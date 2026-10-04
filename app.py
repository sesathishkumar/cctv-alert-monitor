import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from detector import MotionDetector
from weapons import WeaponDetector, model_settings
from violence import ViolenceDetector, violence_settings

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
for directory in ['uploads', 'evidence']:
    (DATA / directory).mkdir(exist_ok=True)
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024


def database():
    connection = sqlite3.connect(DATA / 'alerts.sqlite')
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with database() as db:
        db.execute('CREATE TABLE IF NOT EXISTS alerts (id TEXT PRIMARY KEY, created TEXT, '
                   'source TEXT, event TEXT, evidence TEXT, acknowledged INTEGER DEFAULT 0)')
        columns = {row['name'] for row in db.execute('PRAGMA table_info(alerts)')}
        for name, sql_type in [('category', 'TEXT'), ('confidence', 'REAL'), ('clip', 'TEXT'),
                               ('clip_start', 'REAL'), ('clip_end', 'REAL')]:
            if name not in columns:
                db.execute(f'ALTER TABLE alerts ADD COLUMN {name} {sql_type}')


initialize_database()


class CombinedDetector:
    def __init__(self, **settings):
        self.weapons = WeaponDetector(**settings)
        self.violence = ViolenceDetector(**settings)

    @property
    def evaluations(self):
        return self.violence.evaluations

    def process(self, frame, timestamp):
        annotated, weapon_events, weapon_count = self.weapons.process(frame, timestamp)
        _, fight_events, fight_count = self.violence.process(frame, timestamp)
        if self.violence.score is not None:
            cv2.putText(annotated, f'Fighting model score: {self.violence.score:.2f}',
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 230), 2)
        return annotated, weapon_events + fight_events, weapon_count + fight_count


class Monitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.lifecycle = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.jpeg = None
        self.state = {'status': 'idle', 'source': '', 'frames': 0, 'movement': 0, 'error': ''}

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.thread and self.thread.is_alive():
            raise RuntimeError('Input is still closing. Retry shortly.')
        with self.lock:
            self.state['status'] = 'stopped'

    def start(self, source, label, settings):
        with self.lifecycle:
            self.stop()
            self.stop_event = threading.Event()
            with self.lock:
                self.jpeg = None
                self.state = {'status': 'starting', 'source': label, 'frames': 0,
                              'movement': 0, 'error': ''}
            self.thread = threading.Thread(target=self.run, args=(source, label, settings), daemon=True)
            self.thread.start()

    def run(self, source, label, settings):
        capture = None
        try:
            mode = settings.pop('mode', 'motion')
            detector_class = {'motion': MotionDetector, 'weapons': WeaponDetector,
                              'violence': ViolenceDetector, 'combined': CombinedDetector}[mode]
            detector = detector_class(**settings)
            if source != 'demo':
                capture = cv2.VideoCapture(str(source))
                if not capture.isOpened():
                    raise ValueError('Cannot decode this video. Try an MP4 or AVI file.')
                fps = capture.get(cv2.CAP_PROP_FPS)
                fps = fps if 1 <= fps <= 120 else 25
            else:
                fps = 15
            index = 0
            while not self.stop_event.is_set():
                began = time.monotonic()
                if source == 'demo':
                    frame = np.full((360, 640, 3), 28, np.uint8)
                    cv2.putText(frame, 'SYNTHETIC DEMO', (20, 32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (170, 180, 190), 1)
                    if index > 35:
                        x = 120 + (index * 6) % 350
                        cv2.rectangle(frame, (x, 130), (x + 50, 220), (230, 230, 230), -1)
                else:
                    ok, frame = capture.read()
                    if not ok:
                        if mode in {'violence', 'combined'} and detector.evaluations == 0:
                            raise ValueError('Video too short for a 16-frame fighting window (about two seconds).')
                        with self.lock:
                            self.state['status'] = 'finished'
                        break
                    height, width = frame.shape[:2]
                    if width > 960:
                        frame = cv2.resize(frame, (960, int(height * 960 / width)))
                timestamp = index / fps
                annotated, alerts, movement = detector.process(frame, timestamp)
                if mode == 'motion':
                    alerts = ([{'category': 'motion', 'confidence': None,
                                'event': f'Movement in configured zone at video {timestamp:.1f}s'}]
                              if alerts else [])
                ok, encoded = cv2.imencode('.jpg', annotated)
                if not ok:
                    raise ValueError('Frame encoding failed.')
                for alert in alerts:
                    identifier = uuid.uuid4().hex
                    filename = identifier + '.jpg'
                    (DATA / 'evidence' / filename).write_bytes(encoded.tobytes())
                    clip = None
                    if alert.get('clip_frames'):
                        clip = identifier + '.avi'
                        frames = alert['clip_frames']
                        height, width = frames[0][1].shape[:2]
                        writer = cv2.VideoWriter(str(DATA / 'evidence' / clip),
                                                 cv2.VideoWriter_fourcc(*'MJPG'), alert['clip_fps'], (width, height))
                        if not writer.isOpened():
                            raise ValueError('Cannot write the alert evidence clip.')
                        try:
                            for _, clip_frame in frames:
                                writer.write(clip_frame)
                        finally:
                            writer.release()
                    with database() as db:
                        db.execute('INSERT INTO alerts (id, created, source, event, evidence, category, confidence, '
                                   'clip, clip_start, clip_end) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                   (identifier, datetime.now(timezone.utc).isoformat(), label,
                                    alert['event'], filename, alert['category'], alert['confidence'], clip,
                                    alert.get('clip_start'), alert.get('clip_end')))
                with self.lock:
                    self.jpeg = encoded.tobytes()
                    self.state.update(status='monitoring', frames=index + 1, movement=movement)
                index += 1
                self.stop_event.wait(max(0, 1 / fps - (time.monotonic() - began)))
            if self.stop_event.is_set():
                with self.lock:
                    self.state['status'] = 'stopped'
        except Exception as exc:
            with self.lock:
                self.state.update(status='error', error=str(exc))
        finally:
            if capture is not None:
                capture.release()


monitor = Monitor()


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/status')
def status():
    with monitor.lock:
        return jsonify(monitor.state.copy())


@app.post('/api/start')
def start():
    try:
        mode = request.form.get('mode', 'motion')
        if mode not in {'motion', 'weapons', 'violence', 'combined'}:
            raise ValueError('Choose motion, weapon, or possible fighting detection.')
        confidence = float(request.form.get('confidence', '0.6'))
        if not 0.1 <= confidence <= 0.99:
            raise ValueError('Confidence threshold must be between 0.1 and 0.99.')
        if mode in {'weapons', 'combined'}:
            model_settings()
        if mode in {'violence', 'combined'}:
            violence_settings()
        zone = json.loads(request.form.get('zone', '[0.15,0.15,0.7,0.7]'))
        if (not isinstance(zone, list) or len(zone) != 4
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in zone)):
            raise ValueError('Zone needs four finite numeric values.')
        x, y, w, h = zone
        if not (0 <= x < 1 and 0 <= y < 1 and w >= 0.05 and h >= 0.05
                and x + w <= 1 and y + h <= 1):
            raise ValueError('Zone must fit inside the frame and be at least 5% wide and high.')
        dwell = float(request.form.get('dwell', '0.7'))
        cooldown = float(request.form.get('cooldown', '10'))
        if not (0.1 <= dwell <= 30 and 1 <= cooldown <= 300):
            raise ValueError('Dwell must be 0.1–30 seconds and cooldown 1–300 seconds.')
        source = request.form.get('source', 'demo')
        if source == 'video':
            upload = request.files.get('video')
            if not upload or Path(upload.filename).suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv'}:
                raise ValueError('Choose an MP4, AVI, MOV or MKV video.')
            path = DATA / 'uploads' / (uuid.uuid4().hex + Path(upload.filename).suffix.lower())
            upload.save(path)
            source, label = path, 'Uploaded video'
        elif source == 'demo':
            if mode in {'violence', 'combined'}:
                raise ValueError('Fighting detection requires a video; the motion demo is not a fighting example.')
            label = 'Synthetic demo'
        else:
            raise ValueError('Choose demo or video input.')
        settings = {'zone': zone, 'dwell': dwell, 'cooldown': cooldown, 'mode': mode}
        if mode in {'weapons', 'violence', 'combined'}:
            settings['confidence'] = confidence
        monitor.start(source, label, settings)
        return jsonify(ok=True)
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        return jsonify(error=str(exc)), 400


@app.get('/api/capabilities')
def capabilities():
    try:
        model_settings()
        # Configuration and checksum checks do not establish inference readiness.
        weapons = {'configured': True, 'message': 'Model configured; inference checked when monitoring starts.'}
    except (ValueError, TypeError, OSError) as exc:
        weapons = {'configured': False, 'message': str(exc)}
    try:
        violence_settings()
        violence = {'configured': True, 'message': '16-frame fighting model configured; human review required.'}
    except (ValueError, TypeError, OSError) as exc:
        violence = {'configured': False, 'message': str(exc)}
    return jsonify(motion=True, weapons=weapons, violence=violence)


@app.post('/api/stop')
def stop():
    try:
        with monitor.lifecycle:
            monitor.stop()
        return jsonify(ok=True)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409


@app.get('/api/alerts')
def alerts():
    with database() as db:
        rows = db.execute('SELECT * FROM alerts ORDER BY created DESC LIMIT 100').fetchall()
        return jsonify([dict(row) for row in rows])


@app.post('/api/alerts/<identifier>/acknowledge')
def acknowledge(identifier):
    with database() as db:
        cursor = db.execute('UPDATE alerts SET acknowledged=1 WHERE id=?', (identifier,))
        if not cursor.rowcount:
            return jsonify(error='Alert not found'), 404
    return jsonify(ok=True)


@app.get('/evidence/<filename>')
def evidence(filename):
    return send_from_directory(DATA / 'evidence', filename)


@app.get('/stream')
def stream():
    def frames():
        while True:
            with monitor.lock:
                jpeg = monitor.jpeg
            if jpeg:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            time.sleep(0.1)
    return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True, debug=False)
