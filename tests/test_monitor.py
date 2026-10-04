import io
import time

import cv2
import numpy as np
import pytest

import app as server
from detector import MotionDetector


def frame(index, outside=False):
    image = np.full((240, 320, 3), 28, np.uint8)
    if index >= 35:
        x = 85 + (index * 5) % 110
        y = 5 if outside else 100
        cv2.rectangle(image, (x, y), (x + 30, y + 30), (230, 230, 230), -1)
    return image


def test_motion_requires_zone_duration_and_cooldown():
    detector = MotionDetector(zone=(0.2, 0.2, 0.6, 0.6), dwell=0.3, cooldown=2)
    alerts = []
    for index in range(100):
        _, alert, _ = detector.process(frame(index), index / 15)
        if alert:
            alerts.append(index / 15)
    assert len(alerts) >= 2
    assert alerts[0] >= 35 / 15 + 0.3
    assert all(b - a >= 2 for a, b in zip(alerts, alerts[1:]))
    outside = MotionDetector(zone=(0.2, 0.2, 0.6, 0.6), dwell=0.3)
    assert not any(outside.process(frame(i, outside=True), i / 15)[1] for i in range(100))


@pytest.fixture
def client(tmp_path, monkeypatch):
    server.monitor.stop()
    monkeypatch.setattr(server, 'DATA', tmp_path)
    (tmp_path / 'uploads').mkdir()
    (tmp_path / 'evidence').mkdir()
    server.initialize_database()
    monkeypatch.setattr(server, 'monitor', server.Monitor())
    server.app.config['TESTING'] = True
    yield server.app.test_client()
    server.monitor.stop()


@pytest.mark.parametrize('settings', [
    {'zone': '[0.8,0.1,0.5,0.5]'}, {'zone': '[0,0,0,1]'},
    {'zone': '[0,0,NaN,1]'}, {'zone': '{}'}, {'dwell': 'nan'}, {'cooldown': '0'},
    {'source': 'video'}, {'source': 'unknown'},
])
def test_invalid_settings_are_rejected(client, settings):
    response = client.post('/api/start', data=settings)
    assert response.status_code == 400
    assert response.json['error']
    assert client.get('/api/status').json['status'] == 'idle'


def test_uploaded_video_alert_evidence_acknowledgment_and_eof(client, tmp_path):
    path = tmp_path / 'fixture.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 25, (320, 240))
    assert writer.isOpened()
    for index in range(110):
        writer.write(frame(index))
    writer.release()
    response = client.post('/api/start', data={
        'source': 'video', 'video': (io.BytesIO(path.read_bytes()), 'fixture.avi'),
        'zone': '[0.2,0.2,0.6,0.6]', 'dwell': '0.3', 'cooldown': '1',
    })
    assert response.status_code == 200
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if client.get('/api/status').json['status'] in {'finished', 'error'}:
            break
        time.sleep(0.05)
    state = client.get('/api/status').json
    assert state['status'] == 'finished', state
    assert state['frames'] == 110
    alerts = client.get('/api/alerts').json
    assert alerts and all(a['source'] == 'Uploaded video' for a in alerts)
    evidence = client.get('/evidence/' + alerts[0]['evidence'])
    assert evidence.status_code == 200
    decoded = cv2.imdecode(np.frombuffer(evidence.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (240, 320, 3)
    identifier = alerts[0]['id']
    assert client.post(f'/api/alerts/{identifier}/acknowledge').status_code == 200
    assert client.get('/api/alerts').json[0]['acknowledged'] == 1
    assert client.post('/api/alerts/missing/acknowledge').status_code == 404
    assert client.post('/api/stop').status_code == 200
    with server.database() as db:
        assert db.execute('SELECT COUNT(*) FROM alerts').fetchone()[0] == len(alerts)


def test_undecodable_video_reports_input_failure(client):
    response = client.post('/api/start', data={
        'source': 'video', 'video': (io.BytesIO(b'not a video'), 'bad.mp4')})
    assert response.status_code == 200
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = client.get('/api/status').json
        if state['status'] == 'error':
            break
        time.sleep(0.03)
    assert state['status'] == 'error'
    assert 'Cannot decode' in state['error']
    assert client.get('/api/alerts').json == []
