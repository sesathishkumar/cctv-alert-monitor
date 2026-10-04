import io
import time

import cv2
import numpy as np
import pytest

import app as server
import violence
from test_monitor import client


FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def test_temporal_warmup_confirmation_clip_and_cooldown():
    detector = violence.ViolenceDetector((0, 0, 1, 1), cooldown=5, predictor=lambda frames: 0.95)
    events = []
    for index in range(80):
        _, alerts, _ = detector.process(FRAME, index / 8)
        if index < 23:
            assert not alerts
        events.extend(alerts)
    assert len(events) == 2
    assert events[0]['clip_end'] == 23 / 8
    assert events[1]['clip_end'] - events[0]['clip_end'] >= 5
    assert all(len(event['clip_frames']) == 16 for event in events)
    assert events[0]['clip_start'] == 1
    assert events[0]['category'] == 'possible_fighting'


def test_low_score_resets_confirmation():
    scores = iter([0.95, 0.1, 0.95, 0.95])
    detector = violence.ViolenceDetector((0, 0, 1, 1), predictor=lambda frames: next(scores))
    alerts = []
    for index in range(40):
        alerts.extend(detector.process(FRAME, index / 8)[1])
    assert len(alerts) == 1
    assert alerts[0]['clip_end'] == 39 / 8


def test_nonfinite_model_score_fails_explicitly():
    detector = violence.ViolenceDetector((0, 0, 1, 1), predictor=lambda frames: float('nan'))
    with pytest.raises(ValueError, match='invalid score'):
        for index in range(16):
            detector.process(FRAME, index / 8)


def test_combined_detectors_receive_original_frames(monkeypatch):
    original = FRAME.copy()
    class Weapons:
        def __init__(self, **kwargs):
            pass
        def process(self, image, timestamp):
            assert np.array_equal(image, original)
            return image + 1, [{'category': 'gun'}], 1
    class Fighting:
        score = 0.9
        evaluations = 1
        def __init__(self, **kwargs):
            pass
        def process(self, image, timestamp):
            assert np.array_equal(image, original)  # Never classify weapon overlays as input.
            return image, [{'category': 'possible_fighting'}], 1
    monkeypatch.setattr(server, 'WeaponDetector', Weapons)
    monkeypatch.setattr(server, 'ViolenceDetector', Fighting)
    combined = server.CombinedDetector(zone=(0, 0, 1, 1))
    _, events, count = combined.process(original, 3)
    assert {event['category'] for event in events} == {'gun', 'possible_fighting'}
    assert count == 2 and combined.evaluations == 1


def test_missing_model_and_demo_input_rejected(client, monkeypatch, tmp_path):
    monkeypatch.delenv('VIOLENCE_MODEL_PATH', raising=False)
    monkeypatch.delenv('VIOLENCE_MANIFEST_PATH', raising=False)
    monkeypatch.setattr(violence, 'MODEL_PATH', tmp_path / 'missing.pth')
    assert client.get('/api/capabilities').json['violence']['configured'] is False
    assert client.post('/api/start', data={'mode': 'violence'}).status_code == 400
    monkeypatch.setattr(server, 'violence_settings', lambda: None)
    response = client.post('/api/start', data={'mode': 'violence', 'source': 'demo'})
    assert response.status_code == 400 and 'requires a video' in response.json['error']


def test_video_alert_clip_and_acknowledgment(client, monkeypatch, tmp_path):
    # Fixed high scores validate the alert pipeline, not recognition accuracy.
    monkeypatch.setattr(server, 'violence_settings', lambda: None)
    monkeypatch.setattr(server, 'ViolenceDetector', lambda **settings: violence.ViolenceDetector(
        **settings, predictor=lambda frames: 0.95))
    path = tmp_path / 'fixture.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 8, (160, 120))
    assert writer.isOpened()
    for _ in range(32):
        writer.write(FRAME)
    writer.release()
    response = client.post('/api/start', data={'mode': 'violence', 'source': 'video',
        'video': (io.BytesIO(path.read_bytes()), 'fixture.avi'), 'confidence': '0.7'})
    assert response.status_code == 200
    deadline = time.monotonic() + 7
    while time.monotonic() < deadline:
        state = client.get('/api/status').json
        if state['status'] in {'finished', 'error'}:
            break
        time.sleep(0.05)
    assert state['status'] == 'finished', state
    alerts = client.get('/api/alerts').json
    assert len(alerts) == 1 and alerts[0]['category'] == 'possible_fighting'
    alert = alerts[0]
    assert alert['confidence'] == 0.95 and alert['clip_end'] > alert['clip_start']
    assert client.get('/evidence/' + alert['clip']).status_code == 200
    capture = cv2.VideoCapture(str(tmp_path / 'evidence' / alert['clip']))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 16
    assert capture.read()[0]
    capture.release()
    assert client.post('/api/alerts/' + alert['id'] + '/acknowledge').status_code == 200
    assert client.get('/api/alerts').json[0]['acknowledged'] == 1


@pytest.mark.skipif(not violence.MODEL_PATH.is_file(), reason='Download temporal classifier weights first')
def test_real_temporal_model_inference(monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    monkeypatch.delenv('VIOLENCE_MODEL_PATH', raising=False)
    monkeypatch.delenv('VIOLENCE_MANIFEST_PATH', raising=False)
    detector = violence.ViolenceDetector((0, 0, 1, 1))
    events = []
    for index in range(24):
        events.extend(detector.process(FRAME, index / 8)[1])
    assert detector.evaluations == 2 and 0 <= detector.score <= 1
    assert not events  # This checkpoint does not label the blank control as fighting.
