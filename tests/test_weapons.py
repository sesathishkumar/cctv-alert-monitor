import hashlib
import json

import numpy as np
import pytest

from weapons import WeaponDetector, model_settings
from test_monitor import client


def decoder():
    # Validate parsing/timing with fixtures; no trained-model accuracy claim.
    detector = WeaponDetector.__new__(WeaponDetector)
    detector.classes = [{'label': 'handgun', 'category': 'gun'}, {'label': 'knife', 'category': 'knife'}]
    detector.zone = (0, 0, 1, 1)
    detector.confidence = 0.6
    detector.started = {}
    detector.last_alert = {}
    detector.dwell, detector.cooldown = 0.5, 2
    return detector


def test_both_categories_and_nms():
    output = np.array([[
        [50, 50, 20, 20, 0.9, 0.1],
        [51, 51, 20, 20, 0.85, 0.1],
        [130, 130, 10, 30, 0.1, 0.8],
        [100, 100, 20, 20, 0.2, 0.3],
    ]], dtype=np.float32).transpose(0, 2, 1)
    results = decoder().decode(output, (200, 200, 3), 1, 0, 0)
    assert len(results) == 2
    assert {item['category'] for item in results} == {'gun', 'knife'}
    assert all(item['confidence'] >= 0.6 for item in results)


def test_layout_and_zone_checks():
    detector = decoder()
    with pytest.raises(ValueError, match='layout'):
        detector.decode(np.zeros((1, 8, 20)), (200, 200, 3), 1, 0, 0)
    detector.zone = (0.5, 0.5, 0.5, 0.5)
    output = np.array([[[10, 10, 5, 5, 0.9, 0.1]]]).transpose(0, 2, 1)
    assert detector.decode(output, (200, 200, 3), 1, 0, 0) == []


def test_category_timing_and_disappearance_reset():
    detector = decoder()
    visible = [{'category': 'gun', 'confidence': 0.8, 'box': [10, 10, 10, 10]},
               {'category': 'knife', 'confidence': 0.9, 'box': [30, 30, 10, 10]}]
    detector.infer = lambda frame: visible
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert not detector.process(frame, 0)[1]
    events = detector.process(frame, 0.5)[1]
    assert {event['category'] for event in events} == {'gun', 'knife'}
    assert not detector.process(frame, 1)[1]
    visible.clear()
    assert not detector.process(frame, 2.5)[1]
    visible.append({'category': 'knife', 'confidence': 0.9, 'box': [30, 30, 10, 10]})
    assert not detector.process(frame, 3)[1]
    assert detector.process(frame, 3.5)[1][0]['category'] == 'knife'


def test_manifest_requires_both_classes_and_checksum(tmp_path, monkeypatch):
    model = tmp_path / 'model.onnx'
    model.write_bytes(b'fixture-not-real-onnx')
    manifest = tmp_path / 'manifest.json'
    config = {'format': 'yolov8-detection-onnx', 'input_size': 640,
              'sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
              'classes': [{'label': 'gun', 'category': 'gun'}]}
    manifest.write_text(json.dumps(config))
    monkeypatch.setenv('WEAPON_MODEL_PATH', str(model))
    monkeypatch.setenv('WEAPON_MANIFEST_PATH', str(manifest))
    with pytest.raises(ValueError, match='both gun and knife'):
        model_settings()
    config['classes'].append({'label': 'knife', 'category': 'knife'})
    config['sha256'] = '0' * 64
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='checksum'):
        model_settings()


def test_unconfigured_mode_is_explicitly_unavailable(client, monkeypatch, tmp_path):
    import weapons
    monkeypatch.setattr(weapons, 'DEFAULT_MODEL_PATH', tmp_path / 'missing.onnx')
    monkeypatch.setattr(weapons, 'DEFAULT_MANIFEST_PATH', tmp_path / 'missing.json')
    monkeypatch.delenv('WEAPON_MODEL_PATH', raising=False)
    monkeypatch.delenv('WEAPON_MANIFEST_PATH', raising=False)
    response = client.get('/api/capabilities')
    assert response.json['weapons']['configured'] is False
    assert isinstance(response.json['violence']['configured'], bool)
    response = client.post('/api/start', data={'mode': 'weapons'})
    assert response.status_code == 400
    assert 'WEAPON_MODEL_PATH' in response.json['error']
