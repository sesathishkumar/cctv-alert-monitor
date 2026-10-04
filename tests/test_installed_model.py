"""Optional inference check; skipped when the downloaded weights are absent."""
import numpy as np
import pytest

from weapons import DEFAULT_MODEL_PATH, WeaponDetector, model_settings


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(), reason='Download pretrained weights first')
def test_installed_model_loads_and_runs_on_blank_frame(monkeypatch):
    monkeypatch.delenv('WEAPON_MODEL_PATH', raising=False)
    monkeypatch.delenv('WEAPON_MANIFEST_PATH', raising=False)
    _, classes, size = model_settings()
    assert {item['category'] for item in classes} >= {'gun', 'knife'}
    assert size == 640
    detector = WeaponDetector(zone=(0, 0, 1, 1), confidence=0.6)
    annotated, events, count = detector.process(np.zeros((360, 640, 3), np.uint8), 0)
    assert annotated.shape == (360, 640, 3)
    assert events == [] and count == 0
