"""Temporal fighting classifier. Scores describe a video window, not a person."""
from collections import deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'violence.pth'
MANIFEST_PATH = Path(__file__).resolve().parent / 'models' / 'violence.json'


def violence_settings():
    path = Path(os.environ.get('VIOLENCE_MODEL_PATH') or MODEL_PATH)
    manifest = Path(os.environ.get('VIOLENCE_MANIFEST_PATH') or MANIFEST_PATH)
    if not path.is_file() or not manifest.is_file():
        raise ValueError('Fighting model missing. Run scripts/download_violence_model.py.')
    if importlib.util.find_spec('torch') is None or importlib.util.find_spec('torchvision') is None:
        raise ValueError('Install the optional CPU PyTorch dependencies in requirements-violence.txt.')
    config = json.loads(manifest.read_text())
    if (not isinstance(config, dict) or config.get('format') != 'torchvision-mc3_18-binary'
            or config.get('classes') != ['fight', 'noFight']
            or config.get('sequence_length') != 16):
        raise ValueError('Expected the verified mc3_18 16-frame fight/noFight model manifest.')
    with path.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != config.get('sha256'):
            raise ValueError('Fighting model checksum mismatch.')
    return path, config


class ViolenceDetector:
    def __init__(self, zone, dwell=0.7, cooldown=10, confidence=0.7, predictor=None):
        self.predictor = predictor
        if predictor is None:
            path, config = violence_settings()
            import torch
            from torchvision.models.video import mc3_18
            self.torch = torch
            torch.set_num_threads(min(2, os.cpu_count() or 1))
            self.model = mc3_18(weights=None)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
            # Only tensor weights are loaded; never execute pickled model objects.
            state = torch.load(path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.model.eval()
        self.zone = zone
        self.dwell, self.cooldown, self.confidence = dwell, cooldown, confidence
        self.frames = deque(maxlen=16)
        self.sample_period = 1 / 8
        self.next_sample = 0
        self.sample_count = 0
        self.evaluations = 0
        self.score = None
        self.started = None
        self.positive_windows = 0
        self.last_alert = float('-inf')

    def predict(self, frames):
        prepared = []
        for _, frame in frames:
            height, width = frame.shape[:2]
            x, y, w, h = self.zone
            region = frame[int(y * height):int((y + h) * height),
                           int(x * width):int((x + w) * width)]
            rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (171, 128), interpolation=cv2.INTER_LINEAR)
            # Match upstream Albumentations center-crop and normalization.
            cropped = resized[8:120, 29:141].astype(np.float32) / 255
            normalized = (cropped - np.array([0.43216, 0.394666, 0.37645], np.float32)) / np.array(
                [0.22803, 0.22145, 0.216989], np.float32)
            prepared.append(normalized)
        array = np.stack(prepared).transpose(3, 0, 1, 2)[None]
        with self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(np.ascontiguousarray(array)))
            score = float(self.torch.softmax(logits, dim=1)[0, 0])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Fighting model returned an invalid score.')
        return score

    def process(self, frame, timestamp):
        events = []
        if timestamp + 1e-8 >= self.next_sample:
            self.frames.append((timestamp, frame.copy()))
            self.sample_count += 1
            self.next_sample += self.sample_period
            while self.next_sample <= timestamp:
                self.next_sample += self.sample_period
            if len(self.frames) == 16 and (self.sample_count - 16) % 8 == 0:
                self.evaluations += 1
                self.score = float(self.predictor(self.frames) if self.predictor else self.predict(self.frames))
                if not np.isfinite(self.score) or not 0 <= self.score <= 1:
                    raise ValueError('Fighting model returned an invalid score.')
                if self.score >= self.confidence:
                    self.positive_windows += 1
                    if self.started is None:
                        self.started = timestamp
                    if (self.positive_windows >= 2 and timestamp - self.started >= self.dwell
                            and timestamp - self.last_alert >= self.cooldown):
                        self.last_alert = timestamp
                        events.append({'category': 'possible_fighting', 'confidence': self.score,
                                       'event': f'Possible fighting / physical violence at video {timestamp:.1f}s',
                                       'clip_frames': list(self.frames), 'clip_fps': 8,
                                       'clip_start': self.frames[0][0], 'clip_end': timestamp})
                else:
                    self.positive_windows = 0
                    self.started = None
        annotated = frame.copy()
        height, width = frame.shape[:2]
        x, y, w, h = self.zone
        cv2.rectangle(annotated, (int(x * width), int(y * height)),
                      (int((x + w) * width), int((y + h) * height)), (0, 210, 230), 2)
        label = 'Collecting video window' if self.score is None else f'Fighting model score: {self.score:.2f}'
        cv2.putText(annotated, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 230), 2)
        return annotated, events, int(self.score is not None and self.score >= self.confidence)
