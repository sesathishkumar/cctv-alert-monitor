"""YOLOv8 detection ONNX adapter; requires a model trained for guns AND knives."""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'weapons.onnx'
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / 'models' / 'weapons.json'

def model_settings():
    model = os.environ.get('WEAPON_MODEL_PATH') or DEFAULT_MODEL_PATH
    manifest = os.environ.get('WEAPON_MANIFEST_PATH') or DEFAULT_MANIFEST_PATH
    if not Path(model).is_file() or not Path(manifest).is_file():
        raise ValueError('Weapon model is missing. Run scripts/download_weapon_model.py '
                         'or configure WEAPON_MODEL_PATH and WEAPON_MANIFEST_PATH.')
    config = json.loads(Path(manifest).read_text())
    if not isinstance(config, dict):
        raise ValueError('Weapon manifest must be a JSON object.')
    if config.get('format') != 'yolov8-detection-onnx':
        raise ValueError('Use a YOLOv8 detection ONNX export without embedded NMS.')
    classes = config.get('classes')
    if not isinstance(classes, list) or not classes:
        raise ValueError('The manifest must list model classes in output-index order.')
    for item in classes:
        if (not isinstance(item, dict) or not isinstance(item.get('label'), str)
                or item.get('category') not in {'gun', 'knife', 'ignore'}):
            raise ValueError('Every class needs a label and category: gun, knife, or ignore.')
    if not {'gun', 'knife'} <= {item['category'] for item in classes}:
        raise ValueError('The model must contain both gun and knife classes.')
    size = config.get('input_size', 640)
    if type(size) is not int or not 64 <= size <= 2048:
        raise ValueError('Model input_size must be an integer from 64 to 2048.')
    digest = config.get('sha256')
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError('Provide the expected model SHA-256 in the manifest.')
    try:
        int(digest, 16)
    except ValueError:
        raise ValueError('Model SHA-256 must contain hexadecimal characters.') from None
    path = Path(model)
    with path.open('rb') as handle:
        checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
    if checksum.lower() != digest.lower():
        raise ValueError('Weapon model checksum does not match the manifest.')
    return path, classes, size


class WeaponDetector:
    def __init__(self, zone, dwell=0.7, cooldown=10, confidence=0.6):
        model, self.classes, self.size = model_settings()
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.zone = zone
        self.dwell, self.cooldown, self.confidence = dwell, cooldown, confidence
        self.started = {}
        self.last_alert = {}

    def infer(self, frame):
        height, width = frame.shape[:2]
        scale = self.size / max(height, width)
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
        left = (self.size - resized.shape[1]) // 2
        top = (self.size - resized.shape[0]) // 2
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(padded, 1 / 255, (self.size, self.size), swapRB=True)
        self.net.setInput(blob)
        output = self.net.forward()
        return self.decode(output, frame.shape, scale, left, top)

    def decode(self, output, shape, scale, pad_x, pad_y):
        # Standard YOLOv8 raw output: [1, 4 + num_classes, num_predictions].
        expected = 4 + len(self.classes)
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != expected:
            raise ValueError('Model output does not match the YOLOv8 manifest class count/layout.')
        rows = output[0].T
        if not np.isfinite(rows).all():
            raise ValueError('Weapon model returned non-finite predictions.')
        height, width = shape[:2]
        zx, zy, zw, zh = self.zone
        detections = []
        for row in rows:
            class_id = int(np.argmax(row[4:]))
            score = float(row[4 + class_id])
            category = self.classes[class_id]['category']
            if score < self.confidence or score > 1 or category == 'ignore':
                continue
            cx, cy, bw, bh = map(float, row[:4])
            cx, cy = (cx - pad_x) / scale, (cy - pad_y) / scale
            bw, bh = bw / scale, bh / scale
            if bw <= 0 or bh <= 0 or not (zx * width <= cx <= (zx + zw) * width
                                         and zy * height <= cy <= (zy + zh) * height):
                continue
            x1, y1 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
            x2, y2 = min(width, int(cx + bw / 2)), min(height, int(cy + bh / 2))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({'category': category, 'confidence': score,
                               'label': self.classes[class_id]['label'],
                               'box': [x1, y1, x2 - x1, y2 - y1]})
        result = []
        for category in ('gun', 'knife'):
            items = [item for item in detections if item['category'] == category]
            indices = cv2.dnn.NMSBoxes([item['box'] for item in items],
                                      [item['confidence'] for item in items], self.confidence, 0.45)
            for index in np.asarray(indices).reshape(-1):
                result.append(items[int(index)])
        return result

    def process(self, frame, timestamp):
        detections = self.infer(frame)
        visible = {item['category'] for item in detections}
        events = []
        for category in ('gun', 'knife'):
            if category not in visible:
                self.started.pop(category, None)
                continue
            self.started.setdefault(category, timestamp)
            if (timestamp - self.started[category] >= self.dwell
                    and timestamp - self.last_alert.get(category, float('-inf')) >= self.cooldown):
                self.last_alert[category] = timestamp
                confidence = max(item['confidence'] for item in detections if item['category'] == category)
                events.append({'category': category, 'confidence': confidence,
                               'event': f'Possible {category} visible at video {timestamp:.1f}s'})
        annotated = frame.copy()
        height, width = frame.shape[:2]
        x, y, w, h = self.zone
        cv2.rectangle(annotated, (int(x * width), int(y * height)),
                      (int((x + w) * width), int((y + h) * height)), (0, 210, 230), 2)
        for item in detections:
            x, y, w, h = item['box']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (60, 100, 255), 2)
            cv2.putText(annotated, f"Possible {item['category']} {item['confidence']:.0%}",
                        (x, max(y - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 100, 255), 1)
        return annotated, events, len(detections)
