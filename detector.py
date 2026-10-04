import cv2
import numpy as np


class MotionDetector:
    """Detect sustained foreground movement inside a normalized rectangle."""

    def __init__(self, zone=(0.15, 0.15, 0.7, 0.7), minimum_area=0.008,
                 dwell=0.7, cooldown=10.0):
        self.zone = zone
        self.minimum_area = minimum_area
        self.dwell = dwell
        self.cooldown = cooldown
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=32, detectShadows=True)
        self.frames = 0
        self.started = None
        self.last_alert = float('-inf')

    def process(self, frame, timestamp):
        height, width = frame.shape[:2]
        x, y, w, h = self.zone
        left, top = int(x * width), int(y * height)
        right, bottom = int((x + w) * width), int((y + h) * height)
        region = frame[top:bottom, left:right]
        mask = self.background.apply(region)
        mask = np.where(mask == 255, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) >= region.shape[0] * region.shape[1] * self.minimum_area:
                bx, by, bw, bh = cv2.boundingRect(contour)
                boxes.append((bx + left, by + top, bw, bh))
        self.frames += 1
        # Establish a background before reporting the initial scene as movement.
        if self.frames < 20:
            boxes = []
        if boxes:
            if self.started is None:
                self.started = timestamp
        else:
            self.started = None
        alert = (self.started is not None and timestamp - self.started >= self.dwell
                 and timestamp - self.last_alert >= self.cooldown)
        if alert:
            self.last_alert = timestamp
        annotated = frame.copy()
        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 210, 230), 2)
        for bx, by, bw, bh in boxes:
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (60, 100, 255), 2)
        return annotated, bool(alert), len(boxes)
