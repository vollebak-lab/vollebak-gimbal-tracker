from __future__ import annotations

from typing import Any

from ..camera import require_cv2
from ..models import Detection


class MotionDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.cv2 = require_cv2()
        self.min_area = float(config.get("min_area", 1400))
        self.warmup_frames = int(config.get("warmup_frames", 20))
        self.frame_count = 0
        self.subtractor = self.cv2.createBackgroundSubtractorMOG2(
            history=int(config.get("history", 300)),
            varThreshold=float(config.get("threshold", 32)),
            detectShadows=True,
        )

    def detect(self, frame: Any) -> list[Detection]:
        self.frame_count += 1
        mask = self.subtractor.apply(frame)
        if self.frame_count <= self.warmup_frames:
            return []
        _, mask = self.cv2.threshold(mask, 200, 255, self.cv2.THRESH_BINARY)
        kernel = self.cv2.getStructuringElement(self.cv2.MORPH_ELLIPSE, (5, 5))
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
        mask = self.cv2.dilate(mask, kernel, iterations=2)
        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = self.cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, width, height = self.cv2.boundingRect(contour)
            detections.append(Detection(x, y, width, height, 1.0, "motion"))
        return detections
