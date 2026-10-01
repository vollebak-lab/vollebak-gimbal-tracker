from __future__ import annotations

from typing import Any

import numpy as np

from ..camera import require_cv2
from ..models import Detection


class ColorDetector:
    def __init__(self, config: dict[str, Any]) -> None:
        self.cv2 = require_cv2()
        self.low = np.asarray(config.get("hsv_low", [35, 80, 60]), dtype=np.uint8)
        self.high = np.asarray(config.get("hsv_high", [85, 255, 255]), dtype=np.uint8)
        self.min_area = float(config.get("min_area", 800))

    def detect(self, frame: Any) -> list[Detection]:
        hsv = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2HSV)
        mask = self.cv2.inRange(hsv, self.low, self.high)
        kernel = self.cv2.getStructuringElement(self.cv2.MORPH_ELLIPSE, (5, 5))
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_OPEN, kernel)
        mask = self.cv2.morphologyEx(mask, self.cv2.MORPH_CLOSE, kernel)
        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            if self.cv2.contourArea(contour) < self.min_area:
                continue
            x, y, width, height = self.cv2.boundingRect(contour)
            detections.append(Detection(x, y, width, height, 1.0, "color"))
        return detections
