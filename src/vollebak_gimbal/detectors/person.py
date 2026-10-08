from __future__ import annotations

from typing import Any

from ..camera import require_cv2
from ..models import Detection


class PersonDetector:
    """Dependency-free OpenCV HOG person detector; useful as a baseline on Pi 5."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.cv2 = require_cv2()
        self.scale = float(config.get("processing_scale", 0.5))
        self.hit_threshold = float(config.get("hit_threshold", 0.0))
        self.hog = self.cv2.HOGDescriptor()
        self.hog.setSVMDetector(self.cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: Any) -> list[Detection]:
        working = self.cv2.resize(frame, None, fx=self.scale, fy=self.scale)
        boxes, weights = self.hog.detectMultiScale(
            working,
            hitThreshold=self.hit_threshold,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        inverse = 1.0 / self.scale
        return [
            Detection(
                float(x * inverse),
                float(y * inverse),
                float(width * inverse),
                float(height * inverse),
                float(weight),
                "person",
            )
            for (x, y, width, height), weight in zip(boxes, weights, strict=True)
        ]
