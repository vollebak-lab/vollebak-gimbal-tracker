from __future__ import annotations

from typing import Any, Protocol

from ..models import Detection


class Detector(Protocol):
    def detect(self, frame: Any) -> list[Detection]: ...


def build_detector(config: dict[str, Any]) -> Detector:
    detector_type = str(config.get("type", "motion")).lower()
    if detector_type == "motion":
        from .motion import MotionDetector

        return MotionDetector(config)
    if detector_type == "color":
        from .color import ColorDetector

        return ColorDetector(config)
    if detector_type == "person":
        from .person import PersonDetector

        return PersonDetector(config)
    if detector_type in ("person_model", "person_nn", "yolo", "onnx"):
        from .person_model import PersonModelDetector

        return PersonModelDetector(config)
    raise ValueError(f"Unknown detector type: {detector_type}")
