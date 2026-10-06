from __future__ import annotations

from typing import Any, Self

from .config import CameraConfig


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required. Install the vision extra or python3-opencv on Raspberry Pi."
        ) from exc
    return cv2


class OpenCVCamera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cv2 = require_cv2()
        source = config.source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        backend_names = {
            "dshow": "CAP_DSHOW",
            "msmf": "CAP_MSMF",
            "v4l2": "CAP_V4L2",
        }
        backend_name = backend_names.get(config.backend)
        backend = getattr(self.cv2, backend_name) if backend_name else None
        self.capture = (
            self.cv2.VideoCapture(source, backend)
            if backend is not None and isinstance(source, int)
            else self.cv2.VideoCapture(source)
        )
        if config.backend == "dshow" and hasattr(self.cv2, "VideoWriter_fourcc"):
            self.capture.set(self.cv2.CAP_PROP_FOURCC, self.cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.capture.set(self.cv2.CAP_PROP_FPS, config.fps)
        self.capture.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open camera source {config.source!r}")

    def read(self):
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Camera stopped returning frames")
        return frame

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
