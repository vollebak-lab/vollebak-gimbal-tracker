from __future__ import annotations

import logging
import time

from .calibration import Calibration
from .camera import OpenCVCamera, require_cv2
from .config import AppConfig
from .control import SafeAngleController
from .detectors import build_detector
from .drivers import build_driver
from .models import Angles, Detection

LOGGER = logging.getLogger(__name__)


def choose_target(detections: list[Detection]) -> Detection | None:
    return max(detections, key=lambda detection: detection.area, default=None)


def run_tracker(config: AppConfig, preview: bool = False, max_frames: int | None = None) -> None:
    calibration_path = config.resolve(config.calibration.path)
    if not calibration_path.exists():
        raise RuntimeError(
            f"Calibration not found: {calibration_path}. Run 'gimbal-tracker calibrate' first."
        )
    calibration = Calibration.load(calibration_path)
    detector = build_detector(config.detector)
    driver = build_driver(config.gimbal)
    controller = SafeAngleController(config.gimbal, config.tracking)
    cv2 = require_cv2()
    last_seen: float | None = None
    last_command_time = 0.0
    parked = False
    frames = 0

    try:
        with OpenCVCamera(config.camera) as camera:
            while max_frames is None or frames < max_frames:
                frame = camera.read()
                frames += 1
                height, width = frame.shape[:2]
                if (width, height) != (calibration.camera_width, calibration.camera_height):
                    LOGGER.warning(
                        "Camera is %dx%d but calibration is %dx%d; pixel coordinates will be scaled",
                        width,
                        height,
                        calibration.camera_width,
                        calibration.camera_height,
                    )
                detections = detector.detect(frame)
                target = choose_target(detections)
                now = time.monotonic()
                desired: Angles | None = None

                if target is not None:
                    pixel_x, pixel_y = target.center
                    calibration_x = pixel_x * calibration.camera_width / width
                    calibration_y = pixel_y * calibration.camera_height / height
                    desired = calibration.map_pixel(calibration_x, calibration_y)
                    last_seen = now
                    parked = False
                elif last_seen is not None and now - last_seen >= config.tracking.park_after_s:
                    desired = Angles(config.gimbal.home_pan, config.gimbal.home_tilt)
                    parked = True

                interval = 1.0 / config.tracking.command_hz
                commanded: Angles | None = None
                if desired is not None and now - last_command_time >= interval:
                    commanded = controller.update(desired)
                    driver.move(
                        commanded.pan,
                        commanded.tilt,
                        config.gimbal.speed,
                        config.gimbal.acceleration,
                    )
                    last_command_time = now

                if preview:
                    _draw_preview(frame, detections, target, commanded or controller.current, parked)
                    cv2.imshow("Vollebak Gimbal Tracker", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        driver.close()
        if preview:
            cv2.destroyAllWindows()


def _draw_preview(frame, detections, target, angles, parked: bool) -> None:
    cv2 = require_cv2()
    for detection in detections:
        x, y = int(detection.x), int(detection.y)
        right, bottom = int(detection.x + detection.width), int(detection.y + detection.height)
        color = (0, 255, 255) if detection is target else (160, 160, 160)
        cv2.rectangle(frame, (x, y), (right, bottom), color, 2)
    status = "PARKING" if parked else ("TRACKING" if target else "SEARCHING")
    if angles:
        status += f"  pan={angles.pan:+.1f} tilt={angles.tilt:+.1f}"
    cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)
    cv2.putText(frame, "q/esc: quit", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
