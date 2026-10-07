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


def run_autonomous_tracker(
    config: AppConfig,
    preview: bool = False,
    max_frames: int | None = None,
    laser_hardware: Any = None,
) -> None:
    """Run closed-loop person center-of-mass detection and autonomous laser tracking."""
    from .autonomous_tracker import AutonomousLaserTracker, AutonomousTrackerConfig

    calibration_path = config.resolve(config.calibration.path)
    if not calibration_path.exists():
        raise RuntimeError(
            f"Calibration not found: {calibration_path}. Run 'gimbal-tracker calibrate' first."
        )
    calibration = Calibration.load(calibration_path)
    detector = build_detector(config.detector)
    driver = build_driver(config.gimbal)

    tracker_config = AutonomousTrackerConfig(
        lock_tolerance_deg=float(config.tracking.deadband_deg or 1.2),
        lock_consecutive_frames=3,
        laser_auto_engage=True,
        laser_max_continuous_s=10.0,
        command_hz=config.tracking.command_hz,
    )
    tracker = AutonomousLaserTracker(
        gimbal_config=config.gimbal,
        tracking_config=config.tracking,
        tracker_config=tracker_config,
        calibration=calibration,
        driver=driver,
        laser=laser_hardware,
    )
    cv2 = require_cv2()
    frames = 0
    try:
        with OpenCVCamera(config.camera) as camera:
            while max_frames is None or frames < max_frames:
                frame = camera.read()
                frames += 1
                height, width = frame.shape[:2]
                detections = detector.detect(frame)
                target = choose_target(detections)

                commanded, state, laser_on = tracker.update(target, width, height)

                if preview:
                    _draw_autonomous_preview(
                        frame, detections, target, commanded, state, laser_on, tracker.error_deg
                    )
                    cv2.imshow("Vollebak Autonomous Person Laser Tracker", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        tracker.emergency_stop()
        driver.close()
        if preview:
            cv2.destroyAllWindows()


def _draw_autonomous_preview(
    frame: Any,
    detections: list[Detection],
    target: Detection | None,
    angles: Angles,
    state: Any,
    laser_on: bool,
    error_deg: float,
) -> None:
    cv2 = require_cv2()
    for det in detections:
        x, y = int(det.x), int(det.y)
        r, b = int(det.x + det.width), int(det.y + det.height)
        is_primary = (det is target)
        color = (0, 255, 0) if is_primary else (120, 120, 120)
        cv2.rectangle(frame, (x, y), (r, b), color, 2)

        if is_primary:
            # Draw distinct Center-of-Mass reticle
            cx, cy = int(det.center[0]), int(det.center[1])
            reticle_color = (0, 0, 255) if laser_on else (0, 255, 255)
            cv2.circle(frame, (cx, cy), 8, reticle_color, 2)
            cv2.drawMarker(frame, (cx, cy), reticle_color, cv2.MARKER_CROSS, 24, 2)
            cv2.putText(
                frame,
                f"PERSON CoM ({cx},{cy})",
                (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                reticle_color,
                1,
            )

    # Top status banner
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (10, 10, 10), -1)
    laser_text = "[LASER ARMED // ACTIVE]" if laser_on else "[LASER SAFE // OFF]"
    laser_color = (0, 0, 255) if laser_on else (0, 255, 0)
    state_str = getattr(state, "value", str(state))
    status_msg = f"STATE: {state_str} | PAN: {angles.pan:+.1f} TILT: {angles.tilt:+.1f} | ERR: {error_deg:0.1f}deg"
    cv2.putText(frame, status_msg, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    cv2.putText(frame, laser_text, (max(10, frame.shape[1] - 240), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, laser_color, 2)
