from __future__ import annotations

import json
import logging
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

from .alignment import AlignmentResult, ExtrinsicAlignmentConfig, ExtrinsicParallaxModel
from .autonomous_tracker import LaserHardwareInterface
from .calibration import Calibration
from .camera import OpenCVCamera, require_cv2
from .config import AppConfig
from .control import SafeAngleController, clamp
from .detectors import build_detector
from .drivers import build_driver
from .event_camera import EventCameraMonitor, primary_event_target
from .laser_test import LaserTestController
from .models import Angles, Detection
from .predator_observer import build_observer_telemetry
from .tracker import choose_target

LOGGER = logging.getLogger(__name__)


class DemoCamera:
    """Synthetic scene used to exercise the complete tracker without hardware."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cv2 = require_cv2()
        self.started = time.monotonic()

    def read(self) -> tuple[Any, Detection]:
        elapsed = time.monotonic() - self.started
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (10, 12, 12)
        grid_color = (28, 32, 31)
        for x in range(0, self.width, max(40, self.width // 12)):
            self.cv2.line(frame, (x, 0), (x, self.height), grid_color, 1)
        for y in range(0, self.height, max(40, self.height // 8)):
            self.cv2.line(frame, (0, y), (self.width, y), grid_color, 1)

        x = self.width * (0.5 + 0.34 * math.sin(elapsed * 0.55))
        y = self.height * (0.5 + 0.27 * math.sin(elapsed * 0.83 + 0.7))
        radius = max(18, min(self.width, self.height) // 18)
        self.cv2.circle(frame, (int(x), int(y)), radius + 12, (20, 45, 36), -1)
        self.cv2.circle(frame, (int(x), int(y)), radius, (73, 248, 174), -1)
        self.cv2.circle(frame, (int(x), int(y)), 5, (240, 255, 249), -1)
        self.cv2.putText(
            frame,
            "SYNTHETIC TARGET // SAFE MODE",
            (18, self.height - 20),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (125, 150, 140),
            1,
        )
        detection = Detection(x - radius, y - radius, radius * 2, radius * 2, 1.0, "demo")
        return frame, detection


class DashboardEngine:
    def __init__(
        self,
        config: AppConfig,
        *,
        force_demo: bool = False,
        fallback_demo: bool = True,
    ) -> None:
        self.config = config
        self.force_demo = force_demo
        self.fallback_demo = fallback_demo
        self.cv2 = require_cv2()
        self.detector = build_detector(config.detector)
        self.event_camera = EventCameraMonitor(config.event_camera)
        self.laser_test = LaserTestController(config.laser_test)
        laser_gpio = int(
            getattr(config.autonomous_tracker, "laser_gpio", 17)
            if hasattr(config, "autonomous_tracker")
            else getattr(config.laser_test, "gpio", 17)
        )
        self.laser_hardware = LaserHardwareInterface(gpio=laser_gpio)
        self.auto_laser_enabled = bool(
            hasattr(config, "autonomous_tracker")
            and config.autonomous_tracker.enabled
            and config.autonomous_tracker.laser_auto_engage
        )
        self.driver = build_driver(config.gimbal)
        self.controller = SafeAngleController(config.gimbal, config.tracking)
        self.controller.reset(Angles(config.gimbal.home_pan, config.gimbal.home_tilt))
        calibration_path = config.resolve(config.calibration.path)
        self.calibration = Calibration.load(calibration_path) if calibration_path.exists() else None
        self.alignment_model = ExtrinsicParallaxModel(config.alignment)
        self._manual_aim_active = False
        self._latest_target: Detection | None = None
        self._latest_frame_shape: tuple[int, int] = (config.camera.height, config.camera.width)

        self._lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: bytes | None = None
        self._frame_id = 0
        self._tracking_enabled = bool(
            config.tracking.start_enabled and self.calibration is not None
        )
        initial_angles = Angles(config.gimbal.home_pan, config.gimbal.home_tilt)
        initial_event_camera = self.event_camera.snapshot()
        self._state: dict[str, Any] = {
            "status": "STARTING",
            "camera": "CONNECTING",
            "camera_mode": "unknown",
            "tracking_enabled": self._tracking_enabled,
            "auto_laser_enabled": self.auto_laser_enabled,
            "laser_active": False,
            "driver": config.gimbal.driver,
            "pan": config.gimbal.home_pan,
            "tilt": config.gimbal.home_tilt,
            "target": None,
            "fps": 0.0,
            "frame_width": config.camera.width,
            "frame_height": config.camera.height,
            "calibrated": self.calibration is not None,
            "message": "",
            "event_camera": initial_event_camera,
            "laser_test": self.laser_test.snapshot(),
            "alignment": self.config.alignment.to_dict(),
            "alignment_live": {
                "distance_m": round(self.config.alignment.nominal_distance_m, 2),
                "distance_ft": round(self.config.alignment.nominal_distance_m / 0.3048, 1),
                "distance_source": "nominal",
                "parallax_pan_deg": 0.0,
                "parallax_tilt_deg": 0.0,
                "trim_pan_deg": round(self.config.alignment.trim_pan_deg, 2),
                "trim_tilt_deg": round(self.config.alignment.trim_tilt_deg, 2),
            },
            "limits": {
                "pan_min": config.gimbal.pan_min,
                "pan_max": config.gimbal.pan_max,
                "tilt_min": config.gimbal.tilt_min,
                "tilt_max": config.gimbal.tilt_max,
            },
            "predator": build_observer_telemetry(
                config,
                camera_mode="unknown",
                status="STARTING",
                target=None,
                target_angles=None,
                current_angles=initial_angles,
                event_camera=initial_event_camera,
            ),
        }

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.event_camera.start()
        self._thread = threading.Thread(target=self._run, name="dashboard-tracker", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.event_camera.close()
        self.laser_test.force_off()
        self.laser_hardware.set_state(False)
        self.driver.close()

    def state(self) -> dict[str, Any]:
        with self._lock:
            self._state["laser_test"] = self.laser_test.snapshot()
            return json.loads(json.dumps(self._state))

    def wait_for_frame(self, previous_id: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: self._frame_id != previous_id or self._stop.is_set(), timeout=timeout
            )
            return self._frame_id, self._frame

    def set_tracking(self, enabled: bool) -> None:
        with self._lock:
            if enabled and self.calibration is None:
                raise ValueError("Tracking requires a saved camera calibration")
            if enabled:
                self.laser_test.force_off()
                # If operator manually steered gimbal onto target, auto-lock aim so position does not reset
                if self._manual_aim_active and self._latest_target is not None:
                    try:
                        self.lock_current_aim()
                    except Exception as exc:
                        LOGGER.warning("Could not auto-lock manual aim: %s", exc)
                self._manual_aim_active = False
            else:
                self.laser_hardware.set_state(False)
            self._tracking_enabled = enabled
            self._state["tracking_enabled"] = enabled
            self._state["laser_active"] = False
            self._state["status"] = "SEARCHING" if enabled else "PAUSED"
            self._state["message"] = "Automatic tracking enabled" if enabled else "Manual mode"

    def set_auto_laser(self, enabled: bool) -> None:
        with self._lock:
            self.auto_laser_enabled = enabled
            if not enabled:
                self.laser_hardware.set_state(False)
            self._state["auto_laser_enabled"] = enabled
            self._state["laser_active"] = False
            self._state["message"] = (
                "Autonomous laser engagement armed" if enabled else "Autonomous laser disarmed"
            )

    def pulse_laser_test(self) -> None:
        with self._lock:
            if self._tracking_enabled:
                raise ValueError("Pause tracking before using the laser test pulse")
            self.laser_test.pulse()
            self._state["laser_test"] = self.laser_test.snapshot()
            self._state["message"] = "Laser test pulse complete; output forced OFF"

    def move(self, pan: float, tilt: float) -> Angles:
        angles = Angles(
            clamp(float(pan), self.config.gimbal.pan_min, self.config.gimbal.pan_max),
            clamp(float(tilt), self.config.gimbal.tilt_min, self.config.gimbal.tilt_max),
        )
        with self._command_lock:
            self.driver.move(
                angles.pan,
                angles.tilt,
                self.config.gimbal.speed,
                self.config.gimbal.acceleration,
            )
            self.controller.reset(angles)
        with self._lock:
            self._tracking_enabled = False
            self._manual_aim_active = True
            self.laser_hardware.set_state(False)
            self._state.update(
                tracking_enabled=False,
                laser_active=False,
                status="MANUAL",
                pan=angles.pan,
                tilt=angles.tilt,
                message="Manual command sent",
            )
        return angles

    def home(self) -> Angles:
        angles = self.move(self.config.gimbal.home_pan, self.config.gimbal.home_tilt)
        with self._lock:
            self._manual_aim_active = False
        return angles

    def point(self, normalized_x: float, normalized_y: float) -> Angles:
        if self.calibration is None:
            raise ValueError("Click-to-aim requires a saved camera calibration")
        x = clamp(float(normalized_x), 0.0, 1.0) * self.calibration.camera_width
        y = clamp(float(normalized_y), 0.0, 1.0) * self.calibration.camera_height
        cam_angles = self.calibration.map_pixel(x, y)
        compensated = self.alignment_model.compute_compensated_angles(cam_angles)
        angles = self.move(compensated.angles.pan, compensated.angles.tilt)
        with self._lock:
            self._manual_aim_active = True
        return angles

    def lock_current_aim(self) -> dict[str, Any]:
        """Lock current physical gimbal position to match the detected target crosshair.

        Computes the exact trim delta between the gimbal's current physical angles and
        the un-trimmed optical/parallax model angles, updating trim_pan_deg and trim_tilt_deg
        so tracking engages seamlessly with zero position jump.
        """
        with self._lock:
            if self.calibration is None:
                raise ValueError("Aim lock requires a saved camera calibration")
            if self._latest_target is None:
                raise ValueError("No target currently detected to align against")

            current = self.controller.current or Angles(
                self.config.gimbal.home_pan, self.config.gimbal.home_tilt
            )
            height, width = self._latest_frame_shape
            pixel_x, pixel_y = self._latest_target.center
            cam_angles = self.calibration.map_pixel(
                pixel_x * self.calibration.camera_width / width,
                pixel_y * self.calibration.camera_height / height,
            )
            untrimmed = self.alignment_model.compute_compensated_angles(
                cam_angles,
                target=self._latest_target,
                frame_height_px=height,
                apply_trim=False,
            )
            trim_pan = round(current.pan - untrimmed.angles.pan, 2)
            trim_tilt = round(current.tilt - untrimmed.angles.tilt, 2)

            self.alignment_model.config.trim_pan_deg = trim_pan
            self.alignment_model.config.trim_tilt_deg = trim_tilt
            self._manual_aim_active = False

            snapshot = self.alignment_model.config.to_dict()
            self._state["alignment"] = snapshot
            if "alignment_live" in self._state:
                self._state["alignment_live"]["trim_pan_deg"] = trim_pan
                self._state["alignment_live"]["trim_tilt_deg"] = trim_tilt
            self._state["message"] = (
                f"Aim locked to crosshair: Pan trim {trim_pan:+.1f} deg, "
                f"Tilt trim {trim_tilt:+.1f} deg"
            )
            LOGGER.info(
                "Aim locked to crosshair: target=(%.1f, %.1f), current=(%.2f, %.2f), "
                "untrimmed=(%.2f, %.2f) -> trim=(%.2f, %.2f)",
                pixel_x,
                pixel_y,
                current.pan,
                current.tilt,
                untrimmed.angles.pan,
                untrimmed.angles.tilt,
                trim_pan,
                trim_tilt,
            )
            return snapshot

    def get_alignment(self) -> dict[str, Any]:
        with self._lock:
            return self.alignment_model.config.to_dict()

    def update_alignment(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if updates.get("lock_current"):
                return self.lock_current_aim()

            cfg = self.alignment_model.config
            trim_changed = False
            if "enabled" in updates:
                cfg.enabled = bool(updates["enabled"])
            if "auto_range" in updates:
                cfg.auto_range = bool(updates["auto_range"])
            if "trim_pan_deg" in updates:
                cfg.trim_pan_deg = float(updates["trim_pan_deg"])
                trim_changed = True
            if "trim_tilt_deg" in updates:
                cfg.trim_tilt_deg = float(updates["trim_tilt_deg"])
                trim_changed = True
            if "offset_x_m" in updates:
                cfg.offset_x_m = float(updates["offset_x_m"])
            elif "offset_x_in" in updates:
                cfg.offset_x_m = float(updates["offset_x_in"]) * 0.0254
            if "offset_y_m" in updates:
                cfg.offset_y_m = float(updates["offset_y_m"])
            elif "offset_y_in" in updates:
                cfg.offset_y_m = float(updates["offset_y_in"]) * 0.0254
            if "offset_z_m" in updates:
                cfg.offset_z_m = float(updates["offset_z_m"])
            if "nominal_distance_m" in updates:
                cfg.nominal_distance_m = float(updates["nominal_distance_m"])
            elif "nominal_distance_ft" in updates:
                cfg.nominal_distance_m = float(updates["nominal_distance_ft"]) * 0.3048
            if "camera_vfov_deg" in updates:
                cfg.camera_vfov_deg = float(updates["camera_vfov_deg"])
            if "human_height_m" in updates:
                cfg.human_height_m = float(updates["human_height_m"])

            # Live preview of trim actuation when tracking is paused and target is in view
            if trim_changed and not self._tracking_enabled:
                if self._latest_target is not None and self.calibration is not None:
                    height, width = self._latest_frame_shape
                    pixel_x, pixel_y = self._latest_target.center
                    cam_angles = self.calibration.map_pixel(
                        pixel_x * self.calibration.camera_width / width,
                        pixel_y * self.calibration.camera_height / height,
                    )
                    comp = self.alignment_model.compute_compensated_angles(
                        cam_angles, target=self._latest_target, frame_height_px=height
                    )
                    with self._command_lock:
                        self.driver.move(
                            comp.angles.pan,
                            comp.angles.tilt,
                            self.config.gimbal.speed,
                            self.config.gimbal.acceleration,
                        )
                        self.controller.reset(comp.angles)
                    self._state.update(pan=round(comp.angles.pan, 2), tilt=round(comp.angles.tilt, 2))

            snapshot = cfg.to_dict()
            self._state["alignment"] = snapshot
            if "alignment_live" in self._state:
                self._state["alignment_live"]["trim_pan_deg"] = snapshot["trim_pan_deg"]
                self._state["alignment_live"]["trim_tilt_deg"] = snapshot["trim_tilt_deg"]
            self._state["message"] = (
                f"Alignment updated: Pan trim {cfg.trim_pan_deg:+.1f} deg, "
                f"Tilt trim {cfg.trim_tilt_deg:+.1f} deg"
            )
            return snapshot

    def _run(self) -> None:
        camera: OpenCVCamera | None = None
        demo: DemoCamera | None = None
        camera_message = ""
        try:
            if not self.force_demo:
                try:
                    camera = OpenCVCamera(self.config.camera)
                except RuntimeError as exc:
                    if not self.fallback_demo:
                        raise
                    camera_message = f"Camera unavailable ({exc}); using safe demo feed"
                    LOGGER.warning("%s", camera_message)
            if self.force_demo or camera is None:
                demo = DemoCamera(self.config.camera.width, self.config.camera.height)
                mode = "demo"
            else:
                mode = "live"
            with self._lock:
                self._state.update(
                    camera="ONLINE",
                    camera_mode=mode,
                    status="SEARCHING" if self._tracking_enabled else "PAUSED",
                    message=camera_message,
                )

            last_seen: float | None = None
            last_command_time = 0.0
            last_camera_probe = time.monotonic()
            previous_frame_time = time.monotonic()
            fps = 0.0
            consecutive_locks = 0
            laser_engaged_time: float | None = None
            laser_active = False
            while not self._stop.is_set():
                frame_started = time.monotonic()
                if (
                    demo is not None
                    and not self.force_demo
                    and time.monotonic() - last_camera_probe >= 5.0
                ):
                    last_camera_probe = time.monotonic()
                    try:
                        camera = OpenCVCamera(self.config.camera)
                        demo = None
                        mode = "live"
                        self.detector = build_detector(self.config.detector)
                        with self._lock:
                            self._state.update(
                                camera="ONLINE",
                                camera_mode="live",
                                message="USB camera connected; live tracking active",
                            )
                        LOGGER.info("USB camera connected; switched from demo to live input")
                    except RuntimeError:
                        pass
                if demo is not None:
                    frame, direct_target = demo.read()
                    detections = [direct_target]
                else:
                    try:
                        frame = camera.read()  # type: ignore[union-attr]
                        detections = self.detector.detect(frame)
                    except RuntimeError as exc:
                        if not self.fallback_demo:
                            raise
                        camera.close()  # type: ignore[union-attr]
                        camera = None
                        demo = DemoCamera(self.config.camera.width, self.config.camera.height)
                        mode = "demo"
                        last_camera_probe = time.monotonic()
                        frame, direct_target = demo.read()
                        detections = [direct_target]
                        with self._lock:
                            self._state.update(
                                camera="ONLINE",
                                camera_mode="demo",
                                message=f"Camera disconnected ({exc}); simulation resumed",
                            )
                        LOGGER.warning("Camera disconnected; switched to demo input")
                target = choose_target(detections)
                now = time.monotonic()
                desired: Angles | None = None
                target_angles: Angles | None = None
                event_camera = self.event_camera.snapshot()
                event_target = primary_event_target(event_camera)
                parked = False

                with self._lock:
                    tracking_enabled = self._tracking_enabled
                latest_alignment: AlignmentResult | None = None
                if (
                    event_target is not None
                    and self.config.event_camera.track_targets
                    and tracking_enabled
                ):
                    bearing = event_target.get("bearing", {})
                    target_angles = Angles(
                        self.config.gimbal.home_pan
                        + self.config.event_camera.pan_offset_deg
                        + float(bearing.get("azimuth_deg", 0.0)),
                        self.config.gimbal.home_tilt
                        + self.config.event_camera.tilt_offset_deg
                        + float(bearing.get("elevation_deg", 0.0)),
                    )
                    desired = target_angles
                    last_seen = now
                elif target is not None:
                    last_seen = now
                    height, width = frame.shape[:2]
                    with self._lock:
                        self._latest_target = target
                        self._latest_frame_shape = (height, width)
                    if self.calibration is not None:
                        pixel_x, pixel_y = target.center
                        cam_angles = self.calibration.map_pixel(
                            pixel_x * self.calibration.camera_width / width,
                            pixel_y * self.calibration.camera_height / height,
                        )
                        latest_alignment = self.alignment_model.compute_compensated_angles(
                            cam_angles, target=target, frame_height_px=height
                        )
                        target_angles = latest_alignment.angles
                        if tracking_enabled:
                            desired = target_angles
                elif (
                    tracking_enabled
                    and last_seen is not None
                    and now - last_seen >= self.config.tracking.park_after_s
                ):
                    desired = Angles(self.config.gimbal.home_pan, self.config.gimbal.home_tilt)
                    parked = True

                if target is None and (last_seen is None or now - last_seen >= 1.0):
                    with self._lock:
                        self._latest_target = None

                interval = 1.0 / self.config.tracking.command_hz
                if desired is not None and now - last_command_time >= interval:
                    with self._command_lock:
                        commanded = self.controller.update(desired)
                        self.driver.move(
                            commanded.pan,
                            commanded.tilt,
                            self.config.gimbal.speed,
                            self.config.gimbal.acceleration,
                        )
                    last_command_time = now

                current = self.controller.current or Angles(0.0, 0.0)

                # Closed-loop tracking error evaluation and auto laser engagement
                laser_active = False
                with self._lock:
                    auto_laser_on = self.auto_laser_enabled
                if (
                    tracking_enabled
                    and auto_laser_on
                    and desired is not None
                    and target is not None
                ):
                    err_deg = math.hypot(
                        desired.pan - current.pan, desired.tilt - current.tilt
                    )
                    tolerance = getattr(
                        self.config.autonomous_tracker, "lock_tolerance_deg", 1.2
                    )
                    req_locks = getattr(
                        self.config.autonomous_tracker, "lock_consecutive_frames", 3
                    )
                    max_continuous_s = getattr(
                        self.config.autonomous_tracker, "laser_max_continuous_s", 10.0
                    )
                    if err_deg <= tolerance:
                        consecutive_locks += 1
                    else:
                        consecutive_locks = 0

                    if consecutive_locks >= req_locks:
                        if laser_engaged_time is None:
                            laser_engaged_time = now
                        if now - laser_engaged_time <= max_continuous_s:
                            laser_active = True
                        else:
                            laser_active = False
                    else:
                        laser_engaged_time = None
                else:
                    consecutive_locks = 0
                    laser_engaged_time = None

                self.laser_hardware.set_state(laser_active)

                if not tracking_enabled:
                    status = "MANUAL"
                elif parked:
                    status = "PARKING"
                elif event_target is not None and self.config.event_camera.track_targets:
                    status = "EVENT TRACKING"
                elif target is not None:
                    status = "LOCKED // LASER ENGAGED" if laser_active else "TRACKING"
                else:
                    status = "SEARCHING"
                self._draw_overlay(
                    frame, detections, target, current, status, mode, laser_active, latest_alignment
                )
                ok, encoded = self.cv2.imencode(
                    ".jpg", frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), 82]
                )
                if not ok:
                    continue

                delta = max(now - previous_frame_time, 1e-6)
                instantaneous_fps = min(float(self.config.camera.fps), 1.0 / delta)
                fps = instantaneous_fps if fps == 0 else fps * 0.9 + instantaneous_fps * 0.1
                previous_frame_time = now
                height, width = frame.shape[:2]
                target_state = None
                if event_target is not None:
                    centroid = event_target.get("centroid_px", {})
                    target_state = {
                        "x": float(centroid.get("x", 640.0)) / 1280.0,
                        "y": float(centroid.get("y", 360.0)) / 720.0,
                        "label": "event flicker",
                        "confidence": float(event_target.get("confidence", 0.0)),
                        "source": "event",
                    }
                elif target is not None:
                    target_state = {
                        "x": target.center[0] / width,
                        "y": target.center[1] / height,
                        "label": target.label,
                        "confidence": target.confidence,
                        "source": "rgb",
                    }
                alignment_live = {
                    "distance_m": round(
                        latest_alignment.distance_m
                        if latest_alignment
                        else self.config.alignment.nominal_distance_m,
                        2,
                    ),
                    "distance_ft": round(
                        (
                            latest_alignment.distance_m
                            if latest_alignment
                            else self.config.alignment.nominal_distance_m
                        )
                        / 0.3048,
                        1,
                    ),
                    "distance_source": (
                        latest_alignment.distance_source if latest_alignment else "nominal"
                    ),
                    "parallax_pan_deg": round(
                        latest_alignment.parallax_pan_deg if latest_alignment else 0.0, 2
                    ),
                    "parallax_tilt_deg": round(
                        latest_alignment.parallax_tilt_deg if latest_alignment else 0.0, 2
                    ),
                    "trim_pan_deg": round(
                        latest_alignment.trim_pan_deg
                        if latest_alignment
                        else self.config.alignment.trim_pan_deg,
                        2,
                    ),
                    "trim_tilt_deg": round(
                        latest_alignment.trim_tilt_deg
                        if latest_alignment
                        else self.config.alignment.trim_tilt_deg,
                        2,
                    ),
                }
                with self._frame_ready:
                    self._frame = encoded.tobytes()
                    self._frame_id += 1
                    self._state.update(
                        status=status,
                        pan=round(current.pan, 2),
                        tilt=round(current.tilt, 2),
                        target=target_state,
                        fps=round(fps, 1),
                        frame_width=width,
                        frame_height=height,
                        event_camera=event_camera,
                        laser_active=laser_active,
                        auto_laser_enabled=auto_laser_on,
                        alignment_live=alignment_live,
                        predator=build_observer_telemetry(
                            self.config,
                            camera_mode=mode,
                            status=status,
                            target=target,
                            target_angles=target_angles,
                            current_angles=current,
                            event_camera=event_camera,
                        ),
                    )
                    self._frame_ready.notify_all()

                remaining = (1.0 / max(self.config.camera.fps, 1)) - (
                    time.monotonic() - frame_started
                )
                if remaining > 0:
                    self._stop.wait(remaining)
        except (RuntimeError, OSError, ValueError) as exc:
            LOGGER.exception("Dashboard vision loop stopped")
            with self._lock:
                self._state.update(status="FAULT", camera="OFFLINE", message=str(exc))
        finally:
            self.laser_hardware.set_state(False)
            if camera is not None:
                camera.close()

    def _draw_overlay(
        self,
        frame: Any,
        detections: list[Detection],
        target: Detection | None,
        angles: Angles,
        status: str,
        mode: str,
        laser_active: bool = False,
        alignment: AlignmentResult | None = None,
    ) -> None:
        for detection in detections:
            x, y = int(detection.x), int(detection.y)
            right = int(detection.x + detection.width)
            bottom = int(detection.y + detection.height)
            if detection is target:
                color = (0, 0, 255) if laser_active else (73, 248, 174)
            else:
                color = (125, 125, 125)
            self.cv2.rectangle(frame, (x, y), (right, bottom), color, 2)
            label_text = f"{detection.label.upper()} {int(detection.confidence * 100)}%"
            if detection is target and laser_active:
                label_text += " [LASER ON]"
            self.cv2.putText(
                frame,
                label_text,
                (x, max(18, y - 6)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
            if detection is target:
                center = tuple(int(value) for value in detection.center)
                self.cv2.drawMarker(frame, center, color, self.cv2.MARKER_CROSS, 20, 2)
                self.cv2.circle(frame, center, 14, color, 1)
        self.cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (5, 7, 7), -1)
        hud_color = (0, 0, 255) if laser_active else (73, 248, 174)
        self.cv2.putText(
            frame,
            f"{status}  PAN {angles.pan:+06.1f}  TILT {angles.tilt:+05.1f}",
            (15, 30),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            hud_color,
            2,
        )
        mode_text = (
            f"{mode.upper()} / LASER ARMED"
            if laser_active
            else f"{mode.upper()} / OBSERVE ONLY"
        )
        self.cv2.putText(
            frame,
            mode_text,
            (max(frame.shape[1] - 250, 10), 30),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            hud_color if laser_active else (220, 225, 222),
            1,
        )

        # Bottom HUD bar for distance estimation, parallax compensation, and manual trim
        frame_h, frame_w = frame.shape[:2]
        self.cv2.rectangle(frame, (0, frame_h - 26), (frame_w, frame_h), (5, 7, 7), -1)
        if alignment is not None:
            dist_str = f"RNG {alignment.distance_m:.1f}m ({alignment.distance_m / 0.3048:.1f}ft)"
            par_str = (
                f"PARALLAX P{alignment.parallax_pan_deg:+05.1f} T{alignment.parallax_tilt_deg:+04.1f}"
            )
            trim_str = f"TRIM P{alignment.trim_pan_deg:+04.1f} T{alignment.trim_tilt_deg:+04.1f}"
            hud_align_text = f"{dist_str}  |  {par_str}  |  {trim_str}"
        else:
            cfg = self.alignment_model.config
            hud_align_text = (
                f"NOMINAL {cfg.nominal_distance_m:.1f}m ({cfg.nominal_distance_m / 0.3048:.1f}ft)  |  "
                f"TRIM P{cfg.trim_pan_deg:+04.1f} T{cfg.trim_tilt_deg:+04.1f}"
            )

        self.cv2.putText(
            frame,
            hud_align_text,
            (12, frame_h - 8),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 200, 190),
            1,
        )


class DashboardHandler(BaseHTTPRequestHandler):
    engine: DashboardEngine
    server_version = "PredatorObserver/0.3"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._asset("index.html", "text/html; charset=utf-8")
        elif path == "/app.css":
            self._asset("app.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._asset("app.js", "text/javascript; charset=utf-8")
        elif path == "/api/state":
            self._json(HTTPStatus.OK, self.engine.state())
        elif path == "/api/alignment":
            self._json(HTTPStatus.OK, self.engine.get_alignment())
        elif path == "/api/health":
            self._json(
                HTTPStatus.OK,
                {"ok": True, "mode": "OBSERVATION_ONLY", "engagement_enabled": False},
            )
        elif path == "/stream.mjpg":
            self._rgb_stream()
        elif path == "/event-stream.mjpg":
            self._event_stream()
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/tracking":
                self.engine.set_tracking(bool(body.get("enabled")))
            elif path == "/api/move":
                self.engine.move(float(body["pan"]), float(body["tilt"]))
            elif path == "/api/home":
                self.engine.home()
            elif path == "/api/point":
                self.engine.point(float(body["x"]), float(body["y"]))
            elif path == "/api/laser-test/pulse":
                self.engine.pulse_laser_test()
            elif path == "/api/laser-auto":
                self.engine.set_auto_laser(bool(body.get("enabled")))
            elif path in {"/api/alignment", "/api/alignment/lock", "/api/alignment/calibrate"}:
                if path != "/api/alignment" or body.get("lock_current"):
                    result = self.engine.lock_current_aim()
                else:
                    result = self.engine.update_alignment(body)
                self._json(HTTPStatus.OK, result)
                return
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            self._json(HTTPStatus.OK, self.engine.state())
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except RuntimeError as exc:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})

    def _asset(self, name: str, content_type: str) -> None:
        payload = files("vollebak_gimbal").joinpath("web_assets", name).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        length = min(int(self.headers.get("Content-Length", "0")), 16_384)
        return json.loads(self.rfile.read(length) or b"{}")

    def _rgb_stream(self) -> None:
        """Prefer the original MJPEG bytes so the UI never waits on tracking work."""
        source = self.engine.config.camera.source
        state = self.engine.state()
        if (
            state.get("camera_mode") == "live"
            and isinstance(source, str)
            and source.startswith(("http://", "https://"))
            and self._proxy_stream(source)
        ):
            return
        self._processed_stream()

    def _processed_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        frame_id = -1
        try:
            while True:
                frame_id, frame = self.engine.wait_for_frame(frame_id)
                if frame is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return

    def _event_stream(self) -> None:
        if not self.engine.config.event_camera.enabled:
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Event camera is disabled"})
            return
        self._proxy_stream(self.engine.event_camera.stream_url)

    def _proxy_stream(self, source: str) -> bool:
        """Relay an upstream MJPEG stream in small available chunks.

        ``HTTPResponse.read(65536)`` waits to fill a large buffer and can retain
        several JPEG frames. ``read1`` forwards whatever is available now,
        keeping the browser close to the camera's newest frame.
        """
        try:
            with urlopen(
                source,
                timeout=self.engine.config.event_camera.timeout_s,
            ) as upstream:
                content_type = upstream.headers.get(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                read_available = getattr(upstream, "read1", upstream.read)
                while chunk := read_available(8_192):
                    self.wfile.write(chunk)
                return True
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return True
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            LOGGER.warning("MJPEG stream proxy stopped for %s: %s", source, exc)
            return False

    def log_message(self, message: str, *args: object) -> None:
        LOGGER.debug("HTTP %s - %s", self.address_string(), message % args)


def serve_dashboard(
    config: AppConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    force_demo: bool = False,
    fallback_demo: bool = True,
) -> None:
    engine = DashboardEngine(config, force_demo=force_demo, fallback_demo=fallback_demo)

    class Handler(DashboardHandler):
        pass

    Handler.engine = engine
    server = ThreadingHTTPServer((host, port), Handler)
    engine.start()
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    LOGGER.info("Dashboard ready at http://%s:%d", display_host, port)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        LOGGER.info("Dashboard stopping")
    finally:
        server.shutdown()
        server.server_close()
        engine.close()
