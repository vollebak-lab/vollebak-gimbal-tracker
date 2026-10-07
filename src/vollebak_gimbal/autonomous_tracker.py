from __future__ import annotations

import logging
import math
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .calibration import Calibration
from .config import GimbalConfig, TrackingConfig
from .control import SafeAngleController, clamp
from .drivers.base import GimbalDriver
from .models import Angles, Detection

LOGGER = logging.getLogger(__name__)


class TrackerState(str, Enum):
    SEARCHING = "SEARCHING"
    ACQUIRING = "ACQUIRING"
    LOCKED_ENGAGED = "LOCKED_ENGAGED"
    COASTING = "COASTING"
    LOST = "LOST"


@dataclass
class AutonomousTrackerConfig:
    lock_tolerance_deg: float = 1.2
    lock_consecutive_frames: int = 3
    max_engagement_slew_dps: float = 25.0
    coast_timeout_s: float = 0.35
    park_after_s: float = 0.75
    laser_gpio: int = 17
    laser_auto_engage: bool = True
    laser_max_continuous_s: float = 10.0
    command_hz: float = 50.0


class LaserHardwareInterface:
    """Fail-safe interface to control laser emitter via GPIO pin."""

    def __init__(
        self,
        gpio: int = 17,
        *,
        runner: Callable[..., Any] = subprocess.run,
        pinctrl_path: str | None = None,
    ) -> None:
        self.gpio = gpio
        self._runner = runner
        self._pinctrl = pinctrl_path or shutil.which("pinctrl")
        self._state = False
        self._lock = threading.Lock()

        # Ensure safe initial state (laser OFF)
        self.set_state(False)

    @property
    def is_on(self) -> bool:
        return self._state

    def set_state(self, enabled: bool) -> None:
        with self._lock:
            if not self._pinctrl:
                self._state = enabled
                return
            level = "dh" if enabled else "dl"
            try:
                self._runner(
                    [self._pinctrl, "set", str(self.gpio), "op", level],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=0.5,
                )
                self._state = enabled
            except (OSError, subprocess.SubprocessError) as exc:
                LOGGER.warning("Laser GPIO set failed: %s", exc)
                self._state = False


class AutonomousLaserTracker:
    """Closed-loop visual servoing and autonomous laser director.

    Directs the pan/tilt gimbal at a person's center-of-mass and autonomously
    activates the laser emitter when locked within alignment tolerances.
    """

    def __init__(
        self,
        gimbal_config: GimbalConfig,
        tracking_config: TrackingConfig,
        tracker_config: AutonomousTrackerConfig,
        calibration: Calibration | None,
        driver: GimbalDriver,
        laser: LaserHardwareInterface | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gimbal_config = gimbal_config
        self.tracking_config = tracking_config
        self.tracker_config = tracker_config
        self.calibration = calibration
        self.driver = driver
        self.clock = clock
        self.laser = laser or LaserHardwareInterface(gpio=tracker_config.laser_gpio)
        self.controller = SafeAngleController(gimbal_config, tracking_config)

        self._state = TrackerState.SEARCHING
        self._consecutive_locks = 0
        self._last_seen_time: float | None = None
        self._last_command_time = 0.0
        self._laser_engaged_time: float | None = None
        self._last_desired_angles: Angles | None = None
        self._last_commanded_angles: Angles = Angles(
            gimbal_config.home_pan, gimbal_config.home_tilt
        )
        self._current_error_deg = 0.0
        self._lock = threading.Lock()

        # Initialize gimbal at home position
        self.controller.reset(self._last_commanded_angles)

    @property
    def state(self) -> TrackerState:
        with self._lock:
            return self._state

    @property
    def error_deg(self) -> float:
        with self._lock:
            return self._current_error_deg

    def update(
        self,
        target: Detection | None,
        frame_width: int,
        frame_height: int,
        current_time: float | None = None,
    ) -> tuple[Angles, TrackerState, bool]:
        """Update tracker cycle with target detection.

        Returns:
            (commanded_angles, tracker_state, laser_active)
        """
        now = self.clock() if current_time is None else current_time
        with self._lock:
            desired_angles: Angles | None = None

            if target is not None and self.calibration is not None:
                self._last_seen_time = now
                # Map person center-of-mass to gimbal pan/tilt
                cx, cy = target.center
                scaled_x = cx * self.calibration.camera_width / float(frame_width)
                scaled_y = cy * self.calibration.camera_height / float(frame_height)
                desired_angles = self.calibration.map_pixel(scaled_x, scaled_y)
                self._last_desired_angles = desired_angles

                # Compute current angular alignment error
                current_pan = self._last_commanded_angles.pan
                current_tilt = self._last_commanded_angles.tilt
                d_pan = desired_angles.pan - current_pan
                d_tilt = desired_angles.tilt - current_tilt
                error_deg = math.hypot(d_pan, d_tilt)
                self._current_error_deg = error_deg

                # Check lock criteria
                in_tolerance = error_deg <= self.tracker_config.lock_tolerance_deg
                if in_tolerance:
                    self._consecutive_locks += 1
                else:
                    self._consecutive_locks = 0

                if self._consecutive_locks >= self.tracker_config.lock_consecutive_frames:
                    self._state = TrackerState.LOCKED_ENGAGED
                else:
                    self._state = TrackerState.ACQUIRING

            elif self._last_seen_time is not None:
                elapsed = now - self._last_seen_time
                if elapsed <= self.tracker_config.coast_timeout_s:
                    # Target briefly occluded; coast with last known position
                    self._state = TrackerState.COASTING
                    desired_angles = self._last_desired_angles
                    self._consecutive_locks = 0
                elif elapsed <= self.tracker_config.park_after_s:
                    self._state = TrackerState.LOST
                    desired_angles = self._last_desired_angles
                    self._consecutive_locks = 0
                else:
                    # Return to safe park/home position
                    self._state = TrackerState.SEARCHING
                    desired_angles = Angles(
                        self.gimbal_config.home_pan, self.gimbal_config.home_tilt
                    )
                    self._consecutive_locks = 0
            else:
                self._state = TrackerState.SEARCHING
                desired_angles = Angles(
                    self.gimbal_config.home_pan, self.gimbal_config.home_tilt
                )
                self._consecutive_locks = 0
                self._current_error_deg = 0.0

            # Execute servo movement command rate limiting
            interval = 1.0 / max(1.0, self.tracker_config.command_hz)
            if desired_angles is not None and (now - self._last_command_time) >= interval:
                commanded = self.controller.update(desired_angles)
                self.driver.move(
                    commanded.pan,
                    commanded.tilt,
                    self.gimbal_config.speed,
                    self.gimbal_config.acceleration,
                )
                self._last_commanded_angles = commanded
                self._last_command_time = now

            # Determine laser engagement policy
            should_fire_laser = False
            if (
                self._state == TrackerState.LOCKED_ENGAGED
                and self.tracker_config.laser_auto_engage
            ):
                # Fail-safe check: maximum continuous engagement duration
                if self._laser_engaged_time is None:
                    self._laser_engaged_time = now
                engaged_duration = now - self._laser_engaged_time
                if engaged_duration <= self.tracker_config.laser_max_continuous_s:
                    should_fire_laser = True
                else:
                    LOGGER.warning("Laser thermal/safety limit reached (%0.1fs)", engaged_duration)
                    should_fire_laser = False
            else:
                self._laser_engaged_time = None
                should_fire_laser = False

            # Drive hardware laser state
            if self.laser.is_on != should_fire_laser:
                self.laser.set_state(should_fire_laser)

            return self._last_commanded_angles, self._state, self.laser.is_on

    def emergency_stop(self) -> None:
        """Immediately drop laser and halt tracking."""
        with self._lock:
            self.laser.set_state(False)
            self._state = TrackerState.LOST
            self._consecutive_locks = 0
            self._laser_engaged_time = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state.value,
                "consecutive_locks": self._consecutive_locks,
                "current_error_deg": round(self._current_error_deg, 2),
                "laser_active": self.laser.is_on,
                "commanded_pan": round(self._last_commanded_angles.pan, 2),
                "commanded_tilt": round(self._last_commanded_angles.tilt, 2),
            }
