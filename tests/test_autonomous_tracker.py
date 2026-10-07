from __future__ import annotations

import numpy as np
import pytest

from vollebak_gimbal.autonomous_tracker import (
    AutonomousLaserTracker,
    AutonomousTrackerConfig,
    LaserHardwareInterface,
    TrackerState,
)
from vollebak_gimbal.calibration import Calibration
from vollebak_gimbal.config import GimbalConfig, TrackingConfig
from vollebak_gimbal.drivers.mock import MockGimbal
from vollebak_gimbal.models import Angles, Detection


class MockLaserHardware(LaserHardwareInterface):
    def __init__(self, gpio: int = 17) -> None:
        self.gpio = gpio
        self._state = False

    @property
    def is_on(self) -> bool:
        return self._state

    def set_state(self, enabled: bool) -> None:
        self._state = enabled


def create_test_calibration() -> Calibration:
    # Calibration matrix where (320, 240) maps to (0, 0)
    # pan = (x - 320) * 0.1, tilt = (y - 240) * 0.1
    matrix = np.array([
        [0.1, 0.0, -32.0],
        [0.0, 0.1, -24.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    return Calibration(matrix=matrix, camera_width=640, camera_height=480)


def create_test_tracker(laser: MockLaserHardware | None = None) -> tuple[AutonomousLaserTracker, MockLaserHardware, MockGimbal]:
    gimbal_config = GimbalConfig(
        driver="mock",
        pan_min=-30.0,
        pan_max=30.0,
        tilt_min=-10.0,
        tilt_max=15.0,
        home_pan=0.0,
        home_tilt=0.0,
    )
    tracking_config = TrackingConfig(
        command_hz=100.0,
        smoothing_alpha=1.0,  # Fast response for test determinism
        deadband_deg=0.0,
        max_step_deg=60.0,
    )
    tracker_config = AutonomousTrackerConfig(
        lock_tolerance_deg=1.0,
        lock_consecutive_frames=3,
        laser_auto_engage=True,
        laser_max_continuous_s=5.0,
        coast_timeout_s=0.3,
        park_after_s=0.6,
        command_hz=100.0,
    )
    mock_laser = laser or MockLaserHardware()
    mock_driver = MockGimbal()
    calib = create_test_calibration()

    tracker = AutonomousLaserTracker(
        gimbal_config=gimbal_config,
        tracking_config=tracking_config,
        tracker_config=tracker_config,
        calibration=calib,
        driver=mock_driver,
        laser=mock_laser,
    )
    return tracker, mock_laser, mock_driver


def test_tracker_initial_state():
    tracker, laser, driver = create_test_tracker()
    assert tracker.state == TrackerState.SEARCHING
    assert laser.is_on is False


def test_tracker_lock_and_laser_engagement():
    tracker, laser, driver = create_test_tracker()
    t = 100.0

    # Person at center (320, 240) -> maps to (0.0, 0.0)
    det = Detection(x=280.0, y=200.0, width=80.0, height=80.0, custom_center=(320.0, 240.0))

    # Frame 1: Initial detection -> ACQUIRING, laser OFF
    angles, state, laser_on = tracker.update(det, 640, 480, current_time=t)
    assert state == TrackerState.ACQUIRING
    assert laser_on is False
    assert laser.is_on is False

    # Frame 2: Still ACQUIRING (2 consecutive locks)
    t += 0.02
    angles, state, laser_on = tracker.update(det, 640, 480, current_time=t)
    assert state == TrackerState.ACQUIRING
    assert laser_on is False

    # Frame 3: 3 consecutive locks reached -> LOCKED_ENGAGED, Laser fires!
    t += 0.02
    angles, state, laser_on = tracker.update(det, 640, 480, current_time=t)
    assert state == TrackerState.LOCKED_ENGAGED
    assert laser_on is True
    assert laser.is_on is True


def test_tracker_laser_failsafe_on_target_jump():
    tracker, laser, driver = create_test_tracker()
    t = 100.0

    # Person at center
    det_center = Detection(x=280.0, y=200.0, width=80.0, height=80.0, custom_center=(320.0, 240.0))

    # Lock target
    for _ in range(4):
        t += 0.02
        tracker.update(det_center, 640, 480, current_time=t)

    assert tracker.state == TrackerState.LOCKED_ENGAGED
    assert laser.is_on is True

    # Target suddenly jumps to x=500 -> maps to pan = +18 deg
    det_jump = Detection(x=460.0, y=200.0, width=80.0, height=80.0, custom_center=(500.0, 240.0))
    t += 0.02
    angles, state, laser_on = tracker.update(det_jump, 640, 480, current_time=t)

    # Laser must immediately drop to False!
    assert state == TrackerState.ACQUIRING
    assert laser_on is False
    assert laser.is_on is False


def test_tracker_coasting_and_parking():
    tracker, laser, driver = create_test_tracker()
    t = 100.0

    det_center = Detection(x=280.0, y=200.0, width=80.0, height=80.0, custom_center=(320.0, 240.0))
    for _ in range(4):
        t += 0.02
        tracker.update(det_center, 640, 480, current_time=t)
    assert laser.is_on is True

    # Target disappears
    t += 0.05
    angles, state, laser_on = tracker.update(None, 640, 480, current_time=t)
    assert state == TrackerState.COASTING
    assert laser_on is False  # Immediately extinguished
    assert laser.is_on is False

    # After coast timeout (0.3s)
    t += 0.35
    angles, state, laser_on = tracker.update(None, 640, 480, current_time=t)
    assert state == TrackerState.LOST

    # After park timeout (0.6s)
    t += 0.35
    angles, state, laser_on = tracker.update(None, 640, 480, current_time=t)
    assert state == TrackerState.SEARCHING


def test_tracker_laser_thermal_duration_cutoff():
    tracker, laser, driver = create_test_tracker()
    t = 100.0

    det_center = Detection(x=280.0, y=200.0, width=80.0, height=80.0, custom_center=(320.0, 240.0))
    # Lock target
    for _ in range(4):
        t += 0.02
        tracker.update(det_center, 640, 480, current_time=t)
    assert laser.is_on is True

    # Advance beyond max continuous duration (5.0s)
    t += 5.1
    angles, state, laser_on = tracker.update(det_center, 640, 480, current_time=t)
    # Laser should cut off due to thermal/safety limit
    assert laser_on is False
    assert laser.is_on is False


def test_tracker_emergency_stop():
    tracker, laser, driver = create_test_tracker()
    t = 100.0
    det_center = Detection(x=280.0, y=200.0, width=80.0, height=80.0, custom_center=(320.0, 240.0))
    for _ in range(4):
        t += 0.02
        tracker.update(det_center, 640, 480, current_time=t)
    assert laser.is_on is True

    tracker.emergency_stop()
    assert laser.is_on is False
    assert tracker.state == TrackerState.LOST
