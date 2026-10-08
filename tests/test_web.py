import time
from dataclasses import replace

from vollebak_gimbal.config import load_config
from vollebak_gimbal.web import DashboardEngine


def test_dashboard_starts_disarmed_by_default():
    config = load_config("config/dev.yaml")
    config.tracking = replace(config.tracking, start_enabled=False)
    engine = DashboardEngine(config, force_demo=True)
    try:
        state = engine.state()
        assert state["tracking_enabled"] is False
    finally:
        engine.close()


def test_demo_dashboard_tracks_and_accepts_manual_commands():
    config = load_config("config/dev.yaml")
    config.event_camera.enabled = False
    engine = DashboardEngine(config, force_demo=True)
    engine.start()
    try:
        deadline = time.monotonic() + 2.0
        state = engine.state()
        while state["target"] is None and time.monotonic() < deadline:
            time.sleep(0.02)
            state = engine.state()

        assert state["camera_mode"] == "demo"
        assert state["status"] == "TRACKING"
        assert state["target"]["label"] == "demo"
        assert state["predator"]["operating_mode"] == "OBSERVATION_ONLY"
        assert state["predator"]["engagement"]["enabled"] is False
        assert state["predator"]["layer3"]["track_count"] == 1

        angles = engine.move(999, -999)
        assert angles.pan == config.gimbal.pan_max
        assert angles.tilt == config.gimbal.tilt_min
        assert engine.state()["tracking_enabled"] is False
    finally:
        engine.close()


def test_dashboard_auto_laser_engagement():
    config = load_config("config/dev.yaml")
    config.event_camera.enabled = False
    config.autonomous_tracker = replace(
        config.autonomous_tracker,
        enabled=True,
        laser_auto_engage=True,
        lock_tolerance_deg=100.0,
        lock_consecutive_frames=1,
    )
    engine = DashboardEngine(config, force_demo=True)
    engine.start()
    try:
        deadline = time.monotonic() + 2.0
        state = engine.state()
        while not state.get("laser_active") and time.monotonic() < deadline:
            time.sleep(0.02)
            state = engine.state()

        assert state["auto_laser_enabled"] is True
        assert state["laser_active"] is True
        assert state["status"] == "LOCKED // LASER ENGAGED"

        # Disarming auto laser shuts off output
        engine.set_auto_laser(False)
        assert engine.state()["auto_laser_enabled"] is False
        assert engine.state()["laser_active"] is False

        # Pausing tracking shuts off output
        engine.set_auto_laser(True)
        engine.set_tracking(False)
        assert engine.state()["tracking_enabled"] is False
        assert engine.state()["laser_active"] is False
    finally:
        engine.close()

