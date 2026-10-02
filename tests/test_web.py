import time

from vollebak_gimbal.config import load_config
from vollebak_gimbal.web import DashboardEngine


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
