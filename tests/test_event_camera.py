from vollebak_gimbal.config import EventCameraConfig
from vollebak_gimbal.event_camera import EventCameraMonitor, primary_event_target


def test_normalizes_bart_flicker_detector_stats():
    monitor = EventCameraMonitor(EventCameraConfig(enabled=True))
    state = monitor._normalize(
        {
            "status": "ONLINE",
            "resolution": "1280x720",
            "event_rate_ev_s": 19_250_000,
            "num_targets": 2,
            "targets": [
                {"confidence": 0.4, "bearing": {"azimuth_deg": -3.0}},
                {"confidence": 0.9, "bearing": {"azimuth_deg": 7.0}},
            ],
        }
    )

    assert state["connected"] is True
    assert state["event_rate_mev_s"] == 19.25
    assert state["mode"] == "flicker_detector"
    assert primary_event_target(state)["bearing"]["azimuth_deg"] == 7.0


def test_normalizes_legacy_detector_event_window():
    monitor = EventCameraMonitor(EventCameraConfig(enabled=True))
    state = monitor._normalize(
        {
            "ego_motion": {"total_raw_events": 800_000},
            "num_targets": 0,
            "targets": [],
        }
    )

    assert state["event_rate_mev_s"] == 20.0
