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


def test_normalizes_latest_bart_imu_and_spectral_diagnostics():
    monitor = EventCameraMonitor(EventCameraConfig(enabled=True))
    state = monitor._normalize(
        {
            "num_targets": 0,
            "targets": [],
            "num_tracks": 1,
            "tracks": [{"track_id": 4, "state": "TENTATIVE"}],
            "roi_diagnostics": {
                "total_events": 8100,
                "max_cell_events": 73,
                "max_sieve_hits": 3,
                "active_cells": 12,
            },
            "ego_motion": {
                "imu_connected": True,
                "imu_packets": 4200,
                "trt_suppression_active": True,
                "spectral_combnet_active": True,
                "spectral_eval_cells": 18,
                "spectral_detections": 2,
                "suppressed_events_pct": 37.5,
            },
        }
    )

    assert state["ego_motion"]["imu_connected"] is True
    assert state["ego_motion"]["imu_packets"] == 4200
    assert state["ego_motion"]["spectral_combnet_active"] is True
    assert state["ego_motion"]["spectral_detections"] == 2
    assert state["ego_motion"]["suppressed_events_pct"] == 37.5
    assert state["num_tracks"] == 1
    assert state["tracks"][0]["state"] == "TENTATIVE"
    assert state["roi_diagnostics"]["active_cells"] == 12
    assert state["roi_diagnostics"]["max_sieve_hits"] == 3
