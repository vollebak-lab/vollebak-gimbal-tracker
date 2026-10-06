from vollebak_gimbal.config import load_config
from vollebak_gimbal.models import Angles, Detection
from vollebak_gimbal.predator_observer import build_observer_telemetry


def test_predator_pi_profile_is_observation_only():
    config = load_config("config/dev.yaml")
    target = Detection(10, 20, 30, 40, 0.82, "motion")

    telemetry = build_observer_telemetry(
        config,
        camera_mode="live",
        status="TRACKING",
        target=target,
        target_angles=Angles(12.5, -3.0),
        current_angles=Angles(10.0, -2.0),
    )

    assert telemetry["engagement"] == {
        "enabled": False,
        "commands_available": False,
        "status": "HARD_DISABLED",
    }
    assert telemetry["layer1"]["sensor"] == "USB RGB"
    assert telemetry["layer2"]["status"] == "NOT CONNECTED"
    assert telemetry["layer3"]["azimuth_deg"] == 12.5
    assert telemetry["layer4"]["status"] == "SIMULATED"


def test_predator_uses_live_event_target_bearing():
    config = load_config("config/dev.yaml")
    event_camera = {
        "connected": True,
        "mode": "flicker_detector",
        "event_rate_mev_s": 19.25,
        "ego_motion": {
            "imu_connected": True,
            "imu_packets": 900,
            "spectral_combnet_active": True,
            "spectral_eval_cells": 7,
            "spectral_detections": 1,
        },
        "num_targets": 1,
        "num_tracks": 2,
        "roi_diagnostics": {"active_cells": 14, "max_sieve_hits": 5},
        "targets": [
            {
                "bpf_hz": 180.0,
                "estimated_rpm": 5400.0,
                "confidence": 0.91,
                "snr_db": 14.2,
                "bearing": {"azimuth_deg": 4.5, "elevation_deg": -1.25},
            }
        ],
    }

    telemetry = build_observer_telemetry(
        config,
        camera_mode="live",
        status="EVENT TRACKING",
        target=None,
        target_angles=None,
        current_angles=Angles(4.0, -1.0),
        event_camera=event_camera,
    )

    assert telemetry["layer1"]["event_camera"] == "ONLINE / 19.25 MEV/S"
    assert telemetry["layer1"]["detector"] == "MOTION + BART COMBNET"
    assert telemetry["layer1"]["imu_connected"] is True
    assert telemetry["layer1"]["spectral_detections"] == 1
    assert telemetry["layer1"]["event_tracks"] == 2
    assert telemetry["layer1"]["roi_active_cells"] == 14
    assert telemetry["layer1"]["roi_max_sieve_hits"] == 5
    assert telemetry["layer1"]["bpf_hz"] == 180.0
    assert telemetry["layer3"]["mode"] == "EVENT FLICKER BEARING"
    assert telemetry["layer3"]["azimuth_deg"] == 4.5
    assert telemetry["engagement"]["enabled"] is False
