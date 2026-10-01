from __future__ import annotations

from typing import Any

from .config import AppConfig
from .event_camera import primary_event_target
from .models import Angles, Detection

UPSTREAM_REPOSITORY = "vollebak-lab/predator"
UPSTREAM_COMMIT = "ea46468"
OPERATING_MODE = "OBSERVATION_ONLY"


def build_observer_telemetry(
    config: AppConfig,
    *,
    camera_mode: str,
    status: str,
    target: Detection | None,
    target_angles: Angles | None,
    current_angles: Angles,
    event_camera: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Map the Pi camera tracker into Bart's layered Predator architecture.

    The Pi profile intentionally ends at target observation and gimbal pointing.
    It has no engagement state, command, or actuator API.
    """

    event_camera = event_camera or {}
    event_online = bool(event_camera.get("connected"))
    event_detector_active = event_camera.get("mode") == "flicker_detector"
    event_target = primary_event_target(event_camera)
    target_locked = target is not None or event_target is not None
    detector_name = str(config.detector.get("type", "motion")).upper()
    driver_live = config.gimbal.driver != "mock"
    event_bearing = event_target.get("bearing", {}) if event_target else {}
    event_label = "ONLINE" if event_online else "NOT CONNECTED"
    if event_online:
        event_label += f" / {float(event_camera.get('event_rate_mev_s', 0.0)):.2f} MEV/S"
    layer3_angles = (
        (
            float(event_bearing.get("azimuth_deg", 0.0)),
            float(event_bearing.get("elevation_deg", 0.0)),
        )
        if event_target
        else (
            (round(target_angles.pan, 2), round(target_angles.tilt, 2))
            if target_angles
            else (None, None)
        )
    )

    return {
        "profile": "RASPBERRY_PI_5_OBSERVER",
        "operating_mode": OPERATING_MODE,
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
        },
        "engagement": {
            "enabled": False,
            "commands_available": False,
            "status": "HARD_DISABLED",
        },
        "layer1": {
            "name": "PASSIVE VISION",
            "status": "TRACK" if target_locked else "SEARCH",
            "sensor": (
                "USB RGB + SONY IMX636"
                if event_online and camera_mode == "live"
                else "SONY IMX636"
                if event_online
                else "USB RGB"
                if camera_mode == "live"
                else "SYNTHETIC"
            ),
            "detector": (
                f"{detector_name} + BART FFT"
                if event_detector_active
                else f"{detector_name} + EVENT STREAM"
                if event_online
                else detector_name
            ),
            "event_camera": event_label,
            "event_mode": event_camera.get("mode") if event_online else None,
            "event_rate_mev_s": event_camera.get("event_rate_mev_s") if event_online else None,
            "bpf_hz": event_target.get("bpf_hz") if event_target else None,
            "rotor_rpm": event_target.get("estimated_rpm") if event_target else None,
            "snr_db": event_target.get("snr_db") if event_target else None,
        },
        "layer2": {
            "name": "RADAR",
            "status": "NOT CONNECTED",
            "sensor": "UHNDER S80",
            "simulated": False,
        },
        "layer3": {
            "name": "TRACK FUSION",
            "status": "BEARING LOCK" if target_angles is not None else "STANDBY",
            "mode": "EVENT FLICKER BEARING" if event_target else "RGB BEARING-ONLY",
            "track_count": int(event_camera.get("num_targets", 0)) if event_target else int(target is not None),
            "azimuth_deg": layer3_angles[0],
            "elevation_deg": layer3_angles[1],
            "confidence": (
                round(float(event_target.get("confidence", 0.0)), 3)
                if event_target
                else round(target.confidence, 3)
                if target
                else None
            ),
        },
        "layer4": {
            "name": "GIMBAL POINTING",
            "status": "HARDWARE READY" if driver_live else "SIMULATED",
            "driver": config.gimbal.driver,
            "control_state": status,
            "pan_deg": round(current_angles.pan, 2),
            "tilt_deg": round(current_angles.tilt, 2),
        },
    }
