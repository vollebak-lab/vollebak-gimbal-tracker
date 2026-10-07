from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CameraConfig:
    source: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "auto"


@dataclass(slots=True)
class EventCameraConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8081"
    poll_interval_s: float = 0.25
    timeout_s: float = 1.0
    track_targets: bool = False
    pan_offset_deg: float = 0.0
    tilt_offset_deg: float = 0.0


@dataclass(slots=True)
class CalibrationConfig:
    path: str = "config/calibration.json"


@dataclass(slots=True)
class GimbalConfig:
    driver: str = "mock"
    serial_port: str = "/dev/ttyUSB0"
    baud: int = 115200
    pan_min: float = -170.0
    pan_max: float = 170.0
    tilt_min: float = -25.0
    tilt_max: float = 80.0
    home_pan: float = 0.0
    home_tilt: float = 0.0
    speed: int = 60
    acceleration: int = 20


@dataclass(slots=True)
class TrackingConfig:
    start_enabled: bool = False
    command_hz: float = 10.0
    smoothing_alpha: float = 0.25
    deadband_deg: float = 1.0
    max_step_deg: float = 4.0
    lost_hold_s: float = 1.0
    park_after_s: float = 5.0


@dataclass(slots=True)
class LaserTestConfig:
    enabled: bool = False
    gpio: int = 17
    pulse_s: float = 0.1
    cooldown_s: float = 2.0


@dataclass(slots=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    event_camera: EventCameraConfig = field(default_factory=EventCameraConfig)
    detector: dict[str, Any] = field(default_factory=lambda: {"type": "motion"})
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    laser_test: LaserTestConfig = field(default_factory=LaserTestConfig)
    config_dir: Path = field(default_factory=Path.cwd, repr=False)

    def resolve(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.config_dir / path


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError("Top-level config must be a mapping")

    config = AppConfig(
        camera=CameraConfig(**_section(data, "camera")),
        event_camera=EventCameraConfig(**_section(data, "event_camera")),
        detector=_section(data, "detector"),
        calibration=CalibrationConfig(**_section(data, "calibration")),
        gimbal=GimbalConfig(**_section(data, "gimbal")),
        tracking=TrackingConfig(**_section(data, "tracking")),
        laser_test=LaserTestConfig(**_section(data, "laser_test")),
        config_dir=config_path.parent,
    )
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    g = config.gimbal
    t = config.tracking
    if g.pan_min >= g.pan_max or g.tilt_min >= g.tilt_max:
        raise ValueError("Gimbal minimum limits must be below maximum limits")
    if not (g.pan_min <= g.home_pan <= g.pan_max):
        raise ValueError("home_pan is outside pan limits")
    if not (g.tilt_min <= g.home_tilt <= g.tilt_max):
        raise ValueError("home_tilt is outside tilt limits")
    if g.driver not in {"mock", "waveshare_serial"}:
        raise ValueError("gimbal.driver must be mock or waveshare_serial")
    if config.camera.backend not in {"auto", "dshow", "msmf", "v4l2"}:
        raise ValueError("camera.backend must be auto, dshow, msmf, or v4l2")
    if not (0.0 < t.smoothing_alpha <= 1.0):
        raise ValueError("smoothing_alpha must be in (0, 1]")
    if t.command_hz <= 0 or t.max_step_deg <= 0:
        raise ValueError("command_hz and max_step_deg must be positive")
    if t.park_after_s < t.lost_hold_s:
        raise ValueError("park_after_s must be >= lost_hold_s")
    laser = config.laser_test
    if not (0 <= laser.gpio <= 27):
        raise ValueError("laser_test.gpio must be a Raspberry Pi header GPIO (0-27)")
    if not (0.02 <= laser.pulse_s <= 0.1):
        raise ValueError("laser_test.pulse_s must be between 0.02 and 0.1 seconds")
    if laser.cooldown_s < 1.0:
        raise ValueError("laser_test.cooldown_s must be at least 1 second")
    e = config.event_camera
    if not e.base_url.startswith(("http://", "https://")):
        raise ValueError("event_camera.base_url must be an HTTP(S) URL")
    if e.poll_interval_s <= 0 or e.timeout_s <= 0:
        raise ValueError("event camera polling and timeout values must be positive")
