from __future__ import annotations

from typing import Protocol

from ..config import GimbalConfig


class GimbalDriver(Protocol):
    def move(self, pan: float, tilt: float, speed: int, acceleration: int) -> None: ...

    def close(self) -> None: ...


def build_driver(config: GimbalConfig) -> GimbalDriver:
    if config.driver == "mock":
        from .mock import MockGimbal

        return MockGimbal()
    if config.driver == "waveshare_serial":
        from .waveshare import WaveshareSerialGimbal

        return WaveshareSerialGimbal(config.serial_port, config.baud)
    raise ValueError(f"Unknown gimbal driver: {config.driver}")
