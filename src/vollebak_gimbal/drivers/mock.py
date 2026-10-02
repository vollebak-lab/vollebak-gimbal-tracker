from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


class MockGimbal:
    def __init__(self) -> None:
        self.last_command: tuple[float, float, int, int] | None = None

    def move(self, pan: float, tilt: float, speed: int, acceleration: int) -> None:
        command = (pan, tilt, speed, acceleration)
        if command != self.last_command:
            LOGGER.debug(
                "MOCK gimbal -> pan=%+.2f tilt=%+.2f speed=%d accel=%d",
                pan,
                tilt,
                speed,
                acceleration,
            )
            self.last_command = command

    def close(self) -> None:
        LOGGER.info("MOCK gimbal closed")
