from __future__ import annotations

import json
import logging
import threading

import serial

LOGGER = logging.getLogger(__name__)


class WaveshareSerialGimbal:
    """Waveshare General Driver board JSON-over-UART transport.

    The board consumes one compact JSON object per newline. Command T=133 uses
    X for pan, Y for tilt, SPD for speed, and ACC for acceleration.
    """

    def __init__(self, port: str, baud: int = 115200) -> None:
        self.serial = serial.Serial(port=port, baudrate=baud, timeout=0.1, write_timeout=1.0)
        self._lock = threading.Lock()
        LOGGER.info("Opened Waveshare gimbal on %s at %d baud", port, baud)

    @staticmethod
    def encode_move(pan: float, tilt: float, speed: int, acceleration: int) -> bytes:
        payload = {
            "T": 133,
            "X": round(float(pan), 2),
            "Y": round(float(tilt), 2),
            "SPD": int(speed),
            "ACC": int(acceleration),
        }
        return (json.dumps(payload, separators=(",", ":")) + "\n").encode("ascii")

    def move(self, pan: float, tilt: float, speed: int, acceleration: int) -> None:
        message = self.encode_move(pan, tilt, speed, acceleration)
        with self._lock:
            self.serial.write(message)
            self.serial.flush()
        LOGGER.debug("TX %s", message.decode("ascii").rstrip())

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()
