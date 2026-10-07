from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from .config import LaserTestConfig

LOGGER = logging.getLogger(__name__)


class LaserTestController:
    """Fail-off controller for brief, operator-initiated alignment pulses."""

    def __init__(
        self,
        config: LaserTestConfig,
        *,
        runner: Callable[..., Any] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        pinctrl_path: str | None = None,
    ) -> None:
        self.config = config
        self._runner = runner
        self._sleep = sleeper
        self._clock = clock
        self._pinctrl = pinctrl_path or shutil.which("pinctrl")
        self._lock = threading.Lock()
        self._active = False
        self._last_pulse_at: float | None = None
        self._available = bool(config.enabled and self._pinctrl)
        if config.enabled and not self._available:
            LOGGER.warning("Laser test is enabled but pinctrl was not found; control disabled")
        if self._available:
            self.force_off()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            remaining = 0.0
            if self._last_pulse_at is not None:
                remaining = max(
                    0.0,
                    self.config.cooldown_s - (self._clock() - self._last_pulse_at),
                )
            return {
                "available": self._available,
                "active": self._active,
                "pulse_ms": round(self.config.pulse_s * 1000),
                "cooldown_remaining_s": round(remaining, 2),
                "tracking_interlock": True,
            }

    def pulse(self) -> None:
        with self._lock:
            if not self._available:
                raise ValueError("Laser test control is unavailable")
            now = self._clock()
            if (
                self._last_pulse_at is not None
                and now - self._last_pulse_at < self.config.cooldown_s
            ):
                raise ValueError("Laser test cooldown is active")

            self._write(False)
            self._active = True
            try:
                self._write(True)
                self._sleep(self.config.pulse_s)
            finally:
                try:
                    self._write(False)
                finally:
                    self._active = False
                    self._last_pulse_at = self._clock()

    def force_off(self) -> None:
        with self._lock:
            self._active = False
            if self._available:
                self._write(False)

    def _write(self, enabled: bool) -> None:
        level = "dh" if enabled else "dl"
        try:
            self._runner(
                [self._pinctrl, "set", str(self.config.gpio), "op", level],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Unable to set laser test GPIO safely: {exc}") from exc
