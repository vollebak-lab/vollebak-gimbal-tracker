from __future__ import annotations

import json
import logging
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import EventCameraConfig

LOGGER = logging.getLogger(__name__)


class EventCameraMonitor:
    """Poll Bart's OpenEB service without blocking the RGB tracking loop."""

    def __init__(self, config: EventCameraConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "enabled": config.enabled,
            "connected": False,
            "status": "CONNECTING" if config.enabled else "DISABLED",
            "sensor": "Sony IMX636 HD",
            "integrator": "IDS Imaging Development Systems",
            "resolution": None,
            "event_rate_ev_s": 0.0,
            "event_rate_mev_s": 0.0,
            "stream_fps": 0.0,
            "mode": "unknown",
            "num_targets": 0,
            "targets": [],
            "error": None,
        }

    @property
    def stream_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/stream.mjpg"

    def start(self) -> None:
        if not self.config.enabled or (self._thread and self._thread.is_alive()):
            return
        self._thread = threading.Thread(target=self._run, name="event-camera-monitor", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.config.timeout_s + 0.5))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def _run(self) -> None:
        was_connected: bool | None = None
        while not self._stop.is_set():
            try:
                request = Request(
                    f"{self.config.base_url.rstrip('/')}/stats",
                    headers={"Accept": "application/json", "User-Agent": "VollebakObserver/0.4"},
                )
                with urlopen(request, timeout=self.config.timeout_s) as response:
                    payload = json.load(response)
                if not isinstance(payload, dict):
                    raise TypeError("event-camera /stats response is not an object")
                state = self._normalize(payload)
                connected = True
            except (
                HTTPError,
                URLError,
                TimeoutError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                state = {
                    **self.snapshot(),
                    "connected": False,
                    "status": "OFFLINE",
                    "num_targets": 0,
                    "targets": [],
                    "error": str(exc),
                }
                connected = False

            with self._lock:
                self._state = state
            if connected != was_connected:
                log = LOGGER.info if connected else LOGGER.warning
                log("Event camera %s", "online" if connected else f"offline: {state['error']}")
                was_connected = connected
            self._stop.wait(self.config.poll_interval_s)

    def _normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        targets = payload.get("targets", [])
        if not isinstance(targets, list):
            targets = []
        targets = [target for target in targets if isinstance(target, dict)]
        event_rate = float(payload.get("event_rate_ev_s", 0.0) or 0.0)
        if event_rate <= 0 and isinstance(payload.get("ego_motion"), dict):
            # Older Bart detector builds expose a 25 Hz event-count window only.
            event_rate = float(payload["ego_motion"].get("total_raw_events", 0.0) or 0.0) * 25.0
        return {
            "enabled": True,
            "connected": True,
            "status": str(payload.get("status", "ONLINE")),
            "sensor": str(payload.get("sensor", "Sony IMX636 HD")),
            "integrator": str(
                payload.get("integrator", "IDS Imaging Development Systems")
            ),
            "resolution": payload.get("resolution", "1280x720"),
            "event_rate_ev_s": round(event_rate, 1),
            "event_rate_mev_s": round(event_rate / 1_000_000.0, 3),
            "stream_fps": float(payload.get("stream_fps", 30.0) or 0.0),
            "mode": "flicker_detector" if "num_targets" in payload else "event_viewer",
            "num_targets": int(payload.get("num_targets", len(targets)) or 0),
            "targets": targets,
            "error": None,
        }


def primary_event_target(event_camera: dict[str, Any]) -> dict[str, Any] | None:
    targets = event_camera.get("targets", [])
    if not targets:
        return None
    return max(targets, key=lambda target: float(target.get("confidence", 0.0) or 0.0))
