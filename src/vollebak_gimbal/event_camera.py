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
            "num_tracks": 0,
            "tracks": [],
            "roi_diagnostics": {},
            "ego_motion": {},
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
        tracks = payload.get("tracks", [])
        if not isinstance(tracks, list):
            tracks = []
        tracks = [track for track in tracks if isinstance(track, dict)]
        raw_roi = payload.get("roi_diagnostics", {})
        if not isinstance(raw_roi, dict):
            raw_roi = {}
        roi_diagnostics = {
            "total_events": int(raw_roi.get("total_events", 0) or 0),
            "max_cell_events": int(raw_roi.get("max_cell_events", 0) or 0),
            "max_sieve_hits": int(raw_roi.get("max_sieve_hits", 0) or 0),
            "active_cells": int(raw_roi.get("active_cells", 0) or 0),
        }
        raw_ego_motion = payload.get("ego_motion", {})
        if not isinstance(raw_ego_motion, dict):
            raw_ego_motion = {}
        event_rate = float(payload.get("event_rate_ev_s", 0.0) or 0.0)
        if event_rate <= 0 and raw_ego_motion:
            # Older Bart detector builds expose a 25 Hz event-count window only.
            event_rate = float(raw_ego_motion.get("total_raw_events", 0.0) or 0.0) * 25.0
        ego_motion = {
            "imu_connected": bool(raw_ego_motion.get("imu_connected", False)),
            "imu_packets": int(raw_ego_motion.get("imu_packets", 0) or 0),
            "trt_suppression_active": bool(
                raw_ego_motion.get("trt_suppression_active", False)
            ),
            "spectral_combnet_active": bool(
                raw_ego_motion.get("spectral_combnet_active", False)
            ),
            "spectral_eval_cells": int(raw_ego_motion.get("spectral_eval_cells", 0) or 0),
            "spectral_detections": int(raw_ego_motion.get("spectral_detections", 0) or 0),
            "gyro_speed_deg_s": float(raw_ego_motion.get("gyro_speed_deg_s", 0.0) or 0.0),
            "active_cells": int(raw_ego_motion.get("active_cells", 0) or 0),
            "foliage_dispersion_pct": float(
                raw_ego_motion.get("foliage_dispersion_pct", 0.0) or 0.0
            ),
            "suppressed_events_pct": float(
                raw_ego_motion.get("suppressed_events_pct", 0.0) or 0.0
            ),
            "total_raw_events": int(raw_ego_motion.get("total_raw_events", 0) or 0),
            "retained_imo_events": int(
                raw_ego_motion.get("retained_imo_events", 0) or 0
            ),
        }
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
            "num_tracks": int(payload.get("num_tracks", len(tracks)) or 0),
            "tracks": tracks,
            "roi_diagnostics": roi_diagnostics,
            "ego_motion": ego_motion,
            "error": None,
        }


def primary_event_target(event_camera: dict[str, Any]) -> dict[str, Any] | None:
    targets = event_camera.get("targets", [])
    if not targets:
        return None
    return max(targets, key=lambda target: float(target.get("confidence", 0.0) or 0.0))
