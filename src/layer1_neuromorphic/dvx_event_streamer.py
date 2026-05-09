# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Thread-per-camera USB streaming with graceful reconnect for helmet-mounted event camera array
#   FAILURE_MODE: Single-threaded polling across 4 USB cameras introduced cross-camera latency spikes
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [dv_processing, numpy, pyyaml]
# ---
"""
DVXplorer Micro Multi-Camera Event Streamer.

Opens 4× iniVation DVXplorer Micro cameras via USB using the
``dv_processing`` SDK. Each camera runs in its own thread, yielding
batched event tensors tagged with camera ID and hardware timestamps.

Architecture mirrors ``RadarReaderThread`` from ``radar_belt_engine``
— one dedicated I/O thread per device with queue-based handoff to
the consumer pipeline.

Usage:
    streamer = DvxEventStreamer.from_config("config/dvxplorer_array.yaml")
    streamer.start()
    for batch in streamer.consume():
        # batch.events: np.ndarray (N, 4) [x, y, t_us, polarity]
        # batch.camera_id: int
        # batch.azimuth_offset_deg: float
        pass
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Attempt to import dv_processing; fail gracefully for testing
try:
    import dv_processing as dv

    _DV_AVAILABLE = True
except ImportError:
    dv = None  # type: ignore[assignment]
    _DV_AVAILABLE = False
    logger.warning(
        "dv_processing not installed — running in stub mode. "
        "Install via: pip install dv-processing"
    )


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Configuration for a single DVXplorer Micro camera.

    Attributes:
        camera_id: Unique integer identifier for this camera.
        name: Human-readable name (e.g., 'cam_front').
        usb_path: OS device path (e.g., '/dev/dvxplorer0').
        azimuth_offset_deg: Mounting azimuth relative to operator forward.
        elevation_offset_deg: Mounting elevation offset.
        lens_focal_mm: Lens focal length in mm.
        fov_h_deg: Horizontal field of view in degrees.
        enabled: Whether this camera is active.
    """
    camera_id: int
    name: str
    usb_path: str
    azimuth_offset_deg: float
    elevation_offset_deg: float
    lens_focal_mm: float
    fov_h_deg: float
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class EventBatch:
    """A batch of events from a single camera.

    Attributes:
        events: (N, 4) int64 array with columns [x, y, timestamp_us, polarity].
        camera_id: Source camera identifier.
        camera_name: Human-readable camera name.
        azimuth_offset_deg: Camera mounting azimuth for bearing calculation.
        fov_h_deg: Camera horizontal FOV in degrees.
        resolution_w: Sensor width in pixels.
        resolution_h: Sensor height in pixels.
        hw_timestamp_start_us: First event timestamp in this batch.
        hw_timestamp_end_us: Last event timestamp in this batch.
    """
    events: np.ndarray
    camera_id: int
    camera_name: str
    azimuth_offset_deg: float
    fov_h_deg: float
    resolution_w: int
    resolution_h: int
    hw_timestamp_start_us: int
    hw_timestamp_end_us: int


class CameraReaderThread(threading.Thread):
    """Dedicated I/O thread for a single DVXplorer Micro.

    Opens the camera via ``dv.io.CameraCapture``, reads events in
    configurable-duration batches, and pushes ``EventBatch`` objects
    onto a shared output queue.

    Args:
        config: CameraConfig for this camera.
        output_queue: Shared queue for cross-thread event delivery.
        batch_duration_us: Duration of each event batch in microseconds.
        resolution: Tuple (width, height) of the sensor.
    """

    def __init__(
        self,
        config: CameraConfig,
        output_queue: queue.Queue[EventBatch],
        batch_duration_us: int = 10_000,
        resolution: tuple[int, int] = (640, 480),
    ) -> None:
        super().__init__(daemon=True, name=f"CamReader-{config.name}")
        self._config = config
        self._output_queue = output_queue
        self._batch_duration_us = batch_duration_us
        self._resolution = resolution
        self._stop_event = threading.Event()

        # Stats
        self.batches_produced: int = 0
        self.events_total: int = 0
        self.reconnect_count: int = 0

    def stop(self) -> None:
        """Signal this thread to stop."""
        self._stop_event.set()

    @property
    def is_stopped(self) -> bool:
        """Whether a stop has been requested."""
        return self._stop_event.is_set()

    def _open_camera(self) -> Optional[object]:
        """Attempt to open the DVXplorer Micro via dv_processing.

        Returns:
            A ``dv.io.CameraCapture`` instance, or None on failure.
        """
        if not _DV_AVAILABLE:
            logger.error(
                "Cannot open camera %s — dv_processing not installed",
                self._config.name,
            )
            return None

        try:
            # CameraCapture auto-discovers DVXplorer cameras.
            # If usb_path looks like a device name, use it as cameraName.
            capture = dv.io.CameraCapture(self._config.usb_path)
            logger.info(
                "Opened DVXplorer Micro '%s' at %s (resolution %s)",
                self._config.name,
                self._config.usb_path,
                capture.getEventResolution(),
            )
            return capture
        except Exception:
            logger.exception(
                "Failed to open camera '%s' at %s",
                self._config.name,
                self._config.usb_path,
            )
            return None

    def _events_to_numpy(self, event_store) -> np.ndarray:
        """Convert a dv EventStore to a (N, 4) int64 numpy array.

        Columns: [x, y, timestamp_us, polarity].
        """
        if event_store is None or len(event_store) == 0:
            return np.empty((0, 4), dtype=np.int64)

        # dv_processing EventStore provides numpy() or direct access
        # to x, y, timestamp, polarity arrays.
        timestamps = event_store.timestamps()
        x_coords = event_store.xs()
        y_coords = event_store.ys()
        polarities = event_store.polarities()

        return np.column_stack([
            x_coords.astype(np.int64),
            y_coords.astype(np.int64),
            timestamps.astype(np.int64),
            polarities.astype(np.int64),
        ])

    def run(self) -> None:
        """Main loop: open camera → batch events → push to queue."""
        capture = None

        while not self.is_stopped:
            # Open/reconnect loop
            if capture is None:
                capture = self._open_camera()
                if capture is None:
                    logger.warning(
                        "Camera '%s' unavailable — retrying in 2s",
                        self._config.name,
                    )
                    time.sleep(2.0)
                    self.reconnect_count += 1
                    continue

            try:
                # Read a batch of events
                event_store = capture.getNextEventBatch()

                if event_store is None or len(event_store) == 0:
                    time.sleep(0.001)  # 1ms yield to avoid busy-wait
                    continue

                events = self._events_to_numpy(event_store)

                if events.shape[0] == 0:
                    continue

                batch = EventBatch(
                    events=events,
                    camera_id=self._config.camera_id,
                    camera_name=self._config.name,
                    azimuth_offset_deg=self._config.azimuth_offset_deg,
                    fov_h_deg=self._config.fov_h_deg,
                    resolution_w=self._resolution[0],
                    resolution_h=self._resolution[1],
                    hw_timestamp_start_us=int(events[0, 2]),
                    hw_timestamp_end_us=int(events[-1, 2]),
                )

                # Non-blocking put; drop oldest if queue is full
                try:
                    self._output_queue.put_nowait(batch)
                except queue.Full:
                    # Drop oldest batch to keep pipeline live
                    try:
                        self._output_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._output_queue.put_nowait(batch)

                self.batches_produced += 1
                self.events_total += events.shape[0]

            except Exception:
                logger.exception(
                    "Error reading from camera '%s' — reconnecting",
                    self._config.name,
                )
                capture = None
                self.reconnect_count += 1
                time.sleep(1.0)

        logger.info(
            "CameraReaderThread '%s' stopped — %d batches, %d events, %d reconnects",
            self._config.name,
            self.batches_produced,
            self.events_total,
            self.reconnect_count,
        )


class DvxEventStreamer:
    """Multi-camera event streamer for the DVXplorer Micro array.

    Creates one ``CameraReaderThread`` per enabled camera and merges
    their outputs into a single generator via a shared queue.

    Args:
        cameras: List of CameraConfig for each sensor.
        batch_duration_us: Event batch duration per read cycle.
        resolution: Sensor resolution (W, H).
        queue_depth: Maximum queued batches before dropping.
    """

    def __init__(
        self,
        cameras: list[CameraConfig],
        batch_duration_us: int = 10_000,
        resolution: tuple[int, int] = (640, 480),
        queue_depth: int = 64,
    ) -> None:
        self._cameras = [c for c in cameras if c.enabled]
        self._batch_duration_us = batch_duration_us
        self._resolution = resolution

        self._event_queue: queue.Queue[EventBatch] = queue.Queue(
            maxsize=queue_depth
        )
        self._threads: list[CameraReaderThread] = []
        self._running = False

    @classmethod
    def from_config(cls, config_path: str | Path) -> DvxEventStreamer:
        """Create a streamer from a YAML configuration file.

        Args:
            config_path: Path to ``dvxplorer_array.yaml``.

        Returns:
            Configured ``DvxEventStreamer`` instance.
        """
        config_path = Path(config_path)
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        cameras = []
        for cam in cfg.get("cameras", []):
            cameras.append(CameraConfig(
                camera_id=cam["id"],
                name=cam["name"],
                usb_path=cam["usb_path"],
                azimuth_offset_deg=cam["azimuth_offset_deg"],
                elevation_offset_deg=cam.get("elevation_offset_deg", 0.0),
                lens_focal_mm=cam.get("lens_focal_mm", 6.0),
                fov_h_deg=cam.get("fov_h_deg", 55.0),
                enabled=cam.get("enabled", True),
            ))

        resolution = tuple(cfg.get("array", {}).get("resolution", [640, 480]))
        batch_us = cfg.get("voxelization", {}).get("batch_us", 10_000)

        return cls(
            cameras=cameras,
            batch_duration_us=batch_us,
            resolution=resolution,
        )

    def start(self) -> None:
        """Start all camera reader threads."""
        if self._running:
            logger.warning("DvxEventStreamer already running")
            return

        for cam_cfg in self._cameras:
            thread = CameraReaderThread(
                config=cam_cfg,
                output_queue=self._event_queue,
                batch_duration_us=self._batch_duration_us,
                resolution=self._resolution,
            )
            self._threads.append(thread)
            thread.start()
            logger.info("Started reader thread for camera '%s'", cam_cfg.name)

        self._running = True
        logger.info(
            "DvxEventStreamer started — %d cameras active", len(self._threads)
        )

    def stop(self) -> None:
        """Stop all camera reader threads and wait for them to exit."""
        for thread in self._threads:
            thread.stop()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()
        self._running = False
        logger.info("DvxEventStreamer stopped")

    def consume(
        self, timeout_s: float = 0.1
    ) -> Generator[EventBatch, None, None]:
        """Yield EventBatch objects from all cameras.

        This is the main consumer interface. Blocks up to ``timeout_s``
        seconds per batch. Yields indefinitely until ``stop()`` is called.

        Args:
            timeout_s: Maximum time to wait for a batch.

        Yields:
            EventBatch from any active camera.
        """
        while self._running:
            try:
                batch = self._event_queue.get(timeout=timeout_s)
                yield batch
            except queue.Empty:
                continue

    @property
    def camera_count(self) -> int:
        """Number of active camera threads."""
        return len(self._threads)

    def get_stats(self) -> dict:
        """Return per-camera streaming statistics."""
        stats = {}
        for thread in self._threads:
            stats[thread._config.name] = {
                "batches": thread.batches_produced,
                "events": thread.events_total,
                "reconnects": thread.reconnect_count,
                "alive": thread.is_alive(),
            }
        return stats
