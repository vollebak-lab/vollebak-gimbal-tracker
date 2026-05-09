# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Per-pixel global bearing computation from camera azimuth offset + pixel-to-angle projection for 360-degree tripwire
#   FAILURE_MODE: Without global bearing tags, the fusion layer cannot determine which direction a detection came from
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy]
# ---
"""
4-Camera Event Aggregator with Global Bearing Tags.

Consumes ``EventBatch`` objects from ``DvxEventStreamer`` and enriches
each event with a global bearing (degrees, 0=forward, CW positive)
computed from the camera's mounting azimuth offset and the event's
pixel position within the sensor FOV.

The aggregated output is used downstream by the propeller detector
for multi-camera fusion and by the FSM for radar cueing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.layer1_neuromorphic.dvx_event_streamer import EventBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AggregatedEventBatch:
    """An EventBatch enriched with global bearing information.

    Attributes:
        events: (N, 4) int64 array [x, y, timestamp_us, polarity].
        camera_id: Source camera identifier.
        camera_name: Human-readable camera name.
        global_bearings_deg: (N,) float32 array — per-event global bearing
            in degrees (0=forward, CW positive, wraps at 360).
        centroid_bearing_deg: Scalar bearing of the event batch centroid.
        hw_timestamp_start_us: First event timestamp in this batch.
        hw_timestamp_end_us: Last event timestamp in this batch.
    """
    events: np.ndarray
    camera_id: int
    camera_name: str
    global_bearings_deg: np.ndarray
    centroid_bearing_deg: float
    hw_timestamp_start_us: int
    hw_timestamp_end_us: int


class EventAggregator:
    """Merges multi-camera event streams with global bearing tags.

    For each event, computes:
        global_bearing = camera_azimuth_offset + pixel_to_angle(x)

    Where ``pixel_to_angle(x)`` maps pixel X coordinate to an angular
    offset from the camera boresight using the configured FOV.

    Args:
        resolution_w: Sensor width in pixels (default 640 for DVXplorer Micro).
    """

    def __init__(self, resolution_w: int = 640) -> None:
        self._resolution_w = resolution_w
        self.batches_processed: int = 0
        self.events_total: int = 0

    def compute_pixel_bearings(
        self,
        x_pixels: np.ndarray,
        fov_h_deg: float,
        azimuth_offset_deg: float,
    ) -> np.ndarray:
        """Compute global bearing for each event from its pixel X coordinate.

        Uses a simple pinhole model: linear mapping from pixel to angle.
        The center pixel maps to the camera boresight (azimuth_offset_deg).

        Args:
            x_pixels: (N,) array of pixel X coordinates.
            fov_h_deg: Horizontal field of view in degrees.
            azimuth_offset_deg: Camera mounting azimuth (0=forward, CW).

        Returns:
            (N,) float32 array of global bearings in [0, 360) degrees.
        """
        # Map pixel X to angular offset from boresight
        # x=0 → left edge of FOV, x=resolution_w-1 → right edge
        center_x = self._resolution_w / 2.0
        deg_per_pixel = fov_h_deg / self._resolution_w

        angular_offset = (x_pixels.astype(np.float32) - center_x) * deg_per_pixel

        # Global bearing = camera azimuth + angular offset
        bearings = azimuth_offset_deg + angular_offset

        # Wrap to [0, 360)
        bearings = bearings % 360.0

        return bearings.astype(np.float32)

    def aggregate(self, batch: EventBatch) -> AggregatedEventBatch:
        """Enrich an EventBatch with global bearing information.

        Args:
            batch: Raw EventBatch from ``DvxEventStreamer``.

        Returns:
            AggregatedEventBatch with per-event global bearings.
        """
        events = batch.events
        n_events = events.shape[0]

        if n_events == 0:
            return AggregatedEventBatch(
                events=events,
                camera_id=batch.camera_id,
                camera_name=batch.camera_name,
                global_bearings_deg=np.empty(0, dtype=np.float32),
                centroid_bearing_deg=batch.azimuth_offset_deg,
                hw_timestamp_start_us=batch.hw_timestamp_start_us,
                hw_timestamp_end_us=batch.hw_timestamp_end_us,
            )

        # Extract pixel X coordinates (column 0)
        x_pixels = events[:, 0]

        # Compute global bearings
        bearings = self.compute_pixel_bearings(
            x_pixels=x_pixels,
            fov_h_deg=batch.fov_h_deg,
            azimuth_offset_deg=batch.azimuth_offset_deg,
        )

        # Centroid bearing (mean of all event bearings)
        # Use circular mean to handle wrap-around
        centroid_bearing = self._circular_mean_deg(bearings)

        self.batches_processed += 1
        self.events_total += n_events

        return AggregatedEventBatch(
            events=events,
            camera_id=batch.camera_id,
            camera_name=batch.camera_name,
            global_bearings_deg=bearings,
            centroid_bearing_deg=centroid_bearing,
            hw_timestamp_start_us=batch.hw_timestamp_start_us,
            hw_timestamp_end_us=batch.hw_timestamp_end_us,
        )

    @staticmethod
    def _circular_mean_deg(angles_deg: np.ndarray) -> float:
        """Compute circular mean of angles in degrees.

        Correctly handles wrap-around at 0/360 boundary.

        Args:
            angles_deg: Array of angles in degrees.

        Returns:
            Mean angle in [0, 360) degrees.
        """
        if angles_deg.size == 0:
            return 0.0

        angles_rad = np.deg2rad(angles_deg)
        mean_sin = np.mean(np.sin(angles_rad))
        mean_cos = np.mean(np.cos(angles_rad))

        mean_rad = np.arctan2(mean_sin, mean_cos)
        mean_deg = float(np.rad2deg(mean_rad)) % 360.0

        return mean_deg
