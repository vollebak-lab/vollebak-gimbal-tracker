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
4-Camera Event Aggregator with Global Bearing + Elevation Tags.

Consumes ``EventBatch`` objects from ``DvxEventStreamer`` and enriches
each event with:
- **Global bearing** (degrees, 0=forward, CW positive) from pixel X.
- **Elevation** (degrees, 0=horizon, negative=below) from pixel Y.

Elevation is used downstream by the Waiter mode range estimator to
compute ground-plane distance to close-range ambush drones.

The aggregated output is used downstream by the propeller detector
for multi-camera fusion and by the state machine for radar cueing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.layer1_neuromorphic.dvx_event_streamer import EventBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AggregatedEventBatch:
    """An EventBatch enriched with global bearing and elevation.

    Attributes:
        events: (N, 4) int64 array [x, y, timestamp_us, polarity].
        camera_id: Source camera identifier.
        camera_name: Human-readable camera name.
        global_bearings_deg: (N,) float32 array — per-event global bearing
            in degrees (0=forward, CW positive, wraps at 360).
        centroid_bearing_deg: Scalar bearing of the event batch centroid.
        global_elevations_deg: (N,) float32 array — per-event elevation
            in degrees (0=horizon, negative=below, positive=above).
        centroid_elevation_deg: Scalar elevation of the event batch centroid.
        hw_timestamp_start_us: First event timestamp in this batch.
        hw_timestamp_end_us: Last event timestamp in this batch.
    """
    events: np.ndarray
    camera_id: int
    camera_name: str
    global_bearings_deg: np.ndarray
    centroid_bearing_deg: float
    global_elevations_deg: np.ndarray
    centroid_elevation_deg: float
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
        resolution_h: Sensor height in pixels (default 480 for DVXplorer Micro).
    """

    def __init__(
        self,
        resolution_w: int = 640,
        resolution_h: int = 480,
    ) -> None:
        self._resolution_w = resolution_w
        self._resolution_h = resolution_h
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

    def compute_pixel_elevations(
        self,
        y_pixels: np.ndarray,
        fov_v_deg: float,
        elevation_offset_deg: float,
    ) -> np.ndarray:
        """Compute elevation for each event from its pixel Y coordinate.

        Uses a simple pinhole model: linear mapping from pixel to angle.
        The center pixel maps to the camera boresight elevation.

        Convention: 0° = horizon, negative = below, positive = above.
        Pixel Y=0 is the top of the sensor (above horizon), Y=max is
        the bottom (below horizon).

        Used by the Waiter mode range estimator for ground-plane
        distance computation to close-range ambush drones.

        Args:
            y_pixels: (N,) array of pixel Y coordinates.
            fov_v_deg: Vertical field of view in degrees.
            elevation_offset_deg: Camera mounting elevation offset.

        Returns:
            (N,) float32 array of elevations in degrees.
        """
        center_y = self._resolution_h / 2.0
        deg_per_pixel = fov_v_deg / self._resolution_h

        # Y=0 is top → positive elevation; Y=max is bottom → negative
        # Invert sign: higher pixel Y means lower elevation
        angular_offset = (center_y - y_pixels.astype(np.float32)) * deg_per_pixel

        elevations = elevation_offset_deg + angular_offset

        return elevations.astype(np.float32)

    def aggregate(self, batch: EventBatch) -> AggregatedEventBatch:
        """Enrich an EventBatch with global bearing and elevation.

        Args:
            batch: Raw EventBatch from ``DvxEventStreamer``.

        Returns:
            AggregatedEventBatch with per-event bearings and elevations.
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
                global_elevations_deg=np.empty(0, dtype=np.float32),
                centroid_elevation_deg=getattr(
                    batch, "elevation_offset_deg", 0.0,
                ),
                hw_timestamp_start_us=batch.hw_timestamp_start_us,
                hw_timestamp_end_us=batch.hw_timestamp_end_us,
            )

        # Extract pixel X (column 0) and Y (column 1) coordinates
        x_pixels = events[:, 0]
        y_pixels = events[:, 1]

        # Compute global bearings from pixel X
        bearings = self.compute_pixel_bearings(
            x_pixels=x_pixels,
            fov_h_deg=batch.fov_h_deg,
            azimuth_offset_deg=batch.azimuth_offset_deg,
        )

        # Compute elevations from pixel Y
        fov_v_deg = getattr(batch, "fov_v_deg", 41.0)
        elevation_offset = getattr(batch, "elevation_offset_deg", 0.0)
        elevations = self.compute_pixel_elevations(
            y_pixels=y_pixels,
            fov_v_deg=fov_v_deg,
            elevation_offset_deg=elevation_offset,
        )

        # Centroid bearing (circular mean for wrap-around)
        centroid_bearing = self._circular_mean_deg(bearings)

        # Centroid elevation (arithmetic mean — no wrap-around)
        centroid_elevation = float(np.mean(elevations))

        self.batches_processed += 1
        self.events_total += n_events

        return AggregatedEventBatch(
            events=events,
            camera_id=batch.camera_id,
            camera_name=batch.camera_name,
            global_bearings_deg=bearings,
            centroid_bearing_deg=centroid_bearing,
            global_elevations_deg=elevations,
            centroid_elevation_deg=centroid_elevation,
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
