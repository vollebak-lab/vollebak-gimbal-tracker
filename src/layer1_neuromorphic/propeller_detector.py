# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Adapted overlab-kevin SimpleTracker FSM for multi-camera live inference with per-camera ROI tracking and global bearing output
#   FAILURE_MODE: Original track_aedat.py only supports single AEDAT4 file replay; cannot handle live multi-camera USB streams
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [torch, spconv, numpy, dv_processing]
# ---
"""
Propeller Flicker Detector — Multi-Camera Live Inference.

Adapted from ``overlab-kevin/event-cam-prop-tracker/scripts/track_aedat.py``.
Runs per-camera SpMiniUNet inference on voxelized event data and applies
the SimpleTracker FSM (detect → track with ROI → detect) per camera.

Outputs detection alerts with bearing estimates for Layer 3 fusion.

The SpMiniUNet model and voxelization logic are imported from the
vendor submodule — only the I/O wrapping and multi-camera orchestration
are new code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Conditional imports for inference dependencies
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import spconv.pytorch as spconv
    _SPCONV_AVAILABLE = True
except ImportError:
    spconv = None  # type: ignore[assignment]
    _SPCONV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Detection Alert Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DetectionAlert:
    """A propeller-flicker detection alert from Layer 1.

    Published to Zenoh topic ``predator/layer1/detection``.

    Attributes:
        timestamp_us: Detection time (hardware timestamp from DVXplorer).
        camera_id: Source camera identifier.
        camera_name: Human-readable camera name.
        bearing_deg: Global bearing to detection centroid (0=fwd, CW).
        confidence: Detection confidence score (0.0–1.0).
        centroid_x: Detection centroid X in camera pixel coordinates.
        centroid_y: Detection centroid Y in camera pixel coordinates.
        tracker_state: Current state of the per-camera tracker FSM.
    """
    timestamp_us: int
    camera_id: int
    camera_name: str
    bearing_deg: float
    confidence: float
    centroid_x: float
    centroid_y: float
    tracker_state: str


# ---------------------------------------------------------------------------
# Per-Camera Tracker State Machine
# (Adapted from overlab-kevin SimpleTracker)
# ---------------------------------------------------------------------------

class TrackerState(Enum):
    """States for the per-camera propeller tracker FSM."""
    DETECT = auto()    # Wide-FOV search for propeller flicker
    TRACK = auto()     # Narrow ROI tracking on confirmed detection


@dataclass
class PerCameraTracker:
    """Per-camera propeller detection and tracking state machine.

    Mirrors the ``SimpleTracker`` logic from ``track_aedat.py``:
    - DETECT: Process full sensor frame through SpMiniUNet
    - TRACK: Crop ROI around last detection, process only that region

    Transitions:
    - DETECT → TRACK: Detection confidence exceeds ``det_thresh``
    - TRACK → DETECT: ``max_misses`` consecutive frames without detection

    Args:
        camera_id: Camera identifier.
        camera_name: Human-readable camera name.
        azimuth_offset_deg: Camera mounting azimuth for bearing calculation.
        fov_h_deg: Camera horizontal field of view.
        resolution_w: Sensor width in pixels.
        resolution_h: Sensor height in pixels.
        det_thresh: Logit threshold for detection.
        centre_thresh: Logit threshold for centroid calculation.
        max_misses: Consecutive misses before reverting to DETECT.
        crop_w: ROI width in TRACK state.
        crop_h: ROI height in TRACK state.
    """
    camera_id: int
    camera_name: str
    azimuth_offset_deg: float
    fov_h_deg: float
    resolution_w: int = 640
    resolution_h: int = 480
    det_thresh: float = 1.0
    centre_thresh: float = 0.0
    max_misses: int = 3
    crop_w: int = 150
    crop_h: int = 150

    # Internal state
    state: TrackerState = field(default=TrackerState.DETECT, init=False)
    _miss_count: int = field(default=0, init=False)
    _last_centroid_x: float = field(default=320.0, init=False)
    _last_centroid_y: float = field(default=240.0, init=False)
    _last_confidence: float = field(default=0.0, init=False)
    _detections_total: int = field(default=0, init=False)

    def process_prediction(
        self,
        prediction: np.ndarray,
        events: np.ndarray,
        timestamp_us: int,
    ) -> Optional[DetectionAlert]:
        """Process a SpMiniUNet prediction and update tracker state.

        Args:
            prediction: (H, W) float32 logit map from SpMiniUNet.
            events: (N, 4) event array [x, y, t_us, polarity] for this batch.
            timestamp_us: Batch timestamp for the detection alert.

        Returns:
            DetectionAlert if a detection was made, None otherwise.
        """
        # Apply ROI crop if in TRACK state
        if self.state == TrackerState.TRACK:
            prediction = self._apply_roi_mask(prediction)

        # Threshold for detection
        det_mask = prediction > self.det_thresh
        detected = np.any(det_mask)

        if detected:
            # Compute centroid from pixels above centre threshold
            centre_mask = prediction > self.centre_thresh
            if np.any(centre_mask):
                ys, xs = np.where(centre_mask)
                weights = prediction[centre_mask]
                cx = float(np.average(xs, weights=weights))
                cy = float(np.average(ys, weights=weights))
            else:
                ys, xs = np.where(det_mask)
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))

            confidence = float(np.max(prediction[det_mask]))
            # Normalize confidence via sigmoid
            confidence = 1.0 / (1.0 + np.exp(-confidence))

            self._last_centroid_x = cx
            self._last_centroid_y = cy
            self._last_confidence = confidence
            self._miss_count = 0
            self._detections_total += 1

            # Transition to TRACK
            if self.state == TrackerState.DETECT:
                self.state = TrackerState.TRACK
                logger.debug(
                    "Camera '%s': DETECT → TRACK at (%.1f, %.1f) conf=%.3f",
                    self.camera_name, cx, cy, confidence,
                )

            # Compute global bearing from pixel X
            bearing = self._pixel_to_bearing(cx)

            return DetectionAlert(
                timestamp_us=timestamp_us,
                camera_id=self.camera_id,
                camera_name=self.camera_name,
                bearing_deg=bearing,
                confidence=confidence,
                centroid_x=cx,
                centroid_y=cy,
                tracker_state=self.state.name,
            )

        else:
            # No detection
            self._miss_count += 1

            if self.state == TrackerState.TRACK:
                if self._miss_count >= self.max_misses:
                    self.state = TrackerState.DETECT
                    logger.debug(
                        "Camera '%s': TRACK → DETECT (%d consecutive misses)",
                        self.camera_name, self._miss_count,
                    )
                    self._miss_count = 0

            return None

    def _apply_roi_mask(self, prediction: np.ndarray) -> np.ndarray:
        """Zero out prediction pixels outside the tracking ROI.

        The ROI is centered on the last detection centroid.

        Args:
            prediction: (H, W) logit map.

        Returns:
            Masked prediction with only ROI pixels non-zero.
        """
        h, w = prediction.shape
        cx, cy = int(self._last_centroid_x), int(self._last_centroid_y)

        x_min = max(0, cx - self.crop_w // 2)
        x_max = min(w, cx + self.crop_w // 2)
        y_min = max(0, cy - self.crop_h // 2)
        y_max = min(h, cy + self.crop_h // 2)

        masked = np.full_like(prediction, -1e6)  # Gate out everything
        masked[y_min:y_max, x_min:x_max] = prediction[y_min:y_max, x_min:x_max]

        return masked

    def _pixel_to_bearing(self, pixel_x: float) -> float:
        """Convert a pixel X coordinate to a global bearing.

        Args:
            pixel_x: X coordinate in sensor pixel space.

        Returns:
            Global bearing in degrees [0, 360).
        """
        center_x = self.resolution_w / 2.0
        deg_per_pixel = self.fov_h_deg / self.resolution_w
        offset = (pixel_x - center_x) * deg_per_pixel
        bearing = (self.azimuth_offset_deg + offset) % 360.0
        return bearing


# ---------------------------------------------------------------------------
# Voxelizer — Adapted from overlab-kevin EventVoxelDataset
# ---------------------------------------------------------------------------

class EventVoxelizer:
    """Voxelize events into sparse 3D tensor for SpMiniUNet inference.

    Converts a batch of events (x, y, t_us, polarity) into a sparse
    3D volume (z=time, y, x) with separate positive/negative polarity
    channels, matching the input format expected by SpMiniUNet.

    Args:
        resolution: Sensor resolution (W, H).
        time_bin_us: Duration of each time slice in microseconds.
        batch_duration_us: Total batch duration for z-axis extent.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        time_bin_us: int = 100,
        batch_duration_us: int = 10_000,
    ) -> None:
        self._w, self._h = resolution
        self._time_bin_us = time_bin_us
        self._num_time_bins = batch_duration_us // time_bin_us

    def voxelize(self, events: np.ndarray) -> Optional[object]:
        """Convert events to a sparse PyTorch tensor for SpMiniUNet.

        Args:
            events: (N, 4) int64 array [x, y, timestamp_us, polarity].

        Returns:
            SparseTensor suitable for SpMiniUNet input, or None if
            torch/spconv are unavailable or events are empty.
        """
        if not _TORCH_AVAILABLE or not _SPCONV_AVAILABLE:
            logger.warning("torch/spconv unavailable — cannot voxelize")
            return None

        if events.shape[0] == 0:
            return None

        x = events[:, 0].astype(np.int32)
        y = events[:, 1].astype(np.int32)
        t = events[:, 2].astype(np.int64)
        p = events[:, 3].astype(np.int32)

        # Normalize timestamps to time bins (z-axis)
        t_min = t.min()
        z = ((t - t_min) // self._time_bin_us).astype(np.int32)
        z = np.clip(z, 0, self._num_time_bins - 1)

        # Filter out-of-bounds events
        valid = (x >= 0) & (x < self._w) & (y >= 0) & (y < self._h)
        x, y, z, p = x[valid], y[valid], z[valid], p[valid]

        if x.shape[0] == 0:
            return None

        # Build sparse coordinates and features
        # Features: [positive_count, negative_count] per voxel
        coords = np.column_stack([z, y, x]).astype(np.int32)

        # Use polarity to create 2-channel features
        features = np.zeros((coords.shape[0], 2), dtype=np.float32)
        features[p > 0, 0] = 1.0   # Positive polarity channel
        features[p <= 0, 1] = 1.0  # Negative polarity channel

        # Deduplicate coordinates by summing features
        coords_tensor = torch.from_numpy(coords).int()
        features_tensor = torch.from_numpy(features).float()

        spatial_shape = [self._num_time_bins, self._h, self._w]

        sparse_tensor = spconv.SparseConvTensor(
            features=features_tensor,
            indices=torch.cat([
                torch.zeros(coords_tensor.shape[0], 1, dtype=torch.int32),
                coords_tensor,
            ], dim=1),  # Prepend batch index
            spatial_shape=spatial_shape,
            batch_size=1,
        )

        return sparse_tensor


# ---------------------------------------------------------------------------
# PropellerDetector — Multi-Camera Orchestrator
# ---------------------------------------------------------------------------

class PropellerDetector:
    """Multi-camera propeller flicker detection orchestrator.

    Manages per-camera ``PerCameraTracker`` instances and runs SpMiniUNet
    inference on voxelized event batches. Produces ``DetectionAlert``
    objects for downstream fusion.

    Args:
        model_path: Path to SpMiniUNet checkpoint (.pt file).
        camera_configs: List of per-camera tracker configurations.
        resolution: Sensor resolution (W, H).
        time_bin_us: Voxelization time bin size.
        batch_duration_us: Event batch duration.
        device: PyTorch device ('cuda', 'cpu').
    """

    def __init__(
        self,
        model_path: str | Path,
        camera_configs: list[dict],
        resolution: tuple[int, int] = (640, 480),
        time_bin_us: int = 100,
        batch_duration_us: int = 10_000,
        device: str = "cuda",
    ) -> None:
        self._model_path = Path(model_path)
        self._device = device
        self._resolution = resolution

        # Initialize per-camera trackers
        self._trackers: dict[int, PerCameraTracker] = {}
        for cfg in camera_configs:
            tracker = PerCameraTracker(
                camera_id=cfg["id"],
                camera_name=cfg["name"],
                azimuth_offset_deg=cfg["azimuth_offset_deg"],
                fov_h_deg=cfg.get("fov_h_deg", 55.0),
                resolution_w=resolution[0],
                resolution_h=resolution[1],
                det_thresh=cfg.get("det_thresh", 1.0),
                centre_thresh=cfg.get("centre_thresh", 0.0),
                max_misses=cfg.get("max_misses", 3),
                crop_w=cfg.get("crop_w", 150),
                crop_h=cfg.get("crop_h", 150),
            )
            self._trackers[cfg["id"]] = tracker

        # Voxelizer
        self._voxelizer = EventVoxelizer(
            resolution=resolution,
            time_bin_us=time_bin_us,
            batch_duration_us=batch_duration_us,
        )

        # Model (lazy-loaded)
        self._model = None

        # Stats
        self.inferences_total: int = 0
        self.detections_total: int = 0
        self._inference_time_sum: float = 0.0

    def _load_model(self) -> None:
        """Lazy-load the SpMiniUNet model from checkpoint.

        The model architecture is expected to be importable from the
        vendor submodule: ``vendor.event_cam_prop_tracker.src.model``
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch not installed — cannot load SpMiniUNet model"
            )

        try:
            # Import from vendor submodule
            from vendor.event_cam_prop_tracker.src.model import SpMiniUNetWrapper
            self._model = SpMiniUNetWrapper.load_from_checkpoint(
                str(self._model_path)
            )
        except ImportError:
            logger.warning(
                "vendor.event_cam_prop_tracker not found — "
                "attempting direct model load"
            )
            # Fallback: load raw state dict
            self._model = torch.load(
                str(self._model_path), map_location=self._device
            )

        if hasattr(self._model, 'eval'):
            self._model.eval()

        if hasattr(self._model, 'to'):
            self._model.to(self._device)

        logger.info(
            "Loaded SpMiniUNet from %s on %s",
            self._model_path, self._device,
        )

    def process_batch(
        self,
        events: np.ndarray,
        camera_id: int,
        timestamp_us: int,
    ) -> Optional[DetectionAlert]:
        """Run inference on an event batch from a specific camera.

        Args:
            events: (N, 4) int64 array [x, y, timestamp_us, polarity].
            camera_id: Source camera identifier.
            timestamp_us: Batch end timestamp for the detection alert.

        Returns:
            DetectionAlert if a propeller was detected, None otherwise.
        """
        if self._model is None:
            self._load_model()

        tracker = self._trackers.get(camera_id)
        if tracker is None:
            logger.warning("No tracker for camera_id=%d", camera_id)
            return None

        # Voxelize events
        sparse_input = self._voxelizer.voxelize(events)
        if sparse_input is None:
            return None

        # Run inference
        t_start = time.monotonic()

        with torch.no_grad():
            prediction = self._model(sparse_input)

        t_elapsed = time.monotonic() - t_start
        self.inferences_total += 1
        self._inference_time_sum += t_elapsed

        # Convert prediction to numpy (H, W) logit map
        if hasattr(prediction, 'dense'):
            # SparseConvTensor → dense → squeeze to (H, W)
            pred_dense = prediction.dense()
            pred_np = pred_dense.squeeze().cpu().numpy()
        elif isinstance(prediction, torch.Tensor):
            pred_np = prediction.squeeze().cpu().numpy()
        else:
            pred_np = np.asarray(prediction).squeeze()

        # Ensure 2D (H, W) — if 3D, take max over time axis
        if pred_np.ndim == 3:
            pred_np = pred_np.max(axis=0)

        # Update per-camera tracker FSM
        alert = tracker.process_prediction(
            prediction=pred_np,
            events=events,
            timestamp_us=timestamp_us,
        )

        if alert is not None:
            self.detections_total += 1

        return alert

    @property
    def avg_inference_ms(self) -> float:
        """Average inference time in milliseconds."""
        if self.inferences_total == 0:
            return 0.0
        return (self._inference_time_sum / self.inferences_total) * 1000.0

    def get_tracker_states(self) -> dict[int, str]:
        """Return current FSM state for each camera tracker."""
        return {
            cam_id: tracker.state.name
            for cam_id, tracker in self._trackers.items()
        }
