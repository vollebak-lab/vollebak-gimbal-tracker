# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: 9-state Kalman filter with constant-acceleration model and widened velocity gates for aerial drone kinematics vs ground-based human tracking
#   FAILURE_MODE: radar_belt_engine target_tracker.py assumes constant-velocity ground targets with <2 m/s motion; drones maneuver at 15+ m/s with rapid acceleration
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy, scipy]
# ---
"""
Aerial Target Tracker — Multi-Target Kalman + Hungarian Assignment.

Forked from ``radar_belt_engine/src/target_tracker.py`` and re-tuned
for aerial drone kinematics:

Changes from ground tracker:
    - State vector expanded to 9 dimensions: [x, y, z, vx, vy, vz, ax, ay, az]
    - Constant-acceleration model (vs constant-velocity)
    - Velocity gate widened to 180° (drones can reverse direction)
    - Gate threshold raised for larger innovation (higher target speeds)
    - Measurement noise reduced (radar gives precise aerial returns)
    - Process noise increased (drones accelerate rapidly)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.layer3_fusion.imm_tracker import IMMConfig, IMMTrack, MotionModel
from src.layer3_fusion.jpda_associator import (
    JPDAAssociator,
    JPDAConfig,
    JPDAResult,
    NeuromorphicPrior,
    TrackUpdate,
)

logger = logging.getLogger(__name__)

_GATE_COST: float = 1e6


@dataclass(slots=True)
class AerialTrackerConfig:
    """Configuration tuned for aerial drone tracking.

    Compared to radar_belt_engine TrackerConfig:
    - Higher velocity/acceleration gates
    - Constant-acceleration process model
    - Faster update rate assumption (radar + neuromorphic fusion)
    """
    max_coast_frames: int = 15         # More coasting for aerial targets
    gate_threshold: float = 5.0        # Wider Mahalanobis gate
    velocity_gate_deg: float = 180.0   # No velocity gating for maneuvering targets
    process_noise_accel_std: float = 5.0    # m/s² — drones accelerate fast
    measurement_noise_pos_std: float = 0.5  # m — radar measurement noise
    measurement_noise_vel_std: float = 0.3  # m/s — Doppler noise
    dt: float = 0.1                    # 10Hz update rate
    use_imm: bool = True               # Use IMM instead of single CA model
    use_jpda: bool = True              # Use JPDA instead of Hungarian
    imm_config: Optional[IMMConfig] = None   # IMM-specific config (auto-built if None)
    jpda_config: Optional[JPDAConfig] = None  # JPDA-specific config (auto-built if None)


@dataclass(frozen=True, slots=True)
class AerialTarget:
    """Output for a single tracked aerial target.

    Attributes:
        target_id: Persistent track identifier.
        x, y, z: Position in meters (global frame).
        vx, vy, vz: Velocity in m/s.
        ax, ay, az: Acceleration in m/s².
        doppler_mps: Latest radial Doppler velocity.
        rcs_dbsm: Latest radar cross section.
        bearing_deg: Global bearing from operator.
        elevation_deg: Elevation angle from operator.
        range_m: Slant range from operator.
        track_age_frames: Number of frames since track initiation.
        num_detections: Number of radar detections in this cluster.
    """
    target_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float
    doppler_mps: float
    rcs_dbsm: float
    bearing_deg: float
    elevation_deg: float
    range_m: float
    track_age_frames: int
    num_detections: int


class AerialKalmanTrack:
    """9-state constant-acceleration Kalman filter for a single aerial target.

    State vector: [x, y, z, vx, vy, vz, ax, ay, az]
    Measurement: [x, y, z, doppler]

    Constant-acceleration model accounts for drone maneuvering.

    Args:
        initial_position: (3,) array [x, y, z].
        initial_doppler: Scalar radial Doppler velocity.
        initial_rcs: Radar cross section in dBsm.
        track_id: Unique persistent ID.
        config: AerialTrackerConfig.
    """

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_doppler: float,
        initial_rcs: float,
        track_id: int,
        config: AerialTrackerConfig,
    ) -> None:
        self.id = track_id
        self.doppler = initial_doppler
        self.rcs = initial_rcs
        self.age: int = 0
        self.missed_frames: int = 0
        self.num_detections: int = 1

        dt = config.dt

        # State vector [x, y, z, vx, vy, vz, ax, ay, az]
        self.state = np.zeros(9, dtype=np.float64)
        self.state[:3] = initial_position

        # State transition matrix (constant acceleration)
        self.F = np.eye(9, dtype=np.float64)
        for i in range(3):
            self.F[i, i + 3] = dt          # pos += vel * dt
            self.F[i, i + 6] = 0.5 * dt**2  # pos += 0.5 * acc * dt²
            self.F[i + 3, i + 6] = dt       # vel += acc * dt

        # Measurement matrix: observe [x, y, z] directly
        self.H = np.zeros((3, 9), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise — acceleration jerk model
        q = config.process_noise_accel_std ** 2
        self.Q = np.zeros((9, 9), dtype=np.float64)
        for i in range(3):
            # Position noise
            self.Q[i, i] = q * (dt**4) / 4.0
            self.Q[i, i + 3] = q * (dt**3) / 2.0
            self.Q[i, i + 6] = q * (dt**2) / 2.0
            # Velocity noise
            self.Q[i + 3, i] = q * (dt**3) / 2.0
            self.Q[i + 3, i + 3] = q * dt**2
            self.Q[i + 3, i + 6] = q * dt
            # Acceleration noise
            self.Q[i + 6, i] = q * (dt**2) / 2.0
            self.Q[i + 6, i + 3] = q * dt
            self.Q[i + 6, i + 6] = q

        # Measurement noise
        r_pos = config.measurement_noise_pos_std ** 2
        self.R = np.eye(3, dtype=np.float64) * r_pos

        # State covariance
        self.P = np.eye(9, dtype=np.float64)
        self.P[:3, :3] *= r_pos * 4       # Position uncertainty
        self.P[3:6, 3:6] *= q * 100       # Velocity uncertainty
        self.P[6:9, 6:9] *= q * 200       # Acceleration uncertainty

    @property
    def position(self) -> np.ndarray:
        """Current estimated position [x, y, z]."""
        return self.state[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Current estimated velocity [vx, vy, vz]."""
        return self.state[3:6].copy()

    @property
    def acceleration(self) -> np.ndarray:
        """Current estimated acceleration [ax, ay, az]."""
        return self.state[6:9].copy()

    def predict(self) -> np.ndarray:
        """Predict the next state."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:3].copy()

    def update(
        self,
        measurement: np.ndarray,
        doppler: float,
        rcs: float,
        num_detections: int,
    ) -> None:
        """Kalman update with a new measurement.

        Args:
            measurement: (3,) array [x, y, z].
            doppler: Radial Doppler velocity.
            rcs: Radar cross section in dBsm.
            num_detections: Number of detections in the cluster.
        """
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y

        # Joseph form for numerical stability
        I_KH = np.eye(9) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self.doppler = doppler
        self.rcs = rcs
        self.num_detections = num_detections
        self.missed_frames = 0
        self.age += 1

    def coast(self) -> None:
        """Mark as unmatched for current frame."""
        self.missed_frames += 1
        self.age += 1

    def mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """Mahalanobis distance from predicted position to measurement."""
        residual = measurement - self.state[:3]
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return _GATE_COST
        return float(np.sqrt(residual @ S_inv @ residual))

    def to_aerial_target(self) -> AerialTarget:
        """Convert current state to output AerialTarget."""
        pos = self.position
        vel = self.velocity
        acc = self.acceleration

        # Compute spherical coordinates from operator origin
        range_m = float(np.linalg.norm(pos))
        bearing_deg = float(np.degrees(np.arctan2(pos[0], pos[1]))) % 360.0
        elevation_deg = (
            float(np.degrees(np.arcsin(pos[2] / max(range_m, 1e-6))))
            if range_m > 1e-6 else 0.0
        )

        return AerialTarget(
            target_id=self.id,
            x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
            vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
            ax=float(acc[0]), ay=float(acc[1]), az=float(acc[2]),
            doppler_mps=self.doppler,
            rcs_dbsm=self.rcs,
            bearing_deg=bearing_deg,
            elevation_deg=elevation_deg,
            range_m=range_m,
            track_age_frames=self.age,
            num_detections=self.num_detections,
        )


class AerialMultiTargetTracker:
    """Multi-target tracker with IMM motion models and JPDA association.

    Tracking Pipeline Architecture:
        1. **IMM** (motion model): Runs 4 parallel Kalman models
           (CV, CJ, MSM, STS) per track with Bayesian mixing.
           Replaces single constant-acceleration filter.
        2. **JPDA** (data association): Computes probabilistic
           measurement-to-track associations. Replaces Hungarian.
           Supports neuromorphic bearing as a Bayesian prior.

    Falls back to original single-Kalman + Hungarian when
    ``use_imm=False`` and ``use_jpda=False`` in config.

    Args:
        config: AerialTrackerConfig.
    """

    def __init__(self, config: Optional[AerialTrackerConfig] = None) -> None:
        self._config = config or AerialTrackerConfig()
        self._tracks: list = []  # IMMTrack or AerialKalmanTrack
        self._next_id: int = 0

        # Build JPDA associator if enabled
        self._jpda: Optional[JPDAAssociator] = None
        if self._config.use_jpda:
            jpda_cfg = self._config.jpda_config or JPDAConfig(
                gate_threshold=self._config.gate_threshold,
            )
            self._jpda = JPDAAssociator(config=jpda_cfg)

        # Build IMM config if enabled but not provided
        if self._config.use_imm and self._config.imm_config is None:
            self._config.imm_config = IMMConfig(
                dt=self._config.dt,
                measurement_noise_pos_std=self._config.measurement_noise_pos_std,
                measurement_noise_vel_std=self._config.measurement_noise_vel_std,
            )

    @property
    def active_track_count(self) -> int:
        """Number of currently active tracks."""
        return len(self._tracks)

    def update(
        self,
        positions: np.ndarray,
        dopplers: np.ndarray,
        rcs_values: np.ndarray,
        detection_counts: np.ndarray,
        neuromorphic_priors: Optional[list[NeuromorphicPrior]] = None,
        current_timestamp_us: int = 0,
    ) -> list[AerialTarget]:
        """Process a radar frame of detections.

        Supports two modes:
        - **IMM + JPDA** (default): Probabilistic association with
          multi-model tracking and optional neuromorphic bearing priors.
        - **Legacy** (use_imm=False, use_jpda=False): Single CA Kalman
          with Hungarian assignment.

        Args:
            positions: (M, 3) array of detection positions [x, y, z].
            dopplers: (M,) array of Doppler velocities.
            rcs_values: (M,) array of RCS values.
            detection_counts: (M,) array of detection counts per cluster.
            neuromorphic_priors: Optional list of concurrent Layer 1
                detections for bearing correlation (JPDA mode only).
            current_timestamp_us: Current frame timestamp for L1
                correlation age check.

        Returns:
            List of AerialTarget for all active tracks.
        """
        num_measurements = positions.shape[0] if positions.ndim == 2 else 0

        # Predict all tracks
        for track in self._tracks:
            track.predict()

        num_tracks = len(self._tracks)

        if num_tracks == 0 and num_measurements == 0:
            return []

        if num_tracks == 0:
            return self._spawn_tracks(
                positions, dopplers, rcs_values, detection_counts,
            )

        if num_measurements == 0:
            self._coast_all()
            return self._get_targets()

        # Route to JPDA or legacy Hungarian
        if self._jpda is not None:
            return self._update_jpda(
                positions, dopplers, rcs_values, detection_counts,
                neuromorphic_priors, current_timestamp_us,
            )
        else:
            return self._update_hungarian(
                positions, dopplers, rcs_values, detection_counts,
            )

    def _update_jpda(
        self,
        positions: np.ndarray,
        dopplers: np.ndarray,
        rcs_values: np.ndarray,
        detection_counts: np.ndarray,
        neuromorphic_priors: Optional[list[NeuromorphicPrior]],
        current_timestamp_us: int,
    ) -> list[AerialTarget]:
        """JPDA-based update with neuromorphic bearing correlation.

        Replaces Hungarian assignment with probabilistic multi-hypothesis
        association. Measurements correlating with neuromorphic bearings
        receive boosted association probability.
        """
        result = self._jpda.associate(
            tracks=self._tracks,
            measurements=positions,
            measurement_dopplers=dopplers,
            measurement_rcs=rcs_values,
            measurement_counts=detection_counts,
            neuromorphic_priors=neuromorphic_priors,
            current_timestamp_us=current_timestamp_us,
        )

        # Apply JPDA updates to tracks
        for update in result.track_updates:
            track = self._tracks[update.track_idx]
            if update.has_update and update.weighted_measurement is not None:
                track.update(
                    measurement=update.weighted_measurement,
                    doppler=update.weighted_doppler,
                    rcs=update.weighted_rcs,
                    num_detections=update.weighted_count,
                )
            else:
                track.coast()

        # Spawn new tracks for unassociated measurements
        for j in result.unassociated_meas:
            self._spawn_single_track(
                positions[j], float(dopplers[j]),
                float(rcs_values[j]), int(detection_counts[j]),
            )

        # Prune dead tracks
        self._tracks = [
            t for t in self._tracks
            if t.missed_frames <= self._config.max_coast_frames
        ]

        return self._get_targets()

    def _update_hungarian(
        self,
        positions: np.ndarray,
        dopplers: np.ndarray,
        rcs_values: np.ndarray,
        detection_counts: np.ndarray,
    ) -> list[AerialTarget]:
        """Legacy Hungarian assignment update (backward compatibility)."""
        num_tracks = len(self._tracks)
        num_measurements = positions.shape[0]

        # Build cost matrix
        cost_matrix = np.full((num_tracks, num_measurements), _GATE_COST)

        for i, track in enumerate(self._tracks):
            for j in range(num_measurements):
                maha = track.mahalanobis_distance(positions[j])
                if maha <= self._config.gate_threshold:
                    cost_matrix[i, j] = maha

        # Hungarian assignment
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matched_tracks: set[int] = set()
        matched_meas: set[int] = set()

        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] < _GATE_COST:
                self._tracks[r].update(
                    measurement=positions[c],
                    doppler=float(dopplers[c]),
                    rcs=float(rcs_values[c]),
                    num_detections=int(detection_counts[c]),
                )
                matched_tracks.add(r)
                matched_meas.add(c)

        # Coast unmatched tracks
        for i in range(num_tracks):
            if i not in matched_tracks:
                self._tracks[i].coast()

        # Spawn new tracks for unmatched measurements
        for j in range(num_measurements):
            if j not in matched_meas:
                self._spawn_single_track(
                    positions[j], float(dopplers[j]),
                    float(rcs_values[j]), int(detection_counts[j]),
                )

        # Prune dead tracks
        self._tracks = [
            t for t in self._tracks
            if t.missed_frames <= self._config.max_coast_frames
        ]

        return self._get_targets()

    def _spawn_single_track(
        self,
        position: np.ndarray,
        doppler: float,
        rcs: float,
        detection_count: int,
    ) -> None:
        """Spawn a single new track (IMM or legacy Kalman)."""
        if self._config.use_imm:
            track = IMMTrack(
                initial_position=position,
                initial_doppler=doppler,
                initial_rcs=rcs,
                track_id=self._next_id,
                config=self._config.imm_config,
            )
        else:
            track = AerialKalmanTrack(
                initial_position=position,
                initial_doppler=doppler,
                initial_rcs=rcs,
                track_id=self._next_id,
                config=self._config,
            )
        track.num_detections = detection_count
        self._tracks.append(track)
        self._next_id += 1

    def _spawn_tracks(
        self,
        positions: np.ndarray,
        dopplers: np.ndarray,
        rcs_values: np.ndarray,
        detection_counts: np.ndarray,
    ) -> list[AerialTarget]:
        """Create tracks for all measurements when none exist."""
        targets = []
        for j in range(positions.shape[0]):
            self._spawn_single_track(
                positions[j], float(dopplers[j]),
                float(rcs_values[j]), int(detection_counts[j]),
            )
            targets.append(self._tracks[-1].to_aerial_target())
        return targets

    def _coast_all(self) -> None:
        """Coast all tracks."""
        for track in self._tracks:
            track.coast()
        self._tracks = [
            t for t in self._tracks
            if t.missed_frames <= self._config.max_coast_frames
        ]

    def _get_targets(self) -> list[AerialTarget]:
        """Build AerialTarget list from active tracks."""
        return [track.to_aerial_target() for track in self._tracks]
