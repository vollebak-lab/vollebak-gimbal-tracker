# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: IMM (Interacting Multiple Model) tracker replaces single constant-acceleration Kalman — runs 4 parallel motion models with Bayesian mixing for maneuvering UAS targets
#   FAILURE_MODE: Single CA model diverges on high-G FPV maneuvers and produces lag during hover-to-sprint transitions; IMM adapts process model automatically
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: RoC Research Paper — Section "Algorithmic Tracking" ¶3-4
#   DEPENDENCIES: [numpy]
# ---
"""
Interacting Multiple Model (IMM) Tracker — Multi-Model Kalman Bank.

Replaces the single constant-acceleration Kalman filter in
``aerial_target_tracker.py`` with a bank of 4 parallel motion models.
The IMM framework automatically adapts to the target's current
maneuver mode by mixing model probabilities via a Markov transition
matrix at each update.

Motion Models
~~~~~~~~~~~~~
1. **Constant Velocity (CV)**: Low process noise for cruise/transit.
2. **Constant Jerk (CJ)**: High jerk noise for FPV kamikaze pitch
   maneuvers. Unlike Coordinated Turn (aircraft), FPV drones pitch
   aggressively along the velocity vector — they do not bank.
3. **Move-Stop-Move (MSM)**: Near-zero velocity state for hover/loiter.
4. **Sprint-to-Stop (STS)**: High deceleration for rapid transit-to-
   hover transitions (DJI-class observation drones).

IMM Cycle
~~~~~~~~~
Each update frame executes:
    1. **Interaction (Mix)**: Compute mixed state/covariance for each
       model using Markov transition probabilities.
    2. **Predict**: Each model predicts independently from its mixed
       initial condition.
    3. **Update**: Each model updates with the measurement.
    4. **Combine**: Merge model outputs weighted by their updated
       probabilities to produce a single state estimate.

References:
    - Bar-Shalom, Y., Li, X.R., Kirubarajan, T. (2001).
      "Estimation with Applications to Tracking and Navigation"
    - Radar Fire Control RoC Research Paper (Predator docs)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Motion Model Types
# ---------------------------------------------------------------------------

class MotionModel(Enum):
    """Available IMM motion models for drone tracking."""
    CONSTANT_VELOCITY = auto()    # Cruise/transit
    CONSTANT_JERK = auto()        # FPV aggressive pitch maneuvers
    MOVE_STOP_MOVE = auto()       # Hover/loiter
    SPRINT_TO_STOP = auto()       # Rapid deceleration to hover


# ---------------------------------------------------------------------------
# IMM Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IMMConfig:
    """Configuration for the IMM tracker.

    Attributes:
        dt: Update interval in seconds (default 10Hz).
        measurement_noise_pos_std: Position measurement noise (m).
        measurement_noise_vel_std: Velocity measurement noise (m/s).
        cv_accel_std: CV model process noise — acceleration std (m/s²).
        cj_jerk_std: CJ model process noise — jerk std (m/s³).
        msm_accel_std: MSM model process noise — very low (m/s²).
        sts_accel_std: STS model process noise — high decel (m/s²).
        transition_matrix: (4, 4) Markov transition probability matrix.
            Rows = from-model, Cols = to-model. Each row sums to 1.0.
        initial_model_probs: (4,) initial model probability vector.
    """
    dt: float = 0.1

    measurement_noise_pos_std: float = 0.5
    measurement_noise_vel_std: float = 0.3

    # Per-model process noise tuning
    cv_accel_std: float = 0.5      # Low — steady cruise
    cj_jerk_std: float = 15.0      # High — aggressive pitch
    msm_accel_std: float = 0.1     # Very low — near-stationary
    sts_accel_std: float = 8.0     # High — rapid deceleration

    # Markov transition matrix (4×4)
    # [CV, CJ, MSM, STS]
    # Default: models tend to persist, with moderate transition probs
    transition_matrix: Optional[np.ndarray] = None

    # Initial model probabilities
    initial_model_probs: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Set default transition matrix if not provided."""
        if self.transition_matrix is None:
            # Rows: from [CV, CJ, MSM, STS]
            # Cols: to   [CV, CJ, MSM, STS]
            self.transition_matrix = np.array([
                [0.80, 0.10, 0.05, 0.05],  # CV persists, can enter maneuver
                [0.15, 0.70, 0.05, 0.10],  # CJ → often returns to CV or STS
                [0.10, 0.05, 0.75, 0.10],  # MSM persists, can exit to CV/STS
                [0.10, 0.10, 0.15, 0.65],  # STS → often transitions to MSM
            ], dtype=np.float64)

        if self.initial_model_probs is None:
            # Uniform prior — no assumption about initial maneuver state
            self.initial_model_probs = np.array(
                [0.35, 0.20, 0.25, 0.20], dtype=np.float64,
            )


# ---------------------------------------------------------------------------
# Single Motion Model Filter
# ---------------------------------------------------------------------------

class _SingleModelFilter:
    """A single Kalman filter with a specific motion model.

    State vector: [x, y, z, vx, vy, vz, ax, ay, az] (9 states)
    Measurement: [x, y, z] (position only)

    Each motion model differs in the state transition matrix (F)
    and process noise covariance (Q).

    Args:
        model_type: Which motion model this filter implements.
        dt: Update interval in seconds.
        process_noise_param: Model-specific noise parameter.
        measurement_noise_pos_std: Position measurement noise.
    """

    N_STATES: int = 9
    N_MEAS: int = 3

    def __init__(
        self,
        model_type: MotionModel,
        dt: float,
        process_noise_param: float,
        measurement_noise_pos_std: float,
    ) -> None:
        self.model_type = model_type
        self._dt = dt

        # State vector [x, y, z, vx, vy, vz, ax, ay, az]
        self.state = np.zeros(self.N_STATES, dtype=np.float64)

        # Build model-specific F and Q
        self.F = self._build_transition_matrix(model_type, dt)
        self.Q = self._build_process_noise(
            model_type, dt, process_noise_param,
        )

        # Measurement matrix: observe [x, y, z]
        self.H = np.zeros((self.N_MEAS, self.N_STATES), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Measurement noise
        r_pos = measurement_noise_pos_std ** 2
        self.R = np.eye(self.N_MEAS, dtype=np.float64) * r_pos

        # State covariance — large initial uncertainty
        self.P = np.eye(self.N_STATES, dtype=np.float64)
        self.P[:3, :3] *= r_pos * 4.0
        self.P[3:6, 3:6] *= process_noise_param ** 2 * 100.0
        self.P[6:9, 6:9] *= process_noise_param ** 2 * 200.0

        # Likelihood of last measurement (for model probability update)
        self.likelihood: float = 1.0

    def predict(self) -> None:
        """Predict state forward by dt."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: np.ndarray) -> None:
        """Kalman update with position measurement.

        Also computes measurement likelihood for IMM model
        probability update.

        Args:
            measurement: (3,) array [x, y, z].
        """
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R

        # Compute measurement likelihood for model probability
        try:
            S_inv = np.linalg.inv(S)
            sign, log_det = np.linalg.slogdet(S)
            if sign > 0:
                mahal_sq = float(y @ S_inv @ y)
                self.likelihood = (
                    np.exp(-0.5 * mahal_sq)
                    / np.sqrt((2 * np.pi) ** self.N_MEAS * np.exp(log_det))
                )
            else:
                self.likelihood = 1e-300
        except np.linalg.LinAlgError:
            self.likelihood = 1e-300
            return

        # Kalman gain
        K = self.P @ self.H.T @ S_inv

        # State update
        self.state = self.state + K @ y

        # Joseph form for numerical stability
        I_KH = np.eye(self.N_STATES) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """Mahalanobis distance from predicted position to measurement.

        Args:
            measurement: (3,) position array.

        Returns:
            Mahalanobis distance (scalar).
        """
        residual = measurement - self.state[:3]
        S = self.H @ self.P @ self.H.T + self.R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return 1e6
        return float(np.sqrt(residual @ S_inv @ residual))

    @staticmethod
    def _build_transition_matrix(
        model: MotionModel, dt: float,
    ) -> np.ndarray:
        """Build the state transition matrix for a given motion model.

        All models share the 9-state vector [pos, vel, acc].
        The difference is how acceleration evolves.

        Args:
            model: Motion model type.
            dt: Time step in seconds.

        Returns:
            (9, 9) state transition matrix.
        """
        F = np.eye(9, dtype=np.float64)

        if model == MotionModel.CONSTANT_VELOCITY:
            # Acceleration decays to zero (first-order Gauss-Markov)
            # pos += vel * dt, vel unchanged, acc decays
            for i in range(3):
                F[i, i + 3] = dt
                F[i, i + 6] = 0.0       # No acceleration contribution
                F[i + 3, i + 6] = 0.0
                F[i + 6, i + 6] = 0.1   # Rapid decay of acceleration

        elif model == MotionModel.CONSTANT_JERK:
            # Full constant-acceleration propagation with jerk-driven Q
            for i in range(3):
                F[i, i + 3] = dt
                F[i, i + 6] = 0.5 * dt ** 2
                F[i + 3, i + 6] = dt

        elif model == MotionModel.MOVE_STOP_MOVE:
            # Velocity and acceleration decay rapidly (high damping)
            for i in range(3):
                F[i, i + 3] = dt * 0.5     # Damped velocity contribution
                F[i + 3, i + 3] = 0.3      # Velocity decays each frame
                F[i + 6, i + 6] = 0.1      # Acceleration decays fast

        elif model == MotionModel.SPRINT_TO_STOP:
            # Standard CA model with strong acceleration coupling
            for i in range(3):
                F[i, i + 3] = dt
                F[i, i + 6] = 0.5 * dt ** 2
                F[i + 3, i + 6] = dt
                F[i + 6, i + 6] = 0.7  # Moderate persistence of accel

        return F

    @staticmethod
    def _build_process_noise(
        model: MotionModel, dt: float, sigma: float,
    ) -> np.ndarray:
        """Build process noise covariance for a given motion model.

        Args:
            model: Motion model type.
            dt: Time step.
            sigma: Model-specific noise parameter (accel std or jerk std).

        Returns:
            (9, 9) process noise covariance matrix.
        """
        Q = np.zeros((9, 9), dtype=np.float64)
        q = sigma ** 2

        if model == MotionModel.CONSTANT_VELOCITY:
            # Piecewise white noise acceleration model with low noise
            for i in range(3):
                Q[i, i] = q * dt ** 4 / 4.0
                Q[i, i + 3] = q * dt ** 3 / 2.0
                Q[i + 3, i] = q * dt ** 3 / 2.0
                Q[i + 3, i + 3] = q * dt ** 2
                # Minimal acceleration noise
                Q[i + 6, i + 6] = q * 0.01

        elif model == MotionModel.CONSTANT_JERK:
            # Jerk-driven model: sigma is jerk std (m/s³)
            dt2 = dt ** 2
            dt3 = dt ** 3
            dt4 = dt ** 4
            dt5 = dt ** 5
            for i in range(3):
                Q[i, i] = q * dt5 / 20.0
                Q[i, i + 3] = q * dt4 / 8.0
                Q[i, i + 6] = q * dt3 / 6.0
                Q[i + 3, i] = q * dt4 / 8.0
                Q[i + 3, i + 3] = q * dt3 / 3.0
                Q[i + 3, i + 6] = q * dt2 / 2.0
                Q[i + 6, i] = q * dt3 / 6.0
                Q[i + 6, i + 3] = q * dt2 / 2.0
                Q[i + 6, i + 6] = q * dt

        elif model == MotionModel.MOVE_STOP_MOVE:
            # Very low noise — target expected near-stationary
            for i in range(3):
                Q[i, i] = q * dt ** 2
                Q[i + 3, i + 3] = q * dt
                Q[i + 6, i + 6] = q * 0.5

        elif model == MotionModel.SPRINT_TO_STOP:
            # High acceleration noise for rapid deceleration
            for i in range(3):
                Q[i, i] = q * dt ** 4 / 4.0
                Q[i, i + 3] = q * dt ** 3 / 2.0
                Q[i, i + 6] = q * dt ** 2 / 2.0
                Q[i + 3, i] = q * dt ** 3 / 2.0
                Q[i + 3, i + 3] = q * dt ** 2
                Q[i + 3, i + 6] = q * dt
                Q[i + 6, i] = q * dt ** 2 / 2.0
                Q[i + 6, i + 3] = q * dt
                Q[i + 6, i + 6] = q

        return Q


# ---------------------------------------------------------------------------
# IMM Track
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IMMOutput:
    """Combined IMM output for downstream consumers.

    Attributes:
        position: (3,) estimated position [x, y, z].
        velocity: (3,) estimated velocity [vx, vy, vz].
        acceleration: (3,) estimated acceleration [ax, ay, az].
        covariance: (9, 9) combined state covariance.
        model_probs: Dict mapping MotionModel to its current probability.
        dominant_model: The model with highest probability.
    """
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    covariance: np.ndarray
    model_probs: dict[MotionModel, float]
    dominant_model: MotionModel


class IMMTrack:
    """Interacting Multiple Model track for a single aerial target.

    Runs 4 parallel Kalman filters with different motion models
    and mixes their estimates based on Bayesian model probability
    updated each measurement frame.

    Args:
        initial_position: (3,) array [x, y, z].
        initial_doppler: Scalar radial Doppler velocity.
        initial_rcs: Radar cross section in dBsm.
        track_id: Unique persistent ID.
        config: IMM configuration.
    """

    N_MODELS: int = 4

    def __init__(
        self,
        initial_position: np.ndarray,
        initial_doppler: float,
        initial_rcs: float,
        track_id: int,
        config: IMMConfig,
    ) -> None:
        self.id = track_id
        self.doppler = initial_doppler
        self.rcs = initial_rcs
        self.age: int = 0
        self.missed_frames: int = 0
        self.num_detections: int = 1
        self._config = config

        # Build the 4 model filters
        model_configs = [
            (MotionModel.CONSTANT_VELOCITY, config.cv_accel_std),
            (MotionModel.CONSTANT_JERK, config.cj_jerk_std),
            (MotionModel.MOVE_STOP_MOVE, config.msm_accel_std),
            (MotionModel.SPRINT_TO_STOP, config.sts_accel_std),
        ]

        self._filters: list[_SingleModelFilter] = []
        for model_type, noise_param in model_configs:
            f = _SingleModelFilter(
                model_type=model_type,
                dt=config.dt,
                process_noise_param=noise_param,
                measurement_noise_pos_std=config.measurement_noise_pos_std,
            )
            f.state[:3] = initial_position
            self._filters.append(f)

        # Model probabilities — mutable
        self._model_probs = config.initial_model_probs.copy()

        # Transition probability matrix
        self._transition_matrix = config.transition_matrix.copy()

    @property
    def position(self) -> np.ndarray:
        """Combined IMM position estimate."""
        return self._combined_state()[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Combined IMM velocity estimate."""
        return self._combined_state()[3:6].copy()

    @property
    def acceleration(self) -> np.ndarray:
        """Combined IMM acceleration estimate."""
        return self._combined_state()[6:9].copy()

    @property
    def model_probabilities(self) -> dict[MotionModel, float]:
        """Current model probability distribution."""
        return {
            f.model_type: float(self._model_probs[i])
            for i, f in enumerate(self._filters)
        }

    @property
    def dominant_model(self) -> MotionModel:
        """Model with highest current probability."""
        idx = int(np.argmax(self._model_probs))
        return self._filters[idx].model_type

    def predict(self) -> np.ndarray:
        """IMM predict cycle: interaction (mix) then per-model predict.

        Returns:
            (3,) combined predicted position.
        """
        # Step 1: Interaction (mixing)
        self._interaction_step()

        # Step 2: Per-model prediction
        for f in self._filters:
            f.predict()

        return self._combined_state()[:3].copy()

    def update(
        self,
        measurement: np.ndarray,
        doppler: float,
        rcs: float,
        num_detections: int,
    ) -> None:
        """IMM update cycle: per-model update then combine probabilities.

        Args:
            measurement: (3,) array [x, y, z].
            doppler: Radial Doppler velocity.
            rcs: Radar cross section in dBsm.
            num_detections: Detections in the cluster.
        """
        # Step 3: Per-model measurement update
        for f in self._filters:
            f.update(measurement)

        # Step 4: Update model probabilities
        self._update_model_probabilities()

        # Update ancillary state
        self.doppler = doppler
        self.rcs = rcs
        self.num_detections = num_detections
        self.missed_frames = 0
        self.age += 1

    def coast(self) -> None:
        """Mark track as unmatched for current frame."""
        self.missed_frames += 1
        self.age += 1

    def mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """Minimum Mahalanobis distance across all models.

        Uses the combined IMM covariance for gating.

        Args:
            measurement: (3,) position measurement.

        Returns:
            Mahalanobis distance.
        """
        combined_state = self._combined_state()
        combined_P = self._combined_covariance(combined_state)

        H = self._filters[0].H
        R = self._filters[0].R

        residual = measurement - combined_state[:3]
        S = H @ combined_P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return 1e6
        return float(np.sqrt(residual @ S_inv @ residual))

    def get_imm_output(self) -> IMMOutput:
        """Get the full IMM combined output.

        Returns:
            IMMOutput with combined state and model probabilities.
        """
        combined = self._combined_state()
        return IMMOutput(
            position=combined[:3].copy(),
            velocity=combined[3:6].copy(),
            acceleration=combined[6:9].copy(),
            covariance=self._combined_covariance(combined),
            model_probs=self.model_probabilities,
            dominant_model=self.dominant_model,
        )

    def to_aerial_target(self):
        """Convert to AerialTarget output format.

        Import is deferred to avoid circular dependency.

        Returns:
            AerialTarget dataclass.
        """
        from src.layer3_fusion.aerial_target_tracker import AerialTarget

        combined = self._combined_state()
        pos = combined[:3]
        vel = combined[3:6]
        acc = combined[6:9]

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

    # -------------------------------------------------------------------
    # Internal IMM machinery
    # -------------------------------------------------------------------

    def _interaction_step(self) -> None:
        """IMM interaction (mixing) step.

        Computes mixed initial conditions for each model filter
        using the Markov transition probabilities and current
        model weights.
        """
        n = self.N_MODELS
        mu = self._model_probs  # (4,)
        pi = self._transition_matrix  # (4, 4)

        # Predicted model probabilities (denominator for mixing weights)
        c_bar = pi.T @ mu  # (4,)
        c_bar = np.maximum(c_bar, 1e-300)  # Prevent division by zero

        # Mixing weights: mu_{i|j} = pi_{ij} * mu_i / c_bar_j
        mixing_weights = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                mixing_weights[i, j] = pi[i, j] * mu[i] / c_bar[j]

        # Compute mixed state and covariance for each filter
        for j in range(n):
            # Mixed state
            mixed_state = np.zeros(9, dtype=np.float64)
            for i in range(n):
                mixed_state += mixing_weights[i, j] * self._filters[i].state

            # Mixed covariance (includes spread of means)
            mixed_P = np.zeros((9, 9), dtype=np.float64)
            for i in range(n):
                diff = self._filters[i].state - mixed_state
                mixed_P += mixing_weights[i, j] * (
                    self._filters[i].P + np.outer(diff, diff)
                )

            # Set the mixed initial conditions
            self._filters[j].state = mixed_state
            self._filters[j].P = mixed_P

        # Update predicted model probabilities for next step
        self._model_probs = c_bar

    def _update_model_probabilities(self) -> None:
        """Update model probabilities using measurement likelihoods.

        Bayes' rule: mu_j = c_bar_j * L_j / sum(c_bar_k * L_k)
        where L_j is the measurement likelihood from filter j.
        """
        likelihoods = np.array(
            [f.likelihood for f in self._filters], dtype=np.float64,
        )

        # Prevent numerical underflow
        likelihoods = np.maximum(likelihoods, 1e-300)

        # Unnormalized posterior
        posterior = self._model_probs * likelihoods

        # Normalize
        total = np.sum(posterior)
        if total > 0:
            self._model_probs = posterior / total
        else:
            # Fallback to uniform if all likelihoods collapsed
            self._model_probs = np.ones(self.N_MODELS) / self.N_MODELS
            logger.warning(
                "Track %d: all model likelihoods collapsed — "
                "resetting to uniform",
                self.id,
            )

    def _combined_state(self) -> np.ndarray:
        """Probability-weighted combined state estimate.

        Returns:
            (9,) combined state vector.
        """
        combined = np.zeros(9, dtype=np.float64)
        for i, f in enumerate(self._filters):
            combined += self._model_probs[i] * f.state
        return combined

    def _combined_covariance(
        self, combined_state: np.ndarray,
    ) -> np.ndarray:
        """Probability-weighted combined covariance.

        Includes both within-model uncertainty and spread-of-means
        across models.

        Args:
            combined_state: (9,) combined state from _combined_state().

        Returns:
            (9, 9) combined covariance matrix.
        """
        combined_P = np.zeros((9, 9), dtype=np.float64)
        for i, f in enumerate(self._filters):
            diff = f.state - combined_state
            combined_P += self._model_probs[i] * (
                f.P + np.outer(diff, diff)
            )
        return combined_P
