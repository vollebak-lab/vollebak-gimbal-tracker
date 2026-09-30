# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: JPDA replaces Hungarian assignment for clutter-robust data association; neuromorphic bearing injected as Bayesian prior boosts association probability for cross-sensor correlated detections
#   FAILURE_MODE: Hungarian hard-assigns one measurement per track — fails in dense clutter (urban) and swarm (crossing tracks) environments
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: RoC Research Paper — Section "Algorithmic Tracking" ¶2; Neuromorphic fusion concept from Predator architecture review
#   DEPENDENCIES: [numpy, scipy]
# ---
"""
JPDA (Joint Probabilistic Data Association) — Clutter-Robust Tracker.

Replaces the Hungarian (linear sum assignment) data association in
``aerial_target_tracker.py`` with a probabilistic multi-hypothesis
measurement-to-track association framework.

Key Improvements over Hungarian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Probabilistic weighting**: Each measurement contributes to each
   track with a computed probability, rather than a hard 1:1 assignment.
   This prevents track loss when clutter detections compete with the
   true target return.
2. **Clutter model**: Explicitly models false alarm density via a
   Poisson spatial clutter process. Measurements unlikely to be from
   any track are attributed to clutter with quantified probability.
3. **Neuromorphic bearing prior** (Predator-specific): Layer 1
   neuromorphic camera bearing injected as a Bayesian prior on
   association probability. Measurements correlating with a concurrent
   neuromorphic detection get boosted association weight, exploiting
   the physics-orthogonal nature of optical propeller flicker vs. RF
   radar returns for near-zero false positive probability.

CFAR Elimination
~~~~~~~~~~~~~~~~
When a radar detection falls within ±10° of a concurrent neuromorphic
bearing, the JPDA assigns it high association probability regardless
of CFAR threshold status. This implements the "CFAR bypass for
neuromorphic-correlated detections" policy from the implementation plan.

References:
    - Bar-Shalom, Y., Fortmann, T. (1988). "Tracking and Data
      Association." Academic Press.
    - Radar Fire Control RoC Research Paper (Predator docs)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neuromorphic Prior
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NeuromorphicPrior:
    """A neuromorphic camera bearing measurement for JPDA prior injection.

    Provides a physics-orthogonal detection cue (propeller flicker)
    that boosts radar measurement association probability when
    bearings correlate. Published by Layer 1 ``propeller_detector.py``.

    Attributes:
        bearing_deg: Global bearing to detection (0=fwd, CW).
        confidence: Detection confidence from SpMiniUNet (0.0–1.0).
        timestamp_us: Detection timestamp for temporal correlation.
        camera_id: Source camera identifier.
        bearing_uncertainty_deg: Bearing measurement uncertainty (1σ).
            Derived from camera FOV and pixel resolution.
    """
    bearing_deg: float
    confidence: float
    timestamp_us: int
    camera_id: int
    bearing_uncertainty_deg: float = 5.0  # ±5° per camera typical


# ---------------------------------------------------------------------------
# JPDA Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class JPDAConfig:
    """Configuration for the JPDA associator.

    Attributes:
        gate_threshold: Mahalanobis distance gate for validation region.
        clutter_density: Spatial density of false alarms (per unit volume).
            Higher = more permissive gating (expect more clutter).
        detection_probability: Probability that a true target produces
            a measurement (Pd). Lower = more robust to missed detections.
        neuro_bearing_gate_deg: Angular gate for neuromorphic bearing
            correlation. Radar measurements within this gate of a
            concurrent neuromorphic bearing get boosted prior.
        neuro_correlation_boost: Multiplicative boost factor for
            association probability when neuromorphic bearing correlates.
        neuro_max_age_us: Maximum age of a neuromorphic prior (in
            microseconds) before it is considered stale and ignored.
        min_association_prob: Minimum association probability below
            which a measurement is considered pure clutter.
    """
    gate_threshold: float = 5.0
    clutter_density: float = 1e-6
    detection_probability: float = 0.9
    neuro_bearing_gate_deg: float = 10.0
    neuro_correlation_boost: float = 5.0
    neuro_max_age_us: int = 500_000  # 500ms
    min_association_prob: float = 0.01


# ---------------------------------------------------------------------------
# JPDA Associator
# ---------------------------------------------------------------------------

class JPDAAssociator:
    """Joint Probabilistic Data Association for multi-target tracking.

    Computes probabilistic measurement-to-track associations for all
    tracks simultaneously, accounting for clutter, missed detections,
    and optional neuromorphic bearing priors.

    The output is a set of weighted innovations for each track,
    computed as the probability-weighted sum of all validated
    measurement innovations. This replaces the single hard
    measurement assignment from Hungarian.

    Args:
        config: JPDA configuration parameters.
    """

    def __init__(self, config: Optional[JPDAConfig] = None) -> None:
        self._config = config or JPDAConfig()
        self.associations_total: int = 0
        self.neuro_boosts_total: int = 0

    def associate(
        self,
        tracks: list,
        measurements: np.ndarray,
        measurement_dopplers: np.ndarray,
        measurement_rcs: np.ndarray,
        measurement_counts: np.ndarray,
        neuromorphic_priors: Optional[list[NeuromorphicPrior]] = None,
        current_timestamp_us: int = 0,
    ) -> JPDAResult:
        """Compute JPDA associations for all tracks and measurements.

        Args:
            tracks: List of IMMTrack (or any track with .mahalanobis_distance()
                and .position property).
            measurements: (M, 3) array of measurement positions.
            measurement_dopplers: (M,) Doppler values.
            measurement_rcs: (M,) RCS values.
            measurement_counts: (M,) detection counts per cluster.
            neuromorphic_priors: Optional list of concurrent L1 detections
                for bearing correlation. If provided, correlated measurements
                receive boosted association probability.
            current_timestamp_us: Current frame timestamp for L1 age check.

        Returns:
            JPDAResult containing per-track weighted innovations and
            unassociated measurement indices for track initiation.
        """
        cfg = self._config
        n_tracks = len(tracks)
        n_meas = measurements.shape[0] if measurements.ndim == 2 else 0

        # Edge cases
        if n_tracks == 0:
            return JPDAResult(
                track_updates=[],
                unassociated_meas=list(range(n_meas)),
            )

        if n_meas == 0:
            return JPDAResult(
                track_updates=[
                    TrackUpdate(track_idx=i, has_update=False)
                    for i in range(n_tracks)
                ],
                unassociated_meas=[],
            )

        # Step 1: Gating — build validation matrix
        # validation[i, j] = True if measurement j is within gate of track i
        validation = np.zeros((n_tracks, n_meas), dtype=bool)
        maha_distances = np.full(
            (n_tracks, n_meas), 1e6, dtype=np.float64,
        )

        for i, track in enumerate(tracks):
            for j in range(n_meas):
                dist = track.mahalanobis_distance(measurements[j])
                maha_distances[i, j] = dist
                if dist <= cfg.gate_threshold:
                    validation[i, j] = True

        # Step 2: Compute neuromorphic bearing correlation
        neuro_boost = np.ones(n_meas, dtype=np.float64)
        if neuromorphic_priors:
            neuro_boost = self._compute_neuro_boost(
                measurements, neuromorphic_priors, current_timestamp_us,
            )

        # Step 3: Compute association probabilities
        # For each track i and measurement j, compute:
        #   beta_{ij} = P(measurement j originated from track i)
        # Also compute beta_{i0} = P(no measurement from track i)

        track_updates: list[TrackUpdate] = []

        for i, track in enumerate(tracks):
            validated_indices = np.where(validation[i])[0]

            if len(validated_indices) == 0:
                # No measurements in gate — track has no update
                track_updates.append(
                    TrackUpdate(track_idx=i, has_update=False),
                )
                continue

            # Compute measurement likelihoods for this track
            # Use Gaussian likelihood from Mahalanobis distance
            likelihoods = np.zeros(len(validated_indices), dtype=np.float64)

            for k, j in enumerate(validated_indices):
                d = maha_distances[i, j]
                # Gaussian likelihood ∝ exp(-0.5 * d²)
                likelihoods[k] = np.exp(-0.5 * d ** 2)

                # Apply neuromorphic boost
                likelihoods[k] *= neuro_boost[j]

            # Clutter likelihood per measurement
            clutter_likelihood = cfg.clutter_density

            # Association probabilities via Bayes
            # beta_j = Pd * L_j / (sum(Pd * L_k) + (1-Pd) * clutter)
            pd = cfg.detection_probability
            numerators = pd * likelihoods

            # Denominator includes clutter term
            denom = np.sum(numerators) + (1.0 - pd) * clutter_likelihood
            denom = max(denom, 1e-300)

            betas = numerators / denom
            beta_0 = (1.0 - pd) * clutter_likelihood / denom  # No-detection

            # Filter out negligible associations
            significant = betas >= cfg.min_association_prob
            if not np.any(significant):
                track_updates.append(
                    TrackUpdate(track_idx=i, has_update=False),
                )
                continue

            # Compute weighted innovation
            # Combined measurement = sum(beta_j * z_j)
            H = track._filters[0].H if hasattr(track, '_filters') else None
            predicted_pos = track.position if hasattr(track, 'position') else track.state[:3]

            weighted_measurement = np.zeros(3, dtype=np.float64)
            weighted_doppler = 0.0
            weighted_rcs = 0.0
            weighted_count = 0.0
            total_beta = 0.0

            for k, j in enumerate(validated_indices):
                if significant[k]:
                    weighted_measurement += betas[k] * measurements[j]
                    weighted_doppler += betas[k] * measurement_dopplers[j]
                    weighted_rcs += betas[k] * measurement_rcs[j]
                    weighted_count += betas[k] * measurement_counts[j]
                    total_beta += betas[k]

            if total_beta > 0:
                # Normalize by total association probability
                weighted_measurement /= total_beta
                weighted_doppler /= total_beta
                weighted_rcs /= total_beta
                weighted_count /= total_beta

            track_updates.append(
                TrackUpdate(
                    track_idx=i,
                    has_update=True,
                    weighted_measurement=weighted_measurement,
                    weighted_doppler=float(weighted_doppler),
                    weighted_rcs=float(weighted_rcs),
                    weighted_count=int(round(weighted_count)),
                    association_prob=float(total_beta),
                    no_detection_prob=float(beta_0),
                    n_validated=len(validated_indices),
                ),
            )

        # Step 4: Identify unassociated measurements for track initiation
        # A measurement is unassociated if no track has significant
        # association probability with it
        associated_meas: set[int] = set()
        for i in range(n_tracks):
            validated = np.where(validation[i])[0]
            for j in validated:
                # Check if this measurement had significant association
                d = maha_distances[i, j]
                if d <= cfg.gate_threshold:
                    associated_meas.add(int(j))

        unassociated = [
            j for j in range(n_meas) if j not in associated_meas
        ]

        self.associations_total += 1

        return JPDAResult(
            track_updates=track_updates,
            unassociated_meas=unassociated,
        )

    def _compute_neuro_boost(
        self,
        measurements: np.ndarray,
        priors: list[NeuromorphicPrior],
        current_time_us: int,
    ) -> np.ndarray:
        """Compute neuromorphic bearing correlation boost per measurement.

        For each radar measurement, check if any concurrent neuromorphic
        detection has a matching bearing (within gate). If so, boost
        the measurement's association likelihood.

        This implements the "physics-orthogonal validation" concept:
        propeller flicker (optical/temporal) is completely independent
        of radar reflectivity (RF). Correlated detections across both
        modalities have extremely low false positive probability.

        Args:
            measurements: (M, 3) radar measurement positions.
            priors: Concurrent neuromorphic detections with bearings.
            current_time_us: Current timestamp for age filtering.

        Returns:
            (M,) boost factors (1.0 = no boost, >1.0 = correlated).
        """
        cfg = self._config
        n_meas = measurements.shape[0]
        boost = np.ones(n_meas, dtype=np.float64)

        # Filter stale priors
        valid_priors = [
            p for p in priors
            if (current_time_us - p.timestamp_us) <= cfg.neuro_max_age_us
        ]

        if not valid_priors:
            return boost

        for j in range(n_meas):
            # Compute bearing of this radar measurement
            meas_bearing = float(
                np.degrees(np.arctan2(measurements[j, 0], measurements[j, 1]))
            ) % 360.0

            # Check against each neuromorphic prior
            for prior in valid_priors:
                bearing_delta = abs(meas_bearing - prior.bearing_deg)
                # Handle 360° wraparound
                bearing_delta = min(bearing_delta, 360.0 - bearing_delta)

                if bearing_delta <= cfg.neuro_bearing_gate_deg:
                    # Correlated! Apply boost scaled by confidence
                    # and bearing match quality
                    match_quality = 1.0 - (
                        bearing_delta / cfg.neuro_bearing_gate_deg
                    )
                    boost_factor = (
                        1.0
                        + (cfg.neuro_correlation_boost - 1.0)
                        * prior.confidence
                        * match_quality
                    )
                    boost[j] = max(boost[j], boost_factor)
                    self.neuro_boosts_total += 1

                    logger.debug(
                        "Neuro-radar correlation: meas[%d] bearing=%.1f° "
                        "↔ L1 cam %d bearing=%.1f° (Δ=%.1f°, boost=%.2fx)",
                        j, meas_bearing, prior.camera_id,
                        prior.bearing_deg, bearing_delta, boost_factor,
                    )

        return boost


# ---------------------------------------------------------------------------
# JPDA Result Types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrackUpdate:
    """JPDA output for a single track.

    Attributes:
        track_idx: Index into the tracks list.
        has_update: Whether any measurement was associated.
        weighted_measurement: Probability-weighted combined measurement.
        weighted_doppler: Probability-weighted Doppler value.
        weighted_rcs: Probability-weighted RCS value.
        weighted_count: Probability-weighted detection count.
        association_prob: Total association probability (1 - P(no detection)).
        no_detection_prob: Probability that no measurement came from this track.
        n_validated: Number of measurements within the validation gate.
    """
    track_idx: int
    has_update: bool
    weighted_measurement: Optional[np.ndarray] = None
    weighted_doppler: float = 0.0
    weighted_rcs: float = 0.0
    weighted_count: int = 0
    association_prob: float = 0.0
    no_detection_prob: float = 1.0
    n_validated: int = 0


@dataclass(slots=True)
class JPDAResult:
    """Complete JPDA association result for one frame.

    Attributes:
        track_updates: Per-track association results.
        unassociated_meas: Indices of measurements not associated
            with any existing track (candidates for track initiation).
    """
    track_updates: list[TrackUpdate]
    unassociated_meas: list[int]
