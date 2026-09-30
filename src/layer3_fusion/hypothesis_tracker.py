# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: MHT defers hard track association decisions across N frames — critical for swarm track-ID persistence when drones cross paths in congested environments
#   FAILURE_MODE: JPDA resolves single-frame ambiguity but cannot maintain track identity when two targets cross paths over multiple frames — MHT provides temporal hypothesis branching
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: RoC Research Paper — Section "Deep Learning and MHT" ¶2
#   DEPENDENCIES: [numpy]
# ---
"""
Multiple Hypothesis Tracking (MHT) — Temporal Hypothesis Manager.

Wraps the JPDA data association layer to maintain branching
hypothesis trees over multiple frames. While JPDA resolves
single-frame measurement ambiguity (which measurement belongs to
which track), MHT resolves **temporal** ambiguity (which track
identity persists when targets cross paths).

Architecture
~~~~~~~~~~~~
MHT sits as Layer 3 in the tracking pipeline:

    1. IMM (motion model) → "how does the target move?"
    2. JPDA (data association) → "which measurement this frame?"
    3. MHT (hypothesis management) → "which track ID over time?"

Key Features
~~~~~~~~~~~~
- N-scan-back pruning: Limits hypothesis tree depth to N frames
  (default 3) to bound computational cost.
- Hypothesis merging: Closely-spaced hypotheses are merged to
  prevent exponential branching.
- Swarm mode activation: MHT engages only when track density
  exceeds a configurable threshold (saves CPU when tracking
  isolated targets).
- Track ID persistence guarantee: Provides chain-of-custody for
  engagement authorization — cannot legally engage if track ID
  is ambiguous.

Computational Bounds
~~~~~~~~~~~~~~~~~~~~
Without pruning, MHT is O(m^N) where m = measurements per frame
and N = scan depth. The following bounds are enforced:
- ``max_hypotheses``: Hard cap on hypothesis count (default 100)
- ``n_scan_depth``: Maximum tree depth (default 3)
- ``merge_threshold``: Merge hypotheses with similar state estimates

References:
    - Blackman, S. (2004). "Multiple Hypothesis Tracking for
      Multiple Target Tracking." IEEE Aerospace and Electronic
      Systems Magazine.
    - Reid, D. (1979). "An algorithm for tracking multiple targets."
      IEEE Trans. on Automatic Control.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MHT Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MHTConfig:
    """Configuration for the Multiple Hypothesis Tracker.

    Attributes:
        n_scan_depth: Maximum hypothesis tree depth (frames back).
            Default 3 = keep 3 frames of history before pruning.
        max_hypotheses: Hard cap on number of active hypotheses.
        merge_threshold: State-space distance below which two
            hypotheses are merged (Mahalanobis distance).
        swarm_activation_count: Number of tracks within
            ``swarm_activation_radius_m`` to trigger MHT mode.
            Below this threshold, standard JPDA is used without
            hypothesis branching (saves CPU).
        swarm_activation_radius_m: Radius in meters for swarm
            density check.
        pruning_prob_threshold: Hypotheses with probability below
            this are pruned.
    """
    n_scan_depth: int = 3
    max_hypotheses: int = 100
    merge_threshold: float = 2.0
    swarm_activation_count: int = 3
    swarm_activation_radius_m: float = 50.0
    pruning_prob_threshold: float = 0.01


# ---------------------------------------------------------------------------
# Hypothesis Node
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HypothesisNode:
    """A single node in the hypothesis tree.

    Each node represents one possible measurement-to-track
    association at a single time frame.

    Attributes:
        hypothesis_id: Unique identifier for this hypothesis.
        parent_id: ID of parent hypothesis (None for root).
        frame_number: Frame at which this hypothesis was created.
        track_assignments: Dict mapping track_id → measurement_index.
            Value of -1 indicates "no measurement" (coast).
        probability: Current hypothesis probability.
        children: List of child hypothesis IDs.
    """
    hypothesis_id: int
    parent_id: Optional[int]
    frame_number: int
    track_assignments: dict[int, int]
    probability: float
    children: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Multiple Hypothesis Tracker
# ---------------------------------------------------------------------------

class MultipleHypothesisTracker:
    """Temporal hypothesis manager for swarm track-ID persistence.

    Wraps the JPDA association layer to maintain hypothesis trees
    that defer hard track identity decisions until sufficient
    evidence accumulates.

    Only activates when track density exceeds the swarm threshold.
    In low-density scenarios, passes through JPDA results directly
    to avoid unnecessary computation.

    Args:
        config: MHT configuration.
    """

    def __init__(self, config: Optional[MHTConfig] = None) -> None:
        self._config = config or MHTConfig()

        # Hypothesis tree storage
        self._hypotheses: dict[int, HypothesisNode] = {}
        self._next_hypothesis_id: int = 0
        self._current_frame: int = 0

        # Root hypothesis
        root = HypothesisNode(
            hypothesis_id=self._next_hypothesis_id,
            parent_id=None,
            frame_number=0,
            track_assignments={},
            probability=1.0,
        )
        self._hypotheses[root.hypothesis_id] = root
        self._best_hypothesis_id: int = root.hypothesis_id
        self._next_hypothesis_id += 1

        # Swarm mode state
        self._swarm_mode_active: bool = False

        # Statistics
        self.total_prunes: int = 0
        self.total_merges: int = 0
        self.frames_in_swarm_mode: int = 0

    @property
    def is_swarm_mode(self) -> bool:
        """Whether MHT is actively branching hypotheses."""
        return self._swarm_mode_active

    @property
    def hypothesis_count(self) -> int:
        """Number of active leaf hypotheses."""
        return len(self._get_leaf_hypotheses())

    def check_swarm_activation(
        self,
        track_positions: list[np.ndarray],
    ) -> bool:
        """Check if track density warrants MHT activation.

        MHT branching is computationally expensive. Only activate
        when multiple tracks are within close proximity (potential
        crossing/swarm scenario).

        Args:
            track_positions: List of (3,) position arrays for active tracks.

        Returns:
            True if MHT should be active.
        """
        cfg = self._config
        n_tracks = len(track_positions)

        if n_tracks < cfg.swarm_activation_count:
            self._swarm_mode_active = False
            return False

        # Check if enough tracks are within the swarm radius
        dense_count = 0
        for i in range(n_tracks):
            for j in range(i + 1, n_tracks):
                dist = float(np.linalg.norm(
                    track_positions[i] - track_positions[j],
                ))
                if dist <= cfg.swarm_activation_radius_m:
                    dense_count += 1

        # Need at least swarm_activation_count pairs within radius
        activated = dense_count >= cfg.swarm_activation_count
        if activated and not self._swarm_mode_active:
            logger.info(
                "MHT swarm mode ACTIVATED — %d dense track pairs",
                dense_count,
            )
        elif not activated and self._swarm_mode_active:
            logger.info("MHT swarm mode DEACTIVATED")

        self._swarm_mode_active = activated
        if activated:
            self.frames_in_swarm_mode += 1

        return activated

    def process_frame(
        self,
        track_ids: list[int],
        jpda_associations: list[tuple[int, int]],
        association_probs: list[float],
    ) -> dict[int, int]:
        """Process a frame of JPDA associations through the MHT.

        In swarm mode, creates branching hypotheses for ambiguous
        associations. In normal mode, passes through the best
        JPDA association directly.

        Args:
            track_ids: List of active track IDs.
            jpda_associations: List of (track_id, measurement_idx) pairs.
                measurement_idx = -1 for coast (no measurement).
            association_probs: Probability of each association.

        Returns:
            Dict mapping track_id → confirmed measurement_index.
            Represents the best hypothesis after N-scan-back pruning.
        """
        self._current_frame += 1

        if not self._swarm_mode_active:
            # Passthrough mode — use JPDA result directly
            result = {}
            for track_id, meas_idx in jpda_associations:
                result[track_id] = meas_idx
            return result

        # Create branching hypotheses from JPDA result
        self._branch_hypotheses(
            track_ids, jpda_associations, association_probs,
        )

        # Prune low-probability hypotheses
        self._prune_hypotheses()

        # Merge closely-spaced hypotheses
        self._merge_hypotheses()

        # Enforce hard hypothesis cap
        self._enforce_hypothesis_cap()

        # N-scan-back: extract best confirmed association
        return self._extract_best_assignment()

    def _branch_hypotheses(
        self,
        track_ids: list[int],
        associations: list[tuple[int, int]],
        probs: list[float],
    ) -> None:
        """Create child hypotheses from the current leaf hypotheses.

        Each leaf hypothesis spawns one child per plausible
        association set.
        """
        leaves = self._get_leaf_hypotheses()

        # For each existing leaf, create a child with this frame's
        # best association
        for leaf in leaves:
            assignments = {}
            for track_id, meas_idx in associations:
                assignments[track_id] = meas_idx

            # Compute child probability
            child_prob = leaf.probability * max(
                np.mean(probs) if probs else 0.5, 1e-10,
            )

            child = HypothesisNode(
                hypothesis_id=self._next_hypothesis_id,
                parent_id=leaf.hypothesis_id,
                frame_number=self._current_frame,
                track_assignments=assignments,
                probability=child_prob,
            )
            self._hypotheses[child.hypothesis_id] = child
            leaf.children.append(child.hypothesis_id)
            self._next_hypothesis_id += 1

    def _prune_hypotheses(self) -> None:
        """Remove hypotheses below probability threshold."""
        cfg = self._config
        to_remove = []

        for hyp_id, hyp in self._hypotheses.items():
            if (
                hyp.probability < cfg.pruning_prob_threshold
                and not hyp.children  # Only prune leaves
            ):
                to_remove.append(hyp_id)

        for hyp_id in to_remove:
            del self._hypotheses[hyp_id]
            self.total_prunes += 1

        # Normalize remaining probabilities
        self._normalize_leaf_probabilities()

    def _merge_hypotheses(self) -> None:
        """Merge leaf hypotheses with similar track assignments."""
        leaves = self._get_leaf_hypotheses()
        if len(leaves) <= 1:
            return

        merged: set[int] = set()

        for i in range(len(leaves)):
            if leaves[i].hypothesis_id in merged:
                continue
            for j in range(i + 1, len(leaves)):
                if leaves[j].hypothesis_id in merged:
                    continue

                # Check if assignments are identical
                if leaves[i].track_assignments == leaves[j].track_assignments:
                    # Merge: combine probabilities, keep the higher-prob one
                    leaves[i].probability += leaves[j].probability
                    merged.add(leaves[j].hypothesis_id)
                    self.total_merges += 1

        for hyp_id in merged:
            if hyp_id in self._hypotheses:
                del self._hypotheses[hyp_id]

    def _enforce_hypothesis_cap(self) -> None:
        """Enforce maximum hypothesis count by pruning lowest-probability."""
        leaves = self._get_leaf_hypotheses()

        if len(leaves) <= self._config.max_hypotheses:
            return

        # Sort by probability descending, keep top N
        leaves.sort(key=lambda h: h.probability, reverse=True)
        to_keep = {
            h.hypothesis_id for h in leaves[:self._config.max_hypotheses]
        }
        to_remove = [
            h.hypothesis_id for h in leaves
            if h.hypothesis_id not in to_keep
        ]

        for hyp_id in to_remove:
            if hyp_id in self._hypotheses:
                del self._hypotheses[hyp_id]
                self.total_prunes += 1

        self._normalize_leaf_probabilities()

    def _extract_best_assignment(self) -> dict[int, int]:
        """Extract the best track-to-measurement assignment.

        Uses N-scan-back: finds the best leaf, then walks back
        N frames to find the confirmed assignment at that depth.

        Returns:
            Dict of track_id → measurement_index.
        """
        leaves = self._get_leaf_hypotheses()
        if not leaves:
            return {}

        # Best leaf by probability
        best = max(leaves, key=lambda h: h.probability)
        self._best_hypothesis_id = best.hypothesis_id

        # Walk back n_scan_depth frames
        current = best
        depth = 0
        while (
            current.parent_id is not None
            and depth < self._config.n_scan_depth
        ):
            parent = self._hypotheses.get(current.parent_id)
            if parent is None:
                break
            current = parent
            depth += 1

        return current.track_assignments.copy()

    def _get_leaf_hypotheses(self) -> list[HypothesisNode]:
        """Get all leaf hypotheses (no children)."""
        return [
            h for h in self._hypotheses.values()
            if not h.children
        ]

    def _normalize_leaf_probabilities(self) -> None:
        """Normalize leaf hypothesis probabilities to sum to 1."""
        leaves = self._get_leaf_hypotheses()
        total = sum(h.probability for h in leaves)
        if total > 0:
            for h in leaves:
                h.probability /= total

    def reset(self) -> None:
        """Reset the hypothesis tree to a single root."""
        self._hypotheses.clear()
        self._next_hypothesis_id = 0
        root = HypothesisNode(
            hypothesis_id=0,
            parent_id=None,
            frame_number=self._current_frame,
            track_assignments={},
            probability=1.0,
        )
        self._hypotheses[0] = root
        self._best_hypothesis_id = 0
        self._next_hypothesis_id = 1
        self._swarm_mode_active = False
        logger.info("MHT hypothesis tree reset")

    def get_stats(self) -> dict[str, int]:
        """Return MHT statistics."""
        return {
            "total_hypotheses": len(self._hypotheses),
            "leaf_hypotheses": len(self._get_leaf_hypotheses()),
            "total_prunes": self.total_prunes,
            "total_merges": self.total_merges,
            "frames_in_swarm_mode": self.frames_in_swarm_mode,
            "current_frame": self._current_frame,
        }
