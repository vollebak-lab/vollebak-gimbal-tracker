# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Peer-to-peer cooperative energy management — dynamically elects leader Predator node with best target geometry + battery for FULL_TRACK, subordinate nodes drop to DEEP_SLEEP for 240% energy reduction per node
#   FAILURE_MODE: Without cooperative delegation, all Predator nodes in a network track independently — wasteful battery consumption and unnecessary RF signature
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: RoC Research Paper — Section "Networked Energy Saving via Peer-to-Peer Sensor Fusion"
#   DEPENDENCIES: [numpy]
# ---
"""
Cooperative Tracker — Peer-to-Peer Radar Energy Management (Layer 5).

Implements distributed tracking across a **homogeneous Predator-to-Predator**
network. Each node advertises its state (battery, target geometry, power mode)
and the network dynamically elects a leader to perform FULL_TRACK while
subordinate nodes drop to DEEP_SLEEP.

Key Concepts
~~~~~~~~~~~~
1. **Leader Election**: Node with best aspect angle to target AND highest
   battery reserves wins active tracking duty.
2. **Energy Delegation**: Subordinate nodes cease radar emission, saving
   ~9.5W each. Paper cites 240% energy reduction per node.
3. **Handoff Protocol**: As the leader's battery depletes or target geometry
   degrades, leadership transfers to next-best node seamlessly.

Communication
~~~~~~~~~~~~~
Uses Zenoh pub/sub (already in Predator dependency stack) for:
- Node heartbeat / state advertisement (1Hz)
- Track state sharing (on update)
- Leader election messaging

Scope Limitations
~~~~~~~~~~~~~~~~~
- **Homogeneous only**: All participants must be Predator units.
  Heterogeneous node support (fixed-site radar, EO sensors) is
  deferred to a later milestone.
- **No cooperative engagement**: This module handles tracking
  energy management only, not coordinated engagement authorization.

References:
    - Radar Fire Control RoC Research Paper, Section
      "Networked Energy Saving via Peer-to-Peer Sensor Fusion"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node State Advertisement
# ---------------------------------------------------------------------------

class NodeRole(Enum):
    """Role of this Predator node in the cooperative network."""
    STANDALONE = auto()   # No cooperative network active
    LEADER = auto()       # Active tracker — FULL_TRACK mode
    SUBORDINATE = auto()  # Energy-saving — DEEP_SLEEP mode
    CANDIDATE = auto()    # Participating in leader election


@dataclass(slots=True)
class NodeState:
    """State advertisement broadcast by each Predator node.

    Published at 1Hz via Zenoh pub/sub for peer discovery and
    leader election.

    Attributes:
        node_id: Unique Predator unit identifier.
        battery_fraction: Current battery level (0.0–1.0).
        target_aspect_quality: Quality of this node's viewing angle
            to the primary tracked target (0.0–1.0). Higher = better
            geometry for radar tracking (e.g., broadside aspect).
        target_range_m: Range to primary tracked target.
        current_role: This node's current cooperative role.
        radar_power_w: Current radar power draw.
        timestamp_s: Monotonic timestamp of this advertisement.
        track_count: Number of active tracks this node maintains.
    """
    node_id: str
    battery_fraction: float
    target_aspect_quality: float
    target_range_m: float
    current_role: NodeRole
    radar_power_w: float
    timestamp_s: float
    track_count: int = 0


# ---------------------------------------------------------------------------
# Cooperative Tracker Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CooperativeConfig:
    """Configuration for the cooperative tracker node.

    Attributes:
        heartbeat_interval_s: Interval between state advertisements.
        stale_threshold_s: Time after which a peer is considered lost.
        min_battery_for_leader: Minimum battery fraction to be
            eligible as leader. Below this, node yields leadership.
        handoff_hysteresis: Minimum improvement in election score
            before a handoff is triggered. Prevents oscillation.
        max_peers: Maximum number of cooperative peers tracked.
    """
    heartbeat_interval_s: float = 1.0
    stale_threshold_s: float = 5.0
    min_battery_for_leader: float = 0.15
    handoff_hysteresis: float = 0.1
    max_peers: int = 8


# ---------------------------------------------------------------------------
# Cooperative Tracker Node
# ---------------------------------------------------------------------------

class CooperativeTrackerNode:
    """Peer-to-peer cooperative tracking energy manager.

    Manages leader election and energy delegation for a network
    of homogeneous Predator units. Only the elected leader
    maintains FULL_TRACK radar mode; all other nodes drop to
    DEEP_SLEEP.

    Args:
        node_id: Unique identifier for this Predator unit.
        config: Cooperative tracker configuration.
    """

    def __init__(
        self,
        node_id: str,
        config: Optional[CooperativeConfig] = None,
    ) -> None:
        self._node_id = node_id
        self._config = config or CooperativeConfig()

        # Own state
        self._role = NodeRole.STANDALONE
        self._battery_fraction: float = 1.0
        self._target_aspect_quality: float = 0.0
        self._target_range_m: float = 0.0
        self._radar_power_w: float = 0.05  # Start in deep sleep
        self._track_count: int = 0

        # Peer registry
        self._peers: dict[str, NodeState] = {}
        self._last_heartbeat_s: float = 0.0

        # Statistics
        self.elections_triggered: int = 0
        self.handoffs_completed: int = 0
        self.energy_saved_wh: float = 0.0

    @property
    def role(self) -> NodeRole:
        """Current cooperative role."""
        return self._role

    @property
    def node_id(self) -> str:
        """This node's unique identifier."""
        return self._node_id

    @property
    def peer_count(self) -> int:
        """Number of active cooperative peers."""
        return len(self._peers)

    def update_own_state(
        self,
        battery_fraction: float,
        target_aspect_quality: float,
        target_range_m: float,
        radar_power_w: float,
        track_count: int = 0,
    ) -> None:
        """Update this node's state for cooperative decisions.

        Called each tracker frame with current operational state.

        Args:
            battery_fraction: Current battery level (0.0–1.0).
            target_aspect_quality: Viewing angle quality (0.0–1.0).
            target_range_m: Range to primary target.
            radar_power_w: Current radar power consumption.
            track_count: Number of active tracks.
        """
        self._battery_fraction = battery_fraction
        self._target_aspect_quality = target_aspect_quality
        self._target_range_m = target_range_m
        self._radar_power_w = radar_power_w
        self._track_count = track_count

    def receive_peer_heartbeat(self, peer_state: NodeState) -> None:
        """Process a received peer state advertisement.

        Called when a Zenoh message arrives from a peer node.

        Args:
            peer_state: The peer's current state.
        """
        if peer_state.node_id == self._node_id:
            return  # Ignore own heartbeat

        self._peers[peer_state.node_id] = peer_state

        # If we were standalone and now have peers, enter cooperative mode
        if self._role == NodeRole.STANDALONE and len(self._peers) > 0:
            self._role = NodeRole.CANDIDATE
            logger.info(
                "Node %s entering cooperative mode — %d peers detected",
                self._node_id, len(self._peers),
            )

    def evaluate_election(self) -> NodeRole:
        """Run leader election and determine this node's role.

        Election score combines:
        - Target aspect quality (60% weight) — best geometry wins
        - Battery fraction (40% weight) — energy reserves matter

        The node with highest score becomes LEADER. All others
        become SUBORDINATE.

        Returns:
            This node's role after election.
        """
        if not self._peers:
            self._role = NodeRole.STANDALONE
            return self._role

        # Prune stale peers
        now = time.monotonic()
        stale = [
            pid for pid, ps in self._peers.items()
            if (now - ps.timestamp_s) > self._config.stale_threshold_s
        ]
        for pid in stale:
            del self._peers[pid]
            logger.info("Peer %s pruned (stale)", pid)

        if not self._peers:
            self._role = NodeRole.STANDALONE
            return self._role

        # Compute own election score
        own_score = self._election_score(
            self._target_aspect_quality,
            self._battery_fraction,
        )

        # Check if we meet minimum battery threshold
        if self._battery_fraction < self._config.min_battery_for_leader:
            own_score = 0.0  # Yield leadership

        # Compare with all peers
        best_peer_score = 0.0
        best_peer_id = ""
        for pid, ps in self._peers.items():
            peer_score = self._election_score(
                ps.target_aspect_quality, ps.battery_fraction,
            )
            if ps.battery_fraction < self._config.min_battery_for_leader:
                peer_score = 0.0

            if peer_score > best_peer_score:
                best_peer_score = peer_score
                best_peer_id = pid

        # Apply hysteresis: only handoff if peer is significantly better
        old_role = self._role
        if own_score >= best_peer_score - self._config.handoff_hysteresis:
            self._role = NodeRole.LEADER
        else:
            self._role = NodeRole.SUBORDINATE

        # Track handoffs
        if old_role == NodeRole.LEADER and self._role == NodeRole.SUBORDINATE:
            self.handoffs_completed += 1
            logger.info(
                "HANDOFF: %s → %s (own=%.3f, best_peer=%.3f [%s])",
                self._node_id, best_peer_id,
                own_score, best_peer_score, best_peer_id,
            )
        elif old_role != NodeRole.LEADER and self._role == NodeRole.LEADER:
            logger.info(
                "ELECTED: %s as leader (score=%.3f, beat %d peers)",
                self._node_id, own_score, len(self._peers),
            )

        self.elections_triggered += 1
        return self._role

    def get_state_advertisement(self) -> NodeState:
        """Build this node's state advertisement for peer broadcast.

        Returns:
            NodeState for publication via Zenoh.
        """
        return NodeState(
            node_id=self._node_id,
            battery_fraction=self._battery_fraction,
            target_aspect_quality=self._target_aspect_quality,
            target_range_m=self._target_range_m,
            current_role=self._role,
            radar_power_w=self._radar_power_w,
            timestamp_s=time.monotonic(),
            track_count=self._track_count,
        )

    def get_recommended_power_mode(self) -> str:
        """Get recommended radar power mode based on cooperative role.

        Returns:
            "FULL_TRACK" for leader, "DEEP_SLEEP" for subordinate,
            "SECTOR_SEARCH" for standalone.
        """
        if self._role == NodeRole.LEADER:
            return "FULL_TRACK"
        elif self._role == NodeRole.SUBORDINATE:
            return "DEEP_SLEEP"
        else:
            return "SECTOR_SEARCH"

    @staticmethod
    def _election_score(
        aspect_quality: float, battery_fraction: float,
    ) -> float:
        """Compute election score for a node.

        Args:
            aspect_quality: Target viewing angle quality (0–1).
            battery_fraction: Battery level (0–1).

        Returns:
            Election score (0–1). Higher = better candidate.
        """
        return 0.6 * aspect_quality + 0.4 * battery_fraction
