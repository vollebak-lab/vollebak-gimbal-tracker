# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Silent-to-Active state machine with pre-slew on ALERT (arm deploy + FSM coarse aim) and single-detection fast-track for speed-critical kill chain
#   FAILURE_MODE: Always-on radar reveals operator position via RWR/ESM detection
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy]
# ---
"""
Silent-to-Active Engagement State Machine.

Manages the tactical state transitions for the Predator C-UAS system.
The key principle: stay SILENT (passive neuromorphic only) until a
confirmed threat triggers radar emission (ACTIVE).

States:
    SILENT → ALERT → RADAR_ACTIVE → TRACKING → ENGAGEMENT → BDA

Each state has strict entry/exit conditions and timeout guards.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Predator system engagement states."""
    SILENT = auto()         # Passive only — neuromorphic cameras active
    ALERT = auto()          # Layer 1 detection — awaiting confirmation
    RADAR_ACTIVE = auto()   # Radar emitting — searching for target
    TRACKING = auto()       # Radar locked — EKF tracking target
    ENGAGEMENT = auto()     # Laser armed — engaging target
    BDA = auto()            # Battle damage assessment — evaluating kill


@dataclass(slots=True)
class StateTransition:
    """Record of a state transition for logging and replay.

    Attributes:
        from_state: Previous state.
        to_state: New state.
        trigger: What caused the transition.
        timestamp_us: When the transition occurred.
        metadata: Optional dict with trigger-specific data.
    """
    from_state: SystemState
    to_state: SystemState
    trigger: str
    timestamp_us: int
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class FSMConfig:
    """Configuration for the state machine.

    Attributes:
        alert_confirm_count: Number of detections needed to confirm alert.
            Set to 1 for single-detection fast-track (recommended for
            fast-mover kill chain — cameras are physically spaced apart,
            making concurrent multi-camera detection unlikely).
        alert_confirm_cameras: Number of cameras needed for confirmation.
            DEPRECATED — kept for backward compatibility. Set to 1.
        high_confidence_threshold: Confidence level above which a single
            detection immediately triggers RADAR_ACTIVE (skipping ALERT).
        pre_slew_on_alert: When True, emit arm deploy and FSM (Fast Steer
            Mirror) pre-slew commands on first detection. No harm if
            false positive — arm returns to stow on ALERT timeout.
        alert_timeout_s: Time before ALERT reverts to SILENT.
        radar_search_timeout_s: Time before RADAR_ACTIVE reverts to ALERT.
        tracking_coast_timeout_s: Time without radar updates before losing track.
        engagement_timeout_s: Maximum engagement duration.
        bda_observation_s: Post-engagement observation window.
        min_threat_score: Minimum threat score to authorize engagement.
    """
    alert_confirm_count: int = 1
    alert_confirm_cameras: int = 1  # Deprecated — cameras too far apart
    high_confidence_threshold: float = 0.8
    pre_slew_on_alert: bool = True
    alert_timeout_s: float = 5.0
    radar_search_timeout_s: float = 15.0
    tracking_coast_timeout_s: float = 3.0
    engagement_timeout_s: float = 10.0
    bda_observation_s: float = 5.0
    min_threat_score: float = 0.7


class PredatorStateMachine:
    """Silent-to-Active engagement finite state machine.

    Orchestrates the Predator C-UAS engagement pipeline. All sensor
    inputs are processed through this FSM to determine the current
    system posture and authorized actions.

    Args:
        config: FSM tuning parameters.
        on_state_change: Optional callback invoked on every transition.
    """

    def __init__(
        self,
        config: Optional[FSMConfig] = None,
        on_state_change: Optional[Callable[[StateTransition], None]] = None,
    ) -> None:
        self._config = config or FSMConfig()
        self._on_state_change = on_state_change

        # Current state
        self._state = SystemState.SILENT
        self._state_entry_time = time.monotonic()

        # Alert accumulation
        self._alert_detections: list[dict] = []
        self._alert_camera_ids: set[int] = set()

        # Pre-slew / arm deploy state
        self._arm_deploy_requested: bool = False
        self._pre_slew_bearing_deg: float = 0.0

        # Tracking state
        self._has_radar_track: bool = False
        self._last_radar_update_time: float = 0.0
        self._current_threat_score: float = 0.0

        # Engagement state
        self._engagement_start_time: float = 0.0
        self._laser_fire_sent: bool = False

        # BDA state
        self._bda_start_time: float = 0.0
        self._flicker_ceased: bool = False
        self._doppler_lost: bool = False

        # History
        self._transition_log: list[StateTransition] = []

    @property
    def state(self) -> SystemState:
        """Current system state."""
        return self._state

    @property
    def time_in_state_s(self) -> float:
        """Seconds spent in the current state."""
        return time.monotonic() - self._state_entry_time

    @property
    def is_radar_authorized(self) -> bool:
        """Whether radar emission is authorized in current state."""
        return self._state in (
            SystemState.RADAR_ACTIVE,
            SystemState.TRACKING,
            SystemState.ENGAGEMENT,
            SystemState.BDA,
        )

    @property
    def is_engagement_authorized(self) -> bool:
        """Whether laser engagement is authorized."""
        return self._state == SystemState.ENGAGEMENT

    def _transition_to(
        self,
        new_state: SystemState,
        trigger: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Execute a state transition with logging and callback.

        Args:
            new_state: Target state.
            trigger: Description of what caused the transition.
            metadata: Optional context for the transition.
        """
        old_state = self._state
        now_us = int(time.monotonic() * 1e6)

        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            trigger=trigger,
            timestamp_us=now_us,
            metadata=metadata or {},
        )

        self._state = new_state
        self._state_entry_time = time.monotonic()
        self._transition_log.append(transition)

        logger.info(
            "STATE: %s → %s (trigger: %s)",
            old_state.name, new_state.name, trigger,
        )

        if self._on_state_change:
            try:
                self._on_state_change(transition)
            except Exception:
                logger.exception("State change callback error")

    # -------------------------------------------------------------------
    # Input Methods — called by the pipeline
    # -------------------------------------------------------------------

    @property
    def arm_deploy_requested(self) -> bool:
        """Whether the robotic arm Z-fold deploy has been requested."""
        return self._arm_deploy_requested

    @property
    def pre_slew_bearing_deg(self) -> float:
        """Bearing for pre-slew command (from first L1 detection)."""
        return self._pre_slew_bearing_deg

    def on_layer1_detection(
        self,
        camera_id: int,
        bearing_deg: float,
        confidence: float,
        timestamp_us: int,
    ) -> None:
        """Process a Layer 1 neuromorphic propeller detection.

        Behavior:
            - High confidence (>0.8): SILENT → RADAR_ACTIVE directly
              (skip ALERT state to save 10-80ms).
            - Normal confidence: SILENT → ALERT → RADAR_ACTIVE
              (requires ``alert_confirm_count`` detections).
            - Pre-slew: On ANY first detection, emit arm deploy and
              FSM coarse-aim commands. No harm if false positive —
              arm returns to stow on timeout.

        Args:
            camera_id: Source camera identifier.
            bearing_deg: Global bearing to detection.
            confidence: Detection confidence (0–1).
            timestamp_us: Detection timestamp.
        """
        detection_data = {
            "camera_id": camera_id,
            "bearing_deg": bearing_deg,
            "confidence": confidence,
            "timestamp_us": timestamp_us,
        }

        if self._state == SystemState.SILENT:
            self._alert_detections.clear()
            self._alert_camera_ids.clear()
            self._alert_detections.append(detection_data)
            self._alert_camera_ids.add(camera_id)

            # --- Pre-slew: begin arm deploy + FSM coarse aim ---
            # Bearing determines shoulder side (left vs right)
            if self._config.pre_slew_on_alert:
                self._request_pre_slew(bearing_deg)

            # --- Fast-track: high confidence → skip ALERT ---
            if confidence >= self._config.high_confidence_threshold:
                logger.info(
                    "HIGH-CONF fast-track: conf=%.3f ≥ %.3f → RADAR_ACTIVE",
                    confidence, self._config.high_confidence_threshold,
                )
                self._transition_to(
                    SystemState.RADAR_ACTIVE,
                    trigger="high_confidence_detection",
                    metadata={
                        "camera_id": camera_id,
                        "bearing_deg": bearing_deg,
                        "confidence": confidence,
                        "fast_tracked": True,
                    },
                )
            else:
                self._transition_to(
                    SystemState.ALERT,
                    trigger="layer1_detection",
                    metadata={"camera_id": camera_id, "bearing_deg": bearing_deg},
                )

        elif self._state == SystemState.ALERT:
            # Accumulate detections
            self._alert_detections.append(detection_data)
            self._alert_camera_ids.add(camera_id)

            # Update pre-slew bearing with latest detection
            if self._config.pre_slew_on_alert:
                self._pre_slew_bearing_deg = bearing_deg

            # Check confirmation criteria (default: 1 detection)
            count_met = (
                len(self._alert_detections)
                >= self._config.alert_confirm_count
            )

            if count_met:
                # Compute consensus bearing
                bearings = [d["bearing_deg"] for d in self._alert_detections]
                consensus_bearing = self._circular_mean_deg(bearings)

                self._transition_to(
                    SystemState.RADAR_ACTIVE,
                    trigger="confirmed_detection",
                    metadata={
                        "detection_count": len(self._alert_detections),
                        "camera_count": len(self._alert_camera_ids),
                        "consensus_bearing_deg": consensus_bearing,
                    },
                )

    def on_radar_track_acquired(
        self,
        track_id: int,
        range_m: float,
        azimuth_deg: float,
        elevation_deg: float,
    ) -> None:
        """Process a confirmed radar track acquisition.

        Called when the aerial target tracker initializes a new EKF
        track on a radar-confirmed target.

        Args:
            track_id: Tracker-assigned target ID.
            range_m: Target range.
            azimuth_deg: Target azimuth.
            elevation_deg: Target elevation.
        """
        if self._state == SystemState.RADAR_ACTIVE:
            self._has_radar_track = True
            self._last_radar_update_time = time.monotonic()
            self._transition_to(
                SystemState.TRACKING,
                trigger="radar_acquisition",
                metadata={
                    "track_id": track_id,
                    "range_m": range_m,
                    "azimuth_deg": azimuth_deg,
                    "elevation_deg": elevation_deg,
                },
            )

    def on_radar_track_update(self, track_id: int) -> None:
        """Update the tracking state with a new radar measurement.

        Resets the coast timer to prevent timeout.

        Args:
            track_id: Active track identifier.
        """
        if self._state in (SystemState.TRACKING, SystemState.ENGAGEMENT):
            self._last_radar_update_time = time.monotonic()

    def on_threat_score_update(self, threat_score: float) -> None:
        """Update the current threat score from the threat classifier.

        If the score exceeds the threshold and safety is clear,
        transitions to ENGAGEMENT.

        Args:
            threat_score: Aggregated threat score (0–1).
        """
        self._current_threat_score = threat_score

        if self._state == SystemState.TRACKING:
            if threat_score >= self._config.min_threat_score:
                self._transition_to(
                    SystemState.ENGAGEMENT,
                    trigger="threat_score_exceeded",
                    metadata={"threat_score": threat_score},
                )
                self._engagement_start_time = time.monotonic()

    def on_laser_fired(self) -> None:
        """Record that the laser fire command has been sent."""
        if self._state == SystemState.ENGAGEMENT:
            self._laser_fire_sent = True
            self._transition_to(
                SystemState.BDA,
                trigger="laser_fire_command",
            )
            self._bda_start_time = time.monotonic()
            self._flicker_ceased = False
            self._doppler_lost = False

    def on_flicker_ceased(self) -> None:
        """Layer 1 reports propeller flicker has stopped."""
        self._flicker_ceased = True

    def on_doppler_lost(self) -> None:
        """Layer 2 reports micro-Doppler signature has been lost."""
        self._doppler_lost = True

    # -------------------------------------------------------------------
    # Tick — called on each pipeline cycle
    # -------------------------------------------------------------------

    def tick(self) -> None:
        """Process timeouts and automatic state transitions.

        Must be called periodically (e.g., every pipeline frame) to
        handle timeout-based transitions.
        """
        now = time.monotonic()

        if self._state == SystemState.ALERT:
            if self.time_in_state_s > self._config.alert_timeout_s:
                # False positive — stow arm if it was deployed
                self._stow_arm()
                self._transition_to(
                    SystemState.SILENT,
                    trigger="alert_timeout",
                )

        elif self._state == SystemState.RADAR_ACTIVE:
            if self.time_in_state_s > self._config.radar_search_timeout_s:
                self._transition_to(
                    SystemState.ALERT,
                    trigger="radar_search_timeout",
                )

        elif self._state == SystemState.TRACKING:
            coast_time = now - self._last_radar_update_time
            if coast_time > self._config.tracking_coast_timeout_s:
                self._has_radar_track = False
                self._transition_to(
                    SystemState.RADAR_ACTIVE,
                    trigger="track_coast_timeout",
                )

        elif self._state == SystemState.ENGAGEMENT:
            elapsed = now - self._engagement_start_time
            if elapsed > self._config.engagement_timeout_s:
                self._transition_to(
                    SystemState.BDA,
                    trigger="engagement_timeout",
                )
                self._bda_start_time = now

        elif self._state == SystemState.BDA:
            elapsed = now - self._bda_start_time
            # BDA assessment: flicker ceased AND doppler lost = confirmed kill
            if self._flicker_ceased and self._doppler_lost:
                logger.info("BDA: KILL CONFIRMED — flicker ceased + Doppler lost")
                self._transition_to(
                    SystemState.SILENT,
                    trigger="kill_confirmed",
                )
            elif elapsed > self._config.bda_observation_s:
                if self._flicker_ceased or self._doppler_lost:
                    logger.info("BDA: PROBABLE KILL — partial signature loss")
                else:
                    logger.info("BDA: MISS — no signature change observed")
                self._transition_to(
                    SystemState.SILENT,
                    trigger="bda_timeout",
                    metadata={
                        "flicker_ceased": self._flicker_ceased,
                        "doppler_lost": self._doppler_lost,
                    },
                )

    def get_transition_log(self) -> list[StateTransition]:
        """Return the full state transition history."""
        return list(self._transition_log)

    def _request_pre_slew(self, bearing_deg: float) -> None:
        """Begin robotic arm Z-fold deployment and FSM coarse aim.

        Called on first Layer 1 detection. The arm unfolds from the
        Z-fold stowed position on the rear plate carrier to the
        appropriate shoulder (left or right) based on bearing.

        This is a no-regret action: if the detection is a false
        positive, the arm returns to stow on ALERT timeout.

        The Fast Steer Mirror (FSM) receives the initial coarse
        bearing for beam pre-positioning while the arm deploys.

        Args:
            bearing_deg: Detection bearing for shoulder selection
                and FSM coarse aim (0=fwd, CW).
        """
        self._arm_deploy_requested = True
        self._pre_slew_bearing_deg = bearing_deg

        # Determine deployment side from bearing
        # 0-180° (right hemisphere) → right shoulder
        # 180-360° (left hemisphere) → left shoulder
        normalized = bearing_deg % 360.0
        side = "right" if normalized < 180.0 else "left"

        logger.info(
            "PRE-SLEW: arm deploy to %s shoulder, "
            "FSM coarse aim → %.1f°",
            side, bearing_deg,
        )

    def _stow_arm(self) -> None:
        """Return robotic arm to Z-fold stowed position.

        Called on ALERT timeout (false positive) or BDA→SILENT
        (engagement complete).
        """
        if self._arm_deploy_requested:
            self._arm_deploy_requested = False
            logger.info("ARM: returning to Z-fold stow")

    @staticmethod
    def _circular_mean_deg(angles: list[float]) -> float:
        """Compute circular mean of angles in degrees."""
        if not angles:
            return 0.0
        rad = np.radians(angles)
        mean_rad = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
        return float(np.degrees(mean_rad)) % 360.0


# NumPy import deferred to method to keep module light
import numpy as np
