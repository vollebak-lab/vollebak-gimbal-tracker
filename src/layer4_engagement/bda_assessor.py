# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Multi-indicator BDA fuses flicker cessation, Doppler loss, RCS change, and ballistic onset for high-confidence kill assessment
#   FAILURE_MODE: Single-indicator BDA (e.g., flicker loss alone) cannot distinguish soft-kill from target temporarily leaving FOV
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: [numpy]
# ---
"""
Battle Damage Assessment (BDA) — Post-Engagement Kill Evaluation.

Monitors sensor indicators after laser engagement to determine
whether the soft-kill was effective:
    1. Propeller flicker cessation (Layer 1 stops detecting)
    2. Micro-Doppler signature loss (Layer 2 blade-rate disappears)
    3. RCS change (damaged drone may show different RCS)
    4. Ballistic trajectory onset (drone falls if motors killed)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BDAResult(Enum):
    """Assessment outcome."""
    KILL_CONFIRMED = auto()
    PROBABLE_KILL = auto()
    MISS = auto()
    PENDING = auto()


@dataclass(slots=True)
class BDAConfig:
    """BDA assessment parameters."""
    observation_window_s: float = 5.0
    flicker_absence_threshold_s: float = 2.0
    doppler_absence_threshold_s: float = 2.0
    ballistic_accel_threshold_mps2: float = 7.0   # ~0.7g downward
    min_indicators_for_kill: int = 2
    min_indicators_for_probable: int = 1

    # L1-only BDA (Waiter mode) — only flicker + ballistic available
    l1_flicker_absence_threshold_s: float = 1.0  # Faster—close range = high SNR
    l1_min_indicators_for_kill: int = 1           # Flicker cessation alone = kill


@dataclass(slots=True)
class BDAState:
    """Running BDA observation state."""
    start_time: float = 0.0
    last_flicker_time: float = 0.0
    last_doppler_time: float = 0.0
    pre_engagement_rcs: float = -15.0
    post_engagement_rcs_samples: list[float] = field(default_factory=list)
    vertical_accel_samples: list[float] = field(default_factory=list)
    result: BDAResult = BDAResult.PENDING
    l1_only: bool = False  # True when engaged via Waiter mode (no radar data)


class BDAAssessor:
    """Post-engagement kill assessment engine.

    Begins observation after laser fire command and monitors multiple
    sensor indicators for the configured observation window.

    Args:
        config: BDA assessment parameters.
    """

    def __init__(self, config: Optional[BDAConfig] = None) -> None:
        self._config = config or BDAConfig()
        self._state: Optional[BDAState] = None
        self.assessments_total: int = 0

    def begin_assessment(self, pre_engagement_rcs: float = -15.0) -> None:
        """Start a new BDA observation window.

        Args:
            pre_engagement_rcs: Target RCS before engagement for comparison.
        """
        now = time.monotonic()
        self._state = BDAState(
            start_time=now,
            last_flicker_time=now,
            last_doppler_time=now,
            pre_engagement_rcs=pre_engagement_rcs,
        )
        logger.info("BDA observation started (window=%.1fs)", self._config.observation_window_s)

    def begin_l1_assessment(self) -> None:
        """Start an L1-only BDA observation (Waiter mode).

        L1-only BDA uses only flicker cessation and ballistic onset.
        Doppler and RCS indicators are unavailable because radar was
        never activated (zero RF emission kill chain).

        At close range (<50 m), flicker cessation is high-confidence:
        all 4+ propellers are visible, so cessation strongly indicates
        motor kill from pulsed laser CMOS damage.
        """
        now = time.monotonic()
        self._state = BDAState(
            start_time=now,
            last_flicker_time=now,
            last_doppler_time=now,
            l1_only=True,
        )
        logger.info(
            "BDA observation started (L1-ONLY, window=%.1fs)",
            self._config.observation_window_s,
        )

    def on_flicker_detected(self) -> None:
        """Layer 1 reports propeller flicker is still present."""
        if self._state:
            self._state.last_flicker_time = time.monotonic()

    def on_doppler_detected(self) -> None:
        """Layer 2 reports micro-Doppler is still present."""
        if self._state:
            self._state.last_doppler_time = time.monotonic()

    def on_rcs_measurement(self, rcs_dbsm: float) -> None:
        """Layer 2 provides an RCS measurement."""
        if self._state:
            self._state.post_engagement_rcs_samples.append(rcs_dbsm)

    def on_vertical_acceleration(self, az_mps2: float) -> None:
        """Tracker provides vertical acceleration estimate."""
        if self._state:
            self._state.vertical_accel_samples.append(az_mps2)

    def evaluate(self) -> BDAResult:
        """Evaluate current BDA indicators and return assessment.

        When ``l1_only=True``, only flicker cessation and ballistic
        onset are evaluated. Doppler and RCS are unavailable.

        Should be called periodically during the observation window.

        Returns:
            Current BDA assessment result.
        """
        if self._state is None:
            return BDAResult.PENDING

        now = time.monotonic()
        cfg = self._config
        state = self._state
        elapsed = now - state.start_time

        if state.l1_only:
            return self._evaluate_l1_only(now, elapsed, cfg, state)

        # Standard multi-indicator BDA
        indicators = 0

        # 1. Flicker cessation
        flicker_absent_s = now - state.last_flicker_time
        if flicker_absent_s >= cfg.flicker_absence_threshold_s:
            indicators += 1

        # 2. Doppler loss
        doppler_absent_s = now - state.last_doppler_time
        if doppler_absent_s >= cfg.doppler_absence_threshold_s:
            indicators += 1

        # 3. Ballistic trajectory (strong downward acceleration)
        if state.vertical_accel_samples:
            recent_accel = state.vertical_accel_samples[-min(10, len(state.vertical_accel_samples)):]
            mean_vert_accel = float(np.mean(recent_accel))
            if mean_vert_accel < -cfg.ballistic_accel_threshold_mps2:
                indicators += 1

        # 4. RCS change (>6dB shift indicates structural damage)
        if len(state.post_engagement_rcs_samples) >= 3:
            mean_post_rcs = float(np.mean(state.post_engagement_rcs_samples))
            rcs_change_db = abs(mean_post_rcs - state.pre_engagement_rcs)
            if rcs_change_db > 6.0:
                indicators += 1

        # Determine result
        if indicators >= cfg.min_indicators_for_kill:
            state.result = BDAResult.KILL_CONFIRMED
        elif indicators >= cfg.min_indicators_for_probable:
            state.result = BDAResult.PROBABLE_KILL
        elif elapsed >= cfg.observation_window_s:
            state.result = BDAResult.MISS
            self.assessments_total += 1

        if state.result != BDAResult.PENDING:
            logger.info("BDA result: %s (%d indicators, %.1fs elapsed)",
                        state.result.name, indicators, elapsed)
            self.assessments_total += 1

        return state.result

    def _evaluate_l1_only(
        self,
        now: float,
        elapsed: float,
        cfg: BDAConfig,
        state: BDAState,
    ) -> BDAResult:
        """L1-only BDA: flicker cessation + ballistic onset only.

        At close range (<50 m), flicker cessation is high-confidence
        because all propellers are visible. The pulsed laser damages
        the drone's CMOS sensor, causing navigation failure and
        eventual crash.

        Args:
            now: Current monotonic time.
            elapsed: Seconds since assessment started.
            cfg: BDA configuration.
            state: Current BDA state.

        Returns:
            BDA assessment result.
        """
        indicators = 0

        # 1. Flicker cessation (faster threshold for close range)
        flicker_absent_s = now - state.last_flicker_time
        if flicker_absent_s >= cfg.l1_flicker_absence_threshold_s:
            indicators += 1

        # 2. Ballistic onset (if velocity data available from L1)
        if state.vertical_accel_samples:
            recent = state.vertical_accel_samples[
                -min(5, len(state.vertical_accel_samples)):
            ]
            mean_accel = float(np.mean(recent))
            if mean_accel < -cfg.ballistic_accel_threshold_mps2:
                indicators += 1

        # Determine result
        if indicators >= cfg.l1_min_indicators_for_kill:
            state.result = BDAResult.KILL_CONFIRMED
        elif indicators >= 1 and elapsed >= cfg.observation_window_s / 2:
            state.result = BDAResult.PROBABLE_KILL
        elif elapsed >= cfg.observation_window_s:
            state.result = BDAResult.MISS
            self.assessments_total += 1

        if state.result != BDAResult.PENDING:
            logger.info(
                "BDA (L1-ONLY): %s (%d indicators, %.1fs elapsed)",
                state.result.name, indicators, elapsed,
            )
            self.assessments_total += 1

        return state.result

    @property
    def is_active(self) -> bool:
        """Whether a BDA observation is currently in progress."""
        return self._state is not None and self._state.result == BDAResult.PENDING
