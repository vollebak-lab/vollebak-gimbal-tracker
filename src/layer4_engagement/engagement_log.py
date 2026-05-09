# ---
# VOLLEBAK_ENGINEERING_METADATA:
#   PROJECT_ID: PREDATOR-01
#   TRACK: ECOSYSTEM
#   PHASE: CHALLENGE_HUB
#   CONTRIBUTOR_ID: Antigravity AI
#   THE_DELTA: Append-only JSON-lines engagement log for post-mission forensic review and legal accountability
#   FAILURE_MODE: Without persistent engagement records there is no audit trail for laser employment decisions
#   IP_STATUS: VOLLEBAK_PROPRIETARY
#   USER_FEEDBACK_REF: None
#   DEPENDENCIES: []
# ---
"""
Engagement Log — Append-Only Chronicle.

Records every engagement event as a JSON-lines file for post-mission
review. Each line is a complete, self-contained JSON object.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EngagementRecord:
    """A single engagement event record."""
    timestamp_iso: str
    timestamp_epoch_s: float
    event_type: str          # detection, tracking, engagement, bda
    system_state: str
    target_id: Optional[int]
    target_bearing_deg: Optional[float]
    target_elevation_deg: Optional[float]
    target_range_m: Optional[float]
    threat_score: Optional[float]
    bda_result: Optional[str]
    safety_check: Optional[str]
    metadata: dict


class EngagementLog:
    """Append-only JSON-lines engagement chronicle.

    Each engagement event is appended as a single JSON line to the
    log file. The file is opened in append mode and flushed after
    each write to ensure crash-safety.

    Args:
        log_path: Path to the engagement log file.
    """

    def __init__(self, log_path: str | Path = "engagement_log.jsonl") -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self.records_written: int = 0
        logger.info("Engagement log: %s", self._log_path)

    def record(
        self,
        event_type: str,
        system_state: str,
        target_id: Optional[int] = None,
        target_bearing_deg: Optional[float] = None,
        target_elevation_deg: Optional[float] = None,
        target_range_m: Optional[float] = None,
        threat_score: Optional[float] = None,
        bda_result: Optional[str] = None,
        safety_check: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        """Append an engagement record to the log.

        Args:
            event_type: Type of event being logged.
            system_state: Current FSM state name.
            target_id: Track ID if applicable.
            target_bearing_deg: Target bearing if applicable.
            target_elevation_deg: Target elevation if applicable.
            target_range_m: Target range if applicable.
            threat_score: Threat score if applicable.
            bda_result: BDA assessment if applicable.
            safety_check: Safety check result if applicable.
            **metadata: Additional key-value pairs.
        """
        import datetime

        record = EngagementRecord(
            timestamp_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            timestamp_epoch_s=time.time(),
            event_type=event_type,
            system_state=system_state,
            target_id=target_id,
            target_bearing_deg=target_bearing_deg,
            target_elevation_deg=target_elevation_deg,
            target_range_m=target_range_m,
            threat_score=threat_score,
            bda_result=bda_result,
            safety_check=safety_check,
            metadata=dict(metadata),
        )

        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), default=str) + "\n")
                f.flush()
            self.records_written += 1
        except Exception:
            logger.exception("Failed to write engagement record")
