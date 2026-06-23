"""Append-only lineage events for CurvyTron feedback-loop boundaries."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


LINEAGE_EVENT_SCHEMA_ID = "curvyzero_feedback_loop_lineage_event/v1"
LINEAGE_EVENTS_FILENAME = "lineage_events.jsonl"

LINEAGE_STAGES = (
    "checkpoint_written",
    "checkpoint_intake_seen",
    "checkpoint_intake_enqueued",
    "rating_spawn_claimed",
    "rating_round_started",
    "rating_round_reduced",
    "rating_latest_written",
    "leaderboard_published",
    "training_candidate_assignment_written",
    "assignment_pointer_rewritten",
    "trainer_assignment_loaded",
    "trainer_assignment_applied",
)


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def lineage_event(
    *,
    stage: str,
    status: str = "ok",
    reason: str | None = None,
    observed_at: str | None = None,
    event_id: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    stage_text = str(stage)
    if stage_text not in LINEAGE_STAGES:
        raise ValueError(f"stage must be one of {LINEAGE_STAGES!r}; got {stage!r}")
    status_text = str(status).strip()
    if not status_text:
        raise ValueError("status must be non-empty")
    event: dict[str, Any] = {
        "schema_id": LINEAGE_EVENT_SCHEMA_ID,
        "event_id": event_id or uuid.uuid4().hex,
        "observed_at": observed_at or utc_timestamp(),
        "stage": stage_text,
        "status": status_text,
    }
    if reason is not None:
        event["reason"] = str(reason)
    for key, value in fields.items():
        if value is not None:
            event[str(key)] = _jsonable(value)
    return event


def append_lineage_event(
    path: Path | str,
    *,
    best_effort: bool = True,
    **event_fields: Any,
) -> dict[str, Any]:
    event = lineage_event(**event_fields)
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")
    except Exception as exc:
        if not best_effort:
            raise
        return {
            "ok": False,
            "path": str(target),
            "event": event,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "ok": True,
        "path": str(target),
        "event_id": event["event_id"],
        "event": event,
    }


def lineage_events_path(root: Path | str) -> Path:
    """Return the per-owner feedback-loop lineage path under an artifact root."""

    return Path(root) / "feedback_loop" / LINEAGE_EVENTS_FILENAME


__all__ = [
    "LINEAGE_EVENT_SCHEMA_ID",
    "LINEAGE_EVENTS_FILENAME",
    "LINEAGE_STAGES",
    "append_lineage_event",
    "lineage_event",
    "lineage_events_path",
]
