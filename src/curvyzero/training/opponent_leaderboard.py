"""Pure leaderboard snapshot and opponent assignment helpers.

This module deliberately avoids Modal imports. It is the bridge between a
tournament rating snapshot and the trainer-facing opponent assignment contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from curvyzero.training.lightzero_checkpoints import (
    lightzero_iteration_from_checkpoint_name,
)
from curvyzero.training.opponent_mixture import (
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
)
from curvyzero.training.opponent_registry import (
    OPPONENT_ASSIGNMENT_SCHEMA_ID,
    canonical_assignment_json_sha256,
    parse_opponent_assignment_snapshot,
)


LEADERBOARD_SNAPSHOT_SCHEMA_ID = "curvyzero_opponent_leaderboard_snapshot/v0"
LEADERBOARD_POINTER_SCHEMA_ID = "curvyzero_opponent_leaderboard_pointer/v0"
OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID = "curvyzero_opponent_assignment_audit/v0"

DEFAULT_ACTIVE_MIN_DISTINCT_OPPONENTS = 20
DEFAULT_ACTIVE_MIN_VALID_GAMES = 300
DEFAULT_MAX_FAILURE_RATE = 0.02

DEFAULT_SLOT_WEIGHTS = {
    "champion": 20.0,
    "recent_strong": 12.0,
    "diverse_challenger": 10.0,
    "anchor": 6.0,
    "sentinel": 2.0,
}


def canonical_json_sha256(value: Any) -> str:
    """Return a stable SHA256 for one plain JSON-compatible value."""

    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_leaderboard_snapshot_from_rating_snapshot(
    rating_snapshot: Mapping[str, Any],
    *,
    leaderboard_id: str,
    snapshot_id: str,
    created_at: str | None = None,
    generation: int = 0,
    active_min_distinct_opponents: int = DEFAULT_ACTIVE_MIN_DISTINCT_OPPONENTS,
    active_min_valid_games: int = DEFAULT_ACTIVE_MIN_VALID_GAMES,
    max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE,
) -> dict[str, Any]:
    """Build an immutable public leaderboard snapshot from a rating snapshot."""

    if not leaderboard_id:
        raise ValueError("leaderboard_id is required")
    if not snapshot_id:
        raise ValueError("snapshot_id is required")
    rating_rows = rating_snapshot.get("ratings")
    if not isinstance(rating_rows, list) or not rating_rows:
        raise ValueError("rating snapshot must contain non-empty ratings")
    rows = [
        _leaderboard_row_from_rating_row(
            row,
            active_min_distinct_opponents=active_min_distinct_opponents,
            active_min_valid_games=active_min_valid_games,
            max_failure_rate=max_failure_rate,
        )
        for row in rating_rows
        if isinstance(row, Mapping)
    ]
    if not rows:
        raise ValueError("rating snapshot did not contain valid rating rows")
    snapshot = {
        "schema_id": LEADERBOARD_SNAPSHOT_SCHEMA_ID,
        "leaderboard_id": str(leaderboard_id),
        "snapshot_id": str(snapshot_id),
        "generation": int(generation),
        "created_at": created_at or rating_snapshot.get("created_at"),
        "source": {
            "kind": "checkpoint_tournament_rating",
            "tournament_id": rating_snapshot.get("tournament_id"),
            "rating_run_id": rating_snapshot.get("rating_run_id"),
            "rating_snapshot_ref": rating_snapshot.get("ratings_ref")
            or rating_snapshot.get("latest_ref"),
            "rating_snapshot_sha256": canonical_json_sha256(rating_snapshot),
        },
        "context": {
            "rating_schema_id": rating_snapshot.get("schema_id"),
            "rating_formula_version": rating_snapshot.get("formula_version"),
            "rating_context_hash": rating_snapshot.get("context_hash"),
            "checkpoint_roster_hash": rating_snapshot.get("roster_hash")
            or rating_snapshot.get("pool_hash"),
            "pool_status": "active",
        },
        "maturity_policy": {
            "active_min_distinct_opponents": int(active_min_distinct_opponents),
            "active_min_valid_games": int(active_min_valid_games),
            "max_failure_rate": float(max_failure_rate),
        },
        "rows": rows,
    }
    snapshot["snapshot_sha256"] = _snapshot_sha256(snapshot)
    return snapshot


def validate_leaderboard_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a public leaderboard snapshot."""

    if value.get("schema_id") != LEADERBOARD_SNAPSHOT_SCHEMA_ID:
        raise ValueError(f"leaderboard snapshot requires schema_id {LEADERBOARD_SNAPSHOT_SCHEMA_ID!r}")
    leaderboard_id = value.get("leaderboard_id")
    snapshot_id = value.get("snapshot_id")
    if not isinstance(leaderboard_id, str) or not leaderboard_id:
        raise ValueError("leaderboard snapshot requires leaderboard_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("leaderboard snapshot requires snapshot_id")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("leaderboard snapshot requires non-empty rows")
    normalized_rows = [_validate_leaderboard_row(row) for row in rows]
    normalized = dict(value)
    normalized["rows"] = normalized_rows
    expected_hash = _snapshot_sha256(normalized)
    supplied_hash = normalized.get("snapshot_sha256")
    if supplied_hash is not None and str(supplied_hash) != expected_hash:
        raise ValueError("leaderboard snapshot_sha256 does not match canonical payload")
    normalized["snapshot_sha256"] = expected_hash
    return normalized


def build_leaderboard_pointer(
    snapshot: Mapping[str, Any],
    *,
    snapshot_ref: str,
    published_at: str,
    writer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the small Modal Dict pointer payload for a leaderboard snapshot."""

    normalized = validate_leaderboard_snapshot(snapshot)
    if not snapshot_ref:
        raise ValueError("snapshot_ref is required")
    top_checkpoint_ids = [
        str(row["checkpoint_id"])
        for row in normalized["rows"][:5]
        if row.get("checkpoint_id")
    ]
    active_count = sum(1 for row in normalized["rows"] if row.get("status") == "active")
    return {
        "schema_id": LEADERBOARD_POINTER_SCHEMA_ID,
        "leaderboard_id": normalized["leaderboard_id"],
        "generation": int(normalized.get("generation", 0) or 0),
        "snapshot_id": normalized["snapshot_id"],
        "snapshot_ref": str(snapshot_ref),
        "snapshot_sha256": normalized["snapshot_sha256"],
        "published_at": str(published_at),
        "writer": dict(writer or {}),
        "compact_summary": {
            "row_count": len(normalized["rows"]),
            "active_count": active_count,
            "provisional_count": len(normalized["rows"]) - active_count,
            "top_checkpoint_ids": top_checkpoint_ids,
        },
    }


def validate_leaderboard_pointer(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the small live pointer payload."""

    if value.get("schema_id") != LEADERBOARD_POINTER_SCHEMA_ID:
        raise ValueError(f"leaderboard pointer requires schema_id {LEADERBOARD_POINTER_SCHEMA_ID!r}")
    for key in ("leaderboard_id", "snapshot_id", "snapshot_ref", "snapshot_sha256"):
        if not isinstance(value.get(key), str) or not value.get(key):
            raise ValueError(f"leaderboard pointer requires {key}")
    normalized = dict(value)
    normalized["generation"] = int(normalized.get("generation", 0) or 0)
    return normalized


def select_opponent_assignment_from_leaderboard(
    snapshot: Mapping[str, Any],
    *,
    assignment_id: str,
    source_ref: str,
    seed: int = 0,
    max_slots: int = 5,
    include_blank_sentinel: bool = True,
    allow_provisional: bool = False,
    slot_weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a small trainer assignment and audit from a leaderboard snapshot."""

    if not assignment_id:
        raise ValueError("assignment_id is required")
    normalized = validate_leaderboard_snapshot(snapshot)
    if max_slots < 1:
        raise ValueError("max_slots must be positive")
    weights = {**DEFAULT_SLOT_WEIGHTS, **dict(slot_weights or {})}
    rows = _eligible_assignment_rows(
        normalized["rows"],
        allow_provisional=allow_provisional,
    )
    if not rows:
        raise ValueError("no eligible leaderboard rows with immutable checkpoint refs")
    selected: list[tuple[str, dict[str, Any]]] = []
    used_ids: set[str] = set()

    def add_slot(slot_name: str, row: Mapping[str, Any] | None) -> None:
        if row is None:
            return
        checkpoint_id = str(row["checkpoint_id"])
        if checkpoint_id in used_ids:
            return
        selected.append((slot_name, dict(row)))
        used_ids.add(checkpoint_id)

    add_slot("champion", rows[0])
    add_slot("recent_strong", _first_row(rows, used_ids, latest_for_run=True) or _first_row(rows, used_ids))
    add_slot("diverse_challenger", _diverse_row(rows, used_ids, selected) or _first_row(rows, used_ids))
    add_slot("anchor", _anchor_row(rows, used_ids) or _first_row(rows, used_ids))

    selected_for_assignment = selected[: max_slots - (1 if include_blank_sentinel else 0)]
    assignment_entries = [
        _assignment_entry_from_row(slot_name, row, weight=float(weights.get(slot_name, 1.0)))
        for slot_name, row in selected_for_assignment
    ]
    sentinel_added = False
    if include_blank_sentinel and len(assignment_entries) < max_slots:
        assignment_entries.append(
            {
                "name": "slot_sentinel_blank_canvas",
                "weight": float(weights["sentinel"]),
                "tags": ["slot:sentinel", "scripted", "blank"],
                "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
                "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            }
        )
        sentinel_added = True
    if not assignment_entries:
        raise ValueError("assignment selection produced no entries")
    assignment = {
        "schema_id": OPPONENT_ASSIGNMENT_SCHEMA_ID,
        "assignment_id": str(assignment_id),
        "source_epoch": normalized.get("generation"),
        "source_ref": str(source_ref),
        "seed": int(seed),
        "entries": assignment_entries,
    }
    parse_opponent_assignment_snapshot(assignment)
    assignment_sha256 = canonical_assignment_json_sha256(assignment)
    audit = {
        "schema_id": OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID,
        "assignment_id": str(assignment_id),
        "assignment_sha256": assignment_sha256,
        "source_leaderboard": {
            "leaderboard_id": normalized["leaderboard_id"],
            "snapshot_id": normalized["snapshot_id"],
            "generation": normalized.get("generation"),
            "snapshot_ref": str(source_ref),
            "snapshot_sha256": normalized["snapshot_sha256"],
        },
        "selection": {
            "strategy_id": "top_slots_v0",
            "seed": int(seed),
            "max_slots": int(max_slots),
            "include_blank_sentinel": bool(include_blank_sentinel),
            "allow_provisional": bool(allow_provisional),
            "slot_weights": weights,
        },
        "selected_rows": [
            {
                "entry_name": f"slot_{slot_name}",
                "slot": slot_name,
                "checkpoint_id": row.get("checkpoint_id"),
                "leaderboard_rank": row.get("rank"),
                "leaderboard_status": row.get("status"),
                "opponent_checkpoint_ref": row.get("checkpoint_ref"),
            }
            for slot_name, row in selected_for_assignment
        ]
        + (
            [
                {
                    "entry_name": "slot_sentinel_blank_canvas",
                    "slot": "sentinel",
                    "checkpoint_id": None,
                    "leaderboard_rank": None,
                    "leaderboard_status": "scripted_sentinel",
                    "opponent_checkpoint_ref": None,
                }
            ]
            if sentinel_added
            else []
        ),
    }
    return assignment, audit


def validate_assignment_audit(value: Mapping[str, Any], *, assignment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate an assignment audit payload."""

    if value.get("schema_id") != OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID:
        raise ValueError(
            f"assignment audit requires schema_id {OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID!r}"
        )
    assignment_id = value.get("assignment_id")
    if not isinstance(assignment_id, str) or not assignment_id:
        raise ValueError("assignment audit requires assignment_id")
    if assignment is not None:
        expected = canonical_assignment_json_sha256(assignment)
        if str(value.get("assignment_sha256")) != expected:
            raise ValueError("assignment audit hash does not match assignment")
    return dict(value)


def _leaderboard_row_from_rating_row(
    row: Mapping[str, Any],
    *,
    active_min_distinct_opponents: int,
    active_min_valid_games: int,
    max_failure_rate: float,
) -> dict[str, Any]:
    checkpoint_id = _required_str(row, "checkpoint_id")
    checkpoint_ref = _required_str(row, "checkpoint_ref")
    _validate_immutable_checkpoint_ref(checkpoint_ref)
    games = int(row.get("games", 0) or 0)
    failure_count = int(row.get("failure_count", 0) or 0)
    failure_rate = float(failure_count) / float(games) if games else 0.0
    distinct_opponents = int(row.get("distinct_opponents", 0) or 0)
    status = (
        "active"
        if games >= active_min_valid_games
        and distinct_opponents >= active_min_distinct_opponents
        and failure_rate <= max_failure_rate
        else "provisional"
    )
    source = {
        key: row.get(key)
        for key in (
            "run_id",
            "attempt_id",
            "iteration",
            "label",
            "rank",
            "rating",
            "win_rate",
            "wins",
            "losses",
            "draws",
            "games",
            "rated_battles",
            "last_round_delta",
            "last_battle_ref",
        )
        if row.get(key) is not None
    }
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_ref": checkpoint_ref,
        "run_id": row.get("run_id"),
        "attempt_id": row.get("attempt_id"),
        "iteration": _checkpoint_iteration(checkpoint_ref, row.get("iteration")),
        "label": str(row.get("label") or checkpoint_id),
        "rank": int(row.get("rank", 0) or 0),
        "rating": float(row.get("rating", 0.0) or 0.0),
        "status": status,
        "eligibility": {
            "eligible_for_training_default": status == "active",
            "reasons": _eligibility_reasons(
                games=games,
                distinct_opponents=distinct_opponents,
                failure_rate=failure_rate,
                active_min_valid_games=active_min_valid_games,
                active_min_distinct_opponents=active_min_distinct_opponents,
                max_failure_rate=max_failure_rate,
            ),
        },
        "evidence": {
            "valid_games": games,
            "battle_count": int(row.get("battles", row.get("rated_battles", 0)) or 0),
            "rated_battles": int(row.get("rated_battles", 0) or 0),
            "distinct_opponents": distinct_opponents,
            "failure_rate": failure_rate,
            "draw_rate": float(row.get("draws", 0) or 0) / float(games) if games else 0.0,
        },
        "recency": {
            "latest_for_run": bool(row.get("latest_for_run", False)),
            "checkpoint_mtime_ns": row.get("checkpoint_mtime_ns"),
        },
        "source_rating_row": source,
    }


def _validate_leaderboard_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("leaderboard rows must be objects")
    checkpoint_id = _required_str(row, "checkpoint_id")
    checkpoint_ref = _required_str(row, "checkpoint_ref")
    _validate_immutable_checkpoint_ref(checkpoint_ref)
    normalized = dict(row)
    normalized["checkpoint_id"] = checkpoint_id
    normalized["checkpoint_ref"] = checkpoint_ref
    normalized["rank"] = int(normalized.get("rank", 0) or 0)
    normalized["rating"] = float(normalized.get("rating", 0.0) or 0.0)
    status = str(normalized.get("status") or "provisional")
    if status not in {"active", "provisional", "retired"}:
        raise ValueError(f"unsupported leaderboard row status {status!r}")
    normalized["status"] = status
    return normalized


def _assignment_entry_from_row(slot_name: str, row: Mapping[str, Any], *, weight: float) -> dict[str, Any]:
    return {
        "name": f"slot_{slot_name}",
        "weight": float(weight),
        "age_label": str(row.get("label") or slot_name),
        "tags": ["slot:" + slot_name, "leaderboard", str(row.get("status") or "unknown")],
        "opponent_policy_kind": OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
        "opponent_checkpoint_ref": str(row["checkpoint_ref"]),
    }


def _eligible_assignment_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_provisional: bool,
) -> list[dict[str, Any]]:
    allowed_statuses = {"active", "provisional"} if allow_provisional else {"active"}
    eligible = [
        dict(row)
        for row in rows
        if row.get("status") in allowed_statuses and row.get("checkpoint_ref")
    ]
    return sorted(eligible, key=_assignment_row_sort_key)


def _assignment_row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, float, str]:
    rank = int(row.get("rank", 0) or 0)
    has_rank = rank > 0
    return (
        0 if has_rank else 1,
        rank if has_rank else 0,
        -float(row.get("rating", 0.0) or 0.0),
        str(row.get("checkpoint_id") or ""),
    )


def _first_row(
    rows: Sequence[Mapping[str, Any]],
    used_ids: set[str],
    *,
    latest_for_run: bool | None = None,
) -> dict[str, Any] | None:
    for row in rows:
        if str(row["checkpoint_id"]) in used_ids:
            continue
        if latest_for_run is not None and bool(row.get("latest_for_run", False)) != latest_for_run:
            continue
        return dict(row)
    return None


def _diverse_row(
    rows: Sequence[Mapping[str, Any]],
    used_ids: set[str],
    selected: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any] | None:
    selected_run_ids = {str(row.get("run_id")) for _slot, row in selected if row.get("run_id")}
    for row in rows:
        if str(row["checkpoint_id"]) in used_ids:
            continue
        run_id = str(row.get("run_id") or "")
        if run_id and run_id not in selected_run_ids:
            return dict(row)
    return None


def _anchor_row(rows: Sequence[Mapping[str, Any]], used_ids: set[str]) -> dict[str, Any] | None:
    midpoint = len(rows) // 2
    ordered = list(rows[midpoint:]) + list(rows[:midpoint])
    return _first_row(ordered, used_ids)


def _snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    payload = dict(snapshot)
    payload.pop("snapshot_sha256", None)
    return canonical_json_sha256(payload)


def _required_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"leaderboard row requires {key}")
    return value.strip()


def _checkpoint_iteration(checkpoint_ref: str, fallback: Any = None) -> int | None:
    parsed = lightzero_iteration_from_checkpoint_name(Path(checkpoint_ref).name)
    if parsed is not None:
        return int(parsed)
    if fallback is None:
        return None
    try:
        return int(fallback)
    except (TypeError, ValueError):
        return None


def _validate_immutable_checkpoint_ref(checkpoint_ref: str) -> None:
    filename = Path(str(checkpoint_ref)).name
    if lightzero_iteration_from_checkpoint_name(filename) is None:
        raise ValueError(
            "leaderboard checkpoint refs must be immutable exact iteration_N.pth.tar files; "
            f"got {checkpoint_ref!r}"
        )


def _eligibility_reasons(
    *,
    games: int,
    distinct_opponents: int,
    failure_rate: float,
    active_min_valid_games: int,
    active_min_distinct_opponents: int,
    max_failure_rate: float,
) -> list[str]:
    reasons: list[str] = []
    if games < active_min_valid_games:
        reasons.append("insufficient_games")
    if distinct_opponents < active_min_distinct_opponents:
        reasons.append("insufficient_distinct_opponents")
    if failure_rate > max_failure_rate:
        reasons.append("failure_rate_too_high")
    return reasons
