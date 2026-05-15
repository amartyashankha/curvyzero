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
    OPPONENT_DEATH_MODE_IMMORTAL,
    OPPONENT_DEATH_MODE_NORMAL,
    OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
    OPPONENT_POLICY_KIND_FROZEN_LIGHTZERO_CHECKPOINT,
    OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
    OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
    OPPONENT_RUNTIME_MODE_NORMAL,
)
from curvyzero.training.opponent_registry import (
    OPPONENT_ASSIGNMENT_SCHEMA_ID,
    canonical_assignment_json_sha256,
    parse_opponent_assignment_snapshot,
)


LEADERBOARD_SNAPSHOT_SCHEMA_ID = "curvyzero_opponent_leaderboard_snapshot/v0"
LEADERBOARD_POINTER_SCHEMA_ID = "curvyzero_opponent_leaderboard_pointer/v0"
OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID = "curvyzero_opponent_assignment_audit/v0"

MAX_ASSIGNMENT_SLOT_COUNT = 5

STABLE_SLOTS_V1_STRATEGY_ID = "stable_slots_v1"
STABLE_SLOT_PROFILE_3 = "stable_3"
STABLE_SLOT_PROFILE_5 = "stable_5"
STABLE_SLOT_PROFILES = {STABLE_SLOT_PROFILE_3, STABLE_SLOT_PROFILE_5}
STABLE_SENTINEL_NONE = "none"
STABLE_SENTINEL_BLANK_CANVAS = "blank_canvas"
STABLE_SENTINEL_WALL_AVOIDANT_IMMORTAL = "wall_avoidant_immortal"
STABLE_SENTINELS = {
    STABLE_SENTINEL_NONE,
    STABLE_SENTINEL_BLANK_CANVAS,
    STABLE_SENTINEL_WALL_AVOIDANT_IMMORTAL,
}

DEFAULT_ACTIVE_MIN_DISTINCT_OPPONENTS = 20
DEFAULT_ACTIVE_MIN_VALID_GAMES = 300
DEFAULT_MAX_FAILURE_RATE = 0.02
DEFAULT_MAX_ACTIVE_RANK = 100

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


def validate_rating_snapshot_source(
    rating_snapshot: Mapping[str, Any],
    *,
    expected_round_id: str | None = None,
    expected_round_index: int | None = None,
    expected_rating_context_hash: str | None = None,
    expected_roster_hash: str | None = None,
    expected_rating_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate that a rating snapshot is the exact source the caller expects."""

    actual_round_id = str(rating_snapshot.get("round_id") or "")
    if expected_round_id and actual_round_id != str(expected_round_id):
        raise ValueError(
            "rating snapshot round_id mismatch: "
            f"expected {expected_round_id!r}, got {actual_round_id!r}"
        )
    actual_round_index_raw = rating_snapshot.get("round_index")
    actual_round_index = (
        int(actual_round_index_raw) if actual_round_index_raw is not None else None
    )
    if (
        expected_round_index is not None
        and actual_round_index != int(expected_round_index)
    ):
        raise ValueError(
            "rating snapshot round_index mismatch: "
            f"expected {int(expected_round_index)}, got {actual_round_index!r}"
        )
    actual_context_hash = str(rating_snapshot.get("context_hash") or "")
    if (
        expected_rating_context_hash
        and actual_context_hash != str(expected_rating_context_hash)
    ):
        raise ValueError(
            "rating snapshot context_hash mismatch: "
            f"expected {expected_rating_context_hash!r}, "
            f"got {actual_context_hash!r}"
        )
    actual_roster_hash = str(
        rating_snapshot.get("roster_hash") or rating_snapshot.get("pool_hash") or ""
    )
    if expected_roster_hash and actual_roster_hash != str(expected_roster_hash):
        raise ValueError(
            "rating snapshot roster_hash mismatch: "
            f"expected {expected_roster_hash!r}, got {actual_roster_hash!r}"
        )
    actual_sha256 = canonical_json_sha256(rating_snapshot)
    if (
        expected_rating_snapshot_sha256
        and actual_sha256 != str(expected_rating_snapshot_sha256)
    ):
        raise ValueError(
            "rating snapshot sha256 mismatch: "
            f"expected {expected_rating_snapshot_sha256!r}, got {actual_sha256!r}"
        )
    return {
        "round_id": actual_round_id,
        "round_index": actual_round_index,
        "rating_context_hash": actual_context_hash,
        "roster_hash": actual_roster_hash,
        "rating_snapshot_sha256": actual_sha256,
    }


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
    max_active_rank: int = DEFAULT_MAX_ACTIVE_RANK,
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
            max_active_rank=max_active_rank,
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
            "max_active_rank": int(max_active_rank),
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
    provisional_count = sum(
        1 for row in normalized["rows"] if row.get("status") == "provisional"
    )
    retired_count = sum(1 for row in normalized["rows"] if row.get("status") == "retired")
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
            "provisional_count": provisional_count,
            "retired_count": retired_count,
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
                "opponent_immortal": True,
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


def select_stable_slots_v1_assignment(
    snapshot: Mapping[str, Any],
    *,
    assignment_id: str,
    source_ref: str,
    seed: int = 0,
    profile: str = STABLE_SLOT_PROFILE_3,
    sentinel: str = STABLE_SENTINEL_BLANK_CANVAS,
    allow_recent_provisional: bool = False,
    checkpoint_death_mode: str = OPPONENT_DEATH_MODE_NORMAL,
    expected_rating_context_hash: str | None = None,
    slot_weights: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize stable opponent slots into a trainer assignment.

    This is deliberately not a live slot system. It reads one immutable
    leaderboard snapshot and returns one immutable assignment plus audit. The
    trainer only consumes the assignment JSON.
    """

    if not assignment_id:
        raise ValueError("assignment_id is required")
    normalized = validate_leaderboard_snapshot(snapshot)
    profile = _validate_stable_profile(profile)
    sentinel = _validate_stable_sentinel(sentinel)
    checkpoint_death_mode = _validate_checkpoint_death_mode(checkpoint_death_mode)
    _validate_expected_rating_context_hash(
        expected_rating_context_hash,
        snapshot=normalized,
    )
    weights = {**DEFAULT_SLOT_WEIGHTS, **dict(slot_weights or {})}
    slot_names = _stable_slot_names(profile=profile, sentinel=sentinel)
    active_rows = _eligible_assignment_rows(
        normalized["rows"],
        allow_provisional=False,
    )
    recent_rows = _eligible_assignment_rows(
        normalized["rows"],
        allow_provisional=allow_recent_provisional,
    )
    if not active_rows:
        raise ValueError("stable_slots_v1 requires at least one active checkpoint row")

    assignment_entries: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    hardcoded_slots: list[dict[str, Any]] = []
    slot_plan: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_refs: set[str] = set()

    def add_checkpoint_slot(slot_name: str, row: Mapping[str, Any], *, reason: str) -> None:
        checkpoint_id = str(row["checkpoint_id"])
        checkpoint_ref = str(row["checkpoint_ref"])
        if checkpoint_id in used_ids or checkpoint_ref in used_refs:
            raise ValueError(
                f"stable_slots_v1 selected duplicate checkpoint for {slot_name!r}"
            )
        used_ids.add(checkpoint_id)
        used_refs.add(checkpoint_ref)
        entry = _assignment_entry_from_row(
            slot_name,
            row,
            weight=float(weights.get(slot_name, 1.0)),
        )
        entry["opponent_immortal"] = (
            checkpoint_death_mode == OPPONENT_DEATH_MODE_IMMORTAL
        )
        entry["tags"] = [
            *entry.get("tags", []),
            "strategy:stable_slots_v1",
            (
                "checkpoint_death:immortal"
                if checkpoint_death_mode == OPPONENT_DEATH_MODE_IMMORTAL
                else "checkpoint_death:normal"
            ),
        ]
        assignment_entries.append(entry)
        evidence = {
            "source": "leaderboard",
            "entry_name": entry["name"],
            "slot": slot_name,
            "checkpoint_id": row.get("checkpoint_id"),
            "run_id": row.get("run_id"),
            "leaderboard_rank": row.get("rank"),
            "leaderboard_status": row.get("status"),
            "opponent_checkpoint_ref": row.get("checkpoint_ref"),
            "selection_reason": reason,
            "checkpoint_death_mode": checkpoint_death_mode,
            "recency": row.get("recency", {}),
            "evidence": row.get("evidence", {}),
        }
        selected_rows.append(evidence)
        slot_plan.append(evidence)

    for slot_name in slot_names:
        if slot_name == "sentinel":
            entry, evidence = _stable_sentinel_entry(
                sentinel,
                weight=float(weights.get("sentinel", 1.0)),
            )
            assignment_entries.append(entry)
            hardcoded_slots.append(evidence)
            slot_plan.append(evidence)
            continue

        if slot_name == "champion":
            row = active_rows[0]
            reason = "top_active"
        elif slot_name == "recent_strong":
            row = _first_stable_row(
                recent_rows,
                used_ids,
                used_refs,
                latest_for_run=True,
            )
            if row is None:
                row = _first_stable_row(active_rows, used_ids, used_refs)
            reason = (
                "latest_for_run"
                if row is not None and _row_latest_for_run(row)
                else "best_remaining"
            )
        elif slot_name == "diverse_challenger":
            row = _diverse_stable_row(active_rows, used_ids, used_refs, selected_rows)
            reason = "different_run" if row is not None else "best_remaining"
            if row is None:
                row = _first_stable_row(active_rows, used_ids, used_refs)
        elif slot_name == "anchor":
            row = _anchor_stable_row(active_rows, used_ids, used_refs)
            reason = "middle_ranked_active"
        else:
            raise RuntimeError(f"unknown stable slot {slot_name!r}")

        if row is None:
            raise ValueError(
                f"stable_slots_v1 could not fill required slot {slot_name!r}; "
                f"profile={profile!r}, sentinel={sentinel!r}"
            )
        add_checkpoint_slot(slot_name, row, reason=reason)

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
            "strategy_id": STABLE_SLOTS_V1_STRATEGY_ID,
            "profile": profile,
            "seed": int(seed),
            "sentinel": sentinel,
            "allow_recent_provisional": bool(allow_recent_provisional),
            "checkpoint_death_mode": checkpoint_death_mode,
            "slot_weights": weights,
            "slot_names": slot_names,
            "context_gate": {
                "actual_leaderboard_id": normalized["leaderboard_id"],
                "expected_rating_context_hash": expected_rating_context_hash,
                "actual_rating_context_hash": normalized.get("context", {}).get(
                    "rating_context_hash"
                ),
            },
        },
        "selected_rows": selected_rows,
        "hardcoded_slots": hardcoded_slots,
        "slot_plan": slot_plan,
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
        assignment_payload = dict(assignment)
        if str(assignment_payload.get("assignment_id")) != assignment_id:
            raise ValueError("assignment audit assignment_id does not match assignment")
    selection = value.get("selection")
    strategy_id = selection.get("strategy_id") if isinstance(selection, Mapping) else None
    source_leaderboard = value.get("source_leaderboard")
    if source_leaderboard is not None:
        if not isinstance(source_leaderboard, Mapping):
            raise ValueError("assignment audit source_leaderboard must be an object")
        for key in ("leaderboard_id", "snapshot_id", "snapshot_ref", "snapshot_sha256"):
            if not isinstance(source_leaderboard.get(key), str) or not source_leaderboard.get(key):
                raise ValueError(f"assignment audit source_leaderboard requires {key}")
    if strategy_id == STABLE_SLOTS_V1_STRATEGY_ID:
        if not isinstance(source_leaderboard, Mapping):
            raise ValueError("stable_slots_v1 audit requires source_leaderboard")
        if not isinstance(selection, Mapping):
            raise ValueError("stable_slots_v1 audit requires selection")
        slot_plan = value.get("slot_plan")
        if not isinstance(slot_plan, list) or not slot_plan:
            raise ValueError("stable_slots_v1 audit requires non-empty slot_plan")
        for key in ("profile", "sentinel", "checkpoint_death_mode"):
            if not isinstance(selection.get(key), str) or not selection.get(key):
                raise ValueError(f"stable_slots_v1 audit selection requires {key}")
        if assignment is not None:
            entries = assignment_payload.get("entries")
            if isinstance(entries, list) and len(slot_plan) != len(entries):
                raise ValueError("stable_slots_v1 audit slot_plan length does not match assignment")
    return dict(value)


def _leaderboard_row_from_rating_row(
    row: Mapping[str, Any],
    *,
    active_min_distinct_opponents: int,
    active_min_valid_games: int,
    max_failure_rate: float,
    max_active_rank: int,
) -> dict[str, Any]:
    checkpoint_id = _required_str(row, "checkpoint_id")
    checkpoint_ref = _required_str(row, "checkpoint_ref")
    _validate_immutable_checkpoint_ref(checkpoint_ref)
    games = int(row.get("games", 0) or 0)
    failure_count = int(row.get("failure_count", 0) or 0)
    failure_rate = float(failure_count) / float(games) if games else 0.0
    distinct_opponents = int(row.get("distinct_opponents", 0) or 0)
    rank = int(row.get("rank", 0) or 0)
    source_status = str(row.get("status") or "")
    mature_enough = (
        games >= active_min_valid_games
        and distinct_opponents >= active_min_distinct_opponents
        and failure_rate <= max_failure_rate
    )
    if source_status == "retired" or (
        mature_enough and rank > 0 and rank > int(max_active_rank)
    ):
        status = "retired"
    else:
        status = (
            "active"
            if mature_enough
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
        "rank": rank,
        "rating": float(row.get("rating", 0.0) or 0.0),
        "status": status,
        "eligibility": {
            "eligible_for_training_default": status == "active",
            "reasons": _eligibility_reasons(
                games=games,
                distinct_opponents=distinct_opponents,
                failure_rate=failure_rate,
                rank=rank,
                active_min_valid_games=active_min_valid_games,
                active_min_distinct_opponents=active_min_distinct_opponents,
                max_failure_rate=max_failure_rate,
                max_active_rank=max_active_rank,
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


def _validate_stable_profile(profile: str) -> str:
    normalized = str(profile or "").strip()
    if normalized not in STABLE_SLOT_PROFILES:
        raise ValueError(
            f"stable_slots_v1 profile must be one of {sorted(STABLE_SLOT_PROFILES)!r}"
        )
    return normalized


def _validate_stable_sentinel(sentinel: str) -> str:
    normalized = str(sentinel or "").strip()
    if normalized not in STABLE_SENTINELS:
        raise ValueError(
            f"stable_slots_v1 sentinel must be one of {sorted(STABLE_SENTINELS)!r}"
        )
    return normalized


def _validate_checkpoint_death_mode(checkpoint_death_mode: str) -> str:
    normalized = str(checkpoint_death_mode or OPPONENT_DEATH_MODE_NORMAL).strip()
    if normalized not in {OPPONENT_DEATH_MODE_NORMAL, OPPONENT_DEATH_MODE_IMMORTAL}:
        raise ValueError(
            "stable_slots_v1 checkpoint_death_mode must be "
            f"{OPPONENT_DEATH_MODE_NORMAL!r} or {OPPONENT_DEATH_MODE_IMMORTAL!r}"
        )
    return normalized


def _validate_expected_rating_context_hash(
    expected_rating_context_hash: str | None,
    *,
    snapshot: Mapping[str, Any],
) -> None:
    if expected_rating_context_hash is None:
        return
    actual_context_hash = snapshot.get("context", {}).get("rating_context_hash")
    if str(expected_rating_context_hash) != str(actual_context_hash):
        raise ValueError(
            "stable_slots_v1 expected rating_context_hash "
            f"{expected_rating_context_hash!r}; got {actual_context_hash!r}"
        )


def _stable_slot_names(*, profile: str, sentinel: str) -> list[str]:
    if profile == STABLE_SLOT_PROFILE_3:
        if sentinel == STABLE_SENTINEL_NONE:
            return ["champion", "recent_strong", "diverse_challenger"]
        return ["champion", "recent_strong", "sentinel"]
    if profile == STABLE_SLOT_PROFILE_5:
        if sentinel == STABLE_SENTINEL_NONE:
            raise ValueError("stable_5 requires a sentinel; use stable_3 for checkpoint-only slots")
        slots = ["champion", "recent_strong", "diverse_challenger", "anchor"]
        slots.append("sentinel")
        return slots
    raise RuntimeError(f"unknown stable slot profile {profile!r}")


def _stable_sentinel_entry(
    sentinel: str,
    *,
    weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sentinel == STABLE_SENTINEL_BLANK_CANVAS:
        entry = {
            "name": "slot_sentinel_blank_canvas",
            "weight": float(weight),
            "tags": ["slot:sentinel", "scripted", "blank", "strategy:stable_slots_v1"],
            "opponent_policy_kind": OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_BLANK_CANVAS_NOOP,
            "opponent_immortal": True,
        }
        return entry, {
            "source": "hardcoded",
            "entry_name": entry["name"],
            "slot": "sentinel",
            "slot_kind": STABLE_SENTINEL_BLANK_CANVAS,
            "leaderboard_status": "hardcoded_blank_canvas",
            "selection_reason": "stable_sentinel",
        }
    if sentinel == STABLE_SENTINEL_WALL_AVOIDANT_IMMORTAL:
        entry = {
            "name": "slot_sentinel_wall_avoidant_immortal",
            "weight": float(weight),
            "tags": [
                "slot:sentinel",
                "scripted",
                "immortal",
                "strategy:stable_slots_v1",
            ],
            "opponent_policy_kind": OPPONENT_POLICY_KIND_PROACTIVE_WALL_AVOIDANT,
            "opponent_runtime_mode": OPPONENT_RUNTIME_MODE_NORMAL,
            "opponent_immortal": True,
        }
        return entry, {
            "source": "hardcoded",
            "entry_name": entry["name"],
            "slot": "sentinel",
            "slot_kind": STABLE_SENTINEL_WALL_AVOIDANT_IMMORTAL,
            "leaderboard_status": "hardcoded_wall_avoidant_immortal",
            "selection_reason": "stable_sentinel",
        }
    raise ValueError(f"stable_slots_v1 cannot build sentinel {sentinel!r}")


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
        if latest_for_run is not None and _row_latest_for_run(row) != latest_for_run:
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


def _first_stable_row(
    rows: Sequence[Mapping[str, Any]],
    used_ids: set[str],
    used_refs: set[str],
    *,
    latest_for_run: bool | None = None,
) -> dict[str, Any] | None:
    for row in rows:
        if str(row["checkpoint_id"]) in used_ids:
            continue
        if str(row["checkpoint_ref"]) in used_refs:
            continue
        if latest_for_run is not None and _row_latest_for_run(row) != latest_for_run:
            continue
        return dict(row)
    return None


def _diverse_stable_row(
    rows: Sequence[Mapping[str, Any]],
    used_ids: set[str],
    used_refs: set[str],
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    selected_run_ids = {
        str(row.get("run_id"))
        for row in selected_rows
        if row.get("run_id")
    }
    for row in rows:
        if str(row["checkpoint_id"]) in used_ids:
            continue
        if str(row["checkpoint_ref"]) in used_refs:
            continue
        run_id = str(row.get("run_id") or "")
        if run_id and run_id not in selected_run_ids:
            return dict(row)
    return None


def _anchor_stable_row(
    rows: Sequence[Mapping[str, Any]],
    used_ids: set[str],
    used_refs: set[str],
) -> dict[str, Any] | None:
    midpoint = len(rows) // 2
    ordered = list(rows[midpoint:]) + list(rows[:midpoint])
    return _first_stable_row(ordered, used_ids, used_refs)


def _row_latest_for_run(row: Mapping[str, Any]) -> bool:
    recency = row.get("recency")
    if isinstance(recency, Mapping) and "latest_for_run" in recency:
        return bool(recency.get("latest_for_run"))
    return bool(row.get("latest_for_run", False))


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
    ref_text = str(checkpoint_ref)
    if ref_text.startswith(("runs:", "control:")):
        ref_text = ref_text.split(":", 1)[1]
    filename = Path(ref_text).name
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
    rank: int,
    active_min_valid_games: int,
    active_min_distinct_opponents: int,
    max_failure_rate: float,
    max_active_rank: int,
) -> list[str]:
    reasons: list[str] = []
    if rank > 0 and rank > max_active_rank:
        reasons.append("below_active_rank_limit")
    if games < active_min_valid_games:
        reasons.append("insufficient_games")
    if distinct_opponents < active_min_distinct_opponents:
        reasons.append("insufficient_distinct_opponents")
    if failure_rate > max_failure_rate:
        reasons.append("failure_rate_too_high")
    return reasons
