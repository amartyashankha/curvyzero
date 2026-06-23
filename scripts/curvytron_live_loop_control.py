#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import modal

from curvyzero.infra.modal.curvyzero_checkpoint_tournament_settings import (
    CHECKPOINT_INTAKE_DICT_NAME,
    CURRENT_RATING_RUN_ID,
    CURRENT_TOURNAMENT_ID,
    TRAINING_CANDIDATE_REFRESH_CONFIG_REF,
)


DEFAULT_APP_NAME = "curvyzero-checkpoint-tournament-v2"
DEFAULT_TOURNAMENT_ID = CURRENT_TOURNAMENT_ID
DEFAULT_RATING_RUN_ID = CURRENT_RATING_RUN_ID
DEFAULT_RUN_STATUS_APP_NAME = "curvyzero-lightzero-curvytron-run-status"
DEFAULT_RUN_MANIFEST = (
    "artifacts/local/curvytron_next_batch_manifests/"
    "cz26-full-20260517a/cz26-full-20260517a.json"
)
TOOL_SCHEMA_ID = "curvyzero_live_loop_control_tool/v0"
DEFAULT_DRAIN_REQUEST_STALE_AFTER_SECONDS = 10 * 60
DEFAULT_ASSIGNMENT_PROOF_TAIL_BYTES = 16 * 1024 * 1024
DEFAULT_ASSIGNMENT_PROOF_CHUNK_SIZE = 16
DEFAULT_ASSIGNMENT_PROOF_ROW_LIMIT = 0


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _age_seconds(*, checked_at: Any, updated_at: Any) -> float | None:
    checked = _parse_timestamp(checked_at)
    updated = _parse_timestamp(updated_at)
    if checked is None or updated is None:
        return None
    return max(0.0, (checked - updated).total_seconds())


def _control_decision(status: Mapping[str, Any], *, action: str) -> dict[str, Any]:
    normalized_action = str(action or "status").strip().lower()
    if normalized_action not in {"status", "drain-if-ready", "drain"}:
        raise ValueError("action must be status, drain-if-ready, or drain")
    flags = [str(flag) for flag in status.get("flags") or []]
    intake = status.get("intake") if isinstance(status.get("intake"), Mapping) else {}
    desired_new = _int(intake.get("new_checkpoints_not_in_latest_rating"))
    queue_len = intake.get("queue_len")
    current_batch = status.get("current_game_batch")
    recovery_scan_reason = "spawn_active_game_batch_recovery_scan"
    if normalized_action == "status":
        reason = "status_only"
        spawn = False
    elif "bad_live_all_pairs_config" in flags:
        reason = "blocked_bad_live_all_pairs_config"
        spawn = False
    elif current_batch and (
        "active_game_batch_output_stale" in flags
        or "active_game_batch_not_covering_new_checkpoints" in flags
        or "active_game_batch_partial_reduce_due" in flags
    ):
        reason = recovery_scan_reason
        spawn = True
    elif current_batch:
        reason = "blocked_active_game_batch"
        spawn = False
    elif desired_new <= 0 and not queue_len:
        reason = "nothing_new_to_rate"
        spawn = False
    elif normalized_action == "drain-if-ready" and str(status.get("status") or "") not in {
        "queued_waiting_for_drain",
        "ready_for_next_rating_batch",
    }:
        reason = "status_not_ready_for_drain"
        spawn = False
    else:
        reason = "spawn_next_bounded_rating_batch"
        spawn = True
    return {
        "action": normalized_action,
        "spawn_drain": spawn,
        "reason": reason,
        "status": status.get("status"),
        "new_checkpoints_not_in_latest_rating": desired_new,
        "queue_len": queue_len,
    }


def _decision_is_recovery_scan(decision: Mapping[str, Any]) -> bool:
    return str(decision.get("reason") or "") == "spawn_active_game_batch_recovery_scan"


def _function_call_id(call: object) -> str:
    return str(
        getattr(call, "object_id", None)
        or getattr(call, "id", None)
        or getattr(call, "function_call_id", None)
        or ""
    )


def _drain_request_key(tournament_id: str, rating_run_id: str) -> str:
    return f"live_loop_control:drain_request:{tournament_id}:{rating_run_id}:active"


def _drain_request_summary(request: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(request, Mapping):
        return None
    return {
        "function_call_id": request.get("function_call_id"),
        "requested_at": request.get("requested_at"),
        "stale_after_seconds": request.get("stale_after_seconds"),
        "status_at_request": request.get("status_at_request"),
        "queue_len_at_request": request.get("queue_len_at_request"),
        "new_checkpoints_at_request": request.get("new_checkpoints_at_request"),
        "drain_finished_at": request.get("drain_finished_at"),
        "drain_status": request.get("drain_status"),
        "drain_spawn_skipped_reason": request.get("drain_spawn_skipped_reason"),
    }


def _drain_request_is_fresh(
    request: Mapping[str, Any] | None,
    *,
    stale_after_seconds: int,
) -> bool:
    if not isinstance(request, Mapping):
        return False
    spawned_rating_call = bool(str(request.get("function_call_id") or "").strip()) and (
        request.get("drain_status") == "drain_returned"
    ) and not str(request.get("drain_spawn_skipped_reason") or "").strip()
    if request.get("drain_finished_at") and not spawned_rating_call:
        return False
    if request.get("drain_status") == "drain_returned" and not spawned_rating_call:
        return False
    requested_at = _parse_timestamp(request.get("requested_at"))
    if requested_at is None:
        return False
    age = (datetime.now(UTC) - requested_at).total_seconds()
    return age < max(0, int(stale_after_seconds))


def _drain_result_summary(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, Mapping):
        return None
    recovery_round = result.get("rating_recovery_round")
    recovery_skip = (
        result.get("rating_recovery_skip_decision")
        if isinstance(result.get("rating_recovery_skip_decision"), Mapping)
        else {}
    )
    return {
        "event_count": result.get("event_count"),
        "queue_len_before": result.get("queue_len_before"),
        "queue_len_after_repair": result.get("queue_len_after_repair"),
        "rating_checkpoint_count": result.get("rating_checkpoint_count"),
        "desired_rating_spec_checkpoint_count": result.get(
            "desired_rating_spec_checkpoint_count"
        ),
        "desired_rating_spec_pool_hash": result.get("desired_rating_spec_pool_hash"),
        "latest_rating_checkpoint_count": result.get("latest_rating_checkpoint_count"),
        "desired_pool_new_checkpoint_count": result.get(
            "desired_pool_new_checkpoint_count"
        ),
        "rating_writer_finished": result.get("rating_writer_finished"),
        "rating_claimed": result.get("rating_claimed"),
        "rating_claim_stale": result.get("rating_claim_stale"),
        "rating_claim_repaired": result.get("rating_claim_repaired"),
        "rating_call_id": result.get("rating_call_id"),
        "spawn_skipped_reason": result.get("spawn_skipped_reason"),
        "rating_recovery_round": {
            "round_id": recovery_round.get("round_id"),
            "round_index": recovery_round.get("round_index"),
            "status": recovery_round.get("status"),
        }
        if isinstance(recovery_round, Mapping)
        else None,
        "rating_recovery_claimed": result.get("rating_recovery_claimed"),
        "rating_recovery_reduce_ready": result.get("rating_recovery_reduce_ready"),
        "rating_recovery_partial_reduce_recommended": result.get(
            "rating_recovery_partial_reduce_recommended"
        ),
        "rating_recovery_skip_decision": {
            "skip": bool(recovery_skip.get("skip")),
            "reason": recovery_skip.get("reason"),
            "input_checkpoint_count": recovery_skip.get("input_checkpoint_count"),
            "desired_checkpoint_count": recovery_skip.get("desired_checkpoint_count"),
            "pair_count": recovery_skip.get("pair_count"),
            "game_count": recovery_skip.get("game_count"),
            "completed_game_count": recovery_skip.get("completed_game_count"),
            "started_pair_count": recovery_skip.get("started_pair_count"),
            "stale_after_seconds": recovery_skip.get("stale_after_seconds"),
            "input_age_seconds": recovery_skip.get("input_age_seconds"),
            "stale_age_seconds": recovery_skip.get("stale_age_seconds"),
            "latest_result_ts": recovery_skip.get("latest_result_ts"),
            "is_stale": bool(recovery_skip.get("is_stale")),
            "scan_output_progress": bool(recovery_skip.get("scan_output_progress")),
            "progress_scan_mode": recovery_skip.get("progress_scan_mode"),
            "progress_count_semantics": recovery_skip.get("progress_count_semantics"),
            "progress_scan_error": recovery_skip.get("progress_scan_error"),
            "partial_reduce_recommended": bool(
                recovery_skip.get("partial_reduce_recommended")
            ),
        }
        if recovery_skip
        else None,
    }


def _input_info_summary(item: Any) -> dict[str, Any]:
    status = getattr(item, "status", None)
    children = getattr(item, "children", None) or []
    return {
        "input_id": getattr(item, "input_id", None),
        "function_call_id": getattr(item, "function_call_id", None),
        "task_id": getattr(item, "task_id", None),
        "status": getattr(status, "name", str(status)),
        "function_name": getattr(item, "function_name", None),
        "module_name": getattr(item, "module_name", None),
        "children": [
            _input_info_summary(child) if not isinstance(child, str) else child
            for child in children
        ],
    }


def _flatten_input_info(items: list[Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        status = getattr(item, "status", None)
        children = getattr(item, "children", None) or []
        flat.append(
            {
                "input_id": getattr(item, "input_id", None),
                "function_call_id": getattr(item, "function_call_id", None),
                "task_id": getattr(item, "task_id", None),
                "status": getattr(status, "name", str(status)),
                "function_name": getattr(item, "function_name", None),
                "module_name": getattr(item, "module_name", None),
                "child_count": len(children),
            }
        )
        for child in children:
            if not isinstance(child, str):
                visit(child)

    for root in items:
        if not isinstance(root, str):
            visit(root)
    return flat


def _call_graph_summary(
    items: list[Any],
    *,
    sample_limit: int,
) -> dict[str, Any]:
    flat = _flatten_input_info(items)
    status_counts: Counter[str] = Counter(str(item.get("status") or "") for item in flat)
    function_status_counts: Counter[str] = Counter(
        f"{item.get('function_name') or 'unknown'}:{item.get('status') or 'unknown'}"
        for item in flat
    )
    sample = [
        {
            "input_id": item.get("input_id"),
            "function_call_id": item.get("function_call_id"),
            "function_name": item.get("function_name"),
            "status": item.get("status"),
            "child_count": item.get("child_count"),
        }
        for item in flat[: max(0, int(sample_limit))]
    ]
    return {
        "node_count": len(flat),
        "status_counts": dict(sorted(status_counts.items())),
        "function_status_counts": dict(sorted(function_status_counts.items())),
        "sample_limit": max(0, int(sample_limit)),
        "sample": sample,
    }


def _compact_call_result(value: Any) -> Any:
    if not isinstance(value, Mapping):
        if isinstance(value, list):
            return {"type": "list", "count": len(value)}
        return value
    compact: dict[str, Any] = {}
    omitted: dict[str, int] = {}
    for key, item in value.items():
        if key == "events" and isinstance(item, list):
            omitted[key] = len(item)
            continue
        if key == "previous_pointer_states" and isinstance(item, list):
            omitted[key] = len(item)
            continue
        if key == "rewritten_pointers" and isinstance(item, list):
            omitted[key] = len(item)
            compact["rewritten_pointer_count_from_rows"] = len(item)
            continue
        compact[key] = item
    if omitted:
        compact["omitted_large_fields"] = omitted
    return compact


def _function_call_probe(
    function_call_id: Any,
    *,
    include_full_call_graph: bool,
    call_graph_sample_limit: int,
) -> dict[str, Any] | None:
    call_id = str(function_call_id or "").strip()
    if not call_id:
        return None
    out: dict[str, Any] = {"function_call_id": call_id}
    try:
        call = modal.FunctionCall.from_id(call_id)
    except Exception as exc:
        out["lookup_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return out
    try:
        out["result"] = _compact_call_result(call.get(timeout=0))
        out["state"] = "returned"
    except TimeoutError:
        out["state"] = "pending"
    except Exception as exc:
        out["state"] = "failed"
        out["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    try:
        call_graph = list(call.get_call_graph())
        out["call_graph_summary"] = _call_graph_summary(
            call_graph,
            sample_limit=int(call_graph_sample_limit),
        )
        if include_full_call_graph:
            out["call_graph"] = [_input_info_summary(item) for item in call_graph]
    except Exception as exc:
        out["call_graph_error"] = {"type": type(exc).__name__, "message": str(exc)}
    return out


def _probe_has_pending_work(probe: Mapping[str, Any] | None) -> bool:
    if not isinstance(probe, Mapping):
        return False
    if str(probe.get("state") or "") == "pending":
        return True
    summary = (
        probe.get("call_graph_summary")
        if isinstance(probe.get("call_graph_summary"), Mapping)
        else {}
    )
    status_counts = (
        summary.get("status_counts")
        if isinstance(summary.get("status_counts"), Mapping)
        else {}
    )
    try:
        return int(status_counts.get("PENDING") or 0) > 0
    except (TypeError, ValueError):
        return False


def _status_summary(status: Mapping[str, Any]) -> dict[str, Any]:
    intake = status.get("intake") if isinstance(status.get("intake"), Mapping) else {}
    latest = (
        status.get("latest_rating")
        if isinstance(status.get("latest_rating"), Mapping)
        else {}
    )
    batch = (
        status.get("current_game_batch")
        if isinstance(status.get("current_game_batch"), Mapping)
        else {}
    )
    probe = (
        batch.get("recovery_probe")
        if isinstance(batch.get("recovery_probe"), Mapping)
        else {}
    )
    progress_probe = (
        batch.get("progress_probe")
        if isinstance(batch.get("progress_probe"), Mapping)
        else {}
    )
    refresh = (
        status.get("trainer_refresh")
        if isinstance(status.get("trainer_refresh"), Mapping)
        else {}
    )
    checked_at = status.get("checked_at")
    batch_updated_at = batch.get("updated_at") if batch else None
    intake_checkpoint_count = _int(intake.get("checkpoint_count"))
    latest_checkpoint_count = _int(latest.get("checkpoint_count"))
    active_batch_checkpoint_count = _int(batch.get("checkpoint_count")) if batch else 0
    new_checkpoint_count = _int(intake.get("new_checkpoints_not_in_latest_rating"))
    pool_status = {
        "intake_checkpoint_count": intake_checkpoint_count,
        "latest_rating_checkpoint_count": latest_checkpoint_count,
        "active_game_batch_checkpoint_count": active_batch_checkpoint_count,
        "new_checkpoints_not_in_latest_rating": new_checkpoint_count,
        "active_batch_newer_than_latest": bool(
            batch and active_batch_checkpoint_count > latest_checkpoint_count
        ),
        "active_batch_not_covering_new_checkpoints": bool(
            batch
            and new_checkpoint_count > 0
            and latest_checkpoint_count > 0
            and active_batch_checkpoint_count <= latest_checkpoint_count
        ),
        "active_batch_missing_from_intake_count": max(
            0,
            intake_checkpoint_count - active_batch_checkpoint_count,
        )
        if batch
        else None,
    }
    refresh_rating_source = (
        refresh.get("rating_source")
        if isinstance(refresh.get("rating_source"), Mapping)
        else {}
    )
    latest_round_index = _int(latest.get("round_index"), -1)
    refresh_round_index = _int(refresh_rating_source.get("round_index"), -1)
    proof_chain = {
        "intake": {
            "checkpoint_count": intake_checkpoint_count,
            "queue_len": intake.get("queue_len"),
            "new_checkpoints_not_in_latest_rating": new_checkpoint_count,
        },
        "game_batch": {
            "game_batch_id": batch.get("round_id") if batch else None,
            "game_batch_index": batch.get("round_index") if batch else None,
            "status": batch.get("status") if batch else None,
            "phase": batch.get("phase") if batch else None,
            "checkpoint_count": active_batch_checkpoint_count if batch else 0,
            "pair_count": batch.get("pair_count") if batch else None,
            "game_count": batch.get("game_count") if batch else None,
            "root_completed_game_count": (
                batch.get("completed_game_count") if batch else None
            ),
            "liveness_probe_completed_summary_count": (
                probe.get("completed_game_count") if batch else None
            ),
            "liveness_probe_count_semantics": (
                probe.get("count_semantics") if batch else None
            ),
            "progress_probe_completed_game_count": (
                progress_probe.get("completed_game_count") if batch else None
            ),
            "progress_probe_completion_fraction": (
                progress_probe.get("completion_fraction") if batch else None
            ),
            "progress_probe_count_basis": (
                progress_probe.get("count_basis") if batch else None
            ),
            "covers_current_intake": bool(
                batch and active_batch_checkpoint_count >= intake_checkpoint_count
            ),
            "covers_newer_than_latest_rating": bool(
                batch and active_batch_checkpoint_count > latest_checkpoint_count
            ),
        },
        "latest_rating": {
            "game_batch_id": latest.get("round_id"),
            "game_batch_index": latest.get("round_index"),
            "checkpoint_count": latest_checkpoint_count,
            "rating_count": latest.get("rating_count"),
            "active_count": latest.get("active_count"),
        },
        "trainer_export": {
            "generation": refresh.get("generation"),
            "snapshot_id": refresh.get("snapshot_id"),
            "source_game_batch_id": refresh_rating_source.get("round_id"),
            "source_game_batch_index": refresh_rating_source.get("round_index"),
            "active_count": refresh.get("active_count"),
            "rewritten_pointer_count": refresh.get("rewritten_pointer_count"),
            "assignment_sha_count": len(refresh.get("assignment_sha256s") or []),
            "current_with_latest_rating": bool(
                latest_round_index >= 0 and latest_round_index <= refresh_round_index
            ),
        },
        "trainer_consumption": {
            "status": "unknown_run_trainer_proof",
            "follow_up_command": (
                "uv run --extra modal python scripts/curvytron_live_loop_control.py "
                "--action trainer-proof --activity-probe-pairs 0 --run-limit 0"
            ),
        },
    }
    blockers: list[str] = []
    open_items: list[str] = []
    flags = [str(flag) for flag in status.get("flags") or []]
    if "bad_live_all_pairs_config" in flags:
        blockers.append("bad_live_all_pairs_config")
    if pool_status["active_batch_not_covering_new_checkpoints"]:
        blockers.append("active_game_batch_not_covering_new_checkpoints")
    if batch and active_batch_checkpoint_count < intake_checkpoint_count:
        open_items.append("new_checkpoints_arrived_after_active_game_batch_started")
    if latest_checkpoint_count < intake_checkpoint_count:
        open_items.append("latest_rating_behind_intake")
    if latest_round_index > refresh_round_index:
        blockers.append("trainer_export_stale_vs_latest_rating")
    batch_age = _age_seconds(checked_at=checked_at, updated_at=batch_updated_at)
    visible_output_count = max(
        _int(probe.get("completed_game_count")),
        _int(progress_probe.get("completed_game_count")),
    )
    if batch and visible_output_count <= 0:
        if batch_age is not None and batch_age >= 120:
            blockers.append("active_game_batch_no_probe_output_after_120s")
        else:
            open_items.append("active_game_batch_no_probe_output_yet")
    recent_game_batches = []
    raw_recent = status.get("recent_game_batches")
    if isinstance(raw_recent, list):
        for item in raw_recent[-8:]:
            if not isinstance(item, Mapping):
                continue
            item_updated_at = item.get("updated_at")
            skip_decision = (
                item.get("skip_decision")
                if isinstance(item.get("skip_decision"), Mapping)
                else {}
            )
            recent_game_batches.append(
                {
                    "round_id": item.get("round_id"),
                    "round_index": item.get("round_index"),
                    "status": item.get("status"),
                    "phase": item.get("phase"),
                    "ratings_written": item.get("ratings_written"),
                    "updated_at": item_updated_at,
                    "age_seconds": _age_seconds(
                        checked_at=checked_at,
                        updated_at=item_updated_at,
                    ),
                    "checkpoint_count": item.get("checkpoint_count"),
                    "rating_spec_checkpoint_count": item.get(
                        "rating_spec_checkpoint_count"
                    ),
                    "checkpoint_roster_count": item.get("checkpoint_roster_count"),
                    "pair_count": item.get("pair_count"),
                    "game_count": item.get("game_count"),
                    "completed_game_count": item.get("completed_game_count"),
                    "skip_reason": item.get("skip_reason"),
                    "skip_input_checkpoint_count": skip_decision.get(
                        "input_checkpoint_count"
                    ),
                    "skip_desired_checkpoint_count": skip_decision.get(
                        "desired_checkpoint_count"
                    ),
                    "skip_completed_game_count": skip_decision.get(
                        "completed_game_count"
                    ),
                    "skip_started_pair_count": skip_decision.get("started_pair_count"),
                    "skip_scan_output_progress": skip_decision.get(
                        "scan_output_progress"
                    ),
                    "skip_latest_result_ts": skip_decision.get("latest_result_ts"),
                }
            )
    return {
        "status": status.get("status"),
        "flags": status.get("flags") or [],
        "checked_at": checked_at,
        "operator_next_action": status.get("operator_next_action"),
        "blockers": blockers,
        "open_items": open_items,
        "proof_chain": proof_chain,
        "pool_status": status.get("pool_status")
        if isinstance(status.get("pool_status"), Mapping)
        else pool_status,
        "intake": {
            "checkpoint_count": intake.get("checkpoint_count"),
            "queue_len": intake.get("queue_len"),
            "new_checkpoints_not_in_latest_rating": (
                intake.get("new_checkpoints_not_in_latest_rating")
            ),
            "pair_selection": (intake.get("rating_defaults") or {}).get("pair_selection")
            if isinstance(intake.get("rating_defaults"), Mapping)
            else None,
            "pairs_per_round": (intake.get("rating_defaults") or {}).get("pairs_per_round")
            if isinstance(intake.get("rating_defaults"), Mapping)
            else None,
        },
        "latest_rating": {
            "round_id": latest.get("round_id"),
            "round_index": latest.get("round_index"),
            "checkpoint_count": latest.get("checkpoint_count"),
            "rating_count": latest.get("rating_count"),
            "active_count": latest.get("active_count"),
        },
        "current_game_batch": {
            "round_id": batch.get("round_id"),
            "round_index": batch.get("round_index"),
            "status": batch.get("status"),
            "phase": batch.get("phase"),
            "updated_at": batch_updated_at,
            "age_seconds": _age_seconds(
                checked_at=checked_at,
                updated_at=batch_updated_at,
            ),
            "checkpoint_count": batch.get("checkpoint_count"),
            "rating_spec_checkpoint_count": batch.get("rating_spec_checkpoint_count"),
            "checkpoint_roster_count": batch.get("checkpoint_roster_count"),
            "pair_count": batch.get("pair_count"),
            "game_count": batch.get("game_count"),
            "root_completed_game_count": batch.get("completed_game_count"),
            "completion_fraction": batch.get("completion_fraction"),
            "partial_reduce_due": batch.get("partial_reduce_due"),
            "partial_reduce_after_seconds": batch.get("partial_reduce_after_seconds"),
            "liveness_probe_completed_summary_count": probe.get(
                "completed_game_count"
            ),
            "liveness_probe_sampled_pair_count": probe.get("sampled_pair_count"),
            "liveness_probe_seen_pair_count": probe.get("seen_pair_count"),
            "liveness_probe_stopped_after_first_output": probe.get(
                "stopped_after_first_output"
            ),
            "liveness_probe_count_semantics": probe.get("count_semantics"),
            "liveness_probe_latest_result_age_seconds": probe.get(
                "latest_result_age_seconds"
            ),
            "liveness_probe_scan_output_progress": probe.get("scan_output_progress"),
            "liveness_probe_is_stale": probe.get("is_stale"),
            "progress_probe_completed_game_count": progress_probe.get(
                "completed_game_count"
            ),
            "progress_probe_estimated_seen_game_count": progress_probe.get(
                "estimated_seen_game_count"
            ),
            "progress_probe_started_pair_count": progress_probe.get(
                "started_pair_count"
            ),
            "progress_probe_partial_pair_count": progress_probe.get(
                "partial_pair_count"
            ),
            "progress_probe_completed_pair_count": progress_probe.get(
                "completed_pair_count"
            ),
            "progress_probe_completion_fraction": progress_probe.get(
                "completion_fraction"
            ),
            "progress_probe_latest_result_age_seconds": progress_probe.get(
                "latest_result_age_seconds"
            ),
            "progress_probe_count_basis": progress_probe.get("count_basis"),
            "progress_probe_count_semantics": progress_probe.get("count_semantics"),
            "progress_probe_scan_error_count": progress_probe.get("scan_error_count"),
            "progress_probe_error_type": progress_probe.get("error_type"),
            "progress_probe_error": progress_probe.get("error"),
            "pool_status": batch.get("pool_status")
            if isinstance(batch.get("pool_status"), Mapping)
            else pool_status,
        }
        if batch
        else None,
        "active_game_batch_count": status.get("active_game_batch_count"),
        "recent_game_batches": recent_game_batches,
        "trainer_refresh": {
            "generation": refresh.get("generation"),
            "snapshot_id": refresh.get("snapshot_id"),
            "active_count": refresh.get("active_count"),
            "rewritten_pointer_count": refresh.get("rewritten_pointer_count"),
            "assignment_sha256_prefixes": refresh.get("assignment_sha256_prefixes"),
            "rating_round_id": (refresh.get("rating_source") or {}).get("round_id")
            if isinstance(refresh.get("rating_source"), Mapping)
            else None,
        },
    }


def _refresh_decision(status: Mapping[str, Any]) -> dict[str, Any]:
    latest = (
        status.get("latest_rating")
        if isinstance(status.get("latest_rating"), Mapping)
        else {}
    )
    refresh = (
        status.get("trainer_refresh")
        if isinstance(status.get("trainer_refresh"), Mapping)
        else {}
    )
    rating_source = (
        refresh.get("rating_source")
        if isinstance(refresh.get("rating_source"), Mapping)
        else {}
    )
    latest_round_index = _int(latest.get("round_index"), -1)
    refresh_round_index = _int(rating_source.get("round_index"), -1)
    latest_checkpoint_count = _int(latest.get("checkpoint_count"))
    if not latest.get("exists"):
        return {
            "action": "refresh-if-ready",
            "spawn_refresh": False,
            "reason": "no_latest_rating_snapshot",
            "latest_round_index": latest_round_index,
            "refresh_round_index": refresh_round_index,
            "latest_checkpoint_count": latest_checkpoint_count,
        }
    if latest_round_index <= refresh_round_index:
        return {
            "action": "refresh-if-ready",
            "spawn_refresh": False,
            "reason": "trainer_refresh_already_current",
            "latest_round_index": latest_round_index,
            "refresh_round_index": refresh_round_index,
            "latest_checkpoint_count": latest_checkpoint_count,
        }
    return {
        "action": "refresh-if-ready",
        "spawn_refresh": True,
        "reason": "latest_rating_newer_than_trainer_refresh",
        "latest_round_index": latest_round_index,
        "refresh_round_index": refresh_round_index,
        "latest_checkpoint_count": latest_checkpoint_count,
    }


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_ids_from_manifest(manifest_path: str, *, limit: int = 0) -> list[str]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"manifest {manifest_path!r} has no rows list")
    run_ids = [
        str(row.get("run_id") or "").strip()
        for row in rows
        if isinstance(row, Mapping) and str(row.get("run_id") or "").strip()
    ]
    if limit > 0:
        return run_ids[: int(limit)]
    return run_ids


def _assignment_proof_summary(
    rows: list[Mapping[str, Any]],
    *,
    target_assignment_shas: list[str],
    row_limit: int = DEFAULT_ASSIGNMENT_PROOF_ROW_LIMIT,
) -> dict[str, Any]:
    target_set = {sha for sha in target_assignment_shas if sha}
    latest_applied_counter: Counter[str] = Counter()
    latest_decision_counter: Counter[str] = Counter()
    compact_rows: list[dict[str, Any]] = []
    target_row_count = 0
    target_provider_ok_count = 0
    target_provider_false_count = 0
    target_provider_null_count = 0
    assignment_refresh_event_count = 0
    assignment_refresh_applied_count = 0
    latest_applied_target_count = 0

    for row in rows:
        latest_applied_sha = str(row.get("assignment_refresh_latest_applied_sha256") or "")
        latest_decision = str(row.get("assignment_refresh_latest_decision") or "")
        latest_decision_counter[latest_decision or "none"] += 1
        if latest_applied_sha:
            latest_applied_counter[latest_applied_sha[:8]] += 1
        if target_set and latest_applied_sha in target_set:
            latest_applied_target_count += 1
        assignment_refresh_event_count += _int(row.get("assignment_refresh_event_count"))
        assignment_refresh_applied_count += _int(row.get("assignment_refresh_applied_count"))
        target_row_count += _int(row.get("assignment_env_proof_target_row_count"))
        target_provider_ok_count += _int(
            row.get("assignment_env_proof_target_provider_ok_count")
        )
        target_provider_false_count += _int(
            row.get("assignment_env_proof_target_provider_false_count")
        )
        target_provider_null_count += _int(
            row.get("assignment_env_proof_target_provider_null_count")
        )
        compact_rows.append(
            {
                "run_id": row.get("run_id"),
                "attempt_id": row.get("attempt_id"),
                "latest_decision": row.get("assignment_refresh_latest_decision"),
                "latest_reason": row.get("assignment_refresh_latest_reason"),
                "latest_train_iter": row.get("assignment_refresh_latest_train_iter"),
                "latest_applied_train_iter": row.get(
                    "assignment_refresh_latest_applied_train_iter"
                ),
                "latest_applied_sha_prefix": (
                    latest_applied_sha[:8] if latest_applied_sha else None
                ),
                "target_rows": row.get("assignment_env_proof_target_row_count"),
                "target_provider_ok": row.get(
                    "assignment_env_proof_target_provider_ok_count"
                ),
                "target_provider_false": row.get(
                    "assignment_env_proof_target_provider_false_count"
                ),
                "target_provider_null": row.get(
                    "assignment_env_proof_target_provider_null_count"
                ),
            }
        )
    normalized_row_limit = int(row_limit)
    if normalized_row_limit < 0:
        output_rows = compact_rows
    elif normalized_row_limit > 0:
        output_rows = compact_rows[:normalized_row_limit]
    else:
        output_rows = []
    summary = {
        "run_count": len(rows),
        "target_assignment_count": len(target_set),
        "latest_applied_target_count": latest_applied_target_count,
        "assignment_refresh_event_count": assignment_refresh_event_count,
        "assignment_refresh_applied_count": assignment_refresh_applied_count,
        "latest_decision_counts": dict(sorted(latest_decision_counter.items())),
        "latest_applied_sha_prefix_counts": dict(sorted(latest_applied_counter.items())),
        "target_env_row_count": target_row_count,
        "target_provider_ok_count": target_provider_ok_count,
        "target_provider_false_count": target_provider_false_count,
        "target_provider_null_count": target_provider_null_count,
        "row_output_limit": normalized_row_limit,
        "rows_omitted_count": max(0, len(compact_rows) - len(output_rows)),
    }
    if output_rows:
        summary["rows"] = output_rows
    return summary


def _chunks(items: list[str], chunk_size: int) -> list[list[str]]:
    size = max(1, int(chunk_size))
    return [items[index : index + size] for index in range(0, len(items), size)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Operate the live CurvyTron feedback loop through deployed Modal "
            "functions, without creating a temporary scheduled Modal app."
        )
    )
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME)
    parser.add_argument("--run-status-app-name", default=DEFAULT_RUN_STATUS_APP_NAME)
    parser.add_argument("--tournament-id", default=DEFAULT_TOURNAMENT_ID)
    parser.add_argument("--rating-run-id", default=DEFAULT_RATING_RUN_ID)
    parser.add_argument("--leaderboard-id", default="")
    parser.add_argument(
        "--refresh-config-ref",
        default=TRAINING_CANDIDATE_REFRESH_CONFIG_REF,
        help="Control-volume config ref used when spawning trainer-candidate refresh.",
    )
    parser.add_argument(
        "--action",
        choices=(
            "status",
            "drain-if-ready",
            "drain",
            "refresh-if-ready",
            "trainer-proof",
        ),
        default="status",
    )
    parser.add_argument("--run-manifest", default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--run-limit", type=int, default=0)
    parser.add_argument("--target-assignment-shas", default="")
    parser.add_argument(
        "--assignment-proof-tail-bytes",
        type=int,
        default=DEFAULT_ASSIGNMENT_PROOF_TAIL_BYTES,
    )
    parser.add_argument(
        "--assignment-proof-chunk-size",
        type=int,
        default=DEFAULT_ASSIGNMENT_PROOF_CHUNK_SIZE,
    )
    parser.add_argument(
        "--assignment-proof-row-limit",
        type=int,
        default=DEFAULT_ASSIGNMENT_PROOF_ROW_LIMIT,
        help=(
            "Number of per-run trainer-proof rows to print. "
            "Default 0 omits rows. Use -1 for all rows."
        ),
    )
    parser.add_argument("--lookahead-batches", type=int, default=64)
    parser.add_argument("--activity-probe-pairs", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=2000)
    parser.add_argument("--claim-stale-after-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument("--rating-round-stale-after-seconds", type=int, default=600)
    parser.add_argument(
        "--drain-request-stale-after-seconds",
        type=int,
        default=DEFAULT_DRAIN_REQUEST_STALE_AFTER_SECONDS,
        help=(
            "Local operator lease duration. Prevents repeated drain spawns while "
            "Modal is still scheduling the last requested drain."
        ),
    )
    parser.add_argument(
        "--ignore-drain-request-lease",
        action="store_true",
        help="Ignore a recent operator drain lease and use only deployed status.",
    )
    parser.add_argument(
        "--no-drain-call-probe",
        dest="drain_call_probe",
        action="store_false",
        default=True,
        help=(
            "Do not inspect the recent drain/rating FunctionCall. "
            "By default status includes a compact call-state probe."
        ),
    )
    parser.add_argument(
        "--progress-probe",
        dest="progress_probe",
        action="store_true",
        default=False,
        help=(
            "Ask the deployed function for an active-batch progress scan. This can "
            "be slow on large live batches; routine status still labels the cheap "
            "liveness probe separately."
        ),
    )
    parser.add_argument(
        "--full-drain-call-graph",
        action="store_true",
        help="Include the full nested Modal FunctionCall graph. Default is compact counts only.",
    )
    parser.add_argument("--drain-call-graph-sample-limit", type=int, default=12)
    parser.add_argument(
        "--detach-drain",
        action="store_true",
        help=(
            "Spawn the deployed drain and return only its function call id. "
            "By default the tool waits for the short drain result, while the "
            "rating loop spawned by that drain remains detached."
        ),
    )
    parser.add_argument("--full-status", action="store_true")
    args = parser.parse_args()

    leaderboard_id = args.leaderboard_id or (
        f"{args.tournament_id}-{args.rating_run_id}-training"
    )
    status_fn = modal.Function.from_name(args.app_name, "curvytron_feedback_loop_status")
    status = status_fn.remote(
        {
            "tournament_id": args.tournament_id,
            "rating_run_id": args.rating_run_id,
            "leaderboard_id": leaderboard_id,
            "lookahead_batches": args.lookahead_batches,
            "status_activity_probe_pairs": args.activity_probe_pairs,
            "status_progress_probe": bool(args.progress_probe),
        }
    )
    if args.action == "refresh-if-ready":
        decision = _refresh_decision(status)
    elif args.action == "trainer-proof":
        decision = {
            "action": "trainer-proof",
            "spawn_drain": False,
            "spawn_refresh": False,
            "reason": "read_trainer_assignment_proof",
            "status": status.get("status"),
        }
    else:
        decision = _control_decision(status, action=args.action)
    intake_state = modal.Dict.from_name(CHECKPOINT_INTAKE_DICT_NAME, create_if_missing=True)
    drain_request_key = _drain_request_key(args.tournament_id, args.rating_run_id)
    existing_drain_request = intake_state.get(drain_request_key, None)
    pending_drain_call_probe: dict[str, Any] | None = None
    if (
        bool(decision.get("spawn_drain"))
        and not args.ignore_drain_request_lease
        and _drain_request_is_fresh(
            existing_drain_request,
            stale_after_seconds=args.drain_request_stale_after_seconds,
        )
    ):
        decision = {
            **decision,
            "spawn_drain": False,
            "reason": "blocked_recent_drain_request",
        }
    if (
        bool(decision.get("spawn_drain"))
        and not args.ignore_drain_request_lease
        and not _decision_is_recovery_scan(decision)
        and isinstance(existing_drain_request, Mapping)
        and str(existing_drain_request.get("function_call_id") or "").strip()
    ):
        pending_drain_call_probe = _function_call_probe(
            existing_drain_request.get("function_call_id"),
            include_full_call_graph=False,
            call_graph_sample_limit=int(args.drain_call_graph_sample_limit),
        )
        if _probe_has_pending_work(pending_drain_call_probe):
            decision = {
                **decision,
                "spawn_drain": False,
                "reason": "blocked_pending_drain_function_call",
            }
    output: dict[str, Any] = {
        "schema_id": TOOL_SCHEMA_ID,
        "app_name": args.app_name,
        "tournament_id": args.tournament_id,
        "rating_run_id": args.rating_run_id,
        "leaderboard_id": leaderboard_id,
        "decision": decision,
        "drain_request_key": drain_request_key,
        "pending_drain_request": _drain_request_summary(existing_drain_request),
        "status_summary": _status_summary(status),
    }
    if args.drain_call_probe and isinstance(existing_drain_request, Mapping):
        if pending_drain_call_probe is None or bool(args.full_drain_call_graph):
            pending_drain_call_probe = _function_call_probe(
                existing_drain_request.get("function_call_id"),
                include_full_call_graph=bool(args.full_drain_call_graph),
                call_graph_sample_limit=int(args.drain_call_graph_sample_limit),
            )
        output["pending_drain_call_probe"] = pending_drain_call_probe
    if args.full_status:
        output["status"] = status
    if bool(decision.get("spawn_refresh")):
        refresh_fn = modal.Function.from_name(
            args.app_name,
            "curvytron_training_candidate_refresh_tick",
        )
        refresh_result = refresh_fn.remote(
            {
                "tournament_id": args.tournament_id,
                "rating_run_id": args.rating_run_id,
                "leaderboard_id": leaderboard_id,
                "config_ref": args.refresh_config_ref,
            }
        )
        output["refresh_result"] = refresh_result
    if args.action == "trainer-proof":
        run_ids = _run_ids_from_manifest(args.run_manifest, limit=int(args.run_limit))
        trainer_refresh = (
            status.get("trainer_refresh")
            if isinstance(status.get("trainer_refresh"), Mapping)
            else {}
        )
        target_assignment_shas = _split_csv(args.target_assignment_shas) or [
            str(item)
            for item in trainer_refresh.get("assignment_sha256s") or []
            if str(item).strip()
        ]
        proof_fn = modal.Function.from_name(
            args.run_status_app_name,
            "curvytron_assignment_proof",
        )
        run_id_chunks = _chunks(run_ids, int(args.assignment_proof_chunk_size))
        proof_batches = list(
            proof_fn.starmap(
                [
                    (
                        chunk,
                        None,
                        target_assignment_shas,
                        int(args.assignment_proof_tail_bytes),
                    )
                    for chunk in run_id_chunks
                ]
            )
        )
        proof_rows = [
            row
            for batch in proof_batches
            for row in (batch if isinstance(batch, list) else [])
        ]
        output["trainer_proof"] = _assignment_proof_summary(
            proof_rows,
            target_assignment_shas=target_assignment_shas,
            row_limit=int(args.assignment_proof_row_limit),
        )
        output["trainer_proof"]["chunk_count"] = len(run_id_chunks)
        output["trainer_proof"]["chunk_size"] = int(args.assignment_proof_chunk_size)
        output["trainer_proof"]["tail_bytes"] = int(args.assignment_proof_tail_bytes)
    if decision.get("spawn_drain"):
        request_payload = {
            "schema_id": f"{TOOL_SCHEMA_ID}/drain_request",
            "requested_at": _utc_timestamp(),
            "stale_after_seconds": int(args.drain_request_stale_after_seconds),
            "app_name": args.app_name,
            "tournament_id": args.tournament_id,
            "rating_run_id": args.rating_run_id,
            "leaderboard_id": leaderboard_id,
            "action": args.action,
            "status_at_request": status.get("status"),
            "queue_len_at_request": decision.get("queue_len"),
            "new_checkpoints_at_request": decision.get(
                "new_checkpoints_not_in_latest_rating"
            ),
            "decision_reason": decision.get("reason"),
        }
        intake_state.put(drain_request_key, request_payload)
        drain_fn = modal.Function.from_name(args.app_name, "curvytron_checkpoint_intake_drain")
        output["drain_spawn"] = {
            "max_events": args.max_events,
            "spawn_if_existing": True,
        }
        if args.detach_drain:
            call = drain_fn.spawn(
                {
                    "tournament_id": args.tournament_id,
                    "rating_run_id": args.rating_run_id,
                    "spawn_rating": True,
                    "spawn_if_existing": True,
                    "max_events": args.max_events,
                    "claim_stale_after_seconds": args.claim_stale_after_seconds,
                    "rating_round_stale_after_seconds": args.rating_round_stale_after_seconds,
                    "wait_for_rating": False,
                }
            )
            output["drain_spawn"].update(
                {
                    "status": "spawned",
                    "function_call_id": _function_call_id(call),
                }
            )
        else:
            drain_result = drain_fn.remote(
                {
                    "tournament_id": args.tournament_id,
                    "rating_run_id": args.rating_run_id,
                    "spawn_rating": True,
                    "spawn_if_existing": True,
                    "max_events": args.max_events,
                    "claim_stale_after_seconds": args.claim_stale_after_seconds,
                    "rating_round_stale_after_seconds": args.rating_round_stale_after_seconds,
                    "wait_for_rating": False,
                }
            )
            output["drain_result_summary"] = _drain_result_summary(drain_result)
            output["drain_spawn"].update(
                {
                    "status": "drain_returned",
                    "function_call_id": (
                        (output["drain_result_summary"] or {}).get("rating_call_id") or ""
                    ),
                }
            )
        request_payload = {
            **request_payload,
            "function_call_id": output["drain_spawn"]["function_call_id"],
            "spawned_at": _utc_timestamp(),
            "drain_finished_at": None if args.detach_drain else _utc_timestamp(),
            "drain_status": output["drain_spawn"]["status"],
            "drain_spawn_skipped_reason": (
                (output.get("drain_result_summary") or {}).get("spawn_skipped_reason")
            ),
        }
        intake_state.put(drain_request_key, request_payload)
        output["pending_drain_request"] = _drain_request_summary(request_payload)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
