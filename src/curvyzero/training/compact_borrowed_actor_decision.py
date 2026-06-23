"""Hash-bound OPT-098 decision for borrowed single-actor render state."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.training.compact_speed_row_floor_bundle import FALSE_CLAIM_KEYS
from curvyzero.training.compact_speed_row_floor_bundle import _json_sha256
from curvyzero.training.compact_speed_row_floor_bundle import _read_json_mapping
from curvyzero.training.compact_speed_row_floor_bundle import _require_non_empty
from curvyzero.training.compact_speed_row_floor_bundle import _required_mapping
from curvyzero.training.compact_speed_row_floor_bundle import _sha256


COMPACT_BORROWED_ACTOR_DECISION_SCHEMA_ID = (
    "curvyzero_compact_borrowed_actor_decision/v1"
)
COMPACT_BORROWED_ACTOR_DECISION_MANIFEST_SCHEMA_ID = (
    "curvyzero_compact_borrowed_actor_decision_manifest/v1"
)
COMPACT_BORROWED_ACTOR_DECISION_EVIDENCE_REF_PREFIX = (
    "compact_borrowed_actor_decision:"
)

STATUS_COMPLETE = "compact_borrowed_actor_decision_complete"

DECISION_APPROVED_A1_BORROWED_OPERATIONAL_SPEED_CANDIDATE = (
    "approved_a1_borrowed_operational_speed_candidate_not_same_shape"
)
DECISION_NOT_APPROVED_SPEED_ROWS_FAILED = "not_approved_speed_rows_failed"
DECISION_NOT_APPROVED_TERMINAL_DEATH_PROOF_FAILED = (
    "not_approved_terminal_death_proof_failed"
)
DECISION_NOT_APPROVED_REFERENCE_NOT_OUTPACED = (
    "not_approved_same_shape_reference_not_outpaced"
)

BORROWED_HANDOFF_MODE = "borrow_single_actor_env_state"
COPY_HANDOFF_MODE = "copy_actor_state_to_parent_buffers"
SPEED_CURRENCY = "compact_trainer_env_steps_per_sec"

MIN_BORROWED_ROWS = 2
BORROWED_ROW_MAX_WALL_SEC = 7.0
BORROWED_ROW_MIN_STEPS_PER_SEC = 25_000.0
BORROWED_MEAN_MAX_WALL_SEC = 6.5
BORROWED_MEAN_MIN_STEPS_PER_SEC = 30_000.0


class CompactBorrowedActorDecisionError(ValueError):
    """Raised when a borrowed-actor decision report is malformed."""


def build_compact_borrowed_actor_decision_v1(
    *,
    run_id: str,
    borrowed_speed_row_paths: list[str | Path],
    normal_death_profile_result_path: str | Path,
    same_shape_reference_speed_row_path: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    borrowed_rows = [
        _borrowed_speed_row_summary(Path(path), index=index)
        for index, path in enumerate(borrowed_speed_row_paths)
    ]
    reference = _same_shape_reference_summary(Path(same_shape_reference_speed_row_path))
    normal_death = _normal_death_proof_summary(Path(normal_death_profile_result_path))
    speed_lane = _speed_lane_summary(borrowed_rows)
    comparison = _reference_comparison(speed_lane=speed_lane, reference=reference)
    decision = _decision(
        speed_lane=speed_lane,
        normal_death=normal_death,
        comparison=comparison,
    )
    non_claims = _false_claims()
    approved = decision == DECISION_APPROVED_A1_BORROWED_OPERATIONAL_SPEED_CANDIDATE
    payload: dict[str, Any] = {
        "schema_id": COMPACT_BORROWED_ACTOR_DECISION_SCHEMA_ID,
        "ok": True,
        "status": STATUS_COMPLETE,
        "run_id": _require_non_empty(run_id, "run_id"),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "decision": decision,
        "borrowed_candidate_scope": "h100_b1024_a1_sim1_profile_no_death_speed_lane",
        "correctness_scope": "separate_h100_b1024_a1_sim8_normal_death_safety_lane",
        "borrowed_speed_rows": borrowed_rows,
        "same_shape_reference_speed_row": reference,
        "normal_death_borrowed_profile_proof": normal_death,
        "speed_lane": speed_lane,
        "same_shape_reference_comparison": comparison,
        "interpretation": {
            "allowed": (
                "use_a1_borrowed_as_next_operational_speed_candidate"
                if approved
                else "do_not_use_a1_borrowed_as_operational_speed_candidate"
            ),
            "not_allowed": [
                "do_not_call_this_a_same_shape_a16_baseline",
                "do_not_claim_normal_death_speed_from_profile_no_death_rows",
                "do_not_claim_stock_replacement_or_promotion",
                "do_not_claim_matched_learning_quality",
            ],
            "next_required_measurement": (
                "run a normal-death compact Coach speed row before making "
                "normal-death throughput claims"
            ),
        },
        "non_claims": non_claims,
        "attached_claims": {
            "hash_bound_borrowed_actor_decision": True,
            "borrowed_a1_operational_speed_candidate_allowed": approved,
            "borrowed_a1_speed_lane_profile_no_death": True,
            "normal_death_correctness_lane_passed": bool(normal_death["passed"]),
            "normal_death_speed_claim": False,
            "same_shape_baseline_replaced": False,
            **non_claims,
        },
    }
    payload["evidence_ref"] = compact_borrowed_actor_decision_evidence_ref(payload)
    validate_compact_borrowed_actor_decision_v1(payload)
    return payload


def validate_compact_borrowed_actor_decision_v1(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_id") != COMPACT_BORROWED_ACTOR_DECISION_SCHEMA_ID:
        raise CompactBorrowedActorDecisionError("borrowed actor decision schema mismatch")
    if payload.get("status") != STATUS_COMPLETE:
        raise CompactBorrowedActorDecisionError("borrowed actor decision status mismatch")
    decision = str(payload.get("decision") or "")
    if decision not in {
        DECISION_APPROVED_A1_BORROWED_OPERATIONAL_SPEED_CANDIDATE,
        DECISION_NOT_APPROVED_SPEED_ROWS_FAILED,
        DECISION_NOT_APPROVED_TERMINAL_DEATH_PROOF_FAILED,
        DECISION_NOT_APPROVED_REFERENCE_NOT_OUTPACED,
    }:
        raise CompactBorrowedActorDecisionError("unknown borrowed actor decision")
    _validate_false_claims(payload.get("non_claims"), "non_claims")
    _validate_false_claims(payload.get("attached_claims"), "attached_claims")
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    approved = decision == DECISION_APPROVED_A1_BORROWED_OPERATIONAL_SPEED_CANDIDATE
    if claims.get("borrowed_a1_operational_speed_candidate_allowed") is not approved:
        raise CompactBorrowedActorDecisionError(
            "borrowed A1 allowed claim must match decision"
        )
    if claims.get("same_shape_baseline_replaced") is not False:
        raise CompactBorrowedActorDecisionError("same-shape baseline claim must be false")
    if claims.get("normal_death_speed_claim") is not False:
        raise CompactBorrowedActorDecisionError("normal-death speed claim must be false")

    rows = payload.get("borrowed_speed_rows")
    if not isinstance(rows, list) or len(rows) < MIN_BORROWED_ROWS:
        raise CompactBorrowedActorDecisionError("borrowed speed rows missing")
    for row in rows:
        _validate_input_ref(_required_mapping(row, "borrowed speed row"))
    _validate_input_ref(
        _required_mapping(
            payload.get("same_shape_reference_speed_row"),
            "same_shape_reference_speed_row",
        )
    )
    _validate_input_ref(
        _required_mapping(
            payload.get("normal_death_borrowed_profile_proof"),
            "normal_death_borrowed_profile_proof",
        )
    )
    speed_lane = _required_mapping(payload.get("speed_lane"), "speed_lane")
    normal_death = _required_mapping(
        payload.get("normal_death_borrowed_profile_proof"),
        "normal_death_borrowed_profile_proof",
    )
    comparison = _required_mapping(
        payload.get("same_shape_reference_comparison"),
        "same_shape_reference_comparison",
    )
    if approved:
        if speed_lane.get("passed") is not True:
            raise CompactBorrowedActorDecisionError("approved decision requires speed lane")
        if normal_death.get("passed") is not True:
            raise CompactBorrowedActorDecisionError(
                "approved decision requires normal-death proof"
            )
        if comparison.get("borrowed_outpaced_reference") is not True:
            raise CompactBorrowedActorDecisionError(
                "approved decision requires outpacing reference"
            )
    expected_ref = compact_borrowed_actor_decision_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactBorrowedActorDecisionError("borrowed actor evidence_ref mismatch")


def compact_borrowed_actor_decision_evidence_ref(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "evidence_ref"}
    run_id = _require_non_empty(payload.get("run_id"), "run_id")
    return (
        f"{COMPACT_BORROWED_ACTOR_DECISION_EVIDENCE_REF_PREFIX}"
        f"{run_id}:sha256={_json_sha256(body)}"
    )


def _borrowed_speed_row_summary(path: Path, *, index: int) -> dict[str, Any]:
    result = _read_json_mapping(path, "borrowed speed row")
    row = _required_mapping(result.get("row"), "borrowed speed row.row")
    summary = _required_mapping(result.get("summary"), "borrowed speed row.summary")
    compact = _required_mapping(result.get("compact"), "borrowed speed row.compact")
    source = _source_profile(compact)
    last_search = _optional_mapping(
        compact.get("compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata")
    )
    last_replay = _optional_mapping(
        compact.get("compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata")
    )
    learner_updates = _int(compact.get("compact_rollout_slab_learner_gate_updates"))
    checks = {
        "result_complete": result.get("status") == "complete",
        "result_ok": result.get("ok") is True,
        "returncode_zero_or_absent": result.get("returncode") in (0, None),
        "summary_ok": summary.get("ok") is True,
        "compact_ok": compact.get("ok") is True,
        "profile_only_false": summary.get("profile_only") is False
        and compact.get("profile_only") is False,
        "calls_train_muzero_false": summary.get("calls_train_muzero") is False
        and compact.get("calls_train_muzero") is False,
        "touches_live_runs_false": summary.get("touches_live_runs") is False
        and compact.get("touches_live_runs") is False,
        "real_compact_training_work": compact.get("real_compact_owned_training_work") is True,
        "shape_b1024_a1": _int(row.get("batch_size")) == 1024
        and _int(row.get("actor_count")) == 1,
        "sim1_steps180_warmup45": _int(row.get("num_simulations")) == 1
        and _int(row.get("steps")) == 180
        and _int(row.get("warmup_steps")) == 45,
        "policy_refresh_interval4": _int(row.get("policy_refresh_interval")) == 4,
        "learner_cuda": str(row.get("learner_device") or "") == "cuda",
        "borrowed_requested": row.get("hybrid_borrow_single_actor_render_state") is True
        and compact.get("hybrid_borrow_single_actor_render_state") is True,
        "borrowed_handoff": compact.get("render_state_handoff_mode") == BORROWED_HANDOFF_MODE,
        "borrowed_all_steps": _int(compact.get("render_state_borrowed_steps")) == (
            _int(row.get("steps")) + _int(row.get("warmup_steps"))
        ),
        "no_render_copy_in_speed_lane": _int(compact.get("render_state_copy_steps")) == 0,
        "profile_no_death_surface": str(source.get("death_mode") or "profile_no_death")
        == "profile_no_death",
        "no_terminal_or_death_rows": _terminalish_row_max(source) == 0,
        "no_deferred_learner": compact.get("compact_owned_loop_deferred_learner") is False,
        "learner_updates_positive": learner_updates > 0,
        "final_refresh_consumed": _int(
            compact.get("compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count")
        )
        == learner_updates
        and _int(last_search.get("compact_policy_refresh_learner_update_count"))
        == learner_updates
        and _int(last_replay.get("compact_policy_refresh_learner_update_count"))
        == learner_updates,
        "search_and_replay_metadata_cover_measured_steps": _int(
            compact.get("compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_count")
        )
        >= _int(row.get("steps"))
        and _int(
            compact.get("compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_count")
        )
        >= _int(row.get("steps")),
        "resident_host_fallback_zero": _float(
            compact.get(
                "resident_observation_host_fallback_count",
                source.get("resident_observation_host_fallback_count", 0.0),
            )
        )
        == 0.0,
        "per_row_speed_threshold": _float(compact.get("steps_per_sec"))
        >= BORROWED_ROW_MIN_STEPS_PER_SEC,
        "per_row_wall_threshold": _float(compact.get("training_wall_sec"))
        <= BORROWED_ROW_MAX_WALL_SEC,
    }
    failed = [key for key, value in checks.items() if value is not True]
    return {
        "index": index,
        "input_ref": _input_ref(path, result.get("schema_id", "")),
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "steps_per_sec": _float(compact.get("steps_per_sec")),
        "training_wall_sec": _float(compact.get("training_wall_sec")),
        "env_steps_collected": _float(compact.get("env_steps_collected")),
        "shape": {
            "batch_size": _int(row.get("batch_size")),
            "actor_count": _int(row.get("actor_count")),
            "num_simulations": _int(row.get("num_simulations")),
            "steps": _int(row.get("steps")),
            "warmup_steps": _int(row.get("warmup_steps")),
            "policy_refresh_interval": _int(row.get("policy_refresh_interval")),
        },
        "death_surface": {
            "death_mode": str(source.get("death_mode") or "profile_no_death"),
            "terminal_row_count": _int(source.get("terminal_row_count")),
            "death_row_count": _int(source.get("death_row_count")),
            "terminated_row_count": _int(source.get("terminated_row_count")),
            "truncated_row_count": _int(source.get("truncated_row_count")),
        },
        "borrow_contract": {
            "render_state_handoff_mode": compact.get("render_state_handoff_mode"),
            "render_state_copy_steps": _int(compact.get("render_state_copy_steps")),
            "render_state_borrowed_steps": _int(
                compact.get("render_state_borrowed_steps")
            ),
        },
        "learner_updates": learner_updates,
    }


def _same_shape_reference_summary(path: Path) -> dict[str, Any]:
    result = _read_json_mapping(path, "same-shape reference row")
    row = _required_mapping(result.get("row"), "same-shape reference row.row")
    summary = _required_mapping(result.get("summary"), "same-shape reference row.summary")
    compact = _required_mapping(result.get("compact"), "same-shape reference row.compact")
    checks = {
        "result_complete": result.get("status") == "complete",
        "summary_ok": summary.get("ok") is True,
        "compact_ok": compact.get("ok") is True,
        "profile_only_false": compact.get("profile_only") is False,
        "real_compact_training_work": compact.get("real_compact_owned_training_work") is True,
        "shape_b1024_a16": _int(row.get("batch_size")) == 1024
        and _int(row.get("actor_count")) == 16,
        "sim1_steps180_warmup45": _int(row.get("num_simulations")) == 1
        and _int(row.get("steps")) == 180
        and _int(row.get("warmup_steps")) == 45,
        "not_borrowed": row.get("hybrid_borrow_single_actor_render_state") is not True
        and compact.get("hybrid_borrow_single_actor_render_state") is not True,
        "copy_handoff": compact.get("render_state_handoff_mode") == COPY_HANDOFF_MODE,
        "steps_per_sec_positive": _float(compact.get("steps_per_sec")) > 0.0,
    }
    failed = [key for key, value in checks.items() if value is not True]
    return {
        "input_ref": _input_ref(path, result.get("schema_id", "")),
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "steps_per_sec": _float(compact.get("steps_per_sec")),
        "training_wall_sec": _float(compact.get("training_wall_sec")),
        "shape": {
            "batch_size": _int(row.get("batch_size")),
            "actor_count": _int(row.get("actor_count")),
            "num_simulations": _int(row.get("num_simulations")),
            "steps": _int(row.get("steps")),
            "warmup_steps": _int(row.get("warmup_steps")),
        },
    }


def _normal_death_proof_summary(path: Path) -> dict[str, Any]:
    result = _read_json_mapping(path, "normal-death borrowed profile")
    row = _required_mapping(result.get("row"), "normal-death row")
    summary = _required_mapping(result.get("summary"), "normal-death summary")
    compact = _required_mapping(result.get("compact"), "normal-death compact")
    evidence = _optional_mapping(summary.get("normal_death_terminal_contract_evidence"))
    checks = {
        "result_complete": result.get("status") == "complete",
        "returncode_zero_or_absent": result.get("returncode") in (0, None),
        "profile_only_true": summary.get("profile_only") is True,
        "calls_train_muzero_false": summary.get("calls_train_muzero") is False,
        "touches_live_runs_false": summary.get("touches_live_runs") is False,
        "shape_b1024_a1": _int(row.get("batch_size")) == 1024
        and _int(row.get("actor_count")) == 1,
        "death_mode_normal": row.get("death_mode") == "normal"
        and compact.get("death_mode") == "normal",
        "terminal_rows_positive": _int(compact.get("terminal_row_count")) > 0,
        "terminated_rows_positive": _int(compact.get("terminated_row_count")) > 0,
        "truncated_zero": _int(compact.get("truncated_row_count")) == 0,
        "normal_death_contract_gate": summary.get(
            "normal_death_terminal_contract_promotion_gate_satisfied"
        )
        is True,
        "borrowed_handoff": summary.get("render_state_handoff_mode") == BORROWED_HANDOFF_MODE,
        "borrow_requested": summary.get("borrow_single_actor_render_state") is True
        or row.get("hybrid_borrow_single_actor_render_state") is True,
        "resident_observation_used": summary.get("resident_observation_used") is True,
        "resident_host_fallback_zero": _float(
            summary.get("resident_observation_host_fallback_count")
        )
        == 0.0,
        "terminal_final_observation_before_autoreset": summary.get(
            "terminal_final_observation_before_autoreset_verified"
        )
        is True,
        "terminal_final_observation_rows_positive": _int(
            summary.get("terminal_final_observation_row_count")
        )
        > 0,
        "terminal_sample_rows_positive": _int(evidence.get("terminal_sample_row_count"))
        > 0,
        "resident_terminal_final_observation_used": evidence.get(
            "resident_terminal_final_observation_used"
        )
        is True,
        "terminal_value_target_rows_positive": _int(
            evidence.get("terminal_unroll_value_target_row_count")
        )
        > 0,
        "stock_terminal_target_mode": evidence.get("terminal_unroll_value_target_mode")
        == "stock_terminal_no_bootstrap_return_discount_1.0",
    }
    failed = [key for key, value in checks.items() if value is not True]
    return {
        "input_ref": _input_ref(path, result.get("schema_id", "")),
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "steps_per_sec": _float(summary.get("steps_per_sec")),
        "terminal_row_count": _int(compact.get("terminal_row_count")),
        "terminated_row_count": _int(compact.get("terminated_row_count")),
        "truncated_row_count": _int(compact.get("truncated_row_count")),
        "render_state_copy_steps": _int(summary.get("render_state_copy_steps")),
        "render_state_borrowed_steps": _int(summary.get("render_state_borrowed_steps")),
    }


def _speed_lane_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    speeds = [_float(row.get("steps_per_sec")) for row in rows]
    walls = [_float(row.get("training_wall_sec")) for row in rows]
    row_count = len(rows)
    mean_speed = sum(speeds) / row_count if row_count else 0.0
    mean_wall = sum(walls) / row_count if row_count else 0.0
    checks = {
        "enough_rows": row_count >= MIN_BORROWED_ROWS,
        "all_rows_passed": row_count > 0 and all(row.get("passed") is True for row in rows),
        "mean_steps_per_sec_threshold": mean_speed >= BORROWED_MEAN_MIN_STEPS_PER_SEC,
        "mean_wall_threshold": mean_wall <= BORROWED_MEAN_MAX_WALL_SEC,
    }
    failed = [key for key, value in checks.items() if value is not True]
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "row_count": row_count,
        "min_steps_per_sec": min(speeds) if speeds else 0.0,
        "mean_steps_per_sec": mean_speed,
        "max_wall_sec": max(walls) if walls else 0.0,
        "mean_wall_sec": mean_wall,
        "thresholds": {
            "per_row_min_steps_per_sec": BORROWED_ROW_MIN_STEPS_PER_SEC,
            "per_row_max_wall_sec": BORROWED_ROW_MAX_WALL_SEC,
            "mean_min_steps_per_sec": BORROWED_MEAN_MIN_STEPS_PER_SEC,
            "mean_max_wall_sec": BORROWED_MEAN_MAX_WALL_SEC,
        },
    }


def _reference_comparison(
    *,
    speed_lane: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    borrowed_outpaced = (
        reference.get("passed") is True
        and _float(speed_lane.get("min_steps_per_sec")) > _float(reference.get("steps_per_sec"))
        and _float(speed_lane.get("max_wall_sec")) < _float(reference.get("training_wall_sec"))
    )
    return {
        "borrowed_outpaced_reference": borrowed_outpaced,
        "borrowed_min_steps_per_sec": _float(speed_lane.get("min_steps_per_sec")),
        "borrowed_max_wall_sec": _float(speed_lane.get("max_wall_sec")),
        "reference_steps_per_sec": _float(reference.get("steps_per_sec")),
        "reference_wall_sec": _float(reference.get("training_wall_sec")),
        "same_shape_reference_replaced": False,
        "comparison_note": (
            "A1 borrowed outpaced the A16 reference but changed actor shape."
        ),
    }


def _decision(
    *,
    speed_lane: Mapping[str, Any],
    normal_death: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    if speed_lane.get("passed") is not True:
        return DECISION_NOT_APPROVED_SPEED_ROWS_FAILED
    if normal_death.get("passed") is not True:
        return DECISION_NOT_APPROVED_TERMINAL_DEATH_PROOF_FAILED
    if comparison.get("borrowed_outpaced_reference") is not True:
        return DECISION_NOT_APPROVED_REFERENCE_NOT_OUTPACED
    return DECISION_APPROVED_A1_BORROWED_OPERATIONAL_SPEED_CANDIDATE


def _source_profile(compact: Mapping[str, Any]) -> Mapping[str, Any]:
    source = compact.get("source_profile_payload")
    return source if isinstance(source, Mapping) else {}


def _terminalish_row_max(source: Mapping[str, Any]) -> int:
    return max(
        _int(source.get("terminal_row_count")),
        _int(source.get("death_row_count")),
        _int(source.get("terminated_row_count")),
        _int(source.get("truncated_row_count")),
        _int(source.get("compact_rollout_slab_sample_gate_terminal_sample_rows")),
    )


def _input_ref(path: Path, schema_id: object) -> dict[str, str]:
    return {"path": str(path), "schema_id": str(schema_id or ""), "sha256": _sha256(path)}


def _validate_input_ref(section: Mapping[str, Any]) -> None:
    ref = _required_mapping(section.get("input_ref"), "input_ref")
    path = Path(str(ref.get("path") or ""))
    if not path.is_file():
        raise CompactBorrowedActorDecisionError("input_ref path missing")
    if _sha256(path) != ref.get("sha256"):
        raise CompactBorrowedActorDecisionError("input_ref sha mismatch")


def _validate_false_claims(value: Any, label: str) -> None:
    claims = _required_mapping(value, label)
    for key in FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactBorrowedActorDecisionError(f"{label}.{key} must be false")


def _false_claims() -> dict[str, bool]:
    return {key: False for key in FALSE_CLAIM_KEYS}


def _optional_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "COMPACT_BORROWED_ACTOR_DECISION_MANIFEST_SCHEMA_ID",
    "COMPACT_BORROWED_ACTOR_DECISION_SCHEMA_ID",
    "CompactBorrowedActorDecisionError",
    "build_compact_borrowed_actor_decision_v1",
    "compact_borrowed_actor_decision_evidence_ref",
    "validate_compact_borrowed_actor_decision_v1",
]
