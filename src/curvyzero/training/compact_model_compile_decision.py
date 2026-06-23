"""Hash-bound decision report for compact Torch model compile."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.training.compact_compile_eager_speed_pair import (
    DECISION_APPROVED as SPEED_PAIR_DECISION_APPROVED,
)
from curvyzero.training.compact_compile_eager_speed_pair import (
    COMPACT_COMPILE_EAGER_SPEED_PAIR_SCHEMA_ID,
)
from curvyzero.training.compact_compile_eager_speed_pair import (
    validate_compact_compile_eager_speed_pair_v1,
)
from curvyzero.training.compact_speed_row_floor_bundle import FALSE_CLAIM_KEYS
from curvyzero.training.compact_speed_row_floor_bundle import _json_sha256
from curvyzero.training.compact_speed_row_floor_bundle import _read_json_mapping
from curvyzero.training.compact_speed_row_floor_bundle import _require_non_empty
from curvyzero.training.compact_speed_row_floor_bundle import _required_mapping
from curvyzero.training.compact_speed_row_floor_bundle import _sha256


COMPACT_MODEL_COMPILE_DECISION_SCHEMA_ID = (
    "curvyzero_compact_model_compile_decision/v1"
)
COMPACT_MODEL_COMPILE_DECISION_MANIFEST_SCHEMA_ID = (
    "curvyzero_compact_model_compile_decision_manifest/v1"
)
COMPACT_MODEL_COMPILE_DECISION_EVIDENCE_REF_PREFIX = (
    "compact_model_compile_decision:"
)

STATUS_COMPLETE = "compact_model_compile_decision_complete"

DECISION_APPROVED_SPEED_CANDIDATE = (
    "approved_model_compile_default_speed_candidate"
)
DECISION_PARK_SPEED_UNAPPROVED = "park_model_compile_default_speed_unapproved"
DECISION_ROOT_TAPE_FAILED = "not_approved_root_tape_parity_failed"
DECISION_CLOSED_LOOP_FIDELITY_FAILED = (
    "not_approved_closed_loop_fidelity_failed"
)
DECISION_PAIR_REVIEW_FAILED = "not_approved_pair_review_failed"

MODEL_COMPILE_LABEL = "model_compile_default"
PRIMARY_LABEL = "primary"
MODEL_COMPILE_COMPARISON_KEY = "model_compile_default_vs_primary"


class CompactModelCompileDecisionError(ValueError):
    """Raised when a model-compile decision report is malformed."""


def build_compact_model_compile_decision_v1(
    *,
    run_id: str,
    fixed_root_tape_result_path: str | Path,
    post_guard_speed_pair_report_path: str | Path,
    prior_speed_pair_report_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    root_tape = _root_tape_gate_summary(Path(fixed_root_tape_result_path))
    post_guard = _speed_pair_summary(Path(post_guard_speed_pair_report_path))
    prior = (
        _speed_pair_summary(Path(prior_speed_pair_report_path))
        if prior_speed_pair_report_path is not None
        else None
    )
    decision = _decision(root_tape=root_tape, post_guard=post_guard)
    non_claims = _false_claims()
    payload: dict[str, Any] = {
        "schema_id": COMPACT_MODEL_COMPILE_DECISION_SCHEMA_ID,
        "ok": True,
        "status": STATUS_COMPLETE,
        "run_id": _require_non_empty(run_id, "run_id"),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "decision": decision,
        "model_compile_mode": "default",
        "fixed_root_tape_gate": root_tape,
        "post_guard_closed_loop_pair": post_guard,
        "prior_closed_loop_pair": prior,
        "interpretation": _interpretation(
            decision=decision,
            root_tape=root_tape,
            post_guard=post_guard,
            prior=prior,
        ),
        "non_claims": non_claims,
        "attached_claims": {
            "hash_bound_model_compile_decision": True,
            "model_compile_default_root_fidelity_passed": bool(
                root_tape["root_tape_parity_passed"]
            ),
            "model_compile_default_closed_loop_fidelity_passed": bool(
                post_guard["all_action_trajectory_match"]
            ),
            "model_compile_default_speed_default_allowed": (
                decision == DECISION_APPROVED_SPEED_CANDIDATE
            ),
            **non_claims,
        },
    }
    payload["evidence_ref"] = compact_model_compile_decision_evidence_ref(payload)
    validate_compact_model_compile_decision_v1(payload)
    return payload


def validate_compact_model_compile_decision_v1(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_id") != COMPACT_MODEL_COMPILE_DECISION_SCHEMA_ID:
        raise CompactModelCompileDecisionError("model-compile decision schema mismatch")
    if payload.get("status") != STATUS_COMPLETE:
        raise CompactModelCompileDecisionError("model-compile decision status mismatch")
    decision = str(payload.get("decision") or "")
    if decision not in {
        DECISION_APPROVED_SPEED_CANDIDATE,
        DECISION_PARK_SPEED_UNAPPROVED,
        DECISION_ROOT_TAPE_FAILED,
        DECISION_CLOSED_LOOP_FIDELITY_FAILED,
        DECISION_PAIR_REVIEW_FAILED,
    }:
        raise CompactModelCompileDecisionError("unknown model-compile decision")
    _validate_false_claims(payload.get("non_claims"), "non_claims")
    _validate_false_claims(payload.get("attached_claims"), "attached_claims")
    root_tape = _required_mapping(
        payload.get("fixed_root_tape_gate"),
        "fixed_root_tape_gate",
    )
    post_guard = _required_mapping(
        payload.get("post_guard_closed_loop_pair"),
        "post_guard_closed_loop_pair",
    )
    _validate_input_ref(root_tape, "fixed_root_tape_gate")
    _validate_input_ref(post_guard, "post_guard_closed_loop_pair")
    prior = payload.get("prior_closed_loop_pair")
    if prior is not None:
        _validate_input_ref(
            _required_mapping(prior, "prior_closed_loop_pair"),
            "prior_closed_loop_pair",
        )
    speed_default_allowed = bool(
        _required_mapping(payload.get("attached_claims"), "attached_claims").get(
            "model_compile_default_speed_default_allowed"
        )
    )
    if speed_default_allowed != (decision == DECISION_APPROVED_SPEED_CANDIDATE):
        raise CompactModelCompileDecisionError(
            "model compile speed-default claim must match decision"
        )
    if decision == DECISION_APPROVED_SPEED_CANDIDATE:
        if root_tape.get("root_tape_parity_passed") is not True:
            raise CompactModelCompileDecisionError(
                "approved decision requires root-tape parity"
            )
        if post_guard.get("speed_claim_allowed") is not True:
            raise CompactModelCompileDecisionError(
                "approved decision requires post-guard speed approval"
            )
    expected_ref = compact_model_compile_decision_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactModelCompileDecisionError(
            "model-compile decision evidence_ref mismatch"
        )


def compact_model_compile_decision_evidence_ref(payload: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key != "evidence_ref"
    }
    run_id = _require_non_empty(payload.get("run_id"), "run_id")
    digest = _json_sha256(body)
    return f"{COMPACT_MODEL_COMPILE_DECISION_EVIDENCE_REF_PREFIX}{run_id}:sha256={digest}"


def _root_tape_gate_summary(path: Path) -> dict[str, Any]:
    result = _read_json_mapping(path, "fixed-root model-compile result")
    summary = _required_mapping(result.get("summary"), "fixed-root summary")
    comparison_prefix = f"compact_root_tape_{MODEL_COMPILE_COMPARISON_KEY}"
    backend_prefix = f"compact_root_tape_backend_{MODEL_COMPILE_LABEL}"
    record_count = _int(summary.get(f"{comparison_prefix}_record_count"))
    used_count = _int(summary.get(f"{backend_prefix}_model_compile_used_count"))
    requested_count = _int(
        summary.get(f"{backend_prefix}_model_compile_requested_count")
    )
    service_labels = summary.get("compact_root_tape_service_labels")
    service_labels = service_labels if isinstance(service_labels, list) else []
    parity_checks = {
        "status_complete": result.get("status") == "complete",
        "returncode_zero": _int(result.get("returncode")) == 0,
        "compare_enabled": summary.get("compact_root_tape_compare_enabled") is True,
        "profile_only": summary.get("profile_only") is True,
        "metadata_profile_only": (
            summary.get("compact_root_tape_metadata_profile_only") is True
        ),
        "calls_train_muzero_false": summary.get("calls_train_muzero") is False,
        "touches_live_runs_false": summary.get("touches_live_runs") is False,
        "metadata_calls_train_muzero_false": (
            summary.get("compact_root_tape_metadata_calls_train_muzero") is False
        ),
        "root_noise_zero": float(
            summary.get("compact_root_tape_metadata_root_noise_weight", -1.0)
        )
        == 0.0,
        "service_labels_present": (
            PRIMARY_LABEL in service_labels and MODEL_COMPILE_LABEL in service_labels
        ),
        "record_count_positive": record_count > 0,
        "action_match_exact": float(
            summary.get(f"{comparison_prefix}_action_match_fraction", -1.0)
        )
        == 1.0,
        "visit_l1_zero": float(summary.get(f"{comparison_prefix}_visit_l1_max", -1.0))
        == 0.0,
        "root_value_diff_zero": float(
            summary.get(f"{comparison_prefix}_root_value_abs_diff_max", -1.0)
        )
        == 0.0,
        "model_compile_requested_all": requested_count == record_count,
        "model_compile_used_all": used_count == record_count,
    }
    failed = [key for key, value in parity_checks.items() if value is not True]
    return {
        "input_ref": _input_ref(path, result.get("schema_id", "")),
        "root_tape_parity_passed": not failed,
        "checks": parity_checks,
        "failed_checks": failed,
        "service_labels": list(service_labels),
        "record_count": record_count,
        "skipped_record_count": _int(summary.get("compact_root_tape_skipped_record_count")),
        "active_root_count": _int(summary.get(f"{comparison_prefix}_active_root_count")),
        "action_match_fraction": float(
            summary.get(f"{comparison_prefix}_action_match_fraction", 0.0)
        ),
        "visit_l1_mean": float(summary.get(f"{comparison_prefix}_visit_l1_mean", 0.0)),
        "visit_l1_max": float(summary.get(f"{comparison_prefix}_visit_l1_max", 0.0)),
        "root_value_abs_diff_mean": float(
            summary.get(f"{comparison_prefix}_root_value_abs_diff_mean", 0.0)
        ),
        "root_value_abs_diff_max": float(
            summary.get(f"{comparison_prefix}_root_value_abs_diff_max", 0.0)
        ),
        "model_compile_requested_count": requested_count,
        "model_compile_used_count": used_count,
        "model_compile_cache_hit_count": _int(
            summary.get(f"{backend_prefix}_model_compile_cache_hit_count")
        ),
        "model_compile_runtime_status_counts": dict(
            _required_mapping(
                summary.get(f"{backend_prefix}_model_compile_runtime_status_counts"),
                "model_compile_runtime_status_counts",
            )
        ),
        "primary_run_sec": float(
            summary.get(f"compact_root_tape_backend_{PRIMARY_LABEL}_run_sec", 0.0)
        ),
        "model_compile_run_sec": float(summary.get(f"{backend_prefix}_run_sec", 0.0)),
        "root_tape_is_speed_measurement": False,
    }


def _speed_pair_summary(path: Path) -> dict[str, Any]:
    payload = _read_json_mapping(path, "compile/eager speed-pair report")
    validate_compact_compile_eager_speed_pair_v1(payload)
    if payload.get("schema_id") != COMPACT_COMPILE_EAGER_SPEED_PAIR_SCHEMA_ID:
        raise CompactModelCompileDecisionError("speed-pair schema mismatch")
    aggregate = _required_mapping(payload.get("aggregate"), "speed-pair aggregate")
    return {
        "input_ref": _input_ref(path, payload.get("schema_id", "")),
        "run_id": _require_non_empty(payload.get("run_id"), "speed-pair run_id"),
        "decision": str(aggregate.get("decision") or ""),
        "speed_claim_allowed": bool(aggregate.get("speed_claim_allowed")),
        "pair_count": _int(aggregate.get("pair_count")),
        "all_same_denominator": aggregate.get("all_same_denominator") is True,
        "all_safety_checks_passed": (
            aggregate.get("all_safety_checks_passed") is True
        ),
        "all_compile_faster": aggregate.get("all_compile_faster") is True,
        "all_action_trajectory_match": (
            aggregate.get("all_action_trajectory_match") is True
        ),
        "minimum_wall_win_fraction": float(
            aggregate.get("minimum_wall_win_fraction", 0.0)
        ),
        "maximum_wall_win_fraction": float(
            aggregate.get("maximum_wall_win_fraction", 0.0)
        ),
        "bucket_read": _bucket_read(payload),
        "non_claims": dict(_required_mapping(payload.get("non_claims"), "non_claims")),
    }


def _bucket_read(payload: Mapping[str, Any]) -> dict[str, Any]:
    pairs = payload.get("pairs")
    pairs = pairs if isinstance(pairs, list) else []
    service_total_improved = 0
    model_sec_improved = 0
    comparable_model_sec = 0
    wall_faster = 0
    service_total_deltas: list[float] = []
    model_sec_deltas: list[float] = []
    for pair in pairs:
        pair_map = _required_mapping(pair, "speed-pair pair")
        eager = _required_mapping(pair_map.get("eager"), "eager")
        compiled = _required_mapping(pair_map.get("compile"), "compile")
        eager_timing = _required_mapping(eager.get("timing_buckets"), "eager timing")
        compile_timing = _required_mapping(
            compiled.get("timing_buckets"),
            "compile timing",
        )
        eager_service = float(eager_timing.get("search_service_total_sec", 0.0))
        compile_service = float(compile_timing.get("search_service_total_sec", 0.0))
        service_delta = compile_service - eager_service
        service_total_deltas.append(service_delta)
        if service_delta < 0.0:
            service_total_improved += 1
        eager_model = float(eager_timing.get("model_sec", 0.0))
        compile_model = float(compile_timing.get("model_sec", 0.0))
        if eager_model > 0.0 or compile_model > 0.0:
            comparable_model_sec += 1
            model_delta = compile_model - eager_model
            model_sec_deltas.append(model_delta)
            if model_delta < 0.0:
                model_sec_improved += 1
        speed = _required_mapping(pair_map.get("speed_check"), "speed_check")
        if float(speed.get("wall_win_fraction", 0.0)) > 0.0:
            wall_faster += 1
    pair_count = len(pairs)
    return {
        "pair_count": pair_count,
        "service_total_improved_pair_count": service_total_improved,
        "model_sec_comparable_pair_count": comparable_model_sec,
        "model_sec_improved_pair_count": model_sec_improved,
        "wall_faster_pair_count": wall_faster,
        "service_total_delta_sec_min": min(service_total_deltas)
        if service_total_deltas
        else 0.0,
        "service_total_delta_sec_max": max(service_total_deltas)
        if service_total_deltas
        else 0.0,
        "model_sec_delta_sec_min": min(model_sec_deltas) if model_sec_deltas else 0.0,
        "model_sec_delta_sec_max": max(model_sec_deltas) if model_sec_deltas else 0.0,
        "service_model_buckets_improved_all_pairs": (
            pair_count > 0
            and service_total_improved == pair_count
            and (
                comparable_model_sec == 0
                or model_sec_improved == comparable_model_sec
            )
        ),
    }


def _decision(
    *,
    root_tape: Mapping[str, Any],
    post_guard: Mapping[str, Any],
) -> str:
    if root_tape.get("root_tape_parity_passed") is not True:
        return DECISION_ROOT_TAPE_FAILED
    if post_guard.get("speed_claim_allowed") is True and (
        post_guard.get("decision") == SPEED_PAIR_DECISION_APPROVED
    ):
        return DECISION_APPROVED_SPEED_CANDIDATE
    if post_guard.get("all_action_trajectory_match") is not True:
        return DECISION_CLOSED_LOOP_FIDELITY_FAILED
    if (
        post_guard.get("all_same_denominator") is True
        and post_guard.get("all_safety_checks_passed") is True
        and post_guard.get("all_compile_faster") is not True
    ):
        return DECISION_PARK_SPEED_UNAPPROVED
    return DECISION_PAIR_REVIEW_FAILED


def _interpretation(
    *,
    decision: str,
    root_tape: Mapping[str, Any],
    post_guard: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    bucket_read = _required_mapping(post_guard.get("bucket_read"), "bucket_read")
    return {
        "root_tape_fidelity_read": (
            "compiled_and_eager_outputs_match_on_checked_fixed_roots"
            if root_tape.get("root_tape_parity_passed") is True
            else "compiled_and_eager_fixed_root_outputs_not_trusted"
        ),
        "closed_loop_fidelity_read": (
            "post_guard_action_and_trajectory_checksums_match"
            if post_guard.get("all_action_trajectory_match") is True
            else "post_guard_action_or_trajectory_checksums_diverge"
        ),
        "speed_read": (
            "service_model_buckets_improve_but_wall_speed_not_repeatable"
            if decision == DECISION_PARK_SPEED_UNAPPROVED
            and bucket_read.get("service_model_buckets_improved_all_pairs") is True
            else (
                "post_guard_pair_approves_narrow_speed_candidate"
                if decision == DECISION_APPROVED_SPEED_CANDIDATE
                else "no_speed_default_allowed"
            )
        ),
        "prior_drift_was_seen": (
            bool(prior)
            and prior.get("all_action_trajectory_match") is not True
        ),
        "recommended_next_action": (
            "park_model_compile_default_as_optional_until_a_warmed_repeated_study_shows_end_to_end_wall_win"
            if decision == DECISION_PARK_SPEED_UNAPPROVED
            else (
                "review_for_limited_non_production_enablement"
                if decision == DECISION_APPROVED_SPEED_CANDIDATE
                else "do_not_default_model_compile"
            )
        ),
    }


def _input_ref(path: Path, schema_id: object) -> dict[str, str]:
    return {
        "path": str(path),
        "schema_id": str(schema_id or ""),
        "sha256": _sha256(path),
    }


def _validate_input_ref(section: Mapping[str, Any], label: str) -> None:
    ref = _required_mapping(section.get("input_ref"), f"{label}.input_ref")
    path = Path(str(ref.get("path") or ""))
    if not path.is_file():
        raise CompactModelCompileDecisionError(f"{label}.input_ref missing")
    if _sha256(path) != ref.get("sha256"):
        raise CompactModelCompileDecisionError(f"{label}.input_ref sha mismatch")


def _validate_false_claims(value: Any, label: str) -> None:
    claims = _required_mapping(value, label)
    for key in FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactModelCompileDecisionError(f"{label}.{key} must be false")


def _false_claims() -> dict[str, bool]:
    return {key: False for key in FALSE_CLAIM_KEYS}


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
