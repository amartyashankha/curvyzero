"""Hash-bound OPT-074 decision packet for compact Torch service buckets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from pathlib import Path
from typing import Any

from curvyzero.training.compact_speed_row_floor_bundle import (
    COMPACT_SPEED_ROW_FLOOR_BUNDLE_SCHEMA_ID,
)
from curvyzero.training.compact_speed_row_floor_bundle import FALSE_CLAIM_KEYS
from curvyzero.training.compact_speed_row_floor_bundle import ROLE_COMPACT_TORCH_SIBLING
from curvyzero.training.compact_speed_row_floor_bundle import _json_sha256
from curvyzero.training.compact_speed_row_floor_bundle import _read_json_mapping
from curvyzero.training.compact_speed_row_floor_bundle import _require_non_empty
from curvyzero.training.compact_speed_row_floor_bundle import _required_mapping
from curvyzero.training.compact_speed_row_floor_bundle import _sha256
from curvyzero.training.compact_speed_row_floor_bundle import (
    validate_compact_speed_row_floor_bundle_v1,
)
from curvyzero.training.compact_model_compile_decision import (
    validate_compact_model_compile_decision_v1,
)


COMPACT_TORCH_SERVICE_BUCKET_DECISION_SCHEMA_ID = (
    "curvyzero_compact_torch_service_bucket_decision/v1"
)
COMPACT_TORCH_SERVICE_BUCKET_DECISION_MANIFEST_SCHEMA_ID = (
    "curvyzero_compact_torch_service_bucket_decision_manifest/v1"
)
COMPACT_TORCH_SERVICE_BUCKET_DECISION_EVIDENCE_REF_PREFIX = (
    "compact_torch_service_bucket_decision:"
)

STATUS_COMPLETE = "compact_torch_service_bucket_decision_complete"

DECISION_NEEDS_MORE_MEASUREMENT = "needs_more_measurement"
DECISION_SELECT_INITIAL_MODEL_FORWARD = "select_initial_model_forward_path"
DECISION_SELECT_REPLAY_INDEX_OR_SAMPLE = "select_replay_index_or_sample_gate"
DECISION_SELECT_RECURRENT_DEFERRAL_DESIGN = "select_recurrent_deferral_design_only"
DECISION_SELECT_ACTOR_ENV_BOUNDARY = "select_actor_env_boundary_study"
DECISION_MIXED_OR_NOISY = "mixed_or_noisy_no_code_target"

TARGET_NONE = "no_code_target_selected"
TARGET_INITIAL_MODEL_FORWARD = "initial_model_forward_compile_safe_path"
TARGET_REPLAY_INDEX_OR_SAMPLE = "replay_index_or_sample_gate"
TARGET_RECURRENT_DEFERRAL_DESIGN = "recurrent_deferral_design_only"
TARGET_ACTOR_ENV_BOUNDARY = "actor_env_boundary_study"

_REQUIRED_CUDA_EVENT_TIMING_FIELDS = (
    "search_service_initial_inference_cuda_event_sec",
    "search_service_tree_recurrent_inference_cuda_event_sec",
    "search_service_tree_cuda_event_sec",
)
_REQUIRED_DIRECT_CORE_CUDA_EVENT_TIMING_FIELDS = (
    "search_service_initial_inference_representation_cuda_event_sec",
    "search_service_initial_inference_prediction_cuda_event_sec",
    "search_service_initial_inference_direct_core_cuda_event_sec",
)
_INITIAL_INFERENCE_MODES = ("model_method", "direct_core")


@dataclass(frozen=True)
class ServiceBucketSupportInput:
    role: str
    path: str | Path


class CompactTorchServiceBucketDecisionError(ValueError):
    """Raised when the OPT-074 service-bucket decision would overclaim."""


def build_compact_torch_service_bucket_decision_v1(
    *,
    run_id: str,
    canonical_floor_bundle_path: str | Path,
    repeat_floor_bundle_paths: Iterable[str | Path] = (),
    support_floor_bundles: Iterable[ServiceBucketSupportInput] = (),
    compile_decision_report_path: str | Path | None = None,
    min_repeat_count: int = 2,
    recurrent_refresh_guard_plan_present: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a small, hash-bound target-selection packet from floor bundles."""

    canonical = _bundle_summary(
        role="canonical",
        path=Path(canonical_floor_bundle_path),
    )
    repeats = [
        _bundle_summary(role=f"repeat_{index}", path=Path(path))
        for index, path in enumerate(repeat_floor_bundle_paths, start=1)
    ]
    supports = [
        _bundle_summary(role=input_.role, path=Path(input_.path))
        for input_ in support_floor_bundles
    ]
    compile_decision = (
        _compile_decision_summary(Path(compile_decision_report_path))
        if compile_decision_report_path is not None
        else None
    )
    _validate_evidence_guards(canonical=canonical, repeats=repeats, supports=supports)
    evidence = [canonical, *repeats]
    bucket_summary = _bucket_summary(canonical=canonical, repeats=repeats, supports=supports)
    guard_checks = _guard_checks(
        canonical=canonical,
        repeats=repeats,
        bucket_summary=bucket_summary,
        min_repeat_count=int(min_repeat_count),
        recurrent_refresh_guard_plan_present=bool(recurrent_refresh_guard_plan_present),
    )
    decision, selected_target = _decision(
        bucket_summary=bucket_summary,
        guard_checks=guard_checks,
    )
    non_claims = _false_claims()
    payload: dict[str, Any] = {
        "schema_id": COMPACT_TORCH_SERVICE_BUCKET_DECISION_SCHEMA_ID,
        "ok": True,
        "status": STATUS_COMPLETE,
        "run_id": _require_non_empty(run_id, "run_id"),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "decision": decision,
        "selected_next_target": selected_target,
        "min_repeat_count": int(min_repeat_count),
        "input_refs": _input_refs(
            canonical=canonical,
            repeats=repeats,
            supports=supports,
            compile_decision=compile_decision,
        ),
        "canonical_read": canonical,
        "repeat_read": {
            "bundle_count": len(repeats),
            "bundles": repeats,
            "same_denominator_evidence_count": int(
                guard_checks["same_denominator_evidence_count"]
            ),
            "wall_sec_min": min(float(item["wall_sec"]) for item in evidence),
            "wall_sec_max": max(float(item["wall_sec"]) for item in evidence),
            "steps_per_sec_min": min(float(item["steps_per_sec"]) for item in evidence),
            "steps_per_sec_max": max(float(item["steps_per_sec"]) for item in evidence),
        },
        "support_read": supports,
        "compile_decision": compile_decision,
        "bucket_summary": bucket_summary,
        "guard_checks": guard_checks,
        "parked_options": _parked_options(
            recurrent_refresh_guard_plan_present=bool(
                recurrent_refresh_guard_plan_present
            ),
        ),
        "recommended_next_action": _recommended_next_action(
            decision=decision,
            selected_target=selected_target,
            guard_checks=guard_checks,
        ),
        "non_claims": non_claims,
        "attached_claims": {
            "service_bucket_decision_hash_bound": True,
            "target_selection_claim": decision != DECISION_NEEDS_MORE_MEASUREMENT,
            "next_optimization_code_allowed": decision
            in {
                DECISION_SELECT_INITIAL_MODEL_FORWARD,
                DECISION_SELECT_REPLAY_INDEX_OR_SAMPLE,
            },
            **non_claims,
        },
    }
    payload["evidence_ref"] = compact_torch_service_bucket_decision_evidence_ref(
        payload
    )
    validate_compact_torch_service_bucket_decision_v1(payload)
    return payload


def validate_compact_torch_service_bucket_decision_v1(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_id") != COMPACT_TORCH_SERVICE_BUCKET_DECISION_SCHEMA_ID:
        raise CompactTorchServiceBucketDecisionError(
            "service-bucket decision schema mismatch"
        )
    if payload.get("status") != STATUS_COMPLETE:
        raise CompactTorchServiceBucketDecisionError(
            "service-bucket decision status mismatch"
        )
    decision = str(payload.get("decision") or "")
    if decision not in {
        DECISION_NEEDS_MORE_MEASUREMENT,
        DECISION_SELECT_INITIAL_MODEL_FORWARD,
        DECISION_SELECT_REPLAY_INDEX_OR_SAMPLE,
        DECISION_SELECT_RECURRENT_DEFERRAL_DESIGN,
        DECISION_SELECT_ACTOR_ENV_BOUNDARY,
        DECISION_MIXED_OR_NOISY,
    }:
        raise CompactTorchServiceBucketDecisionError("unknown service-bucket decision")
    _validate_false_claims(payload.get("non_claims"), "non_claims")
    _validate_false_claims(payload.get("attached_claims"), "attached_claims")
    _validate_input_refs(payload.get("input_refs"))
    canonical = _required_mapping(payload.get("canonical_read"), "canonical_read")
    _validate_input_ref(canonical, "canonical_read")
    repeat_read = _required_mapping(payload.get("repeat_read"), "repeat_read")
    repeat_bundles = repeat_read.get("bundles")
    if not isinstance(repeat_bundles, list):
        raise CompactTorchServiceBucketDecisionError("repeat_read.bundles must be a list")
    for index, entry in enumerate(repeat_bundles):
        _validate_input_ref(
            _required_mapping(entry, f"repeat_read.bundles[{index}]"),
            f"repeat_read.bundles[{index}]",
        )
    for label in ("bucket_summary", "guard_checks", "parked_options"):
        _required_mapping(payload.get(label), label)
    support = payload.get("support_read")
    if not isinstance(support, list):
        raise CompactTorchServiceBucketDecisionError("support_read must be a list")
    for index, entry in enumerate(support):
        _validate_input_ref(_required_mapping(entry, f"support_read[{index}]"), "support")
    compile_decision = payload.get("compile_decision")
    if compile_decision is not None:
        _validate_input_ref(
            _required_mapping(compile_decision, "compile_decision"),
            "compile_decision",
        )
    attached = _required_mapping(payload.get("attached_claims"), "attached_claims")
    code_allowed = bool(attached.get("next_optimization_code_allowed"))
    if code_allowed and decision not in {
        DECISION_SELECT_INITIAL_MODEL_FORWARD,
        DECISION_SELECT_REPLAY_INDEX_OR_SAMPLE,
    }:
        raise CompactTorchServiceBucketDecisionError(
            "code permission must match selected implementation target"
        )
    expected_ref = compact_torch_service_bucket_decision_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactTorchServiceBucketDecisionError(
            "service-bucket decision evidence_ref mismatch"
        )


def compact_torch_service_bucket_decision_evidence_ref(
    payload: Mapping[str, Any],
) -> str:
    body = {key: value for key, value in payload.items() if key != "evidence_ref"}
    run_id = _require_non_empty(payload.get("run_id"), "run_id")
    digest = _json_sha256(body)
    return (
        f"{COMPACT_TORCH_SERVICE_BUCKET_DECISION_EVIDENCE_REF_PREFIX}"
        f"{run_id}:sha256={digest}"
    )


def _bundle_summary(*, role: str, path: Path) -> dict[str, Any]:
    bundle = _read_json_mapping(path, f"{role} floor bundle")
    validate_compact_speed_row_floor_bundle_v1(bundle)
    if bundle.get("schema_id") != COMPACT_SPEED_ROW_FLOOR_BUNDLE_SCHEMA_ID:
        raise CompactTorchServiceBucketDecisionError(f"{role} bundle schema mismatch")
    rows = _required_mapping(bundle.get("rows"), f"{role} rows")
    compact = _required_mapping(
        rows.get(ROLE_COMPACT_TORCH_SIBLING),
        f"{role} compact row",
    )
    comparisons = _required_mapping(bundle.get("comparisons"), f"{role} comparisons")
    engineering = _required_mapping(
        bundle.get("engineering_read"),
        f"{role} engineering_read",
    )
    timing = _required_mapping(compact.get("timing_buckets"), f"{role} timing")
    flags = _required_mapping(compact.get("search_service_flags"), f"{role} flags")
    shape = _required_mapping(compact.get("shape"), f"{role} shape")
    profile = _compact_profile_telemetry(compact)
    return {
        "role": _require_non_empty(role, "role"),
        "input_ref": _input_ref(path, bundle.get("schema_id", "")),
        "bundle_run_id": str(bundle.get("run_id") or ""),
        "ok": bundle.get("ok") is True,
        "status": str(bundle.get("status") or ""),
        "same_denominator": _required_mapping(
            bundle.get("denominator_check"),
            f"{role} denominator_check",
        ).get("same_denominator")
        is True,
        "decision_confidence": str(engineering.get("decision_confidence") or ""),
        "engineering_classification": str(engineering.get("classification") or ""),
        "search_dominance_claim": engineering.get("search_dominance_claim") is True,
        "shape": dict(shape),
        "wall_sec": float(comparisons.get("compact_torch_wall_sec", 0.0)),
        "steps_per_sec": float(comparisons.get("compact_torch_steps_per_sec", 0.0)),
        "fixed_floor_steps_per_sec": float(
            comparisons.get("fixed_floor_steps_per_sec", 0.0)
        ),
        "compact_torch_vs_floor_wall_ratio": float(
            comparisons.get("compact_torch_vs_floor_wall_ratio", 0.0)
        ),
        "compact_torch_vs_floor_wall_delta_sec": float(
            comparisons.get("compact_torch_vs_floor_wall_delta_sec", 0.0)
        ),
        "search_delta_sec": float(comparisons.get("search_delta_sec", 0.0)),
        "learner_sample_delta_sec": float(
            comparisons.get("learner_sample_delta_sec", 0.0)
        ),
        "actor_wall_delta_sec": float(comparisons.get("actor_wall_delta_sec", 0.0)),
        "replay_index_rows_build_delta_sec": float(
            _required_mapping(
                comparisons.get("high_level_deltas"),
                f"{role} high_level_deltas",
            ).get("slab_replay_index_rows_build_delta_sec", 0.0)
        ),
        "slab_commit_previous_delta_sec": float(
            _required_mapping(
                comparisons.get("high_level_deltas"),
                f"{role} high_level_deltas",
            ).get("slab_commit_previous_delta_sec", 0.0)
        ),
        "dispatch_residual_delta_sec": float(
            _required_mapping(
                comparisons.get("high_level_deltas"),
                f"{role} high_level_deltas",
            ).get("slab_search_dispatch_residual_delta_sec", 0.0)
        ),
        "timing_buckets": dict(timing),
        "search_service_flags": dict(flags),
        "profile_telemetry": profile,
        "non_claims": dict(_required_mapping(bundle.get("non_claims"), "non_claims")),
    }


def _validate_evidence_guards(
    *,
    canonical: Mapping[str, Any],
    repeats: list[Mapping[str, Any]],
    supports: list[Mapping[str, Any]],
) -> None:
    checks = _canonical_guard_checks(canonical)
    failed = [key for key, passed in checks.items() if passed is not True]
    if failed:
        raise CompactTorchServiceBucketDecisionError(
            "canonical service-bucket guard failed: " + ", ".join(failed)
        )
    for bundle in [*repeats, *supports]:
        role = str(bundle.get("role") or "evidence")
        checks = _canonical_guard_checks(bundle)
        failed = [key for key, passed in checks.items() if passed is not True]
        if failed:
            raise CompactTorchServiceBucketDecisionError(
                f"{role} service-bucket guard failed: " + ", ".join(failed)
            )


def _canonical_guard_checks(bundle: Mapping[str, Any]) -> dict[str, bool]:
    timing = _required_mapping(bundle.get("timing_buckets"), "timing_buckets")
    flags = _required_mapping(bundle.get("search_service_flags"), "search_service_flags")
    shape = _required_mapping(bundle.get("shape"), "shape")
    profile = _required_mapping(bundle.get("profile_telemetry"), "profile_telemetry")
    steps = float(shape.get("steps", 0.0))
    training_before = profile.get("compact_torch_search_model_training_before_inference")
    training_after = profile.get("compact_torch_search_model_training_after_inference")
    initial_mode_requested = str(
        flags.get("initial_inference_mode_requested") or ""
    )
    initial_mode_effective = str(
        flags.get("initial_inference_mode_effective") or ""
    )
    return {
        "bundle_ok": bundle.get("ok") is True,
        "same_denominator": bundle.get("same_denominator") is True,
        "compact_search_service": True,
        "fixed_floor_bound": True,
        "fast_path_count_matches_steps": float(
            timing.get("search_service_one_simulation_fast_path_count", 0.0)
        )
        == steps,
        "softmax_skip_count_matches_steps": float(
            timing.get(
                "search_service_one_simulation_root_prior_softmax_skipped_count",
                0.0,
            )
        )
        == steps,
        "recurrent_call_count_matches_steps": float(
            timing.get("search_service_recurrent_inference_calls", 0.0)
        )
        == steps,
        "selection_mode_masked_logits": (
            flags.get("one_simulation_selection_mode_last") == "masked_logits_argmax"
        ),
        "action_readback_int16_bytes": (
            flags.get("action_readback_bytes_match_int16_selected_actions") is True
        ),
        "replay_payload_d2h_zero": float(timing.get("replay_payload_d2h_bytes", -1.0))
        == 0.0,
        "committed_replay_payload_d2h_zero": float(
            timing.get("committed_replay_payload_d2h_bytes", -1.0)
        )
        == 0.0,
        "resident_observation_host_fallback_zero": float(
            timing.get("resident_observation_host_fallback_count", 0.0)
        )
        == 0.0,
        "cuda_event_timing_complete": not _incomplete_cuda_event_timing_fields(timing),
        "direct_core_cuda_event_timing_complete": (
            not _incomplete_direct_core_cuda_event_timing_fields(timing)
        ),
        "initial_inference_mode_recorded": (
            initial_mode_requested in _INITIAL_INFERENCE_MODES
            and initial_mode_effective in _INITIAL_INFERENCE_MODES
        ),
        "initial_inference_effective_matches_requested": (
            initial_mode_effective == initial_mode_requested
        ),
        "initial_inference_no_fallback": float(
            flags.get("initial_inference_fallback_count", -1.0)
        )
        == 0.0,
        "model_eval_applied_for_inference": (
            profile.get("compact_torch_search_model_eval_applied_for_inference") is True
        ),
        "model_inference_mode_used": (
            profile.get("compact_torch_search_model_inference_mode_used") is True
        ),
        "model_training_state_recorded": (
            training_before is not None and training_after is not None
        ),
        "model_training_state_restored": training_after == training_before,
        "model_compile_not_requested": (
            profile.get("compact_torch_search_model_compile_requested") is not True
        ),
        "model_compile_not_used": (
            profile.get("compact_torch_search_model_compile_used") is not True
        ),
    }


def _guard_checks(
    *,
    canonical: Mapping[str, Any],
    repeats: list[Mapping[str, Any]],
    bucket_summary: Mapping[str, Any],
    min_repeat_count: int,
    recurrent_refresh_guard_plan_present: bool,
) -> dict[str, Any]:
    canonical_checks = _canonical_guard_checks(canonical)
    same_denominator_evidence_count = 1 + sum(
        1 for item in repeats if item.get("same_denominator") is True
    )
    return {
        "canonical_checks": canonical_checks,
        "canonical_all_checks_passed": all(canonical_checks.values()),
        "same_denominator_evidence_count": same_denominator_evidence_count,
        "min_repeat_count": int(min_repeat_count),
        "repeat_requirement_met": same_denominator_evidence_count
        >= int(min_repeat_count),
        "current_cuda_event_timing_present": bool(
            bucket_summary.get("event_timing_present")
        ),
        "trajectory_different_confidence_limited": (
            canonical.get("decision_confidence")
            == "limited_shape_matched_trajectory_different"
        ),
        "speed_claim_allowed": False,
        "gpu_mechanics_selection_allowed": bool(
            bucket_summary.get("actor_env_dominant_across_evidence") is True
        ),
        "recurrent_deferral_implementation_allowed": bool(
            bucket_summary.get("recurrent_bucket_dominant") is True
            and recurrent_refresh_guard_plan_present
        ),
    }


def _bucket_summary(
    *,
    canonical: Mapping[str, Any],
    repeats: list[Mapping[str, Any]],
    supports: list[Mapping[str, Any]],
) -> dict[str, Any]:
    all_bundles = [canonical, *repeats, *supports]
    event_bundle = _latest_event_bundle(all_bundles)
    source = event_bundle or canonical
    timing = _required_mapping(source.get("timing_buckets"), "bucket timing")
    incomplete_event_fields = _incomplete_cuda_event_timing_fields(timing)
    if incomplete_event_fields:
        role = str(source.get("role") or "event evidence")
        raise CompactTorchServiceBucketDecisionError(
            f"{role} cuda-event timing incomplete: "
            + ", ".join(incomplete_event_fields)
        )
    incomplete_direct_event_fields = _incomplete_direct_core_cuda_event_timing_fields(
        timing
    )
    if incomplete_direct_event_fields:
        role = str(source.get("role") or "event evidence")
        raise CompactTorchServiceBucketDecisionError(
            f"{role} direct-core cuda-event timing incomplete: "
            + ", ".join(incomplete_direct_event_fields)
        )
    initial_event = float(timing.get("search_service_initial_inference_cuda_event_sec", 0.0))
    initial_representation_event = float(
        timing.get(
            "search_service_initial_inference_representation_cuda_event_sec",
            0.0,
        )
    )
    initial_prediction_event = float(
        timing.get(
            "search_service_initial_inference_prediction_cuda_event_sec",
            0.0,
        )
    )
    initial_direct_core_event = float(
        timing.get(
            "search_service_initial_inference_direct_core_cuda_event_sec",
            0.0,
        )
    )
    initial_direct_core_residual = float(
        timing.get(
            "search_service_initial_inference_direct_core_cuda_event_residual_sec",
            0.0,
        )
    )
    recurrent_event = float(
        timing.get("search_service_tree_recurrent_inference_cuda_event_sec", 0.0)
    )
    tree_event = float(timing.get("search_service_tree_cuda_event_sec", 0.0))
    search_delta = float(source.get("search_delta_sec", 0.0))
    sample_delta = float(source.get("learner_sample_delta_sec", 0.0))
    replay_index_delta = float(source.get("replay_index_rows_build_delta_sec", 0.0))
    commit_delta = float(source.get("slab_commit_previous_delta_sec", 0.0))
    actor_delta = float(source.get("actor_wall_delta_sec", 0.0))
    replay_commit_delta = max(replay_index_delta, commit_delta)
    return {
        "source_role": str(source.get("role") or ""),
        "event_timing_present": event_bundle is not None,
        "service_total_sec": float(timing.get("search_service_total_sec", 0.0)),
        "service_action_wall_sec": float(
            timing.get("search_service_action_wall_sec", 0.0)
        ),
        "service_action_residual_sec": float(
            timing.get("search_service_action_residual_sec", 0.0)
        ),
        "service_action_readback_sec": float(
            timing.get("search_service_action_readback_sec", 0.0)
        ),
        "service_core_accounted_sec": float(
            timing.get("search_service_core_accounted_sec", 0.0)
        ),
        "service_core_residual_sec": float(
            timing.get("search_service_core_residual_sec", 0.0)
        ),
        "service_inference_guard_total_sec": float(
            timing.get("search_service_inference_guard_total_sec", 0.0)
        ),
        "model_sec": float(timing.get("model_sec", 0.0)),
        "search_sec": float(timing.get("search_sec", 0.0)),
        "initial_cuda_event_sec": initial_event,
        "initial_representation_cuda_event_sec": initial_representation_event,
        "initial_prediction_cuda_event_sec": initial_prediction_event,
        "initial_direct_core_cuda_event_sec": initial_direct_core_event,
        "initial_direct_core_cuda_event_residual_sec": initial_direct_core_residual,
        "initial_direct_core_cuda_event_split_sec": (
            initial_representation_event + initial_prediction_event
        ),
        "initial_enqueue_sec": float(
            timing.get("search_service_initial_inference_enqueue_sec", 0.0)
        ),
        "initial_sync_sec": float(
            timing.get("search_service_initial_inference_sync_sec", 0.0)
        ),
        "recurrent_cuda_event_sec": recurrent_event,
        "recurrent_enqueue_sec": float(
            timing.get("search_service_tree_recurrent_inference_enqueue_sec", 0.0)
        ),
        "tree_cuda_event_sec": tree_event,
        "tree_total_sec": float(timing.get("search_service_tree_total_sec", 0.0)),
        "tree_policy_build_sec": float(
            timing.get("search_service_tree_policy_build_sec", 0.0)
        ),
        "tree_root_prior_build_sec": float(
            timing.get("search_service_tree_root_prior_build_sec", 0.0)
        ),
        "tree_residual_sec": float(
            timing.get("search_service_tree_residual_sec", 0.0)
        ),
        "root_prior_select_sec": float(
            timing.get("search_service_tree_root_prior_select_sec", 0.0)
        ),
        "fixed_shape_masks_sec": float(
            timing.get("search_service_fixed_shape_masks_sec", 0.0)
        ),
        "slab_commit_previous_sec": float(
            timing.get("slab_commit_previous_sec", 0.0)
        ),
        "slab_replay_index_rows_build_sec": float(
            timing.get("slab_replay_index_rows_build_sec", 0.0)
        ),
        "sample_gate_sec": float(timing.get("sample_gate_sec", 0.0)),
        "search_delta_sec": search_delta,
        "learner_sample_delta_sec": sample_delta,
        "replay_index_rows_build_delta_sec": replay_index_delta,
        "slab_commit_previous_delta_sec": commit_delta,
        "replay_or_commit_delta_sec": replay_commit_delta,
        "dispatch_residual_delta_sec": float(
            source.get("dispatch_residual_delta_sec", 0.0)
        ),
        "actor_wall_delta_sec": actor_delta,
        "initial_model_bucket_dominant": bool(
            initial_event >= 0.5
            and initial_event > recurrent_event * 2.0
            and search_delta > sample_delta
            and search_delta > replay_index_delta
        ),
        "recurrent_bucket_dominant": bool(
            recurrent_event >= 0.5 and recurrent_event >= initial_event
        ),
        "replay_or_sample_dominant": bool(
            max(sample_delta, replay_commit_delta) > search_delta
        ),
        "actor_env_dominant_across_evidence": bool(actor_delta > search_delta),
    }


def _decision(
    *,
    bucket_summary: Mapping[str, Any],
    guard_checks: Mapping[str, Any],
) -> tuple[str, str]:
    if guard_checks.get("canonical_all_checks_passed") is not True:
        return DECISION_MIXED_OR_NOISY, TARGET_NONE
    if guard_checks.get("repeat_requirement_met") is not True:
        return DECISION_NEEDS_MORE_MEASUREMENT, TARGET_NONE
    if guard_checks.get("gpu_mechanics_selection_allowed") is True:
        return DECISION_SELECT_ACTOR_ENV_BOUNDARY, TARGET_ACTOR_ENV_BOUNDARY
    if bucket_summary.get("replay_or_sample_dominant") is True:
        return DECISION_SELECT_REPLAY_INDEX_OR_SAMPLE, TARGET_REPLAY_INDEX_OR_SAMPLE
    if guard_checks.get("recurrent_deferral_implementation_allowed") is True:
        return (
            DECISION_SELECT_RECURRENT_DEFERRAL_DESIGN,
            TARGET_RECURRENT_DEFERRAL_DESIGN,
        )
    if (
        guard_checks.get("current_cuda_event_timing_present") is True
        and bucket_summary.get("initial_model_bucket_dominant") is True
    ):
        return DECISION_SELECT_INITIAL_MODEL_FORWARD, TARGET_INITIAL_MODEL_FORWARD
    return DECISION_MIXED_OR_NOISY, TARGET_NONE


def _recommended_next_action(
    *,
    decision: str,
    selected_target: str,
    guard_checks: Mapping[str, Any],
) -> str:
    if decision == DECISION_SELECT_INITIAL_MODEL_FORWARD:
        return (
            "design_the_next_small_initial_model_forward_or_compile_safe_model_path_"
            "optimization_with_fixed_root_or_action_trajectory_guards"
        )
    if decision == DECISION_SELECT_REPLAY_INDEX_OR_SAMPLE:
        return "decompose_replay_index_build_and_sample_gate_before_more_model_work"
    if decision == DECISION_NEEDS_MORE_MEASUREMENT:
        return "collect_repeat_same_denominator_or_current_cuda_event_rows_before_coding"
    if selected_target == TARGET_RECURRENT_DEFERRAL_DESIGN:
        return "write_refresh_hazard_guard_design_before_any_recurrent_deferral_code"
    if selected_target == TARGET_ACTOR_ENV_BOUNDARY:
        return "start_cpu_oracle_actor_env_boundary_study_not_gpu_rewrite"
    if guard_checks.get("repeat_requirement_met") is not True:
        return "repeat_requirement_not_met"
    return "no_code_target_selected_from_current_evidence"


def _parked_options(*, recurrent_refresh_guard_plan_present: bool) -> dict[str, Any]:
    return {
        "gpu_mechanics": {
            "status": "parked",
            "reason": (
                "selected_action_d2h_is_small_and_actor_env_is_not_the_dominant_"
                "same_denominator_delta"
            ),
        },
        "recurrent_deferral": {
            "status": "parked"
            if not recurrent_refresh_guard_plan_present
            else "design_only",
            "reason": (
                "deferred_recurrent_work_moves_across_action_commit_boundary_and_"
                "requires_pending_payload_model_refresh_guards"
            ),
        },
        "model_compile_default": {
            "status": "parked_optional_off_by_default",
            "reason": "OPT-066 decision says speed_default_allowed_false",
        },
    }


def _latest_event_bundle(
    bundles: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    event_bundles = []
    for bundle in bundles:
        timing = _required_mapping(bundle.get("timing_buckets"), "timing_buckets")
        if _cuda_event_timing_enabled(timing):
            event_bundles.append(bundle)
    return event_bundles[-1] if event_bundles else None


def _cuda_event_timing_enabled(timing: Mapping[str, Any]) -> bool:
    try:
        enabled_count = float(
            timing.get("search_service_cuda_event_timing_enabled_count", 0.0)
        )
    except (TypeError, ValueError):
        return False
    return math.isfinite(enabled_count) and enabled_count > 0.0


def _incomplete_cuda_event_timing_fields(timing: Mapping[str, Any]) -> list[str]:
    if not _cuda_event_timing_enabled(timing):
        return []
    incomplete = []
    for key in _REQUIRED_CUDA_EVENT_TIMING_FIELDS:
        try:
            value = float(timing.get(key))
        except (TypeError, ValueError):
            incomplete.append(key)
            continue
        if not math.isfinite(value) or value <= 0.0:
            incomplete.append(key)
    return incomplete


def _direct_core_initial_inference_used(timing: Mapping[str, Any]) -> bool:
    try:
        used_count = float(
            timing.get("search_service_initial_inference_direct_used_count", 0.0)
        )
    except (TypeError, ValueError):
        return False
    return math.isfinite(used_count) and used_count > 0.0


def _incomplete_direct_core_cuda_event_timing_fields(
    timing: Mapping[str, Any],
) -> list[str]:
    if not _cuda_event_timing_enabled(timing) or not _direct_core_initial_inference_used(
        timing
    ):
        return []
    incomplete = []
    for key in _REQUIRED_DIRECT_CORE_CUDA_EVENT_TIMING_FIELDS:
        try:
            value = float(timing.get(key))
        except (TypeError, ValueError):
            incomplete.append(key)
            continue
        if not math.isfinite(value) or value <= 0.0:
            incomplete.append(key)
    return incomplete


def _compact_profile_telemetry(row: Mapping[str, Any]) -> dict[str, Any]:
    result_ref = _required_mapping(row.get("result_ref"), "compact.result_ref")
    result_path = Path(str(result_ref.get("path") or ""))
    result = _read_json_mapping(result_path, "compact speed-row result")
    compact = _required_mapping(result.get("compact"), "compact result")
    source_profile = _required_mapping(
        compact.get("source_profile_payload"),
        "source_profile_payload",
    )
    last = _required_mapping(
        source_profile.get("compact_rollout_slab_last_telemetry"),
        "compact_rollout_slab_last_telemetry",
    )
    profile = last.get("compact_rollout_slab_profile_telemetry")
    return dict(profile if isinstance(profile, Mapping) else {})


def _compile_decision_summary(path: Path) -> dict[str, Any]:
    payload = _read_json_mapping(path, "compile decision report")
    validate_compact_model_compile_decision_v1(payload)
    return {
        "input_ref": _input_ref(path, payload.get("schema_id", "")),
        "run_id": str(payload.get("run_id") or ""),
        "decision": str(payload.get("decision") or ""),
        "model_compile_default_speed_default_allowed": bool(
            _required_mapping(payload.get("attached_claims"), "attached_claims").get(
                "model_compile_default_speed_default_allowed"
            )
        ),
    }


def _input_refs(
    *,
    canonical: Mapping[str, Any],
    repeats: list[Mapping[str, Any]],
    supports: list[Mapping[str, Any]],
    compile_decision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "canonical_floor_bundle": dict(
            _required_mapping(canonical.get("input_ref"), "canonical input_ref")
        ),
        "repeat_floor_bundles": [
            dict(_required_mapping(item.get("input_ref"), "repeat input_ref"))
            for item in repeats
        ],
        "support_floor_bundles": [
            {
                "role": str(item.get("role") or ""),
                "input_ref": dict(
                    _required_mapping(item.get("input_ref"), "support input_ref")
                ),
            }
            for item in supports
        ],
        "compile_decision_report": (
            dict(
                _required_mapping(
                    compile_decision.get("input_ref"),
                    "compile decision input_ref",
                )
            )
            if compile_decision is not None
            else None
        ),
    }


def _validate_input_refs(value: Any) -> None:
    refs = _required_mapping(value, "input_refs")
    _validate_ref_mapping(
        _required_mapping(
            refs.get("canonical_floor_bundle"),
            "input_refs.canonical_floor_bundle",
        ),
        "input_refs.canonical_floor_bundle",
    )
    repeat_refs = refs.get("repeat_floor_bundles")
    if not isinstance(repeat_refs, list):
        raise CompactTorchServiceBucketDecisionError(
            "input_refs.repeat_floor_bundles must be a list"
        )
    for index, ref in enumerate(repeat_refs):
        _validate_ref_mapping(
            _required_mapping(ref, f"input_refs.repeat_floor_bundles[{index}]"),
            f"input_refs.repeat_floor_bundles[{index}]",
        )
    support_refs = refs.get("support_floor_bundles")
    if not isinstance(support_refs, list):
        raise CompactTorchServiceBucketDecisionError(
            "input_refs.support_floor_bundles must be a list"
        )
    for index, item in enumerate(support_refs):
        item_map = _required_mapping(
            item,
            f"input_refs.support_floor_bundles[{index}]",
        )
        _require_non_empty(item_map.get("role"), "support role")
        _validate_ref_mapping(
            _required_mapping(
                item_map.get("input_ref"),
                f"input_refs.support_floor_bundles[{index}].input_ref",
            ),
            f"input_refs.support_floor_bundles[{index}].input_ref",
        )
    compile_ref = refs.get("compile_decision_report")
    if compile_ref is not None:
        _validate_ref_mapping(
            _required_mapping(
                compile_ref,
                "input_refs.compile_decision_report",
            ),
            "input_refs.compile_decision_report",
        )


def _input_ref(path: Path, schema_id: object) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "schema_id": str(schema_id or ""),
        "sha256": _sha256(path.resolve()),
    }


def _validate_input_ref(section: Mapping[str, Any], label: str) -> None:
    ref = _required_mapping(section.get("input_ref"), f"{label}.input_ref")
    _validate_ref_mapping(ref, f"{label}.input_ref")


def _validate_ref_mapping(ref: Mapping[str, Any], label: str) -> None:
    path = Path(str(ref.get("path") or ""))
    if not path.is_file():
        raise CompactTorchServiceBucketDecisionError(f"{label} missing")
    if _sha256(path) != ref.get("sha256"):
        raise CompactTorchServiceBucketDecisionError(
            f"{label} sha mismatch"
        )


def _validate_false_claims(value: Any, label: str) -> None:
    claims = _required_mapping(value, label)
    for key in FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactTorchServiceBucketDecisionError(f"{label}.{key} must be false")


def _false_claims() -> dict[str, bool]:
    return {key: False for key in FALSE_CLAIM_KEYS}
