"""Hash-bound speed-row floor decomposition for compact-owned trainer rows."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
)
from curvyzero.training.compact_coach_speed_row import (
    validate_compact_coach_speed_row_evidence_matches_report_v1,
)
from curvyzero.training.compact_coach_speed_row import (
    validate_compact_coach_speed_row_evidence_v1,
)


COMPACT_SPEED_ROW_FLOOR_BUNDLE_SCHEMA_ID = (
    "curvyzero_compact_speed_row_floor_bundle/v1"
)
COMPACT_SPEED_ROW_FLOOR_BUNDLE_MANIFEST_SCHEMA_ID = (
    "curvyzero_compact_speed_row_floor_bundle_manifest/v1"
)
COMPACT_SPEED_ROW_FLOOR_BUNDLE_EVIDENCE_REF_PREFIX = "compact_speed_row_floor_bundle:"

STATUS_COMPLETE = "same_denominator_speed_row_floor_bundle_complete"
STATUS_DENOMINATOR_MISMATCH = "same_denominator_speed_row_floor_bundle_mismatch"

ROLE_ACCEPTED_COACH_SPEED_ROW = "accepted_coach_speed_row"
ROLE_COMPACT_TORCH_SIBLING = "compact_torch_search_service_sibling"
ROLE_FIXED_FLOOR_SIBLING = "fixed_no_search_floor_sibling"

SEARCH_KIND_DEVICE_TARGET = "device_target"
SEARCH_KIND_COMPACT_TORCH = "compact_torch_search_service"
SEARCH_KIND_FIXED_SHAPE = "fixed_shape_search_owner"

SPEED_CURRENCY = "compact_trainer_env_steps_per_sec"

FALSE_CLAIM_KEYS = (
    "promotion_claim",
    "training_speedup_claim",
    "live_run_safety_claim",
    "stock_resume_claim",
    "rating_or_promotion_quality_claim",
    "automatic_promotion_allowed",
    "calls_train_muzero",
    "touches_live_runs",
)

DENOMINATOR_KEYS = (
    "candidate_checkpoint_id",
    "hardware_class",
    "speed_currency",
    "batch_size",
    "actor_count",
    "steps",
    "warmup_steps",
    "sample_batch_size",
    "sample_interval",
    "replay_pair_capacity",
    "learner_device",
    "learner_train_steps",
    "num_simulations",
    "model_identity_scope",
)

DOMINANT_WALL_DELTA_KEYS = (
    "compact_rollout_slab_delta_sec",
    "actor_wall_delta_sec",
    "learner_sample_delta_sec",
    "observation_delta_sec",
    "compact_batch_build_delta_sec",
    "gather_merge_delta_sec",
    "compact_payload_pickle_delta_sec",
    "scalar_materialization_delta_sec",
)

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


class CompactSpeedRowFloorBundleError(ValueError):
    """Raised when the floor bundle would overclaim or bind stale evidence."""


def build_compact_speed_row_floor_bundle_v1(
    *,
    run_id: str,
    accepted_speed_row_report_path: str | Path,
    compact_torch_sibling_report_path: str | Path,
    fixed_floor_sibling_report_path: str | Path,
    compatibility_report_path: str | Path | None = None,
    unified_lifecycle_report_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the OPT-062 speed-row floor decomposition packet."""

    rows = {
        ROLE_ACCEPTED_COACH_SPEED_ROW: _load_speed_row(
            ROLE_ACCEPTED_COACH_SPEED_ROW,
            accepted_speed_row_report_path,
        ),
        ROLE_COMPACT_TORCH_SIBLING: _load_speed_row(
            ROLE_COMPACT_TORCH_SIBLING,
            compact_torch_sibling_report_path,
        ),
        ROLE_FIXED_FLOOR_SIBLING: _load_speed_row(
            ROLE_FIXED_FLOOR_SIBLING,
            fixed_floor_sibling_report_path,
        ),
    }
    candidate_ids = {str(row["candidate_checkpoint_id"]) for row in rows.values()}
    if len(candidate_ids) != 1:
        raise CompactSpeedRowFloorBundleError("speed-row candidate ids do not match")
    candidate_checkpoint_id = next(iter(candidate_ids))
    denominator_check = _denominator_check(rows)
    comparisons = _comparisons(rows, denominator_check=denominator_check)
    status = (
        STATUS_COMPLETE
        if bool(denominator_check["same_denominator"])
        else STATUS_DENOMINATOR_MISMATCH
    )
    non_claims = _false_claims()
    payload = {
        "schema_id": COMPACT_SPEED_ROW_FLOOR_BUNDLE_SCHEMA_ID,
        "ok": bool(denominator_check["same_denominator"]),
        "status": status,
        "run_id": _require_non_empty(run_id, "run_id"),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "input_reports": {
            "accepted_coach_speed_row": rows[ROLE_ACCEPTED_COACH_SPEED_ROW][
                "report_ref"
            ],
            "compact_torch_search_service_sibling": rows[ROLE_COMPACT_TORCH_SIBLING][
                "report_ref"
            ],
            "fixed_no_search_floor_sibling": rows[ROLE_FIXED_FLOOR_SIBLING][
                "report_ref"
            ],
            **_optional_report_refs(
                compatibility_report_path=compatibility_report_path,
                unified_lifecycle_report_path=unified_lifecycle_report_path,
            ),
        },
        "rows": rows,
        "denominator_check": denominator_check,
        "comparisons": comparisons,
        "engineering_read": _engineering_read(comparisons, denominator_check),
        "limitations": _limitations(denominator_check),
        "non_claims": non_claims,
        "attached_claims": {
            "speed_row_floor_decomposition_hash_bound": True,
            "same_denominator_speed_rows_present": bool(
                denominator_check["same_denominator"]
            ),
            **non_claims,
        },
    }
    payload["evidence_ref"] = compact_speed_row_floor_bundle_evidence_ref(payload)
    validate_compact_speed_row_floor_bundle_v1(payload)
    return payload


def validate_compact_speed_row_floor_bundle_v1(payload: Mapping[str, Any]) -> None:
    """Validate the hash-bound floor bundle and its non-claims."""

    if payload.get("schema_id") != COMPACT_SPEED_ROW_FLOOR_BUNDLE_SCHEMA_ID:
        raise CompactSpeedRowFloorBundleError("speed-row floor bundle schema mismatch")
    if payload.get("status") not in {STATUS_COMPLETE, STATUS_DENOMINATOR_MISMATCH}:
        raise CompactSpeedRowFloorBundleError("speed-row floor bundle status mismatch")
    _validate_false_claims(payload.get("non_claims"), "non_claims")
    _validate_false_claims(payload.get("attached_claims"), "attached_claims")
    rows = _required_mapping(payload.get("rows"), "rows")
    for role in (
        ROLE_ACCEPTED_COACH_SPEED_ROW,
        ROLE_COMPACT_TORCH_SIBLING,
        ROLE_FIXED_FLOOR_SIBLING,
    ):
        if role not in rows:
            raise CompactSpeedRowFloorBundleError(f"missing row role {role}")
        _validate_row(role, _required_mapping(rows[role], role))
    candidate = str(payload.get("candidate_checkpoint_id") or "")
    if not candidate:
        raise CompactSpeedRowFloorBundleError("candidate_checkpoint_id missing")
    if any(str(rows[role].get("candidate_checkpoint_id")) != candidate for role in rows):
        raise CompactSpeedRowFloorBundleError("row candidate_checkpoint_id mismatch")
    denominator_check = _required_mapping(
        payload.get("denominator_check"),
        "denominator_check",
    )
    same_denominator = bool(denominator_check.get("same_denominator"))
    if payload.get("ok") is not same_denominator:
        raise CompactSpeedRowFloorBundleError("ok must match same_denominator")
    if payload.get("status") == STATUS_COMPLETE and not same_denominator:
        raise CompactSpeedRowFloorBundleError("complete bundle requires same denominator")
    if same_denominator and payload.get("status") != STATUS_COMPLETE:
        raise CompactSpeedRowFloorBundleError("same denominator bundle status mismatch")
    comparisons = _required_mapping(payload.get("comparisons"), "comparisons")
    for key in (
        "accepted_vs_compact_torch_wall_ratio",
        "compact_torch_vs_floor_wall_ratio",
        "compact_torch_vs_floor_wall_delta_sec",
        "search_delta_sec",
        "compact_rollout_slab_delta_sec",
        "compact_rollout_slab_non_service_delta_sec",
        "slab_search_dispatch_residual_delta_sec",
        "compact_rollout_slab_non_dispatch_delta_sec",
        "compact_rollout_slab_non_action_service_delta_sec",
        "search_service_action_wall_over_total_delta_sec",
        "top_level_known_accounting_delta_sec",
        "top_level_accounting_residual_delta_sec",
        "legacy_named_unattributed_gap_sec",
        "floor_remaining_wall_sec",
        "slab_search_dispatch_residual_abs_share_of_measured_gap",
    ):
        _require_finite_number(comparisons.get(key), f"comparisons.{key}")
    expected_ref = compact_speed_row_floor_bundle_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactSpeedRowFloorBundleError("speed-row floor bundle evidence_ref mismatch")


def compact_speed_row_floor_bundle_evidence_ref(payload: Mapping[str, Any]) -> str:
    """Return a stable evidence ref for a speed-row floor bundle."""

    body = {
        key: value
        for key, value in payload.items()
        if key != "evidence_ref"
    }
    digest = _json_sha256(body)
    run_id = _require_non_empty(payload.get("run_id"), "run_id")
    candidate = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    return (
        f"{COMPACT_SPEED_ROW_FLOOR_BUNDLE_EVIDENCE_REF_PREFIX}"
        f"{candidate}:{run_id}:sha256={digest}"
    )


def _load_speed_row(role: str, report_path_raw: str | Path) -> dict[str, Any]:
    report_path = Path(report_path_raw).resolve()
    report = _read_json_mapping(report_path, f"{role} report")
    result_path = _resolve_ref(report.get("result_path"), base_path=report_path)
    evidence_path = _resolve_ref(report.get("evidence_path"), base_path=report_path)
    result = _read_json_mapping(result_path, f"{role} result")
    evidence = _read_json_mapping(evidence_path, f"{role} evidence")
    validate_compact_coach_speed_row_evidence_v1(evidence)
    validate_compact_coach_speed_row_evidence_matches_report_v1(
        evidence,
        evidence_ref=str(report.get("evidence_ref") or evidence.get("evidence_ref")),
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        speed_currency=SPEED_CURRENCY,
    )
    summary = _required_mapping(result.get("summary"), f"{role} summary")
    compact = _required_mapping(result.get("compact"), f"{role} compact")
    row = _required_mapping(result.get("row"), f"{role} row")
    _validate_source_false_claims(report, f"{role} report")
    _validate_source_false_claims(result, f"{role} result")
    _validate_source_false_claims(summary, f"{role} summary")
    _validate_source_false_claims(compact, f"{role} compact")
    _validate_source_false_claims(row, f"{role} row")
    source_profile = _source_profile(compact)
    denominator = _required_mapping(evidence.get("denominator"), f"{role} denominator")
    search_impl = _search_impl(summary=summary, compact=compact, source_profile=source_profile)
    search_kind = _search_kind(summary=summary, compact=compact, search_impl=search_impl)
    model_identity = _required_mapping(evidence.get("model_identity"), f"{role} model_identity")
    shape = _shape(
        row=row,
        summary=summary,
        compact=compact,
        source_profile=source_profile,
        report_path=report_path,
    )
    timing_buckets = _timing_buckets(source_profile)
    return {
        "role": role,
        "report_ref": _file_ref(report_path, payload=report),
        "result_ref": _file_ref(result_path, payload=result),
        "evidence_ref": _file_ref(evidence_path, payload=evidence),
        "coach_speed_row_evidence_ref": str(evidence["evidence_ref"]),
        "schema_id": str(result.get("schema_id") or ""),
        "row_id": str(result.get("row_id") or summary.get("row_id") or ""),
        "candidate_checkpoint_id": str(summary.get("candidate_checkpoint_id") or ""),
        "route": str(summary.get("route") or ""),
        "row_purpose": str(summary.get("row_purpose") or ""),
        "search_impl_kind": search_kind,
        "search_impl": search_impl,
        "model_identity_scope": str(model_identity.get("scope") or ""),
        "hardware_class": shape["hardware_class"],
        "shape": shape,
        "speed_currency": str(denominator.get("speed_currency") or ""),
        "numerator_field": str(denominator.get("numerator_field") or ""),
        "numerator_value": _require_finite_number(
            denominator.get("numerator_value"),
            f"{role} numerator",
        ),
        "denominator_field": str(denominator.get("denominator_field") or ""),
        "denominator_value_sec": _require_finite_number(
            denominator.get("denominator_value_sec"),
            f"{role} denominator",
        ),
        "steps_per_sec": _require_finite_number(
            denominator.get("reported_steps_per_sec"),
            f"{role} steps_per_sec",
        ),
        "uses_fallback_denominator": bool(
            denominator.get("uses_fallback_denominator")
        ),
        "profile_only": bool(summary.get("profile_only")),
        "calls_train_muzero": bool(summary.get("calls_train_muzero")),
        "touches_live_runs": bool(summary.get("touches_live_runs")),
        "promotion_claim": bool(summary.get("promotion_claim")),
        "timing_buckets": timing_buckets,
        "timing_accounting": _timing_accounting(timing_buckets),
        "search_service_flags": _search_service_flags(
            source_profile,
            timing_buckets=timing_buckets,
        ),
        "trajectory_fingerprints": _trajectory_fingerprints(source_profile),
        "non_claims": _false_claims(),
    }


def _validate_row(role: str, row: Mapping[str, Any]) -> None:
    if row.get("role") != role:
        raise CompactSpeedRowFloorBundleError(f"{role} role mismatch")
    expected_kind = {
        ROLE_ACCEPTED_COACH_SPEED_ROW: SEARCH_KIND_DEVICE_TARGET,
        ROLE_COMPACT_TORCH_SIBLING: SEARCH_KIND_COMPACT_TORCH,
        ROLE_FIXED_FLOOR_SIBLING: SEARCH_KIND_FIXED_SHAPE,
    }[role]
    if row.get("search_impl_kind") != expected_kind:
        raise CompactSpeedRowFloorBundleError(
            f"{role} search_impl_kind must be {expected_kind}"
        )
    if row.get("speed_currency") != SPEED_CURRENCY:
        raise CompactSpeedRowFloorBundleError(f"{role} speed currency mismatch")
    if row.get("row_purpose") != "coach_speed_row":
        raise CompactSpeedRowFloorBundleError(f"{role} row_purpose mismatch")
    if row.get("route") != COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER:
        raise CompactSpeedRowFloorBundleError(f"{role} route mismatch")
    if row.get("model_identity_scope") != "candidate_loaded_checkpoint":
        raise CompactSpeedRowFloorBundleError(f"{role} must use loaded checkpoint identity")
    if row.get("uses_fallback_denominator") is not False:
        raise CompactSpeedRowFloorBundleError(f"{role} used fallback denominator")
    if row.get("profile_only") is not False:
        raise CompactSpeedRowFloorBundleError(f"{role} must be profile_only=false")
    if row.get("calls_train_muzero") is not False:
        raise CompactSpeedRowFloorBundleError(f"{role} must not call train_muzero")
    if row.get("touches_live_runs") is not False:
        raise CompactSpeedRowFloorBundleError(f"{role} must not touch live runs")
    if row.get("promotion_claim") is not False:
        raise CompactSpeedRowFloorBundleError(f"{role} must not claim promotion")
    _validate_false_claims(row.get("non_claims"), f"{role}.non_claims")
    _validate_compact_torch_timing_guards(role, row)
    for ref_key in ("report_ref", "result_ref", "evidence_ref"):
        ref = _required_mapping(row.get(ref_key), f"{role}.{ref_key}")
        path = Path(str(ref.get("path") or ""))
        if not path.is_file():
            raise CompactSpeedRowFloorBundleError(f"{role}.{ref_key} path missing")
        if _sha256(path) != ref.get("sha256"):
            raise CompactSpeedRowFloorBundleError(f"{role}.{ref_key} sha256 mismatch")


def _validate_compact_torch_timing_guards(
    role: str,
    row: Mapping[str, Any],
) -> None:
    if role != ROLE_COMPACT_TORCH_SIBLING:
        return
    timing = _required_mapping(row.get("timing_buckets"), f"{role}.timing_buckets")
    host_fallback_count = _require_finite_number(
        timing.get("resident_observation_host_fallback_count", 0.0),
        f"{role}.resident_observation_host_fallback_count",
    )
    if host_fallback_count != 0.0:
        raise CompactSpeedRowFloorBundleError(
            f"{role} resident_observation_host_fallback_count must be zero"
        )
    if not _cuda_event_timing_enabled(timing):
        return
    incomplete = _incomplete_cuda_event_timing_fields(timing)
    if incomplete:
        raise CompactSpeedRowFloorBundleError(
            f"{role} cuda-event timing incomplete: " + ", ".join(incomplete)
        )
    if _direct_core_initial_inference_used(timing):
        incomplete_direct = _incomplete_direct_core_cuda_event_timing_fields(timing)
        if incomplete_direct:
            raise CompactSpeedRowFloorBundleError(
                f"{role} direct-core cuda-event timing incomplete: "
                + ", ".join(incomplete_direct)
            )


def _cuda_event_timing_enabled(timing: Mapping[str, Any]) -> bool:
    enabled_count = _require_finite_number(
        timing.get("search_service_cuda_event_timing_enabled_count", 0.0),
        "search_service_cuda_event_timing_enabled_count",
    )
    return enabled_count > 0.0


def _incomplete_cuda_event_timing_fields(timing: Mapping[str, Any]) -> list[str]:
    incomplete = []
    for key in _REQUIRED_CUDA_EVENT_TIMING_FIELDS:
        value = _require_finite_number(timing.get(key), key)
        if value <= 0.0:
            incomplete.append(key)
    return incomplete


def _direct_core_initial_inference_used(timing: Mapping[str, Any]) -> bool:
    return (
        _require_finite_number(
            timing.get("search_service_initial_inference_direct_used_count", 0.0),
            "search_service_initial_inference_direct_used_count",
        )
        > 0.0
    )


def _incomplete_direct_core_cuda_event_timing_fields(
    timing: Mapping[str, Any],
) -> list[str]:
    incomplete = []
    for key in _REQUIRED_DIRECT_CORE_CUDA_EVENT_TIMING_FIELDS:
        value = _require_finite_number(timing.get(key), key)
        if value <= 0.0:
            incomplete.append(key)
    return incomplete


def _validate_source_false_claims(value: Any, name: str) -> None:
    source = _required_mapping(value, name)
    nested = source.get("non_claims")
    if isinstance(nested, Mapping):
        for key in FALSE_CLAIM_KEYS:
            if key in nested and nested.get(key) is not False:
                raise CompactSpeedRowFloorBundleError(
                    f"{name}.non_claims.{key} must be false"
                )
    for key in FALSE_CLAIM_KEYS:
        if key in source and source.get(key) is not False:
            raise CompactSpeedRowFloorBundleError(f"{name}.{key} must be false")


def _denominator_check(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, Any]] = {}
    for key in DENOMINATOR_KEYS:
        values[key] = {
            role: _denominator_value(row, key)
            for role, row in rows.items()
        }
    mismatches = {
        key: by_role
        for key, by_role in values.items()
        if len({_stable_json(value) for value in by_role.values()}) != 1
    }
    same = not mismatches
    trajectory_values = {
        role: row.get("trajectory_fingerprints", {}).get("env_trajectory_checksum_total")
        for role, row in rows.items()
    }
    return {
        "same_denominator": same,
        "checked_keys": list(DENOMINATOR_KEYS),
        "values": values,
        "mismatches": mismatches,
        "comparison_scope": (
            "same_denominator_shape_matched"
            if same
            else "denominator_shape_mismatch"
        ),
        "trajectory_checksum_match": len(
            {_stable_json(value) for value in trajectory_values.values()}
        )
        == 1,
        "trajectory_checksum_values": trajectory_values,
    }


def _comparisons(
    rows: Mapping[str, Mapping[str, Any]],
    *,
    denominator_check: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = rows[ROLE_ACCEPTED_COACH_SPEED_ROW]
    compact = rows[ROLE_COMPACT_TORCH_SIBLING]
    floor = rows[ROLE_FIXED_FLOOR_SIBLING]
    accepted_wall = float(accepted["denominator_value_sec"])
    compact_wall = float(compact["denominator_value_sec"])
    floor_wall = float(floor["denominator_value_sec"])
    compact_timing = _required_mapping(compact.get("timing_buckets"), "compact timing")
    floor_timing = _required_mapping(floor.get("timing_buckets"), "floor timing")
    compact_accounting = _required_mapping(
        compact.get("timing_accounting"),
        "compact timing accounting",
    )
    floor_accounting = _required_mapping(
        floor.get("timing_accounting"),
        "floor timing accounting",
    )
    wall_delta = compact_wall - floor_wall
    search_delta = _timing_delta(compact_timing, floor_timing, "search_service_total_sec")
    learner_sample_delta = (
        _timing_delta(compact_timing, floor_timing, "learner_gate_sec")
        + _timing_delta(compact_timing, floor_timing, "sample_gate_sec")
    )
    observation_delta = _timing_delta(compact_timing, floor_timing, "observation_sec")
    actor_env_delta = _timing_delta(compact_timing, floor_timing, "actor_step_sec")
    actor_wall_delta = _timing_delta(compact_timing, floor_timing, "actor_step_wall_sec")
    compact_rollout_slab_delta = _timing_delta(
        compact_timing,
        floor_timing,
        "compact_rollout_slab_sec",
    )
    compact_rollout_slab_non_service_delta = _accounting_delta(
        compact_accounting,
        floor_accounting,
        "compact_rollout_slab_non_service_sec",
    )
    slab_search_dispatch_residual_delta = _accounting_delta(
        compact_accounting,
        floor_accounting,
        "slab_search_dispatch_residual_sec",
    )
    compact_rollout_slab_non_dispatch_delta = _accounting_delta(
        compact_accounting,
        floor_accounting,
        "compact_rollout_slab_non_dispatch_sec",
    )
    compact_rollout_slab_non_action_service_delta = _accounting_delta(
        compact_accounting,
        floor_accounting,
        "compact_rollout_slab_non_action_service_sec",
    )
    search_service_action_wall_over_total_delta = _accounting_delta(
        compact_accounting,
        floor_accounting,
        "search_service_action_wall_over_total_sec",
    )
    top_level_known_accounting_delta = _accounting_delta(
        compact_accounting,
        floor_accounting,
        "top_level_known_accounting_sec",
    )
    top_level_residual_delta = _accounting_delta(
        compact_accounting,
        floor_accounting,
        "top_level_accounting_residual_sec",
    )
    legacy_named_delta_sum = (
        search_delta
        + learner_sample_delta
        + observation_delta
        + actor_env_delta
    )
    high_level_deltas = {
        "compact_rollout_slab_delta_sec": compact_rollout_slab_delta,
        "actor_wall_delta_sec": actor_wall_delta,
        "learner_sample_delta_sec": learner_sample_delta,
        "observation_delta_sec": observation_delta,
        "compact_batch_build_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "compact_batch_build_sec",
        ),
        "gather_merge_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "gather_merge_sec",
        ),
        "compact_payload_pickle_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "compact_payload_pickle_sec",
        ),
        "scalar_materialization_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "scalar_materialization_sec",
        ),
        "slab_internal_accounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_internal_accounted_sec",
        ),
        "slab_commit_previous_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_commit_previous_sec",
        ),
        "slab_commit_action_check_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_commit_action_check_sec",
        ),
        "slab_replay_payload_flush_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_payload_flush_sec",
        ),
        "slab_replay_payload_validate_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_payload_validate_sec",
        ),
        "slab_replay_payload_materialize_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_payload_materialize_sec",
        ),
        "slab_replay_result_validate_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_result_validate_sec",
        ),
        "slab_replay_index_rows_build_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_index_rows_build_sec",
        ),
        "slab_replay_index_rows_store_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_index_rows_store_sec",
        ),
        "slab_commit_child_accounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_commit_child_accounted_sec",
        ),
        "slab_commit_residual_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_commit_residual_sec",
        ),
        "slab_replay_index_rows_identity_validate_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_index_rows_identity_validate_sec",
        ),
        "slab_replay_index_rows_terminal_prepare_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_index_rows_terminal_prepare_sec",
        ),
        "slab_replay_index_rows_target_tensor_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_index_rows_target_tensor_sec",
        ),
        "slab_replay_index_rows_scalar_host_pack_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_index_rows_scalar_host_pack_sec",
        ),
        "slab_replay_index_rows_scalar_device_transfer_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_index_rows_scalar_device_transfer_sec",
        ),
        "slab_replay_index_rows_metadata_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_replay_index_rows_metadata_sec",
        ),
        "slab_root_batch_build_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_root_batch_build_sec",
        ),
        "slab_root_tape_record_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_root_tape_record_sec",
        ),
        "slab_search_dispatch_wall_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_search_dispatch_wall_sec",
        ),
        "slab_search_dispatch_residual_delta_sec": (
            slab_search_dispatch_residual_delta
        ),
        "compact_rollout_slab_non_dispatch_delta_sec": (
            compact_rollout_slab_non_dispatch_delta
        ),
        "compact_rollout_slab_non_action_service_delta_sec": (
            compact_rollout_slab_non_action_service_delta
        ),
        "search_service_action_wall_over_total_delta_sec": (
            search_service_action_wall_over_total_delta
        ),
        "slab_search_service_action_preamble_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_action_preamble_sec",
        ),
        "slab_search_service_fixed_shape_masks_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_fixed_shape_masks_sec",
        ),
        "slab_search_service_compile_eligibility_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_compile_eligibility_sec",
        ),
        "slab_search_service_helper_cache_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_helper_cache_sec",
        ),
        "slab_search_service_model_cache_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_model_cache_sec",
        ),
        "slab_search_service_inference_guard_enter_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_inference_guard_enter_sec",
        ),
        "slab_search_service_inference_guard_exit_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_inference_guard_exit_sec",
        ),
        "slab_search_service_inference_guard_total_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_inference_guard_total_sec",
        ),
        "slab_search_service_metadata_build_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_metadata_build_sec",
        ),
        "slab_search_service_pending_replay_store_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_pending_replay_store_sec",
        ),
        "slab_search_service_action_step_build_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_action_step_build_sec",
        ),
        "slab_search_service_action_postprocess_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_action_postprocess_sec",
        ),
        "slab_search_service_action_wall_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_action_wall_sec",
        ),
        "slab_search_service_action_accounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_action_accounted_sec",
        ),
        "slab_search_service_action_residual_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_action_residual_sec",
        ),
        "slab_search_service_action_unaccounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_action_unaccounted_sec",
        ),
        "slab_search_service_action_overaccounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_action_overaccounted_sec",
        ),
        "slab_search_service_tensor_prepare_sync_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_tensor_prepare_sync_sec",
        ),
        "slab_search_service_initial_output_decode_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_output_decode_sec",
        ),
        "slab_search_service_root_output_decode_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_root_output_decode_sec",
        ),
        "slab_search_service_initial_inference_enqueue_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_enqueue_sec",
        ),
        "slab_search_service_initial_inference_sync_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_sync_sec",
        ),
        "slab_search_service_initial_inference_cuda_event_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_cuda_event_sec",
        ),
        "slab_search_service_initial_inference_representation_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_representation_sec",
        ),
        "slab_search_service_initial_inference_prediction_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_prediction_sec",
        ),
        "slab_search_service_initial_inference_pack_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_pack_sec",
        ),
        "slab_search_service_initial_inference_representation_cuda_event_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_representation_cuda_event_sec",
        ),
        "slab_search_service_initial_inference_prediction_cuda_event_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_prediction_cuda_event_sec",
        ),
        "slab_search_service_initial_inference_direct_core_cuda_event_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_direct_core_cuda_event_sec",
        ),
        "slab_search_service_initial_inference_direct_core_cuda_event_residual_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_initial_inference_direct_core_cuda_event_residual_sec",
        ),
        "slab_search_service_tree_root_prior_build_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_tree_root_prior_build_sec",
        ),
        "slab_search_service_tree_root_prior_select_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_tree_root_prior_select_sec",
        ),
        "slab_search_service_tree_recurrent_inference_enqueue_delta_sec": (
            _timing_delta(
                compact_timing,
                floor_timing,
                "search_service_tree_recurrent_inference_enqueue_sec",
            )
        ),
        "slab_search_service_tree_recurrent_inference_cuda_event_delta_sec": (
            _timing_delta(
                compact_timing,
                floor_timing,
                "search_service_tree_recurrent_inference_cuda_event_sec",
            )
        ),
        "slab_search_service_tree_sync_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_tree_sync_sec",
        ),
        "slab_search_service_tree_cuda_event_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_tree_cuda_event_sec",
        ),
        "slab_search_service_tree_total_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_tree_total_sec",
        ),
        "slab_search_service_tree_accounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_tree_accounted_sec",
        ),
        "slab_search_service_tree_residual_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_tree_residual_sec",
        ),
        "slab_search_service_tree_overaccounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_tree_overaccounted_sec",
        ),
        "slab_search_service_action_readback_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_action_readback_sec",
        ),
        "slab_search_service_core_accounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_core_accounted_sec",
        ),
        "slab_search_service_core_residual_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_core_residual_sec",
        ),
        "slab_search_service_core_unaccounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_core_unaccounted_sec",
        ),
        "slab_search_service_core_overaccounted_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "search_service_core_overaccounted_sec",
        ),
        "slab_search_identity_validation_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_search_identity_validation_sec",
        ),
        "slab_joint_action_assembly_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_joint_action_assembly_sec",
        ),
        "slab_pending_store_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_pending_store_sec",
        ),
        "slab_telemetry_build_delta_sec": _timing_delta(
            compact_timing,
            floor_timing,
            "slab_telemetry_build_sec",
        ),
    }
    dominant_candidates = {
        key: high_level_deltas[key]
        for key in DOMINANT_WALL_DELTA_KEYS
        if key in high_level_deltas
    }
    dominant_high_level_key = max(
        dominant_candidates,
        key=lambda key: abs(dominant_candidates[key]),
    )
    return {
        "comparison_scope": denominator_check.get("comparison_scope"),
        "accepted_vs_compact_torch_wall_ratio": _safe_ratio(
            accepted_wall,
            compact_wall,
        ),
        "compact_torch_vs_floor_wall_ratio": _safe_ratio(compact_wall, floor_wall),
        "accepted_vs_floor_wall_ratio": _safe_ratio(accepted_wall, floor_wall),
        "accepted_wall_sec": accepted_wall,
        "compact_torch_wall_sec": compact_wall,
        "fixed_floor_wall_sec": floor_wall,
        "compact_torch_vs_floor_wall_delta_sec": wall_delta,
        "floor_remaining_wall_sec": floor_wall,
        "search_delta_sec": search_delta,
        "learner_sample_delta_sec": learner_sample_delta,
        "observation_delta_sec": observation_delta,
        "actor_env_delta_sec": actor_env_delta,
        "actor_wall_delta_sec": actor_wall_delta,
        "compact_rollout_slab_delta_sec": compact_rollout_slab_delta,
        "compact_rollout_slab_non_service_delta_sec": (
            compact_rollout_slab_non_service_delta
        ),
        "slab_search_dispatch_residual_delta_sec": slab_search_dispatch_residual_delta,
        "compact_rollout_slab_non_dispatch_delta_sec": (
            compact_rollout_slab_non_dispatch_delta
        ),
        "compact_rollout_slab_non_action_service_delta_sec": (
            compact_rollout_slab_non_action_service_delta
        ),
        "search_service_action_wall_over_total_delta_sec": (
            search_service_action_wall_over_total_delta
        ),
        "top_level_known_accounting_delta_sec": top_level_known_accounting_delta,
        "top_level_accounting_residual_delta_sec": top_level_residual_delta,
        "legacy_named_bucket_delta_sum_sec": legacy_named_delta_sum,
        "legacy_named_unattributed_gap_sec": wall_delta - legacy_named_delta_sum,
        "dominant_high_level_delta_key": dominant_high_level_key,
        "dominant_high_level_delta_sec": high_level_deltas[dominant_high_level_key],
        "high_level_deltas": high_level_deltas,
        "search_delta_abs_share_of_measured_gap": _safe_share(search_delta, wall_delta),
        "compact_rollout_slab_delta_abs_share_of_measured_gap": _safe_share(
            compact_rollout_slab_delta,
            wall_delta,
        ),
        "compact_rollout_slab_non_service_delta_abs_share_of_measured_gap": (
            _safe_share(compact_rollout_slab_non_service_delta, wall_delta)
        ),
        "slab_search_dispatch_residual_abs_share_of_measured_gap": _safe_share(
            slab_search_dispatch_residual_delta,
            wall_delta,
        ),
        "top_level_accounting_residual_abs_share_of_measured_gap": _safe_share(
            top_level_residual_delta,
            wall_delta,
        ),
        "floor_remaining_wall_over_search_delta_ratio": _safe_ratio(
            floor_wall,
            abs(search_delta),
        ),
        "accepted_steps_per_sec": float(accepted["steps_per_sec"]),
        "compact_torch_steps_per_sec": float(compact["steps_per_sec"]),
        "fixed_floor_steps_per_sec": float(floor["steps_per_sec"]),
    }


def _engineering_read(
    comparisons: Mapping[str, Any],
    denominator_check: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(denominator_check.get("same_denominator")):
        read = "measurement_or_trajectory_confounded"
        dominant_high_level = "unknown"
        next_target = "rerun_same_denominator_control"
        search_dominance_claim = False
        residual_accounting_required = True
    else:
        dominant_high_level = str(comparisons.get("dominant_high_level_delta_key") or "")
        dominant_abs = abs(float(comparisons.get("dominant_high_level_delta_sec", 0.0)))
        top_level_residual_abs = abs(
            float(comparisons.get("top_level_accounting_residual_delta_sec", 0.0))
        )
        search_abs = abs(float(comparisons["search_delta_sec"]))
        slab_non_service_abs = abs(
            float(comparisons.get("compact_rollout_slab_non_service_delta_sec", 0.0))
        )
        high_level_deltas = comparisons.get("high_level_deltas")
        high_level_deltas = (
            high_level_deltas if isinstance(high_level_deltas, Mapping) else {}
        )
        slab_search_dispatch_abs = abs(
            float(high_level_deltas.get("slab_search_dispatch_wall_delta_sec", 0.0))
        )
        if dominant_abs <= 0.0:
            read = "mixed_no_single_wall"
        elif top_level_residual_abs > dominant_abs:
            read = "residual_accounting_required"
        elif dominant_high_level == "compact_rollout_slab_delta_sec":
            if slab_search_dispatch_abs > search_abs:
                read = "compact_rollout_slab_search_dispatch_wall_dominant"
            elif slab_non_service_abs > search_abs:
                read = "compact_rollout_slab_non_service_dominant"
            else:
                read = "compact_rollout_slab_search_service_dominant"
        elif dominant_high_level == "actor_wall_delta_sec":
            read = "actor_wall_dominant"
        elif dominant_high_level == "learner_sample_delta_sec":
            read = "learner_sample_dominant"
        elif dominant_high_level == "observation_delta_sec":
            read = "observation_refresh_dominant"
        else:
            read = "mixed_known_wall_dominant"
        search_dominance_claim = read == "compact_rollout_slab_search_service_dominant"
        residual_accounting_required = (
            read == "residual_accounting_required"
            or read == "compact_rollout_slab_non_service_dominant"
            or read == "compact_rollout_slab_search_dispatch_wall_dominant"
        )
        next_target = {
            "compact_rollout_slab_non_service_dominant": (
                "decompose_compact_rollout_slab_commit_flush_materialization"
            ),
            "compact_rollout_slab_search_service_dominant": (
                "optimize_compact_torch_search_service"
            ),
            "compact_rollout_slab_search_dispatch_wall_dominant": (
                "decompose_compact_torch_search_dispatch_envelope"
            ),
            "actor_wall_dominant": "optimize_actor_runtime_or_actor_wall",
            "learner_sample_dominant": "optimize_learner_sample_gates",
            "observation_refresh_dominant": "optimize_observation_refresh",
            "residual_accounting_required": "decompose_unattributed_wall",
            "mixed_no_single_wall": "repeat_or_expand_accounting",
            "mixed_known_wall_dominant": "inspect_known_wall_deltas",
        }.get(read, "inspect_known_wall_deltas")
    return {
        "classification": read,
        "decision_confidence": (
            "limited_shape_matched_trajectory_different"
            if not bool(denominator_check.get("trajectory_checksum_match"))
            else "shape_matched"
        ),
        "dominant_high_level_delta": dominant_high_level,
        "dominant_high_level_delta_sec": float(
            comparisons.get("dominant_high_level_delta_sec", 0.0)
        ),
        "search_dominance_claim": search_dominance_claim,
        "residual_accounting_required": residual_accounting_required,
        "next_target": next_target,
        "promotion_claim": False,
        "training_speedup_claim": False,
        "notes": [
            "This is an optimizer engineering decomposition only.",
            "It does not compare to stock train_muzero quality or promotion outcomes.",
            (
                "The raw search-service delta is not treated as wall-dominant "
                "unless it dominates slab-internal non-service wall."
            ),
        ],
    }


def _limitations(denominator_check: Mapping[str, Any]) -> list[str]:
    limitations = [
        "accepted baseline is the Coach speed-row device-target trainer smoke",
        "fixed floor is no-search first-legal fixed-shape service",
        "compact Torch sibling is profile-only search-service implementation, not LightZero CTree",
        "no promotion, rating, live-run, or stock-speedup claim is made",
    ]
    if not bool(denominator_check.get("trajectory_checksum_match")):
        limitations.append("trajectory/action checksums differ across sibling services")
    if not bool(denominator_check.get("same_denominator")):
        limitations.append("denominator shape mismatch prevents a complete floor read")
    return limitations


def _source_profile(compact: Mapping[str, Any]) -> Mapping[str, Any]:
    source = compact.get("source_profile_payload")
    if isinstance(source, Mapping):
        return source
    return {}


def _shape(
    *,
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
    compact: Mapping[str, Any],
    source_profile: Mapping[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    return {
        "hardware_class": _hardware_class(report_path),
        "batch_size": _int_from(row, source_profile, "batch_size"),
        "actor_count": _int_from(row, source_profile, "actor_count"),
        "steps": _int_from(row, source_profile, "steps"),
        "warmup_steps": _int_from(row, source_profile, "warmup_steps"),
        "sample_batch_size": _int_from(
            row,
            source_profile,
            "sample_batch_size",
            fallback_key="compact_rollout_slab_sample_gate_batch_size",
        ),
        "sample_interval": _int_from(
            row,
            source_profile,
            "sample_interval",
            fallback_key="compact_rollout_slab_sample_gate_interval",
        ),
        "replay_pair_capacity": _int_from(
            row,
            source_profile,
            "replay_pair_capacity",
            fallback_key="compact_rollout_slab_sample_gate_replay_ring_pair_capacity",
        ),
        "learner_device": str(
            row.get("learner_device")
            or source_profile.get("compact_rollout_slab_learner_gate_device")
            or ""
        ),
        "learner_train_steps": _int_from(
            row,
            source_profile,
            "learner_train_steps",
            fallback_key="compact_rollout_slab_learner_gate_train_steps",
        ),
        "num_simulations": int(
            row.get("num_simulations")
            or compact.get("num_simulations")
            or _nested_get(
                source_profile,
                (
                    "compact_rollout_slab_last_telemetry",
                    "compact_rollout_slab_num_simulations",
                ),
            )
            or source_profile.get("compact_rollout_slab_num_simulations")
            or 0
        ),
        "env_steps_collected": _require_finite_number(
            summary.get("env_steps_collected"),
            "env_steps_collected",
        ),
    }


def _timing_buckets(source_profile: Mapping[str, Any]) -> dict[str, float]:
    totals = source_profile.get("compact_rollout_slab_telemetry_totals")
    totals = totals if isinstance(totals, Mapping) else {}
    timings = source_profile.get("timings")
    timings = timings if isinstance(timings, Mapping) else {}
    return {
        "measured_sec": _float_or_zero(source_profile.get("measured_sec")),
        "total_sec": _float_or_zero(source_profile.get("total_sec")),
        "search_service_total_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_total_sec")
        ),
        "search_service_action_preamble_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_action_preamble_sec")
        ),
        "search_service_fixed_shape_masks_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_fixed_shape_masks_sec")
        ),
        "search_service_compile_eligibility_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_compile_eligibility_sec")
        ),
        "search_service_helper_cache_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_helper_cache_sec")
        ),
        "search_service_model_cache_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_model_cache_sec")
        ),
        "search_service_inference_guard_enter_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_inference_guard_enter_sec")
        ),
        "search_service_inference_guard_exit_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_inference_guard_exit_sec")
        ),
        "search_service_inference_guard_total_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_inference_guard_total_sec")
        ),
        "search_service_metadata_build_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_metadata_build_sec")
        ),
        "search_service_pending_replay_store_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_pending_replay_store_sec")
        ),
        "search_service_action_step_build_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_action_step_build_sec")
        ),
        "search_service_action_postprocess_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_action_postprocess_sec")
        ),
        "search_service_action_wall_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_action_wall_sec")
        ),
        "search_service_action_accounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_action_accounted_sec")
        ),
        "search_service_action_residual_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_action_residual_sec")
        ),
        "search_service_action_unaccounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_action_unaccounted_sec")
        ),
        "search_service_action_overaccounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_action_overaccounted_sec")
        ),
        "search_service_tensor_prepare_sync_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tensor_prepare_sync_sec")
        ),
        "search_service_initial_output_decode_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_initial_output_decode_sec")
        ),
        "search_service_root_output_decode_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_root_output_decode_sec")
        ),
        "search_service_initial_inference_enqueue_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_enqueue_sec"
            )
        ),
        "search_service_initial_inference_sync_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_initial_inference_sync_sec")
        ),
        "search_service_initial_inference_cuda_event_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_cuda_event_sec"
            )
        ),
        "search_service_initial_inference_representation_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_representation_sec"
            )
        ),
        "search_service_initial_inference_prediction_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_prediction_sec"
            )
        ),
        "search_service_initial_inference_pack_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_initial_inference_pack_sec")
        ),
        "search_service_initial_inference_representation_cuda_event_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_representation_cuda_event_sec"
            )
        ),
        "search_service_initial_inference_prediction_cuda_event_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_prediction_cuda_event_sec"
            )
        ),
        "search_service_initial_inference_direct_core_cuda_event_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_direct_core_cuda_event_sec"
            )
        ),
        "search_service_initial_inference_direct_core_cuda_event_residual_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_direct_core_cuda_event_residual_sec"
            )
        ),
        "search_service_initial_inference_direct_requested_count": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_direct_requested"
            )
        ),
        "search_service_initial_inference_direct_used_count": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_direct_used"
            )
        ),
        "search_service_initial_inference_fallback_count": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_initial_inference_fallback_count"
            )
        ),
        "search_service_tree_setup_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_setup_sec")
        ),
        "search_service_tree_root_prior_build_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_root_prior_build_sec")
        ),
        "search_service_tree_root_prior_select_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_root_prior_select_sec")
        ),
        "search_service_tree_select_enqueue_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_select_enqueue_sec")
        ),
        "search_service_tree_recurrent_action_build_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_tree_recurrent_action_build_sec"
            )
        ),
        "search_service_tree_recurrent_inference_enqueue_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_tree_recurrent_inference_enqueue_sec"
            )
        ),
        "search_service_tree_recurrent_inference_cuda_event_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_tree_recurrent_inference_cuda_event_sec"
            )
        ),
        "search_service_tree_recurrent_output_decode_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_tree_recurrent_output_decode_sec"
            )
        ),
        "search_service_tree_backup_enqueue_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_backup_enqueue_sec")
        ),
        "search_service_tree_policy_build_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_policy_build_sec")
        ),
        "search_service_tree_sync_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_sync_sec")
        ),
        "search_service_tree_cuda_event_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_cuda_event_sec")
        ),
        "search_service_tree_total_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_total_sec")
        ),
        "search_service_tree_accounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_accounted_sec")
        ),
        "search_service_tree_residual_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_residual_sec")
        ),
        "search_service_tree_unaccounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_unaccounted_sec")
        ),
        "search_service_tree_overaccounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_tree_overaccounted_sec")
        ),
        "search_service_action_readback_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_action_readback_sec")
        ),
        "search_service_core_accounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_core_accounted_sec")
        ),
        "search_service_core_residual_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_core_residual_sec")
        ),
        "search_service_core_unaccounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_core_unaccounted_sec")
        ),
        "search_service_core_overaccounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_core_overaccounted_sec")
        ),
        "search_service_cuda_event_timing_enabled_count": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_cuda_event_timing_enabled")
        ),
        "search_service_initial_sync_enabled_count": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_initial_sync_enabled")
        ),
        "search_service_one_simulation_fast_path_count": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_one_simulation_fast_path_count"
            )
        ),
        "search_service_one_simulation_root_prior_softmax_skipped_count": _float_or_zero(
            totals.get(
                "compact_rollout_slab_search_service_one_simulation_root_prior_softmax_skipped"
            )
        ),
        "search_service_recurrent_inference_calls": _float_or_zero(
            totals.get("compact_rollout_slab_search_service_recurrent_inference_calls")
        ),
        "search_sec": _float_or_zero(totals.get("compact_rollout_slab_search_sec")),
        "model_sec": _float_or_zero(totals.get("compact_rollout_slab_model_sec")),
        "h2d_sec": _float_or_zero(totals.get("compact_rollout_slab_h2d_sec")),
        "action_d2h_bytes": _float_or_zero(
            totals.get("compact_rollout_slab_action_d2h_bytes")
        ),
        "replay_payload_d2h_bytes": _float_or_zero(
            totals.get("compact_rollout_slab_replay_payload_d2h_bytes")
        ),
        "committed_replay_payload_d2h_bytes": _float_or_zero(
            totals.get("compact_rollout_slab_committed_replay_payload_d2h_bytes")
        ),
        "resident_observation_host_fallback_count": _float_or_zero(
            totals.get("compact_rollout_slab_resident_observation_host_fallback_count")
        ),
        "learner_gate_sec": _float_or_zero(
            source_profile.get("compact_rollout_slab_learner_gate_sec")
        ),
        "sample_gate_sec": _float_or_zero(
            source_profile.get("compact_rollout_slab_sample_gate_sec")
        ),
        "observation_sec": _float_or_zero(timings.get("observation_sec")),
        "actor_step_sec": _float_or_zero(timings.get("actor_step_sec")),
        "actor_step_wall_sec": _float_or_zero(timings.get("actor_step_wall_sec")),
        "parent_send_receive_sec": _float_or_zero(timings.get("parent_send_receive_sec")),
        "gather_merge_sec": _float_or_zero(timings.get("gather_merge_sec")),
        "compact_batch_build_sec": _float_or_zero(
            timings.get("compact_batch_build_sec")
        ),
        "compact_rollout_slab_sec": _float_or_zero(
            timings.get("compact_rollout_slab_sec")
        ),
        "compact_payload_pickle_sec": _float_or_zero(
            timings.get("compact_payload_pickle_sec")
        ),
        "scalar_materialization_sec": _float_or_zero(
            timings.get("scalar_materialization_sec")
        ),
        "slab_internal_accounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_internal_accounted_sec")
        ),
        "slab_commit_previous_sec": _float_or_zero(
            totals.get("compact_rollout_slab_commit_previous_sec")
        ),
        "slab_root_batch_build_sec": _float_or_zero(
            totals.get("compact_rollout_slab_root_batch_build_sec")
        ),
        "slab_root_tape_record_sec": _float_or_zero(
            totals.get("compact_rollout_slab_root_tape_record_sec")
        ),
        "slab_search_dispatch_wall_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_dispatch_wall_sec")
        ),
        "slab_search_identity_validation_sec": _float_or_zero(
            totals.get("compact_rollout_slab_search_identity_validation_sec")
        ),
        "slab_joint_action_assembly_sec": _float_or_zero(
            totals.get("compact_rollout_slab_joint_action_assembly_sec")
        ),
        "slab_pending_store_sec": _float_or_zero(
            totals.get("compact_rollout_slab_pending_store_sec")
        ),
        "slab_telemetry_build_sec": _float_or_zero(
            totals.get("compact_rollout_slab_telemetry_build_sec")
        ),
        "slab_commit_action_check_sec": _float_or_zero(
            totals.get("compact_rollout_slab_commit_action_check_sec")
        ),
        "slab_replay_payload_flush_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_payload_flush_sec")
        ),
        "slab_replay_payload_validate_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_payload_validate_sec")
        ),
        "slab_replay_payload_materialize_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_payload_materialize_sec")
        ),
        "slab_replay_result_validate_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_result_validate_sec")
        ),
        "slab_replay_index_rows_build_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_index_rows_build_sec")
        ),
        "slab_replay_index_rows_store_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_index_rows_store_sec")
        ),
        "slab_commit_child_accounted_sec": _float_or_zero(
            totals.get("compact_rollout_slab_commit_child_accounted_sec")
        ),
        "slab_commit_residual_sec": _float_or_zero(
            totals.get("compact_rollout_slab_commit_residual_sec")
        ),
        "slab_replay_index_rows_identity_validate_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_index_rows_identity_validate_sec")
        ),
        "slab_replay_index_rows_terminal_prepare_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_index_rows_terminal_prepare_sec")
        ),
        "slab_replay_index_rows_target_tensor_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_index_rows_target_tensor_sec")
        ),
        "slab_replay_index_rows_scalar_host_pack_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_index_rows_scalar_host_pack_sec")
        ),
        "slab_replay_index_rows_scalar_device_transfer_sec": _float_or_zero(
            totals.get(
                "compact_rollout_slab_replay_index_rows_scalar_device_transfer_sec"
            )
        ),
        "slab_replay_index_rows_metadata_sec": _float_or_zero(
            totals.get("compact_rollout_slab_replay_index_rows_metadata_sec")
        ),
        "slab_replay_index_rows_scalar_packed_h2d_bytes": _float_or_zero(
            totals.get("compact_rollout_slab_replay_index_rows_scalar_packed_h2d_bytes")
        ),
        "slab_replay_index_rows_scalar_tensor_count": _float_or_zero(
            totals.get("compact_rollout_slab_replay_index_rows_scalar_tensor_count")
        ),
    }


def _search_service_flags(
    source_profile: Mapping[str, Any],
    *,
    timing_buckets: Mapping[str, float],
) -> dict[str, Any]:
    last = source_profile.get("compact_rollout_slab_last_telemetry")
    last = last if isinstance(last, Mapping) else {}
    profile = last.get("compact_rollout_slab_profile_telemetry")
    profile = profile if isinstance(profile, Mapping) else {}
    skipped = _first_present(
        last,
        profile,
        "compact_rollout_slab_search_service_one_simulation_root_prior_softmax_skipped",
        "compact_torch_search_one_simulation_root_prior_softmax_skipped",
    )
    mode = _first_present(
        last,
        profile,
        "compact_rollout_slab_search_service_one_simulation_selection_mode",
        "compact_torch_search_one_simulation_selection_mode",
    )
    initial_mode_requested = _first_present(
        last,
        profile,
        "compact_rollout_slab_search_service_initial_inference_mode_requested",
        "compact_torch_search_initial_inference_mode_requested",
    )
    initial_mode_effective = _first_present(
        last,
        profile,
        "compact_rollout_slab_search_service_initial_inference_mode_effective",
        "compact_torch_search_initial_inference_mode_effective",
    )
    initial_runtime_status = _first_present(
        last,
        profile,
        "compact_rollout_slab_search_service_initial_inference_runtime_status",
        "compact_torch_search_initial_inference_runtime_status",
    )
    action_d2h = float(timing_buckets.get("action_d2h_bytes", 0.0))
    fast_path_count = float(
        timing_buckets.get("search_service_one_simulation_fast_path_count", 0.0)
    )
    active_root_count = _float_or_zero(
        last.get("compact_rollout_slab_active_root_count")
    )
    expected_int16_action_d2h = active_root_count * 2.0 * fast_path_count
    return {
        "one_simulation_root_prior_softmax_skipped_last": bool(skipped)
        if skipped is not None
        else False,
        "one_simulation_root_prior_softmax_skipped_count": float(
            timing_buckets.get(
                "search_service_one_simulation_root_prior_softmax_skipped_count",
                0.0,
            )
        ),
        "one_simulation_selection_mode_last": str(mode or ""),
        "initial_inference_mode_requested": str(initial_mode_requested or ""),
        "initial_inference_mode_effective": str(initial_mode_effective or ""),
        "initial_inference_runtime_status": str(initial_runtime_status or ""),
        "initial_inference_direct_requested": (
            float(
                timing_buckets.get(
                    "search_service_initial_inference_direct_requested_count",
                    0.0,
                )
            )
            > 0.0
        ),
        "initial_inference_direct_used": (
            float(
                timing_buckets.get(
                    "search_service_initial_inference_direct_used_count",
                    0.0,
                )
            )
            > 0.0
        ),
        "initial_inference_fallback_count": float(
            timing_buckets.get("search_service_initial_inference_fallback_count", 0.0)
        ),
        "action_readback_bytes_match_int16_selected_actions": bool(
            expected_int16_action_d2h > 0.0
            and action_d2h == expected_int16_action_d2h
        ),
    }


def _first_present(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    primary_key: str,
    secondary_key: str,
) -> Any:
    if primary_key in primary:
        return primary.get(primary_key)
    if secondary_key in secondary:
        return secondary.get(secondary_key)
    return None


def _timing_accounting(timing_buckets: Mapping[str, Any]) -> dict[str, Any]:
    top_level_keys = (
        "actor_step_wall_sec",
        "gather_merge_sec",
        "observation_sec",
        "compact_batch_build_sec",
        "compact_rollout_slab_sec",
        "sample_gate_sec",
        "scalar_materialization_sec",
        "compact_payload_pickle_sec",
        "learner_gate_sec",
    )
    measured_sec = float(timing_buckets["measured_sec"])
    known_sec = sum(float(timing_buckets[key]) for key in top_level_keys)
    slab_sec = float(timing_buckets["compact_rollout_slab_sec"])
    search_service_total_sec = float(timing_buckets["search_service_total_sec"])
    search_service_action_wall_sec = float(
        timing_buckets["search_service_action_wall_sec"]
    )
    slab_search_dispatch_wall_sec = float(
        timing_buckets["slab_search_dispatch_wall_sec"]
    )
    slab_non_service_sec = slab_sec - search_service_total_sec
    return {
        "top_level_known_accounting_sec": known_sec,
        "top_level_accounting_residual_sec": measured_sec - known_sec,
        "compact_rollout_slab_non_service_sec": slab_non_service_sec,
        "slab_search_dispatch_residual_sec": (
            slab_search_dispatch_wall_sec - search_service_action_wall_sec
        ),
        "compact_rollout_slab_non_dispatch_sec": (
            slab_sec - slab_search_dispatch_wall_sec
        ),
        "compact_rollout_slab_non_action_service_sec": (
            slab_sec - search_service_action_wall_sec
        ),
        "search_service_action_wall_over_total_sec": (
            search_service_action_wall_sec - search_service_total_sec
        ),
        "top_level_accounting_component_keys": list(top_level_keys),
    }


def _timing_delta(
    compact_timing: Mapping[str, Any],
    floor_timing: Mapping[str, Any],
    key: str,
) -> float:
    return float(compact_timing[key]) - float(floor_timing[key])


def _accounting_delta(
    compact_accounting: Mapping[str, Any],
    floor_accounting: Mapping[str, Any],
    key: str,
) -> float:
    return float(compact_accounting[key]) - float(floor_accounting[key])


def _trajectory_fingerprints(source_profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "env_action_checksum_total": source_profile.get("env_action_checksum_total"),
        "env_trajectory_checksum_total": source_profile.get(
            "env_trajectory_checksum_total"
        ),
        "last_env_action_checksum": source_profile.get("last_env_action_checksum"),
        "last_env_trajectory_checksum": source_profile.get(
            "last_env_trajectory_checksum"
        ),
        "terminal_row_count": source_profile.get("terminal_row_count"),
    }


def _search_impl(
    *,
    summary: Mapping[str, Any],
    compact: Mapping[str, Any],
    source_profile: Mapping[str, Any],
) -> str:
    impl = summary.get("search_service_impl") or compact.get("search_service_impl")
    if impl:
        return str(impl)
    last = source_profile.get("compact_rollout_slab_last_telemetry")
    if isinstance(last, Mapping):
        impl = last.get("compact_rollout_slab_search_impl")
    return str(impl or "")


def _search_kind(
    *,
    summary: Mapping[str, Any],
    compact: Mapping[str, Any],
    search_impl: str,
) -> str:
    kind = summary.get("search_service_kind") or compact.get("search_service_kind")
    if kind:
        return str(kind)
    if "compact_torch" in search_impl:
        return SEARCH_KIND_COMPACT_TORCH
    if "fixed_shape" in search_impl:
        return SEARCH_KIND_FIXED_SHAPE
    if "device_target" in search_impl:
        return SEARCH_KIND_DEVICE_TARGET
    return "unknown"


def _denominator_value(row: Mapping[str, Any], key: str) -> Any:
    if key in {"candidate_checkpoint_id", "speed_currency", "model_identity_scope", "hardware_class"}:
        return row.get(key)
    shape = _required_mapping(row.get("shape"), f"{row.get('role')}.shape")
    return shape.get(key)


def _hardware_class(path: Path) -> str:
    lowered = path.as_posix().lower()
    if "h100" in lowered:
        return "h100"
    if "l4" in lowered:
        return "l4"
    if "local" in lowered:
        return "local"
    return "unknown"


def _optional_report_refs(
    *,
    compatibility_report_path: str | Path | None,
    unified_lifecycle_report_path: str | Path | None,
) -> dict[str, Any]:
    refs: dict[str, Any] = {}
    if compatibility_report_path is not None:
        path = Path(compatibility_report_path).resolve()
        refs["compatibility_report"] = _file_ref(path, payload=_read_json_mapping(path, "compatibility report"))
    if unified_lifecycle_report_path is not None:
        path = Path(unified_lifecycle_report_path).resolve()
        refs["unified_lifecycle_report"] = _file_ref(path, payload=_read_json_mapping(path, "unified lifecycle report"))
    return refs


def _file_ref(path: Path, *, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_id": str(payload.get("schema_id") or ""),
    }


def _resolve_ref(value: Any, *, base_path: Path) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise CompactSpeedRowFloorBundleError(f"{base_path} missing referenced path")
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (base_path.parent / path).resolve()


def _int_from(
    row: Mapping[str, Any],
    source_profile: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
) -> int:
    value = row.get(key)
    if value is None:
        value = source_profile.get(fallback_key or key)
    return int(value or 0)


def _nested_get(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return math.inf
    return float(numerator) / float(denominator)


def _safe_share(part: float, whole: float) -> float | None:
    whole_abs = abs(float(whole))
    if whole_abs <= 0.0:
        return None
    return abs(float(part)) / whole_abs


def _float_or_zero(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _require_finite_number(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CompactSpeedRowFloorBundleError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise CompactSpeedRowFloorBundleError(f"{name} must be finite")
    return numeric


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactSpeedRowFloorBundleError(f"{name} must be a JSON object")
    return value


def _require_non_empty(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactSpeedRowFloorBundleError(f"{name} must be non-empty")
    return text


def _validate_false_claims(value: Any, name: str) -> None:
    claims = _required_mapping(value, name)
    for key in FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactSpeedRowFloorBundleError(f"{name}.{key} must be false")


def _false_claims() -> dict[str, bool]:
    return {key: False for key in FALSE_CLAIM_KEYS}


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CompactSpeedRowFloorBundleError(f"{name} must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "COMPACT_SPEED_ROW_FLOOR_BUNDLE_MANIFEST_SCHEMA_ID",
    "COMPACT_SPEED_ROW_FLOOR_BUNDLE_SCHEMA_ID",
    "CompactSpeedRowFloorBundleError",
    "build_compact_speed_row_floor_bundle_v1",
    "compact_speed_row_floor_bundle_evidence_ref",
    "validate_compact_speed_row_floor_bundle_v1",
]
