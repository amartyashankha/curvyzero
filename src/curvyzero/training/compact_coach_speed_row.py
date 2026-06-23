"""Hash-bound Coach speed-row evidence for compact-owned candidates."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from curvyzero.training.compact_death_terminal_contract import (
    validate_compact_death_terminal_contract_v1,
)


COMPACT_COACH_SPEED_ROW_EVIDENCE_SCHEMA_ID = "curvyzero_compact_coach_speed_row_evidence/v1"
COMPACT_COACH_SPEED_ROW_MANIFEST_SCHEMA_ID = "curvyzero_compact_coach_speed_row_manifest/v1"
COMPACT_COACH_SPEED_ROW_RESULT_SCHEMA_ID = "curvyzero_compact_coach_speed_row_result/v1"
COMPACT_COACH_SPEED_ROW_PRODUCER_SCHEMA_ID = "curvyzero_compact_coach_speed_row_producer/v1"
COMPACT_COACH_SPEED_ROW_EVIDENCE_REF_PREFIX = "compact_coach_speed_row:"
COMPACT_COACH_SPEED_ROW_STATUS_VERIFIED = "compact_coach_speed_row_verified"
COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER = "compact_owned_trainer"
COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE = "stock_train_muzero_bridge"
COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID = "curvyzero_compact_unified_lifecycle_smoke/v1"
COMPACT_COACH_MODEL_IDENTITY_SCOPE_SUPPORT_ONLY = "candidate_named_support_only"
COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT = "candidate_loaded_checkpoint"
WHOLE_OWNER_BUFFER_REPLAY_CEILING_PREFIX = (
    "compact_whole_owner_buffer_replay_ceiling_"
)
WHOLE_OWNER_BUFFER_REPLAY_CEILING_SCHEMA_ID = (
    "curvyzero_compact_whole_owner_buffer_replay_ceiling/v1"
)
WHOLE_OWNER_BUFFER_REPLAY_CEILING_SPEED_CURRENCY = "local_projection_no_speed"
WHOLE_OWNER_BUFFER_REPLAY_CEILING_PROJECTION_SOURCE = (
    "measured_owner_search_surface_projection_v1"
)

_ALLOWED_SPEED_CURRENCIES_BY_ROUTE = {
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER: frozenset({"compact_trainer_env_steps_per_sec"}),
    COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE: frozenset(
        {"stock_train_muzero_env_steps_per_sec"}
    ),
}

_PROFILE_ONLY_SPEED_CURRENCIES = frozenset(
    {
        "compact_profile_active_roots_per_sec",
        "compact_profile_physical_rows_per_sec",
        "local_compact_owned_loop_profile_only",
        "compact_trainer_lifecycle_evidence_no_speed",
        "compact_trainer_checkpoint_no_speed",
        "stock_train_muzero_profile_env_steps_per_sec",
    }
)
_NONCLAIM_FALSE_KEYS = (
    "promotion_claim",
    "training_speedup_claim",
    "live_run_safety_claim",
    "stock_resume_claim",
    "rating_or_promotion_quality_claim",
)
_IDENTITY_KEYS = (
    "checkpoint_id",
    "trainer_id",
    "policy_version_ref",
    "model_version_ref",
    "policy_source",
    "model_state_digest",
)
_ALLOWED_MODEL_IDENTITY_SCOPES = frozenset(
    {
        COMPACT_COACH_MODEL_IDENTITY_SCOPE_SUPPORT_ONLY,
        COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT,
    }
)
_REPEATABILITY_SEED_AND_SHAPE_FIELDS = (
    "seed",
    "sample_seed_base",
    "sample_batch_size",
    "sample_interval",
    "replay_pair_capacity",
    "learner_train_steps",
    "policy_refresh_interval",
    "num_simulations",
    "compact_rollout_slab_sample_gate_last_seed",
    "compact_rollout_slab_learner_gate_last_seed",
    "compact_owned_loop_sample_gate_last_metadata_seed",
    "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed",
)
_REPEATABILITY_ENV_TRAJECTORY_FIELDS = (
    "env_action_checksum_total",
    "env_done_checksum_total",
    "env_reward_checksum_total",
    "env_action_mask_checksum_total",
    "env_trajectory_checksum_total",
    "env_trajectory_ordered_checksum_total",
    "env_terminal_row_checksum_total",
    "env_autoreset_row_checksum_total",
    "env_terminal_reason_checksum_total",
    "env_death_count_checksum_total",
    "env_death_cause_checksum_total",
    "env_death_hit_owner_checksum_total",
    "last_env_action_checksum",
    "last_env_trajectory_checksum",
    "last_env_terminal_row_checksum",
    "last_env_autoreset_row_checksum",
)
_REPEATABILITY_SAMPLE_IDENTITY_FIELDS = (
    "compact_rollout_slab_sample_gate_action_checksum",
    "compact_rollout_slab_sample_gate_sample_row_checksum",
    "compact_rollout_slab_sample_gate_sample_action_checksum",
    "compact_rollout_slab_sample_gate_sampled_flat_row_checksum",
    "compact_rollout_slab_sample_gate_sample_position_order_checksum",
    "compact_rollout_slab_sample_gate_source_record_pair_checksum",
    "compact_rollout_slab_sample_gate_source_record_window_checksum",
)
_REPEATABILITY_COUNTER_FIELDS = (
    "compact_owned_loop_record_step_calls",
    "compact_owned_loop_appended_replay_entry_count",
    "compact_rollout_slab_sample_gate_sample_rows",
    "compact_rollout_slab_learner_gate_sample_rows",
    "compact_rollout_slab_sample_gate_opportunities",
    "compact_rollout_slab_sample_gate_skipped_count",
    "compact_rollout_slab_sample_gate_calls",
    "compact_rollout_slab_learner_gate_calls",
    "compact_rollout_slab_learner_gate_updates",
    "compact_owned_trainer_sample_batch_count",
    "compact_owned_trainer_learner_update_count",
    "compact_owned_trainer_policy_refresh_count",
    "compact_rollout_slab_committed_index_row_count",
    "compact_rollout_slab_stored_index_row_count",
    "compact_rollout_slab_policy_refresh_after_learner_gate_calls",
    "compact_rollout_slab_policy_refresh_after_learner_gate_interval",
    "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count",
    "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count",
    "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count",
    "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest",
    "compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind",
    "compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind",
    "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count",
    "compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count",
    (
        "compact_rollout_slab_policy_refresh_after_learner_gate_"
        "parent_model_state_transport_avoided"
    ),
)
_REPEATABILITY_RESULT_SUMMARY_FIELDS = (
    *_REPEATABILITY_SEED_AND_SHAPE_FIELDS,
    *_REPEATABILITY_ENV_TRAJECTORY_FIELDS,
    *_REPEATABILITY_SAMPLE_IDENTITY_FIELDS,
    *_REPEATABILITY_COUNTER_FIELDS,
)


class CompactCoachSpeedRowEvidenceError(ValueError):
    """Raised when a compact Coach speed-row evidence object overclaims."""


def build_compact_coach_speed_row_evidence_v1(
    *,
    route: str,
    candidate_checkpoint_id: str,
    unified_lifecycle_report_path: str | Path,
    manifest_path: str | Path,
    row_id: str,
    result_json_path: str | Path,
    speed_currency: str,
    numerator_field: str,
    denominator_field: str,
    numerator_value: int | float | None = None,
    denominator_value_sec: int | float | None = None,
    reported_steps_per_sec: int | float | None = None,
    uses_fallback_denominator: bool = False,
    evidence_id: str | None = None,
) -> dict[str, Any]:
    """Build a hash-bound speed-row evidence object from row/result artifacts."""

    route_value = str(route).strip()
    checkpoint_id = str(candidate_checkpoint_id).strip()
    speed_currency_value = str(speed_currency).strip()
    if not checkpoint_id:
        raise CompactCoachSpeedRowEvidenceError("candidate_checkpoint_id is required")
    if not speed_currency_value:
        raise CompactCoachSpeedRowEvidenceError("speed_currency is required")
    lifecycle_path = Path(unified_lifecycle_report_path)
    manifest_path_obj = Path(manifest_path)
    result_path_obj = Path(result_json_path)
    lifecycle = _read_json_mapping(lifecycle_path, "unified lifecycle report")
    manifest = _read_json_mapping(manifest_path_obj, "speed-row manifest")
    result = _read_json_mapping(result_path_obj, "speed-row result")
    manifest_row = _manifest_row(manifest, row_id=row_id)
    summary = _result_summary(result)
    compact_payload = result.get("compact")
    if compact_payload is not None and not isinstance(compact_payload, Mapping):
        raise CompactCoachSpeedRowEvidenceError("result compact payload must be a mapping")
    lifecycle_identity = _lifecycle_identity_from_report(
        lifecycle,
        lifecycle_path=lifecycle_path,
    )
    result_identity = (
        _result_loaded_checkpoint_identity(compact_payload)
        if isinstance(compact_payload, Mapping)
        else {}
    )
    model_identity_scope = _model_identity_scope_from_result(compact_payload)
    numerator = (
        _require_positive_number(summary.get(numerator_field), numerator_field)
        if numerator_value is None
        else _require_positive_number(numerator_value, "numerator_value")
    )
    denominator = (
        _require_positive_number(summary.get(denominator_field), denominator_field)
        if denominator_value_sec is None
        else _require_positive_number(denominator_value_sec, "denominator_value_sec")
    )
    reported = (
        _require_positive_number(summary.get("steps_per_sec"), "steps_per_sec")
        if reported_steps_per_sec is None
        else _require_positive_number(reported_steps_per_sec, "reported_steps_per_sec")
    )
    manifest_experiment_id = str(
        manifest.get("experiment_id")
        or manifest_row.get("experiment_id")
        or manifest.get("run_id")
        or "unknown-experiment"
    ).strip()
    if not manifest_experiment_id or manifest_experiment_id == "unknown-experiment":
        raise CompactCoachSpeedRowEvidenceError("manifest experiment_id is required")
    result_sha = _file_sha256(result_path_obj)
    operational_surface = _speed_row_operational_surface(
        manifest=manifest,
        manifest_row=manifest_row,
        summary=summary,
        compact_payload=compact_payload,
    )
    evidence = {
        "schema_id": COMPACT_COACH_SPEED_ROW_EVIDENCE_SCHEMA_ID,
        "compact_coach_speed_row_evidence": True,
        "ok": True,
        "status": COMPACT_COACH_SPEED_ROW_STATUS_VERIFIED,
        "evidence_id": str(evidence_id or f"{manifest_experiment_id}:{row_id}"),
        "route": route_value,
        "candidate_checkpoint_id": checkpoint_id,
        "unified_lifecycle": {
            "path": str(lifecycle_path),
            "sha256": _file_sha256(lifecycle_path),
            "schema_id": lifecycle.get("schema_id"),
            "ok": lifecycle.get("ok"),
            "checkpoint_id": lifecycle.get("checkpoint_id"),
            "lifecycle_gates_complete": lifecycle.get("lifecycle_gates_complete"),
            "missing_required_gates": list(lifecycle.get("missing_required_gates") or []),
            "promotion_eligible": lifecycle.get("promotion_eligible"),
        },
        "manifest": {
            "path": str(manifest_path_obj),
            "sha256": _file_sha256(manifest_path_obj),
            "schema_id": manifest.get("schema_id"),
            "experiment_id": manifest_experiment_id,
            "candidate_checkpoint_id": manifest.get("candidate_checkpoint_id"),
            "route": manifest.get("route"),
            "profile_only": manifest.get("profile_only"),
            "calls_train_muzero": manifest.get("calls_train_muzero"),
            "touches_live_runs": manifest.get("touches_live_runs"),
            "non_claims": {
                key: _first_present(
                    manifest.get(key),
                    manifest.get("non_claims", {}).get(key)
                    if isinstance(manifest.get("non_claims"), Mapping)
                    else None,
                    False,
                )
                for key in _NONCLAIM_FALSE_KEYS
            },
        },
        "row": {
            "row_id": str(row_id),
            "sha256": _json_sha256(manifest_row),
            "candidate_checkpoint_id": manifest_row.get("candidate_checkpoint_id"),
            "route": manifest_row.get("route"),
            "profile_only": manifest_row.get("profile_only"),
            "calls_train_muzero": manifest_row.get("calls_train_muzero"),
            "touches_live_runs": manifest_row.get("touches_live_runs"),
            "row_purpose": manifest_row.get("row_purpose"),
            "speed_currency": manifest_row.get("speed_currency"),
            "promotion_claim": manifest_row.get("promotion_claim"),
            "command_sha256": _json_sha256(manifest_row.get("command", [])),
        },
        "search_config": _speed_row_search_config(
            manifest=manifest,
            manifest_row=manifest_row,
            summary=summary,
            compact_payload=compact_payload,
        ),
        "actor_handoff_config": _speed_row_actor_handoff_config(
            manifest=manifest,
            manifest_row=manifest_row,
            summary=summary,
            compact_payload=compact_payload,
        ),
        "operational_surface": operational_surface,
        "terminal_death_proof": _speed_row_terminal_death_proof(
            operational_surface=operational_surface,
            summary=summary,
            compact_payload=compact_payload,
        ),
        "result": {
            "path": str(result_path_obj),
            "sha256": result_sha,
            "schema_id": result.get("schema_id"),
            "ok": result.get("ok"),
            "status": result.get("status"),
            "problem": result.get("problem"),
            "returncode": result.get("returncode"),
            "run_invocation_id": result.get("run_invocation_id"),
            "candidate_checkpoint_id": result.get("candidate_checkpoint_id"),
            "row_id": str(result.get("row_id") or summary.get("row_id") or ""),
            "row_sha256": (
                _json_sha256(result.get("row")) if isinstance(result.get("row"), Mapping) else None
            ),
            "producer_sha256": (
                _json_sha256(result.get("producer"))
                if isinstance(result.get("producer"), Mapping)
                else None
            ),
            "summary_sha256": _json_sha256(summary),
            "compact_payload_sha256": (
                _json_sha256(compact_payload) if isinstance(compact_payload, Mapping) else None
            ),
            "function_call_id": result.get("function_call_id"),
        },
        "result_summary": {
            "profile_only": summary.get("profile_only"),
            "calls_train_muzero": summary.get("calls_train_muzero"),
            "touches_live_runs": summary.get("touches_live_runs"),
            "status": summary.get("status"),
            "ok": summary.get("ok"),
            "row_id": summary.get("row_id"),
            "candidate_checkpoint_id": summary.get("candidate_checkpoint_id"),
            "route": summary.get("route"),
            "row_purpose": summary.get("row_purpose"),
            "promotion_claim": summary.get("promotion_claim"),
            "speed_currency": summary.get("speed_currency"),
            "search_service_kind": summary.get("search_service_kind"),
            "search_service_impl": summary.get("search_service_impl"),
            "compact_torch_initial_inference_mode": summary.get(
                "compact_torch_initial_inference_mode"
            ),
            "compact_torch_observation_memory_format": summary.get(
                "compact_torch_observation_memory_format"
            ),
            "compact_torch_model_memory_format": summary.get("compact_torch_model_memory_format"),
            "compact_torch_defer_one_simulation_replay_payload_requested": summary.get(
                "compact_torch_defer_one_simulation_replay_payload_requested"
            ),
            "compact_torch_memory_format_applies_to_search_service": summary.get(
                "compact_torch_memory_format_applies_to_search_service"
            ),
            "hybrid_persistent_compact_render_state_buffer": summary.get(
                "hybrid_persistent_compact_render_state_buffer"
            ),
            "hybrid_borrow_single_actor_render_state": summary.get(
                "hybrid_borrow_single_actor_render_state"
            ),
            "render_state_handoff_mode": summary.get("render_state_handoff_mode"),
            "render_state_copy_steps": summary.get("render_state_copy_steps"),
            "render_state_borrowed_steps": summary.get("render_state_borrowed_steps"),
            "render_state_row_overlay_steps": summary.get("render_state_row_overlay_steps"),
            "render_state_row_overlay_rows": summary.get("render_state_row_overlay_rows"),
            "render_state_row_overlay_bytes": summary.get("render_state_row_overlay_bytes"),
            "actor_count": summary.get("actor_count"),
            "batch_size": summary.get("batch_size"),
            "steps": summary.get("steps"),
            "warmup_steps": summary.get("warmup_steps"),
            "death_mode": summary.get("death_mode"),
            "compact_owned_training_loop_owner": summary.get("compact_owned_training_loop_owner"),
            "compact_owned_trainer_config_death_mode": summary.get(
                "compact_owned_trainer_config_death_mode"
            ),
            "normal_death_terminal_contract_owner": summary.get(
                "normal_death_terminal_contract_owner"
            ),
            "terminal_row_count": summary.get("terminal_row_count"),
            "death_row_count": summary.get("death_row_count"),
            "terminated_row_count": summary.get("terminated_row_count"),
            "truncated_row_count": summary.get("truncated_row_count"),
            "terminal_final_observation_row_count": summary.get(
                "terminal_final_observation_row_count"
            ),
            "terminal_final_observation_before_autoreset_verified": summary.get(
                "terminal_final_observation_before_autoreset_verified"
            ),
            "compact_profile_autoreset_direct_count": summary.get(
                "compact_profile_autoreset_direct_count"
            ),
            "compact_profile_autoreset_template_copy_skipped_count": summary.get(
                "compact_profile_autoreset_template_copy_skipped_count"
            ),
            "compact_profile_autoreset_direct_row_count": summary.get(
                "compact_profile_autoreset_direct_row_count"
            ),
            "terminal_sample_row_count": summary.get("terminal_sample_row_count"),
            "terminal_unroll_value_target_mode": summary.get("terminal_unroll_value_target_mode"),
            "terminal_unroll_value_target_row_count": summary.get(
                "terminal_unroll_value_target_row_count"
            ),
            "resident_observation_host_fallback_count": summary.get(
                "resident_observation_host_fallback_count"
            ),
            "normal_death_terminal_contract_promotion_gate_satisfied": summary.get(
                "normal_death_terminal_contract_promotion_gate_satisfied"
            ),
            "learner_num_unroll_steps": summary.get("learner_num_unroll_steps"),
            "compact_owned_loop_fused_learner_batch": summary.get(
                "compact_owned_loop_fused_learner_batch"
            ),
            "compact_owner_search_action_only_result": summary.get(
                "compact_owner_search_action_only_result"
            ),
            "compact_owner_search_owner_materializes_replay": summary.get(
                "compact_owner_search_owner_materializes_replay"
            ),
            "compact_owner_search_parent_slab_commits_replay": summary.get(
                "compact_owner_search_parent_slab_commits_replay"
            ),
            "compact_owner_action_step_boundary_enabled": summary.get(
                "compact_owner_action_step_boundary_enabled"
            ),
            "compact_owner_action_step_boundary_proof_passed": summary.get(
                "compact_owner_action_step_boundary_proof_passed"
            ),
            "compact_owner_action_step_boundary_step_count": summary.get(
                "compact_owner_action_step_boundary_step_count"
            ),
            "compact_owner_mechanics_step_boundary_enabled": summary.get(
                "compact_owner_mechanics_step_boundary_enabled"
            ),
            "compact_owner_mechanics_step_boundary": summary.get(
                "compact_owner_mechanics_step_boundary"
            ),
            "compact_owner_mechanics_step_view_schema_id": summary.get(
                "compact_owner_mechanics_step_view_schema_id"
            ),
            "compact_owner_mechanics_step_frame_slot_schema_id": summary.get(
                "compact_owner_mechanics_step_frame_slot_schema_id"
            ),
            "compact_owner_mechanics_step_boundary_count": summary.get(
                "compact_owner_mechanics_step_boundary_count"
            ),
            "compact_owner_mechanics_parent_compact_batch_builder_call_count": summary.get(
                "compact_owner_mechanics_parent_compact_batch_builder_call_count"
            ),
            "compact_owner_mechanics_parent_compact_batch_object_count": summary.get(
                "compact_owner_mechanics_parent_compact_batch_object_count"
            ),
            "compact_owner_mechanics_parent_compact_batch_builder_used": summary.get(
                "compact_owner_mechanics_parent_compact_batch_builder_used"
            ),
            "compact_owner_mechanics_step_view_object_count": summary.get(
                "compact_owner_mechanics_step_view_object_count"
            ),
            "compact_owner_mechanics_host_observation_bytes_sent": summary.get(
                "compact_owner_mechanics_host_observation_bytes_sent"
            ),
            "compact_owner_mechanics_host_final_observation_bytes_sent": summary.get(
                "compact_owner_mechanics_host_final_observation_bytes_sent"
            ),
            "compact_owner_mechanics_resident_observation_handle_present": summary.get(
                "compact_owner_mechanics_resident_observation_handle_present"
            ),
            "compact_owner_mechanics_step_frame_handle_schema_id": summary.get(
                "compact_owner_mechanics_step_frame_handle_schema_id"
            ),
            "compact_owner_mechanics_step_frame_handle_ring_used": summary.get(
                "compact_owner_mechanics_step_frame_handle_ring_used"
            ),
            "compact_owner_mechanics_step_frame_handle_published": summary.get(
                "compact_owner_mechanics_step_frame_handle_published"
            ),
            "compact_owner_mechanics_step_frame_handle_consumed": summary.get(
                "compact_owner_mechanics_step_frame_handle_consumed"
            ),
            "compact_owner_mechanics_step_frame_handle_publish_count": summary.get(
                "compact_owner_mechanics_step_frame_handle_publish_count"
            ),
            "compact_owner_mechanics_step_frame_handle_consume_count": summary.get(
                "compact_owner_mechanics_step_frame_handle_consume_count"
            ),
            "compact_owner_mechanics_step_frame_handle_ring_slot_count": summary.get(
                "compact_owner_mechanics_step_frame_handle_ring_slot_count"
            ),
            "compact_owner_mechanics_step_frame_handle_slot_id": summary.get(
                "compact_owner_mechanics_step_frame_handle_slot_id"
            ),
            "compact_owner_mechanics_step_frame_handle_generation": summary.get(
                "compact_owner_mechanics_step_frame_handle_generation"
            ),
            "compact_owner_mechanics_step_frame_handle_digest": summary.get(
                "compact_owner_mechanics_step_frame_handle_digest"
            ),
            "compact_owner_mechanics_step_frame_handle_digest_verified": summary.get(
                "compact_owner_mechanics_step_frame_handle_digest_verified"
            ),
            "compact_owner_mechanics_step_frame_handle_owner_digest_verified": summary.get(
                "compact_owner_mechanics_step_frame_handle_owner_digest_verified"
            ),
            "compact_owner_mechanics_step_frame_handle_resident_observation_present": summary.get(
                "compact_owner_mechanics_step_frame_handle_resident_observation_present"
            ),
            "compact_owner_mechanics_step_frame_slot_write_count": summary.get(
                "compact_owner_mechanics_step_frame_slot_write_count"
            ),
            "compact_owner_mechanics_parent_step_frame_build_count": summary.get(
                "compact_owner_mechanics_parent_step_frame_build_count"
            ),
            "compact_owner_step_frame_root_build_request_used": summary.get(
                "compact_owner_step_frame_root_build_request_used"
            ),
            "compact_owner_step_frame_root_build_request_from_batch_helper_used": (
                summary.get(
                    "compact_owner_step_frame_root_build_request_from_batch_helper_used"
                )
            ),
            "compact_owner_step_frame_root_request_sidecar_array_bytes": summary.get(
                "compact_owner_step_frame_root_request_sidecar_array_bytes"
            ),
            "compact_owner_step_frame_root_request_sidecar_field_count": summary.get(
                "compact_owner_step_frame_root_request_sidecar_field_count"
            ),
            "compact_owner_root_action_context_handle_used": summary.get(
                "compact_owner_root_action_context_handle_used"
            ),
            "compact_owner_root_action_context_handle_schema_id": summary.get(
                "compact_owner_root_action_context_handle_schema_id"
            ),
            "compact_owner_root_action_context_handle_id": summary.get(
                "compact_owner_root_action_context_handle_id"
            ),
            "compact_owner_root_action_context_transaction_id": summary.get(
                "compact_owner_root_action_context_transaction_id"
            ),
            "compact_owner_root_action_context_dispatch_id": summary.get(
                "compact_owner_root_action_context_dispatch_id"
            ),
            "compact_owner_root_action_context_root_count": summary.get(
                "compact_owner_root_action_context_root_count"
            ),
            "compact_owner_root_action_context_active_root_count": summary.get(
                "compact_owner_root_action_context_active_root_count"
            ),
            "compact_owner_root_action_context_context_digest": summary.get(
                "compact_owner_root_action_context_context_digest"
            ),
            "compact_owner_root_action_context_owner_store_count": summary.get(
                "compact_owner_root_action_context_owner_store_count"
            ),
            "compact_owner_root_action_context_owner_resolve_count": summary.get(
                "compact_owner_root_action_context_owner_resolve_count"
            ),
            "compact_owner_root_action_context_owner_release_count": summary.get(
                "compact_owner_root_action_context_owner_release_count"
            ),
            "compact_owner_root_action_context_owner_pending_count": summary.get(
                "compact_owner_root_action_context_owner_pending_count"
            ),
            "compact_owner_root_action_context_owner_max_pending_count": summary.get(
                "compact_owner_root_action_context_owner_max_pending_count"
            ),
            "compact_owner_root_action_context_owner_digest_verified": summary.get(
                "compact_owner_root_action_context_owner_digest_verified"
            ),
            "compact_owner_search_pending_root_action_context_stored": summary.get(
                "compact_owner_search_pending_root_action_context_stored"
            ),
            "compact_owner_search_action_dispatch_pending_root_action_context_stored": (
                summary.get(
                    "compact_owner_search_action_dispatch_pending_root_action_context_stored"
                )
            ),
            "compact_owner_search_action_dispatch_pending_root_action_context_store_count": (
                summary.get(
                    "compact_owner_search_action_dispatch_pending_root_action_context_store_count"
                )
            ),
            (
                "compact_owner_search_action_dispatch_pending_root_action_context_"
                "avoided_count"
            ): summary.get(
                "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count"
            ),
            "compact_owner_search_parent_action_context_validation_count": summary.get(
                "compact_owner_search_parent_action_context_validation_count"
            ),
            "compact_owner_search_owner_action_context_validation_count": summary.get(
                "compact_owner_search_owner_action_context_validation_count"
            ),
            "compact_owner_root_search_transaction_boundary_supported": summary.get(
                "compact_owner_root_search_transaction_boundary_supported"
            ),
            "compact_owner_root_search_transaction_requested": summary.get(
                "compact_owner_root_search_transaction_requested"
            ),
            "compact_owner_root_search_transaction_used": summary.get(
                "compact_owner_root_search_transaction_used"
            ),
            "compact_owner_root_search_transaction_schema_id": summary.get(
                "compact_owner_root_search_transaction_schema_id"
            ),
            "compact_owner_root_search_transaction_id": summary.get(
                "compact_owner_root_search_transaction_id"
            ),
            "compact_owner_root_search_transaction_begin_count": summary.get(
                "compact_owner_root_search_transaction_begin_count"
            ),
            "compact_owner_root_search_transaction_submit_count": summary.get(
                "compact_owner_root_search_transaction_submit_count"
            ),
            "compact_owner_root_search_transaction_resolve_count": summary.get(
                "compact_owner_root_search_transaction_resolve_count"
            ),
            "compact_owner_root_search_transaction_pending_count": summary.get(
                "compact_owner_root_search_transaction_pending_count"
            ),
            "compact_owner_root_search_transaction_max_pending_count": summary.get(
                "compact_owner_root_search_transaction_max_pending_count"
            ),
            "compact_owner_root_search_transaction_parent_root_request_build_count": (
                summary.get(
                    "compact_owner_root_search_transaction_parent_root_request_build_count"
                )
            ),
            "compact_owner_root_search_transaction_parent_root_request_stored": (
                summary.get(
                    "compact_owner_root_search_transaction_parent_root_request_stored"
                )
            ),
            "compact_owner_root_search_transaction_parent_compact_batch_stored": (
                summary.get(
                    "compact_owner_root_search_transaction_parent_compact_batch_stored"
                )
            ),
            "compact_owner_root_search_transaction_parent_rebuild_count": summary.get(
                "compact_owner_root_search_transaction_parent_rebuild_count"
            ),
            "compact_owner_root_search_transaction_parent_root_action_context_stored": (
                summary.get(
                    "compact_owner_root_search_transaction_parent_root_action_context_stored"
                )
            ),
            "compact_owner_root_search_transaction_parent_root_action_context_store_count": (
                summary.get(
                    "compact_owner_root_search_transaction_parent_root_action_context_store_count"
                )
            ),
            "compact_owner_root_search_transaction_parent_root_action_context_array_bytes": (
                summary.get(
                    "compact_owner_root_search_transaction_parent_root_action_context_array_bytes"
                )
            ),
            "compact_owner_root_search_transaction_parent_root_action_context_field_count": (
                summary.get(
                    "compact_owner_root_search_transaction_parent_root_action_context_field_count"
                )
            ),
            "compact_owner_root_search_transaction_owner_root_request_build_count": (
                summary.get(
                    "compact_owner_root_search_transaction_owner_root_request_build_count"
                )
            ),
            "compact_owner_root_search_transaction_owner_root_request_build_sec": (
                summary.get(
                    "compact_owner_root_search_transaction_owner_root_request_build_sec"
                )
            ),
            "compact_owner_root_search_transaction_owner_root_store_publish_count": (
                summary.get(
                    "compact_owner_root_search_transaction_owner_root_store_publish_count"
                )
            ),
            "compact_owner_root_search_transaction_frame_generation_verified": (
                summary.get(
                    "compact_owner_root_search_transaction_frame_generation_verified"
                )
            ),
            "compact_owner_root_search_transaction_frame_digest_verified": summary.get(
                "compact_owner_root_search_transaction_frame_digest_verified"
            ),
            "compact_owner_root_search_transaction_action_identity_verified": summary.get(
                "compact_owner_root_search_transaction_action_identity_verified"
            ),
            "compact_owner_root_search_transaction_proxy_transition_closure_used": (
                summary.get(
                    "compact_owner_root_search_transaction_proxy_transition_closure_used"
                )
            ),
            "compact_owner_root_search_transaction_applied_action_mismatch_count": (
                summary.get(
                    "compact_owner_root_search_transaction_applied_action_mismatch_count"
                )
            ),
            "compact_owner_search_learner_update_count": summary.get(
                "compact_owner_search_learner_update_count"
            ),
            "compact_owner_search_owner_train_request_count": summary.get(
                "compact_owner_search_owner_train_request_count"
            ),
            "compact_owner_search_owner_expected_train_request_count": summary.get(
                "compact_owner_search_owner_expected_train_request_count"
            ),
            "compact_owner_search_owner_learner_update_count": summary.get(
                "compact_owner_search_owner_learner_update_count"
            ),
            "compact_owner_search_worker_learner_train_sec": summary.get(
                "compact_owner_search_worker_learner_train_sec"
            ),
            "compact_owner_search_owner_train_wall_sec": summary.get(
                "compact_owner_search_owner_train_wall_sec"
            ),
            "speed_row_total_owner_search_worker_learner_train_sec": summary.get(
                "speed_row_total_owner_search_worker_learner_train_sec"
            ),
            "compact_rollout_slab_sample_gate_calls": summary.get(
                "compact_rollout_slab_sample_gate_calls"
            ),
            "compact_rollout_slab_learner_gate_calls": summary.get(
                "compact_rollout_slab_learner_gate_calls"
            ),
            "compact_muzero_learner_batch_unroll2_specialized_builder": summary.get(
                "compact_muzero_learner_batch_unroll2_specialized_builder"
            ),
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": summary.get(
                "compact_muzero_learner_batch_learner_ready_unroll2_cache"
            ),
            "compact_muzero_learner_batch_tensor_native_replay": summary.get(
                "compact_muzero_learner_batch_tensor_native_replay"
            ),
            "compact_rollout_slab_sample_gate_sec": summary.get(
                "compact_rollout_slab_sample_gate_sec"
            ),
            "compact_rollout_slab_learner_gate_sec": summary.get(
                "compact_rollout_slab_learner_gate_sec"
            ),
            "source_profile_total_sec": summary.get("source_profile_total_sec"),
            "source_profile_warmup_sec": summary.get("source_profile_warmup_sec"),
            "source_profile_measured_sec": summary.get("source_profile_measured_sec"),
            "source_profile_timing_per_timestep_sec": summary.get(
                "source_profile_timing_per_timestep_sec"
            ),
            "speed_row_actor_step_wall_sec": summary.get("speed_row_actor_step_wall_sec"),
            "speed_row_observation_sec": summary.get("speed_row_observation_sec"),
            "speed_row_renderer_stack_update_sec": summary.get(
                "speed_row_renderer_stack_update_sec"
            ),
            "speed_row_compact_rollout_slab_sec": summary.get("speed_row_compact_rollout_slab_sec"),
            "speed_row_sample_gate_sec": summary.get("speed_row_sample_gate_sec"),
            "speed_row_learner_gate_sec": summary.get("speed_row_learner_gate_sec"),
            "speed_row_policy_refresh_sec": summary.get("speed_row_policy_refresh_sec"),
            "speed_row_primary_accounted_sec": summary.get("speed_row_primary_accounted_sec"),
            "speed_row_primary_residual_sec": summary.get("speed_row_primary_residual_sec"),
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": summary.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch"
            ),
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": summary.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"
            ),
            "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": summary.get(
                "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch"
            ),
            "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": summary.get(
                "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"
            ),
            "compact_rollout_slab_sample_gate_host_provider_learner_batch": summary.get(
                "compact_rollout_slab_sample_gate_host_provider_learner_batch"
            ),
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source": summary.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source"
            ),
            "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes": summary.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes"
            ),
            "compact_rollout_slab_learner_gate_prebuilt_batch_used": summary.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_used"
            ),
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order": summary.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order"
            ),
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order": summary.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order"
            ),
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count": summary.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_impl": summary.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_build_impl"
                )
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_direct_build_used": summary.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_direct_build_used"
                )
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count": summary.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_reused_record_count"
                )
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count": summary.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_missing_record_count"
                )
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec": summary.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_concat_sec"
                )
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_sec": summary.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_build_sec"
                )
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec": summary.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_gather_sec"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_prebuilt_path_requested"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_prebuilt_path_eligible"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_prebuilt_path_used"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_prebuilt_fallback_count"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_prebuilt_fallback_reason"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_group_object_count"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_group_object_build_skipped"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_path_requested"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_path_used"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_selected_group_count"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_requested"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_used"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_record_count"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_missing_record_count": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_missing_record_count"
                )
            ),
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_rows": summary.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_rows"
                )
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_requested": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_requested"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_used": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_used"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_schema_id": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_schema_id"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_requested": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_requested"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_used": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_used"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_handle_id": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_handle_id"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_snapshot_version": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_snapshot_version"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_request_checksum": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_request_checksum"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_sample_row_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_sample_row_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_target_row_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_target_row_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_create_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_create_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_resolve_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_resolve_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_inline_resolve_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_inline_resolve_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_reason": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_reason"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_pending_handle_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_pending_handle_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_slot_write_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_slot_write_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_record_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_record_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_table_row_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_table_row_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_fallback_count": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_fallback_count"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec"
            ),
            "compact_rollout_slab_sample_gate_fixed_soa_total_sec": summary.get(
                "compact_rollout_slab_sample_gate_fixed_soa_total_sec"
            ),
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path"
            ),
            "compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled": summary.get(
                "compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled"
            ),
            "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep": summary.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep"
            ),
            "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used": summary.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used"
            ),
            "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count": summary.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count"
            ),
            "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count": summary.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count"
            ),
            "compact_muzero_learner_prebuilt_batch_used": summary.get(
                "compact_muzero_learner_prebuilt_batch_used"
            ),
            **{key: summary.get(key) for key in _REPEATABILITY_RESULT_SUMMARY_FIELDS},
        },
        "denominator": {
            "row_purpose": manifest_row.get("row_purpose") or summary.get("row_purpose"),
            "speed_currency": speed_currency_value,
            "numerator_field": str(numerator_field),
            "numerator_value": numerator,
            "denominator_field": str(denominator_field),
            "denominator_value_sec": denominator,
            "reported_steps_per_sec": reported,
            "uses_fallback_denominator": bool(uses_fallback_denominator),
        },
        "model_identity": {
            "scope": model_identity_scope,
            "lifecycle_identity": lifecycle_identity,
            "result_loaded_checkpoint_identity": result_identity,
        },
        "non_claims": {key: False for key in _NONCLAIM_FALSE_KEYS},
    }
    evidence["evidence_ref"] = _unchecked_compact_coach_speed_row_evidence_ref(evidence)
    validate_compact_coach_speed_row_evidence_v1(evidence)
    return evidence


def validate_compact_coach_speed_row_evidence_v1(
    evidence: Mapping[str, Any],
) -> None:
    """Validate speed-row evidence and referenced file hashes."""

    if evidence.get("schema_id") != COMPACT_COACH_SPEED_ROW_EVIDENCE_SCHEMA_ID:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row evidence schema mismatch")
    if evidence.get("compact_coach_speed_row_evidence") is not True:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row marker missing")
    if evidence.get("ok") is not True:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row evidence must be ok=true")
    if evidence.get("status") != COMPACT_COACH_SPEED_ROW_STATUS_VERIFIED:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row evidence status mismatch")
    route = str(evidence.get("route", "")).strip()
    if route not in {
        COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE,
    }:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row route mismatch")
    if not str(evidence.get("candidate_checkpoint_id", "")).strip():
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row candidate_checkpoint_id missing")
    _validate_lifecycle_section(evidence)
    _validate_manifest_section(evidence)
    _validate_row_section(evidence)
    _validate_result_section(evidence)
    _validate_search_config_section(evidence)
    _validate_actor_handoff_config_section(evidence)
    _validate_operational_surface_section(evidence)
    _validate_terminal_death_proof_section(evidence)
    _validate_denominator_section(evidence)
    _validate_model_identity_section(evidence)
    _validate_nonclaims(evidence.get("non_claims"), "Coach speed-row non_claims")
    expected_ref = _unchecked_compact_coach_speed_row_evidence_ref(evidence)
    if str(evidence.get("evidence_ref", "")) != expected_ref:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row evidence_ref mismatch")


def validate_compact_coach_speed_row_evidence_matches_report_v1(
    evidence: Mapping[str, Any],
    *,
    evidence_ref: str,
    route: str,
    speed_currency: str,
) -> None:
    """Validate speed evidence against a Coach-compatibility report."""

    validate_compact_coach_speed_row_evidence_v1(evidence)
    if str(evidence.get("evidence_ref")) != str(evidence_ref):
        raise CompactCoachSpeedRowEvidenceError(
            "Coach speed-row structured evidence does not match evidence ref"
        )
    if str(evidence.get("route")) != str(route):
        raise CompactCoachSpeedRowEvidenceError(
            "Coach speed-row evidence route does not match compatibility report"
        )
    denominator = evidence.get("denominator")
    if not isinstance(denominator, Mapping):
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row missing denominator")
    if str(denominator.get("speed_currency")) != str(speed_currency):
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row evidence speed_currency mismatch")
    model_identity = _required_mapping(evidence.get("model_identity"), "model_identity")
    if str(model_identity.get("scope")) != COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT:
        raise CompactCoachSpeedRowEvidenceError(
            "Coach speed-row evidence must prove loaded checkpoint identity"
        )


def compact_coach_speed_row_evidence_ref(evidence: Mapping[str, Any]) -> str:
    """Return the stable evidence ref for Coach compatibility metadata."""

    validate_compact_coach_speed_row_evidence_v1(evidence)
    return _unchecked_compact_coach_speed_row_evidence_ref(evidence)


def compact_coach_speed_row_evidence_path(path: str | Path) -> Path:
    """Return the sibling evidence path for a speed-row result JSON."""

    return Path(f"{Path(path)}.compact_coach_speed_row.evidence.json")


def save_compact_coach_speed_row_evidence_v1(
    *,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build and write compact Coach speed-row evidence."""

    evidence = build_compact_coach_speed_row_evidence_v1(**kwargs)
    result_json_path = kwargs.get("result_json_path")
    path = (
        compact_coach_speed_row_evidence_path(Path(str(result_json_path)))
        if output_path is None
        else Path(output_path)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain_value(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"evidence": evidence, "path": path}


def _validate_lifecycle_section(evidence: Mapping[str, Any]) -> None:
    lifecycle = _required_mapping(evidence.get("unified_lifecycle"), "unified_lifecycle")
    _require_sha(lifecycle.get("sha256"), "unified_lifecycle.sha256")
    path = Path(str(lifecycle.get("path", "")))
    loaded = _read_json_mapping(path, "unified lifecycle report")
    if _file_sha256(path) != lifecycle.get("sha256"):
        raise CompactCoachSpeedRowEvidenceError("unified lifecycle sha256 mismatch")
    if loaded.get("schema_id") != COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID:
        raise CompactCoachSpeedRowEvidenceError("unified lifecycle schema mismatch")
    if loaded.get("ok") is not True:
        raise CompactCoachSpeedRowEvidenceError("unified lifecycle ok must be true")
    if loaded.get("lifecycle_gates_complete") is not True:
        raise CompactCoachSpeedRowEvidenceError("unified lifecycle gates not complete")
    if str(loaded.get("checkpoint_id", "")) != str(evidence.get("candidate_checkpoint_id", "")):
        raise CompactCoachSpeedRowEvidenceError("unified lifecycle checkpoint_id mismatch")
    if list(loaded.get("missing_required_gates") or []) != ["coach_speed_row"]:
        raise CompactCoachSpeedRowEvidenceError(
            "unified lifecycle must only be missing coach_speed_row before speed proof"
        )
    if loaded.get("promotion_eligible") is not False:
        raise CompactCoachSpeedRowEvidenceError(
            "unified lifecycle must not already be promotion eligible"
        )
    for key in (
        "schema_id",
        "ok",
        "checkpoint_id",
        "lifecycle_gates_complete",
        "promotion_eligible",
    ):
        if lifecycle.get(key) != loaded.get(key):
            raise CompactCoachSpeedRowEvidenceError(f"unified lifecycle recorded {key} mismatch")


def _validate_manifest_section(evidence: Mapping[str, Any]) -> None:
    manifest_info = _required_mapping(evidence.get("manifest"), "manifest")
    _require_sha(manifest_info.get("sha256"), "manifest.sha256")
    path = Path(str(manifest_info.get("path", "")))
    loaded = _read_json_mapping(path, "speed-row manifest")
    if _file_sha256(path) != manifest_info.get("sha256"):
        raise CompactCoachSpeedRowEvidenceError("manifest sha256 mismatch")
    for key in ("schema_id", "profile_only", "calls_train_muzero", "touches_live_runs"):
        if manifest_info.get(key) != loaded.get(key):
            raise CompactCoachSpeedRowEvidenceError(f"manifest {key} mismatch")
    if loaded.get("profile_only") is not False:
        raise CompactCoachSpeedRowEvidenceError(
            "Coach speed-row manifest must be profile_only=false"
        )
    if loaded.get("schema_id") != COMPACT_COACH_SPEED_ROW_MANIFEST_SCHEMA_ID:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row manifest schema mismatch")
    if manifest_info.get("schema_id") != COMPACT_COACH_SPEED_ROW_MANIFEST_SCHEMA_ID:
        raise CompactCoachSpeedRowEvidenceError("recorded Coach speed-row manifest schema mismatch")
    if str(loaded.get("candidate_checkpoint_id", "")) != str(
        evidence.get("candidate_checkpoint_id", "")
    ):
        raise CompactCoachSpeedRowEvidenceError("manifest candidate_checkpoint_id mismatch")
    if str(manifest_info.get("candidate_checkpoint_id", "")) != str(
        loaded.get("candidate_checkpoint_id", "")
    ):
        raise CompactCoachSpeedRowEvidenceError(
            "recorded manifest candidate_checkpoint_id mismatch"
        )
    if str(loaded.get("route", "")) != str(evidence.get("route", "")):
        raise CompactCoachSpeedRowEvidenceError("manifest route mismatch")
    if str(manifest_info.get("route", "")) != str(loaded.get("route", "")):
        raise CompactCoachSpeedRowEvidenceError("recorded manifest route mismatch")
    if loaded.get("touches_live_runs") is not False:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row manifest must not touch live runs")
    route = str(evidence.get("route"))
    expected_calls = route == COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE
    if loaded.get("calls_train_muzero") is not expected_calls:
        raise CompactCoachSpeedRowEvidenceError(
            "Coach speed-row manifest calls_train_muzero does not match route"
        )
    _validate_nonclaims(_nonclaims_from_loaded(loaded), "loaded manifest non_claims")
    _validate_nonclaims(manifest_info.get("non_claims"), "manifest non_claims")
    if not str(manifest_info.get("experiment_id", "")).strip():
        raise CompactCoachSpeedRowEvidenceError("manifest experiment_id missing")


def _validate_row_section(evidence: Mapping[str, Any]) -> None:
    manifest_info = _required_mapping(evidence.get("manifest"), "manifest")
    row_info = _required_mapping(evidence.get("row"), "row")
    row_id = str(row_info.get("row_id", "")).strip()
    manifest = _read_json_mapping(Path(str(manifest_info["path"])), "speed-row manifest")
    row = _manifest_row(manifest, row_id=row_id)
    if _json_sha256(row) != row_info.get("sha256"):
        raise CompactCoachSpeedRowEvidenceError("manifest row sha256 mismatch")
    if _json_sha256(row.get("command", [])) != row_info.get("command_sha256"):
        raise CompactCoachSpeedRowEvidenceError("manifest row command sha256 mismatch")
    for key in (
        "profile_only",
        "calls_train_muzero",
        "touches_live_runs",
        "row_purpose",
        "speed_currency",
        "promotion_claim",
        "candidate_checkpoint_id",
        "route",
    ):
        if row_info.get(key) != row.get(key):
            raise CompactCoachSpeedRowEvidenceError(f"manifest row {key} mismatch")
    if str(row.get("candidate_checkpoint_id", "")) != str(
        evidence.get("candidate_checkpoint_id", "")
    ):
        raise CompactCoachSpeedRowEvidenceError("manifest row candidate_checkpoint_id mismatch")
    if str(row.get("route", "")) != str(evidence.get("route", "")):
        raise CompactCoachSpeedRowEvidenceError("manifest row route mismatch")
    if row.get("profile_only") is not False:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row row must be profile_only=false")
    expected_calls_train_muzero = (
        str(evidence.get("route")) == COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE
    )
    if row.get("calls_train_muzero") is not expected_calls_train_muzero:
        raise CompactCoachSpeedRowEvidenceError(
            "Coach speed-row row calls_train_muzero does not match route"
        )
    if row.get("touches_live_runs") is not False:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row row must not touch live runs")
    if row.get("promotion_claim") is not False:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row row must not claim promotion")
    if row.get("row_purpose") != "coach_speed_row":
        raise CompactCoachSpeedRowEvidenceError(
            "Coach speed-row row_purpose must be coach_speed_row"
        )
    _validate_nonclaims(_nonclaims_from_loaded(row), "manifest row non_claims")


def _validate_result_section(evidence: Mapping[str, Any]) -> None:
    result_info = _required_mapping(evidence.get("result"), "result")
    summary_info = _required_mapping(evidence.get("result_summary"), "result_summary")
    manifest_info = _required_mapping(evidence.get("manifest"), "manifest")
    row_info = _required_mapping(evidence.get("row"), "row")
    _require_sha(result_info.get("sha256"), "result.sha256")
    path = Path(str(result_info.get("path", "")))
    result = _read_json_mapping(path, "speed-row result")
    if _file_sha256(path) != result_info.get("sha256"):
        raise CompactCoachSpeedRowEvidenceError("result sha256 mismatch")
    summary = _result_summary(result)
    if _json_sha256(summary) != result_info.get("summary_sha256"):
        raise CompactCoachSpeedRowEvidenceError("result summary sha256 mismatch")
    compact_payload = result.get("compact")
    if isinstance(compact_payload, Mapping):
        if _json_sha256(compact_payload) != result_info.get("compact_payload_sha256"):
            raise CompactCoachSpeedRowEvidenceError("result compact payload sha256 mismatch")
    elif result_info.get("compact_payload_sha256") is not None:
        raise CompactCoachSpeedRowEvidenceError(
            "result compact payload hash present without payload"
        )
    else:
        raise CompactCoachSpeedRowEvidenceError("result compact payload missing")
    producer = _required_mapping(result.get("producer"), "result producer")
    if _json_sha256(producer) != result_info.get("producer_sha256"):
        raise CompactCoachSpeedRowEvidenceError("result producer sha256 mismatch")
    if producer.get("schema_id") != COMPACT_COACH_SPEED_ROW_PRODUCER_SCHEMA_ID:
        raise CompactCoachSpeedRowEvidenceError("result producer schema mismatch")
    if not str(producer.get("producer_id", "")).strip():
        raise CompactCoachSpeedRowEvidenceError("result producer_id missing")
    if not str(producer.get("run_id", "")).strip():
        raise CompactCoachSpeedRowEvidenceError("result producer run_id missing")
    if not str(producer.get("produced_by", "")).strip():
        raise CompactCoachSpeedRowEvidenceError("result produced_by missing")
    if result.get("ok") is not True:
        raise CompactCoachSpeedRowEvidenceError("result ok must be true")
    if result_info.get("ok") != result.get("ok"):
        raise CompactCoachSpeedRowEvidenceError("result ok mismatch")
    if result.get("problem") is not None:
        raise CompactCoachSpeedRowEvidenceError("result problem must be null")
    if result_info.get("problem") != result.get("problem"):
        raise CompactCoachSpeedRowEvidenceError("result problem mismatch")
    if result.get("returncode") not in (0, None):
        raise CompactCoachSpeedRowEvidenceError("result returncode must be zero or null")
    if result_info.get("returncode") != result.get("returncode"):
        raise CompactCoachSpeedRowEvidenceError("result returncode mismatch")
    if not str(result.get("run_invocation_id", "")).strip():
        raise CompactCoachSpeedRowEvidenceError("result run_invocation_id missing")
    if result_info.get("run_invocation_id") != result.get("run_invocation_id"):
        raise CompactCoachSpeedRowEvidenceError("result run_invocation_id mismatch")
    if result.get("schema_id") != COMPACT_COACH_SPEED_ROW_RESULT_SCHEMA_ID:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row result schema mismatch")
    if result_info.get("schema_id") != COMPACT_COACH_SPEED_ROW_RESULT_SCHEMA_ID:
        raise CompactCoachSpeedRowEvidenceError("recorded Coach speed-row result schema mismatch")
    for key in ("schema_id", "status", "candidate_checkpoint_id"):
        if result_info.get(key) != result.get(key):
            raise CompactCoachSpeedRowEvidenceError(f"result {key} mismatch")
    if str(result.get("candidate_checkpoint_id", "")) != str(
        evidence.get("candidate_checkpoint_id", "")
    ):
        raise CompactCoachSpeedRowEvidenceError("result candidate_checkpoint_id mismatch")
    result_row = _required_mapping(result.get("row"), "result row")
    manifest = _read_json_mapping(Path(str(manifest_info["path"])), "speed-row manifest")
    manifest_row = _manifest_row(manifest, row_id=str(row_info.get("row_id", "")))
    if _json_sha256(result_row) != result_info.get("row_sha256"):
        raise CompactCoachSpeedRowEvidenceError("result row sha256 mismatch")
    if _json_sha256(result_row) != _json_sha256(manifest_row):
        raise CompactCoachSpeedRowEvidenceError("result row snapshot does not match manifest row")
    row_id = str(result.get("row_id") or summary.get("row_id") or "")
    if str(result_info.get("row_id", "")) != row_id:
        raise CompactCoachSpeedRowEvidenceError("result row_id mismatch")
    if row_id != str(_required_mapping(evidence.get("row"), "row").get("row_id")):
        raise CompactCoachSpeedRowEvidenceError("manifest/result row_id mismatch")
    if result.get("status") != "complete":
        raise CompactCoachSpeedRowEvidenceError("result status must be complete")
    for key in (
        "profile_only",
        "calls_train_muzero",
        "touches_live_runs",
        "status",
        "ok",
        "row_id",
        "candidate_checkpoint_id",
        "route",
        "row_purpose",
        "promotion_claim",
        "speed_currency",
        "search_service_kind",
        "search_service_impl",
        "compact_torch_initial_inference_mode",
        "compact_torch_observation_memory_format",
        "compact_torch_model_memory_format",
        "compact_torch_defer_one_simulation_replay_payload_requested",
        "compact_torch_memory_format_applies_to_search_service",
        "hybrid_persistent_compact_render_state_buffer",
        "hybrid_borrow_single_actor_render_state",
        "render_state_handoff_mode",
        "render_state_copy_steps",
        "render_state_borrowed_steps",
        "render_state_row_overlay_steps",
        "render_state_row_overlay_rows",
        "render_state_row_overlay_bytes",
        "actor_count",
        "batch_size",
        "steps",
        "warmup_steps",
        "death_mode",
        "terminal_row_count",
        "death_row_count",
        "terminated_row_count",
        "truncated_row_count",
        "terminal_final_observation_row_count",
        "terminal_final_observation_before_autoreset_verified",
        "terminal_sample_row_count",
        "terminal_unroll_value_target_mode",
        "terminal_unroll_value_target_row_count",
        "resident_observation_host_fallback_count",
        "normal_death_terminal_contract_promotion_gate_satisfied",
        "learner_num_unroll_steps",
        "compact_owned_loop_fused_learner_batch",
        "compact_owned_training_loop_owner",
        "compact_owner_search_action_only_result",
        "compact_owner_search_owner_materializes_replay",
        "compact_owner_search_parent_slab_commits_replay",
        "compact_owner_search_learner_update_count",
        "compact_owner_search_owner_train_request_count",
        "compact_owner_search_owner_expected_train_request_count",
        "compact_owner_search_owner_learner_update_count",
        "compact_owner_search_worker_learner_train_sec",
        "compact_owner_search_owner_train_wall_sec",
        "speed_row_total_owner_search_worker_learner_train_sec",
        "compact_rollout_slab_sample_gate_calls",
        "compact_rollout_slab_learner_gate_calls",
        "compact_muzero_learner_batch_unroll2_specialized_builder",
        "compact_muzero_learner_batch_learner_ready_unroll2_cache",
        "compact_muzero_learner_batch_tensor_native_replay",
        "compact_rollout_slab_sample_gate_sec",
        "compact_rollout_slab_learner_gate_sec",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only",
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch",
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch",
        "compact_rollout_slab_learner_gate_prebuilt_batch_used",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_impl",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_direct_build_used",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason",
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped",
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested",
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used",
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_missing_record_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_rows",
        "compact_rollout_slab_sample_gate_fixed_soa_requested",
        "compact_rollout_slab_sample_gate_fixed_soa_used",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_schema_id",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_requested",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_used",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_handle_id",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_snapshot_version",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_request_checksum",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_sample_row_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_target_row_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_create_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_resolve_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_inline_resolve_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_reason",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_pending_handle_count",
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_count",
        "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count",
        "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count",
        "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count",
        "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count",
        "compact_rollout_slab_sample_gate_fixed_soa_record_count",
        "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count",
        "compact_rollout_slab_sample_gate_fixed_soa_table_row_count",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count",
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_count",
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason",
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec",
        "compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec",
        "compact_rollout_slab_sample_gate_fixed_soa_total_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path",
        "compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled",
        "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep",
        "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used",
        "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count",
        "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count",
        "compact_muzero_learner_prebuilt_batch_used",
        *_REPEATABILITY_RESULT_SUMMARY_FIELDS,
    ):
        if summary_info.get(key) != summary.get(key):
            raise CompactCoachSpeedRowEvidenceError(f"result summary {key} mismatch")
    if summary.get("profile_only") is not False:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row result must be profile_only=false")
    expected_calls_train_muzero = (
        str(evidence.get("route")) == COMPACT_COACH_ROUTE_STOCK_TRAIN_MUZERO_BRIDGE
    )
    if summary.get("calls_train_muzero") is not expected_calls_train_muzero:
        raise CompactCoachSpeedRowEvidenceError(
            "Coach speed-row result calls_train_muzero does not match route"
        )
    if summary.get("touches_live_runs") is not False:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row result must not touch live runs")
    if summary.get("promotion_claim") is not False:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row result must not claim promotion")
    if str(summary.get("candidate_checkpoint_id", "")) != str(
        evidence.get("candidate_checkpoint_id", "")
    ):
        raise CompactCoachSpeedRowEvidenceError("result summary candidate_checkpoint_id mismatch")
    if str(summary.get("route", "")) != str(evidence.get("route", "")):
        raise CompactCoachSpeedRowEvidenceError("result summary route mismatch")
    if summary.get("row_purpose") != "coach_speed_row":
        raise CompactCoachSpeedRowEvidenceError(
            "result summary row_purpose must be coach_speed_row"
        )
    if summary.get("status") != "complete":
        raise CompactCoachSpeedRowEvidenceError("result summary status must be complete")
    if summary.get("ok") is not True:
        raise CompactCoachSpeedRowEvidenceError("result summary ok must be true")
    if compact_payload.get("ok") is not True:
        raise CompactCoachSpeedRowEvidenceError("result compact payload ok must be true")
    if str(compact_payload.get("candidate_checkpoint_id", "")) != str(
        evidence.get("candidate_checkpoint_id", "")
    ):
        raise CompactCoachSpeedRowEvidenceError(
            "result compact payload candidate_checkpoint_id mismatch"
        )
    if str(compact_payload.get("route", "")) != str(evidence.get("route", "")):
        raise CompactCoachSpeedRowEvidenceError("result compact payload route mismatch")
    if compact_payload.get("profile_only") is not False:
        raise CompactCoachSpeedRowEvidenceError("result compact payload must be profile_only=false")
    if compact_payload.get("calls_train_muzero") is not expected_calls_train_muzero:
        raise CompactCoachSpeedRowEvidenceError(
            "result compact payload calls_train_muzero does not match route"
        )
    if compact_payload.get("touches_live_runs") is not False:
        raise CompactCoachSpeedRowEvidenceError("result compact payload must not touch live runs")
    if compact_payload.get("real_compact_owned_training_work") is not True:
        raise CompactCoachSpeedRowEvidenceError(
            "result compact payload must prove real compact-owned training work"
        )
    _validate_nonclaims(_nonclaims_from_loaded(result), "loaded result non_claims")
    _validate_nonclaims(_nonclaims_from_loaded(summary), "result summary non_claims")
    _validate_nonclaims(_nonclaims_from_loaded(compact_payload), "compact payload non_claims")


def _validate_search_config_section(evidence: Mapping[str, Any]) -> None:
    raw_config = evidence.get("search_config")
    if raw_config is None:
        return
    config = _required_mapping(raw_config, "search_config")
    manifest_info = _required_mapping(evidence.get("manifest"), "manifest")
    row_info = _required_mapping(evidence.get("row"), "row")
    result_info = _required_mapping(evidence.get("result"), "result")
    manifest = _read_json_mapping(Path(str(manifest_info["path"])), "speed-row manifest")
    row = _manifest_row(manifest, row_id=str(row_info.get("row_id", "")))
    result = _read_json_mapping(Path(str(result_info["path"])), "speed-row result")
    summary = _result_summary(result)
    compact_payload = _required_mapping(result.get("compact"), "result compact payload")
    expected = _speed_row_search_config(
        manifest=manifest,
        manifest_row=row,
        summary=summary,
        compact_payload=compact_payload,
    )
    if _json_sha256(config) != _json_sha256(expected):
        raise CompactCoachSpeedRowEvidenceError("search_config mismatch")
    model_format_raw = config.get("compact_torch_model_memory_format")
    model_format = "" if model_format_raw is None else str(model_format_raw).strip()
    if model_format and model_format != "contiguous":
        raise CompactCoachSpeedRowEvidenceError(
            "compact Torch model memory format must remain contiguous"
        )


def _validate_actor_handoff_config_section(evidence: Mapping[str, Any]) -> None:
    raw_config = evidence.get("actor_handoff_config")
    if raw_config is None:
        return
    config = _required_mapping(raw_config, "actor_handoff_config")
    manifest_info = _required_mapping(evidence.get("manifest"), "manifest")
    row_info = _required_mapping(evidence.get("row"), "row")
    result_info = _required_mapping(evidence.get("result"), "result")
    manifest = _read_json_mapping(Path(str(manifest_info["path"])), "speed-row manifest")
    row = _manifest_row(manifest, row_id=str(row_info.get("row_id", "")))
    result = _read_json_mapping(Path(str(result_info["path"])), "speed-row result")
    summary = _result_summary(result)
    compact_payload = _required_mapping(result.get("compact"), "result compact payload")
    expected = _speed_row_actor_handoff_config(
        manifest=manifest,
        manifest_row=row,
        summary=summary,
        compact_payload=compact_payload,
    )
    if _json_sha256(config) != _json_sha256(expected):
        raise CompactCoachSpeedRowEvidenceError("actor_handoff_config mismatch")
    persistent_requested = config.get("hybrid_persistent_compact_render_state_buffer")
    if not isinstance(persistent_requested, bool):
        raise CompactCoachSpeedRowEvidenceError(
            "actor_handoff_config persistent render-state request must be boolean"
        )
    borrow_requested = config.get("hybrid_borrow_single_actor_render_state")
    if not isinstance(borrow_requested, bool):
        raise CompactCoachSpeedRowEvidenceError(
            "actor_handoff_config borrowed render-state request must be boolean"
        )
    if persistent_requested and borrow_requested:
        raise CompactCoachSpeedRowEvidenceError(
            "actor_handoff_config cannot request both persistent and borrowed render-state"
        )
    mode = str(config.get("render_state_handoff_mode") or "").strip()
    if borrow_requested:
        expected_mode = "borrow_single_actor_env_state"
    elif persistent_requested:
        expected_mode = "persistent_compact_render_state_buffer"
    else:
        expected_mode = "copy_actor_state_to_parent_buffers"
    if mode != expected_mode:
        raise CompactCoachSpeedRowEvidenceError("actor_handoff_config handoff mode mismatch")
    borrowed_steps = _actor_handoff_nonnegative_int(config, "render_state_borrowed_steps")
    if borrow_requested and borrowed_steps <= 0:
        raise CompactCoachSpeedRowEvidenceError(
            "borrowed render-state handoff must report borrowed steps"
        )
    if not borrow_requested and borrowed_steps != 0:
        raise CompactCoachSpeedRowEvidenceError(
            "actor_handoff_config borrowed render-state steps require borrowed mode"
        )
    copy_steps = _actor_handoff_nonnegative_int(config, "render_state_copy_steps")
    row_overlay_steps = _actor_handoff_nonnegative_int(
        config,
        "render_state_row_overlay_steps",
    )
    row_overlay_rows = _actor_handoff_nonnegative_int(
        config,
        "render_state_row_overlay_rows",
    )
    _actor_handoff_nonnegative_int(config, "render_state_row_overlay_bytes")
    if not borrow_requested and (row_overlay_steps != 0 or row_overlay_rows != 0):
        raise CompactCoachSpeedRowEvidenceError(
            "render-state row overlays require borrowed render-state mode"
        )
    if persistent_requested and copy_steps != 0:
        raise CompactCoachSpeedRowEvidenceError(
            "direct render-state handoff must not copy actor state rows"
        )
    if borrow_requested and copy_steps != 0:
        surface = (
            _required_mapping(evidence.get("operational_surface"), "operational_surface")
            if isinstance(evidence.get("operational_surface"), Mapping)
            else {}
        )
        if not _borrowed_copy_steps_are_terminal_snapshots(surface):
            raise CompactCoachSpeedRowEvidenceError(
                "borrowed render-state copies require proven terminal snapshots"
            )
    if not persistent_requested and not borrow_requested and copy_steps <= 0:
        raise CompactCoachSpeedRowEvidenceError(
            "copy render-state handoff must report positive copy steps"
        )


def _validate_operational_surface_section(evidence: Mapping[str, Any]) -> None:
    config = _required_mapping(evidence.get("operational_surface"), "operational_surface")
    manifest_info = _required_mapping(evidence.get("manifest"), "manifest")
    row_info = _required_mapping(evidence.get("row"), "row")
    result_info = _required_mapping(evidence.get("result"), "result")
    manifest = _read_json_mapping(Path(str(manifest_info["path"])), "speed-row manifest")
    row = _manifest_row(manifest, row_id=str(row_info.get("row_id", "")))
    result = _read_json_mapping(Path(str(result_info["path"])), "speed-row result")
    summary = _result_summary(result)
    compact_payload = _required_mapping(result.get("compact"), "result compact payload")
    expected = _speed_row_operational_surface(
        manifest=manifest,
        manifest_row=row,
        summary=summary,
        compact_payload=compact_payload,
    )
    if _json_sha256(config) != _json_sha256(expected):
        raise CompactCoachSpeedRowEvidenceError("operational_surface mismatch")

    actor_count = _surface_positive_int(config, "actor_count")
    _surface_positive_int(config, "batch_size")
    _surface_positive_int(config, "steps")
    _surface_nonnegative_int(config, "warmup_steps")

    actor_config = _required_mapping(evidence.get("actor_handoff_config"), "actor_handoff_config")
    borrow_requested = actor_config.get("hybrid_borrow_single_actor_render_state") is True
    if borrow_requested and actor_count != 1:
        raise CompactCoachSpeedRowEvidenceError(
            "borrowed render-state speed evidence requires actor_count=1"
        )

    mode = str(config.get("death_mode") or "").strip()
    if mode not in {"profile_no_death", "normal"}:
        raise CompactCoachSpeedRowEvidenceError(
            "operational_surface death_mode must be profile_no_death or normal"
        )
    owner = str(config.get("compact_owned_training_loop_owner") or "")
    trainer_config_death_mode = str(config.get("compact_owned_trainer_config_death_mode") or "")
    if (
        mode == "normal"
        and owner == "lean_compact_trainer_step"
        and trainer_config_death_mode != "normal"
    ):
        raise CompactCoachSpeedRowEvidenceError(
            "lean normal-death speed evidence requires trainer config death_mode=normal"
        )
    terminal_rows = _surface_nonnegative_int(config, "terminal_row_count")
    death_rows = _surface_nonnegative_int(config, "death_row_count")
    terminated_rows = _surface_nonnegative_int(config, "terminated_row_count")
    truncated_rows = _surface_nonnegative_int(config, "truncated_row_count")
    final_observation_rows = _surface_nonnegative_int(
        config,
        "terminal_final_observation_row_count",
    )
    terminal_sample_rows = _surface_nonnegative_int(config, "terminal_sample_row_count")
    terminal_target_rows = _surface_nonnegative_int(
        config,
        "terminal_unroll_value_target_row_count",
    )
    _validate_sample_learner_fusion_surface(config)

    if mode == "profile_no_death":
        if any(
            value != 0
            for value in (
                terminal_rows,
                death_rows,
                terminated_rows,
                truncated_rows,
                final_observation_rows,
                terminal_sample_rows,
                terminal_target_rows,
            )
        ):
            raise CompactCoachSpeedRowEvidenceError(
                "profile_no_death speed evidence must not contain terminal/death rows"
            )
        if config.get("normal_death_terminal_contract_promotion_gate_satisfied") is True:
            raise CompactCoachSpeedRowEvidenceError(
                "profile_no_death speed evidence cannot claim normal-death contract"
            )
        if borrow_requested:
            copy_steps = _actor_handoff_nonnegative_int(
                actor_config,
                "render_state_copy_steps",
            )
            if copy_steps != 0:
                raise CompactCoachSpeedRowEvidenceError(
                    "profile_no_death borrowed speed evidence must not copy render state"
                )
            row_overlay_rows = _actor_handoff_nonnegative_int(
                actor_config,
                "render_state_row_overlay_rows",
            )
            if row_overlay_rows != 0:
                raise CompactCoachSpeedRowEvidenceError(
                    "profile_no_death borrowed speed evidence must not use row overlays"
                )
        return

    if terminal_rows <= 0 or death_rows <= 0 or terminated_rows <= 0:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence requires terminal, death, and terminated rows"
        )
    if truncated_rows != 0:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence must have truncated_row_count=0"
        )
    if final_observation_rows <= 0:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence requires terminal final observations"
        )
    if config.get("terminal_final_observation_before_autoreset_verified") is not True:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence must verify final observations before autoreset"
        )
    if terminal_sample_rows <= 0 or terminal_target_rows <= 0:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence requires terminal sample and target rows"
        )
    if (
        str(config.get("terminal_unroll_value_target_mode") or "")
        != "stock_terminal_no_bootstrap_return_discount_1.0"
    ):
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence must use stock no-bootstrap terminal targets"
        )
    if config.get("normal_death_terminal_contract_promotion_gate_satisfied") is not True:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence requires the normal-death contract gate"
        )
    if borrow_requested:
        row_overlay_rows = _actor_handoff_nonnegative_int(
            actor_config,
            "render_state_row_overlay_rows",
        )
        if row_overlay_rows <= 0:
            raise CompactCoachSpeedRowEvidenceError(
                "borrowed normal death speed evidence requires render-state row overlays"
            )
    fallback_count = _surface_float(config, "resident_observation_host_fallback_count")
    if fallback_count != 0.0:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence must not use resident host fallback"
        )


def _owner_search_owner_materialized_learner_batch(
    config: Mapping[str, Any],
) -> bool:
    if str(config.get("compact_owned_training_loop_owner") or "") != "owner_search_worker":
        return False
    if config.get("compact_owner_search_action_only_result") is not True:
        return False
    if config.get("compact_owner_search_owner_materializes_replay") is not True:
        return False
    if config.get("compact_owner_search_parent_slab_commits_replay") is not False:
        return False
    if int(config.get("compact_rollout_slab_sample_gate_calls") or 0) != 0:
        return False
    if int(config.get("compact_rollout_slab_learner_gate_calls") or 0) != 0:
        return False
    count_keys = (
        "compact_owner_search_owner_train_request_count",
        "compact_owner_search_owner_expected_train_request_count",
        "compact_owner_search_learner_update_count",
        "compact_owner_search_owner_learner_update_count",
    )
    if not any(key in config and config.get(key) is not None for key in count_keys):
        return True
    train_requests = int(config.get("compact_owner_search_owner_train_request_count") or 0)
    expected_train_requests = int(
        config.get("compact_owner_search_owner_expected_train_request_count") or 0
    )
    learner_updates = int(config.get("compact_owner_search_learner_update_count") or 0)
    owner_learner_updates = int(
        config.get("compact_owner_search_owner_learner_update_count") or 0
    )
    return (
        train_requests > 0
        and expected_train_requests == train_requests
        and max(learner_updates, owner_learner_updates) > 0
    )


def _validate_sample_learner_fusion_surface(config: Mapping[str, Any]) -> None:
    if "compact_owned_loop_fused_learner_batch" not in config:
        return
    fused = config.get("compact_owned_loop_fused_learner_batch") is True
    if config.get("compact_owned_loop_fused_learner_batch") not in {False, True}:
        raise CompactCoachSpeedRowEvidenceError(
            "compact_owned_loop_fused_learner_batch must be boolean"
        )
    unroll2_specialized = bool(
        config.get("compact_muzero_learner_batch_unroll2_specialized_builder", False)
    )
    learner_ready_unroll2_cache = bool(
        config.get("compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
    )
    tensor_native_replay = bool(
        config.get("compact_muzero_learner_batch_tensor_native_replay", False)
    )
    if config.get("compact_muzero_learner_batch_unroll2_specialized_builder", False) not in {
        False,
        True,
    }:
        raise CompactCoachSpeedRowEvidenceError(
            "compact_muzero_learner_batch_unroll2_specialized_builder must be boolean"
        )
    if config.get("compact_muzero_learner_batch_learner_ready_unroll2_cache", False) not in {
        False,
        True,
    }:
        raise CompactCoachSpeedRowEvidenceError(
            "compact_muzero_learner_batch_learner_ready_unroll2_cache must be boolean"
        )
    if config.get("compact_muzero_learner_batch_tensor_native_replay", False) not in {
        False,
        True,
    }:
        raise CompactCoachSpeedRowEvidenceError(
            "compact_muzero_learner_batch_tensor_native_replay must be boolean"
        )
    if config.get("learner_num_unroll_steps") is not None:
        learner_num_unroll_steps = _surface_positive_int(
            config,
            "learner_num_unroll_steps",
        )
    else:
        learner_num_unroll_steps = 0
    sample_sec = _surface_float(config, "compact_rollout_slab_sample_gate_sec")
    learner_sec = _surface_float(config, "compact_rollout_slab_learner_gate_sec")
    proof_keys = (
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only",
        "compact_rollout_slab_learner_gate_prebuilt_batch_used",
    )
    unroll2_proof_keys = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used",
    )
    learner_ready_proof_keys = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used",
    )
    tensor_native_proof_keys = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used",
    )
    resident_proof_keys = (
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch",
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch",
    )
    host_provider_proof_key = "compact_rollout_slab_sample_gate_host_provider_learner_batch"
    learner_ready_requested_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested"
    )
    learner_ready_used_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used"
    )
    learner_ready_impl = str(
        config.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl",
            "",
        )
    )
    learner_batch_builder_path = str(
        config.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path",
            "",
        )
    )
    tensor_native_impl = str(
        config.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
            "",
        )
    )
    tensor_native_table_source = str(
        config.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
            "",
        )
    )
    fixed_soa_tensor_native = (
        tensor_native_impl == "fixed_soa_direct_gather_v1"
        and tensor_native_table_source == "fixed_soa_columns_v1"
    )
    owner_search_owner_batch = _owner_search_owner_materialized_learner_batch(config)
    if not fused:
        has_fusion_proof = any(
            config.get(key) is True
            for key in (
                *proof_keys,
                *unroll2_proof_keys,
                *learner_ready_proof_keys,
                *tensor_native_proof_keys,
                *resident_proof_keys,
                host_provider_proof_key,
            )
        )
        if has_fusion_proof:
            if owner_search_owner_batch:
                return
            raise CompactCoachSpeedRowEvidenceError(
                "fused learner-batch proof requires fused learner-batch config"
            )
        if unroll2_specialized:
            raise CompactCoachSpeedRowEvidenceError(
                "unroll2 specialized builder requires fused learner-batch config"
            )
        if learner_ready_unroll2_cache:
            raise CompactCoachSpeedRowEvidenceError(
                "learner-ready unroll2 cache requires fused learner-batch config"
            )
        if tensor_native_replay:
            raise CompactCoachSpeedRowEvidenceError(
                "tensor-native replay requires fused learner-batch config"
            )
        return
    if learner_num_unroll_steps <= 1:
        raise CompactCoachSpeedRowEvidenceError(
            "fused learner-batch speed evidence requires N-step learner unroll"
        )
    if owner_search_owner_batch and learner_sec <= 0.0:
        learner_sec = max(
            _surface_float(config, "speed_row_total_owner_search_worker_learner_train_sec"),
            _surface_float(config, "compact_owner_search_worker_learner_train_sec"),
            _surface_float(config, "compact_owner_search_owner_train_wall_sec"),
        )
    if sample_sec <= 0.0 or learner_sec <= 0.0:
        raise CompactCoachSpeedRowEvidenceError(
            "fused learner-batch speed evidence requires sample and learner timings"
        )
    parent_proof_keys = (
        tuple(key for key in proof_keys if key != "compact_rollout_slab_learner_gate_prebuilt_batch_used")
        if owner_search_owner_batch
        else proof_keys
    )
    missing = [key for key in parent_proof_keys if config.get(key) is not True]
    resident_fused = all(config.get(key) is True for key in resident_proof_keys)
    host_provider_fused = config.get(host_provider_proof_key) is True
    tensor_native_fused = (
        config.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
        )
        is True
    )
    if not resident_fused and not host_provider_fused and not tensor_native_fused:
        missing.append(
            "resident_grouped_device_learner_batch or host_provider_learner_batch "
            "or tensor_native_replay"
        )
    if unroll2_specialized and not learner_ready_unroll2_cache:
        if learner_num_unroll_steps != 2:
            missing.append("learner_num_unroll_steps == 2")
        missing.extend(key for key in unroll2_proof_keys if config.get(key) is not True)
        if _surface_positive_int(
            config,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count",
        ) <= 0:
            missing.append("unroll2_specialized_builder_call_count")
        fallback_count = int(
            config.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count",
                0,
            )
            or 0
        )
        if fallback_count != 0:
            missing.append("unroll2_specialized_builder_fallback_count")
        fallback_reason = str(
            config.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason",
                "",
            )
        )
        if fallback_reason != "none":
            missing.append("unroll2_specialized_builder_fallback_reason")
        impl = str(
            config.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl",
                "",
            )
        )
        if impl != "unroll2_specialized_v1":
            missing.append("unroll2_specialized_builder_impl")
        if (
            str(
                config.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path",
                    "",
                )
            )
            != "unroll2_specialized"
        ):
            missing.append("unroll2_specialized_builder_path")
    if learner_ready_unroll2_cache or tensor_native_replay:
        if learner_num_unroll_steps != 2:
            missing.append("learner_num_unroll_steps == 2")
        if config.get(learner_ready_requested_key) is not True:
            missing.append(learner_ready_requested_key)
        fallback_count = int(
            config.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count",
                0,
            )
            or 0
        )
        if fallback_count != 0:
            missing.append("learner_ready_unroll2_cache_fallback_count")
        fallback_reason = str(
            config.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason",
                "",
            )
        )
        if fallback_reason != "none":
            missing.append("learner_ready_unroll2_cache_fallback_reason")
        learner_ready_count_keys = (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count",
                "learner_ready_unroll2_cache_available_group_count",
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count",
                "learner_ready_unroll2_cache_eligible_count",
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count",
                "learner_ready_unroll2_cache_call_count",
            ),
        )
        if fixed_soa_tensor_native:
            if config.get(learner_ready_used_key) is not False:
                missing.append("learner_ready_unroll2_cache_used")
            for count_key, label in learner_ready_count_keys:
                if _surface_nonnegative_int(config, count_key) != 0:
                    missing.append(label)
            if learner_ready_impl != "fixed_soa_columns_v1":
                missing.append("learner_ready_unroll2_cache_impl")
            if learner_batch_builder_path != "fixed_soa_direct_gather":
                missing.append("learner_ready_unroll2_cache_builder_path")
        else:
            if config.get(learner_ready_used_key) is not True:
                missing.append("learner_ready_unroll2_cache_used")
            for count_key, label in learner_ready_count_keys:
                if _surface_positive_int(config, count_key) <= 0:
                    missing.append(label)
            if learner_ready_impl != "learner_ready_unroll2_cache_v1":
                missing.append("learner_ready_unroll2_cache_impl")
            if learner_batch_builder_path != "learner_ready_unroll2_cache":
                missing.append("learner_ready_unroll2_cache_builder_path")
    if tensor_native_replay:
        if not learner_ready_unroll2_cache:
            missing.append("compact_muzero_learner_batch_learner_ready_unroll2_cache")
        if learner_num_unroll_steps != 2:
            missing.append("learner_num_unroll_steps == 2")
        missing.extend(key for key in tensor_native_proof_keys if config.get(key) is not True)
        if _surface_positive_int(
            config,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count",
        ) <= 0:
            missing.append("tensor_native_replay_call_count")
        fallback_count = int(
            config.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
                0,
            )
            or 0
        )
        if fallback_count != 0:
            missing.append("tensor_native_replay_fallback_count")
        fallback_reason = str(
            config.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason",
                "",
            )
        )
        if fallback_reason != "none":
            missing.append("tensor_native_replay_fallback_reason")
        if tensor_native_impl not in {
            "maintained_unroll2_table_gather_v1",
            "selected_direct_record_table_gather_v1",
            "selected_maintained_record_table_gather_v1",
            "fixed_soa_direct_gather_v1",
        }:
            missing.append("tensor_native_replay_impl")
        if tensor_native_table_source not in {
            "maintained_record_table_v1",
            "selected_direct_record_table_v1",
            "selected_maintained_record_table_v1",
            "fixed_soa_columns_v1",
        }:
            missing.append("tensor_native_replay_table_source")
        if fixed_soa_tensor_native:
            if config.get("compact_rollout_slab_sample_gate_fixed_soa_used") is not True:
                missing.append("fixed_soa_used")
            fixed_soa_zero_int_fields = (
                "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count",
                "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count",
                "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count",
                "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count",
                "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count",
                "compact_rollout_slab_sample_gate_fixed_soa_fallback_count",
                "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count",
            )
            for field in fixed_soa_zero_int_fields:
                if _surface_nonnegative_int(config, field) != 0:
                    missing.append(field)
            fixed_soa_fallback_reason = str(
                config.get(
                    "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason",
                    "",
                )
            )
            if fixed_soa_fallback_reason != "none":
                missing.append("compact_rollout_slab_sample_gate_fixed_soa_fallback_reason")
            fixed_soa_handle_requested = bool(
                config.get(
                    (
                        "compact_rollout_slab_sample_gate_fixed_soa_"
                        "learner_batch_handle_ring_requested"
                    ),
                    False,
                )
            )
            fixed_soa_handle_sample_rows = _surface_nonnegative_int(
                config,
                (
                    "compact_rollout_slab_sample_gate_fixed_soa_"
                    "learner_batch_handle_ring_sample_row_count"
                ),
            )
            if fixed_soa_handle_requested and fixed_soa_handle_sample_rows > 0:
                if (
                    config.get(
                        (
                            "compact_rollout_slab_sample_gate_fixed_soa_"
                            "learner_batch_handle_ring_used"
                        )
                    )
                    is not True
                ):
                    missing.append("fixed_soa_learner_batch_handle_ring_used")
                fixed_soa_handle_one_fields = (
                    (
                        "compact_rollout_slab_sample_gate_fixed_soa_"
                        "learner_batch_handle_ring_create_count"
                    ),
                    (
                        "compact_rollout_slab_sample_gate_fixed_soa_"
                        "learner_batch_handle_ring_resolve_count"
                    ),
                    (
                        "compact_rollout_slab_sample_gate_fixed_soa_"
                        "learner_batch_handle_ring_inline_resolve_count"
                    ),
                )
                for field in fixed_soa_handle_one_fields:
                    if _surface_nonnegative_int(config, field) != 1:
                        missing.append(field)
                fixed_soa_handle_zero_fields = (
                    (
                        "compact_rollout_slab_sample_gate_fixed_soa_"
                        "learner_batch_handle_ring_fallback_count"
                    ),
                    (
                        "compact_rollout_slab_sample_gate_fixed_soa_"
                        "learner_batch_handle_ring_pending_handle_count"
                    ),
                )
                for field in fixed_soa_handle_zero_fields:
                    if _surface_nonnegative_int(config, field) != 0:
                        missing.append(field)
                fixed_soa_handle_fallback_reason = str(
                    config.get(
                        (
                            "compact_rollout_slab_sample_gate_fixed_soa_"
                            "learner_batch_handle_ring_fallback_reason"
                        ),
                        "",
                    )
                )
                if fixed_soa_handle_fallback_reason != "none":
                    missing.append("fixed_soa_learner_batch_handle_ring_fallback_reason")
            if (
                _surface_float(
                    config,
                    (
                        "compact_rollout_slab_sample_gate_learner_batch_builder_"
                        "tensor_native_replay_table_concat_sec"
                    ),
                )
                != 0.0
            ):
                missing.append("tensor_native_replay_table_concat_sec")
        if _surface_positive_int(
            config,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_reused_record_count"
            ),
        ) <= 0:
            missing.append("tensor_native_replay_table_reused_record_count")
        missing_table_count = int(
            config.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_missing_record_count"
                ),
                -1,
            )
            or 0
        )
        if missing_table_count != 0:
            missing.append("tensor_native_replay_table_missing_record_count")
        if _surface_positive_int(
            config,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows",
        ) <= 0:
            missing.append("tensor_native_replay_table_rows")
        direct_prebuilt_requested = (
            config.get(
                "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested"
            )
            is True
        )
        if direct_prebuilt_requested:
            if (
                config.get(
                    "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used"
                )
                is not True
            ):
                missing.append("tensor_native_direct_prebuilt_path_used")
            direct_fallback_count = int(
                config.get(
                    (
                        "compact_rollout_slab_sample_gate_tensor_native_"
                        "direct_prebuilt_fallback_count"
                    ),
                    0,
                )
                or 0
            )
            if direct_fallback_count != 0:
                missing.append("tensor_native_direct_prebuilt_fallback_count")
            direct_fallback_reason = str(
                config.get(
                    (
                        "compact_rollout_slab_sample_gate_tensor_native_"
                        "direct_prebuilt_fallback_reason"
                    ),
                    "",
                )
            )
            if direct_fallback_reason != "none":
                missing.append("tensor_native_direct_prebuilt_fallback_reason")
            direct_group_object_count = int(
                config.get(
                    (
                        "compact_rollout_slab_sample_gate_tensor_native_"
                        "direct_group_object_count"
                    ),
                    -1,
                )
                or 0
            )
            if direct_group_object_count != 0:
                missing.append("tensor_native_direct_group_object_count")
    if missing:
        raise CompactCoachSpeedRowEvidenceError(
            "fused learner-batch speed evidence missing proof fields: " + ", ".join(missing)
        )


def _speed_row_terminal_death_proof(
    *,
    operational_surface: Mapping[str, Any],
    summary: Mapping[str, Any],
    compact_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    mode = str(operational_surface.get("death_mode") or "")
    if mode != "normal":
        return {
            "death_mode": mode,
            "normal_death_contract_required": False,
            "normal_death_terminal_contract_promotion_gate_satisfied": False,
        }
    contract = None
    if isinstance(compact_payload, Mapping):
        contract = compact_payload.get("normal_death_terminal_contract")
    if not isinstance(contract, Mapping):
        contract = summary.get("normal_death_terminal_contract")
    if not isinstance(contract, Mapping):
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence requires a terminal/death contract"
        )
    try:
        validate_compact_death_terminal_contract_v1(contract)
    except ValueError as exc:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death terminal/death contract is invalid"
        ) from exc
    return {
        "death_mode": "normal",
        "normal_death_contract_required": True,
        "compact_owned_trainer_config_death_mode": _first_present(
            compact_payload.get("compact_owned_trainer_config_death_mode")
            if isinstance(compact_payload, Mapping)
            else None,
            summary.get("compact_owned_trainer_config_death_mode"),
            "",
        ),
        "normal_death_terminal_contract_owner": _first_present(
            contract.get("normal_death_terminal_contract_owner"),
            compact_payload.get("normal_death_terminal_contract_owner")
            if isinstance(compact_payload, Mapping)
            else None,
            summary.get("normal_death_terminal_contract_owner"),
            "none",
        ),
        "normal_death_terminal_contract_source": _first_present(
            compact_payload.get("normal_death_terminal_contract_source")
            if isinstance(compact_payload, Mapping)
            else None,
            summary.get("normal_death_terminal_contract_source"),
            "unknown",
        ),
        "normal_death_terminal_contract_promotion_gate_satisfied": contract.get(
            "compact_death_terminal_contract_promotion_gate_satisfied"
        ),
        "normal_death_terminal_contract_schema_id": contract.get(
            "compact_death_terminal_contract_schema_id"
        ),
        "normal_collision_death_evidence_id": contract.get("normal_collision_death_evidence_id"),
        "normal_collision_death_evidence_refs": list(
            contract.get("normal_collision_death_evidence_refs") or []
        ),
        "death_count_total": contract.get("death_count_total"),
        "terminal_row_count": contract.get("terminal_row_count"),
        "terminated_row_count": contract.get("terminated_row_count"),
        "truncated_row_count": contract.get("truncated_row_count"),
        "death_row_count": contract.get("death_row_count"),
        "normal_collision_death_causes": list(contract.get("normal_collision_death_causes") or []),
        "normal_collision_death_hit_owner_present": contract.get(
            "normal_collision_death_hit_owner_present"
        ),
        "terminal_unroll_value_target_mode": contract.get("terminal_unroll_value_target_mode"),
        "contract_sha256": _json_sha256(contract),
    }


def _validate_terminal_death_proof_section(evidence: Mapping[str, Any]) -> None:
    proof = _required_mapping(evidence.get("terminal_death_proof"), "terminal_death_proof")
    surface = _required_mapping(evidence.get("operational_surface"), "operational_surface")
    mode = str(surface.get("death_mode") or "")
    if proof.get("death_mode") != mode:
        raise CompactCoachSpeedRowEvidenceError("terminal_death_proof death_mode mismatch")
    if mode != "normal":
        if proof.get("normal_death_contract_required") is not False:
            raise CompactCoachSpeedRowEvidenceError(
                "non-normal speed evidence must not require a normal-death contract"
            )
        if proof.get("normal_death_terminal_contract_promotion_gate_satisfied") is True:
            raise CompactCoachSpeedRowEvidenceError(
                "non-normal speed evidence cannot carry normal-death proof"
            )
        return
    if proof.get("normal_death_contract_required") is not True:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death speed evidence requires terminal_death_proof"
        )
    if proof.get("normal_death_terminal_contract_promotion_gate_satisfied") is not True:
        raise CompactCoachSpeedRowEvidenceError(
            "normal death terminal_death_proof must satisfy the contract gate"
        )
    if (
        surface.get("compact_owned_training_loop_owner") == "lean_compact_trainer_step"
        and proof.get("compact_owned_trainer_config_death_mode") != "normal"
    ):
        raise CompactCoachSpeedRowEvidenceError(
            "lean normal-death terminal proof requires trainer config death_mode=normal"
        )
    if (
        surface.get("compact_owned_training_loop_owner") == "lean_compact_trainer_step"
        and proof.get("normal_death_terminal_contract_owner") != "compact_owned_trainer_config"
    ):
        raise CompactCoachSpeedRowEvidenceError(
            "lean normal-death terminal proof requires compact-owned trainer owner"
        )
    for key in (
        "terminal_row_count",
        "terminated_row_count",
        "truncated_row_count",
        "death_row_count",
        "terminal_unroll_value_target_mode",
    ):
        if proof.get(key) != surface.get(key):
            raise CompactCoachSpeedRowEvidenceError(f"terminal_death_proof {key} mismatch")
    if not str(proof.get("normal_collision_death_evidence_id") or "").strip():
        raise CompactCoachSpeedRowEvidenceError(
            "terminal_death_proof requires normal collision evidence id"
        )
    refs = proof.get("normal_collision_death_evidence_refs")
    if not isinstance(refs, list) or not refs:
        raise CompactCoachSpeedRowEvidenceError(
            "terminal_death_proof requires normal collision evidence refs"
        )
    causes = proof.get("normal_collision_death_causes")
    if not isinstance(causes, list) or not causes:
        raise CompactCoachSpeedRowEvidenceError(
            "terminal_death_proof requires normal collision death causes"
        )
    if proof.get("normal_collision_death_hit_owner_present") is not True:
        raise CompactCoachSpeedRowEvidenceError("terminal_death_proof requires hit-owner evidence")
    _require_sha(proof.get("contract_sha256"), "terminal_death_proof contract_sha256")


def _actor_handoff_nonnegative_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if value is None or isinstance(value, bool):
        raise CompactCoachSpeedRowEvidenceError(
            f"actor_handoff_config {key} must be a non-negative integer"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CompactCoachSpeedRowEvidenceError(
            f"actor_handoff_config {key} must be a non-negative integer"
        ) from exc
    if parsed < 0:
        raise CompactCoachSpeedRowEvidenceError(
            f"actor_handoff_config {key} must be a non-negative integer"
        )
    return parsed


def _surface_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = _surface_nonnegative_int(config, key)
    if value <= 0:
        raise CompactCoachSpeedRowEvidenceError(f"{key} must be positive")
    return value


def _surface_nonnegative_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if value is None or isinstance(value, bool):
        raise CompactCoachSpeedRowEvidenceError(f"{key} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CompactCoachSpeedRowEvidenceError(f"{key} must be a non-negative integer") from exc
    if parsed < 0:
        raise CompactCoachSpeedRowEvidenceError(f"{key} must be a non-negative integer")
    return parsed


def _surface_float(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if value is None:
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CompactCoachSpeedRowEvidenceError(f"{key} must be numeric") from exc
    if not math.isfinite(parsed):
        raise CompactCoachSpeedRowEvidenceError(f"{key} must be finite")
    return parsed


def _borrowed_copy_steps_are_terminal_snapshots(surface: Mapping[str, Any]) -> bool:
    return (
        str(surface.get("death_mode") or "") == "normal"
        and int(surface.get("terminal_row_count") or 0) > 0
        and int(surface.get("terminal_final_observation_row_count") or 0) > 0
        and surface.get("terminal_final_observation_before_autoreset_verified") is True
        and surface.get("normal_death_terminal_contract_promotion_gate_satisfied") is True
    )


def _validate_whole_owner_buffer_replay_ceiling_projection(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> None:
    if not any(str(key).startswith(WHOLE_OWNER_BUFFER_REPLAY_CEILING_PREFIX) for key in payload):
        return

    def field(suffix: str) -> str:
        return f"{WHOLE_OWNER_BUFFER_REPLAY_CEILING_PREFIX}{suffix}"

    expected_values = {
        field("schema_id"): WHOLE_OWNER_BUFFER_REPLAY_CEILING_SCHEMA_ID,
        field("projection_only"): True,
        field("production_speed_claim"): False,
        field("touches_live_training"): False,
        field("requires_h100_validation"): True,
        field("speed_currency"): WHOLE_OWNER_BUFFER_REPLAY_CEILING_SPEED_CURRENCY,
        field("projection_source"): WHOLE_OWNER_BUFFER_REPLAY_CEILING_PROJECTION_SOURCE,
        field("h100_validation_status"): "not_run",
        field("variance_interpretation"): "projection_not_measurement",
        field("promotion_eligible"): False,
    }
    for key, expected in expected_values.items():
        if payload.get(key) != expected:
            raise CompactCoachSpeedRowEvidenceError(
                f"{context}.{key} must be {expected!r} for projection-only ceiling"
            )

    numeric_fields = (
        "observed_env_steps",
        "observed_wall_sec",
        "observed_env_steps_per_sec",
        "baseline_env_steps_per_sec",
        "baseline_whole_loop_sec",
        "target_multiplier",
        "target_env_steps_per_sec",
        "target_wall_sec",
        "observed_speedup_vs_opt104",
        "observed_replay_append_sec",
        "observed_owner_train_sample_sec",
        "observed_owner_train_wall_sec",
        "observed_learner_update_sec",
        "observed_worker_search_sec",
        "observed_parent_wait_sec",
        "direct_replay_sample_surface_sec",
        "parent_wait_bounded_surface_sec",
        "preserved_search_update_floor_sec",
        "max_removable_sec",
        "projected_removed_sec",
        "projected_wall_sec",
        "projected_env_steps_per_sec",
        "projected_speedup_vs_opt104",
        "projected_delta_sec",
        "additional_removed_sec_to_2x",
    )
    for suffix in numeric_fields:
        key = field(suffix)
        if key not in payload:
            raise CompactCoachSpeedRowEvidenceError(
                f"{context}.{key} missing from projection-only ceiling"
            )
        try:
            value = float(payload.get(key))
        except (TypeError, ValueError) as exc:
            raise CompactCoachSpeedRowEvidenceError(
                f"{context}.{key} must be finite nonnegative"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise CompactCoachSpeedRowEvidenceError(
                f"{context}.{key} must be finite nonnegative"
            )


def _validate_denominator_section(evidence: Mapping[str, Any]) -> None:
    denominator = _required_mapping(evidence.get("denominator"), "denominator")
    speed_currency = str(denominator.get("speed_currency", "")).strip()
    if not speed_currency:
        raise CompactCoachSpeedRowEvidenceError("denominator speed_currency missing")
    if speed_currency in _PROFILE_ONLY_SPEED_CURRENCIES:
        raise CompactCoachSpeedRowEvidenceError(
            "profile-only speed currency cannot satisfy coach_speed_row"
        )
    allowed = _ALLOWED_SPEED_CURRENCIES_BY_ROUTE.get(str(evidence.get("route")), frozenset())
    if speed_currency not in allowed:
        raise CompactCoachSpeedRowEvidenceError(
            "speed_currency is not allowed for Coach speed-row route"
        )
    row = _required_mapping(evidence.get("row"), "row")
    summary = _required_mapping(evidence.get("result_summary"), "result_summary")
    result_info = _required_mapping(evidence.get("result"), "result")
    result = _read_json_mapping(Path(str(result_info["path"])), "speed-row result")
    loaded_summary = _result_summary(result)
    _validate_whole_owner_buffer_replay_ceiling_projection(
        loaded_summary,
        context="result.summary",
    )
    compact_payload = result.get("compact")
    if isinstance(compact_payload, Mapping):
        _validate_whole_owner_buffer_replay_ceiling_projection(
            compact_payload,
            context="result.compact",
        )
    if str(row.get("speed_currency")) != speed_currency:
        raise CompactCoachSpeedRowEvidenceError("row speed_currency mismatch")
    if str(summary.get("speed_currency")) != speed_currency:
        raise CompactCoachSpeedRowEvidenceError("summary speed_currency mismatch")
    if denominator.get("row_purpose") != "coach_speed_row":
        raise CompactCoachSpeedRowEvidenceError("denominator row_purpose must be coach_speed_row")
    if denominator.get("uses_fallback_denominator") is not False:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row cannot use fallback denominator")
    numerator = _require_positive_number(
        denominator.get("numerator_value"), "denominator numerator_value"
    )
    denom_sec = _require_positive_number(
        denominator.get("denominator_value_sec"),
        "denominator denominator_value_sec",
    )
    reported = _require_positive_number(
        denominator.get("reported_steps_per_sec"),
        "denominator reported_steps_per_sec",
    )
    numerator_field = str(denominator.get("numerator_field", "")).strip()
    denominator_field = str(denominator.get("denominator_field", "")).strip()
    if not numerator_field or not denominator_field:
        raise CompactCoachSpeedRowEvidenceError("denominator fields must be non-empty")
    summary_numerator = _require_positive_number(
        loaded_summary.get(numerator_field),
        f"result summary {numerator_field}",
    )
    summary_denominator = _require_positive_number(
        loaded_summary.get(denominator_field),
        f"result summary {denominator_field}",
    )
    summary_reported = _require_positive_number(
        loaded_summary.get("steps_per_sec"),
        "result summary steps_per_sec",
    )
    _require_same_number(numerator, summary_numerator, "denominator numerator")
    _require_same_number(denom_sec, summary_denominator, "denominator value")
    _require_same_number(reported, summary_reported, "denominator reported speed")
    expected = numerator / denom_sec
    tolerance = max(1.0e-6, abs(expected) * 1.0e-6)
    if abs(reported - expected) > tolerance:
        raise CompactCoachSpeedRowEvidenceError(
            "denominator reported_steps_per_sec arithmetic mismatch"
        )


def _validate_model_identity_section(evidence: Mapping[str, Any]) -> None:
    identity = _required_mapping(evidence.get("model_identity"), "model_identity")
    scope = str(identity.get("scope", "")).strip()
    if scope not in _ALLOWED_MODEL_IDENTITY_SCOPES:
        raise CompactCoachSpeedRowEvidenceError("Coach speed-row model_identity scope mismatch")
    lifecycle_identity = _required_mapping(
        identity.get("lifecycle_identity"),
        "model_identity.lifecycle_identity",
    )
    if str(lifecycle_identity.get("checkpoint_id", "")) != str(
        evidence.get("candidate_checkpoint_id", "")
    ):
        raise CompactCoachSpeedRowEvidenceError("lifecycle model identity checkpoint_id mismatch")
    lifecycle_info = _required_mapping(evidence.get("unified_lifecycle"), "unified_lifecycle")
    loaded_lifecycle = _read_json_mapping(
        Path(str(lifecycle_info["path"])),
        "unified lifecycle report",
    )
    expected_lifecycle_identity = _lifecycle_identity_from_report(
        loaded_lifecycle,
        lifecycle_path=Path(str(lifecycle_info["path"])),
    )
    _require_identity_matches(
        lifecycle_identity,
        expected_lifecycle_identity,
        context="recorded lifecycle model identity",
        require_full=False,
    )

    result_identity = _required_mapping(
        identity.get("result_loaded_checkpoint_identity"),
        "model_identity.result_loaded_checkpoint_identity",
    )
    result_info = _required_mapping(evidence.get("result"), "result")
    result = _read_json_mapping(Path(str(result_info["path"])), "speed-row result")
    compact_payload = _required_mapping(result.get("compact"), "result compact payload")
    loaded_result_identity = _result_loaded_checkpoint_identity(compact_payload)
    if _json_sha256(result_identity) != _json_sha256(loaded_result_identity):
        raise CompactCoachSpeedRowEvidenceError("result loaded checkpoint identity mismatch")

    if scope == COMPACT_COACH_MODEL_IDENTITY_SCOPE_SUPPORT_ONLY:
        return
    if result_identity.get("candidate_loaded_checkpoint") is not True:
        raise CompactCoachSpeedRowEvidenceError("result must mark candidate_loaded_checkpoint=true")
    _require_identity_matches(
        result_identity,
        lifecycle_identity,
        context="result loaded checkpoint identity",
        require_full=True,
    )
    _require_sha(
        result_identity.get("model_state_digest"),
        "result loaded checkpoint model_state_digest",
    )
    compact_checkpoint_sha = result_identity.get("compact_checkpoint_sha256")
    if compact_checkpoint_sha is not None:
        _require_sha(
            compact_checkpoint_sha,
            "result loaded checkpoint compact_checkpoint_sha256",
        )


def _validate_nonclaims(raw: Any, context: str) -> None:
    if not isinstance(raw, Mapping):
        raise CompactCoachSpeedRowEvidenceError(f"{context} missing")
    for key in _NONCLAIM_FALSE_KEYS:
        if raw.get(key) is not False:
            raise CompactCoachSpeedRowEvidenceError(f"{context} {key} must be false")


def _unchecked_compact_coach_speed_row_evidence_ref(
    evidence: Mapping[str, Any],
) -> str:
    manifest = _required_mapping(evidence.get("manifest"), "manifest")
    row = _required_mapping(evidence.get("row"), "row")
    result = _required_mapping(evidence.get("result"), "result")
    return (
        f"{COMPACT_COACH_SPEED_ROW_EVIDENCE_REF_PREFIX}"
        f"{evidence['candidate_checkpoint_id']}:"
        f"{evidence['evidence_id']}:"
        f"row={manifest['experiment_id']}/{row['row_id']}:"
        f"result_sha256={result['sha256']}"
    )


def _manifest_row(manifest: Mapping[str, Any], *, row_id: str) -> Mapping[str, Any]:
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise CompactCoachSpeedRowEvidenceError("manifest rows missing")
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("row_id")) == str(row_id):
            return row
    raise CompactCoachSpeedRowEvidenceError(f"manifest row_id {row_id!r} missing")


def _result_summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise CompactCoachSpeedRowEvidenceError("result summary missing")
    return summary


def _speed_row_search_config(
    *,
    manifest: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    summary: Mapping[str, Any],
    compact_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sources = [manifest, manifest_row, summary]
    if isinstance(compact_payload, Mapping):
        sources.append(compact_payload)
    return {
        key: _coherent_field(key, sources)
        for key in (
            "search_service_kind",
            "search_service_impl",
            "compact_torch_initial_inference_mode",
            "compact_torch_observation_memory_format",
            "compact_torch_model_memory_format",
            "compact_torch_defer_one_simulation_replay_payload_requested",
            "compact_torch_memory_format_applies_to_search_service",
        )
    }


def _speed_row_actor_handoff_config(
    *,
    manifest: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    summary: Mapping[str, Any],
    compact_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sources = [manifest, manifest_row, summary]
    if isinstance(compact_payload, Mapping):
        sources.append(compact_payload)
    persistent_requested = bool(
        _coherent_field(
            "hybrid_persistent_compact_render_state_buffer",
            sources,
        )
    )
    borrow_requested = bool(
        _coherent_field(
            "hybrid_borrow_single_actor_render_state",
            sources,
        )
    )
    raw_mode = _coherent_field("render_state_handoff_mode", sources)
    if raw_mode is None:
        if borrow_requested:
            mode = "borrow_single_actor_env_state"
        elif persistent_requested:
            mode = "persistent_compact_render_state_buffer"
        else:
            mode = "copy_actor_state_to_parent_buffers"
    else:
        mode = str(raw_mode)

    return {
        "hybrid_persistent_compact_render_state_buffer": persistent_requested,
        "hybrid_borrow_single_actor_render_state": borrow_requested,
        "render_state_handoff_mode": mode,
        "render_state_copy_steps": _coherent_int_field(
            "render_state_copy_steps",
            sources,
            default=0 if persistent_requested or borrow_requested else 1,
        ),
        "render_state_borrowed_steps": _coherent_int_field(
            "render_state_borrowed_steps",
            sources,
            default=0,
        ),
        "render_state_row_overlay_steps": _coherent_int_field(
            "render_state_row_overlay_steps",
            sources,
            default=0,
        ),
        "render_state_row_overlay_rows": _coherent_int_field(
            "render_state_row_overlay_rows",
            sources,
            default=0,
        ),
        "render_state_row_overlay_bytes": _coherent_int_field(
            "render_state_row_overlay_bytes",
            sources,
            default=0,
        ),
    }


_SPEED_ROW_ACTOR_OBSERVATION_TIMER_FIELDS = (
    "speed_row_actor_step_sec",
    "speed_row_actor_idle_wait_sec",
    "speed_row_actor_payload_copy_sec",
    "speed_row_actor_compact_write_sec",
    "speed_row_actor_render_state_write_sec",
    "speed_row_actor_autoreset_sec",
    "speed_row_actor_env_runtime_sec",
    "speed_row_actor_env_runtime_step_many_sec",
    "speed_row_actor_env_runtime_movement_sec",
    "speed_row_actor_env_runtime_collision_sec",
    "speed_row_actor_env_runtime_visual_trail_append_sec",
    "speed_row_actor_env_runtime_body_append_sec",
    "speed_row_actor_env_runtime_phase_accounted_sec",
    "speed_row_actor_env_runtime_phase_residual_sec",
    "speed_row_actor_env_public_prepare_sec",
    "speed_row_actor_env_public_info_sec",
    "speed_row_actor_env_compact_action_mask_sec",
    "speed_row_actor_env_reward_sec",
    "speed_row_actor_env_final_observation_sec",
    "speed_row_actor_env_batch_pack_sec",
    "speed_row_actor_env_post_runtime_bookkeeping_sec",
    "speed_row_actor_step_other_sec",
    "speed_row_renderer_render_sec",
    "speed_row_renderer_device_render_sec",
    "speed_row_renderer_host_to_device_sec",
    "speed_row_renderer_device_to_host_sec",
    "speed_row_renderer_production_to_compact_sec",
    "speed_row_renderer_persistent_compact_state_handoff_sec",
    "speed_row_renderer_persistent_delta_pack_sec",
    "speed_row_renderer_persistent_update_sec",
    "speed_row_stack_shift_sec",
    "speed_row_stack_latest_update_sec",
    "speed_row_resident_observation_stack_update_sec",
    "speed_row_resident_observation_frame_view_sec",
    "speed_row_resident_observation_stack_shift_sec",
    "speed_row_resident_observation_latest_write_sec",
    "speed_row_resident_observation_autoreset_sec",
    "speed_row_resident_observation_autoreset_frame_view_sec",
    "speed_row_resident_observation_autoreset_index_build_sec",
    "speed_row_resident_observation_autoreset_zero_sec",
    "speed_row_resident_observation_autoreset_latest_write_sec",
    "speed_row_scalar_materialization_sec",
    "speed_row_resident_observation_replay_snapshot_sec",
    "speed_row_observation_other_sec",
)
_SOURCE_PROFILE_TIMING_TO_SPEED_ROW_FIELD = {
    "actor_step_sec": "speed_row_actor_step_sec",
    "actor_idle_wait_sec": "speed_row_actor_idle_wait_sec",
    "actor_payload_copy_sec": "speed_row_actor_payload_copy_sec",
    "actor_compact_write_sec": "speed_row_actor_compact_write_sec",
    "actor_render_state_write_sec": "speed_row_actor_render_state_write_sec",
    "actor_autoreset_sec": "speed_row_actor_autoreset_sec",
    "actor_env_runtime_sec": "speed_row_actor_env_runtime_sec",
    "actor_env_runtime_step_many_sec": "speed_row_actor_env_runtime_step_many_sec",
    "actor_env_runtime_movement_sec": "speed_row_actor_env_runtime_movement_sec",
    "actor_env_runtime_collision_sec": "speed_row_actor_env_runtime_collision_sec",
    "actor_env_runtime_visual_trail_append_sec": (
        "speed_row_actor_env_runtime_visual_trail_append_sec"
    ),
    "actor_env_runtime_body_append_sec": "speed_row_actor_env_runtime_body_append_sec",
    "actor_env_runtime_phase_accounted_sec": (
        "speed_row_actor_env_runtime_phase_accounted_sec"
    ),
    "actor_env_runtime_phase_residual_sec": (
        "speed_row_actor_env_runtime_phase_residual_sec"
    ),
    "actor_env_public_prepare_sec": "speed_row_actor_env_public_prepare_sec",
    "actor_env_public_info_sec": "speed_row_actor_env_public_info_sec",
    "actor_env_compact_action_mask_sec": "speed_row_actor_env_compact_action_mask_sec",
    "actor_env_reward_sec": "speed_row_actor_env_reward_sec",
    "actor_env_final_observation_sec": "speed_row_actor_env_final_observation_sec",
    "actor_env_batch_pack_sec": "speed_row_actor_env_batch_pack_sec",
    "actor_env_post_runtime_bookkeeping_sec": (
        "speed_row_actor_env_post_runtime_bookkeeping_sec"
    ),
    "renderer_render_sec": "speed_row_renderer_render_sec",
    "renderer_device_render_sec": "speed_row_renderer_device_render_sec",
    "renderer_host_to_device_sec": "speed_row_renderer_host_to_device_sec",
    "renderer_device_to_host_sec": "speed_row_renderer_device_to_host_sec",
    "renderer_production_to_compact_sec": "speed_row_renderer_production_to_compact_sec",
    "renderer_persistent_compact_state_handoff_sec": (
        "speed_row_renderer_persistent_compact_state_handoff_sec"
    ),
    "renderer_persistent_delta_pack_sec": "speed_row_renderer_persistent_delta_pack_sec",
    "renderer_persistent_update_sec": "speed_row_renderer_persistent_update_sec",
    "stack_shift_sec": "speed_row_stack_shift_sec",
    "stack_latest_update_sec": "speed_row_stack_latest_update_sec",
    "resident_observation_stack_update_sec": (
        "speed_row_resident_observation_stack_update_sec"
    ),
    "resident_observation_frame_view_sec": "speed_row_resident_observation_frame_view_sec",
    "resident_observation_stack_shift_sec": "speed_row_resident_observation_stack_shift_sec",
    "resident_observation_latest_write_sec": (
        "speed_row_resident_observation_latest_write_sec"
    ),
    "resident_observation_autoreset_sec": (
        "speed_row_resident_observation_autoreset_sec"
    ),
    "resident_observation_autoreset_frame_view_sec": (
        "speed_row_resident_observation_autoreset_frame_view_sec"
    ),
    "resident_observation_autoreset_index_build_sec": (
        "speed_row_resident_observation_autoreset_index_build_sec"
    ),
    "resident_observation_autoreset_zero_sec": (
        "speed_row_resident_observation_autoreset_zero_sec"
    ),
    "resident_observation_autoreset_latest_write_sec": (
        "speed_row_resident_observation_autoreset_latest_write_sec"
    ),
    "scalar_materialization_sec": "speed_row_scalar_materialization_sec",
    "resident_observation_replay_snapshot_sec": (
        "speed_row_resident_observation_replay_snapshot_sec"
    ),
}


def _float_or_zero(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _source_profile_timing_speed_row_fields(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    timings = profile_payload.get("timings")
    if not isinstance(timings, Mapping):
        return {}
    fields: dict[str, Any] = {
        speed_key: _plain_value(timings.get(timing_key))
        for timing_key, speed_key in _SOURCE_PROFILE_TIMING_TO_SPEED_ROW_FIELD.items()
        if timing_key in timings and timings.get(timing_key) is not None
    }
    actor_step = fields.get("speed_row_actor_step_sec")
    if actor_step is not None and "speed_row_actor_step_other_sec" not in fields:
        actor_child_keys = (
            "speed_row_actor_autoreset_sec",
            "speed_row_actor_env_runtime_sec",
            "speed_row_actor_env_public_prepare_sec",
            "speed_row_actor_env_public_info_sec",
            "speed_row_actor_env_compact_action_mask_sec",
            "speed_row_actor_env_reward_sec",
            "speed_row_actor_env_final_observation_sec",
            "speed_row_actor_env_batch_pack_sec",
            "speed_row_actor_env_post_runtime_bookkeeping_sec",
        )
        fields["speed_row_actor_step_other_sec"] = _float_or_zero(actor_step) - sum(
            _float_or_zero(fields.get(key)) for key in actor_child_keys
        )
    observation = profile_payload.get("observation_sec")
    if observation is None:
        observation = timings.get("observation_sec")
    if observation is not None and "speed_row_observation_other_sec" not in fields:
        observation_child_keys = (
            "speed_row_renderer_render_sec",
            "speed_row_stack_shift_sec",
            "speed_row_stack_latest_update_sec",
            "speed_row_resident_observation_stack_update_sec",
            "speed_row_resident_observation_autoreset_sec",
            "speed_row_scalar_materialization_sec",
            "speed_row_resident_observation_replay_snapshot_sec",
        )
        fields["speed_row_observation_other_sec"] = _float_or_zero(observation) - sum(
            _float_or_zero(fields.get(key)) for key in observation_child_keys
        )
    return fields


def _speed_row_operational_surface(
    *,
    manifest: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
    summary: Mapping[str, Any],
    compact_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    sources: list[Mapping[str, Any]] = [manifest, manifest_row, summary]
    if isinstance(compact_payload, Mapping):
        sources.append(compact_payload)
        profile_payload = compact_payload.get("source_profile_payload")
        if isinstance(profile_payload, Mapping):
            sources.append(profile_payload)
            timing_surface = _source_profile_timing_speed_row_fields(profile_payload)
            if timing_surface:
                sources.append(timing_surface)
    surface = {
        "actor_count": _coherent_surface_value(sources, "actor_count", default=1),
        "batch_size": _coherent_surface_value(sources, "batch_size", default=1),
        "steps": _coherent_surface_value(sources, "steps", default=1),
        "warmup_steps": _coherent_surface_value(sources, "warmup_steps", default=0),
        "death_mode": _coherent_surface_value(
            sources,
            "death_mode",
            default="profile_no_death",
        ),
        "compact_owned_training_loop_owner": _first_surface_value(
            sources,
            "compact_owned_training_loop_owner",
            default="",
        ),
        "compact_owned_trainer_config_death_mode": _first_surface_value(
            sources,
            "compact_owned_trainer_config_death_mode",
            default="",
        ),
        "normal_death_terminal_contract_owner": _first_surface_value(
            sources,
            "normal_death_terminal_contract_owner",
            default="none",
        ),
        "terminal_row_count": _first_surface_value(
            sources,
            "terminal_row_count",
            default=0,
        ),
        "death_row_count": _first_surface_value(
            sources,
            "death_row_count",
            "terminal_death_row_count",
            default=0,
        ),
        "terminated_row_count": _first_surface_value(
            sources,
            "terminated_row_count",
            default=0,
        ),
        "truncated_row_count": _first_surface_value(
            sources,
            "truncated_row_count",
            default=0,
        ),
        "terminal_final_observation_row_count": _first_surface_value(
            sources,
            "terminal_final_observation_row_count",
            "next_final_observation_row_count",
            default=0,
        ),
        "terminal_final_observation_before_autoreset_verified": _first_surface_value(
            sources,
            "terminal_final_observation_before_autoreset_verified",
            default=False,
        ),
        "terminal_sample_row_count": _first_surface_value(
            sources,
            "terminal_sample_row_count",
            "compact_rollout_slab_sample_gate_terminal_sample_rows",
            default=0,
        ),
        "terminal_unroll_value_target_mode": _first_surface_value(
            sources,
            "terminal_unroll_value_target_mode",
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode",
            default="none",
        ),
        "terminal_unroll_value_target_row_count": _first_surface_value(
            sources,
            "terminal_unroll_value_target_row_count",
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_rows",
            default=0,
        ),
        "resident_observation_host_fallback_count": _first_surface_value(
            sources,
            "resident_observation_host_fallback_count",
            default=0.0,
        ),
        "normal_death_terminal_contract_promotion_gate_satisfied": _first_surface_value(
            sources,
            "normal_death_terminal_contract_promotion_gate_satisfied",
            default=False,
        ),
        "source_profile_total_sec": _first_surface_value(
            sources,
            "source_profile_total_sec",
            default=0.0,
        ),
        "source_profile_warmup_sec": _first_surface_value(
            sources,
            "source_profile_warmup_sec",
            default=0.0,
        ),
        "source_profile_measured_sec": _first_surface_value(
            sources,
            "source_profile_measured_sec",
            default=0.0,
        ),
        "source_profile_timing_per_timestep_sec": _first_surface_value(
            sources,
            "source_profile_timing_per_timestep_sec",
            default=0.0,
        ),
        "speed_row_actor_step_wall_sec": _first_surface_value(
            sources,
            "speed_row_actor_step_wall_sec",
            default=0.0,
        ),
        "speed_row_observation_sec": _first_surface_value(
            sources,
            "speed_row_observation_sec",
            default=0.0,
        ),
        "speed_row_renderer_stack_update_sec": _first_surface_value(
            sources,
            "speed_row_renderer_stack_update_sec",
            default=0.0,
        ),
        "speed_row_compact_rollout_slab_sec": _first_surface_value(
            sources,
            "speed_row_compact_rollout_slab_sec",
            default=0.0,
        ),
        "speed_row_sample_gate_sec": _first_surface_value(
            sources,
            "speed_row_sample_gate_sec",
            default=0.0,
        ),
        "speed_row_learner_gate_sec": _first_surface_value(
            sources,
            "speed_row_learner_gate_sec",
            default=0.0,
        ),
        "speed_row_policy_refresh_sec": _first_surface_value(
            sources,
            "speed_row_policy_refresh_sec",
            default=0.0,
        ),
        "speed_row_primary_accounted_sec": _first_surface_value(
            sources,
            "speed_row_primary_accounted_sec",
            default=0.0,
        ),
        "speed_row_primary_residual_sec": _first_surface_value(
            sources,
            "speed_row_primary_residual_sec",
            default=0.0,
        ),
    }
    surface.update(_speed_row_repeatability_surface(sources))
    timer_surface = {
        key: _first_surface_value(sources, key, default=0.0)
        for key in _SPEED_ROW_ACTOR_OBSERVATION_TIMER_FIELDS
        if any(key in source and source.get(key) is not None for source in sources)
    }
    if timer_surface:
        surface.update(timer_surface)
    surface.update(_speed_row_sample_learner_fusion_surface(sources))
    return surface


def _speed_row_repeatability_surface(
    sources: list[Mapping[str, Any]],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in _REPEATABILITY_RESULT_SUMMARY_FIELDS:
        if any(key in source and source.get(key) is not None for source in sources):
            fields[key] = _coherent_surface_field(key, sources)
    return fields


def _speed_row_sample_learner_fusion_surface(
    sources: list[Mapping[str, Any]],
) -> dict[str, Any]:
    keys = (
        "learner_num_unroll_steps",
        "compact_owned_loop_fused_learner_batch",
        "compact_owner_search_action_only_result",
        "compact_owner_search_owner_materializes_replay",
        "compact_owner_search_parent_slab_commits_replay",
        "compact_rollout_slab_sample_gate_calls",
        "compact_rollout_slab_learner_gate_calls",
        "compact_muzero_learner_batch_unroll2_specialized_builder",
        "compact_muzero_learner_batch_learner_ready_unroll2_cache",
        "compact_muzero_learner_batch_tensor_native_replay",
        "compact_rollout_slab_sample_gate_sec",
        "compact_rollout_slab_learner_gate_sec",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only",
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch",
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch",
        "compact_rollout_slab_sample_gate_host_provider_learner_batch",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes",
        "compact_rollout_slab_learner_gate_prebuilt_batch_used",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_impl",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_direct_build_used",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason",
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped",
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested",
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used",
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_missing_record_count",
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_rows",
        "compact_rollout_slab_sample_gate_fixed_soa_requested",
        "compact_rollout_slab_sample_gate_fixed_soa_used",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_schema_id",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_requested",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_used",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_handle_id",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_snapshot_version",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_request_checksum",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_sample_row_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_target_row_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_create_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_resolve_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_inline_resolve_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_reason",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_pending_handle_count",
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_count",
        "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count",
        "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count",
        "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count",
        "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count",
        "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count",
        "compact_rollout_slab_sample_gate_fixed_soa_record_count",
        "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count",
        "compact_rollout_slab_sample_gate_fixed_soa_table_row_count",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count",
        "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count",
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_count",
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason",
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec",
        "compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec",
        "compact_rollout_slab_sample_gate_fixed_soa_total_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path",
        "compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled",
        "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep",
        "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used",
        "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count",
        "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count",
        "compact_muzero_learner_prebuilt_batch_used",
    )
    if not any(any(key in source for key in keys) for source in sources):
        return {}
    return {
        "learner_num_unroll_steps": _first_surface_value(
            sources,
            "learner_num_unroll_steps",
            "compact_rollout_slab_learner_gate_num_unroll_steps",
            default=None,
        ),
        "compact_owned_loop_fused_learner_batch": _first_surface_value(
            sources,
            "compact_owned_loop_fused_learner_batch",
            default=False,
        ),
        "compact_owned_training_loop_owner": _first_surface_value(
            sources,
            "compact_owned_training_loop_owner",
            default="",
        ),
        "compact_owner_search_action_only_result": _first_surface_value(
            sources,
            "compact_owner_search_action_only_result",
            default=False,
        ),
        "compact_owner_search_owner_materializes_replay": _first_surface_value(
            sources,
            "compact_owner_search_owner_materializes_replay",
            default=False,
        ),
        "compact_owner_search_parent_slab_commits_replay": _first_surface_value(
            sources,
            "compact_owner_search_parent_slab_commits_replay",
            default=True,
        ),
        "compact_owner_search_learner_update_count": _first_surface_value(
            sources,
            "compact_owner_search_learner_update_count",
            default=None,
        ),
        "compact_owner_search_owner_train_request_count": _first_surface_value(
            sources,
            "compact_owner_search_owner_train_request_count",
            default=None,
        ),
        "compact_owner_search_owner_expected_train_request_count": _first_surface_value(
            sources,
            "compact_owner_search_owner_expected_train_request_count",
            default=None,
        ),
        "compact_owner_search_owner_learner_update_count": _first_surface_value(
            sources,
            "compact_owner_search_owner_learner_update_count",
            default=None,
        ),
        "compact_owner_search_worker_learner_train_sec": _first_surface_value(
            sources,
            "compact_owner_search_worker_learner_train_sec",
            default=0.0,
        ),
        "compact_owner_search_owner_train_wall_sec": _first_surface_value(
            sources,
            "compact_owner_search_owner_train_wall_sec",
            default=0.0,
        ),
        "speed_row_total_owner_search_worker_learner_train_sec": _first_surface_value(
            sources,
            "speed_row_total_owner_search_worker_learner_train_sec",
            default=0.0,
        ),
        "compact_rollout_slab_sample_gate_calls": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_calls",
            default=0,
        ),
        "compact_rollout_slab_learner_gate_calls": _first_surface_value(
            sources,
            "compact_rollout_slab_learner_gate_calls",
            default=0,
        ),
        "compact_muzero_learner_batch_unroll2_specialized_builder": _first_surface_value(
            sources,
            "compact_muzero_learner_batch_unroll2_specialized_builder",
            default=False,
        ),
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": _first_surface_value(
            sources,
            "compact_muzero_learner_batch_learner_ready_unroll2_cache",
            default=False,
        ),
        "compact_muzero_learner_batch_tensor_native_replay": _first_surface_value(
            sources,
            "compact_muzero_learner_batch_tensor_native_replay",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_sec": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_sec",
            default=0.0,
        ),
        "compact_rollout_slab_learner_gate_sec": _first_surface_value(
            sources,
            "compact_rollout_slab_learner_gate_sec",
            default=0.0,
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_host_provider_learner_batch": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_host_provider_learner_batch",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source",
            default="",
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes",
            default=0,
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_used": _first_surface_value(
            sources,
            "compact_rollout_slab_learner_gate_prebuilt_batch_used",
            default=False,
        ),
        "compact_muzero_learner_prebuilt_batch_used": _first_surface_value(
            sources,
            "compact_muzero_learner_prebuilt_batch_used",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order",
            default="unknown",
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order",
            default=None,
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count": _first_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count",
            default=-1,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_impl": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_build_impl"
            ),
            default="none",
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_direct_build_used": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_direct_build_used"
            ),
            default=False,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_reused_record_count"
            ),
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_missing_record_count"
            ),
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_concat_sec"
            ),
            default=0.0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_sec": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_build_sec"
            ),
            default=0.0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec",
            default=0.0,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_prebuilt_path_requested"
            ),
            default=False,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_prebuilt_path_eligible"
            ),
            default=False,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_prebuilt_path_used"
            ),
            default=False,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_prebuilt_fallback_count"
            ),
            default=0,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_prebuilt_fallback_reason"
            ),
            default="none",
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_group_object_count"
            ),
            default=0,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_group_object_build_skipped"
            ),
            default=False,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_fast_metadata_path_requested"
            ),
            default=False,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_fast_metadata_path_used"
            ),
            default=False,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_fast_metadata_selected_group_count"
            ),
            default=0,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_maintained_table_handle_requested"
            ),
            default=False,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_maintained_table_handle_used"
            ),
            default=False,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_maintained_table_handle_record_count"
            ),
            default=0,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_missing_record_count": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_maintained_table_handle_missing_record_count"
            ),
            default=0,
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_rows": _coherent_surface_value(
            sources,
            (
                "compact_rollout_slab_sample_gate_tensor_native_"
                "direct_maintained_table_handle_rows"
            ),
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_requested": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_requested",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_used": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_used",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_schema_id": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_schema_id",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_requested": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_requested",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_used": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_used",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_handle_id": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_handle_id",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_snapshot_version": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_snapshot_version",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_request_checksum": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_request_checksum",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_sample_row_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_sample_row_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_target_row_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_target_row_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_create_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_create_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_resolve_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_resolve_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_inline_resolve_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_inline_resolve_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_reason": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_reason",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_pending_handle_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_pending_handle_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_slot_write_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_record_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_record_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_table_row_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_table_row_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size",
            default=1,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift",
            default=False,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_count": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_fallback_count",
            default=0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason",
            default="none",
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec",
            default=0.0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec",
            default=0.0,
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_total_sec": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_fixed_soa_total_sec",
            default=0.0,
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": _coherent_surface_value(
            sources,
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path",
            default="none",
        ),
        "compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled": _first_surface_value(
            sources,
            "compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled",
            default=False,
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep": _first_surface_value(
            sources,
            "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep",
            default=True,
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used": _first_surface_value(
            sources,
            "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used",
            default=False,
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count": _first_surface_value(
            sources,
            "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count",
            default=0,
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count": _first_surface_value(
            sources,
            "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count",
            default=0,
        ),
    }


def _coherent_field(key: str, sources: list[Mapping[str, Any]]) -> Any:
    values = [source.get(key) for source in sources if source.get(key) is not None]
    if not values:
        return None
    first = _plain_value(values[0])
    first_hash = _json_sha256(first)
    for value in values[1:]:
        if _json_sha256(value) != first_hash:
            raise CompactCoachSpeedRowEvidenceError(f"search_config {key} mismatch")
    return first


def _coherent_int_field(
    key: str,
    sources: list[Mapping[str, Any]],
    *,
    default: int,
) -> int:
    value = _coherent_field(key, sources)
    if value is None:
        return int(default)
    if isinstance(value, bool):
        raise CompactCoachSpeedRowEvidenceError(
            f"actor_handoff_config {key} must be a non-negative integer"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CompactCoachSpeedRowEvidenceError(
            f"actor_handoff_config {key} must be a non-negative integer"
        ) from exc
    if parsed < 0:
        raise CompactCoachSpeedRowEvidenceError(
            f"actor_handoff_config {key} must be a non-negative integer"
        )
    return parsed


def _coherent_surface_field(key: str, sources: list[Mapping[str, Any]]) -> Any:
    values = [source.get(key) for source in sources if source.get(key) is not None]
    if not values:
        return None
    first = _plain_value(values[0])
    first_hash = _json_sha256(first)
    for value in values[1:]:
        if _json_sha256(value) != first_hash:
            raise CompactCoachSpeedRowEvidenceError(f"operational_surface {key} mismatch")
    return first


def _coherent_surface_value(
    sources: list[Mapping[str, Any]],
    key: str,
    *,
    default: Any,
) -> Any:
    if not any(key in source and source.get(key) is not None for source in sources):
        return default
    return _coherent_surface_field(key, sources)


def _first_surface_value(
    sources: list[Mapping[str, Any]],
    *keys: str,
    default: Any,
) -> Any:
    for source in sources:
        for key in keys:
            if key in source and source.get(key) is not None:
                return _plain_value(source.get(key))
    return default


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise CompactCoachSpeedRowEvidenceError(f"{label} missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CompactCoachSpeedRowEvidenceError(f"{label} must be a mapping")
    return payload


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        raise CompactCoachSpeedRowEvidenceError(f"file missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _plain_value(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha(value: Any, label: str) -> None:
    raw = str(value or "")
    if len(raw) != 64:
        raise CompactCoachSpeedRowEvidenceError(f"{label} must be a sha256 hex")
    try:
        int(raw, 16)
    except ValueError as exc:
        raise CompactCoachSpeedRowEvidenceError(f"{label} must be a sha256 hex") from exc


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactCoachSpeedRowEvidenceError(f"{label} must be a mapping")
    return value


def _lifecycle_identity_from_report(
    lifecycle: Mapping[str, Any],
    *,
    lifecycle_path: Path,
) -> dict[str, Any]:
    current_chain_identity = lifecycle.get("current_chain_identity")
    if isinstance(current_chain_identity, Mapping):
        return _plain_mapping(current_chain_identity)
    checkpoint_path_raw = str(lifecycle.get("compact_checkpoint_path") or "").strip()
    if checkpoint_path_raw:
        checkpoint_path = _resolve_artifact_path(
            checkpoint_path_raw,
            base_dir=lifecycle_path.parent,
        )
        if checkpoint_path.is_file():
            return _compact_checkpoint_identity_from_path(checkpoint_path)
    return {
        "identity_source": "unified_lifecycle_checkpoint_id_only",
        "checkpoint_id": str(lifecycle.get("checkpoint_id") or ""),
    }


def _compact_checkpoint_identity_from_path(path: Path) -> dict[str, Any]:
    from curvyzero.training.compact_policy_refresh_handoff import (
        compact_model_state_digest_v1,
    )
    from curvyzero.training.compact_trainer_checkpoint import (
        load_compact_trainer_checkpoint_v1,
    )

    checkpoint = load_compact_trainer_checkpoint_v1(path)
    metadata = dict(getattr(checkpoint, "metadata", {}) or {})
    return {
        "identity_source": "compact_trainer_checkpoint",
        "candidate_loaded_checkpoint": True,
        "compact_checkpoint_path": str(path),
        "compact_checkpoint_sha256": _file_sha256(path),
        "checkpoint_id": str(metadata.get("checkpoint_id") or ""),
        "trainer_id": str(metadata.get("trainer_id") or ""),
        "policy_version_ref": str(metadata.get("policy_version_ref") or ""),
        "model_version_ref": str(metadata.get("model_version_ref") or ""),
        "policy_source": str(metadata.get("policy_source") or ""),
        "learner_update_count": int(metadata.get("learner_update_count") or 0),
        "model_state_digest": compact_model_state_digest_v1(
            getattr(checkpoint, "model_state_dict")
        ),
    }


def _resolve_artifact_path(raw: str, *, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def _model_identity_scope_from_result(compact_payload: Any) -> str:
    if not isinstance(compact_payload, Mapping):
        return COMPACT_COACH_MODEL_IDENTITY_SCOPE_SUPPORT_ONLY
    raw = compact_payload.get("model_identity_scope")
    if raw is None:
        identity = _result_loaded_checkpoint_identity(compact_payload)
        raw = identity.get("scope")
    scope = str(raw or COMPACT_COACH_MODEL_IDENTITY_SCOPE_SUPPORT_ONLY).strip()
    return scope or COMPACT_COACH_MODEL_IDENTITY_SCOPE_SUPPORT_ONLY


def _result_loaded_checkpoint_identity(compact_payload: Any) -> dict[str, Any]:
    if not isinstance(compact_payload, Mapping):
        return {}
    raw = compact_payload.get("loaded_checkpoint_identity")
    if raw is None:
        raw = compact_payload.get("model_identity")
    if not isinstance(raw, Mapping):
        return {}
    return _plain_mapping(raw)


def _require_identity_matches(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    context: str,
    require_full: bool,
) -> None:
    for key in _IDENTITY_KEYS:
        left_value = str(left.get(key, "")).strip()
        right_value = str(right.get(key, "")).strip()
        if require_full and (not left_value or not right_value):
            raise CompactCoachSpeedRowEvidenceError(f"{context} missing {key}")
        if left_value and right_value and left_value != right_value:
            raise CompactCoachSpeedRowEvidenceError(f"{context} {key} mismatch")
    for key in ("compact_checkpoint_sha256", "compact_checkpoint_path"):
        left_value = str(left.get(key, "")).strip()
        right_value = str(right.get(key, "")).strip()
        if left_value and right_value and left_value != right_value:
            raise CompactCoachSpeedRowEvidenceError(f"{context} {key} mismatch")


def _require_positive_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CompactCoachSpeedRowEvidenceError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise CompactCoachSpeedRowEvidenceError(f"{label} must be finite")
    if number <= 0.0:
        raise CompactCoachSpeedRowEvidenceError(f"{label} must be positive")
    return number


def _require_same_number(left: float, right: float, label: str) -> None:
    tolerance = max(1.0e-9, abs(right) * 1.0e-9)
    if abs(float(left) - float(right)) > tolerance:
        raise CompactCoachSpeedRowEvidenceError(f"{label} must match result summary")


def _nonclaims_from_loaded(raw: Mapping[str, Any]) -> dict[str, Any]:
    nested = raw.get("non_claims")
    return {
        key: _first_present(
            raw.get(key),
            nested.get(key) if isinstance(nested, Mapping) else None,
            False,
        )
        for key in _NONCLAIM_FALSE_KEYS
    }


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    plain = _plain_value(value)
    if not isinstance(plain, dict):
        raise CompactCoachSpeedRowEvidenceError("identity payload must be a mapping")
    return plain


__all__ = [
    "COMPACT_COACH_SPEED_ROW_EVIDENCE_REF_PREFIX",
    "COMPACT_COACH_SPEED_ROW_EVIDENCE_SCHEMA_ID",
    "COMPACT_COACH_SPEED_ROW_MANIFEST_SCHEMA_ID",
    "COMPACT_COACH_SPEED_ROW_PRODUCER_SCHEMA_ID",
    "COMPACT_COACH_SPEED_ROW_RESULT_SCHEMA_ID",
    "COMPACT_COACH_SPEED_ROW_STATUS_VERIFIED",
    "COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT",
    "COMPACT_COACH_MODEL_IDENTITY_SCOPE_SUPPORT_ONLY",
    "CompactCoachSpeedRowEvidenceError",
    "build_compact_coach_speed_row_evidence_v1",
    "compact_coach_speed_row_evidence_path",
    "compact_coach_speed_row_evidence_ref",
    "save_compact_coach_speed_row_evidence_v1",
    "validate_compact_coach_speed_row_evidence_matches_report_v1",
    "validate_compact_coach_speed_row_evidence_v1",
]
