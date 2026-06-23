#!/usr/bin/env python3
"""Launch and collect a Modal/H100 compact Coach speed-row smoke."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.contracts.curvytron import curvytron_runs_volume_name
from curvyzero.env import vector_runtime
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
)
from curvyzero.training.compact_coach_speed_row import (
    compact_coach_speed_row_evidence_ref,
)
from curvyzero.training.compact_coach_speed_row import (
    save_compact_coach_speed_row_evidence_v1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD,
)
from curvyzero.training.compact_owned_loop import COMPACT_SAMPLE_LEARNER_WORKER_KINDS
from curvyzero.training.compact_owned_loop import COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1
from curvyzero.training.compact_owned_loop import COMPACT_REPLAY_APPEND_TRANSPORT_KINDS
from curvyzero.training.compact_owned_loop import COMPACT_MODEL_STATE_TRANSPORT_KINDS
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS,
)
from curvyzero.training.compact_torch_search_service import (
    COMPACT_TORCH_INITIAL_INFERENCE_MODES,
    COMPACT_TORCH_MODEL_COMPILE_MODES,
    COMPACT_TORCH_MEMORY_FORMATS,
)
from curvyzero.training.compact_torch_search_service import COMPACT_TORCH_TIMING_MODES

try:
    import modal
except ImportError:  # pragma: no cover - only when modal extra is absent.
    modal = None  # type: ignore[assignment]


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_coach_speed_row_results")
DEFAULT_RUN_ID = "optimizer-compact-coach-speed-row-h100-20260530"
MODULE = "curvyzero.infra.modal.compact_coach_speed_row"
SPAWN_SCHEMA_ID = "curvyzero_compact_coach_speed_row_spawn/v0"
SPEED_CURRENCY = "compact_trainer_env_steps_per_sec"
ROW_ID = "001"
ACCEPTED_FAST_PATH_PRESET = "compact_trainer_directcore_fused_borrow_lean_b1024a1_normal_unroll2"
ACCEPTED_FAST_PATH_STEP_WINDOW_ACCEPTED = "accepted_180_45"
ACCEPTED_FAST_PATH_STEP_WINDOW_STABILITY_724_180 = "stability_724_180"
ACCEPTED_FAST_PATH_STEP_WINDOW_STABILITY_1084_270 = "stability_1084_270"
ACCEPTED_FAST_PATH_STEP_WINDOW_STABILITY_1444_360 = "stability_1444_360"
ACCEPTED_FAST_PATH_STEP_WINDOWS = {
    ACCEPTED_FAST_PATH_STEP_WINDOW_ACCEPTED: {
        "steps": 180,
        "warmup_steps": 45,
        "comparison_role": "accepted_speed_row",
        "stability_diagnostic": False,
    },
    ACCEPTED_FAST_PATH_STEP_WINDOW_STABILITY_724_180: {
        "steps": 724,
        "warmup_steps": 180,
        "comparison_role": "long_window_stability_diagnostic",
        "stability_diagnostic": True,
    },
    ACCEPTED_FAST_PATH_STEP_WINDOW_STABILITY_1084_270: {
        "steps": 1084,
        "warmup_steps": 270,
        "comparison_role": "long_window_stability_diagnostic",
        "stability_diagnostic": True,
    },
    ACCEPTED_FAST_PATH_STEP_WINDOW_STABILITY_1444_360: {
        "steps": 1444,
        "warmup_steps": 360,
        "comparison_role": "long_window_stability_diagnostic",
        "stability_diagnostic": True,
    },
}


class ModalRemoteLaunchError(RuntimeError):
    """Raised when Modal rejects the detached launch before a FunctionCall exists."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        stdout_tail = str(payload.get("stdout_tail") or "")
        stderr_tail = str(payload.get("stderr_tail") or "")
        super().__init__("remote launch failed:\n" + stdout_tail + stderr_tail)


_ACCEPTED_FAST_PATH_RESULT_REQUIREMENTS = (
    ("summary.seed", "seed", 20260530),
    ("summary.batch_size", "batch_size", 1024),
    ("summary.actor_count", "actor_count", 1),
    ("summary.death_mode", "death_mode", vector_runtime.DEATH_MODE_NORMAL),
    ("summary.sample_seed_base", "sample_seed_base", 20260530),
    ("summary.sample_batch_size", "sample_batch_size", 512),
    ("summary.sample_interval", "sample_interval", 8),
    ("summary.replay_pair_capacity", "replay_pair_capacity", 4096),
    ("summary.learner_train_steps", "learner_train_steps", 1),
    ("summary.learner_num_unroll_steps", "learner_num_unroll_steps", 2),
    ("summary.policy_refresh_interval", "policy_refresh_interval", 4),
    ("summary.num_simulations", "num_simulations", 1),
    ("summary.search_service_kind", "search_service_kind", "compact_torch_search_service"),
    (
        "summary.compact_torch_initial_inference_mode",
        "compact_torch_initial_inference_mode",
        "direct_core",
    ),
    (
        "summary.compact_owned_loop_fused_learner_batch",
        "compact_owned_loop_fused_learner_batch",
        True,
    ),
    ("summary.compact_owned_lean_trainer_step", "compact_owned_lean_trainer_step", True),
    (
        "summary.hybrid_persistent_compact_render_state_buffer",
        "hybrid_persistent_compact_render_state_buffer",
        False,
    ),
    (
        "summary.hybrid_borrow_single_actor_render_state",
        "hybrid_borrow_single_actor_render_state",
        True,
    ),
    (
        "summary.render_state_handoff_mode",
        "render_state_handoff_mode",
        "borrow_single_actor_env_state",
    ),
    ("summary.render_state_copy_steps", "render_state_copy_steps", 0),
    (
        "summary.normal_death_terminal_contract_promotion_gate_satisfied",
        "normal_death_terminal_contract_promotion_gate_satisfied",
        True,
    ),
    ("summary.truncated_row_count", "truncated_row_count", 0),
    (
        "summary.resident_observation_host_fallback_count",
        "resident_observation_host_fallback_count",
        0,
    ),
    ("summary.compact_profile_autoreset_direct_count", "compact_profile_autoreset_direct_count", 0),
)
_ACCEPTED_FAST_PATH_REPEATABILITY_REQUIRED_FIELDS = (
    "compact_rollout_slab_sample_gate_last_seed",
    "compact_rollout_slab_learner_gate_last_seed",
    "compact_owned_loop_sample_gate_last_metadata_seed",
    "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed",
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
    "compact_rollout_slab_sample_gate_action_checksum",
    "compact_rollout_slab_sample_gate_sample_row_checksum",
    "compact_rollout_slab_sample_gate_sample_action_checksum",
    "compact_rollout_slab_sample_gate_sampled_flat_row_checksum",
    "compact_rollout_slab_sample_gate_sample_position_order_checksum",
    "compact_rollout_slab_sample_gate_source_record_pair_checksum",
    "compact_rollout_slab_sample_gate_source_record_window_checksum",
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
_ACCEPTED_FAST_PATH_REPEATABILITY_NONZERO_FIELDS = (
    "env_trajectory_ordered_checksum_total",
    "compact_rollout_slab_sample_gate_sample_position_order_checksum",
    "compact_rollout_slab_sample_gate_source_record_window_checksum",
)
_ACCEPTED_FAST_PATH_REPEATABILITY_POSITIVE_FIELDS = (
    "compact_rollout_slab_sample_gate_sample_rows",
    "compact_rollout_slab_learner_gate_sample_rows",
    "compact_rollout_slab_sample_gate_calls",
    "compact_rollout_slab_learner_gate_updates",
    "compact_rollout_slab_policy_refresh_after_learner_gate_calls",
)
_CUDA_SYNC_TIMING_DIAGNOSTIC_TRUE_FIELDS = (
    "compact_profile_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled",
    ("compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics"),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled"),
    "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled",
)
_CUDA_SYNC_TIMING_DIAGNOSTIC_POSITIVE_FIELDS = (
    "compact_rollout_slab_sample_gate_cuda_sync_count",
    "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count",
    "compact_rollout_slab_learner_gate_cuda_sync_count",
)
_CUDA_SYNC_TIMING_DIAGNOSTIC_SEC_FIELDS = (
    "compact_rollout_slab_sample_gate_cuda_sync_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec",
    "compact_rollout_slab_learner_gate_cuda_sync_sec",
)
_RUNTIME_STEP_TIMING_DIAGNOSTIC_TRUE_FIELDS = ("compact_profile_runtime_step_timing_diagnostics",)
_RUNTIME_STEP_TIMING_DIAGNOSTIC_POSITIVE_FIELDS = ("compact_profile_runtime_step_count",)
_RUNTIME_STEP_TIMING_DIAGNOSTIC_SEC_FIELDS = (
    "compact_profile_runtime_step_sum_sec",
    "compact_profile_runtime_step_min_sec",
    "compact_profile_runtime_step_max_sec",
    "compact_profile_runtime_step_p50_sec",
    "compact_profile_runtime_step_p95_sec",
)
_UNROLL2_SPECIALIZED_BUILDER_KEY = "compact_muzero_learner_batch_unroll2_specialized_builder"
_LEARNER_READY_UNROLL2_CACHE_KEY = "compact_muzero_learner_batch_learner_ready_unroll2_cache"
_TENSOR_NATIVE_REPLAY_KEY = "compact_muzero_learner_batch_tensor_native_replay"
_UNROLL2_SPECIALIZED_BUILDER_REQUESTED_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested"
)
_UNROLL2_SPECIALIZED_BUILDER_USED_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used"
)
_UNROLL2_SPECIALIZED_BUILDER_CALL_COUNT_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count"
)
_UNROLL2_SPECIALIZED_BUILDER_FALLBACK_COUNT_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_"
    "unroll2_specialized_builder_fallback_count"
)
_UNROLL2_SPECIALIZED_BUILDER_PATH_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path"
)
_LEARNER_READY_UNROLL2_CACHE_REQUESTED_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested"
)
_LEARNER_READY_UNROLL2_CACHE_USED_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used"
)
_LEARNER_READY_UNROLL2_CACHE_CALL_COUNT_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count"
)
_LEARNER_READY_UNROLL2_CACHE_FALLBACK_COUNT_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_"
    "learner_ready_unroll2_cache_fallback_count"
)
_LEARNER_READY_UNROLL2_CACHE_PATH_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path"
)
_TENSOR_NATIVE_REPLAY_REQUESTED_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested"
)
_TENSOR_NATIVE_REPLAY_USED_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
)
_TENSOR_NATIVE_REPLAY_CALL_COUNT_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count"
)
_TENSOR_NATIVE_REPLAY_FALLBACK_COUNT_FIELD = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count"
)
_UNROLL2_SPECIALIZED_BUILDER_PROOF_REPORT_FIELDS = (
    _UNROLL2_SPECIALIZED_BUILDER_REQUESTED_FIELD,
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "unroll2_specialized_builder_eligible_count"
    ),
    _UNROLL2_SPECIALIZED_BUILDER_USED_FIELD,
    _UNROLL2_SPECIALIZED_BUILDER_CALL_COUNT_FIELD,
    _UNROLL2_SPECIALIZED_BUILDER_FALLBACK_COUNT_FIELD,
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "unroll2_specialized_builder_fallback_reason"
    ),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl"),
    _UNROLL2_SPECIALIZED_BUILDER_PATH_FIELD,
)
_LEARNER_READY_UNROLL2_CACHE_PROOF_REPORT_FIELDS = (
    _LEARNER_READY_UNROLL2_CACHE_REQUESTED_FIELD,
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "learner_ready_unroll2_cache_available_group_count"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "learner_ready_unroll2_cache_eligible_count"
    ),
    _LEARNER_READY_UNROLL2_CACHE_USED_FIELD,
    _LEARNER_READY_UNROLL2_CACHE_CALL_COUNT_FIELD,
    _LEARNER_READY_UNROLL2_CACHE_FALLBACK_COUNT_FIELD,
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "learner_ready_unroll2_cache_fallback_reason"
    ),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl"),
    _LEARNER_READY_UNROLL2_CACHE_PATH_FIELD,
)
_TENSOR_NATIVE_REPLAY_PROOF_REPORT_FIELDS = (
    _TENSOR_NATIVE_REPLAY_REQUESTED_FIELD,
    _TENSOR_NATIVE_REPLAY_USED_FIELD,
    _TENSOR_NATIVE_REPLAY_CALL_COUNT_FIELD,
    _TENSOR_NATIVE_REPLAY_FALLBACK_COUNT_FIELD,
    ("compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason"),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source"),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "tensor_native_replay_table_build_impl"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "tensor_native_replay_table_direct_build_used"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "tensor_native_replay_table_reused_record_count"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "tensor_native_replay_table_missing_record_count"
    ),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows"),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "tensor_native_replay_table_concat_sec"
    ),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_sec"),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used"),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count"),
    (
        "compact_rollout_slab_sample_gate_tensor_native_direct_"
        "maintained_table_handle_missing_record_count"
    ),
    ("compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_rows"),
    "compact_rollout_slab_sample_gate_fixed_soa_requested",
    "compact_rollout_slab_sample_gate_fixed_soa_used",
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
)
_SAMPLE_GATE_PER_CALL_REPORT_PREFIXES = (
    "compact_rollout_slab_sample_gate_learner_batch_build_per_call",
    "compact_rollout_slab_sample_gate_per_call",
    "compact_rollout_slab_sample_gate_candidate_per_call",
    "compact_rollout_slab_sample_gate_rng_per_call",
    "compact_rollout_slab_sample_gate_residual_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_accounted_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_residual_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_snapshot_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_index_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_observation_per_call",
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "group_loop_terminal_value_bookkeeping_per_call"
    ),
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call",
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_tensor_fallback_per_call"
    ),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_validate_per_call"),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_per_call"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_accounted_per_call"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_residual_per_call"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_presence_per_call"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_select_current_per_call"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_gather_per_call"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_storage_per_call"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_validate_per_call"
    ),
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_builder_select_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_row_index_prepare_per_call",
    ("compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call"),
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_identity_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_build_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_value_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_apply_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_action_stack_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_write_output_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_per_call",
    "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_per_call",
)
_BUILDER_CHILD_CPU_TIME_REPORT_FIELDS = tuple(
    (
        f"compact_rollout_slab_sample_gate_learner_batch_builder_{child_name}_"
        f"{scope}_cpu_time_delta_ns"
    )
    for child_name in (
        "group_loop",
        "group_loop_accounted",
        "group_loop_residual",
        "group_loop_prepare",
        "group_loop_prepare_accounted",
        "group_loop_prepare_residual",
        "group_loop_prepare_snapshot",
        "group_loop_prepare_index",
        "group_loop_prepare_observation",
        "group_loop_terminal_value_bookkeeping",
        "terminal_metadata",
        "terminal_metadata_accounted",
        "terminal_metadata_residual",
        "terminal_metadata_mask",
        "terminal_metadata_tensor_fallback",
        "terminal_metadata_validate",
        "terminal_metadata_final_observation",
        "terminal_metadata_final_observation_accounted",
        "terminal_metadata_final_observation_residual",
        "terminal_metadata_final_observation_presence",
        "terminal_metadata_final_observation_select_current",
        "terminal_metadata_final_observation_gather",
        "terminal_metadata_final_observation_storage",
        "terminal_metadata_final_observation_validate",
        "unroll_terminal_window_hint",
        "unroll_fields",
        "unroll_fields_accounted",
        "unroll_fields_residual",
        "unroll_builder_select",
        "unroll_row_index_prepare",
        "unroll_identity",
        "unroll_stack_fields",
        "unroll_mask_build",
        "unroll_terminal_value",
        "unroll_mask_apply",
        "unroll_action_stack",
        "write_output",
        "order_restore",
        "finalize_outputs",
        "metadata_sync",
        "metadata_build",
    )
    for scope in ("process", "thread")
)
_TERMINAL_FINAL_OBSERVATION_PROOF_REPORT_FIELDS = (
    "compact_rollout_slab_sample_gate_terminal_final_observation_group_count",
    ("compact_rollout_slab_sample_gate_terminal_final_observation_index_fast_path_count"),
    "compact_rollout_slab_sample_gate_terminal_final_observation_fallback_count",
    ("compact_rollout_slab_sample_gate_terminal_final_observation_validate_only_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_materialized_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_final_row_count_sum"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_final_row_count_max"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_dense_storage_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_sparse_storage_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_missing_storage_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_sparse_row_count_sum"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_sparse_row_count_max"),
)
_SAMPLE_LEARNER_TIMER_REPORT_FIELDS = (
    "compact_rollout_slab_sample_gate_candidate_sec",
    "compact_rollout_slab_sample_gate_rng_sec",
    "compact_rollout_slab_sample_gate_resident_check_sec",
    "compact_rollout_slab_sample_gate_group_loop_sec",
    "compact_rollout_slab_sample_gate_metadata_sec",
    "compact_rollout_slab_sample_gate_learner_batch_build_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_accounted_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_residual_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_snapshot_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_index_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_observation_sec",
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "group_loop_terminal_value_bookkeeping_sec"
    ),
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_sec",
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_tensor_fallback_sec"
    ),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_validate_sec"),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_sec"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_accounted_sec"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_residual_sec"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_presence_sec"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_select_current_sec"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_gather_sec"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_storage_sec"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_validate_sec"
    ),
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_builder_select_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_row_index_prepare_sec",
    ("compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_sec"),
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_identity_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_build_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_value_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_apply_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_action_stack_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_write_output_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_sec",
    *_BUILDER_CHILD_CPU_TIME_REPORT_FIELDS,
    *_TERMINAL_FINAL_OBSERVATION_PROOF_REPORT_FIELDS,
    "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled",
    "compact_rollout_slab_sample_gate_cuda_sync_count",
    "compact_rollout_slab_sample_gate_cuda_sync_sec",
    ("compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics"),
    ("compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled"),
    "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count",
    "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec",
    "compact_rollout_slab_sample_gate_sample_batch_build_sec",
    "compact_rollout_slab_sample_gate_accounted_sec",
    "compact_rollout_slab_sample_gate_residual_sec",
    *(
        f"{prefix}_{stat_name}"
        for prefix in _SAMPLE_GATE_PER_CALL_REPORT_PREFIXES
        for stat_name in (
            "count",
            "sum_sec",
            "min_sec",
            "max_sec",
            "p50_sec",
            "p95_sec",
            "slowest_call_index",
            "slowest_iteration",
            "slowest_measured_iteration",
        )
    ),
    "compact_rollout_slab_sample_gate_call_trace_records",
    "compact_rollout_slab_learner_gate_validation_sec",
    "compact_rollout_slab_learner_gate_zero_grad_sec",
    "compact_rollout_slab_learner_gate_target_transform_sec",
    "compact_rollout_slab_learner_gate_initial_inference_sec",
    "compact_rollout_slab_learner_gate_recurrent_inference_sec",
    "compact_rollout_slab_learner_gate_loss_build_sec",
    "compact_rollout_slab_learner_gate_backward_sec",
    "compact_rollout_slab_learner_gate_grad_clip_sec",
    "compact_rollout_slab_learner_gate_optimizer_step_sec",
    "compact_rollout_slab_learner_gate_loss_readback_sec",
    "compact_rollout_slab_learner_gate_final_sync_sec",
    "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled",
    "compact_rollout_slab_learner_gate_cuda_sync_count",
    "compact_rollout_slab_learner_gate_cuda_sync_sec",
    "compact_rollout_slab_learner_gate_accounted_sec",
    "compact_rollout_slab_learner_gate_residual_sec",
    "compact_profile_runtime_step_timing_diagnostics",
    "compact_profile_runtime_step_count",
    "compact_profile_runtime_step_sum_sec",
    "compact_profile_runtime_step_min_sec",
    "compact_profile_runtime_step_max_sec",
    "compact_profile_runtime_step_p50_sec",
    "compact_profile_runtime_step_p95_sec",
    *(
        f"compact_profile_runtime_step_{phase_name}_{stat_name}"
        for phase_name in (
            "actor_step_wall",
            "actor_env_runtime",
            "actor_autoreset",
            "observation",
            "compact_rollout_slab",
            "sample_gate",
            "sample_gate_residual",
            "sample_gate_cuda_sync",
            "sample_gate_builder_group_loop",
            "sample_gate_builder_cuda_sync",
            "learner_gate",
            "policy_refresh",
            "primary_accounted",
            "primary_residual",
        )
        for stat_name in ("sum_sec", "min_sec", "max_sec", "p50_sec", "p95_sec")
    ),
    "compact_profile_runtime_step_slowest_iteration",
    "compact_profile_runtime_step_slowest_measured_iteration",
    "compact_profile_runtime_step_slowest_actor_step_wall_sec",
    "compact_profile_runtime_step_slowest_observation_sec",
    "compact_profile_runtime_step_slowest_compact_rollout_slab_sec",
    "compact_profile_runtime_step_slowest_sample_gate_sec",
    "compact_profile_runtime_step_slowest_learner_gate_sec",
    "compact_profile_runtime_step_slowest_policy_refresh_sec",
    "compact_profile_runtime_step_slowest_primary_accounted_sec",
    "compact_profile_runtime_step_slowest_primary_residual_sec",
    "compact_profile_runtime_step_slowest_env_trajectory_checksum",
    "compact_profile_runtime_step_top_slowest_records",
    *(
        f"compact_profile_runtime_step_{bucket_name}_{stat_name}"
        for bucket_name in (
            "sample_gate_active",
            "sample_gate_inactive",
            "early",
            "mid",
            "late",
        )
        for stat_name in (
            "count",
            "sum_sec",
            "min_sec",
            "max_sec",
            "p50_sec",
            "p95_sec",
            "sample_gate_active_count",
            "actor_step_wall_sum_sec",
            "observation_sum_sec",
            "sample_gate_sum_sec",
            "sample_gate_residual_sum_sec",
            "sample_gate_builder_group_loop_sum_sec",
            "learner_gate_sum_sec",
            "primary_residual_sum_sec",
        )
    ),
    *(
        f"compact_profile_runtime_step_{active_prefix}_{phase_name}_{stat_name}"
        for active_prefix in (
            "sample_gate_active",
            "early_sample_gate_active",
            "mid_sample_gate_active",
            "late_sample_gate_active",
        )
        for phase_name in (
            "sample_gate",
            "sample_gate_residual",
            "sample_gate_builder_group_loop",
            "learner_gate",
            "observation",
            "primary_residual",
        )
        for stat_name in ("count", "sum_sec", "min_sec", "max_sec", "p50_sec", "p95_sec")
    ),
)
_ACTOR_OBSERVATION_TIMER_REPORT_FIELDS = (
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
_GPU_UTILIZATION_REPORT_FIELDS = (
    "speed_row_gpu_utilization_sampling_enabled",
    "speed_row_gpu_utilization_sample_interval_sec",
    "speed_row_gpu_utilization_sample_count",
    "speed_row_gpu_name",
    "speed_row_gpu_utilization_max_percent",
    "speed_row_gpu_utilization_mean_percent",
    "speed_row_gpu_utilization_nonzero_sample_count",
    "speed_row_gpu_utilization_over_50_sample_count",
    "speed_row_gpu_utilization_over_80_sample_count",
    "speed_row_gpu_memory_utilization_max_percent",
    "speed_row_gpu_memory_used_max_mib",
    "speed_row_gpu_power_draw_max_w",
    "speed_row_gpu_utilization_sampling_errors",
)
_SAMPLE_LEARNER_TRANSPORT_PROOF_REPORT_FIELDS = (
    "compact_owned_loop_sample_learner_worker_bootstrap_source",
    "compact_owned_loop_deferred_sample_learner_request_host_only",
    "compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count",
    "compact_owned_loop_deferred_sample_learner_result_host_only",
    "compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count",
    "compact_owned_loop_deferred_sample_learner_snapshot_host_clone_used",
    "compact_owned_loop_deferred_sample_learner_model_state_interval",
    "compact_owned_loop_deferred_sample_learner_model_state_transport_kind",
    "compact_owned_loop_deferred_sample_learner_replay_append_transport_kind",
    "compact_owned_loop_deferred_sample_learner_model_state_return_count",
    "compact_owned_loop_deferred_sample_learner_model_state_omitted_count",
    "compact_owned_loop_deferred_sample_learner_last_model_state_returned",
    "compact_owned_loop_deferred_sample_learner_model_owner_ref_return_count",
    "compact_owned_loop_deferred_sample_learner_last_model_owner_ref_returned",
    "compact_owned_loop_deferred_sample_learner_last_model_owner_ref_digest",
    "compact_owned_loop_deferred_sample_learner_last_model_owner_ref_worker_pid",
    "compact_owned_loop_deferred_sample_learner_model_state_snapshot_return_count",
    "compact_owned_loop_deferred_sample_learner_model_state_snapshot_publish_bytes",
    "compact_owned_loop_deferred_sample_learner_model_state_snapshot_publish_sec",
    "compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_count",
    "compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_bytes",
    "compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_sec",
    "compact_owned_loop_deferred_sample_learner_request_bytes",
    "compact_owned_loop_deferred_sample_learner_result_bytes",
    "compact_owned_loop_deferred_sample_learner_worker_owns_model_state",
    "compact_owned_loop_deferred_sample_learner_worker_owns_replay_store",
    "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent",
    "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_submit_count",
    "compact_owned_loop_deferred_sample_learner_replay_append_entry_count",
    "compact_owned_loop_deferred_sample_learner_replay_append_index_row_count",
    "compact_owned_loop_deferred_sample_learner_last_replay_append_entry_count",
    "compact_owned_loop_deferred_sample_learner_last_replay_append_index_row_count",
    "compact_owned_loop_deferred_sample_learner_replay_append_entry_bytes",
    "compact_owned_loop_deferred_sample_learner_replay_append_host_observation_bytes",
    "compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_count",
    "compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_bytes",
    "compact_owned_loop_deferred_sample_learner_replay_append_compact_batch_bytes",
    "compact_owned_loop_deferred_sample_learner_replay_append_step_payload_bytes",
    "compact_owned_loop_deferred_sample_learner_replay_append_render_state_bytes",
    "compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_count",
    "compact_owned_loop_deferred_sample_learner_last_provider_bootstrap_step_count",
    "compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_bytes",
    "compact_owned_loop_deferred_sample_learner_provider_bootstrap_host_observation_bytes",
    "compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_count",
    "compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_bytes",
    "compact_owned_loop_deferred_sample_learner_provider_bootstrap_render_state_bytes",
    "compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_entry_count",
    "compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_index_row_count",
    "compact_owned_loop_deferred_sample_learner_provider_bootstrap_learner_call_count",
    "compact_owned_loop_deferred_sample_learner_worker_observation_provider_present",
    "compact_owned_loop_deferred_sample_learner_worker_observation_provider_bootstrap_step_count",
    (
        "compact_owned_loop_deferred_sample_learner_"
        "worker_last_observation_provider_bootstrap_step_count"
    ),
    (
        "compact_owned_loop_deferred_sample_learner_"
        "worker_observation_provider_missing_stack_history_count"
    ),
    (
        "compact_owned_loop_deferred_sample_learner_"
        "worker_observation_provider_materialized_entry_count"
    ),
    (
        "compact_owned_loop_deferred_sample_learner_"
        "worker_last_observation_provider_materialized_entry_count"
    ),
    "compact_owned_loop_deferred_sample_learner_worker_model_initialized_count",
    "compact_owned_loop_deferred_sample_learner_worker_completed_count",
    "compact_owned_loop_deferred_sample_learner_worker_job_wall_sec",
    "compact_owned_loop_deferred_sample_learner_worker_inner_job_wall_sec",
    "compact_owned_loop_deferred_sample_learner_worker_replay_prepare_sec",
    "compact_owned_loop_deferred_sample_learner_worker_sample_sec",
    "compact_owned_loop_deferred_sample_learner_worker_learner_sec",
    "compact_owned_loop_deferred_sample_learner_worker_model_state_prepare_sec",
    "compact_owned_loop_deferred_sample_learner_worker_model_state_fn_sec",
    "compact_owned_loop_deferred_sample_learner_worker_model_state_clone_sec",
    "compact_owned_loop_deferred_sample_learner_worker_model_state_digest_sec",
    "compact_owned_loop_deferred_sample_learner_worker_result_public_sec",
    "compact_owned_loop_deferred_sample_learner_worker_result_pickle_sec",
    "compact_owned_loop_deferred_sample_learner_worker_replay_append_count",
    "compact_owned_loop_deferred_sample_learner_worker_replay_entry_count",
    "compact_owned_loop_deferred_sample_learner_worker_replay_index_row_count",
    "compact_owned_loop_deferred_sample_learner_worker_replay_evicted_entry_count",
    "compact_owned_loop_deferred_sample_learner_worker_replay_evicted_index_row_count",
)
_OWNER_SEARCH_SLAB_PROXY_PROOF_REPORT_FIELDS = (
    "compact_owner_search_slab_proxy",
    "compact_owner_search_lazy_slab_proxy",
    "compact_owner_search_inline_slab_proxy",
    "compact_owner_search_inline_background_slab_proxy",
    "compact_owner_search_threaded_slab_proxy",
    "compact_owner_search_slab_bypass",
    "compact_owner_search_slab_bypass_kind",
    "compact_rollout_slab_bypassed",
    "compact_rollout_slab_general_replay_row_builder_used",
    "compact_rollout_slab_resident_host_observation_stub_requested",
    "compact_rollout_slab_resident_host_observation_stubbed",
    "compact_rollout_slab_resident_host_observation_stub_kind",
    "compact_rollout_slab_resident_host_observation_stub_materialized_bytes",
    "compact_rollout_slab_resident_host_observation_stub_logical_bytes",
    "compact_owner_search_direct_root_build_request_requested",
    "compact_rollout_slab_parent_root_batch_build_avoided",
    "compact_rollout_slab_parent_root_batch_builder_used",
    "compact_rollout_slab_parent_root_batch_builder_call_count",
    "compact_rollout_slab_root_batch_build_sec",
    "compact_rollout_slab_root_build_request_sec",
    "compact_owner_action_step_boundary_enabled",
    "compact_owner_action_step_boundary_proof_passed",
    "compact_owner_action_step_boundary_step_count",
    "compact_owner_action_step_boundary_seeded_action_count",
    "compact_owner_action_step_boundary_feedback_action_count",
    "compact_owner_action_step_boundary_action_verified_count",
    "compact_owner_action_step_boundary_next_action_count",
    "compact_owner_action_step_boundary_last_action_source",
    "compact_owner_action_step_boundary_last_applied_action_checksum",
    "compact_owner_action_step_boundary_last_next_action_checksum",
    "compact_owner_action_step_boundary_failure_reason",
    "compact_owner_mechanics_step_boundary_enabled",
    "compact_owner_mechanics_step_boundary",
    "compact_owner_mechanics_step_view_schema_id",
    "compact_owner_mechanics_step_frame_slot_schema_id",
    "compact_owner_mechanics_step_boundary_count",
    "compact_owner_mechanics_parent_compact_batch_builder_call_count",
    "compact_owner_mechanics_parent_compact_batch_object_count",
    "compact_owner_mechanics_parent_compact_batch_builder_used",
    "compact_owner_mechanics_step_view_object_count",
    "compact_owner_mechanics_host_observation_bytes_sent",
    "compact_owner_mechanics_host_final_observation_bytes_sent",
    "compact_owner_mechanics_resident_observation_handle_present",
    "compact_owner_mechanics_step_frame_handle_schema_id",
    "compact_owner_mechanics_step_frame_handle_ring_used",
    "compact_owner_mechanics_step_frame_handle_published",
    "compact_owner_mechanics_step_frame_handle_consumed",
    "compact_owner_mechanics_step_frame_handle_publish_count",
    "compact_owner_mechanics_step_frame_handle_consume_count",
    "compact_owner_mechanics_step_frame_handle_ring_slot_count",
    "compact_owner_mechanics_step_frame_handle_slot_id",
    "compact_owner_mechanics_step_frame_handle_generation",
    "compact_owner_mechanics_step_frame_handle_digest",
    "compact_owner_mechanics_step_frame_handle_digest_verified",
    "compact_owner_mechanics_step_frame_handle_owner_digest_verified",
    "compact_owner_mechanics_step_frame_handle_resident_observation_present",
    "compact_owner_mechanics_step_frame_slot_write_count",
    "compact_owner_mechanics_parent_step_frame_build_count",
    "compact_owner_action_dispatch_step_overlap_enabled",
    "compact_owner_action_dispatch_step_overlap_proof_passed",
    "compact_rollout_slab_action_dispatch_step_overlap_supported",
    "compact_rollout_slab_action_dispatch_step_overlap_used",
    "compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait",
    "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper",
    "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count",
    "compact_rollout_slab_action_dispatch_step_overlap_submit_count",
    "compact_rollout_slab_action_dispatch_step_overlap_resolve_count",
    "compact_rollout_slab_action_dispatch_step_overlap_pending_count",
    "compact_rollout_slab_action_dispatch_step_overlap_max_pending_count",
    "compact_rollout_slab_action_dispatch_step_overlap_submit_wall_sec",
    "compact_rollout_slab_action_dispatch_step_overlap_resolve_wall_sec",
    "compact_rollout_slab_action_dispatch_step_overlap_submit_to_resolve_elapsed_sec",
    "compact_rollout_slab_action_dispatch_step_overlap_parent_work_sec",
    "compact_owner_search_action_dispatch_handle_boundary_supported",
    "compact_owner_search_action_dispatch_handle_used",
    "compact_owner_search_action_dispatch_handle_schema_id",
    "compact_owner_search_action_dispatch_handle_id",
    "compact_owner_search_action_dispatch_handle_submit_no_wait",
    "compact_owner_search_action_dispatch_handle_sync_wrapper",
    "compact_owner_search_action_dispatch_handle_sync_wrapper_count",
    "compact_owner_search_action_dispatch_handle_completed_at_submit_count",
    "compact_owner_search_action_dispatch_handle_submit_count",
    "compact_owner_search_action_dispatch_handle_resolve_count",
    "compact_owner_search_action_dispatch_handle_pending_count",
    "compact_owner_search_action_dispatch_handle_max_pending_count",
    "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count",
    "compact_owner_search_action_dispatch_handle_result_wait_sec",
    "compact_owner_root_action_context_handle_used",
    "compact_owner_root_action_context_handle_schema_id",
    "compact_owner_root_action_context_handle_id",
    "compact_owner_root_action_context_transaction_id",
    "compact_owner_root_action_context_dispatch_id",
    "compact_owner_root_action_context_root_count",
    "compact_owner_root_action_context_active_root_count",
    "compact_owner_root_action_context_context_digest",
    "compact_owner_root_action_context_owner_store_count",
    "compact_owner_root_action_context_owner_resolve_count",
    "compact_owner_root_action_context_owner_release_count",
    "compact_owner_root_action_context_owner_pending_count",
    "compact_owner_root_action_context_owner_max_pending_count",
    "compact_owner_root_action_context_owner_digest_verified",
    "compact_owner_search_pending_root_action_context_stored",
    "compact_owner_search_action_dispatch_pending_root_action_context_stored",
    "compact_owner_search_action_dispatch_pending_root_action_context_store_count",
    "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count",
    "compact_owner_search_parent_action_context_validation_count",
    "compact_owner_search_owner_action_context_validation_count",
    "compact_owner_search_fixed_action_result_buffer_requested",
    "compact_owner_search_fixed_action_result_buffer_used",
    "compact_owner_search_fixed_action_result_buffer_slot_count",
    "compact_owner_search_fixed_action_result_buffer_acquire_count",
    "compact_owner_search_fixed_action_result_buffer_write_count",
    "compact_owner_search_fixed_action_result_buffer_read_count",
    "compact_owner_search_fixed_action_result_buffer_slot_id",
    "compact_owner_search_fixed_action_result_buffer_last_slot_id",
    "compact_owner_search_fixed_action_result_buffer_wire_result_bytes",
    "compact_owner_search_fixed_action_result_buffer_full_result_bytes",
    "compact_owner_search_fixed_action_result_buffer_pending_slot_count",
    "compact_owner_search_slab_bypass_parent_committed_index_rows",
    "compact_owner_search_slab_bypass_parent_stored_index_rows",
    "compact_owner_search_replay_append_transport_entry_count",
    "compact_owner_search_replay_append_transition_batch_count",
    "compact_owner_search_replay_append_transition_batch_entry_count",
    "compact_owner_search_owner_replay_append_staged_transport_entry_count",
    "compact_owner_search_owner_replay_append_suppressed_transport_entry_count",
    "compact_owner_search_owner_replay_append_submitted_transport_entry_count",
    "compact_owner_search_owner_replay_transport_entry_count",
    "compact_owner_search_owner_replay_transport_kind",
    "compact_owner_search_owner_replay_transition_batch_enabled",
    "compact_owner_search_owner_replay_transition_batch_count",
    "compact_owner_search_owner_replay_transition_batch_transition_count",
    "compact_owner_search_owner_replay_transition_legacy_entry_count",
    "compact_owner_search_transition_batch_transport_requested",
    "compact_owner_search_transition_batch_transport_enabled",
    "compact_owner_search_transition_batch_transport_kind",
    "compact_owner_search_transition_batch_schema_id",
    "compact_owner_search_transition_batch_count",
    "compact_owner_search_transition_batch_entry_count",
    "compact_owner_search_transition_batch_transport_entry_count",
    "compact_owner_search_transition_batch_max_entries_per_batch",
    "compact_owner_search_transition_batch_fixed_capacity",
    "compact_owner_search_transition_batch_padding_count",
    "compact_owner_search_transition_batch_overflow_count",
    "compact_owner_search_transition_batch_fallback_count",
    "compact_owner_search_transition_batch_fallback_reason",
    "compact_owner_search_transition_batch_pending_count",
    "compact_owner_search_transition_batch_transport_bytes",
    "compact_owner_search_transition_batch_digest",
    "compact_owner_search_transition_batch_digest_verified",
    "compact_owner_search_transition_batch_build_sec",
    "compact_owner_search_transition_batch_submit_sec",
    "compact_owner_search_owner_local_transition_derivation_requested",
    "compact_owner_search_owner_local_transition_derivation_used",
    "compact_owner_search_owner_local_transition_derivation_schema_id",
    "compact_owner_search_owner_local_transition_derivation_kind",
    "compact_owner_search_owner_local_transition_derivation_batch_count",
    "compact_owner_search_owner_local_transition_derivation_transition_count",
    "compact_owner_search_owner_local_transition_derivation_transport_entry_count",
    "compact_owner_search_owner_local_transition_derivation_pending_count",
    "compact_owner_search_owner_local_transition_derivation_transport_bytes",
    "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes",
    "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count",
    "compact_owner_search_owner_local_transition_derivation_digest",
    "compact_owner_search_owner_local_transition_derivation_digest_verified",
    "compact_owner_search_owner_local_transition_derivation_build_sec",
    "compact_owner_search_owner_local_transition_derivation_submit_sec",
    "compact_owner_search_owner_local_transition_derivation_cache_hit_count",
    "compact_owner_search_owner_local_transition_derivation_cache_miss_count",
    "compact_owner_search_owner_local_transition_derivation_action_checksum_verified_count",
    "compact_owner_search_owner_local_transition_derivation_action_checksum_mismatch_count",
    "compact_owner_search_owner_local_transition_derivation_fallback_count",
    "compact_owner_search_owner_local_transition_derivation_fallback_reason",
    "compact_owner_search_owner_local_transition_derivation_dropped_pending_count",
    "compact_owner_search_owner_proxy_transition_closure_requested",
    "compact_owner_search_owner_proxy_transition_closure_requested_count",
    "compact_owner_search_owner_proxy_transition_closure_used",
    "compact_owner_search_owner_proxy_transition_closure_used_count",
    "compact_owner_search_owner_proxy_transition_closure_source",
    "compact_owner_search_owner_proxy_transition_closure_no_pending_count",
    "compact_owner_search_owner_proxy_transition_closure_closed_count",
    "compact_owner_search_owner_proxy_transition_closure_batch_count",
    "compact_owner_search_owner_proxy_transition_closure_transition_count",
    "compact_owner_search_owner_proxy_transition_closure_transport_entry_count",
    "compact_owner_search_owner_proxy_transition_closure_pending_count",
    "compact_owner_search_owner_proxy_transition_closure_transport_bytes",
    "compact_owner_search_owner_proxy_transition_closure_digest",
    "compact_owner_search_owner_proxy_transition_closure_digest_verified",
    "compact_owner_search_owner_proxy_transition_closure_build_sec",
    "compact_owner_search_owner_proxy_transition_closure_submit_sec",
    "compact_owner_search_owner_proxy_transition_closure_fallback_count",
    "compact_owner_search_owner_proxy_transition_closure_fallback_reason",
    "compact_owner_search_owner_proxy_applied_action_verification_count",
    "compact_owner_search_owner_proxy_applied_action_mismatch_count",
    "compact_owner_search_owner_proxy_applied_action_count",
    "compact_owner_search_owner_proxy_applied_action_checksum",
    "compact_owner_search_owner_proxy_action_frame_pending",
    "compact_owner_search_owner_proxy_action_frame_store_count",
    "compact_owner_search_parent_previous_transition_closure_count",
    "compact_owner_search_parent_previous_transition_closure_avoided_count",
    "compact_owner_search_parent_applied_action_validation_count",
    "compact_owner_search_direct_transition_batch_replay_requested",
    "compact_owner_search_direct_transition_batch_replay_used",
    "compact_owner_search_direct_transition_batch_replay_batch_count",
    "compact_owner_search_direct_transition_batch_replay_transition_count",
    "compact_owner_search_direct_transition_batch_replay_transport_entry_count",
    "compact_owner_search_direct_transition_batch_replay_legacy_expanded_entry_count",
    "compact_owner_search_direct_transition_batch_replay_index_entry_object_count",
    "compact_owner_search_direct_transition_batch_replay_ring_entry_object_count",
    "compact_owner_search_direct_transition_batch_replay_columnar_append_used",
    "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_requested",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_used",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_count",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_entry_view_object_count",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_step_view_object_count",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_learner_ready_object_count",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_entry_object_count",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_concat_count",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_count",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_reason",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_sec",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_successor_index_sec",
    "compact_owner_search_direct_transition_batch_replay_fixed_soa_total_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_record_count",
    "compact_owner_search_direct_transition_batch_replay_columnar_entry_view_object_count",
    "compact_owner_search_direct_transition_batch_replay_columnar_step_view_object_count",
    "compact_owner_search_direct_transition_batch_replay_fallback_count",
    "compact_owner_search_direct_transition_batch_replay_fallback_reason",
    "compact_owner_search_direct_transition_batch_replay_last_append_sec",
    "compact_owner_search_direct_transition_batch_replay_append_sec",
    "compact_owner_search_direct_transition_batch_replay_accounted_sec",
    "compact_owner_search_direct_transition_batch_replay_array_extract_sec",
    "compact_owner_search_direct_transition_batch_replay_transition_validate_sec",
    "compact_owner_search_direct_transition_batch_replay_device_payload_sec",
    "compact_owner_search_direct_transition_batch_replay_index_rows_build_sec",
    "compact_owner_search_direct_transition_batch_replay_step_object_build_sec",
    "compact_owner_search_direct_transition_batch_replay_ring_append_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_prepare_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_register_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_append_store_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_retain_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_evict_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_evict_release_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_candidate_indices_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_cache_refresh_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_cache_rebuild_sec",
    "compact_owner_search_direct_transition_batch_replay_columnar_total_sec",
    "compact_owner_search_owner_maintenance_drained_replay_append_transport_entry_count",
    "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_count",
    "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_entry_count",
    "compact_owner_search_slab_proxy_initialized",
    "compact_owner_search_boundary_kind",
    "compact_owner_search_parent_slab_commits_replay",
    "compact_owner_search_worker_kind",
    "compact_owner_search_worker_resource_scope",
    "compact_owner_search_worker_resource_distinct_from_actor",
    "compact_owner_search_worker_hardware_resource_distinct_from_actor",
    "compact_owner_search_owner_pid",
    "compact_owner_search_root_slot_count",
    "compact_owner_search_active_root_count",
    "compact_owner_search_request_bytes",
    "compact_owner_search_result_bytes",
    "compact_owner_search_request_cuda_tensor_count",
    "compact_owner_search_result_cuda_tensor_count",
    "compact_owner_search_root_observation_bytes_sent",
    "compact_owner_search_parent_reconstructed_search_result",
    "compact_owner_search_action_only_result",
    "compact_owner_search_owner_materializes_replay",
    "compact_owner_search_action_feedback_verified",
    "compact_owner_search_action_feedback_transition_count",
    "compact_owner_search_action_feedback_action_count",
    "compact_owner_search_action_feedback_mismatch_count",
    "compact_owner_search_expected_joint_action_checksum",
    "compact_owner_search_applied_joint_action_checksum",
    "compact_owner_search_replay_action_checksum",
    "compact_owner_search_inner_two_phase_action_step",
    "compact_owner_search_inner_device_replay_payload_deferred",
    "compact_owner_search_use_inner_two_phase_device_replay",
    "compact_owner_search_replay_payload_handle_present",
    "compact_owner_search_model_state_bytes",
    "compact_owner_search_model_state_return_count",
    "compact_owner_search_model_state_snapshot_return_count",
    "compact_owner_search_model_state_snapshot_load_count",
    "compact_owner_search_model_state_snapshot_load_bytes",
    "compact_owner_search_model_state_snapshot_load_sec",
    "compact_owner_search_search_result_payload_bytes",
    "compact_owner_search_search_result_payload_transport_kind",
    "compact_owner_search_search_result_payload_json_safe",
    "compact_owner_search_selected_action_bytes",
    "compact_owner_search_visit_policy_bytes",
    "compact_owner_search_root_value_bytes",
    "compact_owner_search_optional_array_bytes",
    "compact_owner_search_worker_owns_search_state",
    "compact_owner_search_worker_owns_replay_state",
    "compact_owner_search_worker_owns_model_state",
    "compact_owner_search_consumed_learner_update",
    "compact_owner_search_search_refresh_update_count",
    "compact_owner_search_replay_append_entry_count",
    "compact_owner_search_replay_append_count",
    "compact_owner_search_learner_update_count",
    "compact_owner_search_model_owner_ref_returned",
    "compact_owner_search_model_owner_ref_digest",
    "compact_owner_search_owner_replay_append_enabled",
    "compact_owner_search_owner_learning_enabled",
    "compact_owner_search_owner_sample_batch_size",
    "compact_owner_search_owner_train_steps",
    "compact_owner_search_owner_train_interval",
    "compact_owner_search_owner_defer_maintenance",
    "compact_owner_search_owner_loop_schema_id",
    "compact_owner_search_owner_loop_kind",
    "compact_owner_search_owner_loop_persistent",
    "compact_owner_search_owner_action_priority_enabled",
    "compact_owner_search_owner_background_maintenance_thread",
    "compact_owner_search_owner_background_overlap_enabled",
    "compact_owner_search_owner_action_request_count",
    "compact_owner_search_owner_maintenance_request_count",
    "compact_owner_search_owner_run_request_count",
    "compact_owner_search_owner_sample_telemetry",
    "compact_owner_search_owner_learner_telemetry",
    "compact_owner_search_owner_replay_append_staged_entry_count",
    "compact_owner_search_owner_replay_append_suppressed_entry_count",
    "compact_owner_search_owner_replay_append_submitted_entry_count",
    "compact_owner_search_owner_replay_append_request_count",
    "compact_owner_search_owner_replay_append_count",
    "compact_owner_search_owner_train_request_count",
    "compact_owner_search_owner_submitted_learner_update_count",
    "compact_owner_search_owner_learner_update_count",
    "compact_owner_search_owner_pending_replay_append_entry_count",
    "compact_owner_search_owner_maintenance_drain_request_count",
    "compact_owner_search_owner_maintenance_staged_work_item_count",
    "compact_owner_search_owner_maintenance_drained_count",
    "compact_owner_search_owner_maintenance_drained_work_item_count",
    "compact_owner_search_owner_maintenance_drained_replay_append_entry_count",
    "compact_owner_search_owner_maintenance_drained_replay_append_count",
    "compact_owner_search_owner_maintenance_pending_work_count",
    "compact_owner_search_owner_maintenance_inflight",
    "compact_owner_search_owner_maintenance_final_drain_sec",
    "compact_owner_search_owner_maintenance_final_drain_in_measured_sec",
    "compact_owner_search_owner_maintenance_coalescing_kind",
    "compact_owner_search_owner_maintenance_coalesced_skip_count",
    "compact_owner_search_owner_maintenance_eager_append_drain_count",
    "compact_owner_search_owner_async_learner_worker_enabled",
    "compact_owner_search_owner_async_learner_worker_kind",
    "compact_owner_search_owner_async_learner_worker_resource_scope",
    "compact_owner_search_owner_async_learner_worker_resource_id",
    "compact_owner_search_owner_async_learner_actor_resource_id",
    "compact_owner_search_owner_async_learner_worker_parent_pid",
    "compact_owner_search_owner_async_learner_resource_distinct_from_owner",
    ("compact_owner_search_owner_async_learner_hardware_resource_distinct_from_owner"),
    "compact_owner_search_owner_async_learner_max_pending",
    "compact_owner_search_owner_async_learner_submit_count",
    "compact_owner_search_owner_async_learner_completed_count",
    "compact_owner_search_owner_async_learner_pending_count",
    "compact_owner_search_owner_async_learner_max_pending_observed",
    "compact_owner_search_owner_async_learner_wait_count",
    "compact_owner_search_owner_async_learner_wait_sec",
    "compact_owner_search_owner_action_while_async_learner_pending_count",
    "compact_owner_search_owner_async_learner_failed",
    "compact_owner_search_owner_async_learner_request_host_only",
    "compact_owner_search_owner_async_learner_request_cuda_tensor_count",
    "compact_owner_search_owner_async_learner_result_host_only",
    "compact_owner_search_owner_async_learner_result_cuda_tensor_count",
    "compact_owner_search_owner_async_learner_request_bytes",
    "compact_owner_search_owner_async_learner_result_bytes",
    "compact_owner_search_owner_async_learner_worker_pid",
    "compact_owner_search_owner_async_learner_worker_job_wall_sec",
    "compact_owner_search_owner_async_learner_payload_prepare_sec",
    ("compact_owner_search_owner_async_learner_worker_pid_distinct_from_owner"),
    "compact_owner_search_owner_async_learner_worker_owns_model_state",
    "compact_owner_search_owner_policy_lag_current",
    "compact_owner_search_owner_policy_lag_max",
    "compact_owner_search_owner_maintenance_actor_steps_while_pending",
    "compact_owner_search_owner_maintenance_actor_steps_while_policy_lagged",
    "compact_owner_search_owner_action_while_maintenance_pending_count",
    "compact_owner_search_owner_action_while_policy_lagged_count",
    "compact_owner_search_owner_action_served_before_maintenance_count",
    "compact_owner_search_owner_fifo_blocked_action_count",
    "compact_owner_search_owner_maintenance_failed",
    "compact_owner_search_parent_publish_sec",
    "compact_owner_search_parent_submit_sec",
    "compact_owner_search_parent_wait_sec",
    "compact_owner_search_parent_wall_sec",
    "compact_owner_search_worker_wall_sec",
    "compact_owner_search_worker_root_resolve_sec",
    "compact_owner_search_worker_search_sec",
    "compact_owner_search_worker_replay_append_sec",
    "compact_owner_search_worker_learner_train_sec",
    "compact_owner_search_owner_train_wall_sec",
    "compact_owner_search_owner_train_sample_sec",
    "compact_owner_search_owner_train_payload_host_clone_sec",
    "compact_owner_search_owner_train_payload_device_move_sec",
    "compact_owner_search_owner_train_learner_update_sec",
    "compact_owner_search_owner_train_model_state_digest_sec",
    "compact_owner_search_owner_train_model_state_digest_deferred_to_refresh",
    "compact_owner_search_owner_train_model_state_dict_sec",
    "compact_owner_search_owner_train_owner_ref_build_sec",
    "compact_owner_search_owner_train_model_state_snapshot_returned",
    "compact_owner_search_owner_train_model_state_snapshot_bytes",
    "compact_owner_search_owner_train_model_state_snapshot_write_sec",
    "compact_owner_search_owner_train_accounted_sec",
    "compact_owner_search_owner_train_residual_sec",
    "compact_owner_search_owner_train_timing_aggregate_count",
    "compact_owner_search_worker_search_refresh_sec",
    "compact_owner_search_resident_root_bridge_ready",
    "compact_owner_search_resident_root_bridge_kind",
    "compact_owner_search_resident_root_bridge_device",
    "compact_owner_search_resident_root_bridge_h2d_bytes",
    "compact_owner_search_resident_root_bridge_host_observation_copied",
    "compact_owner_search_resident_root_bridge_generation_id",
    "compact_owner_search_resident_root_bridge_final_storage",
    "compact_owner_search_resident_root_bridge_final_sparse_row_count",
    "compact_owner_search_resident_root_bridge_final_sparse_bytes",
    "compact_owner_search_resident_root_bridge_final_dense_clone_avoided_bytes",
    "compact_direct_root_store",
    "compact_direct_root_store_publish_count",
    "compact_direct_root_store_resolve_count",
    "compact_direct_root_store_last_root_slot_count",
    "compact_owner_search_direct_root_handoff",
    "compact_owner_search_direct_root_rebuild_avoided",
    "compact_owner_search_direct_root_resolved",
    "compact_owner_search_direct_root_observation_bytes_sent",
    "compact_owner_search_direct_root_build_request_handoff",
    "compact_owner_step_frame_root_build_request_used",
    "compact_owner_step_frame_root_build_request_from_batch_helper_used",
    "compact_owner_step_frame_root_request_sidecar_array_bytes",
    "compact_owner_step_frame_root_request_sidecar_field_count",
    "compact_owner_root_search_transaction_boundary_supported",
    "compact_owner_root_search_transaction_requested",
    "compact_owner_root_search_transaction_used",
    "compact_owner_root_search_transaction_schema_id",
    "compact_owner_root_search_transaction_id",
    "compact_owner_root_search_transaction_begin_count",
    "compact_owner_root_search_transaction_submit_count",
    "compact_owner_root_search_transaction_resolve_count",
    "compact_owner_root_search_transaction_pending_count",
    "compact_owner_root_search_transaction_max_pending_count",
    "compact_owner_root_search_transaction_parent_root_request_build_count",
    "compact_owner_root_search_transaction_parent_root_request_stored",
    "compact_owner_root_search_transaction_parent_compact_batch_stored",
    "compact_owner_root_search_transaction_parent_rebuild_count",
    "compact_owner_root_search_transaction_parent_root_action_context_stored",
    "compact_owner_root_search_transaction_parent_root_action_context_store_count",
    "compact_owner_root_search_transaction_parent_root_action_context_array_bytes",
    "compact_owner_root_search_transaction_parent_root_action_context_field_count",
    "compact_owner_root_search_transaction_owner_root_request_build_count",
    "compact_owner_root_search_transaction_owner_root_request_build_sec",
    "compact_owner_root_search_transaction_owner_root_store_publish_count",
    "compact_owner_root_search_transaction_frame_generation_verified",
    "compact_owner_root_search_transaction_frame_digest_verified",
    "compact_owner_root_search_transaction_action_identity_verified",
    "compact_owner_root_search_transaction_proxy_transition_closure_used",
    "compact_owner_root_search_transaction_applied_action_mismatch_count",
    "compact_owner_search_direct_root_build_request_schema_id",
    "compact_owner_search_direct_root_build_request_kind",
    "compact_owner_search_direct_root_build_request_publish_count",
    "compact_owner_search_direct_root_build_request_resolve_count",
    "compact_owner_search_direct_root_build_request_root_count",
    "compact_owner_search_direct_root_build_request_active_root_count",
    "compact_owner_search_direct_root_build_request_observation_included",
    "compact_owner_search_direct_root_build_request_observation_bytes_sent",
    "compact_owner_search_direct_root_build_request_resident_handle_present",
    "compact_owner_search_direct_root_parent_build_avoided",
    "compact_owner_search_direct_root_parent_build_call_count",
    "compact_owner_search_direct_root_parent_build_sec",
    "compact_owner_search_direct_root_build_request_sec",
    "compact_owner_search_direct_root_owner_build_used",
    "compact_owner_search_direct_root_owner_build_count",
    "compact_owner_search_direct_root_owner_build_sec",
    "compact_owner_search_parent_compact_root_batch_objects_sent",
    "compact_owner_search_root_build_request_host_observation_bytes_sent",
    "compact_owner_search_resident_root_view_required",
    "compact_owner_search_resident_root_view_proved",
    "compact_owner_search_resident_root_view_kind",
    "compact_owner_search_resident_root_view_generation_id",
    "compact_owner_search_resident_root_view_fresh_for_step_index",
    "compact_owner_search_resident_root_view_device",
    "compact_owner_search_resident_root_view_source_backend",
    "compact_owner_search_resident_root_view_root_shape",
    "compact_owner_search_resident_root_view_stack_shape",
    "compact_owner_search_resident_root_view_h2d_bytes",
    "compact_owner_search_resident_root_view_d2h_bytes",
    "compact_owner_search_resident_root_view_host_fallback_allowed",
    "compact_owner_search_resident_root_view_row_major_order",
    "compact_rollout_slab_committed_index_row_count",
    "compact_rollout_slab_stored_index_row_count",
)


def _uses_compact_torch_search_service(args: argparse.Namespace) -> bool:
    search_kind = str(args.search_service_kind)
    if search_kind == "compact_torch_search_service":
        return True
    return (
        search_kind
        in {
            "owner_search_slab_proxy",
            "owner_search_inline_proxy",
            "owner_search_inline_background_proxy",
            "owner_search_threaded_proxy",
        }
        and str(args.owner_search_inner_search_service_kind) == "compact_torch_search_service"
    )


def _owner_search_config_fields(args: argparse.Namespace) -> dict[str, Any]:
    owner_search = str(args.search_service_kind) in {
        "owner_search_slab_proxy",
        "owner_search_inline_proxy",
        "owner_search_inline_background_proxy",
        "owner_search_threaded_proxy",
    }
    inline_owner_search = str(args.search_service_kind) == "owner_search_inline_proxy"
    inline_background_owner_search = (
        str(args.search_service_kind) == "owner_search_inline_background_proxy"
    )
    threaded_owner_search = str(args.search_service_kind) == "owner_search_threaded_proxy"
    inner_kind = str(args.owner_search_inner_search_service_kind)
    return {
        "owner_search_slab_proxy_requested": owner_search,
        "owner_search_inline_proxy_requested": inline_owner_search,
        "owner_search_inline_background_proxy_requested": (inline_background_owner_search),
        "owner_search_threaded_proxy_requested": threaded_owner_search,
        "owner_search_inner_search_service_kind": inner_kind if owner_search else "",
        "owner_search_compact_torch_resident_root_bridge_ready": bool(
            owner_search
            and not inline_owner_search
            and not inline_background_owner_search
            and not threaded_owner_search
            and inner_kind == "compact_torch_search_service"
        ),
        "owner_search_defer_maintenance_requested": bool(
            owner_search and getattr(args, "owner_search_defer_maintenance", False)
        ),
        "owner_search_slab_bypass_requested": bool(
            owner_search and getattr(args, "owner_search_slab_bypass", False)
        ),
        "owner_search_transition_batch_size_requested": int(
            getattr(args, "owner_search_transition_batch_size", 1)
        )
        if owner_search
        else 1,
        "owner_search_transition_batch_transport_requested": bool(
            owner_search
            and getattr(args, "owner_search_slab_bypass", False)
            and int(getattr(args, "owner_search_transition_batch_size", 1)) > 1
        ),
        "owner_search_direct_transition_batch_replay_requested": bool(
            owner_search and getattr(args, "owner_search_direct_transition_batch_replay", False)
        ),
        "owner_search_owner_local_transition_derivation_requested": bool(
            owner_search and getattr(args, "owner_search_owner_local_transition_derivation", False)
        ),
        "owner_search_owner_proxy_transition_closure_requested": bool(
            owner_search and getattr(args, "owner_search_owner_proxy_transition_closure", False)
        ),
        "owner_search_require_resident_root_view_requested": bool(
            owner_search and getattr(args, "owner_search_require_resident_root_view", False)
        ),
        "owner_search_resident_root_host_observation_stub_requested": bool(
            owner_search
            and getattr(
                args,
                "owner_search_resident_root_host_observation_stub",
                False,
            )
        ),
        "owner_search_direct_root_build_request_requested": bool(
            owner_search and getattr(args, "owner_search_direct_root_build_request", False)
        ),
        "compact_owner_action_step_boundary_requested": bool(
            owner_search and getattr(args, "compact_owner_action_step_boundary", False)
        ),
        "compact_owner_action_dispatch_step_overlap_requested": bool(
            owner_search
            and getattr(args, "compact_owner_action_dispatch_step_overlap", False)
        ),
        "owner_search_fixed_action_result_buffer_requested": bool(
            owner_search and getattr(args, "owner_search_fixed_action_result_buffer", False)
        ),
        "owner_search_action_result_slot_capacity_requested": int(
            getattr(args, "owner_search_action_result_slot_capacity", 4) or 4
        )
        if owner_search
        else 0,
        "owner_search_fixed_soa_replay_requested": bool(
            owner_search and getattr(args, "owner_search_fixed_soa_replay", False)
        ),
        "owner_search_fixed_soa_locality_sample_group_size_requested": int(
            getattr(args, "owner_search_fixed_soa_locality_sample_group_size", 1) or 1
        ),
        "owner_search_defer_model_state_digest_to_refresh_requested": bool(
            owner_search
            and getattr(args, "owner_search_defer_model_state_digest_to_refresh", False)
        ),
        "owner_search_async_learner_worker_requested": bool(
            owner_search and getattr(args, "owner_search_async_learner_worker", False)
        ),
        "owner_search_async_learner_worker_kind_requested": (
            str(
                getattr(
                    args,
                    "owner_search_async_learner_worker_kind",
                    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
                )
            )
            if owner_search
            else ""
        ),
        "owner_search_async_learner_max_pending_requested": (
            int(getattr(args, "owner_search_async_learner_max_pending", 1)) if owner_search else 0
        ),
    }


def _accepted_fast_path_step_window_name(args: argparse.Namespace) -> str:
    return str(
        getattr(
            args,
            "compact_owned_accepted_fast_path_step_window",
            ACCEPTED_FAST_PATH_STEP_WINDOW_ACCEPTED,
        )
        or ACCEPTED_FAST_PATH_STEP_WINDOW_ACCEPTED
    )


def _accepted_fast_path_step_window(args: argparse.Namespace) -> dict[str, Any]:
    name = _accepted_fast_path_step_window_name(args)
    if name not in ACCEPTED_FAST_PATH_STEP_WINDOWS:
        raise ValueError(
            "--compact-owned-accepted-fast-path-step-window must be one of "
            f"{sorted(ACCEPTED_FAST_PATH_STEP_WINDOWS)}"
        )
    return ACCEPTED_FAST_PATH_STEP_WINDOWS[name]


def _validate_accepted_fast_path_step_window_args(args: argparse.Namespace) -> None:
    window_name = _accepted_fast_path_step_window_name(args)
    if window_name != ACCEPTED_FAST_PATH_STEP_WINDOW_ACCEPTED and not bool(
        getattr(args, "compact_owned_accepted_fast_path_preset", False)
    ):
        raise ValueError(
            "--compact-owned-accepted-fast-path-step-window stability diagnostics "
            "require --compact-owned-accepted-fast-path-preset"
        )
    if not bool(getattr(args, "compact_owned_accepted_fast_path_preset", False)):
        return
    conflicts: list[str] = []
    if str(getattr(args, "search_service_kind", "")) in {
        "owner_search_slab_proxy",
        "owner_search_inline_proxy",
        "owner_search_inline_background_proxy",
        "owner_search_threaded_proxy",
    }:
        conflicts.append("--search-service-kind")
    owner_search_checks = (
        ("--owner-search-defer-maintenance", "owner_search_defer_maintenance", False),
        ("--owner-search-slab-bypass", "owner_search_slab_bypass", False),
        (
            "--owner-search-direct-transition-batch-replay",
            "owner_search_direct_transition_batch_replay",
            False,
        ),
        (
            "--owner-search-owner-local-transition-derivation",
            "owner_search_owner_local_transition_derivation",
            False,
        ),
        (
            "--owner-search-owner-proxy-transition-closure",
            "owner_search_owner_proxy_transition_closure",
            False,
        ),
        (
            "--owner-search-require-resident-root-view",
            "owner_search_require_resident_root_view",
            False,
        ),
        (
            "--owner-search-resident-root-host-observation-stub",
            "owner_search_resident_root_host_observation_stub",
            False,
        ),
        (
            "--owner-search-direct-root-build-request",
            "owner_search_direct_root_build_request",
            False,
        ),
        (
            "--compact-owner-action-step-boundary",
            "compact_owner_action_step_boundary",
            False,
        ),
        (
            "--owner-search-fixed-action-result-buffer",
            "owner_search_fixed_action_result_buffer",
            False,
        ),
        (
            "--owner-search-action-result-slot-capacity",
            "owner_search_action_result_slot_capacity",
            4,
        ),
        ("--owner-search-fixed-soa-replay", "owner_search_fixed_soa_replay", False),
        (
            "--owner-search-fixed-soa-locality-sample-group-size",
            "owner_search_fixed_soa_locality_sample_group_size",
            1,
        ),
        (
            "--owner-search-defer-model-state-digest-to-refresh",
            "owner_search_defer_model_state_digest_to_refresh",
            False,
        ),
        ("--owner-search-async-learner-worker", "owner_search_async_learner_worker", False),
    )
    for flag, attr, expected in owner_search_checks:
        value = getattr(args, attr, expected)
        if isinstance(expected, bool):
            mismatched = bool(value) != expected
        else:
            mismatched = int(value) != int(expected)
        if mismatched:
            conflicts.append(flag)
    if int(getattr(args, "owner_search_transition_batch_size", 1)) != 1:
        conflicts.append("--owner-search-transition-batch-size")
    if conflicts:
        raise ValueError(
            "--compact-owned-accepted-fast-path-preset cannot be combined with "
            "owner-search override flags because the preset expands to the plain "
            f"compact_torch_search_service baseline: {', '.join(conflicts)}"
        )


def _accepted_fast_path_step_window_report_fields(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "compact_owned_accepted_fast_path_preset", False)):
        return {
            "compact_owned_accepted_fast_path_step_window": "",
            "compact_owned_accepted_fast_path_stability_diagnostic": False,
            "speed_row_comparison_role": "unclassified_speed_row",
        }
    window_name = _accepted_fast_path_step_window_name(args)
    window = _accepted_fast_path_step_window(args)
    return {
        "compact_owned_accepted_fast_path_step_window": window_name,
        "compact_owned_accepted_fast_path_stability_diagnostic": bool(
            window["stability_diagnostic"]
        ),
        "speed_row_comparison_role": str(window["comparison_role"]),
    }


def _accepted_fast_path_step_window_from_summary(summary: dict[str, Any]) -> str:
    try:
        steps = int(summary.get("steps"))
        warmup_steps = int(summary.get("warmup_steps"))
    except (TypeError, ValueError):
        return ACCEPTED_FAST_PATH_STEP_WINDOW_ACCEPTED
    for name, window in ACCEPTED_FAST_PATH_STEP_WINDOWS.items():
        if steps == int(window["steps"]) and warmup_steps == int(window["warmup_steps"]):
            return name
    return ACCEPTED_FAST_PATH_STEP_WINDOW_ACCEPTED


def _apply_accepted_fast_path_preset(args: argparse.Namespace) -> None:
    _validate_accepted_fast_path_step_window_args(args)
    if not bool(getattr(args, "compact_owned_accepted_fast_path_preset", False)):
        return
    window = _accepted_fast_path_step_window(args)
    args.batch_size = 1024
    args.actor_count = 1
    args.steps = int(window["steps"])
    args.warmup_steps = int(window["warmup_steps"])
    args.death_mode = vector_runtime.DEATH_MODE_NORMAL
    args.sample_batch_size = 512
    args.sample_interval = 8
    args.replay_pair_capacity = 4096
    args.learner_train_steps = 1
    args.learner_num_unroll_steps = 2
    args.policy_refresh_interval = 4
    args.compact_owned_loop_deferred_learner = False
    args.compact_owned_loop_deferred_sample_learner = False
    args.compact_owned_loop_fused_learner_batch = True
    args.compact_owned_lean_trainer_step = True
    args.compact_owned_lean_profile_oracle = False
    args.hybrid_persistent_compact_render_state_buffer = False
    args.hybrid_borrow_single_actor_render_state = True
    args.learner_device = "cuda"
    args.num_simulations = 1
    args.search_service_kind = "compact_torch_search_service"
    args.owner_search_inner_search_service_kind = "compact_torch_search_service"
    args.owner_search_defer_maintenance = False
    args.owner_search_slab_bypass = False
    args.owner_search_transition_batch_size = 1
    args.owner_search_direct_transition_batch_replay = False
    args.owner_search_owner_local_transition_derivation = False
    args.owner_search_owner_proxy_transition_closure = False
    args.owner_search_require_resident_root_view = False
    args.owner_search_resident_root_host_observation_stub = False
    args.owner_search_direct_root_build_request = False
    args.compact_owner_action_step_boundary = False
    args.compact_owner_action_dispatch_step_overlap = False
    args.owner_search_fixed_action_result_buffer = False
    args.owner_search_action_result_slot_capacity = 4
    args.owner_search_async_learner_worker = False
    args.owner_search_async_learner_worker_kind = (
        COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
    )
    args.owner_search_async_learner_max_pending = 1
    args.compact_torch_request_compile = False
    args.compact_torch_request_model_compile = False
    args.compact_torch_timing_mode = "host_phase_sync"
    args.compact_torch_initial_inference_mode = "direct_core"
    args.compact_torch_observation_memory_format = "contiguous"
    args.compact_torch_model_memory_format = "contiguous"
    args.compact_torch_defer_one_simulation_replay_payload = False
    if bool(window["stability_diagnostic"]):
        args.compact_profile_bounded_diagnostics = True


def _accepted_fast_path_preset_violations(
    args: argparse.Namespace,
    result: dict[str, Any],
) -> list[str]:
    if not bool(getattr(args, "compact_owned_accepted_fast_path_preset", False)):
        return []
    window = _accepted_fast_path_step_window(args)
    expected_steps = int(window["steps"])
    expected_warmup_steps = int(window["warmup_steps"])
    stability_diagnostic = bool(window["stability_diagnostic"])
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return ["result.summary must be a JSON object"]
    violations: list[str] = []
    for label, field, expected in (
        ("summary.steps", "steps", expected_steps),
        ("summary.warmup_steps", "warmup_steps", expected_warmup_steps),
    ):
        actual = summary.get(field)
        try:
            matches = actual is not None and int(actual) == expected
        except (TypeError, ValueError):
            matches = False
        if not matches:
            violations.append(f"{label}: expected {expected!r}, got {actual!r}")
    for label, field, expected in _ACCEPTED_FAST_PATH_RESULT_REQUIREMENTS:
        actual = summary.get(field)
        if isinstance(expected, bool):
            matches = actual is expected
        elif isinstance(expected, int):
            matches = actual is not None and int(actual) == expected
        else:
            matches = str(actual) == str(expected)
        if not matches:
            violations.append(f"{label}: expected {expected!r}, got {actual!r}")
    try:
        batch_size = int(summary.get("batch_size"))
        env_steps_collected = float(summary.get("env_steps_collected"))
        expected_env_steps_collected = float(batch_size * expected_steps)
    except (TypeError, ValueError):
        env_steps_collected = -1.0
        expected_env_steps_collected = -2.0
    if env_steps_collected != expected_env_steps_collected:
        violations.append(
            "summary.env_steps_collected: "
            f"expected {expected_env_steps_collected!r}, got "
            f"{summary.get('env_steps_collected')!r}"
        )
    try:
        borrowed_steps = int(summary.get("render_state_borrowed_steps"))
    except (TypeError, ValueError):
        borrowed_steps = -1
    expected_borrowed_steps = expected_steps + expected_warmup_steps
    if borrowed_steps != expected_borrowed_steps:
        violations.append(
            "summary.render_state_borrowed_steps: "
            f"expected {expected_borrowed_steps!r}, got {summary.get('render_state_borrowed_steps')!r}"
        )
    terminal_sample_rows = summary.get("terminal_sample_row_count")
    terminal_target_rows = summary.get("terminal_unroll_value_target_row_count")
    if stability_diagnostic:
        try:
            terminal_sample_count = int(terminal_sample_rows)
            terminal_target_count = int(terminal_target_rows)
        except (TypeError, ValueError):
            terminal_sample_count = -1
            terminal_target_count = -2
        if terminal_sample_count <= 0:
            violations.append(
                "summary.terminal_sample_row_count: required long-window terminal count "
                f"must be positive, got {terminal_sample_rows!r}"
            )
        if terminal_target_count != terminal_sample_count:
            violations.append(
                "summary.terminal_unroll_value_target_row_count: expected to match "
                f"terminal_sample_row_count {terminal_sample_count!r}, got {terminal_target_rows!r}"
            )
    else:
        for label, actual in (
            ("summary.terminal_sample_row_count", terminal_sample_rows),
            ("summary.terminal_unroll_value_target_row_count", terminal_target_rows),
        ):
            try:
                matches = int(actual) == 167
            except (TypeError, ValueError):
                matches = False
            if not matches:
                violations.append(f"{label}: expected 167, got {actual!r}")
    for field in _ACCEPTED_FAST_PATH_REPEATABILITY_REQUIRED_FIELDS:
        if summary.get(field) is None:
            violations.append(f"summary.{field}: required repeatability field missing")
    for field in _ACCEPTED_FAST_PATH_REPEATABILITY_NONZERO_FIELDS:
        try:
            value = float(summary.get(field) or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value == 0:
            violations.append(f"summary.{field}: required repeatability checksum must be nonzero")
    for field in _ACCEPTED_FAST_PATH_REPEATABILITY_POSITIVE_FIELDS:
        try:
            value = float(summary.get(field) or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0:
            violations.append(f"summary.{field}: required repeatability field must be positive")
    digest = str(
        summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest"
        )
        or ""
    )
    if not digest:
        violations.append(
            "summary.compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest: "
            "required repeatability digest missing"
        )
    return violations


def _cuda_sync_timing_diagnostic_violations(
    args: argparse.Namespace,
    result: dict[str, Any],
) -> list[str]:
    if not bool(getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)):
        return []
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return ["result.summary must be a JSON object"]
    violations: list[str] = []
    for field in _CUDA_SYNC_TIMING_DIAGNOSTIC_TRUE_FIELDS:
        if summary.get(field) is not True:
            violations.append(f"summary.{field}: expected True, got {summary.get(field)!r}")
    for field in _CUDA_SYNC_TIMING_DIAGNOSTIC_POSITIVE_FIELDS:
        try:
            value = int(summary.get(field))
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            violations.append(
                f"summary.{field}: expected positive sync count, got {summary.get(field)!r}"
            )
    for field in _CUDA_SYNC_TIMING_DIAGNOSTIC_SEC_FIELDS:
        raw_value = summary.get(field)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            violations.append(
                f"summary.{field}: expected finite nonnegative seconds, got {raw_value!r}"
            )
            continue
        if not math.isfinite(value) or value < 0.0:
            violations.append(
                f"summary.{field}: expected finite nonnegative seconds, got {raw_value!r}"
            )
    return violations


def _runtime_step_timing_diagnostic_violations(
    args: argparse.Namespace,
    result: dict[str, Any],
) -> list[str]:
    requested = bool(
        getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
        or getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
    )
    if not requested:
        return []
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return ["result.summary must be a JSON object"]
    violations: list[str] = []
    for field in _RUNTIME_STEP_TIMING_DIAGNOSTIC_TRUE_FIELDS:
        if summary.get(field) is not True:
            violations.append(f"summary.{field}: expected True, got {summary.get(field)!r}")
    for field in _RUNTIME_STEP_TIMING_DIAGNOSTIC_POSITIVE_FIELDS:
        try:
            value = int(summary.get(field))
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            violations.append(
                f"summary.{field}: expected positive runtime-step count, got {summary.get(field)!r}"
            )
    for field in _RUNTIME_STEP_TIMING_DIAGNOSTIC_SEC_FIELDS:
        raw_value = summary.get(field)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            violations.append(
                f"summary.{field}: expected finite nonnegative seconds, got {raw_value!r}"
            )
            continue
        if not math.isfinite(value) or value < 0.0:
            violations.append(
                f"summary.{field}: expected finite nonnegative seconds, got {raw_value!r}"
            )
    return violations


def _unroll2_specialized_builder_proof_report_fields(
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        field: summary.get(field)
        for field in _UNROLL2_SPECIALIZED_BUILDER_PROOF_REPORT_FIELDS
        if field in summary
    }


def _learner_ready_unroll2_cache_proof_report_fields(
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        field: summary.get(field)
        for field in _LEARNER_READY_UNROLL2_CACHE_PROOF_REPORT_FIELDS
        if field in summary
    }


def _tensor_native_replay_proof_report_fields(
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        field: summary.get(field)
        for field in _TENSOR_NATIVE_REPLAY_PROOF_REPORT_FIELDS
        if field in summary
    }


def _whole_owner_buffer_replay_ceiling_report_fields(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(field): value
        for field, value in summary.items()
        if str(field).startswith("compact_whole_owner_buffer_replay_ceiling_")
    }


def _unroll2_specialized_builder_violations(
    args: argparse.Namespace,
    result: dict[str, Any],
) -> list[str]:
    if not bool(getattr(args, "compact_muzero_learner_batch_unroll2_specialized_builder", False)):
        return []
    if bool(getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)):
        return []
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return ["result.summary must be a JSON object"]
    violations: list[str] = []
    if summary.get(_UNROLL2_SPECIALIZED_BUILDER_KEY) is not True:
        violations.append(
            f"summary.{_UNROLL2_SPECIALIZED_BUILDER_KEY}: expected True, "
            f"got {summary.get(_UNROLL2_SPECIALIZED_BUILDER_KEY)!r}"
        )
    try:
        learner_num_unroll_steps = int(summary.get("learner_num_unroll_steps"))
    except (TypeError, ValueError):
        learner_num_unroll_steps = 0
    if learner_num_unroll_steps != 2:
        violations.append(
            "summary.learner_num_unroll_steps: expected 2, "
            f"got {summary.get('learner_num_unroll_steps')!r}"
        )
    if summary.get(_UNROLL2_SPECIALIZED_BUILDER_REQUESTED_FIELD) is not True:
        violations.append(
            f"summary.{_UNROLL2_SPECIALIZED_BUILDER_REQUESTED_FIELD}: "
            f"expected True, got {summary.get(_UNROLL2_SPECIALIZED_BUILDER_REQUESTED_FIELD)!r}"
        )
    if summary.get(_UNROLL2_SPECIALIZED_BUILDER_USED_FIELD) is not True:
        violations.append(
            f"summary.{_UNROLL2_SPECIALIZED_BUILDER_USED_FIELD}: "
            f"expected True, got {summary.get(_UNROLL2_SPECIALIZED_BUILDER_USED_FIELD)!r}"
        )
    try:
        call_count = int(summary.get(_UNROLL2_SPECIALIZED_BUILDER_CALL_COUNT_FIELD))
    except (TypeError, ValueError):
        call_count = 0
    if call_count <= 0:
        violations.append(
            f"summary.{_UNROLL2_SPECIALIZED_BUILDER_CALL_COUNT_FIELD}: "
            f"expected positive call count, got {summary.get(_UNROLL2_SPECIALIZED_BUILDER_CALL_COUNT_FIELD)!r}"
        )
    try:
        fallback_count = int(summary.get(_UNROLL2_SPECIALIZED_BUILDER_FALLBACK_COUNT_FIELD))
    except (TypeError, ValueError):
        fallback_count = -1
    if fallback_count != 0:
        violations.append(
            f"summary.{_UNROLL2_SPECIALIZED_BUILDER_FALLBACK_COUNT_FIELD}: "
            f"expected 0, got {summary.get(_UNROLL2_SPECIALIZED_BUILDER_FALLBACK_COUNT_FIELD)!r}"
        )
    fallback_reason_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "unroll2_specialized_builder_fallback_reason"
    )
    if str(summary.get(fallback_reason_field) or "") != "none":
        violations.append(
            f"summary.{fallback_reason_field}: expected 'none', "
            f"got {summary.get(fallback_reason_field)!r}"
        )
    impl_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl"
    )
    if str(summary.get(impl_field) or "") != "unroll2_specialized_v1":
        violations.append(
            f"summary.{impl_field}: expected 'unroll2_specialized_v1', "
            f"got {summary.get(impl_field)!r}"
        )
    if str(summary.get(_UNROLL2_SPECIALIZED_BUILDER_PATH_FIELD) or "") != ("unroll2_specialized"):
        violations.append(
            f"summary.{_UNROLL2_SPECIALIZED_BUILDER_PATH_FIELD}: "
            f"expected 'unroll2_specialized', got "
            f"{summary.get(_UNROLL2_SPECIALIZED_BUILDER_PATH_FIELD)!r}"
        )
    return violations


def _learner_ready_unroll2_cache_violations(
    args: argparse.Namespace,
    result: dict[str, Any],
) -> list[str]:
    if not bool(getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)):
        return []
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return ["result.summary must be a JSON object"]
    violations: list[str] = []
    tensor_native_impl = str(
        summary.get(
            ("compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl")
        )
        or ""
    )
    tensor_native_table_source = str(
        summary.get(
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_source"
            )
        )
        or ""
    )
    fixed_soa_tensor_native = (
        tensor_native_impl == "fixed_soa_direct_gather_v1"
        and tensor_native_table_source == "fixed_soa_columns_v1"
    )
    if summary.get(_LEARNER_READY_UNROLL2_CACHE_KEY) is not True:
        violations.append(
            f"summary.{_LEARNER_READY_UNROLL2_CACHE_KEY}: expected True, "
            f"got {summary.get(_LEARNER_READY_UNROLL2_CACHE_KEY)!r}"
        )
    try:
        learner_num_unroll_steps = int(summary.get("learner_num_unroll_steps"))
    except (TypeError, ValueError):
        learner_num_unroll_steps = 0
    if learner_num_unroll_steps != 2:
        violations.append(
            "summary.learner_num_unroll_steps: expected 2, "
            f"got {summary.get('learner_num_unroll_steps')!r}"
        )
    if summary.get(_LEARNER_READY_UNROLL2_CACHE_REQUESTED_FIELD) is not True:
        violations.append(
            f"summary.{_LEARNER_READY_UNROLL2_CACHE_REQUESTED_FIELD}: "
            f"expected True, got {summary.get(_LEARNER_READY_UNROLL2_CACHE_REQUESTED_FIELD)!r}"
        )
    for count_field in (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "learner_ready_unroll2_cache_available_group_count"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "learner_ready_unroll2_cache_eligible_count"
        ),
    ):
        try:
            count_value = int(summary.get(count_field))
        except (TypeError, ValueError):
            count_value = -1 if fixed_soa_tensor_native else 0
        if fixed_soa_tensor_native:
            invalid_count = count_value != 0
            expectation = "0"
        else:
            invalid_count = count_value <= 0
            expectation = "positive"
        if invalid_count:
            violations.append(
                f"summary.{count_field}: expected {expectation} count, "
                f"got {summary.get(count_field)!r}"
            )
    expected_used = False if fixed_soa_tensor_native else True
    if summary.get(_LEARNER_READY_UNROLL2_CACHE_USED_FIELD) is not expected_used:
        violations.append(
            f"summary.{_LEARNER_READY_UNROLL2_CACHE_USED_FIELD}: "
            f"expected {expected_used!r}, got "
            f"{summary.get(_LEARNER_READY_UNROLL2_CACHE_USED_FIELD)!r}"
        )
    try:
        call_count = int(summary.get(_LEARNER_READY_UNROLL2_CACHE_CALL_COUNT_FIELD))
    except (TypeError, ValueError):
        call_count = -1 if fixed_soa_tensor_native else 0
    if fixed_soa_tensor_native:
        invalid_call_count = call_count != 0
        call_expectation = "0"
    else:
        invalid_call_count = call_count <= 0
        call_expectation = "positive"
    if invalid_call_count:
        violations.append(
            f"summary.{_LEARNER_READY_UNROLL2_CACHE_CALL_COUNT_FIELD}: "
            f"expected {call_expectation} call count, got "
            f"{summary.get(_LEARNER_READY_UNROLL2_CACHE_CALL_COUNT_FIELD)!r}"
        )
    try:
        fallback_count = int(summary.get(_LEARNER_READY_UNROLL2_CACHE_FALLBACK_COUNT_FIELD))
    except (TypeError, ValueError):
        fallback_count = -1
    if fallback_count != 0:
        violations.append(
            f"summary.{_LEARNER_READY_UNROLL2_CACHE_FALLBACK_COUNT_FIELD}: "
            f"expected 0, got {summary.get(_LEARNER_READY_UNROLL2_CACHE_FALLBACK_COUNT_FIELD)!r}"
        )
    fallback_reason_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "learner_ready_unroll2_cache_fallback_reason"
    )
    if str(summary.get(fallback_reason_field) or "") != "none":
        violations.append(
            f"summary.{fallback_reason_field}: expected 'none', "
            f"got {summary.get(fallback_reason_field)!r}"
        )
    impl_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl"
    )
    expected_impl = (
        "fixed_soa_columns_v1" if fixed_soa_tensor_native else "learner_ready_unroll2_cache_v1"
    )
    if str(summary.get(impl_field) or "") != expected_impl:
        violations.append(
            f"summary.{impl_field}: expected {expected_impl!r}, got {summary.get(impl_field)!r}"
        )
    expected_path = (
        "fixed_soa_direct_gather" if fixed_soa_tensor_native else "learner_ready_unroll2_cache"
    )
    if str(summary.get(_LEARNER_READY_UNROLL2_CACHE_PATH_FIELD) or "") != expected_path:
        violations.append(
            f"summary.{_LEARNER_READY_UNROLL2_CACHE_PATH_FIELD}: "
            f"expected {expected_path!r}, got "
            f"{summary.get(_LEARNER_READY_UNROLL2_CACHE_PATH_FIELD)!r}"
        )
    return violations


def _tensor_native_replay_violations(
    args: argparse.Namespace,
    result: dict[str, Any],
) -> list[str]:
    if not bool(getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)):
        return []
    summary = result.get("summary")
    if not isinstance(summary, dict):
        return ["result.summary must be a JSON object"]
    violations: list[str] = []
    if summary.get(_TENSOR_NATIVE_REPLAY_KEY) is not True:
        violations.append(
            f"summary.{_TENSOR_NATIVE_REPLAY_KEY}: expected True, "
            f"got {summary.get(_TENSOR_NATIVE_REPLAY_KEY)!r}"
        )
    try:
        learner_num_unroll_steps = int(summary.get("learner_num_unroll_steps"))
    except (TypeError, ValueError):
        learner_num_unroll_steps = 0
    if learner_num_unroll_steps != 2:
        violations.append(
            "summary.learner_num_unroll_steps: expected 2, "
            f"got {summary.get('learner_num_unroll_steps')!r}"
        )
    if summary.get(_TENSOR_NATIVE_REPLAY_REQUESTED_FIELD) is not True:
        violations.append(
            f"summary.{_TENSOR_NATIVE_REPLAY_REQUESTED_FIELD}: "
            f"expected True, got {summary.get(_TENSOR_NATIVE_REPLAY_REQUESTED_FIELD)!r}"
        )
    if summary.get(_TENSOR_NATIVE_REPLAY_USED_FIELD) is not True:
        violations.append(
            f"summary.{_TENSOR_NATIVE_REPLAY_USED_FIELD}: "
            f"expected True, got {summary.get(_TENSOR_NATIVE_REPLAY_USED_FIELD)!r}"
        )
    for count_field, label in (
        (_TENSOR_NATIVE_REPLAY_CALL_COUNT_FIELD, "call_count"),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_reused_record_count"
            ),
            "table_reused_record_count",
        ),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_rows"
            ),
            "table_rows",
        ),
    ):
        try:
            count_value = int(summary.get(count_field))
        except (TypeError, ValueError):
            count_value = 0
        if count_value <= 0:
            violations.append(
                f"summary.{count_field}: expected positive {label}, "
                f"got {summary.get(count_field)!r}"
            )
    try:
        fallback_count = int(summary.get(_TENSOR_NATIVE_REPLAY_FALLBACK_COUNT_FIELD))
    except (TypeError, ValueError):
        fallback_count = -1
    if fallback_count != 0:
        violations.append(
            f"summary.{_TENSOR_NATIVE_REPLAY_FALLBACK_COUNT_FIELD}: "
            f"expected 0, got {summary.get(_TENSOR_NATIVE_REPLAY_FALLBACK_COUNT_FIELD)!r}"
        )
    missing_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "tensor_native_replay_table_missing_record_count"
    )
    try:
        missing_count = int(summary.get(missing_field))
    except (TypeError, ValueError):
        missing_count = -1
    if missing_count != 0:
        violations.append(
            f"summary.{missing_field}: expected 0, got {summary.get(missing_field)!r}"
        )
    if (
        summary.get("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested")
        is True
    ):
        for field, expected in (
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used",
                True,
            ),
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped",
                True,
            ),
        ):
            if summary.get(field) is not expected:
                violations.append(
                    f"summary.{field}: expected {expected!r}, got {summary.get(field)!r}"
                )
        for field, expected in (
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count",
                0,
            ),
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count",
                0,
            ),
        ):
            try:
                value = int(summary.get(field))
            except (TypeError, ValueError):
                value = -1
            if value != expected:
                violations.append(
                    f"summary.{field}: expected {expected}, got {summary.get(field)!r}"
                )
        direct_reason_field = (
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason"
        )
        if str(summary.get(direct_reason_field) or "") != "none":
            violations.append(
                f"summary.{direct_reason_field}: expected 'none', "
                f"got {summary.get(direct_reason_field)!r}"
            )
    fallback_reason_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "tensor_native_replay_fallback_reason"
    )
    if str(summary.get(fallback_reason_field) or "") != "none":
        violations.append(
            f"summary.{fallback_reason_field}: expected 'none', "
            f"got {summary.get(fallback_reason_field)!r}"
        )
    impl_field = "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"
    impl_value = str(summary.get(impl_field) or "")
    if impl_value not in {
        "maintained_unroll2_table_gather_v1",
        "selected_maintained_record_table_gather_v1",
        "selected_direct_record_table_gather_v1",
        "fixed_soa_direct_gather_v1",
    }:
        violations.append(
            f"summary.{impl_field}: expected maintained, selected-direct, or fixed-SoA tensor-native replay, "
            f"got {summary.get(impl_field)!r}"
        )
    table_source_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source"
    )
    table_source_value = str(summary.get(table_source_field) or "")
    if table_source_value not in {
        "maintained_record_table_v1",
        "selected_maintained_record_table_v1",
        "selected_direct_record_table_v1",
        "fixed_soa_columns_v1",
    }:
        violations.append(
            f"summary.{table_source_field}: expected maintained, selected-direct, or fixed-SoA table source, "
            f"got {summary.get(table_source_field)!r}"
        )
    if impl_value == "selected_maintained_record_table_gather_v1":
        if table_source_value != "selected_maintained_record_table_v1":
            violations.append(
                f"summary.{table_source_field}: expected selected_maintained_record_table_v1 "
                f"for selected-maintained gather, got {summary.get(table_source_field)!r}"
            )
        for field in (
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested",
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used",
        ):
            if summary.get(field) is not True:
                violations.append(
                    f"summary.{field}: expected True for selected-maintained gather, "
                    f"got {summary.get(field)!r}"
                )
        selected_group_field = (
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count"
        )
        try:
            selected_group_count = int(summary.get(selected_group_field))
        except (TypeError, ValueError):
            selected_group_count = 0
        if selected_group_count <= 0:
            violations.append(
                f"summary.{selected_group_field}: expected positive selected group count "
                f"for selected-maintained gather, got {summary.get(selected_group_field)!r}"
            )
    if (
        summary.get(
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested"
        )
        is True
    ):
        for field in (
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used",
        ):
            if summary.get(field) is not True:
                violations.append(
                    f"summary.{field}: expected True for maintained table handle, "
                    f"got {summary.get(field)!r}"
                )
        for field, expected_positive in (
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count",
                True,
            ),
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_rows",
                True,
            ),
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_missing_record_count",
                False,
            ),
        ):
            try:
                value = int(summary.get(field))
            except (TypeError, ValueError):
                value = -1
            if expected_positive and value <= 0:
                violations.append(
                    f"summary.{field}: expected positive value for maintained table handle, "
                    f"got {summary.get(field)!r}"
                )
            if not expected_positive and value != 0:
                violations.append(
                    f"summary.{field}: expected 0 for maintained table handle, "
                    f"got {summary.get(field)!r}"
                )
    if impl_value == "fixed_soa_direct_gather_v1":
        fixed_expectations = (
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count",
                0,
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec",
                0.0,
            ),
            ("compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count", 0),
            ("compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count", 0),
            (
                "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count",
                0,
            ),
            ("compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count", 0),
            ("compact_rollout_slab_sample_gate_fixed_soa_table_concat_count", 0),
            ("compact_rollout_slab_sample_gate_fixed_soa_fallback_count", 0),
        )
        if table_source_value != "fixed_soa_columns_v1":
            violations.append(
                f"summary.{table_source_field}: expected fixed_soa_columns_v1 "
                f"for fixed SoA, got {summary.get(table_source_field)!r}"
            )
        if summary.get("compact_rollout_slab_sample_gate_fixed_soa_used") is not True:
            violations.append(
                "summary.compact_rollout_slab_sample_gate_fixed_soa_used: "
                f"expected True, got {summary.get('compact_rollout_slab_sample_gate_fixed_soa_used')!r}"
            )
        for field, expected in fixed_expectations:
            raw_value = summary.get(field)
            try:
                value = float(raw_value) if isinstance(expected, float) else int(raw_value)
            except (TypeError, ValueError):
                value = None
            if value != expected:
                violations.append(f"summary.{field}: expected {expected!r}, got {raw_value!r}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--unified-lifecycle-report", type=Path, required=True)
    parser.add_argument("--compact-checkpoint", type=Path)
    parser.add_argument("--volume-prefix")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--collect-function-call-id")
    parser.add_argument("--launch-timeout-sec", type=float, default=360.0)
    parser.add_argument("--result-timeout-sec", type=float, default=20 * 60.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--actor-count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--death-mode",
        choices=tuple(vector_runtime.DEATH_MODES),
        default=vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
    )
    parser.add_argument("--sample-batch-size", type=int, default=2)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--replay-pair-capacity", type=int, default=16)
    parser.add_argument("--learner-train-steps", type=int, default=1)
    parser.add_argument("--learner-num-unroll-steps", type=int, default=1)
    parser.add_argument("--policy-refresh-interval", type=int, default=1)
    parser.add_argument("--compact-owned-loop-deferred-learner", action="store_true")
    parser.add_argument("--compact-owned-loop-deferred-sample-learner", action="store_true")
    parser.add_argument(
        "--compact-owned-loop-deferred-sample-learner-max-pending",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--compact-owned-loop-sample-learner-worker-kind",
        choices=COMPACT_SAMPLE_LEARNER_WORKER_KINDS,
        default=COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD,
    )
    parser.add_argument(
        "--compact-owned-loop-deferred-sample-learner-replay-append-transport-kind",
        choices=COMPACT_REPLAY_APPEND_TRANSPORT_KINDS,
        default=COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1,
    )
    parser.add_argument(
        "--compact-owned-loop-deferred-sample-learner-model-state-transport-kind",
        choices=COMPACT_MODEL_STATE_TRANSPORT_KINDS,
        default=COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
    )
    parser.add_argument("--compact-owned-loop-fused-learner-batch", action="store_true")
    parser.add_argument(
        "--compact-muzero-learner-batch-unroll2-specialized-builder",
        action="store_true",
    )
    parser.add_argument(
        "--compact-muzero-learner-batch-learner-ready-unroll2-cache",
        action="store_true",
    )
    parser.add_argument(
        "--compact-muzero-learner-batch-tensor-native-replay",
        action="store_true",
    )
    parser.add_argument(
        "--compact-owned-accepted-fast-path-preset",
        action="store_true",
        help=(
            "Expand to the accepted direct compact trainer fast-path shape: "
            "direct_core, fused learner batch, borrowed single-actor render state, "
            "lean trainer step, B1024/A1 normal death, unroll 2."
        ),
    )
    parser.add_argument(
        "--compact-owned-accepted-fast-path-step-window",
        choices=tuple(ACCEPTED_FAST_PATH_STEP_WINDOWS),
        default=ACCEPTED_FAST_PATH_STEP_WINDOW_ACCEPTED,
        help=(
            "Step window to use with --compact-owned-accepted-fast-path-preset. "
            "The default is the accepted 180/45 speed-row denominator. "
            "stability_* windows keep the fast-path flags but label the row as "
            "a long-window stability diagnostic. The long windows keep measured "
            "steps congruent with the accepted shape so a final refresh can be "
            "consumed by a post-refresh actor/search step."
        ),
    )
    parser.add_argument("--compact-owned-lean-trainer-step", action="store_true")
    parser.add_argument("--compact-owned-lean-profile-oracle", action="store_true")
    parser.add_argument(
        "--compact-profile-bounded-diagnostics",
        action="store_true",
        help=(
            "Use bounded diagnostic bookkeeping for long stability rows. "
            "The accepted stability windows enable this automatically."
        ),
    )
    parser.add_argument(
        "--compact-profile-cuda-sync-timing-diagnostics",
        action="store_true",
        help=(
            "Insert diagnostic CUDA synchronizations around sample/learner timing "
            "children so long-window instability can be attributed."
        ),
    )
    parser.add_argument(
        "--compact-profile-runtime-step-timing-diagnostics",
        action="store_true",
        help=(
            "Record measured-step runtime envelope stats without also requiring "
            "CUDA synchronization probes."
        ),
    )
    parser.add_argument(
        "--compact-profile-cpu-perf-stat-diagnostics",
        action="store_true",
        help=(
            "Wrap the remote speed-row producer in perf stat and project CPU "
            "hardware counters when the Modal environment permits it."
        ),
    )
    parser.add_argument(
        "--hybrid-persistent-compact-render-state-buffer",
        action="store_true",
        help=(
            "Use compact persistent renderer buffers for actor render-state handoff. "
            "Default-off actor/render-state wall probe."
        ),
    )
    parser.add_argument(
        "--hybrid-borrow-single-actor-render-state",
        action="store_true",
        help=(
            "Borrow a single actor env render state directly instead of copying "
            "render-state rows. Requires actor-count=1."
        ),
    )
    parser.add_argument("--learner-device", default="cuda")
    parser.add_argument(
        "--gpu-utilization-sampling",
        action="store_true",
        help="Sample nvidia-smi inside the measured remote speed-row run.",
    )
    parser.add_argument(
        "--gpu-utilization-sample-interval-sec",
        type=float,
        default=1.0,
        help="Seconds between nvidia-smi utilization samples when sampling is enabled.",
    )
    parser.add_argument("--num-simulations", type=int, default=1)
    parser.add_argument(
        "--search-service-kind",
        default="device_target",
        choices=(
            "device_target",
            "compact_torch_search_service",
            "fixed_shape_search_owner",
            "owner_search_slab_proxy",
            "owner_search_inline_proxy",
            "owner_search_inline_background_proxy",
            "owner_search_threaded_proxy",
        ),
    )
    parser.add_argument(
        "--owner-search-inner-search-service-kind",
        default="compact_torch_search_service",
        choices=(
            "compact_torch_search_service",
            "fixed_shape_search_owner",
        ),
    )
    parser.add_argument("--owner-search-defer-maintenance", action="store_true")
    parser.add_argument("--owner-search-slab-bypass", action="store_true")
    parser.add_argument("--owner-search-transition-batch-size", type=int, default=1)
    parser.add_argument(
        "--owner-search-direct-transition-batch-replay",
        action="store_true",
    )
    parser.add_argument(
        "--owner-search-owner-local-transition-derivation",
        action="store_true",
    )
    parser.add_argument("--owner-search-owner-proxy-transition-closure", action="store_true")
    parser.add_argument(
        "--owner-search-require-resident-root-view",
        action="store_true",
    )
    parser.add_argument(
        "--owner-search-resident-root-host-observation-stub",
        action="store_true",
    )
    parser.add_argument("--owner-search-direct-root-build-request", action="store_true")
    parser.add_argument("--compact-owner-action-step-boundary", action="store_true")
    parser.add_argument("--compact-owner-action-dispatch-step-overlap", action="store_true")
    parser.add_argument("--owner-search-fixed-action-result-buffer", action="store_true")
    parser.add_argument("--owner-search-action-result-slot-capacity", type=int, default=4)
    parser.add_argument("--owner-search-fixed-soa-replay", action="store_true")
    parser.add_argument(
        "--owner-search-fixed-soa-locality-sample-group-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--owner-search-defer-model-state-digest-to-refresh",
        action="store_true",
        help=(
            "Forward the default-off owner-search same-process digest-deferral "
            "ablation to the remote speed-row producer."
        ),
    )
    parser.add_argument("--owner-search-async-learner-worker", action="store_true")
    parser.add_argument(
        "--owner-search-async-learner-worker-kind",
        choices=COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS,
        default=COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
    )
    parser.add_argument("--owner-search-async-learner-max-pending", type=int, default=1)
    parser.add_argument("--compact-torch-request-compile", action="store_true")
    parser.add_argument("--compact-torch-request-model-compile", action="store_true")
    parser.add_argument(
        "--compact-torch-model-compile-mode",
        default="reduce-overhead",
        choices=COMPACT_TORCH_MODEL_COMPILE_MODES,
    )
    parser.add_argument(
        "--compact-torch-timing-mode",
        default="host_phase_sync",
        choices=COMPACT_TORCH_TIMING_MODES,
    )
    parser.add_argument(
        "--compact-torch-initial-inference-mode",
        default="model_method",
        choices=COMPACT_TORCH_INITIAL_INFERENCE_MODES,
    )
    parser.add_argument(
        "--compact-torch-observation-memory-format",
        default="contiguous",
        choices=COMPACT_TORCH_MEMORY_FORMATS,
    )
    parser.add_argument(
        "--compact-torch-model-memory-format",
        default="contiguous",
        choices=COMPACT_TORCH_MEMORY_FORMATS,
    )
    parser.add_argument(
        "--compact-torch-defer-one-simulation-replay-payload",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--source-max-steps", type=int, default=1048576)
    parser.add_argument("--decision-source-frames", type=int, default=1)
    parser.add_argument("--source-physics-step-ms", type=float, default=16.666666666666668)
    parser.add_argument("--source-max-steps-semantics", default="source_physics_steps")
    args = parser.parse_args()
    _apply_accepted_fast_path_preset(args)
    if bool(args.compact_owned_loop_deferred_learner) and bool(
        args.compact_owned_loop_deferred_sample_learner
    ):
        raise ValueError(
            "--compact-owned-loop-deferred-sample-learner cannot be combined with "
            "--compact-owned-loop-deferred-learner"
        )
    if int(args.compact_owned_loop_deferred_sample_learner_max_pending) <= 0:
        raise ValueError(
            "--compact-owned-loop-deferred-sample-learner-max-pending must be positive"
        )
    if str(
        args.compact_owned_loop_sample_learner_worker_kind
    ) != COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD and not bool(
        args.compact_owned_loop_deferred_sample_learner
    ):
        raise ValueError(
            "--compact-owned-loop-sample-learner-worker-kind requires "
            "--compact-owned-loop-deferred-sample-learner"
        )
    if int(getattr(args, "owner_search_async_learner_max_pending", 1)) <= 0:
        raise ValueError("--owner-search-async-learner-max-pending must be positive")
    if int(getattr(args, "owner_search_action_result_slot_capacity", 4) or 4) <= 0:
        raise ValueError("--owner-search-action-result-slot-capacity must be positive")
    fixed_soa_locality_group_size = int(
        getattr(args, "owner_search_fixed_soa_locality_sample_group_size", 1) or 1
    )
    if fixed_soa_locality_group_size <= 0:
        raise ValueError("--owner-search-fixed-soa-locality-sample-group-size must be positive")
    if fixed_soa_locality_group_size > 1 and not bool(
        getattr(args, "owner_search_fixed_soa_replay", False)
    ):
        raise ValueError(
            "--owner-search-fixed-soa-locality-sample-group-size > 1 requires "
            "--owner-search-fixed-soa-replay"
        )
    if bool(getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)):
        if not bool(args.compact_owned_loop_fused_learner_batch):
            raise ValueError(
                "--compact-muzero-learner-batch-learner-ready-unroll2-cache requires "
                "--compact-owned-loop-fused-learner-batch"
            )
        if int(args.learner_num_unroll_steps) != 2:
            raise ValueError(
                "--compact-muzero-learner-batch-learner-ready-unroll2-cache requires "
                "--learner-num-unroll-steps 2"
            )
    if bool(getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)):
        if not bool(args.compact_owned_loop_fused_learner_batch):
            raise ValueError(
                "--compact-muzero-learner-batch-tensor-native-replay requires "
                "--compact-owned-loop-fused-learner-batch"
            )
        if not bool(
            getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ):
            raise ValueError(
                "--compact-muzero-learner-batch-tensor-native-replay requires "
                "--compact-muzero-learner-batch-learner-ready-unroll2-cache"
            )
        if int(args.learner_num_unroll_steps) != 2:
            raise ValueError(
                "--compact-muzero-learner-batch-tensor-native-replay requires "
                "--learner-num-unroll-steps 2"
            )
    if bool(getattr(args, "owner_search_fixed_soa_replay", False)):
        if not bool(getattr(args, "owner_search_direct_transition_batch_replay", False)):
            raise ValueError(
                "--owner-search-fixed-soa-replay requires "
                "--owner-search-direct-transition-batch-replay"
            )
        if not bool(getattr(args, "owner_search_slab_bypass", False)):
            raise ValueError("--owner-search-fixed-soa-replay requires --owner-search-slab-bypass")
        if int(getattr(args, "owner_search_transition_batch_size", 1) or 1) <= 1:
            raise ValueError(
                "--owner-search-fixed-soa-replay requires --owner-search-transition-batch-size > 1"
            )
        if not bool(getattr(args, "compact_owned_loop_fused_learner_batch", False)):
            raise ValueError(
                "--owner-search-fixed-soa-replay requires --compact-owned-loop-fused-learner-batch"
            )
        if not bool(
            getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ):
            raise ValueError(
                "--owner-search-fixed-soa-replay requires "
                "--compact-muzero-learner-batch-learner-ready-unroll2-cache"
            )
        if not bool(getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)):
            raise ValueError(
                "--owner-search-fixed-soa-replay requires "
                "--compact-muzero-learner-batch-tensor-native-replay"
            )
        if int(getattr(args, "learner_num_unroll_steps", 1) or 1) != 2:
            raise ValueError(
                "--owner-search-fixed-soa-replay requires --learner-num-unroll-steps 2"
            )
    if bool(getattr(args, "owner_search_owner_local_transition_derivation", False)):
        if not bool(getattr(args, "owner_search_direct_transition_batch_replay", False)):
            raise ValueError(
                "--owner-search-owner-local-transition-derivation requires "
                "--owner-search-direct-transition-batch-replay"
            )
        if not bool(getattr(args, "owner_search_slab_bypass", False)):
            raise ValueError(
                "--owner-search-owner-local-transition-derivation requires "
                "--owner-search-slab-bypass"
            )
        if int(getattr(args, "owner_search_transition_batch_size", 1) or 1) <= 1:
            raise ValueError(
                "--owner-search-owner-local-transition-derivation requires "
                "--owner-search-transition-batch-size > 1"
            )
    if bool(getattr(args, "owner_search_owner_proxy_transition_closure", False)):
        if not bool(getattr(args, "owner_search_owner_local_transition_derivation", False)):
            raise ValueError(
                "--owner-search-owner-proxy-transition-closure requires "
                "--owner-search-owner-local-transition-derivation"
            )
        if not bool(getattr(args, "owner_search_direct_root_build_request", False)):
            raise ValueError(
                "--owner-search-owner-proxy-transition-closure requires "
                "--owner-search-direct-root-build-request"
            )
    if bool(getattr(args, "owner_search_resident_root_host_observation_stub", False)) and not (
        bool(getattr(args, "owner_search_slab_bypass", False))
        and bool(getattr(args, "owner_search_require_resident_root_view", False))
    ):
        raise ValueError(
            "--owner-search-resident-root-host-observation-stub requires "
            "--owner-search-slab-bypass and --owner-search-require-resident-root-view"
        )
    if bool(getattr(args, "owner_search_direct_root_build_request", False)) and not (
        bool(getattr(args, "owner_search_slab_bypass", False))
        and bool(getattr(args, "owner_search_require_resident_root_view", False))
        and bool(getattr(args, "owner_search_resident_root_host_observation_stub", False))
    ):
        raise ValueError(
            "--owner-search-direct-root-build-request requires "
            "--owner-search-slab-bypass, --owner-search-require-resident-root-view, "
            "and --owner-search-resident-root-host-observation-stub"
        )
    if bool(getattr(args, "owner_search_direct_root_build_request", False)) and str(
        getattr(args, "search_service_kind", "")
    ) not in {
        "owner_search_inline_proxy",
        "owner_search_inline_background_proxy",
        "owner_search_threaded_proxy",
    }:
        raise ValueError(
            "--owner-search-direct-root-build-request requires an inline, "
            "inline-background, or threaded owner-search proxy"
        )
    if bool(getattr(args, "compact_owner_action_step_boundary", False)):
        if not bool(getattr(args, "owner_search_direct_root_build_request", False)):
            raise ValueError(
                "--compact-owner-action-step-boundary requires "
                "--owner-search-direct-root-build-request"
            )
        if not bool(getattr(args, "owner_search_slab_bypass", False)):
            raise ValueError(
                "--compact-owner-action-step-boundary requires --owner-search-slab-bypass"
            )
    if bool(getattr(args, "compact_owner_action_dispatch_step_overlap", False)):
        if not bool(getattr(args, "compact_owner_action_step_boundary", False)):
            raise ValueError(
                "--compact-owner-action-dispatch-step-overlap requires "
                "--compact-owner-action-step-boundary"
            )
        if not bool(getattr(args, "owner_search_direct_root_build_request", False)):
            raise ValueError(
                "--compact-owner-action-dispatch-step-overlap requires "
                "--owner-search-direct-root-build-request"
            )
    if bool(getattr(args, "owner_search_fixed_action_result_buffer", False)):
        if not bool(getattr(args, "owner_search_defer_maintenance", False)):
            raise ValueError(
                "--owner-search-fixed-action-result-buffer requires "
                "--owner-search-defer-maintenance"
            )
        if not bool(getattr(args, "owner_search_direct_root_build_request", False)):
            raise ValueError(
                "--owner-search-fixed-action-result-buffer requires "
                "--owner-search-direct-root-build-request"
            )
        if str(getattr(args, "search_service_kind", "")) not in {
            "owner_search_inline_proxy",
            "owner_search_inline_background_proxy",
            "owner_search_threaded_proxy",
        }:
            raise ValueError(
                "--owner-search-fixed-action-result-buffer requires an inline, "
                "inline-background, or threaded owner-search proxy"
            )
    if (
        bool(args.compact_owned_loop_deferred_sample_learner)
        and _uses_compact_torch_search_service(args)
        and int(args.sample_interval) > 0
        and (int(args.steps) + int(args.warmup_steps)) % int(args.sample_interval) == 0
    ):
        raise ValueError(
            "compact Torch deferred sample+learner rows need at least one "
            "post-refresh actor/search step; choose steps + warmup_steps that "
            "is not divisible by --sample-interval"
        )
    if str(args.compact_torch_model_memory_format) != "contiguous":
        raise ValueError(
            "compact_torch_model_memory_format=channels_last is parked for the "
            "current LightZero MuZero model because recurrent dynamics uses "
            ".view(); use --compact-torch-model-memory-format contiguous"
        )
    if float(args.gpu_utilization_sample_interval_sec) < 0.0:
        raise ValueError("--gpu-utilization-sample-interval-sec must be non-negative")

    repo_root = Path.cwd()
    lifecycle_path = _resolve_path(args.unified_lifecycle_report, repo_root)
    lifecycle = _load_json(lifecycle_path)
    checkpoint_path = _resolve_checkpoint_path(args, lifecycle, repo_root)
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    volume_prefix = str(
        args.volume_prefix or f"optimizer/compact-coach-speed-row/{_safe_ref_part(args.run_id)}"
    ).strip("/")
    lifecycle_ref = f"{volume_prefix}/unified_lifecycle_report.json"
    checkpoint_ref = f"{volume_prefix}/{checkpoint_path.name}"
    launch_payload: dict[str, Any] | None = None
    if args.collect_function_call_id:
        function_call_id = str(args.collect_function_call_id)
        launch_payload = {
            "schema_id": SPAWN_SCHEMA_ID,
            "status": "provided",
            "function_call_id": function_call_id,
            "result_capture": "modal_function_call_get",
        }
    else:
        if not args.skip_upload:
            _modal_volume_put(lifecycle_path, lifecycle_ref)
            _modal_volume_put(checkpoint_path, checkpoint_ref)
        try:
            launch_payload = _launch_remote(args, lifecycle_ref, checkpoint_ref)
        except ModalRemoteLaunchError as exc:
            report_path = _write_launch_failure_report(
                args=args,
                output_dir=output_dir,
                launch_payload=exc.payload,
            )
            print(json.dumps({"ok": False, "report_path": str(report_path)}, sort_keys=True))
            return 1
        function_call_id = str(launch_payload.get("function_call_id") or "")
        if not function_call_id:
            raise SystemExit("launch payload missing function_call_id")

    _write_json(output_dir / "launch.json", launch_payload)
    bundle = _collect_modal_function_call(function_call_id, args.result_timeout_sec)
    _write_json(output_dir / "remote_bundle.json", bundle)
    if not isinstance(bundle, dict) or bundle.get("ok") is not True:
        failure = {
            "ok": False,
            "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
            "run_id": str(args.run_id),
            "function_call_id": function_call_id,
            "problem": (
                bundle.get("problem")
                if isinstance(bundle, dict)
                else "FunctionCall result was not a JSON object"
            ),
            "compact_torch_observation_memory_format": str(
                args.compact_torch_observation_memory_format
            ),
            "compact_torch_initial_inference_mode": str(args.compact_torch_initial_inference_mode),
            "compact_torch_model_memory_format": str(args.compact_torch_model_memory_format),
            "compact_torch_defer_one_simulation_replay_payload_requested": bool(
                args.compact_torch_defer_one_simulation_replay_payload
            ),
            "compact_torch_memory_format_applies_to_search_service": (
                _uses_compact_torch_search_service(args)
            ),
            **_owner_search_config_fields(args),
            "hybrid_persistent_compact_render_state_buffer": bool(
                args.hybrid_persistent_compact_render_state_buffer
            ),
            "hybrid_borrow_single_actor_render_state": bool(
                args.hybrid_borrow_single_actor_render_state
            ),
            "death_mode": str(args.death_mode),
            "learner_num_unroll_steps": int(args.learner_num_unroll_steps),
            "compact_owned_loop_deferred_learner": bool(args.compact_owned_loop_deferred_learner),
            "compact_owned_loop_deferred_sample_learner": bool(
                args.compact_owned_loop_deferred_sample_learner
            ),
            "compact_owned_loop_deferred_sample_learner_max_pending_requested": int(
                args.compact_owned_loop_deferred_sample_learner_max_pending
            ),
            "compact_owned_loop_sample_learner_worker_kind_requested": str(
                args.compact_owned_loop_sample_learner_worker_kind
            ),
            (
                "compact_owned_loop_deferred_sample_learner_replay_append_transport_kind_requested"
            ): str(args.compact_owned_loop_deferred_sample_learner_replay_append_transport_kind),
            (
                "compact_owned_loop_deferred_sample_learner_model_state_transport_kind_requested"
            ): str(
                getattr(
                    args,
                    ("compact_owned_loop_deferred_sample_learner_model_state_transport_kind"),
                    COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
                )
            ),
            "compact_owned_loop_fused_learner_batch": bool(
                args.compact_owned_loop_fused_learner_batch
            ),
            "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
                args.compact_muzero_learner_batch_unroll2_specialized_builder
            ),
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
                getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
            ),
            "compact_muzero_learner_batch_tensor_native_replay": bool(
                getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
            ),
            "compact_owned_accepted_fast_path_preset": bool(
                getattr(args, "compact_owned_accepted_fast_path_preset", False)
            ),
            "compact_owned_accepted_fast_path_preset_name": (
                ACCEPTED_FAST_PATH_PRESET
                if bool(getattr(args, "compact_owned_accepted_fast_path_preset", False))
                else ""
            ),
            **_accepted_fast_path_step_window_report_fields(args),
            "compact_profile_bounded_diagnostics": bool(
                getattr(args, "compact_profile_bounded_diagnostics", False)
            ),
            "compact_profile_cuda_sync_timing_diagnostics": bool(
                getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
            ),
            "compact_profile_runtime_step_timing_diagnostics": bool(
                getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
                or getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
            ),
            "compact_profile_cpu_perf_stat_diagnostics": bool(
                getattr(args, "compact_profile_cpu_perf_stat_diagnostics", False)
            ),
            **(
                {
                    key: value
                    for key, value in bundle.items()
                    if str(key).startswith("compact_profile_cpu_perf_stat_")
                }
                if isinstance(bundle, dict)
                else {}
            ),
            "compact_owned_lean_trainer_step": bool(args.compact_owned_lean_trainer_step),
            "compact_owned_lean_profile_oracle_requested": bool(
                args.compact_owned_lean_profile_oracle
            ),
            "compact_owned_training_loop_owner": (
                "lean_compact_trainer_step"
                if bool(args.compact_owned_lean_trainer_step)
                else "hybrid_observation_profile_runner"
            ),
            "speed_row_gpu_utilization_sampling_enabled": bool(args.gpu_utilization_sampling),
            "speed_row_gpu_utilization_sample_interval_sec": float(
                args.gpu_utilization_sample_interval_sec
            ),
        }
        _write_json(output_dir / "compact_coach_speed_row_modal_report.json", failure)
        print(json.dumps({"ok": False, "report_path": str(output_dir)}, sort_keys=True))
        return 1

    manifest = _required_mapping(bundle.get("manifest"), "remote manifest")
    result = _required_mapping(bundle.get("result"), "remote result")
    manifest_path = output_dir / "manifest.json"
    result_path = output_dir / f"row_{ROW_ID}_result.json"
    _write_json(manifest_path, manifest)
    _write_json(result_path, result)
    summary = result.get("summary")
    summary_report_fields = summary if isinstance(summary, dict) else {}
    accepted_fast_path_violations = _accepted_fast_path_preset_violations(args, result)
    if accepted_fast_path_violations:
        report_path = output_dir / "compact_coach_speed_row_modal_report.json"
        failure = {
            "ok": False,
            "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
            "run_id": str(args.run_id),
            "function_call_id": function_call_id,
            "problem": "accepted fast-path preset result mismatch",
            "compact_owned_accepted_fast_path_preset": True,
            "compact_owned_accepted_fast_path_preset_name": ACCEPTED_FAST_PATH_PRESET,
            **_accepted_fast_path_step_window_report_fields(args),
            "compact_profile_cuda_sync_timing_diagnostics": bool(
                getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
            ),
            "compact_profile_runtime_step_timing_diagnostics": bool(
                getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
                or getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
            ),
            "compact_profile_cpu_perf_stat_diagnostics": bool(
                getattr(args, "compact_profile_cpu_perf_stat_diagnostics", False)
            ),
            "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
                args.compact_muzero_learner_batch_unroll2_specialized_builder
            ),
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
                getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
            ),
            "compact_muzero_learner_batch_tensor_native_replay": bool(
                getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
            ),
            **_learner_ready_unroll2_cache_proof_report_fields(summary_report_fields),
            **_tensor_native_replay_proof_report_fields(summary_report_fields),
            **_unroll2_specialized_builder_proof_report_fields(summary_report_fields),
            "accepted_fast_path_preset_violations": accepted_fast_path_violations,
            "manifest_path": str(manifest_path),
            "result_path": str(result_path),
            "remote_bundle_path": str(output_dir / "remote_bundle.json"),
            "promotion_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }
        _write_json(report_path, failure)
        print(json.dumps({"ok": False, "report_path": str(report_path)}, sort_keys=True))
        return 1
    cuda_sync_timing_diagnostic_violations = _cuda_sync_timing_diagnostic_violations(
        args,
        result,
    )
    runtime_step_timing_diagnostic_violations = _runtime_step_timing_diagnostic_violations(
        args,
        result,
    )
    if cuda_sync_timing_diagnostic_violations:
        report_path = output_dir / "compact_coach_speed_row_modal_report.json"
        failure = {
            "ok": False,
            "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
            "run_id": str(args.run_id),
            "function_call_id": function_call_id,
            "problem": "cuda sync timing diagnostic result mismatch",
            "compact_profile_cuda_sync_timing_diagnostics": bool(
                getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
            ),
            "compact_profile_runtime_step_timing_diagnostics": bool(
                getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
                or getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
            ),
            "compact_profile_cpu_perf_stat_diagnostics": bool(
                getattr(args, "compact_profile_cpu_perf_stat_diagnostics", False)
            ),
            "cuda_sync_timing_diagnostic_violations": (cuda_sync_timing_diagnostic_violations),
            "accepted_fast_path_preset_violations": accepted_fast_path_violations,
            "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
                args.compact_muzero_learner_batch_unroll2_specialized_builder
            ),
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
                getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
            ),
            "compact_muzero_learner_batch_tensor_native_replay": bool(
                getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
            ),
            **_learner_ready_unroll2_cache_proof_report_fields(summary_report_fields),
            **_tensor_native_replay_proof_report_fields(summary_report_fields),
            **_unroll2_specialized_builder_proof_report_fields(summary_report_fields),
            "unroll2_specialized_builder_violations": [],
            "manifest_path": str(manifest_path),
            "result_path": str(result_path),
            "remote_bundle_path": str(output_dir / "remote_bundle.json"),
            "promotion_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }
        _write_json(report_path, failure)
        print(json.dumps({"ok": False, "report_path": str(report_path)}, sort_keys=True))
        return 1
    if runtime_step_timing_diagnostic_violations:
        report_path = output_dir / "compact_coach_speed_row_modal_report.json"
        failure = {
            "ok": False,
            "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
            "run_id": str(args.run_id),
            "function_call_id": function_call_id,
            "problem": "runtime-step timing diagnostic result mismatch",
            "compact_profile_cuda_sync_timing_diagnostics": bool(
                getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
            ),
            "compact_profile_runtime_step_timing_diagnostics": bool(
                getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
                or getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
            ),
            "compact_profile_cpu_perf_stat_diagnostics": bool(
                getattr(args, "compact_profile_cpu_perf_stat_diagnostics", False)
            ),
            "runtime_step_timing_diagnostic_violations": (
                runtime_step_timing_diagnostic_violations
            ),
            "accepted_fast_path_preset_violations": accepted_fast_path_violations,
            "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
                args.compact_muzero_learner_batch_unroll2_specialized_builder
            ),
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
                getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
            ),
            "compact_muzero_learner_batch_tensor_native_replay": bool(
                getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
            ),
            **_learner_ready_unroll2_cache_proof_report_fields(summary_report_fields),
            **_tensor_native_replay_proof_report_fields(summary_report_fields),
            **_unroll2_specialized_builder_proof_report_fields(summary_report_fields),
            "manifest_path": str(manifest_path),
            "result_path": str(result_path),
            "remote_bundle_path": str(output_dir / "remote_bundle.json"),
            "promotion_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }
        _write_json(report_path, failure)
        print(json.dumps({"ok": False, "report_path": str(report_path)}, sort_keys=True))
        return 1
    unroll2_specialized_builder_violations = _unroll2_specialized_builder_violations(
        args,
        result,
    )
    if unroll2_specialized_builder_violations:
        report_path = output_dir / "compact_coach_speed_row_modal_report.json"
        failure = {
            "ok": False,
            "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
            "run_id": str(args.run_id),
            "function_call_id": function_call_id,
            "problem": "unroll2 specialized builder result mismatch",
            "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
                args.compact_muzero_learner_batch_unroll2_specialized_builder
            ),
            **_unroll2_specialized_builder_proof_report_fields(summary_report_fields),
            **_tensor_native_replay_proof_report_fields(summary_report_fields),
            "unroll2_specialized_builder_violations": (unroll2_specialized_builder_violations),
            "accepted_fast_path_preset_violations": accepted_fast_path_violations,
            "cuda_sync_timing_diagnostic_violations": (cuda_sync_timing_diagnostic_violations),
            "manifest_path": str(manifest_path),
            "result_path": str(result_path),
            "remote_bundle_path": str(output_dir / "remote_bundle.json"),
            "promotion_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }
        _write_json(report_path, failure)
        print(json.dumps({"ok": False, "report_path": str(report_path)}, sort_keys=True))
        return 1
    learner_ready_unroll2_cache_violations = _learner_ready_unroll2_cache_violations(
        args,
        result,
    )
    if learner_ready_unroll2_cache_violations:
        report_path = output_dir / "compact_coach_speed_row_modal_report.json"
        failure = {
            "ok": False,
            "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
            "run_id": str(args.run_id),
            "function_call_id": function_call_id,
            "problem": "learner-ready unroll2 cache result mismatch",
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
                getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
            ),
            **_learner_ready_unroll2_cache_proof_report_fields(summary_report_fields),
            **_tensor_native_replay_proof_report_fields(summary_report_fields),
            "learner_ready_unroll2_cache_violations": (learner_ready_unroll2_cache_violations),
            "accepted_fast_path_preset_violations": accepted_fast_path_violations,
            "cuda_sync_timing_diagnostic_violations": (cuda_sync_timing_diagnostic_violations),
            "manifest_path": str(manifest_path),
            "result_path": str(result_path),
            "remote_bundle_path": str(output_dir / "remote_bundle.json"),
            "promotion_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }
        _write_json(report_path, failure)
        print(json.dumps({"ok": False, "report_path": str(report_path)}, sort_keys=True))
        return 1
    tensor_native_replay_violations = _tensor_native_replay_violations(
        args,
        result,
    )
    if tensor_native_replay_violations:
        report_path = output_dir / "compact_coach_speed_row_modal_report.json"
        failure = {
            "ok": False,
            "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
            "run_id": str(args.run_id),
            "function_call_id": function_call_id,
            "problem": "tensor-native replay result mismatch",
            "compact_muzero_learner_batch_tensor_native_replay": bool(
                getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
            ),
            **_learner_ready_unroll2_cache_proof_report_fields(summary_report_fields),
            **_tensor_native_replay_proof_report_fields(summary_report_fields),
            "tensor_native_replay_violations": tensor_native_replay_violations,
            "accepted_fast_path_preset_violations": accepted_fast_path_violations,
            "cuda_sync_timing_diagnostic_violations": (cuda_sync_timing_diagnostic_violations),
            "manifest_path": str(manifest_path),
            "result_path": str(result_path),
            "remote_bundle_path": str(output_dir / "remote_bundle.json"),
            "promotion_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }
        _write_json(report_path, failure)
        print(json.dumps({"ok": False, "report_path": str(report_path)}, sort_keys=True))
        return 1
    saved = save_compact_coach_speed_row_evidence_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        candidate_checkpoint_id=str(result["candidate_checkpoint_id"]),
        unified_lifecycle_report_path=lifecycle_path,
        manifest_path=manifest_path,
        row_id=ROW_ID,
        result_json_path=result_path,
        speed_currency=SPEED_CURRENCY,
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    evidence = saved["evidence"]
    report = {
        "ok": True,
        "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
        "run_id": str(args.run_id),
        "function_call_id": function_call_id,
        "candidate_checkpoint_id": str(result["candidate_checkpoint_id"]),
        "manifest_path": str(manifest_path),
        "result_path": str(result_path),
        "remote_bundle_path": str(output_dir / "remote_bundle.json"),
        "evidence_path": str(saved["path"]),
        "evidence_ref": compact_coach_speed_row_evidence_ref(evidence),
        "speed_currency": SPEED_CURRENCY,
        "env_steps_collected": result["summary"]["env_steps_collected"],
        "training_wall_sec": result["summary"]["training_wall_sec"],
        "compact_trainer_env_steps_per_sec": result["summary"]["steps_per_sec"],
        "steps_per_sec": result["summary"]["steps_per_sec"],
        "seed": result["summary"].get("seed"),
        "sample_seed_base": result["summary"].get("sample_seed_base"),
        "sample_batch_size": result["summary"].get("sample_batch_size"),
        "sample_interval": result["summary"].get("sample_interval"),
        "replay_pair_capacity": result["summary"].get("replay_pair_capacity"),
        "learner_train_steps": result["summary"].get("learner_train_steps"),
        "policy_refresh_interval": result["summary"].get("policy_refresh_interval"),
        "num_simulations": result["summary"].get("num_simulations"),
        "compact_rollout_slab_sample_gate_last_seed": result["summary"].get(
            "compact_rollout_slab_sample_gate_last_seed"
        ),
        "compact_rollout_slab_learner_gate_last_seed": result["summary"].get(
            "compact_rollout_slab_learner_gate_last_seed"
        ),
        "compact_owned_loop_sample_gate_last_metadata_seed": result["summary"].get(
            "compact_owned_loop_sample_gate_last_metadata_seed"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed": (
            result["summary"].get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed"
            )
        ),
        "search_service_kind": result["summary"].get("search_service_kind"),
        "search_service_impl": result["summary"].get("search_service_impl"),
        "owner_search_slab_proxy_requested": result["summary"].get(
            "owner_search_slab_proxy_requested"
        ),
        "owner_search_inline_proxy_requested": result["summary"].get(
            "owner_search_inline_proxy_requested"
        ),
        "owner_search_inline_background_proxy_requested": result["summary"].get(
            "owner_search_inline_background_proxy_requested"
        ),
        "owner_search_threaded_proxy_requested": result["summary"].get(
            "owner_search_threaded_proxy_requested"
        ),
        "owner_search_inner_search_service_kind": result["summary"].get(
            "owner_search_inner_search_service_kind"
        ),
        "owner_search_inner_search_service_impl": result["summary"].get(
            "owner_search_inner_search_service_impl"
        ),
        "owner_search_compact_torch_resident_root_bridge_ready": result["summary"].get(
            "owner_search_compact_torch_resident_root_bridge_ready"
        ),
        "owner_search_defer_maintenance_requested": result["summary"].get(
            "owner_search_defer_maintenance_requested"
        ),
        "owner_search_slab_bypass_requested": result["summary"].get(
            "owner_search_slab_bypass_requested"
        ),
        "owner_search_transition_batch_size_requested": result["summary"].get(
            "owner_search_transition_batch_size_requested"
        ),
        "owner_search_transition_batch_transport_requested": result["summary"].get(
            "owner_search_transition_batch_transport_requested"
        ),
        "owner_search_direct_transition_batch_replay_requested": result["summary"].get(
            "owner_search_direct_transition_batch_replay_requested"
        ),
        "owner_search_owner_local_transition_derivation_requested": result["summary"].get(
            "owner_search_owner_local_transition_derivation_requested"
        ),
        "owner_search_owner_proxy_transition_closure_requested": result["summary"].get(
            "owner_search_owner_proxy_transition_closure_requested"
        ),
        "owner_search_require_resident_root_view_requested": result["summary"].get(
            "owner_search_require_resident_root_view_requested"
        ),
        "owner_search_resident_root_host_observation_stub_requested": result["summary"].get(
            "owner_search_resident_root_host_observation_stub_requested"
        ),
        "owner_search_direct_root_build_request_requested": result["summary"].get(
            "owner_search_direct_root_build_request_requested"
        ),
        "compact_owner_action_step_boundary_requested": result["summary"].get(
            "compact_owner_action_step_boundary_requested"
        ),
        "owner_search_fixed_action_result_buffer_requested": result["summary"].get(
            "owner_search_fixed_action_result_buffer_requested"
        ),
        "owner_search_action_result_slot_capacity_requested": result["summary"].get(
            "owner_search_action_result_slot_capacity_requested"
        ),
        "compact_owner_search_resident_root_bridge_ready": result["summary"].get(
            "compact_owner_search_resident_root_bridge_ready"
        ),
        "compact_owner_search_resident_root_bridge_kind": result["summary"].get(
            "compact_owner_search_resident_root_bridge_kind"
        ),
        "compact_owner_search_resident_root_bridge_device": result["summary"].get(
            "compact_owner_search_resident_root_bridge_device"
        ),
        "compact_owner_search_resident_root_bridge_h2d_bytes": result["summary"].get(
            "compact_owner_search_resident_root_bridge_h2d_bytes"
        ),
        "compact_owner_search_resident_root_bridge_host_observation_copied": result["summary"].get(
            "compact_owner_search_resident_root_bridge_host_observation_copied"
        ),
        "compact_owner_search_resident_root_bridge_generation_id": result["summary"].get(
            "compact_owner_search_resident_root_bridge_generation_id"
        ),
        "compact_owned_loop_deferred_learner": result["summary"].get(
            "compact_owned_loop_deferred_learner"
        ),
        "compact_owned_loop_deferred_sample_learner": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner"
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_requested": (
            result["summary"].get(
                "compact_owned_loop_deferred_sample_learner_max_pending_requested"
            )
        ),
        "compact_owned_loop_sample_learner_worker_kind_requested": result["summary"].get(
            "compact_owned_loop_sample_learner_worker_kind_requested"
        ),
        "compact_owned_loop_fused_learner_batch": result["summary"].get(
            "compact_owned_loop_fused_learner_batch"
        ),
        "compact_muzero_learner_batch_unroll2_specialized_builder": result["summary"].get(
            "compact_muzero_learner_batch_unroll2_specialized_builder"
        ),
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": result["summary"].get(
            "compact_muzero_learner_batch_learner_ready_unroll2_cache"
        ),
        "compact_muzero_learner_batch_tensor_native_replay": result["summary"].get(
            "compact_muzero_learner_batch_tensor_native_replay"
        ),
        **_learner_ready_unroll2_cache_proof_report_fields(summary_report_fields),
        **_tensor_native_replay_proof_report_fields(summary_report_fields),
        **_whole_owner_buffer_replay_ceiling_report_fields(summary_report_fields),
        **_unroll2_specialized_builder_proof_report_fields(summary_report_fields),
        "compact_owned_accepted_fast_path_preset": bool(
            getattr(args, "compact_owned_accepted_fast_path_preset", False)
        ),
        "compact_owned_accepted_fast_path_preset_name": (
            ACCEPTED_FAST_PATH_PRESET
            if bool(getattr(args, "compact_owned_accepted_fast_path_preset", False))
            else ""
        ),
        **_accepted_fast_path_step_window_report_fields(args),
        "compact_profile_bounded_diagnostics": result["summary"].get(
            "compact_profile_bounded_diagnostics"
        ),
        "compact_profile_cuda_sync_timing_diagnostics": result["summary"].get(
            "compact_profile_cuda_sync_timing_diagnostics"
        ),
        "compact_profile_runtime_step_timing_diagnostics": result["summary"].get(
            "compact_profile_runtime_step_timing_diagnostics"
        ),
        **{
            key: value
            for key, value in summary_report_fields.items()
            if str(key).startswith("compact_profile_cpu_perf_stat_")
        },
        "source_profile_payload_embedded": result["summary"].get("source_profile_payload_embedded"),
        "accepted_fast_path_preset_violations": accepted_fast_path_violations,
        "cuda_sync_timing_diagnostic_violations": cuda_sync_timing_diagnostic_violations,
        "runtime_step_timing_diagnostic_violations": (runtime_step_timing_diagnostic_violations),
        "unroll2_specialized_builder_violations": unroll2_specialized_builder_violations,
        "learner_ready_unroll2_cache_violations": learner_ready_unroll2_cache_violations,
        "tensor_native_replay_violations": tensor_native_replay_violations,
        "resident_replay_snapshot_mode": result["summary"].get("resident_replay_snapshot_mode"),
        "compact_owned_loop_replay_store_retained_resident_snapshot_count": (
            result["summary"].get(
                "compact_owned_loop_replay_store_retained_resident_snapshot_count"
            )
        ),
        "compact_owned_loop_replay_store_retained_resident_snapshot_bytes": (
            result["summary"].get(
                "compact_owned_loop_replay_store_retained_resident_snapshot_bytes"
            )
        ),
        "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_count": (
            result["summary"].get(
                "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_count"
            )
        ),
        "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_bytes": (
            result["summary"].get(
                "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_bytes"
            )
        ),
        "compact_owned_lean_trainer_step": result["summary"].get("compact_owned_lean_trainer_step"),
        "compact_owned_lean_profile_oracle_requested": result["summary"].get(
            "compact_owned_lean_profile_oracle_requested"
        ),
        "compact_owned_training_loop_owner": result["summary"].get(
            "compact_owned_training_loop_owner"
        ),
        "compact_owned_lean_profile_oracle": result["summary"].get(
            "compact_owned_lean_profile_oracle"
        ),
        "compact_rollout_slab_sample_gate_sec": result["summary"].get(
            "compact_rollout_slab_sample_gate_sec"
        ),
        "compact_rollout_slab_learner_gate_sec": result["summary"].get(
            "compact_rollout_slab_learner_gate_sec"
        ),
        "compact_rollout_slab_sample_gate_calls": result["summary"].get(
            "compact_rollout_slab_sample_gate_calls"
        ),
        "compact_rollout_slab_learner_gate_calls": result["summary"].get(
            "compact_rollout_slab_learner_gate_calls"
        ),
        "compact_rollout_slab_learner_gate_updates": result["summary"].get(
            "compact_rollout_slab_learner_gate_updates"
        ),
        "compact_owned_loop_record_step_calls": result["summary"].get(
            "compact_owned_loop_record_step_calls"
        ),
        "compact_owned_loop_appended_replay_entry_count": result["summary"].get(
            "compact_owned_loop_appended_replay_entry_count"
        ),
        "compact_rollout_slab_sample_gate_sample_rows": result["summary"].get(
            "compact_rollout_slab_sample_gate_sample_rows"
        ),
        "compact_rollout_slab_learner_gate_sample_rows": result["summary"].get(
            "compact_rollout_slab_learner_gate_sample_rows"
        ),
        "compact_rollout_slab_sample_gate_opportunities": result["summary"].get(
            "compact_rollout_slab_sample_gate_opportunities"
        ),
        "compact_rollout_slab_sample_gate_skipped_count": result["summary"].get(
            "compact_rollout_slab_sample_gate_skipped_count"
        ),
        "compact_owned_trainer_learner_update_count": result["summary"].get(
            "compact_owned_trainer_learner_update_count"
        ),
        "compact_owned_trainer_sample_batch_count": result["summary"].get(
            "compact_owned_trainer_sample_batch_count"
        ),
        "compact_owned_trainer_policy_refresh_count": result["summary"].get(
            "compact_owned_trainer_policy_refresh_count"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": (
            result["summary"].get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count"
            )
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest": result[
            "summary"
        ].get("compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest"),
        "compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind": result[
            "summary"
        ].get("compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind"),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind": result[
            "summary"
        ].get("compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind"),
        "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count": result[
            "summary"
        ].get("compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count"),
        "compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count": result[
            "summary"
        ].get("compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count"),
        (
            "compact_rollout_slab_policy_refresh_after_learner_gate_"
            "parent_model_state_transport_avoided"
        ): result["summary"].get(
            (
                "compact_rollout_slab_policy_refresh_after_learner_gate_"
                "parent_model_state_transport_avoided"
            )
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_calls": result["summary"].get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_calls"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_interval": result["summary"].get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_interval"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count": (
            result["summary"].get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count"
            )
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count": (
            result["summary"].get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count"
            )
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata_update_count": (
            result["summary"].get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata_update_count"
            )
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata_update_count": (
            result["summary"].get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata_update_count"
            )
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_lag_to_final_update": (
            result["summary"].get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_lag_to_final_update"
            )
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_lag_to_final_update": (
            result["summary"].get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_lag_to_final_update"
            )
        ),
        "source_profile_total_sec": result["summary"].get("source_profile_total_sec"),
        "source_profile_warmup_sec": result["summary"].get("source_profile_warmup_sec"),
        "source_profile_measured_sec": result["summary"].get("source_profile_measured_sec"),
        "source_profile_timing_per_timestep_sec": result["summary"].get(
            "source_profile_timing_per_timestep_sec"
        ),
        "speed_row_actor_step_wall_sec": result["summary"].get("speed_row_actor_step_wall_sec"),
        "speed_row_observation_sec": result["summary"].get("speed_row_observation_sec"),
        "speed_row_renderer_stack_update_sec": result["summary"].get(
            "speed_row_renderer_stack_update_sec"
        ),
        "speed_row_compact_rollout_slab_sec": result["summary"].get(
            "speed_row_compact_rollout_slab_sec"
        ),
        "speed_row_sample_gate_sec": result["summary"].get("speed_row_sample_gate_sec"),
        "speed_row_learner_gate_sec": result["summary"].get("speed_row_learner_gate_sec"),
        "speed_row_policy_refresh_sec": result["summary"].get("speed_row_policy_refresh_sec"),
        "speed_row_primary_accounted_sec": result["summary"].get("speed_row_primary_accounted_sec"),
        "speed_row_primary_residual_sec": result["summary"].get("speed_row_primary_residual_sec"),
        **{key: result["summary"].get(key) for key in _ACTOR_OBSERVATION_TIMER_REPORT_FIELDS},
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": result["summary"].get(
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch"
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": result["summary"].get(
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"
        ),
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": result[
            "summary"
        ].get("compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch"),
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": result[
            "summary"
        ].get("compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"),
        "compact_rollout_slab_sample_gate_host_provider_learner_batch": result["summary"].get(
            "compact_rollout_slab_sample_gate_host_provider_learner_batch"
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source": result[
            "summary"
        ].get("compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source"),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes": result[
            "summary"
        ].get("compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes"),
        **{
            f"{prefix}_stats": result["summary"].get(f"{prefix}_stats")
            for prefix in _SAMPLE_GATE_PER_CALL_REPORT_PREFIXES
        },
        "compact_rollout_slab_learner_gate_prebuilt_batch_used": result["summary"].get(
            "compact_rollout_slab_learner_gate_prebuilt_batch_used"
        ),
        "compact_muzero_learner_prebuilt_batch_used": result["summary"].get(
            "compact_muzero_learner_prebuilt_batch_used"
        ),
        "compact_owned_loop_deferred_learner_submit_count": result["summary"].get(
            "compact_owned_loop_deferred_learner_submit_count"
        ),
        "compact_owned_loop_deferred_learner_completed_count": result["summary"].get(
            "compact_owned_loop_deferred_learner_completed_count"
        ),
        "compact_owned_loop_deferred_learner_pending": result["summary"].get(
            "compact_owned_loop_deferred_learner_pending"
        ),
        "compact_owned_loop_deferred_learner_pending_count": result["summary"].get(
            "compact_owned_loop_deferred_learner_pending_count"
        ),
        "compact_owned_loop_deferred_learner_max_pending": result["summary"].get(
            "compact_owned_loop_deferred_learner_max_pending"
        ),
        "compact_owned_loop_deferred_learner_max_pending_observed": result["summary"].get(
            "compact_owned_loop_deferred_learner_max_pending_observed"
        ),
        "compact_owned_loop_deferred_learner_actor_steps_while_pending": result["summary"].get(
            "compact_owned_loop_deferred_learner_actor_steps_while_pending"
        ),
        "compact_owned_loop_deferred_learner_policy_lag_current": result["summary"].get(
            "compact_owned_loop_deferred_learner_policy_lag_current"
        ),
        "compact_owned_loop_deferred_learner_policy_lag_max": result["summary"].get(
            "compact_owned_loop_deferred_learner_policy_lag_max"
        ),
        "compact_owned_loop_deferred_learner_wait_count": result["summary"].get(
            "compact_owned_loop_deferred_learner_wait_count"
        ),
        "compact_owned_loop_deferred_learner_wait_sec": result["summary"].get(
            "compact_owned_loop_deferred_learner_wait_sec"
        ),
        "compact_owned_loop_deferred_learner_last_wait_sec": result["summary"].get(
            "compact_owned_loop_deferred_learner_last_wait_sec"
        ),
        "compact_owned_loop_deferred_sample_learner_submit_count": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_submit_count"
        ),
        "compact_owned_loop_sample_learner_worker_kind": result["summary"].get(
            "compact_owned_loop_sample_learner_worker_kind"
        ),
        "compact_owned_loop_sample_learner_worker_resource_id": result["summary"].get(
            "compact_owned_loop_sample_learner_worker_resource_id"
        ),
        "compact_owned_loop_actor_search_resource_id": result["summary"].get(
            "compact_owned_loop_actor_search_resource_id"
        ),
        "compact_owned_loop_actor_search_pid": result["summary"].get(
            "compact_owned_loop_actor_search_pid"
        ),
        "compact_owned_loop_sample_learner_worker_parent_pid": result["summary"].get(
            "compact_owned_loop_sample_learner_worker_parent_pid"
        ),
        "compact_owned_loop_sample_learner_worker_resource_scope": result["summary"].get(
            "compact_owned_loop_sample_learner_worker_resource_scope"
        ),
        "compact_owned_loop_sample_learner_worker_start_method": result["summary"].get(
            "compact_owned_loop_sample_learner_worker_start_method"
        ),
        "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings": result[
            "summary"
        ].get("compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings"),
        "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": result[
            "summary"
        ].get("compact_owned_loop_sample_learner_resource_distinct_from_actor_search"),
        "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search": result[
            "summary"
        ].get("compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search"),
        "compact_owned_loop_deferred_sample_learner_completed_count": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_completed_count"
        ),
        "compact_owned_loop_deferred_sample_learner_pending": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_pending"
        ),
        "compact_owned_loop_deferred_sample_learner_pending_count": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_pending_count"
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_max_pending"
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_observed": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_max_pending_observed"
        ),
        "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending": result[
            "summary"
        ].get("compact_owned_loop_deferred_sample_learner_actor_steps_while_pending"),
        "compact_owned_loop_deferred_sample_learner_policy_lag_current": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_policy_lag_current"
        ),
        "compact_owned_loop_deferred_sample_learner_policy_lag_max": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_policy_lag_max"
        ),
        "compact_owned_loop_deferred_sample_learner_last_submitted_request_id": result[
            "summary"
        ].get("compact_owned_loop_deferred_sample_learner_last_submitted_request_id"),
        "compact_owned_loop_deferred_sample_learner_last_completed_request_id": result[
            "summary"
        ].get("compact_owned_loop_deferred_sample_learner_last_completed_request_id"),
        "compact_owned_loop_deferred_sample_learner_last_submitted_snapshot_version": result[
            "summary"
        ].get("compact_owned_loop_deferred_sample_learner_last_submitted_snapshot_version"),
        "compact_owned_loop_deferred_sample_learner_last_completed_snapshot_version": result[
            "summary"
        ].get("compact_owned_loop_deferred_sample_learner_last_completed_snapshot_version"),
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid": result[
            "summary"
        ].get("compact_owned_loop_deferred_sample_learner_last_completed_worker_pid"),
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id": result[
            "summary"
        ].get("compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id"),
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device": result[
            "summary"
        ].get("compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device"),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "last_completed_worker_pid_distinct_from_actor_search"
        ): result["summary"].get(
            (
                "compact_owned_loop_deferred_sample_learner_"
                "last_completed_worker_pid_distinct_from_actor_search"
            )
        ),
        "compact_owned_loop_deferred_sample_learner_model_state_apply_count": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_model_state_apply_count"
        ),
        "compact_owned_loop_deferred_sample_learner_last_model_state_applied": result[
            "summary"
        ].get("compact_owned_loop_deferred_sample_learner_last_model_state_applied"),
        ("compact_owned_loop_deferred_sample_learner_replay_append_transport_kind"): result[
            "summary"
        ].get(("compact_owned_loop_deferred_sample_learner_replay_append_transport_kind")),
        "compact_owned_loop_deferred_sample_learner_wait_count": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_wait_count"
        ),
        "compact_owned_loop_deferred_sample_learner_wait_sec": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_wait_sec"
        ),
        "compact_owned_loop_deferred_sample_learner_last_wait_sec": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_last_wait_sec"
        ),
        "compact_owned_loop_deferred_sample_learner_drained": result["summary"].get(
            "compact_owned_loop_deferred_sample_learner_drained"
        ),
        "compact_owned_loop_final_deferred_drain_sec": result["summary"].get(
            "compact_owned_loop_final_deferred_drain_sec"
        ),
        "compact_owned_loop_final_deferred_sample_learner_drain_sec": result["summary"].get(
            "compact_owned_loop_final_deferred_sample_learner_drain_sec"
        ),
        "compact_owned_loop_final_deferred_learner_drain_sec": result["summary"].get(
            "compact_owned_loop_final_deferred_learner_drain_sec"
        ),
        "compact_owned_loop_final_deferred_drain_in_measured_sec": result["summary"].get(
            "compact_owned_loop_final_deferred_drain_in_measured_sec"
        ),
        "compact_torch_observation_memory_format": result["summary"].get(
            "compact_torch_observation_memory_format"
        ),
        "compact_torch_initial_inference_mode": result["summary"].get(
            "compact_torch_initial_inference_mode"
        ),
        "compact_torch_model_memory_format": result["summary"].get(
            "compact_torch_model_memory_format"
        ),
        "compact_torch_defer_one_simulation_replay_payload_requested": result["summary"].get(
            "compact_torch_defer_one_simulation_replay_payload_requested"
        ),
        "compact_torch_memory_format_applies_to_search_service": result["summary"].get(
            "compact_torch_memory_format_applies_to_search_service"
        ),
        "hybrid_persistent_compact_render_state_buffer": result["summary"].get(
            "hybrid_persistent_compact_render_state_buffer"
        ),
        "hybrid_borrow_single_actor_render_state": result["summary"].get(
            "hybrid_borrow_single_actor_render_state"
        ),
        "render_state_handoff_mode": result["summary"].get("render_state_handoff_mode"),
        "render_state_copy_steps": result["summary"].get("render_state_copy_steps"),
        "render_state_borrowed_steps": result["summary"].get("render_state_borrowed_steps"),
        "render_state_row_overlay_steps": result["summary"].get("render_state_row_overlay_steps"),
        "render_state_row_overlay_rows": result["summary"].get("render_state_row_overlay_rows"),
        "render_state_row_overlay_bytes": result["summary"].get("render_state_row_overlay_bytes"),
        "actor_count": result["summary"].get("actor_count"),
        "batch_size": result["summary"].get("batch_size"),
        "steps": result["summary"].get("steps"),
        "warmup_steps": result["summary"].get("warmup_steps"),
        "death_mode": result["summary"].get("death_mode"),
        "compact_owned_trainer_config_death_mode": result["summary"].get(
            "compact_owned_trainer_config_death_mode"
        ),
        "normal_death_terminal_contract_owner": result["summary"].get(
            "normal_death_terminal_contract_owner"
        ),
        "terminal_row_count": result["summary"].get("terminal_row_count"),
        "death_row_count": result["summary"].get("death_row_count"),
        "terminated_row_count": result["summary"].get("terminated_row_count"),
        "truncated_row_count": result["summary"].get("truncated_row_count"),
        "env_action_checksum_total": result["summary"].get("env_action_checksum_total"),
        "env_done_checksum_total": result["summary"].get("env_done_checksum_total"),
        "env_reward_checksum_total": result["summary"].get("env_reward_checksum_total"),
        "env_action_mask_checksum_total": result["summary"].get("env_action_mask_checksum_total"),
        "env_trajectory_checksum_total": result["summary"].get("env_trajectory_checksum_total"),
        "env_trajectory_ordered_checksum_total": result["summary"].get(
            "env_trajectory_ordered_checksum_total"
        ),
        "env_terminal_row_checksum_total": result["summary"].get("env_terminal_row_checksum_total"),
        "env_autoreset_row_checksum_total": result["summary"].get(
            "env_autoreset_row_checksum_total"
        ),
        "env_terminal_reason_checksum_total": result["summary"].get(
            "env_terminal_reason_checksum_total"
        ),
        "env_death_count_checksum_total": result["summary"].get("env_death_count_checksum_total"),
        "env_death_cause_checksum_total": result["summary"].get("env_death_cause_checksum_total"),
        "env_death_hit_owner_checksum_total": result["summary"].get(
            "env_death_hit_owner_checksum_total"
        ),
        "last_env_action_checksum": result["summary"].get("last_env_action_checksum"),
        "last_env_trajectory_checksum": result["summary"].get("last_env_trajectory_checksum"),
        "last_env_terminal_row_checksum": result["summary"].get("last_env_terminal_row_checksum"),
        "last_env_autoreset_row_checksum": result["summary"].get("last_env_autoreset_row_checksum"),
        "terminal_sample_row_count": result["summary"].get("terminal_sample_row_count"),
        "terminal_unroll_value_target_mode": result["summary"].get(
            "terminal_unroll_value_target_mode"
        ),
        "terminal_unroll_value_target_row_count": result["summary"].get(
            "terminal_unroll_value_target_row_count"
        ),
        "normal_death_terminal_contract_promotion_gate_satisfied": (
            result["summary"].get("normal_death_terminal_contract_promotion_gate_satisfied")
        ),
        "resident_observation_host_fallback_count": result["summary"].get(
            "resident_observation_host_fallback_count"
        ),
        "compact_profile_autoreset_direct_count": result["summary"].get(
            "compact_profile_autoreset_direct_count"
        ),
        "compact_profile_autoreset_template_copy_skipped_count": result["summary"].get(
            "compact_profile_autoreset_template_copy_skipped_count"
        ),
        "compact_profile_autoreset_direct_row_count": result["summary"].get(
            "compact_profile_autoreset_direct_row_count"
        ),
        "compact_rollout_slab_sample_gate_action_checksum": result["summary"].get(
            "compact_rollout_slab_sample_gate_action_checksum"
        ),
        "compact_rollout_slab_sample_gate_sample_row_checksum": result["summary"].get(
            "compact_rollout_slab_sample_gate_sample_row_checksum"
        ),
        "compact_rollout_slab_sample_gate_sample_action_checksum": result["summary"].get(
            "compact_rollout_slab_sample_gate_sample_action_checksum"
        ),
        "compact_rollout_slab_sample_gate_sampled_flat_row_checksum": result["summary"].get(
            "compact_rollout_slab_sample_gate_sampled_flat_row_checksum"
        ),
        "compact_rollout_slab_sample_gate_sample_position_order_checksum": (
            result["summary"].get("compact_rollout_slab_sample_gate_sample_position_order_checksum")
        ),
        "compact_rollout_slab_sample_gate_source_record_pair_checksum": result["summary"].get(
            "compact_rollout_slab_sample_gate_source_record_pair_checksum"
        ),
        "compact_rollout_slab_sample_gate_source_record_window_checksum": result["summary"].get(
            "compact_rollout_slab_sample_gate_source_record_window_checksum"
        ),
        **_gpu_utilization_report_fields(result["summary"]),
        "learner_num_unroll_steps": result["summary"].get("learner_num_unroll_steps"),
        "model_identity_scope": result["compact"]["model_identity_scope"],
        "profile_support_profile_only": result["summary"]["source_profile_support_profile_only"],
        "real_compact_owned_training_work": result["compact"]["real_compact_owned_training_work"],
        "promotion_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    report.update(_sample_learner_timer_report_fields(result["summary"]))
    report.update(_compact_rollout_slab_total_report_fields(result["summary"]))
    report.update(_sample_learner_transport_proof_report_fields(result["summary"]))
    report.update(_owner_search_slab_proxy_proof_report_fields(result["summary"]))
    report_path = output_dir / "compact_coach_speed_row_modal_report.json"
    _write_json(report_path, report)
    print(json.dumps({"ok": True, "report_path": str(report_path)}, sort_keys=True))
    return 0


def _sample_learner_timer_report_fields(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: summary.get(field)
        for field in _SAMPLE_LEARNER_TIMER_REPORT_FIELDS
        if field in summary
    }


def _write_launch_failure_report(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    launch_payload: dict[str, Any],
) -> Path:
    launch_path = output_dir / "launch.json"
    _write_json(launch_path, launch_payload)
    report_path = output_dir / "compact_coach_speed_row_modal_report.json"
    failure = {
        "ok": False,
        "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
        "run_id": str(args.run_id),
        "function_call_id": "",
        "problem": str(launch_payload.get("problem") or "remote launch failed"),
        "failure_stage": "launch",
        "launch_path": str(launch_path),
        "launch_schema_id": launch_payload.get("schema_id"),
        "launch_status": launch_payload.get("status"),
        "modal_launch_returncode": launch_payload.get("returncode"),
        "modal_launch_error_code": str(launch_payload.get("modal_error_code") or ""),
        "modal_launch_resource_exhausted": bool(
            launch_payload.get("modal_resource_exhausted", False)
        ),
        "modal_launch_stdout_tail": str(launch_payload.get("stdout_tail") or ""),
        "modal_launch_stderr_tail": str(launch_payload.get("stderr_tail") or ""),
        "compact_torch_observation_memory_format": str(
            args.compact_torch_observation_memory_format
        ),
        "compact_torch_initial_inference_mode": str(args.compact_torch_initial_inference_mode),
        "compact_torch_model_memory_format": str(args.compact_torch_model_memory_format),
        "compact_torch_defer_one_simulation_replay_payload_requested": bool(
            args.compact_torch_defer_one_simulation_replay_payload
        ),
        "compact_torch_memory_format_applies_to_search_service": (
            _uses_compact_torch_search_service(args)
        ),
        **_owner_search_config_fields(args),
        "hybrid_persistent_compact_render_state_buffer": bool(
            args.hybrid_persistent_compact_render_state_buffer
        ),
        "hybrid_borrow_single_actor_render_state": bool(
            args.hybrid_borrow_single_actor_render_state
        ),
        "death_mode": str(args.death_mode),
        "learner_num_unroll_steps": int(args.learner_num_unroll_steps),
        "compact_owned_loop_deferred_learner": bool(args.compact_owned_loop_deferred_learner),
        "compact_owned_loop_deferred_sample_learner": bool(
            args.compact_owned_loop_deferred_sample_learner
        ),
        "compact_owned_loop_fused_learner_batch": bool(args.compact_owned_loop_fused_learner_batch),
        "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
            args.compact_muzero_learner_batch_unroll2_specialized_builder
        ),
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
            getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ),
        "compact_owned_accepted_fast_path_preset": bool(
            getattr(args, "compact_owned_accepted_fast_path_preset", False)
        ),
        "compact_owned_accepted_fast_path_preset_name": (
            ACCEPTED_FAST_PATH_PRESET
            if bool(getattr(args, "compact_owned_accepted_fast_path_preset", False))
            else ""
        ),
        **_accepted_fast_path_step_window_report_fields(args),
        "compact_profile_bounded_diagnostics": bool(
            getattr(args, "compact_profile_bounded_diagnostics", False)
        ),
        "compact_profile_cuda_sync_timing_diagnostics": bool(
            getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
        ),
        "compact_profile_runtime_step_timing_diagnostics": bool(
            getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
            or getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
        ),
        "compact_profile_cpu_perf_stat_diagnostics": bool(
            getattr(args, "compact_profile_cpu_perf_stat_diagnostics", False)
        ),
        "compact_owned_lean_trainer_step": bool(args.compact_owned_lean_trainer_step),
        "compact_owned_lean_profile_oracle_requested": bool(args.compact_owned_lean_profile_oracle),
        "speed_row_gpu_utilization_sampling_enabled": bool(args.gpu_utilization_sampling),
        "speed_row_gpu_utilization_sample_interval_sec": float(
            args.gpu_utilization_sample_interval_sec
        ),
        "promotion_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    _write_json(report_path, failure)
    return report_path


def _compact_rollout_slab_total_report_fields(summary: dict[str, Any]) -> dict[str, Any]:
    fields = {
        key: value for key, value in summary.items() if str(key).startswith("speed_row_total_")
    }
    if "compact_rollout_slab_telemetry_totals" in summary:
        fields["compact_rollout_slab_telemetry_totals"] = summary.get(
            "compact_rollout_slab_telemetry_totals"
        )
    return fields


def _sample_learner_transport_proof_report_fields(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: summary.get(field)
        for field in _SAMPLE_LEARNER_TRANSPORT_PROOF_REPORT_FIELDS
        if field in summary
    }


def _owner_search_slab_proxy_proof_report_fields(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: summary.get(field)
        for field in _OWNER_SEARCH_SLAB_PROXY_PROOF_REPORT_FIELDS
        if field in summary
    }


def _gpu_utilization_report_fields(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: summary.get(field) for field in _GPU_UTILIZATION_REPORT_FIELDS if field in summary
    }


def _resolve_checkpoint_path(
    args: argparse.Namespace,
    lifecycle: dict[str, Any],
    repo_root: Path,
) -> Path:
    raw = args.compact_checkpoint or Path(str(lifecycle.get("compact_checkpoint_path") or ""))
    if not str(raw):
        raise SystemExit("compact checkpoint path required")
    path = _resolve_path(Path(raw), repo_root)
    if not path.is_file():
        raise SystemExit(f"compact checkpoint not found: {path}")
    return path


def _launch_remote(
    args: argparse.Namespace,
    lifecycle_ref: str,
    checkpoint_ref: str,
) -> dict[str, Any]:
    _apply_accepted_fast_path_preset(args)
    if bool(getattr(args, "owner_search_fixed_action_result_buffer", False)) and not bool(
        getattr(args, "owner_search_defer_maintenance", False)
    ):
        raise ValueError(
            "--owner-search-fixed-action-result-buffer requires --owner-search-defer-maintenance"
        )
    command = [
        "uv",
        "run",
        "--extra",
        "modal",
        "modal",
        "run",
        "--detach",
        "-m",
        MODULE,
        "--speed-row-spawn-result",
        "--run-id",
        str(args.run_id),
        "--unified-lifecycle-report-ref",
        lifecycle_ref,
        "--compact-checkpoint-ref",
        checkpoint_ref,
        "--batch-size",
        str(int(args.batch_size)),
        "--actor-count",
        str(int(args.actor_count)),
        "--steps",
        str(int(args.steps)),
        "--warmup-steps",
        str(int(args.warmup_steps)),
        "--death-mode",
        str(args.death_mode),
        "--sample-batch-size",
        str(int(args.sample_batch_size)),
        "--sample-interval",
        str(int(args.sample_interval)),
        "--replay-pair-capacity",
        str(int(args.replay_pair_capacity)),
        "--learner-train-steps",
        str(int(args.learner_train_steps)),
        "--learner-num-unroll-steps",
        str(int(args.learner_num_unroll_steps)),
        "--policy-refresh-interval",
        str(int(args.policy_refresh_interval)),
        "--learner-device",
        str(args.learner_device),
        "--num-simulations",
        str(int(args.num_simulations)),
        "--search-service-kind",
        str(args.search_service_kind),
        "--owner-search-inner-search-service-kind",
        str(args.owner_search_inner_search_service_kind),
        "--seed",
        str(int(args.seed)),
        "--source-max-steps",
        str(int(args.source_max_steps)),
        "--decision-source-frames",
        str(int(args.decision_source_frames)),
        "--source-physics-step-ms",
        str(float(args.source_physics_step_ms)),
        "--source-max-steps-semantics",
        str(args.source_max_steps_semantics),
    ]
    if bool(args.hybrid_persistent_compact_render_state_buffer):
        command.append("--hybrid-persistent-compact-render-state-buffer")
    if bool(args.hybrid_borrow_single_actor_render_state):
        command.append("--hybrid-borrow-single-actor-render-state")
    if bool(getattr(args, "owner_search_defer_maintenance", False)):
        command.append("--owner-search-defer-maintenance")
    if bool(getattr(args, "owner_search_slab_bypass", False)):
        command.append("--owner-search-slab-bypass")
    if int(getattr(args, "owner_search_transition_batch_size", 1)) > 1:
        command.extend(
            [
                "--owner-search-transition-batch-size",
                str(int(args.owner_search_transition_batch_size)),
            ]
        )
    if bool(getattr(args, "owner_search_direct_transition_batch_replay", False)):
        command.append("--owner-search-direct-transition-batch-replay")
    if bool(getattr(args, "owner_search_owner_local_transition_derivation", False)):
        command.append("--owner-search-owner-local-transition-derivation")
    if bool(getattr(args, "owner_search_owner_proxy_transition_closure", False)):
        command.append("--owner-search-owner-proxy-transition-closure")
    if bool(getattr(args, "owner_search_require_resident_root_view", False)):
        command.append("--owner-search-require-resident-root-view")
    if bool(getattr(args, "owner_search_resident_root_host_observation_stub", False)):
        command.append("--owner-search-resident-root-host-observation-stub")
    if bool(getattr(args, "owner_search_direct_root_build_request", False)):
        command.append("--owner-search-direct-root-build-request")
    if bool(getattr(args, "compact_owner_action_step_boundary", False)):
        command.append("--compact-owner-action-step-boundary")
    if bool(getattr(args, "compact_owner_action_dispatch_step_overlap", False)):
        command.append("--compact-owner-action-dispatch-step-overlap")
    if bool(getattr(args, "owner_search_fixed_action_result_buffer", False)):
        command.append("--owner-search-fixed-action-result-buffer")
        command.extend(
            [
                "--owner-search-action-result-slot-capacity",
                str(int(getattr(args, "owner_search_action_result_slot_capacity", 4) or 4)),
            ]
        )
    if bool(getattr(args, "owner_search_fixed_soa_replay", False)):
        command.append("--owner-search-fixed-soa-replay")
    fixed_soa_locality_group_size = int(
        getattr(args, "owner_search_fixed_soa_locality_sample_group_size", 1) or 1
    )
    if fixed_soa_locality_group_size != 1:
        command.extend(
            [
                "--owner-search-fixed-soa-locality-sample-group-size",
                str(fixed_soa_locality_group_size),
            ]
        )
    if bool(getattr(args, "owner_search_defer_model_state_digest_to_refresh", False)):
        command.append("--owner-search-defer-model-state-digest-to-refresh")
    if bool(getattr(args, "owner_search_async_learner_worker", False)):
        command.append("--owner-search-async-learner-worker")
        command.extend(
            [
                "--owner-search-async-learner-worker-kind",
                str(args.owner_search_async_learner_worker_kind),
                "--owner-search-async-learner-max-pending",
                str(int(args.owner_search_async_learner_max_pending)),
            ]
        )
    if bool(args.compact_owned_loop_deferred_learner):
        command.append("--compact-owned-loop-deferred-learner")
    if bool(args.compact_owned_loop_deferred_sample_learner):
        replay_append_transport_kind = getattr(
            args,
            "compact_owned_loop_deferred_sample_learner_replay_append_transport_kind",
            COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1,
        )
        model_state_transport_kind = getattr(
            args,
            "compact_owned_loop_deferred_sample_learner_model_state_transport_kind",
            COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
        )
        command.append("--compact-owned-loop-deferred-sample-learner")
        command.extend(
            [
                "--compact-owned-loop-deferred-sample-learner-max-pending",
                str(int(args.compact_owned_loop_deferred_sample_learner_max_pending)),
                "--compact-owned-loop-sample-learner-worker-kind",
                str(args.compact_owned_loop_sample_learner_worker_kind),
                "--compact-owned-loop-deferred-sample-learner-replay-append-transport-kind",
                str(replay_append_transport_kind),
                "--compact-owned-loop-deferred-sample-learner-model-state-transport-kind",
                str(model_state_transport_kind),
            ]
        )
    if bool(args.compact_owned_loop_fused_learner_batch):
        command.append("--compact-owned-loop-fused-learner-batch")
    if bool(args.compact_muzero_learner_batch_unroll2_specialized_builder):
        command.append("--compact-muzero-learner-batch-unroll2-specialized-builder")
    if bool(getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)):
        command.append("--compact-muzero-learner-batch-learner-ready-unroll2-cache")
    if bool(getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)):
        command.append("--compact-muzero-learner-batch-tensor-native-replay")
    if bool(args.compact_owned_lean_trainer_step):
        command.append("--compact-owned-lean-trainer-step")
    if bool(args.compact_owned_lean_profile_oracle):
        command.append("--compact-owned-lean-profile-oracle")
    if bool(getattr(args, "compact_profile_bounded_diagnostics", False)):
        command.append("--compact-profile-bounded-diagnostics")
    if bool(getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)):
        command.append("--compact-profile-cuda-sync-timing-diagnostics")
    if bool(getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)):
        command.append("--compact-profile-runtime-step-timing-diagnostics")
    if bool(getattr(args, "compact_profile_cpu_perf_stat_diagnostics", False)):
        command.append("--compact-profile-cpu-perf-stat-diagnostics")
    if bool(args.gpu_utilization_sampling):
        command.append("--gpu-utilization-sampling")
    command.extend(
        [
            "--gpu-utilization-sample-interval-sec",
            str(float(args.gpu_utilization_sample_interval_sec)),
        ]
    )
    if bool(args.compact_torch_request_compile):
        command.append("--compact-torch-request-compile")
    if bool(args.compact_torch_request_model_compile):
        command.append("--compact-torch-request-model-compile")
    command.extend(
        [
            "--compact-torch-model-compile-mode",
            str(args.compact_torch_model_compile_mode),
        ]
    )
    command.extend(["--compact-torch-timing-mode", str(args.compact_torch_timing_mode)])
    command.extend(
        [
            "--compact-torch-initial-inference-mode",
            str(args.compact_torch_initial_inference_mode),
        ]
    )
    command.extend(
        [
            "--compact-torch-observation-memory-format",
            str(args.compact_torch_observation_memory_format),
        ]
    )
    command.extend(
        [
            "--compact-torch-model-memory-format",
            str(args.compact_torch_model_memory_format),
        ]
    )
    if bool(getattr(args, "compact_torch_defer_one_simulation_replay_payload", False)):
        command.append("--compact-torch-defer-one-simulation-replay-payload")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=float(args.launch_timeout_sec),
    )
    if completed.returncode != 0:
        stdout_tail = completed.stdout[-4000:]
        stderr_tail = completed.stderr[-4000:]
        combined_tail = f"{stdout_tail}\n{stderr_tail}"
        resource_exhausted = (
            "RESOURCE_EXHAUSTED" in combined_tail or "memory usage is too high" in combined_tail
        )
        raise ModalRemoteLaunchError(
            {
                "schema_id": SPAWN_SCHEMA_ID,
                "status": "failed",
                "result_capture": "modal_function_call_get",
                "failure_stage": "launch",
                "problem": (
                    "remote launch resource exhausted"
                    if resource_exhausted
                    else "remote launch failed"
                ),
                "modal_error_code": "RESOURCE_EXHAUSTED" if resource_exhausted else "",
                "returncode": int(completed.returncode),
                "command": command,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "modal_resource_exhausted": resource_exhausted,
            }
        )
    payload = _extract_last_json_object(completed.stdout + "\n" + completed.stderr)
    if payload.get("schema_id") != SPAWN_SCHEMA_ID:
        raise SystemExit(f"unexpected spawn schema: {payload.get('schema_id')}")
    function_call_id = str(payload.get("function_call_id") or "")
    status = str(payload.get("status") or ("spawned" if function_call_id else ""))
    if status != "spawned" or not function_call_id:
        combined_tail = (
            str(payload.get("stdout_tail") or "")
            + "\n"
            + str(payload.get("stderr_tail") or "")
            + "\n"
            + str(payload.get("problem") or "")
        )
        resource_exhausted = bool(payload.get("modal_resource_exhausted", False)) or (
            "RESOURCE_EXHAUSTED" in combined_tail or "memory usage is too high" in combined_tail
        )
        failure_payload = dict(payload)
        failure_payload.setdefault("failure_stage", "launch")
        failure_payload.setdefault("problem", "remote spawn failed")
        failure_payload["modal_resource_exhausted"] = resource_exhausted
        if resource_exhausted and not failure_payload.get("modal_error_code"):
            failure_payload["modal_error_code"] = "RESOURCE_EXHAUSTED"
        raise ModalRemoteLaunchError(failure_payload)
    return payload


def _modal_volume_put(local_path: Path, remote_ref: str) -> None:
    completed = subprocess.run(
        [
            "modal",
            "volume",
            "put",
            "--force",
            curvytron_runs_volume_name(),
            str(local_path),
            remote_ref,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5 * 60,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"modal volume put failed for {local_path}:\n"
            + completed.stdout[-4000:]
            + completed.stderr[-4000:]
        )


def _collect_modal_function_call(function_call_id: str, timeout: float) -> dict[str, Any]:
    if modal is None:
        raise SystemExit("modal extra is required to collect FunctionCall results")
    call = modal.FunctionCall.from_id(function_call_id)
    payload = call.get(timeout=float(timeout))
    if not isinstance(payload, dict):
        raise SystemExit("modal FunctionCall result was not a JSON object")
    return payload


def _extract_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    if not candidates:
        raise ValueError("no JSON object found in command output")
    for value in reversed(candidates):
        if value.get("schema_id") == SPAWN_SCHEMA_ID and value.get("function_call_id"):
            return value
    return candidates[-1]


def _required_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return value


def _safe_ref_part(raw: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    clean = "".join(char if char in allowed else "-" for char in str(raw)).strip("-.")
    if not clean:
        raise SystemExit(f"cannot make safe ref from {raw!r}")
    return clean


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
