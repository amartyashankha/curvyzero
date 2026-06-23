"""Dedicated Modal producer for compact Coach speed-row evidence."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import modal

from curvyzero.contracts.curvytron import curvytron_runs_volume_name
from curvyzero.contracts.curvytron import modal_volume_kwargs_for_name
from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.source_state_batched_observation_boundary_profile import (
    gpu_lightzero_image,
)
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
    COMPACT_COACH_SPEED_ROW_RESULT_SCHEMA_ID,
)
from curvyzero.training.compact_torch_search_service import (
    COMPACT_TORCH_INITIAL_INFERENCE_MODES,
    COMPACT_TORCH_MODEL_COMPILE_MODES,
    COMPACT_TORCH_MEMORY_FORMATS,
)


APP_NAME = "curvyzero-compact-coach-speed-row"
SPAWN_SCHEMA_ID = "curvyzero_compact_coach_speed_row_spawn/v0"
RESULT_BUNDLE_SCHEMA_ID = "curvyzero_compact_coach_speed_row_h100_bundle/v0"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
DEFAULT_RUN_ID = "optimizer-compact-coach-speed-row-h100-20260530"
SPEED_CURRENCY = "compact_trainer_env_steps_per_sec"
CPU_PERF_STAT_EVENTS = (
    "task-clock",
    "cycles",
    "ref-cycles",
    "instructions",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
    "LLC-loads",
    "LLC-load-misses",
    "dTLB-loads",
    "dTLB-load-misses",
    "page-faults",
    "context-switches",
    "cpu-migrations",
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
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_validate_per_call"
    ),
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
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "unroll_terminal_window_hint_per_call"
    ),
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
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_index_fast_path_count"
    ),
    "compact_rollout_slab_sample_gate_terminal_final_observation_fallback_count",
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_validate_only_count"
    ),
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_materialized_count"
    ),
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_final_row_count_sum"
    ),
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_final_row_count_max"
    ),
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_dense_storage_count"
    ),
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_sparse_storage_count"
    ),
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_missing_storage_count"
    ),
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_sparse_row_count_sum"
    ),
    (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_sparse_row_count_max"
    ),
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
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_validate_sec"
    ),
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
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "unroll_terminal_window_hint_sec"
    ),
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
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "cuda_sync_timing_diagnostics"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "cuda_sync_timing_enabled"
    ),
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
    "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled",
    "compact_rollout_slab_learner_gate_cuda_sync_count",
    "compact_rollout_slab_learner_gate_cuda_sync_sec",
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
_CUDA_SYNC_TIMING_DIAGNOSTIC_TRUE_FIELDS = (
    "compact_profile_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled",
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "cuda_sync_timing_diagnostics"
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "cuda_sync_timing_enabled"
    ),
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
_RUNTIME_STEP_TIMING_DIAGNOSTIC_TRUE_FIELDS = (
    "compact_profile_runtime_step_timing_diagnostics",
)
_RUNTIME_STEP_TIMING_DIAGNOSTIC_POSITIVE_FIELDS = (
    "compact_profile_runtime_step_count",
)
_RUNTIME_STEP_TIMING_DIAGNOSTIC_SEC_FIELDS = (
    "compact_profile_runtime_step_sum_sec",
    "compact_profile_runtime_step_min_sec",
    "compact_profile_runtime_step_max_sec",
    "compact_profile_runtime_step_p50_sec",
    "compact_profile_runtime_step_p95_sec",
)
_SAMPLE_LEARNER_TRANSPORT_PROOF_FIELDS = (
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
_OWNER_SEARCH_SLAB_PROXY_PROOF_FIELDS = (
    "compact_owner_search_slab_proxy",
    "compact_owner_search_lazy_slab_proxy",
    "compact_owner_search_inline_slab_proxy",
    "compact_owner_search_inline_background_slab_proxy",
    "compact_owner_search_threaded_slab_proxy",
    "compact_owner_search_slab_bypass",
    "compact_owner_search_slab_bypass_kind",
    "compact_rollout_slab_bypassed",
    "compact_rollout_slab_general_replay_row_builder_used",
    "compact_rollout_slab_retains_committed_index_rows",
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
    "compact_owner_search_owner_model_refresh_interval",
    "compact_owner_search_owner_expected_train_request_count",
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
    "compact_owner_search_owner_model_refresh_request_count",
    "compact_owner_search_owner_model_refresh_skipped_count",
    "compact_owner_search_owner_learner_update_count",
    "compact_owner_search_owner_pending_replay_append_entry_count",
    "compact_owner_search_owner_maintenance_drain_request_count",
    "compact_owner_search_owner_maintenance_staged_work_item_count",
    "compact_owner_search_owner_maintenance_drained_count",
    "compact_owner_search_owner_maintenance_drained_work_item_count",
    "compact_owner_search_owner_maintenance_drained_replay_append_entry_count",
    "compact_owner_search_owner_maintenance_drained_replay_append_transport_entry_count",
    (
        "compact_owner_search_owner_maintenance_drained_replay_append_"
        "transition_batch_count"
    ),
    (
        "compact_owner_search_owner_maintenance_drained_replay_append_"
        "transition_batch_entry_count"
    ),
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
    (
        "compact_owner_search_owner_async_learner_hardware_resource_distinct_"
        "from_owner"
    ),
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
    (
        "compact_owner_search_owner_async_learner_worker_pid_distinct_"
        "from_owner"
    ),
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


def _config_uses_compact_torch_search_service(config: Mapping[str, Any]) -> bool:
    search_kind = str(config.get("search_service_kind", "device_target"))
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
        and str(
            config.get(
                "owner_search_inner_search_service_kind",
                "compact_torch_search_service",
            )
        )
        == "compact_torch_search_service"
    )


def _owner_search_config_fields_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    search_kind = str(config.get("search_service_kind", "device_target"))
    owner_search = search_kind in {
        "owner_search_slab_proxy",
        "owner_search_inline_proxy",
        "owner_search_inline_background_proxy",
        "owner_search_threaded_proxy",
    }
    inline_owner_search = search_kind == "owner_search_inline_proxy"
    inline_background_owner_search = search_kind == "owner_search_inline_background_proxy"
    threaded_owner_search = search_kind == "owner_search_threaded_proxy"
    inner_kind = str(
        config.get(
            "owner_search_inner_search_service_kind",
            "compact_torch_search_service",
        )
    )
    return {
        "owner_search_slab_proxy_requested": owner_search,
        "owner_search_inline_proxy_requested": inline_owner_search,
        "owner_search_inline_background_proxy_requested": (
            inline_background_owner_search
        ),
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
            owner_search and config.get("owner_search_defer_maintenance", False)
        ),
        "owner_search_slab_bypass_requested": bool(
            owner_search and config.get("owner_search_slab_bypass", False)
        ),
        "owner_search_transition_batch_size_requested": (
            int(config.get("owner_search_transition_batch_size", 1))
            if owner_search
            else 1
        ),
        "owner_search_transition_batch_transport_requested": bool(
            owner_search
            and config.get("owner_search_slab_bypass", False)
            and int(config.get("owner_search_transition_batch_size", 1)) > 1
        ),
        "owner_search_direct_transition_batch_replay_requested": bool(
            owner_search
            and config.get("owner_search_direct_transition_batch_replay", False)
        ),
        "owner_search_owner_local_transition_derivation_requested": bool(
            owner_search
            and config.get("owner_search_owner_local_transition_derivation", False)
        ),
        "owner_search_owner_proxy_transition_closure_requested": bool(
            owner_search
            and config.get("owner_search_owner_proxy_transition_closure", False)
        ),
        "owner_search_require_resident_root_view_requested": bool(
            owner_search and config.get("owner_search_require_resident_root_view", False)
        ),
        "owner_search_resident_root_host_observation_stub_requested": bool(
            owner_search
            and config.get("owner_search_resident_root_host_observation_stub", False)
        ),
        "owner_search_direct_root_build_request_requested": bool(
            owner_search and config.get("owner_search_direct_root_build_request", False)
        ),
        "compact_owner_action_step_boundary_requested": bool(
            owner_search and config.get("compact_owner_action_step_boundary", False)
        ),
        "compact_owner_action_dispatch_step_overlap_requested": bool(
            owner_search and config.get("compact_owner_action_dispatch_step_overlap", False)
        ),
        "owner_search_fixed_action_result_buffer_requested": bool(
            owner_search and config.get("owner_search_fixed_action_result_buffer", False)
        ),
        "owner_search_action_result_slot_capacity_requested": (
            int(config.get("owner_search_action_result_slot_capacity", 4))
            if owner_search
            else 0
        ),
        "owner_search_fixed_soa_replay_requested": bool(
            owner_search and config.get("owner_search_fixed_soa_replay", False)
        ),
        "owner_search_defer_model_state_digest_to_refresh_requested": bool(
            owner_search
            and config.get("owner_search_defer_model_state_digest_to_refresh", False)
        ),
        "owner_search_fixed_soa_locality_sample_group_size_requested": (
            int(config.get("owner_search_fixed_soa_locality_sample_group_size", 1))
            if owner_search
            else 1
        ),
        "owner_search_async_learner_worker_requested": bool(
            owner_search and config.get("owner_search_async_learner_worker", False)
        ),
        "owner_search_async_learner_worker_kind_requested": (
            str(config.get("owner_search_async_learner_worker_kind", "in_process_thread_v1"))
            if owner_search
            else ""
        ),
        "owner_search_async_learner_max_pending_requested": (
            int(config.get("owner_search_async_learner_max_pending", 1))
            if owner_search
            else 0
        ),
    }


def _result_bundle_config_fields_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compact_torch_initial_inference_mode": str(
            config.get("compact_torch_initial_inference_mode", "model_method")
        ),
        "compact_torch_observation_memory_format": str(
            config.get("compact_torch_observation_memory_format", "contiguous")
        ),
        "compact_torch_model_memory_format": str(
            config.get("compact_torch_model_memory_format", "contiguous")
        ),
        "compact_torch_defer_one_simulation_replay_payload_requested": bool(
            config.get("compact_torch_defer_one_simulation_replay_payload", False)
        ),
        "compact_torch_memory_format_applies_to_search_service": (
            _config_uses_compact_torch_search_service(config)
        ),
        **_owner_search_config_fields_from_config(config),
        "hybrid_persistent_compact_render_state_buffer": bool(
            config.get("hybrid_persistent_compact_render_state_buffer", False)
        ),
        "hybrid_borrow_single_actor_render_state": bool(
            config.get("hybrid_borrow_single_actor_render_state", False)
        ),
        "death_mode": str(config.get("death_mode", "profile_no_death")),
        "learner_num_unroll_steps": int(config.get("learner_num_unroll_steps", 1)),
        "compact_owned_loop_deferred_learner": bool(
            config.get("compact_owned_loop_deferred_learner", False)
        ),
        "compact_owned_loop_deferred_sample_learner": bool(
            config.get("compact_owned_loop_deferred_sample_learner", False)
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_requested": int(
            config.get("compact_owned_loop_deferred_sample_learner_max_pending", 1)
        ),
        "compact_owned_loop_sample_learner_worker_kind_requested": str(
            config.get("compact_owned_loop_sample_learner_worker_kind", "in_process_thread")
        ),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_transport_kind_requested"
        ): str(
            config.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_transport_kind"
                ),
                "durable_entry_v1",
            )
        ),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "model_state_transport_kind_requested"
        ): str(
            config.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "model_state_transport_kind"
                ),
                "result_v1",
            )
        ),
        "compact_owned_loop_fused_learner_batch": bool(
            config.get("compact_owned_loop_fused_learner_batch", False)
        ),
        "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
            config.get("compact_muzero_learner_batch_unroll2_specialized_builder", False)
        ),
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
            config.get("compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ),
        "compact_muzero_learner_batch_tensor_native_replay": bool(
            config.get("compact_muzero_learner_batch_tensor_native_replay", False)
        ),
        "compact_owned_lean_trainer_step": bool(
            config.get("compact_owned_lean_trainer_step", False)
        ),
        "compact_owned_lean_profile_oracle_requested": bool(
            config.get("compact_owned_lean_profile_oracle", False)
        ),
        "compact_profile_bounded_diagnostics": bool(
            config.get("compact_profile_bounded_diagnostics", False)
        ),
        "compact_profile_cuda_sync_timing_diagnostics": bool(
            config.get("compact_profile_cuda_sync_timing_diagnostics", False)
        ),
        "compact_profile_runtime_step_timing_diagnostics": bool(
            config.get("compact_profile_runtime_step_timing_diagnostics", False)
            or config.get("compact_profile_cuda_sync_timing_diagnostics", False)
        ),
        "compact_profile_cpu_perf_stat_diagnostics": bool(
            config.get("compact_profile_cpu_perf_stat_diagnostics", False)
        ),
        "compact_owned_training_loop_owner": (
            "lean_compact_trainer_step"
            if bool(config.get("compact_owned_lean_trainer_step", False))
            else "hybrid_observation_profile_runner"
        ),
        "speed_row_gpu_utilization_sampling_enabled": bool(
            config.get("gpu_utilization_sampling", False)
        ),
        "speed_row_gpu_utilization_sample_interval_sec": float(
            config.get("gpu_utilization_sample_interval_sec", 1.0)
        ),
    }


def _perf_event_field_name(event_name: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(event_name).lower())
    return "_".join(part for part in safe.split("_") if part)


def _parse_perf_float(raw_value: str) -> float | None:
    text = str(raw_value).strip().replace(",", "")
    if not text or text.startswith("<"):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _parse_perf_stat_csv(stderr_text: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "compact_profile_cpu_perf_stat_diagnostics": True,
        "compact_profile_cpu_perf_stat_parse_line_count": 0,
        "compact_profile_cpu_perf_stat_parsed_event_count": 0,
        "compact_profile_cpu_perf_stat_events_requested": list(CPU_PERF_STAT_EVENTS),
    }
    parsed_events: list[str] = []
    for raw_line in str(stderr_text or "").splitlines():
        line = raw_line.strip()
        if not line or "," not in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        raw_value, unit, event = parts[0], parts[1], parts[2]
        if event not in CPU_PERF_STAT_EVENTS:
            continue
        fields["compact_profile_cpu_perf_stat_parse_line_count"] += 1
        value = _parse_perf_float(raw_value)
        event_field = _perf_event_field_name(event)
        fields[f"compact_profile_cpu_perf_stat_{event_field}_unit"] = unit
        if value is None:
            fields[f"compact_profile_cpu_perf_stat_{event_field}_available"] = False
            continue
        fields[f"compact_profile_cpu_perf_stat_{event_field}_available"] = True
        fields[f"compact_profile_cpu_perf_stat_{event_field}"] = value
        parsed_events.append(event)
        if event == "task-clock":
            unit_lower = unit.lower()
            if unit_lower in {"msec", "ms"}:
                fields["compact_profile_cpu_perf_stat_task_clock_sec"] = value / 1000.0
            elif unit_lower in {"sec", "s"}:
                fields["compact_profile_cpu_perf_stat_task_clock_sec"] = value
    fields["compact_profile_cpu_perf_stat_parsed_event_count"] = len(parsed_events)
    fields["compact_profile_cpu_perf_stat_parsed_events"] = parsed_events
    instructions = fields.get("compact_profile_cpu_perf_stat_instructions")
    cycles = fields.get("compact_profile_cpu_perf_stat_cycles")
    if isinstance(instructions, (int, float)) and isinstance(cycles, (int, float)) and cycles > 0:
        fields["compact_profile_cpu_perf_stat_instructions_per_cycle"] = (
            float(instructions) / float(cycles)
        )
    cache_misses = fields.get("compact_profile_cpu_perf_stat_cache_misses")
    cache_refs = fields.get("compact_profile_cpu_perf_stat_cache_references")
    if (
        isinstance(cache_misses, (int, float))
        and isinstance(cache_refs, (int, float))
        and cache_refs > 0
    ):
        fields["compact_profile_cpu_perf_stat_cache_miss_rate"] = (
            float(cache_misses) / float(cache_refs)
        )
    return fields


def _cpu_perf_stat_report_fields(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if str(key).startswith("compact_profile_cpu_perf_stat_")
    }


def _try_load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _load_json(path)
    except Exception as exc:  # pragma: no cover - diagnostic fallback.
        return {
            "_load_problem": f"{type(exc).__name__}: {exc}",
            "_path": str(path),
        }


def _speed_row_failure_bundle(
    *,
    config: Mapping[str, Any],
    run_id: str,
    rc: int,
    result_dir: Path,
) -> dict[str, Any]:
    result_path = result_dir / "row_001_result.json"
    report_path = result_dir / "compact_coach_speed_row_smoke_report.json"
    result = _try_load_json(result_path)
    report = _try_load_json(report_path)
    problem = str(
        report.get("problem")
        or result.get("problem")
        or f"speed-row producer exited with {int(rc)}"
    )
    return {
        "schema_id": RESULT_BUNDLE_SCHEMA_ID,
        "ok": False,
        "status": "failed",
        "problem": problem,
        "run_id": run_id,
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
        "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        "speed_currency": SPEED_CURRENCY,
        "producer_return_code": int(rc),
        "producer_result_path": str(result_path),
        "producer_report_path": str(report_path),
        "producer_result_present": bool(result_path.is_file()),
        "producer_report_present": bool(report_path.is_file()),
        "producer_failure_result": result,
        "producer_failure_report": report,
        **_result_bundle_config_fields_from_config(config),
    }


def _cuda_sync_timing_diagnostic_violations(
    *,
    requested: bool,
    summary: Mapping[str, Any],
) -> list[str]:
    if not requested:
        return []
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
            violations.append(f"summary.{field}: expected finite nonnegative seconds, got {raw_value!r}")
            continue
        if not math.isfinite(value) or value < 0.0:
            violations.append(f"summary.{field}: expected finite nonnegative seconds, got {raw_value!r}")
    return violations


def _runtime_step_timing_diagnostic_violations(
    *,
    requested: bool,
    summary: Mapping[str, Any],
) -> list[str]:
    if not requested:
        return []
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


def _remote_fused_learner_batch_violations(
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> list[str]:
    if not bool(config.get("compact_owned_loop_fused_learner_batch", False)):
        return []
    required_fused = (
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only",
        "compact_rollout_slab_learner_gate_prebuilt_batch_used",
    )
    violations = [key for key in required_fused if summary.get(key) is not True]
    resident_fused = bool(
        summary.get("compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch")
        and summary.get(
            "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"
        )
    )
    host_provider_fused = bool(
        summary.get("compact_rollout_slab_sample_gate_host_provider_learner_batch")
    )
    tensor_native_fused = (
        summary.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
        )
        is True
    )
    if not resident_fused and not host_provider_fused and not tensor_native_fused:
        violations.append(
            "resident_grouped_device_learner_batch or host_provider_learner_batch "
            "or tensor_native_replay"
        )
    return violations


def _tensor_native_replay_violations(
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> list[str]:
    if not bool(config.get("compact_muzero_learner_batch_tensor_native_replay", False)):
        return []
    violations: list[str] = []
    if not bool(config.get("compact_owned_loop_fused_learner_batch", False)):
        violations.append("config.compact_owned_loop_fused_learner_batch: expected True")
    if not bool(
        config.get("compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
    ):
        violations.append(
            "config.compact_muzero_learner_batch_learner_ready_unroll2_cache: expected True"
        )
    learner_ready_impl_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "learner_ready_unroll2_cache_impl"
    )
    learner_ready_path_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path"
    )
    tensor_native_impl_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "tensor_native_replay_impl"
    )
    tensor_native_table_source_field = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "tensor_native_replay_table_source"
    )
    tensor_native_impl = str(summary.get(tensor_native_impl_field) or "")
    tensor_native_table_source = str(summary.get(tensor_native_table_source_field) or "")
    fixed_soa_tensor_native = (
        tensor_native_impl == "fixed_soa_direct_gather_v1"
        and tensor_native_table_source == "fixed_soa_columns_v1"
    )
    for field in (
        "compact_muzero_learner_batch_learner_ready_unroll2_cache",
        "compact_muzero_learner_batch_tensor_native_replay",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used",
    ):
        if summary.get(field) is not True:
            violations.append(f"summary.{field}: expected True, got {summary.get(field)!r}")
    try:
        learner_num_unroll_steps = int(summary.get("learner_num_unroll_steps"))
    except (TypeError, ValueError):
        learner_num_unroll_steps = 0
    if learner_num_unroll_steps != 2:
        violations.append(
            "summary.learner_num_unroll_steps: expected 2, "
            f"got {summary.get('learner_num_unroll_steps')!r}"
        )
    learner_ready_count_fields = (
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
        if (
            summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used"
            )
            is not False
        ):
            violations.append(
                "summary.compact_rollout_slab_sample_gate_learner_batch_builder_"
                "learner_ready_unroll2_cache_used: expected False for fixed SoA, "
                f"got {summary.get('compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used')!r}"
            )
        for field, label in learner_ready_count_fields:
            try:
                value = int(summary.get(field))
            except (TypeError, ValueError):
                value = -1
            if value != 0:
                violations.append(
                    f"summary.{field}: expected 0 {label} for fixed SoA, "
                    f"got {summary.get(field)!r}"
                )
        for field, expected in (
            (learner_ready_impl_field, "fixed_soa_columns_v1"),
            (learner_ready_path_field, "fixed_soa_direct_gather"),
        ):
            if str(summary.get(field) or "") != expected:
                violations.append(
                    f"summary.{field}: expected {expected!r}, got {summary.get(field)!r}"
                )
    else:
        if (
            summary.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used"
            )
            is not True
        ):
            violations.append(
                "summary.compact_rollout_slab_sample_gate_learner_batch_builder_"
                "learner_ready_unroll2_cache_used: expected True, "
                f"got {summary.get('compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used')!r}"
            )
        for field, label in learner_ready_count_fields:
            try:
                value = int(summary.get(field))
            except (TypeError, ValueError):
                value = 0
            if value <= 0:
                violations.append(
                    f"summary.{field}: expected positive {label}, got {summary.get(field)!r}"
                )
        for field, expected in (
            (learner_ready_impl_field, "learner_ready_unroll2_cache_v1"),
            (learner_ready_path_field, "learner_ready_unroll2_cache"),
        ):
            if str(summary.get(field) or "") != expected:
                violations.append(
                    f"summary.{field}: expected {expected!r}, got {summary.get(field)!r}"
                )
    for field, label in (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count",
            "tensor_native_replay_call_count",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count",
            "tensor_native_replay_table_reused_record_count",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows",
            "tensor_native_replay_table_rows",
        ),
    ):
        try:
            value = int(summary.get(field))
        except (TypeError, ValueError):
            value = 0
        if value <= 0:
            violations.append(
                f"summary.{field}: expected positive {label}, got {summary.get(field)!r}"
            )
    for field, label in (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count",
            "learner_ready_unroll2_cache_fallback_count",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
            "tensor_native_replay_fallback_count",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count",
            "tensor_native_replay_table_missing_record_count",
        ),
    ):
        try:
            value = int(summary.get(field))
        except (TypeError, ValueError):
            value = -1
        if value != 0:
            violations.append(
                f"summary.{field}: expected 0 {label}, got {summary.get(field)!r}"
            )
    if (
        summary.get(
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested"
        )
        is True
    ):
        for field, expected, label in (
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used",
                True,
                "tensor_native_direct_prebuilt_path_used",
            ),
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped",
                True,
                "tensor_native_direct_group_object_build_skipped",
            ),
        ):
            if summary.get(field) is not expected:
                violations.append(
                    f"summary.{field}: expected {expected!r} {label}, "
                    f"got {summary.get(field)!r}"
                )
        for field, expected, label in (
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count",
                0,
                "tensor_native_direct_prebuilt_fallback_count",
            ),
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count",
                0,
                "tensor_native_direct_group_object_count",
            ),
        ):
            try:
                value = int(summary.get(field))
            except (TypeError, ValueError):
                value = -1
            if value != expected:
                violations.append(
                    f"summary.{field}: expected {expected} {label}, "
                    f"got {summary.get(field)!r}"
                )
        direct_reason_field = (
            "compact_rollout_slab_sample_gate_tensor_native_"
            "direct_prebuilt_fallback_reason"
        )
        if str(summary.get(direct_reason_field) or "") != "none":
            violations.append(
                f"summary.{direct_reason_field}: expected 'none', "
                f"got {summary.get(direct_reason_field)!r}"
            )
    expected_strings = (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason",
            "none",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason",
            "none",
        ),
    )
    for field, expected in expected_strings:
        if str(summary.get(field) or "") != expected:
            violations.append(
                f"summary.{field}: expected {expected!r}, got {summary.get(field)!r}"
            )
    expected_one_of_strings = (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
            {
        "maintained_unroll2_table_gather_v1",
        "selected_direct_record_table_gather_v1",
        "selected_maintained_record_table_gather_v1",
        "fixed_soa_direct_gather_v1",
            },
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
            {
        "maintained_record_table_v1",
        "selected_direct_record_table_v1",
        "selected_maintained_record_table_v1",
        "fixed_soa_columns_v1",
            },
        ),
    )
    for field, expected_values in expected_one_of_strings:
        if str(summary.get(field) or "") not in expected_values:
            violations.append(
                f"summary.{field}: expected one of {sorted(expected_values)!r}, "
                f"got {summary.get(field)!r}"
            )
    if tensor_native_impl == "fixed_soa_direct_gather_v1":
        if (
            str(
                summary.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source"
                )
                or ""
            )
            != "fixed_soa_columns_v1"
        ):
            violations.append(
                "summary.compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_source: expected fixed_soa_columns_v1"
            )
        for field, expected in (
            (
                "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count",
                0,
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec",
                0.0,
            ),
            ("compact_rollout_slab_sample_gate_fixed_soa_table_concat_count", 0),
            ("compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count", 0),
            ("compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count", 0),
            ("compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count", 0),
            ("compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count", 0),
            ("compact_rollout_slab_sample_gate_fixed_soa_fallback_count", 0),
        ):
            raw_value = summary.get(field)
            try:
                value = float(raw_value) if isinstance(expected, float) else int(raw_value)
            except (TypeError, ValueError):
                value = None
            if value != expected:
                violations.append(
                    f"summary.{field}: expected {expected!r}, got {raw_value!r}"
                )
        if summary.get("compact_rollout_slab_sample_gate_fixed_soa_used") is not True:
            violations.append(
                "summary.compact_rollout_slab_sample_gate_fixed_soa_used: expected True, "
                f"got {summary.get('compact_rollout_slab_sample_gate_fixed_soa_used')!r}"
            )
        if str(summary.get("compact_rollout_slab_sample_gate_fixed_soa_fallback_reason") or "") != (
            "none"
        ):
            violations.append(
                "summary.compact_rollout_slab_sample_gate_fixed_soa_fallback_reason: "
                f"expected 'none', got {summary.get('compact_rollout_slab_sample_gate_fixed_soa_fallback_reason')!r}"
            )
    for field in (
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec",
    ):
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


def _whole_owner_buffer_replay_ceiling_report_fields(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(field): value
        for field, value in summary.items()
        if str(field).startswith("compact_whole_owner_buffer_replay_ceiling_")
    }


app = modal.App(APP_NAME)
runs_volume = modal.Volume.from_name(
    curvytron_runs_volume_name(),
    **modal_volume_kwargs_for_name(curvytron_runs_volume_name()),
)


def _local_file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


_OWNER_SEARCH_SERVICE_SOURCE_SHA256 = _local_file_sha256(
    Path.cwd() / "src" / "curvyzero" / "training" / "compact_owner_search_service.py"
)
_COMPACT_COACH_SPEED_ROW_SMOKE_SOURCE_SHA256 = _local_file_sha256(
    Path.cwd() / "scripts" / "build_compact_coach_speed_row_smoke.py"
)
speed_row_image = (
    gpu_lightzero_image.apt_install("linux-perf")
    .env(
        {
            "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}",
            "CURVYZERO_OWNER_SEARCH_SERVICE_SOURCE_SHA256": (
                _OWNER_SEARCH_SERVICE_SOURCE_SHA256
            ),
            "CURVYZERO_COMPACT_COACH_SPEED_ROW_SMOKE_SOURCE_SHA256": (
                _COMPACT_COACH_SPEED_ROW_SMOKE_SOURCE_SHA256
            ),
        }
    )
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_file(
        Path.cwd() / "scripts" / "build_compact_coach_speed_row_smoke.py",
        remote_path=str(REMOTE_ROOT / "scripts" / "build_compact_coach_speed_row_smoke.py"),
        copy=True,
    )
)


@app.function(
    image=speed_row_image,
    gpu="H100",
    timeout=20 * 60,
    cpu=4.0,
    volumes={RUNS_MOUNT: runs_volume},
)
def run_compact_coach_speed_row_h100(config: dict[str, Any]) -> dict[str, Any]:
    """Run the compact Coach speed-row producer on H100."""

    try:
        return _run_compact_coach_speed_row_h100(config)
    except Exception as exc:  # pragma: no cover - remote diagnostic payload.
        return {
            "schema_id": RESULT_BUNDLE_SCHEMA_ID,
            "ok": False,
            "status": "failed",
            "problem": f"{type(exc).__name__}: {exc}",
            "profile_only": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "promotion_claim": False,
            "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            "speed_currency": SPEED_CURRENCY,
            **_result_bundle_config_fields_from_config(config),
        }


def _run_compact_coach_speed_row_h100(config: dict[str, Any]) -> dict[str, Any]:
    import scripts.build_compact_coach_speed_row_smoke as smoke

    run_id = str(config.get("run_id") or DEFAULT_RUN_ID)
    lifecycle_ref = _required_ref(config, "unified_lifecycle_report_ref")
    checkpoint_ref = _required_ref(config, "compact_checkpoint_ref")
    runs_volume.reload()
    lifecycle_src = runs.volume_path(RUNS_MOUNT, lifecycle_ref)
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    if not lifecycle_src.is_file():
        raise FileNotFoundError(f"unified lifecycle report not found: {lifecycle_src}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"compact checkpoint not found: {checkpoint_path}")

    work_root = Path("/tmp") / "curvyzero_compact_coach_speed_row" / _safe_id(run_id)
    lifecycle_path = work_root / "unified_lifecycle_report.json"
    output_root = work_root / "results"
    lifecycle = _load_json(lifecycle_src)
    lifecycle["compact_checkpoint_path"] = str(checkpoint_path)
    _write_json(lifecycle_path, lifecycle)

    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--unified-lifecycle-report",
        str(lifecycle_path),
        "--load-unified-lifecycle-checkpoint",
        "--omit-loaded-checkpoint-identity-path",
        "--batch-size",
        str(int(config.get("batch_size", 2))),
        "--actor-count",
        str(int(config.get("actor_count", 1))),
        "--steps",
        str(int(config.get("steps", 4))),
        "--warmup-steps",
        str(int(config.get("warmup_steps", 1))),
        "--death-mode",
        str(config.get("death_mode", "profile_no_death")),
        "--sample-batch-size",
        str(int(config.get("sample_batch_size", 2))),
        "--sample-interval",
        str(int(config.get("sample_interval", 1))),
        "--replay-pair-capacity",
        str(int(config.get("replay_pair_capacity", 16))),
        "--learner-train-steps",
        str(int(config.get("learner_train_steps", 1))),
        "--learner-num-unroll-steps",
        str(int(config.get("learner_num_unroll_steps", 1))),
        "--policy-refresh-interval",
        str(int(config.get("policy_refresh_interval", 1))),
        "--learner-device",
        str(config.get("learner_device", "cuda")),
        "--num-simulations",
        str(int(config.get("num_simulations", 1))),
        "--search-service-kind",
        str(config.get("search_service_kind", "device_target")),
        "--owner-search-inner-search-service-kind",
        str(
            config.get(
                "owner_search_inner_search_service_kind",
                "compact_torch_search_service",
            )
        ),
        "--seed",
        str(int(config.get("seed", 20260530))),
    ]
    if bool(config.get("hybrid_persistent_compact_render_state_buffer", False)):
        argv.append("--hybrid-persistent-compact-render-state-buffer")
    if bool(config.get("hybrid_borrow_single_actor_render_state", False)):
        argv.append("--hybrid-borrow-single-actor-render-state")
    if bool(config.get("owner_search_defer_maintenance", False)):
        argv.append("--owner-search-defer-maintenance")
    if bool(config.get("owner_search_slab_bypass", False)):
        argv.append("--owner-search-slab-bypass")
    if int(config.get("owner_search_transition_batch_size", 1)) > 1:
        argv.extend(
            [
                "--owner-search-transition-batch-size",
                str(int(config.get("owner_search_transition_batch_size", 1))),
            ]
        )
    if bool(config.get("owner_search_direct_transition_batch_replay", False)):
        argv.append("--owner-search-direct-transition-batch-replay")
    if bool(config.get("owner_search_owner_local_transition_derivation", False)):
        argv.append("--owner-search-owner-local-transition-derivation")
    if bool(config.get("owner_search_owner_proxy_transition_closure", False)):
        argv.append("--owner-search-owner-proxy-transition-closure")
    if bool(config.get("owner_search_require_resident_root_view", False)):
        argv.append("--owner-search-require-resident-root-view")
    if bool(config.get("owner_search_resident_root_host_observation_stub", False)):
        argv.append("--owner-search-resident-root-host-observation-stub")
    if bool(config.get("owner_search_direct_root_build_request", False)):
        argv.append("--owner-search-direct-root-build-request")
    if bool(config.get("compact_owner_action_step_boundary", False)):
        argv.append("--compact-owner-action-step-boundary")
    if bool(config.get("compact_owner_action_dispatch_step_overlap", False)):
        argv.append("--compact-owner-action-dispatch-step-overlap")
    if bool(config.get("owner_search_fixed_action_result_buffer", False)):
        argv.append("--owner-search-fixed-action-result-buffer")
        argv.extend(
            [
                "--owner-search-action-result-slot-capacity",
                str(int(config.get("owner_search_action_result_slot_capacity", 4))),
            ]
        )
    if bool(config.get("owner_search_fixed_soa_replay", False)):
        argv.append("--owner-search-fixed-soa-replay")
    fixed_soa_locality_group_size = int(
        config.get("owner_search_fixed_soa_locality_sample_group_size", 1) or 1
    )
    if fixed_soa_locality_group_size != 1:
        argv.extend(
            [
                "--owner-search-fixed-soa-locality-sample-group-size",
                str(fixed_soa_locality_group_size),
            ]
        )
    if bool(config.get("owner_search_defer_model_state_digest_to_refresh", False)):
        argv.append("--owner-search-defer-model-state-digest-to-refresh")
    if bool(config.get("owner_search_async_learner_worker", False)):
        argv.append("--owner-search-async-learner-worker")
        argv.extend(
            [
                "--owner-search-async-learner-worker-kind",
                str(
                    config.get(
                        "owner_search_async_learner_worker_kind",
                        "in_process_thread_v1",
                    )
                ),
                "--owner-search-async-learner-max-pending",
                str(int(config.get("owner_search_async_learner_max_pending", 1))),
            ]
        )
    if bool(config.get("compact_owned_loop_deferred_learner", False)):
        argv.append("--compact-owned-loop-deferred-learner")
    if bool(config.get("compact_owned_loop_deferred_sample_learner", False)):
        argv.append("--compact-owned-loop-deferred-sample-learner")
        argv.extend(
            [
                "--compact-owned-loop-deferred-sample-learner-max-pending",
                str(int(config.get("compact_owned_loop_deferred_sample_learner_max_pending", 1))),
                "--compact-owned-loop-sample-learner-worker-kind",
                str(config.get("compact_owned_loop_sample_learner_worker_kind", "in_process_thread")),
                "--compact-owned-loop-deferred-sample-learner-replay-append-transport-kind",
                str(
                    config.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "replay_append_transport_kind"
                        ),
                        "durable_entry_v1",
                    )
                ),
                "--compact-owned-loop-deferred-sample-learner-model-state-transport-kind",
                str(
                    config.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "model_state_transport_kind"
                        ),
                        "result_v1",
                    )
                ),
            ]
        )
    if bool(config.get("compact_owned_loop_fused_learner_batch", False)):
        argv.append("--compact-owned-loop-fused-learner-batch")
    if bool(config.get("compact_muzero_learner_batch_unroll2_specialized_builder", False)):
        argv.append("--compact-muzero-learner-batch-unroll2-specialized-builder")
    if bool(config.get("compact_muzero_learner_batch_learner_ready_unroll2_cache", False)):
        argv.append("--compact-muzero-learner-batch-learner-ready-unroll2-cache")
    if bool(config.get("compact_muzero_learner_batch_tensor_native_replay", False)):
        argv.append("--compact-muzero-learner-batch-tensor-native-replay")
    if bool(config.get("compact_owned_lean_trainer_step", False)):
        argv.append("--compact-owned-lean-trainer-step")
    if bool(config.get("compact_owned_lean_profile_oracle", False)):
        argv.append("--compact-owned-lean-profile-oracle")
    if bool(config.get("compact_profile_bounded_diagnostics", False)):
        argv.append("--compact-profile-bounded-diagnostics")
    if bool(config.get("compact_profile_cuda_sync_timing_diagnostics", False)):
        argv.append("--compact-profile-cuda-sync-timing-diagnostics")
    if bool(config.get("compact_profile_runtime_step_timing_diagnostics", False)):
        argv.append("--compact-profile-runtime-step-timing-diagnostics")
    if bool(config.get("gpu_utilization_sampling", False)):
        argv.append("--gpu-utilization-sampling")
    argv.extend(
        [
            "--gpu-utilization-sample-interval-sec",
            str(float(config.get("gpu_utilization_sample_interval_sec", 1.0))),
        ]
    )
    if bool(config.get("compact_torch_request_compile", False)):
        argv.append("--compact-torch-request-compile")
    if bool(config.get("compact_torch_request_model_compile", False)):
        argv.append("--compact-torch-request-model-compile")
    model_compile_mode = str(config.get("compact_torch_model_compile_mode", "reduce-overhead"))
    if model_compile_mode not in COMPACT_TORCH_MODEL_COMPILE_MODES:
        allowed = ", ".join(COMPACT_TORCH_MODEL_COMPILE_MODES)
        raise ValueError(
            f"compact_torch_model_compile_mode must be one of {allowed}; got {model_compile_mode!r}"
        )
    argv.extend(["--compact-torch-model-compile-mode", model_compile_mode])
    argv.extend(
        [
            "--compact-torch-timing-mode",
            str(config.get("compact_torch_timing_mode", "host_phase_sync")),
        ]
    )
    initial_inference_mode = str(config.get("compact_torch_initial_inference_mode", "model_method"))
    if initial_inference_mode not in COMPACT_TORCH_INITIAL_INFERENCE_MODES:
        allowed = ", ".join(COMPACT_TORCH_INITIAL_INFERENCE_MODES)
        raise ValueError(
            "compact_torch_initial_inference_mode must be one of "
            f"{allowed}; got {initial_inference_mode!r}"
        )
    argv.extend(["--compact-torch-initial-inference-mode", initial_inference_mode])
    observation_memory_format = str(
        config.get("compact_torch_observation_memory_format", "contiguous")
    )
    if observation_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
        allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
        raise ValueError(
            "compact_torch_observation_memory_format must be one of "
            f"{allowed}; got {observation_memory_format!r}"
        )
    argv.extend(["--compact-torch-observation-memory-format", observation_memory_format])
    model_memory_format = str(config.get("compact_torch_model_memory_format", "contiguous"))
    if model_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
        allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
        raise ValueError(
            "compact_torch_model_memory_format must be one of "
            f"{allowed}; got {model_memory_format!r}"
        )
    if model_memory_format != "contiguous":
        raise ValueError(
            "compact_torch_model_memory_format=channels_last is parked for the "
            "current LightZero MuZero model because recurrent dynamics uses "
            ".view(); use compact_torch_model_memory_format='contiguous'"
        )
    argv.extend(["--compact-torch-model-memory-format", model_memory_format])
    if bool(config.get("compact_torch_defer_one_simulation_replay_payload", False)):
        argv.append("--compact-torch-defer-one-simulation-replay-payload")
    if config.get("source_max_steps") is not None:
        argv.extend(["--source-max-steps", str(int(config["source_max_steps"]))])
    if config.get("decision_source_frames") is not None:
        argv.extend(["--decision-source-frames", str(int(config["decision_source_frames"]))])
    if config.get("source_physics_step_ms") is not None:
        argv.extend(["--source-physics-step-ms", str(float(config["source_physics_step_ms"]))])
    if config.get("source_max_steps_semantics"):
        argv.extend(
            [
                "--source-max-steps-semantics",
                str(config["source_max_steps_semantics"]),
            ]
        )

    result_dir = output_root / run_id
    cpu_perf_stat_fields: dict[str, Any] = {
        "compact_profile_cpu_perf_stat_diagnostics": bool(
            config.get("compact_profile_cpu_perf_stat_diagnostics", False)
        ),
    }
    if bool(config.get("compact_profile_cpu_perf_stat_diagnostics", False)):
        result_dir.mkdir(parents=True, exist_ok=True)
        perf_path = shutil.which("perf")
        if not perf_path:
            return {
                **_speed_row_failure_bundle(
                    config=config,
                    run_id=run_id,
                    rc=127,
                    result_dir=result_dir,
                ),
                "problem": "perf stat diagnostic requested but perf was not found",
                "compact_profile_cpu_perf_stat_diagnostics": True,
                "compact_profile_cpu_perf_stat_available": False,
                "compact_profile_cpu_perf_stat_returncode": 127,
            }
        perf_command = [
            perf_path,
            "stat",
            "-x",
            ",",
            "-e",
            ",".join(CPU_PERF_STAT_EVENTS),
            sys.executable,
            str(REMOTE_ROOT / "scripts" / "build_compact_coach_speed_row_smoke.py"),
            *argv,
        ]
        completed = subprocess.run(
            perf_command,
            cwd=str(REMOTE_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        (result_dir / "cpu_perf_stat_stdout.txt").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (result_dir / "cpu_perf_stat_stderr.txt").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        cpu_perf_stat_fields.update(_parse_perf_stat_csv(completed.stderr))
        cpu_perf_stat_fields.update(
            {
                "compact_profile_cpu_perf_stat_available": True,
                "compact_profile_cpu_perf_stat_returncode": int(completed.returncode),
                "compact_profile_cpu_perf_stat_command": perf_command,
                "compact_profile_cpu_perf_stat_stdout_tail": completed.stdout[-4000:],
                "compact_profile_cpu_perf_stat_stderr_tail": completed.stderr[-4000:],
            }
        )
        rc = int(completed.returncode)
    else:
        rc = smoke.main(argv)
    if int(rc) != 0:
        failure = _speed_row_failure_bundle(
            config=config,
            run_id=run_id,
            rc=int(rc),
            result_dir=result_dir,
        )
        failure.update(cpu_perf_stat_fields)
        if bool(config.get("compact_profile_cpu_perf_stat_diagnostics", False)):
            if not (result_dir / "row_001_result.json").is_file():
                failure["problem"] = "perf stat diagnostic failed before speed-row result"
        return failure
    manifest = _load_json(result_dir / "manifest.json")
    result = _load_json(result_dir / "row_001_result.json")
    evidence = _load_json(result_dir / "row_001_result.json.compact_coach_speed_row.evidence.json")
    report = _load_json(result_dir / "compact_coach_speed_row_smoke_report.json")
    if result.get("schema_id") != COMPACT_COACH_SPEED_ROW_RESULT_SCHEMA_ID:
        raise ValueError("remote speed-row result schema mismatch")
    summary = result.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("remote speed-row result summary missing")
    if cpu_perf_stat_fields:
        summary.update(cpu_perf_stat_fields)
        compact_payload = result.get("compact")
        if isinstance(compact_payload, dict):
            compact_payload.update(cpu_perf_stat_fields)
        _write_json(result_dir / "row_001_result.json", result)
    cuda_sync_timing_diagnostic_violations = _cuda_sync_timing_diagnostic_violations(
        requested=bool(config.get("compact_profile_cuda_sync_timing_diagnostics", False)),
        summary=summary,
    )
    if cuda_sync_timing_diagnostic_violations:
        raise ValueError(
            "remote CUDA sync timing diagnostics did not prove active probes: "
            + ", ".join(cuda_sync_timing_diagnostic_violations)
        )
    runtime_step_timing_diagnostic_violations = _runtime_step_timing_diagnostic_violations(
        requested=bool(
            config.get("compact_profile_runtime_step_timing_diagnostics", False)
            or config.get("compact_profile_cuda_sync_timing_diagnostics", False)
        ),
        summary=summary,
    )
    if runtime_step_timing_diagnostic_violations:
        raise ValueError(
            "remote runtime-step timing diagnostics did not prove active probes: "
            + ", ".join(runtime_step_timing_diagnostic_violations)
        )
    fused_learner_batch_violations = _remote_fused_learner_batch_violations(
        config=config,
        summary=summary,
    )
    if fused_learner_batch_violations:
        raise ValueError(
            "remote fused speed row did not prove fused sample/learner path: "
            + ", ".join(fused_learner_batch_violations)
        )
    tensor_native_replay_violations = _tensor_native_replay_violations(
        config=config,
        summary=summary,
    )
    if tensor_native_replay_violations:
        raise ValueError(
            "remote tensor-native replay did not prove maintained replay path: "
            + ", ".join(tensor_native_replay_violations)
        )
    return {
        "schema_id": RESULT_BUNDLE_SCHEMA_ID,
        "ok": True,
        "status": "complete",
        "run_id": run_id,
        "candidate_checkpoint_id": result.get("candidate_checkpoint_id"),
        "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
        "speed_currency": SPEED_CURRENCY,
        "env_steps_collected": summary.get("env_steps_collected"),
        "training_wall_sec": summary.get("training_wall_sec"),
        "compact_trainer_env_steps_per_sec": summary.get("steps_per_sec"),
        "steps_per_sec": summary.get("steps_per_sec"),
        **_whole_owner_buffer_replay_ceiling_report_fields(summary),
        "seed": summary.get("seed"),
        "sample_seed_base": summary.get("sample_seed_base"),
        "sample_batch_size": summary.get("sample_batch_size"),
        "sample_interval": summary.get("sample_interval"),
        "replay_pair_capacity": summary.get("replay_pair_capacity"),
        "learner_train_steps": summary.get("learner_train_steps"),
        "policy_refresh_interval": summary.get("policy_refresh_interval"),
        "num_simulations": summary.get("num_simulations"),
        "compact_rollout_slab_sample_gate_last_seed": summary.get(
            "compact_rollout_slab_sample_gate_last_seed"
        ),
        "compact_rollout_slab_learner_gate_last_seed": summary.get(
            "compact_rollout_slab_learner_gate_last_seed"
        ),
        "compact_owned_loop_sample_gate_last_metadata_seed": summary.get(
            "compact_owned_loop_sample_gate_last_metadata_seed"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed": (
            summary.get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed"
            )
        ),
        "compact_torch_observation_memory_format": str(
            summary.get("compact_torch_observation_memory_format", "contiguous")
        ),
        "compact_torch_initial_inference_mode": str(
            summary.get("compact_torch_initial_inference_mode", "model_method")
        ),
        "compact_torch_model_memory_format": str(
            summary.get("compact_torch_model_memory_format", "contiguous")
        ),
        "compact_torch_defer_one_simulation_replay_payload_requested": bool(
            summary.get(
                "compact_torch_defer_one_simulation_replay_payload_requested",
                False,
            )
        ),
        "compact_torch_memory_format_applies_to_search_service": bool(
            summary.get("compact_torch_memory_format_applies_to_search_service", False)
        ),
        "owner_search_slab_proxy_requested": bool(
            summary.get("owner_search_slab_proxy_requested", False)
        ),
        "owner_search_inline_proxy_requested": bool(
            summary.get("owner_search_inline_proxy_requested", False)
        ),
        "owner_search_inline_background_proxy_requested": bool(
            summary.get("owner_search_inline_background_proxy_requested", False)
        ),
        "owner_search_threaded_proxy_requested": bool(
            summary.get("owner_search_threaded_proxy_requested", False)
        ),
        "owner_search_inner_search_service_kind": str(
            summary.get("owner_search_inner_search_service_kind", "")
        ),
        "owner_search_inner_search_service_impl": str(
            summary.get("owner_search_inner_search_service_impl", "")
        ),
        "owner_search_compact_torch_resident_root_bridge_ready": bool(
            summary.get("owner_search_compact_torch_resident_root_bridge_ready", False)
        ),
        "owner_search_defer_maintenance_requested": bool(
            summary.get("owner_search_defer_maintenance_requested", False)
        ),
        "owner_search_slab_bypass_requested": bool(
            summary.get("owner_search_slab_bypass_requested", False)
        ),
        "owner_search_transition_batch_size_requested": int(
            summary.get("owner_search_transition_batch_size_requested") or 1
        ),
        "owner_search_transition_batch_transport_requested": bool(
            summary.get("owner_search_transition_batch_transport_requested", False)
        ),
        "owner_search_direct_transition_batch_replay_requested": bool(
            summary.get("owner_search_direct_transition_batch_replay_requested", False)
        ),
        "owner_search_owner_local_transition_derivation_requested": bool(
            summary.get("owner_search_owner_local_transition_derivation_requested", False)
        ),
        "owner_search_owner_proxy_transition_closure_requested": bool(
            summary.get("owner_search_owner_proxy_transition_closure_requested", False)
        ),
        "owner_search_require_resident_root_view_requested": bool(
            summary.get("owner_search_require_resident_root_view_requested", False)
        ),
        "owner_search_resident_root_host_observation_stub_requested": bool(
            summary.get(
                "owner_search_resident_root_host_observation_stub_requested",
                False,
            )
        ),
        "owner_search_direct_root_build_request_requested": bool(
            summary.get("owner_search_direct_root_build_request_requested", False)
        ),
        "compact_owner_action_step_boundary_requested": bool(
            summary.get("compact_owner_action_step_boundary_requested", False)
        ),
        "compact_owner_action_dispatch_step_overlap_requested": bool(
            summary.get("compact_owner_action_dispatch_step_overlap_requested", False)
        ),
        "owner_search_fixed_action_result_buffer_requested": bool(
            summary.get("owner_search_fixed_action_result_buffer_requested", False)
        ),
        "owner_search_action_result_slot_capacity_requested": int(
            summary.get("owner_search_action_result_slot_capacity_requested") or 0
        ),
        "owner_search_fixed_soa_replay_requested": bool(
            summary.get("owner_search_fixed_soa_replay_requested", False)
        ),
        "owner_search_defer_model_state_digest_to_refresh_requested": bool(
            summary.get(
                "owner_search_defer_model_state_digest_to_refresh_requested",
                False,
            )
        ),
        "owner_search_fixed_soa_locality_sample_group_size_requested": int(
            summary.get("owner_search_fixed_soa_locality_sample_group_size_requested")
            or 1
        ),
        "owner_search_async_learner_worker_requested": bool(
            summary.get("owner_search_async_learner_worker_requested", False)
        ),
        "owner_search_async_learner_worker_kind_requested": str(
            summary.get("owner_search_async_learner_worker_kind_requested", "")
        ),
        "owner_search_async_learner_max_pending_requested": int(
            summary.get("owner_search_async_learner_max_pending_requested") or 0
        ),
        "compact_owner_search_resident_root_bridge_ready": bool(
            summary.get("compact_owner_search_resident_root_bridge_ready", False)
        ),
        "compact_owner_search_resident_root_bridge_kind": str(
            summary.get("compact_owner_search_resident_root_bridge_kind", "")
        ),
        "compact_owner_search_resident_root_bridge_device": str(
            summary.get("compact_owner_search_resident_root_bridge_device", "")
        ),
        "compact_owner_search_resident_root_bridge_h2d_bytes": float(
            summary.get("compact_owner_search_resident_root_bridge_h2d_bytes") or 0.0
        ),
        "compact_owner_search_resident_root_bridge_host_observation_copied": bool(
            summary.get("compact_owner_search_resident_root_bridge_host_observation_copied", True)
        ),
        "compact_owner_search_resident_root_bridge_generation_id": int(
            summary.get("compact_owner_search_resident_root_bridge_generation_id") or 0
        ),
        "hybrid_persistent_compact_render_state_buffer": bool(
            summary.get("hybrid_persistent_compact_render_state_buffer", False)
        ),
        "hybrid_borrow_single_actor_render_state": bool(
            summary.get("hybrid_borrow_single_actor_render_state", False)
        ),
        "compact_owned_loop_deferred_learner": bool(
            summary.get("compact_owned_loop_deferred_learner", False)
        ),
        "compact_owned_loop_deferred_sample_learner": bool(
            summary.get("compact_owned_loop_deferred_sample_learner", False)
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_requested": summary.get(
            "compact_owned_loop_deferred_sample_learner_max_pending_requested"
        ),
        "compact_owned_loop_sample_learner_worker_kind_requested": summary.get(
            "compact_owned_loop_sample_learner_worker_kind_requested"
        ),
        "compact_owned_loop_fused_learner_batch": bool(
            summary.get("compact_owned_loop_fused_learner_batch", False)
        ),
        "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
            summary.get("compact_muzero_learner_batch_unroll2_specialized_builder", False)
        ),
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
            summary.get("compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ),
        "compact_muzero_learner_batch_tensor_native_replay": bool(
            summary.get("compact_muzero_learner_batch_tensor_native_replay", False)
        ),
        "compact_owned_lean_trainer_step": bool(
            summary.get("compact_owned_lean_trainer_step", False)
        ),
        "compact_owned_lean_profile_oracle_requested": bool(
            summary.get("compact_owned_lean_profile_oracle_requested", False)
        ),
        "compact_owned_training_loop_owner": str(
            summary.get("compact_owned_training_loop_owner", "")
        ),
        "compact_owned_lean_profile_oracle": summary.get("compact_owned_lean_profile_oracle"),
        "speed_row_gpu_utilization_sampling_enabled": bool(
            summary.get("speed_row_gpu_utilization_sampling_enabled", False)
        ),
        "speed_row_gpu_utilization_sample_interval_sec": summary.get(
            "speed_row_gpu_utilization_sample_interval_sec"
        ),
        "speed_row_gpu_utilization_sample_count": summary.get(
            "speed_row_gpu_utilization_sample_count"
        ),
        "speed_row_gpu_name": summary.get("speed_row_gpu_name"),
        "speed_row_gpu_utilization_max_percent": summary.get(
            "speed_row_gpu_utilization_max_percent"
        ),
        "speed_row_gpu_utilization_mean_percent": summary.get(
            "speed_row_gpu_utilization_mean_percent"
        ),
        "speed_row_gpu_utilization_nonzero_sample_count": summary.get(
            "speed_row_gpu_utilization_nonzero_sample_count"
        ),
        "speed_row_gpu_utilization_over_50_sample_count": summary.get(
            "speed_row_gpu_utilization_over_50_sample_count"
        ),
        "speed_row_gpu_utilization_over_80_sample_count": summary.get(
            "speed_row_gpu_utilization_over_80_sample_count"
        ),
        "speed_row_gpu_memory_utilization_max_percent": summary.get(
            "speed_row_gpu_memory_utilization_max_percent"
        ),
        "speed_row_gpu_memory_used_max_mib": summary.get("speed_row_gpu_memory_used_max_mib"),
        "speed_row_gpu_power_draw_max_w": summary.get("speed_row_gpu_power_draw_max_w"),
        "speed_row_gpu_utilization_sampling_errors": summary.get(
            "speed_row_gpu_utilization_sampling_errors"
        ),
        **_cpu_perf_stat_report_fields(summary),
        "compact_rollout_slab_sample_gate_sec": summary.get("compact_rollout_slab_sample_gate_sec"),
        "compact_rollout_slab_learner_gate_sec": summary.get(
            "compact_rollout_slab_learner_gate_sec"
        ),
        "compact_rollout_slab_sample_gate_calls": summary.get(
            "compact_rollout_slab_sample_gate_calls"
        ),
        "compact_rollout_slab_learner_gate_calls": summary.get(
            "compact_rollout_slab_learner_gate_calls"
        ),
        "compact_rollout_slab_learner_gate_updates": summary.get(
            "compact_rollout_slab_learner_gate_updates"
        ),
        "compact_owned_loop_record_step_calls": summary.get(
            "compact_owned_loop_record_step_calls"
        ),
        "compact_owned_loop_appended_replay_entry_count": summary.get(
            "compact_owned_loop_appended_replay_entry_count"
        ),
        "resident_replay_snapshot_mode": summary.get("resident_replay_snapshot_mode"),
        "compact_owned_loop_replay_store_retained_resident_snapshot_count": summary.get(
            "compact_owned_loop_replay_store_retained_resident_snapshot_count"
        ),
        "compact_owned_loop_replay_store_retained_resident_snapshot_bytes": summary.get(
            "compact_owned_loop_replay_store_retained_resident_snapshot_bytes"
        ),
        "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_count": summary.get(
            "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_count"
        ),
        "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_bytes": summary.get(
            "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_bytes"
        ),
        "compact_rollout_slab_sample_gate_sample_rows": summary.get(
            "compact_rollout_slab_sample_gate_sample_rows"
        ),
        "compact_rollout_slab_learner_gate_sample_rows": summary.get(
            "compact_rollout_slab_learner_gate_sample_rows"
        ),
        "compact_rollout_slab_sample_gate_opportunities": summary.get(
            "compact_rollout_slab_sample_gate_opportunities"
        ),
        "compact_rollout_slab_sample_gate_skipped_count": summary.get(
            "compact_rollout_slab_sample_gate_skipped_count"
        ),
        "compact_owned_trainer_learner_update_count": summary.get(
            "compact_owned_trainer_learner_update_count"
        ),
        "compact_owned_trainer_sample_batch_count": summary.get(
            "compact_owned_trainer_sample_batch_count"
        ),
        "compact_owned_trainer_policy_refresh_count": summary.get(
            "compact_owned_trainer_policy_refresh_count"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count"
        ),
        (
            "compact_rollout_slab_policy_refresh_after_learner_gate_"
            "parent_model_state_transport_avoided"
        ): summary.get(
            (
                "compact_rollout_slab_policy_refresh_after_learner_gate_"
                "parent_model_state_transport_avoided"
            )
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_calls": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_calls"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_interval": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_interval"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata_update_count": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata_update_count"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata_update_count": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata_update_count"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_lag_to_final_update": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_lag_to_final_update"
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_lag_to_final_update": summary.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_lag_to_final_update"
        ),
        "source_profile_total_sec": summary.get("source_profile_total_sec"),
        "source_profile_warmup_sec": summary.get("source_profile_warmup_sec"),
        "source_profile_measured_sec": summary.get("source_profile_measured_sec"),
        "source_profile_timing_per_timestep_sec": summary.get(
            "source_profile_timing_per_timestep_sec"
        ),
        "speed_row_actor_step_wall_sec": summary.get("speed_row_actor_step_wall_sec"),
        "speed_row_observation_sec": summary.get("speed_row_observation_sec"),
        "speed_row_renderer_stack_update_sec": summary.get("speed_row_renderer_stack_update_sec"),
        "speed_row_compact_rollout_slab_sec": summary.get("speed_row_compact_rollout_slab_sec"),
        "speed_row_sample_gate_sec": summary.get("speed_row_sample_gate_sec"),
        "speed_row_learner_gate_sec": summary.get("speed_row_learner_gate_sec"),
        "speed_row_policy_refresh_sec": summary.get("speed_row_policy_refresh_sec"),
        "speed_row_primary_accounted_sec": summary.get("speed_row_primary_accounted_sec"),
        "speed_row_primary_residual_sec": summary.get("speed_row_primary_residual_sec"),
        **{key: summary.get(key) for key in _ACTOR_OBSERVATION_TIMER_REPORT_FIELDS},
        **{key: summary.get(key) for key in _SAMPLE_LEARNER_TIMER_REPORT_FIELDS},
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
        **{
            f"{prefix}_stats": summary.get(f"{prefix}_stats")
            for prefix in _SAMPLE_GATE_PER_CALL_REPORT_PREFIXES
        },
        "compact_rollout_slab_learner_gate_prebuilt_batch_used": summary.get(
            "compact_rollout_slab_learner_gate_prebuilt_batch_used"
        ),
        "compact_muzero_learner_prebuilt_batch_used": summary.get(
            "compact_muzero_learner_prebuilt_batch_used"
        ),
        "compact_owned_loop_deferred_learner_submit_count": summary.get(
            "compact_owned_loop_deferred_learner_submit_count"
        ),
        "compact_owned_loop_deferred_learner_completed_count": summary.get(
            "compact_owned_loop_deferred_learner_completed_count"
        ),
        "compact_owned_loop_deferred_learner_pending": summary.get(
            "compact_owned_loop_deferred_learner_pending"
        ),
        "compact_owned_loop_deferred_learner_pending_count": summary.get(
            "compact_owned_loop_deferred_learner_pending_count"
        ),
        "compact_owned_loop_deferred_learner_max_pending": summary.get(
            "compact_owned_loop_deferred_learner_max_pending"
        ),
        "compact_owned_loop_deferred_learner_max_pending_observed": summary.get(
            "compact_owned_loop_deferred_learner_max_pending_observed"
        ),
        "compact_owned_loop_deferred_learner_actor_steps_while_pending": summary.get(
            "compact_owned_loop_deferred_learner_actor_steps_while_pending"
        ),
        "compact_owned_loop_deferred_learner_policy_lag_current": summary.get(
            "compact_owned_loop_deferred_learner_policy_lag_current"
        ),
        "compact_owned_loop_deferred_learner_policy_lag_max": summary.get(
            "compact_owned_loop_deferred_learner_policy_lag_max"
        ),
        "compact_owned_loop_deferred_learner_wait_count": summary.get(
            "compact_owned_loop_deferred_learner_wait_count"
        ),
        "compact_owned_loop_deferred_learner_wait_sec": summary.get(
            "compact_owned_loop_deferred_learner_wait_sec"
        ),
        "compact_owned_loop_deferred_learner_last_wait_sec": summary.get(
            "compact_owned_loop_deferred_learner_last_wait_sec"
        ),
        "compact_owned_loop_deferred_sample_learner_submit_count": summary.get(
            "compact_owned_loop_deferred_sample_learner_submit_count"
        ),
        "compact_owned_loop_sample_learner_worker_kind": summary.get(
            "compact_owned_loop_sample_learner_worker_kind"
        ),
        "compact_owned_loop_sample_learner_worker_resource_id": summary.get(
            "compact_owned_loop_sample_learner_worker_resource_id"
        ),
        "compact_owned_loop_actor_search_resource_id": summary.get(
            "compact_owned_loop_actor_search_resource_id"
        ),
        "compact_owned_loop_actor_search_pid": summary.get(
            "compact_owned_loop_actor_search_pid"
        ),
        "compact_owned_loop_sample_learner_worker_parent_pid": summary.get(
            "compact_owned_loop_sample_learner_worker_parent_pid"
        ),
        "compact_owned_loop_sample_learner_worker_resource_scope": summary.get(
            "compact_owned_loop_sample_learner_worker_resource_scope"
        ),
        "compact_owned_loop_sample_learner_worker_start_method": summary.get(
            "compact_owned_loop_sample_learner_worker_start_method"
        ),
        "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings": summary.get(
            "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings"
        ),
        "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": summary.get(
            "compact_owned_loop_sample_learner_resource_distinct_from_actor_search"
        ),
        "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search": (
            summary.get(
                "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search"
            )
        ),
        "compact_owned_loop_deferred_sample_learner_completed_count": summary.get(
            "compact_owned_loop_deferred_sample_learner_completed_count"
        ),
        "compact_owned_loop_deferred_sample_learner_pending": summary.get(
            "compact_owned_loop_deferred_sample_learner_pending"
        ),
        "compact_owned_loop_deferred_sample_learner_pending_count": summary.get(
            "compact_owned_loop_deferred_sample_learner_pending_count"
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending": summary.get(
            "compact_owned_loop_deferred_sample_learner_max_pending"
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_observed": summary.get(
            "compact_owned_loop_deferred_sample_learner_max_pending_observed"
        ),
        "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending": summary.get(
            "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending"
        ),
        "compact_owned_loop_deferred_sample_learner_policy_lag_current": summary.get(
            "compact_owned_loop_deferred_sample_learner_policy_lag_current"
        ),
        "compact_owned_loop_deferred_sample_learner_policy_lag_max": summary.get(
            "compact_owned_loop_deferred_sample_learner_policy_lag_max"
        ),
        "compact_owned_loop_deferred_sample_learner_last_submitted_request_id": summary.get(
            "compact_owned_loop_deferred_sample_learner_last_submitted_request_id"
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_request_id": summary.get(
            "compact_owned_loop_deferred_sample_learner_last_completed_request_id"
        ),
        "compact_owned_loop_deferred_sample_learner_last_submitted_snapshot_version": summary.get(
            "compact_owned_loop_deferred_sample_learner_last_submitted_snapshot_version"
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_snapshot_version": summary.get(
            "compact_owned_loop_deferred_sample_learner_last_completed_snapshot_version"
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid": summary.get(
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid"
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id": summary.get(
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id"
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device": summary.get(
            "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device"
        ),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "last_completed_worker_pid_distinct_from_actor_search"
        ): summary.get(
            (
                "compact_owned_loop_deferred_sample_learner_"
                "last_completed_worker_pid_distinct_from_actor_search"
            )
        ),
        "compact_owned_loop_deferred_sample_learner_model_state_apply_count": summary.get(
            "compact_owned_loop_deferred_sample_learner_model_state_apply_count"
        ),
        "compact_owned_loop_deferred_sample_learner_last_model_state_applied": summary.get(
            "compact_owned_loop_deferred_sample_learner_last_model_state_applied"
        ),
        **_sample_learner_transport_proof_fields(summary),
        **_owner_search_slab_proxy_proof_fields(summary),
        "compact_owned_loop_deferred_sample_learner_wait_count": summary.get(
            "compact_owned_loop_deferred_sample_learner_wait_count"
        ),
        "compact_owned_loop_deferred_sample_learner_wait_sec": summary.get(
            "compact_owned_loop_deferred_sample_learner_wait_sec"
        ),
        "compact_owned_loop_deferred_sample_learner_last_wait_sec": summary.get(
            "compact_owned_loop_deferred_sample_learner_last_wait_sec"
        ),
        "compact_owned_loop_deferred_sample_learner_drained": summary.get(
            "compact_owned_loop_deferred_sample_learner_drained"
        ),
        "compact_owned_loop_final_deferred_drain_sec": summary.get(
            "compact_owned_loop_final_deferred_drain_sec"
        ),
        "compact_owned_loop_final_deferred_sample_learner_drain_sec": summary.get(
            "compact_owned_loop_final_deferred_sample_learner_drain_sec"
        ),
        "compact_owned_loop_final_deferred_learner_drain_sec": summary.get(
            "compact_owned_loop_final_deferred_learner_drain_sec"
        ),
        "compact_owned_loop_final_deferred_drain_in_measured_sec": summary.get(
            "compact_owned_loop_final_deferred_drain_in_measured_sec"
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
        "compact_owned_trainer_config_death_mode": summary.get(
            "compact_owned_trainer_config_death_mode"
        ),
        "normal_death_terminal_contract_owner": summary.get("normal_death_terminal_contract_owner"),
        "terminal_row_count": summary.get("terminal_row_count"),
        "death_row_count": summary.get("death_row_count"),
        "terminated_row_count": summary.get("terminated_row_count"),
        "truncated_row_count": summary.get("truncated_row_count"),
        "env_action_checksum_total": summary.get("env_action_checksum_total"),
        "env_done_checksum_total": summary.get("env_done_checksum_total"),
        "env_reward_checksum_total": summary.get("env_reward_checksum_total"),
        "env_action_mask_checksum_total": summary.get("env_action_mask_checksum_total"),
        "env_trajectory_checksum_total": summary.get("env_trajectory_checksum_total"),
        "env_trajectory_ordered_checksum_total": summary.get(
            "env_trajectory_ordered_checksum_total"
        ),
        "env_terminal_row_checksum_total": summary.get("env_terminal_row_checksum_total"),
        "env_autoreset_row_checksum_total": summary.get("env_autoreset_row_checksum_total"),
        "env_terminal_reason_checksum_total": summary.get(
            "env_terminal_reason_checksum_total"
        ),
        "env_death_count_checksum_total": summary.get("env_death_count_checksum_total"),
        "env_death_cause_checksum_total": summary.get("env_death_cause_checksum_total"),
        "env_death_hit_owner_checksum_total": summary.get(
            "env_death_hit_owner_checksum_total"
        ),
        "last_env_action_checksum": summary.get("last_env_action_checksum"),
        "last_env_trajectory_checksum": summary.get("last_env_trajectory_checksum"),
        "last_env_terminal_row_checksum": summary.get("last_env_terminal_row_checksum"),
        "last_env_autoreset_row_checksum": summary.get("last_env_autoreset_row_checksum"),
        "terminal_sample_row_count": summary.get("terminal_sample_row_count"),
        "compact_profile_autoreset_direct_count": summary.get(
            "compact_profile_autoreset_direct_count"
        ),
        "compact_profile_autoreset_template_copy_skipped_count": summary.get(
            "compact_profile_autoreset_template_copy_skipped_count"
        ),
        "compact_profile_autoreset_direct_row_count": summary.get(
            "compact_profile_autoreset_direct_row_count"
        ),
        "compact_rollout_slab_sample_gate_action_checksum": summary.get(
            "compact_rollout_slab_sample_gate_action_checksum"
        ),
        "compact_rollout_slab_sample_gate_sample_row_checksum": summary.get(
            "compact_rollout_slab_sample_gate_sample_row_checksum"
        ),
        "compact_rollout_slab_sample_gate_sample_action_checksum": summary.get(
            "compact_rollout_slab_sample_gate_sample_action_checksum"
        ),
        "compact_rollout_slab_sample_gate_sampled_flat_row_checksum": summary.get(
            "compact_rollout_slab_sample_gate_sampled_flat_row_checksum"
        ),
        "compact_rollout_slab_sample_gate_sample_position_order_checksum": summary.get(
            "compact_rollout_slab_sample_gate_sample_position_order_checksum"
        ),
        "compact_rollout_slab_sample_gate_source_record_pair_checksum": summary.get(
            "compact_rollout_slab_sample_gate_source_record_pair_checksum"
        ),
        "compact_rollout_slab_sample_gate_source_record_window_checksum": summary.get(
            "compact_rollout_slab_sample_gate_source_record_window_checksum"
        ),
        "terminal_unroll_value_target_mode": summary.get("terminal_unroll_value_target_mode"),
        "terminal_unroll_value_target_row_count": summary.get(
            "terminal_unroll_value_target_row_count"
        ),
        "normal_death_terminal_contract_promotion_gate_satisfied": summary.get(
            "normal_death_terminal_contract_promotion_gate_satisfied"
        ),
        "resident_observation_host_fallback_count": summary.get(
            "resident_observation_host_fallback_count"
        ),
        "learner_num_unroll_steps": summary.get("learner_num_unroll_steps"),
        "unified_lifecycle_report_ref": lifecycle_ref,
        "compact_checkpoint_ref": checkpoint_ref,
        "manifest": manifest,
        "result": result,
        "evidence": evidence,
        "report": report,
    }


def _sample_learner_transport_proof_fields(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: summary.get(field)
        for field in _SAMPLE_LEARNER_TRANSPORT_PROOF_FIELDS
        if field in summary
    }


def _owner_search_slab_proxy_proof_fields(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: summary.get(field)
        for field in _OWNER_SEARCH_SLAB_PROXY_PROOF_FIELDS
        if field in summary
    }


@app.local_entrypoint()
def main(
    *,
    speed_row_spawn_result: bool = False,
    run_id: str = DEFAULT_RUN_ID,
    unified_lifecycle_report_ref: str,
    compact_checkpoint_ref: str,
    batch_size: int = 2,
    actor_count: int = 1,
    steps: int = 4,
    warmup_steps: int = 1,
    death_mode: str = "profile_no_death",
    sample_batch_size: int = 2,
    sample_interval: int = 1,
    replay_pair_capacity: int = 16,
    learner_train_steps: int = 1,
    learner_num_unroll_steps: int = 1,
    policy_refresh_interval: int = 1,
    hybrid_persistent_compact_render_state_buffer: bool = False,
    hybrid_borrow_single_actor_render_state: bool = False,
    compact_owned_loop_deferred_learner: bool = False,
    compact_owned_loop_deferred_sample_learner: bool = False,
    compact_owned_loop_deferred_sample_learner_max_pending: int = 1,
    compact_owned_loop_sample_learner_worker_kind: str = "in_process_thread",
    compact_owned_loop_deferred_sample_learner_replay_append_transport_kind: str = (
        "durable_entry_v1"
    ),
    compact_owned_loop_deferred_sample_learner_model_state_transport_kind: str = (
        "result_v1"
    ),
    compact_owned_loop_fused_learner_batch: bool = False,
    compact_muzero_learner_batch_unroll2_specialized_builder: bool = False,
    compact_muzero_learner_batch_learner_ready_unroll2_cache: bool = False,
    compact_muzero_learner_batch_tensor_native_replay: bool = False,
    compact_owned_lean_trainer_step: bool = False,
    compact_owned_lean_profile_oracle: bool = False,
    compact_profile_bounded_diagnostics: bool = False,
    compact_profile_cuda_sync_timing_diagnostics: bool = False,
    compact_profile_runtime_step_timing_diagnostics: bool = False,
    compact_profile_cpu_perf_stat_diagnostics: bool = False,
    learner_device: str = "cuda",
    gpu_utilization_sampling: bool = False,
    gpu_utilization_sample_interval_sec: float = 1.0,
    num_simulations: int = 1,
    search_service_kind: str = "device_target",
    owner_search_inner_search_service_kind: str = "compact_torch_search_service",
    owner_search_defer_maintenance: bool = False,
    owner_search_slab_bypass: bool = False,
    owner_search_transition_batch_size: int = 1,
    owner_search_direct_transition_batch_replay: bool = False,
    owner_search_owner_local_transition_derivation: bool = False,
    owner_search_owner_proxy_transition_closure: bool = False,
    owner_search_require_resident_root_view: bool = False,
    owner_search_resident_root_host_observation_stub: bool = False,
    owner_search_direct_root_build_request: bool = False,
    compact_owner_action_step_boundary: bool = False,
    compact_owner_action_dispatch_step_overlap: bool = False,
    owner_search_fixed_action_result_buffer: bool = False,
    owner_search_action_result_slot_capacity: int = 4,
    owner_search_fixed_soa_replay: bool = False,
    owner_search_defer_model_state_digest_to_refresh: bool = False,
    owner_search_fixed_soa_locality_sample_group_size: int = 1,
    owner_search_async_learner_worker: bool = False,
    owner_search_async_learner_worker_kind: str = "in_process_thread_v1",
    owner_search_async_learner_max_pending: int = 1,
    compact_torch_request_compile: bool = False,
    compact_torch_request_model_compile: bool = False,
    compact_torch_model_compile_mode: str = "reduce-overhead",
    compact_torch_timing_mode: str = "host_phase_sync",
    compact_torch_initial_inference_mode: str = "model_method",
    compact_torch_observation_memory_format: str = "contiguous",
    compact_torch_model_memory_format: str = "contiguous",
    compact_torch_defer_one_simulation_replay_payload: bool = False,
    seed: int = 20260530,
    source_max_steps: int = 1048576,
    decision_source_frames: int = 1,
    source_physics_step_ms: float = 16.666666666666668,
    source_max_steps_semantics: str = "source_physics_steps",
) -> None:
    """Launch or run the H100 compact Coach speed-row producer."""

    config = {
        "run_id": str(run_id),
        "unified_lifecycle_report_ref": str(unified_lifecycle_report_ref),
        "compact_checkpoint_ref": str(compact_checkpoint_ref),
        "batch_size": int(batch_size),
        "actor_count": int(actor_count),
        "steps": int(steps),
        "warmup_steps": int(warmup_steps),
        "death_mode": str(death_mode),
        "sample_batch_size": int(sample_batch_size),
        "sample_interval": int(sample_interval),
        "replay_pair_capacity": int(replay_pair_capacity),
        "learner_train_steps": int(learner_train_steps),
        "learner_num_unroll_steps": int(learner_num_unroll_steps),
        "policy_refresh_interval": int(policy_refresh_interval),
        "hybrid_persistent_compact_render_state_buffer": bool(
            hybrid_persistent_compact_render_state_buffer
        ),
        "hybrid_borrow_single_actor_render_state": bool(hybrid_borrow_single_actor_render_state),
        "compact_owned_loop_deferred_learner": bool(compact_owned_loop_deferred_learner),
        "compact_owned_loop_deferred_sample_learner": bool(
            compact_owned_loop_deferred_sample_learner
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending": int(
            compact_owned_loop_deferred_sample_learner_max_pending
        ),
        "compact_owned_loop_sample_learner_worker_kind": str(
            compact_owned_loop_sample_learner_worker_kind
        ),
        "compact_owned_loop_deferred_sample_learner_replay_append_transport_kind": str(
            compact_owned_loop_deferred_sample_learner_replay_append_transport_kind
        ),
        "compact_owned_loop_deferred_sample_learner_model_state_transport_kind": str(
            compact_owned_loop_deferred_sample_learner_model_state_transport_kind
        ),
        "compact_owned_loop_fused_learner_batch": bool(compact_owned_loop_fused_learner_batch),
        "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
            compact_muzero_learner_batch_unroll2_specialized_builder
        ),
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
            compact_muzero_learner_batch_learner_ready_unroll2_cache
        ),
        "compact_muzero_learner_batch_tensor_native_replay": bool(
            compact_muzero_learner_batch_tensor_native_replay
        ),
        "compact_owned_lean_trainer_step": bool(compact_owned_lean_trainer_step),
        "compact_owned_lean_profile_oracle": bool(compact_owned_lean_profile_oracle),
        "compact_profile_bounded_diagnostics": bool(compact_profile_bounded_diagnostics),
        "compact_profile_cuda_sync_timing_diagnostics": bool(
            compact_profile_cuda_sync_timing_diagnostics
        ),
        "compact_profile_runtime_step_timing_diagnostics": bool(
            compact_profile_runtime_step_timing_diagnostics
            or compact_profile_cuda_sync_timing_diagnostics
        ),
        "compact_profile_cpu_perf_stat_diagnostics": bool(
            compact_profile_cpu_perf_stat_diagnostics
        ),
        "learner_device": str(learner_device),
        "gpu_utilization_sampling": bool(gpu_utilization_sampling),
        "gpu_utilization_sample_interval_sec": float(gpu_utilization_sample_interval_sec),
        "num_simulations": int(num_simulations),
        "search_service_kind": str(search_service_kind),
        "owner_search_inner_search_service_kind": str(owner_search_inner_search_service_kind),
        "owner_search_defer_maintenance": bool(owner_search_defer_maintenance),
        "owner_search_slab_bypass": bool(owner_search_slab_bypass),
        "owner_search_transition_batch_size": int(owner_search_transition_batch_size),
        "owner_search_direct_transition_batch_replay": bool(
            owner_search_direct_transition_batch_replay
        ),
        "owner_search_owner_local_transition_derivation": bool(
            owner_search_owner_local_transition_derivation
        ),
        "owner_search_owner_proxy_transition_closure": bool(
            owner_search_owner_proxy_transition_closure
        ),
        "owner_search_require_resident_root_view": bool(
            owner_search_require_resident_root_view
        ),
        "owner_search_resident_root_host_observation_stub": bool(
            owner_search_resident_root_host_observation_stub
        ),
        "owner_search_direct_root_build_request": bool(
            owner_search_direct_root_build_request
        ),
        "compact_owner_action_step_boundary": bool(compact_owner_action_step_boundary),
        "compact_owner_action_dispatch_step_overlap": bool(
            compact_owner_action_dispatch_step_overlap
        ),
        "owner_search_fixed_action_result_buffer": bool(
            owner_search_fixed_action_result_buffer
        ),
        "owner_search_action_result_slot_capacity": int(
            owner_search_action_result_slot_capacity
        ),
        "owner_search_fixed_soa_replay": bool(owner_search_fixed_soa_replay),
        "owner_search_defer_model_state_digest_to_refresh": bool(
            owner_search_defer_model_state_digest_to_refresh
        ),
        "owner_search_fixed_soa_locality_sample_group_size": int(
            owner_search_fixed_soa_locality_sample_group_size
        ),
        "owner_search_async_learner_worker": bool(owner_search_async_learner_worker),
        "owner_search_async_learner_worker_kind": str(owner_search_async_learner_worker_kind),
        "owner_search_async_learner_max_pending": int(owner_search_async_learner_max_pending),
        "compact_torch_request_compile": bool(compact_torch_request_compile),
        "compact_torch_request_model_compile": bool(compact_torch_request_model_compile),
        "compact_torch_model_compile_mode": str(compact_torch_model_compile_mode),
        "compact_torch_timing_mode": str(compact_torch_timing_mode),
        "compact_torch_initial_inference_mode": str(compact_torch_initial_inference_mode),
        "compact_torch_observation_memory_format": str(compact_torch_observation_memory_format),
        "compact_torch_model_memory_format": str(compact_torch_model_memory_format),
        "compact_torch_defer_one_simulation_replay_payload": bool(
            compact_torch_defer_one_simulation_replay_payload
        ),
        "seed": int(seed),
        "source_max_steps": int(source_max_steps),
        "decision_source_frames": int(decision_source_frames),
        "source_physics_step_ms": float(source_physics_step_ms),
        "source_max_steps_semantics": str(source_max_steps_semantics),
    }
    if bool(config["compact_owned_loop_deferred_learner"]) and bool(
        config["compact_owned_loop_deferred_sample_learner"]
    ):
        raise ValueError(
            "compact_owned_loop_deferred_sample_learner cannot be combined with "
            "compact_owned_loop_deferred_learner"
        )
    if int(config["compact_owned_loop_deferred_sample_learner_max_pending"]) <= 0:
        raise ValueError("compact_owned_loop_deferred_sample_learner_max_pending must be positive")
    if (
        str(config["compact_owned_loop_sample_learner_worker_kind"]) != "in_process_thread"
        and not bool(config["compact_owned_loop_deferred_sample_learner"])
    ):
        raise ValueError(
            "compact_owned_loop_sample_learner_worker_kind requires "
            "compact_owned_loop_deferred_sample_learner"
        )
    if float(config["gpu_utilization_sample_interval_sec"]) < 0.0:
        raise ValueError("gpu_utilization_sample_interval_sec must be non-negative")
    if int(config["owner_search_action_result_slot_capacity"]) <= 0:
        raise ValueError("owner_search_action_result_slot_capacity must be positive")
    if bool(config["owner_search_owner_local_transition_derivation"]):
        if not bool(config["owner_search_slab_bypass"]):
            raise ValueError(
                "owner_search_owner_local_transition_derivation requires "
                "owner_search_slab_bypass"
            )
        if int(config["owner_search_transition_batch_size"]) <= 1:
            raise ValueError(
                "owner_search_owner_local_transition_derivation requires "
                "owner_search_transition_batch_size > 1"
            )
        if not bool(config["owner_search_direct_transition_batch_replay"]):
            raise ValueError(
                "owner_search_owner_local_transition_derivation requires "
                "owner_search_direct_transition_batch_replay"
            )
    if bool(config["owner_search_owner_proxy_transition_closure"]):
        if not bool(config["owner_search_owner_local_transition_derivation"]):
            raise ValueError(
                "owner_search_owner_proxy_transition_closure requires "
                "owner_search_owner_local_transition_derivation"
            )
        if not bool(config["owner_search_direct_root_build_request"]):
            raise ValueError(
                "owner_search_owner_proxy_transition_closure requires "
                "owner_search_direct_root_build_request"
            )
    if bool(config["owner_search_resident_root_host_observation_stub"]) and not (
        bool(config["owner_search_slab_bypass"])
        and bool(config["owner_search_require_resident_root_view"])
    ):
        raise ValueError(
            "owner_search_resident_root_host_observation_stub requires "
            "owner_search_slab_bypass and owner_search_require_resident_root_view"
        )
    if bool(config["owner_search_direct_root_build_request"]) and not (
        bool(config["owner_search_slab_bypass"])
        and bool(config["owner_search_require_resident_root_view"])
        and bool(config["owner_search_resident_root_host_observation_stub"])
    ):
        raise ValueError(
            "owner_search_direct_root_build_request requires owner_search_slab_bypass, "
            "owner_search_require_resident_root_view, and "
            "owner_search_resident_root_host_observation_stub"
        )
    if bool(config["owner_search_direct_root_build_request"]) and str(
        config["search_service_kind"]
    ) not in {
        "owner_search_inline_proxy",
        "owner_search_inline_background_proxy",
        "owner_search_threaded_proxy",
    }:
        raise ValueError(
            "owner_search_direct_root_build_request requires an inline, "
            "inline-background, or threaded owner-search proxy"
        )
    if bool(config["compact_owner_action_step_boundary"]):
        if not bool(config["owner_search_direct_root_build_request"]):
            raise ValueError(
                "compact_owner_action_step_boundary requires "
                "owner_search_direct_root_build_request"
            )
        if not bool(config["owner_search_slab_bypass"]):
            raise ValueError(
                "compact_owner_action_step_boundary requires owner_search_slab_bypass"
            )
    if bool(config["compact_owner_action_dispatch_step_overlap"]):
        if not bool(config["compact_owner_action_step_boundary"]):
            raise ValueError(
                "compact_owner_action_dispatch_step_overlap requires "
                "compact_owner_action_step_boundary"
            )
        if not bool(config["owner_search_direct_root_build_request"]):
            raise ValueError(
                "compact_owner_action_dispatch_step_overlap requires "
                "owner_search_direct_root_build_request"
            )
    if bool(config["owner_search_fixed_action_result_buffer"]):
        if not bool(config["owner_search_defer_maintenance"]):
            raise ValueError(
                "owner_search_fixed_action_result_buffer requires "
                "owner_search_defer_maintenance"
            )
        if not bool(config["owner_search_direct_root_build_request"]):
            raise ValueError(
                "owner_search_fixed_action_result_buffer requires "
                "owner_search_direct_root_build_request"
            )
        if str(config["search_service_kind"]) not in {
            "owner_search_inline_proxy",
            "owner_search_inline_background_proxy",
            "owner_search_threaded_proxy",
        }:
            raise ValueError(
                "owner_search_fixed_action_result_buffer requires an inline, "
                "inline-background, or threaded owner-search proxy"
            )
    if speed_row_spawn_result:
        try:
            call = run_compact_coach_speed_row_h100.spawn(config)
        except Exception as exc:
            error_text = str(exc)
            resource_exhausted = (
                "RESOURCE_EXHAUSTED" in error_text
                or "memory usage is too high" in error_text
            )
            print(
                json.dumps(
                    {
                        "schema_id": SPAWN_SCHEMA_ID,
                        "ok": False,
                        "status": "spawn_failed",
                        "launch_failure": True,
                        "failure_stage": "launch",
                        "failure_phase": "modal_spawn",
                        "problem": (
                            "remote spawn resource exhausted"
                            if resource_exhausted
                            else "remote spawn failed"
                        ),
                        "error_type": type(exc).__name__,
                        "modal_error_code": (
                            "RESOURCE_EXHAUSTED" if resource_exhausted else ""
                        ),
                        "modal_resource_exhausted": resource_exhausted,
                        "function_call_id": "",
                        "app_name": APP_NAME,
                        "profile_only": False,
                        "calls_train_muzero": False,
                        "touches_live_runs": False,
                        "promotion_claim": False,
                        "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
                        "speed_currency": SPEED_CURRENCY,
                        "run_id": str(run_id),
                        "unified_lifecycle_report_ref": str(unified_lifecycle_report_ref),
                        "compact_checkpoint_ref": str(compact_checkpoint_ref),
                        "stderr_tail": error_text[-4000:],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        print(
            json.dumps(
                {
                    "schema_id": SPAWN_SCHEMA_ID,
                    "status": "spawned",
                    "launch_mode": "detached_function_call_result",
                    "result_capture": "modal_function_call_get",
                    "function_call_id": str(call.object_id),
                    "app_name": APP_NAME,
                    "profile_only": False,
                    "calls_train_muzero": False,
                    "touches_live_runs": False,
                    "promotion_claim": False,
                    "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
                    "speed_currency": SPEED_CURRENCY,
                    "compact_torch_observation_memory_format": str(
                        compact_torch_observation_memory_format
                    ),
                    "compact_torch_initial_inference_mode": str(
                        compact_torch_initial_inference_mode
                    ),
                    "compact_torch_model_memory_format": str(compact_torch_model_memory_format),
                    "compact_torch_defer_one_simulation_replay_payload_requested": bool(
                        compact_torch_defer_one_simulation_replay_payload
                    ),
                    "compact_torch_memory_format_applies_to_search_service": (
                        _config_uses_compact_torch_search_service(config)
                    ),
                    **_owner_search_config_fields_from_config(config),
                    "hybrid_persistent_compact_render_state_buffer": bool(
                        hybrid_persistent_compact_render_state_buffer
                    ),
                    "hybrid_borrow_single_actor_render_state": bool(
                        hybrid_borrow_single_actor_render_state
                    ),
                    "death_mode": str(death_mode),
                    "learner_num_unroll_steps": int(learner_num_unroll_steps),
                    "compact_owned_loop_deferred_learner": bool(
                        compact_owned_loop_deferred_learner
                    ),
                    "compact_owned_loop_deferred_sample_learner": bool(
                        compact_owned_loop_deferred_sample_learner
                    ),
                    "compact_owned_loop_deferred_sample_learner_max_pending": int(
                        compact_owned_loop_deferred_sample_learner_max_pending
                    ),
                    "compact_owned_loop_sample_learner_worker_kind_requested": str(
                        compact_owned_loop_sample_learner_worker_kind
                    ),
                    (
                        "compact_owned_loop_deferred_sample_learner_"
                        "replay_append_transport_kind_requested"
                    ): str(
                        compact_owned_loop_deferred_sample_learner_replay_append_transport_kind
                    ),
                    (
                        "compact_owned_loop_deferred_sample_learner_"
                        "model_state_transport_kind_requested"
                    ): str(
                        compact_owned_loop_deferred_sample_learner_model_state_transport_kind
                    ),
                    "compact_owned_loop_fused_learner_batch": bool(
                        compact_owned_loop_fused_learner_batch
                    ),
                    "compact_muzero_learner_batch_unroll2_specialized_builder": bool(
                        compact_muzero_learner_batch_unroll2_specialized_builder
                    ),
                    "compact_muzero_learner_batch_learner_ready_unroll2_cache": bool(
                        compact_muzero_learner_batch_learner_ready_unroll2_cache
                    ),
                    "compact_muzero_learner_batch_tensor_native_replay": bool(
                        compact_muzero_learner_batch_tensor_native_replay
                    ),
                    "owner_search_fixed_soa_replay": bool(owner_search_fixed_soa_replay),
                    "owner_search_fixed_soa_locality_sample_group_size": int(
                        owner_search_fixed_soa_locality_sample_group_size
                    ),
                    "compact_owned_lean_trainer_step": bool(compact_owned_lean_trainer_step),
                    "compact_owned_lean_profile_oracle_requested": bool(
                        compact_owned_lean_profile_oracle
                    ),
                    "compact_profile_bounded_diagnostics": bool(
                        compact_profile_bounded_diagnostics
                    ),
                    "compact_profile_cuda_sync_timing_diagnostics": bool(
                        compact_profile_cuda_sync_timing_diagnostics
                    ),
                    "compact_profile_runtime_step_timing_diagnostics": bool(
                        compact_profile_runtime_step_timing_diagnostics
                        or compact_profile_cuda_sync_timing_diagnostics
                    ),
                    "compact_profile_cpu_perf_stat_diagnostics": bool(
                        compact_profile_cpu_perf_stat_diagnostics
                    ),
                    "compact_owned_training_loop_owner": (
                        "lean_compact_trainer_step"
                        if bool(compact_owned_lean_trainer_step)
                        else "hybrid_observation_profile_runner"
                    ),
                    "speed_row_gpu_utilization_sampling_enabled": bool(
                        gpu_utilization_sampling
                    ),
                    "speed_row_gpu_utilization_sample_interval_sec": float(
                        gpu_utilization_sample_interval_sec
                    ),
                    "run_id": str(run_id),
                    "unified_lifecycle_report_ref": str(unified_lifecycle_report_ref),
                    "compact_checkpoint_ref": str(compact_checkpoint_ref),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    result = run_compact_coach_speed_row_h100.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not bool(result.get("ok")):
        raise RuntimeError(str(result.get("problem") or "speed row failed"))


def _required_ref(config: dict[str, Any], key: str) -> str:
    value = str(config.get(key) or "").strip().lstrip("/")
    if not value or ".." in Path(value).parts:
        raise ValueError(f"{key} must be a safe volume ref")
    return value


def _safe_id(raw: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    clean = "".join(char if char in allowed else "-" for char in str(raw)).strip("-.")
    if not clean:
        raise ValueError(f"cannot build safe id from {raw!r}")
    return clean


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "APP_NAME",
    "RESULT_BUNDLE_SCHEMA_ID",
    "SPAWN_SCHEMA_ID",
    "run_compact_coach_speed_row_h100",
]
