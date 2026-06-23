#!/usr/bin/env python3
"""Compare compact Coach speed-row artifacts without changing the denominator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA_ID = "curvyzero_compact_coach_speed_row_comparison/v1"
ACCEPTED_BASELINE_STEPS_PER_SEC = 12689.38
ACCEPTED_BASELINE_WALL_SEC = 14.5255
WHOLE_OWNER_BUFFER_REPLAY_CEILING_TARGET_MULTIPLIER = 2.0
WHOLE_OWNER_BUFFER_REPLAY_CEILING_SPEED_CURRENCY = "local_projection_no_speed"
REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT, REPO_ROOT / "src"):
    raw_import_path = str(import_path)
    if raw_import_path not in sys.path:
        sys.path.insert(0, raw_import_path)

IDENTITY_FIELDS = (
    "seed",
    "batch_size",
    "actor_count",
    "steps",
    "warmup_steps",
    "death_mode",
    "sample_seed_base",
    "sample_batch_size",
    "sample_interval",
    "replay_pair_capacity",
    "learner_train_steps",
    "learner_num_unroll_steps",
    "policy_refresh_interval",
    "num_simulations",
    "search_service_kind",
    "compact_profile_cuda_sync_timing_diagnostics",
    "compact_profile_runtime_step_timing_diagnostics",
    "compact_profile_cpu_perf_stat_diagnostics",
    "compact_profile_runtime_step_count",
    "compact_torch_initial_inference_mode",
    "compact_owned_loop_fused_learner_batch",
    "compact_owned_lean_trainer_step",
    "hybrid_borrow_single_actor_render_state",
    "render_state_copy_steps",
    "render_state_borrowed_steps",
    "terminal_sample_row_count",
    "terminal_unroll_value_target_row_count",
    "normal_death_terminal_contract_promotion_gate_satisfied",
    "truncated_row_count",
    "resident_observation_host_fallback_count",
    "compact_profile_autoreset_direct_count",
    "env_trajectory_ordered_checksum_total",
    "compact_rollout_slab_sample_gate_sample_position_order_checksum",
    "compact_rollout_slab_sample_gate_source_record_window_checksum",
    "compact_rollout_slab_sample_gate_calls",
    "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled",
    "compact_rollout_slab_sample_gate_cuda_sync_count",
    "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled",
    "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count",
    "compact_rollout_slab_learner_gate_updates",
    "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics",
    "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled",
    "compact_rollout_slab_learner_gate_cuda_sync_count",
    "compact_rollout_slab_policy_refresh_after_learner_gate_calls",
)
OPTIONAL_IDENTITY_FIELDS = (
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
SAMPLE_GATE_PER_CALL_TIMING_PREFIXES = (
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
BUILDER_CHILD_CPU_TIME_FIELDS = tuple(
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

TIMING_FIELDS = (
    "training_wall_sec",
    "steps_per_sec",
    "speed_row_actor_step_wall_sec",
    "speed_row_actor_autoreset_sec",
    "speed_row_actor_env_runtime_sec",
    "speed_row_actor_env_public_prepare_sec",
    "speed_row_actor_compact_write_sec",
    "speed_row_observation_sec",
    "speed_row_observation_other_sec",
    "speed_row_sample_gate_sec",
    "compact_rollout_slab_sample_gate_candidate_sec",
    "compact_rollout_slab_sample_gate_rng_sec",
    "compact_rollout_slab_sample_gate_learner_batch_build_sec",
    "compact_rollout_slab_sample_gate_residual_sec",
    "compact_rollout_slab_sample_gate_cuda_sync_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec",
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
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_sec",
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
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_builder_select_sec",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_row_index_prepare_sec",
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
    *BUILDER_CHILD_CPU_TIME_FIELDS,
    "speed_row_compact_rollout_slab_sec",
    "speed_row_learner_gate_sec",
    "compact_owner_search_worker_replay_append_sec",
    "speed_row_total_owner_search_worker_replay_append_sec",
    "compact_owner_search_owner_train_sample_sec",
    "compact_owner_search_owner_train_wall_sec",
    "compact_owner_search_owner_train_learner_update_sec",
    "compact_owner_search_worker_search_sec",
    "speed_row_total_owner_search_worker_search_sec",
    "compact_owner_search_parent_wait_sec",
    "speed_row_total_owner_search_parent_wait_sec",
    "compact_rollout_slab_learner_gate_backward_sec",
    "compact_rollout_slab_learner_gate_initial_inference_sec",
    "compact_rollout_slab_learner_gate_recurrent_inference_sec",
    "compact_rollout_slab_learner_gate_optimizer_step_sec",
    "compact_rollout_slab_learner_gate_cuda_sync_sec",
    "speed_row_policy_refresh_sec",
    "speed_row_resident_observation_stack_shift_sec",
    "speed_row_gpu_utilization_sample_count",
    "speed_row_gpu_utilization_max_percent",
    "speed_row_gpu_utilization_mean_percent",
    "speed_row_gpu_utilization_nonzero_sample_count",
    "speed_row_gpu_utilization_over_50_sample_count",
    "speed_row_gpu_utilization_over_80_sample_count",
    "speed_row_gpu_memory_utilization_max_percent",
    "speed_row_gpu_memory_used_max_mib",
    "speed_row_gpu_power_draw_max_w",
    "compact_profile_cpu_perf_stat_task_clock_sec",
    "compact_profile_cpu_perf_stat_cycles",
    "compact_profile_cpu_perf_stat_ref_cycles",
    "compact_profile_cpu_perf_stat_instructions",
    "compact_profile_cpu_perf_stat_instructions_per_cycle",
    "compact_profile_cpu_perf_stat_branches",
    "compact_profile_cpu_perf_stat_branch_misses",
    "compact_profile_cpu_perf_stat_cache_references",
    "compact_profile_cpu_perf_stat_cache_misses",
    "compact_profile_cpu_perf_stat_cache_miss_rate",
    "compact_profile_cpu_perf_stat_llc_loads",
    "compact_profile_cpu_perf_stat_llc_load_misses",
    "compact_profile_cpu_perf_stat_dtlb_loads",
    "compact_profile_cpu_perf_stat_dtlb_load_misses",
    "compact_profile_cpu_perf_stat_page_faults",
    "compact_profile_cpu_perf_stat_context_switches",
    "compact_profile_cpu_perf_stat_cpu_migrations",
    *(
        f"{prefix}_{stat_name}"
        for prefix in SAMPLE_GATE_PER_CALL_TIMING_PREFIXES
        for stat_name in ("sum_sec", "min_sec", "max_sec", "p50_sec", "p95_sec")
    ),
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
    "compact_profile_runtime_step_slowest_actor_step_wall_sec",
    "compact_profile_runtime_step_slowest_observation_sec",
    "compact_profile_runtime_step_slowest_compact_rollout_slab_sec",
    "compact_profile_runtime_step_slowest_sample_gate_sec",
    "compact_profile_runtime_step_slowest_learner_gate_sec",
    "compact_profile_runtime_step_slowest_policy_refresh_sec",
    "compact_profile_runtime_step_slowest_primary_accounted_sec",
    "compact_profile_runtime_step_slowest_primary_residual_sec",
)
UNROLL2_SPECIALIZED_BUILDER_KEY = (
    "compact_muzero_learner_batch_unroll2_specialized_builder"
)
UNROLL2_SPECIALIZED_BUILDER_PROOF_FIELDS = (
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl",
    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path",
)
GPU_UTILIZATION_INFO_FIELDS = (
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


def _is_wall_attribution_field(field: str) -> bool:
    if field in {"training_wall_sec", "steps_per_sec"}:
        return False
    if field.startswith("speed_row_gpu_"):
        return False
    if "_slowest_" in field:
        return False
    if not field.endswith("_sec"):
        return False
    return not field.endswith(("_min_sec", "_max_sec", "_p50_sec", "_p95_sec"))


WALL_ATTRIBUTION_FIELDS = tuple(
    field for field in TIMING_FIELDS if _is_wall_attribution_field(field)
)


def _load_launcher_module():
    module_name = "_compact_coach_speed_row_modal_smoke_for_compare"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).with_name("run_compact_coach_speed_row_modal_smoke.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


launcher = _load_launcher_module()


@dataclass(frozen=True)
class RowInput:
    label: str
    path: Path


def _summary_from_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: expected top-level JSON object with summary object")
    return summary


def _parse_row(value: str) -> RowInput:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--row must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--row label cannot be empty")
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{path} does not exist")
    return RowInput(label=label, path=path)


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _timing(summary: dict[str, Any], field: str) -> float | None:
    value = summary.get(field)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _identity(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: _jsonable_value(summary.get(field))
        for field in (*IDENTITY_FIELDS, *OPTIONAL_IDENTITY_FIELDS)
    }


def _timings(summary: dict[str, Any]) -> dict[str, float | None]:
    return {field: _timing(summary, field) for field in TIMING_FIELDS}


def _gpu_utilization(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        field: summary.get(field)
        for field in GPU_UTILIZATION_INFO_FIELDS
        if field in summary
    }


def _accepted_fast_path_violations(summary: dict[str, Any]) -> list[str]:
    args = argparse.Namespace(
        compact_owned_accepted_fast_path_preset=True,
        compact_owned_accepted_fast_path_step_window=(
            launcher._accepted_fast_path_step_window_from_summary(summary)
        ),
    )
    return launcher._accepted_fast_path_preset_violations(args, {"summary": summary})


def _accepted_fast_path_window_status(summary: dict[str, Any]) -> dict[str, Any]:
    window_name = launcher._accepted_fast_path_step_window_from_summary(summary)
    args = argparse.Namespace(
        compact_owned_accepted_fast_path_preset=True,
        compact_owned_accepted_fast_path_step_window=window_name,
    )
    return launcher._accepted_fast_path_step_window_report_fields(args)


def _unroll2_specialized_builder_violations(summary: dict[str, Any]) -> list[str]:
    requested_config = bool(summary.get(UNROLL2_SPECIALIZED_BUILDER_KEY, False))
    requested = bool(
        summary.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested",
            False,
        )
    )
    used = bool(
        summary.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used",
            False,
        )
    )
    call_count = int(
        summary.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count",
            0,
        )
        or 0
    )
    fallback_count = int(
        summary.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count",
            0,
        )
        or 0
    )
    fallback_reason = str(
        summary.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason"
        )
        or ""
    )
    impl = str(
        summary.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl"
        )
        or ""
    )
    path = str(
        summary.get("compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path")
        or ""
    )
    violations: list[str] = []
    if not requested_config:
        if requested or used or call_count > 0:
            violations.append("unroll2 specialized builder proof present without config")
        return violations
    if int(summary.get("learner_num_unroll_steps") or 0) != 2:
        violations.append("unroll2 specialized builder requires learner_num_unroll_steps=2")
    if not requested:
        violations.append("unroll2 specialized builder requested proof missing")
    if not used:
        violations.append("unroll2 specialized builder used proof missing")
    if call_count <= 0:
        violations.append("unroll2 specialized builder call_count must be positive")
    if fallback_count != 0:
        violations.append("unroll2 specialized builder fallback_count must be zero")
    if fallback_reason != "none":
        violations.append("unroll2 specialized builder fallback_reason must be none")
    if impl != "unroll2_specialized_v1":
        violations.append("unroll2 specialized builder impl mismatch")
    if path != "unroll2_specialized":
        violations.append("unroll2 specialized builder path mismatch")
    return violations


def _timing_deltas(
    baseline: dict[str, float | None],
    candidate: dict[str, float | None],
) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for field in TIMING_FIELDS:
        left = baseline.get(field)
        right = candidate.get(field)
        if left is None or right is None:
            continue
        delta = right - left
        deltas.append(
            {
                "field": field,
                "baseline": left,
                "candidate": right,
                "delta": delta,
                "abs_delta": abs(delta),
            }
        )
    deltas.sort(key=lambda item: float(item["abs_delta"]), reverse=True)
    return deltas


def _regression_buckets(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overall_fields = {"steps_per_sec", "training_wall_sec"}
    return [
        item
        for item in deltas
        if item["field"] not in overall_fields and float(item["delta"]) > 0.0
    ][:8]


def _identity_status(
    missing_fields: list[dict[str, Any]],
    mismatches: list[dict[str, Any]],
) -> str:
    if mismatches:
        return "mismatch"
    if missing_fields:
        return "partial_legacy"
    return "exact"


def _baseline_speed_status(
    timings: dict[str, float | None],
    *,
    accepted_baseline_steps_per_sec: float,
    accepted_baseline_wall_sec: float,
) -> dict[str, Any]:
    speed = timings.get("steps_per_sec")
    wall = timings.get("training_wall_sec")
    return {
        "accepted_baseline_steps_per_sec": accepted_baseline_steps_per_sec,
        "accepted_baseline_wall_sec": accepted_baseline_wall_sec,
        "speed_delta_vs_accepted_baseline": (
            None if speed is None else speed - accepted_baseline_steps_per_sec
        ),
        "wall_delta_vs_accepted_baseline": None if wall is None else wall - accepted_baseline_wall_sec,
        "beats_accepted_baseline": (
            speed is not None
            and wall is not None
            and speed > accepted_baseline_steps_per_sec
            and wall < accepted_baseline_wall_sec
        ),
    }


def _finite_nonnegative_summary_max(
    summary: dict[str, Any],
    *fields: str,
) -> float:
    values: list[float] = []
    for field in fields:
        value = summary.get(field)
        try:
            number = float(value or 0.0)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0.0:
            values.append(number)
    return max(values) if values else 0.0


def _whole_owner_buffer_replay_ceiling(
    summary: dict[str, Any],
    *,
    accepted_baseline_steps_per_sec: float,
) -> dict[str, Any]:
    steps = _finite_nonnegative_summary_max(summary, "steps")
    batch_size = _finite_nonnegative_summary_max(summary, "batch_size")
    env_steps = _finite_nonnegative_summary_max(summary, "env_steps_collected")
    if env_steps <= 0.0 and steps > 0.0 and batch_size > 0.0:
        env_steps = steps * batch_size
    wall_sec = _finite_nonnegative_summary_max(summary, "training_wall_sec")
    observed_speed = _finite_nonnegative_summary_max(summary, "steps_per_sec")
    if observed_speed <= 0.0 and env_steps > 0.0 and wall_sec > 0.0:
        observed_speed = env_steps / wall_sec

    replay_append_sec = _finite_nonnegative_summary_max(
        summary,
        "compact_owner_search_worker_replay_append_sec",
        "speed_row_total_owner_search_worker_replay_append_sec",
    )
    owner_train_sample_sec = _finite_nonnegative_summary_max(
        summary,
        "compact_owner_search_owner_train_sample_sec",
    )
    owner_train_wall_sec = _finite_nonnegative_summary_max(
        summary,
        "compact_owner_search_owner_train_wall_sec",
    )
    learner_update_sec = _finite_nonnegative_summary_max(
        summary,
        "compact_owner_search_owner_train_learner_update_sec",
    )
    worker_search_sec = _finite_nonnegative_summary_max(
        summary,
        "compact_owner_search_worker_search_sec",
        "speed_row_total_owner_search_worker_search_sec",
    )
    parent_wait_sec = _finite_nonnegative_summary_max(
        summary,
        "compact_owner_search_parent_wait_sec",
        "speed_row_total_owner_search_parent_wait_sec",
    )

    direct_surface_sec = replay_append_sec + owner_train_sample_sec
    parent_wait_bounded_surface_sec = min(parent_wait_sec, direct_surface_sec)
    preserved_floor_sec = worker_search_sec + learner_update_sec
    max_removable_sec = max(0.0, wall_sec - preserved_floor_sec)
    projected_removed_sec = min(parent_wait_bounded_surface_sec, max_removable_sec)
    projected_wall_sec = max(0.0, wall_sec - projected_removed_sec)
    projected_speed = env_steps / projected_wall_sec if projected_wall_sec > 0.0 else 0.0
    target_speed = (
        accepted_baseline_steps_per_sec
        * WHOLE_OWNER_BUFFER_REPLAY_CEILING_TARGET_MULTIPLIER
    )
    target_wall_sec = env_steps / target_speed if env_steps > 0.0 and target_speed > 0.0 else 0.0
    baseline_wall_sec = (
        env_steps / accepted_baseline_steps_per_sec
        if env_steps > 0.0 and accepted_baseline_steps_per_sec > 0.0
        else 0.0
    )
    enabled = (
        env_steps > 0.0
        and wall_sec > 0.0
        and (direct_surface_sec > 0.0 or parent_wait_sec > 0.0)
    )
    return {
        "enabled": enabled,
        "projection_only": True,
        "production_speed_claim": False,
        "touches_live_training": False,
        "requires_h100_validation": True,
        "speed_currency": WHOLE_OWNER_BUFFER_REPLAY_CEILING_SPEED_CURRENCY,
        "projection_source": "compare_rows_existing_summary_fields_v1",
        "basis": "owner_replay_append_train_sample_parent_wait_bound_v1",
        "promotion_eligible": False,
        "observed_env_steps": env_steps,
        "observed_wall_sec": wall_sec,
        "observed_env_steps_per_sec": observed_speed,
        "baseline_env_steps_per_sec": accepted_baseline_steps_per_sec,
        "baseline_whole_loop_sec": baseline_wall_sec,
        "target_multiplier": WHOLE_OWNER_BUFFER_REPLAY_CEILING_TARGET_MULTIPLIER,
        "target_env_steps_per_sec": target_speed,
        "target_wall_sec": target_wall_sec,
        "observed_speedup_vs_accepted_baseline": (
            observed_speed / accepted_baseline_steps_per_sec
            if accepted_baseline_steps_per_sec > 0.0
            else 0.0
        ),
        "observed_replay_append_sec": replay_append_sec,
        "observed_owner_train_sample_sec": owner_train_sample_sec,
        "observed_owner_train_wall_sec": owner_train_wall_sec,
        "observed_learner_update_sec": learner_update_sec,
        "observed_worker_search_sec": worker_search_sec,
        "observed_parent_wait_sec": parent_wait_sec,
        "direct_replay_sample_surface_sec": direct_surface_sec,
        "parent_wait_bounded_surface_sec": parent_wait_bounded_surface_sec,
        "preserved_search_update_floor_sec": preserved_floor_sec,
        "max_removable_sec": max_removable_sec,
        "projected_removed_sec": projected_removed_sec,
        "projected_wall_sec": projected_wall_sec,
        "projected_env_steps_per_sec": projected_speed,
        "projected_speedup_vs_accepted_baseline": (
            projected_speed / accepted_baseline_steps_per_sec
            if accepted_baseline_steps_per_sec > 0.0
            else 0.0
        ),
        "projected_delta_sec": wall_sec - projected_wall_sec,
        "projected_reaches_2x": enabled and projected_speed >= target_speed,
        "additional_removed_sec_to_2x": max(0.0, projected_wall_sec - target_wall_sec),
    }


def _whole_owner_buffer_replay_ceiling_rank(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = []
    for row in rows:
        ceiling = row.get("whole_owner_buffer_replay_ceiling")
        if not isinstance(ceiling, dict) or not bool(ceiling.get("enabled")):
            continue
        ranked.append(
            {
                "label": row["label"],
                "path": row["path"],
                "observed_env_steps_per_sec": ceiling["observed_env_steps_per_sec"],
                "projected_env_steps_per_sec": ceiling["projected_env_steps_per_sec"],
                "projected_speedup_vs_accepted_baseline": ceiling[
                    "projected_speedup_vs_accepted_baseline"
                ],
                "projected_reaches_2x": ceiling["projected_reaches_2x"],
                "additional_removed_sec_to_2x": ceiling[
                    "additional_removed_sec_to_2x"
                ],
                "projected_wall_sec": ceiling["projected_wall_sec"],
                "target_wall_sec": ceiling["target_wall_sec"],
            }
        )
    ranked.sort(
        key=lambda item: float(item["projected_env_steps_per_sec"]),
        reverse=True,
    )
    return ranked


def _wall_delta(comparison: dict[str, Any]) -> float:
    for item in comparison["largest_timing_deltas"]:
        if item["field"] == "training_wall_sec":
            return float(item["delta"])
    return 0.0


def _range_metric(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    values: list[tuple[str, float]] = []
    for row in rows:
        value = row["timings"].get(field)
        if value is None:
            continue
        values.append((str(row["label"]), float(value)))
    if not values:
        return None
    numbers = [value for _label, value in values]
    min_label, min_value = min(values, key=lambda item: item[1])
    max_label, max_value = max(values, key=lambda item: item[1])
    spread = max_value - min_value
    center = float(median(numbers))
    return {
        "field": field,
        "count": len(values),
        "min": min_value,
        "min_label": min_label,
        "max": max_value,
        "max_label": max_label,
        "median": center,
        "range": spread,
        "spread_pct_of_median": None if center == 0.0 else (spread / center) * 100.0,
    }


def _exact_repeat_stability(
    exact_rows: list[dict[str, Any]],
    *,
    exact_candidate_count: int,
) -> dict[str, Any] | None:
    if exact_candidate_count <= 0:
        return None
    ranges = [
        metric
        for field in TIMING_FIELDS
        if (metric := _range_metric(exact_rows, field)) is not None
    ]
    ranges.sort(key=lambda item: abs(float(item["range"])), reverse=True)
    return {
        "exact_row_count": len(exact_rows),
        "exact_candidate_count": exact_candidate_count,
        "labels": [str(row["label"]) for row in exact_rows],
        "training_wall_sec": _range_metric(exact_rows, "training_wall_sec"),
        "steps_per_sec": _range_metric(exact_rows, "steps_per_sec"),
        "largest_timing_ranges": ranges[:12],
        "wall_swing_attribution": _wall_swing_attribution(exact_rows),
    }


def _wall_swing_attribution(exact_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    wall_rows: list[tuple[dict[str, Any], float]] = []
    for row in exact_rows:
        wall = row["timings"].get("training_wall_sec")
        if wall is None:
            continue
        wall_rows.append((row, float(wall)))
    if len(wall_rows) < 2:
        return None
    fastest_row, fastest_wall = min(wall_rows, key=lambda item: item[1])
    slowest_row, slowest_wall = max(wall_rows, key=lambda item: item[1])
    wall_delta = slowest_wall - fastest_wall
    if wall_delta <= 0.0:
        return None
    deltas: list[dict[str, Any]] = []
    for field in WALL_ATTRIBUTION_FIELDS:
        fastest_value = fastest_row["timings"].get(field)
        slowest_value = slowest_row["timings"].get(field)
        if fastest_value is None or slowest_value is None:
            continue
        delta = float(slowest_value) - float(fastest_value)
        deltas.append(
            {
                "field": field,
                "fastest_value": float(fastest_value),
                "slowest_value": float(slowest_value),
                "delta": delta,
                "abs_delta": abs(delta),
                "pct_of_wall_delta": (delta / wall_delta) * 100.0,
                "abs_pct_of_wall_delta": (abs(delta) / wall_delta) * 100.0,
            }
        )
    deltas.sort(key=lambda item: float(item["abs_delta"]), reverse=True)
    positive_deltas = [item for item in deltas if float(item["delta"]) > 0.0]
    return {
        "fastest_label": str(fastest_row["label"]),
        "fastest_wall_sec": fastest_wall,
        "slowest_label": str(slowest_row["label"]),
        "slowest_wall_sec": slowest_wall,
        "wall_delta_sec": wall_delta,
        "wall_delta_pct_of_fastest": (wall_delta / fastest_wall) * 100.0,
        "timing_field_count": len(deltas),
        "largest_positive_deltas": positive_deltas[:12],
        "largest_absolute_deltas": deltas[:12],
    }


def compare_rows(
    rows: list[RowInput],
    *,
    baseline_label: str | None = None,
    accepted_baseline_steps_per_sec: float = ACCEPTED_BASELINE_STEPS_PER_SEC,
    accepted_baseline_wall_sec: float = ACCEPTED_BASELINE_WALL_SEC,
) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("at least two rows are required")
    labels = [row.label for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError("row labels must be unique")
    baseline_label = baseline_label or rows[0].label
    by_label: dict[str, dict[str, Any]] = {}
    for row in rows:
        summary = _summary_from_artifact(row.path)
        timings = _timings(summary)
        by_label[row.label] = {
            "label": row.label,
            "path": str(row.path),
            **_accepted_fast_path_window_status(summary),
            "accepted_fast_path_violations": _accepted_fast_path_violations(summary),
            "unroll2_specialized_builder_violations": (
                _unroll2_specialized_builder_violations(summary)
            ),
            "unroll2_specialized_builder_proof": {
                UNROLL2_SPECIALIZED_BUILDER_KEY: summary.get(UNROLL2_SPECIALIZED_BUILDER_KEY),
                **{
                    field: summary.get(field)
                    for field in UNROLL2_SPECIALIZED_BUILDER_PROOF_FIELDS
                },
            },
            "identity": _identity(summary),
            "timings": timings,
            "whole_owner_buffer_replay_ceiling": _whole_owner_buffer_replay_ceiling(
                summary,
                accepted_baseline_steps_per_sec=accepted_baseline_steps_per_sec,
            ),
            "gpu_utilization": _gpu_utilization(summary),
            "accepted_baseline_status": _baseline_speed_status(
                timings,
                accepted_baseline_steps_per_sec=accepted_baseline_steps_per_sec,
                accepted_baseline_wall_sec=accepted_baseline_wall_sec,
            ),
        }
    if baseline_label not in by_label:
        raise ValueError(f"unknown baseline label: {baseline_label}")
    baseline = by_label[baseline_label]
    comparisons: list[dict[str, Any]] = []
    baseline_identity = baseline["identity"]
    baseline_timings = baseline["timings"]
    for label, row in by_label.items():
        if label == baseline_label:
            continue
        identity_missing_fields: list[dict[str, Any]] = []
        identity_mismatches: list[dict[str, Any]] = []
        for field in IDENTITY_FIELDS:
            baseline_value = baseline_identity.get(field)
            candidate_value = row["identity"].get(field)
            item = {
                "field": field,
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
            if baseline_value is None or candidate_value is None:
                identity_missing_fields.append(item)
            elif baseline_value != candidate_value:
                identity_mismatches.append(item)
        for field in OPTIONAL_IDENTITY_FIELDS:
            baseline_value = baseline_identity.get(field)
            candidate_value = row["identity"].get(field)
            if baseline_value is None and candidate_value is None:
                continue
            item = {
                "field": field,
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
            if baseline_value is None or candidate_value is None:
                identity_missing_fields.append(item)
            elif baseline_value != candidate_value:
                identity_mismatches.append(item)
        timing_deltas = _timing_deltas(baseline_timings, row["timings"])
        identity_status = _identity_status(identity_missing_fields, identity_mismatches)
        comparisons.append(
            {
                "baseline_label": baseline_label,
                "candidate_label": label,
                "identity_status": identity_status,
                "identity_missing_fields": identity_missing_fields,
                "identity_mismatches": identity_mismatches,
                "identity_matches": identity_status == "exact",
                "largest_timing_deltas": timing_deltas[:12],
                "largest_regression_buckets": _regression_buckets(timing_deltas),
            }
        )
    exact_candidate_labels = [
        comparison["candidate_label"]
        for comparison in comparisons
        if comparison["identity_status"] == "exact"
    ]
    exact_labels = [baseline_label, *exact_candidate_labels]
    exact_rows = [by_label[label] for label in exact_labels]
    stable_speed_claim_allowed = bool(exact_candidate_labels) and all(
        (not bool(row["compact_owned_accepted_fast_path_stability_diagnostic"]))
        and bool(row["accepted_baseline_status"]["beats_accepted_baseline"])
        and not bool(row["accepted_fast_path_violations"])
        and not bool(row["unroll2_specialized_builder_violations"])
        for row in exact_rows
    )
    exact_comparisons = [
        comparison for comparison in comparisons if comparison["identity_status"] == "exact"
    ]
    row_values = list(by_label.values())
    return {
        "schema_id": SCHEMA_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_label": baseline_label,
        "accepted_baseline": {
            "steps_per_sec": accepted_baseline_steps_per_sec,
            "wall_sec": accepted_baseline_wall_sec,
        },
        "stable_speed_claim_allowed": stable_speed_claim_allowed,
        "exact_repeat_stability": _exact_repeat_stability(
            exact_rows,
            exact_candidate_count=len(exact_candidate_labels),
        ),
        "largest_exact_wall_swing": (
            None
            if not exact_comparisons
            else max(exact_comparisons, key=lambda comparison: abs(_wall_delta(comparison)))
        ),
        "rows": row_values,
        "whole_owner_buffer_replay_ceiling_rank": (
            _whole_owner_buffer_replay_ceiling_rank(row_values)
        ),
        "comparisons": comparisons,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--row",
        action="append",
        type=_parse_row,
        required=True,
        help="Row artifact in LABEL=PATH form. Repeat at least twice.",
    )
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument(
        "--accepted-baseline-steps-per-sec",
        type=float,
        default=ACCEPTED_BASELINE_STEPS_PER_SEC,
    )
    parser.add_argument(
        "--accepted-baseline-wall-sec",
        type=float,
        default=ACCEPTED_BASELINE_WALL_SEC,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = compare_rows(
        args.row,
        baseline_label=args.baseline_label,
        accepted_baseline_steps_per_sec=float(args.accepted_baseline_steps_per_sec),
        accepted_baseline_wall_sec=float(args.accepted_baseline_wall_sec),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
