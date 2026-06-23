#!/usr/bin/env python3
"""Fixed-action env/outer-loop toy for CurvyTron vector mechanics.

This is architecture proof machinery, not speed evidence. It compares:

1. the existing compact-profile env step path, and
2. a direct fixed-buffer loop that calls the same vector runtime kernel but
   skips public/profile result packing.

The goal is to test whether the outer manager/profile object path can explain
the large primary residual seen after OPT-132BF, before spending another H100
row.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for root in (SCRIPT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402
from curvyzero.env.observation_surface_contract import POLICY_STACK_SHAPE  # noqa: E402
from curvyzero.env import vector_multiplayer_env as vector_env_mod  # noqa: E402
from curvyzero.env import vector_runtime  # noqa: E402
from curvyzero.training.compact_observation_contract import (  # noqa: E402
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
)
from curvyzero.env.vector_visual_observation import (  # noqa: E402
    SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH,
    source_state_canvas_gray64_schema,
)
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv  # noqa: E402
from curvyzero.training.compact_policy_row_bridge import (  # noqa: E402
    build_compact_device_replay_index_rows_v1_from_owner_action_context_payload,
    build_compact_root_batch_v1,
    compact_root_action_context_v1_from_request,
)
from curvyzero.training.compact_rollout_slab import (  # noqa: E402
    CompactRolloutSlab,
    CompactOwnerSearchDirectStepperV1,
)
from curvyzero.training.compact_rollout_slab import (  # noqa: E402
    selected_joint_action_from_action_step,
)
from curvyzero.training.compact_search_service import (  # noqa: E402
    COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
    COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
    CompactDeviceSearchReplayPayloadV1,
    CompactSearchActionStepV1,
    compact_search_array_digest_v1,
    compact_search_deferred_replay_payload_digest_v1,
)
from curvyzero.training.fixed_shape_batched_search_owner import (  # noqa: E402
    FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL,
    FixedShapeBatchedSearchOwnerV0,
)
from curvyzero.training.multiplayer_source_state_target_rows import (  # noqa: E402
    ACTION_COUNT,
    DEFAULT_TO_PLAY,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (  # noqa: E402
    SourceStateMultiplayerTrainerReplayChunkV0,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError  # noqa: E402
from curvyzero.training.source_state_batched_observation_profile import (  # noqa: E402
    CpuOracleBatchedObservationRenderer,
    SourceStateBatchedRenderRequest,
)
from curvyzero.training.source_state_hybrid_observation_profile import (  # noqa: E402
    CompactReplayColumnarAppendRecordV1,
    HybridCompactBatch,
    _CompactReplayRingV1,
    _make_compact_owner_mechanics_step_frame_ring,
    _publish_compact_owner_mechanics_step_frame_slot,
)


SCHEMA_VERSION = "curvyzero_vector_fixed_action_tape_benchmark/v1"
OWNER_BUFFER_CEILING_SCHEMA_VERSION = (
    "curvyzero_vector_fixed_action_tape_owner_buffer_ceiling/v1"
)
DEFAULT_SCENARIO = (
    "scenarios/environment/source_borderless_wrap_skips_destination_body_then_next_frame_kills.json"
)

STATE_CHECKSUM_FIELDS = (
    "tick",
    "episode_step",
    "done",
    "terminated",
    "truncated",
    "terminal_reason",
    "pos",
    "prev_pos",
    "heading",
    "alive",
    "present",
    "score",
    "round_score",
    "winner",
    "draw",
    "death_count",
    "death_player",
    "death_cause",
    "death_hit_owner",
    "world_body_count",
    "body_active",
    "body_pos",
    "body_radius",
    "body_owner",
    "body_num",
    "body_write_cursor",
)

BODY_CHECKSUM_FIELDS = (
    "world_body_count",
    "body_active",
    "body_pos",
    "body_radius",
    "body_owner",
    "body_num",
    "body_write_cursor",
)

DEATH_CHECKSUM_FIELDS = (
    "death_count",
    "death_player",
    "death_cause",
    "death_hit_owner",
    "winner",
    "draw",
    "terminal_reason",
)

OUTPUT_CHECKSUM_FIELDS = (
    "reward",
    "final_reward_map",
    "done",
    "terminated",
    "truncated",
    "terminal_reason",
    "death_count",
    "death_player",
    "death_cause",
    "death_hit_owner",
    "winner",
    "draw",
    "action_mask",
    "terminal_rows",
    "source_physics_substeps_executed",
    "source_physics_elapsed_ms",
)

EXPECTED_DEATH_CAUSE_NAMES_BY_SCENARIO = {
    "source_borderless_wrap_skips_destination_body_then_next_frame_kills": (
        "opponent_trail",
    ),
    "source_normal_wall_death_step": ("wall",),
}

EXPECTED_TERMINAL_ROWS_BY_SCENARIO = {
    "source_normal_wall_death_step": True,
}

SEARCH_METADATA_COMPARE_FIELDS = (
    "search_enabled",
    "search_impl",
    "search_schema_id",
    "root_batch_schema_id",
    "search_call_count",
    "search_root_count",
    "search_active_root_count",
    "search_inactive_root_count",
    "search_selected_action_count",
    "search_max_active_root_count",
    "search_action_count",
    "search_num_simulations",
    "search_first_legal_policy",
    "search_two_phase_action_only",
    "search_ctree_calls",
    "search_tolist_calls",
    "search_per_sim_d2h_bytes",
    "search_root_observation_copy_bytes",
    "search_action_d2h_bytes",
    "search_deferred_replay_payload_d2h_bytes",
    "search_preallocated_buffer_bytes",
    "search_buffer_reused",
    "search_action_step_identity_checked",
    "search_action_step_root_index_matches_active",
    "search_action_step_env_row_matches_root",
    "search_action_step_player_matches_root",
    "search_action_step_policy_env_id_matches_root",
    "search_selected_action_shape_matches",
    "search_selected_action_legal",
    "search_replay_payload_digest_deferred",
    "search_replay_payload_digest_matches_handle",
    "search_selected_action_digest_matches_payload",
    "search_root_batch_observation_source",
    "search_root_batch_observation_copied",
    "search_root_batch_observation_shape",
    "search_root_batch_observation_dtype",
    "search_root_batch_row_major_sidecars_checked",
    "search_done_root_matches_repeat_done",
    "search_active_root_mask_matches_non_done_legal",
    "search_to_play_all_default",
    "search_target_reward_matches_reward",
    "search_root_observation_shares_stack",
    "search_selected_action_digest",
    "search_replay_payload_digest",
    "search_root_batch_checksum",
    "search_action_step_checksum",
    "search_root_observation_checksum",
    "search_selected_action_checksum",
    "search_joint_action_checksum",
)

REPLAY_METADATA_COMPARE_FIELDS = (
    "slab_replay_enabled",
    "slab_search_feedback_closed_loop",
    "slab_replay_tape_bootstrap_action_count",
    "slab_replay_feedback_action_count",
    "slab_replay_measured_feedback_action_count",
    "slab_replay_prev_next_joint_action_match_count",
    "slab_replay_prev_next_joint_action_mismatch_count",
    "slab_replay_feedback_differs_from_tape_count",
    "slab_replay_action_source_sequence_checksum",
    "slab_step_count",
    "slab_root_count",
    "slab_active_root_count",
    "slab_inactive_root_count",
    "slab_selected_action_count",
    "slab_max_active_root_count",
    "slab_action_count",
    "slab_num_simulations",
    "slab_ctree_calls",
    "slab_tolist_calls",
    "slab_per_sim_d2h_bytes",
    "slab_action_d2h_bytes",
    "slab_deferred_replay_payload_d2h_bytes",
    "slab_root_observation_copy_bytes",
    "slab_committed_index_group_count",
    "slab_committed_index_row_count",
    "slab_committed_terminal_row_count",
    "slab_committed_next_final_observation_row_count",
    "slab_replay_payload_flush_count",
    "slab_replay_payload_d2h_bytes",
    "slab_replay_index_rows_observation_materialized",
    "slab_replay_index_rows_next_observation_materialized",
    "slab_replay_sample_batch_built",
    "slab_replay_sample_batch_size",
    "slab_replay_sample_seed",
    "slab_replay_sample_row_id_checksum",
    "slab_replay_sample_action_checksum",
    "slab_replay_sample_observation_checksum",
    "slab_replay_sample_next_observation_checksum",
    "slab_replay_sample_record_index_checksum",
    "slab_replay_sample_policy_row_checksum",
    "slab_replay_index_rows_checksum",
    "slab_replay_joint_action_feedback_checksum",
    "slab_replay_root_batch_checksum",
    "slab_replay_action_step_checksum",
    "slab_next_joint_action_checksum",
    "slab_replay_pending_uncommitted_count",
    "slab_replay_action_check_enforced",
    "slab_replay_root_observation_copied",
    "slab_retains_committed_index_rows",
    "replay_append_count",
    "replay_ring_entry_count",
    "replay_ring_stored_index_row_count",
    "replay_ring_evicted_entry_count",
    "replay_ring_evicted_index_row_count",
    "sample_gate_calls",
    "sample_row_count",
    "sample_target_row_count",
    "sample_seed",
    "sampled_flat_row_checksum",
    "sample_position_order_checksum",
    "source_record_pair_checksum",
    "source_record_window_checksum",
    "sample_row_id_checksum",
    "sample_action_checksum",
    "sample_observation_checksum",
    "sample_next_observation_checksum",
    "sample_reward_checksum",
    "sample_done_checksum",
)

OWNER_SLOT_METADATA_COMPARE_FIELDS = (
    "owner_slot_ceiling_enabled",
    "owner_slot_ceiling_step_count",
    "owner_slot_ceiling_tape_bootstrap_action_count",
    "owner_slot_ceiling_feedback_action_count",
    "owner_slot_ceiling_measured_feedback_action_count",
    "owner_slot_ceiling_prev_next_joint_action_match_count",
    "owner_slot_ceiling_prev_next_joint_action_mismatch_count",
    "owner_slot_ceiling_mechanics_slot_write_count",
    "owner_slot_ceiling_mechanics_slot_generation_verified_count",
    "owner_slot_ceiling_mechanics_slot_digest_verified_count",
    "owner_slot_ceiling_root_request_from_slot_count",
    "owner_slot_ceiling_root_request_from_batch_count",
    "owner_slot_ceiling_hybrid_compact_batch_object_count",
    "owner_slot_ceiling_action_result_write_count",
    "owner_slot_ceiling_action_result_read_count",
    "owner_slot_ceiling_next_action_count",
    "owner_slot_ceiling_root_observation_copy_bytes",
    "owner_slot_ceiling_active_root_count",
    "owner_slot_ceiling_selected_action_count",
    "owner_slot_ceiling_ctree_calls",
    "owner_slot_ceiling_tolist_calls",
    "owner_slot_ceiling_replay_slot_append_count",
    "owner_slot_ceiling_replay_slot_append_row_count",
    "owner_slot_ceiling_replay_slot_object_entry_count",
    "owner_slot_ceiling_parent_replay_object_count",
    "owner_slot_ceiling_selected_group_object_count",
    "owner_slot_ceiling_sample_batch_built",
    "owner_slot_ceiling_sample_gate_calls",
    "owner_slot_ceiling_sample_handle_create_count",
    "owner_slot_ceiling_sample_handle_resolve_count",
    "owner_slot_ceiling_sample_handle_inline_resolve_count",
    "owner_slot_ceiling_sample_handle_pending_count",
    "owner_slot_ceiling_sample_row_count",
    "owner_slot_ceiling_sample_target_row_count",
    "owner_slot_ceiling_replay_slot_window_checksum",
    "owner_slot_ceiling_sample_handle_checksum",
    "owner_slot_ceiling_sample_row_id_checksum",
    "owner_slot_ceiling_sample_action_checksum",
    "owner_slot_ceiling_sample_reward_checksum",
    "owner_slot_ceiling_sample_done_checksum",
    "owner_slot_ceiling_stage_replay_transport_entry_count",
    "owner_slot_ceiling_stage_replay_transition_entry_count",
    "owner_slot_ceiling_stage_replay_payload_cache_hit_count",
    "owner_slot_ceiling_stage_replay_payload_cache_miss_count",
    "owner_slot_ceiling_stage_replay_payload_release_count",
    "owner_slot_ceiling_stage_replay_payload_pending_count",
    "owner_slot_ceiling_stage_replay_pending_record_count",
    "owner_slot_ceiling_stage_replay_ready_record_count",
    "owner_slot_ceiling_stage_replay_drained_record_count",
    "owner_slot_ceiling_stage_replay_index_rows_build_count",
    "owner_slot_ceiling_stage_replay_index_rows_row_count",
    "owner_slot_ceiling_stage_replay_device_index_rows_build_count",
    "owner_slot_ceiling_stage_replay_device_index_rows_row_count",
    "owner_slot_ceiling_stage_replay_slot_append_count",
    "owner_slot_ceiling_stage_replay_slot_append_row_count",
    "owner_slot_ceiling_stage_sample_batch_built",
    "owner_slot_ceiling_stage_sample_gate_calls",
    "owner_slot_ceiling_stage_sample_handle_create_count",
    "owner_slot_ceiling_stage_sample_handle_resolve_count",
    "owner_slot_ceiling_stage_sample_handle_inline_resolve_count",
    "owner_slot_ceiling_stage_sample_handle_pending_count",
    "owner_slot_ceiling_stage_sample_row_count",
    "owner_slot_ceiling_stage_sample_target_row_count",
    "owner_slot_ceiling_stage_replay_slot_window_checksum",
    "owner_slot_ceiling_stage_sample_handle_checksum",
    "owner_slot_ceiling_stage_sample_row_id_checksum",
    "owner_slot_ceiling_stage_sample_action_checksum",
    "owner_slot_ceiling_stage_sample_reward_checksum",
    "owner_slot_ceiling_stage_sample_done_checksum",
    "owner_slot_ceiling_replay_ring_append_record_count",
    "owner_slot_ceiling_replay_ring_append_call_count",
    "owner_slot_ceiling_replay_ring_appended_row_count",
    "owner_slot_ceiling_replay_ring_entry_count",
    "owner_slot_ceiling_replay_ring_stored_index_row_count",
    "owner_slot_ceiling_replay_ring_evicted_entry_count",
    "owner_slot_ceiling_replay_ring_evicted_index_row_count",
    "owner_slot_ceiling_replay_ring_sample_batch_built",
    "owner_slot_ceiling_replay_ring_sample_gate_calls",
    "owner_slot_ceiling_replay_ring_sample_row_count",
    "owner_slot_ceiling_replay_ring_sample_target_row_count",
    "owner_slot_ceiling_replay_ring_sample_source",
    "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample",
    "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample_all",
    "owner_slot_ceiling_replay_ring_sample_resident_device_sample_batch",
    "owner_slot_ceiling_replay_ring_sample_host_observation_fallback_allowed",
    "owner_slot_ceiling_replay_ring_sample_observation_provider_used_count",
    "owner_slot_ceiling_replay_ring_sample_row_id_checksum",
    "owner_slot_ceiling_replay_ring_sample_action_checksum",
    "owner_slot_ceiling_replay_ring_sample_reward_checksum",
    "owner_slot_ceiling_replay_ring_sample_done_checksum",
    "owner_slot_ceiling_replay_ring_sample_observation_checksum",
    "owner_slot_ceiling_replay_ring_sample_next_observation_checksum",
    "owner_slot_ceiling_replay_ring_learner_unroll2_batch_built",
    "owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls",
    "owner_slot_ceiling_replay_ring_learner_unroll2_sample_row_count",
    "owner_slot_ceiling_replay_ring_learner_unroll2_target_row_count",
    "owner_slot_ceiling_replay_ring_learner_unroll2_num_unroll_steps",
    "owner_slot_ceiling_replay_ring_learner_unroll2_require_next_targets",
    "owner_slot_ceiling_replay_ring_learner_unroll2_batch_only",
    "owner_slot_ceiling_replay_ring_learner_unroll2_source",
    "owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source",
    "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count",
    "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count",
    "owner_slot_ceiling_replay_ring_learner_unroll2_observation_provider_used_count",
    "owner_slot_ceiling_replay_ring_learner_unroll2_schema_id",
    "owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source",
    "owner_slot_ceiling_replay_ring_learner_unroll2_host_fallback_allowed",
    "owner_slot_ceiling_replay_ring_learner_unroll2_action_shape",
    "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_shape",
    "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_shape",
    "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_shape",
    "owner_slot_ceiling_replay_ring_learner_unroll2_action_mask_shape",
    "owner_slot_ceiling_replay_ring_learner_unroll2_action_checksum",
    "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_checksum",
    "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_checksum",
    "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_checksum",
    "owner_slot_ceiling_replay_ring_learner_unroll2_source_record_window_checksum",
    "owner_slot_ceiling_replay_ring_columnar_append_call_count",
    "owner_slot_ceiling_replay_ring_columnar_append_record_count",
    "owner_slot_ceiling_replay_ring_columnar_append_entry_view_object_count",
    "owner_slot_ceiling_replay_ring_columnar_append_step_view_object_count",
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    scenario: str = DEFAULT_SCENARIO
    batch_size: int = 32
    measured_steps: int = 32
    warmup_steps: int = 4
    body_capacity: int = 8
    random_tape_capacity_min: int = 16
    render_observation: bool = False
    run_search: bool = False
    run_slab_replay: bool = False
    run_owner_slot_ceiling: bool = False
    seed: int = 132
    event_mode: str = "no-event"
    use_direct_autoreset: bool = True
    require_pass: bool = True


@dataclass(frozen=True, slots=True)
class ActionTape:
    scenario_id: str
    player_count: int
    actions: tuple[np.ndarray, ...]
    step_ms: tuple[float, ...]
    timer_advance_ms: tuple[float, ...]


def _hash_update_array(hasher: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(str(tuple(int(dim) for dim in array.shape)).encode("utf-8"))
    hasher.update(array.tobytes())


def _array_checksum(value: np.ndarray) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    _hash_update_array(hasher, "array", np.asarray(value))
    return hasher.hexdigest()


def _array_checksum_any(value: Any) -> str:
    if hasattr(value, "detach"):
        return _array_checksum(value.detach().cpu().numpy())
    return _array_checksum(np.asarray(value))


def _leading_dim(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.asarray(value).shape
    if not shape:
        return 0
    return int(shape[0])


def _hash_update_text(hasher: Any, name: str, value: str) -> None:
    hasher.update(name.encode("utf-8"))
    payload = str(value).encode("utf-8")
    hasher.update(str(len(payload)).encode("utf-8"))
    hasher.update(payload)


def _state_field_names(state: dict[str, np.ndarray]) -> tuple[str, ...]:
    return tuple(sorted(str(name) for name in state))


def _state_checksum(state: dict[str, np.ndarray], fields: tuple[str, ...]) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    missing = [name for name in fields if name not in state]
    if missing:
        raise KeyError(f"state checksum missing fields: {missing}")
    for name in fields:
        _hash_update_array(hasher, name, np.asarray(state[name]))
    return hasher.hexdigest()


def _state_checksum_all(state: dict[str, np.ndarray]) -> str:
    return _state_checksum(state, _state_field_names(state))


def _hash_update_state_fields(
    hasher: Any,
    prefix: str,
    state: dict[str, np.ndarray],
    fields: tuple[str, ...],
) -> None:
    hasher.update(prefix.encode("utf-8"))
    for name in fields:
        _hash_update_array(hasher, name, np.asarray(state[name]))


def _action_tape_checksum(tape: ActionTape, config: BenchmarkConfig) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    _hash_update_text(hasher, "scenario_id", tape.scenario_id)
    _hash_update_array(
        hasher,
        "player_count",
        np.asarray([int(tape.player_count)], dtype=np.int64),
    )
    _hash_update_array(
        hasher,
        "step_ms",
        np.asarray(tape.step_ms, dtype=np.float64),
    )
    _hash_update_array(
        hasher,
        "timer_advance_ms",
        np.asarray(tape.timer_advance_ms, dtype=np.float64),
    )
    for index, action in enumerate(tape.actions):
        _hash_update_array(hasher, f"action_{index}", np.asarray(action, dtype=np.int16))
    measured_tape_indices = np.asarray(
        [
            int(step_index % len(tape.actions))
            for step_index in range(
                int(config.warmup_steps),
                int(config.warmup_steps) + int(config.measured_steps),
            )
        ],
        dtype=np.int32,
    )
    _hash_update_array(hasher, "measured_tape_indices", measured_tape_indices)
    return hasher.hexdigest()


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "min": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        rank = (len(ordered) - 1) * q
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        blend = rank - lower
        return ordered[lower] * (1.0 - blend) + ordered[upper] * blend

    return {
        "count": len(values),
        "sum": float(sum(values)),
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p95": float(percentile(0.95)),
        "max": float(ordered[-1]),
    }


def _timer_sum(summary: Mapping[str, Any]) -> float:
    try:
        return float(summary.get("sum", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _speedup_if_removed(total_sec: float, removed_sec: float) -> float:
    total = max(0.0, float(total_sec))
    removed = min(max(0.0, float(removed_sec)), total)
    remaining = total - removed
    return float(total / remaining) if remaining > 0.0 else 0.0


def _add_timer(timers: dict[str, float], key: str, started: float) -> None:
    timers[key] = timers.get(key, 0.0) + (time.perf_counter() - started)


def _load_fixture_action_tape(config: BenchmarkConfig) -> tuple[dict[str, np.ndarray], ActionTape]:
    fixture = seed_bridge.seed_fixture(config.scenario, body_capacity=int(config.body_capacity))
    initial_state = vector_compare.array_state_from_seed(fixture)
    steps_payload = fixture.get("action_schedule")
    if not isinstance(steps_payload, list) or not steps_payload:
        raise ValueError("scenario fixture must contain at least one step")

    actions: list[np.ndarray] = []
    step_ms: list[float] = []
    timer_advance_ms: list[float] = []
    for step_index in range(len(steps_payload)):
        prepared = vector_compare.prepare_fixture_array_step(fixture, step_index=step_index)
        source_moves = np.asarray(prepared["source_moves"], dtype=np.int8)
        actions.append((source_moves.astype(np.int16) + 1).reshape(1, -1))
        step_ms.append(float(prepared["step_ms"]))
        timer_advance_ms.append(float(prepared.get("timer_advance_ms", 0.0)))

    return initial_state, ActionTape(
        scenario_id=str(fixture.get("scenario_id", Path(config.scenario).stem)),
        player_count=int(initial_state["pos"].shape[1]),
        actions=tuple(actions),
        step_ms=tuple(step_ms),
        timer_advance_ms=tuple(timer_advance_ms),
    )


def _repeat_state(state: dict[str, np.ndarray], batch_size: int) -> dict[str, np.ndarray]:
    return {
        name: np.repeat(np.asarray(value), repeats=int(batch_size), axis=0)
        for name, value in state.items()
    }


def _pad_random_tape_capacity(
    state: dict[str, np.ndarray],
    *,
    min_capacity: int,
) -> dict[str, np.ndarray]:
    capacity = int(np.asarray(state["random_tape_values"]).shape[1])
    target_capacity = max(capacity, int(min_capacity))
    padded = {name: np.asarray(value).copy() for name, value in state.items()}
    if target_capacity == capacity:
        return padded

    values = np.zeros(
        (int(padded["random_tape_values"].shape[0]), target_capacity),
        dtype=padded["random_tape_values"].dtype,
    )
    values[:, :capacity] = padded["random_tape_values"]
    padded["random_tape_values"] = values
    return padded


def _make_env(
    *,
    config: BenchmarkConfig,
    initial_state: dict[str, np.ndarray],
    tape: ActionTape,
) -> VectorMultiplayerEnv:
    return VectorMultiplayerEnv(
        batch_size=int(config.batch_size),
        player_count=int(tape.player_count),
        decision_ms=max(1.0, float(tape.step_ms[0])),
        body_capacity=int(initial_state["body_active"].shape[1]),
        event_capacity=int(initial_state["event_type"].shape[1]),
        timer_capacity=int(initial_state["timer_active"].shape[1]),
        random_tape_capacity=int(initial_state["random_tape_values"].shape[1]),
        event_mode=str(config.event_mode),
    )


def _set_step_ms(env: VectorMultiplayerEnv, step_ms: float) -> None:
    env.decision_ms = float(step_ms)


def _repeat_action(action: np.ndarray, batch_size: int) -> np.ndarray:
    return np.repeat(np.asarray(action, dtype=np.int16), repeats=int(batch_size), axis=0)


class _ObservationStackTracker:
    def __init__(self, *, batch_size: int, player_count: int) -> None:
        self.batch_size = int(batch_size)
        self.player_count = int(player_count)
        self.stack = np.zeros(
            (self.batch_size, self.player_count, *POLICY_STACK_SHAPE),
            dtype=np.uint8,
        )
        self.render_out = np.zeros(
            (self.batch_size * self.player_count, 1, 64, 64),
            dtype=np.uint8,
        )
        self.row_indices = np.repeat(
            np.arange(self.batch_size, dtype=np.int64),
            self.player_count,
        )
        self.controlled_players = np.tile(
            np.arange(self.player_count, dtype=np.int64),
            self.batch_size,
        )
        self.renderer = CpuOracleBatchedObservationRenderer()
        self.schema = source_state_canvas_gray64_schema()
        self.zero_checksum = _array_checksum(self.stack)
        self.render_call_count = 0
        self.render_row_count = 0
        self.telemetry_sec: dict[str, float] = {}
        self.latest_frame_shape = [int(value) for value in self.render_out.shape]
        self.observation_shape = [int(value) for value in self.stack.shape]
        self.root_observation_shape = [
            int(self.batch_size * self.player_count),
            *[int(value) for value in POLICY_STACK_SHAPE],
        ]

    def update(self, state: dict[str, np.ndarray]) -> np.ndarray:
        self.stack[:, :, :-1] = self.stack[:, :, 1:]
        result = self.renderer.render(
            SourceStateBatchedRenderRequest(
                state=state,
                row_indices=self.row_indices,
                controlled_players=self.controlled_players,
                out=self.render_out,
            )
        )
        frames = np.asarray(result.frames)
        if frames.shape != self.render_out.shape:
            raise RuntimeError(
                "renderer returned unexpected frame shape: "
                f"{frames.shape} != {self.render_out.shape}"
            )
        if frames.dtype != np.uint8:
            raise RuntimeError(f"renderer returned unexpected dtype: {frames.dtype}")
        latest = frames.reshape(self.batch_size, self.player_count, 1, 64, 64)[:, :, 0]
        self.stack[:, :, -1] = latest
        self.render_call_count += 1
        self.render_row_count += int(frames.shape[0])
        for key, value in result.telemetry.items():
            self.telemetry_sec[str(key)] = self.telemetry_sec.get(str(key), 0.0) + float(value)
        return self.stack

    def root_observation(self) -> np.ndarray:
        return self.stack.reshape(self.batch_size * self.player_count, *POLICY_STACK_SHAPE)

    def metadata(self) -> dict[str, Any]:
        nonzero_count = int(np.count_nonzero(self.stack))
        return {
            "observation_schema_id": str(self.schema["schema_id"]),
            "observation_schema_hash": SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH,
            "observation_shape": self.observation_shape,
            "observation_dtype": str(self.stack.dtype),
            "latest_frame_shape": self.latest_frame_shape,
            "root_observation_shape": self.root_observation_shape,
            "render_row_count": int(self.render_row_count),
            "render_call_count": int(self.render_call_count),
            "observation_zero_checksum": self.zero_checksum,
            "observation_nonzero_count": nonzero_count,
            "observation_nonzero_checksum_present": bool(nonzero_count > 0),
            "resident_device_observation_shape": self.observation_shape,
            "resident_root_device_observation_shape": self.root_observation_shape,
            "resident_row_major_order": True,
            "resident_host_fallback_allowed": False,
            "renderer_backend": self.renderer.backend_name,
            "renderer_telemetry_sec": dict(sorted(self.telemetry_sec.items())),
        }


class _SearchRootTracker:
    def __init__(self, *, batch_size: int, player_count: int, num_simulations: int = 1) -> None:
        self.batch_size = int(batch_size)
        self.player_count = int(player_count)
        self.root_count = self.batch_size * self.player_count
        self.service = FixedShapeBatchedSearchOwnerV0(
            root_count=self.root_count,
            num_simulations=int(num_simulations),
        )
        self.root_batch_checksum = hashlib.blake2b(digest_size=16)
        self.action_step_checksum = hashlib.blake2b(digest_size=16)
        self.root_observation_checksum = hashlib.blake2b(digest_size=16)
        self.selected_action_checksum = hashlib.blake2b(digest_size=16)
        self.joint_action_checksum = hashlib.blake2b(digest_size=16)
        self.call_count = 0
        self.total_root_count = 0
        self.total_active_root_count = 0
        self.total_selected_action_count = 0
        self.total_inactive_root_count = 0
        self.total_ctree_calls = 0
        self.total_tolist_calls = 0
        self.total_per_sim_d2h_bytes = 0
        self.total_root_observation_copy_bytes = 0
        self.total_action_d2h_bytes = 0
        self.total_deferred_replay_payload_d2h_bytes = 0
        self.max_active_root_count = 0
        self.search_action_count = 0
        self.search_first_legal_policy = True
        self.search_two_phase_action_only = True
        self.search_buffer_reused = False
        self.search_preallocated_buffer_bytes = 0
        self.action_step_identity_checked = True
        self.action_step_root_index_matches_active = True
        self.action_step_env_row_matches_root = True
        self.action_step_player_matches_root = True
        self.action_step_policy_env_id_matches_root = True
        self.selected_action_shape_matches = True
        self.selected_action_legal = True
        self.replay_payload_digest_deferred = True
        self.replay_payload_digest_matches_handle = True
        self.selected_action_digest_matches_payload = True
        self.root_batch_observation_source = ""
        self.root_batch_observation_copied = False
        self.root_batch_observation_shape: list[int] = []
        self.root_batch_observation_dtype = ""
        self.root_batch_row_major_sidecars_checked = True
        self.done_root_matches_repeat_done = True
        self.active_root_mask_matches_non_done_legal = True
        self.to_play_all_default = True
        self.target_reward_matches_reward = True
        self.root_observation_shares_stack = True
        self.first_selected_action_digest = ""
        self.last_replay_payload_digest = ""
        self.last_profile_telemetry: dict[str, Any] = {}

    def run_step(
        self,
        *,
        observation_tracker: _ObservationStackTracker,
        env: VectorMultiplayerEnv,
        result: Any,
        joint_action: np.ndarray,
        step_index: int,
        tape_index: int,
        mode: str,
    ) -> None:
        root_batch = _build_search_root_batch(
            observation_tracker=observation_tracker,
            env=env,
            result=result,
            joint_action=joint_action,
            step_index=int(step_index),
            tape_index=int(tape_index),
            mode=str(mode),
        )
        action_step = self.service.run_action_step(root_batch)
        action_identity = _validate_search_action_step_identity(root_batch, action_step)
        digest_identity = _validate_search_action_step_digests(action_step)
        root_identity = _validate_search_root_batch_identity(
            root_batch,
            observation_tracker=observation_tracker,
            reward=np.asarray(_result_array(result, "reward"), dtype=np.float32),
            batch_size=self.batch_size,
            player_count=self.player_count,
        )
        search_joint_action = selected_joint_action_from_action_step(
            root_batch,
            action_step,
            batch_size=self.batch_size,
            player_count=self.player_count,
        )
        self.call_count += 1
        active_count = int(root_batch.metadata["active_root_count"])
        self.total_root_count += int(root_batch.metadata["root_count"])
        self.total_active_root_count += active_count
        self.total_inactive_root_count += int(root_batch.metadata["root_count"]) - active_count
        self.total_selected_action_count += int(action_step.selected_action.size)
        self.max_active_root_count = max(self.max_active_root_count, active_count)
        telemetry = action_step.metadata.get("profile_telemetry", {})
        if not isinstance(telemetry, dict):
            raise RuntimeError("search action step missing profile_telemetry")
        self.last_profile_telemetry = dict(telemetry)
        self.total_ctree_calls += int(
            telemetry["fixed_shape_batched_search_owner_ctree_calls"]
        )
        self.total_tolist_calls += int(
            telemetry["fixed_shape_batched_search_owner_tolist_calls"]
        )
        self.total_per_sim_d2h_bytes += int(
            telemetry["fixed_shape_batched_search_owner_per_sim_d2h_bytes"]
        )
        self.total_root_observation_copy_bytes += int(
            telemetry["fixed_shape_batched_search_owner_root_observation_copy_bytes"]
        )
        self.total_action_d2h_bytes += int(
            telemetry["fixed_shape_batched_search_owner_action_d2h_bytes"]
        )
        self.total_deferred_replay_payload_d2h_bytes += int(
            telemetry["fixed_shape_batched_search_owner_deferred_replay_payload_d2h_bytes"]
        )
        self.search_action_count = int(
            telemetry["fixed_shape_batched_search_owner_action_count"]
        )
        self.search_first_legal_policy = bool(
            self.search_first_legal_policy
            and telemetry["fixed_shape_batched_search_owner_first_legal_policy"]
        )
        self.search_two_phase_action_only = bool(
            self.search_two_phase_action_only
            and telemetry["fixed_shape_batched_search_owner_two_phase_action_only"]
        )
        self.search_buffer_reused = bool(
            telemetry["fixed_shape_batched_search_owner_buffer_reused"]
        )
        self.search_preallocated_buffer_bytes = int(
            telemetry["fixed_shape_batched_search_owner_preallocated_buffer_bytes"]
        )
        for key, value in action_identity.items():
            setattr(self, key, bool(getattr(self, key) and value))
        for key, value in digest_identity.items():
            setattr(self, key, bool(getattr(self, key) and value))
        for key, value in root_identity.items():
            if key in {
                "root_batch_observation_source",
                "root_batch_observation_shape",
                "root_batch_observation_dtype",
            }:
                setattr(self, key, value)
            else:
                setattr(self, key, bool(getattr(self, key) and value))
        selected_action_digest = str(action_step.metadata.get("selected_action_digest", ""))
        if not self.first_selected_action_digest:
            self.first_selected_action_digest = selected_action_digest
        self.last_replay_payload_digest = str(
            action_step.metadata.get("search_replay_payload_digest", "")
        )
        _accumulate_root_batch_checksum(self.root_batch_checksum, root_batch)
        _accumulate_action_step_checksum(self.action_step_checksum, action_step)
        _hash_update_array(
            self.root_observation_checksum,
            "root_observation",
            root_batch.observation,
        )
        _hash_update_array(
            self.selected_action_checksum,
            "selected_action",
            action_step.selected_action,
        )
        _hash_update_array(
            self.joint_action_checksum,
            "search_joint_action",
            search_joint_action,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "search_enabled": True,
            "search_impl": FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL,
            "search_schema_id": "curvyzero_compact_search_action_step/v1",
            "root_batch_schema_id": "curvyzero_compact_root_batch/v1",
            "search_call_count": int(self.call_count),
            "search_root_count": int(self.total_root_count),
            "search_active_root_count": int(self.total_active_root_count),
            "search_inactive_root_count": int(self.total_inactive_root_count),
            "search_selected_action_count": int(self.total_selected_action_count),
            "search_max_active_root_count": int(self.max_active_root_count),
            "search_action_count": int(self.search_action_count),
            "search_num_simulations": int(self.service.num_simulations),
            "search_first_legal_policy": bool(self.search_first_legal_policy),
            "search_two_phase_action_only": bool(self.search_two_phase_action_only),
            "search_ctree_calls": int(self.total_ctree_calls),
            "search_tolist_calls": int(self.total_tolist_calls),
            "search_per_sim_d2h_bytes": int(self.total_per_sim_d2h_bytes),
            "search_root_observation_copy_bytes": int(
                self.total_root_observation_copy_bytes
            ),
            "search_action_d2h_bytes": int(self.total_action_d2h_bytes),
            "search_deferred_replay_payload_d2h_bytes": int(
                self.total_deferred_replay_payload_d2h_bytes
            ),
            "search_preallocated_buffer_bytes": int(
                self.search_preallocated_buffer_bytes
            ),
            "search_buffer_reused": bool(self.search_buffer_reused),
            "search_action_step_identity_checked": bool(
                self.action_step_identity_checked
            ),
            "search_action_step_root_index_matches_active": bool(
                self.action_step_root_index_matches_active
            ),
            "search_action_step_env_row_matches_root": bool(
                self.action_step_env_row_matches_root
            ),
            "search_action_step_player_matches_root": bool(
                self.action_step_player_matches_root
            ),
            "search_action_step_policy_env_id_matches_root": bool(
                self.action_step_policy_env_id_matches_root
            ),
            "search_selected_action_shape_matches": bool(
                self.selected_action_shape_matches
            ),
            "search_selected_action_legal": bool(self.selected_action_legal),
            "search_replay_payload_digest_deferred": bool(
                self.replay_payload_digest_deferred
            ),
            "search_replay_payload_digest_matches_handle": bool(
                self.replay_payload_digest_matches_handle
            ),
            "search_selected_action_digest_matches_payload": bool(
                self.selected_action_digest_matches_payload
            ),
            "search_root_batch_observation_source": self.root_batch_observation_source,
            "search_root_batch_observation_copied": bool(
                self.root_batch_observation_copied
            ),
            "search_root_batch_observation_shape": list(
                self.root_batch_observation_shape
            ),
            "search_root_batch_observation_dtype": self.root_batch_observation_dtype,
            "search_root_batch_row_major_sidecars_checked": bool(
                self.root_batch_row_major_sidecars_checked
            ),
            "search_done_root_matches_repeat_done": bool(
                self.done_root_matches_repeat_done
            ),
            "search_active_root_mask_matches_non_done_legal": bool(
                self.active_root_mask_matches_non_done_legal
            ),
            "search_to_play_all_default": bool(self.to_play_all_default),
            "search_target_reward_matches_reward": bool(
                self.target_reward_matches_reward
            ),
            "search_root_observation_shares_stack": bool(
                self.root_observation_shares_stack
            ),
            "search_selected_action_digest": self.first_selected_action_digest,
            "search_replay_payload_digest": self.last_replay_payload_digest,
            "search_root_batch_checksum": self.root_batch_checksum.hexdigest(),
            "search_action_step_checksum": self.action_step_checksum.hexdigest(),
            "search_root_observation_checksum": self.root_observation_checksum.hexdigest(),
            "search_selected_action_checksum": self.selected_action_checksum.hexdigest(),
            "search_joint_action_checksum": self.joint_action_checksum.hexdigest(),
        }


class _SlabReplayTracker:
    def __init__(
        self,
        *,
        batch_size: int,
        player_count: int,
        sample_seed: int,
        num_simulations: int = 1,
    ) -> None:
        self.batch_size = int(batch_size)
        self.player_count = int(player_count)
        self.root_count = self.batch_size * self.player_count
        self.sample_seed = int(sample_seed)
        self.service = FixedShapeBatchedSearchOwnerV0(
            root_count=self.root_count,
            num_simulations=int(num_simulations),
        )
        self.slab = CompactRolloutSlab(
            batch_size=self.batch_size,
            player_count=self.player_count,
            search_service=self.service,
            search_lane="fixed_action_tape_closed_search_feedback_slab",
            policy_source="fixed_action_tape_closed_search_feedback_slab",
            copy_root_observation=False,
            retain_committed_index_rows=False,
        )
        self.replay_ring = _CompactReplayRingV1(
            capacity=max(2, int(self.batch_size) * 2),
            metadata={
                "fixed_action_tape_slab_replay_gate": True,
                "sample_source": "fixed_action_tape_host_replay_ring",
            },
        )
        self.next_joint_action: np.ndarray | None = None
        self.previous_replay_step: Any | None = None
        self.root_batch_checksum = hashlib.blake2b(digest_size=16)
        self.action_step_checksum = hashlib.blake2b(digest_size=16)
        self.index_rows_checksum = hashlib.blake2b(digest_size=16)
        self.feedback_action_checksum = hashlib.blake2b(digest_size=16)
        self.next_joint_action_checksum = hashlib.blake2b(digest_size=16)
        self.action_source_sequence_checksum = hashlib.blake2b(digest_size=16)
        self.sample_row_id_checksum = ""
        self.sample_action_checksum = ""
        self.sample_observation_checksum = ""
        self.sample_next_observation_checksum = ""
        self.sample_reward_checksum = ""
        self.sample_done_checksum = ""
        self.sample_record_index_checksum = ""
        self.sample_policy_row_checksum = ""
        self.sample_position_order_checksum = ""
        self.sampled_flat_row_checksum = ""
        self.source_record_pair_checksum = ""
        self.source_record_window_checksum = ""
        self.step_count = 0
        self.tape_bootstrap_action_count = 0
        self.feedback_action_count = 0
        self.measured_feedback_action_count = 0
        self.prev_next_joint_action_match_count = 0
        self.prev_next_joint_action_mismatch_count = 0
        self.feedback_differs_from_tape_count = 0
        self.root_count_total = 0
        self.active_root_count_total = 0
        self.inactive_root_count_total = 0
        self.selected_action_count_total = 0
        self.max_active_root_count = 0
        self.ctree_calls = 0
        self.tolist_calls = 0
        self.per_sim_d2h_bytes = 0
        self.action_d2h_bytes = 0
        self.deferred_replay_payload_d2h_bytes = 0
        self.root_observation_copy_bytes = 0
        self.committed_group_count = 0
        self.committed_row_count = 0
        self.committed_terminal_row_count = 0
        self.committed_next_final_observation_row_count = 0
        self.replay_payload_flush_count = 0
        self.replay_payload_d2h_bytes = 0
        self.replay_append_count = 0
        self.sample_gate_calls = 0
        self.sample_row_count = 0
        self.sample_target_row_count = 0
        self.last_sample_seed = int(sample_seed)
        self.action_check_enforced = True
        self.root_observation_copied = False
        self.index_rows_observation_materialized = False
        self.index_rows_next_observation_materialized = False

    def action_for_next_step(self) -> np.ndarray | None:
        if self.next_joint_action is None:
            return None
        return np.asarray(self.next_joint_action, dtype=np.int16).copy()

    def record_action_source(
        self,
        *,
        action_source: str,
        action: np.ndarray,
        tape_action: np.ndarray,
        measured: bool,
        step_index: int,
        tape_index: int,
    ) -> None:
        if action_source == "slab_search_feedback":
            self.feedback_action_count += 1
            if bool(measured):
                self.measured_feedback_action_count += 1
            expected_action = self.next_joint_action
            if expected_action is not None and np.array_equal(
                np.asarray(action, dtype=np.int16),
                np.asarray(expected_action, dtype=np.int16),
            ):
                self.prev_next_joint_action_match_count += 1
            else:
                self.prev_next_joint_action_mismatch_count += 1
            if not np.array_equal(
                np.asarray(action, dtype=np.int16),
                np.asarray(tape_action, dtype=np.int16),
            ):
                self.feedback_differs_from_tape_count += 1
        else:
            self.tape_bootstrap_action_count += 1
        _hash_update_text(
            self.action_source_sequence_checksum,
            "action_source",
            (
                f"{int(step_index)}:{int(tape_index)}:{str(action_source)}:"
                f"{int(bool(measured))}"
            ),
        )
        _hash_update_array(
            self.action_source_sequence_checksum,
            "action",
            np.asarray(action, dtype=np.int16),
        )
        _hash_update_array(
            self.feedback_action_checksum,
            str(action_source),
            np.asarray(action, dtype=np.int16),
        )

    def run_step(
        self,
        *,
        observation_tracker: _ObservationStackTracker,
        env: VectorMultiplayerEnv,
        result: Any,
        joint_action: np.ndarray,
        step_index: int,
        tape_index: int,
        mode: str,
    ) -> None:
        compact_batch = _build_hybrid_compact_batch(
            observation_tracker=observation_tracker,
            env=env,
            result=result,
            joint_action=joint_action,
            step_index=int(step_index),
            tape_index=int(tape_index),
            mode=str(mode),
        )
        replay_step = _replay_step_from_compact_batch(compact_batch)
        step = self.slab.step(compact_batch)
        if step.action_step is None:
            raise RuntimeError("slab replay proof requires two-phase action step")
        action_identity = _validate_search_action_step_identity(
            step.root_batch,
            step.action_step,
        )
        digest_identity = _validate_search_action_step_digests(step.action_step)
        root_identity = _validate_search_root_batch_identity(
            step.root_batch,
            observation_tracker=observation_tracker,
            reward=np.asarray(_result_array(result, "reward"), dtype=np.float32),
            batch_size=self.batch_size,
            player_count=self.player_count,
        )
        if not all(bool(value) for value in action_identity.values()):
            raise RuntimeError("slab replay action identity proof failed")
        if not all(bool(value) for value in digest_identity.values()):
            raise RuntimeError("slab replay digest identity proof failed")
        if not all(
            bool(value)
            for key, value in root_identity.items()
            if key
            not in {
                "root_batch_observation_source",
                "root_batch_observation_copied",
                "root_batch_observation_shape",
                "root_batch_observation_dtype",
            }
        ):
            failed = {
                key: value
                for key, value in root_identity.items()
                if isinstance(value, bool) and not value
            }
            raise RuntimeError(f"slab replay root identity proof failed: {failed}")
        self.next_joint_action = np.asarray(step.next_joint_action, dtype=np.int16).copy()
        self.step_count += 1
        root_count = int(step.root_batch.metadata["root_count"])
        active_count = int(step.root_batch.metadata["active_root_count"])
        self.root_count_total += root_count
        self.active_root_count_total += active_count
        self.inactive_root_count_total += root_count - active_count
        self.selected_action_count_total += int(step.action_step.selected_action.size)
        self.max_active_root_count = max(self.max_active_root_count, active_count)
        telemetry = dict(step.telemetry)
        profile_telemetry = dict(step.action_step.metadata.get("profile_telemetry", {}) or {})
        self.ctree_calls += int(
            profile_telemetry.get("fixed_shape_batched_search_owner_ctree_calls", 0)
        )
        self.tolist_calls += int(
            profile_telemetry.get("fixed_shape_batched_search_owner_tolist_calls", 0)
        )
        self.per_sim_d2h_bytes += int(
            profile_telemetry.get("fixed_shape_batched_search_owner_per_sim_d2h_bytes", 0)
        )
        self.action_d2h_bytes += int(
            profile_telemetry.get("fixed_shape_batched_search_owner_action_d2h_bytes", 0)
        )
        self.deferred_replay_payload_d2h_bytes += int(
            profile_telemetry.get(
                "fixed_shape_batched_search_owner_deferred_replay_payload_d2h_bytes",
                0,
            )
        )
        self.root_observation_copy_bytes += int(
            profile_telemetry.get(
                "fixed_shape_batched_search_owner_root_observation_copy_bytes",
                0,
            )
        )
        self.replay_payload_flush_count = int(
            telemetry.get("compact_rollout_slab_replay_payload_flush_count", 0)
        )
        self.replay_payload_d2h_bytes += int(
            telemetry.get("compact_rollout_slab_committed_replay_payload_d2h_bytes", 0)
        )
        self.action_check_enforced = bool(
            self.action_check_enforced
            and telemetry.get("compact_rollout_slab_replay_commit_requires_search_action", False)
        )
        self.root_observation_copied = bool(
            self.root_observation_copied
            or telemetry.get("compact_rollout_slab_observation_copied", False)
        )
        _accumulate_root_batch_checksum(self.root_batch_checksum, step.root_batch)
        _accumulate_action_step_checksum(self.action_step_checksum, step.action_step)
        _hash_update_array(
            self.next_joint_action_checksum,
            "next_joint_action",
            self.next_joint_action,
        )
        if step.committed_index_rows is not None:
            self.committed_group_count += 1
            row_count = int(np.asarray(step.committed_index_rows.action).size)
            self.committed_row_count += row_count
            metadata = dict(step.committed_index_rows.metadata)
            self.index_rows_observation_materialized = bool(
                self.index_rows_observation_materialized
                or metadata.get("observation_materialized", True)
            )
            self.index_rows_next_observation_materialized = bool(
                self.index_rows_next_observation_materialized
                or metadata.get("next_observation_materialized", True)
            )
            _accumulate_replay_index_rows_checksum(
                self.index_rows_checksum,
                step.committed_index_rows,
            )
            self.committed_terminal_row_count += int(
                np.asarray(step.committed_index_rows.done, dtype=np.bool_).sum()
            )
            self.committed_next_final_observation_row_count += int(
                np.asarray(
                    step.committed_index_rows.next_final_observation_row,
                    dtype=np.bool_,
                ).sum()
            )
            if row_count > 0:
                if self.previous_replay_step is None:
                    raise RuntimeError("slab replay commit missing previous replay step")
                self.replay_ring.append(
                    previous_step=self.previous_replay_step,
                    current_step=replay_step,
                    index_rows=step.committed_index_rows,
                )
                self.replay_append_count += 1
                self._sample_replay_ring()
        self.previous_replay_step = replay_step

    def metadata(self) -> dict[str, Any]:
        return {
            "slab_replay_enabled": True,
            "slab_search_feedback_closed_loop": bool(
                self.tape_bootstrap_action_count > 0
                and (
                    self.selected_action_count_total == 0
                    or self.feedback_action_count > 0
                )
            ),
            "slab_replay_tape_bootstrap_action_count": int(
                self.tape_bootstrap_action_count
            ),
            "slab_replay_feedback_action_count": int(self.feedback_action_count),
            "slab_replay_measured_feedback_action_count": int(
                self.measured_feedback_action_count
            ),
            "slab_replay_prev_next_joint_action_match_count": int(
                self.prev_next_joint_action_match_count
            ),
            "slab_replay_prev_next_joint_action_mismatch_count": int(
                self.prev_next_joint_action_mismatch_count
            ),
            "slab_replay_feedback_differs_from_tape_count": int(
                self.feedback_differs_from_tape_count
            ),
            "slab_replay_action_source_sequence_checksum": (
                self.action_source_sequence_checksum.hexdigest()
            ),
            "slab_step_count": int(self.step_count),
            "slab_root_count": int(self.root_count_total),
            "slab_active_root_count": int(self.active_root_count_total),
            "slab_inactive_root_count": int(self.inactive_root_count_total),
            "slab_selected_action_count": int(self.selected_action_count_total),
            "slab_max_active_root_count": int(self.max_active_root_count),
            "slab_action_count": ACTION_COUNT,
            "slab_num_simulations": int(self.service.num_simulations),
            "slab_ctree_calls": int(self.ctree_calls),
            "slab_tolist_calls": int(self.tolist_calls),
            "slab_per_sim_d2h_bytes": int(self.per_sim_d2h_bytes),
            "slab_action_d2h_bytes": int(self.action_d2h_bytes),
            "slab_deferred_replay_payload_d2h_bytes": int(
                self.deferred_replay_payload_d2h_bytes
            ),
            "slab_root_observation_copy_bytes": int(self.root_observation_copy_bytes),
            "slab_committed_index_group_count": int(self.committed_group_count),
            "slab_committed_index_row_count": int(self.committed_row_count),
            "slab_committed_terminal_row_count": int(self.committed_terminal_row_count),
            "slab_committed_next_final_observation_row_count": int(
                self.committed_next_final_observation_row_count
            ),
            "slab_replay_payload_flush_count": int(self.replay_payload_flush_count),
            "slab_replay_payload_d2h_bytes": int(self.replay_payload_d2h_bytes),
            "slab_replay_index_rows_observation_materialized": bool(
                self.index_rows_observation_materialized
            ),
            "slab_replay_index_rows_next_observation_materialized": bool(
                self.index_rows_next_observation_materialized
            ),
            "slab_replay_index_rows_checksum": self.index_rows_checksum.hexdigest(),
            "slab_replay_joint_action_feedback_checksum": (
                self.feedback_action_checksum.hexdigest()
            ),
            "slab_next_joint_action_checksum": self.next_joint_action_checksum.hexdigest(),
            "slab_replay_root_batch_checksum": self.root_batch_checksum.hexdigest(),
            "slab_replay_action_step_checksum": self.action_step_checksum.hexdigest(),
            "slab_replay_pending_uncommitted_count": int(
                max(0, self.step_count - self.committed_group_count)
            ),
            "slab_replay_action_check_enforced": bool(self.action_check_enforced),
            "slab_replay_root_observation_copied": bool(self.root_observation_copied),
            "slab_retains_committed_index_rows": bool(
                self.slab.retain_committed_index_rows
            ),
            "replay_append_count": int(self.replay_append_count),
            "replay_ring_entry_count": int(self.replay_ring.entry_count),
            "replay_ring_stored_index_row_count": int(
                self.replay_ring.stored_index_row_count
            ),
            "replay_ring_evicted_entry_count": int(self.replay_ring.evicted_entry_count),
            "replay_ring_evicted_index_row_count": int(
                self.replay_ring.evicted_index_row_count
            ),
            "sample_gate_calls": int(self.sample_gate_calls),
            "sample_row_count": int(self.sample_row_count),
            "sample_target_row_count": int(self.sample_target_row_count),
            "sample_seed": int(self.last_sample_seed),
            "sampled_flat_row_checksum": self.sampled_flat_row_checksum,
            "sample_position_order_checksum": self.sample_position_order_checksum,
            "source_record_pair_checksum": self.source_record_pair_checksum,
            "source_record_window_checksum": self.source_record_window_checksum,
            "sample_row_id_checksum": self.sample_row_id_checksum,
            "sample_action_checksum": self.sample_action_checksum,
            "sample_observation_checksum": self.sample_observation_checksum,
            "sample_next_observation_checksum": self.sample_next_observation_checksum,
            "sample_reward_checksum": self.sample_reward_checksum,
            "sample_done_checksum": self.sample_done_checksum,
            "slab_replay_sample_batch_built": bool(self.sample_gate_calls > 0),
            "slab_replay_sample_batch_size": int(self.sample_row_count),
            "slab_replay_sample_seed": int(self.last_sample_seed),
            "slab_replay_sample_row_id_checksum": self.sample_row_id_checksum,
            "slab_replay_sample_action_checksum": self.sample_action_checksum,
            "slab_replay_sample_observation_checksum": self.sample_observation_checksum,
            "slab_replay_sample_next_observation_checksum": (
                self.sample_next_observation_checksum
            ),
            "slab_replay_sample_record_index_checksum": self.sample_record_index_checksum,
            "slab_replay_sample_policy_row_checksum": self.sample_policy_row_checksum,
        }

    def _sample_replay_ring(self) -> None:
        sample_batch_size = min(8, int(self.replay_ring.stored_index_row_count))
        if sample_batch_size <= 0:
            return
        sample_seed = int(self.sample_seed) + int(self.sample_gate_calls)
        result = self.replay_ring.sample(
            seed=sample_seed,
            sample_batch_size=sample_batch_size,
            require_next_targets=False,
            num_unroll_steps=1,
            build_compact_muzero_learner_batch=False,
            compact_muzero_learner_batch_only=False,
        )
        sample = result.get("sample_batch")
        if sample is None:
            return
        metadata = dict(result.get("sample_metadata") or getattr(sample, "metadata", {}) or {})
        self.sample_gate_calls += 1
        self.sample_row_count = int(result.get("sample_row_count", sample_batch_size))
        self.sample_target_row_count = int(result.get("target_row_count", self.sample_row_count))
        self.last_sample_seed = sample_seed
        self.sampled_flat_row_checksum = str(metadata.get("sampled_flat_row_checksum", ""))
        self.sample_position_order_checksum = str(
            metadata.get("sample_position_order_checksum", "")
        )
        self.source_record_pair_checksum = str(metadata.get("source_record_pair_checksum", ""))
        self.source_record_window_checksum = str(
            metadata.get("source_record_window_checksum", "")
        )
        self.sample_row_id_checksum = _array_checksum(np.asarray(sample.row_id))
        self.sample_action_checksum = _array_checksum(np.asarray(sample.action))
        self.sample_observation_checksum = _array_checksum(np.asarray(sample.observation))
        self.sample_next_observation_checksum = _array_checksum(
            np.asarray(sample.next_observation)
        )
        self.sample_reward_checksum = _array_checksum(np.asarray(sample.reward))
        self.sample_done_checksum = _array_checksum(np.asarray(sample.done))
        self.sample_record_index_checksum = _array_checksum(np.asarray(sample.record_index))
        self.sample_policy_row_checksum = _array_checksum(np.asarray(sample.policy_row))


class _OwnerSlotCeilingSearchService:
    search_impl = "fixed_action_tape_owner_slot_ceiling_first_legal_v0"
    num_simulations = 1
    supports_two_phase_compact_search = True

    def __init__(self) -> None:
        self.run_count = 0
        self._action_payload_by_handle: dict[str, dict[str, Any]] = {}
        self._pending_staged_replay_records: list[dict[str, Any]] = []
        self._ready_replay_records: list[CompactReplayColumnarAppendRecordV1] = []
        self._replay_slot_capacity = 0
        self._replay_slot_max_rows = 0
        self._replay_slot_generation: np.ndarray | None = None
        self._replay_slot_row_count: np.ndarray | None = None
        self._replay_slot_row_id: np.ndarray | None = None
        self._replay_slot_env_row: np.ndarray | None = None
        self._replay_slot_player: np.ndarray | None = None
        self._replay_slot_action: np.ndarray | None = None
        self._replay_slot_reward: np.ndarray | None = None
        self._replay_slot_done: np.ndarray | None = None
        self._staged_transport_entry_count = 0
        self._staged_transition_entry_count = 0
        self._payload_cache_hit_count = 0
        self._payload_cache_miss_count = 0
        self._payload_release_count = 0
        self._replay_index_rows_build_count = 0
        self._replay_index_rows_row_count = 0
        self._device_replay_index_rows_build_count = 0
        self._device_replay_index_rows_row_count = 0
        self._replay_record_ready_count = 0
        self._replay_record_drain_count = 0
        self._replay_slot_append_count = 0
        self._replay_slot_append_row_count = 0
        self._sample_batch_built = False
        self._sample_gate_calls = 0
        self._sample_handle_create_count = 0
        self._sample_handle_resolve_count = 0
        self._sample_handle_inline_resolve_count = 0
        self._sample_handle_pending_count = 0
        self._sample_row_count = 0
        self._sample_target_row_count = 0
        self._replay_slot_window_checksum = hashlib.blake2b(digest_size=16)
        self._sample_handle_checksum = hashlib.blake2b(digest_size=16)
        self._sample_row_id_checksum = ""
        self._sample_action_checksum = ""
        self._sample_reward_checksum = ""
        self._sample_done_checksum = ""

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "owner_slot_ceiling_stage_replay_schema_id": (
                "curvyzero_owner_slot_stage_replay/v1"
            ),
            "owner_slot_ceiling_stage_replay_transport_entry_count": int(
                self._staged_transport_entry_count
            ),
            "owner_slot_ceiling_stage_replay_transition_entry_count": int(
                self._staged_transition_entry_count
            ),
            "owner_slot_ceiling_stage_replay_payload_cache_hit_count": int(
                self._payload_cache_hit_count
            ),
            "owner_slot_ceiling_stage_replay_payload_cache_miss_count": int(
                self._payload_cache_miss_count
            ),
            "owner_slot_ceiling_stage_replay_payload_release_count": int(
                self._payload_release_count
            ),
            "owner_slot_ceiling_stage_replay_payload_pending_count": int(
                sum(
                    1
                    for payload in self._action_payload_by_handle.values()
                    if int(np.asarray(payload["selected_action"]).size) > 0
                )
            ),
            "owner_slot_ceiling_stage_replay_pending_record_count": int(
                len(self._pending_staged_replay_records)
            ),
            "owner_slot_ceiling_stage_replay_ready_record_count": int(
                self._replay_record_ready_count
            ),
            "owner_slot_ceiling_stage_replay_drained_record_count": int(
                self._replay_record_drain_count
            ),
            "owner_slot_ceiling_stage_replay_index_rows_build_count": int(
                self._replay_index_rows_build_count
            ),
            "owner_slot_ceiling_stage_replay_index_rows_row_count": int(
                self._replay_index_rows_row_count
            ),
            "owner_slot_ceiling_stage_replay_device_index_rows_build_count": int(
                self._device_replay_index_rows_build_count
            ),
            "owner_slot_ceiling_stage_replay_device_index_rows_row_count": int(
                self._device_replay_index_rows_row_count
            ),
            "owner_slot_ceiling_stage_replay_slot_append_count": int(
                self._replay_slot_append_count
            ),
            "owner_slot_ceiling_stage_replay_slot_append_row_count": int(
                self._replay_slot_append_row_count
            ),
            "owner_slot_ceiling_stage_sample_batch_built": bool(
                self._sample_batch_built
            ),
            "owner_slot_ceiling_stage_sample_gate_calls": int(self._sample_gate_calls),
            "owner_slot_ceiling_stage_sample_handle_create_count": int(
                self._sample_handle_create_count
            ),
            "owner_slot_ceiling_stage_sample_handle_resolve_count": int(
                self._sample_handle_resolve_count
            ),
            "owner_slot_ceiling_stage_sample_handle_inline_resolve_count": int(
                self._sample_handle_inline_resolve_count
            ),
            "owner_slot_ceiling_stage_sample_handle_pending_count": int(
                self._sample_handle_pending_count
            ),
            "owner_slot_ceiling_stage_sample_row_count": int(self._sample_row_count),
            "owner_slot_ceiling_stage_sample_target_row_count": int(
                self._sample_target_row_count
            ),
            "owner_slot_ceiling_stage_replay_slot_window_checksum": (
                self._replay_slot_window_checksum.hexdigest()
            ),
            "owner_slot_ceiling_stage_sample_handle_checksum": (
                self._sample_handle_checksum.hexdigest()
            ),
            "owner_slot_ceiling_stage_sample_row_id_checksum": (
                self._sample_row_id_checksum
            ),
            "owner_slot_ceiling_stage_sample_action_checksum": (
                self._sample_action_checksum
            ),
            "owner_slot_ceiling_stage_sample_reward_checksum": (
                self._sample_reward_checksum
            ),
            "owner_slot_ceiling_stage_sample_done_checksum": self._sample_done_checksum,
        }

    def run_action_step(self, root_batch: Any) -> Any:
        del root_batch
        raise ReplayCompatibilityError(
            "owner-slot ceiling must use root-build-request action steps"
        )

    def flush_replay_payload(self, replay_payload_handle: str) -> Any:
        del replay_payload_handle
        raise ReplayCompatibilityError("owner-slot ceiling does not expose replay payloads")

    def stage_replay_append_entries(self, replay_append_entries: Any) -> int:
        if replay_append_entries is None:
            return 0
        if isinstance(replay_append_entries, (list, tuple)):
            entries = tuple(replay_append_entries)
        else:
            entries = (replay_append_entries,)
        staged = 0
        self._staged_transport_entry_count += int(len(entries))
        for entry in entries:
            staged += self._stage_replay_append_entry(entry)
        return int(staged)

    def _stage_replay_append_entry(self, entry: Any) -> int:
        if hasattr(entry, "transition_count") and hasattr(entry, "replay_payload_handles"):
            transition_count = int(getattr(entry, "transition_count"))
            handles = tuple(str(value) for value in getattr(entry, "replay_payload_handles"))
            selected_digests = tuple(
                str(value) for value in getattr(entry, "selected_action_digests")
            )
            search_digests = tuple(
                str(value) for value in getattr(entry, "search_replay_payload_digests")
            )
            next_joint_action = np.asarray(getattr(entry, "next_joint_action"), dtype=np.int16)
            next_reward = np.asarray(getattr(entry, "next_reward"), dtype=np.float32)
            next_done = np.asarray(getattr(entry, "next_done"), dtype=np.bool_)
            next_terminated = np.asarray(getattr(entry, "next_terminated"), dtype=np.bool_)
            next_truncated = np.asarray(getattr(entry, "next_truncated"), dtype=np.bool_)
            next_final_reward_map = np.asarray(
                getattr(entry, "next_final_reward_map"),
                dtype=np.float32,
            )
            next_final_observation_row_mask = np.asarray(
                getattr(entry, "next_final_observation_row_mask"),
                dtype=np.bool_,
            )
            for offset in range(transition_count):
                staged_entry = SimpleNamespace(
                    record_index=int(np.asarray(getattr(entry, "record_indices"))[offset]),
                    next_record_index=int(
                        np.asarray(getattr(entry, "next_record_indices"))[offset]
                    ),
                    replay_payload_handle=handles[offset],
                    selected_action_digest=selected_digests[offset],
                    search_replay_payload_digest=search_digests[offset],
                    next_joint_action=next_joint_action[offset],
                    next_reward=next_reward[offset],
                    next_done=next_done[offset],
                    next_terminated=next_terminated[offset],
                    next_truncated=next_truncated[offset],
                    next_final_reward_map=next_final_reward_map[offset],
                    next_final_observation_row_mask=next_final_observation_row_mask[
                        offset
                    ],
                )
                self._stage_one_transition(staged_entry)
            self._staged_transition_entry_count += int(transition_count)
            return int(transition_count)
        self._stage_one_transition(entry)
        self._staged_transition_entry_count += 1
        return 1

    def _stage_one_transition(self, entry: Any) -> None:
        handle = str(getattr(entry, "replay_payload_handle", ""))
        payload = self._action_payload_by_handle.pop(handle, None)
        if payload is None:
            self._payload_cache_miss_count += 1
            raise ReplayCompatibilityError("owner-slot staged replay payload handle miss")
        self._payload_cache_hit_count += 1
        self._payload_release_count += 1
        selected = np.asarray(payload["selected_action"], dtype=np.int16).reshape(-1)
        entry_metadata = dict(getattr(entry, "metadata", {}) or {})
        expected_selected_digest = str(payload["selected_action_digest"])
        entry_selected_digest = str(
            getattr(entry, "selected_action_digest", "")
            or entry_metadata.get("selected_action_digest", "")
        )
        if entry_selected_digest != expected_selected_digest:
            raise ReplayCompatibilityError("owner-slot staged replay selected digest mismatch")
        expected_search_digest = str(payload["search_replay_payload_digest"])
        entry_search_digest = str(
            getattr(entry, "search_replay_payload_digest", "")
            or entry_metadata.get("search_replay_payload_digest", "")
        )
        if entry_search_digest != expected_search_digest:
            raise ReplayCompatibilityError("owner-slot staged replay payload digest mismatch")
        if selected.size <= 0:
            return
        env_row = np.asarray(payload["env_row"], dtype=np.int64).reshape(-1)
        player = np.asarray(payload["player"], dtype=np.int64).reshape(-1)
        if selected.size != env_row.size or selected.size != player.size:
            raise ReplayCompatibilityError("owner-slot staged replay action/root shape mismatch")
        next_reward = np.asarray(getattr(entry, "next_reward"), dtype=np.float32)
        next_done = np.asarray(getattr(entry, "next_done"), dtype=np.bool_)
        next_joint_action = np.asarray(getattr(entry, "next_joint_action"), dtype=np.int16)
        next_terminated = np.asarray(
            getattr(entry, "next_terminated", next_done),
            dtype=np.bool_,
        )
        next_truncated = np.asarray(
            getattr(entry, "next_truncated", np.zeros_like(next_done, dtype=np.bool_)),
            dtype=np.bool_,
        )
        next_final_reward_map = np.asarray(
            getattr(entry, "next_final_reward_map", next_reward),
            dtype=np.float32,
        )
        next_final_observation_row_mask = np.asarray(
            getattr(
                entry,
                "next_final_observation_row_mask",
                np.zeros_like(next_done, dtype=np.bool_),
            ),
            dtype=np.bool_,
        )
        record_index = int(
            getattr(entry, "record_index", int(payload.get("record_index", 0)))
        )
        next_record_index = int(getattr(entry, "next_record_index", record_index + 1))
        self._pending_staged_replay_records.append(
            {
                "payload": payload,
                "record_index": int(record_index),
                "next_record_index": int(next_record_index),
                "next_joint_action": next_joint_action.copy(),
                "next_reward": next_reward.copy(),
                "next_done": next_done.copy(),
                "next_terminated": next_terminated.copy(),
                "next_truncated": next_truncated.copy(),
                "next_final_reward_map": next_final_reward_map.copy(),
                "next_final_observation_row_mask": (
                    next_final_observation_row_mask.copy()
                ),
            }
        )
        self._append_staged_replay_slot(
            env_row=env_row,
            player=player,
            selected=selected,
            next_reward=next_reward,
            next_done=next_done,
        )

    def _ensure_replay_buffers(self, *, batch_size: int, player_count: int) -> None:
        capacity = max(2, int(batch_size) * 2)
        max_rows = max(1, int(batch_size) * int(player_count))
        if self._replay_slot_generation is not None:
            if self._replay_slot_capacity != capacity or self._replay_slot_max_rows != max_rows:
                raise ReplayCompatibilityError("owner-slot staged replay buffer shape changed")
            return
        self._replay_slot_capacity = int(capacity)
        self._replay_slot_max_rows = int(max_rows)
        replay_shape = (self._replay_slot_capacity, self._replay_slot_max_rows)
        self._replay_slot_generation = np.zeros(self._replay_slot_capacity, dtype=np.int64)
        self._replay_slot_row_count = np.zeros(self._replay_slot_capacity, dtype=np.int32)
        self._replay_slot_row_id = np.zeros(replay_shape, dtype=np.int64)
        self._replay_slot_env_row = np.zeros(replay_shape, dtype=np.int32)
        self._replay_slot_player = np.zeros(replay_shape, dtype=np.int16)
        self._replay_slot_action = np.zeros(replay_shape, dtype=np.int16)
        self._replay_slot_reward = np.zeros(replay_shape, dtype=np.float32)
        self._replay_slot_done = np.zeros(replay_shape, dtype=np.bool_)

    def _append_staged_replay_slot(
        self,
        *,
        env_row: np.ndarray,
        player: np.ndarray,
        selected: np.ndarray,
        next_reward: np.ndarray,
        next_done: np.ndarray,
    ) -> None:
        if self._replay_slot_generation is None:
            raise ReplayCompatibilityError("owner-slot staged replay buffers are missing")
        row_count_array = self._replay_slot_row_count
        row_id_array = self._replay_slot_row_id
        env_row_array = self._replay_slot_env_row
        player_array = self._replay_slot_player
        action_array = self._replay_slot_action
        reward_array = self._replay_slot_reward
        done_array = self._replay_slot_done
        if (
            row_count_array is None
            or row_id_array is None
            or env_row_array is None
            or player_array is None
            or action_array is None
            or reward_array is None
            or done_array is None
        ):
            raise ReplayCompatibilityError("owner-slot staged replay buffer arrays missing")
        slot_id = int(self._replay_slot_append_count % self._replay_slot_capacity)
        generation = int(self._replay_slot_append_count + 1)
        row_count = int(selected.size)
        if row_count > self._replay_slot_max_rows:
            raise ReplayCompatibilityError("owner-slot staged replay slot capacity exceeded")
        row_id = (
            np.arange(row_count, dtype=np.int64)
            + np.int64(generation * self._replay_slot_max_rows)
        )
        row_count_array[slot_id] = np.int32(row_count)
        self._replay_slot_generation[slot_id] = np.int64(generation)
        tail = slice(row_count, self._replay_slot_max_rows)
        row_id_array[slot_id, :row_count] = row_id
        env_row_array[slot_id, :row_count] = env_row.astype(np.int32)
        player_array[slot_id, :row_count] = player.astype(np.int16)
        action_array[slot_id, :row_count] = selected
        reward_array[slot_id, :row_count] = next_reward[env_row, player]
        done_array[slot_id, :row_count] = next_done[env_row]
        row_id_array[slot_id, tail] = 0
        env_row_array[slot_id, tail] = 0
        player_array[slot_id, tail] = 0
        action_array[slot_id, tail] = 0
        reward_array[slot_id, tail] = 0.0
        done_array[slot_id, tail] = False
        self._replay_slot_append_count += 1
        self._replay_slot_append_row_count += row_count
        _hash_update_text(
            self._replay_slot_window_checksum,
            "slot",
            f"{slot_id}:{generation}:{row_count}",
        )
        _hash_update_array(self._replay_slot_window_checksum, "row_id", row_id)
        _hash_update_array(self._replay_slot_window_checksum, "env_row", env_row)
        _hash_update_array(self._replay_slot_window_checksum, "player", player)
        _hash_update_array(self._replay_slot_window_checksum, "action", selected)
        _hash_update_array(
            self._replay_slot_window_checksum,
            "reward",
            reward_array[slot_id, :row_count],
        )
        _hash_update_array(
            self._replay_slot_window_checksum,
            "done",
            done_array[slot_id, :row_count],
        )
        self._sample_staged_replay_slot_handle()

    def _sample_staged_replay_slot_handle(self) -> None:
        row_count_array = self._replay_slot_row_count
        generation_array = self._replay_slot_generation
        row_id_array = self._replay_slot_row_id
        action_array = self._replay_slot_action
        reward_array = self._replay_slot_reward
        done_array = self._replay_slot_done
        if (
            row_count_array is None
            or generation_array is None
            or row_id_array is None
            or action_array is None
            or reward_array is None
            or done_array is None
        ):
            return
        total_rows = int(row_count_array.sum())
        if total_rows <= 0:
            return
        flat_slot = np.empty(total_rows, dtype=np.int32)
        flat_offset = np.empty(total_rows, dtype=np.int32)
        cursor = 0
        for slot_id, row_count_value in enumerate(row_count_array):
            row_count = int(row_count_value)
            if row_count <= 0 or int(generation_array[slot_id]) <= 0:
                continue
            stop = cursor + row_count
            flat_slot[cursor:stop] = int(slot_id)
            flat_offset[cursor:stop] = np.arange(row_count, dtype=np.int32)
            cursor = stop
        flat_slot = flat_slot[:cursor]
        flat_offset = flat_offset[:cursor]
        sample_size = min(8, int(cursor))
        if sample_size <= 0:
            return
        sample_seed = 29_000 + int(self._sample_handle_create_count)
        rng = np.random.default_rng(sample_seed)
        chosen = np.asarray(
            rng.choice(int(cursor), size=sample_size, replace=False),
            dtype=np.int64,
        )
        sample_slot = flat_slot[chosen]
        sample_offset = flat_offset[chosen]
        sample_row_id = row_id_array[sample_slot, sample_offset]
        sample_action = action_array[sample_slot, sample_offset]
        sample_reward = reward_array[sample_slot, sample_offset]
        sample_done = done_array[sample_slot, sample_offset]
        self._sample_handle_create_count += 1
        self._sample_handle_resolve_count += 1
        self._sample_handle_inline_resolve_count += 1
        self._sample_handle_pending_count = 0
        self._sample_gate_calls += 1
        self._sample_batch_built = True
        self._sample_row_count = int(sample_size)
        self._sample_target_row_count = int(sample_size)
        _hash_update_text(
            self._sample_handle_checksum,
            "handle",
            (
                f"{self._sample_handle_create_count}:{sample_seed}:"
                f"{int(sample_size)}:{int(cursor)}"
            ),
        )
        _hash_update_array(self._sample_handle_checksum, "slot", sample_slot)
        _hash_update_array(self._sample_handle_checksum, "offset", sample_offset)
        _hash_update_array(self._sample_handle_checksum, "row_id", sample_row_id)
        self._sample_row_id_checksum = _array_checksum(sample_row_id)
        self._sample_action_checksum = _array_checksum(sample_action)
        self._sample_reward_checksum = _array_checksum(sample_reward)
        self._sample_done_checksum = _array_checksum(sample_done)

    def _materialize_pending_replay_records(
        self,
        *,
        current_resident_observation_replay_snapshot: Any,
    ) -> None:
        if not self._pending_staged_replay_records:
            return
        pending = tuple(self._pending_staged_replay_records)
        self._pending_staged_replay_records.clear()
        for item in pending:
            payload = dict(item["payload"])
            index_rows = build_compact_device_replay_index_rows_v1_from_owner_action_context_payload(
                payload["root_action_context"],
                payload["action_step"],
                payload["replay_payload"],
                record_index=int(item["record_index"]),
                next_joint_action=np.asarray(item["next_joint_action"], dtype=np.int16),
                next_reward=np.asarray(item["next_reward"], dtype=np.float32),
                next_done=np.asarray(item["next_done"], dtype=np.bool_),
                next_terminated=np.asarray(item["next_terminated"], dtype=np.bool_),
                next_truncated=np.asarray(item["next_truncated"], dtype=np.bool_),
                next_final_reward_map=np.asarray(
                    item["next_final_reward_map"],
                    dtype=np.float32,
                ),
                next_final_observation_row_mask=np.asarray(
                    item["next_final_observation_row_mask"],
                    dtype=np.bool_,
                ),
                policy_source="fixed_action_tape_owner_slot_ceiling",
                metadata={
                    "owner_slot_ceiling_stage_replay_ring_record": True,
                    "owner_slot_ceiling_stage_replay_next_record_index": int(
                        item["next_record_index"]
                    ),
                },
            )
            row_count = _leading_dim(index_rows.action)
            self._replay_index_rows_build_count += 1
            self._replay_index_rows_row_count += row_count
            if bool(index_rows.metadata.get("device_replay_index_rows", False)):
                self._device_replay_index_rows_build_count += 1
                self._device_replay_index_rows_row_count += row_count
            self._ready_replay_records.append(
                CompactReplayColumnarAppendRecordV1(
                    previous_resident_observation_replay_snapshot=payload[
                        "resident_snapshot"
                    ],
                    current_resident_observation_replay_snapshot=(
                        current_resident_observation_replay_snapshot
                    ),
                    index_rows=index_rows,
                    previous_action_mask=np.asarray(
                        payload["action_mask"],
                        dtype=np.bool_,
                    ).copy(),
                    current_joint_action=np.asarray(
                        item["next_joint_action"],
                        dtype=np.int16,
                    ).copy(),
                    current_reward=np.asarray(
                        item["next_reward"],
                        dtype=np.float32,
                    ).copy(),
                    current_final_reward_map=np.asarray(
                        item["next_final_reward_map"],
                        dtype=np.float32,
                    ).copy(),
                    current_done=np.asarray(item["next_done"], dtype=np.bool_).copy(),
                    current_terminated=np.asarray(
                        item["next_terminated"],
                        dtype=np.bool_,
                    ).copy(),
                    current_truncated=np.asarray(
                        item["next_truncated"],
                        dtype=np.bool_,
                    ).copy(),
                )
            )
            self._replay_record_ready_count += 1

    def drain_staged_replay_records(
        self,
        *,
        current_resident_observation_replay_snapshot: Any,
    ) -> tuple[CompactReplayColumnarAppendRecordV1, ...]:
        self._materialize_pending_replay_records(
            current_resident_observation_replay_snapshot=(
                current_resident_observation_replay_snapshot
            ),
        )
        if not self._ready_replay_records:
            return ()
        records = tuple(self._ready_replay_records)
        self._ready_replay_records.clear()
        self._replay_record_drain_count += int(len(records))
        return records

    def run_action_step_from_root_build_request(self, root_build_request: Any) -> Any:
        import torch

        self.run_count += 1
        current_resident_snapshot = getattr(root_build_request, "resident_observation", None)
        if current_resident_snapshot is None:
            raise ReplayCompatibilityError("owner-slot root request missing resident snapshot")
        self._materialize_pending_replay_records(
            current_resident_observation_replay_snapshot=current_resident_snapshot
        )
        root_count = int(root_build_request.root_count)
        batch_size = int(root_build_request.batch_size)
        player_count = int(root_build_request.player_count)
        active_root_mask = np.asarray(
            root_build_request.active_root_mask,
            dtype=np.bool_,
        ).reshape(root_count)
        root_index = np.flatnonzero(active_root_mask).astype(np.int64, copy=False)
        legal_mask = np.asarray(root_build_request.action_mask, dtype=np.bool_).reshape(
            root_count,
            ACTION_COUNT,
        )
        selected = np.zeros((root_index.size,), dtype=np.int16)
        if root_index.size:
            selected = legal_mask[root_index].argmax(axis=1).astype(
                np.int16,
                copy=False,
            )
        env_row_all = np.asarray(root_build_request.policy_env_row, dtype=np.int32).reshape(-1)
        player_all = np.asarray(root_build_request.policy_player, dtype=np.int16).reshape(-1)
        policy_env_id_all = np.asarray(root_build_request.policy_env_id, dtype=np.int64).reshape(
            -1
        )
        env_row = env_row_all[root_index].astype(np.int32, copy=False)
        player = player_all[root_index].astype(np.int16, copy=False)
        policy_env_id = policy_env_id_all[root_index].astype(np.int64, copy=False)
        dense = np.zeros((batch_size, player_count), dtype=np.int16)
        if selected.size:
            dense[env_row.astype(np.int64), player.astype(np.int64)] = selected
        handle_text = f"{self.search_impl}:{self.run_count}"
        action_d2h_bytes = int(selected.nbytes)
        replay_payload_bytes = int(
            selected.size * (2 * ACTION_COUNT + 1) * np.dtype(np.float32).itemsize
        )
        selected_digest = compact_search_array_digest_v1(
            selected.astype(np.int16, copy=False)
        )
        search_replay_digest = compact_search_deferred_replay_payload_digest_v1(
            handle_text
        )
        self._ensure_replay_buffers(batch_size=batch_size, player_count=player_count)
        self._action_payload_by_handle[handle_text] = {
            "record_index": int(self.run_count - 1),
            "root_action_context": compact_root_action_context_v1_from_request(
                root_build_request
            ),
            "resident_snapshot": current_resident_snapshot,
            "action_mask": legal_mask.reshape(
                batch_size,
                player_count,
                ACTION_COUNT,
            ).astype(np.bool_, copy=True),
            "env_row": env_row.astype(np.int32, copy=True),
            "player": player.astype(np.int16, copy=True),
            "selected_action": selected.astype(np.int16, copy=True),
            "selected_action_digest": selected_digest,
            "search_replay_payload_digest": search_replay_digest,
        }
        self._action_payload_by_handle[handle_text]["action_step"] = CompactSearchActionStepV1(
            replay_payload_handle=handle_text,
            root_index=root_index.astype(np.int32, copy=True),
            env_row=env_row.astype(np.int32, copy=True),
            player=player.astype(np.int16, copy=True),
            policy_env_id=policy_env_id.astype(np.int64, copy=True),
            selected_action=selected.astype(np.int16, copy=True),
            dense_joint_action=dense.astype(np.int16, copy=True),
            metadata={
                "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                "phase": "action_critical",
                "search_impl": self.search_impl,
                "num_simulations": self.num_simulations,
                "active_root_count": int(selected.size),
                "selected_action_digest": selected_digest,
                "search_replay_payload_digest": search_replay_digest,
                "search_replay_payload_digest_deferred": True,
                "compact_owner_search_owner_materializes_replay": True,
            },
        )
        visit_policy = np.zeros((int(selected.size), ACTION_COUNT), dtype=np.float32)
        if selected.size:
            visit_policy[np.arange(int(selected.size)), selected.astype(np.int64)] = 1.0
        root_value = np.zeros((int(selected.size),), dtype=np.float32)
        self._action_payload_by_handle[handle_text]["replay_payload"] = (
            CompactDeviceSearchReplayPayloadV1(
                replay_payload_handle=handle_text,
                root_index=root_index.astype(np.int32, copy=True),
                env_row=env_row.astype(np.int32, copy=True),
                player=player.astype(np.int16, copy=True),
                policy_env_id=policy_env_id.astype(np.int64, copy=True),
                visit_policy=torch.as_tensor(visit_policy, dtype=torch.float32),
                root_value=torch.as_tensor(root_value, dtype=torch.float32),
                raw_visit_counts=None,
                predicted_value=None,
                predicted_policy_logits=None,
                metadata={
                    "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                    "phase": "replay_critical_device",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(selected.size),
                    "search_replay_payload_digest": search_replay_digest,
                    "device_replay_payload": True,
                    "device_replay_payload_device": "cpu",
                    "host_search_payload_fallback_allowed": False,
                },
            )
        )
        return CompactSearchActionStepV1(
            replay_payload_handle=handle_text,
            root_index=root_index,
            env_row=env_row,
            player=player,
            policy_env_id=policy_env_id,
            selected_action=selected,
            dense_joint_action=dense,
            metadata={
                "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                "phase": "action_critical",
                "search_impl": self.search_impl,
                "num_simulations": self.num_simulations,
                "active_root_count": int(selected.size),
                "selected_action_digest": selected_digest,
                "search_replay_payload_digest": search_replay_digest,
                "search_replay_payload_digest_deferred": True,
                "compact_owner_search_owner_materializes_replay": True,
                "compact_owner_search_action_only_result": True,
                "compact_owner_search_direct_root_build_request_handoff": True,
                "compact_owner_search_direct_root_parent_build_avoided": True,
                "compact_owner_search_dense_joint_action_used": True,
                "compact_owner_root_search_transaction_action_identity_verified": True,
                "profile_telemetry": {
                    "fixed_shape_batched_search_owner_ctree_calls": 0,
                    "fixed_shape_batched_search_owner_tolist_calls": 0,
                    "fixed_shape_batched_search_owner_per_sim_d2h_bytes": 0,
                    "fixed_shape_batched_search_owner_root_observation_copy_bytes": 0,
                    "fixed_shape_batched_search_owner_action_d2h_bytes": (
                        action_d2h_bytes
                    ),
                    "fixed_shape_batched_search_owner_deferred_replay_payload_d2h_bytes": (
                        replay_payload_bytes
                    ),
                },
            },
        )


class _OwnerSlotCeilingTracker:
    def __init__(
        self,
        *,
        batch_size: int,
        player_count: int,
    ) -> None:
        self.batch_size = int(batch_size)
        self.player_count = int(player_count)
        self.root_count = self.batch_size * self.player_count
        self.ring = _make_compact_owner_mechanics_step_frame_ring(
            batch_size=self.batch_size,
            player_count=self.player_count,
        )
        self.search_service = _OwnerSlotCeilingSearchService()
        self.stepper = CompactOwnerSearchDirectStepperV1(
            batch_size=self.batch_size,
            player_count=self.player_count,
            search_service=self.search_service,
            search_lane="fixed_action_tape_owner_slot_ceiling",
            policy_source="fixed_action_tape_owner_slot_ceiling",
            resident_root_host_observation_stub=True,
            direct_root_build_request=True,
        )
        self.next_joint_action: np.ndarray | None = None
        self.step_count = 0
        self.tape_bootstrap_action_count = 0
        self.feedback_action_count = 0
        self.measured_feedback_action_count = 0
        self.prev_next_joint_action_match_count = 0
        self.prev_next_joint_action_mismatch_count = 0
        self.mechanics_slot_write_count = 0
        self.mechanics_slot_generation_verified_count = 0
        self.mechanics_slot_digest_verified_count = 0
        self.root_request_from_slot_count = 0
        self.root_request_from_batch_count = 0
        self.hybrid_compact_batch_object_count = 0
        self.action_result_write_count = 0
        self.action_result_read_count = 0
        self.next_action_count = 0
        self.root_observation_copy_bytes = 0
        self.active_root_count = 0
        self.selected_action_count = 0
        self.ctree_calls = 0
        self.tolist_calls = 0
        self.replay_ring = _CompactReplayRingV1(
            capacity=max(2, int(self.batch_size) * 2),
            metadata={
                "fixed_action_tape_owner_slot_replay_ring_gate": True,
                "sample_source": "fixed_action_tape_owner_slot_replay_ring",
            },
        )
        self.replay_ring_append_record_count = 0
        self.replay_ring_append_call_count = 0
        self.replay_ring_appended_row_count = 0
        self.replay_ring_sample_gate_calls = 0
        self.replay_ring_sample_batch_built = False
        self.replay_ring_sample_row_count = 0
        self.replay_ring_sample_target_row_count = 0
        self.replay_ring_sample_observation_provider_used_count = 0
        self.replay_ring_sample_row_id_checksum = ""
        self.replay_ring_sample_action_checksum = ""
        self.replay_ring_sample_reward_checksum = ""
        self.replay_ring_sample_done_checksum = ""
        self.replay_ring_sample_observation_checksum = ""
        self.replay_ring_sample_next_observation_checksum = ""
        self.replay_ring_sample_source = "none"
        self.replay_ring_sample_device_replay_index_rows_sample = False
        self.replay_ring_sample_device_replay_index_rows_sample_all = False
        self.replay_ring_sample_resident_device_sample_batch = False
        self.replay_ring_sample_host_observation_fallback_allowed = False
        self.replay_ring_learner_unroll2_sample_gate_calls = 0
        self.replay_ring_learner_unroll2_batch_built = False
        self.replay_ring_learner_unroll2_sample_row_count = 0
        self.replay_ring_learner_unroll2_target_row_count = 0
        self.replay_ring_learner_unroll2_num_unroll_steps = 0
        self.replay_ring_learner_unroll2_require_next_targets = False
        self.replay_ring_learner_unroll2_batch_only = False
        self.replay_ring_learner_unroll2_source = "none"
        self.replay_ring_learner_unroll2_candidate_universe_source = "none"
        self.replay_ring_learner_unroll2_explicit_unroll_target_group_count = 0
        self.replay_ring_learner_unroll2_next_target_eligible_pair_count = 0
        self.replay_ring_learner_unroll2_observation_provider_used_count = 0
        self.replay_ring_learner_unroll2_schema_id = "none"
        self.replay_ring_learner_unroll2_prevalidation_source = "none"
        self.replay_ring_learner_unroll2_host_fallback_allowed = False
        self.replay_ring_learner_unroll2_action_shape: list[int] = []
        self.replay_ring_learner_unroll2_target_reward_shape: list[int] = []
        self.replay_ring_learner_unroll2_target_value_shape: list[int] = []
        self.replay_ring_learner_unroll2_target_policy_shape: list[int] = []
        self.replay_ring_learner_unroll2_action_mask_shape: list[int] = []
        self.replay_ring_learner_unroll2_action_checksum = ""
        self.replay_ring_learner_unroll2_target_reward_checksum = ""
        self.replay_ring_learner_unroll2_target_value_checksum = ""
        self.replay_ring_learner_unroll2_target_policy_checksum = ""
        self.replay_ring_learner_unroll2_source_record_window_checksum = ""
        self.replay_slot_capacity = max(2, self.batch_size * 2)
        self.replay_slot_max_rows = max(1, self.root_count)
        replay_shape = (self.replay_slot_capacity, self.replay_slot_max_rows)
        self.replay_slot_generation = np.zeros(self.replay_slot_capacity, dtype=np.int64)
        self.replay_slot_row_count = np.zeros(self.replay_slot_capacity, dtype=np.int32)
        self.replay_slot_row_id = np.zeros(replay_shape, dtype=np.int64)
        self.replay_slot_env_row = np.zeros(replay_shape, dtype=np.int32)
        self.replay_slot_player = np.zeros(replay_shape, dtype=np.int16)
        self.replay_slot_action = np.zeros(replay_shape, dtype=np.int16)
        self.replay_slot_reward = np.zeros(replay_shape, dtype=np.float32)
        self.replay_slot_done = np.zeros(replay_shape, dtype=np.bool_)
        self.replay_slot_terminated = np.zeros(replay_shape, dtype=np.bool_)
        self.replay_slot_truncated = np.zeros(replay_shape, dtype=np.bool_)
        self.previous_action_step: CompactSearchActionStepV1 | None = None
        self.replay_slot_append_count = 0
        self.replay_slot_append_row_count = 0
        self.replay_slot_object_entry_count = 0
        self.parent_replay_object_count = 0
        self.selected_group_object_count = 0
        self.sample_gate_calls = 0
        self.sample_batch_built = False
        self.sample_handle_create_count = 0
        self.sample_handle_resolve_count = 0
        self.sample_handle_inline_resolve_count = 0
        self.sample_handle_pending_count = 0
        self.sample_row_count = 0
        self.sample_target_row_count = 0
        self.action_source_sequence_checksum = hashlib.blake2b(digest_size=16)
        self.slot_digest_checksum = hashlib.blake2b(digest_size=16)
        self.next_joint_action_checksum = hashlib.blake2b(digest_size=16)
        self.replay_slot_window_checksum = hashlib.blake2b(digest_size=16)
        self.sample_handle_checksum = hashlib.blake2b(digest_size=16)
        self.sample_row_id_checksum = ""
        self.sample_action_checksum = ""
        self.sample_reward_checksum = ""
        self.sample_done_checksum = ""
        self.last_slot_id = -1
        self.last_generation = -1

    def action_for_next_step(self) -> np.ndarray | None:
        if self.next_joint_action is None:
            return None
        return np.asarray(self.next_joint_action, dtype=np.int16).copy()

    def record_action_source(
        self,
        *,
        action_source: str,
        action: np.ndarray,
        tape_action: np.ndarray,
        measured: bool,
        step_index: int,
        tape_index: int,
    ) -> None:
        if action_source == "owner_slot_search_feedback":
            self.feedback_action_count += 1
            if bool(measured):
                self.measured_feedback_action_count += 1
            expected_action = self.next_joint_action
            if expected_action is not None and np.array_equal(
                np.asarray(action, dtype=np.int16),
                np.asarray(expected_action, dtype=np.int16),
            ):
                self.prev_next_joint_action_match_count += 1
            else:
                self.prev_next_joint_action_mismatch_count += 1
        else:
            self.tape_bootstrap_action_count += 1
        _hash_update_text(
            self.action_source_sequence_checksum,
            "action_source",
            (
                f"{int(step_index)}:{int(tape_index)}:{str(action_source)}:"
                f"{int(bool(measured))}"
            ),
        )
        _hash_update_array(
            self.action_source_sequence_checksum,
            "action",
            np.asarray(action, dtype=np.int16),
        )
        if tape_action.shape == action.shape and not np.array_equal(action, tape_action):
            _hash_update_text(
                self.action_source_sequence_checksum,
                "differs_from_tape",
                "true",
            )

    def run_step(
        self,
        *,
        observation_tracker: _ObservationStackTracker,
        env: VectorMultiplayerEnv,
        result: Any,
        joint_action: np.ndarray,
        step_index: int,
        tape_index: int,
        mode: str,
    ) -> None:
        del tape_index, mode
        import torch

        device_observation = torch.as_tensor(
            np.ascontiguousarray(observation_tracker.stack.copy())
        )
        root_device_observation = device_observation.reshape(
            self.root_count,
            *POLICY_STACK_SHAPE,
        )
        final_observation_row_mask = np.asarray(
            _result_array(result, "done"),
            dtype=np.bool_,
        ).copy()
        resident = SimpleNamespace(
            device_observation=device_observation,
            root_device_observation=root_device_observation,
            final_device_observation=device_observation,
            root_final_device_observation=root_device_observation,
            final_observation_row_mask=final_observation_row_mask,
            final_device_observation_rows=None,
            final_device_observation_row_indices=None,
            device_frame_history=(),
            generation_id=int(step_index) + 1,
            batch_size=self.batch_size,
            player_count=self.player_count,
            stack_shape=POLICY_STACK_SHAPE,
            dtype=str(device_observation.dtype),
            device=str(device_observation.device),
            row_major_order=True,
            fresh_for_step_index=int(step_index) + 1,
            source_backend="fixed_action_tape_cpu_oracle_resident_handle_shim",
            host_fallback_allowed=False,
            metadata={"local_ceiling_not_h100_speed_evidence": True},
        )
        slot_id = int(step_index) % int(self.ring["slot_generation"].shape[0])
        policy_env_id = (
            np.arange(self.root_count, dtype=np.int32)
            + np.int32(int(step_index) * self.root_count)
        )
        policy_env_row = np.repeat(
            np.arange(self.batch_size, dtype=np.int32),
            self.player_count,
        )
        policy_player = np.tile(
            np.arange(self.player_count, dtype=np.int32),
            self.batch_size,
        )
        terminal_rows = np.asarray(_result_array(result, "terminal_rows"), dtype=np.int32)
        slot = _publish_compact_owner_mechanics_step_frame_slot(
            ring=self.ring,
            action_mask=np.asarray(_result_array(result, "action_mask"), dtype=np.bool_),
            reward=np.asarray(_result_array(result, "reward"), dtype=np.float32),
            final_reward_map=np.asarray(
                _result_array(result, "final_reward_map"),
                dtype=np.float32,
            ),
            done=np.asarray(_result_array(result, "done"), dtype=np.bool_),
            policy_env_id=policy_env_id,
            policy_env_row=policy_env_row,
            policy_player=policy_player,
            terminal_global_rows=terminal_rows,
            autoreset_global_rows=np.zeros(0, dtype=np.int32),
            episode_step=np.asarray(env.state["episode_step"], dtype=np.int32),
            elapsed_ms=np.asarray(
                _result_array(result, "source_physics_elapsed_ms"),
                dtype=np.float64,
            ),
            round_id=np.asarray(
                env.state.get("round_id", np.zeros(self.batch_size, dtype=np.int32)),
                dtype=np.int32,
            ),
            alive=np.asarray(env.state["alive"][:, : self.player_count], dtype=np.bool_),
            joint_action=np.asarray(joint_action, dtype=np.int16),
            batch_size=self.batch_size,
            terminated=np.asarray(_result_array(result, "terminated"), dtype=np.bool_),
            truncated=np.asarray(_result_array(result, "truncated"), dtype=np.bool_),
            terminal_reason=np.asarray(
                _result_array(result, "terminal_reason"),
                dtype=np.int16,
            ),
            death_count=np.asarray(_result_array(result, "death_count"), dtype=np.int32),
            death_player=np.asarray(_result_array(result, "death_player"), dtype=np.int16),
            death_cause=np.asarray(_result_array(result, "death_cause"), dtype=np.int16),
            death_hit_owner=np.asarray(
                _result_array(result, "death_hit_owner"),
                dtype=np.int16,
            ),
            winner=np.asarray(_result_array(result, "winner"), dtype=np.int16),
            draw=np.asarray(_result_array(result, "draw"), dtype=np.bool_),
            resident_observation=resident,
            step_frame_slot_id=slot_id,
            step_frame_generation=int(step_index),
        )
        step = self.stepper.step(slot)
        proof_metadata = {
            **dict(step.telemetry),
            **dict(getattr(step.action_step, "metadata", {}) or {}),
        }
        self.next_joint_action = np.asarray(step.next_joint_action, dtype=np.int16).copy()
        self.step_count += 1
        self.next_action_count += 1
        self.action_result_write_count += 1
        self.action_result_read_count += 1
        self.mechanics_slot_write_count += int(
            slot.metadata.get("compact_owner_mechanics_step_frame_slot_write_count", 0)
        )
        self.mechanics_slot_generation_verified_count += int(
            bool(
                proof_metadata.get(
                    "compact_owner_root_search_transaction_frame_generation_verified",
                    proof_metadata.get(
                        "compact_owner_mechanics_step_frame_handle_consumed",
                        False,
                    ),
                )
            )
        )
        self.mechanics_slot_digest_verified_count += int(
            bool(
                proof_metadata.get(
                    "compact_owner_root_search_transaction_frame_digest_verified",
                    proof_metadata.get(
                        "compact_owner_mechanics_step_frame_handle_digest_verified",
                        False,
                    ),
                )
            )
        )
        self.root_request_from_slot_count += int(
            bool(proof_metadata.get("compact_owner_step_frame_root_build_request_used", False))
        )
        self.root_request_from_batch_count += int(
            bool(
                proof_metadata.get(
                    "compact_owner_step_frame_root_build_request_from_batch_helper_used",
                    False,
                )
            )
        )
        self.hybrid_compact_batch_object_count += 0
        self.root_observation_copy_bytes += int(
            proof_metadata.get(
                "compact_owner_search_direct_root_build_request_observation_bytes_sent",
                0,
            )
            or 0
        )
        self.active_root_count += int(
            proof_metadata.get("compact_rollout_slab_active_root_count", 0) or 0
        )
        self.selected_action_count += int(step.action_step.selected_action.size)
        self.ctree_calls += int(proof_metadata.get("compact_rollout_slab_ctree_calls", 0) or 0)
        self.tolist_calls += int(
            proof_metadata.get("compact_rollout_slab_tolist_calls", 0) or 0
        )
        self.last_slot_id = int(slot_id)
        self.last_generation = int(step_index)
        _hash_update_text(
            self.slot_digest_checksum,
            "slot_digest",
            str(slot.step_frame_handle.digest),
        )
        _hash_update_array(
            self.next_joint_action_checksum,
            "next_joint_action",
            self.next_joint_action,
        )
        self._append_drain_records_to_replay_ring(current_resident=resident)
        self._append_replay_slot_from_previous_action(
            current_result=result,
            current_step_index=int(step_index),
        )
        self.previous_action_step = step.action_step

    def metadata(self) -> dict[str, Any]:
        service_metadata = dict(self.search_service.metadata)
        return {
            "owner_slot_ceiling_enabled": True,
            "owner_slot_ceiling_schema_id": "curvyzero_owner_slot_ceiling/v1",
            "owner_slot_ceiling_step_count": int(self.step_count),
            "owner_slot_ceiling_tape_bootstrap_action_count": int(
                self.tape_bootstrap_action_count
            ),
            "owner_slot_ceiling_feedback_action_count": int(self.feedback_action_count),
            "owner_slot_ceiling_measured_feedback_action_count": int(
                self.measured_feedback_action_count
            ),
            "owner_slot_ceiling_prev_next_joint_action_match_count": int(
                self.prev_next_joint_action_match_count
            ),
            "owner_slot_ceiling_prev_next_joint_action_mismatch_count": int(
                self.prev_next_joint_action_mismatch_count
            ),
            "owner_slot_ceiling_mechanics_slot_write_count": int(
                self.mechanics_slot_write_count
            ),
            "owner_slot_ceiling_mechanics_slot_generation_verified_count": int(
                self.mechanics_slot_generation_verified_count
            ),
            "owner_slot_ceiling_mechanics_slot_digest_verified_count": int(
                self.mechanics_slot_digest_verified_count
            ),
            "owner_slot_ceiling_root_request_from_slot_count": int(
                self.root_request_from_slot_count
            ),
            "owner_slot_ceiling_root_request_from_batch_count": int(
                self.root_request_from_batch_count
            ),
            "owner_slot_ceiling_hybrid_compact_batch_object_count": int(
                self.hybrid_compact_batch_object_count
            ),
            "owner_slot_ceiling_action_result_write_count": int(
                self.action_result_write_count
            ),
            "owner_slot_ceiling_action_result_read_count": int(
                self.action_result_read_count
            ),
            "owner_slot_ceiling_next_action_count": int(self.next_action_count),
            "owner_slot_ceiling_root_observation_copy_bytes": int(
                self.root_observation_copy_bytes
            ),
            "owner_slot_ceiling_active_root_count": int(self.active_root_count),
            "owner_slot_ceiling_selected_action_count": int(self.selected_action_count),
            "owner_slot_ceiling_ctree_calls": int(self.ctree_calls),
            "owner_slot_ceiling_tolist_calls": int(self.tolist_calls),
            "owner_slot_ceiling_replay_ring_append_record_count": int(
                self.replay_ring_append_record_count
            ),
            "owner_slot_ceiling_replay_ring_append_call_count": int(
                self.replay_ring_append_call_count
            ),
            "owner_slot_ceiling_replay_ring_appended_row_count": int(
                self.replay_ring_appended_row_count
            ),
            "owner_slot_ceiling_replay_ring_entry_count": int(
                self.replay_ring.entry_count
            ),
            "owner_slot_ceiling_replay_ring_stored_index_row_count": int(
                self.replay_ring.stored_index_row_count
            ),
            "owner_slot_ceiling_replay_ring_evicted_entry_count": int(
                self.replay_ring.evicted_entry_count
            ),
            "owner_slot_ceiling_replay_ring_evicted_index_row_count": int(
                self.replay_ring.evicted_index_row_count
            ),
            "owner_slot_ceiling_replay_ring_sample_batch_built": bool(
                self.replay_ring_sample_batch_built
            ),
            "owner_slot_ceiling_replay_ring_sample_gate_calls": int(
                self.replay_ring_sample_gate_calls
            ),
            "owner_slot_ceiling_replay_ring_sample_row_count": int(
                self.replay_ring_sample_row_count
            ),
            "owner_slot_ceiling_replay_ring_sample_target_row_count": int(
                self.replay_ring_sample_target_row_count
            ),
            "owner_slot_ceiling_replay_ring_sample_source": str(
                self.replay_ring_sample_source
            ),
            "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample": bool(
                self.replay_ring_sample_device_replay_index_rows_sample
            ),
            "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample_all": bool(
                self.replay_ring_sample_device_replay_index_rows_sample_all
            ),
            "owner_slot_ceiling_replay_ring_sample_resident_device_sample_batch": bool(
                self.replay_ring_sample_resident_device_sample_batch
            ),
            "owner_slot_ceiling_replay_ring_sample_host_observation_fallback_allowed": bool(
                self.replay_ring_sample_host_observation_fallback_allowed
            ),
            "owner_slot_ceiling_replay_ring_sample_observation_provider_used_count": int(
                self.replay_ring_sample_observation_provider_used_count
            ),
            "owner_slot_ceiling_replay_ring_sample_row_id_checksum": (
                self.replay_ring_sample_row_id_checksum
            ),
            "owner_slot_ceiling_replay_ring_sample_action_checksum": (
                self.replay_ring_sample_action_checksum
            ),
            "owner_slot_ceiling_replay_ring_sample_reward_checksum": (
                self.replay_ring_sample_reward_checksum
            ),
            "owner_slot_ceiling_replay_ring_sample_done_checksum": (
                self.replay_ring_sample_done_checksum
            ),
            "owner_slot_ceiling_replay_ring_sample_observation_checksum": (
                self.replay_ring_sample_observation_checksum
            ),
            "owner_slot_ceiling_replay_ring_sample_next_observation_checksum": (
                self.replay_ring_sample_next_observation_checksum
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_batch_built": bool(
                self.replay_ring_learner_unroll2_batch_built
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls": int(
                self.replay_ring_learner_unroll2_sample_gate_calls
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_sample_row_count": int(
                self.replay_ring_learner_unroll2_sample_row_count
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_target_row_count": int(
                self.replay_ring_learner_unroll2_target_row_count
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_num_unroll_steps": int(
                self.replay_ring_learner_unroll2_num_unroll_steps
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_require_next_targets": bool(
                self.replay_ring_learner_unroll2_require_next_targets
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_batch_only": bool(
                self.replay_ring_learner_unroll2_batch_only
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_source": (
                self.replay_ring_learner_unroll2_source
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source": (
                self.replay_ring_learner_unroll2_candidate_universe_source
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count": int(
                self.replay_ring_learner_unroll2_explicit_unroll_target_group_count
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count": int(
                self.replay_ring_learner_unroll2_next_target_eligible_pair_count
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_observation_provider_used_count": int(
                self.replay_ring_learner_unroll2_observation_provider_used_count
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_schema_id": (
                self.replay_ring_learner_unroll2_schema_id
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source": (
                self.replay_ring_learner_unroll2_prevalidation_source
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_host_fallback_allowed": bool(
                self.replay_ring_learner_unroll2_host_fallback_allowed
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_action_shape": list(
                self.replay_ring_learner_unroll2_action_shape
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_shape": list(
                self.replay_ring_learner_unroll2_target_reward_shape
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_shape": list(
                self.replay_ring_learner_unroll2_target_value_shape
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_shape": list(
                self.replay_ring_learner_unroll2_target_policy_shape
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_action_mask_shape": list(
                self.replay_ring_learner_unroll2_action_mask_shape
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_action_checksum": (
                self.replay_ring_learner_unroll2_action_checksum
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_checksum": (
                self.replay_ring_learner_unroll2_target_reward_checksum
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_checksum": (
                self.replay_ring_learner_unroll2_target_value_checksum
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_checksum": (
                self.replay_ring_learner_unroll2_target_policy_checksum
            ),
            "owner_slot_ceiling_replay_ring_learner_unroll2_source_record_window_checksum": (
                self.replay_ring_learner_unroll2_source_record_window_checksum
            ),
            **{
                f"owner_slot_ceiling_replay_ring_columnar_append_{key}": value
                for key, value in self.replay_ring.columnar_append_telemetry_snapshot().items()
            },
            "owner_slot_ceiling_replay_slot_schema_id": (
                "curvyzero_owner_slot_ceiling_fixed_replay_slot/v1"
            ),
            "owner_slot_ceiling_replay_slot_capacity": int(self.replay_slot_capacity),
            "owner_slot_ceiling_replay_slot_max_rows": int(self.replay_slot_max_rows),
            "owner_slot_ceiling_replay_slot_append_count": int(
                self.replay_slot_append_count
            ),
            "owner_slot_ceiling_replay_slot_append_row_count": int(
                self.replay_slot_append_row_count
            ),
            "owner_slot_ceiling_replay_slot_object_entry_count": int(
                self.replay_slot_object_entry_count
            ),
            "owner_slot_ceiling_parent_replay_object_count": int(
                self.parent_replay_object_count
            ),
            "owner_slot_ceiling_selected_group_object_count": int(
                self.selected_group_object_count
            ),
            "owner_slot_ceiling_sample_handle_schema_id": (
                "curvyzero_owner_slot_ceiling_sample_handle/v1"
            ),
            "owner_slot_ceiling_sample_batch_built": bool(self.sample_batch_built),
            "owner_slot_ceiling_sample_gate_calls": int(self.sample_gate_calls),
            "owner_slot_ceiling_sample_handle_create_count": int(
                self.sample_handle_create_count
            ),
            "owner_slot_ceiling_sample_handle_resolve_count": int(
                self.sample_handle_resolve_count
            ),
            "owner_slot_ceiling_sample_handle_inline_resolve_count": int(
                self.sample_handle_inline_resolve_count
            ),
            "owner_slot_ceiling_sample_handle_pending_count": int(
                self.sample_handle_pending_count
            ),
            "owner_slot_ceiling_sample_row_count": int(self.sample_row_count),
            "owner_slot_ceiling_sample_target_row_count": int(
                self.sample_target_row_count
            ),
            "owner_slot_ceiling_last_slot_id": int(self.last_slot_id),
            "owner_slot_ceiling_last_generation": int(self.last_generation),
            "owner_slot_ceiling_action_source_sequence_checksum": (
                self.action_source_sequence_checksum.hexdigest()
            ),
            "owner_slot_ceiling_slot_digest_checksum": self.slot_digest_checksum.hexdigest(),
            "owner_slot_ceiling_next_joint_action_checksum": (
                self.next_joint_action_checksum.hexdigest()
            ),
            "owner_slot_ceiling_replay_slot_window_checksum": (
                self.replay_slot_window_checksum.hexdigest()
            ),
            "owner_slot_ceiling_sample_handle_checksum": (
                self.sample_handle_checksum.hexdigest()
            ),
            "owner_slot_ceiling_sample_row_id_checksum": self.sample_row_id_checksum,
            "owner_slot_ceiling_sample_action_checksum": self.sample_action_checksum,
            "owner_slot_ceiling_sample_reward_checksum": self.sample_reward_checksum,
            "owner_slot_ceiling_sample_done_checksum": self.sample_done_checksum,
            **service_metadata,
        }

    def _append_replay_slot_from_previous_action(
        self,
        *,
        current_result: Any,
        current_step_index: int,
    ) -> None:
        previous = self.previous_action_step
        if previous is None:
            return
        selected = np.asarray(previous.selected_action, dtype=np.int16).reshape(-1)
        if selected.size <= 0:
            return
        env_row = np.asarray(previous.env_row, dtype=np.int64).reshape(-1)
        player = np.asarray(previous.player, dtype=np.int64).reshape(-1)
        if selected.size != env_row.size or selected.size != player.size:
            raise RuntimeError("owner-slot replay slot action/root shape mismatch")
        reward = np.asarray(_result_array(current_result, "reward"), dtype=np.float32)
        done = np.asarray(_result_array(current_result, "done"), dtype=np.bool_)
        terminated = np.asarray(
            _result_array(current_result, "terminated"),
            dtype=np.bool_,
        )
        truncated = np.asarray(
            _result_array(current_result, "truncated"),
            dtype=np.bool_,
        )
        slot_id = int(self.replay_slot_append_count % self.replay_slot_capacity)
        generation = int(self.replay_slot_append_count + 1)
        row_count = int(selected.size)
        if row_count > self.replay_slot_max_rows:
            raise RuntimeError("owner-slot replay slot row capacity exceeded")
        row_id = (
            np.arange(row_count, dtype=np.int64)
            + np.int64(generation * self.replay_slot_max_rows)
        )
        self.replay_slot_generation[slot_id] = np.int64(generation)
        self.replay_slot_row_count[slot_id] = np.int32(row_count)
        tail = slice(row_count, self.replay_slot_max_rows)
        self.replay_slot_row_id[slot_id, :row_count] = row_id
        self.replay_slot_env_row[slot_id, :row_count] = env_row.astype(np.int32)
        self.replay_slot_player[slot_id, :row_count] = player.astype(np.int16)
        self.replay_slot_action[slot_id, :row_count] = selected
        self.replay_slot_reward[slot_id, :row_count] = reward[env_row, player]
        self.replay_slot_done[slot_id, :row_count] = done[env_row]
        self.replay_slot_terminated[slot_id, :row_count] = terminated[env_row]
        self.replay_slot_truncated[slot_id, :row_count] = truncated[env_row]
        self.replay_slot_row_id[slot_id, tail] = 0
        self.replay_slot_env_row[slot_id, tail] = 0
        self.replay_slot_player[slot_id, tail] = 0
        self.replay_slot_action[slot_id, tail] = 0
        self.replay_slot_reward[slot_id, tail] = 0.0
        self.replay_slot_done[slot_id, tail] = False
        self.replay_slot_terminated[slot_id, tail] = False
        self.replay_slot_truncated[slot_id, tail] = False
        self.replay_slot_append_count += 1
        self.replay_slot_append_row_count += row_count
        _hash_update_text(
            self.replay_slot_window_checksum,
            "slot",
            f"{slot_id}:{generation}:{current_step_index}:{row_count}",
        )
        _hash_update_array(self.replay_slot_window_checksum, "row_id", row_id)
        _hash_update_array(self.replay_slot_window_checksum, "env_row", env_row)
        _hash_update_array(self.replay_slot_window_checksum, "player", player)
        _hash_update_array(self.replay_slot_window_checksum, "action", selected)
        _hash_update_array(
            self.replay_slot_window_checksum,
            "reward",
            self.replay_slot_reward[slot_id, :row_count],
        )
        _hash_update_array(
            self.replay_slot_window_checksum,
            "done",
            self.replay_slot_done[slot_id, :row_count],
        )
        self._sample_replay_slot_handle()

    def _append_drain_records_to_replay_ring(self, *, current_resident: Any) -> None:
        records = self.search_service.drain_staged_replay_records(
            current_resident_observation_replay_snapshot=current_resident,
        )
        if not records:
            return
        appended = self.replay_ring.append_columnar_entries(records)
        self.replay_ring_append_call_count += 1
        self.replay_ring_append_record_count += int(appended)
        for record in records:
            row_count = _leading_dim(record.index_rows.action)
            self.replay_ring_appended_row_count += row_count
        self._sample_replay_ring()

    def _sample_replay_ring(self) -> None:
        if self.replay_ring_sample_batch_built:
            self._sample_replay_ring_learner_unroll2()
            return
        sample_batch_size = min(8, int(self.replay_ring.stored_index_row_count))
        if sample_batch_size <= 0:
            return
        sample_seed = 37_000 + int(self.replay_ring_sample_gate_calls)
        result = self.replay_ring.sample(
            seed=sample_seed,
            sample_batch_size=sample_batch_size,
            require_next_targets=False,
            num_unroll_steps=1,
            build_compact_muzero_learner_batch=False,
            compact_muzero_learner_batch_only=False,
        )
        sample = result.get("sample_batch")
        if sample is None:
            return
        metadata = dict(result.get("sample_metadata") or getattr(sample, "metadata", {}) or {})
        self.replay_ring_sample_gate_calls += 1
        self.replay_ring_sample_batch_built = True
        self.replay_ring_sample_row_count = int(
            result.get("sample_row_count", sample_batch_size)
        )
        self.replay_ring_sample_target_row_count = int(
            result.get("target_row_count", self.replay_ring_sample_row_count)
        )
        self.replay_ring_sample_source = str(metadata.get("source", ""))
        self.replay_ring_sample_device_replay_index_rows_sample = bool(
            metadata.get("device_replay_index_rows_sample", False)
        )
        self.replay_ring_sample_device_replay_index_rows_sample_all = bool(
            metadata.get("device_replay_index_rows_sample_all", False)
        )
        self.replay_ring_sample_resident_device_sample_batch = bool(
            metadata.get("resident_device_sample_batch", False)
        )
        self.replay_ring_sample_host_observation_fallback_allowed = bool(
            metadata.get("host_observation_fallback_allowed", False)
        )
        if bool(metadata.get("observation_provider_used", False)):
            self.replay_ring_sample_observation_provider_used_count += 1
        self.replay_ring_sample_row_id_checksum = _array_checksum(
            np.asarray(sample.row_id)
        )
        self.replay_ring_sample_action_checksum = _array_checksum(
            np.asarray(sample.action)
        )
        self.replay_ring_sample_reward_checksum = _array_checksum(
            np.asarray(sample.reward)
        )
        self.replay_ring_sample_done_checksum = _array_checksum(np.asarray(sample.done))
        self.replay_ring_sample_observation_checksum = _array_checksum(
            np.asarray(sample.observation)
        )
        self.replay_ring_sample_next_observation_checksum = _array_checksum(
            np.asarray(sample.next_observation)
        )
        self._sample_replay_ring_learner_unroll2()

    def _sample_replay_ring_learner_unroll2(self) -> None:
        if int(self.replay_ring.entry_count) < 3:
            return
        sample_batch_size = min(8, int(self.replay_ring.stored_index_row_count))
        if sample_batch_size <= 0:
            return
        sample_seed = 47_000 + int(self.replay_ring_learner_unroll2_sample_gate_calls)
        result = self.replay_ring.sample(
            seed=sample_seed,
            sample_batch_size=sample_batch_size,
            require_next_targets=True,
            num_unroll_steps=2,
            build_compact_muzero_learner_batch=True,
            compact_muzero_learner_batch_only=True,
        )
        learner_batch = result.get("learner_batch")
        if learner_batch is None:
            return
        metadata = dict(result.get("sample_metadata") or {})
        learner_metadata = dict(getattr(learner_batch, "metadata", {}) or {})
        telemetry = dict(result.get("telemetry") or {})

        def shape_list(value: Any) -> list[int]:
            shape = getattr(value, "shape", ())
            return [int(dim) for dim in shape]

        self.replay_ring_learner_unroll2_sample_gate_calls += 1
        self.replay_ring_learner_unroll2_batch_built = True
        self.replay_ring_learner_unroll2_sample_row_count = int(
            result.get("sample_row_count", 0)
        )
        self.replay_ring_learner_unroll2_target_row_count = int(
            result.get(
                "target_row_count",
                self.replay_ring_learner_unroll2_sample_row_count,
            )
        )
        self.replay_ring_learner_unroll2_num_unroll_steps = int(
            learner_metadata.get(
                "compact_muzero_learner_num_unroll_steps",
                metadata.get("num_unroll_steps", 0),
            )
        )
        self.replay_ring_learner_unroll2_require_next_targets = bool(
            metadata.get("require_next_targets", False)
        )
        self.replay_ring_learner_unroll2_batch_only = bool(
            telemetry.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only",
                False,
            )
        )
        self.replay_ring_learner_unroll2_source = str(metadata.get("source", ""))
        self.replay_ring_learner_unroll2_candidate_universe_source = str(
            metadata.get("candidate_universe_source", "none")
        )
        self.replay_ring_learner_unroll2_explicit_unroll_target_group_count = int(
            metadata.get("explicit_unroll_target_group_count", 0)
        )
        self.replay_ring_learner_unroll2_next_target_eligible_pair_count = int(
            metadata.get("next_target_eligible_pair_count", 0)
        )
        if bool(metadata.get("observation_provider_used", False)):
            self.replay_ring_learner_unroll2_observation_provider_used_count += 1
        self.replay_ring_learner_unroll2_schema_id = str(
            learner_metadata.get(
                "compact_muzero_learner_batch_schema_id",
                learner_metadata.get("sample_schema_id", "none"),
            )
        )
        self.replay_ring_learner_unroll2_prevalidation_source = str(
            learner_metadata.get("compact_muzero_learner_batch_prevalidation_source", "none")
        )
        self.replay_ring_learner_unroll2_host_fallback_allowed = bool(
            learner_metadata.get("compact_muzero_learner_host_fallback_allowed", True)
        )
        self.replay_ring_learner_unroll2_action_shape = shape_list(learner_batch.action)
        self.replay_ring_learner_unroll2_target_reward_shape = shape_list(
            learner_batch.target_reward
        )
        self.replay_ring_learner_unroll2_target_value_shape = shape_list(
            learner_batch.target_value
        )
        self.replay_ring_learner_unroll2_target_policy_shape = shape_list(
            learner_batch.target_policy
        )
        self.replay_ring_learner_unroll2_action_mask_shape = shape_list(
            learner_batch.action_mask
        )
        self.replay_ring_learner_unroll2_action_checksum = _array_checksum_any(
            learner_batch.action
        )
        self.replay_ring_learner_unroll2_target_reward_checksum = _array_checksum_any(
            learner_batch.target_reward
        )
        self.replay_ring_learner_unroll2_target_value_checksum = _array_checksum_any(
            learner_batch.target_value
        )
        self.replay_ring_learner_unroll2_target_policy_checksum = _array_checksum_any(
            learner_batch.target_policy
        )
        self.replay_ring_learner_unroll2_source_record_window_checksum = str(
            metadata.get("source_record_window_checksum", "")
        )

    def _sample_replay_slot_handle(self) -> None:
        total_rows = int(self.replay_slot_row_count.sum())
        if total_rows <= 0:
            return
        flat_slot = np.empty(total_rows, dtype=np.int32)
        flat_offset = np.empty(total_rows, dtype=np.int32)
        cursor = 0
        for slot_id, row_count_value in enumerate(self.replay_slot_row_count):
            row_count = int(row_count_value)
            if row_count <= 0 or int(self.replay_slot_generation[slot_id]) <= 0:
                continue
            stop = cursor + row_count
            flat_slot[cursor:stop] = int(slot_id)
            flat_offset[cursor:stop] = np.arange(row_count, dtype=np.int32)
            cursor = stop
        flat_slot = flat_slot[:cursor]
        flat_offset = flat_offset[:cursor]
        sample_size = min(8, int(cursor))
        if sample_size <= 0:
            return
        sample_seed = 17_000 + int(self.sample_handle_create_count)
        rng = np.random.default_rng(sample_seed)
        chosen = np.asarray(
            rng.choice(int(cursor), size=sample_size, replace=False),
            dtype=np.int64,
        )
        sample_slot = flat_slot[chosen]
        sample_offset = flat_offset[chosen]
        sample_row_id = self.replay_slot_row_id[sample_slot, sample_offset]
        sample_action = self.replay_slot_action[sample_slot, sample_offset]
        sample_reward = self.replay_slot_reward[sample_slot, sample_offset]
        sample_done = self.replay_slot_done[sample_slot, sample_offset]
        self.sample_handle_create_count += 1
        self.sample_handle_resolve_count += 1
        self.sample_handle_inline_resolve_count += 1
        self.sample_handle_pending_count = 0
        self.sample_gate_calls += 1
        self.sample_batch_built = True
        self.sample_row_count = int(sample_size)
        self.sample_target_row_count = int(sample_size)
        _hash_update_text(
            self.sample_handle_checksum,
            "handle",
            (
                f"{self.sample_handle_create_count}:{sample_seed}:"
                f"{int(sample_size)}:{int(cursor)}"
            ),
        )
        _hash_update_array(self.sample_handle_checksum, "slot", sample_slot)
        _hash_update_array(self.sample_handle_checksum, "offset", sample_offset)
        _hash_update_array(self.sample_handle_checksum, "row_id", sample_row_id)
        self.sample_row_id_checksum = _array_checksum(sample_row_id)
        self.sample_action_checksum = _array_checksum(sample_action)
        self.sample_reward_checksum = _array_checksum(sample_reward)
        self.sample_done_checksum = _array_checksum(sample_done)

    def close(self) -> None:
        self.stepper.close()


def _empty_owner_slot_metadata() -> dict[str, Any]:
    empty_digest = hashlib.blake2b(digest_size=16).hexdigest()
    return {
        "owner_slot_ceiling_enabled": False,
        "owner_slot_ceiling_schema_id": "none",
        "owner_slot_ceiling_step_count": 0,
        "owner_slot_ceiling_tape_bootstrap_action_count": 0,
        "owner_slot_ceiling_feedback_action_count": 0,
        "owner_slot_ceiling_measured_feedback_action_count": 0,
        "owner_slot_ceiling_prev_next_joint_action_match_count": 0,
        "owner_slot_ceiling_prev_next_joint_action_mismatch_count": 0,
        "owner_slot_ceiling_mechanics_slot_write_count": 0,
        "owner_slot_ceiling_mechanics_slot_generation_verified_count": 0,
        "owner_slot_ceiling_mechanics_slot_digest_verified_count": 0,
        "owner_slot_ceiling_root_request_from_slot_count": 0,
        "owner_slot_ceiling_root_request_from_batch_count": 0,
        "owner_slot_ceiling_hybrid_compact_batch_object_count": 0,
        "owner_slot_ceiling_action_result_write_count": 0,
        "owner_slot_ceiling_action_result_read_count": 0,
        "owner_slot_ceiling_next_action_count": 0,
        "owner_slot_ceiling_root_observation_copy_bytes": 0,
        "owner_slot_ceiling_active_root_count": 0,
        "owner_slot_ceiling_selected_action_count": 0,
        "owner_slot_ceiling_ctree_calls": 0,
        "owner_slot_ceiling_tolist_calls": 0,
        "owner_slot_ceiling_replay_slot_schema_id": "none",
        "owner_slot_ceiling_replay_slot_capacity": 0,
        "owner_slot_ceiling_replay_slot_max_rows": 0,
        "owner_slot_ceiling_replay_slot_append_count": 0,
        "owner_slot_ceiling_replay_slot_append_row_count": 0,
        "owner_slot_ceiling_replay_slot_object_entry_count": 0,
        "owner_slot_ceiling_parent_replay_object_count": 0,
        "owner_slot_ceiling_selected_group_object_count": 0,
        "owner_slot_ceiling_sample_handle_schema_id": "none",
        "owner_slot_ceiling_sample_batch_built": False,
        "owner_slot_ceiling_sample_gate_calls": 0,
        "owner_slot_ceiling_sample_handle_create_count": 0,
        "owner_slot_ceiling_sample_handle_resolve_count": 0,
        "owner_slot_ceiling_sample_handle_inline_resolve_count": 0,
        "owner_slot_ceiling_sample_handle_pending_count": 0,
        "owner_slot_ceiling_sample_row_count": 0,
        "owner_slot_ceiling_sample_target_row_count": 0,
        "owner_slot_ceiling_last_slot_id": -1,
        "owner_slot_ceiling_last_generation": -1,
        "owner_slot_ceiling_action_source_sequence_checksum": empty_digest,
        "owner_slot_ceiling_slot_digest_checksum": empty_digest,
        "owner_slot_ceiling_next_joint_action_checksum": empty_digest,
        "owner_slot_ceiling_replay_slot_window_checksum": empty_digest,
        "owner_slot_ceiling_sample_handle_checksum": empty_digest,
        "owner_slot_ceiling_sample_row_id_checksum": "",
        "owner_slot_ceiling_sample_action_checksum": "",
        "owner_slot_ceiling_sample_reward_checksum": "",
        "owner_slot_ceiling_sample_done_checksum": "",
        "owner_slot_ceiling_stage_replay_schema_id": "none",
        "owner_slot_ceiling_stage_replay_transport_entry_count": 0,
        "owner_slot_ceiling_stage_replay_transition_entry_count": 0,
        "owner_slot_ceiling_stage_replay_payload_cache_hit_count": 0,
        "owner_slot_ceiling_stage_replay_payload_cache_miss_count": 0,
        "owner_slot_ceiling_stage_replay_payload_release_count": 0,
        "owner_slot_ceiling_stage_replay_payload_pending_count": 0,
        "owner_slot_ceiling_stage_replay_pending_record_count": 0,
        "owner_slot_ceiling_stage_replay_ready_record_count": 0,
        "owner_slot_ceiling_stage_replay_drained_record_count": 0,
        "owner_slot_ceiling_stage_replay_index_rows_build_count": 0,
        "owner_slot_ceiling_stage_replay_index_rows_row_count": 0,
        "owner_slot_ceiling_stage_replay_device_index_rows_build_count": 0,
        "owner_slot_ceiling_stage_replay_device_index_rows_row_count": 0,
        "owner_slot_ceiling_stage_replay_slot_append_count": 0,
        "owner_slot_ceiling_stage_replay_slot_append_row_count": 0,
        "owner_slot_ceiling_stage_sample_batch_built": False,
        "owner_slot_ceiling_stage_sample_gate_calls": 0,
        "owner_slot_ceiling_stage_sample_handle_create_count": 0,
        "owner_slot_ceiling_stage_sample_handle_resolve_count": 0,
        "owner_slot_ceiling_stage_sample_handle_inline_resolve_count": 0,
        "owner_slot_ceiling_stage_sample_handle_pending_count": 0,
        "owner_slot_ceiling_stage_sample_row_count": 0,
        "owner_slot_ceiling_stage_sample_target_row_count": 0,
        "owner_slot_ceiling_stage_replay_slot_window_checksum": empty_digest,
        "owner_slot_ceiling_stage_sample_handle_checksum": empty_digest,
        "owner_slot_ceiling_stage_sample_row_id_checksum": "",
        "owner_slot_ceiling_stage_sample_action_checksum": "",
        "owner_slot_ceiling_stage_sample_reward_checksum": "",
        "owner_slot_ceiling_stage_sample_done_checksum": "",
        "owner_slot_ceiling_replay_ring_append_record_count": 0,
        "owner_slot_ceiling_replay_ring_append_call_count": 0,
        "owner_slot_ceiling_replay_ring_appended_row_count": 0,
        "owner_slot_ceiling_replay_ring_entry_count": 0,
        "owner_slot_ceiling_replay_ring_stored_index_row_count": 0,
        "owner_slot_ceiling_replay_ring_evicted_entry_count": 0,
        "owner_slot_ceiling_replay_ring_evicted_index_row_count": 0,
        "owner_slot_ceiling_replay_ring_sample_batch_built": False,
        "owner_slot_ceiling_replay_ring_sample_gate_calls": 0,
        "owner_slot_ceiling_replay_ring_sample_row_count": 0,
        "owner_slot_ceiling_replay_ring_sample_target_row_count": 0,
        "owner_slot_ceiling_replay_ring_sample_source": "none",
        "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample": False,
        "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample_all": False,
        "owner_slot_ceiling_replay_ring_sample_resident_device_sample_batch": False,
        "owner_slot_ceiling_replay_ring_sample_host_observation_fallback_allowed": False,
        "owner_slot_ceiling_replay_ring_sample_observation_provider_used_count": 0,
        "owner_slot_ceiling_replay_ring_sample_row_id_checksum": "",
        "owner_slot_ceiling_replay_ring_sample_action_checksum": "",
        "owner_slot_ceiling_replay_ring_sample_reward_checksum": "",
        "owner_slot_ceiling_replay_ring_sample_done_checksum": "",
        "owner_slot_ceiling_replay_ring_sample_observation_checksum": "",
        "owner_slot_ceiling_replay_ring_sample_next_observation_checksum": "",
        "owner_slot_ceiling_replay_ring_learner_unroll2_batch_built": False,
        "owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls": 0,
        "owner_slot_ceiling_replay_ring_learner_unroll2_sample_row_count": 0,
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_row_count": 0,
        "owner_slot_ceiling_replay_ring_learner_unroll2_num_unroll_steps": 0,
        "owner_slot_ceiling_replay_ring_learner_unroll2_require_next_targets": False,
        "owner_slot_ceiling_replay_ring_learner_unroll2_batch_only": False,
        "owner_slot_ceiling_replay_ring_learner_unroll2_source": "none",
        "owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source": "none",
        "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count": 0,
        "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count": 0,
        "owner_slot_ceiling_replay_ring_learner_unroll2_observation_provider_used_count": 0,
        "owner_slot_ceiling_replay_ring_learner_unroll2_schema_id": "none",
        "owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source": "none",
        "owner_slot_ceiling_replay_ring_learner_unroll2_host_fallback_allowed": False,
        "owner_slot_ceiling_replay_ring_learner_unroll2_action_shape": [],
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_shape": [],
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_shape": [],
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_shape": [],
        "owner_slot_ceiling_replay_ring_learner_unroll2_action_mask_shape": [],
        "owner_slot_ceiling_replay_ring_learner_unroll2_action_checksum": "",
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_checksum": "",
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_checksum": "",
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_checksum": "",
        "owner_slot_ceiling_replay_ring_learner_unroll2_source_record_window_checksum": "",
        "owner_slot_ceiling_replay_ring_columnar_append_call_count": 0.0,
        "owner_slot_ceiling_replay_ring_columnar_append_record_count": 0.0,
        "owner_slot_ceiling_replay_ring_columnar_append_entry_view_object_count": 0.0,
        "owner_slot_ceiling_replay_ring_columnar_append_step_view_object_count": 0.0,
    }


def _empty_replay_metadata() -> dict[str, Any]:
    empty_digest = hashlib.blake2b(digest_size=16).hexdigest()
    return {
        "slab_replay_enabled": False,
        "slab_search_feedback_closed_loop": False,
        "slab_replay_tape_bootstrap_action_count": 0,
        "slab_replay_feedback_action_count": 0,
        "slab_replay_measured_feedback_action_count": 0,
        "slab_replay_prev_next_joint_action_match_count": 0,
        "slab_replay_prev_next_joint_action_mismatch_count": 0,
        "slab_replay_feedback_differs_from_tape_count": 0,
        "slab_replay_action_source_sequence_checksum": empty_digest,
        "slab_step_count": 0,
        "slab_root_count": 0,
        "slab_active_root_count": 0,
        "slab_inactive_root_count": 0,
        "slab_selected_action_count": 0,
        "slab_max_active_root_count": 0,
        "slab_action_count": 0,
        "slab_num_simulations": 0,
        "slab_ctree_calls": 0,
        "slab_tolist_calls": 0,
        "slab_per_sim_d2h_bytes": 0,
        "slab_action_d2h_bytes": 0,
        "slab_deferred_replay_payload_d2h_bytes": 0,
        "slab_root_observation_copy_bytes": 0,
        "slab_committed_index_group_count": 0,
        "slab_committed_index_row_count": 0,
        "slab_committed_terminal_row_count": 0,
        "slab_committed_next_final_observation_row_count": 0,
        "slab_replay_payload_flush_count": 0,
        "slab_replay_payload_d2h_bytes": 0,
        "slab_replay_index_rows_observation_materialized": False,
        "slab_replay_index_rows_next_observation_materialized": False,
        "slab_replay_sample_batch_built": False,
        "slab_replay_sample_batch_size": 0,
        "slab_replay_sample_seed": 0,
        "slab_replay_sample_row_id_checksum": "",
        "slab_replay_sample_action_checksum": "",
        "slab_replay_sample_observation_checksum": "",
        "slab_replay_sample_next_observation_checksum": "",
        "slab_replay_sample_record_index_checksum": "",
        "slab_replay_sample_policy_row_checksum": "",
        "slab_replay_index_rows_checksum": empty_digest,
        "slab_replay_joint_action_feedback_checksum": empty_digest,
        "slab_replay_root_batch_checksum": empty_digest,
        "slab_replay_action_step_checksum": empty_digest,
        "slab_next_joint_action_checksum": empty_digest,
        "slab_replay_pending_uncommitted_count": 0,
        "slab_replay_action_check_enforced": False,
        "slab_replay_root_observation_copied": False,
        "slab_retains_committed_index_rows": False,
        "replay_append_count": 0,
        "replay_ring_entry_count": 0,
        "replay_ring_stored_index_row_count": 0,
        "replay_ring_evicted_entry_count": 0,
        "replay_ring_evicted_index_row_count": 0,
        "sample_gate_calls": 0,
        "sample_row_count": 0,
        "sample_target_row_count": 0,
        "sample_seed": 0,
        "sampled_flat_row_checksum": "",
        "sample_position_order_checksum": "",
        "source_record_pair_checksum": "",
        "source_record_window_checksum": "",
        "sample_row_id_checksum": "",
        "sample_action_checksum": "",
        "sample_observation_checksum": "",
        "sample_next_observation_checksum": "",
        "sample_reward_checksum": "",
        "sample_done_checksum": "",
    }


def _accumulate_replay_index_rows_checksum(hasher: Any, index_rows: Any) -> None:
    for name in (
        "compact_root_row",
        "policy_env_id",
        "policy_row",
        "env_row",
        "player",
        "action",
        "action_mask",
        "policy_target",
        "root_value",
        "reward",
        "final_reward",
        "done",
        "terminated",
        "truncated",
        "next_final_observation_row",
        "to_play",
    ):
        _hash_update_array(hasher, name, np.asarray(getattr(index_rows, name)))
    _hash_update_text(hasher, "policy_source", str(index_rows.policy_source))


def _replay_chunk_from_compact_batches(
    compact_batches: list[HybridCompactBatch],
) -> SourceStateMultiplayerTrainerReplayChunkV0:
    if not compact_batches:
        empty_observation = np.zeros((0, 0, 0, *POLICY_STACK_SHAPE), dtype=np.float32)
        arrays = {
            "observation": empty_observation,
            "legal_action_mask": np.zeros((0, 0, 0, ACTION_COUNT), dtype=np.bool_),
            "lightzero_action_mask": np.zeros((0, 0, 0, ACTION_COUNT), dtype=np.bool_),
            "live_mask": np.zeros((0, 0, 0), dtype=np.bool_),
            "joint_action": np.zeros((0, 0, 0), dtype=np.int16),
            "reward": np.zeros((0, 0, 0), dtype=np.float32),
            "done": np.zeros((0, 0), dtype=np.bool_),
            "terminated": np.zeros((0, 0), dtype=np.bool_),
            "truncated": np.zeros((0, 0), dtype=np.bool_),
            "final_observation": empty_observation.copy(),
            "final_observation_row_mask": np.zeros((0, 0), dtype=np.bool_),
            "final_reward_map": np.zeros((0, 0, 0), dtype=np.float32),
        }
        return SourceStateMultiplayerTrainerReplayChunkV0(
            metadata={"producer": "fixed_action_tape_closed_search_feedback_slab"},
            arrays=arrays,
            policy_rows=(),
            records=(),
        )
    arrays = {
        "observation": np.stack(
            [
                np.asarray(batch.observation, dtype=np.float32)
                for batch in compact_batches
            ],
            axis=0,
        ),
        "legal_action_mask": np.stack(
            [
                np.asarray(batch.action_mask, dtype=np.bool_)
                for batch in compact_batches
            ],
            axis=0,
        ),
        "lightzero_action_mask": np.stack(
            [
                np.asarray(batch.action_mask, dtype=np.bool_)
                for batch in compact_batches
            ],
            axis=0,
        ),
        "live_mask": np.stack(
            [
                np.asarray(batch.active_root_mask, dtype=np.bool_).reshape(
                    np.asarray(batch.reward).shape[:2]
                )
                for batch in compact_batches
            ],
            axis=0,
        ),
        "joint_action": np.stack(
            [
                np.asarray(batch.joint_action, dtype=np.int16)
                for batch in compact_batches
            ],
            axis=0,
        ),
        "reward": np.stack(
            [np.asarray(batch.reward, dtype=np.float32) for batch in compact_batches],
            axis=0,
        ),
        "done": np.stack(
            [np.asarray(batch.done, dtype=np.bool_) for batch in compact_batches],
            axis=0,
        ),
        "terminated": np.stack(
            [
                np.asarray(batch.terminated, dtype=np.bool_)
                for batch in compact_batches
            ],
            axis=0,
        ),
        "truncated": np.stack(
            [
                np.asarray(batch.truncated, dtype=np.bool_)
                for batch in compact_batches
            ],
            axis=0,
        ),
        "final_observation": np.stack(
            [
                np.asarray(
                    batch.final_observation
                    if batch.final_observation is not None
                    else batch.observation,
                    dtype=np.float32,
                )
                for batch in compact_batches
            ],
            axis=0,
        ),
        "final_observation_row_mask": np.stack(
            [
                np.asarray(batch.final_observation_row_mask, dtype=np.bool_)
                for batch in compact_batches
            ],
            axis=0,
        ),
        "final_reward_map": np.stack(
            [
                np.asarray(batch.final_reward_map, dtype=np.float32)
                for batch in compact_batches
            ],
            axis=0,
        ),
    }
    policy_rows = tuple(
        _policy_rows_from_compact_batch(batch) for batch in compact_batches
    )
    records = tuple(
        {
            "sequence_index": int(record_index),
            "policy_row_count": int(policy_rows[record_index]["policy_env_row"].size),
            "done_rows": np.flatnonzero(arrays["done"][record_index]).tolist(),
            "final_observation_rows": np.flatnonzero(
                arrays["final_observation_row_mask"][record_index]
            ).tolist(),
        }
        for record_index in range(len(compact_batches))
    )
    return SourceStateMultiplayerTrainerReplayChunkV0(
        metadata={
            "producer": "fixed_action_tape_closed_search_feedback_slab",
            "source_fidelity_claim": "local_toy_closed_loop_slab_replay_sample",
            "observation_materialized_for_sample_edge_only": True,
        },
        arrays=arrays,
        policy_rows=policy_rows,
        records=records,
    )


def _policy_rows_from_compact_batch(batch: HybridCompactBatch) -> dict[str, np.ndarray]:
    reward = np.asarray(batch.reward)
    batch_size, player_count = int(reward.shape[0]), int(reward.shape[1])
    active = np.asarray(batch.active_root_mask, dtype=np.bool_).reshape(
        batch_size,
        player_count,
    )
    env_row, player = np.nonzero(active)
    env_row = env_row.astype(np.int32, copy=False)
    player = player.astype(np.int16, copy=False)
    return {
        "policy_observation": np.asarray(batch.observation, dtype=np.float32)[
            env_row,
            player,
        ].copy(),
        "policy_action_mask": np.asarray(batch.action_mask, dtype=np.bool_)[
            env_row,
            player,
        ].copy(),
        "policy_env_row": env_row.copy(),
        "policy_player": player.copy(),
    }


def _validate_search_action_step_identity(root_batch: Any, action_step: Any) -> dict[str, bool]:
    root_index = np.asarray(action_step.root_index, dtype=np.int64).reshape(-1)
    active_root_index = np.flatnonzero(root_batch.active_root_mask).astype(
        np.int64,
        copy=False,
    )
    root_index_matches = bool(np.array_equal(root_index, active_root_index))
    env_row_matches = bool(np.array_equal(action_step.env_row, root_batch.env_row[root_index]))
    player_matches = bool(np.array_equal(action_step.player, root_batch.player[root_index]))
    policy_env_id_matches = bool(
        np.array_equal(action_step.policy_env_id, root_batch.policy_env_id[root_index])
    )
    selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
    selected_shape_matches = bool(selected.shape == root_index.shape)
    selected_legal = True
    if selected.size:
        selected_index = selected.astype(np.int64, copy=False)
        selected_legal = bool(root_batch.legal_mask[root_index, selected_index].all())
    checked = bool(
        root_index_matches
        and env_row_matches
        and player_matches
        and policy_env_id_matches
        and selected_shape_matches
        and selected_legal
    )
    if not checked:
        raise RuntimeError("search action step identity does not match root batch")
    return {
        "action_step_identity_checked": checked,
        "action_step_root_index_matches_active": root_index_matches,
        "action_step_env_row_matches_root": env_row_matches,
        "action_step_player_matches_root": player_matches,
        "action_step_policy_env_id_matches_root": policy_env_id_matches,
        "selected_action_shape_matches": selected_shape_matches,
        "selected_action_legal": selected_legal,
    }


def _validate_search_action_step_digests(action_step: Any) -> dict[str, bool]:
    metadata = action_step.metadata
    expected_replay_digest = compact_search_deferred_replay_payload_digest_v1(
        action_step.replay_payload_handle
    )
    expected_selected_digest = compact_search_array_digest_v1(
        np.asarray(action_step.selected_action, dtype=np.int16)
    )
    replay_deferred = bool(metadata.get("search_replay_payload_digest_deferred") is True)
    replay_digest_matches = bool(
        metadata.get("search_replay_payload_digest") == expected_replay_digest
    )
    selected_digest_matches = bool(
        metadata.get("selected_action_digest") == expected_selected_digest
    )
    if not (replay_deferred and replay_digest_matches and selected_digest_matches):
        raise RuntimeError("search action step digest proof failed")
    return {
        "replay_payload_digest_deferred": replay_deferred,
        "replay_payload_digest_matches_handle": replay_digest_matches,
        "selected_action_digest_matches_payload": selected_digest_matches,
    }


def _validate_search_root_batch_identity(
    root_batch: Any,
    *,
    observation_tracker: _ObservationStackTracker,
    reward: np.ndarray,
    batch_size: int,
    player_count: int,
) -> dict[str, Any]:
    batch_size = int(batch_size)
    player_count = int(player_count)
    root_count = batch_size * player_count
    expected_env_row = np.repeat(np.arange(batch_size, dtype=np.int32), player_count)
    expected_player = np.tile(np.arange(player_count, dtype=np.int16), batch_size)
    row_major_checked = bool(
        np.array_equal(root_batch.env_row, expected_env_row)
        and np.array_equal(root_batch.player, expected_player)
    )
    done_root = np.asarray(root_batch.done_root, dtype=np.bool_).reshape(
        batch_size,
        player_count,
    )
    done_root_matches = bool((done_root == done_root[:, :1]).all())
    legal_mask = np.asarray(root_batch.legal_mask, dtype=np.bool_)
    active_expected = np.logical_and(
        ~np.asarray(root_batch.done_root, dtype=np.bool_).reshape(-1),
        legal_mask.any(axis=1),
    )
    active_matches = bool(
        np.array_equal(root_batch.active_root_mask.reshape(-1), active_expected)
    )
    to_play_default = bool((np.asarray(root_batch.to_play) == DEFAULT_TO_PLAY).all())
    target_reward_matches = bool(
        np.allclose(
            np.asarray(root_batch.target_reward, dtype=np.float32).reshape(root_count),
            np.asarray(reward, dtype=np.float32).reshape(root_count),
            atol=1e-6,
        )
    )
    observation_shape = [int(dim) for dim in np.asarray(root_batch.observation).shape]
    observation_dtype = str(np.asarray(root_batch.observation).dtype)
    observation_shares_stack = bool(
        np.shares_memory(root_batch.observation, observation_tracker.stack)
    )
    observation_copied = bool(root_batch.metadata.get("observation_copied"))
    checked = bool(
        row_major_checked
        and done_root_matches
        and active_matches
        and to_play_default
        and target_reward_matches
        and observation_shape == [root_count, *[int(dim) for dim in POLICY_STACK_SHAPE]]
        and observation_dtype == "uint8"
        and observation_shares_stack
        and not observation_copied
    )
    if not checked:
        raise RuntimeError("search root batch identity proof failed")
    return {
        "root_batch_observation_source": str(root_batch.metadata["observation_source"]),
        "root_batch_observation_copied": observation_copied,
        "root_batch_observation_shape": observation_shape,
        "root_batch_observation_dtype": observation_dtype,
        "root_batch_row_major_sidecars_checked": row_major_checked,
        "done_root_matches_repeat_done": done_root_matches,
        "active_root_mask_matches_non_done_legal": active_matches,
        "to_play_all_default": to_play_default,
        "target_reward_matches_reward": target_reward_matches,
        "root_observation_shares_stack": observation_shares_stack,
    }


def _empty_search_metadata() -> dict[str, Any]:
    empty_digest = hashlib.blake2b(digest_size=16).hexdigest()
    return {
        "search_enabled": False,
        "search_impl": "none",
        "search_schema_id": "none",
        "root_batch_schema_id": "none",
        "search_call_count": 0,
        "search_root_count": 0,
        "search_active_root_count": 0,
        "search_inactive_root_count": 0,
        "search_selected_action_count": 0,
        "search_max_active_root_count": 0,
        "search_action_count": 0,
        "search_num_simulations": 0,
        "search_first_legal_policy": False,
        "search_two_phase_action_only": False,
        "search_ctree_calls": 0,
        "search_tolist_calls": 0,
        "search_per_sim_d2h_bytes": 0,
        "search_root_observation_copy_bytes": 0,
        "search_action_d2h_bytes": 0,
        "search_deferred_replay_payload_d2h_bytes": 0,
        "search_preallocated_buffer_bytes": 0,
        "search_buffer_reused": False,
        "search_action_step_identity_checked": False,
        "search_action_step_root_index_matches_active": False,
        "search_action_step_env_row_matches_root": False,
        "search_action_step_player_matches_root": False,
        "search_action_step_policy_env_id_matches_root": False,
        "search_selected_action_shape_matches": False,
        "search_selected_action_legal": False,
        "search_replay_payload_digest_deferred": False,
        "search_replay_payload_digest_matches_handle": False,
        "search_selected_action_digest_matches_payload": False,
        "search_root_batch_observation_source": "none",
        "search_root_batch_observation_copied": False,
        "search_root_batch_observation_shape": [],
        "search_root_batch_observation_dtype": "none",
        "search_root_batch_row_major_sidecars_checked": False,
        "search_done_root_matches_repeat_done": False,
        "search_active_root_mask_matches_non_done_legal": False,
        "search_to_play_all_default": False,
        "search_target_reward_matches_reward": False,
        "search_root_observation_shares_stack": False,
        "search_selected_action_digest": "",
        "search_replay_payload_digest": "",
        "search_root_batch_checksum": empty_digest,
        "search_action_step_checksum": empty_digest,
        "search_root_observation_checksum": empty_digest,
        "search_selected_action_checksum": empty_digest,
        "search_joint_action_checksum": empty_digest,
    }


def _build_hybrid_compact_batch(
    *,
    observation_tracker: _ObservationStackTracker,
    env: VectorMultiplayerEnv,
    result: Any,
    joint_action: np.ndarray,
    step_index: int,
    tape_index: int,
    mode: str,
) -> Any:
    batch_size = int(observation_tracker.batch_size)
    player_count = int(observation_tracker.player_count)
    root_count = batch_size * player_count
    reward = np.asarray(_result_array(result, "reward"), dtype=np.float32)
    done = np.asarray(_result_array(result, "done"), dtype=np.bool_)
    terminated = np.asarray(_result_array(result, "terminated"), dtype=np.bool_)
    truncated = np.asarray(_result_array(result, "truncated"), dtype=np.bool_)
    terminal_reason = np.asarray(_result_array(result, "terminal_reason"), dtype=np.int16)
    action_mask = np.asarray(_result_array(result, "action_mask"), dtype=np.bool_)
    terminal_rows = np.asarray(_result_array(result, "terminal_rows"), dtype=np.int32)
    done_root = np.repeat(done, player_count)
    active_root_mask = np.logical_and(
        ~done_root,
        action_mask.reshape(root_count, action_mask.shape[-1]).any(axis=1),
    )
    final_observation = observation_tracker.stack.copy() if bool(done.any()) else None
    compact_batch = HybridCompactBatch(
        observation=observation_tracker.stack,
        action_mask=action_mask,
        reward=reward,
        final_reward_map=np.asarray(_result_array(result, "final_reward_map"), dtype=np.float32),
        done=done,
        terminated=terminated,
        truncated=truncated,
        terminal_reason=terminal_reason,
        death_count=np.asarray(_result_array(result, "death_count"), dtype=np.int32),
        death_player=np.asarray(_result_array(result, "death_player"), dtype=np.int16),
        death_cause=np.asarray(_result_array(result, "death_cause"), dtype=np.int16),
        death_hit_owner=np.asarray(_result_array(result, "death_hit_owner"), dtype=np.int16),
        winner=np.asarray(_result_array(result, "winner"), dtype=np.int16),
        draw=np.asarray(_result_array(result, "draw"), dtype=np.bool_),
        policy_env_id=(
            np.arange(root_count, dtype=np.int64) + np.int64(int(step_index) * root_count)
        ),
        policy_env_row=np.repeat(np.arange(batch_size, dtype=np.int32), player_count),
        policy_player=np.tile(np.arange(player_count, dtype=np.int32), batch_size),
        target_reward=reward.reshape(root_count, 1),
        done_root=done_root,
        to_play=np.full(root_count, DEFAULT_TO_PLAY, dtype=np.int64),
        active_root_mask=active_root_mask,
        final_observation=final_observation,
        final_observation_row_mask=done.copy() if final_observation is not None else np.zeros(
            batch_size,
            dtype=np.bool_,
        ),
        terminal_row_mask=_row_mask_from_indices(terminal_rows, batch_size=batch_size),
        autoreset_row_mask=np.zeros(batch_size, dtype=np.bool_),
        terminal_global_rows=terminal_rows,
        autoreset_global_rows=np.zeros(0, dtype=np.int32),
        episode_step=np.asarray(env.state["episode_step"], dtype=np.int32).copy(),
        elapsed_ms=np.asarray(
            _result_array(result, "source_physics_elapsed_ms"),
            dtype=np.float64,
        ).copy(),
        round_id=np.asarray(
            env.state.get("round_id", np.zeros(batch_size, dtype=np.int32)),
            dtype=np.int32,
        ).copy(),
        alive=np.asarray(env.state["alive"][:, :player_count], dtype=np.bool_).copy(),
        joint_action=np.asarray(joint_action, dtype=np.int16).copy(),
    )
    return compact_batch


def _replay_step_from_compact_batch(compact_batch: HybridCompactBatch) -> Any:
    return SimpleNamespace(
        observation=np.asarray(compact_batch.observation).copy(),
        action_mask=np.asarray(compact_batch.action_mask, dtype=np.bool_).copy(),
        reward=np.asarray(compact_batch.reward, dtype=np.float32).copy(),
        final_reward_map=np.asarray(compact_batch.final_reward_map, dtype=np.float32).copy(),
        done=np.asarray(compact_batch.done, dtype=np.bool_).copy(),
        terminated=np.asarray(compact_batch.terminated, dtype=np.bool_).copy(),
        truncated=np.asarray(compact_batch.truncated, dtype=np.bool_).copy(),
        payload={"joint_action": np.asarray(compact_batch.joint_action, dtype=np.int16).copy()},
        compact_batch=compact_batch,
    )


def _build_search_root_batch(
    *,
    observation_tracker: _ObservationStackTracker,
    env: VectorMultiplayerEnv,
    result: Any,
    joint_action: np.ndarray,
    step_index: int,
    tape_index: int,
    mode: str,
) -> Any:
    compact_batch = _build_hybrid_compact_batch(
        observation_tracker=observation_tracker,
        env=env,
        result=result,
        joint_action=joint_action,
        step_index=int(step_index),
        tape_index=int(tape_index),
        mode=str(mode),
    )
    return build_compact_root_batch_v1(
        compact_batch,
        search_lane="fixed_action_tape_fixed_shape_search",
        metadata={
            "fixed_action_tape_search_gate": True,
            "fixed_action_tape_step_index": int(step_index),
            "fixed_action_tape_tape_index": int(tape_index),
            "fixed_action_tape_loop_mode": str(mode),
        },
        copy_observation=False,
    )


def _row_mask_from_indices(rows: np.ndarray, *, batch_size: int) -> np.ndarray:
    mask = np.zeros(int(batch_size), dtype=np.bool_)
    row_values = np.asarray(rows, dtype=np.int32).reshape(-1)
    if row_values.size:
        if bool((row_values < 0).any()) or bool((row_values >= int(batch_size)).any()):
            raise RuntimeError(f"row ids out of range for batch_size={batch_size}: {row_values}")
        mask[row_values] = True
    return mask


def _accumulate_root_batch_checksum(hasher: Any, root_batch: Any) -> None:
    for name in (
        "observation",
        "legal_mask",
        "active_root_mask",
        "to_play",
        "env_row",
        "player",
        "policy_env_id",
        "target_reward",
        "done_root",
        "final_observation_row_mask",
        "terminal_row_mask",
        "autoreset_row_mask",
    ):
        _hash_update_array(hasher, f"root_batch.{name}", np.asarray(getattr(root_batch, name)))
    has_final_observation = root_batch.final_observation is not None
    _hash_update_array(
        hasher,
        "root_batch.has_final_observation",
        np.asarray([has_final_observation], dtype=np.bool_),
    )
    if has_final_observation:
        _hash_update_array(
            hasher,
            "root_batch.final_observation",
            np.asarray(root_batch.final_observation),
        )
    for key in (
        "schema_id",
        "search_lane",
        "observation_source",
        "batch_size",
        "player_count",
        "root_count",
        "active_root_count",
        "fixed_opponent_to_play",
        "env_row_player_semantics",
    ):
        _hash_update_text(hasher, f"root_batch.metadata.{key}", root_batch.metadata[key])


def _accumulate_action_step_checksum(hasher: Any, action_step: Any) -> None:
    for name in ("root_index", "env_row", "player", "policy_env_id", "selected_action"):
        _hash_update_array(hasher, f"action_step.{name}", np.asarray(getattr(action_step, name)))
    for key in (
        "schema_id",
        "phase",
        "search_impl",
        "num_simulations",
        "active_root_count",
        "selected_action_digest",
        "search_replay_payload_digest",
        "search_replay_payload_digest_deferred",
    ):
        _hash_update_text(hasher, f"action_step.metadata.{key}", action_step.metadata[key])


def _fixed_buffer_direct_step(
    env: VectorMultiplayerEnv,
    actions: np.ndarray,
    *,
    timer_advance_ms: float | np.ndarray | None,
    collect_collision_diagnostics: bool = False,
) -> tuple[dict[str, Any], dict[str, float]]:
    timers: dict[str, float] = {}

    started = time.perf_counter()
    env._require_reset()
    if bool(env._needs_reset.any()):
        rows = np.flatnonzero(env._needs_reset).astype(np.int32)
        raise RuntimeError(
            "reset must be called before stepping rows that ended; "
            f"pending rows={rows.tolist()}"
        )
    warmdown_pending = env._warmdown_pending_mask()
    if bool(warmdown_pending.any()):
        rows = np.flatnonzero(warmdown_pending).astype(np.int32)
        raise RuntimeError(
            "advance_warmdown must be called before stepping rows between rounds; "
            f"pending rows={rows.tolist()}"
        )
    pre_alive = env.state["alive"][:, : env.player_count].copy()
    pre_death_count = env.state["death_count"].copy()
    pre_active = ~env.state["done"].copy()
    env.state["bonus_catch_count_step"][:, : env.player_count] = 0
    source_moves, _action_sidecar = env._source_moves_and_action_sidecar(
        actions,
        pre_alive=pre_alive,
    )
    timer_advance = vector_env_mod._step_timer_advance_ms(
        timer_advance_ms,
        batch_size=env.batch_size,
    )
    disabled_mask = env._disabled_player_mask(None)
    env._ensure_seed_generated_random_tape_headroom(
        pre_active,
        min_available=max(16, env.player_count * 4 * max(1, env.decision_source_frames or 1)),
    )
    _add_timer(timers, "prepare_sec", started)

    started = time.perf_counter()
    runtime_phase_timers: dict[str, float] = {}
    runtime_result = env._advance_runtime_for_public_step(
        pre_active=pre_active,
        source_moves=source_moves,
        timer_advance=timer_advance,
        disabled_player_mask=disabled_mask,
        collect_collision_diagnostics=bool(collect_collision_diagnostics),
        runtime_phase_timers=runtime_phase_timers,
    )
    runtime_sec = time.perf_counter() - started
    timers["runtime_sec"] = timers.get("runtime_sec", 0.0) + runtime_sec
    for key, value in runtime_phase_timers.items():
        timers[f"runtime_phase_{key}"] = timers.get(f"runtime_phase_{key}", 0.0) + float(value)

    started = time.perf_counter()
    env._correct_leave_adjusted_death_scores(
        pre_alive=pre_alive,
        pre_death_count=pre_death_count,
    )
    env._append_new_deaths(pre_alive)
    env.state["episode_step"][pre_active] += 1
    env._mark_overflow_truncations(pre_active)
    env._mark_timeout_truncations(pre_active)

    transition_mask = env.state["done"].copy()
    round_transition_mask = transition_mask & env.state["terminated"] & ~env.state["truncated"]
    if bool(round_transition_mask.any()):
        if env.episode_end_mode == vector_env_mod.EPISODE_END_MODE_MATCH:
            env._stage_match_mode_warmdown_rows(round_transition_mask)
        else:
            env._mark_public_round_warmdown_rows(round_transition_mask)
    public_terminal_mask = env.state["done"].copy()
    if bool(public_terminal_mask.any()):
        env.state["in_round"][public_terminal_mask] = False
        env._needs_reset |= public_terminal_mask

    terminal_reason = env.state["terminal_reason"].copy()
    death_count = env.state["death_count"].copy()
    death_player = env.state["death_player"].copy()
    death_cause = env.state["death_cause"].copy()
    death_hit_owner = env.state["death_hit_owner"].copy()
    winner = env.state["winner"].copy()
    draw = env.state["draw"].copy()
    done = env.state["done"].copy()
    terminated = env.state["terminated"].copy()
    truncated = env.state["truncated"].copy()
    _add_timer(timers, "post_runtime_bookkeeping_sec", started)

    started = time.perf_counter()
    reward = env._reward()
    _add_timer(timers, "reward_sec", started)

    final_reward_map = np.zeros_like(reward, dtype=np.float32)
    terminal_rows = np.flatnonzero(public_terminal_mask).astype(np.int32)
    if bool(public_terminal_mask.any()):
        final_reward_map[public_terminal_mask] = reward[public_terminal_mask]

    started = time.perf_counter()
    action_mask = env._action_mask()
    _add_timer(timers, "compact_action_mask_sec", started)

    result = {
        "reward": np.asarray(reward, dtype=np.float32).copy(),
        "final_reward_map": final_reward_map.copy(),
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "terminal_reason": terminal_reason,
        "death_count": death_count,
        "death_player": death_player,
        "death_cause": death_cause,
        "death_hit_owner": death_hit_owner,
        "winner": winner,
        "draw": draw,
        "action_mask": action_mask.copy(),
        "terminal_rows": terminal_rows.copy(),
        "source_physics_substeps_executed": runtime_result[
            "source_physics_substeps_executed"
        ].copy(),
        "source_physics_elapsed_ms": runtime_result["source_physics_elapsed_ms"].copy(),
    }
    return result, timers


def _empty_proof_accumulator() -> dict[str, Any]:
    accumulator = {
        name: hashlib.blake2b(digest_size=16)
        for name in OUTPUT_CHECKSUM_FIELDS
    }
    accumulator["death"] = hashlib.blake2b(digest_size=16)
    accumulator["observation"] = hashlib.blake2b(digest_size=16)
    return accumulator


def _empty_autoreset_accumulator() -> Any:
    return hashlib.blake2b(digest_size=16)


def _accumulate_result(accumulator: dict[str, Any], result: Any, *, batch_size: int) -> None:
    for name in OUTPUT_CHECKSUM_FIELDS:
        _hash_update_array(accumulator[name], name, _result_array(result, name))
    for name in ("death_count", "death_player", "death_cause", "death_hit_owner"):
        _hash_update_array(accumulator["death"], name, _result_array(result, name))


def _accumulate_zero_observation(accumulator: dict[str, Any], *, batch_size: int) -> None:
    zero_observation = np.zeros((int(batch_size), 1), dtype=np.uint8)
    _hash_update_array(accumulator["observation"], "zero_observation", zero_observation)


def _accumulate_rendered_observation(
    accumulator: dict[str, Any],
    tracker: _ObservationStackTracker,
) -> None:
    _hash_update_array(accumulator["observation"], "observation_stack", tracker.stack)
    _hash_update_array(
        accumulator["observation"],
        "root_observation",
        tracker.root_observation(),
    )
    _hash_update_array(
        accumulator["observation"],
        "row_indices",
        tracker.row_indices,
    )
    _hash_update_array(
        accumulator["observation"],
        "controlled_players",
        tracker.controlled_players,
    )


def _finalize_accumulator(accumulator: dict[str, Any]) -> dict[str, str]:
    return {name: hasher.hexdigest() for name, hasher in accumulator.items()}


def _result_array(result: Any, name: str) -> np.ndarray:
    return np.asarray(result[name] if isinstance(result, dict) else getattr(result, name))


def _empty_trajectory_accumulator() -> dict[str, Any]:
    return {
        "state": hashlib.blake2b(digest_size=16),
        "body": hashlib.blake2b(digest_size=16),
        "death": hashlib.blake2b(digest_size=16),
    }


def _accumulate_trajectory_state(
    accumulator: dict[str, Any],
    state: dict[str, np.ndarray],
) -> None:
    _hash_update_state_fields(
        accumulator["state"],
        "per_step_full_state",
        state,
        _state_field_names(state),
    )
    _hash_update_state_fields(accumulator["body"], "per_step_body", state, BODY_CHECKSUM_FIELDS)
    _hash_update_state_fields(accumulator["death"], "per_step_death", state, DEATH_CHECKSUM_FIELDS)


def _new_death_evidence(
    *,
    pre_death_count: np.ndarray,
    post_death_count: np.ndarray,
    death_cause: np.ndarray,
) -> tuple[int, tuple[str, ...]]:
    pre_count = np.asarray(pre_death_count, dtype=np.int64).reshape(-1)
    post_count = np.asarray(post_death_count, dtype=np.int64).reshape(-1)
    causes = np.asarray(death_cause)
    total = 0
    names: set[str] = set()
    for row, (before, after) in enumerate(zip(pre_count, post_count, strict=True)):
        start = max(0, int(before))
        stop = min(max(start, int(after)), int(causes.shape[1]))
        if stop <= start:
            continue
        total += stop - start
        cause_names = vector_runtime.death_cause_name_array(causes[row : row + 1, start:stop])
        names.update(str(name) for name in cause_names.reshape(-1) if name != "none")
    return int(total), tuple(sorted(names))


def _loop(
    *,
    label: str,
    env: VectorMultiplayerEnv,
    tape: ActionTape,
    config: BenchmarkConfig,
    mode: str,
) -> dict[str, Any]:
    total_steps = int(config.warmup_steps) + int(config.measured_steps)
    step_wall_sec: list[float] = []
    whole_loop_wall_sec: list[float] = []
    action_source_wall_sec: list[float] = []
    observation_wall_sec: list[float] = []
    search_wall_sec: list[float] = []
    slab_replay_wall_sec: list[float] = []
    owner_slot_wall_sec: list[float] = []
    autoreset_wall_sec: list[float] = []
    child_timer_values: dict[str, list[float]] = {}
    output_accumulator = _empty_proof_accumulator()
    trajectory_accumulator = _empty_trajectory_accumulator()
    autoreset_accumulator = _empty_autoreset_accumulator()
    observation_tracker = (
        _ObservationStackTracker(
            batch_size=int(config.batch_size),
            player_count=int(tape.player_count),
        )
        if bool(config.render_observation)
        else None
    )
    search_tracker = (
        _SearchRootTracker(
            batch_size=int(config.batch_size),
            player_count=int(tape.player_count),
        )
        if bool(config.run_search)
        else None
    )
    slab_tracker = (
        _SlabReplayTracker(
            batch_size=int(config.batch_size),
            player_count=int(tape.player_count),
            sample_seed=int(config.seed),
        )
        if bool(config.run_slab_replay)
        else None
    )
    owner_slot_tracker = (
        _OwnerSlotCeilingTracker(
            batch_size=int(config.batch_size),
            player_count=int(tape.player_count),
        )
        if bool(config.run_owner_slot_ceiling)
        else None
    )
    state_fields = _state_field_names(env.state)
    terminal_row_count = 0
    death_row_count = 0
    new_death_row_count = 0
    measured_new_death_row_count = 0
    done_invariant_violation_count = 0
    autoreset_call_count = 0
    autoreset_row_count = 0
    measured_env_rows = 0
    reward_shape: list[int] | None = None
    done_shape: list[int] | None = None
    action_mask_shape: list[int] | None = None
    death_cause_names: set[str] = set()
    new_death_cause_names: set[str] = set()
    measured_new_death_cause_names: set[str] = set()
    death_transition_step_indices: list[int] = []
    measured_death_transition_step_indices: list[int] = []
    measured_tape_indices: list[int] = []

    for step_index in range(total_steps):
        whole_loop_started = time.perf_counter()
        action_source_started = time.perf_counter()
        tape_index = step_index % len(tape.actions)
        measured = step_index >= int(config.warmup_steps)
        tape_action = _repeat_action(tape.actions[tape_index], config.batch_size)
        staged_action = (
            owner_slot_tracker.action_for_next_step()
            if owner_slot_tracker is not None
            else (
                slab_tracker.action_for_next_step()
                if slab_tracker is not None
                else None
            )
        )
        if staged_action is None:
            action = tape_action
            action_source = "tape_bootstrap"
        elif owner_slot_tracker is not None:
            action = staged_action
            action_source = "owner_slot_search_feedback"
        else:
            action = staged_action
            action_source = "slab_search_feedback"
        if slab_tracker is not None:
            slab_tracker.record_action_source(
                action_source=action_source,
                action=action,
                tape_action=tape_action,
                measured=measured,
                step_index=step_index,
                tape_index=tape_index,
            )
        if owner_slot_tracker is not None:
            owner_slot_tracker.record_action_source(
                action_source=action_source,
                action=action,
                tape_action=tape_action,
                measured=measured,
                step_index=step_index,
                tape_index=tape_index,
            )
        action_source_elapsed_sec = time.perf_counter() - action_source_started
        step_ms = float(tape.step_ms[tape_index])
        timer_advance_ms = float(tape.timer_advance_ms[tape_index])
        _set_step_ms(env, step_ms)

        pre_death_count = env.state["death_count"].copy()
        started = time.perf_counter()
        if mode == "compact_profile":
            profile_timers: dict[str, float] = {}
            result = env.step_compact_profile(
                action,
                timer_advance_ms=timer_advance_ms,
                profile_timers=profile_timers,
                collect_collision_diagnostics=False,
            )
            child_timers = dict(profile_timers)
        elif mode == "fixed_buffer_direct":
            result, child_timers = _fixed_buffer_direct_step(
                env,
                action,
                timer_advance_ms=timer_advance_ms,
                collect_collision_diagnostics=False,
            )
        else:
            raise ValueError(f"unknown loop mode {mode!r}")
        elapsed = time.perf_counter() - started

        terminal_rows = _result_array(result, "terminal_rows")
        death_cause = _result_array(result, "death_cause")
        post_death_count = _result_array(result, "death_count")
        step_new_death_count, step_new_death_names = _new_death_evidence(
            pre_death_count=pre_death_count,
            post_death_count=post_death_count,
            death_cause=death_cause,
        )
        if step_new_death_count:
            new_death_row_count += int(step_new_death_count)
            new_death_cause_names.update(step_new_death_names)
            death_transition_step_indices.append(int(step_index))
        observation_updated = False
        observation_elapsed_sec = 0.0
        search_elapsed_sec = 0.0
        slab_replay_elapsed_sec = 0.0
        owner_slot_elapsed_sec = 0.0
        if observation_tracker is not None and (
            measured or slab_tracker is not None or owner_slot_tracker is not None
        ):
            observation_started = time.perf_counter()
            observation_tracker.update(env.state)
            observation_elapsed_sec += time.perf_counter() - observation_started
            observation_updated = True
        if measured:
            measured_tape_indices.append(int(tape_index))
            step_wall_sec.append(float(elapsed))
            measured_env_rows += int(config.batch_size)
            terminal_row_count += int(np.asarray(terminal_rows).size)
            _accumulate_result(output_accumulator, result, batch_size=int(config.batch_size))
            _accumulate_trajectory_state(trajectory_accumulator, env.state)
            if observation_tracker is None:
                _accumulate_zero_observation(
                    output_accumulator,
                    batch_size=int(config.batch_size),
                )
            else:
                if not observation_updated:
                    observation_started = time.perf_counter()
                    observation_tracker.update(env.state)
                    observation_elapsed_sec += time.perf_counter() - observation_started
                    observation_updated = True
                _accumulate_rendered_observation(output_accumulator, observation_tracker)
            if search_tracker is not None:
                if observation_tracker is None:
                    raise RuntimeError("search root proof requires rendered observation")
                search_started = time.perf_counter()
                search_tracker.run_step(
                    observation_tracker=observation_tracker,
                    env=env,
                    result=result,
                    joint_action=action,
                    step_index=int(step_index),
                    tape_index=int(tape_index),
                    mode=mode,
                )
                search_elapsed_sec += time.perf_counter() - search_started
            reward = _result_array(result, "reward")
            done = _result_array(result, "done")
            terminated = _result_array(result, "terminated")
            truncated = _result_array(result, "truncated")
            action_mask = _result_array(result, "action_mask")
            reward_shape = [int(value) for value in reward.shape]
            done_shape = [int(value) for value in done.shape]
            action_mask_shape = [int(value) for value in action_mask.shape]
            done_invariant_violation_count += int(
                np.count_nonzero(done != (terminated | truncated))
            )
            death_row_count += int(np.count_nonzero(death_cause != vector_runtime.DEATH_CAUSE_NONE))
            names = vector_runtime.death_cause_name_array(death_cause)
            death_cause_names.update(str(name) for name in names.reshape(-1) if name != "none")
            if step_new_death_count:
                measured_new_death_row_count += int(step_new_death_count)
                measured_new_death_cause_names.update(step_new_death_names)
                measured_death_transition_step_indices.append(int(step_index))
            for key, value in child_timers.items():
                child_timer_values.setdefault(key, []).append(float(value))

        if slab_tracker is not None:
            if observation_tracker is None:
                raise RuntimeError("slab replay proof requires rendered observation")
            if not observation_updated:
                observation_started = time.perf_counter()
                observation_tracker.update(env.state)
                observation_elapsed_sec += time.perf_counter() - observation_started
                observation_updated = True
            slab_started = time.perf_counter()
            slab_tracker.run_step(
                observation_tracker=observation_tracker,
                env=env,
                result=result,
                joint_action=action,
                step_index=int(step_index),
                tape_index=int(tape_index),
                mode=mode,
            )
            slab_replay_elapsed_sec += time.perf_counter() - slab_started

        if owner_slot_tracker is not None:
            if observation_tracker is None:
                raise RuntimeError("owner slot ceiling requires rendered observation")
            if not observation_updated:
                observation_started = time.perf_counter()
                observation_tracker.update(env.state)
                observation_elapsed_sec += time.perf_counter() - observation_started
                observation_updated = True
            owner_slot_started = time.perf_counter()
            owner_slot_tracker.run_step(
                observation_tracker=observation_tracker,
                env=env,
                result=result,
                joint_action=action,
                step_index=int(step_index),
                tape_index=int(tape_index),
                mode=mode,
            )
            owner_slot_elapsed_sec += time.perf_counter() - owner_slot_started

        if bool(np.asarray(terminal_rows).size):
            reset_seed = np.arange(int(config.batch_size), dtype=np.uint64)
            reset_seed += np.uint64(int(config.seed) + 10_000 + step_index)
            started = time.perf_counter()
            reset_info = env.autoreset_done_rows_compact_profile(
                seed=reset_seed,
                use_direct_reset=bool(config.use_direct_autoreset),
            )
            reset_elapsed = time.perf_counter() - started
            if measured:
                autoreset_call_count += 1
                autoreset_wall_sec.append(float(reset_elapsed))
                reset_rows = np.asarray(reset_info["reset_rows"], dtype=np.int32)
                autoreset_row_count += int(reset_rows.size)
                _hash_update_array(autoreset_accumulator, "autoreset_rows", reset_rows)

        if measured:
            whole_loop_wall_sec.append(time.perf_counter() - whole_loop_started)
            action_source_wall_sec.append(float(action_source_elapsed_sec))
            observation_wall_sec.append(float(observation_elapsed_sec))
            search_wall_sec.append(float(search_elapsed_sec))
            slab_replay_wall_sec.append(float(slab_replay_elapsed_sec))
            owner_slot_wall_sec.append(float(owner_slot_elapsed_sec))

    child_sums = {key: float(sum(values)) for key, values in child_timer_values.items()}
    named_child_sec = sum(
        value
        for key, value in child_sums.items()
        if key
        in {
            "public_prepare_sec",
            "prepare_sec",
            "runtime_sec",
            "post_runtime_bookkeeping_sec",
            "reward_sec",
            "compact_action_mask_sec",
        }
    )
    step_wall_total = float(sum(step_wall_sec))
    whole_loop_wall_total = float(sum(whole_loop_wall_sec))
    autoreset_total = float(sum(autoreset_wall_sec))
    outer_residual_sec = max(0.0, step_wall_total + autoreset_total - named_child_sec)
    whole_loop_named_surface_sec = (
        named_child_sec
        + float(sum(action_source_wall_sec))
        + float(sum(observation_wall_sec))
        + float(sum(search_wall_sec))
        + float(sum(slab_replay_wall_sec))
        + float(sum(owner_slot_wall_sec))
        + autoreset_total
    )
    whole_loop_outer_residual_sec = max(
        0.0,
        whole_loop_wall_total - whole_loop_named_surface_sec,
    )

    output_checksums = _finalize_accumulator(output_accumulator)
    trajectory_checksums = _finalize_accumulator(trajectory_accumulator)
    observation_metadata = (
        observation_tracker.metadata()
        if observation_tracker is not None
        else {
            "observation_schema_id": "zero_observation_stub",
            "observation_schema_hash": "zero_observation_stub",
            "observation_shape": [int(config.batch_size), 1],
            "observation_dtype": "uint8",
            "latest_frame_shape": [],
            "root_observation_shape": [],
            "render_row_count": 0,
            "render_call_count": 0,
            "observation_zero_checksum": _array_checksum(
                np.zeros((int(config.batch_size), 1), dtype=np.uint8)
            ),
            "observation_nonzero_count": 0,
            "observation_nonzero_checksum_present": False,
            "resident_device_observation_shape": [],
            "resident_root_device_observation_shape": [],
            "resident_row_major_order": True,
            "resident_host_fallback_allowed": False,
            "renderer_backend": "none",
            "renderer_telemetry_sec": {},
        }
    )
    search_metadata = (
        search_tracker.metadata() if search_tracker is not None else _empty_search_metadata()
    )
    replay_metadata = (
        slab_tracker.metadata() if slab_tracker is not None else _empty_replay_metadata()
    )
    owner_slot_metadata = (
        owner_slot_tracker.metadata()
        if owner_slot_tracker is not None
        else _empty_owner_slot_metadata()
    )
    if owner_slot_tracker is not None:
        owner_slot_tracker.close()
    current_state_fields = _state_field_names(env.state)
    if current_state_fields != state_fields:
        raise RuntimeError("env state field set changed during benchmark")
    return {
        "label": label,
        "mode": mode,
        "measured_steps": int(config.measured_steps),
        "measured_env_rows": int(measured_env_rows),
        "terminal_row_count": int(terminal_row_count),
        "death_row_count": int(death_row_count),
        "new_death_row_count": int(new_death_row_count),
        "measured_new_death_row_count": int(measured_new_death_row_count),
        "death_cause_names": sorted(death_cause_names),
        "new_death_cause_names": sorted(new_death_cause_names),
        "measured_new_death_cause_names": sorted(measured_new_death_cause_names),
        "death_transition_step_indices": death_transition_step_indices,
        "measured_death_transition_step_indices": measured_death_transition_step_indices,
        "done_invariant_violation_count": int(done_invariant_violation_count),
        "reward_shape": reward_shape or [],
        "done_shape": done_shape or [],
        "action_mask_shape": action_mask_shape or [],
        "measured_tape_indices": measured_tape_indices,
        "first_measured_tape_index": (
            int(measured_tape_indices[0]) if measured_tape_indices else None
        ),
        "autoreset_call_count": int(autoreset_call_count),
        "autoreset_row_count": int(autoreset_row_count),
        "autoreset_rows_checksum": autoreset_accumulator.hexdigest(),
        "step_wall_sec": _summary(step_wall_sec),
        "whole_loop_wall_sec": _summary(whole_loop_wall_sec),
        "action_source_wall_sec": _summary(action_source_wall_sec),
        "observation_wall_sec": _summary(observation_wall_sec),
        "search_wall_sec": _summary(search_wall_sec),
        "slab_replay_wall_sec": _summary(slab_replay_wall_sec),
        "owner_slot_wall_sec": _summary(owner_slot_wall_sec),
        "autoreset_wall_sec": _summary(autoreset_wall_sec),
        "child_timer_sec": {key: _summary(values) for key, values in child_timer_values.items()},
        "child_timer_sum_sec": child_sums,
        "named_child_sum_sec": float(named_child_sec),
        "outer_residual_sec": float(outer_residual_sec),
        "whole_loop_named_surface_sec": float(whole_loop_named_surface_sec),
        "whole_loop_outer_residual_sec": float(whole_loop_outer_residual_sec),
        "env_rows_per_sec": (
            float(measured_env_rows) / (step_wall_total + autoreset_total)
            if (step_wall_total + autoreset_total) > 0.0
            else 0.0
        ),
        "whole_loop_env_rows_per_sec": (
            float(measured_env_rows) / whole_loop_wall_total
            if whole_loop_wall_total > 0.0
            else 0.0
        ),
        "state_checksum": _state_checksum(env.state, STATE_CHECKSUM_FIELDS),
        "full_state_checksum": _state_checksum_all(env.state),
        "full_state_field_count": len(state_fields),
        "unhashed_state_fields": [],
        "body_checksum": _state_checksum(env.state, BODY_CHECKSUM_FIELDS),
        "death_checksum": _state_checksum(env.state, DEATH_CHECKSUM_FIELDS),
        "per_step_state_checksum": trajectory_checksums["state"],
        "per_step_body_checksum": trajectory_checksums["body"],
        "per_step_death_checksum": trajectory_checksums["death"],
        "needs_reset_checksum": _array_checksum(env._needs_reset),
        "output_compared_fields": list(OUTPUT_CHECKSUM_FIELDS),
        "uncompared_output_fields": [],
        "observation_metadata": observation_metadata,
        "search_metadata": search_metadata,
        "replay_metadata": replay_metadata,
        "owner_slot_metadata": owner_slot_metadata,
        "output_checksums": output_checksums,
    }


def _compare_loops(compact: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "state_checksum",
        "full_state_checksum",
        "body_checksum",
        "death_checksum",
        "per_step_state_checksum",
        "per_step_body_checksum",
        "per_step_death_checksum",
        "autoreset_rows_checksum",
        "needs_reset_checksum",
    )
    field_matches = {
        field: compact[field] == fixed[field]
        for field in fields
    }
    output_matches = {
        name: compact["output_checksums"][name] == fixed["output_checksums"][name]
        for name in compact["output_checksums"]
    }
    scalar_matches = {
        "death_row_count": int(compact["death_row_count"]) == int(fixed["death_row_count"]),
        "new_death_row_count": int(compact["new_death_row_count"])
        == int(fixed["new_death_row_count"]),
        "measured_new_death_row_count": int(compact["measured_new_death_row_count"])
        == int(fixed["measured_new_death_row_count"]),
        "terminal_row_count": int(compact["terminal_row_count"])
        == int(fixed["terminal_row_count"]),
        "autoreset_row_count": int(compact["autoreset_row_count"])
        == int(fixed["autoreset_row_count"]),
        "autoreset_call_count": int(compact["autoreset_call_count"])
        == int(fixed["autoreset_call_count"]),
        "full_state_field_count": int(compact["full_state_field_count"])
        == int(fixed["full_state_field_count"]),
        "unhashed_state_fields": list(compact["unhashed_state_fields"])
        == list(fixed["unhashed_state_fields"])
        == [],
        "uncompared_output_fields": list(compact["uncompared_output_fields"])
        == list(fixed["uncompared_output_fields"])
        == [],
        "done_invariant_violation_count": int(compact["done_invariant_violation_count"]) == 0
        and int(fixed["done_invariant_violation_count"]) == 0,
        "reward_shape": list(compact["reward_shape"]) == list(fixed["reward_shape"]),
        "done_shape": list(compact["done_shape"]) == list(fixed["done_shape"]),
        "action_mask_shape": list(compact["action_mask_shape"])
        == list(fixed["action_mask_shape"]),
        "death_cause_names": list(compact["death_cause_names"])
        == list(fixed["death_cause_names"]),
        "new_death_cause_names": list(compact["new_death_cause_names"])
        == list(fixed["new_death_cause_names"]),
        "measured_new_death_cause_names": list(compact["measured_new_death_cause_names"])
        == list(fixed["measured_new_death_cause_names"]),
        "death_transition_step_indices": list(compact["death_transition_step_indices"])
        == list(fixed["death_transition_step_indices"]),
        "measured_death_transition_step_indices": list(
            compact["measured_death_transition_step_indices"]
        )
        == list(fixed["measured_death_transition_step_indices"]),
        "measured_tape_indices": list(compact["measured_tape_indices"])
        == list(fixed["measured_tape_indices"]),
        "observation_metadata": {
            key: compact["observation_metadata"][key]
            for key in (
                "observation_schema_id",
                "observation_schema_hash",
                "observation_shape",
                "observation_dtype",
                "latest_frame_shape",
                "root_observation_shape",
                "render_row_count",
                "render_call_count",
                "observation_zero_checksum",
                "observation_nonzero_count",
                "observation_nonzero_checksum_present",
                "resident_device_observation_shape",
                "resident_root_device_observation_shape",
                "resident_row_major_order",
                "resident_host_fallback_allowed",
                "renderer_backend",
            )
        }
        == {
            key: fixed["observation_metadata"][key]
            for key in (
                "observation_schema_id",
                "observation_schema_hash",
                "observation_shape",
                "observation_dtype",
                "latest_frame_shape",
                "root_observation_shape",
                "render_row_count",
                "render_call_count",
                "observation_zero_checksum",
                "observation_nonzero_count",
                "observation_nonzero_checksum_present",
                "resident_device_observation_shape",
                "resident_root_device_observation_shape",
                "resident_row_major_order",
                "resident_host_fallback_allowed",
                "renderer_backend",
            )
        },
        "search_metadata": {
            key: compact["search_metadata"][key] for key in SEARCH_METADATA_COMPARE_FIELDS
        }
        == {
            key: fixed["search_metadata"][key] for key in SEARCH_METADATA_COMPARE_FIELDS
        },
        "replay_metadata": {
            key: compact["replay_metadata"][key] for key in REPLAY_METADATA_COMPARE_FIELDS
        }
        == {
            key: fixed["replay_metadata"][key] for key in REPLAY_METADATA_COMPARE_FIELDS
        },
        "owner_slot_metadata": {
            key: compact["owner_slot_metadata"][key]
            for key in OWNER_SLOT_METADATA_COMPARE_FIELDS
        }
        == {
            key: fixed["owner_slot_metadata"][key]
            for key in OWNER_SLOT_METADATA_COMPARE_FIELDS
        },
    }
    fixed_wall = float(fixed["step_wall_sec"]["sum"]) + float(fixed["autoreset_wall_sec"]["sum"])
    compact_wall = float(compact["step_wall_sec"]["sum"]) + float(
        compact["autoreset_wall_sec"]["sum"]
    )
    fixed_whole_loop_wall = _timer_sum(fixed["whole_loop_wall_sec"])
    compact_whole_loop_wall = _timer_sum(compact["whole_loop_wall_sec"])
    return {
        "field_matches": field_matches,
        "output_matches": output_matches,
        "scalar_matches": scalar_matches,
        "passed": bool(
            all(field_matches.values())
            and all(output_matches.values())
            and all(scalar_matches.values())
        ),
        "compact_wall_sec": float(compact_wall),
        "fixed_wall_sec": float(fixed_wall),
        "compact_whole_loop_wall_sec": float(compact_whole_loop_wall),
        "fixed_whole_loop_wall_sec": float(fixed_whole_loop_wall),
        "fixed_vs_compact_speedup": (
            float(compact_wall) / float(fixed_wall) if fixed_wall > 0.0 else 0.0
        ),
        "fixed_vs_compact_whole_loop_speedup": (
            float(compact_whole_loop_wall) / float(fixed_whole_loop_wall)
            if fixed_whole_loop_wall > 0.0
            else 0.0
        ),
        "fixed_outer_residual_minus_named_sec": float(fixed["outer_residual_sec"]),
        "compact_outer_residual_minus_named_sec": float(compact["outer_residual_sec"]),
        "fixed_whole_loop_outer_residual_sec": float(
            fixed["whole_loop_outer_residual_sec"]
        ),
        "compact_whole_loop_outer_residual_sec": float(
            compact["whole_loop_outer_residual_sec"]
        ),
    }


def _owner_buffer_ceiling_block(
    *,
    config: BenchmarkConfig,
    compact: dict[str, Any],
    fixed: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    fixed_total_sec = _timer_sum(fixed["whole_loop_wall_sec"])
    compact_total_sec = _timer_sum(compact["whole_loop_wall_sec"])
    fixed_step_sec = _timer_sum(fixed["step_wall_sec"])
    fixed_action_source_sec = _timer_sum(fixed["action_source_wall_sec"])
    fixed_observation_sec = _timer_sum(fixed["observation_wall_sec"])
    fixed_search_sec = _timer_sum(fixed["search_wall_sec"])
    fixed_slab_replay_sec = _timer_sum(fixed["slab_replay_wall_sec"])
    fixed_owner_slot_sec = _timer_sum(fixed["owner_slot_wall_sec"])
    fixed_autoreset_sec = _timer_sum(fixed["autoreset_wall_sec"])
    fixed_residual_sec = float(fixed.get("whole_loop_outer_residual_sec", 0.0) or 0.0)
    owner_transport_sec = (
        fixed_action_source_sec
        + fixed_observation_sec
        + fixed_search_sec
        + fixed_slab_replay_sec
        + fixed_owner_slot_sec
        + fixed_residual_sec
    )
    mechanics_floor_sec = fixed_step_sec + fixed_autoreset_sec

    def share(value: float) -> float:
        return float(value / fixed_total_sec) if fixed_total_sec > 0.0 else 0.0

    enabled = bool(
        fixed_total_sec > 0.0
        and bool(config.render_observation)
        and (
            bool(config.run_search)
            or bool(config.run_slab_replay)
            or bool(config.run_owner_slot_ceiling)
        )
    )
    transport_removed_speedup = _speedup_if_removed(
        fixed_total_sec,
        owner_transport_sec,
    )
    observation_removed_speedup = _speedup_if_removed(
        fixed_total_sec,
        fixed_observation_sec,
    )
    slab_removed_speedup = _speedup_if_removed(
        fixed_total_sec,
        fixed_slab_replay_sec,
    )
    return {
        "schema_id": OWNER_BUFFER_CEILING_SCHEMA_VERSION,
        "enabled": enabled,
        "profile_scope": "local_fixed_action_tape_whole_loop_not_h100_speed_evidence",
        "production_speed_claim": False,
        "touches_live_training": False,
        "same_work_reference": "compact_profile_same_local_toy_loop",
        "whole_loop_timer_includes": [
            "action_source",
            "mechanics_step",
            "observation_update",
            "search_action_selection",
            "slab_replay_append_sample",
            "owner_slot_root_action",
            "autoreset",
            "outer_residual",
        ],
        "puffer_style_requirement": (
            "fixed owners must keep observation/root/action/replay/sample data "
            "resident and exchange handles, not rebuilt parent Python payloads"
        ),
        "fixed_whole_loop_wall_sec": float(fixed_total_sec),
        "compact_whole_loop_wall_sec": float(compact_total_sec),
        "fixed_vs_compact_whole_loop_speedup": float(
            comparison["fixed_vs_compact_whole_loop_speedup"]
        ),
        "fixed_whole_loop_env_rows_per_sec": float(
            fixed.get("whole_loop_env_rows_per_sec", 0.0) or 0.0
        ),
        "compact_whole_loop_env_rows_per_sec": float(
            compact.get("whole_loop_env_rows_per_sec", 0.0) or 0.0
        ),
        "fixed_mechanics_step_sec": float(fixed_step_sec),
        "fixed_mechanics_step_share": share(fixed_step_sec),
        "fixed_autoreset_sec": float(fixed_autoreset_sec),
        "fixed_autoreset_share": share(fixed_autoreset_sec),
        "fixed_action_source_sec": float(fixed_action_source_sec),
        "fixed_action_source_share": share(fixed_action_source_sec),
        "fixed_observation_sec": float(fixed_observation_sec),
        "fixed_observation_share": share(fixed_observation_sec),
        "fixed_search_sec": float(fixed_search_sec),
        "fixed_search_share": share(fixed_search_sec),
        "fixed_slab_replay_sample_sec": float(fixed_slab_replay_sec),
        "fixed_slab_replay_sample_share": share(fixed_slab_replay_sec),
        "fixed_owner_slot_root_action_sec": float(fixed_owner_slot_sec),
        "fixed_owner_slot_root_action_share": share(fixed_owner_slot_sec),
        "fixed_outer_residual_sec": float(fixed_residual_sec),
        "fixed_outer_residual_share": share(fixed_residual_sec),
        "preserved_mechanics_autoreset_floor_sec": float(mechanics_floor_sec),
        "preserved_mechanics_autoreset_floor_share": share(mechanics_floor_sec),
        "owner_transport_candidate_sec": float(owner_transport_sec),
        "owner_transport_candidate_share": share(owner_transport_sec),
        "needed_removed_fraction_for_2x": 0.5,
        "owner_transport_removed_speedup_ceiling": float(transport_removed_speedup),
        "observation_removed_speedup_ceiling": float(observation_removed_speedup),
        "slab_replay_sample_removed_speedup_ceiling": float(slab_removed_speedup),
        "owner_transport_removal_reaches_2x": bool(
            transport_removed_speedup >= 2.0
        ),
        "interpretation": (
            "A 2x same-work win needs about half the measured wall to disappear "
            "or overlap. If this local owner-transport fraction is below 0.5, "
            "single-surface buffer work cannot justify a 2x claim."
        ),
    }


def _search_metadata_passed(
    *,
    config: BenchmarkConfig,
    tape: ActionTape,
    search_metadata: dict[str, Any],
    search_metadata_match: bool,
) -> bool:
    if not bool(config.run_search):
        return True
    root_count_per_step = int(config.batch_size) * int(tape.player_count)
    expected_root_count = int(config.measured_steps) * root_count_per_step
    selected_count = int(search_metadata["search_selected_action_count"])
    selected_action_bytes = int(np.dtype(np.int16).itemsize)
    replay_payload_bytes = int(
        (2 * ACTION_COUNT + 1) * np.dtype(np.float32).itemsize
    )
    preallocated_bytes = int(
        root_count_per_step * (selected_action_bytes + replay_payload_bytes)
    )
    expected_root_shape = [root_count_per_step, *[int(dim) for dim in POLICY_STACK_SHAPE]]
    required_true_fields = (
        "search_enabled",
        "search_first_legal_policy",
        "search_two_phase_action_only",
        "search_action_step_identity_checked",
        "search_action_step_root_index_matches_active",
        "search_action_step_env_row_matches_root",
        "search_action_step_player_matches_root",
        "search_action_step_policy_env_id_matches_root",
        "search_selected_action_shape_matches",
        "search_selected_action_legal",
        "search_replay_payload_digest_deferred",
        "search_replay_payload_digest_matches_handle",
        "search_selected_action_digest_matches_payload",
        "search_root_batch_row_major_sidecars_checked",
        "search_done_root_matches_repeat_done",
        "search_active_root_mask_matches_non_done_legal",
        "search_to_play_all_default",
        "search_target_reward_matches_reward",
        "search_root_observation_shares_stack",
    )
    return bool(
        search_metadata_match
        and all(bool(search_metadata[field]) for field in required_true_fields)
        and int(search_metadata["search_call_count"]) == int(config.measured_steps)
        and int(search_metadata["search_root_count"]) == expected_root_count
        and int(search_metadata["search_active_root_count"])
        + int(search_metadata["search_inactive_root_count"])
        == int(search_metadata["search_root_count"])
        and selected_count == int(search_metadata["search_active_root_count"])
        and int(search_metadata["search_max_active_root_count"]) <= root_count_per_step
        and int(search_metadata["search_action_count"]) == ACTION_COUNT
        and int(search_metadata["search_num_simulations"]) == 1
        and int(search_metadata["search_ctree_calls"]) == 0
        and int(search_metadata["search_tolist_calls"]) == 0
        and int(search_metadata["search_per_sim_d2h_bytes"]) == 0
        and int(search_metadata["search_root_observation_copy_bytes"]) == 0
        and int(search_metadata["search_action_d2h_bytes"])
        == selected_count * selected_action_bytes
        and int(search_metadata["search_deferred_replay_payload_d2h_bytes"])
        == selected_count * replay_payload_bytes
        and int(search_metadata["search_preallocated_buffer_bytes"]) == preallocated_bytes
        and str(search_metadata["search_root_batch_observation_source"])
        == COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
        and not bool(search_metadata["search_root_batch_observation_copied"])
        and list(search_metadata["search_root_batch_observation_shape"])
        == expected_root_shape
        and str(search_metadata["search_root_batch_observation_dtype"]) == "uint8"
    )


def _replay_metadata_failure_reasons(
    *,
    config: BenchmarkConfig,
    replay_metadata: dict[str, Any],
    replay_metadata_match: bool,
) -> tuple[str, ...]:
    if not bool(config.run_slab_replay):
        return ()
    failure_reasons: list[str] = []

    def require(name: str, condition: bool) -> None:
        if not bool(condition):
            failure_reasons.append(str(name))

    root_count_per_step = int(replay_metadata["slab_root_count"]) // max(
        1,
        int(replay_metadata["slab_step_count"]),
    )
    total_steps = int(config.warmup_steps) + int(config.measured_steps)
    expected_root_count = total_steps * root_count_per_step
    selected_count = int(replay_metadata["slab_selected_action_count"])
    selected_action_bytes = int(np.dtype(np.int16).itemsize)
    replay_payload_bytes = int(
        (2 * ACTION_COUNT + 1) * np.dtype(np.float32).itemsize
    )
    has_active_replay_rows = bool(selected_count > 0 and int(config.measured_steps) > 1)
    sample_required = has_active_replay_rows
    expected_measured_feedback_actions = (
        int(config.measured_steps)
        if int(config.warmup_steps) > 0
        else max(0, int(config.measured_steps) - 1)
    )
    require("replay_metadata_match", bool(replay_metadata_match))
    require("slab_replay_enabled", bool(replay_metadata["slab_replay_enabled"]))
    require(
        "slab_search_feedback_closed_loop",
        bool(replay_metadata["slab_search_feedback_closed_loop"]),
    )
    require(
        "slab_step_count",
        int(replay_metadata["slab_step_count"]) == total_steps,
    )
    require(
        "slab_root_count",
        int(replay_metadata["slab_root_count"]) == expected_root_count,
    )
    require(
        "slab_active_inactive_root_count",
        int(replay_metadata["slab_active_root_count"])
        + int(replay_metadata["slab_inactive_root_count"])
        == int(replay_metadata["slab_root_count"]),
    )
    require(
        "slab_selected_action_count",
        selected_count == int(replay_metadata["slab_active_root_count"]),
    )
    require(
        "slab_max_active_root_count",
        int(replay_metadata["slab_max_active_root_count"]) <= root_count_per_step,
    )
    require("slab_action_count", int(replay_metadata["slab_action_count"]) == ACTION_COUNT)
    require("slab_num_simulations", int(replay_metadata["slab_num_simulations"]) == 1)
    require("slab_ctree_calls", int(replay_metadata["slab_ctree_calls"]) == 0)
    require("slab_tolist_calls", int(replay_metadata["slab_tolist_calls"]) == 0)
    require(
        "slab_per_sim_d2h_bytes",
        int(replay_metadata["slab_per_sim_d2h_bytes"]) == 0,
    )
    require(
        "slab_root_observation_copy_bytes",
        int(replay_metadata["slab_root_observation_copy_bytes"]) == 0,
    )
    require(
        "slab_action_d2h_bytes",
        int(replay_metadata["slab_action_d2h_bytes"])
        == selected_count * selected_action_bytes,
    )
    require(
        "slab_deferred_replay_payload_d2h_bytes",
        int(replay_metadata["slab_deferred_replay_payload_d2h_bytes"])
        == selected_count * replay_payload_bytes,
    )
    require(
        "slab_replay_tape_bootstrap_action_count",
        int(replay_metadata["slab_replay_tape_bootstrap_action_count"]) == 1,
    )
    require(
        "slab_replay_measured_feedback_action_count",
        int(replay_metadata["slab_replay_measured_feedback_action_count"])
        == expected_measured_feedback_actions,
    )
    require(
        "slab_replay_prev_next_joint_action_match_count",
        int(replay_metadata["slab_replay_prev_next_joint_action_match_count"])
        == int(replay_metadata["slab_replay_feedback_action_count"]),
    )
    require(
        "slab_replay_prev_next_joint_action_mismatch_count",
        int(replay_metadata["slab_replay_prev_next_joint_action_mismatch_count"]) == 0,
    )
    require(
        "slab_committed_index_group_count",
        (not has_active_replay_rows)
        or int(replay_metadata["slab_committed_index_group_count"]) > 0,
    )
    require(
        "slab_committed_index_row_count",
        (not has_active_replay_rows)
        or int(replay_metadata["slab_committed_index_row_count"]) > 0,
    )
    require(
        "slab_replay_payload_flush_count",
        (not has_active_replay_rows)
        or int(replay_metadata["slab_replay_payload_flush_count"]) > 0,
    )
    require(
        "slab_replay_index_rows_observation_materialized",
        not bool(replay_metadata["slab_replay_index_rows_observation_materialized"]),
    )
    require(
        "slab_replay_index_rows_next_observation_materialized",
        not bool(
            replay_metadata["slab_replay_index_rows_next_observation_materialized"]
        ),
    )
    require(
        "slab_replay_sample_batch_built",
        (not sample_required)
        or bool(replay_metadata["slab_replay_sample_batch_built"]),
    )
    require(
        "slab_replay_sample_batch_size",
        (not sample_required)
        or int(replay_metadata["slab_replay_sample_batch_size"]) > 0,
    )
    require(
        "slab_replay_action_check_enforced",
        bool(replay_metadata["slab_replay_action_check_enforced"]),
    )
    require(
        "slab_replay_root_observation_copied",
        not bool(replay_metadata["slab_replay_root_observation_copied"]),
    )
    require(
        "slab_replay_pending_uncommitted_count",
        int(replay_metadata["slab_replay_pending_uncommitted_count"]) == 1,
    )
    require(
        "slab_retains_committed_index_rows",
        not bool(replay_metadata["slab_retains_committed_index_rows"]),
    )
    require(
        "replay_ring_stored_index_row_count",
        int(replay_metadata["replay_ring_stored_index_row_count"])
        == int(replay_metadata["slab_committed_index_row_count"]),
    )
    require(
        "replay_append_count",
        (not sample_required) or int(replay_metadata["replay_append_count"]) > 0,
    )
    require(
        "sample_gate_calls",
        (not sample_required) or int(replay_metadata["sample_gate_calls"]) > 0,
    )
    require(
        "sample_row_count",
        sample_required
        or int(replay_metadata["sample_gate_calls"]) == 0
        or int(replay_metadata["sample_row_count"]) > 0,
    )
    return tuple(failure_reasons)


def _replay_metadata_passed(
    *,
    config: BenchmarkConfig,
    replay_metadata: dict[str, Any],
    replay_metadata_match: bool,
) -> bool:
    return not _replay_metadata_failure_reasons(
        config=config,
        replay_metadata=replay_metadata,
        replay_metadata_match=replay_metadata_match,
    )


def _owner_slot_metadata_failure_reasons(
    *,
    config: BenchmarkConfig,
    owner_slot_metadata: dict[str, Any],
    owner_slot_metadata_match: bool,
) -> tuple[str, ...]:
    if not bool(config.run_owner_slot_ceiling):
        return ()
    failure_reasons: list[str] = []

    def require(name: str, condition: bool) -> None:
        if not bool(condition):
            failure_reasons.append(str(name))

    total_steps = int(config.warmup_steps) + int(config.measured_steps)
    expected_measured_feedback_actions = (
        int(config.measured_steps)
        if int(config.warmup_steps) > 0
        else max(0, int(config.measured_steps) - 1)
    )
    has_selected_actions = int(owner_slot_metadata["owner_slot_ceiling_selected_action_count"]) > 0
    expected_replay_slot_appends = (
        max(0, total_steps - 1) if has_selected_actions else 0
    )
    expected_stage_payload_pending = 1 if has_selected_actions else 0
    learner_unroll2_required = bool(has_selected_actions and expected_replay_slot_appends >= 3)
    require("owner_slot_metadata_match", bool(owner_slot_metadata_match))
    require("owner_slot_ceiling_enabled", bool(owner_slot_metadata["owner_slot_ceiling_enabled"]))
    require(
        "owner_slot_ceiling_step_count",
        int(owner_slot_metadata["owner_slot_ceiling_step_count"]) == total_steps,
    )
    require(
        "owner_slot_ceiling_tape_bootstrap_action_count",
        int(owner_slot_metadata["owner_slot_ceiling_tape_bootstrap_action_count"]) == 1,
    )
    require(
        "owner_slot_ceiling_measured_feedback_action_count",
        int(owner_slot_metadata["owner_slot_ceiling_measured_feedback_action_count"])
        == expected_measured_feedback_actions,
    )
    require(
        "owner_slot_ceiling_prev_next_joint_action_match_count",
        int(owner_slot_metadata["owner_slot_ceiling_prev_next_joint_action_match_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_feedback_action_count"]),
    )
    require(
        "owner_slot_ceiling_prev_next_joint_action_mismatch_count",
        int(owner_slot_metadata["owner_slot_ceiling_prev_next_joint_action_mismatch_count"])
        == 0,
    )
    require(
        "owner_slot_ceiling_mechanics_slot_write_count",
        int(owner_slot_metadata["owner_slot_ceiling_mechanics_slot_write_count"])
        == total_steps,
    )
    require(
        "owner_slot_ceiling_mechanics_slot_generation_verified_count",
        int(
            owner_slot_metadata[
                "owner_slot_ceiling_mechanics_slot_generation_verified_count"
            ]
        )
        == total_steps,
    )
    require(
        "owner_slot_ceiling_mechanics_slot_digest_verified_count",
        int(owner_slot_metadata["owner_slot_ceiling_mechanics_slot_digest_verified_count"])
        == total_steps,
    )
    require(
        "owner_slot_ceiling_root_request_from_slot_count",
        int(owner_slot_metadata["owner_slot_ceiling_root_request_from_slot_count"])
        == total_steps,
    )
    require(
        "owner_slot_ceiling_root_request_from_batch_count",
        int(owner_slot_metadata["owner_slot_ceiling_root_request_from_batch_count"]) == 0,
    )
    require(
        "owner_slot_ceiling_hybrid_compact_batch_object_count",
        int(owner_slot_metadata["owner_slot_ceiling_hybrid_compact_batch_object_count"]) == 0,
    )
    require(
        "owner_slot_ceiling_action_result_write_count",
        int(owner_slot_metadata["owner_slot_ceiling_action_result_write_count"]) == total_steps,
    )
    require(
        "owner_slot_ceiling_action_result_read_count",
        int(owner_slot_metadata["owner_slot_ceiling_action_result_read_count"]) == total_steps,
    )
    require(
        "owner_slot_ceiling_next_action_count",
        int(owner_slot_metadata["owner_slot_ceiling_next_action_count"]) == total_steps,
    )
    require(
        "owner_slot_ceiling_root_observation_copy_bytes",
        int(owner_slot_metadata["owner_slot_ceiling_root_observation_copy_bytes"]) == 0,
    )
    require(
        "owner_slot_ceiling_selected_action_count",
        int(owner_slot_metadata["owner_slot_ceiling_selected_action_count"]) >= 0,
    )
    require(
        "owner_slot_ceiling_ctree_calls",
        int(owner_slot_metadata["owner_slot_ceiling_ctree_calls"]) == 0,
    )
    require(
        "owner_slot_ceiling_tolist_calls",
        int(owner_slot_metadata["owner_slot_ceiling_tolist_calls"]) == 0,
    )
    require(
        "owner_slot_ceiling_replay_slot_append_count",
        int(owner_slot_metadata["owner_slot_ceiling_replay_slot_append_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_replay_slot_append_row_count",
        (not has_selected_actions)
        or int(owner_slot_metadata["owner_slot_ceiling_replay_slot_append_row_count"]) > 0,
    )
    require(
        "owner_slot_ceiling_replay_slot_object_entry_count",
        int(owner_slot_metadata["owner_slot_ceiling_replay_slot_object_entry_count"]) == 0,
    )
    require(
        "owner_slot_ceiling_parent_replay_object_count",
        int(owner_slot_metadata["owner_slot_ceiling_parent_replay_object_count"]) == 0,
    )
    require(
        "owner_slot_ceiling_selected_group_object_count",
        int(owner_slot_metadata["owner_slot_ceiling_selected_group_object_count"]) == 0,
    )
    require(
        "owner_slot_ceiling_sample_gate_calls",
        (
            int(owner_slot_metadata["owner_slot_ceiling_sample_gate_calls"]) > 0
            if has_selected_actions
            else int(owner_slot_metadata["owner_slot_ceiling_sample_gate_calls"]) == 0
        ),
    )
    require(
        "owner_slot_ceiling_sample_batch_built",
        bool(owner_slot_metadata["owner_slot_ceiling_sample_batch_built"])
        == bool(has_selected_actions),
    )
    require(
        "owner_slot_ceiling_sample_handle_create_count",
        (
            int(owner_slot_metadata["owner_slot_ceiling_sample_handle_create_count"]) > 0
            if has_selected_actions
            else int(owner_slot_metadata["owner_slot_ceiling_sample_handle_create_count"]) == 0
        ),
    )
    require(
        "owner_slot_ceiling_sample_handle_resolve_count",
        int(owner_slot_metadata["owner_slot_ceiling_sample_handle_resolve_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_sample_handle_create_count"]),
    )
    require(
        "owner_slot_ceiling_sample_handle_inline_resolve_count",
        int(owner_slot_metadata["owner_slot_ceiling_sample_handle_inline_resolve_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_sample_handle_resolve_count"]),
    )
    require(
        "owner_slot_ceiling_sample_handle_pending_count",
        int(owner_slot_metadata["owner_slot_ceiling_sample_handle_pending_count"]) == 0,
    )
    require(
        "owner_slot_ceiling_sample_row_count",
        (
            int(owner_slot_metadata["owner_slot_ceiling_sample_row_count"]) > 0
            if has_selected_actions
            else int(owner_slot_metadata["owner_slot_ceiling_sample_row_count"]) == 0
        ),
    )
    require(
        "owner_slot_ceiling_sample_target_row_count",
        int(owner_slot_metadata["owner_slot_ceiling_sample_target_row_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_sample_row_count"]),
    )
    require(
        "owner_slot_ceiling_stage_replay_transport_entry_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_transport_entry_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_stage_replay_transition_entry_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_transition_entry_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_stage_replay_payload_cache_hit_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_payload_cache_hit_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_stage_replay_payload_cache_miss_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_payload_cache_miss_count"])
        == 0,
    )
    require(
        "owner_slot_ceiling_stage_replay_payload_release_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_payload_release_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_stage_replay_payload_pending_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_payload_pending_count"])
        == expected_stage_payload_pending,
    )
    require(
        "owner_slot_ceiling_stage_replay_pending_record_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_pending_record_count"])
        == 0,
    )
    require(
        "owner_slot_ceiling_stage_replay_ready_record_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_ready_record_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_stage_replay_drained_record_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_drained_record_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_stage_replay_index_rows_build_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_index_rows_build_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_stage_replay_index_rows_row_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_index_rows_row_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_replay_slot_append_row_count"]),
    )
    require(
        "owner_slot_ceiling_stage_replay_device_index_rows_build_count",
        int(
            owner_slot_metadata[
                "owner_slot_ceiling_stage_replay_device_index_rows_build_count"
            ]
        )
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_stage_replay_device_index_rows_row_count",
        int(
            owner_slot_metadata[
                "owner_slot_ceiling_stage_replay_device_index_rows_row_count"
            ]
        )
        == int(owner_slot_metadata["owner_slot_ceiling_stage_replay_index_rows_row_count"]),
    )
    require(
        "owner_slot_ceiling_stage_replay_slot_append_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_slot_append_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_stage_replay_slot_append_row_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_replay_slot_append_row_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_replay_slot_append_row_count"]),
    )
    require(
        "owner_slot_ceiling_stage_sample_batch_built",
        bool(owner_slot_metadata["owner_slot_ceiling_stage_sample_batch_built"])
        == bool(has_selected_actions),
    )
    require(
        "owner_slot_ceiling_stage_sample_gate_calls",
        int(owner_slot_metadata["owner_slot_ceiling_stage_sample_gate_calls"])
        == int(owner_slot_metadata["owner_slot_ceiling_sample_gate_calls"]),
    )
    require(
        "owner_slot_ceiling_stage_sample_handle_create_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_sample_handle_create_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_sample_handle_create_count"]),
    )
    require(
        "owner_slot_ceiling_stage_sample_handle_resolve_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_sample_handle_resolve_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_stage_sample_handle_create_count"]),
    )
    require(
        "owner_slot_ceiling_stage_sample_handle_inline_resolve_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_sample_handle_inline_resolve_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_stage_sample_handle_resolve_count"]),
    )
    require(
        "owner_slot_ceiling_stage_sample_handle_pending_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_sample_handle_pending_count"]) == 0,
    )
    require(
        "owner_slot_ceiling_stage_sample_row_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_sample_row_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_sample_row_count"]),
    )
    require(
        "owner_slot_ceiling_stage_sample_target_row_count",
        int(owner_slot_metadata["owner_slot_ceiling_stage_sample_target_row_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_stage_sample_row_count"]),
    )
    require(
        "owner_slot_ceiling_replay_ring_append_record_count",
        int(owner_slot_metadata["owner_slot_ceiling_replay_ring_append_record_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_replay_ring_append_call_count",
        (
            int(owner_slot_metadata["owner_slot_ceiling_replay_ring_append_call_count"]) > 0
            if has_selected_actions
            else int(owner_slot_metadata["owner_slot_ceiling_replay_ring_append_call_count"]) == 0
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_appended_row_count",
        int(owner_slot_metadata["owner_slot_ceiling_replay_ring_appended_row_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_stage_replay_index_rows_row_count"]),
    )
    require(
        "owner_slot_ceiling_replay_ring_entry_count",
        int(owner_slot_metadata["owner_slot_ceiling_replay_ring_entry_count"])
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_replay_ring_stored_index_row_count",
        int(owner_slot_metadata["owner_slot_ceiling_replay_ring_stored_index_row_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_replay_ring_appended_row_count"]),
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_batch_built",
        bool(owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_batch_built"])
        == bool(has_selected_actions),
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_gate_calls",
        (
            int(owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_gate_calls"]) > 0
            if has_selected_actions
            else int(owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_gate_calls"]) == 0
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_row_count",
        (
            int(owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_row_count"]) > 0
            if has_selected_actions
            else int(owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_row_count"]) == 0
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_target_row_count",
        int(owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_target_row_count"])
        == int(owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_row_count"]),
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_source",
        (
            str(owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_source"])
            == "compact_replay_ring_resident_sample_gate"
            if has_selected_actions
            else str(owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_source"])
            == "none"
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample",
        bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample"
            ]
        )
        == bool(has_selected_actions),
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample_all",
        bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample_all"
            ]
        )
        == bool(has_selected_actions),
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_resident_device_sample_batch",
        bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_resident_device_sample_batch"
            ]
        )
        == bool(has_selected_actions),
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_host_observation_fallback_allowed",
        bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_host_observation_fallback_allowed"
            ]
        )
        is False,
    )
    require(
        "owner_slot_ceiling_replay_ring_sample_observation_provider_used_count",
        int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_observation_provider_used_count"
            ]
        )
        == 0,
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_batch_built",
        bool(owner_slot_metadata["owner_slot_ceiling_replay_ring_learner_unroll2_batch_built"])
        == bool(learner_unroll2_required),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls",
        (
            int(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls"
                ]
            )
            > 0
            if learner_unroll2_required
            else int(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls"
                ]
            )
            == 0
        ),
    )
    learner_unroll2_rows = int(
        owner_slot_metadata[
            "owner_slot_ceiling_replay_ring_learner_unroll2_sample_row_count"
        ]
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_sample_row_count",
        learner_unroll2_rows > 0 if learner_unroll2_required else learner_unroll2_rows == 0,
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_row_count",
        int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_row_count"
            ]
        )
        == learner_unroll2_rows,
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_num_unroll_steps",
        int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_num_unroll_steps"
            ]
        )
        == (2 if learner_unroll2_required else 0),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_require_next_targets",
        bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_require_next_targets"
            ]
        )
        == bool(learner_unroll2_required),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_batch_only",
        bool(owner_slot_metadata["owner_slot_ceiling_replay_ring_learner_unroll2_batch_only"])
        == bool(learner_unroll2_required),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_source",
        (
            str(owner_slot_metadata["owner_slot_ceiling_replay_ring_learner_unroll2_source"])
            == "compact_rollout_slab_resident_device_replay_grouped_learner_batch"
            if learner_unroll2_required
            else str(owner_slot_metadata["owner_slot_ceiling_replay_ring_learner_unroll2_source"])
            == "none"
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source",
        (
            str(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source"
                ]
            )
            != "none"
            if learner_unroll2_required
            else str(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source"
                ]
            )
            == "none"
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count",
        (
            int(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count"
                ]
            )
            >= 0
            if learner_unroll2_required
            else int(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count"
                ]
            )
            == 0
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count",
        (
            int(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count"
                ]
            )
            > 0
            if learner_unroll2_required
            else int(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count"
                ]
            )
            == 0
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_observation_provider_used_count",
        int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_observation_provider_used_count"
            ]
        )
        == 0,
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_schema_id",
        (
            str(owner_slot_metadata["owner_slot_ceiling_replay_ring_learner_unroll2_schema_id"])
            != "none"
            if learner_unroll2_required
            else str(owner_slot_metadata["owner_slot_ceiling_replay_ring_learner_unroll2_schema_id"])
            == "none"
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source",
        (
            str(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source"
                ]
            )
            == "resident_grouped_device_learner_batch_builder_v1"
            if learner_unroll2_required
            else str(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source"
                ]
            )
            == "none"
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_host_fallback_allowed",
        bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_host_fallback_allowed"
            ]
        )
        is False,
    )
    expected_action_shape = [learner_unroll2_rows, 2] if learner_unroll2_required else []
    expected_target_shape = [learner_unroll2_rows, 3, ACTION_COUNT] if learner_unroll2_required else []
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_action_shape",
        list(owner_slot_metadata["owner_slot_ceiling_replay_ring_learner_unroll2_action_shape"])
        == expected_action_shape,
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_shape",
        list(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_shape"
            ]
        )
        == expected_action_shape,
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_shape",
        list(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_shape"
            ]
        )
        == ([learner_unroll2_rows, 3] if learner_unroll2_required else []),
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_shape",
        list(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_shape"
            ]
        )
        == expected_target_shape,
    )
    require(
        "owner_slot_ceiling_replay_ring_learner_unroll2_action_mask_shape",
        list(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_action_mask_shape"
            ]
        )
        == expected_target_shape,
    )
    for checksum_key in (
        "owner_slot_ceiling_replay_ring_learner_unroll2_action_checksum",
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_checksum",
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_checksum",
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_checksum",
        "owner_slot_ceiling_replay_ring_learner_unroll2_source_record_window_checksum",
    ):
        require(
            checksum_key,
            bool(owner_slot_metadata[checksum_key])
            if learner_unroll2_required
            else str(owner_slot_metadata[checksum_key]) == "",
        )
    require(
        "owner_slot_ceiling_replay_ring_columnar_append_call_count",
        (
            float(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_columnar_append_call_count"
                ]
            )
            > 0.0
            if has_selected_actions
            else float(
                owner_slot_metadata[
                    "owner_slot_ceiling_replay_ring_columnar_append_call_count"
                ]
            )
            == 0.0
        ),
    )
    require(
        "owner_slot_ceiling_replay_ring_columnar_append_record_count",
        int(
            round(
                float(
                    owner_slot_metadata[
                        "owner_slot_ceiling_replay_ring_columnar_append_record_count"
                    ]
                )
            )
        )
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_replay_ring_columnar_append_entry_view_object_count",
        int(
            round(
                float(
                    owner_slot_metadata[
                        "owner_slot_ceiling_replay_ring_columnar_append_entry_view_object_count"
                    ]
                )
            )
        )
        == expected_replay_slot_appends,
    )
    require(
        "owner_slot_ceiling_replay_ring_columnar_append_step_view_object_count",
        int(
            round(
                float(
                    owner_slot_metadata[
                        "owner_slot_ceiling_replay_ring_columnar_append_step_view_object_count"
                    ]
                )
            )
        )
        == expected_replay_slot_appends * 2,
    )
    return tuple(failure_reasons)


def _proof_block(
    *,
    config: BenchmarkConfig,
    tape: ActionTape,
    compact: dict[str, Any],
    fixed: dict[str, Any],
    comparison: dict[str, Any],
    action_tape_checksum: str,
) -> dict[str, Any]:
    autoreset_rows = np.asarray(
        [int(compact["autoreset_row_count"]), int(fixed["autoreset_row_count"])],
        dtype=np.int64,
    )
    expected_death_cause_names = list(
        EXPECTED_DEATH_CAUSE_NAMES_BY_SCENARIO.get(tape.scenario_id, ())
    )
    expects_terminal_rows = bool(
        EXPECTED_TERMINAL_ROWS_BY_SCENARIO.get(tape.scenario_id, False)
    )
    expected_death_cause_names_present = all(
        name in compact["new_death_cause_names"] for name in expected_death_cause_names
    )
    expected_measured_death_cause_names_present = all(
        name in compact["measured_new_death_cause_names"]
        for name in expected_death_cause_names
    )
    expected_death_evidence_present = (
        not expected_death_cause_names
        or (
            int(compact["new_death_row_count"]) > 0
            and bool(expected_death_cause_names_present)
        )
    )
    expected_measured_death_evidence_present = (
        not expected_death_cause_names
        or (
            int(compact["measured_new_death_row_count"]) > 0
            and bool(expected_measured_death_cause_names_present)
        )
    )
    terminal_rows_equal_autoreset_rows = int(compact["terminal_row_count"]) == int(
        compact["autoreset_row_count"]
    )
    expected_terminal_autoreset_evidence_present = (
        not expects_terminal_rows
        or (
            int(compact["terminal_row_count"]) > 0
            and int(compact["autoreset_row_count"]) > 0
            and bool(terminal_rows_equal_autoreset_rows)
        )
    )
    search_metadata = dict(compact["search_metadata"])
    search_metadata_match = bool(comparison["scalar_matches"]["search_metadata"])
    search_proof_passed = _search_metadata_passed(
        config=config,
        tape=tape,
        search_metadata=search_metadata,
        search_metadata_match=search_metadata_match,
    )
    replay_metadata = dict(compact["replay_metadata"])
    replay_metadata_match = bool(comparison["scalar_matches"]["replay_metadata"])
    replay_failure_reasons = _replay_metadata_failure_reasons(
        config=config,
        replay_metadata=replay_metadata,
        replay_metadata_match=replay_metadata_match,
    )
    replay_proof_passed = not replay_failure_reasons
    owner_slot_metadata = dict(compact["owner_slot_metadata"])
    owner_slot_metadata_match = bool(comparison["scalar_matches"]["owner_slot_metadata"])
    owner_slot_failure_reasons = _owner_slot_metadata_failure_reasons(
        config=config,
        owner_slot_metadata=owner_slot_metadata,
        owner_slot_metadata_match=owner_slot_metadata_match,
    )
    owner_slot_proof_passed = not owner_slot_failure_reasons
    replay_root_count_per_step = int(replay_metadata["slab_root_count"]) // max(
        1,
        int(replay_metadata["slab_step_count"]),
    )
    replay_expected_total_steps = int(config.warmup_steps) + int(config.measured_steps)
    replay_expected_measured_feedback_action_count = (
        int(config.measured_steps)
        if int(config.warmup_steps) > 0
        else max(0, int(config.measured_steps) - 1)
    )
    replay_expected_append_count = (
        max(0, int(replay_metadata["slab_step_count"]) - 1)
        if int(replay_metadata["slab_selected_action_count"]) > 0
        else 0
    )
    proof_passed = bool(
        comparison["passed"]
        and expected_death_evidence_present
        and (not expects_terminal_rows or expected_measured_death_evidence_present)
        and expected_terminal_autoreset_evidence_present
        and (
            not bool(config.render_observation)
            or bool(compact["observation_metadata"]["observation_nonzero_checksum_present"])
        )
        and search_proof_passed
        and replay_proof_passed
        and owner_slot_proof_passed
        and list(compact["unhashed_state_fields"]) == []
        and list(compact["uncompared_output_fields"]) == []
    )
    observation_metadata = dict(compact["observation_metadata"])
    return {
        "required_pass": bool(config.require_pass),
        "passed": proof_passed,
        "speed_claim_scope": "local_architecture_only_not_h100_speed_evidence",
        "whole_loop_wall_sec_includes_observation_search_replay_sample": bool(
            config.render_observation
            and (
                config.run_search
                or config.run_slab_replay
                or config.run_owner_slot_ceiling
            )
        ),
        "compact_whole_loop_wall_sec": float(
            compact["whole_loop_wall_sec"]["sum"]
        ),
        "fixed_whole_loop_wall_sec": float(fixed["whole_loop_wall_sec"]["sum"]),
        "fixed_vs_compact_whole_loop_speedup": float(
            comparison["fixed_vs_compact_whole_loop_speedup"]
        ),
        "state_checksum": compact["state_checksum"],
        "full_state_checksum": compact["full_state_checksum"],
        "full_state_checksum_match": bool(comparison["field_matches"]["full_state_checksum"]),
        "full_state_field_count": int(compact["full_state_field_count"]),
        "unhashed_state_fields": list(compact["unhashed_state_fields"]),
        "body_checksum": compact["body_checksum"],
        "trajectory_checksum": compact["per_step_state_checksum"],
        "per_step_state_checksum": compact["per_step_state_checksum"],
        "per_step_body_checksum": compact["per_step_body_checksum"],
        "per_step_death_checksum": compact["per_step_death_checksum"],
        "observation_checksum": compact["output_checksums"]["observation"],
        "action_tape_checksum": action_tape_checksum,
        "state_checksum_match": bool(comparison["field_matches"]["state_checksum"]),
        "body_checksum_match": bool(comparison["field_matches"]["body_checksum"]),
        "per_step_state_checksum_match": bool(
            comparison["field_matches"]["per_step_state_checksum"]
        ),
        "per_step_body_checksum_match": bool(
            comparison["field_matches"]["per_step_body_checksum"]
        ),
        "per_step_death_checksum_match": bool(
            comparison["field_matches"]["per_step_death_checksum"]
        ),
        "trajectory_checksum_match": bool(
            comparison["field_matches"]["per_step_state_checksum"]
        ),
        "observation_checksum_match": bool(comparison["output_matches"]["observation"]),
        "observation_schema_id": observation_metadata["observation_schema_id"],
        "observation_schema_hash": observation_metadata["observation_schema_hash"],
        "observation_shape": observation_metadata["observation_shape"],
        "observation_dtype": observation_metadata["observation_dtype"],
        "latest_frame_shape": observation_metadata["latest_frame_shape"],
        "root_observation_shape": observation_metadata["root_observation_shape"],
        "render_row_count": int(observation_metadata["render_row_count"]),
        "render_call_count": int(observation_metadata["render_call_count"]),
        "observation_zero_checksum": observation_metadata["observation_zero_checksum"],
        "observation_nonzero_count": int(observation_metadata["observation_nonzero_count"]),
        "observation_nonzero_checksum_present": bool(
            observation_metadata["observation_nonzero_checksum_present"]
        ),
        "resident_device_observation_shape": observation_metadata[
            "resident_device_observation_shape"
        ],
        "resident_root_device_observation_shape": observation_metadata[
            "resident_root_device_observation_shape"
        ],
        "resident_row_major_order": bool(observation_metadata["resident_row_major_order"]),
        "resident_host_fallback_allowed": bool(
            observation_metadata["resident_host_fallback_allowed"]
        ),
        "renderer_backend": observation_metadata["renderer_backend"],
        "search_enabled": bool(search_metadata["search_enabled"]),
        "search_metadata_match": bool(comparison["scalar_matches"]["search_metadata"]),
        "search_proof_passed": bool(search_proof_passed),
        "slab_replay_enabled": bool(replay_metadata["slab_replay_enabled"]),
        "slab_replay_metadata_match": bool(
            comparison["scalar_matches"]["replay_metadata"]
        ),
        "slab_replay_proof_passed": bool(replay_proof_passed),
        "slab_replay_failure_reasons": list(replay_failure_reasons),
        "owner_slot_ceiling_enabled": bool(owner_slot_metadata["owner_slot_ceiling_enabled"]),
        "owner_slot_ceiling_metadata_match": bool(owner_slot_metadata_match),
        "owner_slot_ceiling_proof_passed": bool(owner_slot_proof_passed),
        "owner_slot_ceiling_failure_reasons": list(owner_slot_failure_reasons),
        "owner_slot_ceiling_step_count": int(owner_slot_metadata["owner_slot_ceiling_step_count"]),
        "owner_slot_ceiling_tape_bootstrap_action_count": int(
            owner_slot_metadata["owner_slot_ceiling_tape_bootstrap_action_count"]
        ),
        "owner_slot_ceiling_feedback_action_count": int(
            owner_slot_metadata["owner_slot_ceiling_feedback_action_count"]
        ),
        "owner_slot_ceiling_measured_feedback_action_count": int(
            owner_slot_metadata["owner_slot_ceiling_measured_feedback_action_count"]
        ),
        "owner_slot_ceiling_prev_next_joint_action_match_count": int(
            owner_slot_metadata["owner_slot_ceiling_prev_next_joint_action_match_count"]
        ),
        "owner_slot_ceiling_prev_next_joint_action_mismatch_count": int(
            owner_slot_metadata["owner_slot_ceiling_prev_next_joint_action_mismatch_count"]
        ),
        "owner_slot_ceiling_mechanics_slot_write_count": int(
            owner_slot_metadata["owner_slot_ceiling_mechanics_slot_write_count"]
        ),
        "owner_slot_ceiling_mechanics_slot_generation_verified_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_mechanics_slot_generation_verified_count"
            ]
        ),
        "owner_slot_ceiling_mechanics_slot_digest_verified_count": int(
            owner_slot_metadata["owner_slot_ceiling_mechanics_slot_digest_verified_count"]
        ),
        "owner_slot_ceiling_root_request_from_slot_count": int(
            owner_slot_metadata["owner_slot_ceiling_root_request_from_slot_count"]
        ),
        "owner_slot_ceiling_root_request_from_batch_count": int(
            owner_slot_metadata["owner_slot_ceiling_root_request_from_batch_count"]
        ),
        "owner_slot_ceiling_hybrid_compact_batch_object_count": int(
            owner_slot_metadata["owner_slot_ceiling_hybrid_compact_batch_object_count"]
        ),
        "owner_slot_ceiling_action_result_write_count": int(
            owner_slot_metadata["owner_slot_ceiling_action_result_write_count"]
        ),
        "owner_slot_ceiling_action_result_read_count": int(
            owner_slot_metadata["owner_slot_ceiling_action_result_read_count"]
        ),
        "owner_slot_ceiling_next_action_count": int(
            owner_slot_metadata["owner_slot_ceiling_next_action_count"]
        ),
        "owner_slot_ceiling_root_observation_copy_bytes": int(
            owner_slot_metadata["owner_slot_ceiling_root_observation_copy_bytes"]
        ),
        "owner_slot_ceiling_active_root_count": int(
            owner_slot_metadata["owner_slot_ceiling_active_root_count"]
        ),
        "owner_slot_ceiling_selected_action_count": int(
            owner_slot_metadata["owner_slot_ceiling_selected_action_count"]
        ),
        "owner_slot_ceiling_ctree_calls": int(
            owner_slot_metadata["owner_slot_ceiling_ctree_calls"]
        ),
        "owner_slot_ceiling_tolist_calls": int(
            owner_slot_metadata["owner_slot_ceiling_tolist_calls"]
        ),
        "owner_slot_ceiling_replay_slot_schema_id": owner_slot_metadata[
            "owner_slot_ceiling_replay_slot_schema_id"
        ],
        "owner_slot_ceiling_replay_slot_capacity": int(
            owner_slot_metadata["owner_slot_ceiling_replay_slot_capacity"]
        ),
        "owner_slot_ceiling_replay_slot_max_rows": int(
            owner_slot_metadata["owner_slot_ceiling_replay_slot_max_rows"]
        ),
        "owner_slot_ceiling_replay_slot_append_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_slot_append_count"]
        ),
        "owner_slot_ceiling_replay_slot_append_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_slot_append_row_count"]
        ),
        "owner_slot_ceiling_replay_slot_object_entry_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_slot_object_entry_count"]
        ),
        "owner_slot_ceiling_parent_replay_object_count": int(
            owner_slot_metadata["owner_slot_ceiling_parent_replay_object_count"]
        ),
        "owner_slot_ceiling_selected_group_object_count": int(
            owner_slot_metadata["owner_slot_ceiling_selected_group_object_count"]
        ),
        "owner_slot_ceiling_sample_handle_schema_id": owner_slot_metadata[
            "owner_slot_ceiling_sample_handle_schema_id"
        ],
        "owner_slot_ceiling_sample_batch_built": bool(
            owner_slot_metadata["owner_slot_ceiling_sample_batch_built"]
        ),
        "owner_slot_ceiling_sample_gate_calls": int(
            owner_slot_metadata["owner_slot_ceiling_sample_gate_calls"]
        ),
        "owner_slot_ceiling_sample_handle_create_count": int(
            owner_slot_metadata["owner_slot_ceiling_sample_handle_create_count"]
        ),
        "owner_slot_ceiling_sample_handle_resolve_count": int(
            owner_slot_metadata["owner_slot_ceiling_sample_handle_resolve_count"]
        ),
        "owner_slot_ceiling_sample_handle_inline_resolve_count": int(
            owner_slot_metadata["owner_slot_ceiling_sample_handle_inline_resolve_count"]
        ),
        "owner_slot_ceiling_sample_handle_pending_count": int(
            owner_slot_metadata["owner_slot_ceiling_sample_handle_pending_count"]
        ),
        "owner_slot_ceiling_sample_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_sample_row_count"]
        ),
        "owner_slot_ceiling_sample_target_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_sample_target_row_count"]
        ),
        "owner_slot_ceiling_action_source_sequence_checksum": owner_slot_metadata[
            "owner_slot_ceiling_action_source_sequence_checksum"
        ],
        "owner_slot_ceiling_slot_digest_checksum": owner_slot_metadata[
            "owner_slot_ceiling_slot_digest_checksum"
        ],
        "owner_slot_ceiling_next_joint_action_checksum": owner_slot_metadata[
            "owner_slot_ceiling_next_joint_action_checksum"
        ],
        "owner_slot_ceiling_replay_slot_window_checksum": owner_slot_metadata[
            "owner_slot_ceiling_replay_slot_window_checksum"
        ],
        "owner_slot_ceiling_sample_handle_checksum": owner_slot_metadata[
            "owner_slot_ceiling_sample_handle_checksum"
        ],
        "owner_slot_ceiling_sample_row_id_checksum": owner_slot_metadata[
            "owner_slot_ceiling_sample_row_id_checksum"
        ],
        "owner_slot_ceiling_sample_action_checksum": owner_slot_metadata[
            "owner_slot_ceiling_sample_action_checksum"
        ],
        "owner_slot_ceiling_sample_reward_checksum": owner_slot_metadata[
            "owner_slot_ceiling_sample_reward_checksum"
        ],
        "owner_slot_ceiling_sample_done_checksum": owner_slot_metadata[
            "owner_slot_ceiling_sample_done_checksum"
        ],
        "owner_slot_ceiling_stage_replay_schema_id": owner_slot_metadata[
            "owner_slot_ceiling_stage_replay_schema_id"
        ],
        "owner_slot_ceiling_stage_replay_transport_entry_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_transport_entry_count"]
        ),
        "owner_slot_ceiling_stage_replay_transition_entry_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_transition_entry_count"]
        ),
        "owner_slot_ceiling_stage_replay_payload_cache_hit_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_payload_cache_hit_count"]
        ),
        "owner_slot_ceiling_stage_replay_payload_cache_miss_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_payload_cache_miss_count"]
        ),
        "owner_slot_ceiling_stage_replay_payload_release_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_payload_release_count"]
        ),
        "owner_slot_ceiling_stage_replay_payload_pending_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_payload_pending_count"]
        ),
        "owner_slot_ceiling_stage_replay_pending_record_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_pending_record_count"]
        ),
        "owner_slot_ceiling_stage_replay_ready_record_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_ready_record_count"]
        ),
        "owner_slot_ceiling_stage_replay_drained_record_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_drained_record_count"]
        ),
        "owner_slot_ceiling_stage_replay_index_rows_build_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_index_rows_build_count"]
        ),
        "owner_slot_ceiling_stage_replay_index_rows_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_index_rows_row_count"]
        ),
        "owner_slot_ceiling_stage_replay_device_index_rows_build_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_stage_replay_device_index_rows_build_count"
            ]
        ),
        "owner_slot_ceiling_stage_replay_device_index_rows_row_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_stage_replay_device_index_rows_row_count"
            ]
        ),
        "owner_slot_ceiling_stage_replay_slot_append_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_slot_append_count"]
        ),
        "owner_slot_ceiling_stage_replay_slot_append_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_replay_slot_append_row_count"]
        ),
        "owner_slot_ceiling_stage_sample_batch_built": bool(
            owner_slot_metadata["owner_slot_ceiling_stage_sample_batch_built"]
        ),
        "owner_slot_ceiling_stage_sample_gate_calls": int(
            owner_slot_metadata["owner_slot_ceiling_stage_sample_gate_calls"]
        ),
        "owner_slot_ceiling_stage_sample_handle_create_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_sample_handle_create_count"]
        ),
        "owner_slot_ceiling_stage_sample_handle_resolve_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_sample_handle_resolve_count"]
        ),
        "owner_slot_ceiling_stage_sample_handle_inline_resolve_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_stage_sample_handle_inline_resolve_count"
            ]
        ),
        "owner_slot_ceiling_stage_sample_handle_pending_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_sample_handle_pending_count"]
        ),
        "owner_slot_ceiling_stage_sample_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_sample_row_count"]
        ),
        "owner_slot_ceiling_stage_sample_target_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_stage_sample_target_row_count"]
        ),
        "owner_slot_ceiling_stage_replay_slot_window_checksum": owner_slot_metadata[
            "owner_slot_ceiling_stage_replay_slot_window_checksum"
        ],
        "owner_slot_ceiling_stage_sample_handle_checksum": owner_slot_metadata[
            "owner_slot_ceiling_stage_sample_handle_checksum"
        ],
        "owner_slot_ceiling_stage_sample_row_id_checksum": owner_slot_metadata[
            "owner_slot_ceiling_stage_sample_row_id_checksum"
        ],
        "owner_slot_ceiling_stage_sample_action_checksum": owner_slot_metadata[
            "owner_slot_ceiling_stage_sample_action_checksum"
        ],
        "owner_slot_ceiling_stage_sample_reward_checksum": owner_slot_metadata[
            "owner_slot_ceiling_stage_sample_reward_checksum"
        ],
        "owner_slot_ceiling_stage_sample_done_checksum": owner_slot_metadata[
            "owner_slot_ceiling_stage_sample_done_checksum"
        ],
        "owner_slot_ceiling_replay_ring_append_record_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_append_record_count"]
        ),
        "owner_slot_ceiling_replay_ring_append_call_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_append_call_count"]
        ),
        "owner_slot_ceiling_replay_ring_appended_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_appended_row_count"]
        ),
        "owner_slot_ceiling_replay_ring_entry_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_entry_count"]
        ),
        "owner_slot_ceiling_replay_ring_stored_index_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_stored_index_row_count"]
        ),
        "owner_slot_ceiling_replay_ring_evicted_entry_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_evicted_entry_count"]
        ),
        "owner_slot_ceiling_replay_ring_evicted_index_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_evicted_index_row_count"]
        ),
        "owner_slot_ceiling_replay_ring_sample_batch_built": bool(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_batch_built"]
        ),
        "owner_slot_ceiling_replay_ring_sample_gate_calls": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_gate_calls"]
        ),
        "owner_slot_ceiling_replay_ring_sample_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_row_count"]
        ),
        "owner_slot_ceiling_replay_ring_sample_target_row_count": int(
            owner_slot_metadata["owner_slot_ceiling_replay_ring_sample_target_row_count"]
        ),
        "owner_slot_ceiling_replay_ring_sample_source": owner_slot_metadata[
            "owner_slot_ceiling_replay_ring_sample_source"
        ],
        "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample": bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample"
            ]
        ),
        "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample_all": bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample_all"
            ]
        ),
        "owner_slot_ceiling_replay_ring_sample_resident_device_sample_batch": bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_resident_device_sample_batch"
            ]
        ),
        "owner_slot_ceiling_replay_ring_sample_host_observation_fallback_allowed": bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_host_observation_fallback_allowed"
            ]
        ),
        "owner_slot_ceiling_replay_ring_sample_observation_provider_used_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_observation_provider_used_count"
            ]
        ),
        "owner_slot_ceiling_replay_ring_sample_row_id_checksum": owner_slot_metadata[
            "owner_slot_ceiling_replay_ring_sample_row_id_checksum"
        ],
        "owner_slot_ceiling_replay_ring_sample_action_checksum": owner_slot_metadata[
            "owner_slot_ceiling_replay_ring_sample_action_checksum"
        ],
        "owner_slot_ceiling_replay_ring_sample_reward_checksum": owner_slot_metadata[
            "owner_slot_ceiling_replay_ring_sample_reward_checksum"
        ],
        "owner_slot_ceiling_replay_ring_sample_done_checksum": owner_slot_metadata[
            "owner_slot_ceiling_replay_ring_sample_done_checksum"
        ],
        "owner_slot_ceiling_replay_ring_sample_observation_checksum": owner_slot_metadata[
            "owner_slot_ceiling_replay_ring_sample_observation_checksum"
        ],
        "owner_slot_ceiling_replay_ring_sample_next_observation_checksum": (
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_sample_next_observation_checksum"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_batch_built": bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_batch_built"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls": int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_sample_row_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_sample_row_count"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_row_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_row_count"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_num_unroll_steps": int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_num_unroll_steps"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_require_next_targets": bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_require_next_targets"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_batch_only": bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_batch_only"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_source": owner_slot_metadata[
            "owner_slot_ceiling_replay_ring_learner_unroll2_source"
        ],
        "owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source": (
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_observation_provider_used_count": int(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_observation_provider_used_count"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_schema_id": owner_slot_metadata[
            "owner_slot_ceiling_replay_ring_learner_unroll2_schema_id"
        ],
        "owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source": (
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_host_fallback_allowed": bool(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_host_fallback_allowed"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_action_shape": list(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_action_shape"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_shape": list(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_shape"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_shape": list(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_shape"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_shape": list(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_shape"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_action_mask_shape": list(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_action_mask_shape"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_action_checksum": (
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_action_checksum"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_checksum": (
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_checksum"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_checksum": (
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_value_checksum"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_checksum": (
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_checksum"
            ]
        ),
        "owner_slot_ceiling_replay_ring_learner_unroll2_source_record_window_checksum": (
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_learner_unroll2_source_record_window_checksum"
            ]
        ),
        "owner_slot_ceiling_replay_ring_columnar_append_call_count": float(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_columnar_append_call_count"
            ]
        ),
        "owner_slot_ceiling_replay_ring_columnar_append_record_count": float(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_columnar_append_record_count"
            ]
        ),
        "owner_slot_ceiling_replay_ring_columnar_append_entry_view_object_count": float(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_columnar_append_entry_view_object_count"
            ]
        ),
        "owner_slot_ceiling_replay_ring_columnar_append_step_view_object_count": float(
            owner_slot_metadata[
                "owner_slot_ceiling_replay_ring_columnar_append_step_view_object_count"
            ]
        ),
        "slab_replay_expected_total_steps": int(replay_expected_total_steps),
        "slab_replay_expected_root_count": int(
            replay_expected_total_steps * replay_root_count_per_step
        ),
        "slab_replay_expected_measured_feedback_action_count": int(
            replay_expected_measured_feedback_action_count
        ),
        "slab_replay_expected_append_count": int(replay_expected_append_count),
        "slab_search_feedback_closed_loop": bool(
            replay_metadata["slab_search_feedback_closed_loop"]
        ),
        "slab_replay_tape_bootstrap_action_count": int(
            replay_metadata["slab_replay_tape_bootstrap_action_count"]
        ),
        "slab_replay_feedback_action_count": int(
            replay_metadata["slab_replay_feedback_action_count"]
        ),
        "slab_replay_measured_feedback_action_count": int(
            replay_metadata["slab_replay_measured_feedback_action_count"]
        ),
        "slab_replay_prev_next_joint_action_match_count": int(
            replay_metadata["slab_replay_prev_next_joint_action_match_count"]
        ),
        "slab_replay_prev_next_joint_action_mismatch_count": int(
            replay_metadata["slab_replay_prev_next_joint_action_mismatch_count"]
        ),
        "slab_replay_feedback_differs_from_tape_count": int(
            replay_metadata["slab_replay_feedback_differs_from_tape_count"]
        ),
        "slab_replay_action_source_sequence_checksum": replay_metadata[
            "slab_replay_action_source_sequence_checksum"
        ],
        "slab_step_count": int(replay_metadata["slab_step_count"]),
        "slab_committed_index_group_count": int(
            replay_metadata["slab_committed_index_group_count"]
        ),
        "slab_committed_index_row_count": int(
            replay_metadata["slab_committed_index_row_count"]
        ),
        "slab_replay_payload_flush_count": int(
            replay_metadata["slab_replay_payload_flush_count"]
        ),
        "slab_replay_payload_d2h_bytes": int(
            replay_metadata["slab_replay_payload_d2h_bytes"]
        ),
        "slab_replay_index_rows_observation_materialized": bool(
            replay_metadata["slab_replay_index_rows_observation_materialized"]
        ),
        "slab_replay_index_rows_next_observation_materialized": bool(
            replay_metadata["slab_replay_index_rows_next_observation_materialized"]
        ),
        "slab_replay_sample_batch_built": bool(
            replay_metadata["slab_replay_sample_batch_built"]
        ),
        "slab_replay_sample_batch_size": int(
            replay_metadata["slab_replay_sample_batch_size"]
        ),
        "slab_replay_sample_seed": int(replay_metadata["slab_replay_sample_seed"]),
        "slab_replay_sample_row_id_checksum": replay_metadata[
            "slab_replay_sample_row_id_checksum"
        ],
        "slab_replay_sample_action_checksum": replay_metadata[
            "slab_replay_sample_action_checksum"
        ],
        "slab_replay_sample_observation_checksum": replay_metadata[
            "slab_replay_sample_observation_checksum"
        ],
        "slab_replay_sample_next_observation_checksum": replay_metadata[
            "slab_replay_sample_next_observation_checksum"
        ],
        "slab_replay_sample_record_index_checksum": replay_metadata[
            "slab_replay_sample_record_index_checksum"
        ],
        "slab_replay_sample_policy_row_checksum": replay_metadata[
            "slab_replay_sample_policy_row_checksum"
        ],
        "slab_replay_index_rows_checksum": replay_metadata[
            "slab_replay_index_rows_checksum"
        ],
        "slab_replay_joint_action_feedback_checksum": replay_metadata[
            "slab_replay_joint_action_feedback_checksum"
        ],
        "slab_replay_root_batch_checksum": replay_metadata[
            "slab_replay_root_batch_checksum"
        ],
        "slab_replay_action_step_checksum": replay_metadata[
            "slab_replay_action_step_checksum"
        ],
        "slab_replay_pending_uncommitted_count": int(
            replay_metadata["slab_replay_pending_uncommitted_count"]
        ),
        "slab_replay_action_check_enforced": bool(
            replay_metadata["slab_replay_action_check_enforced"]
        ),
        "slab_replay_root_observation_copied": bool(
            replay_metadata["slab_replay_root_observation_copied"]
        ),
        "slab_root_count": int(replay_metadata["slab_root_count"]),
        "slab_active_root_count": int(replay_metadata["slab_active_root_count"]),
        "slab_inactive_root_count": int(replay_metadata["slab_inactive_root_count"]),
        "slab_selected_action_count": int(replay_metadata["slab_selected_action_count"]),
        "slab_max_active_root_count": int(replay_metadata["slab_max_active_root_count"]),
        "slab_action_count": int(replay_metadata["slab_action_count"]),
        "slab_num_simulations": int(replay_metadata["slab_num_simulations"]),
        "slab_ctree_calls": int(replay_metadata["slab_ctree_calls"]),
        "slab_tolist_calls": int(replay_metadata["slab_tolist_calls"]),
        "slab_per_sim_d2h_bytes": int(replay_metadata["slab_per_sim_d2h_bytes"]),
        "slab_action_d2h_bytes": int(replay_metadata["slab_action_d2h_bytes"]),
        "slab_deferred_replay_payload_d2h_bytes": int(
            replay_metadata["slab_deferred_replay_payload_d2h_bytes"]
        ),
        "slab_root_observation_copy_bytes": int(
            replay_metadata["slab_root_observation_copy_bytes"]
        ),
        "slab_committed_terminal_row_count": int(
            replay_metadata["slab_committed_terminal_row_count"]
        ),
        "slab_committed_next_final_observation_row_count": int(
            replay_metadata["slab_committed_next_final_observation_row_count"]
        ),
        "slab_next_joint_action_checksum": replay_metadata[
            "slab_next_joint_action_checksum"
        ],
        "slab_retains_committed_index_rows": bool(
            replay_metadata["slab_retains_committed_index_rows"]
        ),
        "replay_append_count": int(replay_metadata["replay_append_count"]),
        "replay_ring_entry_count": int(replay_metadata["replay_ring_entry_count"]),
        "replay_ring_stored_index_row_count": int(
            replay_metadata["replay_ring_stored_index_row_count"]
        ),
        "replay_ring_evicted_entry_count": int(
            replay_metadata["replay_ring_evicted_entry_count"]
        ),
        "replay_ring_evicted_index_row_count": int(
            replay_metadata["replay_ring_evicted_index_row_count"]
        ),
        "sample_gate_calls": int(replay_metadata["sample_gate_calls"]),
        "sample_row_count": int(replay_metadata["sample_row_count"]),
        "sample_target_row_count": int(replay_metadata["sample_target_row_count"]),
        "sample_seed": int(replay_metadata["sample_seed"]),
        "sampled_flat_row_checksum": replay_metadata["sampled_flat_row_checksum"],
        "sample_position_order_checksum": replay_metadata[
            "sample_position_order_checksum"
        ],
        "source_record_pair_checksum": replay_metadata["source_record_pair_checksum"],
        "source_record_window_checksum": replay_metadata["source_record_window_checksum"],
        "sample_row_id_checksum": replay_metadata["sample_row_id_checksum"],
        "sample_action_checksum": replay_metadata["sample_action_checksum"],
        "sample_observation_checksum": replay_metadata["sample_observation_checksum"],
        "sample_next_observation_checksum": replay_metadata[
            "sample_next_observation_checksum"
        ],
        "sample_reward_checksum": replay_metadata["sample_reward_checksum"],
        "sample_done_checksum": replay_metadata["sample_done_checksum"],
        "search_impl": search_metadata["search_impl"],
        "search_schema_id": search_metadata["search_schema_id"],
        "root_batch_schema_id": search_metadata["root_batch_schema_id"],
        "search_call_count": int(search_metadata["search_call_count"]),
        "search_root_count": int(search_metadata["search_root_count"]),
        "search_active_root_count": int(search_metadata["search_active_root_count"]),
        "search_inactive_root_count": int(search_metadata["search_inactive_root_count"]),
        "search_selected_action_count": int(search_metadata["search_selected_action_count"]),
        "search_max_active_root_count": int(search_metadata["search_max_active_root_count"]),
        "search_action_count": int(search_metadata["search_action_count"]),
        "search_num_simulations": int(search_metadata["search_num_simulations"]),
        "search_first_legal_policy": bool(search_metadata["search_first_legal_policy"]),
        "search_two_phase_action_only": bool(search_metadata["search_two_phase_action_only"]),
        "search_ctree_calls": int(search_metadata["search_ctree_calls"]),
        "search_tolist_calls": int(search_metadata["search_tolist_calls"]),
        "search_per_sim_d2h_bytes": int(search_metadata["search_per_sim_d2h_bytes"]),
        "search_root_observation_copy_bytes": int(
            search_metadata["search_root_observation_copy_bytes"]
        ),
        "search_action_d2h_bytes": int(search_metadata["search_action_d2h_bytes"]),
        "search_deferred_replay_payload_d2h_bytes": int(
            search_metadata["search_deferred_replay_payload_d2h_bytes"]
        ),
        "search_preallocated_buffer_bytes": int(
            search_metadata["search_preallocated_buffer_bytes"]
        ),
        "search_buffer_reused": bool(search_metadata["search_buffer_reused"]),
        "search_action_step_identity_checked": bool(
            search_metadata["search_action_step_identity_checked"]
        ),
        "search_action_step_root_index_matches_active": bool(
            search_metadata["search_action_step_root_index_matches_active"]
        ),
        "search_action_step_env_row_matches_root": bool(
            search_metadata["search_action_step_env_row_matches_root"]
        ),
        "search_action_step_player_matches_root": bool(
            search_metadata["search_action_step_player_matches_root"]
        ),
        "search_action_step_policy_env_id_matches_root": bool(
            search_metadata["search_action_step_policy_env_id_matches_root"]
        ),
        "search_selected_action_shape_matches": bool(
            search_metadata["search_selected_action_shape_matches"]
        ),
        "search_selected_action_legal": bool(
            search_metadata["search_selected_action_legal"]
        ),
        "search_replay_payload_digest_deferred": bool(
            search_metadata["search_replay_payload_digest_deferred"]
        ),
        "search_replay_payload_digest_matches_handle": bool(
            search_metadata["search_replay_payload_digest_matches_handle"]
        ),
        "search_selected_action_digest_matches_payload": bool(
            search_metadata["search_selected_action_digest_matches_payload"]
        ),
        "search_root_batch_observation_source": search_metadata[
            "search_root_batch_observation_source"
        ],
        "search_root_batch_observation_copied": bool(
            search_metadata["search_root_batch_observation_copied"]
        ),
        "search_root_batch_observation_shape": list(
            search_metadata["search_root_batch_observation_shape"]
        ),
        "search_root_batch_observation_dtype": search_metadata[
            "search_root_batch_observation_dtype"
        ],
        "search_root_batch_row_major_sidecars_checked": bool(
            search_metadata["search_root_batch_row_major_sidecars_checked"]
        ),
        "search_done_root_matches_repeat_done": bool(
            search_metadata["search_done_root_matches_repeat_done"]
        ),
        "search_active_root_mask_matches_non_done_legal": bool(
            search_metadata["search_active_root_mask_matches_non_done_legal"]
        ),
        "search_to_play_all_default": bool(search_metadata["search_to_play_all_default"]),
        "search_target_reward_matches_reward": bool(
            search_metadata["search_target_reward_matches_reward"]
        ),
        "search_root_observation_shares_stack": bool(
            search_metadata["search_root_observation_shares_stack"]
        ),
        "search_selected_action_digest": search_metadata["search_selected_action_digest"],
        "search_replay_payload_digest": search_metadata["search_replay_payload_digest"],
        "search_root_batch_checksum": search_metadata["search_root_batch_checksum"],
        "search_action_step_checksum": search_metadata["search_action_step_checksum"],
        "search_root_observation_checksum": search_metadata[
            "search_root_observation_checksum"
        ],
        "search_selected_action_checksum": search_metadata[
            "search_selected_action_checksum"
        ],
        "search_joint_action_checksum": search_metadata["search_joint_action_checksum"],
        "autoreset_rows_checksum": compact["autoreset_rows_checksum"],
        "autoreset_rows_checksum_match": bool(
            comparison["field_matches"]["autoreset_rows_checksum"]
        ),
        "env_action_checksum_total": action_tape_checksum,
        "env_done_checksum_total": compact["output_checksums"]["done"],
        "env_reward_checksum_total": compact["output_checksums"]["reward"],
        "env_action_mask_checksum_total": compact["output_checksums"]["action_mask"],
        "env_trajectory_checksum_total": compact["per_step_state_checksum"],
        "env_terminal_row_checksum_total": compact["output_checksums"]["terminal_rows"],
        "env_autoreset_row_checksum_total": compact["autoreset_rows_checksum"],
        "env_autoreset_row_count_pair_checksum": _array_checksum(autoreset_rows),
        "env_death_cause_checksum_total": compact["output_checksums"]["death"],
        "output_compared_fields": list(compact["output_compared_fields"]),
        "uncompared_output_fields": list(compact["uncompared_output_fields"]),
        "reward_shape": compact["reward_shape"],
        "done_shape": compact["done_shape"],
        "terminated_shape": compact["done_shape"],
        "truncated_shape": compact["done_shape"],
        "action_mask_shape": compact["action_mask_shape"],
        "done_equals_terminated_or_truncated": bool(
            comparison["scalar_matches"]["done_invariant_violation_count"]
        ),
        "death_row_count": int(compact["death_row_count"]),
        "new_death_row_count": int(compact["new_death_row_count"]),
        "measured_new_death_row_count": int(compact["measured_new_death_row_count"]),
        "death_cause_names": list(compact["death_cause_names"]),
        "new_death_cause_names": list(compact["new_death_cause_names"]),
        "measured_new_death_cause_names": list(compact["measured_new_death_cause_names"]),
        "death_transition_step_indices": list(compact["death_transition_step_indices"]),
        "measured_death_transition_step_indices": list(
            compact["measured_death_transition_step_indices"]
        ),
        "expected_death_cause_names": expected_death_cause_names,
        "expected_death_cause_names_present": bool(expected_death_cause_names_present),
        "expected_measured_death_cause_names_present": bool(
            expected_measured_death_cause_names_present
        ),
        "expected_death_evidence_present": bool(expected_death_evidence_present),
        "expected_measured_death_evidence_present": bool(
            expected_measured_death_evidence_present
        ),
        "expects_terminal_rows": bool(expects_terminal_rows),
        "expected_terminal_autoreset_evidence_present": bool(
            expected_terminal_autoreset_evidence_present
        ),
        "terminal_row_count": int(compact["terminal_row_count"]),
        "autoreset_call_count": int(compact["autoreset_call_count"]),
        "autoreset_row_count": int(compact["autoreset_row_count"]),
        "terminal_rows_equal_autoreset_rows": bool(terminal_rows_equal_autoreset_rows),
        "measured_tape_indices": list(compact["measured_tape_indices"]),
        "first_measured_tape_index": compact["first_measured_tape_index"],
        "measured_initial_fixture_transition_exercised": bool(
            int(compact["measured_new_death_row_count"]) > 0
            and all(
                tape_index in compact["measured_tape_indices"]
                for tape_index in range(len(tape.actions))
            )
        ),
        "zero_observation_stub": not bool(config.render_observation),
        "replay_index_rows_exposed": bool(
            int(replay_metadata["slab_committed_index_row_count"]) > 0
            or int(
                owner_slot_metadata[
                    "owner_slot_ceiling_stage_replay_index_rows_row_count"
                ]
            )
            > 0
        ),
    }


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    if int(config.batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(config.measured_steps) <= 0:
        raise ValueError("measured_steps must be positive")
    if int(config.warmup_steps) < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if int(config.body_capacity) <= 0:
        raise ValueError("body_capacity must be positive")
    if int(config.random_tape_capacity_min) <= 0:
        raise ValueError("random_tape_capacity_min must be positive")
    if bool(config.run_search) and not bool(config.render_observation):
        raise ValueError("run_search requires render_observation")
    if bool(config.run_slab_replay) and not bool(config.render_observation):
        raise ValueError("run_slab_replay requires render_observation")
    if bool(config.run_owner_slot_ceiling) and not bool(config.render_observation):
        raise ValueError("run_owner_slot_ceiling requires render_observation")
    enabled_proof_gates = sum(
        int(flag)
        for flag in (
            bool(config.run_search),
            bool(config.run_slab_replay),
            bool(config.run_owner_slot_ceiling),
        )
    )
    if enabled_proof_gates > 1:
        raise ValueError(
            "run_search, run_slab_replay, and run_owner_slot_ceiling are separate proof gates"
        )

    initial_state, tape = _load_fixture_action_tape(config)
    initial_state = _pad_random_tape_capacity(
        initial_state,
        min_capacity=int(config.random_tape_capacity_min),
    )
    batched_state = _repeat_state(initial_state, int(config.batch_size))
    compact_env = _make_env(config=config, initial_state=initial_state, tape=tape)
    fixed_env = _make_env(config=config, initial_state=initial_state, tape=tape)
    reset_seed = np.arange(int(config.batch_size), dtype=np.uint64)
    reset_seed += np.uint64(int(config.seed))
    compact_env.reset_from_state_arrays(batched_state, reset_seed=reset_seed)
    fixed_env.reset_from_state_arrays(batched_state, reset_seed=reset_seed)

    compact = _loop(
        label="compact_profile",
        env=compact_env,
        tape=tape,
        config=config,
        mode="compact_profile",
    )
    fixed = _loop(
        label="fixed_buffer_direct",
        env=fixed_env,
        tape=tape,
        config=config,
        mode="fixed_buffer_direct",
    )
    comparison = _compare_loops(compact, fixed)
    action_tape_checksum = _action_tape_checksum(tape, config)
    proof = _proof_block(
        config=config,
        tape=tape,
        compact=compact,
        fixed=fixed,
        comparison=comparison,
        action_tape_checksum=action_tape_checksum,
    )
    owner_buffer_ceiling = _owner_buffer_ceiling_block(
        config=config,
        compact=compact,
        fixed=fixed,
        comparison=comparison,
    )
    result = {
        "schema": SCHEMA_VERSION,
        "benchmark": "vector_fixed_action_tape",
        "status": "pass" if proof["passed"] else "fail",
        "config": {
            "scenario": str(config.scenario),
            "batch_size": int(config.batch_size),
            "measured_steps": int(config.measured_steps),
            "warmup_steps": int(config.warmup_steps),
            "body_capacity": int(config.body_capacity),
            "random_tape_capacity_min": int(config.random_tape_capacity_min),
            "render_observation": bool(config.render_observation),
            "run_search": bool(config.run_search),
            "run_slab_replay": bool(config.run_slab_replay),
            "run_owner_slot_ceiling": bool(config.run_owner_slot_ceiling),
            "seed": int(config.seed),
            "event_mode": str(config.event_mode),
            "use_direct_autoreset": bool(config.use_direct_autoreset),
        },
        "tape": {
            "scenario_id": tape.scenario_id,
            "player_count": int(tape.player_count),
            "steps": len(tape.actions),
            "step_ms": [float(value) for value in tape.step_ms],
            "timer_advance_ms": [float(value) for value in tape.timer_advance_ms],
            "action_checksum": action_tape_checksum,
            "action_array_checksum": _array_checksum(np.concatenate(tape.actions, axis=0)),
        },
        "proof": proof,
        "compact_profile": compact,
        "fixed_buffer_direct": fixed,
        "comparison": comparison,
        "owner_buffer_ceiling": owner_buffer_ceiling,
        "known_limits": _known_limits(config),
    }
    if bool(config.require_pass) and result["status"] != "pass":
        raise RuntimeError(
            "fixed-action tape proof failed: "
            f"{json.dumps(result['proof'], sort_keys=True)}"
        )
    return result


def _known_limits(config: BenchmarkConfig) -> list[str]:
    limits = [
        "toy benchmark only; not H100 speed evidence",
        "fixed_buffer_direct intentionally uses private env internals",
    ]
    if bool(config.render_observation):
        limits.append(
            "rendered observation uses the CPU oracle renderer; not optimized H100 resident "
            "observation speed evidence"
        )
    else:
        limits.append(
            "zero-observation checksum is a stub until the observation variant is added"
        )
    if bool(config.run_search):
        limits.append(
            "fixed-shape search uses profile-only first-legal action owner; not compact Torch "
            "or MCTS speed evidence"
        )
    if bool(config.run_slab_replay):
        limits.append(
            "slab replay uses profile-only first-legal action owner and host replay-ring "
            "sample snapshots; not compact Torch, MCTS, learner, or H100 speed evidence"
        )
        limits.append("no learner update, policy refresh, resident replay, or GPU search claim")
    elif bool(config.run_owner_slot_ceiling):
        limits.append(
            "owner-slot ceiling uses a CPU-oracle resident-handle shim and profile-only "
            "first-legal action owner; not compact Torch, MCTS, learner, or H100 speed evidence"
        )
        limits.append(
            "owner-slot ceiling drains staged owner rows into the real compact replay ring "
            "and builds a resident device learner unroll-2 batch locally; fixed resident "
            "row/window slots or handle-ring sampling are the next rung"
        )
    else:
        limits.append(
            "no compact Torch search, slab commit, replay append, sample, learner, or policy refresh"
        )
    return limits


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--measured-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--body-capacity", type=int, default=8)
    parser.add_argument("--random-tape-capacity-min", type=int, default=16)
    parser.add_argument("--render-observation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-slab-replay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--run-owner-slot-ceiling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=132)
    parser.add_argument("--event-mode", default="no-event", choices=("no-event", "debug-event"))
    parser.add_argument("--use-direct-autoreset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-require-pass", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_benchmark(
        BenchmarkConfig(
            scenario=str(args.scenario),
            batch_size=int(args.batch_size),
            measured_steps=int(args.measured_steps),
            warmup_steps=int(args.warmup_steps),
            body_capacity=int(args.body_capacity),
            random_tape_capacity_min=int(args.random_tape_capacity_min),
            render_observation=bool(args.render_observation),
            run_search=bool(args.run_search),
            run_slab_replay=bool(args.run_slab_replay),
            run_owner_slot_ceiling=bool(args.run_owner_slot_ceiling),
            seed=int(args.seed),
            event_mode=str(args.event_mode),
            use_direct_autoreset=bool(args.use_direct_autoreset),
            require_pass=not bool(args.no_require_pass),
        )
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
