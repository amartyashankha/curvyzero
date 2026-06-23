import pickle
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import torch as _torch
except Exception:  # pragma: no cover - torch is optional for module import.
    _torch = None

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.env import vector_runtime
from curvyzero.env import vector_reset
from curvyzero.training.source_state_batched_observation_profile import (
    SourceStateBatchedRenderResult,
)
from curvyzero.training import source_state_hybrid_observation_profile as hybrid_profile
from curvyzero.training import exploration_bonus as xb
from curvyzero.training.compact_policy_row_bridge import CompactReplayIndexRowsV1
from curvyzero.training.compact_policy_row_bridge import CompactDeviceReplayIndexRowsV1
from curvyzero.training.compact_policy_row_bridge import (
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1
from curvyzero.training.compact_policy_row_bridge import build_compact_root_batch_v1
from curvyzero.training.compact_policy_row_bridge import (
    compact_root_action_context_v1_from_request,
)
from curvyzero.training.compact_policy_row_bridge import validate_compact_search_result_v1
from curvyzero.training.compact_policy_refresh_handoff import compact_model_state_digest_v1
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_policy_refresh_metadata_from_state_v1,
)
from curvyzero.training.compact_observation_contract import ResidentObservationBatchV1
from curvyzero.training.compact_muzero_learner import CompactMuZeroLearnerConfigV1
from curvyzero.training.compact_muzero_learner import (
    build_compact_muzero_learner_batch_v1,
)
from curvyzero.training.compact_owned_loop import COMPACT_OWNED_LOOP_SCHEMA_ID
from curvyzero.training.compact_owned_loop import (
    COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS,
)
from curvyzero.training.compact_search_service import COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID
from curvyzero.training.compact_search_service import (
    COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import CompactDeviceSearchReplayPayloadV1
from curvyzero.training.compact_search_service import CompactSearchActionStepV1
from curvyzero.training.compact_search_service import compact_search_array_digest_v1
from curvyzero.training.compact_search_service import compact_search_action_step_v1_from_result
from curvyzero.training.compact_search_service import (
    compact_search_deferred_replay_payload_digest_v1,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM,
)
from curvyzero.training.compact_owner_search_service import (
    CompactLazyThreadedOwnerSearchSlabProxyV1,
)
from curvyzero.training.compact_owner_search_service import CompactOwnerSearchRequestV1
from curvyzero.training.compact_rollout_slab import CompactOwnerSearchDirectStepperV1
from curvyzero.training.compact_rollout_slab import (
    CompactOwnerSearchReplayAppendDerivedTransitionBatchV1,
)
from curvyzero.training.compact_rollout_slab import CompactRolloutSlab
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.source_state_hybrid_observation_profile import (
    COMPACT_REPLAY_RENDER_STATE_SNAPSHOT_SCHEMA_ID,
    COMPACT_REPLAY_STORE_STATE_SCHEMA_ID,
    CompactReplayColumnarAppendRecordV1,
    CompactReplayRendererBackedObservationProviderV1,
    CompactReplayRenderStateSnapshotV1,
    CompactResidentFrameStackReplaySnapshotV1,
    CompactResidentSampleBatchV1,
    HYBRID_OBSERVATION_TIMING_FIELDS,
    HYBRID_OBSERVATION_MODE_RENDERER_BACKED,
    HybridActorStepPayload,
    HybridBatchedStackProbeResult,
    HybridBatchedObservationProfileManager,
    HybridCompactBatch,
    HybridObservationProfileConfig,
    HybridPolicySearchProbeResult,
    InProcessHybridCurvyTronActor,
    PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_KEYS,
    PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_MARKER,
    PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
    PERSISTENT_GPU_PROFILE_RENDER_STATE_KEYS,
    RESIDENT_REPLAY_SNAPSHOT_MODE_LATEST_FRAME_HISTORY,
    HybridObservationProfileStep,
    _CompactReplayRingV1,
    _build_compact_resident_grouped_device_learner_batch_fast,
    _build_compact_resident_grouped_device_sample_batch_fast,
    _build_compact_resident_sample_batch_from_device_index_rows_fast,
    _build_compact_resident_sample_batch_from_index_rows_fast,
    _build_compact_sample_batch_from_index_rows_fast,
    _concat_compact_resident_sample_batches_fast,
    _concat_compact_sample_batches_fast,
    _compact_rollout_slab_search_model,
    _joint_action_from_compact_search_action_step,
    _make_compact_batch,
    _refresh_compact_rollout_slab_search_from_owner_ref,
    _take_compact_index_rows,
    _trainer_replay_step_from_hybrid_step,
    run_hybrid_observation_profile,
)


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402


if _torch is not None:

    class _TinyMuZeroForRefreshTest(_torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _torch.nn.Sequential(
                _torch.nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
                _torch.nn.ReLU(),
                _torch.nn.AdaptiveAvgPool2d((1, 1)),
                _torch.nn.Flatten(),
                _torch.nn.Linear(8, 16),
                _torch.nn.Tanh(),
            )
            self.action_embedding = _torch.nn.Embedding(ACTION_COUNT, 16)
            self.policy_head = _torch.nn.Linear(16, ACTION_COUNT)
            self.value_head = _torch.nn.Linear(16, 3)
            self.reward_head = _torch.nn.Linear(16, 3)

        def initial_inference(self, obs):
            from lzero.model.common import MZNetworkOutput

            latent = self.encoder(obs)
            value = self.value_head(latent)
            return MZNetworkOutput(
                value,
                _torch.zeros_like(value),
                self.policy_head(latent),
                latent,
            )

        def recurrent_inference(self, latent_state, action):
            from lzero.model.common import MZNetworkOutput

            next_latent = _torch.tanh(
                latent_state + self.action_embedding(action.reshape(-1).long())
            )
            return MZNetworkOutput(
                self.value_head(next_latent),
                self.reward_head(next_latent),
                self.policy_head(next_latent),
                next_latent,
            )

else:
    _TinyMuZeroForRefreshTest = None


class _OwnerManagerBoundaryReplayStore:
    def __init__(self) -> None:
        self.direct_append_call_count = 0
        self.legacy_append_call_count = 0
        self.append_count = 0
        self.last_transition_batches = ()
        self.last_root_batch_cache_keys = ()

    def append_owner_search_transition_batches(
        self,
        *,
        replay_append_transition_batches,
        root_batch,
        search_result,
        request,
        root_batch_cache=None,
    ):
        assert isinstance(root_batch, CompactRootBatchV1)
        assert isinstance(search_result, CompactSearchResultV1)
        assert isinstance(request, CompactOwnerSearchRequestV1)
        self.direct_append_call_count += 1
        self.last_transition_batches = tuple(replay_append_transition_batches)
        self.last_root_batch_cache_keys = tuple(
            sorted(int(key) for key in dict(root_batch_cache or {}).keys())
        )
        assert self.last_transition_batches
        for batch in self.last_transition_batches:
            assert isinstance(batch, CompactOwnerSearchReplayAppendDerivedTransitionBatchV1)
        transition_count = sum(
            int(batch.transition_count) for batch in self.last_transition_batches
        )
        action_count = sum(
            int(np.asarray(batch.applied_action_counts, dtype=np.int64).sum())
            for batch in self.last_transition_batches
        )
        action_checksum = sum(
            int(np.asarray(batch.applied_action_checksums, dtype=np.int64).sum())
            for batch in self.last_transition_batches
        )
        self.append_count += int(transition_count)
        return {
            "appended_count": int(transition_count),
            "owner_action_feedback": {
                "compact_owner_search_action_feedback_verified": True,
                "compact_owner_search_action_feedback_transition_count": int(transition_count),
                "compact_owner_search_action_feedback_action_count": int(action_count),
                "compact_owner_search_action_feedback_mismatch_count": 0,
                "compact_owner_search_expected_joint_action_checksum": int(action_checksum),
                "compact_owner_search_applied_joint_action_checksum": int(action_checksum),
                "compact_owner_search_replay_action_checksum": int(action_checksum),
            },
            "compact_owner_search_direct_transition_batch_replay_requested": True,
            "compact_owner_search_direct_transition_batch_replay_used": True,
            "compact_owner_search_direct_transition_batch_replay_batch_count": len(
                self.last_transition_batches
            ),
            "compact_owner_search_direct_transition_batch_replay_transition_count": int(
                transition_count
            ),
            "compact_owner_search_direct_transition_batch_replay_transport_entry_count": len(
                self.last_transition_batches
            ),
            "compact_owner_search_direct_transition_batch_replay_legacy_expanded_entry_count": 0,
            "compact_owner_search_direct_transition_batch_replay_index_entry_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_ring_entry_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_columnar_append_used": True,
            "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count": int(
                transition_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fallback_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fallback_reason": "none",
        }

    def append_owner_search_replay(self, **kwargs):
        del kwargs
        self.legacy_append_call_count += 1
        raise AssertionError("production manager boundary must not use legacy replay append")


class _OwnerManagerBoundaryLearner:
    def __init__(self) -> None:
        self.train_calls = 0
        self.update_count = 0

    def train_owner_search_step(
        self,
        *,
        replay_store,
        root_batch,
        search_result,
        sample_batch_size,
        train_steps,
        request,
    ):
        assert replay_store.append_count > 0
        assert isinstance(root_batch, CompactRootBatchV1)
        assert isinstance(search_result, CompactSearchResultV1)
        assert isinstance(request, CompactOwnerSearchRequestV1)
        self.train_calls += 1
        self.update_count += int(train_steps)
        return {
            "learner_update_count": int(train_steps),
            "sample_metadata": {
                "compact_rollout_slab_sample_gate_sample_row_count": int(sample_batch_size),
                "compact_rollout_slab_sample_gate_target_row_count": int(sample_batch_size),
                "compact_rollout_slab_sample_gate_requested_sample_row_count": int(
                    sample_batch_size
                ),
                "compact_rollout_slab_sample_gate_require_next_targets": True,
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
            },
            "learner_telemetry": {
                "compact_owner_search_owner_train_wall_sec": 0.01,
                "compact_owner_search_owner_train_sample_sec": 0.001,
                "compact_owner_search_owner_train_learner_update_sec": 0.002,
                "compact_owner_search_owner_train_model_state_digest_sec": 0.0,
                "compact_owner_search_owner_train_model_state_dict_sec": 0.0,
                "compact_owner_search_owner_train_owner_ref_build_sec": 0.0,
                "compact_owner_search_owner_train_accounted_sec": 0.003,
                "compact_owner_search_owner_train_residual_sec": 0.0,
            },
        }


def test_hybrid_profile_zero_observation_shape_and_counts_are_deterministic():
    config = HybridObservationProfileConfig(
        batch_size=3,
        actor_count=2,
        steps=2,
        warmup_steps=1,
        seed=17,
    )

    first = run_hybrid_observation_profile(config)
    second = run_hybrid_observation_profile(config)

    assert first["profile_only"] is True
    assert first["calls_train_muzero"] is False
    assert first["observation_mode"] == "zero_observation_stack"
    assert first["policy_search_probe_backend_name"] == "none"
    assert first["policy_search_probe_calls"] == 0
    assert first["batched_stack_probe_backend_name"] == "none"
    assert first["batched_stack_probe_calls"] == 0
    assert first["materialize_scalar_timestep"] is True
    assert first["last_observation_shape"] == [3, 2, 4, 64, 64]
    assert first["last_flat_obs_shape"] == [6, 4, 64, 64]
    assert first["last_target_reward_shape"] == [6, 1]
    assert first["ready_count"] == 6
    assert first["timestep_count"] == 12
    assert first["materialized_timestep_count"] == 12
    assert first["live_physical_row_count"] == 3
    assert first["death_mode"] == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
    assert first["last_policy_env_id"] == [0, 1, 2, 3, 4, 5]
    assert first["last_policy_env_row"] == second["last_policy_env_row"]
    assert first["last_policy_player"] == second["last_policy_player"]
    assert first["last_payload_summary"]["global_rows"] == [0, 1, 2]


def test_hybrid_profile_config_passes_normal_death_mode_to_actors():
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            death_mode=vector_runtime.DEATH_MODE_NORMAL,
        )
    )

    assert manager.actors[0].env.death_mode == vector_runtime.DEATH_MODE_NORMAL


def _source_opponent_trail_death_fixture_state_and_actions() -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    float,
]:
    fixture = seed_bridge.seed_fixture(
        "scenarios/environment/source_collision_head_head_reverse_order_single_death_step.json",
        body_capacity=8,
    )
    prepared_step = vector_compare.prepare_fixture_array_step(fixture, step_index=0)
    source_moves = np.asarray(prepared_step["source_moves"], dtype=np.int8)
    actions = source_moves.astype(np.int16).reshape(1, -1) + 1
    step_ms = float(prepared_step["step_ms"])
    return vector_compare.array_state_from_seed(fixture), actions, step_ms


def _fixture_state_for_env_capacity(env, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    adjusted: dict[str, np.ndarray] = {}
    for name, value in state.items():
        array = np.asarray(value)
        target = np.asarray(env.state[name])
        if array.shape == target.shape:
            adjusted[name] = array
            continue
        if array.ndim != target.ndim or any(
            int(size) > int(limit) for size, limit in zip(array.shape, target.shape, strict=True)
        ):
            adjusted[name] = array
            continue
        expanded = np.asarray(env.reset_template[name]).copy()
        slices = tuple(slice(0, int(size)) for size in array.shape)
        expanded[slices] = array
        adjusted[name] = expanded
    return adjusted


def test_hybrid_compact_native_path_preserves_normal_collision_death_fixture():
    state, actions, step_ms = _source_opponent_trail_death_fixture_state_and_actions()
    actor = InProcessHybridCurvyTronActor(
        actor_id=0,
        global_rows=np.asarray([0], dtype=np.int32),
        player_count=2,
        seed=123,
        max_ticks=2_000,
        body_capacity=8,
        death_mode=vector_runtime.DEATH_MODE_NORMAL,
        decision_source_frames=1,
        source_physics_step_ms=step_ms,
    )
    actor.env.reset_from_state_arrays(
        _fixture_state_for_env_capacity(actor.env, state),
        reset_seed=np.asarray([123], dtype=np.uint64),
    )

    payload = actor.step(
        actions,
        autoreset_terminal_rows=False,
        include_render_state=False,
    )

    np.testing.assert_array_equal(payload.done, np.asarray([True], dtype=np.bool_))
    np.testing.assert_array_equal(payload.terminated, np.asarray([True], dtype=np.bool_))
    np.testing.assert_array_equal(payload.truncated, np.asarray([False], dtype=np.bool_))
    np.testing.assert_array_equal(
        payload.terminal_reason,
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(payload.death_count, np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        payload.death_player,
        np.asarray([[0, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        payload.death_cause,
        np.asarray(
            [
                [
                    vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
                    vector_runtime.DEATH_CAUSE_NONE,
                ]
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        payload.death_hit_owner,
        np.asarray([[1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(payload.winner, np.asarray([1], dtype=np.int16))
    np.testing.assert_array_equal(payload.draw, np.asarray([False], dtype=np.bool_))
    np.testing.assert_array_equal(payload.terminal_global_rows, np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        payload.reward,
        np.asarray([[-1.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(payload.final_reward_map, payload.reward)

    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=1,
            actor_count=1,
            player_count=2,
            seed=123,
            body_capacity=8,
            death_mode=vector_runtime.DEATH_MODE_NORMAL,
            decision_source_frames=1,
            source_physics_step_ms=step_ms,
            autoreset_terminal_rows=False,
            native_actor_buffer=True,
            pickle_payload=False,
            materialize_scalar_timestep=False,
        ),
        batched_stack_probe=_CaptureCompactBatchProbe(),
    )
    actor_state = _fixture_state_for_env_capacity(manager.actors[0].env, state)
    manager.actors[0].env.reset_from_state_arrays(
        actor_state,
        reset_seed=np.asarray([123], dtype=np.uint64),
    )

    step = manager.step(actions)

    assert step.compact_batch is not None
    compact_batch = step.compact_batch
    np.testing.assert_array_equal(step.done, payload.done)
    np.testing.assert_array_equal(step.terminated, payload.terminated)
    np.testing.assert_array_equal(step.truncated, payload.truncated)
    np.testing.assert_array_equal(step.death_count, payload.death_count)
    np.testing.assert_array_equal(step.death_player, payload.death_player)
    np.testing.assert_array_equal(step.death_cause, payload.death_cause)
    np.testing.assert_array_equal(step.death_hit_owner, payload.death_hit_owner)
    np.testing.assert_array_equal(step.winner, payload.winner)
    np.testing.assert_array_equal(step.draw, payload.draw)
    np.testing.assert_array_equal(compact_batch.terminated, payload.terminated)
    np.testing.assert_array_equal(compact_batch.truncated, payload.truncated)
    np.testing.assert_array_equal(compact_batch.death_count, payload.death_count)
    np.testing.assert_array_equal(compact_batch.death_player, payload.death_player)
    np.testing.assert_array_equal(
        compact_batch.terminal_global_rows, np.asarray([0], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        compact_batch.done_root,
        np.asarray([True, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        compact_batch.target_reward,
        np.asarray([[-1.0], [1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        compact_batch.active_root_mask,
        np.asarray([False, False], dtype=np.bool_),
    )
    np.testing.assert_array_equal(compact_batch.terminal_row_mask, np.asarray([True]))
    np.testing.assert_array_equal(
        compact_batch.final_observation_row_mask,
        np.asarray([True]),
    )
    np.testing.assert_array_equal(compact_batch.final_reward_map, compact_batch.reward)


def test_hybrid_profile_manager_exposes_row_player_scalar_ids():
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(batch_size=2, actor_count=2, seed=23)
    )
    action = np.asarray([[0, 1], [2, 1]], dtype=np.int16)

    step = manager.step(action)

    np.testing.assert_array_equal(step.policy_env_id, np.asarray([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(step.policy_env_row, np.asarray([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(step.policy_player, np.asarray([0, 1, 0, 1], dtype=np.int32))
    assert manager.scalar_env_id(row=1, player=0) == 2
    assert manager.row_player_for_scalar_env_id(3) == (1, 1)
    assert [item["row"] for item in step.timestep.info] == [0, 0, 1, 1]
    assert [item["player"] for item in step.timestep.info] == [0, 1, 0, 1]
    assert all(item["profile_only"] is True for item in step.timestep.info)


def test_make_compact_batch_owns_deferred_arrays():
    observation = np.arange(2 * 2 * 4 * 64 * 64, dtype=np.float32).reshape(
        2,
        2,
        4,
        64,
        64,
    )
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=np.bool_)
    reward = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    final_reward_map = reward + 10.0
    done = np.asarray([False, True], dtype=np.bool_)
    policy_env_id = np.asarray([0, 1, 2, 3], dtype=np.int32)
    policy_env_row = np.asarray([0, 0, 1, 1], dtype=np.int32)
    policy_player = np.asarray([0, 1, 0, 1], dtype=np.int32)
    target_reward = reward.reshape(-1, 1)
    done_root = np.asarray([False, False, True, True], dtype=np.bool_)
    to_play = np.full((4,), -1, dtype=np.int64)
    active_root_mask = np.ones((4,), dtype=np.bool_)
    final_observation = observation + 100.0
    terminal_global_rows = np.asarray([1], dtype=np.int32)
    autoreset_global_rows = np.asarray([], dtype=np.int32)
    episode_step = np.asarray([5, 6], dtype=np.int32)
    elapsed_ms = np.asarray([50.0, 60.0], dtype=np.float64)
    round_id = np.asarray([7, 8], dtype=np.int32)
    alive = np.ones((2, 2), dtype=np.bool_)
    joint_action = np.asarray([[0, 1], [2, 0]], dtype=np.int16)

    batch = _make_compact_batch(
        observation=observation,
        action_mask=action_mask,
        reward=reward,
        final_reward_map=final_reward_map,
        done=done,
        policy_env_id=policy_env_id,
        policy_env_row=policy_env_row,
        policy_player=policy_player,
        target_reward=target_reward,
        done_root=done_root,
        to_play=to_play,
        active_root_mask=active_root_mask,
        final_observation=final_observation,
        terminal_global_rows=terminal_global_rows,
        autoreset_global_rows=autoreset_global_rows,
        episode_step=episode_step,
        elapsed_ms=elapsed_ms,
        round_id=round_id,
        alive=alive,
        joint_action=joint_action,
        batch_size=2,
    )

    observation[...] = -999.0
    action_mask[...] = False
    reward[...] = -999.0
    final_reward_map[...] = -999.0
    policy_env_id[...] = -999
    joint_action[...] = -999
    final_observation[...] = -999.0

    assert batch.observation[0, 0, 0, 0, 0] == 0.0
    assert bool(batch.action_mask[0, 0, 0]) is True
    assert batch.reward[0, 0] == 1.0
    assert batch.final_reward_map[0, 0] == 11.0
    assert batch.policy_env_id[0] == 0
    assert batch.joint_action[0, 1] == 1
    assert batch.final_observation is not None
    assert batch.final_observation[0, 0, 0, 0, 0] == 100.0


def test_make_compact_batch_can_borrow_stable_observation_copy():
    observation = np.zeros((1, 2, 4, 64, 64), dtype=np.uint8)
    final_observation = observation.copy()

    batch = _make_compact_batch(
        observation=observation,
        action_mask=np.ones((1, 2, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((1, 2), dtype=np.float32),
        final_reward_map=np.zeros((1, 2), dtype=np.float32),
        done=np.zeros((1,), dtype=np.bool_),
        policy_env_id=np.asarray([0, 1], dtype=np.int32),
        policy_env_row=np.asarray([0, 0], dtype=np.int32),
        policy_player=np.asarray([0, 1], dtype=np.int32),
        target_reward=np.zeros((2, 1), dtype=np.float32),
        done_root=np.zeros((2,), dtype=np.bool_),
        to_play=np.full((2,), -1, dtype=np.int64),
        active_root_mask=np.ones((2,), dtype=np.bool_),
        final_observation=final_observation,
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.zeros((1,), dtype=np.int32),
        elapsed_ms=np.zeros((1,), dtype=np.float64),
        round_id=np.zeros((1,), dtype=np.int32),
        alive=np.ones((1, 2), dtype=np.bool_),
        joint_action=np.zeros((1, 2), dtype=np.int16),
        batch_size=1,
        copy_observation=False,
    )

    assert np.shares_memory(batch.observation, observation)
    assert batch.final_observation is not None
    assert np.shares_memory(batch.final_observation, final_observation)


def test_resident_compact_batch_final_observation_fails_at_root_boundary():
    torch = pytest.importorskip("torch")
    observation = np.zeros((1, 2, 4, 64, 64), dtype=np.uint8)
    final_observation = observation.copy()
    resident = SimpleNamespace(
        device_observation=torch.zeros((1, 2, 4, 64, 64), dtype=torch.uint8),
        root_device_observation=torch.zeros((2, 4, 64, 64), dtype=torch.uint8),
        generation_id=1,
        batch_size=1,
        player_count=2,
        stack_shape=(4, 64, 64),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=1,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
    )
    batch = _make_compact_batch(
        observation=observation,
        action_mask=np.ones((1, 2, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((1, 2), dtype=np.float32),
        final_reward_map=np.zeros((1, 2), dtype=np.float32),
        done=np.asarray([True], dtype=np.bool_),
        policy_env_id=np.asarray([0, 1], dtype=np.int32),
        policy_env_row=np.asarray([0, 0], dtype=np.int32),
        policy_player=np.asarray([0, 1], dtype=np.int32),
        target_reward=np.zeros((2, 1), dtype=np.float32),
        done_root=np.ones((2,), dtype=np.bool_),
        to_play=np.full((2,), -1, dtype=np.int64),
        active_root_mask=np.zeros((2,), dtype=np.bool_),
        final_observation=final_observation,
        terminal_global_rows=np.asarray([0], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.zeros((1,), dtype=np.int32),
        elapsed_ms=np.zeros((1,), dtype=np.float64),
        round_id=np.zeros((1,), dtype=np.int32),
        alive=np.zeros((1, 2), dtype=np.bool_),
        joint_action=np.zeros((1, 2), dtype=np.int16),
        batch_size=1,
        resident_observation=resident,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    )

    with pytest.raises(ReplayCompatibilityError, match="resident-owned"):
        build_compact_root_batch_v1(batch, search_lane="unit_test_resident_terminal")


def test_resident_compact_batch_accepts_resident_final_observation_at_root_boundary():
    torch = pytest.importorskip("torch")
    observation = np.zeros((1, 2, 4, 64, 64), dtype=np.uint8)
    final_device_observation = torch.full((1, 2, 4, 64, 64), 42, dtype=torch.uint8)
    resident = ResidentObservationBatchV1(
        device_observation=torch.zeros((1, 2, 4, 64, 64), dtype=torch.uint8),
        root_device_observation=torch.zeros((2, 4, 64, 64), dtype=torch.uint8),
        generation_id=1,
        batch_size=1,
        player_count=2,
        stack_shape=(4, 64, 64),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=1,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
        final_device_observation=final_device_observation,
        root_final_device_observation=final_device_observation.reshape(2, 4, 64, 64),
        final_observation_row_mask=np.asarray([True], dtype=np.bool_),
    )
    batch = _make_compact_batch(
        observation=observation,
        action_mask=np.ones((1, 2, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((1, 2), dtype=np.float32),
        final_reward_map=np.zeros((1, 2), dtype=np.float32),
        done=np.asarray([True], dtype=np.bool_),
        policy_env_id=np.asarray([0, 1], dtype=np.int32),
        policy_env_row=np.asarray([0, 0], dtype=np.int32),
        policy_player=np.asarray([0, 1], dtype=np.int32),
        target_reward=np.zeros((2, 1), dtype=np.float32),
        done_root=np.ones((2,), dtype=np.bool_),
        to_play=np.full((2,), -1, dtype=np.int64),
        active_root_mask=np.zeros((2,), dtype=np.bool_),
        final_observation=None,
        terminal_global_rows=np.asarray([0], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.zeros((1,), dtype=np.int32),
        elapsed_ms=np.zeros((1,), dtype=np.float64),
        round_id=np.zeros((1,), dtype=np.int32),
        alive=np.zeros((1, 2), dtype=np.bool_),
        joint_action=np.zeros((1, 2), dtype=np.int16),
        batch_size=1,
        resident_observation=resident,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    )

    root_batch = build_compact_root_batch_v1(
        batch,
        search_lane="unit_test_resident_terminal",
    )

    assert root_batch.metadata["resident_final_device_observation_present"] is True
    assert root_batch.metadata["resident_final_observation_row_count"] == 1
    assert root_batch.final_observation is None
    np.testing.assert_array_equal(root_batch.final_observation_row_mask, [True])


def test_owner_mechanics_step_frame_handle_digest_mismatch_fails_closed():
    import curvyzero.training.compact_rollout_slab as slab_module

    torch = pytest.importorskip("torch")
    resident = SimpleNamespace(
        device_observation=torch.zeros((1, 2, 4, 64, 64), dtype=torch.uint8),
        root_device_observation=torch.zeros((2, 4, 64, 64), dtype=torch.uint8),
        generation_id=1,
        batch_size=1,
        player_count=2,
        stack_shape=(4, 64, 64),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=1,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
    )
    view = hybrid_profile._make_compact_owner_mechanics_step_view(
        action_mask=np.ones((1, 2, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((1, 2), dtype=np.float32),
        final_reward_map=np.zeros((1, 2), dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        terminated=np.asarray([False], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        terminal_reason=np.asarray([0], dtype=np.int16),
        death_count=np.zeros((1,), dtype=np.int32),
        death_player=np.full((1, 2), -1, dtype=np.int16),
        death_cause=np.zeros((1, 2), dtype=np.int16),
        death_hit_owner=np.full((1, 2), -1, dtype=np.int16),
        winner=np.full((1,), -1, dtype=np.int16),
        draw=np.asarray([False], dtype=np.bool_),
        policy_env_id=np.asarray([0, 1], dtype=np.int32),
        policy_env_row=np.asarray([0, 0], dtype=np.int32),
        policy_player=np.asarray([0, 1], dtype=np.int32),
        target_reward=np.zeros((2, 1), dtype=np.float32),
        done_root=np.asarray([False, False], dtype=np.bool_),
        to_play=np.full((2,), -1, dtype=np.int64),
        active_root_mask=np.asarray([True, True], dtype=np.bool_),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.zeros((1,), dtype=np.int32),
        elapsed_ms=np.zeros((1,), dtype=np.float64),
        round_id=np.zeros((1,), dtype=np.int32),
        alive=np.ones((1, 2), dtype=np.bool_),
        joint_action=np.zeros((1, 2), dtype=np.int16),
        batch_size=1,
        resident_observation=resident,
        step_frame_slot_id=0,
        step_frame_generation=0,
    )
    corrupted = replace(
        view,
        metadata={
            **view.metadata,
            "compact_owner_mechanics_step_frame_handle_digest": "stale-digest",
        },
    )

    with pytest.raises(
        ReplayCompatibilityError,
        match="owner mechanics step frame metadata digest mismatch",
    ):
        slab_module._owner_mechanics_step_frame_handle_metadata_v1(corrupted)


def test_owner_mechanics_step_frame_handle_stale_generation_fails_before_submit():
    import curvyzero.training.compact_rollout_slab as slab_module

    torch = pytest.importorskip("torch")

    class NoSubmitSearchService:
        supports_two_phase_compact_search = True

        def __init__(self) -> None:
            self.submit_count = 0
            self.resolve_count = 0
            self.replay_append_count = 0
            self.proxy_closure_count = 0

        def run_action_step(self, root_batch):
            del root_batch
            raise AssertionError("stale step frame must not run search")

        def flush_replay_payload(self, handle):
            del handle
            raise AssertionError("stale step frame must not flush replay")

        def submit_action_step_from_root_build_request(self, root_build_request):
            del root_build_request
            self.submit_count += 1
            raise AssertionError("stale step frame must not submit search")

        def resolve_action_step_handle(self, handle, *, sync_wrapper=False):
            del handle, sync_wrapper
            self.resolve_count += 1
            raise AssertionError("stale step frame must not resolve search")

        def stage_replay_append_entries(self, entry):
            del entry
            self.replay_append_count += 1
            raise AssertionError("stale step frame must not stage replay")

        def stage_owner_proxy_transition_from_root_build_request(self, root_request, **kwargs):
            del root_request, kwargs
            self.proxy_closure_count += 1
            raise AssertionError("stale step frame must not close proxy transition")

    resident = SimpleNamespace(
        device_observation=torch.zeros((1, 2, 4, 64, 64), dtype=torch.uint8),
        root_device_observation=torch.zeros((2, 4, 64, 64), dtype=torch.uint8),
        generation_id=1,
        batch_size=1,
        player_count=2,
        stack_shape=(4, 64, 64),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=1,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
    )
    ring = hybrid_profile._make_compact_owner_mechanics_step_frame_ring(
        batch_size=1,
        player_count=2,
    )
    batch = hybrid_profile._publish_compact_owner_mechanics_step_frame_slot(
        ring=ring,
        action_mask=np.ones((1, 2, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((1, 2), dtype=np.float32),
        final_reward_map=np.zeros((1, 2), dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        terminated=np.asarray([False], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        terminal_reason=np.asarray([0], dtype=np.int16),
        death_count=np.zeros((1,), dtype=np.int32),
        death_player=np.full((1, 2), -1, dtype=np.int16),
        death_cause=np.zeros((1, 2), dtype=np.int16),
        death_hit_owner=np.full((1, 2), -1, dtype=np.int16),
        winner=np.full((1,), -1, dtype=np.int16),
        draw=np.asarray([False], dtype=np.bool_),
        policy_env_id=np.asarray([0, 1], dtype=np.int32),
        policy_env_row=np.asarray([0, 0], dtype=np.int32),
        policy_player=np.asarray([0, 1], dtype=np.int32),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.zeros((1,), dtype=np.int32),
        elapsed_ms=np.zeros((1,), dtype=np.float64),
        round_id=np.zeros((1,), dtype=np.int32),
        alive=np.ones((1, 2), dtype=np.bool_),
        joint_action=np.zeros((1, 2), dtype=np.int16),
        batch_size=1,
        resident_observation=resident,
        step_frame_slot_id=0,
        step_frame_generation=0,
    )
    metadata_generation_mismatch = replace(
        batch,
        metadata={
            **batch.metadata,
            "compact_owner_mechanics_step_frame_handle_generation": 1,
        },
    )
    with pytest.raises(
        ReplayCompatibilityError,
        match="owner mechanics step frame metadata generation mismatch",
    ):
        slab_module._owner_mechanics_step_frame_handle_metadata_v1(
            metadata_generation_mismatch
        )

    consumed_probe_stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=1,
        player_count=2,
        search_service=NoSubmitSearchService(),
        search_lane="unit_test_consumed_step_frame",
        policy_source="unit_test_consumed_step_frame",
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )
    try:
        root_metadata = slab_module._owner_mechanics_step_frame_handle_metadata_v1(batch)
        consumed_probe_stepper._validate_owner_mechanics_step_frame_unconsumed(
            root_metadata
        )
        consumed_probe_stepper._mark_owner_mechanics_step_frame_consumed(root_metadata)
        with pytest.raises(
            ReplayCompatibilityError,
            match="owner mechanics step frame stale generation",
        ):
            consumed_probe_stepper._validate_owner_mechanics_step_frame_unconsumed(
                root_metadata
            )
    finally:
        consumed_probe_stepper.close()

    ring["slot_generation"][0] = 4
    search_service = NoSubmitSearchService()
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=1,
        player_count=2,
        search_service=search_service,
        search_lane="unit_test_stale_step_frame",
        policy_source="unit_test_stale_step_frame",
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )
    try:
        pending_sentinel = SimpleNamespace(
            compact_batch=None,
            root_batch=None,
            search_result=None,
            action_step=None,
            next_joint_action=None,
        )
        stepper._pending = pending_sentinel
        record_index_before = stepper._record_index
        with pytest.raises(
            ReplayCompatibilityError,
            match="owner mechanics step frame stale generation",
        ):
            stepper.submit_step(batch)
        assert stepper._pending is pending_sentinel
        assert stepper._record_index == record_index_before
        assert search_service.submit_count == 0
        assert search_service.resolve_count == 0
        assert search_service.replay_append_count == 0
        assert search_service.proxy_closure_count == 0
        assert stepper._pending_direct_step_dispatch is None
        assert stepper.metadata[
            "compact_rollout_slab_action_dispatch_step_overlap_submit_count"
        ] == 0
    finally:
        stepper.close()


def test_direct_stepper_slot_transaction_pending_does_not_store_compact_batch(
    monkeypatch,
):
    import curvyzero.training.compact_owner_search_service as owner_service_module
    import curvyzero.training.compact_rollout_slab as slab_module

    torch = pytest.importorskip("torch")

    class SlotTransactionSearchService:
        supports_two_phase_compact_search = True

        def __init__(self) -> None:
            self.submit_count = 0

        def run_action_step(self, root_batch):
            del root_batch
            raise AssertionError("pending storage guard must not pass a root batch")

        def submit_owner_root_search_transaction_from_step_frame_slot(
            self,
            compact_batch,
            *,
            search_lane,
            metadata=None,
            copy_observation=False,
            resident_host_observation_stub=True,
            close_previous_transition=False,
            max_entries_per_batch=0,
            policy_source="",
        ):
            del close_previous_transition, max_entries_per_batch, policy_source
            self.submit_count += 1
            root_request = (
                owner_service_module._owner_root_search_transaction_request_from_step_frame_slot_v1(
                    compact_batch,
                    search_lane=search_lane,
                    metadata={
                        **dict(metadata or {}),
                        "compact_owner_root_search_transaction_requested": True,
                        "compact_owner_root_search_transaction_used": True,
                        "compact_owner_root_search_transaction_id": self.submit_count,
                    },
                    copy_observation=copy_observation,
                    resident_host_observation_stub=resident_host_observation_stub,
                )
            )
            root_action_context = compact_root_action_context_v1_from_request(root_request)
            root_action_context_handle = SimpleNamespace(
                schema_id="curvyzero_compact_owner_root_action_context_handle/v1",
                context_id=self.submit_count,
                transaction_id=self.submit_count,
                dispatch_id=self.submit_count,
                root_count=int(root_action_context.root_count),
                active_root_count=int(root_action_context.active_root_index.size),
                context_digest="unit-test-context-digest",
                metadata={
                    "compact_owner_root_action_context_handle_used": True,
                    "compact_owner_root_action_context_root_count": int(
                        root_action_context.root_count
                    ),
                },
            )
            return SimpleNamespace(
                action_dispatch_handle=SimpleNamespace(dispatch_id=self.submit_count),
                root_action_context_handle=root_action_context_handle,
                commit_timing={},
                metadata={
                    **dict(root_request.metadata),
                    **dict(root_action_context_handle.metadata),
                    "compact_owner_root_search_transaction_begin_count": self.submit_count,
                    "compact_owner_root_search_transaction_submit_count": self.submit_count,
                    "compact_owner_root_search_transaction_pending_count": 1,
                    "compact_owner_root_search_transaction_max_pending_count": 1,
                    "compact_owner_root_search_transaction_owner_root_request_build_count": (
                        self.submit_count
                    ),
                    "compact_owner_root_search_transaction_owner_root_store_publish_count": (
                        self.submit_count
                    ),
                },
            )

        def resolve_action_step_handle(self, handle, *, sync_wrapper=False):
            del handle, sync_wrapper
            raise AssertionError("pending storage guard must not resolve")

        def flush_replay_payload(self, handle):
            del handle
            raise AssertionError("pending storage guard must not flush replay")

    def fail_parent_step_frame_root_request_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent step-frame root request builder must not be called")

    monkeypatch.setattr(
        slab_module,
        "_root_build_request_from_owner_step_frame_slot_v1",
        fail_parent_step_frame_root_request_builder,
    )
    resident = SimpleNamespace(
        device_observation=torch.zeros((1, 2, 4, 64, 64), dtype=torch.uint8),
        root_device_observation=torch.zeros((2, 4, 64, 64), dtype=torch.uint8),
        generation_id=11,
        batch_size=1,
        player_count=2,
        stack_shape=(4, 64, 64),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=11,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
    )
    ring = hybrid_profile._make_compact_owner_mechanics_step_frame_ring(
        batch_size=1,
        player_count=2,
    )
    batch = hybrid_profile._publish_compact_owner_mechanics_step_frame_slot(
        ring=ring,
        action_mask=np.ones((1, 2, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((1, 2), dtype=np.float32),
        final_reward_map=np.zeros((1, 2), dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        terminated=np.asarray([False], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        terminal_reason=np.asarray([0], dtype=np.int16),
        death_count=np.zeros((1,), dtype=np.int32),
        death_player=np.full((1, 2), -1, dtype=np.int16),
        death_cause=np.zeros((1, 2), dtype=np.int16),
        death_hit_owner=np.full((1, 2), -1, dtype=np.int16),
        winner=np.full((1,), -1, dtype=np.int16),
        draw=np.asarray([False], dtype=np.bool_),
        policy_env_id=np.asarray([0, 1], dtype=np.int32),
        policy_env_row=np.asarray([0, 0], dtype=np.int32),
        policy_player=np.asarray([0, 1], dtype=np.int32),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.zeros((1,), dtype=np.int32),
        elapsed_ms=np.zeros((1,), dtype=np.float64),
        round_id=np.zeros((1,), dtype=np.int32),
        alive=np.ones((1, 2), dtype=np.bool_),
        joint_action=np.zeros((1, 2), dtype=np.int16),
        batch_size=1,
        resident_observation=resident,
        step_frame_slot_id=0,
        step_frame_generation=0,
    )
    search_service = SlotTransactionSearchService()
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=1,
        player_count=2,
        search_service=search_service,
        search_lane="unit_test_slot_transaction_pending",
        policy_source="unit_test_slot_transaction_pending",
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )
    try:
        handle = stepper.submit_step(batch)
        assert handle.dispatch_id == 1
        assert search_service.submit_count == 1
        assert stepper._pending_direct_step_dispatch is not None
        assert stepper._pending_direct_step_dispatch.compact_batch is None
        assert stepper._pending_direct_step_dispatch.root_action_context is None
        assert stepper._pending_direct_step_dispatch.root_action_context_handle.root_count == 2
        assert (
            stepper._pending_direct_step_dispatch.root_metadata[
                "compact_owner_root_search_transaction_parent_compact_batch_stored"
            ]
            is False
        )
        assert (
            stepper.metadata["compact_owner_search_pending_root_action_context_stored"]
            is False
        )
        assert not hasattr(stepper._pending_direct_step_dispatch, "root_build_request")
        with pytest.raises(ReplayCompatibilityError, match="dispatch pending at close"):
            stepper.close()
    finally:
        if not stepper._closed:
            stepper._pending_direct_step_dispatch = None
            stepper.close()


def test_compact_action_step_must_match_current_batch_identity():
    batch = _make_compact_batch(
        observation=np.zeros((2, 2, 4, 64, 64), dtype=np.float32),
        action_mask=np.ones((2, 2, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((2, 2), dtype=np.float32),
        final_reward_map=np.zeros((2, 2), dtype=np.float32),
        done=np.zeros((2,), dtype=np.bool_),
        policy_env_id=np.asarray([0, 1, 2, 3], dtype=np.int32),
        policy_env_row=np.asarray([0, 0, 1, 1], dtype=np.int32),
        policy_player=np.asarray([0, 1, 0, 1], dtype=np.int32),
        target_reward=np.zeros((4, 1), dtype=np.float32),
        done_root=np.zeros((4,), dtype=np.bool_),
        to_play=np.full((4,), -1, dtype=np.int64),
        active_root_mask=np.ones((4,), dtype=np.bool_),
        final_observation=None,
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.zeros((2,), dtype=np.int32),
        elapsed_ms=np.zeros((2,), dtype=np.float64),
        round_id=np.zeros((2,), dtype=np.int32),
        alive=np.ones((2, 2), dtype=np.bool_),
        joint_action=np.zeros((2, 2), dtype=np.int16),
        batch_size=2,
    )
    action_step = SimpleNamespace(
        root_index=np.asarray([0, 1, 2, 3], dtype=np.int32),
        env_row=np.asarray([0, 0, 1, 1], dtype=np.int32),
        player=np.asarray([0, 1, 0, 1], dtype=np.int16),
        policy_env_id=np.asarray([0, 1, 2, 3], dtype=np.int64),
        selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
    )

    joint_action = _joint_action_from_compact_search_action_step(batch, action_step)
    np.testing.assert_array_equal(
        joint_action,
        np.asarray([[0, 1], [2, 0]], dtype=np.int16),
    )

    stale_action_step = SimpleNamespace(
        **{**action_step.__dict__, "policy_env_id": np.asarray([0, 1, 3, 2])}
    )
    with pytest.raises(ValueError, match="policy_env_id"):
        _joint_action_from_compact_search_action_step(batch, stale_action_step)

    reordered_action_step = SimpleNamespace(
        **{**action_step.__dict__, "root_index": np.asarray([1, 0, 2, 3])}
    )
    with pytest.raises(ValueError, match="roots"):
        _joint_action_from_compact_search_action_step(batch, reordered_action_step)


def test_trainer_replay_step_preserves_final_reward_map():
    observation = np.zeros((2, 2, 4, 64, 64), dtype=np.float32)
    action_mask = np.ones((2, 2, ACTION_COUNT), dtype=np.bool_)
    reward = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    final_reward_map = reward + 20.0
    step = HybridObservationProfileStep(
        observation=observation,
        timestep=None,
        flat_obs=observation.reshape(4, 4, 64, 64),
        target_reward=reward.reshape(4, 1),
        policy_env_id=np.asarray([0, 1, 2, 3], dtype=np.int32),
        policy_env_row=np.asarray([0, 0, 1, 1], dtype=np.int32),
        policy_player=np.asarray([0, 1, 0, 1], dtype=np.int32),
        reward=reward,
        final_reward_map=final_reward_map,
        done=np.asarray([False, True], dtype=np.bool_),
        terminated=np.asarray([False, True], dtype=np.bool_),
        truncated=np.asarray([False, False], dtype=np.bool_),
        terminal_reason=np.asarray([0, 1], dtype=np.int16),
        death_count=np.asarray([0, 1], dtype=np.int32),
        death_player=np.asarray([[-1, -1], [0, -1]], dtype=np.int16),
        death_cause=np.asarray([[0, 0], [1, 0]], dtype=np.int16),
        death_hit_owner=np.asarray([[-1, -1], [-1, -1]], dtype=np.int16),
        winner=np.asarray([-1, 1], dtype=np.int16),
        draw=np.asarray([False, False], dtype=np.bool_),
        action_mask=action_mask,
        payload={
            "alive": np.ones((2, 2), dtype=np.bool_),
            "joint_action": np.asarray([[0, 1], [2, 0]], dtype=np.int16),
        },
        timings={},
        batched_stack_probe_telemetry={},
        compact_rollout_slab_telemetry={},
        compact_rollout_slab_step=None,
        compact_batch=None,
        compact_payload_bytes=0,
    )

    replay_step = _trainer_replay_step_from_hybrid_step(step)

    np.testing.assert_array_equal(replay_step.final_reward_map, final_reward_map)
    np.testing.assert_array_equal(replay_step.terminated, [False, True])
    np.testing.assert_array_equal(replay_step.truncated, [False, False])
    np.testing.assert_array_equal(replay_step.info["death_count"], [0, 1])


def test_hybrid_profile_manager_can_use_profile_only_compact_rollout_slab():
    class FakeSearchService:
        search_impl = "unit_test_manager_compact_rollout_slab"
        num_simulations = 0

        def run(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            selected = np.argmax(root_batch.legal_mask[active_roots], axis=1).astype(
                np.int16,
                copy=False,
            )
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata={
                    "compact_owner_search_slab_proxy": True,
                    "compact_owner_search_lazy_slab_proxy": True,
                    "compact_owner_search_slab_proxy_initialized": True,
                    "compact_owner_search_boundary_kind": (
                        "worker_search_parent_slab_commit"
                    ),
                    "compact_owner_search_parent_slab_commits_replay": True,
                    "compact_owner_search_owner_pid": 123,
                    "compact_owner_search_request_bytes": 55,
                    "compact_owner_search_result_bytes": 88,
                    "compact_owner_search_request_cuda_tensor_count": 0,
                    "compact_owner_search_result_cuda_tensor_count": 0,
                    "compact_owner_search_root_observation_bytes_sent": 0,
                    "compact_owner_search_parent_reconstructed_search_result": True,
                    "compact_owner_search_model_state_bytes": 0,
                    "compact_owner_search_model_state_return_count": 0,
                    "compact_owner_search_model_state_snapshot_return_count": 0,
                    "compact_owner_search_search_result_payload_bytes": 77,
                    "compact_owner_search_search_result_payload_transport_kind": (
                        "numpy_ndarray_ipc_v1"
                    ),
                    "compact_owner_search_search_result_payload_json_safe": False,
                    "compact_owner_search_selected_action_bytes": 8,
                    "compact_owner_search_visit_policy_bytes": 24,
                    "compact_owner_search_root_value_bytes": 8,
                    "compact_owner_search_optional_array_bytes": 24,
                    "compact_owner_search_worker_owns_search_state": True,
                    "compact_owner_search_worker_owns_replay_state": False,
                    "compact_owner_search_worker_owns_model_state": False,
                    "compact_owner_search_consumed_learner_update": True,
                    "compact_owner_search_search_refresh_update_count": 0,
                    "compact_owner_search_parent_wait_sec": 0.01,
                    "compact_owner_search_worker_wall_sec": 0.02,
                    "compact_owner_search_worker_search_sec": 0.015,
                    "compact_owner_search_resident_root_bridge_ready": True,
                    "compact_owner_search_resident_root_bridge_kind": (
                        "shared_memory_host_root_to_owner_resident_tensor_v1"
                    ),
                    "compact_owner_search_resident_root_bridge_device": "cpu",
                    "compact_owner_search_resident_root_bridge_h2d_bytes": 1024.0,
                    "compact_owner_search_resident_root_bridge_generation_id": 7,
                    "owner_search_compact_torch_resident_root_bridge_ready": True,
                    "profile_telemetry": {
                        "lightzero_mcts_arrays_boundary_total_sec": 0.5,
                        "lightzero_mcts_arrays_boundary_initial_inference_sec": 0.1,
                        "lightzero_mcts_arrays_boundary_recurrent_inference_sec": 0.2,
                        "lightzero_mcts_arrays_boundary_search_sec": 0.3,
                        "lightzero_consumer_h2d_sec": 0.05,
                        "compact_torch_search_service_action_preamble_sec": 0.07,
                        "compact_torch_search_service_fixed_shape_masks_sec": 0.01,
                        "compact_torch_search_service_compile_eligibility_sec": 0.02,
                        "compact_torch_search_service_helper_cache_sec": 0.03,
                        "compact_torch_search_service_model_cache_sec": 0.04,
                        "compact_torch_search_service_metadata_build_sec": 0.005,
                        "compact_torch_search_service_pending_replay_store_sec": 0.006,
                        "compact_torch_search_service_action_step_build_sec": 0.007,
                        "compact_torch_search_service_action_postprocess_sec": 0.018,
                        "compact_torch_search_service_action_wall_sec": 0.588,
                        "compact_torch_search_service_action_unaccounted_sec": 0.0,
                    }
                },
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeSearchService(),
        search_lane="unit_test_manager_compact_rollout_slab",
        policy_source="unit_test_manager_compact_rollout_slab",
    )
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=20260523,
            materialize_scalar_timestep=False,
        ),
        compact_rollout_slab=slab,
    )

    first = manager.step(np.asarray([[0, 1], [2, 0]], dtype=np.int16))
    assert first.compact_batch is not None
    assert first.compact_rollout_slab_step is not None
    assert first.compact_rollout_slab_step.committed_index_rows is None
    assert first.compact_rollout_slab_telemetry["compact_rollout_slab_profile_only"] is True
    assert (
        first.compact_rollout_slab_telemetry["compact_rollout_slab_internal_accounted_sec"] >= 0.0
    )
    assert (
        first.compact_rollout_slab_telemetry[
            "compact_rollout_slab_owner_search_parent_wait_sec"
        ]
        == pytest.approx(0.01)
    )
    assert (
        first.compact_rollout_slab_telemetry[
            "compact_rollout_slab_owner_search_worker_wall_sec"
        ]
        == pytest.approx(0.02)
    )
    assert (
        first.compact_rollout_slab_telemetry[
            "compact_rollout_slab_owner_search_worker_search_sec"
        ]
        == pytest.approx(0.015)
    )
    assert first.timings["compact_rollout_slab_sec"] >= 0.0

    second = manager.step(first.compact_rollout_slab_step.next_joint_action)
    assert second.compact_rollout_slab_step is not None
    committed = second.compact_rollout_slab_step.committed_index_rows
    assert committed is not None
    assert committed.metadata["observation_materialized"] is False
    np.testing.assert_array_equal(
        committed.action,
        first.compact_rollout_slab_step.search_result.selected_action,
    )


def test_hybrid_profile_run_can_use_compact_rollout_slab_without_scalar_rows():
    compact_torch_profile_fields = {
        "compact_torch_search_service_action_preamble_sec": 0.07,
        "compact_torch_search_service_fixed_shape_masks_sec": 0.01,
        "compact_torch_search_service_compile_eligibility_sec": 0.02,
        "compact_torch_search_service_helper_cache_sec": 0.03,
        "compact_torch_search_service_model_cache_sec": 0.04,
        "compact_torch_search_service_metadata_build_sec": 0.005,
        "compact_torch_search_service_pending_replay_store_sec": 0.006,
        "compact_torch_search_service_action_step_build_sec": 0.007,
        "compact_torch_search_service_action_postprocess_sec": 0.018,
        "compact_torch_search_service_action_wall_sec": 0.588,
        "compact_torch_search_service_action_accounted_sec": 0.560,
        "compact_torch_search_service_action_residual_sec": 0.028,
        "compact_torch_search_service_action_unaccounted_sec": 0.0,
        "compact_torch_search_service_action_overaccounted_sec": 0.0,
        "compact_torch_search_service_tensor_prepare_sync_sec": 0.001,
        "compact_torch_search_service_initial_output_decode_sec": 0.010,
        "compact_torch_search_service_root_output_decode_sec": 0.010,
        "compact_torch_search_service_initial_inference_enqueue_sec": 0.011,
        "compact_torch_search_service_initial_inference_sync_sec": 0.012,
        "compact_torch_search_service_initial_inference_cuda_event_sec": 0.013,
        "compact_torch_search_service_tree_setup_sec": 0.0,
        "compact_torch_search_service_tree_root_prior_build_sec": 0.020,
        "compact_torch_search_service_tree_root_prior_select_sec": 0.021,
        "compact_torch_search_service_tree_select_enqueue_sec": 0.0,
        "compact_torch_search_service_tree_recurrent_action_build_sec": 0.022,
        "compact_torch_search_service_tree_recurrent_inference_enqueue_sec": 0.023,
        "compact_torch_search_service_tree_recurrent_inference_cuda_event_sec": 0.014,
        "compact_torch_search_service_tree_recurrent_output_decode_sec": 0.024,
        "compact_torch_search_service_tree_backup_enqueue_sec": 0.0,
        "compact_torch_search_service_tree_policy_build_sec": 0.025,
        "compact_torch_search_service_tree_sync_sec": 0.026,
        "compact_torch_search_service_tree_cuda_event_sec": 0.015,
        "compact_torch_search_service_tree_total_sec": 0.150,
        "compact_torch_search_service_tree_accounted_sec": 0.140,
        "compact_torch_search_service_tree_residual_sec": 0.010,
        "compact_torch_search_service_tree_unaccounted_sec": 0.0,
        "compact_torch_search_service_tree_overaccounted_sec": 0.0,
        "compact_torch_search_service_action_readback_sec": 0.009,
        "compact_torch_search_service_core_accounted_sec": 0.300,
        "compact_torch_search_service_core_residual_sec": 0.002,
        "compact_torch_search_service_core_unaccounted_sec": 0.002,
        "compact_torch_search_service_core_overaccounted_sec": 0.0,
        "compact_torch_search_service_cuda_event_timing_enabled": 1.0,
        "compact_torch_search_service_initial_sync_enabled": 1.0,
        "compact_torch_search_one_simulation_fast_path": True,
        "compact_torch_search_one_simulation_root_prior_softmax_skipped": True,
        "compact_torch_search_recurrent_inference_calls": 1.0,
    }

    class FakeSearchService:
        search_impl = "unit_test_run_compact_rollout_slab"
        num_simulations = 0

        def run(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            selected = np.argmax(root_batch.legal_mask[active_roots], axis=1).astype(
                np.int16,
                copy=False,
            )
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata={
                    "compact_owner_search_slab_proxy": True,
                    "compact_owner_search_lazy_slab_proxy": True,
                    "compact_owner_search_slab_proxy_initialized": True,
                    "compact_owner_search_boundary_kind": (
                        "worker_search_parent_slab_commit"
                    ),
                    "compact_owner_search_parent_slab_commits_replay": True,
                    "compact_owner_search_owner_pid": 123,
                    "compact_owner_search_request_bytes": 55,
                    "compact_owner_search_result_bytes": 88,
                    "compact_owner_search_request_cuda_tensor_count": 0,
                    "compact_owner_search_result_cuda_tensor_count": 0,
                    "compact_owner_search_root_observation_bytes_sent": 0,
                    "compact_owner_search_parent_reconstructed_search_result": True,
                    "compact_owner_search_model_state_bytes": 0,
                    "compact_owner_search_model_state_return_count": 0,
                    "compact_owner_search_model_state_snapshot_return_count": 0,
                    "compact_owner_search_search_result_payload_bytes": 77,
                    "compact_owner_search_search_result_payload_transport_kind": (
                        "numpy_ndarray_ipc_v1"
                    ),
                    "compact_owner_search_search_result_payload_json_safe": False,
                    "compact_owner_search_selected_action_bytes": 8,
                    "compact_owner_search_visit_policy_bytes": 24,
                    "compact_owner_search_root_value_bytes": 8,
                    "compact_owner_search_optional_array_bytes": 24,
                    "compact_owner_search_worker_owns_search_state": True,
                    "compact_owner_search_worker_owns_replay_state": False,
                    "compact_owner_search_worker_owns_model_state": False,
                    "compact_owner_search_consumed_learner_update": True,
                    "compact_owner_search_search_refresh_update_count": 0,
                    "compact_owner_search_parent_wait_sec": 0.01,
                    "compact_owner_search_worker_wall_sec": 0.02,
                    "compact_owner_search_worker_search_sec": 0.015,
                    "compact_owner_search_resident_root_bridge_ready": True,
                    "compact_owner_search_resident_root_bridge_kind": (
                        "shared_memory_host_root_to_owner_resident_tensor_v1"
                    ),
                    "compact_owner_search_resident_root_bridge_device": "cpu",
                    "compact_owner_search_resident_root_bridge_h2d_bytes": 1024.0,
                    "compact_owner_search_resident_root_bridge_generation_id": 7,
                    "owner_search_compact_torch_resident_root_bridge_ready": True,
                    "profile_telemetry": {
                        "lightzero_mcts_arrays_boundary_total_sec": 0.5,
                        "lightzero_mcts_arrays_boundary_initial_inference_sec": 0.1,
                        "lightzero_mcts_arrays_boundary_recurrent_inference_sec": 0.2,
                        "lightzero_mcts_arrays_boundary_search_sec": 0.3,
                        "lightzero_consumer_h2d_sec": 0.05,
                        **compact_torch_profile_fields,
                    }
                },
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeSearchService(),
        search_lane="unit_test_run_compact_rollout_slab",
        policy_source="unit_test_run_compact_rollout_slab",
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=20260523,
            materialize_scalar_timestep=False,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=2,
            compact_rollout_slab_sample_gate_interval=2,
        ),
        compact_rollout_slab=slab,
    )

    assert result["compact_rollout_slab_enabled"] is True
    assert result["compact_rollout_slab_profile_only"] is True
    assert result["compact_rollout_slab_calls"] == 2
    assert result["compact_rollout_slab_total_roots"] > 0
    assert result["compact_rollout_slab_committed_index_row_count"] > 0
    assert result["compact_rollout_slab_stored_index_group_count"] == 2
    assert (
        result["compact_rollout_slab_stored_index_row_count"]
        == result["compact_rollout_slab_committed_index_row_count"]
    )
    assert result["compact_rollout_slab_dropped_pending_search_count"] == 1
    assert result["materialized_timestep_count"] == 0
    assert result["contract"]["compact_rollout_slab"] is True
    assert result["contract"]["compact_rollout_slab_sample_gate"] is True
    assert result["contract"]["compact_rollout_slab_sample_gate_batch_size"] == 2
    assert result["contract"]["compact_rollout_slab_sample_gate_interval"] == 2
    assert result["compact_rollout_slab_sample_gate_enabled"] is True
    assert result["compact_rollout_slab_sample_gate_calls"] == 1
    assert result["compact_rollout_slab_sample_gate_opportunities"] == 2
    assert result["compact_rollout_slab_sample_gate_skipped_count"] == 1
    assert result["compact_rollout_slab_sample_gate_replay_ring_entry_count"] == 2
    assert (
        result["compact_rollout_slab_sample_gate_index_row_count"]
        == result["compact_rollout_slab_sample_gate_replay_ring_index_row_count"]
    )
    assert result["compact_rollout_slab_sample_gate_target_row_count"] > 0
    assert result["compact_rollout_slab_sample_gate_batch_size"] == 2
    assert result["compact_rollout_slab_sample_gate_interval"] == 2
    assert result["compact_rollout_slab_sample_gate_sample_row_count"] == 2
    assert result["compact_rollout_slab_sample_gate_mock_base_env_timestep_rows"] == 0
    assert result["last_flat_obs_shape"] == [0, 4, 64, 64]
    assert result["last_target_reward_shape"] == [0, 1]
    assert result["compact_rollout_slab_sample_gate_last_telemetry"][
        "compact_rollout_slab_sample_gate_observation_shape"
    ][1:] == [4, 64, 64]
    assert result["compact_rollout_slab_sample_gate_last_telemetry"][
        "compact_rollout_slab_sample_gate_next_observation_shape"
    ][1:] == [4, 64, 64]
    assert (
        result["compact_rollout_slab_sample_gate_last_telemetry"][
            "compact_rollout_slab_sample_gate_requested_sample_row_count"
        ]
        == 2
    )
    assert (
        result["compact_rollout_slab_sample_gate_last_telemetry"][
            "compact_rollout_slab_sample_gate_mode"
        ]
        == "compact_replay_ring_fast_sample_batch"
    )
    assert (
        result["compact_rollout_slab_sample_gate_last_telemetry"][
            "compact_rollout_slab_sample_gate_stored_pair_count"
        ]
        == 2
    )
    assert (
        result["compact_rollout_slab_sample_gate_last_telemetry"][
            "compact_rollout_slab_sample_gate_stored_index_row_count"
        ]
        == result["compact_rollout_slab_sample_gate_replay_ring_index_row_count"]
    )
    assert (
        result["compact_rollout_slab_last_telemetry"]["compact_rollout_slab_search_impl"]
        == "unit_test_run_compact_rollout_slab"
    )
    assert result["compact_owner_search_slab_proxy"] is True
    assert result["compact_owner_search_lazy_slab_proxy"] is True
    assert result["compact_owner_search_boundary_kind"] == "worker_search_parent_slab_commit"
    assert result["compact_owner_search_parent_slab_commits_replay"] is True
    assert result["compact_owner_search_request_bytes"] == 55
    assert result["compact_owner_search_result_bytes"] == 88
    assert result["compact_owner_search_request_cuda_tensor_count"] == 0
    assert result["compact_owner_search_result_cuda_tensor_count"] == 0
    assert result["compact_owner_search_root_observation_bytes_sent"] == 0
    assert result["compact_owner_search_parent_reconstructed_search_result"] is True
    assert result["compact_owner_search_model_state_bytes"] == 0
    assert result["compact_owner_search_model_state_return_count"] == 0
    assert result["compact_owner_search_search_result_payload_bytes"] == 77
    assert (
        result["compact_owner_search_search_result_payload_transport_kind"]
        == "numpy_ndarray_ipc_v1"
    )
    assert result["compact_owner_search_search_result_payload_json_safe"] is False
    assert result["compact_owner_search_selected_action_bytes"] == 8
    assert result["compact_owner_search_visit_policy_bytes"] == 24
    assert result["compact_owner_search_root_value_bytes"] == 8
    assert result["compact_owner_search_worker_owns_search_state"] is True
    assert result["compact_owner_search_worker_owns_replay_state"] is False
    assert result["compact_owner_search_consumed_learner_update"] is True
    assert result["compact_owner_search_parent_wait_sec"] == 0.01
    assert result["compact_owner_search_worker_wall_sec"] == 0.02
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_owner_search_parent_wait_sec"
    ] == pytest.approx(0.02)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_owner_search_worker_wall_sec"
    ] == pytest.approx(0.04)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_owner_search_worker_search_sec"
    ] == pytest.approx(0.03)
    assert result["compact_owner_search_resident_root_bridge_ready"] is True
    assert result["compact_owner_search_resident_root_bridge_kind"] == (
        "shared_memory_host_root_to_owner_resident_tensor_v1"
    )
    assert result["compact_owner_search_resident_root_bridge_device"] == "cpu"
    assert result["compact_owner_search_resident_root_bridge_h2d_bytes"] == 1024.0
    assert result["compact_owner_search_resident_root_bridge_generation_id"] == 7
    assert result["owner_search_compact_torch_resident_root_bridge_ready"] is True
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_total_sec"
    ] == pytest.approx(1.0)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_model_sec"
    ] == pytest.approx(0.6)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_h2d_sec"
    ] == pytest.approx(0.1)
    assert (
        result["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_internal_accounted_sec"
        ]
        >= 0.0
    )
    assert (
        result["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_dispatch_wall_sec"
        ]
        >= 0.0
    )
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_dispatch_service_envelope_sec"
    ] == pytest.approx(1.176)
    assert (
        result["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_dispatch_residual_sec"
        ]
        <= 0.0
    )
    assert (
        result["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_dispatch_positive_residual_sec"
        ]
        >= 0.0
    )
    assert (
        result["compact_rollout_slab_telemetry_totals"][
            "compact_rollout_slab_search_dispatch_overaccounted_sec"
        ]
        >= 0.0
    )
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_action_preamble_sec"
    ] == pytest.approx(0.14)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_action_wall_sec"
    ] == pytest.approx(1.176)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_action_accounted_sec"
    ] == pytest.approx(1.12)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_action_residual_sec"
    ] == pytest.approx(0.056)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_action_postprocess_sec"
    ] == pytest.approx(0.036)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_one_simulation_fast_path_count"
    ] == pytest.approx(2.0)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_one_simulation_root_prior_softmax_skipped"
    ] == pytest.approx(2.0)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_recurrent_inference_calls"
    ] == pytest.approx(2.0)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_initial_inference_sync_sec"
    ] == pytest.approx(0.024)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_initial_inference_cuda_event_sec"
    ] == pytest.approx(0.026)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_initial_output_decode_sec"
    ] == pytest.approx(0.020)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_tree_root_prior_build_sec"
    ] == pytest.approx(0.040)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_tree_root_prior_select_sec"
    ] == pytest.approx(0.042)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_tree_recurrent_inference_enqueue_sec"
    ] == pytest.approx(0.046)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_tree_recurrent_inference_cuda_event_sec"
    ] == pytest.approx(0.028)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_tree_sync_sec"
    ] == pytest.approx(0.052)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_tree_cuda_event_sec"
    ] == pytest.approx(0.03)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_tree_total_sec"
    ] == pytest.approx(0.300)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_core_accounted_sec"
    ] == pytest.approx(0.600)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_cuda_event_timing_enabled"
    ] == pytest.approx(2.0)
    assert result["compact_rollout_slab_telemetry_totals"][
        "compact_rollout_slab_search_service_initial_sync_enabled"
    ] == pytest.approx(2.0)


def test_hybrid_profile_guarded_owner_action_step_boundary_drives_next_action():
    class BoundarySearchService:
        search_impl = "unit_test_owner_action_step_boundary"
        num_simulations = 1

        def __init__(self) -> None:
            self.run_count = 0
            self.dense_actions: list[np.ndarray] = []
            self.sync_wrapper_count = 0

        def run(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            env_row = np.asarray(root_batch.env_row, dtype=np.int64)[active_roots]
            player = np.asarray(root_batch.player, dtype=np.int64)[active_roots]
            selected = (
                env_row + player + int(self.run_count)
            ).astype(np.int16, copy=False) % np.int16(ACTION_COUNT)
            dense = np.zeros((2, 2), dtype=np.int16)
            dense[env_row, player] = selected
            self.dense_actions.append(dense)
            self.run_count += 1
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata={
                    "compact_owner_search_action_only_result": True,
                    "compact_owner_search_owner_materializes_replay": True,
                    "compact_owner_search_parent_slab_commits_replay": False,
                    "compact_owner_search_parent_reconstructed_search_result": False,
                    "compact_owner_search_search_result_payload_bytes": 0,
                },
            )

    search_service = BoundarySearchService()
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=search_service,
        search_lane="unit_test_owner_action_step_boundary",
        policy_source="unit_test_owner_action_step_boundary",
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=20260608,
            materialize_scalar_timestep=False,
            compact_owner_action_step_boundary=True,
        ),
        compact_rollout_slab=slab,
    )

    assert result["compact_owner_action_step_boundary_enabled"] is True
    assert result["compact_owner_action_step_boundary_proof_passed"] is True
    assert result["compact_owner_action_step_boundary_step_count"] == 3
    assert result["compact_owner_action_step_boundary_seeded_action_count"] == 1
    assert result["compact_owner_action_step_boundary_feedback_action_count"] == 2
    assert result["compact_owner_action_step_boundary_action_verified_count"] == 3
    assert result["compact_owner_action_step_boundary_next_action_count"] == 3
    assert result["compact_owner_action_step_boundary_failure_reason"] == "none"
    assert result["compact_owner_action_step_boundary_last_action_source"] == "search_feedback"
    assert result["materialized_timestep_count"] == 0
    assert result["contract"]["compact_owner_action_step_boundary"] is True
    assert search_service.run_count == 3
    assert (
        result["compact_owner_action_step_boundary_last_applied_action_checksum"]
        == hybrid_profile._joint_action_checksum(search_service.dense_actions[1])
    )
    assert (
        result["compact_owner_action_step_boundary_last_next_action_checksum"]
        == hybrid_profile._joint_action_checksum(search_service.dense_actions[2])
    )


def test_hybrid_profile_owner_action_step_boundary_uses_direct_root_build_request(
    monkeypatch,
):
    import curvyzero.training.compact_rollout_slab as slab_module

    torch = pytest.importorskip("torch")

    def fail_parent_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent root batch builder must not be called")

    def fail_parent_compact_batch_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent compact batch builder must not be called")

    def fail_parent_mechanics_step_view_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent mechanics step view builder must not be called")

    def fail_parent_root_build_request_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent root build request builder must not be called")

    def fail_parent_step_frame_root_request_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent step-frame root request builder must not be called")

    def fail_parent_dense_action_reconstruction(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent dense action reconstruction must not be called")

    class BoundaryRootBuildRequestSearchService:
        search_impl = "unit_test_owner_boundary_root_build_request"
        num_simulations = 1

        def __init__(self) -> None:
            self.run_count = 0
            self.dense_actions: list[np.ndarray] = []
            self.sync_wrapper_count = 0

        def run(self, root_batch):
            assert root_batch.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
            assert root_batch.resident_observation is not None
            assert root_batch.metadata["compact_owner_search_direct_root_build_request_handoff"]
            assert root_batch.metadata["compact_owner_search_direct_root_parent_build_avoided"]
            assert root_batch.metadata["compact_owner_search_direct_root_owner_build_used"]
            assert root_batch.metadata["compact_owner_search_resident_root_view_proved"]
            assert (
                root_batch.metadata[
                    "compact_owner_search_direct_root_build_request_observation_included"
                ]
                is False
            )
            assert (
                root_batch.metadata[
                    "compact_owner_search_direct_root_build_request_observation_bytes_sent"
                ]
                == 0
            )
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            env_row = np.asarray(root_batch.env_row, dtype=np.int64)[active_roots]
            player = np.asarray(root_batch.player, dtype=np.int64)[active_roots]
            selected = (
                env_row + player + int(self.run_count)
            ).astype(np.int16, copy=False) % np.int16(ACTION_COUNT)
            dense = np.zeros((2, 2), dtype=np.int16)
            dense[env_row, player] = selected
            self.dense_actions.append(dense)
            self.run_count += 1
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata={
                    "compact_owner_search_direct_root_build_request_handoff": True,
                    "compact_owner_search_direct_root_parent_build_avoided": True,
                    "compact_owner_search_direct_root_owner_build_used": True,
                    "compact_owner_search_action_only_result": True,
                    "compact_owner_search_owner_materializes_replay": True,
                    "compact_owner_search_parent_slab_commits_replay": False,
                    "compact_owner_search_parent_reconstructed_search_result": False,
                    "compact_owner_search_search_result_payload_bytes": 0,
                    "compact_owner_search_visit_policy_bytes": 0,
                    "compact_owner_search_root_value_bytes": 0,
                    "compact_owner_search_action_dispatch_handle_sync_wrapper_count": int(
                        self.sync_wrapper_count
                    ),
                    "compact_owner_search_action_dispatch_handle_completed_at_submit_count": 0,
                    "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count": 0,
                },
            )

    search_service = BoundaryRootBuildRequestSearchService()
    monkeypatch.setattr(slab_module, "build_compact_root_batch_v1", fail_parent_builder)
    monkeypatch.setattr(
        hybrid_profile,
        "_make_compact_batch",
        fail_parent_compact_batch_builder,
    )
    monkeypatch.setattr(
        hybrid_profile,
        "_make_compact_owner_mechanics_step_view",
        fail_parent_mechanics_step_view_builder,
    )
    monkeypatch.setattr(
        slab_module,
        "compact_root_build_request_v1_from_batch",
        fail_parent_root_build_request_builder,
    )
    monkeypatch.setattr(
        slab_module,
        "_root_build_request_from_owner_step_frame_slot_v1",
        fail_parent_step_frame_root_request_builder,
    )
    monkeypatch.setattr(
        slab_module,
        "_selected_joint_action_from_compact_sidecars",
        fail_parent_dense_action_reconstruction,
    )

    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=lambda: search_service,
        owner_replay_append_enabled=False,
        owner_defer_maintenance=True,
        root_store_capacity=8,
        require_resident_root_view=True,
        fixed_action_result_buffer=True,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=2,
        player_count=2,
        search_service=proxy,
        search_lane="unit_test_owner_boundary_root_build_request",
        policy_source="unit_test_owner_boundary_root_build_request",
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )

    try:
        result = run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                steps=2,
                warmup_steps=1,
                seed=2026060801,
                stack_storage_dtype="uint8",
                update_host_observation_stack=False,
                resident_observation_search=True,
                materialize_scalar_timestep=False,
                native_actor_buffer=True,
                compact_owner_action_step_boundary=True,
                compact_owner_mechanics_step_boundary=True,
                compact_owner_minimal_step_payload_snapshot=True,
            ),
            observation_renderer=_PersistentBackendSentinelRenderer(),
            compact_rollout_slab=stepper,
        )
    finally:
        stepper.close()
        proxy.close()

    assert torch is not None
    assert result["compact_owner_action_step_boundary_enabled"] is True
    assert result["compact_owner_action_step_boundary_proof_passed"] is True
    assert result["compact_owner_action_step_boundary_step_count"] == 3
    assert result["compact_owner_action_step_boundary_seeded_action_count"] == 1
    assert result["compact_owner_action_step_boundary_feedback_action_count"] == 2
    assert result["compact_owner_action_step_boundary_action_verified_count"] == 3
    assert result["compact_owner_action_step_boundary_next_action_count"] == 3
    assert result["compact_owner_action_step_boundary_last_action_source"] == "search_feedback"
    assert result["compact_owner_action_step_boundary_failure_reason"] == "none"
    assert result["compact_owner_mechanics_step_boundary_enabled"] is True
    assert result["contract"]["compact_owner_mechanics_step_boundary"] is True
    assert (
        result["contract"]["compact_owner_mechanics_step_view_schema_id"]
        == hybrid_profile.COMPACT_OWNER_MECHANICS_STEP_VIEW_SCHEMA_ID
    )
    assert (
        result["contract"]["compact_owner_mechanics_step_frame_slot_schema_id"]
        == hybrid_profile.COMPACT_OWNER_MECHANICS_STEP_FRAME_SLOT_SCHEMA_ID
    )
    assert (
        result["contract"]["compact_owner_mechanics_step_frame_handle_schema_id"]
        == hybrid_profile.COMPACT_OWNER_MECHANICS_STEP_FRAME_HANDLE_SCHEMA_ID
    )
    assert (
        result["contract"]["compact_owner_mechanics_step_frame_handle_ring_slot_count"]
        == hybrid_profile.COMPACT_OWNER_MECHANICS_STEP_FRAME_HANDLE_RING_SLOT_COUNT
    )
    assert result["contract"]["compact_owner_minimal_step_payload_snapshot"] is True
    assert result["contract"]["compact_step_payload_snapshot_mode"] == "owner_minimal_v1"
    assert (
        result["timings"]["compact_owner_mechanics_step_boundary_count"]
        == pytest.approx(2.0)
    )
    assert (
        result["timings"][
            "compact_owner_mechanics_parent_compact_batch_builder_call_count"
        ]
        == pytest.approx(0.0)
    )
    assert (
        result["timings"]["compact_owner_mechanics_parent_compact_batch_object_count"]
        == pytest.approx(0.0)
    )
    assert (
        result["timings"]["compact_owner_mechanics_step_view_object_count"]
        == pytest.approx(0.0)
    )
    assert (
        result["timings"]["compact_owner_mechanics_step_frame_handle_publish_count"]
        == pytest.approx(2.0)
    )
    assert (
        result["timings"]["compact_owner_mechanics_step_frame_slot_write_count"]
        == pytest.approx(2.0)
    )
    assert (
        result["timings"]["compact_owner_mechanics_parent_step_frame_build_count"]
        == pytest.approx(0.0)
    )
    assert (
        result["timings"]["compact_owner_mechanics_host_observation_bytes_sent"]
        == pytest.approx(0.0)
    )
    assert (
        result["timings"]["compact_owner_mechanics_host_final_observation_bytes_sent"]
        == pytest.approx(0.0)
    )
    assert (
        result["timings"]["compact_step_payload_minimal_snapshot_count"]
        == pytest.approx(2.0)
    )
    assert result["timings"][
        "compact_step_payload_minimal_snapshot_full_bytes_elided"
    ] > 0.0
    assert (
        result["timings"][
            "compact_step_payload_minimal_snapshot_full_key_count_elided"
        ]
        > 0.0
    )
    assert (
        result["timings"][
            "compact_step_payload_minimal_snapshot_retained_key_count"
        ]
        == pytest.approx(16.0)
    )
    assert search_service.run_count == 3
    assert (
        result["compact_owner_action_step_boundary_last_applied_action_checksum"]
        == hybrid_profile._joint_action_checksum(search_service.dense_actions[1])
    )
    assert (
        result["compact_owner_action_step_boundary_last_next_action_checksum"]
        == hybrid_profile._joint_action_checksum(search_service.dense_actions[2])
    )
    assert result["compact_owner_search_direct_root_build_request_requested"] is True
    assert result["compact_owner_mechanics_step_boundary"] is True
    assert result["compact_owner_mechanics_step_view_schema_id"] == ""
    assert result["compact_owner_mechanics_step_boundary_count"] == 1
    assert (
        result["compact_owner_mechanics_parent_compact_batch_builder_call_count"]
        == 0
    )
    assert result["compact_owner_mechanics_parent_compact_batch_object_count"] == 0
    assert result["compact_owner_mechanics_parent_compact_batch_builder_used"] is False
    assert result["compact_owner_mechanics_step_view_object_count"] == 0
    assert result["compact_owner_mechanics_step_frame_slot_schema_id"] == (
        hybrid_profile.COMPACT_OWNER_MECHANICS_STEP_FRAME_SLOT_SCHEMA_ID
    )
    assert result["compact_owner_mechanics_step_frame_handle_ring_used"] is True
    assert result["compact_owner_mechanics_step_frame_slot_write_count"] == 1
    assert result["compact_owner_mechanics_parent_step_frame_build_count"] == 0
    assert result["compact_owner_mechanics_host_observation_bytes_sent"] == 0
    assert result["compact_owner_mechanics_host_final_observation_bytes_sent"] == 0
    assert result["compact_owner_mechanics_resident_observation_handle_present"] is True
    assert (
        result["compact_owner_mechanics_step_frame_handle_schema_id"]
        == hybrid_profile.COMPACT_OWNER_MECHANICS_STEP_FRAME_HANDLE_SCHEMA_ID
    )
    assert result["compact_owner_mechanics_step_frame_handle_published"] is True
    assert result["compact_owner_mechanics_step_frame_handle_consumed"] is True
    assert result["compact_owner_mechanics_step_frame_handle_publish_count"] == 1
    assert result["compact_owner_mechanics_step_frame_handle_consume_count"] == 1
    assert (
        result["compact_owner_mechanics_step_frame_handle_ring_slot_count"]
        == hybrid_profile.COMPACT_OWNER_MECHANICS_STEP_FRAME_HANDLE_RING_SLOT_COUNT
    )
    assert result["compact_owner_mechanics_step_frame_handle_slot_id"] == 2
    assert result["compact_owner_mechanics_step_frame_handle_generation"] == 2
    assert result["compact_owner_mechanics_step_frame_handle_digest"]
    assert result["compact_owner_mechanics_step_frame_handle_digest_verified"] is True
    assert (
        result["compact_owner_mechanics_step_frame_handle_owner_digest_verified"]
        is True
    )
    assert (
        result[
            "compact_owner_mechanics_step_frame_handle_resident_observation_present"
        ]
        is True
    )
    assert result["compact_owner_search_next_joint_action_published"] is True
    assert result["compact_owner_search_dense_joint_action_used"] is True
    assert result["compact_owner_search_dense_joint_action_owner_assembled"] is True
    assert result["compact_owner_search_dense_joint_action_parent_assembly_avoided"] is True
    assert result["compact_owner_search_dense_joint_action_present"] is True
    assert result["compact_owner_search_dense_joint_action_fallback_count"] == 0
    assert result["compact_owner_search_dense_joint_action_fallback_reason"] == "none"
    assert result["compact_owner_search_dense_joint_action_checksum"] != 0
    assert result["compact_owner_search_dense_joint_action_bytes"] == 8
    assert result["compact_owner_search_dense_joint_action_mismatch_count"] == 0
    assert result["compact_rollout_slab_parent_dense_action_reconstruction_count"] == 0
    assert result["compact_rollout_slab_parent_dense_action_reconstruction_used"] is False
    assert result["compact_rollout_slab_dense_joint_action_validation_count"] == 1
    assert result["compact_rollout_slab_next_joint_action_checksum"] == (
        result["compact_owner_search_dense_joint_action_checksum"]
    )
    assert result["compact_owner_search_pending_compact_batch_sidecar_stored"] is False
    assert result["compact_owner_search_pending_compact_batch_sidecar_storage_avoided"] is True
    assert result["compact_owner_search_pending_compact_batch_sidecar_store_count"] == 0
    assert result["compact_owner_search_pending_compact_batch_sidecar_store_avoided_count"] >= 1
    assert result["compact_owner_search_pending_root_batch_sidecar_stored"] is False
    assert result["compact_owner_search_pending_root_batch_sidecar_storage_avoided"] is True
    assert result["compact_owner_search_pending_root_batch_sidecar_store_count"] == 0
    assert result["compact_owner_search_pending_root_batch_sidecar_store_avoided_count"] >= 1
    assert result["compact_owner_search_pending_action_step_identity_handle_stored"] is True
    assert (
        result["compact_owner_search_pending_action_step_identity_handle_store_count"]
        >= 1
    )
    assert result["compact_owner_search_pending_root_build_request_stored"] is False
    assert result["compact_owner_search_pending_root_action_context_stored"] is False
    assert result["compact_owner_root_action_context_handle_used"] is True
    assert result["compact_owner_root_action_context_owner_store_count"] == 3
    assert result["compact_owner_root_action_context_owner_resolve_count"] == 3
    assert result["compact_owner_root_action_context_owner_release_count"] == 3
    assert result["compact_owner_root_action_context_owner_pending_count"] == 0
    assert result["compact_owner_root_action_context_owner_digest_verified"] is True
    assert result["compact_owner_search_parent_action_context_validation_count"] == 0
    assert result["compact_owner_search_owner_action_context_validation_count"] >= 1
    assert result["compact_owner_search_direct_root_build_request_handoff"] is True
    assert result["compact_owner_search_direct_root_parent_build_avoided"] is True
    assert result["compact_owner_root_search_transaction_used"] is True
    assert result["compact_owner_root_search_transaction_requested"] is True
    assert result["compact_owner_root_search_transaction_begin_count"] == 3
    assert result["compact_owner_root_search_transaction_submit_count"] == 3
    assert result["compact_owner_root_search_transaction_resolve_count"] == 3
    assert result["compact_owner_root_search_transaction_pending_count"] == 0
    assert result["compact_owner_root_search_transaction_max_pending_count"] == 1
    assert (
        result["compact_owner_root_search_transaction_parent_root_request_build_count"]
        == 0
    )
    assert result["compact_owner_root_search_transaction_parent_root_request_stored"] is False
    assert result["compact_owner_root_search_transaction_parent_compact_batch_stored"] is False
    assert result["compact_owner_root_search_transaction_parent_rebuild_count"] == 0
    assert (
        result["compact_owner_root_search_transaction_parent_root_action_context_stored"]
        is False
    )
    assert (
        result["compact_owner_root_search_transaction_parent_root_action_context_store_count"]
        == 0
    )
    assert (
        result["compact_owner_root_search_transaction_parent_root_action_context_array_bytes"]
        == 0
    )
    assert (
        result["compact_owner_root_search_transaction_parent_root_action_context_field_count"]
        == 0
    )
    assert (
        result["compact_owner_root_search_transaction_owner_root_request_build_count"]
        == 3
    )
    assert (
        result["compact_owner_root_search_transaction_owner_root_store_publish_count"]
        == 3
    )
    assert result["compact_owner_root_search_transaction_frame_generation_verified"] is True
    assert result["compact_owner_root_search_transaction_frame_digest_verified"] is True
    assert result["compact_owner_root_search_transaction_action_identity_verified"] is True
    assert (
        result["compact_owner_root_search_transaction_applied_action_mismatch_count"]
        == 0
    )
    assert result["compact_owner_step_frame_root_build_request_used"] is True
    assert (
        result["compact_owner_step_frame_root_build_request_from_batch_helper_used"]
        is False
    )
    assert result["compact_owner_step_frame_root_request_sidecar_array_bytes"] == 0
    assert result["compact_owner_step_frame_root_request_sidecar_field_count"] == 0
    assert result["compact_rollout_slab_parent_root_batch_builder_used"] is False
    assert result["compact_rollout_slab_parent_root_batch_builder_call_count"] == 0
    assert result["compact_rollout_slab_return_root_batch_sidecar_stored"] is False
    assert result["compact_rollout_slab_return_root_batch_sidecar_storage_avoided"] is True
    assert result["compact_rollout_slab_return_root_batch_sidecar_build_count"] == 0
    assert result["compact_owner_search_direct_root_build_request_observation_included"] is False
    assert result["compact_owner_search_direct_root_build_request_observation_bytes_sent"] == 0
    assert result["compact_owner_search_direct_root_owner_build_used"] is True
    assert result["compact_owner_search_direct_root_owner_build_count"] == 3
    assert result["compact_owner_search_resident_root_view_proved"] is True
    assert result["compact_owner_search_resident_root_view_h2d_bytes"] == 0.0
    assert result["compact_owner_search_resident_root_view_d2h_bytes"] == 0.0
    assert result["compact_owner_search_search_result_payload_bytes"] == 0
    assert result["compact_rollout_slab_committed_index_row_count"] == 0
    assert result["compact_rollout_slab_stored_index_row_count"] == 0


def test_hybrid_profile_owner_manager_boundary_stages_replay_without_parent_materialization(
    monkeypatch,
):
    import curvyzero.training.compact_rollout_slab as slab_module

    pytest.importorskip("torch")

    def fail_parent_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent root batch builder must not be called")

    def fail_parent_compact_batch_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent compact batch builder must not be called")

    def fail_parent_mechanics_step_view_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent mechanics step view builder must not be called")

    def fail_parent_root_build_request_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent root build request builder must not be called")

    def fail_parent_step_frame_root_request_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent step-frame root request builder must not be called")

    def fail_parent_dense_action_reconstruction(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent dense action reconstruction must not be called")

    def fail_parent_previous_transition_closure(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent previous-transition closure must not run")

    class BoundaryReplaySearchService:
        search_impl = "unit_test_owner_manager_replay_boundary"
        num_simulations = 1

        def __init__(self) -> None:
            self.run_count = 0

        def run(self, root_batch):
            assert root_batch.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
            assert root_batch.resident_observation is not None
            assert root_batch.metadata["compact_owner_search_direct_root_build_request_handoff"]
            assert root_batch.metadata["compact_owner_search_direct_root_parent_build_avoided"]
            assert root_batch.metadata["compact_owner_search_direct_root_owner_build_used"]
            assert root_batch.metadata["compact_owner_search_resident_root_view_proved"]
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            env_row = np.asarray(root_batch.env_row, dtype=np.int64)[active_roots]
            player = np.asarray(root_batch.player, dtype=np.int64)[active_roots]
            selected = (
                env_row + player + int(self.run_count)
            ).astype(np.int16, copy=False) % np.int16(ACTION_COUNT)
            self.run_count += 1
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata={
                    "compact_owner_search_direct_root_build_request_handoff": True,
                    "compact_owner_search_direct_root_parent_build_avoided": True,
                    "compact_owner_search_direct_root_owner_build_used": True,
                    "compact_owner_search_action_only_result": True,
                    "compact_owner_search_owner_materializes_replay": True,
                    "compact_owner_search_parent_slab_commits_replay": False,
                    "compact_owner_search_parent_reconstructed_search_result": False,
                    "compact_owner_search_search_result_payload_bytes": 0,
                    "compact_owner_search_visit_policy_bytes": 0,
                    "compact_owner_search_root_value_bytes": 0,
                    "compact_owner_search_action_dispatch_handle_sync_wrapper_count": 0,
                    "compact_owner_search_action_dispatch_handle_completed_at_submit_count": 0,
                    "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count": 0,
                },
            )

    replay_store = _OwnerManagerBoundaryReplayStore()
    learner = _OwnerManagerBoundaryLearner()
    search_service = BoundaryReplaySearchService()
    monkeypatch.setattr(slab_module, "build_compact_root_batch_v1", fail_parent_builder)
    monkeypatch.setattr(
        hybrid_profile,
        "_make_compact_batch",
        fail_parent_compact_batch_builder,
    )
    monkeypatch.setattr(
        hybrid_profile,
        "_make_compact_owner_mechanics_step_view",
        fail_parent_mechanics_step_view_builder,
    )
    monkeypatch.setattr(
        slab_module,
        "compact_root_build_request_v1_from_batch",
        fail_parent_root_build_request_builder,
    )
    monkeypatch.setattr(
        slab_module,
        "_root_build_request_from_owner_step_frame_slot_v1",
        fail_parent_step_frame_root_request_builder,
    )
    monkeypatch.setattr(
        slab_module,
        "_selected_joint_action_from_compact_sidecars",
        fail_parent_dense_action_reconstruction,
    )
    monkeypatch.setattr(
        CompactOwnerSearchDirectStepperV1,
        "_stage_previous_derived_transition",
        fail_parent_previous_transition_closure,
    )
    monkeypatch.setattr(
        CompactOwnerSearchDirectStepperV1,
        "_flush_derived_transition_batch",
        fail_parent_previous_transition_closure,
    )
    monkeypatch.setattr(
        CompactOwnerSearchDirectStepperV1,
        "_commit_previous_transition",
        fail_parent_previous_transition_closure,
    )

    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=lambda: search_service,
        replay_store_factory=lambda: replay_store,
        learner_factory=lambda: learner,
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_model_refresh_interval=100,
        owner_defer_maintenance=True,
        root_store_capacity=8,
        require_resident_root_view=True,
        fixed_action_result_buffer=True,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=2,
        player_count=2,
        search_service=proxy,
        search_lane="unit_test_owner_manager_replay_boundary",
        policy_source="unit_test_owner_manager_replay_boundary",
        transition_batch_size=2,
        owner_local_transition_derivation=True,
        owner_proxy_transition_closure=True,
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )

    try:
        result = run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                steps=3,
                warmup_steps=0,
                seed=2026060902,
                stack_storage_dtype="uint8",
                update_host_observation_stack=False,
                resident_observation_search=True,
                materialize_scalar_timestep=False,
                native_actor_buffer=True,
                compact_owner_action_step_boundary=True,
                compact_owner_mechanics_step_boundary=True,
                compact_owner_minimal_step_payload_snapshot=True,
            ),
            observation_renderer=_PersistentBackendSentinelRenderer(),
            compact_rollout_slab=stepper,
        )
        proxy.drain_owner_maintenance(wait=True)
        proxy_metadata = dict(proxy.metadata)
        stepper_metadata = dict(stepper.metadata)
    finally:
        stepper.close()
        proxy.close()

    assert result["compact_owner_action_step_boundary_proof_passed"] is True
    assert result["compact_owner_mechanics_parent_compact_batch_object_count"] == 0
    assert result["compact_owner_mechanics_parent_step_frame_build_count"] == 0
    assert result["compact_owner_mechanics_host_observation_bytes_sent"] == 0
    assert result["compact_owner_mechanics_host_final_observation_bytes_sent"] == 0
    assert result["compact_owner_step_frame_root_build_request_used"] is True
    assert (
        result["compact_owner_step_frame_root_build_request_from_batch_helper_used"]
        is False
    )
    assert result["compact_rollout_slab_parent_dense_action_reconstruction_count"] == 0
    assert result["compact_rollout_slab_parent_root_batch_builder_call_count"] == 0
    assert result["compact_rollout_slab_return_root_batch_sidecar_stored"] is False
    assert result["compact_owner_search_search_result_payload_bytes"] == 0
    assert result["compact_rollout_slab_committed_index_row_count"] == 0
    assert result["compact_rollout_slab_stored_index_row_count"] == 0

    assert proxy_metadata["compact_owner_search_owner_replay_append_submitted_entry_count"] > 0
    assert proxy_metadata["compact_owner_search_owner_replay_append_count"] > 0
    assert proxy_metadata["compact_owner_search_direct_transition_batch_replay_used"] is True
    assert (
        proxy_metadata[
            "compact_owner_search_direct_transition_batch_replay_legacy_expanded_entry_count"
        ]
        == 0
    )
    assert proxy_metadata["compact_owner_search_owner_train_request_count"] >= 1
    assert proxy_metadata["compact_owner_search_owner_submitted_learner_update_count"] > 0
    assert proxy_metadata["compact_owner_search_owner_learner_update_count"] > 0
    assert proxy_metadata["compact_owner_search_owner_maintenance_pending_work_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_policy_lag_current"] >= 1
    assert proxy_metadata["compact_owner_search_owner_proxy_transition_closure_used"] is True
    assert (
        proxy_metadata["compact_owner_search_owner_proxy_transition_closure_closed_count"]
        > 0
    )
    assert proxy_metadata["compact_owner_search_owner_proxy_transition_closure_batch_count"] > 0
    assert (
        proxy_metadata["compact_owner_search_owner_proxy_transition_closure_transition_count"]
        > 0
    )
    assert (
        proxy_metadata["compact_owner_search_owner_proxy_transition_closure_pending_count"]
        == 0
    )
    assert proxy_metadata["compact_owner_search_parent_previous_transition_closure_count"] == 0
    assert proxy_metadata["compact_owner_search_action_feedback_verified"] is True
    assert proxy_metadata["compact_owner_search_action_feedback_mismatch_count"] == 0

    assert stepper_metadata["compact_owner_search_parent_previous_transition_closure_count"] == 0
    assert stepper_metadata["compact_owner_search_owner_proxy_transition_closure_used"] is True
    assert stepper_metadata["compact_owner_search_transition_batch_count"] > 0
    assert stepper_metadata["compact_owner_search_transition_batch_entry_count"] > 0
    assert (
        stepper_metadata["compact_owner_search_pending_action_step_identity_handle_stored"]
        is False
    )


def test_hybrid_profile_owner_action_dispatch_step_overlap_resolves_after_parent_payload(
    monkeypatch,
):
    torch = pytest.importorskip("torch")

    class OverlapRootBuildRequestSearchService:
        supports_two_phase_compact_search = True
        search_impl = "unit_test_owner_action_dispatch_step_overlap"
        num_simulations = 1

        def __init__(self) -> None:
            self.submit_count = 0
            self.resolve_count = 0
            self.sync_wrapper_count = 0
            self.parent_payload_snapshot_count = 0

        def run_action_step(self, root_batch):
            del root_batch
            raise AssertionError("overlap path must not pass a parent root batch")

        def run_action_step_from_root_build_request(self, root_build_request):
            del root_build_request
            raise AssertionError("overlap path must not use the sync wrapper")

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"overlap action-only test flushed {replay_payload_handle}")

        def stage_replay_append_entries(self, replay_append_entries):
            del replay_append_entries
            return 0

        def submit_action_step_from_root_build_request(self, root_build_request):
            self.submit_count += 1
            return SimpleNamespace(
                dispatch_id=self.submit_count,
                root_build_request=root_build_request,
                min_parent_payload_snapshot_count=self.parent_payload_snapshot_count + 1,
            )

        def resolve_action_step_handle(self, handle, *, sync_wrapper=False):
            if self.parent_payload_snapshot_count < int(
                handle.min_parent_payload_snapshot_count
            ):
                raise AssertionError("owner action resolved before parent payload work")
            self.resolve_count += 1
            self.sync_wrapper_count += 1 if bool(sync_wrapper) else 0
            root_build_request = handle.root_build_request
            root_index = np.flatnonzero(root_build_request.active_root_mask).astype(
                np.int64,
                copy=False,
            )
            env_row_all = np.asarray(
                root_build_request.policy_env_row,
                dtype=np.int32,
            ).reshape(-1)
            player_all = np.asarray(
                root_build_request.policy_player,
                dtype=np.int16,
            ).reshape(-1)
            policy_env_id_all = np.asarray(
                root_build_request.policy_env_id,
                dtype=np.int64,
            ).reshape(-1)
            legal_mask = np.asarray(
                root_build_request.action_mask,
                dtype=np.bool_,
            ).reshape(int(root_build_request.root_count), ACTION_COUNT)
            selected = legal_mask[root_index].argmax(axis=1).astype(
                np.int16,
                copy=False,
            )
            env_row = env_row_all[root_index].astype(np.int32, copy=False)
            player = player_all[root_index].astype(np.int16, copy=False)
            policy_env_id = policy_env_id_all[root_index].astype(np.int64, copy=False)
            dense = np.zeros(
                (
                    int(root_build_request.batch_size),
                    int(root_build_request.player_count),
                ),
                dtype=np.int16,
            )
            dense[env_row.astype(np.int64), player.astype(np.int64)] = selected
            handle_text = f"unit-overlap:{self.resolve_count}"
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
                    "selected_action_digest": compact_search_array_digest_v1(
                        selected.astype(np.int16, copy=False)
                    ),
                    "search_replay_payload_digest": (
                        compact_search_deferred_replay_payload_digest_v1(handle_text)
                    ),
                    "search_replay_payload_digest_deferred": True,
                    "compact_owner_search_owner_materializes_replay": True,
                    "compact_owner_search_action_only_result": True,
                    "compact_owner_search_direct_root_build_request_handoff": True,
                    "compact_owner_search_direct_root_parent_build_avoided": True,
                    "compact_owner_search_parent_slab_commits_replay": False,
                    "compact_owner_search_parent_reconstructed_search_result": False,
                    "compact_owner_search_search_result_payload_bytes": 0,
                    "compact_owner_search_visit_policy_bytes": 0,
                    "compact_owner_search_root_value_bytes": 0,
                    "compact_owner_search_action_dispatch_handle_sync_wrapper_count": int(
                        self.sync_wrapper_count
                    ),
                    "compact_owner_search_action_dispatch_handle_completed_at_submit_count": 0,
                    "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count": 0,
                },
            )

    search_service = OverlapRootBuildRequestSearchService()
    original_minimal_snapshot = hybrid_profile._copy_minimal_compact_payload_for_owner_step

    def mark_parent_payload_snapshot(compact):
        search_service.parent_payload_snapshot_count += 1
        return original_minimal_snapshot(compact)

    monkeypatch.setattr(
        hybrid_profile,
        "_copy_minimal_compact_payload_for_owner_step",
        mark_parent_payload_snapshot,
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=2,
        player_count=2,
        search_service=search_service,
        search_lane="unit_test_owner_action_dispatch_step_overlap",
        policy_source="unit_test_owner_action_dispatch_step_overlap",
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )

    try:
        result = run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                steps=2,
                warmup_steps=1,
                seed=2026060802,
                stack_storage_dtype="uint8",
                update_host_observation_stack=False,
                resident_observation_search=True,
                materialize_scalar_timestep=False,
                native_actor_buffer=True,
                compact_owner_action_step_boundary=True,
                compact_owner_mechanics_step_boundary=True,
                compact_owner_action_dispatch_step_overlap=True,
                compact_owner_minimal_step_payload_snapshot=True,
            ),
            observation_renderer=_PersistentBackendSentinelRenderer(),
            compact_rollout_slab=stepper,
        )
    finally:
        stepper.close()

    assert torch is not None
    assert search_service.submit_count == 3
    assert search_service.resolve_count == 3
    assert search_service.sync_wrapper_count == 0
    assert search_service.parent_payload_snapshot_count == 3
    assert result["compact_owner_action_step_boundary_proof_passed"] is True
    assert result["compact_owner_action_dispatch_step_overlap_enabled"] is True
    assert result["compact_owner_action_dispatch_step_overlap_proof_passed"] is True
    assert result["contract"]["compact_owner_action_dispatch_step_overlap"] is True
    assert result["compact_rollout_slab_action_dispatch_step_overlap_supported"] is True
    assert result["compact_rollout_slab_action_dispatch_step_overlap_used"] is True
    assert result["compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait"] is True
    assert result["compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper"] is False
    assert result["compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count"] == 0
    assert result["compact_owner_search_action_dispatch_handle_sync_wrapper_count"] == 0
    assert result["compact_owner_search_action_dispatch_handle_completed_at_submit_count"] == 0
    assert result["compact_owner_search_action_dispatch_handle_result_wait_in_submit_count"] == 0
    assert result["compact_rollout_slab_action_dispatch_step_overlap_submit_count"] == 3
    assert result["compact_rollout_slab_action_dispatch_step_overlap_resolve_count"] == 3
    assert result["compact_rollout_slab_action_dispatch_step_overlap_pending_count"] == 0
    assert result["compact_rollout_slab_action_dispatch_step_overlap_max_pending_count"] == 1
    assert result["compact_rollout_slab_action_dispatch_step_overlap_parent_work_sec"] > 0.0
    assert result["timings"][
        "compact_rollout_slab_action_dispatch_step_overlap_count"
    ] == pytest.approx(2.0)
    assert result["timings"][
        "compact_rollout_slab_action_dispatch_step_overlap_parent_work_sec"
    ] > 0.0
    assert result["timings"]["compact_rollout_slab_submit_sec"] >= 0.0
    assert result["timings"]["compact_rollout_slab_resolve_sec"] >= 0.0


def test_hybrid_profile_guarded_owner_action_step_boundary_requires_no_scalar_timestep():
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=SimpleNamespace(),
        search_lane="unit_test_owner_action_step_boundary_guard",
        policy_source="unit_test_owner_action_step_boundary_guard",
    )

    with pytest.raises(ValueError, match="materialize_scalar_timestep=False"):
        run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                steps=1,
                warmup_steps=0,
                compact_owner_action_step_boundary=True,
            ),
            compact_rollout_slab=slab,
        )


def test_hybrid_profile_minimal_step_payload_requires_owner_action_boundary():
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=SimpleNamespace(),
        search_lane="unit_test_minimal_step_payload_guard",
        policy_source="unit_test_minimal_step_payload_guard",
    )

    with pytest.raises(
        ValueError,
        match="compact_owner_minimal_step_payload_snapshot requires "
        "compact_owner_action_step_boundary",
    ):
        run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                steps=1,
                warmup_steps=0,
                materialize_scalar_timestep=False,
                compact_owner_minimal_step_payload_snapshot=True,
            ),
            compact_rollout_slab=slab,
        )


def test_hybrid_profile_owner_mechanics_step_boundary_requires_direct_root_request():
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=2,
        player_count=2,
        search_service=SimpleNamespace(),
        search_lane="unit_test_owner_mechanics_boundary_guard",
        policy_source="unit_test_owner_mechanics_boundary_guard",
        resident_root_host_observation_stub=True,
        direct_root_build_request=False,
    )

    with pytest.raises(
        ValueError,
        match="compact_owner_mechanics_step_boundary requires "
        "direct_root_build_request=True",
    ):
        run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                steps=1,
                warmup_steps=0,
                stack_storage_dtype="uint8",
                update_host_observation_stack=False,
                resident_observation_search=True,
                materialize_scalar_timestep=False,
                native_actor_buffer=True,
                compact_owner_action_step_boundary=True,
                compact_owner_mechanics_step_boundary=True,
            ),
            observation_renderer=_PersistentBackendSentinelRenderer(),
            compact_rollout_slab=stepper,
        )


def test_hybrid_profile_owner_mechanics_step_boundary_rejects_batched_probe():
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=2,
        player_count=2,
        search_service=SimpleNamespace(),
        search_lane="unit_test_owner_mechanics_boundary_probe_guard",
        policy_source="unit_test_owner_mechanics_boundary_probe_guard",
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )

    with pytest.raises(
        ValueError,
        match="compact_owner_mechanics_step_boundary cannot feed batched_stack_probe",
    ):
        run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                steps=1,
                warmup_steps=0,
                stack_storage_dtype="uint8",
                update_host_observation_stack=False,
                resident_observation_search=True,
                materialize_scalar_timestep=False,
                native_actor_buffer=True,
                compact_owner_action_step_boundary=True,
                compact_owner_mechanics_step_boundary=True,
            ),
            observation_renderer=_PersistentBackendSentinelRenderer(),
            batched_stack_probe=SimpleNamespace(),
            compact_rollout_slab=stepper,
        )


def test_hybrid_profile_projects_action_only_owner_search_deferred_maintenance():
    class FakeActionOnlyOwnerSearchService:
        search_impl = "unit_test_action_only_owner_search"
        num_simulations = 0
        supports_two_phase_compact_search = True

        def __init__(self):
            self.action_request_count = 0
            self.owner_replay_append_staged_entry_count = 0
            self.owner_replay_append_suppressed_entry_count = 0
            self.owner_replay_append_submitted_entry_count = 0
            self.owner_replay_append_request_count = 0
            self.owner_replay_append_count = 0
            self.owner_train_request_count = 0
            self.owner_learner_update_count = 0
            self.owner_maintenance_drain_request_count = 0
            self.owner_maintenance_staged_work_item_count = 0
            self.owner_maintenance_drained_count = 0
            self.owner_maintenance_drained_work_item_count = 0
            self.owner_maintenance_drained_replay_append_entry_count = 0
            self.owner_maintenance_drained_replay_append_count = 0
            self.owner_learning_enabled = True
            self.owner_learning_enabled_history: list[bool] = []
            self._metadata: dict[str, object] = {}
            self._refresh_metadata()

        @property
        def metadata(self):
            return dict(self._metadata)

        def _refresh_metadata(self):
            self._metadata = {
                "compact_owner_search_slab_proxy": True,
                "compact_owner_search_lazy_slab_proxy": True,
                "compact_owner_search_slab_proxy_initialized": True,
                "compact_owner_search_boundary_kind": (
                    "worker_search_parent_slab_commit"
                ),
                "compact_owner_search_parent_slab_commits_replay": False,
                "compact_owner_search_worker_kind": "local_process_owner_search_v1",
                "compact_owner_search_worker_resource_scope": "persistent_process",
                "compact_owner_search_worker_resource_distinct_from_actor": True,
                "compact_owner_search_worker_hardware_resource_distinct_from_actor": False,
                "compact_owner_search_owner_pid": 123,
                "compact_owner_search_root_slot_count": 2,
                "compact_owner_search_active_root_count": 2,
                "compact_owner_search_request_bytes": 55,
                "compact_owner_search_result_bytes": 88,
                "compact_owner_search_request_cuda_tensor_count": 0,
                "compact_owner_search_result_cuda_tensor_count": 0,
                "compact_owner_search_root_observation_bytes_sent": 0,
                "compact_owner_search_parent_reconstructed_search_result": False,
                "compact_owner_search_action_only_result": True,
                "compact_owner_search_owner_materializes_replay": True,
                "compact_owner_search_action_feedback_verified": (
                    self.owner_replay_append_submitted_entry_count > 0
                ),
                "compact_owner_search_action_feedback_transition_count": (
                    self.owner_replay_append_submitted_entry_count
                ),
                "compact_owner_search_action_feedback_action_count": (
                    self.owner_replay_append_submitted_entry_count
                ),
                "compact_owner_search_action_feedback_mismatch_count": 0,
                "compact_owner_search_expected_joint_action_checksum": (
                    11 * self.owner_replay_append_submitted_entry_count
                ),
                "compact_owner_search_applied_joint_action_checksum": (
                    11 * self.owner_replay_append_submitted_entry_count
                ),
                "compact_owner_search_replay_action_checksum": (
                    11 * self.owner_replay_append_submitted_entry_count
                ),
                "compact_owner_search_search_result_payload_bytes": 0,
                "compact_owner_search_search_result_payload_transport_kind": (
                    "action_only_owner_cached_replay_v1"
                ),
                "compact_owner_search_search_result_payload_json_safe": True,
                "compact_owner_search_selected_action_bytes": 8,
                "compact_owner_search_visit_policy_bytes": 0,
                "compact_owner_search_root_value_bytes": 0,
                "compact_owner_search_optional_array_bytes": 0,
                "compact_owner_search_inner_two_phase_action_step": True,
                "compact_owner_search_inner_device_replay_payload_deferred": True,
                "compact_owner_search_inner_device_replay_payload_flushed_count": 4,
                "compact_owner_search_inner_deferred_one_simulation_replay_payload_flush_count": 4,
                "compact_owner_search_inner_deferred_one_simulation_replay_materialized_on_flush_count": 4,
                "compact_owner_search_inner_deferred_one_simulation_replay_recurrent_inference_calls": 4.0,
                "compact_owner_search_inner_deferred_one_simulation_model_identity_match_count": 4,
                "compact_owner_search_inner_deferred_one_simulation_model_refresh_crossed_count": 0,
                "compact_owner_search_inner_pending_deferred_replay_payload_count_max": 4,
                "compact_owner_search_inner_pending_deferred_replay_payload_final_count": 0,
                "compact_owner_search_inner_deferred_one_simulation_replay_flush_sec": 0.004,
                "compact_owner_search_inner_device_replay_payload_flush_sec": 0.006,
                "compact_owner_search_inner_replay_payload_d2h_bytes": 0.0,
                "compact_owner_search_inner_deferred_one_simulation_action_model_state_digest": "digest-a",
                "compact_owner_search_inner_deferred_one_simulation_flush_model_state_digest": "digest-a",
                "compact_owner_search_worker_owns_search_state": True,
                "compact_owner_search_worker_owns_replay_state": True,
                "compact_owner_search_worker_owns_model_state": True,
                "compact_owner_search_consumed_learner_update": (
                    self.owner_learner_update_count > 0
                ),
                "compact_owner_search_search_refresh_update_count": (
                    self.owner_learner_update_count
                ),
                "compact_owner_search_replay_append_entry_count": (
                    self.owner_replay_append_submitted_entry_count
                ),
                "compact_owner_search_replay_append_count": self.owner_replay_append_count,
                "compact_owner_search_learner_update_count": (
                    self.owner_learner_update_count
                ),
                "compact_owner_search_model_owner_ref_returned": (
                    self.owner_learner_update_count > 0
                ),
                "compact_owner_search_model_owner_ref_digest": (
                    "owner-digest-1" if self.owner_learner_update_count > 0 else ""
                ),
                "compact_owner_search_owner_replay_append_enabled": True,
                "compact_owner_search_owner_learning_enabled": (
                    self.owner_learning_enabled
                ),
                "compact_owner_search_owner_sample_batch_size": 1,
                "compact_owner_search_owner_train_steps": 1,
                "compact_owner_search_owner_train_interval": 1,
                "compact_owner_search_owner_defer_maintenance": True,
                "compact_owner_search_owner_loop_schema_id": (
                    "curvyzero_compact_owner_search_priority_loop/v1"
                ),
                "compact_owner_search_owner_loop_kind": (
                    "persistent_priority_owner_loop_v1"
                ),
                "compact_owner_search_owner_loop_persistent": True,
                "compact_owner_search_owner_action_priority_enabled": True,
                "compact_owner_search_owner_action_request_count": (
                    self.action_request_count
                ),
                "compact_owner_search_owner_maintenance_request_count": (
                    self.owner_replay_append_request_count
                ),
                "compact_owner_search_owner_run_request_count": self.action_request_count,
                "compact_owner_search_owner_learner_telemetry": {
                    "compact_owner_search_owner_train_wall_sec": (
                        0.011 if self.owner_learner_update_count > 0 else 0.0
                    ),
                    "compact_owner_search_owner_train_sample_sec": (
                        0.001 if self.owner_learner_update_count > 0 else 0.0
                    ),
                    "compact_owner_search_owner_train_learner_update_sec": (
                        0.002 if self.owner_learner_update_count > 0 else 0.0
                    ),
                    "compact_owner_search_owner_train_model_state_digest_sec": (
                        0.003 if self.owner_learner_update_count > 0 else 0.0
                    ),
                    "compact_owner_search_owner_train_model_state_dict_sec": (
                        0.004 if self.owner_learner_update_count > 0 else 0.0
                    ),
                    "compact_owner_search_owner_train_owner_ref_build_sec": (
                        0.0005 if self.owner_learner_update_count > 0 else 0.0
                    ),
                    "compact_owner_search_owner_train_accounted_sec": (
                        0.0105 if self.owner_learner_update_count > 0 else 0.0
                    ),
                    "compact_owner_search_owner_train_residual_sec": (
                        0.0005 if self.owner_learner_update_count > 0 else 0.0
                    ),
                    "compact_owner_search_owner_train_timing_aggregate_count": (
                        1 if self.owner_learner_update_count > 0 else 0
                    ),
                },
                "compact_owner_search_owner_replay_append_staged_entry_count": (
                    self.owner_replay_append_staged_entry_count
                ),
                "compact_owner_search_owner_replay_append_suppressed_entry_count": (
                    self.owner_replay_append_suppressed_entry_count
                ),
                "compact_owner_search_owner_replay_append_submitted_entry_count": (
                    self.owner_replay_append_submitted_entry_count
                ),
                "compact_owner_search_owner_replay_append_request_count": (
                    self.owner_replay_append_request_count
                ),
                "compact_owner_search_owner_replay_append_count": (
                    self.owner_replay_append_count
                ),
                "compact_owner_search_owner_train_request_count": (
                    self.owner_train_request_count
                ),
                "compact_owner_search_owner_submitted_learner_update_count": (
                    self.owner_learner_update_count
                ),
                "compact_owner_search_owner_learner_update_count": (
                    self.owner_learner_update_count
                ),
                "compact_owner_search_owner_pending_replay_append_entry_count": 0,
                "compact_owner_search_owner_maintenance_drain_request_count": (
                    self.owner_maintenance_drain_request_count
                ),
                "compact_owner_search_owner_maintenance_staged_work_item_count": (
                    self.owner_maintenance_staged_work_item_count
                ),
                "compact_owner_search_owner_maintenance_drained_count": (
                    self.owner_maintenance_drained_count
                ),
                "compact_owner_search_owner_maintenance_drained_work_item_count": (
                    self.owner_maintenance_drained_work_item_count
                ),
                "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": (
                    self.owner_maintenance_drained_replay_append_entry_count
                ),
                "compact_owner_search_owner_maintenance_drained_replay_append_count": (
                    self.owner_maintenance_drained_replay_append_count
                ),
                "compact_owner_search_owner_maintenance_pending_work_count": 0,
                "compact_owner_search_owner_maintenance_inflight": False,
                "compact_owner_search_owner_maintenance_final_drain_sec": 0.0,
                "compact_owner_search_owner_maintenance_coalescing_kind": (
                    "eager_append_or_train_boundary_v1"
                ),
                "compact_owner_search_owner_maintenance_coalesced_skip_count": 0,
                "compact_owner_search_owner_maintenance_eager_append_drain_count": 0,
                "compact_owner_search_owner_async_learner_worker_enabled": False,
                "compact_owner_search_owner_async_learner_worker_kind": "none",
                "compact_owner_search_owner_async_learner_worker_resource_scope": "",
                "compact_owner_search_owner_async_learner_worker_resource_id": "",
                "compact_owner_search_owner_async_learner_actor_resource_id": "",
                "compact_owner_search_owner_async_learner_worker_parent_pid": 0,
                (
                    "compact_owner_search_owner_async_learner_resource_distinct_"
                    "from_owner"
                ): False,
                (
                    "compact_owner_search_owner_async_learner_hardware_resource_"
                    "distinct_from_owner"
                ): False,
                "compact_owner_search_owner_async_learner_max_pending": 0,
                "compact_owner_search_owner_async_learner_submit_count": 0,
                "compact_owner_search_owner_async_learner_completed_count": 0,
                "compact_owner_search_owner_async_learner_pending_count": 0,
                "compact_owner_search_owner_async_learner_max_pending_observed": 0,
                "compact_owner_search_owner_async_learner_wait_count": 0,
                "compact_owner_search_owner_async_learner_wait_sec": 0.0,
                "compact_owner_search_owner_action_while_async_learner_pending_count": 0,
                "compact_owner_search_owner_async_learner_failed": False,
                "compact_owner_search_owner_async_learner_request_host_only": False,
                "compact_owner_search_owner_async_learner_request_cuda_tensor_count": -1,
                "compact_owner_search_owner_async_learner_result_host_only": False,
                "compact_owner_search_owner_async_learner_result_cuda_tensor_count": -1,
                "compact_owner_search_owner_async_learner_request_bytes": 0,
                "compact_owner_search_owner_async_learner_result_bytes": 0,
                "compact_owner_search_owner_async_learner_worker_pid": 0,
                "compact_owner_search_owner_async_learner_worker_job_wall_sec": 0.0,
                "compact_owner_search_owner_async_learner_payload_prepare_sec": 0.0,
                (
                    "compact_owner_search_owner_async_learner_worker_pid_distinct_"
                    "from_owner"
                ): False,
                "compact_owner_search_owner_async_learner_worker_owns_model_state": (
                    False
                ),
                "compact_owner_search_owner_policy_lag_current": 0,
                "compact_owner_search_owner_policy_lag_max": 0,
                "compact_owner_search_owner_maintenance_actor_steps_while_pending": 0,
                "compact_owner_search_owner_maintenance_actor_steps_while_policy_lagged": 0,
                "compact_owner_search_owner_action_while_maintenance_pending_count": 0,
                "compact_owner_search_owner_action_while_policy_lagged_count": 0,
                "compact_owner_search_owner_action_served_before_maintenance_count": 0,
                "compact_owner_search_owner_fifo_blocked_action_count": 0,
                "compact_owner_search_owner_maintenance_failed": False,
                "compact_owner_search_parent_publish_sec": 0.001,
                "compact_owner_search_parent_submit_sec": 0.001,
                "compact_owner_search_parent_wait_sec": 0.001,
                "compact_owner_search_parent_wall_sec": 0.003,
                "compact_owner_search_worker_wall_sec": 0.004,
                "compact_owner_search_worker_root_resolve_sec": 0.001,
                "compact_owner_search_worker_search_sec": 0.002,
                "compact_owner_search_worker_replay_append_sec": (
                    0.001 if self.owner_replay_append_count > 0 else 0.0
                ),
                "compact_owner_search_worker_learner_train_sec": (
                    0.001 if self.owner_learner_update_count > 0 else 0.0
                ),
                "compact_owner_search_worker_search_refresh_sec": (
                    0.001 if self.owner_learner_update_count > 0 else 0.0
                ),
            }

        def run_action_step(self, root_batch):
            self.action_request_count += 1
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            selected = np.argmax(root_batch.legal_mask[active_roots], axis=1).astype(
                np.int16,
                copy=False,
            )
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            self._refresh_metadata()
            result = validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata=self.metadata,
            )
            action_step = compact_search_action_step_v1_from_result(
                result,
                replay_payload_handle=f"owner-action-only-{self.action_request_count}",
                metadata=self.metadata,
            )
            action_step.metadata[
                "search_replay_payload_digest"
            ] = compact_search_deferred_replay_payload_digest_v1(
                action_step.replay_payload_handle
            )
            action_step.metadata["search_replay_payload_digest_deferred"] = True
            return action_step

        def set_owner_learning_enabled(self, enabled):
            self.owner_learning_enabled = bool(enabled)
            self.owner_learning_enabled_history.append(self.owner_learning_enabled)
            self._refresh_metadata()

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(
                f"parent must not flush owner-materialized replay {replay_payload_handle}"
            )

        def stage_replay_append_entries(self, replay_append_entries):
            if replay_append_entries is None:
                return 0
            if isinstance(replay_append_entries, (list, tuple)):
                count = len(replay_append_entries)
            else:
                count = 1
            if not self.owner_learning_enabled:
                self.owner_replay_append_suppressed_entry_count += count
                self._refresh_metadata()
                return 0
            self.owner_replay_append_staged_entry_count += count
            self._refresh_metadata()
            return count

        def drain_owner_maintenance(self, *, wait=True):
            del wait
            self.owner_maintenance_drain_request_count += 1
            submitted = self.owner_replay_append_staged_entry_count
            self.owner_replay_append_submitted_entry_count = submitted
            self.owner_replay_append_request_count = 1 if submitted else 0
            self.owner_replay_append_count = submitted
            self.owner_train_request_count = 1 if submitted else 0
            self.owner_learner_update_count = 1 if submitted else 0
            self.owner_maintenance_staged_work_item_count = (
                self.owner_replay_append_request_count
            )
            self.owner_maintenance_drained_count = self.owner_replay_append_request_count
            self.owner_maintenance_drained_work_item_count = (
                self.owner_replay_append_request_count
            )
            self.owner_maintenance_drained_replay_append_entry_count = submitted
            self.owner_maintenance_drained_replay_append_count = submitted
            self._refresh_metadata()
            return self.metadata

        def close(self):
            self._refresh_metadata()

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeActionOnlyOwnerSearchService(),
        search_lane="unit_test_action_only_owner_search",
        policy_source="unit_test_action_only_owner_search",
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=2,
            warmup_steps=2,
            seed=20260604,
            materialize_scalar_timestep=False,
        ),
        compact_rollout_slab=slab,
    )

    assert result["compact_owner_search_action_only_result"] is True
    assert result["compact_owner_search_owner_materializes_replay"] is True
    assert result["compact_owner_search_action_feedback_verified"] is True
    assert result["compact_owner_search_action_feedback_transition_count"] == result[
        "compact_owner_search_owner_replay_append_submitted_entry_count"
    ]
    assert result["compact_owner_search_action_feedback_action_count"] > 0
    assert result["compact_owner_search_action_feedback_mismatch_count"] == 0
    assert result["compact_owner_search_expected_joint_action_checksum"] == result[
        "compact_owner_search_applied_joint_action_checksum"
    ]
    assert result["compact_owner_search_expected_joint_action_checksum"] == result[
        "compact_owner_search_replay_action_checksum"
    ]
    assert result["compact_owner_search_parent_slab_commits_replay"] is False
    assert result["compact_owner_search_parent_reconstructed_search_result"] is False
    assert result["compact_owner_search_search_result_payload_bytes"] == 0
    assert result["compact_owner_search_visit_policy_bytes"] == 0
    assert result["compact_owner_search_root_value_bytes"] == 0
    assert (
        result["compact_owner_search_search_result_payload_transport_kind"]
        == "action_only_owner_cached_replay_v1"
    )
    assert result["compact_rollout_slab_committed_index_row_count"] == 0
    assert result["compact_rollout_slab_stored_index_row_count"] == 0
    assert result["compact_rollout_slab_sample_gate_calls"] == 0
    assert result["compact_rollout_slab_learner_gate_calls"] == 0
    assert result["compact_owner_search_owner_replay_append_staged_entry_count"] > 0
    assert result["compact_owner_search_owner_learning_enabled"] is True
    assert result["compact_owner_search_owner_replay_append_suppressed_entry_count"] > 0
    assert (
        result["compact_owner_search_owner_replay_append_submitted_entry_count"]
        == result["compact_owner_search_owner_replay_append_staged_entry_count"]
    )
    assert (
        result["compact_owner_search_replay_append_entry_count"]
        == result["compact_owner_search_owner_replay_append_submitted_entry_count"]
    )
    assert (
        result["compact_owner_search_replay_append_count"]
        == result["compact_owner_search_owner_replay_append_count"]
    )
    assert result["compact_owner_search_owner_train_request_count"] == 1
    assert result["compact_owner_search_owner_learner_update_count"] == 1
    assert result["compact_owner_search_consumed_learner_update"] is True
    assert result["compact_owner_search_search_refresh_update_count"] == 1
    assert result["compact_owner_search_owner_maintenance_drain_request_count"] == 1
    assert result["compact_owner_search_owner_maintenance_drained_count"] == 1
    assert result[
        "compact_owner_search_owner_maintenance_staged_work_item_count"
    ] == 1
    assert result[
        "compact_owner_search_owner_maintenance_drained_work_item_count"
    ] == 1
    assert result[
        "compact_owner_search_owner_maintenance_drained_replay_append_entry_count"
    ] == result["compact_owner_search_owner_replay_append_submitted_entry_count"]
    assert result[
        "compact_owner_search_owner_maintenance_drained_replay_append_count"
    ] == result["compact_owner_search_owner_replay_append_count"]
    assert result["compact_owner_search_owner_maintenance_pending_work_count"] == 0
    assert result["compact_owner_search_owner_maintenance_inflight"] is False
    assert (
        result["compact_owner_search_owner_maintenance_final_drain_in_measured_sec"]
        is True
    )
    assert result["compact_owner_search_owner_train_wall_sec"] == pytest.approx(0.011)
    assert result["compact_owner_search_owner_train_sample_sec"] == pytest.approx(0.001)
    assert result[
        "compact_owner_search_owner_train_learner_update_sec"
    ] == pytest.approx(0.002)
    assert result[
        "compact_owner_search_owner_train_model_state_digest_sec"
    ] == pytest.approx(0.003)
    assert result[
        "compact_owner_search_owner_train_model_state_dict_sec"
    ] == pytest.approx(0.004)
    assert result[
        "compact_owner_search_owner_train_owner_ref_build_sec"
    ] == pytest.approx(0.0005)
    assert result["compact_owner_search_owner_train_accounted_sec"] == pytest.approx(
        0.0105
    )
    assert result["compact_owner_search_owner_train_residual_sec"] == pytest.approx(
        0.0005
    )
    assert result["compact_owner_search_owner_train_timing_aggregate_count"] == 1
    assert result["compact_owner_search_worker_owns_replay_state"] is True
    assert result["compact_owner_search_worker_owns_model_state"] is True
    assert result["compact_owner_search_inner_two_phase_action_step"] is True
    assert result["compact_owner_search_inner_device_replay_payload_deferred"] is True
    assert result["compact_owner_search_inner_device_replay_payload_flushed_count"] == 4
    assert (
        result[
            "compact_owner_search_inner_deferred_one_simulation_replay_payload_flush_count"
        ]
        == 4
    )
    assert (
        result[
            "compact_owner_search_inner_deferred_one_simulation_replay_materialized_on_flush_count"
        ]
        == 4
    )
    assert result[
        "compact_owner_search_inner_deferred_one_simulation_replay_recurrent_inference_calls"
    ] == pytest.approx(4.0)
    assert (
        result[
            "compact_owner_search_inner_deferred_one_simulation_model_identity_match_count"
        ]
        == 4
    )
    assert (
        result[
            "compact_owner_search_inner_deferred_one_simulation_model_refresh_crossed_count"
        ]
        == 0
    )
    assert result["compact_owner_search_inner_pending_deferred_replay_payload_count_max"] == 4
    assert (
        result["compact_owner_search_inner_pending_deferred_replay_payload_final_count"]
        == 0
    )
    assert result[
        "compact_owner_search_inner_deferred_one_simulation_replay_flush_sec"
    ] == pytest.approx(0.004)
    assert result[
        "compact_owner_search_inner_device_replay_payload_flush_sec"
    ] == pytest.approx(0.006)
    assert result["compact_owner_search_inner_replay_payload_d2h_bytes"] == pytest.approx(
        0.0
    )
    assert (
        result[
            "compact_owner_search_inner_deferred_one_simulation_action_model_state_digest"
        ]
        == "digest-a"
    )
    assert (
        result[
            "compact_owner_search_inner_deferred_one_simulation_flush_model_state_digest"
        ]
        == "digest-a"
    )


def test_compact_rollout_slab_sample_gate_uses_bounded_replay_ring():
    class FakeSearchService:
        search_impl = "unit_test_bounded_replay_ring"
        num_simulations = 0

        def run(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            selected = np.argmax(root_batch.legal_mask[active_roots], axis=1).astype(
                np.int16,
                copy=False,
            )
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeSearchService(),
        search_lane="unit_test_bounded_replay_ring",
        policy_source="unit_test_bounded_replay_ring",
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=5,
            warmup_steps=1,
            seed=20260526,
            materialize_scalar_timestep=False,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=0,
            compact_rollout_slab_sample_gate_interval=1,
            compact_rollout_slab_sample_gate_replay_pair_capacity=2,
        ),
        compact_rollout_slab=slab,
    )

    assert result["materialized_timestep_count"] == 0
    assert (
        result["compact_rollout_slab_sample_gate_calls"]
        == result["compact_rollout_slab_sample_gate_opportunities"]
    )
    assert result["compact_rollout_slab_sample_gate_replay_ring_pair_capacity"] == 2
    assert result["compact_rollout_slab_sample_gate_replay_ring_entry_count"] == 2
    assert result["compact_rollout_slab_sample_gate_replay_ring_evicted_pair_count"] > 0
    assert (
        result["compact_rollout_slab_committed_index_row_count"]
        > result["compact_rollout_slab_sample_gate_replay_ring_index_row_count"]
    )
    telemetry = result["compact_rollout_slab_sample_gate_last_telemetry"]
    assert telemetry["compact_rollout_slab_sample_gate_mode"] == (
        "compact_replay_ring_fast_sample_batch"
    )
    assert telemetry["compact_rollout_slab_sample_gate_stored_pair_count"] == 2
    assert telemetry["compact_rollout_slab_sample_gate_evicted_pair_count"] > 0
    source_pairs = telemetry["compact_rollout_slab_sample_gate_source_record_pairs"]
    assert len(source_pairs) <= 2
    assert all(pair[0] >= 3 for pair in source_pairs)


def test_compact_replay_ring_empty_sample_is_explicit_noop():
    ring = _CompactReplayRingV1(capacity=2)

    result = ring.sample(seed=17, sample_batch_size=32)

    assert result["sample_batch"] is None
    assert result["index_row_count"] == 0
    assert result["target_row_count"] == 0
    assert result["sample_row_count"] == 0
    telemetry = result["telemetry"]
    assert telemetry["compact_rollout_slab_sample_gate_mode"] == "empty_replay_ring_skipped"
    assert telemetry["compact_rollout_slab_sample_gate_replay_ring_pair_capacity"] == 2
    assert telemetry["compact_rollout_slab_sample_gate_evicted_pair_count"] == 0


def test_compact_replay_ring_tracks_retained_resident_snapshot_bytes():
    if _torch is None:
        pytest.skip("torch unavailable")
    ring = _CompactReplayRingV1(capacity=1)
    snapshot0 = _resident_snapshot_for_terminal_sample_test(
        _torch,
        latest_value=1,
        generation_id=1,
    )
    snapshot1 = _resident_snapshot_for_terminal_sample_test(
        _torch,
        latest_value=2,
        generation_id=2,
    )
    snapshot2 = _resident_snapshot_for_terminal_sample_test(
        _torch,
        latest_value=3,
        generation_id=3,
    )
    step0 = _resident_step_for_terminal_sample_test(snapshot0, action=0, done=False)
    step1 = _resident_step_for_terminal_sample_test(snapshot1, action=1, done=False)
    step2 = _resident_step_for_terminal_sample_test(snapshot2, action=2, done=False)
    snapshot_bytes = int(
        snapshot0.device_observation.numel() * snapshot0.device_observation.element_size()
    )

    ring.append(
        previous_step=step0,
        current_step=step1,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            _torch,
            record_index=0,
            action=0,
            done=False,
        ),
    )

    assert ring.retained_resident_snapshot_count == 2
    assert ring.retained_resident_snapshot_bytes == 2 * snapshot_bytes

    ring.append(
        previous_step=step1,
        current_step=step2,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            _torch,
            record_index=1,
            action=1,
            done=False,
        ),
    )

    assert ring.entry_count == 1
    assert ring.evicted_entry_count == 1
    assert ring.retained_resident_snapshot_count == 2
    assert ring.retained_resident_snapshot_bytes == 2 * snapshot_bytes


def test_compact_replay_ring_sample_snapshot_is_stable_after_append():
    ring = _CompactReplayRingV1(capacity=4, metadata={"unit": "snapshot"})
    action_mask = np.ones((2, 1, ACTION_COUNT), dtype=np.bool_)

    def make_step(*, actions, rewards):
        return SimpleNamespace(
            observation=np.zeros((2, 1, 4, 64, 64), dtype=np.float32),
            action_mask=action_mask,
            reward=np.asarray([[rewards[0]], [rewards[1]]], dtype=np.float32),
            final_reward_map=np.asarray([[99.0], [99.0]], dtype=np.float32),
            done=np.asarray([False, False], dtype=np.bool_),
            payload={"joint_action": np.asarray([[actions[0]], [actions[1]]], dtype=np.int16)},
            compact_batch=None,
        )

    def make_rows(*, record_index, actions, rewards):
        return CompactReplayIndexRowsV1(
            metadata={},
            record_index=record_index,
            next_record_index=record_index + 1,
            compact_root_row=np.asarray([0, 1], dtype=np.int32),
            policy_env_id=np.asarray([0, 1], dtype=np.int32),
            policy_row=np.asarray([0, 1], dtype=np.int32),
            env_row=np.asarray([0, 1], dtype=np.int32),
            player=np.asarray([0, 0], dtype=np.int16),
            action=np.asarray(actions, dtype=np.int16),
            action_mask=np.ones((2, ACTION_COUNT), dtype=np.bool_),
            policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[np.asarray(actions)],
            root_value=np.asarray([0.0, 0.0], dtype=np.float32),
            reward=np.asarray(rewards, dtype=np.float32),
            final_reward=np.asarray(rewards, dtype=np.float32),
            done=np.asarray([False, False], dtype=np.bool_),
            terminated=np.asarray([False, False], dtype=np.bool_),
            truncated=np.asarray([False, False], dtype=np.bool_),
            next_final_observation_row=np.asarray([False, False], dtype=np.bool_),
            to_play=np.asarray([-1, -1], dtype=np.int64),
            policy_source="unit_test_snapshot",
        )

    ring.append(
        previous_step=make_step(actions=[0, 1], rewards=[10.0, 11.0]),
        current_step=make_step(actions=[0, 1], rewards=[10.0, 11.0]),
        index_rows=make_rows(record_index=0, actions=[0, 1], rewards=[10.0, 11.0]),
    )
    snapshot = ring.snapshot_for_sample()
    ring.append(
        previous_step=make_step(actions=[2, 0], rewards=[12.0, 13.0]),
        current_step=make_step(actions=[2, 0], rewards=[12.0, 13.0]),
        index_rows=make_rows(record_index=1, actions=[2, 0], rewards=[12.0, 13.0]),
    )
    result = ring.sample(seed=1, sample_batch_size=2)

    assert len(snapshot.entries) == 1
    assert snapshot.stored_index_row_count == 2
    assert ring.entry_count == 2
    assert result["telemetry"]["compact_rollout_slab_sample_gate_snapshot_version"] > (
        snapshot.snapshot_version
    )


def test_compact_replay_ring_preserves_rng_sample_order_across_pairs():
    ring = _CompactReplayRingV1(capacity=2)
    action_mask = np.ones((2, 1, ACTION_COUNT), dtype=np.bool_)

    def make_step(*, actions, rewards):
        return SimpleNamespace(
            observation=np.zeros((2, 1, 4, 64, 64), dtype=np.float32),
            action_mask=action_mask,
            reward=np.asarray([[rewards[0]], [rewards[1]]], dtype=np.float32),
            final_reward_map=np.asarray([[99.0], [99.0]], dtype=np.float32),
            done=np.asarray([False, False], dtype=np.bool_),
            payload={"joint_action": np.asarray([[actions[0]], [actions[1]]], dtype=np.int16)},
            compact_batch=None,
        )

    def make_rows(*, record_index, actions, rewards):
        return CompactReplayIndexRowsV1(
            metadata={},
            record_index=record_index,
            next_record_index=record_index + 1,
            compact_root_row=np.asarray([0, 1], dtype=np.int32),
            policy_env_id=np.asarray([0, 1], dtype=np.int32),
            policy_row=np.asarray([0, 1], dtype=np.int32),
            env_row=np.asarray([0, 1], dtype=np.int32),
            player=np.asarray([0, 0], dtype=np.int16),
            action=np.asarray(actions, dtype=np.int16),
            action_mask=np.ones((2, ACTION_COUNT), dtype=np.bool_),
            policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[np.asarray(actions)],
            root_value=np.asarray([0.0, 0.0], dtype=np.float32),
            reward=np.asarray(rewards, dtype=np.float32),
            final_reward=np.asarray(rewards, dtype=np.float32),
            done=np.asarray([False, False], dtype=np.bool_),
            terminated=np.asarray([False, False], dtype=np.bool_),
            truncated=np.asarray([False, False], dtype=np.bool_),
            next_final_observation_row=np.asarray([False, False], dtype=np.bool_),
            to_play=np.asarray([-1, -1], dtype=np.int64),
            policy_source="unit_test_rng_order",
        )

    ring.append(
        previous_step=make_step(actions=[0, 1], rewards=[10.0, 11.0]),
        current_step=make_step(actions=[0, 1], rewards=[10.0, 11.0]),
        index_rows=make_rows(record_index=0, actions=[0, 1], rewards=[10.0, 11.0]),
    )
    ring.append(
        previous_step=make_step(actions=[2, 0], rewards=[12.0, 13.0]),
        current_step=make_step(actions=[2, 0], rewards=[12.0, 13.0]),
        index_rows=make_rows(record_index=1, actions=[2, 0], rewards=[12.0, 13.0]),
    )

    result = ring.sample(seed=1, sample_batch_size=4)
    batch = result["sample_batch"]

    np.testing.assert_array_equal(batch.row_id, np.asarray([1, 2, 0, 3], dtype=np.int64))
    np.testing.assert_array_equal(batch.action, np.asarray([1, 2, 0, 0], dtype=np.int16))
    assert batch.metadata["sample_metadata_schema_id"]
    assert batch.metadata["source_target_contract_id"]
    assert batch.metadata["replay_ring_pair_capacity"] == 2
    assert batch.metadata["seed"] == 1
    assert batch.metadata["replace"] is False
    assert batch.metadata["learner_update_claim"] is False


def test_compact_replay_store_durable_state_round_trips_eviction_and_sample_order():
    ring = _CompactReplayRingV1(capacity=2)
    action_mask = np.ones((2, 1, ACTION_COUNT), dtype=np.bool_)

    def make_step(*, actions, rewards, observation_base):
        observation = np.zeros((2, 1, 4, 64, 64), dtype=np.float32)
        observation[0, 0] = float(observation_base)
        observation[1, 0] = float(observation_base + 1)
        return SimpleNamespace(
            observation=observation,
            action_mask=action_mask,
            reward=np.asarray([[rewards[0]], [rewards[1]]], dtype=np.float32),
            final_reward_map=np.asarray([[99.0], [99.0]], dtype=np.float32),
            done=np.asarray([False, False], dtype=np.bool_),
            payload={
                "joint_action": np.asarray(
                    [[actions[0]], [actions[1]]],
                    dtype=np.int16,
                )
            },
            compact_batch=None,
        )

    def make_rows(*, record_index, actions, rewards):
        return CompactReplayIndexRowsV1(
            metadata={"policy_version_ref": "unit-test-policy-v1"},
            record_index=record_index,
            next_record_index=record_index + 1,
            compact_root_row=np.asarray([0, 1], dtype=np.int32),
            policy_env_id=np.asarray([record_index * 10, record_index * 10 + 1]),
            policy_row=np.asarray([0, 1], dtype=np.int32),
            env_row=np.asarray([0, 1], dtype=np.int32),
            player=np.asarray([0, 0], dtype=np.int16),
            action=np.asarray(actions, dtype=np.int16),
            action_mask=np.ones((2, ACTION_COUNT), dtype=np.bool_),
            policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[np.asarray(actions)],
            root_value=np.asarray([0.0, 0.0], dtype=np.float32),
            reward=np.asarray(rewards, dtype=np.float32),
            final_reward=np.asarray(rewards, dtype=np.float32),
            done=np.asarray([False, False], dtype=np.bool_),
            terminated=np.asarray([False, False], dtype=np.bool_),
            truncated=np.asarray([False, False], dtype=np.bool_),
            next_final_observation_row=np.asarray([False, False], dtype=np.bool_),
            to_play=np.asarray([-1, -1], dtype=np.int64),
            policy_source="unit_test_durable_replay_store",
        )

    for record_index, actions, rewards, observation_base in (
        (0, [0, 1], [10.0, 11.0], 0),
        (1, [2, 0], [12.0, 13.0], 10),
        (2, [1, 2], [14.0, 15.0], 20),
    ):
        step = make_step(
            actions=actions,
            rewards=rewards,
            observation_base=observation_base,
        )
        ring.append(
            previous_step=step,
            current_step=step,
            index_rows=make_rows(
                record_index=record_index,
                actions=actions,
                rewards=rewards,
            ),
        )

    with pytest.raises(ValueError, match="policy_version_ref"):
        ring.snapshot_durable_state(policy_version_ref="")

    state = ring.snapshot_durable_state(
        policy_version_ref="unit-test-policy-v1",
        policy_source="unit_test_durable_replay_store",
        model_version_ref="unit-test-model-v1",
        metadata={"support_scale": 9},
    )
    assert [entry.index_rows.record_index for entry in state.entries] == [1, 2]
    assert state.metadata["compact_replay_store_restart_load_tested"] is False

    ring._entries[0].previous_step.observation[1, 0, 0, 0, 0] = 123.0
    ring._entries[0].index_rows.action[1] = 2

    loaded_state = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    restored = _CompactReplayRingV1.from_durable_state(loaded_state)
    result = restored.sample(seed=1, sample_batch_size=4)
    batch = result["sample_batch"]

    assert restored.entry_count == 2
    assert restored.stored_index_row_count == 4
    assert restored.evicted_entry_count == 1
    assert restored.evicted_index_row_count == 2
    np.testing.assert_array_equal(batch.row_id, np.asarray([1, 2, 0, 3], dtype=np.int64))
    np.testing.assert_array_equal(batch.action, np.asarray([0, 1, 2, 2], dtype=np.int16))
    np.testing.assert_array_equal(
        batch.observation[:, 0, 0, 0],
        np.asarray([11.0, 20.0, 10.0, 21.0], dtype=np.float32),
    )
    assert batch.metadata["compact_replay_store_loaded"] is True
    assert batch.metadata["compact_replay_store_load_strict"] is True
    assert batch.metadata["policy_version_ref"] == "unit-test-policy-v1"
    assert batch.metadata["model_version_ref"] == "unit-test-model-v1"
    assert batch.metadata["support_scale"] == 9
    assert batch.metadata["profile_only"] is True
    assert batch.metadata["calls_train_muzero"] is False
    assert batch.metadata["touches_live_runs"] is False
    assert batch.metadata["training_speed_claim"] is False
    assert batch.metadata["stock_replay_storage_claim"] is False
    assert batch.metadata["source_record_pairs"] == [[1, 2], [2, 3]]


def test_compact_replay_ring_scalar_ref_append_strips_observation_payloads():
    ring = _CompactReplayRingV1(capacity=2)
    observation = np.ones((2, 1, 4, 64, 64), dtype=np.float32)
    final_observation = np.full((2, 1, 4, 64, 64), 9.0, dtype=np.float32)
    action_mask = np.ones((2, 1, ACTION_COUNT), dtype=np.bool_)
    step = SimpleNamespace(
        observation=observation,
        action_mask=action_mask,
        reward=np.asarray([[1.0], [2.0]], dtype=np.float32),
        final_reward_map=np.asarray([[10.0], [20.0]], dtype=np.float32),
        done=np.asarray([False, True], dtype=np.bool_),
        payload={
            "joint_action": np.asarray([[1], [2]], dtype=np.int16),
            "unused_large_payload": observation.copy(),
        },
        compact_batch=SimpleNamespace(final_observation=final_observation),
        resident_observation_replay_snapshot=object(),
    )
    index_rows = CompactReplayIndexRowsV1(
        metadata={"policy_version_ref": "unit-test-policy-v1"},
        record_index=3,
        next_record_index=4,
        compact_root_row=np.asarray([0, 1], dtype=np.int32),
        policy_env_id=np.asarray([30, 31], dtype=np.int32),
        policy_row=np.asarray([0, 1], dtype=np.int32),
        env_row=np.asarray([0, 1], dtype=np.int32),
        player=np.asarray([0, 0], dtype=np.int16),
        action=np.asarray([1, 2], dtype=np.int16),
        action_mask=np.ones((2, ACTION_COUNT), dtype=np.bool_),
        policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[np.asarray([1, 2])],
        root_value=np.asarray([0.0, 0.0], dtype=np.float32),
        reward=np.asarray([1.0, 2.0], dtype=np.float32),
        final_reward=np.asarray([10.0, 20.0], dtype=np.float32),
        done=np.asarray([False, True], dtype=np.bool_),
        terminated=np.asarray([False, True], dtype=np.bool_),
        truncated=np.asarray([False, False], dtype=np.bool_),
        next_final_observation_row=np.asarray([False, True], dtype=np.bool_),
        to_play=np.asarray([-1, -1], dtype=np.int64),
        policy_source="unit_test_scalar_ref_append",
    )

    entry = ring.make_scalar_append_delta_entry(
        previous_step=step,
        current_step=step,
        index_rows=index_rows,
    )

    assert entry is not None
    assert entry.previous_step.observation is None
    assert entry.current_step.observation is None
    assert entry.previous_step.resident_observation_replay_snapshot is None
    assert entry.current_step.resident_observation_replay_snapshot is None
    assert entry.current_step.compact_batch is not None
    assert entry.current_step.compact_batch.final_observation is None
    assert "joint_action" in entry.current_step.payload
    assert "unused_large_payload" not in entry.current_step.payload
    assert entry.index_rows.metadata["compact_replay_scalar_ref_append"] is True
    assert entry.index_rows.metadata["compact_replay_append_transport_kind"] == (
        COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
    )


def test_compact_replay_ring_scalar_ref_append_keeps_selected_render_state_rows():
    ring = _CompactReplayRingV1(capacity=2)
    observation = np.ones((3, 1, 4, 64, 64), dtype=np.float32)
    action_mask = np.ones((3, 1, ACTION_COUNT), dtype=np.bool_)
    render_state = {
        "head_x": np.asarray([[10.0], [11.0], [12.0]], dtype=np.float32),
        "head_alive": np.asarray([[1], [0], [1]], dtype=np.uint8),
        "visual_trail_active": np.arange(12, dtype=np.uint8).reshape(3, 4),
        "scalar_state_marker": np.asarray(7, dtype=np.int16),
    }
    autoreset_render_state = {
        "head_x": np.asarray([[100.0], [101.0], [102.0]], dtype=np.float32),
        "head_alive": np.asarray([[0], [1], [0]], dtype=np.uint8),
    }
    step = SimpleNamespace(
        observation=observation,
        action_mask=action_mask,
        reward=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        final_reward_map=np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32),
        done=np.asarray([False, True, False], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1], [2], [0]], dtype=np.int16)},
        compact_batch=None,
        resident_observation_replay_snapshot=object(),
        render_state_snapshot=render_state,
        autoreset_render_state_snapshot=autoreset_render_state,
    )
    index_rows = CompactReplayIndexRowsV1(
        metadata={"policy_version_ref": "unit-test-policy-v1"},
        record_index=3,
        next_record_index=4,
        compact_root_row=np.asarray([0, 1], dtype=np.int32),
        policy_env_id=np.asarray([32, 30], dtype=np.int32),
        policy_row=np.asarray([0, 1], dtype=np.int32),
        env_row=np.asarray([2, 0], dtype=np.int32),
        player=np.asarray([0, 0], dtype=np.int16),
        action=np.asarray([1, 2], dtype=np.int16),
        action_mask=np.ones((2, ACTION_COUNT), dtype=np.bool_),
        policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[np.asarray([1, 2])],
        root_value=np.asarray([0.0, 0.0], dtype=np.float32),
        reward=np.asarray([3.0, 1.0], dtype=np.float32),
        final_reward=np.asarray([30.0, 10.0], dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        terminated=np.asarray([False, False], dtype=np.bool_),
        truncated=np.asarray([False, False], dtype=np.bool_),
        next_final_observation_row=np.asarray([False, False], dtype=np.bool_),
        to_play=np.asarray([-1, -1], dtype=np.int64),
        policy_source="unit_test_scalar_ref_render_state_append",
    )

    entry = ring.make_scalar_append_delta_entry(
        previous_step=step,
        current_step=step,
        index_rows=index_rows,
    )

    assert entry is not None
    assert entry.current_step.observation is None
    assert entry.current_step.resident_observation_replay_snapshot is None
    snapshot = entry.current_step.render_state_snapshot
    assert isinstance(snapshot, CompactReplayRenderStateSnapshotV1)
    assert snapshot.schema_id == COMPACT_REPLAY_RENDER_STATE_SNAPSHOT_SCHEMA_ID
    assert snapshot.source_batch_size == 3
    assert snapshot.player_count == 1
    np.testing.assert_array_equal(snapshot.rows, np.asarray([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(
        snapshot.state["head_x"],
        np.asarray([[10.0], [12.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        snapshot.state["visual_trail_active"],
        np.asarray([[0, 1, 2, 3], [8, 9, 10, 11]], dtype=np.uint8),
    )
    assert "scalar_state_marker" not in snapshot.state
    autoreset_snapshot = entry.current_step.autoreset_render_state_snapshot
    assert isinstance(autoreset_snapshot, CompactReplayRenderStateSnapshotV1)
    np.testing.assert_array_equal(
        autoreset_snapshot.rows,
        np.asarray([0, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        autoreset_snapshot.state["head_x"],
        np.asarray([[100.0], [101.0], [102.0]], dtype=np.float32),
    )


def test_compact_replay_ring_scalar_ref_append_fails_closed_without_provider():
    ring = _CompactReplayRingV1(capacity=2)
    observation = np.ones((1, 1, 4, 64, 64), dtype=np.float32)
    action_mask = np.ones((1, 1, ACTION_COUNT), dtype=np.bool_)
    step = SimpleNamespace(
        observation=observation,
        action_mask=action_mask,
        reward=np.asarray([[1.0]], dtype=np.float32),
        final_reward_map=np.asarray([[1.0]], dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1]], dtype=np.int16)},
        compact_batch=None,
    )
    index_rows = CompactReplayIndexRowsV1(
        metadata={},
        record_index=3,
        next_record_index=4,
        compact_root_row=np.asarray([0], dtype=np.int32),
        policy_env_id=np.asarray([30], dtype=np.int32),
        policy_row=np.asarray([0], dtype=np.int32),
        env_row=np.asarray([0], dtype=np.int32),
        player=np.asarray([0], dtype=np.int16),
        action=np.asarray([1], dtype=np.int16),
        action_mask=np.ones((1, ACTION_COUNT), dtype=np.bool_),
        policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[np.asarray([1])],
        root_value=np.asarray([0.0], dtype=np.float32),
        reward=np.asarray([1.0], dtype=np.float32),
        final_reward=np.asarray([1.0], dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        terminated=np.asarray([False], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        next_final_observation_row=np.asarray([False], dtype=np.bool_),
        to_play=np.asarray([-1], dtype=np.int64),
        policy_source="unit_test_scalar_ref_append",
    )
    entry = ring.make_scalar_append_delta_entry(
        previous_step=step,
        current_step=step,
        index_rows=index_rows,
    )
    assert entry is not None
    ring.append_entry(entry)

    with pytest.raises(ReplayCompatibilityError, match="observation reconstruction provider"):
        ring.sample(seed=1, sample_batch_size=1)


class _FrameValueObservationRenderer:
    backend_name = "unit_test_frame_value_renderer"

    def render(self, request):
        values = np.asarray(request.state["frame_value"], dtype=np.uint8)
        out = np.asarray(request.out)
        for output_row, (state_row, player) in enumerate(
            zip(request.row_indices, request.controlled_players, strict=True)
        ):
            if values.ndim >= 2:
                value = values[int(state_row), int(player)]
            else:
                value = values[int(state_row)]
            out[output_row, 0].fill(np.uint8(value))
        return SourceStateBatchedRenderResult(
            frames=out,
            telemetry={"unit_test_rendered_frame_count": float(out.shape[0])},
        )


def _frame_value_render_state(values) -> dict[str, np.ndarray]:
    return {"frame_value": np.asarray(values, dtype=np.uint8)}


def _provider_bootstrap_step(values, *, done=None, autoreset_values=None):
    values_array = np.asarray(values, dtype=np.uint8)
    batch_size = int(values_array.shape[0])
    done_array = (
        np.zeros(batch_size, dtype=np.bool_)
        if done is None
        else np.asarray(done, dtype=np.bool_)
    )
    return SimpleNamespace(
        observation=None,
        resident_observation_replay_snapshot=None,
        render_state_snapshot=_frame_value_render_state(values_array),
        autoreset_render_state_snapshot=(
            None
            if autoreset_values is None
            else _frame_value_render_state(autoreset_values)
        ),
        done=done_array,
    )


def _solid_stack(batch_size: int, row_values: Mapping[int, list[int]]) -> np.ndarray:
    observation = np.zeros((batch_size, 1, 4, 64, 64), dtype=np.float32)
    for row, values in row_values.items():
        for depth, value in enumerate(values):
            observation[int(row), 0, int(depth)].fill(float(value) / 255.0)
    return observation


def _scalar_ref_index_rows_for_provider_test(
    *,
    env_row: int = 0,
    record_index: int = 0,
    action: int = 1,
    reward: float = 1.0,
    final_reward: float = 1.0,
    done: bool = False,
) -> CompactReplayIndexRowsV1:
    metadata = {
        "done_row_count": int(bool(done)),
        "next_final_observation_row_count": int(bool(done)),
    }
    if bool(done):
        metadata["done_row_indices"] = (0,)
        metadata["next_final_observation_row_indices"] = (0,)
    return CompactReplayIndexRowsV1(
        metadata=metadata,
        record_index=int(record_index),
        next_record_index=int(record_index) + 1,
        compact_root_row=np.asarray([0], dtype=np.int32),
        policy_env_id=np.asarray([30 + int(env_row)], dtype=np.int32),
        policy_row=np.asarray([0], dtype=np.int32),
        env_row=np.asarray([int(env_row)], dtype=np.int32),
        player=np.asarray([0], dtype=np.int16),
        action=np.asarray([int(action)], dtype=np.int16),
        action_mask=np.ones((1, ACTION_COUNT), dtype=np.bool_),
        policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[np.asarray([int(action)])],
        root_value=np.asarray([0.25], dtype=np.float32),
        reward=np.asarray([float(reward)], dtype=np.float32),
        final_reward=np.asarray([float(final_reward)], dtype=np.float32),
        done=np.asarray([bool(done)], dtype=np.bool_),
        terminated=np.asarray([bool(done)], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        next_final_observation_row=np.asarray([bool(done)], dtype=np.bool_),
        to_play=np.asarray([-1], dtype=np.int64),
        policy_source="unit_test_renderer_provider",
    )


def _provider_replay_step(
    *,
    batch_size: int,
    observation: np.ndarray | None,
    frame_values,
    done: bool = False,
    autoreset_values=None,
    action: int = 1,
    reward: float = 1.0,
    final_reward: float = 1.0,
):
    return SimpleNamespace(
        observation=observation,
        action_mask=np.ones((batch_size, 1, ACTION_COUNT), dtype=np.bool_),
        reward=np.full((batch_size, 1), float(reward), dtype=np.float32),
        final_reward_map=np.full((batch_size, 1), float(final_reward), dtype=np.float32),
        done=np.full(batch_size, bool(done), dtype=np.bool_),
        payload={"joint_action": np.full((batch_size, 1), int(action), dtype=np.int16)},
        compact_batch=SimpleNamespace(
            final_observation=None,
            observation_source=COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
        ),
        resident_observation_replay_snapshot=None,
        render_state_snapshot=_frame_value_render_state(frame_values),
        autoreset_render_state_snapshot=(
            None
            if autoreset_values is None
            else _frame_value_render_state(autoreset_values)
        ),
    )


def test_compact_replay_renderer_provider_matches_durable_row_sliced_sample():
    provider = CompactReplayRendererBackedObservationProviderV1(
        batch_size=2,
        player_count=1,
        renderer=_FrameValueObservationRenderer(),
    )
    for value in (1, 2, 3, 4):
        provider.bootstrap_compact_replay_step(
            _provider_bootstrap_step([[99], [value]])
        )
    previous_observation = _solid_stack(2, {1: [1, 2, 3, 4]})
    current_observation = _solid_stack(2, {1: [2, 3, 4, 5]})
    previous_step = _provider_replay_step(
        batch_size=2,
        observation=previous_observation,
        frame_values=[[99], [4]],
    )
    current_step = _provider_replay_step(
        batch_size=2,
        observation=current_observation,
        frame_values=[[77], [5]],
        reward=3.0,
        final_reward=3.0,
    )
    index_rows = _scalar_ref_index_rows_for_provider_test(
        env_row=1,
        reward=3.0,
        final_reward=3.0,
    )
    durable = _CompactReplayRingV1(capacity=2)
    durable.append(
        previous_step=previous_step,
        current_step=current_step,
        index_rows=index_rows,
    )
    scalar = _CompactReplayRingV1(capacity=2)
    scalar_entry = scalar.make_scalar_append_delta_entry(
        previous_step=previous_step,
        current_step=current_step,
        index_rows=index_rows,
    )
    assert scalar_entry is not None
    assert scalar_entry.current_step.render_state_snapshot is not None
    np.testing.assert_array_equal(
        scalar_entry.current_step.render_state_snapshot.rows,
        np.asarray([1], dtype=np.int32),
    )
    scalar.append_entry(scalar_entry)
    scalar.set_observation_provider(provider)

    durable_batch = durable.sample(seed=3, sample_batch_size=1)["sample_batch"]
    scalar_result = scalar.sample(seed=3, sample_batch_size=1)
    scalar_batch = scalar_result["sample_batch"]

    np.testing.assert_allclose(scalar_batch.observation, durable_batch.observation)
    np.testing.assert_allclose(scalar_batch.next_observation, durable_batch.next_observation)
    np.testing.assert_array_equal(scalar_batch.action_mask, durable_batch.action_mask)
    np.testing.assert_array_equal(scalar_batch.action, durable_batch.action)
    np.testing.assert_allclose(scalar_batch.reward, durable_batch.reward)
    assert scalar_batch.metadata["observation_provider_used"] is True
    assert provider.materialized_entry_count == 1


def test_compact_replay_renderer_provider_captures_terminal_before_autoreset():
    provider = CompactReplayRendererBackedObservationProviderV1(
        batch_size=1,
        player_count=1,
        renderer=_FrameValueObservationRenderer(),
    )
    for value in (1, 2, 3, 4):
        provider.bootstrap_compact_replay_step(_provider_bootstrap_step([[value]]))
    terminal_entry = SimpleNamespace(
        previous_step=_provider_replay_step(
            batch_size=1,
            observation=None,
            frame_values=[[4]],
        ),
        current_step=_provider_replay_step(
            batch_size=1,
            observation=None,
            frame_values=[[5]],
            done=True,
            autoreset_values=[[9]],
            reward=5.0,
            final_reward=50.0,
        ),
        index_rows=_scalar_ref_index_rows_for_provider_test(
            done=True,
            reward=5.0,
            final_reward=50.0,
        ),
    )
    terminal_entry.previous_step.compact_batch.observation_source = (
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    )
    terminal_entry.current_step.compact_batch.observation_source = (
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    )

    materialized_terminal = provider.materialize_compact_replay_entry(terminal_entry)
    final_observation = materialized_terminal.current_step.compact_batch.final_observation
    assert materialized_terminal.previous_step.compact_batch.observation_source == (
        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
    )
    assert materialized_terminal.current_step.compact_batch.observation_source == (
        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
    )
    np.testing.assert_allclose(
        final_observation[0, 0, :, 0, 0],
        np.asarray([2, 3, 4, 5], dtype=np.float32) / 255.0,
    )

    next_entry = SimpleNamespace(
        previous_step=_provider_replay_step(
            batch_size=1,
            observation=None,
            frame_values=[[9]],
        ),
        current_step=_provider_replay_step(
            batch_size=1,
            observation=None,
            frame_values=[[10]],
        ),
        index_rows=_scalar_ref_index_rows_for_provider_test(record_index=1),
    )
    next_entry.previous_step.compact_batch.observation_source = (
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    )
    next_entry.current_step.compact_batch.observation_source = (
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    )
    materialized_next = provider.materialize_compact_replay_entry(next_entry)
    assert materialized_next.previous_step.compact_batch.observation_source == (
        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
    )
    assert materialized_next.current_step.compact_batch.observation_source == (
        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
    )
    np.testing.assert_allclose(
        materialized_next.previous_step.observation[0, 0, :, 0, 0],
        np.asarray([0, 0, 0, 9], dtype=np.float32) / 255.0,
    )
    np.testing.assert_allclose(
        materialized_next.current_step.observation[0, 0, :, 0, 0],
        np.asarray([0, 0, 9, 10], dtype=np.float32) / 255.0,
    )


def test_compact_replay_renderer_provider_fails_without_bootstrap_history():
    provider = CompactReplayRendererBackedObservationProviderV1(
        batch_size=1,
        player_count=1,
        renderer=_FrameValueObservationRenderer(),
    )
    entry = SimpleNamespace(
        previous_step=_provider_replay_step(
            batch_size=1,
            observation=None,
            frame_values=[[0]],
        ),
        current_step=_provider_replay_step(
            batch_size=1,
            observation=None,
            frame_values=[[1]],
        ),
        index_rows=_scalar_ref_index_rows_for_provider_test(),
    )

    with pytest.raises(ReplayCompatibilityError, match="missing bootstrap stack history"):
        provider.materialize_compact_replay_entry(entry)
    assert provider.missing_stack_history_count == 1


def test_compact_replay_ring_scalar_ref_observation_provider_matches_durable_sample():
    previous_observation = np.zeros((1, 1, 4, 64, 64), dtype=np.float32)
    previous_observation[0, 0] = 3.0
    current_observation = np.zeros((1, 1, 4, 64, 64), dtype=np.float32)
    current_observation[0, 0] = 5.0
    final_observation = np.zeros((1, 1, 4, 64, 64), dtype=np.float32)
    final_observation[0, 0] = 42.0
    action_mask = np.ones((1, 1, ACTION_COUNT), dtype=np.bool_)
    previous_step = SimpleNamespace(
        observation=previous_observation,
        action_mask=action_mask,
        reward=np.asarray([[0.0]], dtype=np.float32),
        final_reward_map=np.asarray([[0.0]], dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1]], dtype=np.int16)},
        compact_batch=None,
    )
    current_step = SimpleNamespace(
        observation=current_observation,
        action_mask=action_mask,
        reward=np.asarray([[7.0]], dtype=np.float32),
        final_reward_map=np.asarray([[70.0]], dtype=np.float32),
        done=np.asarray([True], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1]], dtype=np.int16)},
        compact_batch=SimpleNamespace(
            final_observation=final_observation,
            observation_source=COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
        ),
    )
    index_rows = CompactReplayIndexRowsV1(
        metadata={},
        record_index=3,
        next_record_index=4,
        compact_root_row=np.asarray([0], dtype=np.int32),
        policy_env_id=np.asarray([30], dtype=np.int32),
        policy_row=np.asarray([0], dtype=np.int32),
        env_row=np.asarray([0], dtype=np.int32),
        player=np.asarray([0], dtype=np.int16),
        action=np.asarray([1], dtype=np.int16),
        action_mask=np.ones((1, ACTION_COUNT), dtype=np.bool_),
        policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[np.asarray([1])],
        root_value=np.asarray([0.25], dtype=np.float32),
        reward=np.asarray([7.0], dtype=np.float32),
        final_reward=np.asarray([70.0], dtype=np.float32),
        done=np.asarray([True], dtype=np.bool_),
        terminated=np.asarray([True], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        next_final_observation_row=np.asarray([True], dtype=np.bool_),
        to_play=np.asarray([-1], dtype=np.int64),
        policy_source="unit_test_scalar_ref_provider",
    )
    durable = _CompactReplayRingV1(capacity=2)
    durable.append(
        previous_step=previous_step,
        current_step=current_step,
        index_rows=index_rows,
    )
    scalar = _CompactReplayRingV1(capacity=2)
    scalar_entry = scalar.make_scalar_append_delta_entry(
        previous_step=previous_step,
        current_step=current_step,
        index_rows=index_rows,
    )
    assert scalar_entry is not None
    scalar.append_entry(scalar_entry)

    class Provider:
        def materialize_compact_replay_entry(self, entry):
            return replace(
                entry,
                previous_step=previous_step,
                current_step=current_step,
            )

    scalar.set_observation_provider(Provider())
    durable_result = durable.sample(seed=1, sample_batch_size=1)
    scalar_result = scalar.sample_from_snapshot(
        scalar.snapshot_for_sample(),
        seed=1,
        sample_batch_size=1,
    )
    durable_batch = durable_result["sample_batch"]
    scalar_batch = scalar_result["sample_batch"]

    np.testing.assert_array_equal(scalar_batch.observation, durable_batch.observation)
    np.testing.assert_array_equal(
        scalar_batch.next_observation,
        durable_batch.next_observation,
    )
    np.testing.assert_array_equal(scalar_batch.action, durable_batch.action)
    np.testing.assert_array_equal(scalar_batch.action_mask, durable_batch.action_mask)
    np.testing.assert_array_equal(scalar_batch.reward, durable_batch.reward)
    np.testing.assert_array_equal(scalar_batch.final_reward, durable_batch.final_reward)
    np.testing.assert_array_equal(scalar_batch.done, durable_batch.done)
    assert scalar_batch.metadata["observation_provider_used"] is True
    assert scalar_batch.metadata["observation_provider_materialized_entry_count"] == 1
    assert scalar_result["telemetry"][
        "compact_rollout_slab_sample_gate_observation_provider_used"
    ] is True


def test_compact_replay_ring_scalar_ref_provider_matches_durable_unroll2_sample():
    torch = pytest.importorskip("torch")
    snapshot0 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=3,
        generation_id=1,
    )
    snapshot1 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=4,
        generation_id=2,
    )
    snapshot2_terminal = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=99,
        generation_id=3,
        final_latest_value=42,
    )
    snapshot3_reset = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=6,
        generation_id=4,
    )
    step0 = _resident_step_for_terminal_sample_test(snapshot0, done=False)
    step1 = _resident_step_for_terminal_sample_test(
        snapshot1,
        action=1,
        reward=2.0,
        final_reward=2.0,
        done=False,
    )
    step2_terminal = _resident_step_for_terminal_sample_test(
        snapshot2_terminal,
        action=2,
        reward=5.0,
        final_reward=50.0,
        done=True,
    )
    step3_reset = _resident_step_for_terminal_sample_test(
        snapshot3_reset,
        action=0,
        reward=0.25,
        final_reward=0.25,
        done=False,
    )
    entries = (
        (
            step0,
            step1,
            _resident_device_index_rows_for_terminal_sample_test(
                torch,
                record_index=0,
                action=1,
                policy_target=np.asarray([0.2, 0.7, 0.1], dtype=np.float32),
                root_value=-11.0,
                reward=2.0,
                final_reward=2.0,
                done=False,
            ),
        ),
        (
            step1,
            step2_terminal,
            _resident_device_index_rows_for_terminal_sample_test(
                torch,
                record_index=1,
                action=2,
                policy_target=np.asarray([0.1, 0.3, 0.6], dtype=np.float32),
                root_value=-22.0,
                reward=5.0,
                final_reward=50.0,
                done=True,
            ),
        ),
        (
            step2_terminal,
            step3_reset,
            _resident_device_index_rows_for_terminal_sample_test(
                torch,
                record_index=2,
                action=0,
                policy_target=np.asarray([0.8, 0.1, 0.1], dtype=np.float32),
                root_value=-33.0,
                reward=0.25,
                final_reward=0.25,
                done=False,
            ),
        ),
    )
    durable = _CompactReplayRingV1(capacity=4)
    scalar = _CompactReplayRingV1(capacity=4)
    originals = {}
    for previous_step, current_step, index_rows in entries:
        durable.append(
            previous_step=previous_step,
            current_step=current_step,
            index_rows=index_rows,
        )
        scalar_entry = scalar.make_scalar_append_delta_entry(
            previous_step=previous_step,
            current_step=current_step,
            index_rows=index_rows,
        )
        assert scalar_entry is not None
        scalar.append_entry(scalar_entry)
        originals[int(getattr(index_rows, "record_index"))] = (
            previous_step,
            current_step,
        )

    class Provider:
        def materialize_compact_replay_entry(self, entry):
            record_index = int(getattr(entry.index_rows, "record_index"))
            previous_step, current_step = originals[record_index]
            return replace(
                entry,
                previous_step=previous_step,
                current_step=current_step,
            )

    scalar.set_observation_provider(Provider())
    sample_kwargs = {
        "seed": 11,
        "sample_batch_size": 0,
        "require_next_targets": True,
        "num_unroll_steps": 2,
    }
    durable_result = durable.sample(**sample_kwargs)
    scalar_result = scalar.sample_from_snapshot(
        scalar.snapshot_for_sample(),
        **sample_kwargs,
    )
    durable_batch = durable_result["sample_batch"]
    scalar_batch = scalar_result["sample_batch"]

    for field in (
        "observation",
        "next_observation",
        "action",
        "action_mask",
        "reward",
        "final_reward",
        "done",
        "terminated",
        "truncated",
        "next_action_mask",
        "unroll_action",
        "unroll_reward",
        "unroll_action_mask",
        "unroll_action_valid_mask",
        "unroll_reward_valid_mask",
        "unroll_policy_valid_mask",
        "unroll_value_valid_mask",
        "unroll_done",
        "unroll_terminated",
        "unroll_truncated",
    ):
        torch.testing.assert_close(
            getattr(scalar_batch, field),
            getattr(durable_batch, field),
        )
    assert scalar_batch.metadata["explicit_unroll_targets"] is True
    assert scalar_batch.metadata["observation_provider_used"] is True
    assert scalar_batch.metadata["observation_provider_materialized_entry_count"] > 0
    assert scalar_result["telemetry"][
        "compact_rollout_slab_sample_gate_observation_provider_used"
    ] is True


def test_compact_replay_ring_host_scalar_ref_unroll2_builds_learner_batch():
    torch = pytest.importorskip("torch")
    step0 = _provider_replay_step(
        batch_size=1,
        observation=_solid_stack(1, {0: [1, 2, 3, 4]}),
        frame_values=[[4]],
        action=1,
        reward=0.0,
        final_reward=0.0,
        done=False,
    )
    step1 = _provider_replay_step(
        batch_size=1,
        observation=_solid_stack(1, {0: [2, 3, 4, 5]}),
        frame_values=[[5]],
        action=1,
        reward=2.0,
        final_reward=2.0,
        done=False,
    )
    step2_terminal = _provider_replay_step(
        batch_size=1,
        observation=_solid_stack(1, {0: [3, 4, 5, 6]}),
        frame_values=[[6]],
        autoreset_values=[[9]],
        action=2,
        reward=5.0,
        final_reward=50.0,
        done=True,
    )
    step2_terminal.compact_batch.final_observation = _solid_stack(1, {0: [3, 4, 5, 6]})
    step3_reset = _provider_replay_step(
        batch_size=1,
        observation=_solid_stack(1, {0: [0, 0, 0, 9]}),
        frame_values=[[9]],
        action=0,
        reward=0.25,
        final_reward=0.25,
        done=False,
    )
    entries = (
        (
            step0,
            step1,
            _scalar_ref_index_rows_for_provider_test(
                record_index=0,
                action=1,
                reward=2.0,
                final_reward=2.0,
                done=False,
            ),
        ),
        (
            step1,
            step2_terminal,
            _scalar_ref_index_rows_for_provider_test(
                record_index=1,
                action=2,
                reward=5.0,
                final_reward=50.0,
                done=True,
            ),
        ),
        (
            step2_terminal,
            step3_reset,
            _scalar_ref_index_rows_for_provider_test(
                record_index=2,
                action=0,
                reward=0.25,
                final_reward=0.25,
                done=False,
            ),
        ),
    )
    durable = _CompactReplayRingV1(capacity=4)
    scalar = _CompactReplayRingV1(capacity=4)
    originals = {}
    for previous_step, current_step, index_rows in entries:
        durable.append(
            previous_step=previous_step,
            current_step=current_step,
            index_rows=index_rows,
        )
        scalar_entry = scalar.make_scalar_append_delta_entry(
            previous_step=previous_step,
            current_step=current_step,
            index_rows=index_rows,
        )
        assert scalar_entry is not None
        scalar.append_entry(scalar_entry)
        originals[int(index_rows.record_index)] = (previous_step, current_step)

    class Provider:
        def materialize_compact_replay_entry(self, entry):
            previous_step, current_step = originals[int(entry.index_rows.record_index)]
            return replace(
                entry,
                previous_step=previous_step,
                current_step=current_step,
            )

    scalar.set_observation_provider(Provider())
    sample_kwargs = {
        "seed": 11,
        "sample_batch_size": 0,
        "require_next_targets": True,
        "num_unroll_steps": 2,
    }
    durable_batch = durable.sample(**sample_kwargs)["sample_batch"]
    scalar_result = scalar.sample(**sample_kwargs)
    scalar_batch = scalar_result["sample_batch"]

    assert scalar_batch.metadata["explicit_unroll_targets"] is True
    assert scalar_batch.metadata["observation_provider_used"] is True
    assert scalar_batch.metadata["observation_provider_materialized_entry_count"] > 0
    assert scalar_result["telemetry"][
        "compact_rollout_slab_sample_gate_explicit_unroll_targets"
    ] is True
    assert scalar_result["telemetry"][
        "compact_rollout_slab_sample_gate_terminal_sample_row_count"
    ] == 1, (
        scalar_batch.record_index.tolist(),
        scalar_batch.done.tolist(),
        scalar_batch.metadata,
        scalar_result["telemetry"],
    )
    assert scalar_result["telemetry"][
        "compact_rollout_slab_sample_gate_next_final_observation_row_count"
    ] == 1
    assert scalar_result["telemetry"][
        "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count"
    ] > 0
    assert scalar_batch.metadata["terminal_sample_row_count"] == 1
    assert scalar_batch.metadata["next_final_observation_row_count"] == 1
    assert scalar_batch.metadata["terminal_unroll_value_target_row_count"] > 0
    assert bool(scalar_batch.done.any())
    terminal_position = int(np.flatnonzero(scalar_batch.done)[0])
    np.testing.assert_allclose(
        scalar_batch.next_observation[terminal_position, :, 0, 0],
        np.asarray([3, 4, 5, 6], dtype=np.float32) / 255.0,
    )
    for field in (
        "observation",
        "next_observation",
        "action",
        "action_mask",
        "policy_target",
        "root_value",
        "reward",
        "final_reward",
        "done",
        "terminated",
        "truncated",
        "next_action_mask",
        "next_policy_target",
        "next_root_value",
        "unroll_action",
        "unroll_reward",
        "unroll_policy_target",
        "unroll_root_value",
        "unroll_action_mask",
        "unroll_action_valid_mask",
        "unroll_reward_valid_mask",
        "unroll_policy_valid_mask",
        "unroll_value_valid_mask",
        "unroll_done",
        "unroll_terminated",
        "unroll_truncated",
    ):
        np.testing.assert_array_equal(
            getattr(scalar_batch, field),
            getattr(durable_batch, field),
        )
    learner_batch = build_compact_muzero_learner_batch_v1(
        scalar_batch,
        config=CompactMuZeroLearnerConfigV1(
            device="cpu",
            num_unroll_steps=2,
            require_resident_sample=False,
        ),
    )
    assert learner_batch.action.shape[1] == 2
    assert learner_batch.target_reward.shape[1] == 2
    assert learner_batch.target_policy.shape[1:] == (3, ACTION_COUNT)
    assert learner_batch.target_value.shape[1] == 3
    torch.testing.assert_close(
        learner_batch.action[:, 0].cpu(),
        torch.as_tensor(scalar_batch.action, dtype=torch.long),
        rtol=0,
        atol=0,
    )
    fused_result = scalar.sample(
        **sample_kwargs,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    fused_batch = fused_result["learner_batch"]
    assert fused_result["sample_batch"] is None
    assert fused_batch.metadata["host_provider_direct_write_learner_batch"] is True
    assert (
        fused_batch.metadata["compact_muzero_learner_batch_prevalidation_source"]
        == "host_provider_sample_batch_builder_v1"
    )
    assert fused_batch.metadata["terminal_sample_row_count"] == 1
    assert fused_result["telemetry"][
        "compact_rollout_slab_sample_gate_host_provider_learner_batch"
    ] is True
    assert fused_result["telemetry"][
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"
    ] is True
    assert (
        fused_result["telemetry"][
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source"
        ]
        == "host_provider_sample_batch_builder_v1"
    )
    for field in (
        "observation",
        "action",
        "action_mask",
        "target_reward",
        "target_value",
        "target_policy",
        "target_reward_mask",
        "target_value_mask",
        "target_policy_mask",
        "action_valid_mask",
        "done",
        "terminated",
        "truncated",
    ):
        torch.testing.assert_close(getattr(fused_batch, field), getattr(learner_batch, field))


def test_compact_fast_sample_uses_index_reward_and_terminal_final_observation():
    observation_shape = (1, 2, 4, 64, 64)
    previous_observation = np.zeros(observation_shape, dtype=np.float32)
    previous_observation[0, 0] = 3.0
    current_observation = np.zeros(observation_shape, dtype=np.float32)
    current_observation[0, 0] = 5.0
    final_observation = np.zeros(observation_shape, dtype=np.float32)
    final_observation[0, 0] = 42.0
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=np.bool_)

    previous_step = SimpleNamespace(
        observation=previous_observation,
        action_mask=action_mask,
    )
    current_step = SimpleNamespace(
        observation=current_observation,
        action_mask=action_mask,
        reward=np.asarray([[7.0, 0.0]], dtype=np.float32),
        final_reward_map=np.asarray([[70.0, 0.0]], dtype=np.float32),
        done=np.asarray([True], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1, 0]], dtype=np.int16)},
        compact_batch=SimpleNamespace(final_observation=final_observation),
    )
    index_rows = SimpleNamespace(
        env_row=np.asarray([0], dtype=np.int32),
        player=np.asarray([0], dtype=np.int16),
        action=np.asarray([1], dtype=np.int16),
        action_mask=np.asarray([[True, True, True]], dtype=np.bool_),
        policy_target=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        root_value=np.asarray([0.25], dtype=np.float32),
        reward=np.asarray([7.0], dtype=np.float32),
        final_reward=np.asarray([70.0], dtype=np.float32),
        done=np.asarray([True], dtype=np.bool_),
        terminated=np.asarray([True], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        next_final_observation_row=np.asarray([True], dtype=np.bool_),
        policy_row=np.asarray([0], dtype=np.int32),
        record_index=11,
        next_record_index=12,
    )

    batch = _build_compact_sample_batch_from_index_rows_fast(
        previous_step=previous_step,
        current_step=current_step,
        source_index_row=np.asarray([5], dtype=np.int64),
        index_rows=index_rows,
        seed=19,
        replace=False,
    )

    assert batch.final_reward.tolist() == [70.0]
    assert batch.reward.tolist() == [7.0]
    assert batch.done.tolist() == [True]
    np.testing.assert_array_equal(batch.observation[0], previous_observation[0, 0])
    np.testing.assert_array_equal(batch.next_observation[0], final_observation[0, 0])


def test_host_compact_sample_batch_preserves_explicit_next_targets_through_concat():
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=np.bool_)
    previous_step = SimpleNamespace(
        observation=np.zeros((1, 2, 4, 64, 64), dtype=np.float32),
        action_mask=action_mask,
    )
    current_step = SimpleNamespace(
        observation=np.ones((1, 2, 4, 64, 64), dtype=np.float32),
        action_mask=action_mask,
        reward=np.asarray([[1.0, 2.0]], dtype=np.float32),
        final_reward_map=np.asarray([[1.0, 20.0]], dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1, 2]], dtype=np.int16)},
        compact_batch=SimpleNamespace(final_observation=None),
    )
    index_rows = SimpleNamespace(
        env_row=np.asarray([0, 0], dtype=np.int32),
        player=np.asarray([0, 1], dtype=np.int16),
        action=np.asarray([1, 2], dtype=np.int16),
        action_mask=np.ones((2, ACTION_COUNT), dtype=np.bool_),
        policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[[1, 2]],
        root_value=np.asarray([0.25, 0.5], dtype=np.float32),
        reward=np.asarray([1.0, 2.0], dtype=np.float32),
        final_reward=np.asarray([1.0, 2.0], dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        terminated=np.asarray([False, False], dtype=np.bool_),
        truncated=np.asarray([False, False], dtype=np.bool_),
        next_final_observation_row=np.asarray([False, False], dtype=np.bool_),
        policy_row=np.asarray([0, 1], dtype=np.int32),
        record_index=11,
        next_record_index=12,
    )
    next_index_rows = SimpleNamespace(
        action_mask=np.asarray(
            [[True, False, True], [False, True, True]],
            dtype=np.bool_,
        ),
        policy_target=np.asarray(
            [[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        root_value=np.asarray([3.0, 4.0], dtype=np.float32),
    )

    first = _build_compact_sample_batch_from_index_rows_fast(
        previous_step=previous_step,
        current_step=current_step,
        source_index_row=np.asarray([5], dtype=np.int64),
        index_rows=SimpleNamespace(
            **{
                key: (value[:1] if isinstance(value, np.ndarray) else value)
                for key, value in vars(index_rows).items()
            }
        ),
        next_index_rows=SimpleNamespace(
            action_mask=next_index_rows.action_mask[:1],
            policy_target=next_index_rows.policy_target[:1],
            root_value=next_index_rows.root_value[:1],
        ),
        seed=19,
        replace=False,
    )
    second = _build_compact_sample_batch_from_index_rows_fast(
        previous_step=previous_step,
        current_step=current_step,
        source_index_row=np.asarray([6], dtype=np.int64),
        index_rows=SimpleNamespace(
            **{
                key: (value[1:] if isinstance(value, np.ndarray) else value)
                for key, value in vars(index_rows).items()
            }
        ),
        next_index_rows=SimpleNamespace(
            action_mask=next_index_rows.action_mask[1:],
            policy_target=next_index_rows.policy_target[1:],
            root_value=next_index_rows.root_value[1:],
        ),
        seed=20,
        replace=False,
    )
    combined = _concat_compact_sample_batches_fast(
        [first, second],
        metadata={"source": "unit_test_host_next_targets"},
        sample_position_order=np.asarray([1, 0], dtype=np.int64),
    )

    assert first.metadata["explicit_next_targets"] is True
    assert second.metadata["explicit_next_targets"] is True
    assert combined.metadata["explicit_next_targets"] is True
    assert combined.next_action_mask is not None
    assert combined.next_policy_target is not None
    assert combined.next_root_value is not None
    np.testing.assert_array_equal(combined.row_id, np.asarray([6, 5], dtype=np.int64))
    np.testing.assert_array_equal(combined.next_action_mask, next_index_rows.action_mask[[1, 0]])
    np.testing.assert_allclose(combined.next_policy_target, next_index_rows.policy_target[[1, 0]])
    np.testing.assert_allclose(combined.next_root_value, next_index_rows.root_value[[1, 0]])


def _resident_snapshot_for_terminal_sample_test(
    torch,
    *,
    latest_value: int,
    generation_id: int,
    final_latest_value: int | None = None,
) -> ResidentObservationBatchV1:
    device_observation = torch.zeros((1, 1, 4, 8, 8), dtype=torch.uint8)
    device_observation[0, 0, -1, 0, 0] = int(latest_value)
    final_device_observation = None
    root_final_device_observation = None
    final_observation_row_mask = np.asarray([False], dtype=np.bool_)
    if final_latest_value is not None:
        final_device_observation = device_observation.clone()
        final_device_observation[0, 0, -1, 0, 0] = int(final_latest_value)
        root_final_device_observation = final_device_observation.reshape(1, 4, 8, 8)
        final_observation_row_mask = np.asarray([True], dtype=np.bool_)
    return ResidentObservationBatchV1(
        device_observation=device_observation,
        root_device_observation=device_observation.reshape(1, 4, 8, 8),
        generation_id=int(generation_id),
        batch_size=1,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=int(generation_id),
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
        final_device_observation=final_device_observation,
        root_final_device_observation=root_final_device_observation,
        final_observation_row_mask=final_observation_row_mask,
    )


def _resident_frame_history_snapshot_from_full_test(
    torch,
    snapshot: ResidentObservationBatchV1,
) -> CompactResidentFrameStackReplaySnapshotV1:
    frame_history = tuple(
        snapshot.device_observation[:, :, frame_index].detach().clone().contiguous()
        for frame_index in range(int(snapshot.stack_shape[0]))
    )
    final_mask = np.asarray(
        snapshot.final_observation_row_mask
        if snapshot.final_observation_row_mask is not None
        else np.zeros((int(snapshot.batch_size),), dtype=np.bool_),
        dtype=np.bool_,
    ).reshape(-1)
    final_rows = np.flatnonzero(final_mask).astype(np.int64, copy=False)
    final_device_observation_rows = None
    final_device_observation_row_indices = None
    if final_rows.size:
        source = (
            snapshot.final_device_observation
            if snapshot.final_device_observation is not None
            else snapshot.device_observation
        )
        final_device_observation_rows = source.index_select(
            0,
            torch.as_tensor(final_rows, dtype=torch.long, device=source.device),
        ).detach().clone().contiguous()
        final_device_observation_row_indices = final_rows.astype(np.int32, copy=True)
    return CompactResidentFrameStackReplaySnapshotV1(
        device_frame_history=frame_history,
        generation_id=int(snapshot.generation_id),
        batch_size=int(snapshot.batch_size),
        player_count=int(snapshot.player_count),
        stack_shape=tuple(int(dim) for dim in snapshot.stack_shape),
        dtype=str(snapshot.dtype),
        device=str(snapshot.device),
        row_major_order=True,
        fresh_for_step_index=int(snapshot.fresh_for_step_index),
        source_backend=str(snapshot.source_backend),
        host_fallback_allowed=False,
        metadata={"resident_observation_replay_snapshot_mode": "latest_frame_history"},
        final_observation_row_mask=final_mask.copy(),
        final_device_observation_rows=final_device_observation_rows,
        final_device_observation_row_indices=final_device_observation_row_indices,
    )


def _resident_step_for_terminal_sample_test(
    snapshot,
    *,
    action: int = 1,
    reward: float = 7.0,
    final_reward: float = 70.0,
    done: bool = True,
):
    return SimpleNamespace(
        observation=np.zeros((1, 1, 4, 8, 8), dtype=np.float32),
        action_mask=np.ones((1, 1, ACTION_COUNT), dtype=np.bool_),
        reward=np.asarray([[reward]], dtype=np.float32),
        final_reward_map=np.asarray([[final_reward]], dtype=np.float32),
        done=np.asarray([done], dtype=np.bool_),
        payload={"joint_action": np.asarray([[action]], dtype=np.int16)},
        compact_batch=SimpleNamespace(
            observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        ),
        resident_observation_replay_snapshot=snapshot,
    )


def _resident_device_index_rows_for_terminal_sample_test(
    torch,
    *,
    record_index: int = 0,
    action: int = 1,
    policy_target: np.ndarray | None = None,
    root_value: float = 0.25,
    reward: float = 7.0,
    final_reward: float = 70.0,
    done: bool = True,
) -> CompactDeviceReplayIndexRowsV1:
    if policy_target is None:
        policy = torch.zeros((1, ACTION_COUNT), dtype=torch.float32)
        policy[0, int(action)] = 1.0
    else:
        policy = torch.as_tensor(
            np.asarray(policy_target, dtype=np.float32).reshape(1, ACTION_COUNT),
            dtype=torch.float32,
        )
    done_tensor = torch.tensor([done], dtype=torch.bool)
    return CompactDeviceReplayIndexRowsV1(
        metadata={
            "device_replay_index_rows": True,
            "done_row_count": int(done),
            "next_final_observation_row_count": int(done),
            "host_search_payload_fallback_allowed": False,
        },
        record_index=int(record_index),
        next_record_index=int(record_index) + 1,
        compact_root_row=torch.tensor([0], dtype=torch.int32),
        policy_env_id=torch.tensor([record_index], dtype=torch.int64),
        policy_row=torch.tensor([0], dtype=torch.int32),
        env_row=torch.tensor([0], dtype=torch.int32),
        player=torch.tensor([0], dtype=torch.int16),
        action=torch.tensor([action], dtype=torch.int16),
        action_mask=torch.ones((1, ACTION_COUNT), dtype=torch.bool),
        policy_target=policy,
        root_value=torch.tensor([root_value], dtype=torch.float32),
        reward=torch.tensor([reward], dtype=torch.float32),
        final_reward=torch.tensor([final_reward], dtype=torch.float32),
        done=done_tensor,
        terminated=done_tensor.clone(),
        truncated=torch.zeros((1,), dtype=torch.bool),
        next_final_observation_row=done_tensor.clone(),
        to_play=torch.tensor([-1], dtype=torch.int64),
        policy_source="unit_test_resident_terminal",
    )


def test_compact_replay_ring_resident_device_sample_uses_terminal_final_observation():
    torch = pytest.importorskip("torch")
    previous_snapshot = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=3,
        generation_id=1,
    )
    current_snapshot = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=99,
        generation_id=2,
        final_latest_value=42,
    )
    previous_step = _resident_step_for_terminal_sample_test(
        previous_snapshot,
        done=False,
    )
    current_step = _resident_step_for_terminal_sample_test(current_snapshot)
    ring = _CompactReplayRingV1(capacity=2)
    ring.append(
        previous_step=previous_step,
        current_step=current_step,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(torch),
    )

    result = ring.sample(seed=3, sample_batch_size=1)
    sample = result["sample_batch"]

    assert sample.metadata["resident_device_sample_batch"] is True
    assert sample.metadata["device_replay_index_rows_sample"] is True
    assert sample.metadata["resident_grouped_device_replay_sample_batch"] is True
    assert sample.metadata["resident_terminal_final_observation_used"] is True
    assert sample.done.cpu().tolist() == [True]
    assert sample.next_final_observation_row.cpu().tolist() == [True]
    assert sample.reward.cpu().tolist() == [7.0]
    assert sample.final_reward.cpu().tolist() == [70.0]
    assert int(sample.observation[0, -1, 0, 0].item()) == 3
    assert int(sample.next_observation[0, -1, 0, 0].item()) == 42
    assert int(sample.next_observation[0, -1, 0, 0].item()) != 99


def test_latest_frame_history_resident_sample_matches_full_stack_terminal_sample():
    torch = pytest.importorskip("torch")
    previous_observation = torch.zeros((1, 1, 4, 8, 8), dtype=torch.uint8)
    current_observation = torch.zeros((1, 1, 4, 8, 8), dtype=torch.uint8)
    final_observation = torch.zeros((1, 1, 4, 8, 8), dtype=torch.uint8)
    for frame_index, value in enumerate((3, 5, 7, 9)):
        previous_observation[0, 0, frame_index, 0, 0] = value
    for frame_index, value in enumerate((11, 13, 15, 17)):
        current_observation[0, 0, frame_index, 0, 0] = value
    for frame_index, value in enumerate((21, 23, 25, 27)):
        final_observation[0, 0, frame_index, 0, 0] = value

    previous_snapshot = ResidentObservationBatchV1(
        device_observation=previous_observation,
        root_device_observation=previous_observation.reshape(1, 4, 8, 8),
        generation_id=1,
        batch_size=1,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=1,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
        final_observation_row_mask=np.asarray([False], dtype=np.bool_),
    )
    current_snapshot = ResidentObservationBatchV1(
        device_observation=current_observation,
        root_device_observation=current_observation.reshape(1, 4, 8, 8),
        generation_id=2,
        batch_size=1,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=2,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
        final_device_observation=final_observation,
        root_final_device_observation=final_observation.reshape(1, 4, 8, 8),
        final_observation_row_mask=np.asarray([True], dtype=np.bool_),
    )
    index_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        done=True,
        reward=7.0,
        final_reward=70.0,
    )

    full_batch = _build_compact_resident_sample_batch_from_device_index_rows_fast(
        previous_step=_resident_step_for_terminal_sample_test(previous_snapshot, done=False),
        current_step=_resident_step_for_terminal_sample_test(current_snapshot, done=True),
        source_index_row=np.asarray([5], dtype=np.int64),
        index_rows=index_rows,
        seed=19,
        replace=False,
    )
    frame_batch = _build_compact_resident_sample_batch_from_device_index_rows_fast(
        previous_step=_resident_step_for_terminal_sample_test(
            _resident_frame_history_snapshot_from_full_test(torch, previous_snapshot),
            done=False,
        ),
        current_step=_resident_step_for_terminal_sample_test(
            _resident_frame_history_snapshot_from_full_test(torch, current_snapshot),
            done=True,
        ),
        source_index_row=np.asarray([5], dtype=np.int64),
        index_rows=index_rows,
        seed=19,
        replace=False,
    )

    assert torch.equal(frame_batch.observation, full_batch.observation)
    assert torch.equal(frame_batch.next_observation, full_batch.next_observation)
    assert torch.equal(frame_batch.done, full_batch.done)
    assert torch.equal(frame_batch.final_reward, full_batch.final_reward)
    assert frame_batch.metadata["resident_terminal_final_observation_used"] is True
    assert frame_batch.metadata["resident_current_generation_id"] == 2
    assert frame_batch.next_observation[0, :, 0, 0].cpu().tolist() == [21, 23, 25, 27]


def test_compact_replay_ring_resident_device_sample_accepts_sparse_terminal_final_observation():
    torch = pytest.importorskip("torch")
    previous_observation = torch.zeros((2, 1, 4, 8, 8), dtype=torch.uint8)
    previous_observation[1, 0, -1, 0, 0] = 3
    current_observation = torch.zeros((2, 1, 4, 8, 8), dtype=torch.uint8)
    current_observation[1, 0, -1, 0, 0] = 99
    sparse_final = current_observation.index_select(
        0,
        torch.tensor([1], dtype=torch.long),
    ).clone()
    sparse_final[0, 0, -1, 0, 0] = 42
    previous_snapshot = ResidentObservationBatchV1(
        device_observation=previous_observation,
        root_device_observation=previous_observation.reshape(2, 4, 8, 8),
        generation_id=1,
        batch_size=2,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=1,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
        final_observation_row_mask=np.asarray([False, False], dtype=np.bool_),
    )
    current_snapshot = ResidentObservationBatchV1(
        device_observation=current_observation,
        root_device_observation=current_observation.reshape(2, 4, 8, 8),
        generation_id=2,
        batch_size=2,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=2,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={
            "resident_final_device_observation_storage": "sparse_rows",
            "resident_final_device_observation_sparse_row_count": 1,
        },
        final_observation_row_mask=np.asarray([False, True], dtype=np.bool_),
        final_device_observation_rows=sparse_final,
        final_device_observation_row_indices=np.asarray([1], dtype=np.int32),
    )
    previous_step = SimpleNamespace(
        observation=np.zeros((2, 1, 4, 8, 8), dtype=np.float32),
        action_mask=np.ones((2, 1, ACTION_COUNT), dtype=np.bool_),
        reward=np.asarray([[0.0], [0.0]], dtype=np.float32),
        final_reward_map=np.asarray([[0.0], [0.0]], dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1], [1]], dtype=np.int16)},
        compact_batch=SimpleNamespace(
            observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        ),
        resident_observation_replay_snapshot=previous_snapshot,
    )
    current_step = SimpleNamespace(
        observation=np.zeros((2, 1, 4, 8, 8), dtype=np.float32),
        action_mask=np.ones((2, 1, ACTION_COUNT), dtype=np.bool_),
        reward=np.asarray([[0.0], [7.0]], dtype=np.float32),
        final_reward_map=np.asarray([[0.0], [70.0]], dtype=np.float32),
        done=np.asarray([False, True], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1], [1]], dtype=np.int16)},
        compact_batch=SimpleNamespace(
            observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        ),
        resident_observation_replay_snapshot=current_snapshot,
    )
    index_rows = _resident_device_index_rows_for_terminal_sample_test(torch)
    index_rows = replace(
        index_rows,
        policy_env_id=torch.tensor([1], dtype=torch.int64),
        env_row=torch.tensor([1], dtype=torch.int32),
        record_index=7,
        next_record_index=8,
    )
    ring = _CompactReplayRingV1(capacity=2)
    ring.append(previous_step=previous_step, current_step=current_step, index_rows=index_rows)

    result = ring.sample(seed=3, sample_batch_size=1)
    sample = result["sample_batch"]

    assert sample.metadata["resident_terminal_final_observation_used"] is True
    assert sample.done.cpu().tolist() == [True]
    assert int(sample.observation[0, -1, 0, 0].item()) == 3
    assert int(sample.next_observation[0, -1, 0, 0].item()) == 42
    assert int(sample.next_observation[0, -1, 0, 0].item()) != 99


def test_grouped_direct_write_resident_sample_matches_per_entry_concat():
    torch = pytest.importorskip("torch")

    terminal_previous_step = _resident_step_for_terminal_sample_test(
        _resident_snapshot_for_terminal_sample_test(torch, latest_value=3, generation_id=1),
        done=False,
    )
    terminal_current_step = _resident_step_for_terminal_sample_test(
        _resident_snapshot_for_terminal_sample_test(
            torch,
            latest_value=99,
            generation_id=2,
            final_latest_value=42,
        ),
        action=1,
        reward=7.0,
        final_reward=70.0,
        done=True,
    )
    live_previous_step = _resident_step_for_terminal_sample_test(
        _resident_snapshot_for_terminal_sample_test(torch, latest_value=5, generation_id=3),
        done=False,
    )
    live_current_step = _resident_step_for_terminal_sample_test(
        _resident_snapshot_for_terminal_sample_test(torch, latest_value=6, generation_id=4),
        action=2,
        reward=11.0,
        final_reward=11.0,
        done=False,
    )
    terminal_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=0,
        action=1,
        reward=7.0,
        final_reward=70.0,
        done=True,
    )
    live_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=1,
        action=2,
        root_value=0.5,
        reward=11.0,
        final_reward=11.0,
        done=False,
    )
    terminal_next_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=1,
        action=2,
        root_value=0.75,
        reward=11.0,
        final_reward=11.0,
        done=False,
    )
    live_next_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=2,
        action=0,
        root_value=0.875,
        reward=0.0,
        final_reward=0.0,
        done=False,
    )
    group_specs = [
        (
            terminal_previous_step,
            terminal_current_step,
            terminal_rows,
            terminal_next_rows,
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([10, 10], dtype=np.int64),
        ),
        (
            live_previous_step,
            live_current_step,
            live_rows,
            live_next_rows,
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([20, 20], dtype=np.int64),
        ),
    ]
    sample_position_order = np.asarray([2, 0, 1, 3], dtype=np.int64)
    metadata = {
        "sample_row_count": 4,
        "require_next_targets": True,
        "num_unroll_steps": 1,
    }

    per_entry_batches = []
    grouped_samples = []
    for previous_step, current_step, rows, next_rows, local_rows, source_rows in group_specs:
        per_entry_batches.append(
            _build_compact_resident_sample_batch_from_index_rows_fast(
                previous_step=previous_step,
                current_step=current_step,
                source_index_row=source_rows,
                index_rows=_take_compact_index_rows(rows, local_rows),
                next_index_rows=_take_compact_index_rows(next_rows, local_rows),
                seed=123,
                replace=True,
            )
        )
        grouped_samples.append(
            {
                "previous_step": previous_step,
                "current_step": current_step,
                "source_index_row": source_rows,
                "index_rows": rows,
                "local_rows": local_rows,
                "next_index_rows": next_rows,
            }
        )

    expected = _concat_compact_resident_sample_batches_fast(
        per_entry_batches,
        metadata=metadata,
        sample_position_order=sample_position_order,
    )
    actual = _build_compact_resident_grouped_device_sample_batch_fast(
        group_samples=grouped_samples,
        metadata=metadata,
        sample_position_order=sample_position_order,
        seed=123,
        replace=True,
    )

    tensor_fields = [
        "row_id",
        "observation",
        "action",
        "action_mask",
        "policy_target",
        "root_value",
        "reward",
        "final_reward",
        "done",
        "terminated",
        "truncated",
        "next_observation",
        "to_play",
        "env_row",
        "player",
        "record_index",
        "next_record_index",
        "policy_row",
        "next_action_mask",
        "next_policy_target",
        "next_root_value",
        "next_final_observation_row",
        "unroll_action",
        "unroll_reward",
        "unroll_policy_target",
        "unroll_root_value",
        "unroll_action_mask",
        "unroll_action_valid_mask",
        "unroll_reward_valid_mask",
        "unroll_policy_valid_mask",
        "unroll_value_valid_mask",
        "unroll_done",
        "unroll_terminated",
        "unroll_truncated",
    ]
    for field in tensor_fields:
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if expected_value is None:
            assert actual_value is None, field
            continue
        assert actual_value is not None, field
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    for key in (
        "terminal_sample_row_count",
        "next_final_observation_row_count",
        "resident_terminal_final_observation_used",
        "terminal_zero_metadata_fast_path_count",
        "terminal_tensor_check_count",
        "device_replay_index_rows_sample",
        "device_replay_index_rows_sample_all",
        "explicit_next_targets",
    ):
        assert actual.metadata.get(key) == expected.metadata.get(key), key
    assert actual.metadata["resident_grouped_device_direct_write_sample_batch"] is True
    assert actual.done.cpu().tolist() == [True, False, True, False]
    assert [int(value) for value in actual.next_observation[:, -1, 0, 0].cpu().tolist()] == [
        42,
        6,
        42,
        6,
    ]


def test_grouped_direct_write_resident_sample_supports_unroll_windows(monkeypatch):
    torch = pytest.importorskip("torch")

    terminal_previous_step = _resident_step_for_terminal_sample_test(
        _resident_snapshot_for_terminal_sample_test(torch, latest_value=3, generation_id=1),
        done=False,
    )
    terminal_current_step = _resident_step_for_terminal_sample_test(
        _resident_snapshot_for_terminal_sample_test(
            torch,
            latest_value=99,
            generation_id=2,
            final_latest_value=42,
        ),
        action=1,
        reward=7.0,
        final_reward=70.0,
        done=True,
    )
    live_previous_step = _resident_step_for_terminal_sample_test(
        _resident_snapshot_for_terminal_sample_test(torch, latest_value=5, generation_id=3),
        done=False,
    )
    live_current_step = _resident_step_for_terminal_sample_test(
        _resident_snapshot_for_terminal_sample_test(torch, latest_value=6, generation_id=4),
        action=2,
        reward=11.0,
        final_reward=11.0,
        done=False,
    )
    terminal_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=0,
        action=1,
        root_value=-1.0,
        reward=7.0,
        final_reward=70.0,
        done=True,
    )
    terminal_next_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=1,
        action=2,
        root_value=-2.0,
        reward=11.0,
        final_reward=11.0,
        done=False,
    )
    terminal_next2_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=2,
        action=0,
        root_value=-3.0,
        reward=0.25,
        final_reward=0.25,
        done=False,
    )
    live_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=3,
        action=2,
        root_value=0.5,
        reward=11.0,
        final_reward=11.0,
        done=False,
    )
    live_next_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=4,
        action=0,
        root_value=0.75,
        reward=0.25,
        final_reward=0.25,
        done=False,
    )
    live_next2_rows = _resident_device_index_rows_for_terminal_sample_test(
        torch,
        record_index=5,
        action=1,
        root_value=0.875,
        reward=0.5,
        final_reward=0.5,
        done=False,
    )
    group_specs = [
        (
            terminal_previous_step,
            terminal_current_step,
            terminal_rows,
            (terminal_next_rows, terminal_next2_rows),
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([10, 10], dtype=np.int64),
        ),
        (
            live_previous_step,
            live_current_step,
            live_rows,
            (live_next_rows, live_next2_rows),
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([20, 20], dtype=np.int64),
        ),
    ]
    sample_position_order = np.asarray([2, 0, 1, 3], dtype=np.int64)
    metadata = {
        "sample_row_count": 4,
        "require_next_targets": True,
        "num_unroll_steps": 2,
    }

    per_entry_batches = []
    grouped_samples = []
    for previous_step, current_step, rows, window, local_rows, source_rows in group_specs:
        materialized_rows = _take_compact_index_rows(rows, local_rows)
        materialized_window = tuple(_take_compact_index_rows(item, local_rows) for item in window)
        per_entry_batches.append(
            _build_compact_resident_sample_batch_from_index_rows_fast(
                previous_step=previous_step,
                current_step=current_step,
                source_index_row=source_rows,
                index_rows=materialized_rows,
                next_index_rows=materialized_window[0],
                unroll_index_rows=(materialized_rows, *materialized_window),
                seed=123,
                replace=True,
            )
        )
        grouped_samples.append(
            {
                "previous_step": previous_step,
                "current_step": current_step,
                "source_index_row": source_rows,
                "index_rows": rows,
                "local_rows": local_rows,
                "next_index_rows": window[0],
                "unroll_index_rows_window": window,
            }
        )

    expected = _concat_compact_resident_sample_batches_fast(
        per_entry_batches,
        metadata=metadata,
        sample_position_order=sample_position_order,
    )
    actual = _build_compact_resident_grouped_device_sample_batch_fast(
        group_samples=grouped_samples,
        metadata=metadata,
        sample_position_order=sample_position_order,
        seed=123,
        replace=True,
    )

    for field in (
        "row_id",
        "observation",
        "next_observation",
        "action",
        "action_mask",
        "policy_target",
        "root_value",
        "reward",
        "final_reward",
        "done",
        "terminated",
        "truncated",
        "next_action_mask",
        "next_policy_target",
        "next_root_value",
        "unroll_action",
        "unroll_reward",
        "unroll_policy_target",
        "unroll_root_value",
        "unroll_action_mask",
        "unroll_action_valid_mask",
        "unroll_reward_valid_mask",
        "unroll_policy_valid_mask",
        "unroll_value_valid_mask",
        "unroll_done",
        "unroll_terminated",
        "unroll_truncated",
    ):
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        assert actual_value is not None, field
        torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)

    assert actual.metadata["resident_grouped_device_direct_write_sample_batch"] is True
    assert actual.metadata["explicit_unroll_targets"] is True
    assert actual.metadata["terminal_unroll_value_target_row_count"] == 2
    assert actual.metadata["terminal_unroll_value_target_mode"] == (
        "stock_terminal_no_bootstrap_return_discount_1.0"
    )
    expected_learner = build_compact_muzero_learner_batch_v1(
        actual,
        config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=2),
    )

    def fail_final_observation_materialization(*_args, **_kwargs):
        raise AssertionError("grouped learner should validate final observations only")

    monkeypatch.setattr(
        hybrid_profile,
        "_resident_final_next_observation_for_rows",
        fail_final_observation_materialization,
    )
    direct_learner = _build_compact_resident_grouped_device_learner_batch_fast(
        group_samples=grouped_samples,
        metadata=metadata,
        sample_position_order=sample_position_order,
    )
    for field in (
        "observation",
        "action",
        "action_mask",
        "target_reward",
        "target_value",
        "target_policy",
        "target_reward_mask",
        "target_value_mask",
        "target_policy_mask",
        "action_valid_mask",
        "weights",
        "done",
        "terminated",
        "truncated",
    ):
        torch.testing.assert_close(
            getattr(direct_learner, field),
            getattr(expected_learner, field),
            rtol=0,
            atol=0,
        )
    assert direct_learner.source_sample_batch is None
    assert direct_learner.metadata["resident_grouped_device_direct_write_learner_batch"] is True
    assert direct_learner.metadata["compact_muzero_learner_batch_prevalidated"] is True
    assert (
        direct_learner.metadata[
            "compact_muzero_learner_batch_builder_cuda_sync_timing_diagnostics"
        ]
        is False
    )
    assert (
        direct_learner.metadata[
            "compact_muzero_learner_batch_builder_cuda_sync_timing_enabled"
        ]
        is False
    )
    assert direct_learner.metadata["compact_muzero_learner_batch_builder_cuda_sync_count"] == 0
    assert direct_learner.metadata["compact_muzero_learner_batch_builder_cuda_sync_sec"] == 0.0
    assert (
        direct_learner.metadata[
            "compact_muzero_learner_batch_unroll2_specialized_builder_requested"
        ]
        is False
    )
    assert (
        direct_learner.metadata["compact_muzero_learner_batch_unroll2_specialized_builder_used"]
        is False
    )
    assert (
        direct_learner.metadata[
            "compact_muzero_learner_batch_unroll2_specialized_builder_call_count"
        ]
        == 0
    )
    assert (
        direct_learner.metadata[
            "compact_muzero_learner_batch_unroll2_specialized_builder_fallback_count"
        ]
        == 0
    )
    assert direct_learner.metadata["compact_muzero_learner_batch_unroll_builder_path"] == "generic"
    assert (
        direct_learner.metadata["compact_muzero_learner_batch_late_observation_normalize"]
        is True
    )
    assert (
        direct_learner.metadata["compact_muzero_learner_batch_observation_storage_dtype"]
        == "torch.uint8"
    )
    assert direct_learner.observation.dtype == torch.float32
    assert direct_learner.metadata["compact_muzero_learner_batch_grouped_order_restore_once"] is True
    assert direct_learner.metadata["compact_muzero_learner_batch_sample_order"] == "rng"
    assert direct_learner.metadata["compact_muzero_learner_batch_preserves_sample_order"] is True
    assert (
        direct_learner.metadata["compact_muzero_learner_batch_order_restore_index_copy_count"] > 0
    )
    assert direct_learner.metadata["terminal_unroll_value_target_row_count"] == 2
    assert direct_learner.metadata["terminal_final_observation_group_count"] == 1
    assert direct_learner.metadata["terminal_final_observation_fallback_count"] == 1
    assert direct_learner.metadata["terminal_final_observation_validate_only_count"] == 1
    assert direct_learner.metadata["terminal_final_observation_materialized_count"] == 0
    assert (
        direct_learner.metadata[
            "compact_muzero_learner_batch_builder_terminal_metadata_final_observation_gather_sec"
        ]
        == 0.0
    )
    successor_by_record_index = {
        1: terminal_next_rows,
        2: terminal_next2_rows,
        4: live_next_rows,
        5: live_next2_rows,
    }
    cached_grouped_samples = []
    for group in grouped_samples:
        entry = hybrid_profile._CompactReplayRingEntry(
            previous_step=group["previous_step"],
            current_step=group["current_step"],
            index_rows=group["index_rows"],
        )
        successor_window = hybrid_profile._successor_index_row_window_for_entry(
            entry,
            successor_by_record_index=successor_by_record_index,
            num_unroll_steps=2,
            allow_terminal_padding=True,
        )
        assert successor_window is not None
        targets = hybrid_profile._compact_replay_learner_ready_unroll2_targets_for_entry(
            entry,
            successor_by_record_index=successor_by_record_index,
        )
        assert targets is not None
        cached_grouped_samples.append(
            {
                **group,
                "next_index_rows": successor_window[0],
                "unroll_index_rows_window": successor_window,
                "learner_ready_unroll2_targets": targets,
            }
        )
    cache_direct_learner = _build_compact_resident_grouped_device_learner_batch_fast(
        group_samples=cached_grouped_samples,
        metadata=metadata,
        sample_position_order=sample_position_order,
    )
    cached_learner = _build_compact_resident_grouped_device_learner_batch_fast(
        group_samples=cached_grouped_samples,
        metadata={
            **metadata,
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
        },
        sample_position_order=sample_position_order,
    )
    for field in (
        "observation",
        "action",
        "action_mask",
        "target_reward",
        "target_value",
        "target_policy",
        "target_reward_mask",
        "target_value_mask",
        "target_policy_mask",
        "action_valid_mask",
        "weights",
        "done",
        "terminated",
        "truncated",
    ):
        torch.testing.assert_close(
            getattr(cached_learner, field),
            getattr(cache_direct_learner, field),
            rtol=0,
            atol=0,
            msg=field,
        )
    assert (
        cached_learner.metadata[
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_requested"
        ]
        is True
    )
    assert (
        cached_learner.metadata[
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_used"
        ]
        is True
    )
    assert (
        cached_learner.metadata[
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_call_count"
        ]
        == len(cached_grouped_samples)
    )
    assert (
        cached_learner.metadata[
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_fallback_count"
        ]
        == 0
    )
    assert (
        cached_learner.metadata["compact_muzero_learner_batch_unroll_builder_path"]
        == "learner_ready_unroll2_cache"
    )
    assert (
        cached_learner.metadata["terminal_unroll_value_target_row_count"]
        == cache_direct_learner.metadata["terminal_unroll_value_target_row_count"]
    )
    specialized_learner = _build_compact_resident_grouped_device_learner_batch_fast(
        group_samples=grouped_samples,
        metadata={
            **metadata,
            "compact_muzero_learner_batch_unroll2_specialized_builder": True,
        },
        sample_position_order=sample_position_order,
    )
    for field in (
        "observation",
        "action",
        "action_mask",
        "target_reward",
        "target_value",
        "target_policy",
        "target_reward_mask",
        "target_value_mask",
        "target_policy_mask",
        "action_valid_mask",
        "weights",
        "done",
        "terminated",
        "truncated",
    ):
        torch.testing.assert_close(
            getattr(specialized_learner, field),
            getattr(direct_learner, field),
            rtol=0,
            atol=0,
        )
    assert (
        specialized_learner.metadata[
            "compact_muzero_learner_batch_unroll2_specialized_builder_requested"
        ]
        is True
    )
    assert (
        specialized_learner.metadata[
            "compact_muzero_learner_batch_unroll2_specialized_builder_used"
        ]
        is True
    )
    assert (
        specialized_learner.metadata[
            "compact_muzero_learner_batch_unroll2_specialized_builder_call_count"
        ]
        == len(grouped_samples)
    )
    assert (
        specialized_learner.metadata[
            "compact_muzero_learner_batch_unroll2_specialized_builder_fallback_count"
        ]
        == 0
    )
    assert (
        specialized_learner.metadata["compact_muzero_learner_batch_unroll_builder_path"]
        == "unroll2_specialized"
    )
    assert (
        specialized_learner.metadata["terminal_unroll_value_target_row_count"]
        == direct_learner.metadata["terminal_unroll_value_target_row_count"]
    )
    assert (
        specialized_learner.metadata["terminal_unroll_value_target_mode"]
        == direct_learner.metadata["terminal_unroll_value_target_mode"]
    )
    assert (
        specialized_learner.metadata["compact_muzero_learner_batch_sample_row_checksum"]
        == direct_learner.metadata["compact_muzero_learner_batch_sample_row_checksum"]
    )
    grouped_order_learner = _build_compact_resident_grouped_device_learner_batch_fast(
        group_samples=grouped_samples,
        metadata=metadata,
        sample_position_order=sample_position_order,
        preserve_sample_order=False,
    )
    assert grouped_order_learner.metadata["compact_muzero_learner_batch_sample_order"] == "grouped"
    assert (
        grouped_order_learner.metadata["compact_muzero_learner_batch_preserves_sample_order"]
        is False
    )
    assert (
        grouped_order_learner.metadata["compact_muzero_learner_batch_grouped_order_restore_once"]
        is False
    )
    assert (
        grouped_order_learner.metadata[
            "compact_muzero_learner_batch_order_restore_index_copy_count"
        ]
        == 0
    )
    assert grouped_order_learner.metadata["compact_muzero_learner_batch_sample_row_checksum"] == (
        10 * 1 + 10 * 2 + 20 * 3 + 20 * 4
    )
    assert grouped_order_learner.observation.shape == expected_learner.observation.shape
    assert grouped_order_learner.action.shape == expected_learner.action.shape


def test_compact_replay_ring_resident_terminal_masks_feed_compact_muzero_batch():
    torch = pytest.importorskip("torch")
    previous_snapshot = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=3,
        generation_id=1,
    )
    terminal_snapshot = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=99,
        generation_id=2,
        final_latest_value=42,
    )
    successor_snapshot = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=5,
        generation_id=3,
    )
    previous_step = _resident_step_for_terminal_sample_test(previous_snapshot, done=False)
    terminal_step = _resident_step_for_terminal_sample_test(terminal_snapshot)
    successor_step = _resident_step_for_terminal_sample_test(
        successor_snapshot,
        action=2,
        reward=0.0,
        final_reward=0.0,
        done=False,
    )
    ring = _CompactReplayRingV1(capacity=3)
    ring.append(
        previous_step=previous_step,
        current_step=terminal_step,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=0,
            action=1,
            done=True,
        ),
    )
    ring.append(
        previous_step=terminal_step,
        current_step=successor_step,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=1,
            action=2,
            reward=0.0,
            final_reward=0.0,
            done=False,
        ),
    )

    result = ring.sample(seed=0, sample_batch_size=1, require_next_targets=True)
    sample = result["sample_batch"]
    batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=CompactMuZeroLearnerConfigV1(
            device="cpu",
            require_device_replay_rows=True,
        ),
    )

    assert sample.unroll_action_valid_mask.cpu().tolist() == [[True]]
    assert sample.unroll_reward_valid_mask.cpu().tolist() == [[True]]
    assert sample.unroll_policy_valid_mask.cpu().tolist() == [[True, False]]
    assert sample.unroll_value_valid_mask.cpu().tolist() == [[True, False]]
    assert bool(sample.next_action_mask[0].any().item()) is False
    assert float(sample.next_policy_target[0].sum().item()) == 0.0
    assert float(sample.next_root_value[0].item()) == 0.0
    assert batch.metadata["compact_muzero_learner_done_count"] == 1
    assert batch.metadata["compact_muzero_learner_action_valid_count"] == 1
    assert batch.metadata["compact_muzero_learner_reward_valid_count"] == 1
    assert batch.metadata["compact_muzero_learner_policy_valid_count"] == 1
    assert batch.metadata["compact_muzero_learner_value_valid_count"] == 1
    assert bool(batch.target_policy_mask[0, 0].item()) is True
    assert bool(batch.target_policy_mask[0, 1].item()) is False
    assert bool(batch.target_value_mask[0, 1].item()) is False
    assert bool(batch.next_action_mask[0].any().item()) is False


def test_compact_replay_ring_resident_terminal_n_step_masks_feed_compact_muzero_batch():
    torch = pytest.importorskip("torch")
    snapshot0 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=3,
        generation_id=1,
    )
    snapshot1 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=99,
        generation_id=2,
        final_latest_value=42,
    )
    snapshot2 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=5,
        generation_id=3,
    )
    snapshot3 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=6,
        generation_id=4,
    )
    step0 = _resident_step_for_terminal_sample_test(snapshot0, done=False)
    step1_terminal = _resident_step_for_terminal_sample_test(snapshot1)
    step2_reset = _resident_step_for_terminal_sample_test(
        snapshot2,
        action=2,
        reward=0.5,
        final_reward=0.5,
        done=False,
    )
    step3_reset = _resident_step_for_terminal_sample_test(
        snapshot3,
        action=0,
        reward=0.25,
        final_reward=0.25,
        done=False,
    )
    ring = _CompactReplayRingV1(capacity=4)
    ring.append(
        previous_step=step0,
        current_step=step1_terminal,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=0,
            action=1,
            done=True,
        ),
    )
    ring.append(
        previous_step=step1_terminal,
        current_step=step2_reset,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=1,
            action=2,
            reward=0.5,
            final_reward=0.5,
            done=False,
        ),
    )
    ring.append(
        previous_step=step2_reset,
        current_step=step3_reset,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=2,
            action=0,
            reward=0.25,
            final_reward=0.25,
            done=False,
        ),
    )

    result = ring.sample(
        seed=0,
        sample_batch_size=1,
        require_next_targets=True,
        num_unroll_steps=2,
    )
    sample = result["sample_batch"]
    batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=CompactMuZeroLearnerConfigV1(
            device="cpu",
            num_unroll_steps=2,
            require_device_replay_rows=True,
        ),
    )

    assert sample.metadata["explicit_unroll_targets"] is True
    assert sample.unroll_action_valid_mask.cpu().tolist() == [[True, False]]
    assert sample.unroll_reward_valid_mask.cpu().tolist() == [[True, False]]
    assert sample.unroll_policy_valid_mask.cpu().tolist() == [[True, False, False]]
    assert sample.unroll_value_valid_mask.cpu().tolist() == [[True, False, False]]
    assert sample.unroll_done.cpu().tolist() == [[True, False]]
    assert float(sample.unroll_reward[0, 1].item()) == 0.0
    assert float(sample.unroll_policy_target[0, 1:].sum().item()) == 0.0
    assert float(sample.unroll_root_value[0, 1:].abs().sum().item()) == 0.0
    assert bool(sample.unroll_action_mask[0, 1:].any().item()) is False
    assert batch.metadata["compact_muzero_learner_done_count"] == 1
    assert batch.metadata["compact_muzero_learner_action_valid_count"] == 1
    assert batch.metadata["compact_muzero_learner_reward_valid_count"] == 1
    assert batch.metadata["compact_muzero_learner_policy_valid_count"] == 1
    assert batch.metadata["compact_muzero_learner_value_valid_count"] == 1

    fused_result = ring.sample(
        seed=0,
        sample_batch_size=1,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    fused_batch = fused_result["learner_batch"]
    assert fused_result["sample_batch"] is None
    assert (
        fused_result["telemetry"][
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"
        ]
        is True
    )
    assert (
        fused_result["telemetry"][
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order"
        ]
        == "rng"
    )
    assert (
        fused_result["telemetry"][
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order"
        ]
        is True
    )
    assert (
        fused_result["telemetry"][
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count"
        ]
        > 0
    )
    for field in (
        "observation",
        "action",
        "action_mask",
        "target_reward",
        "target_value",
        "target_policy",
        "target_reward_mask",
        "target_value_mask",
        "target_policy_mask",
        "action_valid_mask",
        "done",
        "terminated",
        "truncated",
    ):
        torch.testing.assert_close(getattr(fused_batch, field), getattr(batch, field))
    assert fused_batch.metadata["resident_grouped_device_direct_write_learner_batch"] is True
    assert fused_batch.metadata["terminal_unroll_value_target_row_count"] == 1


def test_compact_replay_ring_resident_terminal_two_step_targets_match_stock_semantics():
    torch = pytest.importorskip("torch")
    snapshot0 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=3,
        generation_id=1,
    )
    snapshot1 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=4,
        generation_id=2,
    )
    snapshot2_terminal = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=99,
        generation_id=3,
        final_latest_value=42,
    )
    snapshot3_reset = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=6,
        generation_id=4,
    )
    step0 = _resident_step_for_terminal_sample_test(snapshot0, done=False)
    step1 = _resident_step_for_terminal_sample_test(
        snapshot1,
        action=1,
        reward=2.0,
        final_reward=2.0,
        done=False,
    )
    step2_terminal = _resident_step_for_terminal_sample_test(
        snapshot2_terminal,
        action=2,
        reward=5.0,
        final_reward=50.0,
        done=True,
    )
    step3_reset = _resident_step_for_terminal_sample_test(
        snapshot3_reset,
        action=0,
        reward=0.25,
        final_reward=0.25,
        done=False,
    )

    ring = _CompactReplayRingV1(capacity=4)
    ring.append(
        previous_step=step0,
        current_step=step1,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=0,
            action=1,
            policy_target=np.asarray([0.2, 0.7, 0.1], dtype=np.float32),
            root_value=-11.0,
            reward=2.0,
            final_reward=2.0,
            done=False,
        ),
    )
    ring.append(
        previous_step=step1,
        current_step=step2_terminal,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=1,
            action=2,
            policy_target=np.asarray([0.1, 0.1, 0.8], dtype=np.float32),
            root_value=-22.0,
            reward=5.0,
            final_reward=50.0,
            done=True,
        ),
    )
    ring.append(
        previous_step=step2_terminal,
        current_step=step3_reset,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=2,
            action=0,
            policy_target=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            root_value=-33.0,
            reward=0.25,
            final_reward=0.25,
            done=False,
        ),
    )

    state = ring.snapshot_durable_state(
        policy_version_ref="resident-terminal-policy-v1",
        policy_source="unit_test_resident_terminal",
        model_version_ref="resident-terminal-model-v1",
        metadata={"support_scale": 9},
    )
    loaded_state = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    restored = _CompactReplayRingV1.from_durable_state(loaded_state)

    result = restored.sample(
        seed=11,
        sample_batch_size=1,
        require_next_targets=True,
        num_unroll_steps=2,
    )
    sample = result["sample_batch"]
    batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=CompactMuZeroLearnerConfigV1(
            device="cpu",
            num_unroll_steps=2,
            require_device_replay_rows=True,
        ),
    )

    expected_policy = torch.tensor(
        [[[0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    expected_value = torch.tensor([[7.0, 5.0, 0.0]], dtype=torch.float32)
    expected_reward = torch.tensor([[2.0, 5.0]], dtype=torch.float32)
    expected_state_valid = torch.tensor([[True, True, False]], dtype=torch.bool)

    assert restored.entry_count == 3
    assert sample.metadata["compact_replay_store_loaded"] is True
    assert sample.metadata["policy_version_ref"] == "resident-terminal-policy-v1"
    assert sample.metadata["model_version_ref"] == "resident-terminal-model-v1"
    assert sample.metadata["profile_only"] is True
    assert sample.metadata["calls_train_muzero"] is False
    assert sample.metadata["training_speed_claim"] is False
    assert sample.metadata["terminal_unroll_value_target_row_count"] == 1
    assert sample.metadata["terminal_unroll_value_target_mode"] == (
        "stock_terminal_no_bootstrap_return_discount_1.0"
    )
    assert (
        result["telemetry"][
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count"
        ]
        == 1
    )
    assert (
        result["telemetry"]["compact_rollout_slab_sample_gate_terminal_unroll_windows_supported"]
        is True
    )
    assert sample.unroll_done.cpu().tolist() == [[False, True]]
    assert sample.unroll_action_valid_mask.cpu().tolist() == [[True, True]]
    assert sample.unroll_reward_valid_mask.cpu().tolist() == [[True, True]]
    assert torch.equal(sample.unroll_policy_valid_mask.cpu(), expected_state_valid)
    assert torch.equal(sample.unroll_value_valid_mask.cpu(), expected_state_valid)
    assert not bool(sample.unroll_action_mask[0, 2].any().item())
    assert float(sample.root_value[0].item()) == -11.0
    assert torch.allclose(sample.unroll_policy_target.cpu(), expected_policy, atol=1e-6)
    assert torch.allclose(sample.unroll_root_value.cpu(), expected_value, atol=1e-6)
    assert torch.allclose(batch.target_reward.cpu(), expected_reward, atol=1e-6)
    assert torch.allclose(batch.target_policy.cpu(), expected_policy, atol=1e-6)
    assert torch.allclose(batch.target_value.cpu(), expected_value, atol=1e-6)


def test_compact_replay_ring_resident_terminal_two_step_targets_pad_without_reset_successor():
    torch = pytest.importorskip("torch")
    snapshot0 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=3,
        generation_id=1,
    )
    snapshot1 = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=4,
        generation_id=2,
    )
    snapshot2_terminal = _resident_snapshot_for_terminal_sample_test(
        torch,
        latest_value=99,
        generation_id=3,
        final_latest_value=42,
    )
    step0 = _resident_step_for_terminal_sample_test(snapshot0, done=False)
    step1 = _resident_step_for_terminal_sample_test(
        snapshot1,
        action=1,
        reward=2.0,
        final_reward=2.0,
        done=False,
    )
    step2_terminal = _resident_step_for_terminal_sample_test(
        snapshot2_terminal,
        action=2,
        reward=5.0,
        final_reward=50.0,
        done=True,
    )

    ring = _CompactReplayRingV1(capacity=4)
    ring.append(
        previous_step=step0,
        current_step=step1,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=0,
            action=1,
            policy_target=np.asarray([0.2, 0.7, 0.1], dtype=np.float32),
            root_value=-11.0,
            reward=2.0,
            final_reward=2.0,
            done=False,
        ),
    )
    ring.append(
        previous_step=step1,
        current_step=step2_terminal,
        index_rows=_resident_device_index_rows_for_terminal_sample_test(
            torch,
            record_index=1,
            action=2,
            policy_target=np.asarray([0.1, 0.1, 0.8], dtype=np.float32),
            root_value=-22.0,
            reward=5.0,
            final_reward=50.0,
            done=True,
        ),
    )

    window_result = ring.sample(
        seed=1,
        sample_batch_size=1,
        require_next_targets=True,
        num_unroll_steps=2,
    )
    window_sample = window_result["sample_batch"]
    window_telemetry = window_result["telemetry"]

    assert window_sample.record_index.cpu().tolist() == [0]
    assert window_telemetry["compact_rollout_slab_sample_gate_terminal_row_available_count"] == 1
    assert (
        window_telemetry["compact_rollout_slab_sample_gate_terminal_window_row_available_count"]
        == 2
    )
    assert window_telemetry["compact_rollout_slab_sample_gate_terminal_row_forced"] is False
    assert window_telemetry["compact_rollout_slab_sample_gate_terminal_sample_row_count"] == 0
    assert (
        window_telemetry["compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count"]
        == 1
    )

    result = ring.sample(
        seed=0,
        sample_batch_size=0,
        require_next_targets=True,
        num_unroll_steps=2,
    )
    sample = result["sample_batch"]
    batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=CompactMuZeroLearnerConfigV1(
            device="cpu",
            num_unroll_steps=2,
            require_device_replay_rows=True,
        ),
    )
    order = torch.argsort(sample.record_index.cpu())
    value_by_record = sample.unroll_root_value.cpu()[order].tolist()
    valid_by_record = batch.target_value_mask.cpu()[order].tolist()

    assert result["telemetry"]["compact_rollout_slab_sample_gate_target_row_count"] == 2
    assert result["telemetry"]["compact_rollout_slab_sample_gate_explicit_unroll_targets"] is True
    assert (
        result["telemetry"]["compact_rollout_slab_sample_gate_terminal_unroll_windows_supported"]
        is True
    )
    assert sample.record_index.cpu()[order].tolist() == [0, 1]
    assert value_by_record == [[7.0, 5.0, 0.0], [5.0, 0.0, 0.0]]
    assert valid_by_record == [[True, True, False], [True, False, False]]


def test_compact_replay_ring_samples_partial_terminal_entry_without_successor():
    torch = pytest.importorskip("torch")

    previous_observation = torch.zeros((2, 1, 4, 8, 8), dtype=torch.uint8)
    current_observation = torch.ones((2, 1, 4, 8, 8), dtype=torch.uint8)
    final_observation = current_observation.clone()
    final_observation[1, 0, -1, 0, 0] = 42
    previous_snapshot = ResidentObservationBatchV1(
        device_observation=previous_observation,
        root_device_observation=previous_observation.reshape(2, 4, 8, 8),
        generation_id=1,
        batch_size=2,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=1,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
    )
    current_snapshot = ResidentObservationBatchV1(
        device_observation=current_observation,
        root_device_observation=current_observation.reshape(2, 4, 8, 8),
        generation_id=2,
        batch_size=2,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=2,
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={},
        final_device_observation=final_observation,
        root_final_device_observation=final_observation.reshape(2, 4, 8, 8),
        final_observation_row_mask=np.asarray([False, True], dtype=np.bool_),
    )
    previous_step = SimpleNamespace(
        observation=np.zeros((2, 1, 4, 8, 8), dtype=np.float32),
        action_mask=np.ones((2, 1, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((2, 1), dtype=np.float32),
        final_reward_map=np.zeros((2, 1), dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1], [2]], dtype=np.int16)},
        compact_batch=SimpleNamespace(
            observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        ),
        resident_observation_replay_snapshot=previous_snapshot,
    )
    current_step = SimpleNamespace(
        observation=np.zeros((2, 1, 4, 8, 8), dtype=np.float32),
        action_mask=np.ones((2, 1, ACTION_COUNT), dtype=np.bool_),
        reward=np.asarray([[0.0], [5.0]], dtype=np.float32),
        final_reward_map=np.asarray([[0.0], [50.0]], dtype=np.float32),
        done=np.asarray([False, True], dtype=np.bool_),
        payload={"joint_action": np.asarray([[1], [2]], dtype=np.int16)},
        compact_batch=SimpleNamespace(
            observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        ),
        resident_observation_replay_snapshot=current_snapshot,
    )
    policy = torch.zeros((2, ACTION_COUNT), dtype=torch.float32)
    policy[0, 1] = 1.0
    policy[1, 2] = 1.0
    tensor_native_metadata = {
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
        "compact_muzero_learner_batch_tensor_native_replay": True,
        "compact_muzero_learner_batch_tensor_native_replay_selected_maintained_gather": True,
    }
    partial_terminal_index_rows = CompactDeviceReplayIndexRowsV1(
        metadata={
            "device_replay_index_rows": True,
            "done_row_count": 1,
            "next_final_observation_row_count": 1,
            "host_search_payload_fallback_allowed": False,
        },
        record_index=10,
        next_record_index=11,
        compact_root_row=torch.tensor([0, 1], dtype=torch.int32),
        policy_env_id=torch.tensor([0, 1], dtype=torch.int64),
        policy_row=torch.tensor([0, 1], dtype=torch.int32),
        env_row=torch.tensor([0, 1], dtype=torch.int32),
        player=torch.tensor([0, 0], dtype=torch.int16),
        action=torch.tensor([1, 2], dtype=torch.int16),
        action_mask=torch.ones((2, ACTION_COUNT), dtype=torch.bool),
        policy_target=policy,
        root_value=torch.tensor([0.0, -2.0], dtype=torch.float32),
        reward=torch.tensor([0.0, 5.0], dtype=torch.float32),
        final_reward=torch.tensor([0.0, 50.0], dtype=torch.float32),
        done=torch.tensor([False, True], dtype=torch.bool),
        terminated=torch.tensor([False, True], dtype=torch.bool),
        truncated=torch.zeros((2,), dtype=torch.bool),
        next_final_observation_row=torch.tensor([False, True], dtype=torch.bool),
        to_play=torch.tensor([-1, -1], dtype=torch.int64),
        policy_source="unit_test_partial_terminal",
    )
    ring = _CompactReplayRingV1(
        capacity=2,
        metadata=tensor_native_metadata,
    )
    ring.append(
        previous_step=previous_step,
        current_step=current_step,
        index_rows=partial_terminal_index_rows,
    )

    result = ring.sample(
        seed=1,
        sample_batch_size=1,
        require_next_targets=True,
        num_unroll_steps=2,
    )
    sample = result["sample_batch"]
    telemetry = result["telemetry"]

    assert sample.env_row.cpu().tolist() == [1]
    assert sample.done.cpu().tolist() == [True]
    assert int(sample.next_observation[0, -1, 0, 0].item()) == 42
    assert telemetry["compact_rollout_slab_sample_gate_terminal_row_available_count"] == 1
    assert telemetry["compact_rollout_slab_sample_gate_terminal_sample_row_count"] == 1
    assert telemetry["compact_rollout_slab_sample_gate_next_final_observation_row_count"] == 1
    assert telemetry["compact_rollout_slab_sample_gate_terminal_zero_metadata_fast_path_count"] == 0
    assert telemetry["compact_rollout_slab_sample_gate_terminal_tensor_check_count"] > 0
    assert telemetry["compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count"] == 1
    assert (
        telemetry["compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode"]
        == "stock_terminal_no_bootstrap_return_discount_1.0"
    )

    fused_result = ring.sample(
        seed=1,
        sample_batch_size=1,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    fused_batch = fused_result["learner_batch"]
    fused_telemetry = fused_result["telemetry"]

    assert fused_result["sample_batch"] is None
    assert fused_batch.metadata["compact_muzero_learner_batch_prevalidation_source"] == (
        "tensor_native_replay_selected_maintained_record_table_v1"
    )
    assert fused_batch.metadata["compact_muzero_learner_batch_tensor_native_replay_used"] is True
    assert (
        fused_batch.metadata["compact_muzero_learner_batch_tensor_native_replay_impl"]
        == "selected_maintained_record_table_gather_v1"
    )
    assert (
        fused_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_source"
        ]
        == "selected_maintained_record_table_v1"
    )
    assert (
        fused_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_build_impl"
        ]
        == "direct_record_table_v1"
    )
    assert (
        fused_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_reused_record_count"
        ]
        == 1
    )
    assert (
        fused_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_missing_record_count"
        ]
        == 0
    )
    assert fused_batch.metadata["compact_muzero_learner_batch_tensor_native_replay_table_rows"] == 1
    assert (
        fused_batch.metadata[
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_fallback_count"
        ]
        == 0
    )
    assert fused_batch.done.cpu().tolist() == [[True, False]]
    assert fused_batch.target_value_mask.cpu().tolist() == [[True, False, False]]
    assert fused_batch.metadata["terminal_sample_row_count"] == 1
    assert fused_batch.metadata["next_final_observation_row_count"] == 1
    assert fused_batch.metadata["resident_terminal_final_observation_used"] is True
    assert fused_batch.metadata["terminal_unroll_value_target_row_count"] == 1
    assert (
        fused_batch.metadata["terminal_unroll_value_target_mode"]
        == "stock_terminal_no_bootstrap_return_discount_1.0"
    )
    assert (
        fused_telemetry[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
        ]
        is True
    )
    assert (
        fused_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested"
        ]
        is True
    )
    assert (
        fused_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used"
        ]
        is True
    )
    assert (
        fused_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count"
        ]
        == 0
    )
    assert (
        fused_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested"
        ]
        is True
    )
    assert (
        fused_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used"
        ]
        is True
    )
    assert (
        fused_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count"
        ]
        == 1
    )
    assert (
        fused_telemetry["compact_rollout_slab_sample_gate_candidate_universe_source"]
        == "maintained_sample_universe_v1"
    )
    assert (
        fused_telemetry["compact_rollout_slab_sample_gate_candidate_universe_cache_hit"]
        is True
    )
    assert (
        fused_telemetry[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows"
        ]
        == 1
    )
    assert (
        fused_telemetry["compact_rollout_slab_sample_gate_terminal_sample_row_count"]
        == 1
    )
    assert (
        fused_telemetry[
            "compact_rollout_slab_sample_gate_next_final_observation_row_count"
        ]
        == 1
    )

    columnar_ring = _CompactReplayRingV1(
        capacity=2,
        metadata=tensor_native_metadata,
    )
    assert (
        columnar_ring.append_columnar_entries(
            (
                CompactReplayColumnarAppendRecordV1(
                    previous_resident_observation_replay_snapshot=previous_snapshot,
                    current_resident_observation_replay_snapshot=current_snapshot,
                    index_rows=partial_terminal_index_rows,
                ),
            )
        )
        == 1
    )
    columnar_append_telemetry = columnar_ring.columnar_append_telemetry_snapshot()
    assert columnar_append_telemetry["record_count"] == 1.0
    assert columnar_append_telemetry["entry_view_object_count"] == 1.0
    assert columnar_append_telemetry["step_view_object_count"] == 2.0
    assert columnar_append_telemetry["total_sec"] >= 0.0
    assert columnar_append_telemetry["cache_refresh_sec"] >= 0.0
    columnar_result = columnar_ring.sample(
        seed=1,
        sample_batch_size=1,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    columnar_batch = columnar_result["learner_batch"]
    columnar_telemetry = columnar_result["telemetry"]
    assert columnar_batch.metadata[
        "compact_muzero_learner_batch_tensor_native_replay_table_build_impl"
    ] == "direct_record_table_v1"
    assert columnar_batch.metadata["terminal_sample_row_count"] == 1
    assert columnar_batch.metadata["resident_terminal_final_observation_used"] is True
    assert (
        columnar_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used"
        ]
        is True
    )
    assert (
        columnar_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count"
        ]
        == 0
    )

    lazy_columnar_metadata = {
        **tensor_native_metadata,
        "compact_muzero_learner_batch_tensor_native_replay_lazy_table_build": True,
    }
    lazy_columnar_ring = _CompactReplayRingV1(
        capacity=2,
        metadata=lazy_columnar_metadata,
    )
    assert (
        lazy_columnar_ring.append_columnar_entries(
            (
                CompactReplayColumnarAppendRecordV1(
                    previous_resident_observation_replay_snapshot=previous_snapshot,
                    current_resident_observation_replay_snapshot=current_snapshot,
                    index_rows=partial_terminal_index_rows,
                ),
            )
        )
        == 1
    )
    lazy_columnar_result = lazy_columnar_ring.sample(
        seed=1,
        sample_batch_size=1,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    lazy_columnar_batch = lazy_columnar_result["learner_batch"]
    lazy_columnar_telemetry = lazy_columnar_result["telemetry"]
    assert (
        lazy_columnar_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_impl"
        ]
        == "selected_direct_record_table_gather_v1"
    )
    assert (
        lazy_columnar_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_source"
        ]
        == "selected_direct_record_table_v1"
    )
    assert (
        lazy_columnar_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_rows"
        ]
        == 1
    )
    assert (
        lazy_columnar_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested"
        ]
        is False
    )
    assert (
        lazy_columnar_batch.metadata[
            "compact_rollout_slab_sample_gate_tensor_native_direct_selected_path_used"
        ]
        is True
    )
    assert lazy_columnar_batch.metadata["terminal_sample_row_count"] == 1
    assert lazy_columnar_batch.metadata["resident_terminal_final_observation_used"] is True

    live_snapshots = [
        _resident_snapshot_for_terminal_sample_test(
            torch,
            latest_value=50 + index,
            generation_id=10 + index,
        )
        for index in range(4)
    ]
    live_steps = [
        _resident_step_for_terminal_sample_test(
            snapshot,
            action=index % ACTION_COUNT,
            reward=float(index),
            final_reward=float(index),
            done=False,
        )
        for index, snapshot in enumerate(live_snapshots)
    ]
    mixed_ring = _CompactReplayRingV1(capacity=8, metadata=tensor_native_metadata)
    for record_index in range(3):
        mixed_ring.append(
            previous_step=live_steps[record_index],
            current_step=live_steps[record_index + 1],
            index_rows=_resident_device_index_rows_for_terminal_sample_test(
                torch,
                record_index=record_index,
                action=(record_index + 1) % ACTION_COUNT,
                root_value=float(record_index),
                reward=float(record_index + 1),
                final_reward=float(record_index + 1),
                done=False,
            ),
        )
    mixed_ring.append(
        previous_step=previous_step,
        current_step=current_step,
        index_rows=partial_terminal_index_rows,
    )
    mixed_result = mixed_ring.sample(
        seed=1,
        sample_batch_size=2,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    mixed_batch = mixed_result["learner_batch"]
    mixed_telemetry = mixed_result["telemetry"]

    assert mixed_result["sample_batch"] is None
    assert mixed_telemetry["compact_rollout_slab_sample_gate_stored_pair_count"] == 2
    assert (
        mixed_telemetry["compact_rollout_slab_sample_gate_candidate_universe_source"]
        == "maintained_sample_universe_v1"
    )
    assert (
        mixed_telemetry["compact_rollout_slab_sample_gate_candidate_universe_cache_hit"]
        is True
    )
    assert mixed_batch.metadata["compact_muzero_learner_batch_tensor_native_replay_used"] is True
    assert (
        mixed_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_reused_record_count"
        ]
        == 2
    )
    assert (
        mixed_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_missing_record_count"
        ]
        == 0
    )
    assert mixed_batch.metadata["compact_muzero_learner_batch_tensor_native_replay_table_rows"] == 2
    assert (
        mixed_batch.metadata[
            "compact_muzero_learner_batch_learner_ready_unroll2_cache_fallback_count"
        ]
        == 0
    )

    fixed_soa_metadata = {
        **tensor_native_metadata,
        hybrid_profile.COMPACT_REPLAY_FIXED_SOA_UNROLL2_BUFFER_REQUESTED_KEY: True,
        hybrid_profile.COMPACT_REPLAY_FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY: True,
    }
    fixed_soa_ring = _CompactReplayRingV1(capacity=8, metadata=fixed_soa_metadata)
    for record_index in range(3):
        assert (
            fixed_soa_ring.append_fixed_soa_columnar_records(
                (
                    CompactReplayColumnarAppendRecordV1(
                        previous_resident_observation_replay_snapshot=live_snapshots[
                            record_index
                        ],
                        current_resident_observation_replay_snapshot=live_snapshots[
                            record_index + 1
                        ],
                        index_rows=_resident_device_index_rows_for_terminal_sample_test(
                            torch,
                            record_index=record_index,
                            action=(record_index + 1) % ACTION_COUNT,
                            root_value=float(record_index),
                            reward=float(record_index + 1),
                            final_reward=float(record_index + 1),
                            done=False,
                        ),
                    ),
                )
            )
            == 1
        )
    assert (
        fixed_soa_ring.append_fixed_soa_columnar_records(
            (
                CompactReplayColumnarAppendRecordV1(
                    previous_resident_observation_replay_snapshot=previous_snapshot,
                    current_resident_observation_replay_snapshot=current_snapshot,
                    index_rows=partial_terminal_index_rows,
                ),
            )
        )
        == 1
    )
    assert fixed_soa_ring.columnar_append_telemetry_snapshot()["record_count"] == 0.0
    fixed_soa_append_telemetry = fixed_soa_ring.fixed_soa_append_telemetry_snapshot()
    assert fixed_soa_append_telemetry["slot_write_count"] == 4.0
    assert fixed_soa_append_telemetry["entry_view_object_count"] == 0.0
    assert fixed_soa_append_telemetry["step_view_object_count"] == 0.0

    fixed_soa_result = fixed_soa_ring.sample(
        seed=1,
        sample_batch_size=2,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    fixed_soa_batch = fixed_soa_result["learner_batch"]
    fixed_soa_telemetry = fixed_soa_result["telemetry"]

    assert fixed_soa_result["sample_batch"] is None
    for field_name in (
        "observation",
        "action",
        "action_mask",
        "target_reward",
        "target_value",
        "target_policy",
        "target_reward_mask",
        "target_value_mask",
        "target_policy_mask",
        "action_valid_mask",
        "weights",
        "done",
        "terminated",
        "truncated",
    ):
        torch.testing.assert_close(
            getattr(fixed_soa_batch, field_name),
            getattr(mixed_batch, field_name),
            rtol=0,
            atol=0,
        )
    assert fixed_soa_batch.next_action_mask is None
    assert mixed_batch.next_action_mask is None
    assert (
        fixed_soa_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_impl"
        ]
        == "fixed_soa_direct_gather_v1"
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_muzero_learner_batch_fixed_soa_slot_candidate_path"
        ]
        is True
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_source"
        ]
        == "fixed_soa_columns_v1"
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_requested"
        ]
        is True
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_used"
        ]
        is True
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_create_count"
        ]
        == 1
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_resolve_count"
        ]
        == 1
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_inline_resolve_count"
        ]
        == 1
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_fallback_count"
        ]
        == 0
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_fallback_reason"
        ]
        == "none"
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_pending_handle_count"
        ]
        == 0
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_sample_row_count"
        ]
        == fixed_soa_result["sample_row_count"]
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_target_row_count"
        ]
        == fixed_soa_result["target_row_count"]
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_concat_sec"
        ]
        == 0.0
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_muzero_learner_batch_fixed_soa_table_entry_object_count"
        ]
        == 0
    )
    assert (
        fixed_soa_batch.metadata[
            "compact_muzero_learner_batch_fixed_soa_learner_ready_object_count"
        ]
        == 0
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_used"
        ]
        is True
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_candidate_universe_slot_candidate_path"
        ]
        is True
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count"
        ]
        == 0
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_selected_path_used"
        ]
        is False
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_requested"
        ]
        is True
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_used"
        ]
        is True
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_create_count"
        ]
        == 1
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_resolve_count"
        ]
        == 1
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_count"
        ]
        == 0
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_reason"
        ]
        == "none"
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_request_checksum"
        ]
        == fixed_soa_batch.metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_request_checksum"
        ]
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec"
        ]
        == 0.0
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count"
        ]
        == 0
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count"
        ]
        == 0
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count"
        ]
        == 0
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count"
        ]
        == 0
    )
    assert (
        fixed_soa_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count"
        ]
        == 0
    )


def test_fixed_soa_locality_sampler_reduces_selected_record_count_and_marks_drift():
    torch = pytest.importorskip("torch")

    records = []
    snapshots = [
        _resident_snapshot_for_terminal_sample_test(
            torch,
            latest_value=100 + index,
            generation_id=index,
        )
        for index in range(6)
    ]
    for record_index in range(6):
        records.append(
            CompactReplayColumnarAppendRecordV1(
                previous_resident_observation_replay_snapshot=snapshots[record_index],
                current_resident_observation_replay_snapshot=snapshots[
                    record_index + 1
                ]
                if record_index + 1 < len(snapshots)
                else snapshots[record_index],
                index_rows=_resident_device_index_rows_for_terminal_sample_test(
                    torch,
                    record_index=record_index,
                    action=(record_index + 1) % ACTION_COUNT,
                    root_value=float(record_index),
                    reward=float(record_index + 1),
                    done=False,
                ),
            )
        )

    base_metadata = {
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
        "compact_muzero_learner_batch_tensor_native_replay": True,
        hybrid_profile.COMPACT_REPLAY_FIXED_SOA_UNROLL2_BUFFER_REQUESTED_KEY: True,
    }
    exact_ring = _CompactReplayRingV1(capacity=8, metadata=base_metadata)
    assert exact_ring.append_fixed_soa_columnar_records(records) == len(records)
    exact_result = exact_ring.sample(
        seed=7,
        sample_batch_size=4,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    exact_metadata = exact_result["learner_batch"].metadata
    assert exact_metadata["fixed_soa_locality_sample_used"] is False
    assert exact_metadata["compact_muzero_learner_batch_fixed_soa_slot_candidate_path"] is True
    assert exact_metadata["compact_muzero_learner_batch_fixed_soa_selected_record_count"] == 4

    explicit_off_metadata = {
        **base_metadata,
        hybrid_profile.COMPACT_REPLAY_FIXED_SOA_LOCALITY_SAMPLE_GROUP_SIZE_KEY: 1,
    }
    explicit_off_ring = _CompactReplayRingV1(
        capacity=8,
        metadata=explicit_off_metadata,
    )
    assert explicit_off_ring.append_fixed_soa_columnar_records(records) == len(records)
    explicit_off_result = explicit_off_ring.sample(
        seed=7,
        sample_batch_size=4,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    explicit_off_batch = explicit_off_result["learner_batch"]
    for field_name in (
        "observation",
        "action",
        "target_reward",
        "target_value",
        "target_policy",
        "done",
        "terminated",
        "truncated",
    ):
        torch.testing.assert_close(
            getattr(explicit_off_batch, field_name),
            getattr(exact_result["learner_batch"], field_name),
            rtol=0,
            atol=0,
        )
    assert explicit_off_batch.metadata["fixed_soa_locality_sample_used"] is False
    assert (
        explicit_off_batch.metadata["sampled_flat_row_checksum"]
        == exact_result["learner_batch"].metadata["sampled_flat_row_checksum"]
    )

    locality_metadata = {
        **base_metadata,
        hybrid_profile.COMPACT_REPLAY_FIXED_SOA_LOCALITY_SAMPLE_GROUP_SIZE_KEY: 4,
    }
    locality_ring = _CompactReplayRingV1(capacity=8, metadata=locality_metadata)
    assert locality_ring.append_fixed_soa_columnar_records(records) == len(records)
    locality_result = locality_ring.sample(
        seed=7,
        sample_batch_size=4,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    locality_batch = locality_result["learner_batch"]
    locality_telemetry = locality_result["telemetry"]
    assert locality_batch.observation.shape[0] == 4
    assert locality_batch.metadata["fixed_soa_locality_sample_used"] is True
    assert (
        locality_batch.metadata["fixed_soa_locality_sample_semantic_drift"]
        is True
    )
    assert (
        locality_batch.metadata[
            "compact_muzero_learner_batch_fixed_soa_selected_record_count"
        ]
        == 1
    )
    assert (
        locality_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used"
        ]
        is True
    )
    assert (
        locality_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift"
        ]
        is True
    )
    assert (
        locality_telemetry[
            "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count"
        ]
        == 1
    )


def test_fixed_soa_samples_row_level_successors_for_coalesced_transition_batch():
    torch = pytest.importorskip("torch")

    def snapshot(values, *, generation_id, final_values=None, final_mask=None):
        values_array = list(values)
        device_observation = torch.zeros(
            (len(values_array), 1, 4, 8, 8),
            dtype=torch.uint8,
        )
        for row, value in enumerate(values_array):
            device_observation[row, 0, -1, 0, 0] = int(value)
        final_device_observation = None
        root_final_device_observation = None
        final_row_mask = np.zeros((len(values_array),), dtype=np.bool_)
        if final_values is not None:
            final_device_observation = device_observation.clone()
            for row, value in enumerate(list(final_values)):
                final_device_observation[row, 0, -1, 0, 0] = int(value)
            root_final_device_observation = final_device_observation.reshape(
                len(values_array),
                4,
                8,
                8,
            )
            final_row_mask = np.asarray(final_mask, dtype=np.bool_).reshape(-1)
        return ResidentObservationBatchV1(
            device_observation=device_observation,
            root_device_observation=device_observation.reshape(
                len(values_array),
                4,
                8,
                8,
            ),
            generation_id=int(generation_id),
            batch_size=len(values_array),
            player_count=1,
            stack_shape=(4, 8, 8),
            dtype="torch.uint8",
            device="cpu",
            row_major_order=True,
            fresh_for_step_index=int(generation_id),
            source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
            host_fallback_allowed=False,
            metadata={},
            final_device_observation=final_device_observation,
            root_final_device_observation=root_final_device_observation,
            final_observation_row_mask=final_row_mask,
        )

    def rows(*, record_index, env_rows, actions, rewards, done):
        env = torch.tensor(env_rows, dtype=torch.int32)
        action = torch.tensor(actions, dtype=torch.int16)
        done_tensor = torch.tensor(done, dtype=torch.bool)
        policy = torch.zeros((len(env_rows), ACTION_COUNT), dtype=torch.float32)
        policy[torch.arange(len(env_rows)), action.to(dtype=torch.long)] = 1.0
        done_indices = np.flatnonzero(np.asarray(done, dtype=np.bool_)).astype(
            np.int64,
            copy=False,
        )
        return CompactDeviceReplayIndexRowsV1(
            metadata={
                "device_replay_index_rows": True,
                "done_row_count": int(done_indices.size),
                "done_row_indices": done_indices.tolist(),
                "next_final_observation_row_count": int(done_indices.size),
                "next_final_observation_row_indices": done_indices.tolist(),
                "host_search_payload_fallback_allowed": False,
            },
            record_index=int(record_index),
            next_record_index=int(record_index) + 1,
            compact_root_row=torch.arange(len(env_rows), dtype=torch.int32),
            policy_env_id=torch.as_tensor(env_rows, dtype=torch.int64),
            policy_row=torch.arange(len(env_rows), dtype=torch.int32),
            env_row=env,
            player=torch.zeros((len(env_rows),), dtype=torch.int16),
            action=action,
            action_mask=torch.ones((len(env_rows), ACTION_COUNT), dtype=torch.bool),
            policy_target=policy,
            root_value=torch.arange(len(env_rows), dtype=torch.float32)
            + float(record_index),
            reward=torch.tensor(rewards, dtype=torch.float32),
            final_reward=torch.tensor(rewards, dtype=torch.float32),
            done=done_tensor,
            terminated=done_tensor.clone(),
            truncated=torch.zeros((len(env_rows),), dtype=torch.bool),
            next_final_observation_row=done_tensor.clone(),
            to_play=torch.full((len(env_rows),), -1, dtype=torch.int64),
            policy_source="unit_test_fixed_soa_coalesced",
        )

    snapshots = [
        snapshot([10, 20], generation_id=0),
        snapshot(
            [11, 21],
            generation_id=1,
            final_values=[11, 77],
            final_mask=[False, True],
        ),
        snapshot([12, 22], generation_id=2),
        snapshot([13, 23], generation_id=3),
    ]
    records = (
        CompactReplayColumnarAppendRecordV1(
            previous_resident_observation_replay_snapshot=snapshots[0],
            current_resident_observation_replay_snapshot=snapshots[1],
            index_rows=rows(
                record_index=0,
                env_rows=[1, 0],
                actions=[1, 0],
                rewards=[2.0, 1.0],
                done=[True, False],
            ),
        ),
        CompactReplayColumnarAppendRecordV1(
            previous_resident_observation_replay_snapshot=snapshots[1],
            current_resident_observation_replay_snapshot=snapshots[2],
            index_rows=rows(
                record_index=1,
                env_rows=[0, 1],
                actions=[2, 1],
                rewards=[3.0, 4.0],
                done=[False, False],
            ),
        ),
        CompactReplayColumnarAppendRecordV1(
            previous_resident_observation_replay_snapshot=snapshots[2],
            current_resident_observation_replay_snapshot=snapshots[3],
            index_rows=rows(
                record_index=2,
                env_rows=[1, 0],
                actions=[0, 2],
                rewards=[5.0, 6.0],
                done=[False, False],
            ),
        ),
    )
    incremental_ring = _CompactReplayRingV1(
        capacity=8,
        metadata={
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
            "compact_muzero_learner_batch_tensor_native_replay": True,
            hybrid_profile.COMPACT_REPLAY_FIXED_SOA_UNROLL2_BUFFER_REQUESTED_KEY: True,
        },
    )
    for record, expected_eligible_rows in zip(records, (1, 1, 2), strict=True):
        assert incremental_ring.append_fixed_soa_columnar_records((record,)) == 1
        incremental_result = incremental_ring.sample(
            seed=1,
            sample_batch_size=0,
            require_next_targets=True,
            num_unroll_steps=2,
            build_compact_muzero_learner_batch=True,
            compact_muzero_learner_batch_only=True,
        )
        assert (
            incremental_result["telemetry"][
                "compact_rollout_slab_sample_gate_stored_index_row_count"
            ]
            == expected_eligible_rows
        )

    fixed_soa_ring = _CompactReplayRingV1(
        capacity=8,
        metadata={
            "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
            "compact_muzero_learner_batch_tensor_native_replay": True,
            hybrid_profile.COMPACT_REPLAY_FIXED_SOA_UNROLL2_BUFFER_REQUESTED_KEY: True,
        },
    )
    assert fixed_soa_ring.append_fixed_soa_columnar_records(records) == 3

    result = fixed_soa_ring.sample(
        seed=1,
        sample_batch_size=0,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )

    batch = result["learner_batch"]
    telemetry = result["telemetry"]
    assert result["sample_batch"] is None
    assert telemetry["compact_rollout_slab_sample_gate_fixed_soa_used"] is True
    assert telemetry["compact_rollout_slab_sample_gate_sample_row_count"] == 2
    assert telemetry["compact_rollout_slab_sample_gate_stored_index_row_count"] == 2
    assert telemetry["compact_rollout_slab_sample_gate_raw_stored_index_row_count"] == 6
    assert batch.metadata["compact_muzero_learner_done_count"] == 1
    assert batch.metadata["terminal_sample_row_count"] == 1
    assert batch.metadata["next_final_observation_row_count"] == 1
    assert batch.done.to(dtype=torch.int64).sum().item() == 1
    current_done = batch.done[:, 0].to(dtype=torch.bool)
    assert int(current_done.sum().item()) == 1
    assert int((~current_done).sum().item()) == 1
    assert batch.metadata["terminal_final_observation_group_count"] == 1
    assert (
        batch.metadata["compact_muzero_learner_batch_tensor_native_replay_impl"]
        == "fixed_soa_direct_gather_v1"
    )
    assert batch.metadata["compact_muzero_learner_batch_fixed_soa_slot_candidate_path"] is True


def test_compact_replay_ring_batch_append_deduplicates_tensor_native_refresh(monkeypatch):
    torch = pytest.importorskip("torch")
    tensor_native_metadata = {
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
        "compact_muzero_learner_batch_tensor_native_replay": True,
        "compact_muzero_learner_batch_tensor_native_replay_selected_maintained_gather": True,
    }
    snapshots = [
        _resident_snapshot_for_terminal_sample_test(
            torch,
            latest_value=100 + index,
            generation_id=100 + index,
        )
        for index in range(5)
    ]
    steps = [
        _resident_step_for_terminal_sample_test(
            snapshot,
            action=index % ACTION_COUNT,
            reward=float(index),
            final_reward=float(index),
            done=False,
        )
        for index, snapshot in enumerate(snapshots)
    ]
    original_build = hybrid_profile._compact_tensor_native_unroll2_table_entry_for_entry
    table_build_record_indices: list[int] = []

    def wrapped_table_build(entry, *args, **kwargs):
        table_build_record_indices.append(int(getattr(entry.index_rows, "record_index")))
        return original_build(entry, *args, **kwargs)

    monkeypatch.setattr(
        hybrid_profile,
        "_compact_tensor_native_unroll2_table_entry_for_entry",
        wrapped_table_build,
    )

    ring = _CompactReplayRingV1(capacity=8, metadata=tensor_native_metadata)
    entries = tuple(
        hybrid_profile._CompactReplayRingEntry(
            previous_step=steps[record_index],
            current_step=steps[record_index + 1],
            index_rows=_resident_device_index_rows_for_terminal_sample_test(
                torch,
                record_index=record_index,
                action=(record_index + 1) % ACTION_COUNT,
                root_value=float(record_index),
                reward=float(record_index + 1),
                final_reward=float(record_index + 1),
                done=False,
            ),
        )
        for record_index in range(4)
    )

    assert ring.append_entries(entries) == 4
    assert ring.snapshot_version == 4
    assert table_build_record_indices == [0, 1]

    result = ring.sample(
        seed=0,
        sample_batch_size=2,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    learner_batch = result["learner_batch"]
    telemetry = result["telemetry"]

    assert result["sample_batch"] is None
    assert learner_batch.metadata["compact_muzero_learner_batch_tensor_native_replay_used"] is True
    assert (
        learner_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_build_impl"
        ]
        == "direct_record_table_v1"
    )
    assert (
        learner_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_reused_record_count"
        ]
        == 2
    )
    assert (
        learner_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_missing_record_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source"
        ]
        == "selected_maintained_record_table_v1"
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"
        ]
        == "selected_maintained_record_table_gather_v1"
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested"
        ]
        is True
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used"
        ]
        is True
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count"
        ]
        == learner_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_reused_record_count"
        ]
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used"
        ]
        is True
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count"
        ]
        == 0
    )


def test_compact_replay_ring_tensor_native_uses_maintained_table_handle():
    torch = pytest.importorskip("torch")
    tensor_native_metadata = {
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
        "compact_muzero_learner_batch_tensor_native_replay": True,
        "compact_muzero_learner_batch_tensor_native_replay_maintained_table_handle": True,
    }
    snapshots = [
        _resident_snapshot_for_terminal_sample_test(
            torch,
            latest_value=150 + index,
            generation_id=150 + index,
        )
        for index in range(5)
    ]
    steps = [
        _resident_step_for_terminal_sample_test(
            snapshot,
            action=index % ACTION_COUNT,
            reward=float(index),
            final_reward=float(index),
            done=False,
        )
        for index, snapshot in enumerate(snapshots)
    ]
    ring = _CompactReplayRingV1(capacity=8, metadata=tensor_native_metadata)
    entries = tuple(
        hybrid_profile._CompactReplayRingEntry(
            previous_step=steps[record_index],
            current_step=steps[record_index + 1],
            index_rows=_resident_device_index_rows_for_terminal_sample_test(
                torch,
                record_index=record_index,
                action=(record_index + 1) % ACTION_COUNT,
                root_value=float(record_index),
                reward=float(record_index + 1),
                final_reward=float(record_index + 1),
                done=False,
            ),
        )
        for record_index in range(4)
    )

    assert ring.append_entries(entries) == 4
    snapshot = ring.snapshot_for_sample()
    handle = snapshot.tensor_native_unroll2_table_handle
    assert handle is not None
    assert handle.schema_id == "curvyzero_compact_replay_tensor_native_unroll2_table_handle/v1"
    assert handle.candidate_record_indices == (0, 1)
    assert handle.missing_record_count == 0
    assert handle.table_row_count == 2

    result = ring.sample_from_snapshot(
        snapshot,
        seed=0,
        sample_batch_size=2,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    learner_batch = result["learner_batch"]
    telemetry = result["telemetry"]

    assert result["sample_batch"] is None
    assert (
        learner_batch.metadata["compact_muzero_learner_batch_tensor_native_replay_impl"]
        == "maintained_unroll2_table_gather_v1"
    )
    assert (
        learner_batch.metadata["compact_muzero_learner_batch_tensor_native_replay_table_source"]
        == "maintained_record_table_v1"
    )
    assert (
        learner_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_maintained_table_handle_requested"
        ]
        is True
    )
    assert (
        learner_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_maintained_table_handle_used"
        ]
        is True
    )
    assert (
        learner_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_maintained_table_handle_record_count"
        ]
        == 2
    )
    assert (
        learner_batch.metadata[
            "compact_muzero_learner_batch_tensor_native_replay_maintained_table_handle_rows"
        ]
        == 2
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested"
        ]
        is True
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used"
        ]
        is True
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count"
        ]
        == 2
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_missing_record_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used"
        ]
        is False
    )


def test_compact_fast_sample_nonterminal_final_reward_uses_index_rows():
    observation_shape = (1, 2, 4, 64, 64)
    previous_observation = np.zeros(observation_shape, dtype=np.float32)
    current_observation = np.ones(observation_shape, dtype=np.float32)
    action_mask = np.ones((1, 2, ACTION_COUNT), dtype=np.bool_)

    previous_step = SimpleNamespace(
        observation=previous_observation,
        action_mask=action_mask,
    )
    current_step = SimpleNamespace(
        observation=current_observation,
        action_mask=action_mask,
        reward=np.asarray([[3.0, 0.0]], dtype=np.float32),
        final_reward_map=np.asarray([[99.0, 0.0]], dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        payload={"joint_action": np.asarray([[2, 0]], dtype=np.int16)},
        compact_batch=None,
    )
    index_rows = SimpleNamespace(
        env_row=np.asarray([0], dtype=np.int32),
        player=np.asarray([0], dtype=np.int16),
        action=np.asarray([2], dtype=np.int16),
        action_mask=np.asarray([[True, True, True]], dtype=np.bool_),
        policy_target=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        root_value=np.asarray([0.5], dtype=np.float32),
        reward=np.asarray([3.0], dtype=np.float32),
        final_reward=np.asarray([3.0], dtype=np.float32),
        done=np.asarray([False], dtype=np.bool_),
        terminated=np.asarray([False], dtype=np.bool_),
        truncated=np.asarray([False], dtype=np.bool_),
        next_final_observation_row=np.asarray([False], dtype=np.bool_),
        policy_row=np.asarray([0], dtype=np.int32),
        record_index=21,
        next_record_index=22,
    )

    batch = _build_compact_sample_batch_from_index_rows_fast(
        previous_step=previous_step,
        current_step=current_step,
        source_index_row=np.asarray([0], dtype=np.int64),
        index_rows=index_rows,
        seed=23,
        replace=False,
    )

    assert batch.final_reward.tolist() == [3.0]
    np.testing.assert_array_equal(batch.next_observation[0], current_observation[0, 0])


def test_hybrid_profile_compact_rollout_slab_learner_gate_consumes_samples():
    class FakeSearchService:
        search_impl = "unit_test_compact_learner_gate"
        num_simulations = 1

        def run(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            legal = root_batch.legal_mask[active_roots]
            selected = np.argmax(legal, axis=1).astype(np.int16, copy=False)
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeSearchService(),
        search_lane="unit_test_compact_learner_gate",
        policy_source="unit_test_compact_learner_gate",
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=20260524,
            materialize_scalar_timestep=False,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=2,
            compact_rollout_slab_sample_gate_interval=1,
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_device="cpu",
            compact_rollout_slab_learner_gate_include_rnd=True,
        ),
        compact_rollout_slab=slab,
    )

    assert result["materialized_timestep_count"] == 0
    assert result["compact_rollout_slab_sample_gate_calls"] > 0
    assert result["compact_rollout_slab_learner_gate_enabled"] is True
    assert (
        result["compact_rollout_slab_learner_gate_calls"]
        == result["compact_rollout_slab_sample_gate_calls"]
    )
    assert (
        result["compact_rollout_slab_learner_gate_updates"]
        == result["compact_rollout_slab_learner_gate_calls"]
    )
    assert (
        result["compact_rollout_slab_learner_gate_sample_row_count"]
        == result["compact_rollout_slab_sample_gate_sample_row_count"]
    )
    assert result["compact_rollout_slab_learner_gate_input_bytes"] > 0
    assert result["compact_rollout_slab_learner_gate_sec"] > 0.0
    assert (
        result["compact_rollout_slab_learner_gate_last_telemetry"][
            "compact_rollout_slab_learner_gate_include_rnd"
        ]
        is True
    )
    assert result["contract"]["compact_rollout_slab_learner_gate"] is True


def test_hybrid_profile_compact_owned_loop_entrypoint_surfaces_handoff_claims():
    class FakeSearchService:
        search_impl = "unit_test_compact_owned_loop_entrypoint"
        num_simulations = 1

        def run(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            legal = root_batch.legal_mask[active_roots]
            selected = np.argmax(legal, axis=1).astype(np.int16, copy=False)
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeSearchService(),
        search_lane="unit_test_compact_owned_loop_entrypoint",
        policy_source="unit_test_compact_owned_loop_entrypoint",
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=2026052801,
            materialize_scalar_timestep=False,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=2,
            compact_rollout_slab_sample_gate_interval=1,
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_device="cpu",
            compact_owned_loop_entrypoint=True,
            compact_owned_loop_policy_version_ref="unit-owned-profile-policy-v1",
            compact_owned_loop_model_version_ref="unit-owned-profile-model-v1",
            compact_owned_loop_policy_source="unit_test_compact_owned_loop_entrypoint",
            compact_owned_loop_capture_replay_store_state=True,
        ),
        compact_rollout_slab=slab,
    )

    assert result["materialized_timestep_count"] == 0
    assert result["compact_owned_loop_entrypoint_enabled"] is True
    assert result["compact_owned_loop_schema_id"] == COMPACT_OWNED_LOOP_SCHEMA_ID
    assert result["compact_owned_loop_profile_only"] is True
    assert result["compact_owned_loop_calls_train_muzero"] is False
    assert result["compact_owned_loop_touches_live_runs"] is False
    assert result["compact_owned_loop_replay_store_owned"] is True
    assert result["compact_owned_loop_policy_version_handoff"] is True
    assert result["compact_owned_loop_policy_version_ref"] == "unit-owned-profile-policy-v1"
    assert result["compact_owned_loop_model_version_ref"] == "unit-owned-profile-model-v1"
    assert result["compact_owned_loop_policy_source"] == "unit_test_compact_owned_loop_entrypoint"
    assert result["compact_rollout_slab_sample_gate_calls"] > 0
    assert (
        result["compact_rollout_slab_learner_gate_calls"]
        == result["compact_rollout_slab_sample_gate_calls"]
    )
    state_metadata = result["compact_owned_loop_replay_store_state_metadata"]
    assert state_metadata["schema_id"] == COMPACT_REPLAY_STORE_STATE_SCHEMA_ID
    assert state_metadata["compact_owned_loop_entrypoint"] is True
    assert state_metadata["compact_owned_loop_replay_store_owned"] is True
    telemetry = result["compact_owned_loop_telemetry"]
    sample_metadata = telemetry["compact_owned_loop_sample_gate_last_sample_metadata"]
    assert sample_metadata["compact_owned_loop_entrypoint"] is True
    assert sample_metadata["compact_owned_loop_replay_store_owned"] is True
    assert sample_metadata["compact_owned_loop_policy_version_ref"] == (
        "unit-owned-profile-policy-v1"
    )
    assert sample_metadata["compact_owned_loop_model_version_ref"] == (
        "unit-owned-profile-model-v1"
    )
    assert sample_metadata["profile_only"] is True
    assert sample_metadata["calls_train_muzero"] is False
    assert sample_metadata["touches_live_runs"] is False
    assert result["contract"]["compact_owned_loop_entrypoint"] is True
    assert result["contract"]["compact_owned_loop_policy_version_handoff"] is True
    assert result["contract"]["compact_owned_loop_calls_train_muzero"] is False


def test_resident_sample_gate_learner_uses_resident_observation_batch():
    class FakeSearchService:
        search_impl = "unit_test_resident_sample_learner_gate"
        num_simulations = 1

        def run(self, root_batch):
            assert root_batch.observation_source == "resident_device_v1"
            assert root_batch.resident_observation is not None
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            legal = root_batch.legal_mask[active_roots]
            selected = np.argmax(legal, axis=1).astype(np.int16, copy=False)
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeSearchService(),
        search_lane="unit_test_resident_sample_learner_gate",
        policy_source="unit_test_resident_sample_learner_gate",
        copy_root_observation=False,
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=20260526,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=2,
            compact_rollout_slab_sample_gate_interval=1,
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_device="cpu",
            compact_rollout_slab_learner_gate_include_rnd=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        compact_rollout_slab=slab,
    )

    assert result["materialized_timestep_count"] == 0
    assert (
        result["compact_rollout_slab_sample_gate_last_telemetry"][
            "compact_rollout_slab_sample_gate_resident_sample_batch"
        ]
        is True
    )
    last_learner = result["compact_rollout_slab_learner_gate_last_telemetry"]
    assert last_learner["compact_rollout_slab_learner_gate_resident_sample_used"] is True
    assert last_learner["compact_rollout_slab_learner_gate_observation_h2d_bytes"] == 0


def test_resident_sample_gate_uses_device_replay_targets():
    torch = pytest.importorskip("torch")

    class DeviceTargetSearchService:
        supports_two_phase_compact_search = True
        search_impl = "unit_test_device_target_sample_gate"
        num_simulations = 1

        def __init__(self):
            self.counter = 0
            self.pending = {}
            self.host_flushes = 0

        def run_action_step(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            legal = root_batch.legal_mask[active_roots]
            selected = np.argmax(legal, axis=1).astype(np.int16, copy=False)
            handle = f"device-target:{self.counter}"
            self.counter += 1
            self.pending[handle] = (
                active_roots,
                root_batch.env_row[active_roots].astype(np.int32, copy=True),
                root_batch.player[active_roots].astype(np.int16, copy=True),
                root_batch.policy_env_id[active_roots].astype(np.int64, copy=True),
                selected.copy(),
            )
            return CompactSearchActionStepV1(
                replay_payload_handle=handle,
                root_index=active_roots,
                env_row=self.pending[handle][1],
                player=self.pending[handle][2],
                policy_env_id=self.pending[handle][3],
                selected_action=selected,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(active_roots.size),
                    "replay_payload_origin": f"{self.search_impl}:{handle}",
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                    "search_replay_payload_digest": (
                        compact_search_deferred_replay_payload_digest_v1(handle)
                    ),
                    "search_replay_payload_digest_deferred": True,
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            self.host_flushes += 1
            raise AssertionError(f"unexpected host replay flush: {replay_payload_handle}")

        def flush_device_replay_payload(self, replay_payload_handle):
            root_index, env_row, player, policy_env_id, selected = self.pending.pop(
                replay_payload_handle
            )
            visit_policy = torch.zeros((root_index.size, ACTION_COUNT), dtype=torch.float32)
            visit_policy[
                torch.arange(root_index.size),
                torch.as_tensor(selected, dtype=torch.long),
            ] = 1.0
            root_value = torch.zeros((root_index.size,), dtype=torch.float32)
            return CompactDeviceSearchReplayPayloadV1(
                replay_payload_handle=str(replay_payload_handle),
                root_index=root_index,
                env_row=env_row,
                player=player,
                policy_env_id=policy_env_id,
                visit_policy=visit_policy,
                root_value=root_value,
                raw_visit_counts=visit_policy.clone(),
                predicted_value=None,
                predicted_policy_logits=None,
                metadata={
                    "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                    "phase": "replay_critical_device",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(root_index.size),
                    "replay_payload_origin": (f"{self.search_impl}:{str(replay_payload_handle)}"),
                    "device_replay_payload": True,
                    "host_search_payload_fallback_allowed": False,
                },
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=DeviceTargetSearchService(),
        search_lane="unit_test_device_target_sample_gate",
        policy_source="unit_test_device_target_sample_gate",
        copy_root_observation=False,
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=20260527,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=2,
            compact_rollout_slab_sample_gate_interval=1,
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_device="cpu",
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        compact_rollout_slab=slab,
    )

    assert (
        result["compact_rollout_slab_last_telemetry"][
            "compact_rollout_slab_committed_replay_payload_d2h_bytes"
        ]
        == 0
    )
    assert (
        result["compact_rollout_slab_sample_gate_last_telemetry"][
            "compact_rollout_slab_sample_gate_resident_sample_batch"
        ]
        is True
    )
    assert (
        result["compact_rollout_slab_sample_gate_last_telemetry"][
            "compact_rollout_slab_sample_gate_device_replay_index_rows"
        ]
        is True
    )
    sample_telemetry = result["compact_rollout_slab_sample_gate_last_telemetry"]
    assert (
        sample_telemetry["compact_rollout_slab_sample_gate_resident_grouped_device_sample_batch"]
        is True
    )
    assert (
        sample_telemetry["compact_rollout_slab_sample_gate_resident_grouped_device_direct_write"]
        is True
    )
    assert (
        sample_telemetry["compact_rollout_slab_sample_gate_terminal_zero_metadata_fast_path_count"]
        > 0
    )
    assert sample_telemetry["compact_rollout_slab_sample_gate_terminal_tensor_check_count"] == 0
    last_learner = result["compact_rollout_slab_learner_gate_last_telemetry"]
    assert last_learner["compact_rollout_slab_learner_gate_resident_sample_used"] is True
    assert last_learner["compact_rollout_slab_learner_gate_device_replay_index_rows_sample"] is True
    assert last_learner["compact_rollout_slab_learner_gate_observation_h2d_bytes"] == 0
    assert last_learner["compact_rollout_slab_learner_gate_input_h2d_bytes"] == 0


def test_compact_muzero_learner_gate_consumes_two_step_resident_sample():
    torch = pytest.importorskip("torch")

    class TinyMuZero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(4 * 64 * 64, 16),
                torch.nn.Tanh(),
            )
            self.action_embedding = torch.nn.Embedding(ACTION_COUNT, 16)
            self.policy_head = torch.nn.Linear(16, ACTION_COUNT)
            self.value_head = torch.nn.Linear(16, 3)
            self.reward_head = torch.nn.Linear(16, 3)
            self.recurrent_actions = []

        def initial_inference(self, obs):
            from lzero.model.common import MZNetworkOutput

            latent = self.encoder(obs)
            return MZNetworkOutput(
                self.value_head(latent),
                torch.zeros_like(self.value_head(latent)),
                self.policy_head(latent),
                latent,
            )

        def recurrent_inference(self, latent_state, action):
            from lzero.model.common import MZNetworkOutput

            self.recurrent_actions.append(action.detach().cpu().clone())
            next_latent = torch.tanh(
                latent_state + self.action_embedding(action.reshape(-1).long())
            )
            return MZNetworkOutput(
                self.value_head(next_latent),
                self.reward_head(next_latent),
                self.policy_head(next_latent),
                next_latent,
            )

    class DeviceTargetSearchService:
        supports_two_phase_compact_search = True
        search_impl = "unit_test_compact_muzero_unroll_gate"
        num_simulations = 1

        def __init__(self):
            self._model = TinyMuZero()
            self.counter = 0
            self.pending = {}

        def run_action_step(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = np.asarray(
                [self.counter % ACTION_COUNT for _ in range(active_roots.size)],
                dtype=np.int16,
            )
            handle = f"muzero-unroll:{self.counter}"
            self.counter += 1
            self.pending[handle] = (
                active_roots,
                root_batch.env_row[active_roots].astype(np.int32, copy=True),
                root_batch.player[active_roots].astype(np.int16, copy=True),
                root_batch.policy_env_id[active_roots].astype(np.int64, copy=True),
                selected.copy(),
            )
            return CompactSearchActionStepV1(
                replay_payload_handle=handle,
                root_index=active_roots,
                env_row=self.pending[handle][1],
                player=self.pending[handle][2],
                policy_env_id=self.pending[handle][3],
                selected_action=selected,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(active_roots.size),
                    "replay_payload_origin": f"{self.search_impl}:{handle}",
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                    "search_replay_payload_digest": (
                        compact_search_deferred_replay_payload_digest_v1(handle)
                    ),
                    "search_replay_payload_digest_deferred": True,
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"unexpected host replay flush: {replay_payload_handle}")

        def flush_device_replay_payload(self, replay_payload_handle):
            root_index, env_row, player, policy_env_id, selected = self.pending.pop(
                replay_payload_handle
            )
            visit_policy = torch.zeros((root_index.size, ACTION_COUNT), dtype=torch.float32)
            visit_policy[
                torch.arange(root_index.size),
                torch.as_tensor(selected, dtype=torch.long),
            ] = 1.0
            return CompactDeviceSearchReplayPayloadV1(
                replay_payload_handle=str(replay_payload_handle),
                root_index=root_index,
                env_row=env_row,
                player=player,
                policy_env_id=policy_env_id,
                visit_policy=visit_policy,
                root_value=torch.zeros((root_index.size,), dtype=torch.float32),
                raw_visit_counts=visit_policy.clone(),
                predicted_value=None,
                predicted_policy_logits=None,
                metadata={
                    "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                    "phase": "replay_critical_device",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(root_index.size),
                    "replay_payload_origin": (f"{self.search_impl}:{str(replay_payload_handle)}"),
                    "device_replay_payload": True,
                    "host_search_payload_fallback_allowed": False,
                },
            )

    search_service = DeviceTargetSearchService()
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=search_service,
        search_lane="unit_test_compact_muzero_unroll_gate",
        policy_source="unit_test_compact_muzero_unroll_gate",
        copy_root_observation=False,
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=5,
            warmup_steps=1,
            seed=2026052801,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=2,
            compact_rollout_slab_sample_gate_interval=1,
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_impl="compact_muzero",
            compact_rollout_slab_learner_gate_device="cpu",
            compact_rollout_slab_learner_gate_num_unroll_steps=2,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        compact_rollout_slab=slab,
    )

    last_sample = result["compact_rollout_slab_sample_gate_last_telemetry"]
    last_learner = result["compact_rollout_slab_learner_gate_last_telemetry"]
    assert result["compact_rollout_slab_learner_gate_calls"] > 0
    assert last_sample["compact_rollout_slab_sample_gate_explicit_unroll_targets"] is True
    assert last_sample["compact_rollout_slab_sample_gate_num_unroll_steps"] == 2
    assert last_learner["compact_rollout_slab_learner_gate_num_unroll_steps"] == 2
    assert last_learner["compact_rollout_slab_learner_gate_resident_sample_used"] is True
    assert last_learner["compact_rollout_slab_learner_gate_input_h2d_bytes"] == 0
    assert len(search_service._model.recurrent_actions) >= 2


def test_compact_replay_ring_resident_sample_attaches_committed_next_targets():
    torch = pytest.importorskip("torch")

    class DeviceTargetSearchService:
        supports_two_phase_compact_search = True
        search_impl = "unit_test_explicit_next_targets"
        num_simulations = 1

        def __init__(self):
            self.counter = 0
            self.pending = {}
            self.host_flushes = 0
            self.device_flushes = 0

        def run_action_step(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = np.asarray(
                [self.counter % ACTION_COUNT for _ in range(active_roots.size)],
                dtype=np.int16,
            )
            handle = f"explicit-next:{self.counter}"
            self.counter += 1
            self.pending[handle] = (
                active_roots,
                root_batch.env_row[active_roots].astype(np.int32, copy=True),
                root_batch.player[active_roots].astype(np.int16, copy=True),
                root_batch.policy_env_id[active_roots].astype(np.int64, copy=True),
                selected.copy(),
                self.counter,
            )
            return CompactSearchActionStepV1(
                replay_payload_handle=handle,
                root_index=active_roots,
                env_row=self.pending[handle][1],
                player=self.pending[handle][2],
                policy_env_id=self.pending[handle][3],
                selected_action=selected,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(active_roots.size),
                    "replay_payload_origin": f"{self.search_impl}:{handle}",
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                    "search_replay_payload_digest": (
                        compact_search_deferred_replay_payload_digest_v1(handle)
                    ),
                    "search_replay_payload_digest_deferred": True,
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            self.host_flushes += 1
            raise AssertionError(f"unexpected host replay flush: {replay_payload_handle}")

        def flush_device_replay_payload(self, replay_payload_handle):
            self.device_flushes += 1
            root_index, env_row, player, policy_env_id, selected, search_id = self.pending.pop(
                replay_payload_handle
            )
            visit_policy = torch.zeros((root_index.size, ACTION_COUNT), dtype=torch.float32)
            visit_policy[
                torch.arange(root_index.size),
                torch.as_tensor(selected, dtype=torch.long),
            ] = 1.0
            root_value = torch.full(
                (root_index.size,),
                float(search_id),
                dtype=torch.float32,
            )
            return CompactDeviceSearchReplayPayloadV1(
                replay_payload_handle=str(replay_payload_handle),
                root_index=root_index,
                env_row=env_row,
                player=player,
                policy_env_id=policy_env_id,
                visit_policy=visit_policy,
                root_value=root_value,
                raw_visit_counts=visit_policy.clone(),
                predicted_value=None,
                predicted_policy_logits=None,
                metadata={
                    "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                    "phase": "replay_critical_device",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(root_index.size),
                    "replay_payload_origin": (f"{self.search_impl}:{str(replay_payload_handle)}"),
                    "device_replay_payload": True,
                    "host_search_payload_fallback_allowed": False,
                },
            )

    search_service = DeviceTargetSearchService()
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=search_service,
        search_lane="unit_test_explicit_next_targets",
        policy_source="unit_test_explicit_next_targets",
        copy_root_observation=False,
    )
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=2026052601,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            compact_rollout_slab_sample_gate=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        compact_rollout_slab=slab,
    )
    ring = _CompactReplayRingV1(capacity=8)
    previous_sample_step = None
    action = np.zeros((2, 2), dtype=np.int16)
    sample = None
    for _ in range(5):
        step = manager.step(action)
        action = step.compact_rollout_slab_step.next_joint_action.astype(np.int16)
        committed = step.compact_rollout_slab_step.committed_index_rows
        if committed is not None and previous_sample_step is not None:
            ring.append(
                previous_step=previous_sample_step,
                current_step=step,
                index_rows=committed,
            )
            sample = ring.sample(seed=99, sample_batch_size=2, require_next_targets=True)
        previous_sample_step = step

    assert sample is not None
    assert search_service.host_flushes == 0
    assert search_service.device_flushes >= 3
    batch = sample["sample_batch"]
    assert batch.metadata["explicit_next_targets"] is True
    assert batch.next_policy_target is not None
    assert batch.next_root_value is not None
    assert batch.next_action_mask is not None
    assert isinstance(batch.next_policy_target, torch.Tensor)
    assert isinstance(batch.next_root_value, torch.Tensor)
    assert batch.next_policy_target.shape == batch.policy_target.shape
    assert batch.next_root_value.shape == batch.root_value.shape
    assert sample["telemetry"]["compact_rollout_slab_sample_gate_explicit_next_targets"] is True
    assert sample["telemetry"]["compact_rollout_slab_sample_gate_require_next_targets"] is True
    assert (
        sample["telemetry"]["compact_rollout_slab_sample_gate_next_target_excluded_pair_count"] >= 0
    )
    assert (
        sample["telemetry"]["compact_rollout_slab_sample_gate_raw_stored_pair_count"]
        >= (sample["telemetry"]["compact_rollout_slab_sample_gate_stored_pair_count"])
    )

    unroll_sample = ring.sample(
        seed=123,
        sample_batch_size=2,
        require_next_targets=True,
        num_unroll_steps=2,
    )
    unroll_batch = unroll_sample["sample_batch"]
    assert unroll_batch.metadata["explicit_unroll_targets"] is True
    assert (
        unroll_sample["telemetry"]["compact_rollout_slab_sample_gate_explicit_unroll_targets"]
        is True
    )
    assert unroll_sample["telemetry"]["compact_rollout_slab_sample_gate_num_unroll_steps"] == 2
    assert unroll_batch.unroll_action.shape == (2, 2)
    assert unroll_batch.unroll_reward.shape == (2, 2)
    assert unroll_batch.unroll_policy_target.shape == (2, 3, ACTION_COUNT)
    assert unroll_batch.unroll_root_value.shape == (2, 3)
    assert unroll_batch.unroll_action_mask.shape == (2, 3, ACTION_COUNT)
    assert torch.equal(unroll_batch.unroll_action[:, 0], unroll_batch.action)
    assert torch.equal(unroll_batch.unroll_action_mask[:, 0], unroll_batch.action_mask)
    assert torch.allclose(unroll_batch.unroll_policy_target[:, 0], unroll_batch.policy_target)
    assert torch.allclose(unroll_batch.unroll_root_value[:, 0], unroll_batch.root_value)
    assert torch.allclose(unroll_batch.unroll_reward[:, 0], unroll_batch.reward)


def test_compact_replay_ring_reports_rows_excluded_without_successor_targets():
    ring = _CompactReplayRingV1(capacity=4)
    ring.append(
        previous_step=object(),
        current_step=object(),
        index_rows=SimpleNamespace(
            record_index=10,
            next_record_index=11,
            action=np.asarray([0, 1], dtype=np.int16),
        ),
    )

    result = ring.sample(seed=7, sample_batch_size=2, require_next_targets=True)

    telemetry = result["telemetry"]
    assert result["sample_batch"] is None
    assert telemetry["compact_rollout_slab_sample_gate_raw_stored_pair_count"] == 1
    assert telemetry["compact_rollout_slab_sample_gate_raw_stored_index_row_count"] == 2
    assert telemetry["compact_rollout_slab_sample_gate_stored_pair_count"] == 0
    assert telemetry["compact_rollout_slab_sample_gate_stored_index_row_count"] == 0
    assert telemetry["compact_rollout_slab_sample_gate_next_target_eligible_pair_count"] == 0
    assert telemetry["compact_rollout_slab_sample_gate_next_target_eligible_index_row_count"] == 0
    assert telemetry["compact_rollout_slab_sample_gate_next_target_excluded_pair_count"] == 1
    assert telemetry["compact_rollout_slab_sample_gate_next_target_excluded_index_row_count"] == 2


def test_resident_sample_concat_rejects_mixed_device_replay_ownership():
    torch = pytest.importorskip("torch")

    def make_batch(*, device_replay: bool, row_id: int) -> CompactResidentSampleBatchV1:
        observation = torch.zeros((1, 4, 8, 8), dtype=torch.uint8)
        return CompactResidentSampleBatchV1(
            metadata={
                "resident_device_sample_batch": True,
                "device_replay_index_rows_sample": device_replay,
            },
            row_id=torch.tensor([row_id], dtype=torch.int64),
            observation=observation,
            action=torch.tensor([0], dtype=torch.int16),
            action_mask=torch.ones((1, ACTION_COUNT), dtype=torch.bool),
            policy_target=torch.zeros((1, ACTION_COUNT), dtype=torch.float32),
            root_value=torch.zeros((1,), dtype=torch.float32),
            reward=torch.zeros((1,), dtype=torch.float32),
            final_reward=torch.zeros((1,), dtype=torch.float32),
            done=torch.zeros((1,), dtype=torch.bool),
            terminated=torch.zeros((1,), dtype=torch.bool),
            truncated=torch.zeros((1,), dtype=torch.bool),
            next_observation=observation.clone(),
            to_play=torch.zeros((1,), dtype=torch.int64),
            env_row=torch.zeros((1,), dtype=torch.int32),
            player=torch.zeros((1,), dtype=torch.int16),
            record_index=torch.tensor([row_id], dtype=torch.int32),
            next_record_index=torch.tensor([row_id + 1], dtype=torch.int32),
            policy_row=torch.zeros((1,), dtype=torch.int32),
        )

    with pytest.raises(ValueError, match="mix device and host replay"):
        _concat_compact_resident_sample_batches_fast(
            [
                make_batch(device_replay=True, row_id=0),
                make_batch(device_replay=False, row_id=1),
            ],
            metadata={},
            sample_position_order=np.asarray([0, 1], dtype=np.int64),
        )


def test_hybrid_profile_compact_muzero_learner_gate_uses_explicit_next_targets():
    torch = pytest.importorskip("torch")

    class TinyMuZero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(8, 16),
                torch.nn.Tanh(),
            )
            self.action_embedding = torch.nn.Embedding(ACTION_COUNT, 16)
            self.policy_head = torch.nn.Linear(16, ACTION_COUNT)
            self.value_head = torch.nn.Linear(16, 3)
            self.reward_head = torch.nn.Linear(16, 3)

        def initial_inference(self, obs):
            from lzero.model.common import MZNetworkOutput

            latent = self.encoder(obs)
            policy_logits = self.policy_head(latent)
            value = self.value_head(latent)
            reward = torch.zeros_like(value)
            return MZNetworkOutput(value, reward, policy_logits, latent)

        def recurrent_inference(self, latent_state, action):
            from lzero.model.common import MZNetworkOutput

            next_latent = torch.tanh(
                latent_state + self.action_embedding(action.reshape(-1).long())
            )
            policy_logits = self.policy_head(next_latent)
            value = self.value_head(next_latent)
            reward = self.reward_head(next_latent)
            return MZNetworkOutput(value, reward, policy_logits, next_latent)

    class DeviceTargetSearchService:
        supports_two_phase_compact_search = True
        search_impl = "unit_test_compact_muzero_learner_gate"
        num_simulations = 1

        def __init__(self):
            self._model = TinyMuZero()
            self.counter = 0
            self.pending = {}

        def run_action_step(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = np.asarray(
                [self.counter % ACTION_COUNT for _ in range(active_roots.size)],
                dtype=np.int16,
            )
            handle = f"compact-muzero:{self.counter}"
            self.counter += 1
            self.pending[handle] = (
                active_roots,
                root_batch.env_row[active_roots].astype(np.int32, copy=True),
                root_batch.player[active_roots].astype(np.int16, copy=True),
                root_batch.policy_env_id[active_roots].astype(np.int64, copy=True),
                selected.copy(),
                self.counter,
            )
            return CompactSearchActionStepV1(
                replay_payload_handle=handle,
                root_index=active_roots,
                env_row=self.pending[handle][1],
                player=self.pending[handle][2],
                policy_env_id=self.pending[handle][3],
                selected_action=selected,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(active_roots.size),
                    "replay_payload_origin": f"{self.search_impl}:{handle}",
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                    "search_replay_payload_digest": (
                        compact_search_deferred_replay_payload_digest_v1(handle)
                    ),
                    "search_replay_payload_digest_deferred": True,
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"unexpected host replay flush: {replay_payload_handle}")

        def flush_device_replay_payload(self, replay_payload_handle):
            root_index, env_row, player, policy_env_id, selected, search_id = self.pending.pop(
                replay_payload_handle
            )
            visit_policy = torch.zeros((root_index.size, ACTION_COUNT), dtype=torch.float32)
            visit_policy[
                torch.arange(root_index.size),
                torch.as_tensor(selected, dtype=torch.long),
            ] = 1.0
            root_value = torch.full((root_index.size,), float(search_id), dtype=torch.float32)
            return CompactDeviceSearchReplayPayloadV1(
                replay_payload_handle=str(replay_payload_handle),
                root_index=root_index,
                env_row=env_row,
                player=player,
                policy_env_id=policy_env_id,
                visit_policy=visit_policy,
                root_value=root_value,
                raw_visit_counts=visit_policy.clone(),
                predicted_value=None,
                predicted_policy_logits=None,
                metadata={
                    "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                    "phase": "replay_critical_device",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(root_index.size),
                    "replay_payload_origin": (f"{self.search_impl}:{str(replay_payload_handle)}"),
                    "device_replay_payload": True,
                    "host_search_payload_fallback_allowed": False,
                },
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=DeviceTargetSearchService(),
        search_lane="unit_test_compact_muzero_learner_gate",
        policy_source="unit_test_compact_muzero_learner_gate",
        copy_root_observation=False,
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=4,
            warmup_steps=1,
            seed=2026052602,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=2,
            compact_rollout_slab_sample_gate_interval=1,
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_impl="compact_muzero",
            compact_rollout_slab_learner_gate_device="cpu",
            compact_rollout_slab_learner_gate_support_scale=1,
            compact_profile_cuda_sync_timing_diagnostics=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        compact_rollout_slab=slab,
    )

    last_sample = result["compact_rollout_slab_sample_gate_last_telemetry"]
    last_learner = result["compact_rollout_slab_learner_gate_last_telemetry"]
    assert result["compact_rollout_slab_learner_gate_impl"] == "compact_muzero"
    assert result["compact_rollout_slab_learner_gate_real_muzero_update"] is True
    assert result["compact_rollout_slab_learner_gate_toy_probe"] is False
    assert result["compact_rollout_slab_learner_gate_calls"] > 0
    for key in (
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
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_tensor_fallback_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_validate_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_accounted_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_residual_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_storage_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_builder_select_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_row_index_prepare_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_sec",
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
        "compact_rollout_slab_sample_gate_sample_batch_build_sec",
        "compact_rollout_slab_sample_gate_accounted_sec",
        "compact_rollout_slab_sample_gate_residual_sec",
    ):
        assert key in last_sample
        assert key in result
        assert result["timings"][key] == pytest.approx(result[key])
    stats = result["compact_rollout_slab_sample_gate_learner_batch_build_per_call_stats"]
    assert stats["count"] == result["compact_rollout_slab_sample_gate_calls"]
    assert stats["sum_sec"] == pytest.approx(
        result["compact_rollout_slab_sample_gate_learner_batch_build_sec"]
    )
    assert stats["max_sec"] >= stats["min_sec"]
    assert stats["slowest_call_index"] >= 1
    assert stats["slowest_iteration"] >= 1
    assert stats["slowest_measured_iteration"] >= 1
    sample_gate_stats = result["compact_rollout_slab_sample_gate_per_call_stats"]
    assert sample_gate_stats["count"] == result["compact_rollout_slab_sample_gate_calls"]
    assert sample_gate_stats["sum_sec"] == pytest.approx(
        result["compact_rollout_slab_sample_gate_sec"]
    )
    assert sample_gate_stats["max_sec"] >= sample_gate_stats["min_sec"]
    assert sample_gate_stats["slowest_iteration"] >= 1
    assert sample_gate_stats["slowest_measured_iteration"] >= 1
    trace_records = result["compact_rollout_slab_sample_gate_call_trace_records"]
    assert len(trace_records) == result["compact_rollout_slab_sample_gate_calls"]
    assert {record["measured_window_third"] for record in trace_records} <= {
        "early",
        "mid",
        "late",
        "warmup_or_unknown",
    }
    assert any(record["sample_row_count"] > 0 for record in trace_records)
    assert all(record["replay_ring_pair_capacity"] > 0 for record in trace_records)
    assert all(record["sample_gate_sec"] >= 0.0 for record in trace_records)
    assert any(record["stored_pair_count"] > 0 for record in trace_records)
    for record in trace_records:
        assert record["runtime_snapshot_diagnostics"] in (0, 1)
        assert record["cuda_memory_snapshot_enabled"] == 0
        assert record["cuda_memory_allocated_before_bytes"] == 0
        assert record["cuda_memory_allocated_after_bytes"] == 0
        assert record["cuda_memory_allocated_delta_bytes"] == 0
        assert record["learner_batch_build_cuda_memory_allocated_delta_bytes"] == 0
        assert record["python_gc_gen0_before"] >= 0
        assert record["python_gc_gen0_after"] >= 0
        for generation in range(3):
            assert record[f"python_gc_gen{generation}_collections_before"] >= 0
            assert record[f"python_gc_gen{generation}_collections_after"] >= 0
            assert record[f"python_gc_gen{generation}_collections_delta"] >= 0
            assert record[f"python_gc_gen{generation}_collected_delta"] >= 0
            assert record[f"python_gc_gen{generation}_uncollectable_delta"] >= 0
        assert isinstance(record["process_maxrss_after_raw"], int)
        assert record["process_cpu_time_delta_ns"] >= 0
        assert record["thread_cpu_time_delta_ns"] >= 0
        assert record["learner_batch_build_process_cpu_time_delta_ns"] >= 0
        assert record["learner_batch_build_thread_cpu_time_delta_ns"] >= 0
        for key in (
            "terminal_final_observation_group_count",
            "terminal_final_observation_index_fast_path_count",
            "terminal_final_observation_fallback_count",
            "terminal_final_observation_validate_only_count",
            "terminal_final_observation_materialized_count",
            "terminal_final_observation_final_row_count_sum",
            "terminal_final_observation_final_row_count_max",
            "terminal_final_observation_dense_storage_count",
            "terminal_final_observation_sparse_storage_count",
            "terminal_final_observation_missing_storage_count",
            "terminal_final_observation_sparse_row_count_sum",
            "terminal_final_observation_sparse_row_count_max",
        ):
            assert record[key] >= 0
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
        ):
            for scope in ("process", "thread"):
                assert record[f"builder_{child_name}_{scope}_cpu_time_delta_ns"] >= 0
                assert (
                    result[
                        (
                            "compact_rollout_slab_sample_gate_learner_batch_builder_"
                            f"{child_name}_{scope}_cpu_time_delta_ns"
                        )
                    ]
                    >= 0
                )
        for resource_prefix in (
            "process_resource",
            "thread_resource",
            "learner_batch_build_process_resource",
            "learner_batch_build_thread_resource",
        ):
            for field_name in (
                "user_cpu_time_delta_ns",
                "system_cpu_time_delta_ns",
                "minor_page_faults_delta",
                "major_page_faults_delta",
                "voluntary_context_switches_delta",
                "involuntary_context_switches_delta",
            ):
                assert record[f"{resource_prefix}_{field_name}"] >= 0
        for key in (
            "builder_group_loop_prepare_sec",
            "builder_group_loop_prepare_accounted_sec",
            "builder_group_loop_prepare_residual_sec",
            "builder_group_loop_prepare_snapshot_sec",
            "builder_group_loop_prepare_index_sec",
            "builder_group_loop_prepare_observation_sec",
            "builder_group_loop_terminal_value_bookkeeping_sec",
            "builder_terminal_metadata_accounted_sec",
            "builder_terminal_metadata_residual_sec",
            "builder_terminal_metadata_mask_sec",
            "builder_terminal_metadata_tensor_fallback_sec",
            "builder_terminal_metadata_validate_sec",
            "builder_terminal_metadata_final_observation_sec",
            "builder_terminal_metadata_final_observation_accounted_sec",
            "builder_terminal_metadata_final_observation_residual_sec",
            "builder_terminal_metadata_final_observation_presence_sec",
            "builder_terminal_metadata_final_observation_select_current_sec",
            "builder_terminal_metadata_final_observation_gather_sec",
            "builder_terminal_metadata_final_observation_storage_sec",
            "builder_terminal_metadata_final_observation_validate_sec",
            "builder_unroll_fields_accounted_sec",
            "builder_unroll_fields_residual_sec",
            "builder_unroll_builder_select_sec",
            "builder_unroll_row_index_prepare_sec",
            "builder_unroll_terminal_window_hint_sec",
            "builder_unroll_identity_sec",
            "builder_unroll_stack_fields_sec",
            "builder_unroll_mask_build_sec",
            "builder_unroll_terminal_value_sec",
            "builder_unroll_mask_apply_sec",
            "builder_unroll_action_stack_sec",
            "builder_order_restore_sec",
            "builder_finalize_outputs_sec",
            "builder_metadata_sync_sec",
            "builder_metadata_build_sec",
            "sample_gate_cuda_sync_sec",
        ):
            assert key in record
    for key in (
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
            "terminal_final_observation_sparse_row_count_max"
        ),
    ):
        assert last_sample[key] >= 0
    assert (
        last_sample[
            "compact_rollout_slab_sample_gate_"
            "terminal_final_observation_materialized_count"
        ]
        == 0
    )
    assert (
        last_sample[
            "compact_rollout_slab_sample_gate_"
            "terminal_final_observation_validate_only_count"
        ]
        == last_sample[
            "compact_rollout_slab_sample_gate_terminal_final_observation_fallback_count"
        ]
    )
    for stats_key, total_key in (
        (
            "compact_rollout_slab_sample_gate_candidate_per_call_stats",
            "compact_rollout_slab_sample_gate_candidate_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_rng_per_call_stats",
            "compact_rollout_slab_sample_gate_rng_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_residual_per_call_stats",
            "compact_rollout_slab_sample_gate_residual_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_accounted_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_accounted_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_residual_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_residual_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_snapshot_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_snapshot_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_index_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_index_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_observation_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_observation_sec",
        ),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "group_loop_terminal_value_bookkeeping_per_call_stats"
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "group_loop_terminal_value_bookkeeping_sec"
            ),
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_tensor_fallback_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_tensor_fallback_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_validate_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_validate_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_sec",
        ),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_accounted_per_call_stats"
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_accounted_sec"
            ),
        ),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_residual_per_call_stats"
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_residual_sec"
            ),
        ),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_presence_per_call_stats"
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_presence_sec"
            ),
        ),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_select_current_per_call_stats"
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_select_current_sec"
            ),
        ),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_gather_per_call_stats"
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_gather_sec"
            ),
        ),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_storage_per_call_stats"
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_storage_sec"
            ),
        ),
        (
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_validate_per_call_stats"
            ),
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "terminal_metadata_final_observation_validate_sec"
            ),
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_builder_select_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_builder_select_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_row_index_prepare_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_row_index_prepare_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_identity_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_identity_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_build_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_build_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_value_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_value_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_apply_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_apply_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_action_stack_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_action_stack_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_write_output_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_write_output_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_sec",
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_per_call_stats",
            "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec",
        ),
    ):
        child_stats = result[stats_key]
        assert child_stats["count"] == result["compact_rollout_slab_sample_gate_calls"]
        assert child_stats["sum_sec"] == pytest.approx(result[total_key])
        assert child_stats["max_sec"] >= child_stats["min_sec"]
    runtime_stats = result["compact_profile_runtime_step_timing_stats"]
    assert result["compact_profile_runtime_step_timing_diagnostics"] is True
    assert runtime_stats["count"] == 4
    assert runtime_stats["sum_sec"] > 0.0
    assert runtime_stats["max_sec"] >= runtime_stats["min_sec"]
    assert runtime_stats["p95_sec"] >= runtime_stats["p50_sec"]
    assert 1 <= runtime_stats["slowest_measured_iteration"] <= 4
    assert runtime_stats["slowest_iteration"] >= 1
    assert runtime_stats["slowest_actor_step_wall_sec"] >= 0.0
    assert runtime_stats["slowest_observation_sec"] >= 0.0
    assert runtime_stats["slowest_compact_rollout_slab_sec"] >= 0.0
    assert runtime_stats["slowest_sample_gate_sec"] >= 0.0
    assert runtime_stats["slowest_learner_gate_sec"] >= 0.0
    assert runtime_stats["slowest_policy_refresh_sec"] >= 0.0
    assert runtime_stats["slowest_primary_accounted_sec"] >= 0.0
    assert (
        runtime_stats["sample_gate_active_count"]
        + runtime_stats["sample_gate_inactive_count"]
        == runtime_stats["count"]
    )
    assert (
        runtime_stats["early_count"]
        + runtime_stats["mid_count"]
        + runtime_stats["late_count"]
        == runtime_stats["count"]
    )
    assert (
        runtime_stats["early_sum_sec"]
        + runtime_stats["mid_sum_sec"]
        + runtime_stats["late_sum_sec"]
    ) == pytest.approx(runtime_stats["sum_sec"])
    assert (
        runtime_stats["early_sample_gate_sum_sec"]
        + runtime_stats["mid_sample_gate_sum_sec"]
        + runtime_stats["late_sample_gate_sum_sec"]
    ) == pytest.approx(runtime_stats["sample_gate_sum_sec"])
    assert runtime_stats["sample_gate_active_sample_gate_count"] == runtime_stats[
        "sample_gate_active_count"
    ]
    assert runtime_stats["sample_gate_active_sample_gate_sum_sec"] == pytest.approx(
        runtime_stats["sample_gate_sum_sec"]
    )
    for bucket_name in ("early", "mid", "late"):
        active_prefix = f"{bucket_name}_sample_gate_active"
        assert runtime_stats[f"{active_prefix}_sample_gate_count"] == runtime_stats[
            f"{bucket_name}_sample_gate_active_count"
        ]
        assert runtime_stats[f"{active_prefix}_sample_gate_sum_sec"] == pytest.approx(
            runtime_stats[f"{bucket_name}_sample_gate_sum_sec"]
        )
        assert runtime_stats[
            f"{active_prefix}_sample_gate_builder_group_loop_sum_sec"
        ] == pytest.approx(
            runtime_stats[f"{bucket_name}_sample_gate_builder_group_loop_sum_sec"]
        )
        assert runtime_stats[f"{active_prefix}_sample_gate_p95_sec"] >= runtime_stats[
            f"{active_prefix}_sample_gate_p50_sec"
        ]
        assert runtime_stats[
            f"{active_prefix}_sample_gate_builder_group_loop_p95_sec"
        ] >= runtime_stats[f"{active_prefix}_sample_gate_builder_group_loop_p50_sec"]
    assert len(runtime_stats["top_slowest_records"]) <= 8
    assert runtime_stats["top_slowest_records"][0]["sec"] == pytest.approx(
        runtime_stats["max_sec"]
    )
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
    ):
        assert runtime_stats[f"{phase_name}_sum_sec"] >= 0.0
        assert runtime_stats[f"{phase_name}_max_sec"] >= runtime_stats[
            f"{phase_name}_min_sec"
        ]
        assert runtime_stats[f"{phase_name}_p95_sec"] >= runtime_stats[
            f"{phase_name}_p50_sec"
        ]
    assert last_learner["compact_rollout_slab_learner_gate_impl"] == "compact_muzero"
    assert last_learner["compact_rollout_slab_learner_gate_real_muzero_update"] is True
    assert last_learner["compact_rollout_slab_learner_gate_resident_sample_used"] is True
    assert last_learner["compact_rollout_slab_learner_gate_device_replay_index_rows_sample"] is True
    assert last_learner["compact_rollout_slab_learner_gate_input_h2d_bytes"] == 0
    for key in (
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
        "compact_rollout_slab_learner_gate_accounted_sec",
        "compact_rollout_slab_learner_gate_residual_sec",
    ):
        assert key in last_learner
        assert key in result
        assert result["timings"][key] == pytest.approx(result[key])
    compact_telemetry = last_learner["compact_rollout_slab_learner_gate_compact_muzero_telemetry"]
    assert compact_telemetry["compact_muzero_learner_target_mode"] == "explicit_next_targets"
    assert compact_telemetry["compact_muzero_learner_next_action_mask_present"] is True
    assert compact_telemetry["compact_muzero_learner_accounted_sec"] > 0.0


def test_runtime_step_timing_diagnostics_can_run_without_cuda_sync():
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=3,
            warmup_steps=1,
            seed=2026060301,
            compact_profile_runtime_step_timing_diagnostics=True,
            compact_profile_cuda_sync_timing_diagnostics=False,
        )
    )

    runtime_stats = result["compact_profile_runtime_step_timing_stats"]
    assert result["compact_profile_cuda_sync_timing_diagnostics"] is False
    assert result["compact_profile_runtime_step_timing_diagnostics"] is True
    assert runtime_stats["count"] == 3
    assert runtime_stats["sum_sec"] > 0.0
    assert runtime_stats["max_sec"] >= runtime_stats["min_sec"]
    assert (
        runtime_stats["sample_gate_active_count"]
        + runtime_stats["sample_gate_inactive_count"]
        == runtime_stats["count"]
    )
    assert (
        runtime_stats["early_count"]
        + runtime_stats["mid_count"]
        + runtime_stats["late_count"]
        == runtime_stats["count"]
    )
    assert (
        runtime_stats["early_sum_sec"]
        + runtime_stats["mid_sum_sec"]
        + runtime_stats["late_sum_sec"]
    ) == pytest.approx(runtime_stats["sum_sec"])
    assert len(runtime_stats["top_slowest_records"]) == 3
    assert runtime_stats["top_slowest_records"][0]["sec"] == pytest.approx(
        runtime_stats["max_sec"]
    )
    for record in runtime_stats["top_slowest_records"]:
        assert 1 <= record["measured_iteration"] <= 3
        assert "sample_gate_call_index" in record
        assert "sample_gate_builder_group_loop_sec" in record


def test_compact_owned_loop_owner_ref_refresh_requires_explicit_search_hook():
    owner_ref = {
        "schema_id": "curvyzero_compact_owned_loop_model_owner_ref/v1",
        "transport_kind": COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
        "request_id": 7,
        "worker_pid": 123,
        "model_state_digest": "owner-digest-7",
        "model_object_id": 456,
    }
    slab_without_hook = SimpleNamespace(search_service=SimpleNamespace())
    with pytest.raises(ValueError, match="refresh_model_owner_ref"):
        _refresh_compact_rollout_slab_search_from_owner_ref(
            compact_rollout_slab=slab_without_hook,
            owner_ref=owner_ref,
            policy_version_ref="policy:learner_update_7",
            model_version_ref="model:learner_update_7",
            policy_source="unit_test_owner_ref",
            learner_update_count=7,
            expected_model_state_digest="owner-digest-7",
        )

    class OwnerRefAwareSearchService:
        def __init__(self):
            self.calls = []

        def refresh_model_owner_ref(
            self,
            *,
            owner_ref,
            policy_version_ref,
            model_version_ref,
            policy_source,
            learner_update_count,
            expected_model_state_digest=None,
        ):
            self.calls.append(
                {
                    "owner_ref": dict(owner_ref),
                    "policy_version_ref": str(policy_version_ref),
                    "model_version_ref": str(model_version_ref),
                    "policy_source": str(policy_source),
                    "learner_update_count": int(learner_update_count),
                    "expected_model_state_digest": expected_model_state_digest,
                }
            )
            return {
                "schema_id": "curvyzero_compact_policy_refresh_search_worker_state/v1",
                "search_impl": "unit_test_owner_ref_search",
                "policy_version_ref": str(policy_version_ref),
                "model_version_ref": str(model_version_ref),
                "policy_source": str(policy_source),
                "learner_update_count": int(learner_update_count),
                "model_state_digest": str(expected_model_state_digest),
                "search_worker_model_object_id": 999,
                "refresh_count": 1,
                "refresh_applied": True,
                "cache_cleared": True,
                "refresh_total_sec": 0.001,
                "state_load_sec": 0.0,
                "model_state_digest_sec": 0.0,
            }

    search_service = OwnerRefAwareSearchService()
    refresh = _refresh_compact_rollout_slab_search_from_owner_ref(
        compact_rollout_slab=SimpleNamespace(search_service=search_service),
        owner_ref=owner_ref,
        policy_version_ref="policy:learner_update_7",
        model_version_ref="model:learner_update_7",
        policy_source="unit_test_owner_ref",
        learner_update_count=7,
        expected_model_state_digest="owner-digest-7",
    )

    assert refresh["learner_update_count"] == 7
    assert refresh["model_state_digest"] == "owner-digest-7"
    assert refresh["learner_model_object_id"] == 456
    assert refresh["search_worker_model_object_id"] == 999
    assert search_service.calls == [
        {
            "owner_ref": owner_ref,
            "policy_version_ref": "policy:learner_update_7",
            "model_version_ref": "model:learner_update_7",
            "policy_source": "unit_test_owner_ref",
            "learner_update_count": 7,
            "expected_model_state_digest": "owner-digest-7",
        }
    ]


@pytest.mark.parametrize(
    "defer_mode",
    [
        "sync",
        "defer_learner",
        "defer_sample_learner",
        "defer_sample_learner_owner_ref",
        "defer_sample_learner_local_process_snapshot",
    ],
)
def test_compact_owned_loop_refreshes_separate_search_worker_after_muzero_update(
    defer_mode: str,
):
    torch = pytest.importorskip("torch")
    defer_learner = defer_mode == "defer_learner"
    defer_sample_learner = defer_mode in {
        "defer_sample_learner",
        "defer_sample_learner_owner_ref",
        "defer_sample_learner_local_process_snapshot",
    }
    owner_ref_only = defer_mode == "defer_sample_learner_owner_ref"
    local_process_snapshot = defer_mode == "defer_sample_learner_local_process_snapshot"
    assert _TinyMuZeroForRefreshTest is not None

    class RefreshAwareSearchService:
        supports_two_phase_compact_search = True
        search_impl = "unit_test_refresh_aware_compact_search"
        num_simulations = 1

        def __init__(self, *, owner_ref_only: bool = False):
            self.owner_ref_only = bool(owner_ref_only)
            self._model = _TinyMuZeroForRefreshTest()
            self.counter = 0
            self.pending = {}
            self.refresh_calls = []
            self.root_metadata = []
            self._policy_version_ref = "initial-policy"
            self._model_version_ref = "initial-model"
            self._policy_source = "unit_test_refresh_aware_compact_search"
            self._refresh_count = 0
            self._learner_update_count = 0
            self._model_state_digest = ""
            self.refresh_expected_model_state_digests = []

        def __getattribute__(self, name):
            if name == "refresh_model_state":
                owner_ref_only = object.__getattribute__(self, "owner_ref_only")
                if bool(owner_ref_only):
                    raise AttributeError(name)
            return object.__getattribute__(self, name)

        def refresh_model_state(
            self,
            *,
            model_state_dict,
            policy_version_ref,
            model_version_ref,
            policy_source,
            learner_update_count,
            expected_model_state_digest=None,
        ):
            return self._refresh_model_state_impl(
                model_state_dict=model_state_dict,
                policy_version_ref=policy_version_ref,
                model_version_ref=model_version_ref,
                policy_source=policy_source,
                learner_update_count=learner_update_count,
                expected_model_state_digest=expected_model_state_digest,
            )

        def refresh_model_owner_ref(
            self,
            *,
            owner_ref,
            policy_version_ref,
            model_version_ref,
            policy_source,
            learner_update_count,
            expected_model_state_digest=None,
        ):
            assert owner_ref["transport_kind"] == COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1
            self.refresh_expected_model_state_digests.append(expected_model_state_digest)
            digest = str(expected_model_state_digest or owner_ref["model_state_digest"])
            assert digest
            self._policy_version_ref = str(policy_version_ref)
            self._model_version_ref = str(model_version_ref)
            self._policy_source = str(policy_source)
            self._learner_update_count = int(learner_update_count)
            self._refresh_count += 1
            self._model_state_digest = digest
            state = self.policy_refresh_search_worker_state()
            self.refresh_calls.append(dict(state))
            return state

        def _refresh_model_state_impl(
            self,
            *,
            model_state_dict,
            policy_version_ref,
            model_version_ref,
            policy_source,
            learner_update_count,
            expected_model_state_digest=None,
        ):
            self.refresh_expected_model_state_digests.append(expected_model_state_digest)
            self._model.load_state_dict(dict(model_state_dict))
            digest = compact_model_state_digest_v1(self._model)
            if expected_model_state_digest is not None:
                assert digest == expected_model_state_digest
            self._policy_version_ref = str(policy_version_ref)
            self._model_version_ref = str(model_version_ref)
            self._policy_source = str(policy_source)
            self._learner_update_count = int(learner_update_count)
            self._refresh_count += 1
            self._model_state_digest = digest
            state = self.policy_refresh_search_worker_state()
            self.refresh_calls.append(dict(state))
            return state

        def policy_refresh_search_worker_state(self):
            digest = self._model_state_digest or compact_model_state_digest_v1(self._model)
            return {
                "schema_id": "curvyzero_compact_policy_refresh_search_worker_state/v1",
                "search_impl": self.search_impl,
                "policy_version_ref": self._policy_version_ref,
                "model_version_ref": self._model_version_ref,
                "policy_source": self._policy_source,
                "learner_update_count": int(self._learner_update_count),
                "model_state_digest": digest,
                "search_worker_model_object_id": int(id(self._model)),
                "search_worker_object_id": int(id(self)),
                "refresh_count": int(self._refresh_count),
                "refresh_applied": bool(self._refresh_count > 0),
                "cache_cleared": bool(self._refresh_count > 0),
                "refresh_total_sec": 0.001,
                "state_load_sec": 0.0004,
                "model_state_digest_sec": 0.0006,
                "total_state_load_sec": float(self._refresh_count) * 0.0004,
                "total_model_state_digest_sec": float(self._refresh_count) * 0.0006,
                "model_state_digest_source": "unit_test_after_load",
                "calls_train_muzero": False,
                "touches_live_runs": False,
            }

        def _refresh_metadata(self):
            if self._refresh_count <= 0:
                return {}
            return compact_policy_refresh_metadata_from_state_v1(
                self.policy_refresh_search_worker_state()
            )

        def run_action_step(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = np.asarray(
                [self.counter % ACTION_COUNT for _ in range(active_roots.size)],
                dtype=np.int16,
            )
            handle = f"refresh-aware:{self.counter}"
            self.counter += 1
            self.root_metadata.append(dict(root_batch.metadata))
            self.pending[handle] = (
                active_roots,
                root_batch.env_row[active_roots].astype(np.int32, copy=True),
                root_batch.player[active_roots].astype(np.int16, copy=True),
                root_batch.policy_env_id[active_roots].astype(np.int64, copy=True),
                selected.copy(),
                self._refresh_metadata(),
            )
            return CompactSearchActionStepV1(
                replay_payload_handle=handle,
                root_index=active_roots,
                env_row=self.pending[handle][1],
                player=self.pending[handle][2],
                policy_env_id=self.pending[handle][3],
                selected_action=selected,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(active_roots.size),
                    "replay_payload_origin": f"{self.search_impl}:{handle}",
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                    "search_replay_payload_digest": (
                        compact_search_deferred_replay_payload_digest_v1(handle)
                    ),
                    "search_replay_payload_digest_deferred": True,
                    **self._refresh_metadata(),
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"unexpected host replay flush: {replay_payload_handle}")

        def flush_device_replay_payload(self, replay_payload_handle):
            root_index, env_row, player, policy_env_id, selected, refresh_metadata = (
                self.pending.pop(replay_payload_handle)
            )
            visit_policy = torch.zeros((root_index.size, ACTION_COUNT), dtype=torch.float32)
            visit_policy[
                torch.arange(root_index.size),
                torch.as_tensor(selected, dtype=torch.long),
            ] = 1.0
            return CompactDeviceSearchReplayPayloadV1(
                replay_payload_handle=str(replay_payload_handle),
                root_index=root_index,
                env_row=env_row,
                player=player,
                policy_env_id=policy_env_id,
                visit_policy=visit_policy,
                root_value=torch.zeros((root_index.size,), dtype=torch.float32),
                raw_visit_counts=visit_policy.clone(),
                predicted_value=None,
                predicted_policy_logits=None,
                metadata={
                    "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                    "phase": "replay_critical_device",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(root_index.size),
                    "replay_payload_origin": (f"{self.search_impl}:{str(replay_payload_handle)}"),
                    "device_replay_payload": True,
                    "host_search_payload_fallback_allowed": False,
                    **refresh_metadata,
                },
            )

    learner_model = _TinyMuZeroForRefreshTest()
    search_service = RefreshAwareSearchService(owner_ref_only=owner_ref_only)
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=search_service,
        search_lane="unit_test_refresh_aware_compact_search",
        policy_source="unit_test_refresh_aware_compact_search",
        copy_root_observation=False,
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=8,
            warmup_steps=1,
            seed=2026053101,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=2,
            compact_rollout_slab_sample_gate_interval=3,
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_impl="compact_muzero",
            compact_rollout_slab_learner_gate_device="cpu",
            compact_owned_loop_entrypoint=True,
            compact_owned_loop_policy_version_ref="unit-policy",
            compact_owned_loop_model_version_ref="unit-model",
            compact_owned_loop_policy_source="unit-policy-source",
            compact_owned_loop_defer_learner_gate=defer_learner,
            compact_owned_loop_defer_sample_learner_gate=defer_sample_learner,
            compact_owned_loop_sample_learner_worker_kind=(
                COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS
                if local_process_snapshot
                else "in_process_thread"
            ),
            compact_owned_loop_defer_sample_learner_model_state_transport_kind=(
                COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1
                if local_process_snapshot
                else COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1
                if owner_ref_only
                else "result_v1"
            ),
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        compact_rollout_slab=slab,
        compact_rollout_slab_learner_model=learner_model,
        compact_rollout_slab_refresh_search_after_learner_gate=True,
        compact_rollout_slab_refresh_search_after_learner_gate_interval=2,
    )

    learner_updates = result["compact_rollout_slab_learner_gate_updates"]
    assert learner_updates > 0
    assert result["compact_owned_loop_defer_learner_gate"] is defer_learner
    assert result["compact_owned_loop_defer_sample_learner_gate"] is defer_sample_learner
    if defer_learner:
        assert result["compact_owned_loop_deferred_learner_submit_count"] > 0
        assert (
            result["compact_owned_loop_deferred_learner_completed_count"]
            == (result["compact_owned_loop_deferred_learner_submit_count"])
        )
        assert result["compact_owned_loop_deferred_learner_pending"] is False
    if defer_sample_learner:
        assert result["compact_owned_loop_deferred_sample_learner_submit_count"] > 0
        assert (
            result["compact_owned_loop_deferred_sample_learner_completed_count"]
            == result["compact_owned_loop_deferred_sample_learner_submit_count"]
        )
        assert result["compact_owned_loop_deferred_sample_learner_pending"] is False
        assert result["compact_owned_loop_deferred_sample_learner_pending_count"] == 0
        assert result["compact_owned_loop_deferred_sample_learner_drained"] is True
        assert result["compact_owned_loop_final_deferred_drain_in_measured_sec"] is True
        assert result["compact_rollout_slab_sample_gate_calls"] == (
            result["compact_owned_loop_deferred_sample_learner_completed_count"]
        )
    if local_process_snapshot:
        assert (
            result["compact_owned_loop_deferred_sample_learner_model_state_transport_kind"]
            == COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1
        )
        assert (
            result[
                "compact_owned_loop_deferred_sample_learner_model_state_snapshot_return_count"
            ]
            > 0
        )
        assert (
            result[
                "compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_count"
            ]
            > 0
        )
    assert search_service._model is not learner_model
    assert search_service.refresh_calls
    assert search_service.refresh_calls[-1]["learner_update_count"] == learner_updates
    assert search_service.refresh_calls[-1]["model_state_digest"] == (
        compact_model_state_digest_v1(learner_model)
    )
    if defer_learner or defer_sample_learner:
        assert all(
            isinstance(value, str) and value
            for value in search_service.refresh_expected_model_state_digests
        )
    else:
        assert all(value is None for value in search_service.refresh_expected_model_state_digests)
    assert result["compact_rollout_slab_policy_refresh_after_learner_gate_enabled"] is True
    assert result["compact_rollout_slab_policy_refresh_after_learner_gate_interval"] == 2
    assert result["compact_rollout_slab_policy_refresh_after_learner_gate_calls"] > 0
    if owner_ref_only:
        assert (
            result["compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind"]
            == COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1
        )
        assert (
            result["compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind"]
            == COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1
        )
        assert (
            result[
                "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count"
            ]
            == result["compact_rollout_slab_policy_refresh_after_learner_gate_calls"]
        )
        assert (
            result[
                "compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count"
            ]
            == 0
        )
        assert (
            result[
                (
                    "compact_rollout_slab_policy_refresh_after_learner_gate_"
                    "parent_model_state_transport_avoided"
                )
            ]
            is True
        )
    else:
        assert (
            result["compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count"]
            == 0
        )
        assert (
            result["compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count"]
            == result["compact_rollout_slab_policy_refresh_after_learner_gate_calls"]
        )
    if defer_learner or defer_sample_learner:
        assert result["compact_rollout_slab_policy_refresh_after_learner_gate_calls"] <= (
            learner_updates
        )
    else:
        assert result["compact_rollout_slab_policy_refresh_after_learner_gate_calls"] < (
            learner_updates
        )
        assert result["compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count"] > 0
    assert result["compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count"] >= 0
    assert (
        result["compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count"]
        == learner_updates
    )
    assert (
        result[
            "compact_rollout_slab_policy_refresh_after_learner_gate_search_worker_distinct_from_learner"
        ]
        is True
    )
    assert result["compact_rollout_slab_policy_refresh_after_learner_gate_service_total_sec"] > 0.0
    assert result["compact_rollout_slab_policy_refresh_after_learner_gate_state_load_sec"] > 0.0
    assert result["compact_rollout_slab_policy_refresh_after_learner_gate_model_digest_sec"] > 0.0
    assert result["compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_count"]
    assert result["compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_count"]
    assert any(
        metadata.get("compact_policy_refresh_search_worker_refreshed") is True
        for metadata in search_service.root_metadata
    )
    search_metadata = result[
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata"
    ]
    replay_metadata = result[
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata"
    ]
    assert search_metadata["compact_policy_refresh_learner_update_count"] == learner_updates
    assert replay_metadata["compact_policy_refresh_learner_update_count"] == learner_updates


def test_compact_muzero_learner_gate_finds_model_through_profile_probe_wrapper():
    model = object()
    wrapped = SimpleNamespace(
        search_service=SimpleNamespace(
            _probe=SimpleNamespace(_policy=SimpleNamespace(_model=model))
        )
    )

    assert _compact_rollout_slab_search_model(wrapped) is model


def test_hybrid_profile_scripted_slab_actions_hold_env_trajectory_fixed():
    class FakeSearchService:
        num_simulations = 0

        def __init__(self, *, search_impl: str, select_last_legal: bool) -> None:
            self.search_impl = search_impl
            self.select_last_legal = bool(select_last_legal)

        def run(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            legal = root_batch.legal_mask[active_roots]
            if self.select_last_legal:
                selected = (ACTION_COUNT - 1 - np.argmax(legal[:, ::-1], axis=1)).astype(
                    np.int16, copy=False
                )
            else:
                selected = np.argmax(legal, axis=1).astype(np.int16, copy=False)
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
            )

    def run_one(*, select_last_legal: bool) -> dict[str, object]:
        slab = CompactRolloutSlab(
            batch_size=4,
            player_count=2,
            search_service=FakeSearchService(
                search_impl=f"unit_test_scripted_{select_last_legal}",
                select_last_legal=select_last_legal,
            ),
            search_lane=f"unit_test_scripted_{select_last_legal}",
            policy_source=f"unit_test_scripted_{select_last_legal}",
            action_feedback_mode=COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM,
        )
        return run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=4,
                actor_count=1,
                steps=3,
                warmup_steps=1,
                seed=20260523,
                materialize_scalar_timestep=False,
                compact_rollout_slab_action_mode=(COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM),
            ),
            compact_rollout_slab=slab,
        )

    first = run_one(select_last_legal=False)
    second = run_one(select_last_legal=True)

    assert first["env_action_checksum_total"] == second["env_action_checksum_total"]
    assert first["last_env_action_checksum"] == second["last_env_action_checksum"]
    assert first["env_trajectory_checksum_total"] == second["env_trajectory_checksum_total"]
    assert first["last_env_trajectory_checksum"] == second["last_env_trajectory_checksum"]
    assert first["compact_rollout_slab_action_mode"] == (
        COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM
    )
    assert second["compact_rollout_slab_action_mode"] == (
        COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM
    )
    assert first["compact_rollout_slab_action_override_drop_count"] > 0
    assert second["compact_rollout_slab_action_override_drop_count"] > 0
    assert first["compact_rollout_slab_committed_index_row_count"] == 0
    assert second["compact_rollout_slab_committed_index_row_count"] == 0
    assert first["contract"]["compact_rollout_slab_action_mode"] == (
        COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM
    )


def test_hybrid_profile_reports_bounded_slab_history_counters_without_retention():
    class FakeSearchService:
        num_simulations = 0

        def run(self, root_batch):
            active_roots = np.flatnonzero(root_batch.active_root_mask)
            legal = root_batch.legal_mask[active_roots]
            selected = np.argmax(legal, axis=1).astype(np.int16, copy=False)
            visit_policy = np.zeros((active_roots.size, ACTION_COUNT), dtype=np.float32)
            visit_policy[np.arange(active_roots.size), selected] = 1.0
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected,
                visit_policy=visit_policy,
                root_value=np.zeros((active_roots.size,), dtype=np.float32),
                search_impl="unit_test_bounded_slab_history",
                num_simulations=self.num_simulations,
            )

    slab = CompactRolloutSlab(
        batch_size=4,
        player_count=2,
        search_service=FakeSearchService(),
        search_lane="unit_test_bounded_slab_history",
        policy_source="unit_test_bounded_slab_history",
        retain_committed_index_rows=False,
    )

    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=4,
            actor_count=1,
            steps=3,
            warmup_steps=0,
            seed=20260602,
            materialize_scalar_timestep=False,
        ),
        compact_rollout_slab=slab,
    )

    assert result["compact_rollout_slab_committed_index_row_count"] > 0
    assert result["compact_rollout_slab_stored_index_group_count"] > 0
    assert result["compact_rollout_slab_stored_index_row_count"] == (
        result["compact_rollout_slab_committed_index_row_count"]
    )
    assert result["compact_rollout_slab_retains_committed_index_rows"] is False
    assert slab.committed_index_rows == ()


def test_hybrid_profile_metadata_stays_profile_only_and_does_not_touch_trainer_defaults():
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(batch_size=2, actor_count=1, steps=1, warmup_steps=0)
    )
    contract = result["contract"]

    assert result["profile_only"] is True
    assert result["calls_train_muzero"] is False
    assert result["stock_lightzero_integrated"] is False
    assert result["trainer_defaults_changed"] is False
    assert result["touches_live_runs"] is False
    assert contract["profile_only"] is True
    assert contract["calls_train_muzero"] is False
    assert contract["trainer_defaults_changed"] is False
    assert contract["future_gpu_render_boundary"]["not_implemented_here"] is True


def test_hybrid_profile_reports_timing_and_byte_fields():
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(batch_size=4, actor_count=2, steps=2, warmup_steps=0)
    )

    assert set(HYBRID_OBSERVATION_TIMING_FIELDS) <= set(result["timings"])
    for field in HYBRID_OBSERVATION_TIMING_FIELDS:
        assert isinstance(result["timings"][field], float)
        assert result["timings"][field] >= 0.0
        assert field in result["timing_per_timestep_sec"]
    assert result["timings"]["actor_step_sec"] > 0.0
    assert result["timings"]["actor_env_runtime_sec"] > 0.0
    assert result["timings"]["actor_env_runtime_step_many_sec"] > 0.0
    assert result["timings"]["actor_env_runtime_outer_bookkeeping_sec"] >= 0.0
    assert result["timings"]["actor_env_runtime_natural_bonus_spawn_sec"] >= 0.0
    assert result["timings"]["actor_env_runtime_context_build_sec"] >= 0.0
    assert result["timings"]["actor_env_runtime_phase_accounted_sec"] > 0.0
    assert result["timings"]["actor_env_runtime_phase_residual_sec"] >= 0.0
    assert result["timings"]["actor_env_runtime_movement_sec"] > 0.0
    assert (
        result["timings"]["actor_env_runtime_phase_accounted_sec"]
        >= result["timings"]["actor_env_runtime_movement_sec"]
    )
    assert result["timings"]["actor_env_batch_pack_sec"] > 0.0
    assert result["timings"]["actor_autoreset_sec"] >= 0.0
    assert result["timings"]["gather_merge_sec"] >= 0.0
    assert result["timings"]["observation_sec"] >= 0.0
    assert result["timings"]["scalar_materialization_sec"] >= 0.0
    assert result["timings"]["policy_search_probe_sec"] == 0.0
    assert result["timings"]["batched_stack_probe_sec"] == 0.0
    assert result["compact_payload_bytes_total"] > 0
    assert result["compact_payload_bytes_per_step"] > 0.0
    assert result["rendered_stack_bytes_per_step"] == 4 * 2 * 4 * 64 * 64 * 4
    assert result["compact_vs_rendered_stack_ratio"] < 1.0
    assert result["steps_per_sec"] is not None
    assert result["measured_sec"] > 0.0
    assert result["total_sec"] >= result["measured_sec"]
    assert result["warmup_sec"] >= 0.0


def test_native_actor_buffer_matches_payload_merge_for_core_arrays():
    action = np.asarray(
        [
            [0, 1],
            [2, 0],
            [1, 2],
            [0, 0],
        ],
        dtype=np.int16,
    )
    baseline = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(batch_size=4, actor_count=2, seed=41)
    )
    native = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=4,
            actor_count=2,
            seed=41,
            native_actor_buffer=True,
        )
    )

    baseline_step = baseline.step(action)
    native_step = native.step(action)

    np.testing.assert_array_equal(native_step.policy_env_id, baseline_step.policy_env_id)
    np.testing.assert_array_equal(native_step.policy_env_row, baseline_step.policy_env_row)
    np.testing.assert_array_equal(native_step.policy_player, baseline_step.policy_player)
    np.testing.assert_array_equal(native_step.reward, baseline_step.reward)
    np.testing.assert_array_equal(native_step.done, baseline_step.done)
    np.testing.assert_array_equal(native_step.action_mask, baseline_step.action_mask)
    np.testing.assert_array_equal(
        native_step.timestep.obs["observation"],
        baseline_step.timestep.obs["observation"],
    )
    np.testing.assert_array_equal(
        native_step.timestep.obs["action_mask"],
        baseline_step.timestep.obs["action_mask"],
    )
    assert native_step.timestep.info == baseline_step.timestep.info
    assert native_step.timings["gather_merge_sec"] <= baseline_step.timings["gather_merge_sec"]


def test_native_actor_buffer_bypasses_public_env_packaging_for_nonterminal_step():
    action = np.asarray([[0, 1], [2, 0]], dtype=np.int16)
    baseline = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(batch_size=2, actor_count=1, seed=2026052603)
    )
    native = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=2026052603,
            native_actor_buffer=True,
        )
    )
    baseline_step = baseline.step(action)

    def fail_public_packaging(*_args, **_kwargs):
        raise AssertionError("compact actor path touched public env packaging")

    actor = native.actors[0]
    actor.env._public_info = fail_public_packaging
    actor.env._batch = fail_public_packaging
    actor.env._observe_array = fail_public_packaging

    native_step = native.step(action)

    np.testing.assert_array_equal(native_step.reward, baseline_step.reward)
    np.testing.assert_array_equal(native_step.done, baseline_step.done)
    np.testing.assert_array_equal(native_step.action_mask, baseline_step.action_mask)
    np.testing.assert_array_equal(
        native_step.payload["episode_step"],
        baseline_step.payload["episode_step"],
    )
    np.testing.assert_array_equal(
        native_step.payload["elapsed_ms"],
        baseline_step.payload["elapsed_ms"],
    )
    np.testing.assert_array_equal(
        native_step.payload["round_id"], baseline_step.payload["round_id"]
    )
    np.testing.assert_array_equal(native_step.payload["alive"], baseline_step.payload["alive"])
    np.testing.assert_array_equal(
        native_step.payload["joint_action"],
        baseline_step.payload["joint_action"],
    )
    assert native_step.timings["actor_env_public_info_sec"] == 0.0
    assert native_step.timings["actor_env_batch_pack_sec"] == 0.0
    assert native_step.timings["actor_env_compact_action_mask_sec"] >= 0.0
    assert native_step.timings["actor_env_runtime_step_many_sec"] > 0.0
    assert native_step.timings["actor_env_runtime_outer_bookkeeping_sec"] >= 0.0
    assert native_step.timings["actor_env_runtime_context_build_sec"] >= 0.0
    assert native_step.timings["actor_env_runtime_phase_accounted_sec"] > 0.0
    assert native_step.timings["actor_env_runtime_movement_sec"] > 0.0


def test_native_actor_buffer_matches_payload_merge_over_fixed_action_sequence():
    rng = np.random.default_rng(2026052604)
    baseline = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(batch_size=5, actor_count=2, seed=2026052604)
    )
    native = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=5,
            actor_count=2,
            seed=2026052604,
            native_actor_buffer=True,
        )
    )

    for _step_index in range(16):
        action = rng.integers(0, ACTION_COUNT, size=(5, 2), dtype=np.int16)
        baseline_step = baseline.step(action)
        native_step = native.step(action)
        np.testing.assert_array_equal(native_step.reward, baseline_step.reward)
        np.testing.assert_array_equal(native_step.done, baseline_step.done)
        np.testing.assert_array_equal(native_step.action_mask, baseline_step.action_mask)
        for name in ("episode_step", "elapsed_ms", "round_id", "alive", "joint_action"):
            np.testing.assert_array_equal(
                native_step.payload[name],
                baseline_step.payload[name],
            )
        assert not bool(native_step.done.any())


def test_native_actor_buffer_rejects_duplicate_row_partitions():
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=4,
            actor_count=2,
            seed=43,
            native_actor_buffer=True,
        )
    )
    manager._row_partitions = [
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1, 2], dtype=np.int32),
    ]

    with pytest.raises(ValueError, match="duplicate rows"):
        manager.step(np.zeros((4, 2), dtype=np.int16))


def test_hybrid_profile_autoreset_terminal_rows_are_counted():
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=2,
            steps=1,
            warmup_steps=0,
            seed=29,
            max_ticks=1,
        )
    )

    assert result["terminal_row_count"] == 2
    assert result["autoreset_row_count"] == 2
    assert result["done_rows"] == 2
    assert result["last_payload_summary"]["terminal_global_rows"] == [0, 1]
    assert result["last_payload_summary"]["autoreset_global_rows"] == [0, 1]
    assert all(item["final_observation_present"] is True for item in result["last_info"])
    assert all(item["final_observation"].shape == (4, 64, 64) for item in result["last_info"])


def test_hybrid_renderer_backed_mode_preserves_row_major_player_order():
    renderer = _SentinelRenderer()
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=5,
            actor_count=2,
            steps=1,
            warmup_steps=0,
            seed=31,
            pickle_payload=False,
        ),
        observation_renderer=renderer,
    )

    assert result["observation_mode"] == HYBRID_OBSERVATION_MODE_RENDERER_BACKED
    assert result["renderer_backend_name"] == "sentinel_renderer"
    assert result["timings"]["renderer_render_sec"] == 0.123
    assert result["timings"]["renderer_device_render_sec"] == 0.045
    assert result["last_policy_env_row"] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert result["last_policy_player"] == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(batch_size=5, actor_count=2, seed=31),
        observation_renderer=_SentinelRenderer(),
    )
    step = manager.step(np.zeros((5, 2), dtype=np.int16))
    latest = step.flat_obs[:, -1, 0, 0]
    expected = np.asarray([11, 12, 21, 22, 31, 32, 41, 42, 51, 52], dtype=np.float32) / 255.0
    np.testing.assert_allclose(latest, expected)


def test_hybrid_renderer_backed_terminal_final_observation_uses_terminal_frame():
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(batch_size=2, actor_count=2, seed=37, max_ticks=1),
        observation_renderer=_SentinelRenderer(),
    )

    step = manager.step(np.zeros((2, 2), dtype=np.int16))

    assert bool(step.done.all())
    assert [item["final_observation_present"] for item in step.timestep.info] == [
        True,
        True,
        True,
        True,
    ]
    expected = np.asarray([11, 12, 21, 22], dtype=np.float32) / 255.0
    actual = np.asarray(
        [item["final_observation"][-1, 0, 0] for item in step.timestep.info],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected)


def test_native_actor_buffer_supports_renderer_backed_rows_and_matches_payload_merge():
    action = np.asarray([[0, 1], [2, 0], [1, 2]], dtype=np.int16)
    baseline_probe = _CaptureCompactBatchProbe()
    native_probe = _CaptureCompactBatchProbe()
    baseline = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=2,
            seed=61,
            stack_storage_dtype="uint8",
            materialize_scalar_timestep=False,
        ),
        observation_renderer=_SentinelRenderer(),
        batched_stack_probe=baseline_probe,
    )
    native = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=2,
            seed=61,
            stack_storage_dtype="uint8",
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=_SentinelRenderer(),
        batched_stack_probe=native_probe,
    )

    baseline_step = baseline.step(action)
    native_step = native.step(action)
    baseline_batch = baseline_probe.last_batch
    native_batch = native_probe.last_batch

    assert baseline_batch is not None
    assert native_batch is not None
    np.testing.assert_array_equal(native_step.observation, baseline_step.observation)
    np.testing.assert_array_equal(native_step.reward, baseline_step.reward)
    np.testing.assert_array_equal(native_step.done, baseline_step.done)
    np.testing.assert_array_equal(native_step.action_mask, baseline_step.action_mask)
    for name in (
        "policy_env_id",
        "policy_env_row",
        "policy_player",
        "reward",
        "done",
        "target_reward",
        "done_root",
        "to_play",
        "active_root_mask",
        "episode_step",
        "elapsed_ms",
        "round_id",
        "alive",
        "action_mask",
        "joint_action",
    ):
        np.testing.assert_array_equal(getattr(native_batch, name), getattr(baseline_batch, name))
    assert native_step.timings["gather_merge_sec"] <= baseline_step.timings["gather_merge_sec"]
    assert native.contract()["native_actor_buffer"] is True
    assert native.contract()["observation_mode"] == HYBRID_OBSERVATION_MODE_RENDERER_BACKED


def test_native_actor_buffer_filters_persistent_gpu_render_state_keys():
    renderer = _PersistentBackendSentinelRenderer()
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=4,
            actor_count=2,
            seed=63,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
        ),
        observation_renderer=renderer,
    )

    step = manager.step(np.ones((4, 2), dtype=np.int16))

    assert step.timings["actor_render_state_write_sec"] >= 0.0
    assert step.timings["actor_render_state_write_visual_trail_sec"] >= 0.0
    assert step.timings["actor_render_state_write_player_sec"] >= 0.0
    assert step.timings["actor_render_state_write_bonus_sec"] >= 0.0
    assert step.timings["actor_render_state_write_other_sec"] >= 0.0
    split_sum = (
        step.timings["actor_render_state_write_visual_trail_sec"]
        + step.timings["actor_render_state_write_player_sec"]
        + step.timings["actor_render_state_write_bonus_sec"]
        + step.timings["actor_render_state_write_other_sec"]
    )
    assert split_sum <= step.timings["actor_render_state_write_sec"]
    assert renderer.seen_keys
    assert set(renderer.seen_keys[-1]).issubset(PERSISTENT_GPU_PROFILE_RENDER_STATE_KEYS)
    assert "visual_trail_pos" in renderer.seen_keys[-1]
    assert "episode_step" not in renderer.seen_keys[-1]


def test_native_actor_buffer_incremental_visual_trail_matches_actor_state():
    renderer = _PersistentBackendSentinelRenderer()
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=4,
            actor_count=2,
            seed=631,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
        ),
        observation_renderer=renderer,
    )

    manager.step(np.asarray([[0, 1], [1, 2], [2, 0], [0, 2]], dtype=np.int16))
    second = manager.step(np.asarray([[1, 2], [2, 0], [0, 1], [2, 1]], dtype=np.int16))

    rendered_state = renderer.seen_states[-1]
    for actor in manager.actors:
        rows = np.asarray(actor.global_rows, dtype=np.int64)
        for key in (
            "visual_trail_active",
            "visual_trail_write_cursor",
            "visual_trail_pos",
            "visual_trail_radius",
            "visual_trail_owner",
            "visual_trail_break_before",
        ):
            np.testing.assert_array_equal(rendered_state[key][rows], actor.env.state[key])
    assert second.timings["actor_render_state_write_visual_trail_sec"] >= 0.0


def test_native_actor_buffer_incremental_visual_trail_invalidates_after_autoreset():
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=632,
            stack_storage_dtype="uint8",
            max_ticks=1,
            native_actor_buffer=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
    )

    first = manager.step(np.asarray([[0, 1], [1, 2]], dtype=np.int16))

    assert bool(first.done.all())
    state = manager._native_render_state
    assert state is not None
    np.testing.assert_array_equal(
        np.asarray(state["visual_trail_write_cursor"]),
        np.asarray([-1, -1], dtype=np.int32),
    )


def test_persistent_compact_render_state_buffer_sends_compact_keys_only():
    renderer = _PersistentBackendSentinelRenderer()
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=4,
            actor_count=2,
            seed=63,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
            persistent_compact_render_state_buffer=True,
        ),
        observation_renderer=renderer,
    )

    step = manager.step(np.ones((4, 2), dtype=np.int16))

    assert step.timings["actor_render_state_write_sec"] >= 0.0
    contract = manager.contract()
    assert contract["render_state_handoff_mode"] == "persistent_compact_render_state_buffer"
    assert contract["render_state_copy_steps"] == 0
    assert contract["render_state_borrowed_steps"] == 0
    state = renderer.seen_states[-1]
    assert PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_MARKER in state
    assert set(PERSISTENT_GPU_PROFILE_COMPACT_RENDER_STATE_KEYS) <= set(state)
    assert "visual_trail_pos" not in state
    assert "pos" not in state
    assert state["trail_x"].shape == (4, 4096)
    assert state["head_x"].shape == (4, 2)
    assert state["trail_write_cursor"].shape == (4,)
    assert state["trail_x"].dtype == np.float32
    assert state["trail_owner"].dtype == np.int32
    assert state["trail_active"].dtype == np.uint8
    assert state["head_alive"].dtype == np.uint8


def test_persistent_compact_render_state_buffer_matches_native_step_outputs():
    action = np.asarray([[0, 1], [2, 0]], dtype=np.int16)
    copied = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=68,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
    )
    compact = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=68,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
            persistent_compact_render_state_buffer=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
    )

    copied_step = copied.step(action)
    compact_step = compact.step(action)

    np.testing.assert_array_equal(compact_step.observation, copied_step.observation)
    np.testing.assert_array_equal(compact_step.reward, copied_step.reward)
    np.testing.assert_array_equal(compact_step.done, copied_step.done)
    np.testing.assert_array_equal(compact_step.action_mask, copied_step.action_mask)
    assert compact.contract()["render_state_copy_steps"] == 0
    assert copied.contract()["render_state_copy_steps"] == 1


def test_native_actor_buffer_can_borrow_single_actor_render_state():
    action = np.asarray([[0, 1], [2, 0], [1, 2]], dtype=np.int16)
    copied_probe = _CaptureCompactBatchProbe()
    borrowed_probe = _CaptureCompactBatchProbe()
    copied = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=1,
            seed=66,
            stack_storage_dtype="uint8",
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        batched_stack_probe=copied_probe,
    )
    borrowed_renderer = _PersistentBackendSentinelRenderer()
    borrowed = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=1,
            seed=66,
            stack_storage_dtype="uint8",
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            borrow_single_actor_render_state=True,
        ),
        observation_renderer=borrowed_renderer,
        batched_stack_probe=borrowed_probe,
    )

    copied_step = copied.step(action)
    borrowed_step = borrowed.step(action)
    copied_batch = copied_probe.last_batch
    borrowed_batch = borrowed_probe.last_batch

    assert copied_batch is not None
    assert borrowed_batch is not None
    np.testing.assert_array_equal(borrowed_step.observation, copied_step.observation)
    np.testing.assert_array_equal(borrowed_step.reward, copied_step.reward)
    np.testing.assert_array_equal(borrowed_step.done, copied_step.done)
    np.testing.assert_array_equal(borrowed_step.action_mask, copied_step.action_mask)
    for name in (
        "policy_env_id",
        "policy_env_row",
        "policy_player",
        "reward",
        "done",
        "target_reward",
        "done_root",
        "to_play",
        "active_root_mask",
        "episode_step",
        "elapsed_ms",
        "round_id",
        "alive",
        "action_mask",
        "joint_action",
    ):
        np.testing.assert_array_equal(getattr(borrowed_batch, name), getattr(copied_batch, name))
    assert borrowed_step.timings["actor_render_state_write_sec"] == 0.0
    assert borrowed.contract()["render_state_handoff_mode"] == "borrow_single_actor_env_state"
    assert borrowed.contract()["render_state_borrowed_steps"] == 1
    assert borrowed.contract()["render_state_copy_steps"] == 0
    assert copied.contract()["render_state_handoff_mode"] == "copy_actor_state_to_parent_buffers"
    assert copied.contract()["render_state_copy_steps"] == 1
    assert borrowed_renderer.seen_keys
    assert "visual_trail_pos" in borrowed_renderer.seen_keys[-1]


def test_borrowed_single_actor_render_state_handles_terminal_rows():
    renderer = _SequenceRenderer([30, 90])
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=67,
            max_ticks=1,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
            borrow_single_actor_render_state=True,
        ),
        observation_renderer=renderer,
    )

    step = manager.step(np.zeros((2, 2), dtype=np.int16))

    terminal = np.asarray([[30, 31], [32, 33]], dtype=np.uint8)
    reset = np.asarray([[90, 91], [92, 93]], dtype=np.uint8)
    assert bool(step.done.all())
    np.testing.assert_array_equal(step.observation[:, :, -1, 0, 0], terminal)
    np.testing.assert_allclose(
        [item["final_observation"][-1, 0, 0] for item in step.timestep.info],
        terminal.reshape(-1).astype(np.float32) / 255.0,
    )
    np.testing.assert_array_equal(manager._zero_stack[:, :, :3, 0, 0], np.zeros((2, 2, 3)))
    np.testing.assert_array_equal(manager._zero_stack[:, :, -1, 0, 0], reset)
    np.testing.assert_array_equal(step.payload["terminal_global_rows"], np.asarray([0, 1]))
    np.testing.assert_array_equal(step.payload["autoreset_global_rows"], np.asarray([0, 1]))
    assert manager.contract()["render_state_handoff_mode"] == "borrow_single_actor_env_state"
    assert manager.contract()["render_state_borrowed_steps"] == 1
    assert manager.contract()["render_state_copy_steps"] == 0
    assert manager.contract()["render_state_row_overlay_steps"] == 1
    assert manager.contract()["render_state_row_overlay_rows"] == 2
    assert renderer.calls[0]["rows"] == [0, 0, 1, 1]
    assert renderer.calls[0]["state_row_overlay_rows"] == [[0, 1]]
    assert renderer.calls[1]["rows"] == [0, 0, 1, 1]
    assert renderer.calls[1]["state_row_overlay_rows"] == []


def test_borrowed_single_actor_render_state_handles_mixed_terminal_live_rows():
    renderer = _SequenceRenderer([30, 90])
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=67,
            max_ticks=100,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
            borrow_single_actor_render_state=True,
        ),
        observation_renderer=renderer,
    )
    manager.actors[0].env.state["tick"][0] = manager.actors[0].env.max_ticks

    step = manager.step(np.zeros((2, 2), dtype=np.int16))

    terminal_row_latest = np.asarray([30, 31], dtype=np.uint8)
    live_row_latest = np.asarray([32, 33], dtype=np.uint8)
    reset_row_latest = np.asarray([90, 91], dtype=np.uint8)
    np.testing.assert_array_equal(step.done, np.asarray([True, False]))
    np.testing.assert_array_equal(step.observation[0, :, -1, 0, 0], terminal_row_latest)
    np.testing.assert_array_equal(step.observation[1, :, -1, 0, 0], live_row_latest)
    np.testing.assert_allclose(
        [item["final_observation"][-1, 0, 0] for item in step.timestep.info[:2]],
        terminal_row_latest.astype(np.float32) / 255.0,
    )
    assert [item["final_observation_present"] for item in step.timestep.info] == [
        True,
        True,
        False,
        False,
    ]
    np.testing.assert_array_equal(manager._zero_stack[0, :, :3, 0, 0], np.zeros((2, 3)))
    np.testing.assert_array_equal(manager._zero_stack[0, :, -1, 0, 0], reset_row_latest)
    np.testing.assert_array_equal(manager._zero_stack[1, :, -1, 0, 0], live_row_latest)
    np.testing.assert_array_equal(step.payload["terminal_global_rows"], np.asarray([0]))
    np.testing.assert_array_equal(step.payload["autoreset_global_rows"], np.asarray([0]))
    assert renderer.calls[0]["rows"] == [0, 0, 1, 1]
    assert renderer.calls[0]["state_row_overlay_rows"] == [[0]]
    assert renderer.calls[1]["rows"] == [0, 0]
    assert renderer.calls[1]["state_row_overlay_rows"] == []


def test_borrowed_single_actor_render_state_resets_resident_stack_after_terminal_autoreset():
    renderer = _DeviceSequenceRenderer([30, 90])
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=67,
            max_ticks=1,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            borrow_single_actor_render_state=True,
        ),
        observation_renderer=renderer,
    )

    step = manager.step(np.zeros((2, 2), dtype=np.int16))
    resident = manager._last_resident_observation
    resident_stack = manager._resident_observation_stack

    assert bool(step.done.all())
    assert resident is not None
    assert resident_stack is not None
    assert resident.final_device_observation is None
    assert resident.final_device_observation_rows is not None
    np.testing.assert_array_equal(
        resident.final_device_observation_row_indices,
        np.asarray([0, 1], dtype=np.int32),
    )
    assert resident.metadata["resident_final_device_observation_storage"] == "sparse_rows"
    assert step.timings["resident_final_host_observation_dense_elided_count"] == 1.0
    assert step.timings["resident_final_host_observation_dense_elided_row_count"] == 2.0
    assert (
        step.timings["resident_final_host_observation_dense_elided_bytes"]
        == step.observation.nbytes
    )
    assert step.timings["resident_observation_stack_update_sec"] >= 0.0
    assert step.timings["resident_observation_frame_view_sec"] >= 0.0
    assert step.timings["resident_observation_stack_shift_sec"] >= 0.0
    assert step.timings["resident_observation_latest_write_sec"] >= 0.0
    assert step.timings["resident_observation_autoreset_sec"] >= 0.0
    assert step.timings["resident_observation_autoreset_frame_view_sec"] >= 0.0
    assert step.timings["resident_observation_autoreset_index_build_sec"] >= 0.0
    assert step.timings["resident_observation_autoreset_zero_sec"] >= 0.0
    assert step.timings["resident_observation_autoreset_latest_write_sec"] >= 0.0
    terminal = np.asarray([[30, 31], [32, 33]], dtype=np.uint8)
    reset = np.asarray([[90, 91], [92, 93]], dtype=np.uint8)
    np.testing.assert_array_equal(
        resident.final_device_observation_rows[:, :, -1, 0, 0].cpu().numpy(),
        terminal,
    )
    np.testing.assert_array_equal(
        resident_stack[:, :, :3, 0, 0].cpu().numpy(),
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        resident_stack[:, :, -1, 0, 0].cpu().numpy(),
        reset,
    )
    np.testing.assert_array_equal(step.payload["terminal_global_rows"], np.asarray([0, 1]))
    np.testing.assert_array_equal(step.payload["autoreset_global_rows"], np.asarray([0, 1]))
    assert int(step.payload["compact_profile_autoreset_direct_count"]) == 1
    assert int(step.payload["compact_profile_autoreset_template_copy_skipped_count"]) == 1
    assert int(step.payload["compact_profile_autoreset_direct_row_count"]) == 2
    assert step.timings["resident_observation_autoreset_row_count"] == 2.0
    assert renderer.calls[0]["device_only"] is True
    assert renderer.calls[1]["device_only"] is True
    assert manager.contract()["render_state_handoff_mode"] == "borrow_single_actor_env_state"
    assert manager.contract()["render_state_copy_steps"] == 0
    assert manager.contract()["render_state_row_overlay_steps"] == 1
    assert manager.contract()["render_state_row_overlay_rows"] == 2


def test_latest_frame_history_replay_snapshot_rebuilds_after_terminal_autoreset():
    renderer = _DeviceSequenceRenderer([30, 90])
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=67,
            max_ticks=1,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            resident_replay_snapshot_mode=RESIDENT_REPLAY_SNAPSHOT_MODE_LATEST_FRAME_HISTORY,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            borrow_single_actor_render_state=True,
            compact_rollout_slab_sample_gate=True,
        ),
        observation_renderer=renderer,
    )

    step = manager.step(np.zeros((2, 2), dtype=np.int16))
    snapshot = step.resident_observation_replay_snapshot

    assert isinstance(snapshot, CompactResidentFrameStackReplaySnapshotV1)
    assert bool(step.done.all())
    assert len(snapshot.device_frame_history) == 4
    for frame in snapshot.device_frame_history[:3]:
        np.testing.assert_array_equal(
            frame[:, :, 0, 0].cpu().numpy(),
            np.zeros((2, 2), dtype=np.uint8),
        )
    np.testing.assert_array_equal(
        snapshot.device_frame_history[-1][:, :, 0, 0].cpu().numpy(),
        np.asarray([[90, 91], [92, 93]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        snapshot.final_device_observation_rows[:, :, -1, 0, 0].cpu().numpy(),
        np.asarray([[30, 31], [32, 33]], dtype=np.uint8),
    )


def test_resident_autoreset_accepts_full_batch_device_frames_for_partial_reset():
    torch = pytest.importorskip("torch")

    class FullBatchResetDeviceRenderer(_DeviceSequenceRenderer):
        def render(self, request):
            result = super().render(request)
            if len(self.calls) != 2:
                return result
            rows = np.asarray(request.row_indices, dtype=np.int64)
            players = np.asarray(request.controlled_players, dtype=np.int64)
            full = np.full((2, 2, 1, 64, 64), 7, dtype=np.uint8)
            frames = np.asarray(result.frames).reshape(rows.shape[0], 1, 64, 64)
            for item_index, (row, player) in enumerate(zip(rows, players, strict=True)):
                full[int(row), int(player)] = frames[item_index]
            return SourceStateBatchedRenderResult(
                frames=result.frames,
                telemetry=result.telemetry,
                device_frames=torch.as_tensor(full),
            )

    renderer = FullBatchResetDeviceRenderer([30, 90])
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=67,
            max_ticks=100,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            borrow_single_actor_render_state=True,
        ),
        observation_renderer=renderer,
    )
    manager.actors[0].env.state["tick"][0] = manager.actors[0].env.max_ticks

    step = manager.step(np.zeros((2, 2), dtype=np.int16))
    resident_stack = manager._resident_observation_stack

    assert resident_stack is not None
    np.testing.assert_array_equal(step.done, np.asarray([True, False]))
    np.testing.assert_array_equal(
        resident_stack[:, :, -1, 0, 0].cpu().numpy(),
        np.asarray([[90, 91], [32, 33]], dtype=np.uint8),
    )
    assert int(step.payload["compact_profile_autoreset_direct_count"]) == 1
    assert int(step.payload["compact_profile_autoreset_template_copy_skipped_count"]) == 1
    assert int(step.payload["compact_profile_autoreset_direct_row_count"]) == 1
    assert renderer.calls[0]["device_only"] is True
    assert renderer.calls[1]["device_only"] is True


def test_resident_autoreset_stack_reset_requires_device_frames():
    class MissingResetDeviceFramesRenderer(_DeviceSequenceRenderer):
        def render(self, request):
            result = super().render(request)
            if len(self.calls) == 2:
                return SourceStateBatchedRenderResult(
                    frames=result.frames,
                    telemetry=result.telemetry,
                )
            return result

    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=67,
            max_ticks=1,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            borrow_single_actor_render_state=True,
        ),
        observation_renderer=MissingResetDeviceFramesRenderer([30, 90]),
    )

    with pytest.raises(ValueError, match="resident autoreset stack reset requires"):
        manager.step(np.zeros((2, 2), dtype=np.int16))


def test_persistent_device_only_mode_skips_host_stack_update():
    renderer = _PersistentBackendSentinelRenderer()
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=64,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            native_actor_buffer=True,
        ),
        observation_renderer=renderer,
    )

    step = manager.step(np.ones((2, 2), dtype=np.int16))

    assert renderer.device_only_calls == [True]
    assert manager.contract()["update_host_observation_stack"] is False
    assert step.timings["renderer_device_to_host_sec"] == 0.0
    assert not bool(step.observation.any())


def test_resident_observation_step_builds_fresh_device_stack_without_host_fallback():
    renderer = _PersistentBackendSentinelRenderer()
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=640,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=renderer,
        batched_stack_probe=_CountingBatchedStackProbe(),
    )

    first = manager.step(np.asarray([[0, 1], [2, 0]], dtype=np.int16))
    second = manager.step(np.asarray([[1, 2], [0, 1]], dtype=np.int16))

    assert renderer.device_only_calls == [True, True]
    assert not bool(second.observation.any())
    assert second.compact_batch is not None
    resident = second.compact_batch.resident_observation
    assert second.compact_batch.observation_source == "resident_device_v1"
    assert resident is not None
    assert resident.generation_id == 2
    assert resident.host_fallback_allowed is False
    assert resident.row_major_order is True
    assert tuple(resident.root_device_observation.shape) == (4, 4, 64, 64)
    assert str(resident.root_device_observation.dtype) == "torch.uint8"
    assert first.timings["resident_observation_host_fallback_count"] == 0.0
    assert second.timings["resident_observation_h2d_bytes"] == 0.0

    expected_latest = np.asarray(renderer.last_output_device)[:, :, 0]
    np.testing.assert_array_equal(
        resident.device_observation[:, :, -1].cpu().numpy(),
        expected_latest,
    )


def test_compact_rollout_slab_consumes_resident_observation_source():
    class ResidentAssertingSearchService:
        supports_two_phase_compact_search = True
        search_impl = "resident_asserting_search"
        num_simulations = 1

        def __init__(self):
            self.root_batches = []

        def run_action_step(self, root_batch):
            self.root_batches.append(root_batch)
            assert root_batch.observation_source == "resident_device_v1"
            assert root_batch.resident_observation is not None
            root_index = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = np.zeros((root_index.size,), dtype=np.int16)
            return CompactSearchActionStepV1(
                replay_payload_handle="resident-asserting-search:0",
                root_index=root_index,
                env_row=root_batch.env_row[root_index].astype(np.int32, copy=True),
                player=root_batch.player[root_index].astype(np.int16, copy=True),
                policy_env_id=root_batch.policy_env_id[root_index].astype(
                    np.int64,
                    copy=True,
                ),
                selected_action=selected,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(
                f"unexpected flush in first resident step: {replay_payload_handle}"
            )

    search_service = ResidentAssertingSearchService()
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=search_service,
        search_lane="resident_asserting_search",
        policy_source="unit_test",
        copy_root_observation=False,
    )
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=642,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        compact_rollout_slab=slab,
    )

    step = manager.step(np.asarray([[0, 1], [2, 0]], dtype=np.int16))

    assert step.compact_rollout_slab_step is not None
    assert len(search_service.root_batches) == 1
    root_batch = search_service.root_batches[0]
    assert root_batch.metadata["resident_device_observation_authoritative"] is True
    assert root_batch.metadata["resident_observation_host_fallback_used"] is False


def test_compact_rollout_slab_resident_path_ignores_poisoned_host_frames():
    class PoisonedHostPersistentRenderer(_PersistentBackendSentinelRenderer):
        def render(self, request):
            result = super().render(request)
            frames = np.asarray(request.out)
            frames[...] = 255
            return SourceStateBatchedRenderResult(
                frames=frames,
                telemetry=result.telemetry,
                device_frames=result.device_frames,
            )

    class PixelAssertingSearchService:
        supports_two_phase_compact_search = True
        search_impl = "pixel_asserting_resident_search"
        num_simulations = 1

        def __init__(self):
            self.latest_pixels = []
            self.generations = []

        def run_action_step(self, root_batch):
            assert root_batch.observation_source == "resident_device_v1"
            resident = root_batch.resident_observation
            assert resident is not None
            self.generations.append(int(resident.generation_id))
            self.latest_pixels.append(
                resident.root_device_observation[:, -1, 0, 0].detach().cpu().numpy().copy()
            )
            root_index = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = np.zeros((root_index.size,), dtype=np.int16)
            return CompactSearchActionStepV1(
                replay_payload_handle=f"pixel-assert:{len(self.generations)}",
                root_index=root_index,
                env_row=root_batch.env_row[root_index].astype(np.int32, copy=True),
                player=root_batch.player[root_index].astype(np.int16, copy=True),
                policy_env_id=root_batch.policy_env_id[root_index].astype(np.int64, copy=True),
                selected_action=selected,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"unexpected flush: {replay_payload_handle}")

    search_service = PixelAssertingSearchService()
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=search_service,
        search_lane="pixel_asserting_resident_search",
        policy_source="unit_test",
        copy_root_observation=False,
    )
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=643,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=PoisonedHostPersistentRenderer(),
        compact_rollout_slab=slab,
    )

    step = manager.step(np.asarray([[0, 1], [2, 0]], dtype=np.int16))

    assert not bool(step.observation.any())
    assert search_service.generations == [1]
    np.testing.assert_array_equal(
        search_service.latest_pixels[0],
        np.asarray([11, 12, 21, 22], dtype=np.uint8),
    )


def test_compact_rollout_slab_resident_path_consumes_current_generation_each_step():
    class GenerationAssertingSearchService:
        supports_two_phase_compact_search = True
        search_impl = "generation_asserting_resident_search"
        num_simulations = 1

        def __init__(self):
            self.generations = []
            self.latest_pixels = []

        def run_action_step(self, root_batch):
            resident = root_batch.resident_observation
            assert resident is not None
            self.generations.append(int(resident.generation_id))
            self.latest_pixels.append(
                resident.root_device_observation[:, -1, 0, 0].detach().cpu().numpy().copy()
            )
            root_index = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = np.zeros((root_index.size,), dtype=np.int16)
            return CompactSearchActionStepV1(
                replay_payload_handle=f"generation-assert:{len(self.generations)}",
                root_index=root_index,
                env_row=root_batch.env_row[root_index].astype(np.int32, copy=True),
                player=root_batch.player[root_index].astype(np.int16, copy=True),
                policy_env_id=root_batch.policy_env_id[root_index].astype(np.int64, copy=True),
                selected_action=selected,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"unexpected flush: {replay_payload_handle}")

    search_service = GenerationAssertingSearchService()
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=search_service,
        search_lane="generation_asserting_resident_search",
        policy_source="unit_test",
        copy_root_observation=False,
        action_feedback_mode=COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM,
    )
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=644,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        compact_rollout_slab=slab,
    )

    manager.step(np.asarray([[0, 1], [2, 0]], dtype=np.int16))
    manager.step(np.asarray([[1, 2], [0, 1]], dtype=np.int16))

    assert search_service.generations == [1, 2]
    np.testing.assert_array_equal(
        search_service.latest_pixels[0],
        np.asarray([11, 12, 21, 22], dtype=np.uint8),
    )
    np.testing.assert_array_equal(search_service.latest_pixels[1], search_service.latest_pixels[0])


def test_compact_rollout_slab_resident_commit_uses_device_replay_payload():
    torch = pytest.importorskip("torch")

    class DeviceReplaySearchService:
        supports_two_phase_compact_search = True
        search_impl = "device_replay_resident_search"
        num_simulations = 3

        def __init__(self):
            self.counter = 0
            self.device_flushes = 0
            self.host_flushes = 0

        def run_action_step(self, root_batch):
            root_index = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = np.zeros((root_index.size,), dtype=np.int16)
            handle = f"device-replay:{self.counter}"
            self.counter += 1
            return CompactSearchActionStepV1(
                replay_payload_handle=handle,
                root_index=root_index,
                env_row=root_batch.env_row[root_index].astype(np.int32, copy=True),
                player=root_batch.player[root_index].astype(np.int16, copy=True),
                policy_env_id=root_batch.policy_env_id[root_index].astype(
                    np.int64,
                    copy=True,
                ),
                selected_action=selected,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": int(root_index.size),
                    "replay_payload_origin": f"{self.search_impl}:{handle}",
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                    "search_replay_payload_digest": (
                        compact_search_deferred_replay_payload_digest_v1(handle)
                    ),
                    "search_replay_payload_digest_deferred": True,
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            self.host_flushes += 1
            raise AssertionError(f"unexpected host replay flush: {replay_payload_handle}")

        def flush_device_replay_payload(self, replay_payload_handle):
            self.device_flushes += 1
            root_count = 4
            visit_policy = torch.full((root_count, ACTION_COUNT), 1.0 / ACTION_COUNT)
            root_value = torch.linspace(0.0, 0.3, root_count, dtype=torch.float32)
            raw_counts = torch.full((root_count, ACTION_COUNT), 2.0)
            root_index = np.arange(root_count, dtype=np.int32)
            env_row = np.asarray([0, 0, 1, 1], dtype=np.int32)
            player = np.asarray([0, 1, 0, 1], dtype=np.int16)
            return CompactDeviceSearchReplayPayloadV1(
                replay_payload_handle=str(replay_payload_handle),
                root_index=root_index,
                env_row=env_row,
                player=player,
                policy_env_id=np.arange(root_count, dtype=np.int64),
                visit_policy=visit_policy,
                root_value=root_value,
                raw_visit_counts=raw_counts,
                predicted_value=None,
                predicted_policy_logits=None,
                metadata={
                    "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                    "phase": "replay_critical_device",
                    "search_impl": self.search_impl,
                    "num_simulations": self.num_simulations,
                    "active_root_count": root_count,
                    "replay_payload_origin": (f"{self.search_impl}:{str(replay_payload_handle)}"),
                    "device_replay_payload": True,
                    "host_search_payload_fallback_allowed": False,
                    "profile_telemetry": {
                        "compact_torch_search_service_replay_payload_d2h_bytes": 0.0,
                    },
                },
            )

    search_service = DeviceReplaySearchService()
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=search_service,
        search_lane="device_replay_resident_search",
        policy_source="unit_test",
        copy_root_observation=False,
    )
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=645,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        compact_rollout_slab=slab,
    )

    manager.step(np.zeros((2, 2), dtype=np.int16))
    step = manager.step(np.zeros((2, 2), dtype=np.int16))

    assert search_service.host_flushes == 0
    assert search_service.device_flushes == 1
    assert step.compact_rollout_slab_step is not None
    committed = step.compact_rollout_slab_step.committed_index_rows
    assert committed is not None
    assert committed.metadata["device_replay_index_rows"] is True
    assert committed.metadata["replay_index_rows_builder_variant"] == ("device_packed_scalar_v1")
    assert isinstance(committed.metadata["env_player_identity_digest"], str)
    assert len(committed.metadata["env_player_identity_digest"]) == 64
    assert committed.metadata["replay_index_rows_scalar_tensor_count"] == 5
    assert committed.metadata["compact_terminal_metadata_variant"] == ("counts_checksums_v1")
    assert "next_terminal_reason" not in committed.metadata
    assert committed.metadata["next_terminal_reason_shape"] == [2]
    assert committed.metadata["next_death_cause_shape"] == [2, 2]
    assert isinstance(committed.policy_target, torch.Tensor)
    assert isinstance(committed.root_value, torch.Tensor)
    assert (
        step.compact_rollout_slab_step.telemetry[
            "compact_rollout_slab_committed_replay_payload_d2h_bytes"
        ]
        == 0
    )
    assert (
        step.compact_rollout_slab_step.telemetry["compact_rollout_slab_device_replay_index_rows"]
        is True
    )
    assert (
        step.compact_rollout_slab_step.telemetry[
            "compact_rollout_slab_replay_index_rows_builder_variant"
        ]
        == "device_packed_scalar_v1"
    )
    assert (
        step.compact_rollout_slab_step.telemetry[
            "compact_rollout_slab_replay_index_rows_scalar_tensor_count"
        ]
        == 5.0
    )
    assert (
        step.compact_rollout_slab_step.telemetry["compact_rollout_slab_commit_residual_sec"] >= 0.0
    )


def test_resident_observation_rejects_renderer_without_device_frames():
    class HostOnlyPersistentRenderer(_PersistentBackendSentinelRenderer):
        def render(self, request):
            result = super().render(request)
            return SourceStateBatchedRenderResult(
                frames=result.frames,
                telemetry=result.telemetry,
            )

    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=641,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=HostOnlyPersistentRenderer(),
        batched_stack_probe=_CountingBatchedStackProbe(),
    )

    with pytest.raises(ValueError, match="requires renderer device_frames"):
        manager.step(np.asarray([[0, 1], [2, 0]], dtype=np.int16))


def test_persistent_device_only_latest_matches_host_stack_order_for_same_actions():
    host_renderer = _PersistentBackendSentinelRenderer()
    resident_renderer = _PersistentBackendSentinelRenderer()
    host = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=65,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
        ),
        observation_renderer=host_renderer,
    )
    resident = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=65,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            native_actor_buffer=True,
        ),
        observation_renderer=resident_renderer,
    )
    resident_stack = np.zeros((2, 2, 4, 64, 64), dtype=np.uint8)

    for action in (
        np.asarray([[0, 1], [2, 0]], dtype=np.int16),
        np.asarray([[1, 2], [0, 1]], dtype=np.int16),
    ):
        host_step = host.step(action)
        resident.step(action)
        latest_device = np.asarray(resident_renderer.last_output_device)

        assert latest_device.dtype == np.uint8
        assert latest_device.shape == (2, 2, 1, 64, 64)
        np.testing.assert_array_equal(latest_device[:, :, 0], host_step.observation[:, :, -1])
        resident_stack = np.concatenate((resident_stack[:, :, 1:], latest_device), axis=2)
        np.testing.assert_array_equal(resident_stack, host_step.observation)


def test_persistent_resident_model_facing_tensor_matches_stateless_host_tensor():
    host_probe = _ModelFacingTensorCaptureProbe()
    resident_probe = _ModelFacingTensorCaptureProbe()
    host = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=650,
            stack_storage_dtype="uint8",
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=_SentinelRenderer(),
        batched_stack_probe=host_probe,
    )
    resident = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=650,
            stack_storage_dtype="uint8",
            update_host_observation_stack=False,
            resident_observation_search=True,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        observation_renderer=_PersistentBackendSentinelRenderer(),
        batched_stack_probe=resident_probe,
    )

    actions = (
        np.asarray([[0, 1], [2, 0]], dtype=np.int16),
        np.asarray([[1, 2], [0, 1]], dtype=np.int16),
    )
    resident_steps = []
    for action in actions:
        host.step(action)
        resident_steps.append(resident.step(action))

    assert host_probe.sources == [
        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
    ]
    assert resident_probe.sources == [
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    ]
    for host_tensor, resident_tensor in zip(
        host_probe.model_tensors,
        resident_probe.model_tensors,
        strict=True,
    ):
        np.testing.assert_array_equal(host_tensor, resident_tensor)
    for host_mask, resident_mask in zip(
        host_probe.active_root_masks,
        resident_probe.active_root_masks,
        strict=True,
    ):
        np.testing.assert_array_equal(host_mask, resident_mask)
    for host_action_mask, resident_action_mask in zip(
        host_probe.action_masks,
        resident_probe.action_masks,
        strict=True,
    ):
        np.testing.assert_array_equal(host_action_mask, resident_action_mask)
    for step in resident_steps:
        assert not bool(step.observation.any())
        assert step.timings["resident_observation_host_fallback_count"] == 0.0
        assert step.timings["resident_observation_h2d_bytes"] == 0.0
        assert step.timings["renderer_device_to_host_sec"] == 0.0


def test_borrow_single_actor_render_state_matches_copied_persistent_state():
    action = np.asarray([[0, 1], [2, 0]], dtype=np.int16)
    copied_renderer = _PersistentBackendSentinelRenderer()
    borrowed_renderer = _PersistentBackendSentinelRenderer()
    copied = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=66,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
        ),
        observation_renderer=copied_renderer,
    )
    borrowed = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=66,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
            borrow_single_actor_render_state=True,
        ),
        observation_renderer=borrowed_renderer,
    )

    copied_step = copied.step(action)
    borrowed_step = borrowed.step(action)

    np.testing.assert_array_equal(borrowed_step.observation, copied_step.observation)
    np.testing.assert_array_equal(borrowed_step.reward, copied_step.reward)
    np.testing.assert_array_equal(borrowed_step.done, copied_step.done)
    np.testing.assert_array_equal(borrowed_step.action_mask, copied_step.action_mask)
    assert borrowed_step.timings["actor_render_state_write_sec"] == 0.0
    assert borrowed.contract()["render_state_handoff_mode"] == "borrow_single_actor_env_state"
    assert borrowed.contract()["render_state_borrowed_steps"] == 1
    assert borrowed.contract()["render_state_copy_steps"] == 0
    assert set(borrowed_renderer.seen_keys[-1]).issubset(PERSISTENT_GPU_PROFILE_RENDER_STATE_KEYS)
    assert "visual_trail_pos" in borrowed_renderer.seen_keys[-1]
    assert "episode_step" not in borrowed_renderer.seen_keys[-1]


def test_borrow_single_actor_render_state_terminal_path_matches_copied_path():
    copied_renderer = _SequenceRenderer([40, 80])
    borrowed_renderer = _SequenceRenderer([40, 80])
    copied = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=67,
            max_ticks=1,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
        ),
        observation_renderer=copied_renderer,
    )
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            seed=67,
            max_ticks=1,
            stack_storage_dtype="uint8",
            native_actor_buffer=True,
            borrow_single_actor_render_state=True,
        ),
        observation_renderer=borrowed_renderer,
    )

    copied_step = copied.step(np.ones((2, 2), dtype=np.int16))
    borrowed_step = manager.step(np.ones((2, 2), dtype=np.int16))

    np.testing.assert_array_equal(borrowed_step.observation, copied_step.observation)
    np.testing.assert_array_equal(borrowed_step.reward, copied_step.reward)
    np.testing.assert_array_equal(borrowed_step.done, copied_step.done)
    np.testing.assert_array_equal(borrowed_step.action_mask, copied_step.action_mask)
    np.testing.assert_array_equal(
        borrowed_step.payload["terminal_global_rows"],
        copied_step.payload["terminal_global_rows"],
    )
    np.testing.assert_array_equal(
        borrowed_step.payload["autoreset_global_rows"],
        copied_step.payload["autoreset_global_rows"],
    )
    assert manager.contract()["render_state_handoff_mode"] == "borrow_single_actor_env_state"
    assert manager.contract()["render_state_borrowed_steps"] == 1
    assert manager.contract()["render_state_copy_steps"] == 0
    assert manager.contract()["render_state_row_overlay_steps"] == 1
    assert manager.contract()["render_state_row_overlay_rows"] == 2
    assert borrowed_renderer.calls[0]["state_row_overlay_rows"] == [[0, 1]]
    assert borrowed_renderer.calls[0]["synchronize_device"] is True


def test_device_only_mode_requires_persistent_renderer():
    with pytest.raises(ValueError, match="update_host_observation_stack=False requires"):
        HybridBatchedObservationProfileManager(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                stack_storage_dtype="uint8",
                update_host_observation_stack=False,
            ),
            observation_renderer=_SentinelRenderer(),
        )


def test_native_actor_buffer_renderer_backed_autoreset_matches_terminal_then_reset_frames():
    renderer = _SequenceRenderer([30, 90])
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            stack_storage_dtype="uint8",
            max_ticks=1,
            native_actor_buffer=True,
        ),
        observation_renderer=renderer,
    )

    step = manager.step(np.zeros((2, 2), dtype=np.int16))

    terminal = np.asarray([[30, 31], [32, 33]], dtype=np.uint8)
    reset = np.asarray([[90, 91], [92, 93]], dtype=np.uint8)
    assert bool(step.done.all())
    np.testing.assert_array_equal(step.observation[:, :, -1, 0, 0], terminal)
    np.testing.assert_array_equal(manager._zero_stack[:, :, :3, 0, 0], np.zeros((2, 2, 3)))
    np.testing.assert_array_equal(manager._zero_stack[:, :, -1, 0, 0], reset)
    assert renderer.calls[0]["rows"] == [0, 0, 1, 1]
    assert renderer.calls[1]["rows"] == [0, 0, 1, 1]


def test_hybrid_renderer_backed_uint8_stack_stores_bytes_but_scalarizes_float32():
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=1,
            warmup_steps=0,
            stack_storage_dtype="uint8",
        ),
        observation_renderer=_SentinelRenderer(),
    )

    step = manager.step(np.zeros((2, 2), dtype=np.int16))

    assert step.observation.dtype == np.uint8
    assert step.flat_obs.dtype == np.float32
    np.testing.assert_allclose(
        step.flat_obs[:, -1, 0, 0],
        np.asarray([11, 12, 21, 22], dtype=np.float32) / 255.0,
    )


def test_hybrid_renderer_backed_uint8_stack_fifo_preserves_row_major_player_order():
    renderer = _SequenceRenderer([40, 80])
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            stack_storage_dtype="uint8",
            max_ticks=100,
        ),
        observation_renderer=renderer,
    )

    manager.step(np.zeros((2, 2), dtype=np.int16))
    step = manager.step(np.zeros((2, 2), dtype=np.int16))

    assert step.observation.dtype == np.uint8
    np.testing.assert_array_equal(step.observation[:, :, 0, 0, 0], np.zeros((2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(step.observation[:, :, 1, 0, 0], np.zeros((2, 2), dtype=np.uint8))
    np.testing.assert_array_equal(
        step.observation[:, :, 2, 0, 0],
        np.asarray([[40, 41], [42, 43]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        step.observation[:, :, 3, 0, 0],
        np.asarray([[80, 81], [82, 83]], dtype=np.uint8),
    )
    assert renderer.calls[0]["rows"] == [0, 0, 1, 1]
    assert renderer.calls[0]["players"] == [0, 1, 0, 1]


def test_hybrid_renderer_backed_autoreset_resets_internal_stack_after_terminal_copy():
    renderer = _SequenceRenderer([30, 90])
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            stack_storage_dtype="uint8",
            max_ticks=1,
        ),
        observation_renderer=renderer,
    )

    step = manager.step(np.zeros((2, 2), dtype=np.int16))

    terminal = np.asarray([[30, 31], [32, 33]], dtype=np.uint8)
    reset = np.asarray([[90, 91], [92, 93]], dtype=np.uint8)
    assert bool(step.done.all())
    np.testing.assert_array_equal(step.observation[:, :, -1, 0, 0], terminal)
    np.testing.assert_allclose(
        [item["final_observation"][-1, 0, 0] for item in step.timestep.info],
        terminal.reshape(-1).astype(np.float32) / 255.0,
    )
    np.testing.assert_array_equal(manager._zero_stack[:, :, :3, 0, 0], np.zeros((2, 2, 3)))
    np.testing.assert_array_equal(manager._zero_stack[:, :, -1, 0, 0], reset)
    assert renderer.calls[1]["rows"] == [0, 0, 1, 1]
    assert renderer.calls[1]["players"] == [0, 1, 0, 1]


def test_hybrid_renderer_backed_uint8_terminal_final_observation_scalarizes_float32():
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            max_ticks=1,
            stack_storage_dtype="uint8",
        ),
        observation_renderer=_SentinelRenderer(),
    )

    step = manager.step(np.zeros((2, 2), dtype=np.int16))

    assert step.observation.dtype == np.uint8
    assert step.flat_obs.dtype == np.float32
    expected = np.asarray([11, 12, 21, 22], dtype=np.float32) / 255.0
    actual = np.asarray(
        [item["final_observation"][-1, 0, 0] for item in step.timestep.info],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected)


def test_hybrid_profile_runs_policy_search_probe_only_for_measured_steps():
    probe = _CountingProbe()
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=41,
            pickle_payload=False,
        ),
        policy_search_probe=probe,
    )

    assert result["policy_search_probe_backend_name"] == "counting_probe"
    assert result["policy_search_probe_semantics"] == "synthetic_counting_probe"
    assert result["policy_search_probe_calls"] == 2
    assert probe.calls == 3
    np.testing.assert_allclose(result["timings"]["policy_search_probe_sec"], 0.02)
    np.testing.assert_allclose(
        result["timings"]["policy_search_probe_host_to_device_sec"],
        0.004,
    )
    np.testing.assert_allclose(result["timings"]["policy_search_probe_device_sec"], 0.014)
    np.testing.assert_allclose(result["timings"]["policy_search_probe_readback_sec"], 0.002)


def test_hybrid_profile_runs_batched_stack_probe_before_scalar_materialization():
    probe = _CountingBatchedStackProbe()
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=43,
            pickle_payload=False,
            stack_storage_dtype="uint8",
        ),
        observation_renderer=_SentinelRenderer(),
        batched_stack_probe=probe,
    )

    assert result["batched_stack_probe_backend_name"] == "counting_batched_stack_probe"
    assert result["batched_stack_probe_semantics"] == "synthetic_batched_stack_probe"
    assert result["batched_stack_probe_calls"] == 2
    assert result["batched_stack_probe_total_roots"] == 12
    assert result["batched_stack_probe_input_shape"] == [3, 2, 4, 64, 64]
    assert result["batched_stack_probe_input_dtype"] == "uint8"
    assert probe.calls == 3
    np.testing.assert_allclose(result["timings"]["batched_stack_probe_sec"], 0.06)
    np.testing.assert_allclose(
        result["timings"]["batched_stack_probe_host_to_device_sec"],
        0.012,
    )
    np.testing.assert_allclose(result["timings"]["batched_stack_probe_normalize_sec"], 0.008)
    np.testing.assert_allclose(result["timings"]["batched_stack_probe_device_sec"], 0.036)
    np.testing.assert_allclose(result["timings"]["batched_stack_probe_readback_sec"], 0.004)
    assert (
        result["batched_stack_probe_ledger_totals"]["lightzero_array_ceiling_obs_h2d_bytes"]
        == 200.0
    )
    assert (
        result["batched_stack_probe_ledger_totals"]["lightzero_array_ceiling_action_d2h_bytes"]
        == 20.0
    )


def test_hybrid_profile_can_skip_scalar_materialization_for_batched_stack_probe():
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=1,
            warmup_steps=0,
            seed=47,
            materialize_scalar_timestep=False,
        ),
        batched_stack_probe=_CountingBatchedStackProbe(),
    )

    assert result["materialize_scalar_timestep"] is False
    assert result["materialized_timestep_count"] == 0
    assert result["last_flat_obs_shape"] == [0, 4, 64, 64]
    assert result["last_target_reward_shape"] == [0, 1]
    assert result["last_timestep_reward_shape"] == [0]
    assert result["last_info"] == []
    assert result["timings"]["scalar_materialization_sec"] == 0.0
    assert result["timings"]["batched_stack_probe_sec"] > 0.0


def test_hybrid_profile_passes_compact_batch_sidecars_to_probe():
    probe = _CaptureCompactBatchProbe()
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=2,
            seed=53,
            max_ticks=1,
            stack_storage_dtype="uint8",
        ),
        observation_renderer=_SentinelRenderer(),
        batched_stack_probe=probe,
    )

    step = manager.step(np.zeros((2, 2), dtype=np.int16))
    batch = probe.last_batch

    assert batch is not None
    np.testing.assert_array_equal(batch.policy_env_id, np.asarray([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(batch.policy_env_row, np.asarray([0, 0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(batch.policy_player, np.asarray([0, 1, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(batch.action_mask, step.action_mask)
    np.testing.assert_array_equal(batch.reward, step.reward)
    np.testing.assert_array_equal(batch.done, np.asarray([True, True], dtype=np.bool_))
    np.testing.assert_array_equal(batch.done_root, np.asarray([True, True, True, True]))
    np.testing.assert_array_equal(batch.to_play, np.asarray([-1, -1, -1, -1], dtype=np.int64))
    np.testing.assert_array_equal(batch.active_root_mask, np.asarray([False] * 4))
    np.testing.assert_array_equal(batch.terminal_row_mask, np.asarray([True, True]))
    np.testing.assert_array_equal(batch.autoreset_row_mask, np.asarray([True, True]))
    np.testing.assert_array_equal(batch.final_observation_row_mask, np.asarray([True, True]))
    np.testing.assert_array_equal(batch.terminal_global_rows, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(batch.autoreset_global_rows, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(batch.target_reward, step.reward.reshape(4, 1))
    assert batch.final_observation is not None
    assert batch.final_observation.dtype == np.uint8
    expected_latest = np.asarray([11, 12, 21, 22], dtype=np.uint8)
    actual_latest = batch.final_observation.reshape(4, 4, 64, 64)[:, -1, 0, 0]
    np.testing.assert_array_equal(actual_latest, expected_latest)


def test_native_actor_buffer_matches_payload_merge_for_terminal_sidecars():
    action = np.asarray([[0, 1], [2, 0], [1, 2]], dtype=np.int16)
    baseline_probe = _CaptureCompactBatchProbe()
    native_probe = _CaptureCompactBatchProbe()
    baseline = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=2,
            seed=59,
            max_ticks=1,
            materialize_scalar_timestep=False,
        ),
        batched_stack_probe=baseline_probe,
    )
    native = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=2,
            seed=59,
            max_ticks=1,
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
        ),
        batched_stack_probe=native_probe,
    )

    baseline_step = baseline.step(action)
    native_step = native.step(action)
    baseline_batch = baseline_probe.last_batch
    native_batch = native_probe.last_batch

    assert baseline_batch is not None
    assert native_batch is not None
    for name in (
        "policy_env_id",
        "policy_env_row",
        "policy_player",
        "reward",
        "done",
        "target_reward",
        "done_root",
        "to_play",
        "active_root_mask",
        "terminal_row_mask",
        "autoreset_row_mask",
        "terminal_global_rows",
        "autoreset_global_rows",
        "episode_step",
        "elapsed_ms",
        "round_id",
        "alive",
        "action_mask",
        "joint_action",
    ):
        np.testing.assert_array_equal(getattr(native_batch, name), getattr(baseline_batch, name))
    np.testing.assert_array_equal(native_step.reward, baseline_step.reward)
    np.testing.assert_array_equal(native_step.done, baseline_step.done)
    np.testing.assert_array_equal(native_step.action_mask, baseline_step.action_mask)


def test_compact_batch_can_feed_rnd_latest_frame_without_scalar_timestep():
    probe = _CompactRNDLatestFrameProbe()
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=1,
            warmup_steps=0,
            stack_storage_dtype="uint8",
            materialize_scalar_timestep=False,
        ),
        observation_renderer=_SentinelRenderer(),
        batched_stack_probe=probe,
    )

    assert result["materialized_timestep_count"] == 0
    assert result["last_flat_obs_shape"] == [0, 4, 64, 64]
    assert probe.rnd_input is not None
    assert probe.rnd_input.shape == (4, 1, 64, 64)
    assert probe.rnd_input.dtype == np.float32
    np.testing.assert_allclose(
        probe.rnd_input[:, 0, 0, 0],
        np.asarray([11, 12, 21, 22], dtype=np.float32) / np.float32(255.0),
    )
    telemetry = result["batched_stack_probe_last_telemetry"]
    assert telemetry["compact_rnd_batch_contract"] == "compact_row_player_sidecar_v1"
    assert telemetry["compact_rnd_latest_frame_count"] == 4.0
    assert telemetry["compact_rnd_target_reward_shape"] == [4, 1]


def test_compact_service_replay_proof_steps_with_search_actions_and_builds_targets():
    probe = _CompactServiceReplayProofProbe()
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=20260522,
            materialize_scalar_timestep=False,
            compact_service_replay_proof=True,
        ),
        batched_stack_probe=probe,
    )

    assert result["compact_service_replay_proof_enabled"] is True
    assert result["compact_service_replay_proof_calls"] == 1
    assert result["compact_service_replay_proof_skipped_count"] == 0
    assert result["compact_service_replay_proof_target_row_count"] == 4
    assert result["compact_service_replay_proof_warmup_seeded_calls"] == 1
    assert result["compact_service_replay_proof_warmup_seeded_target_row_count"] == 4
    assert result["compact_service_replay_proof_sec"] > 0.0
    assert result["timings"]["compact_service_replay_proof_sec"] == pytest.approx(
        result["compact_service_replay_proof_sec"]
    )
    assert result["compact_service_replay_proof_sec_per_call"] is not None
    assert result["compact_service_replay_proof_sec_per_target_row"] is not None
    assert probe.actions_seen[1] == probe.selected_joint_actions[0]
    assert probe.actions_seen[2] == probe.selected_joint_actions[1]
    telemetry = result["compact_service_replay_proof_last_telemetry"]
    assert telemetry["compact_service_replay_proof_enabled"] is True
    assert telemetry["compact_service_replay_proof_mode"] == "index_rows_v1"
    assert telemetry["compact_service_replay_search_array_source"] == "direct_mcts_arrays"
    assert telemetry["compact_service_replay_chunk_schema_id"] == (
        "curvyzero_compact_replay_index_rows/v1"
    )
    assert telemetry["compact_service_replay_target_row_count"] == 4.0
    assert telemetry["compact_service_replay_action_feedback_verified"] is True
    assert telemetry["compact_service_replay_identity_feedback_verified"] is True
    assert telemetry["compact_service_replay_rnd_latest_verified"] is True
    assert telemetry["compact_service_replay_two_phase_payload_verified"] is True
    assert telemetry["compact_service_replay_action_step_drives_next_action_verified"] is True
    assert telemetry["compact_service_replay_sample_visible_before_payload_flush"] is False
    assert telemetry["compact_service_replay_sample_visible_after_payload_flush"] is True
    assert telemetry["compact_service_replay_payload_gate_pending_count"] == 0.0
    assert telemetry["compact_service_replay_payload_gate_complete_count"] == 1.0
    assert (
        telemetry["compact_service_replay_expected_joint_action_checksum"]
        == (telemetry["compact_service_replay_applied_joint_action_checksum"])
    )
    assert telemetry["compact_service_replay_compact_root_row_checksum"] == 20.0
    assert telemetry["compact_service_replay_player_checksum"] == 6.0
    assert telemetry["compact_service_replay_policy_env_id_checksum"] == 20.0
    assert np.isfinite(telemetry["compact_service_replay_rnd_latest_checksum"])


def test_compact_service_replay_proof_accepts_array_ceiling_compact_search_arrays():
    probe = _ArrayCeilingCompactServiceReplayProofProbe()
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=2,
            warmup_steps=1,
            seed=20260522,
            materialize_scalar_timestep=False,
            compact_service_replay_proof=True,
        ),
        batched_stack_probe=probe,
    )

    assert result["compact_service_replay_proof_enabled"] is True
    assert result["compact_service_replay_proof_calls"] == 1
    assert result["compact_service_replay_proof_skipped_count"] == 0
    assert result["compact_service_replay_proof_warmup_seeded_calls"] == 1
    assert probe.actions_seen[1] == probe.selected_joint_actions[0]
    telemetry = result["compact_service_replay_proof_last_telemetry"]
    assert telemetry["compact_service_replay_search_impl"] == "service_tax_probe"
    assert telemetry["compact_service_replay_num_simulations"] == 5.0
    assert telemetry["compact_service_replay_requested_simulations"] == 5.0
    assert telemetry["compact_service_replay_search_array_source"] == (
        "array_ceiling_compact_search"
    )
    assert telemetry["compact_service_replay_target_row_count"] == 4.0
    assert telemetry["compact_service_replay_action_feedback_verified"] is True
    assert telemetry["compact_service_replay_identity_feedback_verified"] is True
    assert telemetry["compact_service_replay_rnd_latest_verified"] is True
    assert telemetry["compact_service_replay_two_phase_payload_verified"] is True
    assert telemetry["compact_service_replay_action_step_drives_next_action_verified"] is True
    assert telemetry["compact_service_replay_sample_visible_before_payload_flush"] is False
    assert telemetry["compact_service_replay_sample_visible_after_payload_flush"] is True
    assert (
        telemetry["compact_service_replay_expected_joint_action_checksum"]
        == (telemetry["compact_service_replay_applied_joint_action_checksum"])
    )
    assert telemetry["compact_service_replay_compact_root_row_checksum"] == 20.0
    assert telemetry["compact_service_replay_player_checksum"] == 6.0
    assert telemetry["compact_service_replay_policy_env_id_checksum"] == 20.0
    assert np.isfinite(telemetry["compact_service_replay_rnd_latest_checksum"])


def test_compact_service_replay_proof_reports_warmup_seed_separately():
    probe = _CompactServiceReplayProofProbe()
    result = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=2,
            actor_count=1,
            steps=1,
            warmup_steps=1,
            seed=20260522,
            materialize_scalar_timestep=False,
            compact_service_replay_proof=True,
        ),
        batched_stack_probe=probe,
    )

    assert result["compact_service_replay_proof_calls"] == 0
    assert result["compact_service_replay_proof_target_row_count"] == 0
    assert result["compact_service_replay_proof_warmup_seeded_calls"] == 1
    assert result["compact_service_replay_proof_warmup_seeded_target_row_count"] == 4


def test_compact_service_replay_proof_requires_warmup_search_action_seed():
    with pytest.raises(ValueError, match="warmup_steps >= 1"):
        run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                steps=1,
                warmup_steps=0,
                seed=20260522,
                materialize_scalar_timestep=False,
                compact_service_replay_proof=True,
            ),
            batched_stack_probe=_CompactServiceReplayProofProbe(),
        )


def test_hybrid_manager_preserves_action_mask_order_for_probe_and_scalar_timestep():
    action_mask = np.asarray(
        [
            [[True, False, True], [False, True, True]],
            [[True, True, False], [False, False, True]],
            [[False, True, False], [True, False, False]],
        ],
        dtype=bool,
    )
    probe = _CaptureBatchedStackProbe()
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=2,
            pickle_payload=False,
        ),
        batched_stack_probe=probe,
    )
    manager.actors = [
        _PayloadActor(_hybrid_payload(rows=[0, 1], action_mask=action_mask[[0, 1]])),
        _PayloadActor(_hybrid_payload(rows=[2], action_mask=action_mask[[2]])),
    ]

    step = manager.step(np.zeros((3, 2), dtype=np.int16))

    np.testing.assert_array_equal(step.action_mask, action_mask)
    np.testing.assert_array_equal(probe.last_action_mask, action_mask)
    np.testing.assert_array_equal(
        step.timestep.obs["action_mask"],
        action_mask.reshape(3 * 2, ACTION_COUNT),
    )


def test_hybrid_manager_preserves_final_reward_map_from_actor_payloads():
    action_mask = np.ones((3, 2, ACTION_COUNT), dtype=bool)
    final_reward_map = np.asarray(
        [
            [0.0, 0.0],
            [11.0, 12.0],
            [21.0, 22.0],
        ],
        dtype=np.float32,
    )
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=2,
            pickle_payload=False,
            materialize_scalar_timestep=False,
        ),
        batched_stack_probe=_CaptureCompactBatchProbe(),
    )
    manager.actors = [
        _PayloadActor(
            _hybrid_payload(
                rows=[0, 1],
                action_mask=action_mask[[0, 1]],
                final_reward_map=final_reward_map[[0, 1]],
            )
        ),
        _PayloadActor(
            _hybrid_payload(
                rows=[2],
                action_mask=action_mask[[2]],
                final_reward_map=final_reward_map[[2]],
            )
        ),
    ]

    step = manager.step(np.zeros((3, 2), dtype=np.int16))

    np.testing.assert_array_equal(step.final_reward_map, final_reward_map)
    assert step.compact_batch is not None
    np.testing.assert_array_equal(step.compact_batch.final_reward_map, final_reward_map)


def test_hybrid_manager_preserves_terminal_death_metadata_from_actor_payloads():
    action_mask = np.ones((3, 2, ACTION_COUNT), dtype=bool)
    done = np.asarray([False, True, True], dtype=np.bool_)
    terminated = np.asarray([False, True, False], dtype=np.bool_)
    truncated = np.asarray([False, False, True], dtype=np.bool_)
    terminal_reason = np.asarray([0, 1, 2], dtype=np.int16)
    death_count = np.asarray([0, 1, 0], dtype=np.int32)
    death_player = np.asarray([[-1, -1], [0, -1], [-1, -1]], dtype=np.int16)
    death_cause = np.asarray([[0, 0], [1, 0], [0, 0]], dtype=np.int16)
    death_hit_owner = np.asarray([[-1, -1], [-1, -1], [-1, -1]], dtype=np.int16)
    winner = np.asarray([-1, 1, -1], dtype=np.int16)
    draw = np.asarray([False, False, True], dtype=np.bool_)
    manager = HybridBatchedObservationProfileManager(
        HybridObservationProfileConfig(
            batch_size=3,
            actor_count=2,
            pickle_payload=False,
            materialize_scalar_timestep=False,
        ),
        batched_stack_probe=_CaptureCompactBatchProbe(),
    )
    manager.actors = [
        _PayloadActor(
            _hybrid_payload(
                rows=[0, 1],
                action_mask=action_mask[[0, 1]],
                done=done[[0, 1]],
                terminated=terminated[[0, 1]],
                truncated=truncated[[0, 1]],
                terminal_reason=terminal_reason[[0, 1]],
                death_count=death_count[[0, 1]],
                death_player=death_player[[0, 1]],
                death_cause=death_cause[[0, 1]],
                death_hit_owner=death_hit_owner[[0, 1]],
                winner=winner[[0, 1]],
                draw=draw[[0, 1]],
            )
        ),
        _PayloadActor(
            _hybrid_payload(
                rows=[2],
                action_mask=action_mask[[2]],
                done=done[[2]],
                terminated=terminated[[2]],
                truncated=truncated[[2]],
                terminal_reason=terminal_reason[[2]],
                death_count=death_count[[2]],
                death_player=death_player[[2]],
                death_cause=death_cause[[2]],
                death_hit_owner=death_hit_owner[[2]],
                winner=winner[[2]],
                draw=draw[[2]],
            )
        ),
    ]

    step = manager.step(np.zeros((3, 2), dtype=np.int16))

    np.testing.assert_array_equal(step.done, done)
    np.testing.assert_array_equal(step.terminated, terminated)
    np.testing.assert_array_equal(step.truncated, truncated)
    np.testing.assert_array_equal(step.terminal_reason, terminal_reason)
    np.testing.assert_array_equal(step.death_count, death_count)
    assert step.compact_batch is not None
    np.testing.assert_array_equal(step.compact_batch.death_player, death_player)
    np.testing.assert_array_equal(step.compact_batch.death_cause, death_cause)
    np.testing.assert_array_equal(step.compact_batch.death_hit_owner, death_hit_owner)
    np.testing.assert_array_equal(step.compact_batch.winner, winner)
    np.testing.assert_array_equal(step.compact_batch.draw, draw)


def test_hybrid_profile_rejects_skipping_scalar_materialization_without_consumer():
    try:
        run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=2,
                actor_count=1,
                steps=1,
                warmup_steps=0,
                materialize_scalar_timestep=False,
            )
        )
    except ValueError as exc:
        assert "requires batched_stack_probe" in str(exc)
    else:
        raise AssertionError("expected ValueError")


class _SentinelRenderer:
    backend_name = "sentinel_renderer"

    def render(self, request):
        rows = np.asarray(request.row_indices, dtype=np.int64)
        players = np.asarray(request.controlled_players, dtype=np.int64)
        values = ((rows + 1) * 10 + players + 1).astype(np.uint8)
        out = np.asarray(request.out)
        out.fill(0)
        out[:, 0, :, :] = values[:, None, None]
        return SourceStateBatchedRenderResult(
            frames=out,
            telemetry={
                "render_sec": 0.123,
                "device_render_sec": 0.045,
            },
        )


class _SequenceRenderer:
    backend_name = "sequence_renderer"

    def __init__(self, bases):
        self._bases = [int(base) for base in bases]
        self.calls = []

    def render(self, request):
        rows = np.asarray(request.row_indices, dtype=np.int64)
        players = np.asarray(request.controlled_players, dtype=np.int64)
        call_index = len(self.calls)
        base = self._bases[min(call_index, len(self._bases) - 1)]
        values = (base + rows * 2 + players).astype(np.uint8)
        out = np.asarray(request.out)
        out.fill(0)
        out[:, 0, :, :] = values[:, None, None]
        self.calls.append(
            {
                "rows": rows.astype(int).tolist(),
                "players": players.astype(int).tolist(),
                "out_shape": list(out.shape),
                "device_only": bool(request.device_only),
                "synchronize_device": bool(request.synchronize_device),
                "state_row_overlay_rows": [
                    np.asarray(overlay.rows, dtype=np.int64).astype(int).tolist()
                    for overlay in request.state_row_overlays
                ],
            }
        )
        return SourceStateBatchedRenderResult(
            frames=out,
            telemetry={
                "render_sec": 0.001,
                "device_render_sec": 0.0005,
            },
        )


class _DeviceSequenceRenderer(_SequenceRenderer):
    backend_name = PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME

    def render(self, request):
        result = super().render(request)
        import torch

        return SourceStateBatchedRenderResult(
            frames=result.frames,
            telemetry=result.telemetry,
            device_frames=torch.as_tensor(np.asarray(result.frames).copy()),
        )


class _PersistentBackendSentinelRenderer(_SentinelRenderer):
    backend_name = PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME

    def __init__(self):
        self.seen_keys = []
        self.seen_states = []
        self.device_only_calls = []
        self.last_output_device = None

    def render(self, request):
        self.seen_keys.append(set(request.state.keys()))
        self.seen_states.append(
            {str(key): np.asarray(value).copy() for key, value in request.state.items()}
        )
        self.device_only_calls.append(bool(request.device_only))
        if bool(request.device_only):
            import torch

            out = np.asarray(request.out)
            rows = np.asarray(request.row_indices, dtype=np.int64)
            players = np.asarray(request.controlled_players, dtype=np.int64)
            values = ((rows + 1) * 10 + players + 1).astype(np.uint8)
            device_frames = np.zeros_like(out)
            device_frames[:, 0, :, :] = values[:, None, None]
            batch_size = int(rows.max(initial=-1) + 1)
            player_count = int(players.max(initial=-1) + 1)
            self.last_output_device = device_frames.reshape(batch_size, player_count, 1, 64, 64)
            return SourceStateBatchedRenderResult(
                frames=out,
                telemetry={
                    "render_sec": 0.123,
                    "device_render_sec": 0.045,
                    "device_to_host_sec": 0.0,
                },
                device_frames=torch.as_tensor(self.last_output_device),
            )
        result = super().render(request)
        frames = np.asarray(result.frames)
        rows = np.asarray(request.row_indices, dtype=np.int64)
        players = np.asarray(request.controlled_players, dtype=np.int64)
        batch_size = int(rows.max(initial=-1) + 1)
        player_count = int(players.max(initial=-1) + 1)
        self.last_output_device = frames.reshape(batch_size, player_count, 1, 64, 64).copy()
        return SourceStateBatchedRenderResult(
            frames=result.frames,
            telemetry=result.telemetry,
            device_frames=self.last_output_device,
        )


class _CountingProbe:
    backend_name = "counting_probe"
    semantics = "synthetic_counting_probe"

    def __init__(self):
        self.calls = 0

    def run(self, flat_obs):
        self.calls += 1
        assert flat_obs.ndim == 4
        return HybridPolicySearchProbeResult(
            telemetry={
                "total_sec": 0.01,
                "host_to_device_sec": 0.002,
                "device_sec": 0.007,
                "readback_sec": 0.001,
            }
        )


class _ModelFacingTensorCaptureProbe:
    backend_name = "model_facing_tensor_capture_probe"
    semantics = "capture_normalized_model_facing_tensor"

    def __init__(self):
        self.model_tensors = []
        self.active_root_masks = []
        self.action_masks = []
        self.sources = []

    def run_compact_batch(self, batch):
        self.sources.append(str(batch.observation_source))
        self.active_root_masks.append(np.asarray(batch.active_root_mask, dtype=bool).copy())
        self.action_masks.append(
            np.asarray(batch.action_mask, dtype=bool).reshape(-1, ACTION_COUNT).copy()
        )
        if batch.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1:
            resident = batch.resident_observation
            assert resident is not None
            tensor = resident.root_device_observation
            assert tensor is not None
            model_tensor = tensor.float().div(255.0).detach().cpu().numpy()
        else:
            observation = np.asarray(batch.observation)
            model_tensor = observation.reshape(-1, *observation.shape[2:]).astype(
                np.float32,
                copy=False,
            )
            if observation.dtype == np.uint8:
                model_tensor = model_tensor / np.float32(255.0)
        assert model_tensor.shape == (4, 4, 64, 64)
        self.model_tensors.append(np.asarray(model_tensor, dtype=np.float32).copy())
        return HybridBatchedStackProbeResult(
            telemetry={
                "total_sec": 0.0,
                "host_to_device_sec": 0.0,
                "normalize_sec": 0.0,
                "device_sec": 0.0,
                "readback_sec": 0.0,
            }
        )


class _CountingBatchedStackProbe:
    backend_name = "counting_batched_stack_probe"
    semantics = "synthetic_batched_stack_probe"

    def __init__(self):
        self.calls = 0

    def run(self, observation, action_mask):
        self.calls += 1
        assert observation.ndim == 5
        assert action_mask.shape == observation.shape[:2] + (3,)
        return HybridBatchedStackProbeResult(
            telemetry={
                "total_sec": 0.03,
                "host_to_device_sec": 0.006,
                "normalize_sec": 0.004,
                "device_sec": 0.018,
                "readback_sec": 0.002,
                "lightzero_array_ceiling_obs_h2d_bytes": 100.0,
                "lightzero_array_ceiling_action_d2h_bytes": 10.0,
            }
        )


class _CaptureBatchedStackProbe:
    backend_name = "capture_batched_stack_probe"
    semantics = "capture_action_mask_order"

    def __init__(self):
        self.last_action_mask = None

    def run(self, observation, action_mask):
        self.last_action_mask = np.asarray(action_mask, dtype=bool).copy()
        assert observation.shape[:2] == action_mask.shape[:2]
        return HybridBatchedStackProbeResult(
            telemetry={
                "total_sec": 0.001,
                "host_to_device_sec": 0.0,
                "normalize_sec": 0.0,
                "device_sec": 0.0,
                "readback_sec": 0.0,
            }
        )


class _CompactRNDLatestFrameProbe:
    backend_name = "compact_rnd_latest_frame_probe"
    semantics = "compact_rnd_latest_frame_extraction_no_scalar_timestep"

    def __init__(self):
        self.rnd_input = None

    def run_compact_batch(self, batch: HybridCompactBatch):
        self.rnd_input = xb.extract_policy_gray64_latest_for_rnd_from_compact_observation(
            batch.observation,
            batch.target_reward,
        )
        return HybridBatchedStackProbeResult(
            telemetry={
                "total_sec": 0.001,
                "host_to_device_sec": 0.0,
                "normalize_sec": 0.0,
                "device_sec": 0.0,
                "readback_sec": 0.0,
                "compact_rnd_batch_contract": "compact_row_player_sidecar_v1",
                "compact_rnd_latest_frame_count": float(self.rnd_input.shape[0]),
                "compact_rnd_target_reward_shape": list(batch.target_reward.shape),
            }
        )


class _CompactServiceReplayProofProbe:
    backend_name = "compact_service_replay_proof_probe"
    semantics = "direct_ctree_compact_service_replay_proof_test"
    _arrays_boundary_impl = "direct_ctree_gpu_latent"
    _num_simulations = 8

    def __init__(self):
        self._last_direct_mcts_arrays = None
        self.actions_seen = []
        self.selected_joint_actions = []

    def run_compact_batch(self, batch: HybridCompactBatch):
        self.actions_seen.append(np.asarray(batch.joint_action, dtype=np.int16).tolist())
        active_root_mask = np.asarray(batch.active_root_mask, dtype=np.bool_).reshape(-1)
        active_root_indices = np.flatnonzero(active_root_mask)
        env_rows = np.asarray(batch.policy_env_row, dtype=np.int64).reshape(-1)
        players = np.asarray(batch.policy_player, dtype=np.int64).reshape(-1)
        selected = np.zeros((active_root_indices.size,), dtype=np.int16)
        visit_policy = np.zeros((active_root_indices.size, ACTION_COUNT), dtype=np.float32)
        joint_action = np.zeros(batch.joint_action.shape, dtype=np.int16)
        for output_row, root_index in enumerate(active_root_indices):
            env_row = int(env_rows[int(root_index)])
            player = int(players[int(root_index)])
            action = np.int16((env_row + player + len(self.actions_seen)) % ACTION_COUNT)
            selected[output_row] = action
            visit_policy[output_row, int(action)] = 1.0
            joint_action[env_row, player] = action
        root_value = np.linspace(-0.25, 0.25, active_root_indices.size, dtype=np.float32)
        self._last_direct_mcts_arrays = {
            "selected_action": selected.copy(),
            "visit_policy": visit_policy.copy(),
            "root_value": root_value.copy(),
            "predicted_value": root_value.copy(),
            "predicted_policy_logits": visit_policy.copy(),
            "array_source": "direct_mcts_arrays",
            "search_impl": self._arrays_boundary_impl,
            "actual_search_simulations": self._num_simulations,
            "requested_simulations": self._num_simulations,
        }
        self.selected_joint_actions.append(joint_action.tolist())
        return HybridBatchedStackProbeResult(
            telemetry={
                "total_sec": 0.001,
                "host_to_device_sec": 0.0,
                "normalize_sec": 0.0,
                "device_sec": 0.0,
                "readback_sec": 0.0,
                "lightzero_mcts_arrays_boundary_impl": self._arrays_boundary_impl,
                "lightzero_consumer_num_simulations": float(self._num_simulations),
                "lightzero_root_count": float(active_root_indices.size),
                "compact_service_contract_v1_enabled": True,
            }
        )


class _ArrayCeilingCompactServiceReplayProofProbe(_CompactServiceReplayProofProbe):
    semantics = "service_tax_compact_service_replay_proof_test"
    _mode = "service_tax_probe"
    _num_simulations = 5

    def __init__(self):
        super().__init__()
        self._last_compact_search_arrays = None

    def run_compact_batch(self, batch: HybridCompactBatch):
        result = super().run_compact_batch(batch)
        self._last_compact_search_arrays = dict(self._last_direct_mcts_arrays)
        self._last_compact_search_arrays.update(
            {
                "array_source": "array_ceiling_compact_search",
                "search_impl": self._mode,
                "actual_search_simulations": self._num_simulations,
                "requested_simulations": self._num_simulations,
            }
        )
        self._last_direct_mcts_arrays = None
        telemetry = dict(result.telemetry)
        telemetry.pop("lightzero_mcts_arrays_boundary_impl", None)
        telemetry.pop("lightzero_consumer_num_simulations", None)
        telemetry["lightzero_array_ceiling_mode"] = self._mode
        telemetry["lightzero_array_ceiling_requested_simulations"] = float(self._num_simulations)
        telemetry["lightzero_array_ceiling_compact_search_arrays_stored"] = 1.0
        return HybridBatchedStackProbeResult(telemetry=telemetry)


class _CaptureCompactBatchProbe:
    backend_name = "capture_compact_batch_probe"
    semantics = "capture_compact_batch_v1"

    def __init__(self):
        self.last_batch = None

    def run_compact_batch(self, batch):
        self.last_batch = _copy_compact_batch_for_test(batch)
        return HybridBatchedStackProbeResult(
            telemetry={
                "total_sec": 0.001,
                "host_to_device_sec": 0.0,
                "normalize_sec": 0.0,
                "device_sec": 0.0,
                "readback_sec": 0.0,
                "compact_batch_contract": "compact_row_player_sidecar_v1",
            }
        )

    def run(self, observation, action_mask):
        raise AssertionError("compact batch probe should use run_compact_batch")


def _copy_compact_batch_for_test(batch: HybridCompactBatch) -> HybridCompactBatch:
    return HybridCompactBatch(
        observation=np.asarray(batch.observation).copy(),
        action_mask=np.asarray(batch.action_mask).copy(),
        reward=np.asarray(batch.reward).copy(),
        final_reward_map=np.asarray(batch.final_reward_map).copy(),
        done=np.asarray(batch.done).copy(),
        terminated=np.asarray(batch.terminated).copy(),
        truncated=np.asarray(batch.truncated).copy(),
        terminal_reason=np.asarray(batch.terminal_reason).copy(),
        death_count=np.asarray(batch.death_count).copy(),
        death_player=np.asarray(batch.death_player).copy(),
        death_cause=np.asarray(batch.death_cause).copy(),
        death_hit_owner=np.asarray(batch.death_hit_owner).copy(),
        winner=np.asarray(batch.winner).copy(),
        draw=np.asarray(batch.draw).copy(),
        policy_env_id=np.asarray(batch.policy_env_id).copy(),
        policy_env_row=np.asarray(batch.policy_env_row).copy(),
        policy_player=np.asarray(batch.policy_player).copy(),
        target_reward=np.asarray(batch.target_reward).copy(),
        done_root=np.asarray(batch.done_root).copy(),
        to_play=np.asarray(batch.to_play).copy(),
        active_root_mask=np.asarray(batch.active_root_mask).copy(),
        final_observation=(
            None if batch.final_observation is None else np.asarray(batch.final_observation).copy()
        ),
        final_observation_row_mask=np.asarray(batch.final_observation_row_mask).copy(),
        terminal_row_mask=np.asarray(batch.terminal_row_mask).copy(),
        autoreset_row_mask=np.asarray(batch.autoreset_row_mask).copy(),
        terminal_global_rows=np.asarray(batch.terminal_global_rows).copy(),
        autoreset_global_rows=np.asarray(batch.autoreset_global_rows).copy(),
        episode_step=np.asarray(batch.episode_step).copy(),
        elapsed_ms=np.asarray(batch.elapsed_ms).copy(),
        round_id=np.asarray(batch.round_id).copy(),
        alive=np.asarray(batch.alive).copy(),
        joint_action=np.asarray(batch.joint_action).copy(),
    )


class _PayloadActor:
    def __init__(self, payload):
        self._payload = payload

    def step(
        self,
        _joint_action,
        *,
        autoreset_terminal_rows,
        include_render_state,
        render_state_keys=None,
    ):
        assert autoreset_terminal_rows is True
        assert include_render_state is False
        assert render_state_keys is None
        return self._payload


def _hybrid_payload(
    *,
    rows,
    action_mask,
    final_reward_map=None,
    done=None,
    terminated=None,
    truncated=None,
    terminal_reason=None,
    death_count=None,
    death_player=None,
    death_cause=None,
    death_hit_owner=None,
    winner=None,
    draw=None,
):
    global_rows = np.asarray(rows, dtype=np.int32)
    row_count = int(global_rows.shape[0])
    if final_reward_map is None:
        final_reward_map = np.zeros((row_count, 2), dtype=np.float32)
    done_array = (
        np.zeros(row_count, dtype=np.bool_) if done is None else np.asarray(done, dtype=np.bool_)
    )
    terminated_array = (
        done_array.copy() if terminated is None else np.asarray(terminated, dtype=np.bool_)
    )
    truncated_array = (
        np.zeros(row_count, dtype=np.bool_)
        if truncated is None
        else np.asarray(truncated, dtype=np.bool_)
    )
    return HybridActorStepPayload(
        actor_id=0,
        local_rows=np.arange(row_count, dtype=np.int32),
        global_rows=global_rows,
        reward=np.zeros((row_count, 2), dtype=np.float32),
        final_reward_map=np.asarray(final_reward_map, dtype=np.float32),
        done=done_array,
        terminated=terminated_array,
        truncated=truncated_array,
        terminal_reason=np.zeros(row_count, dtype=np.int16)
        if terminal_reason is None
        else np.asarray(terminal_reason, dtype=np.int16),
        death_count=np.zeros(row_count, dtype=np.int32)
        if death_count is None
        else np.asarray(death_count, dtype=np.int32),
        death_player=np.full((row_count, 2), -1, dtype=np.int16)
        if death_player is None
        else np.asarray(death_player, dtype=np.int16),
        death_cause=np.zeros((row_count, 2), dtype=np.int16)
        if death_cause is None
        else np.asarray(death_cause, dtype=np.int16),
        death_hit_owner=np.full((row_count, 2), -1, dtype=np.int16)
        if death_hit_owner is None
        else np.asarray(death_hit_owner, dtype=np.int16),
        winner=np.full(row_count, -1, dtype=np.int16)
        if winner is None
        else np.asarray(winner, dtype=np.int16),
        draw=np.zeros(row_count, dtype=np.bool_)
        if draw is None
        else np.asarray(draw, dtype=np.bool_),
        episode_step=np.ones(row_count, dtype=np.int32),
        elapsed_ms=np.zeros(row_count, dtype=np.float64),
        round_id=np.zeros(row_count, dtype=np.int32),
        alive=np.ones((row_count, 2), dtype=np.bool_),
        action_mask=np.asarray(action_mask, dtype=np.bool_),
        joint_action=np.zeros((row_count, 2), dtype=np.int16),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        render_state=None,
        autoreset_render_state=None,
        actor_env_step_sec=0.0,
        actor_autoreset_sec=0.0,
    )
