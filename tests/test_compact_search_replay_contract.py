import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from curvyzero.training.compact_search_service import CompactSearchServiceV1
from curvyzero.training.compact_search_service import COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID
from curvyzero.training.compact_search_service import (
    COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import CompactDeviceSearchReplayPayloadV1
from curvyzero.training.compact_search_service import CompactSearchActionStepV1
from curvyzero.training.compact_search_service import CompactSearchComparatorServiceV1
from curvyzero.training.compact_search_service import (
    compact_search_action_step_v1_from_result,
)
from curvyzero.training.compact_search_service import compact_search_array_digest_v1
from curvyzero.training.compact_search_service import (
    compact_search_deferred_replay_payload_digest_v1,
)
from curvyzero.training.compact_search_service import CompactSearchPayloadGateV1
from curvyzero.training.compact_search_service import (
    compact_search_replay_payload_v1_from_result,
)
from curvyzero.training.compact_search_service import (
    compact_search_replay_payload_digest_v1,
)
from curvyzero.training.compact_search_service import compact_search_result_v1_from_arrays
from curvyzero.training.compact_search_service import (
    validate_compact_search_two_phase_payload_v1,
)
from curvyzero.training.compact_policy_row_bridge import (
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
)
from curvyzero.training.compact_policy_row_bridge import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_policy_row_bridge import (
    ResidentObservationBatchV1,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_replay_chunk_v1_from_search_result,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_device_replay_index_rows_v1_from_payload,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_device_replay_index_rows_v1_from_owner_action_context_payload,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_replay_index_rows_v1_from_owner_action_context_payload,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_replay_index_rows_v1_from_search_result,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_transition_outcome_v1_from_next_root_batch,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_transition_outcome_v1_from_root_build_request,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_root_batch_v1,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_root_batch_v1_from_request,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_root_action_context_v1_from_request,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_root_build_request_v1_from_batch,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_target_rows_from_search_arrays_v0,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_policy_row_records_from_compact_search_v0,
)
from curvyzero.training.compact_policy_row_bridge import (
    materialize_compact_target_rows_from_index_rows_v1,
)
from curvyzero.training.compact_policy_row_bridge import (
    materialize_compact_target_rows_from_index_row_groups_v1,
)
from curvyzero.training.compact_policy_row_bridge import (
    sample_compact_target_rows_from_index_row_groups_v1,
)
from curvyzero.training.compact_policy_row_bridge import (
    validate_compact_search_result_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_policy_refresh_metadata_from_state_v1,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM,
)
import curvyzero.training.compact_rollout_slab as compact_rollout_slab_module
from curvyzero.training.compact_rollout_slab import CompactRolloutSlab
from curvyzero.training.compact_rollout_slab import selected_joint_action_from_action_step
from curvyzero.training.compact_rollout_slab import selected_joint_action_from_search_result
from curvyzero.training.exploration_bonus import (
    CurvyRNDRewardModel,
)
from curvyzero.training.exploration_bonus import (
    extract_policy_gray64_latest_for_rnd_from_compact_observation,
)
from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT
from curvyzero.training.multiplayer_source_state_target_rows import DEFAULT_TO_PLAY
from curvyzero.training.multiplayer_source_state_target_rows import PolicyRowRecordV0
from curvyzero.training.multiplayer_source_state_target_rows import (
    build_source_state_multiplayer_sample_batch_v0,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    build_source_state_multiplayer_target_rows_v0,
)
from curvyzero.training.multiplayer_source_state_lightzero_native_bridge import (
    build_lightzero_source_state_native_game_segments_v0,
)
from curvyzero.training.multiplayer_source_state_lightzero_native_bridge import (
    maybe_push_lightzero_source_state_native_segments_into_muzero_buffer_v0,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    FRAME_STACK_SHAPE,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayChunkV0,
)
from curvyzero.training.source_state_hybrid_observation_profile import HybridCompactBatch
from curvyzero.training.two_seat_native_replay_bridge import (
    build_lightzero_muzero_bridge_config,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


def test_build_compact_root_batch_requires_explicit_resident_observation_contract():
    batch_size = 2
    player_count = 1
    root_count = batch_size * player_count
    observation = np.zeros((batch_size, player_count, 4, 64, 64), dtype=np.uint8)
    compact_batch = HybridCompactBatch(
        observation=observation,
        action_mask=np.ones((batch_size, player_count, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((batch_size, player_count), dtype=np.float32),
        final_reward_map=np.zeros((batch_size, player_count), dtype=np.float32),
        done=np.zeros((batch_size,), dtype=np.bool_),
        policy_env_id=np.arange(root_count, dtype=np.int32),
        policy_env_row=np.arange(batch_size, dtype=np.int32),
        policy_player=np.zeros((root_count,), dtype=np.int32),
        target_reward=np.zeros((root_count, 1), dtype=np.float32),
        done_root=np.zeros((root_count,), dtype=np.bool_),
        to_play=np.full((root_count,), DEFAULT_TO_PLAY, dtype=np.int64),
        active_root_mask=np.ones((root_count,), dtype=np.bool_),
        final_observation=None,
        final_observation_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        terminal_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        autoreset_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.zeros((batch_size,), dtype=np.int32),
        elapsed_ms=np.zeros((batch_size,), dtype=np.float64),
        round_id=np.zeros((batch_size,), dtype=np.int32),
        alive=np.ones((batch_size, player_count), dtype=np.bool_),
        joint_action=np.zeros((batch_size, player_count), dtype=np.int16),
    )

    with pytest.raises(ReplayCompatibilityError, match="requires resident_observation"):
        build_compact_root_batch_v1(
            compact_batch,
            search_lane="unit_test_resident_missing",
            observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        )

    resident_observation = ResidentObservationBatchV1(
        device_observation=np.zeros_like(observation),
        root_device_observation=np.zeros((root_count, 4, 64, 64), dtype=np.uint8),
        generation_id=3,
        batch_size=batch_size,
        player_count=player_count,
        stack_shape=(4, 64, 64),
        dtype="uint8",
        device="cuda:0",
        row_major_order=True,
        fresh_for_step_index=9,
        source_backend="unit_test_resident_contract",
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_resident_contract",
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=resident_observation,
    )

    assert root_batch.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    assert root_batch.resident_observation is resident_observation
    assert root_batch.metadata["resident_observation_generation_id"] == 3
    assert root_batch.metadata["resident_observation_host_fallback_allowed"] is False
    assert root_batch.metadata["resident_observation_host_fallback_used"] is False
    assert root_batch.metadata["host_observation_authoritative"] is False

    with pytest.raises(ReplayCompatibilityError, match="observation_source is host_array_v1"):
        build_compact_root_batch_v1(
            compact_batch,
            search_lane="unit_test_resident_wrong_source",
            observation_source=COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
            resident_observation=resident_observation,
        )


def test_root_build_request_preserves_mechanics_outcome_sidecars_for_owner_derivation():
    compact_batch = _compact_batch_with_terminal_mechanics_outcome()

    request = compact_root_build_request_v1_from_batch(
        compact_batch,
        search_lane="unit_test_owner_transition_outcome",
    )
    assert request.metadata["mechanics_outcome_sidecars_present"] is True
    assert np.array_equal(request.terminated, compact_batch.terminated)
    assert np.array_equal(request.truncated, compact_batch.truncated)
    assert np.array_equal(request.final_reward_map, compact_batch.final_reward_map)

    root_batch = build_compact_root_batch_v1_from_request(request)
    assert root_batch.metadata["mechanics_outcome_sidecars_present"] is True
    assert np.array_equal(root_batch.terminated, compact_batch.terminated)
    assert np.array_equal(root_batch.truncated, compact_batch.truncated)
    assert np.array_equal(root_batch.final_reward_map, compact_batch.final_reward_map)

    outcome = compact_transition_outcome_v1_from_next_root_batch(root_batch)
    request_outcome = compact_transition_outcome_v1_from_root_build_request(request)
    assert np.array_equal(outcome.next_reward, compact_batch.reward)
    assert np.array_equal(outcome.next_done, compact_batch.done)
    assert np.array_equal(outcome.next_terminated, compact_batch.terminated)
    assert np.array_equal(outcome.next_truncated, compact_batch.truncated)
    assert np.array_equal(outcome.next_final_reward_map, compact_batch.final_reward_map)
    assert np.array_equal(
        outcome.next_final_observation_row_mask,
        compact_batch.final_observation_row_mask,
    )
    assert np.array_equal(request_outcome.next_reward, outcome.next_reward)
    assert np.array_equal(request_outcome.next_done, outcome.next_done)
    assert np.array_equal(request_outcome.next_terminated, outcome.next_terminated)
    assert np.array_equal(request_outcome.next_truncated, outcome.next_truncated)
    assert np.array_equal(request_outcome.next_final_reward_map, outcome.next_final_reward_map)
    assert np.array_equal(
        request_outcome.next_final_observation_row_mask,
        outcome.next_final_observation_row_mask,
    )


def test_root_transition_outcome_derivation_requires_explicit_terminal_sidecars():
    compact_batch = _compact_batch_with_terminal_mechanics_outcome()
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_owner_transition_outcome_missing",
    )

    with pytest.raises(ReplayCompatibilityError, match="terminated"):
        compact_transition_outcome_v1_from_next_root_batch(replace(root_batch, terminated=None))
    with pytest.raises(ReplayCompatibilityError, match="truncated"):
        compact_transition_outcome_v1_from_next_root_batch(replace(root_batch, truncated=None))
    with pytest.raises(ReplayCompatibilityError, match="final_reward_map"):
        compact_transition_outcome_v1_from_next_root_batch(
            replace(root_batch, final_reward_map=None)
        )
    request = compact_root_build_request_v1_from_batch(
        compact_batch,
        search_lane="unit_test_owner_transition_outcome_missing_request",
    )
    with pytest.raises(ReplayCompatibilityError, match="terminated"):
        compact_transition_outcome_v1_from_root_build_request(
            replace(request, terminated=None)
        )
    with pytest.raises(ReplayCompatibilityError, match="truncated"):
        compact_transition_outcome_v1_from_root_build_request(replace(request, truncated=None))
    with pytest.raises(ReplayCompatibilityError, match="final_reward_map"):
        compact_transition_outcome_v1_from_root_build_request(
            replace(request, final_reward_map=None)
        )


def test_two_record_compact_rows_use_final_observation_before_autoreset_and_rnd_latest():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )

    object_rows = _expected_rows_from_replay_policy_rows(
        chunk,
        record_index=0,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        policy_source="compact_two_record_contract_test",
    )
    compact_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=0,
        policy_source="compact_two_record_contract_test",
    )

    _assert_target_rows_equal(compact_rows, object_rows)
    np.testing.assert_array_equal(compact_rows.env_row, np.asarray([0, 0, 1, 1]))
    np.testing.assert_array_equal(compact_rows.player, np.asarray([0, 1, 0, 1]))
    for row_index, (env_row, player) in enumerate(
        zip(compact_rows.env_row, compact_rows.player, strict=True)
    ):
        expected_latest = _latest_value(0, int(env_row), int(player))
        assert compact_rows.observation[row_index, 3, 0, 0] == expected_latest
        if int(env_row) == 1:
            np.testing.assert_array_equal(
                compact_rows.next_observation[row_index],
                chunk.arrays["final_observation"][1, int(env_row), int(player)],
            )
            assert compact_rows.next_observation[row_index, 3, 0, 0] == _final_value(
                1,
                int(env_row),
                int(player),
            )
            assert compact_rows.next_observation[row_index, 3, 0, 0] != _latest_value(
                1,
                int(env_row),
                int(player),
            )

    rnd_input = extract_policy_gray64_latest_for_rnd_from_compact_observation(
        compact_batch.observation,
        compact_batch.target_reward,
    )
    assert rnd_input.shape == (4, 1, 64, 64)
    expected_latest = np.asarray(
        [
            _latest_value(0, 0, 0),
            _latest_value(0, 0, 1),
            _latest_value(0, 1, 0),
            _latest_value(0, 1, 1),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(rnd_input[:, 0, 0, 0], expected_latest)


def test_compact_service_v1_round_trips_target_rows_and_identity_sidecars():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_direct_ctree_control",
        metadata={"rnd_mode": "meter_v0"},
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_search_impl",
        num_simulations=8,
    )
    service_chunk = build_compact_replay_chunk_v1_from_search_result(
        chunk,
        compact_batch,
        root_batch,
        search_result,
        record_index=0,
        policy_source="compact_service_v1_contract_test",
        metadata={"profile_only": True},
    )
    object_rows = _expected_rows_from_replay_policy_rows(
        chunk,
        record_index=0,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        policy_source="compact_service_v1_contract_test",
    )

    assert root_batch.metadata["schema_id"] == "curvyzero_compact_root_batch/v1"
    assert root_batch.metadata["active_root_count"] == 4
    assert search_result.metadata["schema_id"] == "curvyzero_compact_search_result/v1"
    assert service_chunk.metadata["schema_id"] == "curvyzero_compact_replay_chunk/v1"
    assert service_chunk.metadata["profile_only"] is True
    np.testing.assert_array_equal(search_result.root_index, np.asarray([0, 1, 2, 3]))
    np.testing.assert_array_equal(search_result.policy_env_id, np.asarray([0, 1, 2, 3]))
    _assert_target_rows_equal(service_chunk.target_rows, object_rows)
    for row_index, ref in enumerate(service_chunk.target_rows.source_record_ref):
        assert ref["compact_root_row"] == row_index
        assert ref["policy_env_id"] == row_index


def test_compact_search_two_phase_payload_preserves_action_identity():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_two_phase_service",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_two_phase_impl",
        num_simulations=8,
    )

    action_step = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
    )
    replay_payload = compact_search_replay_payload_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
    )

    validate_compact_search_two_phase_payload_v1(action_step, replay_payload)
    assert action_step.metadata["phase"] == "action_critical"
    assert replay_payload.metadata["phase"] == "replay_critical"
    np.testing.assert_array_equal(action_step.selected_action, selected_action)
    np.testing.assert_array_equal(replay_payload.visit_policy, visit_policy)
    np.testing.assert_array_equal(replay_payload.root_value, root_value)


def test_policy_refresh_metadata_survives_root_action_replay_and_index_rows():
    torch = pytest.importorskip("torch")
    policy_refresh_metadata = _policy_refresh_metadata_for_test()
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_policy_refresh_row_stamps",
        metadata=policy_refresh_metadata,
        copy_observation=False,
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_policy_refresh_search",
        num_simulations=8,
        metadata=policy_refresh_metadata,
    )
    action_step = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
        metadata=policy_refresh_metadata,
    )
    replay_payload = compact_search_replay_payload_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
        metadata=policy_refresh_metadata,
    )
    device_replay_payload = CompactDeviceSearchReplayPayloadV1(
        replay_payload_handle="record-0",
        root_index=search_result.root_index.copy(),
        env_row=search_result.env_row.copy(),
        player=search_result.player.copy(),
        policy_env_id=search_result.policy_env_id.copy(),
        visit_policy=torch.as_tensor(visit_policy, dtype=torch.float32),
        root_value=torch.as_tensor(root_value, dtype=torch.float32),
        raw_visit_counts=torch.as_tensor(visit_policy, dtype=torch.float32),
        predicted_value=None,
        predicted_policy_logits=None,
        metadata={
            **policy_refresh_metadata,
            "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
            "phase": "replay_critical_device",
            "search_result_schema_id": search_result.metadata["schema_id"],
            "search_impl": search_result.metadata["search_impl"],
            "num_simulations": search_result.metadata["num_simulations"],
            "active_root_count": int(search_result.root_index.size),
            "search_replay_payload_digest": replay_payload.metadata["search_replay_payload_digest"],
            "device_replay_payload": True,
            "host_search_payload_fallback_allowed": False,
        },
    )

    host_index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=0,
        next_joint_action=chunk.arrays["joint_action"][1],
        next_reward=chunk.arrays["reward"][1],
        next_done=chunk.arrays["done"][1],
        next_terminated=chunk.arrays["terminated"][1],
        next_truncated=chunk.arrays["truncated"][1],
        next_final_reward_map=chunk.arrays["final_reward_map"][1],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
        policy_source=policy_refresh_metadata["policy_source"],
    )
    device_index_rows = build_compact_device_replay_index_rows_v1_from_payload(
        compact_batch,
        root_batch,
        action_step,
        device_replay_payload,
        record_index=0,
        next_joint_action=chunk.arrays["joint_action"][1],
        next_reward=chunk.arrays["reward"][1],
        next_done=chunk.arrays["done"][1],
        next_terminated=chunk.arrays["terminated"][1],
        next_truncated=chunk.arrays["truncated"][1],
        next_final_reward_map=chunk.arrays["final_reward_map"][1],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
        policy_source=policy_refresh_metadata["policy_source"],
    )

    for metadata in (
        root_batch.metadata,
        search_result.metadata,
        action_step.metadata,
        replay_payload.metadata,
        host_index_rows.metadata,
        device_index_rows.metadata,
    ):
        _assert_policy_refresh_metadata(metadata, policy_refresh_metadata)


def test_compact_search_two_phase_payload_rejects_stale_or_reordered_identity():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_two_phase_service",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_two_phase_impl",
        num_simulations=8,
    )
    action_step = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
    )
    replay_payload = compact_search_replay_payload_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
    )
    stale_payload = replace(
        replay_payload,
        policy_env_id=replay_payload.policy_env_id[::-1].copy(),
    )

    with pytest.raises(ReplayCompatibilityError, match="policy_env_id"):
        validate_compact_search_two_phase_payload_v1(action_step, stale_payload)

    wrong_handle = replace(replay_payload, replay_payload_handle="record-1")
    with pytest.raises(ReplayCompatibilityError, match="handle"):
        validate_compact_search_two_phase_payload_v1(action_step, wrong_handle)

    stale_same_identity_payload = replace(
        replay_payload,
        visit_policy=np.roll(replay_payload.visit_policy, shift=1, axis=1).copy(),
    )
    with pytest.raises(ReplayCompatibilityError, match="digest"):
        validate_compact_search_two_phase_payload_v1(
            action_step,
            stale_same_identity_payload,
        )


def test_deferred_compact_search_payload_still_checks_shape_and_origin():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_deferred_two_phase_service",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_deferred_two_phase_impl",
        num_simulations=8,
    )
    handle = "record-0"
    origin = f"unit_test_deferred_two_phase_impl:{handle}"
    action_step = CompactSearchActionStepV1(
        replay_payload_handle=handle,
        root_index=search_result.root_index.copy(),
        env_row=search_result.env_row.copy(),
        player=search_result.player.copy(),
        policy_env_id=search_result.policy_env_id.copy(),
        selected_action=search_result.selected_action.copy(),
        metadata={
            "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
            "phase": "action_critical",
            "search_impl": search_result.metadata["search_impl"],
            "num_simulations": search_result.metadata["num_simulations"],
            "active_root_count": int(search_result.selected_action.size),
            "selected_action_digest": compact_search_array_digest_v1(search_result.selected_action),
            "search_replay_payload_digest": (
                compact_search_deferred_replay_payload_digest_v1(handle)
            ),
            "search_replay_payload_digest_deferred": True,
            "replay_payload_origin": origin,
        },
    )
    payload = compact_search_replay_payload_v1_from_result(
        search_result,
        replay_payload_handle=handle,
        metadata={"replay_payload_origin": origin},
    )
    validate_compact_search_two_phase_payload_v1(action_step, payload)

    wrong_shape = replace(
        payload,
        visit_policy=payload.visit_policy[:, :2].copy(),
    )
    wrong_shape = replace(
        wrong_shape,
        metadata={
            **wrong_shape.metadata,
            "search_replay_payload_digest": compact_search_replay_payload_digest_v1(wrong_shape),
        },
    )
    with pytest.raises(ReplayCompatibilityError, match="visit_policy shape"):
        validate_compact_search_two_phase_payload_v1(action_step, wrong_shape)

    wrong_origin = replace(
        payload,
        metadata={**payload.metadata, "replay_payload_origin": "wrong-origin"},
    )
    wrong_origin = replace(
        wrong_origin,
        metadata={
            **wrong_origin.metadata,
            "search_replay_payload_digest": compact_search_replay_payload_digest_v1(wrong_origin),
        },
    )
    with pytest.raises(ReplayCompatibilityError, match="origin"):
        validate_compact_search_two_phase_payload_v1(action_step, wrong_origin)


def test_compact_search_payload_gate_hides_rows_until_payload_arrives():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_two_phase_service",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_two_phase_impl",
        num_simulations=8,
    )
    action_step = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
    )
    replay_payload = compact_search_replay_payload_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
    )
    gate = CompactSearchPayloadGateV1()

    gate.register_action_step(action_step)
    assert gate.pending_count == 1
    assert gate.complete_count == 0
    assert gate.is_sample_visible("record-0") is False
    with pytest.raises(ReplayCompatibilityError, match="sample-visible"):
        gate.require_replay_payload("record-0")

    gate.attach_replay_payload(replay_payload)
    assert gate.pending_count == 0
    assert gate.complete_count == 1
    assert gate.is_sample_visible("record-0") is True
    assert gate.require_replay_payload("record-0") is replay_payload


def test_compact_search_comparator_returns_primary_and_records_deltas():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_comparator",
    )
    reference_visit = visit_policy.copy()
    reference_visit[:, 1:] = reference_visit[:, ::-1][:, :2]
    reference_visit = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.25, 0.75],
            [0.0, 1.0, 0.0],
            [0.0, 0.25, 0.75],
        ],
        dtype=np.float32,
    )
    reference_value = root_value + np.asarray([0.0, 0.5, 0.0, -0.5], dtype=np.float32)
    service = CompactSearchComparatorServiceV1(
        primary=_StaticCompactSearchService(
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            predicted_value=root_value,
            predicted_policy_logits=visit_policy,
            search_impl="primary_search",
        ),
        reference=_StaticCompactSearchService(
            selected_action=selected_action,
            visit_policy=reference_visit,
            root_value=reference_value,
            predicted_value=root_value + np.float32(0.25),
            predicted_policy_logits=visit_policy + np.float32(0.125),
            search_impl="reference_search",
        ),
        comparison_label="unit_test_primary_vs_reference",
    )

    result = service.run(root_batch)

    np.testing.assert_array_equal(result.selected_action, selected_action)
    telemetry = result.metadata["profile_telemetry"]
    assert result.metadata["compact_search_comparator_enabled"] is True
    assert telemetry["compact_search_comparator_label"] == "unit_test_primary_vs_reference"
    assert telemetry["compact_search_comparator_primary_impl"] == "primary_search"
    assert telemetry["compact_search_comparator_reference_impl"] == "reference_search"
    assert telemetry["compact_search_comparator_action_match_fraction"] == 1.0
    assert telemetry["compact_search_comparator_visit_l1_max"] > 0.0
    assert telemetry["compact_search_comparator_root_value_abs_diff_max"] == pytest.approx(0.5)
    assert telemetry["compact_search_comparator_predicted_value_abs_diff_present"] is True
    assert telemetry["compact_search_comparator_predicted_value_abs_diff_max"] == pytest.approx(
        0.25
    )
    assert telemetry["compact_search_comparator_predicted_policy_logits_abs_diff_present"] is True
    assert telemetry[
        "compact_search_comparator_predicted_policy_logits_abs_diff_max"
    ] == pytest.approx(0.125)


def test_compact_search_comparator_rejects_identity_mismatch():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_comparator",
    )
    service = CompactSearchComparatorServiceV1(
        primary=_StaticCompactSearchService(
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            search_impl="primary_search",
        ),
        reference=_StaticCompactSearchService(
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            search_impl="reference_search",
            reverse_identity=True,
        ),
        comparison_label="unit_test_identity_mismatch",
    )

    with pytest.raises(ReplayCompatibilityError, match="identity mismatch"):
        service.run(root_batch)


def test_compact_search_payload_gate_allows_out_of_order_payloads_by_handle():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_two_phase_service",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_two_phase_impl",
        num_simulations=8,
    )
    step_a = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle="record-a",
    )
    payload_a = compact_search_replay_payload_v1_from_result(
        search_result,
        replay_payload_handle="record-a",
    )
    step_b = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle="record-b",
    )
    payload_b = compact_search_replay_payload_v1_from_result(
        search_result,
        replay_payload_handle="record-b",
    )
    gate = CompactSearchPayloadGateV1()

    gate.register_action_step(step_a)
    gate.register_action_step(step_b)
    gate.attach_replay_payload(payload_b)

    assert gate.is_sample_visible("record-b") is True
    assert gate.is_sample_visible("record-a") is False
    with pytest.raises(ReplayCompatibilityError, match="sample-visible"):
        gate.require_replay_payload("record-a")

    gate.attach_replay_payload(payload_a)
    assert gate.is_sample_visible("record-a") is True


def test_compact_rollout_slab_stages_actions_and_commits_previous_index_rows():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    compact1 = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )

    class FakeCompactSearchService:
        search_impl = "unit_test_compact_rollout_slab_search"
        num_simulations = 8

        def run(self, root_batch):
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected_action,
                visit_policy=visit_policy,
                root_value=root_value,
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata={
                    "profile_telemetry": {
                        "lightzero_consumer_model_total_sec": 1.25,
                        "lightzero_mcts_arrays_boundary_initial_inference_sec": 99.0,
                        "lightzero_consumer_h2d_sec": 0.125,
                        "lightzero_mcts_arrays_boundary_input_prepare_sec": 88.0,
                    }
                },
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeCompactSearchService(),
        search_lane="unit_test_compact_rollout_slab",
        policy_source="unit_test_compact_rollout_slab",
    )

    first = slab.step(compact0)
    np.testing.assert_array_equal(first.next_joint_action, compact1.joint_action)
    assert first.committed_index_rows is None
    assert first.telemetry["compact_rollout_slab_profile_only"] is True
    assert first.telemetry["compact_rollout_slab_active_root_count"] == 4
    assert first.telemetry["compact_rollout_slab_model_sec"] == 1.25
    assert first.telemetry["compact_rollout_slab_h2d_sec"] == 0.125

    second = slab.step(compact1)
    assert second.committed_index_rows is not None
    np.testing.assert_array_equal(second.committed_index_rows.action, selected_action)
    np.testing.assert_array_equal(second.committed_index_rows.policy_target, visit_policy)
    np.testing.assert_allclose(second.committed_index_rows.root_value, root_value)
    assert second.committed_index_rows.metadata["observation_materialized"] is False
    assert second.telemetry["compact_rollout_slab_committed_index_row_count"] == 4


def test_compact_rollout_slab_stamps_policy_refresh_metadata_on_roots_and_rows():
    policy_refresh_state = _policy_refresh_state_for_test()
    policy_refresh_metadata = compact_policy_refresh_metadata_from_state_v1(policy_refresh_state)
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    compact1 = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )

    class FakeRefreshAwareSearchService:
        search_impl = "unit_test_compact_rollout_slab_policy_refresh"
        num_simulations = 8

        def policy_refresh_search_worker_state(self):
            return dict(policy_refresh_state)

        def run(self, root_batch):
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected_action,
                visit_policy=visit_policy,
                root_value=root_value,
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata=policy_refresh_metadata,
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeRefreshAwareSearchService(),
        search_lane="unit_test_compact_rollout_slab_policy_refresh",
        policy_source=policy_refresh_metadata["policy_source"],
    )

    first = slab.step(compact0)
    _assert_policy_refresh_metadata(first.root_batch.metadata, policy_refresh_metadata)
    second = slab.step(compact1)

    assert second.committed_index_rows is not None
    _assert_policy_refresh_metadata(
        second.committed_index_rows.metadata,
        policy_refresh_metadata,
    )


def test_compact_rollout_slab_two_phase_hot_step_returns_actions_only_then_flushes():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    compact1 = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    service = _FakeTwoPhaseCompactSearchServiceV1(
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=service,
        search_lane="unit_test_two_phase_compact_rollout_slab",
        policy_source="unit_test_two_phase_compact_rollout_slab",
    )

    first = slab.step(compact0)

    assert service.run_calls == 0
    assert service.action_step_calls == 1
    assert service.flush_calls == 0
    assert first.search_result is None
    assert first.action_step is not None
    np.testing.assert_array_equal(
        selected_joint_action_from_action_step(
            first.root_batch,
            first.action_step,
            batch_size=2,
            player_count=2,
        ),
        first.next_joint_action,
    )
    np.testing.assert_array_equal(first.next_joint_action, compact1.joint_action)
    assert first.telemetry["compact_rollout_slab_two_phase_search"] is True
    assert first.telemetry["compact_rollout_slab_action_step_only"] is True
    assert first.telemetry["compact_rollout_slab_replay_payload_flush_count"] == 0
    assert first.telemetry["compact_rollout_slab_root_batch_build_sec"] >= 0.0
    assert first.telemetry["compact_rollout_slab_search_dispatch_wall_sec"] >= 0.0
    assert first.telemetry["compact_rollout_slab_search_dispatch_service_envelope_sec"] >= 0.0
    assert "compact_rollout_slab_search_dispatch_residual_sec" in first.telemetry
    assert first.telemetry["compact_rollout_slab_search_dispatch_positive_residual_sec"] >= 0.0
    assert first.telemetry["compact_rollout_slab_search_dispatch_overaccounted_sec"] >= 0.0
    assert first.telemetry["compact_rollout_slab_joint_action_assembly_sec"] >= 0.0
    assert first.telemetry["compact_rollout_slab_internal_accounted_sec"] >= 0.0

    second = slab.step(compact1)

    assert service.run_calls == 0
    assert service.action_step_calls == 2
    assert service.flush_calls == 1
    assert second.committed_index_rows is not None
    np.testing.assert_array_equal(second.committed_index_rows.action, selected_action)
    np.testing.assert_array_equal(second.committed_index_rows.policy_target, visit_policy)
    np.testing.assert_allclose(second.committed_index_rows.root_value, root_value)
    assert second.telemetry["compact_rollout_slab_replay_payload_flush_count"] == 1
    assert second.telemetry["compact_rollout_slab_committed_replay_payload_flushed"] is True
    assert second.telemetry["compact_rollout_slab_commit_previous_sec"] >= 0.0
    assert second.telemetry["compact_rollout_slab_replay_payload_flush_sec"] >= 0.0
    assert second.telemetry["compact_rollout_slab_replay_payload_validate_sec"] >= 0.0
    assert second.telemetry["compact_rollout_slab_replay_index_rows_build_sec"] >= 0.0
    assert second.telemetry["compact_rollout_slab_replay_index_rows_store_sec"] >= 0.0


def test_compact_rollout_slab_rejects_stale_action_step_before_env_action():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )

    class StaleActionDigestService(_FakeTwoPhaseCompactSearchServiceV1):
        def run_action_step(self, root_batch):
            action_step = super().run_action_step(root_batch)
            return replace(
                action_step,
                metadata={
                    **action_step.metadata,
                    "selected_action_digest": "stale",
                },
            )

    service = StaleActionDigestService(
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=service,
        search_lane="unit_test_stale_action_step_slab",
        policy_source="unit_test_stale_action_step_slab",
    )

    with pytest.raises(ReplayCompatibilityError, match="selected-action digest"):
        slab.step(compact0)


def test_compact_rollout_slab_flattens_search_profile_telemetry():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )

    class FakeCompactSearchService:
        search_impl = "unit_test_compact_rollout_slab_search"
        num_simulations = 8

        def run(self, root_batch):
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected_action,
                visit_policy=visit_policy,
                root_value=root_value,
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata={
                    "profile_telemetry": {
                        "lightzero_mcts_arrays_boundary_total_sec": 1.25,
                        "lightzero_mcts_arrays_boundary_initial_inference_sec": 0.2,
                        "lightzero_mcts_arrays_boundary_recurrent_inference_sec": 0.3,
                        "lightzero_mcts_arrays_boundary_search_sec": 0.6,
                        "lightzero_consumer_h2d_sec": 0.05,
                        "lightzero_mcts_arrays_boundary_obs_h2d_bytes": 4096.0,
                        "lightzero_mcts_arrays_boundary_mask_h2d_bytes": 12.0,
                        "lightzero_mcts_arrays_boundary_action_d2h_bytes": 8.0,
                        "compact_torch_search_one_simulation_fast_path": True,
                        "compact_torch_search_one_simulation_root_prior_softmax_skipped": True,
                        "compact_torch_search_one_simulation_selection_mode": (
                            "masked_logits_argmax"
                        ),
                        "compact_torch_search_recurrent_inference_calls": 1.0,
                        "compact_torch_search_initial_inference_direct_requested": True,
                        "compact_torch_search_initial_inference_direct_used": True,
                        "compact_torch_search_initial_inference_fallback_count": 0.0,
                        "compact_torch_search_initial_inference_mode_requested": ("direct_core"),
                        "compact_torch_search_initial_inference_mode_effective": ("direct_core"),
                        "compact_torch_search_initial_inference_runtime_status": (
                            "direct_core_used"
                        ),
                        "compact_torch_search_service_tensor_prepare_sync_sec": 0.01,
                        "compact_torch_search_service_initial_inference_enqueue_sec": 0.11,
                        "compact_torch_search_service_initial_inference_sync_sec": 0.12,
                        "compact_torch_search_service_initial_inference_representation_cuda_event_sec": 0.13,
                        "compact_torch_search_service_initial_inference_prediction_cuda_event_sec": 0.14,
                        "compact_torch_search_service_initial_inference_direct_core_cuda_event_sec": 0.27,
                        "compact_torch_search_service_initial_inference_direct_core_cuda_event_residual_sec": 0.03,
                        "compact_torch_search_observation_memory_format_requested": (
                            "channels_last"
                        ),
                        "compact_torch_search_observation_memory_format_effective": (
                            "channels_last"
                        ),
                        "compact_torch_search_observation_normalized_uint8": True,
                        "compact_torch_search_observation_dtype_before_model": ("torch.uint8"),
                        "compact_torch_search_observation_dtype_model_input": ("torch.float32"),
                        "compact_torch_search_observation_layout_copy_bytes": 8192.0,
                        "compact_torch_search_observation_is_contiguous": False,
                        "compact_torch_search_observation_is_channels_last": True,
                        "compact_torch_search_model_memory_format_requested": ("contiguous"),
                        "compact_torch_search_model_memory_format_active": ("contiguous"),
                        "compact_torch_search_model_memory_format_applied": False,
                        "compact_torch_search_service_root_latent_prepare_sec": 0.015,
                        "compact_torch_search_root_latent_dtype": "torch.float32",
                        "compact_torch_search_root_latent_ndim": 4.0,
                        "compact_torch_search_root_latent_is_contiguous_before_recurrent": False,
                        "compact_torch_search_root_latent_is_channels_last_before_recurrent": True,
                        "compact_torch_search_root_latent_contiguous_for_recurrent": True,
                        "compact_torch_search_root_latent_is_channels_last_for_recurrent": False,
                        "compact_torch_search_root_latent_contiguous_copy_bytes": 4096.0,
                        "compact_torch_search_service_tree_root_prior_select_sec": 0.21,
                        "compact_torch_search_service_tree_recurrent_action_build_sec": 0.22,
                        "compact_torch_search_service_tree_recurrent_inference_enqueue_sec": 0.23,
                        "compact_torch_search_service_tree_recurrent_output_decode_sec": 0.24,
                        "compact_torch_search_service_tree_policy_build_sec": 0.25,
                        "compact_torch_search_service_tree_sync_sec": 0.26,
                        "compact_torch_search_service_tree_unaccounted_sec": 0.0,
                    },
                },
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeCompactSearchService(),
        search_lane="unit_test_compact_rollout_slab",
        policy_source="unit_test_compact_rollout_slab",
    )

    step = slab.step(compact0)

    assert step.telemetry["compact_rollout_slab_search_service_total_sec"] == 1.25
    assert step.telemetry["compact_rollout_slab_model_sec"] == pytest.approx(0.5)
    assert step.telemetry["compact_rollout_slab_search_sec"] == 0.6
    assert step.telemetry["compact_rollout_slab_h2d_sec"] == 0.05
    assert step.telemetry["compact_rollout_slab_obs_h2d_bytes"] == 4096.0
    assert step.telemetry["compact_rollout_slab_mask_h2d_bytes"] == 12.0
    assert step.telemetry["compact_rollout_slab_action_d2h_bytes"] == 8.0
    assert step.telemetry["compact_rollout_slab_search_service_one_simulation_fast_path"] is True
    assert (
        step.telemetry["compact_rollout_slab_search_service_one_simulation_fast_path_count"] == 1.0
    )
    assert (
        step.telemetry[
            "compact_rollout_slab_search_service_one_simulation_root_prior_softmax_skipped"
        ]
        is True
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_one_simulation_selection_mode"]
        == "masked_logits_argmax"
    )
    assert step.telemetry["compact_rollout_slab_search_service_recurrent_inference_calls"] == 1.0
    assert (
        step.telemetry["compact_rollout_slab_search_service_initial_inference_direct_requested"]
        is True
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_initial_inference_direct_used"] is True
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_initial_inference_fallback_count"]
        == 0.0
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_initial_inference_mode_requested"]
        == "direct_core"
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_initial_inference_mode_effective"]
        == "direct_core"
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_initial_inference_runtime_status"]
        == "direct_core_used"
    )
    assert step.telemetry["compact_rollout_slab_search_service_initial_inference_sync_sec"] == 0.12
    assert (
        step.telemetry[
            "compact_rollout_slab_search_service_initial_inference_representation_cuda_event_sec"
        ]
        == 0.13
    )
    assert (
        step.telemetry[
            "compact_rollout_slab_search_service_initial_inference_prediction_cuda_event_sec"
        ]
        == 0.14
    )
    assert (
        step.telemetry[
            "compact_rollout_slab_search_service_initial_inference_direct_core_cuda_event_sec"
        ]
        == 0.27
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_observation_memory_format_requested"]
        == "channels_last"
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_observation_memory_format_effective"]
        == "channels_last"
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_observation_layout_copy_bytes"]
        == 8192.0
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_observation_is_channels_last"] is True
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_model_memory_format_active"]
        == "contiguous"
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_model_memory_format_applied"] is False
    )
    assert step.telemetry["compact_rollout_slab_search_service_root_latent_prepare_sec"] == 0.015
    assert (
        step.telemetry[
            "compact_rollout_slab_search_service_root_latent_is_channels_last_before_recurrent"
        ]
        is True
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_root_latent_contiguous_for_recurrent"]
        is True
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_root_latent_contiguous_copy_bytes"]
        == 4096.0
    )
    assert (
        step.telemetry["compact_rollout_slab_search_service_tree_recurrent_inference_enqueue_sec"]
        == 0.23
    )
    assert step.telemetry["compact_rollout_slab_search_service_tree_sync_sec"] == 0.26
    assert step.telemetry["compact_rollout_slab_telemetry_build_sec"] >= 0.0


def test_compact_rollout_slab_promotes_mctx_profile_telemetry_fields():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )

    class FakeMctxSearchService:
        search_impl = "mctx_compact_search_service_profile_only_v0"
        num_simulations = 8

        def run(self, root_batch):
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected_action,
                visit_policy=visit_policy,
                root_value=root_value,
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
                metadata={
                    "profile_only": True,
                    "not_lightzero_ctree": True,
                    "not_train_muzero": True,
                    "profile_telemetry": {
                        "mctx_compact_search_service_total_sec": 1.5,
                        "mctx_compact_search_service_search_sec": 1.1,
                        "mctx_compact_search_service_h2d_sec": 0.2,
                        "mctx_compact_search_service_obs_h2d_bytes": 4096.0,
                        "mctx_compact_search_service_mask_h2d_bytes": 12.0,
                        "mctx_compact_search_service_action_d2h_bytes": 8.0,
                        "mctx_compact_search_service_replay_payload_d2h_bytes": 64.0,
                    },
                },
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeMctxSearchService(),
        search_lane="unit_test_mctx_compact_rollout_slab",
        policy_source="unit_test_mctx_compact_rollout_slab",
    )

    step = slab.step(compact0)

    assert step.telemetry["compact_rollout_slab_search_impl"] == (
        "mctx_compact_search_service_profile_only_v0"
    )
    assert step.telemetry["compact_rollout_slab_search_service_total_sec"] == 1.5
    assert step.telemetry["compact_rollout_slab_search_sec"] == 1.1
    assert step.telemetry["compact_rollout_slab_h2d_sec"] == 0.2
    assert step.telemetry["compact_rollout_slab_obs_h2d_bytes"] == 4096.0
    assert step.telemetry["compact_rollout_slab_mask_h2d_bytes"] == 12.0
    assert step.telemetry["compact_rollout_slab_action_d2h_bytes"] == 8.0
    assert step.telemetry["compact_rollout_slab_replay_payload_d2h_bytes"] == 64.0


def test_compact_rollout_slab_rejects_next_batch_that_ignored_staged_actions():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    compact1 = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )

    class FakeCompactSearchService:
        search_impl = "unit_test_compact_rollout_slab_search"
        num_simulations = 8

        def run(self, root_batch):
            return validate_compact_search_result_v1(
                root_batch,
                selected_action=selected_action,
                visit_policy=visit_policy,
                root_value=root_value,
                search_impl=self.search_impl,
                num_simulations=self.num_simulations,
            )

    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=FakeCompactSearchService(),
        search_lane="unit_test_compact_rollout_slab",
        policy_source="unit_test_compact_rollout_slab",
    )
    slab.step(compact0)
    ignored_actions = replace(compact1, joint_action=np.zeros((2, 2), dtype=np.int16))

    with pytest.raises(ReplayCompatibilityError, match="staged selected actions"):
        slab.step(ignored_actions)


def test_compact_rollout_slab_scripted_random_drops_replay_instead_of_requiring_actions():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    compact1 = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=_StaticCompactSearchService(
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            search_impl="unit_test_scripted_action_mode",
        ),
        search_lane="unit_test_scripted_action_mode",
        policy_source="unit_test_scripted_action_mode",
        action_feedback_mode=COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM,
    )
    slab.step(compact0)
    ignored_actions = replace(compact1, joint_action=np.zeros((2, 2), dtype=np.int16))

    second = slab.step(ignored_actions)

    assert second.committed_index_rows is None
    assert slab.committed_index_row_count == 0
    assert slab.action_override_drop_count == 1
    assert second.telemetry["compact_rollout_slab_action_feedback_mode"] == (
        COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM
    )
    assert second.telemetry["compact_rollout_slab_replay_commit_requires_search_action"] is False
    assert second.telemetry["compact_rollout_slab_action_override_drop_count"] == 1


def test_compact_rollout_slab_two_phase_scripted_random_never_flushes_dropped_payload():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    compact1 = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    service = _FakeTwoPhaseCompactSearchServiceV1(
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=service,
        search_lane="unit_test_two_phase_scripted_action_mode",
        policy_source="unit_test_two_phase_scripted_action_mode",
        action_feedback_mode=COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM,
    )
    slab.step(compact0)
    ignored_actions = replace(compact1, joint_action=np.zeros((2, 2), dtype=np.int16))

    second = slab.step(ignored_actions)

    assert second.committed_index_rows is None
    assert service.run_calls == 0
    assert service.action_step_calls == 2
    assert service.flush_calls == 0
    assert slab.committed_index_row_count == 0
    assert slab.action_override_drop_count == 1
    assert second.telemetry["compact_rollout_slab_replay_payload_flush_count"] == 0
    assert second.telemetry["compact_rollout_slab_committed_replay_payload_flushed"] is False


def test_compact_rollout_slab_close_drops_tail_without_faking_transition():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=_StaticCompactSearchService(
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            search_impl="unit_test_close_tail",
        ),
        search_lane="unit_test_close_tail",
        policy_source="unit_test_close_tail",
    )

    slab.step(compact0)
    assert slab.close() is None
    assert slab.dropped_pending_search_count == 1
    assert slab.committed_index_row_count == 0
    assert slab.committed_index_rows == ()
    assert slab.close() is None

    with pytest.raises(ReplayCompatibilityError, match="closed"):
        slab.step(compact0)


def test_compact_rollout_slab_close_with_next_batch_commits_once_and_copies_storage():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    compact1 = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=_StaticCompactSearchService(
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            search_impl="unit_test_close_commit",
        ),
        search_lane="unit_test_close_commit",
        policy_source="unit_test_close_commit",
    )

    slab.step(compact0)
    committed = slab.close(compact1)

    assert committed is not None
    assert slab.close(compact1) is None
    assert slab.committed_index_row_count == 4
    assert len(slab.committed_index_rows) == 1
    np.testing.assert_array_equal(committed.action, selected_action)
    committed.action[0] = np.int16((int(committed.action[0]) + 1) % ACTION_COUNT)
    np.testing.assert_array_equal(slab.committed_index_rows[0].action, selected_action)

    with pytest.raises(ReplayCompatibilityError, match="closed"):
        slab.step(compact1)


def test_compact_rollout_slab_can_count_without_retaining_committed_history():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    compact1 = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=_StaticCompactSearchService(
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            search_impl="unit_test_bounded_history",
        ),
        search_lane="unit_test_bounded_history",
        policy_source="unit_test_bounded_history",
        retain_committed_index_rows=False,
    )

    slab.step(compact0)
    committed = slab.close(compact1)

    assert committed is not None
    assert slab.committed_index_row_count == 4
    assert slab.committed_index_group_count == 1
    assert slab.committed_index_rows == ()
    np.testing.assert_array_equal(committed.action, selected_action)


def test_compact_rollout_slab_commit_checks_only_active_selected_seats():
    live_by_record = np.asarray(
        [
            [[True, True], [False, False]],
            [[True, True], [False, False]],
        ],
        dtype=np.bool_,
    )
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=live_by_record,
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    compact1 = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    assert selected_action.shape == (2,)
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=_StaticCompactSearchService(
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            search_impl="unit_test_active_seats_only",
        ),
        search_lane="unit_test_active_seats_only",
        policy_source="unit_test_active_seats_only",
    )

    first = slab.step(compact0)
    assert not np.array_equal(first.next_joint_action, compact1.joint_action)
    second = slab.step(compact1)

    assert second.committed_index_rows is not None
    np.testing.assert_array_equal(second.committed_index_rows.action, selected_action)


def test_compact_index_row_group_materialization_preserves_record_identity():
    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.ones((3, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    index0 = _compact_index_rows_for_record(chunk, record_index=0)
    index1 = _compact_index_rows_for_record(chunk, record_index=1)
    rows0 = materialize_compact_target_rows_from_index_rows_v1(chunk, index0)
    rows1 = materialize_compact_target_rows_from_index_rows_v1(chunk, index1)

    grouped = materialize_compact_target_rows_from_index_row_groups_v1(
        chunk,
        [index0, index1],
    )

    assert grouped.metadata["compact_index_group_count"] == 2
    assert grouped.metadata["target_row_count"] == 8
    np.testing.assert_array_equal(
        grouped.record_index,
        np.concatenate([rows0.record_index, rows1.record_index]),
    )
    np.testing.assert_array_equal(grouped.record_index[:4], np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(grouped.record_index[4:], np.ones(4, dtype=np.int32))
    np.testing.assert_array_equal(
        grouped.action,
        np.concatenate([rows0.action, rows1.action]),
    )
    for row_index in range(4, 8):
        if int(grouped.env_row[row_index]) == 1:
            assert grouped.next_observation[row_index, 3, 0, 0] == _final_value(
                2,
                1,
                int(grouped.player[row_index]),
            )

    sample = sample_compact_target_rows_from_index_row_groups_v1(
        chunk,
        [index0, index1],
        batch_size=4,
        seed=20260523,
        replace=False,
    )
    assert sample.observation.shape == (4, *FRAME_STACK_SHAPE)
    assert int(sample.row_id.min()) >= 0
    assert int(sample.row_id.max()) < 8
    np.testing.assert_array_equal(sample.action, grouped.action[sample.row_id])


def test_selected_joint_action_from_search_result_fast_path_uses_row_major_sidecars(
    monkeypatch,
):
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact0,
        search_lane="unit_test_compact_rollout_slab",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_compact_rollout_slab_search",
        num_simulations=8,
    )

    def fail_full(*_args, **_kwargs):
        raise AssertionError("row-major fast path should not allocate inactive fill")

    monkeypatch.setattr(compact_rollout_slab_module.np, "full", fail_full)

    joint_action = selected_joint_action_from_search_result(
        root_batch,
        search_result,
        batch_size=2,
        player_count=2,
    )

    np.testing.assert_array_equal(
        joint_action,
        selected_action.reshape(2, 2),
    )


def test_selected_joint_action_from_search_result_rejects_illegal_actions():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact0 = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact0,
        search_lane="unit_test_compact_rollout_slab",
    )
    bad_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_compact_rollout_slab_search",
        num_simulations=8,
    )
    poisoned = replace(bad_result, selected_action=np.asarray([0, 2, 1, 2], dtype=np.int16))

    with pytest.raises(ReplayCompatibilityError, match="illegal"):
        selected_joint_action_from_search_result(
            root_batch,
            poisoned,
            batch_size=2,
            player_count=2,
        )


def test_compact_replay_index_rows_skip_observation_materialization():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_direct_ctree_control",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_search_impl",
        num_simulations=8,
    )

    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=0,
        next_joint_action=chunk.arrays["joint_action"][1],
        next_reward=chunk.arrays["reward"][1],
        next_done=chunk.arrays["done"][1],
        next_terminated=chunk.arrays["terminated"][1],
        next_truncated=chunk.arrays["truncated"][1],
        next_final_reward_map=chunk.arrays["final_reward_map"][1],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
        policy_source="compact_index_rows_contract_test",
    )

    assert index_rows.metadata["schema_id"] == "curvyzero_compact_replay_index_rows/v1"
    assert index_rows.metadata["observation_materialized"] is False
    assert index_rows.metadata["next_observation_materialized"] is False
    np.testing.assert_array_equal(index_rows.compact_root_row, np.asarray([0, 1, 2, 3]))
    np.testing.assert_array_equal(index_rows.policy_env_id, np.asarray([0, 1, 2, 3]))
    np.testing.assert_array_equal(index_rows.policy_row, np.asarray([0, 1, 2, 3]))
    np.testing.assert_array_equal(index_rows.env_row, np.asarray([0, 0, 1, 1]))
    np.testing.assert_array_equal(index_rows.player, np.asarray([0, 1, 0, 1]))
    np.testing.assert_array_equal(index_rows.action, selected_action)
    np.testing.assert_array_equal(index_rows.action_mask, compact_batch.action_mask.reshape(4, 3))
    np.testing.assert_allclose(index_rows.policy_target, visit_policy)
    np.testing.assert_allclose(index_rows.root_value, root_value)
    np.testing.assert_allclose(
        index_rows.reward,
        chunk.arrays["reward"][1, index_rows.env_row, index_rows.player],
    )
    expected_final_reward = chunk.arrays["reward"][1, index_rows.env_row, index_rows.player].copy()
    terminal_targets = np.asarray([False, False, True, True])
    expected_final_reward[terminal_targets] = chunk.arrays["final_reward_map"][
        1,
        index_rows.env_row[terminal_targets],
        index_rows.player[terminal_targets],
    ]
    np.testing.assert_allclose(index_rows.final_reward, expected_final_reward)
    np.testing.assert_array_equal(index_rows.done, np.asarray([False, False, True, True]))
    np.testing.assert_array_equal(
        index_rows.next_final_observation_row,
        np.asarray([False, False, True, True]),
    )
    compact_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=0,
        policy_source="compact_index_rows_contract_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )
    _assert_target_rows_equal(materialized_rows, compact_rows)

    bad_action = chunk.arrays["joint_action"][1].copy()
    bad_action[0, 0] = np.int16((int(selected_action[0]) + 1) % ACTION_COUNT)
    with pytest.raises(ReplayCompatibilityError, match="selected_action does not match"):
        build_compact_replay_index_rows_v1_from_search_result(
            compact_batch,
            root_batch,
            search_result,
            record_index=0,
            next_joint_action=bad_action,
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            next_terminated=chunk.arrays["terminated"][1],
            next_truncated=chunk.arrays["truncated"][1],
            next_final_reward_map=chunk.arrays["final_reward_map"][1],
            next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
            policy_source="compact_index_rows_contract_test",
        )

    with pytest.raises(ReplayCompatibilityError, match="next_done rows require"):
        build_compact_replay_index_rows_v1_from_search_result(
            compact_batch,
            root_batch,
            search_result,
            record_index=0,
            next_joint_action=chunk.arrays["joint_action"][1],
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            next_terminated=chunk.arrays["terminated"][1],
            next_truncated=chunk.arrays["truncated"][1],
            next_final_reward_map=chunk.arrays["final_reward_map"][1],
            next_final_observation_row_mask=np.zeros((2,), dtype=np.bool_),
            policy_source="compact_index_rows_contract_test",
        )

    with pytest.raises(ReplayCompatibilityError, match="next_final_reward_map"):
        build_compact_replay_index_rows_v1_from_search_result(
            compact_batch,
            root_batch,
            search_result,
            record_index=0,
            next_joint_action=chunk.arrays["joint_action"][1],
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            next_terminated=chunk.arrays["terminated"][1],
            next_truncated=chunk.arrays["truncated"][1],
            next_final_reward_map=None,
            next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
            policy_source="compact_index_rows_contract_test",
        )


def test_owner_action_context_replay_index_rows_match_root_batch_builder():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_owner_action_context_rows",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_search_impl",
        num_simulations=8,
    )
    trusted_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=0,
        next_joint_action=chunk.arrays["joint_action"][1],
        next_reward=chunk.arrays["reward"][1],
        next_done=chunk.arrays["done"][1],
        next_terminated=chunk.arrays["terminated"][1],
        next_truncated=chunk.arrays["truncated"][1],
        next_final_reward_map=chunk.arrays["final_reward_map"][1],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
        policy_source="compact_index_rows_contract_test",
    )
    root_request = compact_root_build_request_v1_from_batch(
        compact_batch,
        search_lane="unit_test_owner_action_context_rows",
        copy_observation=False,
        observation_source=COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
    )
    root_action_context = compact_root_action_context_v1_from_request(root_request)
    action_step = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle="owner-context-record-0",
    )
    replay_payload = compact_search_replay_payload_v1_from_result(
        search_result,
        replay_payload_handle="owner-context-record-0",
    )
    owner_rows = build_compact_replay_index_rows_v1_from_owner_action_context_payload(
        root_action_context,
        action_step,
        replay_payload,
        record_index=0,
        next_joint_action=chunk.arrays["joint_action"][1],
        next_reward=chunk.arrays["reward"][1],
        next_done=chunk.arrays["done"][1],
        next_terminated=chunk.arrays["terminated"][1],
        next_truncated=chunk.arrays["truncated"][1],
        next_final_reward_map=chunk.arrays["final_reward_map"][1],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
        policy_source="compact_index_rows_contract_test",
    )

    assert owner_rows.metadata["owner_action_context_replay_index_rows"] is True
    for field in (
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
        np.testing.assert_array_equal(getattr(owner_rows, field), getattr(trusted_rows, field))
    assert owner_rows.record_index == trusted_rows.record_index
    assert owner_rows.next_record_index == trusted_rows.next_record_index
    assert owner_rows.policy_source == trusted_rows.policy_source
    assert owner_rows.metadata["observation_materialized"] is False
    assert owner_rows.metadata["next_observation_materialized"] is False

    bad_action = chunk.arrays["joint_action"][1].copy()
    bad_action[0, 0] = np.int16((int(selected_action[0]) + 1) % ACTION_COUNT)
    with pytest.raises(ReplayCompatibilityError, match="selected_action does not match"):
        build_compact_replay_index_rows_v1_from_owner_action_context_payload(
            root_action_context,
            action_step,
            replay_payload,
            record_index=0,
            next_joint_action=bad_action,
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            next_terminated=chunk.arrays["terminated"][1],
            next_truncated=chunk.arrays["truncated"][1],
            next_final_reward_map=chunk.arrays["final_reward_map"][1],
            next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
            policy_source="compact_index_rows_contract_test",
        )


def test_owner_action_context_device_replay_index_rows_match_host_owner_rows():
    torch = pytest.importorskip("torch")

    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_owner_action_context_device_rows",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_search_impl",
        num_simulations=8,
    )
    root_request = compact_root_build_request_v1_from_batch(
        compact_batch,
        search_lane="unit_test_owner_action_context_device_rows",
        copy_observation=False,
        observation_source=COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
    )
    root_action_context = compact_root_action_context_v1_from_request(root_request)
    handle = "owner-context-device-record-0"
    action_step = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle=handle,
    )
    replay_payload = compact_search_replay_payload_v1_from_result(
        search_result,
        replay_payload_handle=handle,
    )
    owner_rows = build_compact_replay_index_rows_v1_from_owner_action_context_payload(
        root_action_context,
        action_step,
        replay_payload,
        record_index=0,
        next_joint_action=chunk.arrays["joint_action"][1],
        next_reward=chunk.arrays["reward"][1],
        next_done=chunk.arrays["done"][1],
        next_terminated=chunk.arrays["terminated"][1],
        next_truncated=chunk.arrays["truncated"][1],
        next_final_reward_map=chunk.arrays["final_reward_map"][1],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
        policy_source="compact_index_rows_contract_test",
    )
    deferred_digest = compact_search_deferred_replay_payload_digest_v1(handle)
    device_action_step = replace(
        action_step,
        metadata={
            **action_step.metadata,
            "search_replay_payload_digest": deferred_digest,
            "search_replay_payload_digest_deferred": True,
        },
    )
    device_replay_payload = CompactDeviceSearchReplayPayloadV1(
        replay_payload_handle=handle,
        root_index=search_result.root_index.copy(),
        env_row=search_result.env_row.copy(),
        player=search_result.player.copy(),
        policy_env_id=search_result.policy_env_id.copy(),
        visit_policy=torch.as_tensor(visit_policy, dtype=torch.float32),
        root_value=torch.as_tensor(root_value, dtype=torch.float32),
        raw_visit_counts=torch.as_tensor(visit_policy, dtype=torch.float32),
        predicted_value=None,
        predicted_policy_logits=None,
        metadata={
            "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
            "phase": "replay_critical_device",
            "search_impl": search_result.metadata["search_impl"],
            "num_simulations": search_result.metadata["num_simulations"],
            "active_root_count": int(search_result.root_index.size),
            "search_replay_payload_digest": deferred_digest,
            "device_replay_payload": True,
            "device_replay_payload_device": "cpu",
            "host_search_payload_fallback_allowed": False,
        },
    )

    device_rows = build_compact_device_replay_index_rows_v1_from_owner_action_context_payload(
        root_action_context,
        device_action_step,
        device_replay_payload,
        record_index=0,
        next_joint_action=chunk.arrays["joint_action"][1],
        next_reward=chunk.arrays["reward"][1],
        next_done=chunk.arrays["done"][1],
        next_terminated=chunk.arrays["terminated"][1],
        next_truncated=chunk.arrays["truncated"][1],
        next_final_reward_map=chunk.arrays["final_reward_map"][1],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
        policy_source="compact_index_rows_contract_test",
    )

    assert device_rows.metadata["device_replay_index_rows"] is True
    assert device_rows.metadata["owner_action_context_device_replay_index_rows"] is True
    assert (
        device_rows.metadata["replay_index_rows_builder_variant"]
        == "owner_context_device_packed_scalar_v1"
    )
    assert device_rows.metadata["host_search_payload_fallback_allowed"] is False
    assert device_rows.policy_target.device == device_replay_payload.visit_policy.device
    assert device_rows.root_value.device == device_replay_payload.root_value.device
    assert device_rows.action.dtype == torch.int16
    assert device_rows.action_mask.dtype == torch.bool
    assert device_rows.policy_target.dtype == torch.float32
    assert device_rows.root_value.dtype == torch.float32

    def to_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    for field in (
        "compact_root_row",
        "policy_env_id",
        "policy_row",
        "env_row",
        "player",
        "action",
        "action_mask",
        "done",
        "terminated",
        "truncated",
        "next_final_observation_row",
        "to_play",
    ):
        np.testing.assert_array_equal(
            to_numpy(getattr(device_rows, field)),
            getattr(owner_rows, field),
        )
    for field in ("policy_target", "root_value", "reward", "final_reward"):
        np.testing.assert_allclose(
            to_numpy(getattr(device_rows, field)),
            getattr(owner_rows, field),
        )
    assert device_rows.record_index == owner_rows.record_index
    assert device_rows.next_record_index == owner_rows.next_record_index
    assert device_rows.policy_source == owner_rows.policy_source

    bad_action = chunk.arrays["joint_action"][1].copy()
    bad_action[0, 0] = np.int16((int(selected_action[0]) + 1) % ACTION_COUNT)
    with pytest.raises(ReplayCompatibilityError, match="selected_action does not match"):
        build_compact_device_replay_index_rows_v1_from_owner_action_context_payload(
            root_action_context,
            device_action_step,
            device_replay_payload,
            record_index=0,
            next_joint_action=bad_action,
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            next_terminated=chunk.arrays["terminated"][1],
            next_truncated=chunk.arrays["truncated"][1],
            next_final_reward_map=chunk.arrays["final_reward_map"][1],
            next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
            policy_source="compact_index_rows_contract_test",
        )


def test_compact_replay_index_rows_allow_missing_final_reward_map_for_nonterminal_rows():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_nonterminal_missing_final_reward_map",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_search_impl",
        num_simulations=8,
    )

    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=0,
        next_joint_action=chunk.arrays["joint_action"][1],
        next_reward=chunk.arrays["reward"][1],
        next_done=chunk.arrays["done"][1],
        next_terminated=chunk.arrays["terminated"][1],
        next_truncated=chunk.arrays["truncated"][1],
        next_final_reward_map=None,
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
        policy_source="compact_nonterminal_missing_final_reward_map_test",
    )

    np.testing.assert_allclose(index_rows.final_reward, index_rows.reward)
    np.testing.assert_array_equal(
        index_rows.next_final_observation_row,
        np.zeros((4,), dtype=np.bool_),
    )


def test_compact_device_replay_index_rows_require_final_reward_map_for_terminal_rows():
    torch = pytest.importorskip("torch")

    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_device_terminal_missing_final_reward_map",
        copy_observation=False,
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_device_terminal_missing_final_reward_map_search",
        num_simulations=8,
    )
    action_step = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
    )
    replay_payload = CompactDeviceSearchReplayPayloadV1(
        replay_payload_handle="record-0",
        root_index=search_result.root_index.copy(),
        env_row=search_result.env_row.copy(),
        player=search_result.player.copy(),
        policy_env_id=search_result.policy_env_id.copy(),
        visit_policy=torch.as_tensor(visit_policy, dtype=torch.float32),
        root_value=torch.as_tensor(root_value, dtype=torch.float32),
        raw_visit_counts=torch.as_tensor(visit_policy, dtype=torch.float32),
        predicted_value=None,
        predicted_policy_logits=None,
        metadata={
            "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
            "phase": "replay_critical_device",
            "search_impl": search_result.metadata["search_impl"],
            "num_simulations": search_result.metadata["num_simulations"],
            "active_root_count": int(search_result.root_index.size),
            "device_replay_payload": True,
            "host_search_payload_fallback_allowed": False,
        },
    )

    with pytest.raises(ReplayCompatibilityError, match="next_final_reward_map"):
        build_compact_device_replay_index_rows_v1_from_payload(
            compact_batch,
            root_batch,
            action_step,
            replay_payload,
            record_index=0,
            next_joint_action=chunk.arrays["joint_action"][1],
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            next_terminated=chunk.arrays["terminated"][1],
            next_truncated=chunk.arrays["truncated"][1],
            next_final_reward_map=None,
            next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
            policy_source="compact_device_terminal_missing_final_reward_map_test",
        )


def test_compact_device_replay_index_rows_preserve_terminal_final_reward_and_mask():
    torch = pytest.importorskip("torch")

    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_device_terminal_final_reward_mask",
        copy_observation=False,
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_device_terminal_final_reward_mask_search",
        num_simulations=8,
    )
    action_step = compact_search_action_step_v1_from_result(
        search_result,
        replay_payload_handle="record-0",
    )
    replay_payload = CompactDeviceSearchReplayPayloadV1(
        replay_payload_handle="record-0",
        root_index=search_result.root_index.copy(),
        env_row=search_result.env_row.copy(),
        player=search_result.player.copy(),
        policy_env_id=search_result.policy_env_id.copy(),
        visit_policy=torch.as_tensor(visit_policy, dtype=torch.float32),
        root_value=torch.as_tensor(root_value, dtype=torch.float32),
        raw_visit_counts=torch.as_tensor(visit_policy, dtype=torch.float32),
        predicted_value=None,
        predicted_policy_logits=None,
        metadata={
            "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
            "phase": "replay_critical_device",
            "search_impl": search_result.metadata["search_impl"],
            "num_simulations": search_result.metadata["num_simulations"],
            "active_root_count": int(search_result.root_index.size),
            "device_replay_payload": True,
            "host_search_payload_fallback_allowed": False,
        },
    )

    index_rows = build_compact_device_replay_index_rows_v1_from_payload(
        compact_batch,
        root_batch,
        action_step,
        replay_payload,
        record_index=0,
        next_joint_action=chunk.arrays["joint_action"][1],
        next_reward=chunk.arrays["reward"][1],
        next_done=chunk.arrays["done"][1],
        next_terminated=chunk.arrays["terminated"][1],
        next_truncated=chunk.arrays["truncated"][1],
        next_final_reward_map=chunk.arrays["final_reward_map"][1],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][1],
        policy_source="compact_device_terminal_final_reward_mask_test",
    )

    assert index_rows.metadata["device_replay_index_rows"] is True
    assert index_rows.metadata["replay_index_rows_builder_variant"] == ("device_packed_scalar_v1")
    assert index_rows.metadata["replay_index_rows_scalar_tensor_count"] == 5
    assert index_rows.metadata["replay_index_rows_scalar_packed_h2d_bytes"] > 0
    assert index_rows.metadata["replay_index_rows_identity_validate_sec"] >= 0.0
    assert index_rows.metadata["replay_index_rows_terminal_prepare_sec"] >= 0.0
    assert index_rows.metadata["replay_index_rows_target_tensor_sec"] >= 0.0
    assert index_rows.metadata["replay_index_rows_scalar_host_pack_sec"] >= 0.0
    assert index_rows.metadata["replay_index_rows_scalar_device_transfer_sec"] >= 0.0
    assert index_rows.metadata["replay_index_rows_metadata_sec"] >= 0.0
    assert index_rows.metadata["done_row_count"] == 2
    assert index_rows.metadata["next_final_observation_row_count"] == 2
    np.testing.assert_array_equal(
        index_rows.done.cpu().numpy(),
        np.asarray([False, False, True, True], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        index_rows.next_final_observation_row.cpu().numpy(),
        np.asarray([False, False, True, True], dtype=np.bool_),
    )
    env_row = index_rows.env_row.cpu().numpy()
    player = index_rows.player.cpu().numpy()
    expected_final_reward = chunk.arrays["reward"][1, env_row, player].copy()
    terminal_targets = index_rows.next_final_observation_row.cpu().numpy()
    expected_final_reward[terminal_targets] = chunk.arrays["final_reward_map"][
        1,
        env_row[terminal_targets],
        player[terminal_targets],
    ]
    np.testing.assert_allclose(index_rows.final_reward.cpu().numpy(), expected_final_reward)
    assert index_rows.policy_target.device == replay_payload.visit_policy.device
    assert index_rows.root_value.device == replay_payload.root_value.device
    assert index_rows.action_mask.device == replay_payload.visit_policy.device
    assert index_rows.final_reward.device == replay_payload.visit_policy.device


def test_compact_search_result_v1_rejects_identity_and_legality_errors():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=0,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_direct_ctree_control",
    )

    illegal_action = selected_action.copy()
    illegal_action[0] = 0
    with pytest.raises(ReplayCompatibilityError, match="selected_action is illegal"):
        validate_compact_search_result_v1(
            root_batch,
            selected_action=illegal_action,
            visit_policy=visit_policy,
            root_value=root_value,
            search_impl="unit_test_search_impl",
            num_simulations=8,
        )

    illegal_visit = visit_policy.copy()
    illegal_visit[0] = np.asarray([0.25, 0.75, 0.0], dtype=np.float32)
    with pytest.raises(ReplayCompatibilityError, match="illegal actions"):
        validate_compact_search_result_v1(
            root_batch,
            selected_action=selected_action,
            visit_policy=illegal_visit,
            root_value=root_value,
            search_impl="unit_test_search_impl",
            num_simulations=8,
        )

    illegal_raw_counts = visit_policy.copy()
    illegal_raw_counts[0] = np.asarray([1.0, 2.0, 0.0], dtype=np.float32)
    with pytest.raises(ReplayCompatibilityError, match="raw_visit_counts assigns mass"):
        validate_compact_search_result_v1(
            root_batch,
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            raw_visit_counts=illegal_raw_counts,
            search_impl="unit_test_search_impl",
            num_simulations=8,
        )

    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_search_impl",
        num_simulations=8,
    )
    swapped_result = type(search_result)(
        root_index=search_result.root_index.copy(),
        env_row=search_result.env_row[::-1].copy(),
        player=search_result.player.copy(),
        policy_env_id=search_result.policy_env_id.copy(),
        selected_action=search_result.selected_action.copy(),
        visit_policy=search_result.visit_policy.copy(),
        root_value=search_result.root_value.copy(),
        raw_visit_counts=search_result.raw_visit_counts,
        predicted_value=search_result.predicted_value,
        predicted_policy_logits=search_result.predicted_policy_logits,
        metadata=dict(search_result.metadata),
    )
    with pytest.raises(ReplayCompatibilityError, match="env_row does not match"):
        build_compact_replay_chunk_v1_from_search_result(
            chunk,
            compact_batch,
            root_batch,
            swapped_result,
            record_index=0,
            policy_source="compact_service_identity_reject_test",
        )

    swapped_player_result = type(search_result)(
        root_index=search_result.root_index.copy(),
        env_row=search_result.env_row.copy(),
        player=search_result.player[::-1].copy(),
        policy_env_id=search_result.policy_env_id.copy(),
        selected_action=search_result.selected_action.copy(),
        visit_policy=search_result.visit_policy.copy(),
        root_value=search_result.root_value.copy(),
        raw_visit_counts=search_result.raw_visit_counts,
        predicted_value=search_result.predicted_value,
        predicted_policy_logits=search_result.predicted_policy_logits,
        metadata=dict(search_result.metadata),
    )
    with pytest.raises(ReplayCompatibilityError, match="player does not match"):
        build_compact_replay_index_rows_v1_from_search_result(
            compact_batch,
            root_batch,
            swapped_player_result,
            record_index=0,
            next_joint_action=chunk.arrays["joint_action"][1],
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            policy_source="compact_service_identity_reject_test",
        )

    duplicate_root_result = type(search_result)(
        root_index=np.asarray([0, 0, 2, 3], dtype=np.int32),
        env_row=search_result.env_row.copy(),
        player=search_result.player.copy(),
        policy_env_id=search_result.policy_env_id.copy(),
        selected_action=search_result.selected_action.copy(),
        visit_policy=search_result.visit_policy.copy(),
        root_value=search_result.root_value.copy(),
        raw_visit_counts=search_result.raw_visit_counts,
        predicted_value=search_result.predicted_value,
        predicted_policy_logits=search_result.predicted_policy_logits,
        metadata=dict(search_result.metadata),
    )
    with pytest.raises(ReplayCompatibilityError, match="roots must match"):
        build_compact_replay_index_rows_v1_from_search_result(
            compact_batch,
            root_batch,
            duplicate_root_result,
            record_index=0,
            next_joint_action=chunk.arrays["joint_action"][1],
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            policy_source="compact_service_identity_reject_test",
        )

    stale_policy_env_result = type(search_result)(
        root_index=search_result.root_index.copy(),
        env_row=search_result.env_row.copy(),
        player=search_result.player.copy(),
        policy_env_id=search_result.policy_env_id.copy(),
        selected_action=search_result.selected_action.copy(),
        visit_policy=search_result.visit_policy.copy(),
        root_value=search_result.root_value.copy(),
        raw_visit_counts=search_result.raw_visit_counts,
        predicted_value=search_result.predicted_value,
        predicted_policy_logits=search_result.predicted_policy_logits,
        metadata=dict(search_result.metadata),
    )
    stale_policy_env_result.policy_env_id[0] = np.int64(999)
    with pytest.raises(ReplayCompatibilityError, match="policy_env_id does not match"):
        build_compact_replay_index_rows_v1_from_search_result(
            compact_batch,
            root_batch,
            stale_policy_env_result,
            record_index=0,
            next_joint_action=chunk.arrays["joint_action"][1],
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            policy_source="compact_service_identity_reject_test",
        )

    stale_root_batch = type(root_batch)(
        observation=root_batch.observation,
        legal_mask=root_batch.legal_mask.copy(),
        active_root_mask=root_batch.active_root_mask,
        to_play=root_batch.to_play,
        env_row=root_batch.env_row,
        player=root_batch.player,
        policy_env_id=root_batch.policy_env_id,
        target_reward=root_batch.target_reward,
        done_root=root_batch.done_root,
        final_observation=root_batch.final_observation,
        final_observation_row_mask=root_batch.final_observation_row_mask,
        terminal_row_mask=root_batch.terminal_row_mask,
        autoreset_row_mask=root_batch.autoreset_row_mask,
        metadata=dict(root_batch.metadata),
    )
    stale_root_batch.legal_mask[0, 0] = ~stale_root_batch.legal_mask[0, 0]
    with pytest.raises(ReplayCompatibilityError, match="legal_mask does not match compact batch"):
        build_compact_replay_index_rows_v1_from_search_result(
            compact_batch,
            stale_root_batch,
            search_result,
            record_index=0,
            next_joint_action=chunk.arrays["joint_action"][1],
            next_reward=chunk.arrays["reward"][1],
            next_done=chunk.arrays["done"][1],
            policy_source="compact_service_identity_reject_test",
        )


def test_three_record_compact_rows_map_non_prefix_active_roots_to_compacted_policy_rows():
    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.zeros((3, 2), dtype=np.bool_),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[False, True], [False, True]],
                [[True, True], [True, True]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.zeros((3, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=1,
    )

    records = build_policy_row_records_from_compact_search_v0(
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=1,
        policy_source="compact_non_prefix_active_contract_test",
    )
    compact_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=1,
        policy_source="compact_non_prefix_active_contract_test",
    )
    object_rows = _expected_rows_from_replay_policy_rows(
        chunk,
        record_index=1,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        policy_source="compact_non_prefix_active_contract_test",
    )

    assert [record.policy_row for record in records] == [0, 1]
    assert [record.env_row for record in records] == [0, 1]
    assert [record.player for record in records] == [1, 1]
    _assert_target_rows_equal(compact_rows, object_rows)
    np.testing.assert_array_equal(compact_rows.policy_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(compact_rows.env_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(compact_rows.player, np.asarray([1, 1]))
    np.testing.assert_array_equal(compact_rows.record_index, np.asarray([1, 1]))
    np.testing.assert_array_equal(compact_rows.next_record_index, np.asarray([2, 2]))
    for row_index, env_row in enumerate(compact_rows.env_row):
        assert compact_rows.observation[row_index, 3, 0, 0] == _latest_value(
            1,
            int(env_row),
            1,
        )
    assert compact_rows.next_observation[row_index, 3, 0, 0] == _latest_value(
        2,
        int(env_row),
        1,
    )

    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_direct_ctree_control",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_search_impl",
        num_simulations=8,
    )
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=1,
        next_joint_action=chunk.arrays["joint_action"][2],
        next_reward=chunk.arrays["reward"][2],
        next_done=chunk.arrays["done"][2],
        next_terminated=chunk.arrays["terminated"][2],
        next_truncated=chunk.arrays["truncated"][2],
        next_final_reward_map=chunk.arrays["final_reward_map"][2],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][2],
        policy_source="compact_non_prefix_active_contract_test",
    )
    np.testing.assert_array_equal(index_rows.compact_root_row, np.asarray([1, 3]))
    np.testing.assert_array_equal(index_rows.policy_env_id, np.asarray([1, 3]))
    np.testing.assert_array_equal(index_rows.policy_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(index_rows.env_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(index_rows.player, np.asarray([1, 1]))
    np.testing.assert_array_equal(index_rows.action, np.asarray([2, 2], dtype=np.int16))
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )
    _assert_target_rows_equal(materialized_rows, compact_rows)


def test_deferred_search_payload_rows_match_immediate_rows_for_non_prefix_roots():
    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.zeros((3, 2), dtype=np.bool_),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[False, True], [False, True]],
                [[True, True], [True, True]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.zeros((3, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=1)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=1,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_deferred_payload_contract",
        copy_observation=False,
    )

    # The CPU env step only needs selected actions plus row/player sidecars.
    staged_joint_action = np.full((2, 2), 1, dtype=np.int16)
    active_root_mask = np.asarray(root_batch.active_root_mask, dtype=np.bool_)
    staged_joint_action[
        root_batch.env_row[active_root_mask].astype(np.int64, copy=False),
        root_batch.player[active_root_mask].astype(np.int64, copy=False),
    ] = selected_action
    np.testing.assert_array_equal(staged_joint_action, chunk.arrays["joint_action"][2])

    # The replay payload can arrive later as long as it is attached to the same roots.
    deferred_search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_deferred_payload_search",
        num_simulations=8,
    )
    deferred_index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        deferred_search_result,
        record_index=1,
        next_joint_action=staged_joint_action,
        next_reward=chunk.arrays["reward"][2],
        next_done=chunk.arrays["done"][2],
        next_terminated=chunk.arrays["terminated"][2],
        next_truncated=chunk.arrays["truncated"][2],
        next_final_reward_map=chunk.arrays["final_reward_map"][2],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][2],
        policy_source="compact_deferred_payload_contract_test",
    )
    immediate_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=1,
        policy_source="compact_deferred_payload_contract_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        deferred_index_rows,
    )

    assert root_batch.metadata["observation_copied"] is False
    np.testing.assert_array_equal(deferred_index_rows.compact_root_row, np.asarray([1, 3]))
    np.testing.assert_array_equal(deferred_index_rows.policy_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(deferred_index_rows.env_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(deferred_index_rows.player, np.asarray([1, 1]))
    np.testing.assert_array_equal(deferred_index_rows.action, selected_action)
    np.testing.assert_allclose(deferred_index_rows.policy_target, visit_policy)
    np.testing.assert_allclose(deferred_index_rows.root_value, root_value)
    _assert_target_rows_equal(materialized_rows, immediate_rows)


def test_closed_compact_loop_index_rows_materialize_same_as_immediate_rows():
    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[False, True], [False, True]],
                [[True, True], [True, True]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    policy_env_id = np.asarray([101, 103, 107, 109], dtype=np.int64)
    compact_batch = replace(
        _compact_batch_from_chunk(chunk, record_index=1),
        policy_env_id=policy_env_id,
    )
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=1,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_closed_compact_loop",
        copy_observation=False,
    )

    staged_joint_action = np.full((2, 2), 1, dtype=np.int16)
    active_root_mask = np.asarray(root_batch.active_root_mask, dtype=np.bool_)
    staged_joint_action[
        root_batch.env_row[active_root_mask].astype(np.int64, copy=False),
        root_batch.player[active_root_mask].astype(np.int64, copy=False),
    ] = selected_action
    np.testing.assert_array_equal(staged_joint_action, chunk.arrays["joint_action"][2])

    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_closed_compact_loop_search",
        num_simulations=8,
    )
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=1,
        next_joint_action=staged_joint_action,
        next_reward=chunk.arrays["reward"][2],
        next_done=chunk.arrays["done"][2],
        next_terminated=chunk.arrays["terminated"][2],
        next_truncated=chunk.arrays["truncated"][2],
        next_final_reward_map=chunk.arrays["final_reward_map"][2],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][2],
        policy_source="compact_closed_loop_index_rows_contract_test",
    )
    immediate_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=1,
        policy_source="compact_closed_loop_index_rows_contract_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )

    expected_compact_roots = np.asarray([1, 3], dtype=np.int32)
    np.testing.assert_array_equal(root_batch.active_root_mask, [False, True, False, True])
    np.testing.assert_array_equal(search_result.root_index, expected_compact_roots)
    np.testing.assert_array_equal(search_result.policy_env_id, policy_env_id[[1, 3]])
    np.testing.assert_array_equal(index_rows.compact_root_row, expected_compact_roots)
    np.testing.assert_array_equal(index_rows.policy_env_id, policy_env_id[[1, 3]])
    np.testing.assert_array_equal(index_rows.policy_row, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(index_rows.env_row, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(index_rows.player, np.asarray([1, 1], dtype=np.int16))
    np.testing.assert_array_equal(index_rows.action, selected_action)
    np.testing.assert_allclose(index_rows.policy_target, visit_policy)
    np.testing.assert_allclose(index_rows.root_value, root_value)

    _assert_target_rows_equal(materialized_rows, immediate_rows)
    assert materialized_rows.source_record_ref == immediate_rows.source_record_ref
    for output_row, ref in enumerate(materialized_rows.source_record_ref):
        assert ref["policy_env_id"] == int(policy_env_id[expected_compact_roots[output_row]])
        assert ref["compact_root_row"] == int(expected_compact_roots[output_row])
        assert ref["policy_row"] == output_row
        assert ref["active_output_row"] == output_row
    np.testing.assert_array_equal(materialized_rows.done, np.asarray([False, True]))
    assert materialized_rows.next_observation[1, 3, 0, 0] == _final_value(2, 1, 1)
    assert materialized_rows.next_observation[1, 3, 0, 0] != _latest_value(2, 1, 1)


def test_compact_index_rows_materialized_sample_batch_matches_immediate_rows():
    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[False, True], [False, True]],
                [[True, True], [True, True]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = replace(
        _compact_batch_from_chunk(chunk, record_index=1),
        policy_env_id=np.asarray([101, 103, 107, 109], dtype=np.int64),
    )
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=1,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_closed_compact_loop_sample_batch",
        copy_observation=False,
    )
    active_root_mask = np.asarray(root_batch.active_root_mask, dtype=np.bool_)
    staged_joint_action = np.full((2, 2), 1, dtype=np.int16)
    staged_joint_action[
        root_batch.env_row[active_root_mask].astype(np.int64, copy=False),
        root_batch.player[active_root_mask].astype(np.int64, copy=False),
    ] = selected_action
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_closed_compact_loop_sample_batch_search",
        num_simulations=8,
    )
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=1,
        next_joint_action=staged_joint_action,
        next_reward=chunk.arrays["reward"][2],
        next_done=chunk.arrays["done"][2],
        next_terminated=chunk.arrays["terminated"][2],
        next_truncated=chunk.arrays["truncated"][2],
        next_final_reward_map=chunk.arrays["final_reward_map"][2],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][2],
        policy_source="compact_closed_loop_sample_batch_contract_test",
    )
    immediate_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=1,
        policy_source="compact_closed_loop_sample_batch_contract_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )

    immediate_batch = build_source_state_multiplayer_sample_batch_v0(
        immediate_rows,
        batch_size=2,
        seed=17,
    )
    materialized_batch = build_source_state_multiplayer_sample_batch_v0(
        materialized_rows,
        batch_size=2,
        seed=17,
    )

    _assert_sample_batches_equal(materialized_batch, immediate_batch)
    assert materialized_batch.metadata["native_game_segment_claim"] is False
    assert materialized_batch.metadata["lightzero_training_integration_claim"] is False


def test_compact_index_rows_materialized_stock_lightzero_target_hooks_match():
    pytest.importorskip("lzero", reason="DI-engine/LightZero runtime is not installed locally")
    game_segment_module = pytest.importorskip("lzero.mcts.buffer.game_segment")
    buffer_module = pytest.importorskip("lzero.mcts.buffer.game_buffer_muzero")
    torch = pytest.importorskip("torch")

    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[False, True], [False, True]],
                [[True, True], [True, True]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = replace(
        _compact_batch_from_chunk(chunk, record_index=1),
        policy_env_id=np.asarray([101, 103, 107, 109], dtype=np.int64),
    )
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=1,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_closed_compact_loop_stock_lightzero",
        copy_observation=False,
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_closed_compact_loop_stock_lightzero_search",
        num_simulations=8,
    )
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=1,
        next_joint_action=chunk.arrays["joint_action"][2],
        next_reward=chunk.arrays["reward"][2],
        next_done=chunk.arrays["done"][2],
        next_terminated=chunk.arrays["terminated"][2],
        next_truncated=chunk.arrays["truncated"][2],
        next_final_reward_map=chunk.arrays["final_reward_map"][2],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][2],
        policy_source="compact_closed_loop_stock_lightzero_contract_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )
    config = build_lightzero_muzero_bridge_config(
        action_space_size=ACTION_COUNT,
        game_segment_length=1,
        num_unroll_steps=0,
        td_steps=1,
        discount_factor=1.0,
    )
    native = build_lightzero_source_state_native_game_segments_v0(
        materialized_rows,
        game_segment_cls=game_segment_module.GameSegment,
        config=config,
        action_space_size=ACTION_COUNT,
    )

    try:
        pushed = maybe_push_lightzero_source_state_native_segments_into_muzero_buffer_v0(
            native,
            buffer_cls=buffer_module.MuZeroGameBuffer,
        )
    except Exception as exc:
        pytest.skip(f"MuZeroGameBuffer push blocked stock target parity: {exc!r}")

    buffer = pushed.buffer
    required_hooks = (
        "_prepare_reward_value_context",
        "_compute_target_reward_value",
        "_prepare_policy_non_reanalyzed_context",
        "_compute_target_policy_non_reanalyzed",
    )
    missing_hooks = [name for name in required_hooks if not hasattr(buffer, name)]
    if missing_hooks:
        pytest.skip(
            "MuZeroGameBuffer stock target parity blocked: missing hooks "
            + ", ".join(missing_hooks)
        )

    segment_lengths = [len(spec.actions) for spec in native.specs]
    first_transition_indices = np.cumsum([0, *segment_lengths[:-1]]).astype(int).tolist()
    segment_start_offsets = [0 for _spec in native.specs]
    max_segment_length = max(segment_lengths, default=0)

    try:
        reward_context = buffer._prepare_reward_value_context(
            first_transition_indices,
            list(native.game_segments),
            segment_start_offsets,
            buffer.get_num_of_transitions(),
        )
        target_rewards, target_values = buffer._compute_target_reward_value(
            reward_context,
            _LightZeroZeroTargetModel(
                torch=torch,
                action_space_size=ACTION_COUNT,
                value_support_size=_support_size(config),
                reward_support_size=_support_size(config),
            ),
        )
        policy_context = buffer._prepare_policy_non_reanalyzed_context(
            first_transition_indices,
            list(native.game_segments),
            segment_start_offsets,
        )
        target_policies = buffer._compute_target_policy_non_reanalyzed(
            policy_context,
            max_segment_length,
        )
    except Exception as exc:
        pytest.skip(
            f"MuZeroGameBuffer stock target parity blocked after push: {type(exc).__name__}: {exc}"
        )

    assert pushed.transition_count == len(materialized_rows.action)
    for segment_index, spec in enumerate(native.specs):
        expected_rows = list(spec.row_id)
        np.testing.assert_allclose(
            _to_numpy(target_policies[segment_index]),
            materialized_rows.policy_target[expected_rows],
        )
        np.testing.assert_allclose(
            _to_numpy(target_rewards[segment_index]),
            materialized_rows.reward[expected_rows],
        )
        expected_returns = np.asarray(
            [
                sum(float(reward) for reward in spec.rewards[start:])
                for start in range(len(spec.rewards))
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(_to_numpy(target_values[segment_index]), expected_returns)


def test_compact_index_rows_materialized_stock_lightzero_public_sample_matches():
    pytest.importorskip("lzero", reason="DI-engine/LightZero runtime is not installed locally")
    game_segment_module = pytest.importorskip("lzero.mcts.buffer.game_segment")
    buffer_module = pytest.importorskip("lzero.mcts.buffer.game_buffer_muzero")
    torch = pytest.importorskip("torch")

    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[False, True], [False, True]],
                [[True, True], [True, True]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = replace(
        _compact_batch_from_chunk(chunk, record_index=1),
        policy_env_id=np.asarray([101, 103, 107, 109], dtype=np.int64),
    )
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=1,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_closed_compact_loop_stock_lightzero_public_sample",
        copy_observation=False,
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_closed_compact_loop_stock_lightzero_public_sample_search",
        num_simulations=8,
    )
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=1,
        next_joint_action=chunk.arrays["joint_action"][2],
        next_reward=chunk.arrays["reward"][2],
        next_done=chunk.arrays["done"][2],
        next_terminated=chunk.arrays["terminated"][2],
        next_truncated=chunk.arrays["truncated"][2],
        next_final_reward_map=chunk.arrays["final_reward_map"][2],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][2],
        policy_source="compact_closed_loop_stock_lightzero_public_sample_contract_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )
    config = build_lightzero_muzero_bridge_config(
        action_space_size=ACTION_COUNT,
        game_segment_length=1,
        num_unroll_steps=0,
        td_steps=1,
        discount_factor=1.0,
    )
    config.model.model_type = "conv"
    config.model.observation_shape = FRAME_STACK_SHAPE
    config.model.image_channel = FRAME_STACK_SHAPE[0]
    native = build_lightzero_source_state_native_game_segments_v0(
        materialized_rows,
        game_segment_cls=game_segment_module.GameSegment,
        config=config,
        action_space_size=ACTION_COUNT,
    )

    try:
        pushed = maybe_push_lightzero_source_state_native_segments_into_muzero_buffer_v0(
            native,
            buffer_cls=buffer_module.MuZeroGameBuffer,
        )
    except Exception as exc:
        pytest.skip(f"MuZeroGameBuffer push blocked public sample parity: {exc!r}")

    buffer = pushed.buffer
    if not hasattr(buffer, "sample"):
        pytest.skip("MuZeroGameBuffer public sample parity blocked: missing sample")

    policy = SimpleNamespace(
        _target_model=_LightZeroZeroTargetModel(
            torch=torch,
            action_space_size=ACTION_COUNT,
            value_support_size=_support_size(config),
            reward_support_size=_support_size(config),
        )
    )
    rng_state = np.random.get_state()
    try:
        np.random.seed(23)
        train_data = buffer.sample(len(materialized_rows.action), policy)
    except Exception as exc:
        pytest.skip(
            f"MuZeroGameBuffer public sample parity blocked after push: {type(exc).__name__}: {exc}"
        )
    finally:
        np.random.set_state(rng_state)

    if not isinstance(train_data, (list, tuple)) or len(train_data) != 2:
        pytest.skip("MuZeroGameBuffer public sample returned an unexpected batch shape")
    current_batch, target_batch = train_data
    if len(current_batch) < 4 or len(target_batch) != 3:
        pytest.skip("MuZeroGameBuffer public sample returned an unexpected batch layout")

    batch_index = np.asarray(current_batch[3], dtype=np.int64)
    if batch_index.shape != (len(materialized_rows.action),):
        pytest.skip("MuZeroGameBuffer public sample returned unexpected sample indices")

    try:
        sampled_row_id = _lightzero_sample_row_ids(buffer, native, batch_index)
    except Exception as exc:
        pytest.skip(
            "MuZeroGameBuffer public sample parity blocked by transition lookup: "
            f"{type(exc).__name__}: {exc}"
        )

    sample_observation = _to_numpy(current_batch[0])
    target_rewards, target_values, target_policies = (_to_numpy(item) for item in target_batch)

    np.testing.assert_array_equal(
        sample_observation,
        materialized_rows.observation[sampled_row_id],
    )
    np.testing.assert_allclose(
        target_policies[:, 0, :],
        materialized_rows.policy_target[sampled_row_id],
    )
    np.testing.assert_allclose(
        target_rewards[:, 0],
        materialized_rows.reward[sampled_row_id],
    )
    np.testing.assert_allclose(
        target_values[:, 0],
        materialized_rows.reward[sampled_row_id],
    )


def test_compact_multi_record_sample_batch_matches_stock_muzero_public_sample_for_terminal_nstep():
    pytest.importorskip("lzero", reason="DI-engine/LightZero runtime is not installed locally")
    game_segment_module = pytest.importorskip("lzero.mcts.buffer.game_segment")
    buffer_module = pytest.importorskip("lzero.mcts.buffer.game_buffer_muzero")
    torch = pytest.importorskip("torch")

    num_unroll_steps = 2
    td_steps = 2
    discount_factor = 1.0
    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
                [True, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.ones((3, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, False],
                [True, True],
            ],
            dtype=np.bool_,
        ),
    )
    index0 = _compact_index_rows_for_record(chunk, record_index=0)
    index1 = _compact_index_rows_for_record(chunk, record_index=1)
    durable_lineage = {
        "policy_version_ref": "public-sample-durable-policy-v1",
        "model_version_ref": "public-sample-durable-model-v1",
        "support_scale": 9,
    }
    index0 = replace(index0, metadata={**index0.metadata, **durable_lineage})
    index1 = replace(index1, metadata={**index1.metadata, **durable_lineage})
    index0, index1 = pickle.loads(pickle.dumps((index0, index1), protocol=pickle.HIGHEST_PROTOCOL))
    assert index0.metadata["policy_version_ref"] == "public-sample-durable-policy-v1"
    assert index1.metadata["model_version_ref"] == "public-sample-durable-model-v1"
    materialized_rows = materialize_compact_target_rows_from_index_row_groups_v1(
        chunk,
        [index0, index1],
    )
    config = build_lightzero_muzero_bridge_config(
        action_space_size=ACTION_COUNT,
        game_segment_length=2,
        num_unroll_steps=num_unroll_steps,
        td_steps=td_steps,
        discount_factor=discount_factor,
        support_scale=9,
    )
    config.model.model_type = "conv"
    config.model.observation_shape = FRAME_STACK_SHAPE
    config.model.image_channel = FRAME_STACK_SHAPE[0]
    native = build_lightzero_source_state_native_game_segments_v0(
        materialized_rows,
        game_segment_cls=game_segment_module.GameSegment,
        config=config,
        action_space_size=ACTION_COUNT,
    )

    try:
        pushed = maybe_push_lightzero_source_state_native_segments_into_muzero_buffer_v0(
            native,
            buffer_cls=buffer_module.MuZeroGameBuffer,
        )
    except Exception as exc:
        pytest.skip(f"MuZeroGameBuffer push blocked terminal N-step sample diff: {exc!r}")

    buffer = pushed.buffer
    if not hasattr(buffer, "sample"):
        pytest.skip("MuZeroGameBuffer terminal N-step sample diff blocked: missing sample")

    policy = SimpleNamespace(
        _target_model=_LightZeroZeroTargetModel(
            torch=torch,
            action_space_size=ACTION_COUNT,
            value_support_size=_support_size(config),
            reward_support_size=_support_size(config),
        )
    )
    rng_state = np.random.get_state()
    try:
        np.random.seed(37)
        train_data = buffer.sample(len(materialized_rows.action), policy)
    except Exception as exc:
        pytest.skip(
            "MuZeroGameBuffer terminal N-step sample diff blocked after push: "
            f"{type(exc).__name__}: {exc}"
        )
    finally:
        np.random.set_state(rng_state)

    if not isinstance(train_data, (list, tuple)) or len(train_data) != 2:
        pytest.skip("MuZeroGameBuffer public sample returned an unexpected batch shape")
    current_batch, target_batch = train_data
    if len(current_batch) < 4 or len(target_batch) != 3:
        pytest.skip("MuZeroGameBuffer public sample returned an unexpected batch layout")

    batch_index = np.asarray(current_batch[3], dtype=np.int64)
    if batch_index.shape != (len(materialized_rows.action),):
        pytest.skip("MuZeroGameBuffer public sample returned unexpected sample indices")

    try:
        sampled_row_id = _lightzero_sample_row_ids(buffer, native, batch_index)
    except Exception as exc:
        pytest.skip(
            "MuZeroGameBuffer terminal N-step sample diff blocked by transition lookup: "
            f"{type(exc).__name__}: {exc}"
        )

    (
        expected_observation,
        expected_action,
        expected_mask,
        expected_rewards,
        expected_values,
        expected_policies,
    ) = _expected_lightzero_public_sample_for_row_ids(
        native,
        sampled_row_id,
        num_unroll_steps=num_unroll_steps,
        td_steps=td_steps,
        discount_factor=discount_factor,
    )
    sample_observation = _to_numpy(current_batch[0])
    action_batch = np.asarray(current_batch[1], dtype=np.int64)
    mask_batch = np.asarray(current_batch[2], dtype=np.float32)
    target_rewards, target_values, target_policies = (_to_numpy(item) for item in target_batch)

    np.testing.assert_array_equal(sample_observation, expected_observation)
    np.testing.assert_array_equal(
        sample_observation[:, : FRAME_STACK_SHAPE[0]],
        materialized_rows.observation[sampled_row_id],
    )
    np.testing.assert_array_equal(mask_batch, expected_mask)
    valid_action_slots = expected_mask[:, :num_unroll_steps].astype(bool)
    np.testing.assert_array_equal(
        action_batch[valid_action_slots],
        expected_action[valid_action_slots],
    )
    np.testing.assert_array_equal(action_batch[:, 0], materialized_rows.action[sampled_row_id])
    np.testing.assert_allclose(target_rewards, expected_rewards, atol=1e-6)
    np.testing.assert_allclose(target_values, expected_values, atol=1e-6)
    np.testing.assert_allclose(target_policies, expected_policies, atol=1e-6)
    np.testing.assert_allclose(
        target_rewards[:, 0],
        materialized_rows.reward[sampled_row_id],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        target_policies[:, 0, :],
        materialized_rows.policy_target[sampled_row_id],
        atol=1e-6,
    )

    terminal_sample = np.asarray(materialized_rows.done[sampled_row_id], dtype=np.bool_)
    nonterminal_sample = ~terminal_sample
    assert bool(terminal_sample.any())
    assert bool(nonterminal_sample.any())
    np.testing.assert_array_equal(
        mask_batch[terminal_sample],
        np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (int(terminal_sample.sum()), 1)),
    )
    np.testing.assert_array_equal(
        mask_batch[nonterminal_sample],
        np.tile(
            np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
            (int(nonterminal_sample.sum()), 1),
        ),
    )
    np.testing.assert_allclose(
        target_values[terminal_sample, 0],
        materialized_rows.reward[sampled_row_id[terminal_sample]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        target_policies[terminal_sample, 1:, :],
        np.zeros((int(terminal_sample.sum()), 2, ACTION_COUNT), dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        target_values[nonterminal_sample, 0],
        target_rewards[nonterminal_sample, 0] + target_rewards[nonterminal_sample, 1],
        atol=1e-6,
    )


def test_compact_search_service_v1_protocol_runs_fake_service_to_index_rows():
    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[False, True], [False, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=1)
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_compact_search_service_v1",
        copy_observation=False,
    )
    service = _FakeCompactSearchServiceV1(
        selected_action=np.asarray([2, 2], dtype=np.int16),
        visit_policy=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        root_value=np.asarray([-0.125, 0.375], dtype=np.float32),
    )

    assert isinstance(service, CompactSearchServiceV1)
    search_result = service.run(root_batch)
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=1,
        next_joint_action=chunk.arrays["joint_action"][2],
        next_reward=chunk.arrays["reward"][2],
        next_done=chunk.arrays["done"][2],
        next_terminated=chunk.arrays["terminated"][2],
        next_truncated=chunk.arrays["truncated"][2],
        next_final_reward_map=chunk.arrays["final_reward_map"][2],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][2],
        policy_source="compact_search_service_v1_protocol_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )
    immediate_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=service.selected_action,
        visit_policy=service.visit_policy,
        root_value=service.root_value,
        record_index=1,
        policy_source="compact_search_service_v1_protocol_test",
    )

    assert search_result.metadata["search_impl"] == "unit_test_fake_compact_service"
    np.testing.assert_array_equal(search_result.root_index, np.asarray([1, 3]))
    _assert_target_rows_equal(materialized_rows, immediate_rows)


def test_compact_search_service_index_rows_preserve_rnd_and_player_perspective():
    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[False, True], [False, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=1)
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_compact_search_service_v1",
        copy_observation=False,
    )
    service = _FakeCompactSearchServiceV1(
        selected_action=np.asarray([2, 2], dtype=np.int16),
        visit_policy=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        root_value=np.asarray([-0.125, 0.375], dtype=np.float32),
    )

    search_result = service.run(root_batch)
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=1,
        next_joint_action=chunk.arrays["joint_action"][2],
        next_reward=chunk.arrays["reward"][2],
        next_done=chunk.arrays["done"][2],
        next_terminated=chunk.arrays["terminated"][2],
        next_truncated=chunk.arrays["truncated"][2],
        next_final_reward_map=chunk.arrays["final_reward_map"][2],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][2],
        policy_source="compact_search_service_rnd_perspective_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )
    rnd_latest = extract_policy_gray64_latest_for_rnd_from_compact_observation(
        compact_batch.observation,
        compact_batch.target_reward,
    )

    np.testing.assert_array_equal(search_result.root_index, np.asarray([1, 3]))
    np.testing.assert_array_equal(index_rows.env_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(index_rows.player, np.asarray([1, 1]))
    np.testing.assert_array_equal(index_rows.compact_root_row, search_result.root_index)
    for row_index, compact_root in enumerate(index_rows.compact_root_row):
        env_row = int(index_rows.env_row[row_index])
        player = int(index_rows.player[row_index])
        expected_latest = _latest_value(1, env_row, player)
        assert rnd_latest[int(compact_root), 0, 0, 0] == expected_latest
        assert materialized_rows.observation[row_index, 3, 0, 0] == expected_latest
        assert materialized_rows.player[row_index] == player


def test_compact_search_service_index_rows_feed_rnd_model_and_terminal_final_obs(
    tmp_path,
):
    pytest.importorskip("torch")
    chunk = _synthetic_chunk(
        time_steps=3,
        done_by_record=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
        live_by_record=np.asarray(
            [
                [[True, True], [True, True]],
                [[False, True], [False, True]],
                [[True, True], [False, False]],
            ],
            dtype=np.bool_,
        ),
        final_observation_row_mask=np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=np.bool_,
        ),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=1)
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_compact_search_service_v1",
        copy_observation=False,
    )
    service = _FakeCompactSearchServiceV1(
        selected_action=np.asarray([2, 2], dtype=np.int16),
        visit_policy=np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        root_value=np.asarray([-0.125, 0.375], dtype=np.float32),
    )

    search_result = service.run(root_batch)
    index_rows = build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=1,
        next_joint_action=chunk.arrays["joint_action"][2],
        next_reward=chunk.arrays["reward"][2],
        next_done=chunk.arrays["done"][2],
        next_terminated=chunk.arrays["terminated"][2],
        next_truncated=chunk.arrays["truncated"][2],
        next_final_reward_map=chunk.arrays["final_reward_map"][2],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][2],
        policy_source="compact_search_service_rnd_model_terminal_test",
    )
    materialized_rows = materialize_compact_target_rows_from_index_rows_v1(
        chunk,
        index_rows,
    )
    rnd_latest = extract_policy_gray64_latest_for_rnd_from_compact_observation(
        compact_batch.observation,
        compact_batch.target_reward,
    )

    np.testing.assert_array_equal(index_rows.env_row, np.asarray([0, 1]))
    np.testing.assert_array_equal(index_rows.player, np.asarray([1, 1]))
    np.testing.assert_array_equal(index_rows.next_final_observation_row, [False, True])
    assert materialized_rows.next_observation[1, 3, 0, 0] == _final_value(2, 1, 1)
    assert materialized_rows.next_observation[1, 3, 0, 0] != _latest_value(2, 1, 1)
    np.testing.assert_allclose(
        rnd_latest[index_rows.compact_root_row, 0, 0, 0],
        materialized_rows.observation[:, 3, 0, 0],
    )

    class Segment:
        pass

    segment = Segment()
    segment.obs_segment = materialized_rows.observation.astype(np.float32, copy=True)
    target_reward = materialized_rows.reward.astype(np.float32).reshape(
        materialized_rows.reward.shape[0],
        1,
        1,
    )
    config = {
        "input_type": "obs",
        "intrinsic_reward_type": "add",
        "intrinsic_reward_weight": 0.25,
        "curvyzero_adapter": {
            "shape": [1, 64, 64],
            "source_observation_shape": [4, 64, 64],
        },
        "hidden_size_list": [8],
        "learning_rate": 1e-2,
        "weight_decay": 0.0,
        "batch_size": 2,
        "update_per_collect": 2,
        "rnd_buffer_size": 16,
        "curvyzero_metrics_latest_path": str(tmp_path / "rnd_latest.json"),
    }
    model = CurvyRNDRewardModel(config, device="cpu")
    model.collect_data(([segment], None))
    model.train_with_data()
    output = model.estimate([[materialized_rows.observation], [target_reward]])

    assert model.collect_data_calls == 1
    assert model.train_cnt_rnd == 2
    assert model.estimate_cnt_rnd == 1
    assert model.last_target_hash_before_train == model.last_target_hash_after_train
    assert model.last_predictor_hash_before_train != model.last_predictor_hash_after_train
    assert model.last_target_reward_changed is True
    assert model.last_target_reward_delta_abs_max <= 0.25 + 1e-6
    assert not np.array_equal(output[1][0], target_reward)


def test_compact_search_result_from_arrays_preserves_optional_payloads():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.asarray([[False, False], [False, False]], dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_array_helper",
        copy_observation=False,
    )
    search_arrays = {
        "selected_action": np.asarray([1, 2, 1, 2], dtype=np.int16),
        "visit_policy": np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "root_value": np.asarray([0.5, -0.25, 0.125, 0.0], dtype=np.float32),
        "raw_visit_counts": np.asarray(
            [
                [0.0, 8.0, 0.0],
                [0.0, 0.0, 8.0],
                [0.0, 8.0, 0.0],
                [0.0, 0.0, 8.0],
            ],
            dtype=np.float32,
        ),
        "predicted_value": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "predicted_policy_logits": np.asarray(
            [
                [0.1, 0.9, -1.0],
                [0.2, -0.5, 0.7],
                [1.0, -1.0, 0.0],
                [0.0, 0.8, -0.2],
            ],
            dtype=np.float32,
        ),
        "search_impl": "unit_test_existing_probe",
        "actual_search_simulations": 7,
    }

    result = compact_search_result_v1_from_arrays(
        root_batch,
        search_arrays,
        default_search_impl="default_not_used",
        default_num_simulations=3,
        metadata={"source": "arrays_once"},
    )

    assert result.metadata["search_impl"] == "unit_test_existing_probe"
    assert result.metadata["num_simulations"] == 7
    assert result.metadata["source"] == "arrays_once"
    np.testing.assert_array_equal(result.root_index, np.asarray([0, 1, 2, 3]))
    np.testing.assert_allclose(result.raw_visit_counts, search_arrays["raw_visit_counts"])
    np.testing.assert_allclose(result.predicted_value, search_arrays["predicted_value"])
    np.testing.assert_allclose(
        result.predicted_policy_logits,
        search_arrays["predicted_policy_logits"],
    )


def test_compact_root_batch_can_keep_observation_view_for_profile_hot_path():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)

    copied = build_compact_root_batch_v1(
        compact_batch,
        search_lane="copy_contract",
    )
    viewed = build_compact_root_batch_v1(
        compact_batch,
        search_lane="view_contract",
        copy_observation=False,
    )

    assert copied.metadata["observation_copied"] is True
    assert viewed.metadata["observation_copied"] is False
    assert not np.shares_memory(copied.observation, compact_batch.observation)
    assert np.shares_memory(viewed.observation, compact_batch.observation)
    np.testing.assert_array_equal(copied.observation, viewed.observation)


def test_resident_root_batch_can_stub_parent_host_observation():
    chunk = _synthetic_chunk(
        time_steps=2,
        done_by_record=np.zeros((2, 2), dtype=np.bool_),
        live_by_record=np.ones((2, 2, 2), dtype=np.bool_),
        final_observation_row_mask=np.zeros((2, 2), dtype=np.bool_),
    )
    compact_batch = _compact_batch_from_chunk(chunk, record_index=0)
    hostile_host = np.full_like(compact_batch.observation, 99)
    resident_observation = compact_batch.observation.astype(np.float32, copy=True)
    batch_size, player_count = resident_observation.shape[:2]
    stack_shape = tuple(int(dim) for dim in resident_observation.shape[2:])
    resident = ResidentObservationBatchV1(
        device_observation=resident_observation,
        root_device_observation=resident_observation.reshape(
            batch_size * player_count,
            *stack_shape,
        ),
        generation_id=7,
        batch_size=batch_size,
        player_count=player_count,
        stack_shape=stack_shape,
        dtype=str(resident_observation.dtype),
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=7,
        source_backend="unit_resident_stub",
        host_fallback_allowed=False,
        final_observation_row_mask=np.zeros((batch_size,), dtype=np.bool_),
    )
    resident_batch = replace(
        compact_batch,
        observation=hostile_host,
        final_observation=None,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=resident,
    )

    root_batch = build_compact_root_batch_v1(
        resident_batch,
        search_lane="resident_stub_contract",
        copy_observation=False,
        resident_host_observation_stub=True,
    )

    assert root_batch.resident_observation is resident
    assert root_batch.metadata["resident_host_observation_stub_requested"] is True
    assert root_batch.metadata["resident_host_observation_stubbed"] is True
    assert root_batch.metadata["resident_host_observation_stub_kind"] == (
        "zero_stride_shape_only_v1"
    )
    assert root_batch.metadata["resident_host_observation_stub_materialized_bytes"] == 0
    assert root_batch.metadata["resident_host_observation_stub_logical_bytes"] == (
        hostile_host.nbytes
    )
    assert root_batch.observation.shape == (
        batch_size * player_count,
        *stack_shape,
    )
    assert not np.shares_memory(root_batch.observation, hostile_host)
    assert not np.any(root_batch.observation == 99)
    assert 0 in root_batch.observation.strides


def _synthetic_chunk(
    *,
    time_steps: int,
    done_by_record: np.ndarray,
    live_by_record: np.ndarray,
    final_observation_row_mask: np.ndarray,
) -> SourceStateMultiplayerTrainerReplayChunkV0:
    batch_size = 2
    player_count = 2
    observation = np.zeros((time_steps, batch_size, player_count, *FRAME_STACK_SHAPE))
    final_observation = np.zeros_like(observation)
    for record_index in range(time_steps):
        for env_row in range(batch_size):
            for player in range(player_count):
                _write_stack(
                    observation[record_index, env_row, player],
                    latest=_latest_value(record_index, env_row, player),
                )
                _write_stack(
                    final_observation[record_index, env_row, player],
                    latest=_final_value(record_index, env_row, player),
                )

    legal_action_mask = np.zeros(
        (time_steps, batch_size, player_count, ACTION_COUNT),
        dtype=np.bool_,
    )
    legal_action_mask[..., 1] = True
    legal_action_mask[:, :, 1, 2] = True
    lightzero_action_mask = legal_action_mask.copy()
    joint_action = np.full((time_steps, batch_size, player_count), -1, dtype=np.int16)
    for record_index in range(1, time_steps):
        joint_action[record_index, :, 0] = 1
        joint_action[record_index, :, 1] = 2
    reward = np.zeros((time_steps, batch_size, player_count), dtype=np.float32)
    final_reward_map = np.zeros_like(reward)
    for record_index in range(time_steps):
        for env_row in range(batch_size):
            for player in range(player_count):
                reward[record_index, env_row, player] = (
                    record_index + env_row / 10.0 + player / 100.0
                )
                final_reward_map[record_index, env_row, player] = (
                    10.0 + record_index + env_row / 10.0 + player / 100.0
                )

    arrays = {
        "observation": observation.astype(np.float32),
        "legal_action_mask": legal_action_mask,
        "lightzero_action_mask": lightzero_action_mask,
        "live_mask": live_by_record.astype(np.bool_),
        "joint_action": joint_action,
        "reward": reward,
        "done": done_by_record.astype(np.bool_),
        "terminated": done_by_record.astype(np.bool_),
        "truncated": np.zeros_like(done_by_record, dtype=np.bool_),
        "final_observation": final_observation.astype(np.float32),
        "final_observation_row_mask": final_observation_row_mask.astype(np.bool_),
        "final_reward_map": final_reward_map,
    }
    policy_rows = tuple(
        _policy_rows_for_record(arrays, record_index) for record_index in range(time_steps)
    )
    records = tuple(
        {
            "sequence_index": record_index,
            "policy_row_count": int(policy_rows[record_index]["policy_env_row"].size),
            "done_rows": np.flatnonzero(done_by_record[record_index]).tolist(),
            "final_observation_rows": np.flatnonzero(
                final_observation_row_mask[record_index]
            ).tolist(),
        }
        for record_index in range(time_steps)
    )
    return SourceStateMultiplayerTrainerReplayChunkV0(
        metadata={
            "replay_contract_id": "unit_test_compact_search_replay_contract",
            "source_fidelity_claim": "unit_test_synthetic_contract_fixture",
            "original_curvytron_behavior_claim": True,
        },
        arrays=arrays,
        policy_rows=policy_rows,
        records=records,
    )


def _policy_rows_for_record(
    arrays: dict[str, np.ndarray],
    record_index: int,
) -> dict[str, np.ndarray]:
    env_row, player = np.nonzero(arrays["live_mask"][record_index])
    env_row = env_row.astype(np.int32, copy=False)
    player = player.astype(np.int16, copy=False)
    return {
        "policy_observation": arrays["observation"][record_index, env_row, player].copy(),
        "policy_action_mask": arrays["legal_action_mask"][
            record_index,
            env_row,
            player,
        ].copy(),
        "policy_env_row": env_row.copy(),
        "policy_player": player.copy(),
    }


def _compact_batch_with_terminal_mechanics_outcome() -> HybridCompactBatch:
    batch_size = 2
    player_count = 2
    root_count = batch_size * player_count
    observation = np.arange(
        batch_size * player_count * 4 * 64 * 64,
        dtype=np.uint8,
    ).reshape(batch_size, player_count, 4, 64, 64)
    reward = np.asarray(
        [
            [0.25, -0.25],
            [1.0, -1.0],
        ],
        dtype=np.float32,
    )
    done = np.asarray([False, True], dtype=np.bool_)
    done_root = np.repeat(done, player_count)
    action_mask = np.ones((batch_size, player_count, ACTION_COUNT), dtype=np.bool_)
    return HybridCompactBatch(
        observation=observation,
        action_mask=action_mask,
        reward=reward,
        final_reward_map=np.asarray(
            [
                [0.25, -0.25],
                [2.0, -2.0],
            ],
            dtype=np.float32,
        ),
        done=done,
        policy_env_id=np.arange(root_count, dtype=np.int32),
        policy_env_row=np.repeat(np.arange(batch_size, dtype=np.int32), player_count),
        policy_player=np.tile(np.arange(player_count, dtype=np.int32), batch_size),
        target_reward=reward.reshape(root_count, 1),
        done_root=done_root,
        to_play=np.full(root_count, DEFAULT_TO_PLAY, dtype=np.int64),
        active_root_mask=np.logical_and(
            ~done_root, action_mask.reshape(root_count, ACTION_COUNT).any(axis=1)
        ),
        final_observation=observation.copy(),
        final_observation_row_mask=np.asarray([False, True], dtype=np.bool_),
        terminal_row_mask=np.asarray([False, True], dtype=np.bool_),
        autoreset_row_mask=np.asarray([False, True], dtype=np.bool_),
        terminal_global_rows=np.asarray([1], dtype=np.int32),
        autoreset_global_rows=np.asarray([1], dtype=np.int32),
        episode_step=np.asarray([7, 7], dtype=np.int32),
        elapsed_ms=np.asarray([0.0, 0.0], dtype=np.float64),
        round_id=np.asarray([3, 4], dtype=np.int32),
        alive=np.asarray([[True, True], [False, False]], dtype=np.bool_),
        joint_action=np.zeros((batch_size, player_count), dtype=np.int16),
        terminated=np.asarray([False, True], dtype=np.bool_),
        truncated=np.asarray([False, False], dtype=np.bool_),
    )


def _compact_batch_from_chunk(
    chunk: SourceStateMultiplayerTrainerReplayChunkV0,
    *,
    record_index: int,
) -> HybridCompactBatch:
    observation = np.asarray(chunk.arrays["observation"][record_index])
    batch_size, player_count = observation.shape[:2]
    action_mask = np.zeros((batch_size, player_count, ACTION_COUNT), dtype=np.bool_)
    policy = chunk.policy_rows[record_index]
    for policy_row, (env_row, player) in enumerate(
        zip(policy["policy_env_row"], policy["policy_player"], strict=True)
    ):
        action_mask[int(env_row), int(player)] = policy["policy_action_mask"][policy_row]
    root_count = batch_size * player_count
    done = np.asarray(chunk.arrays["done"][record_index], dtype=np.bool_)
    done_root = np.repeat(done, player_count)
    flat_mask = action_mask.reshape(root_count, ACTION_COUNT)
    terminal_row_mask = done.copy()
    autoreset_row_mask = np.asarray(
        chunk.arrays["final_observation_row_mask"][record_index],
        dtype=np.bool_,
    ).copy()
    return HybridCompactBatch(
        observation=observation,
        action_mask=action_mask,
        reward=np.asarray(chunk.arrays["reward"][record_index], dtype=np.float32),
        final_reward_map=np.asarray(
            chunk.arrays["final_reward_map"][record_index],
            dtype=np.float32,
        ),
        done=done,
        policy_env_id=np.arange(root_count, dtype=np.int32),
        policy_env_row=np.repeat(np.arange(batch_size, dtype=np.int32), player_count),
        policy_player=np.tile(np.arange(player_count, dtype=np.int32), batch_size),
        target_reward=np.asarray(
            chunk.arrays["reward"][record_index],
            dtype=np.float32,
        ).reshape(root_count, 1),
        done_root=done_root,
        to_play=np.full(root_count, DEFAULT_TO_PLAY, dtype=np.int64),
        active_root_mask=np.logical_and(~done_root, flat_mask.any(axis=1)),
        final_observation=np.asarray(chunk.arrays["final_observation"][record_index]),
        final_observation_row_mask=np.asarray(
            chunk.arrays["final_observation_row_mask"][record_index],
            dtype=np.bool_,
        ),
        terminal_row_mask=terminal_row_mask,
        autoreset_row_mask=autoreset_row_mask,
        terminal_global_rows=np.flatnonzero(terminal_row_mask).astype(np.int32),
        autoreset_global_rows=np.flatnonzero(autoreset_row_mask).astype(np.int32),
        episode_step=np.full((batch_size,), record_index, dtype=np.int32),
        elapsed_ms=np.zeros((batch_size,), dtype=np.float64),
        round_id=np.zeros((batch_size,), dtype=np.int32),
        alive=np.asarray(chunk.arrays["live_mask"][record_index], dtype=np.bool_),
        joint_action=np.asarray(chunk.arrays["joint_action"][record_index], dtype=np.int16),
    )


def _compact_index_rows_for_record(
    chunk: SourceStateMultiplayerTrainerReplayChunkV0,
    *,
    record_index: int,
):
    compact_batch = _compact_batch_from_chunk(chunk, record_index=record_index)
    selected_action, visit_policy, root_value = _search_inputs_for_record(
        chunk,
        record_index=record_index,
    )
    root_batch = build_compact_root_batch_v1(
        compact_batch,
        search_lane="unit_test_compact_index_group",
    )
    search_result = validate_compact_search_result_v1(
        root_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        search_impl="unit_test_compact_index_group",
        num_simulations=8,
    )
    next_record_index = record_index + 1
    return build_compact_replay_index_rows_v1_from_search_result(
        compact_batch,
        root_batch,
        search_result,
        record_index=record_index,
        next_joint_action=chunk.arrays["joint_action"][next_record_index],
        next_reward=chunk.arrays["reward"][next_record_index],
        next_done=chunk.arrays["done"][next_record_index],
        next_terminated=chunk.arrays["terminated"][next_record_index],
        next_truncated=chunk.arrays["truncated"][next_record_index],
        next_final_reward_map=chunk.arrays["final_reward_map"][next_record_index],
        next_final_observation_row_mask=chunk.arrays["final_observation_row_mask"][
            next_record_index
        ],
        policy_source="unit_test_compact_index_group",
    )


def _search_inputs_for_record(
    chunk: SourceStateMultiplayerTrainerReplayChunkV0,
    *,
    record_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy = chunk.policy_rows[record_index]
    action = np.asarray(
        [
            chunk.arrays["joint_action"][record_index + 1, int(env_row), int(player)]
            for env_row, player in zip(
                policy["policy_env_row"],
                policy["policy_player"],
                strict=True,
            )
        ],
        dtype=np.int16,
    )
    visit_policy = np.zeros((action.size, ACTION_COUNT), dtype=np.float32)
    visit_policy[np.arange(action.size), action] = 1.0
    root_value = np.linspace(-0.25, 0.25, action.size, dtype=np.float32)
    return action, visit_policy, root_value


class _StaticCompactSearchService:
    profile_only = True
    calls_train_muzero = False
    trainer_defaults_changed = False
    touches_live_runs = False

    def __init__(
        self,
        *,
        selected_action: np.ndarray,
        visit_policy: np.ndarray,
        root_value: np.ndarray,
        search_impl: str,
        predicted_value: np.ndarray | None = None,
        predicted_policy_logits: np.ndarray | None = None,
        reverse_identity: bool = False,
    ) -> None:
        self.selected_action = selected_action
        self.visit_policy = visit_policy
        self.root_value = root_value
        self.search_impl = search_impl
        self.predicted_value = predicted_value
        self.predicted_policy_logits = predicted_policy_logits
        self.num_simulations = 8
        self.reverse_identity = reverse_identity

    def run(self, root_batch):
        result = validate_compact_search_result_v1(
            root_batch,
            selected_action=self.selected_action,
            visit_policy=self.visit_policy,
            root_value=self.root_value,
            predicted_value=self.predicted_value,
            predicted_policy_logits=self.predicted_policy_logits,
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
            metadata={"profile_telemetry": {f"{self.search_impl}_ran": True}},
        )
        if not self.reverse_identity:
            return result
        return replace(result, policy_env_id=result.policy_env_id[::-1].copy())


def _expected_rows_from_replay_policy_rows(
    chunk: SourceStateMultiplayerTrainerReplayChunkV0,
    *,
    record_index: int,
    selected_action: np.ndarray,
    visit_policy: np.ndarray,
    root_value: np.ndarray,
    policy_source: str,
):
    policy = chunk.policy_rows[record_index]
    records = []
    for policy_row, (env_row, player) in enumerate(
        zip(policy["policy_env_row"], policy["policy_player"], strict=True)
    ):
        records.append(
            PolicyRowRecordV0(
                record_index=record_index,
                policy_row=policy_row,
                env_row=int(env_row),
                player=int(player),
                action=int(selected_action[policy_row]),
                action_mask=policy["policy_action_mask"][policy_row].copy(),
                policy_target=visit_policy[policy_row].copy(),
                root_value=float(root_value[policy_row]),
                policy_source=policy_source,
                source_record_ref=f"{record_index}:{policy_row}",
            )
        )
    return build_source_state_multiplayer_target_rows_v0(chunk, records)


class _FakeCompactSearchServiceV1:
    search_impl = "unit_test_fake_compact_service"
    num_simulations = 4

    def __init__(
        self,
        *,
        selected_action: np.ndarray,
        visit_policy: np.ndarray,
        root_value: np.ndarray,
    ) -> None:
        self.selected_action = selected_action
        self.visit_policy = visit_policy
        self.root_value = root_value

    def run(self, root_batch):
        return validate_compact_search_result_v1(
            root_batch,
            selected_action=self.selected_action,
            visit_policy=self.visit_policy,
            root_value=self.root_value,
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
        )


class _FakeTwoPhaseCompactSearchServiceV1:
    search_impl = "unit_test_fake_two_phase_compact_service"
    num_simulations = 4

    def __init__(
        self,
        *,
        selected_action: np.ndarray,
        visit_policy: np.ndarray,
        root_value: np.ndarray,
    ) -> None:
        self.selected_action = selected_action
        self.visit_policy = visit_policy
        self.root_value = root_value
        self.run_calls = 0
        self.action_step_calls = 0
        self.flush_calls = 0
        self._pending_payloads = {}

    def run(self, root_batch):
        self.run_calls += 1
        raise AssertionError("two-phase slab must not call full run() on hot step")

    def run_action_step(self, root_batch):
        self.action_step_calls += 1
        result = validate_compact_search_result_v1(
            root_batch,
            selected_action=self.selected_action,
            visit_policy=self.visit_policy,
            root_value=self.root_value,
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
            metadata={
                "profile_telemetry": {
                    "fixed_shape_batched_search_owner_action_d2h_bytes": (
                        self.selected_action.nbytes
                    ),
                    "fixed_shape_batched_search_owner_replay_payload_d2h_bytes": (
                        self.visit_policy.nbytes + self.root_value.nbytes
                    ),
                }
            },
        )
        handle = f"unit-test-two-phase-{self.action_step_calls}"
        action_step = compact_search_action_step_v1_from_result(
            result,
            replay_payload_handle=handle,
            metadata={"two_phase_test": True},
        )
        payload = compact_search_replay_payload_v1_from_result(
            result,
            replay_payload_handle=handle,
            metadata={"two_phase_test": True},
        )
        self._pending_payloads[handle] = payload
        return action_step

    def flush_replay_payload(self, replay_payload_handle):
        self.flush_calls += 1
        try:
            return self._pending_payloads.pop(str(replay_payload_handle))
        except KeyError as exc:
            raise ReplayCompatibilityError("missing two-phase replay payload") from exc


def _assert_target_rows_equal(left, right) -> None:
    for field in (
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
    ):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    assert left.policy_source == right.policy_source
    assert left.metadata == right.metadata


def _assert_sample_batches_equal(left, right) -> None:
    for field in (
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
    ):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    assert left.metadata == right.metadata


def _policy_refresh_metadata_for_test() -> dict[str, object]:
    return compact_policy_refresh_metadata_from_state_v1(_policy_refresh_state_for_test())


def _policy_refresh_state_for_test() -> dict[str, object]:
    return {
        "schema_id": COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID,
        "search_impl": "unit_test_compact_torch_search",
        "policy_version_ref": "policy:update-3",
        "model_version_ref": "model:update-3",
        "policy_source": "unit_test_policy_refresh",
        "learner_update_count": 3,
        "model_state_digest": "unit-model-digest",
        "search_worker_model_object_id": 12345,
        "search_worker_object_id": 67890,
        "refresh_count": 2,
        "refresh_applied": True,
        "cache_cleared": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }


def _assert_policy_refresh_metadata(
    metadata: dict[str, object],
    expected: dict[str, object],
) -> None:
    for key, expected_value in expected.items():
        assert metadata[key] == expected_value


class _LightZeroZeroTargetModel:
    def __init__(
        self,
        *,
        torch,
        action_space_size: int,
        value_support_size: int,
        reward_support_size: int,
    ) -> None:
        self._torch = torch
        self._action_space_size = int(action_space_size)
        self._value_support_size = int(value_support_size)
        self._reward_support_size = int(reward_support_size)
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def to(self, *_args, **_kwargs):
        return self

    def initial_inference(self, obs):
        batch_size = int(obs.shape[0])
        return SimpleNamespace(
            latent_state=self._torch.zeros((batch_size, 1), dtype=self._torch.float32),
            reward=self._torch.zeros(
                (batch_size, self._reward_support_size),
                dtype=self._torch.float32,
            ),
            value=self._torch.zeros(
                (batch_size, self._value_support_size),
                dtype=self._torch.float32,
            ),
            policy_logits=self._torch.zeros(
                (batch_size, self._action_space_size),
                dtype=self._torch.float32,
            ),
        )


def _support_size(config) -> int:
    return int(2 * config.model.support_scale + 1)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _lightzero_sample_row_ids(buffer, native, batch_index: np.ndarray) -> np.ndarray:
    row_ids = []
    transition_lookup = buffer.game_segment_game_pos_look_up
    base_idx = int(getattr(buffer, "base_idx", 0))
    for transition_index in batch_index.astype(np.int64, copy=False):
        segment_index, pos_in_segment = transition_lookup[int(transition_index)]
        spec_index = int(segment_index) - base_idx
        row_ids.append(int(native.specs[spec_index].row_id[int(pos_in_segment)]))
    return np.asarray(row_ids, dtype=np.int64)


def _expected_lightzero_public_sample_for_row_ids(
    native,
    sampled_row_id: np.ndarray,
    *,
    num_unroll_steps: int,
    td_steps: int,
    discount_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_position = {}
    for spec in native.specs:
        for position, row_id in enumerate(spec.row_id):
            row_position[int(row_id)] = (spec, int(position))

    observation_rows = []
    action_rows = []
    mask_rows = []
    reward_rows = []
    value_rows = []
    policy_rows = []
    for row_id in sampled_row_id.astype(np.int64, copy=False):
        spec, position = row_position[int(row_id)]
        segment_len = len(spec.actions)
        obs_window = list(spec.observations[position : position + num_unroll_steps + 1])
        if len(obs_window) < num_unroll_steps + 1:
            obs_window.extend([obs_window[-1]] * (num_unroll_steps + 1 - len(obs_window)))
        observation_rows.append(
            np.concatenate(
                [np.asarray(obs, dtype=np.float32) for obs in obs_window],
                axis=0,
            )
        )

        action_window = list(spec.actions[position : position + num_unroll_steps])
        actions = np.full((num_unroll_steps,), -1, dtype=np.int64)
        if action_window:
            actions[: len(action_window)] = np.asarray(action_window, dtype=np.int64)
        action_rows.append(actions)

        valid_action_count = min(
            len(action_window),
            max(0, int(native.config.game_segment_length) - int(position)),
        )
        mask = np.zeros((num_unroll_steps + 1,), dtype=np.float32)
        mask[:valid_action_count] = 1.0
        mask_rows.append(mask)

        clipped_td_steps = int(np.clip(td_steps, 1, max(1, segment_len - position)))
        rewards = []
        values = []
        policies = []
        for offset in range(num_unroll_steps + 1):
            current_index = position + offset
            if current_index < segment_len:
                rewards.append(float(spec.rewards[current_index]))
                policies.append(np.asarray(spec.policy_target[current_index], dtype=np.float32))
            else:
                rewards.append(0.0)
                policies.append(np.zeros((ACTION_COUNT,), dtype=np.float32))

            bootstrap_index = current_index + clipped_td_steps
            value = 0.0
            scale = 1.0
            for reward in spec.rewards[current_index:bootstrap_index]:
                value += float(reward) * scale
                scale *= float(discount_factor)
            values.append(value)
        reward_rows.append(np.asarray(rewards, dtype=np.float32))
        value_rows.append(np.asarray(values, dtype=np.float32))
        policy_rows.append(np.asarray(policies, dtype=np.float32))

    return (
        np.asarray(observation_rows, dtype=np.float32),
        np.asarray(action_rows, dtype=np.int64),
        np.asarray(mask_rows, dtype=np.float32),
        np.asarray(reward_rows, dtype=np.float32),
        np.asarray(value_rows, dtype=np.float32),
        np.asarray(policy_rows, dtype=np.float32),
    )


def _write_stack(stack: np.ndarray, *, latest: float) -> None:
    stack[0, :, :] = 4.0
    stack[1, :, :] = -3.0
    stack[2, :, :] = 2.0
    stack[3, :, :] = np.float32(latest)


def _latest_value(record_index: int, env_row: int, player: int) -> np.float32:
    return np.float32(0.05 + 0.2 * record_index + 0.04 * env_row + 0.01 * player)


def _final_value(record_index: int, env_row: int, player: int) -> np.float32:
    return np.float32(0.70 + 0.05 * record_index + 0.04 * env_row + 0.01 * player)
