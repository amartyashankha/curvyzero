import math

import numpy as np
import pytest

from curvyzero.env import vector_runtime
from curvyzero.training.multiplayer_source_state_target_rows import DEFAULT_TO_PLAY
from curvyzero.training.multiplayer_source_state_target_rows import (
    PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_target_rows import PolicyRowRecordV0
from curvyzero.training.multiplayer_source_state_target_rows import (
    build_source_state_multiplayer_sample_batch_v0,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    build_source_state_multiplayer_target_rows_v0,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_target_rows_from_search_arrays_v0,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_policy_row_records_from_compact_search_v0,
)
from curvyzero.training.compact_policy_row_bridge import validate_compact_policy_search_arrays_v0
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayChunkV0,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.exploration_bonus import (
    extract_policy_gray64_latest_for_rnd_from_compact_observation,
)
from curvyzero.training.source_state_hybrid_observation_profile import HybridCompactBatch


def test_target_rows_build_reset_to_step_alignment_without_lightzero_claims():
    chunk = _nonterminal_chunk(player_count=2)
    records = _policy_records_for_record(chunk, record_index=0)

    rows = build_source_state_multiplayer_target_rows_v0(chunk, records)

    assert rows.metadata["target_contract_id"] == (SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID)
    assert rows.metadata["source_replay_contract_id"] == (
        SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID
    )
    assert rows.metadata["native_game_segment_claim"] is False
    assert rows.metadata["lightzero_training_integration_claim"] is False
    assert rows.metadata["target_row_count"] == 2
    assert rows.observation.shape == (2, 4, 64, 64)
    np.testing.assert_array_equal(rows.to_play, np.full(2, DEFAULT_TO_PLAY))
    np.testing.assert_array_equal(rows.env_row, np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(rows.player, np.asarray([0, 1], dtype=np.int16))
    np.testing.assert_array_equal(rows.policy_row, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(rows.action, np.asarray([1, 1], dtype=np.int16))
    np.testing.assert_array_equal(
        rows.reward,
        chunk.arrays["reward"][1, 0, :2],
    )
    np.testing.assert_array_equal(
        rows.observation,
        chunk.arrays["observation"][0, 0, :2],
    )
    np.testing.assert_array_equal(
        rows.next_observation,
        chunk.arrays["observation"][1, 0, :2],
    )


def test_target_rows_accept_compact_mcts_visit_policy_targets():
    chunk = _nonterminal_chunk(player_count=2)
    records = _policy_records_for_record(chunk, record_index=0)
    visit_policy_targets = (
        np.asarray([0.25, 0.5, 0.25], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    records = [
        PolicyRowRecordV0(
            record_index=record.record_index,
            policy_row=record.policy_row,
            env_row=record.env_row,
            player=record.player,
            action=record.action,
            action_mask=record.action_mask,
            policy_target=visit_policy_targets[index],
            root_value=record.root_value + 1.0,
            policy_source="direct_ctree_arrays_profile_test",
            source_record_ref=record.source_record_ref,
        )
        for index, record in enumerate(records)
    ]

    rows = build_source_state_multiplayer_target_rows_v0(chunk, records)

    np.testing.assert_allclose(rows.policy_target, np.stack(visit_policy_targets))
    np.testing.assert_array_equal(rows.action, np.asarray([1, 1], dtype=np.int16))
    np.testing.assert_allclose(rows.root_value, np.asarray([1.0, 1.1], dtype=np.float32))
    assert rows.policy_source == (
        "direct_ctree_arrays_profile_test",
        "direct_ctree_arrays_profile_test",
    )


def test_compact_search_outputs_build_checked_policy_records_for_target_rows():
    chunk = _nonterminal_chunk(player_count=2)
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=0)
    selected_action = chunk.arrays["joint_action"][1, 0, :2].astype(np.int16)
    visit_policy = np.asarray(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    root_value = np.asarray([0.25, -0.5], dtype=np.float32)

    records = build_policy_row_records_from_compact_search_v0(
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=0,
        policy_source="compact_direct_ctree_arrays_profile_test",
    )
    rows = build_source_state_multiplayer_target_rows_v0(chunk, records)

    assert len(records) == 2
    np.testing.assert_array_equal(rows.action, selected_action)
    np.testing.assert_allclose(rows.policy_target, visit_policy)
    np.testing.assert_allclose(rows.root_value, root_value)
    np.testing.assert_array_equal(rows.policy_row, np.asarray([0, 1], dtype=np.int32))
    assert rows.policy_source == (
        "compact_direct_ctree_arrays_profile_test",
        "compact_direct_ctree_arrays_profile_test",
    )
    assert rows.source_record_ref[0]["contract_id"].endswith(
        "compact_policy_row_records_from_search/v0"
    )


def test_compact_search_arrays_validate_without_policy_record_objects():
    chunk = _nonterminal_chunk(player_count=2)
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=0)
    selected_action = chunk.arrays["joint_action"][1, 0, :2].astype(np.int16)
    visit_policy = np.asarray(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    root_value = np.asarray([0.25, -0.5], dtype=np.float32)

    arrays = validate_compact_policy_search_arrays_v0(
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=0,
        policy_source="compact_array_target_test",
    )

    np.testing.assert_array_equal(arrays.policy_row, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(arrays.env_row, np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(arrays.player, np.asarray([0, 1], dtype=np.int16))
    np.testing.assert_array_equal(arrays.action, selected_action)
    np.testing.assert_allclose(arrays.policy_target, visit_policy)
    np.testing.assert_allclose(arrays.root_value, root_value)
    assert arrays.policy_source == "compact_array_target_test"


def test_compact_search_arrays_build_target_rows_without_policy_record_objects():
    chunk = _nonterminal_chunk(player_count=2)
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=0)
    selected_action = chunk.arrays["joint_action"][1, 0, :2].astype(np.int16)
    visit_policy = np.asarray(
        [
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )
    root_value = np.asarray([0.25, -0.5], dtype=np.float32)

    object_rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        build_policy_row_records_from_compact_search_v0(
            compact_batch,
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            record_index=0,
            policy_source="compact_target_array_parity_test",
        ),
    )
    compact_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=0,
        policy_source="compact_target_array_parity_test",
    )

    _assert_target_rows_equal(compact_rows, object_rows)


def test_compact_search_arrays_use_terminal_final_observation_without_records():
    chunk = _terminal_chunk()
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=0)
    selected_action = chunk.arrays["joint_action"][1, 0, :2].astype(np.int16)
    visit_policy = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    root_value = np.asarray([1.25, -1.5], dtype=np.float32)

    object_rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        build_policy_row_records_from_compact_search_v0(
            compact_batch,
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            record_index=0,
            policy_source="compact_terminal_target_array_parity_test",
        ),
    )
    compact_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=0,
        policy_source="compact_terminal_target_array_parity_test",
    )

    _assert_target_rows_equal(compact_rows, object_rows)
    for index, player in enumerate(compact_rows.player):
        np.testing.assert_array_equal(
            compact_rows.next_observation[index],
            chunk.arrays["final_observation"][1, 0, int(player)],
        )


def test_compact_target_rows_map_non_prefix_active_roots_to_replay_policy_rows():
    chunk = _chunk_with_live_policy_seats([(0, 1), (1, 0), (1, 1)])
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=0)
    compact_batch = _replace_compact_batch_for_test(
        compact_batch,
        policy_env_id=np.asarray([101, 103, 107, 109], dtype=np.int32),
    )
    selected_action = np.asarray([1, 1, 1], dtype=np.int16)
    visit_policy = np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (3, 1))
    root_value = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)

    rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=0,
        policy_source="compact_non_prefix_policy_row_test",
    )

    np.testing.assert_array_equal(rows.env_row, np.asarray([0, 1, 1], dtype=np.int32))
    np.testing.assert_array_equal(rows.player, np.asarray([1, 0, 1], dtype=np.int16))
    np.testing.assert_array_equal(rows.policy_row, np.asarray([0, 1, 2], dtype=np.int32))
    np.testing.assert_allclose(rows.root_value, root_value)
    assert [ref["compact_root_row"] for ref in rows.source_record_ref] == [1, 2, 3]
    assert [ref["policy_env_id"] for ref in rows.source_record_ref] == [103, 107, 109]


def test_compact_target_rows_reject_swapped_player_perspective_batch():
    chunk = _nonterminal_chunk(batch_size=1, player_count=2)
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=0)
    swapped_observation = compact_batch.observation.copy()
    swapped_observation[:, [0, 1]] = swapped_observation[:, [1, 0]]

    with pytest.raises(ReplayCompatibilityError, match="observation does not match"):
        build_compact_target_rows_from_search_arrays_v0(
            chunk,
            _replace_compact_batch_for_test(compact_batch, observation=swapped_observation),
            selected_action=chunk.arrays["joint_action"][1, 0, :2].astype(np.int16),
            visit_policy=np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (2, 1)),
            root_value=np.asarray([0.0, 0.0], dtype=np.float32),
            record_index=0,
            policy_source="compact_swapped_perspective_reject_test",
        )


def test_compact_target_rows_use_record_index_one_in_three_record_chunk():
    chunk = _three_record_chunk(player_count=2)
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=1)
    selected_action = chunk.arrays["joint_action"][2, 0, :2].astype(np.int16)
    visit_policy = np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (2, 1))
    root_value = np.asarray([0.6, -0.4], dtype=np.float32)

    object_rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        build_policy_row_records_from_compact_search_v0(
            compact_batch,
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            record_index=1,
            policy_source="compact_record_one_parity_test",
        ),
    )
    compact_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=1,
        policy_source="compact_record_one_parity_test",
    )

    _assert_target_rows_equal(compact_rows, object_rows)
    np.testing.assert_array_equal(compact_rows.observation, chunk.arrays["observation"][1, 0, :2])
    np.testing.assert_array_equal(
        compact_rows.next_observation,
        chunk.arrays["observation"][2, 0, :2],
    )
    np.testing.assert_array_equal(compact_rows.record_index, np.asarray([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(
        compact_rows.next_record_index,
        np.asarray([2, 2], dtype=np.int32),
    )


def test_compact_target_rows_preserve_rnd_latest_frame_order_without_records():
    chunk = _chunk_with_latest_channel_sentinels([(0, 1), (1, 0), (1, 1)])
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=0)
    selected_action = np.asarray([1, 1, 1], dtype=np.int16)

    rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        compact_batch,
        selected_action=selected_action,
        visit_policy=np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (3, 1)),
        root_value=np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
        record_index=0,
        policy_source="compact_rnd_latest_order_test",
    )
    rnd_input = extract_policy_gray64_latest_for_rnd_from_compact_observation(
        rows.observation,
        np.zeros((3, 1), dtype=np.float32),
    )

    assert rnd_input.shape == (3, 1, 64, 64)
    np.testing.assert_allclose(rnd_input[:, 0, 0, 0], np.asarray([0.21, 0.31, 0.41]))


def test_compact_search_policy_records_skip_done_roots_before_target_rows():
    chunk = _nonterminal_chunk(player_count=2)
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=0)
    compact_batch = _replace_compact_batch_for_test(
        compact_batch,
        done=np.asarray([True], dtype=np.bool_),
        done_root=np.asarray([True, True], dtype=np.bool_),
        active_root_mask=np.asarray([False, False], dtype=np.bool_),
    )

    records = build_policy_row_records_from_compact_search_v0(
        compact_batch,
        selected_action=np.zeros((0,), dtype=np.int16),
        visit_policy=np.zeros((0, 3), dtype=np.float32),
        root_value=np.zeros((0,), dtype=np.float32),
        record_index=0,
        policy_source="compact_done_root_skip_test",
    )
    rows = build_source_state_multiplayer_target_rows_v0(chunk, records)

    assert records == []
    assert rows.metadata["target_row_count"] == 0
    assert rows.observation.shape == (0, 4, 64, 64)


def test_compact_search_policy_records_keep_active_root_order_for_p4_and_mixed_done():
    p4_chunk = _nonterminal_chunk(player_count=4)
    p4_batch = _compact_batch_from_chunk_policy_record(p4_chunk, record_index=0)

    p4_records = build_policy_row_records_from_compact_search_v0(
        p4_batch,
        selected_action=p4_chunk.arrays["joint_action"][1, 0, :4].astype(np.int16),
        visit_policy=np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32), (4, 1)),
        root_value=np.arange(4, dtype=np.float32),
        record_index=0,
        policy_source="compact_p4_active_order_test",
    )
    p4_rows = build_source_state_multiplayer_target_rows_v0(p4_chunk, p4_records)

    np.testing.assert_array_equal(p4_rows.player, np.arange(4, dtype=np.int16))
    np.testing.assert_array_equal(p4_rows.policy_row, np.arange(4, dtype=np.int32))
    np.testing.assert_allclose(p4_rows.root_value, np.arange(4, dtype=np.float32))

    mixed_chunk = _nonterminal_chunk(batch_size=2, player_count=2)
    mixed_batch = _compact_batch_from_chunk_policy_record(mixed_chunk, record_index=0)
    mixed_batch = _replace_compact_batch_for_test(
        mixed_batch,
        done=np.asarray([False, True], dtype=np.bool_),
        done_root=np.asarray([False, False, True, True], dtype=np.bool_),
        active_root_mask=np.asarray([True, True, False, False], dtype=np.bool_),
    )

    mixed_records = build_policy_row_records_from_compact_search_v0(
        mixed_batch,
        selected_action=mixed_chunk.arrays["joint_action"][1, 0, :2].astype(np.int16),
        visit_policy=np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        root_value=np.asarray([3.0, 4.0], dtype=np.float32),
        record_index=0,
        policy_source="compact_mixed_active_order_test",
    )
    mixed_rows = build_source_state_multiplayer_target_rows_v0(mixed_chunk, mixed_records)

    np.testing.assert_array_equal(mixed_rows.env_row, np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_array_equal(mixed_rows.player, np.asarray([0, 1], dtype=np.int16))
    np.testing.assert_array_equal(mixed_rows.policy_row, np.asarray([0, 1], dtype=np.int32))
    np.testing.assert_allclose(mixed_rows.root_value, np.asarray([3.0, 4.0], dtype=np.float32))


def test_compact_search_policy_records_reject_bad_sidecars_before_replay():
    chunk = _nonterminal_chunk(player_count=2)
    compact_batch = _compact_batch_from_chunk_policy_record(chunk, record_index=0)
    selected_action = np.asarray([1, 1], dtype=np.int16)
    visit_policy = np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    root_value = np.asarray([0.0, 0.0], dtype=np.float32)

    fractional_mask = compact_batch.action_mask.astype(np.float32)
    fractional_mask[0, 0, 0] = 0.5
    with pytest.raises(ReplayCompatibilityError, match="action_mask must be binary"):
        build_policy_row_records_from_compact_search_v0(
            _replace_compact_batch_for_test(compact_batch, action_mask=fractional_mask),
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            record_index=0,
            policy_source="compact_bad_mask_test",
        )

    illegal_mask = compact_batch.action_mask.copy()
    illegal_mask[0, 0, 1] = False
    with pytest.raises(ReplayCompatibilityError, match="selected_action is illegal"):
        build_policy_row_records_from_compact_search_v0(
            _replace_compact_batch_for_test(compact_batch, action_mask=illegal_mask),
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            record_index=0,
            policy_source="compact_illegal_action_test",
        )

    illegal_visit = visit_policy.copy()
    illegal_visit[0] = np.asarray([0.2, 0.8, 0.0], dtype=np.float32)
    illegal_visit_mask = compact_batch.action_mask.copy()
    illegal_visit_mask[0, 0, 0] = False
    with pytest.raises(ReplayCompatibilityError, match="illegal actions"):
        build_policy_row_records_from_compact_search_v0(
            _replace_compact_batch_for_test(compact_batch, action_mask=illegal_visit_mask),
            selected_action=selected_action,
            visit_policy=illegal_visit,
            root_value=root_value,
            record_index=0,
            policy_source="compact_illegal_visit_test",
        )

    with pytest.raises(ReplayCompatibilityError, match="active_root_mask"):
        build_policy_row_records_from_compact_search_v0(
            _replace_compact_batch_for_test(
                compact_batch,
                active_root_mask=np.asarray([True, False], dtype=np.bool_),
            ),
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=root_value,
            record_index=0,
            policy_source="compact_bad_active_test",
        )


def test_target_rows_use_terminal_final_observation_and_final_reward_map():
    chunk = _terminal_chunk()
    records = _policy_records_for_record(chunk, record_index=0)

    rows = build_source_state_multiplayer_target_rows_v0(chunk, records)

    np.testing.assert_array_equal(rows.done, np.asarray([True, True], dtype=bool))
    np.testing.assert_array_equal(rows.terminated, np.asarray([True, True], dtype=bool))
    np.testing.assert_array_equal(rows.truncated, np.asarray([False, False], dtype=bool))
    for index, player in enumerate(rows.player):
        np.testing.assert_array_equal(
            rows.next_observation[index],
            chunk.arrays["final_observation"][1, 0, int(player)],
        )
        assert rows.final_reward[index] == pytest.approx(
            float(chunk.arrays["final_reward_map"][1, 0, int(player)])
        )


def test_target_rows_map_p4_live_rows_and_keep_to_play_non_board_game():
    chunk = _nonterminal_chunk(player_count=4)
    records = _policy_records_for_record(chunk, record_index=0)

    rows = build_source_state_multiplayer_target_rows_v0(chunk, records)

    assert rows.metadata["target_row_count"] == 4
    np.testing.assert_array_equal(rows.env_row, np.zeros(4, dtype=np.int32))
    np.testing.assert_array_equal(rows.player, np.arange(4, dtype=np.int16))
    np.testing.assert_array_equal(rows.to_play, np.full(4, DEFAULT_TO_PLAY))
    np.testing.assert_array_equal(rows.action, np.ones(4, dtype=np.int16))


def test_target_rows_reject_bad_policy_target_and_action_mismatch():
    chunk = _nonterminal_chunk(player_count=2)
    records = _policy_records_for_record(chunk, record_index=0)
    records[0] = PolicyRowRecordV0(
        record_index=records[0].record_index,
        policy_row=records[0].policy_row,
        env_row=records[0].env_row,
        player=records[0].player,
        action=records[0].action,
        action_mask=np.asarray([True, False, True], dtype=bool),
        policy_target=np.asarray([0.2, 0.3, 0.5], dtype=np.float32),
        root_value=records[0].root_value,
        policy_source=records[0].policy_source,
    )
    with pytest.raises(ReplayCompatibilityError, match="action_mask"):
        build_source_state_multiplayer_target_rows_v0(chunk, records)

    records = _policy_records_for_record(chunk, record_index=0)
    records[0] = PolicyRowRecordV0(
        record_index=records[0].record_index,
        policy_row=records[0].policy_row,
        env_row=records[0].env_row,
        player=records[0].player,
        action=2,
        action_mask=records[0].action_mask,
        policy_target=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        root_value=records[0].root_value,
        policy_source=records[0].policy_source,
    )
    with pytest.raises(ReplayCompatibilityError, match="joint_action"):
        build_source_state_multiplayer_target_rows_v0(chunk, records)


def test_target_rows_reject_policy_record_when_next_record_is_leave_event():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=20260513,
        decision_source_frames=1,
        natural_bonus_spawn=False,
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = surface.reset(
        seed=20260513,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )
    leave_step = surface.remove_player(1)
    recorder.record(reset_step, source_ref="reset")
    recorder.record(leave_step, source_ref="remove_player")
    chunk = recorder.build_chunk()

    assert chunk.records[1]["trainer_surface_api"] == "remove_player"
    np.testing.assert_array_equal(
        chunk.arrays["joint_action"][1],
        np.full((1, 2), -1, dtype=np.int16),
    )

    action_mask = chunk.policy_rows[0]["policy_action_mask"][0].copy()
    policy_target = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    records = [
        PolicyRowRecordV0(
            record_index=0,
            policy_row=0,
            env_row=0,
            player=0,
            action=1,
            action_mask=action_mask,
            policy_target=policy_target,
            root_value=0.0,
            policy_source="unit_test_should_not_train_leave_event",
        )
    ]

    with pytest.raises(ReplayCompatibilityError, match="joint_action"):
        build_source_state_multiplayer_target_rows_v0(chunk, records)


def test_target_rows_copy_arrays_instead_of_aliasing_replay_arrays():
    chunk = _nonterminal_chunk(player_count=2)
    rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        _policy_records_for_record(chunk, record_index=0),
    )

    rows.observation[0, ...] = -3.0
    rows.next_observation[0, ...] = -4.0
    rows.action_mask[0, ...] = False
    rows.policy_target[0, ...] = 0.0

    assert float(chunk.arrays["observation"][0, 0, 0].min()) >= 0.0
    assert float(chunk.arrays["observation"][1, 0, 0].min()) >= 0.0
    assert bool(chunk.arrays["legal_action_mask"][0, 0, 0].all())


def test_target_rows_preserve_profile_no_death_restricted_metadata():
    chunk = _nonterminal_chunk(
        player_count=2,
        death_mode=vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
    )

    rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        _policy_records_for_record(chunk, record_index=0),
    )

    assert rows.metadata["death_mode"] == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
    assert rows.metadata["death_suppression_for_profile"] is True
    assert rows.metadata["death_suppression_claim"] == "profile_only_not_source_fidelity"
    assert rows.metadata["original_curvytron_behavior_claim"] is False
    assert rows.metadata["source_fidelity_claim"] == (
        PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM
    )
    assert rows.metadata["project_training_helper_active"] is True


def test_target_rows_preserve_death_immunity_restricted_metadata():
    chunk = _nonterminal_chunk(player_count=2, death_immunity_player_ids=(1,))

    rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        _policy_records_for_record(chunk, record_index=0),
    )

    assert rows.metadata["death_immunity_player_ids"] == [1]
    assert rows.metadata["death_immunity_mask"] == [[False, True]]
    assert rows.metadata["death_immunity_diagnostic"] is True
    assert rows.metadata["death_immunity_claim"] == "diagnostic_not_source_faithful"
    assert rows.metadata["original_curvytron_behavior_claim"] is False
    assert rows.metadata["project_training_helper_active"] is True


def test_sample_batch_is_deterministic_for_same_seed_and_tracks_row_ids():
    chunk = _nonterminal_chunk(player_count=4)
    rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        _policy_records_for_record(chunk, record_index=0),
    )
    batch = build_source_state_multiplayer_sample_batch_v0(
        rows,
        batch_size=4,
        seed=11,
    )
    repeat = build_source_state_multiplayer_sample_batch_v0(
        rows,
        batch_size=4,
        seed=11,
    )
    expected_row_id = np.random.default_rng(11).choice(4, size=4, replace=False)

    np.testing.assert_array_equal(batch.row_id, expected_row_id)
    np.testing.assert_array_equal(repeat.row_id, batch.row_id)
    np.testing.assert_array_equal(batch.observation, rows.observation[batch.row_id])
    np.testing.assert_array_equal(batch.action, rows.action[batch.row_id])
    assert batch.metadata["sample_contract_id"] == (
        SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_CONTRACT_ID
    )
    assert batch.metadata["source_target_contract_id"] == (
        SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID
    )
    assert batch.metadata["sample_row_count"] == 4
    assert batch.metadata["seed"] == 11
    assert batch.metadata["replace"] is False
    assert batch.metadata["native_game_segment_claim"] is False
    assert batch.metadata["lightzero_training_integration_claim"] is False


def test_sample_batch_different_seed_can_select_different_rows():
    chunk = _nonterminal_chunk(player_count=4)
    rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        _policy_records_for_record(chunk, record_index=0),
    )

    first = build_source_state_multiplayer_sample_batch_v0(
        rows,
        batch_size=2,
        seed=0,
    )
    second = build_source_state_multiplayer_sample_batch_v0(
        rows,
        batch_size=2,
        seed=1,
    )

    assert not np.array_equal(first.row_id, second.row_id)


def test_sample_batch_copies_arrays_instead_of_aliasing_target_rows():
    chunk = _nonterminal_chunk(player_count=4)
    rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        _policy_records_for_record(chunk, record_index=0),
    )
    batch = build_source_state_multiplayer_sample_batch_v0(
        rows,
        batch_size=2,
        seed=0,
    )
    first_row = int(batch.row_id[0])
    original_observation = rows.observation[first_row].copy()
    original_next_observation = rows.next_observation[first_row].copy()
    original_action_mask = rows.action_mask[first_row].copy()
    original_policy_target = rows.policy_target[first_row].copy()

    batch.observation[0, ...] = -30.0
    batch.next_observation[0, ...] = -40.0
    batch.action_mask[0, ...] = False
    batch.policy_target[0, ...] = 0.0

    np.testing.assert_array_equal(rows.observation[first_row], original_observation)
    np.testing.assert_array_equal(
        rows.next_observation[first_row],
        original_next_observation,
    )
    np.testing.assert_array_equal(rows.action_mask[first_row], original_action_mask)
    np.testing.assert_array_equal(rows.policy_target[first_row], original_policy_target)


def test_sample_batch_preserves_profile_no_death_metadata():
    chunk = _nonterminal_chunk(
        player_count=2,
        death_mode=vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
    )
    rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        _policy_records_for_record(chunk, record_index=0),
    )

    batch = build_source_state_multiplayer_sample_batch_v0(
        rows,
        batch_size=1,
        seed=3,
    )

    assert batch.metadata["death_mode"] == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH
    assert batch.metadata["death_suppression_for_profile"] is True
    assert batch.metadata["death_suppression_claim"] == "profile_only_not_source_fidelity"
    assert batch.metadata["original_curvytron_behavior_claim"] is False
    assert batch.metadata["source_fidelity_claim"] == (
        PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM
    )
    assert batch.metadata["project_training_helper_active"] is True


def test_sample_batch_rejects_invalid_batch_size():
    chunk = _nonterminal_chunk(player_count=2)
    rows = build_source_state_multiplayer_target_rows_v0(
        chunk,
        _policy_records_for_record(chunk, record_index=0),
    )

    with pytest.raises(ReplayCompatibilityError, match="positive"):
        build_source_state_multiplayer_sample_batch_v0(rows, batch_size=0)
    with pytest.raises(ReplayCompatibilityError, match="cannot exceed"):
        build_source_state_multiplayer_sample_batch_v0(rows, batch_size=3)


def _nonterminal_chunk(
    *,
    player_count: int,
    batch_size: int = 1,
    death_mode: str = vector_runtime.DEATH_MODE_NORMAL,
    death_immunity_player_ids: tuple[int, ...] = (),
):
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=batch_size,
        player_count=player_count,
        seed=100 + player_count,
        death_mode=death_mode,
        death_immunity_player_ids=death_immunity_player_ids,
        decision_source_frames=1,
        natural_bonus_spawn=False,
    )
    reset_step = surface.reset(seed=100 + player_count)
    step = surface.step(np.ones((batch_size, player_count), dtype=np.int16))
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    recorder.record(reset_step)
    recorder.record(step)
    return recorder.build_chunk()


def _terminal_chunk():
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=2,
        seed=211,
        decision_ms=100.0,
        natural_bonus_spawn=False,
    )
    reset_step = surface.reset(seed=211)
    terminal_step = _terminal_step(surface)
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    recorder.record(reset_step)
    recorder.record(terminal_step)
    return recorder.build_chunk()


def _three_record_chunk(*, player_count: int):
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=player_count,
        seed=311 + player_count,
        decision_source_frames=1,
        natural_bonus_spawn=False,
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    recorder.record(surface.reset(seed=311 + player_count))
    recorder.record(surface.step(np.ones((1, player_count), dtype=np.int16)))
    recorder.record(surface.step(np.ones((1, player_count), dtype=np.int16)))
    return recorder.build_chunk()


def _chunk_with_live_policy_seats(
    live_seats: list[tuple[int, int]],
) -> SourceStateMultiplayerTrainerReplayChunkV0:
    chunk = _nonterminal_chunk(batch_size=2, player_count=2)
    return _replace_record_policy_seats(chunk, record_index=0, live_seats=live_seats)


def _chunk_with_latest_channel_sentinels(
    live_seats: list[tuple[int, int]],
) -> SourceStateMultiplayerTrainerReplayChunkV0:
    sentinels = {
        (0, 1): 0.21,
        (1, 0): 0.31,
        (1, 1): 0.41,
    }
    chunk = _nonterminal_chunk(batch_size=2, player_count=2)
    arrays = {key: value.copy() for key, value in chunk.arrays.items()}
    for (env_row, player), value in sentinels.items():
        arrays["observation"][0, env_row, player, 3, :, :] = np.float32(value)
    updated = SourceStateMultiplayerTrainerReplayChunkV0(
        metadata=dict(chunk.metadata),
        arrays=arrays,
        policy_rows=chunk.policy_rows,
        records=chunk.records,
    )
    return _replace_record_policy_seats(updated, record_index=0, live_seats=live_seats)


def _replace_record_policy_seats(
    chunk,
    *,
    record_index: int,
    live_seats: list[tuple[int, int]],
) -> SourceStateMultiplayerTrainerReplayChunkV0:
    arrays = {key: value.copy() for key, value in chunk.arrays.items()}
    batch_size, player_count = arrays["live_mask"].shape[1:3]
    live_mask = np.zeros((batch_size, player_count), dtype=np.bool_)
    for env_row, player in live_seats:
        live_mask[int(env_row), int(player)] = True
    arrays["live_mask"][record_index] = live_mask

    policy_env_row = np.asarray([env_row for env_row, _player in live_seats], dtype=np.int32)
    policy_player = np.asarray([player for _env_row, player in live_seats], dtype=np.int16)
    policy_rows = list(chunk.policy_rows)
    policy_rows[record_index] = {
        "policy_observation": arrays["observation"][record_index, policy_env_row, policy_player]
        .astype(np.float32, copy=True),
        "policy_action_mask": arrays["legal_action_mask"][
            record_index,
            policy_env_row,
            policy_player,
        ].astype(bool, copy=True),
        "policy_env_row": policy_env_row,
        "policy_player": policy_player,
    }
    return SourceStateMultiplayerTrainerReplayChunkV0(
        metadata=dict(chunk.metadata),
        arrays=arrays,
        policy_rows=tuple(policy_rows),
        records=chunk.records,
    )


def _policy_records_for_record(chunk, *, record_index: int) -> list[PolicyRowRecordV0]:
    policy = chunk.policy_rows[record_index]
    records: list[PolicyRowRecordV0] = []
    for policy_row, (env_row, player) in enumerate(
        zip(policy["policy_env_row"], policy["policy_player"], strict=True)
    ):
        env_row = int(env_row)
        player = int(player)
        action = int(chunk.arrays["joint_action"][record_index + 1, env_row, player])
        action_mask = policy["policy_action_mask"][policy_row].copy()
        policy_target = np.zeros(3, dtype=np.float32)
        policy_target[action] = 1.0
        records.append(
            PolicyRowRecordV0(
                record_index=record_index,
                policy_row=policy_row,
                env_row=env_row,
                player=player,
                action=action,
                action_mask=action_mask,
                policy_target=policy_target,
                root_value=float(policy_row) / 10.0,
                policy_source="unit_test_one_hot",
                source_record_ref=f"{record_index}:{policy_row}",
            )
        )
    return records


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


def _compact_batch_from_chunk_policy_record(
    chunk,
    *,
    record_index: int,
) -> HybridCompactBatch:
    observation = np.asarray(chunk.arrays["observation"][record_index])
    batch_size, player_count = observation.shape[:2]
    action_mask = np.zeros((batch_size, player_count, 3), dtype=bool)
    policy = chunk.policy_rows[record_index]
    for policy_row, (env_row, player) in enumerate(
        zip(policy["policy_env_row"], policy["policy_player"], strict=True)
    ):
        action_mask[int(env_row), int(player)] = policy["policy_action_mask"][policy_row]
    root_count = int(batch_size * player_count)
    done = np.asarray(chunk.arrays["done"][record_index], dtype=np.bool_)
    done_root = np.repeat(done, player_count)
    flat_mask = action_mask.reshape(root_count, 3)
    terminal_row_mask = done.copy()
    terminal_global_rows = np.flatnonzero(terminal_row_mask).astype(np.int32, copy=False)
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
        autoreset_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        terminal_global_rows=terminal_global_rows,
        autoreset_global_rows=np.zeros((0,), dtype=np.int32),
        episode_step=np.arange(batch_size, dtype=np.int32),
        elapsed_ms=np.zeros((batch_size,), dtype=np.float64),
        round_id=np.zeros((batch_size,), dtype=np.int32),
        alive=np.asarray(chunk.arrays["live_mask"][record_index], dtype=np.bool_),
        joint_action=np.asarray(chunk.arrays["joint_action"][record_index], dtype=np.int16),
    )


def _replace_compact_batch_for_test(
    batch: HybridCompactBatch,
    **replacements,
) -> HybridCompactBatch:
    fields = {
        "observation": batch.observation,
        "action_mask": batch.action_mask,
        "reward": batch.reward,
        "final_reward_map": batch.final_reward_map,
        "done": batch.done,
        "policy_env_id": batch.policy_env_id,
        "policy_env_row": batch.policy_env_row,
        "policy_player": batch.policy_player,
        "target_reward": batch.target_reward,
        "done_root": batch.done_root,
        "to_play": batch.to_play,
        "active_root_mask": batch.active_root_mask,
        "final_observation": batch.final_observation,
        "final_observation_row_mask": batch.final_observation_row_mask,
        "terminal_row_mask": batch.terminal_row_mask,
        "autoreset_row_mask": batch.autoreset_row_mask,
        "terminal_global_rows": batch.terminal_global_rows,
        "autoreset_global_rows": batch.autoreset_global_rows,
        "episode_step": batch.episode_step,
        "elapsed_ms": batch.elapsed_ms,
        "round_id": batch.round_id,
        "alive": batch.alive,
        "joint_action": batch.joint_action,
    }
    fields.update(replacements)
    return HybridCompactBatch(**fields)


def _terminal_step(surface: SourceStateMultiplayerTrainerSurface):
    env = surface.env
    env.state["pos"][0] = np.asarray(
        [[5.0, 5.0], [87.0, 44.0]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray([math.pi / 4.0, 0.0], dtype=np.float64)
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0, 0] = 8.0
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]
    surface.stack.reset_rows(env, np.asarray([True], dtype=bool))

    step = surface.step(np.asarray([[1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(step.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        step.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    return step
