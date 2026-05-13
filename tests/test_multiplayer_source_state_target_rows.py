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
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


def test_target_rows_build_reset_to_step_alignment_without_lightzero_claims():
    chunk = _nonterminal_chunk(player_count=2)
    records = _policy_records_for_record(chunk, record_index=0)

    rows = build_source_state_multiplayer_target_rows_v0(chunk, records)

    assert rows.metadata["target_contract_id"] == (
        SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID
    )
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
    death_mode: str = vector_runtime.DEATH_MODE_NORMAL,
    death_immunity_player_ids: tuple[int, ...] = (),
):
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=player_count,
        seed=100 + player_count,
        death_mode=death_mode,
        death_immunity_player_ids=death_immunity_player_ids,
        decision_source_frames=1,
        natural_bonus_spawn=False,
    )
    reset_step = surface.reset(seed=100 + player_count)
    step = surface.step(np.ones((1, player_count), dtype=np.int16))
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
