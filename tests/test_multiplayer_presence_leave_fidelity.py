import math

import numpy as np
import pytest

from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


@pytest.mark.parametrize("player_count", [3, 4])
def test_public_presence_leave_active_and_warmdown_rows(player_count: int):
    env = _make_public_env(player_count)
    reset_batch = env.reset(
        seed=np.asarray([101, 202], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )

    np.testing.assert_array_equal(
        reset_batch.info["present"],
        np.ones((2, player_count), dtype=bool),
    )
    np.testing.assert_array_equal(
        reset_batch.action_mask,
        np.ones((2, player_count, 3), dtype=bool),
    )

    _force_player0_round_win(env, row=1, player_count=player_count)
    actions = np.ones((2, player_count), dtype=np.int16)
    round_batch = env.step(actions)

    np.testing.assert_array_equal(round_batch.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(
        round_batch.info["round_done"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.info["warmdown_pending"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        round_batch.action_mask,
        np.asarray(
            [
                np.ones((player_count, 3), dtype=bool),
                np.zeros((player_count, 3), dtype=bool),
            ],
            dtype=bool,
        ),
    )

    leave_batch = env.remove_player(
        np.asarray([1, 0], dtype=np.int16),
        row_mask=np.asarray([True, True], dtype=bool),
    )

    expected_present = np.ones((2, player_count), dtype=bool)
    expected_present[0, 1] = False
    expected_present[1, 0] = False
    expected_alive_after_leave = expected_present.copy()
    expected_alive_after_leave[1] = False
    expected_leave_mask = np.zeros((2, player_count, 3), dtype=bool)
    expected_leave_mask[0, expected_present[0]] = True

    assert leave_batch.info["leave_metadata_only"] is True
    assert leave_batch.info["leave_trainer_claim"] is False
    np.testing.assert_array_equal(
        leave_batch.info["leave_rows"],
        np.asarray([0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_player_ids"],
        np.asarray([1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_source_player_ids"],
        np.asarray([2, 1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_warmdown_rows"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        leave_batch.info["leave_immediate_terminal_rows"],
        np.asarray([], dtype=np.int32),
    )
    np.testing.assert_array_equal(leave_batch.info["present"], expected_present)
    np.testing.assert_array_equal(leave_batch.info["alive"], expected_alive_after_leave)
    np.testing.assert_array_equal(
        leave_batch.info["round_done"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        leave_batch.info["warmdown_pending"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(leave_batch.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(leave_batch.action_mask, expected_leave_mask)
    np.testing.assert_array_equal(
        leave_batch.reward,
        np.zeros((2, player_count), dtype=np.float32),
    )
    _assert_public_final_rows(leave_batch, [])

    warmdown_batch = env.advance_warmdown(5000.0)
    expected_alive_after_warmdown = expected_present.copy()
    expected_live_mask = np.repeat(expected_present[:, :, None], 3, axis=2)

    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(
        warmdown_batch.info["next_round_rows"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(warmdown_batch.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(warmdown_batch.info["present"], expected_present)
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        expected_alive_after_warmdown,
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.zeros(2, dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_pending"],
        np.zeros(2, dtype=bool),
    )
    np.testing.assert_array_equal(warmdown_batch.action_mask, expected_live_mask)
    _assert_public_final_rows(warmdown_batch, [])

    next_actions = np.ones((2, player_count), dtype=np.int16)
    next_actions[0, 1] = -1
    next_actions[1, 0] = -1
    next_batch = env.step(next_actions)

    expected_action_source = np.full(
        (2, player_count),
        "external_joint_action",
        dtype=object,
    )
    expected_action_source[0, 1] = "absent_noop"
    expected_action_source[1, 0] = "absent_noop"
    expected_native = np.zeros((2, player_count), dtype=np.int8)

    np.testing.assert_array_equal(next_batch.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(next_batch.info["present"], expected_present)
    np.testing.assert_array_equal(next_batch.info["alive"], expected_alive_after_warmdown)
    np.testing.assert_array_equal(next_batch.action_mask, expected_live_mask)
    np.testing.assert_array_equal(
        next_batch.info["action_sidecar"]["action_required"],
        expected_present,
    )
    np.testing.assert_array_equal(
        next_batch.info["action_sidecar"]["action_source"],
        expected_action_source,
    )
    np.testing.assert_array_equal(
        next_batch.info["action_sidecar"]["native_control_value"],
        expected_native,
    )


@pytest.mark.parametrize("player_count", [3, 4])
def test_trainer_replay_preserves_presence_leave_rows(player_count: int):
    surface = SourceStateMultiplayerTrainerSurface(
        batch_size=2,
        player_count=player_count,
        seed=20260513,
        decision_ms=100.0,
        max_score=99,
        episode_end_mode="match",
        body_capacity=64,
        event_capacity=64,
        timer_capacity=max(4, player_count),
        random_tape_capacity=512,
        natural_bonus_spawn=False,
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()

    reset_step = surface.reset(
        seed=np.asarray([101, 202], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )
    recorder.record(reset_step, source_ref="reset")

    _force_player0_round_win(surface.env, row=1, player_count=player_count)
    actions = np.ones((2, player_count), dtype=np.int16)
    round_step = surface.step(actions)
    recorder.record(round_step, source_ref="round-end")

    leave_step = surface.remove_player(
        np.asarray([1, 0], dtype=np.int16),
        row_mask=np.asarray([True, True], dtype=bool),
    )
    recorder.record(leave_step, source_ref="mixed-active-warmdown-leave")

    expected_present = np.ones((2, player_count), dtype=bool)
    expected_present[0, 1] = False
    expected_present[1, 0] = False
    expected_live_after_leave = np.zeros((2, player_count), dtype=bool)
    expected_live_after_leave[0] = expected_present[0]
    expected_leave_mask = np.zeros((2, player_count, 3), dtype=bool)
    expected_leave_mask[0, expected_present[0]] = True
    expected_leave_reward = expected_live_after_leave.astype(np.float32)

    assert leave_step.info["trainer_surface_api"] == "remove_player"
    assert leave_step.info["underlying_env_metadata_only"] is True
    np.testing.assert_array_equal(leave_step.info["present"], expected_present)
    np.testing.assert_array_equal(leave_step.live_mask, expected_live_after_leave)
    np.testing.assert_array_equal(leave_step.legal_action_mask, expected_leave_mask)
    np.testing.assert_array_equal(leave_step.reward, expected_leave_reward)
    np.testing.assert_array_equal(
        leave_step.joint_action,
        np.full((2, player_count), -1, dtype=np.int16),
    )
    _assert_policy_rows(
        leave_step,
        [(0, player) for player in range(player_count) if player != 1],
    )
    np.testing.assert_array_equal(
        leave_step.final_observation_row_mask,
        np.zeros(2, dtype=bool),
    )

    warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(warmdown_step, source_ref="warmdown")
    chunk = recorder.build_chunk()

    expected_live_after_warmdown = expected_present.copy()
    expected_warmdown_reward = np.zeros((2, player_count), dtype=np.float32)
    expected_warmdown_mask = np.repeat(expected_present[:, :, None], 3, axis=2)
    expected_warmdown_policy_rows = (
        [(0, player) for player in range(player_count) if player != 1]
        + [(1, player) for player in range(player_count) if player != 0]
    )

    assert warmdown_step.info["trainer_surface_api"] == "advance_warmdown"
    np.testing.assert_array_equal(warmdown_step.info["present"], expected_present)
    np.testing.assert_array_equal(warmdown_step.live_mask, expected_live_after_warmdown)
    np.testing.assert_array_equal(warmdown_step.legal_action_mask, expected_warmdown_mask)
    np.testing.assert_array_equal(warmdown_step.reward, expected_warmdown_reward)
    _assert_policy_rows(warmdown_step, expected_warmdown_policy_rows)
    np.testing.assert_array_equal(
        warmdown_step.final_observation_row_mask,
        np.zeros(2, dtype=bool),
    )

    assert chunk.metadata["record_count"] == 4
    assert chunk.metadata["closed_by_terminal"] is False
    assert chunk.metadata["player_count"] == player_count
    np.testing.assert_array_equal(
        chunk.arrays["live_mask"][2],
        expected_live_after_leave,
    )
    np.testing.assert_array_equal(
        chunk.arrays["legal_action_mask"][2],
        expected_leave_mask,
    )
    np.testing.assert_array_equal(
        chunk.arrays["joint_action"][2],
        np.full((2, player_count), -1, dtype=np.int16),
    )
    np.testing.assert_array_equal(
        chunk.arrays["live_mask"][3],
        expected_live_after_warmdown,
    )
    assert chunk.records[2]["trainer_surface_api"] == "remove_player"
    assert chunk.records[2]["policy_row_count"] == player_count - 1
    assert chunk.records[3]["trainer_surface_api"] == "advance_warmdown"
    assert chunk.records[3]["policy_row_count"] == 2 * (player_count - 1)


def _make_public_env(player_count: int) -> VectorMultiplayerEnv:
    return VectorMultiplayerEnv(
        batch_size=2,
        player_count=player_count,
        seed=20260513,
        decision_ms=100.0,
        max_score=99,
        episode_end_mode="match",
        body_capacity=64,
        event_capacity=64,
        timer_capacity=max(4, player_count),
        random_tape_capacity=512,
    )


def _force_player0_round_win(
    env: VectorMultiplayerEnv,
    *,
    row: int,
    player_count: int,
) -> None:
    if player_count == 3:
        env.state["pos"][row, :3] = np.asarray(
            [[5.0, 5.0], [1.0, 47.5], [93.0, 47.5]],
            dtype=np.float64,
        )
        env.state["heading"][row, :3] = np.asarray(
            [math.pi / 4.0, math.pi, 0.0],
            dtype=np.float64,
        )
        env.state["speed"][row, :3] = np.asarray([8.0, 16.0, 16.0], dtype=np.float64)
    else:
        env.state["pos"][row, :4] = np.asarray(
            [[10.0, 50.5], [50.5, 1.0], [1.0, 50.5], [99.0, 50.5]],
            dtype=np.float64,
        )
        env.state["heading"][row, :4] = np.asarray(
            [0.0, math.tau * 0.75, math.pi, 0.0],
            dtype=np.float64,
        )
        env.state["speed"][row, :4] = np.asarray(
            [8.0, 16.0, 16.0, 16.0],
            dtype=np.float64,
        )

    env.state["prev_pos"][row, :player_count] = env.state["pos"][row, :player_count]
    env.state["print_manager_distance"][row, 0] = 999.0
    env.state["print_manager_last_pos"][row, 0] = env.state["pos"][row, 0]


def _assert_public_final_rows(batch, rows: list[int]) -> None:
    rows_array = np.asarray(rows, dtype=np.int32)
    row_mask = np.zeros(batch.done.shape, dtype=bool)
    row_mask[rows_array] = True

    np.testing.assert_array_equal(batch.info["final_observation_rows"], rows_array)
    np.testing.assert_array_equal(batch.info["final_observation_row_mask"], row_mask)
    np.testing.assert_array_equal(batch.info["final_reward_rows"], rows_array)
    np.testing.assert_array_equal(batch.info["final_reward_row_mask"], row_mask)
    if rows:
        assert batch.final_observation is not None
        assert batch.final_reward is not None
    else:
        assert batch.final_observation is None
        assert batch.final_reward is None


def _assert_policy_rows(step, expected: list[tuple[int, int]]) -> None:
    expected_env_row = np.asarray([row for row, _ in expected], dtype=np.int32)
    expected_player = np.asarray([player for _, player in expected], dtype=np.int16)
    np.testing.assert_array_equal(step.policy_env_row, expected_env_row)
    np.testing.assert_array_equal(step.policy_player, expected_player)
    assert step.policy_observation.shape == (len(expected), 4, 64, 64)
    assert step.policy_action_mask.shape == (len(expected), 3)
    for policy_row, (row, player) in enumerate(expected):
        np.testing.assert_array_equal(
            step.policy_observation[policy_row],
            step.observation[row, player],
        )
        np.testing.assert_array_equal(
            step.policy_action_mask[policy_row],
            step.legal_action_mask[row, player],
        )
