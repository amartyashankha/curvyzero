import math

import numpy as np
import pytest

from curvyzero.env import vector_reset
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)


@pytest.mark.parametrize("player_count", [3, 4])
def test_trainer_replay_preserves_multiplayer_warmdown_lifecycle(player_count: int):
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
        random_tape_capacity=256,
        natural_bonus_spawn=False,
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()

    reset_step = surface.reset(
        seed=np.asarray([101, 202], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )
    recorder.record(reset_step, source_ref="reset")

    np.testing.assert_array_equal(reset_step.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(
        reset_step.live_mask,
        np.ones((2, player_count), dtype=bool),
    )
    _assert_policy_rows(reset_step, [(0, p) for p in range(player_count)] + [(1, p) for p in range(player_count)])

    surface.env.state["max_score"][1] = player_count - 1
    _force_player0_round_win(surface, row=0, player_count=player_count)
    _force_player0_round_win(surface, row=1, player_count=player_count)

    actions = np.ones((2, player_count), dtype=np.int16)
    round_step = surface.step(actions)
    recorder.record(round_step, source_ref="round-end")

    np.testing.assert_array_equal(round_step.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(round_step.info["round_done"], np.ones(2, dtype=bool))
    np.testing.assert_array_equal(
        round_step.info["warmdown_pending"],
        np.ones(2, dtype=bool),
    )
    np.testing.assert_array_equal(round_step.info["match_done"], np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(
        round_step.legal_action_mask,
        np.zeros((2, player_count, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        round_step.live_mask,
        np.zeros((2, player_count), dtype=bool),
    )
    _assert_policy_rows(round_step, [])
    expected_round_reward = np.zeros((2, player_count), dtype=np.float32)
    expected_round_reward[:, 0] = 1.0
    np.testing.assert_array_equal(round_step.reward, expected_round_reward)
    np.testing.assert_array_equal(
        round_step.final_observation_row_mask,
        np.zeros(2, dtype=bool),
    )

    with pytest.raises(RuntimeError, match="advance_warmdown must be called"):
        surface.step(actions)

    warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(warmdown_step, source_ref="warmdown")
    chunk = recorder.build_chunk()

    assert warmdown_step.info["trainer_surface_api"] == "advance_warmdown"
    assert warmdown_step.info["warmdown_waited"] is True
    assert warmdown_step.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_step.info["warmdown_info"]["match_end_count"] == 1
    np.testing.assert_array_equal(
        warmdown_step.info["next_round_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_step.info["terminal_rows"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(warmdown_step.done, np.asarray([False, True]))
    np.testing.assert_array_equal(warmdown_step.terminated, np.asarray([False, True]))
    np.testing.assert_array_equal(warmdown_step.truncated, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(
        warmdown_step.info["terminal_reason"],
        np.asarray(
            [vector_reset.TERMINAL_REASON_NONE, vector_reset.TERMINAL_REASON_SURVIVOR_WIN],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        warmdown_step.legal_action_mask,
        np.asarray(
            [
                np.ones((player_count, 3), dtype=bool),
                np.zeros((player_count, 3), dtype=bool),
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        warmdown_step.live_mask,
        np.asarray(
            [
                np.ones(player_count, dtype=bool),
                np.zeros(player_count, dtype=bool),
            ],
            dtype=bool,
        ),
    )
    _assert_policy_rows(warmdown_step, [(0, p) for p in range(player_count)])
    np.testing.assert_array_equal(
        warmdown_step.reward,
        np.zeros((2, player_count), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        warmdown_step.final_reward_map,
        np.zeros((2, player_count), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        warmdown_step.final_observation_row_mask,
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_step.final_observation[0],
        np.zeros((player_count, 4, 64, 64), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        warmdown_step.final_observation[1],
        warmdown_step.observation[1],
    )
    assert int(np.count_nonzero(warmdown_step.final_observation[1])) > 0
    np.testing.assert_array_equal(
        warmdown_step.observation[0, :, :-1],
        np.zeros((player_count, 3, 64, 64), dtype=np.float32),
    )
    assert int(np.count_nonzero(warmdown_step.observation[0, :, -1])) > 0

    assert chunk.metadata["record_count"] == 3
    assert chunk.metadata["closed_by_terminal"] is True
    assert chunk.metadata["player_count"] == player_count
    np.testing.assert_array_equal(
        chunk.arrays["done"],
        np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"],
        np.asarray(
            [
                [False, False],
                [False, False],
                [False, True],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(chunk.arrays["live_mask"][1], round_step.live_mask)
    np.testing.assert_array_equal(
        chunk.arrays["live_mask"][2],
        warmdown_step.live_mask,
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_observation"][2],
        warmdown_step.final_observation,
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_reward_map"][2],
        warmdown_step.final_reward_map,
    )
    assert [rows["policy_env_row"].size for rows in chunk.policy_rows] == [
        2 * player_count,
        0,
        player_count,
    ]
    assert chunk.records[1]["policy_row_count"] == 0
    assert chunk.records[2]["trainer_surface_api"] == "advance_warmdown"
    assert chunk.records[2]["done_rows"] == [1]
    assert chunk.records[2]["final_observation_rows"] == [1]


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


def _force_player0_round_win(
    surface: SourceStateMultiplayerTrainerSurface,
    *,
    row: int,
    player_count: int,
) -> None:
    env = surface.env
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
