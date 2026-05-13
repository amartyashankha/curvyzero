import math

import numpy as np
import pytest

from curvyzero.env import vector_reset
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


@pytest.mark.parametrize("player_count", [3, 4])
def test_public_match_lifecycle_mixes_next_round_and_match_end_rows(player_count: int):
    env = VectorMultiplayerEnv(
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
    )

    reset_batch = env.reset(
        seed=np.asarray([101, 202], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )

    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    assert reset_batch.info["warmup_info"]["game_start_fires"] == 2
    np.testing.assert_array_equal(env.state["started"], np.ones(2, dtype=bool))
    np.testing.assert_array_equal(env.state["in_round"], np.ones(2, dtype=bool))
    np.testing.assert_array_equal(
        env.state["printing"][:, :player_count],
        np.ones((2, player_count), dtype=bool),
    )
    np.testing.assert_array_equal(
        env.state["print_manager_active"][:, :player_count],
        np.ones((2, player_count), dtype=bool),
    )
    np.testing.assert_array_equal(reset_batch.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(reset_batch.info["round_done"], np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(
        reset_batch.info["warmdown_pending"],
        np.zeros(2, dtype=bool),
    )
    np.testing.assert_array_equal(reset_batch.info["match_done"], np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(
        reset_batch.action_mask,
        np.ones((2, player_count, 3), dtype=bool),
    )
    _assert_public_final_rows(reset_batch, [])

    env.state["max_score"][1] = player_count - 1
    _force_player0_round_win(env, row=0, player_count=player_count)
    _force_player0_round_win(env, row=1, player_count=player_count)

    actions = np.ones((2, player_count), dtype=np.int16)
    round_batch = env.step(actions)

    np.testing.assert_array_equal(round_batch.done, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(round_batch.terminated, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(round_batch.truncated, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(round_batch.info["round_done"], np.ones(2, dtype=bool))
    np.testing.assert_array_equal(
        round_batch.info["warmdown_pending"],
        np.ones(2, dtype=bool),
    )
    np.testing.assert_array_equal(round_batch.info["match_done"], np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(round_batch.info["needs_reset"], np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(
        round_batch.info["terminal_reason"],
        np.full(2, vector_reset.TERMINAL_REASON_SURVIVOR_WIN, dtype=np.int16),
    )
    np.testing.assert_array_equal(round_batch.info["winner"], np.zeros(2, dtype=np.int16))
    np.testing.assert_array_equal(
        round_batch.info["round_winner"],
        np.zeros(2, dtype=np.int16),
    )
    np.testing.assert_array_equal(
        round_batch.info["match_winner"],
        np.full(2, -1, dtype=np.int16),
    )
    assert round_batch.info["round_winner_ids"] == [[0], [0]]
    assert round_batch.info["match_winner_ids"] == [[], []]
    expected_reward = np.full((2, player_count), -1.0, dtype=np.float32)
    expected_reward[:, 0] = 1.0
    np.testing.assert_array_equal(round_batch.reward, expected_reward)
    np.testing.assert_array_equal(
        round_batch.action_mask,
        np.zeros((2, player_count, 3), dtype=bool),
    )
    _assert_public_final_rows(round_batch, [])

    with pytest.raises(RuntimeError, match="advance_warmdown must be called"):
        env.step(actions)

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 1
    np.testing.assert_array_equal(
        warmdown_batch.info["next_round_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["terminal_rows"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False, True]))
    np.testing.assert_array_equal(warmdown_batch.terminated, np.asarray([False, True]))
    np.testing.assert_array_equal(warmdown_batch.truncated, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["warmdown_pending"],
        np.zeros(2, dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False, True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_winner"],
        np.asarray([-1, 0], dtype=np.int16),
    )
    assert warmdown_batch.info["round_winner_ids"] == [[], [0]]
    assert warmdown_batch.info["match_winner_ids"] == [[], [0]]
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"][0],
        np.ones(player_count, dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"][0],
        np.full(player_count, -1, dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_id"],
        np.asarray([2, 1]),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.asarray(
            [
                np.ones((player_count, 3), dtype=bool),
                np.zeros((player_count, 3), dtype=bool),
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        warmdown_batch.reward,
        np.zeros((2, player_count), dtype=np.float32),
    )
    _assert_public_final_rows(
        warmdown_batch,
        [1],
        expected_reward=np.zeros((2, player_count), dtype=np.float32),
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


def _assert_public_final_rows(
    batch,
    rows: list[int],
    *,
    expected_reward: np.ndarray | None = None,
) -> None:
    rows_array = np.asarray(rows, dtype=np.int32)
    row_mask = np.zeros(batch.done.shape, dtype=bool)
    row_mask[rows_array] = True

    np.testing.assert_array_equal(batch.info["final_observation_rows"], rows_array)
    np.testing.assert_array_equal(batch.info["final_observation_row_mask"], row_mask)
    np.testing.assert_array_equal(batch.info["final_reward_rows"], rows_array)
    np.testing.assert_array_equal(batch.info["final_reward_row_mask"], row_mask)

    if not rows:
        assert batch.final_observation is None
        assert batch.final_reward is None
        return

    assert batch.final_observation is not None
    expected_final_observation = np.zeros_like(batch.observation)
    expected_final_observation[rows_array] = batch.observation[rows_array]
    np.testing.assert_array_equal(batch.final_observation, expected_final_observation)
    np.testing.assert_array_equal(
        batch.info["final_observation"],
        expected_final_observation,
    )

    assert expected_reward is not None
    assert batch.final_reward is not None
    expected_final_reward = np.zeros_like(batch.reward)
    expected_final_reward[rows_array] = expected_reward[rows_array]
    np.testing.assert_array_equal(batch.final_reward, expected_final_reward)
    np.testing.assert_array_equal(batch.info["final_reward_map"], expected_final_reward)
