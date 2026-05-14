import json
import math
from pathlib import Path

import numpy as np
import pytest

from curvyzero.env import vector_reset
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)


SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenarios" / "environment"
PRESENT_ABSENT_3P = np.asarray([[True, False, True]], dtype=bool)
PRESENT_ABSENT_3P_ACTION_MASK = np.asarray(
    [[[True, True, True], [False, False, False], [True, True, True]]],
    dtype=bool,
)
PRESENT_ABSENT_4P = np.asarray([[True, False, True, True]], dtype=bool)
PRESENT_ABSENT_4P_ACTION_MASK = np.asarray(
    [
        [
            [True, True, True],
            [False, False, False],
            [True, True, True],
            [True, True, True],
        ]
    ],
    dtype=bool,
)


def test_round_new_present_absent_3p_fixture_public_and_replay_masks():
    scenario_name = "source_lifecycle_present_absent_3p_round_new.json"
    env, batch = _reset_public(
        scenario_name,
        player_count=3,
        present=PRESENT_ABSENT_3P,
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    np.testing.assert_array_equal(batch.info["present"], PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(batch.info["alive"], PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(batch.action_mask, PRESENT_ABSENT_3P_ACTION_MASK)
    np.testing.assert_allclose(
        env.state["pos"][0, [0, 2]],
        [[68.575, 47.5], [26.425, 47.5]],
    )
    np.testing.assert_allclose(
        env.state["heading"][0, [0, 2]],
        [math.tau * 0.75, math.tau * 0.25],
    )
    assert int(batch.info["random_tape_cursor"][0]) == 6
    _assert_public_final_rows(batch, [], expected_reward=None)

    surface = _make_surface(scenario_name, player_count=3)
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = surface.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=PRESENT_ABSENT_3P,
        source_fixture_random_tape_values=_lifecycle_random_tape(scenario_name),
        source_fixture_ref=f"scenarios/environment/{scenario_name}",
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    recorder.record(reset_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(reset_step.info["present"], PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(reset_step.live_mask, PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(
        reset_step.legal_action_mask,
        PRESENT_ABSENT_3P_ACTION_MASK,
    )
    _assert_policy_rows(reset_step, [(0, 0), (0, 2)])
    assert chunk.metadata["record_count"] == 1
    assert chunk.metadata["closed_by_terminal"] is False
    assert chunk.records[0]["source_ref"] == scenario_name
    assert chunk.records[0]["policy_row_count"] == 2


def test_round_new_present_absent_4p_fixture_public_and_replay_masks():
    scenario_name = "source_lifecycle_present_absent_4p_round_new.json"
    env, batch = _reset_public(
        scenario_name,
        player_count=4,
        present=PRESENT_ABSENT_4P,
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )

    np.testing.assert_array_equal(batch.info["present"], PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(batch.info["alive"], PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[1, -1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(batch.action_mask, PRESENT_ABSENT_4P_ACTION_MASK)
    np.testing.assert_allclose(
        env.state["pos"][0, [0, 2, 3]],
        [[77.41, 50.5], [41.53, 50.5], [23.59, 50.5]],
    )
    np.testing.assert_allclose(
        env.state["heading"][0, [0, 2, 3]],
        [math.tau * 0.75, 0.1, math.tau * 0.25],
    )
    assert int(batch.info["random_tape_cursor"][0]) == 9
    _assert_public_final_rows(batch, [], expected_reward=None)

    surface = _make_surface(scenario_name, player_count=4)
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = surface.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=PRESENT_ABSENT_4P,
        source_fixture_random_tape_values=_lifecycle_random_tape(scenario_name),
        source_fixture_ref=f"scenarios/environment/{scenario_name}",
        source_fixture_new_round_time_ms=3000.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    recorder.record(reset_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(reset_step.info["present"], PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(reset_step.live_mask, PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(
        reset_step.legal_action_mask,
        PRESENT_ABSENT_4P_ACTION_MASK,
    )
    _assert_policy_rows(reset_step, [(0, 0), (0, 2), (0, 3)])
    assert chunk.metadata["record_count"] == 1
    assert chunk.metadata["closed_by_terminal"] is False
    assert chunk.records[0]["source_ref"] == scenario_name
    assert chunk.records[0]["policy_row_count"] == 3


def test_survivor_present_absent_3p_fixture_public_and_replay_final_rows():
    scenario_name = "source_lifecycle_present_absent_3p_survivor_score_round_end.json"
    env, reset_batch = _reset_public(
        scenario_name,
        player_count=3,
        present=PRESENT_ABSENT_3P,
    )

    assert int(reset_batch.info["random_tape_cursor"][0]) == 9
    _force_present_absent_survivor_score(env)
    batch = env.step(np.asarray([[1, -1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(batch.info["present"], PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([[True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["score"],
        np.asarray([[2, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([[1, 2, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(batch.info["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(batch.info["match_done"], np.asarray([False]))
    np.testing.assert_array_equal(batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.reward,
        np.asarray([[1.0, 0.0, -1.0]], dtype=np.float32),
    )
    _assert_public_final_rows(batch, [0], expected_reward=batch.reward)

    surface = _make_surface(scenario_name, player_count=3)
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = _reset_surface(surface, scenario_name, present=PRESENT_ABSENT_3P)
    recorder.record(reset_step, source_ref=scenario_name)
    _assert_policy_rows(reset_step, [(0, 0), (0, 2)])

    _force_present_absent_survivor_score(surface.env)
    terminal_step = surface.step(np.asarray([[1, -1, 1]], dtype=np.int16))
    recorder.record(terminal_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(terminal_step.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_step.info["present"], PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(
        terminal_step.info["score"],
        np.asarray([[2, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(terminal_step.live_mask, np.zeros((1, 3), dtype=bool))
    np.testing.assert_array_equal(
        terminal_step.reward,
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_step.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    _assert_policy_rows(terminal_step, [])

    assert chunk.metadata["record_count"] == 2
    assert chunk.metadata["closed_by_terminal"] is True
    assert [rows["policy_env_row"].size for rows in chunk.policy_rows] == [2, 0]
    assert chunk.records[1]["source_ref"] == scenario_name
    assert chunk.records[1]["done_rows"] == [0]
    assert chunk.records[1]["final_observation_rows"] == [0]
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][1],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(chunk.arrays["final_reward_map"][1], terminal_step.reward)


def test_survivor_present_absent_4p_fixture_public_and_replay_final_rows():
    scenario_name = "source_lifecycle_present_absent_4p_survivor_score_round_end.json"
    env, reset_batch = _reset_public(
        scenario_name,
        player_count=4,
        present=PRESENT_ABSENT_4P,
    )

    assert int(reset_batch.info["random_tape_cursor"][0]) == 13
    _force_present_absent_4p_first_death(env)
    first_death = env.step(np.asarray([[1, -1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(first_death.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_death.info["alive"],
        np.asarray([[True, False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death.info["death_player"],
        np.asarray([[1, 3, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_death.action_mask,
        np.asarray(
            [
                [
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                    [False, False, False],
                ]
            ],
            dtype=bool,
        ),
    )
    _assert_public_final_rows(first_death, [], expected_reward=None)

    _force_present_absent_4p_terminal_death(env)
    terminal_batch = env.step(np.asarray([[1, -1, 1, -1]], dtype=np.int16))

    np.testing.assert_array_equal(terminal_batch.info["present"], PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(
        terminal_batch.info["alive"],
        np.asarray([[True, False, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[3, 0, 2, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[1, 3, 2, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["winner"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.info["match_done"], np.asarray([False]))
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.reward,
        np.asarray([[1.0, 0.0, -1.0, -1.0]], dtype=np.float32),
    )
    _assert_public_final_rows(
        terminal_batch,
        [0],
        expected_reward=terminal_batch.reward,
    )

    surface = _make_surface(scenario_name, player_count=4)
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = _reset_surface(surface, scenario_name, present=PRESENT_ABSENT_4P)
    recorder.record(reset_step, source_ref=scenario_name)
    _assert_policy_rows(reset_step, [(0, 0), (0, 2), (0, 3)])

    _force_present_absent_4p_first_death(surface.env)
    first_death_step = surface.step(np.asarray([[1, -1, 1, 1]], dtype=np.int16))
    recorder.record(first_death_step, source_ref=scenario_name)
    np.testing.assert_array_equal(
        first_death_step.live_mask,
        np.asarray([[True, False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death_step.reward,
        np.asarray([[1.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    )
    _assert_policy_rows(first_death_step, [(0, 0), (0, 2)])

    _force_present_absent_4p_terminal_death(surface.env)
    terminal_step = surface.step(np.asarray([[1, -1, 1, -1]], dtype=np.int16))
    recorder.record(terminal_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(terminal_step.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_step.info["present"], PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(
        terminal_step.info["score"],
        np.asarray([[3, 0, 2, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(terminal_step.live_mask, np.zeros((1, 4), dtype=bool))
    np.testing.assert_array_equal(
        terminal_step.reward,
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        terminal_step.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    _assert_policy_rows(terminal_step, [])

    assert chunk.metadata["record_count"] == 3
    assert chunk.metadata["closed_by_terminal"] is True
    assert [rows["policy_env_row"].size for rows in chunk.policy_rows] == [3, 2, 0]
    assert chunk.records[2]["source_ref"] == scenario_name
    assert chunk.records[2]["done_rows"] == [0]
    assert chunk.records[2]["final_observation_rows"] == [0]
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][2],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(chunk.arrays["final_reward_map"][2], terminal_step.reward)


def test_next_round_present_absent_3p_fixture_public_and_replay_masks():
    scenario_name = "source_lifecycle_present_absent_3p_next_round.json"
    env, _ = _reset_public(
        scenario_name,
        player_count=3,
        present=PRESENT_ABSENT_3P,
    )

    _force_present_absent_draw(env)
    terminal_batch = env.step(np.asarray([[1, -1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[1, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[1, 2, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.info["match_done"], np.asarray([False]))
    _assert_public_final_rows(
        terminal_batch,
        [0],
        expected_reward=terminal_batch.reward,
    )

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(warmdown_batch.info["present"], PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(warmdown_batch.info["alive"], PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[1, 0, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(warmdown_batch.action_mask, PRESENT_ABSENT_3P_ACTION_MASK)
    _assert_public_final_rows(warmdown_batch, [], expected_reward=None)

    surface = _make_surface(
        scenario_name,
        player_count=3,
        episode_end_mode="match",
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = _reset_surface(surface, scenario_name, present=PRESENT_ABSENT_3P)
    recorder.record(reset_step, source_ref=scenario_name)
    _assert_policy_rows(reset_step, [(0, 0), (0, 2)])

    _force_present_absent_draw(surface.env)
    round_step = surface.step(np.asarray([[1, -1, 1]], dtype=np.int16))
    recorder.record(round_step, source_ref=scenario_name)

    np.testing.assert_array_equal(round_step.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(round_step.info["round_done"], np.asarray([True]))
    np.testing.assert_array_equal(round_step.live_mask, np.zeros((1, 3), dtype=bool))
    np.testing.assert_array_equal(
        round_step.final_observation_row_mask,
        np.asarray([False], dtype=bool),
    )
    _assert_policy_rows(round_step, [])

    warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(warmdown_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    assert warmdown_step.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_step.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(warmdown_step.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(warmdown_step.info["present"], PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(warmdown_step.live_mask, PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(
        warmdown_step.legal_action_mask,
        PRESENT_ABSENT_3P_ACTION_MASK,
    )
    _assert_policy_rows(warmdown_step, [(0, 0), (0, 2)])

    assert chunk.metadata["record_count"] == 3
    assert chunk.metadata["closed_by_terminal"] is False
    assert [rows["policy_env_row"].size for rows in chunk.policy_rows] == [2, 0, 2]
    assert [record["source_ref"] for record in chunk.records] == [
        scenario_name,
        scenario_name,
        scenario_name,
    ]
    np.testing.assert_array_equal(chunk.arrays["live_mask"][2], PRESENT_ABSENT_3P)
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"],
        np.zeros((3, 1), dtype=bool),
    )


@pytest.mark.parametrize(
    "scenario_name",
    [
        "source_lifecycle_present_absent_4p_next_round.json",
        "source_lifecycle_present_absent_4p_tie_at_max_score.json",
    ],
)
def test_draw_present_absent_4p_fixtures_public_and_replay_next_round(
    scenario_name: str,
):
    env, _ = _reset_public(
        scenario_name,
        player_count=4,
        present=PRESENT_ABSENT_4P,
    )

    _force_present_absent_4p_draw(env)
    terminal_batch = env.step(np.asarray([[1, -1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(
        terminal_batch.info["score"],
        np.asarray([[1, 0, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[1, 3, 2, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.info["match_done"], np.asarray([False]))
    _assert_public_final_rows(
        terminal_batch,
        [0],
        expected_reward=terminal_batch.reward,
    )

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(warmdown_batch.info["present"], PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(warmdown_batch.info["alive"], PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[1, 0, 1, 1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["death_player"],
        np.asarray([[1, -1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        PRESENT_ABSENT_4P_ACTION_MASK,
    )
    _assert_public_final_rows(warmdown_batch, [], expected_reward=None)

    surface = _make_surface(
        scenario_name,
        player_count=4,
        episode_end_mode="match",
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = _reset_surface(surface, scenario_name, present=PRESENT_ABSENT_4P)
    recorder.record(reset_step, source_ref=scenario_name)
    _assert_policy_rows(reset_step, [(0, 0), (0, 2), (0, 3)])

    _force_present_absent_4p_draw(surface.env)
    round_step = surface.step(np.asarray([[1, -1, 1, 1]], dtype=np.int16))
    recorder.record(round_step, source_ref=scenario_name)

    np.testing.assert_array_equal(round_step.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(round_step.info["round_done"], np.asarray([True]))
    np.testing.assert_array_equal(round_step.live_mask, np.zeros((1, 4), dtype=bool))
    np.testing.assert_array_equal(
        round_step.final_observation_row_mask,
        np.asarray([False], dtype=bool),
    )
    _assert_policy_rows(round_step, [])

    warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(warmdown_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    assert warmdown_step.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_step.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(warmdown_step.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(warmdown_step.info["present"], PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(warmdown_step.live_mask, PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(
        warmdown_step.legal_action_mask,
        PRESENT_ABSENT_4P_ACTION_MASK,
    )
    _assert_policy_rows(warmdown_step, [(0, 0), (0, 2), (0, 3)])

    assert chunk.metadata["record_count"] == 3
    assert chunk.metadata["closed_by_terminal"] is False
    assert [rows["policy_env_row"].size for rows in chunk.policy_rows] == [3, 0, 3]
    np.testing.assert_array_equal(chunk.arrays["live_mask"][2], PRESENT_ABSENT_4P)
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"],
        np.zeros((3, 1), dtype=bool),
    )


@pytest.mark.parametrize(
    ("scenario_name", "player_count", "expected_score"),
    [
        ("source_lifecycle_match_end_at_max_score_2p.json", 2, [[1, 0]]),
        ("source_lifecycle_match_end_at_max_score_3p.json", 3, [[2, 0, 0]]),
    ],
)
def test_match_end_source_fixtures_public_and_replay_final_rows(
    scenario_name: str,
    player_count: int,
    expected_score: list[list[int]],
):
    env, _ = _reset_public(
        scenario_name,
        player_count=player_count,
        episode_end_mode="match",
    )
    _force_player0_round_win(env, player_count=player_count)
    round_batch = env.step(np.ones((1, player_count), dtype=np.int16))

    np.testing.assert_array_equal(round_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(round_batch.info["round_done"], np.asarray([True]))
    np.testing.assert_array_equal(round_batch.info["match_done"], np.asarray([False]))
    np.testing.assert_array_equal(
        round_batch.info["score"],
        np.asarray(expected_score, dtype=np.int32),
    )
    np.testing.assert_array_equal(round_batch.info["winner"], np.asarray([0]))
    np.testing.assert_array_equal(round_batch.action_mask, np.zeros((1, player_count, 3)))
    _assert_public_final_rows(round_batch, [], expected_reward=None)

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 1
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(warmdown_batch.info["match_done"], np.asarray([True]))
    np.testing.assert_array_equal(
        warmdown_batch.info["match_winner"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray(expected_score, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.zeros((1, player_count, 3), dtype=bool),
    )
    _assert_public_final_rows(
        warmdown_batch,
        [0],
        expected_reward=np.zeros((1, player_count), dtype=np.float32),
    )

    surface = _make_surface(
        scenario_name,
        player_count=player_count,
        episode_end_mode="match",
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = _reset_surface(surface, scenario_name)
    recorder.record(reset_step, source_ref=scenario_name)
    _assert_policy_rows(reset_step, [(0, player) for player in range(player_count)])

    _force_player0_round_win(surface.env, player_count=player_count)
    round_step = surface.step(np.ones((1, player_count), dtype=np.int16))
    recorder.record(round_step, source_ref=scenario_name)
    warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(warmdown_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(round_step.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(round_step.live_mask, np.zeros((1, player_count)))
    _assert_policy_rows(round_step, [])
    np.testing.assert_array_equal(warmdown_step.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_step.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_step.final_reward_map,
        np.zeros((1, player_count), dtype=np.float32),
    )
    _assert_policy_rows(warmdown_step, [])

    assert chunk.metadata["record_count"] == 3
    assert chunk.metadata["closed_by_terminal"] is True
    assert [rows["policy_env_row"].size for rows in chunk.policy_rows] == [
        player_count,
        0,
        0,
    ]
    assert chunk.records[2]["trainer_surface_api"] == "advance_warmdown"
    assert chunk.records[2]["source_ref"] == scenario_name
    assert chunk.records[2]["done_rows"] == [0]
    assert chunk.records[2]["final_observation_rows"] == [0]
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][2],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_reward_map"][2],
        np.zeros((1, player_count), dtype=np.float32),
    )


def test_match_end_4p_source_fixture_public_and_replay_final_rows():
    scenario_name = "source_lifecycle_match_end_at_max_score_4p.json"
    env, reset_batch = _reset_public(
        scenario_name,
        player_count=4,
        episode_end_mode="match",
    )

    assert int(reset_batch.info["max_score_by_row"][0]) == 3
    _force_4p_unique_leader_first_death(env)
    first_death = env.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(first_death.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        first_death.info["alive"],
        np.asarray([[True, True, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        first_death.info["death_player"],
        np.asarray([[3, -1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        first_death.info["score"],
        np.zeros((1, 4), dtype=np.int32),
    )

    _force_4p_unique_leader_second_death(env)
    second_death = env.step(np.asarray([[1, 1, 1, -1]], dtype=np.int16))

    np.testing.assert_array_equal(second_death.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        second_death.info["alive"],
        np.asarray([[True, True, False, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        second_death.info["death_player"],
        np.asarray([[3, 2, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second_death.info["score"],
        np.zeros((1, 4), dtype=np.int32),
    )

    _force_4p_unique_leader_terminal_death(env)
    round_batch = env.step(np.asarray([[1, 1, -1, -1]], dtype=np.int16))

    np.testing.assert_array_equal(round_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(round_batch.info["round_done"], np.asarray([True]))
    np.testing.assert_array_equal(round_batch.info["match_done"], np.asarray([False]))
    np.testing.assert_array_equal(
        round_batch.info["score"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        round_batch.info["death_player"],
        np.asarray([[3, 2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(round_batch.info["winner"], np.asarray([0]))
    np.testing.assert_array_equal(round_batch.action_mask, np.zeros((1, 4, 3)))
    _assert_public_final_rows(round_batch, [], expected_reward=None)

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 1
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(warmdown_batch.info["match_done"], np.asarray([True]))
    np.testing.assert_array_equal(
        warmdown_batch.info["match_winner"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["score"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.zeros((1, 4, 3), dtype=bool),
    )
    _assert_public_final_rows(
        warmdown_batch,
        [0],
        expected_reward=np.zeros((1, 4), dtype=np.float32),
    )

    surface = _make_surface(
        scenario_name,
        player_count=4,
        episode_end_mode="match",
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = _reset_surface(surface, scenario_name)
    recorder.record(reset_step, source_ref=scenario_name)
    _assert_policy_rows(reset_step, [(0, player) for player in range(4)])

    _force_4p_unique_leader_first_death(surface.env)
    first_death_step = surface.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))
    recorder.record(first_death_step, source_ref=scenario_name)
    np.testing.assert_array_equal(
        first_death_step.live_mask,
        np.asarray([[True, True, True, False]], dtype=bool),
    )
    _assert_policy_rows(first_death_step, [(0, 0), (0, 1), (0, 2)])

    _force_4p_unique_leader_second_death(surface.env)
    second_death_step = surface.step(np.asarray([[1, 1, 1, -1]], dtype=np.int16))
    recorder.record(second_death_step, source_ref=scenario_name)
    np.testing.assert_array_equal(
        second_death_step.live_mask,
        np.asarray([[True, True, False, False]], dtype=bool),
    )
    _assert_policy_rows(second_death_step, [(0, 0), (0, 1)])

    _force_4p_unique_leader_terminal_death(surface.env)
    round_step = surface.step(np.asarray([[1, 1, -1, -1]], dtype=np.int16))
    recorder.record(round_step, source_ref=scenario_name)

    np.testing.assert_array_equal(round_step.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(round_step.live_mask, np.zeros((1, 4), dtype=bool))
    np.testing.assert_array_equal(
        round_step.reward,
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        round_step.final_observation_row_mask,
        np.asarray([False], dtype=bool),
    )
    _assert_policy_rows(round_step, [])

    warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(warmdown_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(warmdown_step.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_step.info["match_done"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_step.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_step.final_reward_map,
        np.zeros((1, 4), dtype=np.float32),
    )
    _assert_policy_rows(warmdown_step, [])

    assert chunk.metadata["record_count"] == 5
    assert chunk.metadata["closed_by_terminal"] is True
    assert [rows["policy_env_row"].size for rows in chunk.policy_rows] == [
        4,
        3,
        2,
        0,
        0,
    ]
    assert chunk.records[-1]["trainer_surface_api"] == "advance_warmdown"
    assert chunk.records[-1]["source_ref"] == scenario_name
    assert chunk.records[-1]["done_rows"] == [0]
    assert chunk.records[-1]["final_observation_rows"] == [0]
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][-1],
        np.asarray([True], dtype=bool),
    )


def test_multi_round_3p_match_end_fixture_public_and_replay():
    scenario_name = "source_lifecycle_multi_round_match_end_3p.json"
    env, _ = _reset_public(
        scenario_name,
        player_count=3,
        episode_end_mode="match",
    )

    _force_player0_round_win(env, player_count=3)
    first_round = env.step(np.ones((1, 3), dtype=np.int16))
    first_warmdown = env.advance_warmdown(5000.0)

    np.testing.assert_array_equal(first_round.info["score"], np.asarray([[2, 0, 0]]))
    np.testing.assert_array_equal(first_round.done, np.asarray([False], dtype=bool))
    assert first_warmdown.info["warmdown_info"]["next_round_count"] == 1
    assert first_warmdown.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(first_warmdown.done, np.asarray([False], dtype=bool))
    _assert_public_final_rows(first_warmdown, [], expected_reward=None)

    game_start = env.advance_warmup(3000.0)
    assert game_start.info["warmup_info"]["game_start_fires"] == 1
    _prepare_second_round_safe_3p(env)
    print_start = env.advance_warmup(3000.0)
    assert print_start.info["warmup_info"]["print_manager_delayed_start_fires"] == 3

    _force_second_round_3p_player0_win(env)
    second_round = env.step(np.ones((1, 3), dtype=np.int16))
    second_warmdown = env.advance_warmdown(5000.0)

    np.testing.assert_array_equal(second_round.info["score"], np.asarray([[4, 0, 0]]))
    np.testing.assert_array_equal(second_round.done, np.asarray([False], dtype=bool))
    _assert_public_final_rows(second_round, [], expected_reward=None)
    assert second_warmdown.info["warmdown_info"]["next_round_count"] == 0
    assert second_warmdown.info["warmdown_info"]["match_end_count"] == 1
    np.testing.assert_array_equal(second_warmdown.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(second_warmdown.info["match_done"], np.asarray([True]))
    np.testing.assert_array_equal(second_warmdown.info["score"], np.asarray([[4, 0, 0]]))
    _assert_public_final_rows(
        second_warmdown,
        [0],
        expected_reward=np.zeros((1, 3), dtype=np.float32),
    )

    surface = _make_surface(
        scenario_name,
        player_count=3,
        episode_end_mode="match",
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = _reset_surface(surface, scenario_name)
    recorder.record(reset_step, source_ref=scenario_name)

    _force_player0_round_win(surface.env, player_count=3)
    first_round_step = surface.step(np.ones((1, 3), dtype=np.int16))
    recorder.record(first_round_step, source_ref=scenario_name)
    first_warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(first_warmdown_step, source_ref=scenario_name)
    game_start_step = surface.advance_warmup(3000.0)
    recorder.record(game_start_step, source_ref=scenario_name)
    _prepare_second_round_safe_3p(surface.env)
    print_start_step = surface.advance_warmup(3000.0)
    recorder.record(print_start_step, source_ref=scenario_name)
    _force_second_round_3p_player0_win(surface.env)
    second_round_step = surface.step(np.ones((1, 3), dtype=np.int16))
    recorder.record(second_round_step, source_ref=scenario_name)
    second_warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(second_warmdown_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(first_round_step.live_mask, np.zeros((1, 3)))
    np.testing.assert_array_equal(first_warmdown_step.live_mask, np.ones((1, 3)))
    np.testing.assert_array_equal(game_start_step.live_mask, np.ones((1, 3)))
    np.testing.assert_array_equal(print_start_step.live_mask, np.ones((1, 3)))
    np.testing.assert_array_equal(second_round_step.live_mask, np.zeros((1, 3)))
    np.testing.assert_array_equal(second_warmdown_step.live_mask, np.zeros((1, 3)))
    np.testing.assert_array_equal(
        second_warmdown_step.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        second_warmdown_step.final_reward_map,
        np.zeros((1, 3), dtype=np.float32),
    )

    assert chunk.metadata["record_count"] == 7
    assert chunk.metadata["closed_by_terminal"] is True
    assert [rows["policy_env_row"].size for rows in chunk.policy_rows] == [
        3,
        0,
        3,
        3,
        3,
        0,
        0,
    ]
    assert chunk.records[-1]["trainer_surface_api"] == "advance_warmdown"
    assert chunk.records[-1]["source_ref"] == scenario_name
    assert chunk.records[-1]["done_rows"] == [0]
    assert chunk.records[-1]["final_observation_rows"] == [0]
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][-1],
        np.asarray([True], dtype=bool),
    )


def test_multi_round_4p_match_end_fixture_public_and_replay():
    scenario_name = "source_lifecycle_multi_round_match_end_4p.json"
    env, reset_batch = _reset_public(
        scenario_name,
        player_count=4,
        episode_end_mode="match",
    )

    assert int(reset_batch.info["max_score_by_row"][0]) == 5
    _force_player0_round_win(env, player_count=4)
    first_round = env.step(np.ones((1, 4), dtype=np.int16))
    first_warmdown = env.advance_warmdown(5000.0)

    np.testing.assert_array_equal(first_round.info["score"], np.asarray([[3, 0, 0, 0]]))
    np.testing.assert_array_equal(
        first_round.info["death_player"],
        np.asarray([[3, 2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(first_round.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(first_round.info["match_done"], np.asarray([False]))
    _assert_public_final_rows(first_round, [], expected_reward=None)
    assert first_warmdown.info["warmdown_info"]["next_round_count"] == 1
    assert first_warmdown.info["warmdown_info"]["match_end_count"] == 0
    np.testing.assert_array_equal(first_warmdown.done, np.asarray([False], dtype=bool))
    _assert_public_final_rows(first_warmdown, [], expected_reward=None)

    game_start = env.advance_warmup(3000.0)
    assert game_start.info["warmup_info"]["game_start_fires"] == 1
    _force_player0_round_win(env, player_count=4)
    print_start = env.advance_warmup(3000.0)
    assert print_start.info["warmup_info"]["print_manager_delayed_start_fires"] == 4

    _force_player0_round_win(env, player_count=4)
    second_round = env.step(np.ones((1, 4), dtype=np.int16))
    second_warmdown = env.advance_warmdown(5000.0)

    np.testing.assert_array_equal(
        second_round.info["score"],
        np.asarray([[6, 0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(second_round.done, np.asarray([False], dtype=bool))
    _assert_public_final_rows(second_round, [], expected_reward=None)
    assert second_warmdown.info["warmdown_info"]["next_round_count"] == 0
    assert second_warmdown.info["warmdown_info"]["match_end_count"] == 1
    np.testing.assert_array_equal(second_warmdown.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(second_warmdown.info["match_done"], np.asarray([True]))
    np.testing.assert_array_equal(
        second_warmdown.info["match_winner"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        second_warmdown.info["score"],
        np.asarray([[6, 0, 0, 0]], dtype=np.int32),
    )
    _assert_public_final_rows(
        second_warmdown,
        [0],
        expected_reward=np.zeros((1, 4), dtype=np.float32),
    )

    surface = _make_surface(
        scenario_name,
        player_count=4,
        episode_end_mode="match",
    )
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    reset_step = _reset_surface(surface, scenario_name)
    recorder.record(reset_step, source_ref=scenario_name)

    _force_player0_round_win(surface.env, player_count=4)
    first_round_step = surface.step(np.ones((1, 4), dtype=np.int16))
    recorder.record(first_round_step, source_ref=scenario_name)
    first_warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(first_warmdown_step, source_ref=scenario_name)
    game_start_step = surface.advance_warmup(3000.0)
    recorder.record(game_start_step, source_ref=scenario_name)
    _force_player0_round_win(surface.env, player_count=4)
    print_start_step = surface.advance_warmup(3000.0)
    recorder.record(print_start_step, source_ref=scenario_name)
    _force_player0_round_win(surface.env, player_count=4)
    second_round_step = surface.step(np.ones((1, 4), dtype=np.int16))
    recorder.record(second_round_step, source_ref=scenario_name)
    second_warmdown_step = surface.advance_warmdown(5000.0)
    recorder.record(second_warmdown_step, source_ref=scenario_name)
    chunk = recorder.build_chunk()

    np.testing.assert_array_equal(first_round_step.live_mask, np.zeros((1, 4)))
    np.testing.assert_array_equal(first_warmdown_step.live_mask, np.ones((1, 4)))
    np.testing.assert_array_equal(game_start_step.live_mask, np.ones((1, 4)))
    np.testing.assert_array_equal(print_start_step.live_mask, np.ones((1, 4)))
    np.testing.assert_array_equal(second_round_step.live_mask, np.zeros((1, 4)))
    np.testing.assert_array_equal(second_warmdown_step.live_mask, np.zeros((1, 4)))
    np.testing.assert_array_equal(
        second_warmdown_step.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        second_warmdown_step.final_reward_map,
        np.zeros((1, 4), dtype=np.float32),
    )

    assert chunk.metadata["record_count"] == 7
    assert chunk.metadata["closed_by_terminal"] is True
    assert [rows["policy_env_row"].size for rows in chunk.policy_rows] == [
        4,
        0,
        4,
        4,
        4,
        0,
        0,
    ]
    assert chunk.records[-1]["trainer_surface_api"] == "advance_warmdown"
    assert chunk.records[-1]["source_ref"] == scenario_name
    assert chunk.records[-1]["done_rows"] == [0]
    assert chunk.records[-1]["final_observation_rows"] == [0]
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][-1],
        np.asarray([True], dtype=bool),
    )


def _reset_public(
    scenario_name: str,
    *,
    player_count: int,
    episode_end_mode: str = "round",
    present: np.ndarray | None = None,
    source_fixture_new_round_time_ms: float = 0.0,
    source_fixture_warmup_advance_ms: float = 3000.0,
) -> tuple[VectorMultiplayerEnv, object]:
    tape = _lifecycle_random_tape(scenario_name)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        seed=555,
        decision_ms=100.0,
        max_score=_lifecycle_max_score(scenario_name),
        episode_end_mode=episode_end_mode,
        body_capacity=64,
        event_capacity=64,
        timer_capacity=max(4, player_count),
        random_tape_capacity=tape.shape[1],
    )
    batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=present,
        source_fixture_random_tape_values=tape,
        source_fixture_ref=f"scenarios/environment/{scenario_name}",
        source_fixture_new_round_time_ms=source_fixture_new_round_time_ms,
        source_fixture_warmup_advance_ms=source_fixture_warmup_advance_ms,
    )
    return env, batch


def _make_surface(
    scenario_name: str,
    *,
    player_count: int,
    episode_end_mode: str = "round",
) -> SourceStateMultiplayerTrainerSurface:
    tape = _lifecycle_random_tape(scenario_name)
    return SourceStateMultiplayerTrainerSurface(
        batch_size=1,
        player_count=player_count,
        seed=555,
        decision_ms=100.0,
        max_score=_lifecycle_max_score(scenario_name),
        episode_end_mode=episode_end_mode,
        body_capacity=64,
        event_capacity=64,
        timer_capacity=max(4, player_count),
        random_tape_capacity=tape.shape[1],
        natural_bonus_spawn=False,
    )


def _reset_surface(
    surface: SourceStateMultiplayerTrainerSurface,
    scenario_name: str,
    *,
    present: np.ndarray | None = None,
) -> object:
    return surface.reset(
        seed=np.asarray([555], dtype=np.uint64),
        present=present,
        source_fixture_random_tape_values=_lifecycle_random_tape(scenario_name),
        source_fixture_ref=f"scenarios/environment/{scenario_name}",
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )


def _lifecycle_random_tape(scenario_name: str) -> np.ndarray:
    sequence = _lifecycle_payload(scenario_name)["source_setup"]["random"][
        "math_random_sequence"
    ]
    return np.asarray([sequence], dtype=np.float64)


def _lifecycle_max_score(scenario_name: str) -> int:
    return int(_lifecycle_payload(scenario_name)["source_setup"]["room"]["max_score"])


def _lifecycle_payload(scenario_name: str) -> dict[str, object]:
    with (SCENARIO_ROOT / scenario_name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _force_present_absent_survivor_score(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0, 0] = np.asarray([20.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 0] = 0.0
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]


def _force_present_absent_draw(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    env.state["pos"][0, 0] = np.asarray([1.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 0] = math.pi
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]


def _force_present_absent_4p_first_death(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0, 0] = np.asarray([10.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 0] = 0.0
    env.state["speed"][0, 0] = 8.0
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["pos"][0, 3] = np.asarray([99.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 3] = 0.0
    env.state["prev_pos"][0, 3] = env.state["pos"][0, 3]


def _force_present_absent_4p_terminal_death(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 2] = math.pi
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]


def _force_present_absent_4p_draw(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0, 3] = np.asarray([99.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 3] = 0.0
    env.state["prev_pos"][0, 3] = env.state["pos"][0, 3]
    env.state["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    env.state["heading"][0, 2] = math.pi
    env.state["prev_pos"][0, 2] = env.state["pos"][0, 2]
    env.state["pos"][0, 0] = np.asarray([50.5, 1.0], dtype=np.float64)
    env.state["heading"][0, 0] = math.tau * 0.75
    env.state["prev_pos"][0, 0] = env.state["pos"][0, 0]


def _force_4p_unique_leader_first_death(env: VectorMultiplayerEnv) -> None:
    _force_present_absent_4p_first_death(env)


def _force_4p_unique_leader_second_death(env: VectorMultiplayerEnv) -> None:
    _force_present_absent_4p_terminal_death(env)


def _force_4p_unique_leader_terminal_death(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0, 1] = np.asarray([50.5, 1.0], dtype=np.float64)
    env.state["heading"][0, 1] = math.tau * 0.75
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]


def _force_player0_round_win(
    env: VectorMultiplayerEnv,
    *,
    player_count: int,
) -> None:
    if player_count == 2:
        env.state["pos"][0] = np.asarray(
            [[5.0, 5.0], [87.0, 44.0]],
            dtype=np.float64,
        )
        env.state["heading"][0] = np.asarray([math.pi / 4.0, 0.0], dtype=np.float64)
        env.state["speed"][0, 0] = 8.0
    elif player_count == 3:
        env.state["pos"][0] = np.asarray(
            [[5.0, 5.0], [1.0, 47.5], [93.0, 47.5]],
            dtype=np.float64,
        )
        env.state["heading"][0] = np.asarray(
            [math.pi / 4.0, math.pi, 0.0],
            dtype=np.float64,
        )
        env.state["speed"][0] = np.asarray([8.0, 16.0, 16.0], dtype=np.float64)
    elif player_count == 4:
        env.state["pos"][0] = np.asarray(
            [[10.0, 50.5], [50.5, 1.0], [1.0, 50.5], [99.0, 50.5]],
            dtype=np.float64,
        )
        env.state["heading"][0] = np.asarray(
            [0.0, math.tau * 0.75, math.pi, 0.0],
            dtype=np.float64,
        )
        env.state["speed"][0] = np.asarray([8.0, 16.0, 16.0, 16.0], dtype=np.float64)
    else:
        raise AssertionError(f"unexpected player_count={player_count}")

    env.state["prev_pos"][0, :player_count] = env.state["pos"][0, :player_count]
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]


def _prepare_second_round_safe_3p(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0] = np.asarray(
        [[5.0, 5.0], [70.0, 20.0], [70.0, 70.0]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray(
        [math.pi / 4.0, math.pi / 4.0, math.tau * 0.875],
        dtype=np.float64,
    )
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["speed"][0] = np.asarray([8.0, 8.0, 8.0], dtype=np.float64)


def _force_second_round_3p_player0_win(env: VectorMultiplayerEnv) -> None:
    env.state["pos"][0, 0] = np.asarray([5.0, 5.0], dtype=np.float64)
    env.state["heading"][0, 0] = math.pi / 4.0
    env.state["speed"][0, 0] = 8.0
    env.state["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 2] = 0.0
    env.state["speed"][0, 2] = 16.0
    env.state["pos"][0, 1] = np.asarray([1.0, 47.5], dtype=np.float64)
    env.state["heading"][0, 1] = math.pi
    env.state["speed"][0, 1] = 16.0
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["print_manager_distance"][0, 0] = 999.0
    env.state["print_manager_last_pos"][0, 0] = env.state["pos"][0, 0]


def _assert_public_final_rows(
    batch,
    rows: list[int],
    *,
    expected_reward: np.ndarray | None,
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
        assert expected_reward is None
        return

    assert batch.final_observation is not None
    assert batch.final_reward is not None
    assert expected_reward is not None
    expected_final_observation = np.zeros_like(batch.observation)
    expected_final_observation[rows_array] = batch.observation[rows_array]
    expected_final_reward = np.zeros_like(batch.reward)
    expected_final_reward[rows_array] = expected_reward[rows_array]
    np.testing.assert_array_equal(batch.final_observation, expected_final_observation)
    np.testing.assert_array_equal(batch.final_reward, expected_final_reward)
    np.testing.assert_array_equal(batch.info["final_reward_map"], expected_final_reward)


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
