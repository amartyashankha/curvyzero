import numpy as np

from curvyzero.env import vector_reset
from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)


def test_3p_terminal_hit_owner_reaches_public_trainer_replay_and_debug_events():
    expected_death_player = [2, 0, -1]
    expected_death_hit_owner = [1, 1, -1]
    expected_death_cause = [
        vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
        vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
        vector_runtime.DEATH_CAUSE_NONE,
    ]

    public_env = _hit_owner_env(player_count=3)
    _seed_body_on_player(public_env, owner=1, victim=0)
    _seed_body_on_player(public_env, owner=1, victim=2)
    public_batch = public_env.step(_straight_actions(player_count=3))

    _assert_public_death_facts(
        public_batch,
        expected_alive=[False, True, False],
        expected_death_player=expected_death_player,
        expected_death_hit_owner=expected_death_hit_owner,
        expected_death_cause=expected_death_cause,
    )
    np.testing.assert_array_equal(public_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        public_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    assert public_batch.info["terminal_reason_name"].tolist() == ["round_survivor_win"]
    np.testing.assert_array_equal(public_batch.info["winner"], np.asarray([1], dtype=np.int16))
    assert public_batch.info["winner_ids"] == [[1]]
    assert public_batch.info["loser_ids"] == [[0, 2]]
    np.testing.assert_array_equal(
        public_batch.reward,
        np.asarray([[-1.0, 1.0, -1.0]], dtype=np.float32),
    )
    assert public_batch.final_observation is not None
    _assert_die_events(public_env, [(0, 1, 0), (2, 1, 0)])

    surface = _hit_owner_surface(player_count=3)
    _seed_body_on_player(surface.env, owner=1, victim=0)
    _seed_body_on_player(surface.env, owner=1, victim=2)
    step = surface.step(_straight_actions(player_count=3))

    _assert_public_death_facts(
        step,
        expected_alive=[False, True, False],
        expected_death_player=expected_death_player,
        expected_death_hit_owner=expected_death_hit_owner,
        expected_death_cause=expected_death_cause,
    )
    np.testing.assert_array_equal(step.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(step.final_observation_row_mask, np.asarray([True]))
    np.testing.assert_array_equal(step.final_observation[0], step.observation[0])
    assert int(np.count_nonzero(step.final_observation[0])) > 0
    np.testing.assert_array_equal(
        step.reward,
        np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(step.final_reward_map, step.reward)
    assert step.info["winner_ids"] == [[1]]
    assert step.info["loser_ids"] == [[0, 2]]
    _assert_die_events(surface.env, [(0, 1, 0), (2, 1, 0)])

    chunk = _record_one(step, source_ref="3p-terminal-hit-owner")
    record = chunk.records[0]
    assert chunk.metadata["closed_by_terminal"] is True
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(chunk.arrays["final_observation"][0], step.final_observation)
    np.testing.assert_array_equal(chunk.arrays["final_reward_map"][0], step.final_reward_map)
    assert record["terminal_or_final"] is True
    assert record["final_observation_rows"] == [0]
    assert record["death_player"] == [expected_death_player]
    assert record["death_hit_owner"] == [expected_death_hit_owner]
    assert record["death_cause"] == [expected_death_cause]
    assert record["death_cause_name"] == [["opponent_trail", "opponent_trail", "none"]]
    assert record["winner_ids"] == [[1]]
    assert record["loser_ids"] == [[0, 2]]
    assert record["alive"] == [[False, True, False]]
    assert record["score"] == [[0, 2, 0]]
    assert record["source_ref"] == "3p-terminal-hit-owner"
    assert record["final_observation_policy"]["metadata_only"] is False


def test_4p_nonterminal_two_victim_hit_owner_reaches_trainer_replay_and_debug_events():
    expected_death_player = [2, 0, -1, -1]
    expected_death_hit_owner = [1, 3, -1, -1]
    expected_death_cause = [
        vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
        vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
        vector_runtime.DEATH_CAUSE_NONE,
        vector_runtime.DEATH_CAUSE_NONE,
    ]

    public_env = _hit_owner_env(player_count=4)
    _seed_body_on_player(public_env, owner=1, victim=2)
    _seed_body_on_player(public_env, owner=3, victim=0)
    public_batch = public_env.step(_straight_actions(player_count=4))

    _assert_public_death_facts(
        public_batch,
        expected_alive=[False, True, False, True],
        expected_death_player=expected_death_player,
        expected_death_hit_owner=expected_death_hit_owner,
        expected_death_cause=expected_death_cause,
    )
    np.testing.assert_array_equal(public_batch.done, np.asarray([False], dtype=bool))
    assert public_batch.final_observation is None
    _assert_die_events(public_env, [(0, 3, 0), (2, 1, 0)])

    surface = _hit_owner_surface(player_count=4)
    _seed_body_on_player(surface.env, owner=1, victim=2)
    _seed_body_on_player(surface.env, owner=3, victim=0)
    step = surface.step(_straight_actions(player_count=4))

    _assert_public_death_facts(
        step,
        expected_alive=[False, True, False, True],
        expected_death_player=expected_death_player,
        expected_death_hit_owner=expected_death_hit_owner,
        expected_death_cause=expected_death_cause,
    )
    np.testing.assert_array_equal(step.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(step.final_observation_row_mask, np.asarray([False]))
    np.testing.assert_array_equal(step.final_observation, np.zeros_like(step.final_observation))
    np.testing.assert_array_equal(
        step.reward,
        np.asarray([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32),
    )
    _assert_die_events(surface.env, [(0, 3, 0), (2, 1, 0)])

    chunk = _record_one(step, source_ref="4p-nonterminal-two-victim-hit-owner")
    record = chunk.records[0]
    assert chunk.metadata["closed_by_terminal"] is False
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([False], dtype=bool),
    )
    assert record["terminal_or_final"] is False
    assert record["final_observation_rows"] == []
    assert record["death_player"] == [expected_death_player]
    assert record["death_hit_owner"] == [expected_death_hit_owner]
    assert record["death_cause"] == [expected_death_cause]
    assert record["death_cause_name"] == [
        ["opponent_trail", "opponent_trail", "none", "none"]
    ]
    assert record["alive"] == [[False, True, False, True]]
    assert record["source_ref"] == "4p-nonterminal-two-victim-hit-owner"


def _hit_owner_surface(*, player_count: int) -> SourceStateMultiplayerTrainerSurface:
    env = _hit_owner_env(player_count=player_count)
    surface = SourceStateMultiplayerTrainerSurface(env=env)
    surface.stack.reset_rows(env, np.asarray([True], dtype=bool))
    return surface


def _hit_owner_env(*, player_count: int) -> VectorMultiplayerEnv:
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        decision_ms=1.0,
        body_capacity=16,
        event_capacity=32,
        timer_capacity=max(4, player_count),
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        {name: array.copy() for name, array in env.reset_template.items()},
        reset_seed=np.asarray([123], dtype=np.uint64),
    )
    _clear_for_zero_ms_body_hit(env)
    env.decision_ms = 0.0
    return env


def _clear_for_zero_ms_body_hit(env: VectorMultiplayerEnv) -> None:
    state = env.state
    player_count = env.player_count
    size = float(state["map_size"][0])
    positions = np.asarray(
        [
            [20.0, 20.0],
            [size - 20.0, 20.0],
            [20.0, size - 20.0],
            [size - 20.0, size - 20.0],
        ],
        dtype=np.float64,
    )
    state["timer_active"][0] = False
    state["done"][0] = False
    state["terminated"][0] = False
    state["truncated"][0] = False
    state["terminal_reason"][0] = vector_reset.TERMINAL_REASON_NONE
    state["reset_pending"][0] = False
    state["round_done"][0] = False
    state["warmdown_pending"][0] = False
    state["match_done"][0] = False
    state["started"][0] = True
    state["in_round"][0] = True
    state["world_active"][0] = True
    state["draw"][0] = False
    state["winner"][0] = -1
    state["alive"][0, :player_count] = True
    state["present"][0, :player_count] = True
    state["score"][0, :player_count] = 0
    state["round_score"][0, :player_count] = 0
    state["pos"][0, :player_count] = positions[:player_count]
    state["prev_pos"][0, :player_count] = positions[:player_count]
    state["heading"][0, :player_count] = 0.0
    state["angular_velocity_per_ms"][0, :player_count] = 0.0
    state["printing"][0, :player_count] = False
    state["print_manager_active"][0, :player_count] = False
    state["death_count"][0] = 0
    state["death_player"][0] = -1
    state["death_cause"][0] = vector_runtime.DEATH_CAUSE_NONE
    state["death_hit_owner"][0] = -1
    state["body_active"][0] = False
    state["body_owner"][0] = -1
    state["body_num"][0] = -1
    state["body_insert_tick"][0] = -1
    state["body_insert_kind"][0] = -1
    state["body_break_before"][0] = False
    state["body_write_cursor"][0] = 0
    state["body_count"][0, :player_count] = 0
    state["live_body_num"][0, :player_count] = 0
    state["world_body_count"][0] = 0
    state["event_count"][0] = 0
    state["event_mask"][0] = False
    state["event_type"][0] = vector_runtime.EVENT_NONE
    state["event_player"][0] = -1
    state["event_other"][0] = -1
    state["event_bool"][0] = -1
    state["event_value_i"][0] = 0
    state["event_value_f"][0] = 0.0
    state["event_overflow"][0] = False
    state["event_overflow_attempts"][0] = 0


def _seed_body_on_player(
    env: VectorMultiplayerEnv,
    *,
    owner: int,
    victim: int,
) -> None:
    state = env.state
    slot = int(state["body_write_cursor"][0])
    state["body_active"][0, slot] = True
    state["body_pos"][0, slot] = state["pos"][0, victim]
    state["body_radius"][0, slot] = state["radius"][0, owner]
    state["body_owner"][0, slot] = owner
    state["body_num"][0, slot] = int(state["body_count"][0, owner])
    state["body_insert_tick"][0, slot] = int(state["tick"][0])
    state["body_insert_kind"][0, slot] = vector_runtime.BODY_KIND_NORMAL
    state["body_write_cursor"][0] = slot + 1
    state["world_body_count"][0] += 1
    state["body_count"][0, owner] += 1


def _straight_actions(*, player_count: int) -> np.ndarray:
    return np.ones((1, player_count), dtype=np.int16)


def _assert_public_death_facts(
    batch,
    *,
    expected_alive: list[bool],
    expected_death_player: list[int],
    expected_death_hit_owner: list[int],
    expected_death_cause: list[int],
) -> None:
    np.testing.assert_array_equal(
        batch.info["alive"],
        np.asarray([expected_alive], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["death_count"],
        np.asarray(
            [sum(1 for player in expected_death_player if player >= 0)],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(
        batch.info["death_player"],
        np.asarray([expected_death_player], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["death_hit_owner"],
        np.asarray([expected_death_hit_owner], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        batch.info["death_cause"],
        np.asarray([expected_death_cause], dtype=np.int16),
    )


def _assert_die_events(
    env: VectorMultiplayerEnv,
    expected: list[tuple[int, int, int]],
) -> None:
    count = int(env.state["event_count"][0])
    event_types = env.state["event_type"][0, :count]
    die_indices = np.flatnonzero(event_types == vector_runtime.EVENT_DIE)
    actual = sorted(
        (
            int(env.state["event_player"][0, index]),
            int(env.state["event_other"][0, index]),
            int(env.state["event_bool"][0, index]),
        )
        for index in die_indices
    )
    assert actual == sorted(expected)


def _record_one(step, *, source_ref: str):
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    recorder.record(step, source_ref=source_ref)
    return recorder.build_chunk()
