import json
from pathlib import Path
import sys

import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayRecorder,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    SourceStateMultiplayerTrainerSurface,
)


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402


SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenarios" / "environment"


def test_replay_preserves_bonus_self_fast_stack_death_terminal_facts():
    surface, fixture = _source_fixture_surface(
        "source_bonus_self_fast_stack_death_late_expiry_step.json",
    )
    env = surface.env
    env.seed_active_bonus(
        row=0,
        bonus_type="BonusSelfFast",
        x=20.0,
        y=20.0,
        bonus_id=1,
        bonus_capacity=3,
        stack_capacity=vector_runtime.SOURCE_MAX_ACTIVE_BONUSES,
    )
    env.state["bonus_active"][0, 1:] = True
    env.state["bonus_type"][0, 1:] = vector_runtime.BONUS_TYPE_SELF_FAST
    env.state["bonus_id"][0, 1:] = np.asarray([2, 3], dtype=np.int32)
    env.state["bonus_pos"][0, 1:] = np.asarray([[20.0, 20.0], [20.0, 20.0]])
    env.state["bonus_radius"][0, 1:] = vector_runtime.SOURCE_BONUS_RADIUS
    env.state["bonus_count"][0] = 3
    env.state["bonus_world_body_count"][0] = 3

    steps = []
    for step_index, expected_stack_count in enumerate((1, 2, 3)):
        step = _surface_fixture_step(surface, fixture, step_index)
        assert step.info["step_counters"]["bonus_self_fast_catches"] == 1
        np.testing.assert_array_equal(
            step.info["bonus_support"]["stack_count"],
            np.asarray([[expected_stack_count, 0]], dtype=np.int16),
        )
        steps.append(step)

    terminal = _surface_fixture_step(surface, fixture, 3)
    steps.append(terminal)

    _assert_terminal_visual_reward_and_bonus_death_facts(terminal)
    assert terminal.info["step_counters"]["normal_wall_deaths"] == 1
    assert terminal.info["step_counters"]["bonus_self_fast_expiries"] == 0
    np.testing.assert_array_equal(
        terminal.info["bonus_support"]["stack_count"],
        np.asarray([[0, 0]], dtype=np.int16),
    )
    assert env.state["speed"][0, 0] == 52.0

    chunk = _record_chunk(steps)
    _assert_terminal_replay_facts(
        chunk,
        terminal,
        terminal_index=3,
        expected_step_counters={
            "normal_wall_deaths": 1,
            "bonus_self_fast_expiries": 0,
        },
    )


def test_replay_preserves_bonus_self_fast_expiry_then_wall_death_terminal_facts():
    scenario_name = "source_bonus_self_fast_expiry_then_wall_death_same_tick_step.json"
    surface, fixture = _source_fixture_surface(scenario_name)
    bonus = _first_active_bonus(scenario_name)
    surface.env.seed_active_bonus(
        row=0,
        bonus_type=str(bonus["type"]),
        x=float(bonus["x"]),
        y=float(bonus["y"]),
        bonus_id=1,
    )

    catch = _surface_fixture_step(surface, fixture, 0)
    assert catch.info["step_counters"]["bonus_self_fast_catches"] == 1
    np.testing.assert_array_equal(
        catch.info["bonus_support"]["stack_count"],
        np.asarray([[1, 0]], dtype=np.int16),
    )

    terminal = _surface_fixture_step(surface, fixture, 1)
    _assert_terminal_visual_reward_and_bonus_death_facts(terminal)
    assert terminal.info["step_counters"]["bonus_self_fast_expiries"] == 1
    assert terminal.info["step_counters"]["normal_wall_deaths"] == 1
    np.testing.assert_array_equal(
        terminal.info["bonus_support"]["stack_count"],
        np.asarray([[0, 0]], dtype=np.int16),
    )
    assert surface.env.state["speed"][0, 0] == 16.0
    np.testing.assert_allclose(surface.env.state["pos"][0, 0], [-1.4, 20.0])

    chunk = _record_chunk([catch, terminal])
    _assert_terminal_replay_facts(
        chunk,
        terminal,
        terminal_index=1,
        expected_step_counters={
            "normal_wall_deaths": 1,
            "bonus_self_fast_expiries": 1,
        },
    )


def test_4p_replay_preserves_bonus_enemy_slow_stack_wall_death_terminal_facts():
    surface = _bonus_enemy_slow_4p_terminal_surface()

    catch = surface.step(np.ones((1, 4), dtype=np.int16))

    assert catch.info["step_counters"]["bonus_enemy_slow_catches"] == 1
    assert catch.info["step_counters"]["bonus_stack_appends"] == 3
    np.testing.assert_allclose(
        surface.env.state["speed"],
        np.asarray([[16.0, 8.0, 8.0, 8.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        catch.info["bonus_support"]["stack_count"],
        np.asarray([[0, 1, 1, 1]], dtype=np.int16),
    )

    _move_bonus_enemy_targets_to_wall_death(surface.env)
    terminal = surface.step(np.ones((1, 4), dtype=np.int16))

    np.testing.assert_array_equal(terminal.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal.truncated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        terminal.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(terminal.final_observation[0], terminal.observation[0])
    assert int(np.count_nonzero(terminal.final_observation[0])) > 0
    np.testing.assert_array_equal(
        terminal.final_reward_map,
        np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(terminal.reward, terminal.final_reward_map)
    assert terminal.info["step_counters"]["normal_wall_deaths"] == 3
    assert terminal.info["step_counters"]["bonus_enemy_slow_expiries"] == 0
    np.testing.assert_array_equal(
        terminal.info["bonus_support"]["stack_count"],
        np.asarray([[0, 0, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_allclose(
        surface.env.state["speed"],
        np.asarray([[16.0, 8.0, 8.0, 8.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        terminal.info["death_player"],
        np.asarray([[3, 2, 1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal.info["death_cause"],
        np.asarray(
            [
                [
                    vector_runtime.DEATH_CAUSE_WALL,
                    vector_runtime.DEATH_CAUSE_WALL,
                    vector_runtime.DEATH_CAUSE_WALL,
                    vector_runtime.DEATH_CAUSE_NONE,
                ]
            ],
            dtype=np.int16,
        ),
    )
    assert terminal.info["winner_ids"] == [[0]]
    assert terminal.info["loser_ids"] == [[1, 2, 3]]

    chunk = _record_chunk([catch, terminal])
    catch_record = chunk.records[0]
    terminal_record = chunk.records[1]
    assert chunk.metadata["closed_by_terminal"] is True
    assert catch_record["bonus_support"]["stack_count"] == [[0, 1, 1, 1]]
    assert catch_record["step_counters"]["bonus_enemy_slow_catches"] == 1
    assert terminal_record["terminal_or_final"] is True
    assert terminal_record["final_observation_rows"] == [0]
    np.testing.assert_array_equal(
        chunk.arrays["final_observation"][1],
        terminal.final_observation,
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_reward_map"][1],
        terminal.final_reward_map,
    )
    assert terminal_record["bonus_support"]["stack_count"] == [[0, 0, 0, 0]]
    assert terminal_record["death_player"] == [[3, 2, 1, -1]]
    assert terminal_record["death_cause"] == [
        [
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_WALL,
            vector_runtime.DEATH_CAUSE_NONE,
        ]
    ]
    assert terminal_record["death_cause_name"] == [["wall", "wall", "wall", "none"]]
    assert terminal_record["winner_ids"] == [[0]]
    assert terminal_record["loser_ids"] == [[1, 2, 3]]
    assert terminal_record["alive"] == [[True, False, False, False]]
    assert terminal_record["score"] == [[3, 0, 0, 0]]
    assert terminal_record["step_counters"]["normal_wall_deaths"] == 3
    assert terminal_record["step_counters"]["bonus_enemy_slow_expiries"] == 0


def _source_fixture_surface(
    scenario_name: str,
) -> tuple[SourceStateMultiplayerTrainerSurface, dict[str, object]]:
    scenario_path = f"scenarios/environment/{scenario_name}"
    fixture = seed_bridge.seed_fixture(scenario_path, body_capacity=8)
    state = vector_compare.array_state_from_seed(fixture)
    player_count = int(state["pos"].shape[1])
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        decision_ms=1.0,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=max(4, player_count),
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset_from_state_arrays(
        state,
        reset_seed=np.asarray([101], dtype=np.uint64),
    )
    surface = SourceStateMultiplayerTrainerSurface(env=env)
    surface.stack.reset_rows(env, np.asarray([True], dtype=bool))
    return surface, fixture


def _bonus_enemy_slow_4p_terminal_surface() -> SourceStateMultiplayerTrainerSurface:
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        decision_ms=1.0,
        body_capacity=16,
        event_capacity=64,
        timer_capacity=8,
        random_tape_capacity=32,
        event_mode="debug-event",
    )
    env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.decision_ms = 0.0
    env.state["timer_active"][0] = False
    env.state["pos"][0] = np.asarray(
        [[50.0, 50.0], [20.0, 20.0], [20.0, 40.0], [20.0, 60.0]],
        dtype=np.float64,
    )
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["heading"][0] = 0.0
    env.state["alive"][0] = True
    env.state["present"][0] = True
    env.state["printing"][0] = False
    env.state["print_manager_active"][0] = False
    env.seed_active_bonus(
        row=0,
        bonus_type="BonusEnemySlow",
        x=50.0,
        y=50.0,
        bonus_id=1,
        stack_capacity=vector_runtime.SOURCE_MAX_ACTIVE_BONUSES,
    )
    surface = SourceStateMultiplayerTrainerSurface(env=env)
    surface.stack.reset_rows(env, np.asarray([True], dtype=bool))
    return surface


def _move_bonus_enemy_targets_to_wall_death(env: VectorMultiplayerEnv) -> None:
    env.decision_ms = 400.0
    env.state["timer_active"][0] = False
    env.state["pos"][0, 0] = np.asarray([50.0, 50.0], dtype=np.float64)
    env.state["heading"][0, 0] = 0.0
    env.state["pos"][0, 1:4] = np.asarray(
        [[1.0, 20.0], [1.0, 40.0], [1.0, 60.0]],
        dtype=np.float64,
    )
    env.state["heading"][0, 1:4] = np.pi
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["print_manager_last_pos"][0] = env.state["pos"][0]


def _surface_fixture_step(
    surface: SourceStateMultiplayerTrainerSurface,
    fixture: dict[str, object],
    step_index: int,
):
    prepared_step = vector_compare.prepare_fixture_array_step(
        fixture,
        step_index=step_index,
    )
    source_moves = np.asarray(prepared_step["source_moves"], dtype=np.int8)
    actions = source_moves.astype(np.int16).reshape(1, -1) + 1
    surface.env.decision_ms = float(prepared_step["step_ms"])
    return surface.step(
        actions,
        timer_advance_ms=float(prepared_step.get("timer_advance_ms", 0.0)),
    )


def _first_active_bonus(scenario_name: str) -> dict[str, object]:
    payload = json.loads((SCENARIO_ROOT / scenario_name).read_text())
    active_bonuses = payload["initial_state"]["active_bonuses"]
    assert len(active_bonuses) == 1
    bonus = active_bonuses[0]
    assert isinstance(bonus, dict)
    return bonus


def _record_chunk(steps):
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    for index, step in enumerate(steps):
        recorder.record(step, source_ref=f"bonus-terminal-proof#{index}")
    return recorder.build_chunk()


def _assert_terminal_visual_reward_and_bonus_death_facts(terminal) -> None:
    np.testing.assert_array_equal(terminal.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal.truncated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        terminal.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(terminal.final_observation[0], terminal.observation[0])
    assert int(np.count_nonzero(terminal.final_observation[0])) > 0
    np.testing.assert_array_equal(
        terminal.final_reward_map,
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(terminal.reward, terminal.final_reward_map)
    np.testing.assert_array_equal(
        terminal.info["death_player"],
        np.asarray([[0, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal.info["death_cause"],
        np.asarray(
            [[vector_runtime.DEATH_CAUSE_WALL, vector_runtime.DEATH_CAUSE_NONE]],
            dtype=np.int16,
        ),
    )
    assert terminal.info["winner_ids"] == [[1]]
    assert terminal.info["loser_ids"] == [[0]]


def _assert_terminal_replay_facts(
    chunk,
    terminal,
    *,
    terminal_index: int,
    expected_step_counters: dict[str, int],
) -> None:
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.arange(chunk.arrays["final_observation_row_mask"].shape[0]) == terminal_index,
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_observation"][terminal_index],
        terminal.final_observation,
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_reward_map"][terminal_index],
        terminal.final_reward_map,
    )
    assert int(np.count_nonzero(chunk.arrays["final_observation"][terminal_index])) > 0
    assert chunk.metadata["closed_by_terminal"] is True

    record = chunk.records[terminal_index]
    assert record["terminal_or_final"] is True
    assert record["final_observation_rows"] == [0]
    assert record["bonus_support"]["stack_count"] == [[0, 0]]
    assert record["bonus_support"]["active_count"] == [0]
    assert record["death_player"] == [[0, -1]]
    assert record["death_cause"] == [
        [vector_runtime.DEATH_CAUSE_WALL, vector_runtime.DEATH_CAUSE_NONE]
    ]
    assert record["death_cause_name"] == [["wall", "none"]]
    assert record["winner_ids"] == [[1]]
    assert record["loser_ids"] == [[0]]
    assert record["score"] == [[0, 1]]
    assert record["alive"] == [[False, True]]
    assert record["final_observation_policy"]["metadata_only"] is False
    for key, expected in expected_step_counters.items():
        assert record["step_counters"][key] == expected
