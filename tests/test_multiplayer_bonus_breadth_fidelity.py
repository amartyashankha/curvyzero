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


def test_source_fixture_borderless_catch_expiry_preserves_trainer_replay_audit():
    scenario_name = "source_bonus_game_borderless_expiry_restore_step.json"
    surface, fixture = _source_fixture_surface(scenario_name, body_capacity=4)
    _seed_first_fixture_bonus(surface.env, scenario_name)

    catch = _surface_fixture_step(surface, fixture, step_index=0)
    _assert_seeded_bonus_public_claim(catch)
    assert catch.info["step_counters"]["bonus_game_borderless_catches"] == 1
    assert catch.info["step_counters"]["bonus_game_borderless_expiries"] == 0
    np.testing.assert_array_equal(catch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(catch.info["borderless"], np.asarray([True]))
    np.testing.assert_array_equal(
        catch.info["bonus_support"]["game_stack_count"],
        np.asarray([1], dtype=np.int16),
    )
    assert int(surface.env.state["bonus_game_stack_borderless"][0, 0]) == 1

    expiry = _surface_fixture_step(surface, fixture, step_index=1)
    _assert_seeded_bonus_public_claim(expiry)
    assert expiry.info["step_counters"]["bonus_game_borderless_catches"] == 0
    assert expiry.info["step_counters"]["bonus_game_borderless_expiries"] == 1
    assert expiry.info["step_counters"]["normal_wall_deaths"] == 0
    np.testing.assert_array_equal(expiry.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(expiry.info["borderless"], np.asarray([False]))
    np.testing.assert_array_equal(
        expiry.info["bonus_support"]["game_stack_count"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(surface.env.state["borderless"], np.asarray([False]))

    chunk = _record_chunk([catch, expiry], source_prefix=scenario_name)

    assert chunk.metadata["closed_by_terminal"] is False
    np.testing.assert_array_equal(
        chunk.arrays["done"][:, 0],
        np.asarray([False, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_observation_row_mask"][:, 0],
        np.asarray([False, False], dtype=bool),
    )
    assert chunk.records[0]["borderless"] == [True]
    assert chunk.records[1]["borderless"] == [False]
    assert chunk.records[0]["bonus_support"]["game_stack_count"] == [1]
    assert chunk.records[1]["bonus_support"]["game_stack_count"] == [0]
    assert chunk.records[0]["step_counters"]["bonus_game_borderless_catches"] == 1
    assert chunk.records[1]["step_counters"]["bonus_game_borderless_expiries"] == 1


def test_source_fixture_self_small_expiry_preserves_trainer_replay_audit():
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    surface, fixture = _source_fixture_surface(scenario_name, body_capacity=8)
    _seed_first_fixture_bonus(surface.env, scenario_name)

    catch = _surface_fixture_step(surface, fixture, step_index=0)
    _assert_seeded_bonus_public_claim(catch)
    assert catch.info["step_counters"]["bonus_self_small_catches"] == 1
    assert catch.info["step_counters"]["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(
        catch.info["bonus_catch_count_step"],
        np.asarray([[1, 0]], dtype=np.int16),
    )
    np.testing.assert_allclose(surface.env.state["radius"], np.asarray([[0.3, 0.6]]))
    np.testing.assert_array_equal(
        catch.info["bonus_support"]["stack_count"],
        np.asarray([[1, 0]], dtype=np.int16),
    )

    expiry = _surface_fixture_step(surface, fixture, step_index=1)
    _assert_seeded_bonus_public_claim(expiry)
    assert expiry.info["step_counters"]["bonus_self_small_expiries"] == 1
    assert expiry.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(expiry.done, np.asarray([False], dtype=bool))
    np.testing.assert_allclose(surface.env.state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(
        expiry.info["bonus_support"]["stack_count"],
        np.asarray([[0, 0]], dtype=np.int16),
    )
    assert int(surface.env.state["bonus_stack_type"][0, 0, 0]) == (
        vector_runtime.BONUS_TYPE_NONE
    )

    chunk = _record_chunk([catch, expiry], source_prefix=scenario_name)

    assert chunk.metadata["closed_by_terminal"] is False
    assert chunk.records[0]["bonus_catch_count_step"] == [[1, 0]]
    assert chunk.records[0]["bonus_support"]["stack_count"] == [[1, 0]]
    assert chunk.records[1]["bonus_support"]["stack_count"] == [[0, 0]]
    assert chunk.records[0]["step_counters"]["bonus_self_small_catches"] == 1
    assert chunk.records[1]["step_counters"]["bonus_self_small_expiries"] == 1


def test_source_fixture_self_small_wall_death_no_catch_preserves_terminal_replay():
    scenario_name = "source_bonus_self_small_wall_death_no_catch_step.json"
    surface, fixture = _source_fixture_surface(scenario_name, body_capacity=8)
    _seed_first_fixture_bonus(surface.env, scenario_name)

    terminal = _surface_fixture_step(surface, fixture, step_index=0)

    _assert_seeded_bonus_public_claim(terminal)
    np.testing.assert_array_equal(terminal.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal.truncated, np.asarray([False], dtype=bool))
    assert terminal.info["step_counters"]["normal_wall_deaths"] == 1
    assert terminal.info["step_counters"]["terminal_score_rows"] == 1
    assert terminal.info["step_counters"]["bonus_self_small_catches"] == 0
    assert terminal.info["step_counters"]["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(
        terminal.info["bonus_support"]["active_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal.info["bonus_support"]["stack_count"],
        np.asarray([[0, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal.info["bonus_support"]["bonus_active"],
        np.asarray([[True]], dtype=bool),
    )
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
    np.testing.assert_array_equal(
        terminal.final_observation_row_mask,
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(terminal.final_observation[0], terminal.observation[0])
    np.testing.assert_array_equal(
        terminal.final_reward_map,
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )

    chunk = _record_chunk([terminal], source_prefix=scenario_name)
    record = chunk.records[0]

    assert chunk.metadata["closed_by_terminal"] is True
    assert record["terminal_or_final"] is True
    assert record["final_observation_rows"] == [0]
    assert record["bonus_support"]["active_count"] == [1]
    assert record["bonus_support"]["stack_count"] == [[0, 0]]
    assert record["bonus_support"]["bonus_active"] == [[True]]
    assert record["death_player"] == [[0, -1]]
    assert record["death_cause"] == [
        [vector_runtime.DEATH_CAUSE_WALL, vector_runtime.DEATH_CAUSE_NONE]
    ]
    assert record["death_cause_name"] == [["wall", "none"]]
    assert record["winner_ids"] == [[1]]
    assert record["loser_ids"] == [[0]]
    assert record["score"] == [[0, 1]]
    assert record["alive"] == [[False, True]]
    assert record["step_counters"]["bonus_self_small_catches"] == 0
    np.testing.assert_array_equal(
        chunk.arrays["final_observation"][0],
        terminal.final_observation,
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_reward_map"][0],
        terminal.final_reward_map,
    )


def _source_fixture_surface(
    scenario_name: str,
    *,
    body_capacity: int,
) -> tuple[SourceStateMultiplayerTrainerSurface, dict[str, object]]:
    scenario_path = f"scenarios/environment/{scenario_name}"
    fixture = seed_bridge.seed_fixture(scenario_path, body_capacity=body_capacity)
    state = vector_compare.array_state_from_seed(fixture)
    player_count = int(state["pos"].shape[1])
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        decision_ms=1.0,
        body_capacity=body_capacity,
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


def _seed_first_fixture_bonus(
    env: VectorMultiplayerEnv,
    scenario_name: str,
) -> dict[str, object]:
    bonus = _first_active_bonus(scenario_name)
    return env.seed_active_bonus(
        row=0,
        bonus_type=str(bonus["type"]),
        x=float(bonus["x"]),
        y=float(bonus["y"]),
        bonus_id=1,
        stack_capacity=vector_runtime.SOURCE_MAX_ACTIVE_BONUSES,
    )


def _first_active_bonus(scenario_name: str) -> dict[str, object]:
    payload = json.loads((SCENARIO_ROOT / scenario_name).read_text(encoding="utf-8"))
    active_bonuses = payload["initial_state"]["active_bonuses"]
    assert len(active_bonuses) == 1
    bonus = active_bonuses[0]
    assert isinstance(bonus, dict)
    return bonus


def _surface_fixture_step(
    surface: SourceStateMultiplayerTrainerSurface,
    fixture: dict[str, object],
    *,
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


def _assert_seeded_bonus_public_claim(step) -> None:
    assert step.info["bonus_support_mode"] == "seeded"
    support = step.info["bonus_support"]
    assert support["mode"] == "seeded"
    assert support["natural_bonus_spawn"] is False
    np.testing.assert_array_equal(
        support["enabled_by_row"],
        np.asarray([True], dtype=bool),
    )


def _record_chunk(steps, *, source_prefix: str):
    recorder = SourceStateMultiplayerTrainerReplayRecorder()
    for index, step in enumerate(steps):
        recorder.record(step, source_ref=f"{source_prefix}#{index}")
    return recorder.build_chunk()
