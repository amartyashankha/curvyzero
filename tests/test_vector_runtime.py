import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from curvyzero.env import vector_runtime
from curvyzero.env.config import CurvyTronReferenceDefaults


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "environment"

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402


def _expected_source_angular_velocity_for_speed(speed: float) -> float:
    ratio = float(speed) / vector_runtime.SOURCE_AVATAR_SPEED
    return (
        ratio * vector_runtime.SOURCE_AVATAR_ANGULAR_VELOCITY_PER_MS
        + math.log(1.0 / ratio) / 1000.0
    )


def _load_lifecycle_scenario(name: str) -> dict[str, object]:
    with (SCENARIO_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expected_lifecycle_spawn_random_calls(
    payload: dict[str, object],
) -> list[tuple[int, str, int, float]]:
    expectations = payload["expectations"]
    assert isinstance(expectations, dict)
    event_order = expectations["event_order"]
    assert isinstance(event_order, list)
    calls: list[tuple[int, str, int, float]] = []
    for event in event_order:
        assert isinstance(event, dict)
        if event.get("event") != "random":
            continue
        data = event["data"]
        assert isinstance(data, dict)
        site = str(data["site"])
        if not site.startswith("spawn."):
            continue
        calls.append(
            (
                int(data["index"]),
                site,
                int(data["avatar"]) - 1,
                float(data["value"]),
            )
        )
    return calls


def _warmdown_timer_state_from_lifecycle_payload(
    payload: dict[str, object],
    *,
    random_tape_cursor: int,
    terminal_reason: int,
    death_players: list[int],
    body_capacity: int,
) -> dict[str, np.ndarray]:
    player_count = int(payload["player_count"])
    source_setup = payload["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    random_values = np.asarray(random_setup["math_random_sequence"], dtype=np.float64)
    map_size = float(CurvyTronReferenceDefaults().arena_size_for_players(player_count))
    timer_capacity = player_count

    return {
        "tick": np.asarray([1], dtype=np.int32),
        "done": np.asarray([False], dtype=bool),
        "overflow": np.asarray([False], dtype=bool),
        "started": np.asarray([True], dtype=bool),
        "in_round": np.asarray([True], dtype=bool),
        "world_active": np.asarray([True], dtype=bool),
        "world_body_count": np.asarray([body_capacity], dtype=np.int32),
        "round_done": np.asarray([True], dtype=bool),
        "warmdown_pending": np.asarray([True], dtype=bool),
        "match_done": np.asarray([False], dtype=bool),
        "round_winner": np.asarray([-1], dtype=np.int16),
        "match_winner": np.asarray([-1], dtype=np.int16),
        "round_id": np.asarray([0], dtype=np.int32),
        "terminal_reason": np.asarray([terminal_reason], dtype=np.int16),
        "draw": np.asarray([terminal_reason == vector_runtime.TERMINAL_REASON_ALL_DEAD_DRAW]),
        "winner": np.asarray([-1], dtype=np.int16),
        "timer_active": np.asarray([[True, *([False] * (timer_capacity - 1))]], dtype=bool),
        "timer_remaining_ms": np.asarray(
            [[5000.0, *([0.0] * (timer_capacity - 1))]],
            dtype=np.float64,
        ),
        "timer_kind": np.asarray(
            [[vector_runtime.TIMER_KIND_WARMDOWN_END, *([0] * (timer_capacity - 1))]],
            dtype=np.int16,
        ),
        "timer_player": np.full(
            (1, timer_capacity),
            vector_runtime.TIMER_PLAYER_NONE,
            dtype=np.int16,
        ),
        "timer_seq": np.zeros((1, timer_capacity), dtype=np.int32),
        "timer_overflow": np.asarray([False], dtype=bool),
        "present": np.ones((1, player_count), dtype=bool),
        "alive": np.zeros((1, player_count), dtype=bool),
        "pos": np.zeros((1, player_count, 2), dtype=np.float64),
        "prev_pos": np.zeros((1, player_count, 2), dtype=np.float64),
        "heading": np.zeros((1, player_count), dtype=np.float64),
        "printing": np.ones((1, player_count), dtype=bool),
        "print_manager_active": np.ones((1, player_count), dtype=bool),
        "print_manager_distance": np.ones((1, player_count), dtype=np.float64),
        "print_manager_last_pos": np.ones((1, player_count, 2), dtype=np.float64),
        "map_size": np.asarray([map_size], dtype=np.float64),
        "radius": np.full((1, player_count), 0.6, dtype=np.float64),
        "score": np.zeros((1, player_count), dtype=np.int32),
        "round_score": np.zeros((1, player_count), dtype=np.int32),
        "body_count": np.ones((1, player_count), dtype=np.int32),
        "live_body_num": np.ones((1, player_count), dtype=np.int32),
        "death_tick": np.ones((1, player_count), dtype=np.int32),
        "death_count": np.asarray([len(death_players)], dtype=np.int32),
        "death_player": np.asarray([death_players], dtype=np.int16),
        "random_tape_values": random_values.reshape(1, -1).copy(),
        "random_tape_length": np.asarray([random_values.size], dtype=np.int32),
        "random_tape_cursor": np.asarray([random_tape_cursor], dtype=np.int32),
        "random_tape_draw_count": np.asarray([random_tape_cursor], dtype=np.int32),
        "random_tape_exhausted": np.asarray([False], dtype=bool),
        "body_active": np.ones((1, body_capacity), dtype=bool),
        "body_pos": np.ones((1, body_capacity, 2), dtype=np.float64),
        "body_radius": np.ones((1, body_capacity), dtype=np.float64),
        "body_owner": np.ones((1, body_capacity), dtype=np.int16),
        "body_num": np.ones((1, body_capacity), dtype=np.int32),
        "body_insert_tick": np.ones((1, body_capacity), dtype=np.int32),
        "body_insert_kind": np.ones((1, body_capacity), dtype=np.int16),
        "body_write_cursor": np.asarray([body_capacity], dtype=np.int32),
        "body_overflow": np.asarray([True], dtype=bool),
        "visible_trail_count": np.ones((1, player_count), dtype=np.int32),
        "has_visible_trail_last": np.ones((1, player_count), dtype=bool),
        "visible_trail_last_pos": np.ones((1, player_count, 2), dtype=np.float64),
        "has_draw_cursor": np.ones((1, player_count), dtype=bool),
        "draw_cursor_pos": np.ones((1, player_count, 2), dtype=np.float64),
    }


def _step_input(
    *,
    row_count: int = 2,
    player_count: int = 2,
    step_ms: np.ndarray | None = None,
    source_moves: np.ndarray | None = None,
    print_manager_mode: np.ndarray | None = None,
) -> vector_runtime.VectorStepInput:
    return vector_runtime.VectorStepInput(
        state={"tick": np.zeros(row_count, dtype=np.int32)},
        step_ms=(np.full(row_count, 16.0, dtype=np.float64) if step_ms is None else step_ms),
        source_moves=(
            np.zeros((row_count, player_count), dtype=np.int8)
            if source_moves is None
            else source_moves
        ),
        player_count=player_count,
        print_manager_mode=print_manager_mode,
    )


def _random_tape_state() -> dict[str, np.ndarray]:
    return {
        "tick": np.zeros(2, dtype=np.int32),
        "event_count": np.asarray([2, 3], dtype=np.int16),
        "event_overflow_attempts": np.asarray([0, 1], dtype=np.int32),
        "printing": np.asarray([[True], [False]], dtype=bool),
        "print_manager_distance": np.zeros((2, 1), dtype=np.float64),
        "random_tape_values": np.asarray(
            [
                [0.25, 0.75],
                [0.50, 0.00],
            ],
            dtype=np.float64,
        ),
        "random_tape_length": np.asarray([2, 0], dtype=np.int32),
        "random_tape_cursor": np.zeros(2, dtype=np.int32),
        "random_tape_draw_count": np.zeros(2, dtype=np.int32),
        "random_tape_exhausted": np.zeros(2, dtype=bool),
    }


def _warmup_timer_state(*, body_capacity: int = 4) -> dict[str, np.ndarray]:
    state = {
        "tick": np.zeros(1, dtype=np.int32),
        "done": np.asarray([False], dtype=bool),
        "overflow": np.asarray([False], dtype=bool),
        "started": np.asarray([True], dtype=bool),
        "in_round": np.asarray([True], dtype=bool),
        "world_active": np.asarray([False], dtype=bool),
        "world_body_count": np.asarray([0], dtype=np.int32),
        "alive": np.asarray([[True, True]], dtype=bool),
        "present": np.asarray([[True, True]], dtype=bool),
        "pos": np.asarray([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float64),
        "radius": np.asarray([[0.6, 0.6]], dtype=np.float64),
        "printing": np.asarray([[False, False]], dtype=bool),
        "print_manager_active": np.asarray([[False, False]], dtype=bool),
        "print_manager_distance": np.zeros((1, 2), dtype=np.float64),
        "print_manager_last_pos": np.zeros((1, 2, 2), dtype=np.float64),
        "timer_active": np.asarray([[True, False, False, False]], dtype=bool),
        "timer_remaining_ms": np.asarray([[3000.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        "timer_kind": np.asarray(
            [[vector_runtime.TIMER_KIND_GAME_START, 0, 0, 0]],
            dtype=np.int16,
        ),
        "timer_player": np.asarray([[-1, -1, -1, -1]], dtype=np.int16),
        "timer_seq": np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        "timer_overflow": np.asarray([False], dtype=bool),
        "random_tape_values": np.asarray([[0.25, 0.50, 0.0, 0.0]], dtype=np.float64),
        "random_tape_length": np.asarray([2], dtype=np.int32),
        "random_tape_cursor": np.asarray([0], dtype=np.int32),
        "random_tape_draw_count": np.asarray([0], dtype=np.int32),
        "random_tape_exhausted": np.asarray([False], dtype=bool),
        "body_active": np.zeros((1, body_capacity), dtype=bool),
        "body_pos": np.zeros((1, body_capacity, 2), dtype=np.float64),
        "body_radius": np.zeros((1, body_capacity), dtype=np.float64),
        "body_owner": np.full((1, body_capacity), -1, dtype=np.int16),
        "body_num": np.full((1, body_capacity), -1, dtype=np.int32),
        "body_insert_tick": np.full((1, body_capacity), -1, dtype=np.int32),
        "body_insert_kind": np.full((1, body_capacity), -1, dtype=np.int16),
        "body_write_cursor": np.asarray([0], dtype=np.int32),
        "body_count": np.asarray([[0, 0]], dtype=np.int32),
        "body_overflow": np.asarray([False], dtype=bool),
        "visible_trail_count": np.zeros((1, 2), dtype=np.int32),
        "has_visible_trail_last": np.asarray([[False, False]], dtype=bool),
        "visible_trail_last_pos": np.zeros((1, 2, 2), dtype=np.float64),
        "has_draw_cursor": np.asarray([[False, False]], dtype=bool),
        "draw_cursor_pos": np.zeros((1, 2, 2), dtype=np.float64),
    }
    return state


def _prepared_step_batch(prepared_step: dict[str, object]) -> dict[str, np.ndarray | int]:
    return {
        "player_count": prepared_step["player_count"],
        "step_ms": np.asarray([prepared_step["step_ms"]], dtype=np.float64),
        "source_moves": np.asarray([prepared_step["source_moves"]], dtype=np.int8),
        "print_manager_mode": np.asarray(
            [prepared_step.get("print_manager_mode", "none")],
            dtype=object,
        ),
        "timer_advance_ms": np.asarray(
            [prepared_step.get("timer_advance_ms", 0.0)],
            dtype=np.float64,
        ),
    }


def _bonus_type_metadata_state(alive_rows: list[list[bool]]) -> dict[str, np.ndarray]:
    alive = np.asarray(alive_rows, dtype=bool)
    return {
        "tick": np.zeros(alive.shape[0], dtype=np.int32),
        "alive": alive,
        "present": np.ones_like(alive, dtype=bool),
    }


def _bonus_spawn_state(
    *,
    bonus_capacity: int = 2,
    body_capacity: int = 4,
    event_capacity: int = 4,
    player_count: int = 2,
) -> dict[str, np.ndarray]:
    state = _bonus_type_metadata_state([[True] * player_count])
    state.update(
        {
            "map_size": np.asarray([88.0], dtype=np.float64),
            "radius": np.full((1, player_count), 0.6, dtype=np.float64),
            "world_active": np.asarray([False], dtype=bool),
            "world_body_count": np.asarray([0], dtype=np.int32),
            "body_active": np.zeros((1, body_capacity), dtype=bool),
            "body_pos": np.zeros((1, body_capacity, 2), dtype=np.float64),
            "body_radius": np.zeros((1, body_capacity), dtype=np.float64),
            "body_owner": np.full((1, body_capacity), -1, dtype=np.int16),
            "body_num": np.full((1, body_capacity), -1, dtype=np.int32),
            "body_insert_tick": np.full((1, body_capacity), -1, dtype=np.int32),
            "body_insert_kind": np.full((1, body_capacity), -1, dtype=np.int16),
            "body_write_cursor": np.zeros(1, dtype=np.int32),
            "body_count": np.zeros((1, player_count), dtype=np.int32),
            "body_overflow": np.zeros(1, dtype=bool),
            "bonus_world_active": np.ones(1, dtype=bool),
            "bonus_active": np.zeros((1, bonus_capacity), dtype=bool),
            "bonus_type": np.full(
                (1, bonus_capacity),
                vector_runtime.BONUS_TYPE_NONE,
                dtype=np.int16,
            ),
            "bonus_id": np.full((1, bonus_capacity), -1, dtype=np.int32),
            "bonus_pos": np.zeros((1, bonus_capacity, 2), dtype=np.float64),
            "bonus_radius": np.zeros((1, bonus_capacity), dtype=np.float64),
            "bonus_count": np.zeros(1, dtype=np.int32),
            "bonus_world_body_count": np.zeros(1, dtype=np.int32),
            "event_count": np.zeros(1, dtype=np.int16),
            "event_mask": np.zeros((1, event_capacity), dtype=bool),
            "event_type": np.full(
                (1, event_capacity),
                vector_runtime.EVENT_NONE,
                dtype=np.int16,
            ),
            "event_player": np.full((1, event_capacity), -1, dtype=np.int16),
            "event_other": np.full((1, event_capacity), -1, dtype=np.int16),
            "event_bool": np.full((1, event_capacity), -1, dtype=np.int16),
            "event_value_i": np.zeros((1, event_capacity, 2), dtype=np.int32),
            "event_value_f": np.zeros((1, event_capacity, 2), dtype=np.float64),
            "event_overflow": np.zeros(1, dtype=bool),
            "event_overflow_attempts": np.zeros(1, dtype=np.int32),
        }
    )
    return state


def _seed_spawn_bonus(
    state: dict[str, np.ndarray],
    *,
    slot: int,
    bonus_id: int,
    x: float,
    y: float,
    bonus_type: int = vector_runtime.BONUS_TYPE_SELF_SMALL,
) -> None:
    state["bonus_active"][0, slot] = True
    state["bonus_type"][0, slot] = bonus_type
    state["bonus_id"][0, slot] = bonus_id
    state["bonus_pos"][0, slot] = (x, y)
    state["bonus_radius"][0, slot] = vector_runtime.SOURCE_BONUS_RADIUS
    state["bonus_count"][0] = int(state["bonus_active"][0].sum())
    state["bonus_world_body_count"][0] = int(state["bonus_count"][0])


def _runtime_fixture_state(
    path: str,
    *,
    body_capacity: int = 8,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    fixture = seed_bridge.seed_fixture(path, body_capacity=body_capacity)
    state = vector_compare.array_state_from_seed(fixture)
    state["terminated"] = np.asarray([False], dtype=bool)
    state["truncated"] = np.asarray([False], dtype=bool)
    state["reset_pending"] = np.asarray([False], dtype=bool)
    state["terminal_reason"] = np.asarray(
        [vector_compare.TERMINAL_REASON_NONE],
        dtype=np.int16,
    )
    state["draw"] = np.asarray([False], dtype=bool)
    state["winner"] = np.asarray([-1], dtype=np.int16)
    return fixture, state


def _expected_runtime_counters(counters: dict[str, int]) -> dict[str, int]:
    expected = {name: 0 for name in vector_runtime.STEP_COUNTER_NAMES}
    expected.update(counters)
    return expected


def _add_forced_bonus_self_small_arrays(
    state: dict[str, np.ndarray],
    scenario_name: str,
    *,
    bonus_type_code: int = vector_runtime.BONUS_TYPE_SELF_SMALL,
    stack_capacity: int = 2,
) -> None:
    scenario = _load_lifecycle_scenario(scenario_name)
    initial_state = scenario["initial_state"]
    assert isinstance(initial_state, dict)
    active_bonuses = initial_state["active_bonuses"]
    assert isinstance(active_bonuses, list)
    assert len(active_bonuses) == 1
    bonus = active_bonuses[0]
    assert isinstance(bonus, dict)
    assert bonus["type"] == "BonusSelfSmall"
    row_count, player_count = state["radius"].shape

    state["bonus_world_active"] = np.ones(row_count, dtype=bool)
    state["bonus_active"] = np.ones((row_count, 1), dtype=bool)
    state["bonus_type"] = np.asarray([[bonus_type_code]], dtype=np.int16)
    state["bonus_id"] = np.asarray([[1]], dtype=np.int32)
    state["bonus_pos"] = np.asarray(
        [[[float(bonus["x"]), float(bonus["y"])]]],
        dtype=np.float64,
    )
    state["bonus_radius"] = np.asarray([[3.0]], dtype=np.float64)
    state["bonus_count"] = np.asarray([1], dtype=np.int32)
    state["bonus_world_body_count"] = np.asarray([1], dtype=np.int32)
    state["base_radius"] = state["radius"].copy()
    if "speed" in state:
        state["base_speed"] = state["speed"].copy()
    if "angular_velocity_per_ms" in state:
        state["base_angular_velocity_per_ms"] = state["angular_velocity_per_ms"].copy()
    state["inverse"] = np.zeros((row_count, player_count), dtype=bool)
    state["base_inverse"] = state["inverse"].copy()
    state["invincible"] = np.zeros((row_count, player_count), dtype=bool)
    state["base_invincible"] = state["invincible"].copy()
    state["avatar_color"] = np.tile(
        np.arange(player_count, dtype=np.int16),
        (row_count, 1),
    )
    state["base_avatar_color"] = state["avatar_color"].copy()
    state["radius_power"] = np.zeros((row_count, player_count), dtype=np.int16)
    state["bonus_stack_count"] = np.zeros((row_count, player_count), dtype=np.int16)
    state["bonus_stack_id"] = np.full(
        (row_count, player_count, stack_capacity),
        -1,
        dtype=np.int32,
    )
    state["bonus_stack_type"] = np.full(
        (row_count, player_count, stack_capacity),
        vector_runtime.BONUS_TYPE_NONE,
        dtype=np.int16,
    )
    state["bonus_stack_duration_ms"] = np.zeros(
        (row_count, player_count, stack_capacity),
        dtype=np.int32,
    )
    state["bonus_stack_radius_power"] = np.zeros(
        (row_count, player_count, stack_capacity),
        dtype=np.int16,
    )
    state["bonus_stack_velocity_delta"] = np.zeros(
        (row_count, player_count, stack_capacity),
        dtype=np.float64,
    )
    state["bonus_stack_inverse_delta"] = np.zeros(
        (row_count, player_count, stack_capacity),
        dtype=np.int16,
    )
    state["bonus_stack_angular_velocity_per_ms"] = np.zeros(
        (row_count, player_count, stack_capacity),
        dtype=np.float64,
    )
    state["bonus_stack_invincible_delta"] = np.zeros(
        (row_count, player_count, stack_capacity),
        dtype=np.int16,
    )
    state["bonus_stack_printing_delta"] = np.zeros(
        (row_count, player_count, stack_capacity),
        dtype=np.int16,
    )
    state["bonus_stack_color"] = np.full(
        (row_count, player_count, stack_capacity),
        -1,
        dtype=np.int16,
    )


def _add_forced_bonus_game_clear_arrays(
    state: dict[str, np.ndarray],
    scenario_name: str,
) -> None:
    scenario = _load_lifecycle_scenario(scenario_name)
    initial_state = scenario["initial_state"]
    assert isinstance(initial_state, dict)
    active_bonuses = initial_state["active_bonuses"]
    assert isinstance(active_bonuses, list)
    assert len(active_bonuses) == 1
    bonus = active_bonuses[0]
    assert isinstance(bonus, dict)
    assert bonus["type"] == "BonusGameClear"
    row_count = state["radius"].shape[0]

    state["bonus_world_active"] = np.ones(row_count, dtype=bool)
    state["bonus_active"] = np.ones((row_count, 1), dtype=bool)
    state["bonus_type"] = np.asarray(
        [[vector_runtime.BONUS_TYPE_GAME_CLEAR]],
        dtype=np.int16,
    )
    state["bonus_id"] = np.asarray([[1]], dtype=np.int32)
    state["bonus_pos"] = np.asarray(
        [[[float(bonus["x"]), float(bonus["y"])]]],
        dtype=np.float64,
    )
    state["bonus_radius"] = np.asarray([[3.0]], dtype=np.float64)
    state["bonus_count"] = np.asarray([1], dtype=np.int32)
    state["bonus_world_body_count"] = np.asarray([1], dtype=np.int32)


def _add_forced_bonus_game_borderless_arrays(
    state: dict[str, np.ndarray],
    scenario_name: str,
    *,
    stack_capacity: int = 2,
) -> None:
    scenario = _load_lifecycle_scenario(scenario_name)
    initial_state = scenario["initial_state"]
    assert isinstance(initial_state, dict)
    active_bonuses = initial_state["active_bonuses"]
    assert isinstance(active_bonuses, list)
    assert len(active_bonuses) == 1
    bonus = active_bonuses[0]
    assert isinstance(bonus, dict)
    assert bonus["type"] == "BonusGameBorderless"
    row_count = state["radius"].shape[0]

    state["bonus_world_active"] = np.ones(row_count, dtype=bool)
    state["bonus_active"] = np.ones((row_count, 1), dtype=bool)
    state["bonus_type"] = np.asarray(
        [[vector_runtime.BONUS_TYPE_GAME_BORDERLESS]],
        dtype=np.int16,
    )
    state["bonus_id"] = np.asarray([[1]], dtype=np.int32)
    state["bonus_pos"] = np.asarray(
        [[[float(bonus["x"]), float(bonus["y"])]]],
        dtype=np.float64,
    )
    state["bonus_radius"] = np.asarray([[3.0]], dtype=np.float64)
    state["bonus_count"] = np.asarray([1], dtype=np.int32)
    state["bonus_world_body_count"] = np.asarray([1], dtype=np.int32)
    _add_empty_bonus_game_borderless_stack_arrays(
        state,
        stack_capacity=stack_capacity,
    )


def _add_empty_bonus_game_borderless_stack_arrays(
    state: dict[str, np.ndarray],
    *,
    stack_capacity: int = 2,
) -> None:
    row_count = state["borderless"].shape[0]
    state["bonus_game_stack_count"] = np.zeros(row_count, dtype=np.int16)
    state["bonus_game_stack_id"] = np.full(
        (row_count, stack_capacity),
        -1,
        dtype=np.int32,
    )
    state["bonus_game_stack_type"] = np.full(
        (row_count, stack_capacity),
        vector_runtime.BONUS_TYPE_NONE,
        dtype=np.int16,
    )
    state["bonus_game_stack_duration_ms"] = np.zeros(
        (row_count, stack_capacity),
        dtype=np.int32,
    )
    state["bonus_game_stack_borderless"] = np.zeros(
        (row_count, stack_capacity),
        dtype=np.int16,
    )


def _step_runtime_fixture(
    fixture: dict[str, object],
    state: dict[str, np.ndarray],
    *,
    step_index: int,
) -> dict[str, int]:
    prepared_step = vector_compare.prepare_fixture_array_step(
        fixture,
        step_index=step_index,
    )
    return vector_runtime.step_many(
        vector_runtime.VectorStepInput.from_mapping(
            state,
            _prepared_step_batch(prepared_step),
        ),
    )


def _slice_runtime_state_to_player_count(
    state: dict[str, np.ndarray],
    *,
    player_count: int,
) -> None:
    player_axis_1_arrays = (
        "heading",
        "alive",
        "death_tick",
        "score",
        "round_score",
        "printing",
        "radius",
        "speed",
        "angular_velocity_per_ms",
        "trail_latency",
        "body_count",
        "live_body_num",
        "visible_trail_count",
        "has_visible_trail_last",
        "has_draw_cursor",
        "print_manager_active",
        "print_manager_distance",
        "present",
        "inverse",
        "invincible",
        "radius_power",
        "avatar_color",
        "base_avatar_color",
        "base_radius",
        "base_speed",
        "base_angular_velocity_per_ms",
        "base_inverse",
        "base_invincible",
    )
    player_axis_2_arrays = (
        "pos",
        "prev_pos",
        "visible_trail_last_pos",
        "draw_cursor_pos",
        "print_manager_last_pos",
    )
    for name in player_axis_1_arrays:
        if name in state:
            state[name] = state[name][:, :player_count].copy()
    for name in player_axis_2_arrays:
        if name in state:
            state[name] = state[name][:, :player_count, :].copy()

    state["death_count"] = np.zeros(1, dtype=np.int32)
    state["death_player"] = np.full((1, player_count), -1, dtype=np.int16)
    state["death_cause"] = np.full(
        (1, player_count),
        vector_runtime.DEATH_CAUSE_NONE,
        dtype=np.int16,
    )
    state["death_hit_owner"] = np.full((1, player_count), -1, dtype=np.int16)


def _step_runtime_fixture_with_player_count(
    fixture: dict[str, object],
    state: dict[str, np.ndarray],
    *,
    step_index: int,
    player_count: int,
) -> dict[str, int]:
    prepared_step = vector_compare.prepare_fixture_array_step(
        fixture,
        step_index=step_index,
    )
    prepared_batch = _prepared_step_batch(prepared_step)
    prepared_batch["player_count"] = player_count
    prepared_batch["source_moves"] = prepared_batch["source_moves"][:, :player_count]
    return vector_runtime.step_many(
        vector_runtime.VectorStepInput.from_mapping(state, prepared_batch),
    )


def _step_runtime_timer_only(
    state: dict[str, np.ndarray],
    *,
    timer_advance_ms: float,
    player_count: int = 2,
) -> dict[str, int]:
    return vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=state,
            step_ms=np.zeros(1, dtype=np.float64),
            source_moves=np.zeros((1, player_count), dtype=np.int8),
            player_count=player_count,
            timer_advance_ms=np.asarray([timer_advance_ms], dtype=np.float64),
            event_mode=vector_runtime.EVENT_MODE_DEBUG,
        ),
    )


def _print_manager_step_state(
    row_count: int,
    *,
    body_capacity: int = 8,
) -> dict[str, np.ndarray]:
    pos = np.repeat(
        np.asarray([[[10.0, 10.0]]], dtype=np.float64),
        row_count,
        axis=0,
    )
    return {
        "tick": np.zeros(row_count, dtype=np.int32),
        "done": np.zeros(row_count, dtype=bool),
        "overflow": np.zeros(row_count, dtype=bool),
        "world_body_count": np.zeros(row_count, dtype=np.int32),
        "alive": np.ones((row_count, 1), dtype=bool),
        "death_tick": np.full((row_count, 1), -1, dtype=np.int32),
        "pos": pos.copy(),
        "prev_pos": np.zeros((row_count, 1, 2), dtype=np.float64),
        "heading": np.zeros((row_count, 1), dtype=np.float64),
        "angular_velocity_per_ms": np.zeros((row_count, 1), dtype=np.float64),
        "speed": np.full((row_count, 1), 2.0, dtype=np.float64),
        "live_body_num": np.zeros((row_count, 1), dtype=np.int32),
        "trail_latency": np.zeros((row_count, 1), dtype=np.int32),
        "borderless": np.zeros(row_count, dtype=bool),
        "map_size": np.full(row_count, 100.0, dtype=np.float64),
        "radius": np.full((row_count, 1), 0.5, dtype=np.float64),
        "printing": np.zeros((row_count, 1), dtype=bool),
        "print_manager_active": np.ones((row_count, 1), dtype=bool),
        "print_manager_distance": np.ones((row_count, 1), dtype=np.float64),
        "print_manager_last_pos": pos.copy(),
        "body_active": np.zeros((row_count, body_capacity), dtype=bool),
        "body_pos": np.zeros((row_count, body_capacity, 2), dtype=np.float64),
        "body_radius": np.zeros((row_count, body_capacity), dtype=np.float64),
        "body_owner": np.full((row_count, body_capacity), -1, dtype=np.int16),
        "body_num": np.full((row_count, body_capacity), -1, dtype=np.int32),
        "body_insert_tick": np.full((row_count, body_capacity), -1, dtype=np.int32),
        "body_insert_kind": np.full((row_count, body_capacity), -1, dtype=np.int16),
        "body_write_cursor": np.zeros(row_count, dtype=np.int32),
        "body_count": np.zeros((row_count, 1), dtype=np.int32),
        "body_overflow": np.zeros(row_count, dtype=bool),
        "visible_trail_count": np.zeros((row_count, 1), dtype=np.int32),
        "has_visible_trail_last": np.zeros((row_count, 1), dtype=bool),
        "visible_trail_last_pos": np.zeros((row_count, 1, 2), dtype=np.float64),
        "has_draw_cursor": np.ones((row_count, 1), dtype=bool),
        "draw_cursor_pos": pos.copy(),
        "score": np.zeros((row_count, 1), dtype=np.int32),
        "round_score": np.zeros((row_count, 1), dtype=np.int32),
        "event_count": np.zeros(row_count, dtype=np.int16),
        "event_overflow_attempts": np.zeros(row_count, dtype=np.int32),
        "random_tape_values": np.full((row_count, 2), 0.25, dtype=np.float64),
        "random_tape_length": np.full(row_count, 2, dtype=np.int32),
        "random_tape_cursor": np.zeros(row_count, dtype=np.int32),
        "random_tape_draw_count": np.zeros(row_count, dtype=np.int32),
        "random_tape_exhausted": np.zeros(row_count, dtype=bool),
    }


def test_step_many_rejects_step_ms_shape_that_does_not_match_rows():
    step_input = _step_input(step_ms=np.asarray([16.0], dtype=np.float64))

    with pytest.raises(vector_runtime.VectorRuntimeError, match=r"step_ms .* shape \[B\]"):
        vector_runtime.step_many(step_input)


def test_step_many_rejects_source_moves_shape_that_does_not_match_rows_and_players():
    step_input = _step_input(source_moves=np.zeros((2, 3), dtype=np.int8))

    with pytest.raises(
        vector_runtime.VectorRuntimeError,
        match=r"source_moves .* shape \[B,P\]",
    ):
        vector_runtime.step_many(step_input)


def test_step_many_rejects_print_manager_mode_shape_that_does_not_match_rows():
    step_input = _step_input(print_manager_mode=np.asarray(["none"], dtype=object))

    with pytest.raises(
        vector_runtime.VectorRuntimeError,
        match=r"print_manager_mode must have shape \[B\]",
    ):
        vector_runtime.step_many(step_input)


def test_prepare_step_input_normalizes_mapping_values_and_default_print_modes():
    state = {"tick": np.asarray([0, 1], dtype=np.int32)}
    prepared_batch = {
        "player_count": 2,
        "step_ms": [16, 33],
        "source_moves": [[0, 1], [-1, 0]],
    }

    step_input = vector_runtime.prepare_step_input(
        state,
        prepared_batch,
        event_mode=vector_runtime.EVENT_MODE_NONE,
    )

    assert step_input.state is state
    assert step_input.player_count == 2
    assert step_input.event_mode == vector_runtime.EVENT_MODE_NONE
    assert step_input.step_ms.dtype == np.float64
    assert step_input.source_moves.dtype == np.int8
    np.testing.assert_array_equal(
        step_input.print_manager_mode,
        np.asarray(["none", "none"], dtype=object),
    )


def test_vector_step_input_from_mapping_rejects_unknown_print_mode():
    state = {"tick": np.asarray([0], dtype=np.int32)}
    prepared_batch = {
        "player_count": 1,
        "step_ms": [16.0],
        "source_moves": [[0]],
        "print_manager_mode": ["mystery"],
    }

    with pytest.raises(vector_runtime.VectorRuntimeError, match="known modes"):
        vector_runtime.VectorStepInput.from_mapping(state, prepared_batch)


def test_step_many_rejects_unknown_print_mode():
    step_input = _step_input(
        print_manager_mode=np.asarray(["natural_toggle", "mystery"], dtype=object),
    )

    with pytest.raises(vector_runtime.VectorRuntimeError, match="known modes"):
        vector_runtime.step_many(step_input)


def test_empty_step_counters_returns_complete_zeroed_counter_set():
    counters = vector_runtime.empty_step_counters()

    assert set(counters) == set(vector_runtime.VectorStepCounters.__annotations__)
    assert all(value == 0 for value in counters.values())


def test_empty_step_counters_can_omit_random_tape_for_legacy_callers():
    counters = vector_runtime.empty_step_counters(include_random_tape=False)

    assert "random_tape_draws" not in counters
    assert "random_tape_exhaustions" not in counters
    assert counters["movement_updates"] == 0


def test_phase_timer_helpers_tolerate_none_and_empty_timer_dict():
    vector_runtime._timer_add(None, "phase_sec", 0.0)

    phase_timers: dict[str, float] = {}
    started = vector_runtime._timer_start(phase_timers)
    vector_runtime._timer_add(phase_timers, "phase_sec", started)

    assert phase_timers["phase_sec"] >= 0.0


def test_print_manager_random_distances_consume_row_local_tape_and_default_draws():
    state = _random_tape_state()

    vector_runtime.assign_print_manager_random_distances(
        state,
        player=0,
        mask=np.asarray([True, True], dtype=bool),
    )

    assert state["print_manager_distance"][0, 0] == pytest.approx(28.5)
    assert state["print_manager_distance"][1, 0] == pytest.approx(
        vector_runtime.PRINT_MANAGER_RANDOM_HALF_HOLE_DISTANCE,
    )
    np.testing.assert_array_equal(state["random_tape_cursor"], np.asarray([1, 0]))
    np.testing.assert_array_equal(state["random_tape_draw_count"], np.asarray([1, 1]))
    np.testing.assert_array_equal(state["random_tape_exhausted"], np.asarray([False, False]))


def test_next_print_manager_random_distance_raises_and_marks_finite_tape_exhaustion():
    state = _random_tape_state()
    state["random_tape_length"][0] = 1
    state["random_tape_cursor"][0] = 1
    state["random_tape_draw_count"][0] = 1

    with pytest.raises(
        vector_runtime.VectorRuntimeError,
        match=r"row 0 Math\.random tape exhausted after 1 calls",
    ):
        vector_runtime.next_print_manager_random_distance(
            state,
            row=0,
            printing=True,
        )

    np.testing.assert_array_equal(state["random_tape_exhausted"], [True, False])
    np.testing.assert_array_equal(state["random_tape_cursor"], [1, 0])
    np.testing.assert_array_equal(state["random_tape_draw_count"], [1, 0])


def test_natural_toggle_toggles_printing_and_updates_distance_like_toggle():
    state = _print_manager_step_state(4)
    state["print_manager_distance"][:, 0] = [1.0, 1.0, 5.0, 5.0]

    counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=state,
            step_ms=np.full(4, 1000.0, dtype=np.float64),
            source_moves=np.zeros((4, 1), dtype=np.int8),
            player_count=1,
            print_manager_mode=np.asarray(
                ["natural_toggle", "toggle", "natural_toggle", "toggle"],
                dtype=object,
            ),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        ),
    )

    np.testing.assert_array_equal(state["printing"][:, 0], [True, True, False, False])
    np.testing.assert_allclose(state["print_manager_distance"][:2, 0], [28.5, 28.5])
    np.testing.assert_allclose(state["print_manager_distance"][2:, 0], [3.0, 3.0])
    np.testing.assert_allclose(state["print_manager_last_pos"][:, 0], state["pos"][:, 0])
    np.testing.assert_array_equal(state["random_tape_cursor"], [1, 1, 0, 0])
    np.testing.assert_array_equal(state["body_count"][:, 0], [1, 1, 0, 0])
    np.testing.assert_array_equal(
        state["body_insert_kind"][:2, 0],
        [
            vector_runtime.BODY_KIND_IMPORTANT,
            vector_runtime.BODY_KIND_IMPORTANT,
        ],
    )
    np.testing.assert_array_equal(
        state["printing"][[0, 2], 0],
        state["printing"][[1, 3], 0],
    )
    np.testing.assert_allclose(
        state["print_manager_distance"][[0, 2], 0],
        state["print_manager_distance"][[1, 3], 0],
    )
    assert counters["print_manager_toggle_updates"] == 2
    assert counters["print_manager_toggle_rows_unhandled"] == 2
    assert counters["print_manager_death_stops"] == 0


@pytest.mark.parametrize("death_kind", ["wall", "body"])
def test_natural_toggle_stops_and_clears_on_death_like_death_stop(death_kind):
    state = _print_manager_step_state(2)
    state["printing"][:, 0] = True
    state["print_manager_distance"][:, 0] = 7.0
    state["visible_trail_count"][:, 0] = 3
    state["has_visible_trail_last"][:, 0] = True
    state["visible_trail_last_pos"][:, 0] = state["pos"][:, 0]

    if death_kind == "wall":
        state["pos"][:, 0, 0] = 0.25
        state["speed"][:, 0] = 0.0
        state["print_manager_last_pos"][:, 0] = state["pos"][:, 0]
        state["draw_cursor_pos"][:, 0] = state["pos"][:, 0]
    else:
        state["body_active"][:, 0] = True
        state["body_pos"][:, 0] = [12.0, 10.0]
        state["body_radius"][:, 0] = 1.0
        state["body_num"][:, 0] = 0
        state["body_write_cursor"][:] = 1
        state["world_body_count"][:] = 1
        state["draw_cursor_pos"][:, 0] = [12.0, 10.0]

    counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=state,
            step_ms=np.full(2, 1000.0, dtype=np.float64),
            source_moves=np.zeros((2, 1), dtype=np.int8),
            player_count=1,
            print_manager_mode=np.asarray(
                ["natural_toggle", "death_stop"],
                dtype=object,
            ),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        ),
    )

    np.testing.assert_array_equal(state["print_manager_active"][:, 0], [False, False])
    np.testing.assert_array_equal(state["printing"][:, 0], [False, False])
    np.testing.assert_allclose(state["print_manager_distance"][:, 0], [0.0, 0.0])
    np.testing.assert_allclose(state["print_manager_last_pos"][:, 0], 0.0)
    np.testing.assert_array_equal(state["visible_trail_count"][:, 0], [0, 0])
    np.testing.assert_array_equal(state["has_visible_trail_last"][:, 0], [False, False])
    np.testing.assert_array_equal(state["has_draw_cursor"][:, 0], [False, False])
    np.testing.assert_array_equal(state["random_tape_cursor"], [1, 1])
    np.testing.assert_array_equal(state["random_tape_draw_count"], [1, 1])
    np.testing.assert_array_equal(state["body_count"][:, 0], [2, 2])
    assert counters["print_manager_death_stops"] == 2
    assert counters["print_manager_death_stop_points"] == 2
    assert counters["print_manager_death_stop_visual_clears"] == 2
    assert counters["print_manager_toggle_updates"] == 0


@pytest.mark.parametrize(
    ("body_owner", "body_num", "expected_cause"),
    [
        (0, -1, vector_runtime.DEATH_CAUSE_OWN_TRAIL),
        (1, 0, vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL),
    ],
)
def test_step_many_body_death_records_trail_owner_cause_in_no_event_mode(
    body_owner,
    body_num,
    expected_cause,
):
    pos = np.asarray([[[10.0, 10.0], [60.0, 60.0]]], dtype=np.float64)
    state = {
        "tick": np.zeros(1, dtype=np.int32),
        "done": np.zeros(1, dtype=bool),
        "overflow": np.zeros(1, dtype=bool),
        "world_body_count": np.asarray([1], dtype=np.int32),
        "alive": np.asarray([[True, True]], dtype=bool),
        "death_tick": np.full((1, 2), -1, dtype=np.int32),
        "death_count": np.zeros(1, dtype=np.int32),
        "death_player": np.full((1, 2), -1, dtype=np.int16),
        "death_cause": np.full(
            (1, 2),
            vector_runtime.DEATH_CAUSE_NONE,
            dtype=np.int16,
        ),
        "death_hit_owner": np.full((1, 2), -1, dtype=np.int16),
        "pos": pos.copy(),
        "prev_pos": pos.copy(),
        "heading": np.zeros((1, 2), dtype=np.float64),
        "angular_velocity_per_ms": np.zeros((1, 2), dtype=np.float64),
        "speed": np.asarray([[2.0, 0.0]], dtype=np.float64),
        "live_body_num": np.zeros((1, 2), dtype=np.int32),
        "trail_latency": np.zeros((1, 2), dtype=np.int32),
        "borderless": np.zeros(1, dtype=bool),
        "map_size": np.full(1, 100.0, dtype=np.float64),
        "radius": np.full((1, 2), 0.5, dtype=np.float64),
        "printing": np.zeros((1, 2), dtype=bool),
        "print_manager_active": np.zeros((1, 2), dtype=bool),
        "print_manager_distance": np.zeros((1, 2), dtype=np.float64),
        "print_manager_last_pos": pos.copy(),
        "body_active": np.asarray([[True, False, False]], dtype=bool),
        "body_pos": np.asarray([[[12.0, 10.0], [0.0, 0.0], [0.0, 0.0]]]),
        "body_radius": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
        "body_owner": np.asarray([[body_owner, -1, -1]], dtype=np.int16),
        "body_num": np.asarray([[body_num, -1, -1]], dtype=np.int32),
        "body_insert_tick": np.asarray([[0, -1, -1]], dtype=np.int32),
        "body_insert_kind": np.asarray(
            [[vector_runtime.BODY_KIND_NORMAL, -1, -1]],
            dtype=np.int16,
        ),
        "body_write_cursor": np.asarray([1], dtype=np.int32),
        "body_count": np.zeros((1, 2), dtype=np.int32),
        "body_overflow": np.zeros(1, dtype=bool),
        "visible_trail_count": np.zeros((1, 2), dtype=np.int32),
        "has_visible_trail_last": np.zeros((1, 2), dtype=bool),
        "visible_trail_last_pos": np.zeros((1, 2, 2), dtype=np.float64),
        "has_draw_cursor": np.zeros((1, 2), dtype=bool),
        "draw_cursor_pos": np.zeros((1, 2, 2), dtype=np.float64),
        "score": np.zeros((1, 2), dtype=np.int32),
        "round_score": np.zeros((1, 2), dtype=np.int32),
        "event_count": np.zeros(1, dtype=np.int16),
        "event_overflow_attempts": np.zeros(1, dtype=np.int32),
        "random_tape_values": np.zeros((1, 1), dtype=np.float64),
        "random_tape_length": np.ones(1, dtype=np.int32),
        "random_tape_cursor": np.zeros(1, dtype=np.int32),
        "random_tape_draw_count": np.zeros(1, dtype=np.int32),
        "random_tape_exhausted": np.zeros(1, dtype=bool),
    }

    counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=state,
            step_ms=np.asarray([1000.0], dtype=np.float64),
            source_moves=np.zeros((1, 2), dtype=np.int8),
            player_count=2,
            event_mode=vector_runtime.EVENT_MODE_NONE,
        ),
    )

    assert counters["body_hits"] == 1
    np.testing.assert_array_equal(state["death_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        state["death_player"],
        np.asarray([[0, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray([[expected_cause, vector_runtime.DEATH_CAUSE_NONE]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["death_hit_owner"],
        np.asarray([[body_owner, -1]], dtype=np.int16),
    )


def test_finalize_random_tape_counters_accounts_draw_delta_and_exhaustions():
    state = _random_tape_state()
    before = vector_runtime.snapshot_random_tape_counters(state)
    counters = vector_runtime.empty_step_counters()
    state["random_tape_draw_count"][:] = [2, 3]
    state["random_tape_exhausted"][:] = [False, True]

    vector_runtime.finalize_random_tape_counters(counters, state, before)

    assert counters["random_tape_draws"] == 5
    assert counters["random_tape_exhaustions"] == 1


def test_finalize_step_counters_normalizes_state_derived_counts():
    state = _random_tape_state()
    before = vector_runtime.snapshot_random_tape_counters(state)
    counters = vector_runtime.empty_step_counters()
    counters["movement_updates"] = 7
    counters["events_emitted"] = 99
    counters["random_tape_draws"] = 99
    state["random_tape_draw_count"][:] = [1, 4]

    finalized = vector_runtime.finalize_step_counters(
        counters,
        state,
        random_tape_draw_count_before=before,
    )

    assert finalized["movement_updates"] == 7
    assert finalized["events_emitted"] == 5
    assert finalized["event_overflow_attempts"] == 1
    assert finalized["random_tape_draws"] == 5
    assert finalized["random_tape_exhaustions"] == 0


def test_advance_warmup_1v1_no_bonus_timers_starts_game_and_print_managers():
    state = _warmup_timer_state()

    info = vector_runtime.advance_warmup_1v1_no_bonus_timers(state, 6000.0)

    assert info["schema"] == vector_runtime.WARMUP_TIMER_ADVANCE_INFO_SCHEMA_ID
    assert info["surface"] == vector_runtime.WARMUP_TIMER_ADVANCE_SURFACE
    assert info["pre_step_timer_advances"] == 1
    assert info["pre_step_timer_fires"] == 3
    assert info["game_start_fires"] == 1
    assert info["scheduled_print_manager_start_count"] == 2
    assert info["print_manager_delayed_start_fires"] == 2
    assert info["print_manager_delayed_start_points"] == 2
    assert info["body_overflow_attempts"] == 0
    assert info["random_tape_draws"] == 2
    np.testing.assert_array_equal(info["game_start_rows"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        info["scheduled_print_manager_start_slots"],
        np.asarray([0, 1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        info["scheduled_print_manager_start_players"],
        np.asarray([1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        info["print_manager_start_players"],
        np.asarray([1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        info["random_draw_count_delta"],
        np.asarray([2], dtype=np.int32),
    )

    assert bool(state["world_active"][0]) is True
    assert int(state["world_body_count"][0]) == 2
    np.testing.assert_array_equal(state["timer_active"][0], np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(
        state["timer_kind"][0],
        np.zeros(4, dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["timer_player"][0],
        np.full(4, vector_runtime.TIMER_PLAYER_NONE, dtype=np.int16),
    )

    np.testing.assert_array_equal(state["print_manager_active"][0], [True, True])
    np.testing.assert_array_equal(state["printing"][0], [True, True])
    np.testing.assert_allclose(state["print_manager_last_pos"][0], state["pos"][0])
    assert state["print_manager_distance"][0, 1] == pytest.approx(28.5)
    assert state["print_manager_distance"][0, 0] == pytest.approx(39.0)
    np.testing.assert_array_equal(state["random_tape_cursor"], np.asarray([2]))
    np.testing.assert_array_equal(state["random_tape_draw_count"], np.asarray([2]))

    np.testing.assert_array_equal(state["body_active"][0, :2], [True, True])
    np.testing.assert_allclose(state["body_pos"][0, :2], [[30.0, 40.0], [10.0, 20.0]])
    np.testing.assert_allclose(state["body_radius"][0, :2], [0.6, 0.6])
    np.testing.assert_array_equal(state["body_owner"][0, :2], [1, 0])
    np.testing.assert_array_equal(state["body_num"][0, :2], [0, 0])
    np.testing.assert_array_equal(
        state["body_insert_kind"][0, :2],
        [vector_runtime.BODY_KIND_IMPORTANT, vector_runtime.BODY_KIND_IMPORTANT],
    )
    assert int(state["body_write_cursor"][0]) == 2
    np.testing.assert_array_equal(state["body_count"][0], [1, 1])
    np.testing.assert_array_equal(state["visible_trail_count"][0], [1, 1])
    np.testing.assert_array_equal(state["has_visible_trail_last"][0], [True, True])
    np.testing.assert_allclose(state["visible_trail_last_pos"][0], state["pos"][0])


def test_advance_warmup_1v1_no_bonus_timers_leaves_not_due_timer_pending():
    state = _warmup_timer_state()

    info = vector_runtime.advance_warmup_1v1_no_bonus_timers(state, 1000.0)

    assert info["pre_step_timer_advances"] == 1
    assert info["pre_step_timer_fires"] == 0
    assert bool(state["world_active"][0]) is False
    assert bool(state["timer_active"][0, 0]) is True
    assert state["timer_remaining_ms"][0, 0] == pytest.approx(2000.0)
    np.testing.assert_array_equal(state["print_manager_active"][0], [False, False])


def test_advance_warmdown_no_bonus_timers_spawns_3p_next_round_from_fixture_tape():
    payload = _load_lifecycle_scenario("source_lifecycle_spawn_rng_3p_next_round.json")
    player_count = int(payload["player_count"])
    source_setup = payload["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    random_values = np.asarray(random_setup["math_random_sequence"], dtype=np.float64)
    map_size = float(CurvyTronReferenceDefaults().arena_size_for_players(player_count))
    body_capacity = 8

    state = {
        "tick": np.asarray([1], dtype=np.int32),
        "done": np.asarray([False], dtype=bool),
        "overflow": np.asarray([False], dtype=bool),
        "started": np.asarray([True], dtype=bool),
        "in_round": np.asarray([True], dtype=bool),
        "world_active": np.asarray([True], dtype=bool),
        "world_body_count": np.asarray([7], dtype=np.int32),
        "round_done": np.asarray([True], dtype=bool),
        "warmdown_pending": np.asarray([True], dtype=bool),
        "match_done": np.asarray([False], dtype=bool),
        "round_winner": np.asarray([-1], dtype=np.int16),
        "match_winner": np.asarray([-1], dtype=np.int16),
        "round_id": np.asarray([0], dtype=np.int32),
        "terminal_reason": np.asarray(
            [vector_runtime.TERMINAL_REASON_ALL_DEAD_DRAW],
            dtype=np.int16,
        ),
        "draw": np.asarray([True], dtype=bool),
        "winner": np.asarray([-1], dtype=np.int16),
        "timer_active": np.asarray([[True, False, False]], dtype=bool),
        "timer_remaining_ms": np.asarray([[5000.0, 0.0, 0.0]], dtype=np.float64),
        "timer_kind": np.asarray(
            [[vector_runtime.TIMER_KIND_WARMDOWN_END, 0, 0]],
            dtype=np.int16,
        ),
        "timer_player": np.full((1, 3), vector_runtime.TIMER_PLAYER_NONE, dtype=np.int16),
        "timer_seq": np.asarray([[0, 0, 0]], dtype=np.int32),
        "timer_overflow": np.asarray([False], dtype=bool),
        "present": np.ones((1, player_count), dtype=bool),
        "alive": np.zeros((1, player_count), dtype=bool),
        "pos": np.zeros((1, player_count, 2), dtype=np.float64),
        "prev_pos": np.zeros((1, player_count, 2), dtype=np.float64),
        "heading": np.zeros((1, player_count), dtype=np.float64),
        "printing": np.ones((1, player_count), dtype=bool),
        "print_manager_active": np.ones((1, player_count), dtype=bool),
        "print_manager_distance": np.ones((1, player_count), dtype=np.float64),
        "print_manager_last_pos": np.ones((1, player_count, 2), dtype=np.float64),
        "map_size": np.asarray([map_size], dtype=np.float64),
        "radius": np.full((1, player_count), 0.6, dtype=np.float64),
        "score": np.zeros((1, player_count), dtype=np.int32),
        "round_score": np.zeros((1, player_count), dtype=np.int32),
        "body_count": np.ones((1, player_count), dtype=np.int32),
        "live_body_num": np.ones((1, player_count), dtype=np.int32),
        "death_tick": np.asarray([[1, 1, 1]], dtype=np.int32),
        "death_count": np.asarray([3], dtype=np.int32),
        "death_player": np.asarray([[2, 1, 0]], dtype=np.int16),
        "random_tape_values": random_values.reshape(1, -1).copy(),
        "random_tape_length": np.asarray([random_values.size], dtype=np.int32),
        "random_tape_cursor": np.asarray([15], dtype=np.int32),
        "random_tape_draw_count": np.asarray([15], dtype=np.int32),
        "random_tape_exhausted": np.asarray([False], dtype=bool),
        "body_active": np.ones((1, body_capacity), dtype=bool),
        "body_pos": np.ones((1, body_capacity, 2), dtype=np.float64),
        "body_radius": np.ones((1, body_capacity), dtype=np.float64),
        "body_owner": np.ones((1, body_capacity), dtype=np.int16),
        "body_num": np.ones((1, body_capacity), dtype=np.int32),
        "body_insert_tick": np.ones((1, body_capacity), dtype=np.int32),
        "body_insert_kind": np.ones((1, body_capacity), dtype=np.int16),
        "body_write_cursor": np.asarray([body_capacity], dtype=np.int32),
        "body_overflow": np.asarray([True], dtype=bool),
        "visible_trail_count": np.ones((1, player_count), dtype=np.int32),
        "has_visible_trail_last": np.ones((1, player_count), dtype=bool),
        "visible_trail_last_pos": np.ones((1, player_count, 2), dtype=np.float64),
        "has_draw_cursor": np.ones((1, player_count), dtype=bool),
        "draw_cursor_pos": np.ones((1, player_count, 2), dtype=np.float64),
    }

    info = vector_runtime.advance_warmdown_no_bonus_timers(
        state,
        5000.0,
        player_count=player_count,
    )

    assert info["schema"] == vector_runtime.WARMDOWN_TIMER_ADVANCE_NO_BONUS_INFO_SCHEMA_ID
    assert info["surface"] == vector_runtime.WARMDOWN_TIMER_ADVANCE_NO_BONUS_SURFACE
    assert info["warmdown_end_fires"] == 1
    assert info["game_stop_count"] == 1
    assert info["round_new_count"] == 1
    assert info["next_round_spawn_count"] == 1
    assert info["scheduled_game_start_count"] == 1
    np.testing.assert_array_equal(info["random_draw_count_delta"], np.asarray([9]))
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in info["spawn_infos"][0]["random_calls"]
    ] == [
        (15, "spawn.position_x", 2, 0.25),
        (16, "spawn.position_y", 2, 0.5),
        (17, "spawn.angle_attempt_0", 2, 0.25),
        (18, "spawn.position_x", 1, 0.5),
        (19, "spawn.position_y", 1, 0.5),
        (20, "spawn.angle_attempt_0", 1, 0.5),
        (21, "spawn.position_x", 0, 0.75),
        (22, "spawn.position_y", 0, 0.5),
        (23, "spawn.angle_attempt_0", 0, 0.75),
    ]

    np.testing.assert_array_equal(state["alive"], np.asarray([[True, True, True]]))
    np.testing.assert_allclose(
        state["pos"][0],
        [[68.575, 47.5], [47.5, 47.5], [26.425, 47.5]],
    )
    np.testing.assert_allclose(
        state["heading"][0],
        [1.5 * np.pi, np.pi, 0.5 * np.pi],
    )
    np.testing.assert_array_equal(state["printing"], np.zeros((1, 3), dtype=bool))
    np.testing.assert_array_equal(
        state["print_manager_active"],
        np.zeros((1, 3), dtype=bool),
    )
    assert bool(state["world_active"][0]) is False
    assert bool(state["round_done"][0]) is False
    assert bool(state["warmdown_pending"][0]) is False
    assert int(state["round_id"][0]) == 1
    assert int(state["death_count"][0]) == 0
    np.testing.assert_array_equal(state["death_player"], np.asarray([[-1, -1, -1]]))
    assert int(state["random_tape_cursor"][0]) == 24
    assert int(state["random_tape_draw_count"][0]) == 24
    np.testing.assert_array_equal(
        state["timer_kind"],
        np.asarray([[vector_runtime.TIMER_KIND_GAME_START, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(state["timer_active"], np.asarray([[True, False, False]]))
    np.testing.assert_allclose(state["timer_remaining_ms"], [[0.0, 0.0, 0.0]])
    assert int(state["body_write_cursor"][0]) == 0
    np.testing.assert_array_equal(state["body_active"], np.zeros((1, body_capacity), dtype=bool))


def test_advance_warmdown_no_bonus_timers_spawns_4p_next_round_from_fixture_tape():
    payload = _load_lifecycle_scenario("source_lifecycle_spawn_rng_4p_next_round.json")
    player_count = int(payload["player_count"])
    body_capacity = 10
    state = _warmdown_timer_state_from_lifecycle_payload(
        payload,
        random_tape_cursor=20,
        terminal_reason=vector_runtime.TERMINAL_REASON_ALL_DEAD_DRAW,
        death_players=[3, 2, 1, 0],
        body_capacity=body_capacity,
    )

    info = vector_runtime.advance_warmdown_no_bonus_timers(
        state,
        5000.0,
        player_count=player_count,
    )

    assert info["schema"] == vector_runtime.WARMDOWN_TIMER_ADVANCE_NO_BONUS_INFO_SCHEMA_ID
    assert info["surface"] == vector_runtime.WARMDOWN_TIMER_ADVANCE_NO_BONUS_SURFACE
    assert info["player_count"] == 4
    assert info["warmdown_end_fires"] == 1
    assert info["game_stop_count"] == 1
    assert info["round_new_count"] == 1
    assert info["next_round_spawn_count"] == 1
    assert info["scheduled_game_start_count"] == 1
    np.testing.assert_array_equal(info["random_draw_count_delta"], np.asarray([12]))
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in info["spawn_infos"][0]["random_calls"]
    ] == _expected_lifecycle_spawn_random_calls(payload)[12:]

    np.testing.assert_array_equal(state["alive"], np.asarray([[True, True, True, True]]))
    np.testing.assert_allclose(
        state["pos"][0],
        [[77.41, 50.5], [59.47, 50.5], [41.53, 50.5], [23.59, 50.5]],
    )
    np.testing.assert_allclose(
        state["heading"][0],
        [1.5 * np.pi, np.pi, 0.1, 0.5 * np.pi],
    )
    np.testing.assert_array_equal(state["printing"], np.zeros((1, 4), dtype=bool))
    np.testing.assert_array_equal(
        state["print_manager_active"],
        np.zeros((1, 4), dtype=bool),
    )
    assert bool(state["world_active"][0]) is False
    assert bool(state["round_done"][0]) is False
    assert bool(state["warmdown_pending"][0]) is False
    assert int(state["round_id"][0]) == 1
    assert int(state["death_count"][0]) == 0
    np.testing.assert_array_equal(state["death_player"], np.asarray([[-1, -1, -1, -1]]))
    assert int(state["random_tape_cursor"][0]) == 32
    assert int(state["random_tape_draw_count"][0]) == 32
    np.testing.assert_array_equal(
        state["timer_kind"],
        np.asarray([[vector_runtime.TIMER_KIND_GAME_START, 0, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["timer_active"],
        np.asarray([[True, False, False, False]]),
    )
    np.testing.assert_allclose(state["timer_remaining_ms"], [[0.0, 0.0, 0.0, 0.0]])
    assert int(state["body_write_cursor"][0]) == 0
    np.testing.assert_array_equal(state["body_active"], np.zeros((1, body_capacity), dtype=bool))


def test_advance_player_movement_updates_only_live_rows_with_source_math():
    state = {
        "tick": np.asarray([0, 1], dtype=np.int32),
        "pos": np.asarray(
            [
                [[0.0, 0.0], [10.0, 20.0]],
                [[5.0, 7.0], [30.0, 40.0]],
            ],
            dtype=np.float64,
        ),
        "prev_pos": np.full((2, 2, 2), -1.0, dtype=np.float64),
        "heading": np.asarray(
            [
                [0.0, 0.25],
                [1.0, 1.25],
            ],
            dtype=np.float64,
        ),
        "angular_velocity_per_ms": np.asarray(
            [
                [0.0, 0.01],
                [0.0, 0.50],
            ],
            dtype=np.float64,
        ),
        "speed": np.asarray(
            [
                [0.0, 20.0],
                [0.0, 99.0],
            ],
            dtype=np.float64,
        ),
        "live_body_num": np.full((2, 2), -1, dtype=np.int32),
        "body_count": np.asarray([[3, 4], [5, 6]], dtype=np.int32),
    }
    original_inactive_pos = state["pos"][1, 1].copy()
    original_inactive_heading = float(state["heading"][1, 1])
    step_ms = np.asarray([100.0, 200.0], dtype=np.float64)
    moves = np.asarray([[0, -1], [0, 1]], dtype=np.int8)

    updates = vector_runtime.advance_player_movement(
        state,
        player=1,
        live_mask=np.asarray([True, False], dtype=bool),
        step_ms=step_ms,
        source_moves=moves,
    )

    expected_heading = 0.25 - 0.01 * 100.0
    expected_distance = 20.0 * 100.0 / 1000.0
    assert updates == 1
    assert state["heading"][0, 1] == pytest.approx(expected_heading)
    np.testing.assert_allclose(
        state["pos"][0, 1],
        [
            10.0 + np.cos(expected_heading) * expected_distance,
            20.0 + np.sin(expected_heading) * expected_distance,
        ],
    )
    np.testing.assert_allclose(state["prev_pos"][0, 1], [10.0, 20.0])
    assert state["live_body_num"][0, 1] == 4
    np.testing.assert_allclose(state["pos"][1, 1], original_inactive_pos)
    assert state["heading"][1, 1] == pytest.approx(original_inactive_heading)
    np.testing.assert_allclose(state["prev_pos"][1, 1], [-1.0, -1.0])
    assert state["live_body_num"][1, 1] == -1


def test_advance_player_movement_inverts_turn_direction_when_inverse_is_present():
    state = {
        "tick": np.asarray([0], dtype=np.int32),
        "pos": np.asarray([[[10.0, 20.0]]], dtype=np.float64),
        "prev_pos": np.full((1, 1, 2), -1.0, dtype=np.float64),
        "heading": np.asarray([[0.25]], dtype=np.float64),
        "angular_velocity_per_ms": np.asarray([[0.01]], dtype=np.float64),
        "speed": np.asarray([[20.0]], dtype=np.float64),
        "inverse": np.asarray([[True]], dtype=bool),
        "live_body_num": np.full((1, 1), -1, dtype=np.int32),
        "body_count": np.asarray([[4]], dtype=np.int32),
    }

    vector_runtime.advance_player_movement(
        state,
        player=0,
        live_mask=np.asarray([True], dtype=bool),
        step_ms=np.asarray([100.0], dtype=np.float64),
        source_moves=np.asarray([[1]], dtype=np.int8),
    )

    expected_heading = 0.25 - 0.01 * 100.0
    assert state["heading"][0, 0] == pytest.approx(expected_heading)


def test_apply_borderless_wrap_preserves_source_axis_priority_and_strict_edges():
    state = {
        "tick": np.zeros(8, dtype=np.int32),
        "borderless": np.asarray(
            [True, True, True, True, True, True, False, True],
            dtype=bool,
        ),
        "map_size": np.full(8, 100.0, dtype=np.float64),
        "pos": np.asarray(
            [
                [[-1.0, -2.0]],  # x wraps first; y is left untouched.
                [[101.0, 50.0]],
                [[25.0, -0.5]],
                [[25.0, 100.5]],
                [[0.0, 50.0]],
                [[100.0, 50.0]],
                [[-3.0, 50.0]],
                [[-4.0, 50.0]],
            ],
            dtype=np.float64,
        ),
    }

    wrap_count, wrapped = vector_runtime.apply_borderless_wrap(
        state,
        player=0,
        live_mask=np.asarray([True, True, True, True, True, True, True, False]),
    )

    assert wrap_count == 4
    np.testing.assert_array_equal(
        wrapped,
        np.asarray([True, True, True, True, False, False, False, False]),
    )
    np.testing.assert_allclose(
        state["pos"][:, 0],
        np.asarray(
            [
                [100.0, -2.0],
                [0.0, 50.0],
                [25.0, 100.0],
                [25.0, 0.0],
                [0.0, 50.0],
                [100.0, 50.0],
                [-3.0, 50.0],
                [-4.0, 50.0],
            ],
            dtype=np.float64,
        ),
    )


def test_apply_borderless_wrap_rejects_live_mask_shape_mismatch():
    state = {
        "tick": np.zeros(2, dtype=np.int32),
        "borderless": np.ones(2, dtype=bool),
        "map_size": np.full(2, 100.0, dtype=np.float64),
        "pos": np.zeros((2, 1, 2), dtype=np.float64),
    }

    with pytest.raises(vector_runtime.VectorRuntimeError, match=r"live_mask .* shape \[B\]"):
        vector_runtime.apply_borderless_wrap(
            state,
            player=0,
            live_mask=np.asarray([True], dtype=bool),
        )


def test_normal_wall_hit_mask_uses_source_strict_radius_edges():
    state = {
        "tick": np.zeros(10, dtype=np.int32),
        "borderless": np.asarray(
            [False, False, False, False, False, False, False, False, True, False],
            dtype=bool,
        ),
        "map_size": np.full(10, 100.0, dtype=np.float64),
        "radius": np.full((10, 1), 5.0, dtype=np.float64),
        "pos": np.asarray(
            [
                [[5.0, 50.0]],
                [[4.999, 50.0]],
                [[95.0, 50.0]],
                [[95.001, 50.0]],
                [[50.0, 5.0]],
                [[50.0, 4.999]],
                [[50.0, 95.0]],
                [[50.0, 95.001]],
                [[1.0, 50.0]],
                [[1.0, 50.0]],
            ],
            dtype=np.float64,
        ),
    }

    hit_mask = vector_runtime.normal_wall_hit_mask(
        state,
        player=0,
        live_mask=np.asarray([True, True, True, True, True, True, True, True, True, False]),
    )

    np.testing.assert_array_equal(
        hit_mask,
        np.asarray([False, True, False, True, False, True, False, True, False, False]),
    )


def test_normal_wall_hit_mask_rejects_live_mask_shape_mismatch():
    state = {
        "tick": np.zeros(2, dtype=np.int32),
        "borderless": np.zeros(2, dtype=bool),
        "map_size": np.full(2, 100.0, dtype=np.float64),
        "radius": np.ones((2, 1), dtype=np.float64),
        "pos": np.zeros((2, 1, 2), dtype=np.float64),
    }

    with pytest.raises(vector_runtime.VectorRuntimeError, match=r"live_mask .* shape \[B\]"):
        vector_runtime.normal_wall_hit_mask(
            state,
            player=0,
            live_mask=np.asarray([True], dtype=bool),
        )


def test_normal_wall_hit_mask_rejects_radius_shape_mismatch():
    state = {
        "tick": np.zeros(2, dtype=np.int32),
        "borderless": np.zeros(2, dtype=bool),
        "map_size": np.full(2, 100.0, dtype=np.float64),
        "radius": np.ones(2, dtype=np.float64),
        "pos": np.zeros((2, 1, 2), dtype=np.float64),
    }

    with pytest.raises(vector_runtime.VectorRuntimeError, match=r"radius .* shape \[B,P\]"):
        vector_runtime.normal_wall_hit_mask(
            state,
            player=0,
            live_mask=np.ones(2, dtype=bool),
        )


def test_step_many_1v1_wall_death_marks_terminal_lifecycle_and_preserves_events():
    fixture = seed_bridge.seed_fixture(
        "scenarios/environment/source_normal_wall_death_step.json",
        body_capacity=4,
    )
    initial_state = vector_compare.array_state_from_seed(fixture)
    expected_state = vector_compare.copy_array_state(initial_state)
    actual_state = vector_compare.copy_array_state(initial_state)
    actual_state["terminated"] = np.asarray([False], dtype=bool)
    actual_state["truncated"] = np.asarray([False], dtype=bool)
    actual_state["reset_pending"] = np.asarray([False], dtype=bool)
    actual_state["terminal_reason"] = np.asarray(
        [vector_compare.TERMINAL_REASON_NONE],
        dtype=np.int16,
    )
    actual_state["draw"] = np.asarray([True], dtype=bool)
    actual_state["winner"] = np.asarray([-1], dtype=np.int16)
    player_count = actual_state["alive"].shape[1]
    actual_state["death_count"] = np.zeros(1, dtype=np.int32)
    actual_state["death_player"] = np.full((1, player_count), -1, dtype=np.int16)
    actual_state["death_cause"] = np.full_like(
        actual_state["death_player"],
        vector_runtime.DEATH_CAUSE_NONE,
    )
    actual_state["death_hit_owner"] = np.full_like(actual_state["death_player"], -1)
    prepared_step = vector_compare.prepare_fixture_array_step(fixture, step_index=0)

    expected_counters = vector_compare.step_prepared_arrays(expected_state, prepared_step)
    actual_counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput.from_mapping(
            actual_state,
            _prepared_step_batch(prepared_step),
        ),
    )

    assert actual_counters == _expected_runtime_counters(expected_counters)
    assert actual_counters["normal_wall_deaths"] == 1
    assert actual_counters["terminal_score_rows"] == 1
    np.testing.assert_array_equal(actual_state["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        actual_state["terminated"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        actual_state["truncated"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        actual_state["reset_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        actual_state["terminal_reason"],
        np.asarray([vector_runtime.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(actual_state["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(actual_state["winner"], np.asarray([1], dtype=np.int16))
    np.testing.assert_array_equal(actual_state["score"], expected_state["score"])
    np.testing.assert_array_equal(actual_state["round_score"], expected_state["round_score"])
    np.testing.assert_array_equal(actual_state["death_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        actual_state["death_cause"],
        np.asarray([[vector_runtime.DEATH_CAUSE_WALL, vector_runtime.DEATH_CAUSE_NONE]]),
    )
    np.testing.assert_array_equal(
        actual_state["death_hit_owner"],
        np.asarray([[-1, -1]], dtype=np.int16),
    )

    for name in (
        "event_count",
        "event_mask",
        "event_type",
        "event_player",
        "event_other",
        "event_bool",
        "event_value_i",
        "event_value_f",
        "event_overflow",
        "event_overflow_attempts",
    ):
        np.testing.assert_array_equal(actual_state[name], expected_state[name])


def test_step_many_terminal_lifecycle_tolerates_missing_extended_arrays():
    fixture = seed_bridge.seed_fixture(
        "scenarios/environment/source_normal_wall_death_step.json",
        body_capacity=4,
    )
    state = vector_compare.array_state_from_seed(fixture)
    prepared_step = vector_compare.prepare_fixture_array_step(fixture, step_index=0)

    counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput.from_mapping(
            state,
            _prepared_step_batch(prepared_step),
        ),
    )

    assert counters["terminal_score_rows"] == 1
    np.testing.assert_array_equal(state["done"], np.asarray([True], dtype=bool))
    assert "terminated" not in state
    assert "reset_pending" not in state
    assert "terminal_reason" not in state


def test_step_many_3p_no_bonus_wall_survivor_scoring_matches_promoted_fixture():
    fixture, state = _runtime_fixture_state(
        "scenarios/environment/source_normal_wall_3p_two_die_one_survivor_step.json",
    )

    counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert counters["movement_updates"] == 3
    assert counters["normal_wall_deaths"] == 2
    assert counters["terminal_score_rows"] == 1
    np.testing.assert_array_equal(state["alive"], np.asarray([[True, False, False]]))
    np.testing.assert_array_equal(state["score"], np.asarray([[2, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(
        state["round_score"],
        np.asarray([[0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(state["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(state["terminated"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        state["terminal_reason"],
        np.asarray([vector_runtime.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(state["draw"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(state["winner"], np.asarray([0], dtype=np.int16))

    event_count = int(state["event_count"][0])
    np.testing.assert_array_equal(
        state["event_type"][0, :event_count],
        np.asarray(
            [
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_POINT,
                vector_runtime.EVENT_DIE,
                vector_runtime.EVENT_SCORE_ROUND,
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_POINT,
                vector_runtime.EVENT_DIE,
                vector_runtime.EVENT_SCORE_ROUND,
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_SCORE_ROUND,
                vector_runtime.EVENT_SCORE,
                vector_runtime.EVENT_SCORE,
                vector_runtime.EVENT_SCORE,
                vector_runtime.EVENT_ROUND_END,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["event_player"][0, [2, 6, 9, 10, 11, 12]],
        np.asarray([2, 1, 0, 2, 1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["event_value_i"][0, [3, 7, 9, 10, 11, 12], 1],
        np.asarray([0, 0, 2, 0, 0, 2], dtype=np.int32),
    )
    assert int(state["event_other"][0, 13]) == 0


def test_step_many_4p_no_bonus_ordered_wall_deaths_keep_source_scores():
    fixture, state = _runtime_fixture_state(
        "scenarios/environment/source_normal_wall_4p_ordered_deaths_survivor_score.json",
    )

    first = _step_runtime_fixture(fixture, state, step_index=0)
    assert first["normal_wall_deaths"] == 1
    assert first["terminal_score_rows"] == 0
    np.testing.assert_array_equal(state["alive"], np.asarray([[True, False, True, True]]))
    np.testing.assert_array_equal(
        state["round_score"],
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
    )

    second = _step_runtime_fixture(fixture, state, step_index=1)
    assert second["normal_wall_deaths"] == 1
    assert second["terminal_score_rows"] == 0
    np.testing.assert_array_equal(state["alive"], np.asarray([[True, False, False, True]]))
    np.testing.assert_array_equal(
        state["round_score"],
        np.asarray([[0, 0, 1, 0]], dtype=np.int32),
    )

    third = _step_runtime_fixture(fixture, state, step_index=2)
    assert third["normal_wall_deaths"] == 1
    assert third["terminal_score_rows"] == 1
    np.testing.assert_array_equal(state["alive"], np.asarray([[True, False, False, False]]))
    np.testing.assert_array_equal(
        state["score"],
        np.asarray([[3, 0, 1, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        state["round_score"],
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(state["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(state["terminated"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        state["terminal_reason"],
        np.asarray([vector_runtime.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(state["winner"], np.asarray([0], dtype=np.int16))

    event_count = int(state["event_count"][0])
    np.testing.assert_array_equal(
        state["event_type"][0, :event_count],
        np.asarray(
            [
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_POINT,
                vector_runtime.EVENT_DIE,
                vector_runtime.EVENT_SCORE_ROUND,
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_SCORE_ROUND,
                vector_runtime.EVENT_SCORE,
                vector_runtime.EVENT_SCORE,
                vector_runtime.EVENT_SCORE,
                vector_runtime.EVENT_SCORE,
                vector_runtime.EVENT_ROUND_END,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["event_player"][0, [2, 3, 5, 6, 7, 8, 9]],
        np.asarray([3, 3, 0, 3, 2, 1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["event_value_i"][0, [3, 5, 6, 7, 8, 9], 1],
        np.asarray([2, 3, 2, 1, 0, 3], dtype=np.int32),
    )
    assert int(state["event_other"][0, 10]) == 0


def test_bonus_type_selection_metadata_pins_reduced_game_clear_edges():
    state = _bonus_type_metadata_state(
        [
            [True, True, False, False],
            [True, True, False, False],
        ],
    )

    info = vector_runtime.bonus_type_selection_metadata(
        state,
        np.asarray([0.945, 0.965], dtype=np.float64),
        player_count=4,
    )

    assert info["schema"] == vector_runtime.BONUS_TYPE_SELECTION_METADATA_SCHEMA_ID
    assert info["surface"] == vector_runtime.BONUS_TYPE_SELECTION_METADATA_SURFACE
    np.testing.assert_array_equal(info["eligible_rows"], np.asarray([True, True]))
    np.testing.assert_allclose(info["game_clear_probability"], [0.5, 0.5])
    np.testing.assert_allclose(info["total_weight"], [10.7, 10.7])
    np.testing.assert_allclose(info["weighted_draw"], [10.1115, 10.3255])
    np.testing.assert_array_equal(
        info["selected_type_code"],
        np.asarray(
            [
                vector_runtime.BONUS_TYPE_ALL_COLOR,
                vector_runtime.BONUS_TYPE_GAME_CLEAR,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        info["selected_type_name"],
        np.asarray(["BonusAllColor", "BonusGameClear"], dtype=object),
    )


def test_bonus_type_selection_metadata_pins_full_probability_game_clear_edge():
    state = _bonus_type_metadata_state(
        [
            [True, True, False, True],
            [True, True, False, False],
        ],
    )

    info = vector_runtime.bonus_type_selection_metadata(
        state,
        np.asarray([0.93, 0.93], dtype=np.float64),
        player_count=4,
    )

    np.testing.assert_allclose(info["game_clear_probability"], [1.0, 0.5])
    np.testing.assert_allclose(info["total_weight"], [11.2, 10.7])
    np.testing.assert_allclose(info["weighted_draw"], [10.416, 9.951])
    np.testing.assert_array_equal(
        info["selected_type_code"],
        np.asarray(
            [
                vector_runtime.BONUS_TYPE_GAME_CLEAR,
                vector_runtime.BONUS_TYPE_ALL_COLOR,
            ],
            dtype=np.int16,
        ),
    )


def test_bonus_type_selection_metadata_is_metadata_only_and_requires_present_array():
    state = _bonus_type_metadata_state([[True, True, False, False]])
    state["bonus_active"] = np.asarray([[False]], dtype=bool)
    state["bonus_count"] = np.asarray([0], dtype=np.int32)
    original_bonus_active = state["bonus_active"].copy()
    original_bonus_count = state["bonus_count"].copy()

    vector_runtime.bonus_type_selection_metadata(state, [0.965], player_count=4)

    np.testing.assert_array_equal(state["bonus_active"], original_bonus_active)
    np.testing.assert_array_equal(state["bonus_count"], original_bonus_count)

    with pytest.raises(
        vector_runtime.VectorRuntimeError,
        match="state is missing required step array 'present'",
    ):
        vector_runtime.bonus_type_selection_metadata(
            {
                "tick": np.zeros(1, dtype=np.int32),
                "alive": np.asarray([[True, True, False, False]], dtype=bool),
            },
            [0.965],
            player_count=4,
        )


def test_bonus_spawn_cap_metadata_pins_source_cap_gate_for_type_selection():
    cap_scenario = _load_lifecycle_scenario("source_bonus_spawn_cap_twenty_step.json")
    cap_initial_state = cap_scenario["initial_state"]
    assert isinstance(cap_initial_state, dict)
    active_bonuses = cap_initial_state["active_bonuses"]
    assert isinstance(active_bonuses, list)
    assert len(active_bonuses) == vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    cap_random = cap_scenario["source_setup"]["random"]
    assert isinstance(cap_random, dict)
    assert [entry["label"] for entry in cap_random["math_random_sequence"]] == [
        "bonus.start_delay",
        "bonus.next_delay_after_pop",
    ]

    one_type_scenario = _load_lifecycle_scenario(
        "source_bonus_spawn_type_position_rng_step.json"
    )
    one_type_random = one_type_scenario["source_setup"]["random"]
    assert isinstance(one_type_random, dict)
    type_draw = one_type_random["math_random_sequence"][2]
    assert type_draw == {"label": "bonus.type.BonusSelfSmall", "value": 0.2}

    state = _bonus_type_metadata_state(
        [
            [True, True],
            [True, True],
            [True, True],
        ],
    )
    state["bonus_count"] = np.asarray([20, 19, 20], dtype=np.int32)

    cap_info = vector_runtime.bonus_spawn_cap_metadata(
        state,
        eligible_rows=np.asarray([True, True, False], dtype=bool),
    )

    assert cap_info["schema"] == vector_runtime.BONUS_SPAWN_CAP_METADATA_SCHEMA_ID
    assert cap_info["surface"] == vector_runtime.BONUS_SPAWN_CAP_METADATA_SURFACE
    assert cap_info["source_max_active_bonuses"] == 20
    np.testing.assert_array_equal(
        cap_info["eligible_rows"],
        np.asarray([True, True, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        cap_info["capped_rows"],
        np.asarray([True, False, False], dtype=bool),
    )
    np.testing.assert_array_equal(
        cap_info["selection_rows"],
        np.asarray([False, True, False], dtype=bool),
    )

    selection_info = vector_runtime.bonus_type_selection_metadata(
        state,
        np.asarray([0.2, 0.2, 0.2], dtype=np.float64),
        player_count=2,
        eligible_rows=cap_info["selection_rows"],
        enabled_type_codes=np.asarray(
            [vector_runtime.BONUS_TYPE_SELF_SMALL],
            dtype=np.int16,
        ),
    )

    np.testing.assert_array_equal(
        selection_info["selected_type_code"],
        np.asarray(
            [
                vector_runtime.BONUS_TYPE_NONE,
                vector_runtime.BONUS_TYPE_SELF_SMALL,
                vector_runtime.BONUS_TYPE_NONE,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        selection_info["selected_type_name"],
        np.asarray(["None", "BonusSelfSmall", "None"], dtype=object),
    )


def test_bonus_spawn_due_rows_spawns_type_position_from_source_fixture():
    scenario = _load_lifecycle_scenario("source_bonus_spawn_type_position_rng_step.json")
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    sequence = random_setup["math_random_sequence"]
    assert [entry["label"] for entry in sequence] == [
        "bonus.start_delay",
        "bonus.next_delay_after_pop",
        "bonus.type.BonusSelfSmall",
        "bonus.position.x",
        "bonus.position.y",
    ]

    state = _bonus_spawn_state()
    info = vector_runtime.bonus_spawn_due_rows(
        state,
        player_count=2,
        due_rows=np.asarray([True], dtype=bool),
        type_draws=np.asarray([sequence[2]["value"]], dtype=np.float64),
        position_draws=np.asarray(
            [[[sequence[3]["value"], sequence[4]["value"]]]],
            dtype=np.float64,
        ),
        enabled_type_codes=np.asarray(
            [vector_runtime.BONUS_TYPE_SELF_SMALL],
            dtype=np.int16,
        ),
        events_enabled=True,
    )

    assert info["schema"] == vector_runtime.BONUS_SPAWN_DUE_ROWS_SCHEMA_ID
    assert info["surface"] == vector_runtime.BONUS_SPAWN_DUE_ROWS_SURFACE
    np.testing.assert_array_equal(info["spawn_rows"], np.asarray([True]))
    np.testing.assert_array_equal(
        info["selected_type_code"],
        np.asarray([vector_runtime.BONUS_TYPE_SELF_SMALL], dtype=np.int16),
    )
    np.testing.assert_array_equal(info["spawned_bonus_id"], np.asarray([1]))
    np.testing.assert_array_equal(info["spawned_slot"], np.asarray([0]))
    np.testing.assert_allclose(info["spawned_pos"], [[23.94, 64.06]])
    np.testing.assert_array_equal(info["accepted_position_attempt"], np.asarray([0]))
    np.testing.assert_array_equal(info["position_attempt_count"], np.asarray([1]))
    np.testing.assert_allclose(info["position_margin"], [3.88])
    np.testing.assert_allclose(info["position_span"], [80.24])

    np.testing.assert_array_equal(state["bonus_active"], np.asarray([[True, False]]))
    assert int(state["bonus_type"][0, 0]) == vector_runtime.BONUS_TYPE_SELF_SMALL
    assert int(state["bonus_id"][0, 0]) == 1
    np.testing.assert_allclose(state["bonus_pos"][0, 0], [23.94, 64.06])
    assert float(state["bonus_radius"][0, 0]) == vector_runtime.SOURCE_BONUS_RADIUS
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([1]))
    np.testing.assert_array_equal(state["bonus_world_body_count"], np.asarray([1]))

    assert info["spawn_events"][0]["event"] == "bonus:pop"
    assert info["spawn_events"][0]["bonus"] == 1
    assert info["spawn_events"][0]["type"] == "BonusSelfSmall"
    np.testing.assert_allclose(
        [info["spawn_events"][0]["x"], info["spawn_events"][0]["y"]],
        [23.94, 64.06],
    )
    assert int(state["event_count"][0]) == 1
    assert int(state["event_type"][0, 0]) == vector_runtime.EVENT_BONUS_POP
    np.testing.assert_array_equal(
        state["event_value_i"][0, 0],
        np.asarray([1, vector_runtime.BONUS_TYPE_SELF_SMALL], dtype=np.int32),
    )
    np.testing.assert_allclose(state["event_value_f"][0, 0], [23.94, 64.06])


def test_bonus_spawn_due_rows_retries_against_game_world_from_source_fixture():
    scenario = _load_lifecycle_scenario("source_bonus_spawn_game_world_retry_step.json")
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    sequence = random_setup["math_random_sequence"]
    assert [entry["label"] for entry in sequence] == [
        "bonus.start_delay",
        "bonus.next_delay_after_pop",
        "bonus.type.BonusSelfSmall",
        "bonus.position.x",
        "bonus.position.y",
        "bonus.position.retry_1.x",
        "bonus.position.retry_1.y",
    ]

    initial_state = scenario["initial_state"]
    assert isinstance(initial_state, dict)
    world_body = initial_state["world_bodies"][0]
    assert isinstance(world_body, dict)
    state = _bonus_spawn_state()
    state["world_active"][0] = True
    state["world_body_count"][0] = 1
    state["body_active"][0, 0] = True
    state["body_pos"][0, 0] = (float(world_body["x"]), float(world_body["y"]))
    state["body_radius"][0, 0] = float(world_body["radius"])
    state["body_owner"][0, 0] = 0
    state["body_num"][0, 0] = int(world_body["num"])
    state["body_insert_kind"][0, 0] = vector_runtime.BODY_KIND_NORMAL
    state["body_write_cursor"][0] = 1

    info = vector_runtime.bonus_spawn_due_rows(
        state,
        player_count=2,
        due_rows=np.asarray([True], dtype=bool),
        type_draws=np.asarray([sequence[2]["value"]], dtype=np.float64),
        position_draws=np.asarray(
            [
                [
                    [sequence[3]["value"], sequence[4]["value"]],
                    [sequence[5]["value"], sequence[6]["value"]],
                ]
            ],
            dtype=np.float64,
        ),
        enabled_type_codes=np.asarray(
            [vector_runtime.BONUS_TYPE_SELF_SMALL],
            dtype=np.int16,
        ),
    )

    np.testing.assert_array_equal(info["accepted_position_attempt"], np.asarray([1]))
    np.testing.assert_array_equal(info["position_attempt_count"], np.asarray([2]))
    np.testing.assert_array_equal(info["rejected_game_world_attempts"], np.asarray([1]))
    np.testing.assert_array_equal(info["rejected_bonus_world_attempts"], np.asarray([0]))
    np.testing.assert_allclose(info["spawned_pos"], [[68.072, 19.928]])
    np.testing.assert_allclose(state["bonus_pos"][0, 0], [68.072, 19.928])
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([1]))


def test_bonus_spawn_due_rows_retries_against_bonus_world_from_source_fixture():
    scenario = _load_lifecycle_scenario("source_bonus_spawn_bonus_world_retry_step.json")
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    sequence = random_setup["math_random_sequence"]
    assert [entry["label"] for entry in sequence] == [
        "bonus.start_delay",
        "bonus.next_delay_after_pop",
        "bonus.type.BonusSelfSmall",
        "bonus.position.x",
        "bonus.position.y",
        "bonus.position.retry_1.x",
        "bonus.position.retry_1.y",
    ]

    initial_state = scenario["initial_state"]
    assert isinstance(initial_state, dict)
    seeded_bonus = initial_state["active_bonuses"][0]
    assert isinstance(seeded_bonus, dict)
    state = _bonus_spawn_state()
    _seed_spawn_bonus(
        state,
        slot=0,
        bonus_id=1,
        x=float(seeded_bonus["x"]),
        y=float(seeded_bonus["y"]),
    )

    info = vector_runtime.bonus_spawn_due_rows(
        state,
        player_count=2,
        due_rows=np.asarray([True], dtype=bool),
        type_draws=np.asarray([sequence[2]["value"]], dtype=np.float64),
        position_draws=np.asarray(
            [
                [
                    [sequence[3]["value"], sequence[4]["value"]],
                    [sequence[5]["value"], sequence[6]["value"]],
                ]
            ],
            dtype=np.float64,
        ),
        enabled_type_codes=np.asarray(
            [vector_runtime.BONUS_TYPE_SELF_SMALL],
            dtype=np.int16,
        ),
    )

    np.testing.assert_array_equal(info["accepted_position_attempt"], np.asarray([1]))
    np.testing.assert_array_equal(info["position_attempt_count"], np.asarray([2]))
    np.testing.assert_array_equal(info["rejected_game_world_attempts"], np.asarray([0]))
    np.testing.assert_array_equal(info["rejected_bonus_world_attempts"], np.asarray([1]))
    np.testing.assert_array_equal(info["spawned_bonus_id"], np.asarray([2]))
    np.testing.assert_array_equal(info["spawned_slot"], np.asarray([1]))
    np.testing.assert_allclose(info["spawned_pos"], [[68.072, 19.928]])
    np.testing.assert_array_equal(state["bonus_active"], np.asarray([[True, True]]))
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([2]))
    np.testing.assert_array_equal(state["bonus_world_body_count"], np.asarray([2]))
    assert int(state["bonus_id"][0, 1]) == 2
    np.testing.assert_allclose(state["bonus_pos"][0, 1], [68.072, 19.928])


def test_bonus_spawn_due_rows_raises_when_all_position_candidates_hit_body_world():
    state = _bonus_spawn_state(body_capacity=2)
    position_draws = np.asarray(
        [
            [
                [0.25, 0.25],
                [0.75, 0.75],
            ]
        ],
        dtype=np.float64,
    )
    margin = vector_runtime.SOURCE_BONUS_RADIUS + (
        vector_runtime.SOURCE_BONUS_POSITION_MARGIN_FRACTION
        * float(state["map_size"][0])
    )
    span = float(state["map_size"][0]) - margin * 2.0
    candidate_positions = margin + position_draws[0] * span

    state["world_active"][0] = True
    state["world_body_count"][0] = 2
    state["body_active"][0, :] = True
    state["body_pos"][0, :] = candidate_positions
    state["body_radius"][0, :] = vector_runtime.SOURCE_BONUS_RADIUS
    state["body_owner"][0, :] = [0, 1]
    state["body_num"][0, :] = [0, 0]
    state["body_insert_kind"][0, :] = vector_runtime.BODY_KIND_NORMAL
    state["body_write_cursor"][0] = 2
    state["body_count"][0, :] = [1, 1]

    with pytest.raises(
        vector_runtime.VectorRuntimeError,
        match="row 0 position_draws did not include an accepted candidate",
    ):
        vector_runtime.bonus_spawn_due_rows(
            state,
            player_count=2,
            due_rows=np.asarray([True], dtype=bool),
            type_draws=np.asarray([0.0], dtype=np.float64),
            position_draws=position_draws,
            enabled_type_codes=np.asarray(
                [vector_runtime.BONUS_TYPE_SELF_SMALL],
                dtype=np.int16,
            ),
        )

    np.testing.assert_array_equal(state["bonus_active"], np.asarray([[False, False]]))
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([0]))
    np.testing.assert_array_equal(state["bonus_world_body_count"], np.asarray([0]))


def test_bonus_spawn_due_rows_skips_type_position_at_source_cap():
    scenario = _load_lifecycle_scenario("source_bonus_spawn_cap_twenty_step.json")
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    sequence = random_setup["math_random_sequence"]
    assert [entry["label"] for entry in sequence] == [
        "bonus.start_delay",
        "bonus.next_delay_after_pop",
    ]

    state = _bonus_spawn_state(
        bonus_capacity=vector_runtime.SOURCE_MAX_ACTIVE_BONUSES,
        event_capacity=2,
    )
    for slot in range(vector_runtime.SOURCE_MAX_ACTIVE_BONUSES):
        _seed_spawn_bonus(state, slot=slot, bonus_id=slot + 1, x=44.0, y=44.0)

    info = vector_runtime.bonus_spawn_due_rows(
        state,
        player_count=2,
        due_rows=np.asarray([True], dtype=bool),
        events_enabled=True,
    )

    np.testing.assert_array_equal(info["capped_rows"], np.asarray([True]))
    np.testing.assert_array_equal(info["selection_rows"], np.asarray([False]))
    np.testing.assert_array_equal(info["spawn_rows"], np.asarray([False]))
    np.testing.assert_array_equal(info["selected_type_code"], np.asarray([0]))
    np.testing.assert_array_equal(info["position_attempt_count"], np.asarray([0]))
    assert info["type_selection_info"] is None
    assert info["spawn_events"] == []
    np.testing.assert_array_equal(
        state["bonus_count"],
        np.asarray([vector_runtime.SOURCE_MAX_ACTIVE_BONUSES]),
    )
    np.testing.assert_array_equal(
        state["bonus_world_body_count"],
        np.asarray([vector_runtime.SOURCE_MAX_ACTIVE_BONUSES]),
    )
    assert int(state["event_count"][0]) == 0


@pytest.mark.parametrize(
    (
        "scenario_name",
        "expected_alive",
        "expected_death_count",
        "expected_death_cause",
        "expected_death_hit_owner",
        "expected_world_body_count",
    ),
    [
        (
            "source_body_opponent_tangent_safe_step.json",
            [True, True],
            0,
            vector_runtime.DEATH_CAUSE_NONE,
            -1,
            1,
        ),
        (
            "source_body_opponent_overlap_kills_step.json",
            [False, True],
            1,
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL,
            1,
            2,
        ),
        (
            "source_body_own_delta3_safe_step.json",
            [True, True],
            0,
            vector_runtime.DEATH_CAUSE_NONE,
            -1,
            1,
        ),
        (
            "source_body_own_delta4_kills_step.json",
            [False, True],
            1,
            vector_runtime.DEATH_CAUSE_OWN_TRAIL,
            0,
            2,
        ),
    ],
    ids=[
        "opponent-tangent-safe-2p",
        "opponent-overlap-kills-2p",
        "own-delta3-safe-2p",
        "own-delta4-kills-2p",
    ],
)
def test_step_many_2p_body_canary_fixture_slice_matches_expected_deaths(
    scenario_name: str,
    expected_alive: list[bool],
    expected_death_count: int,
    expected_death_cause: int,
    expected_death_hit_owner: int,
    expected_world_body_count: int,
):
    fixture, state = _runtime_fixture_state(
        f"scenarios/environment/{scenario_name}",
        body_capacity=8,
    )
    _slice_runtime_state_to_player_count(state, player_count=2)

    counters = _step_runtime_fixture_with_player_count(
        fixture,
        state,
        step_index=0,
        player_count=2,
    )

    np.testing.assert_array_equal(state["alive"], np.asarray([expected_alive]))
    np.testing.assert_array_equal(
        state["death_count"],
        np.asarray([expected_death_count], dtype=np.int32),
    )
    expected_death_player = [0, -1] if expected_death_count else [-1, -1]
    expected_death_causes = [
        expected_death_cause if expected_death_count else vector_runtime.DEATH_CAUSE_NONE,
        vector_runtime.DEATH_CAUSE_NONE,
    ]
    expected_hit_owners = [expected_death_hit_owner, -1]
    np.testing.assert_array_equal(
        state["death_player"],
        np.asarray([expected_death_player], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray([expected_death_causes], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["death_hit_owner"],
        np.asarray([expected_hit_owners], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["world_body_count"],
        np.asarray([expected_world_body_count], dtype=np.int32),
    )
    assert counters["body_hits"] == expected_death_count
    assert counters["terminal_score_rows"] == expected_death_count
    np.testing.assert_array_equal(
        state["done"],
        np.asarray([bool(expected_death_count)], dtype=bool),
    )
    np.testing.assert_array_equal(
        state["winner"],
        np.asarray([1 if expected_death_count else -1], dtype=np.int16),
    )
    if expected_death_count:
        np.testing.assert_array_equal(state["score"], np.asarray([[0, 1]]))
        np.testing.assert_array_equal(
            state["terminal_reason"],
            np.asarray([vector_runtime.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
        )


@pytest.mark.parametrize(
    (
        "scenario_name",
        "expected_alive",
        "expected_death_count",
        "expected_world_body_count",
        "expected_printing",
    ),
    [
        (
            "source_trail_gap_hole_space_safe_step.json",
            [True, True],
            0,
            1,
            [False, False],
        ),
        (
            "source_trail_gap_stored_body_still_kills_step.json",
            [False, True],
            1,
            2,
            [False, False],
        ),
        (
            "source_trail_gap_print_to_hole_boundary_kills_step.json",
            [False, True],
            1,
            2,
            [False, False],
        ),
        (
            "source_trail_gap_hole_to_print_boundary_kills_step.json",
            [False, True],
            1,
            2,
            [False, True],
        ),
    ],
    ids=[
        "hole-space-safe-2p",
        "stored-body-still-kills-2p",
        "print-to-hole-boundary-kills-2p",
        "hole-to-print-boundary-kills-2p",
    ],
)
def test_step_many_2p_trail_gap_fixture_slice_matches_expected_deaths(
    scenario_name: str,
    expected_alive: list[bool],
    expected_death_count: int,
    expected_world_body_count: int,
    expected_printing: list[bool],
):
    fixture, state = _runtime_fixture_state(
        f"scenarios/environment/{scenario_name}",
        body_capacity=8,
    )
    _slice_runtime_state_to_player_count(state, player_count=2)

    counters = _step_runtime_fixture_with_player_count(
        fixture,
        state,
        step_index=0,
        player_count=2,
    )

    np.testing.assert_array_equal(state["alive"], np.asarray([expected_alive]))
    np.testing.assert_array_equal(
        state["printing"],
        np.asarray([expected_printing], dtype=bool),
    )
    np.testing.assert_array_equal(
        state["death_count"],
        np.asarray([expected_death_count], dtype=np.int32),
    )
    expected_death_player = [0, -1] if expected_death_count else [-1, -1]
    expected_death_causes = [
        (
            vector_runtime.DEATH_CAUSE_OPPONENT_TRAIL
            if expected_death_count
            else vector_runtime.DEATH_CAUSE_NONE
        ),
        vector_runtime.DEATH_CAUSE_NONE,
    ]
    np.testing.assert_array_equal(
        state["death_player"],
        np.asarray([expected_death_player], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["death_cause"],
        np.asarray([expected_death_causes], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["death_hit_owner"],
        np.asarray([[1, -1] if expected_death_count else [-1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["world_body_count"],
        np.asarray([expected_world_body_count], dtype=np.int32),
    )
    assert counters["body_hits"] == expected_death_count
    assert counters["terminal_score_rows"] == expected_death_count
    np.testing.assert_array_equal(
        state["done"],
        np.asarray([bool(expected_death_count)], dtype=bool),
    )
    np.testing.assert_array_equal(
        state["winner"],
        np.asarray([1 if expected_death_count else -1], dtype=np.int16),
    )
    if expected_death_count:
        np.testing.assert_array_equal(state["score"], np.asarray([[0, 1]]))
        np.testing.assert_array_equal(
            state["terminal_reason"],
            np.asarray([vector_runtime.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
        )


def test_step_many_bonus_fixture_without_optional_arrays_runs_like_no_bonus():
    fixture, state = _runtime_fixture_state(
        "scenarios/environment/source_bonus_self_small_catch_step.json",
        body_capacity=4,
    )

    counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert counters["bonus_self_small_catches"] == 0
    assert counters["bonus_stack_appends"] == 0
    assert "bonus_active" not in state
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 0.6]]))
    assert int(state["event_count"][0]) == 2
    np.testing.assert_array_equal(
        state["event_type"][0, :2],
        np.asarray(
            [vector_runtime.EVENT_POSITION, vector_runtime.EVENT_POSITION],
            dtype=np.int16,
        ),
    )


def test_step_many_bonus_game_clear_without_optional_arrays_runs_like_no_bonus():
    fixture, state = _runtime_fixture_state(
        "scenarios/environment/source_bonus_game_clear_immediate_step.json",
        body_capacity=4,
    )

    counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert counters["bonus_game_clear_catches"] == 0
    assert counters["bonus_stack_appends"] == 0
    assert "bonus_active" not in state
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(state["world_body_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(state["body_active"][0], [True, False, False, False])
    assert int(state["event_count"][0]) == 2
    np.testing.assert_array_equal(
        state["event_type"][0, :2],
        np.asarray(
            [vector_runtime.EVENT_POSITION, vector_runtime.EVENT_POSITION],
            dtype=np.int16,
        ),
    )


def test_step_many_bonus_game_borderless_without_optional_arrays_runs_like_no_bonus():
    fixture, state = _runtime_fixture_state(
        "scenarios/environment/source_bonus_game_borderless_catch_step.json",
        body_capacity=4,
    )

    counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert counters["bonus_game_borderless_catches"] == 0
    assert counters["bonus_stack_appends"] == 0
    assert "bonus_active" not in state
    np.testing.assert_array_equal(state["borderless"], np.asarray([False], dtype=bool))
    assert int(state["event_count"][0]) == 2
    np.testing.assert_array_equal(
        state["event_type"][0, :2],
        np.asarray(
            [vector_runtime.EVENT_POSITION, vector_runtime.EVENT_POSITION],
            dtype=np.int16,
        ),
    )


def test_step_many_catches_forced_bonus_self_small_after_print_manager_like_fixture():
    scenario_name = "source_bonus_self_small_catch_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(state, scenario_name)

    counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert counters["bonus_self_small_catches"] == 1
    assert counters["bonus_stack_appends"] == 1
    np.testing.assert_allclose(state["radius"], np.asarray([[0.3, 0.6]]))
    np.testing.assert_array_equal(state["radius_power"], np.asarray([[-1, 0]], dtype=np.int16))
    np.testing.assert_array_equal(state["bonus_active"], np.asarray([[False]]))
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        state["bonus_world_body_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[1, 0]]))
    assert int(state["bonus_stack_id"][0, 0, 0]) == 1
    assert int(state["bonus_stack_type"][0, 0, 0]) == vector_runtime.BONUS_TYPE_SELF_SMALL
    assert int(state["bonus_stack_duration_ms"][0, 0, 0]) == 7500
    assert int(state["bonus_stack_radius_power"][0, 0, 0]) == -1

    assert int(state["event_count"][0]) == 5
    np.testing.assert_array_equal(
        state["event_type"][0, :5],
        np.asarray(
            [
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_BONUS_CLEAR,
                vector_runtime.EVENT_PROPERTY,
                vector_runtime.EVENT_BONUS_STACK,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["event_player"][0, :5],
        np.asarray([1, 0, -1, 0, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["event_value_i"][0, 2:5],
        np.asarray(
            [
                [1, 0],
                [vector_runtime.PROPERTY_RADIUS, 0],
                [1, vector_runtime.BONUS_TYPE_SELF_SMALL],
            ],
            dtype=np.int32,
        ),
    )
    assert state["event_value_f"][0, 3, 0] == pytest.approx(0.3)
    assert int(state["event_bool"][0, 4]) == vector_runtime.BONUS_STACK_METHOD_ADD
    np.testing.assert_allclose(
        state["event_value_f"][0, 4],
        np.asarray([7500.0, -1.0], dtype=np.float64),
    )


def test_step_many_forced_bonus_self_small_expiry_restores_radius_like_fixture():
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(state, scenario_name)

    catch_counters = _step_runtime_fixture(fixture, state, step_index=0)
    expiry_counters = _step_runtime_fixture(fixture, state, step_index=1)

    assert catch_counters["bonus_self_small_catches"] == 1
    assert catch_counters["bonus_stack_appends"] == 1
    assert expiry_counters["bonus_self_small_expiries"] == 1
    assert expiry_counters["bonus_self_small_catches"] == 0
    assert expiry_counters["bonus_stack_appends"] == 0
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(state["radius_power"], np.asarray([[0, 0]], dtype=np.int16))
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(state["bonus_stack_id"][0, 0, 0]) == -1
    assert int(state["bonus_stack_type"][0, 0, 0]) == vector_runtime.BONUS_TYPE_NONE
    assert int(state["bonus_stack_duration_ms"][0, 0, 0]) == 0
    assert int(state["bonus_stack_radius_power"][0, 0, 0]) == 0
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        state["bonus_world_body_count"],
        np.asarray([1], dtype=np.int32),
    )

    assert int(state["event_count"][0]) == 4
    np.testing.assert_array_equal(
        state["event_type"][0, :4],
        np.asarray(
            [
                vector_runtime.EVENT_PROPERTY,
                vector_runtime.EVENT_BONUS_STACK,
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_POSITION,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["event_player"][0, :4],
        np.asarray([0, 0, 1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["event_value_i"][0, :2],
        np.asarray(
            [
                [vector_runtime.PROPERTY_RADIUS, 0],
                [1, vector_runtime.BONUS_TYPE_SELF_SMALL],
            ],
            dtype=np.int32,
        ),
    )
    assert state["event_value_f"][0, 0, 0] == pytest.approx(0.6)
    assert int(state["event_bool"][0, 1]) == vector_runtime.BONUS_STACK_METHOD_REMOVE
    np.testing.assert_allclose(
        state["event_value_f"][0, 1],
        np.asarray([7500.0, -1.0], dtype=np.float64),
    )


def test_step_many_catches_forced_bonus_self_slow_and_expiry_restores_speed():
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(
        state,
        scenario_name,
        bonus_type_code=vector_runtime.BONUS_TYPE_SELF_SLOW,
    )

    catch_counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert catch_counters["bonus_self_slow_catches"] == 1
    assert catch_counters["bonus_stack_appends"] == 1
    np.testing.assert_allclose(state["speed"], np.asarray([[8.0, 16.0]]))
    assert state["angular_velocity_per_ms"][0, 0] == pytest.approx(
        _expected_source_angular_velocity_for_speed(8.0)
    )
    assert state["angular_velocity_per_ms"][0, 1] == pytest.approx(
        vector_runtime.SOURCE_AVATAR_ANGULAR_VELOCITY_PER_MS
    )
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[1, 0]]))
    assert int(state["bonus_stack_type"][0, 0, 0]) == vector_runtime.BONUS_TYPE_SELF_SLOW
    assert int(state["bonus_stack_duration_ms"][0, 0, 0]) == (
        vector_runtime.BONUS_SELF_SLOW_DURATION_MS
    )
    assert state["event_value_i"][0, 3, 0] == vector_runtime.PROPERTY_VELOCITY
    assert state["event_value_f"][0, 3, 0] == pytest.approx(8.0)

    expiry_counters = _step_runtime_timer_only(
        state,
        timer_advance_ms=vector_runtime.BONUS_SELF_SLOW_DURATION_MS,
    )

    assert expiry_counters["bonus_self_slow_expiries"] == 1
    np.testing.assert_allclose(state["speed"], np.asarray([[16.0, 16.0]]))
    np.testing.assert_allclose(
        state["angular_velocity_per_ms"],
        np.full((1, 2), vector_runtime.SOURCE_AVATAR_ANGULAR_VELOCITY_PER_MS),
    )
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(state["bonus_stack_type"][0, 0, 0]) == vector_runtime.BONUS_TYPE_NONE


@pytest.mark.parametrize(
    (
        "bonus_type_code",
        "catch_counter",
        "expiry_counter",
        "target_player",
        "expected_speed",
        "expected_delta",
        "duration_ms",
    ),
    [
        (
            vector_runtime.BONUS_TYPE_SELF_FAST,
            "bonus_self_fast_catches",
            "bonus_self_fast_expiries",
            0,
            28.0,
            vector_runtime.BONUS_SELF_FAST_VELOCITY_DELTA,
            vector_runtime.BONUS_SELF_FAST_DURATION_MS,
        ),
        (
            vector_runtime.BONUS_TYPE_ENEMY_SLOW,
            "bonus_enemy_slow_catches",
            "bonus_enemy_slow_expiries",
            1,
            8.0,
            vector_runtime.BONUS_ENEMY_SLOW_VELOCITY_DELTA,
            vector_runtime.BONUS_ENEMY_SLOW_DURATION_MS,
        ),
        (
            vector_runtime.BONUS_TYPE_ENEMY_FAST,
            "bonus_enemy_fast_catches",
            "bonus_enemy_fast_expiries",
            1,
            28.0,
            vector_runtime.BONUS_ENEMY_FAST_VELOCITY_DELTA,
            vector_runtime.BONUS_ENEMY_FAST_DURATION_MS,
        ),
    ],
)
def test_step_many_catches_forced_velocity_bonus_and_expiry_restores_speed(
    bonus_type_code,
    catch_counter,
    expiry_counter,
    target_player,
    expected_speed,
    expected_delta,
    duration_ms,
):
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(
        state,
        scenario_name,
        bonus_type_code=bonus_type_code,
    )

    catch_counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert catch_counters[catch_counter] == 1
    assert catch_counters["bonus_stack_appends"] == 1
    assert state["speed"][0, target_player] == pytest.approx(expected_speed)
    assert state["speed"][0, 1 - target_player] == pytest.approx(16.0)
    assert state["angular_velocity_per_ms"][0, target_player] == pytest.approx(
        _expected_source_angular_velocity_for_speed(expected_speed)
    )
    assert state["angular_velocity_per_ms"][0, 1 - target_player] == pytest.approx(
        vector_runtime.SOURCE_AVATAR_ANGULAR_VELOCITY_PER_MS
    )
    np.testing.assert_array_equal(
        state["bonus_stack_count"],
        np.asarray([[int(target_player == 0), int(target_player == 1)]]),
    )
    assert int(state["bonus_stack_type"][0, target_player, 0]) == bonus_type_code
    assert int(state["bonus_stack_duration_ms"][0, target_player, 0]) == duration_ms
    assert state["bonus_stack_velocity_delta"][0, target_player, 0] == pytest.approx(
        expected_delta
    )

    expiry_counters = _step_runtime_timer_only(state, timer_advance_ms=duration_ms)

    assert expiry_counters[expiry_counter] == 1
    np.testing.assert_allclose(state["speed"], np.asarray([[16.0, 16.0]]))
    np.testing.assert_allclose(
        state["angular_velocity_per_ms"],
        np.full((1, 2), vector_runtime.SOURCE_AVATAR_ANGULAR_VELOCITY_PER_MS),
    )
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(state["bonus_stack_type"][0, target_player, 0]) == (
        vector_runtime.BONUS_TYPE_NONE
    )


def test_step_many_catches_forced_bonus_enemy_big_targets_other_alive_avatar():
    scenario_name = "source_bonus_self_small_catch_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(
        state,
        scenario_name,
        bonus_type_code=vector_runtime.BONUS_TYPE_ENEMY_BIG,
    )

    catch_counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert catch_counters["bonus_enemy_big_catches"] == 1
    assert catch_counters["bonus_stack_appends"] == 1
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 1.2]]))
    np.testing.assert_array_equal(state["radius_power"], np.asarray([[0, 1]], dtype=np.int16))
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 1]]))
    assert int(state["bonus_stack_type"][0, 1, 0]) == vector_runtime.BONUS_TYPE_ENEMY_BIG
    assert int(state["bonus_stack_duration_ms"][0, 1, 0]) == (
        vector_runtime.BONUS_ENEMY_BIG_DURATION_MS
    )

    expiry_counters = _step_runtime_timer_only(
        state,
        timer_advance_ms=vector_runtime.BONUS_ENEMY_BIG_DURATION_MS,
    )

    assert expiry_counters["bonus_enemy_big_expiries"] == 1
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(state["radius_power"], np.asarray([[0, 0]], dtype=np.int16))
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))


def test_step_many_catches_forced_bonus_enemy_inverse_and_expiry_restores_direction():
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(
        state,
        scenario_name,
        bonus_type_code=vector_runtime.BONUS_TYPE_ENEMY_INVERSE,
    )

    catch_counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert catch_counters["bonus_enemy_inverse_catches"] == 1
    assert catch_counters["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(state["inverse"], np.asarray([[False, True]]))
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 1]]))
    assert int(state["bonus_stack_type"][0, 1, 0]) == (
        vector_runtime.BONUS_TYPE_ENEMY_INVERSE
    )
    assert int(state["bonus_stack_inverse_delta"][0, 1, 0]) == 1

    expiry_counters = _step_runtime_timer_only(
        state,
        timer_advance_ms=vector_runtime.BONUS_ENEMY_INVERSE_DURATION_MS,
    )

    assert expiry_counters["bonus_enemy_inverse_expiries"] == 1
    np.testing.assert_array_equal(state["inverse"], np.asarray([[False, False]]))
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))


def test_step_many_catches_forced_bonus_enemy_straight_angle_and_expiry_restores_turn_rate():
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(
        state,
        scenario_name,
        bonus_type_code=vector_runtime.BONUS_TYPE_ENEMY_STRAIGHT_ANGLE,
    )
    base_angular_velocity = state["base_angular_velocity_per_ms"].copy()

    catch_counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert catch_counters["bonus_enemy_straight_angle_catches"] == 1
    assert catch_counters["bonus_stack_appends"] == 1
    np.testing.assert_allclose(
        state["angular_velocity_per_ms"],
        np.asarray(
            [
                [
                    base_angular_velocity[0, 0],
                    vector_runtime.SOURCE_STRAIGHT_ANGLE_RADIANS,
                ]
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 1]]))
    assert int(state["bonus_stack_type"][0, 1, 0]) == (
        vector_runtime.BONUS_TYPE_ENEMY_STRAIGHT_ANGLE
    )
    assert int(state["bonus_stack_duration_ms"][0, 1, 0]) == (
        vector_runtime.BONUS_ENEMY_STRAIGHT_ANGLE_DURATION_MS
    )
    assert state["bonus_stack_angular_velocity_per_ms"][0, 1, 0] == pytest.approx(
        vector_runtime.SOURCE_STRAIGHT_ANGLE_RADIANS
    )

    expiry_counters = _step_runtime_timer_only(
        state,
        timer_advance_ms=vector_runtime.BONUS_ENEMY_STRAIGHT_ANGLE_DURATION_MS,
    )

    assert expiry_counters["bonus_enemy_straight_angle_expiries"] == 1
    np.testing.assert_allclose(state["angular_velocity_per_ms"], base_angular_velocity)
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(state["bonus_stack_type"][0, 1, 0]) == vector_runtime.BONUS_TYPE_NONE


def test_step_many_catches_forced_bonus_self_master_and_expiry_restores_state():
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(
        state,
        scenario_name,
        bonus_type_code=vector_runtime.BONUS_TYPE_SELF_MASTER,
    )
    state["printing"][:, :] = True
    state["print_manager_active"][:, :] = True

    catch_counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert catch_counters["bonus_self_master_catches"] == 1
    assert catch_counters["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(state["invincible"], np.asarray([[True, False]]))
    np.testing.assert_array_equal(state["printing"], np.asarray([[False, True]]))
    np.testing.assert_array_equal(
        state["print_manager_active"],
        np.asarray([[False, True]]),
    )
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[1, 0]]))
    assert int(state["bonus_stack_type"][0, 0, 0]) == (
        vector_runtime.BONUS_TYPE_SELF_MASTER
    )
    assert int(state["bonus_stack_duration_ms"][0, 0, 0]) == (
        vector_runtime.BONUS_SELF_MASTER_DURATION_MS
    )
    assert int(state["bonus_stack_invincible_delta"][0, 0, 0]) == 1
    assert int(state["bonus_stack_printing_delta"][0, 0, 0]) == -1

    expiry_counters = _step_runtime_timer_only(
        state,
        timer_advance_ms=vector_runtime.BONUS_SELF_MASTER_DURATION_MS,
    )

    assert expiry_counters["bonus_self_master_expiries"] == 1
    np.testing.assert_array_equal(state["invincible"], np.asarray([[False, False]]))
    np.testing.assert_array_equal(state["printing"], np.asarray([[True, True]]))
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(state["bonus_stack_type"][0, 0, 0]) == vector_runtime.BONUS_TYPE_NONE
    assert int(state["bonus_stack_invincible_delta"][0, 0, 0]) == 0
    assert int(state["bonus_stack_printing_delta"][0, 0, 0]) == 0


def test_step_many_catches_forced_bonus_all_color_and_expiry_restores_colors():
    scenario_name = "source_bonus_self_small_expiry_restore_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(
        state,
        scenario_name,
        bonus_type_code=vector_runtime.BONUS_TYPE_ALL_COLOR,
    )

    catch_counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert catch_counters["bonus_all_color_catches"] == 1
    assert catch_counters["bonus_stack_appends"] == 2
    np.testing.assert_array_equal(state["avatar_color"], np.asarray([[1, 0]]))
    np.testing.assert_array_equal(state["base_avatar_color"], np.asarray([[0, 1]]))
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[1, 1]]))
    np.testing.assert_array_equal(
        state["bonus_stack_type"][0, :, 0],
        np.asarray([vector_runtime.BONUS_TYPE_ALL_COLOR] * 2, dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["bonus_stack_color"][0, :, 0],
        np.asarray([1, 0], dtype=np.int16),
    )

    expiry_counters = _step_runtime_timer_only(
        state,
        timer_advance_ms=vector_runtime.BONUS_ALL_COLOR_DURATION_MS,
    )

    assert expiry_counters["bonus_all_color_expiries"] == 2
    np.testing.assert_array_equal(state["avatar_color"], np.asarray([[0, 1]]))
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))
    np.testing.assert_array_equal(
        state["bonus_stack_color"][0, :, 0],
        np.asarray([-1, -1], dtype=np.int16),
    )


def test_step_many_forced_bonus_self_small_tangent_does_not_catch_like_fixture():
    scenario_name = "source_bonus_self_small_tangent_no_catch_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(state, scenario_name)

    counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert counters["bonus_self_small_catches"] == 0
    assert counters["bonus_stack_appends"] == 0
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(state["radius_power"], np.asarray([[0, 0]], dtype=np.int16))
    np.testing.assert_array_equal(state["bonus_active"], np.asarray([[True]]))
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        state["bonus_world_body_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))
    assert int(state["event_count"][0]) == 2
    assert vector_runtime.EVENT_BONUS_CLEAR not in set(state["event_type"][0, :2])
    assert vector_runtime.EVENT_BONUS_STACK not in set(state["event_type"][0, :2])


def test_step_many_forced_bonus_self_small_wall_death_does_not_catch_like_fixture():
    scenario_name = "source_bonus_self_small_wall_death_no_catch_step.json"
    fixture, state = _runtime_fixture_state(f"scenarios/environment/{scenario_name}")
    _add_forced_bonus_self_small_arrays(state, scenario_name)

    counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert counters["normal_wall_deaths"] == 1
    assert counters["terminal_score_rows"] == 1
    assert counters["bonus_self_small_catches"] == 0
    assert counters["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(state["alive"], np.asarray([[False, True]]))
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(state["radius_power"], np.asarray([[0, 0]], dtype=np.int16))
    np.testing.assert_array_equal(state["bonus_active"], np.asarray([[True]]))
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        state["bonus_world_body_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(state["bonus_stack_count"], np.asarray([[0, 0]]))
    np.testing.assert_array_equal(state["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(state["winner"], np.asarray([1], dtype=np.int16))
    event_types = state["event_type"][0, : int(state["event_count"][0])]
    assert vector_runtime.EVENT_BONUS_CLEAR not in set(event_types)
    assert vector_runtime.EVENT_BONUS_STACK not in set(event_types)


def test_step_many_catches_forced_bonus_game_clear_immediately_like_fixture():
    scenario_name = "source_bonus_game_clear_immediate_step.json"
    fixture, state = _runtime_fixture_state(
        f"scenarios/environment/{scenario_name}",
        body_capacity=4,
    )
    state["world_active"] = np.asarray([True], dtype=bool)
    _add_forced_bonus_game_clear_arrays(state, scenario_name)

    counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert counters["bonus_self_small_catches"] == 0
    assert counters["bonus_game_clear_catches"] == 1
    assert counters["bonus_stack_appends"] == 0
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(state["world_active"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(state["world_body_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(state["body_active"], np.zeros((1, 4), dtype=bool))
    np.testing.assert_allclose(state["body_pos"], np.zeros((1, 4, 2), dtype=np.float64))
    np.testing.assert_allclose(state["body_radius"], np.zeros((1, 4), dtype=np.float64))
    np.testing.assert_array_equal(state["body_owner"], np.full((1, 4), -1, dtype=np.int16))
    np.testing.assert_array_equal(state["body_num"], np.full((1, 4), -1, dtype=np.int32))
    np.testing.assert_array_equal(
        state["body_insert_tick"],
        np.full((1, 4), -1, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        state["body_insert_kind"],
        np.full((1, 4), -1, dtype=np.int16),
    )
    np.testing.assert_array_equal(state["body_write_cursor"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(state["body_count"], np.asarray([[0, 1]], dtype=np.int32))
    np.testing.assert_array_equal(
        state["visible_trail_count"],
        np.asarray([[0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(state["bonus_active"], np.asarray([[False]]))
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        state["bonus_world_body_count"],
        np.asarray([1], dtype=np.int32),
    )
    assert "bonus_stack_count" not in state

    assert int(state["event_count"][0]) == 4
    np.testing.assert_array_equal(
        state["event_type"][0, :4],
        np.asarray(
            [
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_BONUS_CLEAR,
                vector_runtime.EVENT_CLEAR,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["event_player"][0, :4],
        np.asarray([1, 0, -1, -1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["event_value_i"][0, 2],
        np.asarray([1, 0], dtype=np.int32),
    )
    event_types = state["event_type"][0, : int(state["event_count"][0])]
    assert vector_runtime.EVENT_PROPERTY not in set(event_types)
    assert vector_runtime.EVENT_BONUS_STACK not in set(event_types)


def test_step_many_catches_forced_bonus_game_borderless_like_js_fixture():
    scenario_name = "source_bonus_game_borderless_catch_step.json"
    fixture, state = _runtime_fixture_state(
        f"scenarios/environment/{scenario_name}",
        body_capacity=4,
    )
    _add_forced_bonus_game_borderless_arrays(state, scenario_name)

    counters = _step_runtime_fixture(fixture, state, step_index=0)

    assert counters["bonus_self_small_catches"] == 0
    assert counters["bonus_game_clear_catches"] == 0
    assert counters["bonus_game_borderless_catches"] == 1
    assert counters["bonus_stack_appends"] == 1
    np.testing.assert_array_equal(state["borderless"], np.asarray([True], dtype=bool))
    np.testing.assert_allclose(state["radius"], np.asarray([[0.6, 0.6]]))
    np.testing.assert_array_equal(state["bonus_active"], np.asarray([[False]]))
    np.testing.assert_array_equal(state["bonus_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        state["bonus_world_body_count"],
        np.asarray([1], dtype=np.int32),
    )
    assert "bonus_stack_count" not in state
    np.testing.assert_array_equal(state["bonus_game_stack_count"], np.asarray([1]))
    assert int(state["bonus_game_stack_id"][0, 0]) == 1
    assert int(state["bonus_game_stack_type"][0, 0]) == (
        vector_runtime.BONUS_TYPE_GAME_BORDERLESS
    )
    assert int(state["bonus_game_stack_duration_ms"][0, 0]) == (
        vector_runtime.BONUS_GAME_BORDERLESS_DURATION_MS
    )
    assert int(state["bonus_game_stack_borderless"][0, 0]) == 1

    assert int(state["event_count"][0]) == 4
    np.testing.assert_array_equal(
        state["event_type"][0, :4],
        np.asarray(
            [
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_BONUS_CLEAR,
                vector_runtime.EVENT_BORDERLESS,
            ],
            dtype=np.int16,
        ),
    )
    np.testing.assert_array_equal(
        state["event_player"][0, :4],
        np.asarray([1, 0, -1, -1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        state["event_value_i"][0, 2],
        np.asarray([1, 0], dtype=np.int32),
    )
    assert int(state["event_bool"][0, 3]) == 1
    event_types = state["event_type"][0, : int(state["event_count"][0])]
    assert vector_runtime.EVENT_CLEAR not in set(event_types)
    assert vector_runtime.EVENT_PROPERTY not in set(event_types)
    assert vector_runtime.EVENT_BONUS_STACK not in set(event_types)


def test_step_many_forced_bonus_game_borderless_source_read_expiry_restores_false():
    scenario_name = "source_bonus_game_borderless_catch_step.json"
    fixture, state = _runtime_fixture_state(
        f"scenarios/environment/{scenario_name}",
        body_capacity=4,
    )
    _add_forced_bonus_game_borderless_arrays(state, scenario_name)

    catch_counters = _step_runtime_fixture(fixture, state, step_index=0)
    before_expiry_counters = _step_runtime_timer_only(
        state,
        timer_advance_ms=vector_runtime.BONUS_GAME_BORDERLESS_DURATION_MS - 1,
    )

    assert catch_counters["bonus_game_borderless_catches"] == 1
    assert catch_counters["bonus_stack_appends"] == 1
    assert before_expiry_counters["bonus_game_borderless_expiries"] == 0
    np.testing.assert_array_equal(state["borderless"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(state["bonus_game_stack_count"], np.asarray([1]))
    assert int(state["bonus_game_stack_duration_ms"][0, 0]) == 1
    event_types = state["event_type"][0, : int(state["event_count"][0])]
    assert vector_runtime.EVENT_BORDERLESS not in set(event_types)

    expiry_counters = _step_runtime_timer_only(state, timer_advance_ms=1.0)

    assert expiry_counters["bonus_game_borderless_expiries"] == 1
    assert expiry_counters["bonus_game_borderless_catches"] == 0
    assert expiry_counters["bonus_stack_appends"] == 0
    np.testing.assert_array_equal(state["borderless"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(state["bonus_game_stack_count"], np.asarray([0]))
    assert int(state["bonus_game_stack_id"][0, 0]) == -1
    assert int(state["bonus_game_stack_type"][0, 0]) == vector_runtime.BONUS_TYPE_NONE
    assert int(state["bonus_game_stack_duration_ms"][0, 0]) == 0
    assert int(state["bonus_game_stack_borderless"][0, 0]) == 0

    assert int(state["event_count"][0]) == 3
    np.testing.assert_array_equal(
        state["event_type"][0, :3],
        np.asarray(
            [
                vector_runtime.EVENT_BORDERLESS,
                vector_runtime.EVENT_POSITION,
                vector_runtime.EVENT_POSITION,
            ],
            dtype=np.int16,
        ),
    )
    assert int(state["event_bool"][0, 0]) == 0

    no_stack_counters = _step_runtime_timer_only(state, timer_advance_ms=1.0)

    assert no_stack_counters["bonus_game_borderless_expiries"] == 0
    np.testing.assert_array_equal(state["borderless"], np.asarray([False], dtype=bool))
    event_types = state["event_type"][0, : int(state["event_count"][0])]
    assert vector_runtime.EVENT_BORDERLESS not in set(event_types)


def test_step_many_runs_supported_fixture_row_against_scalar_array_step():
    fixture = seed_bridge.seed_fixture(
        "scenarios/environment/source_borderless_wrap_step.json",
        body_capacity=4,
    )
    initial_state = vector_compare.array_state_from_seed(fixture)
    expected_state = vector_compare.copy_array_state(initial_state)
    actual_state = vector_compare.copy_array_state(initial_state)
    prepared_step = vector_compare.prepare_fixture_array_step(fixture, step_index=0)
    expected_counters = vector_compare.step_prepared_arrays(expected_state, prepared_step)
    prepared_batch = {
        "player_count": prepared_step["player_count"],
        "step_ms": np.asarray([prepared_step["step_ms"]], dtype=np.float64),
        "source_moves": np.asarray([prepared_step["source_moves"]], dtype=np.int8),
        "print_manager_mode": np.asarray(
            [prepared_step.get("print_manager_mode", "none")],
            dtype=object,
        ),
        "timer_advance_ms": np.asarray(
            [prepared_step.get("timer_advance_ms", 0.0)],
            dtype=np.float64,
        ),
    }

    actual_counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput.from_mapping(actual_state, prepared_batch),
    )

    assert actual_counters == _expected_runtime_counters(expected_counters)
    for name, expected_array in expected_state.items():
        actual_array = actual_state[name]
        if np.issubdtype(expected_array.dtype, np.floating):
            np.testing.assert_allclose(actual_array, expected_array, rtol=0.0, atol=1e-12)
        else:
            np.testing.assert_array_equal(actual_array, expected_array)
