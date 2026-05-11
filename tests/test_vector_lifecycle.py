import json
import math
from pathlib import Path

import numpy as np

from curvyzero.env import (
    vector_autoreset,
    vector_lifecycle,
    vector_reset,
    vector_runtime,
    vector_trainer_observation,
)
from curvyzero.env.config import CurvyTronReferenceDefaults
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios" / "environment"


def _reset_spawn_template_and_target() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    template_tape = np.zeros((2, 8), dtype=np.float64)
    template_tape[0, :4] = [0.1, 0.2, 0.3, 0.4]
    template_tape[1, :7] = [
        0.0,
        0.5,
        0.5,
        0.25,
        0.6794871794871795,
        0.5,
        0.5159154943091895,
    ]

    reset_template = {
        "episode_id": np.asarray([100, 900], dtype=np.int64),
        "episode_step": np.asarray([0, 0], dtype=np.int32),
        "env_active": np.asarray([True, True], dtype=bool),
        "reset_pending": np.asarray([False, False], dtype=bool),
        "done": np.asarray([False, False], dtype=bool),
        "terminated": np.asarray([False, False], dtype=bool),
        "truncated": np.asarray([False, False], dtype=bool),
        "terminal_reason": np.asarray(
            [vector_reset.TERMINAL_REASON_NONE, vector_reset.TERMINAL_REASON_NONE],
            dtype=np.int16,
        ),
        "reset_seed": np.asarray([1001, 9001], dtype=np.uint64),
        "reset_source": np.asarray(
            [vector_reset.RESET_SOURCE_FIXTURE, vector_reset.RESET_SOURCE_FIXTURE],
            dtype=np.int16,
        ),
        "tick": np.asarray([0, 0], dtype=np.int32),
        "elapsed_ms": np.asarray([0.0, 0.0], dtype=np.float64),
        "pos": np.zeros((2, 2, 2), dtype=np.float64),
        "prev_pos": np.zeros((2, 2, 2), dtype=np.float64),
        "heading": np.zeros((2, 2), dtype=np.float64),
        "alive": np.zeros((2, 2), dtype=bool),
        "present": np.asarray([[True, False], [True, True]], dtype=bool),
        "map_size": np.asarray([88.0, 88.0], dtype=np.float64),
        "random_tape_values": template_tape,
        "random_tape_length": np.asarray([4, 7], dtype=np.int32),
        "random_tape_cursor": np.asarray([1, 0], dtype=np.int32),
        "random_tape_exhausted": np.asarray([False, True], dtype=bool),
        "random_tape_draw_count": np.asarray([1, 0], dtype=np.int32),
    }

    target = {name: array.copy() for name, array in reset_template.items()}
    target.update(
        {
            "episode_id": np.asarray([10, 20], dtype=np.int64),
            "episode_step": np.asarray([3, 9], dtype=np.int32),
            "env_active": np.asarray([True, False], dtype=bool),
            "reset_pending": np.asarray([False, True], dtype=bool),
            "done": np.asarray([False, True], dtype=bool),
            "terminated": np.asarray([False, True], dtype=bool),
            "truncated": np.asarray([False, False], dtype=bool),
            "terminal_reason": np.asarray(
                [
                    vector_reset.TERMINAL_REASON_NONE,
                    vector_reset.TERMINAL_REASON_SURVIVOR_WIN,
                ],
                dtype=np.int16,
            ),
            "reset_seed": np.asarray([111, 222], dtype=np.uint64),
            "reset_source": np.asarray(
                [vector_reset.RESET_SOURCE_REPLAY, vector_reset.RESET_SOURCE_MANUAL],
                dtype=np.int16,
            ),
            "tick": np.asarray([5, 41], dtype=np.int32),
            "elapsed_ms": np.asarray([16.0, 250.0], dtype=np.float64),
            "random_tape_length": np.asarray([6, 0], dtype=np.int32),
            "random_tape_cursor": np.asarray([5, 0], dtype=np.int32),
            "random_tape_exhausted": np.asarray([False, True], dtype=bool),
            "random_tape_draw_count": np.asarray([5, 13], dtype=np.int32),
        }
    )
    target["pos"][0] = [[1.0, 2.0], [3.0, 4.0]]
    target["pos"][1] = [[99.0, 98.0], [97.0, 96.0]]
    target["prev_pos"][:] = target["pos"]
    target["heading"][0] = [0.1, 0.2]
    target["heading"][1] = [9.0, 8.0]
    target["alive"][0] = [True, False]
    target["random_tape_values"][1] = 0.99
    return reset_template, target


def _add_optional_1v1_lifecycle_arrays(
    reset_template: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    timer_capacity: int = 4,
) -> None:
    for state in (reset_template, target):
        state.update(
            {
                "started": np.asarray([False, False], dtype=bool),
                "in_round": np.asarray([False, False], dtype=bool),
                "world_active": np.asarray([False, False], dtype=bool),
                "world_body_count": np.asarray([0, 0], dtype=np.int32),
                "timer_active": np.zeros((2, timer_capacity), dtype=bool),
                "timer_remaining_ms": np.zeros((2, timer_capacity), dtype=np.float64),
                "timer_kind": np.zeros((2, timer_capacity), dtype=np.int16),
                "timer_player": np.full((2, timer_capacity), -1, dtype=np.int16),
                "timer_seq": np.zeros((2, timer_capacity), dtype=np.int32),
                "timer_overflow": np.asarray([False, False], dtype=bool),
            }
        )

    target["started"][0] = True
    target["in_round"][0] = True
    target["world_active"][0] = True
    target["world_body_count"][0] = 17
    target["timer_active"][0, 2] = True
    target["timer_remaining_ms"][0, 2] = 123.0
    target["timer_kind"][0, 2] = vector_lifecycle.TIMER_KIND_PRINT_MANAGER_START
    target["timer_player"][0, 2] = 0
    target["timer_seq"][0, 2] = 9

    target["world_active"][1] = True
    target["world_body_count"][1] = 8
    target["timer_active"][1, 3] = True
    target["timer_remaining_ms"][1, 3] = 42.0
    target["timer_kind"][1, 3] = vector_lifecycle.TIMER_KIND_PRINT_MANAGER_START
    target["timer_player"][1, 3] = 1
    target["timer_seq"][1, 3] = 7


def _add_1v1_warmup_round_local_arrays(
    reset_template: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    body_capacity: int = 3,
) -> None:
    for state in (reset_template, target):
        state.update(
            {
                "score": np.asarray([[0, 0], [4, 3]], dtype=np.int32),
                "round_score": np.asarray([[0, 0], [9, 8]], dtype=np.int32),
                "printing": np.asarray([[True, True], [True, True]], dtype=bool),
                "print_manager_active": np.asarray([[True, True], [True, True]], dtype=bool),
                "print_manager_distance": np.asarray(
                    [[10.0, 20.0], [30.0, 40.0]],
                    dtype=np.float64,
                ),
                "print_manager_last_pos": np.arange(
                    2 * 2 * 2,
                    dtype=np.float64,
                ).reshape(2, 2, 2),
                "death_count": np.asarray([2, 2], dtype=np.int32),
                "death_player": np.asarray([[1, 0], [1, 0]], dtype=np.int16),
                "body_active": np.ones((2, body_capacity), dtype=bool),
                "body_pos": np.ones((2, body_capacity, 2), dtype=np.float64),
                "body_radius": np.ones((2, body_capacity), dtype=np.float64),
                "body_owner": np.ones((2, body_capacity), dtype=np.int16),
                "body_num": np.ones((2, body_capacity), dtype=np.int32),
                "body_insert_tick": np.ones((2, body_capacity), dtype=np.int32),
                "body_insert_kind": np.ones((2, body_capacity), dtype=np.int16),
                "body_write_cursor": np.asarray([body_capacity, body_capacity], dtype=np.int32),
                "body_count": np.asarray([[4, 5], [6, 7]], dtype=np.int32),
            }
        )


def _load_lifecycle_scenario(name: str) -> dict[str, object]:
    with (SCENARIO_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expected_spawn_random_calls(payload: dict[str, object]) -> list[tuple[int, str, int, float]]:
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


def _source_events(payload: dict[str, object]) -> list[str]:
    expectations = payload["expectations"]
    assert isinstance(expectations, dict)
    event_order = expectations["event_order"]
    assert isinstance(event_order, list)
    return [str(event["event"]) for event in event_order if isinstance(event, dict)]


def _source_score_triplets(payload: dict[str, object]) -> list[tuple[int, int, int]]:
    expectations = payload["expectations"]
    assert isinstance(expectations, dict)
    event_order = expectations["event_order"]
    assert isinstance(event_order, list)
    triplets: list[tuple[int, int, int]] = []
    for event in event_order:
        assert isinstance(event, dict)
        if event.get("event") != "score":
            continue
        data = event["data"]
        assert isinstance(data, dict)
        triplets.append(
            (
                int(data["avatar"]),
                int(data["score"]),
                int(data["roundScore"]),
            )
        )
    return triplets


def _source_room_max_score(payload: dict[str, object]) -> int:
    source_setup = payload["source_setup"]
    assert isinstance(source_setup, dict)
    room = source_setup["room"]
    assert isinstance(room, dict)
    return int(room["max_score"])


def _source_random_tape(payload: dict[str, object]) -> np.ndarray:
    source_setup = payload["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    return np.asarray([random_setup["math_random_sequence"]], dtype=np.float64)


def _warmup_template_and_target_from_lifecycle_payload(
    payload: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    player_count = int(payload["player_count"])
    source_setup = payload["source_setup"]
    assert isinstance(source_setup, dict)
    random_setup = source_setup["random"]
    assert isinstance(random_setup, dict)
    random_values = np.asarray(
        random_setup["math_random_sequence"],
        dtype=np.float64,
    )
    batch_size = 1
    timer_capacity = max(1, player_count)
    body_capacity = player_count
    map_size = float(CurvyTronReferenceDefaults().arena_size_for_players(player_count))

    reset_template = {
        "episode_id": np.asarray([100], dtype=np.int64),
        "episode_step": np.asarray([0], dtype=np.int32),
        "env_active": np.asarray([True], dtype=bool),
        "reset_pending": np.asarray([False], dtype=bool),
        "done": np.asarray([False], dtype=bool),
        "terminated": np.asarray([False], dtype=bool),
        "truncated": np.asarray([False], dtype=bool),
        "terminal_reason": np.asarray(
            [vector_reset.TERMINAL_REASON_NONE],
            dtype=np.int16,
        ),
        "reset_seed": np.asarray([1001], dtype=np.uint64),
        "reset_source": np.asarray([vector_reset.RESET_SOURCE_FIXTURE], dtype=np.int16),
        "tick": np.asarray([0], dtype=np.int32),
        "elapsed_ms": np.asarray([0.0], dtype=np.float64),
        "pos": np.zeros((batch_size, player_count, 2), dtype=np.float64),
        "prev_pos": np.zeros((batch_size, player_count, 2), dtype=np.float64),
        "heading": np.zeros((batch_size, player_count), dtype=np.float64),
        "alive": np.zeros((batch_size, player_count), dtype=bool),
        "present": np.ones((batch_size, player_count), dtype=bool),
        "map_size": np.asarray([map_size], dtype=np.float64),
        "random_tape_values": random_values.reshape(1, -1).copy(),
        "random_tape_length": np.asarray([random_values.size], dtype=np.int32),
        "random_tape_cursor": np.asarray([0], dtype=np.int32),
        "random_tape_exhausted": np.asarray([False], dtype=bool),
        "random_tape_draw_count": np.asarray([0], dtype=np.int32),
        "started": np.asarray([False], dtype=bool),
        "in_round": np.asarray([False], dtype=bool),
        "world_active": np.asarray([False], dtype=bool),
        "world_body_count": np.asarray([0], dtype=np.int32),
        "timer_active": np.zeros((batch_size, timer_capacity), dtype=bool),
        "timer_remaining_ms": np.zeros((batch_size, timer_capacity), dtype=np.float64),
        "timer_kind": np.zeros((batch_size, timer_capacity), dtype=np.int16),
        "timer_player": np.full(
            (batch_size, timer_capacity),
            vector_lifecycle.TIMER_PLAYER_NONE,
            dtype=np.int16,
        ),
        "timer_seq": np.zeros((batch_size, timer_capacity), dtype=np.int32),
        "timer_overflow": np.asarray([False], dtype=bool),
        "score": np.zeros((batch_size, player_count), dtype=np.int32),
        "round_score": np.full((batch_size, player_count), 9, dtype=np.int32),
        "printing": np.ones((batch_size, player_count), dtype=bool),
        "print_manager_active": np.ones((batch_size, player_count), dtype=bool),
        "print_manager_distance": np.full(
            (batch_size, player_count),
            17.0,
            dtype=np.float64,
        ),
        "print_manager_last_pos": np.ones(
            (batch_size, player_count, 2),
            dtype=np.float64,
        ),
        "death_count": np.asarray([player_count], dtype=np.int32),
        "death_player": np.arange(player_count, dtype=np.int16).reshape(1, -1),
        "body_active": np.ones((batch_size, body_capacity), dtype=bool),
        "body_pos": np.ones((batch_size, body_capacity, 2), dtype=np.float64),
        "body_radius": np.ones((batch_size, body_capacity), dtype=np.float64),
        "body_owner": np.ones((batch_size, body_capacity), dtype=np.int16),
        "body_num": np.ones((batch_size, body_capacity), dtype=np.int32),
        "body_insert_tick": np.ones((batch_size, body_capacity), dtype=np.int32),
        "body_insert_kind": np.ones((batch_size, body_capacity), dtype=np.int16),
        "body_write_cursor": np.asarray([body_capacity], dtype=np.int32),
        "body_count": np.ones((batch_size, player_count), dtype=np.int32),
    }
    target = {name: array.copy() for name, array in reset_template.items()}
    target["episode_id"][:] = 200
    target["done"][:] = True
    target["terminated"][:] = True
    target["terminal_reason"][:] = vector_reset.TERMINAL_REASON_SURVIVOR_WIN
    target["random_tape_cursor"][:] = random_values.size
    target["random_tape_draw_count"][:] = random_values.size
    target["timer_active"][0, -1] = True
    target["timer_remaining_ms"][0, -1] = 42.0
    target["timer_kind"][0, -1] = vector_lifecycle.TIMER_KIND_PRINT_MANAGER_START
    target["timer_player"][0, -1] = player_count - 1
    target["timer_seq"][0, -1] = 7
    return reset_template, target


def _start_3p_no_bonus_lifecycle_fixture(
    payload: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    reset_template, target = _warmup_template_and_target_from_lifecycle_payload(payload)
    _add_no_bonus_runtime_step_arrays(
        reset_template,
        target,
        player_count=3,
    )

    reset_info = vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        player_count=3,
        reset_seed=555,
        first_warmup_ms=0.0,
        snapshot_array_names=("done", "timer_active"),
    )
    warmup_info = vector_runtime.advance_warmup_no_bonus_timers(
        target,
        np.asarray([3000.0], dtype=np.float64),
        player_count=3,
    )
    return target, reset_info, warmup_info


def _step_3p_no_bonus_lifecycle(target: dict[str, np.ndarray]) -> dict[str, int]:
    return dict(
        vector_runtime.step_many(
            vector_runtime.VectorStepInput(
                state=target,
                step_ms=np.asarray([100.0], dtype=np.float64),
                source_moves=np.zeros((1, 3), dtype=np.int8),
                player_count=3,
                print_manager_mode=np.asarray(["death_stop"], dtype=object),
                event_mode=vector_runtime.EVENT_MODE_NONE,
            )
        )
    )


def _slice_state_rows(
    state: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: array[rows, ...].copy() for name, array in state.items()}


def _one_row_wall_loop_template_and_target() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    reset_template, target = _reset_spawn_template_and_target()
    _add_optional_1v1_lifecycle_arrays(reset_template, target)
    _add_1v1_warmup_round_local_arrays(reset_template, target, body_capacity=6)

    selected_rows = np.asarray([1], dtype=np.int64)
    reset_template = _slice_state_rows(reset_template, selected_rows)
    target = _slice_state_rows(target, selected_rows)
    _add_1v1_no_bonus_runtime_step_arrays(reset_template, target)

    spawn_and_print_tape = np.zeros((1, 10), dtype=np.float64)
    spawn_and_print_tape[0, :10] = [
        0.0,
        0.5,
        0.5,
        0.25,
        0.6794871794871795,
        0.5,
        0.5159154943091895,
        0.25,
        0.5,
        0.5,
    ]
    for state in (reset_template, target):
        state["random_tape_values"] = spawn_and_print_tape.copy()
        state["random_tape_length"] = np.asarray([10], dtype=np.int32)
        state["random_tape_cursor"] = np.asarray([0], dtype=np.int32)
        state["random_tape_draw_count"] = np.asarray([0], dtype=np.int32)
        state["random_tape_exhausted"] = np.asarray([False], dtype=bool)

    return reset_template, target


def _add_1v1_no_bonus_runtime_step_arrays(
    reset_template: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    event_capacity: int = 16,
) -> None:
    batch_size = target["episode_id"].shape[0]
    player_count = 2
    for state in (reset_template, target):
        state.update(
            {
                "overflow": np.zeros(batch_size, dtype=bool),
                "borderless": np.zeros(batch_size, dtype=bool),
                "radius": np.full((batch_size, player_count), 0.6, dtype=np.float64),
                "speed": np.asarray([[0.0, 200.0]], dtype=np.float64).repeat(
                    batch_size,
                    axis=0,
                ),
                "angular_velocity_per_ms": np.zeros(
                    (batch_size, player_count),
                    dtype=np.float64,
                ),
                "live_body_num": np.zeros((batch_size, player_count), dtype=np.int32),
                "trail_latency": np.full(
                    (batch_size, player_count),
                    3,
                    dtype=np.int16,
                ),
                "death_tick": np.full(
                    (batch_size, player_count),
                    -1,
                    dtype=np.int32,
                ),
                "draw": np.ones(batch_size, dtype=bool),
                "winner": np.full(batch_size, -1, dtype=np.int16),
                "body_overflow": np.zeros(batch_size, dtype=bool),
                "visible_trail_count": np.zeros(
                    (batch_size, player_count),
                    dtype=np.int32,
                ),
                "has_visible_trail_last": np.zeros(
                    (batch_size, player_count),
                    dtype=bool,
                ),
                "visible_trail_last_pos": np.zeros(
                    (batch_size, player_count, 2),
                    dtype=np.float64,
                ),
                "has_draw_cursor": np.zeros((batch_size, player_count), dtype=bool),
                "draw_cursor_pos": np.zeros(
                    (batch_size, player_count, 2),
                    dtype=np.float64,
                ),
                "event_count": np.zeros(batch_size, dtype=np.int16),
                "event_mask": np.zeros((batch_size, event_capacity), dtype=bool),
                "event_type": np.zeros((batch_size, event_capacity), dtype=np.int16),
                "event_player": np.full(
                    (batch_size, event_capacity),
                    -1,
                    dtype=np.int16,
                ),
                "event_other": np.full(
                    (batch_size, event_capacity),
                    -1,
                    dtype=np.int16,
                ),
                "event_bool": np.full(
                    (batch_size, event_capacity),
                    -1,
                    dtype=np.int8,
                ),
                "event_value_i": np.zeros(
                    (batch_size, event_capacity, 2),
                    dtype=np.int32,
                ),
                "event_value_f": np.zeros(
                    (batch_size, event_capacity, 2),
                    dtype=np.float64,
                ),
                "event_overflow": np.zeros(batch_size, dtype=bool),
                "event_overflow_attempts": np.zeros(batch_size, dtype=np.int32),
            }
        )


def _add_no_bonus_runtime_step_arrays(
    reset_template: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    *,
    player_count: int,
    body_capacity: int = 24,
    event_capacity: int = 64,
) -> None:
    batch_size = target["episode_id"].shape[0]
    for state in (reset_template, target):
        state.update(
            {
                "overflow": np.zeros(batch_size, dtype=bool),
                "borderless": np.zeros(batch_size, dtype=bool),
                "radius": np.full(
                    (batch_size, player_count),
                    0.6,
                    dtype=np.float64,
                ),
                "speed": np.full(
                    (batch_size, player_count),
                    16.0,
                    dtype=np.float64,
                ),
                "angular_velocity_per_ms": np.zeros(
                    (batch_size, player_count),
                    dtype=np.float64,
                ),
                "live_body_num": np.zeros(
                    (batch_size, player_count),
                    dtype=np.int32,
                ),
                "trail_latency": np.full(
                    (batch_size, player_count),
                    3,
                    dtype=np.int16,
                ),
                "death_tick": np.full(
                    (batch_size, player_count),
                    -1,
                    dtype=np.int32,
                ),
                "draw": np.zeros(batch_size, dtype=bool),
                "winner": np.full(batch_size, -1, dtype=np.int16),
                "round_done": np.zeros(batch_size, dtype=bool),
                "warmdown_pending": np.zeros(batch_size, dtype=bool),
                "match_done": np.zeros(batch_size, dtype=bool),
                "round_winner": np.full(batch_size, -1, dtype=np.int16),
                "match_winner": np.full(batch_size, -1, dtype=np.int16),
                "max_score": np.full(batch_size, 10, dtype=np.int32),
                "body_active": np.zeros((batch_size, body_capacity), dtype=bool),
                "body_pos": np.zeros((batch_size, body_capacity, 2), dtype=np.float64),
                "body_radius": np.zeros((batch_size, body_capacity), dtype=np.float64),
                "body_owner": np.full((batch_size, body_capacity), -1, dtype=np.int16),
                "body_num": np.full((batch_size, body_capacity), -1, dtype=np.int32),
                "body_insert_tick": np.full(
                    (batch_size, body_capacity),
                    -1,
                    dtype=np.int32,
                ),
                "body_insert_kind": np.full(
                    (batch_size, body_capacity),
                    -1,
                    dtype=np.int16,
                ),
                "body_write_cursor": np.zeros(batch_size, dtype=np.int32),
                "body_count": np.zeros((batch_size, player_count), dtype=np.int32),
                "body_overflow": np.zeros(batch_size, dtype=bool),
                "visible_trail_count": np.zeros(
                    (batch_size, player_count),
                    dtype=np.int32,
                ),
                "has_visible_trail_last": np.zeros(
                    (batch_size, player_count),
                    dtype=bool,
                ),
                "visible_trail_last_pos": np.zeros(
                    (batch_size, player_count, 2),
                    dtype=np.float64,
                ),
                "has_draw_cursor": np.zeros((batch_size, player_count), dtype=bool),
                "draw_cursor_pos": np.zeros(
                    (batch_size, player_count, 2),
                    dtype=np.float64,
                ),
                "event_count": np.zeros(batch_size, dtype=np.int16),
                "event_mask": np.zeros((batch_size, event_capacity), dtype=bool),
                "event_type": np.zeros((batch_size, event_capacity), dtype=np.int16),
                "event_player": np.full(
                    (batch_size, event_capacity),
                    -1,
                    dtype=np.int16,
                ),
                "event_other": np.full(
                    (batch_size, event_capacity),
                    -1,
                    dtype=np.int16,
                ),
                "event_bool": np.full(
                    (batch_size, event_capacity),
                    -1,
                    dtype=np.int8,
                ),
                "event_value_i": np.zeros(
                    (batch_size, event_capacity, 2),
                    dtype=np.int32,
                ),
                "event_value_f": np.zeros(
                    (batch_size, event_capacity, 2),
                    dtype=np.float64,
                ),
                "event_overflow": np.zeros(batch_size, dtype=bool),
                "event_overflow_attempts": np.zeros(batch_size, dtype=np.int32),
            }
        )


def test_reset_and_spawn_round_rows_resets_and_spawns_selected_rows_from_template_tape():
    reset_template, target = _reset_spawn_template_and_target()

    info = vector_lifecycle.reset_and_spawn_round_rows(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        player_count=2,
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=("done", "terminal_reason", "pos"),
    )

    assert info["schema"] == vector_lifecycle.RESET_SPAWN_INFO_SCHEMA_ID
    assert info["surface"] == vector_lifecycle.RESET_SPAWN_SURFACE
    assert info["full_lifecycle"] is False
    assert info["can_compose"] is True
    assert info["reset_count"] == 1
    assert info["spawn_count"] == 1
    np.testing.assert_array_equal(info["reset_rows"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(info["spawn_rows"], np.asarray([1], dtype=np.int32))

    assert int(target["episode_id"][1]) == 21
    assert int(target["episode_step"][1]) == 0
    assert bool(target["env_active"][1]) is True
    assert bool(target["reset_pending"][1]) is False
    assert bool(target["done"][1]) is False
    assert bool(target["terminated"][1]) is False
    assert int(target["terminal_reason"][1]) == vector_reset.TERMINAL_REASON_NONE
    assert int(target["reset_seed"][1]) == 555
    assert int(target["reset_source"][1]) == vector_reset.RESET_SOURCE_AUTORESET

    np.testing.assert_allclose(target["pos"][1], [[58.0, 44.0], [5.0, 44.0]])
    np.testing.assert_allclose(target["prev_pos"][1], target["pos"][1])
    np.testing.assert_allclose(
        target["heading"][1],
        [0.5159154943091895 * math.tau, 0.25 * math.tau],
    )
    np.testing.assert_array_equal(target["alive"][1], np.asarray([True, True], dtype=bool))
    assert int(target["random_tape_cursor"][1]) == 7
    assert int(target["random_tape_draw_count"][1]) == 7
    assert bool(target["random_tape_exhausted"][1]) is False
    np.testing.assert_array_equal(
        info["random_draw_count_delta"],
        np.asarray([0, 7], dtype=np.int32),
    )
    assert info["scheduled_timer_count"] == 0
    timer_info = info["lifecycle_schedule_info"]["delayed_start_timers"]
    assert timer_info["missing_arrays"] == list(
        vector_lifecycle._OPTIONAL_DELAYED_START_TIMER_ARRAYS
    )
    assert timer_info["unsupported_reasons"] == [
        "missing_delayed_start_timer_arrays"
    ]


def test_reset_and_spawn_round_rows_schedules_optional_1v1_delayed_start_timers():
    reset_template, target = _reset_spawn_template_and_target()
    _add_optional_1v1_lifecycle_arrays(reset_template, target)
    skipped_before = {
        name: target[name][0].copy()
        for name in (
            "started",
            "in_round",
            "world_active",
            "world_body_count",
            "timer_active",
            "timer_remaining_ms",
            "timer_kind",
            "timer_player",
            "timer_seq",
            "timer_overflow",
        )
    }

    info = vector_lifecycle.reset_and_spawn_round_rows(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        player_count=2,
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=("done",),
    )

    schedule_info = info["lifecycle_schedule_info"]
    assert schedule_info["surface"] == vector_lifecycle.RESET_SPAWN_LIFECYCLE_SURFACE
    assert schedule_info["full_lifecycle"] is False
    assert info["full_lifecycle"] is False
    assert info["scheduled_timer_count"] == 2

    world_flags = schedule_info["world_flags"]
    assert world_flags["missing_arrays"] == []
    assert world_flags["unsupported_reasons"] == []
    assert world_flags["applied"] is True
    assert world_flags["applied_arrays"] == [
        "started",
        "in_round",
        "world_active",
        "world_body_count",
    ]
    np.testing.assert_array_equal(world_flags["rows"], np.asarray([1], dtype=np.int32))
    assert bool(target["started"][1]) is True
    assert bool(target["in_round"][1]) is True
    assert bool(target["world_active"][1]) is False
    assert int(target["world_body_count"][1]) == 0

    timers = schedule_info["delayed_start_timers"]
    assert timers["missing_arrays"] == []
    assert timers["unsupported_reasons"] == []
    assert timers["scheduled"] is True
    assert timers["scheduled_timer_count"] == 2
    assert timers["timer_kind_code"] == vector_lifecycle.TIMER_KIND_PRINT_MANAGER_START
    assert timers["delay_ms"] == vector_lifecycle.SOURCE_TRAIL_START_DELAY_MS
    np.testing.assert_array_equal(
        timers["scheduled_timer_rows"],
        np.asarray([1, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        timers["scheduled_timer_slots"],
        np.asarray([0, 1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        timers["scheduled_timer_players"],
        np.asarray([1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        timers["timer_overflow_rows"],
        np.asarray([], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        target["timer_active"][1],
        np.asarray([True, True, False, False], dtype=bool),
    )
    np.testing.assert_allclose(
        target["timer_remaining_ms"][1],
        [3000.0, 3000.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        target["timer_kind"][1],
        np.asarray([1, 1, 0, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        target["timer_player"][1],
        np.asarray([1, 0, -1, -1], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        target["timer_seq"][1],
        np.asarray([0, 1, 0, 0], dtype=np.int32),
    )
    assert bool(target["timer_overflow"][1]) is False

    for name, before in skipped_before.items():
        np.testing.assert_array_equal(target[name][0], before)


def test_reset_and_spawn_round_rows_preserves_terminal_snapshot_through_spawn():
    reset_template, target = _reset_spawn_template_and_target()

    info = vector_lifecycle.reset_and_spawn_round_rows(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        player_count=2,
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=(
            "done",
            "terminal_reason",
            "pos",
            "random_tape_cursor",
            "random_tape_draw_count",
        ),
    )

    snapshot = info["terminal_transition_snapshot"]
    assert snapshot is info["reset_info"]["terminal_transition_snapshot"]
    np.testing.assert_array_equal(snapshot["final_rows"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(snapshot["arrays"]["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        snapshot["arrays"]["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_allclose(snapshot["arrays"]["pos"][0], [[99.0, 98.0], [97.0, 96.0]])
    np.testing.assert_array_equal(
        snapshot["arrays"]["random_tape_cursor"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["random_tape_draw_count"],
        np.asarray([13], dtype=np.int32),
    )

    target["pos"][1, 0] = [-1.0, -1.0]
    np.testing.assert_allclose(snapshot["arrays"]["pos"][0], [[99.0, 98.0], [97.0, 96.0]])


def test_reset_and_spawn_round_rows_snapshots_before_spawn_and_lifecycle_schedule():
    reset_template, target = _reset_spawn_template_and_target()
    _add_optional_1v1_lifecycle_arrays(reset_template, target)

    info = vector_lifecycle.reset_and_spawn_round_rows(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        player_count=2,
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=(
            "pos",
            "random_tape_cursor",
            "world_active",
            "timer_active",
            "timer_remaining_ms",
        ),
    )

    snapshot = info["terminal_transition_snapshot"]
    np.testing.assert_allclose(snapshot["arrays"]["pos"][0], [[99.0, 98.0], [97.0, 96.0]])
    np.testing.assert_array_equal(
        snapshot["arrays"]["random_tape_cursor"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["world_active"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["timer_active"][0],
        np.asarray([False, False, False, True], dtype=bool),
    )
    np.testing.assert_allclose(
        snapshot["arrays"]["timer_remaining_ms"][0],
        [0.0, 0.0, 0.0, 42.0],
    )

    np.testing.assert_allclose(target["pos"][1], [[58.0, 44.0], [5.0, 44.0]])
    assert bool(target["world_active"][1]) is False
    np.testing.assert_array_equal(
        target["timer_active"][1],
        np.asarray([True, True, False, False], dtype=bool),
    )


def test_reset_and_spawn_round_rows_leaves_skipped_rows_and_cursors_untouched():
    reset_template, target = _reset_spawn_template_and_target()
    skipped_before = {name: array[0].copy() for name, array in target.items()}

    vector_lifecycle.reset_and_spawn_round_rows(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        player_count=2,
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=("done",),
    )

    for name, before in skipped_before.items():
        np.testing.assert_array_equal(target[name][0], before)
    assert int(target["random_tape_cursor"][0]) == 5
    assert int(target["random_tape_draw_count"][0]) == 5


def test_reset_and_spawn_round_rows_returns_exact_missing_arrays_without_mutating():
    reset_template, target = _reset_spawn_template_and_target()
    target.pop("random_tape_values")
    target["target_only_debug"] = np.asarray([1, 2], dtype=np.int32)
    reset_template["template_only_debug"] = np.asarray([3, 4], dtype=np.int32)
    target_before = {name: array.copy() for name, array in target.items()}

    info = vector_lifecycle.reset_and_spawn_round_rows(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        player_count=2,
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=("done",),
    )

    assert info["can_compose"] is False
    assert info["reset_info"] is None
    assert info["spawn_info"] is None
    assert info["terminal_transition_snapshot"] is None
    assert info["missing_target_arrays"] == ["random_tape_values", "template_only_debug"]
    assert info["missing_reset_template_arrays"] == ["target_only_debug"]
    assert info["target_only_arrays"] == ["target_only_debug"]
    assert info["reset_template_only_arrays"] == [
        "random_tape_values",
        "template_only_debug",
    ]
    assert info["missing_target_required_arrays"] == ["random_tape_values"]
    assert info["missing_reset_template_required_arrays"] == []

    for name, before in target_before.items():
        np.testing.assert_array_equal(target[name], before)


def test_reset_spawn_warmup_1v1_no_bonus_schedules_game_start_only():
    reset_template, target = _reset_spawn_template_and_target()
    _add_optional_1v1_lifecycle_arrays(reset_template, target)
    _add_1v1_warmup_round_local_arrays(reset_template, target)
    skipped_before = {
        name: target[name][0].copy()
        for name in (
            "started",
            "in_round",
            "world_active",
            "world_body_count",
            "timer_active",
            "timer_remaining_ms",
            "timer_kind",
            "timer_player",
            "timer_seq",
            "timer_overflow",
            "printing",
            "print_manager_active",
            "death_count",
            "body_active",
            "body_write_cursor",
        )
    }

    info = vector_lifecycle.reset_spawn_warmup_1v1_no_bonus_rows(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=("done", "world_active", "timer_active"),
    )

    assert info["schema"] == vector_lifecycle.RESET_SPAWN_WARMUP_INFO_SCHEMA_ID
    assert info["surface"] == vector_lifecycle.RESET_SPAWN_WARMUP_SURFACE
    assert info["full_lifecycle"] is False
    assert info["can_compose"] is True
    assert info["reset_count"] == 1
    assert info["spawn_count"] == 1
    assert info["scheduled_timer_count"] == 1
    assert info["scheduled_timer_kind"] == "game:start"
    assert info["scheduled_timer_kind_code"] == vector_lifecycle.TIMER_KIND_GAME_START
    np.testing.assert_array_equal(
        info["scheduled_timer_rows"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        info["scheduled_timer_slots"],
        np.asarray([0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        info["timer_overflow_rows"],
        np.asarray([], dtype=np.int32),
    )

    np.testing.assert_allclose(target["pos"][1], [[58.0, 44.0], [5.0, 44.0]])
    np.testing.assert_array_equal(target["alive"][1], np.asarray([True, True], dtype=bool))
    assert bool(target["started"][1]) is True
    assert bool(target["in_round"][1]) is True
    assert bool(target["world_active"][1]) is False
    assert int(target["world_body_count"][1]) == 0
    np.testing.assert_array_equal(
        target["timer_active"][1],
        np.asarray([True, False, False, False], dtype=bool),
    )
    np.testing.assert_allclose(target["timer_remaining_ms"][1], [3000.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        target["timer_kind"][1],
        np.asarray([vector_lifecycle.TIMER_KIND_GAME_START, 0, 0, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        target["timer_player"][1],
        np.asarray([-1, -1, -1, -1], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["timer_seq"][1], np.asarray([0, 0, 0, 0]))
    assert bool(target["timer_overflow"][1]) is False

    np.testing.assert_array_equal(target["printing"][1], np.asarray([False, False]))
    np.testing.assert_array_equal(
        target["print_manager_active"][1],
        np.asarray([False, False]),
    )
    np.testing.assert_allclose(target["print_manager_distance"][1], [0.0, 0.0])
    np.testing.assert_allclose(target["print_manager_last_pos"][1], np.zeros((2, 2)))
    assert int(target["death_count"][1]) == 0
    np.testing.assert_array_equal(target["death_player"][1], np.asarray([-1, -1]))
    np.testing.assert_array_equal(target["round_score"][1], np.asarray([0, 0]))
    np.testing.assert_array_equal(target["body_active"][1], np.asarray([False, False, False]))
    np.testing.assert_allclose(target["body_pos"][1], np.zeros((3, 2)))
    assert int(target["body_write_cursor"][1]) == 0
    np.testing.assert_array_equal(target["body_count"][1], np.asarray([0, 0]))

    snapshot = info["terminal_transition_snapshot"]
    np.testing.assert_array_equal(snapshot["final_rows"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        snapshot["arrays"]["world_active"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        snapshot["arrays"]["timer_active"][0],
        np.asarray([False, False, False, True], dtype=bool),
    )

    for name, before in skipped_before.items():
        np.testing.assert_array_equal(target[name][0], before)


def test_reset_spawn_warmup_no_bonus_rows_matches_3p_spawn_rng_fixture():
    payload = _load_lifecycle_scenario(
        "source_lifecycle_spawn_rng_warmup_print_start_3p.json"
    )
    reset_template, target = _warmup_template_and_target_from_lifecycle_payload(payload)

    info = vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        player_count=3,
        reset_seed=555,
        first_warmup_ms=0.0,
        snapshot_array_names=("done", "timer_active"),
    )

    assert info["schema"] == vector_lifecycle.RESET_SPAWN_WARMUP_NO_BONUS_INFO_SCHEMA_ID
    assert info["surface"] == vector_lifecycle.RESET_SPAWN_WARMUP_NO_BONUS_SURFACE
    assert info["player_count"] == 3
    assert info["can_compose"] is True
    assert info["scheduled_timer_count"] == 1
    assert info["scheduled_timer_kind"] == "game:start"
    np.testing.assert_array_equal(info["scheduled_timer_rows"], np.asarray([0], dtype=np.int32))
    np.testing.assert_allclose(target["timer_remaining_ms"][0], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        target["timer_kind"][0],
        np.asarray([vector_lifecycle.TIMER_KIND_GAME_START, 0, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, True, True]]))
    np.testing.assert_allclose(
        target["pos"][0],
        [[68.575, 47.5], [30.0, 47.5], [26.425, 47.5]],
    )
    np.testing.assert_array_equal(target["printing"], np.zeros((1, 3), dtype=bool))
    np.testing.assert_array_equal(
        target["print_manager_active"],
        np.zeros((1, 3), dtype=bool),
    )
    assert int(target["death_count"][0]) == 0
    np.testing.assert_array_equal(target["death_player"][0], np.asarray([-1, -1, -1]))
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in info["spawn_info"]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[:9]
    assert int(target["random_tape_cursor"][0]) == 9
    assert int(target["random_tape_draw_count"][0]) == 9


def test_reset_spawn_warmup_no_bonus_rows_matches_4p_spawn_rng_fixture():
    payload = _load_lifecycle_scenario("source_lifecycle_spawn_rng_order_4p.json")
    reset_template, target = _warmup_template_and_target_from_lifecycle_payload(payload)

    info = vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        player_count=4,
        reset_seed=555,
        first_warmup_ms=0.0,
        snapshot_array_names=("done", "timer_active"),
    )

    assert info["schema"] == vector_lifecycle.RESET_SPAWN_WARMUP_NO_BONUS_INFO_SCHEMA_ID
    assert info["surface"] == vector_lifecycle.RESET_SPAWN_WARMUP_NO_BONUS_SURFACE
    assert info["player_count"] == 4
    assert info["can_compose"] is True
    assert info["scheduled_timer_count"] == 1
    np.testing.assert_array_equal(
        target["timer_kind"][0],
        np.asarray([vector_lifecycle.TIMER_KIND_GAME_START, 0, 0, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, True, True, True]]))
    np.testing.assert_allclose(
        target["pos"][0],
        [[77.41, 50.5], [59.47, 50.5], [41.53, 50.5], [23.59, 50.5]],
    )
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in info["spawn_info"]["random_calls"]
    ] == _expected_spawn_random_calls(payload)
    assert int(target["random_tape_cursor"][0]) == 12
    assert int(target["random_tape_draw_count"][0]) == 12


def test_no_bonus_3p_round_end_warmdown_spawns_next_round_from_source_fixture_rng():
    payload = _load_lifecycle_scenario("source_lifecycle_spawn_rng_3p_next_round.json")
    reset_template, target = _warmup_template_and_target_from_lifecycle_payload(payload)
    _add_no_bonus_runtime_step_arrays(
        reset_template,
        target,
        player_count=3,
    )

    reset_info = vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        player_count=3,
        reset_seed=555,
        first_warmup_ms=0.0,
        snapshot_array_names=("done", "timer_active"),
    )
    warmup_info = vector_runtime.advance_warmup_no_bonus_timers(
        target,
        np.asarray([3000.0], dtype=np.float64),
        player_count=3,
    )
    assert reset_info["scheduled_timer_kind"] == "game:start"
    assert warmup_info["game_start_fires"] == 1
    np.testing.assert_array_equal(
        warmup_info["print_manager_start_players"],
        np.asarray([2, 1, 0], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 12

    target["pos"][0] = np.asarray(
        [[47.5, 1.0], [1.0, 47.5], [93.0, 47.5]],
        dtype=np.float64,
    )
    target["heading"][0] = np.asarray(
        [math.tau * 0.75, math.pi, 0.0],
        dtype=np.float64,
    )
    target["prev_pos"][0] = target["pos"][0]

    step_counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=target,
            step_ms=np.asarray([100.0], dtype=np.float64),
            source_moves=np.zeros((1, 3), dtype=np.int8),
            player_count=3,
            print_manager_mode=np.asarray(["death_stop"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )

    assert step_counters["normal_wall_deaths"] == 3
    assert step_counters["terminal_score_rows"] == 1
    assert step_counters["print_manager_death_stops"] == 3
    assert step_counters["random_tape_draws"] == 3
    np.testing.assert_array_equal(target["alive"], np.asarray([[False, False, False]]))
    np.testing.assert_array_equal(target["score"], np.asarray([[0, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(target["match_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["round_winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(target["death_count"], np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[2, 1, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        target["timer_kind"],
        np.asarray([[vector_lifecycle.TIMER_KIND_WARMDOWN_END, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_allclose(target["timer_remaining_ms"][0], [5000.0, 0.0, 0.0])
    assert int(target["random_tape_cursor"][0]) == 15

    warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=3,
    )

    assert warmdown_info["schema"] == (
        vector_lifecycle.WARMDOWN_ADVANCE_NO_BONUS_INFO_SCHEMA_ID
    )
    assert warmdown_info["surface"] == vector_lifecycle.WARMDOWN_ADVANCE_NO_BONUS_SURFACE
    assert warmdown_info["warmdown_end_fires"] == 1
    assert warmdown_info["game_stop_fires"] == 1
    assert warmdown_info["next_round_count"] == 1
    assert warmdown_info["match_end_count"] == 0
    assert warmdown_info["random_tape_draws"] == 9
    np.testing.assert_array_equal(
        warmdown_info["warmdown_end_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_info["next_round_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_info["scheduled_timer_rows"],
        np.asarray([0], dtype=np.int32),
    )
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in warmdown_info["spawn_infos"][0]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[9:]

    np.testing.assert_array_equal(target["alive"], np.asarray([[True, True, True]]))
    np.testing.assert_array_equal(target["round_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(target["match_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["death_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(target["death_player"], np.asarray([[-1, -1, -1]]))
    np.testing.assert_allclose(
        target["pos"][0],
        [[68.575, 47.5], [47.5, 47.5], [26.425, 47.5]],
    )
    np.testing.assert_allclose(
        target["heading"][0],
        [math.tau * 0.75, math.pi, math.tau * 0.25],
    )
    np.testing.assert_array_equal(
        target["timer_kind"],
        np.asarray([[vector_lifecycle.TIMER_KIND_GAME_START, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_allclose(target["timer_remaining_ms"][0], [3000.0, 0.0, 0.0])
    assert int(target["random_tape_cursor"][0]) == 24
    assert int(target["random_tape_draw_count"][0]) == 24


def test_no_bonus_3p_present_absent_draw_warmdown_resizes_next_round_to_present_arena():
    payload = _load_lifecycle_scenario(
        "source_lifecycle_present_absent_3p_next_round.json"
    )
    reset_template, target = _warmup_template_and_target_from_lifecycle_payload(payload)
    _add_no_bonus_runtime_step_arrays(
        reset_template,
        target,
        player_count=3,
    )

    reference = CurvyTronReferenceDefaults()
    present_mask = np.asarray([[True, False, True]], dtype=bool)
    for state in (reset_template, target):
        state["present"][:] = present_mask
        state["pos"][0, 1] = [reference.avatar_radius, reference.avatar_radius]
        state["prev_pos"][0, 1] = state["pos"][0, 1]

    reset_info = vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        player_count=3,
        reset_seed=555,
        first_warmup_ms=0.0,
        snapshot_array_names=("done", "timer_active"),
    )

    assert float(target["map_size"][0]) == reference.arena_size_for_players(3)
    np.testing.assert_array_equal(target["present"], present_mask)
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, False, True]]))
    np.testing.assert_array_equal(target["death_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_allclose(
        target["pos"][0],
        [[68.575, 47.5], [0.6, 0.6], [26.425, 47.5]],
    )
    np.testing.assert_allclose(target["heading"][0], [math.pi, 0.0, 0.1])
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in reset_info["spawn_info"]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[:6]
    assert int(target["random_tape_cursor"][0]) == 6

    warmup_info = vector_runtime.advance_warmup_no_bonus_timers(
        target,
        np.asarray([3000.0], dtype=np.float64),
        player_count=3,
    )

    assert warmup_info["game_start_fires"] == 1
    np.testing.assert_array_equal(
        warmup_info["print_manager_start_players"],
        np.asarray([2, 1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        target["printing"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        target["print_manager_active"],
        np.asarray([[True, True, True]], dtype=bool),
    )
    assert int(target["random_tape_cursor"][0]) == 9

    target["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    target["heading"][0, 2] = 0.0
    target["prev_pos"][0, 2] = target["pos"][0, 2]
    target["pos"][0, 0] = np.asarray([1.0, 47.5], dtype=np.float64)
    target["heading"][0, 0] = math.pi
    target["prev_pos"][0, 0] = target["pos"][0, 0]

    step_counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=target,
            step_ms=np.asarray([100.0], dtype=np.float64),
            source_moves=np.zeros((1, 3), dtype=np.int8),
            player_count=3,
            print_manager_mode=np.asarray(["death_stop"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )

    assert step_counters["normal_wall_deaths"] == 2
    assert step_counters["terminal_score_rows"] == 1
    assert step_counters["print_manager_death_stops"] == 2
    assert step_counters["random_tape_draws"] == 2
    np.testing.assert_array_equal(target["alive"], np.asarray([[False, False, False]]))
    np.testing.assert_array_equal(target["score"], np.asarray([[1, 0, 1]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_score"], np.zeros((1, 3), dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        target["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["round_winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(target["death_count"], np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[1, 2, 0]], dtype=np.int16),
    )
    assert float(target["map_size"][0]) == reference.arena_size_for_players(3)
    assert int(target["random_tape_cursor"][0]) == 11

    warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=3,
    )

    assert warmdown_info["warmdown_end_fires"] == 1
    assert warmdown_info["game_stop_fires"] == 1
    assert warmdown_info["next_round_count"] == 1
    assert warmdown_info["match_end_count"] == 0
    assert warmdown_info["round_clear_print_manager_stops"] == 0
    assert warmdown_info["random_tape_draws"] == 6
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in warmdown_info["spawn_infos"][0]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[6:]

    assert float(target["map_size"][0]) == reference.arena_size_for_players(2)
    np.testing.assert_array_equal(target["present"], present_mask)
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, False, True]]))
    np.testing.assert_array_equal(target["death_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["score"], np.asarray([[1, 0, 1]], dtype=np.int32))
    np.testing.assert_allclose(
        target["pos"][0],
        [[63.5, 44.0], [0.6, 0.6], [24.5, 44.0]],
    )
    np.testing.assert_allclose(
        target["heading"][0],
        [math.tau * 0.75, 0.0, math.tau * 0.25],
    )
    np.testing.assert_array_equal(
        target["printing"],
        np.asarray([[False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        target["print_manager_active"],
        np.asarray([[False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        target["timer_kind"],
        np.asarray([[vector_lifecycle.TIMER_KIND_GAME_START, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_allclose(target["timer_remaining_ms"][0], [3000.0, 0.0, 0.0])
    assert int(target["random_tape_cursor"][0]) == 17
    assert int(target["random_tape_draw_count"][0]) == 17


def test_no_bonus_3p_survivor_warmdown_death_does_not_rescore_before_next_round():
    payload = _load_lifecycle_scenario("source_lifecycle_survivor_score_3p_next_round.json")
    reset_template, target = _warmup_template_and_target_from_lifecycle_payload(payload)
    _add_no_bonus_runtime_step_arrays(
        reset_template,
        target,
        player_count=3,
    )

    vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        player_count=3,
        reset_seed=555,
        first_warmup_ms=0.0,
        snapshot_array_names=("done", "timer_active"),
    )
    vector_runtime.advance_warmup_no_bonus_timers(
        target,
        np.asarray([3000.0], dtype=np.float64),
        player_count=3,
    )
    assert int(target["random_tape_cursor"][0]) == 12

    target["pos"][0] = np.asarray(
        [[47.5, 47.5], [1.0, 47.5], [93.0, 47.5]],
        dtype=np.float64,
    )
    target["heading"][0] = np.asarray([0.0, math.pi, 0.0], dtype=np.float64)
    target["prev_pos"][0] = target["pos"][0]

    round_end_counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=target,
            step_ms=np.asarray([100.0], dtype=np.float64),
            source_moves=np.zeros((1, 3), dtype=np.int8),
            player_count=3,
            print_manager_mode=np.asarray(["death_stop"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )

    assert round_end_counters["normal_wall_deaths"] == 2
    assert round_end_counters["terminal_score_rows"] == 1
    assert round_end_counters["print_manager_death_stops"] == 2
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, False, False]]))
    np.testing.assert_array_equal(target["score"], np.asarray([[2, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_score"], np.zeros((1, 3), dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(
        target["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["round_winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(target["death_count"], np.asarray([2], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[2, 1, -1]], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 14

    target["pos"][0, 0] = np.asarray([93.0, 47.5], dtype=np.float64)
    target["heading"][0, 0] = 0.0
    target["prev_pos"][0, 0] = target["pos"][0, 0]

    warmdown_death_counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=target,
            step_ms=np.asarray([100.0], dtype=np.float64),
            source_moves=np.zeros((1, 3), dtype=np.int8),
            player_count=3,
            print_manager_mode=np.asarray(["death_stop"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )

    assert warmdown_death_counters["normal_wall_deaths"] == 1
    assert warmdown_death_counters["terminal_score_rows"] == 0
    assert warmdown_death_counters["print_manager_death_stops"] == 1
    np.testing.assert_array_equal(target["alive"], np.asarray([[False, False, False]]))
    np.testing.assert_array_equal(target["score"], np.asarray([[2, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(
        target["round_score"],
        np.asarray([[2, 0, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(target["death_count"], np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[2, 1, 0]], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 15

    warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=3,
    )

    assert warmdown_info["warmdown_end_fires"] == 1
    assert warmdown_info["game_stop_fires"] == 1
    assert warmdown_info["next_round_count"] == 1
    assert warmdown_info["match_end_count"] == 0
    assert warmdown_info["random_tape_draws"] == 9
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in warmdown_info["spawn_infos"][0]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[9:]

    np.testing.assert_array_equal(target["alive"], np.asarray([[True, True, True]]))
    np.testing.assert_array_equal(target["round_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(target["death_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(target["death_player"], np.asarray([[-1, -1, -1]]))
    np.testing.assert_array_equal(target["round_score"], np.zeros((1, 3), dtype=np.int32))
    np.testing.assert_allclose(
        target["pos"][0],
        [[68.575, 47.5], [47.5, 47.5], [26.425, 47.5]],
    )
    np.testing.assert_allclose(
        target["heading"][0],
        [math.tau * 0.75, math.pi, math.tau * 0.25],
    )
    assert int(target["random_tape_cursor"][0]) == 24
    assert int(target["random_tape_draw_count"][0]) == 24


def test_no_bonus_3p_unique_max_score_leader_match_ends_on_warmdown():
    payload = _load_lifecycle_scenario(
        "source_lifecycle_match_end_at_max_score_3p.json"
    )
    assert _source_events(payload)[-2:] == ["game:stop", "end"]
    assert _source_events(payload).count("round:new") == 1
    assert _source_score_triplets(payload)[-3:] == [
        (3, 0, 0),
        (2, 0, 0),
        (1, 2, 2),
    ]

    target, reset_info, warmup_info = _start_3p_no_bonus_lifecycle_fixture(payload)

    assert reset_info["scheduled_timer_kind"] == "game:start"
    assert warmup_info["game_start_fires"] == 1
    np.testing.assert_array_equal(
        warmup_info["print_manager_start_players"],
        np.asarray([2, 1, 0], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 12

    target["pos"][0] = np.asarray(
        [[5.0, 5.0], [1.0, 47.5], [93.0, 47.5]],
        dtype=np.float64,
    )
    target["heading"][0] = np.asarray([math.pi / 4.0, math.pi, 0.0], dtype=np.float64)
    target["prev_pos"][0] = target["pos"][0]
    target["speed"][0, 0] = 8.0

    round_end_counters = _step_3p_no_bonus_lifecycle(target)

    assert round_end_counters["normal_wall_deaths"] == 2
    assert round_end_counters["terminal_score_rows"] == 1
    assert round_end_counters["print_manager_death_stops"] == 2
    assert round_end_counters["random_tape_draws"] == 2
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, False, False]]))
    np.testing.assert_array_equal(target["score"], np.asarray([[2, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(target["match_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["round_winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(target["death_count"], np.asarray([2], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[2, 1, -1]], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 14

    target["max_score"][:] = _source_room_max_score(payload)
    warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=3,
    )

    assert warmdown_info["warmdown_end_fires"] == 1
    assert warmdown_info["game_stop_fires"] == 1
    assert warmdown_info["next_round_count"] == 0
    assert warmdown_info["match_end_count"] == 1
    assert warmdown_info["random_tape_draws"] == 0
    np.testing.assert_array_equal(
        warmdown_info["match_end_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_info["next_round_rows"],
        np.asarray([], dtype=np.int32),
    )
    assert warmdown_info["spawn_infos"] == []
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(target["match_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["terminated"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["reset_pending"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["match_winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(target["started"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["in_round"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["world_active"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["timer_active"], np.zeros((1, 3), dtype=bool))
    assert int(target["random_tape_cursor"][0]) == 14
    assert int(target["random_tape_draw_count"][0]) == 14


def test_no_bonus_3p_tied_max_score_leaders_continue_to_next_round():
    payload = _load_lifecycle_scenario("source_lifecycle_tie_at_max_score_3p.json")
    source_events = _source_events(payload)
    game_stop_index = source_events.index("game:stop")
    assert source_events[game_stop_index : game_stop_index + 2] == [
        "game:stop",
        "round:new",
    ]
    assert "end" not in source_events
    assert _source_score_triplets(payload)[-3:] == [
        (3, 0, 0),
        (2, 1, 1),
        (1, 1, 1),
    ]

    target, reset_info, warmup_info = _start_3p_no_bonus_lifecycle_fixture(payload)

    assert reset_info["scheduled_timer_kind"] == "game:start"
    assert warmup_info["game_start_fires"] == 1
    assert int(target["random_tape_cursor"][0]) == 12

    target["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    target["heading"][0, 2] = 0.0
    target["prev_pos"][0, 2] = target["pos"][0, 2]

    first_death_counters = _step_3p_no_bonus_lifecycle(target)

    assert first_death_counters["normal_wall_deaths"] == 1
    assert first_death_counters["terminal_score_rows"] == 0
    assert first_death_counters["print_manager_death_stops"] == 1
    assert first_death_counters["random_tape_draws"] == 1
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, True, False]]))
    np.testing.assert_array_equal(target["score"], np.zeros((1, 3), dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["death_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[2, -1, -1]], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 13

    target["pos"][0, 1] = np.asarray([1.0, 47.5], dtype=np.float64)
    target["heading"][0, 1] = math.pi
    target["prev_pos"][0, 1] = target["pos"][0, 1]
    target["pos"][0, 0] = np.asarray([47.5, 1.0], dtype=np.float64)
    target["heading"][0, 0] = math.tau * 0.75
    target["prev_pos"][0, 0] = target["pos"][0, 0]

    terminal_counters = _step_3p_no_bonus_lifecycle(target)

    assert terminal_counters["normal_wall_deaths"] == 2
    assert terminal_counters["terminal_score_rows"] == 1
    assert terminal_counters["print_manager_death_stops"] == 2
    assert terminal_counters["random_tape_draws"] == 2
    np.testing.assert_array_equal(target["alive"], np.asarray([[False, False, False]]))
    np.testing.assert_array_equal(target["score"], np.asarray([[1, 1, 0]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_score"], np.zeros((1, 3), dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(target["match_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["round_winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(target["death_count"], np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[2, 1, 0]], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 15

    target["max_score"][:] = _source_room_max_score(payload)
    warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=3,
    )

    assert warmdown_info["warmdown_end_fires"] == 1
    assert warmdown_info["game_stop_fires"] == 1
    assert warmdown_info["next_round_count"] == 1
    assert warmdown_info["match_end_count"] == 0
    assert warmdown_info["random_tape_draws"] == 9
    np.testing.assert_array_equal(
        warmdown_info["next_round_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_info["match_end_rows"],
        np.asarray([], dtype=np.int32),
    )
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in warmdown_info["spawn_infos"][0]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[9:]

    np.testing.assert_array_equal(target["alive"], np.asarray([[True, True, True]]))
    np.testing.assert_array_equal(target["score"], np.asarray([[1, 1, 0]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(target["match_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["death_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(target["death_player"], np.asarray([[-1, -1, -1]]))
    np.testing.assert_array_equal(
        target["timer_kind"],
        np.asarray([[vector_lifecycle.TIMER_KIND_GAME_START, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_allclose(target["timer_remaining_ms"][0], [3000.0, 0.0, 0.0])
    assert int(target["random_tape_cursor"][0]) == 24
    assert int(target["random_tape_draw_count"][0]) == 24


def test_no_bonus_3p_multi_round_match_ends_on_second_warmdown():
    payload = _load_lifecycle_scenario(
        "source_lifecycle_multi_round_match_end_3p.json"
    )
    source_events = _source_events(payload)
    game_stop_indexes = [
        index for index, event in enumerate(source_events) if event == "game:stop"
    ]
    assert [source_events[index + 1] for index in game_stop_indexes] == [
        "round:new",
        "end",
    ]
    assert _source_score_triplets(payload)[-6:] == [
        (3, 0, 0),
        (2, 0, 0),
        (1, 2, 2),
        (3, 0, 0),
        (2, 0, 0),
        (1, 4, 2),
    ]

    target, _, _ = _start_3p_no_bonus_lifecycle_fixture(payload)

    target["pos"][0] = np.asarray(
        [[5.0, 5.0], [1.0, 47.5], [93.0, 47.5]],
        dtype=np.float64,
    )
    target["heading"][0] = np.asarray([math.pi / 4.0, math.pi, 0.0], dtype=np.float64)
    target["prev_pos"][0] = target["pos"][0]
    target["speed"][0, 0] = 8.0

    first_round_counters = _step_3p_no_bonus_lifecycle(target)

    assert first_round_counters["normal_wall_deaths"] == 2
    assert first_round_counters["terminal_score_rows"] == 1
    assert first_round_counters["random_tape_draws"] == 2
    np.testing.assert_array_equal(target["score"], np.asarray([[2, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    assert int(target["random_tape_cursor"][0]) == 14

    target["max_score"][:] = _source_room_max_score(payload)
    first_warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=3,
    )

    assert first_warmdown_info["warmdown_end_fires"] == 1
    assert first_warmdown_info["game_stop_fires"] == 1
    assert first_warmdown_info["next_round_count"] == 1
    assert first_warmdown_info["match_end_count"] == 0
    assert first_warmdown_info["round_clear_print_manager_stops"] == 1
    assert first_warmdown_info["random_tape_draws"] == 10
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in first_warmdown_info["spawn_infos"][0]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[9:]
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, True, True]]))
    np.testing.assert_array_equal(target["score"], np.asarray([[2, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([False], dtype=bool))
    assert int(target["random_tape_cursor"][0]) == 24

    target["max_score"][:] = 10
    game_start_info = vector_runtime.advance_warmup_no_bonus_timers(
        target,
        np.asarray([3000.0], dtype=np.float64),
        player_count=3,
    )

    assert game_start_info["game_start_fires"] == 1
    assert game_start_info["scheduled_print_manager_start_count"] == 3
    assert game_start_info["print_manager_delayed_start_fires"] == 0
    assert game_start_info["random_tape_draws"] == 0
    assert int(target["random_tape_cursor"][0]) == 24

    target["pos"][0] = np.asarray(
        [[5.0, 5.0], [70.0, 20.0], [70.0, 70.0]],
        dtype=np.float64,
    )
    target["heading"][0] = np.asarray(
        [math.pi / 4.0, math.pi / 4.0, math.tau * 0.875],
        dtype=np.float64,
    )
    target["prev_pos"][0] = target["pos"][0]
    target["speed"][0] = np.asarray([8.0, 8.0, 8.0], dtype=np.float64)

    print_start_info = vector_runtime.advance_warmup_no_bonus_timers(
        target,
        np.asarray([3000.0], dtype=np.float64),
        player_count=3,
    )

    assert print_start_info["game_start_fires"] == 0
    assert print_start_info["print_manager_delayed_start_fires"] == 3
    assert print_start_info["random_tape_draws"] == 3
    assert int(target["random_tape_cursor"][0]) == 27

    target["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    target["heading"][0, 2] = 0.0
    target["prev_pos"][0, 2] = target["pos"][0, 2]
    target["speed"][0, 2] = 16.0
    target["pos"][0, 1] = np.asarray([1.0, 47.5], dtype=np.float64)
    target["heading"][0, 1] = math.pi
    target["prev_pos"][0, 1] = target["pos"][0, 1]
    target["speed"][0, 1] = 16.0

    second_round_counters = _step_3p_no_bonus_lifecycle(target)

    assert second_round_counters["normal_wall_deaths"] == 2
    assert second_round_counters["terminal_score_rows"] == 1
    assert second_round_counters["random_tape_draws"] == 2
    np.testing.assert_array_equal(target["score"], np.asarray([[4, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    assert int(target["random_tape_cursor"][0]) == 29

    target["max_score"][:] = _source_room_max_score(payload)
    second_warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=3,
    )

    assert second_warmdown_info["warmdown_end_fires"] == 1
    assert second_warmdown_info["game_stop_fires"] == 1
    assert second_warmdown_info["next_round_count"] == 0
    assert second_warmdown_info["match_end_count"] == 1
    assert second_warmdown_info["random_tape_draws"] == 0
    assert second_warmdown_info["spawn_infos"] == []
    np.testing.assert_array_equal(target["match_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["terminated"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["reset_pending"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["match_winner"], np.asarray([0], dtype=np.int16))
    assert int(target["random_tape_cursor"][0]) == 29
    assert int(target["random_tape_draw_count"][0]) == 29


def test_no_bonus_3p_present_absent_warmdown_resizes_next_round_to_present_count():
    payload = _load_lifecycle_scenario("source_lifecycle_present_absent_3p_next_round.json")
    reset_template, target = _warmup_template_and_target_from_lifecycle_payload(payload)
    for state in (reset_template, target):
        state["present"][0] = np.asarray([True, False, True], dtype=bool)
    _add_no_bonus_runtime_step_arrays(
        reset_template,
        target,
        player_count=3,
    )

    vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        player_count=3,
        reset_seed=555,
        first_warmup_ms=0.0,
        snapshot_array_names=("done", "timer_active"),
    )
    warmup_info = vector_runtime.advance_warmup_no_bonus_timers(
        target,
        np.asarray([3000.0], dtype=np.float64),
        player_count=3,
    )

    assert warmup_info["game_start_fires"] == 1
    np.testing.assert_array_equal(
        warmup_info["print_manager_start_players"],
        np.asarray([2, 1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, False, True]]))
    np.testing.assert_array_equal(target["death_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 9
    np.testing.assert_allclose(target["map_size"], [95.0])

    target["pos"][0, 2] = np.asarray([93.0, 47.5], dtype=np.float64)
    target["heading"][0, 2] = 0.0
    target["prev_pos"][0, 2] = target["pos"][0, 2]
    target["pos"][0, 0] = np.asarray([1.0, 47.5], dtype=np.float64)
    target["heading"][0, 0] = math.pi
    target["prev_pos"][0, 0] = target["pos"][0, 0]

    step_counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=target,
            step_ms=np.asarray([100.0], dtype=np.float64),
            source_moves=np.zeros((1, 3), dtype=np.int8),
            player_count=3,
            print_manager_mode=np.asarray(["death_stop"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )

    assert step_counters["normal_wall_deaths"] == 2
    assert step_counters["terminal_score_rows"] == 1
    assert step_counters["print_manager_death_stops"] == 2
    assert step_counters["random_tape_draws"] == 2
    np.testing.assert_array_equal(target["alive"], np.asarray([[False, False, False]]))
    np.testing.assert_array_equal(target["score"], np.asarray([[1, 0, 1]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(target["match_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["draw"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(target["death_count"], np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[1, 2, 0]], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 11

    warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=3,
    )

    assert warmdown_info["warmdown_end_fires"] == 1
    assert warmdown_info["game_stop_fires"] == 1
    assert warmdown_info["next_round_count"] == 1
    assert warmdown_info["match_end_count"] == 0
    assert warmdown_info["round_clear_print_manager_stops"] == 0
    assert warmdown_info["random_tape_draws"] == 6
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in warmdown_info["spawn_infos"][0]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[6:]

    np.testing.assert_allclose(target["map_size"], [88.0])
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, False, True]]))
    np.testing.assert_array_equal(target["round_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(target["death_count"], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[1, -1, -1]], dtype=np.int16),
    )
    np.testing.assert_allclose(
        target["pos"][0, [0, 2]],
        [[63.5, 44.0], [24.5, 44.0]],
    )
    np.testing.assert_allclose(
        target["heading"][0, [0, 2]],
        [math.tau * 0.75, math.tau * 0.25],
    )
    np.testing.assert_array_equal(
        target["timer_kind"],
        np.asarray([[vector_lifecycle.TIMER_KIND_GAME_START, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_allclose(target["timer_remaining_ms"][0], [3000.0, 0.0, 0.0])
    assert int(target["random_tape_cursor"][0]) == 17
    assert int(target["random_tape_draw_count"][0]) == 17


def test_no_bonus_4p_round_end_warmdown_spawns_next_round_from_source_fixture_rng():
    payload = _load_lifecycle_scenario("source_lifecycle_spawn_rng_4p_next_round.json")
    reset_template, target = _warmup_template_and_target_from_lifecycle_payload(payload)
    _add_no_bonus_runtime_step_arrays(
        reset_template,
        target,
        player_count=4,
    )

    reset_info = vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        player_count=4,
        reset_seed=555,
        first_warmup_ms=0.0,
        snapshot_array_names=("done", "timer_active"),
    )
    warmup_info = vector_runtime.advance_warmup_no_bonus_timers(
        target,
        np.asarray([3000.0], dtype=np.float64),
        player_count=4,
    )
    assert reset_info["scheduled_timer_kind"] == "game:start"
    assert warmup_info["game_start_fires"] == 1
    np.testing.assert_array_equal(
        warmup_info["print_manager_start_players"],
        np.asarray([3, 2, 1, 0], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 16

    target["pos"][0] = np.asarray(
        [[50.5, 99.0], [50.5, 1.0], [1.0, 50.5], [99.0, 50.5]],
        dtype=np.float64,
    )
    target["heading"][0] = np.asarray(
        [math.tau * 0.25, math.tau * 0.75, math.pi, 0.0],
        dtype=np.float64,
    )
    target["prev_pos"][0] = target["pos"][0]

    step_counters = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=target,
            step_ms=np.asarray([100.0], dtype=np.float64),
            source_moves=np.zeros((1, 4), dtype=np.int8),
            player_count=4,
            print_manager_mode=np.asarray(["death_stop"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )

    assert step_counters["movement_updates"] == 4
    assert step_counters["normal_wall_deaths"] == 4
    assert step_counters["terminal_score_rows"] == 1
    assert step_counters["print_manager_death_stops"] == 4
    assert step_counters["random_tape_draws"] == 4
    np.testing.assert_array_equal(
        target["alive"],
        np.asarray([[False, False, False, False]]),
    )
    np.testing.assert_array_equal(target["score"], np.asarray([[0, 0, 0, 0]], dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(target["match_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["round_winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(target["death_count"], np.asarray([4], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        target["timer_kind"],
        np.asarray(
            [[vector_lifecycle.TIMER_KIND_WARMDOWN_END, 0, 0, 0]],
            dtype=np.int16,
        ),
    )
    np.testing.assert_allclose(target["timer_remaining_ms"][0], [5000.0, 0.0, 0.0, 0.0])
    assert int(target["random_tape_cursor"][0]) == 20

    warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=4,
    )

    assert warmdown_info["schema"] == (
        vector_lifecycle.WARMDOWN_ADVANCE_NO_BONUS_INFO_SCHEMA_ID
    )
    assert warmdown_info["surface"] == vector_lifecycle.WARMDOWN_ADVANCE_NO_BONUS_SURFACE
    assert warmdown_info["warmdown_end_fires"] == 1
    assert warmdown_info["game_stop_fires"] == 1
    assert warmdown_info["next_round_count"] == 1
    assert warmdown_info["match_end_count"] == 0
    assert warmdown_info["random_tape_draws"] == 12
    np.testing.assert_array_equal(
        warmdown_info["warmdown_end_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_info["next_round_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmdown_info["scheduled_timer_rows"],
        np.asarray([0], dtype=np.int32),
    )
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in warmdown_info["spawn_infos"][0]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[12:]

    np.testing.assert_array_equal(
        target["alive"],
        np.asarray([[True, True, True, True]]),
    )
    np.testing.assert_array_equal(target["round_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(target["match_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["death_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(target["death_player"], np.asarray([[-1, -1, -1, -1]]))
    np.testing.assert_allclose(
        target["pos"][0],
        [[77.41, 50.5], [59.47, 50.5], [41.53, 50.5], [23.59, 50.5]],
    )
    np.testing.assert_allclose(
        target["heading"][0],
        [math.tau * 0.75, math.pi, 0.1, math.tau * 0.25],
    )
    np.testing.assert_array_equal(
        target["timer_kind"],
        np.asarray([[vector_lifecycle.TIMER_KIND_GAME_START, 0, 0, 0]], dtype=np.int16),
    )
    np.testing.assert_allclose(target["timer_remaining_ms"][0], [3000.0, 0.0, 0.0, 0.0])
    assert int(target["random_tape_cursor"][0]) == 32
    assert int(target["random_tape_draw_count"][0]) == 32


def test_public_4p_all_dead_warmdown_continues_to_next_round_metadata():
    payload = _load_lifecycle_scenario("source_lifecycle_spawn_rng_4p_next_round.json")
    tape = _source_random_tape(payload)
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=4,
        seed=555,
        decision_ms=100.0,
        body_capacity=24,
        event_capacity=64,
        timer_capacity=4,
        random_tape_capacity=tape.shape[1],
    )
    reset_batch = env.reset(
        seed=np.asarray([555], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_ref="scenarios/environment/source_lifecycle_spawn_rng_4p_next_round.json",
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=3000.0,
    )

    assert reset_batch.info["metadata_only"] is True
    assert reset_batch.info["trainer_observation_claim"] is False
    assert reset_batch.info["player_count"] == 4
    assert reset_batch.info["reset_info"]["random_tape_source"] == (
        "source_fixture_random_tape_values"
    )
    assert reset_batch.info["warmup_info"]["game_start_fires"] == 1
    np.testing.assert_array_equal(
        reset_batch.info["warmup_info"]["print_manager_start_players"],
        np.asarray([3, 2, 1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        reset_batch.action_mask,
        np.ones((1, 4, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        reset_batch.info["alive"],
        np.asarray([[True, True, True, True]], dtype=bool),
    )
    assert int(reset_batch.info["random_tape_cursor"][0]) == 16

    env.state["pos"][0] = np.asarray(
        [[50.5, 99.0], [50.5, 1.0], [1.0, 50.5], [99.0, 50.5]],
        dtype=np.float64,
    )
    env.state["heading"][0] = np.asarray(
        [math.tau * 0.25, math.tau * 0.75, math.pi, 0.0],
        dtype=np.float64,
    )
    env.state["prev_pos"][0] = env.state["pos"][0]

    terminal_batch = env.step(np.asarray([[1, 1, 1, 1]], dtype=np.int16))

    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW], dtype=np.int16),
    )
    np.testing.assert_array_equal(terminal_batch.info["draw"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.info["winner"], np.asarray([-1], dtype=np.int16))
    np.testing.assert_array_equal(
        terminal_batch.info["death_player"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["death_count"],
        np.asarray([4], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["needs_reset"],
        np.asarray([True], dtype=bool),
    )
    assert int(terminal_batch.info["random_tape_cursor"][0]) == 20
    assert terminal_batch.final_observation is not None
    assert terminal_batch.final_reward is not None

    warmdown_batch = env.advance_warmdown(5000.0)

    assert warmdown_batch.info["metadata_only"] is True
    assert warmdown_batch.info["trainer_observation_claim"] is False
    assert warmdown_batch.info["warmdown_waited"] is True
    assert warmdown_batch.info["warmdown_info"]["warmdown_end_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["game_stop_fires"] == 1
    assert warmdown_batch.info["warmdown_info"]["next_round_count"] == 1
    assert warmdown_batch.info["warmdown_info"]["match_end_count"] == 0
    assert warmdown_batch.info["warmdown_info"]["random_tape_draws"] == 12
    np.testing.assert_array_equal(
        warmdown_batch.info["terminal_rows"],
        np.asarray([], dtype=np.int32),
    )
    np.testing.assert_array_equal(warmdown_batch.done, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        warmdown_batch.info["needs_reset"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["alive"],
        np.asarray([[True, True, True, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["round_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.info["match_done"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(
        warmdown_batch.action_mask,
        np.ones((1, 4, 3), dtype=bool),
    )
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in warmdown_batch.info["warmdown_info"]["spawn_infos"][0]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[12:]
    np.testing.assert_allclose(
        env.state["pos"][0],
        [[77.41, 50.5], [59.47, 50.5], [41.53, 50.5], [23.59, 50.5]],
    )
    np.testing.assert_allclose(
        env.state["heading"][0],
        [math.tau * 0.75, math.pi, 0.1, math.tau * 0.25],
    )
    assert int(warmdown_batch.info["random_tape_cursor"][0]) == 32
    assert int(warmdown_batch.info["random_tape_draw_count"][0]) == 32
    assert warmdown_batch.final_observation is None
    assert warmdown_batch.final_reward is None


def test_no_bonus_4p_survivor_score_warmdown_spawns_next_round_from_source_fixture_rng():
    payload = _load_lifecycle_scenario("source_lifecycle_survivor_score_4p_next_round.json")
    reset_template, target = _warmup_template_and_target_from_lifecycle_payload(payload)
    _add_no_bonus_runtime_step_arrays(
        reset_template,
        target,
        player_count=4,
    )

    vector_lifecycle.reset_spawn_warmup_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        player_count=4,
        reset_seed=555,
        first_warmup_ms=0.0,
        snapshot_array_names=("done", "timer_active"),
    )
    vector_runtime.advance_warmup_no_bonus_timers(
        target,
        np.asarray([3000.0], dtype=np.float64),
        player_count=4,
    )
    assert int(target["random_tape_cursor"][0]) == 16

    target["pos"][0] = np.asarray(
        [[10.0, 50.5], [30.0, 30.0], [70.0, 70.0], [99.0, 50.5]],
        dtype=np.float64,
    )
    target["heading"][0] = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    target["prev_pos"][0] = target["pos"][0]
    target["speed"][0, 0] = 8.0

    first_death = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=target,
            step_ms=np.asarray([100.0], dtype=np.float64),
            source_moves=np.zeros((1, 4), dtype=np.int8),
            player_count=4,
            print_manager_mode=np.asarray(["death_stop"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )
    assert first_death["normal_wall_deaths"] == 1
    assert first_death["terminal_score_rows"] == 0
    np.testing.assert_array_equal(
        target["alive"],
        np.asarray([[True, True, True, False]]),
    )
    np.testing.assert_array_equal(
        target["round_score"],
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
    )
    assert int(target["random_tape_cursor"][0]) == 17

    target["pos"][0, 2] = np.asarray([1.0, 50.5], dtype=np.float64)
    target["heading"][0, 2] = math.pi
    target["prev_pos"][0, 2] = target["pos"][0, 2]
    second_death = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=target,
            step_ms=np.asarray([100.0], dtype=np.float64),
            source_moves=np.zeros((1, 4), dtype=np.int8),
            player_count=4,
            print_manager_mode=np.asarray(["death_stop"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )
    assert second_death["normal_wall_deaths"] == 1
    assert second_death["terminal_score_rows"] == 0
    np.testing.assert_array_equal(
        target["alive"],
        np.asarray([[True, True, False, False]]),
    )
    np.testing.assert_array_equal(
        target["round_score"],
        np.asarray([[0, 0, 1, 0]], dtype=np.int32),
    )
    assert int(target["random_tape_cursor"][0]) == 18

    target["pos"][0, 1] = np.asarray([50.5, 1.0], dtype=np.float64)
    target["heading"][0, 1] = math.tau * 0.75
    target["prev_pos"][0, 1] = target["pos"][0, 1]
    terminal_death = vector_runtime.step_many(
        vector_runtime.VectorStepInput(
            state=target,
            step_ms=np.asarray([100.0], dtype=np.float64),
            source_moves=np.zeros((1, 4), dtype=np.int8),
            player_count=4,
            print_manager_mode=np.asarray(["death_stop"], dtype=object),
            event_mode=vector_runtime.EVENT_MODE_NONE,
        )
    )
    assert terminal_death["normal_wall_deaths"] == 1
    assert terminal_death["terminal_score_rows"] == 1
    np.testing.assert_array_equal(
        target["alive"],
        np.asarray([[True, False, False, False]]),
    )
    np.testing.assert_array_equal(
        target["score"],
        np.asarray([[3, 2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(target["round_score"], np.zeros((1, 4), dtype=np.int32))
    np.testing.assert_array_equal(target["round_done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([True], dtype=bool),
    )
    np.testing.assert_array_equal(target["round_winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(target["death_count"], np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(
        target["death_player"],
        np.asarray([[3, 2, 1, -1]], dtype=np.int16),
    )
    assert int(target["random_tape_cursor"][0]) == 19

    warmdown_info = vector_lifecycle.advance_warmdown_no_bonus_rows(
        target,
        np.asarray([5000.0], dtype=np.float64),
        player_count=4,
    )

    assert warmdown_info["warmdown_end_fires"] == 1
    assert warmdown_info["game_stop_fires"] == 1
    assert warmdown_info["next_round_count"] == 1
    assert warmdown_info["match_end_count"] == 0
    assert warmdown_info["round_clear_print_manager_stops"] == 1
    assert warmdown_info["random_tape_draws"] == 13
    np.testing.assert_array_equal(
        warmdown_info["round_clear_print_manager_stop_players"],
        np.asarray([0], dtype=np.int16),
    )
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in warmdown_info["spawn_infos"][0]["random_calls"]
    ] == _expected_spawn_random_calls(payload)[12:]

    np.testing.assert_array_equal(
        target["alive"],
        np.asarray([[True, True, True, True]]),
    )
    np.testing.assert_array_equal(target["round_done"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        target["warmdown_pending"],
        np.asarray([False], dtype=bool),
    )
    np.testing.assert_array_equal(target["death_count"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(target["death_player"], np.asarray([[-1, -1, -1, -1]]))
    np.testing.assert_allclose(
        target["pos"][0],
        [[77.41, 50.5], [59.47, 50.5], [41.53, 50.5], [23.59, 50.5]],
    )
    np.testing.assert_allclose(
        target["heading"][0],
        [math.tau * 0.75, math.pi, 0.1, math.tau * 0.25],
    )
    assert int(target["random_tape_cursor"][0]) == 32
    assert int(target["random_tape_draw_count"][0]) == 32


def test_reset_spawn_warmup_1v1_no_bonus_reports_missing_required_arrays_without_mutating():
    reset_template, target = _reset_spawn_template_and_target()
    _add_optional_1v1_lifecycle_arrays(reset_template, target)
    _add_1v1_warmup_round_local_arrays(reset_template, target)
    target_before = {name: array.copy() for name, array in target.items()}
    target.pop("timer_active")

    info = vector_lifecycle.reset_spawn_warmup_1v1_no_bonus_rows(
        target,
        reset_template,
        np.asarray([False, True], dtype=bool),
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        snapshot_array_names=("done",),
    )

    assert info["can_compose"] is False
    assert info["reset_info"] is None
    assert info["spawn_info"] is None
    assert info["terminal_transition_snapshot"] is None
    assert "timer_active" in info["warmup_missing_target_arrays"]
    for name, before in target_before.items():
        if name in target:
            np.testing.assert_array_equal(target[name], before)


def test_run_warmup_start_step_1v1_no_bonus_rows_reaches_autoreset_wall_death():
    reset_template, target = _one_row_wall_loop_template_and_target()

    info = vector_lifecycle.run_warmup_start_step_1v1_no_bonus_rows(
        target,
        reset_template,
        np.asarray([True], dtype=bool),
        reset_seed=555,
        reset_source=vector_reset.RESET_SOURCE_AUTORESET,
        runtime_steps={
            "player_count": 2,
            "step_ms": np.asarray([300.0], dtype=np.float64),
            "source_moves": np.asarray([[0, 0]], dtype=np.int8),
            "print_manager_mode": np.asarray(["death_stop"], dtype=object),
            "timer_advance_ms": np.asarray([0.0], dtype=np.float64),
        },
        snapshot_array_names=("done", "terminal_reason"),
    )

    assert info["schema"] == (
        vector_lifecycle.WARMUP_START_STEP_1V1_NO_BONUS_INFO_SCHEMA_ID
    )
    assert info["surface"] == vector_lifecycle.WARMUP_START_STEP_1V1_NO_BONUS_SURFACE
    assert info["row_count"] == 1
    assert info["runtime_step_count"] == 1
    reset_spawn_info = info["reset_spawn_info"]
    assert reset_spawn_info["can_compose"] is True
    assert reset_spawn_info["scheduled_timer_kind"] == "game:start"
    np.testing.assert_array_equal(
        reset_spawn_info["scheduled_timer_rows"],
        np.asarray([0], dtype=np.int32),
    )

    warmup_timer_info = info["warmup_timer_info"]
    assert warmup_timer_info["game_start_fires"] == 1
    assert warmup_timer_info["print_manager_delayed_start_fires"] == 2
    np.testing.assert_array_equal(
        warmup_timer_info["game_start_rows"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        warmup_timer_info["print_manager_start_players"],
        np.asarray([1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        warmup_timer_info["random_draw_count_delta"],
        np.asarray([2], dtype=np.int32),
    )

    step_counters = info["runtime_step_counters"][0]
    assert step_counters["normal_wall_deaths"] == 1
    assert step_counters["terminal_score_rows"] == 1
    assert step_counters["print_manager_death_stops"] == 1

    np.testing.assert_allclose(target["pos"][0, 1], [5.0, 104.0])
    np.testing.assert_array_equal(target["alive"], np.asarray([[True, False]], dtype=bool))
    np.testing.assert_array_equal(target["done"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["terminated"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(target["truncated"], np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(target["reset_pending"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(
        target["terminal_reason"],
        np.asarray([vector_reset.TERMINAL_REASON_SURVIVOR_WIN], dtype=np.int16),
    )
    np.testing.assert_array_equal(target["winner"], np.asarray([0], dtype=np.int16))
    np.testing.assert_array_equal(
        target["print_manager_active"],
        np.asarray([[True, False]], dtype=bool),
    )
    assert int(target["random_tape_cursor"][0]) == 10
    assert int(target["random_tape_draw_count"][0]) == 10

    transition = (
        vector_trainer_observation.build_final_trainer_transition_1v1_no_bonus_rows(
            target,
            np.asarray([True], dtype=bool),
            decision_ms=300.0,
            max_ticks=100,
        )
    )
    assert transition["final_observation"].shape == (1, 2, 106)
    np.testing.assert_array_equal(
        transition["final_reward_map"],
        np.asarray([[1.0, -1.0]], dtype=np.float32),
    )

    plan = vector_autoreset.plan_autoreset_rows(
        target,
        final_observation=transition["final_observation"],
        final_reward_map=transition["final_reward_map"],
        reset_seed=target["reset_seed"],
        reset_source=target["reset_source"],
    )

    assert plan["autoreset_count"] == 1
    np.testing.assert_array_equal(plan["eligible_mask"], np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(plan["row_ids"], np.asarray([0], dtype=np.int32))
