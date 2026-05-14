import json
from pathlib import Path

import numpy as np
import pytest

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import SOURCE_BONUS_POPING_TIME_MS
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


SCENARIO_ROOT = Path(__file__).resolve().parents[1] / "scenarios" / "environment"


@pytest.mark.parametrize(
    ("scenario_name", "expected"),
    (
        (
            "source_bonus_spawn_type_position_rng_step.json",
            {
                "spawned_bonus_id": 1,
                "spawned_slot": 0,
                "spawned_pos": (23.94, 64.06),
                "accepted_position_attempt": 0,
                "position_attempt_count": 1,
                "rejected_game_world_attempts": 0,
                "rejected_bonus_world_attempts": 0,
                "world_body_count": 0,
                "bonus_count": 1,
            },
        ),
        (
            "source_bonus_spawn_game_world_retry_step.json",
            {
                "spawned_bonus_id": 1,
                "spawned_slot": 0,
                "spawned_pos": (68.072, 19.928),
                "accepted_position_attempt": 1,
                "position_attempt_count": 2,
                "rejected_game_world_attempts": 1,
                "rejected_bonus_world_attempts": 0,
                "world_body_count": 1,
                "bonus_count": 1,
            },
        ),
        (
            "source_bonus_spawn_bonus_world_retry_step.json",
            {
                "spawned_bonus_id": 2,
                "spawned_slot": 1,
                "spawned_pos": (68.072, 19.928),
                "accepted_position_attempt": 1,
                "position_attempt_count": 2,
                "rejected_game_world_attempts": 0,
                "rejected_bonus_world_attempts": 1,
                "world_body_count": 0,
                "bonus_count": 2,
            },
        ),
    ),
)
def test_public_env_promotes_source_bonus_spawn_rng_and_retry_fixtures(
    scenario_name: str,
    expected: dict[str, object],
):
    env, reset_batch, scenario, random_entries, spawn_prefix_length = (
        _source_bonus_spawn_fixture_env(scenario_name)
    )
    step = scenario["steps"][0]

    assert reset_batch.info["bonus_support_mode"] == "natural_spawn"
    assert reset_batch.info["bonus_support"]["natural_bonus_spawn"] is True
    assert reset_batch.info["bonus_support"]["natural_bonus_rate"] == pytest.approx(1.0)
    assert reset_batch.info["natural_bonus_reset_info"]["source_bonus_poping_time_ms"] == (
        pytest.approx(1500.0)
    )
    assert reset_batch.info["natural_bonus_reset_info"]["random_calls"][-1]["label"] == (
        "bonus.start_delay"
    )
    assert reset_batch.info["natural_bonus_reset_info"]["random_calls"][-1][
        "tape_index"
    ] == spawn_prefix_length
    np.testing.assert_allclose(
        reset_batch.info["natural_bonus_timer_remaining_ms"],
        np.asarray([1500.0], dtype=np.float64),
    )

    batch = env.step(
        _actions_for_source_step(step),
        timer_advance_ms=np.asarray([float(step["advance_timers_ms"])], dtype=np.float64),
    )

    natural_info = batch.info["natural_bonus_info"]
    expected_step_entries = random_entries[1:]
    assert [call["label"] for call in natural_info["random_calls"]] == [
        entry["label"] for entry in expected_step_entries
    ]
    assert [call["tape_index"] for call in natural_info["random_calls"]] == list(
        range(spawn_prefix_length + 1, spawn_prefix_length + len(random_entries))
    )
    np.testing.assert_allclose(
        [call["value"] for call in natural_info["random_calls"]],
        [entry["value"] for entry in expected_step_entries],
    )
    assert int(natural_info["random_tape_draws"]) == len(expected_step_entries)
    assert int(batch.info["random_tape_cursor"][0]) == (
        spawn_prefix_length + len(random_entries)
    )
    assert int(batch.info["random_tape_draw_count"][0]) == (
        spawn_prefix_length + len(random_entries)
    )

    assert natural_info["schedule_calls"] == [
        {
            "row": 0,
            "label": "bonus.next_delay_after_pop",
            "delay_draw": 0.5,
            "delay_ms": 2250.0,
        }
    ]
    np.testing.assert_allclose(
        natural_info["remaining_ms"],
        np.asarray([2250.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        natural_info["next_due_elapsed_ms"],
        np.asarray([3750.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(natural_info["due_rows"], np.asarray([True]))
    np.testing.assert_array_equal(natural_info["active_rows"], np.asarray([True]))
    np.testing.assert_array_equal(natural_info["pop_count"], np.asarray([1]))

    spawn_info = natural_info["spawn_infos"][0]
    np.testing.assert_array_equal(spawn_info["due_rows"], np.asarray([True]))
    np.testing.assert_array_equal(spawn_info["capped_rows"], np.asarray([False]))
    np.testing.assert_array_equal(spawn_info["selection_rows"], np.asarray([True]))
    np.testing.assert_array_equal(spawn_info["spawn_rows"], np.asarray([True]))
    assert str(spawn_info["selected_type_name"][0]) == "BonusSelfSmall"
    assert int(spawn_info["selected_type_code"][0]) == (
        vector_runtime.BONUS_TYPE_SELF_SMALL
    )
    np.testing.assert_array_equal(
        spawn_info["spawned_bonus_id"],
        np.asarray([expected["spawned_bonus_id"]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["spawned_slot"],
        np.asarray([expected["spawned_slot"]], dtype=np.int32),
    )
    np.testing.assert_allclose(spawn_info["spawned_pos"], [expected["spawned_pos"]])
    np.testing.assert_array_equal(
        spawn_info["accepted_position_attempt"],
        np.asarray([expected["accepted_position_attempt"]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["position_attempt_count"],
        np.asarray([expected["position_attempt_count"]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["rejected_game_world_attempts"],
        np.asarray([expected["rejected_game_world_attempts"]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["rejected_bonus_world_attempts"],
        np.asarray([expected["rejected_bonus_world_attempts"]], dtype=np.int32),
    )
    assert len(spawn_info["spawn_events"]) == 1
    spawn_event = spawn_info["spawn_events"][0]
    assert spawn_event["event"] == "bonus:pop"
    assert spawn_event["row"] == 0
    assert spawn_event["bonus"] == expected["spawned_bonus_id"]
    assert spawn_event["type"] == "BonusSelfSmall"
    np.testing.assert_allclose(
        [spawn_event["x"], spawn_event["y"]],
        expected["spawned_pos"],
    )

    assert int(env.state["world_body_count"][0]) == expected["world_body_count"]
    assert int(env.state["bonus_count"][0]) == expected["bonus_count"]
    assert int(env.state["bonus_world_body_count"][0]) == expected["bonus_count"]
    assert int(env.state["bonus_type"][0, int(expected["spawned_slot"])]) == (
        vector_runtime.BONUS_TYPE_SELF_SMALL
    )
    np.testing.assert_allclose(
        env.state["bonus_pos"][0, int(expected["spawned_slot"])],
        expected["spawned_pos"],
    )

    assert batch.info["bonus_support"]["active_count"].tolist() == [
        expected["bonus_count"]
    ]
    assert batch.info["step_counters"]["random_tape_draws"] == 0
    assert batch.info["step_counters"]["print_manager_delayed_start_fires"] == 0


def test_public_env_promotes_source_bonus_spawn_cap_twenty_fixture():
    env, reset_batch, scenario, random_entries, spawn_prefix_length = (
        _source_bonus_spawn_fixture_env("source_bonus_spawn_cap_twenty_step.json")
    )
    step = scenario["steps"][0]

    assert [entry["label"] for entry in random_entries] == [
        "bonus.start_delay",
        "bonus.next_delay_after_pop",
    ]
    assert reset_batch.info["bonus_support_mode"] == "natural_spawn"
    assert reset_batch.info["bonus_support"]["natural_bonus_rate"] == pytest.approx(1.0)
    assert reset_batch.info["natural_bonus_reset_info"]["random_calls"][-1][
        "tape_index"
    ] == spawn_prefix_length
    assert reset_batch.info["natural_bonus_reset_info"]["source_bonus_poping_time_ms"] == (
        pytest.approx(1500.0)
    )
    np.testing.assert_allclose(
        reset_batch.info["natural_bonus_timer_remaining_ms"],
        np.asarray([1500.0], dtype=np.float64),
    )

    batch = env.step(
        _actions_for_source_step(step),
        timer_advance_ms=np.asarray([float(step["advance_timers_ms"])], dtype=np.float64),
    )

    natural_info = batch.info["natural_bonus_info"]
    assert [call["label"] for call in natural_info["random_calls"]] == [
        "bonus.next_delay_after_pop"
    ]
    assert [call["tape_index"] for call in natural_info["random_calls"]] == [
        spawn_prefix_length + 1
    ]
    np.testing.assert_allclose(
        [call["value"] for call in natural_info["random_calls"]],
        [random_entries[1]["value"]],
    )
    assert int(natural_info["random_tape_draws"]) == 1
    assert int(batch.info["random_tape_cursor"][0]) == spawn_prefix_length + 2
    assert int(batch.info["random_tape_draw_count"][0]) == spawn_prefix_length + 2

    assert natural_info["schedule_calls"] == [
        {
            "row": 0,
            "label": "bonus.next_delay_after_pop",
            "delay_draw": 0.5,
            "delay_ms": 2250.0,
        }
    ]
    np.testing.assert_allclose(
        natural_info["remaining_ms"],
        np.asarray([2250.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(natural_info["due_rows"], np.asarray([True]))
    np.testing.assert_array_equal(natural_info["pop_count"], np.asarray([1]))

    spawn_info = natural_info["spawn_infos"][0]
    np.testing.assert_array_equal(spawn_info["due_rows"], np.asarray([True]))
    np.testing.assert_array_equal(spawn_info["capped_rows"], np.asarray([True]))
    np.testing.assert_array_equal(spawn_info["selection_rows"], np.asarray([False]))
    np.testing.assert_array_equal(spawn_info["spawn_rows"], np.asarray([False]))
    assert spawn_info["type_selection_info"] is None
    assert spawn_info["spawn_events"] == []
    assert str(spawn_info["selected_type_name"][0]) == "None"
    assert int(spawn_info["selected_type_code"][0]) == vector_runtime.BONUS_TYPE_NONE
    np.testing.assert_array_equal(
        spawn_info["spawned_bonus_id"],
        np.asarray([-1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["spawned_slot"],
        np.asarray([-1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["position_attempt_count"],
        np.asarray([0], dtype=np.int32),
    )
    assert int(spawn_info["source_max_active_bonuses"]) == 20

    assert int(env.state["bonus_count"][0]) == vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    assert int(env.state["bonus_world_body_count"][0]) == (
        vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    )
    np.testing.assert_array_equal(
        env.state["bonus_active"][0, : vector_runtime.SOURCE_MAX_ACTIVE_BONUSES],
        np.ones(vector_runtime.SOURCE_MAX_ACTIVE_BONUSES, dtype=bool),
    )
    assert batch.info["bonus_support"]["active_count"].tolist() == [
        vector_runtime.SOURCE_MAX_ACTIVE_BONUSES
    ]
    assert batch.info["step_counters"]["random_tape_draws"] == 0


def test_public_env_promotes_source_bonus_default_weights_type_rng_fixture():
    env, reset_batch, scenario, random_entries, spawn_prefix_length = (
        _source_bonus_spawn_fixture_env("source_bonus_default_weights_type_rng_step.json")
    )
    step = scenario["steps"][0]

    assert [entry["label"] for entry in random_entries] == [
        "bonus.start_delay",
        "bonus.next_delay_after_pop",
        "bonus.type.BonusAllColor",
        "bonus.position.x",
        "bonus.position.y",
    ]
    assert reset_batch.info["bonus_support_mode"] == "natural_spawn"
    assert reset_batch.info["bonus_support"]["natural_bonus_rate"] == pytest.approx(1.0)
    assert reset_batch.info["natural_bonus_reset_info"]["random_calls"][-1][
        "tape_index"
    ] == spawn_prefix_length
    assert reset_batch.info["natural_bonus_reset_info"]["source_bonus_poping_time_ms"] == (
        pytest.approx(1500.0)
    )
    np.testing.assert_allclose(
        reset_batch.info["natural_bonus_timer_remaining_ms"],
        np.asarray([1500.0], dtype=np.float64),
    )

    batch = env.step(
        _actions_for_source_step(step),
        timer_advance_ms=np.asarray([float(step["advance_timers_ms"])], dtype=np.float64),
    )

    natural_info = batch.info["natural_bonus_info"]
    expected_step_entries = random_entries[1:]
    assert [call["label"] for call in natural_info["random_calls"]] == [
        entry["label"] for entry in expected_step_entries
    ]
    assert [call["tape_index"] for call in natural_info["random_calls"]] == list(
        range(spawn_prefix_length + 1, spawn_prefix_length + len(random_entries))
    )
    np.testing.assert_allclose(
        [call["value"] for call in natural_info["random_calls"]],
        [entry["value"] for entry in expected_step_entries],
    )
    assert int(natural_info["random_tape_draws"]) == len(expected_step_entries)
    assert int(batch.info["random_tape_cursor"][0]) == (
        spawn_prefix_length + len(random_entries)
    )
    assert int(batch.info["random_tape_draw_count"][0]) == (
        spawn_prefix_length + len(random_entries)
    )

    assert natural_info["schedule_calls"] == [
        {
            "row": 0,
            "label": "bonus.next_delay_after_pop",
            "delay_draw": 0.5,
            "delay_ms": 2250.0,
        }
    ]
    np.testing.assert_allclose(
        natural_info["remaining_ms"],
        np.asarray([2250.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        natural_info["next_due_elapsed_ms"],
        np.asarray([3750.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(natural_info["due_rows"], np.asarray([True]))
    np.testing.assert_array_equal(natural_info["active_rows"], np.asarray([True]))
    np.testing.assert_array_equal(natural_info["pop_count"], np.asarray([1]))

    spawn_info = natural_info["spawn_infos"][0]
    np.testing.assert_array_equal(spawn_info["due_rows"], np.asarray([True]))
    np.testing.assert_array_equal(spawn_info["capped_rows"], np.asarray([False]))
    np.testing.assert_array_equal(spawn_info["selection_rows"], np.asarray([True]))
    np.testing.assert_array_equal(spawn_info["spawn_rows"], np.asarray([True]))
    assert str(spawn_info["selected_type_name"][0]) == "BonusAllColor"
    assert int(spawn_info["selected_type_code"][0]) == (
        vector_runtime.BONUS_TYPE_ALL_COLOR
    )
    type_info = spawn_info["type_selection_info"]
    assert type_info is not None
    np.testing.assert_array_equal(type_info["eligible_rows"], np.asarray([True]))
    np.testing.assert_allclose(type_info["type_draw"], [0.945])
    np.testing.assert_allclose(type_info["game_clear_probability"], [0.5])
    np.testing.assert_allclose(type_info["total_weight"], [10.7])
    np.testing.assert_allclose(type_info["weighted_draw"], [10.1115])
    np.testing.assert_array_equal(
        type_info["selected_type_code"],
        np.asarray([vector_runtime.BONUS_TYPE_ALL_COLOR], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        type_info["selected_type_name"],
        np.asarray(["BonusAllColor"], dtype=object),
    )
    np.testing.assert_array_equal(
        spawn_info["spawned_bonus_id"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["spawned_slot"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_allclose(spawn_info["spawned_pos"], [[27.255, 73.745]])
    np.testing.assert_array_equal(
        spawn_info["accepted_position_attempt"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["position_attempt_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["rejected_game_world_attempts"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["rejected_bonus_world_attempts"],
        np.asarray([0], dtype=np.int32),
    )
    assert len(spawn_info["spawn_events"]) == 1
    spawn_event = spawn_info["spawn_events"][0]
    assert spawn_event["event"] == "bonus:pop"
    assert spawn_event["row"] == 0
    assert spawn_event["bonus"] == 1
    assert spawn_event["type"] == "BonusAllColor"
    np.testing.assert_allclose(
        [spawn_event["x"], spawn_event["y"]],
        [27.255, 73.745],
    )

    np.testing.assert_array_equal(
        env.state["alive"][0],
        np.asarray([True, True, False, False], dtype=bool),
    )
    assert int(env.state["bonus_count"][0]) == 1
    assert int(env.state["bonus_world_body_count"][0]) == 1
    assert int(env.state["bonus_type"][0, 0]) == vector_runtime.BONUS_TYPE_ALL_COLOR
    np.testing.assert_allclose(env.state["bonus_pos"][0, 0], [27.255, 73.745])
    assert batch.info["bonus_support"]["active_count"].tolist() == [1]
    assert batch.info["bonus_support"]["natural_bonus_spawn"] is True
    assert batch.info["step_counters"]["random_tape_draws"] == 0


@pytest.mark.parametrize(
    ("scenario_name", "expected"),
    (
        (
            "source_bonus_default_weights_select_game_clear_step.json",
            {
                "alive": [True, True, False, False],
                "type_draw": 0.965,
                "game_clear_probability": 0.5,
                "total_weight": 10.7,
                "weighted_draw": 10.3255,
            },
        ),
        (
            "source_bonus_default_weights_game_clear_full_probability_step.json",
            {
                "alive": [True, True, False, True],
                "type_draw": 0.93,
                "game_clear_probability": 1.0,
                "total_weight": 11.2,
                "weighted_draw": 10.416,
            },
        ),
    ),
)
def test_public_env_promotes_source_bonus_game_clear_probability_fixtures(
    scenario_name: str,
    expected: dict[str, object],
):
    env, reset_batch, scenario, random_entries, spawn_prefix_length = (
        _source_bonus_spawn_fixture_env(scenario_name)
    )
    step = scenario["steps"][0]

    expected_labels = [
        "bonus.start_delay",
        "bonus.next_delay_after_pop",
        "bonus.type.BonusGameClear",
        "bonus.position.x",
        "bonus.position.y",
    ]
    assert [entry["label"] for entry in random_entries] == expected_labels
    assert reset_batch.info["bonus_support_mode"] == "natural_spawn"
    assert reset_batch.info["bonus_support"]["natural_bonus_rate"] == pytest.approx(1.0)
    assert reset_batch.info["bonus_support"]["source_bonus_base_poping_time_ms"] == (
        pytest.approx(SOURCE_BONUS_POPING_TIME_MS)
    )
    assert reset_batch.info["bonus_support"]["source_bonus_poping_time_ms"] == (
        pytest.approx(1500.0)
    )
    reset_info = reset_batch.info["natural_bonus_reset_info"]
    assert reset_info["natural_bonus_rate"] == pytest.approx(1.0)
    assert reset_info["source_bonus_base_poping_time_ms"] == pytest.approx(
        SOURCE_BONUS_POPING_TIME_MS
    )
    assert reset_info["source_bonus_poping_time_ms"] == pytest.approx(1500.0)
    assert reset_info["random_calls"][-1]["label"] == "bonus.start_delay"
    assert reset_info["random_calls"][-1]["tape_index"] == spawn_prefix_length
    assert reset_info["random_calls"][-1]["draw_ordinal"] == spawn_prefix_length
    np.testing.assert_allclose(reset_info["delay_draw"], np.asarray([0.0]))
    np.testing.assert_allclose(reset_info["delay_ms"], np.asarray([1500.0]))
    np.testing.assert_allclose(
        reset_batch.info["natural_bonus_timer_remaining_ms"],
        np.asarray([1500.0], dtype=np.float64),
    )

    batch = env.step(
        _actions_for_source_step(step),
        timer_advance_ms=np.asarray([float(step["advance_timers_ms"])], dtype=np.float64),
    )

    natural_info = batch.info["natural_bonus_info"]
    expected_step_entries = random_entries[1:]
    assert [call["label"] for call in natural_info["random_calls"]] == [
        entry["label"] for entry in expected_step_entries
    ]
    assert [call["tape_index"] for call in natural_info["random_calls"]] == list(
        range(spawn_prefix_length + 1, spawn_prefix_length + len(random_entries))
    )
    assert [call["draw_ordinal"] for call in natural_info["random_calls"]] == list(
        range(spawn_prefix_length + 1, spawn_prefix_length + len(random_entries))
    )
    np.testing.assert_allclose(
        [call["value"] for call in natural_info["random_calls"]],
        [entry["value"] for entry in expected_step_entries],
    )
    assert int(natural_info["random_tape_draws"]) == len(expected_step_entries)
    assert int(batch.info["random_tape_cursor"][0]) == (
        spawn_prefix_length + len(random_entries)
    )
    assert int(batch.info["random_tape_draw_count"][0]) == (
        spawn_prefix_length + len(random_entries)
    )

    assert natural_info["source_bonus_base_poping_time_ms"] == pytest.approx(
        SOURCE_BONUS_POPING_TIME_MS
    )
    assert natural_info["source_bonus_poping_time_ms"] == pytest.approx(1500.0)
    assert natural_info["natural_bonus_rate"] == pytest.approx(1.0)
    assert natural_info["schedule_calls"] == [
        {
            "row": 0,
            "label": "bonus.next_delay_after_pop",
            "delay_draw": 0.5,
            "delay_ms": 2250.0,
        }
    ]
    np.testing.assert_allclose(
        natural_info["remaining_ms"],
        np.asarray([2250.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        natural_info["next_due_elapsed_ms"],
        np.asarray([3750.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(natural_info["due_rows"], np.asarray([True]))
    np.testing.assert_array_equal(natural_info["active_rows"], np.asarray([True]))
    np.testing.assert_array_equal(natural_info["pop_count"], np.asarray([1]))

    spawn_info = natural_info["spawn_infos"][0]
    np.testing.assert_array_equal(spawn_info["due_rows"], np.asarray([True]))
    np.testing.assert_array_equal(spawn_info["capped_rows"], np.asarray([False]))
    np.testing.assert_array_equal(spawn_info["selection_rows"], np.asarray([True]))
    np.testing.assert_array_equal(spawn_info["spawn_rows"], np.asarray([True]))
    assert str(spawn_info["selected_type_name"][0]) == "BonusGameClear"
    assert int(spawn_info["selected_type_code"][0]) == (
        vector_runtime.BONUS_TYPE_GAME_CLEAR
    )

    type_info = spawn_info["type_selection_info"]
    assert type_info is not None
    np.testing.assert_array_equal(type_info["eligible_rows"], np.asarray([True]))
    np.testing.assert_allclose(type_info["type_draw"], [expected["type_draw"]])
    np.testing.assert_allclose(
        type_info["game_clear_probability"],
        [expected["game_clear_probability"]],
    )
    np.testing.assert_allclose(type_info["total_weight"], [expected["total_weight"]])
    np.testing.assert_allclose(type_info["weighted_draw"], [expected["weighted_draw"]])
    np.testing.assert_array_equal(
        type_info["selected_type_code"],
        np.asarray([vector_runtime.BONUS_TYPE_GAME_CLEAR], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        type_info["selected_type_name"],
        np.asarray(["BonusGameClear"], dtype=object),
    )

    np.testing.assert_array_equal(
        spawn_info["spawned_bonus_id"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["spawned_slot"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_allclose(spawn_info["spawned_pos"], [[27.255, 73.745]])
    np.testing.assert_array_equal(
        spawn_info["accepted_position_attempt"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["position_attempt_count"],
        np.asarray([1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["rejected_game_world_attempts"],
        np.asarray([0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        spawn_info["rejected_bonus_world_attempts"],
        np.asarray([0], dtype=np.int32),
    )
    assert len(spawn_info["spawn_events"]) == 1
    spawn_event = spawn_info["spawn_events"][0]
    assert spawn_event["event"] == "bonus:pop"
    assert spawn_event["row"] == 0
    assert spawn_event["bonus"] == 1
    assert spawn_event["type"] == "BonusGameClear"
    np.testing.assert_allclose([spawn_event["x"], spawn_event["y"]], [27.255, 73.745])

    np.testing.assert_array_equal(
        env.state["alive"][0],
        np.asarray(expected["alive"], dtype=bool),
    )
    assert int(env.state["bonus_count"][0]) == 1
    assert int(env.state["bonus_world_body_count"][0]) == 1
    assert int(env.state["bonus_type"][0, 0]) == vector_runtime.BONUS_TYPE_GAME_CLEAR
    assert int(env.state["bonus_id"][0, 0]) == 1
    np.testing.assert_allclose(env.state["bonus_pos"][0, 0], [27.255, 73.745])
    assert batch.info["bonus_support"]["natural_bonus_spawn"] is True
    assert batch.info["bonus_support"]["active_count"].tolist() == [1]
    assert bool(batch.info["bonus_support"]["bonus_active"][0, 0]) is True
    assert int(batch.info["bonus_support"]["bonus_type"][0, 0]) == (
        vector_runtime.BONUS_TYPE_GAME_CLEAR
    )
    assert batch.info["step_counters"]["random_tape_draws"] == 0


def _source_bonus_spawn_fixture_env(
    scenario_name: str,
) -> tuple[VectorMultiplayerEnv, object, dict[str, object], list[dict[str, float]], int]:
    scenario = _load_scenario(scenario_name)
    player_count = int(scenario["player_count"])
    source_setup = scenario["source_setup"]
    room = source_setup["room"]
    random_entries = _fixture_random_entries(scenario)
    spawn_prefix = _spawn_prefix_values(player_count)
    tape = np.asarray(
        [[*spawn_prefix, *[entry["value"] for entry in random_entries]]],
        dtype=np.float64,
    )
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
        map_size=float(source_setup["map_size"]),
        max_score=int(room["max_score"]),
        decision_ms=1.0,
        body_capacity=4,
        event_capacity=16,
        timer_capacity=max(4, player_count),
        random_tape_capacity=tape.shape[1],
        event_mode="debug-event",
        natural_bonus_spawn=True,
        natural_bonus_type_codes=tuple(room["bonuses"]),
        natural_bonus_rate=float(room["bonus_rate"]),
    )
    reset_batch = env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_random_tape_values=tape,
        source_fixture_ref=scenario_name,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    _apply_source_fixture_state(env, scenario)
    env.decision_ms = 0.0
    return env, reset_batch, scenario, random_entries, len(spawn_prefix)


def _apply_source_fixture_state(
    env: VectorMultiplayerEnv,
    scenario: dict[str, object],
) -> None:
    source_setup = scenario["source_setup"]
    game = source_setup["game"]
    env.state["started"][0] = bool(game["started"])
    env.state["in_round"][0] = bool(game["in_round"])
    env.state["world_active"][0] = bool(game["world_active"])
    env.state["borderless"][0] = bool(game["borderless"])

    players = scenario["players"]
    for player_index, player in enumerate(players):
        initial = player["initial"]
        env.state["pos"][0, player_index] = (
            float(initial["x"]),
            float(initial["y"]),
        )
        env.state["prev_pos"][0, player_index] = env.state["pos"][0, player_index]
        env.state["heading"][0, player_index] = float(initial["angle_rad"])
        env.state["printing"][0, player_index] = bool(initial["printing"])
        env.state["alive"][0, player_index] = bool(initial.get("alive", True))

    _seed_fixture_world_bodies(env, scenario)
    _seed_fixture_active_bonuses(env, scenario)


def _seed_fixture_world_bodies(
    env: VectorMultiplayerEnv,
    scenario: dict[str, object],
) -> None:
    initial_state = scenario.get("initial_state", {})
    world_bodies = initial_state.get("world_bodies", [])
    env.state["body_active"][0, :] = False
    env.state["body_pos"][0, :, :] = 0.0
    env.state["body_radius"][0, :] = 0.0
    env.state["body_owner"][0, :] = -1
    env.state["body_num"][0, :] = -1
    env.state["body_insert_kind"][0, :] = -1
    env.state["body_write_cursor"][0] = len(world_bodies)
    env.state["world_body_count"][0] = len(world_bodies)
    env.state["world_active"][0] = bool(world_bodies) or bool(
        scenario["source_setup"]["game"]["world_active"]
    )
    for slot, body in enumerate(world_bodies):
        env.state["body_active"][0, slot] = True
        env.state["body_pos"][0, slot] = (float(body["x"]), float(body["y"]))
        env.state["body_radius"][0, slot] = float(body["radius"])
        env.state["body_owner"][0, slot] = _player_ref_to_index(str(body["player_id"]))
        env.state["body_num"][0, slot] = int(body["num"])
        env.state["body_insert_kind"][0, slot] = vector_runtime.BODY_KIND_NORMAL


def _seed_fixture_active_bonuses(
    env: VectorMultiplayerEnv,
    scenario: dict[str, object],
) -> None:
    initial_state = scenario.get("initial_state", {})
    active_bonuses = initial_state.get("active_bonuses", [])
    env.state["bonus_active"][0, :] = False
    env.state["bonus_type"][0, :] = vector_runtime.BONUS_TYPE_NONE
    env.state["bonus_id"][0, :] = -1
    env.state["bonus_pos"][0, :, :] = 0.0
    env.state["bonus_radius"][0, :] = 0.0
    for slot, bonus in enumerate(active_bonuses):
        env.state["bonus_active"][0, slot] = True
        env.state["bonus_type"][0, slot] = _bonus_type_code(str(bonus["type"]))
        env.state["bonus_id"][0, slot] = slot + 1
        env.state["bonus_pos"][0, slot] = (float(bonus["x"]), float(bonus["y"]))
        env.state["bonus_radius"][0, slot] = vector_runtime.SOURCE_BONUS_RADIUS
    env.state["bonus_count"][0] = len(active_bonuses)
    env.state["bonus_world_body_count"][0] = len(active_bonuses)
    env.state["bonus_world_active"][0] = bool(active_bonuses)
    env.state["bonus_next_id"][0] = len(active_bonuses) + 1


def _load_scenario(scenario_name: str) -> dict[str, object]:
    return json.loads((SCENARIO_ROOT / scenario_name).read_text(encoding="utf-8"))


def _fixture_random_entries(scenario: dict[str, object]) -> list[dict[str, float]]:
    sequence = scenario["source_setup"]["random"]["math_random_sequence"]
    entries: list[dict[str, float]] = []
    for entry in sequence:
        assert isinstance(entry, dict)
        entries.append({"label": str(entry["label"]), "value": float(entry["value"])})
    assert entries[0]["label"] == "bonus.start_delay"
    assert entries[1]["label"] == "bonus.next_delay_after_pop"
    return entries


def _spawn_prefix_values(player_count: int) -> list[float]:
    fixture_name_by_player_count = {
        2: "source_lifecycle_spawn_rng_2p_next_round.json",
        4: "source_lifecycle_spawn_rng_order_4p.json",
    }
    sequence = _load_scenario(fixture_name_by_player_count[player_count])[
        "source_setup"
    ]["random"]["math_random_sequence"]
    values: list[float] = []
    for entry in sequence[: player_count * 3]:
        if isinstance(entry, dict):
            values.append(float(entry["value"]))
        else:
            values.append(float(entry))
    return values


def _actions_for_source_step(step: dict[str, object]) -> np.ndarray:
    moves = [int(move["move"]) for move in step["moves"]]
    return np.asarray([moves], dtype=np.int16) + 1


def _bonus_type_code(name: str) -> int:
    return int(vector_runtime.BONUS_TYPE_NAME_BY_CODE.index(name))


def _player_ref_to_index(player_ref: str) -> int:
    assert player_ref.startswith("p")
    return int(player_ref[1:])
