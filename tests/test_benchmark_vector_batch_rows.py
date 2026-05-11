import importlib.util
from pathlib import Path
import sys

import numpy as np


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_vector_batch_rows.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_vector_batch_rows", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
batch_benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = batch_benchmark
_SPEC.loader.exec_module(batch_benchmark)


def _repeat_prepared_step(prepared_step, batch_size):
    return {
        "player_count": prepared_step["player_count"],
        "step_ms": np.full(
            batch_size,
            float(prepared_step["step_ms"]),
            dtype=np.float64,
        ),
        "source_moves": np.stack(
            [np.asarray(prepared_step["source_moves"], dtype=np.int8) for _ in range(batch_size)],
            axis=0,
        ),
        "print_manager_mode": np.full(
            batch_size,
            str(prepared_step.get("print_manager_mode", "none")),
            dtype=object,
        ),
        "timer_advance_ms": np.full(
            batch_size,
            float(prepared_step.get("timer_advance_ms", 0.0)),
            dtype=np.float64,
        ),
    }


def test_batch_preflight_matches_single_row_event_arrays_for_supported_fixtures():
    summary = batch_benchmark.benchmark_inputs(
        [
            "scenarios/environment/source_body_canary_batch.json",
            "scenarios/environment/source_borderless_wrap_step.json",
            "scenarios/environment/source_normal_wall_death_step.json",
        ],
        body_capacity=4,
        batch_sizes=(1,),
        repeat=1,
        warmup=0,
    )

    assert summary["summary"] == {
        "passed": 8,
        "failed": 0,
        "unsupported": 0,
        "batch_preflight_failed": False,
        "status": "pass",
    }
    for group in summary["groups"]:
        preflight = group["preflight"]
        assert preflight["state_match"] is True
        assert preflight["event_match"] is True
        assert preflight["event_mismatches"] == []
        assert preflight["event_counter_mismatches"] == []
        assert (
            preflight["batch_counters"]["events_emitted"]
            == preflight["expected_scalar_counters"]["events_emitted"]
        )


def test_batch_benchmark_compares_debug_event_and_no_event_modes():
    summary = batch_benchmark.benchmark_inputs(
        batch_benchmark.DEFAULT_PATHS,
        body_capacity=4,
        batch_sizes=(2,),
        event_modes=(
            batch_benchmark.EVENT_MODE_DEBUG,
            batch_benchmark.EVENT_MODE_NONE,
        ),
        repeat=1,
        warmup=0,
    )

    assert summary["summary"]["status"] == "pass"
    for group in summary["groups"]:
        assert group["preflight"]["event_match"] is True
        assert group["no_event_preflight"]["state_match"] is True
        assert len(group["event_mode_comparisons"]) == 1

        by_mode = {batch["event_mode"]: batch for batch in group["batches"]}
        debug_batch = by_mode[batch_benchmark.EVENT_MODE_DEBUG]
        no_event_batch = by_mode[batch_benchmark.EVENT_MODE_NONE]

        assert debug_batch["timed_counters"]["events_emitted"] > 0
        assert no_event_batch["timed_counters"]["events_emitted"] == 0
        assert no_event_batch["timing_sec"]["event_overhead_sec"] == 0


def test_default_paths_keep_delayed_start_timer_fixture_explicitly_excluded():
    assert batch_benchmark.DELAYED_START_TIMER_PATH not in batch_benchmark.DEFAULT_PATHS
    assert (
        "full two-tick reset/timer trace"
        in batch_benchmark.DEFAULT_EXCLUDED_PATHS[batch_benchmark.DELAYED_START_TIMER_PATH]
    )


def test_step_batched_arrays_uses_runtime_borderless_wrap(monkeypatch):
    fixture = batch_benchmark.seed_bridge.seed_fixture(
        "scenarios/environment/source_borderless_wrap_step.json",
        body_capacity=4,
    )
    initial_state = batch_benchmark._array_state_from_seed(fixture)
    prepared_step = batch_benchmark._prepare_fixture_array_step(fixture, step_index=0)
    template = {
        "initial_state": initial_state,
        "prepared_step": prepared_step,
        "player_count": 2,
        "scenario_id": fixture["scenario_id"],
    }
    batch = batch_benchmark._build_batch([template], 1)
    working_state = batch_benchmark.vector_compare.copy_array_state(batch["initial_state"])
    original = batch_benchmark.vector_runtime.apply_borderless_wrap
    calls = []

    def spy_apply_borderless_wrap(state, *, player, live_mask):
        calls.append((player, live_mask.copy()))
        return original(state, player=player, live_mask=live_mask)

    monkeypatch.setattr(
        batch_benchmark.vector_runtime,
        "apply_borderless_wrap",
        spy_apply_borderless_wrap,
    )

    counters = batch_benchmark.step_batched_arrays(
        working_state,
        batch["prepared_batch"],
        event_mode=batch_benchmark.EVENT_MODE_NONE,
    )

    assert counters["borderless_wraps"] == 1
    assert [player for player, _live_mask in calls] == [1, 0]


def test_step_batched_arrays_uses_runtime_normal_wall_hit_mask(monkeypatch):
    fixture = batch_benchmark.seed_bridge.seed_fixture(
        "scenarios/environment/source_normal_wall_death_step.json",
        body_capacity=4,
    )
    initial_state = batch_benchmark._array_state_from_seed(fixture)
    prepared_step = batch_benchmark._prepare_fixture_array_step(fixture, step_index=0)
    template = {
        "initial_state": initial_state,
        "prepared_step": prepared_step,
        "player_count": 2,
        "scenario_id": fixture["scenario_id"],
    }
    batch = batch_benchmark._build_batch([template], 1)
    working_state = batch_benchmark.vector_compare.copy_array_state(batch["initial_state"])
    original = batch_benchmark.vector_runtime.normal_wall_hit_mask
    calls = []

    def spy_normal_wall_hit_mask(state, *, player, live_mask):
        mask = original(state, player=player, live_mask=live_mask)
        calls.append((player, live_mask.copy(), mask.copy()))
        return mask

    monkeypatch.setattr(
        batch_benchmark.vector_runtime,
        "normal_wall_hit_mask",
        spy_normal_wall_hit_mask,
    )

    counters = batch_benchmark.step_batched_arrays(
        working_state,
        batch["prepared_batch"],
        event_mode=batch_benchmark.EVENT_MODE_NONE,
    )

    assert counters["normal_wall_deaths"] == 1
    assert [player for player, _live_mask, _hit_mask in calls] == [1, 0]
    assert [hit_mask.tolist() for _player, _live_mask, hit_mask in calls] == [[False], [True]]


def test_batched_delayed_start_pre_step_timer_second_tick_matches_scalar():
    fixture = batch_benchmark.seed_bridge.seed_fixture(
        batch_benchmark.DELAYED_START_TIMER_PATH,
        body_capacity=4,
    )
    scenario_id = fixture["scenario_id"]
    initial_state = batch_benchmark._array_state_from_seed(fixture)
    first_step = batch_benchmark._prepare_fixture_array_step(fixture, step_index=0)
    second_step = batch_benchmark._prepare_fixture_array_step(fixture, step_index=1)
    template = {
        "initial_state": initial_state,
        "prepared_step": first_step,
        "player_count": 1,
        "scenario_id": scenario_id,
    }
    batch = batch_benchmark._build_batch([template], 2)
    working_state = batch_benchmark.vector_compare.copy_array_state(batch["initial_state"])

    first_counters = batch_benchmark.step_batched_arrays(
        working_state,
        batch["prepared_batch"],
    )
    second_counters = batch_benchmark.step_batched_arrays(
        working_state,
        _repeat_prepared_step(second_step, batch_size=2),
    )

    expected_state = batch_benchmark._array_state_from_seed(fixture)
    batch_benchmark._step_single_template(expected_state, first_step)
    expected_second_counters = batch_benchmark._step_single_template(expected_state, second_step)

    assert first_counters["pre_step_timer_advances"] == 2
    assert first_counters["pre_step_timer_fires"] == 0
    assert second_counters["pre_step_timer_advances"] == 2
    assert second_counters["pre_step_timer_fires"] == 2
    assert second_counters["print_manager_delayed_start_fires"] == 2
    assert second_counters["print_manager_delayed_start_points"] == 2
    assert second_counters["events_emitted"] == 6
    assert expected_second_counters["events_emitted"] == 3
    assert working_state["event_type"][0, :3].tolist() == [
        batch_benchmark.vector_compare.EVENT_POINT,
        batch_benchmark.vector_compare.EVENT_PROPERTY,
        batch_benchmark.vector_compare.EVENT_POSITION,
    ]
    assert working_state["event_type"][1, :3].tolist() == [
        batch_benchmark.vector_compare.EVENT_POINT,
        batch_benchmark.vector_compare.EVENT_PROPERTY,
        batch_benchmark.vector_compare.EVENT_POSITION,
    ]

    for row_index in range(2):
        assert (
            batch_benchmark._row_state_mismatches(
                expected_state,
                working_state,
                row_index=row_index,
                scenario_id=scenario_id,
                max_mismatches=8,
            )
            == []
        )
        assert (
            batch_benchmark._row_event_mismatches(
                expected_state,
                working_state,
                row_index=row_index,
                scenario_id=scenario_id,
                max_mismatches=8,
            )
            == []
        )


def test_batched_delayed_start_timer_no_event_mode_updates_state_without_events():
    fixture = batch_benchmark.seed_bridge.seed_fixture(
        batch_benchmark.DELAYED_START_TIMER_PATH,
        body_capacity=4,
    )
    scenario_id = fixture["scenario_id"]
    initial_state = batch_benchmark._array_state_from_seed(fixture)
    first_step = batch_benchmark._prepare_fixture_array_step(fixture, step_index=0)
    second_step = batch_benchmark._prepare_fixture_array_step(fixture, step_index=1)
    template = {
        "initial_state": initial_state,
        "prepared_step": first_step,
        "player_count": 1,
        "scenario_id": scenario_id,
    }
    batch = batch_benchmark._build_batch([template], 2)
    working_state = batch_benchmark.vector_compare.copy_array_state(batch["initial_state"])

    batch_benchmark.step_batched_arrays(
        working_state,
        batch["prepared_batch"],
        event_mode=batch_benchmark.EVENT_MODE_NONE,
    )
    second_counters = batch_benchmark.step_batched_arrays(
        working_state,
        _repeat_prepared_step(second_step, batch_size=2),
        event_mode=batch_benchmark.EVENT_MODE_NONE,
    )

    expected_state = batch_benchmark._array_state_from_seed(fixture)
    batch_benchmark._step_single_template(expected_state, first_step)
    batch_benchmark._step_single_template(expected_state, second_step)

    assert second_counters["pre_step_timer_fires"] == 2
    assert second_counters["events_emitted"] == 0
    assert working_state["event_count"].tolist() == [0, 0]
    for row_index in range(2):
        assert (
            batch_benchmark._row_state_mismatches(
                expected_state,
                working_state,
                row_index=row_index,
                scenario_id=scenario_id,
                max_mismatches=8,
            )
            == []
        )
