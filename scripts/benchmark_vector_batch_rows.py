"""Benchmark stacked fixture-seeded vector rows in one array-step call.

This is the B>1 companion to ``benchmark_vector_array_steps.py``. It keeps the
source/common-trace proof as a preflight, then stacks compatible fixture seeds
along the leading row axis and steps many rows with one batch call. The hot loop
does not call the source runner, common-trace projection, or per-fixture scalar
step helper.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
for root in (SCRIPT_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import compare_vector_arrays_to_fidelity as vector_compare  # noqa: E402
import seed_vector_state_from_fixtures as seed_bridge  # noqa: E402
from curvyzero.env import vector_runtime  # noqa: E402


SCHEMA_VERSION = "curvyzero_vector_fixture_batch_row_timing/v1"
BENCHMARK_ID = "fixture_seeded_numpy_batch_rows_supported_transition"
BODY_KIND_NORMAL = 0
BODY_KIND_IMPORTANT = 1
BODY_KIND_DEATH = 2
DEFAULT_PATHS = (
    "scenarios/environment/source_body_canary_batch.json",
    "scenarios/environment/source_borderless_wrap_step.json",
    "scenarios/environment/source_normal_wall_death_step.json",
    "scenarios/environment/source_print_manager_no_toggle_control_step.json",
    "scenarios/environment/source_print_manager_print_to_hole_step.json",
    "scenarios/environment/source_print_manager_hole_to_print_step.json",
    "scenarios/environment/source_print_manager_exact_zero_toggle_step.json",
    "scenarios/environment/source_print_manager_active_stop_on_death_step.json",
    "scenarios/environment/source_print_manager_active_hole_stop_on_death_step.json",
    "scenarios/environment/source_print_manager_body_collision_stop_on_death_step.json",
    "scenarios/environment/source_trail_gap_batch.json",
)
DELAYED_START_TIMER_PATH = (
    "scenarios/environment/source_print_manager_delayed_start_timer_step.json"
)
DEFAULT_EXCLUDED_PATHS = {
    DELAYED_START_TIMER_PATH: (
        "delayed-start comparator support is a full two-tick reset/timer trace; "
        "the default batch-row benchmark remains the supported one-step timing slice"
    ),
}
DEFAULT_BATCH_SIZES = (1, 8, 32, 128)
EVENT_MODE_DEBUG = vector_runtime.EVENT_MODE_DEBUG
EVENT_MODE_NONE = vector_runtime.EVENT_MODE_NONE
EVENT_MODE_CHOICES = tuple(sorted(vector_runtime.EVENT_MODES))
DEFAULT_EVENT_MODES = (EVENT_MODE_DEBUG,)
NEXT_PRODUCTION_GATES = [
    "broaden compact event rows beyond the current supported fixture slice",
    "PrintManager delayed-start full-trace batch-row timing and broader trail-gap state transitions",
    "observation, reward, done/truncated, and reset/autoreset arrays",
    "policy/search/MCTS batch connection and replay staging",
]
EVENT_ARRAY_NAMES = frozenset(
    {
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
    }
)
EVENT_COUNTER_NAMES = frozenset({"events_emitted", "event_overflow_attempts"})


def benchmark_inputs(
    paths: Sequence[str | Path],
    *,
    body_capacity: int = 4,
    step_index: int = 0,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    event_modes: Sequence[str] = DEFAULT_EVENT_MODES,
    repeat: int = 1_000,
    warmup: int = 100,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Benchmark supported fixture rows stacked into fixed-profile batches."""

    if body_capacity < 0:
        raise vector_compare.VectorCompareError("body_capacity must be zero or greater")
    if step_index < 0:
        raise vector_compare.VectorCompareError("step_index must be zero or greater")
    if repeat <= 0:
        raise vector_compare.VectorCompareError("repeat must be greater than zero")
    if warmup < 0:
        raise vector_compare.VectorCompareError("warmup must be zero or greater")
    batch_sizes = tuple(int(size) for size in batch_sizes)
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise vector_compare.VectorCompareError("batch_sizes must contain positive integers")
    event_modes = _normalize_event_modes(event_modes)

    wall_started = time.perf_counter()
    timers = {
        "seed_inputs_sec": 0.0,
        "source_preflight_sec": 0.0,
        "array_prepare_sec": 0.0,
        "batch_state_preflight_sec": 0.0,
    }

    started = time.perf_counter()
    seeded = seed_bridge.seed_inputs(paths, body_capacity=body_capacity)
    timers["seed_inputs_sec"] = time.perf_counter() - started

    templates: list[dict[str, Any]] = []
    fixture_results: list[dict[str, Any]] = []
    for fixture in seeded["fixtures"]:
        scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")

        started = time.perf_counter()
        source_preflight = _compare_fixture_seed(
            fixture,
            step_index=step_index,
            require_verified=require_verified,
        )
        timers["source_preflight_sec"] += time.perf_counter() - started

        fixture_result = {
            "scenario_id": scenario_id,
            "path": fixture.get("path"),
            "status": source_preflight["status"],
            "match": source_preflight["match"],
            "compared_fields": len(source_preflight["compared_fields"]),
            "skipped_fields": len(source_preflight["skipped_fields"]),
            "unsupported_mechanics": source_preflight["unsupported_mechanics"],
            "covered_mechanics": source_preflight["covered_mechanics"],
        }
        if source_preflight["status"] == "fail":
            fixture_result["mismatches"] = source_preflight["mismatches"]
        fixture_results.append(fixture_result)
        if source_preflight["status"] != "pass":
            continue

        started = time.perf_counter()
        initial_state = _array_state_from_seed(fixture)
        prepared_step = _prepare_fixture_array_step(
            fixture,
            step_index=step_index,
        )
        timers["array_prepare_sec"] += time.perf_counter() - started
        profile = _mapping(fixture.get("profile"), "fixture.profile")
        templates.append(
            {
                "fixture": fixture,
                "scenario_id": scenario_id,
                "group_key": _group_key(profile, initial_state),
                "player_count": _int(profile.get("P"), "fixture.profile.P"),
                "body_capacity": _int(profile.get("K"), "fixture.profile.K"),
                "initial_state": initial_state,
                "prepared_step": prepared_step,
            }
        )

    passed = sum(result["status"] == "pass" for result in fixture_results)
    failed = sum(result["status"] == "fail" for result in fixture_results)
    unsupported = sum(result["status"] == "unsupported" for result in fixture_results)

    groups_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for template in templates:
        groups_by_key[template["group_key"]].append(template)

    groups = []
    can_time = failed == 0 and bool(templates)
    for group_key, group_templates in sorted(groups_by_key.items()):
        started = time.perf_counter()
        preflight = _batch_state_preflight(group_templates)
        timers["batch_state_preflight_sec"] += time.perf_counter() - started
        no_event_preflight = None
        if EVENT_MODE_NONE in event_modes:
            started = time.perf_counter()
            no_event_preflight = _no_event_state_preflight(group_templates)
            timers["batch_state_preflight_sec"] += time.perf_counter() - started

        batch_results = []
        can_time_group = (
            can_time
            and preflight["match"]
            and (no_event_preflight is None or no_event_preflight["match"])
        )
        if can_time_group:
            for batch_size in batch_sizes:
                for event_mode in event_modes:
                    batch_results.append(
                        _benchmark_group_batch(
                            group_templates,
                            batch_size=batch_size,
                            event_mode=event_mode,
                            repeat=repeat,
                            warmup=warmup,
                        )
                    )

        groups.append(
            {
                "group_id": group_key,
                "player_count": group_templates[0]["player_count"],
                "body_capacity": group_templates[0]["body_capacity"],
                "supported_fixture_count": len(group_templates),
                "fixture_ids": [template["scenario_id"] for template in group_templates],
                "preflight": preflight,
                "no_event_preflight": no_event_preflight,
                "batches": batch_results,
                "event_mode_comparisons": _batch_event_mode_comparisons(batch_results),
            }
        )

    batch_preflight_failed = any(
        not group["preflight"]["match"]
        or (group["no_event_preflight"] is not None and not group["no_event_preflight"]["match"])
        for group in groups
    )
    wall_elapsed_sec = time.perf_counter() - wall_started
    return {
        "schema": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "source_fidelity_claim": (
            "source/common-trace state and fixed event rows are compared once per "
            "supported fixture before timing; batch preflight compares the B>1 "
            "row path against the single-row comparator output. Repeated timing "
            "covers fixture-seeded NumPy rows stacked into fixed-profile B>1 calls"
        ),
        "trust_level": (
            "This is a fixture-seeded batch-row microbenchmark. It is not a full "
            "environment, not a GPU claim, and not production self-play throughput."
        ),
        "batch_path_label": (
            "The hot call takes one stacked state with leading row axis B plus "
            "row-specific source_moves[B,P] and step_ms[B]. Rows are grouped by "
            "fixed P/K array profile; P=1, P=2, and P=3 fixtures are not padded "
            "together."
        ),
        "config": {
            "paths": [str(path) for path in paths],
            "body_capacity": body_capacity,
            "step_index": step_index,
            "batch_sizes": list(batch_sizes),
            "event_modes": list(event_modes),
            "repeat": repeat,
            "warmup": warmup,
            "require_verified": require_verified,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "input_count": seeded["input_count"],
        "fixture_count": len(fixture_results),
        "supported_fixture_count": len(templates),
        "summary": {
            "passed": passed,
            "failed": failed,
            "unsupported": unsupported,
            "batch_preflight_failed": batch_preflight_failed,
            "status": (
                "fail"
                if failed or batch_preflight_failed
                else "pass"
                if passed and not unsupported
                else "mixed"
            ),
        },
        "timing_sec": {
            **timers,
            "setup_sec": timers["seed_inputs_sec"] + timers["array_prepare_sec"],
            "preflight_total_sec": timers["source_preflight_sec"]
            + timers["batch_state_preflight_sec"],
            "wall_elapsed_sec": wall_elapsed_sec,
        },
        "fixtures": fixture_results,
        "groups": groups,
        "hot_loop_exclusions": {
            "source_trace_calls": 0,
            "projection_calls": 0,
            "comparison_calls": 0,
            "per_fixture_scalar_step_calls": 0,
            "note": (
                "timed loop performs reset-copy plus one batched NumPy row step per "
                "repeat; source, common-trace comparison, and scalar B=1 helpers "
                "are preflight only"
            ),
        },
        "known_fake_or_incomplete": [
            "batch rows are made by cycling the current supported fixture seeds",
            "fixtures are grouped by fixed P/K profile instead of one padded mixed-P production batch",
            "only the currently supported one-step mechanics are timed",
            *NEXT_PRODUCTION_GATES,
        ],
        "next_production_gates": NEXT_PRODUCTION_GATES,
    }


def step_batched_arrays(
    state: dict[str, np.ndarray],
    prepared_batch: Mapping[str, Any],
    *,
    phase_timers: dict[str, float] | None = None,
    event_mode: str = EVENT_MODE_DEBUG,
) -> dict[str, int]:
    """Run one source-ordered in-place array tick for B stacked rows."""

    try:
        step_input = vector_runtime.prepare_step_input(
            state,
            prepared_batch,
            event_mode=event_mode,
        )
        if phase_timers is None:
            return vector_runtime.step_many(step_input)
        return vector_runtime._step_many_kernel(  # noqa: SLF001
            step_input,
            phase_timers=phase_timers,
        )
    except vector_runtime.VectorRuntimeError as exc:
        raise vector_compare.VectorCompareError(str(exc)) from exc


def print_plain(summary: Mapping[str, Any]) -> None:
    counts = _mapping(summary.get("summary"), "summary")
    timing = _mapping(summary.get("timing_sec"), "timing_sec")
    config = _mapping(summary.get("config"), "config")
    print(f"benchmark={summary['benchmark_id']}")
    print(f"trust_level={summary['trust_level']}")
    print(f"batch_path={summary['batch_path_label']}")
    print("event_modes=" + ",".join(str(mode) for mode in config["event_modes"]))
    print(
        "summary="
        f"passed:{counts['passed']} failed:{counts['failed']} "
        f"unsupported:{counts['unsupported']} "
        f"batch_preflight_failed:{counts['batch_preflight_failed']}"
    )
    print(
        "preflight_sec="
        f"source:{timing['source_preflight_sec']:.6f} "
        f"batch_state:{timing['batch_state_preflight_sec']:.6f} "
        f"setup:{timing['setup_sec']:.6f}"
    )
    for group in _list(summary.get("groups"), "groups"):
        group = _mapping(group, "group")
        preflight = _mapping(group["preflight"], "group.preflight")
        print(
            "group="
            f"{group['group_id']} P={group['player_count']} K={group['body_capacity']} "
            f"fixtures={group['supported_fixture_count']} "
            f"batch_state_match={preflight['state_match']} "
            f"batch_event_match={preflight['event_match']}"
        )
        no_event_preflight = group.get("no_event_preflight")
        if no_event_preflight is not None:
            no_event_preflight = _mapping(
                no_event_preflight,
                "group.no_event_preflight",
            )
            print(
                "no_event_preflight="
                f"state_match:{no_event_preflight['state_match']} "
                f"match:{no_event_preflight['match']}"
            )
        if not preflight["match"]:
            print(f"batch_preflight_mismatches={preflight['mismatches']}")
            continue
        for batch in _list(group.get("batches"), "group.batches"):
            batch = _mapping(batch, "batch")
            rates = _mapping(batch["rates"], "batch.rates")
            timing_sec = _mapping(batch["timing_sec"], "batch.timing_sec")
            phase = _mapping(batch["phase_timing_sec"], "batch.phase_timing_sec")
            top_phase = _top_phase_label(phase)
            print(
                "batch="
                f"event_mode:{batch['event_mode']} "
                f"B:{batch['batch_size']} repeats:{batch['repeat']} "
                f"rows:{rates['timed_rows']} "
                f"rows_per_step_sec:{rates['rows_per_step_sec']:.1f} "
                f"rows_per_reset_plus_step_sec:{rates['rows_per_reset_plus_step_sec']:.1f} "
                f"reset_sec:{timing_sec['reset_copy_sec']:.6f} "
                f"step_sec:{timing_sec['env_step_sec']:.6f} "
                f"event_sec:{timing_sec['event_overhead_sec']:.6f} "
                f"event_pct:{rates['event_overhead_step_pct']:.1f} "
                f"top_phase:{top_phase}"
            )
        for comparison in _list(
            group.get("event_mode_comparisons", []),
            "group.event_mode_comparisons",
        ):
            comparison = _mapping(comparison, "event_mode_comparison")
            print(
                "event_compare="
                f"B:{comparison['batch_size']} rows:{comparison['timed_rows']} "
                f"debug_step_sec:{comparison['debug_event_step_sec']:.6f} "
                f"no_event_step_sec:{comparison['no_event_step_sec']:.6f} "
                f"debug_minus_no_event_step_sec:"
                f"{comparison['debug_minus_no_event_step_sec']:.6f} "
                f"debug_minus_no_event_step_pct:"
                f"{comparison['debug_minus_no_event_step_pct']:.1f} "
                f"debug_rows_per_step_sec:"
                f"{comparison['debug_event_rows_per_step_sec']:.1f} "
                f"no_event_rows_per_step_sec:"
                f"{comparison['no_event_rows_per_step_sec']:.1f} "
                f"no_event_step_speedup_vs_debug:"
                f"{comparison['no_event_step_speedup_vs_debug']:.3f} "
                f"reset_plus_step_speedup_vs_debug:"
                f"{comparison['no_event_reset_plus_step_speedup_vs_debug']:.3f}"
            )
    print("known_fake_or_incomplete=" + "; ".join(summary["known_fake_or_incomplete"]))


def _compare_fixture_seed(
    fixture: Mapping[str, Any],
    *,
    step_index: int,
    require_verified: bool,
) -> dict[str, Any]:
    scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")
    support = vector_compare.fixture_transition_support(
        fixture,
        step_index=step_index,
        require_verified=require_verified,
    )
    base = {
        "scenario_id": scenario_id,
        "path": fixture.get("path"),
        "verification": fixture.get("verification"),
        "supported_transition": support["supported"],
        "covered_mechanics": support["covered_mechanics"],
        "unsupported_mechanics": support["unsupported_mechanics"],
    }
    if not support["supported"]:
        return {
            **base,
            "status": "unsupported",
            "match": None,
            "mismatches": [],
            "compared_fields": [],
            "skipped_fields": [],
            "array_counters": {},
        }

    source_trace = vector_compare.source_common_trace_for_fixture(fixture)
    state = _array_state_from_seed(fixture)
    counters = _step_single_template(
        state,
        _prepare_fixture_array_step(fixture, step_index=step_index),
    )
    actual_trace = vector_compare.project_array_state_to_common_trace(
        fixture,
        state,
        step_index=step_index,
    )
    compare = vector_compare.compare_common_trace_fields(
        actual_trace,
        source_trace,
        fixture.get("comparison", {}),
    )
    return {
        **base,
        "status": "pass" if compare["match"] else "fail",
        "match": compare["match"],
        "mismatches": compare["mismatches"],
        "compared_fields": compare["compared_fields"],
        "skipped_fields": compare["skipped_fields"],
        "array_counters": counters,
    }


def _array_state_from_seed(fixture: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return vector_compare.array_state_from_seed(fixture)


def _array_from_payload(payload: Any, name: str) -> np.ndarray:
    data = _mapping(payload, f"arrays.{name}")
    dtype = _numpy_dtype(_string(data.get("dtype"), f"arrays.{name}.dtype"))
    return np.asarray(data.get("values"), dtype=dtype)


def _project_array_state_to_common_trace(
    fixture: Mapping[str, Any],
    state: Mapping[str, np.ndarray],
    *,
    step_index: int,
) -> dict[str, Any]:
    profile = _mapping(fixture.get("profile"), "fixture.profile")
    player_ids = _list(profile.get("player_ids"), "fixture.profile.player_ids")
    action_schedule = _list(fixture.get("action_schedule"), "fixture.action_schedule")
    raw_step = _mapping(action_schedule[step_index], f"fixture.action_schedule[{step_index}]")
    players = []
    for player_index, player_id in enumerate(player_ids):
        last_trail_point: list[float] | None = None
        if bool(state["has_visible_trail_last"][0, player_index]):
            last_trail_point = [
                float(state["visible_trail_last_pos"][0, player_index, 0]),
                float(state["visible_trail_last_pos"][0, player_index, 1]),
            ]
        players.append(
            {
                "player_id": str(player_id),
                "x": float(state["pos"][0, player_index, 0]),
                "y": float(state["pos"][0, player_index, 1]),
                "angle": float(state["heading"][0, player_index]),
                "alive": bool(state["alive"][0, player_index]),
                "score": int(state["score"][0, player_index]),
                "roundScore": int(state["round_score"][0, player_index]),
                "trailPointCount": int(state["visible_trail_count"][0, player_index]),
                "lastTrailPoint": last_trail_point,
                "bodyNum": int(state["live_body_num"][0, player_index]),
                "bodyCount": int(state["body_count"][0, player_index]),
                "printing": bool(state["printing"][0, player_index]),
            }
        )
    return {
        "schema": "curvyzero_common_trace/v1",
        "scenario_id": _string(fixture.get("scenario_id"), "fixture.scenario_id"),
        "map_size": float(state["map_size"][0]),
        "steps": [
            {
                "step_index": step_index,
                "step_ms": float(raw_step["step_ms"]),
                "players": players,
                "worldBodyCount": int(state["world_body_count"][0]),
            }
        ],
    }


def _source_trace_without_events(trace: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    stripped = dict(trace)
    stripped_steps = []
    skipped = []
    for step_index, raw_step in enumerate(_list(trace.get("steps"), "source_trace.steps")):
        step = dict(_mapping(raw_step, f"source_trace.steps[{step_index}]"))
        if "events" in step:
            skipped.append(f"$.steps[{step_index}].events")
            step.pop("events")
        stripped_steps.append(step)
    stripped["steps"] = stripped_steps
    return stripped, skipped


def _prepare_fixture_array_step(
    fixture: Mapping[str, Any],
    *,
    step_index: int = 0,
) -> dict[str, Any]:
    return dict(vector_compare.prepare_fixture_array_step(fixture, step_index=step_index))


def _step_single_template(
    state: dict[str, np.ndarray],
    prepared_step: Mapping[str, Any],
) -> dict[str, int]:
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
    return step_batched_arrays(state, prepared_batch)


def _benchmark_group_batch(
    templates: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    event_mode: str,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    event_mode = _normalize_event_mode(event_mode)
    batch = _build_batch(templates, batch_size)
    warmup_working_state = vector_compare.copy_array_state(batch["initial_state"])
    warmup_timers, warmup_counters = _run_repeated_batch_steps(
        warmup_working_state,
        batch["initial_state"],
        batch["prepared_batch"],
        repeat=warmup,
        collect_phase_timing=False,
        event_mode=event_mode,
    )

    working_state = vector_compare.copy_array_state(batch["initial_state"])
    timed_timers, timed_counters = _run_repeated_batch_steps(
        working_state,
        batch["initial_state"],
        batch["prepared_batch"],
        repeat=repeat,
        collect_phase_timing=True,
        event_mode=event_mode,
    )

    timed_rows = batch_size * repeat
    timed_loop_sec = timed_timers["reset_copy_sec"] + timed_timers["env_step_sec"]
    event_overhead_sec = _event_phase_sec(timed_timers["phase_timing_sec"])
    timed_timers["event_overhead_sec"] = event_overhead_sec
    return {
        "batch_size": batch_size,
        "event_mode": event_mode,
        "events_enabled": _events_enabled(event_mode),
        "repeat": repeat,
        "warmup": warmup,
        "row_source_counts": batch["row_source_counts"],
        "timing_sec": timed_timers,
        "phase_timing_sec": timed_timers["phase_timing_sec"],
        "rates": {
            "warmup_rows": batch_size * warmup,
            "timed_rows": timed_rows,
            "rows_per_step_sec": _rate(timed_rows, timed_timers["env_step_sec"]),
            "rows_per_reset_plus_step_sec": _rate(timed_rows, timed_loop_sec),
            "event_overhead_step_pct": _rate(
                event_overhead_sec * 100.0, timed_timers["env_step_sec"]
            ),
        },
        "warmup_counters": warmup_counters,
        "timed_counters": timed_counters,
    }


def _run_repeated_batch_steps(
    working_state: dict[str, np.ndarray],
    initial_state: Mapping[str, np.ndarray],
    prepared_batch: Mapping[str, Any],
    *,
    repeat: int,
    collect_phase_timing: bool,
    event_mode: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    timers: dict[str, Any] = {
        "reset_copy_sec": 0.0,
        "env_step_sec": 0.0,
        "phase_timing_sec": _empty_phase_timers(),
    }
    counters: dict[str, int] = {}
    for _ in range(repeat):
        started = time.perf_counter()
        vector_compare.reset_array_state(working_state, initial_state)
        timers["reset_copy_sec"] += time.perf_counter() - started

        phase_timers = timers["phase_timing_sec"] if collect_phase_timing else None
        started = time.perf_counter()
        step_counters = step_batched_arrays(
            working_state,
            prepared_batch,
            phase_timers=phase_timers,
            event_mode=event_mode,
        )
        timers["env_step_sec"] += time.perf_counter() - started
        _add_counters(counters, step_counters)
    return timers, counters


def _batch_state_preflight(templates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = _build_batch(templates, len(templates))
    working_state = vector_compare.copy_array_state(batch["initial_state"])
    batch_counters = step_batched_arrays(working_state, batch["prepared_batch"])
    expected_counters: dict[str, int] = {}
    state_mismatches: list[dict[str, Any]] = []
    event_mismatches: list[dict[str, Any]] = []
    for row_index, template in enumerate(templates):
        expected_state = vector_compare.copy_array_state(template["initial_state"])
        scalar_counters = _step_single_template(
            expected_state,
            template["prepared_step"],
        )
        _add_counters(expected_counters, scalar_counters)
        state_mismatches.extend(
            _row_state_mismatches(
                expected_state,
                working_state,
                row_index=row_index,
                scenario_id=str(template["scenario_id"]),
                max_mismatches=max(0, 8 - len(state_mismatches)),
            )
        )
        event_mismatches.extend(
            _row_event_mismatches(
                expected_state,
                working_state,
                row_index=row_index,
                scenario_id=str(template["scenario_id"]),
                max_mismatches=max(0, 8 - len(event_mismatches)),
            )
        )
        if len(state_mismatches) >= 8 and len(event_mismatches) >= 8:
            break

    counter_mismatches = []
    event_counter_mismatches = []
    for key in sorted(set(expected_counters) | set(batch_counters)):
        if int(expected_counters.get(key, 0)) != int(batch_counters.get(key, 0)):
            detail = {
                "counter": key,
                "expected": int(expected_counters.get(key, 0)),
                "actual": int(batch_counters.get(key, 0)),
            }
            counter_mismatches.append(detail)
            if key in EVENT_COUNTER_NAMES:
                event_counter_mismatches.append(detail)
    state_counter_mismatches = [
        mismatch
        for mismatch in counter_mismatches
        if mismatch["counter"] not in EVENT_COUNTER_NAMES
    ]
    state_match = not state_mismatches and not state_counter_mismatches
    event_match = not event_mismatches and not event_counter_mismatches
    return {
        "row_count": len(templates),
        "match": state_match and event_match,
        "state_match": state_match,
        "event_match": event_match,
        "mismatches": [*state_mismatches, *event_mismatches],
        "state_mismatches": state_mismatches,
        "event_mismatches": event_mismatches,
        "counter_mismatches": counter_mismatches,
        "event_counter_mismatches": event_counter_mismatches,
        "batch_counters": batch_counters,
        "expected_scalar_counters": expected_counters,
    }


def _no_event_state_preflight(templates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = _build_batch(templates, len(templates))
    working_state = vector_compare.copy_array_state(batch["initial_state"])
    batch_counters = step_batched_arrays(
        working_state,
        batch["prepared_batch"],
        event_mode=EVENT_MODE_NONE,
    )
    expected_counters: dict[str, int] = {}
    state_mismatches: list[dict[str, Any]] = []
    for row_index, template in enumerate(templates):
        expected_state = vector_compare.copy_array_state(template["initial_state"])
        scalar_counters = _step_single_template(
            expected_state,
            template["prepared_step"],
        )
        _add_counters(expected_counters, scalar_counters)
        state_mismatches.extend(
            _row_state_mismatches(
                expected_state,
                working_state,
                row_index=row_index,
                scenario_id=str(template["scenario_id"]),
                max_mismatches=max(0, 8 - len(state_mismatches)),
            )
        )
        if len(state_mismatches) >= 8:
            break

    counter_mismatches = [
        {
            "counter": key,
            "expected": int(expected_counters.get(key, 0)),
            "actual": int(batch_counters.get(key, 0)),
        }
        for key in sorted(set(expected_counters) | set(batch_counters))
        if key not in EVENT_COUNTER_NAMES
        and int(expected_counters.get(key, 0)) != int(batch_counters.get(key, 0))
    ]
    state_match = not state_mismatches and not counter_mismatches
    return {
        "row_count": len(templates),
        "event_mode": EVENT_MODE_NONE,
        "match": state_match,
        "state_match": state_match,
        "state_mismatches": state_mismatches,
        "counter_mismatches": counter_mismatches,
        "batch_counters": batch_counters,
        "expected_scalar_counters": expected_counters,
        "note": "compares non-event state and non-event counters against scalar debug-event output",
    }


def _batch_event_mode_comparisons(
    batch_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_size: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for batch in batch_results:
        by_size[int(batch["batch_size"])][str(batch["event_mode"])] = batch

    comparisons = []
    for batch_size, by_mode in sorted(by_size.items()):
        debug_batch = by_mode.get(EVENT_MODE_DEBUG)
        no_event_batch = by_mode.get(EVENT_MODE_NONE)
        if debug_batch is None or no_event_batch is None:
            continue
        debug_timing = _mapping(debug_batch["timing_sec"], "debug_batch.timing_sec")
        no_event_timing = _mapping(no_event_batch["timing_sec"], "no_event_batch.timing_sec")
        debug_rates = _mapping(debug_batch["rates"], "debug_batch.rates")
        no_event_rates = _mapping(no_event_batch["rates"], "no_event_batch.rates")
        debug_step_sec = float(debug_timing["env_step_sec"])
        no_event_step_sec = float(no_event_timing["env_step_sec"])
        debug_loop_sec = float(debug_timing["reset_copy_sec"]) + debug_step_sec
        no_event_loop_sec = float(no_event_timing["reset_copy_sec"]) + no_event_step_sec
        step_delta = debug_step_sec - no_event_step_sec
        comparisons.append(
            {
                "batch_size": batch_size,
                "timed_rows": int(debug_rates["timed_rows"]),
                "debug_event_step_sec": debug_step_sec,
                "no_event_step_sec": no_event_step_sec,
                "debug_minus_no_event_step_sec": step_delta,
                "debug_minus_no_event_step_pct": _rate(step_delta * 100.0, debug_step_sec),
                "debug_event_rows_per_step_sec": float(debug_rates["rows_per_step_sec"]),
                "no_event_rows_per_step_sec": float(no_event_rates["rows_per_step_sec"]),
                "no_event_step_speedup_vs_debug": _rate(
                    float(no_event_rates["rows_per_step_sec"]),
                    float(debug_rates["rows_per_step_sec"]),
                ),
                "debug_event_reset_plus_step_sec": debug_loop_sec,
                "no_event_reset_plus_step_sec": no_event_loop_sec,
                "no_event_reset_plus_step_speedup_vs_debug": _rate(
                    float(no_event_rates["rows_per_reset_plus_step_sec"]),
                    float(debug_rates["rows_per_reset_plus_step_sec"]),
                ),
            }
        )
    return comparisons


def _build_batch(
    templates: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    if not templates:
        raise vector_compare.VectorCompareError("cannot build a batch from no templates")
    row_templates = [templates[index % len(templates)] for index in range(batch_size)]
    states = [template["initial_state"] for template in row_templates]
    initial_state = {
        name: np.concatenate([state[name] for state in states], axis=0) for name in states[0]
    }
    player_count = _int(row_templates[0]["player_count"], "template.player_count")
    step_ms = np.asarray(
        [
            _number(template["prepared_step"]["step_ms"], "prepared_step.step_ms")
            for template in row_templates
        ],
        dtype=np.float64,
    )
    source_moves = np.stack(
        [
            np.asarray(template["prepared_step"]["source_moves"], dtype=np.int8)
            for template in row_templates
        ],
        axis=0,
    )
    print_manager_mode = np.asarray(
        [
            str(template["prepared_step"].get("print_manager_mode", "none"))
            for template in row_templates
        ],
        dtype=object,
    )
    timer_advance_ms = np.asarray(
        [
            _number(
                template["prepared_step"].get("timer_advance_ms", 0.0),
                "prepared_step.timer_advance_ms",
            )
            for template in row_templates
        ],
        dtype=np.float64,
    )
    return {
        "initial_state": initial_state,
        "prepared_batch": {
            "player_count": player_count,
            "step_ms": step_ms,
            "source_moves": source_moves,
            "print_manager_mode": print_manager_mode,
            "timer_advance_ms": timer_advance_ms,
        },
        "row_source_counts": dict(
            Counter(str(template["scenario_id"]) for template in row_templates)
        ),
    }


def _append_body_points_batched(
    state: dict[str, np.ndarray],
    *,
    player: int,
    write_mask: np.ndarray,
    insert_kind: int,
) -> tuple[int, int]:
    rows = np.flatnonzero(write_mask)
    if rows.size == 0:
        return 0, 0

    capacity = state["body_active"].shape[1]
    cursor = state["body_write_cursor"][rows].astype(np.int64, copy=False)
    can_insert = cursor < capacity
    overflow_rows = rows[~can_insert]
    if overflow_rows.size:
        state["body_overflow"][overflow_rows] = True
        state["overflow"][overflow_rows] = True

    insert_rows = rows[can_insert]
    insert_cursor = cursor[can_insert]
    if insert_rows.size == 0:
        return 0, int(overflow_rows.size)

    body_num = state["body_count"][insert_rows, player].copy()
    state["body_active"][insert_rows, insert_cursor] = True
    state["body_pos"][insert_rows, insert_cursor] = state["pos"][insert_rows, player]
    state["body_radius"][insert_rows, insert_cursor] = state["radius"][insert_rows, player]
    state["body_owner"][insert_rows, insert_cursor] = player
    state["body_num"][insert_rows, insert_cursor] = body_num
    state["body_insert_tick"][insert_rows, insert_cursor] = state["tick"][insert_rows]
    state["body_insert_kind"][insert_rows, insert_cursor] = insert_kind
    state["body_write_cursor"][insert_rows] += 1
    state["world_body_count"][insert_rows] += 1
    state["body_count"][insert_rows, player] += 1
    state["visible_trail_count"][insert_rows, player] += 1
    state["has_visible_trail_last"][insert_rows, player] = True
    state["visible_trail_last_pos"][insert_rows, player] = state["pos"][insert_rows, player]
    state["has_draw_cursor"][insert_rows, player] = True
    state["draw_cursor_pos"][insert_rows, player] = state["pos"][insert_rows, player]
    return int(insert_rows.size), int(overflow_rows.size)


def _update_print_manager_no_toggle_batched(
    state: dict[str, np.ndarray],
    *,
    player: int,
    live_mask: np.ndarray,
) -> tuple[int, int]:
    active = live_mask & state["print_manager_active"][:, player]
    if not active.any():
        return 0, 0

    dx = state["print_manager_last_pos"][:, player, 0] - state["pos"][:, player, 0]
    dy = state["print_manager_last_pos"][:, player, 1] - state["pos"][:, player, 1]
    next_distance = state["print_manager_distance"][:, player] - np.hypot(dx, dy)
    no_toggle = active & (next_distance > 0.0)
    unhandled_toggle = active & ~no_toggle

    state["print_manager_distance"][no_toggle, player] = next_distance[no_toggle]
    state["print_manager_last_pos"][no_toggle, player] = state["pos"][no_toggle, player]
    return int(no_toggle.sum()), int(unhandled_toggle.sum())


def _update_print_manager_toggle_batched(
    state: dict[str, np.ndarray],
    *,
    player: int,
    live_mask: np.ndarray,
    events_enabled: bool,
) -> tuple[int, int, int]:
    active = live_mask & state["print_manager_active"][:, player]
    if not active.any():
        return 0, 0, 0

    dx = state["print_manager_last_pos"][:, player, 0] - state["pos"][:, player, 0]
    dy = state["print_manager_last_pos"][:, player, 1] - state["pos"][:, player, 1]
    next_distance = state["print_manager_distance"][:, player] - np.hypot(dx, dy)
    toggle = active & (next_distance <= 0.0)
    no_toggle = active & ~toggle
    old_printing = state["printing"][:, player].copy()
    toggled_to_hole = toggle & old_printing

    state["print_manager_distance"][no_toggle, player] = next_distance[no_toggle]
    state["print_manager_last_pos"][active, player] = state["pos"][active, player]

    if toggle.any():
        state["printing"][toggle, player] = ~old_printing[toggle]
        _append_body_points_batched(
            state,
            player=player,
            write_mask=toggle,
            insert_kind=BODY_KIND_IMPORTANT,
        )
        if events_enabled:
            _emit_point_events_batched(state, player, toggle, important=True)
        if toggled_to_hole.any():
            _clear_visible_trail_batched(state, player, toggled_to_hole)
        if events_enabled:
            _emit_printing_property_events_batched(state, player, toggle)
        try:
            vector_runtime.assign_print_manager_random_distances(
                state,
                player=player,
                mask=toggle,
            )
        except vector_runtime.VectorRuntimeError as exc:
            raise vector_compare.VectorCompareError(str(exc)) from exc

    return int(toggle.sum()), int(no_toggle.sum()), int(toggled_to_hole.sum())


def _stop_print_manager_on_death_batched(
    state: dict[str, np.ndarray],
    *,
    player: int,
    death_mask: np.ndarray,
    events_enabled: bool,
) -> tuple[int, int, int]:
    active = death_mask & state["print_manager_active"][:, player]
    if not active.any():
        return 0, 0, 0

    was_printing = state["printing"][:, player].copy()
    important_stop = active & was_printing

    if important_stop.any():
        _append_body_points_batched(
            state,
            player=player,
            write_mask=important_stop,
            insert_kind=BODY_KIND_IMPORTANT,
        )
        if events_enabled:
            _emit_point_events_batched(state, player, important_stop, important=True)

    state["printing"][active, player] = False
    if important_stop.any():
        _clear_visible_trail_batched(state, player, important_stop)
    if events_enabled:
        _emit_printing_property_events_batched(state, player, active)

    state["print_manager_active"][active, player] = False
    state["print_manager_distance"][active, player] = 0.0
    state["print_manager_last_pos"][active, player] = 0.0
    return int(active.sum()), int(important_stop.sum()), int(important_stop.sum())


def _advance_pre_step_timers_batched(
    state: dict[str, np.ndarray],
    prepared_batch: Mapping[str, Any],
    *,
    events_enabled: bool,
) -> dict[str, int]:
    row_count = state["tick"].shape[0]
    timer_advance_ms = np.asarray(
        prepared_batch.get("timer_advance_ms", np.zeros(row_count)),
        dtype=np.float64,
    )
    if timer_advance_ms.shape != (row_count,):
        raise vector_compare.VectorCompareError(
            "prepared_batch.timer_advance_ms must have shape [B]"
        )
    if np.any(timer_advance_ms < 0.0):
        raise vector_compare.VectorCompareError(
            "prepared_batch.timer_advance_ms must be non-negative"
        )

    counters = {
        "pre_step_timer_advances": 0,
        "pre_step_timer_fires": 0,
        "print_manager_delayed_start_fires": 0,
        "print_manager_delayed_start_points": 0,
        "body_overflow_attempts": 0,
    }
    if "timer_active" not in state or not state["timer_active"].any():
        return counters

    active_rows = np.flatnonzero(~state["done"] & ~state["overflow"])
    for row in active_rows:
        row_int = int(row)
        active_slots = np.flatnonzero(state["timer_active"][row_int])
        if active_slots.size == 0:
            continue

        counters["pre_step_timer_advances"] += 1
        state["timer_remaining_ms"][row_int, active_slots] -= timer_advance_ms[row_int]
        due_slots = [
            int(slot)
            for slot in active_slots
            if float(state["timer_remaining_ms"][row_int, int(slot)]) <= 0.0
        ]
        due_slots.sort(key=lambda slot: int(state["timer_seq"][row_int, slot]))

        for slot in due_slots:
            kind = int(state["timer_kind"][row_int, slot])
            if kind == vector_compare.TIMER_KIND_PRINT_MANAGER_START:
                player = int(state["timer_player"][row_int, slot])
                fired, points, overflowed = _fire_print_manager_start_timer_batched(
                    state,
                    row=row_int,
                    player=player,
                    events_enabled=events_enabled,
                )
                counters["pre_step_timer_fires"] += fired
                counters["print_manager_delayed_start_fires"] += fired
                counters["print_manager_delayed_start_points"] += points
                counters["body_overflow_attempts"] += overflowed
            vector_compare._clear_timer_slot(state, row=row_int, slot=slot)

    return counters


def _fire_print_manager_start_timer_batched(
    state: dict[str, np.ndarray],
    *,
    row: int,
    player: int,
    events_enabled: bool,
) -> tuple[int, int, int]:
    if player < 0 or player >= state["print_manager_active"].shape[1]:
        raise vector_compare.VectorCompareError(f"timer references unknown player index {player}")
    if bool(state["print_manager_active"][row, player]):
        return 0, 0, 0

    state["print_manager_active"][row, player] = True
    state["print_manager_last_pos"][row, player] = state["pos"][row, player]

    row_mask = np.zeros(state["tick"].shape[0], dtype=bool)
    row_mask[row] = True
    points = 0
    overflowed = 0
    if not bool(state["printing"][row, player]):
        state["printing"][row, player] = True
        points, overflowed = _append_body_points_batched(
            state,
            player=player,
            write_mask=row_mask,
            insert_kind=BODY_KIND_IMPORTANT,
        )
        if events_enabled:
            _emit_point_events_batched(state, player, row_mask, important=True)

    if events_enabled:
        _emit_printing_property_events_batched(state, player, row_mask)
    try:
        vector_runtime.assign_print_manager_random_distances(
            state,
            player=player,
            mask=row_mask,
        )
    except vector_runtime.VectorRuntimeError as exc:
        raise vector_compare.VectorCompareError(str(exc)) from exc
    return 1, points, overflowed


def _clear_visible_trail_batched(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
) -> None:
    state["visible_trail_count"][mask, player] = 0
    state["has_visible_trail_last"][mask, player] = False
    state["visible_trail_last_pos"][mask, player] = 0.0
    state["has_draw_cursor"][mask, player] = False
    state["draw_cursor_pos"][mask, player] = 0.0


def _apply_terminal_score_for_rows_batched(
    state: dict[str, np.ndarray],
    death_rows: np.ndarray,
    *,
    player_count: int,
    phase_timers: dict[str, float] | None = None,
    event_mode: str = EVENT_MODE_DEBUG,
) -> int:
    events_enabled = _events_enabled(event_mode)
    alive_counts = state["alive"][:, :player_count].sum(axis=1)
    terminal_rows = death_rows & (alive_counts <= 1)
    rows = np.flatnonzero(terminal_rows)
    if rows.size == 0:
        return 0

    for row in rows:
        row_int = int(row)
        started = _timer_start(phase_timers)
        live_players = np.flatnonzero(state["alive"][row_int, :player_count])
        winner_value = -1
        if live_players.size == 1:
            winner_value = int(live_players[0])
            state["round_score"][row_int, winner_value] += max(player_count - 1, 1)
        _timer_add(phase_timers, "terminal_score_state_sec", started)

        if events_enabled and winner_value >= 0:
            started = _timer_start(phase_timers)
            vector_compare._emit_score_row(
                state,
                row=row_int,
                player=winner_value,
                event_type=vector_compare.EVENT_SCORE_ROUND,
            )
            _timer_add(phase_timers, "event_emit_sec", started)

        started = _timer_start(phase_timers)
        state["score"][row_int, :player_count] += state["round_score"][row_int, :player_count]
        _timer_add(phase_timers, "terminal_score_state_sec", started)

        if events_enabled:
            started = _timer_start(phase_timers)
            for player in range(player_count - 1, -1, -1):
                vector_compare._emit_score_row(
                    state,
                    row=row_int,
                    player=player,
                    event_type=vector_compare.EVENT_SCORE,
                )
            _timer_add(phase_timers, "event_emit_sec", started)

        started = _timer_start(phase_timers)
        state["round_score"][row_int, :player_count] = 0
        _timer_add(phase_timers, "terminal_score_state_sec", started)

        if events_enabled:
            started = _timer_start(phase_timers)
            vector_compare._emit_round_end_row(state, row=row_int, winner=winner_value)
            _timer_add(phase_timers, "event_emit_sec", started)
    return int(rows.size)


def _emit_position_events_batched(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=vector_compare.EVENT_POSITION,
        player=player,
        float_values=state["pos"][rows, player],
    )


def _emit_point_events_batched(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
    *,
    important: bool,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=vector_compare.EVENT_POINT,
        player=player,
        bool_value=1 if important else 0,
        float_values=state["pos"][rows, player],
    )


def _emit_printing_property_events_batched(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=vector_compare.EVENT_PROPERTY,
        player=player,
        bool_value=state["printing"][rows, player].astype(np.int8),
        int_values=np.tile(
            np.asarray([vector_compare.PROPERTY_PRINTING, 0], dtype=np.int32),
            (rows.size, 1),
        ),
    )


def _emit_die_events_batched(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
    *,
    other_player: np.ndarray | None = None,
    old: bool | None = None,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=vector_compare.EVENT_DIE,
        player=player,
        other_values=other_player,
        bool_value=-1 if old is None else 1 if old else 0,
    )


def _emit_score_events_batched(
    state: dict[str, np.ndarray],
    player: int,
    mask: np.ndarray,
    *,
    event_type: int,
) -> None:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return
    _emit_event_rows_batched(
        state,
        rows,
        event_type=event_type,
        player=player,
        int_values=np.stack(
            (state["score"][rows, player], state["round_score"][rows, player]),
            axis=1,
        ),
    )


def _emit_event_rows_batched(
    state: dict[str, np.ndarray],
    rows: np.ndarray,
    *,
    event_type: int,
    player: int = -1,
    other_values: np.ndarray | None = None,
    bool_value: int | np.ndarray = -1,
    int_values: np.ndarray | None = None,
    float_values: np.ndarray | None = None,
) -> None:
    cursors = state["event_count"][rows].astype(np.int64, copy=False)
    capacity = state["event_type"].shape[1]
    can_emit = cursors < capacity

    overflow_rows = rows[~can_emit]
    if overflow_rows.size:
        state["event_overflow"][overflow_rows] = True
        state["event_overflow_attempts"][overflow_rows] += 1

    emit_rows = rows[can_emit]
    if emit_rows.size == 0:
        return

    emit_cursors = cursors[can_emit]
    state["event_count"][emit_rows] += 1
    state["event_mask"][emit_rows, emit_cursors] = True
    state["event_type"][emit_rows, emit_cursors] = event_type
    state["event_player"][emit_rows, emit_cursors] = player
    state["event_other"][emit_rows, emit_cursors] = -1
    if other_values is not None:
        state["event_other"][emit_rows, emit_cursors] = other_values[emit_rows]
    if isinstance(bool_value, np.ndarray):
        state["event_bool"][emit_rows, emit_cursors] = bool_value[can_emit]
    else:
        state["event_bool"][emit_rows, emit_cursors] = bool_value
    if int_values is not None:
        state["event_value_i"][emit_rows, emit_cursors] = int_values[can_emit]
    if float_values is not None:
        state["event_value_f"][emit_rows, emit_cursors] = float_values[can_emit]


def _body_collision_rows(
    state: Mapping[str, np.ndarray],
    player: int,
    live_mask: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    capacity = state["body_active"].shape[1]
    if capacity == 0:
        return np.asarray([], dtype=np.int64), 0, 0

    radius = state["radius"][:, player][:, None]
    dx = state["body_pos"][:, :, 0] - state["pos"][:, player, 0][:, None]
    dy = state["body_pos"][:, :, 1] - state["pos"][:, player, 1][:, None]
    dist_sq = dx * dx + dy * dy
    hit_radius_sq = (radius + state["body_radius"]) ** 2
    own_body = state["body_owner"] == player
    own_delta = state["live_body_num"][:, player][:, None] - state["body_num"]
    own_too_young = own_body & (own_delta <= state["trail_latency"][:, player][:, None])
    candidate = state["body_active"] & ~own_too_young
    hit_mask = candidate & (dist_sq < hit_radius_sq)
    hit_rows = np.flatnonzero(live_mask & hit_mask.any(axis=1))
    scanned_slots = int(live_mask.sum()) * capacity
    return hit_rows, int(candidate.sum()), scanned_slots


def _row_state_mismatches(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    row_index: int,
    scenario_id: str,
    max_mismatches: int,
) -> list[dict[str, Any]]:
    if max_mismatches <= 0:
        return []
    mismatches: list[dict[str, Any]] = []
    for name, expected_array in expected.items():
        if name in EVENT_ARRAY_NAMES:
            continue
        actual_array = actual[name][row_index : row_index + 1]
        if np.issubdtype(expected_array.dtype, np.floating):
            match = np.allclose(expected_array, actual_array, rtol=0.0, atol=1e-12)
        else:
            match = np.array_equal(expected_array, actual_array)
        if not match:
            detail: dict[str, Any] = {
                "scenario_id": scenario_id,
                "row_index": row_index,
                "array": name,
                "expected_shape": list(expected_array.shape),
                "actual_shape": list(actual_array.shape),
            }
            if np.issubdtype(expected_array.dtype, np.floating):
                detail["max_abs_diff"] = float(np.max(np.abs(expected_array - actual_array)))
            mismatches.append(detail)
            if len(mismatches) >= max_mismatches:
                break
    return mismatches


def _row_event_mismatches(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
    *,
    row_index: int,
    scenario_id: str,
    max_mismatches: int,
) -> list[dict[str, Any]]:
    if max_mismatches <= 0:
        return []
    mismatches: list[dict[str, Any]] = []
    for name in sorted(EVENT_ARRAY_NAMES):
        expected_array = expected[name]
        actual_array = actual[name][row_index : row_index + 1]
        if np.issubdtype(expected_array.dtype, np.floating):
            match = np.allclose(expected_array, actual_array, rtol=0.0, atol=1e-12)
        else:
            match = np.array_equal(expected_array, actual_array)
        if not match:
            detail: dict[str, Any] = {
                "scenario_id": scenario_id,
                "row_index": row_index,
                "array": name,
                "expected_shape": list(expected_array.shape),
                "actual_shape": list(actual_array.shape),
            }
            if np.issubdtype(expected_array.dtype, np.floating):
                detail["max_abs_diff"] = float(np.max(np.abs(expected_array - actual_array)))
            mismatches.append(detail)
            if len(mismatches) >= max_mismatches:
                break
    return mismatches


def _group_key(profile: Mapping[str, Any], state: Mapping[str, np.ndarray]) -> str:
    player_count = _int(profile.get("P"), "fixture.profile.P")
    body_capacity = _int(profile.get("K"), "fixture.profile.K")
    # Production profiles should be explicit fixed shapes. For the current
    # supported fixtures, P and K fully split the compatible array profiles.
    return f"P{player_count}_K{body_capacity}"


def _empty_phase_timers() -> dict[str, float]:
    return {
        "event_reset_sec": 0.0,
        "event_emit_sec": 0.0,
        "event_body_hit_owner_sec": 0.0,
        "movement_sec": 0.0,
        "normal_point_mask_sec": 0.0,
        "normal_point_append_sec": 0.0,
        "border_wrap_sec": 0.0,
        "wall_check_sec": 0.0,
        "wall_death_apply_sec": 0.0,
        "body_collision_sec": 0.0,
        "body_death_apply_sec": 0.0,
        "pre_step_timer_sec": 0.0,
        "print_manager_update_sec": 0.0,
        "print_manager_death_stop_sec": 0.0,
        "terminal_score_state_sec": 0.0,
        "tick_sec": 0.0,
    }


def _timer_start(phase_timers: dict[str, float] | None) -> float:
    if phase_timers is None:
        return 0.0
    return time.perf_counter()


def _timer_add(
    phase_timers: dict[str, float] | None,
    key: str,
    started: float,
) -> None:
    if phase_timers is None:
        return
    phase_timers[key] += time.perf_counter() - started


def _top_phase_label(phase_timers: Mapping[str, Any]) -> str:
    if not phase_timers:
        return "none"
    key, value = max(phase_timers.items(), key=lambda item: float(item[1]))
    return f"{key}:{float(value):.6f}s"


def _event_phase_sec(phase_timers: Mapping[str, Any]) -> float:
    return sum(float(value) for key, value in phase_timers.items() if key.startswith("event_"))


def _normalize_event_modes(event_modes: Sequence[str]) -> tuple[str, ...]:
    modes = tuple(_normalize_event_mode(mode) for mode in event_modes)
    if not modes:
        raise vector_compare.VectorCompareError("event_modes must contain at least one mode")
    duplicates = [mode for mode, count in Counter(modes).items() if count > 1]
    if duplicates:
        raise vector_compare.VectorCompareError(
            f"event_modes contains duplicate mode(s): {', '.join(sorted(duplicates))}"
        )
    return modes


def _normalize_event_mode(event_mode: str) -> str:
    if event_mode not in EVENT_MODE_CHOICES:
        choices = ", ".join(EVENT_MODE_CHOICES)
        raise vector_compare.VectorCompareError(
            f"event_mode must be one of {choices}; got {event_mode!r}"
        )
    return event_mode


def _events_enabled(event_mode: str) -> bool:
    return _normalize_event_mode(event_mode) == EVENT_MODE_DEBUG


def _rows_to_mask(rows: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[rows] = True
    return mask


def _add_counters(total: dict[str, int], increment: Mapping[str, int]) -> None:
    for key, value in increment.items():
        total[key] = total.get(key, 0) + int(value)


def _rate(numerator: int, denominator_sec: float) -> float:
    if denominator_sec <= 0:
        return 0.0
    return numerator / denominator_sec


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise vector_compare.VectorCompareError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise vector_compare.VectorCompareError(f"{field} must be a list")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise vector_compare.VectorCompareError(f"{field} must be a non-empty string")
    return value


def _int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise vector_compare.VectorCompareError(f"{field} must be an integer")
    return value


def _number(value: Any, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise vector_compare.VectorCompareError(f"{field} must be a number")
    return float(value)


def _numpy_dtype(name: str) -> Any:
    dtypes = {
        "bool": bool,
        "float64": np.float64,
        "float32": np.float32,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
    }
    if name not in dtypes:
        raise vector_compare.VectorCompareError(f"unsupported array dtype {name!r}")
    return dtypes[name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark B>1 fixture-seeded NumPy rows in one vectorized step call."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="Scenario JSON files or batch manifests. Defaults to the current supported set.",
    )
    parser.add_argument(
        "--body-capacity",
        type=_nonnegative_int,
        default=4,
        help="Fixed K for the seeded body buffer. Defaults to the narrow K=4 profile.",
    )
    parser.add_argument("--step-index", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--batch-sizes", type=_positive_int, nargs="+", default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--event-modes",
        choices=EVENT_MODE_CHOICES,
        nargs="+",
        default=list(DEFAULT_EVENT_MODES),
        help=(
            "Event logging modes to time. Use 'debug-event no-event' to isolate "
            "debug event row cost. Defaults to debug-event."
        ),
    )
    parser.add_argument("--repeat", type=_positive_int, default=1_000)
    parser.add_argument("--warmup", type=_nonnegative_int, default=100)
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow fixtures that are not marked python-runner-verified.",
    )
    parser.add_argument(
        "--fail-on-unsupported",
        action="store_true",
        help="Exit nonzero when any fixture is unsupported.",
    )
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    args = parser.parse_args()

    summary = benchmark_inputs(
        args.paths,
        body_capacity=args.body_capacity,
        step_index=args.step_index,
        batch_sizes=args.batch_sizes,
        event_modes=args.event_modes,
        repeat=args.repeat,
        warmup=args.warmup,
        require_verified=not args.allow_unverified,
    )
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)

    counts = summary["summary"]
    if (
        counts["failed"]
        or counts["batch_preflight_failed"]
        or (args.fail_on_unsupported and counts["unsupported"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
