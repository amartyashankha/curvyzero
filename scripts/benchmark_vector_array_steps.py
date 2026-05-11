"""Time repeated supported fixture-seeded array transitions.

This is the speed companion to ``compare_vector_arrays_to_fidelity.py``. It
uses the same narrow supported fixture set, runs source/common-trace comparison
once as a preflight check, and keeps source runner, projection, and comparison
work out of the repeated env-step timing loop.
"""

from __future__ import annotations

import argparse
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


SCHEMA_VERSION = "curvyzero_vector_fixture_step_timing/v1"
BENCHMARK_ID = "fixture_seeded_numpy_supported_transition_repeated"
DEFAULT_PATHS = (
    "scenarios/environment/source_body_canary_batch.json",
    "scenarios/environment/source_borderless_wrap_step.json",
    "scenarios/environment/source_normal_wall_death_step.json",
)
NEXT_VECTOR_BLOCKER = (
    "compact fixed event arrays for point/die/score rows, then PrintManager/trail-gap "
    "state, before claiming production B>1 self-play batching"
)


def benchmark_inputs(
    paths: Sequence[str | Path],
    *,
    body_capacity: int = 4,
    step_index: int = 0,
    repeat: int = 10_000,
    warmup: int = 500,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Benchmark repeated supported one-step transitions from fixture seeds."""

    if body_capacity < 0:
        raise vector_compare.VectorCompareError("body_capacity must be zero or greater")
    if step_index < 0:
        raise vector_compare.VectorCompareError("step_index must be zero or greater")
    if repeat <= 0:
        raise vector_compare.VectorCompareError("repeat must be greater than zero")
    if warmup < 0:
        raise vector_compare.VectorCompareError("warmup must be zero or greater")

    wall_started = time.perf_counter()
    timers = {
        "seed_inputs_sec": 0.0,
        "array_prepare_sec": 0.0,
        "preflight_state_copy_sec": 0.0,
        "preflight_source_trace_sec": 0.0,
        "preflight_env_step_sec": 0.0,
        "preflight_projection_sec": 0.0,
        "preflight_comparison_sec": 0.0,
        "warmup_reset_copy_sec": 0.0,
        "warmup_env_step_sec": 0.0,
        "timed_reset_copy_sec": 0.0,
        "timed_env_step_sec": 0.0,
    }

    started = time.perf_counter()
    seeded = seed_bridge.seed_inputs(paths, body_capacity=body_capacity)
    timers["seed_inputs_sec"] = time.perf_counter() - started

    prepared_items = []
    fixture_results = []
    for fixture in seeded["fixtures"]:
        support = vector_compare.fixture_transition_support(
            fixture,
            step_index=step_index,
            require_verified=require_verified,
        )
        scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")
        if not support["supported"]:
            fixture_results.append(
                {
                    "scenario_id": scenario_id,
                    "path": fixture.get("path"),
                    "status": "unsupported",
                    "match": None,
                    "compared_fields": 0,
                    "skipped_fields": 0,
                    "unsupported_mechanics": support["unsupported_mechanics"],
                    "covered_mechanics": support["covered_mechanics"],
                }
            )
            continue

        started = time.perf_counter()
        initial_state = vector_compare.array_state_from_seed(fixture)
        working_state = vector_compare.copy_array_state(initial_state)
        prepared_step = vector_compare.prepare_fixture_array_step(
            fixture,
            step_index=step_index,
        )
        timers["array_prepare_sec"] += time.perf_counter() - started
        prepared_items.append(
            {
                "fixture": fixture,
                "support": support,
                "initial_state": initial_state,
                "working_state": working_state,
                "prepared_step": prepared_step,
            }
        )

    for item in prepared_items:
        fixture = item["fixture"]
        scenario_id = _string(fixture.get("scenario_id"), "fixture.scenario_id")

        started = time.perf_counter()
        source_trace = vector_compare.source_common_trace_for_fixture(fixture)
        timers["preflight_source_trace_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        preflight_state = vector_compare.copy_array_state(item["initial_state"])
        timers["preflight_state_copy_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        counters = vector_compare.step_prepared_arrays(
            preflight_state,
            item["prepared_step"],
        )
        timers["preflight_env_step_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        actual_trace = vector_compare.project_array_state_to_common_trace(
            fixture,
            preflight_state,
            step_index=step_index,
        )
        timers["preflight_projection_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        comparison = vector_compare.compare_common_trace_fields(
            actual_trace,
            source_trace,
            fixture.get("comparison", {}),
        )
        timers["preflight_comparison_sec"] += time.perf_counter() - started

        fixture_results.append(
            {
                "scenario_id": scenario_id,
                "path": fixture.get("path"),
                "status": "pass" if comparison["match"] else "fail",
                "match": comparison["match"],
                "compared_fields": len(comparison["compared_fields"]),
                "skipped_fields": len(comparison["skipped_fields"]),
                "mismatches": comparison["mismatches"],
                "unsupported_mechanics": item["support"]["unsupported_mechanics"],
                "covered_mechanics": item["support"]["covered_mechanics"],
                "preflight_array_counters": counters,
            }
        )

    passed = sum(result["status"] == "pass" for result in fixture_results)
    failed = sum(result["status"] == "fail" for result in fixture_results)
    unsupported = sum(result["status"] == "unsupported" for result in fixture_results)

    if prepared_items and failed == 0:
        warmup_timers, warmup_counters = _run_repeated_steps(prepared_items, warmup)
        timers["warmup_reset_copy_sec"] = warmup_timers["reset_copy_sec"]
        timers["warmup_env_step_sec"] = warmup_timers["env_step_sec"]

        timed_timers, timed_counters = _run_repeated_steps(prepared_items, repeat)
        timers["timed_reset_copy_sec"] = timed_timers["reset_copy_sec"]
        timers["timed_env_step_sec"] = timed_timers["env_step_sec"]
    else:
        warmup_counters = {}
        timed_counters = {}

    measured_env_steps = len(prepared_items) * repeat if failed == 0 else 0
    warmup_env_steps = len(prepared_items) * warmup if failed == 0 else 0
    timed_step_sec = timers["timed_env_step_sec"]
    timed_loop_sec = timers["timed_env_step_sec"] + timers["timed_reset_copy_sec"]
    setup_sec = timers["seed_inputs_sec"] + timers["array_prepare_sec"]
    wall_elapsed_sec = time.perf_counter() - wall_started

    return {
        "schema": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "source_fidelity_claim": (
            "source/common-trace comparison is run once as preflight for the same "
            "narrow fixture set; repeated timing covers only supported NumPy "
            "array transitions plus a separately reported reset-copy bucket"
        ),
        "trust_level": (
            "This is a fixture-seeded B=1 microbenchmark. It is not a production "
            "backend, not stacked B>1 batching, and not a trainer throughput claim."
        ),
        "config": {
            "paths": [str(path) for path in paths],
            "body_capacity": body_capacity,
            "step_index": step_index,
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
        "supported_fixture_count": len(prepared_items),
        "summary": {
            "passed": passed,
            "failed": failed,
            "unsupported": unsupported,
            "status": "fail" if failed else "pass" if passed and not unsupported else "mixed",
        },
        "timing_sec": {
            **timers,
            "setup_sec": setup_sec,
            "preflight_total_sec": (
                timers["preflight_state_copy_sec"]
                + timers["preflight_source_trace_sec"]
                + timers["preflight_env_step_sec"]
                + timers["preflight_projection_sec"]
                + timers["preflight_comparison_sec"]
            ),
            "timed_loop_sec": timed_loop_sec,
            "wall_elapsed_sec": wall_elapsed_sec,
        },
        "rates": {
            "warmup_env_steps": warmup_env_steps,
            "timed_env_steps": measured_env_steps,
            "timed_env_steps_per_step_sec": _rate(measured_env_steps, timed_step_sec),
            "timed_env_steps_per_reset_plus_step_sec": _rate(measured_env_steps, timed_loop_sec),
        },
        "hot_loop_exclusions": {
            "source_trace_calls": 0,
            "projection_calls": 0,
            "comparison_calls": 0,
            "note": "hot loop performs reset-copy plus step only; reset-copy is timed separately",
        },
        "warmup_counters": warmup_counters,
        "timed_counters": timed_counters,
        "fixtures": fixture_results,
        "global_unsupported_mechanics": vector_compare.GLOBAL_UNSUPPORTED_MECHANICS,
        "next_vector_blocker": NEXT_VECTOR_BLOCKER,
    }


def print_plain(summary: Mapping[str, Any]) -> None:
    counts = _mapping(summary.get("summary"), "summary")
    timing = _mapping(summary.get("timing_sec"), "timing_sec")
    rates = _mapping(summary.get("rates"), "rates")
    exclusions = _mapping(summary.get("hot_loop_exclusions"), "hot_loop_exclusions")

    print(f"benchmark={summary['benchmark_id']}")
    print(f"trust_level={summary['trust_level']}")
    print(
        "summary="
        f"passed:{counts['passed']} failed:{counts['failed']} unsupported:{counts['unsupported']}"
    )
    print(
        "timing_sec="
        f"setup:{timing['setup_sec']:.6f} "
        f"preflight_source_trace:{timing['preflight_source_trace_sec']:.6f} "
        f"preflight_env_step:{timing['preflight_env_step_sec']:.6f} "
        f"preflight_projection:{timing['preflight_projection_sec']:.6f} "
        f"preflight_comparison:{timing['preflight_comparison_sec']:.6f} "
        f"timed_reset_copy:{timing['timed_reset_copy_sec']:.6f} "
        f"timed_env_step:{timing['timed_env_step_sec']:.6f}"
    )
    print(
        "rates="
        f"timed_env_steps:{rates['timed_env_steps']} "
        f"env_steps_per_step_sec:{rates['timed_env_steps_per_step_sec']:.1f} "
        "env_steps_per_reset_plus_step_sec:"
        f"{rates['timed_env_steps_per_reset_plus_step_sec']:.1f}"
    )
    print(
        "hot_loop_exclusions="
        f"source_trace_calls:{exclusions['source_trace_calls']} "
        f"projection_calls:{exclusions['projection_calls']} "
        f"comparison_calls:{exclusions['comparison_calls']}"
    )
    for result in _list(summary.get("fixtures"), "fixtures"):
        fixture = _mapping(result, "fixture")
        print(
            "fixture="
            f"{fixture['scenario_id']} status={fixture['status']} "
            f"compared_fields={fixture['compared_fields']} "
            f"skipped_fields={fixture['skipped_fields']}"
        )
        if fixture["status"] == "fail":
            first = _mapping(_list(fixture["mismatches"], "mismatches")[0], "first_mismatch")
            print(
                "first_mismatch="
                f"{first['path']} expected={first.get('expected')} "
                f"actual={first.get('actual')} reason={first.get('reason')}"
            )
        if fixture["status"] == "unsupported":
            mechanics = _list(fixture["unsupported_mechanics"], "unsupported_mechanics")
            reason = _mapping(mechanics[0], "unsupported_mechanics[0]") if mechanics else {}
            print(f"unsupported_reason={reason.get('reason', 'not supported by this gate')}")
    print(f"next_vector_blocker={summary['next_vector_blocker']}")


def _run_repeated_steps(
    prepared_items: Sequence[Mapping[str, Any]],
    repeat: int,
) -> tuple[dict[str, float], dict[str, int]]:
    timers = {"reset_copy_sec": 0.0, "env_step_sec": 0.0}
    counters: dict[str, int] = {}
    if repeat == 0:
        return timers, counters

    for _ in range(repeat):
        for item in prepared_items:
            started = time.perf_counter()
            vector_compare.reset_array_state(item["working_state"], item["initial_state"])
            timers["reset_copy_sec"] += time.perf_counter() - started

            started = time.perf_counter()
            step_counters = vector_compare.step_prepared_arrays(
                item["working_state"],
                item["prepared_step"],
            )
            timers["env_step_sec"] += time.perf_counter() - started
            _add_counters(counters, step_counters)

    return timers, counters


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark repeated supported fixture-seeded NumPy array steps."
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
        help="Fixed K for the seeded body buffer. Defaults to the narrow K=4 timing profile.",
    )
    parser.add_argument("--step-index", type=_nonnegative_int, default=0)
    parser.add_argument("--repeat", type=_positive_int, default=10_000)
    parser.add_argument("--warmup", type=_nonnegative_int, default=500)
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
        repeat=args.repeat,
        warmup=args.warmup,
        require_verified=not args.allow_unverified,
    )
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)

    counts = summary["summary"]
    if counts["failed"] or (args.fail_on_unsupported and counts["unsupported"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
