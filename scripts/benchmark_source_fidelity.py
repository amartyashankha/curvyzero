"""Smoke benchmark for the existing source-fidelity scenario runner surface.

This is intentionally narrow. It times read-only calls into the current
source-fidelity scenario runners and the common-trace projection. It does not
instrument source internals, so movement, point insertion, collision scan, and
PrintManager buckets are reported as scaffolded, not measured.

Pass --profile to attach a cProfile/pstats summary for the measured runner
surface loop. That profile excludes warmup and manifest collection.
"""

from __future__ import annotations

import argparse
import cProfile
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import pstats
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.scenario_schema import LoadedScenario, load_scenario  # noqa: E402
from curvyzero.env.trace_compare import project_common_trace  # noqa: E402
from curvyzero.fidelity.source_runners import (  # noqa: E402
    SOURCE_BODY_CANARY_SCENARIO_IDS,
    SOURCE_PRINT_MANAGER_SCENARIO_IDS,
    SOURCE_TRAIL_CADENCE_SCENARIO_IDS,
    SOURCE_TRAIL_GAP_SCENARIO_IDS,
    run_source_body_canary_scenario,
    run_source_borderless_wrap_scenario,
    run_source_kinematics_scenario,
    run_source_normal_wall_scenario,
    run_source_print_manager_scenario,
    run_source_trail_cadence_scenario,
    run_source_trail_gap_scenario,
)


DEFAULT_MANIFESTS = (
    "scenarios/environment/source_kinematics_batch.json",
    "scenarios/environment/source_body_canary_batch.json",
    "scenarios/environment/source_trail_cadence_batch.json",
    "scenarios/environment/source_trail_gap_batch.json",
    "scenarios/environment/source_print_manager_batch.json",
)

TIMER_DEFINITIONS = {
    "scenario_load": "Scenario JSON load and shared schema parsing.",
    "runner_call": "Inclusive source-fidelity runner call; internal mechanics are not split.",
    "wrapper_payload": "Runner result to_payload conversion.",
    "common_trace_projection": "project_common_trace(payload).",
    "json_encode": "JSON serialization of the source runner payload.",
    "loop_overhead": "Elapsed benchmark time outside measured timers.",
}

REQUESTED_BUCKET_STATUS = {
    "movement": {
        "status": "not_measured",
        "covered_by": "runner_call",
        "reason": "Movement is internal to source-fidelity trace generation.",
    },
    "point_insertion": {
        "status": "not_measured",
        "covered_by": "runner_call",
        "reason": "Point/body insertion is interleaved with source player-order mechanics.",
    },
    "collision_scan": {
        "status": "not_measured",
        "covered_by": "runner_call",
        "reason": "Body scans and wall checks are not exposed as separate spans.",
    },
    "print_manager": {
        "status": "not_measured",
        "covered_by": "runner_call",
        "reason": "PrintManager update/toggle work is nested inside runner paths.",
    },
    "common_trace_projection": {
        "status": "measured",
        "measured_as": "common_trace_projection",
    },
    "wrapper_overhead": {
        "status": "partially_measured",
        "measured_as": ["wrapper_payload", "json_encode"],
        "reason": "Measures source payload wrapping, not trainer-facing env wrappers.",
    },
}


def _git_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, check=False, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_manifest() -> dict[str, Any]:
    status = _git_text(["git", "status", "--short"])
    status_lines = status.splitlines() if status else []
    return {
        "revision": _git_text(["git", "rev-parse", "--short", "HEAD"]) or "unavailable",
        "working_tree_status": "changes-present" if status_lines else "clean-or-unavailable",
        "status_entry_count": len(status_lines),
    }


def _runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def _manifest_scenarios(manifest: Path) -> list[Path]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
        entries = payload["scenarios"]
    else:
        raise ValueError(f"{manifest} must be a scenario list or batch object")

    paths = []
    for index, entry in enumerate(entries):
        value = entry if isinstance(entry, str) else entry.get("path", entry.get("scenario_path"))
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{manifest} scenario entry {index} has no path")
        path = Path(value).expanduser()
        if path.is_absolute():
            paths.append(path.resolve())
            continue
        manifest_path = (manifest.parent / path).resolve()
        paths.append(manifest_path if manifest_path.exists() else (REPO_ROOT / path).resolve())
    return paths


def _runner_for(scenario: LoadedScenario) -> tuple[str, Any, tuple[str, ...]]:
    scenario_id = scenario.scenario_id
    if scenario_id == "forced_two_player_turn_step" or scenario_id.startswith(
        "source_kinematics_"
    ):
        return "source-kinematics", run_source_kinematics_scenario, ("movement",)
    if scenario_id.startswith("source_normal_wall_"):
        return "source-normal-wall", run_source_normal_wall_scenario, ("movement",)
    if scenario_id.startswith("source_borderless_"):
        return "source-borderless-wrap", run_source_borderless_wrap_scenario, ("movement",)
    if scenario_id in SOURCE_BODY_CANARY_SCENARIO_IDS:
        return (
            "source-body-canary",
            run_source_body_canary_scenario,
            ("movement", "point_insertion", "collision_scan"),
        )
    if scenario_id in SOURCE_TRAIL_CADENCE_SCENARIO_IDS:
        return (
            "source-trail-cadence",
            run_source_trail_cadence_scenario,
            ("movement", "point_insertion"),
        )
    if scenario_id in SOURCE_TRAIL_GAP_SCENARIO_IDS:
        return (
            "source-trail-gap",
            run_source_trail_gap_scenario,
            ("movement", "point_insertion", "collision_scan", "print_manager"),
        )
    if scenario_id in SOURCE_PRINT_MANAGER_SCENARIO_IDS:
        return (
            "source-print-manager",
            run_source_print_manager_scenario,
            ("movement", "point_insertion", "print_manager"),
        )
    raise ValueError(f"no source-fidelity runner for scenario {scenario_id!r}")


def _timed_call(fn: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - started


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _run_one(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    scenario, load_sec = _timed_call(lambda: load_scenario(path))
    runner_name, runner, feature_buckets = _runner_for(scenario)
    run_result, runner_sec = _timed_call(lambda: runner(scenario))
    payload, wrapper_sec = _timed_call(run_result.to_payload)
    common_trace, projection_sec = _timed_call(lambda: project_common_trace(payload))
    encoded, encode_sec = _timed_call(
        lambda: json.dumps(payload, separators=(",", ":"), sort_keys=True)
    )
    steps = common_trace.get("steps", [])
    return (
        {
            "scenario_id": scenario.scenario_id,
            "scenario_path": _display(path),
            "runner": runner_name,
            "feature_buckets": list(feature_buckets),
            "common_trace_step_count": len(steps) if isinstance(steps, list) else None,
            "json_payload_bytes": len(encoded.encode("utf-8")),
        },
        {
            "scenario_load": load_sec,
            "runner_call": runner_sec,
            "wrapper_payload": wrapper_sec,
            "common_trace_projection": projection_sec,
            "json_encode": encode_sec,
        },
    )


def run(
    manifests: list[str | Path],
    repeat: int,
    warmup: int,
    *,
    profiler: cProfile.Profile | None = None,
) -> dict[str, Any]:
    if repeat <= 0:
        raise ValueError("--repeat must be greater than zero")
    if warmup < 0:
        raise ValueError("--warmup must be zero or greater")

    manifest_paths = [_resolve(path) for path in manifests]
    scenario_paths = [
        scenario_path
        for manifest in manifest_paths
        for scenario_path in _manifest_scenarios(manifest)
    ]
    if not scenario_paths:
        raise ValueError("no scenarios selected")

    for _ in range(warmup):
        for scenario_path in scenario_paths:
            _run_one(scenario_path)

    timings: dict[str, float] = defaultdict(float)
    scenarios: dict[str, dict[str, Any]] = {}
    if profiler is not None:
        profiler.enable()
    try:
        started = time.perf_counter()
        for _ in range(repeat):
            for scenario_path in scenario_paths:
                metadata, one_timings = _run_one(scenario_path)
                scenarios.setdefault(metadata["scenario_id"], metadata)
                for bucket, seconds in one_timings.items():
                    timings[bucket] += seconds
        elapsed = time.perf_counter() - started
    finally:
        if profiler is not None:
            profiler.disable()
    measured = sum(timings.values())
    timings["loop_overhead"] = max(0.0, elapsed - measured)
    iterations = repeat * len(scenario_paths)

    return {
        "schema_version": "curvyzero_source_fidelity_benchmark/v1",
        "benchmark_id": "source_fidelity_runner_surface_timing",
        "source_fidelity_claim": (
            "Real CurvyTron source-fidelity runner calls for selected fixtures; "
            "internal mechanics buckets are scaffolded only."
        ),
        "command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "workload": {
            "manifest_paths": [_display(path) for path in manifest_paths],
            "scenario_count": len(scenario_paths),
            "repeat": repeat,
            "warmup": warmup,
            "scenario_iterations": iterations,
        },
        "source": _git_manifest(),
        "runtime": _runtime_manifest(),
        "timer_schema": {
            "definitions": TIMER_DEFINITIONS,
            "requested_bucket_status": REQUESTED_BUCKET_STATUS,
            "caveat": (
                "Do not infer movement, point insertion, collision scan, or "
                "PrintManager split timings from inclusive runner_call."
            ),
        },
        "elapsed_sec": elapsed,
        "scenario_iterations_per_sec": iterations / elapsed if elapsed else 0.0,
        "timings_sec": dict(timings),
        "timing_counts": {
            "scenario_load": iterations,
            "runner_call": iterations,
            "wrapper_payload": iterations,
            "common_trace_projection": iterations,
            "json_encode": iterations,
        },
        "requested_bucket_status": REQUESTED_BUCKET_STATUS,
        "scenarios": list(scenarios.values()),
    }


def _profile_rows(
    profiler: cProfile.Profile,
    *,
    sort_by: str,
    limit: int,
) -> dict[str, Any]:
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats(sort_by)
    top_functions = []
    for rank, func in enumerate((stats.fcn_list or [])[:limit], start=1):
        primitive_calls, total_calls, tottime, cumtime, _callers = stats.stats[func]
        top_functions.append(
            {
                "rank": rank,
                "function": pstats.func_std_string(func),
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "tottime_sec": tottime,
                "cumtime_sec": cumtime,
                "per_call_tottime_sec": tottime / total_calls if total_calls else 0.0,
                "per_call_cumtime_sec": cumtime / primitive_calls
                if primitive_calls
                else 0.0,
            }
        )
    return {
        "profiler": "cProfile",
        "profile_scope": (
            "source-fidelity runner-surface measured-loop profile; includes "
            "scenario loading, inclusive runner calls, payload/json wrapping, "
            "common-trace projection, and benchmark loop work. Excludes warmup "
            "and manifest collection."
        ),
        "profile_not": "production environment stepping or source-internal split timers",
        "sort": sort_by,
        "limit": limit,
        "primitive_calls": stats.prim_calls,
        "total_calls": stats.total_calls,
        "total_tottime_sec": stats.total_tt,
        "top_functions": top_functions,
    }


def run_profiled(
    manifests: list[str | Path],
    repeat: int,
    warmup: int,
    *,
    profile_sort: str,
    profile_limit: int,
) -> dict[str, Any]:
    profiler = cProfile.Profile()
    summary = run(
        manifests=manifests,
        repeat=repeat,
        warmup=warmup,
        profiler=profiler,
    )
    summary["profile"] = _profile_rows(
        profiler,
        sort_by=profile_sort,
        limit=profile_limit,
    )
    return summary


def print_plain(summary: dict[str, Any]) -> None:
    timings = summary["timings_sec"]
    workload = summary["workload"]
    print(f"benchmark={summary['benchmark_id']}")
    print(f"source_fidelity_claim={summary['source_fidelity_claim']}")
    print(f"scenario_count={workload['scenario_count']}")
    print(f"repeat={workload['repeat']}")
    print(f"warmup={workload['warmup']}")
    print(f"scenario_iterations={workload['scenario_iterations']}")
    print(f"elapsed_sec={summary['elapsed_sec']:.6f}")
    print(f"scenario_iterations_per_sec={summary['scenario_iterations_per_sec']:.1f}")
    for bucket in TIMER_DEFINITIONS:
        print(f"{bucket}_sec={timings[bucket]:.6f}")
    print(
        "split_timer_status=movement/point_insertion/collision_scan/"
        "print_manager are scaffolded, not measured"
    )
    profile = summary.get("profile")
    if profile:
        print(f"profile_scope={profile['profile_scope']}")
        print(f"profile_not={profile['profile_not']}")
        print(
            "profile_totals="
            f"primitive_calls:{profile['primitive_calls']} "
            f"total_calls:{profile['total_calls']} "
            f"tottime_sec:{profile['total_tottime_sec']:.6f}"
        )
        print(f"profile_top_functions_sort={profile['sort']}")
        for row in profile["top_functions"]:
            print(
                "profile_top_function="
                f"rank:{row['rank']} "
                f"primitive_calls:{row['primitive_calls']} "
                f"total_calls:{row['total_calls']} "
                f"tottime_sec:{row['tottime_sec']:.6f} "
                f"cumtime_sec:{row['cumtime_sec']:.6f} "
                f"function:{row['function']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", dest="manifests")
    parser.add_argument("--repeat", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--format", choices=["plain", "json"], default="plain")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Attach a cProfile/pstats summary for the source-fidelity runner surface.",
    )
    parser.add_argument(
        "--profile-sort",
        choices=["cumulative", "tottime", "calls"],
        default="cumulative",
    )
    parser.add_argument("--profile-limit", type=_positive_int, default=20)
    args = parser.parse_args()
    manifests = args.manifests if args.manifests is not None else list(DEFAULT_MANIFESTS)
    if args.profile:
        summary = run_profiled(
            manifests=manifests,
            repeat=args.repeat,
            warmup=args.warmup,
            profile_sort=args.profile_sort,
            profile_limit=args.profile_limit,
        )
    else:
        summary = run(manifests=manifests, repeat=args.repeat, warmup=args.warmup)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)


if __name__ == "__main__":
    main()
