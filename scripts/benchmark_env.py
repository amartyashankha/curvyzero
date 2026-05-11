"""Benchmark the reference CurvyZero environment.

This is a smoke benchmark, not a final performance target. It exists so every
backend change can report steps/sec and episodes/sec from the beginning.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np

from curvyzero.env import CurvyTronConfig, CurvyTronEnv


TIMER_DEFINITIONS = {
    "reset": (
        "Inclusive env.reset wall time. Includes state allocation, initial occupancy "
        "marks, and reset observation generation."
    ),
    "action_sample": "Random joint-action dict construction in the benchmark harness.",
    "step": (
        "Inclusive env.step wall time. Includes physics ticks, observation generation, "
        "reward/terminal/truncation/info dicts, and StepResult construction."
    ),
    "loop_overhead": (
        "Elapsed benchmark time not covered by reset, action_sample, or step timers."
    ),
}


REQUESTED_SPLIT_TIMER_STATUS = {
    "movement": {
        "status": "not_measured",
        "reason": (
            "Movement is inside CurvyTronEnv._physics_tick with collision lookup and "
            "trail drawing."
        ),
        "next_step": (
            "Add source-level spans or refactor the toy-v0 transition before reporting "
            "movement time."
        ),
    },
    "trail_collision_writes": {
        "status": "not_measured",
        "reason": (
            "Collision lookup, death flags, and trail writes are not exposed as benchmark-visible "
            "spans in the current implementation."
        ),
        "next_step": (
            "Instrument or split the transition at the environment layer before claiming "
            "this bucket."
        ),
    },
    "observation_generation": {
        "status": "not_measured",
        "reason": "Observation generation is nested inside reset and step return construction.",
        "next_step": (
            "Expose a measured observation span before using this as optimization "
            "evidence."
        ),
    },
    "reset_autoreset": {
        "status": "partially_measured",
        "measured_as": "reset",
        "reason": (
            "The benchmark measures explicit env.reset calls. There is no autoreset "
            "wrapper in this smoke."
        ),
        "next_step": (
            "Add wrapper/autoreset benchmarks only after the public reset/step contract "
            "is stable."
        ),
    },
    "wrapper_dict_output_overhead": {
        "status": "not_measured",
        "reason": "Dict and StepResult construction are included in the coarse step timer.",
        "next_step": (
            "Add an explicit wrapper/output span or isolated microbenchmark before "
            "optimizing it."
        ),
    },
}


def _git_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, check=False, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_manifest() -> dict[str, Any]:
    revision = _git_text(["git", "rev-parse", "--short", "HEAD"])
    status = _git_text(["git", "status", "--short"])
    status_lines = status.splitlines() if status else []
    return {
        "revision": revision or "unavailable",
        "working_tree_status": "changes-present" if status_lines else "clean-or-unavailable",
        "status_entry_count": len(status_lines),
    }


def _config_manifest(config: CurvyTronConfig) -> dict[str, Any]:
    return {
        "ruleset": config.ruleset,
        "rules_hash": config.rules_hash,
        "players": config.players,
        "width": config.width,
        "height": config.height,
        "action_repeat": config.action_repeat,
        "max_ticks": config.max_ticks,
        "action_count": config.action_count,
    }


def _runtime_manifest() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _build_manifest(
    *,
    config: CurvyTronConfig,
    seed: int,
    episodes: int,
    max_steps: int,
    observations: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "schema_version": "toy-v0-benchmark-manifest-v1",
        "benchmark_id": "curvytron_env_step_throughput",
        "benchmark_label": "toy-v0 single-env random-action smoke",
        "backend": "python_curvytron_env_single",
        "source_fidelity_claim": "none; simplified toy-v0 measurement scaffold",
        "command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "workload": {
            "seed": seed,
            "episodes": episodes,
            "max_steps_per_episode": max_steps,
            "batch_size": 1,
            "action_policy": "uniform_random_per_agent",
            "scenario_set": "none",
            "autoreset": "none; benchmark calls reset once per episode",
        },
        "environment": {
            "env_id": "curvyzero-v0",
            "config": _config_manifest(config),
            "agents": [f"player_{idx}" for idx in range(config.players)],
            "observation_shapes": {
                agent: list(observation.shape) for agent, observation in observations.items()
            },
        },
        "source": _git_manifest(),
        "runtime": _runtime_manifest(),
        "timer_schema": {
            "version": "toy-v0-timers-v1",
            "definitions": TIMER_DEFINITIONS,
            "requested_split_timer_status": REQUESTED_SPLIT_TIMER_STATUS,
            "caveat": "This benchmark reports only externally measured coarse timers.",
        },
    }


def run(seed: int, episodes: int, max_steps: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    config = CurvyTronConfig(action_repeat=1)
    env = CurvyTronEnv(config)
    total_steps = 0
    reset_sec = 0.0
    action_sec = 0.0
    step_sec = 0.0
    last_observations: dict[str, np.ndarray] | None = None
    started = time.perf_counter()

    for episode in range(episodes):
        reset_started = time.perf_counter()
        last_observations = env.reset(seed=seed + episode)
        reset_sec += time.perf_counter() - reset_started
        for _ in range(max_steps):
            action_started = time.perf_counter()
            actions = {
                "player_0": int(rng.integers(env.config.action_count)),
                "player_1": int(rng.integers(env.config.action_count)),
            }
            action_sec += time.perf_counter() - action_started
            step_started = time.perf_counter()
            result = env.step(actions)
            step_sec += time.perf_counter() - step_started
            last_observations = result.observations
            total_steps += 1
            if result.terminated["player_0"] or result.truncated["player_0"]:
                break

    elapsed = time.perf_counter() - started
    measured_sec = reset_sec + action_sec + step_sec
    steps_per_sec = total_steps / elapsed if elapsed else 0.0
    episodes_per_sec = episodes / elapsed if elapsed else 0.0
    observations = last_observations or {}
    return {
        "benchmark": "curvytron_env_step_throughput",
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "steps": total_steps,
        "elapsed_sec": elapsed,
        "steps_per_sec": steps_per_sec,
        "episodes_per_sec": episodes_per_sec,
        "step_calls": total_steps,
        "config": _config_manifest(config),
        "manifest": _build_manifest(
            config=config,
            seed=seed,
            episodes=episodes,
            max_steps=max_steps,
            observations=observations,
        ),
        "timings_sec": {
            "reset": reset_sec,
            "action_sample": action_sec,
            "step": step_sec,
            "loop_overhead": max(0.0, elapsed - measured_sec),
        },
        "timing_counts": {
            "reset": episodes,
            "action_sample": total_steps,
            "step": total_steps,
        },
        "requested_split_timer_status": REQUESTED_SPLIT_TIMER_STATUS,
    }


def print_plain(summary: dict[str, Any]) -> None:
    timings = summary["timings_sec"]
    print(f"episodes={summary['episodes']}")
    print(f"steps={summary['steps']}")
    print(f"step_calls={summary['step_calls']}")
    print(f"elapsed_sec={summary['elapsed_sec']:.4f}")
    print(f"steps_per_sec={summary['steps_per_sec']:.1f}")
    print(f"episodes_per_sec={summary['episodes_per_sec']:.1f}")
    print(f"reset_sec={timings['reset']:.4f}")
    print(f"action_sample_sec={timings['action_sample']:.4f}")
    print(f"step_sec={timings['step']:.4f}")
    print(f"loop_overhead_sec={timings['loop_overhead']:.4f}")
    print("split_timer_status=coarse_only; movement/trail/observation/wrapper splits not measured")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1_000)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--format", choices=["plain", "json"], default="plain")
    args = parser.parse_args()
    summary = run(seed=args.seed, episodes=args.episodes, max_steps=args.max_steps)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)


if __name__ == "__main__":
    main()
