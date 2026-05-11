"""Benchmark the debug CurvyTron visual observation smoke surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env.source_env import CurvyTronSourceEnv  # noqa: E402
from curvyzero.training.curvytron_visual_observation import (  # noqa: E402
    DebugOccupancyGray64FrameStack,
    DebugOccupancyGray64Renderer,
    debug_occupancy_gray64_schema,
    normalize_debug_occupancy_gray64_for_lightzero,
)


def run_benchmark(
    *,
    batch_size: int,
    rollout_steps: int,
    step_ms: float,
    warmup_ms: float,
    startup_advance_ms: float,
    seed: int,
    random_tape_length: int,
    stack: bool,
    stack_copy: bool,
    latency_samples: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    timers = {
        "reset_sec": 0.0,
        "source_env_step_sec": 0.0,
        "snapshot_world_body_extraction_sec": 0.0,
        "render_frame_sec": 0.0,
        "normalize_frame_sec": 0.0,
        "frame_stack_update_copy_sec": 0.0,
    }
    latency_sec: dict[str, list[float]] = {key: [] for key in timers}
    reset_started = time.perf_counter()
    envs = [
        CurvyTronSourceEnv(random_values=rng.random(random_tape_length).tolist())
        for _ in range(batch_size)
    ]
    renderer = DebugOccupancyGray64Renderer()
    lightzero_frames = [np.empty((1, 64, 64), dtype=np.float32) for _ in range(batch_size)]
    stacks = [DebugOccupancyGray64FrameStack() for _ in range(batch_size)] if stack else []
    snapshots = []
    for env in envs:
        env.reset(player_count=2, warmup_ms=warmup_ms)
        env.advance_timers(startup_advance_ms)
        snapshots.append(env.snapshot("after_profiler_startup_advance"))
    reset_duration = time.perf_counter() - reset_started
    timers["reset_sec"] = reset_duration
    if latency_samples:
        latency_sec["reset_sec"].append(reset_duration)

    frames = 0
    nonzero_pixels_total = 0
    world_body_count_total = 0
    max_world_body_count = 0
    live_avatar_count_total = 0
    in_round_frame_count = 0
    world_active_frame_count = 0
    last_raw_frame: np.ndarray | None = None
    last_lightzero_frame: np.ndarray | None = None
    last_stack: np.ndarray | None = None
    started = time.perf_counter()

    for _ in range(rollout_steps):
        for row, env in enumerate(envs):
            actions = [int(rng.integers(3)) - 1, int(rng.integers(3)) - 1]

            step_started = time.perf_counter()
            snapshots[row] = env.step(actions, elapsed_ms=step_ms)
            step_duration = time.perf_counter() - step_started
            timers["source_env_step_sec"] += step_duration
            if latency_samples:
                latency_sec["source_env_step_sec"].append(step_duration)

            extract_started = time.perf_counter()
            world_bodies = env.world_bodies_snapshot()
            snapshot = snapshots[row]
            world_body_count = len(world_bodies)
            world_body_count_total += world_body_count
            max_world_body_count = max(max_world_body_count, world_body_count)
            avatars = snapshot.get("avatars", ())
            live_avatar_count_total += sum(
                1 for avatar in avatars if isinstance(avatar, dict) and bool(avatar.get("alive", True))
            )
            game = snapshot.get("game", {})
            if isinstance(game, dict) and bool(game.get("inRound", False)):
                in_round_frame_count += 1
            if isinstance(game, dict) and bool(game.get("worldActive", False)):
                world_active_frame_count += 1
            extract_duration = time.perf_counter() - extract_started
            timers["snapshot_world_body_extraction_sec"] += extract_duration
            if latency_samples:
                latency_sec["snapshot_world_body_extraction_sec"].append(extract_duration)

            render_started = time.perf_counter()
            last_raw_frame = renderer.render(snapshot, world_bodies=world_bodies)
            render_duration = time.perf_counter() - render_started
            timers["render_frame_sec"] += render_duration
            if latency_samples:
                latency_sec["render_frame_sec"].append(render_duration)
            nonzero_pixels_total += int(np.count_nonzero(last_raw_frame))

            normalize_started = time.perf_counter()
            last_lightzero_frame = normalize_debug_occupancy_gray64_for_lightzero(
                last_raw_frame,
                out=lightzero_frames[row],
            )
            normalize_duration = time.perf_counter() - normalize_started
            timers["normalize_frame_sec"] += normalize_duration
            if latency_samples:
                latency_sec["normalize_frame_sec"].append(normalize_duration)

            if stack:
                stack_started = time.perf_counter()
                last_stack = stacks[row].update(last_lightzero_frame, copy=stack_copy)
                stack_duration = time.perf_counter() - stack_started
                timers["frame_stack_update_copy_sec"] += stack_duration
                if latency_samples:
                    latency_sec["frame_stack_update_copy_sec"].append(stack_duration)

            frames += 1

    elapsed = time.perf_counter() - started
    measured = sum(timers.values())
    schema = debug_occupancy_gray64_schema()
    visual_observation_sec = (
        timers["snapshot_world_body_extraction_sec"]
        + timers["render_frame_sec"]
        + timers["normalize_frame_sec"]
        + timers["frame_stack_update_copy_sec"]
    )
    return {
        "schema": schema,
        "workload": {
            "batch_size": batch_size,
            "rollout_steps": rollout_steps,
            "frames": frames,
            "step_ms": step_ms,
            "reset_warmup_ms": warmup_ms,
            "startup_advance_ms": startup_advance_ms,
            "seed": seed,
            "source_random_mode": "seeded_math_random_tape",
            "random_tape_length_per_env": random_tape_length,
            "stack_enabled": stack,
            "stack_copy_enabled": bool(stack and stack_copy),
            "latency_samples_enabled": latency_samples,
        },
        "surface": {
            "name": schema["surface"],
            "schema_id": schema["schema_id"],
            "truth_level": schema["truth_level"],
            "source_fidelity_level": schema["source_fidelity_level"],
            "source_backed_observation_fidelity": schema[
                "source_backed_observation_fidelity"
            ],
            "uses_ale": schema["uses_ale"],
            "ale_usage": schema["ale_usage"],
        },
        "timed_components": {
            "env_step": True,
            "render": True,
            "stack": bool(stack),
            "stack_copy": bool(stack and stack_copy),
            "reset": True,
            "policy_search": False,
            "replay": False,
            "learner": False,
        },
        "denominators": {
            "loop_elapsed_excludes_reset": True,
            "loop_elapsed_includes_env_step": True,
            "loop_elapsed_includes_render": True,
            "loop_elapsed_includes_normalize": True,
            "loop_elapsed_includes_stack": bool(stack),
            "loop_elapsed_includes_stack_copy": bool(stack and stack_copy),
            "visual_observation_includes_snapshot_world_bodies": True,
            "visual_observation_includes_render": True,
            "visual_observation_includes_normalize": True,
            "visual_observation_includes_stack": bool(stack),
            "visual_observation_includes_stack_copy": bool(stack and stack_copy),
            "visual_observation_excludes_env_step": True,
            "policy_search_included": False,
            "replay_included": False,
        },
        "timing_sec": {
            **timers,
            "elapsed": elapsed,
            "loop_overhead_sec": max(0.0, elapsed - (measured - timers["reset_sec"])),
        },
        "latency_us": _latency_report(latency_sec) if latency_samples else {},
        "throughput": {
            "whole_loop_env_transitions_per_sec": frames / elapsed if elapsed else 0.0,
            "env_step_only_fps": frames / timers["source_env_step_sec"]
            if timers["source_env_step_sec"]
            else 0.0,
            "render_only_fps": frames / timers["render_frame_sec"]
            if timers["render_frame_sec"]
            else 0.0,
            "visual_observation_fps": frames / visual_observation_sec
            if visual_observation_sec
            else 0.0,
        },
        "density": {
            "mean_nonzero_pixels_per_frame": nonzero_pixels_total / frames if frames else 0.0,
            "mean_world_bodies_per_frame": world_body_count_total / frames if frames else 0.0,
            "max_world_bodies_per_frame": max_world_body_count,
            "mean_live_avatars_per_frame": live_avatar_count_total / frames if frames else 0.0,
            "in_round_frame_ratio": in_round_frame_count / frames if frames else 0.0,
            "world_active_frame_ratio": world_active_frame_count / frames if frames else 0.0,
        },
        "observation": {
            "raw_frame_shape": (
                list(last_raw_frame.shape) if last_raw_frame is not None else schema["shape"]
            ),
            "raw_frame_dtype": (
                str(last_raw_frame.dtype)
                if last_raw_frame is not None
                else schema["raw_renderer_dtype"]
            ),
            "lightzero_frame_shape": (
                list(last_lightzero_frame.shape)
                if last_lightzero_frame is not None
                else schema["shape"]
            ),
            "lightzero_frame_dtype": (
                str(last_lightzero_frame.dtype)
                if last_lightzero_frame is not None
                else schema["lightzero_payload_dtype"]
            ),
            "stack_shape": (
                list(last_stack.shape)
                if last_stack is not None
                else (schema["stack_shape"] if stack else None)
            ),
            "stack_dtype": str(last_stack.dtype) if last_stack is not None else None,
        },
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _latency_report(latency_sec: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for key, values in latency_sec.items():
        if not values:
            continue
        arr_us = np.asarray(values, dtype=np.float64) * 1_000_000.0
        report[key] = {
            "samples": int(arr_us.size),
            "p50": float(np.percentile(arr_us, 50)),
            "p95": float(np.percentile(arr_us, 95)),
            "p99": float(np.percentile(arr_us, 99)),
            "max": float(np.max(arr_us)),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--rollout-steps", type=_positive_int, default=8)
    parser.add_argument("--step-ms", type=_nonnegative_float, default=1000.0 / 60.0)
    parser.add_argument("--warmup-ms", type=_nonnegative_float, default=0.0)
    parser.add_argument("--startup-advance-ms", type=_nonnegative_float, default=3000.0)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--random-tape-length", type=_positive_int, default=8192)
    parser.add_argument("--stack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stack-copy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latency-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_benchmark(
        batch_size=int(args.batch_size),
        rollout_steps=int(args.rollout_steps),
        step_ms=float(args.step_ms),
        warmup_ms=float(args.warmup_ms),
        startup_advance_ms=float(args.startup_advance_ms),
        seed=int(args.seed),
        random_tape_length=int(args.random_tape_length),
        stack=bool(args.stack),
        stack_copy=bool(args.stack_copy),
        latency_samples=bool(args.latency_samples),
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    timing = report["timing_sec"]
    throughput = report["throughput"]
    observation = report["observation"]
    stack_summary = (
        f"{observation['stack_dtype']}{tuple(observation['stack_shape'])}"
        if observation["stack_shape"] is not None
        else "disabled"
    )
    print(
        f"{report['schema']['label']} "
        f"surface={report['surface']['name']} "
        f"B={report['workload']['batch_size']} T={report['workload']['rollout_steps']} "
        f"startup_advance={report['workload']['startup_advance_ms']:.1f}ms "
        f"reset={timing['reset_sec']:.6f}s "
        f"source_env_step={timing['source_env_step_sec']:.6f}s "
        f"snapshot/world_bodies={timing['snapshot_world_body_extraction_sec']:.6f}s "
        f"render_frame={timing['render_frame_sec']:.6f}s "
        f"normalize_frame={timing['normalize_frame_sec']:.6f}s "
        f"frame_stack_update/copy={timing['frame_stack_update_copy_sec']:.6f}s "
        f"stack_copy={report['workload']['stack_copy_enabled']} "
        f"throughput={throughput['whole_loop_env_transitions_per_sec']:.1f} steps/s "
        f"nonzero={report['density']['mean_nonzero_pixels_per_frame']:.1f}/frame "
        f"bodies={report['density']['mean_world_bodies_per_frame']:.1f}/frame "
        f"frame={observation['lightzero_frame_dtype']}"
        f"{tuple(observation['lightzero_frame_shape'])} "
        f"stack={stack_summary}"
    )


if __name__ == "__main__":
    main()
