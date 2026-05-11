"""Compare local serial, thread, and process sharding for toy self-play rows.

This is a practical scout for the current Python object-env bridge. It does not
claim source fidelity, does not run MCTS, and does not benchmark the vector
fixture path. It answers one smaller question: when the hot loop is still
Python env objects plus small NumPy policy work, does coarse local parallelism
help enough to justify the added orchestration cost?
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env import CurvyTronConfig  # noqa: E402
from curvyzero.env import CurvyTronEnv  # noqa: E402


BENCHMARK_ID = "toy_curvytron_selfplay_parallel_bridge_v0"
SOURCE_FIDELITY_CLAIM = (
    "none; simplified toy CurvyTron object env, synthetic NumPy policy, local sharding"
)
BOTTLENECK_SUMMARY_SCHEMA = "curvyzero_selfplay_parallel_bridge_bottleneck_summary/v0"
TIMER_DEFINITIONS = {
    "obs_batch_build": "Copy per-env observation dicts into one [B_shard, P, obs_dim] array.",
    "policy_batch": "Synthetic NumPy tanh MLP over ego rows; not MCTS or a checkpoint.",
    "action_assembly": "Convert action ids back into per-env joint-action dicts.",
    "env_step": "Sequential CurvyTronEnv.step calls inside one shard.",
    "autoreset": "Reset terminal/truncated env rows after the terminal transition.",
    "replay_stage": "Copy obs/actions/done into a preallocated in-memory array buffer.",
    "loop_overhead": "Measured loop wall time not covered by the named buckets.",
}


@dataclass(frozen=True)
class ShardProfile:
    shard_index: int
    env_count: int
    steps: int
    warmup: int
    seed: int
    hidden_dim: int
    action_repeat: int
    max_ticks: int


@dataclass
class ShardState:
    envs: list[CurvyTronEnv]
    observations: list[dict[str, np.ndarray]]
    agents: list[str]
    weights_in: np.ndarray
    bias_hidden: np.ndarray
    weights_out: np.ndarray
    bias_out: np.ndarray
    obs_dim: int
    replay_obs: np.ndarray
    replay_actions: np.ndarray
    replay_done: np.ndarray
    row_cursor: int = 0
    checksum: float = 0.0
    reset_counter: int = 0


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
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _make_state(profile: ShardProfile, *, steps: int) -> ShardState:
    config = CurvyTronConfig(
        action_repeat=profile.action_repeat,
        max_ticks=profile.max_ticks,
    )
    envs = [CurvyTronEnv(config) for _ in range(profile.env_count)]
    seed_base = profile.seed + profile.shard_index * 100_000
    observations = [env.reset(seed=seed_base + index) for index, env in enumerate(envs)]
    agents = envs[0].agents
    obs_dim = int(observations[0][agents[0]].shape[0])

    rng = np.random.default_rng(seed_base)
    weights_in = rng.normal(0.0, 0.05, size=(obs_dim, profile.hidden_dim)).astype(np.float32)
    bias_hidden = rng.normal(0.0, 0.01, size=(profile.hidden_dim,)).astype(np.float32)
    weights_out = rng.normal(0.0, 0.05, size=(profile.hidden_dim, 3)).astype(np.float32)
    bias_out = rng.normal(0.0, 0.01, size=(3,)).astype(np.float32)

    rows = profile.env_count * len(agents) * steps
    return ShardState(
        envs=envs,
        observations=observations,
        agents=agents,
        weights_in=weights_in,
        bias_hidden=bias_hidden,
        weights_out=weights_out,
        bias_out=bias_out,
        obs_dim=obs_dim,
        replay_obs=np.empty((rows, obs_dim), dtype=np.float32),
        replay_actions=np.empty(rows, dtype=np.int8),
        replay_done=np.empty(rows, dtype=bool),
    )


def _build_obs_batch(state: ShardState) -> np.ndarray:
    obs_batch = np.empty((len(state.envs), len(state.agents), state.obs_dim), dtype=np.float32)
    for env_index, observations in enumerate(state.observations):
        for player_index, agent in enumerate(state.agents):
            obs_batch[env_index, player_index] = observations[agent]
    return obs_batch


def _policy_actions(obs_batch: np.ndarray, state: ShardState) -> np.ndarray:
    rows = obs_batch.reshape(-1, state.obs_dim)
    hidden = np.tanh(rows @ state.weights_in + state.bias_hidden)
    logits = hidden @ state.weights_out + state.bias_out
    return np.argmax(logits, axis=1).reshape(obs_batch.shape[:2]).astype(np.int8)


def _assemble_actions(actions: np.ndarray, agents: list[str]) -> list[dict[str, int]]:
    return [
        {agent: int(actions[env_index, player_index]) for player_index, agent in enumerate(agents)}
        for env_index in range(actions.shape[0])
    ]


def _step_envs(
    state: ShardState,
    joint_actions: list[dict[str, int]],
) -> tuple[np.ndarray, int]:
    done = np.zeros(len(state.envs), dtype=bool)
    terminal_count = 0
    next_observations: list[dict[str, np.ndarray]] = []
    for env_index, env in enumerate(state.envs):
        step = env.step(joint_actions[env_index])
        row_done = any(step.terminated.values()) or any(step.truncated.values())
        done[env_index] = row_done
        terminal_count += int(row_done)
        next_observations.append(step.observations)
    state.observations = next_observations
    return done, terminal_count


def _autoreset_done_rows(state: ShardState, done: np.ndarray, profile: ShardProfile) -> int:
    reset_rows = np.flatnonzero(done)
    seed_base = profile.seed + profile.shard_index * 100_000 + 1_000_000
    for env_index in reset_rows:
        state.reset_counter += 1
        reset_seed = seed_base + state.reset_counter
        state.observations[int(env_index)] = state.envs[int(env_index)].reset(seed=reset_seed)
    return int(reset_rows.size)


def _stage_replay(
    state: ShardState,
    obs_batch: np.ndarray,
    actions: np.ndarray,
    done: np.ndarray,
) -> int:
    rows = obs_batch.reshape(-1, state.obs_dim)
    flat_actions = actions.reshape(-1)
    flat_done = np.repeat(done, len(state.agents))
    row_count = rows.shape[0]
    start = state.row_cursor
    end = start + row_count
    state.replay_obs[start:end] = rows
    state.replay_actions[start:end] = flat_actions
    state.replay_done[start:end] = flat_done
    state.row_cursor = end
    state.checksum += float(flat_actions.sum()) + float(flat_done.sum())
    return row_count


def _run_steps(
    profile: ShardProfile,
    state: ShardState,
    *,
    steps: int,
    collect_latency: bool,
) -> tuple[dict[str, float], dict[str, int], float, list[float]]:
    timers = {name: 0.0 for name in TIMER_DEFINITIONS if name != "loop_overhead"}
    counts = {
        "env_steps": 0,
        "ego_rows": 0,
        "replay_rows": 0,
        "terminal_env_steps": 0,
        "autoreset_rows": 0,
    }
    actor_step_latency_sec: list[float] = []
    started_loop = time.perf_counter()
    for _ in range(steps):
        actor_started = time.perf_counter()

        started = time.perf_counter()
        obs_batch = _build_obs_batch(state)
        timers["obs_batch_build"] += time.perf_counter() - started

        started = time.perf_counter()
        actions = _policy_actions(obs_batch, state)
        timers["policy_batch"] += time.perf_counter() - started

        started = time.perf_counter()
        joint_actions = _assemble_actions(actions, state.agents)
        timers["action_assembly"] += time.perf_counter() - started

        started = time.perf_counter()
        done, terminal_count = _step_envs(state, joint_actions)
        timers["env_step"] += time.perf_counter() - started

        started = time.perf_counter()
        reset_count = _autoreset_done_rows(state, done, profile)
        timers["autoreset"] += time.perf_counter() - started

        started = time.perf_counter()
        replay_rows = _stage_replay(state, obs_batch, actions, done)
        timers["replay_stage"] += time.perf_counter() - started

        counts["env_steps"] += len(state.envs)
        counts["ego_rows"] += len(state.envs) * len(state.agents)
        counts["replay_rows"] += replay_rows
        counts["terminal_env_steps"] += terminal_count
        counts["autoreset_rows"] += reset_count
        if collect_latency:
            actor_step_latency_sec.append(time.perf_counter() - actor_started)

    elapsed = time.perf_counter() - started_loop
    measured = sum(timers.values())
    timers["loop_overhead"] = max(0.0, elapsed - measured)
    return timers, counts, elapsed, actor_step_latency_sec


def _run_shard(profile: ShardProfile) -> dict[str, Any]:
    if profile.env_count <= 0:
        raise ValueError("env_count must be greater than zero")

    if profile.warmup:
        warmup_state = _make_state(profile, steps=profile.warmup)
        warmup_started = time.perf_counter()
        _run_steps(profile, warmup_state, steps=profile.warmup, collect_latency=False)
        warmup_sec = time.perf_counter() - warmup_started
    else:
        warmup_sec = 0.0

    setup_started = time.perf_counter()
    state = _make_state(profile, steps=profile.steps)
    setup_sec = time.perf_counter() - setup_started
    timers, counts, elapsed, latency = _run_steps(
        profile,
        state,
        steps=profile.steps,
        collect_latency=True,
    )
    return {
        "profile": asdict(profile),
        "setup_sec": setup_sec,
        "warmup_sec": warmup_sec,
        "elapsed_sec": elapsed,
        "timings_sec": timers,
        "counts": counts,
        "actor_step_latency_sec": latency,
        "checksum": state.checksum,
    }


def _split_batch(batch: int, workers: int) -> list[int]:
    if workers <= 0:
        raise ValueError("workers must be greater than zero")
    if workers > batch:
        workers = batch
    base, remainder = divmod(batch, workers)
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def _profiles(
    *,
    batch: int,
    workers: int,
    steps: int,
    warmup: int,
    seed: int,
    hidden_dim: int,
    action_repeat: int,
    max_ticks: int,
    sharded: bool,
) -> list[ShardProfile]:
    env_counts = _split_batch(batch, workers) if sharded else [batch]
    return [
        ShardProfile(
            shard_index=index,
            env_count=env_count,
            steps=steps,
            warmup=warmup,
            seed=seed,
            hidden_dim=hidden_dim,
            action_repeat=action_repeat,
            max_ticks=max_ticks,
        )
        for index, env_count in enumerate(env_counts)
    ]


def _latency_summary(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(data.size),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(data.max()),
    }


def _estimated_sync_step_latency(mode: str, shard_results: list[dict[str, Any]]) -> list[float]:
    latency_by_shard = [
        np.asarray(result["actor_step_latency_sec"], dtype=np.float64) for result in shard_results
    ]
    if not latency_by_shard:
        return []
    if len(latency_by_shard) == 1:
        return latency_by_shard[0].tolist()

    stacked = np.stack(latency_by_shard, axis=0)
    if mode == "serial-sharded":
        return stacked.sum(axis=0).tolist()
    return stacked.max(axis=0).tolist()


def _rank_bucket_percentages(
    bucket_pct: Mapping[str, float],
    *,
    min_pct: float = 0.0,
) -> list[dict[str, float | str]]:
    return [
        {"bucket": name, "pct_serialized_shard_loop": float(pct)}
        for name, pct in sorted(
            bucket_pct.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
        if float(pct) >= min_pct
    ]


def _bottleneck_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    bucket_pct = result["bucket_pct_serialized_shard_loop"]
    ranked = _rank_bucket_percentages(bucket_pct)
    top = ranked[0] if ranked else {"bucket": "none", "pct_serialized_shard_loop": 0.0}
    env_pct = float(bucket_pct.get("env_step", 0.0))
    policy_pct = float(bucket_pct.get("policy_batch", 0.0))
    latency = result["latency_sec"]["estimated_sync_actor_step"]
    return {
        "schema": BOTTLENECK_SUMMARY_SCHEMA,
        "ranked_buckets": ranked,
        "top_bucket": top["bucket"],
        "top_bucket_pct_serialized": top["pct_serialized_shard_loop"],
        "env_step_pct_serialized": env_pct,
        "non_env_step_pct_serialized": max(0.0, 100.0 - env_pct),
        "policy_batch_pct_serialized": policy_pct,
        "policy_to_env_time_ratio": policy_pct / env_pct if env_pct else 0.0,
        "env_steps_per_sec_wall": result["rates"]["env_steps_per_sec_wall"],
        "env_steps_per_sec_measured_loop_estimate": result["rates"][
            "env_steps_per_sec_measured_loop_estimate"
        ],
        "estimated_sync_actor_step_p95_ms": float(latency["p95"]) * 1000.0,
        "estimated_sync_actor_step_p99_ms": float(latency["p99"]) * 1000.0,
        "scope": (
            "toy object-env actor shard timing; read throughput together with "
            "estimated synchronous action latency and bucket shares"
        ),
    }


def _run_mode(
    mode: str,
    *,
    batch: int,
    workers: int,
    steps: int,
    warmup: int,
    seed: int,
    hidden_dim: int,
    action_repeat: int,
    max_ticks: int,
) -> dict[str, Any]:
    if mode == "serial":
        profiles = _profiles(
            batch=batch,
            workers=1,
            steps=steps,
            warmup=warmup,
            seed=seed,
            hidden_dim=hidden_dim,
            action_repeat=action_repeat,
            max_ticks=max_ticks,
            sharded=False,
        )
    elif mode in {"serial-sharded", "thread", "process"}:
        profiles = _profiles(
            batch=batch,
            workers=workers,
            steps=steps,
            warmup=warmup,
            seed=seed,
            hidden_dim=hidden_dim,
            action_repeat=action_repeat,
            max_ticks=max_ticks,
            sharded=True,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    started = time.perf_counter()
    if mode in {"serial", "serial-sharded"}:
        shard_results = [_run_shard(profile) for profile in profiles]
    elif mode == "thread":
        with ThreadPoolExecutor(max_workers=len(profiles)) as executor:
            shard_results = list(executor.map(_run_shard, profiles))
    else:
        with ProcessPoolExecutor(max_workers=len(profiles)) as executor:
            shard_results = list(executor.map(_run_shard, profiles))
    wall_elapsed_sec = time.perf_counter() - started

    timer_totals = {name: 0.0 for name in TIMER_DEFINITIONS}
    count_totals = {
        "env_steps": 0,
        "ego_rows": 0,
        "replay_rows": 0,
        "terminal_env_steps": 0,
        "autoreset_rows": 0,
    }
    checksum = 0.0
    for result in shard_results:
        for name, value in result["timings_sec"].items():
            timer_totals[name] += float(value)
        for name, value in result["counts"].items():
            count_totals[name] += int(value)
        checksum += float(result["checksum"])

    shard_loop_elapsed = [float(result["elapsed_sec"]) for result in shard_results]
    if mode == "serial-sharded":
        measured_loop_wall_estimate = sum(shard_loop_elapsed)
    elif mode == "serial":
        measured_loop_wall_estimate = shard_loop_elapsed[0]
    else:
        measured_loop_wall_estimate = max(shard_loop_elapsed) if shard_loop_elapsed else 0.0

    sync_latency = _estimated_sync_step_latency(mode, shard_results)
    env_steps = count_totals["env_steps"]
    ego_rows = count_totals["ego_rows"]
    serialized_loop_sum = sum(shard_loop_elapsed)
    bucket_pct_serialized = {
        name: (value * 100.0 / serialized_loop_sum if serialized_loop_sum else 0.0)
        for name, value in timer_totals.items()
    }
    result = {
        "mode": mode,
        "requested_workers": workers if mode != "serial" else 1,
        "actual_shards": len(shard_results),
        "shard_env_counts": [result["profile"]["env_count"] for result in shard_results],
        "wall_elapsed_sec": wall_elapsed_sec,
        "setup_sec_sum": sum(float(result["setup_sec"]) for result in shard_results),
        "warmup_sec_sum": sum(float(result["warmup_sec"]) for result in shard_results),
        "measured_loop_wall_estimate_sec": measured_loop_wall_estimate,
        "serialized_shard_loop_sum_sec": serialized_loop_sum,
        "timings_sec_sum": timer_totals,
        "counts": count_totals,
        "rates": {
            "env_steps_per_sec_wall": env_steps / wall_elapsed_sec if wall_elapsed_sec else 0.0,
            "ego_rows_per_sec_wall": ego_rows / wall_elapsed_sec if wall_elapsed_sec else 0.0,
            "env_steps_per_sec_measured_loop_estimate": (
                env_steps / measured_loop_wall_estimate if measured_loop_wall_estimate else 0.0
            ),
            "ego_rows_per_sec_measured_loop_estimate": (
                ego_rows / measured_loop_wall_estimate if measured_loop_wall_estimate else 0.0
            ),
            "env_steps_per_sec_serialized_shard_sum": (
                env_steps / serialized_loop_sum if serialized_loop_sum else 0.0
            ),
        },
        "latency_sec": {
            "estimated_sync_actor_step": _latency_summary(sync_latency),
            "all_shard_actor_steps": _latency_summary(
                value
                for result in shard_results
                for value in result["actor_step_latency_sec"]
            ),
        },
        "bucket_pct_serialized_shard_loop": bucket_pct_serialized,
        "checksum": checksum,
        "shards": shard_results,
    }
    result["bottleneck_summary"] = _bottleneck_summary(result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    modes = list(dict.fromkeys(args.modes))
    results = [
        _run_mode(
            mode,
            batch=args.batch,
            workers=args.workers,
            steps=args.steps,
            warmup=args.warmup,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            action_repeat=args.action_repeat,
            max_ticks=args.max_ticks,
        )
        for mode in modes
    ]
    serial_rate = next(
        (
            result["rates"]["env_steps_per_sec_measured_loop_estimate"]
            for result in results
            if result["mode"] == "serial"
        ),
        None,
    )
    for result in results:
        if serial_rate and serial_rate > 0:
            rate = result["rates"]["env_steps_per_sec_measured_loop_estimate"]
            speedup = rate / serial_rate
            result["speedup_vs_serial_measured_loop_estimate"] = speedup
            result["parallel_efficiency_vs_serial"] = speedup / result["actual_shards"]
        else:
            result["speedup_vs_serial_measured_loop_estimate"] = None
            result["parallel_efficiency_vs_serial"] = None

    return {
        "schema_version": "curvyzero_selfplay_parallel_bridge_benchmark/v0",
        "benchmark_id": BENCHMARK_ID,
        "source_fidelity_claim": SOURCE_FIDELITY_CLAIM,
        "trust_level": (
            "Local timing scout only. It uses the toy object env, synthetic NumPy "
            "policy work, array replay staging, and coarse shard parallelism. It is "
            "not source fidelity, not MCTS, not Modal, and not production throughput."
        ),
        "command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "workload": {
            "batch": args.batch,
            "steps": args.steps,
            "warmup": args.warmup,
            "workers": args.workers,
            "hidden_dim": args.hidden_dim,
            "action_repeat": args.action_repeat,
            "max_ticks": args.max_ticks,
            "modes": modes,
            "env": "curvyzero.env.CurvyTronEnv toy v0",
            "policy": "synthetic NumPy tanh MLP logits, one call per shard tick",
            "parallelism_scope": (
                "coarse actor shards; no per-step IPC, no central inference queue, "
                "and no shared MCTS batch"
            ),
        },
        "source": _git_manifest(),
        "runtime": _runtime_manifest(),
        "timer_schema": TIMER_DEFINITIONS,
        "results": results,
    }


def print_plain(summary: dict[str, Any]) -> None:
    workload = summary["workload"]
    print(f"benchmark={summary['benchmark_id']}")
    print(f"source_fidelity_claim={summary['source_fidelity_claim']}")
    print(f"trust_level={summary['trust_level']}")
    print(
        "profile="
        f"batch:{workload['batch']} steps:{workload['steps']} warmup:{workload['warmup']} "
        f"workers:{workload['workers']} hidden:{workload['hidden_dim']} "
        f"modes:{','.join(workload['modes'])}"
    )
    for result in summary["results"]:
        counts = result["counts"]
        rates = result["rates"]
        latency = result["latency_sec"]["estimated_sync_actor_step"]
        buckets = result["bucket_pct_serialized_shard_loop"]
        bottleneck = result["bottleneck_summary"]
        top3 = ",".join(
            f"{item['bucket']}:{item['pct_serialized_shard_loop']:.1f}"
            for item in bottleneck["ranked_buckets"][:3]
        )
        speedup = result["speedup_vs_serial_measured_loop_estimate"]
        efficiency = result["parallel_efficiency_vs_serial"]
        speedup_text = "n/a" if speedup is None else f"{speedup:.3f}"
        efficiency_text = "n/a" if efficiency is None else f"{efficiency:.3f}"
        print(
            "mode="
            f"{result['mode']} shards:{result['actual_shards']} "
            f"shard_env_counts:{result['shard_env_counts']} "
            f"wall_elapsed_sec:{result['wall_elapsed_sec']:.6f} "
            f"measured_loop_wall_est_sec:{result['measured_loop_wall_estimate_sec']:.6f} "
            f"serialized_shard_loop_sum_sec:{result['serialized_shard_loop_sum_sec']:.6f} "
            f"env_steps:{counts['env_steps']} ego_rows:{counts['ego_rows']} "
            f"env_steps_per_sec_wall:{rates['env_steps_per_sec_wall']:.1f} "
            f"env_steps_per_sec_measured_loop_est:"
            f"{rates['env_steps_per_sec_measured_loop_estimate']:.1f} "
            f"speedup_vs_serial:{speedup_text} efficiency:{efficiency_text} "
            f"actor_step_p50_ms:{latency['p50'] * 1000.0:.3f} "
            f"actor_step_p95_ms:{latency['p95'] * 1000.0:.3f} "
            f"actor_step_p99_ms:{latency['p99'] * 1000.0:.3f} "
            f"env_step_pct_serialized:{buckets['env_step']:.1f} "
            f"policy_pct_serialized:{buckets['policy_batch']:.1f} "
            f"replay_pct_serialized:{buckets['replay_stage']:.1f} "
            f"checksum:{result['checksum']:.6f}"
        )
        print(
            "bottleneck="
            f"mode:{result['mode']} "
            f"top_bucket:{bottleneck['top_bucket']} "
            f"top_pct_serialized:{bottleneck['top_bucket_pct_serialized']:.1f} "
            f"top3:{top3} "
            f"env_pct_serialized:{bottleneck['env_step_pct_serialized']:.1f} "
            f"non_env_pct_serialized:{bottleneck['non_env_step_pct_serialized']:.1f} "
            f"policy_to_env_time_ratio:{bottleneck['policy_to_env_time_ratio']:.3f} "
            f"actor_step_p95_ms:{bottleneck['estimated_sync_actor_step_p95_ms']:.3f} "
            f"actor_step_p99_ms:{bottleneck['estimated_sync_actor_step_p99_ms']:.3f} "
            "scope:toy_object_env_throughput_and_latency"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=_positive_int, default=128)
    parser.add_argument("--steps", type=_positive_int, default=200)
    parser.add_argument("--warmup", type=_nonnegative_int, default=20)
    parser.add_argument("--workers", type=_positive_int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=_positive_int, default=64)
    parser.add_argument("--action-repeat", type=_positive_int, default=1)
    parser.add_argument("--max-ticks", type=_positive_int, default=2_000)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("serial", "serial-sharded", "thread", "process"),
        default=("serial", "serial-sharded", "thread", "process"),
    )
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    args = parser.parse_args()

    summary = run(args)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)


if __name__ == "__main__":
    main()
