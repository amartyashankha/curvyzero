"""Benchmark the current self-play bridge between env rows and policy batches.

This is an isolated timing scout. It uses the current simplified CurvyTron toy
environment plus a synthetic NumPy policy/model batch. It does not claim source
fidelity, it does not run MuZero/MCTS, and it does not add a production
``reset_many`` or ``step_many`` API.
"""

from __future__ import annotations

import argparse
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


BENCHMARK_ID = "toy_curvytron_selfplay_batch_bridge_v0"
SOURCE_FIDELITY_CLAIM = (
    "none; current toy CurvyTron object env plus synthetic NumPy policy/model batch"
)
TIMER_DEFINITIONS = {
    "obs_batch_build": (
        "Copy current per-env observation dicts into one fixed [B, P, obs_dim] array."
    ),
    "policy_batch": (
        "Synthetic batched NumPy policy/model forward over B*P ego rows. This is "
        "not MCTS and not a learned CurvyZero checkpoint."
    ),
    "action_assembly": "Convert [B, P] action ids back to the current per-env action dicts.",
    "env_step": "Sequential calls to the current single-env CurvyTronEnv.step API.",
    "autoreset": "Reset finished toy env rows so the fixed batch can keep running.",
    "replay_capture": (
        "Optional hot-loop replay staging as arrays, Python dict rows, or small JSON rows."
    ),
    "loop_overhead": "Timed loop wall time not covered by the buckets above.",
}


@dataclass(frozen=True)
class Profile:
    batch: int
    steps: int
    warmup: int
    seed: int
    hidden_dim: int
    action_repeat: int
    max_ticks: int
    replay_mode: str


@dataclass
class LoopState:
    envs: list[CurvyTronEnv]
    observations: list[dict[str, np.ndarray]]
    agents: list[str]
    weights_in: np.ndarray
    bias_hidden: np.ndarray
    weights_out: np.ndarray
    bias_out: np.ndarray
    obs_dim: int
    row_cursor: int
    replay_obs: np.ndarray | None
    replay_actions: np.ndarray | None
    replay_done: np.ndarray | None
    dict_rows: list[dict[str, object]]
    json_bytes: int
    reset_counter: int


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


def _make_loop_state(profile: Profile) -> LoopState:
    config = CurvyTronConfig(action_repeat=profile.action_repeat, max_ticks=profile.max_ticks)
    envs = [CurvyTronEnv(config) for _ in range(profile.batch)]
    observations = [env.reset(seed=profile.seed + index) for index, env in enumerate(envs)]
    agents = envs[0].agents
    first_obs = observations[0][agents[0]]
    obs_dim = int(first_obs.shape[0])

    rng = np.random.default_rng(profile.seed)
    weights_in = rng.normal(0.0, 0.05, size=(obs_dim, profile.hidden_dim)).astype(np.float32)
    bias_hidden = rng.normal(0.0, 0.01, size=(profile.hidden_dim,)).astype(np.float32)
    weights_out = rng.normal(0.0, 0.05, size=(profile.hidden_dim, 3)).astype(np.float32)
    bias_out = rng.normal(0.0, 0.01, size=(3,)).astype(np.float32)

    rows = profile.batch * len(agents) * profile.steps
    replay_obs: np.ndarray | None = None
    replay_actions: np.ndarray | None = None
    replay_done: np.ndarray | None = None
    if profile.replay_mode == "array":
        replay_obs = np.empty((rows, obs_dim), dtype=np.float32)
        replay_actions = np.empty(rows, dtype=np.int8)
        replay_done = np.empty(rows, dtype=bool)

    return LoopState(
        envs=envs,
        observations=observations,
        agents=agents,
        weights_in=weights_in,
        bias_hidden=bias_hidden,
        weights_out=weights_out,
        bias_out=bias_out,
        obs_dim=obs_dim,
        row_cursor=0,
        replay_obs=replay_obs,
        replay_actions=replay_actions,
        replay_done=replay_done,
        dict_rows=[],
        json_bytes=0,
        reset_counter=0,
    )


def _build_obs_batch(state: LoopState, profile: Profile) -> np.ndarray:
    obs_batch = np.empty((profile.batch, len(state.agents), state.obs_dim), dtype=np.float32)
    for env_index, observations in enumerate(state.observations):
        for player_index, agent in enumerate(state.agents):
            obs_batch[env_index, player_index] = observations[agent]
    return obs_batch


def _policy_actions(obs_batch: np.ndarray, state: LoopState) -> np.ndarray:
    rows = obs_batch.reshape(-1, state.obs_dim)
    hidden = np.tanh(rows @ state.weights_in + state.bias_hidden)
    logits = hidden @ state.weights_out + state.bias_out
    return np.argmax(logits, axis=1).reshape(obs_batch.shape[:2]).astype(np.int8)


def _assemble_actions(actions: np.ndarray, agents: list[str]) -> list[dict[str, int]]:
    return [
        {agent: int(actions[env_index, player_index]) for player_index, agent in enumerate(agents)}
        for env_index in range(actions.shape[0])
    ]


def _env_step(
    state: LoopState,
    profile: Profile,
    joint_actions: list[dict[str, int]],
) -> tuple[np.ndarray, int]:
    done = np.zeros(profile.batch, dtype=bool)
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


def _autoreset_done_rows(state: LoopState, done: np.ndarray, profile: Profile) -> int:
    reset_rows = np.flatnonzero(done)
    for env_index in reset_rows:
        state.reset_counter += 1
        reset_seed = profile.seed + 1_000_000 + state.reset_counter
        state.observations[int(env_index)] = state.envs[int(env_index)].reset(seed=reset_seed)
    return int(reset_rows.size)


def _capture_replay(
    state: LoopState,
    profile: Profile,
    obs_batch: np.ndarray,
    actions: np.ndarray,
    done: np.ndarray,
    step_index: int,
) -> int:
    if profile.replay_mode == "none":
        return 0

    rows = obs_batch.reshape(-1, state.obs_dim)
    flat_actions = actions.reshape(-1)
    flat_done = np.repeat(done, len(state.agents))
    row_count = rows.shape[0]

    if profile.replay_mode == "array":
        assert state.replay_obs is not None
        assert state.replay_actions is not None
        assert state.replay_done is not None
        start = state.row_cursor
        end = start + row_count
        state.replay_obs[start:end] = rows
        state.replay_actions[start:end] = flat_actions
        state.replay_done[start:end] = flat_done
        state.row_cursor = end
        return row_count

    for env_index in range(profile.batch):
        for player_index, agent in enumerate(state.agents):
            row = {
                "step": step_index,
                "env": env_index,
                "ego_agent": agent,
                "action": int(actions[env_index, player_index]),
                "done": bool(done[env_index]),
                "obs0": float(obs_batch[env_index, player_index, 0]),
                "obs1": float(obs_batch[env_index, player_index, 1]),
            }
            if profile.replay_mode == "dict":
                state.dict_rows.append(row)
            elif profile.replay_mode == "dict_json":
                encoded = json.dumps(row, separators=(",", ":"), sort_keys=True)
                state.json_bytes += len(encoded) + 1
            else:
                raise ValueError(f"unknown replay_mode {profile.replay_mode!r}")
    state.row_cursor += row_count
    return row_count


def _run_measured_loop(
    profile: Profile,
    state: LoopState,
) -> tuple[dict[str, float], dict[str, int], float]:
    timers = {name: 0.0 for name in TIMER_DEFINITIONS if name != "loop_overhead"}
    counts = {
        "env_steps": 0,
        "agent_decisions": 0,
        "replay_rows": 0,
        "terminal_env_steps": 0,
        "autoreset_rows": 0,
    }
    started_loop = time.perf_counter()
    for step_index in range(profile.steps):
        started = time.perf_counter()
        obs_batch = _build_obs_batch(state, profile)
        timers["obs_batch_build"] += time.perf_counter() - started

        started = time.perf_counter()
        actions = _policy_actions(obs_batch, state)
        timers["policy_batch"] += time.perf_counter() - started

        started = time.perf_counter()
        joint_actions = _assemble_actions(actions, state.agents)
        timers["action_assembly"] += time.perf_counter() - started

        started = time.perf_counter()
        done, terminal_count = _env_step(state, profile, joint_actions)
        timers["env_step"] += time.perf_counter() - started

        started = time.perf_counter()
        reset_count = _autoreset_done_rows(state, done, profile)
        timers["autoreset"] += time.perf_counter() - started

        started = time.perf_counter()
        replay_rows = _capture_replay(state, profile, obs_batch, actions, done, step_index)
        timers["replay_capture"] += time.perf_counter() - started

        counts["env_steps"] += profile.batch
        counts["agent_decisions"] += profile.batch * len(state.agents)
        counts["replay_rows"] += replay_rows
        counts["terminal_env_steps"] += terminal_count
        counts["autoreset_rows"] += reset_count

    elapsed = time.perf_counter() - started_loop
    measured = sum(timers.values())
    timers["loop_overhead"] = max(0.0, elapsed - measured)
    return timers, counts, elapsed


def run(profile: Profile) -> dict[str, Any]:
    if profile.warmup:
        warmup_profile = Profile(**{**asdict(profile), "steps": profile.warmup, "warmup": 0})
        warmup_state = _make_loop_state(warmup_profile)
        warmup_started = time.perf_counter()
        _run_measured_loop(warmup_profile, warmup_state)
        warmup_sec = time.perf_counter() - warmup_started
    else:
        warmup_sec = 0.0

    setup_started = time.perf_counter()
    state = _make_loop_state(profile)
    setup_sec = time.perf_counter() - setup_started
    timers, counts, elapsed = _run_measured_loop(profile, state)

    env_step_sec = timers["env_step"]
    policy_sec = timers["policy_batch"]
    replay_sec = timers["replay_capture"]
    env_steps = counts["env_steps"]
    decisions = counts["agent_decisions"]
    rows = counts["replay_rows"]
    replay_memory_bytes = 0
    if state.replay_obs is not None:
        replay_memory_bytes += int(state.replay_obs.nbytes)
    if state.replay_actions is not None:
        replay_memory_bytes += int(state.replay_actions.nbytes)
    if state.replay_done is not None:
        replay_memory_bytes += int(state.replay_done.nbytes)

    config = state.envs[0].config
    return {
        "schema_version": "curvyzero_selfplay_batch_bridge_benchmark/v1",
        "benchmark_id": BENCHMARK_ID,
        "source_fidelity_claim": SOURCE_FIDELITY_CLAIM,
        "trust_level": (
            "Bridge timing only. It measures current toy object-env stepping, "
            "observation/action packing, and a synthetic batched policy call. "
            "It is not production vectorization, not source fidelity, and not MuZero/MCTS."
        ),
        "command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "workload": {
            **asdict(profile),
            "env": "curvyzero.env.CurvyTronEnv toy v0",
            "players": len(state.agents),
            "obs_shape": [profile.batch, len(state.agents), state.obs_dim],
            "policy_batch_rows_per_step": profile.batch * len(state.agents),
            "policy_model": "synthetic NumPy tanh MLP logits, one call per env tick",
            "replay_mode_meaning": {
                "none": "no replay row staging",
                "array": "preallocated array staging",
                "dict": "small Python dict row staging",
                "dict_json": "small JSON encoding scaffold, not full production replay",
            },
        },
        "env_config": {
            "ruleset": config.ruleset,
            "rules_hash": config.rules_hash,
            "action_repeat": config.action_repeat,
            "max_ticks": config.max_ticks,
            "action_count": config.action_count,
            "width": config.width,
            "height": config.height,
        },
        "source": _git_manifest(),
        "runtime": _runtime_manifest(),
        "timer_schema": TIMER_DEFINITIONS,
        "setup_sec": setup_sec,
        "warmup_sec": warmup_sec,
        "elapsed_sec": elapsed,
        "timings_sec": timers,
        "counts": {
            **counts,
            "dict_rows_stored": len(state.dict_rows),
            "json_bytes": state.json_bytes,
            "replay_memory_bytes": replay_memory_bytes,
        },
        "rates": {
            "env_steps_per_sec_total_loop": env_steps / elapsed if elapsed else 0.0,
            "agent_decisions_per_sec_total_loop": decisions / elapsed if elapsed else 0.0,
            "env_steps_per_sec_env_step_bucket": env_steps / env_step_sec if env_step_sec else 0.0,
            "agent_rows_per_sec_policy_bucket": decisions / policy_sec if policy_sec else 0.0,
            "replay_rows_per_sec_replay_bucket": rows / replay_sec if replay_sec else 0.0,
        },
        "next_measurement_hint": (
            "Repeat this shape after a real fixture-seeded array backend exists, "
            "then replace the synthetic policy batch with the Mctx/JAX search benchmark."
        ),
    }


def print_plain(summary: dict[str, Any]) -> None:
    rates = summary["rates"]
    counts = summary["counts"]
    timings = summary["timings_sec"]
    workload = summary["workload"]
    print(f"benchmark={summary['benchmark_id']}")
    print(f"source_fidelity_claim={summary['source_fidelity_claim']}")
    print(f"trust_level={summary['trust_level']}")
    print(
        "profile="
        f"B{workload['batch']}_steps{workload['steps']}_hidden{workload['hidden_dim']}_"
        f"repeat{workload['action_repeat']}_replay-{workload['replay_mode']}"
    )
    print(f"setup_sec={summary['setup_sec']:.6f}")
    print(f"warmup_sec={summary['warmup_sec']:.6f}")
    print(f"elapsed_sec={summary['elapsed_sec']:.6f}")
    print(f"env_steps={counts['env_steps']}")
    print(f"agent_decisions={counts['agent_decisions']}")
    print(f"terminal_env_steps={counts['terminal_env_steps']}")
    print(f"autoreset_rows={counts['autoreset_rows']}")
    print(f"env_steps_per_sec_total_loop={rates['env_steps_per_sec_total_loop']:.1f}")
    print(f"agent_decisions_per_sec_total_loop={rates['agent_decisions_per_sec_total_loop']:.1f}")
    print(f"env_steps_per_sec_env_step_bucket={rates['env_steps_per_sec_env_step_bucket']:.1f}")
    print(f"agent_rows_per_sec_policy_bucket={rates['agent_rows_per_sec_policy_bucket']:.1f}")
    print(f"replay_rows_per_sec_replay_bucket={rates['replay_rows_per_sec_replay_bucket']:.1f}")
    for name in TIMER_DEFINITIONS:
        print(f"{name}_sec={timings[name]:.6f}")
    print(f"replay_rows={counts['replay_rows']}")
    print(f"dict_rows_stored={counts['dict_rows_stored']}")
    print(f"json_bytes={counts['json_bytes']}")
    print(f"replay_memory_bytes={counts['replay_memory_bytes']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=_positive_int, default=128)
    parser.add_argument("--steps", type=_positive_int, default=200)
    parser.add_argument("--warmup", type=_nonnegative_int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=_positive_int, default=64)
    parser.add_argument("--action-repeat", type=_positive_int, default=1)
    parser.add_argument("--max-ticks", type=_positive_int, default=2_000)
    parser.add_argument(
        "--replay-mode",
        choices=("none", "array", "dict", "dict_json"),
        default="array",
    )
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    args = parser.parse_args()

    profile = Profile(
        batch=args.batch,
        steps=args.steps,
        warmup=args.warmup,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        action_repeat=args.action_repeat,
        max_ticks=args.max_ticks,
        replay_mode=args.replay_mode,
    )
    summary = run(profile)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)


if __name__ == "__main__":
    main()
