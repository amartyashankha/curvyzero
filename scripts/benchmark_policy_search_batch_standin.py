"""CPU stand-in benchmark for policy/search batching and array movement.

This script is deliberately plain. It does not import JAX or Mctx, it does not
run a real MuZero search, and it does not touch the CurvyTron simulator. It
answers one narrower question: for fixed self-play batch shapes, what is the
local CPU cost of packing ego rows, running repeated recurrent-model-like
matrix work, unpacking actions, and copying search targets into arrays?
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np


BENCHMARK_ID = "numpy_policy_search_batch_standin_v0"
SOURCE_FIDELITY_CLAIM = (
    "none; CPU NumPy stand-in for fixed policy/search batch shapes and array copies"
)

TIMER_DEFINITIONS = {
    "policy_row_pack": (
        "Copy [B_env, P, obs_dim] observations into padded fixed [B_policy, obs_dim] "
        "ego rows and copy active-row metadata."
    ),
    "array_copy_in": (
        "Optional NumPy host array copy before model/search work. This is not a GPU "
        "transfer or PCIe measurement."
    ),
    "root_model": (
        "Synthetic representation plus prediction heads for one root policy/value batch."
    ),
    "recurrent_search_loop": (
        "Repeated synthetic dynamics/prediction calls and visit-count bookkeeping. "
        "This is shaped like batched search work but is not Mctx/MCTS."
    ),
    "search_target_build": (
        "Normalize fake visit counts into action weights and masked root values."
    ),
    "action_unmap": "Map selected ego-row actions back to [B_env, P] joint action arrays.",
    "array_copy_out": (
        "Optional NumPy host array copy for action weights, root values, and selected "
        "actions into replay-like staging arrays."
    ),
    "loop_overhead": "Timed loop wall time not covered by the buckets above.",
}


@dataclass(frozen=True)
class Profile:
    env_batch: int
    players: int
    obs_dim: int
    hidden_dim: int
    action_count: int
    simulations: int
    decision_batches: int
    warmup: int
    live_fraction: float
    seed: int
    copy_mode: str


@dataclass
class BenchState:
    env_obs: np.ndarray
    alive: np.ndarray
    policy_obs: np.ndarray
    model_obs: np.ndarray
    active_rows: np.ndarray
    row_env_id: np.ndarray
    row_player_id: np.ndarray
    packed_env_id: np.ndarray
    packed_player_id: np.ndarray
    representation_w: np.ndarray
    representation_b: np.ndarray
    dynamics_w: np.ndarray
    dynamics_b: np.ndarray
    action_embed: np.ndarray
    policy_w: np.ndarray
    policy_b: np.ndarray
    value_w: np.ndarray
    reward_w: np.ndarray
    joint_action: np.ndarray
    staged_action_weights: np.ndarray
    staged_root_value: np.ndarray
    staged_action: np.ndarray
    row_index: np.ndarray
    checksum: float = 0.0


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


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in the range (0, 1]")
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


def _module_status(name: str) -> dict[str, Any]:
    available = importlib.util.find_spec(name) is not None
    version = "missing"
    if available:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return {
        "available": available,
        "version": version,
        "status": "detected only; not imported by this benchmark",
    }


def _runtime_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "optional_backends": {
            "jax": _module_status("jax"),
            "mctx": _module_status("mctx"),
            "torch": _module_status("torch"),
        },
    }


def _make_state(profile: Profile) -> BenchState:
    rng = np.random.default_rng(profile.seed)
    env_obs = rng.normal(
        0.0,
        0.5,
        size=(profile.env_batch, profile.players, profile.obs_dim),
    ).astype(np.float32)
    alive = rng.random((profile.env_batch, profile.players)) < profile.live_fraction
    if not alive.any():
        alive[0, 0] = True

    policy_rows = profile.env_batch * profile.players
    row_env_id = np.repeat(np.arange(profile.env_batch, dtype=np.int32), profile.players)
    row_player_id = np.tile(np.arange(profile.players, dtype=np.int16), profile.env_batch)

    def matrix(rows: int, cols: int, scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, size=(rows, cols)).astype(np.float32)

    return BenchState(
        env_obs=env_obs,
        alive=alive,
        policy_obs=np.empty((policy_rows, profile.obs_dim), dtype=np.float32),
        model_obs=np.empty((policy_rows, profile.obs_dim), dtype=np.float32),
        active_rows=np.empty(policy_rows, dtype=bool),
        row_env_id=row_env_id,
        row_player_id=row_player_id,
        packed_env_id=np.empty(policy_rows, dtype=np.int32),
        packed_player_id=np.empty(policy_rows, dtype=np.int16),
        representation_w=matrix(profile.obs_dim, profile.hidden_dim, 0.06),
        representation_b=rng.normal(0.0, 0.01, size=profile.hidden_dim).astype(np.float32),
        dynamics_w=matrix(profile.hidden_dim, profile.hidden_dim, 0.025),
        dynamics_b=rng.normal(0.0, 0.01, size=profile.hidden_dim).astype(np.float32),
        action_embed=matrix(profile.action_count, profile.hidden_dim, 0.08),
        policy_w=matrix(profile.hidden_dim, profile.action_count, 0.05),
        policy_b=rng.normal(0.0, 0.01, size=profile.action_count).astype(np.float32),
        value_w=rng.normal(0.0, 0.04, size=profile.hidden_dim).astype(np.float32),
        reward_w=rng.normal(0.0, 0.03, size=profile.hidden_dim).astype(np.float32),
        joint_action=np.ones((profile.env_batch, profile.players), dtype=np.int16),
        staged_action_weights=np.empty((policy_rows, profile.action_count), dtype=np.float32),
        staged_root_value=np.empty(policy_rows, dtype=np.float32),
        staged_action=np.empty(policy_rows, dtype=np.int16),
        row_index=np.arange(policy_rows),
    )


def _pack_policy_rows(state: BenchState) -> None:
    state.policy_obs[:] = state.env_obs.reshape(state.policy_obs.shape)
    state.active_rows[:] = state.alive.reshape(-1)
    state.packed_env_id[:] = state.row_env_id
    state.packed_player_id[:] = state.row_player_id


def _root_model(obs: np.ndarray, state: BenchState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.tanh(obs @ state.representation_w + state.representation_b)
    logits = hidden @ state.policy_w + state.policy_b
    value = np.tanh(hidden @ state.value_w)
    return hidden, logits, value


def _recurrent_search_loop(
    hidden: np.ndarray,
    logits: np.ndarray,
    root_value: np.ndarray,
    state: BenchState,
    profile: Profile,
) -> tuple[np.ndarray, np.ndarray]:
    visits = np.zeros((hidden.shape[0], profile.action_count), dtype=np.float32)
    value_accum = root_value.astype(np.float32, copy=True)
    current_logits = logits
    current_hidden = hidden
    for sim_index in range(profile.simulations):
        tie_break = np.float32((sim_index + 1) * 1.0e-4)
        action = np.argmax(
            current_logits + tie_break * np.arange(profile.action_count, dtype=np.float32),
            axis=1,
        )
        np.add.at(visits, (state.row_index, action), 1.0)
        action_delta = state.action_embed[action]
        current_hidden = np.tanh(
            current_hidden + action_delta + current_hidden @ state.dynamics_w + state.dynamics_b
        )
        current_logits = current_hidden @ state.policy_w + state.policy_b
        value = np.tanh(current_hidden @ state.value_w)
        reward = np.float32(0.05) * np.tanh(current_hidden @ state.reward_w)
        value_accum += reward + np.float32(0.99) * value
    return visits, value_accum


def _build_search_targets(
    visits: np.ndarray,
    value_accum: np.ndarray,
    active_rows: np.ndarray,
    profile: Profile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denom = np.float32(max(1, profile.simulations))
    action_weights = visits / denom
    selected = np.argmax(action_weights, axis=1).astype(np.int16)
    root_value = value_accum / np.float32(profile.simulations + 1)
    action_weights[~active_rows] = 0.0
    selected[~active_rows] = 1 if profile.action_count > 1 else 0
    root_value[~active_rows] = 0.0
    return action_weights, root_value.astype(np.float32, copy=False), selected


def _unmap_actions(state: BenchState, selected: np.ndarray) -> None:
    state.joint_action[:] = selected.reshape(state.joint_action.shape)
    state.joint_action[~state.alive] = 1 if state.joint_action.max(initial=0) >= 1 else 0


def _copy_out(
    state: BenchState,
    action_weights: np.ndarray,
    root_value: np.ndarray,
    selected: np.ndarray,
) -> None:
    state.staged_action_weights[:] = action_weights
    state.staged_root_value[:] = root_value
    state.staged_action[:] = selected


def _run_batches(
    profile: Profile,
    state: BenchState,
) -> tuple[dict[str, float], dict[str, int], float]:
    timers = {name: 0.0 for name in TIMER_DEFINITIONS if name != "loop_overhead"}
    counts = {
        "env_decision_batches": 0,
        "policy_rows": 0,
        "active_policy_rows": 0,
        "padded_policy_rows": 0,
        "recurrent_model_calls": 0,
        "fake_visit_updates": 0,
    }
    started_loop = time.perf_counter()
    for batch_index in range(profile.decision_batches):
        state.env_obs[:, :, 0] += np.float32(1.0e-4 * ((batch_index % 7) - 3))

        started = time.perf_counter()
        _pack_policy_rows(state)
        timers["policy_row_pack"] += time.perf_counter() - started

        started = time.perf_counter()
        if profile.copy_mode == "copy":
            state.model_obs[:] = state.policy_obs
            model_obs = state.model_obs
        else:
            model_obs = state.policy_obs
        timers["array_copy_in"] += time.perf_counter() - started

        started = time.perf_counter()
        hidden, logits, root_value = _root_model(model_obs, state)
        timers["root_model"] += time.perf_counter() - started

        started = time.perf_counter()
        visits, value_accum = _recurrent_search_loop(hidden, logits, root_value, state, profile)
        timers["recurrent_search_loop"] += time.perf_counter() - started

        started = time.perf_counter()
        action_weights, searched_value, selected = _build_search_targets(
            visits,
            value_accum,
            state.active_rows,
            profile,
        )
        timers["search_target_build"] += time.perf_counter() - started

        started = time.perf_counter()
        _unmap_actions(state, selected)
        timers["action_unmap"] += time.perf_counter() - started

        started = time.perf_counter()
        if profile.copy_mode == "copy":
            _copy_out(state, action_weights, searched_value, selected)
        else:
            state.staged_action_weights = action_weights
            state.staged_root_value = searched_value
            state.staged_action = selected
        timers["array_copy_out"] += time.perf_counter() - started

        active_count = int(state.active_rows.sum())
        row_count = int(state.active_rows.size)
        state.checksum += float(searched_value[: min(4, searched_value.size)].sum())
        counts["env_decision_batches"] += profile.env_batch
        counts["policy_rows"] += row_count
        counts["active_policy_rows"] += active_count
        counts["padded_policy_rows"] += row_count - active_count
        counts["recurrent_model_calls"] += row_count * profile.simulations
        counts["fake_visit_updates"] += row_count * profile.simulations

    elapsed = time.perf_counter() - started_loop
    measured = sum(timers.values())
    timers["loop_overhead"] = max(0.0, elapsed - measured)
    return timers, counts, elapsed


def _array_bytes(profile: Profile) -> dict[str, int]:
    policy_rows = profile.env_batch * profile.players
    obs_bytes = policy_rows * profile.obs_dim * np.dtype(np.float32).itemsize
    weights_bytes = policy_rows * profile.action_count * np.dtype(np.float32).itemsize
    value_bytes = policy_rows * np.dtype(np.float32).itemsize
    action_bytes = policy_rows * np.dtype(np.int16).itemsize
    hidden_tree_bytes = (
        policy_rows
        * (profile.simulations + 1)
        * profile.hidden_dim
        * np.dtype(np.float32).itemsize
    )
    return {
        "policy_obs_bytes": obs_bytes,
        "action_weights_bytes": weights_bytes,
        "root_value_bytes": value_bytes,
        "selected_action_bytes": action_bytes,
        "mctx_hidden_tree_bytes_lower_bound": hidden_tree_bytes,
    }


def run(profile: Profile) -> dict[str, Any]:
    if profile.warmup:
        warmup_profile = Profile(**{**asdict(profile), "decision_batches": profile.warmup, "warmup": 0})
        warmup_state = _make_state(warmup_profile)
        warmup_started = time.perf_counter()
        _run_batches(warmup_profile, warmup_state)
        warmup_sec = time.perf_counter() - warmup_started
    else:
        warmup_sec = 0.0

    setup_started = time.perf_counter()
    state = _make_state(profile)
    setup_sec = time.perf_counter() - setup_started
    timers, counts, elapsed = _run_batches(profile, state)

    policy_rows_per_batch = profile.env_batch * profile.players
    active_fraction = (
        counts["active_policy_rows"] / counts["policy_rows"] if counts["policy_rows"] else 0.0
    )
    bytes_shape = _array_bytes(profile)
    copied_bytes_per_batch = 0
    if profile.copy_mode == "copy":
        copied_bytes_per_batch = (
            bytes_shape["policy_obs_bytes"]
            + bytes_shape["action_weights_bytes"]
            + bytes_shape["root_value_bytes"]
            + bytes_shape["selected_action_bytes"]
        )

    return {
        "schema_version": "curvyzero_policy_search_batch_standin/v1",
        "benchmark_id": BENCHMARK_ID,
        "source_fidelity_claim": SOURCE_FIDELITY_CLAIM,
        "trust_level": (
            "CPU NumPy stand-in only. It measures fixed-shape batch packing, "
            "matrix-heavy recurrent-loop work, action unmapping, and NumPy copies. "
            "It is not JAX, not Mctx, not GPU, not real MCTS, and not CurvyTron gameplay."
        ),
        "command": {
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
        },
        "workload": {
            **asdict(profile),
            "policy_rows_per_batch": policy_rows_per_batch,
            "batch_connection": (
                "[B_env, P, obs_dim] env observations become padded [B_policy, obs_dim] "
                "ego rows, then selected ego actions reshape back to [B_env, P]."
            ),
            "copy_mode_meaning": {
                "copy": "copy policy input and search outputs through NumPy host arrays",
                "none": "reuse arrays where possible; still allocates intermediate NumPy results",
            },
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
            "checksum": state.checksum,
            "copied_bytes_per_decision_batch": copied_bytes_per_batch,
            "copied_bytes_total": copied_bytes_per_batch * profile.decision_batches,
        },
        "memory_shape": bytes_shape,
        "rates": {
            "decision_batches_per_sec": profile.decision_batches / elapsed if elapsed else 0.0,
            "env_decisions_per_sec": (
                counts["env_decision_batches"] / elapsed if elapsed else 0.0
            ),
            "policy_rows_per_sec": counts["policy_rows"] / elapsed if elapsed else 0.0,
            "active_policy_rows_per_sec": (
                counts["active_policy_rows"] / elapsed if elapsed else 0.0
            ),
            "recurrent_model_calls_per_sec": (
                counts["recurrent_model_calls"] / timers["recurrent_search_loop"]
                if timers["recurrent_search_loop"]
                else 0.0
            ),
            "active_fraction": active_fraction,
        },
        "hard_caveats": [
            "Mctx tree selection, backup, Gumbel sampling, JAX compile time, GPU kernels, and device transfer are absent.",
            "The repeated recurrent loop is only a local CPU proxy for batch shape and rough matrix-work scale.",
            "The hidden-tree byte estimate is a lower bound for Mctx-style embedding storage, not measured memory.",
        ],
    }


def print_plain(summary: dict[str, Any]) -> None:
    workload = summary["workload"]
    rates = summary["rates"]
    timings = summary["timings_sec"]
    counts = summary["counts"]
    runtime = summary["runtime"]
    memory = summary["memory_shape"]
    print(f"benchmark={summary['benchmark_id']}")
    print(f"source_fidelity_claim={summary['source_fidelity_claim']}")
    print(f"trust_level={summary['trust_level']}")
    print(
        "profile="
        f"Benv{workload['env_batch']}_P{workload['players']}_Rows{workload['policy_rows_per_batch']}_"
        f"obs{workload['obs_dim']}_hidden{workload['hidden_dim']}_"
        f"A{workload['action_count']}_sims{workload['simulations']}_"
        f"batches{workload['decision_batches']}_copy-{workload['copy_mode']}"
    )
    print(
        "optional_backends="
        f"jax:{runtime['optional_backends']['jax']['available']} "
        f"mctx:{runtime['optional_backends']['mctx']['available']} "
        f"torch:{runtime['optional_backends']['torch']['available']} "
        "(detected only, not imported)"
    )
    print(f"setup_sec={summary['setup_sec']:.6f}")
    print(f"warmup_sec={summary['warmup_sec']:.6f}")
    print(f"elapsed_sec={summary['elapsed_sec']:.6f}")
    print(f"env_decisions_per_sec={rates['env_decisions_per_sec']:.1f}")
    print(f"policy_rows_per_sec={rates['policy_rows_per_sec']:.1f}")
    print(f"active_policy_rows_per_sec={rates['active_policy_rows_per_sec']:.1f}")
    print(f"recurrent_model_calls_per_sec={rates['recurrent_model_calls_per_sec']:.1f}")
    print(f"active_fraction={rates['active_fraction']:.3f}")
    for bucket in TIMER_DEFINITIONS:
        print(f"{bucket}_sec={timings[bucket]:.6f}")
    print(f"policy_rows={counts['policy_rows']}")
    print(f"active_policy_rows={counts['active_policy_rows']}")
    print(f"padded_policy_rows={counts['padded_policy_rows']}")
    print(f"recurrent_model_calls={counts['recurrent_model_calls']}")
    print(f"copied_bytes_total={counts['copied_bytes_total']}")
    print(f"mctx_hidden_tree_bytes_lower_bound={memory['mctx_hidden_tree_bytes_lower_bound']}")
    print(f"checksum={counts['checksum']:.6f}")
    print("not_real=Mctx,MCTS,JAX,GPU,device_transfer,CurvyTron_gameplay")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-batch", type=_positive_int, default=256)
    parser.add_argument("--players", type=_positive_int, default=2)
    parser.add_argument("--obs-dim", type=_positive_int, default=64)
    parser.add_argument("--hidden-dim", type=_positive_int, default=64)
    parser.add_argument("--action-count", type=_positive_int, default=3)
    parser.add_argument(
        "--simulations",
        type=_positive_int,
        default=16,
        help=(
            "Scale repeated synthetic recurrent-model-like work. Use this as a "
            "calibrated stand-in knob for batch-shape and copy scouts only; it is "
            "not real MCTS, Mctx, JAX, GPU, or learned-model timing."
        ),
    )
    parser.add_argument("--decision-batches", type=_positive_int, default=20)
    parser.add_argument("--warmup", type=_nonnegative_int, default=3)
    parser.add_argument("--live-fraction", type=_unit_float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copy-mode", choices=("copy", "none"), default="copy")
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    args = parser.parse_args()

    profile = Profile(
        env_batch=args.env_batch,
        players=args.players,
        obs_dim=args.obs_dim,
        hidden_dim=args.hidden_dim,
        action_count=args.action_count,
        simulations=args.simulations,
        decision_batches=args.decision_batches,
        warmup=args.warmup,
        live_fraction=args.live_fraction,
        seed=args.seed,
        copy_mode=args.copy_mode,
    )
    summary = run(profile)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_plain(summary)


if __name__ == "__main__":
    main()
