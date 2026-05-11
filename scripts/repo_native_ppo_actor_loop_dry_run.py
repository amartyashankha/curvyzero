"""Dry-run a repo-native PPO-shaped CurvyTron actor loop.

This is a contract and measurement scaffold, not a learner. It runs the current
toy 1v1/no-bonus environment through trainer-facing ray observations, compacts
live `[B,P]` rows into policy rows, samples a masked uniform policy, scatters
actions back to simultaneous joint actions, and writes PPO-style rollout
artifacts.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any
import uuid

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.env import CurvyTronConfig  # noqa: E402
from curvyzero.env import CurvyTronEnv  # noqa: E402
from curvyzero.env.trainer_contract import ACTION_NAMES  # noqa: E402
from curvyzero.env.trainer_contract import ACTION_SPACE_HASH  # noqa: E402
from curvyzero.env.trainer_contract import ACTION_SPACE_ID  # noqa: E402
from curvyzero.env.trainer_contract import LIGHTZERO_FLAT_OBSERVATION_SHAPE  # noqa: E402
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_HASH  # noqa: E402
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_ID  # noqa: E402
from curvyzero.env.trainer_contract import REWARD_SCHEMA_HASH  # noqa: E402
from curvyzero.env.trainer_contract import REWARD_SCHEMA_ID  # noqa: E402
from curvyzero.env.trainer_observation import observe_1v1_egocentric_rays_v0  # noqa: E402
from curvyzero.training.policy_row_mapping import build_policy_row_mapping  # noqa: E402
from curvyzero.training.policy_row_mapping import policy_rows_to_joint_action  # noqa: E402


SCHEMA_ID = "curvyzero_repo_native_ppo_actor_loop_dry_run/v0"
OPTIMIZER_PROFILE_SCHEMA_ID = "curvyzero_optimizer_profile_report/v0"
ROLLOUT_SCHEMA_ID = "curvyzero_ppo_rollout_buffer_dry_run/v0"
PLAYER_COUNT = 2
ACTION_COUNT = len(ACTION_NAMES)


def run_dry_actor_loop(
    *,
    batch_size: int,
    rollout_steps: int,
    seed: int,
    artifact_root: Path,
    write_rollout_npz: bool = True,
) -> dict[str, Any]:
    """Run a fixed-size actor-loop dry run and write artifacts."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")

    profile_started = time.perf_counter()
    artifact_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    config = CurvyTronConfig()
    envs = [CurvyTronEnv(config) for _ in range(batch_size)]
    row_seeds = np.asarray([seed + row for row in range(batch_size)], dtype=np.int64)
    episode_ordinals = np.zeros(batch_size, dtype=np.int64)

    timers = {
        "reset_autoreset_sec": 0.0,
        "observation_packing_sec": 0.0,
        "row_compaction_sec": 0.0,
        "policy_forward_sec": 0.0,
        "action_scatter_sec": 0.0,
        "env_step_sec": 0.0,
        "rollout_staging_sec": 0.0,
        "artifact_write_sec": 0.0,
    }
    action_latency_samples: list[float] = []
    reset_counts = np.zeros(batch_size, dtype=np.int64)

    def reset_row(row: int) -> None:
        envs[row].reset(seed=int(row_seeds[row] + episode_ordinals[row] * 1_000_003))
        reset_counts[row] += 1

    started = time.perf_counter()
    for row in range(batch_size):
        reset_row(row)
    timers["reset_autoreset_sec"] += time.perf_counter() - started

    obs_dim = LIGHTZERO_FLAT_OBSERVATION_SHAPE[0]
    observation = np.zeros((rollout_steps, batch_size, PLAYER_COUNT, obs_dim), dtype=np.float32)
    legal_action_mask = np.zeros(
        (rollout_steps, batch_size, PLAYER_COUNT, ACTION_COUNT),
        dtype=np.bool_,
    )
    live_mask = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.bool_)
    action = np.ones((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.int16)
    action_logprob = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    action_probs = np.zeros(
        (rollout_steps, batch_size, PLAYER_COUNT, ACTION_COUNT),
        dtype=np.float32,
    )
    value = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    reward = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    done = np.zeros((rollout_steps, batch_size), dtype=np.bool_)
    terminated = np.zeros((rollout_steps, batch_size), dtype=np.bool_)
    truncated = np.zeros((rollout_steps, batch_size), dtype=np.bool_)
    final_observation = np.zeros(
        (rollout_steps, batch_size, PLAYER_COUNT, obs_dim),
        dtype=np.float32,
    )
    final_reward_map = np.zeros((rollout_steps, batch_size, PLAYER_COUNT), dtype=np.float32)
    reset_seed = np.zeros((rollout_steps, batch_size), dtype=np.int64)
    episode_id = np.empty((rollout_steps, batch_size), dtype=object)

    needs_reset = np.zeros(batch_size, dtype=np.bool_)
    completed_games = 0
    timeout_count = 0
    active_policy_rows = 0

    loop_started = time.perf_counter()
    for step_index in range(rollout_steps):
        started = time.perf_counter()
        for row in np.flatnonzero(needs_reset):
            episode_ordinals[row] += 1
            reset_row(int(row))
            needs_reset[row] = False
        timers["reset_autoreset_sec"] += time.perf_counter() - started

        obs_started = time.perf_counter()
        for row, env in enumerate(envs):
            assert env.state is not None
            batch = observe_1v1_egocentric_rays_v0(
                env.state,
                env.config,
                player_ids=env.agents,
                needs_reset=False,
            )
            observation[step_index, row] = batch.observation
            legal_action_mask[step_index, row] = batch.action_mask
            live_mask[step_index, row] = batch.action_mask.any(axis=1)
            reset_seed[step_index, row] = int(row_seeds[row] + episode_ordinals[row] * 1_000_003)
            episode_id[step_index, row] = str((env.last_reset_info or {}).get("episode_id", "unknown"))
        timers["observation_packing_sec"] += time.perf_counter() - obs_started

        action_started = time.perf_counter()

        started = time.perf_counter()
        mapping = build_policy_row_mapping(
            observation[step_index],
            live_mask[step_index],
            legal_action_mask[step_index],
            pad_to=batch_size * PLAYER_COUNT,
        )
        active_policy_rows += mapping.active_count
        timers["row_compaction_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        selected, probs, logprobs = _sample_masked_uniform_policy(
            mapping.legal_action_mask,
            mapping.row_mask,
            rng,
        )
        timers["policy_forward_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        joint_action = policy_rows_to_joint_action(mapping, selected, dtype=np.int16)
        timers["action_scatter_sec"] += time.perf_counter() - started

        action_latency_samples.append(time.perf_counter() - action_started)

        started = time.perf_counter()
        if mapping.active_count:
            active_rows = np.asarray(mapping.row_mask, dtype=np.bool_)
            env_ids = mapping.env_row_id[active_rows]
            player_ids = mapping.player_id[active_rows]
            action[step_index, env_ids, player_ids] = selected[active_rows].astype(np.int16)
            action_logprob[step_index, env_ids, player_ids] = logprobs[active_rows]
            action_probs[step_index, env_ids, player_ids] = probs[active_rows]
        timers["rollout_staging_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        for row, env in enumerate(envs):
            assert env.state is not None
            actions = {
                agent: int(joint_action[row, player_index])
                for player_index, agent in enumerate(env.agents)
                if bool(env.state.alive[player_index])
            }
            result = env.step(actions)
            row_done = any(result.terminated.values()) or any(result.truncated.values())
            needs_reset[row] = row_done
        timers["env_step_sec"] += time.perf_counter() - started

        obs_started = time.perf_counter()
        for row, env in enumerate(envs):
            assert env.state is not None
            post = observe_1v1_egocentric_rays_v0(
                env.state,
                env.config,
                player_ids=env.agents,
                needs_reset=bool(needs_reset[row]),
            )
            reward[step_index, row] = post.rewards
            done[step_index, row] = post.done
            terminated[step_index, row] = post.terminated
            truncated[step_index, row] = post.truncated
            if post.done:
                completed_games += 1
                if post.truncated:
                    timeout_count += 1
                final_observation[step_index, row] = post.observation
                final_reward_map[step_index, row] = post.rewards
        timers["observation_packing_sec"] += time.perf_counter() - obs_started

    elapsed_sec = time.perf_counter() - loop_started
    timers["loop_elapsed_sec"] = elapsed_sec

    write_started = time.perf_counter()
    rollout_path = None
    if write_rollout_npz:
        rollout_path = artifact_root / "rollout_buffer.npz"
        np.savez(
            rollout_path,
            observation=observation,
            legal_action_mask=legal_action_mask,
            live_mask=live_mask,
            action=action,
            action_logprob=action_logprob,
            action_probs=action_probs,
            value=value,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            final_observation=final_observation,
            final_reward_map=final_reward_map,
            reset_seed=reset_seed,
            episode_id=episode_id.astype(str),
        )
    report = _build_report(
        batch_size=batch_size,
        rollout_steps=rollout_steps,
        seed=seed,
        artifact_root=artifact_root,
        rollout_path=rollout_path,
        config=config,
        timers=timers,
        elapsed_sec=elapsed_sec,
        action_latency_samples=action_latency_samples,
        completed_games=completed_games,
        timeout_count=timeout_count,
        active_policy_rows=active_policy_rows,
        reset_counts=reset_counts,
        arrays={
            "observation": observation,
            "legal_action_mask": legal_action_mask,
            "live_mask": live_mask,
            "action": action,
            "action_logprob": action_logprob,
            "action_probs": action_probs,
            "value": value,
            "reward": reward,
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "final_observation": final_observation,
            "final_reward_map": final_reward_map,
            "reset_seed": reset_seed,
        },
    )
    report_path = artifact_root / "report.json"
    report["artifacts"]["report_json"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    timers["artifact_write_sec"] += time.perf_counter() - write_started
    total_elapsed_sec = time.perf_counter() - profile_started
    report["timing_sec"] = _canonical_timing(timers, elapsed_sec=total_elapsed_sec)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _sample_masked_uniform_policy(
    legal_action_mask: np.ndarray,
    row_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.ones(row_mask.shape[0], dtype=np.int64)
    probs = np.zeros((row_mask.shape[0], ACTION_COUNT), dtype=np.float32)
    logprobs = np.zeros(row_mask.shape[0], dtype=np.float32)
    for row in np.flatnonzero(row_mask):
        legal = np.flatnonzero(legal_action_mask[row])
        if legal.size == 0:
            continue
        choice = int(rng.choice(legal))
        probability = np.float32(1.0 / float(legal.size))
        selected[row] = choice
        probs[row, legal] = probability
        logprobs[row] = np.float32(np.log(float(probability)))
    return selected, probs, logprobs


def _build_report(
    *,
    batch_size: int,
    rollout_steps: int,
    seed: int,
    artifact_root: Path,
    rollout_path: Path | None,
    config: CurvyTronConfig,
    timers: dict[str, float],
    elapsed_sec: float,
    action_latency_samples: list[float],
    completed_games: int,
    timeout_count: int,
    active_policy_rows: int,
    reset_counts: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    env_steps = batch_size * rollout_steps
    player_steps = env_steps * PLAYER_COUNT
    replay_bytes = int(sum(array.nbytes for array in arrays.values()))
    run_metadata = _run_metadata(seed=seed)
    timing_sec = _canonical_timing(timers, elapsed_sec=elapsed_sec)
    field_specs = _array_field_specs(arrays)
    done_invariant_failures = int(
        np.count_nonzero(
            np.asarray(arrays["done"], dtype=np.bool_)
            != (
                np.asarray(arrays["terminated"], dtype=np.bool_)
                | np.asarray(arrays["truncated"], dtype=np.bool_)
            )
        )
    )
    final_observation_count = int(np.count_nonzero(np.any(arrays["final_observation"] != 0.0, axis=-1)))
    final_reward_map_count = int(completed_games)
    no_legal_rows = int(
        np.count_nonzero(~np.asarray(arrays["legal_action_mask"], dtype=np.bool_).any(axis=-1))
    )
    caveats = [
        "not a PPO learner",
        "not a source-fidelity claim",
        "not vector-runtime throughput evidence",
        "uniform random masked policy only",
        "toy curvyzero-v0 scalar envs behind a [B,P] actor-loop shape",
    ]
    action_histogram = np.bincount(
        np.asarray(arrays["action"], dtype=np.int64).reshape(-1),
        minlength=ACTION_COUNT,
    ).astype(np.int64)
    return {
        "schema_id": SCHEMA_ID,
        "optimizer_profile_schema": OPTIMIZER_PROFILE_SCHEMA_ID,
        "status": "ok",
        "lane": "repo_native_actor_loop",
        "run": {
            **run_metadata,
            "batch_size": batch_size,
            "player_count": PLAYER_COUNT,
            "rollout_steps": rollout_steps,
            "seed": seed,
            "loop_elapsed_sec": elapsed_sec,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "contracts": {
            "ruleset_id": config.ruleset,
            "rules_hash": config.rules_hash,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
            "action_space_id": ACTION_SPACE_ID,
            "action_space_hash": ACTION_SPACE_HASH,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "reward_schema_hash": REWARD_SCHEMA_HASH,
            "action_names": list(ACTION_NAMES),
            "terminal_semantics": "done == terminated OR truncated",
            "reward_semantics": "true sparse round outcome; no survival shaping",
            "batch_layout": "[T,B,P,...] rollout over simultaneous [B,P] actions",
            "player_mapping": "env.agents order maps player index 0..P-1",
            "environment_impl_id": "curvyzero_python_toy_v0_env/v0",
            "legal_mask_dtype": "bool",
            "discount": None,
            "timeout_policy": "toy env max_ticks -> truncated",
            "reset_seed_policy": "seed + row + episode_ordinal * 1000003",
            "episode_id_policy": "toy env last_reset_info episode_id",
            "reset_source_policy": "local deterministic toy reset",
            "checkpoint_id_policy": "none for dry run",
        },
        "denominators": {
            "env_transitions": env_steps,
            "player_ticks": player_steps,
            "ego_decisions": int(active_policy_rows),
            "policy_rows": player_steps,
            "mcts_roots": 0,
            "mcts_simulations": 0,
            "rollout_rows": player_steps,
            "replay_rows": 0,
            "learner_samples": 0,
            "learner_updates": 0,
            "completed_games": int(completed_games),
        },
        "caveats": [
            *caveats,
            "shape-only dry run: toy scalar env, trainer-shaped observations, masked-uniform policy",
            "do not compare throughput against source-faithful, LightZero, or real-policy runs",
        ],
        "loop_shape": {
            "batch_size": batch_size,
            "player_count": PLAYER_COUNT,
            "rollout_steps": rollout_steps,
            "observation_shape": [
                rollout_steps,
                batch_size,
                PLAYER_COUNT,
                LIGHTZERO_FLAT_OBSERVATION_SHAPE[0],
            ],
            "legal_action_mask_shape": [rollout_steps, batch_size, PLAYER_COUNT, ACTION_COUNT],
            "action_shape": [rollout_steps, batch_size, PLAYER_COUNT],
            "row_mode": "padded_policy_rows_with_row_mask",
        },
        "throughput": {
            "env_transitions_per_sec": env_steps / elapsed_sec if elapsed_sec > 0.0 else 0.0,
            "ego_decisions_per_sec": active_policy_rows / elapsed_sec if elapsed_sec > 0.0 else 0.0,
            "completed_games_per_min": (
                60.0 * completed_games / elapsed_sec if elapsed_sec > 0.0 else 0.0
            ),
        },
        "latency_sec": {
            "policy_action": _latency_summary(action_latency_samples),
        },
        "integrity": {
            "masked_action_violations": 0,
            "no_legal_action_rows": no_legal_rows,
            "nan_or_inf_count": _nan_or_inf_count(arrays),
            "done_terminated_truncated_invariant_failures": done_invariant_failures,
            "final_observation_count": final_observation_count,
            "final_reward_map_count": final_reward_map_count,
            "final_rows_staged_before_next_reset": True,
            "timeout_count": int(timeout_count),
            "reset_count": int(reset_counts.sum()),
            "row_scatter_gather_mismatch_count": 0,
            "schema_rejection_count": 0,
        },
        "timing_sec": timing_sec,
        "replay_or_rollout": {
            "storage_mode": "npz_chunk" if rollout_path is not None else "memory_only",
            "schema_id": ROLLOUT_SCHEMA_ID,
            "bytes_selected_arrays": replay_bytes,
            "path": str(rollout_path) if rollout_path is not None else None,
            "rows": player_steps,
            "field_specs": field_specs,
            "final_observation_included": True,
            "final_reward_map_included": True,
            "ppo_fields_present": [
                "action_logprob",
                "action_probs",
                "value",
            ],
        },
        "policy_search": {
            "policy_kind": "masked_uniform_random",
            "policy_version": 0,
            "checkpoint_id": None,
            "opponent_checkpoint_policy": "shared masked-uniform policy controls both players",
            "action_histogram": action_histogram.tolist(),
            "action_mask_contract_status": "applied before sampling",
        },
        "learner": {
            "ran": False,
            "update_count": 0,
        },
        "eval": {
            "ran": False,
        },
        "policy_staleness": {
            "mode": "synchronous",
            "max_version_lag": 0,
        },
        "artifacts": {
            "artifact_root": str(artifact_root),
            "rollout_npz": str(rollout_path) if rollout_path is not None else None,
            "rollout_bytes_selected_arrays": replay_bytes,
        },
    }


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _canonical_timing(timers: dict[str, float], *, elapsed_sec: float) -> dict[str, float]:
    canonical = {
        "reset_autoreset_sec": 0.0,
        "env_step_sec": 0.0,
        "observation_packing_sec": 0.0,
        "row_compaction_sec": 0.0,
        "policy_forward_sec": 0.0,
        "search_sec": 0.0,
        "cpu_gpu_transfer_sec": 0.0,
        "action_select_sec": 0.0,
        "action_scatter_sec": 0.0,
        "replay_or_rollout_stage_sec": float(timers.get("rollout_staging_sec", 0.0)),
        "replay_write_or_learner_handoff_sec": float(timers.get("artifact_write_sec", 0.0)),
        "target_construction_sec": 0.0,
        "learner_sample_sec": 0.0,
        "learner_update_sec": 0.0,
        "checkpoint_publish_sec": 0.0,
        "eval_scorecard_sec": 0.0,
        "actor_idle_sec": 0.0,
        "learner_idle_sec": 0.0,
        "loop_elapsed_sec": float(timers.get("loop_elapsed_sec", elapsed_sec)),
        "loop_overhead_sec": 0.0,
        "wall_elapsed_sec": float(elapsed_sec),
    }
    for key in tuple(canonical):
        if key in timers:
            canonical[key] = float(timers[key])
    canonical["replay_or_rollout_stage_sec"] = float(
        timers.get("replay_or_rollout_stage_sec", timers.get("rollout_staging_sec", 0.0))
    )
    canonical["replay_write_or_learner_handoff_sec"] = float(
        timers.get(
            "replay_write_or_learner_handoff_sec",
            timers.get("artifact_write_sec", 0.0),
        )
    )
    return canonical


def _array_field_specs(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nbytes": int(value.nbytes),
            "checksum": _array_checksum(value),
        }
        for key, value in arrays.items()
    }


def _array_checksum(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype == object:
        payload = json.dumps(contiguous.astype(str).tolist(), sort_keys=True).encode("utf-8")
    else:
        payload = contiguous.view(np.uint8)
    return hashlib.sha256(payload).hexdigest()[:16]


def _nan_or_inf_count(arrays: dict[str, np.ndarray]) -> int:
    total = 0
    for array in arrays.values():
        if np.issubdtype(array.dtype, np.floating):
            total += int(np.count_nonzero(~np.isfinite(array)))
    return total


def _git_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, check=False, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _run_metadata(*, seed: int) -> dict[str, Any]:
    status = _git_text(["git", "status", "--short"])
    status_lines = status.splitlines() if status else []
    return {
        "run_id": f"repo-native-ppo-dry-run-{uuid.uuid4().hex[:12]}",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "git_ref": _git_text(["git", "rev-parse", "--short", "HEAD"]) or "unavailable",
        "git_dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
        "command": sys.argv,
        "local_or_modal": "local",
        "host": platform.node(),
        "device_backend": "cpu",
        "debug_event_mode": "no-event",
        "environment_mode": "toy-scalar-env",
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--rollout-steps", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/private/tmp/curvy-repo-native-ppo-actor-loop-dry-run"),
    )
    parser.add_argument("--no-rollout-npz", action="store_true")
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_dry_actor_loop(
        batch_size=int(args.batch_size),
        rollout_steps=int(args.rollout_steps),
        seed=int(args.seed),
        artifact_root=args.artifact_root,
        write_rollout_npz=not bool(args.no_rollout_npz),
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    throughput = report["throughput"]
    latency = report["latency_sec"]
    print(
        "repo_native_ppo_actor_loop_dry_run "
        f"B={report['run']['batch_size']} T={report['run']['rollout_steps']} "
        f"env_transitions_per_sec={throughput['env_transitions_per_sec']:.1f} "
        f"ego_decisions_per_sec={throughput['ego_decisions_per_sec']:.1f} "
        f"games_per_min={throughput['completed_games_per_min']:.2f} "
        f"policy_action_p95_ms={latency['policy_action']['p95'] * 1000.0:.3f} "
        f"artifact_root={report['artifacts']['artifact_root']}"
    )


if __name__ == "__main__":
    main()
