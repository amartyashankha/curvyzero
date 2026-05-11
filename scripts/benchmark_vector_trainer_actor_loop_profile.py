"""Profile the native strict 1v1/no-bonus vector trainer actor loop.

This is optimizer plumbing evidence, not a learning-quality or full CurvyTron
fidelity claim. It measures the public ``VectorTrainerEnv1v1NoBonus`` surface:

    reset -> [B,2,106] observations/masks -> policy row mapping
      -> tiny policy/search stand-in -> joint_action[B,2]
      -> env.step -> replay-v0 recorder/chunk

The source-backed ``CurvyTronSourceEnv`` profile remains the semantic oracle for
broader source claims.
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

from curvyzero.env import VectorTrainerEnv1v1NoBonus  # noqa: E402
from curvyzero.env import vector_runtime  # noqa: E402
from curvyzero.env.trainer_contract import ACTION_NAMES  # noqa: E402
from curvyzero.env.trainer_contract import ACTION_SPACE_HASH  # noqa: E402
from curvyzero.env.trainer_contract import ACTION_SPACE_ID  # noqa: E402
from curvyzero.env.trainer_contract import LIGHTZERO_FLAT_OBSERVATION_SHAPE  # noqa: E402
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_HASH  # noqa: E402
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_ID  # noqa: E402
from curvyzero.env.trainer_contract import REWARD_SCHEMA_HASH  # noqa: E402
from curvyzero.env.trainer_contract import REWARD_SCHEMA_ID  # noqa: E402
from curvyzero.env.vector_trainer_observation import (  # noqa: E402
    observe_vector_1v1_egocentric_rays_batch_arrays_v0,
)
from curvyzero.training.policy_row_mapping import build_policy_row_mapping  # noqa: E402
from curvyzero.training.policy_row_mapping import policy_rows_to_joint_action  # noqa: E402
from curvyzero.training import replay_chunk_v0  # noqa: E402
from curvyzero.training.vector_env_replay_recorder import VectorEnvReplayRecorder  # noqa: E402


SCHEMA_ID = "curvyzero_vector_trainer_actor_loop_profile/v0"
OPTIMIZER_PROFILE_SCHEMA_ID = "curvyzero_optimizer_profile_report/v0"
PLAYER_COUNT = 2
ACTION_COUNT = len(ACTION_NAMES)
RULESET_ID = "curvyzero-vector-trainer-1v1-no-bonus-strict-profile"
RULES_HASH = replay_chunk_v0.stable_contract_hash(
    {
        "ruleset_id": RULESET_ID,
        "env_impl_id": "VectorTrainerEnv1v1NoBonus",
        "feature_flags": ("1v1", "no_bonus", "P=2"),
        "control_model": "fixed_decision_wrapper_over_elapsed_ms_source_controls",
    }
)


class TinyNumpyPolicy:
    """Small deterministic model for timing policy-shaped CPU work."""

    def __init__(self, *, obs_dim: int, hidden_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.04, size=(obs_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.wp = rng.normal(0.0, 0.04, size=(hidden_dim, ACTION_COUNT)).astype(np.float32)
        self.bp = np.zeros(ACTION_COUNT, dtype=np.float32)
        self.wv = rng.normal(0.0, 0.03, size=hidden_dim).astype(np.float32)

    def __call__(
        self,
        observation: np.ndarray,
        legal_action_mask: np.ndarray,
        row_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = np.ones(row_mask.shape[0], dtype=np.int64)
        probs = np.zeros((row_mask.shape[0], ACTION_COUNT), dtype=np.float32)
        values = np.zeros(row_mask.shape[0], dtype=np.float32)
        active = np.asarray(row_mask, dtype=np.bool_)
        if not bool(active.any()):
            return selected, probs, values

        hidden = np.tanh(observation[active] @ self.w1 + self.b1)
        logits = hidden @ self.wp + self.bp
        legal = legal_action_mask[active]
        masked_logits = np.where(legal, logits, -1.0e9)
        shifted = masked_logits - masked_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted).astype(np.float32) * legal.astype(np.float32)
        denom = exp_logits.sum(axis=1, keepdims=True)
        active_probs = np.divide(
            exp_logits,
            denom,
            out=np.zeros_like(exp_logits),
            where=denom > 0.0,
        )
        selected[active] = active_probs.argmax(axis=1).astype(np.int64)
        probs[active] = active_probs
        values[active] = np.tanh(hidden @ self.wv).astype(np.float32)
        return selected, probs, values


def run_profile(
    *,
    batch_size: int,
    rollout_steps: int,
    decision_ms: float,
    max_ticks: int,
    seed: int,
    hidden_dim: int,
    event_mode: str,
    action_mode: str,
    artifact_root: Path,
    profile_observation_phases: bool = False,
    write_replay: bool = True,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if action_mode not in ("straight", "policy"):
        raise ValueError("action_mode must be 'straight' or 'policy'")
    if event_mode not in vector_runtime.EVENT_MODES:
        raise ValueError("event_mode must be 'debug-event' or 'no-event'")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if seed > np.iinfo(np.int64).max - batch_size:
        raise ValueError("seed is too large for reset_seed range")

    profile_started = time.perf_counter()
    artifact_root.mkdir(parents=True, exist_ok=True)
    timers = {
        "reset_autoreset_sec": 0.0,
        "policy_row_mapping_sec": 0.0,
        "policy_forward_sec": 0.0,
        "action_scatter_sec": 0.0,
        "policy_output_scatter_sec": 0.0,
        "env_step_public_sec": 0.0,
        "recorder_record_step_sec": 0.0,
        "replay_build_validate_sec": 0.0,
        "replay_write_sec": 0.0,
        "replay_read_validate_sec": 0.0,
        "observation_phase_probe_sec": 0.0,
    }
    obs_dim = int(LIGHTZERO_FLAT_OBSERVATION_SHAPE[0])
    policy = TinyNumpyPolicy(obs_dim=obs_dim, hidden_dim=hidden_dim, seed=seed + 10_000)

    env = VectorTrainerEnv1v1NoBonus(
        batch_size=batch_size,
        seed=seed,
        decision_ms=decision_ms,
        max_ticks=max_ticks,
        event_mode=event_mode,
    )
    reset_seed = np.arange(seed, seed + batch_size, dtype=np.uint64)

    started = time.perf_counter()
    batch = env.reset(seed=reset_seed)
    timers["reset_autoreset_sec"] += time.perf_counter() - started

    recorder = VectorEnvReplayRecorder()
    episode_id = np.asarray(
        [f"vector-profile-row-{row}-episode-0" for row in range(batch_size)],
        dtype=str,
    )
    reset_source = np.asarray(["manual" for _ in range(batch_size)], dtype="<U6")

    active_policy_rows = 0
    completed_games = 0
    step_batches_recorded = 0
    legal_action_violations = 0
    selected_positive_weight_violations = 0
    action_latency_samples: list[float] = []
    loop_started = time.perf_counter()
    for _step_index in range(rollout_steps):
        action_started = time.perf_counter()
        started = time.perf_counter()
        live_mask = batch.action_mask.any(axis=-1) & ~batch.done[:, None]
        mapping = build_policy_row_mapping(
            batch.observation,
            live_mask,
            batch.action_mask,
            pad_to=batch_size * PLAYER_COUNT,
        )
        active_policy_rows += mapping.active_count
        timers["policy_row_mapping_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        selected, probs, values = policy(
            mapping.observations,
            mapping.legal_action_mask,
            mapping.row_mask,
        )
        timers["policy_forward_sec"] += time.perf_counter() - started

        if action_mode == "straight":
            selected, probs = _straight_action_outputs(mapping)

        started = time.perf_counter()
        joint_action = policy_rows_to_joint_action(mapping, selected, dtype=np.int16)
        timers["action_scatter_sec"] += time.perf_counter() - started
        legal_action_violations += _legal_action_violations(joint_action, batch.action_mask)
        action_latency_samples.append(time.perf_counter() - action_started)

        started = time.perf_counter()
        action_weights, root_value = _scatter_policy_outputs(mapping, probs, values)
        timers["policy_output_scatter_sec"] += time.perf_counter() - started
        selected_positive_weight_violations += _selected_positive_weight_violations(
            joint_action,
            action_weights,
            batch.action_mask,
        )

        started = time.perf_counter()
        step_batch = env.step(joint_action)
        timers["env_step_public_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        recorder.record_step(
            step_batch,
            actions=joint_action,
            action_weights=action_weights,
            root_value=root_value,
        )
        timers["recorder_record_step_sec"] += time.perf_counter() - started
        step_batches_recorded += 1
        completed_games += int(np.count_nonzero(step_batch.done))
        batch = step_batch
        if bool(step_batch.done.any()):
            break
    loop_elapsed_sec = time.perf_counter() - loop_started

    started = time.perf_counter()
    chunk = recorder.build_chunk(
        episode_id=episode_id,
        reset_seed=reset_seed,
        reset_source=reset_source,
        rules_hash=RULES_HASH,
        ruleset_id=RULESET_ID,
        producer=SCHEMA_ID,
        created_at=_utc_now(),
    )
    timers["replay_build_validate_sec"] += time.perf_counter() - started

    replay_path = artifact_root / "replay_v0_chunk.npz"
    replay_read_validated = False
    if write_replay:
        started = time.perf_counter()
        replay_chunk_v0.write_replay_chunk_v0(
            replay_path,
            arrays=chunk.arrays,
            metadata=chunk.metadata,
        )
        timers["replay_write_sec"] += time.perf_counter() - started

        started = time.perf_counter()
        replay_chunk_v0.read_replay_chunk_v0(
            replay_path,
            expected_metadata=chunk.metadata,
        )
        timers["replay_read_validate_sec"] += time.perf_counter() - started
        replay_read_validated = True

    observation_phase_timers: dict[str, float] | None = None
    if profile_observation_phases:
        observation_phase_timers = {}
        started = time.perf_counter()
        observe_vector_1v1_egocentric_rays_batch_arrays_v0(
            env.state,
            player_ids=env.player_ids,
            decision_ms=decision_ms,
            max_ticks=max_ticks,
            profile_timers=observation_phase_timers,
        )
        timers["observation_phase_probe_sec"] += time.perf_counter() - started

    total_elapsed_sec = time.perf_counter() - profile_started
    denominators = {
        "env_transitions": step_batches_recorded * batch_size,
        "player_ticks": step_batches_recorded * batch_size * PLAYER_COUNT,
        "ego_decisions": int(active_policy_rows),
        "replay_steps": step_batches_recorded,
        "completed_games": int(completed_games),
    }
    report = {
        "schema_id": SCHEMA_ID,
        "optimizer_profile_schema": OPTIMIZER_PROFILE_SCHEMA_ID,
        "status": "ok",
        "run": _run_metadata(seed=seed, event_mode=event_mode),
        "lane": "vector_trainer_actor_loop_profile",
        "contracts": {
            "environment_impl_id": "VectorTrainerEnv1v1NoBonus",
            "source_truth_oracle": "CurvyTronSourceEnv/JS oracle, not this profile",
            "feature_flags": ["strict_1v1", "no_bonus", "P=2"],
            "ruleset_id": RULESET_ID,
            "rules_hash": RULES_HASH,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
            "action_space_id": ACTION_SPACE_ID,
            "action_space_hash": ACTION_SPACE_HASH,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "reward_schema_hash": REWARD_SCHEMA_HASH,
            "event_mode": event_mode,
            "decision_ms": decision_ms,
            "action_mode": action_mode,
            "replay_semantics_caveat": (
                "recorder packs returned step batches; use as throughput "
                "plumbing until learner-aligned pre-action semantics are pinned"
            ),
        },
        "loop_shape": {
            "batch_size": batch_size,
            "player_count": PLAYER_COUNT,
            "requested_rollout_steps": rollout_steps,
            "recorded_rollout_steps": step_batches_recorded,
            "observation_shape": list(chunk.arrays["observation"].shape),
            "action_shape": list(chunk.arrays["action"].shape),
        },
        "timing_sec": {
            **timers,
            "loop_elapsed_sec": loop_elapsed_sec,
            "wall_elapsed_sec": total_elapsed_sec,
        },
        "timing_notes": {
            "env_step_public_sec": (
                "public VectorTrainerEnv.step call; includes runtime advance plus returned "
                "observation, mask, reward, done, terminal, and truncation packing"
            ),
            "policy_forward_sec": (
                "tiny NumPy policy timing; in straight action_mode it is timed but not "
                "used for action selection"
            ),
            "observation_phase_probe_sec": (
                "optional separate diagnostic over current env.state; not nested inside "
                "env_step_public_sec"
            ),
        },
        "latency_sec": {"policy_action": _latency_summary(action_latency_samples)},
        "throughput": _throughput(denominators, loop_elapsed_sec),
        "denominators": denominators,
        "integrity": {
            "done_terminated_truncated_invariant_failures": int(
                np.count_nonzero(
                    chunk.arrays["done"]
                    != np.logical_or(chunk.arrays["terminated"], chunk.arrays["truncated"])
                )
            ),
            "masked_action_violations": int(legal_action_violations),
            "selected_action_positive_weight_violations": int(
                selected_positive_weight_violations
            ),
            "nan_or_inf_count": _nan_or_inf_count(chunk.arrays),
            "terminal_batch_stopped_loop": bool(completed_games > 0),
            "replay_schema_read_validated": replay_read_validated,
        },
        "replay_or_rollout": {
            "path": str(replay_path) if write_replay else None,
            "schema_id": chunk.metadata["replay_schema_id"],
            "schema_hash": chunk.metadata["replay_schema_hash"],
            "field_specs": _field_specs(chunk.arrays),
        },
        "policy_search": {
            "policy_kind": "tiny_numpy_mlp_argmax_timed",
            "action_mode": action_mode,
            "policy_forward_used_for_action_selection": action_mode == "policy",
            "search": "none",
            "hidden_dim": hidden_dim,
        },
        "observation_phase_profile": {
            "enabled": profile_observation_phases,
            "timing_sec": dict(sorted((observation_phase_timers or {}).items())),
            "note": (
                "separate diagnostic probe over current env.state; not included "
                "inside env_step_public_sec"
            ),
        },
        "artifacts": {
            "artifact_root": str(artifact_root),
            "report_json": str(artifact_root / "profile_report.json"),
            "replay_v0_chunk": str(replay_path) if write_replay else None,
        },
        "caveats": [
            "strict 1v1/no-bonus vector trainer profile only",
            "not full CurvyTron fidelity, bonuses, broad lifecycle, 3P/4P, or visual LightZero",
            "tiny NumPy policy is for timing/plumbing only",
            "source-backed CurvyTronSourceEnv/JS remains the semantic oracle",
        ],
    }
    report_path = artifact_root / "profile_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _straight_action_outputs(mapping: Any) -> tuple[np.ndarray, np.ndarray]:
    selected = np.full(mapping.capacity, 1, dtype=np.int64)
    probs = np.zeros((mapping.capacity, mapping.action_count), dtype=np.float32)
    active = np.asarray(mapping.row_mask, dtype=bool)
    if bool(active.any()):
        legal = np.asarray(mapping.legal_action_mask, dtype=bool)
        straight_legal = legal[active, 1]
        active_selected = np.full(int(active.sum()), 1, dtype=np.int64)
        active_probs = np.zeros((int(active.sum()), mapping.action_count), dtype=np.float32)
        active_probs[:, 1] = straight_legal.astype(np.float32)
        fallback = ~straight_legal
        if bool(fallback.any()):
            first_legal = legal[active][fallback].argmax(axis=1)
            active_probs[fallback, first_legal] = 1.0
            active_selected[fallback] = first_legal.astype(np.int64, copy=False)
        selected[active] = active_selected
        probs[active] = active_probs
    return selected, probs


def _scatter_policy_outputs(
    mapping: Any,
    probs: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, player_count = mapping.source_shape
    action_weights = np.zeros((batch_size, player_count, mapping.action_count), dtype=np.float32)
    root_value = np.zeros((batch_size, player_count), dtype=np.float32)
    active = np.asarray(mapping.row_mask, dtype=bool)
    if bool(active.any()):
        env_ids = np.asarray(mapping.env_row_id, dtype=np.int64)[active]
        player_ids = np.asarray(mapping.player_id, dtype=np.int64)[active]
        action_weights[env_ids, player_ids] = probs[active]
        root_value[env_ids, player_ids] = values[active]
    return action_weights, root_value


def _throughput(denominators: dict[str, int], elapsed: float) -> dict[str, float]:
    if elapsed <= 0.0:
        return {
            "env_transitions_per_sec": 0.0,
            "ego_decisions_per_sec": 0.0,
            "completed_games_per_min": 0.0,
        }
    return {
        "env_transitions_per_sec": denominators["env_transitions"] / elapsed,
        "ego_decisions_per_sec": denominators["ego_decisions"] / elapsed,
        "completed_games_per_min": 60.0 * denominators["completed_games"] / elapsed,
    }


def _legal_action_violations(action: np.ndarray, legal_action_mask: np.ndarray) -> int:
    active = np.asarray(legal_action_mask, dtype=bool).any(axis=-1)
    if not bool(active.any()):
        return 0
    selected_legal = np.take_along_axis(legal_action_mask, action[..., None], axis=-1)[..., 0]
    return int(np.count_nonzero(active & ~selected_legal))


def _selected_positive_weight_violations(
    action: np.ndarray,
    action_weights: np.ndarray,
    legal_action_mask: np.ndarray,
) -> int:
    active = np.asarray(legal_action_mask, dtype=bool).any(axis=-1)
    if not bool(active.any()):
        return 0
    positive_weight = np.take_along_axis(action_weights > 0.0, action[..., None], axis=-1)[
        ...,
        0,
    ]
    return int(np.count_nonzero(active & ~positive_weight))


def _nan_or_inf_count(arrays: dict[str, np.ndarray]) -> int:
    total = 0
    for array in arrays.values():
        if np.issubdtype(array.dtype, np.floating):
            total += int(np.count_nonzero(~np.isfinite(array)))
    return total


def _field_specs(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "nbytes": int(array.nbytes),
            "checksum": _array_checksum(array),
        }
        for key, array in arrays.items()
    }


def _array_checksum(array: np.ndarray) -> str:
    if array.dtype.kind == "U":
        payload = json.dumps(array.tolist(), sort_keys=True).encode("utf-8")
    else:
        payload = np.ascontiguousarray(array).view(np.uint8)
    return hashlib.sha256(payload).hexdigest()[:16]


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


def _git_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, check=False, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _run_metadata(*, seed: int, event_mode: str) -> dict[str, Any]:
    status = _git_text(["git", "status", "--short"])
    status_lines = status.splitlines() if status else []
    return {
        "run_id": f"vector-trainer-profile-{uuid.uuid4().hex[:12]}",
        "created_at_utc": _utc_now(),
        "git_ref": _git_text(["git", "rev-parse", "--short", "HEAD"]) or "unavailable",
        "git_dirty": bool(status_lines),
        "status_entry_count": len(status_lines),
        "command": sys.argv,
        "local_or_modal": "local",
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device_backend": "cpu",
        "debug_event_mode": event_mode,
        "lane": "vector_trainer_actor_loop_profile",
        "seed": seed,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--rollout-steps", type=_positive_int, default=64)
    parser.add_argument("--decision-ms", type=_positive_float, default=1000.0 / 60.0)
    parser.add_argument("--max-ticks", type=_positive_int, default=10_000)
    parser.add_argument("--seed", type=_nonnegative_int, default=123)
    parser.add_argument("--hidden-dim", type=_positive_int, default=64)
    parser.add_argument(
        "--event-mode",
        choices=sorted(vector_runtime.EVENT_MODES),
        default=vector_runtime.EVENT_MODE_NONE,
    )
    parser.add_argument("--action-mode", choices=("straight", "policy"), default="straight")
    parser.add_argument("--profile-observation-phases", action="store_true")
    parser.add_argument("--no-write-replay", action="store_true")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("/private/tmp/curvy-vector-trainer-profile"),
    )
    parser.add_argument("--format", choices=("json", "plain"), default="plain")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_profile(
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        decision_ms=args.decision_ms,
        max_ticks=args.max_ticks,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        event_mode=args.event_mode,
        action_mode=args.action_mode,
        artifact_root=args.artifact_root,
        profile_observation_phases=args.profile_observation_phases,
        write_replay=not args.no_write_replay,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    timing = report["timing_sec"]
    throughput = report["throughput"]
    print(f"schema: {report['schema_id']}")
    print(f"shape: B={args.batch_size}, T={report['loop_shape']['recorded_rollout_steps']}")
    print(f"loop_elapsed_sec: {timing['loop_elapsed_sec']:.6f}")
    print(f"env_step_public_sec: {timing['env_step_public_sec']:.6f}")
    print(f"policy_row_mapping_sec: {timing['policy_row_mapping_sec']:.6f}")
    print(f"policy_forward_sec: {timing['policy_forward_sec']:.6f}")
    print(f"recorder_record_step_sec: {timing['recorder_record_step_sec']:.6f}")
    print(f"replay_build_validate_sec: {timing['replay_build_validate_sec']:.6f}")
    print(f"env_transitions_per_sec: {throughput['env_transitions_per_sec']:.1f}")
    if report["observation_phase_profile"]["enabled"]:
        phases = report["observation_phase_profile"]["timing_sec"]
        print(f"observation_phase_probe_sec: {timing['observation_phase_probe_sec']:.6f}")
        print(f"observation_phase_ray_cast_sec: {phases.get('ray_cast_sec', 0.0):.6f}")
    print(f"report_json: {report['artifacts']['report_json']}")


if __name__ == "__main__":
    main()
