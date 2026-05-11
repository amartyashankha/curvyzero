"""Evaluator for the dummy solo survival task.

This is an EVAL1 debugging harness, not a leaderboard. It evaluates simple
baselines and optional learned checkpoints on fixed seeds and writes a small
JSON table that can catch obvious survival, crash, and action-collapse
regressions.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from curvyzero.training.dummy_survival import DummyMuZeroModel
from curvyzero.training.dummy_survival import DummyPlanner
from curvyzero.training.dummy_survival import SoloTurningSurvivalEnv
from curvyzero.training.dummy_survival import SurvivalConfig
from curvyzero.training.dummy_survival import SurvivalObservation

ACTION_LABELS = ("left", "straight", "right")
UNTRAINED_PLANNER_POLICY_ID = "untrained_model_same_planner"
DEFAULT_POLICIES = ("random_uniform", "one_step_safe", UNTRAINED_PLANNER_POLICY_ID)
LEARNED_POLICY_PREFIX = "learned:"
POLICY_RNG_OFFSETS = {
    "random_uniform": 1_000_003,
    "one_step_safe": 2_000_006,
    UNTRAINED_PLANNER_POLICY_ID: 2_500_007,
}

PolicyFn = Callable[[SurvivalObservation, np.random.Generator], int]


@dataclass(frozen=True, slots=True)
class EvalEpisodeRow:
    policy_id: str
    episode_index: int
    env_seed: int
    steps: int
    terminal_reward: float
    crashed: bool
    survived: bool
    action_counts: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class EvalPolicy:
    policy_id: str
    policy_kind: str
    policy_fn: PolicyFn
    checkpoint_path: Path | None = None
    checkpoint_id: str | None = None
    checkpoint_metadata: dict[str, object] | None = None
    learned_state_count: int | None = None
    learned_dynamics_edges: int | None = None


def run_dummy_survival_eval(
    *,
    episodes: int,
    seed: int,
    output_dir: Path | None = None,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
    checkpoint_policies: tuple[str, ...] = (),
    split_id: str = "dummy_survival_monitor_v0",
    split_role: str = "monitor",
) -> dict[str, object]:
    """Evaluate dummy-survival baselines/checkpoints and optionally write artifacts."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if not policies and not checkpoint_policies:
        raise ValueError("at least one policy must be requested")

    config = SurvivalConfig()
    resolved_policies = _resolve_policies(
        policies=policies,
        checkpoint_policies=checkpoint_policies,
        config=config,
    )
    policy_ids = [policy.policy_id for policy in resolved_policies]
    env_seeds = _episode_seeds(seed=seed, episodes=episodes)
    policy_rng_seeds = {
        policy.policy_id: _policy_rng_seed(seed=seed, policy=policy, index=index)
        for index, policy in enumerate(resolved_policies)
    }

    started = time.perf_counter()
    rows: list[EvalEpisodeRow] = []
    for policy in resolved_policies:
        policy_rng = np.random.default_rng(policy_rng_seeds[policy.policy_id])
        for episode_index, env_seed in enumerate(env_seeds):
            rows.append(
                _run_episode(
                    config=config,
                    policy_id=policy.policy_id,
                    policy_fn=policy.policy_fn,
                    episode_index=episode_index,
                    env_seed=env_seed,
                    policy_rng=policy_rng,
                )
            )
    elapsed_seconds = time.perf_counter() - started

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_survival_eval",
        "note": "Fixed dummy baselines plus optional learned checkpoint policies; no ratings.",
        "seed": seed,
        "episodes_per_policy": episodes,
        "policies": policy_ids,
        "policy_specs": [_policy_summary(policy) for policy in resolved_policies],
        "checkpoint_policies": [
            _policy_summary(policy)
            for policy in resolved_policies
            if policy.policy_kind == "learned_checkpoint"
        ],
        "action_labels": list(ACTION_LABELS),
        "config": asdict(config),
        "seed_handling": {
            "env_seed_source": "np.random.default_rng(seed).integers(2**31 - 1)",
            "shared_env_seeds_across_policies": True,
            "policy_rng_seeds": policy_rng_seeds,
        },
        "eval_split": {
            "split_id": split_id,
            "split_role": split_role,
            "seed_generation": "rng_integers",
            "base_seed": seed,
            "seed_count": len(env_seeds),
            "seed_list_hash": _seed_list_hash(env_seeds),
            "paired_seat": False,
        },
        "table": [_summarize_policy(policy, rows) for policy in resolved_policies],
        "eval": {
            "total_episodes": len(rows),
            "elapsed_seconds": elapsed_seconds,
            "episodes_per_second": len(rows) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0,
        },
    }

    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, rows=rows)
    return summary


def _run_episode(
    *,
    config: SurvivalConfig,
    policy_id: str,
    policy_fn: PolicyFn,
    episode_index: int,
    env_seed: int,
    policy_rng: np.random.Generator,
) -> EvalEpisodeRow:
    env = SoloTurningSurvivalEnv(config)
    observation = env.reset(seed=env_seed)
    action_counts = [0, 0, 0]
    terminal_reward = 0.0
    crashed = False
    survived = False

    while True:
        action = policy_fn(observation, policy_rng)
        if action < 0 or action >= config.action_count:
            raise ValueError(f"policy {policy_id!r} returned invalid action {action!r}")
        action_counts[action] += 1
        step = env.step(action)
        observation = step.observation
        if step.terminated:
            terminal_reward = float(step.reward)
            crashed = bool(step.info["crashed"])
            survived = bool(step.info["survived"])
            break

    return EvalEpisodeRow(
        policy_id=policy_id,
        episode_index=episode_index,
        env_seed=env_seed,
        steps=sum(action_counts),
        terminal_reward=terminal_reward,
        crashed=crashed,
        survived=survived,
        action_counts=tuple(action_counts),
    )


def _random_uniform_policy(
    observation: SurvivalObservation, rng: np.random.Generator
) -> int:
    del observation
    return int(rng.integers(3))


def _one_step_safe_policy(
    observation: SurvivalObservation, rng: np.random.Generator
) -> int:
    del rng
    clearances = (
        observation.left_clearance,
        observation.straight_clearance,
        observation.right_clearance,
    )
    best_clearance = max(clearances)
    preferred_tie_order = (1, 0, 2)
    for action in preferred_tie_order:
        if clearances[action] == best_clearance:
            return action
    raise AssertionError("unreachable")


def _untrained_model_same_planner_policy(config: SurvivalConfig) -> PolicyFn:
    model = DummyMuZeroModel(action_count=config.action_count)
    planner = DummyPlanner(discount=config.discount)

    def policy_fn(observation: SurvivalObservation, rng: np.random.Generator) -> int:
        return planner.select_action(
            model=model,
            observation=observation,
            rng=rng,
            epsilon=0.0,
            explore_unknown=False,
        )

    return policy_fn


def _policy_by_id(policy_id: str, config: SurvivalConfig) -> PolicyFn:
    if policy_id == "random_uniform":
        return _random_uniform_policy
    if policy_id == "one_step_safe":
        return _one_step_safe_policy
    if policy_id == UNTRAINED_PLANNER_POLICY_ID:
        return _untrained_model_same_planner_policy(config)
    raise ValueError(f"unknown policy {policy_id!r}; expected one of {DEFAULT_POLICIES!r}")


def _resolve_policies(
    *,
    policies: tuple[str, ...],
    checkpoint_policies: tuple[str, ...],
    config: SurvivalConfig,
) -> list[EvalPolicy]:
    resolved = [
        EvalPolicy(
            policy_id=policy_id,
            policy_kind="baseline",
            policy_fn=_policy_by_id(policy_id, config),
        )
        for policy_id in policies
    ]
    resolved.extend(
        _learned_checkpoint_policy(spec=policy_spec, config=config)
        for policy_spec in checkpoint_policies
    )
    return resolved


def _learned_checkpoint_policy(*, spec: str, config: SurvivalConfig) -> EvalPolicy:
    if not spec.startswith(LEARNED_POLICY_PREFIX):
        raise ValueError(
            f"unknown checkpoint policy {spec!r}; expected {LEARNED_POLICY_PREFIX}<path>"
        )
    checkpoint_path = Path(spec.removeprefix(LEARNED_POLICY_PREFIX))
    model = DummyMuZeroModel.load_checkpoint(checkpoint_path)
    planner = DummyPlanner(discount=config.discount)

    def policy_fn(observation: SurvivalObservation, rng: np.random.Generator) -> int:
        return planner.select_action(
            model=model,
            observation=observation,
            rng=rng,
            epsilon=0.0,
            explore_unknown=False,
        )

    checkpoint_id = _checkpoint_id(checkpoint_path)
    return EvalPolicy(
        policy_id=f"learned:{checkpoint_id}",
        policy_kind="learned_checkpoint",
        policy_fn=policy_fn,
        checkpoint_path=checkpoint_path,
        checkpoint_id=checkpoint_id,
        checkpoint_metadata=DummyMuZeroModel.checkpoint_metadata(checkpoint_path),
        learned_state_count=len(model.q_values),
        learned_dynamics_edges=len(model.transition_counts),
    )


def _checkpoint_id(path: Path) -> str:
    if path.name == "checkpoint.npz" and path.parent.name:
        return f"{path.parent.name}/{path.stem}"
    return path.stem


def _policy_rng_seed(*, seed: int, policy: EvalPolicy, index: int) -> int:
    if policy.policy_id in POLICY_RNG_OFFSETS:
        return seed + POLICY_RNG_OFFSETS[policy.policy_id]
    return seed + 3_000_009 + index * 1_000_003


def _episode_seeds(*, seed: int, episodes: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(2**31 - 1, size=episodes)]


def _seed_list_hash(seeds: list[int]) -> str:
    payload = json.dumps(seeds, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summarize_policy(policy: EvalPolicy, rows: list[EvalEpisodeRow]) -> dict[str, object]:
    policy_id = policy.policy_id
    policy_rows = [row for row in rows if row.policy_id == policy_id]
    if not policy_rows:
        return _with_checkpoint_fields(
            policy,
            {
                "policy_id": policy_id,
                "policy_kind": policy.policy_kind,
                "episodes": 0,
                "mean_terminal_reward": 0.0,
                "survival_rate": 0.0,
                "crash_rate": 0.0,
                "mean_steps": 0.0,
                "max_steps": 0,
                "action_histogram": [0, 0, 0],
                "action_histogram_by_label": dict.fromkeys(ACTION_LABELS, 0),
            },
        )

    action_histogram = [
        sum(row.action_counts[action] for row in policy_rows)
        for action in range(len(ACTION_LABELS))
    ]
    return _with_checkpoint_fields(
        policy,
        {
            "policy_id": policy_id,
            "policy_kind": policy.policy_kind,
            "episodes": len(policy_rows),
            "mean_terminal_reward": float(np.mean([row.terminal_reward for row in policy_rows])),
            "survival_rate": float(np.mean([row.survived for row in policy_rows])),
            "crash_rate": float(np.mean([row.crashed for row in policy_rows])),
            "mean_steps": float(np.mean([row.steps for row in policy_rows])),
            "max_steps": int(max(row.steps for row in policy_rows)),
            "action_histogram": action_histogram,
            "action_histogram_by_label": dict(zip(ACTION_LABELS, action_histogram)),
        },
    )


def _policy_summary(policy: EvalPolicy) -> dict[str, object]:
    summary = {
        "policy_id": policy.policy_id,
        "policy_kind": policy.policy_kind,
    }
    return _with_checkpoint_fields(policy, summary)


def _with_checkpoint_fields(
    policy: EvalPolicy,
    summary: dict[str, object],
) -> dict[str, object]:
    if policy.policy_kind != "learned_checkpoint":
        return summary
    summary.update(
        {
            "checkpoint_path": str(policy.checkpoint_path),
            "checkpoint_id": policy.checkpoint_id,
            "checkpoint_metadata": policy.checkpoint_metadata or {},
            "learned_state_count": policy.learned_state_count,
            "learned_dynamics_edges": policy.learned_dynamics_edges,
        }
    )
    return summary


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "episodes_jsonl": output_dir / "episodes.jsonl",
    }


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    rows: list[EvalEpisodeRow],
) -> None:
    summary_path = artifacts["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    episodes_path = artifacts["episodes_jsonl"]
    with episodes_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
