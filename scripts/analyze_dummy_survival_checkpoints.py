"""Diagnose dummy-survival checkpoint degradation without retraining.

This script compares the planner-only baseline against learned checkpoints on
shared eval seeds, then surfaces where learned dynamics/Q values change the
planner's safety-prior action.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.training.dummy_survival import DummyMuZeroModel
from curvyzero.training.dummy_survival import DummyPlanner
from curvyzero.training.dummy_survival import SoloTurningSurvivalEnv
from curvyzero.training.dummy_survival import StateKey
from curvyzero.training.dummy_survival import SurvivalConfig
from curvyzero.training.dummy_survival import SurvivalObservation

ACTION_LABELS = ("left", "straight", "right")
UNTRAINED_POLICY_ID = "untrained_model_same_planner"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/local/dummy_survival_safety_planner_i8_e20_seed0_c2/checkpoints"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/local/dummy_survival_degradation_diagnosis_seed123_e20"),
    )
    args = parser.parse_args()

    summary = analyze_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def analyze_checkpoints(
    *,
    checkpoint_dir: Path,
    episodes: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    checkpoint_paths = sorted(checkpoint_dir.glob("*.npz"))
    if not checkpoint_paths:
        raise ValueError(f"no .npz checkpoints found in {checkpoint_dir}")

    config = SurvivalConfig()
    planner = DummyPlanner(discount=config.discount)
    eval_seeds = _episode_seeds(seed=seed, episodes=episodes)
    policy_specs = [(UNTRAINED_POLICY_ID, DummyMuZeroModel(action_count=config.action_count), None)]
    policy_specs.extend(
        (f"learned:{path.stem}", DummyMuZeroModel.load_checkpoint(path), path)
        for path in checkpoint_paths
    )

    all_policy_rows = []
    episode_rows = []
    override_rows = []
    traces_by_policy: dict[str, list[dict[str, Any]]] = {}
    for policy_id, model, checkpoint_path in policy_specs:
        traces = [
            _run_traced_episode(
                config=config,
                planner=planner,
                model=model,
                policy_id=policy_id,
                episode_index=episode_index,
                env_seed=env_seed,
            )
            for episode_index, env_seed in enumerate(eval_seeds)
        ]
        traces_by_policy[policy_id] = traces
        all_policy_rows.append(
            _summarize_policy(
                policy_id=policy_id,
                checkpoint_path=checkpoint_path,
                model=model,
                traces=traces,
            )
        )
        episode_rows.extend(_episode_row(trace) for trace in traces)
        for trace in traces:
            override_rows.extend(_override_rows(trace))

    early_policy_id = f"learned:{checkpoint_paths[0].stem}"
    latest_policy_id = f"learned:{checkpoint_paths[-1].stem}"
    degradation_examples = _degradation_examples(
        traces_by_policy=traces_by_policy,
        early_policy_id=early_policy_id,
        latest_policy_id=latest_policy_id,
    )
    summary: dict[str, Any] = {
        "kind": "curvyzero_dummy_survival_checkpoint_degradation_diagnosis",
        "note": "Analysis-only replay of existing checkpoints; no training or pytest.",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "seed": seed,
        "episodes": episodes,
        "eval_seeds": eval_seeds,
        "config": asdict(config),
        "policy_table": all_policy_rows,
        "degradation_examples": degradation_examples,
        "top_override_states": _top_override_states(override_rows),
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "episodes_jsonl": str(output_dir / "episodes.jsonl"),
            "overrides_jsonl": str(output_dir / "overrides.jsonl"),
        },
    }
    _write_outputs(
        output_dir=output_dir,
        summary=summary,
        episode_rows=episode_rows,
        override_rows=override_rows,
    )
    return summary


def _run_traced_episode(
    *,
    config: SurvivalConfig,
    planner: DummyPlanner,
    model: DummyMuZeroModel,
    policy_id: str,
    episode_index: int,
    env_seed: int,
) -> dict[str, Any]:
    env = SoloTurningSurvivalEnv(config)
    observation = env.reset(seed=env_seed)
    action_counts = [0, 0, 0]
    steps = []

    while True:
        decision = _decision_snapshot(
            model=model,
            config=config,
            planner=planner,
            observation=observation,
        )
        action = int(decision["selected_action"])
        action_counts[action] += 1
        step = env.step(action)
        steps.append(
            {
                **decision,
                "step_index": len(steps),
                "reward": float(step.reward),
                "terminated": bool(step.terminated),
                "crashed": bool(step.info["crashed"]),
                "survived": bool(step.info["survived"]),
            }
        )
        observation = step.observation
        if step.terminated:
            return {
                "policy_id": policy_id,
                "episode_index": episode_index,
                "env_seed": env_seed,
                "steps": len(steps),
                "terminal_reward": float(step.reward),
                "crashed": bool(step.info["crashed"]),
                "survived": bool(step.info["survived"]),
                "action_counts": action_counts,
                "trace": steps,
            }


def _decision_snapshot(
    *,
    model: DummyMuZeroModel,
    config: SurvivalConfig,
    planner: DummyPlanner,
    observation: SurvivalObservation,
) -> dict[str, Any]:
    state_key = model.representation(observation)
    clearances = (
        observation.left_clearance,
        observation.straight_clearance,
        observation.right_clearance,
    )
    safety_action = _safety_action(clearances)
    scores = []
    next_keys = []
    rewards = []
    next_values = []
    next_argmax_q = []
    for action in range(config.action_count):
        next_key, reward = model.dynamics(state_key, action)
        next_key_tuple = tuple(next_key) if next_key is not None else None
        q_values = None if next_key is None else model.q_values.get(next_key)
        next_value = float(np.max(q_values)) if q_values is not None else 0.0
        score = float(reward + planner.discount * next_value)
        if clearances[action] <= 0:
            score -= planner.safety_penalty
        scores.append(score)
        next_keys.append(next_key_tuple)
        rewards.append(float(reward))
        next_values.append(next_value)
        next_argmax_q.append(None if q_values is None else [float(value) for value in q_values])
    selected_action = max(
        range(config.action_count),
        key=lambda action: (
            scores[action],
            clearances[action],
            1 if action == 1 else 0,
            -action,
        ),
    )
    current_q = model.q_values.get(state_key)
    return {
        "state_key": tuple(state_key),
        "observation": asdict(observation),
        "clearances": clearances,
        "scores": scores,
        "selected_action": selected_action,
        "selected_label": ACTION_LABELS[selected_action],
        "safety_action": safety_action,
        "safety_label": ACTION_LABELS[safety_action],
        "selected_clearance": clearances[selected_action],
        "safety_clearance": clearances[safety_action],
        "score_delta_vs_safety": float(scores[selected_action] - scores[safety_action]),
        "override_safety_action": selected_action != safety_action,
        "lower_clearance_than_safety": clearances[selected_action] < clearances[safety_action],
        "selected_zero_clearance": clearances[selected_action] <= 0,
        "next_keys": next_keys,
        "model_rewards": rewards,
        "next_values": next_values,
        "next_argmax_q": next_argmax_q,
        "current_q": None if current_q is None else [float(value) for value in current_q],
    }


def _summarize_policy(
    *,
    policy_id: str,
    checkpoint_path: Path | None,
    model: DummyMuZeroModel,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    action_histogram = [
        sum(trace["action_counts"][action] for trace in traces)
        for action in range(len(ACTION_LABELS))
    ]
    flat_steps = [step for trace in traces for step in trace["trace"]]
    overrides = [step for step in flat_steps if step["override_safety_action"]]
    lower_clearance = [step for step in overrides if step["lower_clearance_than_safety"]]
    zero_clearance = [step for step in flat_steps if step["selected_zero_clearance"]]
    crash_traces = [trace for trace in traces if trace["crashed"]]
    return {
        "policy_id": policy_id,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "episodes": len(traces),
        "survival_rate": float(np.mean([trace["survived"] for trace in traces])),
        "crash_rate": float(np.mean([trace["crashed"] for trace in traces])),
        "mean_steps": float(np.mean([trace["steps"] for trace in traces])),
        "max_steps": int(max(trace["steps"] for trace in traces)),
        "mean_terminal_reward": float(np.mean([trace["terminal_reward"] for trace in traces])),
        "action_histogram": action_histogram,
        "action_histogram_by_label": dict(zip(ACTION_LABELS, action_histogram)),
        "override_safety_action_steps": len(overrides),
        "lower_clearance_override_steps": len(lower_clearance),
        "selected_zero_clearance_steps": len(zero_clearance),
        "crash_seeds": [trace["env_seed"] for trace in crash_traces],
        "first_crash_examples": [_crash_example(trace) for trace in crash_traces[:5]],
        "model_stats": _model_stats(model),
    }


def _model_stats(model: DummyMuZeroModel) -> dict[str, Any]:
    q_rows = list(model.q_values.values())
    q_values = np.vstack(q_rows) if q_rows else np.zeros((0, model.action_count))
    rewards = np.array(
        [
            model.reward_sums[key] / count
            for key, count in model.transition_counts.items()
            if count > 0
        ],
        dtype=np.float64,
    )
    action_edges = Counter(key[1] for key in model.transition_counts)
    terminal_edges = sum(1 for _, _, next_key in model.transition_counts if next_key is None)
    return {
        "learned_state_count": len(model.q_values),
        "learned_dynamics_edges": len(model.transition_counts),
        "terminal_dynamics_edges": terminal_edges,
        "action_dynamics_edges": {
            ACTION_LABELS[action]: int(action_edges[action])
            for action in range(model.action_count)
        },
        "q_min": None if q_values.size == 0 else float(np.min(q_values)),
        "q_max": None if q_values.size == 0 else float(np.max(q_values)),
        "q_mean": None if q_values.size == 0 else float(np.mean(q_values)),
        "q_negative_count": int(np.sum(q_values < 0.0)),
        "q_positive_count": int(np.sum(q_values > 0.0)),
        "q_zero_count": int(np.sum(q_values == 0.0)),
        "visit_total": float(sum(float(np.sum(visits)) for visits in model.visit_counts.values())),
        "mean_model_reward_min": None if rewards.size == 0 else float(np.min(rewards)),
        "mean_model_reward_max": None if rewards.size == 0 else float(np.max(rewards)),
        "negative_reward_edges": int(np.sum(rewards < 0.0)) if rewards.size else 0,
        "positive_reward_edges": int(np.sum(rewards > 0.0)) if rewards.size else 0,
    }


def _degradation_examples(
    *,
    traces_by_policy: dict[str, list[dict[str, Any]]],
    early_policy_id: str,
    latest_policy_id: str,
) -> list[dict[str, Any]]:
    untrained_by_seed = _by_seed(traces_by_policy[UNTRAINED_POLICY_ID])
    early_by_seed = _by_seed(traces_by_policy[early_policy_id])
    latest_by_seed = _by_seed(traces_by_policy[latest_policy_id])
    examples = []
    for env_seed, latest_trace in latest_by_seed.items():
        early_trace = early_by_seed[env_seed]
        untrained_trace = untrained_by_seed[env_seed]
        if latest_trace["crashed"] and early_trace["survived"] and untrained_trace["survived"]:
            examples.append(
                {
                    "env_seed": env_seed,
                    "early_policy_id": early_policy_id,
                    "latest_policy_id": latest_policy_id,
                    "untrained_steps": untrained_trace["steps"],
                    "early_steps": early_trace["steps"],
                    "latest_steps": latest_trace["steps"],
                    "latest_first_bad_override": _first_bad_override(latest_trace),
                    "latest_crash": _crash_example(latest_trace),
                }
            )
    return examples[:10]


def _first_bad_override(trace: dict[str, Any]) -> dict[str, Any] | None:
    for step in trace["trace"]:
        if step["lower_clearance_than_safety"] or step["selected_zero_clearance"]:
            return _compact_step(step)
    return None


def _crash_example(trace: dict[str, Any]) -> dict[str, Any]:
    first_bad = _first_bad_override(trace)
    return {
        "env_seed": trace["env_seed"],
        "steps": trace["steps"],
        "terminal_reward": trace["terminal_reward"],
        "final_step": _compact_step(trace["trace"][-1]),
        "first_bad_override": first_bad,
    }


def _override_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for step in trace["trace"]:
        if not step["override_safety_action"]:
            continue
        rows.append(
            {
                "policy_id": trace["policy_id"],
                "episode_index": trace["episode_index"],
                "env_seed": trace["env_seed"],
                "crashed": trace["crashed"],
                **_compact_step(step),
            }
        )
    return rows


def _top_override_states(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, tuple[int, ...], int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["policy_id"],
            tuple(row["state_key"]),
            row["selected_action"],
            row["safety_action"],
        )
        item = grouped.setdefault(
            key,
            {
                "policy_id": row["policy_id"],
                "state_key": row["state_key"],
                "selected_action": row["selected_action"],
                "selected_label": row["selected_label"],
                "safety_action": row["safety_action"],
                "safety_label": row["safety_label"],
                "count": 0,
                "crash_episode_count": 0,
                "max_score_delta_vs_safety": row["score_delta_vs_safety"],
                "sample": row,
            },
        )
        item["count"] += 1
        item["crash_episode_count"] += int(row["crashed"])
        item["max_score_delta_vs_safety"] = max(
            item["max_score_delta_vs_safety"],
            row["score_delta_vs_safety"],
        )
    return sorted(
        grouped.values(),
        key=lambda item: (item["crash_episode_count"], item["count"], item["max_score_delta_vs_safety"]),
        reverse=True,
    )[:20]


def _episode_row(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_id": trace["policy_id"],
        "episode_index": trace["episode_index"],
        "env_seed": trace["env_seed"],
        "steps": trace["steps"],
        "terminal_reward": trace["terminal_reward"],
        "crashed": trace["crashed"],
        "survived": trace["survived"],
        "action_counts": trace["action_counts"],
        "override_safety_action_steps": sum(
            1 for step in trace["trace"] if step["override_safety_action"]
        ),
        "lower_clearance_override_steps": sum(
            1 for step in trace["trace"] if step["lower_clearance_than_safety"]
        ),
        "selected_zero_clearance_steps": sum(
            1 for step in trace["trace"] if step["selected_zero_clearance"]
        ),
        "first_bad_override": _first_bad_override(trace),
    }


def _compact_step(step: dict[str, Any]) -> dict[str, Any]:
    selected_action = step["selected_action"]
    safety_action = step["safety_action"]
    return {
        "step_index": step["step_index"],
        "state_key": list(step["state_key"]),
        "observation": step["observation"],
        "clearances": list(step["clearances"]),
        "selected_action": selected_action,
        "selected_label": step["selected_label"],
        "safety_action": safety_action,
        "safety_label": step["safety_label"],
        "selected_clearance": step["selected_clearance"],
        "safety_clearance": step["safety_clearance"],
        "score_delta_vs_safety": step["score_delta_vs_safety"],
        "scores": step["scores"],
        "model_rewards": step["model_rewards"],
        "next_values": step["next_values"],
        "selected_next_key": _jsonable_key(step["next_keys"][selected_action]),
        "safety_next_key": _jsonable_key(step["next_keys"][safety_action]),
        "selected_next_q": step["next_argmax_q"][selected_action],
        "safety_next_q": step["next_argmax_q"][safety_action],
        "current_q": step["current_q"],
        "terminated": step["terminated"],
        "reward": step["reward"],
    }


def _safety_action(clearances: tuple[int, int, int]) -> int:
    best_clearance = max(clearances)
    for action in (1, 0, 2):
        if clearances[action] == best_clearance:
            return action
    raise AssertionError("unreachable")


def _episode_seeds(*, seed: int, episodes: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(2**31 - 1, size=episodes)]


def _by_seed(traces: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(trace["env_seed"]): trace for trace in traces}


def _jsonable_key(key: tuple[int, ...] | StateKey | None) -> list[int] | None:
    return None if key is None else [int(value) for value in key]


def _write_outputs(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    override_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in episode_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "overrides.jsonl").open("w", encoding="utf-8") as handle:
        for row in override_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
