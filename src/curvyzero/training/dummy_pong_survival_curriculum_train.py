"""Tiny raster Pong policy-gradient trainer with explicit survival reward.

Run from the repository root:

    uv run python -m curvyzero.training.dummy_pong_survival_curriculum_train \
      --epochs 4 \
      --games-per-epoch 4 \
      --eval-games 2 \
      --output-dir artifacts/local/dummy-pong-survival-curriculum-smoke

This is intentionally small. It is not MuZero, PPO, or a league. It trains the
same linear raster policy checkpoint shape that the existing Pong scoreboard can
load, but it optimizes an on-policy survival/loss-delay return instead of the
old offline random self-play rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.training.dummy_pong import ACTION_LABELS
from curvyzero.training.dummy_pong import ACTION_SCHEMA_ID
from curvyzero.training.dummy_pong import AGENTS
from curvyzero.training.dummy_pong import RASTER_LEGEND
from curvyzero.training.dummy_pong import RASTER_OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import REWARD_SCHEMA_ID
from curvyzero.training.dummy_pong import RULESET_ID
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv
from curvyzero.training.dummy_pong import PongObservation
from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
from curvyzero.training.dummy_pong_eval import TrackBallPolicy
from curvyzero.training.dummy_pong_imitation_train import DummyPongImitationPolicy
from curvyzero.training.dummy_pong_imitation_train import _agent_index
from curvyzero.training.dummy_pong_imitation_train import _encode_raster_grid
from curvyzero.training.dummy_pong_imitation_train import _feature_dim
from curvyzero.training.dummy_pong_imitation_train import _softmax

SURVIVAL_CURRICULUM_TRAIN_SUMMARY_SCHEMA_ID = (
    "dummy_pong_survival_curriculum_train_summary_v0"
)
SURVIVAL_CURRICULUM_POLICY_CHECKPOINT_SCHEMA_ID = (
    "dummy_pong_survival_curriculum_policy_checkpoint_v0"
)
SURVIVAL_CURRICULUM_REWARD_SCHEMA_ID = (
    "dummy_pong_score_plus_survival_loss_delay_return_v0"
)
LOSS_DELAY_REWARD_SCHEMA_ID = "dummy_pong_loss_delay_training_return_v0"
FEATURE_ENCODING_ID = "dummy_pong_raster_one_hot_plus_geometry_v0"


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    opponent_policy_id: str
    learner_agent: str
    seed: int
    steps: int
    max_steps: int
    winner: str | None
    terminated: bool
    truncated: bool
    score_return: float
    survival_fraction: float
    training_return: float
    action_counts: list[int]


class NoisyTrackBallPolicy:
    """Track-ball policy with random-action noise for a weak curriculum stage."""

    name = "weak_track_ball"

    def __init__(self, *, epsilon: float, seed: int):
        self._epsilon = epsilon
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)
        self._track_ball = TrackBallPolicy()

    def reset(self, episode_seed: int, agent: str) -> None:
        self._track_ball.reset(episode_seed, agent)
        self._rng = np.random.default_rng(self._base_seed + episode_seed + _agent_seed(agent))

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int:
        if float(self._rng.random()) < self._epsilon:
            return int(self._rng.integers(len(ACTION_LABELS)))
        return int(self._track_ball.action(observation, raster_grid, agent))


def train_dummy_pong_survival_curriculum(
    *,
    output_dir: Path | None = None,
    seed: int = 0,
    epochs: int = 20,
    games_per_epoch: int = 16,
    eval_games: int = 8,
    max_steps: int = 120,
    learning_rate: float = 0.05,
    l2: float = 0.0001,
    survival_weight: float = 0.5,
    truncation_bonus: float = 0.25,
    reward_mode: str = "score_plus_survival",
    weak_track_ball_epsilon: float = 0.35,
    random_phase_fraction: float = 0.34,
    weak_phase_fraction: float = 0.34,
    initial_checkpoint: Path | None = None,
) -> dict[str, object]:
    """Train a small visual Pong policy with explicit survival/loss-delay reward."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if games_per_epoch < 1:
        raise ValueError("games_per_epoch must be at least 1")
    if eval_games < 1:
        raise ValueError("eval_games must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    if survival_weight < 0.0:
        raise ValueError("survival_weight must be non-negative")
    if truncation_bonus < 0.0:
        raise ValueError("truncation_bonus must be non-negative")
    if reward_mode not in {"score_plus_survival", "loss_delay"}:
        raise ValueError("reward_mode must be 'score_plus_survival' or 'loss_delay'")
    if not 0.0 <= weak_track_ball_epsilon <= 1.0:
        raise ValueError("weak_track_ball_epsilon must be in [0.0, 1.0]")
    if random_phase_fraction < 0.0 or weak_phase_fraction < 0.0:
        raise ValueError("curriculum phase fractions must be non-negative")
    if random_phase_fraction + weak_phase_fraction > 1.0:
        raise ValueError("random_phase_fraction + weak_phase_fraction must be <= 1.0")

    config = PongConfig(max_steps=max_steps)
    raster_shape = (config.height, config.width)
    weights, bias = _initial_policy_parameters(
        initial_checkpoint=initial_checkpoint,
        raster_shape=raster_shape,
    )
    rng = np.random.default_rng(seed)
    history = []

    for epoch in range(epochs):
        opponent_policy_id = _curriculum_opponent(
            completed_epoch=epoch,
            epochs=epochs,
            random_phase_fraction=random_phase_fraction,
            weak_phase_fraction=weak_phase_fraction,
        )
        rollout = _collect_epoch_rollout(
            config=config,
            games=games_per_epoch,
            seed_rng=rng,
            policy_seed=seed + epoch * 10_000,
            opponent_policy_id=opponent_policy_id,
            weak_track_ball_epsilon=weak_track_ball_epsilon,
            weights=weights,
            bias=bias,
            survival_weight=survival_weight,
            truncation_bonus=truncation_bonus,
            reward_mode=reward_mode,
            sample_actions=True,
        )
        _policy_gradient_update(
            rollout=rollout,
            weights=weights,
            bias=bias,
            learning_rate=learning_rate,
            l2=l2,
        )
        eval_metrics = _eval_policy(
            config=config,
            games_per_opponent=eval_games,
            base_seed=seed + 1_000_000 + epoch * 10_000,
            weak_track_ball_epsilon=weak_track_ball_epsilon,
            weights=weights,
            bias=bias,
            survival_weight=survival_weight,
            truncation_bonus=truncation_bonus,
            reward_mode=reward_mode,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "curriculum_opponent": opponent_policy_id,
                "train": _rollout_metrics(rollout["outcomes"]),
                "eval": eval_metrics,
            }
        )

    final_eval = _eval_policy(
        config=config,
        games_per_opponent=eval_games,
        base_seed=seed + 9_000_000,
        weak_track_ball_epsilon=weak_track_ball_epsilon,
        weights=weights,
        bias=bias,
        survival_weight=survival_weight,
        truncation_bonus=truncation_bonus,
        reward_mode=reward_mode,
    )

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_survival_curriculum_training",
        "note": (
            "Tiny on-policy raster policy-gradient trainer for dummy Pong. The "
            "objective explicitly rewards score, survival length, and max-step "
            "truncation; wins are reported but are not the only signal."
        ),
        "summary_schema_id": SURVIVAL_CURRICULUM_TRAIN_SUMMARY_SCHEMA_ID,
        "checkpoint_schema_id": SURVIVAL_CURRICULUM_POLICY_CHECKPOINT_SCHEMA_ID,
        "reward_schema_id": _reward_schema_id(reward_mode),
        "seed": seed,
        "epochs": epochs,
        "games_per_epoch": games_per_epoch,
        "eval_games": eval_games,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "l2": l2,
        "reward_mode": reward_mode,
        "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
        "curriculum": {
            "stages": [
                {
                    "policy_id": "random_uniform",
                    "phase_fraction": random_phase_fraction,
                },
                {
                    "policy_id": "weak_track_ball",
                    "phase_fraction": weak_phase_fraction,
                    "epsilon_random_action": weak_track_ball_epsilon,
                },
                {
                    "policy_id": "track_ball",
                    "phase_fraction": 1.0 - random_phase_fraction - weak_phase_fraction,
                },
            ],
            "learner_seating": "paired seat; each game index alternates learner agent",
        },
        "training_return": _reward_rule_summary(
            survival_weight=survival_weight,
            truncation_bonus=truncation_bonus,
            reward_mode=reward_mode,
        ),
        "schemas": _schemas_summary(),
        "input": {
            "raster_shape": list(raster_shape),
            "feature_shape": [_feature_dim(raster_shape)],
            "feature_encoding": FEATURE_ENCODING_ID,
            "raster_legend": RASTER_LEGEND,
        },
        "model": {
            "type": "per_ego_agent_softmax_linear_numpy_on_policy_reinforce",
            "weights_shape": list(weights.shape),
            "bias_shape": list(bias.shape),
            "ego_agents": list(AGENTS),
        },
        "action_labels": list(ACTION_LABELS),
        "metrics": {
            "final_eval": final_eval,
            "last_epoch_train": history[-1]["train"],
            "last_epoch_eval": history[-1]["eval"],
        },
        "history": history,
        "plain_language": {
            "proves": (
                "The learner can collect raster observations on-policy, optimize "
                "a visible survival/loss-delay shaped return, and write a "
                "checkpoint the existing Pong scoreboard can load."
            ),
            "does_not_prove": (
                "This is not MuZero or a strong Pong result. Track-ball wins, "
                "episode length, truncation rate, and shaped return must all be "
                "read together before scaling."
            ),
        },
    }
    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(
            artifacts=artifacts,
            summary=summary,
            weights=weights,
            bias=bias,
            raster_shape=raster_shape,
        )
    return summary


def _initial_policy_parameters(
    *,
    initial_checkpoint: Path | None,
    raster_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    expected_weights_shape = (len(AGENTS), _feature_dim(raster_shape), len(ACTION_LABELS))
    if initial_checkpoint is None:
        return (
            np.zeros(expected_weights_shape, dtype=np.float64),
            np.zeros((len(AGENTS), len(ACTION_LABELS)), dtype=np.float64),
        )
    policy = DummyPongImitationPolicy.load_checkpoint(initial_checkpoint)
    if policy.raster_shape != raster_shape:
        raise ValueError(
            f"initial checkpoint raster shape {policy.raster_shape!r} "
            f"does not match expected {raster_shape!r}"
        )
    if policy.weights.shape != expected_weights_shape:
        raise ValueError(f"unexpected initial checkpoint weight shape {policy.weights.shape!r}")
    return np.array(policy.weights, copy=True), np.array(policy.bias, copy=True)


def _curriculum_opponent(
    *,
    completed_epoch: int,
    epochs: int,
    random_phase_fraction: float,
    weak_phase_fraction: float,
) -> str:
    progress = completed_epoch / max(epochs, 1)
    if progress < random_phase_fraction:
        return "random_uniform"
    if progress < random_phase_fraction + weak_phase_fraction:
        return "weak_track_ball"
    return "track_ball"


def _collect_epoch_rollout(
    *,
    config: PongConfig,
    games: int,
    seed_rng: np.random.Generator,
    policy_seed: int,
    opponent_policy_id: str,
    weak_track_ball_epsilon: float,
    weights: np.ndarray,
    bias: np.ndarray,
    survival_weight: float,
    truncation_bonus: float,
    reward_mode: str,
    sample_actions: bool,
) -> dict[str, Any]:
    features = []
    actions = []
    probs = []
    agent_indices = []
    row_returns = []
    outcomes: list[EpisodeOutcome] = []
    for game_index in range(games):
        learner_agent = AGENTS[game_index % len(AGENTS)]
        episode_seed = int(seed_rng.integers(2**31 - 1))
        episode = _run_episode(
            config=config,
            episode_seed=episode_seed,
            policy_seed=policy_seed + game_index * 997,
            learner_agent=learner_agent,
            opponent_policy_id=opponent_policy_id,
            weak_track_ball_epsilon=weak_track_ball_epsilon,
            weights=weights,
            bias=bias,
            survival_weight=survival_weight,
            truncation_bonus=truncation_bonus,
            reward_mode=reward_mode,
            sample_actions=sample_actions,
        )
        features.extend(episode["features"])
        actions.extend(episode["actions"])
        probs.extend(episode["probs"])
        agent_indices.extend(episode["agent_indices"])
        row_returns.extend([episode["outcome"].training_return] * len(episode["actions"]))
        outcomes.append(episode["outcome"])
    return {
        "features": np.vstack(features) if features else np.empty((0, _feature_dim((config.height, config.width)))),
        "actions": np.asarray(actions, dtype=np.int64),
        "probs": np.vstack(probs) if probs else np.empty((0, len(ACTION_LABELS))),
        "agent_indices": np.asarray(agent_indices, dtype=np.int64),
        "returns": np.asarray(row_returns, dtype=np.float64),
        "outcomes": outcomes,
    }


def _run_episode(
    *,
    config: PongConfig,
    episode_seed: int,
    policy_seed: int,
    learner_agent: str,
    opponent_policy_id: str,
    weak_track_ball_epsilon: float,
    weights: np.ndarray,
    bias: np.ndarray,
    survival_weight: float,
    truncation_bonus: float,
    reward_mode: str,
    sample_actions: bool,
) -> dict[str, Any]:
    env = PongEnv(config)
    observations = env.reset(seed=episode_seed)
    opponent_agent = _opponent(learner_agent)
    opponent_policy = _make_opponent_policy(
        opponent_policy_id,
        epsilon=weak_track_ball_epsilon,
        seed=policy_seed + 31,
    )
    opponent_policy.reset(episode_seed, opponent_agent)
    learner_rng = np.random.default_rng(policy_seed + episode_seed + _agent_seed(learner_agent))

    features = []
    actions = []
    probs = []
    agent_indices = []
    action_counts = [0 for _ in ACTION_LABELS]

    final_step = None
    while True:
        raster_grid = env.raster_observation()
        learner_features = _encode_raster_grid(raster_grid, expected_shape=(config.height, config.width))
        learner_agent_index = _agent_index(learner_agent)
        learner_probs = _softmax(learner_features @ weights[learner_agent_index] + bias[learner_agent_index])
        if sample_actions:
            learner_action = int(learner_rng.choice(len(ACTION_LABELS), p=learner_probs))
        else:
            learner_action = int(np.argmax(learner_probs))
        opponent_action = int(
            opponent_policy.action(observations[opponent_agent], raster_grid, opponent_agent)
        )
        joint_action = {
            learner_agent: learner_action,
            opponent_agent: opponent_action,
        }
        step = env.step(joint_action)

        features.append(learner_features)
        actions.append(learner_action)
        probs.append(learner_probs)
        agent_indices.append(learner_agent_index)
        action_counts[learner_action] += 1

        observations = step.observations
        final_step = step
        if step.terminated or step.truncated:
            break

    if final_step is None:
        raise RuntimeError("dummy Pong curriculum episode produced no steps")

    steps = int(next(iter(final_step.observations.values())).step)
    winner = final_step.infos["winner"] if final_step.terminated else None
    score_return = _score_return(
        learner_agent=learner_agent,
        winner=winner,
        terminated=final_step.terminated,
    )
    survival_fraction = steps / config.max_steps
    training_return = _training_return(
        score_return=score_return,
        survival_fraction=survival_fraction,
        truncated=final_step.truncated,
        survival_weight=survival_weight,
        truncation_bonus=truncation_bonus,
        reward_mode=reward_mode,
    )
    return {
        "features": features,
        "actions": actions,
        "probs": probs,
        "agent_indices": agent_indices,
        "outcome": EpisodeOutcome(
            opponent_policy_id=opponent_policy_id,
            learner_agent=learner_agent,
            seed=episode_seed,
            steps=steps,
            max_steps=config.max_steps,
            winner=winner,
            terminated=final_step.terminated,
            truncated=final_step.truncated,
            score_return=score_return,
            survival_fraction=survival_fraction,
            training_return=training_return,
            action_counts=action_counts,
        ),
    }


def _policy_gradient_update(
    *,
    rollout: dict[str, Any],
    weights: np.ndarray,
    bias: np.ndarray,
    learning_rate: float,
    l2: float,
) -> None:
    if len(rollout["actions"]) == 0:
        return
    returns = rollout["returns"]
    advantages = returns - float(np.mean(returns))
    scale = float(np.std(advantages))
    if scale > 1e-8:
        advantages = advantages / scale
    advantages = np.clip(advantages, -3.0, 3.0)
    weight_grad = np.zeros_like(weights)
    bias_grad = np.zeros_like(bias)
    for row_index, agent_index in enumerate(rollout["agent_indices"]):
        action = int(rollout["actions"][row_index])
        grad_logits = -np.asarray(rollout["probs"][row_index], dtype=np.float64)
        grad_logits[action] += 1.0
        grad_logits *= advantages[row_index]
        feature = rollout["features"][row_index]
        weight_grad[agent_index] += np.outer(feature, grad_logits)
        bias_grad[agent_index] += grad_logits
    row_count = float(len(rollout["actions"]))
    if l2:
        weight_grad -= l2 * weights
    weights += learning_rate * weight_grad / row_count
    bias += learning_rate * bias_grad / row_count


def _eval_policy(
    *,
    config: PongConfig,
    games_per_opponent: int,
    base_seed: int,
    weak_track_ball_epsilon: float,
    weights: np.ndarray,
    bias: np.ndarray,
    survival_weight: float,
    truncation_bonus: float,
    reward_mode: str,
) -> dict[str, object]:
    metrics = {}
    for opponent_index, opponent_policy_id in enumerate(
        ("random_uniform", "weak_track_ball", "track_ball")
    ):
        outcomes = []
        seed_rng = np.random.default_rng(base_seed + opponent_index * 100_000)
        for game_index in range(games_per_opponent * len(AGENTS)):
            learner_agent = AGENTS[game_index % len(AGENTS)]
            episode_seed = int(seed_rng.integers(2**31 - 1))
            episode = _run_episode(
                config=config,
                episode_seed=episode_seed,
                policy_seed=base_seed + opponent_index * 10_000 + game_index * 997,
                learner_agent=learner_agent,
                opponent_policy_id=opponent_policy_id,
                weak_track_ball_epsilon=weak_track_ball_epsilon,
                weights=weights,
                bias=bias,
                survival_weight=survival_weight,
                truncation_bonus=truncation_bonus,
                reward_mode=reward_mode,
                sample_actions=False,
            )
            outcomes.append(episode["outcome"])
        metrics[opponent_policy_id] = _rollout_metrics(outcomes)
    return metrics


def _rollout_metrics(outcomes: list[EpisodeOutcome]) -> dict[str, object]:
    if not outcomes:
        return {
            "episodes": 0,
            "wins": 0,
            "losses": 0,
            "truncations": 0,
            "win_rate": 0.0,
            "truncation_rate": 0.0,
            "mean_steps": 0.0,
            "mean_survival_fraction": 0.0,
            "mean_score_return": 0.0,
            "mean_training_return": 0.0,
        }
    wins = sum(outcome.score_return > 0.0 for outcome in outcomes)
    losses = sum(outcome.score_return < 0.0 for outcome in outcomes)
    truncations = sum(outcome.truncated for outcome in outcomes)
    action_counts = Counter()
    for outcome in outcomes:
        for action_id, count in enumerate(outcome.action_counts):
            action_counts[ACTION_LABELS[action_id]] += count
    return {
        "episodes": len(outcomes),
        "wins": int(wins),
        "losses": int(losses),
        "truncations": int(truncations),
        "win_rate": float(wins / len(outcomes)),
        "truncation_rate": float(truncations / len(outcomes)),
        "mean_steps": float(np.mean([outcome.steps for outcome in outcomes])),
        "mean_survival_fraction": float(
            np.mean([outcome.survival_fraction for outcome in outcomes])
        ),
        "mean_score_return": float(np.mean([outcome.score_return for outcome in outcomes])),
        "mean_training_return": float(
            np.mean([outcome.training_return for outcome in outcomes])
        ),
        "action_histogram": {
            label: int(action_counts[label])
            for label in ACTION_LABELS
        },
        "by_learner_agent": {
            agent: _rollout_metrics_for_agent(outcomes, agent)
            for agent in AGENTS
        },
    }


def _rollout_metrics_for_agent(
    outcomes: list[EpisodeOutcome],
    agent: str,
) -> dict[str, object]:
    agent_outcomes = [outcome for outcome in outcomes if outcome.learner_agent == agent]
    if not agent_outcomes:
        return {
            "episodes": 0,
            "wins": 0,
            "losses": 0,
            "truncations": 0,
            "mean_steps": 0.0,
            "mean_training_return": 0.0,
        }
    wins = sum(outcome.score_return > 0.0 for outcome in agent_outcomes)
    losses = sum(outcome.score_return < 0.0 for outcome in agent_outcomes)
    return {
        "episodes": len(agent_outcomes),
        "wins": int(wins),
        "losses": int(losses),
        "truncations": int(sum(outcome.truncated for outcome in agent_outcomes)),
        "mean_steps": float(np.mean([outcome.steps for outcome in agent_outcomes])),
        "mean_training_return": float(
            np.mean([outcome.training_return for outcome in agent_outcomes])
        ),
    }


def _score_return(
    *,
    learner_agent: str,
    winner: str | None,
    terminated: bool,
) -> float:
    if not terminated or winner is None:
        return 0.0
    return 1.0 if winner == learner_agent else -1.0


def _training_return(
    *,
    score_return: float,
    survival_fraction: float,
    truncated: bool,
    survival_weight: float,
    truncation_bonus: float,
    reward_mode: str,
) -> float:
    if reward_mode == "loss_delay":
        if score_return > 0.0:
            return 1.0
        if score_return < 0.0:
            return -1.0 + survival_weight * survival_fraction
        return truncation_bonus if truncated else 0.0
    if reward_mode != "score_plus_survival":
        raise ValueError(f"unknown reward_mode {reward_mode!r}")
    return (
        score_return
        + survival_weight * survival_fraction
        + (truncation_bonus if truncated else 0.0)
    )


def _make_opponent_policy(
    policy_id: str,
    *,
    epsilon: float,
    seed: int,
) -> RandomUniformPolicy | TrackBallPolicy | NoisyTrackBallPolicy:
    if policy_id == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    if policy_id == "weak_track_ball":
        return NoisyTrackBallPolicy(epsilon=epsilon, seed=seed)
    if policy_id == "track_ball":
        return TrackBallPolicy()
    raise ValueError(f"unknown opponent policy {policy_id!r}")


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "checkpoint_npz": output_dir / "checkpoint.npz",
    }


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    weights: np.ndarray,
    bias: np.ndarray,
    raster_shape: tuple[int, int],
) -> None:
    artifacts["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_policy_checkpoint(
        path=artifacts["checkpoint_npz"],
        weights=weights,
        bias=bias,
        raster_shape=raster_shape,
        metadata=_checkpoint_metadata(summary),
    )


def _write_policy_checkpoint(
    *,
    path: Path,
    weights: np.ndarray,
    bias: np.ndarray,
    raster_shape: tuple[int, int],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=weights,
        bias=bias,
        raster_shape=np.asarray(raster_shape, dtype=np.int64),
        action_labels=np.asarray(ACTION_LABELS),
        ego_agents=np.asarray(AGENTS),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _checkpoint_metadata(summary: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "curvyzero_dummy_pong_survival_curriculum_policy",
        "checkpoint_schema_id": summary["checkpoint_schema_id"],
        "summary_schema_id": summary["summary_schema_id"],
        "reward_schema_id": summary["reward_schema_id"],
        "seed": summary["seed"],
        "epochs": summary["epochs"],
        "games_per_epoch": summary["games_per_epoch"],
        "max_steps": summary["max_steps"],
        "learning_rate": summary["learning_rate"],
        "l2": summary["l2"],
        "initial_checkpoint": summary["initial_checkpoint"],
        "curriculum": summary["curriculum"],
        "training_return": summary["training_return"],
        "schemas": summary["schemas"],
        "input": summary["input"],
        "model": summary["model"],
        "action_labels": summary["action_labels"],
        "metrics": summary["metrics"],
        "note": summary["note"],
    }


def _reward_rule_summary(
    *,
    survival_weight: float,
    truncation_bonus: float,
    reward_mode: str,
) -> dict[str, object]:
    if reward_mode == "loss_delay":
        formula = (
            "win -> +1.0; loss -> -1.0 + survival_weight * "
            "(episode_steps / max_steps); timeout/draw -> truncation_bonus"
        )
    else:
        formula = (
            "score_return + survival_weight * (episode_steps / max_steps) "
            "+ truncation_bonus_if_max_steps"
        )
    return {
        "mode": reward_mode,
        "formula": formula,
        "score_return": "win=+1.0, loss=-1.0, truncation_or_draw=0.0",
        "survival_weight": survival_weight,
        "truncation_bonus": truncation_bonus,
        "warning": (
            "This is a labeled shaped-objective training return. Keep "
            "mean_score_return and wins/losses separate from mean_training_return; "
            "do not compare this run as stock sparse-reward Pong."
        ),
        "reported_eval_fields": [
            "wins",
            "losses",
            "truncations",
            "win_rate",
            "truncation_rate",
            "mean_steps",
            "mean_survival_fraction",
            "mean_score_return",
            "mean_training_return",
        ],
    }


def _reward_schema_id(reward_mode: str) -> str:
    if reward_mode == "loss_delay":
        return LOSS_DELAY_REWARD_SCHEMA_ID
    return SURVIVAL_CURRICULUM_REWARD_SCHEMA_ID


def _schemas_summary() -> dict[str, object]:
    return {
        "ruleset_id": RULESET_ID,
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "reward_schema_id": REWARD_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "survival_curriculum_train_summary_schema_id": (
            SURVIVAL_CURRICULUM_TRAIN_SUMMARY_SCHEMA_ID
        ),
        "survival_curriculum_policy_checkpoint_schema_id": (
            SURVIVAL_CURRICULUM_POLICY_CHECKPOINT_SCHEMA_ID
        ),
        "survival_curriculum_reward_schema_id": SURVIVAL_CURRICULUM_REWARD_SCHEMA_ID,
        "loss_delay_reward_schema_id": LOSS_DELAY_REWARD_SCHEMA_ID,
        "feature_encoding_id": FEATURE_ENCODING_ID,
    }


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent {agent!r}")


def _agent_seed(agent: str) -> int:
    if agent == "player_0":
        return 17
    if agent == "player_1":
        return 29
    raise ValueError(f"unknown agent {agent!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--games-per-epoch", type=int, default=16)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--survival-weight", type=float, default=0.5)
    parser.add_argument("--truncation-bonus", type=float, default=0.25)
    parser.add_argument(
        "--reward-mode",
        choices=("score_plus_survival", "loss_delay"),
        default="score_plus_survival",
    )
    parser.add_argument("--weak-track-ball-epsilon", type=float, default=0.35)
    parser.add_argument("--random-phase-fraction", type=float, default=0.34)
    parser.add_argument("--weak-phase-fraction", type=float, default=0.34)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    summary = train_dummy_pong_survival_curriculum(
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        games_per_epoch=args.games_per_epoch,
        eval_games=args.eval_games,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        l2=args.l2,
        survival_weight=args.survival_weight,
        truncation_bonus=args.truncation_bonus,
        reward_mode=args.reward_mode,
        weak_track_ball_epsilon=args.weak_track_ball_epsilon,
        random_phase_fraction=args.random_phase_fraction,
        weak_phase_fraction=args.weak_phase_fraction,
        initial_checkpoint=args.initial_checkpoint,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
