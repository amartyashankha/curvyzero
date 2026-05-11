"""Tiny CEM/random-search Pong learner with score-primary selection.

Run from the repository root:

    uv run python -m curvyzero.training.dummy_pong_cem_train \
      --generations 2 \
      --population-size 8 \
      --elite-count 3 \
      --eval-games 4 \
      --output-dir artifacts/local/dummy-pong-cem-smoke

This is intentionally narrow. It searches a compact geometry-only slice of the
existing raster linear policy, then writes the normal checkpoint shape that the
dummy Pong scoreboard can load.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
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
from curvyzero.training.dummy_pong_eval import AngleControlPolicy
from curvyzero.training.dummy_pong_eval import LaggedTrackBallPolicy
from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
from curvyzero.training.dummy_pong_eval import TrackBallPolicy
from curvyzero.training.dummy_pong_imitation_train import _agent_index
from curvyzero.training.dummy_pong_imitation_train import _encode_raster_grid
from curvyzero.training.dummy_pong_imitation_train import _feature_dim
from curvyzero.training.dummy_pong_imitation_train import _softmax

CEM_TRAIN_SUMMARY_SCHEMA_ID = "dummy_pong_geometry_cem_train_summary_v0"
CEM_POLICY_CHECKPOINT_SCHEMA_ID = "dummy_pong_geometry_cem_policy_checkpoint_v0"
CEM_SELECTION_SCHEMA_ID = "dummy_pong_score_primary_loss_delay_selection_v1"
FEATURE_ENCODING_ID = "dummy_pong_raster_one_hot_plus_geometry_v0"
CEM_OPPONENT_POLICY_IDS = (
    "random_uniform",
    "weak_track_ball",
    "lagged_track_ball_1",
    "track_ball",
    "angle_control",
)
_SHAPED_SELECTION_TIEBREAK_SCALE = 1.0e-3

_GEOMETRY_FEATURES = (
    "ball_x_over_width_minus_1",
    "ball_y_over_height_minus_1",
    "player_0_center_y_over_height_minus_1",
    "player_1_center_y_over_height_minus_1",
    "ball_y_minus_player_0_center_y_over_height_minus_1",
    "ball_y_minus_player_1_center_y_over_height_minus_1",
)
_SEARCH_PARAMETER_DIM = len(AGENTS) * (
    len(_GEOMETRY_FEATURES) * len(ACTION_LABELS) + len(ACTION_LABELS)
)


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    opponent_policy_id: str
    learner_agent: str
    seed: int
    steps: int
    max_steps: int
    winner: str | None
    truncated: bool
    score_return: float
    shaped_proxy: float
    action_counts: list[int]


class NoisyTrackBallPolicy:
    """Track-ball policy with random-action noise for weak target ladders."""

    name = "weak_track_ball"

    def __init__(self, *, epsilon: float, seed: int):
        self._epsilon = epsilon
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)
        self._track_ball = TrackBallPolicy()

    def reset(self, episode_seed: int, agent: str) -> None:
        self._track_ball.reset(episode_seed, agent)
        self._rng = np.random.default_rng(
            self._base_seed + episode_seed + _agent_seed(agent)
        )

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int:
        if float(self._rng.random()) < self._epsilon:
            return int(self._rng.integers(len(ACTION_LABELS)))
        return int(self._track_ball.action(observation, raster_grid, agent))


def train_dummy_pong_cem(
    *,
    output_dir: Path | None = None,
    seed: int = 0,
    width: int = 15,
    height: int = 9,
    paddle_height: int = 3,
    generations: int = 3,
    population_size: int = 12,
    elite_count: int = 4,
    eval_games: int = 8,
    max_steps: int = 120,
    initial_sigma: float = 0.75,
    sigma_decay: float = 0.7,
    min_sigma: float = 0.05,
    track_ball_prior_strength: float = 12.0,
    stay_bias: float = 0.75,
    random_opponent_weight: float = 0.25,
    track_ball_opponent_weight: float = 0.75,
    opponent_weights: dict[str, float] | None = None,
    target_opponent_id: str = "track_ball",
    weak_track_ball_epsilon: float = 0.35,
    loss_delay_weight: float = 0.5,
    truncation_value: float = 0.0,
) -> dict[str, object]:
    """Search a small geometry policy and save a scoreboard-loadable checkpoint."""

    if generations < 1:
        raise ValueError("generations must be at least 1")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if elite_count < 1 or elite_count > population_size:
        raise ValueError("elite_count must be in [1, population_size]")
    if eval_games < 1:
        raise ValueError("eval_games must be at least 1")
    if width < 7:
        raise ValueError("width must be at least 7")
    if height < paddle_height + 2:
        raise ValueError("height must leave room around the paddle")
    if paddle_height < 1:
        raise ValueError("paddle_height must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if initial_sigma < 0.0:
        raise ValueError("initial_sigma must be non-negative")
    if sigma_decay <= 0.0:
        raise ValueError("sigma_decay must be positive")
    if min_sigma < 0.0:
        raise ValueError("min_sigma must be non-negative")
    if random_opponent_weight < 0.0 or track_ball_opponent_weight < 0.0:
        raise ValueError("opponent weights must be non-negative")
    if not 0.0 <= weak_track_ball_epsilon <= 1.0:
        raise ValueError("weak_track_ball_epsilon must be in [0.0, 1.0]")
    if loss_delay_weight < 0.0:
        raise ValueError("loss_delay_weight must be non-negative")

    config = PongConfig(
        width=width,
        height=height,
        paddle_height=paddle_height,
        max_steps=max_steps,
    )
    raster_shape = (config.height, config.width)
    rng = np.random.default_rng(seed)
    mean = _track_ball_geometry_vector(
        track_ball_prior_strength=track_ball_prior_strength,
        stay_bias=stay_bias,
    )
    sigma = np.full_like(mean, initial_sigma, dtype=np.float64)
    if opponent_weights is None:
        opponent_weights = {
            "random_uniform": random_opponent_weight,
            "track_ball": track_ball_opponent_weight,
        }
    opponent_weights = _validated_opponent_weights(opponent_weights)
    if target_opponent_id not in CEM_OPPONENT_POLICY_IDS:
        raise ValueError(f"unknown target opponent {target_opponent_id!r}")
    if opponent_weights.get(target_opponent_id, 0.0) <= 0.0:
        raise ValueError("target_opponent_id must have a positive opponent weight")

    best_candidate: dict[str, Any] | None = None
    history = []
    for generation in range(generations):
        candidate_vectors = [np.array(mean, copy=True)]
        while len(candidate_vectors) < population_size:
            candidate_vectors.append(mean + rng.normal(0.0, sigma))

        scored = []
        for candidate_index, vector in enumerate(candidate_vectors):
            weights, bias = _full_policy_from_geometry_vector(
                vector,
                raster_shape=raster_shape,
            )
            metrics = _eval_candidate(
                config=config,
                weights=weights,
                bias=bias,
                eval_games=eval_games,
                base_seed=seed + generation * 1_000_000 + candidate_index * 10_000,
                opponent_weights=opponent_weights,
                weak_track_ball_epsilon=weak_track_ball_epsilon,
                loss_delay_weight=loss_delay_weight,
                truncation_value=truncation_value,
            )
            scored.append(
                {
                    "generation": generation + 1,
                    "candidate_index": candidate_index,
                    "vector": vector,
                    "selection_score": metrics["selection_score"],
                    "metrics": metrics,
                }
            )

        scored.sort(
            key=lambda candidate: _candidate_sort_key(
                candidate,
                target_opponent_id=target_opponent_id,
            ),
            reverse=True,
        )
        generation_best = scored[0]
        if best_candidate is None or _candidate_sort_key(
            generation_best,
            target_opponent_id=target_opponent_id,
        ) > _candidate_sort_key(
            best_candidate,
            target_opponent_id=target_opponent_id,
        ):
            best_candidate = generation_best
        elite_vectors = np.vstack([candidate["vector"] for candidate in scored[:elite_count]])
        mean = np.mean(elite_vectors, axis=0)
        sigma = np.maximum(np.std(elite_vectors, axis=0) * sigma_decay, min_sigma)
        history.append(
            {
                "generation": generation + 1,
                "best": _candidate_summary(generation_best),
                "elite_mean_selection_score": float(
                    np.mean([candidate["selection_score"] for candidate in scored[:elite_count]])
                ),
                "population_selection_score": {
                    "mean": float(np.mean([candidate["selection_score"] for candidate in scored])),
                    "min": float(np.min([candidate["selection_score"] for candidate in scored])),
                    "max": float(np.max([candidate["selection_score"] for candidate in scored])),
                },
                "sigma_mean": float(np.mean(sigma)),
            }
        )

    if best_candidate is None:
        raise RuntimeError("CEM search produced no candidates")
    best_weights, best_bias = _full_policy_from_geometry_vector(
        best_candidate["vector"],
        raster_shape=raster_shape,
    )
    final_eval = _eval_candidate(
        config=config,
        weights=best_weights,
        bias=best_bias,
        eval_games=eval_games,
        base_seed=seed + 9_000_000,
        opponent_weights=opponent_weights,
        weak_track_ball_epsilon=weak_track_ball_epsilon,
        loss_delay_weight=loss_delay_weight,
        truncation_value=truncation_value,
    )

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_geometry_cem_training",
        "note": (
            "Tiny cross-entropy/random-search baseline over only the geometry "
            "slice of the existing raster linear policy. Selection is direct "
            "paired-seat eval against configured opponents using score return "
            "as the primary objective and survival/loss-delay only as a "
            "secondary tie-breaker."
        ),
        "summary_schema_id": CEM_TRAIN_SUMMARY_SCHEMA_ID,
        "checkpoint_schema_id": CEM_POLICY_CHECKPOINT_SCHEMA_ID,
        "selection_schema_id": CEM_SELECTION_SCHEMA_ID,
        "seed": seed,
        "config": asdict(config),
        "target_opponent_id": target_opponent_id,
        "generations": generations,
        "population_size": population_size,
        "elite_count": elite_count,
        "eval_games": eval_games,
        "max_steps": max_steps,
        "initial_sigma": initial_sigma,
        "sigma_decay": sigma_decay,
        "min_sigma": min_sigma,
        "track_ball_prior_strength": track_ball_prior_strength,
        "stay_bias": stay_bias,
        "weak_track_ball_epsilon": weak_track_ball_epsilon,
        "selection_return": _selection_rule_summary(
            opponent_weights=opponent_weights,
            target_opponent_id=target_opponent_id,
            loss_delay_weight=loss_delay_weight,
            truncation_value=truncation_value,
        ),
        "schemas": _schemas_summary(),
        "input": {
            "raster_shape": list(raster_shape),
            "feature_shape": [_feature_dim(raster_shape)],
            "feature_encoding": FEATURE_ENCODING_ID,
            "raster_legend": RASTER_LEGEND,
            "searched_geometry_features": list(_GEOMETRY_FEATURES),
        },
        "model": {
            "type": "per_ego_agent_softmax_linear_numpy_geometry_cem",
            "weights_shape": list(best_weights.shape),
            "bias_shape": list(best_bias.shape),
            "ego_agents": list(AGENTS),
            "searched_parameter_count": _SEARCH_PARAMETER_DIM,
            "checkpoint_parameter_count": int(best_weights.size + best_bias.size),
        },
        "action_labels": list(ACTION_LABELS),
        "metrics": {
            "best_search_candidate": _candidate_summary(best_candidate),
            "final_eval": final_eval,
        },
        "history": history,
        "plain_language": {
            "proves": (
                "A tiny black-box learner can directly select a loadable raster "
                "linear policy with score return primary and survival/loss "
                "delay as a secondary diagnostic."
            ),
            "does_not_prove": (
                "This is not MuZero, a strategic Pong breakthrough, or evidence "
                "that copying track_ball can beat track_ball. It is a compact "
                "scoreable baseline and failure probe."
            ),
        },
    }
    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(
            artifacts=artifacts,
            summary=summary,
            weights=best_weights,
            bias=best_bias,
            raster_shape=raster_shape,
        )
    return summary


def _track_ball_geometry_vector(
    *,
    track_ball_prior_strength: float,
    stay_bias: float,
) -> np.ndarray:
    weights = np.zeros((len(AGENTS), len(_GEOMETRY_FEATURES), len(ACTION_LABELS)))
    bias = np.zeros((len(AGENTS), len(ACTION_LABELS)))
    for agent in AGENTS:
        agent_index = _agent_index(agent)
        dy_feature_index = 4 if agent == "player_0" else 5
        weights[agent_index, dy_feature_index, 0] = -track_ball_prior_strength
        weights[agent_index, dy_feature_index, 2] = track_ball_prior_strength
        bias[agent_index, 1] = stay_bias
    return _pack_geometry_vector(weights, bias)


def _pack_geometry_vector(geometry_weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.concatenate([geometry_weights.reshape(-1), bias.reshape(-1)]).astype(np.float64)


def _unpack_geometry_vector(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected = _SEARCH_PARAMETER_DIM
    if vector.shape != (expected,):
        raise ValueError(f"expected vector shape {(expected,)!r}, got {vector.shape!r}")
    weight_count = len(AGENTS) * len(_GEOMETRY_FEATURES) * len(ACTION_LABELS)
    geometry_weights = vector[:weight_count].reshape(
        len(AGENTS),
        len(_GEOMETRY_FEATURES),
        len(ACTION_LABELS),
    )
    bias = vector[weight_count:].reshape(len(AGENTS), len(ACTION_LABELS))
    return geometry_weights, bias


def _full_policy_from_geometry_vector(
    vector: np.ndarray,
    *,
    raster_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    geometry_weights, bias = _unpack_geometry_vector(vector)
    weights = np.zeros((len(AGENTS), _feature_dim(raster_shape), len(ACTION_LABELS)))
    weights[:, -len(_GEOMETRY_FEATURES) :, :] = geometry_weights
    return weights, np.array(bias, copy=True)


def _eval_candidate(
    *,
    config: PongConfig,
    weights: np.ndarray,
    bias: np.ndarray,
    eval_games: int,
    base_seed: int,
    opponent_weights: dict[str, float],
    weak_track_ball_epsilon: float,
    loss_delay_weight: float,
    truncation_value: float,
) -> dict[str, object]:
    by_opponent = {}
    weighted_score_returns = []
    weighted_shaped_proxies = []
    total_weight = 0.0
    for opponent_index, (opponent_policy_id, opponent_weight) in enumerate(
        opponent_weights.items()
    ):
        if opponent_weight <= 0.0:
            continue
        outcomes = []
        for game_index in range(eval_games * len(AGENTS)):
            learner_agent = AGENTS[game_index % len(AGENTS)]
            outcomes.append(
                _run_episode(
                    config=config,
                    weights=weights,
                    bias=bias,
                    learner_agent=learner_agent,
                    opponent_policy_id=opponent_policy_id,
                    episode_seed=base_seed + opponent_index * 100_000 + game_index,
                    policy_seed=base_seed + opponent_index * 10_000 + game_index * 997,
                    weak_track_ball_epsilon=weak_track_ball_epsilon,
                    loss_delay_weight=loss_delay_weight,
                    truncation_value=truncation_value,
                )
            )
        metrics = _rollout_metrics(outcomes)
        by_opponent[opponent_policy_id] = metrics
        weighted_score_returns.append(opponent_weight * float(metrics["mean_score_return"]))
        weighted_shaped_proxies.append(opponent_weight * float(metrics["mean_shaped_proxy"]))
        total_weight += opponent_weight
    mean_score_return = float(sum(weighted_score_returns) / total_weight)
    mean_shaped_proxy = float(sum(weighted_shaped_proxies) / total_weight)
    selection_score = mean_score_return + _SHAPED_SELECTION_TIEBREAK_SCALE * mean_shaped_proxy
    track_ball_metrics = by_opponent.get("track_ball", {})
    return {
        "selection_score": selection_score,
        "selection_mean_score_return": mean_score_return,
        "selection_mean_shaped_proxy": mean_shaped_proxy,
        "selection_shaped_tiebreak_scale": _SHAPED_SELECTION_TIEBREAK_SCALE,
        "opponent_weights": dict(opponent_weights),
        "by_opponent": by_opponent,
        "track_ball_mean_steps": float(track_ball_metrics.get("mean_steps", 0.0)),
        "track_ball_truncation_rate": float(track_ball_metrics.get("truncation_rate", 0.0)),
        "track_ball_win_rate": float(track_ball_metrics.get("win_rate", 0.0)),
    }


def _run_episode(
    *,
    config: PongConfig,
    weights: np.ndarray,
    bias: np.ndarray,
    learner_agent: str,
    opponent_policy_id: str,
    episode_seed: int,
    policy_seed: int,
    weak_track_ball_epsilon: float,
    loss_delay_weight: float,
    truncation_value: float,
) -> EpisodeOutcome:
    env = PongEnv(config)
    observations = env.reset(seed=episode_seed)
    opponent_agent = _opponent(learner_agent)
    opponent_policy = _make_opponent_policy(
        opponent_policy_id,
        weak_track_ball_epsilon=weak_track_ball_epsilon,
        seed=policy_seed + 31,
    )
    opponent_policy.reset(episode_seed, opponent_agent)
    action_counts = [0 for _ in ACTION_LABELS]

    final_step = None
    while True:
        raster_grid = env.raster_observation()
        features = _encode_raster_grid(
            raster_grid,
            expected_shape=(config.height, config.width),
        )
        agent_index = _agent_index(learner_agent)
        learner_probs = _softmax(features @ weights[agent_index] + bias[agent_index])
        learner_action = int(np.argmax(learner_probs))
        opponent_action = int(
            opponent_policy.action(observations[opponent_agent], raster_grid, opponent_agent)
        )
        final_step = env.step(
            {
                learner_agent: learner_action,
                opponent_agent: opponent_action,
            }
        )
        action_counts[learner_action] += 1
        observations = final_step.observations
        if final_step.terminated or final_step.truncated:
            break

    steps = int(next(iter(final_step.observations.values())).step)
    winner = final_step.infos["winner"] if final_step.terminated else None
    score_return = _score_return(
        learner_agent=learner_agent,
        winner=winner,
        truncated=final_step.truncated,
    )
    shaped_proxy = _shaped_proxy(
        score_return=score_return,
        steps=steps,
        max_steps=config.max_steps,
        truncated=final_step.truncated,
        loss_delay_weight=loss_delay_weight,
        truncation_value=truncation_value,
    )
    return EpisodeOutcome(
        opponent_policy_id=opponent_policy_id,
        learner_agent=learner_agent,
        seed=episode_seed,
        steps=steps,
        max_steps=config.max_steps,
        winner=winner,
        truncated=final_step.truncated,
        score_return=score_return,
        shaped_proxy=shaped_proxy,
        action_counts=action_counts,
    )


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
            "median_steps": 0.0,
            "mean_score_return": 0.0,
            "mean_shaped_proxy": 0.0,
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
        "median_steps": float(np.median([outcome.steps for outcome in outcomes])),
        "mean_score_return": float(np.mean([outcome.score_return for outcome in outcomes])),
        "mean_shaped_proxy": float(np.mean([outcome.shaped_proxy for outcome in outcomes])),
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
        return {"episodes": 0, "wins": 0, "losses": 0, "truncations": 0, "mean_steps": 0.0}
    wins = sum(outcome.score_return > 0.0 for outcome in agent_outcomes)
    losses = sum(outcome.score_return < 0.0 for outcome in agent_outcomes)
    return {
        "episodes": len(agent_outcomes),
        "wins": int(wins),
        "losses": int(losses),
        "truncations": int(sum(outcome.truncated for outcome in agent_outcomes)),
        "mean_steps": float(np.mean([outcome.steps for outcome in agent_outcomes])),
        "mean_shaped_proxy": float(np.mean([outcome.shaped_proxy for outcome in agent_outcomes])),
    }


def _score_return(
    *,
    learner_agent: str,
    winner: str | None,
    truncated: bool,
) -> float:
    if truncated or winner is None:
        return 0.0
    return 1.0 if winner == learner_agent else -1.0


def _shaped_proxy(
    *,
    score_return: float,
    steps: int,
    max_steps: int,
    truncated: bool,
    loss_delay_weight: float,
    truncation_value: float,
) -> float:
    if truncated:
        return truncation_value
    if score_return > 0.0:
        return 1.0
    return -1.0 + loss_delay_weight * (steps / max_steps)


def _validated_opponent_weights(opponent_weights: dict[str, float]) -> dict[str, float]:
    validated = {}
    for policy_id, weight in opponent_weights.items():
        if policy_id not in CEM_OPPONENT_POLICY_IDS:
            raise ValueError(f"unknown opponent policy {policy_id!r}")
        if weight < 0.0:
            raise ValueError("opponent weights must be non-negative")
        validated[policy_id] = float(weight)
    if sum(validated.values()) <= 0.0:
        raise ValueError("at least one opponent weight must be positive")
    return validated


def _make_opponent_policy(
    policy_id: str,
    *,
    weak_track_ball_epsilon: float,
    seed: int,
) -> (
    RandomUniformPolicy
    | NoisyTrackBallPolicy
    | LaggedTrackBallPolicy
    | TrackBallPolicy
    | AngleControlPolicy
):
    if policy_id == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    if policy_id == "weak_track_ball":
        return NoisyTrackBallPolicy(epsilon=weak_track_ball_epsilon, seed=seed)
    if policy_id == "lagged_track_ball_1":
        return LaggedTrackBallPolicy(delay=1)
    if policy_id == "track_ball":
        return TrackBallPolicy()
    if policy_id == "angle_control":
        return AngleControlPolicy()
    raise ValueError(f"unknown opponent policy {policy_id!r}")


def _agent_seed(agent: str) -> int:
    return 0 if agent == "player_0" else 10_000


def _candidate_sort_key(
    candidate: dict[str, Any],
    *,
    target_opponent_id: str,
) -> tuple[float, float, float, float, float]:
    metrics = candidate["metrics"]
    by_opponent = metrics["by_opponent"]
    target = by_opponent.get(target_opponent_id, {})
    return (
        float(metrics["selection_mean_score_return"]),
        float(target.get("win_rate", 0.0)),
        float(metrics["selection_score"]),
        float(target.get("truncation_rate", 0.0)),
        float(target.get("mean_steps", 0.0)),
    )


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, object]:
    return {
        "generation": int(candidate["generation"]),
        "candidate_index": int(candidate["candidate_index"]),
        "selection_score": float(candidate["selection_score"]),
        "metrics": candidate["metrics"],
    }


def _selection_rule_summary(
    *,
    opponent_weights: dict[str, float],
    target_opponent_id: str,
    loss_delay_weight: float,
    truncation_value: float,
) -> dict[str, object]:
    return {
        "formula": (
            "selection_score = weighted_mean_score_return + "
            "0.001 * weighted_mean_shaped_proxy; score return is "
            "win=+1.0, loss=-1.0, truncation=0.0; shaped proxy is "
            "win=+1.0, loss=-1.0 + loss_delay_weight * steps/max_steps, "
            "truncation=truncation_value"
        ),
        "sort_order": [
            "weighted_mean_score_return",
            "target_opponent_win_rate",
            "selection_score",
            "target_opponent_truncation_rate",
            "target_opponent_mean_steps",
        ],
        "target_opponent_id": target_opponent_id,
        "opponent_weights": dict(opponent_weights),
        "loss_delay_weight": loss_delay_weight,
        "truncation_value": truncation_value,
        "shaped_tiebreak_scale": _SHAPED_SELECTION_TIEBREAK_SCALE,
        "reported_eval_fields": [
            "wins",
            "losses",
            "truncations",
            "win_rate",
            "truncation_rate",
            "mean_steps",
            "median_steps",
            "mean_score_return",
            "mean_shaped_proxy",
        ],
    }


def _schemas_summary() -> dict[str, object]:
    return {
        "ruleset_id": RULESET_ID,
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "reward_schema_id": REWARD_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "cem_train_summary_schema_id": CEM_TRAIN_SUMMARY_SCHEMA_ID,
        "cem_policy_checkpoint_schema_id": CEM_POLICY_CHECKPOINT_SCHEMA_ID,
        "cem_selection_schema_id": CEM_SELECTION_SCHEMA_ID,
        "feature_encoding_id": FEATURE_ENCODING_ID,
    }


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
        "kind": "curvyzero_dummy_pong_geometry_cem_policy",
        "checkpoint_schema_id": summary["checkpoint_schema_id"],
        "summary_schema_id": summary["summary_schema_id"],
        "selection_schema_id": summary["selection_schema_id"],
        "config": summary["config"],
        "target_opponent_id": summary["target_opponent_id"],
        "seed": summary["seed"],
        "generations": summary["generations"],
        "population_size": summary["population_size"],
        "elite_count": summary["elite_count"],
        "eval_games": summary["eval_games"],
        "max_steps": summary["max_steps"],
        "selection_return": summary["selection_return"],
        "schemas": summary["schemas"],
        "input": summary["input"],
        "model": summary["model"],
        "action_labels": summary["action_labels"],
        "metrics": summary["metrics"],
        "note": summary["note"],
    }


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent {agent!r}")


def _parse_opponent_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        policy_id, separator, weight_text = value.partition("=")
        if not separator:
            raise ValueError("--opponent-weight entries must use POLICY=WEIGHT")
        if policy_id in weights:
            raise ValueError(f"duplicate opponent weight for {policy_id!r}")
        try:
            weights[policy_id] = float(weight_text)
        except ValueError as exc:
            raise ValueError(f"invalid weight for {policy_id!r}: {weight_text!r}") from exc
    return _validated_opponent_weights(weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--paddle-height", type=int, default=3)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=12)
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--initial-sigma", type=float, default=0.75)
    parser.add_argument("--sigma-decay", type=float, default=0.7)
    parser.add_argument("--min-sigma", type=float, default=0.05)
    parser.add_argument("--track-ball-prior-strength", type=float, default=12.0)
    parser.add_argument("--stay-bias", type=float, default=0.75)
    parser.add_argument("--random-opponent-weight", type=float, default=0.25)
    parser.add_argument("--track-ball-opponent-weight", type=float, default=0.75)
    parser.add_argument(
        "--opponent-weight",
        action="append",
        default=None,
        metavar="POLICY=WEIGHT",
        help=(
            "Configured CEM evaluation opponent. May repeat and overrides the "
            "legacy random/track-ball weight flags. Known policies: "
            + ", ".join(CEM_OPPONENT_POLICY_IDS)
        ),
    )
    parser.add_argument("--target-opponent-id", default="track_ball")
    parser.add_argument("--weak-track-ball-epsilon", type=float, default=0.35)
    parser.add_argument("--loss-delay-weight", type=float, default=0.5)
    parser.add_argument("--truncation-value", type=float, default=0.0)
    args = parser.parse_args()
    opponent_weights = None
    if args.opponent_weight is not None:
        try:
            opponent_weights = _parse_opponent_weights(args.opponent_weight)
        except ValueError as exc:
            parser.error(str(exc))

    summary = train_dummy_pong_cem(
        output_dir=args.output_dir,
        seed=args.seed,
        width=args.width,
        height=args.height,
        paddle_height=args.paddle_height,
        generations=args.generations,
        population_size=args.population_size,
        elite_count=args.elite_count,
        eval_games=args.eval_games,
        max_steps=args.max_steps,
        initial_sigma=args.initial_sigma,
        sigma_decay=args.sigma_decay,
        min_sigma=args.min_sigma,
        track_ball_prior_strength=args.track_ball_prior_strength,
        stay_bias=args.stay_bias,
        random_opponent_weight=args.random_opponent_weight,
        track_ball_opponent_weight=args.track_ball_opponent_weight,
        opponent_weights=opponent_weights,
        target_opponent_id=args.target_opponent_id,
        weak_track_ball_epsilon=args.weak_track_ball_epsilon,
        loss_delay_weight=args.loss_delay_weight,
        truncation_value=args.truncation_value,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
