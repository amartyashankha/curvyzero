"""Tiny policy/value trainer for dummy Pong self-play replay."""

from __future__ import annotations

import json
from collections import Counter
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
from curvyzero.training.dummy_pong_imitation_train import DummyPongImitationPolicy
from curvyzero.training.dummy_pong_imitation_train import _agent_index
from curvyzero.training.dummy_pong_imitation_train import _encode_raster_grid
from curvyzero.training.dummy_pong_imitation_train import _feature_dim
from curvyzero.training.dummy_pong_imitation_train import _row_raster_shape
from curvyzero.training.dummy_pong_imitation_train import _softmax_rows
from curvyzero.training.dummy_pong_imitation_train import _split_indices

SELFPLAY_REPLAY_ROW_SCHEMA_ID = "dummy_pong_selfplay_replay_row_v0"
SELFPLAY_TRAIN_SUMMARY_SCHEMA_ID = "dummy_pong_selfplay_train_summary_v0"
SELFPLAY_POLICY_CHECKPOINT_SCHEMA_ID = "dummy_pong_selfplay_policy_checkpoint_v0"
SELFPLAY_TARGET_SCHEMA_ID = "dummy_pong_score_plus_longevity_return_target_v0"
FEATURE_ENCODING_ID = "dummy_pong_raster_one_hot_plus_geometry_v0"


def train_dummy_pong_selfplay(
    *,
    replay_path: Path,
    output_dir: Path | None = None,
    seed: int = 0,
    epochs: int = 100,
    policy_learning_rate: float = 0.1,
    value_learning_rate: float = 0.001,
    action_diversity_beta: float = 0.01,
    validation_fraction: float = 0.2,
    l2: float = 0.0,
    initial_checkpoint: Path | None = None,
    checkpoint_every_epochs: int | None = None,
) -> dict[str, object]:
    """Train a tiny raster policy/value model from Pong self-play rows."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if policy_learning_rate <= 0.0:
        raise ValueError("policy_learning_rate must be positive")
    if value_learning_rate <= 0.0:
        raise ValueError("value_learning_rate must be positive")
    if action_diversity_beta < 0.0:
        raise ValueError("action_diversity_beta must be non-negative")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    if checkpoint_every_epochs is not None and checkpoint_every_epochs < 1:
        raise ValueError("checkpoint_every_epochs must be at least 1 when set")
    if checkpoint_every_epochs is not None and output_dir is None:
        raise ValueError("output_dir is required when checkpoint_every_epochs is set")

    replay_rows_path = _resolve_replay_rows_path(replay_path)
    rows = _load_selfplay_rows(replay_rows_path)
    dataset = _dataset_from_rows(rows)
    rng = np.random.default_rng(seed)
    train_indices, validation_indices = _split_indices(
        row_count=len(rows),
        validation_fraction=validation_fraction,
        rng=rng,
    )

    weights, bias = _initial_policy_parameters(
        initial_checkpoint=initial_checkpoint,
        raster_shape=dataset["raster_shape"],
    )
    value_weights = np.zeros((len(AGENTS), dataset["features"].shape[1]), dtype=np.float64)
    value_bias = np.zeros(len(AGENTS), dtype=np.float64)

    history = []
    periodic_checkpoints = []
    for epoch in range(epochs):
        advantages = _advantages(dataset["shaped_returns"], train_indices=train_indices)
        _train_epoch(
            features=dataset["features"],
            actions=dataset["actions"],
            agent_indices=dataset["agent_indices"],
            shaped_returns=dataset["shaped_returns"],
            advantages=advantages,
            indices=train_indices,
            weights=weights,
            bias=bias,
            value_weights=value_weights,
            value_bias=value_bias,
            policy_learning_rate=policy_learning_rate,
            value_learning_rate=value_learning_rate,
            action_diversity_beta=action_diversity_beta,
            l2=l2,
        )
        completed_epoch = epoch + 1
        if epoch == 0 or epoch == epochs - 1:
            history.append(
                {
                    "epoch": completed_epoch,
                    "train": _metrics(
                        dataset=dataset,
                        indices=train_indices,
                        weights=weights,
                        bias=bias,
                        value_weights=value_weights,
                        value_bias=value_bias,
                    ),
                    "validation": _metrics(
                        dataset=dataset,
                        indices=validation_indices,
                        weights=weights,
                        bias=bias,
                        value_weights=value_weights,
                        value_bias=value_bias,
                    ),
                }
            )
        if (
            output_dir is not None
            and checkpoint_every_epochs is not None
            and completed_epoch % checkpoint_every_epochs == 0
        ):
            checkpoint_path = _periodic_checkpoint_path(output_dir, completed_epoch)
            _write_policy_checkpoint(
                path=checkpoint_path,
                weights=weights,
                bias=bias,
                value_weights=value_weights,
                value_bias=value_bias,
                raster_shape=dataset["raster_shape"],
                metadata=_checkpoint_metadata(
                    replay_rows_path=replay_rows_path,
                    seed=seed,
                    epochs=epochs,
                    completed_epochs=completed_epoch,
                    policy_learning_rate=policy_learning_rate,
                    value_learning_rate=value_learning_rate,
                    action_diversity_beta=action_diversity_beta,
                    validation_fraction=validation_fraction,
                    l2=l2,
                    initial_checkpoint=initial_checkpoint,
                    dataset=dataset,
                ),
            )
            periodic_checkpoints.append(
                {
                    "epoch": completed_epoch,
                    "path": str(checkpoint_path),
                }
            )

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_selfplay_training",
        "note": (
            "Tiny NumPy policy/value update from dummy Pong self-play rows. "
            "This is the first self-play training scaffold, not MuZero search."
        ),
        "summary_schema_id": SELFPLAY_TRAIN_SUMMARY_SCHEMA_ID,
        "checkpoint_schema_id": SELFPLAY_POLICY_CHECKPOINT_SCHEMA_ID,
        "target_schema_id": SELFPLAY_TARGET_SCHEMA_ID,
        "seed": seed,
        "source_replay_path": str(replay_rows_path),
        "epochs": epochs,
        "policy_learning_rate": policy_learning_rate,
        "value_learning_rate": value_learning_rate,
        "action_diversity_beta": action_diversity_beta,
        "validation_fraction": validation_fraction,
        "l2": l2,
        "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
        "schemas": _schemas_summary(),
        "input": {
            "raster_shape": list(dataset["raster_shape"]),
            "feature_shape": [int(dataset["features"].shape[1])],
            "feature_encoding": FEATURE_ENCODING_ID,
            "raster_legend": RASTER_LEGEND,
        },
        "target_construction": {
            "score_return_field": "score_return",
            "training_target_field": "shaped_return",
            "training_target_rule": (
                "win => +1.0; loss => -1.0 + 0.5 * episode_steps / max_steps; "
                "timeout/draw => 0.0"
            ),
            "policy_update": (
                "Increase actions from above-average shaped-return rows and "
                "decrease actions from below-average shaped-return rows. "
                "A small action-diversity term pushes logits away from collapse."
            ),
        },
        "model": {
            "type": "per_ego_agent_linear_softmax_policy_plus_linear_value_numpy",
            "weights_shape": list(weights.shape),
            "bias_shape": list(bias.shape),
            "value_weights_shape": list(value_weights.shape),
            "value_bias_shape": list(value_bias.shape),
            "ego_agents": list(AGENTS),
        },
        "action_labels": list(ACTION_LABELS),
        "data": _data_summary(dataset, train_indices, validation_indices),
        "metrics": {
            "train": _metrics(
                dataset=dataset,
                indices=train_indices,
                weights=weights,
                bias=bias,
                value_weights=value_weights,
                value_bias=value_bias,
            ),
            "validation": _metrics(
                dataset=dataset,
                indices=validation_indices,
                weights=weights,
                bias=bias,
                value_weights=value_weights,
                value_bias=value_bias,
            ),
            "all_rows": _metrics(
                dataset=dataset,
                indices=np.arange(len(rows)),
                weights=weights,
                bias=bias,
                value_weights=value_weights,
                value_bias=value_bias,
            ),
        },
        "history": history,
        "plain_language": {
            "proves": (
                "Self-play replay can update a visual policy/value checkpoint "
                "that the existing Pong scoreboard can load."
            ),
            "does_not_prove": (
                "This does not prove MuZero search, strong Pong play, or that "
                "the policy beats track_ball. The scoreboard must check that."
            ),
        },
    }
    if checkpoint_every_epochs is not None:
        summary["checkpoints"] = {
            "periodic_enabled": True,
            "every_epochs": checkpoint_every_epochs,
            "count": len(periodic_checkpoints),
            "refs": periodic_checkpoints,
            "latest": periodic_checkpoints[-1] if periodic_checkpoints else None,
        }

    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(
            artifacts=artifacts,
            summary=summary,
            weights=weights,
            bias=bias,
            value_weights=value_weights,
            value_bias=value_bias,
            raster_shape=dataset["raster_shape"],
        )
    return summary


def _resolve_replay_rows_path(replay_path: Path) -> Path:
    if replay_path.is_dir():
        replay_path = replay_path / "replay_rows.jsonl"
    if not replay_path.exists():
        raise FileNotFoundError(f"replay rows not found: {replay_path}")
    return replay_path


def _load_selfplay_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("schema_id") != SELFPLAY_REPLAY_ROW_SCHEMA_ID:
                raise ValueError(
                    f"{path}:{line_number} has unexpected schema_id {row.get('schema_id')!r}"
                )
            _validate_row(row, path=path, line_number=line_number)
            rows.append(row)
    if not rows:
        raise ValueError(f"replay rows file is empty: {path}")
    return rows


def _validate_row(row: dict[str, Any], *, path: Path, line_number: int) -> None:
    required_fields = (
        "schema_id",
        "raster_observation_schema_id",
        "raster_shape",
        "raster_grid",
        "ego_agent",
        "behavior_action_id",
        "behavior_action_label",
        "joint_action_by_agent",
        "reward_after_step",
        "score_return",
        "shaped_return",
    )
    missing = [field for field in required_fields if field not in row]
    if missing:
        raise ValueError(f"{path}:{line_number} missing required fields {missing!r}")
    if row["raster_observation_schema_id"] != RASTER_OBSERVATION_SCHEMA_ID:
        raise ValueError(
            f"{path}:{line_number} has unexpected raster_observation_schema_id "
            f"{row['raster_observation_schema_id']!r}"
        )
    ego_agent = str(row["ego_agent"])
    _agent_index(ego_agent)
    action_id = _action_id(row["behavior_action_id"], path=path, line_number=line_number)
    if row["behavior_action_label"] != ACTION_LABELS[action_id]:
        raise ValueError(f"{path}:{line_number} behavior_action_label does not match action id")
    joint_action_by_agent = row["joint_action_by_agent"]
    if not isinstance(joint_action_by_agent, dict) or set(joint_action_by_agent) != set(AGENTS):
        raise ValueError(f"{path}:{line_number} joint_action_by_agent must contain both agents")
    if int(joint_action_by_agent[ego_agent]["action_id"]) != action_id:
        raise ValueError(f"{path}:{line_number} ego joint action does not match behavior action")
    raster_shape = _row_raster_shape(row)
    _encode_raster_grid(row["raster_grid"], expected_shape=raster_shape)
    float(row["reward_after_step"])
    float(row["score_return"])
    float(row["shaped_return"])


def _dataset_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raster_shape = _row_raster_shape(rows[0])
    features = []
    actions = []
    agent_indices = []
    shaped_returns = []
    score_returns = []
    rows_by_ego_agent = Counter()
    action_histogram_by_agent = {agent: Counter() for agent in AGENTS}
    policy_ids = set()
    game_keys = set()

    for row in rows:
        if _row_raster_shape(row) != raster_shape:
            raise ValueError("self-play replay has mixed raster shapes")
        ego_agent = str(row["ego_agent"])
        action_id = int(row["behavior_action_id"])
        features.append(_encode_raster_grid(row["raster_grid"], expected_shape=raster_shape))
        actions.append(action_id)
        agent_indices.append(_agent_index(ego_agent))
        shaped_returns.append(float(row["shaped_return"]))
        score_returns.append(float(row["score_return"]))
        rows_by_ego_agent[ego_agent] += 1
        action_histogram_by_agent[ego_agent][ACTION_LABELS[action_id]] += 1
        policy_ids.add(str(row.get("behavior_policy_id", row.get("target_policy_id", "unknown"))))
        game_keys.add((row.get("run_id"), row.get("game_index"), ego_agent))

    shaped = np.asarray(shaped_returns, dtype=np.float64)
    score = np.asarray(score_returns, dtype=np.float64)
    return {
        "features": np.vstack(features),
        "actions": np.asarray(actions, dtype=np.int64),
        "agent_indices": np.asarray(agent_indices, dtype=np.int64),
        "shaped_returns": shaped,
        "score_returns": score,
        "raster_shape": raster_shape,
        "rows_by_ego_agent": {agent: int(rows_by_ego_agent[agent]) for agent in AGENTS},
        "action_histogram_by_agent": {
            agent: {label: int(action_histogram_by_agent[agent][label]) for label in ACTION_LABELS}
            for agent in AGENTS
        },
        "behavior_policy_ids": sorted(policy_ids),
        "return_groups": len(game_keys),
        "score_return_stats": _stats(score),
        "shaped_return_stats": _stats(shaped),
    }


def _initial_policy_parameters(
    *,
    initial_checkpoint: Path | None,
    raster_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (len(AGENTS), _feature_dim(raster_shape), len(ACTION_LABELS))
    if initial_checkpoint is None:
        return (
            np.zeros(expected_shape, dtype=np.float64),
            np.zeros((len(AGENTS), len(ACTION_LABELS)), dtype=np.float64),
        )
    policy = DummyPongImitationPolicy.load_checkpoint(initial_checkpoint)
    if policy.raster_shape != raster_shape:
        raise ValueError(
            f"initial checkpoint raster shape {policy.raster_shape!r} "
            f"does not match replay shape {raster_shape!r}"
        )
    if policy.weights.shape != expected_shape:
        raise ValueError(f"unexpected initial checkpoint weight shape {policy.weights.shape!r}")
    return np.array(policy.weights, copy=True), np.array(policy.bias, copy=True)


def _advantages(shaped_returns: np.ndarray, *, train_indices: np.ndarray) -> np.ndarray:
    baseline = float(np.mean(shaped_returns[train_indices]))
    advantages = shaped_returns - baseline
    scale = float(np.std(advantages[train_indices]))
    if scale > 1e-8:
        advantages = advantages / scale
    return np.clip(advantages, -2.0, 2.0)


def _train_epoch(
    *,
    features: np.ndarray,
    actions: np.ndarray,
    agent_indices: np.ndarray,
    shaped_returns: np.ndarray,
    advantages: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    value_weights: np.ndarray,
    value_bias: np.ndarray,
    policy_learning_rate: float,
    value_learning_rate: float,
    action_diversity_beta: float,
    l2: float,
) -> None:
    row_count = float(len(indices))
    policy_weight_grad = np.zeros_like(weights)
    policy_bias_grad = np.zeros_like(bias)
    value_weight_grad = np.zeros_like(value_weights)
    value_bias_grad = np.zeros_like(value_bias)

    for agent_index in range(len(AGENTS)):
        agent_rows = indices[agent_indices[indices] == agent_index]
        if len(agent_rows) == 0:
            continue
        agent_features = features[agent_rows]
        probs = _softmax_rows(agent_features @ weights[agent_index] + bias[agent_index])
        policy_grad = probs.copy()
        policy_grad[np.arange(len(agent_rows)), actions[agent_rows]] -= 1.0
        policy_grad *= advantages[agent_rows, None]
        if action_diversity_beta:
            uniform_probs = np.full_like(probs, 1.0 / len(ACTION_LABELS))
            policy_grad += action_diversity_beta * (probs - uniform_probs)
        policy_weight_grad[agent_index] = agent_features.T @ policy_grad
        policy_bias_grad[agent_index] = np.sum(policy_grad, axis=0)

        values = agent_features @ value_weights[agent_index] + value_bias[agent_index]
        value_error = values - shaped_returns[agent_rows]
        value_weight_grad[agent_index] = agent_features.T @ value_error
        value_bias_grad[agent_index] = float(np.sum(value_error))

    if l2:
        policy_weight_grad += l2 * weights
        value_weight_grad += l2 * value_weights
    weights -= policy_learning_rate * policy_weight_grad / row_count
    bias -= policy_learning_rate * policy_bias_grad / row_count
    value_weights -= value_learning_rate * value_weight_grad / row_count
    value_bias -= value_learning_rate * value_bias_grad / row_count


def _metrics(
    *,
    dataset: dict[str, Any],
    indices: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    value_weights: np.ndarray,
    value_bias: np.ndarray,
) -> dict[str, object]:
    if len(indices) == 0:
        return {
            "rows": 0,
            "policy_loss": None,
            "value_mse": None,
            "mean_score_return": None,
            "mean_shaped_return": None,
        }

    policy_losses = []
    value_errors = []
    action_counts = Counter()
    correct_sign_actions = 0
    advantages = dataset["shaped_returns"] - float(np.mean(dataset["shaped_returns"][indices]))
    for agent_index in range(len(AGENTS)):
        agent_rows = indices[dataset["agent_indices"][indices] == agent_index]
        if len(agent_rows) == 0:
            continue
        features = dataset["features"][agent_rows]
        probs = _softmax_rows(features @ weights[agent_index] + bias[agent_index])
        chosen_probs = np.maximum(probs[np.arange(len(agent_rows)), dataset["actions"][agent_rows]], 1e-12)
        policy_losses.extend(float(item) for item in -advantages[agent_rows] * np.log(chosen_probs))
        predictions = np.argmax(probs, axis=1)
        for predicted in predictions:
            action_counts[ACTION_LABELS[int(predicted)]] += 1
        correct_sign_actions += int(np.sum(predictions == dataset["actions"][agent_rows]))
        values = features @ value_weights[agent_index] + value_bias[agent_index]
        value_errors.extend(float(item) for item in values - dataset["shaped_returns"][agent_rows])

    return {
        "rows": int(len(indices)),
        "policy_loss": float(np.mean(policy_losses)),
        "behavior_action_match_rate": float(correct_sign_actions / len(indices)),
        "value_mse": float(np.mean(np.square(value_errors))),
        "mean_score_return": float(np.mean(dataset["score_returns"][indices])),
        "mean_shaped_return": float(np.mean(dataset["shaped_returns"][indices])),
        "mean_policy_entropy": _mean_policy_entropy(
            dataset=dataset,
            indices=indices,
            weights=weights,
            bias=bias,
        ),
        "predicted_action_histogram": {
            label: int(action_counts[label])
            for label in ACTION_LABELS
        },
    }


def _data_summary(
    dataset: dict[str, Any],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, object]:
    return {
        "rows": int(len(dataset["actions"])),
        "train_rows": int(len(train_indices)),
        "validation_rows": int(len(validation_indices)),
        "return_groups": int(dataset["return_groups"]),
        "behavior_policy_ids": dataset["behavior_policy_ids"],
        "rows_by_ego_agent": dataset["rows_by_ego_agent"],
        "action_histogram_by_agent": dataset["action_histogram_by_agent"],
        "score_return_stats": dataset["score_return_stats"],
        "shaped_return_stats": dataset["shaped_return_stats"],
    }


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "checkpoint_npz": output_dir / "checkpoint.npz",
    }


def _periodic_checkpoint_path(output_dir: Path, completed_epoch: int) -> Path:
    return output_dir / "checkpoints" / f"epoch-{completed_epoch:06d}" / "checkpoint.npz"


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    weights: np.ndarray,
    bias: np.ndarray,
    value_weights: np.ndarray,
    value_bias: np.ndarray,
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
        value_weights=value_weights,
        value_bias=value_bias,
        raster_shape=raster_shape,
        metadata=_checkpoint_metadata_from_summary(summary),
    )


def _write_policy_checkpoint(
    *,
    path: Path,
    weights: np.ndarray,
    bias: np.ndarray,
    value_weights: np.ndarray,
    value_bias: np.ndarray,
    raster_shape: tuple[int, int],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weights=weights,
        bias=bias,
        value_weights=value_weights,
        value_bias=value_bias,
        raster_shape=np.asarray(raster_shape, dtype=np.int64),
        action_labels=np.asarray(ACTION_LABELS),
        ego_agents=np.asarray(AGENTS),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _checkpoint_metadata(
    *,
    replay_rows_path: Path,
    seed: int,
    epochs: int,
    completed_epochs: int,
    policy_learning_rate: float,
    value_learning_rate: float,
    action_diversity_beta: float,
    validation_fraction: float,
    l2: float,
    initial_checkpoint: Path | None,
    dataset: dict[str, Any],
) -> dict[str, object]:
    return {
        "kind": "curvyzero_dummy_pong_selfplay_policy_checkpoint",
        "checkpoint_schema_id": SELFPLAY_POLICY_CHECKPOINT_SCHEMA_ID,
        "target_schema_id": SELFPLAY_TARGET_SCHEMA_ID,
        "source_replay_path": str(replay_rows_path),
        "seed": seed,
        "epochs": epochs,
        "completed_epochs": completed_epochs,
        "policy_learning_rate": policy_learning_rate,
        "value_learning_rate": value_learning_rate,
        "action_diversity_beta": action_diversity_beta,
        "validation_fraction": validation_fraction,
        "l2": l2,
        "initial_checkpoint": None if initial_checkpoint is None else str(initial_checkpoint),
        "schemas": _schemas_summary(),
        "input_summary": {
            "rows": int(len(dataset["actions"])),
            "raster_shape": list(dataset["raster_shape"]),
            "shaped_return_stats": dataset["shaped_return_stats"],
        },
    }


def _checkpoint_metadata_from_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "curvyzero_dummy_pong_selfplay_policy_checkpoint",
        "checkpoint_schema_id": SELFPLAY_POLICY_CHECKPOINT_SCHEMA_ID,
        "target_schema_id": SELFPLAY_TARGET_SCHEMA_ID,
        "summary_schema_id": SELFPLAY_TRAIN_SUMMARY_SCHEMA_ID,
        "source_replay_path": summary["source_replay_path"],
        "seed": summary["seed"],
        "epochs": summary["epochs"],
        "completed_epochs": summary["epochs"],
        "policy_learning_rate": summary["policy_learning_rate"],
        "value_learning_rate": summary["value_learning_rate"],
        "action_diversity_beta": summary["action_diversity_beta"],
        "validation_fraction": summary["validation_fraction"],
        "l2": summary["l2"],
        "initial_checkpoint": summary["initial_checkpoint"],
        "schemas": summary["schemas"],
        "input_summary": summary["data"],
    }


def _schemas_summary() -> dict[str, object]:
    return {
        "ruleset_id": RULESET_ID,
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "reward_schema_id": REWARD_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "selfplay_replay_row_schema_id": SELFPLAY_REPLAY_ROW_SCHEMA_ID,
        "selfplay_train_summary_schema_id": SELFPLAY_TRAIN_SUMMARY_SCHEMA_ID,
        "selfplay_policy_checkpoint_schema_id": SELFPLAY_POLICY_CHECKPOINT_SCHEMA_ID,
        "selfplay_target_schema_id": SELFPLAY_TARGET_SCHEMA_ID,
        "feature_encoding_id": FEATURE_ENCODING_ID,
    }


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _mean_policy_entropy(
    *,
    dataset: dict[str, Any],
    indices: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> float:
    entropies = []
    for agent_index in range(len(AGENTS)):
        agent_rows = indices[dataset["agent_indices"][indices] == agent_index]
        if len(agent_rows) == 0:
            continue
        probs = _softmax_rows(dataset["features"][agent_rows] @ weights[agent_index] + bias[agent_index])
        row_entropies = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        entropies.extend(float(item) for item in row_entropies)
    return float(np.mean(entropies)) if entropies else 0.0


def _action_id(value: object, *, path: Path, line_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path}:{line_number} action id must be an integer")
    action_id = int(value)
    if not 0 <= action_id < len(ACTION_LABELS):
        raise ValueError(f"{path}:{line_number} action id out of range: {action_id!r}")
    return action_id
