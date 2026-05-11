"""Tiny value-target learner for dummy Pong scoring replay."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.training.dummy_pong import AGENTS
from curvyzero.training.dummy_pong import RASTER_LEGEND
from curvyzero.training.dummy_pong import RASTER_OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import REWARD_SCHEMA_ID
from curvyzero.training.dummy_pong import RULESET_ID
from curvyzero.training.dummy_pong_imitation_train import _agent_index
from curvyzero.training.dummy_pong_imitation_train import _encode_raster_grid
from curvyzero.training.dummy_pong_imitation_train import _npz_text
from curvyzero.training.dummy_pong_imitation_train import _row_raster_shape
from curvyzero.training.dummy_pong_imitation_train import _split_indices
from curvyzero.training.dummy_pong_scoring_replay import SCORING_REPLAY_ROW_SCHEMA_ID

VALUE_TRAIN_SUMMARY_SCHEMA_ID = "dummy_pong_value_train_summary_v0"
VALUE_CHECKPOINT_SCHEMA_ID = "dummy_pong_value_checkpoint_v0"
VALUE_TARGET_SCHEMA_ID = "dummy_pong_score_delta_return_target_v0"
VALUE_FEATURE_ENCODING_ID = "dummy_pong_raster_one_hot_plus_geometry_plus_ego_v0"


class DummyPongValueRegressor:
    """Linear scalar value regressor over raster features plus ego-agent id."""

    def __init__(
        self,
        *,
        weights: np.ndarray,
        bias: float,
        raster_shape: tuple[int, int],
        ego_agents: tuple[str, ...],
        metadata: dict[str, object],
    ):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.raster_shape = raster_shape
        self.ego_agents = ego_agents
        self.metadata = dict(metadata)
        expected_shape = (_feature_dim(raster_shape),)
        if self.weights.shape != expected_shape:
            raise ValueError(f"unexpected weight shape {self.weights.shape!r}")
        if self.ego_agents != AGENTS:
            raise ValueError(f"unexpected ego agents {self.ego_agents!r}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> "DummyPongValueRegressor":
        """Load a saved dummy Pong value checkpoint."""

        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(_npz_text(payload["metadata"].item()))
            raster_shape = tuple(int(item) for item in payload["raster_shape"].tolist())
            ego_agents = tuple(_npz_text(item) for item in payload["ego_agents"])
            return cls(
                weights=np.asarray(payload["weights"], dtype=np.float64),
                bias=float(payload["bias"]),
                raster_shape=(raster_shape[0], raster_shape[1]),
                ego_agents=ego_agents,
                metadata=metadata,
            )

    def predict_value(self, raster_grid: list[str] | np.ndarray, ego_agent: str) -> float:
        """Predict target_return for one ego raster observation."""

        features = _encode_value_features(
            raster_grid,
            ego_agent=ego_agent,
            expected_shape=self.raster_shape,
        )
        return float(features @ self.weights + self.bias)


def train_dummy_pong_value(
    *,
    replay_path: Path,
    output_dir: Path | None = None,
    seed: int = 0,
    validation_fraction: float = 0.2,
    discount: float = 1.0,
    ridge_l2: float = 1e-6,
) -> dict[str, object]:
    """Train a deterministic linear value regressor from scoring replay rows."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    if discount < 0.0:
        raise ValueError("discount must be non-negative")
    if ridge_l2 < 0.0:
        raise ValueError("ridge_l2 must be non-negative")

    replay_rows_path = _resolve_replay_rows_path(replay_path)
    rows = _load_scoring_rows(replay_rows_path)
    target_returns = _target_returns_by_game_and_ego(rows, discount=discount)
    dataset = _dataset_from_rows(rows, target_returns)
    rng = np.random.default_rng(seed)
    train_indices, validation_indices = _split_indices(
        row_count=len(rows),
        validation_fraction=validation_fraction,
        rng=rng,
    )

    weights, bias = _fit_ridge(
        features=dataset["features"],
        targets=dataset["target_returns"],
        indices=train_indices,
        ridge_l2=ridge_l2,
    )

    train_metrics = _metrics(
        features=dataset["features"],
        targets=dataset["target_returns"],
        indices=train_indices,
        weights=weights,
        bias=bias,
    )
    validation_metrics = _metrics(
        features=dataset["features"],
        targets=dataset["target_returns"],
        indices=validation_indices,
        weights=weights,
        bias=bias,
    )
    all_metrics = _metrics(
        features=dataset["features"],
        targets=dataset["target_returns"],
        indices=np.arange(len(rows)),
        weights=weights,
        bias=bias,
    )

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_value_training",
        "note": (
            "Tiny deterministic NumPy value regressor trained from score-delta "
            "return targets in dummy Pong scoring replay. This proves target "
            "construction and value-fitting plumbing only; it is not policy "
            "improvement, MuZero, self-play, or action cloning."
        ),
        "summary_schema_id": VALUE_TRAIN_SUMMARY_SCHEMA_ID,
        "checkpoint_schema_id": VALUE_CHECKPOINT_SCHEMA_ID,
        "target_schema_id": VALUE_TARGET_SCHEMA_ID,
        "seed": seed,
        "source_replay_path": str(replay_rows_path),
        "validation_fraction": validation_fraction,
        "discount": discount,
        "ridge_l2": ridge_l2,
        "schemas": {
            "ruleset_id": RULESET_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "scoring_replay_row_schema_id": SCORING_REPLAY_ROW_SCHEMA_ID,
            "value_train_summary_schema_id": VALUE_TRAIN_SUMMARY_SCHEMA_ID,
            "value_checkpoint_schema_id": VALUE_CHECKPOINT_SCHEMA_ID,
            "value_target_schema_id": VALUE_TARGET_SCHEMA_ID,
            "feature_encoding_id": VALUE_FEATURE_ENCODING_ID,
        },
        "target_construction": {
            "target_field": "target_return",
            "group_by": ["game_index", "ego_agent"],
            "backup": "target_return[t] = reward_after_step[t] + discount * target_return[t+1]",
            "discount": discount,
            "source_reward_field": "reward_after_step",
        },
        "input": {
            "raster_shape": list(dataset["raster_shape"]),
            "feature_shape": [int(dataset["features"].shape[1])],
            "feature_encoding": VALUE_FEATURE_ENCODING_ID,
            "raster_legend": RASTER_LEGEND,
        },
        "model": {
            "type": "linear_ridge_numpy_scalar_value",
            "weights_shape": list(weights.shape),
            "bias_shape": [],
            "ego_agents": list(AGENTS),
        },
        "data": {
            "rows": len(rows),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "return_groups": dataset["return_groups"],
            "row_schema_ids": sorted(dataset["row_schema_ids"]),
            "behavior_policy_ids": sorted(dataset["behavior_policy_ids"]),
            "reward_values": dataset["reward_values"],
            "reward_stats": dataset["reward_stats"],
            "target_return_stats": dataset["target_return_stats"],
            "nonzero_reward_rows": dataset["nonzero_reward_rows"],
            "positive_reward_rows": dataset["positive_reward_rows"],
            "negative_reward_rows": dataset["negative_reward_rows"],
            "terminated_rows": dataset["terminated_rows"],
            "truncated_rows": dataset["truncated_rows"],
            "rows_by_ego_agent": dataset["rows_by_ego_agent"],
            "rows_by_behavior_policy_id": dataset["rows_by_behavior_policy_id"],
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "all_rows": all_metrics,
        },
        "plain_language": {
            "proves": (
                "Scoring replay rows can be grouped by game and ego agent, backed up "
                "into scalar score-delta returns, fit by a tiny raster value model, "
                "saved, and reloaded."
            ),
            "does_not_prove": (
                "This does not improve a policy, search with MuZero, collect self-play, "
                "or tell the agent which action to take."
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
            raster_shape=dataset["raster_shape"],
        )
    return summary


def _resolve_replay_rows_path(replay_path: Path) -> Path:
    if replay_path.is_dir():
        replay_path = replay_path / "replay_rows.jsonl"
    if not replay_path.exists():
        raise FileNotFoundError(f"replay rows not found: {replay_path}")
    return replay_path


def _load_scoring_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("schema_id") != SCORING_REPLAY_ROW_SCHEMA_ID:
                raise ValueError(
                    f"{path}:{line_number} has unexpected schema_id {row.get('schema_id')!r}"
                )
            _validate_scoring_row(row, path=path, line_number=line_number)
            rows.append(row)
    if not rows:
        raise ValueError(f"replay rows file is empty: {path}")
    return rows


def _validate_scoring_row(row: dict[str, Any], *, path: Path, line_number: int) -> None:
    required_fields = (
        "schema_id",
        "game_index",
        "step_index",
        "ego_agent",
        "behavior_policy_id",
        "raster_observation_schema_id",
        "raster_shape",
        "raster_grid",
        "reward_after_step",
        "terminated",
        "truncated",
    )
    missing_fields = [field for field in required_fields if field not in row]
    if missing_fields:
        raise ValueError(f"{path}:{line_number} missing required fields {missing_fields!r}")
    if row["raster_observation_schema_id"] != RASTER_OBSERVATION_SCHEMA_ID:
        raise ValueError(
            f"{path}:{line_number} has unexpected raster_observation_schema_id "
            f"{row['raster_observation_schema_id']!r}"
        )
    _agent_index(str(row["ego_agent"]))
    int(row["game_index"])
    int(row["step_index"])
    float(row["reward_after_step"])
    raster_shape = _row_raster_shape(row)
    _encode_raster_grid(row["raster_grid"], expected_shape=raster_shape)


def _target_returns_by_game_and_ego(
    rows: list[dict[str, Any]],
    *,
    discount: float,
) -> np.ndarray:
    target_returns = np.zeros(len(rows), dtype=np.float64)
    grouped_indices: dict[tuple[int, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (int(row["game_index"]), str(row["ego_agent"]))
        grouped_indices.setdefault(key, []).append(index)

    for indices in grouped_indices.values():
        running_return = 0.0
        ordered_indices = sorted(indices, key=lambda index: (int(rows[index]["step_index"]), index))
        for index in reversed(ordered_indices):
            running_return = float(rows[index]["reward_after_step"]) + discount * running_return
            target_returns[index] = running_return
    return target_returns


def _dataset_from_rows(rows: list[dict[str, Any]], target_returns: np.ndarray) -> dict[str, Any]:
    first_shape = _row_raster_shape(rows[0])
    features = []
    row_schema_ids = set()
    behavior_policy_ids = set()
    reward_values = set()
    rewards = []
    nonzero_reward_rows = 0
    positive_reward_rows = 0
    negative_reward_rows = 0
    terminated_rows = 0
    truncated_rows = 0
    rows_by_ego_agent = Counter()
    rows_by_behavior_policy_id = Counter()
    return_group_keys = set()

    for row in rows:
        raster_shape = _row_raster_shape(row)
        if raster_shape != first_shape:
            raise ValueError(f"mixed raster shapes: {first_shape!r} and {raster_shape!r}")
        ego_agent = str(row["ego_agent"])
        behavior_policy_id = str(row["behavior_policy_id"])
        reward = float(row["reward_after_step"])

        features.append(
            _encode_value_features(
                row["raster_grid"],
                ego_agent=ego_agent,
                expected_shape=first_shape,
            )
        )
        row_schema_ids.add(str(row["schema_id"]))
        behavior_policy_ids.add(behavior_policy_id)
        reward_values.add(reward)
        rewards.append(reward)
        if reward != 0.0:
            nonzero_reward_rows += 1
        if reward > 0.0:
            positive_reward_rows += 1
        if reward < 0.0:
            negative_reward_rows += 1
        if bool(row.get("terminated", row.get("terminated_after_step", False))):
            terminated_rows += 1
        if bool(row.get("truncated", row.get("truncated_after_step", False))):
            truncated_rows += 1
        rows_by_ego_agent[ego_agent] += 1
        rows_by_behavior_policy_id[behavior_policy_id] += 1
        return_group_keys.add((int(row["game_index"]), ego_agent))

    return {
        "features": np.vstack(features),
        "target_returns": target_returns,
        "raster_shape": first_shape,
        "return_groups": len(return_group_keys),
        "row_schema_ids": row_schema_ids,
        "behavior_policy_ids": behavior_policy_ids,
        "reward_values": sorted(float(value) for value in reward_values),
        "reward_stats": _stats(np.asarray(rewards, dtype=np.float64)),
        "target_return_stats": _stats(target_returns),
        "nonzero_reward_rows": nonzero_reward_rows,
        "positive_reward_rows": positive_reward_rows,
        "negative_reward_rows": negative_reward_rows,
        "terminated_rows": terminated_rows,
        "truncated_rows": truncated_rows,
        "rows_by_ego_agent": {
            agent: int(rows_by_ego_agent[agent])
            for agent in AGENTS
        },
        "rows_by_behavior_policy_id": {
            policy_id: int(rows_by_behavior_policy_id[policy_id])
            for policy_id in sorted(rows_by_behavior_policy_id)
        },
    }


def _encode_value_features(
    raster_grid: list[str] | np.ndarray,
    *,
    ego_agent: str,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    ego_one_hot = np.zeros(len(AGENTS), dtype=np.float64)
    ego_one_hot[_agent_index(ego_agent)] = 1.0
    return np.concatenate(
        [
            _encode_raster_grid(raster_grid, expected_shape=expected_shape),
            ego_one_hot,
        ]
    )


def _fit_ridge(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    ridge_l2: float,
) -> tuple[np.ndarray, float]:
    train_features = features[indices]
    train_targets = targets[indices]
    design = np.column_stack(
        [
            train_features,
            np.ones(len(train_features), dtype=np.float64),
        ]
    )
    gram = design.T @ design
    penalty = ridge_l2 * np.eye(gram.shape[0], dtype=np.float64)
    penalty[-1, -1] = 0.0
    rhs = design.T @ train_targets
    try:
        params = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        params = np.linalg.pinv(gram + penalty) @ rhs
    return np.asarray(params[:-1], dtype=np.float64), float(params[-1])


def _metrics(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> dict[str, object]:
    if len(indices) == 0:
        return {
            "rows": 0,
            "mse": None,
            "rmse": None,
            "mae": None,
            "prediction_stats": None,
            "target_stats": None,
        }
    predictions = features[indices] @ weights + bias
    residuals = predictions - targets[indices]
    mse = float(np.mean(residuals * residuals))
    return {
        "rows": int(len(indices)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residuals))),
        "prediction_stats": _stats(predictions),
        "target_stats": _stats(targets[indices]),
    }


def _stats(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    positive = int(np.sum(values > 0.0))
    negative = int(np.sum(values < 0.0))
    zero = int(np.sum(values == 0.0))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_count": positive,
        "zero_count": zero,
        "negative_count": negative,
    }


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "checkpoint_npz": output_dir / "checkpoint.npz",
    }


def _feature_dim(raster_shape: tuple[int, int]) -> int:
    raster_values = len(RASTER_LEGEND)
    return raster_shape[0] * raster_shape[1] * raster_values + 6 + len(AGENTS)


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    weights: np.ndarray,
    bias: float,
    raster_shape: tuple[int, int],
) -> None:
    summary_path = artifacts["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        artifacts["checkpoint_npz"],
        weights=weights,
        bias=np.asarray(bias, dtype=np.float64),
        raster_shape=np.asarray(raster_shape, dtype=np.int64),
        ego_agents=np.asarray(AGENTS),
        metadata=np.asarray(json.dumps(_checkpoint_metadata(summary), sort_keys=True)),
    )


def _checkpoint_metadata(summary: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "curvyzero_dummy_pong_value_regressor",
        "checkpoint_schema_id": summary["checkpoint_schema_id"],
        "summary_schema_id": summary["summary_schema_id"],
        "target_schema_id": summary["target_schema_id"],
        "seed": summary["seed"],
        "source_replay_path": summary["source_replay_path"],
        "validation_fraction": summary["validation_fraction"],
        "discount": summary["discount"],
        "ridge_l2": summary["ridge_l2"],
        "schemas": summary["schemas"],
        "target_construction": summary["target_construction"],
        "input": summary["input"],
        "model": summary["model"],
        "metrics": summary["metrics"],
        "note": summary["note"],
    }
