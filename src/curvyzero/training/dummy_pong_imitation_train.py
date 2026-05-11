"""Tiny supervised raster imitation learner for dummy Pong."""

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
from curvyzero.training.dummy_pong import RULESET_ID
from curvyzero.training.dummy_pong_lookahead_replay import LOOKAHEAD_REPLAY_ROW_SCHEMA_ID
from curvyzero.training.dummy_pong_replay import IMITATION_REPLAY_ROW_SCHEMA_ID
from curvyzero.training.dummy_pong_scoring_replay import SCORING_REPLAY_ROW_SCHEMA_ID

IMITATION_TRAIN_SUMMARY_SCHEMA_ID = "dummy_pong_imitation_train_summary_v0"
IMITATION_POLICY_CHECKPOINT_SCHEMA_ID = "dummy_pong_imitation_policy_checkpoint_v0"
IMITATION_MLP_POLICY_CHECKPOINT_SCHEMA_ID = "dummy_pong_imitation_mlp_policy_checkpoint_v0"
FEATURE_ENCODING_ID = "dummy_pong_raster_one_hot_plus_geometry_v0"
RASTER_ONLY_FEATURE_ENCODING_ID = "dummy_pong_raster_one_hot_v0"
FRAME_STACK_FEATURE_ENCODING_ID = "dummy_pong_raster_frame_stack_one_hot_plus_geometry_v0"
RASTER_ONLY_FRAME_STACK_FEATURE_ENCODING_ID = "dummy_pong_raster_frame_stack_one_hot_v0"
DEFAULT_FEATURE_MODE = "raster_plus_geometry"
DEFAULT_MODEL_TYPE = "linear"
FEATURE_ENCODING_BY_MODE = {
    DEFAULT_FEATURE_MODE: FEATURE_ENCODING_ID,
    "raster_only": RASTER_ONLY_FEATURE_ENCODING_ID,
}
FRAME_STACK_FEATURE_ENCODING_BY_MODE = {
    DEFAULT_FEATURE_MODE: FRAME_STACK_FEATURE_ENCODING_ID,
    "raster_only": RASTER_ONLY_FRAME_STACK_FEATURE_ENCODING_ID,
}
FEATURE_MODES = tuple(FEATURE_ENCODING_BY_MODE)
MODEL_TYPES = ("linear", "mlp")

_RASTER_VALUES = tuple(int(value) for value in RASTER_LEGEND)
_SUPPORTED_REPLAY_ROW_SCHEMA_IDS = (
    IMITATION_REPLAY_ROW_SCHEMA_ID,
    SCORING_REPLAY_ROW_SCHEMA_ID,
    LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
)


class DummyPongImitationPolicy:
    """Linear softmax raster policy with one head per ego agent."""

    def __init__(
        self,
        *,
        weights: np.ndarray,
        bias: np.ndarray,
        raster_shape: tuple[int, int],
        action_labels: tuple[str, ...],
        metadata: dict[str, object],
        feature_mode: str | None = None,
        frame_stack: int | None = None,
    ):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.raster_shape = raster_shape
        self.action_labels = action_labels
        self.metadata = dict(metadata)
        self.feature_mode = _checkpoint_feature_mode(
            self.metadata,
            explicit_feature_mode=feature_mode,
        )
        self.frame_stack = _checkpoint_frame_stack(
            self.metadata,
            explicit_frame_stack=frame_stack,
        )
        if self.weights.shape != (
            len(AGENTS),
            _feature_dim(
                raster_shape,
                feature_mode=self.feature_mode,
                frame_stack=self.frame_stack,
            ),
            len(action_labels),
        ):
            raise ValueError(f"unexpected weight shape {self.weights.shape!r}")
        if self.bias.shape != (len(AGENTS), len(action_labels)):
            raise ValueError(f"unexpected bias shape {self.bias.shape!r}")

    @classmethod
    def load_checkpoint(cls, path: Path) -> "DummyPongImitationPolicy":
        """Load a saved imitation checkpoint."""

        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(_npz_text(payload["metadata"].item()))
            raster_shape = tuple(int(item) for item in payload["raster_shape"].tolist())
            action_labels = tuple(_npz_text(item) for item in payload["action_labels"])
            return cls(
                weights=np.asarray(payload["weights"], dtype=np.float64),
                bias=np.asarray(payload["bias"], dtype=np.float64),
                raster_shape=(raster_shape[0], raster_shape[1]),
                action_labels=action_labels,
                metadata=metadata,
            )

    def predict_proba(self, raster_grid: list[str] | np.ndarray, ego_agent: str) -> np.ndarray:
        """Return action probabilities for one ego agent from a raster grid."""

        agent_index = _agent_index(ego_agent)
        features = _encode_raster_grid(
            raster_grid,
            expected_shape=self.raster_shape,
            feature_mode=self.feature_mode,
            frame_stack=self.frame_stack,
        )
        logits = features @ self.weights[agent_index] + self.bias[agent_index]
        return _softmax(logits)

    def predict_action_id(self, raster_grid: list[str] | np.ndarray, ego_agent: str) -> int:
        """Return the deterministic argmax action id for one ego agent."""

        return int(np.argmax(self.predict_proba(raster_grid, ego_agent)))

    def predict_action_label(self, raster_grid: list[str] | np.ndarray, ego_agent: str) -> str:
        """Return the deterministic argmax action label for one ego agent."""

        return self.action_labels[self.predict_action_id(raster_grid, ego_agent)]


class DummyPongMlpImitationPolicy:
    """One-hidden-layer softmax raster policy with one MLP per ego agent."""

    def __init__(
        self,
        *,
        hidden_weights: np.ndarray,
        hidden_bias: np.ndarray,
        output_weights: np.ndarray,
        output_bias: np.ndarray,
        raster_shape: tuple[int, int],
        action_labels: tuple[str, ...],
        metadata: dict[str, object],
        feature_mode: str | None = None,
        frame_stack: int | None = None,
    ):
        self.hidden_weights = np.asarray(hidden_weights, dtype=np.float64)
        self.hidden_bias = np.asarray(hidden_bias, dtype=np.float64)
        self.output_weights = np.asarray(output_weights, dtype=np.float64)
        self.output_bias = np.asarray(output_bias, dtype=np.float64)
        self.raster_shape = raster_shape
        self.action_labels = action_labels
        self.metadata = dict(metadata)
        self.feature_mode = _checkpoint_feature_mode(
            self.metadata,
            explicit_feature_mode=feature_mode,
        )
        self.frame_stack = _checkpoint_frame_stack(
            self.metadata,
            explicit_frame_stack=frame_stack,
        )
        input_dim = _feature_dim(
            raster_shape,
            feature_mode=self.feature_mode,
            frame_stack=self.frame_stack,
        )
        if self.hidden_weights.ndim != 3:
            raise ValueError(f"unexpected hidden weight shape {self.hidden_weights.shape!r}")
        hidden_dim = self.hidden_weights.shape[2]
        expected_shapes = {
            "hidden_weights": (len(AGENTS), input_dim, hidden_dim),
            "hidden_bias": (len(AGENTS), hidden_dim),
            "output_weights": (len(AGENTS), hidden_dim, len(action_labels)),
            "output_bias": (len(AGENTS), len(action_labels)),
        }
        actual_shapes = {
            "hidden_weights": self.hidden_weights.shape,
            "hidden_bias": self.hidden_bias.shape,
            "output_weights": self.output_weights.shape,
            "output_bias": self.output_bias.shape,
        }
        for name, expected_shape in expected_shapes.items():
            if actual_shapes[name] != expected_shape:
                raise ValueError(
                    f"unexpected {name} shape {actual_shapes[name]!r}, "
                    f"expected {expected_shape!r}"
                )

    @classmethod
    def load_checkpoint(cls, path: Path) -> "DummyPongMlpImitationPolicy":
        """Load a saved MLP imitation checkpoint."""

        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(_npz_text(payload["metadata"].item()))
            raster_shape = tuple(int(item) for item in payload["raster_shape"].tolist())
            action_labels = tuple(_npz_text(item) for item in payload["action_labels"])
            return cls(
                hidden_weights=np.asarray(payload["hidden_weights"], dtype=np.float64),
                hidden_bias=np.asarray(payload["hidden_bias"], dtype=np.float64),
                output_weights=np.asarray(payload["output_weights"], dtype=np.float64),
                output_bias=np.asarray(payload["output_bias"], dtype=np.float64),
                raster_shape=(raster_shape[0], raster_shape[1]),
                action_labels=action_labels,
                metadata=metadata,
            )

    def predict_proba(self, raster_grid: list[str] | np.ndarray, ego_agent: str) -> np.ndarray:
        """Return action probabilities for one ego agent from a raster grid."""

        agent_index = _agent_index(ego_agent)
        features = _encode_raster_grid(
            raster_grid,
            expected_shape=self.raster_shape,
            feature_mode=self.feature_mode,
            frame_stack=self.frame_stack,
        )
        hidden = np.maximum(
            features @ self.hidden_weights[agent_index] + self.hidden_bias[agent_index],
            0.0,
        )
        logits = hidden @ self.output_weights[agent_index] + self.output_bias[agent_index]
        return _softmax(logits)

    def predict_action_id(self, raster_grid: list[str] | np.ndarray, ego_agent: str) -> int:
        """Return the deterministic argmax action id for one ego agent."""

        return int(np.argmax(self.predict_proba(raster_grid, ego_agent)))

    def predict_action_label(self, raster_grid: list[str] | np.ndarray, ego_agent: str) -> str:
        """Return the deterministic argmax action label for one ego agent."""

        return self.action_labels[self.predict_action_id(raster_grid, ego_agent)]


def load_dummy_pong_imitation_checkpoint(path: Path) -> DummyPongImitationPolicy | DummyPongMlpImitationPolicy:
    """Load any supervised dummy Pong imitation checkpoint supported by eval."""

    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(_npz_text(payload["metadata"].item()))
        schema_id = str(metadata.get("checkpoint_schema_id", IMITATION_POLICY_CHECKPOINT_SCHEMA_ID))
        has_mlp_keys = "hidden_weights" in payload and "output_weights" in payload

    if schema_id == IMITATION_MLP_POLICY_CHECKPOINT_SCHEMA_ID or has_mlp_keys:
        return DummyPongMlpImitationPolicy.load_checkpoint(path)
    if schema_id == IMITATION_POLICY_CHECKPOINT_SCHEMA_ID:
        return DummyPongImitationPolicy.load_checkpoint(path)
    raise ValueError(f"unsupported dummy Pong imitation checkpoint schema {schema_id!r}")


def train_dummy_pong_imitation(
    *,
    replay_path: Path,
    output_dir: Path | None = None,
    seed: int = 0,
    epochs: int = 1000,
    learning_rate: float = 1.0,
    validation_fraction: float = 0.2,
    l2: float = 0.0,
    checkpoint_every_epochs: int | None = None,
    class_weighting: str = "none",
    feature_mode: str = DEFAULT_FEATURE_MODE,
    frame_stack: int = 1,
    model_type: str = DEFAULT_MODEL_TYPE,
    hidden_dim: int = 64,
) -> dict[str, object]:
    """Train a tiny supervised policy from dummy Pong replay rows."""

    model_type = _validate_model_type(model_type)
    if model_type == "mlp":
        return train_dummy_pong_mlp_imitation(
            replay_path=replay_path,
            output_dir=output_dir,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            l2=l2,
            checkpoint_every_epochs=checkpoint_every_epochs,
            class_weighting=class_weighting,
            feature_mode=feature_mode,
            frame_stack=frame_stack,
            hidden_dim=hidden_dim,
        )

    feature_mode = _validate_feature_mode(feature_mode)
    frame_stack = _validate_frame_stack(frame_stack)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    if checkpoint_every_epochs is not None and checkpoint_every_epochs < 1:
        raise ValueError("checkpoint_every_epochs must be at least 1 when set")
    if checkpoint_every_epochs is not None and output_dir is None:
        raise ValueError("output_dir is required when checkpoint_every_epochs is set")
    if class_weighting not in ("none", "balanced"):
        raise ValueError("class_weighting must be one of: none, balanced")

    replay_rows_path = _resolve_replay_rows_path(replay_path)
    rows = _load_replay_rows(replay_rows_path)
    dataset = _dataset_from_rows(rows, feature_mode=feature_mode, frame_stack=frame_stack)
    rng = np.random.default_rng(seed)
    train_indices, validation_indices = _split_indices(
        row_count=len(rows),
        validation_fraction=validation_fraction,
        rng=rng,
    )
    class_weights = _class_weights(
        targets=dataset["targets"],
        agent_indices=dataset["agent_indices"],
        indices=train_indices,
        mode=class_weighting,
    )

    weights = np.zeros(
        (
            len(AGENTS),
            dataset["features"].shape[1],
            len(ACTION_LABELS),
        ),
        dtype=np.float64,
    )
    bias = np.zeros((len(AGENTS), len(ACTION_LABELS)), dtype=np.float64)

    history = []
    periodic_checkpoints = []
    for epoch in range(epochs):
        _train_epoch(
            features=dataset["features"],
            targets=dataset["targets"],
            agent_indices=dataset["agent_indices"],
            indices=train_indices,
            weights=weights,
            bias=bias,
            learning_rate=learning_rate,
            l2=l2,
            class_weights=class_weights,
        )
        if epoch == 0 or epoch == epochs - 1:
            history.append(
                {
                    "epoch": epoch + 1,
                    "train": _metrics(
                        features=dataset["features"],
                        targets=dataset["targets"],
                        agent_indices=dataset["agent_indices"],
                        indices=train_indices,
                        weights=weights,
                        bias=bias,
                        l2=l2,
                        class_weights=class_weights,
                    ),
                    "validation": _metrics(
                        features=dataset["features"],
                        targets=dataset["targets"],
                        agent_indices=dataset["agent_indices"],
                        indices=validation_indices,
                        weights=weights,
                        bias=bias,
                        l2=l2,
                        class_weights=class_weights,
                    ),
                }
            )
        completed_epoch = epoch + 1
        if (
            output_dir is not None
            and checkpoint_every_epochs is not None
            and completed_epoch % checkpoint_every_epochs == 0
        ):
            checkpoint_metrics = _metrics_bundle(
                dataset=dataset,
                rows=rows,
                train_indices=train_indices,
                validation_indices=validation_indices,
                weights=weights,
                bias=bias,
                l2=l2,
                class_weights=class_weights,
            )
            checkpoint_path = _periodic_checkpoint_path(output_dir, completed_epoch)
            _write_policy_checkpoint(
                path=checkpoint_path,
                weights=weights,
                bias=bias,
                raster_shape=dataset["raster_shape"],
                metadata=_periodic_checkpoint_metadata(
                    seed=seed,
                    source_replay_path=str(replay_rows_path),
                    epochs=epochs,
                    completed_epochs=completed_epoch,
                    learning_rate=learning_rate,
                    validation_fraction=validation_fraction,
                    l2=l2,
                    class_weighting=class_weighting,
                    class_weights=class_weights,
                    feature_mode=feature_mode,
                    frame_stack=frame_stack,
                    schemas=_schemas_summary(
                        feature_mode=feature_mode,
                        frame_stack=frame_stack,
                    ),
                    input_summary=_input_summary(
                        dataset,
                        feature_mode=feature_mode,
                        frame_stack=frame_stack,
                    ),
                    model_summary=_model_summary(weights, bias),
                    metrics=checkpoint_metrics,
                ),
            )
            periodic_checkpoints.append(
                {
                    "epoch": completed_epoch,
                    "path": str(checkpoint_path),
                }
            )

    train_metrics = _metrics(
        features=dataset["features"],
        targets=dataset["targets"],
        agent_indices=dataset["agent_indices"],
        indices=train_indices,
        weights=weights,
        bias=bias,
        l2=l2,
        class_weights=class_weights,
    )
    validation_metrics = _metrics(
        features=dataset["features"],
        targets=dataset["targets"],
        agent_indices=dataset["agent_indices"],
        indices=validation_indices,
        weights=weights,
        bias=bias,
        l2=l2,
        class_weights=class_weights,
    )
    all_metrics = _metrics(
        features=dataset["features"],
        targets=dataset["targets"],
        agent_indices=dataset["agent_indices"],
        indices=np.arange(len(rows)),
        weights=weights,
        bias=bias,
        l2=l2,
        class_weights=class_weights,
    )

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_imitation_training",
        "note": (
            "Tiny supervised raster policy that copies target_action labels from "
            "dummy Pong replay rows. Labels may come from scripted imitation or "
            "short-lookahead relabeling; this trainer is not MuZero or self-play."
        ),
        "summary_schema_id": IMITATION_TRAIN_SUMMARY_SCHEMA_ID,
        "checkpoint_schema_id": IMITATION_POLICY_CHECKPOINT_SCHEMA_ID,
        "seed": seed,
        "source_replay_path": str(replay_rows_path),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "l2": l2,
        "class_weighting": class_weighting,
        "feature_mode": feature_mode,
        "frame_stack": frame_stack,
        "class_weights_by_agent": _class_weights_summary(class_weights),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "imitation_replay_row_schema_id": IMITATION_REPLAY_ROW_SCHEMA_ID,
            "scoring_replay_row_schema_id": SCORING_REPLAY_ROW_SCHEMA_ID,
            "lookahead_replay_row_schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
            "imitation_train_summary_schema_id": IMITATION_TRAIN_SUMMARY_SCHEMA_ID,
            "imitation_policy_checkpoint_schema_id": IMITATION_POLICY_CHECKPOINT_SCHEMA_ID,
            "feature_encoding_id": _feature_encoding_id(
                feature_mode,
                frame_stack=frame_stack,
            ),
        },
        "input": {
            "raster_shape": list(dataset["raster_shape"]),
            "feature_shape": [int(dataset["features"].shape[1])],
            "feature_mode": feature_mode,
            "frame_stack": frame_stack,
            "feature_encoding": _feature_encoding_id(
                feature_mode,
                frame_stack=frame_stack,
            ),
            "raster_legend": RASTER_LEGEND,
        },
        "model": {
            "type": "per_ego_agent_softmax_linear_numpy",
            "weights_shape": list(weights.shape),
            "bias_shape": list(bias.shape),
            "ego_agents": list(AGENTS),
        },
        "action_labels": list(ACTION_LABELS),
        "data": {
            "rows": len(rows),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "row_schema_ids": sorted(dataset["row_schema_ids"]),
            "replay_row_schema_ids_supported": list(_SUPPORTED_REPLAY_ROW_SCHEMA_IDS),
            "target_policy_ids": sorted(dataset["target_policy_ids"]),
            "behavior_policy_ids": sorted(dataset["behavior_policy_ids"]),
            "reward_values": dataset["reward_values"],
            "nonzero_reward_rows": dataset["nonzero_reward_rows"],
            "positive_reward_rows": dataset["positive_reward_rows"],
            "negative_reward_rows": dataset["negative_reward_rows"],
            "terminated_rows": dataset["terminated_rows"],
            "truncated_rows": dataset["truncated_rows"],
            "action_histogram_by_agent": dataset["action_histogram_by_agent"],
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "all_rows": all_metrics,
        },
        "history": history,
        "plain_language": {
            "proves": (
                "The replay can feed a supervised raster learner, save a checkpoint, "
                "reload it later, and predict target_action-style actions from raster grids. "
                "With feature_mode=raster_only, logits use one-hot raster cells plus the "
                "per-ego policy head only, with no decoded geometry suffix. With "
                "frame_stack > 1, logits receive chronological raster frames so the "
                "linear policy can see one-step visual velocity."
            ),
            "does_not_prove": (
                "This does not prove Pong reinforcement learning, MuZero planning, "
                "self-play improvement, or reward-driven behavior. Score-bearing rows "
                "are still used here only as supervised action labels."
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
        if checkpoint_every_epochs is not None:
            artifacts["periodic_checkpoint_dir"] = output_dir / "checkpoints"
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(
            artifacts=artifacts,
            summary=summary,
            weights=weights,
            bias=bias,
            raster_shape=dataset["raster_shape"],
        )
    return summary


def train_dummy_pong_mlp_imitation(
    *,
    replay_path: Path,
    output_dir: Path | None = None,
    seed: int = 0,
    epochs: int = 1000,
    learning_rate: float = 0.01,
    validation_fraction: float = 0.2,
    l2: float = 0.0,
    checkpoint_every_epochs: int | None = None,
    class_weighting: str = "none",
    feature_mode: str = DEFAULT_FEATURE_MODE,
    frame_stack: int = 1,
    hidden_dim: int = 64,
) -> dict[str, object]:
    """Train a tiny one-hidden-layer NumPy MLP from dummy Pong replay rows."""

    feature_mode = _validate_feature_mode(feature_mode)
    frame_stack = _validate_frame_stack(frame_stack)
    hidden_dim = _validate_hidden_dim(hidden_dim)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    if checkpoint_every_epochs is not None and checkpoint_every_epochs < 1:
        raise ValueError("checkpoint_every_epochs must be at least 1 when set")
    if checkpoint_every_epochs is not None and output_dir is None:
        raise ValueError("output_dir is required when checkpoint_every_epochs is set")
    if class_weighting not in ("none", "balanced"):
        raise ValueError("class_weighting must be one of: none, balanced")

    replay_rows_path = _resolve_replay_rows_path(replay_path)
    rows = _load_replay_rows(replay_rows_path)
    dataset = _dataset_from_rows(rows, feature_mode=feature_mode, frame_stack=frame_stack)
    rng = np.random.default_rng(seed)
    train_indices, validation_indices = _split_indices(
        row_count=len(rows),
        validation_fraction=validation_fraction,
        rng=rng,
    )
    class_weights = _class_weights(
        targets=dataset["targets"],
        agent_indices=dataset["agent_indices"],
        indices=train_indices,
        mode=class_weighting,
    )
    params = _init_mlp_params(
        rng=rng,
        input_dim=int(dataset["features"].shape[1]),
        hidden_dim=hidden_dim,
    )
    adam_state = _init_mlp_adam_state(params)

    history = []
    periodic_checkpoints = []
    for epoch in range(epochs):
        _train_mlp_epoch(
            features=dataset["features"],
            targets=dataset["targets"],
            agent_indices=dataset["agent_indices"],
            indices=train_indices,
            params=params,
            adam_state=adam_state,
            learning_rate=learning_rate,
            l2=l2,
            class_weights=class_weights,
            step=epoch + 1,
        )
        if epoch == 0 or epoch == epochs - 1:
            history.append(
                {
                    "epoch": epoch + 1,
                    "train": _mlp_metrics(
                        features=dataset["features"],
                        targets=dataset["targets"],
                        agent_indices=dataset["agent_indices"],
                        indices=train_indices,
                        params=params,
                        l2=l2,
                        class_weights=class_weights,
                    ),
                    "validation": _mlp_metrics(
                        features=dataset["features"],
                        targets=dataset["targets"],
                        agent_indices=dataset["agent_indices"],
                        indices=validation_indices,
                        params=params,
                        l2=l2,
                        class_weights=class_weights,
                    ),
                }
            )
        completed_epoch = epoch + 1
        if (
            output_dir is not None
            and checkpoint_every_epochs is not None
            and completed_epoch % checkpoint_every_epochs == 0
        ):
            checkpoint_metrics = _mlp_metrics_bundle(
                dataset=dataset,
                rows=rows,
                train_indices=train_indices,
                validation_indices=validation_indices,
                params=params,
                l2=l2,
                class_weights=class_weights,
            )
            checkpoint_path = _periodic_checkpoint_path(output_dir, completed_epoch)
            _write_mlp_policy_checkpoint(
                path=checkpoint_path,
                params=params,
                raster_shape=dataset["raster_shape"],
                metadata=_periodic_mlp_checkpoint_metadata(
                    seed=seed,
                    source_replay_path=str(replay_rows_path),
                    epochs=epochs,
                    completed_epochs=completed_epoch,
                    learning_rate=learning_rate,
                    validation_fraction=validation_fraction,
                    l2=l2,
                    class_weighting=class_weighting,
                    feature_mode=feature_mode,
                    frame_stack=frame_stack,
                    hidden_dim=hidden_dim,
                    class_weights=class_weights,
                    schemas=_schemas_summary(
                        feature_mode=feature_mode,
                        frame_stack=frame_stack,
                    ),
                    input_summary=_input_summary(
                        dataset,
                        feature_mode=feature_mode,
                        frame_stack=frame_stack,
                    ),
                    model_summary=_mlp_model_summary(params),
                    metrics=checkpoint_metrics,
                ),
            )
            periodic_checkpoints.append(
                {
                    "epoch": completed_epoch,
                    "path": str(checkpoint_path),
                }
            )

    train_metrics = _mlp_metrics(
        features=dataset["features"],
        targets=dataset["targets"],
        agent_indices=dataset["agent_indices"],
        indices=train_indices,
        params=params,
        l2=l2,
        class_weights=class_weights,
    )
    validation_metrics = _mlp_metrics(
        features=dataset["features"],
        targets=dataset["targets"],
        agent_indices=dataset["agent_indices"],
        indices=validation_indices,
        params=params,
        l2=l2,
        class_weights=class_weights,
    )
    all_metrics = _mlp_metrics(
        features=dataset["features"],
        targets=dataset["targets"],
        agent_indices=dataset["agent_indices"],
        indices=np.arange(len(rows)),
        params=params,
        l2=l2,
        class_weights=class_weights,
    )

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_imitation_training",
        "note": (
            "Tiny supervised raster MLP policy that copies target_action labels "
            "from dummy Pong replay rows. This is a CPU-local nonlinear behavior "
            "cloning smoke, not MuZero or self-play."
        ),
        "summary_schema_id": IMITATION_TRAIN_SUMMARY_SCHEMA_ID,
        "checkpoint_schema_id": IMITATION_MLP_POLICY_CHECKPOINT_SCHEMA_ID,
        "seed": seed,
        "source_replay_path": str(replay_rows_path),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "l2": l2,
        "class_weighting": class_weighting,
        "feature_mode": feature_mode,
        "frame_stack": frame_stack,
        "class_weights_by_agent": _class_weights_summary(class_weights),
        "schemas": {
            **_schemas_summary(feature_mode=feature_mode, frame_stack=frame_stack),
            "imitation_mlp_policy_checkpoint_schema_id": IMITATION_MLP_POLICY_CHECKPOINT_SCHEMA_ID,
        },
        "input": _input_summary(
            dataset,
            feature_mode=feature_mode,
            frame_stack=frame_stack,
        ),
        "model": _mlp_model_summary(params),
        "action_labels": list(ACTION_LABELS),
        "data": {
            "rows": len(rows),
            "train_rows": int(len(train_indices)),
            "validation_rows": int(len(validation_indices)),
            "row_schema_ids": sorted(dataset["row_schema_ids"]),
            "replay_row_schema_ids_supported": list(_SUPPORTED_REPLAY_ROW_SCHEMA_IDS),
            "target_policy_ids": sorted(dataset["target_policy_ids"]),
            "behavior_policy_ids": sorted(dataset["behavior_policy_ids"]),
            "reward_values": dataset["reward_values"],
            "nonzero_reward_rows": dataset["nonzero_reward_rows"],
            "positive_reward_rows": dataset["positive_reward_rows"],
            "negative_reward_rows": dataset["negative_reward_rows"],
            "terminated_rows": dataset["terminated_rows"],
            "truncated_rows": dataset["truncated_rows"],
            "action_histogram_by_agent": dataset["action_histogram_by_agent"],
        },
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "all_rows": all_metrics,
        },
        "history": history,
        "plain_language": {
            "proves": (
                "The replay can feed a nonlinear raster learner, save a separate "
                "MLP checkpoint, reload it in the existing Pong eval path, and "
                "predict from raster observations plus optional frame history."
            ),
            "does_not_prove": (
                "This does not prove reinforcement learning or planning. With "
                "feature_mode=raster_only it avoids decoded geometry helpers, but "
                "the labels are still supervised exact-trace actions."
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
        if checkpoint_every_epochs is not None:
            artifacts["periodic_checkpoint_dir"] = output_dir / "checkpoints"
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_mlp_artifacts(
            artifacts=artifacts,
            summary=summary,
            params=params,
            raster_shape=dataset["raster_shape"],
        )
    return summary


def _resolve_replay_rows_path(replay_path: Path) -> Path:
    if replay_path.is_dir():
        replay_path = replay_path / "replay_rows.jsonl"
    if not replay_path.exists():
        raise FileNotFoundError(f"replay rows not found: {replay_path}")
    return replay_path


def _load_replay_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if row.get("schema_id") not in _SUPPORTED_REPLAY_ROW_SCHEMA_IDS:
                raise ValueError(
                    f"{path}:{line_number} has unexpected schema_id {row.get('schema_id')!r}"
                )
            _validate_supervised_replay_row(row, path=path, line_number=line_number)
            rows.append(row)
    if not rows:
        raise ValueError(f"replay rows file is empty: {path}")
    return rows


def _dataset_from_rows(
    rows: list[dict[str, Any]],
    *,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    frame_stack: int = 1,
) -> dict[str, Any]:
    feature_mode = _validate_feature_mode(feature_mode)
    frame_stack = _validate_frame_stack(frame_stack)
    first_shape = _row_raster_shape(rows[0])
    features = []
    targets = []
    agent_indices = []
    row_schema_ids = set()
    target_policy_ids = set()
    behavior_policy_ids = set()
    reward_values = set()
    nonzero_reward_rows = 0
    positive_reward_rows = 0
    negative_reward_rows = 0
    terminated_rows = 0
    truncated_rows = 0
    action_histogram_by_agent = {agent: Counter() for agent in AGENTS}

    for row in rows:
        raster_shape = _row_raster_shape(row)
        if raster_shape != first_shape:
            raise ValueError(f"mixed raster shapes: {first_shape!r} and {raster_shape!r}")
        ego_agent = str(row["ego_agent"])
        target_action_id = int(row["target_action_id"])
        reward = float(row["reward_after_step"])

        features.append(
            _encode_raster_grid(
                _row_raster_input(row, frame_stack=frame_stack),
                expected_shape=first_shape,
                feature_mode=feature_mode,
                frame_stack=frame_stack,
            )
        )
        targets.append(target_action_id)
        agent_indices.append(_agent_index(ego_agent))
        row_schema_ids.add(str(row["schema_id"]))
        target_policy_ids.add(str(row["target_policy_id"]))
        behavior_policy_ids.add(
            str(row.get("behavior_policy_id", row.get("collector_policy_id", row["target_policy_id"])))
        )
        reward_values.add(reward)
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
        action_histogram_by_agent[ego_agent][ACTION_LABELS[target_action_id]] += 1

    return {
        "features": np.vstack(features),
        "targets": np.asarray(targets, dtype=np.int64),
        "agent_indices": np.asarray(agent_indices, dtype=np.int64),
        "raster_shape": first_shape,
        "row_schema_ids": row_schema_ids,
        "target_policy_ids": target_policy_ids,
        "behavior_policy_ids": behavior_policy_ids,
        "reward_values": sorted(float(value) for value in reward_values),
        "nonzero_reward_rows": nonzero_reward_rows,
        "positive_reward_rows": positive_reward_rows,
        "negative_reward_rows": negative_reward_rows,
        "terminated_rows": terminated_rows,
        "truncated_rows": truncated_rows,
        "action_histogram_by_agent": {
            agent: {label: int(action_histogram_by_agent[agent][label]) for label in ACTION_LABELS}
            for agent in AGENTS
        },
    }


def _validate_supervised_replay_row(
    row: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> None:
    required_fields = (
        "schema_id",
        "raster_observation_schema_id",
        "raster_shape",
        "raster_grid",
        "ego_agent",
        "target_policy_id",
        "target_action_id",
        "target_action_label",
        "joint_action_by_agent",
        "reward_after_step",
        "next_raster_grid",
    )
    missing_fields = [field for field in required_fields if field not in row]
    if missing_fields:
        raise ValueError(f"{path}:{line_number} missing required fields {missing_fields!r}")

    if row["raster_observation_schema_id"] != RASTER_OBSERVATION_SCHEMA_ID:
        raise ValueError(
            f"{path}:{line_number} has unexpected raster_observation_schema_id "
            f"{row['raster_observation_schema_id']!r}"
        )

    ego_agent = str(row["ego_agent"])
    _agent_index(ego_agent)

    target_action_id = _validate_action_id(
        row["target_action_id"],
        field="target_action_id",
        path=path,
        line_number=line_number,
    )
    _validate_action_label(
        row["target_action_label"],
        action_id=target_action_id,
        field="target_action_label",
        path=path,
        line_number=line_number,
    )

    behavior_action_id = row.get("behavior_action_id")
    if behavior_action_id is not None:
        behavior_action_id = _validate_action_id(
            behavior_action_id,
            field="behavior_action_id",
            path=path,
            line_number=line_number,
        )
        if behavior_action_id != target_action_id:
            raise ValueError(
                f"{path}:{line_number} behavior_action_id {behavior_action_id!r} "
                f"does not match target_action_id {target_action_id!r}"
            )
        if "behavior_action_label" not in row:
            raise ValueError(f"{path}:{line_number} missing behavior_action_label")
        _validate_action_label(
            row["behavior_action_label"],
            action_id=target_action_id,
            field="behavior_action_label",
            path=path,
            line_number=line_number,
        )

    joint_action_by_agent = row["joint_action_by_agent"]
    if not isinstance(joint_action_by_agent, dict):
        raise ValueError(f"{path}:{line_number} joint_action_by_agent must be an object")
    if set(joint_action_by_agent) != set(AGENTS):
        raise ValueError(
            f"{path}:{line_number} joint_action_by_agent agents "
            f"{sorted(joint_action_by_agent)!r} do not match {list(AGENTS)!r}"
        )
    for agent, action_payload in joint_action_by_agent.items():
        if not isinstance(action_payload, dict):
            raise ValueError(f"{path}:{line_number} joint action for {agent!r} must be an object")
        action_id = _validate_action_id(
            action_payload.get("action_id"),
            field=f"joint_action_by_agent[{agent!r}].action_id",
            path=path,
            line_number=line_number,
        )
        _validate_action_label(
            action_payload.get("label"),
            action_id=action_id,
            field=f"joint_action_by_agent[{agent!r}].label",
            path=path,
            line_number=line_number,
        )

    if int(joint_action_by_agent[ego_agent]["action_id"]) != target_action_id:
        raise ValueError(
            f"{path}:{line_number} ego joint action does not match target_action_id "
            f"{target_action_id!r}"
        )

    raster_shape = _row_raster_shape(row)
    raster_grid = _raster_array(row["raster_grid"])
    next_raster_grid = _raster_array(row["next_raster_grid"])
    if raster_shape != raster_grid.shape:
        raise ValueError(f"{path}:{line_number} raster_shape does not match raster_grid")
    if next_raster_grid.shape != raster_shape:
        raise ValueError(f"{path}:{line_number} next_raster_grid shape does not match raster_shape")
    if "raster_frame_stack" in row:
        frame_stack = _validate_frame_stack(row.get("frame_stack", len(row["raster_frame_stack"])))
        raster_frame_stack = _raster_stack_array(
            row["raster_frame_stack"],
            expected_shape=raster_shape,
            frame_stack=frame_stack,
        )
        if not np.array_equal(raster_frame_stack[-1], raster_grid):
            raise ValueError(
                f"{path}:{line_number} final raster_frame_stack frame does not match raster_grid"
            )
        if "next_raster_frame_stack" not in row:
            raise ValueError(f"{path}:{line_number} missing next_raster_frame_stack")
        next_raster_frame_stack = _raster_stack_array(
            row["next_raster_frame_stack"],
            expected_shape=raster_shape,
            frame_stack=frame_stack,
        )
        if not np.array_equal(next_raster_frame_stack[-1], next_raster_grid):
            raise ValueError(
                f"{path}:{line_number} final next_raster_frame_stack frame does not match "
                "next_raster_grid"
            )
    float(row["reward_after_step"])


def _validate_action_id(
    value: object,
    *,
    field: str,
    path: Path,
    line_number: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path}:{line_number} {field} must be an integer, got {value!r}")
    try:
        action_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{line_number} {field} must be an integer") from exc
    if action_id != value and not isinstance(value, np.integer):
        raise ValueError(f"{path}:{line_number} {field} must be an integer, got {value!r}")
    if not 0 <= action_id < len(ACTION_LABELS):
        raise ValueError(f"{path}:{line_number} {field} out of range: {action_id!r}")
    return action_id


def _validate_action_label(
    value: object,
    *,
    action_id: int,
    field: str,
    path: Path,
    line_number: int,
) -> None:
    expected = ACTION_LABELS[action_id]
    if value != expected:
        raise ValueError(
            f"{path}:{line_number} {field} {value!r} does not match "
            f"action {action_id!r} label {expected!r}"
        )


def _split_indices(
    *,
    row_count: int,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(row_count)
    rng.shuffle(indices)
    validation_count = int(round(row_count * validation_fraction))
    if row_count > 1 and validation_fraction > 0.0:
        validation_count = max(1, min(validation_count, row_count - 1))
    validation_indices = np.sort(indices[:validation_count])
    train_indices = np.sort(indices[validation_count:])
    if len(train_indices) < 1:
        raise ValueError("training split is empty")
    return train_indices, validation_indices


def _train_epoch(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    agent_indices: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    learning_rate: float,
    l2: float,
    class_weights: np.ndarray,
) -> None:
    row_count = float(len(indices))
    weight_grad = np.zeros_like(weights)
    bias_grad = np.zeros_like(bias)
    for agent_index in range(len(AGENTS)):
        agent_mask = agent_indices[indices] == agent_index
        agent_rows = indices[agent_mask]
        if len(agent_rows) == 0:
            continue
        agent_features = features[agent_rows]
        probs = _softmax_rows(agent_features @ weights[agent_index] + bias[agent_index])
        probs[np.arange(len(agent_rows)), targets[agent_rows]] -= 1.0
        row_weights = class_weights[agent_index, targets[agent_rows]]
        probs *= row_weights[:, np.newaxis]
        weight_grad[agent_index] = agent_features.T @ probs
        bias_grad[agent_index] = np.sum(probs, axis=0)
    if l2:
        weight_grad += l2 * weights
    weights -= learning_rate * weight_grad / row_count
    bias -= learning_rate * bias_grad / row_count


def _class_weights(
    *,
    targets: np.ndarray,
    agent_indices: np.ndarray,
    indices: np.ndarray,
    mode: str,
) -> np.ndarray:
    weights = np.ones((len(AGENTS), len(ACTION_LABELS)), dtype=np.float64)
    if mode == "none":
        return weights
    if mode != "balanced":
        raise ValueError(f"unsupported class weighting mode {mode!r}")

    weights.fill(0.0)
    for agent_index in range(len(AGENTS)):
        agent_rows = indices[agent_indices[indices] == agent_index]
        if len(agent_rows) == 0:
            continue
        counts = np.bincount(targets[agent_rows], minlength=len(ACTION_LABELS))
        present = counts > 0
        if not np.any(present):
            continue
        weights[agent_index, present] = (
            len(agent_rows) / (float(np.sum(present)) * counts[present])
        )
    return weights


def _metrics(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    agent_indices: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    l2: float,
    class_weights: np.ndarray,
) -> dict[str, object]:
    if len(indices) == 0:
        return {"rows": 0, "loss": None, "accuracy": None}

    losses = []
    weighted_losses = []
    correct = 0
    action_counts = Counter()
    correct_by_class = Counter()
    target_counts = Counter()
    for agent_index in range(len(AGENTS)):
        agent_mask = agent_indices[indices] == agent_index
        agent_rows = indices[agent_mask]
        if len(agent_rows) == 0:
            continue
        probs = _softmax_rows(features[agent_rows] @ weights[agent_index] + bias[agent_index])
        predictions = np.argmax(probs, axis=1)
        for predicted in predictions:
            action_counts[ACTION_LABELS[int(predicted)]] += 1
        correct += int(np.sum(predictions == targets[agent_rows]))
        row_losses = -np.log(np.maximum(probs[np.arange(len(agent_rows)), targets[agent_rows]], 1e-12))
        row_weights = class_weights[agent_index, targets[agent_rows]]
        weighted_losses.extend(float(loss * weight) for loss, weight in zip(row_losses, row_weights))
        losses.extend(float(item) for item in row_losses)
        for target, predicted in zip(targets[agent_rows], predictions):
            label = ACTION_LABELS[int(target)]
            target_counts[label] += 1
            if int(target) == int(predicted):
                correct_by_class[label] += 1

    loss = float(np.mean(losses))
    weighted_loss = float(np.mean(weighted_losses))
    if l2:
        loss += float(0.5 * l2 * np.mean(weights * weights))
        weighted_loss += float(0.5 * l2 * np.mean(weights * weights))
    return {
        "rows": int(len(indices)),
        "loss": loss,
        "weighted_loss": weighted_loss,
        "accuracy": float(correct / len(indices)),
        "target_action_histogram": {
            label: int(target_counts[label])
            for label in ACTION_LABELS
        },
        "predicted_action_histogram": {
            label: int(action_counts[label])
            for label in ACTION_LABELS
        },
        "accuracy_by_class": {
            label: (
                None
                if target_counts[label] == 0
                else float(correct_by_class[label] / target_counts[label])
            )
            for label in ACTION_LABELS
        },
    }


def _encode_raster_grid(
    raster_grid: list[str] | np.ndarray,
    *,
    expected_shape: tuple[int, int],
    feature_mode: str = DEFAULT_FEATURE_MODE,
    frame_stack: int = 1,
) -> np.ndarray:
    feature_mode = _validate_feature_mode(feature_mode)
    frame_stack = _validate_frame_stack(frame_stack)
    stack = _raster_stack_array(
        raster_grid,
        expected_shape=expected_shape,
        frame_stack=frame_stack,
    )
    if np.any((stack < 0) | (stack >= len(_RASTER_VALUES))):
        raise ValueError("raster grid contains values outside the known legend")
    eye = np.eye(len(_RASTER_VALUES), dtype=np.float64)
    raster_features = eye[stack].reshape(-1)
    if feature_mode == "raster_only":
        return raster_features
    return np.concatenate([raster_features, _geometry_features(stack[-1])])


def _row_raster_input(row: dict[str, Any], *, frame_stack: int) -> object:
    if frame_stack == 1:
        return row["raster_grid"]
    if "raster_frame_stack" not in row:
        raise ValueError(
            "frame_stack > 1 requires replay rows with raster_frame_stack; "
            "rebuild the lag-1 trace replay with --frame-stack"
        )
    row_frame_stack = row.get("frame_stack")
    if row_frame_stack is not None and int(row_frame_stack) != frame_stack:
        raise ValueError(
            f"row frame_stack {int(row_frame_stack)!r} does not match requested {frame_stack!r}"
        )
    return row["raster_frame_stack"]


def _geometry_features(grid: np.ndarray) -> np.ndarray:
    """Decode small position features from the raster, without reading tabular rows."""

    height, width = grid.shape
    ball_positions = np.argwhere((grid == 3) | (grid == 4))
    if len(ball_positions) != 1:
        raise ValueError(f"expected exactly one ball cell, got {len(ball_positions)}")
    ball_y, ball_x = (int(item) for item in ball_positions[0])

    player_0_center = _paddle_center_y(grid, paddle_value=1, fallback_x=1)
    player_1_center = _paddle_center_y(grid, paddle_value=2, fallback_x=width - 2)
    scale_y = max(height - 1, 1)
    scale_x = max(width - 1, 1)
    return np.asarray(
        [
            ball_x / scale_x,
            ball_y / scale_y,
            player_0_center / scale_y,
            player_1_center / scale_y,
            (ball_y - player_0_center) / scale_y,
            (ball_y - player_1_center) / scale_y,
        ],
        dtype=np.float64,
    )


def _paddle_center_y(grid: np.ndarray, *, paddle_value: int, fallback_x: int) -> float:
    rows = [int(row) for row, col in np.argwhere(grid == paddle_value)]
    ball_on_paddle_rows = [
        int(row)
        for row, col in np.argwhere(grid == 4)
        if int(col) == fallback_x
    ]
    rows.extend(ball_on_paddle_rows)
    if not rows:
        raise ValueError(f"could not locate paddle value {paddle_value}")
    return float(np.mean(rows))


def _raster_array(raster_grid: list[str] | np.ndarray) -> np.ndarray:
    if isinstance(raster_grid, np.ndarray):
        grid = np.asarray(raster_grid, dtype=np.int64)
        if grid.ndim != 2:
            raise ValueError(f"raster grid must be 2D, got shape {grid.shape!r}")
        return grid
    if not raster_grid:
        raise ValueError("raster grid must not be empty")
    width = len(raster_grid[0])
    rows = []
    for row in raster_grid:
        if len(row) != width:
            raise ValueError("raster grid rows must have equal widths")
        rows.append([int(cell) for cell in row])
    return np.asarray(rows, dtype=np.int64)


def _raster_stack_array(
    raster_grid: object,
    *,
    expected_shape: tuple[int, int],
    frame_stack: int,
) -> np.ndarray:
    if isinstance(raster_grid, np.ndarray):
        array = np.asarray(raster_grid, dtype=np.int64)
        if array.ndim == 2:
            array = array[np.newaxis, :, :]
        if array.ndim != 3:
            raise ValueError(f"raster stack must be 3D, got shape {array.shape!r}")
    else:
        if (
            isinstance(raster_grid, list)
            and raster_grid
            and all(isinstance(row, str) for row in raster_grid)
        ):
            array = _raster_array(raster_grid)[np.newaxis, :, :]
        elif isinstance(raster_grid, list):
            array = np.stack([_raster_array(frame) for frame in raster_grid])
        else:
            array = _raster_array(raster_grid)[np.newaxis, :, :]

    if array.shape != (frame_stack, expected_shape[0], expected_shape[1]):
        raise ValueError(
            "expected raster stack shape "
            f"{(frame_stack, expected_shape[0], expected_shape[1])!r}, got {array.shape!r}"
        )
    return array


def _row_raster_shape(row: dict[str, Any]) -> tuple[int, int]:
    shape = tuple(int(item) for item in row["raster_shape"])
    if len(shape) != 2:
        raise ValueError(f"raster_shape must have two dimensions: {shape!r}")
    return (shape[0], shape[1])


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "checkpoint_npz": output_dir / "checkpoint.npz",
    }


def _periodic_checkpoint_path(output_dir: Path, completed_epoch: int) -> Path:
    return output_dir / "checkpoints" / f"epoch-{completed_epoch:06d}" / "checkpoint.npz"


def _feature_dim(
    raster_shape: tuple[int, int],
    *,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    frame_stack: int = 1,
) -> int:
    feature_mode = _validate_feature_mode(feature_mode)
    frame_stack = _validate_frame_stack(frame_stack)
    raster_dim = raster_shape[0] * raster_shape[1] * len(_RASTER_VALUES) * frame_stack
    if feature_mode == "raster_only":
        return raster_dim
    return raster_dim + 6


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    weights: np.ndarray,
    bias: np.ndarray,
    raster_shape: tuple[int, int],
) -> None:
    summary_path = artifacts["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _write_mlp_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    params: dict[str, np.ndarray],
    raster_shape: tuple[int, int],
) -> None:
    summary_path = artifacts["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_mlp_policy_checkpoint(
        path=artifacts["checkpoint_npz"],
        params=params,
        raster_shape=raster_shape,
        metadata=_mlp_checkpoint_metadata(summary),
    )


def _write_mlp_policy_checkpoint(
    *,
    path: Path,
    params: dict[str, np.ndarray],
    raster_shape: tuple[int, int],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hidden_weights=params["hidden_weights"],
        hidden_bias=params["hidden_bias"],
        output_weights=params["output_weights"],
        output_bias=params["output_bias"],
        raster_shape=np.asarray(raster_shape, dtype=np.int64),
        action_labels=np.asarray(ACTION_LABELS),
        ego_agents=np.asarray(AGENTS),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _init_mlp_params(
    *,
    rng: np.random.Generator,
    input_dim: int,
    hidden_dim: int,
) -> dict[str, np.ndarray]:
    hidden_scale = np.sqrt(2.0 / max(input_dim, 1))
    output_scale = np.sqrt(2.0 / max(hidden_dim, 1))
    return {
        "hidden_weights": rng.normal(
            loc=0.0,
            scale=hidden_scale,
            size=(len(AGENTS), input_dim, hidden_dim),
        ),
        "hidden_bias": np.zeros((len(AGENTS), hidden_dim), dtype=np.float64),
        "output_weights": rng.normal(
            loc=0.0,
            scale=output_scale,
            size=(len(AGENTS), hidden_dim, len(ACTION_LABELS)),
        ),
        "output_bias": np.zeros((len(AGENTS), len(ACTION_LABELS)), dtype=np.float64),
    }


def _init_mlp_adam_state(params: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {
            "m": np.zeros_like(value),
            "v": np.zeros_like(value),
        }
        for name, value in params.items()
    }


def _train_mlp_epoch(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    agent_indices: np.ndarray,
    indices: np.ndarray,
    params: dict[str, np.ndarray],
    adam_state: dict[str, dict[str, np.ndarray]],
    learning_rate: float,
    l2: float,
    class_weights: np.ndarray,
    step: int,
) -> None:
    row_count = float(len(indices))
    grads = {name: np.zeros_like(value) for name, value in params.items()}
    for agent_index in range(len(AGENTS)):
        agent_mask = agent_indices[indices] == agent_index
        agent_rows = indices[agent_mask]
        if len(agent_rows) == 0:
            continue
        agent_features = features[agent_rows]
        hidden_pre = (
            agent_features @ params["hidden_weights"][agent_index]
            + params["hidden_bias"][agent_index]
        )
        hidden = np.maximum(hidden_pre, 0.0)
        logits = hidden @ params["output_weights"][agent_index] + params["output_bias"][agent_index]
        probs = _softmax_rows(logits)
        probs[np.arange(len(agent_rows)), targets[agent_rows]] -= 1.0
        row_weights = class_weights[agent_index, targets[agent_rows]]
        probs *= row_weights[:, np.newaxis]

        grads["output_weights"][agent_index] = hidden.T @ probs
        grads["output_bias"][agent_index] = np.sum(probs, axis=0)
        hidden_grad = probs @ params["output_weights"][agent_index].T
        hidden_grad *= hidden_pre > 0.0
        grads["hidden_weights"][agent_index] = agent_features.T @ hidden_grad
        grads["hidden_bias"][agent_index] = np.sum(hidden_grad, axis=0)

    if l2:
        grads["hidden_weights"] += l2 * params["hidden_weights"]
        grads["output_weights"] += l2 * params["output_weights"]
    for name, value in params.items():
        _adam_update(
            value=value,
            grad=grads[name] / row_count,
            state=adam_state[name],
            learning_rate=learning_rate,
            step=step,
        )


def _adam_update(
    *,
    value: np.ndarray,
    grad: np.ndarray,
    state: dict[str, np.ndarray],
    learning_rate: float,
    step: int,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    state["m"] = beta1 * state["m"] + (1.0 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1.0 - beta2) * (grad * grad)
    m_hat = state["m"] / (1.0 - beta1**step)
    v_hat = state["v"] / (1.0 - beta2**step)
    value -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


def _mlp_logits(
    *,
    features: np.ndarray,
    params: dict[str, np.ndarray],
    agent_index: int,
) -> np.ndarray:
    hidden = np.maximum(
        features @ params["hidden_weights"][agent_index] + params["hidden_bias"][agent_index],
        0.0,
    )
    return hidden @ params["output_weights"][agent_index] + params["output_bias"][agent_index]


def _mlp_metrics(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    agent_indices: np.ndarray,
    indices: np.ndarray,
    params: dict[str, np.ndarray],
    l2: float,
    class_weights: np.ndarray,
) -> dict[str, object]:
    if len(indices) == 0:
        return {"rows": 0, "loss": None, "accuracy": None}

    losses = []
    weighted_losses = []
    correct = 0
    action_counts = Counter()
    correct_by_class = Counter()
    target_counts = Counter()
    for agent_index in range(len(AGENTS)):
        agent_mask = agent_indices[indices] == agent_index
        agent_rows = indices[agent_mask]
        if len(agent_rows) == 0:
            continue
        probs = _softmax_rows(
            _mlp_logits(
                features=features[agent_rows],
                params=params,
                agent_index=agent_index,
            )
        )
        predictions = np.argmax(probs, axis=1)
        for predicted in predictions:
            action_counts[ACTION_LABELS[int(predicted)]] += 1
        correct += int(np.sum(predictions == targets[agent_rows]))
        row_losses = -np.log(np.maximum(probs[np.arange(len(agent_rows)), targets[agent_rows]], 1e-12))
        row_weights = class_weights[agent_index, targets[agent_rows]]
        weighted_losses.extend(float(loss * weight) for loss, weight in zip(row_losses, row_weights))
        losses.extend(float(item) for item in row_losses)
        for target, predicted in zip(targets[agent_rows], predictions):
            label = ACTION_LABELS[int(target)]
            target_counts[label] += 1
            if int(target) == int(predicted):
                correct_by_class[label] += 1

    loss = float(np.mean(losses))
    weighted_loss = float(np.mean(weighted_losses))
    if l2:
        weight_penalty = 0.5 * l2 * (
            float(np.mean(params["hidden_weights"] * params["hidden_weights"]))
            + float(np.mean(params["output_weights"] * params["output_weights"]))
        )
        loss += weight_penalty
        weighted_loss += weight_penalty
    return {
        "rows": int(len(indices)),
        "loss": loss,
        "weighted_loss": weighted_loss,
        "accuracy": float(correct / len(indices)),
        "target_action_histogram": {
            label: int(target_counts[label])
            for label in ACTION_LABELS
        },
        "predicted_action_histogram": {
            label: int(action_counts[label])
            for label in ACTION_LABELS
        },
        "accuracy_by_class": {
            label: (
                None
                if target_counts[label] == 0
                else float(correct_by_class[label] / target_counts[label])
            )
            for label in ACTION_LABELS
        },
    }


def _mlp_metrics_bundle(
    *,
    dataset: dict[str, Any],
    rows: list[dict[str, Any]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    params: dict[str, np.ndarray],
    l2: float,
    class_weights: np.ndarray,
) -> dict[str, object]:
    return {
        "train": _mlp_metrics(
            features=dataset["features"],
            targets=dataset["targets"],
            agent_indices=dataset["agent_indices"],
            indices=train_indices,
            params=params,
            l2=l2,
            class_weights=class_weights,
        ),
        "validation": _mlp_metrics(
            features=dataset["features"],
            targets=dataset["targets"],
            agent_indices=dataset["agent_indices"],
            indices=validation_indices,
            params=params,
            l2=l2,
            class_weights=class_weights,
        ),
        "all_rows": _mlp_metrics(
            features=dataset["features"],
            targets=dataset["targets"],
            agent_indices=dataset["agent_indices"],
            indices=np.arange(len(rows)),
            params=params,
            l2=l2,
            class_weights=class_weights,
        ),
    }


def _metrics_bundle(
    *,
    dataset: dict[str, Any],
    rows: list[dict[str, Any]],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    l2: float,
    class_weights: np.ndarray,
) -> dict[str, object]:
    return {
        "train": _metrics(
            features=dataset["features"],
            targets=dataset["targets"],
            agent_indices=dataset["agent_indices"],
            indices=train_indices,
            weights=weights,
            bias=bias,
            l2=l2,
            class_weights=class_weights,
        ),
        "validation": _metrics(
            features=dataset["features"],
            targets=dataset["targets"],
            agent_indices=dataset["agent_indices"],
            indices=validation_indices,
            weights=weights,
            bias=bias,
            l2=l2,
            class_weights=class_weights,
        ),
        "all_rows": _metrics(
            features=dataset["features"],
            targets=dataset["targets"],
            agent_indices=dataset["agent_indices"],
            indices=np.arange(len(rows)),
            weights=weights,
            bias=bias,
            l2=l2,
            class_weights=class_weights,
        ),
    }


def _schemas_summary(
    *,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    frame_stack: int = 1,
) -> dict[str, object]:
    feature_mode = _validate_feature_mode(feature_mode)
    frame_stack = _validate_frame_stack(frame_stack)
    return {
        "ruleset_id": RULESET_ID,
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "imitation_replay_row_schema_id": IMITATION_REPLAY_ROW_SCHEMA_ID,
        "scoring_replay_row_schema_id": SCORING_REPLAY_ROW_SCHEMA_ID,
        "lookahead_replay_row_schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
        "imitation_train_summary_schema_id": IMITATION_TRAIN_SUMMARY_SCHEMA_ID,
        "imitation_policy_checkpoint_schema_id": IMITATION_POLICY_CHECKPOINT_SCHEMA_ID,
        "feature_encoding_id": _feature_encoding_id(
            feature_mode,
            frame_stack=frame_stack,
        ),
    }


def _input_summary(
    dataset: dict[str, Any],
    *,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    frame_stack: int = 1,
) -> dict[str, object]:
    feature_mode = _validate_feature_mode(feature_mode)
    frame_stack = _validate_frame_stack(frame_stack)
    return {
        "raster_shape": list(dataset["raster_shape"]),
        "feature_shape": [int(dataset["features"].shape[1])],
        "feature_mode": feature_mode,
        "frame_stack": frame_stack,
        "feature_encoding": _feature_encoding_id(
            feature_mode,
            frame_stack=frame_stack,
        ),
        "raster_legend": RASTER_LEGEND,
    }


def _model_summary(weights: np.ndarray, bias: np.ndarray) -> dict[str, object]:
    return {
        "type": "per_ego_agent_softmax_linear_numpy",
        "weights_shape": list(weights.shape),
        "bias_shape": list(bias.shape),
        "ego_agents": list(AGENTS),
    }


def _mlp_model_summary(params: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "type": "per_ego_agent_one_hidden_layer_mlp_numpy",
        "activation": "relu",
        "optimizer": "adam_full_batch",
        "hidden_dim": int(params["hidden_bias"].shape[1]),
        "hidden_weights_shape": list(params["hidden_weights"].shape),
        "hidden_bias_shape": list(params["hidden_bias"].shape),
        "output_weights_shape": list(params["output_weights"].shape),
        "output_bias_shape": list(params["output_bias"].shape),
        "ego_agents": list(AGENTS),
    }


def _class_weights_summary(class_weights: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        agent: {
            label: float(class_weights[agent_index, action_index])
            for action_index, label in enumerate(ACTION_LABELS)
        }
        for agent_index, agent in enumerate(AGENTS)
    }


def _checkpoint_metadata(summary: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "curvyzero_dummy_pong_imitation_policy",
        "checkpoint_schema_id": summary["checkpoint_schema_id"],
        "summary_schema_id": summary["summary_schema_id"],
        "seed": summary["seed"],
        "source_replay_path": summary["source_replay_path"],
        "epochs": summary["epochs"],
        "learning_rate": summary["learning_rate"],
        "validation_fraction": summary["validation_fraction"],
        "l2": summary["l2"],
        "class_weighting": summary["class_weighting"],
        "feature_mode": summary["feature_mode"],
        "frame_stack": summary["frame_stack"],
        "class_weights_by_agent": summary["class_weights_by_agent"],
        "schemas": summary["schemas"],
        "input": summary["input"],
        "model": summary["model"],
        "action_labels": summary["action_labels"],
        "metrics": summary["metrics"],
        "note": summary["note"],
    }


def _mlp_checkpoint_metadata(summary: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "curvyzero_dummy_pong_imitation_mlp_policy",
        "checkpoint_schema_id": summary["checkpoint_schema_id"],
        "summary_schema_id": summary["summary_schema_id"],
        "seed": summary["seed"],
        "source_replay_path": summary["source_replay_path"],
        "epochs": summary["epochs"],
        "learning_rate": summary["learning_rate"],
        "validation_fraction": summary["validation_fraction"],
        "l2": summary["l2"],
        "class_weighting": summary["class_weighting"],
        "feature_mode": summary["feature_mode"],
        "frame_stack": summary["frame_stack"],
        "class_weights_by_agent": summary["class_weights_by_agent"],
        "schemas": summary["schemas"],
        "input": summary["input"],
        "model": summary["model"],
        "action_labels": summary["action_labels"],
        "metrics": summary["metrics"],
        "note": summary["note"],
    }


def _periodic_checkpoint_metadata(
    *,
    seed: int,
    source_replay_path: str,
    epochs: int,
    completed_epochs: int,
    learning_rate: float,
    validation_fraction: float,
    l2: float,
    class_weighting: str,
    feature_mode: str,
    frame_stack: int,
    class_weights: np.ndarray,
    schemas: dict[str, object],
    input_summary: dict[str, object],
    model_summary: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "kind": "curvyzero_dummy_pong_imitation_policy",
        "checkpoint_schema_id": IMITATION_POLICY_CHECKPOINT_SCHEMA_ID,
        "summary_schema_id": IMITATION_TRAIN_SUMMARY_SCHEMA_ID,
        "seed": seed,
        "source_replay_path": source_replay_path,
        "epochs": epochs,
        "completed_epochs": completed_epochs,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "l2": l2,
        "class_weighting": class_weighting,
        "feature_mode": feature_mode,
        "frame_stack": frame_stack,
        "class_weights_by_agent": _class_weights_summary(class_weights),
        "schemas": schemas,
        "input": input_summary,
        "model": model_summary,
        "action_labels": list(ACTION_LABELS),
        "metrics": metrics,
        "note": (
            "Tiny supervised raster policy that copies target_action labels from "
            "dummy Pong replay rows. Labels may come from scripted imitation or "
            "short-lookahead relabeling; this trainer is not MuZero or self-play."
        ),
    }


def _periodic_mlp_checkpoint_metadata(
    *,
    seed: int,
    source_replay_path: str,
    epochs: int,
    completed_epochs: int,
    learning_rate: float,
    validation_fraction: float,
    l2: float,
    class_weighting: str,
    feature_mode: str,
    frame_stack: int,
    hidden_dim: int,
    class_weights: np.ndarray,
    schemas: dict[str, object],
    input_summary: dict[str, object],
    model_summary: dict[str, object],
    metrics: dict[str, object],
) -> dict[str, object]:
    del hidden_dim
    return {
        "kind": "curvyzero_dummy_pong_imitation_mlp_policy",
        "checkpoint_schema_id": IMITATION_MLP_POLICY_CHECKPOINT_SCHEMA_ID,
        "summary_schema_id": IMITATION_TRAIN_SUMMARY_SCHEMA_ID,
        "seed": seed,
        "source_replay_path": source_replay_path,
        "epochs": epochs,
        "completed_epochs": completed_epochs,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "l2": l2,
        "class_weighting": class_weighting,
        "feature_mode": feature_mode,
        "frame_stack": frame_stack,
        "class_weights_by_agent": _class_weights_summary(class_weights),
        "schemas": {
            **schemas,
            "imitation_mlp_policy_checkpoint_schema_id": IMITATION_MLP_POLICY_CHECKPOINT_SCHEMA_ID,
        },
        "input": input_summary,
        "model": model_summary,
        "action_labels": list(ACTION_LABELS),
        "metrics": metrics,
        "note": (
            "Tiny supervised raster MLP policy that copies target_action labels "
            "from dummy Pong replay rows. This trainer is not MuZero or self-play."
        ),
    }


def _agent_index(agent: str) -> int:
    if agent == "player_0":
        return 0
    if agent == "player_1":
        return 1
    raise ValueError(f"unknown agent {agent!r}")


def _validate_feature_mode(feature_mode: str) -> str:
    if feature_mode not in FEATURE_ENCODING_BY_MODE:
        raise ValueError(
            f"feature_mode must be one of {list(FEATURE_MODES)!r}, got {feature_mode!r}"
        )
    return feature_mode


def _validate_model_type(model_type: str) -> str:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {list(MODEL_TYPES)!r}, got {model_type!r}")
    return model_type


def _validate_hidden_dim(hidden_dim: int) -> int:
    if isinstance(hidden_dim, bool):
        raise ValueError("hidden_dim must be an integer")
    hidden_dim = int(hidden_dim)
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be at least 1")
    return hidden_dim


def _validate_frame_stack(frame_stack: int) -> int:
    if isinstance(frame_stack, bool):
        raise ValueError("frame_stack must be an integer")
    frame_stack = int(frame_stack)
    if frame_stack < 1:
        raise ValueError("frame_stack must be at least 1")
    return frame_stack


def _feature_encoding_id(feature_mode: str, *, frame_stack: int = 1) -> str:
    feature_mode = _validate_feature_mode(feature_mode)
    frame_stack = _validate_frame_stack(frame_stack)
    if frame_stack == 1:
        return FEATURE_ENCODING_BY_MODE[feature_mode]
    return FRAME_STACK_FEATURE_ENCODING_BY_MODE[feature_mode]


def _checkpoint_feature_mode(
    metadata: dict[str, object],
    *,
    explicit_feature_mode: str | None,
) -> str:
    if explicit_feature_mode is not None:
        return _validate_feature_mode(explicit_feature_mode)

    metadata_mode = metadata.get("feature_mode")
    if isinstance(metadata_mode, str):
        return _validate_feature_mode(metadata_mode)

    input_summary = metadata.get("input")
    if not isinstance(input_summary, dict):
        input_summary = metadata.get("input_summary")
    if isinstance(input_summary, dict):
        input_mode = input_summary.get("feature_mode")
        if isinstance(input_mode, str):
            return _validate_feature_mode(input_mode)
        input_encoding = input_summary.get("feature_encoding")
        if isinstance(input_encoding, str):
            return _feature_mode_from_encoding_id(input_encoding)

    schemas = metadata.get("schemas")
    if isinstance(schemas, dict):
        schema_encoding = schemas.get("feature_encoding_id")
        if isinstance(schema_encoding, str):
            return _feature_mode_from_encoding_id(schema_encoding)

    return DEFAULT_FEATURE_MODE


def _checkpoint_frame_stack(
    metadata: dict[str, object],
    *,
    explicit_frame_stack: int | None,
) -> int:
    if explicit_frame_stack is not None:
        return _validate_frame_stack(explicit_frame_stack)

    metadata_frame_stack = metadata.get("frame_stack")
    if metadata_frame_stack is not None:
        return _validate_frame_stack(int(metadata_frame_stack))

    input_summary = metadata.get("input")
    if not isinstance(input_summary, dict):
        input_summary = metadata.get("input_summary")
    if isinstance(input_summary, dict):
        input_frame_stack = input_summary.get("frame_stack")
        if input_frame_stack is not None:
            return _validate_frame_stack(int(input_frame_stack))

    return 1


def _feature_mode_from_encoding_id(feature_encoding_id: str) -> str:
    for encoding_by_mode in (FEATURE_ENCODING_BY_MODE, FRAME_STACK_FEATURE_ENCODING_BY_MODE):
        for mode, encoding_id in encoding_by_mode.items():
            if feature_encoding_id == encoding_id:
                return mode
    raise ValueError(f"unsupported feature_encoding_id {feature_encoding_id!r}")


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _npz_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
