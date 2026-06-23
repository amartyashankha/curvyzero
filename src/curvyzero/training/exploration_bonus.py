"""Training-only exploration bonus contracts and RND adapters."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

EXPLORATION_BONUS_MODE_NONE = "none"
EXPLORATION_BONUS_MODE_RND_METER_V0 = "rnd_meter_v0"
EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0 = "rnd_replay_target_v0"
EXPLORATION_BONUS_MODE_CHOICES = (
    EXPLORATION_BONUS_MODE_NONE,
    EXPLORATION_BONUS_MODE_RND_METER_V0,
    EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0,
)

RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0 = "policy_gray64_latest/v0"
RND_FEATURE_SOURCE_CHOICES = (RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0,)
RND_DEFAULT_UPDATE_PER_COLLECT = 100

RND_INPUT_SHAPE_POLICY_GRAY64_LATEST_V0 = (1, 64, 64)
RND_SOURCE_OBSERVATION_SHAPE = (4, 64, 64)
RND_METER_SCHEMA_ID = "curvyzero_exploration_bonus/rnd_meter_v0"
RND_REPLAY_TARGET_SCHEMA_ID = "curvyzero_exploration_bonus/rnd_replay_target_v0"
RND_SCHEMA_ID = RND_METER_SCHEMA_ID
EXPLORATION_BONUS_CONFIG_KEYS = frozenset(
    {
        "schema_id",
        "mode",
        "weight",
        "feature_source",
        "training_only",
        "training_effect",
        "target_reward_effect",
        "trainer_effect",
        "rnd_batch_size",
        "rnd_update_per_collect",
        "rnd_buffer_size",
        "rnd_learning_rate",
        "rnd_weight_decay",
        "rnd_input_norm",
        "input_spec",
        "config_hash",
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and callable(value.detach):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_plain(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _get_attr_or_item(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping) and key in value:
        return value[key]
    return getattr(value, key, default)


def _to_numpy_cpu(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _strict_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a bool")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be integer-valued; got {value!r}")
        parsed = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text.isdecimal():
            raise ValueError(f"{name} must be a base-10 integer string; got {value!r}")
        parsed = int(text)
    else:
        raise ValueError(f"{name} must be an integer; got {type(value).__name__}")
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1")
    return parsed


def _strict_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a strict bool; got {value!r}")


class _ConfigView(dict):
    """Tiny attr-access wrapper for LightZero-style EasyDict configs in tests."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _as_config_view(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return _ConfigView(
        {
            str(key): _as_config_view(item) if isinstance(item, Mapping) else item
            for key, item in value.items()
        }
    )


@dataclass(frozen=True)
class RndInputSpec:
    feature_source: str = RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0
    shape: tuple[int, int, int] = RND_INPUT_SHAPE_POLICY_GRAY64_LATEST_V0
    source_observation_shape: tuple[int, int, int] = RND_SOURCE_OBSERVATION_SHAPE
    layout: str = "NCHW"
    dtype: str = "float32"
    value_min: float = 0.0
    value_max: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_source": self.feature_source,
            "shape": list(self.shape),
            "source_observation_shape": list(self.source_observation_shape),
            "layout": self.layout,
            "dtype": self.dtype,
            "value_min": self.value_min,
            "value_max": self.value_max,
        }


@dataclass(frozen=True)
class ExplorationBonusSpec:
    mode: str = EXPLORATION_BONUS_MODE_NONE
    weight: float = 0.0
    feature_source: str = RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0
    rnd_batch_size: int = 64
    rnd_update_per_collect: int = RND_DEFAULT_UPDATE_PER_COLLECT
    rnd_buffer_size: int = 100_000
    rnd_learning_rate: float = 3e-4
    rnd_weight_decay: float = 1e-4
    rnd_input_norm: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode != EXPLORATION_BONUS_MODE_NONE

    @property
    def schema_id(self) -> str:
        if self.mode == EXPLORATION_BONUS_MODE_NONE:
            return "curvyzero_exploration_bonus/none/v0"
        if self.mode == EXPLORATION_BONUS_MODE_RND_METER_V0:
            return f"{RND_METER_SCHEMA_ID}/v0"
        if self.mode == EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0:
            return f"{RND_REPLAY_TARGET_SCHEMA_ID}/v0"
        raise ValueError(f"unknown exploration_bonus_mode {self.mode!r}")

    @property
    def training_effect(self) -> str:
        if self.mode == EXPLORATION_BONUS_MODE_RND_METER_V0:
            return "reward_target_unchanged"
        if self.mode == EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0:
            return "reward_target_augmented_by_intrinsic_rnd"
        return "disabled"

    @property
    def trainer_effect(self) -> str:
        if self.mode == EXPLORATION_BONUS_MODE_RND_METER_V0:
            return "uses_reward_model_entrypoint_and_trains_rnd_meter"
        if self.mode == EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0:
            return "uses_reward_model_entrypoint_and_trains_rnd_replay_target"
        return "uses_stock_muzero_entrypoint"

    @property
    def target_reward_effect(self) -> str:
        if self.mode == EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0:
            return "intrinsic_weighted_addition"
        return "unchanged"

    @property
    def input_spec(self) -> RndInputSpec | None:
        if not self.enabled:
            return None
        return RndInputSpec(feature_source=self.feature_source)

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_id": self.schema_id,
            "mode": self.mode,
            "weight": float(self.weight),
            "feature_source": self.feature_source,
            "training_only": True,
            "training_effect": self.training_effect,
            "target_reward_effect": self.target_reward_effect,
            "trainer_effect": self.trainer_effect,
            "rnd_batch_size": int(self.rnd_batch_size),
            "rnd_update_per_collect": int(self.rnd_update_per_collect),
            "rnd_buffer_size": int(self.rnd_buffer_size),
            "rnd_learning_rate": float(self.rnd_learning_rate),
            "rnd_weight_decay": float(self.rnd_weight_decay),
            "rnd_input_norm": bool(self.rnd_input_norm),
            "input_spec": self.input_spec.as_dict() if self.input_spec else None,
        }
        if include_hash:
            payload["config_hash"] = self.config_hash()
        return payload

    def config_hash(self) -> str:
        return _stable_hash(self.as_dict(include_hash=False))


def normalize_exploration_bonus_spec(
    *,
    mode: str = EXPLORATION_BONUS_MODE_NONE,
    weight: float = 0.0,
    feature_source: str = RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0,
    rnd_batch_size: int = 64,
    rnd_update_per_collect: int = RND_DEFAULT_UPDATE_PER_COLLECT,
    rnd_buffer_size: int = 100_000,
    rnd_learning_rate: float = 3e-4,
    rnd_weight_decay: float = 1e-4,
    rnd_input_norm: bool = False,
) -> ExplorationBonusSpec:
    mode = str(mode)
    if mode not in EXPLORATION_BONUS_MODE_CHOICES:
        raise ValueError(
            f"unknown exploration_bonus_mode {mode!r}; "
            f"expected one of {EXPLORATION_BONUS_MODE_CHOICES!r}"
        )
    feature_source = str(feature_source)
    if feature_source not in RND_FEATURE_SOURCE_CHOICES:
        raise ValueError(
            f"unknown exploration_bonus_feature_source {feature_source!r}; "
            f"expected one of {RND_FEATURE_SOURCE_CHOICES!r}"
        )
    weight = float(weight)
    if mode == EXPLORATION_BONUS_MODE_NONE and weight != 0.0:
        raise ValueError("exploration_bonus_weight must be 0.0 when mode='none'")
    if mode == EXPLORATION_BONUS_MODE_RND_METER_V0 and weight != 0.0:
        raise ValueError("rnd_meter_v0 is metric-only; exploration_bonus_weight must be 0.0")
    if mode == EXPLORATION_BONUS_MODE_RND_REPLAY_TARGET_V0:
        if weight <= 0.0:
            raise ValueError("rnd_replay_target_v0 requires exploration_bonus_weight > 0.0")
        if weight > 1.0:
            raise ValueError("rnd_replay_target_v0 exploration_bonus_weight must be <= 1.0")
    rnd_batch_size = _strict_positive_int("rnd_batch_size", rnd_batch_size)
    rnd_update_per_collect = _strict_positive_int("rnd_update_per_collect", rnd_update_per_collect)
    rnd_buffer_size = _strict_positive_int("rnd_buffer_size", rnd_buffer_size)
    if float(rnd_learning_rate) <= 0.0:
        raise ValueError("rnd_learning_rate must be positive")
    if float(rnd_weight_decay) < 0.0:
        raise ValueError("rnd_weight_decay must be non-negative")
    rnd_input_norm = _strict_bool("rnd_input_norm", rnd_input_norm)
    return ExplorationBonusSpec(
        mode=mode,
        weight=weight,
        feature_source=feature_source,
        rnd_batch_size=rnd_batch_size,
        rnd_update_per_collect=rnd_update_per_collect,
        rnd_buffer_size=rnd_buffer_size,
        rnd_learning_rate=float(rnd_learning_rate),
        rnd_weight_decay=float(rnd_weight_decay),
        rnd_input_norm=rnd_input_norm,
    )


def normalize_exploration_bonus_config(value: Any = None) -> ExplorationBonusSpec:
    if value is None:
        return normalize_exploration_bonus_spec()
    if isinstance(value, ExplorationBonusSpec):
        return value
    if isinstance(value, str):
        return normalize_exploration_bonus_spec(mode=value)
    if isinstance(value, Mapping):
        unknown_keys = sorted(
            str(key) for key in value if str(key) not in EXPLORATION_BONUS_CONFIG_KEYS
        )
        if unknown_keys:
            raise ValueError("unknown exploration_bonus config fields: " + ", ".join(unknown_keys))
        spec = normalize_exploration_bonus_spec(
            mode=value.get("mode", EXPLORATION_BONUS_MODE_NONE),
            weight=value.get("weight", 0.0),
            feature_source=value.get("feature_source", RND_FEATURE_SOURCE_POLICY_GRAY64_LATEST_V0),
            rnd_batch_size=value.get("rnd_batch_size", 64),
            rnd_update_per_collect=value.get(
                "rnd_update_per_collect", RND_DEFAULT_UPDATE_PER_COLLECT
            ),
            rnd_buffer_size=value.get("rnd_buffer_size", 100_000),
            rnd_learning_rate=value.get("rnd_learning_rate", 3e-4),
            rnd_weight_decay=value.get("rnd_weight_decay", 1e-4),
            rnd_input_norm=value.get("rnd_input_norm", False),
        )
        expected_hash = value.get("config_hash")
        if expected_hash is not None and str(expected_hash) != spec.config_hash():
            raise ValueError(
                "exploration_bonus config_hash does not match normalized config: "
                f"expected {spec.config_hash()!r}, got {expected_hash!r}"
            )
        expected_schema = value.get("schema_id")
        if expected_schema is not None and str(expected_schema) != spec.schema_id:
            raise ValueError(
                "exploration_bonus schema_id does not match normalized config: "
                f"expected {spec.schema_id!r}, got {expected_schema!r}"
            )
        expected_training_effect = value.get("training_effect")
        if (
            expected_training_effect is not None
            and str(expected_training_effect) != spec.training_effect
        ):
            raise ValueError(
                "exploration_bonus training_effect does not match normalized config: "
                f"expected {spec.training_effect!r}, got {expected_training_effect!r}"
            )
        expected_training_only = value.get("training_only")
        if expected_training_only is not None and not _strict_bool(
            "training_only",
            expected_training_only,
        ):
            raise ValueError("exploration_bonus training_only must be true")
        expected_target_reward_effect = value.get("target_reward_effect")
        target_reward_effect = spec.as_dict(include_hash=False)["target_reward_effect"]
        if (
            expected_target_reward_effect is not None
            and str(expected_target_reward_effect) != target_reward_effect
        ):
            raise ValueError(
                "exploration_bonus target_reward_effect does not match normalized config: "
                f"expected {target_reward_effect!r}, got {expected_target_reward_effect!r}"
            )
        expected_trainer_effect = value.get("trainer_effect")
        if (
            expected_trainer_effect is not None
            and str(expected_trainer_effect) != spec.trainer_effect
        ):
            raise ValueError(
                "exploration_bonus trainer_effect does not match normalized config: "
                f"expected {spec.trainer_effect!r}, got {expected_trainer_effect!r}"
            )
        missing = object()
        expected_input_spec = value.get("input_spec", missing)
        if expected_input_spec is not missing:
            if _plain(expected_input_spec) != _plain(
                spec.input_spec.as_dict() if spec.input_spec else None
            ):
                raise ValueError("exploration_bonus input_spec does not match normalized config")
        return spec
    raise TypeError(f"unsupported exploration bonus config: {type(value).__name__}")


def lightzero_entrypoint_name(spec: ExplorationBonusSpec) -> str:
    if spec.enabled:
        return "train_muzero_with_reward_model"
    return "train_muzero"


def lightzero_trainer_entrypoint_ref(spec: ExplorationBonusSpec) -> str:
    return f"lzero.entry.{lightzero_entrypoint_name(spec)}"


def lightzero_reward_model_config(
    spec: ExplorationBonusSpec,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    if not spec.enabled:
        return {}
    input_spec = spec.input_spec
    if input_spec is None:
        raise ValueError("RND mode requires an input spec")
    payload = {
        "type": "rnd_muzero",
        "intrinsic_reward_type": "add",
        "input_type": "obs",
        "intrinsic_reward_weight": float(spec.weight),
        "obs_shape": list(input_spec.shape),
        "latent_state_dim": 512,
        "hidden_size_list": [64, 64, 128],
        "learning_rate": float(spec.rnd_learning_rate),
        "weight_decay": float(spec.rnd_weight_decay),
        "batch_size": int(spec.rnd_batch_size),
        "update_per_collect": int(spec.rnd_update_per_collect),
        "rnd_buffer_size": int(spec.rnd_buffer_size),
        "input_norm": bool(spec.rnd_input_norm),
        "input_norm_clamp_max": 5,
        "input_norm_clamp_min": -5,
        "extrinsic_reward_norm": False,
        "extrinsic_reward_norm_max": 1,
        "curvyzero_adapter": {
            "schema_id": "curvyzero_rnd_replay_batch_adapter/v0",
            **input_spec.as_dict(),
        },
        "curvyzero_exploration_bonus": spec.as_dict(),
    }
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


def _set_path(mapping: MutableMapping[str, Any], path: Sequence[str], value: Any) -> dict[str, Any]:
    current: MutableMapping[str, Any] = mapping
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, MutableMapping):
            child = {}
            current[key] = child
        current = child
    leaf = path[-1]
    old = current.get(leaf)
    current[leaf] = value
    return {"path": ".".join(path), "old": _plain(old), "new": _plain(value)}


def apply_lightzero_exploration_bonus_config(
    main_config: MutableMapping[str, Any],
    create_config: MutableMapping[str, Any],
    spec: ExplorationBonusSpec,
    *,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    del create_config
    if not spec.enabled:
        return []
    metadata = spec.as_dict()
    reward_model = lightzero_reward_model_config(spec, seed=seed)
    patches = [
        _set_path(main_config, ("reward_model",), reward_model),
        _set_path(main_config, ("policy", "use_rnd_model"), True),
        _set_path(main_config, ("policy", "use_momentum_representation_network"), True),
        _set_path(
            main_config,
            ("policy", "target_model_for_intrinsic_reward_update_type"),
            "assign",
        ),
        _set_path(main_config, ("policy", "target_update_freq_for_intrinsic_reward"), 1000),
        _set_path(main_config, ("policy", "target_update_theta_for_intrinsic_reward"), 0.005),
        _set_path(main_config, ("env", "exploration_bonus"), metadata),
    ]
    for patch in patches:
        patch["reason"] = "enable CurvyZero RND exploration plumbing"
    return patches


def extract_policy_gray64_latest_for_rnd(
    obs_batch: Any,
    target_reward: Any,
    *,
    source_observation_shape: tuple[int, int, int] = RND_SOURCE_OBSERVATION_SHAPE,
) -> np.ndarray:
    obs = _to_numpy_cpu(obs_batch, dtype=np.float32)
    target = _to_numpy_cpu(target_reward)
    if target.ndim < 2:
        raise ValueError("target_reward must expose batch and unroll dimensions")
    batch_size = int(target.shape[0])
    unroll_len = int(target.shape[1])
    channels, height, width = source_observation_shape
    flat_frame = channels * height * width
    flat_sequence = unroll_len * flat_frame

    if (
        obs.ndim == 5
        and obs.shape[:2] == (batch_size, unroll_len)
        and obs.shape[2:] == source_observation_shape
    ):
        stacked = obs
    elif obs.ndim == 4 and obs.shape[0] == batch_size * unroll_len:
        stacked = obs.reshape(batch_size, unroll_len, channels, height, width)
    elif (
        obs.ndim == 4
        and obs.shape[0] == batch_size
        and obs.shape[1:] == (unroll_len * channels, height, width)
    ):
        stacked = obs.reshape(batch_size, unroll_len, channels, height, width)
    elif (
        obs.ndim == 4
        and obs.shape[0] == batch_size
        and obs.shape[1:] == source_observation_shape
        and unroll_len == 1
    ):
        stacked = obs.reshape(batch_size, 1, channels, height, width)
    elif obs.ndim == 3 and obs.shape == (batch_size, unroll_len, flat_frame):
        stacked = obs.reshape(batch_size, unroll_len, channels, height, width)
    elif obs.ndim == 2 and obs.shape == (batch_size, flat_sequence):
        stacked = obs.reshape(batch_size, unroll_len, channels, height, width)
    else:
        raise ValueError(
            "obs_batch shape is not compatible with Curvy RND latest-frame extraction: "
            f"obs_shape={tuple(obs.shape)!r}, target_reward_shape={tuple(target.shape)!r}, "
            f"source_observation_shape={source_observation_shape!r}"
        )
    latest = stacked[:, :, -1:, :, :].reshape(batch_size * unroll_len, 1, height, width)
    if not np.isfinite(latest).all():
        raise ValueError("RND input contains non-finite values")
    if latest.size and (float(latest.min()) < -1e-6 or float(latest.max()) > 1.0 + 1e-6):
        raise ValueError(
            "RND input must be normalized to [0, 1]; "
            f"observed min={float(latest.min()):g}, max={float(latest.max()):g}"
        )
    return latest.astype(np.float32, copy=False)


def normalize_policy_gray64_stack_for_rnd(
    obs_batch: Any,
    *,
    source_observation_shape: tuple[int, int, int] = RND_SOURCE_OBSERVATION_SHAPE,
) -> np.ndarray:
    """Return normalized row/root policy stacks for Curvy RND.

    Accepts either scalar/root-major `[N,4,64,64]` stacks or compact
    row/player `[B,P,4,64,64]` stacks. `uint8` inputs are interpreted as
    policy pixels and scaled to `[0,1]`; floating inputs must already be
    normalized.
    """

    obs = _to_numpy_cpu(obs_batch)
    channels, height, width = source_observation_shape
    if obs.ndim == 5 and tuple(obs.shape[2:]) == source_observation_shape:
        stack = obs.reshape(int(obs.shape[0]) * int(obs.shape[1]), channels, height, width)
    elif obs.ndim == 4 and tuple(obs.shape[1:]) == source_observation_shape:
        stack = obs
    else:
        raise ValueError(
            "obs_batch shape is not compatible with compact Curvy RND stack extraction: "
            f"obs_shape={tuple(obs.shape)!r}, "
            f"source_observation_shape={source_observation_shape!r}"
        )
    if np.issubdtype(stack.dtype, np.integer):
        if stack.dtype != np.uint8:
            raise ValueError(f"integer RND stack inputs must be uint8, got {stack.dtype}")
        normalized = stack.astype(np.float32) * np.float32(1.0 / 255.0)
    else:
        normalized = stack.astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError("RND input contains non-finite values")
    if normalized.size and (
        float(normalized.min()) < -1e-6 or float(normalized.max()) > 1.0 + 1e-6
    ):
        raise ValueError(
            "RND input must be normalized to [0, 1]; "
            f"observed min={float(normalized.min()):g}, max={float(normalized.max()):g}"
        )
    return np.ascontiguousarray(normalized, dtype=np.float32)


def extract_policy_gray64_latest_for_rnd_from_compact_observation(
    obs_batch: Any,
    target_reward: Any,
    *,
    source_observation_shape: tuple[int, int, int] = RND_SOURCE_OBSERVATION_SHAPE,
) -> np.ndarray:
    """Extract RND latest frames from compact row/player policy observations."""

    obs = _to_numpy_cpu(obs_batch)
    channels, height, width = source_observation_shape
    if obs.ndim == 5 and tuple(obs.shape[2:]) == source_observation_shape:
        root_count = int(obs.shape[0]) * int(obs.shape[1])
        latest = obs[:, :, channels - 1 : channels, :, :].reshape(root_count, 1, height, width)
    elif obs.ndim == 4 and tuple(obs.shape[1:]) == source_observation_shape:
        root_count = int(obs.shape[0])
        latest = obs[:, channels - 1 : channels, :, :]
    else:
        raise ValueError(
            "obs_batch shape is not compatible with compact Curvy RND latest-frame extraction: "
            f"obs_shape={tuple(obs.shape)!r}, "
            f"source_observation_shape={source_observation_shape!r}"
        )
    target = _to_numpy_cpu(target_reward)
    if target.shape != (root_count, 1):
        raise ValueError(
            "compact RND target_reward must have shape [B*P, 1]; "
            f"got target_reward_shape={tuple(target.shape)!r}, root_count={root_count!r}"
        )
    if np.issubdtype(latest.dtype, np.integer):
        if latest.dtype != np.uint8:
            raise ValueError(f"integer RND latest-frame inputs must be uint8, got {latest.dtype}")
        normalized = latest.astype(np.float32)
        normalized *= np.float32(1.0 / 255.0)
    else:
        normalized = latest.astype(np.float32, copy=False)
        if not np.isfinite(normalized).all():
            raise ValueError("RND input contains non-finite values")
        if normalized.size and (
            float(normalized.min()) < -1e-6 or float(normalized.max()) > 1.0 + 1e-6
        ):
            raise ValueError(
                "RND input must be normalized to [0, 1]; "
                f"observed min={float(normalized.min()):g}, max={float(normalized.max()):g}"
            )
    return normalized


class CurvyRNDRewardModel:
    """Curvy image-shape-safe RND reward model for LightZero's reward-model entrypoint."""

    def __init__(
        self,
        config: Any,
        device: str = "cpu",
        tb_logger: Any = None,
        representation_network: Any = None,
        target_representation_network: Any = None,
        use_momentum_representation_network: bool = True,
    ) -> None:
        del (
            representation_network,
            target_representation_network,
            use_momentum_representation_network,
        )
        self.cfg = _as_config_view(config)
        self.device = str(device)
        self.tb_logger = tb_logger
        self.input_type = str(_get_attr_or_item(self.cfg, "input_type", "obs"))
        if self.input_type != "obs":
            raise ValueError("CurvyRNDRewardModel supports only input_type='obs'")
        adapter = _get_attr_or_item(self.cfg, "curvyzero_adapter", {}) or {}
        self.source_observation_shape = tuple(
            int(item)
            for item in _get_attr_or_item(
                adapter, "source_observation_shape", RND_SOURCE_OBSERVATION_SHAPE
            )
        )
        self.input_shape = tuple(
            int(item)
            for item in _get_attr_or_item(adapter, "shape", RND_INPUT_SHAPE_POLICY_GRAY64_LATEST_V0)
        )
        if self.input_shape != RND_INPUT_SHAPE_POLICY_GRAY64_LATEST_V0:
            raise ValueError(f"unsupported Curvy RND input shape: {self.input_shape!r}")
        self.intrinsic_reward_type = str(
            _get_attr_or_item(self.cfg, "intrinsic_reward_type", "add")
        )
        if self.intrinsic_reward_type != "add":
            raise ValueError("rnd_meter_v0 supports only intrinsic_reward_type='add'")
        self.train_obs: list[Any] = []
        seed_raw = _get_attr_or_item(self.cfg, "seed", None)
        self.seed = None if seed_raw is None else int(seed_raw)
        self._sample_rng = random.Random(self.seed)
        self.estimate_cnt_rnd = 0
        self.train_cnt_rnd = 0
        self.collect_data_calls = 0
        self.train_with_data_calls = 0
        self.train_with_data_skipped_small_buffer_count = 0
        self.estimate_calls = 0
        self.last_train_loss: float | None = None
        self.last_raw_mse_mean: float | None = None
        self.last_raw_mse_min: float | None = None
        self.last_raw_mse_max: float | None = None
        self.last_raw_mse_std: float | None = None
        self.last_raw_mse_p50: float | None = None
        self.last_raw_mse_p95: float | None = None
        self.last_intrinsic_mean: float | None = None
        self.last_intrinsic_min: float | None = None
        self.last_intrinsic_max: float | None = None
        self.last_target_reward_changed: bool | None = None
        self.last_target_reward_delta_abs_mean: float | None = None
        self.last_target_reward_delta_abs_max: float | None = None
        self.last_predictor_hash_before_train: str | None = None
        self.last_predictor_hash_after_train: str | None = None
        self.last_target_hash_before_train: str | None = None
        self.last_target_hash_after_train: str | None = None
        self._metrics_write_errors: list[str] = []
        self._torch = None
        self.reward_model = None
        self._optimizer_rnd = None
        self._init_torch_model()
        self._write_metrics_snapshot("init")

    def _init_torch_model(self) -> None:
        import torch
        import torch.nn as nn

        if bool(_get_attr_or_item(self.cfg, "disable_cudnn", False)):
            torch.backends.cudnn.enabled = False

        class _TinyRNDNetwork(nn.Module):
            def __init__(self, input_shape: tuple[int, int, int], feature_dim: int) -> None:
                super().__init__()
                channels, height, width = input_shape

                def tower() -> nn.Sequential:
                    return nn.Sequential(
                        nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(32 * ((height + 3) // 4) * ((width + 3) // 4), feature_dim),
                        nn.ReLU(),
                    )

                self.target = tower()
                self.predictor = tower()
                for param in self.target.parameters():
                    param.requires_grad = False

            def forward(self, obs: Any) -> tuple[Any, Any]:
                prediction = self.predictor(obs)
                with torch.no_grad():
                    target = self.target(obs)
                return prediction, target

        self._torch = torch
        feature_dim = int((_get_attr_or_item(self.cfg, "hidden_size_list", [128]) or [128])[-1])
        if self.seed is None:
            self.reward_model = _TinyRNDNetwork(self.input_shape, feature_dim).to(self.device)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(self.seed))
                self.reward_model = _TinyRNDNetwork(self.input_shape, feature_dim).to(self.device)
        self._optimizer_rnd = torch.optim.Adam(
            self.reward_model.predictor.parameters(),
            lr=float(_get_attr_or_item(self.cfg, "learning_rate", 3e-4)),
            weight_decay=float(_get_attr_or_item(self.cfg, "weight_decay", 1e-4)),
        )

    def _log_scalar(self, name: str, value: Any, step: int) -> None:
        if self.tb_logger is not None and hasattr(self.tb_logger, "add_scalar"):
            self.tb_logger.add_scalar(name, value, step)

    def _state_hash(self, module_name: str) -> str | None:
        module = getattr(self.reward_model, module_name, None)
        if module is None:
            return None
        digest = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(str(name).encode("utf-8"))
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str(array.shape).encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def model_state_hash(self, module_name: str) -> str | None:
        if module_name not in {"predictor", "target"}:
            raise ValueError("module_name must be 'predictor' or 'target'")
        return self._state_hash(module_name)

    def metrics_snapshot(self, *, reason: str = "snapshot") -> dict[str, Any]:
        return {
            "schema_id": "curvyzero_rnd_reward_model_metrics/v0",
            "reason": str(reason),
            "constructed": True,
            "device": self.device,
            "disable_cudnn": bool(_get_attr_or_item(self.cfg, "disable_cudnn", False)),
            "input_type": self.input_type,
            "input_shape": list(self.input_shape),
            "source_observation_shape": list(self.source_observation_shape),
            "seed": self.seed,
            "intrinsic_reward_weight": float(
                _get_attr_or_item(self.cfg, "intrinsic_reward_weight", 0.0)
            ),
            "buffer_count": len(self.train_obs),
            "collect_data_calls": int(self.collect_data_calls),
            "train_with_data_calls": int(self.train_with_data_calls),
            "train_with_data_skipped_small_buffer_count": int(
                self.train_with_data_skipped_small_buffer_count
            ),
            "train_cnt_rnd": int(self.train_cnt_rnd),
            "estimate_calls": int(self.estimate_calls),
            "estimate_cnt_rnd": int(self.estimate_cnt_rnd),
            "train_cnt_per_estimate": (
                None
                if int(self.estimate_cnt_rnd) <= 0
                else float(self.train_cnt_rnd) / float(self.estimate_cnt_rnd)
            ),
            "train_with_data_calls_per_collect": (
                None
                if int(self.collect_data_calls) <= 0
                else float(self.train_with_data_calls) / float(self.collect_data_calls)
            ),
            "last_train_loss": self.last_train_loss,
            "last_raw_mse_mean": self.last_raw_mse_mean,
            "last_raw_mse_min": self.last_raw_mse_min,
            "last_raw_mse_max": self.last_raw_mse_max,
            "last_raw_mse_std": self.last_raw_mse_std,
            "last_raw_mse_p50": self.last_raw_mse_p50,
            "last_raw_mse_p95": self.last_raw_mse_p95,
            "last_intrinsic_mean": self.last_intrinsic_mean,
            "last_intrinsic_min": self.last_intrinsic_min,
            "last_intrinsic_max": self.last_intrinsic_max,
            "last_target_reward_changed": self.last_target_reward_changed,
            "last_target_reward_delta_abs_mean": self.last_target_reward_delta_abs_mean,
            "last_target_reward_delta_abs_max": self.last_target_reward_delta_abs_max,
            "last_predictor_hash_before_train": self.last_predictor_hash_before_train,
            "last_predictor_hash_after_train": self.last_predictor_hash_after_train,
            "last_target_hash_before_train": self.last_target_hash_before_train,
            "last_target_hash_after_train": self.last_target_hash_after_train,
            "predictor_hash": self._state_hash("predictor"),
            "target_hash": self._state_hash("target"),
            "metrics_write_errors": list(self._metrics_write_errors[-5:]),
        }

    def _write_metrics_snapshot(self, reason: str) -> None:
        latest_path_raw = _get_attr_or_item(self.cfg, "curvyzero_metrics_latest_path", None)
        jsonl_path_raw = _get_attr_or_item(self.cfg, "curvyzero_metrics_jsonl_path", None)
        if not latest_path_raw and not jsonl_path_raw:
            return
        try:
            payload = self.metrics_snapshot(reason=reason)
            text = json.dumps(_plain(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n"
            if latest_path_raw:
                latest_path = Path(str(latest_path_raw))
                latest_path.parent.mkdir(parents=True, exist_ok=True)
                latest_path.write_text(text, encoding="utf-8")
            if jsonl_path_raw:
                jsonl_path = Path(str(jsonl_path_raw))
                jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(_plain(payload), sort_keys=True) + "\n")
        except Exception as exc:  # pragma: no cover - sidecar diagnostics only.
            self._metrics_write_errors.append(f"{type(exc).__name__}: {exc}")

    def collect_data(self, data: Any) -> None:
        self.collect_data_calls += 1
        segments = data[0] if isinstance(data, (list, tuple)) and data else []
        limit = int(_get_attr_or_item(self.cfg, "rnd_buffer_size", 100_000))
        for game_segment in segments:
            obs_segment = getattr(game_segment, "obs_segment", None)
            if obs_segment is None:
                continue
            obs = np.asarray(obs_segment, dtype=np.float32)
            if obs.ndim != 4 or tuple(obs.shape[1:]) != self.source_observation_shape:
                continue
            latest = obs[:, -1:, :, :]
            if not np.isfinite(latest).all():
                continue
            if latest.size and (float(latest.min()) < -1e-6 or float(latest.max()) > 1.0 + 1e-6):
                continue
            compact_latest = np.ascontiguousarray(latest)
            self.train_obs.extend(
                self._torch.from_numpy(np.ascontiguousarray(frame)).clone()
                for frame in compact_latest
            )
        if len(self.train_obs) > limit:
            self.train_obs = self.train_obs[-limit:]
        self._write_metrics_snapshot("collect_data")

    def train_with_data(self) -> None:
        self.train_with_data_calls += 1
        batch_size = int(_get_attr_or_item(self.cfg, "batch_size", 64))
        if len(self.train_obs) < batch_size:
            self.train_with_data_skipped_small_buffer_count += 1
            self._write_metrics_snapshot("train_with_data_skipped_small_buffer")
            return
        import torch.nn.functional as F

        updates = int(_get_attr_or_item(self.cfg, "update_per_collect", 1))
        self.last_predictor_hash_before_train = self._state_hash("predictor")
        self.last_target_hash_before_train = self._state_hash("target")
        for _ in range(updates):
            train_data = self._torch.stack(self._sample_rng.sample(self.train_obs, batch_size)).to(
                self.device
            )
            prediction, target = self.reward_model(train_data)
            loss = F.mse_loss(prediction, target)
            self._optimizer_rnd.zero_grad()
            loss.backward()
            self._optimizer_rnd.step()
            self.last_train_loss = float(loss.detach().cpu().item())
            self._log_scalar(
                "rnd_reward_model/rnd_mse_loss",
                self.last_train_loss,
                self.train_cnt_rnd,
            )
            self.train_cnt_rnd += 1
        self.last_predictor_hash_after_train = self._state_hash("predictor")
        self.last_target_hash_after_train = self._state_hash("target")
        self._write_metrics_snapshot("train_with_data")

    def estimate(self, data: Any) -> Any:
        self.estimate_calls += 1
        obs_batch_orig = data[0][0]
        target_reward = data[1][0]
        rnd_input = extract_policy_gray64_latest_for_rnd(
            obs_batch_orig,
            target_reward,
            source_observation_shape=self.source_observation_shape,
        )
        with self._torch.no_grad():
            input_tensor = self._torch.from_numpy(rnd_input).to(self.device)
            prediction, target = self.reward_model(input_tensor)
            mse = self._torch.nn.functional.mse_loss(prediction, target, reduction="none").mean(
                dim=1
            )
            mse_cpu = mse.detach().cpu().numpy()
            self.last_raw_mse_mean = float(mse_cpu.mean()) if mse_cpu.size else 0.0
            self.last_raw_mse_min = float(mse_cpu.min()) if mse_cpu.size else 0.0
            self.last_raw_mse_max = float(mse_cpu.max()) if mse_cpu.size else 0.0
            self.last_raw_mse_std = float(mse_cpu.std()) if mse_cpu.size else 0.0
            self.last_raw_mse_p50 = float(np.percentile(mse_cpu, 50)) if mse_cpu.size else 0.0
            self.last_raw_mse_p95 = float(np.percentile(mse_cpu, 95)) if mse_cpu.size else 0.0
            rnd_reward = (mse - mse.min()) / (mse.max() - mse.min() + 1e-6)
            self.estimate_cnt_rnd += 1
            self.last_intrinsic_mean = float(rnd_reward.mean().detach().cpu().item())
            self.last_intrinsic_min = float(rnd_reward.min().detach().cpu().item())
            self.last_intrinsic_max = float(rnd_reward.max().detach().cpu().item())
            self._log_scalar(
                "rnd_reward_model/rnd_reward_mean",
                self.last_intrinsic_mean,
                self.estimate_cnt_rnd,
            )
        weight = float(_get_attr_or_item(self.cfg, "intrinsic_reward_weight", 0.0))
        if weight != 0.0:
            target_reward_augmented = copy.deepcopy(target_reward)
            reshaped = rnd_reward.unsqueeze(1).cpu().numpy().reshape(target_reward.shape)
            target_reward_augmented = target_reward_augmented + reshaped * weight
            self.last_target_reward_changed = not np.array_equal(
                _to_numpy_cpu(target_reward),
                _to_numpy_cpu(target_reward_augmented),
            )
            delta = np.abs(
                _to_numpy_cpu(target_reward_augmented, dtype=np.float32)
                - _to_numpy_cpu(target_reward, dtype=np.float32)
            )
            self.last_target_reward_delta_abs_mean = float(delta.mean()) if delta.size else 0.0
            self.last_target_reward_delta_abs_max = float(delta.max()) if delta.size else 0.0
        else:
            target_reward_augmented = target_reward
            self.last_target_reward_changed = False
            self.last_target_reward_delta_abs_mean = 0.0
            self.last_target_reward_delta_abs_max = 0.0
        output = list(data)
        output[1] = list(output[1])
        output[1][0] = target_reward_augmented
        self._write_metrics_snapshot("estimate")
        return output

    def clear_old_data(self) -> None:
        limit = int(_get_attr_or_item(self.cfg, "rnd_buffer_size", 100_000))
        if len(self.train_obs) > limit:
            self.train_obs = self.train_obs[-limit:]

    def state_dict(self) -> dict[str, Any]:
        return {
            "reward_model": self.reward_model.state_dict(),
            "optimizer": self._optimizer_rnd.state_dict(),
            "estimate_cnt_rnd": int(self.estimate_cnt_rnd),
            "train_cnt_rnd": int(self.train_cnt_rnd),
            "train_with_data_skipped_small_buffer_count": int(
                self.train_with_data_skipped_small_buffer_count
            ),
            "sample_rng_state": self._sample_rng.getstate(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if "reward_model" in state_dict:
            self.reward_model.load_state_dict(state_dict["reward_model"])
        else:
            self.reward_model.load_state_dict(state_dict)
        if "optimizer" in state_dict:
            self._optimizer_rnd.load_state_dict(state_dict["optimizer"])
        self.estimate_cnt_rnd = int(state_dict.get("estimate_cnt_rnd", self.estimate_cnt_rnd))
        self.train_cnt_rnd = int(state_dict.get("train_cnt_rnd", self.train_cnt_rnd))
        if "sample_rng_state" in state_dict:
            self._sample_rng.setstate(state_dict["sample_rng_state"])
        self.train_with_data_skipped_small_buffer_count = int(
            state_dict.get(
                "train_with_data_skipped_small_buffer_count",
                self.train_with_data_skipped_small_buffer_count,
            )
        )

    def clear_data(self) -> None:
        self.train_obs.clear()

    def train(self) -> None:
        self.train_with_data()
