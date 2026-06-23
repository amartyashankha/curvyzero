"""Frozen LightZero checkpoint provider for snapshot-backed opponents."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from curvyzero.env.observation_surface_contract import (
    DEFAULT_POLICY_OBSERVATION_BACKEND,
    POLICY_BONUS_RENDER_MODE,
    POLICY_OBSERVATION_CONTRACT_ID,
    POLICY_OBSERVATION_PERSPECTIVE,
    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
    POLICY_OBSERVATION_SEAT_MAPPING,
    POLICY_STACK_SHAPE,
    POLICY_TRAIL_RENDER_MODE,
)
from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
)
from curvyzero.training.multiplayer_opponent_policy import (
    OpponentActionChoice,
    SnapshotBackedOpponentPolicy,
)


LIGHTZERO_CHECKPOINT_OPPONENT_PROVIDER_ID = (
    "curvyzero_lightzero_checkpoint_opponent_provider"
)
LIGHTZERO_CHECKPOINT_OPPONENT_PROVIDER_VERSION = "v0.2026-05-10"

_POLICY_OBSERVATION_METADATA_FIELD_SOURCES = {
    "policy_trail_render_mode": (
        "policy_trail_render_mode",
        "trail_render_mode",
        "source_state_trail_render_mode",
    ),
    "policy_bonus_render_mode": (
        "policy_bonus_render_mode",
        "bonus_render_mode",
        "source_state_bonus_render_mode",
    ),
    "policy_observation_backend": (
        "policy_observation_backend",
        "observation_backend",
    ),
    "policy_observation_contract_id": (
        "policy_observation_contract_id",
        "observation_contract_id",
    ),
    "policy_observation_perspective_schema_id": (
        "policy_observation_perspective_schema_id",
        "player_perspective_schema_id",
        "perspective_schema_id",
    ),
}

_POLICY_OBSERVATION_CONTRACT_FIELD_SOURCES = {
    "policy_trail_render_mode": "trail_render_mode",
    "policy_bonus_render_mode": "bonus_render_mode",
    "policy_observation_backend": "backend",
    "policy_observation_contract_id": "contract_id",
    "policy_observation_perspective_schema_id": "perspective_schema_id",
    "policy_observation_perspective": "perspective",
    "policy_observation_seat_mapping": "seat_mapping",
    "policy_observation_stack_shape": "stack_shape",
}

_POLICY_OBSERVATION_EXPECTED_VALUES = {
    "policy_trail_render_mode": POLICY_TRAIL_RENDER_MODE,
    "policy_bonus_render_mode": POLICY_BONUS_RENDER_MODE,
    "policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
    "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
    "policy_observation_perspective_schema_id": POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
    "policy_observation_perspective": POLICY_OBSERVATION_PERSPECTIVE,
    "policy_observation_seat_mapping": POLICY_OBSERVATION_SEAT_MAPPING,
    "policy_observation_stack_shape": list(POLICY_STACK_SHAPE),
}


@dataclass(slots=True)
class LightZeroCheckpointOpponentProvider:
    """Lazy frozen-checkpoint action provider for one opponent observation row.

    The provider matches the current CurvyTron debug visual survival LightZero
    surface: MuZero conv model, stacked ``[4,64,64]`` observation, and `A=3`.
    It does not refresh snapshots or share weights with a live learner.
    """

    checkpoint_path: str | Path
    seed: int = 0
    num_simulations: int = 8
    batch_size: int = 16
    use_cuda: bool = False
    state_key: str | None = None
    illegal_action_policy: str = "raise"
    provider_id: str = LIGHTZERO_CHECKPOINT_OPPONENT_PROVIDER_ID
    provider_version: str = LIGHTZERO_CHECKPOINT_OPPONENT_PROVIDER_VERSION
    _policy: Any | None = field(default=None, init=False, repr=False)
    _device: Any | None = field(default=None, init=False, repr=False)
    _load_summary: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.illegal_action_policy not in ("raise", "first_legal"):
            raise ValueError("illegal_action_policy must be 'raise' or 'first_legal'")

    @property
    def load_summary(self) -> dict[str, Any] | None:
        """Return checkpoint load metadata after the lazy load has happened."""

        if self._load_summary is None:
            return None
        return dict(self._load_summary)

    def select_action(
        self,
        *,
        observation: np.ndarray | None,
        legal_action_mask: np.ndarray,
        decision_index: int,
        env_row: int,
        player_id: int,
        action_seed: int,
        snapshot_ref: str,
        checkpoint_ref: str | None = None,
    ) -> OpponentActionChoice:
        del decision_index, env_row, action_seed, snapshot_ref, checkpoint_ref
        if observation is None:
            raise ValueError("LightZero checkpoint provider requires an observation row")
        obs = _validate_observation_row(observation)
        legal = _validate_legal_mask(legal_action_mask)
        policy = self._loaded_policy()
        output = _policy_eval_forward(
            policy,
            observation=obs,
            legal_action_mask=legal,
            player_id=player_id,
            device=self._device,
        )
        action_id = _extract_eval_action(output)
        logp = _extract_action_logp(output, action_id)
        action_is_legal = 0 <= action_id < legal.shape[0] and bool(legal[action_id])
        if not action_is_legal:
            if self.illegal_action_policy == "raise":
                raise ValueError(
                    f"LightZero checkpoint selected illegal action {action_id!r} "
                    f"under mask {legal.astype(int).tolist()}"
                )
            action_id = int(np.flatnonzero(legal)[0])
            logp = None
        return OpponentActionChoice(action_id=int(action_id), action_logp=logp)

    def _loaded_policy(self) -> Any:
        if self._policy is not None:
            return self._policy
        policy, device, load_summary = load_lightzero_curvytron_visual_survival_policy(
            checkpoint_path=self.checkpoint_path,
            seed=self.seed,
            num_simulations=self.num_simulations,
            batch_size=self.batch_size,
            use_cuda=self.use_cuda,
            state_key=self.state_key,
        )
        self._policy = policy
        self._device = device
        self._load_summary = load_summary
        return policy


def checkpoint_policy_metadata_sidecar_path(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    return path.with_name(f"{path.name}.metadata.json")


def _load_checkpoint_policy_metadata_sidecar(checkpoint_path: Path) -> dict[str, Any]:
    sidecar_path = checkpoint_policy_metadata_sidecar_path(checkpoint_path)
    if not sidecar_path.is_file():
        return {}
    try:
        with sidecar_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ValueError(
            f"could not read LightZero checkpoint policy metadata sidecar "
            f"{sidecar_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"LightZero checkpoint policy metadata sidecar {sidecar_path} "
            "must contain a JSON object"
        )
    return payload


def _policy_observation_metadata_containers(
    payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = [payload]
    for key in ("metadata", "config"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    return containers


def _collect_policy_observation_metadata_values(
    payloads: list[Mapping[str, Any]],
) -> dict[str, list[tuple[str, Any]]]:
    values: dict[str, list[tuple[str, Any]]] = {
        key: [] for key in _POLICY_OBSERVATION_EXPECTED_VALUES
    }
    for payload_index, payload in enumerate(payloads):
        for container_index, container in enumerate(
            _policy_observation_metadata_containers(payload)
        ):
            prefix = f"payload[{payload_index}].container[{container_index}]"
            for (
                metadata_field,
                source_keys,
            ) in _POLICY_OBSERVATION_METADATA_FIELD_SOURCES.items():
                for source_key in source_keys:
                    if container.get(source_key) is not None:
                        values[metadata_field].append(
                            (f"{prefix}.{source_key}", container[source_key])
                        )
            contract = container.get("observation_contract")
            if not isinstance(contract, Mapping):
                continue
            for (
                metadata_field,
                contract_key,
            ) in _POLICY_OBSERVATION_CONTRACT_FIELD_SOURCES.items():
                if contract.get(contract_key) is not None:
                    values[metadata_field].append(
                        (
                            f"{prefix}.observation_contract.{contract_key}",
                            contract[contract_key],
                        )
                    )
    return values


def _metadata_value_matches_expected(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        try:
            return list(value) == expected
        except TypeError:
            return False
    return str(value) == str(expected)


def require_checkpoint_policy_observation_metadata(
    *,
    checkpoint_path: str | Path,
    checkpoint_payload: Any,
) -> dict[str, Any]:
    """Validate the frozen checkpoint sees the current model-facing tensor."""

    path = Path(checkpoint_path)
    payloads: list[Mapping[str, Any]] = []
    if isinstance(checkpoint_payload, Mapping):
        payloads.append(checkpoint_payload)
    sidecar = _load_checkpoint_policy_metadata_sidecar(path)
    if sidecar:
        payloads.append(sidecar)
    values = _collect_policy_observation_metadata_values(payloads)
    required_fields = (
        "policy_trail_render_mode",
        "policy_bonus_render_mode",
        "policy_observation_backend",
        "policy_observation_contract_id",
        "policy_observation_perspective_schema_id",
    )
    missing = [field for field in required_fields if not values[field]]
    if missing:
        raise ValueError(
            "LightZero checkpoint opponent is missing required policy observation "
            f"metadata: {', '.join(missing)} for {path}. Frozen checkpoint "
            "opponents must declare the tensor surface they were trained on."
        )
    for metadata_field, expected in _POLICY_OBSERVATION_EXPECTED_VALUES.items():
        for source, actual in values[metadata_field]:
            if not _metadata_value_matches_expected(actual, expected):
                raise ValueError(
                    "LightZero checkpoint opponent policy observation metadata "
                    f"{source} must be {expected!r}; got {actual!r} for {path}."
                )
    return {
        "policy_trail_render_mode": POLICY_TRAIL_RENDER_MODE,
        "policy_bonus_render_mode": POLICY_BONUS_RENDER_MODE,
        "policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
        "policy_observation_perspective_schema_id": (
            POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
        ),
    }


def snapshot_backed_lightzero_checkpoint_opponent_policy(
    *,
    checkpoint_path: str | Path,
    snapshot_ref: str,
    checkpoint_ref: str | None = None,
    model_id: str | None = None,
    seed: int = 0,
    num_simulations: int = 8,
    batch_size: int = 16,
    use_cuda: bool = False,
    state_key: str | None = None,
    illegal_action_policy: str = "raise",
) -> SnapshotBackedOpponentPolicy:
    """Build a snapshot-backed opponent policy around a frozen LightZero checkpoint."""

    provider = LightZeroCheckpointOpponentProvider(
        checkpoint_path=checkpoint_path,
        seed=seed,
        num_simulations=num_simulations,
        batch_size=batch_size,
        use_cuda=use_cuda,
        state_key=state_key,
        illegal_action_policy=illegal_action_policy,
    )
    return SnapshotBackedOpponentPolicy(
        provider=provider,
        snapshot_ref=snapshot_ref,
        checkpoint_ref=checkpoint_ref or str(checkpoint_path),
        model_id=model_id or "lightzero_muzero_curvytron_debug_visual_survival",
        seed=seed,
    )


def build_lightzero_checkpoint_multiplayer_ego_wrapper(
    env: Any,
    *,
    checkpoint_path: str | Path,
    snapshot_ref: str,
    checkpoint_ref: str | None = None,
    model_id: str | None = None,
    seed: int = 0,
    num_simulations: int = 8,
    batch_size: int = 16,
    use_cuda: bool = False,
    state_key: str | None = None,
    illegal_action_policy: str = "raise",
    ego_player_id: int | np.ndarray = 0,
    pad_to: int | None = None,
) -> Any:
    """Wrap a multiplayer env with a frozen LightZero checkpoint opponent.

    The wrapped env must already expose observations shaped ``[B,P,4,64,64]``
    and legal masks shaped ``[B,P,3]``. Checkpoint loading stays outside the env.
    """

    from curvyzero.env.multiplayer_ego_wrapper import MetadataOnlyMultiplayerEgoWrapper

    opponent_policy = snapshot_backed_lightzero_checkpoint_opponent_policy(
        checkpoint_path=checkpoint_path,
        snapshot_ref=snapshot_ref,
        checkpoint_ref=checkpoint_ref,
        model_id=model_id,
        seed=seed,
        num_simulations=num_simulations,
        batch_size=batch_size,
        use_cuda=use_cuda,
        state_key=state_key,
        illegal_action_policy=illegal_action_policy,
    )
    return MetadataOnlyMultiplayerEgoWrapper(
        env,
        ego_player_id=ego_player_id,
        opponent_policy=opponent_policy,
        pad_to=pad_to,
    )


def load_lightzero_curvytron_visual_survival_policy(
    *,
    checkpoint_path: str | Path,
    seed: int = 0,
    num_simulations: int = 8,
    batch_size: int = 16,
    use_cuda: bool = False,
    state_key: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load a LightZero MuZeroPolicy matching the current visual survival trainer."""

    try:
        import torch
        from ding.config import compile_config
        from easydict import EasyDict
        from lzero.policy.muzero import MuZeroPolicy
    except ImportError as exc:
        raise ImportError(
            "LightZero checkpoint opponents require LightZero/DI-engine runtime "
            "dependencies. Install/use the same runtime as the visual survival trainer."
        ) from exc

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"LightZero checkpoint does not exist: {path}")

    payload = _torch_load(torch, path)
    policy_observation_metadata = require_checkpoint_policy_observation_metadata(
        checkpoint_path=path,
        checkpoint_payload=payload,
    )
    found_key, state_dict = _state_dict_from_payload(payload, state_key=state_key)
    inferred_support_config = _infer_model_support_config_from_state_dict(state_dict)

    main_config, create_config = _lightzero_policy_configs(
        seed=seed,
        num_simulations=num_simulations,
        batch_size=batch_size,
        use_cuda=use_cuda,
        EasyDict=EasyDict,
    )
    _apply_inferred_model_support_config(main_config, inferred_support_config)
    cfg = compile_config(
        copy.deepcopy(main_config),
        seed=int(seed),
        auto=True,
        create_cfg=copy.deepcopy(create_config),
        save_cfg=False,
    )
    cfg.policy.cuda = bool(use_cuda)
    cfg.policy.device = "cuda" if use_cuda else "cpu"

    policy = MuZeroPolicy(cfg.policy)
    model = getattr(policy, "_model", None)
    if model is None:
        raise AttributeError("MuZeroPolicy has no _model attribute")

    load_summary = _load_state_dict_strict(model, state_dict)
    if not load_summary.get("ok"):
        raise RuntimeError(f"strict LightZero checkpoint load failed: {load_summary}")
    if hasattr(model, "eval"):
        model.eval()
    device = _policy_model_device(policy, torch=torch)
    load_summary.update(
        {
            "checkpoint_path": str(path),
            "state_key": found_key,
            "model_observation_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
            "action_space_size": 3,
            "num_simulations": int(num_simulations),
            "batch_size": int(batch_size),
            "use_cuda": bool(use_cuda),
            "device": str(device),
            "checkpoint_inferred_model_support_config": inferred_support_config,
            **policy_observation_metadata,
        }
    )
    return policy, device, load_summary


def _infer_model_support_config_from_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    def first_output_size(suffixes: tuple[str, ...]) -> int | None:
        for key, value in state_dict.items():
            key_text = str(key)
            if not any(key_text.endswith(suffix) for suffix in suffixes):
                continue
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) < 1:
                continue
            return int(shape[0])
        return None

    reward_size = first_output_size(
        (
            "dynamics_network.fc_reward_head.3.weight",
            "dynamics_network.reward_head.3.weight",
            "reward_head.3.weight",
        )
    )
    value_size = first_output_size(
        (
            "prediction_network.fc_value.3.weight",
            "prediction_network.value_head.3.weight",
            "value_head.3.weight",
        )
    )
    config: dict[str, Any] = {}
    if reward_size is not None:
        config["reward_support_size"] = reward_size
    if value_size is not None:
        config["value_support_size"] = value_size
    sizes = [size for size in (reward_size, value_size) if size is not None]
    if sizes and all(size == sizes[0] for size in sizes) and sizes[0] % 2 == 1:
        config["support_scale"] = (sizes[0] - 1) // 2
    return config


def _apply_inferred_model_support_config(
    main_config: Any,
    support_config: dict[str, Any],
) -> None:
    if not support_config:
        return
    model_config = main_config["policy"]["model"]
    for key in ("support_scale", "reward_support_size", "value_support_size"):
        if key in support_config:
            model_config[key] = int(support_config[key])


def _lightzero_policy_configs(
    *,
    seed: int,
    num_simulations: int,
    batch_size: int,
    use_cuda: bool,
    EasyDict: Any,
) -> tuple[Any, Any]:
    import importlib

    module = importlib.import_module("zoo.atari.config.atari_muzero_config")
    main_config = copy.deepcopy(module.main_config)
    main_config["exp_name"] = "/tmp/curvyzero_lightzero_checkpoint_opponent"
    main_config["policy"]["cuda"] = bool(use_cuda)
    main_config["policy"]["env_type"] = "not_board_games"
    main_config["policy"]["collector_env_num"] = 1
    main_config["policy"]["evaluator_env_num"] = 1
    main_config["policy"]["n_episode"] = 1
    main_config["policy"]["num_simulations"] = int(num_simulations)
    main_config["policy"]["batch_size"] = int(batch_size)
    main_config["policy"]["model"]["model_type"] = "conv"
    main_config["policy"]["model"]["image_channel"] = 4
    main_config["policy"]["model"]["frame_stack_num"] = 1
    main_config["policy"]["model"]["observation_shape"] = list(
        DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE
    )
    main_config["policy"]["model"]["action_space_size"] = 3
    main_config["env"]["seed"] = int(seed)
    main_config["env"]["env_id"] = "CurvyZeroStackedDebugVisualSurvivalLightZero-v0"
    main_config["env"]["observation_shape"] = list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE)
    main_config["env"]["action_space_size"] = 3
    create_config = EasyDict(
        {
            "env": {
                "type": "curvyzero_stacked_debug_visual_survival_lightzero",
                "import_names": [
                    "curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env"
                ],
            },
            "env_manager": {"type": "base"},
            "policy": {"type": "muzero", "import_names": ["lzero.policy.muzero"]},
        }
    )
    return main_config, create_config


def _policy_eval_forward(
    policy: Any,
    *,
    observation: np.ndarray,
    legal_action_mask: np.ndarray,
    player_id: int,
    device: Any,
) -> Any:
    del player_id
    import torch

    obs_tensor = torch.as_tensor(
        np.asarray([observation], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    action_mask = np.asarray([legal_action_mask], dtype=np.float32)
    with torch.no_grad():
        return policy.eval_mode.forward(
            obs_tensor,
            action_mask=action_mask,
            to_play=[-1],
            ready_env_id=np.asarray([0]),
        )


def _extract_eval_action(output: Any) -> int:
    root = _first_policy_output(output)
    if isinstance(root, dict) and "action" in root:
        return int(_plain_scalar(root["action"]))
    raise ValueError(f"could not extract LightZero action from output: {output!r}")


def _extract_action_logp(output: Any, action_id: int) -> float | None:
    root = _first_policy_output(output)
    if not isinstance(root, dict):
        return None
    distribution = None
    for key in ("visit_count_distribution", "visit_count_distributions"):
        if key in root:
            distribution = root[key]
            break
    if distribution is None:
        return None
    probs = np.asarray(distribution, dtype=np.float64).reshape(-1)
    if action_id < 0 or action_id >= probs.shape[0]:
        return None
    prob = float(probs[action_id])
    if prob <= 0.0 or not math.isfinite(prob):
        return None
    return float(math.log(prob))


def _first_policy_output(output: Any) -> Any:
    if isinstance(output, dict):
        for key in (0, "0"):
            if key in output:
                return _first_policy_output(output[key])
        return output
    if isinstance(output, (list, tuple)) and output:
        return _first_policy_output(output[0])
    return output


def _validate_observation_row(observation: np.ndarray) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE:
        raise ValueError(
            f"LightZero checkpoint opponent observation shape {obs.shape!r}; "
            f"expected {DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE!r}"
        )
    return obs


def _validate_legal_mask(legal_action_mask: np.ndarray) -> np.ndarray:
    legal = np.asarray(legal_action_mask, dtype=bool)
    if legal.shape != (3,):
        raise ValueError(f"legal_action_mask shape {legal.shape!r}; expected (3,)")
    if not bool(legal.any()):
        raise ValueError("legal_action_mask must include at least one legal action")
    return legal


def _torch_load(torch: Any, path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_dict_from_payload(
    payload: Any,
    *,
    state_key: str | None,
) -> tuple[str, dict[str, Any]]:
    if state_key is not None:
        if not isinstance(payload, dict) or state_key not in payload:
            raise KeyError(f"checkpoint payload does not contain state_key={state_key!r}")
        value = payload[state_key]
        if not isinstance(value, dict):
            raise TypeError(f"checkpoint state_key={state_key!r} is not a state dict")
        return state_key, value

    candidates: list[tuple[str, Any]] = [("<root>", payload)]
    if isinstance(payload, dict):
        for key in (
            "model",
            "state_dict",
            "model_state_dict",
            "network",
            "target_model",
            "policy",
            "_model",
            "_learn_model",
        ):
            if key in payload:
                candidates.append((key, payload[key]))
        for key, value in payload.items():
            if isinstance(value, dict):
                for nested_key in ("model", "state_dict", "model_state_dict", "_model", "_learn_model"):
                    if nested_key in value:
                        candidates.append((f"{key}.{nested_key}", value[nested_key]))

    best: tuple[str, dict[str, Any], int] | None = None
    for key, value in candidates:
        if not isinstance(value, dict):
            continue
        tensor_count = sum(1 for item in value.values() if _is_tensor_like(item))
        if tensor_count == 0:
            continue
        keys = [str(item) for item in value]
        score = tensor_count
        if any("representation_network" in item for item in keys):
            score += 1000
        if any("prediction_network" in item for item in keys):
            score += 1000
        if any("dynamics_network" in item for item in keys):
            score += 1000
        if best is None or score > best[2]:
            best = (key, value, score)
    if best is None:
        raise ValueError("checkpoint payload does not contain a tensor state dict")
    return best[0], best[1]


def _load_state_dict_strict(module: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        ("as_is", state_dict),
        ("strip_module", _strip_prefix(state_dict, "module.")),
        ("strip_model", _strip_prefix(state_dict, "model.")),
        ("strip_learn_model", _strip_prefix(state_dict, "_learn_model.")),
    ]
    errors = []
    for name, candidate in candidates:
        try:
            loaded = module.load_state_dict(candidate, strict=True)
            return {
                "ok": True,
                "candidate": name,
                "strict": True,
                "missing_keys": list(getattr(loaded, "missing_keys", [])),
                "unexpected_keys": list(getattr(loaded, "unexpected_keys", [])),
            }
        except Exception as exc:
            errors.append({"candidate": name, "error": str(exc)})
    return {"ok": False, "strict": True, "errors": errors[-8:]}


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not any(str(key).startswith(prefix) for key in state_dict):
        return state_dict
    return {str(key).removeprefix(prefix): value for key, value in state_dict.items()}


def _policy_model_device(policy: Any, *, torch: Any) -> Any:
    model = getattr(policy, "_model", None)
    if model is None:
        return torch.device("cpu")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _plain_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


__all__ = [
    "LIGHTZERO_CHECKPOINT_OPPONENT_PROVIDER_ID",
    "LIGHTZERO_CHECKPOINT_OPPONENT_PROVIDER_VERSION",
    "LightZeroCheckpointOpponentProvider",
    "build_lightzero_checkpoint_multiplayer_ego_wrapper",
    "checkpoint_policy_metadata_sidecar_path",
    "load_lightzero_curvytron_visual_survival_policy",
    "require_checkpoint_policy_observation_metadata",
    "snapshot_backed_lightzero_checkpoint_opponent_policy",
]
