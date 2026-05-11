"""LightZero checkpoint policies for the project-owned dummy Pong eval."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.training.dummy_pong import ACTION_LABELS
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongObservation
from curvyzero.training.lightzero_dummy_pong_features import encode_lightzero_observation
from curvyzero.training.lightzero_dummy_pong_features import lightzero_feature_schema_id

LIGHTZERO_POLICY_PREFIX = "lightzero:"
LIGHTZERO_POLICY_HEAD_GREEDY_SCHEMA_ID = (
    "curvyzero_lightzero_dummy_pong_policy_head_greedy_no_mcts/v0"
)
LIGHTZERO_POLICY_HEAD_GREEDY_LABEL = "LightZero policy-head greedy, no MCTS"
LIGHTZERO_MCTS_EVAL_MODE_SCHEMA_ID = "curvyzero_lightzero_dummy_pong_mcts_eval_mode/v0"
LIGHTZERO_MCTS_EVAL_MODE_LABEL = "LightZero MCTS eval-mode"


@dataclass(frozen=True, slots=True)
class LightZeroPolicyHeadGreedySpec:
    policy_id: str
    checkpoint_path: Path
    policy: Any
    checkpoint_metadata: dict[str, object]
    checkpoint_schema_id: str
    feature_schema_id: str
    feature_mode: str
    adapter_schema_id: str
    adapter_label: str


@dataclass(frozen=True, slots=True)
class LightZeroMCTSEvalModeSpec:
    policy_id: str
    checkpoint_path: Path
    policy: Any
    checkpoint_metadata: dict[str, object]
    checkpoint_schema_id: str
    feature_schema_id: str
    feature_mode: str
    adapter_schema_id: str
    adapter_label: str
    num_simulations: int


class LightZeroPolicyHeadGreedyPolicy:
    """Greedy argmax over ``MuZeroModelMLP.initial_inference().policy_logits``."""

    def __init__(
        self,
        *,
        policy_id: str,
        model: Any,
        torch_module: Any,
        encoder_config: PongConfig,
        feature_mode: str,
    ) -> None:
        self.name = policy_id
        self._model = model
        self._torch = torch_module
        self._encoder_config = encoder_config
        self._feature_mode = feature_mode

    def reset(self, episode_seed: int, agent: str) -> None:
        del episode_seed, agent

    def action(
        self,
        observation: PongObservation,
        raster_grid: Any,
        agent: str,
    ) -> int:
        del agent
        encoded = encode_lightzero_observation(
            feature_mode=self._feature_mode,
            observation=observation,
            config=self._encoder_config,
            raster_grid=raster_grid,
        )
        obs_tensor = self._torch.as_tensor(encoded[None, :], dtype=self._torch.float32)
        with self._torch.no_grad():
            network_output = self._model.initial_inference(obs_tensor)
        policy_logits = network_output.policy_logits.detach().cpu().reshape(-1)
        action_id = int(self._torch.argmax(policy_logits).item())
        if action_id < 0 or action_id >= len(ACTION_LABELS):
            raise ValueError(f"LightZero policy produced invalid action {action_id!r}")
        return action_id


class LightZeroMCTSEvalModePolicy:
    """Action adapter around ``MuZeroPolicy.eval_mode.forward`` with MCTS."""

    def __init__(
        self,
        *,
        policy_id: str,
        policy: Any,
        torch_module: Any,
        encoder_config: PongConfig,
        feature_mode: str,
    ) -> None:
        self.name = policy_id
        self._policy = policy
        self._torch = torch_module
        self._encoder_config = encoder_config
        self._feature_mode = feature_mode
        self.last_forward_output: Any | None = None

    def reset(self, episode_seed: int, agent: str) -> None:
        del episode_seed, agent
        self.last_forward_output = None

    def action(
        self,
        observation: PongObservation,
        raster_grid: Any,
        agent: str,
    ) -> int:
        del agent
        encoded = encode_lightzero_observation(
            feature_mode=self._feature_mode,
            observation=observation,
            config=self._encoder_config,
            raster_grid=raster_grid,
        )
        obs_tensor = self._torch.as_tensor(encoded[None, :], dtype=self._torch.float32)
        action_mask = np.ones((1, len(ACTION_LABELS)), dtype=np.float32)
        with self._torch.no_grad():
            output = self._policy.eval_mode.forward(
                obs_tensor,
                action_mask=action_mask,
                to_play=[-1],
                ready_env_id=np.asarray([0]),
            )
        self.last_forward_output = output
        action_id = _extract_eval_mode_action(output)
        if action_id < 0 or action_id >= len(ACTION_LABELS):
            raise ValueError(f"LightZero MCTS eval-mode produced invalid action {action_id!r}")
        return action_id


def load_lightzero_policy_head_greedy_checkpoint(
    *,
    policy_id: str,
    checkpoint_path: Path,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
) -> LightZeroPolicyHeadGreedySpec:
    feature_schema_id = lightzero_feature_schema_id(feature_mode)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LightZero checkpoint file not found: {checkpoint_path}")

    import torch
    from lzero.model.muzero_model_mlp import MuZeroModelMLP

    from curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_COLLECTOR_ENV_NUM,
        DEFAULT_EVALUATOR_ENV_NUM,
        DEFAULT_NUM_SIMULATIONS,
        DEFAULT_UPDATE_PER_COLLECT,
        patched_dummy_pong_configs,
    )
    from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import _find_state_dict
    from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import _torch_load
    from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import _to_plain

    checkpoint = _torch_load(checkpoint_path)
    state_candidate = _find_state_dict(checkpoint)
    if state_candidate is None:
        raise ValueError(
            "no tensor state dict found under common LightZero checkpoint keys "
            f"in {checkpoint_path}"
        )
    state_path, state_dict = state_candidate

    patched = patched_dummy_pong_configs(
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        collector_env_num=DEFAULT_COLLECTOR_ENV_NUM,
        evaluator_env_num=DEFAULT_EVALUATOR_ENV_NUM,
        n_evaluator_episode=1,
        num_simulations=DEFAULT_NUM_SIMULATIONS,
        batch_size=DEFAULT_BATCH_SIZE,
        update_per_collect=DEFAULT_UPDATE_PER_COLLECT,
    )
    model_cfg = dict(patched["main_config"]["policy"]["model"])
    model_type = str(model_cfg.pop("model_type", "mlp"))
    if model_type != "mlp":
        raise ValueError(f"expected LightZero model_type='mlp', got {model_type!r}")
    model_cfg = _model_cfg_for_checkpoint_state_dict(model_cfg, state_dict)

    model = MuZeroModelMLP(**model_cfg)
    load_state_dict = load_state_dict_policy_head_safe(model, state_dict)
    model.eval()

    policy = LightZeroPolicyHeadGreedyPolicy(
        policy_id=policy_id,
        model=model,
        torch_module=torch,
        encoder_config=PongConfig(max_steps=max_env_step),
        feature_mode=feature_mode,
    )
    checkpoint_metadata: dict[str, object] = {
        "adapter_label": LIGHTZERO_POLICY_HEAD_GREEDY_LABEL,
        "adapter_schema_id": LIGHTZERO_POLICY_HEAD_GREEDY_SCHEMA_ID,
        "checkpoint_summary": _checkpoint_summary(checkpoint_path),
        "state_dict_path": state_path,
        "state_dict_tensor_count": sum(1 for value in state_dict.values() if hasattr(value, "shape")),
        "load_state_dict": load_state_dict,
        "strict_full_model_load_ok": bool(load_state_dict["strict"]),
        "direct_policy_head_possible": True,
        "model_class": "lzero.model.muzero_model_mlp.MuZeroModelMLP",
        "model_config": _to_plain(model_cfg),
        "config_surface": _to_plain(patched["patched_surface"]),
        "inference_boundary": (
            f"model.initial_inference({feature_mode}[None, :]); "
            "action=argmax(policy_logits); no MuZeroPolicy, no MCTS"
        ),
        "mcts_parity_claim": False,
    }
    return LightZeroPolicyHeadGreedySpec(
        policy_id=policy_id,
        checkpoint_path=checkpoint_path,
        policy=policy,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_schema_id="lightzero_pth_tar",
        feature_schema_id=feature_schema_id,
        feature_mode=feature_mode,
        adapter_schema_id=LIGHTZERO_POLICY_HEAD_GREEDY_SCHEMA_ID,
        adapter_label=LIGHTZERO_POLICY_HEAD_GREEDY_LABEL,
    )


def load_lightzero_mcts_eval_mode_checkpoint(
    *,
    policy_id: str,
    checkpoint_path: Path,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    num_simulations: int,
) -> LightZeroMCTSEvalModeSpec:
    feature_schema_id = lightzero_feature_schema_id(feature_mode)
    if num_simulations < 1:
        raise ValueError("num_simulations must be at least 1")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LightZero checkpoint file not found: {checkpoint_path}")

    import torch
    from ding.config import compile_config
    from lzero.policy.muzero import MuZeroPolicy

    from curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_COLLECTOR_ENV_NUM,
        DEFAULT_EVALUATOR_ENV_NUM,
        DEFAULT_UPDATE_PER_COLLECT,
        patched_dummy_pong_configs,
    )
    from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import _find_state_dict
    from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import _torch_load
    from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import _to_plain

    checkpoint = _torch_load(checkpoint_path)
    state_candidate = _find_state_dict(checkpoint)
    if state_candidate is None:
        raise ValueError(
            "no tensor state dict found under common LightZero checkpoint keys "
            f"in {checkpoint_path}"
        )
    state_path, state_dict = state_candidate

    patched = patched_dummy_pong_configs(
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        collector_env_num=DEFAULT_COLLECTOR_ENV_NUM,
        evaluator_env_num=DEFAULT_EVALUATOR_ENV_NUM,
        n_evaluator_episode=1,
        num_simulations=num_simulations,
        batch_size=DEFAULT_BATCH_SIZE,
        update_per_collect=DEFAULT_UPDATE_PER_COLLECT,
    )
    main_config = copy.deepcopy(patched["main_config"])
    model_cfg = dict(main_config["policy"]["model"])
    model_type = str(model_cfg.get("model_type", "mlp"))
    if model_type != "mlp":
        raise ValueError(f"expected LightZero model_type='mlp', got {model_type!r}")
    model_cfg_no_type = dict(model_cfg)
    model_cfg_no_type.pop("model_type", None)
    strict_model_cfg = _model_cfg_for_checkpoint_state_dict(model_cfg_no_type, state_dict)
    if strict_model_cfg.get("res_connection_in_dynamics") is True:
        main_config["policy"]["model"]["res_connection_in_dynamics"] = True

    cfg = compile_config(
        main_config,
        seed=seed,
        auto=True,
        create_cfg=copy.deepcopy(patched["create_config"]),
        save_cfg=False,
    )
    if not hasattr(cfg.policy, "device"):
        cfg.policy.device = "cpu"
    policy = MuZeroPolicy(cfg.policy)
    policy_model = getattr(policy, "_model", None)
    if policy_model is None:
        raise AttributeError("MuZeroPolicy has no _model attribute for checkpoint load")
    load_state_dict = load_state_dict_strict_full_model(policy_model, state_dict)
    policy_model.eval()

    adapter = LightZeroMCTSEvalModePolicy(
        policy_id=policy_id,
        policy=policy,
        torch_module=torch,
        encoder_config=PongConfig(max_steps=max_env_step),
        feature_mode=feature_mode,
    )
    checkpoint_metadata: dict[str, object] = {
        "adapter_label": LIGHTZERO_MCTS_EVAL_MODE_LABEL,
        "adapter_schema_id": LIGHTZERO_MCTS_EVAL_MODE_SCHEMA_ID,
        "checkpoint_summary": _checkpoint_summary(checkpoint_path),
        "state_dict_path": state_path,
        "state_dict_tensor_count": sum(1 for value in state_dict.values() if hasattr(value, "shape")),
        "load_state_dict": load_state_dict,
        "strict_full_model_load_ok": True,
        "strict_full_model_load_variant": (
            "res_connection_in_dynamics_true"
            if strict_model_cfg.get("res_connection_in_dynamics") is True
            else "as_is"
        ),
        "model_class": "lzero.policy.muzero.MuZeroPolicy",
        "model_config": _to_plain(strict_model_cfg),
        "config_surface": _to_plain(patched["patched_surface"]),
        "num_simulations": num_simulations,
        "inference_boundary": (
            f"MuZeroPolicy.eval_mode.forward({feature_mode}[None, :], "
            "action_mask=ones[1,3], to_play=[-1], ready_env_id=[0]); uses MCTS"
        ),
        "mcts_parity_claim": True,
    }
    return LightZeroMCTSEvalModeSpec(
        policy_id=policy_id,
        checkpoint_path=checkpoint_path,
        policy=adapter,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_schema_id="lightzero_pth_tar",
        feature_schema_id=feature_schema_id,
        feature_mode=feature_mode,
        adapter_schema_id=LIGHTZERO_MCTS_EVAL_MODE_SCHEMA_ID,
        adapter_label=LIGHTZERO_MCTS_EVAL_MODE_LABEL,
        num_simulations=num_simulations,
    )


def load_state_dict_policy_head_safe(model: Any, state_dict: dict[str, Any]) -> dict[str, object]:
    """Load a LightZero checkpoint for policy-head eval.

    A strict full-model load is preferred. If the only incompatibility is under
    ``dynamics_network.*``, allow ``strict=False`` because this eval never calls
    recurrent dynamics or MCTS.
    """

    strict_errors = []
    for candidate_name, candidate in _state_dict_candidates(state_dict):
        try:
            loaded = model.load_state_dict(candidate, strict=True)
            return {
                "ok": True,
                "candidate": candidate_name,
                "strict": True,
                "missing_keys": list(getattr(loaded, "missing_keys", [])),
                "unexpected_keys": list(getattr(loaded, "unexpected_keys", [])),
            }
        except Exception as exc:
            strict_errors.append({"candidate": candidate_name, "error": str(exc)})

    relaxed_errors = []
    for candidate_name, candidate in _state_dict_candidates(state_dict):
        try:
            loaded = model.load_state_dict(candidate, strict=False)
        except Exception as exc:
            relaxed_errors.append({"candidate": candidate_name, "error": str(exc)})
            continue

        missing_keys = list(getattr(loaded, "missing_keys", []))
        unexpected_keys = list(getattr(loaded, "unexpected_keys", []))
        mismatch_keys = missing_keys + unexpected_keys
        if mismatch_keys and _only_dynamics_network_keys(mismatch_keys):
            return {
                "ok": True,
                "candidate": candidate_name,
                "strict": False,
                "relaxed_reason": "dynamics_network_only_mismatch_policy_head_eval",
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "strict_errors": strict_errors,
            }
        relaxed_errors.append(
            {
                "candidate": candidate_name,
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "reason": "non-dynamics mismatch is not allowed",
            }
        )

    raise RuntimeError(
        "could not load LightZero state dict for policy-head eval; "
        f"strict_errors={strict_errors}; relaxed_errors={relaxed_errors}"
    )


def load_state_dict_strict_full_model(model: Any, state_dict: dict[str, Any]) -> dict[str, object]:
    errors = []
    for candidate_name, candidate in _state_dict_candidates(state_dict):
        try:
            loaded = model.load_state_dict(candidate, strict=True)
            return {
                "ok": True,
                "candidate": candidate_name,
                "strict": True,
                "missing_keys": list(getattr(loaded, "missing_keys", [])),
                "unexpected_keys": list(getattr(loaded, "unexpected_keys", [])),
            }
        except Exception as exc:
            errors.append({"candidate": candidate_name, "error": str(exc)})
    raise RuntimeError(f"could not strict-load LightZero full model; errors={errors}")


def _extract_eval_mode_action(output: Any) -> int:
    if isinstance(output, dict):
        for key in (0, "0"):
            if key in output:
                return _extract_eval_mode_action(output[key])
        if "action" in output:
            return int(np.asarray(output["action"]).item())
        for key in ("actions", "selected_action", "selected_actions"):
            if key in output:
                return int(np.asarray(output[key]).reshape(-1)[0].item())
    if isinstance(output, (list, tuple)):
        if len(output) == 1:
            return _extract_eval_mode_action(output[0])
        if output and isinstance(output[0], dict):
            return _extract_eval_mode_action(output[0])
    if hasattr(output, "action"):
        return int(np.asarray(getattr(output, "action")).item())
    raise ValueError(f"could not extract action from MuZeroPolicy.eval_mode.forward output: {output!r}")


def _model_cfg_for_checkpoint_state_dict(
    model_cfg: dict[str, Any],
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    if _has_split_residual_dynamics_keys(state_dict):
        return {**model_cfg, "res_connection_in_dynamics": True}
    return model_cfg


def _has_split_residual_dynamics_keys(state_dict: dict[str, Any]) -> bool:
    keys = [str(key) for key in state_dict]
    return any(key.startswith("dynamics_network.fc_dynamics_1.") for key in keys) and any(
        key.startswith("dynamics_network.fc_dynamics_2.") for key in keys
    )


def _only_dynamics_network_keys(keys: list[str]) -> bool:
    return bool(keys) and all(str(key).startswith("dynamics_network.") for key in keys)


def _state_dict_candidates(state_dict: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("as_is", state_dict),
        ("strip_module", _strip_prefix(state_dict, "module.")),
        ("strip_model", _strip_prefix(state_dict, "model.")),
        ("strip_learn_model", _strip_prefix(state_dict, "_learn_model.")),
    ]


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not any(str(key).startswith(prefix) for key in state_dict):
        return state_dict
    return {
        str(key).removeprefix(prefix): value
        for key, value in state_dict.items()
    }


def _checkpoint_summary(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
