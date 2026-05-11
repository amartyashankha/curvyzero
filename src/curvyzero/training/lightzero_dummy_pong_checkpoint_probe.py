"""Inspect a LightZero dummy Pong checkpoint and try one tabular action.

This is an implementation-prep probe, not the independent scoreboard. It is
intended to run inside the same LightZero runtime used by the tiny trainer:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke ...

or any environment with torch, LightZero, DI-engine, and curvyzero on
PYTHONPATH. Local runs without those packages can still read this file but
cannot load a ``.pth.tar`` checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROBE_SCHEMA_ID = "curvyzero_lightzero_dummy_pong_checkpoint_probe/v0"
DEFAULT_TABULAR_OBSERVATION = (
    0.5,
    0.5,
    0.0,
    1.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.5,
    0.0,
)
ACTION_LABELS = ("up", "stay", "down")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--feature-mode", default="tabular_ego", choices=("tabular_ego",))
    parser.add_argument("--env", default="dummy_pong_lag1")
    parser.add_argument("--opponent-policy", default="random_uniform")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-env-step", type=int, default=64)
    parser.add_argument(
        "--observation",
        default=",".join(str(value) for value in DEFAULT_TABULAR_OBSERVATION),
        help="Comma-separated 10-float tabular_ego row.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    result = probe_lightzero_dummy_pong_checkpoint(
        checkpoint_path=args.checkpoint,
        feature_mode=args.feature_mode,
        env=args.env,
        opponent_policy=args.opponent_policy,
        seed=args.seed,
        max_env_step=args.max_env_step,
        observation=_parse_observation(args.observation),
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text, end="")


def probe_lightzero_dummy_pong_checkpoint(
    *,
    checkpoint_path: Path,
    feature_mode: str,
    env: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    observation: list[float],
) -> dict[str, Any]:
    if feature_mode != "tabular_ego":
        raise ValueError("this first probe only supports feature_mode='tabular_ego'")
    if len(observation) != 10:
        raise ValueError("tabular_ego observation must contain exactly 10 floats")

    result: dict[str, Any] = {
        "schema": PROBE_SCHEMA_ID,
        "checkpoint": _checkpoint_summary(checkpoint_path),
        "feature_mode": feature_mode,
        "env": env,
        "opponent_policy": opponent_policy,
        "seed": seed,
        "max_env_step": max_env_step,
        "observation_shape": [10],
        "action_labels": list(ACTION_LABELS),
        "load": {"ok": False},
        "state_dict": {"ok": False},
        "action_probe": {"ok": False},
        "implementation_boundary": {
            "model_api": "lzero.model.muzero_model_mlp.MuZeroModelMLP",
            "inference_api": "model.initial_inference(torch.tensor(obs)[None, :])",
            "action_selection": "argmax over output.policy_logits with action_mask=[1,1,1]",
            "policy_api_not_used_yet": (
                "lzero.policy.muzero.MuZeroPolicy.eval_mode.forward would run MCTS, "
                "but this probe first validates direct network reconstruction."
            ),
        },
    }

    try:
        checkpoint = _torch_load(checkpoint_path)
    except Exception as exc:  # pragma: no cover - depends on optional torch/runtime.
        result["load"] = _exception("torch_load_failed", exc)
        return result

    result["load"] = {
        "ok": True,
        "payload_summary": _summarize_value(checkpoint),
    }
    state_candidate = _find_state_dict(checkpoint)
    if state_candidate is None:
        result["state_dict"] = {
            "ok": False,
            "reason": "no tensor state dict found under common LightZero checkpoint keys",
        }
        return result

    state_path, state_dict = state_candidate
    result["state_dict"] = {
        "ok": True,
        "path": state_path,
        "tensor_count": len(state_dict),
        "keys_sample": list(state_dict)[:30],
        "tensor_sample": {
            key: _summarize_value(value)
            for key, value in list(state_dict.items())[:12]
        },
    }

    try:
        action_probe = _try_direct_model_action(
            state_dict=state_dict,
            env=env,
            feature_mode=feature_mode,
            opponent_policy=opponent_policy,
            seed=seed,
            max_env_step=max_env_step,
            observation=observation,
        )
    except Exception as exc:  # pragma: no cover - depends on optional LightZero runtime.
        result["action_probe"] = _exception("direct_model_action_failed", exc)
        return result

    result["action_probe"] = action_probe
    return result


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _try_direct_model_action(
    *,
    state_dict: dict[str, Any],
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    observation: list[float],
) -> dict[str, Any]:
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
    load_result = _load_model_state_dict(model, state_dict)
    model.eval()
    obs_tensor = torch.tensor([observation], dtype=torch.float32)
    with torch.no_grad():
        network_output = model.initial_inference(obs_tensor)
    policy_logits = network_output.policy_logits.detach().cpu().reshape(-1).tolist()
    action_id = int(max(range(len(policy_logits)), key=policy_logits.__getitem__))
    return {
        "ok": True,
        "model_class": "lzero.model.muzero_model_mlp.MuZeroModelMLP",
        "model_config": _to_plain(model_cfg),
        "load_state_dict": load_result,
        "policy_logits": [float(value) for value in policy_logits],
        "action_id": action_id,
        "action_label": ACTION_LABELS[action_id] if action_id < len(ACTION_LABELS) else None,
    }


def _load_model_state_dict(model: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        ("as_is", state_dict),
        ("strip_module", _strip_prefix(state_dict, "module.")),
        ("strip_model", _strip_prefix(state_dict, "model.")),
        ("strip_learn_model", _strip_prefix(state_dict, "_learn_model.")),
    ]
    errors = []
    for name, candidate in candidates:
        try:
            loaded = model.load_state_dict(candidate, strict=True)
            return {
                "ok": True,
                "candidate": name,
                "strict": True,
                "missing_keys": list(getattr(loaded, "missing_keys", [])),
                "unexpected_keys": list(getattr(loaded, "unexpected_keys", [])),
            }
        except Exception as exc:
            errors.append({"candidate": name, "strict": True, "error": str(exc)})

    for name, candidate in candidates:
        try:
            loaded = model.load_state_dict(candidate, strict=False)
            return {
                "ok": True,
                "candidate": name,
                "strict": False,
                "missing_keys": list(getattr(loaded, "missing_keys", []))[:40],
                "unexpected_keys": list(getattr(loaded, "unexpected_keys", []))[:40],
                "note": "non-strict load is only a probe signal; do not score from it",
            }
        except Exception as exc:
            errors.append({"candidate": name, "strict": False, "error": str(exc)})

    raise RuntimeError(f"could not load state dict into MuZeroModelMLP: {errors}")


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


def _find_state_dict(payload: Any) -> tuple[str, dict[str, Any]] | None:
    candidates: list[tuple[str, Any]] = [("<root>", payload)]
    if isinstance(payload, dict):
        for key in (
            "model",
            "state_dict",
            "model_state_dict",
            "network",
            "target_model",
            "policy",
        ):
            if key in payload:
                candidates.append((key, payload[key]))
        for key, value in payload.items():
            if isinstance(value, dict):
                for nested_key in ("model", "state_dict", "model_state_dict"):
                    if nested_key in value:
                        candidates.append((f"{key}.{nested_key}", value[nested_key]))

    best: tuple[str, dict[str, Any], int] | None = None
    for path, value in candidates:
        if not isinstance(value, dict):
            continue
        tensor_count = sum(1 for item in value.values() if _is_tensor_like(item))
        if tensor_count == 0:
            continue
        score = tensor_count
        if any("representation_network" in str(key) for key in value):
            score += 1000
        if any("prediction_network" in str(key) for key in value):
            score += 1000
        if best is None or score > best[2]:
            best = (path, value, score)
    if best is None:
        return None
    return best[0], best[1]


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        summary["bytes"] = path.stat().st_size
        summary["sha256"] = _sha256(path)
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_value(value: Any, *, depth: int = 0) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        summary["shape"] = [int(item) for item in shape]
    if dtype is not None:
        summary["dtype"] = str(dtype)
    if hasattr(value, "numel"):
        try:
            summary["numel"] = int(value.numel())
        except Exception:
            pass
    if isinstance(value, dict):
        keys = [str(key) for key in value.keys()]
        summary["len"] = len(value)
        summary["keys_sample"] = keys[:40]
        if depth < 1:
            summary["items_sample"] = {
                str(key): _summarize_value(item, depth=depth + 1)
                for key, item in list(value.items())[:8]
            }
    elif isinstance(value, (list, tuple)):
        summary["len"] = len(value)
        if depth < 1:
            summary["items_sample"] = [
                _summarize_value(item, depth=depth + 1)
                for item in list(value)[:8]
            ]
    else:
        text = repr(_to_plain(value))
        summary["repr"] = text if len(text) <= 240 else text[:237] + "..."
    return summary


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not any(str(key).startswith(prefix) for key in state_dict):
        return state_dict
    return {
        str(key).removeprefix(prefix): value
        for key, value in state_dict.items()
    }


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _parse_observation(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _exception(stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
