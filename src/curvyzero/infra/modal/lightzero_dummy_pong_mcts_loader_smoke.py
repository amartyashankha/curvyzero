"""Modal strict-load and eval-path audit for LightZero dummy Pong checkpoints.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_mcts_loader_smoke

This smoke is intentionally conservative. It may try a key rename only after
proving that the installed LightZero model has the same tensor shapes behind
the old and new names. It does not call that a valid MCTS path unless strict
load and a LightZero eval forward both succeed.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTOR_ENV_NUM,
    DEFAULT_ENV,
    DEFAULT_EVALUATOR_ENV_NUM,
    DEFAULT_FEATURE_MODE,
    DEFAULT_MAX_ENV_STEP,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_OPPONENT_POLICY,
    DEFAULT_SEED,
    DEFAULT_UPDATE_PER_COLLECT,
    LIGHTZERO_VERSION,
    patched_dummy_pong_configs,
)
from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import (
    DEFAULT_TABULAR_OBSERVATION,
)


APP_NAME = "curvyzero-lightzero-dummy-pong-mcts-loader-smoke"
TASK_ID = "lightzero-dummy-pong"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_RUN_ID = "lz-dpong-20260509T141607Z-3696aa333028"
DEFAULT_ATTEMPT_ID = "attempt-20260509T141607Z-98662e4917b4"
DEFAULT_CHECKPOINT_REF = (
    "training/lightzero-dummy-pong/"
    "lz-dpong-20260509T141607Z-3696aa333028/"
    "checkpoints/lightzero/ckpt_best.pth.tar"
)
DIAGNOSTIC_OBSERVATIONS = {
    "default_center": DEFAULT_TABULAR_OBSERVATION,
    "ball_above_left": (0.18, 0.22, -0.7, -0.6, 0.0, 0.0, -0.6, -0.7, 0.72, 0.45),
    "ball_below_right": (0.82, 0.78, 0.65, 0.5, 0.0, 0.0, 0.55, 0.65, 0.28, -0.45),
    "near_paddle_center": (0.12, 0.5, -0.95, 0.0, 0.0, 0.0, -0.9, 0.0, 0.5, 0.0),
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}", "numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


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


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-16:],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_dict_from_checkpoint(payload: Any) -> tuple[str, dict[str, Any]]:
    from curvyzero.training.lightzero_dummy_pong_checkpoint_probe import _find_state_dict

    candidate = _find_state_dict(payload)
    if candidate is None:
        raise ValueError("no tensor state dict found in checkpoint")
    return candidate


def _summarize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, int] = {}
    shapes: dict[str, list[int]] = {}
    for key, value in state_dict.items():
        prefix = str(key).split(".", 1)[0]
        groups[prefix] = groups.get(prefix, 0) + 1
        shape = getattr(value, "shape", None)
        if shape is not None and len(shapes) < 60:
            shapes[str(key)] = [int(item) for item in shape]
    return {
        "tensor_count": len(state_dict),
        "prefix_counts": groups,
        "keys_sample": list(state_dict)[:80],
        "shape_sample": shapes,
        "policy_head_tensor_stats": _tensor_stats_for_keys(
            state_dict,
            ("policy_head",),
        ),
    }


def _tensor_stats_for_keys(
    state_dict: dict[str, Any],
    key_terms: tuple[str, ...],
) -> dict[str, Any]:
    import torch

    stats: dict[str, Any] = {}
    for key, value in state_dict.items():
        key_text = str(key)
        if not all(term in key_text.lower() for term in key_terms):
            continue
        if not hasattr(value, "detach"):
            continue
        tensor = value.detach().cpu().float().reshape(-1)
        if tensor.numel() == 0:
            continue
        stats[key_text] = {
            "shape": [int(item) for item in value.shape],
            "numel": int(tensor.numel()),
            "min": float(torch.min(tensor).item()),
            "max": float(torch.max(tensor).item()),
            "mean": float(torch.mean(tensor).item()),
            "std": float(torch.std(tensor, unbiased=False).item()),
            "l2": float(torch.linalg.vector_norm(tensor).item()),
            "nonzero": int(torch.count_nonzero(tensor).item()),
        }
    return stats


def _model_config(
    *,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    num_simulations: int,
) -> dict[str, Any]:
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
    model_cfg = dict(patched["main_config"]["policy"]["model"])
    model_type = str(model_cfg.pop("model_type", "mlp"))
    if model_type != "mlp":
        raise ValueError(f"expected model_type='mlp', got {model_type!r}")
    return {
        "main_config": patched["main_config"],
        "create_config": patched["create_config"],
        "model_cfg": model_cfg,
        "surface": patched["patched_surface"],
    }


def _source_excerpt(obj: Any, *, needle: str | None = None, radius: int = 24) -> dict[str, Any]:
    try:
        source = inspect.getsource(obj)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    lines = source.splitlines()
    if needle is None:
        return {"ok": True, "line_count": len(lines), "head": lines[:80]}
    matches = [idx for idx, line in enumerate(lines) if needle in line]
    if not matches:
        return {"ok": True, "line_count": len(lines), "matches": [], "head": lines[:80]}
    start = max(0, matches[0] - radius)
    end = min(len(lines), matches[0] + radius)
    return {
        "ok": True,
        "line_count": len(lines),
        "first_match_line": matches[0] + 1,
        "excerpt": lines[start:end],
    }


def _inspect_lightzero_surface(model_cfg: dict[str, Any]) -> dict[str, Any]:
    import lzero
    import torch
    from importlib import metadata
    from lzero.model.muzero_model_mlp import MuZeroModelMLP
    from lzero.policy.muzero import MuZeroPolicy

    model = MuZeroModelMLP(**model_cfg)
    state_keys = list(model.state_dict())
    model_module = importlib.import_module("lzero.model.muzero_model_mlp")
    policy_module = importlib.import_module("lzero.policy.muzero")
    model_classes = [
        name
        for name, value in inspect.getmembers(model_module, inspect.isclass)
        if "MuZero" in name or "Dynamics" in name
    ]
    policy_classes = [
        name
        for name, value in inspect.getmembers(policy_module, inspect.isclass)
        if "MuZero" in name or "MCTS" in name
    ]
    return {
        "packages": {
            "LightZero": metadata.version("LightZero"),
            "DI-engine": metadata.version("DI-engine"),
            "torch": torch.__version__,
            "lzero_file": getattr(lzero, "__file__", None),
            "model_module_file": getattr(model_module, "__file__", None),
            "policy_module_file": getattr(policy_module, "__file__", None),
        },
        "model_class": "lzero.model.muzero_model_mlp.MuZeroModelMLP",
        "model_signature": str(inspect.signature(MuZeroModelMLP)),
        "policy_signature": str(inspect.signature(MuZeroPolicy)),
        "model_classes": model_classes,
        "policy_classes": policy_classes,
        "constructed_state": {
            "key_count": len(state_keys),
            "keys_sample": state_keys[:80],
            "has_fc_dynamics": any("dynamics_network.fc_dynamics." in key for key in state_keys),
            "has_fc_dynamics_1": any("dynamics_network.fc_dynamics_1." in key for key in state_keys),
            "has_fc_dynamics_2": any("dynamics_network.fc_dynamics_2." in key for key in state_keys),
        },
        "model_source_fc_dynamics": _source_excerpt(MuZeroModelMLP, needle="fc_dynamics"),
        "policy_source_eval": _source_excerpt(MuZeroPolicy, needle="eval_mode"),
        "policy_source_forward_eval_action_mask": _source_excerpt(
            MuZeroPolicy._forward_eval,
            needle="action_mask",
            radius=80,
        ),
    }


def _try_strict_load(model: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        loaded = model.load_state_dict(state_dict, strict=True)
        return {
            "ok": True,
            "strict": True,
            "missing_keys": list(getattr(loaded, "missing_keys", [])),
            "unexpected_keys": list(getattr(loaded, "unexpected_keys", [])),
        }
    except Exception as exc:
        return _exception_result(exc)


def _try_relaxed_dynamics_only_load(model: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        loaded = model.load_state_dict(state_dict, strict=False)
    except Exception as exc:
        return _exception_result(exc)
    missing = list(getattr(loaded, "missing_keys", []))
    unexpected = list(getattr(loaded, "unexpected_keys", []))
    mismatch = [*missing, *unexpected]
    ok = bool(mismatch) and all(str(key).startswith("dynamics_network.") for key in mismatch)
    return {
        "ok": ok,
        "strict": False,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "reason": (
            "dynamics_network_only_mismatch_policy_head_diagnostic"
            if ok
            else "non_dynamics_mismatch"
        ),
    }


def _strict_load_attempts(model_cfg: dict[str, Any], state_dict: dict[str, Any]) -> dict[str, Any]:
    from lzero.model.muzero_model_mlp import MuZeroModelMLP

    model = MuZeroModelMLP(**model_cfg)
    as_is = _try_strict_load(model, state_dict)
    residual_dynamics_cfg = {**model_cfg, "res_connection_in_dynamics": True}
    residual_model = MuZeroModelMLP(**residual_dynamics_cfg)
    residual_dynamics = _try_strict_load(residual_model, state_dict)
    residual_dynamics["model_config_delta"] = {"res_connection_in_dynamics": True}
    mapped_state = _map_split_dynamics_to_single_if_shape_safe(model.state_dict(), state_dict)
    mapped_result: dict[str, Any]
    if mapped_state["ok"]:
        mapped_model = MuZeroModelMLP(**model_cfg)
        mapped_result = _try_strict_load(mapped_model, mapped_state["state_dict"])
        mapped_result["mapping"] = {
            key: value for key, value in mapped_state.items() if key != "state_dict"
        }
    else:
        mapped_result = {"ok": False, "mapping": mapped_state}
    return {
        "as_is": as_is,
        "res_connection_in_dynamics_true": residual_dynamics,
        "mapped_split_dynamics": mapped_result,
    }


def _policy_head_diagnostic(model_cfg: dict[str, Any], state_dict: dict[str, Any]) -> dict[str, Any]:
    import torch
    from lzero.model.muzero_model_mlp import MuZeroModelMLP

    model = MuZeroModelMLP(**model_cfg)
    strict = _try_strict_load(model, state_dict)
    load_result = strict
    load_kind = "strict"
    if not strict.get("ok"):
        residual_cfg = {**model_cfg, "res_connection_in_dynamics": True}
        residual_model = MuZeroModelMLP(**residual_cfg)
        residual = _try_strict_load(residual_model, state_dict)
        if residual.get("ok"):
            model = residual_model
            load_result = residual
            load_result["model_config_delta"] = {"res_connection_in_dynamics": True}
            load_kind = "strict_res_connection_in_dynamics_true"
        else:
            model = MuZeroModelMLP(**model_cfg)
            relaxed = _try_relaxed_dynamics_only_load(model, state_dict)
            load_result = relaxed
            load_kind = "relaxed_dynamics_only"
            if not relaxed.get("ok"):
                return {
                    "ok": False,
                    "load_kind": load_kind,
                    "load_state_dict": relaxed,
                    "strict_error": strict,
                    "strict_res_connection_in_dynamics_true_error": residual,
                    "reason": "cannot load even policy-head diagnostic safely",
                }

    model.eval()
    observations = {
        name: list(values)
        for name, values in DIAGNOSTIC_OBSERVATIONS.items()
    }
    obs_tensor = torch.tensor(list(observations.values()), dtype=torch.float32)
    with torch.no_grad():
        network_output = model.initial_inference(obs_tensor)
    logits_tensor = network_output.policy_logits.detach().cpu().float()
    rows: dict[str, Any] = {}
    all_logits = []
    for idx, name in enumerate(observations):
        logits = [float(value) for value in logits_tensor[idx].reshape(-1).tolist()]
        spread = max(logits) - min(logits)
        rows[name] = {
            "observation": observations[name],
            "policy_logits": logits,
            "argmax_action": int(max(range(len(logits)), key=logits.__getitem__)),
            "spread": float(spread),
            "max_abs": float(max(abs(value) for value in logits)),
            "nearly_equal": bool(spread <= 1e-6),
        }
        all_logits.extend(logits)
    unique_rows = {
        tuple(round(value, 8) for value in row["policy_logits"])
        for row in rows.values()
    }
    return {
        "ok": True,
        "load_kind": load_kind,
        "load_state_dict": load_result,
        "rows": rows,
        "all_zero_or_near_zero": bool(all(abs(value) <= 1e-8 for value in all_logits)),
        "all_rows_identical_rounded_1e_8": len(unique_rows) == 1,
        "max_spread": float(max(row["spread"] for row in rows.values())),
        "note": "Direct initial_inference policy-head diagnostic only; no recurrent dynamics, no MCTS.",
    }


def _map_split_dynamics_to_single_if_shape_safe(
    model_state: dict[str, Any],
    checkpoint_state: dict[str, Any],
) -> dict[str, Any]:
    """Map ``fc_dynamics_1`` to ``fc_dynamics`` only when it is shape-identical.

    ``fc_dynamics_2`` has no same-shaped destination in the installed model, so
    a strict full load still cannot be claimed after this mapping. Keeping that
    failure explicit avoids hiding a model-variant mismatch.
    """

    mapped = OrderedDict()
    rewrites: dict[str, str] = {}
    dropped: list[str] = []
    unsafe: list[dict[str, Any]] = []
    for key, value in checkpoint_state.items():
        key_text = str(key)
        if key_text.startswith("dynamics_network.fc_dynamics_1."):
            target_key = key_text.replace("dynamics_network.fc_dynamics_1.", "dynamics_network.fc_dynamics.", 1)
            target = model_state.get(target_key)
            if target is None:
                unsafe.append({"source": key_text, "target": target_key, "reason": "target_missing"})
                mapped[key_text] = value
                continue
            if tuple(target.shape) != tuple(value.shape):
                unsafe.append(
                    {
                        "source": key_text,
                        "target": target_key,
                        "reason": "shape_mismatch",
                        "source_shape": [int(item) for item in value.shape],
                        "target_shape": [int(item) for item in target.shape],
                    }
                )
                mapped[key_text] = value
                continue
            mapped[target_key] = value
            rewrites[key_text] = target_key
        elif key_text.startswith("dynamics_network.fc_dynamics_2."):
            dropped.append(key_text)
        else:
            mapped[key_text] = value

    return {
        "ok": not unsafe,
        "state_dict": mapped,
        "rewrites": rewrites,
        "dropped_unmapped_fc_dynamics_2": dropped,
        "unsafe": unsafe,
        "reason": (
            "fc_dynamics_1 is shape-identical to installed fc_dynamics; "
            "fc_dynamics_2 has no destination in this installed class"
        ),
    }


def _main_config_with_residual_dynamics(main_config: Any, *, enabled: bool) -> Any:
    if not enabled:
        return main_config
    cfg = copy.deepcopy(main_config)
    cfg["policy"]["model"]["res_connection_in_dynamics"] = True
    return cfg


def _eval_mode_forward_attempt(
    *,
    main_config: Any,
    create_config: Any,
    checkpoint_path: Path,
    seed: int,
) -> dict[str, Any]:
    from ding.config import compile_config
    from lzero.policy.muzero import MuZeroPolicy

    cfg = compile_config(
        copy.deepcopy(main_config),
        seed=seed,
        auto=True,
        create_cfg=copy.deepcopy(create_config),
        save_cfg=False,
    )
    try:
        if not hasattr(cfg.policy, "device"):
            cfg.policy.device = "cpu"
        policy = MuZeroPolicy(cfg.policy)
        policy_model = getattr(policy, "_model", None)
        if policy_model is None:
            raise AttributeError("MuZeroPolicy has no _model attribute for checkpoint load")
        policy_model.load_state_dict(_torch_load(checkpoint_path)["model"], strict=True)
    except Exception as exc:
        return {
            "ok": False,
            "stage": "policy_construct_or_load_state_dict",
            **_exception_result(exc),
        }
    try:
        import numpy as np
        import torch
        from curvyzero.training.lightzero_dummy_pong_env import DummyPongLightZeroEnv

        env = DummyPongLightZeroEnv(cfg.env)
        env.seed(seed, dynamic_seed=False)
        observation = env.reset()
        obs_tensor = torch.as_tensor(np.asarray([observation["observation"]]), dtype=torch.float32)
        action_mask = np.asarray([observation["action_mask"]], dtype=np.float32)
        output = policy.eval_mode.forward(
            obs_tensor,
            action_mask=action_mask,
            to_play=[int(observation["to_play"])],
            ready_env_id=np.array([0]),
        )
        return {
            "ok": True,
            "output_type": type(output).__name__,
            "output": _to_plain(output),
            "call": {
                "api": "MuZeroPolicy.eval_mode.forward",
                "data_shape": [int(item) for item in obs_tensor.shape],
                "action_mask": _to_plain(action_mask),
                "to_play": [int(observation["to_play"])],
                "ready_env_id": [0],
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "stage": "eval_mode_forward",
            **_exception_result(exc),
        }


def _output_ref(run_id: str, attempt_id: str) -> str:
    return (
        runs.attempt_root_ref(TASK_ID, run_id, attempt_id)
        / "probe"
        / f"lightzero_mcts_loader_smoke_{runs.utc_stamp()}.json"
    ).as_posix()


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=10 * 60)
def lightzero_dummy_pong_mcts_loader_smoke(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    opponent_policy: str = DEFAULT_OPPONENT_POLICY,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
) -> dict[str, Any]:
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    output_ref = output_ref or _output_ref(run_id, attempt_id)
    output_path = runs.volume_path(RUNS_MOUNT, output_ref)
    started_at = runs.utc_timestamp()
    try:
        payload = _torch_load(checkpoint_path)
        state_path, state_dict = _state_dict_from_checkpoint(payload)
        config = _model_config(
            env=env,
            feature_mode=feature_mode,
            opponent_policy=opponent_policy,
            seed=seed,
            max_env_step=max_env_step,
            num_simulations=num_simulations,
        )
        lightzero_surface = _inspect_lightzero_surface(config["model_cfg"])
        strict_load = _strict_load_attempts(config["model_cfg"], state_dict)
        strict_full_model_load_ok = bool(
            strict_load["as_is"].get("ok")
            or strict_load["res_connection_in_dynamics_true"].get("ok")
        )
        eval_attempt = (
            _eval_mode_forward_attempt(
                main_config=_main_config_with_residual_dynamics(
                    config["main_config"],
                    enabled=bool(strict_load["res_connection_in_dynamics_true"].get("ok")),
                ),
                create_config=config["create_config"],
                checkpoint_path=checkpoint_path,
                seed=seed,
            )
            if strict_full_model_load_ok
            else {
                "ok": False,
                "stage": "not_attempted",
                "reason": "strict full model load failed; do not fake MCTS/eval",
            }
        )
        result: dict[str, Any] = {
            "schema": "curvyzero_lightzero_dummy_pong_mcts_loader_smoke/v0",
            "ok": bool(strict_full_model_load_ok and eval_attempt.get("ok")),
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "config": {
                "checkpoint_ref": checkpoint_ref,
                "checkpoint_path": str(checkpoint_path),
                "output_ref": output_ref,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "env": env,
                "feature_mode": feature_mode,
                "opponent_policy": opponent_policy,
                "seed": seed,
                "max_env_step": max_env_step,
                "num_simulations": num_simulations,
                "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            },
            "checkpoint": {
                "exists": checkpoint_path.is_file(),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
                "last_iter": _to_plain(payload.get("last_iter")) if isinstance(payload, dict) else None,
                "last_step": _to_plain(payload.get("last_step")) if isinstance(payload, dict) else None,
                "state_dict_path": state_path,
                "state_dict": _summarize_state_dict(state_dict),
            },
            "model_config": _to_plain(config["model_cfg"]),
            "config_surface": _to_plain(config["surface"]),
            "lightzero_surface": _to_plain(lightzero_surface),
            "strict_load": _to_plain(strict_load),
            "strict_full_model_load_ok": strict_full_model_load_ok,
            "strict_full_model_load_variant": (
                "as_is"
                if strict_load["as_is"].get("ok")
                else (
                    "res_connection_in_dynamics_true"
                    if strict_load["res_connection_in_dynamics_true"].get("ok")
                    else None
                )
            ),
            "policy_head_diagnostic": _to_plain(
                _policy_head_diagnostic(config["model_cfg"], state_dict)
            ),
            "eval_mode_forward": _to_plain(eval_attempt),
            "mcts_eval_status": (
                "ok"
                if strict_full_model_load_ok and eval_attempt.get("ok")
                else (
                    "blocked_eval_forward_failed"
                    if strict_full_model_load_ok
                    else "blocked_strict_load_failed"
                )
            ),
            "plain_language": {
                "strict_load": (
                    "The installed LightZero MuZeroModelMLP does not strictly load this checkpoint "
                    "unless strict_load.as_is.ok is true."
                ),
                "mapping": (
                    "A fc_dynamics_1 to fc_dynamics mapping is inspected only for shape safety. "
                    "fc_dynamics_2 remains unmapped, so this is not a valid full-model fix."
                ),
                "mcts": (
                    "MuZeroPolicy.eval_mode.forward is only attempted after strict full model load."
                ),
            },
        }
    except Exception as exc:  # pragma: no cover - remote diagnosis.
        result = {
            "schema": "curvyzero_lightzero_dummy_pong_mcts_loader_smoke/v0",
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "wrapper_error": _exception_result(exc),
        }

    json_safe_result = json.loads(json.dumps(_to_plain(result)))
    runs.write_json(output_path, json_safe_result)
    result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    json_safe_result["artifact"] = result["artifact"]
    runs_volume.commit()
    print(json.dumps(json_safe_result, indent=2, sort_keys=True))
    return json_safe_result


@app.local_entrypoint()
def main(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    opponent_policy: str = DEFAULT_OPPONENT_POLICY,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
) -> None:
    result = lightzero_dummy_pong_mcts_loader_smoke.remote(
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        num_simulations=num_simulations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
