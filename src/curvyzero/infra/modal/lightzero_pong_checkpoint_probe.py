"""Modal checkpoint-load probe for stock LightZero visual Atari Pong.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_checkpoint_probe

This is the checkpoint-load follow-up to ``lightzero_pong_tiny_train_smoke``.
It loads the mirrored stock Atari ``iteration_1.pth.tar`` into the matching
LightZero MuZero conv model/policy/config surface and runs only a cheap direct
zero-observation forward. It does not score gameplay.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_atari_rom_image import build_lightzero_atari_rom_image
from curvyzero.infra.modal.lightzero_pong_dry_config_smoke import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTOR_ENV_NUM,
    DEFAULT_ENV_ID,
    DEFAULT_EVALUATOR_ENV_NUM,
    DEFAULT_MAX_ENV_STEP,
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_SEED,
    DEFAULT_UPDATE_PER_COLLECT,
    LIGHTZERO_VERSION,
)
from curvyzero.infra.modal.lightzero_pong_tiny_train_smoke import (
    DEFAULT_GAME_SEGMENT_LENGTH,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_MAX_TRAIN_ITER,
    _patched_stock_atari_pong_configs,
    _version_or_missing,
)


APP_NAME = "curvyzero-lightzero-pong-checkpoint-probe"
TASK_ID = "lightzero-official-visual-pong"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_RUN_ID = "lz-visual-pong-20260509T171834Z-1798cd6bef57"
DEFAULT_ATTEMPT_ID = "attempt-20260509T171834Z-fd4b5559bec6"
DEFAULT_CHECKPOINT_REF = (
    "training/lightzero-official-visual-pong/"
    "lz-visual-pong-20260509T171834Z-1798cd6bef57/"
    "checkpoints/lightzero/iteration_1.pth.tar"
)

image = (
    build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_to_plain(item) for item in sorted(value, key=repr)]
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
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_summary(path: Path, *, include_sha256: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        summary["bytes"] = path.stat().st_size
        if include_sha256:
            summary["sha256"] = _sha256(path)
    return summary


def _torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


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
        summary["len"] = len(value)
        summary["keys_sample"] = [str(key) for key in value.keys()][:40]
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
            "_model",
            "_learn_model",
        ):
            if key in payload:
                candidates.append((key, payload[key]))
        for key, value in payload.items():
            if isinstance(value, dict):
                for nested_key in (
                    "model",
                    "state_dict",
                    "model_state_dict",
                    "_model",
                    "_learn_model",
                ):
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
        keys = [str(key) for key in value]
        if any("representation_network" in key for key in keys):
            score += 1000
        if any("prediction_network" in key for key in keys):
            score += 1000
        if any("dynamics_network" in key for key in keys):
            score += 1000
        if best is None or score > best[2]:
            best = (path, value, score)
    if best is None:
        return None
    return best[0], best[1]


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not any(str(key).startswith(prefix) for key in state_dict):
        return state_dict
    return {str(key).removeprefix(prefix): value for key, value in state_dict.items()}


def _load_state_dict_probe(module: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
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
            errors.append({"candidate": name, "strict": True, "error": str(exc)})

    for name, candidate in candidates:
        try:
            loaded = module.load_state_dict(candidate, strict=False)
            return {
                "ok": True,
                "candidate": name,
                "strict": False,
                "missing_keys": list(getattr(loaded, "missing_keys", []))[:80],
                "unexpected_keys": list(getattr(loaded, "unexpected_keys", []))[:80],
                "note": "non-strict load is only a probe signal; do not score from it",
            }
        except Exception as exc:
            errors.append({"candidate": name, "strict": False, "error": str(exc)})

    return {
        "ok": False,
        "strict": False,
        "errors": errors[-8:],
    }


def _module_shape(module: Any, *, limit: int = 20) -> dict[str, Any]:
    named_parameters = list(module.named_parameters())
    total_parameters = sum(int(param.numel()) for _, param in named_parameters)
    trainable_parameters = sum(
        int(param.numel()) for _, param in named_parameters if param.requires_grad
    )
    return {
        "class": f"{type(module).__module__}.{type(module).__name__}",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_sample": [
            {
                "name": name,
                "shape": [int(item) for item in param.shape],
                "dtype": str(param.dtype),
            }
            for name, param in named_parameters[:limit]
        ],
    }


def _network_output_summary(output: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in ("value", "reward", "policy_logits", "latent_state"):
        if hasattr(output, name):
            item = getattr(output, name)
            summary[name] = _summarize_value(item)
            if name == "policy_logits" and hasattr(item, "detach"):
                logits = item.detach().cpu().reshape(-1).tolist()
                summary["policy_logits_sample"] = [float(value) for value in logits[:12]]
                if logits:
                    summary["greedy_action_id"] = int(max(range(len(logits)), key=logits.__getitem__))
    return summary


def _direct_model_probe(
    *,
    model_cfg: dict[str, Any],
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from lzero.model.muzero_model import MuZeroModel

    model = MuZeroModel(**model_cfg)
    load = _load_state_dict_probe(model, state_dict)
    model.eval()
    observation_shape = [int(item) for item in model_cfg["observation_shape"]]
    forward: dict[str, Any] = {"ok": False}
    if load["ok"]:
        try:
            obs = torch.zeros((1, *observation_shape), dtype=torch.float32)
            with torch.no_grad():
                output = model.initial_inference(obs)
            forward = {
                "ok": True,
                "input_shape": [int(item) for item in obs.shape],
                "output": _network_output_summary(output),
            }
        except Exception as exc:
            forward = _exception_result(exc)
    return {
        "ok": bool(load["ok"] and forward.get("ok")),
        "model_class": "lzero.model.muzero_model.MuZeroModel",
        "model_signature": str(inspect.signature(MuZeroModel)),
        "model_config": _to_plain(model_cfg),
        "shape": _module_shape(model),
        "load_state_dict": load,
        "forward": forward,
    }


def _policy_probe(
    *,
    main_config: Any,
    create_config: Any,
    state_dict: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    from ding.config import compile_config
    from lzero.policy.muzero import MuZeroPolicy

    config = copy.deepcopy(main_config)
    config["policy"]["cuda"] = False
    cfg = compile_config(
        config,
        seed=seed,
        auto=True,
        create_cfg=copy.deepcopy(create_config),
        save_cfg=False,
    )
    if not hasattr(cfg.policy, "device"):
        cfg.policy.device = "cpu"
    cfg.policy.cuda = False
    policy = MuZeroPolicy(cfg.policy)
    policy_model = getattr(policy, "_model", None)
    if policy_model is None:
        raise AttributeError("MuZeroPolicy has no _model attribute")
    load = _load_state_dict_probe(policy_model, state_dict)
    if hasattr(policy_model, "eval"):
        policy_model.eval()
    return {
        "ok": bool(load["ok"]),
        "policy_class": f"{type(policy).__module__}.{type(policy).__name__}",
        "policy_signature": str(inspect.signature(MuZeroPolicy)),
        "policy_model_shape": _module_shape(policy_model),
        "load_state_dict": load,
        "compiled_surface": {
            "cuda": bool(cfg.policy.cuda),
            "device": str(getattr(cfg.policy, "device", "missing")),
            "model_type": str(cfg.policy.model.model_type),
            "observation_shape": _to_plain(cfg.policy.model.observation_shape),
            "action_space_size": int(cfg.policy.model.action_space_size),
            "num_simulations": int(cfg.policy.num_simulations),
            "batch_size": int(cfg.policy.batch_size),
            "env_type": str(cfg.policy.env_type),
        },
    }


def _default_output_ref(*, run_id: str, attempt_id: str) -> str:
    return (
        runs.attempt_root_ref(TASK_ID, run_id, attempt_id)
        / "probe"
        / f"lightzero_visual_pong_checkpoint_load_{runs.utc_stamp()}.json"
    ).as_posix()


def _status(result: dict[str, Any]) -> dict[str, Any]:
    direct_load = result.get("direct_model_probe", {}).get("load_state_dict", {})
    policy_load = result.get("policy_probe", {}).get("load_state_dict", {})
    return {
        "checkpoint_load_ok": bool(result.get("load", {}).get("ok")),
        "state_dict_ok": bool(result.get("state_dict", {}).get("ok")),
        "strict_direct_model_load_ok": bool(direct_load.get("ok") and direct_load.get("strict")),
        "partial_direct_model_load_ok": bool(direct_load.get("ok") and not direct_load.get("strict")),
        "strict_policy_model_load_ok": bool(policy_load.get("ok") and policy_load.get("strict")),
        "partial_policy_model_load_ok": bool(policy_load.get("ok") and not policy_load.get("strict")),
        "direct_forward_ok": bool(result.get("direct_model_probe", {}).get("forward", {}).get("ok")),
    }


def _probe_checkpoint(
    *,
    checkpoint_path: Path,
    run_id: str,
    attempt_id: str,
    env_id: str,
    seed: int,
    max_env_step: int,
    max_train_iter: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    max_episode_steps: int,
    game_segment_length: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "curvyzero_lightzero_visual_pong_checkpoint_load_probe/v0",
        "checkpoint": _checkpoint_summary(checkpoint_path),
        "load": {"ok": False},
        "state_dict": {"ok": False},
        "direct_model_probe": {"ok": False},
        "policy_probe": {"ok": False},
    }

    checkpoint = _torch_load(checkpoint_path)
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
        "keys_sample": list(state_dict)[:40],
        "tensor_sample": {
            key: _summarize_value(value)
            for key, value in list(state_dict.items())[:12]
        },
    }

    patched = _patched_stock_atari_pong_configs(
        env_id=env_id,
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
        use_cuda=False,
    )
    model_cfg = dict(patched["main_config"]["policy"]["model"])
    model_type = str(model_cfg.pop("model_type", "conv"))
    if model_type != "conv":
        raise ValueError(f"expected stock Atari model_type='conv', got {model_type!r}")

    result["stock_example"] = {
        "task": env_id,
        "algorithm": "MuZero",
        "module": patched["module"],
        "trainer_entrypoint": "lzero.entry.train_muzero",
        "original_surface": patched["original_surface"],
        "patched_surface": patched["patched_surface"],
        "patches": patched["patches"],
    }
    result["direct_model_probe"] = _direct_model_probe(
        model_cfg=model_cfg,
        state_dict=state_dict,
    )
    try:
        result["policy_probe"] = _policy_probe(
            main_config=patched["main_config"],
            create_config=patched["create_config"],
            state_dict=state_dict,
            seed=seed,
        )
    except Exception as exc:
        result["policy_probe"] = _exception_result(exc)
    return result


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60, cpu=1.0)
def lightzero_pong_checkpoint_probe(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
) -> dict[str, Any]:
    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    output_ref = output_ref or _default_output_ref(run_id=run_id, attempt_id=attempt_id)
    output_path = runs.volume_path(RUNS_MOUNT, output_ref)
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "gym": _version_or_missing("gym"),
        "gymnasium": _version_or_missing("gymnasium"),
        "ale-py": _version_or_missing("ale-py", "ale_py"),
        "opencv-python-headless": _version_or_missing("opencv-python-headless"),
        "AutoROM": _version_or_missing("AutoROM"),
    }
    config = {
        "job_kind": "lightzero_official_visual_pong_checkpoint_load_probe",
        "checkpoint_ref": checkpoint_ref,
        "output_ref": output_ref,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "env_id": env_id,
        "seed": seed,
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "collector_env_num": collector_env_num,
        "evaluator_env_num": evaluator_env_num,
        "num_simulations": num_simulations,
        "batch_size": batch_size,
        "update_per_collect": update_per_collect,
        "max_episode_steps": max_episode_steps,
        "game_segment_length": game_segment_length,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }

    try:
        probe = _probe_checkpoint(
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            attempt_id=attempt_id,
            env_id=env_id,
            seed=seed,
            max_env_step=max_env_step,
            max_train_iter=max_train_iter,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            num_simulations=num_simulations,
            batch_size=batch_size,
            update_per_collect=update_per_collect,
            max_episode_steps=max_episode_steps,
            game_segment_length=game_segment_length,
        )
        result: dict[str, Any] = {
            **probe,
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
        }
        result["status"] = _status(result)
        result["ok"] = bool(
            result["status"]["checkpoint_load_ok"]
            and result["status"]["state_dict_ok"]
            and result["status"]["strict_direct_model_load_ok"]
            and result["status"]["strict_policy_model_load_ok"]
            and result["status"]["direct_forward_ok"]
        )
    except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
        result = {
            "schema": "curvyzero_lightzero_visual_pong_checkpoint_load_probe/v0",
            "ok": False,
            "started_at": started_at,
            "ended_at": runs.utc_timestamp(),
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
            "config": config,
            "packages": packages,
            "checkpoint": _checkpoint_summary(checkpoint_path),
            "status": {
                "checkpoint_load_ok": False,
                "state_dict_ok": False,
                "strict_direct_model_load_ok": False,
                "partial_direct_model_load_ok": False,
                "strict_policy_model_load_ok": False,
                "partial_policy_model_load_ok": False,
                "direct_forward_ok": False,
            },
            "wrapper_error": _exception_result(exc),
        }

    runs.write_json(output_path, _to_plain(result))
    result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.local_entrypoint()
def main(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    output_ref: str | None = None,
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
) -> None:
    result = lightzero_pong_checkpoint_probe.remote(
        checkpoint_ref=checkpoint_ref,
        output_ref=output_ref,
        run_id=run_id,
        attempt_id=attempt_id,
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        max_episode_steps=max_episode_steps,
        game_segment_length=game_segment_length,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
