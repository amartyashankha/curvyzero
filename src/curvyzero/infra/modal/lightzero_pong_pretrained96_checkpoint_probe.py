"""Strict-load probe for the OpenDILab HF 96x96 Atari Pong MuZero checkpoint.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_pretrained96_checkpoint_probe

This is intentionally separate from ``lightzero_pong_checkpoint_probe``. The
current stock Atari wrapper uses LightZero's 64x64 ``atari_muzero_config`` path;
this module recreates the older model-card-compatible 96x96/downsample surface
for ``OpenDILabCommunity/PongNoFrameskip-v4-MuZero``.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import runpy
import time
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs
from curvyzero.infra.modal.lightzero_atari_rom_image import build_lightzero_atari_rom_image
from curvyzero.infra.modal.lightzero_pong_checkpoint_probe import (
    _checkpoint_summary,
    _exception_result,
    _find_state_dict,
    _module_shape,
    _network_output_summary,
    _summarize_value,
    _torch_load,
    _to_plain,
)
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
    TASK_ID as STOCK_TASK_ID,
    VOLUME_NAME,
    _set_path,
    _set_path_creating_dicts,
    _set_path_if_present,
    _version_or_missing,
)


APP_NAME = "curvyzero-lightzero-pong-pretrained96-checkpoint-probe"
TASK_ID = "lightzero-official-visual-pong-pretrained96"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_RUN_ID = "lz-visual-pong-pretrained96-opendilab-hf"
DEFAULT_ATTEMPT_ID = "strict-load-forward-pretrained96-hf"
DEFAULT_CHECKPOINT_REF = (
    f"training/{STOCK_TASK_ID}/pretrained/"
    "OpenDILabCommunity/PongNoFrameskip-v4-MuZero/pytorch_model.bin"
)
DEFAULT_POLICY_CONFIG_REF = (
    f"training/{STOCK_TASK_ID}/pretrained/"
    "OpenDILabCommunity/PongNoFrameskip-v4-MuZero/policy_config.py"
)
EXPECTED_OBSERVATION_SHAPE = [4, 96, 96]
EXPECTED_ACTION_SPACE_SIZE = 6

image = (
    build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _extract_pretrained96_surface(
    main_config: Any,
    create_config: Any,
    *,
    max_env_step: int,
    max_train_iter: int,
) -> dict[str, Any]:
    return {
        "env_id": main_config["env"]["env_id"],
        "policy_type": create_config["policy"]["type"],
        "env_type": create_config["env"]["type"],
        "env_manager_type": create_config["env_manager"]["type"],
        "model_type": main_config["policy"]["model"].get("model_type", "conv"),
        "observation_shape": _to_plain(main_config["policy"]["model"]["observation_shape"]),
        "action_space_size": main_config["policy"]["model"]["action_space_size"],
        "downsample": bool(main_config["policy"]["model"].get("downsample", False)),
        "collector_env_num": main_config["env"]["collector_env_num"],
        "evaluator_env_num": main_config["env"]["evaluator_env_num"],
        "n_evaluator_episode": main_config["env"].get("n_evaluator_episode"),
        "collect_max_episode_steps": main_config["env"].get("collect_max_episode_steps"),
        "eval_max_episode_steps": main_config["env"].get("eval_max_episode_steps"),
        "num_simulations": main_config["policy"]["num_simulations"],
        "batch_size": main_config["policy"]["batch_size"],
        "update_per_collect": main_config["policy"].get("update_per_collect"),
        "n_episode": main_config["policy"].get("n_episode"),
        "game_segment_length": main_config["policy"].get("game_segment_length"),
        "cuda": main_config["policy"]["cuda"],
        "max_env_step": max_env_step,
        "max_train_iter": max_train_iter,
        "exp_name": str(main_config["exp_name"]),
    }


def _capture_pretrained96_configs(*, env_id: str, seed: int) -> dict[str, Any]:
    module_name = "zoo.atari.config.atari_muzero_segment_config"
    module = importlib.import_module(module_name)
    action_map_module = importlib.import_module("zoo.atari.config.atari_env_action_space_map")
    action_map = getattr(action_map_module, "atari_env_action_space_map")
    entry_module = importlib.import_module("lzero.entry")
    original_train_muzero_segment = entry_module.train_muzero_segment
    captured: dict[str, Any] = {}

    def capture_train_muzero_segment(configs, *, seed: int, max_env_step: int, **kwargs):
        captured["configs"] = configs
        captured["seed"] = seed
        captured["max_env_step"] = max_env_step
        captured["kwargs"] = kwargs
        return {"captured": True}

    entry_module.train_muzero_segment = capture_train_muzero_segment
    try:
        module.main(env_id, seed)
    finally:
        entry_module.train_muzero_segment = original_train_muzero_segment

    main_config, create_config = captured["configs"]
    return {
        "module": module_name,
        "main_config": main_config,
        "create_config": create_config,
        "captured_seed": captured["seed"],
        "captured_max_env_step": captured["max_env_step"],
        "captured_kwargs": _to_plain(captured["kwargs"]),
        "default_env_action_space_size": action_map[env_id],
    }


def _load_hf_policy_config(policy_config_ref: str) -> dict[str, Any]:
    from easydict import EasyDict

    config_path = runs.volume_path(RUNS_MOUNT, policy_config_ref)
    payload = runpy.run_path(config_path)
    exp_config = payload.get("exp_config")
    if not isinstance(exp_config, dict):
        raise ValueError(f"{policy_config_ref} did not define dict exp_config")
    main_config = exp_config.get("main_config")
    create_config = exp_config.get("create_config")
    if not isinstance(main_config, dict) or not isinstance(create_config, dict):
        raise ValueError(f"{policy_config_ref} exp_config missing main_config/create_config dicts")
    return {
        "module": "hf_policy_config.py",
        "main_config": EasyDict(main_config),
        "create_config": EasyDict(create_config),
        "policy_config_ref": policy_config_ref,
        "captured_seed": main_config.get("seed"),
        "captured_max_env_step": 500000,
        "captured_kwargs": {},
        "default_env_action_space_size": main_config.get("policy", {})
        .get("model", {})
        .get("action_space_size"),
    }


def _patched_pretrained96_atari_pong_configs(
    *,
    env_id: str,
    seed: int,
    run_id: str,
    attempt_id: str,
    policy_config_ref: str | None,
    max_env_step: int,
    max_train_iter: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    max_episode_steps: int,
    game_segment_length: int,
    use_cuda: bool,
) -> dict[str, Any]:
    captured = (
        _load_hf_policy_config(policy_config_ref)
        if policy_config_ref
        else _capture_pretrained96_configs(env_id=env_id, seed=seed)
    )
    main_config = copy.deepcopy(captured["main_config"])
    create_config = copy.deepcopy(captured["create_config"])
    original_surface = _extract_pretrained96_surface(
        captured["main_config"],
        captured["create_config"],
        max_env_step=captured["captured_max_env_step"],
        max_train_iter=int(1e10),
    )
    exp_name = Path("/tmp") / "curvyzero-lightzero-pretrained96-pong" / run_id / attempt_id / "exp"
    patches = [
        _set_path(main_config, ("exp_name",), str(exp_name)),
        _set_path(main_config, ("env", "env_id"), env_id),
        _set_path(main_config, ("env", "collector_env_num"), collector_env_num),
        _set_path(main_config, ("env", "evaluator_env_num"), evaluator_env_num),
        _set_path_if_present(main_config, ("env", "n_evaluator_episode"), evaluator_env_num),
        _set_path_creating_dicts(main_config, ("env", "collect_max_episode_steps"), max_episode_steps),
        _set_path_creating_dicts(main_config, ("env", "eval_max_episode_steps"), max_episode_steps),
        _set_path_if_present(main_config, ("policy", "collector_env_num"), collector_env_num),
        _set_path_if_present(main_config, ("policy", "evaluator_env_num"), evaluator_env_num),
        _set_path_if_present(main_config, ("policy", "n_episode"), collector_env_num),
        _set_path(main_config, ("policy", "cuda"), use_cuda),
        _set_path(main_config, ("policy", "model", "observation_shape"), tuple(EXPECTED_OBSERVATION_SHAPE)),
        _set_path(main_config, ("policy", "model", "action_space_size"), EXPECTED_ACTION_SPACE_SIZE),
        _set_path(main_config, ("policy", "model", "downsample"), True),
        _set_path(main_config, ("policy", "num_simulations"), num_simulations),
        _set_path(main_config, ("policy", "batch_size"), batch_size),
        _set_path_if_present(main_config, ("policy", "update_per_collect"), update_per_collect),
        _set_path_if_present(main_config, ("policy", "game_segment_length"), game_segment_length),
        _set_path_if_present(main_config, ("policy", "eval_freq"), 1),
    ]
    return {
        "module": captured["module"],
        "main_config": main_config,
        "create_config": create_config,
        "captured": captured,
        "original_surface": original_surface,
        "patched_surface": _extract_pretrained96_surface(
            main_config,
            create_config,
            max_env_step=max_env_step,
            max_train_iter=max_train_iter,
        ),
        "patches": patches,
    }


def _validate_pretrained96_surface(surface: dict[str, Any], *, use_cuda: bool) -> list[str]:
    problems: list[str] = []
    expected = {
        "env_id": DEFAULT_ENV_ID,
        "policy_type": "muzero",
        "env_type": "atari_lightzero",
        "model_type": "conv",
        "observation_shape": EXPECTED_OBSERVATION_SHAPE,
        "action_space_size": EXPECTED_ACTION_SPACE_SIZE,
        "downsample": True,
        "cuda": use_cuda,
    }
    for key, value in expected.items():
        if surface[key] != value:
            problems.append(f"pretrained96 surface {key}={surface[key]!r}, expected {value!r}")
    return problems


def _pretrained96_policy_shape_probe(*, main_config: Any, create_config: Any, seed: int) -> dict[str, Any]:
    from ding.config import compile_config
    from lzero.policy.muzero import MuZeroPolicy

    cfg = compile_config(
        copy.deepcopy(main_config),
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
    return {
        "ok": True,
        "policy_class": f"{type(policy).__module__}.{type(policy).__name__}",
        "policy_model_shape": _module_shape(policy_model),
        "compiled_surface": {
            "cuda": bool(cfg.policy.cuda),
            "device": str(getattr(cfg.policy, "device", "missing")),
            "model_type": str(cfg.policy.model.model_type),
            "observation_shape": _to_plain(cfg.policy.model.observation_shape),
            "action_space_size": int(cfg.policy.model.action_space_size),
            "downsample": bool(cfg.policy.model.downsample),
            "num_simulations": int(cfg.policy.num_simulations),
            "batch_size": int(cfg.policy.batch_size),
            "env_type": str(cfg.policy.env_type),
        },
    }


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not any(str(key).startswith(prefix) for key in state_dict):
        return state_dict
    return {str(key).removeprefix(prefix): value for key, value in state_dict.items()}


def _strict_load_state_dict_probe(module: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "ok": False,
        "strict": True,
        "errors": errors,
        "first_error": errors[0]["error"] if errors else None,
    }


def _direct_model_strict_probe(
    *,
    model_cfg: dict[str, Any],
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from lzero.model.muzero_model import MuZeroModel

    model = MuZeroModel(**model_cfg)
    load = _strict_load_state_dict_probe(model, state_dict)
    model.eval()
    observation_shape = [int(item) for item in model_cfg["observation_shape"]]
    forward: dict[str, Any] = {
        "ok": False,
        "skipped": True,
        "reason": "strict load failed",
    }
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


def _policy_strict_probe(
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
    load = _strict_load_state_dict_probe(policy_model, state_dict)
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
            "downsample": bool(cfg.policy.model.downsample),
            "num_simulations": int(cfg.policy.num_simulations),
            "batch_size": int(cfg.policy.batch_size),
            "env_type": str(cfg.policy.env_type),
        },
    }


def _status(result: dict[str, Any]) -> dict[str, Any]:
    direct_load = result.get("direct_model_probe", {}).get("load_state_dict", {})
    policy_load = result.get("policy_probe", {}).get("load_state_dict", {})
    return {
        "checkpoint_load_ok": bool(result.get("load", {}).get("ok")),
        "state_dict_ok": bool(result.get("state_dict", {}).get("ok")),
        "surface_ok": not result.get("surface_problems"),
        "strict_direct_model_load_ok": bool(direct_load.get("ok") and direct_load.get("strict")),
        "strict_policy_model_load_ok": bool(policy_load.get("ok") and policy_load.get("strict")),
        "direct_forward_ok": bool(result.get("direct_model_probe", {}).get("forward", {}).get("ok")),
    }


def _probe_pretrained96_checkpoint(
    *,
    checkpoint_path: Path,
    run_id: str,
    attempt_id: str,
    env_id: str,
    seed: int,
    policy_config_ref: str | None,
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
        "schema": "curvyzero_lightzero_visual_pong_pretrained96_checkpoint_probe/v0",
        "checkpoint": _checkpoint_summary(checkpoint_path),
        "load": {"ok": False},
        "state_dict": {"ok": False},
        "surface_problems": [],
        "direct_model_probe": {"ok": False},
        "policy_shape_probe": {"ok": False},
        "policy_probe": {"ok": False},
    }
    checkpoint = _torch_load(checkpoint_path)
    result["load"] = {"ok": True, "payload_summary": _summarize_value(checkpoint)}
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
        "tensor_sample": {key: _summarize_value(value) for key, value in list(state_dict.items())[:12]},
    }
    patched = _patched_pretrained96_atari_pong_configs(
        env_id=env_id,
        seed=seed,
        run_id=run_id,
        attempt_id=attempt_id,
        policy_config_ref=policy_config_ref,
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
    result["surface_problems"] = _validate_pretrained96_surface(
        patched["patched_surface"],
        use_cuda=False,
    )
    model_cfg = dict(patched["main_config"]["policy"]["model"])
    model_type = str(model_cfg.pop("model_type", "conv"))
    if model_type != "conv":
        raise ValueError(f"expected pretrained96 model_type='conv', got {model_type!r}")

    result["pretrained_example"] = {
        "source": "OpenDILabCommunity/PongNoFrameskip-v4-MuZero",
        "checkpoint_config_surface": (
            patched["captured"].get("policy_config_ref")
            or "policy_config.py model-card-compatible recreation"
        ),
        "module": patched["module"],
        "original_surface": patched["original_surface"],
        "patched_surface": patched["patched_surface"],
        "patches": patched["patches"],
    }
    result["direct_model_probe"] = _direct_model_strict_probe(
        model_cfg=model_cfg,
        state_dict=state_dict,
    )
    try:
        result["policy_shape_probe"] = _pretrained96_policy_shape_probe(
            main_config=patched["main_config"],
            create_config=patched["create_config"],
            seed=seed,
        )
    except Exception as exc:
        result["policy_shape_probe"] = _exception_result(exc)
    try:
        result["policy_probe"] = _policy_strict_probe(
            main_config=patched["main_config"],
            create_config=patched["create_config"],
            state_dict=state_dict,
            seed=seed,
        )
    except Exception as exc:
        result["policy_probe"] = _exception_result(exc)
    return result


def _default_output_ref() -> str:
    return (
        Path("training")
        / TASK_ID
        / "pretrained"
        / "OpenDILabCommunity"
        / "PongNoFrameskip-v4-MuZero"
        / "probe"
        / f"lightzero_visual_pong_pretrained96_strict_load_{runs.utc_stamp()}.json"
    ).as_posix()


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=8 * 60, cpu=1.0)
def lightzero_pong_pretrained96_checkpoint_probe(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    policy_config_ref: str | None = DEFAULT_POLICY_CONFIG_REF,
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
    output_ref = output_ref or _default_output_ref()
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
        "job_kind": "lightzero_official_visual_pong_pretrained96_checkpoint_probe",
        "checkpoint_ref": checkpoint_ref,
        "policy_config_ref": policy_config_ref,
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
        probe = _probe_pretrained96_checkpoint(
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            attempt_id=attempt_id,
            env_id=env_id,
            seed=seed,
            policy_config_ref=policy_config_ref,
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
        result["ok"] = all(bool(value) for value in result["status"].values())
    except Exception as exc:  # pragma: no cover - remote wrapper diagnosis.
        result = {
            "schema": "curvyzero_lightzero_visual_pong_pretrained96_checkpoint_probe/v0",
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
                "surface_ok": False,
                "strict_direct_model_load_ok": False,
                "strict_policy_model_load_ok": False,
                "direct_forward_ok": False,
            },
            "wrapper_error": _exception_result(exc),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(output_path, _to_plain(result))
    result["artifact"] = runs.file_summary(output_path, mount=RUNS_MOUNT)
    runs_volume.commit()
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.local_entrypoint()
def main(
    checkpoint_ref: str = DEFAULT_CHECKPOINT_REF,
    policy_config_ref: str | None = DEFAULT_POLICY_CONFIG_REF,
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
    result = lightzero_pong_pretrained96_checkpoint_probe.remote(
        checkpoint_ref=checkpoint_ref,
        policy_config_ref=policy_config_ref,
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
