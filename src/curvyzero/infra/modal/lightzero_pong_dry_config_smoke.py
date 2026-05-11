"""Dry-only stock LightZero Atari Pong MuZero config smoke.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_dry_config_smoke

This intentionally does not call the LightZero Atari trainer and does not
create an Atari environment. It imports the stock Pong MuZero segment config,
monkeypatches ``train_muzero_segment`` to capture the generated configs, then
reports the tiny CPU patches that would be needed before any future trainer
attempt.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

APP_NAME = "curvyzero-lightzero-pong-dry-config-smoke"
LIGHTZERO_VERSION = "0.2.0"

DEFAULT_ENV_ID = "PongNoFrameskip-v4"
DEFAULT_SEED = 0
DEFAULT_MAX_ENV_STEP = 4
DEFAULT_COLLECTOR_ENV_NUM = 1
DEFAULT_EVALUATOR_ENV_NUM = 1
DEFAULT_NUM_SIMULATIONS = 2
DEFAULT_BATCH_SIZE = 4
DEFAULT_UPDATE_PER_COLLECT = 1

image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    f"LightZero=={LIGHTZERO_VERSION}",
)

app = modal.App(APP_NAME)


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _get_path(mapping: Any, path: tuple[str, ...]) -> Any:
    current = mapping
    for part in path:
        current = current[part]
    return current


def _set_path(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        current = current[part]
    key = path[-1]
    old_value = current[key]
    current[key] = value
    return {
        "path": ".".join(path),
        "old": _to_plain(old_value),
        "new": _to_plain(value),
    }


def _set_path_if_present(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    try:
        _get_path(mapping, path)
    except KeyError:
        return {
            "path": ".".join(path),
            "old": "<missing>",
            "new": _to_plain(value),
            "skipped": True,
        }
    return _set_path(mapping, path, value)


def _extract_surface(main_config: Any, create_config: Any, *, max_env_step: int) -> dict[str, Any]:
    return {
        "env_id": main_config["env"]["env_id"],
        "policy_type": create_config["policy"]["type"],
        "env_type": create_config["env"]["type"],
        "model_type": main_config["policy"]["model"]["model_type"],
        "observation_shape": _to_plain(main_config["policy"]["model"]["observation_shape"]),
        "action_space_size": main_config["policy"]["model"]["action_space_size"],
        "collector_env_num": main_config["env"]["collector_env_num"],
        "evaluator_env_num": main_config["env"]["evaluator_env_num"],
        "n_evaluator_episode": main_config["env"].get("n_evaluator_episode"),
        "num_simulations": main_config["policy"]["num_simulations"],
        "batch_size": main_config["policy"]["batch_size"],
        "update_per_collect": main_config["policy"].get("update_per_collect"),
        "n_episode": main_config["policy"].get("n_episode"),
        "cuda": main_config["policy"]["cuda"],
        "max_env_step": max_env_step,
        "exp_name": str(main_config["exp_name"]),
    }


def _capture_stock_pong_configs(*, env_id: str, seed: int) -> dict[str, Any]:
    module_name = "zoo.atari.config.atari_muzero_segment_config"
    module = importlib.import_module(module_name)
    action_map_module = importlib.import_module("zoo.atari.config.atari_env_action_space_map")
    action_map = getattr(action_map_module, "atari_env_action_space_map")
    entry_module = importlib.import_module("lzero.entry")
    original_train_muzero_segment = entry_module.train_muzero_segment
    main_source = inspect.getsource(module.main)
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
        "captured_kwargs": captured["kwargs"],
        "default_env_action_space_size": action_map[env_id],
        "source_line_count": len(main_source.splitlines()),
        "source_tokens_present": {
            token: token in main_source
            for token in (
                "train_muzero_segment",
                "num_simulations = 50",
                "batch_size = 256",
                "max_env_step = int(5e5)",
            )
        },
    }


def _patched_pong_configs(
    *,
    env_id: str,
    seed: int,
    max_env_step: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
) -> dict[str, Any]:
    captured = _capture_stock_pong_configs(env_id=env_id, seed=seed)
    main_config = copy.deepcopy(captured["main_config"])
    create_config = copy.deepcopy(captured["create_config"])

    original_surface = _extract_surface(
        captured["main_config"],
        captured["create_config"],
        max_env_step=captured["captured_max_env_step"],
    )
    patches = [
        _set_path(
            main_config,
            ("exp_name",),
            str(Path("/tmp") / "curvyzero-lightzero-pong-dry" / f"seed-{seed}"),
        ),
        _set_path(main_config, ("env", "collector_env_num"), collector_env_num),
        _set_path(main_config, ("env", "evaluator_env_num"), evaluator_env_num),
        _set_path_if_present(main_config, ("env", "n_evaluator_episode"), evaluator_env_num),
        _set_path(main_config, ("policy", "cuda"), False),
        _set_path_if_present(main_config, ("policy", "collector_env_num"), collector_env_num),
        _set_path_if_present(main_config, ("policy", "evaluator_env_num"), evaluator_env_num),
        _set_path_if_present(main_config, ("policy", "n_episode"), 1),
        _set_path(main_config, ("policy", "num_simulations"), num_simulations),
        _set_path(main_config, ("policy", "batch_size"), batch_size),
        _set_path_if_present(main_config, ("policy", "update_per_collect"), update_per_collect),
    ]

    return {
        "module": captured["module"],
        "main_config": main_config,
        "create_config": create_config,
        "captured": {
            "seed": captured["captured_seed"],
            "max_env_step": captured["captured_max_env_step"],
            "kwargs": _to_plain(captured["captured_kwargs"]),
            "default_env_action_space_size": captured["default_env_action_space_size"],
            "source_line_count": captured["source_line_count"],
            "source_tokens_present": captured["source_tokens_present"],
        },
        "original_surface": original_surface,
        "patched_surface": _extract_surface(
            main_config,
            create_config,
            max_env_step=max_env_step,
        ),
        "patches": patches,
    }


def _validate_patched_surface(surface: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    expected = {
        "env_id": DEFAULT_ENV_ID,
        "policy_type": "muzero",
        "env_type": "atari_lightzero",
        "model_type": "conv",
        "action_space_size": 6,
        "cuda": False,
    }
    for key, value in expected.items():
        if surface[key] != value:
            problems.append(f"patched Pong surface {key}={surface[key]!r}, expected {value!r}")
    caps = {
        "max_env_step": 8,
        "collector_env_num": 1,
        "evaluator_env_num": 1,
        "num_simulations": 2,
        "batch_size": 8,
    }
    for key, ceiling in caps.items():
        if int(surface[key]) > ceiling:
            problems.append(f"patched Pong cap {key}={surface[key]!r} exceeds {ceiling}")
    optional_caps = {
        "n_evaluator_episode": 1,
        "n_episode": 1,
        "update_per_collect": 1,
    }
    for key, ceiling in optional_caps.items():
        if surface[key] is not None and int(surface[key]) > ceiling:
            problems.append(f"patched Pong optional cap {key}={surface[key]!r} exceeds {ceiling}")
    return problems


def _run_lightzero_pong_dry_config_smoke(
    *,
    env_id: str,
    seed: int,
    max_env_step: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    problems: list[str] = []
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "easydict": _version_or_missing("easydict"),
    }

    patched = _patched_pong_configs(
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
    )
    patched_surface = patched["patched_surface"]
    problems.extend(_validate_patched_surface(patched_surface))

    captured = patched["captured"]
    if captured["default_env_action_space_size"] != patched_surface["action_space_size"]:
        problems.append(
            "Pong action map size "
            f"{captured['default_env_action_space_size']!r} does not match config "
            f"action_space_size {patched_surface['action_space_size']!r}"
        )
    missing_tokens = [
        token
        for token, present in captured["source_tokens_present"].items()
        if not present
    ]
    if missing_tokens:
        problems.append(f"Pong config main() source missing expected tokens: {missing_tokens}")

    entry_module = importlib.import_module("lzero.entry")
    train_muzero_segment = entry_module.train_muzero_segment

    result = {
        "ok": not problems,
        "label": "stock LightZero Atari Pong MuZero dry config smoke",
        "mode": "dry",
        "call_policy": "dry_config_patch_only_trainer_entrypoint_monkeypatched",
        "problems": problems,
        "packages": packages,
        "stock_example": {
            "task": env_id,
            "algorithm": "MuZero",
            "module": patched["module"],
            "trainer_entrypoint": "lzero.entry.train_muzero_segment",
            "trainer_signature": str(inspect.signature(train_muzero_segment)),
            "captured": captured,
            "original_surface": patched["original_surface"],
            "patched_surface": patched_surface,
            "patches": patched["patches"],
            "trainer_args": {
                "seed": seed,
                "max_env_step": max_env_step,
            },
        },
        "train_result": None,
        "note": (
            "Dry config only. This does not instantiate ALE/Gym/EnvPool, does "
            "not require Atari ROMs, and does not call the LightZero Pong "
            "trainer."
        ),
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, timeout=8 * 60)
def lightzero_pong_dry_config_smoke(
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
) -> dict[str, Any]:
    return _run_lightzero_pong_dry_config_smoke(
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
    )


@app.local_entrypoint()
def main(
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
) -> None:
    result = lightzero_pong_dry_config_smoke.remote(
        env_id=env_id,
        seed=seed,
        max_env_step=max_env_step,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
