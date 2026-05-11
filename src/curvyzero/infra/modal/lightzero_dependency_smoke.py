"""Contained Modal dependency smoke for stock LightZero MuZero examples.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dependency_smoke

This is not a trainer. It only verifies that Modal can build a CPU image with
LightZero, import the stock CartPole/Pong MuZero config modules, and inspect the
cheap config surface while monkeypatching the Pong training entrypoint to a
no-op capture function.
"""

from __future__ import annotations

import importlib
import inspect
import json
import time
from importlib import metadata
from typing import Any

import modal

APP_NAME = "curvyzero-lightzero-dependency-smoke"
LIGHTZERO_VERSION = "0.2.0"

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


def _extract_cartpole_config() -> dict[str, Any]:
    module_name = "zoo.classic_control.cartpole.config.cartpole_muzero_config"
    module = importlib.import_module(module_name)
    main_config = _to_plain(module.main_config)
    create_config = _to_plain(module.create_config)
    return {
        "module": module_name,
        "env_id": main_config["env"]["env_id"],
        "policy_type": create_config["policy"]["type"],
        "env_type": create_config["env"]["type"],
        "model_type": main_config["policy"]["model"]["model_type"],
        "observation_shape": main_config["policy"]["model"]["observation_shape"],
        "action_space_size": main_config["policy"]["model"]["action_space_size"],
        "collector_env_num": main_config["env"]["collector_env_num"],
        "evaluator_env_num": main_config["env"]["evaluator_env_num"],
        "num_simulations": main_config["policy"]["num_simulations"],
        "batch_size": main_config["policy"]["batch_size"],
        "cuda": main_config["policy"]["cuda"],
        "max_env_step": int(getattr(module, "max_env_step")),
    }


def _extract_pong_config_surface() -> dict[str, Any]:
    module_name = "zoo.atari.config.atari_muzero_segment_config"
    module = importlib.import_module(module_name)
    action_map_module = importlib.import_module("zoo.atari.config.atari_env_action_space_map")
    action_map = getattr(action_map_module, "atari_env_action_space_map")
    main_source = inspect.getsource(module.main)
    expected_main_tokens = [
        "train_muzero_segment",
        "num_simulations = 50",
        "batch_size = 256",
        "max_env_step = int(5e5)",
    ]
    captured: dict[str, Any] = {}

    entry_module = importlib.import_module("lzero.entry")
    original_train_muzero_segment = entry_module.train_muzero_segment

    def capture_train_muzero_segment(configs, *, seed: int, max_env_step: int, **kwargs):
        captured["configs"] = configs
        captured["seed"] = seed
        captured["max_env_step"] = max_env_step
        captured["kwargs"] = kwargs
        return {"captured": True}

    entry_module.train_muzero_segment = capture_train_muzero_segment
    try:
        module.main("PongNoFrameskip-v4", 0)
    finally:
        entry_module.train_muzero_segment = original_train_muzero_segment

    main_config, create_config = [_to_plain(config) for config in captured["configs"]]
    return {
        "module": module_name,
        "call_policy": "trainer_entrypoint_monkeypatched_to_capture_config",
        "main_callable": callable(getattr(module, "main", None)),
        "default_env": "PongNoFrameskip-v4",
        "default_env_action_space_size": action_map["PongNoFrameskip-v4"],
        "captured_seed": captured["seed"],
        "captured_max_env_step": captured["max_env_step"],
        "env_id": main_config["env"]["env_id"],
        "policy_type": create_config["policy"]["type"],
        "env_type": create_config["env"]["type"],
        "model_type": main_config["policy"]["model"]["model_type"],
        "observation_shape": main_config["policy"]["model"]["observation_shape"],
        "action_space_size": main_config["policy"]["model"]["action_space_size"],
        "collector_env_num": main_config["env"]["collector_env_num"],
        "evaluator_env_num": main_config["env"]["evaluator_env_num"],
        "num_simulations": main_config["policy"]["num_simulations"],
        "batch_size": main_config["policy"]["batch_size"],
        "cuda": main_config["policy"]["cuda"],
        "expected_main_tokens_present": {
            token: token in main_source for token in expected_main_tokens
        },
        "source_line_count": len(main_source.splitlines()),
    }


def _run_lightzero_import_config_smoke() -> dict[str, Any]:
    started = time.perf_counter()
    problems: list[str] = []

    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "easydict": _version_or_missing("easydict"),
    }

    imports: dict[str, str] = {}
    for module_name in ("lzero", "ding", "torch", "easydict"):
        try:
            importlib.import_module(module_name)
            imports[module_name] = "ok"
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            imports[module_name] = f"{type(exc).__name__}: {exc}"
            problems.append(f"failed to import {module_name}: {type(exc).__name__}: {exc}")

    cartpole: dict[str, Any] | None = None
    pong: dict[str, Any] | None = None
    try:
        cartpole = _extract_cartpole_config()
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        problems.append(f"failed to inspect CartPole config: {type(exc).__name__}: {exc}")

    try:
        pong = _extract_pong_config_surface()
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        problems.append(f"failed to inspect Atari Pong config surface: {type(exc).__name__}: {exc}")

    if cartpole is not None:
        expected_cartpole = {
            "env_id": "CartPole-v0",
            "policy_type": "muzero",
            "env_type": "cartpole_lightzero",
            "model_type": "mlp",
            "action_space_size": 2,
            "cuda": True,
        }
        for key, expected in expected_cartpole.items():
            if cartpole[key] != expected:
                problems.append(f"CartPole config {key}={cartpole[key]!r}, expected {expected!r}")

    if pong is not None:
        expected_pong = {
            "env_id": "PongNoFrameskip-v4",
            "policy_type": "muzero",
            "env_type": "atari_lightzero",
            "model_type": "conv",
            "action_space_size": 6,
            "captured_seed": 0,
            "captured_max_env_step": 500000,
            "num_simulations": 50,
            "batch_size": 256,
            "cuda": True,
        }
        for key, expected in expected_pong.items():
            if pong[key] != expected:
                problems.append(f"Pong config {key}={pong[key]!r}, expected {expected!r}")
        if pong["default_env_action_space_size"] != pong["action_space_size"]:
            problems.append(
                "Pong action map size "
                f"{pong['default_env_action_space_size']!r} does not match config "
                f"action_space_size {pong['action_space_size']!r}"
            )
        missing_tokens = [
            token
            for token, present in pong["expected_main_tokens_present"].items()
            if not present
        ]
        if missing_tokens:
            problems.append(f"Pong config main() source missing expected tokens: {missing_tokens}")

    result = {
        "ok": not problems,
        "problems": problems,
        "packages": packages,
        "imports": imports,
        "stock_examples": {
            "cartpole": cartpole,
            "pong": pong,
        },
        "note": (
            "This smoke does not call train_muzero or train_muzero_segment. "
            "CartPole is the smallest stock MuZero replication path; Pong is "
            "the later visual reference."
        ),
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.function(image=image, timeout=10 * 60)
def lightzero_import_config_smoke() -> dict[str, Any]:
    return _run_lightzero_import_config_smoke()


@app.local_entrypoint()
def main() -> None:
    result = lightzero_import_config_smoke.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
