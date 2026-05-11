"""Stock LightZero Atari Pong environment creation smoke.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_pong_env_smoke

This is intentionally not a trainer. It imports the stock LightZero Atari Pong
MuZero config, patches the config down to one tiny CPU env surface, then tries
to create, reset, and step the Pong environment through LightZero/DI-engine
paths. Missing ALE/Gym/ROM/EnvPool dependencies are returned as structured
diagnostics instead of hidden behind a trainer failure.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import time
import traceback
from importlib import metadata
from typing import Any, Callable

import modal

from curvyzero.infra.modal.lightzero_atari_rom_image import (
    ATARI_ROM_LICENSE_NOTICE,
    build_lightzero_atari_rom_image,
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
    _patched_pong_configs,
    _to_plain,
    _validate_patched_surface,
)

APP_NAME = "curvyzero-lightzero-pong-env-smoke"

image = build_lightzero_atari_rom_image(lightzero_version=LIGHTZERO_VERSION)

app = modal.App(APP_NAME)


def _version_or_missing(*packages: str) -> str:
    for package in packages:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return "missing"


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _summarize_value(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        summary["shape"] = [int(item) for item in shape]
    if dtype is not None:
        summary["dtype"] = str(dtype)
    if isinstance(value, dict):
        summary["keys"] = [str(key) for key in list(value.keys())[:10]]
        summary["size"] = len(value)
    elif isinstance(value, (list, tuple)):
        summary["len"] = len(value)
        if value:
            summary["first"] = _summarize_value(value[0])
    else:
        text = repr(value)
        summary["repr"] = text if len(text) <= 200 else text[:197] + "..."
    return summary


def _import_status(module_names: tuple[str, ...]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            statuses[module_name] = "ok"
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            statuses[module_name] = f"{type(exc).__name__}: {exc}"
    return statuses


def _merge_env_cfg(main_env_cfg: Any, create_env_cfg: Any) -> Any:
    try:
        from easydict import EasyDict
    except Exception:  # pragma: no cover - LightZero dependency diagnosis.
        EasyDict = dict  # type: ignore[assignment]

    merged = EasyDict(copy.deepcopy(_to_plain(main_env_cfg)))
    for key, value in _to_plain(create_env_cfg).items():
        if key == "type" or key not in merged:
            merged[key] = value
    return merged


def _try_call(name: str, call: Callable[[], Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        value = call()
        return {
            "name": name,
            "ok": True,
            "elapsed_sec": round(time.perf_counter() - started, 6),
            "value": value,
        }
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        result = {
            "name": name,
            "ok": False,
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
        result.update(_exception_result(exc))
        return result


def _seed_env(env: Any, seed: int) -> dict[str, Any]:
    if not hasattr(env, "seed"):
        return {"called": False, "reason": "env has no seed method"}
    for args, kwargs in (
        ((seed,), {}),
        ((seed,), {"dynamic_seed": False}),
        ((), {"seed": seed}),
    ):
        try:
            env.seed(*args, **kwargs)
            return {"called": True, "args": list(args), "kwargs": kwargs, "ok": True}
        except TypeError:
            continue
    try:
        env.seed(seed)
        return {"called": True, "args": [seed], "kwargs": {}, "ok": True}
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        result = {"called": True, "ok": False}
        result.update(_exception_result(exc))
        return result


def _reset_env(env: Any, seed: int) -> tuple[Any, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for name, call in (
        ("reset()", lambda: env.reset()),
        ("reset(seed=seed)", lambda: env.reset(seed=seed)),
    ):
        result = _try_call(name, call)
        attempts.append({key: value for key, value in result.items() if key != "value"})
        if result["ok"]:
            return result["value"], {
                "ok": True,
                "attempt": name,
                "observation": _summarize_value(result["value"]),
                "attempts": attempts,
            }
    return None, {"ok": False, "attempts": attempts}


def _step_env(env: Any) -> tuple[Any, dict[str, Any]]:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - LightZero pulls numpy in practice.
        np = None  # type: ignore[assignment]

    action_candidates: list[tuple[str, Any]] = [("int_0", 0)]
    action_space = getattr(env, "action_space", None)
    if action_space is not None and hasattr(action_space, "sample"):
        try:
            action_candidates.append(("action_space_sample", action_space.sample()))
        except Exception:
            pass
    if np is not None:
        action_candidates.extend(
            [
                ("np_int64_0", np.int64(0)),
                ("np_array_0", np.array(0)),
            ]
        )

    attempts: list[dict[str, Any]] = []
    for name, action in action_candidates:
        result = _try_call(f"step({name})", lambda action=action: env.step(action))
        attempts.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"value"}
            }
        )
        if result["ok"]:
            return result["value"], {
                "ok": True,
                "attempt": f"step({name})",
                "action_summary": _summarize_value(action),
                "timestep": _summarize_value(result["value"]),
                "attempts": attempts,
            }
    return None, {"ok": False, "attempts": attempts}


def _close_env(env: Any) -> dict[str, Any]:
    if not hasattr(env, "close"):
        return {"called": False, "reason": "env has no close method"}
    result = _try_call("close()", env.close)
    return {key: value for key, value in result.items() if key != "value"}


def _exercise_env(env: Any, *, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "env_type": type(env).__module__ + "." + type(env).__name__,
        "action_space": _summarize_value(getattr(env, "action_space", None)),
        "observation_space": _summarize_value(getattr(env, "observation_space", None)),
    }
    result["seed"] = _seed_env(env, seed)
    _, reset_result = _reset_env(env, seed)
    result["reset"] = reset_result
    if reset_result["ok"]:
        _, step_result = _step_env(env)
    else:
        step_result = {"ok": False, "skipped": True, "reason": "reset failed"}
    result["step"] = step_result
    result["close"] = _close_env(env)
    result["ok"] = bool(reset_result["ok"] and step_result["ok"])
    return result


def _compile_config_candidates(main_config: Any, create_config: Any, seed: int) -> list[dict[str, Any]]:
    from ding.config import compile_config

    candidates: tuple[tuple[str, Callable[[], Any]], ...] = (
        (
            "compile_config(main_config, create_cfg=create_config, auto=True)",
            lambda: compile_config(
                copy.deepcopy(main_config),
                seed=seed,
                auto=True,
                create_cfg=copy.deepcopy(create_config),
                save_cfg=False,
            ),
        ),
        (
            "compile_config([main_config, create_config], auto=True)",
            lambda: compile_config(
                [copy.deepcopy(main_config), copy.deepcopy(create_config)],
                seed=seed,
                auto=True,
                save_cfg=False,
            ),
        ),
    )
    results: list[dict[str, Any]] = []
    for name, call in candidates:
        result = _try_call(name, call)
        if result["ok"]:
            compiled = result.pop("value")
            result["env_cfg_summary"] = {
                "type": getattr(getattr(compiled, "env", None), "type", None),
                "env_id": getattr(getattr(compiled, "env", None), "env_id", None),
                "collector_env_num": getattr(getattr(compiled, "env", None), "collector_env_num", None),
                "evaluator_env_num": getattr(getattr(compiled, "env", None), "evaluator_env_num", None),
            }
            result["compiled_config"] = compiled
        results.append(result)
    return results


def _attempt_ding_env_factory(main_config: Any, create_config: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "name": "ding.envs.get_vec_env_setting direct env",
        "ok": False,
        "compile_config_attempts": [],
        "env_cfg_attempts": [],
    }
    try:
        from ding.envs import get_vec_env_setting
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        result.update(_exception_result(exc))
        result["elapsed_sec"] = round(time.perf_counter() - started, 6)
        return result

    env_cfgs: list[tuple[str, Any]] = []
    try:
        compile_results = _compile_config_candidates(main_config, create_config, seed)
        result["compile_config_attempts"] = [
            {key: value for key, value in item.items() if key != "compiled_config"}
            for item in compile_results
        ]
        for item in compile_results:
            if item["ok"] and "compiled_config" in item:
                env_cfgs.append((item["name"] + ".env", item["compiled_config"].env))
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        result["compile_config_exception"] = _exception_result(exc)

    env_cfgs.extend(
        [
            ("merged main_config.env + create_config.env", _merge_env_cfg(main_config["env"], create_config["env"])),
            ("main_config.env", copy.deepcopy(main_config["env"])),
        ]
    )

    for env_cfg_name, env_cfg in env_cfgs:
        attempt: dict[str, Any] = {
            "env_cfg_name": env_cfg_name,
            "env_cfg_type": getattr(env_cfg, "type", None),
            "env_id": getattr(env_cfg, "env_id", None),
        }
        try:
            env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(env_cfg)
            attempt["get_vec_env_setting"] = {
                "ok": True,
                "env_fn": getattr(env_fn, "__module__", "") + "." + getattr(env_fn, "__name__", repr(env_fn)),
                "collector_env_cfg_count": len(collector_env_cfg),
                "evaluator_env_cfg_count": len(evaluator_env_cfg),
            }
            one_env_cfg = collector_env_cfg[0] if collector_env_cfg else env_cfg
            try:
                env = env_fn(cfg=one_env_cfg)
            except TypeError:
                env = env_fn(one_env_cfg)
            attempt["exercise"] = _exercise_env(env, seed=seed)
            attempt["ok"] = bool(attempt["exercise"]["ok"])
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            attempt["ok"] = False
            attempt.update(_exception_result(exc))
        result["env_cfg_attempts"].append(attempt)
        if attempt["ok"]:
            result["ok"] = True
            break

    result["elapsed_sec"] = round(time.perf_counter() - started, 6)
    return result


def _attempt_direct_lightzero_classes(main_config: Any, create_config: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    merged_cfg = _merge_env_cfg(main_config["env"], create_config["env"])
    candidates = (
        ("zoo.atari.envs.atari_lightzero_env", "AtariLightZeroEnv"),
        ("zoo.atari.envs.atari_env", "AtariEnv"),
        ("dizoo.atari.envs.atari_env", "AtariEnv"),
        ("dizoo.atari.envs.atari_lightzero_env", "AtariLightZeroEnv"),
    )
    attempts: list[dict[str, Any]] = []
    for module_name, class_name in candidates:
        attempt = {"module": module_name, "class": class_name, "ok": False}
        try:
            module = importlib.import_module(module_name)
            env_class = getattr(module, class_name)
            try:
                env = env_class(cfg=copy.deepcopy(merged_cfg))
            except TypeError:
                env = env_class(copy.deepcopy(merged_cfg))
            attempt["exercise"] = _exercise_env(env, seed=seed)
            attempt["ok"] = bool(attempt["exercise"]["ok"])
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            attempt.update(_exception_result(exc))
        attempts.append(attempt)
        if attempt["ok"]:
            break
    return {
        "name": "direct LightZero/DI-engine Atari env class candidates",
        "ok": any(item["ok"] for item in attempts),
        "attempts": attempts,
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }


def _attempt_plain_gym(env_id: str, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    for module_name in ("gym", "gymnasium"):
        attempt: dict[str, Any] = {"module": module_name, "ok": False}
        try:
            gym_module = importlib.import_module(module_name)
            env = gym_module.make(env_id)
            attempt["exercise"] = _exercise_env(env, seed=seed)
            attempt["ok"] = bool(attempt["exercise"]["ok"])
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            attempt.update(_exception_result(exc))
        attempts.append(attempt)
        if attempt["ok"]:
            break
    return {
        "name": "plain gym/gymnasium diagnostic fallback",
        "ok": any(item["ok"] for item in attempts),
        "attempts": attempts,
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }


def _run_lightzero_pong_env_smoke(
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
        "gym": _version_or_missing("gym"),
        "gymnasium": _version_or_missing("gymnasium"),
        "ale-py": _version_or_missing("ale-py", "ale_py"),
        "opencv-python-headless": _version_or_missing("opencv-python-headless"),
        "AutoROM": _version_or_missing("AutoROM"),
        "atari-py": _version_or_missing("atari-py", "atari_py"),
        "envpool": _version_or_missing("envpool"),
    }
    imports = _import_status(
        (
            "lzero",
            "ding",
            "torch",
            "gym",
            "gymnasium",
            "cv2",
            "ale_py",
            "atari_py",
            "envpool",
            "zoo.atari.config.atari_muzero_segment_config",
            "zoo.atari.envs.atari_lightzero_env",
            "zoo.atari.envs.atari_env",
            "dizoo.atari.envs.atari_env",
        )
    )

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

    main_config = patched["main_config"]
    create_config = patched["create_config"]

    entry_module = importlib.import_module("lzero.entry")
    train_muzero_segment = entry_module.train_muzero_segment
    attempts = [
        _attempt_ding_env_factory(main_config, create_config, seed),
        _attempt_direct_lightzero_classes(main_config, create_config, seed),
        _attempt_plain_gym(env_id, seed),
    ]
    lightzero_path_ok = bool(attempts[0]["ok"] or attempts[1]["ok"])
    env_ok = any(attempt["ok"] for attempt in attempts)

    if not lightzero_path_ok:
        problems.append("LightZero/DI-engine Pong env create/reset/step did not complete")
    if not env_ok:
        problems.append("No Pong env create/reset/step path completed")

    result = {
        "ok": not problems,
        "env_ok": env_ok,
        "lightzero_path_ok": lightzero_path_ok,
        "label": "stock LightZero Atari Pong environment creation smoke",
        "mode": "env_create_reset_step_only",
        "call_policy": "does_not_train; exercises env create/reset/step",
        "problems": problems,
        "packages": packages,
        "imports": imports,
        "stock_example": {
            "task": env_id,
            "algorithm": "MuZero",
            "module": patched["module"],
            "trainer_entrypoint": "lzero.entry.train_muzero_segment",
            "trainer_signature": str(inspect.signature(train_muzero_segment)),
            "captured": patched["captured"],
            "original_surface": patched["original_surface"],
            "patched_surface": patched_surface,
            "patches": patched["patches"],
        },
        "env_attempts": attempts,
        "train_result": None,
        "rom_unblocker": {
            "license_acceptance": ATARI_ROM_LICENSE_NOTICE,
            "modal_image_step": [
                'uv_pip_install("AutoROM[accept-rom-license]==0.6.1")',
                'run_commands("AutoROM --accept-license")',
            ],
            "check": (
                "This smoke is the ROM-enabled image check. It must pass "
                "create/reset/step before any stock Pong trainer call."
            ),
        },
        "note": (
            "This smoke intentionally does not call train_muzero_segment. A "
            "failure here is expected to identify missing Atari runtime pieces "
            "such as ALE, ROM discovery, Gym/Gymnasium Atari extras, or "
            "EnvPool. This image installs OpenCV for the stock LightZero Atari "
            "wrappers and explicitly accepts the AutoROM Atari ROM license at "
            "image build time."
        ),
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, timeout=8 * 60)
def lightzero_pong_env_smoke(
    env_id: str = DEFAULT_ENV_ID,
    seed: int = DEFAULT_SEED,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
) -> dict[str, Any]:
    return _run_lightzero_pong_env_smoke(
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
    result = lightzero_pong_env_smoke.remote(
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
