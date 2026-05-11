"""No-train probes for the debug visual CurvyZero LightZero boundary."""

from __future__ import annotations

import importlib
import json
import time
import traceback
from dataclasses import asdict
from dataclasses import is_dataclass
from importlib import metadata
from typing import Any

import numpy as np

from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_LABEL,
    DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
    DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
    DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
    DEBUG_OCCUPANCY_GRAY64_SHAPE,
    DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
    DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE,
)


CURVYZERO_DEBUG_VISUAL_IMPORT = "curvyzero.training.curvyzero_debug_visual_lightzero_env"
CURVYZERO_DEBUG_VISUAL_ENV_TYPE = "curvyzero_debug_visual_tensor_lightzero"
CURVYZERO_DEBUG_VISUAL_ENV_ID = "CurvyZeroDebugVisualTensorLightZero-v0"


def run_curvyzero_debug_visual_lightzero_runtime_probe(
    *,
    seed: int = 0,
    include_env_factory: bool = True,
    require_installed_runtime: bool = True,
) -> dict[str, Any]:
    """Exercise the debug visual LightZero env without calling a trainer."""

    started = time.perf_counter()
    problems: list[str] = []
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "gym": _version_or_missing("gym"),
        "numpy": _version_or_missing("numpy"),
    }
    imports = _import_status(("lzero", "ding", "torch", "gym", CURVYZERO_DEBUG_VISUAL_IMPORT))
    for module_name, status in imports.items():
        if status != "ok" and (require_installed_runtime or module_name == CURVYZERO_DEBUG_VISUAL_IMPORT):
            problems.append(f"failed to import {module_name}: {status}")

    create_config = build_curvyzero_debug_visual_lightzero_create_config()
    env_cfg = build_curvyzero_debug_visual_lightzero_env_cfg(seed=seed)
    model_config = build_curvyzero_debug_visual_lightzero_model_config()
    identity = _identity_report(
        create_config=create_config,
        env_cfg=env_cfg,
        model_config=model_config,
    )
    if not identity["ok"]:
        problems.extend(identity["problems"])

    factory_probe = (
        _exercise_env_factory(seed=seed, require_installed_runtime=require_installed_runtime)
        if include_env_factory
        else {"ok": True, "skipped": True, "reason": "include_env_factory=false"}
    )
    direct_probe = _exercise_direct_env(
        seed=seed,
        require_installed_runtime=require_installed_runtime,
    )
    if not factory_probe["ok"]:
        problems.append("DI-engine env factory did not create/reset/step debug visual env")
    if not direct_probe["ok"]:
        problems.append("direct CurvyZeroDebugVisualLightZeroEnv reset/step failed")

    result = {
        "ok": not problems,
        "label": "CurvyZero debug visual LightZero installed-runtime config/import smoke",
        "mode": "no_train_env_create_reset_step_only",
        "call_policy": "does_not_train; does_not_call_lzero_entrypoints",
        "problems": problems,
        "packages": packages,
        "imports": imports,
        "config_surface": {
            "create_config": _to_plain(create_config),
            "env_cfg": _to_plain(env_cfg),
            "model_config": _to_plain(model_config),
            "expected_env_type": CURVYZERO_DEBUG_VISUAL_ENV_TYPE,
            "expected_env_id": CURVYZERO_DEBUG_VISUAL_ENV_ID,
            "simulator_kind": "project_owned_curvytron_source_env",
            "lightzero_wrapper_kind": "curvyzero_debug_visual_tensor_lightzero_env",
            "emulator": "none",
            "ale_usage": "none",
            "observation_schema_id": DEBUG_OCCUPANCY_GRAY64_LABEL,
            "observation_schema_hash": DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
            "renderer_impl_id": DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
            "surface": "debug_visual_tensor",
            "truth_level": "debug_non_fidelity",
            "source_fidelity_level": "none",
            "source_backed_observation_fidelity": False,
            "uses_ale": False,
            "shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
            "dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
            "range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
            "env_raw_frame_shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
            "model_observation_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
            "frame_stack": 4,
            "frame_stack_owner": "optimizer",
            "action_space_n": 3,
            "model_kind": "conv",
        },
        "identity": identity,
        "env_factory": factory_probe,
        "direct_env": direct_probe,
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    return _to_plain(result)


def build_curvyzero_debug_visual_lightzero_create_config() -> Any:
    """Return the tiny DI-engine create_config surface for this custom env."""

    return _maybe_easydict(
        {
            "env": {
                "type": CURVYZERO_DEBUG_VISUAL_ENV_TYPE,
                "import_names": [CURVYZERO_DEBUG_VISUAL_IMPORT],
            },
            "env_manager": {"type": "base"},
            "policy": {
                "type": "muzero",
                "import_names": ["lzero.policy.muzero"],
            },
        }
    )


def build_curvyzero_debug_visual_lightzero_env_cfg(*, seed: int) -> Any:
    return _maybe_easydict(
        {
            "type": CURVYZERO_DEBUG_VISUAL_ENV_TYPE,
            "import_names": [CURVYZERO_DEBUG_VISUAL_IMPORT],
            "env_id": CURVYZERO_DEBUG_VISUAL_ENV_ID,
            "collector_env_num": 1,
            "evaluator_env_num": 1,
            "n_evaluator_episode": 1,
            "seed": int(seed),
            "dynamic_seed": False,
            "ego_player_id": "player_0",
            "opponent_action_id": 1,
            "source_step_ms": 1000.0 / 60.0,
            "source_max_steps": 2000,
        }
    )


def build_curvyzero_debug_visual_lightzero_model_config() -> Any:
    return _maybe_easydict(
        {
            "model": {
                "observation_shape": DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
                "action_space_size": 3,
                "model_type": "conv",
                "frame_stack_num": 4,
            },
            "collector": {
                "env_raw_frame_shape": DEBUG_OCCUPANCY_GRAY64_SHAPE,
                "frame_stack_owner": "optimizer",
            },
        }
    )


def _exercise_env_factory(
    *,
    seed: int,
    require_installed_runtime: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        _import_curvyzero_debug_visual_env_module()
        from ding.envs import BaseEnvTimestep
        from ding.envs import get_vec_env_setting

        env_cfg = build_curvyzero_debug_visual_lightzero_env_cfg(seed=seed)
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(env_cfg)
        one_env_cfg = collector_env_cfg[0] if collector_env_cfg else env_cfg
        try:
            env = env_fn(cfg=one_env_cfg)
        except TypeError:
            env = env_fn(one_env_cfg)
        exercise = _exercise_env(
            env,
            seed=seed,
            base_env_timestep_cls=BaseEnvTimestep,
            require_real_base_timestep=require_installed_runtime,
        )
        problems = list(exercise["problems"])
        if not exercise["identity"]["ok"]:
            problems.extend(exercise["identity"]["problems"])
        return {
            "ok": exercise["ok"] and not problems,
            "problems": problems,
            "collector_env_cfg_count": len(collector_env_cfg),
            "evaluator_env_cfg_count": len(evaluator_env_cfg),
            "exercise": exercise,
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:  # pragma: no cover - installed-runtime diagnosis.
        result = {"ok": False, "elapsed_sec": round(time.perf_counter() - started, 6)}
        result.update(_exception_result(exc))
        return result


def _exercise_direct_env(
    *,
    seed: int,
    require_installed_runtime: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        module = _import_curvyzero_debug_visual_env_module()
        base_env_timestep_cls = _optional_ding_base_env_timestep_cls()
        env = module.CurvyZeroDebugVisualLightZeroEnv(
            build_curvyzero_debug_visual_lightzero_env_cfg(seed=seed)
        )
        exercise = _exercise_env(
            env,
            seed=seed,
            base_env_timestep_cls=base_env_timestep_cls,
            require_real_base_timestep=require_installed_runtime,
        )
        return {
            "ok": exercise["ok"],
            "problems": exercise["problems"],
            "exercise": exercise,
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:
        result = {"ok": False, "elapsed_sec": round(time.perf_counter() - started, 6)}
        result.update(_exception_result(exc))
        return result


def _exercise_env(
    env: Any,
    *,
    seed: int,
    base_env_timestep_cls: Any | None,
    require_real_base_timestep: bool,
) -> dict[str, Any]:
    problems: list[str] = []
    identity = _env_identity_report(env)
    if not identity["ok"]:
        problems.extend(identity["problems"])
    seed_result = _seed_env(env, seed)
    reset_obs = _reset_env(env, seed)
    reset_summary = _summarize_lightzero_observation(reset_obs)
    problems.extend(_validate_lightzero_observation(reset_obs, label="reset"))
    step_timestep = env.step(np.int64(1))
    step_summary = _summarize_timestep(
        step_timestep,
        base_env_timestep_cls=base_env_timestep_cls,
    )
    problems.extend(
        _validate_timestep(
            step_timestep,
            label="step",
            base_env_timestep_cls=base_env_timestep_cls,
            require_real_base_timestep=require_real_base_timestep,
        )
    )
    close_result = _close_env(env)
    return {
        "ok": not problems,
        "problems": problems,
        "identity": identity,
        "seed": seed_result,
        "spaces": {
            "observation_space": _summarize_value(getattr(env, "observation_space", None)),
            "action_space": _summarize_value(getattr(env, "action_space", None)),
            "reward_space": _summarize_value(getattr(env, "reward_space", None)),
            "legal_actions": _summarize_value(getattr(env, "legal_actions", None)),
        },
        "reset": reset_summary,
        "step": step_summary,
        "close": close_result,
    }


def _seed_env(env: Any, seed: int) -> dict[str, Any]:
    if not hasattr(env, "seed"):
        return {"called": False, "reason": "env has no seed method"}
    try:
        env.seed(seed, dynamic_seed=False)
        return {"called": True, "args": [int(seed)], "kwargs": {"dynamic_seed": False}, "ok": True}
    except TypeError:
        env.seed(seed)
        return {"called": True, "args": [int(seed)], "kwargs": {}, "ok": True}


def _reset_env(env: Any, seed: int) -> Any:
    try:
        return env.reset(seed=seed)
    except TypeError:
        return env.reset()


def _close_env(env: Any) -> dict[str, Any]:
    if not hasattr(env, "close"):
        return {"called": False, "reason": "env has no close method"}
    env.close()
    return {"called": True, "ok": True}


def _validate_lightzero_observation(obs: Any, *, label: str) -> list[str]:
    problems: list[str] = []
    if not isinstance(obs, dict):
        return [f"{label} observation is {type(obs).__name__}, expected dict"]
    if set(obs) != {"observation", "action_mask", "to_play", "timestep"}:
        problems.append(f"{label} observation keys were {sorted(obs)}, expected LightZero keys")
    observation = np.asarray(obs.get("observation"))
    if tuple(observation.shape) != DEBUG_OCCUPANCY_GRAY64_SHAPE:
        problems.append(
            f"{label} observation shape {tuple(observation.shape)!r}, "
            f"expected {DEBUG_OCCUPANCY_GRAY64_SHAPE!r}"
        )
    if observation.dtype != np.float32:
        problems.append(f"{label} observation dtype {observation.dtype}, expected float32")
    if observation.size and (float(observation.min()) < 0.0 or float(observation.max()) > 1.0):
        problems.append(f"{label} observation range escaped [0, 1]")
    action_mask = np.asarray(obs.get("action_mask"))
    if tuple(action_mask.shape) != (3,):
        problems.append(f"{label} action_mask shape {tuple(action_mask.shape)!r}, expected (3,)")
    if action_mask.dtype != np.int8:
        problems.append(f"{label} action_mask dtype {action_mask.dtype}, expected int8")
    if obs.get("to_play") != -1:
        problems.append(f"{label} to_play {obs.get('to_play')!r}, expected -1")
    return problems


def _validate_timestep(
    timestep: Any,
    *,
    label: str,
    base_env_timestep_cls: Any | None,
    require_real_base_timestep: bool,
) -> list[str]:
    problems: list[str] = []
    for attr in ("obs", "reward", "done", "info"):
        if not hasattr(timestep, attr):
            problems.append(f"{label} timestep missing {attr}")
    if problems:
        return problems
    problems.extend(_validate_lightzero_observation(timestep.obs, label=f"{label}.obs"))
    if not isinstance(timestep.info, dict):
        problems.append(f"{label}.info is {type(timestep.info).__name__}, expected dict")
    for key, expected in {
        "surface": "debug_visual_tensor",
        "observation_schema_id": DEBUG_OCCUPANCY_GRAY64_LABEL,
        "observation_schema_hash": DEBUG_OCCUPANCY_GRAY64_SCHEMA_HASH,
        "truth_level": "debug_non_fidelity",
        "source_fidelity_level": "none",
        "source_backed_observation_fidelity": False,
        "renderer_impl_id": DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
        "shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
        "dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
        "range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
        "uses_ale": False,
        "ale_usage": "none",
        "frame_stack_owner": "optimizer",
    }.items():
        if timestep.info.get(key) != expected:
            problems.append(f"{label}.info[{key!r}]={timestep.info.get(key)!r}, expected {expected!r}")
    if base_env_timestep_cls is None:
        if require_real_base_timestep:
            problems.append("ding.envs.BaseEnvTimestep was not importable")
    elif not isinstance(timestep, base_env_timestep_cls):
        problems.append(
            f"{label} timestep type {type(timestep).__name__} is not ding.envs.BaseEnvTimestep"
        )
    return problems


def _summarize_timestep(
    timestep: Any,
    *,
    base_env_timestep_cls: Any | None,
) -> dict[str, Any]:
    info = getattr(timestep, "info", None)
    return {
        "type": type(timestep).__module__ + "." + type(timestep).__name__,
        "base_env_timestep_like": all(
            hasattr(timestep, attr) for attr in ("obs", "reward", "done", "info")
        ),
        "real_ding_base_env_timestep": bool(
            base_env_timestep_cls is not None and isinstance(timestep, base_env_timestep_cls)
        ),
        "obs": _summarize_lightzero_observation(getattr(timestep, "obs", None)),
        "reward": _to_plain(getattr(timestep, "reward", None)),
        "done": bool(getattr(timestep, "done", False)),
        "info": _summarize_info(info),
    }


def _summarize_lightzero_observation(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {"type": type(obs).__name__, "repr": repr(obs)}
    return {
        "type": "dict",
        "keys": sorted(str(key) for key in obs),
        "observation": _summarize_value(obs.get("observation")),
        "observation_min": float(np.asarray(obs.get("observation")).min()),
        "observation_max": float(np.asarray(obs.get("observation")).max()),
        "action_mask": _summarize_value(obs.get("action_mask")),
        "action_mask_values": _to_plain(obs.get("action_mask")),
        "to_play": _to_plain(obs.get("to_play")),
        "timestep": _to_plain(obs.get("timestep")),
    }


def _summarize_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {"type": type(info).__name__, "repr": repr(info)}
    selected_keys = (
        "surface",
        "observation_schema_id",
        "observation_schema_hash",
        "schema_hash",
        "truth_level",
        "source_fidelity_level",
        "source_backed_observation_fidelity",
        "renderer_impl_id",
        "shape",
        "dtype",
        "range",
        "value_range",
        "uses_ale",
        "ale_usage",
        "frame_stack_owner",
        "adapter_timestep",
        "source_timer_clock_advances_on_step",
        "source_at_ms",
        "joint_action",
        "joint_source_move",
        "opponent_policy_id",
        "done",
        "terminated",
        "truncated",
        "trace_hash",
    )
    return {
        "type": "dict",
        "keys": sorted(str(key) for key in info),
        "selected": {key: _to_plain(info.get(key)) for key in selected_keys if key in info},
        "final_observation": _summarize_lightzero_observation(info.get("final_observation"))
        if "final_observation" in info and info.get("final_observation") is not None
        else None,
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
        summary["keys"] = sorted(str(key) for key in value)
    elif isinstance(value, (list, tuple)):
        summary["len"] = len(value)
        summary["value"] = _to_plain(value)
    else:
        plain = _to_plain(value)
        text = repr(plain)
        summary["repr"] = text if len(text) <= 240 else text[:237] + "..."
    return summary


def _identity_report(
    *,
    create_config: Any,
    env_cfg: Any,
    model_config: Any,
) -> dict[str, Any]:
    problems: list[str] = []
    env_type = _cfg_get(_cfg_get(create_config, "env", {}), "type", None)
    env_import_names = tuple(_cfg_get(_cfg_get(create_config, "env", {}), "import_names", ()))
    env_cfg_type = _cfg_get(env_cfg, "type", None)
    env_id = _cfg_get(env_cfg, "env_id", None)
    model = _cfg_get(model_config, "model", {})
    model_shape = tuple(_cfg_get(model, "observation_shape", ()))
    frame_stack = _cfg_get(model, "frame_stack_num", None)
    model_type = _cfg_get(model, "model_type", None)
    action_space = _cfg_get(model, "action_space_size", None)
    if env_type != CURVYZERO_DEBUG_VISUAL_ENV_TYPE:
        problems.append(f"create_config.env.type={env_type!r}, expected {CURVYZERO_DEBUG_VISUAL_ENV_TYPE!r}")
    if env_cfg_type != CURVYZERO_DEBUG_VISUAL_ENV_TYPE:
        problems.append(f"env_cfg.type={env_cfg_type!r}, expected {CURVYZERO_DEBUG_VISUAL_ENV_TYPE!r}")
    if env_id != CURVYZERO_DEBUG_VISUAL_ENV_ID:
        problems.append(f"env_cfg.env_id={env_id!r}, expected {CURVYZERO_DEBUG_VISUAL_ENV_ID!r}")
    if CURVYZERO_DEBUG_VISUAL_IMPORT not in env_import_names:
        problems.append("create_config.env.import_names does not import debug visual env")
    if model_shape != DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE:
        problems.append(f"model observation_shape={model_shape!r}, expected {DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE!r}")
    if frame_stack != 4:
        problems.append(f"model frame_stack_num={frame_stack!r}, expected 4")
    if action_space != 3:
        problems.append(f"model action_space_size={action_space!r}, expected 3")
    if model_type != "conv":
        problems.append(f"model_type={model_type!r}, expected 'conv'")
    disallowed = _disallowed_identity_hits([env_type, env_cfg_type, env_id, *env_import_names])
    if disallowed:
        problems.append(f"config identity contains disallowed external markers: {disallowed}")
    return {
        "ok": not problems,
        "problems": problems,
        "create_config_env_type": env_type,
        "create_config_env_import_names": list(env_import_names),
        "env_cfg_type": env_cfg_type,
        "env_cfg_env_id": env_id,
        "model_observation_shape": list(model_shape),
        "frame_stack_num": frame_stack,
        "action_space_size": action_space,
        "model_type": model_type,
        "identity_clean": not disallowed,
        "disallowed_identity_hits": disallowed,
    }


def _env_identity_report(env: Any) -> dict[str, Any]:
    env_class = type(env).__module__ + "." + type(env).__name__
    env_id = getattr(env, "env_id", None)
    env_type = getattr(env, "lightzero_env_type", None)
    text_values = [env_class, env_id, env_type, repr(env)]
    disallowed = _disallowed_identity_hits(text_values)
    problems: list[str] = []
    if env_class != (
        "curvyzero.training.curvyzero_debug_visual_lightzero_env."
        "CurvyZeroDebugVisualLightZeroEnv"
    ):
        problems.append(f"env class was {env_class!r}, expected CurvyZeroDebugVisualLightZeroEnv")
    if env_id != CURVYZERO_DEBUG_VISUAL_ENV_ID:
        problems.append(f"env.env_id={env_id!r}, expected {CURVYZERO_DEBUG_VISUAL_ENV_ID!r}")
    if env_type != CURVYZERO_DEBUG_VISUAL_ENV_TYPE:
        problems.append(
            f"env.lightzero_env_type={env_type!r}, expected {CURVYZERO_DEBUG_VISUAL_ENV_TYPE!r}"
        )
    if disallowed:
        problems.append(f"env identity contains disallowed external markers: {disallowed}")
    return {
        "ok": not problems,
        "problems": problems,
        "env_class": env_class,
        "env_id": env_id,
        "lightzero_env_type": env_type,
        "repr": repr(env),
        "identity_clean": not disallowed,
        "disallowed_identity_hits": disallowed,
    }


def _disallowed_identity_hits(values: list[Any]) -> list[str]:
    hits: list[str] = []
    for value in values:
        text = str(value).lower()
        for token in ("at" "ari", "a" "le", "po" "ng"):
            if token in text:
                hits.append(f"{token}:{value}")
    return hits


def _optional_ding_base_env_timestep_cls() -> Any | None:
    try:
        from ding.envs import BaseEnvTimestep
    except ImportError:
        return None
    return BaseEnvTimestep


def _import_curvyzero_debug_visual_env_module() -> Any:
    return importlib.import_module(CURVYZERO_DEBUG_VISUAL_IMPORT)


def _import_status(module_names: tuple[str, ...]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            statuses[module_name] = "ok"
        except Exception as exc:  # pragma: no cover - dependency diagnosis.
            statuses[module_name] = f"{type(exc).__name__}: {exc}"
    return statuses


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


def _maybe_easydict(value: dict[str, Any]) -> Any:
    try:
        from easydict import EasyDict
    except Exception:
        return value
    return EasyDict(value)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    print(
        json.dumps(
            run_curvyzero_debug_visual_lightzero_runtime_probe(
                include_env_factory=False,
                require_installed_runtime=False,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
