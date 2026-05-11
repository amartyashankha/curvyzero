"""No-train probes for the installed LightZero CurvyZero env boundary."""

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

from curvyzero.env import CurvyTronConfig
from curvyzero.env import trainer_contract as contract


CURVYZERO_LIGHTZERO_IMPORT = "curvyzero.training.curvyzero_lightzero_env"
CURVYZERO_LIGHTZERO_ENV_TYPE = "curvyzero_v0_lightzero"
CURVYZERO_LIGHTZERO_ENV_ID = "CurvyZeroLightZero-v0"


def run_curvyzero_lightzero_runtime_probe(
    *,
    seed: int = 0,
    include_env_factory: bool = True,
    include_terminal: bool = True,
    require_installed_runtime: bool = True,
) -> dict[str, Any]:
    """Exercise the CurvyZero LightZero env without calling a trainer."""

    started = time.perf_counter()
    problems: list[str] = []
    packages = {
        "LightZero": _version_or_missing("LightZero", "lightzero"),
        "DI-engine": _version_or_missing("DI-engine", "ding"),
        "torch": _version_or_missing("torch"),
        "gym": _version_or_missing("gym"),
        "numpy": _version_or_missing("numpy"),
    }
    imports = _import_status(
        (
            "lzero",
            "ding",
            "torch",
            "gym",
            CURVYZERO_LIGHTZERO_IMPORT,
        )
    )
    for module_name, status in imports.items():
        if status != "ok" and (require_installed_runtime or module_name == CURVYZERO_LIGHTZERO_IMPORT):
            problems.append(f"failed to import {module_name}: {status}")

    create_config = build_curvyzero_lightzero_create_config()
    env_cfg = build_curvyzero_lightzero_env_cfg(seed=seed)
    identity = _identity_report(create_config=create_config, env_cfg=env_cfg)
    if not identity["ok"]:
        problems.extend(identity["problems"])

    factory_probe = (
        _exercise_env_factory(seed=seed, require_installed_runtime=require_installed_runtime)
        if include_env_factory
        else {"ok": True, "skipped": True, "reason": "include_env_factory=false"}
    )
    direct_probe = _exercise_direct_env(
        seed=seed,
        env_config=CurvyTronConfig(action_repeat=1),
        require_installed_runtime=require_installed_runtime,
    )
    terminal_probe = (
        _exercise_terminal_env(seed=seed, require_installed_runtime=require_installed_runtime)
        if include_terminal
        else {"ok": True, "skipped": True, "reason": "include_terminal=false"}
    )

    if not factory_probe["ok"]:
        problems.append("DI-engine env factory did not create/reset/step curvyzero_v0_lightzero")
    if not direct_probe["ok"]:
        problems.append("direct CurvyZeroLightZeroEnv reset/step failed")
    if not terminal_probe["ok"]:
        problems.append("tiny terminal CurvyZeroLightZeroEnv probe failed")

    result = {
        "ok": not problems,
        "label": "CurvyZero LightZero installed-runtime config/import smoke",
        "mode": "no_train_env_create_reset_step_only",
        "call_policy": "does_not_train; does_not_call_lzero_entrypoints",
        "problems": problems,
        "packages": packages,
        "imports": imports,
        "config_surface": {
            "create_config": _to_plain(create_config),
            "env_cfg": _to_plain(env_cfg),
            "expected_env_type": CURVYZERO_LIGHTZERO_ENV_TYPE,
            "expected_env_id": CURVYZERO_LIGHTZERO_ENV_ID,
            "simulator_kind": "project_owned_curvytron_simulator",
            "lightzero_wrapper_kind": "curvyzero_state_vector_lightzero_env",
            "emulator": "none",
            "ale_usage": "none; ALE is only for separate stock Atari ROM Pong smokes",
            "atari_usage": "none",
            "observation_shape": list(contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE),
            "action_mask_shape": [len(contract.ACTION_NAMES)],
            "action_mask_dtype": contract.LIGHTZERO_ACTION_MASK_DTYPE,
            "to_play": -1,
        },
        "identity": identity,
        "env_factory": factory_probe,
        "direct_env": direct_probe,
        "terminal_env": terminal_probe,
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    return _to_plain(result)


def build_curvyzero_lightzero_create_config() -> Any:
    """Return the tiny DI-engine create_config surface for this custom env."""

    return _maybe_easydict(
        {
            "env": {
                "type": CURVYZERO_LIGHTZERO_ENV_TYPE,
                "import_names": [CURVYZERO_LIGHTZERO_IMPORT],
            },
            "env_manager": {"type": "base"},
            "policy": {
                "type": "muzero",
                "import_names": ["lzero.policy.muzero"],
            },
        }
    )


def build_curvyzero_lightzero_env_cfg(
    *,
    seed: int,
    env_config: CurvyTronConfig | None = None,
) -> Any:
    cfg: dict[str, Any] = {
        "type": CURVYZERO_LIGHTZERO_ENV_TYPE,
        "import_names": [CURVYZERO_LIGHTZERO_IMPORT],
        "env_id": CURVYZERO_LIGHTZERO_ENV_ID,
        "collector_env_num": 1,
        "evaluator_env_num": 1,
        "n_evaluator_episode": 1,
        "seed": int(seed),
        "dynamic_seed": False,
        "ego_player_id": "player_0",
        "opponent_action_id": 1,
    }
    if env_config is not None:
        cfg["env_config"] = env_config
    return _maybe_easydict(cfg)


def _exercise_env_factory(
    *,
    seed: int,
    require_installed_runtime: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        _import_curvyzero_lightzero_env_module()
        from ding.envs import BaseEnvTimestep
        from ding.envs import get_vec_env_setting

        env_cfg = build_curvyzero_lightzero_env_cfg(seed=seed)
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
            "env_fn": getattr(env_fn, "__module__", "") + "." + getattr(env_fn, "__name__", repr(env_fn)),
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
    env_config: CurvyTronConfig,
    require_installed_runtime: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        module = _import_curvyzero_lightzero_env_module()
        base_env_timestep_cls = _optional_ding_base_env_timestep_cls()
        env = module.CurvyZeroLightZeroEnv(
            build_curvyzero_lightzero_env_cfg(seed=seed, env_config=env_config)
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


def _exercise_terminal_env(
    *,
    seed: int,
    require_installed_runtime: bool,
) -> dict[str, Any]:
    terminal_config = CurvyTronConfig(
        width=10,
        height=30,
        max_ticks=100,
        action_repeat=1,
        speed=9.1,
        turn_rate_radians=float(np.pi / 2),
        spawn_margin=1.0,
    )
    result = _exercise_direct_env(
        seed=seed,
        env_config=terminal_config,
        require_installed_runtime=require_installed_runtime,
    )
    if not result["ok"]:
        return result
    exercise = result["exercise"]
    timestep = exercise["step"]
    terminal_info = timestep.get("info", {})
    terminal_info_keys = set(terminal_info.get("keys", ()))
    terminal_selected = terminal_info.get("selected", {})
    missing_terminal_keys = [
        key for key in contract.LIGHTZERO_TERMINAL_INFO_KEYS if key not in terminal_info_keys
    ]
    problems = list(result["problems"])
    if timestep.get("done") is not True:
        problems.append("terminal config did not finish in one step")
    if terminal_selected.get("terminated") is not True:
        problems.append("terminal config did not report terminated=true")
    if terminal_selected.get("truncated") is not False:
        problems.append("terminal config unexpectedly reported truncated=true")
    if missing_terminal_keys:
        problems.append(f"terminal info missing keys: {missing_terminal_keys}")
    return {
        **result,
        "ok": not problems,
        "problems": problems,
        "terminal_checks": {
            "done": timestep.get("done"),
            "terminated": terminal_selected.get("terminated"),
            "truncated": terminal_selected.get("truncated"),
            "terminal_reason": terminal_selected.get("terminal_reason"),
            "winner_ids": terminal_selected.get("winner_ids"),
            "loser_ids": terminal_selected.get("loser_ids"),
            "missing_terminal_info_keys": missing_terminal_keys,
            "final_observation": terminal_info.get("final_observation"),
            "final_reward_map": terminal_info.get("final_reward_map"),
            "eval_episode_return": terminal_selected.get("eval_episode_return"),
        },
    }


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
    if tuple(observation.shape) != contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE:
        problems.append(
            f"{label} observation shape {tuple(observation.shape)!r}, "
            f"expected {contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE!r}"
        )
    if observation.dtype != np.float32:
        problems.append(f"{label} observation dtype {observation.dtype}, expected float32")

    action_mask = np.asarray(obs.get("action_mask"))
    if tuple(action_mask.shape) != (len(contract.ACTION_NAMES),):
        problems.append(
            f"{label} action_mask shape {tuple(action_mask.shape)!r}, "
            f"expected {(len(contract.ACTION_NAMES),)!r}"
        )
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
    if not isinstance(bool(timestep.done), bool):
        problems.append(f"{label}.done could not be interpreted as bool")
    try:
        float(timestep.reward)
    except (TypeError, ValueError):
        problems.append(f"{label}.reward could not be interpreted as float")

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
    base_like = all(hasattr(timestep, attr) for attr in ("obs", "reward", "done", "info"))
    real_base = bool(base_env_timestep_cls is not None and isinstance(timestep, base_env_timestep_cls))
    return {
        "type": type(timestep).__module__ + "." + type(timestep).__name__,
        "base_env_timestep_like": base_like,
        "real_ding_base_env_timestep": real_base,
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
        "action_mask": _summarize_value(obs.get("action_mask")),
        "action_mask_values": _to_plain(obs.get("action_mask")),
        "to_play": _to_plain(obs.get("to_play")),
        "timestep": _to_plain(obs.get("timestep")),
    }


def _summarize_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {"type": type(info).__name__, "repr": repr(info)}
    selected_keys = (
        "ego_player_id",
        "opponent_player_id",
        "joint_action",
        "opponent_action_id",
        "opponent_policy_id",
        "terminal_reason",
        "winner_ids",
        "loser_ids",
        "done",
        "terminated",
        "truncated",
        "timeout",
        "truncation_reason",
        "needs_reset",
        "eval_episode_return",
        "trace_hash",
    )
    return {
        "type": "dict",
        "keys": sorted(str(key) for key in info),
        "selected": {key: _to_plain(info.get(key)) for key in selected_keys if key in info},
        "final_observation": _summarize_lightzero_observation(info.get("final_observation"))
        if "final_observation" in info
        else None,
        "final_reward_map": _to_plain(info.get("final_reward_map"))
        if "final_reward_map" in info
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


def _identity_report(*, create_config: Any, env_cfg: Any) -> dict[str, Any]:
    problems: list[str] = []
    env_type = _cfg_get(_cfg_get(create_config, "env", {}), "type", None)
    env_import_names = tuple(_cfg_get(_cfg_get(create_config, "env", {}), "import_names", ()))
    env_cfg_type = _cfg_get(env_cfg, "type", None)
    env_id = _cfg_get(env_cfg, "env_id", None)
    if env_type != CURVYZERO_LIGHTZERO_ENV_TYPE:
        problems.append(f"create_config.env.type={env_type!r}, expected {CURVYZERO_LIGHTZERO_ENV_TYPE!r}")
    if env_cfg_type != CURVYZERO_LIGHTZERO_ENV_TYPE:
        problems.append(f"env_cfg.type={env_cfg_type!r}, expected {CURVYZERO_LIGHTZERO_ENV_TYPE!r}")
    if env_id != CURVYZERO_LIGHTZERO_ENV_ID:
        problems.append(f"env_cfg.env_id={env_id!r}, expected {CURVYZERO_LIGHTZERO_ENV_ID!r}")
    if CURVYZERO_LIGHTZERO_IMPORT not in env_import_names:
        problems.append("create_config.env.import_names does not import curvyzero LightZero env")
    disallowed = _disallowed_identity_hits(
        [env_type, env_cfg_type, env_id, *env_import_names]
    )
    if disallowed:
        problems.append(f"config identity contains CartPole/Atari/ALE markers: {disallowed}")
    return {
        "ok": not problems,
        "problems": problems,
        "create_config_env_type": env_type,
        "create_config_env_import_names": list(env_import_names),
        "env_cfg_type": env_cfg_type,
        "env_cfg_env_id": env_id,
        "not_cartpole_atari_ale": not disallowed,
        "disallowed_identity_hits": disallowed,
    }


def _env_identity_report(env: Any) -> dict[str, Any]:
    env_class = type(env).__module__ + "." + type(env).__name__
    env_id = getattr(env, "env_id", None)
    env_type = getattr(env, "lightzero_env_type", None)
    text_values = [env_class, env_id, env_type, repr(env)]
    disallowed = _disallowed_identity_hits(text_values)
    problems: list[str] = []
    if env_class != "curvyzero.training.curvyzero_lightzero_env.CurvyZeroLightZeroEnv":
        problems.append(f"env class was {env_class!r}, expected CurvyZeroLightZeroEnv")
    if env_id != CURVYZERO_LIGHTZERO_ENV_ID:
        problems.append(f"env.env_id={env_id!r}, expected {CURVYZERO_LIGHTZERO_ENV_ID!r}")
    if env_type != CURVYZERO_LIGHTZERO_ENV_TYPE:
        problems.append(
            f"env.lightzero_env_type={env_type!r}, expected {CURVYZERO_LIGHTZERO_ENV_TYPE!r}"
        )
    if disallowed:
        problems.append(f"env identity contains CartPole/Atari/ALE markers: {disallowed}")
    return {
        "ok": not problems,
        "problems": problems,
        "env_class": env_class,
        "env_id": env_id,
        "lightzero_env_type": env_type,
        "repr": repr(env),
        "not_cartpole_atari_ale": not disallowed,
        "disallowed_identity_hits": disallowed,
    }


def _disallowed_identity_hits(values: list[Any]) -> list[str]:
    hits: list[str] = []
    for value in values:
        text = str(value).lower()
        for token in ("cartpole", "atari", "ale", "pongnoframeskip"):
            if token in text:
                hits.append(f"{token}:{value}")
    return hits


def _optional_ding_base_env_timestep_cls() -> Any | None:
    try:
        from ding.envs import BaseEnvTimestep
    except ImportError:
        return None
    return BaseEnvTimestep


def _import_curvyzero_lightzero_env_module() -> Any:
    return importlib.import_module(CURVYZERO_LIGHTZERO_IMPORT)


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
            run_curvyzero_lightzero_runtime_probe(
                include_env_factory=False,
                require_installed_runtime=False,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
