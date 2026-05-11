"""Feature-fit and config/import smoke for LightZero dummy Pong.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_dummy_pong_config_import_smoke \
      --mode feature-fit \
      --env dummy_pong_lag1 \
      --feature-mode tabular_ego \
      --seed 0

This does not train. It verifies that the custom dummy Pong env can be
imported, reset, stepped, and targeted by a tiny LightZero MuZero config.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

APP_NAME = "curvyzero-lightzero-dummy-pong-config-import-smoke"
LIGHTZERO_VERSION = "0.2.0"
REMOTE_ROOT = Path("/repo")

DEFAULT_ENV = "dummy_pong_lag1"
DEFAULT_FEATURE_MODE = "tabular_ego"
DEFAULT_OPPONENT_POLICY = "random_uniform"
DEFAULT_SEED = 0
DEFAULT_MAX_ENV_STEP = 64
DEFAULT_COLLECTOR_ENV_NUM = 1
DEFAULT_EVALUATOR_ENV_NUM = 1
DEFAULT_N_EVALUATOR_EPISODE = 1
DEFAULT_NUM_SIMULATIONS = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_UPDATE_PER_COLLECT = 1
DEFAULT_N_EPISODE = 1
DEFAULT_GAME_SEGMENT_LENGTH = 50
DEFAULT_RANDOM_COLLECT_EPISODE_NUM = 0
DEFAULT_EPS_GREEDY_EXPLORATION_IN_COLLECT = False
DEFAULT_EPS_START = 1.0
DEFAULT_EPS_END = 0.05
DEFAULT_EPS_DECAY = 100_000
DEFAULT_FIXED_TEMPERATURE_VALUE = 0.25
DEFAULT_ACTION_TYPE = "fixed_action_space"
DEFAULT_PONG_EPISODE_MAX_STEPS: int | None = None
DEFAULT_PONG_RESET_PROFILE = "default"
DEFAULT_PONG_RESET_PRESSURE_AGENT = "ego"
DEFAULT_TD_STEPS: int | None = None
DEFAULT_NUM_UNROLL_STEPS: int | None = None
DEFAULT_DISCOUNT_FACTOR: float | None = None
DEFAULT_SUPPORT_SCALE: int | None = None
DEFAULT_REWARD_SUPPORT_MIN: float | None = None
DEFAULT_REWARD_SUPPORT_MAX: float | None = None
DEFAULT_REWARD_SUPPORT_DELTA: float | None = None
DEFAULT_VALUE_SUPPORT_MIN: float | None = None
DEFAULT_VALUE_SUPPORT_MAX: float | None = None
DEFAULT_VALUE_SUPPORT_DELTA: float | None = None

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}", "numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
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
    if hasattr(value, "tolist"):
        return _to_plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _exception_result(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _import_status(module_names: tuple[str, ...]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
            statuses[module_name] = "ok"
        except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
            statuses[module_name] = f"{type(exc).__name__}: {exc}"
    return statuses


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


def _set_or_add_path(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        current = current[part]
    key = path[-1]
    try:
        old_value = current[key]
    except KeyError:
        old_value = "<missing>"
    current[key] = value
    return {
        "path": ".".join(path),
        "old": _to_plain(old_value),
        "new": _to_plain(value),
    }


def _support_range_from_parts(
    name: str,
    *,
    minimum: float | None,
    maximum: float | None,
    delta: float | None,
) -> tuple[float, float, float] | None:
    values = (minimum, maximum, delta)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{name} support range requires min, max, and delta together")
    if delta is None or delta <= 0:
        raise ValueError(f"{name} support delta must be positive")
    if minimum is None or maximum is None or maximum <= minimum:
        raise ValueError(f"{name} support max must be greater than min")
    return (float(minimum), float(maximum), float(delta))


def _support_size_from_scale(support_scale: int | None) -> int | None:
    if support_scale is None:
        return None
    support_scale = int(support_scale)
    if support_scale < 1:
        raise ValueError("support_scale must be at least 1")
    return support_scale * 2 + 1


def _support_range_size(support_range: tuple[float, float, float]) -> int:
    minimum, maximum, delta = support_range
    span = (maximum - minimum) / delta
    rounded = round(span)
    if abs(span - rounded) > 1e-9:
        raise ValueError(f"support range {support_range!r} does not have an integer size")
    return int(rounded)


def _support_scale_from_range(
    name: str,
    support_range: tuple[float, float, float],
) -> int:
    minimum, maximum, delta = support_range
    if delta != 1.0:
        raise ValueError(f"{name} support range must have delta=1 for LightZero v0.2.0")
    scale = int(abs(minimum))
    if minimum != -float(scale) or maximum != float(scale + 1):
        raise ValueError(
            f"{name} support range {support_range!r} must be symmetric [-n, n + 1, 1]"
        )
    return scale


def _range_from_support_scale(support_scale: int) -> tuple[float, float, float]:
    return (-float(support_scale), float(support_scale + 1), 1.0)


def _resolve_v020_support_scale(
    *,
    support_scale: int | None,
    reward_support_range: tuple[float, float, float] | None,
    value_support_range: tuple[float, float, float] | None,
) -> tuple[int | None, tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Resolve the v0.2.0 MuZero support_scale and coherent range metadata."""

    if support_scale is not None:
        resolved_scale = int(support_scale)
        _support_size_from_scale(resolved_scale)
    elif reward_support_range is not None and value_support_range is not None:
        reward_scale = _support_scale_from_range("reward", reward_support_range)
        value_scale = _support_scale_from_range("value", value_support_range)
        if reward_scale != value_scale:
            raise ValueError(
                "LightZero v0.2.0 has one model.support_scale; reward and value ranges "
                f"resolve to different scales: {reward_scale} vs {value_scale}"
            )
        resolved_scale = reward_scale
    else:
        resolved_scale = None

    if resolved_scale is None:
        return None, reward_support_range, value_support_range

    expected_range = _range_from_support_scale(resolved_scale)
    if reward_support_range is None:
        reward_support_range = expected_range
    if value_support_range is None:
        value_support_range = expected_range
    for name, support_range in (
        ("reward", reward_support_range),
        ("value", value_support_range),
    ):
        if support_range != expected_range:
            raise ValueError(
                f"{name} support range {support_range!r} is not coherent with "
                f"support_scale={resolved_scale}; expected {expected_range!r}"
            )
        if _support_range_size(support_range) != _support_size_from_scale(resolved_scale):
            raise ValueError(f"{name} support size does not match support_scale")
    return resolved_scale, reward_support_range, value_support_range


def resolve_pong_episode_max_steps(
    *, max_env_step: int, pong_episode_max_steps: int | None
) -> int:
    if max_env_step < 1:
        raise ValueError("max_env_step must be at least 1")
    if pong_episode_max_steps is None:
        return int(max_env_step)
    if pong_episode_max_steps < 1:
        raise ValueError("pong_episode_max_steps must be at least 1")
    return int(pong_episode_max_steps)


def _pong_episode_horizon_source(pong_episode_max_steps: int | None) -> str:
    return "max_env_step_legacy_default" if pong_episode_max_steps is None else "explicit"


def _extract_surface(
    main_config: Any,
    create_config: Any,
    *,
    max_env_step: int,
    pong_episode_max_steps: int | None,
) -> dict[str, Any]:
    effective_pong_episode_max_steps = int(main_config["env"]["max_steps"])
    return {
        "env_id": main_config["env"]["env_id"],
        "policy_type": create_config["policy"]["type"],
        "env_type": create_config["env"]["type"],
        "model_type": main_config["policy"]["model"]["model_type"],
        "observation_shape": _to_plain(main_config["policy"]["model"]["observation_shape"]),
        "action_space_size": main_config["policy"]["model"]["action_space_size"],
        "reward_support_range": _to_plain(
            main_config["policy"]["model"].get("reward_support_range")
        ),
        "value_support_range": _to_plain(
            main_config["policy"]["model"].get("value_support_range")
        ),
        "support_scale": main_config["policy"]["model"].get("support_scale"),
        "reward_support_size": main_config["policy"]["model"].get("reward_support_size"),
        "value_support_size": main_config["policy"]["model"].get("value_support_size"),
        "categorical_distribution": main_config["policy"]["model"].get(
            "categorical_distribution"
        ),
        "collector_env_num": main_config["env"]["collector_env_num"],
        "evaluator_env_num": main_config["env"]["evaluator_env_num"],
        "n_evaluator_episode": main_config["env"]["n_evaluator_episode"],
        "num_simulations": main_config["policy"]["num_simulations"],
        "batch_size": main_config["policy"]["batch_size"],
        "update_per_collect": main_config["policy"]["update_per_collect"],
        "td_steps": main_config["policy"].get("td_steps"),
        "num_unroll_steps": main_config["policy"].get("num_unroll_steps"),
        "discount_factor": main_config["policy"].get("discount_factor"),
        "n_episode": main_config["policy"].get("n_episode"),
        "game_segment_length": main_config["policy"].get("game_segment_length"),
        "random_collect_episode_num": main_config["policy"].get("random_collect_episode_num"),
        "eps_greedy_exploration_in_collect": main_config["policy"].get("eps", {}).get(
            "eps_greedy_exploration_in_collect"
        ),
        "eps_start": main_config["policy"].get("eps", {}).get("start"),
        "eps_end": main_config["policy"].get("eps", {}).get("end"),
        "eps_decay": main_config["policy"].get("eps", {}).get("decay"),
        "fixed_temperature_value": main_config["policy"].get("fixed_temperature_value"),
        "action_type": main_config["policy"].get("action_type"),
        "eval_freq": main_config["policy"].get("eval_freq"),
        "cuda": main_config["policy"]["cuda"],
        "max_env_step": max_env_step,
        "max_env_step_role": "lightzero_training_budget",
        "requested_pong_episode_max_steps": pong_episode_max_steps,
        "pong_episode_max_steps": effective_pong_episode_max_steps,
        "effective_pong_episode_max_steps": effective_pong_episode_max_steps,
        "pong_episode_max_steps_source": _pong_episode_horizon_source(
            pong_episode_max_steps
        ),
        "pong_reset_profile": main_config["env"].get("pong_reset_profile"),
        "pong_reset_pressure_agent": main_config["env"].get("pong_reset_pressure_agent"),
        "env_max_steps": effective_pong_episode_max_steps,
        "curvyzero_env": main_config["env"].get("curvyzero_env"),
        "feature_mode": main_config["env"].get("feature_mode"),
        "opponent_policy": main_config["env"].get("opponent_policy"),
        "ego_agent": main_config["env"].get("ego_agent"),
        "opponent_checkpoint_path": main_config["env"].get("opponent_checkpoint_path"),
        "opponent_checkpoint_label": main_config["env"].get("opponent_checkpoint_label"),
        "opponent_checkpoint_adapter": main_config["env"].get("opponent_checkpoint_adapter"),
        "opponent_checkpoint_num_simulations": main_config["env"].get(
            "opponent_checkpoint_num_simulations"
        ),
        "opponent_checkpoint_source_ref": main_config["env"].get(
            "opponent_checkpoint_source_ref"
        ),
        "opponent_checkpoint_sha256": main_config["env"].get("opponent_checkpoint_sha256"),
        "opponent_checkpoint_state_key": main_config["env"].get(
            "opponent_checkpoint_state_key"
        ),
        "dynamic_seed": main_config["env"].get("dynamic_seed"),
        "telemetry_path": main_config["env"].get("telemetry_path"),
        "exp_name": str(main_config["exp_name"]),
    }


def patched_dummy_pong_configs(
    *,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    pong_episode_max_steps: int | None = DEFAULT_PONG_EPISODE_MAX_STEPS,
    pong_reset_profile: str = DEFAULT_PONG_RESET_PROFILE,
    pong_reset_pressure_agent: str = DEFAULT_PONG_RESET_PRESSURE_AGENT,
    n_episode: int = DEFAULT_N_EPISODE,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    random_collect_episode_num: int = DEFAULT_RANDOM_COLLECT_EPISODE_NUM,
    eps_greedy_exploration_in_collect: bool = DEFAULT_EPS_GREEDY_EXPLORATION_IN_COLLECT,
    eps_start: float = DEFAULT_EPS_START,
    eps_end: float = DEFAULT_EPS_END,
    eps_decay: int = DEFAULT_EPS_DECAY,
    fixed_temperature_value: float = DEFAULT_FIXED_TEMPERATURE_VALUE,
    td_steps: int | None = DEFAULT_TD_STEPS,
    num_unroll_steps: int | None = DEFAULT_NUM_UNROLL_STEPS,
    discount_factor: float | None = DEFAULT_DISCOUNT_FACTOR,
    support_scale: int | None = DEFAULT_SUPPORT_SCALE,
    reward_support_min: float | None = DEFAULT_REWARD_SUPPORT_MIN,
    reward_support_max: float | None = DEFAULT_REWARD_SUPPORT_MAX,
    reward_support_delta: float | None = DEFAULT_REWARD_SUPPORT_DELTA,
    value_support_min: float | None = DEFAULT_VALUE_SUPPORT_MIN,
    value_support_max: float | None = DEFAULT_VALUE_SUPPORT_MAX,
    value_support_delta: float | None = DEFAULT_VALUE_SUPPORT_DELTA,
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE,
    telemetry_path: str | None = None,
    dynamic_seed: bool = True,
    ego_agent: str = "player_0",
    opponent_checkpoint_path: str | None = None,
    opponent_checkpoint_label: str | None = None,
    opponent_checkpoint_adapter: str | None = None,
    opponent_checkpoint_num_simulations: int | None = None,
    opponent_checkpoint_sha256: str | None = None,
    opponent_checkpoint_source_ref: str | None = None,
    opponent_checkpoint_state_key: str | None = None,
    opponent_checkpoint_config_opponent_policy: str | None = None,
) -> dict[str, Any]:
    from easydict import EasyDict
    from curvyzero.training.dummy_pong import PongConfig
    from curvyzero.training.lightzero_dummy_pong_features import lightzero_observation_shape

    module_name = "zoo.classic_control.cartpole.config.cartpole_muzero_config"
    module = importlib.import_module(module_name)
    main_config = copy.deepcopy(module.main_config)
    create_config = copy.deepcopy(module.create_config)
    effective_pong_episode_max_steps = resolve_pong_episode_max_steps(
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
    )
    reward_support_range = _support_range_from_parts(
        "reward",
        minimum=reward_support_min,
        maximum=reward_support_max,
        delta=reward_support_delta,
    )
    value_support_range = _support_range_from_parts(
        "value",
        minimum=value_support_min,
        maximum=value_support_max,
        delta=value_support_delta,
    )
    resolved_support_scale, reward_support_range, value_support_range = (
        _resolve_v020_support_scale(
            support_scale=support_scale,
            reward_support_range=reward_support_range,
            value_support_range=value_support_range,
        )
    )

    patches = [
        _set_path(
            main_config,
            ("exp_name",),
            str(Path("/tmp") / "curvyzero-lightzero-dummy-pong" / f"seed-{seed}"),
        ),
        _set_path(main_config, ("env", "env_id"), "DummyPongLightZero-v0"),
        _set_path(main_config, ("env", "collector_env_num"), collector_env_num),
        _set_path(main_config, ("env", "evaluator_env_num"), evaluator_env_num),
        _set_path(main_config, ("env", "n_evaluator_episode"), n_evaluator_episode),
        _set_path_if_present(main_config, ("env", "continuous"), False),
        _set_path_if_present(main_config, ("env", "manually_discretization"), False),
        _set_path(main_config, ("policy", "cuda"), False),
        _set_path_if_present(main_config, ("policy", "env_type"), "not_board_games"),
        _set_path(main_config, ("policy", "collector_env_num"), collector_env_num),
        _set_path(main_config, ("policy", "evaluator_env_num"), evaluator_env_num),
        _set_path(main_config, ("policy", "n_episode"), n_episode),
        _set_path(main_config, ("policy", "game_segment_length"), game_segment_length),
        _set_path(main_config, ("policy", "num_simulations"), num_simulations),
        _set_path(main_config, ("policy", "batch_size"), batch_size),
        _set_path(main_config, ("policy", "update_per_collect"), update_per_collect),
        _set_path_if_present(
            main_config,
            ("policy", "random_collect_episode_num"),
            random_collect_episode_num,
        ),
        _set_path_if_present(
            main_config,
            ("policy", "fixed_temperature_value"),
            fixed_temperature_value,
        ),
        _set_path(main_config, ("policy", "action_type"), DEFAULT_ACTION_TYPE),
        _set_path_if_present(
            main_config,
            ("policy", "eps", "eps_greedy_exploration_in_collect"),
            eps_greedy_exploration_in_collect,
        ),
        _set_path_if_present(main_config, ("policy", "eps", "start"), eps_start),
        _set_path_if_present(main_config, ("policy", "eps", "end"), eps_end),
        _set_path_if_present(main_config, ("policy", "eps", "decay"), eps_decay),
        _set_path_if_present(main_config, ("policy", "eval_freq"), 1),
        _set_path(main_config, ("policy", "model", "model_type"), "mlp"),
        _set_path(
            main_config,
            ("policy", "model", "observation_shape"),
            lightzero_observation_shape(feature_mode, PongConfig()),
        ),
        _set_path(main_config, ("policy", "model", "action_space_size"), 3),
    ]
    if resolved_support_scale is not None:
        support_size = _support_size_from_scale(resolved_support_scale)
        patches.extend(
            [
                _set_or_add_path(
                    main_config,
                    ("policy", "model", "support_scale"),
                    int(resolved_support_scale),
                ),
                _set_or_add_path(
                    main_config,
                    ("policy", "model", "reward_support_size"),
                    int(support_size),
                ),
                _set_or_add_path(
                    main_config,
                    ("policy", "model", "value_support_size"),
                    int(support_size),
                ),
            ]
        )
    if td_steps is not None:
        patches.append(_set_or_add_path(main_config, ("policy", "td_steps"), int(td_steps)))
    if num_unroll_steps is not None:
        patches.append(
            _set_or_add_path(main_config, ("policy", "num_unroll_steps"), int(num_unroll_steps))
        )
    if discount_factor is not None:
        patches.append(
            _set_or_add_path(main_config, ("policy", "discount_factor"), float(discount_factor))
        )
    if reward_support_range is not None:
        patches.append(
            _set_or_add_path(
                main_config,
                ("policy", "model", "reward_support_range"),
                reward_support_range,
            )
        )
    if value_support_range is not None:
        patches.append(
            _set_or_add_path(
                main_config,
                ("policy", "model", "value_support_range"),
                value_support_range,
            )
        )

    create_config = EasyDict(
        {
            "env": {
                "type": "dummy_pong_lightzero",
                "import_names": ["curvyzero.training.lightzero_dummy_pong_env"],
            },
            "env_manager": {"type": "base"},
            "policy": {
                "type": "muzero",
                "import_names": ["lzero.policy.muzero"],
            },
        }
    )
    env_cfg = EasyDict(
        {
            **_to_plain(main_config["env"]),
            "type": "dummy_pong_lightzero",
            "env_id": "DummyPongLightZero-v0",
            "curvyzero_env": env,
            "feature_mode": feature_mode,
            "opponent_policy": opponent_policy,
            "ego_agent": ego_agent,
            "opponent_checkpoint_path": opponent_checkpoint_path,
            "opponent_checkpoint_label": opponent_checkpoint_label,
            "opponent_checkpoint_adapter": opponent_checkpoint_adapter,
            "opponent_checkpoint_num_simulations": opponent_checkpoint_num_simulations or 2,
            "opponent_checkpoint_sha256": opponent_checkpoint_sha256,
            "opponent_checkpoint_source_ref": opponent_checkpoint_source_ref,
            "opponent_checkpoint_state_key": opponent_checkpoint_state_key or "model",
            "opponent_checkpoint_config_opponent_policy": (
                opponent_checkpoint_config_opponent_policy or "random_uniform"
            ),
            "dynamic_seed": bool(dynamic_seed),
            "seed": seed,
            "max_steps": effective_pong_episode_max_steps,
            "pong_episode_max_steps": effective_pong_episode_max_steps,
            "pong_reset_profile": pong_reset_profile,
            "pong_reset_pressure_agent": pong_reset_pressure_agent,
            "telemetry_path": telemetry_path,
            "continuous": False,
            "manually_discretization": False,
        }
    )
    main_config["env"] = env_cfg
    patches.append(
        {
            "path": "env.dynamic_seed",
            "old": "<missing-or-template-default>",
            "new": bool(dynamic_seed),
        }
    )
    patches.append(
        {
            "path": "env.pong_episode_max_steps",
            "old": "<missing-or-template-default>",
            "new": effective_pong_episode_max_steps,
            "source": _pong_episode_horizon_source(pong_episode_max_steps),
        }
    )
    patches.append(
        {
            "path": "env.pong_reset_profile",
            "old": "<missing-or-template-default>",
            "new": pong_reset_profile,
        }
    )
    patches.append(
        {
            "path": "env.pong_reset_pressure_agent",
            "old": "<missing-or-template-default>",
            "new": pong_reset_pressure_agent,
        }
    )
    if opponent_checkpoint_path is not None:
        patches.append(
            {
                "path": "env.opponent_checkpoint",
                "old": "<missing-or-template-default>",
                "new": {
                    "path": opponent_checkpoint_path,
                    "label": opponent_checkpoint_label,
                    "adapter": opponent_checkpoint_adapter,
                    "num_simulations": opponent_checkpoint_num_simulations,
                    "source_ref": opponent_checkpoint_source_ref,
                    "sha256": opponent_checkpoint_sha256,
                    "state_key": opponent_checkpoint_state_key or "model",
                },
            }
        )

    return {
        "module": module_name,
        "main_config": main_config,
        "create_config": create_config,
        "patched_surface": _extract_surface(
            main_config,
            create_config,
            max_env_step=max_env_step,
            pong_episode_max_steps=pong_episode_max_steps,
        ),
        "patches": patches,
    }


def validate_dummy_pong_surface(
    surface: dict[str, Any],
    *,
    max_env_step: int,
    pong_episode_max_steps: int | None = DEFAULT_PONG_EPISODE_MAX_STEPS,
    pong_reset_profile: str = DEFAULT_PONG_RESET_PROFILE,
    expected_dynamic_seed: bool = True,
) -> list[str]:
    """Return simple config problems before calling the LightZero trainer."""

    problems = []
    effective_pong_episode_max_steps = resolve_pong_episode_max_steps(
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
    )
    expected = {
        "env_id": "DummyPongLightZero-v0",
        "env_type": "dummy_pong_lightzero",
        "policy_type": "muzero",
        "model_type": "mlp",
        "action_space_size": 3,
        "cuda": False,
        "max_env_step": max_env_step,
        "pong_episode_max_steps": effective_pong_episode_max_steps,
        "effective_pong_episode_max_steps": effective_pong_episode_max_steps,
        "env_max_steps": effective_pong_episode_max_steps,
    }
    for key, value in expected.items():
        if surface.get(key) != value:
            problems.append(f"{key}={surface.get(key)!r}, expected {value!r}")
    if int(surface.get("collector_env_num", 0)) < 1:
        problems.append("collector_env_num must be at least 1")
    if int(surface.get("evaluator_env_num", 0)) < 1:
        problems.append("evaluator_env_num must be at least 1")
    if int(surface.get("n_evaluator_episode", 0)) < 1:
        problems.append("n_evaluator_episode must be at least 1")
    if surface.get("feature_mode") not in {"tabular_ego", "raster_flat"}:
        problems.append(f"unknown feature_mode {surface.get('feature_mode')!r}")
    if surface.get("dynamic_seed") is not expected_dynamic_seed:
        problems.append(
            f"dynamic_seed={surface.get('dynamic_seed')!r}, expected {expected_dynamic_seed!r}"
        )
    if surface.get("pong_reset_profile") != pong_reset_profile:
        problems.append(
            "pong_reset_profile="
            f"{surface.get('pong_reset_profile')!r}, expected {pong_reset_profile!r}"
        )
    from curvyzero.training.dummy_pong import PongConfig
    from curvyzero.training.lightzero_dummy_pong_features import lightzero_observation_shape

    expected_shape = lightzero_observation_shape(
        str(surface.get("feature_mode")),
        PongConfig(),
    )
    if surface.get("observation_shape") != expected_shape:
        problems.append(
            f"observation_shape={surface.get('observation_shape')!r}, expected {expected_shape!r}"
        )
    return problems


def _summarize_value(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        summary["shape"] = [int(item) for item in shape]
    if dtype is not None:
        summary["dtype"] = str(dtype)
    if isinstance(value, dict):
        summary["keys"] = [str(key) for key in value.keys()]
    else:
        text = repr(_to_plain(value))
        summary["repr"] = text if len(text) <= 240 else text[:237] + "..."
    return summary


def _exercise_direct_env(
    *,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    pong_episode_max_steps: int | None,
    pong_reset_profile: str,
    pong_reset_pressure_agent: str,
) -> dict[str, Any]:
    from curvyzero.training.lightzero_dummy_pong_env import DummyPongLightZeroEnv

    started = time.perf_counter()
    problems: list[str] = []
    effective_pong_episode_max_steps = resolve_pong_episode_max_steps(
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
    )
    wrapper = DummyPongLightZeroEnv(
        {
            "curvyzero_env": env,
            "feature_mode": feature_mode,
            "opponent_policy": opponent_policy,
            "seed": seed,
            "max_steps": effective_pong_episode_max_steps,
            "pong_episode_max_steps": effective_pong_episode_max_steps,
            "pong_reset_profile": pong_reset_profile,
            "pong_reset_pressure_agent": pong_reset_pressure_agent,
        }
    )
    wrapper.seed(seed, dynamic_seed=False)
    reset_obs = wrapper.reset()
    step_result = wrapper.step(1)
    curvy_info = step_result.info.get("curvyzero_pong", {})
    required_info_fields = (
        "base_seed",
        "episode_seed",
        "ego_agent",
        "opponent_agent",
        "opponent_policy_id",
        "configured_dynamic_seed",
        "effective_dynamic_seed",
        "seed_call_dynamic_seed_arg",
        "seed_source",
        "steps",
        "max_steps",
        "pong_episode_max_steps",
        "winner",
        "terminated",
        "truncated",
        "score_return",
        "survival_fraction",
        "action_counts_by_agent",
        "trace_hash",
        "pong_reset_profile",
        "pong_reset_pressure_agent",
        "pong_reset",
    )
    for field in required_info_fields:
        if field not in curvy_info:
            problems.append(f"curvyzero_pong info missing {field}")
    if curvy_info.get("opponent_policy_id") != opponent_policy:
        problems.append("opponent_policy_id did not round-trip")
    if reset_obs.get("to_play") != -1:
        problems.append("reset to_play was not -1")
    if reset_obs.get("action_mask") is None or list(reset_obs["action_mask"]) != [1, 1, 1]:
        problems.append("reset action_mask was not all ones for A=3")

    return {
        "ok": not problems,
        "problems": problems,
        "env_class": type(wrapper).__module__ + "." + type(wrapper).__name__,
        "observation_space": _summarize_value(wrapper.observation_space),
        "action_space": _summarize_value(wrapper.action_space),
        "reset_observation": _summarize_value(reset_obs),
        "reset_observation_leaf": _summarize_value(reset_obs["observation"]),
        "step": {
            "reward": float(step_result.reward),
            "done": bool(step_result.done),
            "observation": _summarize_value(step_result.obs),
            "curvyzero_pong": _to_plain(curvy_info),
        },
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }


def _exercise_seed_policy(
    *,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    pong_episode_max_steps: int | None,
    pong_reset_profile: str,
    pong_reset_pressure_agent: str,
) -> dict[str, Any]:
    from curvyzero.training.lightzero_dummy_pong_env import DummyPongLightZeroEnv
    effective_pong_episode_max_steps = resolve_pong_episode_max_steps(
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
    )

    def collect(
        *,
        configured_dynamic_seed: bool,
        seed_call_dynamic_seed: bool,
    ) -> dict[str, Any]:
        wrapper = DummyPongLightZeroEnv(
            {
                "curvyzero_env": env,
                "feature_mode": feature_mode,
                "opponent_policy": opponent_policy,
                "dynamic_seed": configured_dynamic_seed,
                "seed": seed,
                "max_steps": effective_pong_episode_max_steps,
                "pong_episode_max_steps": effective_pong_episode_max_steps,
                "pong_reset_profile": pong_reset_profile,
                "pong_reset_pressure_agent": pong_reset_pressure_agent,
            }
        )
        wrapper.seed(seed, dynamic_seed=seed_call_dynamic_seed)
        episode_seeds = []
        for _ in range(4):
            wrapper.reset()
            episode_seeds.append(int(wrapper._episode_seed))
        return {
            "configured_dynamic_seed": bool(wrapper._configured_dynamic_seed),
            "effective_dynamic_seed": bool(wrapper._dynamic_seed),
            "seed_call_dynamic_seed_arg": bool(wrapper._last_seed_call_dynamic_seed_arg),
            "seed_source": str(wrapper._last_seed_source),
            "episode_seeds": episode_seeds,
        }

    dynamic_cfg_env_manager_false = collect(
        configured_dynamic_seed=True,
        seed_call_dynamic_seed=False,
    )
    fixed_cfg_env_manager_false = collect(
        configured_dynamic_seed=False,
        seed_call_dynamic_seed=False,
    )
    fixed_cfg_env_manager_true = collect(
        configured_dynamic_seed=False,
        seed_call_dynamic_seed=True,
    )
    problems = []
    expected_dynamic = [seed + offset for offset in range(4)]
    expected_fixed = [seed for _ in range(4)]
    if dynamic_cfg_env_manager_false["episode_seeds"] != expected_dynamic:
        problems.append(
            "dynamic cfg did not override env.seed(..., dynamic_seed=False): "
            f"{dynamic_cfg_env_manager_false['episode_seeds']!r}"
        )
    if fixed_cfg_env_manager_false["episode_seeds"] != expected_fixed:
        problems.append(
            "fixed cfg with env.seed(..., dynamic_seed=False) did not stay fixed: "
            f"{fixed_cfg_env_manager_false['episode_seeds']!r}"
        )
    if fixed_cfg_env_manager_true["episode_seeds"] != expected_fixed:
        problems.append(
            "fixed cfg with env.seed(..., dynamic_seed=True) did not stay fixed: "
            f"{fixed_cfg_env_manager_true['episode_seeds']!r}"
        )
    return {
        "ok": not problems,
        "problems": problems,
        "dynamic_cfg_env_manager_false": dynamic_cfg_env_manager_false,
        "fixed_cfg_env_manager_false": fixed_cfg_env_manager_false,
        "fixed_cfg_env_manager_true": fixed_cfg_env_manager_true,
    }


def _compile_config(main_config: Any, create_config: Any, seed: int) -> dict[str, Any]:
    from ding.config import compile_config

    started = time.perf_counter()
    try:
        compiled = compile_config(
            copy.deepcopy(main_config),
            seed=seed,
            auto=True,
            create_cfg=copy.deepcopy(create_config),
            save_cfg=False,
        )
        env_cfg = getattr(compiled, "env", {})
        model_cfg = getattr(getattr(compiled, "policy", {}), "model", {})
        return {
            "ok": True,
            "env_cfg": {
                "type": getattr(env_cfg, "type", None),
                "env_id": getattr(env_cfg, "env_id", None),
                "curvyzero_env": getattr(env_cfg, "curvyzero_env", None),
                "feature_mode": getattr(env_cfg, "feature_mode", None),
                "opponent_policy": getattr(env_cfg, "opponent_policy", None),
                "dynamic_seed": getattr(env_cfg, "dynamic_seed", None),
                "max_steps": getattr(env_cfg, "max_steps", None),
                "pong_episode_max_steps": getattr(env_cfg, "pong_episode_max_steps", None),
                "pong_reset_profile": getattr(env_cfg, "pong_reset_profile", None),
                "pong_reset_pressure_agent": getattr(
                    env_cfg, "pong_reset_pressure_agent", None
                ),
                "collector_env_num": getattr(env_cfg, "collector_env_num", None),
                "evaluator_env_num": getattr(env_cfg, "evaluator_env_num", None),
            },
            "policy_model_cfg": {
                "support_scale": model_cfg.get("support_scale"),
                "reward_support_size": model_cfg.get("reward_support_size"),
                "value_support_size": model_cfg.get("value_support_size"),
                "reward_support_range": _to_plain(model_cfg.get("reward_support_range")),
                "value_support_range": _to_plain(model_cfg.get("value_support_range")),
                "categorical_distribution": model_cfg.get("categorical_distribution"),
            },
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:  # pragma: no cover - remote dependency diagnosis.
        result = {"ok": False, "elapsed_sec": round(time.perf_counter() - started, 6)}
        result.update(_exception_result(exc))
        return result


def _trainer_surface() -> dict[str, Any]:
    entry_module = importlib.import_module("lzero.entry")
    train_muzero = entry_module.train_muzero
    return {
        "trainer_entrypoint": "lzero.entry.train_muzero",
        "trainer_signature": str(inspect.signature(train_muzero)),
    }


def _feature_fit_report(
    *,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    direct_env: dict[str, Any],
    compiled_config: dict[str, Any],
    trainer_surface: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "env_reset_step": direct_env["ok"],
        "observation_shape": direct_env["reset_observation_leaf"].get("shape") in ([10], [135]),
        "legal_actions": direct_env["reset_observation"].get("keys") is not None,
        "reward_info_telemetry": bool(direct_env["step"]["curvyzero_pong"].get("trace_hash")),
        "trainer_entrypoint_fit": "max_env_step" in trainer_surface["trainer_signature"],
        "checkpoint_discovery_plan": True,
        "independent_scorecard_plan": True,
        "compiled_config_targets_custom_env": compiled_config.get("env_cfg", {}).get("type")
        == "dummy_pong_lightzero",
    }
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "ok": not missing,
        "checks": checks,
        "missing": missing,
        "env": env,
        "feature_mode": feature_mode,
        "opponent_policy": opponent_policy,
        "seed": seed,
        "legal_action_handling": "all three dummy Pong actions are always legal; action_mask is ones",
        "required_telemetry": (
            "wins/losses, survival steps mean/median/p90/std, truncation rate, "
            "score return stats, shaped loss-delay stats, seeds, actions, trace hash"
        ),
        "checkpoint_discovery_plan": (
            "LightZero tiny train smoke will scan the exp_name tree for "
            "ckpt_best.pth.tar and iteration_*.pth.tar, then mirror them to "
            "curvyzero-runs under training/lightzero-dummy-pong/<run_id>/."
        ),
        "independent_scorecard_plan_doc": (
            "docs/working/lightzero_pong_scorecard_plan_2026-05-09.md"
        ),
    }


def _run_lightzero_dummy_pong_config_import_smoke(
    *,
    mode: str,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    pong_episode_max_steps: int | None,
    pong_reset_profile: str,
    pong_reset_pressure_agent: str,
    collector_env_num: int,
    evaluator_env_num: int,
    num_simulations: int,
    batch_size: int,
    update_per_collect: int,
    n_episode: int,
    game_segment_length: int,
    random_collect_episode_num: int,
    eps_greedy_exploration_in_collect: bool,
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    fixed_temperature_value: float,
    td_steps: int | None,
    num_unroll_steps: int | None,
    discount_factor: float | None,
    support_scale: int | None,
    reward_support_min: float | None,
    reward_support_max: float | None,
    reward_support_delta: float | None,
    value_support_min: float | None,
    value_support_max: float | None,
    value_support_delta: float | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    if mode not in {"feature-fit", "config-import"}:
        raise ValueError("mode must be 'feature-fit' or 'config-import'")

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
            "curvyzero.training.dummy_pong",
            "curvyzero.training.lightzero_dummy_pong_env",
        )
    )
    problems = [
        f"failed to import {module}: {status}"
        for module, status in imports.items()
        if status != "ok"
    ]

    patched = patched_dummy_pong_configs(
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
        pong_reset_profile=pong_reset_profile,
        pong_reset_pressure_agent=pong_reset_pressure_agent,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        n_episode=n_episode,
        game_segment_length=game_segment_length,
        random_collect_episode_num=random_collect_episode_num,
        eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        fixed_temperature_value=fixed_temperature_value,
        td_steps=td_steps,
        num_unroll_steps=num_unroll_steps,
        discount_factor=discount_factor,
        support_scale=support_scale,
        reward_support_min=reward_support_min,
        reward_support_max=reward_support_max,
        reward_support_delta=reward_support_delta,
        value_support_min=value_support_min,
        value_support_max=value_support_max,
        value_support_delta=value_support_delta,
    )
    direct_env = _exercise_direct_env(
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
        pong_reset_profile=pong_reset_profile,
        pong_reset_pressure_agent=pong_reset_pressure_agent,
    )
    seed_policy = _exercise_seed_policy(
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
        pong_reset_profile=pong_reset_profile,
        pong_reset_pressure_agent=pong_reset_pressure_agent,
    )
    compiled_config = _compile_config(patched["main_config"], patched["create_config"], seed)
    trainer_surface = _trainer_surface()
    feature_fit = _feature_fit_report(
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        direct_env=direct_env,
        compiled_config=compiled_config,
        trainer_surface=trainer_surface,
    )

    if not direct_env["ok"]:
        problems.append("direct DummyPongLightZeroEnv reset/step failed")
    if not seed_policy["ok"]:
        problems.append("DummyPongLightZeroEnv seed policy smoke failed")
    if not compiled_config["ok"]:
        problems.append("compile_config failed for dummy Pong LightZero config")
    if not feature_fit["ok"]:
        problems.append(f"feature fit missing required checks: {feature_fit['missing']}")
    compiled_model = compiled_config.get("policy_model_cfg", {})
    if (
        patched["patched_surface"].get("support_scale") is not None
        and patched["patched_surface"].get("support_scale") != compiled_model.get("support_scale")
    ):
        problems.append(
            "compiled policy.model.support_scale did not match patched surface: "
            f"{compiled_model.get('support_scale')!r} vs "
            f"{patched['patched_surface'].get('support_scale')!r}"
        )

    result = {
        "ok": not problems,
        "label": "LightZero custom dummy Pong config/import smoke",
        "mode": mode,
        "call_policy": "does_not_train; imports env, exercises reset/step, captures config",
        "algorithm": "LightZero MuZero config only; no trainer call",
        "problems": problems,
        "packages": packages,
        "imports": imports,
        "patched_config": {
            "source_module": patched["module"],
            "surface": patched["patched_surface"],
            "patches": patched["patches"],
            "create_config": _to_plain(patched["create_config"]),
        },
        "direct_env": direct_env,
        "seed_policy": seed_policy,
        "compiled_config": compiled_config,
        "trainer_surface": trainer_surface,
        "feature_fit": feature_fit,
        "next_command_if_ok": (
            "uv run --extra modal modal run -m "
            "curvyzero.infra.modal.lightzero_dummy_pong_tiny_train_smoke "
            f"--mode progression --env dummy_pong_lag1 --feature-mode {feature_mode} "
            "--seed 0 --opponent-policy random_uniform --max-env-step 64 "
            "--max-train-iter 2 --num-simulations 2 --batch-size 8 "
            "--update-per-collect 1 --n-evaluator-episode 1"
        ),
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    print(json.dumps(_to_plain(result), indent=2, sort_keys=True))
    return _to_plain(result)


@app.function(image=image, timeout=8 * 60)
def lightzero_dummy_pong_config_import_smoke(
    mode: str = "config-import",
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    opponent_policy: str = DEFAULT_OPPONENT_POLICY,
    seed: int = 0,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    pong_episode_max_steps: int | None = DEFAULT_PONG_EPISODE_MAX_STEPS,
    pong_reset_profile: str = DEFAULT_PONG_RESET_PROFILE,
    pong_reset_pressure_agent: str = DEFAULT_PONG_RESET_PRESSURE_AGENT,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    n_episode: int = DEFAULT_N_EPISODE,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    random_collect_episode_num: int = DEFAULT_RANDOM_COLLECT_EPISODE_NUM,
    eps_greedy_exploration_in_collect: bool = DEFAULT_EPS_GREEDY_EXPLORATION_IN_COLLECT,
    eps_start: float = DEFAULT_EPS_START,
    eps_end: float = DEFAULT_EPS_END,
    eps_decay: int = DEFAULT_EPS_DECAY,
    fixed_temperature_value: float = DEFAULT_FIXED_TEMPERATURE_VALUE,
    td_steps: int | None = DEFAULT_TD_STEPS,
    num_unroll_steps: int | None = DEFAULT_NUM_UNROLL_STEPS,
    discount_factor: float | None = DEFAULT_DISCOUNT_FACTOR,
    support_scale: int | None = DEFAULT_SUPPORT_SCALE,
    reward_support_min: float | None = DEFAULT_REWARD_SUPPORT_MIN,
    reward_support_max: float | None = DEFAULT_REWARD_SUPPORT_MAX,
    reward_support_delta: float | None = DEFAULT_REWARD_SUPPORT_DELTA,
    value_support_min: float | None = DEFAULT_VALUE_SUPPORT_MIN,
    value_support_max: float | None = DEFAULT_VALUE_SUPPORT_MAX,
    value_support_delta: float | None = DEFAULT_VALUE_SUPPORT_DELTA,
) -> dict[str, Any]:
    return _run_lightzero_dummy_pong_config_import_smoke(
        mode=mode,
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
        pong_reset_profile=pong_reset_profile,
        pong_reset_pressure_agent=pong_reset_pressure_agent,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        n_episode=n_episode,
        game_segment_length=game_segment_length,
        random_collect_episode_num=random_collect_episode_num,
        eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        fixed_temperature_value=fixed_temperature_value,
        td_steps=td_steps,
        num_unroll_steps=num_unroll_steps,
        discount_factor=discount_factor,
        support_scale=support_scale,
        reward_support_min=reward_support_min,
        reward_support_max=reward_support_max,
        reward_support_delta=reward_support_delta,
        value_support_min=value_support_min,
        value_support_max=value_support_max,
        value_support_delta=value_support_delta,
    )


@app.local_entrypoint()
def main(
    mode: str = "config-import",
    env: str = DEFAULT_ENV,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    opponent_policy: str = DEFAULT_OPPONENT_POLICY,
    seed: int = 0,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    pong_episode_max_steps: int | None = DEFAULT_PONG_EPISODE_MAX_STEPS,
    pong_reset_profile: str = DEFAULT_PONG_RESET_PROFILE,
    pong_reset_pressure_agent: str = DEFAULT_PONG_RESET_PRESSURE_AGENT,
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM,
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT,
    n_episode: int = DEFAULT_N_EPISODE,
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH,
    random_collect_episode_num: int = DEFAULT_RANDOM_COLLECT_EPISODE_NUM,
    eps_greedy_exploration_in_collect: bool = DEFAULT_EPS_GREEDY_EXPLORATION_IN_COLLECT,
    eps_start: float = DEFAULT_EPS_START,
    eps_end: float = DEFAULT_EPS_END,
    eps_decay: int = DEFAULT_EPS_DECAY,
    fixed_temperature_value: float = DEFAULT_FIXED_TEMPERATURE_VALUE,
    td_steps: int | None = DEFAULT_TD_STEPS,
    num_unroll_steps: int | None = DEFAULT_NUM_UNROLL_STEPS,
    discount_factor: float | None = DEFAULT_DISCOUNT_FACTOR,
    support_scale: int | None = DEFAULT_SUPPORT_SCALE,
    reward_support_min: float | None = DEFAULT_REWARD_SUPPORT_MIN,
    reward_support_max: float | None = DEFAULT_REWARD_SUPPORT_MAX,
    reward_support_delta: float | None = DEFAULT_REWARD_SUPPORT_DELTA,
    value_support_min: float | None = DEFAULT_VALUE_SUPPORT_MIN,
    value_support_max: float | None = DEFAULT_VALUE_SUPPORT_MAX,
    value_support_delta: float | None = DEFAULT_VALUE_SUPPORT_DELTA,
) -> None:
    result = lightzero_dummy_pong_config_import_smoke.remote(
        mode=mode,
        env=env,
        feature_mode=feature_mode,
        opponent_policy=opponent_policy,
        seed=seed,
        max_env_step=max_env_step,
        pong_episode_max_steps=pong_episode_max_steps,
        pong_reset_profile=pong_reset_profile,
        pong_reset_pressure_agent=pong_reset_pressure_agent,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        num_simulations=num_simulations,
        batch_size=batch_size,
        update_per_collect=update_per_collect,
        n_episode=n_episode,
        game_segment_length=game_segment_length,
        random_collect_episode_num=random_collect_episode_num,
        eps_greedy_exploration_in_collect=eps_greedy_exploration_in_collect,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        fixed_temperature_value=fixed_temperature_value,
        td_steps=td_steps,
        num_unroll_steps=num_unroll_steps,
        discount_factor=discount_factor,
        support_scale=support_scale,
        reward_support_min=reward_support_min,
        reward_support_max=reward_support_max,
        reward_support_delta=reward_support_delta,
        value_support_min=value_support_min,
        value_support_max=value_support_max,
        value_support_delta=value_support_delta,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
