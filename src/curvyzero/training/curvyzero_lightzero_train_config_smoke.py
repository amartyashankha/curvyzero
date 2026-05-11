"""Dry scalar CurvyTron LightZero trainer-plumbing config scaffold.

This module intentionally does not train. Its default local path does not
import LightZero or DI-engine; it builds and validates the tiny scalar/ray
config surface Coach would use for a first CurvyTron ``train_muzero`` plumbing
smoke. When an installed LightZero runtime is available, callers can require
the real CartPole MLP template and optionally run DI-engine ``compile_config``
inspection before any trainer entrypoint is called.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import is_dataclass
import importlib
import json
from pathlib import Path
import time
import traceback
from typing import Any

from curvyzero.env import trainer_contract as contract
from curvyzero.training.curvyzero_lightzero_runtime_probe import (
    CURVYZERO_LIGHTZERO_ENV_ID,
    CURVYZERO_LIGHTZERO_ENV_TYPE,
    CURVYZERO_LIGHTZERO_IMPORT,
)


DEFAULT_TEMPLATE_MODULE = "zoo.classic_control.cartpole.config.cartpole_muzero_config"
DEFAULT_MAX_ENV_STEP = 8
DEFAULT_MAX_TRAIN_ITER = 1
DEFAULT_COLLECTOR_ENV_NUM = 1
DEFAULT_EVALUATOR_ENV_NUM = 1
DEFAULT_N_EVALUATOR_EPISODE = 1
DEFAULT_NUM_SIMULATIONS = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_UPDATE_PER_COLLECT = 1
DEFAULT_N_EPISODE = 1
DEFAULT_GAME_SEGMENT_LENGTH = 8
DEFAULT_RANDOM_COLLECT_EPISODE_NUM = 0
DEFAULT_FIXED_TEMPERATURE_VALUE = 0.25
DEFAULT_SUPPORT_SCALE = 5
DEFAULT_EPS_START = 1.0
DEFAULT_EPS_END = 0.05
DEFAULT_EPS_DECAY = 100_000


@dataclass(frozen=True, slots=True)
class CurvyZeroLightZeroTrainSmokeRequest:
    """Tiny scalar config request for a trainer-plumbing-only smoke."""

    seed: int = 0
    exp_name: str | None = None
    max_env_step: int = DEFAULT_MAX_ENV_STEP
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER
    collector_env_num: int = DEFAULT_COLLECTOR_ENV_NUM
    evaluator_env_num: int = DEFAULT_EVALUATOR_ENV_NUM
    n_evaluator_episode: int = DEFAULT_N_EVALUATOR_EPISODE
    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    batch_size: int = DEFAULT_BATCH_SIZE
    update_per_collect: int = DEFAULT_UPDATE_PER_COLLECT
    n_episode: int = DEFAULT_N_EPISODE
    game_segment_length: int = DEFAULT_GAME_SEGMENT_LENGTH
    random_collect_episode_num: int = DEFAULT_RANDOM_COLLECT_EPISODE_NUM
    fixed_temperature_value: float = DEFAULT_FIXED_TEMPERATURE_VALUE
    support_scale: int = DEFAULT_SUPPORT_SCALE
    dynamic_seed: bool = False
    ego_player_id: str = "player_0"
    opponent_action_id: int = 1
    cuda: bool = False
    eps_start: float = DEFAULT_EPS_START
    eps_end: float = DEFAULT_EPS_END
    eps_decay: int = DEFAULT_EPS_DECAY
    template_module: str = DEFAULT_TEMPLATE_MODULE
    lightzero_env_type: str = CURVYZERO_LIGHTZERO_ENV_TYPE
    lightzero_env_import: str = CURVYZERO_LIGHTZERO_IMPORT
    lightzero_env_id: str = CURVYZERO_LIGHTZERO_ENV_ID
    reward_schema_id: str = contract.REWARD_SCHEMA_ID
    opponent_policy_id: str = "curvyzero_fixed_action_opponent"
    target_boundary: str = "scalar/ray curvyzero_v0_lightzero only"
    target_surface: str = "scalar_rays_only"

    @property
    def resolved_exp_name(self) -> str:
        if self.exp_name is not None:
            return self.exp_name
        return str(Path("/tmp") / "curvyzero-lightzero-curvytron-scalar" / f"seed-{self.seed}")


def build_curvyzero_lightzero_train_smoke_report(
    request: CurvyZeroLightZeroTrainSmokeRequest | None = None,
    *,
    require_lightzero_template: bool = False,
    compile_installed_lightzero: bool = False,
    include_configs: bool = False,
) -> dict[str, Any]:
    """Build a dry scalar trainer-smoke report without calling ``train_muzero``."""

    request = request or CurvyZeroLightZeroTrainSmokeRequest()
    patched = build_curvyzero_lightzero_train_smoke_configs(
        request,
        require_lightzero_template=require_lightzero_template,
    )
    validation_problems = validate_curvyzero_lightzero_train_smoke_surface(
        patched["patched_surface"],
        request=request,
    )
    compiled_config: dict[str, Any]
    if compile_installed_lightzero:
        compiled_config = compile_installed_lightzero_config_summary(
            patched["main_config"],
            patched["create_config"],
            seed=request.seed,
        )
        validation_problems.extend(
            _validate_compiled_config_summary(
                compiled_config,
                surface=patched["patched_surface"],
            )
        )
    else:
        compiled_config = {
            "ok": True,
            "skipped": True,
            "reason": "compile_installed_lightzero=false",
        }

    result: dict[str, Any] = {
        "ok": not validation_problems and bool(compiled_config.get("ok", False)),
        "label": "CurvyTron scalar LightZero MuZero dry trainer-plumbing scaffold",
        "mode": "dry_config_builder_validator_only",
        "call_policy": "does_not_train; does_not_call_lzero_entrypoints",
        "quality_claim": "none",
        "source_fidelity_claim": "none",
        "trainer_entrypoint": "lzero.entry.train_muzero",
        "called_train_muzero": False,
        "target_boundary": request.target_boundary,
        "request": asdict(request),
        "template": patched["template"],
        "patched_surface": patched["patched_surface"],
        "validation_problems": validation_problems,
        "compiled_config": compiled_config,
        "notes": [
            "This is a Coach-facing config scaffold, not a learning result.",
            "Two-simulation search is trainer plumbing only and must not be reported as quality.",
            "The visual/debug adapter lane is intentionally not touched by this helper.",
        ],
    }
    if include_configs:
        result["configs"] = {
            "main_config": patched["main_config"],
            "create_config": patched["create_config"],
            "patches": patched["patches"],
        }
    return _to_plain(result)


def build_curvyzero_lightzero_train_smoke_configs(
    request: CurvyZeroLightZeroTrainSmokeRequest,
    *,
    require_lightzero_template: bool = False,
) -> dict[str, Any]:
    """Build patched ``main_config``/``create_config`` for dry inspection."""

    template = _load_template(request.template_module, require_lightzero_template)
    main_config = copy.deepcopy(template["main_config"])
    create_config = copy.deepcopy(template["create_config"])
    patches = _apply_curvyzero_scalar_train_smoke_patches(main_config, create_config, request)
    return {
        "template": template["metadata"],
        "main_config": main_config,
        "create_config": create_config,
        "patched_surface": extract_curvyzero_lightzero_train_smoke_surface(
            main_config,
            create_config,
            request=request,
        ),
        "patches": patches,
    }


def extract_curvyzero_lightzero_train_smoke_surface(
    main_config: Any,
    create_config: Any,
    *,
    request: CurvyZeroLightZeroTrainSmokeRequest,
) -> dict[str, Any]:
    """Return the scalar LightZero trainer-smoke fields Coach should inspect."""

    env_cfg = _cfg_get(main_config, "env", {})
    policy_cfg = _cfg_get(main_config, "policy", {})
    model_cfg = _cfg_get(policy_cfg, "model", {})
    create_env = _cfg_get(create_config, "env", {})
    create_env_manager = _cfg_get(create_config, "env_manager", {})
    create_policy = _cfg_get(create_config, "policy", {})
    support_scale = _cfg_get(model_cfg, "support_scale", None)
    return {
        "env_type": _cfg_get(create_env, "type", None),
        "env_import_names": _to_plain(_cfg_get(create_env, "import_names", ())),
        "env_manager_type": _cfg_get(create_env_manager, "type", None),
        "policy_type": _cfg_get(create_policy, "type", None),
        "policy_import_names": _to_plain(_cfg_get(create_policy, "import_names", ())),
        "env_id": _cfg_get(env_cfg, "env_id", None),
        "model_type": _cfg_get(model_cfg, "model_type", None),
        "observation_shape": _to_plain(_cfg_get(model_cfg, "observation_shape", None)),
        "flat_observation_shape": list(contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE),
        "observation_schema_id": contract.OBSERVATION_SCHEMA_ID,
        "action_space_size": _cfg_get(model_cfg, "action_space_size", None),
        "action_space_id": contract.ACTION_SPACE_ID,
        "reward_schema_id": request.reward_schema_id,
        "frame_stack_num": _cfg_get(env_cfg, "frame_stack_num", None),
        "collector_env_num": _cfg_get(env_cfg, "collector_env_num", None),
        "policy_collector_env_num": _cfg_get(policy_cfg, "collector_env_num", None),
        "evaluator_env_num": _cfg_get(env_cfg, "evaluator_env_num", None),
        "policy_evaluator_env_num": _cfg_get(policy_cfg, "evaluator_env_num", None),
        "n_evaluator_episode": _cfg_get(env_cfg, "n_evaluator_episode", None),
        "num_simulations": _cfg_get(policy_cfg, "num_simulations", None),
        "batch_size": _cfg_get(policy_cfg, "batch_size", None),
        "update_per_collect": _cfg_get(policy_cfg, "update_per_collect", None),
        "n_episode": _cfg_get(policy_cfg, "n_episode", None),
        "game_segment_length": _cfg_get(policy_cfg, "game_segment_length", None),
        "random_collect_episode_num": _cfg_get(policy_cfg, "random_collect_episode_num", None),
        "fixed_temperature_value": _cfg_get(policy_cfg, "fixed_temperature_value", None),
        "eps_start": _cfg_get(_cfg_get(policy_cfg, "eps", {}), "start", None),
        "eps_end": _cfg_get(_cfg_get(policy_cfg, "eps", {}), "end", None),
        "eps_decay": _cfg_get(_cfg_get(policy_cfg, "eps", {}), "decay", None),
        "support_scale": support_scale,
        "reward_support_size": _cfg_get(model_cfg, "reward_support_size", None),
        "value_support_size": _cfg_get(model_cfg, "value_support_size", None),
        "reward_support_range": _to_plain(_cfg_get(model_cfg, "reward_support_range", None)),
        "value_support_range": _to_plain(_cfg_get(model_cfg, "value_support_range", None)),
        "cuda": _cfg_get(policy_cfg, "cuda", None),
        "max_env_step": request.max_env_step,
        "max_train_iter": request.max_train_iter,
        "dynamic_seed": _cfg_get(env_cfg, "dynamic_seed", None),
        "ego_player_id": _cfg_get(env_cfg, "ego_player_id", None),
        "opponent_action_id": _cfg_get(env_cfg, "opponent_action_id", None),
        "opponent_policy_id": request.opponent_policy_id,
        "exp_name": str(_cfg_get(main_config, "exp_name", "")),
        "target_surface": request.target_surface,
        "visual_adapter_touched": False,
        "train_quality_claim": "none",
    }


def validate_curvyzero_lightzero_train_smoke_surface(
    surface: dict[str, Any],
    *,
    request: CurvyZeroLightZeroTrainSmokeRequest,
) -> list[str]:
    """Return config problems before Coach attempts any trainer call."""

    problems: list[str] = []
    expected = {
        "env_type": request.lightzero_env_type,
        "env_import_names": [request.lightzero_env_import],
        "env_manager_type": "base",
        "policy_type": "muzero",
        "policy_import_names": ["lzero.policy.muzero"],
        "env_id": request.lightzero_env_id,
        "model_type": "mlp",
        "observation_shape": contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE[0],
        "action_space_size": len(contract.ACTION_NAMES),
        "reward_schema_id": request.reward_schema_id,
        "frame_stack_num": 1,
        "collector_env_num": request.collector_env_num,
        "policy_collector_env_num": request.collector_env_num,
        "evaluator_env_num": request.evaluator_env_num,
        "policy_evaluator_env_num": request.evaluator_env_num,
        "n_evaluator_episode": request.n_evaluator_episode,
        "num_simulations": request.num_simulations,
        "batch_size": request.batch_size,
        "update_per_collect": request.update_per_collect,
        "n_episode": request.n_episode,
        "game_segment_length": request.game_segment_length,
        "random_collect_episode_num": request.random_collect_episode_num,
        "support_scale": request.support_scale,
        "cuda": request.cuda,
        "max_env_step": request.max_env_step,
        "max_train_iter": request.max_train_iter,
        "dynamic_seed": request.dynamic_seed,
        "ego_player_id": request.ego_player_id,
        "opponent_action_id": request.opponent_action_id,
        "opponent_policy_id": request.opponent_policy_id,
        "target_surface": request.target_surface,
        "visual_adapter_touched": False,
        "train_quality_claim": "none",
    }
    for key, value in expected.items():
        if surface.get(key) != value:
            problems.append(f"{key}={surface.get(key)!r}, expected {value!r}")

    support_size = _support_size_from_scale(request.support_scale)
    if surface.get("reward_support_size") != support_size:
        problems.append(
            f"reward_support_size={surface.get('reward_support_size')!r}, "
            f"expected {support_size!r}"
        )
    if surface.get("value_support_size") != support_size:
        problems.append(
            f"value_support_size={surface.get('value_support_size')!r}, "
            f"expected {support_size!r}"
        )
    support_range = list(_support_range_from_scale(request.support_scale))
    if surface.get("reward_support_range") != support_range:
        problems.append(
            f"reward_support_range={surface.get('reward_support_range')!r}, "
            f"expected {support_range!r}"
        )
    if surface.get("value_support_range") != support_range:
        problems.append(
            f"value_support_range={surface.get('value_support_range')!r}, "
            f"expected {support_range!r}"
        )
    for key in (
        "collector_env_num",
        "evaluator_env_num",
        "n_evaluator_episode",
        "num_simulations",
        "batch_size",
        "update_per_collect",
        "n_episode",
        "game_segment_length",
        "max_env_step",
        "max_train_iter",
    ):
        try:
            value = int(surface.get(key, 0))
        except (TypeError, ValueError):
            problems.append(f"{key}={surface.get(key)!r} must be an integer")
            continue
        if value < 1:
            problems.append(f"{key} must be at least 1")
    disallowed = _disallowed_active_identity_hits(surface)
    if disallowed:
        problems.append(f"active config identity contains non-CurvyTron markers: {disallowed}")
    return problems


def compile_installed_lightzero_config_summary(
    main_config: Any,
    create_config: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    """Compile the dry config in an installed DI-engine runtime, but do not train."""

    started = time.perf_counter()
    try:
        from ding.config import compile_config

        compiled = compile_config(
            copy.deepcopy(main_config),
            seed=seed,
            auto=True,
            create_cfg=copy.deepcopy(create_config),
            save_cfg=False,
        )
        env_cfg = getattr(compiled, "env", {})
        policy_cfg = getattr(compiled, "policy", {})
        model_cfg = _cfg_get(policy_cfg, "model", {})
        return {
            "ok": True,
            "call_policy": "does_not_train; compile_config_only",
            "env_cfg": {
                "type": _cfg_get(env_cfg, "type", None),
                "env_id": _cfg_get(env_cfg, "env_id", None),
                "collector_env_num": _cfg_get(env_cfg, "collector_env_num", None),
                "evaluator_env_num": _cfg_get(env_cfg, "evaluator_env_num", None),
                "frame_stack_num": _cfg_get(env_cfg, "frame_stack_num", None),
                "dynamic_seed": _cfg_get(env_cfg, "dynamic_seed", None),
                "ego_player_id": _cfg_get(env_cfg, "ego_player_id", None),
                "opponent_action_id": _cfg_get(env_cfg, "opponent_action_id", None),
            },
            "policy_model_cfg": {
                "model_type": _cfg_get(model_cfg, "model_type", None),
                "observation_shape": _to_plain(_cfg_get(model_cfg, "observation_shape", None)),
                "action_space_size": _cfg_get(model_cfg, "action_space_size", None),
                "support_scale": _cfg_get(model_cfg, "support_scale", None),
                "reward_support_size": _cfg_get(model_cfg, "reward_support_size", None),
                "value_support_size": _cfg_get(model_cfg, "value_support_size", None),
                "reward_support_range": _to_plain(
                    _cfg_get(model_cfg, "reward_support_range", None)
                ),
                "value_support_range": _to_plain(_cfg_get(model_cfg, "value_support_range", None)),
            },
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:  # pragma: no cover - installed-runtime diagnosis.
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }


def _apply_curvyzero_scalar_train_smoke_patches(
    main_config: Any,
    create_config: Any,
    request: CurvyZeroLightZeroTrainSmokeRequest,
) -> list[dict[str, Any]]:
    support_size = _support_size_from_scale(request.support_scale)
    support_range = _support_range_from_scale(request.support_scale)
    env_cfg = {
        **_to_plain(_cfg_get(main_config, "env", {})),
        "type": request.lightzero_env_type,
        "import_names": [request.lightzero_env_import],
        "env_id": request.lightzero_env_id,
        "collector_env_num": request.collector_env_num,
        "evaluator_env_num": request.evaluator_env_num,
        "n_evaluator_episode": request.n_evaluator_episode,
        "seed": request.seed,
        "dynamic_seed": request.dynamic_seed,
        "ego_player_id": request.ego_player_id,
        "opponent_action_id": request.opponent_action_id,
        "frame_stack_num": 1,
        "observation_shape": contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE[0],
        "continuous": False,
        "manually_discretization": False,
    }
    patches = [
        _set_or_add_path(main_config, ("exp_name",), request.resolved_exp_name),
        _set_or_add_path(main_config, ("env",), env_cfg),
        _set_or_add_path(main_config, ("policy", "cuda"), request.cuda),
        _set_or_add_path(main_config, ("policy", "env_type"), "not_board_games"),
        _set_or_add_path(main_config, ("policy", "collector_env_num"), request.collector_env_num),
        _set_or_add_path(main_config, ("policy", "evaluator_env_num"), request.evaluator_env_num),
        _set_or_add_path(main_config, ("policy", "n_episode"), request.n_episode),
        _set_or_add_path(
            main_config,
            ("policy", "game_segment_length"),
            request.game_segment_length,
        ),
        _set_or_add_path(main_config, ("policy", "num_simulations"), request.num_simulations),
        _set_or_add_path(main_config, ("policy", "batch_size"), request.batch_size),
        _set_or_add_path(
            main_config,
            ("policy", "update_per_collect"),
            request.update_per_collect,
        ),
        _set_or_add_path(
            main_config,
            ("policy", "random_collect_episode_num"),
            request.random_collect_episode_num,
        ),
        _set_or_add_path(
            main_config,
            ("policy", "fixed_temperature_value"),
            request.fixed_temperature_value,
        ),
        _set_or_add_path(main_config, ("policy", "eps", "start"), request.eps_start),
        _set_or_add_path(main_config, ("policy", "eps", "end"), request.eps_end),
        _set_or_add_path(main_config, ("policy", "eps", "decay"), request.eps_decay),
        _set_or_add_path(main_config, ("policy", "eval_freq"), 1),
        _set_or_add_path(main_config, ("policy", "model", "model_type"), "mlp"),
        _set_or_add_path(
            main_config,
            ("policy", "model", "observation_shape"),
            contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE[0],
        ),
        _set_or_add_path(
            main_config,
            ("policy", "model", "action_space_size"),
            len(contract.ACTION_NAMES),
        ),
        _set_or_add_path(
            main_config,
            ("policy", "model", "support_scale"),
            request.support_scale,
        ),
        _set_or_add_path(
            main_config,
            ("policy", "model", "reward_support_size"),
            support_size,
        ),
        _set_or_add_path(
            main_config,
            ("policy", "model", "value_support_size"),
            support_size,
        ),
        _set_or_add_path(
            main_config,
            ("policy", "model", "reward_support_range"),
            support_range,
        ),
        _set_or_add_path(
            main_config,
            ("policy", "model", "value_support_range"),
            support_range,
        ),
        _set_or_add_path(create_config, ("env", "type"), request.lightzero_env_type),
        _set_or_add_path(create_config, ("env", "import_names"), [request.lightzero_env_import]),
        _set_or_add_path(create_config, ("env_manager", "type"), "base"),
        _set_or_add_path(create_config, ("policy", "type"), "muzero"),
        _set_or_add_path(create_config, ("policy", "import_names"), ["lzero.policy.muzero"]),
    ]
    return patches


def _load_template(module_name: str, require_lightzero_template: bool) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        return {
            "main_config": module.main_config,
            "create_config": module.create_config,
            "metadata": {
                "status": "installed_lightzero_template_loaded",
                "module": module_name,
                "fallback": False,
            },
        }
    except Exception as exc:
        if require_lightzero_template:
            raise RuntimeError(
                "installed LightZero CartPole MLP template is required for this dry "
                f"scaffold but could not be imported from {module_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return {
            "main_config": _fallback_main_config(),
            "create_config": _fallback_create_config(),
            "metadata": {
                "status": "fallback_surface_only_template",
                "module": module_name,
                "fallback": True,
                "import_error_type": type(exc).__name__,
                "import_error": str(exc),
                "trainable_config_status": "not_verified_without_installed_lightzero_template",
            },
        }


def _fallback_main_config() -> dict[str, Any]:
    return {
        "exp_name": "",
        "env": {},
        "policy": {
            "cuda": False,
            "env_type": "not_board_games",
            "collector_env_num": DEFAULT_COLLECTOR_ENV_NUM,
            "evaluator_env_num": DEFAULT_EVALUATOR_ENV_NUM,
            "n_episode": DEFAULT_N_EPISODE,
            "game_segment_length": DEFAULT_GAME_SEGMENT_LENGTH,
            "num_simulations": DEFAULT_NUM_SIMULATIONS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "update_per_collect": DEFAULT_UPDATE_PER_COLLECT,
            "random_collect_episode_num": DEFAULT_RANDOM_COLLECT_EPISODE_NUM,
            "fixed_temperature_value": DEFAULT_FIXED_TEMPERATURE_VALUE,
            "model_path": None,
            "eval_freq": 1,
            "eps": {
                "start": DEFAULT_EPS_START,
                "end": DEFAULT_EPS_END,
                "decay": DEFAULT_EPS_DECAY,
            },
            "model": {},
        },
    }


def _fallback_create_config() -> dict[str, Any]:
    return {
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


def _validate_compiled_config_summary(
    compiled_config: dict[str, Any],
    *,
    surface: dict[str, Any],
) -> list[str]:
    if not compiled_config.get("ok", False):
        return [f"installed LightZero compile_config failed: {compiled_config.get('error')}"]
    if compiled_config.get("skipped"):
        return []
    problems: list[str] = []
    env_cfg = compiled_config.get("env_cfg", {})
    model_cfg = compiled_config.get("policy_model_cfg", {})
    for key in ("type", "env_id", "frame_stack_num", "dynamic_seed", "opponent_action_id"):
        surface_key = "env_type" if key == "type" else key
        if env_cfg.get(key) != surface.get(surface_key):
            problems.append(
                f"compiled env.{key}={env_cfg.get(key)!r}, "
                f"expected patched surface {surface.get(surface_key)!r}"
            )
    for key in (
        "model_type",
        "observation_shape",
        "action_space_size",
        "support_scale",
        "reward_support_size",
        "value_support_size",
        "reward_support_range",
        "value_support_range",
    ):
        if model_cfg.get(key) != surface.get(key):
            problems.append(
                f"compiled policy.model.{key}={model_cfg.get(key)!r}, "
                f"expected patched surface {surface.get(key)!r}"
            )
    return problems


def _support_size_from_scale(support_scale: int) -> int:
    support_scale = int(support_scale)
    if support_scale < 1:
        raise ValueError("support_scale must be at least 1")
    return support_scale * 2 + 1


def _support_range_from_scale(support_scale: int) -> tuple[float, float, float]:
    support_scale = int(support_scale)
    return (-float(support_scale), float(support_scale + 1), 1.0)


def _set_or_add_path(mapping: Any, path: tuple[str, ...], value: Any) -> dict[str, Any]:
    current = mapping
    for part in path[:-1]:
        child = _cfg_get(current, part, None)
        if child is None:
            child = {}
            current[part] = child
        current = child
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


def _cfg_get(mapping: Any, key: str, default: Any) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


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


def _disallowed_active_identity_hits(surface: dict[str, Any]) -> list[str]:
    values = [
        surface.get("env_type"),
        surface.get("env_id"),
        surface.get("env_import_names"),
        surface.get("target_surface"),
    ]
    hits: list[str] = []
    for value in values:
        text = str(value).lower()
        for token in ("cartpole", "atari", "ale", "pongnoframeskip", "debug_visual", "occupancy"):
            if token in text:
                hits.append(f"{token}:{value}")
    return hits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-env-step", type=int, default=DEFAULT_MAX_ENV_STEP)
    parser.add_argument("--max-train-iter", type=int, default=DEFAULT_MAX_TRAIN_ITER)
    parser.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--support-scale", type=int, default=DEFAULT_SUPPORT_SCALE)
    parser.add_argument("--require-lightzero-template", action="store_true")
    parser.add_argument("--compile-installed-lightzero", action="store_true")
    parser.add_argument("--include-configs", action="store_true")
    args = parser.parse_args(argv)

    request = CurvyZeroLightZeroTrainSmokeRequest(
        seed=args.seed,
        max_env_step=args.max_env_step,
        max_train_iter=args.max_train_iter,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        support_scale=args.support_scale,
    )
    report = build_curvyzero_lightzero_train_smoke_report(
        request,
        require_lightzero_template=args.require_lightzero_template,
        compile_installed_lightzero=args.compile_installed_lightzero,
        include_configs=args.include_configs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
