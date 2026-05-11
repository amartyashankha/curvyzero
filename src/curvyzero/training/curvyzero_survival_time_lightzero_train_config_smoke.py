"""Dry CurvyTron survival-time LightZero trainer-plumbing config scaffold.

This module intentionally does not train. It reuses the scalar/ray MuZero dry
config builder, but points the active env identity and reward schema at the
separate survival-time wrapper.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from curvyzero.training.curvyzero_lightzero_train_config_smoke import (
    DEFAULT_BATCH_SIZE,
)
from curvyzero.training.curvyzero_lightzero_train_config_smoke import (
    DEFAULT_MAX_ENV_STEP,
)
from curvyzero.training.curvyzero_lightzero_train_config_smoke import (
    DEFAULT_MAX_TRAIN_ITER,
)
from curvyzero.training.curvyzero_lightzero_train_config_smoke import (
    DEFAULT_NUM_SIMULATIONS,
)
from curvyzero.training.curvyzero_lightzero_train_config_smoke import (
    CurvyZeroLightZeroTrainSmokeRequest,
)
from curvyzero.training.curvyzero_lightzero_train_config_smoke import (
    build_curvyzero_lightzero_train_smoke_report,
)
from curvyzero.training.curvyzero_survival_time_lightzero_env import (
    LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID,
)
from curvyzero.training.curvyzero_survival_time_lightzero_env import (
    LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE,
)
from curvyzero.training.curvyzero_survival_time_lightzero_env import (
    LIGHTZERO_CURVYZERO_SURVIVAL_TIME_IMPORT_NAMES,
)
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    SURVIVAL_TIME_REWARD_SCHEMA_ID,
)


DEFAULT_SURVIVAL_SUPPORT_SCALE = 8
SURVIVAL_TIME_OPPONENT_POLICY_ID = "curvyzero_fixed_action_opponent"


def build_curvyzero_survival_time_lightzero_train_smoke_request(
    *,
    seed: int = 0,
    exp_name: str | None = None,
    max_env_step: int = DEFAULT_MAX_ENV_STEP,
    max_train_iter: int = DEFAULT_MAX_TRAIN_ITER,
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    support_scale: int = DEFAULT_SURVIVAL_SUPPORT_SCALE,
) -> CurvyZeroLightZeroTrainSmokeRequest:
    """Return the tiny dry config request for the survival-time wrapper."""

    if exp_name is None:
        exp_name = str(
            Path("/tmp")
            / "curvyzero-lightzero-curvytron-survival-time"
            / f"seed-{int(seed)}"
        )
    import_names = tuple(LIGHTZERO_CURVYZERO_SURVIVAL_TIME_IMPORT_NAMES)
    if len(import_names) != 1:
        raise ValueError(
            "survival-time LightZero env must expose exactly one import name for "
            "this dry config smoke"
        )
    return CurvyZeroLightZeroTrainSmokeRequest(
        seed=seed,
        exp_name=exp_name,
        max_env_step=max_env_step,
        max_train_iter=max_train_iter,
        num_simulations=num_simulations,
        batch_size=batch_size,
        support_scale=support_scale,
        lightzero_env_type=LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE,
        lightzero_env_import=import_names[0],
        lightzero_env_id=LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID,
        reward_schema_id=SURVIVAL_TIME_REWARD_SCHEMA_ID,
        opponent_policy_id=SURVIVAL_TIME_OPPONENT_POLICY_ID,
        target_boundary="scalar/ray curvyzero_survival_time_lightzero only",
        target_surface="scalar_rays_survival_time_only",
    )


def build_curvyzero_survival_time_lightzero_train_smoke_report(
    request: CurvyZeroLightZeroTrainSmokeRequest | None = None,
    *,
    require_lightzero_template: bool = False,
    compile_installed_lightzero: bool = False,
    include_configs: bool = False,
) -> dict[str, Any]:
    """Build a dry survival-time trainer config report without calling trainers."""

    request = request or build_curvyzero_survival_time_lightzero_train_smoke_request()
    report = build_curvyzero_lightzero_train_smoke_report(
        request,
        require_lightzero_template=require_lightzero_template,
        compile_installed_lightzero=compile_installed_lightzero,
        include_configs=include_configs,
    )
    report["label"] = "CurvyTron survival-time LightZero dry trainer-plumbing scaffold"
    report["reward_policy"] = {
        "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
        "survival_only": True,
        "terminal_outcome_bonus": 0.0,
        "loser_penalty": 0.0,
        "winner_bonus": 0.0,
    }
    report["request"] = asdict(request)
    report["notes"].append(
        "The active env is the separate survival-time wrapper; sparse outcome "
        "reward is not used."
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-env-step", type=int, default=DEFAULT_MAX_ENV_STEP)
    parser.add_argument("--max-train-iter", type=int, default=DEFAULT_MAX_TRAIN_ITER)
    parser.add_argument("--num-simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--support-scale", type=int, default=DEFAULT_SURVIVAL_SUPPORT_SCALE)
    parser.add_argument("--require-lightzero-template", action="store_true")
    parser.add_argument("--compile-installed-lightzero", action="store_true")
    parser.add_argument("--include-configs", action="store_true")
    args = parser.parse_args(argv)

    request = build_curvyzero_survival_time_lightzero_train_smoke_request(
        seed=args.seed,
        max_env_step=args.max_env_step,
        max_train_iter=args.max_train_iter,
        num_simulations=args.num_simulations,
        batch_size=args.batch_size,
        support_scale=args.support_scale,
    )
    report = build_curvyzero_survival_time_lightzero_train_smoke_report(
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


__all__ = [
    "DEFAULT_SURVIVAL_SUPPORT_SCALE",
    "SURVIVAL_TIME_OPPONENT_POLICY_ID",
    "build_curvyzero_survival_time_lightzero_train_smoke_report",
    "build_curvyzero_survival_time_lightzero_train_smoke_request",
]
