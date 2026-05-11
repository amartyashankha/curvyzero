"""DI-engine registration wrapper for the CurvyZero LightZero boundary.

This module intentionally reuses ``CurvyZeroLightZeroLocalSmokeEnv`` so the
registered env cannot drift into a second set of CurvyTron semantics. In a
LightZero/DI-engine runtime it registers ``curvyzero_v0_lightzero`` and returns
real ``BaseEnvTimestep`` rows. In the local dev environment it remains
importable and uses small fallback classes for smoke tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from curvyzero.env import trainer_contract as contract
from curvyzero.training.curvyzero_lightzero_smoke import (
    CurvyZeroLightZeroLocalSmokeEnv,
)


try:  # Imported inside the Modal LightZero image.
    import gym
    from ding.envs import BaseEnv
    from ding.envs import BaseEnvTimestep
    from ding.utils import ENV_REGISTRY
except ImportError as exc:  # pragma: no cover - local tree can compile without LightZero.
    _LIGHTZERO_IMPORT_ERROR: ImportError | None = exc
    gym = None

    class BaseEnv:  # type: ignore[no-redef]
        pass

    class BaseEnvTimestep:  # type: ignore[no-redef]
        def __init__(self, obs: Any, reward: float, done: bool, info: dict[str, Any]):
            self.obs = obs
            self.reward = reward
            self.done = done
            self.info = info

    class _MissingEnvRegistry:
        def register(self, _name: str):
            def decorator(cls):
                return cls

            return decorator

    ENV_REGISTRY = _MissingEnvRegistry()
else:  # pragma: no cover - exercised only when LightZero/DI-engine is installed.
    _LIGHTZERO_IMPORT_ERROR = None


LIGHTZERO_CURVYZERO_ENV_TYPE = "curvyzero_v0_lightzero"
LIGHTZERO_CURVYZERO_ENV_ID = "CurvyZeroLightZero-v0"
LIGHTZERO_CURVYZERO_IMPORT_NAMES = ("curvyzero.training.curvyzero_lightzero_env",)


@ENV_REGISTRY.register(LIGHTZERO_CURVYZERO_ENV_TYPE)
class CurvyZeroLightZeroEnv(CurvyZeroLightZeroLocalSmokeEnv, BaseEnv):
    """Registered single-ego LightZero wrapper around the local CurvyZero smoke env."""

    config = dict(CurvyZeroLightZeroLocalSmokeEnv.config)
    config.update(
        {
            "env_id": LIGHTZERO_CURVYZERO_ENV_ID,
            "lightzero_env_type": LIGHTZERO_CURVYZERO_ENV_TYPE,
            "lightzero_import_names": LIGHTZERO_CURVYZERO_IMPORT_NAMES,
        }
    )

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        super().__init__(cfg)
        self.env_id = str(_cfg_get(cfg, "env_id", LIGHTZERO_CURVYZERO_ENV_ID))
        self.lightzero_env_type = LIGHTZERO_CURVYZERO_ENV_TYPE
        if gym is not None:
            self._action_space = gym.spaces.Discrete(len(contract.ACTION_NAMES))
            self._observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE,
                dtype=np.float32,
            )
            self._reward_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(),
                dtype=np.float32,
            )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    def step(self, action: Any) -> BaseEnvTimestep:
        local_timestep = super().step(action)
        return local_timestep.to_base_env_timestep(BaseEnvTimestep)

    def __repr__(self) -> str:
        return (
            "CurvyZeroLightZeroEnv("
            f"env_id={self.env_id!r}, "
            f"ego_player_id={self.ego_player_id!r}, "
            f"opponent_player_id={self.opponent_player_id!r}, "
            f"opponent_policy_id={self.opponent_policy.policy_id!r})"
        )


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


__all__ = [
    "BaseEnvTimestep",
    "CurvyZeroLightZeroEnv",
    "LIGHTZERO_CURVYZERO_ENV_ID",
    "LIGHTZERO_CURVYZERO_IMPORT_NAMES",
    "LIGHTZERO_CURVYZERO_ENV_TYPE",
]
