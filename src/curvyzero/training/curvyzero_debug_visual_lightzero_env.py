"""DI-engine registration wrapper for the debug visual CurvyZero boundary."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvyzero.training.curvyzero_debug_visual_lightzero_smoke import (
    CurvyZeroDebugVisualLightZeroLocalSmokeEnv,
)
from curvyzero.training.curvytron_visual_observation import DEBUG_OCCUPANCY_GRAY64_SHAPE


try:  # Imported inside a LightZero/DI-engine runtime.
    import gym
    from ding.envs import BaseEnv
    from ding.envs import BaseEnvTimestep
    from ding.utils import ENV_REGISTRY
except ImportError as exc:  # pragma: no cover - local tree can compile without DI-engine.
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


LIGHTZERO_DEBUG_VISUAL_ENV_TYPE = "curvyzero_debug_visual_tensor_lightzero"
LIGHTZERO_DEBUG_VISUAL_ENV_ID = "CurvyZeroDebugVisualTensorLightZero-v0"
LIGHTZERO_DEBUG_VISUAL_IMPORT_NAMES = (
    "curvyzero.training.curvyzero_debug_visual_lightzero_env",
)


@ENV_REGISTRY.register(LIGHTZERO_DEBUG_VISUAL_ENV_TYPE)
class CurvyZeroDebugVisualLightZeroEnv(CurvyZeroDebugVisualLightZeroLocalSmokeEnv, BaseEnv):
    """Registered single-ego debug visual LightZero wrapper."""

    config = dict(CurvyZeroDebugVisualLightZeroLocalSmokeEnv.config)
    config.update(
        {
            "env_id": LIGHTZERO_DEBUG_VISUAL_ENV_ID,
            "lightzero_env_type": LIGHTZERO_DEBUG_VISUAL_ENV_TYPE,
            "lightzero_import_names": LIGHTZERO_DEBUG_VISUAL_IMPORT_NAMES,
        }
    )

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        super().__init__(cfg)
        self.env_id = str(_cfg_get(cfg, "env_id", LIGHTZERO_DEBUG_VISUAL_ENV_ID))
        self.lightzero_env_type = LIGHTZERO_DEBUG_VISUAL_ENV_TYPE
        if gym is not None:
            self._action_space = gym.spaces.Discrete(3)
            self._observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=DEBUG_OCCUPANCY_GRAY64_SHAPE,
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
            "CurvyZeroDebugVisualLightZeroEnv("
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
    "CurvyZeroDebugVisualLightZeroEnv",
    "LIGHTZERO_DEBUG_VISUAL_ENV_ID",
    "LIGHTZERO_DEBUG_VISUAL_IMPORT_NAMES",
    "LIGHTZERO_DEBUG_VISUAL_ENV_TYPE",
]
