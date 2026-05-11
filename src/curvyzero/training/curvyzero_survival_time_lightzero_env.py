"""DI-engine registration wrapper for CurvyZero survival-time LightZero env."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvyzero.env import trainer_contract as contract
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    CurvyZeroSurvivalTimeLightZeroLocalSmokeEnv,
)
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID as LOCAL_SURVIVAL_ENV_ID,
)
from curvyzero.training.curvyzero_survival_time_lightzero_smoke import (
    LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE as LOCAL_SURVIVAL_ENV_TYPE,
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


LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE = "curvyzero_survival_time_lightzero"
LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID = "CurvyZeroSurvivalTimeLightZero-v0"
LIGHTZERO_CURVYZERO_SURVIVAL_TIME_IMPORT_NAMES = (
    "curvyzero.training.curvyzero_survival_time_lightzero_env",
)


@ENV_REGISTRY.register(LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE)
class CurvyZeroSurvivalTimeLightZeroEnv(
    CurvyZeroSurvivalTimeLightZeroLocalSmokeEnv,
    BaseEnv,
):
    """Registered scalar LightZero wrapper with survival-time rewards."""

    config = dict(CurvyZeroSurvivalTimeLightZeroLocalSmokeEnv.config)
    config.update(
        {
            "env_id": LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID,
            "lightzero_env_type": LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE,
            "lightzero_import_names": LIGHTZERO_CURVYZERO_SURVIVAL_TIME_IMPORT_NAMES,
            "local_smoke_env_id": LOCAL_SURVIVAL_ENV_ID,
            "local_smoke_env_type": LOCAL_SURVIVAL_ENV_TYPE,
        }
    )

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        super().__init__(cfg)
        self.env_id = str(
            _cfg_get(cfg, "env_id", LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID)
        )
        self.lightzero_env_type = LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE
        if gym is not None:
            self._action_space = gym.spaces.Discrete(len(contract.ACTION_NAMES))
            self._observation_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE,
                dtype=np.float32,
            )
            self._reward_space = gym.spaces.Box(
                low=0.0,
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
            "CurvyZeroSurvivalTimeLightZeroEnv("
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
    "CurvyZeroSurvivalTimeLightZeroEnv",
    "LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID",
    "LIGHTZERO_CURVYZERO_SURVIVAL_TIME_IMPORT_NAMES",
    "LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE",
]
