"""DI-engine registration wrapper for stacked debug visual survival CurvyZero env."""

from __future__ import annotations

from typing import Any

import numpy as np

from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID as LOCAL_STACKED_ENV_ID,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE as LOCAL_STACKED_ENV_TYPE,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID as LOCAL_TURN_COMMIT_ENV_ID,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_smoke import (
    LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE as LOCAL_TURN_COMMIT_ENV_TYPE,
)


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


LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE = (
    "curvyzero_stacked_debug_visual_survival_lightzero"
)
LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID = (
    "CurvyZeroStackedDebugVisualSurvivalLightZero-v0"
)
LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_IMPORT_NAMES = (
    "curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env",
)
LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE = (
    "curvyzero_stacked_debug_visual_turn_commit_lightzero"
)
LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID = (
    "CurvyZeroStackedDebugVisualTurnCommitLightZero-v0"
)
LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_IMPORT_NAMES = (
    "curvyzero.training.curvyzero_stacked_debug_visual_survival_lightzero_env",
)


@ENV_REGISTRY.register(LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE)
class CurvyZeroStackedDebugVisualSurvivalLightZeroEnv(
    CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv,
    BaseEnv,
):
    """Registered debug visual wrapper with wrapper-owned frame stack."""

    config = dict(CurvyZeroStackedDebugVisualSurvivalLightZeroLocalSmokeEnv.config)
    config.update(
        {
            "env_id": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID,
            "lightzero_env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE,
            "lightzero_import_names": LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_IMPORT_NAMES,
            "local_smoke_env_id": LOCAL_STACKED_ENV_ID,
            "local_smoke_env_type": LOCAL_STACKED_ENV_TYPE,
        }
    )

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        super().__init__(cfg)
        self.env_id = str(
            _cfg_get(cfg, "env_id", LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID)
        )
        self.lightzero_env_type = LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE
        if gym is not None:
            self._action_space = gym.spaces.Discrete(3)
            self._observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
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
            "CurvyZeroStackedDebugVisualSurvivalLightZeroEnv("
            f"env_id={self.env_id!r}, "
            f"ego_player_id={self.ego_player_id!r}, "
            f"opponent_player_id={self.opponent_player_id!r}, "
            f"opponent_policy_id={self.opponent_policy.policy_id!r})"
        )


@ENV_REGISTRY.register(LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE)
class CurvyZeroStackedDebugVisualTurnCommitLightZeroEnv(
    CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv,
    BaseEnv,
):
    """Registered native LightZero turn-commit wrapper for CurvyTron."""

    config = dict(CurvyZeroStackedDebugVisualTurnCommitLightZeroLocalSmokeEnv.config)
    config.update(
        {
            "env_id": LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID,
            "lightzero_env_type": LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE,
            "lightzero_import_names": LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_IMPORT_NAMES,
            "local_smoke_env_id": LOCAL_TURN_COMMIT_ENV_ID,
            "local_smoke_env_type": LOCAL_TURN_COMMIT_ENV_TYPE,
        }
    )

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        super().__init__(cfg)
        self.env_id = str(
            _cfg_get(cfg, "env_id", LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID)
        )
        self.lightzero_env_type = LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE
        if gym is not None:
            self._action_space = gym.spaces.Discrete(3)
            self._observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
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
            "CurvyZeroStackedDebugVisualTurnCommitLightZeroEnv("
            f"env_id={self.env_id!r}, "
            f"active_player_id={self.active_player_id!r})"
        )


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


__all__ = [
    "BaseEnvTimestep",
    "CurvyZeroStackedDebugVisualSurvivalLightZeroEnv",
    "CurvyZeroStackedDebugVisualTurnCommitLightZeroEnv",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_ID",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_IMPORT_NAMES",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_SURVIVAL_ENV_TYPE",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_ID",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_IMPORT_NAMES",
    "LIGHTZERO_STACKED_DEBUG_VISUAL_TURN_COMMIT_ENV_TYPE",
]
