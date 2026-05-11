"""Opt-in survival-shaped Atari env for LightZero control ablations.

The stock LightZero Atari env remains registered as ``atari_lightzero``. This
module registers a separate env type so shaped runs cannot silently mix with
official/control Atari Pong results.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

_LIGHTZERO_IMPORT_ERROR: ImportError | None = None
try:  # Imported inside the Modal LightZero image.
    from ding.envs import BaseEnvTimestep
    from ding.utils import ENV_REGISTRY
    from zoo.atari.envs.atari_lightzero_env import AtariEnvLightZero
except ImportError as exc:  # pragma: no cover - local tree can compile without LightZero.
    _LIGHTZERO_IMPORT_ERROR = exc

    class BaseEnvTimestep:  # type: ignore[no-redef]
        def __init__(self, obs: Any, reward: Any, done: bool, info: dict[str, Any]):
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

    class AtariEnvLightZero:  # type: ignore[no-redef]
        config: dict[str, Any] = {}


LIGHTZERO_ATARI_SURVIVAL_ENV_TYPE = "atari_lightzero_survival_shaped"
REWARD_SHAPING_SCHEMA_ID = "curvyzero_lightzero_atari_survival_step_reward/v1"


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def _reward_scalar(reward: Any) -> float:
    array = np.asarray(reward, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(array.reshape(-1)[0])


@ENV_REGISTRY.register(LIGHTZERO_ATARI_SURVIVAL_ENV_TYPE)
class AtariSurvivalShapedEnv(AtariEnvLightZero):
    """LightZero Atari env with a small per-surviving-step reward bonus."""

    config = copy.deepcopy(getattr(AtariEnvLightZero, "config", {}))
    config.update(
        {
            "survival_reward_per_step": 0.0,
            "survival_reward_apply_on_done": False,
            "reward_shaping_schema_id": REWARD_SHAPING_SCHEMA_ID,
        }
    )

    def __init__(self, cfg: Any) -> None:
        if _LIGHTZERO_IMPORT_ERROR is not None:
            raise ImportError(
                "AtariSurvivalShapedEnv requires LightZero/DI-engine runtime packages"
            ) from _LIGHTZERO_IMPORT_ERROR
        super().__init__(cfg)
        self.survival_reward_per_step = float(_cfg_get(cfg, "survival_reward_per_step", 0.0))
        if self.survival_reward_per_step < 0.0:
            raise ValueError("survival_reward_per_step must be non-negative")
        self.survival_reward_apply_on_done = bool(
            _cfg_get(cfg, "survival_reward_apply_on_done", False)
        )
        self.reward_shaping_schema_id = str(
            _cfg_get(cfg, "reward_shaping_schema_id", REWARD_SHAPING_SCHEMA_ID)
        )
        self._episode_survival_bonus = 0.0

    def reset(self) -> dict[str, Any]:
        self._episode_survival_bonus = 0.0
        return super().reset()

    def step(self, action: int) -> BaseEnvTimestep:
        timestep = super().step(action)
        base_reward = timestep.reward
        apply_bonus = self.survival_reward_apply_on_done or not bool(timestep.done)
        bonus = self.survival_reward_per_step if apply_bonus else 0.0
        shaped_reward = (np.asarray(base_reward, dtype=np.float32) + bonus).astype(np.float32)
        self._episode_survival_bonus += float(bonus)
        self.reward = shaped_reward
        self._eval_episode_return = self._eval_episode_return + bonus

        info = dict(timestep.info)
        if bool(timestep.done):
            info["eval_episode_return"] = self._eval_episode_return
        info["curvyzero_reward_shaping"] = {
            "schema_id": self.reward_shaping_schema_id,
            "mode": "per_step_survival",
            "base_reward": _reward_scalar(base_reward),
            "survival_reward_per_step": self.survival_reward_per_step,
            "survival_reward_apply_on_done": self.survival_reward_apply_on_done,
            "applied_bonus": float(bonus),
            "episode_survival_bonus": float(self._episode_survival_bonus),
            "shaped_reward": _reward_scalar(shaped_reward),
        }
        return BaseEnvTimestep(timestep.obs, shaped_reward, bool(timestep.done), info)
