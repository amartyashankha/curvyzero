"""No-train LightZero-shaped smoke adapter for ``CurvyTronEnv``.

This module deliberately avoids importing LightZero, DI-engine, Gym, torch, or
the vector/source fidelity paths. It proves the local reset/step boundary that a
real DI-engine ``BaseEnv`` wrapper will need later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from curvyzero.env import CurvyTronConfig
from curvyzero.env import CurvyTronEnv
from curvyzero.env import trainer_contract as contract
from curvyzero.env.trainer_observation import observe_egocentric_rays_v0


LIGHTZERO_CURVYZERO_ENV_TYPE = "curvyzero_v0_lightzero_local_smoke"
LIGHTZERO_CURVYZERO_ENV_ID = "CurvyZeroLightZeroLocalSmoke-v0"
ADAPTER_IMPL_ID = "curvyzero_lightzero_local_smoke_adapter/v0"
OPPONENT_POLICY_ID = "curvyzero_fixed_action_opponent"
OPPONENT_POLICY_VERSION = "v0.2026-05-09"


@dataclass(frozen=True, slots=True)
class LocalLightZeroTimestep:
    """Small local stand-in for DI-engine's ``BaseEnvTimestep``."""

    obs: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]

    def to_base_env_timestep(self, base_env_timestep_cls: Any | None = None) -> Any:
        """Convert this local row to DI-engine's timestep type when it is available."""

        return to_base_env_timestep(self, base_env_timestep_cls=base_env_timestep_cls)


def optional_base_env_timestep_cls() -> Any | None:
    """Return DI-engine's ``BaseEnvTimestep`` class if the runtime provides it."""

    try:
        from ding.envs import BaseEnvTimestep
    except ImportError:
        return None
    return BaseEnvTimestep


def to_base_env_timestep(
    timestep: LocalLightZeroTimestep,
    *,
    base_env_timestep_cls: Any | None = None,
) -> Any:
    """Build a real ``BaseEnvTimestep`` without importing DI-engine at module import time."""

    cls = base_env_timestep_cls or optional_base_env_timestep_cls()
    if cls is None:
        raise ImportError(
            "ding.envs.BaseEnvTimestep is not available; install DI-engine/LightZero "
            "to convert LocalLightZeroTimestep into the real env-manager row type"
        )
    return cls(timestep.obs, timestep.reward, timestep.done, timestep.info)


@dataclass(frozen=True, slots=True)
class FixedActionOpponentPolicy:
    """Deterministic one-action opponent for adapter plumbing tests."""

    action_id: int = 1
    policy_id: str = OPPONENT_POLICY_ID
    policy_version: str = OPPONENT_POLICY_VERSION

    def action(self, legal_action_mask: np.ndarray) -> int:
        """Return the fixed action if it is legal for the live opponent."""

        mask = np.asarray(legal_action_mask, dtype=np.bool_)
        action_id = int(self.action_id)
        if action_id < 0 or action_id >= mask.shape[0]:
            raise ValueError(f"opponent action {action_id!r} is outside mask shape {mask.shape}")
        if not bool(mask[action_id]):
            raise ValueError(f"opponent action {action_id!r} is not legal under mask {mask}")
        return action_id


class CurvyZeroLightZeroLocalSmokeEnv:
    """Single-ego, LightZero-shaped wrapper around the toy ``CurvyTronEnv``."""

    config = {
        "env_id": LIGHTZERO_CURVYZERO_ENV_ID,
        "ego_player_id": "player_0",
        "opponent_action_id": 1,
        "seed": 0,
        "dynamic_seed": False,
    }

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        env_config = _cfg_get(cfg, "env_config", None) or CurvyTronConfig()
        if not isinstance(env_config, CurvyTronConfig):
            raise TypeError("env_config must be a CurvyTronConfig")
        if int(env_config.players) != 2:
            raise ValueError("CurvyZeroLightZeroLocalSmokeEnv currently supports exactly 2 players")
        if not bool(env_config.allow_straight_action):
            raise ValueError("CurvyZeroLightZeroLocalSmokeEnv requires turn3 action ids 0/1/2")

        self.cfg = cfg
        self.env_id = str(_cfg_get(cfg, "env_id", LIGHTZERO_CURVYZERO_ENV_ID))
        self._env = CurvyTronEnv(env_config)
        self.ego_player_id = str(_cfg_get(cfg, "ego_player_id", "player_0"))
        if self.ego_player_id not in self._env.agents:
            raise ValueError(f"unknown ego_player_id: {self.ego_player_id!r}")
        self.opponent_player_id = _other_player(self._env.agents, self.ego_player_id)
        self.opponent_policy = FixedActionOpponentPolicy(
            action_id=int(_cfg_get(cfg, "opponent_action_id", 1))
        )
        self._base_seed = int(_cfg_get(cfg, "seed", 0))
        self._dynamic_seed = bool(_cfg_get(cfg, "dynamic_seed", False))
        self._episode_index = 0
        self._episode_seed = self._base_seed
        self._episode_return = 0.0
        self._action_trace: list[dict[str, Any]] = []
        self._needs_reset = False
        self._closed = False
        self.last_reset_info: dict[str, Any] | None = None

        action_space_n = len(contract.ACTION_NAMES)
        observation_shape = contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE
        self._action_space = {"type": "Discrete", "n": action_space_n}
        self._observation_space = {
            "type": "Box",
            "shape": observation_shape,
            "dtype": contract.OBSERVATION_DTYPE,
        }
        self._reward_space = {"type": "Box", "shape": (), "dtype": "float32"}

    @property
    def curvyzero_env(self) -> CurvyTronEnv:
        """Expose the wrapped env for narrow local smoke assertions."""

        return self._env

    @property
    def observation_space(self) -> dict[str, Any]:
        return dict(self._observation_space)

    @property
    def action_space(self) -> dict[str, Any]:
        return dict(self._action_space)

    @property
    def reward_space(self) -> dict[str, Any]:
        return dict(self._reward_space)

    @property
    def legal_actions(self) -> np.ndarray:
        return np.arange(len(contract.ACTION_NAMES), dtype=np.int64)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """Set the base seed policy used by future ``reset()`` calls."""

        self._base_seed = int(seed)
        self._dynamic_seed = bool(dynamic_seed)
        self._episode_index = 0
        self._episode_seed = self._base_seed

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset one deterministic CurvyZero episode and return a LightZero dict."""

        self._episode_seed = int(seed) if seed is not None else self._next_episode_seed()
        self._env.reset(seed=self._episode_seed)
        self._episode_return = 0.0
        self._action_trace = []
        self._needs_reset = False
        self._episode_index += 1
        self.last_reset_info = self._base_info()
        self.last_reset_info.update(
            {
                "ego_player_id": self.ego_player_id,
                "opponent_player_id": self.opponent_player_id,
                "opponent_policy_id": self.opponent_policy.policy_id,
                "opponent_policy_version": self.opponent_policy.policy_version,
                "episode_index": self._episode_index - 1,
                "needs_reset": False,
            }
        )
        return self._lightzero_observation(needs_reset=False)

    def step(self, action: Any) -> LocalLightZeroTimestep:
        """Apply one ego action, fill opponent action, and step the joint env once."""

        if self._env.state is None:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError(
                "reset must be called before stepping after a terminal or truncated episode"
            )
        ego_action = self._validate_ego_action(action)
        joint_action = self._joint_action(ego_action)

        result = self._env.step(joint_action)
        reward_observation = self._trainer_observation(
            self.ego_player_id,
            needs_reset=any(result.terminated.values()) or any(result.truncated.values()),
        )
        reward_info = reward_observation.reward_info
        reward = float(reward_observation.reward)
        done = bool(reward_info["done"])
        self._needs_reset = done
        self._episode_return += reward

        self._action_trace.append(
            {
                "step_index": len(self._action_trace),
                "tick_index": int(self._env.state.tick),
                "joint_action": dict(joint_action),
                "reward": reward,
                "done": done,
                "terminated": bool(reward_info["terminated"]),
                "truncated": bool(reward_info["truncated"]),
            }
        )

        next_obs = self._lightzero_observation(needs_reset=done)
        info = self._step_info(
            joint_action=joint_action,
            done=done,
            reward_info=reward_info,
            env_info=result.infos[self.ego_player_id],
            next_obs=next_obs,
        )
        return LocalLightZeroTimestep(next_obs, reward, done, info)

    def random_action(self) -> int:
        """Return a deterministic legal action sample from the episode seed."""

        rng = np.random.default_rng(self._episode_seed + len(self._action_trace))
        return int(rng.integers(len(contract.ACTION_NAMES)))

    def close(self) -> None:
        self._closed = True

    def render(self, mode: str = "state_realtime_mode") -> None:
        del mode
        return None

    def __repr__(self) -> str:
        return (
            "CurvyZeroLightZeroLocalSmokeEnv("
            f"env_id={self.env_id!r}, "
            f"ego_player_id={self.ego_player_id!r}, "
            f"opponent_player_id={self.opponent_player_id!r}, "
            f"opponent_policy_id={self.opponent_policy.policy_id!r})"
        )

    def _next_episode_seed(self) -> int:
        if self._dynamic_seed:
            return self._base_seed + self._episode_index
        return self._base_seed

    def _validate_ego_action(self, action: Any) -> int:
        try:
            action_id = int(np.asarray(action).item())
        except ValueError as exc:
            raise ValueError(f"ego action must be scalar, got {action!r}") from exc

        if action_id < 0 or action_id >= len(contract.ACTION_NAMES):
            raise ValueError(f"invalid CurvyZero action id: {action_id!r}")
        ego_mask = self._trainer_observation(self.ego_player_id).action_mask
        if not bool(ego_mask[action_id]):
            raise ValueError(f"ego action {action_id!r} is not legal under mask {ego_mask}")
        return action_id

    def _joint_action(self, ego_action: int) -> dict[str, int]:
        assert self._env.state is not None
        joint_action = {self.ego_player_id: int(ego_action)}
        opponent_index = self._env.agents.index(self.opponent_player_id)
        if bool(self._env.state.alive[opponent_index]):
            opponent_mask = self._trainer_observation(self.opponent_player_id).action_mask
            joint_action[self.opponent_player_id] = self.opponent_policy.action(opponent_mask)
        return joint_action

    def _trainer_observation(
        self,
        player_id: str,
        *,
        needs_reset: bool | None = None,
    ):
        assert self._env.state is not None
        return observe_egocentric_rays_v0(
            self._env.state,
            self._env.config,
            player_id,
            player_ids=self._env.agents,
            needs_reset=self._needs_reset if needs_reset is None else bool(needs_reset),
        )

    def _lightzero_observation(self, *, needs_reset: bool) -> dict[str, Any]:
        observed = self._trainer_observation(self.ego_player_id, needs_reset=needs_reset)
        payload = observed.lightzero_payload()
        assert self._env.state is not None
        payload["timestep"] = int(self._env.state.tick)
        return payload

    def _base_info(self) -> dict[str, Any]:
        core_info = dict(self._env.last_reset_info or {})
        core_info.update(
            {
                "seed": self._episode_seed,
                "ruleset_id": self._env.config.ruleset,
                "rules_hash": self._env.config.rules_hash,
                "observation_schema_id": contract.OBSERVATION_SCHEMA_ID,
                "observation_schema_hash": contract.OBSERVATION_SCHEMA_HASH,
                "action_space_id": contract.ACTION_SPACE_ID,
                "action_space_hash": contract.ACTION_SPACE_HASH,
                "reward_schema_id": contract.REWARD_SCHEMA_ID,
                "reward_schema_hash": contract.REWARD_SCHEMA_HASH,
                "trainer_adapter_contract_id": contract.TRAINER_ADAPTER_CONTRACT_ID,
                "trainer_adapter_contract_hash": contract.TRAINER_ADAPTER_CONTRACT_HASH,
                "adapter_impl_id": ADAPTER_IMPL_ID,
                "lightzero_adapter_kind": "local_no_train_smoke",
                "player_ids": list(self._env.agents),
                "max_players": int(self._env.config.players),
            }
        )
        return core_info

    def _step_info(
        self,
        *,
        joint_action: Mapping[str, int],
        done: bool,
        reward_info: dict[str, Any],
        env_info: Mapping[str, Any],
        next_obs: dict[str, Any],
    ) -> dict[str, Any]:
        info = self._base_info()
        info.update(
            {
                "ego_player_id": self.ego_player_id,
                "opponent_player_id": self.opponent_player_id,
                "step_index": int(env_info["step_index"]),
                "tick_index": int(env_info["tick_index"]),
                "joint_action": {player: int(action) for player, action in joint_action.items()},
                "opponent_action_id": (
                    int(joint_action[self.opponent_player_id])
                    if self.opponent_player_id in joint_action
                    else None
                ),
                "opponent_policy_id": self.opponent_policy.policy_id,
                "opponent_policy_version": self.opponent_policy.policy_version,
                "terminal_reason": reward_info["terminal_reason"],
                "winner_ids": tuple(reward_info["winner_ids"]),
                "loser_ids": tuple(reward_info["loser_ids"]),
                "death_player_ids": tuple(env_info.get("death_player_ids", ())),
                "draw": bool(reward_info["draw"]),
                "timeout": bool(reward_info["timeout"]),
                "truncation_reason": reward_info["truncation_reason"],
                "done": done,
                "terminated": bool(reward_info["terminated"]),
                "truncated": bool(reward_info["truncated"]),
                "needs_reset": self._needs_reset,
                "final_observation": _copy_lightzero_observation(next_obs) if done else None,
                "final_reward_map": self._reward_map() if done else None,
                "event_ref": env_info.get("event_ref"),
                "event_range": env_info.get("event_range"),
                "state_ref": env_info.get("state_ref"),
                "trace_ref": env_info.get("trace_ref"),
                "trace_hash": self._trace_hash(),
                "eval_episode_return": float(self._episode_return) if done else None,
            }
        )
        return info

    def _reward_map(self) -> dict[str, float]:
        assert self._env.state is not None
        return {
            player_id: float(
                observe_egocentric_rays_v0(
                    self._env.state,
                    self._env.config,
                    player_id,
                    player_ids=self._env.agents,
                    needs_reset=self._needs_reset,
                ).reward
            )
            for player_id in self._env.agents
        }

    def _trace_hash(self) -> str:
        payload = {
            "adapter_impl_id": ADAPTER_IMPL_ID,
            "episode_seed": self._episode_seed,
            "ego_player_id": self.ego_player_id,
            "opponent_player_id": self.opponent_player_id,
            "opponent_policy_id": self.opponent_policy.policy_id,
            "opponent_policy_version": self.opponent_policy.policy_version,
            "trace": self._action_trace,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def _copy_lightzero_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _other_player(player_ids: list[str], ego_player_id: str) -> str:
    opponents = [player_id for player_id in player_ids if player_id != ego_player_id]
    if len(opponents) != 1:
        raise ValueError(f"expected exactly one opponent for {ego_player_id!r}")
    return opponents[0]


__all__ = [
    "ADAPTER_IMPL_ID",
    "LIGHTZERO_CURVYZERO_ENV_ID",
    "LIGHTZERO_CURVYZERO_ENV_TYPE",
    "LocalLightZeroTimestep",
    "CurvyZeroLightZeroLocalSmokeEnv",
    "FixedActionOpponentPolicy",
    "OPPONENT_POLICY_ID",
    "OPPONENT_POLICY_VERSION",
    "optional_base_env_timestep_cls",
    "to_base_env_timestep",
]
