"""No-train LightZero-shaped debug visual adapter for ``CurvyTronSourceEnv``.

This path is intentionally separate from the scalar ``curvyzero_v0_lightzero``
adapter. It renders the current source-env snapshot into the coarse debug
occupancy tensor and exposes a LightZero-style observation dict for smoke and
runtime plumbing checks only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from curvyzero.env import CurvyTronSourceEnv
from curvyzero.env import trainer_contract as contract
from curvyzero.env.config import CurvyTronReferenceDefaults
from curvyzero.training.curvytron_visual_observation import (
    DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
    DEBUG_OCCUPANCY_GRAY64_RAW_VALUE_RANGE,
    DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
    DEBUG_OCCUPANCY_GRAY64_SHAPE,
    DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL,
    DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY,
    DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE,
    DEBUG_OCCUPANCY_GRAY64_SURFACE,
    DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL,
    DEBUG_OCCUPANCY_GRAY64_USES_ALE,
    DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE,
    DebugOccupancyGray64Renderer,
    debug_occupancy_gray64_metadata,
    debug_occupancy_gray64_schema,
    normalize_debug_occupancy_gray64_for_lightzero,
)


LIGHTZERO_DEBUG_VISUAL_ENV_TYPE = "curvyzero_debug_visual_tensor_lightzero"
LIGHTZERO_DEBUG_VISUAL_ENV_ID = "CurvyZeroDebugVisualTensorLightZero-v0"
ADAPTER_IMPL_ID = "curvyzero_debug_visual_tensor_lightzero_smoke_adapter/v0"
OPPONENT_POLICY_ID = "curvyzero_debug_visual_fixed_straight_opponent"
OPPONENT_POLICY_VERSION = "v0.2026-05-10"
SOURCE_TIMER_ACTIVATION_POLICY_ID = "source_warmup_zero_activate_then_elapsed_decision_step/v0"
DEFAULT_SOURCE_STEP_MS = CurvyTronReferenceDefaults().tick_ms
ACTION_ID_TO_SOURCE_MOVE = contract.ACTION_ID_TO_SOURCE_MOVE
DEATH_CAUSE_NONE = 0
DEATH_CAUSE_WALL = 1
DEATH_CAUSE_OWN_TRAIL = 2
DEATH_CAUSE_OPPONENT_TRAIL = 3
DEATH_CAUSE_BODY_UNKNOWN = 4

_DEATH_CAUSE_BY_NAME = {
    "none": DEATH_CAUSE_NONE,
    "wall": DEATH_CAUSE_WALL,
    "own_trail": DEATH_CAUSE_OWN_TRAIL,
    "opponent_trail": DEATH_CAUSE_OPPONENT_TRAIL,
    "body_unknown": DEATH_CAUSE_BODY_UNKNOWN,
}
_DEATH_CAUSE_NAMES_BY_ID = {
    value: key for key, value in _DEATH_CAUSE_BY_NAME.items()
}


@dataclass(frozen=True, slots=True)
class LocalDebugVisualLightZeroTimestep:
    """Small local stand-in for DI-engine's ``BaseEnvTimestep``."""

    obs: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]

    def to_base_env_timestep(self, base_env_timestep_cls: Any | None = None) -> Any:
        """Convert this local row to DI-engine's timestep type when available."""

        return to_base_env_timestep(self, base_env_timestep_cls=base_env_timestep_cls)


def optional_base_env_timestep_cls() -> Any | None:
    """Return DI-engine's ``BaseEnvTimestep`` class if the runtime provides it."""

    try:
        from ding.envs import BaseEnvTimestep
    except ImportError:
        return None
    return BaseEnvTimestep


def to_base_env_timestep(
    timestep: LocalDebugVisualLightZeroTimestep,
    *,
    base_env_timestep_cls: Any | None = None,
) -> Any:
    """Build a real ``BaseEnvTimestep`` without importing DI-engine at import time."""

    cls = base_env_timestep_cls or optional_base_env_timestep_cls()
    if cls is None:
        raise ImportError(
            "ding.envs.BaseEnvTimestep is not available; install DI-engine/LightZero "
            "to convert LocalDebugVisualLightZeroTimestep into the real env-manager row type"
        )
    return cls(timestep.obs, timestep.reward, timestep.done, timestep.info)


@dataclass(frozen=True, slots=True)
class FixedStraightSourceOpponentPolicy:
    """Deterministic straight-control opponent for adapter plumbing tests."""

    action_id: int = 1
    policy_id: str = OPPONENT_POLICY_ID
    policy_version: str = OPPONENT_POLICY_VERSION

    def action(self, legal_action_mask: np.ndarray) -> int:
        mask = np.asarray(legal_action_mask, dtype=np.bool_)
        action_id = int(self.action_id)
        if action_id < 0 or action_id >= mask.shape[0]:
            raise ValueError(f"opponent action {action_id!r} is outside mask shape {mask.shape}")
        if not bool(mask[action_id]):
            raise ValueError(f"opponent action {action_id!r} is not legal under mask {mask}")
        return action_id


class CurvyZeroDebugVisualLightZeroLocalSmokeEnv:
    """Single-ego debug visual wrapper around the source-shaped CurvyTron env."""

    config = {
        "env_id": LIGHTZERO_DEBUG_VISUAL_ENV_ID,
        "ego_player_id": "player_0",
        "opponent_action_id": 1,
        "seed": 0,
        "dynamic_seed": False,
        "source_step_ms": DEFAULT_SOURCE_STEP_MS,
        "source_max_steps": 2000,
    }

    def __init__(self, cfg: Any | None = None):
        cfg = cfg or {}
        self.cfg = cfg
        self.env_id = str(_cfg_get(cfg, "env_id", LIGHTZERO_DEBUG_VISUAL_ENV_ID))
        self.ego_player_id = str(_cfg_get(cfg, "ego_player_id", "player_0"))
        self.player_ids = ("player_0", "player_1")
        if self.ego_player_id not in self.player_ids:
            raise ValueError(f"unknown ego_player_id: {self.ego_player_id!r}")
        self.opponent_player_id = _other_player(self.player_ids, self.ego_player_id)
        self._avatar_ids_by_player = {"player_0": 1, "player_1": 2}
        self._player_ids_by_avatar = {
            avatar_id: player_id for player_id, avatar_id in self._avatar_ids_by_player.items()
        }
        self.opponent_policy = FixedStraightSourceOpponentPolicy(
            action_id=int(_cfg_get(cfg, "opponent_action_id", 1))
        )
        if self.opponent_policy.action_id != 1:
            raise ValueError("debug visual smoke opponent is intentionally fixed to straight")
        self._base_seed = int(_cfg_get(cfg, "seed", 0))
        self._dynamic_seed = bool(_cfg_get(cfg, "dynamic_seed", False))
        self._source_step_ms = float(_cfg_get(cfg, "source_step_ms", DEFAULT_SOURCE_STEP_MS))
        if self._source_step_ms < 0.0:
            raise ValueError("source_step_ms must be non-negative")
        self._source_max_steps = int(_cfg_get(cfg, "source_max_steps", 2000))
        self._episode_index = 0
        self._episode_seed = self._base_seed
        self._episode_return = 0.0
        self._step_index = 0
        self._action_trace: list[dict[str, Any]] = []
        self._needs_reset = False
        self._closed = False
        self._env = self._make_source_env(seed=self._episode_seed)
        self._renderer = DebugOccupancyGray64Renderer()
        self._raw_frame = np.zeros(DEBUG_OCCUPANCY_GRAY64_SHAPE, dtype=np.uint8)
        self._normalized_frame = np.zeros(DEBUG_OCCUPANCY_GRAY64_SHAPE, dtype=np.float32)
        self._last_snapshot: dict[str, Any] | None = None
        self.last_reset_info: dict[str, Any] | None = None

        self._action_space = {"type": "Discrete", "n": len(contract.ACTION_NAMES)}
        self._observation_space = {
            "type": "Box",
            "shape": DEBUG_OCCUPANCY_GRAY64_SHAPE,
            "dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
            "low": DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE[0],
            "high": DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE[1],
        }
        self._reward_space = {"type": "Box", "shape": (), "dtype": "float32"}

    @property
    def curvytron_source_env(self) -> CurvyTronSourceEnv:
        """Expose the wrapped source env for narrow local smoke assertions."""

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
        """Reset and explicitly activate one deterministic source-backed round."""

        self._episode_seed = int(seed) if seed is not None else self._next_episode_seed()
        self._env = self._make_source_env(seed=self._episode_seed)
        snapshot = self._env.reset(
            player_count=2,
            players=self._source_players(),
            warmup_ms=0.0,
        )
        self._env.advance_timers(0.0)
        snapshot = self._env.snapshot("after_reset_timer_activation")
        self._last_snapshot = dict(snapshot)
        self._episode_return = 0.0
        self._step_index = 0
        self._action_trace = []
        self._needs_reset = False
        self._episode_index += 1
        self.last_reset_info = self._base_info(snapshot)
        self.last_reset_info.update(
            {
                "ego_player_id": self.ego_player_id,
                "opponent_player_id": self.opponent_player_id,
                "opponent_policy_id": self.opponent_policy.policy_id,
                "opponent_policy_version": self.opponent_policy.policy_version,
                "episode_index": self._episode_index - 1,
                "needs_reset": False,
                "source_reset_snapshot_label": snapshot.get("label"),
            }
        )
        return self._lightzero_observation(snapshot=snapshot, needs_reset=False)

    def step(self, action: Any) -> LocalDebugVisualLightZeroTimestep:
        """Apply one ego action, hold opponent straight, and render the next frame."""

        if self._last_snapshot is None:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError(
                "reset must be called before stepping after a terminal or truncated episode"
            )
        ego_action = self._validate_ego_action(action)
        ego_move = _action_id_to_source_move(ego_action)
        opponent_action = self._opponent_action(
            snapshot=self._last_snapshot,
            legal_action_mask=self._action_mask(active=True),
        )
        opponent_move = _action_id_to_source_move(opponent_action)
        joint_source_action = {
            self._avatar_ids_by_player[self.ego_player_id]: ego_move,
            self._avatar_ids_by_player[self.opponent_player_id]: opponent_move,
        }
        joint_action = {
            self.ego_player_id: int(ego_action),
            self.opponent_player_id: int(opponent_action),
        }

        self._env.advance_timers(0.0)
        snapshot = self._env.step(joint_source_action, elapsed_ms=self._source_step_ms)
        self._last_snapshot = dict(snapshot)
        terminated = self._terminated(snapshot)
        truncated = self._truncated()
        done = bool(terminated or truncated)
        self._needs_reset = done
        reward = self._reward(snapshot, terminated=terminated)
        self._episode_return += reward
        self._step_index += 1

        next_obs = self._lightzero_observation(snapshot=snapshot, needs_reset=done)
        self._action_trace.append(
            {
                "step_index": self._step_index - 1,
                "source_at_ms": snapshot.get("atMs"),
                "joint_action": dict(joint_action),
                "joint_source_move": {
                    player: float(
                        joint_source_action[self._avatar_ids_by_player[player]]
                    )
                    for player in joint_action
                },
                "reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        info = self._step_info(
            snapshot=snapshot,
            joint_action=joint_action,
            joint_source_action=joint_source_action,
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            next_obs=next_obs,
        )
        return LocalDebugVisualLightZeroTimestep(next_obs, reward, done, info)

    def random_action(self) -> int:
        """Return a deterministic legal action sample from the episode seed."""

        rng = np.random.default_rng(self._episode_seed + self._step_index)
        return int(rng.integers(len(contract.ACTION_NAMES)))

    def close(self) -> None:
        self._closed = True

    def render(self, mode: str = "debug_visual_tensor") -> np.ndarray | None:
        if mode != "debug_visual_tensor":
            return None
        if self._last_snapshot is None:
            return None
        return self._render_normalized(self._last_snapshot).copy()

    def __repr__(self) -> str:
        return (
            "CurvyZeroDebugVisualLightZeroLocalSmokeEnv("
            f"env_id={self.env_id!r}, "
            f"ego_player_id={self.ego_player_id!r}, "
            f"opponent_player_id={self.opponent_player_id!r}, "
            f"opponent_policy_id={self.opponent_policy.policy_id!r})"
        )

    def _next_episode_seed(self) -> int:
        if self._dynamic_seed:
            return self._base_seed + self._episode_index
        return self._base_seed

    def _make_source_env(self, *, seed: int) -> CurvyTronSourceEnv:
        return CurvyTronSourceEnv(
            random_values=_seed_random_values(seed),
            include_deaths_snapshot=True,
            drain_frame_timers=False,
            bonus_types=(),
        )

    def _source_players(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {"id": 1, "name": "player_0"},
            {"id": 2, "name": "player_1"},
        )

    def _validate_ego_action(self, action: Any) -> int:
        try:
            action_id = int(np.asarray(action).item())
        except ValueError as exc:
            raise ValueError(f"ego action must be scalar, got {action!r}") from exc
        if action_id < 0 or action_id >= len(contract.ACTION_NAMES):
            raise ValueError(f"invalid CurvyZero action id: {action_id!r}")
        if not bool(self._action_mask(active=not self._needs_reset)[action_id]):
            raise ValueError(f"ego action {action_id!r} is not legal")
        return action_id

    def _opponent_action(
        self,
        *,
        snapshot: Mapping[str, Any],
        legal_action_mask: np.ndarray,
    ) -> int:
        """Select the source opponent action for the current decision."""

        del snapshot
        return int(self.opponent_policy.action(legal_action_mask))

    def _lightzero_observation(
        self,
        *,
        snapshot: Mapping[str, Any],
        needs_reset: bool,
    ) -> dict[str, Any]:
        frame = self._render_normalized(snapshot)
        ego_alive = self._player_alive(snapshot, self.ego_player_id)
        return {
            "observation": frame.copy(),
            "action_mask": self._action_mask(active=ego_alive and not needs_reset),
            "to_play": -1,
            "timestep": int(self._step_index),
        }

    def _render_normalized(
        self,
        snapshot: Mapping[str, Any],
        *,
        controlled_player_id: str | int | None = None,
    ) -> np.ndarray:
        raw = self._renderer.render(
            snapshot,
            world_bodies=self._env.world_bodies_snapshot(),
            controlled_player_id=controlled_player_id,
            out=self._raw_frame,
        )
        return normalize_debug_occupancy_gray64_for_lightzero(raw, out=self._normalized_frame)

    def _action_mask(self, *, active: bool) -> np.ndarray:
        if active:
            return np.ones(len(contract.ACTION_NAMES), dtype=np.int8)
        return np.zeros(len(contract.ACTION_NAMES), dtype=np.int8)

    def _base_info(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        schema = debug_occupancy_gray64_schema()
        info = debug_occupancy_gray64_metadata(includes_render_cost=True)
        info.update(
            {
                "seed": self._episode_seed,
                "reset_seed": self._episode_seed,
                "dynamic_seed": self._dynamic_seed,
                "env_id": self.env_id,
                "lightzero_env_type": LIGHTZERO_DEBUG_VISUAL_ENV_TYPE,
                "adapter_impl_id": ADAPTER_IMPL_ID,
                "lightzero_adapter_kind": "debug_visual_tensor_no_train_smoke",
                "schema_hash": schema["schema_hash"],
                "observation_schema_hash": schema["schema_hash"],
                "raw_frame_shape": list(DEBUG_OCCUPANCY_GRAY64_SHAPE),
                "model_observation_shape": list(DEBUG_OCCUPANCY_GRAY64_STACK_SHAPE),
                "raw_frame_dtype": "uint8",
                "lightzero_payload_dtype": DEBUG_OCCUPANCY_GRAY64_OBSERVATION_DTYPE,
                "raw_value_range": list(DEBUG_OCCUPANCY_GRAY64_RAW_VALUE_RANGE),
                "value_range": list(DEBUG_OCCUPANCY_GRAY64_VALUE_RANGE),
                "source_visual_fidelity": DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY,
                "uses_ale": DEBUG_OCCUPANCY_GRAY64_USES_ALE,
                "ale_usage": "none",
                "surface": DEBUG_OCCUPANCY_GRAY64_SURFACE,
                "truth_level": DEBUG_OCCUPANCY_GRAY64_TRUTH_LEVEL,
                "source_fidelity_level": DEBUG_OCCUPANCY_GRAY64_SOURCE_FIDELITY_LEVEL,
                "source_backed_observation_fidelity": DEBUG_OCCUPANCY_GRAY64_SOURCE_VISUAL_FIDELITY,
                "renderer_impl_id": DEBUG_OCCUPANCY_GRAY64_RENDERER_IMPL_ID,
                "action_space_id": contract.ACTION_SPACE_ID,
                "action_space_hash": contract.ACTION_SPACE_HASH,
                "action_id_to_source_move": list(ACTION_ID_TO_SOURCE_MOVE),
                "source_control_model_id": contract.NATIVE_CONTROL_MODEL_ID,
                "frame_stack_owner": "optimizer",
                "player_ids": list(self.player_ids),
                "max_players": 2,
                "source_activation_policy_id": SOURCE_TIMER_ACTIVATION_POLICY_ID,
                "source_reset_warmup_ms": 0.0,
                "source_reset_advance_timers_ms": 0.0,
                "source_step_pre_advance_timers_ms": 0.0,
                "source_step_ms": self._source_step_ms,
                "source_timer_clock_advances_on_step": False,
                "source_max_steps": self._source_max_steps,
                "source_game_started": _snapshot_game_bool(snapshot, "started"),
                "source_game_in_round": _snapshot_game_bool(snapshot, "inRound"),
                "source_world_active": _snapshot_game_bool(snapshot, "worldActive"),
                "source_frame_scheduled": _snapshot_game_bool(snapshot, "frameScheduled"),
                "source_at_ms": snapshot.get("atMs"),
            }
        )
        return info

    def _step_info(
        self,
        *,
        snapshot: Mapping[str, Any],
        joint_action: Mapping[str, int],
        joint_source_action: Mapping[int, float],
        reward: float,
        done: bool,
        terminated: bool,
        truncated: bool,
        next_obs: dict[str, Any],
    ) -> dict[str, Any]:
        info = self._base_info(snapshot)
        winner_ids = self._winner_ids(snapshot) if terminated else ()
        loser_ids = self._loser_ids(snapshot) if terminated else ()
        terminal_reason = "survivor_win" if winner_ids else ("all_dead_draw" if terminated else "none")
        truncation_reason = "source_max_steps" if truncated else None
        info.update(
            {
                "ego_player_id": self.ego_player_id,
                "opponent_player_id": self.opponent_player_id,
                "step_index": self._step_index - 1,
                "tick_index": self._step_index,
                "adapter_timestep": self._step_index,
                "joint_action": {player: int(action) for player, action in joint_action.items()},
                "joint_source_move": {
                    player: float(joint_source_action[self._avatar_ids_by_player[player]])
                    for player in joint_action
                },
                "opponent_action_id": int(joint_action[self.opponent_player_id]),
                "opponent_policy_id": self.opponent_policy.policy_id,
                "opponent_policy_version": self.opponent_policy.policy_version,
                "terminal_reason": terminal_reason,
                "winner_ids": winner_ids,
                "loser_ids": loser_ids,
                "death_player_ids": loser_ids,
                "draw": bool(terminated and not winner_ids),
                "timeout": bool(truncated),
                "truncation_reason": truncation_reason,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
                "needs_reset": self._needs_reset,
                "final_observation": _copy_lightzero_observation(next_obs) if done else None,
                "final_reward_map": self._reward_map(snapshot, terminated=terminated)
                if done
                else None,
                "event_ref": None,
                "event_range": [0, len(self._env.events)],
                "state_ref": None,
                "trace_ref": None,
                "trace_hash": self._trace_hash(),
                "eval_episode_return": float(self._episode_return) if done else None,
                "reward": reward,
                "source_snapshot_label": snapshot.get("label"),
                "source_event_count": len(self._env.events),
            }
        )
        info.update(self._source_death_summary())
        return info

    def _source_death_summary(self) -> dict[str, Any]:
        deaths: list[dict[str, Any]] = []
        for event in self._env.events:
            if event.get("event") != "die":
                continue
            data = event.get("data")
            if not isinstance(data, Mapping):
                continue
            avatar_id = _int_or_none(data.get("avatar"))
            if avatar_id is None:
                continue
            player_id = self._player_ids_by_avatar.get(avatar_id)
            if player_id is None:
                continue
            killer_avatar_id = _int_or_none(data.get("killer"))
            hit_owner = self._player_ids_by_avatar.get(killer_avatar_id)
            if killer_avatar_id is None:
                cause_name = "wall"
            elif killer_avatar_id == avatar_id:
                cause_name = "own_trail"
            elif hit_owner is not None:
                cause_name = "opponent_trail"
            else:
                cause_name = "body_unknown"
            deaths.append(
                {
                    "player_id": player_id,
                    "player_index": self.player_ids.index(player_id),
                    "cause_name": cause_name,
                    "cause": _DEATH_CAUSE_BY_NAME[cause_name],
                    "hit_owner": -1 if hit_owner is None else self.player_ids.index(hit_owner),
                    "source_avatar": avatar_id,
                    "source_killer_avatar": killer_avatar_id,
                }
            )
        player_count = len(self.player_ids)
        death_player = [-1] * player_count
        death_cause = [DEATH_CAUSE_NONE] * player_count
        death_cause_name = [_DEATH_CAUSE_NAMES_BY_ID[DEATH_CAUSE_NONE]] * player_count
        death_hit_owner = [-1] * player_count
        death_source_avatar = [-1] * player_count
        death_source_killer_avatar = [-1] * player_count
        for slot, death in enumerate(deaths[:player_count]):
            death_player[slot] = int(death["player_index"])
            death_cause[slot] = int(death["cause"])
            death_cause_name[slot] = str(death["cause_name"])
            death_hit_owner[slot] = int(death["hit_owner"])
            death_source_avatar[slot] = int(death["source_avatar"])
            killer = death["source_killer_avatar"]
            death_source_killer_avatar[slot] = -1 if killer is None else int(killer)
        return {
            "death_count": [min(len(deaths), player_count)],
            "death_player": [death_player],
            "death_cause": [death_cause],
            "death_cause_name": [death_cause_name],
            "death_hit_owner": [death_hit_owner],
            "death_source_avatar": [death_source_avatar],
            "death_source_killer_avatar": [death_source_killer_avatar],
        }

    def _reward(self, snapshot: Mapping[str, Any], *, terminated: bool) -> float:
        if not terminated:
            return 0.0
        alive = self._player_alive(snapshot, self.ego_player_id)
        opponent_alive = self._player_alive(snapshot, self.opponent_player_id)
        if alive and not opponent_alive:
            return 1.0
        if opponent_alive and not alive:
            return -1.0
        return 0.0

    def _reward_map(self, snapshot: Mapping[str, Any], *, terminated: bool) -> dict[str, float]:
        if not terminated:
            return {player_id: 0.0 for player_id in self.player_ids}
        return {
            player_id: self._reward_for_player(snapshot, player_id)
            for player_id in self.player_ids
        }

    def _reward_for_player(self, snapshot: Mapping[str, Any], player_id: str) -> float:
        alive = self._player_alive(snapshot, player_id)
        other_alive = self._player_alive(snapshot, _other_player(self.player_ids, player_id))
        if alive and not other_alive:
            return 1.0
        if other_alive and not alive:
            return -1.0
        return 0.0

    def _terminated(self, snapshot: Mapping[str, Any]) -> bool:
        return sum(self._player_alive(snapshot, player_id) for player_id in self.player_ids) <= 1

    def _truncated(self) -> bool:
        return self._source_max_steps > 0 and self._step_index + 1 >= self._source_max_steps

    def _winner_ids(self, snapshot: Mapping[str, Any]) -> tuple[str, ...]:
        alive_players = tuple(
            player_id for player_id in self.player_ids if self._player_alive(snapshot, player_id)
        )
        return alive_players if len(alive_players) == 1 else ()

    def _loser_ids(self, snapshot: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            player_id for player_id in self.player_ids if not self._player_alive(snapshot, player_id)
        )

    def _player_alive(self, snapshot: Mapping[str, Any], player_id: str) -> bool:
        avatar_id = self._avatar_ids_by_player[player_id]
        for avatar in _snapshot_avatars(snapshot):
            if int(avatar.get("id", -1)) == avatar_id:
                return bool(avatar.get("alive", False))
        return False

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


def _seed_random_values(seed: int, count: int = 64) -> tuple[float, ...]:
    rng = np.random.default_rng(int(seed))
    return tuple(float(value) for value in rng.random(count))


def _action_id_to_source_move(action_id: int) -> float:
    return float(ACTION_ID_TO_SOURCE_MOVE[int(action_id)])


def _copy_lightzero_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        copied[key] = value.copy() if isinstance(value, np.ndarray) else value
    return copied


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _other_player(player_ids: tuple[str, ...], ego_player_id: str) -> str:
    opponents = [player_id for player_id in player_ids if player_id != ego_player_id]
    if len(opponents) != 1:
        raise ValueError(f"expected exactly one opponent for {ego_player_id!r}")
    return opponents[0]


def _snapshot_game_bool(snapshot: Mapping[str, Any], key: str) -> bool:
    game = snapshot.get("game")
    if not isinstance(game, Mapping):
        return False
    return bool(game.get(key, False))


def _snapshot_avatars(snapshot: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    avatars = snapshot.get("avatars")
    if not isinstance(avatars, tuple | list):
        return ()
    return tuple(avatar for avatar in avatars if isinstance(avatar, Mapping))


__all__ = [
    "ACTION_ID_TO_SOURCE_MOVE",
    "ADAPTER_IMPL_ID",
    "LIGHTZERO_DEBUG_VISUAL_ENV_ID",
    "LIGHTZERO_DEBUG_VISUAL_ENV_TYPE",
    "LocalDebugVisualLightZeroTimestep",
    "CurvyZeroDebugVisualLightZeroLocalSmokeEnv",
    "FixedStraightSourceOpponentPolicy",
    "OPPONENT_POLICY_ID",
    "OPPONENT_POLICY_VERSION",
    "SOURCE_TIMER_ACTIVATION_POLICY_ID",
    "optional_base_env_timestep_cls",
    "to_base_env_timestep",
]
