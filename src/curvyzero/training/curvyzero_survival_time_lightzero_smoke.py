"""No-train survival-time LightZero smoke adapter for ``CurvyTronEnv``.

This module keeps the proven scalar/ray LightZero surface from
``CurvyZeroLightZeroLocalSmokeEnv`` and changes only the reward contract. The
survival reward is +1 for a transition whose post-step ego player is alive, and
0 otherwise. There is no terminal outcome bonus and no loser penalty.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from typing import Any

import numpy as np

from curvyzero.env import CurvyTronConfig
from curvyzero.env import trainer_contract as contract
from curvyzero.env.state import EnvState
from curvyzero.training.curvyzero_lightzero_smoke import (
    CurvyZeroLightZeroLocalSmokeEnv,
)
from curvyzero.training.curvyzero_lightzero_smoke import (
    FixedActionOpponentPolicy,
)
from curvyzero.training.curvyzero_lightzero_smoke import (
    LocalLightZeroTimestep,
)
from curvyzero.training.curvyzero_lightzero_smoke import (
    optional_base_env_timestep_cls,
)
from curvyzero.training.curvyzero_lightzero_smoke import (
    to_base_env_timestep,
)


LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE = "curvyzero_survival_time_lightzero_local_smoke"
LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID = "CurvyZeroSurvivalTimeLightZeroLocalSmoke-v0"
SURVIVAL_TIME_ADAPTER_IMPL_ID = "curvyzero_survival_time_lightzero_local_smoke_adapter/v0"
SURVIVAL_TIME_REWARD_SCHEMA_ID = "curvyzero_survival_time/v0"
SURVIVAL_TIME_REWARD_SCHEMA = {
    "schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
    "dtype": contract.REWARD_DTYPE,
    "episode_unit": "one_round",
    "perspective": "ego_player",
    "alignment": "reward_t_plus_1_after_wrapper_decision_t",
    "reward_unit": "one_fixed_wrapper_decision_step",
    "terminal_step_counting_rule": "post_transition_alive",
    "post_transition_alive_reward": 1.0,
    "post_transition_dead_reward": 0.0,
    "terminal_outcome_bonus": 0.0,
    "loser_penalty": 0.0,
    "winner_bonus": 0.0,
    "draw_bonus": 0.0,
    "truncation_bonus": 0.0,
    "episode_return": "sum of post-transition alive rewards for the ego player",
}
SURVIVAL_TIME_REWARD_SCHEMA_HASH = contract.stable_contract_hash(
    SURVIVAL_TIME_REWARD_SCHEMA
)
SURVIVAL_TIME_TRAINER_ADAPTER_CONTRACT_HASH = contract.stable_contract_hash(
    {
        "adapter": {
            **contract.TRAINER_ADAPTER_CONTRACT,
            "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
        },
        "action_space": contract.ACTION_SPACE_SCHEMA,
        "observation": contract.OBSERVATION_SCHEMA,
        "reward": SURVIVAL_TIME_REWARD_SCHEMA,
    }
)


class CurvyZeroSurvivalTimeLightZeroLocalSmokeEnv(CurvyZeroLightZeroLocalSmokeEnv):
    """Single-ego scalar LightZero smoke env with survival-time rewards."""

    config = dict(CurvyZeroLightZeroLocalSmokeEnv.config)
    config.update(
        {
            "env_id": LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID,
            "lightzero_env_type": LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE,
            "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
            "survival_terminal_step_counting_rule": "post_transition_alive",
        }
    )

    def __init__(self, cfg: Any | None = None):
        super().__init__(cfg)
        self.env_id = str(
            _cfg_get(cfg or {}, "env_id", LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID)
        )

    def _trainer_observation(
        self,
        player_id: str,
        *,
        needs_reset: bool | None = None,
    ):
        observed = super()._trainer_observation(player_id, needs_reset=needs_reset)
        assert self._env.state is not None
        reward, reward_info = survival_time_reward_v0(
            self._env.state,
            self._env.config,
            player_id,
            player_ids=self._env.agents,
        )
        return replace(observed, reward=reward, reward_info=reward_info)

    def _base_info(self) -> dict[str, Any]:
        info = super()._base_info()
        info.update(
            {
                "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
                "reward_schema_hash": SURVIVAL_TIME_REWARD_SCHEMA_HASH,
                "trainer_adapter_contract_hash": (
                    SURVIVAL_TIME_TRAINER_ADAPTER_CONTRACT_HASH
                ),
                "adapter_impl_id": SURVIVAL_TIME_ADAPTER_IMPL_ID,
                "lightzero_adapter_kind": "local_survival_time_no_train_smoke",
                "survival_terminal_step_counting_rule": "post_transition_alive",
            }
        )
        return info

    def _trace_hash(self) -> str:
        payload = {
            "adapter_impl_id": SURVIVAL_TIME_ADAPTER_IMPL_ID,
            "episode_seed": self._episode_seed,
            "ego_player_id": self.ego_player_id,
            "opponent_player_id": self.opponent_player_id,
            "opponent_policy_id": self.opponent_policy.policy_id,
            "opponent_policy_version": self.opponent_policy.policy_version,
            "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
            "trace": self._action_trace,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            "CurvyZeroSurvivalTimeLightZeroLocalSmokeEnv("
            f"env_id={self.env_id!r}, "
            f"ego_player_id={self.ego_player_id!r}, "
            f"opponent_player_id={self.opponent_player_id!r}, "
            f"opponent_policy_id={self.opponent_policy.policy_id!r})"
        )


def survival_time_reward_v0(
    state: EnvState,
    config: CurvyTronConfig,
    ego_player_id: str | int,
    *,
    player_ids: tuple[str, ...] | list[str] | None = None,
) -> tuple[np.float32, dict[str, Any]]:
    """Return post-transition-alive survival reward and terminal metadata."""

    ids = tuple(player_ids or ("player_0", "player_1"))
    if len(ids) != 2:
        raise ValueError(f"player_ids must contain exactly 2 ids, got {len(ids)}")
    ego_idx = _player_index(ego_player_id, ids)
    alive = np.asarray(state.alive, dtype=np.bool_)
    if alive.shape != (2,):
        raise ValueError(f"{SURVIVAL_TIME_REWARD_SCHEMA_ID} requires alive shape (2,)")

    alive_count = int(alive.sum())
    terminated = alive_count <= 1
    truncated = bool(config.max_ticks > 0 and int(state.tick) >= int(config.max_ticks))
    done = terminated or truncated

    terminal_reason = "none"
    winner_ids: tuple[str, ...] = ()
    loser_ids: tuple[str, ...] = ()
    draw = False
    if terminated and alive_count == 1:
        winner_idx = int(np.flatnonzero(alive)[0])
        terminal_reason = "survivor_win"
        winner_ids = (ids[winner_idx],)
        loser_ids = tuple(
            player_id for index, player_id in enumerate(ids) if index != winner_idx
        )
    elif terminated:
        terminal_reason = "all_dead_draw"
        draw = True
    elif truncated:
        terminal_reason = "timeout"

    reward = np.float32(1.0 if bool(alive[ego_idx]) else 0.0)
    reward_info = {
        "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
        "reward_schema_hash": SURVIVAL_TIME_REWARD_SCHEMA_HASH,
        "observation_schema_id": contract.OBSERVATION_SCHEMA_ID,
        "observation_schema_hash": contract.OBSERVATION_SCHEMA_HASH,
        "ego_player_id": ids[ego_idx],
        "terminal_reason": terminal_reason,
        "winner_ids": winner_ids,
        "loser_ids": loser_ids,
        "draw": draw,
        "timeout": truncated,
        "truncation_reason": "max_ticks" if truncated else None,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "survival_terminal_step_counting_rule": "post_transition_alive",
    }
    return reward, reward_info


def run_survival_time_lightzero_smoke(seed: int = 0) -> dict[str, Any]:
    """Run a tiny no-train smoke for the survival-time scalar wrapper."""

    nonterminal_env = CurvyZeroSurvivalTimeLightZeroLocalSmokeEnv(
        {"env_config": CurvyTronConfig(action_repeat=1), "seed": seed}
    )
    reset_obs = nonterminal_env.reset(seed=seed)
    first_step = nonterminal_env.step(1)

    terminal_config = CurvyTronConfig(
        width=10,
        height=30,
        max_ticks=100,
        action_repeat=1,
        speed=9.1,
        turn_rate_radians=float(np.pi / 2),
        spawn_margin=1.0,
    )
    terminal_env = CurvyZeroSurvivalTimeLightZeroLocalSmokeEnv(
        {"env_config": terminal_config, "seed": seed}
    )
    terminal_env.reset(seed=5)
    terminal_step = terminal_env.step(1)

    problems: list[str] = []
    problems.extend(_validate_reset(reset_obs))
    if first_step.reward != 1.0:
        problems.append(f"first nonterminal survival reward was {first_step.reward!r}")
    if first_step.info["reward_schema_id"] != SURVIVAL_TIME_REWARD_SCHEMA_ID:
        problems.append("nonterminal step did not report survival reward schema")
    if terminal_step.reward != 0.0:
        problems.append(f"ego-death terminal reward was {terminal_step.reward!r}")
    if terminal_step.info["final_reward_map"] != {"player_0": 0.0, "player_1": 0.0}:
        problems.append(
            f"ego-death final_reward_map was {terminal_step.info['final_reward_map']!r}"
        )
    if terminal_step.info["eval_episode_return"] != 0.0:
        problems.append(
            f"ego-death eval_episode_return was {terminal_step.info['eval_episode_return']!r}"
        )

    return {
        "ok": not problems,
        "mode": "no_train_local_survival_time_reset_step_only",
        "problems": problems,
        "env_type": LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE,
        "env_id": LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID,
        "reward_schema_id": SURVIVAL_TIME_REWARD_SCHEMA_ID,
        "reward_schema_hash": SURVIVAL_TIME_REWARD_SCHEMA_HASH,
        "survival_terminal_step_counting_rule": "post_transition_alive",
        "reset": _summarize_observation(reset_obs),
        "first_step": _summarize_timestep(first_step),
        "ego_death_terminal_step": _summarize_timestep(terminal_step),
    }


def _validate_reset(obs: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if set(obs) != {"observation", "action_mask", "to_play", "timestep"}:
        problems.append(f"reset keys were {sorted(obs)}")
    observation = np.asarray(obs.get("observation"))
    action_mask = np.asarray(obs.get("action_mask"))
    if tuple(observation.shape) != contract.LIGHTZERO_FLAT_OBSERVATION_SHAPE:
        problems.append("reset observation shape drifted")
    if observation.dtype != np.float32:
        problems.append("reset observation dtype drifted")
    if tuple(action_mask.shape) != (len(contract.ACTION_NAMES),):
        problems.append("reset action_mask shape drifted")
    if action_mask.dtype != np.int8:
        problems.append("reset action_mask dtype drifted")
    if obs.get("to_play") != -1:
        problems.append("reset to_play drifted")
    return problems


def _summarize_timestep(timestep: LocalLightZeroTimestep) -> dict[str, Any]:
    return {
        "reward": float(timestep.reward),
        "done": bool(timestep.done),
        "obs": _summarize_observation(timestep.obs),
        "info": {
            "reward_schema_id": timestep.info.get("reward_schema_id"),
            "terminal_reason": timestep.info.get("terminal_reason"),
            "winner_ids": list(timestep.info.get("winner_ids", ())),
            "loser_ids": list(timestep.info.get("loser_ids", ())),
            "death_player_ids": list(timestep.info.get("death_player_ids", ())),
            "terminated": timestep.info.get("terminated"),
            "truncated": timestep.info.get("truncated"),
            "final_reward_map": timestep.info.get("final_reward_map"),
            "eval_episode_return": timestep.info.get("eval_episode_return"),
            "trace_hash": timestep.info.get("trace_hash"),
        },
    }


def _summarize_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    observation = np.asarray(obs.get("observation"))
    action_mask = np.asarray(obs.get("action_mask"))
    return {
        "keys": sorted(str(key) for key in obs),
        "observation_shape": [int(item) for item in observation.shape],
        "observation_dtype": str(observation.dtype),
        "action_mask_shape": [int(item) for item in action_mask.shape],
        "action_mask_dtype": str(action_mask.dtype),
        "action_mask_values": action_mask.tolist(),
        "to_play": int(obs.get("to_play")),
        "timestep": int(obs.get("timestep")),
    }


def _player_index(ego_player_id: str | int, player_ids: tuple[str, ...]) -> int:
    if isinstance(ego_player_id, int):
        if 0 <= ego_player_id < len(player_ids):
            return int(ego_player_id)
        raise ValueError(f"unknown player index {ego_player_id!r}")
    try:
        return player_ids.index(ego_player_id)
    except ValueError as exc:
        raise ValueError(f"unknown player {ego_player_id!r}") from exc


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def main() -> None:
    result = run_survival_time_lightzero_smoke()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "CurvyZeroSurvivalTimeLightZeroLocalSmokeEnv",
    "FixedActionOpponentPolicy",
    "LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_ID",
    "LIGHTZERO_CURVYZERO_SURVIVAL_TIME_ENV_TYPE",
    "LocalLightZeroTimestep",
    "SURVIVAL_TIME_ADAPTER_IMPL_ID",
    "SURVIVAL_TIME_REWARD_SCHEMA",
    "SURVIVAL_TIME_REWARD_SCHEMA_HASH",
    "SURVIVAL_TIME_REWARD_SCHEMA_ID",
    "SURVIVAL_TIME_TRAINER_ADAPTER_CONTRACT_HASH",
    "optional_base_env_timestep_cls",
    "run_survival_time_lightzero_smoke",
    "survival_time_reward_v0",
    "to_base_env_timestep",
]
