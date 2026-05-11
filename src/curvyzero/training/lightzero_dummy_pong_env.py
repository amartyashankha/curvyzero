"""LightZero adapter for the project-owned dummy Pong environment.

This module is intentionally small. LightZero controls one ego paddle, while
the wrapper supplies the opponent action from a scripted policy and keeps the
Pong telemetry visible in ``info``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.training.dummy_pong import ACTION_LABELS
from curvyzero.training.dummy_pong import AGENTS
from curvyzero.training.dummy_pong import OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import PONG_RESET_PROFILES
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv
from curvyzero.training.dummy_pong import PongObservation
from curvyzero.training.dummy_pong_eval import LaggedTrackBallPolicy
from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
from curvyzero.training.dummy_pong_eval import TrackBallPolicy
from curvyzero.training.lightzero_dummy_pong_features import RASTER_FLAT_FEATURE_SCHEMA_ID
from curvyzero.training.lightzero_dummy_pong_features import TABULAR_FEATURE_SCHEMA_ID
from curvyzero.training.lightzero_dummy_pong_features import encode_tabular_ego_observation
from curvyzero.training.lightzero_dummy_pong_features import lightzero_observation_shape

_LIGHTZERO_IMPORT_ERROR: ImportError | None = None
try:  # Imported inside the Modal LightZero image.
    import gym
    from ding.envs import BaseEnv
    from ding.envs import BaseEnvTimestep
    from ding.utils import ENV_REGISTRY
except ImportError as exc:  # pragma: no cover - local tree can compile without LightZero.
    _LIGHTZERO_IMPORT_ERROR = exc
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

LIGHTZERO_DUMMY_PONG_ENV_TYPE = "dummy_pong_lightzero"
LIGHTZERO_DUMMY_PONG_ENV_ID = "DummyPongLightZero-v0"
EPISODE_INFO_SCHEMA_ID = "curvyzero_lightzero_dummy_pong_episode/v1"


@ENV_REGISTRY.register(LIGHTZERO_DUMMY_PONG_ENV_TYPE)
class DummyPongLightZeroEnv(BaseEnv):
    """Single-ego LightZero wrapper around simultaneous dummy Pong."""

    config = {
        "env_id": LIGHTZERO_DUMMY_PONG_ENV_ID,
        "feature_mode": "tabular_ego",
        "opponent_policy": "random_uniform",
        "ego_agent": "player_0",
        "opponent_checkpoint_path": None,
        "opponent_checkpoint_label": None,
        "opponent_checkpoint_adapter": None,
        "opponent_checkpoint_num_simulations": 2,
        "opponent_checkpoint_source_ref": None,
        "opponent_checkpoint_sha256": None,
        "opponent_checkpoint_state_key": "model",
        "dynamic_seed": True,
        "max_steps": 120,
        "pong_episode_max_steps": None,
        "pong_reset_profile": "default",
        "pong_reset_pressure_agent": "ego",
    }

    def __init__(self, cfg: Any | None = None):
        if _LIGHTZERO_IMPORT_ERROR is not None:
            raise ImportError(
                "DummyPongLightZeroEnv requires LightZero/DI-engine runtime packages"
            ) from _LIGHTZERO_IMPORT_ERROR
        self.cfg = cfg or {}
        self.env_id = _cfg_get(self.cfg, "env_id", LIGHTZERO_DUMMY_PONG_ENV_ID)
        self.curvyzero_env = _cfg_get(self.cfg, "curvyzero_env", "dummy_pong_lag1")
        self.feature_mode = _cfg_get(self.cfg, "feature_mode", "tabular_ego")
        self.opponent_policy_id = _cfg_get(self.cfg, "opponent_policy", "random_uniform")
        self.opponent_checkpoint_path = _cfg_get(self.cfg, "opponent_checkpoint_path", None)
        self.opponent_checkpoint_label = _cfg_get(self.cfg, "opponent_checkpoint_label", None)
        self.opponent_checkpoint_adapter = _cfg_get(self.cfg, "opponent_checkpoint_adapter", None)
        opponent_checkpoint_num_simulations = _cfg_get(
            self.cfg, "opponent_checkpoint_num_simulations", 2
        )
        self.opponent_checkpoint_num_simulations = int(
            2
            if opponent_checkpoint_num_simulations is None
            else opponent_checkpoint_num_simulations
        )
        self.opponent_checkpoint_source_ref = _cfg_get(
            self.cfg, "opponent_checkpoint_source_ref", None
        )
        self.opponent_checkpoint_sha256 = _cfg_get(self.cfg, "opponent_checkpoint_sha256", None)
        self.opponent_checkpoint_state_key = _cfg_get(
            self.cfg, "opponent_checkpoint_state_key", "model"
        )
        self.opponent_checkpoint_config_opponent_policy = _cfg_get(
            self.cfg, "opponent_checkpoint_config_opponent_policy", "random_uniform"
        )
        self.ego_agent = _cfg_get(self.cfg, "ego_agent", "player_0")
        if self.ego_agent not in AGENTS:
            raise ValueError(f"unknown ego_agent: {self.ego_agent!r}")
        self.opponent_agent = _opponent(self.ego_agent)
        pong_episode_max_steps = _cfg_get(self.cfg, "pong_episode_max_steps", None)
        max_steps = (
            _cfg_get(self.cfg, "max_steps", 120)
            if pong_episode_max_steps is None
            else pong_episode_max_steps
        )
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.pong_reset_profile = str(_cfg_get(self.cfg, "pong_reset_profile", "default"))
        if self.pong_reset_profile not in PONG_RESET_PROFILES:
            raise ValueError(
                f"pong_reset_profile must be one of {PONG_RESET_PROFILES!r}, "
                f"got {self.pong_reset_profile!r}"
            )
        self.pong_reset_pressure_agent = self._resolve_reset_pressure_agent(
            _cfg_get(self.cfg, "pong_reset_pressure_agent", "ego")
        )
        self._pong_config = PongConfig(
            max_steps=self.max_steps,
            reset_profile=self.pong_reset_profile,
            reset_pressure_agent=self.pong_reset_pressure_agent,
        )
        self._env = PongEnv(self._pong_config)
        self._base_seed = int(_cfg_get(self.cfg, "seed", 0))
        self._configured_dynamic_seed = bool(_cfg_get(self.cfg, "dynamic_seed", True))
        self._dynamic_seed = self._configured_dynamic_seed
        self._last_seed_call_dynamic_seed_arg: bool | None = None
        self._last_seed_source = _seed_source(
            configured_dynamic_seed=self._configured_dynamic_seed,
            seed_call_dynamic_seed_arg=None,
        )
        self._episode_index = 0
        self._episode_seed = self._base_seed
        self._random_action_rng = np.random.default_rng(self._episode_seed + 20_000)
        self._observations: dict[str, PongObservation] | None = None
        self._opponent_checkpoint_metadata: dict[str, Any] | None = None
        self._opponent_policy = self._make_opponent_policy(seed=self._base_seed)
        telemetry_path = _cfg_get(self.cfg, "telemetry_path", None)
        self._telemetry_path = Path(telemetry_path) if telemetry_path else None
        self._episode_return = 0.0
        self._action_trace: list[dict[str, Any]] = []
        self._action_counts_by_agent = {agent: Counter() for agent in AGENTS}
        self._closed = False

        self._action_space = gym.spaces.Discrete(len(ACTION_LABELS))
        self._observation_space = self._make_observation_space()

    def reset(self) -> dict[str, Any]:
        self._episode_seed = self._next_episode_seed()
        self._observations = self._env.reset(seed=self._episode_seed)
        self._opponent_policy.reset(self._episode_seed, self.opponent_agent)
        self._random_action_rng = np.random.default_rng(self._episode_seed + 20_000)
        self._episode_return = 0.0
        self._action_trace = []
        self._action_counts_by_agent = {agent: Counter() for agent in AGENTS}
        self._episode_index += 1
        return self._lightzero_observation()

    def step(self, action: Any) -> BaseEnvTimestep:
        if self._observations is None:
            raise RuntimeError("reset must be called before step")
        ego_action = int(np.asarray(action).item())
        if ego_action < 0 or ego_action >= len(ACTION_LABELS):
            raise ValueError(f"invalid dummy Pong action: {ego_action!r}")

        raster_grid = self._env.raster_observation()
        opponent_action = int(
            self._opponent_policy.action(
                self._observations[self.opponent_agent],
                raster_grid,
                self.opponent_agent,
            )
        )
        joint_action = {
            self.ego_agent: ego_action,
            self.opponent_agent: opponent_action,
        }
        for agent, played_action in joint_action.items():
            self._action_counts_by_agent[agent][ACTION_LABELS[int(played_action)]] += 1

        step_index = int(self._observations[self.ego_agent].step)
        pong_step = self._env.step(joint_action)
        self._observations = pong_step.observations
        reward = float(pong_step.rewards[self.ego_agent])
        self._episode_return += reward
        done = bool(pong_step.terminated or pong_step.truncated)
        self._action_trace.append(
            {
                "step": step_index,
                "joint_action": dict(joint_action),
                "reward": reward,
            }
        )

        info = self._info(
            done=done,
            terminated=bool(pong_step.terminated),
            truncated=bool(pong_step.truncated),
            reward=reward,
            winner=pong_step.infos["winner"] if pong_step.terminated else None,
            last_hit=pong_step.infos["last_hit"],
            final_rewards={
                agent: float(pong_step.rewards[agent])
                for agent in AGENTS
            },
        )
        if done:
            info["eval_episode_return"] = self._episode_return
            self._write_episode_row(info["curvyzero_pong"])

        return BaseEnvTimestep(self._lightzero_observation(), reward, done, info)

    def seed(self, seed: int, dynamic_seed: bool | None = None) -> None:
        self._base_seed = int(seed)
        self._last_seed_call_dynamic_seed_arg = (
            bool(dynamic_seed) if dynamic_seed is not None else None
        )
        self._dynamic_seed = self._configured_dynamic_seed
        self._last_seed_source = _seed_source(
            configured_dynamic_seed=self._configured_dynamic_seed,
            seed_call_dynamic_seed_arg=self._last_seed_call_dynamic_seed_arg,
        )
        self._episode_index = 0
        self._episode_seed = self._base_seed
        self._random_action_rng = np.random.default_rng(self._episode_seed + 20_000)
        self._opponent_policy = self._make_opponent_policy(seed=self._base_seed)

    def close(self) -> None:
        self._closed = True

    def render(self, mode: str = "state_realtime_mode") -> None:
        del mode
        return None

    def __repr__(self) -> str:
        return (
            "DummyPongLightZeroEnv("
            f"env_id={self.env_id!r}, "
            f"curvyzero_env={self.curvyzero_env!r}, "
            f"feature_mode={self.feature_mode!r}, "
            f"opponent_policy={self.opponent_policy_id!r}, "
            f"pong_reset_profile={self.pong_reset_profile!r}, "
            f"ego_agent={self.ego_agent!r})"
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    @property
    def legal_actions(self) -> np.ndarray:
        return np.arange(len(ACTION_LABELS), dtype=np.int64)

    def random_action(self) -> int:
        return int(self._random_action_rng.integers(len(ACTION_LABELS)))

    def _make_observation_space(self):
        if self.feature_mode == "tabular_ego":
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        if self.feature_mode == "raster_flat":
            shape = (lightzero_observation_shape(self.feature_mode, self._pong_config),)
            return gym.spaces.Box(low=0.0, high=4.0, shape=shape, dtype=np.float32)
        raise ValueError(
            f"unknown feature_mode {self.feature_mode!r}; expected tabular_ego or raster_flat"
        )

    def _next_episode_seed(self) -> int:
        if self._dynamic_seed:
            return self._base_seed + self._episode_index
        return self._base_seed

    def _make_opponent_policy(self, *, seed: int):
        if self.opponent_policy_id == "random_uniform":
            return RandomUniformPolicy(seed=seed + 10_000)
        if self.opponent_policy_id == "track_ball":
            return TrackBallPolicy()
        if self.opponent_policy_id == "lagged_track_ball_1":
            return LaggedTrackBallPolicy(delay=1)
        if self.opponent_policy_id in {
            "lightzero_policy_head_checkpoint",
            "lightzero_mcts_checkpoint",
        }:
            return self._make_lightzero_checkpoint_opponent_policy(seed=seed)
        raise ValueError(f"unknown opponent_policy: {self.opponent_policy_id!r}")

    def _make_lightzero_checkpoint_opponent_policy(self, *, seed: int):
        if not self.opponent_checkpoint_path:
            raise ValueError(
                f"{self.opponent_policy_id} requires opponent_checkpoint_path"
            )
        checkpoint_path = Path(str(self.opponent_checkpoint_path))
        label = str(self.opponent_checkpoint_label or checkpoint_path.name)
        policy_id = f"frozen_checkpoint:{label}"
        adapter = self.opponent_checkpoint_adapter
        if adapter is None:
            adapter = (
                "policy_head_greedy"
                if self.opponent_policy_id == "lightzero_policy_head_checkpoint"
                else "mcts_eval_mode"
            )

        if adapter == "policy_head_greedy":
            from curvyzero.training.lightzero_dummy_pong_policy import (
                load_lightzero_policy_head_greedy_checkpoint,
            )

            spec = load_lightzero_policy_head_greedy_checkpoint(
                policy_id=policy_id,
                checkpoint_path=checkpoint_path,
                env=self.curvyzero_env,
                feature_mode=self.feature_mode,
                opponent_policy=self.opponent_checkpoint_config_opponent_policy,
                seed=seed,
                max_env_step=self.max_steps,
            )
        elif adapter == "mcts_eval_mode":
            from curvyzero.training.lightzero_dummy_pong_policy import (
                load_lightzero_mcts_eval_mode_checkpoint,
            )

            spec = load_lightzero_mcts_eval_mode_checkpoint(
                policy_id=policy_id,
                checkpoint_path=checkpoint_path,
                env=self.curvyzero_env,
                feature_mode=self.feature_mode,
                opponent_policy=self.opponent_checkpoint_config_opponent_policy,
                seed=seed,
                max_env_step=self.max_steps,
                num_simulations=self.opponent_checkpoint_num_simulations,
            )
        else:
            raise ValueError(
                "opponent_checkpoint_adapter must be one of "
                "'policy_head_greedy' or 'mcts_eval_mode'"
            )

        self._opponent_checkpoint_metadata = {
            "policy_id": spec.policy_id,
            "checkpoint_path": str(spec.checkpoint_path),
            "checkpoint_schema_id": spec.checkpoint_schema_id,
            "feature_schema_id": spec.feature_schema_id,
            "feature_mode": spec.feature_mode,
            "adapter": adapter,
            "adapter_schema_id": spec.adapter_schema_id,
            "adapter_label": spec.adapter_label,
            "checkpoint_summary": spec.checkpoint_metadata.get("checkpoint_summary", {}),
        }
        if hasattr(spec, "num_simulations"):
            self._opponent_checkpoint_metadata["num_simulations"] = int(spec.num_simulations)
        return spec.policy

    def _lightzero_observation(self) -> dict[str, Any]:
        if self._observations is None:
            raise RuntimeError("reset must be called before observing")
        observation = self._observations[self.ego_agent]
        return {
            "observation": self._encode_observation(observation),
            "action_mask": np.ones(len(ACTION_LABELS), dtype=np.int8),
            "to_play": -1,
            "timestep": int(observation.step),
        }

    def _encode_observation(self, observation: PongObservation) -> np.ndarray:
        if self.feature_mode == "raster_flat":
            return self._env.raster_observation().astype(np.float32).reshape(-1)

        return encode_tabular_ego_observation(observation, self._pong_config)

    def _info(
        self,
        *,
        done: bool,
        terminated: bool,
        truncated: bool,
        reward: float,
        winner: str | None,
        last_hit: str | None,
        final_rewards: dict[str, float],
    ) -> dict[str, Any]:
        steps = int(self._observations[self.ego_agent].step) if self._observations else 0
        score_return = float(self._episode_return)
        shaped_return = (
            _shaped_loss_delay_return(
                score_return=score_return,
                episode_steps=steps,
                max_steps=self._pong_config.max_steps,
            )
            if done
            else None
        )
        trace_hash = hashlib.sha256(
            json.dumps(self._action_trace, sort_keys=True).encode("utf-8")
        ).hexdigest()
        action_counts = {
            agent: {label: int(self._action_counts_by_agent[agent][label]) for label in ACTION_LABELS}
            for agent in AGENTS
        }
        return {
            "curvyzero_pong": {
                "schema": EPISODE_INFO_SCHEMA_ID,
                "env_id": self.env_id,
                "curvyzero_env": self.curvyzero_env,
                "feature_mode": self.feature_mode,
                "feature_schema_id": _feature_schema_id(self.feature_mode),
                "observation_schema_id": OBSERVATION_SCHEMA_ID,
                "base_seed": self._base_seed,
                "episode_seed": self._episode_seed,
                "episode_index": self._episode_index - 1,
                "configured_dynamic_seed": self._configured_dynamic_seed,
                "effective_dynamic_seed": self._dynamic_seed,
                "seed_call_dynamic_seed_arg": self._last_seed_call_dynamic_seed_arg,
                "seed_source": self._last_seed_source,
                "ego_agent": self.ego_agent,
                "opponent_agent": self.opponent_agent,
                "opponent_policy_id": self.opponent_policy_id,
                "pong_reset_profile": self.pong_reset_profile,
                "pong_reset_pressure_agent": self.pong_reset_pressure_agent,
                "pong_reset": dict(self._env._reset_info),
                "learner_control_kind": "live",
                "learner_policy_kind": "lightzero_train_muzero",
                "opponent_control_kind": _opponent_control_kind(self.opponent_policy_id),
                **self._opponent_checkpoint_info(),
                "steps": steps,
                "max_steps": self._pong_config.max_steps,
                "pong_episode_max_steps": self._pong_config.max_steps,
                "winner": winner,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
                "reward_after_step": reward,
                "score_return": score_return,
                "shaped_loss_delay_return": shaped_return,
                "survival_fraction": steps / self._pong_config.max_steps,
                "action_counts_by_agent": action_counts,
                "action_trace": list(self._action_trace),
                "last_hit": last_hit,
                "final_rewards": final_rewards,
                "trace_hash": trace_hash,
            }
        }

    def _resolve_reset_pressure_agent(self, configured_agent: Any) -> str:
        value = str(configured_agent)
        if value == "ego":
            return self.ego_agent
        if value == "opponent":
            return self.opponent_agent
        if value in {*AGENTS, "random"}:
            return value
        raise ValueError(
            "pong_reset_pressure_agent must be one of "
            f"{(*AGENTS, 'random', 'ego', 'opponent')!r}, got {value!r}"
        )

    def _opponent_checkpoint_info(self) -> dict[str, Any]:
        if self.opponent_policy_id not in {
            "lightzero_policy_head_checkpoint",
            "lightzero_mcts_checkpoint",
        }:
            return {}
        metadata = self._opponent_checkpoint_metadata or {}
        checkpoint_summary = dict(metadata.get("checkpoint_summary") or {})
        return {
            "opponent_checkpoint_label": self.opponent_checkpoint_label,
            "opponent_checkpoint_path": str(self.opponent_checkpoint_path),
            "opponent_checkpoint_source_ref": self.opponent_checkpoint_source_ref,
            "opponent_checkpoint_sha256": (
                self.opponent_checkpoint_sha256 or checkpoint_summary.get("sha256")
            ),
            "opponent_checkpoint_bytes": checkpoint_summary.get("bytes"),
            "opponent_checkpoint_adapter": metadata.get(
                "adapter", self.opponent_checkpoint_adapter
            ),
            "opponent_checkpoint_adapter_schema_id": metadata.get("adapter_schema_id"),
            "opponent_checkpoint_adapter_label": metadata.get("adapter_label"),
            "opponent_checkpoint_num_simulations": metadata.get(
                "num_simulations",
                self.opponent_checkpoint_num_simulations
                if self.opponent_checkpoint_adapter == "mcts_eval_mode"
                else None,
            ),
            "opponent_checkpoint_state_key": self.opponent_checkpoint_state_key,
            "opponent_checkpoint_policy_id": metadata.get("policy_id"),
            "opponent_checkpoint_config_opponent_policy": (
                self.opponent_checkpoint_config_opponent_policy
            ),
        }

    def _write_episode_row(self, row: dict[str, Any]) -> None:
        if self._telemetry_path is None:
            return
        self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        with self._telemetry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(row), sort_keys=True) + "\n")


def load_episode_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load terminal dummy Pong telemetry rows from a JSONL sidecar."""

    row_path = Path(path)
    if not row_path.exists():
        return []
    rows = []
    for line in row_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize env-side Pong telemetry without depending on LightZero logs."""

    if not rows:
        return {
            "schema": "curvyzero_lightzero_dummy_pong_scorecard/v1",
            "episodes": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "truncation_rate": 0.0,
            "survival_steps": _numeric_stats([]),
            "p90_survival_steps": 0.0,
            "score_return": _numeric_stats([]),
            "shaped_loss_delay_return": _numeric_stats([]),
            "seeds": [],
            "seed_unique_count": 0,
            "seed_most_common": None,
            "seed_most_common_count": 0,
            "seed_most_common_fraction": 0.0,
            "seed_top5": [],
            "seed_dominance_warning": False,
            "seed_dominance_warning_reason": None,
            "action_counts_by_agent": {},
            "learner_control_kinds": [],
            "learner_policy_kinds": [],
            "opponent_control_kinds": [],
            "opponent_policy_ids": [],
            "opponent_is_frozen_checkpoint": False,
            "opponent_checkpoint_refs": [],
            "trace_hashes": [],
        }

    steps = [float(row.get("steps", 0)) for row in rows]
    score_returns = [float(row.get("score_return", 0.0)) for row in rows]
    shaped_returns = [
        float(row.get("shaped_loss_delay_return", 0.0) or 0.0)
        for row in rows
    ]
    ego_wins = sum(1 for row in rows if row.get("winner") == row.get("ego_agent"))
    terminal_losses = sum(
        1
        for row in rows
        if row.get("winner") is not None and row.get("winner") != row.get("ego_agent")
    )
    timeouts = sum(1 for row in rows if bool(row.get("truncated")))
    action_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for agent, counts in dict(row.get("action_counts_by_agent", {})).items():
            dest = action_counts.setdefault(
                str(agent),
                {label: 0 for label in ACTION_LABELS},
            )
            for label in ACTION_LABELS:
                dest[label] += int(dict(counts).get(label, 0))
    seeds = [int(row.get("episode_seed", 0)) for row in rows]
    effective_dynamic_seed_values = [
        bool(row["effective_dynamic_seed"])
        for row in rows
        if isinstance(row.get("effective_dynamic_seed"), bool)
    ]
    seed_distribution = _seed_distribution(
        seeds,
        effective_dynamic_seed_values=effective_dynamic_seed_values,
    )
    opponent_checkpoint_refs = _opponent_checkpoint_refs(rows)

    return {
        "schema": "curvyzero_lightzero_dummy_pong_scorecard/v1",
        "episodes": len(rows),
        "wins": int(ego_wins),
        "losses": int(terminal_losses),
        "timeouts": int(timeouts),
        "truncation_rate": float(timeouts / len(rows)),
        "survival_steps": _numeric_stats(steps),
        "p90_survival_steps": _numeric_stats(steps)["p90"],
        "score_return": _numeric_stats(score_returns),
        "shaped_loss_delay_return": _numeric_stats(shaped_returns),
        "seeds": seeds,
        **seed_distribution,
        "action_counts_by_agent": action_counts,
        "learner_control_kinds": sorted(
            {str(row.get("learner_control_kind", "live")) for row in rows}
        ),
        "learner_policy_kinds": sorted(
            {str(row.get("learner_policy_kind", "")) for row in rows if row.get("learner_policy_kind")}
        ),
        "opponent_control_kinds": sorted(
            {
                str(row.get("opponent_control_kind") or _opponent_control_kind(row.get("opponent_policy_id")))
                for row in rows
            }
        ),
        "opponent_policy_ids": sorted(
            {str(row.get("opponent_policy_id", "")) for row in rows if row.get("opponent_policy_id")}
        ),
        "opponent_is_frozen_checkpoint": bool(opponent_checkpoint_refs),
        "opponent_checkpoint_refs": opponent_checkpoint_refs,
        "trace_hashes": [str(row.get("trace_hash", "")) for row in rows],
    }


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _opponent(agent: str) -> str:
    return "player_1" if agent == "player_0" else "player_0"


def _opponent_control_kind(opponent_policy_id: Any) -> str:
    if opponent_policy_id in {
        "lightzero_policy_head_checkpoint",
        "lightzero_mcts_checkpoint",
    }:
        return "frozen_checkpoint"
    if opponent_policy_id in {
        "random_uniform",
        "track_ball",
        "lagged_track_ball_1",
    }:
        return "scripted"
    return "unknown"


def _opponent_checkpoint_refs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if _opponent_control_kind(row.get("opponent_policy_id")) != "frozen_checkpoint":
            continue
        ref = {
            "opponent_policy_id": row.get("opponent_policy_id"),
            "opponent_checkpoint_label": row.get("opponent_checkpoint_label"),
            "opponent_checkpoint_path": row.get("opponent_checkpoint_path"),
            "opponent_checkpoint_source_ref": row.get("opponent_checkpoint_source_ref"),
            "opponent_checkpoint_sha256": row.get("opponent_checkpoint_sha256"),
            "opponent_checkpoint_adapter": row.get("opponent_checkpoint_adapter"),
            "opponent_checkpoint_adapter_schema_id": row.get(
                "opponent_checkpoint_adapter_schema_id"
            ),
            "opponent_checkpoint_num_simulations": row.get(
                "opponent_checkpoint_num_simulations"
            ),
            "opponent_checkpoint_state_key": row.get("opponent_checkpoint_state_key"),
        }
        key = tuple(ref.items())
        refs_by_key.setdefault(key, ref)
    return list(refs_by_key.values())


def _seed_source(
    *,
    configured_dynamic_seed: bool,
    seed_call_dynamic_seed_arg: bool | None,
) -> str:
    if seed_call_dynamic_seed_arg is None:
        return "config_dynamic" if configured_dynamic_seed else "config_fixed"
    if seed_call_dynamic_seed_arg == configured_dynamic_seed:
        return "env_seed_arg_matches_config"
    if configured_dynamic_seed:
        return "config_dynamic_overrode_env_seed_arg_false"
    return "config_fixed_overrode_env_seed_arg_true"


def _scale_unit(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return float(value) / float(maximum)


def _scale_signed(value: float, maximum_abs: float) -> float:
    if maximum_abs <= 0:
        return 0.0
    return float(np.clip(float(value) / float(maximum_abs), -1.0, 1.0))


def _feature_schema_id(feature_mode: str) -> str:
    if feature_mode == "tabular_ego":
        return TABULAR_FEATURE_SCHEMA_ID
    if feature_mode == "raster_flat":
        return RASTER_FLAT_FEATURE_SCHEMA_ID
    return f"unknown:{feature_mode}"


def _shaped_loss_delay_return(
    *,
    score_return: float,
    episode_steps: int,
    max_steps: int,
) -> float:
    if score_return > 0.0:
        return 1.0
    if score_return < 0.0:
        return -1.0 + 0.5 * (episode_steps / max_steps)
    return 0.0


def _numeric_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _seed_distribution(
    seeds: list[int],
    *,
    effective_dynamic_seed_values: list[bool] | None = None,
) -> dict[str, Any]:
    if not seeds:
        return {
            "seed_unique_count": 0,
            "seed_most_common": None,
            "seed_most_common_count": 0,
            "seed_most_common_fraction": 0.0,
            "seed_top5": [],
            "seed_dominance_warning": False,
            "seed_dominance_warning_reason": None,
        }

    counts = Counter(seeds)
    top5 = [
        {
            "seed": int(seed),
            "count": int(count),
            "fraction": float(count / len(seeds)),
        }
        for seed, count in counts.most_common(5)
    ]
    most_common = top5[0]
    dynamic_mode_known = bool(effective_dynamic_seed_values)
    dynamic_mode_active = any(effective_dynamic_seed_values or [])
    dominance_warning = bool(
        most_common["fraction"] >= 0.5 and (not dynamic_mode_known or dynamic_mode_active)
    )
    if dominance_warning and len(counts) == 1:
        warning_reason = (
            "one episode_seed accounts for all rows while dynamic seeding was "
            "effective or unspecified"
        )
    elif dominance_warning:
        warning_reason = "one episode_seed accounts for at least half of rows"
    else:
        warning_reason = None
    return {
        "seed_unique_count": len(counts),
        "seed_most_common": most_common["seed"],
        "seed_most_common_count": most_common["count"],
        "seed_most_common_fraction": most_common["fraction"],
        "seed_top5": top5,
        "seed_dominance_warning": dominance_warning,
        "seed_dominance_warning_reason": warning_reason,
    }
