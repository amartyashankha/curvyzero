"""Reference CurvyTron-like environment.

This is intentionally small and conservative. It establishes deterministic
reset/step semantics before source-derived CurvyTron details are fully mined.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np

from curvyzero.env.config import CurvyTronConfig
from curvyzero.env.state import EnvState

ActionMap = Mapping[str, int]

OBSERVATION_SCHEMA_ID = "curvyzero_debug_global_player_obs/v0"
REWARD_SCHEMA_ID = "curvyzero_sparse_round_outcome/v0"
ENV_IMPL_ID = "curvyzero_python_toy_v0_env/v0"


def _schema_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


OBSERVATION_SCHEMA_HASH = _schema_hash(
    "curvyzero_debug_global_player_obs/v0|float32[9]|global-privileged-debug"
)
REWARD_SCHEMA_HASH = _schema_hash("curvyzero_sparse_round_outcome/v0|winner-1-loser-minus1")


@dataclass(frozen=True, slots=True)
class StepResult:
    observations: dict[str, np.ndarray]
    rewards: dict[str, float]
    terminated: dict[str, bool]
    truncated: dict[str, bool]
    infos: dict[str, dict]


class CurvyTronEnv:
    """Minimal deterministic 1v1 no-bonus environment."""

    def __init__(self, config: CurvyTronConfig | None = None):
        self.config = config or CurvyTronConfig()
        if self.config.players != 2:
            raise ValueError("curvyzero-v0 currently supports exactly 2 players")
        self.state: EnvState | None = None
        self._episode_counter = 0
        self._episode_id: str | None = None
        self._seed: int | None = None
        self._step_index = 0
        self._needs_reset = False
        self.last_reset_info: dict[str, object] | None = None

    @property
    def agents(self) -> list[str]:
        return [f"player_{idx}" for idx in range(self.config.players)]

    @property
    def action_space_id(self) -> str:
        if self.config.allow_straight_action:
            return "curvyzero_turn3/v0"
        return "curvyzero_turn2_toy/v0"

    @property
    def action_space_hash(self) -> str:
        return _schema_hash(f"{self.action_space_id}|action_count={self.config.action_count}")

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        cfg = self.config
        y0 = cfg.height / 2.0
        positions = np.array(
            [[cfg.spawn_margin, y0], [cfg.width - cfg.spawn_margin, y0]], dtype=np.float32
        )
        headings = np.array([0.0, np.pi], dtype=np.float32)
        alive = np.ones(cfg.players, dtype=np.bool_)
        death_tick = np.full(cfg.players, -1, dtype=np.int32)
        occupancy = np.zeros((cfg.height, cfg.width), dtype=np.int16)
        self.state = EnvState(
            tick=0,
            positions=positions,
            headings=headings,
            alive=alive,
            death_tick=death_tick,
            occupancy=occupancy,
            rng=rng,
        )
        self._episode_counter += 1
        self._episode_id = f"curvyzero-toy-v0-episode-{self._episode_counter}"
        self._seed = seed
        self._step_index = 0
        self._needs_reset = False
        self._mark_players()
        self.last_reset_info = self._base_info()
        self.last_reset_info["needs_reset"] = False
        return self._observations()

    def observe(self, ego_player: str) -> np.ndarray:
        """Return the current toy-v0 debug observation for one player."""

        if self.state is None:
            raise RuntimeError("reset must be called before observe")
        self._player_index(ego_player)
        return self._observation_vector().copy()

    def legal_action_mask(self, ego_player: str) -> np.ndarray:
        """Return legal toy action ids for one player in the current state."""

        if self.state is None:
            raise RuntimeError("reset must be called before legal_action_mask")
        idx = self._player_index(ego_player)
        mask = np.zeros(self.config.action_count, dtype=np.bool_)
        if self.state.alive[idx] and not self._is_terminal() and not self._needs_reset:
            mask[:] = True
        return mask

    def step(self, actions: ActionMap) -> StepResult:
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        if self._needs_reset:
            raise RuntimeError(
                "reset must be called before stepping after a terminal or truncated episode"
            )
        self._validate_joint_action(actions)

        cfg = self.config
        pre_alive = self.state.alive.copy()
        rewards = {agent: 0.0 for agent in self.agents}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}

        for _ in range(cfg.action_repeat):
            if not self.state.alive.any():
                break
            self._physics_tick(actions)
            if self.state.tick >= cfg.max_ticks:
                for agent in self.agents:
                    truncated[agent] = True
                break

        if self._is_terminal():
            rewards = self._terminal_rewards()
            terminated = {agent: True for agent in self.agents}

        self._step_index += 1
        observations = self._observations()
        self._needs_reset = any(terminated.values()) or any(truncated.values())
        step_info = self._step_info(pre_alive, terminated=terminated, truncated=truncated)
        infos = {}
        for agent in self.agents:
            done = terminated[agent] or truncated[agent]
            info = dict(step_info)
            info.update(
                {
                    "done": done,
                    "terminated": terminated[agent],
                    "truncated": truncated[agent],
                    "needs_reset": self._needs_reset,
                    "final_observation": observations[agent].copy() if done else None,
                }
            )
            infos[agent] = info

        return StepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
        )

    def _physics_tick(self, actions: ActionMap) -> None:
        assert self.state is not None
        cfg = self.config
        state = self.state
        state.tick += 1

        old_positions = state.positions.copy()
        for idx, agent in enumerate(self.agents):
            if not state.alive[idx]:
                continue
            action = actions[agent]
            turn = self._action_to_turn(action)
            state.headings[idx] += turn * cfg.turn_rate_radians
            delta = np.array(
                [np.cos(state.headings[idx]), np.sin(state.headings[idx])], dtype=np.float32
            )
            state.positions[idx] += delta * cfg.speed

        deaths = np.zeros(cfg.players, dtype=np.bool_)
        for idx in range(cfg.players):
            if not state.alive[idx]:
                continue
            x, y = state.positions[idx]
            if x < 0 or x >= cfg.width or y < 0 or y >= cfg.height:
                deaths[idx] = True
                continue
            cell_x = int(round(float(x)))
            cell_y = int(round(float(y)))
            if cell_x < 0 or cell_x >= cfg.width or cell_y < 0 or cell_y >= cfg.height:
                deaths[idx] = True
            elif state.occupancy[cell_y, cell_x] != 0:
                deaths[idx] = True

        for idx, died in enumerate(deaths):
            if died:
                state.alive[idx] = False
                state.death_tick[idx] = state.tick

        self._draw_segments(old_positions, state.positions, deaths)

    def _action_to_turn(self, action: int) -> int:
        if self.config.allow_straight_action:
            if action == 0:
                return -1
            if action == 1:
                return 0
            if action == 2:
                return 1
        else:
            if action == 0:
                return -1
            if action == 1:
                return 1
        raise ValueError(f"invalid action {action!r} for action_count={self.config.action_count}")

    def _mark_players(self) -> None:
        assert self.state is not None
        for idx, position in enumerate(self.state.positions):
            self._mark_cell(position, idx + 1)

    def _draw_segments(self, starts: np.ndarray, ends: np.ndarray, deaths: np.ndarray) -> None:
        assert self.state is not None
        for idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if deaths[idx]:
                continue
            self._mark_segment(start, end, idx + 1)

    def _mark_segment(self, start: np.ndarray, end: np.ndarray, value: int) -> None:
        distance = float(np.linalg.norm(end - start))
        steps = max(1, int(np.ceil(distance * 2)))
        for t in np.linspace(0.0, 1.0, steps + 1):
            self._mark_cell(start + (end - start) * t, value)

    def _mark_cell(self, position: np.ndarray, value: int) -> None:
        assert self.state is not None
        x = int(round(float(position[0])))
        y = int(round(float(position[1])))
        if 0 <= x < self.config.width and 0 <= y < self.config.height:
            self.state.occupancy[y, x] = value

    def _player_index(self, player_id: str) -> int:
        try:
            return self.agents.index(player_id)
        except ValueError as exc:
            raise ValueError(f"unknown player {player_id!r}") from exc

    def _validate_joint_action(self, actions: ActionMap) -> None:
        assert self.state is not None
        known_agents = set(self.agents)
        provided_agents = set(actions)
        unknown = sorted(provided_agents - known_agents)
        if unknown:
            raise ValueError(f"unknown player actions: {', '.join(unknown)}")

        missing_live = [
            agent
            for idx, agent in enumerate(self.agents)
            if bool(self.state.alive[idx]) and agent not in actions
        ]
        if missing_live:
            raise ValueError(f"missing actions for live players: {', '.join(missing_live)}")

        for agent, action in actions.items():
            if agent in known_agents:
                self._action_to_turn(action)

    def _base_info(self) -> dict[str, object]:
        return {
            "episode_id": self._episode_id,
            "seed": self._seed,
            "ruleset_id": self.config.ruleset,
            "rules_hash": self.config.rules_hash,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
            "action_space_id": self.action_space_id,
            "action_space_hash": self.action_space_hash,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "reward_schema_hash": REWARD_SCHEMA_HASH,
            "player_ids": self.agents,
            "max_players": self.config.players,
            "env_impl_id": ENV_IMPL_ID,
        }

    def _step_info(
        self,
        pre_alive: np.ndarray,
        *,
        terminated: Mapping[str, bool],
        truncated: Mapping[str, bool],
    ) -> dict[str, object]:
        assert self.state is not None
        info = self._base_info()
        death_player_ids = [
            agent
            for idx, agent in enumerate(self.agents)
            if bool(pre_alive[idx]) and not bool(self.state.alive[idx])
        ]
        winner_ids: list[str] = []
        loser_ids: list[str] = []
        terminal_reason = "none"
        draw = False

        if any(terminated.values()):
            alive_ids = [
                agent for idx, agent in enumerate(self.agents) if bool(self.state.alive[idx])
            ]
            if len(alive_ids) == 1:
                terminal_reason = "survivor_win"
                winner_ids = alive_ids
                loser_ids = [agent for agent in self.agents if agent not in winner_ids]
            else:
                terminal_reason = "all_dead_draw"
                draw = True
        elif any(truncated.values()):
            terminal_reason = "timeout"

        timeout = any(truncated.values())
        info.update(
            {
                "step_index": self._step_index,
                "tick_index": int(self.state.tick),
                "terminal_reason": terminal_reason,
                "winner_ids": winner_ids,
                "loser_ids": loser_ids,
                "death_player_ids": death_player_ids,
                "draw": draw,
                "timeout": timeout,
                "truncation_reason": "max_ticks" if timeout else None,
                "event_ref": None,
                "event_range": None,
                "state_ref": None,
                "trace_ref": None,
            }
        )
        return info

    def _observation_vector(self) -> np.ndarray:
        assert self.state is not None
        return np.concatenate(
            [
                self.state.positions.reshape(-1),
                self.state.headings,
                self.state.alive.astype(np.float32),
                np.array([self.state.tick / self.config.max_ticks], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _observations(self) -> dict[str, np.ndarray]:
        assert self.state is not None
        flat = self._observation_vector()
        return {agent: flat.copy() for agent in self.agents}

    def _is_terminal(self) -> bool:
        assert self.state is not None
        return int(self.state.alive.sum()) <= 1

    def _terminal_rewards(self) -> dict[str, float]:
        assert self.state is not None
        if self.state.alive.sum() == 1:
            winner = int(np.flatnonzero(self.state.alive)[0])
            return {agent: (1.0 if idx == winner else -1.0) for idx, agent in enumerate(self.agents)}
        return {agent: 0.0 for agent in self.agents}
