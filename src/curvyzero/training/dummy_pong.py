"""Tiny deterministic two-player Pong-like environment.

This is project-owned infrastructure scaffolding: simple enough to inspect,
but shaped to exercise simultaneous two-player controls, ego observations,
ball dynamics, terminal scoring rewards, and time-limit truncation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AGENTS = ("player_0", "player_1")
RULESET_ID = "dummy_pong_v0"
OBSERVATION_SCHEMA_ID = "pong_ego_tabular_v0"
REWARD_SCHEMA_ID = "win_loss_score_v0"
ACTION_SCHEMA_ID = "pong_vertical_actions_v0"
ACTION_LABELS = ("up", "stay", "down")
PONG_RESET_PROFILES = ("default", "contact_pressure")
PONG_RESET_PRESSURE_AGENTS = (*AGENTS, "random")
RASTER_OBSERVATION_SCHEMA_ID = "dummy_pong_raster_grid_v0"
RASTER_LEGEND = {
    "0": "empty",
    "1": "player_0_paddle",
    "2": "player_1_paddle",
    "3": "ball",
    "4": "ball_on_paddle",
}
PADDLE_BOUNCE_SCHEMA_ID = "dummy_pong_paddle_offset_bounce_v0"
PADDLE_BOUNCE_RULE = {
    "description": (
        "On paddle hits, outgoing ball_vy is the sign of the ball's impact "
        "offset from the paddle center."
    ),
    "default_paddle_height_examples": [
        {"impact": "top", "impact_offset": -1, "outgoing_ball_vy": -1},
        {"impact": "center", "impact_offset": 0, "outgoing_ball_vy": 0},
        {"impact": "bottom", "impact_offset": 1, "outgoing_ball_vy": 1},
    ],
}


@dataclass(frozen=True, slots=True)
class PongConfig:
    width: int = 15
    height: int = 9
    paddle_height: int = 3
    max_steps: int = 120
    action_count: int = 3
    players: int = 2
    reset_profile: str = "default"
    reset_pressure_agent: str = "random"
    reset_contact_distance_min: int = 2
    reset_contact_distance_max: int = 3


@dataclass(frozen=True, slots=True)
class PongObservation:
    ego_paddle_y: int
    opponent_paddle_y: int
    ego_paddle_x: int
    opponent_paddle_x: int
    ball_dx_forward: int
    ball_dy_from_ego_center: int
    ball_vx_forward: int
    ball_vy: int
    ball_y: int
    step: int
    ego_agent: str


@dataclass(frozen=True, slots=True)
class PongStep:
    observations: dict[str, PongObservation]
    rewards: dict[str, float]
    terminated: bool
    truncated: bool
    infos: dict[str, object]


class PongEnv:
    """Small Pong-like simultaneous-move game with sparse terminal rewards."""

    def __init__(self, config: PongConfig | None = None):
        self.config = config or PongConfig()
        if self.config.width < 7:
            raise ValueError("width must be at least 7")
        if self.config.height < self.config.paddle_height + 2:
            raise ValueError("height must leave room around the paddle")
        if self.config.paddle_height < 1:
            raise ValueError("paddle_height must be positive")
        if self.config.reset_profile not in PONG_RESET_PROFILES:
            raise ValueError(
                f"reset_profile must be one of {PONG_RESET_PROFILES!r}, "
                f"got {self.config.reset_profile!r}"
            )
        if self.config.reset_pressure_agent not in PONG_RESET_PRESSURE_AGENTS:
            raise ValueError(
                "reset_pressure_agent must be one of "
                f"{PONG_RESET_PRESSURE_AGENTS!r}, got {self.config.reset_pressure_agent!r}"
            )
        if self.config.reset_contact_distance_min < 1:
            raise ValueError("reset_contact_distance_min must be at least 1")
        if self.config.reset_contact_distance_max < self.config.reset_contact_distance_min:
            raise ValueError(
                "reset_contact_distance_max must be >= reset_contact_distance_min"
            )
        self._paddle_x = {"player_0": 1, "player_1": self.config.width - 2}
        self._paddle_y = {agent: 0 for agent in AGENTS}
        self._ball_x = self.config.width // 2
        self._ball_y = self.config.height // 2
        self._ball_vx = 1
        self._ball_vy = 1
        self._step = 0
        self._done = False
        self._last_hit: str | None = None
        self._last_hit_impact: dict[str, int | str] | None = None
        self._score_agent: str | None = None
        self._reset_info: dict[str, object] = {
            "profile": self.config.reset_profile,
            "curriculum_enabled": self.config.reset_profile != "default",
        }

    def reset(self, seed: int | None = None) -> dict[str, PongObservation]:
        rng = np.random.default_rng(seed)
        center_y = (self.config.height - self.config.paddle_height) // 2
        self._paddle_y = {agent: center_y for agent in AGENTS}
        self._step = 0
        self._done = False
        self._last_hit = None
        self._last_hit_impact = None
        self._score_agent = None
        if self.config.reset_profile == "default":
            self._reset_info = self._reset_default(rng)
        elif self.config.reset_profile == "contact_pressure":
            self._reset_info = self._reset_contact_pressure(rng)
        else:  # Defensive guard; __init__ validates this.
            raise ValueError(f"unknown reset_profile: {self.config.reset_profile!r}")
        return self.observations()

    def step(self, actions_by_agent: dict[str, int]) -> PongStep:
        if self._done:
            raise RuntimeError("reset must be called before stepping a finished episode")
        if set(actions_by_agent) != set(AGENTS):
            raise ValueError(f"actions must be provided for both agents: {AGENTS!r}")
        for agent, action in actions_by_agent.items():
            if action < 0 or action >= self.config.action_count:
                raise ValueError(f"invalid action for {agent}: {action!r}")

        for agent in AGENTS:
            self._move_paddle(agent, actions_by_agent[agent])

        next_x = self._ball_x + self._ball_vx
        next_y = self._ball_y + self._ball_vy
        if next_y < 0:
            next_y = 1
            self._ball_vy = 1
        elif next_y >= self.config.height:
            next_y = self.config.height - 2
            self._ball_vy = -1

        hit_agent = self._paddle_hit_agent(next_x, next_y)
        if hit_agent is not None:
            self._ball_vx *= -1
            next_x = self._paddle_x[hit_agent] + self._ball_vx
            self._last_hit = hit_agent
            impact_offset = self._paddle_hit_offset(hit_agent, next_y)
            self._ball_vy = self._vy_after_hit(hit_agent, next_y)
            self._last_hit_impact = {
                "agent": hit_agent,
                "hit_y": int(next_y),
                "paddle_center_y": int(
                    self._paddle_y[hit_agent] + self.config.paddle_height // 2
                ),
                "impact_offset": int(impact_offset),
                "outgoing_ball_vy": int(self._ball_vy),
            }

        self._ball_x = next_x
        self._ball_y = next_y
        self._step += 1

        rewards = {agent: 0.0 for agent in AGENTS}
        terminated = False
        if self._ball_x < 0:
            terminated = True
            self._score_agent = "player_1"
            rewards = {"player_0": -1.0, "player_1": 1.0}
        elif self._ball_x >= self.config.width:
            terminated = True
            self._score_agent = "player_0"
            rewards = {"player_0": 1.0, "player_1": -1.0}
        truncated = self._step >= self.config.max_steps and not terminated
        self._done = terminated or truncated

        return PongStep(
            observations=self.observations(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos={
                "winner": self._score_agent,
                "score_agent": self._score_agent,
                "last_hit": self._last_hit,
                "last_hit_impact": (
                    None if self._last_hit_impact is None else dict(self._last_hit_impact)
                ),
                "reset": dict(self._reset_info),
                "ball": {
                    "x": self._ball_x,
                    "y": self._ball_y,
                    "vx": self._ball_vx,
                    "vy": self._ball_vy,
                },
                "paddles": dict(self._paddle_y),
            },
        )

    def observations(self) -> dict[str, PongObservation]:
        return {agent: self.observation(agent) for agent in AGENTS}

    def observation(self, ego_agent: str) -> PongObservation:
        if ego_agent not in AGENTS:
            raise ValueError(f"unknown agent {ego_agent!r}")
        opponent_agent = _opponent(ego_agent)
        forward = 1 if ego_agent == "player_0" else -1
        ego_center = self._paddle_y[ego_agent] + self.config.paddle_height // 2
        return PongObservation(
            ego_paddle_y=self._paddle_y[ego_agent],
            opponent_paddle_y=self._paddle_y[opponent_agent],
            ego_paddle_x=self._paddle_x[ego_agent],
            opponent_paddle_x=self._paddle_x[opponent_agent],
            ball_dx_forward=(self._ball_x - self._paddle_x[ego_agent]) * forward,
            ball_dy_from_ego_center=self._ball_y - ego_center,
            ball_vx_forward=self._ball_vx * forward,
            ball_vy=self._ball_vy,
            ball_y=self._ball_y,
            step=self._step,
            ego_agent=ego_agent,
        )

    def raster_observation(self) -> np.ndarray:
        """Return a tiny visual observation grid for Pong experiments."""

        grid = np.zeros((self.config.height, self.config.width), dtype=np.uint8)
        for agent, value in (("player_0", 1), ("player_1", 2)):
            x = self._paddle_x[agent]
            y = self._paddle_y[agent]
            for row in range(y, min(y + self.config.paddle_height, self.config.height)):
                grid[row, x] = value
        if 0 <= self._ball_x < self.config.width and 0 <= self._ball_y < self.config.height:
            grid[self._ball_y, self._ball_x] = 4 if grid[self._ball_y, self._ball_x] else 3
        return grid

    @property
    def winner(self) -> str | None:
        return self._score_agent

    def _move_paddle(self, agent: str, action: int) -> None:
        delta = action - 1
        max_y = self.config.height - self.config.paddle_height
        self._paddle_y[agent] = min(max(self._paddle_y[agent] + delta, 0), max_y)

    def _paddle_hit_agent(self, next_x: int, next_y: int) -> str | None:
        candidate = "player_0" if self._ball_vx < 0 else "player_1"
        if next_x != self._paddle_x[candidate]:
            return None
        top = self._paddle_y[candidate]
        if top <= next_y < top + self.config.paddle_height:
            return candidate
        return None

    def _vy_after_hit(self, agent: str, hit_y: int) -> int:
        impact_offset = self._paddle_hit_offset(agent, hit_y)
        if impact_offset < 0:
            return -1
        if impact_offset > 0:
            return 1
        return 0

    def _paddle_hit_offset(self, agent: str, hit_y: int) -> int:
        center = self._paddle_y[agent] + self.config.paddle_height // 2
        return hit_y - center

    def _reset_default(self, rng: np.random.Generator) -> dict[str, object]:
        self._ball_x = self.config.width // 2
        self._ball_y = int(rng.integers(2, self.config.height - 2))
        self._ball_vx = -1 if int(rng.integers(2)) == 0 else 1
        self._ball_vy = -1 if int(rng.integers(2)) == 0 else 1
        return {
            "profile": "default",
            "curriculum_enabled": False,
            "pressure_agent": None,
            "contact_distance": None,
        }

    def _reset_contact_pressure(self, rng: np.random.Generator) -> dict[str, object]:
        pressure_agent = self._resolve_reset_pressure_agent(rng)
        pressure_x = self._paddle_x[pressure_agent]
        available_distance = (
            self.config.width - 2 - pressure_x
            if pressure_agent == "player_0"
            else pressure_x - 1
        )
        max_distance = min(self.config.reset_contact_distance_max, available_distance)
        min_distance = min(self.config.reset_contact_distance_min, max_distance)
        contact_distance = int(rng.integers(min_distance, max_distance + 1))
        self._ball_vx = -1 if pressure_agent == "player_0" else 1
        self._ball_x = pressure_x - self._ball_vx * contact_distance

        target_hit_y = int(rng.integers(1, self.config.height - 1))
        desired_impact_offset = int(rng.choice(np.asarray([-1, 0, 1], dtype=np.int64)))
        max_paddle_y = self.config.height - self.config.paddle_height
        target_center_y = target_hit_y - desired_impact_offset
        self._paddle_y[pressure_agent] = int(
            np.clip(target_center_y - self.config.paddle_height // 2, 0, max_paddle_y)
        )

        opponent_agent = _opponent(pressure_agent)
        center_y = (self.config.height - self.config.paddle_height) // 2
        self._paddle_y[opponent_agent] = int(
            np.clip(center_y + int(rng.integers(-1, 2)), 0, max_paddle_y)
        )

        vy_candidates = list(rng.permutation(np.asarray([-1, 0, 1], dtype=np.int64)))
        for candidate_vy in vy_candidates:
            y_candidates = [
                y
                for y in range(self.config.height)
                if _advance_y(y, int(candidate_vy), contact_distance, self.config.height)
                == target_hit_y
            ]
            if y_candidates:
                self._ball_vy = int(candidate_vy)
                self._ball_y = int(rng.choice(np.asarray(y_candidates, dtype=np.int64)))
                break
        else:
            self._ball_vy = 0
            self._ball_y = target_hit_y

        return {
            "profile": "contact_pressure",
            "curriculum_enabled": True,
            "pressure_agent": pressure_agent,
            "contact_distance": contact_distance,
            "target_hit_y": target_hit_y,
            "desired_impact_offset": desired_impact_offset,
            "incoming_ball_vy": int(self._ball_vy),
        }

    def _resolve_reset_pressure_agent(self, rng: np.random.Generator) -> str:
        if self.config.reset_pressure_agent == "random":
            return str(AGENTS[int(rng.integers(len(AGENTS)))])
        return self.config.reset_pressure_agent


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent {agent!r}")


def _advance_y(ball_y: int, ball_vy: int, steps: int, height: int) -> int:
    next_y = int(ball_y)
    vy = int(ball_vy)
    for _ in range(steps):
        next_y += vy
        if next_y < 0:
            next_y = 1
            vy = 1
        elif next_y >= height:
            next_y = height - 2
            vy = -1
    return next_y
