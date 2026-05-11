"""Dummy two-player line-duel training scaffold.

This module is infrastructure scaffolding, not a real MuZero implementation.
It keeps the solo dummy trainer's tabular shape while exercising simultaneous
two-player stepping, ego-perspective replay rows, and shared-policy updates.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

AGENTS = ("player_0", "player_1")
RULESET_ID = "dummy_line_duel_v0"
OBSERVATION_SCHEMA_ID = "line_duel_ego_tabular_v0"
REWARD_SCHEMA_ID = "win_loss_draw_v0"

StateKey = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LineDuelConfig:
    width: int = 11
    height: int = 11
    max_steps: int = 80
    discount: float = 0.997
    replay_capacity: int = 512
    action_count: int = 3
    players: int = 2


@dataclass(frozen=True, slots=True)
class LineDuelObservation:
    ego_x: int
    ego_y: int
    ego_heading: int
    left_clearance: int
    straight_clearance: int
    right_clearance: int
    opponent_dx_forward: int
    opponent_dy_right: int
    opponent_heading_relative: int
    opponent_alive: bool
    step: int


@dataclass(frozen=True, slots=True)
class LineDuelStep:
    observations: dict[str, LineDuelObservation]
    rewards: dict[str, float]
    terminated: bool
    truncated: bool
    infos: dict[str, object]


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    episode_id: int
    seed: int
    step: int
    ego_agent: str
    opponent_agent: str
    observation_key: StateKey
    ego_action: int
    joint_action_by_agent: dict[str, int]
    reward: float
    next_observation_key: StateKey | None
    done: bool
    truncated: bool
    target_return: float
    opponent_policy_id: str
    ruleset_id: str = RULESET_ID
    observation_schema_id: str = OBSERVATION_SCHEMA_ID
    reward_schema_id: str = REWARD_SCHEMA_ID


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    seed: int
    episode_id: int
    steps: int
    winner: str | None
    draw: bool
    truncated: bool
    rewards: dict[str, float]
    action_counts_by_agent: dict[str, tuple[int, int, int]]
    death_causes: dict[str, tuple[str, ...]]


class LineDuelEnv:
    """Tiny two-player task with simultaneous turns and trail collisions."""

    _DIRECTIONS = (
        np.array([0, -1], dtype=np.int16),
        np.array([1, 0], dtype=np.int16),
        np.array([0, 1], dtype=np.int16),
        np.array([-1, 0], dtype=np.int16),
    )

    def __init__(self, config: LineDuelConfig | None = None):
        self.config = config or LineDuelConfig()
        self._grid = np.zeros((self.config.height, self.config.width), dtype=np.bool_)
        self._positions = {agent: np.zeros(2, dtype=np.int16) for agent in AGENTS}
        self._headings = {agent: 1 for agent in AGENTS}
        self._alive = {agent: True for agent in AGENTS}
        self._death_causes: dict[str, list[str]] = {agent: [] for agent in AGENTS}
        self._step = 0
        self._done = False

    def reset(self, seed: int | None = None) -> dict[str, LineDuelObservation]:
        del seed
        self._grid.fill(False)
        self._positions = {
            "player_0": np.array([2, self.config.height // 2], dtype=np.int16),
            "player_1": np.array([self.config.width - 3, self.config.height // 2], dtype=np.int16),
        }
        self._headings = {"player_0": 1, "player_1": 3}
        self._alive = {agent: True for agent in AGENTS}
        self._death_causes = {agent: [] for agent in AGENTS}
        self._step = 0
        self._done = False
        for agent in AGENTS:
            self._mark_position(self._positions[agent])
        return self.observations()

    def step(self, actions_by_agent: dict[str, int]) -> LineDuelStep:
        if self._done:
            raise RuntimeError("reset must be called before stepping a finished episode")

        live_agents = tuple(agent for agent in AGENTS if self._alive[agent])
        if set(actions_by_agent) != set(live_agents):
            raise ValueError(f"actions must be provided for live agents only: {live_agents!r}")
        for agent, action in actions_by_agent.items():
            if action < 0 or action >= self.config.action_count:
                raise ValueError(f"invalid action for {agent}: {action!r}")

        old_positions = {agent: self._positions[agent].copy() for agent in live_agents}
        proposed: dict[str, np.ndarray] = {}
        for agent in live_agents:
            self._headings[agent] = (self._headings[agent] + actions_by_agent[agent] - 1) % 4
            proposed[agent] = self._positions[agent] + self._DIRECTIONS[self._headings[agent]]

        deaths: dict[str, list[str]] = {agent: [] for agent in live_agents}
        for agent, next_position in proposed.items():
            if self._out_of_bounds(next_position):
                deaths[agent].append("out_of_bounds")
            elif self._grid[int(next_position[1]), int(next_position[0])]:
                deaths[agent].append("occupied_cell")

        if len(live_agents) == 2:
            a, b = live_agents
            if np.array_equal(proposed[a], proposed[b]):
                deaths[a].append("same_cell")
                deaths[b].append("same_cell")
            if np.array_equal(proposed[a], old_positions[b]) and np.array_equal(
                proposed[b], old_positions[a]
            ):
                deaths[a].append("cross_swap")
                deaths[b].append("cross_swap")

        for agent in live_agents:
            if deaths[agent]:
                self._alive[agent] = False
                self._death_causes[agent] = deaths[agent]
            else:
                self._positions[agent] = proposed[agent]
                self._mark_position(proposed[agent])

        self._step += 1
        alive_count = sum(self._alive.values())
        terminated = alive_count <= 1
        truncated = self._step >= self.config.max_steps and alive_count == len(AGENTS)
        self._done = terminated or truncated

        rewards = {agent: 0.0 for agent in AGENTS}
        if terminated and alive_count == 1:
            winner = next(agent for agent in AGENTS if self._alive[agent])
            rewards[winner] = 1.0
            for agent in AGENTS:
                if agent != winner:
                    rewards[agent] = -1.0

        return LineDuelStep(
            observations=self.observations(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos={
                "alive": dict(self._alive),
                "death_causes": {
                    agent: tuple(causes)
                    for agent, causes in self._death_causes.items()
                },
                "winner": self.winner(),
            },
        )

    def observations(self) -> dict[str, LineDuelObservation]:
        return {agent: self.observation(agent) for agent in AGENTS}

    def observation(self, ego_agent: str) -> LineDuelObservation:
        opponent_agent = _opponent(ego_agent)
        ego_position = self._positions[ego_agent]
        opponent_position = self._positions[opponent_agent]
        ego_heading = self._headings[ego_agent]
        forward = self._DIRECTIONS[ego_heading]
        right = self._DIRECTIONS[(ego_heading + 1) % 4]
        delta = opponent_position - ego_position
        return LineDuelObservation(
            ego_x=int(ego_position[0]),
            ego_y=int(ego_position[1]),
            ego_heading=ego_heading,
            left_clearance=self._clearance(ego_position, (ego_heading - 1) % 4),
            straight_clearance=self._clearance(ego_position, ego_heading),
            right_clearance=self._clearance(ego_position, (ego_heading + 1) % 4),
            opponent_dx_forward=int(np.dot(delta, forward)),
            opponent_dy_right=int(np.dot(delta, right)),
            opponent_heading_relative=(self._headings[opponent_agent] - ego_heading) % 4,
            opponent_alive=self._alive[opponent_agent],
            step=self._step,
        )

    def winner(self) -> str | None:
        live_agents = [agent for agent in AGENTS if self._alive[agent]]
        return live_agents[0] if len(live_agents) == 1 else None

    @property
    def death_causes(self) -> dict[str, tuple[str, ...]]:
        return {agent: tuple(causes) for agent, causes in self._death_causes.items()}

    def _out_of_bounds(self, position: np.ndarray) -> bool:
        x = int(position[0])
        y = int(position[1])
        return x < 0 or x >= self.config.width or y < 0 or y >= self.config.height

    def _mark_position(self, position: np.ndarray) -> None:
        if not self._out_of_bounds(position):
            self._grid[int(position[1]), int(position[0])] = True

    def _clearance(self, position: np.ndarray, heading: int) -> int:
        probe = position.copy()
        distance = 0
        while True:
            probe = probe + self._DIRECTIONS[heading]
            if self._out_of_bounds(probe) or self._grid[int(probe[1]), int(probe[0])]:
                return distance
            distance += 1


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._episodes: list[list[ReplayTransition]] = []

    def add_episode(self, transitions: list[ReplayTransition]) -> None:
        self._episodes.append(transitions)
        if len(self._episodes) > self.capacity:
            self._episodes = self._episodes[-self.capacity :]

    def sample(self, rng: np.random.Generator, batch_size: int) -> list[ReplayTransition]:
        transitions = list(self.transitions())
        if not transitions:
            return []
        indices = rng.integers(len(transitions), size=min(batch_size, len(transitions)))
        return [transitions[int(index)] for index in indices]

    def transitions(self) -> Iterable[ReplayTransition]:
        for episode in self._episodes:
            yield from episode

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def transition_count(self) -> int:
        return sum(len(episode) for episode in self._episodes)


class DummyMuZeroModel:
    """Shared tabular learner behind MuZero-shaped method names."""

    def __init__(self, action_count: int, width: int = 11, height: int = 11):
        self.action_count = action_count
        self.width = width
        self.height = height
        self.q_values: dict[StateKey, np.ndarray] = {}
        self.visit_counts: dict[StateKey, np.ndarray] = {}
        self.transition_counts: Counter[tuple[StateKey, int, StateKey | None]] = Counter()
        self.reward_sums: Counter[tuple[StateKey, int, StateKey | None]] = Counter()
        self.checkpoint_metadata: dict[str, object] = {}

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        action_count: int | None = None,
        width: int = 11,
        height: int = 11,
    ) -> "DummyMuZeroModel":
        """Load the compact npz format written by :meth:`checkpoint`."""

        with np.load(path, allow_pickle=False) as data:
            q_values = np.asarray(data["q_values"], dtype=np.float64)
            if q_values.ndim != 2:
                raise ValueError(f"checkpoint q_values must be rank-2: {path}")
            loaded_action_count = int(q_values.shape[1])
            if action_count is not None and loaded_action_count != action_count:
                raise ValueError(
                    f"checkpoint action_count {loaded_action_count} does not match "
                    f"expected {action_count}: {path}"
                )

            model = cls(
                action_count=loaded_action_count,
                width=width,
                height=height,
            )

            state_keys = [_state_key_from_json(raw) for raw in data["state_keys"]]
            if len(state_keys) != q_values.shape[0]:
                raise ValueError(f"checkpoint state_keys/q_values length mismatch: {path}")
            for state_key, row in zip(state_keys, q_values, strict=True):
                model.q_values[state_key] = row.copy()

            visit_counts = np.asarray(data["visit_counts"], dtype=np.float64)
            if visit_counts.shape != q_values.shape:
                raise ValueError(f"checkpoint visit_counts/q_values shape mismatch: {path}")
            for state_key, row in zip(state_keys, visit_counts, strict=True):
                model.visit_counts[state_key] = row.copy()

            dynamics_keys = list(data["dynamics_keys"])
            dynamics_counts = np.asarray(data["dynamics_counts"], dtype=np.int64)
            reward_sums = np.asarray(data["reward_sums"], dtype=np.float64)
            if len(dynamics_keys) != len(dynamics_counts) or len(dynamics_keys) != len(reward_sums):
                raise ValueError(f"checkpoint dynamics arrays length mismatch: {path}")
            for raw_key, count, reward_sum in zip(
                dynamics_keys,
                dynamics_counts,
                reward_sums,
                strict=True,
            ):
                key = _dynamics_key_from_json(raw_key)
                model.transition_counts[key] = int(count)
                model.reward_sums[key] = float(reward_sum)

            model.checkpoint_metadata = json.loads(_array_text(data["metadata"]))
            return model

    def representation(self, observation: LineDuelObservation) -> StateKey:
        return (
            _third_bucket(observation.ego_x, self.width),
            _third_bucket(observation.ego_y, self.height),
            observation.ego_heading,
            _bucket_clearance(observation.left_clearance),
            _bucket_clearance(observation.straight_clearance),
            _bucket_clearance(observation.right_clearance),
            _signed_bucket(observation.opponent_dx_forward),
            _signed_bucket(observation.opponent_dy_right),
            observation.opponent_heading_relative,
            int(observation.opponent_alive),
            min(8, observation.step // 10),
        )

    def prediction(self, state_key: StateKey) -> tuple[np.ndarray, float]:
        q_values = self._q(state_key).copy()
        value = float(np.max(q_values))
        policy = _softmax(q_values)
        return policy, value

    def dynamics(self, state_key: StateKey, action: int) -> tuple[StateKey | None, float]:
        candidates = {
            key: count
            for key, count in self.transition_counts.items()
            if key[0] == state_key and key[1] == action
        }
        if not candidates:
            return None, 0.0
        best_key = max(candidates, key=candidates.get)
        reward = float(self.reward_sums[best_key] / self.transition_counts[best_key])
        return best_key[2], reward

    def record_transition(
        self, state_key: StateKey, action: int, next_state_key: StateKey | None, reward: float
    ) -> None:
        key = (state_key, action, next_state_key)
        self.transition_counts[key] += 1
        self.reward_sums[key] += reward

    def update_q(
        self,
        state_key: StateKey,
        action: int,
        target: float,
        learning_rate: float,
    ) -> None:
        q_values = self._q(state_key)
        visits = self._visits(state_key)
        q_values[action] += learning_rate * (target - q_values[action])
        visits[action] += 1.0

    def checkpoint(self, path: Path, metadata: dict[str, object]) -> None:
        keys = sorted(self.q_values)
        q_values = (
            np.vstack([self.q_values[key] for key in keys])
            if keys
            else np.zeros((0, self.action_count))
        )
        visits = (
            np.vstack(
                [
                    self.visit_counts.get(key, np.zeros(self.action_count, dtype=np.float64))
                    for key in keys
                ]
            )
            if keys
            else np.zeros((0, self.action_count))
        )
        dynamics_keys = [
            json.dumps([list(state_key), action, None if next_key is None else list(next_key)])
            for state_key, action, next_key in self.transition_counts
        ]
        dynamics_counts = [self.transition_counts[key] for key in self.transition_counts]
        reward_sums = [self.reward_sums[key] for key in self.transition_counts]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            state_keys=np.array([json.dumps(list(key)) for key in keys]),
            q_values=q_values,
            visit_counts=visits,
            dynamics_keys=np.array(dynamics_keys),
            dynamics_counts=np.array(dynamics_counts, dtype=np.int64),
            reward_sums=np.array(reward_sums, dtype=np.float64),
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )

    def _q(self, state_key: StateKey) -> np.ndarray:
        if state_key not in self.q_values:
            self.q_values[state_key] = np.zeros(self.action_count, dtype=np.float64)
        return self.q_values[state_key]

    def _visits(self, state_key: StateKey) -> np.ndarray:
        if state_key not in self.visit_counts:
            self.visit_counts[state_key] = np.zeros(self.action_count, dtype=np.float64)
        return self.visit_counts[state_key]


class DummyPlanner:
    """One-step tabular planner; shaped like planning, intentionally tiny."""

    def __init__(self, discount: float, exploration_scale: float = 0.08):
        self.discount = discount
        self.exploration_scale = exploration_scale

    def select_action(
        self,
        model: DummyMuZeroModel,
        observation: LineDuelObservation,
        rng: np.random.Generator,
        epsilon: float,
        explore_unknown: bool,
    ) -> int:
        if float(rng.random()) < epsilon:
            return int(rng.integers(model.action_count))
        state_key = model.representation(observation)
        scores = []
        visits = model._visits(state_key)
        total = float(np.sum(visits))
        for action in range(model.action_count):
            next_key, reward = model.dynamics(state_key, action)
            _, next_value = model.prediction(next_key) if next_key is not None else (None, 0.0)
            score = reward + self.discount * next_value
            if explore_unknown:
                score += self.exploration_scale * np.sqrt(
                    np.log(total + 2.0) / (visits[action] + 1.0)
                )
            scores.append(score)
        return int(np.argmax(np.array(scores, dtype=np.float64)))


class DummyUpdater:
    def __init__(self, learning_rate: float = 0.25):
        self.learning_rate = learning_rate

    def update(
        self,
        model: DummyMuZeroModel,
        replay: ReplayBuffer,
        rng: np.random.Generator,
        batches: int,
        batch_size: int,
    ) -> dict[str, float]:
        losses = []
        for _ in range(batches):
            batch = replay.sample(rng, batch_size)
            for transition in batch:
                old_q = model._q(transition.observation_key)[transition.ego_action]
                model.update_q(
                    transition.observation_key,
                    transition.ego_action,
                    transition.target_return,
                    self.learning_rate,
                )
                losses.append(float((transition.target_return - old_q) ** 2))
        return {
            "updates": float(len(losses)),
            "mean_squared_td_error": float(np.mean(losses)) if losses else 0.0,
        }


def run_dummy_line_duel_training(
    *,
    iterations: int,
    episodes_per_iter: int,
    seed: int,
    output_dir: Path | None = None,
    eval_episodes: int = 20,
) -> dict[str, object]:
    """Run the dummy two-player line-duel training scaffold and optionally write artifacts."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if episodes_per_iter < 1:
        raise ValueError("episodes_per_iter must be at least 1")

    config = LineDuelConfig()
    _assert_env_canaries(config)
    rng = np.random.default_rng(seed)
    model = DummyMuZeroModel(
        action_count=config.action_count,
        width=config.width,
        height=config.height,
    )
    planner = DummyPlanner(discount=config.discount)
    updater = DummyUpdater()
    replay = ReplayBuffer(capacity=config.replay_capacity)

    iteration_metrics = []
    next_episode_id = 0
    for iteration in range(iterations):
        epsilon = _epsilon_for_iteration(iteration, iterations)
        train_stats = []
        for _ in range(episodes_per_iter):
            episode_seed = int(rng.integers(2**31 - 1))
            transitions, stats = _run_episode(
                config=config,
                model=model,
                planner=planner,
                seed=episode_seed,
                episode_id=next_episode_id,
                epsilon=epsilon,
                explore_unknown=True,
            )
            next_episode_id += 1
            for transition in transitions:
                model.record_transition(
                    transition.observation_key,
                    transition.ego_action,
                    transition.next_observation_key,
                    transition.reward,
                )
            replay.add_episode(transitions)
            train_stats.append(stats)

        update_metrics = updater.update(
            model=model,
            replay=replay,
            rng=rng,
            batches=max(1, episodes_per_iter),
            batch_size=32,
        )
        eval_metrics = evaluate_dummy_line_duel(
            model=model,
            planner=planner,
            config=config,
            episodes=eval_episodes,
            seed=seed + 50_000 + iteration * 1_000,
        )
        iteration_metrics.append(
            {
                "iteration": iteration,
                "epsilon": epsilon,
                "train": _summarize_episode_stats(train_stats),
                "update": update_metrics,
                "eval": eval_metrics,
                "replay": {
                    "episodes": replay.episode_count,
                    "transitions": replay.transition_count,
                },
            }
        )

    final_eval = evaluate_dummy_line_duel(
        model=model,
        planner=planner,
        config=config,
        episodes=eval_episodes,
        seed=seed + 900_000,
    )
    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_line_duel_training",
        "note": "Infrastructure scaffold only: shared tabular dummy trainer, not real MuZero.",
        "seed": seed,
        "iterations": iterations,
        "episodes_per_iter": episodes_per_iter,
        "eval_episodes": eval_episodes,
        "config": asdict(config),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
        },
        "rollout_policy": "shared_dummy_self_play",
        "model": {
            "type": "shared_tabular_numpy_dummy",
            "states": len(model.q_values),
            "learned_dynamics_edges": len(model.transition_counts),
        },
        "iteration_metrics": iteration_metrics,
        "final_eval": final_eval,
    }

    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(
            artifacts=artifacts,
            summary=summary,
            model=model,
            replay=replay,
            iteration_metrics=iteration_metrics,
        )
    return summary


def evaluate_dummy_line_duel(
    *,
    model: DummyMuZeroModel,
    planner: DummyPlanner,
    config: LineDuelConfig,
    episodes: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    stats = []
    for episode_id in range(episodes):
        episode_seed = int(rng.integers(2**31 - 1))
        _, episode_stats = _run_episode(
            config=config,
            model=model,
            planner=planner,
            seed=episode_seed,
            episode_id=episode_id,
            epsilon=0.0,
            explore_unknown=False,
        )
        stats.append(episode_stats)
    return _summarize_episode_stats(stats)


def _run_episode(
    *,
    config: LineDuelConfig,
    model: DummyMuZeroModel,
    planner: DummyPlanner,
    seed: int,
    episode_id: int,
    epsilon: float,
    explore_unknown: bool,
) -> tuple[list[ReplayTransition], EpisodeStats]:
    rng = np.random.default_rng(seed)
    env = LineDuelEnv(config)
    observations = env.reset(seed=seed)
    pending = []
    action_counts = {agent: [0, 0, 0] for agent in AGENTS}

    while True:
        joint_action = {}
        observation_keys = {}
        for agent in AGENTS:
            observation_keys[agent] = model.representation(observations[agent])
            action = planner.select_action(
                model=model,
                observation=observations[agent],
                rng=rng,
                epsilon=epsilon,
                explore_unknown=explore_unknown,
            )
            joint_action[agent] = action
            action_counts[agent][action] += 1

        step = env.step(joint_action)
        done = step.terminated or step.truncated
        for agent in AGENTS:
            next_key = None if done else model.representation(step.observations[agent])
            pending.append(
                (
                    episode_id,
                    seed,
                    env_step_index(step),
                    agent,
                    _opponent(agent),
                    observation_keys[agent],
                    joint_action[agent],
                    dict(joint_action),
                    step.rewards[agent],
                    next_key,
                    done,
                    step.truncated,
                )
            )

        observations = step.observations
        if done:
            break

    transitions = _attach_returns(pending, config.discount)
    final_rewards = {agent: 0.0 for agent in AGENTS}
    for transition in transitions[-len(AGENTS) :]:
        final_rewards[transition.ego_agent] = transition.reward
    truncated = any(transition.truncated for transition in transitions[-len(AGENTS) :])
    winner = env.winner() if not truncated else None
    stats = EpisodeStats(
        seed=seed,
        episode_id=episode_id,
        steps=max((transition.step for transition in transitions), default=-1) + 1,
        winner=winner,
        draw=winner is None and not truncated,
        truncated=truncated,
        rewards=final_rewards,
        action_counts_by_agent={agent: tuple(counts) for agent, counts in action_counts.items()},
        death_causes=env.death_causes,
    )
    return transitions, stats


def env_step_index(step: LineDuelStep) -> int:
    return int(next(iter(step.observations.values())).step - 1)


def _attach_returns(
    pending: list[
        tuple[
            int,
            int,
            int,
            str,
            str,
            StateKey,
            int,
            dict[str, int],
            float,
            StateKey | None,
            bool,
            bool,
        ]
    ],
    discount: float,
) -> list[ReplayTransition]:
    targets = {agent: 0.0 for agent in AGENTS}
    transitions = []
    for (
        episode_id,
        seed,
        step,
        ego_agent,
        opponent_agent,
        observation_key,
        ego_action,
        joint_action_by_agent,
        reward,
        next_observation_key,
        done,
        truncated,
    ) in reversed(pending):
        targets[ego_agent] = reward + discount * targets[ego_agent]
        transitions.append(
            ReplayTransition(
                episode_id=episode_id,
                seed=seed,
                step=step,
                ego_agent=ego_agent,
                opponent_agent=opponent_agent,
                observation_key=observation_key,
                ego_action=ego_action,
                joint_action_by_agent=joint_action_by_agent,
                reward=reward,
                next_observation_key=next_observation_key,
                done=done,
                truncated=truncated,
                target_return=targets[ego_agent],
                opponent_policy_id="shared_dummy_self_play",
            )
        )
    transitions.reverse()
    return transitions


def _summarize_episode_stats(stats: list[EpisodeStats]) -> dict[str, object]:
    if not stats:
        return {
            "episodes": 0,
            "mean_steps": 0.0,
            "max_steps": 0,
            "wins_by_player": {agent: 0 for agent in AGENTS},
            "draws": 0,
            "truncations": 0,
            "mean_reward_by_player": {agent: 0.0 for agent in AGENTS},
            "action_histogram_by_player": {agent: [0, 0, 0] for agent in AGENTS},
        }

    action_histogram = {
        agent: [
            sum(stat.action_counts_by_agent[agent][action] for stat in stats)
            for action in range(3)
        ]
        for agent in AGENTS
    }
    return {
        "episodes": len(stats),
        "mean_steps": float(np.mean([stat.steps for stat in stats])),
        "max_steps": int(max(stat.steps for stat in stats)),
        "wins_by_player": {
            agent: sum(1 for stat in stats if stat.winner == agent)
            for agent in AGENTS
        },
        "draws": sum(1 for stat in stats if stat.draw),
        "truncations": sum(1 for stat in stats if stat.truncated),
        "mean_reward_by_player": {
            agent: float(np.mean([stat.rewards[agent] for stat in stats]))
            for agent in AGENTS
        },
        "action_histogram_by_player": action_histogram,
    }


def _assert_env_canaries(config: LineDuelConfig) -> None:
    env = LineDuelEnv(config)
    env.reset(seed=0)
    for _ in range(2):
        env.step({"player_0": 1, "player_1": 1})
    step = env.step({"player_0": 1, "player_1": 1})
    if not step.terminated or step.rewards != {"player_0": 0.0, "player_1": 0.0}:
        raise AssertionError("same-cell draw canary failed")
    causes = step.infos["death_causes"]
    if "same_cell" not in causes["player_0"] or "same_cell" not in causes["player_1"]:
        raise AssertionError("same-cell death cause canary failed")

    env = _canary_env(
        config,
        positions={"player_0": (0, 5), "player_1": (5, 5)},
        headings={"player_0": 3, "player_1": 0},
    )
    step = env.step({"player_0": 1, "player_1": 1})
    if step.rewards != {"player_0": -1.0, "player_1": 1.0} or step.infos["winner"] != "player_1":
        raise AssertionError("wall-death winner reward canary failed")

    env = _canary_env(
        config,
        positions={"player_0": (4, 5), "player_1": (5, 5)},
        headings={"player_0": 1, "player_1": 3},
    )
    step = env.step({"player_0": 1, "player_1": 1})
    if not step.terminated or step.rewards != {"player_0": 0.0, "player_1": 0.0}:
        raise AssertionError("cross-swap draw canary failed")
    causes = step.infos["death_causes"]
    if "cross_swap" not in causes["player_0"] or "cross_swap" not in causes["player_1"]:
        raise AssertionError("cross-swap death cause canary failed")


def _canary_env(
    config: LineDuelConfig,
    *,
    positions: dict[str, tuple[int, int]],
    headings: dict[str, int],
) -> LineDuelEnv:
    env = LineDuelEnv(config)
    env.reset(seed=0)
    env._grid.fill(False)
    env._positions = {
        agent: np.array([positions[agent][0], positions[agent][1]], dtype=np.int16)
        for agent in AGENTS
    }
    env._headings = dict(headings)
    env._alive = {agent: True for agent in AGENTS}
    env._death_causes = {agent: [] for agent in AGENTS}
    env._step = 0
    env._done = False
    for agent in AGENTS:
        env._mark_position(env._positions[agent])
    return env


def _epsilon_for_iteration(iteration: int, iterations: int) -> float:
    if iterations == 1:
        return 0.2
    progress = iteration / (iterations - 1)
    return float(0.35 * (1.0 - progress) + 0.05 * progress)


def _third_bucket(value: int, width: int) -> int:
    return min(2, max(0, int(value * 3 / width)))


def _bucket_clearance(value: int) -> int:
    return min(value, 5)


def _signed_bucket(value: int) -> int:
    return max(0, min(10, value + 5))


def _array_text(value: object) -> str:
    if isinstance(value, np.ndarray):
        return str(value.item())
    return str(value)


def _state_key_from_json(value: object) -> StateKey:
    return tuple(int(part) for part in json.loads(_array_text(value)))


def _dynamics_key_from_json(value: object) -> tuple[StateKey, int, StateKey | None]:
    state_key, action, next_key = json.loads(_array_text(value))
    return (
        tuple(int(part) for part in state_key),
        int(action),
        None if next_key is None else tuple(int(part) for part in next_key),
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent {agent!r}")


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "checkpoint_npz": output_dir / "checkpoint.npz",
        "iteration_metrics_jsonl": output_dir / "iteration_metrics.jsonl",
        "replay_rows_jsonl": output_dir / "replay_rows.jsonl",
    }


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    model: DummyMuZeroModel,
    replay: ReplayBuffer,
    iteration_metrics: list[dict[str, object]],
) -> None:
    summary_path = artifacts["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model.checkpoint(
        artifacts["checkpoint_npz"],
        metadata={
            "kind": summary["kind"],
            "seed": summary["seed"],
            "iterations": summary["iterations"],
            "episodes_per_iter": summary["episodes_per_iter"],
            "note": summary["note"],
            "ruleset_id": RULESET_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
        },
    )
    with artifacts["iteration_metrics_jsonl"].open("w", encoding="utf-8") as handle:
        for item in iteration_metrics:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    with artifacts["replay_rows_jsonl"].open("w", encoding="utf-8") as handle:
        for transition in replay.transitions():
            handle.write(json.dumps(_transition_to_json(transition), sort_keys=True) + "\n")


def _transition_to_json(transition: ReplayTransition) -> dict[str, object]:
    return {
        "episode_id": transition.episode_id,
        "seed": transition.seed,
        "step": transition.step,
        "ego_agent": transition.ego_agent,
        "opponent_agent": transition.opponent_agent,
        "observation_key": list(transition.observation_key),
        "ego_action": transition.ego_action,
        "joint_action_by_agent": transition.joint_action_by_agent,
        "reward": transition.reward,
        "next_observation_key": (
            None
            if transition.next_observation_key is None
            else list(transition.next_observation_key)
        ),
        "done": transition.done,
        "truncated": transition.truncated,
        "target_return": transition.target_return,
        "opponent_policy_id": transition.opponent_policy_id,
        "ruleset_id": transition.ruleset_id,
        "observation_schema_id": transition.observation_schema_id,
        "reward_schema_id": transition.reward_schema_id,
    }
