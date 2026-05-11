"""Dummy single-player turning-survival training scaffold.

This module is infrastructure scaffolding, not a real MuZero implementation.
It keeps MuZero-shaped seams around representation, dynamics, prediction,
planning, replay, and updates while using a tiny tabular NumPy learner.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

StateKey = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class SurvivalConfig:
    width: int = 11
    height: int = 11
    max_steps: int = 80
    discount: float = 0.997
    replay_capacity: int = 512
    action_count: int = 3


@dataclass(frozen=True, slots=True)
class SurvivalObservation:
    x: int
    y: int
    heading: int
    left_clearance: int
    straight_clearance: int
    right_clearance: int
    step: int


@dataclass(frozen=True, slots=True)
class SurvivalStep:
    observation: SurvivalObservation
    reward: float
    terminated: bool
    info: dict[str, object]


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    state_key: StateKey
    action: int
    reward: float
    next_state_key: StateKey | None
    done: bool
    target_return: float


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    seed: int
    steps: int
    terminal_reward: float
    crashed: bool
    survived: bool
    action_counts: tuple[int, int, int]


class SoloTurningSurvivalEnv:
    """Tiny solo task: turn left/straight/right and avoid walls plus own trail."""

    _DIRECTIONS = (
        np.array([0, -1], dtype=np.int16),
        np.array([1, 0], dtype=np.int16),
        np.array([0, 1], dtype=np.int16),
        np.array([-1, 0], dtype=np.int16),
    )

    def __init__(self, config: SurvivalConfig | None = None):
        self.config = config or SurvivalConfig()
        self._grid = np.zeros((self.config.height, self.config.width), dtype=np.bool_)
        self._position = np.zeros(2, dtype=np.int16)
        self._heading = 1
        self._step = 0
        self._done = False

    def reset(self, seed: int | None = None) -> SurvivalObservation:
        rng = np.random.default_rng(seed)
        self._grid.fill(False)
        self._position = np.array([self.config.width // 2, self.config.height // 2], dtype=np.int16)
        self._heading = int(rng.integers(4))
        self._step = 0
        self._done = False
        self._mark_position()
        return self.observation()

    def step(self, action: int) -> SurvivalStep:
        if self._done:
            raise RuntimeError("reset must be called before stepping a finished episode")
        if action < 0 or action >= self.config.action_count:
            raise ValueError(f"invalid action {action!r}")

        self._heading = (self._heading + action - 1) % 4
        self._position = self._position + self._DIRECTIONS[self._heading]
        self._step += 1

        crashed = self._is_crash(self._position)
        survived = self._step >= self.config.max_steps and not crashed
        self._done = crashed or survived
        reward = -1.0 if crashed else 1.0 if survived else 0.0
        if not crashed:
            self._mark_position()

        return SurvivalStep(
            observation=self.observation(),
            reward=reward,
            terminated=self._done,
            info={"crashed": crashed, "survived": survived},
        )

    def observation(self) -> SurvivalObservation:
        return SurvivalObservation(
            x=int(self._position[0]),
            y=int(self._position[1]),
            heading=self._heading,
            left_clearance=self._clearance((self._heading - 1) % 4),
            straight_clearance=self._clearance(self._heading),
            right_clearance=self._clearance((self._heading + 1) % 4),
            step=self._step,
        )

    def _is_crash(self, position: np.ndarray) -> bool:
        x = int(position[0])
        y = int(position[1])
        return x < 0 or x >= self.config.width or y < 0 or y >= self.config.height or self._grid[y, x]

    def _mark_position(self) -> None:
        x = int(self._position[0])
        y = int(self._position[1])
        if 0 <= x < self.config.width and 0 <= y < self.config.height:
            self._grid[y, x] = True

    def _clearance(self, heading: int) -> int:
        position = self._position.copy()
        distance = 0
        while True:
            position = position + self._DIRECTIONS[heading]
            if self._is_crash(position):
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
    """Tabular learner behind MuZero-shaped method names."""

    def __init__(self, action_count: int):
        self.action_count = action_count
        self.planner_unknown_next_value: float | None = None
        self.q_values: dict[StateKey, np.ndarray] = {}
        self.visit_counts: dict[StateKey, np.ndarray] = {}
        self.transition_counts: Counter[tuple[StateKey, int, StateKey | None]] = Counter()
        self.reward_sums: Counter[tuple[StateKey, int, StateKey | None]] = Counter()

    def representation(self, observation: SurvivalObservation) -> StateKey:
        def bucket_clearance(value: int) -> int:
            return min(value, 5)

        return (
            _third_bucket(observation.x, 11),
            _third_bucket(observation.y, 11),
            observation.heading,
            bucket_clearance(observation.left_clearance),
            bucket_clearance(observation.straight_clearance),
            bucket_clearance(observation.right_clearance),
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

    def has_dynamics(self, state_key: StateKey, action: int) -> bool:
        return any(
            key[0] == state_key and key[1] == action
            for key in self.transition_counts
        )

    def record_transition(
        self, state_key: StateKey, action: int, next_state_key: StateKey | None, reward: float
    ) -> None:
        key = (state_key, action, next_state_key)
        self.transition_counts[key] += 1
        self.reward_sums[key] += reward

    def update_q(self, state_key: StateKey, action: int, target: float, learning_rate: float) -> None:
        q_values = self._q(state_key)
        visits = self._visits(state_key)
        q_values[action] += learning_rate * (target - q_values[action])
        visits[action] += 1.0

    def checkpoint(self, path: Path, metadata: dict[str, object]) -> None:
        keys = sorted(self.q_values)
        q_values = np.vstack([self.q_values[key] for key in keys]) if keys else np.zeros((0, 3))
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

    @classmethod
    def load_checkpoint(cls, path: Path) -> DummyMuZeroModel:
        with np.load(path, allow_pickle=False) as payload:
            state_keys = [_state_key_from_json(item) for item in payload["state_keys"]]
            q_values = np.asarray(payload["q_values"], dtype=np.float64)
            if q_values.ndim != 2:
                raise ValueError(f"checkpoint {path} q_values must be a 2D array")
            if len(state_keys) != q_values.shape[0]:
                raise ValueError(
                    f"checkpoint {path} has {len(state_keys)} state keys but "
                    f"{q_values.shape[0]} q rows"
                )

            model = cls(action_count=int(q_values.shape[1]))
            for state_key, values in zip(state_keys, q_values, strict=True):
                model.q_values[state_key] = values.copy()

            visit_counts = np.asarray(payload["visit_counts"], dtype=np.float64)
            if visit_counts.shape != q_values.shape:
                raise ValueError(
                    f"checkpoint {path} visit_counts shape {visit_counts.shape!r} does not "
                    f"match q_values shape {q_values.shape!r}"
                )
            for state_key, values in zip(state_keys, visit_counts, strict=True):
                model.visit_counts[state_key] = values.copy()

            for encoded_key, count, reward_sum in zip(
                payload["dynamics_keys"],
                payload["dynamics_counts"],
                payload["reward_sums"],
                strict=True,
            ):
                key = _dynamics_key_from_json(encoded_key)
                model.transition_counts[key] = int(count)
                model.reward_sums[key] = float(reward_sum)

            if "metadata" in payload:
                metadata = json.loads(_npz_text(payload["metadata"].item()))
                model.planner_unknown_next_value = _optional_float(
                    metadata.get("planner_unknown_next_value")
                )
        return model

    @staticmethod
    def checkpoint_metadata(path: Path) -> dict[str, object]:
        with np.load(path, allow_pickle=False) as payload:
            if "metadata" not in payload:
                return {}
            return json.loads(_npz_text(payload["metadata"].item()))

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

    def __init__(
        self,
        discount: float,
        exploration_scale: float = 0.08,
        safety_penalty: float = 10.0,
        unknown_next_value: float | None = None,
    ):
        self.discount = discount
        self.exploration_scale = exploration_scale
        self.safety_penalty = safety_penalty
        self.unknown_next_value = unknown_next_value

    def select_action(
        self,
        model: DummyMuZeroModel,
        observation: SurvivalObservation,
        rng: np.random.Generator,
        epsilon: float,
        explore_unknown: bool,
        safety_filter_epsilon: bool = False,
    ) -> int:
        clearances = _action_clearances(observation)
        if float(rng.random()) < epsilon:
            if safety_filter_epsilon:
                positive_actions = [
                    action for action, clearance in enumerate(clearances) if clearance > 0
                ]
                if positive_actions:
                    return int(rng.choice(positive_actions))
            return int(rng.integers(model.action_count))
        state_key = model.representation(observation)
        scores = []
        for action in range(model.action_count):
            has_dynamics = model.has_dynamics(state_key, action)
            next_key, reward = model.dynamics(state_key, action)
            if next_key is not None:
                _, next_value = model.prediction(next_key)
            elif has_dynamics:
                next_value = 0.0
            else:
                next_value = self._unknown_next_value(model)
            score = reward + self.discount * next_value
            if clearances[action] <= 0:
                score -= self.safety_penalty
            if explore_unknown:
                visits = model._visits(state_key)
                total = float(np.sum(visits))
                score += self.exploration_scale * np.sqrt(np.log(total + 2.0) / (visits[action] + 1.0))
            scores.append(score)
        return max(
            range(model.action_count),
            key=lambda action: (scores[action], clearances[action], 1 if action == 1 else 0, -action),
        )

    def _unknown_next_value(self, model: DummyMuZeroModel) -> float:
        if self.unknown_next_value is not None:
            return self.unknown_next_value
        if model.planner_unknown_next_value is not None:
            return model.planner_unknown_next_value
        return 0.0


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
                old_q = model._q(transition.state_key)[transition.action]
                model.update_q(
                    transition.state_key,
                    transition.action,
                    transition.target_return,
                    self.learning_rate,
                )
                losses.append(float((transition.target_return - old_q) ** 2))
        return {
            "updates": float(len(losses)),
            "mean_squared_td_error": float(np.mean(losses)) if losses else 0.0,
        }


def run_dummy_survival_training(
    *,
    iterations: int,
    episodes_per_iter: int,
    seed: int,
    output_dir: Path | None = None,
    eval_episodes: int = 20,
    checkpoint_every_iterations: int | None = None,
    safety_filter_epsilon: bool = False,
    planner_unknown_next_value: float | None = None,
) -> dict[str, object]:
    """Run the dummy solo survival training scaffold and optionally write artifacts."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if episodes_per_iter < 1:
        raise ValueError("episodes_per_iter must be at least 1")
    if checkpoint_every_iterations is not None and checkpoint_every_iterations < 1:
        raise ValueError("checkpoint_every_iterations must be at least 1 when set")
    if checkpoint_every_iterations is not None and output_dir is None:
        raise ValueError("output_dir is required when checkpoint_every_iterations is set")

    config = SurvivalConfig()
    rng = np.random.default_rng(seed)
    model = DummyMuZeroModel(action_count=config.action_count)
    model.planner_unknown_next_value = planner_unknown_next_value
    planner = DummyPlanner(discount=config.discount)
    updater = DummyUpdater()
    replay = ReplayBuffer(capacity=config.replay_capacity)

    iteration_metrics = []
    periodic_checkpoints = []
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
                epsilon=epsilon,
                explore_unknown=True,
                safety_filter_epsilon=safety_filter_epsilon,
            )
            for transition in transitions:
                model.record_transition(
                    transition.state_key,
                    transition.action,
                    transition.next_state_key,
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
        eval_metrics = evaluate_dummy_survival(
            model=model,
            planner=planner,
            config=config,
            episodes=eval_episodes,
            seed=seed + 50_000 + iteration * 1_000,
        )
        iteration_metric = {
            "iteration": iteration,
            "completed_iterations": iteration + 1,
            "epsilon": epsilon,
            "train": _summarize_episode_stats(train_stats),
            "update": update_metrics,
            "eval": eval_metrics,
            "replay": {
                "episodes": replay.episode_count,
                "transitions": replay.transition_count,
            },
        }
        iteration_metrics.append(iteration_metric)

        if (
            output_dir is not None
            and checkpoint_every_iterations is not None
            and (iteration + 1) % checkpoint_every_iterations == 0
        ):
            checkpoint_path = _write_periodic_checkpoint(
                model=model,
                output_dir=output_dir,
                seed=seed,
                iterations=iterations,
                episodes_per_iter=episodes_per_iter,
                iteration=iteration,
                eval_metrics=eval_metrics,
                safety_filter_epsilon=safety_filter_epsilon,
                planner_unknown_next_value=planner_unknown_next_value,
            )
            periodic_checkpoints.append(
                {
                    "iteration": iteration,
                    "completed_iterations": iteration + 1,
                    "path": str(checkpoint_path),
                    "eval": eval_metrics,
                }
            )

    final_eval = evaluate_dummy_survival(
        model=model,
        planner=planner,
        config=config,
        episodes=eval_episodes,
        seed=seed + 900_000,
    )
    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_survival_training",
        "note": "Infrastructure scaffold only: MuZero-shaped dummy trainer, not real MuZero.",
        "seed": seed,
        "iterations": iterations,
        "episodes_per_iter": episodes_per_iter,
        "eval_episodes": eval_episodes,
        "checkpoint_every_iterations": checkpoint_every_iterations,
        "safety_filter_epsilon": safety_filter_epsilon,
        "planner_unknown_next_value": planner_unknown_next_value,
        "config": asdict(config),
        "model": {
            "type": "tabular_numpy_dummy",
            "states": len(model.q_values),
            "learned_dynamics_edges": len(model.transition_counts),
        },
        "iteration_metrics": iteration_metrics,
        "periodic_checkpoints": periodic_checkpoints,
        "final_eval": final_eval,
    }

    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        if checkpoint_every_iterations is None:
            artifacts.pop("periodic_checkpoint_dir")
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(
            artifacts=artifacts,
            summary=summary,
            model=model,
            iteration_metrics=iteration_metrics,
        )
    return summary


def evaluate_dummy_survival(
    *,
    model: DummyMuZeroModel,
    planner: DummyPlanner,
    config: SurvivalConfig,
    episodes: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(episodes):
        episode_seed = int(rng.integers(2**31 - 1))
        _, episode_stats = _run_episode(
            config=config,
            model=model,
            planner=planner,
            seed=episode_seed,
            epsilon=0.0,
            explore_unknown=False,
        )
        stats.append(episode_stats)
    return _summarize_episode_stats(stats)


def _run_episode(
    *,
    config: SurvivalConfig,
    model: DummyMuZeroModel,
    planner: DummyPlanner,
    seed: int,
    epsilon: float,
    explore_unknown: bool,
    safety_filter_epsilon: bool = False,
) -> tuple[list[ReplayTransition], EpisodeStats]:
    rng = np.random.default_rng(seed)
    env = SoloTurningSurvivalEnv(config)
    observation = env.reset(seed=seed)
    pending = []
    action_counts = [0, 0, 0]
    terminal_reward = 0.0
    crashed = False
    survived = False

    while True:
        state_key = model.representation(observation)
        action = planner.select_action(
            model=model,
            observation=observation,
            rng=rng,
            epsilon=epsilon,
            explore_unknown=explore_unknown,
            safety_filter_epsilon=safety_filter_epsilon,
        )
        action_counts[action] += 1
        step = env.step(action)
        next_key = None if step.terminated else model.representation(step.observation)
        pending.append((state_key, action, step.reward, next_key, step.terminated))
        observation = step.observation
        if step.terminated:
            terminal_reward = step.reward
            crashed = bool(step.info["crashed"])
            survived = bool(step.info["survived"])
            break

    transitions = _attach_returns(pending, config.discount)
    stats = EpisodeStats(
        seed=seed,
        steps=len(transitions),
        terminal_reward=terminal_reward,
        crashed=crashed,
        survived=survived,
        action_counts=tuple(action_counts),
    )
    return transitions, stats


def _attach_returns(
    pending: list[tuple[StateKey, int, float, StateKey | None, bool]], discount: float
) -> list[ReplayTransition]:
    target = 0.0
    transitions = []
    for state_key, action, reward, next_key, done in reversed(pending):
        target = reward + discount * target
        transitions.append(
            ReplayTransition(
                state_key=state_key,
                action=action,
                reward=reward,
                next_state_key=next_key,
                done=done,
                target_return=target,
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
            "crash_rate": 0.0,
            "survival_rate": 0.0,
            "mean_terminal_reward": 0.0,
            "action_histogram": [0, 0, 0],
        }
    action_histogram = [
        sum(stat.action_counts[action] for stat in stats)
        for action in range(3)
    ]
    return {
        "episodes": len(stats),
        "mean_steps": float(np.mean([stat.steps for stat in stats])),
        "max_steps": int(max(stat.steps for stat in stats)),
        "crash_rate": float(np.mean([stat.crashed for stat in stats])),
        "survival_rate": float(np.mean([stat.survived for stat in stats])),
        "mean_terminal_reward": float(np.mean([stat.terminal_reward for stat in stats])),
        "action_histogram": action_histogram,
    }


def _epsilon_for_iteration(iteration: int, iterations: int) -> float:
    if iterations == 1:
        return 0.2
    progress = iteration / (iterations - 1)
    return float(0.35 * (1.0 - progress) + 0.05 * progress)


def _third_bucket(value: int, width: int) -> int:
    return min(2, max(0, int(value * 3 / width)))


def _action_clearances(observation: SurvivalObservation) -> tuple[int, int, int]:
    return (
        observation.left_clearance,
        observation.straight_clearance,
        observation.right_clearance,
    )


def _state_key_from_json(value: object) -> StateKey:
    items = json.loads(_npz_text(value))
    return _coerce_state_key(items)


def _dynamics_key_from_json(value: object) -> tuple[StateKey, int, StateKey | None]:
    state_key, action, next_key = json.loads(_npz_text(value))
    return (
        _coerce_state_key(state_key),
        int(action),
        None if next_key is None else _coerce_state_key(next_key),
    )


def _coerce_state_key(items: list[object]) -> StateKey:
    values = tuple(int(item) for item in items)
    if len(values) != 6:
        raise ValueError(f"state key must have 6 items, got {len(values)}")
    return (values[0], values[1], values[2], values[3], values[4], values[5])


def _npz_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "checkpoint_npz": output_dir / "checkpoint.npz",
        "periodic_checkpoint_dir": output_dir / "checkpoints",
        "iteration_metrics_jsonl": output_dir / "iteration_metrics.jsonl",
    }


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    model: DummyMuZeroModel,
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
            "checkpoint_every_iterations": summary["checkpoint_every_iterations"],
            "safety_filter_epsilon": summary["safety_filter_epsilon"],
            "planner_unknown_next_value": summary["planner_unknown_next_value"],
            "note": summary["note"],
        },
    )
    with artifacts["iteration_metrics_jsonl"].open("w", encoding="utf-8") as handle:
        for item in iteration_metrics:
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def _write_periodic_checkpoint(
    *,
    model: DummyMuZeroModel,
    output_dir: Path,
    seed: int,
    iterations: int,
    episodes_per_iter: int,
    iteration: int,
    eval_metrics: dict[str, object],
    safety_filter_epsilon: bool,
    planner_unknown_next_value: float | None,
) -> Path:
    completed_iterations = iteration + 1
    checkpoint_path = output_dir / "checkpoints" / f"iteration-{completed_iterations:04d}.npz"
    model.checkpoint(
        checkpoint_path,
        metadata={
            "kind": "curvyzero_dummy_survival_training_periodic_checkpoint",
            "seed": seed,
            "iterations": iterations,
            "episodes_per_iter": episodes_per_iter,
            "iteration": iteration,
            "completed_iterations": completed_iterations,
            "safety_filter_epsilon": safety_filter_epsilon,
            "planner_unknown_next_value": planner_unknown_next_value,
            "eval": eval_metrics,
            "note": "Periodic dummy survival checkpoint for checkpoint-selection sweeps.",
        },
    )
    return checkpoint_path
