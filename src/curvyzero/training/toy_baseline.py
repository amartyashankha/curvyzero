"""Local toy baselines for the CurvyTron environment.

This module is intentionally small: it gives the training-run coach a stable
local harness before any real learner exists.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from curvyzero.env import CurvyTronConfig
from curvyzero.env import CurvyTronEnv
from curvyzero.env import StepResult


class Policy(Protocol):
    name: str

    def reset(self, episode_seed: int) -> None: ...

    def action(self, env: CurvyTronEnv, player_idx: int) -> int: ...


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode: int
    seed: int
    policy_pair: str
    steps: int
    rewards: dict[str, float]
    winner: str | None
    terminated: bool
    truncated: bool
    death_ticks: dict[str, int]


@dataclass(frozen=True, slots=True)
class MatchupSummary:
    policy_pair: str
    episodes: int
    seed: int
    steps_total: int
    steps_mean: float
    reward_mean: dict[str, float]
    wins: dict[str, int]
    win_rate: dict[str, float]
    draws: int
    draw_rate: float
    terminated: int
    truncated: int


class RandomPolicy:
    """Uniform random legal actions."""

    name = "random"

    def __init__(self, action_count: int, seed: int):
        self._action_count = action_count
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, episode_seed: int) -> None:
        self._rng = np.random.default_rng(self._base_seed + episode_seed)

    def action(self, env: CurvyTronEnv, player_idx: int) -> int:
        return int(self._rng.integers(self._action_count))


class PrivilegedSurvivalHeuristic:
    """One-step privileged policy that steers toward open space.

    It reads ``env.state`` directly and scores each legal action by ray-casting
    ahead from the resulting heading. Ties prefer going straight when available.
    """

    name = "privileged_survival_heuristic"

    def reset(self, episode_seed: int) -> None:
        del episode_seed

    def action(self, env: CurvyTronEnv, player_idx: int) -> int:
        if env.state is None:
            raise RuntimeError("env must be reset before policy action")
        if not bool(env.state.alive[player_idx]):
            return 1 if env.config.allow_straight_action else 0

        legal_actions = range(env.config.action_count)
        scores = {
            action: self._score_action(env, player_idx=player_idx, action=action)
            for action in legal_actions
        }
        return max(scores, key=lambda action: (scores[action], self._tie_break(env, action)))

    def _score_action(self, env: CurvyTronEnv, player_idx: int, action: int) -> float:
        assert env.state is not None
        state = env.state
        heading = float(
            state.headings[player_idx] + env._action_to_turn(action) * env.config.turn_rate_radians
        )
        start = state.positions[player_idx]
        direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)

        clearance = self._ray_clearance(env, start, direction, player_idx)
        left = self._ray_clearance(env, start, self._rotated(direction, 0.45), player_idx)
        right = self._ray_clearance(env, start, self._rotated(direction, -0.45), player_idx)
        return clearance + 0.25 * min(left, right)

    def _ray_clearance(
        self, env: CurvyTronEnv, start: np.ndarray, direction: np.ndarray, player_idx: int
    ) -> float:
        assert env.state is not None
        cfg = env.config
        step = max(0.5, float(cfg.speed) * 0.5)
        max_distance = float(max(cfg.width, cfg.height))
        distance = step
        own_marker = player_idx + 1

        while distance <= max_distance:
            point = start + direction * distance
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if x < 0 or x >= cfg.width or y < 0 or y >= cfg.height:
                return distance
            occupied = int(env.state.occupancy[y, x])
            if occupied != 0 and not (occupied == own_marker and distance <= cfg.speed):
                return distance
            distance += step
        return max_distance

    def _tie_break(self, env: CurvyTronEnv, action: int) -> int:
        if env.config.allow_straight_action:
            return 1 if action == 1 else 0
        return 0

    def _rotated(self, direction: np.ndarray, radians: float) -> np.ndarray:
        cos_theta = float(np.cos(radians))
        sin_theta = float(np.sin(radians))
        return np.array(
            [
                direction[0] * cos_theta - direction[1] * sin_theta,
                direction[0] * sin_theta + direction[1] * cos_theta,
            ],
            dtype=np.float32,
        )


def run_toy_baselines(
    *,
    episodes: int,
    seed: int,
    max_steps: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Evaluate random-vs-random and heuristic-vs-random matchups."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    config = CurvyTronConfig()
    env = CurvyTronEnv(config)
    step_limit = config.max_ticks if max_steps is None else max_steps
    if step_limit < 1:
        raise ValueError("max_steps must be at least 1")
    matchups = [
        (
            RandomPolicy(config.action_count, seed=seed + 101),
            RandomPolicy(config.action_count, seed=seed + 202),
        ),
        (
            PrivilegedSurvivalHeuristic(),
            RandomPolicy(config.action_count, seed=seed + 303),
        ),
    ]

    summaries: list[MatchupSummary] = []
    records_by_matchup: dict[str, list[EpisodeRecord]] = {}
    for matchup_idx, (player_0_policy, player_1_policy) in enumerate(matchups):
        policy_pair = f"{player_0_policy.name}_vs_{player_1_policy.name}"
        matchup_seed = seed + matchup_idx * 100_000
        records = _run_matchup(
            env=env,
            policy_pair=policy_pair,
            player_0_policy=player_0_policy,
            player_1_policy=player_1_policy,
            episodes=episodes,
            seed=matchup_seed,
            max_steps=step_limit,
        )
        summaries.append(_summarize_records(policy_pair, episodes, matchup_seed, records))
        records_by_matchup[policy_pair] = records

    payload: dict[str, object] = {
        "kind": "curvyzero_toy_baseline",
        "seed": seed,
        "episodes": episodes,
        "max_steps": step_limit,
        "env": {
            "ruleset": config.ruleset,
            "rules_hash": config.rules_hash,
            "width": config.width,
            "height": config.height,
            "action_count": config.action_count,
            "action_repeat": config.action_repeat,
            "max_ticks": config.max_ticks,
        },
        "matchups": [asdict(summary) for summary in summaries],
    }
    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        payload["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts, payload, records_by_matchup)
    return payload


def _run_matchup(
    *,
    env: CurvyTronEnv,
    policy_pair: str,
    player_0_policy: Policy,
    player_1_policy: Policy,
    episodes: int,
    seed: int,
    max_steps: int,
) -> list[EpisodeRecord]:
    records = []
    for episode in range(episodes):
        episode_seed = seed + episode
        env.reset(seed=episode_seed)
        player_0_policy.reset(episode_seed)
        player_1_policy.reset(episode_seed)

        result: StepResult | None = None
        for step_idx in range(max_steps):
            actions = {
                "player_0": player_0_policy.action(env, 0),
                "player_1": player_1_policy.action(env, 1),
            }
            result = env.step(actions)
            if result.terminated["player_0"] or result.truncated["player_0"]:
                break

        assert result is not None
        step_count = step_idx + 1
        records.append(_episode_record(episode, episode_seed, policy_pair, step_count, env, result))
    return records


def _episode_record(
    episode: int,
    episode_seed: int,
    policy_pair: str,
    step_count: int,
    env: CurvyTronEnv,
    result: StepResult,
) -> EpisodeRecord:
    assert env.state is not None
    winner = None
    if result.rewards["player_0"] > result.rewards["player_1"]:
        winner = "player_0"
    elif result.rewards["player_1"] > result.rewards["player_0"]:
        winner = "player_1"

    return EpisodeRecord(
        episode=episode,
        seed=episode_seed,
        policy_pair=policy_pair,
        steps=step_count,
        rewards={agent: float(reward) for agent, reward in result.rewards.items()},
        winner=winner,
        terminated=bool(result.terminated["player_0"]),
        truncated=bool(result.truncated["player_0"]),
        death_ticks={agent: int(env.state.death_tick[idx]) for idx, agent in enumerate(env.agents)},
    )


def _summarize_records(
    policy_pair: str, episodes: int, seed: int, records: list[EpisodeRecord]
) -> MatchupSummary:
    agents = ("player_0", "player_1")
    wins = {agent: sum(record.winner == agent for record in records) for agent in agents}
    draws = sum(record.winner is None for record in records)
    steps_total = sum(record.steps for record in records)
    reward_mean = {
        agent: float(np.mean([record.rewards[agent] for record in records])) for agent in agents
    }
    return MatchupSummary(
        policy_pair=policy_pair,
        episodes=episodes,
        seed=seed,
        steps_total=steps_total,
        steps_mean=float(np.mean([record.steps for record in records])),
        reward_mean=reward_mean,
        wins=wins,
        win_rate={agent: wins[agent] / episodes for agent in agents},
        draws=draws,
        draw_rate=draws / episodes,
        terminated=sum(record.terminated for record in records),
        truncated=sum(record.truncated for record in records),
    )


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "episodes_jsonl": output_dir / "episodes.jsonl",
    }


def _write_artifacts(
    artifacts: dict[str, Path],
    payload: dict[str, object],
    records_by_matchup: dict[str, list[EpisodeRecord]],
) -> None:
    summary_path = artifacts["summary_json"]
    episodes_path = artifacts["episodes_jsonl"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with episodes_path.open("w", encoding="utf-8") as handle:
        for records in records_by_matchup.values():
            for record in records:
                handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
