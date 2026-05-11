"""Build self-play replay datasets for the visual dummy Pong lane."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.training.dummy_pong import ACTION_LABELS
from curvyzero.training.dummy_pong import ACTION_SCHEMA_ID
from curvyzero.training.dummy_pong import AGENTS
from curvyzero.training.dummy_pong import OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import RASTER_LEGEND
from curvyzero.training.dummy_pong import RASTER_OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import REWARD_SCHEMA_ID
from curvyzero.training.dummy_pong import RULESET_ID
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv
from curvyzero.training.dummy_pong import PongObservation
from curvyzero.training.dummy_pong_eval import BaselinePolicy
from curvyzero.training.dummy_pong_eval import LEARNED_POLICY_PREFIX
from curvyzero.training.dummy_pong_eval import LearnedCheckpointPolicy
from curvyzero.training.dummy_pong_eval import LearnedCheckpointPolicySpec
from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
from curvyzero.training.dummy_pong_eval import _load_learned_checkpoint_policies

SELFPLAY_REPLAY_SUMMARY_SCHEMA_ID = "dummy_pong_selfplay_replay_summary_v0"
SELFPLAY_REPLAY_ROW_SCHEMA_ID = "dummy_pong_selfplay_replay_row_v0"


@dataclass(frozen=True, slots=True)
class SelfPlayPolicySpec:
    policy_arg: str
    policy_id: str
    kind: str
    source: str
    learned_spec: LearnedCheckpointPolicySpec | None = None


class EpsilonGreedyPolicy:
    """Apply a small random-action override around an existing Pong policy."""

    def __init__(self, policy: BaselinePolicy, *, epsilon: float, seed: int):
        self.name = policy.name
        self._policy = policy
        self._epsilon = epsilon
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, episode_seed: int, agent: str) -> None:
        self._policy.reset(episode_seed, agent)
        self._rng = np.random.default_rng(self._base_seed + episode_seed + _agent_seed(agent))

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int:
        if self._epsilon > 0.0 and float(self._rng.random()) < self._epsilon:
            return int(self._rng.integers(len(ACTION_LABELS)))
        return int(self._policy.action(observation, raster_grid, agent))


def build_dummy_pong_selfplay_replay(
    *,
    games: int,
    seed: int,
    output_dir: Path | None = None,
    max_steps: int = 120,
    policy: str = "random_uniform",
    epsilon: float = 0.0,
) -> dict[str, object]:
    """Collect visual Pong self-play rows for both ego seats."""

    if games < 1:
        raise ValueError("games must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0.0, 1.0]")

    config = PongConfig(max_steps=max_steps)
    policy_spec = _load_policy_spec(policy)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    games_summary: list[dict[str, object]] = []
    run_id = (
        f"dummy-pong-selfplay-seed-{seed}-games-{games}-max-{max_steps}"
        f"-policy-{_policy_slug(policy_spec.policy_id)}-eps-{_epsilon_slug(epsilon)}"
    )
    match_id = f"{policy_spec.policy_id}_p0__{policy_spec.policy_id}_p1"

    for game_index in range(games):
        game_seed = int(rng.integers(2**31 - 1))
        game_rows, game_summary = _run_game(
            config=config,
            run_id=run_id,
            match_id=match_id,
            game_index=game_index,
            seed=game_seed,
            policy_seed=seed + game_index * 10_000,
            policy_spec=policy_spec,
            epsilon=epsilon,
        )
        rows.extend(game_rows)
        games_summary.append(game_summary)

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_selfplay_replay",
        "note": (
            "Small visual Pong self-play replay. The same behavior policy controls "
            "both seats, and each environment step emits one trainer-friendly row "
            "per ego agent."
        ),
        "run_id": run_id,
        "summary_schema_id": SELFPLAY_REPLAY_SUMMARY_SCHEMA_ID,
        "row_schema_id": SELFPLAY_REPLAY_ROW_SCHEMA_ID,
        "seed": seed,
        "games": games,
        "max_steps": max_steps,
        "policy": policy,
        "policy_id": policy_spec.policy_id,
        "epsilon": epsilon,
        "match_id": match_id,
        "total_rows": len(rows),
        "total_steps": sum(int(game["steps"]) for game in games_summary),
        "config": asdict(config),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "selfplay_replay_summary_schema_id": SELFPLAY_REPLAY_SUMMARY_SCHEMA_ID,
            "selfplay_replay_row_schema_id": SELFPLAY_REPLAY_ROW_SCHEMA_ID,
        },
        "raster": {
            "shape": [config.height, config.width],
            "encoding": "row-major digit strings over the raster legend",
            "legend": RASTER_LEGEND,
        },
        "selfplay": {
            "policy_by_agent": {agent: policy_spec.policy_id for agent in AGENTS},
            "epsilon_greedy_random_action_probability": epsilon,
            "row_filter": "all ego agents at every environment step",
            "row_action_fields": (
                "behavior_action_id/action_taken_id/target_action_id are all "
                "the ego action actually played by the self-play policy."
            ),
        },
        "return_targets": {
            "score_return": "terminal environment score from the ego perspective",
            "shaped_return": (
                "win=+1.0, loss=-1.0 + 0.5 * episode_steps/max_steps, "
                "timeout_or_draw=0.0"
            ),
        },
        "policy_spec": _policy_spec_summary(policy_spec),
        "action_labels": list(ACTION_LABELS),
        "action_histogram_by_ego_agent": _action_histogram(rows),
        "return_summary": _return_summary(rows),
        "outcome_summary": _outcome_summary(games_summary),
        "games_summary": games_summary,
    }
    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, rows=rows)
    return summary


def _run_game(
    *,
    config: PongConfig,
    run_id: str,
    match_id: str,
    game_index: int,
    seed: int,
    policy_seed: int,
    policy_spec: SelfPlayPolicySpec,
    epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    env = PongEnv(config)
    observations = env.reset(seed=seed)
    policy_by_agent = {agent: policy_spec.policy_id for agent in AGENTS}
    policies = {
        agent: _make_policy(policy_spec, seed=policy_seed + index * 997, epsilon=epsilon)
        for index, agent in enumerate(AGENTS)
    }
    for agent, agent_policy in policies.items():
        agent_policy.reset(seed, agent)

    rows: list[dict[str, Any]] = []
    total_rewards = {agent: 0.0 for agent in AGENTS}
    final_step = None

    while True:
        step_index = int(next(iter(observations.values())).step)
        raster_observation = env.raster_observation()
        raster_grid = _raster_grid(raster_observation)
        joint_action = {
            agent: _policy_action(
                policy=policies[agent],
                observation=observations[agent],
                raster_observation=raster_observation,
                agent=agent,
                config=config,
            )
            for agent in AGENTS
        }
        step = env.step(joint_action)
        next_raster_grid = _raster_grid(env.raster_observation())
        for agent in AGENTS:
            total_rewards[agent] += float(step.rewards[agent])
            rows.append(
                _replay_row(
                    run_id=run_id,
                    match_id=match_id,
                    game_index=game_index,
                    seed=seed,
                    step_index=step_index,
                    ego_agent=agent,
                    policy_by_agent=policy_by_agent,
                    observation=observations[agent],
                    raster_grid=raster_grid,
                    action_taken=joint_action[agent],
                    joint_action=joint_action,
                    reward_after_step=float(step.rewards[agent]),
                    next_raster_grid=next_raster_grid,
                    next_observation=step.observations[agent],
                    terminated=step.terminated,
                    truncated=step.truncated,
                    winner=step.infos["winner"] if step.terminated else None,
                    terminal_cause=_terminal_cause(
                        terminated=step.terminated,
                        truncated=step.truncated,
                        winner=step.infos["winner"] if step.terminated else None,
                    ),
                )
            )

        observations = step.observations
        final_step = step
        if step.terminated or step.truncated:
            break

    if final_step is None:
        raise RuntimeError("dummy Pong self-play game produced no steps")

    episode_steps = int(next(iter(final_step.observations.values())).step)
    winner = final_step.infos["winner"] if final_step.terminated else None
    terminal_cause = _terminal_cause(
        terminated=final_step.terminated,
        truncated=final_step.truncated,
        winner=winner,
    )
    score_returns_by_agent = {
        agent: _score_return(ego_agent=agent, winner=winner, terminated=final_step.terminated)
        for agent in AGENTS
    }
    shaped_returns_by_agent = {
        agent: _shaped_return(
            ego_agent=agent,
            winner=winner,
            terminated=final_step.terminated,
            episode_steps=episode_steps,
            max_steps=config.max_steps,
        )
        for agent in AGENTS
    }
    terminal_metadata = {
        "episode_steps": episode_steps,
        "max_steps": config.max_steps,
        "terminated": final_step.terminated,
        "truncated": final_step.truncated,
        "winner": winner,
        "terminal_cause": terminal_cause,
        "total_rewards": total_rewards,
        "score_returns_by_agent": score_returns_by_agent,
        "shaped_returns_by_agent": shaped_returns_by_agent,
        "last_hit": final_step.infos["last_hit"],
    }
    for row in rows:
        ego_agent = str(row["ego_agent"])
        row.update(
            {
                "episode_steps": episode_steps,
                "episode_terminated": final_step.terminated,
                "episode_truncated": final_step.truncated,
                "episode_winner": winner,
                "episode_terminal_cause": terminal_cause,
                "episode_terminal": dict(terminal_metadata),
                "score_return": float(score_returns_by_agent[ego_agent]),
                "shaped_return": float(shaped_returns_by_agent[ego_agent]),
            }
        )

    game_summary = {
        "match_id": match_id,
        "game_index": game_index,
        "seed": seed,
        "policy_by_agent": policy_by_agent,
        "steps": episode_steps,
        "winner": winner,
        "terminated": final_step.terminated,
        "truncated": final_step.truncated,
        "terminal_cause": terminal_cause,
        "total_rewards": total_rewards,
        "score_returns_by_agent": score_returns_by_agent,
        "shaped_returns_by_agent": shaped_returns_by_agent,
        "last_hit": final_step.infos["last_hit"],
    }
    return rows, game_summary


def _replay_row(
    *,
    run_id: str,
    match_id: str,
    game_index: int,
    seed: int,
    step_index: int,
    ego_agent: str,
    policy_by_agent: dict[str, str],
    observation: PongObservation,
    raster_grid: list[str],
    action_taken: int,
    joint_action: dict[str, int],
    reward_after_step: float,
    next_raster_grid: list[str],
    next_observation: PongObservation,
    terminated: bool,
    truncated: bool,
    winner: str | None,
    terminal_cause: str | None,
) -> dict[str, Any]:
    behavior_policy_id = policy_by_agent[ego_agent]
    action_label = ACTION_LABELS[action_taken]
    return {
        "schema_id": SELFPLAY_REPLAY_ROW_SCHEMA_ID,
        "run_id": run_id,
        "match_id": match_id,
        "game_index": game_index,
        "seed": seed,
        "step_index": step_index,
        "ego_agent": ego_agent,
        "opponent_agent": _opponent(ego_agent),
        "selfplay_policy_id": behavior_policy_id,
        "behavior_policy_id": behavior_policy_id,
        "target_policy_id": behavior_policy_id,
        "action_taken": {
            "action_id": action_taken,
            "label": action_label,
        },
        "action_taken_id": action_taken,
        "action_taken_label": action_label,
        "behavior_action_id": action_taken,
        "behavior_action_label": action_label,
        "target_action_id": action_taken,
        "target_action_label": action_label,
        "policy_by_agent": dict(policy_by_agent),
        "behavior_policy_by_agent": dict(policy_by_agent),
        "target_policy_by_agent": dict(policy_by_agent),
        "joint_action": {agent: int(action) for agent, action in joint_action.items()},
        "joint_action_by_agent": {
            agent: {
                "policy_id": policy_by_agent[agent],
                "behavior_policy_id": policy_by_agent[agent],
                "target_policy_id": policy_by_agent[agent],
                "action_id": int(action),
                "label": ACTION_LABELS[int(action)],
            }
            for agent, action in joint_action.items()
        },
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "raster_shape": [len(raster_grid), len(raster_grid[0]) if raster_grid else 0],
        "raster_grid": raster_grid,
        "observation": asdict(observation),
        "reward_after_step": reward_after_step,
        "next_raster_grid": next_raster_grid,
        "next_observation": asdict(next_observation),
        "terminated": terminated,
        "truncated": truncated,
        "terminated_after_step": terminated,
        "truncated_after_step": truncated,
        "winner_after_step": winner,
        "terminal_cause_after_step": terminal_cause,
    }


def _load_policy_spec(policy_arg: str) -> SelfPlayPolicySpec:
    if policy_arg == "random_uniform":
        return SelfPlayPolicySpec(
            policy_arg=policy_arg,
            policy_id="random_uniform",
            kind="rng_baseline",
            source="curvyzero.training.dummy_pong_eval.RandomUniformPolicy",
        )
    if policy_arg.startswith(LEARNED_POLICY_PREFIX):
        learned_specs = _load_learned_checkpoint_policies([policy_arg])
        learned_spec = learned_specs[0]
        return SelfPlayPolicySpec(
            policy_arg=policy_arg,
            policy_id=learned_spec.policy_id,
            kind="learned_checkpoint",
            source="curvyzero.training.dummy_pong_eval.LearnedCheckpointPolicy",
            learned_spec=learned_spec,
        )
    raise ValueError(
        f"policy must be 'random_uniform' or {LEARNED_POLICY_PREFIX}<checkpoint.npz>"
    )


def _make_policy(
    policy_spec: SelfPlayPolicySpec,
    *,
    seed: int,
    epsilon: float,
) -> BaselinePolicy:
    if policy_spec.policy_id == "random_uniform":
        base_policy: BaselinePolicy = RandomUniformPolicy(seed=seed)
    elif policy_spec.learned_spec is not None:
        base_policy = LearnedCheckpointPolicy(spec=policy_spec.learned_spec)
    else:
        raise ValueError(f"unknown self-play policy {policy_spec.policy_id!r}")
    if epsilon == 0.0:
        return base_policy
    return EpsilonGreedyPolicy(base_policy, epsilon=epsilon, seed=seed)


def _policy_action(
    *,
    policy: BaselinePolicy,
    observation: PongObservation,
    raster_observation: np.ndarray,
    agent: str,
    config: PongConfig,
) -> int:
    action = int(policy.action(observation, raster_observation, agent))
    if action < 0 or action >= config.action_count:
        raise ValueError(f"policy {policy.name!r} returned invalid action {action!r}")
    return action


def _score_return(
    *,
    ego_agent: str,
    winner: str | None,
    terminated: bool,
) -> float:
    if not terminated or winner is None:
        return 0.0
    return 1.0 if winner == ego_agent else -1.0


def _shaped_return(
    *,
    ego_agent: str,
    winner: str | None,
    terminated: bool,
    episode_steps: int,
    max_steps: int,
) -> float:
    if not terminated or winner is None:
        return 0.0
    if winner == ego_agent:
        return 1.0
    return -1.0 + 0.5 * (episode_steps / max_steps)


def _terminal_cause(
    *,
    terminated: bool,
    truncated: bool,
    winner: str | None,
) -> str | None:
    if terminated:
        return f"{winner}_scored"
    if truncated:
        return "max_steps"
    return None


def _raster_grid(grid: np.ndarray) -> list[str]:
    return ["".join(str(int(cell)) for cell in row) for row in grid.tolist()]


def _action_histogram(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    histograms: dict[str, Counter[str]] = {agent: Counter() for agent in AGENTS}
    for row in rows:
        histograms[row["ego_agent"]][row["action_taken_label"]] += 1
    return {
        agent: {label: int(histograms[agent][label]) for label in ACTION_LABELS}
        for agent in AGENTS
    }


def _return_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    return {
        "score_return_by_ego_agent": {
            agent: _return_stats(
                [float(row["score_return"]) for row in rows if row["ego_agent"] == agent]
            )
            for agent in AGENTS
        },
        "shaped_return_by_ego_agent": {
            agent: _return_stats(
                [float(row["shaped_return"]) for row in rows if row["ego_agent"] == agent]
            )
            for agent in AGENTS
        },
    }


def _return_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _outcome_summary(games_summary: list[dict[str, object]]) -> dict[str, object]:
    winners = Counter(row["winner"] for row in games_summary)
    return {
        "wins_by_agent": {agent: int(winners[agent]) for agent in AGENTS},
        "draws_or_truncations": int(winners[None]),
        "mean_steps": float(np.mean([row["steps"] for row in games_summary])),
        "truncations": int(sum(bool(row["truncated"]) for row in games_summary)),
    }


def _policy_spec_summary(policy_spec: SelfPlayPolicySpec) -> dict[str, object]:
    summary: dict[str, object] = {
        "policy_arg": policy_spec.policy_arg,
        "policy_id": policy_spec.policy_id,
        "kind": policy_spec.kind,
        "source": policy_spec.source,
    }
    if policy_spec.learned_spec is not None:
        learned_spec = policy_spec.learned_spec
        summary.update(
            {
                "checkpoint_spec": f"{LEARNED_POLICY_PREFIX}{learned_spec.checkpoint_path}",
                "checkpoint_path": str(learned_spec.checkpoint_path),
                "checkpoint_schema_id": learned_spec.checkpoint_schema_id,
                "checkpoint_metadata": learned_spec.checkpoint_metadata,
                "feature_encoding_id": learned_spec.feature_encoding_id,
            }
        )
    return summary


def _policy_slug(policy_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in policy_id).strip("_")


def _epsilon_slug(epsilon: float) -> str:
    return str(epsilon).replace(".", "p")


def _agent_seed(agent: str) -> int:
    if agent == "player_0":
        return 17
    if agent == "player_1":
        return 29
    raise ValueError(f"unknown agent {agent!r}")


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent {agent!r}")


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "replay_rows_jsonl": output_dir / "replay_rows.jsonl",
    }


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    rows: list[dict[str, Any]],
) -> None:
    artifacts["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with artifacts["replay_rows_jsonl"].open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
