"""Build score-reward replay datasets for the dummy Pong lane."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
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
from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
from curvyzero.training.dummy_pong_eval import TrackBallPolicy

SCORING_REPLAY_SUMMARY_SCHEMA_ID = "dummy_pong_scoring_replay_summary_v0"
SCORING_REPLAY_ROW_SCHEMA_ID = "dummy_pong_scoring_replay_row_v0"


def build_dummy_pong_scoring_replay(
    *,
    games_per_seat: int,
    seed: int,
    output_dir: Path | None = None,
    max_steps: int = 120,
    row_policy: str = "track_ball",
) -> dict[str, object]:
    """Collect scoring ego rows against random opponents in both seats."""

    if games_per_seat < 1:
        raise ValueError("games_per_seat must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if row_policy not in {"track_ball", "all"}:
        raise ValueError("row_policy must be 'track_ball' or 'all'")

    config = PongConfig(max_steps=max_steps)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    games_summary: list[dict[str, object]] = []
    run_id = f"dummy-pong-scoring-seed-{seed}-games-per-seat-{games_per_seat}-max-{max_steps}"
    seatings = (
        {"player_0": "track_ball", "player_1": "random_uniform"},
        {"player_0": "random_uniform", "player_1": "track_ball"},
    )

    for seating_index, behavior_policy_by_agent in enumerate(seatings):
        match_id = _match_id(behavior_policy_by_agent)
        for game_index in range(games_per_seat):
            seed_for_game = int(rng.integers(2**31 - 1))
            global_game_index = seating_index * games_per_seat + game_index
            game_rows, game_summary = _run_game(
                config=config,
                run_id=run_id,
                match_id=match_id,
                game_index=global_game_index,
                seat_game_index=game_index,
                seed=seed_for_game,
                behavior_policy_by_agent=behavior_policy_by_agent,
                row_policy=row_policy,
            )
            rows.extend(game_rows)
            games_summary.append(game_summary)

    row_filter = _row_filter_summary(row_policy)
    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_scoring_replay",
        "note": _summary_note(row_policy),
        "run_id": run_id,
        "summary_schema_id": SCORING_REPLAY_SUMMARY_SCHEMA_ID,
        "row_schema_id": SCORING_REPLAY_ROW_SCHEMA_ID,
        "seed": seed,
        "games_per_seat": games_per_seat,
        "games": len(games_summary),
        "max_steps": max_steps,
        "row_policy": row_policy,
        "total_rows": len(rows),
        "total_steps": sum(int(game["steps"]) for game in games_summary),
        "config": asdict(config),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "scoring_replay_summary_schema_id": SCORING_REPLAY_SUMMARY_SCHEMA_ID,
            "scoring_replay_row_schema_id": SCORING_REPLAY_ROW_SCHEMA_ID,
        },
        "raster": {
            "shape": [config.height, config.width],
            "encoding": "row-major digit strings over the raster legend",
            "legend": RASTER_LEGEND,
        },
        "row_filter": row_filter,
        "reward_source": {
            "policy": "score_delta_only",
            "environment_field": "PongStep.rewards[ego_agent]",
            "reward_schema_id": REWARD_SCHEMA_ID,
        },
        "action_labels": list(ACTION_LABELS),
        "matchups": _summarize_matchups(games_summary),
        "action_histogram_by_ego_agent": _action_histogram(rows),
        "outcome_summary": _outcome_summary(games_summary, rows),
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
    seat_game_index: int,
    seed: int,
    behavior_policy_by_agent: dict[str, str],
    row_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    env = PongEnv(config)
    observations = env.reset(seed=seed)
    policies = {
        agent: _make_policy(policy_id, seed=seed + index * 1000)
        for index, (agent, policy_id) in enumerate(behavior_policy_by_agent.items())
    }
    for agent, policy in policies.items():
        policy.reset(seed, agent)

    rows: list[dict[str, Any]] = []
    total_rewards = {agent: 0.0 for agent in AGENTS}
    final_step = None

    while True:
        step_index = int(next(iter(observations.values())).step)
        raster_observation = env.raster_observation()
        raster_grid = _raster_grid(raster_observation)
        joint_action = {
            agent: policies[agent].action(observations[agent], raster_observation, agent)
            for agent in AGENTS
        }
        step = env.step(joint_action)
        next_raster_grid = _raster_grid(env.raster_observation())
        for agent in AGENTS:
            total_rewards[agent] += float(step.rewards[agent])
            if not _include_ego_row(
                row_policy=row_policy,
                behavior_policy_id=behavior_policy_by_agent[agent],
            ):
                continue
            rows.append(
                _replay_row(
                    run_id=run_id,
                    match_id=match_id,
                    game_index=game_index,
                    seat_game_index=seat_game_index,
                    seed=seed,
                    step_index=step_index,
                    ego_agent=agent,
                    behavior_policy_by_agent=behavior_policy_by_agent,
                    observation=observations[agent],
                    raster_grid=raster_grid,
                    behavior_action=joint_action[agent],
                    joint_action=joint_action,
                    reward_after_step=float(step.rewards[agent]),
                    next_raster_grid=next_raster_grid,
                    next_observation=step.observations[agent],
                    terminated=step.terminated,
                    truncated=step.truncated,
                )
            )

        observations = step.observations
        final_step = step
        if step.terminated or step.truncated:
            break

    if final_step is None:
        raise RuntimeError("dummy Pong scoring game produced no steps")

    game_summary = {
        "match_id": match_id,
        "game_index": game_index,
        "seat_game_index": seat_game_index,
        "seed": seed,
        "behavior_policy_by_agent": dict(behavior_policy_by_agent),
        "track_ball_agent": _track_ball_agent(behavior_policy_by_agent),
        "steps": int(next(iter(final_step.observations.values())).step),
        "winner": final_step.infos["winner"] if final_step.terminated else None,
        "terminated": final_step.terminated,
        "truncated": final_step.truncated,
        "total_rewards": total_rewards,
        "last_hit": final_step.infos["last_hit"],
    }
    return rows, game_summary


def _replay_row(
    *,
    run_id: str,
    match_id: str,
    game_index: int,
    seat_game_index: int,
    seed: int,
    step_index: int,
    ego_agent: str,
    behavior_policy_by_agent: dict[str, str],
    observation: PongObservation,
    raster_grid: list[str],
    behavior_action: int,
    joint_action: dict[str, int],
    reward_after_step: float,
    next_raster_grid: list[str],
    next_observation: PongObservation,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    return {
        "schema_id": SCORING_REPLAY_ROW_SCHEMA_ID,
        "run_id": run_id,
        "match_id": match_id,
        "game_index": game_index,
        "seat_game_index": seat_game_index,
        "seed": seed,
        "step_index": step_index,
        "ego_agent": ego_agent,
        "opponent_agent": _opponent(ego_agent),
        "behavior_policy_id": behavior_policy_by_agent[ego_agent],
        "target_policy_id": behavior_policy_by_agent[ego_agent],
        "behavior_action_id": behavior_action,
        "behavior_action_label": ACTION_LABELS[behavior_action],
        "target_action_id": behavior_action,
        "target_action_label": ACTION_LABELS[behavior_action],
        "behavior_policy_by_agent": dict(behavior_policy_by_agent),
        "joint_action_by_agent": {
            agent: {
                "behavior_policy_id": behavior_policy_by_agent[agent],
                "action_id": action,
                "label": ACTION_LABELS[action],
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
    }


def _make_policy(policy_id: str, *, seed: int) -> TrackBallPolicy | RandomUniformPolicy:
    if policy_id == "track_ball":
        return TrackBallPolicy()
    if policy_id == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    raise ValueError(f"unknown policy_id {policy_id!r}")


def _include_ego_row(*, row_policy: str, behavior_policy_id: str) -> bool:
    if row_policy == "track_ball":
        return behavior_policy_id == "track_ball"
    if row_policy == "all":
        return True
    raise ValueError(f"unknown row_policy {row_policy!r}")


def _summary_note(row_policy: str) -> str:
    base = (
        "Score-delta reward replay from track_ball versus random_uniform in both seats. "
        "This is scoring replay data, not MuZero, self-play, or a learned policy."
    )
    if row_policy == "all":
        return (
            base
            + " Rows are emitted for both ego policies, so terminal reward rows include "
            "positive winning examples and negative losing examples."
        )
    return base + " Rows are emitted only for the track_ball-controlled ego."


def _row_filter_summary(row_policy: str) -> dict[str, object]:
    if row_policy == "all":
        return {
            "row_policy": "all",
            "included_ego_policy_ids": ["random_uniform", "track_ball"],
            "excluded_ego_policy_ids": [],
        }
    return {
        "row_policy": "track_ball",
        "included_ego_policy_id": "track_ball",
        "excluded_ego_policy_id": "random_uniform",
    }


def _raster_grid(grid: np.ndarray) -> list[str]:
    return ["".join(str(int(cell)) for cell in row) for row in grid.tolist()]


def _match_id(behavior_policy_by_agent: dict[str, str]) -> str:
    return "__".join(
        f"{behavior_policy_by_agent[agent]}_{agent.replace('player_', 'p')}"
        for agent in AGENTS
    )


def _track_ball_agent(behavior_policy_by_agent: dict[str, str]) -> str:
    agents = [
        agent
        for agent, policy_id in behavior_policy_by_agent.items()
        if policy_id == "track_ball"
    ]
    if len(agents) != 1:
        raise ValueError(f"expected exactly one track_ball agent, got {agents!r}")
    return agents[0]


def _action_histogram(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    histograms: dict[str, Counter[str]] = {agent: Counter() for agent in AGENTS}
    for row in rows:
        histograms[row["ego_agent"]][row["target_action_label"]] += 1
    return {
        agent: {label: int(histograms[agent][label]) for label in ACTION_LABELS}
        for agent in AGENTS
        if histograms[agent]
    }


def _outcome_summary(
    games_summary: list[dict[str, object]],
    rows: list[dict[str, Any]],
) -> dict[str, object]:
    winners = Counter(row["winner"] for row in games_summary)
    nonzero_reward_rows = [row for row in rows if float(row["reward_after_step"]) != 0.0]
    positive_reward_rows = [row for row in rows if float(row["reward_after_step"]) > 0.0]
    negative_reward_rows = [row for row in rows if float(row["reward_after_step"]) < 0.0]
    reward_totals: Counter[str] = Counter()
    for row in rows:
        reward_totals[row["ego_agent"]] += float(row["reward_after_step"])
    return {
        "wins_by_agent": {agent: int(winners[agent]) for agent in AGENTS},
        "draws_or_truncations": int(winners[None]),
        "mean_steps": float(np.mean([row["steps"] for row in games_summary])),
        "truncations": int(sum(bool(row["truncated"]) for row in games_summary)),
        "nonzero_reward_rows": len(nonzero_reward_rows),
        "positive_reward_rows": len(positive_reward_rows),
        "negative_reward_rows": len(negative_reward_rows),
        "reward_total_by_ego_agent": {
            agent: float(reward_totals[agent])
            for agent in AGENTS
            if reward_totals[agent] != 0.0
        },
    }


def _summarize_matchups(games_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in games_summary:
        groups.setdefault(str(row["match_id"]), []).append(row)
    summaries = []
    for match_id, rows in sorted(groups.items()):
        winners = Counter(row["winner"] for row in rows)
        summaries.append(
            {
                "match_id": match_id,
                "games": len(rows),
                "behavior_policy_by_agent": dict(rows[0]["behavior_policy_by_agent"]),
                "track_ball_agent": rows[0]["track_ball_agent"],
                "wins_by_agent": {agent: int(winners[agent]) for agent in AGENTS},
                "truncations": int(sum(bool(row["truncated"]) for row in rows)),
                "mean_steps": float(np.mean([row["steps"] for row in rows])),
            }
        )
    return summaries


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
