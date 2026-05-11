"""Build small imitation replay datasets for the dummy Pong lane."""

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
from curvyzero.training.dummy_pong_eval import TrackBallPolicy

IMITATION_REPLAY_SUMMARY_SCHEMA_ID = "dummy_pong_imitation_replay_summary_v0"
IMITATION_REPLAY_ROW_SCHEMA_ID = "dummy_pong_imitation_replay_row_v0"


def build_dummy_pong_imitation_replay(
    *,
    games: int,
    seed: int,
    output_dir: Path | None = None,
    max_steps: int = 120,
) -> dict[str, object]:
    """Collect track-ball imitation rows over raster Pong observations."""

    if games < 1:
        raise ValueError("games must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    config = PongConfig(max_steps=max_steps)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    games_summary: list[dict[str, object]] = []
    run_id = f"dummy-pong-imitation-seed-{seed}-games-{games}-max-{max_steps}"

    for game_index in range(games):
        game_seed = int(rng.integers(2**31 - 1))
        game_rows, game_summary = _run_game(
            config=config,
            run_id=run_id,
            game_index=game_index,
            seed=game_seed,
        )
        rows.extend(game_rows)
        games_summary.append(game_summary)

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_imitation_replay",
        "note": (
            "Small supervised replay from the scripted track_ball policy over "
            "Pong raster observations. This is data prep, not a learned policy."
        ),
        "run_id": run_id,
        "summary_schema_id": IMITATION_REPLAY_SUMMARY_SCHEMA_ID,
        "row_schema_id": IMITATION_REPLAY_ROW_SCHEMA_ID,
        "seed": seed,
        "games": games,
        "max_steps": max_steps,
        "total_rows": len(rows),
        "total_steps": len(rows) // len(AGENTS),
        "config": asdict(config),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "imitation_replay_summary_schema_id": IMITATION_REPLAY_SUMMARY_SCHEMA_ID,
            "imitation_replay_row_schema_id": IMITATION_REPLAY_ROW_SCHEMA_ID,
        },
        "raster": {
            "shape": [config.height, config.width],
            "encoding": "row-major digit strings over the raster legend",
            "legend": RASTER_LEGEND,
        },
        "target_policy": {
            "policy_id": "track_ball",
            "source": "curvyzero.training.dummy_pong_eval.TrackBallPolicy",
            "target": "behavioral_cloning_action",
        },
        "action_labels": list(ACTION_LABELS),
        "action_histogram_by_agent": _action_histogram(rows),
        "games_summary": games_summary,
        "outcome_summary": _outcome_summary(games_summary),
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
    game_index: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    env = PongEnv(config)
    observations = env.reset(seed=seed)
    policies = {agent: TrackBallPolicy() for agent in AGENTS}
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
            rows.append(
                _replay_row(
                    run_id=run_id,
                    game_index=game_index,
                    seed=seed,
                    step_index=step_index,
                    ego_agent=agent,
                    observation=observations[agent],
                    raster_grid=raster_grid,
                    target_action=joint_action[agent],
                    joint_action=joint_action,
                    reward_after_step=float(step.rewards[agent]),
                    next_raster_grid=next_raster_grid,
                    next_observation=step.observations[agent],
                    terminated_after_step=step.terminated,
                    truncated_after_step=step.truncated,
                )
            )

        observations = step.observations
        final_step = step
        if step.terminated or step.truncated:
            break

    if final_step is None:
        raise RuntimeError("dummy Pong imitation game produced no steps")

    game_summary = {
        "game_index": game_index,
        "seed": seed,
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
    game_index: int,
    seed: int,
    step_index: int,
    ego_agent: str,
    observation: PongObservation,
    raster_grid: list[str],
    target_action: int,
    joint_action: dict[str, int],
    reward_after_step: float,
    next_raster_grid: list[str],
    next_observation: PongObservation,
    terminated_after_step: bool,
    truncated_after_step: bool,
) -> dict[str, Any]:
    return {
        "schema_id": IMITATION_REPLAY_ROW_SCHEMA_ID,
        "run_id": run_id,
        "game_index": game_index,
        "seed": seed,
        "step_index": step_index,
        "ego_agent": ego_agent,
        "opponent_agent": _opponent(ego_agent),
        "target_policy_id": "track_ball",
        "target_action_id": target_action,
        "target_action_label": ACTION_LABELS[target_action],
        "joint_action_by_agent": {
            agent: {"action_id": action, "label": ACTION_LABELS[action]}
            for agent, action in joint_action.items()
        },
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "raster_shape": [len(raster_grid), len(raster_grid[0]) if raster_grid else 0],
        "raster_grid": raster_grid,
        "observation": asdict(observation),
        "reward_after_step": reward_after_step,
        "next_raster_grid": next_raster_grid,
        "next_observation": asdict(next_observation),
        "terminated_after_step": terminated_after_step,
        "truncated_after_step": truncated_after_step,
    }


def _raster_grid(grid: np.ndarray) -> list[str]:
    return ["".join(str(int(cell)) for cell in row) for row in grid.tolist()]


def _action_histogram(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    histograms: dict[str, Counter[str]] = {agent: Counter() for agent in AGENTS}
    for row in rows:
        histograms[row["ego_agent"]][row["target_action_label"]] += 1
    return {
        agent: {label: histograms[agent][label] for label in ACTION_LABELS}
        for agent in AGENTS
    }


def _outcome_summary(games_summary: list[dict[str, object]]) -> dict[str, object]:
    winners = Counter(row["winner"] for row in games_summary)
    return {
        "wins_by_agent": {
            agent: int(winners[agent])
            for agent in AGENTS
        },
        "draws_or_truncations": int(winners[None]),
        "mean_steps": float(np.mean([row["steps"] for row in games_summary])),
        "truncations": int(sum(bool(row["truncated"]) for row in games_summary)),
    }


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
