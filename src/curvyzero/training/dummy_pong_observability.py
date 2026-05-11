"""Deterministic trace artifact harness for the dummy Pong lane."""

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
from curvyzero.training.dummy_pong import PADDLE_BOUNCE_RULE
from curvyzero.training.dummy_pong import PADDLE_BOUNCE_SCHEMA_ID
from curvyzero.training.dummy_pong import RASTER_LEGEND
from curvyzero.training.dummy_pong import RASTER_OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import REWARD_SCHEMA_ID
from curvyzero.training.dummy_pong import RULESET_ID
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv
from curvyzero.training.dummy_pong import PongObservation
from curvyzero.training.dummy_pong import PongStep
from curvyzero.training.dummy_pong_eval import BaselinePolicy
from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
from curvyzero.training.dummy_pong_eval import TrackBallPolicy

OBSERVABILITY_SUMMARY_SCHEMA_ID = "dummy_pong_observability_summary_v0"
OBSERVABILITY_GAME_SCHEMA_ID = "dummy_pong_observability_game_v0"
OBSERVABILITY_STEP_SCHEMA_ID = "dummy_pong_observability_step_v0"
OBSERVABILITY_FRAME_SCHEMA_ID = "dummy_pong_observability_frame_v0"
DEFAULT_OBSERVABILITY_MATCHUPS = (
    ("random_uniform", "track_ball"),
    ("track_ball", "random_uniform"),
    ("track_ball", "track_ball"),
)


def run_dummy_pong_observability(
    *,
    games_per_match: int,
    seed: int,
    output_dir: Path | None = None,
    max_steps: int = 40,
) -> dict[str, object]:
    """Play a small fixed policy set and optionally write trace artifacts."""

    if games_per_match < 1:
        raise ValueError("games_per_match must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    config = PongConfig(max_steps=max_steps)
    game_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    run_id = f"dummy-pong-observability-seed-{seed}-gpm-{games_per_match}-max-{max_steps}"

    for matchup_index, (player_0_policy, player_1_policy) in enumerate(
        DEFAULT_OBSERVABILITY_MATCHUPS
    ):
        match_id = f"{player_0_policy}_p0__{player_1_policy}_p1"
        for local_game_index in range(games_per_match):
            game_index = matchup_index * games_per_match + local_game_index
            game_seed = seed + matchup_index * 10_000 + local_game_index
            policy_seed = seed + matchup_index * 1_000 + local_game_index * 100
            game_row, rows, frames = _run_game(
                config=config,
                run_id=run_id,
                match_id=match_id,
                game_index=game_index,
                local_game_index=local_game_index,
                seed=game_seed,
                policy_seed=policy_seed,
                policy_by_agent={"player_0": player_0_policy, "player_1": player_1_policy},
            )
            game_rows.append(game_row)
            step_rows.extend(rows)
            frame_rows.extend(frames)

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_observability",
        "note": "Compact deterministic step traces for the project-owned dummy Pong lane.",
        "run_id": run_id,
        "summary_schema_id": OBSERVABILITY_SUMMARY_SCHEMA_ID,
        "game_row_schema_id": OBSERVABILITY_GAME_SCHEMA_ID,
        "step_row_schema_id": OBSERVABILITY_STEP_SCHEMA_ID,
        "frame_row_schema_id": OBSERVABILITY_FRAME_SCHEMA_ID,
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "seed": seed,
        "games_per_match": games_per_match,
        "max_steps": max_steps,
        "total_games": len(game_rows),
        "total_step_rows": len(step_rows),
        "total_frame_rows": len(frame_rows),
        "config": asdict(config),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "observability_summary_schema_id": OBSERVABILITY_SUMMARY_SCHEMA_ID,
            "observability_game_schema_id": OBSERVABILITY_GAME_SCHEMA_ID,
            "observability_step_schema_id": OBSERVABILITY_STEP_SCHEMA_ID,
            "observability_frame_schema_id": OBSERVABILITY_FRAME_SCHEMA_ID,
        },
        "raster": {
            "shape": [config.height, config.width],
            "encoding": "row-major digit strings over the raster legend",
            "legend": RASTER_LEGEND,
            "note": (
                "Raster frames are the intended visual/MuZero-facing Pong observation path; "
                "tabular ego observations remain debug and eval scaffolding."
            ),
        },
        "paddle_bounce": {
            "schema_id": PADDLE_BOUNCE_SCHEMA_ID,
            "rule": PADDLE_BOUNCE_RULE,
            "mini_north_star": (
                "Learn to choose off-center paddle returns to beat track_ball, "
                "not merely track the ball row."
            ),
        },
        "action_labels": list(ACTION_LABELS),
        "policies": _policy_specs(),
        "matchups": _summarize_matchups(game_rows),
    }
    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(
            artifacts=artifacts,
            summary=summary,
            game_rows=game_rows,
            step_rows=step_rows,
            frame_rows=frame_rows,
        )
    return summary


def _run_game(
    *,
    config: PongConfig,
    run_id: str,
    match_id: str,
    game_index: int,
    local_game_index: int,
    seed: int,
    policy_seed: int,
    policy_by_agent: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    env = PongEnv(config)
    observations = env.reset(seed=seed)
    policies = {
        agent: _make_policy(policy_id, seed=policy_seed + agent_index * 997)
        for agent_index, (agent, policy_id) in enumerate(policy_by_agent.items())
    }
    for agent, policy in policies.items():
        policy.reset(seed, agent)

    total_rewards = {agent: 0.0 for agent in AGENTS}
    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = [
        _frame_row(
            run_id=run_id,
            match_id=match_id,
            game_index=game_index,
            local_game_index=local_game_index,
            seed=seed,
            policy_by_agent=policy_by_agent,
            observations=observations,
            infos=_infos_from_observations(observations=observations),
            rewards={agent: 0.0 for agent in AGENTS},
            terminated=False,
            truncated=False,
            frame_id=_frame_id(match_id=match_id, game_index=game_index, step_index=0),
            config=config,
        )
    ]
    final_step: PongStep | None = None
    while True:
        raster_observation = env.raster_observation()
        joint_action = {
            agent: policies[agent].action(observations[agent], raster_observation, agent)
            for agent in AGENTS
        }
        step = env.step(joint_action)
        for agent in AGENTS:
            total_rewards[agent] += float(step.rewards[agent])
        step_index = int(next(iter(step.observations.values())).step)
        frame_id = _frame_id(match_id=match_id, game_index=game_index, step_index=step_index)
        rows.append(
            _step_row(
                run_id=run_id,
                match_id=match_id,
                game_index=game_index,
                local_game_index=local_game_index,
                seed=seed,
                policy_by_agent=policy_by_agent,
                step=step,
                joint_action=joint_action,
                frame_id=frame_id,
            )
        )
        frames.append(
            _frame_row(
                run_id=run_id,
                match_id=match_id,
                game_index=game_index,
                local_game_index=local_game_index,
                seed=seed,
                policy_by_agent=policy_by_agent,
                observations=step.observations,
                infos=step.infos,
                rewards=step.rewards,
                terminated=step.terminated,
                truncated=step.truncated,
                frame_id=frame_id,
                config=config,
            )
        )
        observations = step.observations
        final_step = step
        if step.terminated or step.truncated:
            break

    if final_step is None:
        raise RuntimeError("dummy Pong game produced no steps")

    final_state = _state_from_step(final_step)
    game_row = {
        "schema_id": OBSERVABILITY_GAME_SCHEMA_ID,
        "run_id": run_id,
        "match_id": match_id,
        "game_index": game_index,
        "local_game_index": local_game_index,
        "seed": seed,
        "policy_by_agent": dict(policy_by_agent),
        "steps": len(rows),
        "terminated": final_step.terminated,
        "truncated": final_step.truncated,
        "terminal_cause": _terminal_cause(final_step),
        "winner": final_step.infos["winner"] if final_step.terminated else None,
        "last_hit": final_step.infos["last_hit"],
        "last_hit_impact": final_step.infos["last_hit_impact"],
        "total_rewards": total_rewards,
        "final_ball": final_state["ball"],
        "final_paddles": final_state["paddles"],
    }
    return game_row, rows, frames


def _step_row(
    *,
    run_id: str,
    match_id: str,
    game_index: int,
    local_game_index: int,
    seed: int,
    policy_by_agent: dict[str, str],
    step: PongStep,
    joint_action: dict[str, int],
    frame_id: str,
) -> dict[str, Any]:
    state = _state_from_step(step)
    step_index = int(next(iter(step.observations.values())).step)
    return {
        "schema_id": OBSERVABILITY_STEP_SCHEMA_ID,
        "run_id": run_id,
        "match_id": match_id,
        "game_index": game_index,
        "local_game_index": local_game_index,
        "seed": seed,
        "step_index": step_index,
        "raster_frame_id": frame_id,
        "policy_by_agent": dict(policy_by_agent),
        "joint_action": {
            agent: {"action_id": action, "label": ACTION_LABELS[action]}
            for agent, action in joint_action.items()
        },
        "ball": state["ball"],
        "paddles": state["paddles"],
        "rewards": {agent: float(step.rewards[agent]) for agent in AGENTS},
        "terminated": step.terminated,
        "truncated": step.truncated,
        "terminal_cause": _terminal_cause(step),
        "winner": step.infos["winner"] if step.terminated else None,
        "last_hit": step.infos["last_hit"],
        "last_hit_impact": step.infos["last_hit_impact"],
        "observations_by_agent": {
            agent: _observation_dict(observation)
            for agent, observation in step.observations.items()
        },
    }


def _frame_row(
    *,
    run_id: str,
    match_id: str,
    game_index: int,
    local_game_index: int,
    seed: int,
    policy_by_agent: dict[str, str],
    observations: dict[str, PongObservation],
    infos: dict[str, object],
    rewards: dict[str, float],
    terminated: bool,
    truncated: bool,
    frame_id: str,
    config: PongConfig,
) -> dict[str, Any]:
    step_index = int(next(iter(observations.values())).step)
    return {
        "schema_id": OBSERVABILITY_FRAME_SCHEMA_ID,
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "frame_id": frame_id,
        "run_id": run_id,
        "match_id": match_id,
        "game_index": game_index,
        "local_game_index": local_game_index,
        "seed": seed,
        "step_index": step_index,
        "policy_by_agent": dict(policy_by_agent),
        "shape": [config.height, config.width],
        "encoding": "row-major digit strings over the raster legend",
        "legend": RASTER_LEGEND,
        "grid": _raster_grid(observations=observations, infos=infos, config=config),
        "terminated": terminated,
        "truncated": truncated,
        "terminal_cause": _terminal_cause_from_flags(infos=infos, terminated=terminated, truncated=truncated),
        "rewards": {agent: float(rewards[agent]) for agent in AGENTS},
    }


def _raster_grid(
    *,
    observations: dict[str, PongObservation],
    infos: dict[str, object],
    config: PongConfig,
) -> list[str]:
    grid = [[0 for _ in range(config.width)] for _ in range(config.height)]
    state = _state_from_observations(observations=observations, infos=infos)
    paddles = state["paddles"]
    for agent, value in (("player_0", 1), ("player_1", 2)):
        paddle = paddles[agent]
        x = int(paddle["x"])
        y = int(paddle["y"])
        for row in range(y, min(y + config.paddle_height, config.height)):
            if 0 <= x < config.width:
                grid[row][x] = value

    ball = state["ball"]
    ball_x = int(ball["x"])
    ball_y = int(ball["y"])
    if 0 <= ball_x < config.width and 0 <= ball_y < config.height:
        grid[ball_y][ball_x] = 4 if grid[ball_y][ball_x] else 3
    return ["".join(str(cell) for cell in row) for row in grid]


def _frame_id(*, match_id: str, game_index: int, step_index: int) -> str:
    return f"{match_id}:game-{game_index:04d}:step-{step_index:04d}"


def _state_from_step(step: PongStep) -> dict[str, object]:
    return _state_from_observations(observations=step.observations, infos=step.infos)


def _state_from_observations(
    *,
    observations: dict[str, PongObservation],
    infos: dict[str, object],
) -> dict[str, object]:
    ball = infos["ball"]
    paddle_y_by_agent = infos["paddles"]
    return {
        "ball": {
            "x": int(ball["x"]),
            "y": int(ball["y"]),
            "vx": int(ball["vx"]),
            "vy": int(ball["vy"]),
        },
        "paddles": {
            agent: {
                "x": int(observations[agent].ego_paddle_x),
                "y": int(paddle_y_by_agent[agent]),
            }
            for agent in AGENTS
        },
    }


def _terminal_cause(step: PongStep) -> str | None:
    return _terminal_cause_from_flags(
        infos=step.infos,
        terminated=step.terminated,
        truncated=step.truncated,
    )


def _terminal_cause_from_flags(
    *,
    infos: dict[str, object],
    terminated: bool,
    truncated: bool,
) -> str | None:
    if terminated:
        winner = infos["winner"]
        return f"{winner}_scored"
    if truncated:
        return "max_steps"
    return None


def _infos_from_observations(*, observations: dict[str, PongObservation]) -> dict[str, object]:
    player_0_observation = observations["player_0"]
    return {
        "winner": None,
        "score_agent": None,
        "last_hit": None,
        "last_hit_impact": None,
        "ball": {
            "x": player_0_observation.ego_paddle_x + player_0_observation.ball_dx_forward,
            "y": player_0_observation.ball_y,
            "vx": player_0_observation.ball_vx_forward,
            "vy": player_0_observation.ball_vy,
        },
        "paddles": {
            agent: observations[agent].ego_paddle_y
            for agent in AGENTS
        },
    }


def _observation_dict(observation: PongObservation) -> dict[str, object]:
    return asdict(observation)


def _make_policy(policy_id: str, *, seed: int) -> BaselinePolicy:
    if policy_id == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    if policy_id == "track_ball":
        return TrackBallPolicy()
    raise ValueError(f"unknown dummy Pong policy {policy_id!r}")


def _policy_specs() -> list[dict[str, object]]:
    return [
        {
            "policy_id": "random_uniform",
            "source": "curvyzero.training.dummy_pong_eval.RandomUniformPolicy",
            "kind": "rng_baseline",
        },
        {
            "policy_id": "track_ball",
            "source": "curvyzero.training.dummy_pong_eval.TrackBallPolicy",
            "kind": "scripted_baseline",
        },
    ]


def _summarize_matchups(game_rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in game_rows:
        grouped.setdefault(row["match_id"], []).append(row)

    summaries = []
    for match_id, rows in sorted(grouped.items()):
        wins_by_agent = Counter(row["winner"] for row in rows if row["winner"] is not None)
        wins_by_policy: Counter[str] = Counter()
        terminal_causes = Counter(row["terminal_cause"] for row in rows)
        for row in rows:
            winner = row["winner"]
            if winner is not None:
                wins_by_policy[row["policy_by_agent"][winner]] += 1
        summaries.append(
            {
                "match_id": match_id,
                "games": len(rows),
                "policy_by_agent": dict(rows[0]["policy_by_agent"]),
                "wins_by_agent": dict(sorted(wins_by_agent.items())),
                "wins_by_policy": dict(sorted(wins_by_policy.items())),
                "terminal_causes": dict(sorted(terminal_causes.items())),
                "mean_steps": float(np.mean([row["steps"] for row in rows])),
                "truncations": sum(row["truncated"] for row in rows),
            }
        )
    return summaries


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "games_jsonl": output_dir / "games.jsonl",
        "steps_jsonl": output_dir / "steps.jsonl",
        "frames_jsonl": output_dir / "frames.jsonl",
    }


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    game_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> None:
    artifacts["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with artifacts["games_jsonl"].open("w", encoding="utf-8") as handle:
        for row in game_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with artifacts["steps_jsonl"].open("w", encoding="utf-8") as handle:
        for row in step_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with artifacts["frames_jsonl"].open("w", encoding="utf-8") as handle:
        for row in frame_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
