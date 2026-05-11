"""Probe scoreability of dummy Pong contact-pressure reset states.

This is a local diagnostic, not training. It samples real
``pong_reset_profile=contact_pressure`` starts, clones each start, sweeps the
legal ego actions, and rolls out with simple scripted policies under the true
sparse environment reward.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.training.dummy_pong import ACTION_LABELS
from curvyzero.training.dummy_pong import AGENTS
from curvyzero.training.dummy_pong import PADDLE_BOUNCE_SCHEMA_ID
from curvyzero.training.dummy_pong import REWARD_SCHEMA_ID
from curvyzero.training.dummy_pong import RULESET_ID
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv
from curvyzero.training.dummy_pong_eval import BaselinePolicy
from curvyzero.training.dummy_pong_eval import LaggedTrackBallPolicy
from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
from curvyzero.training.dummy_pong_eval import TrackBallPolicy

PROBE_SCHEMA_ID = "dummy_pong_contact_pressure_scoreability_probe_v0"
DEFAULT_OPPONENT_POLICIES = ("track_ball", "lagged_track_ball_1", "stay")


@dataclass(frozen=True, slots=True)
class EnvSnapshot:
    player_0_y: int
    player_1_y: int
    ball_x: int
    ball_y: int
    ball_vx: int
    ball_vy: int
    step: int
    reset_info: dict[str, object]


class StayPolicy:
    name = "stay"

    def reset(self, episode_seed: int, agent: str) -> None:
        del episode_seed, agent

    def action(self, observation: object, raster_grid: np.ndarray, agent: str) -> int:
        del observation, raster_grid, agent
        return 1


def probe_contact_pressure_scoreability(
    *,
    states_per_pressure_agent: int,
    seed: int,
    max_steps: int = 64,
    width: int = 15,
    height: int = 9,
    paddle_height: int = 3,
    reset_contact_distance_min: int = 2,
    reset_contact_distance_max: int = 3,
    opponent_policies: tuple[str, ...] = DEFAULT_OPPONENT_POLICIES,
    ego_action_mode: str = "until_contact",
    output_dir: Path | None = None,
) -> dict[str, object]:
    if states_per_pressure_agent < 1:
        raise ValueError("states_per_pressure_agent must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if ego_action_mode not in {"first_step", "until_contact"}:
        raise ValueError("ego_action_mode must be first_step or until_contact")

    rows: list[dict[str, Any]] = []
    reset_summaries: list[dict[str, object]] = []
    for pressure_agent in AGENTS:
        config = PongConfig(
            width=width,
            height=height,
            paddle_height=paddle_height,
            max_steps=max_steps,
            reset_profile="contact_pressure",
            reset_pressure_agent=pressure_agent,
            reset_contact_distance_min=reset_contact_distance_min,
            reset_contact_distance_max=reset_contact_distance_max,
        )
        for state_index in range(states_per_pressure_agent):
            reset_seed = seed + _agent_seed_offset(pressure_agent) + state_index
            snapshot = _sample_reset_snapshot(config=config, seed=reset_seed)
            reset_summaries.append(
                {
                    "reset_state_id": _reset_state_id(pressure_agent, reset_seed),
                    "seed": reset_seed,
                    "pressure_agent": pressure_agent,
                    "snapshot": asdict(snapshot),
                }
            )
            for opponent_policy in opponent_policies:
                for ego_action in range(config.action_count):
                    rows.append(
                        _rollout_candidate(
                            config=config,
                            snapshot=snapshot,
                            reset_seed=reset_seed,
                            pressure_agent=pressure_agent,
                            opponent_policy_name=opponent_policy,
                            ego_action=ego_action,
                            ego_action_mode=ego_action_mode,
                        )
                    )

    summary = _summarize(
        rows=rows,
        reset_summaries=reset_summaries,
        states_per_pressure_agent=states_per_pressure_agent,
        seed=seed,
        max_steps=max_steps,
        opponent_policies=opponent_policies,
        ego_action_mode=ego_action_mode,
        config={
            "width": width,
            "height": height,
            "paddle_height": paddle_height,
            "reset_contact_distance_min": reset_contact_distance_min,
            "reset_contact_distance_max": reset_contact_distance_max,
        },
    )
    if output_dir is not None:
        artifacts = {
            "summary_json": output_dir / "summary.json",
            "rows_jsonl": output_dir / "rows.jsonl",
        }
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, rows=rows)
    return summary


def _sample_reset_snapshot(*, config: PongConfig, seed: int) -> EnvSnapshot:
    env = PongEnv(config)
    env.reset(seed=seed)
    return _snapshot(env)


def _rollout_candidate(
    *,
    config: PongConfig,
    snapshot: EnvSnapshot,
    reset_seed: int,
    pressure_agent: str,
    opponent_policy_name: str,
    ego_action: int,
    ego_action_mode: str,
) -> dict[str, Any]:
    env = PongEnv(config)
    env.reset(seed=reset_seed)
    _install_snapshot(env, snapshot)

    ego_agent = pressure_agent
    opponent_agent = _opponent(ego_agent)
    ego_policy = TrackBallPolicy()
    opponent_policy = _make_policy(
        opponent_policy_name,
        seed=reset_seed + ego_action * 10_000,
    )
    ego_policy.reset(reset_seed, ego_agent)
    opponent_policy.reset(reset_seed, opponent_agent)

    first_contact: dict[str, object] | None = None
    final_step = None
    score_return = 0.0
    nonzero_reward_steps = 0
    released_to_ego_baseline = False
    action_trace: list[dict[str, int]] = []
    for rollout_step in range(config.max_steps):
        observations = env.observations()
        raster_grid = env.raster_observation()
        if released_to_ego_baseline:
            current_ego_action = int(
                ego_policy.action(observations[ego_agent], raster_grid, ego_agent)
            )
        else:
            current_ego_action = ego_action
        joint_action = {
            ego_agent: current_ego_action,
            opponent_agent: int(
                opponent_policy.action(
                    observations[opponent_agent],
                    raster_grid,
                    opponent_agent,
                )
            ),
        }
        if len(action_trace) < 12:
            action_trace.append(dict(joint_action))

        final_step = env.step(joint_action)
        ego_reward = float(final_step.rewards[ego_agent])
        score_return += ego_reward
        if ego_reward != 0.0:
            nonzero_reward_steps += 1

        impact = final_step.infos["last_hit_impact"]
        if first_contact is None and impact is not None:
            first_contact = {
                "step": int(next(iter(final_step.observations.values())).step),
                **dict(impact),
            }
            if ego_action_mode == "until_contact":
                released_to_ego_baseline = True
        if ego_action_mode == "first_step" and rollout_step == 0:
            released_to_ego_baseline = True

        if final_step.terminated or final_step.truncated:
            break

    if final_step is None:
        raise RuntimeError("candidate rollout did not run")

    terminal_step = int(next(iter(final_step.observations.values())).step)
    return {
        "schema_id": PROBE_SCHEMA_ID,
        "ruleset_id": RULESET_ID,
        "reward_schema_id": REWARD_SCHEMA_ID,
        "paddle_bounce_schema_id": PADDLE_BOUNCE_SCHEMA_ID,
        "reset_state_id": _reset_state_id(pressure_agent, reset_seed),
        "reset_seed": reset_seed,
        "pressure_agent": pressure_agent,
        "ego_agent": ego_agent,
        "opponent_agent": opponent_agent,
        "opponent_policy": opponent_policy_name,
        "ego_continuation_policy": "track_ball",
        "ego_action_mode": ego_action_mode,
        "candidate_ego_action": ego_action,
        "candidate_ego_action_label": ACTION_LABELS[ego_action],
        "reset": dict(snapshot.reset_info),
        "initial_state": asdict(snapshot),
        "first_contact": first_contact,
        "first_contact_agent": None if first_contact is None else first_contact["agent"],
        "first_contact_step": None if first_contact is None else first_contact["step"],
        "impact_offset": None if first_contact is None else first_contact["impact_offset"],
        "outgoing_ball_vy": (
            None if first_contact is None else first_contact["outgoing_ball_vy"]
        ),
        "score_return": score_return,
        "ego_reward": float(final_step.rewards[ego_agent]),
        "winner": final_step.infos["winner"],
        "terminated": bool(final_step.terminated),
        "truncated": bool(final_step.truncated),
        "survival_steps": terminal_step,
        "nonzero_reward_steps": nonzero_reward_steps,
        "final_state": {
            "ball": dict(final_step.infos["ball"]),
            "paddles": dict(final_step.infos["paddles"]),
        },
        "action_trace_prefix": action_trace,
    }


def _summarize(
    *,
    rows: list[dict[str, Any]],
    reset_summaries: list[dict[str, object]],
    states_per_pressure_agent: int,
    seed: int,
    max_steps: int,
    opponent_policies: tuple[str, ...],
    ego_action_mode: str,
    config: dict[str, object],
) -> dict[str, object]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["reset_state_id"]), str(row["opponent_policy"])), []).append(row)

    group_summaries = [
        _summarize_group(reset_state_id, opponent_policy, group_rows)
        for (reset_state_id, opponent_policy), group_rows in sorted(groups.items())
    ]
    action_sensitive_groups = [
        group for group in group_summaries if group["action_sensitive_any"]
    ]
    scoreable_groups = [group for group in group_summaries if group["scoreable"]]
    score_return_spread_groups = [
        group for group in group_summaries if group["score_return_differs"]
    ]

    row_wins = sum(float(row["ego_reward"]) > 0.0 for row in rows)
    row_scores = sum(float(row["ego_reward"]) != 0.0 for row in rows)
    terminal_histogram = Counter(
        _terminal_signature(row)
        for row in rows
    )
    best_action_histogram = Counter()
    for group in group_summaries:
        best_actions = group["best_ego_actions_by_score_then_survival"]
        if len(best_actions) == len(ACTION_LABELS):
            best_action_histogram["all_tied"] += 1
        else:
            for action in best_actions:
                best_action_histogram[str(action)] += 1

    return {
        "kind": "curvyzero_dummy_pong_contact_pressure_scoreability_probe",
        "probe_schema_id": PROBE_SCHEMA_ID,
        "question": (
            "Do real contact-pressure reset states have legal ego actions that "
            "change contact angle, sparse terminal score, or survival?"
        ),
        "note": (
            "This probe preserves PongEnv sparse reward. It changes only seeded "
            "reset profile, candidate ego action, and scripted rollout policies."
        ),
        "seed": seed,
        "states_per_pressure_agent": states_per_pressure_agent,
        "pressure_agents": list(AGENTS),
        "sampled_reset_states": len(reset_summaries),
        "opponent_policies": list(opponent_policies),
        "ego_action_mode": ego_action_mode,
        "max_steps": max_steps,
        "config": {
            **config,
            "reset_profile": "contact_pressure",
            "reset_pressure_agent": "swept_player_0_player_1",
        },
        "action_labels": list(ACTION_LABELS),
        "reward_contract": {
            "schema_id": REWARD_SCHEMA_ID,
            "terminal_score": {"ego_scores": 1.0, "ego_loses": -1.0},
            "non_score_step": 0.0,
        },
        "rows": len(rows),
        "groups": len(group_summaries),
        "row_score_probability": 0.0 if not rows else row_wins / len(rows),
        "row_nonzero_score_rate": 0.0 if not rows else row_scores / len(rows),
        "scoreable_group_count": len(scoreable_groups),
        "scoreable_group_rate": (
            0.0 if not group_summaries else len(scoreable_groups) / len(group_summaries)
        ),
        "action_sensitive_group_count": len(action_sensitive_groups),
        "action_sensitive_group_rate": (
            0.0
            if not group_summaries
            else len(action_sensitive_groups) / len(group_summaries)
        ),
        "score_return_spread_group_count": len(score_return_spread_groups),
        "score_return_spread_group_rate": (
            0.0
            if not group_summaries
            else len(score_return_spread_groups) / len(group_summaries)
        ),
        "contact_angle_sensitive_group_count": sum(
            group["contact_angle_differs"] for group in group_summaries
        ),
        "terminal_outcome_sensitive_group_count": sum(
            group["terminal_outcome_differs"] for group in group_summaries
        ),
        "survival_sensitive_group_count": sum(
            group["survival_steps_differs"] for group in group_summaries
        ),
        "terminal_signature_histogram": dict(sorted(terminal_histogram.items())),
        "best_action_histogram": dict(sorted(best_action_histogram.items())),
        "mean_survival_steps_by_candidate_action": _mean_by_action(rows, "survival_steps"),
        "mean_score_return_by_candidate_action": _mean_by_action(rows, "score_return"),
        "by_opponent_policy": _summarize_by_opponent(group_summaries),
        "examples": action_sensitive_groups[:8],
        "group_summaries": group_summaries,
    }


def _summarize_group(
    reset_state_id: str,
    opponent_policy: str,
    rows: list[dict[str, Any]],
) -> dict[str, object]:
    if len(rows) != len(ACTION_LABELS):
        raise ValueError(
            f"expected {len(ACTION_LABELS)} rows for {reset_state_id}/{opponent_policy}, "
            f"got {len(rows)}"
        )
    rows = sorted(rows, key=lambda row: int(row["candidate_ego_action"]))
    score_returns = {float(row["score_return"]) for row in rows}
    terminal_outcomes = {_terminal_signature(row) for row in rows}
    contact_angles = {
        (row["impact_offset"], row["outgoing_ball_vy"], row["first_contact_agent"])
        for row in rows
    }
    survival_steps = {int(row["survival_steps"]) for row in rows}
    best_score = max(float(row["score_return"]) for row in rows)
    best_survival = max(
        int(row["survival_steps"])
        for row in rows
        if float(row["score_return"]) == best_score
    )
    best_actions = [
        int(row["candidate_ego_action"])
        for row in rows
        if float(row["score_return"]) == best_score
        and int(row["survival_steps"]) == best_survival
    ]
    scoreable = any(float(row["ego_reward"]) > 0.0 for row in rows)
    score_return_differs = len(score_returns) > 1
    terminal_outcome_differs = len(terminal_outcomes) > 1
    contact_angle_differs = len(contact_angles) > 1
    survival_steps_differs = len(survival_steps) > 1
    return {
        "reset_state_id": reset_state_id,
        "reset_seed": rows[0]["reset_seed"],
        "pressure_agent": rows[0]["pressure_agent"],
        "opponent_policy": opponent_policy,
        "reset": rows[0]["reset"],
        "scoreable": scoreable,
        "score_probability_across_actions": (
            sum(float(row["ego_reward"]) > 0.0 for row in rows) / len(rows)
        ),
        "score_return_differs": score_return_differs,
        "terminal_outcome_differs": terminal_outcome_differs,
        "contact_angle_differs": contact_angle_differs,
        "survival_steps_differs": survival_steps_differs,
        "action_sensitive_any": (
            score_return_differs
            or terminal_outcome_differs
            or contact_angle_differs
            or survival_steps_differs
        ),
        "best_ego_actions_by_score_then_survival": best_actions,
        "candidate_results": [
            {
                "action": int(row["candidate_ego_action"]),
                "action_label": row["candidate_ego_action_label"],
                "score_return": row["score_return"],
                "winner": row["winner"],
                "terminated": row["terminated"],
                "truncated": row["truncated"],
                "survival_steps": row["survival_steps"],
                "first_contact_agent": row["first_contact_agent"],
                "first_contact_step": row["first_contact_step"],
                "impact_offset": row["impact_offset"],
                "outgoing_ball_vy": row["outgoing_ball_vy"],
            }
            for row in rows
        ],
    }


def _summarize_by_opponent(group_summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    by_opponent: dict[str, list[dict[str, object]]] = {}
    for group in group_summaries:
        by_opponent.setdefault(str(group["opponent_policy"]), []).append(group)
    summaries = []
    for opponent_policy, groups in sorted(by_opponent.items()):
        summaries.append(
            {
                "opponent_policy": opponent_policy,
                "groups": len(groups),
                "scoreable_group_count": sum(bool(group["scoreable"]) for group in groups),
                "scoreable_group_rate": (
                    sum(bool(group["scoreable"]) for group in groups) / len(groups)
                ),
                "action_sensitive_group_count": sum(
                    bool(group["action_sensitive_any"]) for group in groups
                ),
                "action_sensitive_group_rate": (
                    sum(bool(group["action_sensitive_any"]) for group in groups)
                    / len(groups)
                ),
                "score_return_spread_group_count": sum(
                    bool(group["score_return_differs"]) for group in groups
                ),
                "contact_angle_sensitive_group_count": sum(
                    bool(group["contact_angle_differs"]) for group in groups
                ),
                "survival_sensitive_group_count": sum(
                    bool(group["survival_steps_differs"]) for group in groups
                ),
            }
        )
    return summaries


def _mean_by_action(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    means = {}
    for action in range(len(ACTION_LABELS)):
        values = [
            float(row[key])
            for row in rows
            if int(row["candidate_ego_action"]) == action
        ]
        means[str(action)] = 0.0 if not values else float(np.mean(values))
    return means


def _terminal_signature(row: dict[str, Any]) -> str:
    return (
        f"reward={float(row['ego_reward']):+.1f}|winner={row['winner']}|"
        f"terminated={bool(row['terminated'])}|truncated={bool(row['truncated'])}"
    )


def _make_policy(policy_name: str, *, seed: int) -> BaselinePolicy:
    if policy_name == "track_ball":
        return TrackBallPolicy()
    if policy_name == "lagged_track_ball_1":
        return LaggedTrackBallPolicy(delay=1)
    if policy_name == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    if policy_name == "stay":
        return StayPolicy()
    raise ValueError(f"unknown policy {policy_name!r}")


def _snapshot(env: PongEnv) -> EnvSnapshot:
    return EnvSnapshot(
        player_0_y=int(env._paddle_y["player_0"]),
        player_1_y=int(env._paddle_y["player_1"]),
        ball_x=int(env._ball_x),
        ball_y=int(env._ball_y),
        ball_vx=int(env._ball_vx),
        ball_vy=int(env._ball_vy),
        step=int(env._step),
        reset_info=dict(env._reset_info),
    )


def _install_snapshot(env: PongEnv, snapshot: EnvSnapshot) -> None:
    env._paddle_y = {
        "player_0": snapshot.player_0_y,
        "player_1": snapshot.player_1_y,
    }
    env._ball_x = snapshot.ball_x
    env._ball_y = snapshot.ball_y
    env._ball_vx = snapshot.ball_vx
    env._ball_vy = snapshot.ball_vy
    env._step = snapshot.step
    env._done = False
    env._last_hit = None
    env._last_hit_impact = None
    env._score_agent = None
    env._reset_info = dict(snapshot.reset_info)


def _agent_seed_offset(agent: str) -> int:
    if agent == "player_0":
        return 0
    if agent == "player_1":
        return 100_000
    raise ValueError(f"unknown agent {agent!r}")


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent {agent!r}")


def _reset_state_id(pressure_agent: str, reset_seed: int) -> str:
    return f"{pressure_agent}-seed-{reset_seed}"


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
    with artifacts["rows_jsonl"].open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-per-pressure-agent", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--paddle-height", type=int, default=3)
    parser.add_argument("--reset-contact-distance-min", type=int, default=2)
    parser.add_argument("--reset-contact-distance-max", type=int, default=3)
    parser.add_argument(
        "--opponent-policy",
        action="append",
        choices=("track_ball", "lagged_track_ball_1", "stay", "random_uniform"),
        dest="opponent_policies",
        default=None,
    )
    parser.add_argument(
        "--ego-action-mode",
        choices=("first_step", "until_contact"),
        default="until_contact",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = probe_contact_pressure_scoreability(
        states_per_pressure_agent=args.states_per_pressure_agent,
        seed=args.seed,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
        paddle_height=args.paddle_height,
        reset_contact_distance_min=args.reset_contact_distance_min,
        reset_contact_distance_max=args.reset_contact_distance_max,
        opponent_policies=tuple(args.opponent_policies or DEFAULT_OPPONENT_POLICIES),
        ego_action_mode=args.ego_action_mode,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
