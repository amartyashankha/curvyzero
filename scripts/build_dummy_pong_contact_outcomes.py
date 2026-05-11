"""Build a tiny contact-outcome dataset for dummy Pong.

This is a local probe, not a training loop. It creates controlled near-contact
states, evaluates top/center/bottom ego paddle contacts, then rolls out a short
horizon against the scripted track_ball baseline.
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
from curvyzero.training.dummy_pong import RASTER_OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import REWARD_SCHEMA_ID
from curvyzero.training.dummy_pong import RULESET_ID
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv
from curvyzero.training.dummy_pong_eval import TrackBallPolicy

CONTACT_OUTCOME_SCHEMA_ID = "dummy_pong_contact_outcome_rows_v0"
CONTACT_OFFSETS = (-1, 0, 1)
CONTACT_LABEL_BY_OFFSET = {-1: "top", 0: "center", 1: "bottom"}


@dataclass(frozen=True, slots=True)
class ControlledContactState:
    state_id: str
    state_index: int
    state_seed: int
    ego_agent: str
    opponent_agent: str
    incoming_hit_y: int
    incoming_ball_vy: int
    ball_x: int
    ball_y: int
    ball_vx: int
    paddle_y_by_agent: dict[str, int]


def build_dummy_pong_contact_outcomes(
    *,
    states: int,
    seed: int,
    horizon: int,
    width: int = 15,
    height: int = 9,
    paddle_height: int = 3,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if states < 1:
        raise ValueError("states must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    config = PongConfig(
        width=width,
        height=height,
        paddle_height=paddle_height,
        max_steps=horizon,
    )
    _validate_config(config)
    rng = np.random.default_rng(seed)
    state_specs = [
        _make_controlled_state(config=config, rng=rng, seed=seed, state_index=index)
        for index in range(states)
    ]
    rows = [
        _rollout_candidate(config=config, state=state, candidate_impact_offset=offset)
        for state in state_specs
        for offset in CONTACT_OFFSETS
    ]
    track_ball_baseline_rows = [
        _rollout_track_ball_baseline(config=config, state=state)
        for state in state_specs
    ]
    summary = _summarize_rows(
        rows=rows,
        track_ball_baseline_rows=track_ball_baseline_rows,
        state_specs=state_specs,
        config=config,
        states=states,
        seed=seed,
        horizon=horizon,
    )

    if output_dir is not None:
        artifacts = {
            "summary_json": output_dir / "summary.json",
            "contact_rows_jsonl": output_dir / "contact_rows.jsonl",
        }
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, rows=rows)
    return summary


def _validate_config(config: PongConfig) -> None:
    if config.paddle_height != 3:
        raise ValueError(
            "this first contact probe assumes the default three-cell paddle"
        )
    feasible_hit_ys = _feasible_hit_ys(config)
    if not feasible_hit_ys:
        raise ValueError("config has no hit rows that support top/center/bottom choices")


def _make_controlled_state(
    *,
    config: PongConfig,
    rng: np.random.Generator,
    seed: int,
    state_index: int,
) -> ControlledContactState:
    ego_agent = str(rng.choice(AGENTS))
    opponent_agent = _opponent(ego_agent)
    ball_vx = -1 if ego_agent == "player_0" else 1
    incoming_hit_y = int(rng.choice(_feasible_hit_ys(config)))
    incoming_ball_vy = int(rng.choice([-1, 0, 1]))
    ball_y = incoming_hit_y - incoming_ball_vy
    paddle_x = {"player_0": 1, "player_1": config.width - 2}
    ball_x = paddle_x[ego_agent] - ball_vx
    center_y = (config.height - config.paddle_height) // 2
    opponent_paddle_y = _clamp_paddle_y(incoming_hit_y - config.paddle_height // 2, config)
    return ControlledContactState(
        state_id=f"state-{state_index:04d}",
        state_index=state_index,
        state_seed=seed + state_index,
        ego_agent=ego_agent,
        opponent_agent=opponent_agent,
        incoming_hit_y=incoming_hit_y,
        incoming_ball_vy=incoming_ball_vy,
        ball_x=ball_x,
        ball_y=ball_y,
        ball_vx=ball_vx,
        paddle_y_by_agent={
            ego_agent: center_y,
            opponent_agent: opponent_paddle_y,
        },
    )


def _feasible_hit_ys(config: PongConfig) -> list[int]:
    feasible = []
    for hit_y in range(config.height):
        if all(
            0 <= _paddle_y_for_impact(config=config, hit_y=hit_y, impact_offset=offset)
            <= config.height - config.paddle_height
            for offset in CONTACT_OFFSETS
        ):
            feasible.append(hit_y)
    return feasible


def _rollout_candidate(
    *,
    config: PongConfig,
    state: ControlledContactState,
    candidate_impact_offset: int,
) -> dict[str, Any]:
    env = PongEnv(config)
    env.reset(seed=state.state_seed)
    candidate_paddle_y = _paddle_y_for_impact(
        config=config,
        hit_y=state.incoming_hit_y,
        impact_offset=candidate_impact_offset,
    )
    target_center_y = state.incoming_hit_y - candidate_impact_offset
    base_ego_center_y = (
        state.paddle_y_by_agent[state.ego_agent] + config.paddle_height // 2
    )
    target_center_within_bounds = (
        0 <= candidate_paddle_y <= config.height - config.paddle_height
    )
    reachable_target_center = (
        target_center_within_bounds
        and abs(target_center_y - base_ego_center_y) <= 1
    )
    _install_state(env, state=state, ego_paddle_y=candidate_paddle_y)
    pre_contact_raster = _raster_strings(env.raster_observation())
    pre_contact_state = _env_debug_state(env)

    track_ball = {agent: TrackBallPolicy() for agent in AGENTS}
    for agent, policy in track_ball.items():
        policy.reset(state.state_seed, agent)

    score_delta_return = 0.0
    post_contact_score_delta_return = 0.0
    first_contact: dict[str, object] | None = None
    contact_seen = False
    final_step = None
    actions_by_step: list[dict[str, int]] = []
    for rollout_step in range(config.max_steps):
        observations = env.observations()
        raster_grid = env.raster_observation()
        joint_action = {}
        for agent in AGENTS:
            if rollout_step == 0 and agent == state.ego_agent:
                action = 1
            else:
                action = track_ball[agent].action(observations[agent], raster_grid, agent)
            joint_action[agent] = int(action)
        actions_by_step.append(dict(joint_action))

        final_step = env.step(joint_action)
        ego_reward = float(final_step.rewards[state.ego_agent])
        score_delta_return += ego_reward
        impact = final_step.infos["last_hit_impact"]
        if first_contact is None and impact is not None:
            first_contact = dict(impact)
            contact_seen = True
        if contact_seen:
            post_contact_score_delta_return += ego_reward
        if final_step.terminated or final_step.truncated:
            break

    if final_step is None:
        raise RuntimeError("rollout did not run")

    contact_happened = (
        first_contact is not None
        and first_contact.get("agent") == state.ego_agent
    )
    actual_impact_offset = (
        None if first_contact is None else int(first_contact["impact_offset"])
    )
    actual_outgoing_ball_vy = (
        None if first_contact is None else int(first_contact["outgoing_ball_vy"])
    )
    winner = final_step.infos["winner"]
    return {
        "schema_id": CONTACT_OUTCOME_SCHEMA_ID,
        "ruleset_id": RULESET_ID,
        "reward_schema_id": REWARD_SCHEMA_ID,
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "paddle_bounce_schema_id": PADDLE_BOUNCE_SCHEMA_ID,
        "state_id": state.state_id,
        "state_index": state.state_index,
        "state_seed": state.state_seed,
        "ego_agent": state.ego_agent,
        "opponent_agent": state.opponent_agent,
        "opponent_policy": "track_ball",
        "candidate_contact_label": CONTACT_LABEL_BY_OFFSET[candidate_impact_offset],
        "candidate_impact_offset": candidate_impact_offset,
        "desired_impact_offset": candidate_impact_offset,
        "predicted_hit_y": state.incoming_hit_y,
        "target_center_y": target_center_y,
        "base_ego_paddle_center_y": base_ego_center_y,
        "target_center_within_bounds": target_center_within_bounds,
        "reachable_target_center": reachable_target_center,
        "uses_controlled_paddle_placement": True,
        "candidate_ego_paddle_y": candidate_paddle_y,
        "contact_happened": contact_happened,
        "actual_impact_offset": actual_impact_offset,
        "actual_outgoing_ball_vy": actual_outgoing_ball_vy,
        "actual_hit_y": None if first_contact is None else int(first_contact["hit_y"]),
        "actual_paddle_center_y": (
            None if first_contact is None else int(first_contact["paddle_center_y"])
        ),
        "score_delta_return": score_delta_return,
        "post_contact_score_delta_return": post_contact_score_delta_return,
        "terminated": bool(final_step.terminated),
        "truncated": bool(final_step.truncated),
        "winner": winner,
        "ego_reward": float(final_step.rewards[state.ego_agent]),
        "steps": int(next(iter(final_step.observations.values())).step),
        "horizon": config.max_steps,
        "pre_contact_state": pre_contact_state,
        "pre_contact_raster": pre_contact_raster,
        "final_state": _env_debug_state(env),
        "first_actions_by_agent": actions_by_step[0],
        "action_trace": actions_by_step,
    }


def _rollout_track_ball_baseline(
    *,
    config: PongConfig,
    state: ControlledContactState,
) -> dict[str, Any]:
    env = PongEnv(config)
    env.reset(seed=state.state_seed)
    _install_state(
        env,
        state=state,
        ego_paddle_y=state.paddle_y_by_agent[state.ego_agent],
    )
    pre_contact_state = _env_debug_state(env)
    track_ball = {agent: TrackBallPolicy() for agent in AGENTS}
    for agent, policy in track_ball.items():
        policy.reset(state.state_seed, agent)

    score_delta_return = 0.0
    post_contact_score_delta_return = 0.0
    first_contact: dict[str, object] | None = None
    contact_seen = False
    final_step = None
    first_actions_by_agent: dict[str, int] | None = None
    for _rollout_step in range(config.max_steps):
        observations = env.observations()
        raster_grid = env.raster_observation()
        joint_action = {
            agent: int(track_ball[agent].action(observations[agent], raster_grid, agent))
            for agent in AGENTS
        }
        if first_actions_by_agent is None:
            first_actions_by_agent = dict(joint_action)
        final_step = env.step(joint_action)
        ego_reward = float(final_step.rewards[state.ego_agent])
        score_delta_return += ego_reward
        impact = final_step.infos["last_hit_impact"]
        if first_contact is None and impact is not None:
            first_contact = dict(impact)
            contact_seen = True
        if contact_seen:
            post_contact_score_delta_return += ego_reward
        if final_step.terminated or final_step.truncated:
            break

    if final_step is None or first_actions_by_agent is None:
        raise RuntimeError("baseline rollout did not run")

    return {
        "state_id": state.state_id,
        "state_index": state.state_index,
        "state_seed": state.state_seed,
        "ego_agent": state.ego_agent,
        "opponent_agent": state.opponent_agent,
        "policy": "track_ball",
        "predicted_hit_y": state.incoming_hit_y,
        "contact_happened": (
            first_contact is not None
            and first_contact.get("agent") == state.ego_agent
        ),
        "actual_impact_offset": (
            None if first_contact is None else int(first_contact["impact_offset"])
        ),
        "actual_outgoing_ball_vy": (
            None if first_contact is None else int(first_contact["outgoing_ball_vy"])
        ),
        "score_delta_return": score_delta_return,
        "post_contact_score_delta_return": post_contact_score_delta_return,
        "terminated": bool(final_step.terminated),
        "truncated": bool(final_step.truncated),
        "winner": final_step.infos["winner"],
        "steps": int(next(iter(final_step.observations.values())).step),
        "pre_contact_state": pre_contact_state,
        "final_state": _env_debug_state(env),
        "first_actions_by_agent": first_actions_by_agent,
    }


def _install_state(
    env: PongEnv,
    *,
    state: ControlledContactState,
    ego_paddle_y: int,
) -> None:
    # This probe intentionally uses private toy-env fields to create compact,
    # controlled near-contact states before the first real action.
    env._paddle_y = dict(state.paddle_y_by_agent)
    env._paddle_y[state.ego_agent] = ego_paddle_y
    env._ball_x = state.ball_x
    env._ball_y = state.ball_y
    env._ball_vx = state.ball_vx
    env._ball_vy = state.incoming_ball_vy
    env._step = 0
    env._done = False
    env._last_hit = None
    env._last_hit_impact = None
    env._score_agent = None


def _paddle_y_for_impact(
    *,
    config: PongConfig,
    hit_y: int,
    impact_offset: int,
) -> int:
    paddle_center_y = hit_y - impact_offset
    return paddle_center_y - config.paddle_height // 2


def _clamp_paddle_y(paddle_y: int, config: PongConfig) -> int:
    return min(max(paddle_y, 0), config.height - config.paddle_height)


def _env_debug_state(env: PongEnv) -> dict[str, object]:
    return {
        "step": int(env._step),
        "ball": {
            "x": int(env._ball_x),
            "y": int(env._ball_y),
            "vx": int(env._ball_vx),
            "vy": int(env._ball_vy),
        },
        "paddles": {
            agent: {
                "x": int(env._paddle_x[agent]),
                "y": int(env._paddle_y[agent]),
            }
            for agent in AGENTS
        },
    }


def _raster_strings(raster_grid: np.ndarray) -> list[str]:
    return ["".join(str(int(cell)) for cell in row) for row in raster_grid]


def _summarize_rows(
    *,
    rows: list[dict[str, Any]],
    track_ball_baseline_rows: list[dict[str, Any]],
    state_specs: list[ControlledContactState],
    config: PongConfig,
    states: int,
    seed: int,
    horizon: int,
) -> dict[str, object]:
    by_state: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_state.setdefault(str(row["state_id"]), []).append(row)

    outgoing_diff_state_count = 0
    return_diff_state_count = 0
    terminal_diff_state_count = 0
    state_summaries = []
    for state_id, state_rows in sorted(by_state.items()):
        outgoing_values = {row["actual_outgoing_ball_vy"] for row in state_rows}
        returns = {row["score_delta_return"] for row in state_rows}
        terminal_outcomes = {
            (row["terminated"], row["truncated"], row["winner"])
            for row in state_rows
        }
        outgoing_diff_state_count += len(outgoing_values) > 1
        return_diff_state_count += len(returns) > 1
        terminal_diff_state_count += len(terminal_outcomes) > 1
        best_return = max(float(row["score_delta_return"]) for row in state_rows)
        best_offsets = [
            int(row["candidate_impact_offset"])
            for row in state_rows
            if float(row["score_delta_return"]) == best_return
        ]
        state_summaries.append(
            {
                "state_id": state_id,
                "ego_agent": state_rows[0]["ego_agent"],
                "incoming_hit_y": state_rows[0]["pre_contact_state"]["ball"]["y"]
                + state_rows[0]["pre_contact_state"]["ball"]["vy"],
                "score_delta_returns_by_offset": {
                    str(row["candidate_impact_offset"]): row["score_delta_return"]
                    for row in sorted(state_rows, key=lambda item: item["candidate_impact_offset"])
                },
                "post_contact_score_delta_returns_by_offset": {
                    str(row["candidate_impact_offset"]): row[
                        "post_contact_score_delta_return"
                    ]
                    for row in sorted(state_rows, key=lambda item: item["candidate_impact_offset"])
                },
                "outgoing_ball_vy_by_offset": {
                    str(row["candidate_impact_offset"]): row["actual_outgoing_ball_vy"]
                    for row in sorted(state_rows, key=lambda item: item["candidate_impact_offset"])
                },
                "track_ball_baseline": _baseline_summary_for_state(
                    track_ball_baseline_rows,
                    state_id,
                ),
                "best_return": best_return,
                "best_candidate_impact_offsets": best_offsets,
            }
        )

    contacts = [row for row in rows if row["contact_happened"]]
    return_histogram = Counter(str(row["score_delta_return"]) for row in rows)
    all_candidate_returns_zero = all(
        float(row["score_delta_return"]) == 0.0
        for row in rows
    )
    best_offset_histogram: Counter[str] = Counter()
    for state_summary in state_summaries:
        best_offsets = state_summary["best_candidate_impact_offsets"]
        if len(best_offsets) == len(CONTACT_OFFSETS):
            best_offset_histogram["all_tied"] += 1
        else:
            for offset in best_offsets:
                best_offset_histogram[str(offset)] += 1

    return {
        "kind": "curvyzero_dummy_pong_contact_outcomes",
        "schema_id": CONTACT_OUTCOME_SCHEMA_ID,
        "note": (
            "Controlled near-contact snapshots compare top/center/bottom ego "
            "contacts, then roll out a short horizon against track_ball."
        ),
        "private_state_access": (
            "This first toy probe writes PongEnv private fields to create cloned "
            "near-contact states and candidate paddle placements."
        ),
        "seed": seed,
        "states": states,
        "horizon": horizon,
        "rows": len(rows),
        "config": asdict(config),
        "action_labels": list(ACTION_LABELS),
        "candidate_impact_offsets": list(CONTACT_OFFSETS),
        "candidate_contact_labels": {
            str(offset): label
            for offset, label in CONTACT_LABEL_BY_OFFSET.items()
        },
        "state_specs": [asdict(state) for state in state_specs],
        "contact_count": len(contacts),
        "missing_contact_count": len(rows) - len(contacts),
        "return_histogram": dict(sorted(return_histogram.items())),
        "mean_score_delta_return_by_offset": _mean_by_offset(rows, "score_delta_return"),
        "mean_post_contact_score_delta_return_by_offset": _mean_by_offset(
            rows,
            "post_contact_score_delta_return",
        ),
        "terminated_count_by_offset": _count_true_by_offset(rows, "terminated"),
        "truncated_count_by_offset": _count_true_by_offset(rows, "truncated"),
        "outgoing_ball_vy_histogram_by_offset": _histogram_by_offset(
            rows,
            "actual_outgoing_ball_vy",
        ),
        "track_ball_baseline": _summarize_track_ball_baseline(track_ball_baseline_rows),
        "outcome_differences": {
            "outgoing_ball_vy_differs_state_count": outgoing_diff_state_count,
            "score_delta_return_differs_state_count": return_diff_state_count,
            "terminal_outcome_differs_state_count": terminal_diff_state_count,
            "choices_differ_on_any_sampled_state": (
                outgoing_diff_state_count > 0
                or return_diff_state_count > 0
                or terminal_diff_state_count > 0
            ),
            "score_delta_choices_differ_on_any_sampled_state": (
                return_diff_state_count > 0
            ),
        },
        "best_candidate_impact_offset_histogram": dict(
            sorted(best_offset_histogram.items())
        ),
        "geometry_note": (
            "All candidate score-delta returns were zero against track_ball; "
            "the default geometry may be too forgiving. Next knobs to try are "
            "smaller width, smaller paddle, or faster ball."
            if all_candidate_returns_zero
            else None
        ),
        "state_summaries": state_summaries,
    }


def _baseline_summary_for_state(
    baseline_rows: list[dict[str, Any]],
    state_id: str,
) -> dict[str, object]:
    matches = [row for row in baseline_rows if row["state_id"] == state_id]
    if len(matches) != 1:
        raise ValueError(f"expected one baseline row for {state_id}, got {len(matches)}")
    row = matches[0]
    return {
        "score_delta_return": row["score_delta_return"],
        "post_contact_score_delta_return": row["post_contact_score_delta_return"],
        "contact_happened": row["contact_happened"],
        "actual_impact_offset": row["actual_impact_offset"],
        "actual_outgoing_ball_vy": row["actual_outgoing_ball_vy"],
        "terminated": row["terminated"],
        "truncated": row["truncated"],
        "winner": row["winner"],
    }


def _summarize_track_ball_baseline(
    baseline_rows: list[dict[str, Any]],
) -> dict[str, object]:
    return {
        "rows": len(baseline_rows),
        "contact_count": sum(bool(row["contact_happened"]) for row in baseline_rows),
        "return_histogram": dict(
            sorted(Counter(str(row["score_delta_return"]) for row in baseline_rows).items())
        ),
        "mean_score_delta_return": (
            0.0
            if not baseline_rows
            else float(np.mean([row["score_delta_return"] for row in baseline_rows]))
        ),
        "actual_impact_offset_histogram": dict(
            sorted(
                Counter(str(row["actual_impact_offset"]) for row in baseline_rows).items()
            )
        ),
        "rows_by_state": [
            {
                "state_id": row["state_id"],
                "ego_agent": row["ego_agent"],
                "score_delta_return": row["score_delta_return"],
                "post_contact_score_delta_return": row[
                    "post_contact_score_delta_return"
                ],
                "contact_happened": row["contact_happened"],
                "actual_impact_offset": row["actual_impact_offset"],
                "actual_outgoing_ball_vy": row["actual_outgoing_ball_vy"],
                "terminated": row["terminated"],
                "truncated": row["truncated"],
                "winner": row["winner"],
            }
            for row in baseline_rows
        ],
    }


def _mean_by_offset(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    means = {}
    for offset in CONTACT_OFFSETS:
        values = [
            float(row[key])
            for row in rows
            if row["candidate_impact_offset"] == offset
        ]
        means[str(offset)] = 0.0 if not values else float(np.mean(values))
    return means


def _count_true_by_offset(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return {
        str(offset): sum(
            bool(row[key])
            for row in rows
            if row["candidate_impact_offset"] == offset
        )
        for offset in CONTACT_OFFSETS
    }


def _histogram_by_offset(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    histograms: dict[str, dict[str, int]] = {}
    for offset in CONTACT_OFFSETS:
        counter = Counter(
            str(row[key])
            for row in rows
            if row["candidate_impact_offset"] == offset
        )
        histograms[str(offset)] = dict(sorted(counter.items()))
    return histograms


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    rows: list[dict[str, Any]],
) -> None:
    summary_path = artifacts["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with artifacts["contact_rows_jsonl"].open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent {agent!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--paddle-height", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = build_dummy_pong_contact_outcomes(
        states=args.states,
        seed=args.seed,
        horizon=args.horizon,
        width=args.width,
        height=args.height,
        paddle_height=args.paddle_height,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
