"""Build supervised raster replay from exact lag-1 Pong winning traces."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from curvyzero.training.dummy_pong import ACTION_LABELS
from curvyzero.training.dummy_pong import ACTION_SCHEMA_ID
from curvyzero.training.dummy_pong import AGENTS
from curvyzero.training.dummy_pong import RASTER_LEGEND
from curvyzero.training.dummy_pong import RASTER_OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import REWARD_SCHEMA_ID
from curvyzero.training.dummy_pong import RULESET_ID
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv
from curvyzero.training.dummy_pong_lookahead_replay import LOOKAHEAD_REPLAY_ROW_SCHEMA_ID

from probe_dummy_pong_target_ladder import LadderState
from probe_dummy_pong_target_ladder import TargetSpec
from probe_dummy_pong_target_ladder import _find_winning_trace
from probe_dummy_pong_target_ladder import _initial_opponent_memory
from probe_dummy_pong_target_ladder import _opponent_action
from probe_dummy_pong_target_ladder import _update_opponent_memory
from probe_dummy_pong_track_ball_beatable import SearchState
from probe_dummy_pong_track_ball_beatable import _normal_reset_states
from probe_dummy_pong_track_ball_beatable import _opponent
from probe_dummy_pong_track_ball_beatable import _state_id

TRACE_REPLAY_SUMMARY_SCHEMA_ID = "dummy_pong_lag1_trace_replay_summary_v0"
TRACE_TARGET_POLICY_ID = "exact_dp_lagged_track_ball_1_winning_trace_v0"
BALANCE_ACTION_CHOICES = ("none", "oversample")


def build_dummy_pong_lag1_trace_replay(
    *,
    output_dir: Path | None = None,
    max_steps: int = 120,
    repeats: int = 1,
    frame_stack: int = 1,
    include_vertical_mirror: bool = False,
    balance_actions: str = "none",
    balance_seed: int = 0,
) -> dict[str, object]:
    """Emit raster action-label rows from exact traces that beat lagged track-ball."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if frame_stack < 1:
        raise ValueError("frame_stack must be at least 1")
    if balance_actions not in BALANCE_ACTION_CHOICES:
        raise ValueError(f"balance_actions must be one of {BALANCE_ACTION_CHOICES!r}")

    config = PongConfig(max_steps=max_steps)
    spec = TargetSpec(
        target_id="lag1_track_ball_normal",
        description="Default geometry; opponent tracks the previous ball row.",
        config=config,
        opponent_policy="lagged_track_ball_1",
    )
    rows: list[dict[str, Any]] = []
    trace_summaries = []
    reset_states = _normal_reset_states(config)
    run_id_parts = [
        "dummy-pong-lag1-trace-replay",
        f"max-{max_steps}",
        f"repeats-{repeats}",
    ]
    if frame_stack != 1:
        run_id_parts.append(f"stack-{frame_stack}")
    if include_vertical_mirror:
        run_id_parts.append("mirror-y")
    if balance_actions != "none":
        run_id_parts.append(f"balance-{balance_actions}")
    run_id = "-".join(run_id_parts)

    for state_index, reset_state in enumerate(reset_states):
        reset_state_id = _state_id(state_index, reset_state)
        for ego_agent in AGENTS:
            outcome = _find_winning_trace(
                spec=spec,
                initial_state=reset_state,
                ego_agent=ego_agent,
            )
            if outcome.trace is None:
                raise RuntimeError(
                    f"lag-1 target produced no trace for {reset_state_id} {ego_agent}"
                )
            for repeat_index in range(repeats):
                rows.extend(
                    _trace_rows(
                        config=config,
                        spec=spec,
                        run_id=run_id,
                        reset_state_id=reset_state_id,
                        reset_state=reset_state,
                        ego_agent=ego_agent,
                        actions=outcome.trace.actions,
                        repeat_index=repeat_index,
                        frame_stack=frame_stack,
                    )
                )
            trace_summaries.append(
                {
                    "reset_state_id": reset_state_id,
                    "reset_state": asdict(reset_state),
                    "ego_agent": ego_agent,
                    "opponent_agent": _opponent(ego_agent),
                    "winning_steps": outcome.trace.steps,
                    "winning_actions": outcome.trace.actions,
                    "winning_action_labels": [
                        ACTION_LABELS[action] for action in outcome.trace.actions
                    ],
                    "memoized_states": outcome.memoized_states,
                }
            )

    exact_rows = list(rows)
    transform_summaries: list[dict[str, object]] = []
    if include_vertical_mirror:
        rows, mirror_summary = _include_vertical_mirror(rows, config=config)
        transform_summaries.append(mirror_summary)
    rows, balance_summary = _balance_rows_by_action(
        rows,
        strategy=balance_actions,
        seed=balance_seed,
    )
    transform_summaries.append(balance_summary)

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_lag1_trace_replay",
        "summary_schema_id": TRACE_REPLAY_SUMMARY_SCHEMA_ID,
        "row_schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
        "target_policy_id": TRACE_TARGET_POLICY_ID,
        "run_id": run_id,
        "note": (
            "Supervised visual-policy replay from exact DP winning traces against "
            "lagged_track_ball_1. Rows contain raster_grid and target_action labels; "
            "this is a smallest score-pressure visual lane, not CEM or self-play."
        ),
        "config": asdict(config),
        "max_steps": max_steps,
        "repeats": repeats,
        "frame_stack": frame_stack,
        "include_vertical_mirror": include_vertical_mirror,
        "balance_actions": balance_actions,
        "balance_seed": balance_seed,
        "total_rows": len(rows),
        "exact_trace_rows": len(exact_rows),
        "trace_count": len(trace_summaries),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "trace_replay_summary_schema_id": TRACE_REPLAY_SUMMARY_SCHEMA_ID,
            "replay_row_schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
        },
        "raster": {
            "shape": [config.height, config.width],
            "encoding": "row-major digit strings over the raster legend",
            "frame_stack": {
                "length": frame_stack,
                "field": "raster_frame_stack",
                "order": "oldest-to-newest; final frame equals raster_grid",
                "initial_padding": "duplicate earliest available frame",
            },
            "legend": RASTER_LEGEND,
        },
        "action_labels": list(ACTION_LABELS),
        "exact_trace_data": _summarize_rows(exact_rows),
        "transforms": transform_summaries,
        "data": _summarize_rows(rows),
        "traces": {
            "count": len(trace_summaries),
            "winning_steps_min": min(int(trace["winning_steps"]) for trace in trace_summaries),
            "winning_steps_max": max(int(trace["winning_steps"]) for trace in trace_summaries),
            "examples": trace_summaries[:4],
        },
        "plain_language": {
            "proves": (
                "The target-ladder DP can be converted into raster supervised rows "
                "that ask a policy to imitate score-producing actions."
            ),
            "does_not_prove": (
                "That the trained visual policy generalizes. That claim requires "
                "the checkpoint scoreboard against lagged_track_ball_1, random, "
                "and default track_ball."
            ),
        },
    }
    if output_dir is not None:
        artifacts = {
            "summary_json": output_dir / "summary.json",
            "replay_rows_jsonl": output_dir / "replay_rows.jsonl",
        }
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, rows=rows)
    return summary


def _include_vertical_mirror(
    rows: list[dict[str, Any]],
    *,
    config: PongConfig,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    mirrored_rows = [
        _vertical_mirror_row(row, row_index=index, config=config)
        for index, row in enumerate(rows)
    ]
    return (
        rows + mirrored_rows,
        {
            "name": "vertical_mirror",
            "description": (
                "Mirror raster rows across the horizontal centerline and swap "
                "up/down actions. This preserves the lagged-track-ball game "
                "symmetry while turning upward corrections into downward ones."
            ),
            "input_rows": len(rows),
            "added_rows": len(mirrored_rows),
            "output_rows": len(rows) + len(mirrored_rows),
            "action_histogram_before": _action_histogram_by_agent(rows),
            "action_histogram_after": _action_histogram_by_agent(rows + mirrored_rows),
        },
    )


def _vertical_mirror_row(
    row: dict[str, Any],
    *,
    row_index: int,
    config: PongConfig,
) -> dict[str, Any]:
    mirrored = deepcopy(row)
    mirrored["reset_state_id"] = f"{row['reset_state_id']}|mirror_y"
    mirrored["match_id"] = f"{row.get('match_id', 'exact_dp')}|mirror_y"
    mirrored["raster_grid"] = _mirror_grid_y(row["raster_grid"])
    mirrored["next_raster_grid"] = _mirror_grid_y(row["next_raster_grid"])
    if "raster_frame_stack" in mirrored:
        mirrored["raster_frame_stack"] = _mirror_grid_stack_y(row["raster_frame_stack"])
    if "next_raster_frame_stack" in mirrored:
        mirrored["next_raster_frame_stack"] = _mirror_grid_stack_y(row["next_raster_frame_stack"])
    mirrored["reset_state"] = _mirror_state_y(row["reset_state"], config=config)
    mirrored["observation"] = _mirror_observation_y(row["observation"], config=config)
    mirrored["next_observation"] = _mirror_observation_y(
        row["next_observation"],
        config=config,
    )
    mirrored["target_action_id"] = _mirror_action_id(row["target_action_id"])
    mirrored["target_action_label"] = ACTION_LABELS[mirrored["target_action_id"]]
    mirrored["behavior_action_id"] = mirrored["target_action_id"]
    mirrored["behavior_action_label"] = mirrored["target_action_label"]
    mirrored["joint_action_by_agent"] = {
        agent: {
            "action_id": _mirror_action_id(action["action_id"]),
            "label": ACTION_LABELS[_mirror_action_id(action["action_id"])],
        }
        for agent, action in row["joint_action_by_agent"].items()
    }
    mirrored["augmentation"] = {
        "type": "vertical_mirror",
        "source_replay_row_index": row_index,
        "source_reset_state_id": row["reset_state_id"],
    }
    return mirrored


def _mirror_grid_y(grid: list[str]) -> list[str]:
    return list(reversed(grid))


def _mirror_grid_stack_y(stack: list[list[str]]) -> list[list[str]]:
    return [_mirror_grid_y(grid) for grid in stack]


def _mirror_state_y(state: dict[str, Any], *, config: PongConfig) -> dict[str, Any]:
    mirrored = dict(state)
    max_paddle_y = config.height - config.paddle_height
    if "player_0_y" in mirrored:
        mirrored["player_0_y"] = max_paddle_y - int(mirrored["player_0_y"])
    if "player_1_y" in mirrored:
        mirrored["player_1_y"] = max_paddle_y - int(mirrored["player_1_y"])
    if "ball_y" in mirrored:
        mirrored["ball_y"] = config.height - 1 - int(mirrored["ball_y"])
    if "ball_vy" in mirrored:
        mirrored["ball_vy"] = -int(mirrored["ball_vy"])
    return mirrored


def _mirror_observation_y(
    observation: dict[str, Any],
    *,
    config: PongConfig,
) -> dict[str, Any]:
    mirrored = dict(observation)
    max_paddle_y = config.height - config.paddle_height
    if "ego_paddle_y" in mirrored:
        mirrored["ego_paddle_y"] = max_paddle_y - int(mirrored["ego_paddle_y"])
    if "opponent_paddle_y" in mirrored:
        mirrored["opponent_paddle_y"] = max_paddle_y - int(mirrored["opponent_paddle_y"])
    if "ball_y" in mirrored:
        mirrored["ball_y"] = config.height - 1 - int(mirrored["ball_y"])
    if "ball_dy_from_ego_center" in mirrored:
        mirrored["ball_dy_from_ego_center"] = -int(mirrored["ball_dy_from_ego_center"])
    if "ball_vy" in mirrored:
        mirrored["ball_vy"] = -int(mirrored["ball_vy"])
    return mirrored


def _mirror_action_id(action_id: object) -> int:
    action = int(action_id)
    if action == 0:
        return 2
    if action == 2:
        return 0
    return action


def _balance_rows_by_action(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    before = _action_histogram_by_agent(rows)
    if strategy == "none":
        return (
            rows,
            {
                "name": "action_balance",
                "strategy": "none",
                "seed": seed,
                "input_rows": len(rows),
                "added_rows": 0,
                "output_rows": len(rows),
                "action_histogram_before": before,
                "action_histogram_after": before,
            },
        )

    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {
        (agent, label): []
        for agent in AGENTS
        for label in ACTION_LABELS
    }
    for row in rows:
        groups[(str(row["ego_agent"]), str(row["target_action_label"]))].append(row)
    target_count = max((len(group) for group in groups.values()), default=0)

    balanced_rows = list(rows)
    added_rows = 0
    for agent in AGENTS:
        for label in ACTION_LABELS:
            group = groups[(agent, label)]
            if not group:
                continue
            needed = target_count - len(group)
            for duplicate_index in range(needed):
                duplicate = deepcopy(rng.choice(group))
                duplicate["replay_balance"] = {
                    "strategy": strategy,
                    "group": {"ego_agent": agent, "target_action_label": label},
                    "duplicate_index": duplicate_index,
                    "target_rows_per_group": target_count,
                }
                balanced_rows.append(duplicate)
                added_rows += 1

    after = _action_histogram_by_agent(balanced_rows)
    return (
        balanced_rows,
        {
            "name": "action_balance",
            "strategy": strategy,
            "seed": seed,
            "input_rows": len(rows),
            "added_rows": added_rows,
            "output_rows": len(balanced_rows),
            "target_rows_per_agent_action": target_count,
            "action_histogram_before": before,
            "action_histogram_after": after,
        },
    )


def _trace_rows(
    *,
    config: PongConfig,
    spec: TargetSpec,
    run_id: str,
    reset_state_id: str,
    reset_state: SearchState,
    ego_agent: str,
    actions: list[int],
    repeat_index: int,
    frame_stack: int,
) -> list[dict[str, Any]]:
    env = _env_from_state(config, reset_state)
    opponent_agent = _opponent(ego_agent)
    ladder_state = LadderState(
        core=reset_state,
        opponent_memory=_initial_opponent_memory(spec, reset_state),
    )
    rows = []
    raster_history: list[list[str]] = []
    for trace_step_index, ego_action in enumerate(actions):
        step_index = int(next(iter(env.observations().values())).step)
        raster_grid = _raster_grid(env.raster_observation())
        raster_history = (raster_history + [raster_grid])[-frame_stack:]
        raster_frame_stack = _pad_frame_stack(raster_history, frame_stack=frame_stack)
        observations = env.observations()
        opponent_action = _opponent_action(spec, ladder_state, opponent_agent)
        joint_action = {
            ego_agent: ego_action,
            opponent_agent: opponent_action,
        }
        step = env.step(joint_action)
        next_raster_grid = _raster_grid(env.raster_observation())
        next_raster_frame_stack = _pad_frame_stack(
            (raster_history + [next_raster_grid])[-frame_stack:],
            frame_stack=frame_stack,
        )
        rows.append(
            {
                "schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
                "run_id": run_id,
                "target_policy_id": TRACE_TARGET_POLICY_ID,
                "target_schema_id": TRACE_REPLAY_SUMMARY_SCHEMA_ID,
                "replay_semantics_id": "dummy_pong_exact_dp_trace_supervised_replay_v0",
                "match_id": "exact_dp_vs_lagged_track_ball_1",
                "reset_state_id": reset_state_id,
                "reset_state": asdict(reset_state),
                "repeat_index": repeat_index,
                "trace_step_index": trace_step_index,
                "step_index": step_index,
                "ego_agent": ego_agent,
                "opponent_agent": opponent_agent,
                "collector_policy_id": TRACE_TARGET_POLICY_ID,
                "behavior_policy_id": TRACE_TARGET_POLICY_ID,
                "opponent_policy_id": "lagged_track_ball_1",
                "target_action_id": ego_action,
                "target_action_label": ACTION_LABELS[ego_action],
                "behavior_action_id": ego_action,
                "behavior_action_label": ACTION_LABELS[ego_action],
                "joint_action_by_agent": {
                    agent: {
                        "action_id": int(action),
                        "label": ACTION_LABELS[int(action)],
                    }
                    for agent, action in joint_action.items()
                },
                "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
                "raster_shape": [config.height, config.width],
                "raster_grid": raster_grid,
                "raster_frame_stack": raster_frame_stack,
                "frame_stack": frame_stack,
                "observation": asdict(observations[ego_agent]),
                "reward_after_step": float(step.rewards[ego_agent]),
                "next_raster_grid": next_raster_grid,
                "next_raster_frame_stack": next_raster_frame_stack,
                "next_observation": asdict(step.observations[ego_agent]),
                "terminated_after_step": bool(step.terminated),
                "truncated_after_step": bool(step.truncated),
                "terminated": bool(step.terminated),
                "truncated": bool(step.truncated),
                "winner_after_step": step.infos["winner"] if step.terminated else None,
            }
        )
        ladder_state = LadderState(
            core=_state_from_env(env),
            opponent_memory=_update_opponent_memory(spec, ladder_state),
        )
        if step.terminated or step.truncated:
            break
    if not rows or rows[-1]["winner_after_step"] != ego_agent:
        raise RuntimeError(f"trace did not score for {reset_state_id} {ego_agent}")
    return rows


def _pad_frame_stack(
    history: list[list[str]],
    *,
    frame_stack: int,
) -> list[list[str]]:
    if not history:
        raise ValueError("cannot pad an empty raster history")
    if len(history) >= frame_stack:
        return [list(frame) for frame in history[-frame_stack:]]
    padding = [list(history[0]) for _ in range(frame_stack - len(history))]
    return padding + [list(frame) for frame in history]


def _env_from_state(config: PongConfig, state: SearchState) -> PongEnv:
    env = PongEnv(config)
    env._paddle_y = {
        "player_0": int(state.player_0_y),
        "player_1": int(state.player_1_y),
    }
    env._ball_x = int(state.ball_x)
    env._ball_y = int(state.ball_y)
    env._ball_vx = int(state.ball_vx)
    env._ball_vy = int(state.ball_vy)
    env._step = int(state.step)
    env._done = False
    env._last_hit = None
    env._last_hit_impact = None
    env._score_agent = None
    return env


def _state_from_env(env: PongEnv) -> SearchState:
    return SearchState(
        player_0_y=int(env._paddle_y["player_0"]),
        player_1_y=int(env._paddle_y["player_1"]),
        ball_x=int(env._ball_x),
        ball_y=int(env._ball_y),
        ball_vx=int(env._ball_vx),
        ball_vy=int(env._ball_vy),
        step=int(env._step),
    )


def _raster_grid(grid: Any) -> list[str]:
    return ["".join(str(int(cell)) for cell in row) for row in grid]


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, object]:
    target_histogram = _action_histogram_by_agent(rows)
    terminal_rewards = []
    for row in rows:
        if bool(row["terminated_after_step"]):
            terminal_rewards.append(float(row["reward_after_step"]))
    return {
        "rows": len(rows),
        "target_action_histogram_by_agent": {
            agent: {label: int(target_histogram[agent][label]) for label in ACTION_LABELS}
            for agent in AGENTS
        },
        "terminated_rows": sum(1 for row in rows if bool(row["terminated_after_step"])),
        "truncated_rows": sum(1 for row in rows if bool(row["truncated_after_step"])),
        "positive_reward_rows": sum(1 for row in rows if float(row["reward_after_step"]) > 0.0),
        "negative_reward_rows": sum(1 for row in rows if float(row["reward_after_step"]) < 0.0),
        "terminal_rewards": terminal_rewards,
    }


def _action_histogram_by_agent(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    target_histogram = {agent: Counter() for agent in AGENTS}
    for row in rows:
        target_histogram[str(row["ego_agent"])][str(row["target_action_label"])] += 1
    return {
        agent: {label: int(target_histogram[agent][label]) for label in ACTION_LABELS}
        for agent in AGENTS
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help=(
            "Number of chronological raster frames to store per row. Values above "
            "1 emit raster_frame_stack and next_raster_frame_stack for velocity cues."
        ),
    )
    parser.add_argument(
        "--include-vertical-mirror",
        action="store_true",
        help="Add valid top/bottom mirrored rows with up/down actions swapped.",
    )
    parser.add_argument(
        "--balance-actions",
        choices=BALANCE_ACTION_CHOICES,
        default="none",
        help="Replay-level action balancing. 'oversample' duplicates rare actions per ego agent.",
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=0,
        help="Seed for deterministic oversampling when action balancing is enabled.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = build_dummy_pong_lag1_trace_replay(
        max_steps=args.max_steps,
        repeats=args.repeats,
        frame_stack=args.frame_stack,
        include_vertical_mirror=args.include_vertical_mirror,
        balance_actions=args.balance_actions,
        balance_seed=args.balance_seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
