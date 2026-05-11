"""Exact sweep for small scoreable dummy Pong target replacements.

The default dummy Pong ``track_ball`` target is known to be unwinnable from
normal resets. This script keeps the same bounded dynamic-programming evidence
style, but sweeps a compact ladder of deterministic target changes:

- weaker track_ball timing policies;
- smaller symmetric paddles;
- narrow geometry;
- biased near-opponent-contact starts.

It reports whether any legal ego action sequence can score before the episode
cap, and keeps enough trace data to make the next curriculum target auditable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from curvyzero.training.dummy_pong import ACTION_LABELS
from curvyzero.training.dummy_pong import AGENTS
from curvyzero.training.dummy_pong import PongConfig

from probe_dummy_pong_track_ball_beatable import SearchState
from probe_dummy_pong_track_ball_beatable import TransitionResult
from probe_dummy_pong_track_ball_beatable import _normal_reset_states
from probe_dummy_pong_track_ball_beatable import _opponent
from probe_dummy_pong_track_ball_beatable import _paddle_y
from probe_dummy_pong_track_ball_beatable import _state_id
from probe_dummy_pong_track_ball_beatable import _step_state

PROBE_SCHEMA_ID = "dummy_pong_target_ladder_probe_v0"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    target_id: str
    description: str
    config: PongConfig
    opponent_policy: str
    reset_mode: str = "normal"
    reset_distance: int | None = None


@dataclass(frozen=True, slots=True)
class LadderState:
    core: SearchState
    opponent_memory: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class LadderTrace:
    target_id: str
    ego_agent: str
    reset_state_id: str
    reset_state: SearchState
    actions: list[int]
    steps: int
    winner: str
    final_state: SearchState


@dataclass(frozen=True, slots=True)
class LadderOutcome:
    trace: LadderTrace | None
    memoized_states: int


def probe_target_ladder(
    *,
    max_steps: int = 120,
    output_dir: Path | None = None,
) -> dict[str, object]:
    specs = _default_specs(max_steps=max_steps)
    all_rows: list[dict[str, object]] = []
    candidate_summaries = []
    for spec in specs:
        rows = _probe_spec(spec)
        all_rows.extend(rows)
        candidate_summaries.append(_summarize_spec(spec, rows))

    recommendation = _recommend(candidate_summaries)
    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_target_ladder_probe",
        "probe_schema_id": PROBE_SCHEMA_ID,
        "question": (
            "What is the smallest exact-scoreable target after default track_ball "
            "proved unwinnable from normal dummy Pong resets?"
        ),
        "search": {
            "type": "exact_dynamic_programming_over_finite_state_space",
            "ego_action_branching": len(ACTION_LABELS),
            "max_depth_steps": max_steps,
            "terminal_scoring": "same as PongEnv pure transition model",
        },
        "action_labels": list(ACTION_LABELS),
        "candidate_summaries": candidate_summaries,
        "recommendation": recommendation,
        "rows": all_rows,
    }
    if output_dir is not None:
        artifacts = {
            "summary_json": output_dir / "summary.json",
            "rows_jsonl": output_dir / "rows.jsonl",
        }
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, rows=all_rows)
    return summary


def _default_specs(*, max_steps: int) -> list[TargetSpec]:
    base = PongConfig(max_steps=max_steps)
    return [
        TargetSpec(
            target_id="default_track_ball_normal",
            description="Control: default geometry and deterministic track_ball.",
            config=base,
            opponent_policy="track_ball",
        ),
        TargetSpec(
            target_id="lag1_track_ball_normal",
            description="Default geometry; opponent tracks the previous ball row.",
            config=base,
            opponent_policy="lagged_track_ball_1",
        ),
        TargetSpec(
            target_id="lag2_track_ball_normal",
            description="Default geometry; opponent tracks the ball row from two states ago.",
            config=base,
            opponent_policy="lagged_track_ball_2",
        ),
        TargetSpec(
            target_id="every2_track_ball_normal",
            description="Default geometry; opponent tracks only on even steps, otherwise stays.",
            config=base,
            opponent_policy="track_ball_every_2",
        ),
        TargetSpec(
            target_id="every3_track_ball_normal",
            description="Default geometry; opponent tracks every third step, otherwise stays.",
            config=base,
            opponent_policy="track_ball_every_3",
        ),
        TargetSpec(
            target_id="paddle2_track_ball_normal",
            description="Symmetric smaller paddles; otherwise default track_ball.",
            config=PongConfig(paddle_height=2, max_steps=max_steps),
            opponent_policy="track_ball",
        ),
        TargetSpec(
            target_id="paddle1_track_ball_normal",
            description="Symmetric one-cell paddles; otherwise default track_ball.",
            config=PongConfig(paddle_height=1, max_steps=max_steps),
            opponent_policy="track_ball",
        ),
        TargetSpec(
            target_id="width9_track_ball_normal",
            description="Narrow width-9 geometry; otherwise default track_ball.",
            config=PongConfig(width=9, max_steps=max_steps),
            opponent_policy="track_ball",
        ),
        TargetSpec(
            target_id="height11_track_ball_normal",
            description="Taller height-11 geometry; otherwise default track_ball.",
            config=PongConfig(height=11, max_steps=max_steps),
            opponent_policy="track_ball",
        ),
        TargetSpec(
            target_id="default_track_ball_near_contact_d2",
            description=(
                "Default track_ball with starts two horizontal steps before the "
                "opponent paddle."
            ),
            config=base,
            opponent_policy="track_ball",
            reset_mode="near_opponent_contact",
            reset_distance=2,
        ),
        TargetSpec(
            target_id="default_track_ball_near_contact_d1",
            description=(
                "Default track_ball with starts one horizontal step before the "
                "opponent paddle."
            ),
            config=base,
            opponent_policy="track_ball",
            reset_mode="near_opponent_contact",
            reset_distance=1,
        ),
    ]


def _probe_spec(spec: TargetSpec) -> list[dict[str, object]]:
    rows = []
    for ego_agent in AGENTS:
        reset_states = _reset_states_for_ego(spec, ego_agent)
        for state_index, reset_state in enumerate(reset_states):
            reset_state_id = _state_id(state_index, reset_state)
            outcome = _find_winning_trace(spec=spec, initial_state=reset_state, ego_agent=ego_agent)
            trace = outcome.trace
            rows.append(
                {
                    "target_id": spec.target_id,
                    "description": spec.description,
                    "config": asdict(spec.config),
                    "opponent_policy": spec.opponent_policy,
                    "reset_mode": spec.reset_mode,
                    "reset_distance": spec.reset_distance,
                    "reset_state_id": reset_state_id,
                    "reset_state": asdict(reset_state),
                    "ego_agent": ego_agent,
                    "opponent_agent": _opponent(ego_agent),
                    "can_score": trace is not None,
                    "memoized_states": outcome.memoized_states,
                    "winning_steps": None if trace is None else trace.steps,
                    "winning_actions": None if trace is None else trace.actions,
                    "winning_action_labels": (
                        None
                        if trace is None
                        else [ACTION_LABELS[action] for action in trace.actions]
                    ),
                    "final_state": None if trace is None else asdict(trace.final_state),
                }
            )
    return rows


def _reset_states_for_ego(spec: TargetSpec, ego_agent: str) -> list[SearchState]:
    if spec.reset_mode == "normal":
        return _normal_reset_states(spec.config)
    if spec.reset_mode == "near_opponent_contact":
        if spec.reset_distance is None:
            raise ValueError("near_opponent_contact requires reset_distance")
        return _near_opponent_contact_states(spec.config, ego_agent, spec.reset_distance)
    raise ValueError(f"unknown reset_mode {spec.reset_mode!r}")


def _near_opponent_contact_states(
    config: PongConfig,
    ego_agent: str,
    distance: int,
) -> list[SearchState]:
    if distance < 1:
        raise ValueError("distance must be positive")
    center_y = (config.height - config.paddle_height) // 2
    if ego_agent == "player_0":
        ball_x = config.width - 2 - distance
        ball_vx = 1
    elif ego_agent == "player_1":
        ball_x = 1 + distance
        ball_vx = -1
    else:
        raise ValueError(f"unknown ego agent {ego_agent!r}")
    states = []
    for ball_y in range(2, config.height - 2):
        for ball_vy in (-1, 1):
            states.append(
                SearchState(
                    player_0_y=center_y,
                    player_1_y=center_y,
                    ball_x=ball_x,
                    ball_y=ball_y,
                    ball_vx=ball_vx,
                    ball_vy=ball_vy,
                    step=0,
                )
            )
    return states


def _find_winning_trace(
    *,
    spec: TargetSpec,
    initial_state: SearchState,
    ego_agent: str,
) -> LadderOutcome:
    initial = LadderState(
        core=initial_state,
        opponent_memory=_initial_opponent_memory(spec, initial_state),
    )
    memo: dict[LadderState, list[int] | None] = {}

    def solve(state: LadderState) -> list[int] | None:
        if state.core.step >= spec.config.max_steps:
            return None
        if state in memo:
            return memo[state]
        opponent_agent = _opponent(ego_agent)
        for ego_action in range(spec.config.action_count):
            actions = {
                ego_agent: ego_action,
                opponent_agent: _opponent_action(spec, state, opponent_agent),
            }
            transition = _step_state(state=state.core, actions=actions, config=spec.config)
            next_state = LadderState(
                core=transition.state,
                opponent_memory=_update_opponent_memory(spec, state),
            )
            if transition.winner == ego_agent:
                memo[state] = [ego_action]
                return memo[state]
            if transition.winner is not None or transition.truncated:
                continue
            suffix = solve(next_state)
            if suffix is not None:
                memo[state] = [ego_action] + suffix
                return memo[state]
        memo[state] = None
        return None

    actions = solve(initial)
    if actions is None:
        return LadderOutcome(trace=None, memoized_states=len(memo))
    final_transition = _rollout_actions(
        spec=spec,
        initial_state=initial_state,
        ego_agent=ego_agent,
        ego_actions=actions,
    )
    if final_transition.winner != ego_agent:
        raise RuntimeError("winning trace did not reproduce an ego score")
    return LadderOutcome(
        trace=LadderTrace(
            target_id=spec.target_id,
            ego_agent=ego_agent,
            reset_state_id="",
            reset_state=initial_state,
            actions=actions,
            steps=final_transition.state.step,
            winner=final_transition.winner,
            final_state=final_transition.state,
        ),
        memoized_states=len(memo),
    )


def _rollout_actions(
    *,
    spec: TargetSpec,
    initial_state: SearchState,
    ego_agent: str,
    ego_actions: list[int],
) -> TransitionResult:
    state = LadderState(
        core=initial_state,
        opponent_memory=_initial_opponent_memory(spec, initial_state),
    )
    transition = TransitionResult(
        state=initial_state,
        winner=None,
        truncated=False,
        hit_agent=None,
        impact_offset=None,
    )
    opponent_agent = _opponent(ego_agent)
    for ego_action in ego_actions:
        actions = {
            ego_agent: ego_action,
            opponent_agent: _opponent_action(spec, state, opponent_agent),
        }
        transition = _step_state(state=state.core, actions=actions, config=spec.config)
        state = LadderState(
            core=transition.state,
            opponent_memory=_update_opponent_memory(spec, state),
        )
        if transition.winner is not None or transition.truncated:
            break
    return transition


def _initial_opponent_memory(spec: TargetSpec, state: SearchState) -> tuple[int, ...]:
    delay = _lag_delay(spec.opponent_policy)
    if delay == 0:
        return ()
    return tuple([state.ball_y] * delay)


def _update_opponent_memory(spec: TargetSpec, state: LadderState) -> tuple[int, ...]:
    delay = _lag_delay(spec.opponent_policy)
    if delay == 0:
        return ()
    return state.opponent_memory[1:] + (state.core.ball_y,)


def _opponent_action(spec: TargetSpec, state: LadderState, agent: str) -> int:
    policy = spec.opponent_policy
    if policy == "track_ball":
        return _track_row_action(state.core, agent, spec.config, target_y=state.core.ball_y)
    if policy.startswith("lagged_track_ball_"):
        if not state.opponent_memory:
            raise ValueError(f"missing memory for {policy}")
        return _track_row_action(
            state.core,
            agent,
            spec.config,
            target_y=state.opponent_memory[0],
        )
    if policy.startswith("track_ball_every_"):
        period = int(policy.removeprefix("track_ball_every_"))
        if period < 1:
            raise ValueError(f"bad period in {policy!r}")
        if state.core.step % period != 0:
            return 1
        return _track_row_action(state.core, agent, spec.config, target_y=state.core.ball_y)
    raise ValueError(f"unknown opponent policy {policy!r}")


def _lag_delay(policy: str) -> int:
    if not policy.startswith("lagged_track_ball_"):
        return 0
    delay = int(policy.removeprefix("lagged_track_ball_"))
    if delay < 1:
        raise ValueError(f"bad delay in {policy!r}")
    return delay


def _track_row_action(
    state: SearchState,
    agent: str,
    config: PongConfig,
    *,
    target_y: int,
) -> int:
    center_y = _paddle_y(agent, state.player_0_y, state.player_1_y) + config.paddle_height // 2
    dy = target_y - center_y
    if dy < 0:
        return 0
    if dy > 0:
        return 2
    return 1


def _summarize_spec(spec: TargetSpec, rows: list[dict[str, object]]) -> dict[str, object]:
    winning_rows = [row for row in rows if row["can_score"]]
    winning_steps = [int(row["winning_steps"]) for row in winning_rows]
    shortest = None
    if winning_rows:
        shortest_row = min(winning_rows, key=lambda row: int(row["winning_steps"]))
        shortest = {
            "ego_agent": shortest_row["ego_agent"],
            "reset_state_id": shortest_row["reset_state_id"],
            "reset_state": shortest_row["reset_state"],
            "steps": shortest_row["winning_steps"],
            "actions": shortest_row["winning_actions"],
            "action_labels": shortest_row["winning_action_labels"],
            "final_state": shortest_row["final_state"],
        }
    return {
        "target_id": spec.target_id,
        "description": spec.description,
        "config": asdict(spec.config),
        "opponent_policy": spec.opponent_policy,
        "reset_mode": spec.reset_mode,
        "reset_distance": spec.reset_distance,
        "checked_reset_player_cases": len(rows),
        "cases_that_can_score": len(winning_rows),
        "all_cases_can_score": len(winning_rows) == len(rows),
        "win_coverage": len(winning_rows) / len(rows) if rows else 0.0,
        "memoized_states_total": sum(int(row["memoized_states"]) for row in rows),
        "winning_steps_min": min(winning_steps) if winning_steps else None,
        "winning_steps_median": median(winning_steps) if winning_steps else None,
        "winning_steps_max": max(winning_steps) if winning_steps else None,
        "shortest_winning_trace": shortest,
    }


def _recommend(candidate_summaries: list[dict[str, object]]) -> dict[str, object]:
    normal = [
        summary
        for summary in candidate_summaries
        if summary["reset_mode"] == "normal" and int(summary["cases_that_can_score"]) > 0
    ]
    complete_normal = [summary for summary in normal if summary["all_cases_can_score"]]
    if complete_normal:
        winner = sorted(
            complete_normal,
            key=lambda summary: (
                _change_rank(str(summary["target_id"])),
                int(summary["winning_steps_median"]),
            ),
        )[0]
        rationale = (
            "smallest normal-reset target with score traces in every checked "
            "reset/player case"
        )
    elif normal:
        winner = sorted(
            normal,
            key=lambda summary: (
                -float(summary["win_coverage"]),
                _change_rank(str(summary["target_id"])),
                int(summary["winning_steps_median"]),
            ),
        )[0]
        rationale = (
            "best normal-reset target with at least one exact scoring trace; "
            "coverage should be improved before making it a hard gate"
        )
    else:
        biased = [
            summary
            for summary in candidate_summaries
            if int(summary["cases_that_can_score"]) > 0
        ]
        winner = None if not biased else max(biased, key=lambda summary: float(summary["win_coverage"]))
        rationale = (
            "no normal-reset candidate scored; use biased starts only as a "
            "diagnostic pressure dataset"
        )
    if winner is None:
        return {
            "target_id": None,
            "rationale": "No candidate in the compact sweep had an exact scoring trace.",
        }
    return {
        "target_id": winner["target_id"],
        "rationale": rationale,
        "summary": winner,
    }


def _change_rank(target_id: str) -> int:
    if target_id.startswith("lag1_"):
        return 0
    if target_id.startswith("lag2_"):
        return 1
    if target_id.startswith("every2_"):
        return 2
    if target_id.startswith("every3_"):
        return 3
    if target_id.startswith("paddle2_"):
        return 4
    if target_id.startswith("paddle1_"):
        return 5
    if target_id.startswith("width9_"):
        return 6
    if target_id.startswith("height11_"):
        return 7
    return 99


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


def _console_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "kind": summary["kind"],
        "probe_schema_id": summary["probe_schema_id"],
        "recommendation": summary["recommendation"],
        "candidate_summaries": summary["candidate_summaries"],
        "artifacts": summary.get("artifacts"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--include-rows-in-stdout",
        action="store_true",
        help="Print the full rows payload instead of the compact summary.",
    )
    args = parser.parse_args()

    summary = probe_target_ladder(max_steps=args.max_steps, output_dir=args.output_dir)
    if args.include_rows_in_stdout:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(_console_summary(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
