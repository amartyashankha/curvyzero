"""Exact bounded probe for whether dummy Pong can beat track_ball.

The current default reset has only 20 unique initial geometry states:

- ball_y in [2, height - 3]
- ball_vx in {-1, 1}
- ball_vy in {-1, 1}
- both paddles centered

This script does dynamic programming over the finite deterministic state space
for one ego player against a fixed track_ball opponent. It answers whether
there exists any legal ego action sequence that scores before max_steps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from curvyzero.training.dummy_pong import ACTION_LABELS
from curvyzero.training.dummy_pong import AGENTS
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv

PROBE_SCHEMA_ID = "dummy_pong_track_ball_beatable_probe_v0"


@dataclass(frozen=True, slots=True)
class SearchState:
    player_0_y: int
    player_1_y: int
    ball_x: int
    ball_y: int
    ball_vx: int
    ball_vy: int
    step: int


@dataclass(frozen=True, slots=True)
class TransitionResult:
    state: SearchState
    winner: str | None
    truncated: bool
    hit_agent: str | None
    impact_offset: int | None


@dataclass(frozen=True, slots=True)
class WinningTrace:
    ego_agent: str
    reset_state_id: str
    reset_state: SearchState
    example_seed: int | None
    actions: list[int]
    steps: int
    winner: str
    final_state: SearchState


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    winning_trace: WinningTrace | None
    memoized_states: int


def probe_track_ball_beatability(
    *,
    width: int = 15,
    height: int = 9,
    paddle_height: int = 3,
    max_steps: int = 120,
    seed_search_limit: int = 10_000,
    output_dir: Path | None = None,
) -> dict[str, object]:
    config = PongConfig(
        width=width,
        height=height,
        paddle_height=paddle_height,
        max_steps=max_steps,
    )
    reset_states = _normal_reset_states(config)
    example_seed_by_state = _example_seed_by_state(
        config=config,
        states=reset_states,
        seed_search_limit=seed_search_limit,
    )
    transition_validation = _validate_transition_against_env(config, reset_states)

    rows = []
    winning_traces: list[WinningTrace] = []
    memoized_state_counts = []
    for state_index, reset_state in enumerate(reset_states):
        reset_state_id = _state_id(state_index, reset_state)
        for ego_agent in AGENTS:
            search = _find_winning_trace(config=config, initial_state=reset_state, ego_agent=ego_agent)
            trace = search.winning_trace
            memoized_state_counts.append(search.memoized_states)
            row = {
                "reset_state_id": reset_state_id,
                "reset_state": asdict(reset_state),
                "example_seed": example_seed_by_state.get(reset_state),
                "ego_agent": ego_agent,
                "opponent_agent": _opponent(ego_agent),
                "can_score": trace is not None,
                "memoized_states": search.memoized_states,
                "winning_steps": None if trace is None else trace.steps,
                "winning_actions": None if trace is None else trace.actions,
                "winning_action_labels": (
                    None
                    if trace is None
                    else [ACTION_LABELS[action] for action in trace.actions]
                ),
            }
            rows.append(row)
            if trace is not None:
                winning_traces.append(
                    WinningTrace(
                        ego_agent=ego_agent,
                        reset_state_id=reset_state_id,
                        reset_state=reset_state,
                        example_seed=example_seed_by_state.get(reset_state),
                        actions=trace.actions,
                        steps=trace.steps,
                        winner=trace.winner,
                        final_state=trace.final_state,
                    )
                )

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_track_ball_beatable_probe",
        "probe_schema_id": PROBE_SCHEMA_ID,
        "question": (
            "Can any legal ego action sequence score against deterministic track_ball "
            "from the normal reset support?"
        ),
        "config": asdict(config),
        "action_labels": list(ACTION_LABELS),
        "normal_reset_support": {
            "states": len(reset_states),
            "player_seatings": len(reset_states) * len(AGENTS),
            "definition": (
                "Both paddles centered; ball_x=width//2; ball_y sampled from "
                "integers(2, height-2); ball_vx and ball_vy sampled from {-1, 1}."
            ),
            "example_seed_search_limit": seed_search_limit,
            "example_seed_coverage": sum(
                1 for state in reset_states if state in example_seed_by_state
            ),
        },
        "search": {
            "type": "exact_dynamic_programming_over_finite_state_space",
            "ego_action_branching": len(ACTION_LABELS),
            "opponent_policy": "track_ball",
            "max_depth_steps": max_steps,
            "terminal_scoring": "same as PongEnv: score beats max-step truncation",
            "state_bound": {
                "paddle_y_values_per_agent": config.height - config.paddle_height + 1,
                "ball_x_values_nonterminal": config.width,
                "ball_y_values": config.height,
                "ball_vx_values": 2,
                "ball_vy_values": 3,
                "steps": max_steps + 1,
                "loose_state_count": (
                    (config.height - config.paddle_height + 1) ** 2
                    * config.width
                    * config.height
                    * 2
                    * 3
                    * (max_steps + 1)
                ),
            },
        },
        "transition_validation": transition_validation,
        "result": {
            "can_any_policy_score": bool(winning_traces),
            "winning_trace_count": len(winning_traces),
            "checked_reset_player_cases": len(rows),
            "cases_that_can_score": sum(1 for row in rows if row["can_score"]),
            "memoized_states_total_over_cases": sum(memoized_state_counts),
            "memoized_states_min_per_case": min(memoized_state_counts),
            "memoized_states_max_per_case": max(memoized_state_counts),
            "shortest_winning_trace": (
                None
                if not winning_traces
                else _winning_trace_summary(min(winning_traces, key=lambda trace: trace.steps))
            ),
        },
        "rows": rows,
        "interpretation": _interpretation(bool(winning_traces), max_steps=max_steps),
    }

    if output_dir is not None:
        artifacts = {
            "summary_json": output_dir / "summary.json",
            "rows_jsonl": output_dir / "rows.jsonl",
        }
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, rows=rows)
    return summary


def _normal_reset_states(config: PongConfig) -> list[SearchState]:
    center_y = (config.height - config.paddle_height) // 2
    states = []
    for ball_y in range(2, config.height - 2):
        for ball_vx in (-1, 1):
            for ball_vy in (-1, 1):
                states.append(
                    SearchState(
                        player_0_y=center_y,
                        player_1_y=center_y,
                        ball_x=config.width // 2,
                        ball_y=ball_y,
                        ball_vx=ball_vx,
                        ball_vy=ball_vy,
                        step=0,
                    )
                )
    return states


def _example_seed_by_state(
    *,
    config: PongConfig,
    states: list[SearchState],
    seed_search_limit: int,
) -> dict[SearchState, int]:
    wanted = set(states)
    found: dict[SearchState, int] = {}
    for seed in range(seed_search_limit):
        state = _reset_state_from_seed(config, seed)
        if state in wanted and state not in found:
            found[state] = seed
        if len(found) == len(wanted):
            break
    return found


def _reset_state_from_seed(config: PongConfig, seed: int) -> SearchState:
    rng = np.random.default_rng(seed)
    center_y = (config.height - config.paddle_height) // 2
    return SearchState(
        player_0_y=center_y,
        player_1_y=center_y,
        ball_x=config.width // 2,
        ball_y=int(rng.integers(2, config.height - 2)),
        ball_vx=-1 if int(rng.integers(2)) == 0 else 1,
        ball_vy=-1 if int(rng.integers(2)) == 0 else 1,
        step=0,
    )


def _find_winning_trace(
    *,
    config: PongConfig,
    initial_state: SearchState,
    ego_agent: str,
) -> SearchOutcome:
    memo: dict[SearchState, list[int] | None] = {}

    def solve(state: SearchState) -> list[int] | None:
        if state.step >= config.max_steps:
            return None
        if state in memo:
            return memo[state]
        for ego_action in range(config.action_count):
            opponent_agent = _opponent(ego_agent)
            actions = {
                ego_agent: ego_action,
                opponent_agent: _track_ball_action(state, opponent_agent, config),
            }
            transition = _step_state(state=state, actions=actions, config=config)
            if transition.winner == ego_agent:
                memo[state] = [ego_action]
                return memo[state]
            if transition.winner is not None or transition.truncated:
                continue
            suffix = solve(transition.state)
            if suffix is not None:
                memo[state] = [ego_action] + suffix
                return memo[state]
        memo[state] = None
        return None

    actions = solve(initial_state)
    if actions is None:
        return SearchOutcome(winning_trace=None, memoized_states=len(memo))
    final_transition = _rollout_actions(
        config=config,
        initial_state=initial_state,
        ego_agent=ego_agent,
        ego_actions=actions,
    )
    if final_transition.winner != ego_agent:
        raise RuntimeError("winning trace did not reproduce an ego score")
    return SearchOutcome(
        winning_trace=WinningTrace(
            ego_agent=ego_agent,
            reset_state_id="",
            reset_state=initial_state,
            example_seed=None,
            actions=actions,
            steps=final_transition.state.step,
            winner=final_transition.winner,
            final_state=final_transition.state,
        ),
        memoized_states=len(memo),
    )


def _rollout_actions(
    *,
    config: PongConfig,
    initial_state: SearchState,
    ego_agent: str,
    ego_actions: list[int],
) -> TransitionResult:
    state = initial_state
    transition = TransitionResult(
        state=state,
        winner=None,
        truncated=False,
        hit_agent=None,
        impact_offset=None,
    )
    for ego_action in ego_actions:
        actions = {
            ego_agent: ego_action,
            _opponent(ego_agent): _track_ball_action(state, _opponent(ego_agent), config),
        }
        transition = _step_state(state=state, actions=actions, config=config)
        state = transition.state
        if transition.winner is not None or transition.truncated:
            break
    return transition


def _step_state(
    *,
    state: SearchState,
    actions: dict[str, int],
    config: PongConfig,
) -> TransitionResult:
    p0_y = _move_paddle(state.player_0_y, actions["player_0"], config)
    p1_y = _move_paddle(state.player_1_y, actions["player_1"], config)

    ball_vx = state.ball_vx
    ball_vy = state.ball_vy
    next_x = state.ball_x + ball_vx
    next_y = state.ball_y + ball_vy
    if next_y < 0:
        next_y = 1
        ball_vy = 1
    elif next_y >= config.height:
        next_y = config.height - 2
        ball_vy = -1

    hit_agent = _paddle_hit_agent(
        next_x=next_x,
        next_y=next_y,
        ball_vx=ball_vx,
        player_0_y=p0_y,
        player_1_y=p1_y,
        config=config,
    )
    impact_offset = None
    if hit_agent is not None:
        ball_vx *= -1
        next_x = _paddle_x(config, hit_agent) + ball_vx
        impact_offset = next_y - (_paddle_y(hit_agent, p0_y, p1_y) + config.paddle_height // 2)
        if impact_offset < 0:
            ball_vy = -1
        elif impact_offset > 0:
            ball_vy = 1
        else:
            ball_vy = 0

    next_step = state.step + 1
    winner = None
    if next_x < 0:
        winner = "player_1"
    elif next_x >= config.width:
        winner = "player_0"
    truncated = next_step >= config.max_steps and winner is None
    return TransitionResult(
        state=SearchState(
            player_0_y=p0_y,
            player_1_y=p1_y,
            ball_x=next_x,
            ball_y=next_y,
            ball_vx=ball_vx,
            ball_vy=ball_vy,
            step=next_step,
        ),
        winner=winner,
        truncated=truncated,
        hit_agent=hit_agent,
        impact_offset=impact_offset,
    )


def _move_paddle(paddle_y: int, action: int, config: PongConfig) -> int:
    delta = action - 1
    return min(max(paddle_y + delta, 0), config.height - config.paddle_height)


def _track_ball_action(state: SearchState, agent: str, config: PongConfig) -> int:
    center_y = _paddle_y(agent, state.player_0_y, state.player_1_y) + config.paddle_height // 2
    ball_dy_from_center = state.ball_y - center_y
    if ball_dy_from_center < 0:
        return 0
    if ball_dy_from_center > 0:
        return 2
    return 1


def _paddle_hit_agent(
    *,
    next_x: int,
    next_y: int,
    ball_vx: int,
    player_0_y: int,
    player_1_y: int,
    config: PongConfig,
) -> str | None:
    candidate = "player_0" if ball_vx < 0 else "player_1"
    if next_x != _paddle_x(config, candidate):
        return None
    top = _paddle_y(candidate, player_0_y, player_1_y)
    if top <= next_y < top + config.paddle_height:
        return candidate
    return None


def _paddle_x(config: PongConfig, agent: str) -> int:
    if agent == "player_0":
        return 1
    if agent == "player_1":
        return config.width - 2
    raise ValueError(f"unknown agent {agent!r}")


def _paddle_y(agent: str, player_0_y: int, player_1_y: int) -> int:
    if agent == "player_0":
        return player_0_y
    if agent == "player_1":
        return player_1_y
    raise ValueError(f"unknown agent {agent!r}")


def _opponent(agent: str) -> str:
    if agent == "player_0":
        return "player_1"
    if agent == "player_1":
        return "player_0"
    raise ValueError(f"unknown agent {agent!r}")


def _state_id(index: int, state: SearchState) -> str:
    return (
        f"reset-{index:02d}-y{state.ball_y}-vx{state.ball_vx:+d}-"
        f"vy{state.ball_vy:+d}"
    )


def _winning_trace_summary(trace: WinningTrace) -> dict[str, object]:
    return {
        "ego_agent": trace.ego_agent,
        "reset_state_id": trace.reset_state_id,
        "reset_state": asdict(trace.reset_state),
        "example_seed": trace.example_seed,
        "steps": trace.steps,
        "winner": trace.winner,
        "actions": trace.actions,
        "action_labels": [ACTION_LABELS[action] for action in trace.actions],
        "final_state": asdict(trace.final_state),
    }


def _validate_transition_against_env(
    config: PongConfig,
    reset_states: list[SearchState],
) -> dict[str, object]:
    checked = 0
    for state in reset_states:
        for ego_agent in AGENTS:
            opponent_agent = _opponent(ego_agent)
            for ego_action in range(config.action_count):
                actions = {
                    ego_agent: ego_action,
                    opponent_agent: _track_ball_action(state, opponent_agent, config),
                }
                expected = _step_state(state=state, actions=actions, config=config)
                actual = _step_env_from_state(config=config, state=state, actions=actions)
                checked += 1
                if expected != actual:
                    return {
                        "passed": False,
                        "checked_single_step_cases": checked,
                        "mismatch": {
                            "state": asdict(state),
                            "actions": actions,
                            "expected": _transition_dict(expected),
                            "actual": _transition_dict(actual),
                        },
                    }
    return {
        "passed": True,
        "checked_single_step_cases": checked,
        "scope": "all normal reset support states, both ego seats, all ego actions",
    }


def _step_env_from_state(
    *,
    config: PongConfig,
    state: SearchState,
    actions: dict[str, int],
) -> TransitionResult:
    env = PongEnv(config)
    env.reset(seed=0)
    env._paddle_y = {"player_0": state.player_0_y, "player_1": state.player_1_y}
    env._ball_x = state.ball_x
    env._ball_y = state.ball_y
    env._ball_vx = state.ball_vx
    env._ball_vy = state.ball_vy
    env._step = state.step
    env._done = False
    env._last_hit = None
    env._last_hit_impact = None
    env._score_agent = None
    result = env.step(actions)
    impact = result.infos["last_hit_impact"]
    return TransitionResult(
        state=SearchState(
            player_0_y=int(result.infos["paddles"]["player_0"]),
            player_1_y=int(result.infos["paddles"]["player_1"]),
            ball_x=int(result.infos["ball"]["x"]),
            ball_y=int(result.infos["ball"]["y"]),
            ball_vx=int(result.infos["ball"]["vx"]),
            ball_vy=int(result.infos["ball"]["vy"]),
            step=int(next(iter(result.observations.values())).step),
        ),
        winner=result.infos["winner"] if result.terminated else None,
        truncated=result.truncated,
        hit_agent=result.infos["last_hit"],
        impact_offset=None if impact is None else int(impact["impact_offset"]),
    )


def _transition_dict(transition: TransitionResult) -> dict[str, object]:
    return {
        "state": asdict(transition.state),
        "winner": transition.winner,
        "truncated": transition.truncated,
        "hit_agent": transition.hit_agent,
        "impact_offset": transition.impact_offset,
    }


def _interpretation(can_score: bool, *, max_steps: int) -> str:
    if can_score:
        return (
            "At least one normal reset state has a legal ego action sequence that "
            "scores against track_ball under the current geometry."
        )
    return (
        "No legal ego action sequence scores against track_ball from any normal "
        f"reset support state within the {max_steps}-step episode cap. Under this "
        "exact bound, track_ball is a bad hard win baseline for the toy because "
        "perfect play can at best force the same full-survival tie."
    )


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
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--height", type=int, default=9)
    parser.add_argument("--paddle-height", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed-search-limit", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = probe_track_ball_beatability(
        width=args.width,
        height=args.height,
        paddle_height=args.paddle_height,
        max_steps=args.max_steps,
        seed_search_limit=args.seed_search_limit,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
