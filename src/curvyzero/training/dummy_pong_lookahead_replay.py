"""Build short-lookahead policy-improvement replay for dummy Pong."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from itertools import product
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
from curvyzero.training.dummy_pong_eval import AngleControlPolicy
from curvyzero.training.dummy_pong_eval import BaselinePolicy
from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
from curvyzero.training.dummy_pong_eval import TrackBallPolicy

LOOKAHEAD_REPLAY_SUMMARY_SCHEMA_ID = "dummy_pong_lookahead_replay_summary_v0"
LOOKAHEAD_REPLAY_ROW_SCHEMA_ID = "dummy_pong_lookahead_replay_row_v0"
LOOKAHEAD_TARGET_SCHEMA_ID = "dummy_pong_score_delta_short_lookahead_target_v0"
LOOKAHEAD_LOSS_DELAY_TARGET_SCHEMA_ID = (
    "dummy_pong_score_delta_loss_delay_short_lookahead_target_v0"
)
LOOKAHEAD_REPLAY_SEMANTICS_ID = "dummy_pong_counterfactual_supervised_action_replay_v0"
COLLECTOR_POLICIES = ("random_uniform", "track_ball", "angle_control")
TIE_BREAK_POLICIES = ("track_ball", "angle_control")


def build_dummy_pong_lookahead_replay(
    *,
    games_per_seat: int,
    seed: int,
    output_dir: Path | None = None,
    max_steps: int = 120,
    lookahead_steps: int = 24,
    ego_sequence_depth: int = 1,
    collector_policy: str = "random_uniform",
    include_ties: bool = False,
    tie_break_policy: str = "track_ball",
    loss_delay_alpha: float = 0.0,
) -> dict[str, object]:
    """Collect states and relabel ego actions by short score-delta lookahead."""

    if games_per_seat < 1:
        raise ValueError("games_per_seat must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if lookahead_steps < 1:
        raise ValueError("lookahead_steps must be at least 1")
    if ego_sequence_depth not in (1, 2):
        raise ValueError("ego_sequence_depth must be 1 or 2")
    if collector_policy not in COLLECTOR_POLICIES:
        raise ValueError(f"collector_policy must be one of {COLLECTOR_POLICIES!r}")
    if tie_break_policy not in TIE_BREAK_POLICIES:
        raise ValueError(f"tie_break_policy must be one of {TIE_BREAK_POLICIES!r}")
    if loss_delay_alpha < 0.0:
        raise ValueError("loss_delay_alpha must be non-negative")

    config = PongConfig(max_steps=max_steps)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    games_summary: list[dict[str, object]] = []
    shaping_id = _shaping_id(loss_delay_alpha)
    depth_id = "" if ego_sequence_depth == 1 else f"-ego-depth-{ego_sequence_depth}"
    run_id = (
        f"dummy-pong-lookahead-seed-{seed}-games-per-seat-{games_per_seat}"
        f"-max-{max_steps}-horizon-{lookahead_steps}{depth_id}-{shaping_id}"
    )
    target_policy_id = _target_policy_id(
        lookahead_steps=lookahead_steps,
        ego_sequence_depth=ego_sequence_depth,
        include_ties=include_ties,
        tie_break_policy=tie_break_policy,
        loss_delay_alpha=loss_delay_alpha,
    )
    target_schema_id = _target_schema_id(loss_delay_alpha)

    for seating_index, ego_agent in enumerate(AGENTS):
        policy_by_agent = {
            agent: (collector_policy if agent == ego_agent else "track_ball")
            for agent in AGENTS
        }
        for seat_game_index in range(games_per_seat):
            game_seed = int(rng.integers(2**31 - 1))
            game_index = seating_index * games_per_seat + seat_game_index
            game_rows, game_summary = _run_game(
                config=config,
                run_id=run_id,
                game_index=game_index,
                seat_game_index=seat_game_index,
                seed=game_seed,
                policy_seed=seed + game_index * 10_000,
                ego_agent=ego_agent,
                policy_by_agent=policy_by_agent,
                lookahead_steps=lookahead_steps,
                ego_sequence_depth=ego_sequence_depth,
                target_policy_id=target_policy_id,
                target_schema_id=target_schema_id,
                include_ties=include_ties,
                tie_break_policy=tie_break_policy,
                loss_delay_alpha=loss_delay_alpha,
            )
            rows.extend(game_rows)
            games_summary.append(game_summary)

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_lookahead_replay",
        "note": (
            "Short score-delta lookahead relabeling from raster Pong states. "
            "Each target is the immediate ego action with the best rollout "
            "return against fixed track_ball play; this is a small policy-input "
            "improvement lane, not MuZero or learned dynamics."
        ),
        "run_id": run_id,
        "summary_schema_id": LOOKAHEAD_REPLAY_SUMMARY_SCHEMA_ID,
        "row_schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
        "target_schema_id": target_schema_id,
        "replay_semantics_id": LOOKAHEAD_REPLAY_SEMANTICS_ID,
        "target_policy_id": target_policy_id,
        "seed": seed,
        "games_per_seat": games_per_seat,
        "games": len(games_summary),
        "max_steps": max_steps,
        "lookahead_steps": lookahead_steps,
        "ego_sequence_depth": ego_sequence_depth,
        "loss_delay_alpha": loss_delay_alpha,
        "collector_policy": collector_policy,
        "opponent_policy": "track_ball",
        "include_ties": include_ties,
        "tie_break_policy": tie_break_policy,
        "total_rows": len(rows),
        "total_collector_steps": sum(int(game["steps"]) for game in games_summary),
        "config": asdict(config),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "lookahead_replay_summary_schema_id": LOOKAHEAD_REPLAY_SUMMARY_SCHEMA_ID,
            "lookahead_replay_row_schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
            "lookahead_target_schema_id": LOOKAHEAD_TARGET_SCHEMA_ID,
            "lookahead_loss_delay_target_schema_id": LOOKAHEAD_LOSS_DELAY_TARGET_SCHEMA_ID,
            "lookahead_replay_semantics_id": LOOKAHEAD_REPLAY_SEMANTICS_ID,
        },
        "raster": {
            "shape": [config.height, config.width],
            "encoding": "row-major digit strings over the raster legend",
            "legend": RASTER_LEGEND,
        },
        "target_construction": {
            "candidate_actions": list(ACTION_LABELS),
            "candidate_ego_sequences": [
                list(_sequence_labels(sequence))
                for sequence in _candidate_sequences(ego_sequence_depth)
            ],
            "return": "sum of ego score-delta rewards over lookahead_steps",
            "target_return": _target_return_description(loss_delay_alpha),
            "first_step": (
                "candidate ego action plus fixed-opponent track_ball action from "
                "the sampled state"
            ),
            "forced_ego_sequence_depth": ego_sequence_depth,
            "forced_opponent_policy": "track_ball on every forced step",
            "rollout_after_first_step": (
                "track_ball for both agents"
                if ego_sequence_depth == 1
                else "continue forced ego sequence, then track_ball for both agents"
            ),
            "rollout_after_forced_steps": "track_ball for both agents",
            "tie_break": f"prefer the current {tie_break_policy} action when it is tied best",
            "row_filter": (
                "all sampled states" if include_ties else "states where candidate returns differ"
            ),
            "horizon": (
                "up to lookahead_steps, stopping early when the candidate rollout "
                "terminates or truncates"
            ),
            "replay_semantics": (
                "Rows are supervised action labels from counterfactual candidate "
                "steps. reward_after_step and next_* describe the chosen target "
                "action, not the collector action that gathered the state."
            ),
            "shaping_warning": (
                "The loss-delay adjustment is for training labels only. Pong eval "
                "still uses true score wins, losses, and truncations."
            )
            if loss_delay_alpha > 0.0
            else None,
        },
        "action_labels": list(ACTION_LABELS),
        "data": _summarize_rows(rows),
        "collector_games": _outcome_summary(games_summary),
        "games_summary": games_summary,
        "plain_language": {
            "proves": (
                "The raster learner can receive action labels chosen by score-delta "
                "lookahead instead of plain action cloning."
            ),
            "does_not_prove": (
                "Policy improvement by itself. Progress still has to come from "
                "checkpoint scoreboard wins or pressure against track_ball."
            ),
        },
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
    seat_game_index: int,
    seed: int,
    policy_seed: int,
    ego_agent: str,
    policy_by_agent: dict[str, str],
    lookahead_steps: int,
    ego_sequence_depth: int,
    target_policy_id: str,
    target_schema_id: str,
    include_ties: bool,
    tie_break_policy: str,
    loss_delay_alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    env = PongEnv(config)
    observations = env.reset(seed=seed)
    policies = {
        agent: _make_policy(policy_by_agent[agent], seed=policy_seed + index * 997)
        for index, agent in enumerate(AGENTS)
    }
    for agent, policy in policies.items():
        policy.reset(seed, agent)

    rows: list[dict[str, Any]] = []
    total_rewards = {agent: 0.0 for agent in AGENTS}
    sampled_states = 0
    filtered_tie_states = 0
    final_step = None

    while True:
        step_index = int(next(iter(observations.values())).step)
        raster_observation = env.raster_observation()
        collector_joint_action = {
            agent: int(policies[agent].action(observations[agent], raster_observation, agent))
            for agent in AGENTS
        }
        sampled_states += 1
        row = _lookahead_row(
            env=env,
            config=config,
            run_id=run_id,
            game_index=game_index,
            seat_game_index=seat_game_index,
            seed=seed,
            step_index=step_index,
            ego_agent=ego_agent,
            policy_by_agent=policy_by_agent,
            collector_joint_action=collector_joint_action,
            observation=observations[ego_agent],
            raster_observation=raster_observation,
            lookahead_steps=lookahead_steps,
            ego_sequence_depth=ego_sequence_depth,
            target_policy_id=target_policy_id,
            target_schema_id=target_schema_id,
            tie_break_policy=tie_break_policy,
            loss_delay_alpha=loss_delay_alpha,
        )
        if include_ties or float(row["lookahead"]["return_spread"]) > 0.0:
            rows.append(row)
        else:
            filtered_tie_states += 1

        final_step = env.step(collector_joint_action)
        for agent in AGENTS:
            total_rewards[agent] += float(final_step.rewards[agent])
        observations = final_step.observations
        if final_step.terminated or final_step.truncated:
            break

    if final_step is None:
        raise RuntimeError("dummy Pong lookahead collector game produced no steps")

    return rows, {
        "game_index": game_index,
        "seat_game_index": seat_game_index,
        "seed": seed,
        "ego_agent": ego_agent,
        "policy_by_agent": dict(policy_by_agent),
        "steps": int(next(iter(final_step.observations.values())).step),
        "sampled_states": sampled_states,
        "emitted_rows": len(rows),
        "filtered_tie_states": filtered_tie_states,
        "winner": final_step.infos["winner"] if final_step.terminated else None,
        "terminated": final_step.terminated,
        "truncated": final_step.truncated,
        "total_rewards": total_rewards,
        "last_hit": final_step.infos["last_hit"],
    }


def _lookahead_row(
    *,
    env: PongEnv,
    config: PongConfig,
    run_id: str,
    game_index: int,
    seat_game_index: int,
    seed: int,
    step_index: int,
    ego_agent: str,
    policy_by_agent: dict[str, str],
    collector_joint_action: dict[str, int],
    observation: PongObservation,
    raster_observation: np.ndarray,
    lookahead_steps: int,
    ego_sequence_depth: int,
    target_policy_id: str,
    target_schema_id: str,
    tie_break_policy: str,
    loss_delay_alpha: float,
) -> dict[str, Any]:
    opponent_agent = _opponent(ego_agent)
    baseline_policy = TrackBallPolicy()
    baseline_policy.reset(seed, ego_agent)
    baseline_action = int(baseline_policy.action(observation, raster_observation, ego_agent))
    tie_break_action = _tie_break_action(
        policy_id=tie_break_policy,
        observation=observation,
        raster_observation=raster_observation,
        ego_agent=ego_agent,
        seed=seed,
    )
    candidate_sequences = _candidate_sequences(ego_sequence_depth)
    candidate_sequence_results = {
        sequence: _evaluate_candidate_sequence(
            env=env,
            config=config,
            ego_agent=ego_agent,
            ego_action_sequence=sequence,
            seed=seed,
            lookahead_steps=lookahead_steps,
            loss_delay_alpha=loss_delay_alpha,
        )
        for sequence in candidate_sequences
    }
    if ego_sequence_depth == 1:
        candidate_results = {
            sequence[0]: result
            for sequence, result in candidate_sequence_results.items()
        }
        target_action, target_source, best_actions = _choose_target_action(
            candidate_results=candidate_results,
            tie_break_action=tie_break_action,
            tie_break_policy=tie_break_policy,
        )
        chosen_sequence = (target_action,)
        best_sequences = [(action,) for action in best_actions]
    else:
        chosen_sequence, target_source, best_sequences = _choose_target_sequence(
            candidate_results=candidate_sequence_results,
            tie_break_action=tie_break_action,
            tie_break_policy=tie_break_policy,
        )
        target_action = chosen_sequence[0]
        best_actions = sorted({sequence[0] for sequence in best_sequences})

    candidate_results_by_first_action = _best_results_by_first_action(
        candidate_sequence_results
    )
    chosen = candidate_sequence_results[chosen_sequence]
    opponent_action = int(chosen["first_step_opponent_action"])
    return_values = [
        float(candidate_sequence_results[sequence]["target_return"])
        for sequence in candidate_sequences
    ]
    score_delta_return_values = [
        float(candidate_sequence_results[sequence]["score_delta_return"])
        for sequence in candidate_sequences
    ]
    return_spread = max(return_values) - min(return_values)
    score_delta_return_spread = max(score_delta_return_values) - min(score_delta_return_values)

    return {
        "schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
        "run_id": run_id,
        "game_index": game_index,
        "seat_game_index": seat_game_index,
        "seed": seed,
        "step_index": step_index,
        "ego_agent": ego_agent,
        "opponent_agent": opponent_agent,
        "collector_policy_id": policy_by_agent[ego_agent],
        "behavior_policy_id": policy_by_agent[ego_agent],
        "opponent_policy_id": policy_by_agent[opponent_agent],
        "target_policy_id": target_policy_id,
        "target_schema_id": target_schema_id,
        "replay_semantics_id": LOOKAHEAD_REPLAY_SEMANTICS_ID,
        "target_action_id": target_action,
        "target_action_label": ACTION_LABELS[target_action],
        "target_action_source": target_source,
        "chosen_ego_sequence_depth": ego_sequence_depth,
        "chosen_ego_sequence_action_ids": list(chosen_sequence),
        "chosen_ego_sequence_action_labels": list(_sequence_labels(chosen_sequence)),
        "chosen_ego_sequence_label": _sequence_label(chosen_sequence),
        "target_return": float(chosen["target_return"]),
        "target_return_kind": str(chosen["target_return_kind"]),
        "target_score_delta_return": float(chosen["score_delta_return"]),
        "target_loss_delay_bonus": float(chosen["loss_delay_bonus"]),
        "loss_delay_alpha": loss_delay_alpha,
        "collector_action_id": int(collector_joint_action[ego_agent]),
        "collector_action_label": ACTION_LABELS[int(collector_joint_action[ego_agent])],
        "track_ball_action_id": baseline_action,
        "track_ball_action_label": ACTION_LABELS[baseline_action],
        "tie_break_policy_id": tie_break_policy,
        "tie_break_action_id": tie_break_action,
        "tie_break_action_label": ACTION_LABELS[tie_break_action],
        "joint_action_by_agent": {
            ego_agent: {
                "action_id": target_action,
                "label": ACTION_LABELS[target_action],
                "action_source": "lookahead_target",
            },
            opponent_agent: {
                "action_id": opponent_action,
                "label": ACTION_LABELS[opponent_action],
                "action_source": "fixed_opponent_track_ball",
            },
        },
        "collector_joint_action_by_agent": {
            agent: {
                "action_id": int(action),
                "label": ACTION_LABELS[int(action)],
                "policy_id": policy_by_agent[agent],
            }
            for agent, action in collector_joint_action.items()
        },
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "raster_shape": [int(raster_observation.shape[0]), int(raster_observation.shape[1])],
        "raster_grid": _raster_grid(raster_observation),
        "observation": asdict(observation),
        "reward_after_step": float(chosen["first_step_reward"]),
        "next_raster_grid": chosen["next_raster_grid"],
        "next_observation": chosen["next_observation"],
        "terminated_after_step": bool(chosen["first_step_terminated"]),
        "truncated_after_step": bool(chosen["first_step_truncated"]),
        "lookahead": {
            "lookahead_steps": lookahead_steps,
            "ego_sequence_depth": ego_sequence_depth,
            "candidate_returns_by_action_label": {
                ACTION_LABELS[action]: float(
                    candidate_results_by_first_action[action]["target_return"]
                )
                for action in range(len(ACTION_LABELS))
            },
            "candidate_score_delta_returns_by_action_label": {
                ACTION_LABELS[action]: float(
                    candidate_results_by_first_action[action]["score_delta_return"]
                )
                for action in range(len(ACTION_LABELS))
            },
            "candidate_loss_delay_bonuses_by_action_label": {
                ACTION_LABELS[action]: float(
                    candidate_results_by_first_action[action]["loss_delay_bonus"]
                )
                for action in range(len(ACTION_LABELS))
            },
            "candidate_first_step_rewards_by_action_label": {
                ACTION_LABELS[action]: float(
                    candidate_results_by_first_action[action]["first_step_reward"]
                )
                for action in range(len(ACTION_LABELS))
            },
            "candidate_terminal_by_action_label": {
                ACTION_LABELS[action]: {
                    "terminated": bool(candidate_results_by_first_action[action]["terminated"]),
                    "truncated": bool(candidate_results_by_first_action[action]["truncated"]),
                    "winner": candidate_results_by_first_action[action]["winner"],
                    "steps_run": int(candidate_results_by_first_action[action]["steps_run"]),
                }
                for action in range(len(ACTION_LABELS))
            },
            "candidate_sequence_returns_by_label": {
                _sequence_label(sequence): float(
                    candidate_sequence_results[sequence]["target_return"]
                )
                for sequence in candidate_sequences
            },
            "candidate_sequence_score_delta_returns_by_label": {
                _sequence_label(sequence): float(
                    candidate_sequence_results[sequence]["score_delta_return"]
                )
                for sequence in candidate_sequences
            },
            "candidate_sequence_loss_delay_bonuses_by_label": {
                _sequence_label(sequence): float(
                    candidate_sequence_results[sequence]["loss_delay_bonus"]
                )
                for sequence in candidate_sequences
            },
            "candidate_sequence_results": [
                _sequence_result_summary(sequence, candidate_sequence_results[sequence])
                for sequence in candidate_sequences
            ],
            "chosen_ego_sequence": _sequence_debug(chosen_sequence),
            "best_ego_sequences": [
                _sequence_debug(sequence)
                for sequence in best_sequences
            ],
            "best_action_ids": best_actions,
            "best_action_labels": [ACTION_LABELS[action] for action in best_actions],
            "return_spread": float(return_spread),
            "score_delta_return_spread": float(score_delta_return_spread),
            "target_return_kind": str(chosen["target_return_kind"]),
            "loss_delay_alpha": loss_delay_alpha,
            "tie_break_policy": tie_break_policy,
            "rollout_after_first_step": (
                "track_ball"
                if ego_sequence_depth == 1
                else "continue_forced_ego_sequence_then_track_ball"
            ),
            "rollout_after_forced_steps": "track_ball",
        },
        "lookahead_terminated": bool(chosen["terminated"]),
        "lookahead_truncated": bool(chosen["truncated"]),
        "terminated": bool(chosen["terminated"]),
        "truncated": bool(chosen["truncated"]),
        "lookahead_winner": chosen["winner"],
    }


def _evaluate_candidate_sequence(
    *,
    env: PongEnv,
    config: PongConfig,
    ego_agent: str,
    ego_action_sequence: tuple[int, ...],
    seed: int,
    lookahead_steps: int,
    loss_delay_alpha: float,
) -> dict[str, Any]:
    clone = _clone_env(env, config)
    opponent_agent = _opponent(ego_agent)
    forced_opponent_policy = TrackBallPolicy()
    forced_opponent_policy.reset(seed, opponent_agent)

    first_step = None
    first_step_opponent_action = None
    next_raster_grid = None
    next_observation = None
    final_step = None
    score_delta_return = 0.0
    steps_run = 0

    for candidate_action in ego_action_sequence:
        if steps_run >= lookahead_steps:
            break
        observations = clone.observations()
        raster_observation = clone.raster_observation()
        opponent_action = int(
            forced_opponent_policy.action(
                observations[opponent_agent],
                raster_observation,
                opponent_agent,
            )
        )
        joint_action = {
            ego_agent: candidate_action,
            opponent_agent: opponent_action,
        }
        step = clone.step(joint_action)
        score_delta_return += float(step.rewards[ego_agent])
        steps_run += 1
        if first_step is None:
            first_step = step
            first_step_opponent_action = opponent_action
            next_raster_grid = _raster_grid(clone.raster_observation())
            next_observation = asdict(step.observations[ego_agent])
        final_step = step
        if step.terminated or step.truncated:
            break

    if first_step is None or final_step is None:
        raise RuntimeError("candidate sequence must contain at least one ego action")

    rollout_policies = {agent: TrackBallPolicy() for agent in AGENTS}
    for agent, policy in rollout_policies.items():
        policy.reset(seed, agent)

    while steps_run < lookahead_steps and not (final_step.terminated or final_step.truncated):
        observations = clone.observations()
        raster_observation = clone.raster_observation()
        rollout_action = {
            agent: int(rollout_policies[agent].action(observations[agent], raster_observation, agent))
            for agent in AGENTS
        }
        final_step = clone.step(rollout_action)
        score_delta_return += float(final_step.rewards[ego_agent])
        steps_run += 1

    loss_delay_bonus = _loss_delay_bonus(
        score_delta_return=score_delta_return,
        steps_run=steps_run,
        lookahead_steps=lookahead_steps,
        loss_delay_alpha=loss_delay_alpha,
    )
    target_return = score_delta_return + loss_delay_bonus
    return {
        "target_return": target_return,
        "target_return_kind": (
            "score_delta_plus_loss_delay"
            if loss_delay_alpha > 0.0
            else "score_delta"
        ),
        "score_delta_return": score_delta_return,
        "loss_delay_bonus": loss_delay_bonus,
        "loss_delay_alpha": loss_delay_alpha,
        "first_step_reward": float(first_step.rewards[ego_agent]),
        "first_step_opponent_action": int(first_step_opponent_action),
        "first_step_terminated": first_step.terminated,
        "first_step_truncated": first_step.truncated,
        "next_raster_grid": next_raster_grid,
        "next_observation": next_observation,
        "terminated": final_step.terminated,
        "truncated": final_step.truncated,
        "winner": final_step.infos["winner"] if final_step.terminated else None,
        "steps_run": steps_run,
    }


def _loss_delay_bonus(
    *,
    score_delta_return: float,
    steps_run: int,
    lookahead_steps: int,
    loss_delay_alpha: float,
) -> float:
    if loss_delay_alpha == 0.0 or score_delta_return >= 0.0:
        return 0.0
    return loss_delay_alpha * (steps_run / lookahead_steps)


def _choose_target_action(
    *,
    candidate_results: dict[int, dict[str, Any]],
    tie_break_action: int,
    tie_break_policy: str,
) -> tuple[int, str, list[int]]:
    returns = {
        action: float(result["target_return"])
        for action, result in candidate_results.items()
    }
    max_return = max(returns.values())
    min_return = min(returns.values())
    best_actions = [action for action in sorted(returns) if returns[action] == max_return]
    if max_return == min_return:
        return tie_break_action, f"all_actions_tied_{tie_break_policy}", best_actions
    if len(best_actions) == 1:
        return best_actions[0], "unique_best_return", best_actions
    if tie_break_action in best_actions:
        return tie_break_action, f"best_return_tie_break_{tie_break_policy}", best_actions
    return best_actions[0], "best_return_tie_break_lowest_action", best_actions


def _choose_target_sequence(
    *,
    candidate_results: dict[tuple[int, ...], dict[str, Any]],
    tie_break_action: int,
    tie_break_policy: str,
) -> tuple[tuple[int, ...], str, list[tuple[int, ...]]]:
    returns = {
        sequence: float(result["target_return"])
        for sequence, result in candidate_results.items()
    }
    max_return = max(returns.values())
    min_return = min(returns.values())
    best_sequences = [
        sequence
        for sequence in sorted(returns)
        if returns[sequence] == max_return
    ]
    if max_return == min_return:
        return (
            _tie_break_sequence(best_sequences, tie_break_action),
            f"all_sequences_tied_{tie_break_policy}",
            best_sequences,
        )
    if len(best_sequences) == 1:
        return best_sequences[0], "unique_best_sequence_return", best_sequences
    tie_break_sequences = [
        sequence
        for sequence in best_sequences
        if sequence[0] == tie_break_action
    ]
    if tie_break_sequences:
        return (
            sorted(tie_break_sequences)[0],
            f"best_sequence_return_tie_break_{tie_break_policy}",
            best_sequences,
        )
    return best_sequences[0], "best_sequence_return_tie_break_lowest_sequence", best_sequences


def _tie_break_sequence(
    sequences: list[tuple[int, ...]],
    tie_break_action: int,
) -> tuple[int, ...]:
    first_action_matches = [
        sequence
        for sequence in sequences
        if sequence[0] == tie_break_action
    ]
    if first_action_matches:
        return sorted(first_action_matches)[0]
    return sorted(sequences)[0]


def _best_results_by_first_action(
    candidate_results: dict[tuple[int, ...], dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    best_by_action: dict[int, tuple[tuple[int, ...], dict[str, Any]]] = {}
    for sequence, result in sorted(candidate_results.items()):
        first_action = sequence[0]
        existing = best_by_action.get(first_action)
        if existing is None or float(result["target_return"]) > float(existing[1]["target_return"]):
            best_by_action[first_action] = (sequence, result)
    return {
        action: result
        for action, (_, result) in best_by_action.items()
    }


def _tie_break_action(
    *,
    policy_id: str,
    observation: PongObservation,
    raster_observation: np.ndarray,
    ego_agent: str,
    seed: int,
) -> int:
    policy = _make_policy(policy_id, seed=seed)
    policy.reset(seed, ego_agent)
    return int(policy.action(observation, raster_observation, ego_agent))


def _target_policy_id(
    *,
    lookahead_steps: int,
    ego_sequence_depth: int,
    include_ties: bool,
    tie_break_policy: str,
    loss_delay_alpha: float,
) -> str:
    tie_part = f"include_ties_{tie_break_policy}" if include_ties else "strict"
    depth_part = "" if ego_sequence_depth == 1 else f"_ego_d{ego_sequence_depth}"
    return (
        f"lookahead_{_shaping_id(loss_delay_alpha)}_h{lookahead_steps}{depth_part}"
        f"_vs_track_ball_{tie_part}"
    )


def _target_schema_id(loss_delay_alpha: float) -> str:
    if loss_delay_alpha == 0.0:
        return LOOKAHEAD_TARGET_SCHEMA_ID
    return LOOKAHEAD_LOSS_DELAY_TARGET_SCHEMA_ID


def _shaping_id(loss_delay_alpha: float) -> str:
    if loss_delay_alpha == 0.0:
        return "score_delta"
    return f"score_delta_plus_loss_delay_{_float_id(loss_delay_alpha)}"


def _float_id(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def _candidate_sequences(ego_sequence_depth: int) -> list[tuple[int, ...]]:
    return list(product(range(len(ACTION_LABELS)), repeat=ego_sequence_depth))


def _sequence_label(sequence: tuple[int, ...]) -> str:
    return "/".join(_sequence_labels(sequence))


def _sequence_labels(sequence: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(ACTION_LABELS[action] for action in sequence)


def _sequence_debug(sequence: tuple[int, ...]) -> dict[str, object]:
    return {
        "action_ids": list(sequence),
        "action_labels": list(_sequence_labels(sequence)),
        "label": _sequence_label(sequence),
    }


def _sequence_result_summary(
    sequence: tuple[int, ...],
    result: dict[str, Any],
) -> dict[str, object]:
    return {
        **_sequence_debug(sequence),
        "target_return": float(result["target_return"]),
        "score_delta_return": float(result["score_delta_return"]),
        "loss_delay_bonus": float(result["loss_delay_bonus"]),
        "first_step_reward": float(result["first_step_reward"]),
        "terminated": bool(result["terminated"]),
        "truncated": bool(result["truncated"]),
        "winner": result["winner"],
        "steps_run": int(result["steps_run"]),
    }


def _target_return_description(loss_delay_alpha: float) -> str:
    if loss_delay_alpha == 0.0:
        return "score_delta_return"
    return (
        "score_delta_return + "
        f"{loss_delay_alpha:g} * steps_run / lookahead_steps, only when score_delta_return < 0"
    )


def _clone_env(env: PongEnv, config: PongConfig) -> PongEnv:
    # The toy probes use private env fields to avoid adding clone API surface yet.
    clone = PongEnv(config)
    clone._paddle_y = dict(env._paddle_y)
    clone._ball_x = int(env._ball_x)
    clone._ball_y = int(env._ball_y)
    clone._ball_vx = int(env._ball_vx)
    clone._ball_vy = int(env._ball_vy)
    clone._step = int(env._step)
    clone._done = bool(env._done)
    clone._last_hit = env._last_hit
    clone._last_hit_impact = (
        None if env._last_hit_impact is None else dict(env._last_hit_impact)
    )
    clone._score_agent = env._score_agent
    return clone


def _make_policy(policy_id: str, *, seed: int) -> BaselinePolicy:
    if policy_id == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    if policy_id == "track_ball":
        return TrackBallPolicy()
    if policy_id == "angle_control":
        return AngleControlPolicy()
    raise ValueError(f"unknown policy_id {policy_id!r}")


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, object]:
    target_action_histogram = _action_histogram(rows, "target_action_label")
    collector_action_histogram = _action_histogram(rows, "collector_action_label")
    chosen_sequence_histogram = _sequence_histogram(rows)
    target_source_counts = Counter(str(row["target_action_source"]) for row in rows)
    target_return_values = np.asarray(
        [float(row.get("target_return", row["target_score_delta_return"])) for row in rows],
        dtype=np.float64,
    )
    score_delta_return_values = np.asarray(
        [float(row["target_score_delta_return"]) for row in rows],
        dtype=np.float64,
    )
    return_spreads = np.asarray(
        [float(row["lookahead"]["return_spread"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "rows": len(rows),
        "target_action_histogram_by_agent": target_action_histogram,
        "collector_action_histogram_by_agent": collector_action_histogram,
        "chosen_ego_sequence_histogram_by_agent": chosen_sequence_histogram,
        "target_action_source_counts": dict(sorted(target_source_counts.items())),
        "target_differs_from_track_ball_rows": int(
            sum(row["target_action_id"] != row["track_ball_action_id"] for row in rows)
        ),
        "target_differs_from_collector_rows": int(
            sum(row["target_action_id"] != row["collector_action_id"] for row in rows)
        ),
        "positive_return_rows": int(np.sum(target_return_values > 0.0)) if len(rows) else 0,
        "negative_return_rows": int(np.sum(target_return_values < 0.0)) if len(rows) else 0,
        "zero_return_rows": int(np.sum(target_return_values == 0.0)) if len(rows) else 0,
        "positive_score_delta_return_rows": (
            int(np.sum(score_delta_return_values > 0.0)) if len(rows) else 0
        ),
        "negative_score_delta_return_rows": (
            int(np.sum(score_delta_return_values < 0.0)) if len(rows) else 0
        ),
        "zero_score_delta_return_rows": (
            int(np.sum(score_delta_return_values == 0.0)) if len(rows) else 0
        ),
        "target_return_stats": _stats(target_return_values),
        "target_score_delta_return_stats": _stats(score_delta_return_values),
        "return_spread_stats": _stats(return_spreads),
        "return_spread_histogram": dict(
            sorted(Counter(str(float(value)) for value in return_spreads).items())
        ),
        "candidate_sequence_target_return_stats_by_label": (
            _candidate_sequence_stats(rows, "candidate_sequence_returns_by_label")
        ),
        "candidate_sequence_score_delta_return_stats_by_label": (
            _candidate_sequence_stats(rows, "candidate_sequence_score_delta_returns_by_label")
        ),
    }


def _action_histogram(rows: list[dict[str, Any]], label_field: str) -> dict[str, dict[str, int]]:
    histograms: dict[str, Counter[str]] = {agent: Counter() for agent in AGENTS}
    for row in rows:
        histograms[row["ego_agent"]][row[label_field]] += 1
    return {
        agent: {label: int(histograms[agent][label]) for label in ACTION_LABELS}
        for agent in AGENTS
    }


def _sequence_histogram(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    labels = sorted(
        {
            str(row["chosen_ego_sequence_label"])
            for row in rows
        }
    )
    histograms: dict[str, Counter[str]] = {agent: Counter() for agent in AGENTS}
    for row in rows:
        histograms[row["ego_agent"]][str(row["chosen_ego_sequence_label"])] += 1
    return {
        agent: {label: int(histograms[agent][label]) for label in labels}
        for agent in AGENTS
    }


def _candidate_sequence_stats(
    rows: list[dict[str, Any]],
    field_name: str,
) -> dict[str, dict[str, object] | None]:
    labels = sorted(
        {
            label
            for row in rows
            for label in row["lookahead"][field_name]
        }
    )
    return {
        label: _stats(
            np.asarray(
                [
                    float(row["lookahead"][field_name][label])
                    for row in rows
                    if label in row["lookahead"][field_name]
                ],
                dtype=np.float64,
            )
        )
        for label in labels
    }


def _outcome_summary(games_summary: list[dict[str, object]]) -> dict[str, object]:
    winners = Counter(row["winner"] for row in games_summary)
    return {
        "wins_by_agent": {agent: int(winners[agent]) for agent in AGENTS},
        "draws_or_truncations": int(winners[None]),
        "mean_steps": float(np.mean([row["steps"] for row in games_summary])),
        "truncations": int(sum(bool(row["truncated"]) for row in games_summary)),
        "emitted_rows": int(sum(int(row["emitted_rows"]) for row in games_summary)),
        "sampled_states": int(sum(int(row["sampled_states"]) for row in games_summary)),
        "filtered_tie_states": int(sum(int(row["filtered_tie_states"]) for row in games_summary)),
    }


def _stats(values: np.ndarray) -> dict[str, object] | None:
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_count": int(np.sum(values > 0.0)),
        "zero_count": int(np.sum(values == 0.0)),
        "negative_count": int(np.sum(values < 0.0)),
    }


def _raster_grid(grid: np.ndarray) -> list[str]:
    return ["".join(str(int(cell)) for cell in row) for row in grid.tolist()]


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
