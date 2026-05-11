"""Append closed-loop lag-1 Pong states with exact-DP action labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
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
from curvyzero.training.dummy_pong_imitation_train import DummyPongImitationPolicy
from curvyzero.training.dummy_pong_lookahead_replay import LOOKAHEAD_REPLAY_ROW_SCHEMA_ID

from build_dummy_pong_lag1_trace_replay import TRACE_TARGET_POLICY_ID
from build_dummy_pong_lag1_trace_replay import _env_from_state
from build_dummy_pong_lag1_trace_replay import _raster_grid
from build_dummy_pong_lag1_trace_replay import _state_from_env
from probe_dummy_pong_target_ladder import LadderState
from probe_dummy_pong_target_ladder import TargetSpec
from probe_dummy_pong_target_ladder import _initial_opponent_memory
from probe_dummy_pong_target_ladder import _opponent_action
from probe_dummy_pong_target_ladder import _update_opponent_memory
from probe_dummy_pong_track_ball_beatable import TransitionResult
from probe_dummy_pong_track_ball_beatable import _opponent
from probe_dummy_pong_track_ball_beatable import _step_state

DAGGER_REPLAY_SUMMARY_SCHEMA_ID = "dummy_pong_lag1_dagger_replay_summary_v0"
DAGGER_COLLECTOR_POLICY_ID = "closed_loop_visual_policy_lag1_dagger_v0"
DAGGER_REPLAY_SEMANTICS_ID = "dummy_pong_closed_loop_exact_dp_lag1_relabel_v0"


def build_dummy_pong_lag1_dagger_replay(
    *,
    source_replay_path: Path,
    checkpoint_path: Path | None = None,
    checkpoint_paths: list[Path] | None = None,
    output_dir: Path | None = None,
    episodes_per_seating: int = 2,
    seed_count: int = 1,
    rollouts_per_seed: int = 1,
    seed: int = 0,
    max_steps: int = 120,
    max_rows: int | None = None,
    ego_agents: list[str] | None = None,
    exploration_epsilon: float = 0.0,
    max_unlabelable_examples: int = 20,
) -> dict[str, object]:
    """Roll out a visual policy, exact-label visited states, and append replay."""

    if episodes_per_seating < 1:
        raise ValueError("episodes_per_seating must be at least 1")
    if seed_count < 1:
        raise ValueError("seed_count must be at least 1")
    if rollouts_per_seed < 1:
        raise ValueError("rollouts_per_seed must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be at least 1 when set")
    if not 0.0 <= exploration_epsilon <= 1.0:
        raise ValueError("exploration_epsilon must be in [0, 1]")
    if max_unlabelable_examples < 0:
        raise ValueError("max_unlabelable_examples must be non-negative")
    checkpoint_paths = _normalize_checkpoint_paths(
        checkpoint_path=checkpoint_path,
        checkpoint_paths=checkpoint_paths,
    )
    ego_agents = _normalize_ego_agents(ego_agents)

    source_rows_path = _resolve_replay_rows_path(source_replay_path)
    source_rows = _load_rows(source_rows_path)
    config = PongConfig(max_steps=max_steps)
    spec = TargetSpec(
        target_id="lag1_track_ball_normal",
        description="Default geometry; opponent tracks the previous ball row.",
        config=config,
        opponent_policy="lagged_track_ball_1",
    )
    policies = [
        (path, DummyPongImitationPolicy.load_checkpoint(path))
        for path in checkpoint_paths
    ]
    for path, policy in policies:
        if policy.raster_shape != (config.height, config.width):
            raise ValueError(
                f"checkpoint {path} raster_shape {policy.raster_shape!r} "
                f"does not match {(config.height, config.width)!r}"
            )

    dagger_rows: list[dict[str, Any]] = []
    rollout_summaries = []
    unlabelable_examples: list[dict[str, Any]] = []
    for seed_index in range(seed_count):
        for rollout_index in range(rollouts_per_seed):
            for seating_index, ego_agent in enumerate(ego_agents):
                for episode_index in range(episodes_per_seating):
                    for checkpoint_index, (policy_checkpoint_path, policy) in enumerate(policies):
                        if max_rows is not None and len(dagger_rows) >= max_rows:
                            break
                        episode_seed = (
                            seed
                            + seed_index * 1_000_000
                            + seating_index * 100_000
                            + episode_index
                        )
                        behavior_seed = (
                            episode_seed
                            + checkpoint_index * 10_000_000
                            + rollout_index * 10_000
                        )
                        rollout_summaries.append(
                            _collect_episode_rows(
                                config=config,
                                spec=spec,
                                policy=policy,
                                checkpoint_index=checkpoint_index,
                                checkpoint_path=policy_checkpoint_path,
                                ego_agent=ego_agent,
                                episode_index=episode_index,
                                episode_seed=episode_seed,
                                seed_index=seed_index,
                                rollout_index=rollout_index,
                                behavior_seed=behavior_seed,
                                exploration_epsilon=exploration_epsilon,
                                dagger_rows=dagger_rows,
                                max_rows=max_rows,
                                unlabelable_examples=unlabelable_examples,
                                max_unlabelable_examples=max_unlabelable_examples,
                            )
                        )
                    if max_rows is not None and len(dagger_rows) >= max_rows:
                        break
                if max_rows is not None and len(dagger_rows) >= max_rows:
                    break
            if max_rows is not None and len(dagger_rows) >= max_rows:
                break

    rows = source_rows + dagger_rows
    rollout_summary = _summarize_rollouts(rollout_summaries)
    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_lag1_dagger_replay",
        "summary_schema_id": DAGGER_REPLAY_SUMMARY_SCHEMA_ID,
        "row_schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
        "target_policy_id": TRACE_TARGET_POLICY_ID,
        "collector_policy_id": DAGGER_COLLECTOR_POLICY_ID,
        "replay_semantics_id": DAGGER_REPLAY_SEMANTICS_ID,
        "note": (
            "Tiny DAgger-style replay: a learned visual checkpoint is rolled out "
            "against lagged_track_ball_1, visited raster states are relabeled by "
            "the exact DP teacher from the current lagged-opponent memory, and "
            "the rows are appended to an existing supervised replay."
        ),
        "config": asdict(config),
        "seed": seed,
        "seed_count": seed_count,
        "rollouts_per_seed": rollouts_per_seed,
        "episodes_per_seating": episodes_per_seating,
        "ego_agents": ego_agents,
        "exploration_epsilon": exploration_epsilon,
        "max_rows": max_rows,
        "max_unlabelable_examples": max_unlabelable_examples,
        "source_replay_path": str(source_rows_path),
        "source_rows": len(source_rows),
        "dagger_rows": len(dagger_rows),
        "total_rows": len(rows),
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "checkpoint_path": str(checkpoint_paths[0]),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "replay_row_schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
        },
        "raster": {
            "shape": [config.height, config.width],
            "encoding": "row-major digit strings over the raster legend",
            "legend": RASTER_LEGEND,
        },
        "action_labels": list(ACTION_LABELS),
        "source_data": _summarize_rows(source_rows),
        "dagger_data": _summarize_rows(dagger_rows),
        "combined_data": _summarize_rows(rows),
        "closed_loop_collection": rollout_summary,
        "unlabelable_states": {
            "count": rollout_summary["skipped_unlabelable_states"],
            "examples": unlabelable_examples,
        },
        "rollouts": rollout_summaries,
        "plain_language": {
            "proves": (
                "The visual policy's own closed-loop states can be collected and "
                "turned into exact-DP supervised labels without changing the trainer."
            ),
            "does_not_prove": (
                "That one small aggregation is enough to beat the CEM-v2 baseline. "
                "This is a minimal coverage repair for off-trace drift."
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


def _collect_episode_rows(
    *,
    config: PongConfig,
    spec: TargetSpec,
    policy: DummyPongImitationPolicy,
    checkpoint_index: int,
    checkpoint_path: Path,
    ego_agent: str,
    episode_index: int,
    episode_seed: int,
    seed_index: int,
    rollout_index: int,
    behavior_seed: int,
    exploration_epsilon: float,
    dagger_rows: list[dict[str, Any]],
    max_rows: int | None,
    unlabelable_examples: list[dict[str, Any]],
    max_unlabelable_examples: int,
) -> dict[str, object]:
    env = PongEnv(config)
    observations = env.reset(seed=episode_seed)
    opponent_agent = _opponent(ego_agent)
    ladder_state = LadderState(
        core=_state_from_env(env),
        opponent_memory=_initial_opponent_memory(spec, _state_from_env(env)),
    )
    match_id = (
        "dagger"
        f"_ckpt{checkpoint_index}"
        f"_{ego_agent}_vs_lagged_track_ball_1"
        f"_seed_{episode_seed}"
        f"_rollout_{rollout_index}"
    )
    rng = random.Random(behavior_seed)
    teacher_action_counts = Counter()
    behavior_action_counts = Counter()
    exploration_action_counts = Counter()
    added_rows = 0
    skipped_unlabelable = 0

    while True:
        if max_rows is not None and len(dagger_rows) >= max_rows:
            break
        raster_array = env.raster_observation()
        raster_grid = _raster_grid(raster_array)
        policy_action = policy.predict_action_id(raster_array, ego_agent)
        behavior_action, used_exploration = _behavior_action(
            policy_action=policy_action,
            action_count=config.action_count,
            rng=rng,
            exploration_epsilon=exploration_epsilon,
        )
        behavior_action_counts[ACTION_LABELS[behavior_action]] += 1
        if used_exploration:
            exploration_action_counts[ACTION_LABELS[behavior_action]] += 1
        teacher_actions = _find_winning_actions_from_ladder_state(
            spec=spec,
            initial_state=ladder_state,
            ego_agent=ego_agent,
        )
        if teacher_actions is None:
            skipped_unlabelable += 1
            _record_unlabelable_example(
                examples=unlabelable_examples,
                max_examples=max_unlabelable_examples,
                match_id=match_id,
                checkpoint_index=checkpoint_index,
                checkpoint_path=checkpoint_path,
                ego_agent=ego_agent,
                opponent_agent=opponent_agent,
                episode_index=episode_index,
                episode_seed=episode_seed,
                seed_index=seed_index,
                rollout_index=rollout_index,
                behavior_seed=behavior_seed,
                step_index=int(ladder_state.core.step),
                ladder_state=ladder_state,
                policy_action=policy_action,
                behavior_action=behavior_action,
                used_exploration=used_exploration,
            )
        else:
            teacher_action = int(teacher_actions[0])
            dagger_rows.append(
                _dagger_row(
                    config=config,
                    spec=spec,
                    checkpoint_path=checkpoint_path,
                    match_id=match_id,
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    ego_agent=ego_agent,
                    opponent_agent=opponent_agent,
                    ladder_state=ladder_state,
                    raster_grid=raster_grid,
                    observation=asdict(observations[ego_agent]),
                    checkpoint_index=checkpoint_index,
                    seed_index=seed_index,
                    rollout_index=rollout_index,
                    behavior_seed=behavior_seed,
                    exploration_epsilon=exploration_epsilon,
                    policy_action=policy_action,
                    used_exploration=used_exploration,
                    behavior_action=behavior_action,
                    teacher_action=teacher_action,
                    teacher_trace_len=len(teacher_actions),
                    added_index=added_rows,
                )
            )
            teacher_action_counts[ACTION_LABELS[teacher_action]] += 1
            added_rows += 1

        opponent_action = _opponent_action(spec, ladder_state, opponent_agent)
        step = env.step({ego_agent: behavior_action, opponent_agent: opponent_action})
        previous_ladder_state = ladder_state
        ladder_state = LadderState(
            core=_state_from_env(env),
            opponent_memory=_update_opponent_memory(spec, previous_ladder_state),
        )
        observations = step.observations
        if step.terminated or step.truncated:
            return {
                "match_id": match_id,
                "ego_agent": ego_agent,
                "opponent_agent": opponent_agent,
                "checkpoint_index": checkpoint_index,
                "checkpoint_path": str(checkpoint_path),
                "episode_seed": episode_seed,
                "seed_index": seed_index,
                "rollout_index": rollout_index,
                "behavior_seed": behavior_seed,
                "exploration_epsilon": exploration_epsilon,
                "added_rows": added_rows,
                "skipped_unlabelable_states": skipped_unlabelable,
                "behavior_winner": step.infos["winner"] if step.terminated else None,
                "behavior_truncated": bool(step.truncated),
                "behavior_steps": int(ladder_state.core.step),
                "behavior_action_histogram": _action_histogram(behavior_action_counts),
                "exploration_action_histogram": _action_histogram(exploration_action_counts),
                "teacher_action_histogram": {
                    label: int(teacher_action_counts[label])
                    for label in ACTION_LABELS
                },
            }

    return {
        "match_id": match_id,
        "ego_agent": ego_agent,
        "opponent_agent": opponent_agent,
        "checkpoint_index": checkpoint_index,
        "checkpoint_path": str(checkpoint_path),
        "episode_seed": episode_seed,
        "seed_index": seed_index,
        "rollout_index": rollout_index,
        "behavior_seed": behavior_seed,
        "exploration_epsilon": exploration_epsilon,
        "added_rows": added_rows,
        "skipped_unlabelable_states": skipped_unlabelable,
        "stopped_by_max_rows": True,
        "behavior_steps": int(ladder_state.core.step),
        "behavior_action_histogram": _action_histogram(behavior_action_counts),
        "exploration_action_histogram": _action_histogram(exploration_action_counts),
        "teacher_action_histogram": {
            label: int(teacher_action_counts[label])
            for label in ACTION_LABELS
        },
    }


def _dagger_row(
    *,
    config: PongConfig,
    spec: TargetSpec,
    checkpoint_path: Path,
    match_id: str,
    episode_index: int,
    episode_seed: int,
    ego_agent: str,
    opponent_agent: str,
    ladder_state: LadderState,
    raster_grid: list[str],
    observation: dict[str, Any],
    checkpoint_index: int,
    seed_index: int,
    rollout_index: int,
    behavior_seed: int,
    exploration_epsilon: float,
    policy_action: int,
    used_exploration: bool,
    behavior_action: int,
    teacher_action: int,
    teacher_trace_len: int,
    added_index: int,
) -> dict[str, Any]:
    opponent_action = _opponent_action(spec, ladder_state, opponent_agent)
    transition = _step_state(
        state=ladder_state.core,
        actions={ego_agent: teacher_action, opponent_agent: opponent_action},
        config=config,
    )
    next_env = _env_from_state(config, transition.state)
    reward = _reward_for_transition(transition, ego_agent)
    return {
        "schema_id": LOOKAHEAD_REPLAY_ROW_SCHEMA_ID,
        "run_id": "dummy-pong-lag1-dagger-replay",
        "target_policy_id": TRACE_TARGET_POLICY_ID,
        "target_schema_id": DAGGER_REPLAY_SUMMARY_SCHEMA_ID,
        "replay_semantics_id": DAGGER_REPLAY_SEMANTICS_ID,
        "match_id": match_id,
        "reset_state_id": f"closed_loop_seed_{episode_seed}",
        "reset_state": asdict(ladder_state.core),
        "episode_index": episode_index,
        "episode_seed": episode_seed,
        "checkpoint_index": checkpoint_index,
        "seed_index": seed_index,
        "rollout_index": rollout_index,
        "behavior_seed": behavior_seed,
        "dagger_row_index": added_index,
        "step_index": int(ladder_state.core.step),
        "ego_agent": ego_agent,
        "opponent_agent": opponent_agent,
        "collector_policy_id": DAGGER_COLLECTOR_POLICY_ID,
        "behavior_policy_id": f"learned_visual_checkpoint:{checkpoint_path}",
        "behavior_exploration_epsilon": float(exploration_epsilon),
        "closed_loop_policy_action_id": int(policy_action),
        "closed_loop_policy_action_label": ACTION_LABELS[int(policy_action)],
        "closed_loop_behavior_action_id": int(behavior_action),
        "closed_loop_behavior_action_label": ACTION_LABELS[int(behavior_action)],
        "closed_loop_behavior_used_exploration": bool(used_exploration),
        "opponent_policy_id": "lagged_track_ball_1",
        "opponent_memory": list(ladder_state.opponent_memory),
        "teacher_trace_remaining_steps": teacher_trace_len,
        "target_action_id": int(teacher_action),
        "target_action_label": ACTION_LABELS[int(teacher_action)],
        "joint_action_by_agent": {
            ego_agent: {
                "action_id": int(teacher_action),
                "label": ACTION_LABELS[int(teacher_action)],
            },
            opponent_agent: {
                "action_id": int(opponent_action),
                "label": ACTION_LABELS[int(opponent_action)],
            },
        },
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "raster_shape": [config.height, config.width],
        "raster_grid": raster_grid,
        "observation": observation,
        "reward_after_step": reward,
        "next_raster_grid": _raster_grid(next_env.raster_observation()),
        "next_observation": asdict(next_env.observations()[ego_agent]),
        "terminated_after_step": bool(transition.winner is not None),
        "truncated_after_step": bool(transition.truncated),
        "terminated": bool(transition.winner is not None),
        "truncated": bool(transition.truncated),
        "winner_after_step": transition.winner,
    }


def _find_winning_actions_from_ladder_state(
    *,
    spec: TargetSpec,
    initial_state: LadderState,
    ego_agent: str,
) -> list[int] | None:
    memo: dict[LadderState, list[int] | None] = {}
    opponent_agent = _opponent(ego_agent)

    def solve(state: LadderState) -> list[int] | None:
        if state.core.step >= spec.config.max_steps:
            return None
        if state in memo:
            return memo[state]
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

    return solve(initial_state)


def _reward_for_transition(transition: TransitionResult, ego_agent: str) -> float:
    if transition.winner is None:
        return 0.0
    return 1.0 if transition.winner == ego_agent else -1.0


def _normalize_checkpoint_paths(
    *,
    checkpoint_path: Path | None,
    checkpoint_paths: list[Path] | None,
) -> list[Path]:
    paths = list(checkpoint_paths or [])
    if checkpoint_path is not None:
        paths.insert(0, checkpoint_path)
    if not paths:
        raise ValueError("at least one checkpoint path is required")
    return paths


def _normalize_ego_agents(ego_agents: list[str] | None) -> list[str]:
    if ego_agents is None or len(ego_agents) == 0:
        return list(AGENTS)
    unknown = [agent for agent in ego_agents if agent not in AGENTS]
    if unknown:
        raise ValueError(f"unknown ego agent(s): {unknown!r}; expected one of {AGENTS!r}")
    return list(dict.fromkeys(ego_agents))


def _behavior_action(
    *,
    policy_action: int,
    action_count: int,
    rng: random.Random,
    exploration_epsilon: float,
) -> tuple[int, bool]:
    if exploration_epsilon <= 0.0 or rng.random() >= exploration_epsilon:
        return int(policy_action), False
    return int(rng.randrange(action_count)), True


def _record_unlabelable_example(
    *,
    examples: list[dict[str, Any]],
    max_examples: int,
    match_id: str,
    checkpoint_index: int,
    checkpoint_path: Path,
    ego_agent: str,
    opponent_agent: str,
    episode_index: int,
    episode_seed: int,
    seed_index: int,
    rollout_index: int,
    behavior_seed: int,
    step_index: int,
    ladder_state: LadderState,
    policy_action: int,
    behavior_action: int,
    used_exploration: bool,
) -> None:
    if len(examples) >= max_examples:
        return
    examples.append(
        {
            "match_id": match_id,
            "checkpoint_index": checkpoint_index,
            "checkpoint_path": str(checkpoint_path),
            "ego_agent": ego_agent,
            "opponent_agent": opponent_agent,
            "episode_index": episode_index,
            "episode_seed": episode_seed,
            "seed_index": seed_index,
            "rollout_index": rollout_index,
            "behavior_seed": behavior_seed,
            "step_index": step_index,
            "state": asdict(ladder_state.core),
            "opponent_memory": list(ladder_state.opponent_memory),
            "closed_loop_policy_action_id": int(policy_action),
            "closed_loop_policy_action_label": ACTION_LABELS[int(policy_action)],
            "closed_loop_behavior_action_id": int(behavior_action),
            "closed_loop_behavior_action_label": ACTION_LABELS[int(behavior_action)],
            "closed_loop_behavior_used_exploration": bool(used_exploration),
        }
    )


def _action_histogram(counts: Counter[str]) -> dict[str, int]:
    return {label: int(counts[label]) for label in ACTION_LABELS}


def _summarize_rollouts(rollouts: list[dict[str, object]]) -> dict[str, object]:
    teacher_counts = Counter()
    behavior_counts = Counter()
    exploration_counts = Counter()
    behavior_winners = Counter()
    added_rows = 0
    skipped_unlabelable = 0
    truncated = 0
    stopped_by_max_rows = 0
    for rollout in rollouts:
        added_rows += int(rollout.get("added_rows", 0))
        skipped_unlabelable += int(rollout.get("skipped_unlabelable_states", 0))
        if bool(rollout.get("behavior_truncated", False)):
            truncated += 1
        if bool(rollout.get("stopped_by_max_rows", False)):
            stopped_by_max_rows += 1
        winner = rollout.get("behavior_winner")
        behavior_winners[str(winner) if winner is not None else "none"] += 1
        for label, count in dict(rollout.get("teacher_action_histogram", {})).items():
            teacher_counts[str(label)] += int(count)
        for label, count in dict(rollout.get("behavior_action_histogram", {})).items():
            behavior_counts[str(label)] += int(count)
        for label, count in dict(rollout.get("exploration_action_histogram", {})).items():
            exploration_counts[str(label)] += int(count)
    return {
        "rollouts": len(rollouts),
        "added_rows": added_rows,
        "skipped_unlabelable_states": skipped_unlabelable,
        "behavior_truncations": truncated,
        "stopped_by_max_rows_rollouts": stopped_by_max_rows,
        "behavior_winners": {
            winner: int(behavior_winners[winner])
            for winner in sorted(behavior_winners)
        },
        "teacher_action_histogram": _action_histogram(teacher_counts),
        "behavior_action_histogram": _action_histogram(behavior_counts),
        "exploration_action_histogram": _action_histogram(exploration_counts),
    }


def _resolve_replay_rows_path(replay_path: Path) -> Path:
    if replay_path.is_dir():
        replay_path = replay_path / "replay_rows.jsonl"
    if not replay_path.exists():
        raise FileNotFoundError(f"replay rows not found: {replay_path}")
    return replay_path


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    if not rows:
        raise ValueError(f"replay rows file is empty: {path}")
    return rows


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, object]:
    histogram = {agent: Counter() for agent in AGENTS}
    for row in rows:
        histogram[str(row["ego_agent"])][str(row["target_action_label"])] += 1
    return {
        "rows": len(rows),
        "target_action_histogram_by_agent": {
            agent: {label: int(histogram[agent][label]) for label in ACTION_LABELS}
            for agent in AGENTS
        },
        "positive_reward_rows": sum(1 for row in rows if float(row["reward_after_step"]) > 0.0),
        "negative_reward_rows": sum(1 for row in rows if float(row["reward_after_step"]) < 0.0),
        "terminated_rows": sum(1 for row in rows if bool(row.get("terminated_after_step", False))),
        "truncated_rows": sum(1 for row in rows if bool(row.get("truncated_after_step", False))),
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
    parser.add_argument(
        "--source-replay-path",
        type=Path,
        required=True,
        help="Path to replay_rows.jsonl or a replay directory containing it.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        action="append",
        required=True,
        default=[],
        help="Checkpoint to roll out. May be repeated to aggregate multiple behavior policies.",
    )
    parser.add_argument("--episodes-per-seating", type=int, default=2)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="Number of base seed blocks to collect for each checkpoint/seat.",
    )
    parser.add_argument(
        "--rollouts-per-seed",
        type=int,
        default=1,
        help=(
            "Repeat each env seed with a fresh behavior RNG. Useful with "
            "--exploration-epsilon to visit different closed-loop branches."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--ego-agent",
        choices=AGENTS,
        action="append",
        default=[],
        help="Ego seat to collect. May be repeated; defaults to both seats.",
    )
    parser.add_argument(
        "--exploration-epsilon",
        type=float,
        default=0.0,
        help="Probability of replacing the checkpoint action with a random behavior action.",
    )
    parser.add_argument(
        "--max-unlabelable-examples",
        type=int,
        default=20,
        help="Maximum exact-DP-unlabelable visited states to store in summary.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = build_dummy_pong_lag1_dagger_replay(
        source_replay_path=args.source_replay_path,
        checkpoint_paths=args.checkpoint_path,
        output_dir=args.output_dir,
        episodes_per_seating=args.episodes_per_seating,
        seed_count=args.seed_count,
        rollouts_per_seed=args.rollouts_per_seed,
        seed=args.seed,
        max_steps=args.max_steps,
        max_rows=args.max_rows,
        ego_agents=args.ego_agent,
        exploration_epsilon=args.exploration_epsilon,
        max_unlabelable_examples=args.max_unlabelable_examples,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
