"""Fixed baseline evaluation matrix for the dummy Pong-like task."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Protocol

import numpy as np

from curvyzero.training.dummy_pong import ACTION_LABELS
from curvyzero.training.dummy_pong import ACTION_SCHEMA_ID
from curvyzero.training.dummy_pong import AGENTS
from curvyzero.training.dummy_pong import OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import RASTER_OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_pong import REWARD_SCHEMA_ID
from curvyzero.training.dummy_pong import RULESET_ID
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv
from curvyzero.training.dummy_pong import PongObservation
from curvyzero.training.lightzero_dummy_pong_policy import LIGHTZERO_POLICY_HEAD_GREEDY_LABEL
from curvyzero.training.lightzero_dummy_pong_policy import LIGHTZERO_POLICY_HEAD_GREEDY_SCHEMA_ID
from curvyzero.training.lightzero_dummy_pong_policy import LIGHTZERO_MCTS_EVAL_MODE_LABEL
from curvyzero.training.lightzero_dummy_pong_policy import LIGHTZERO_MCTS_EVAL_MODE_SCHEMA_ID
from curvyzero.training.lightzero_dummy_pong_policy import LIGHTZERO_POLICY_PREFIX
from curvyzero.training.lightzero_dummy_pong_policy import LightZeroMCTSEvalModeSpec
from curvyzero.training.lightzero_dummy_pong_policy import LightZeroPolicyHeadGreedySpec
from curvyzero.training.lightzero_dummy_pong_policy import (
    load_lightzero_mcts_eval_mode_checkpoint,
    load_lightzero_policy_head_greedy_checkpoint,
)

EVAL_CONFIG_SCHEMA_ID = "dummy_pong_eval_config_v0"
EVAL_SUMMARY_SCHEMA_ID = "dummy_pong_eval_summary_v0"
POLICY_NAMES = ("random_uniform", "lagged_track_ball_1", "track_ball")
EXTENDED_POLICY_NAMES = (*POLICY_NAMES, "stay")
LEARNED_POLICY_PREFIX = "learned:"
FEATURE_ENCODING_ID = "dummy_pong_raster_one_hot_plus_geometry_v0"
DEFAULT_FEATURE_MODE = "raster_plus_geometry"
IMITATION_POLICY_CHECKPOINT_SCHEMA_ID = "dummy_pong_imitation_policy_checkpoint_v0"
DEFAULT_LIGHTZERO_ENV = "dummy_pong_lag1"
DEFAULT_LIGHTZERO_FEATURE_MODE = "tabular_ego"
DEFAULT_LIGHTZERO_OPPONENT_POLICY = "random_uniform"
DEFAULT_LIGHTZERO_MAX_ENV_STEP = 64
DEFAULT_LIGHTZERO_NUM_SIMULATIONS = 2
DEFAULT_PONG_RESET_PROFILE = "default"
DEFAULT_PONG_RESET_PRESSURE_AGENT = "random"
LightZeroCheckpointPolicySpec = LightZeroPolicyHeadGreedySpec | LightZeroMCTSEvalModeSpec


class BaselinePolicy(Protocol):
    name: str

    def reset(self, episode_seed: int, agent: str) -> None: ...

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    match_id: str
    pair_group_id: str
    episode: int
    seed: int
    player_policy_by_agent: dict[str, str]
    steps: int
    max_steps: int
    winner: str | None
    truncated: bool
    rewards: dict[str, float]
    action_counts_by_agent: dict[str, list[int]]
    last_hit: str | None


@dataclass(frozen=True, slots=True)
class LearnedCheckpointPolicySpec:
    policy_id: str
    checkpoint_path: Path
    policy: Any
    checkpoint_metadata: dict[str, object]
    checkpoint_schema_id: str
    feature_encoding_id: str
    feature_mode: str
    frame_stack: int


class RandomUniformPolicy:
    name = "random_uniform"

    def __init__(self, seed: int):
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, episode_seed: int, agent: str) -> None:
        self._rng = np.random.default_rng(self._base_seed + episode_seed + _agent_seed(agent))

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int:
        del observation, raster_grid, agent
        return int(self._rng.integers(len(ACTION_LABELS)))


class TrackBallPolicy:
    name = "track_ball"

    def reset(self, episode_seed: int, agent: str) -> None:
        del episode_seed, agent

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int:
        del raster_grid, agent
        if observation.ball_dy_from_ego_center < 0:
            return 0
        if observation.ball_dy_from_ego_center > 0:
            return 2
        return 1


class LaggedTrackBallPolicy:
    name = "lagged_track_ball_1"

    def __init__(self, *, delay: int = 1):
        if delay < 1:
            raise ValueError("delay must be at least 1")
        self.name = f"lagged_track_ball_{delay}"
        self._delay = delay
        self._ball_y_history: list[int] = []

    def reset(self, episode_seed: int, agent: str) -> None:
        del episode_seed, agent
        self._ball_y_history = []

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int:
        del raster_grid, agent
        if len(self._ball_y_history) < self._delay:
            target_y = observation.ball_y
        else:
            target_y = self._ball_y_history[0]
        self._ball_y_history = (self._ball_y_history + [observation.ball_y])[-self._delay :]
        ego_center_y = observation.ball_y - observation.ball_dy_from_ego_center
        if target_y < ego_center_y:
            return 0
        if target_y > ego_center_y:
            return 2
        return 1


class StayPolicy:
    name = "stay"

    def reset(self, episode_seed: int, agent: str) -> None:
        del episode_seed, agent

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int:
        del observation, raster_grid, agent
        return 1


class AngleControlPolicy:
    name = "angle_control"

    def __init__(self) -> None:
        self._zero_vy_offset = 1

    def reset(self, episode_seed: int, agent: str) -> None:
        self._zero_vy_offset = -1 if (episode_seed + _agent_seed(agent)) % 2 == 0 else 1

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int:
        del agent
        target_center_y = observation.ball_y
        if observation.ball_vx_forward < 0:
            predicted_hit_y = _predict_contact_y(
                ball_y=observation.ball_y,
                ball_vy=observation.ball_vy,
                steps=max(observation.ball_dx_forward, 1),
                height=int(raster_grid.shape[0]),
            )
            desired_impact_offset = _edge_seeking_impact_offset(
                predicted_hit_y,
                height=int(raster_grid.shape[0]),
                default=self._zero_vy_offset,
            )
            target_center_y = predicted_hit_y - desired_impact_offset

        ego_center_y = observation.ball_y - observation.ball_dy_from_ego_center
        if target_center_y < ego_center_y:
            return 0
        if target_center_y > ego_center_y:
            return 2
        return 1


class LearnedCheckpointPolicy:
    def __init__(self, spec: LearnedCheckpointPolicySpec):
        self.name = spec.policy_id
        self._policy = spec.policy
        self._frame_stack = max(1, int(spec.frame_stack))
        self._raster_history_by_agent: dict[str, list[np.ndarray]] = {}

    def reset(self, episode_seed: int, agent: str) -> None:
        del episode_seed
        self._raster_history_by_agent[agent] = []

    def action(
        self,
        observation: PongObservation,
        raster_grid: np.ndarray,
        agent: str,
    ) -> int:
        del observation
        if self._frame_stack == 1:
            return self._policy.predict_action_id(raster_grid, agent)

        history = self._raster_history_by_agent.setdefault(agent, [])
        history.append(np.asarray(raster_grid, dtype=np.int64).copy())
        del history[:-self._frame_stack]
        raster_input = _pad_raster_history(history, frame_stack=self._frame_stack)
        return self._policy.predict_action_id(raster_input, agent)


def run_dummy_pong_eval(
    *,
    episodes: int,
    seed: int,
    output_dir: Path | None = None,
    checkpoint_policies: list[str | Path] | None = None,
    lightzero_checkpoint_policies: list[str | Path] | None = None,
    lightzero_mcts_checkpoint_policies: list[str | Path] | None = None,
    lightzero_env: str = DEFAULT_LIGHTZERO_ENV,
    lightzero_feature_mode: str = DEFAULT_LIGHTZERO_FEATURE_MODE,
    lightzero_opponent_policy: str = DEFAULT_LIGHTZERO_OPPONENT_POLICY,
    lightzero_max_env_step: int = DEFAULT_LIGHTZERO_MAX_ENV_STEP,
    lightzero_num_simulations: int = DEFAULT_LIGHTZERO_NUM_SIMULATIONS,
    pong_reset_profile: str = DEFAULT_PONG_RESET_PROFILE,
    pong_reset_pressure_agent: str = DEFAULT_PONG_RESET_PRESSURE_AGENT,
    paired_seats: bool = True,
    baseline_policy_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Evaluate fixed Pong baselines/checkpoints and optionally write artifacts."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    uses_lightzero_checkpoint = bool(
        lightzero_checkpoint_policies or lightzero_mcts_checkpoint_policies
    )
    config = (
        PongConfig(
            max_steps=lightzero_max_env_step,
            reset_profile=pong_reset_profile,
            reset_pressure_agent=pong_reset_pressure_agent,
        )
        if uses_lightzero_checkpoint
        else PongConfig(
            reset_profile=pong_reset_profile,
            reset_pressure_agent=pong_reset_pressure_agent,
        )
    )
    baseline_policy_names = _resolve_baseline_policy_names(baseline_policy_names)
    learned_specs = _load_learned_checkpoint_policies(checkpoint_policies or [])
    used_policy_ids = set(baseline_policy_names) | {spec.policy_id for spec in learned_specs}
    lightzero_specs = _load_lightzero_checkpoint_policies(
        lightzero_checkpoint_policies or [],
        env=lightzero_env,
        feature_mode=lightzero_feature_mode,
        opponent_policy=lightzero_opponent_policy,
        seed=seed,
        max_env_step=lightzero_max_env_step,
        used_policy_ids=used_policy_ids,
    )
    lightzero_mcts_specs = _load_lightzero_mcts_checkpoint_policies(
        lightzero_mcts_checkpoint_policies or [],
        env=lightzero_env,
        feature_mode=lightzero_feature_mode,
        opponent_policy=lightzero_opponent_policy,
        seed=seed,
        max_env_step=lightzero_max_env_step,
        num_simulations=lightzero_num_simulations,
        used_policy_ids=used_policy_ids,
    )
    checkpoint_specs_by_id = {
        **{spec.policy_id: spec for spec in learned_specs},
        **{spec.policy_id: spec for spec in lightzero_specs},
        **{spec.policy_id: spec for spec in lightzero_mcts_specs},
    }
    checkpoint_policy_ids = set(checkpoint_specs_by_id)
    all_records: list[EpisodeRecord] = []
    matchups = []
    for pair_index, (policy_a, policy_b) in enumerate(
        _policy_pairs(checkpoint_specs_by_id, baseline_policy_names=baseline_policy_names)
    ):
        pair_group_id = f"{policy_a}_vs_{policy_b}"
        pair_seed = seed + pair_index * 100_000
        seatings = _seatings_for_pair(
            policy_a,
            policy_b,
            checkpoint_policy_ids=checkpoint_policy_ids,
            baseline_policy_ids=set(baseline_policy_names),
            paired_seats=paired_seats,
        )
        for seating_index, (player_0_policy, player_1_policy) in enumerate(seatings):
            match_id = f"{player_0_policy}_p0__{player_1_policy}_p1"
            records = _run_matchup(
                config=config,
                match_id=match_id,
                pair_group_id=pair_group_id,
                player_policy_names={"player_0": player_0_policy, "player_1": player_1_policy},
                episodes=episodes,
                seed=pair_seed,
                policy_seed=seed + pair_index * 10_000 + seating_index * 1_000,
                checkpoint_specs_by_id=checkpoint_specs_by_id,
            )
            all_records.extend(records)
            matchups.append(_summarize_records(records))

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_eval",
        "note": (
            "Fixed baseline matrix plus optional supervised raster checkpoint "
            "policies for a tiny project-owned Pong-like environment."
        ),
        "eval_config_schema_id": EVAL_CONFIG_SCHEMA_ID,
        "eval_summary_schema_id": EVAL_SUMMARY_SCHEMA_ID,
        "seed": seed,
        "episodes_per_match": episodes,
        "total_episodes": len(all_records),
        "paired_seats": paired_seats,
        "eval_seating": _eval_seating_summary(paired_seats=paired_seats),
        "config": asdict(config),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
        },
        "action_labels": list(ACTION_LABELS),
        "baseline_policy_names": list(baseline_policy_names),
        "policies": _policy_specs(
            learned_specs,
            lightzero_specs + lightzero_mcts_specs,
            baseline_policy_names=baseline_policy_names,
        ),
        "checkpoint_policies": [
            _learned_policy_spec_summary(spec)
            for spec in learned_specs
        ] + [
            _lightzero_policy_spec_summary(spec)
            for spec in lightzero_specs
        ] + [
            _lightzero_policy_spec_summary(spec)
            for spec in lightzero_mcts_specs
        ],
        "matchups": matchups,
        "pair_groups": _summarize_pair_groups(all_records),
    }
    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, records=all_records)
    return summary


def _policy_pairs(
    checkpoint_specs_by_id: dict[str, LearnedCheckpointPolicySpec | LightZeroCheckpointPolicySpec],
    *,
    baseline_policy_names: tuple[str, ...],
) -> list[tuple[str, str]]:
    pairs = []
    for index, policy_a in enumerate(baseline_policy_names):
        for policy_b in baseline_policy_names[index:]:
            pairs.append((policy_a, policy_b))
    for checkpoint_policy_id in checkpoint_specs_by_id:
        for baseline_policy_id in baseline_policy_names:
            pairs.append((checkpoint_policy_id, baseline_policy_id))
    checkpoint_policy_ids = list(checkpoint_specs_by_id)
    for index, policy_a in enumerate(checkpoint_policy_ids):
        for policy_b in checkpoint_policy_ids[index + 1 :]:
            pairs.append((policy_a, policy_b))
    return pairs


def _seatings_for_pair(
    policy_a: str,
    policy_b: str,
    *,
    checkpoint_policy_ids: set[str],
    baseline_policy_ids: set[str],
    paired_seats: bool,
) -> list[tuple[str, str]]:
    if policy_a == policy_b:
        return [(policy_a, policy_b)]
    if paired_seats:
        return [(policy_a, policy_b), (policy_b, policy_a)]
    if policy_a in checkpoint_policy_ids and policy_b in baseline_policy_ids:
        return [(policy_a, policy_b)]
    if policy_b in checkpoint_policy_ids and policy_a in baseline_policy_ids:
        return [(policy_b, policy_a)]
    return [(policy_a, policy_b), (policy_b, policy_a)]


def _eval_seating_summary(*, paired_seats: bool) -> dict[str, object]:
    if paired_seats:
        return {
            "mode": "paired_seats",
            "paired_seats": True,
            "checkpoint_vs_baseline": "checkpoint_as_player_0_and_player_1",
            "training_seat_control": False,
        }
    return {
        "mode": "player0_only_checkpoint_vs_baseline",
        "paired_seats": False,
        "checkpoint_vs_baseline": "checkpoint_as_player_0_baseline_as_player_1",
        "training_seat_control": True,
    }


def _policy_specs(
    learned_specs: list[LearnedCheckpointPolicySpec],
    lightzero_specs: list[LightZeroCheckpointPolicySpec],
    *,
    baseline_policy_names: tuple[str, ...] = POLICY_NAMES,
) -> list[dict[str, object]]:
    baseline_specs = {
        "random_uniform": {
            "policy_id": "random_uniform",
            "kind": "rng_baseline",
            "description": "Uniform random up/stay/down action on every decision.",
        },
        "lagged_track_ball_1": {
            "policy_id": "lagged_track_ball_1",
            "kind": "scripted_baseline",
            "description": "Move the paddle center toward the previous ball row.",
        },
        "track_ball": {
            "policy_id": "track_ball",
            "kind": "scripted_baseline",
            "description": "Move the paddle center toward the current ball row.",
        },
        "stay": {
            "policy_id": "stay",
            "kind": "scripted_baseline",
            "description": "Always choose the stay action.",
        },
    }
    specs: list[dict[str, object]] = [baseline_specs[name] for name in baseline_policy_names]
    specs.extend(_learned_policy_spec_summary(spec) for spec in learned_specs)
    specs.extend(_lightzero_policy_spec_summary(spec) for spec in lightzero_specs)
    return specs


def _learned_policy_spec_summary(spec: LearnedCheckpointPolicySpec) -> dict[str, object]:
    return {
        "policy_id": spec.policy_id,
        "kind": "learned_checkpoint",
        "checkpoint_spec": f"{LEARNED_POLICY_PREFIX}{spec.checkpoint_path}",
        "checkpoint_path": str(spec.checkpoint_path),
        "checkpoint_schema_id": spec.checkpoint_schema_id,
        "checkpoint_metadata": spec.checkpoint_metadata,
        "raster_observation_schema_id": RASTER_OBSERVATION_SCHEMA_ID,
        "feature_mode": spec.feature_mode,
        "feature_encoding_id": spec.feature_encoding_id,
        "frame_stack": spec.frame_stack,
        "description": (
            "Loaded raster policy checkpoint; predicts from env.raster_observation(), "
            "optional per-agent raster history, and the ego agent."
        ),
    }


def _lightzero_policy_spec_summary(spec: LightZeroCheckpointPolicySpec) -> dict[str, object]:
    kind = (
        "lightzero_mcts_eval_mode"
        if isinstance(spec, LightZeroMCTSEvalModeSpec)
        else "lightzero_policy_head_greedy_no_mcts"
    )
    description = (
        f"{LIGHTZERO_MCTS_EVAL_MODE_LABEL}; calls MuZeroPolicy.eval_mode.forward "
        f"with a {spec.feature_mode} observation, all-ones action mask, to_play=[-1], "
        "ready_env_id=[0], and LightZero MCTS."
        if isinstance(spec, LightZeroMCTSEvalModeSpec)
        else (
            f"{LIGHTZERO_POLICY_HEAD_GREEDY_LABEL}; direct MuZeroModelMLP "
            f"initial_inference policy logits from {spec.feature_mode} only. "
            "This does not use MuZeroPolicy or MCTS and does not claim "
            "LightZero evaluator parity."
        )
    )
    return {
        "policy_id": spec.policy_id,
        "kind": kind,
        "checkpoint_spec": f"{LIGHTZERO_POLICY_PREFIX}{spec.checkpoint_path}",
        "checkpoint_path": str(spec.checkpoint_path),
        "checkpoint_schema_id": spec.checkpoint_schema_id,
        "checkpoint_metadata": spec.checkpoint_metadata,
        "feature_mode": spec.feature_mode,
        "feature_schema_id": spec.feature_schema_id,
        "adapter_schema_id": spec.adapter_schema_id,
        "adapter_label": spec.adapter_label,
        "description": description,
    }


def _run_matchup(
    *,
    config: PongConfig,
    match_id: str,
    pair_group_id: str,
    player_policy_names: dict[str, str],
    episodes: int,
    seed: int,
    policy_seed: int,
    checkpoint_specs_by_id: dict[str, LearnedCheckpointPolicySpec | LightZeroCheckpointPolicySpec],
) -> list[EpisodeRecord]:
    records = []
    for episode in range(episodes):
        episode_seed = seed + episode
        policies = {
            agent: _make_policy(
                policy_name,
                seed=policy_seed + index * 100,
                checkpoint_specs_by_id=checkpoint_specs_by_id,
            )
            for index, (agent, policy_name) in enumerate(player_policy_names.items())
        }
        for agent, policy in policies.items():
            policy.reset(episode_seed, agent)
        records.append(
            _run_episode(
                config=config,
                match_id=match_id,
                pair_group_id=pair_group_id,
                episode=episode,
                episode_seed=episode_seed,
                player_policy_by_agent=player_policy_names,
                policies=policies,
            )
        )
    return records


def _run_episode(
    *,
    config: PongConfig,
    match_id: str,
    pair_group_id: str,
    episode: int,
    episode_seed: int,
    player_policy_by_agent: dict[str, str],
    policies: dict[str, BaselinePolicy],
) -> EpisodeRecord:
    env = PongEnv(config)
    observations = env.reset(seed=episode_seed)
    action_counts = {agent: [0, 0, 0] for agent in AGENTS}

    final_step = None
    while True:
        joint_action = {}
        raster_grid = env.raster_observation()
        for agent in AGENTS:
            action = policies[agent].action(observations[agent], raster_grid, agent)
            if action < 0 or action >= config.action_count:
                raise ValueError(
                    f"policy {policies[agent].name!r} returned invalid action {action!r}"
                )
            joint_action[agent] = action
            action_counts[agent][action] += 1
        final_step = env.step(joint_action)
        observations = final_step.observations
        if final_step.terminated or final_step.truncated:
            break

    return EpisodeRecord(
        match_id=match_id,
        pair_group_id=pair_group_id,
        episode=episode,
        seed=episode_seed,
        player_policy_by_agent=dict(player_policy_by_agent),
        steps=int(next(iter(final_step.observations.values())).step),
        max_steps=config.max_steps,
        winner=final_step.infos["winner"] if not final_step.truncated else None,
        truncated=final_step.truncated,
        rewards={agent: float(final_step.rewards[agent]) for agent in AGENTS},
        action_counts_by_agent=action_counts,
        last_hit=final_step.infos["last_hit"],
    )


def _make_policy(
    policy_name: str,
    *,
    seed: int,
    checkpoint_specs_by_id: dict[str, LearnedCheckpointPolicySpec | LightZeroCheckpointPolicySpec],
) -> BaselinePolicy:
    if policy_name == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    if policy_name == "lagged_track_ball_1":
        return LaggedTrackBallPolicy(delay=1)
    if policy_name == "track_ball":
        return TrackBallPolicy()
    if policy_name == "stay":
        return StayPolicy()
    if policy_name in checkpoint_specs_by_id:
        spec = checkpoint_specs_by_id[policy_name]
        if isinstance(spec, LearnedCheckpointPolicySpec):
            return LearnedCheckpointPolicy(spec=spec)
        if isinstance(spec, (LightZeroPolicyHeadGreedySpec, LightZeroMCTSEvalModeSpec)):
            return spec.policy
    raise ValueError(f"unknown baseline policy {policy_name!r}")


def _resolve_baseline_policy_names(
    baseline_policy_names: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if baseline_policy_names is None:
        return POLICY_NAMES
    names = tuple(str(name) for name in baseline_policy_names)
    if not names:
        raise ValueError("baseline_policy_names must include at least one policy")
    unknown = sorted(set(names) - set(EXTENDED_POLICY_NAMES))
    if unknown:
        raise ValueError(
            f"unknown baseline policies {unknown!r}; expected subset of {EXTENDED_POLICY_NAMES!r}"
        )
    return names


def _summarize_records(records: list[EpisodeRecord]) -> dict[str, object]:
    if not records:
        raise ValueError("cannot summarize an empty matchup")

    player_policy_by_agent = records[0].player_policy_by_agent
    policy_names = sorted(set(player_policy_by_agent.values()))
    wins_by_policy: Counter[str] = Counter()
    action_histogram_by_player = {agent: [0, 0, 0] for agent in AGENTS}
    action_histogram_by_policy = {policy_name: [0, 0, 0] for policy_name in policy_names}
    rewards_by_policy: dict[str, list[float]] = {policy_name: [] for policy_name in policy_names}
    shaped_returns_by_policy: dict[str, list[float]] = {
        policy_name: [] for policy_name in policy_names
    }
    hits_by_player: Counter[str] = Counter()
    step_values = np.asarray([record.steps for record in records], dtype=np.float64)
    truncations = int(sum(record.truncated for record in records))

    for record in records:
        if record.winner is not None:
            wins_by_policy[player_policy_by_agent[record.winner]] += 1
        if record.last_hit is not None:
            hits_by_player[record.last_hit] += 1
        for agent in AGENTS:
            policy_name = player_policy_by_agent[agent]
            rewards_by_policy[policy_name].append(record.rewards[agent])
            shaped_returns_by_policy[policy_name].append(
                _shaped_loss_delay_return(
                    score_return=record.rewards[agent],
                    episode_steps=record.steps,
                    max_steps=record.max_steps,
                )
            )
            for action, count in enumerate(record.action_counts_by_agent[agent]):
                action_histogram_by_player[agent][action] += count
                action_histogram_by_policy[policy_name][action] += count

    return {
        "match_id": records[0].match_id,
        "pair_group_id": records[0].pair_group_id,
        "episodes": len(records),
        "seed_start": records[0].seed,
        "player_policy_by_agent": dict(player_policy_by_agent),
        "wins_by_player": {
            agent: sum(record.winner == agent for record in records)
            for agent in AGENTS
        },
        "wins_by_policy": {
            policy_name: wins_by_policy[policy_name]
            for policy_name in policy_names
        },
        "truncations": truncations,
        "truncation_rate": float(truncations / len(records)),
        "mean_steps": float(np.mean(step_values)),
        "median_steps": float(np.median(step_values)),
        "p90_steps": float(np.percentile(step_values, 90)),
        "std_steps": float(np.std(step_values)),
        "survival_steps": _numeric_stats(step_values),
        "mean_reward_by_player": {
            agent: float(np.mean([record.rewards[agent] for record in records]))
            for agent in AGENTS
        },
        "mean_reward_by_policy": {
            policy_name: float(np.mean(rewards))
            for policy_name, rewards in rewards_by_policy.items()
        },
        "score_return_stats_by_policy": {
            policy_name: _numeric_stats(np.asarray(rewards, dtype=np.float64))
            for policy_name, rewards in rewards_by_policy.items()
        },
        "mean_shaped_loss_delay_return_by_policy": {
            policy_name: float(np.mean(returns))
            for policy_name, returns in shaped_returns_by_policy.items()
        },
        "shaped_loss_delay_return_stats_by_policy": {
            policy_name: _numeric_stats(np.asarray(returns, dtype=np.float64))
            for policy_name, returns in shaped_returns_by_policy.items()
        },
        "action_histogram_by_player": action_histogram_by_player,
        "action_histogram_by_policy": action_histogram_by_policy,
        "last_hits_by_player": {
            agent: hits_by_player[agent]
            for agent in AGENTS
        },
    }


def _summarize_pair_groups(records: list[EpisodeRecord]) -> list[dict[str, object]]:
    groups: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        groups.setdefault(record.pair_group_id, []).append(record)

    summaries = []
    for pair_group_id, group_records in sorted(groups.items()):
        rewards_by_policy: dict[str, list[float]] = {}
        shaped_returns_by_policy: dict[str, list[float]] = {}
        wins_by_policy: Counter[str] = Counter()
        for record in group_records:
            if record.winner is not None:
                wins_by_policy[record.player_policy_by_agent[record.winner]] += 1
            for agent in AGENTS:
                policy_name = record.player_policy_by_agent[agent]
                rewards_by_policy.setdefault(policy_name, []).append(record.rewards[agent])
                shaped_returns_by_policy.setdefault(policy_name, []).append(
                    _shaped_loss_delay_return(
                        score_return=record.rewards[agent],
                        episode_steps=record.steps,
                        max_steps=record.max_steps,
                    )
                )
        step_values = np.asarray([record.steps for record in group_records], dtype=np.float64)
        truncations = int(sum(record.truncated for record in group_records))
        summaries.append(
            {
                "pair_group_id": pair_group_id,
                "episodes": len(group_records),
                "wins_by_policy": dict(sorted(wins_by_policy.items())),
                "truncations": truncations,
                "truncation_rate": float(truncations / len(group_records)),
                "mean_steps": float(np.mean(step_values)),
                "median_steps": float(np.median(step_values)),
                "p90_steps": float(np.percentile(step_values, 90)),
                "std_steps": float(np.std(step_values)),
                "survival_steps": _numeric_stats(step_values),
                "mean_reward_by_policy": {
                    policy_name: float(np.mean(rewards))
                    for policy_name, rewards in sorted(rewards_by_policy.items())
                },
                "score_return_stats_by_policy": {
                    policy_name: _numeric_stats(np.asarray(rewards, dtype=np.float64))
                    for policy_name, rewards in sorted(rewards_by_policy.items())
                },
                "mean_shaped_loss_delay_return_by_policy": {
                    policy_name: float(np.mean(returns))
                    for policy_name, returns in sorted(shaped_returns_by_policy.items())
                },
                "shaped_loss_delay_return_stats_by_policy": {
                    policy_name: _numeric_stats(np.asarray(returns, dtype=np.float64))
                    for policy_name, returns in sorted(shaped_returns_by_policy.items())
                },
            }
        )
    return summaries


def _shaped_loss_delay_return(
    *,
    score_return: float,
    episode_steps: int,
    max_steps: int,
) -> float:
    if score_return > 0.0:
        return 1.0
    if score_return < 0.0:
        return -1.0 + 0.5 * (episode_steps / max_steps)
    return 0.0


def _numeric_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _agent_seed(agent: str) -> int:
    if agent == "player_0":
        return 17
    if agent == "player_1":
        return 29
    raise ValueError(f"unknown agent {agent!r}")


def _edge_seeking_impact_offset(hit_y: int, *, height: int, default: int) -> int:
    middle_y = height // 2
    if hit_y < middle_y:
        return -1
    if hit_y > middle_y:
        return 1
    return default


def _predict_contact_y(*, ball_y: int, ball_vy: int, steps: int, height: int) -> int:
    predicted_y = ball_y
    predicted_vy = ball_vy
    for _ in range(steps):
        next_y = predicted_y + predicted_vy
        if next_y < 0:
            next_y = 1
            predicted_vy = 1
        elif next_y >= height:
            next_y = height - 2
            predicted_vy = -1
        predicted_y = next_y
    return predicted_y


def _load_learned_checkpoint_policies(
    checkpoint_policies: list[str | Path],
) -> list[LearnedCheckpointPolicySpec]:
    from curvyzero.training.dummy_pong_imitation_train import load_dummy_pong_imitation_checkpoint

    specs = []
    used_policy_ids = set(POLICY_NAMES)
    for index, value in enumerate(checkpoint_policies):
        label, checkpoint_path = _checkpoint_policy_spec(value)
        policy = load_dummy_pong_imitation_checkpoint(checkpoint_path)
        if tuple(policy.action_labels) != tuple(ACTION_LABELS):
            raise ValueError(
                f"checkpoint {checkpoint_path} action labels {policy.action_labels!r} "
                f"do not match eval action labels {ACTION_LABELS!r}"
            )
        policy_id = _learned_policy_id(
            checkpoint_path,
            label=label,
            index=index,
            used_policy_ids=used_policy_ids,
        )
        used_policy_ids.add(policy_id)
        checkpoint_metadata = dict(policy.metadata)
        schemas = checkpoint_metadata.get("schemas")
        if not isinstance(schemas, dict):
            schemas = {}
        specs.append(
            LearnedCheckpointPolicySpec(
                policy_id=policy_id,
                checkpoint_path=checkpoint_path,
                policy=policy,
                checkpoint_metadata=checkpoint_metadata,
                checkpoint_schema_id=str(
                    checkpoint_metadata.get(
                        "checkpoint_schema_id",
                        IMITATION_POLICY_CHECKPOINT_SCHEMA_ID,
                    )
                ),
                feature_encoding_id=str(
                    schemas.get(
                        "feature_encoding_id",
                        FEATURE_ENCODING_ID,
                    )
                ),
                feature_mode=str(getattr(policy, "feature_mode", DEFAULT_FEATURE_MODE)),
                frame_stack=int(getattr(policy, "frame_stack", 1)),
            )
        )
    return specs


def _load_lightzero_checkpoint_policies(
    checkpoint_policies: list[str | Path],
    *,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    used_policy_ids: set[str],
) -> list[LightZeroPolicyHeadGreedySpec]:
    specs = []
    for index, value in enumerate(checkpoint_policies):
        label, checkpoint_path = _lightzero_checkpoint_policy_spec(value)
        policy_id = _lightzero_policy_id(
            checkpoint_path,
            label=label,
            index=index,
            used_policy_ids=used_policy_ids,
        )
        used_policy_ids.add(policy_id)
        spec = load_lightzero_policy_head_greedy_checkpoint(
            policy_id=policy_id,
            checkpoint_path=checkpoint_path,
            env=env,
            feature_mode=feature_mode,
            opponent_policy=opponent_policy,
            seed=seed,
            max_env_step=max_env_step,
        )
        if spec.adapter_schema_id != LIGHTZERO_POLICY_HEAD_GREEDY_SCHEMA_ID:
            raise ValueError(
                f"unexpected LightZero adapter schema {spec.adapter_schema_id!r}"
            )
        specs.append(spec)
    return specs


def _load_lightzero_mcts_checkpoint_policies(
    checkpoint_policies: list[str | Path],
    *,
    env: str,
    feature_mode: str,
    opponent_policy: str,
    seed: int,
    max_env_step: int,
    num_simulations: int,
    used_policy_ids: set[str],
) -> list[LightZeroMCTSEvalModeSpec]:
    specs = []
    for index, value in enumerate(checkpoint_policies):
        label, checkpoint_path = _lightzero_checkpoint_policy_spec(value)
        policy_id = _lightzero_policy_id(
            checkpoint_path,
            label=label,
            index=index,
            used_policy_ids=used_policy_ids,
        )
        used_policy_ids.add(policy_id)
        spec = load_lightzero_mcts_eval_mode_checkpoint(
            policy_id=policy_id,
            checkpoint_path=checkpoint_path,
            env=env,
            feature_mode=feature_mode,
            opponent_policy=opponent_policy,
            seed=seed,
            max_env_step=max_env_step,
            num_simulations=num_simulations,
        )
        if spec.adapter_schema_id != LIGHTZERO_MCTS_EVAL_MODE_SCHEMA_ID:
            raise ValueError(
                f"unexpected LightZero MCTS adapter schema {spec.adapter_schema_id!r}"
            )
        specs.append(spec)
    return specs


def _pad_raster_history(history: list[np.ndarray], *, frame_stack: int) -> list[np.ndarray]:
    if not history:
        raise ValueError("cannot pad an empty raster history")
    if len(history) >= frame_stack:
        return [frame.copy() for frame in history[-frame_stack:]]
    padding = [history[0].copy() for _ in range(frame_stack - len(history))]
    return padding + [frame.copy() for frame in history]


def _checkpoint_policy_spec(value: str | Path) -> tuple[str | None, Path]:
    if isinstance(value, Path):
        return None, value
    if not value.startswith(LEARNED_POLICY_PREFIX):
        raise ValueError(
            f"checkpoint policy must use {LEARNED_POLICY_PREFIX}<checkpoint.npz> syntax: {value!r}"
        )
    spec_text = value[len(LEARNED_POLICY_PREFIX) :]
    label = None
    path_text = spec_text
    if "=" in spec_text:
        label_text, path_text = spec_text.split("=", 1)
        label = label_text.strip()
        if not label:
            raise ValueError("checkpoint policy label must not be empty")
    if not path_text:
        raise ValueError("checkpoint policy path must not be empty")
    return label, Path(path_text)


def _lightzero_checkpoint_policy_spec(value: str | Path) -> tuple[str | None, Path]:
    if isinstance(value, Path):
        return None, value
    if not value.startswith(LIGHTZERO_POLICY_PREFIX):
        raise ValueError(
            f"LightZero checkpoint policy must use "
            f"{LIGHTZERO_POLICY_PREFIX}LABEL=checkpoint.pth.tar syntax: {value!r}"
        )
    spec_text = value[len(LIGHTZERO_POLICY_PREFIX) :]
    label = None
    path_text = spec_text
    if "=" in spec_text:
        label_text, path_text = spec_text.split("=", 1)
        label = label_text.strip()
        if not label:
            raise ValueError("LightZero checkpoint policy label must not be empty")
    if not path_text:
        raise ValueError("LightZero checkpoint policy path must not be empty")
    return label, Path(path_text)


def _learned_policy_id(
    checkpoint_path: Path,
    *,
    label: str | None = None,
    index: int,
    used_policy_ids: set[str],
) -> str:
    source = label or (
        checkpoint_path.parent.name
        if checkpoint_path.name == "checkpoint.npz"
        else checkpoint_path.stem
    )
    slug = "".join(character if character.isalnum() else "_" for character in source).strip("_")
    base_policy_id = f"learned_{slug or 'checkpoint'}"
    policy_id = base_policy_id
    suffix = index
    while policy_id in used_policy_ids:
        suffix += 1
        policy_id = f"{base_policy_id}_{suffix}"
    return policy_id


def _lightzero_policy_id(
    checkpoint_path: Path,
    *,
    label: str | None = None,
    index: int,
    used_policy_ids: set[str],
) -> str:
    source = label or (
        checkpoint_path.parent.name
        if checkpoint_path.name == "ckpt_best.pth.tar"
        else checkpoint_path.stem.replace(".pth", "")
    )
    slug = "".join(character if character.isalnum() else "_" for character in source).strip("_")
    base_policy_id = f"lightzero_{slug or 'checkpoint'}"
    policy_id = base_policy_id
    suffix = index
    while policy_id in used_policy_ids:
        suffix += 1
        policy_id = f"{base_policy_id}_{suffix}"
    return policy_id


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "summary.json",
        "episodes_jsonl": output_dir / "episodes.jsonl",
    }


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    records: list[EpisodeRecord],
) -> None:
    summary_path = artifacts["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with artifacts["episodes_jsonl"].open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
