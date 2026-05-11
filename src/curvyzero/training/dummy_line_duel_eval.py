"""Fixed baseline evaluation matrix for the dummy Tiny Line Duel task."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from curvyzero.training.dummy_line_duel import AGENTS
from curvyzero.training.dummy_line_duel import OBSERVATION_SCHEMA_ID
from curvyzero.training.dummy_line_duel import REWARD_SCHEMA_ID
from curvyzero.training.dummy_line_duel import RULESET_ID
from curvyzero.training.dummy_line_duel import DummyMuZeroModel
from curvyzero.training.dummy_line_duel import DummyPlanner
from curvyzero.training.dummy_line_duel import LineDuelConfig
from curvyzero.training.dummy_line_duel import LineDuelEnv
from curvyzero.training.dummy_line_duel import LineDuelObservation

EVAL_CONFIG_SCHEMA_ID = "dummy_line_duel_eval_config_v0"
EVAL_SUMMARY_SCHEMA_ID = "dummy_line_duel_eval_summary_v0"
ACTION_SCHEMA_ID = "line_duel_turn_actions_v0"
ACTION_LABELS = ("left", "straight", "right")
POLICY_NAMES = ("random_uniform", "random_sticky", "one_step_safe")


class BaselinePolicy(Protocol):
    name: str

    def reset(self, episode_seed: int, agent: str) -> None: ...

    def action(self, observation: LineDuelObservation) -> int: ...


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    match_id: str
    pair_group_id: str
    episode: int
    seed: int
    player_policy_by_agent: dict[str, str]
    steps: int
    winner: str | None
    draw: bool
    truncated: bool
    rewards: dict[str, float]
    action_counts_by_agent: dict[str, list[int]]
    death_causes: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class LearnedCheckpointPolicySpec:
    policy_id: str
    checkpoint_path: Path
    model: DummyMuZeroModel
    checkpoint_state_count: int
    checkpoint_dynamics_edge_count: int


class RandomUniformPolicy:
    name = "random_uniform"

    def __init__(self, seed: int):
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, episode_seed: int, agent: str) -> None:
        self._rng = np.random.default_rng(self._base_seed + episode_seed + _agent_seed(agent))

    def action(self, observation: LineDuelObservation) -> int:
        del observation
        return int(self._rng.integers(len(ACTION_LABELS)))


class RandomStickyPolicy:
    name = "random_sticky"

    def __init__(self, seed: int, stick_probability: float = 0.8):
        self._base_seed = seed
        self._stick_probability = stick_probability
        self._rng = np.random.default_rng(seed)
        self._previous_action: int | None = None

    def reset(self, episode_seed: int, agent: str) -> None:
        self._rng = np.random.default_rng(self._base_seed + episode_seed + _agent_seed(agent))
        self._previous_action = None

    def action(self, observation: LineDuelObservation) -> int:
        del observation
        if self._previous_action is None or float(self._rng.random()) >= self._stick_probability:
            self._previous_action = int(self._rng.integers(len(ACTION_LABELS)))
        return self._previous_action


class OneStepSafePolicy:
    name = "one_step_safe"

    def reset(self, episode_seed: int, agent: str) -> None:
        del episode_seed, agent

    def action(self, observation: LineDuelObservation) -> int:
        clearances = (
            observation.left_clearance,
            observation.straight_clearance,
            observation.right_clearance,
        )
        return max(
            range(len(ACTION_LABELS)),
            key=lambda action: (clearances[action], 1 if action == 1 else 0, -action),
        )


class LearnedCheckpointPolicy:
    def __init__(self, spec: LearnedCheckpointPolicySpec, config: LineDuelConfig, seed: int):
        self.name = spec.policy_id
        self._model = spec.model
        self._planner = DummyPlanner(discount=config.discount)
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, episode_seed: int, agent: str) -> None:
        self._rng = np.random.default_rng(self._base_seed + episode_seed + _agent_seed(agent))

    def action(self, observation: LineDuelObservation) -> int:
        return self._planner.select_action(
            model=self._model,
            observation=observation,
            rng=self._rng,
            epsilon=0.0,
            explore_unknown=False,
        )


def run_dummy_line_duel_eval(
    *,
    episodes: int,
    seed: int,
    output_dir: Path | None = None,
    checkpoint_policies: list[str | Path] | None = None,
    split_id: str = "dummy_line_duel_monitor_v0",
    split_role: str = "monitor",
) -> dict[str, object]:
    """Evaluate fixed Tiny Line Duel baselines and optional learned checkpoints."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    config = LineDuelConfig()
    learned_specs = _load_learned_checkpoint_policies(
        checkpoint_policies or [],
        config=config,
    )
    learned_specs_by_id = {spec.policy_id: spec for spec in learned_specs}
    all_records: list[EpisodeRecord] = []
    matchups = []
    for pair_index, (policy_a, policy_b) in enumerate(_policy_pairs(learned_specs_by_id)):
        pair_group_id = f"{policy_a}_vs_{policy_b}"
        pair_seed = seed + pair_index * 100_000
        seatings = [(policy_a, policy_b)] if policy_a == policy_b else [(policy_a, policy_b), (policy_b, policy_a)]
        for seating_index, (player_0_policy, player_1_policy) in enumerate(seatings):
            match_seed = pair_seed
            match_id = f"{player_0_policy}_p0__{player_1_policy}_p1"
            records = _run_matchup(
                config=config,
                match_id=match_id,
                pair_group_id=pair_group_id,
                player_policy_names={"player_0": player_0_policy, "player_1": player_1_policy},
                episodes=episodes,
                seed=match_seed,
                policy_seed=seed + pair_index * 10_000 + seating_index * 1_000,
                learned_specs_by_id=learned_specs_by_id,
            )
            all_records.extend(records)
            matchups.append(_summarize_records(records))

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_line_duel_eval",
        "note": "Fixed baseline matrix plus optional learned-vs-baseline checkpoints; no league or ratings.",
        "eval_config_schema_id": EVAL_CONFIG_SCHEMA_ID,
        "eval_summary_schema_id": EVAL_SUMMARY_SCHEMA_ID,
        "seed": seed,
        "episodes_per_match": episodes,
        "total_episodes": len(all_records),
        "paired_seat_group_count": _paired_seat_group_count(all_records),
        "config": asdict(config),
        "schemas": {
            "ruleset_id": RULESET_ID,
            "observation_schema_id": OBSERVATION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
        },
        "action_labels": list(ACTION_LABELS),
        "eval_split": {
            "split_id": split_id,
            "split_role": split_role,
            "seed_generation": "sequential_pair_seed_plus_episode",
            "base_seed": seed,
            "seed_count": len({record.seed for record in all_records}),
            "seed_list_hash": _seed_list_hash(sorted({record.seed for record in all_records})),
            "paired_seat": True,
        },
        "policies": _policy_specs(learned_specs),
        "matchups": matchups,
        "pair_groups": _summarize_pair_groups(all_records),
    }
    if output_dir is not None:
        artifacts = _artifact_paths(output_dir)
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, records=all_records)
    return summary


def _policy_pairs(
    learned_specs_by_id: dict[str, LearnedCheckpointPolicySpec],
) -> list[tuple[str, str]]:
    pairs = []
    for index, policy_a in enumerate(POLICY_NAMES):
        for policy_b in POLICY_NAMES[index:]:
            pairs.append((policy_a, policy_b))
    for learned_policy_id in learned_specs_by_id:
        for baseline_policy_id in POLICY_NAMES:
            pairs.append((learned_policy_id, baseline_policy_id))
    return pairs


def _policy_specs(
    learned_specs: list[LearnedCheckpointPolicySpec],
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "policy_id": "random_uniform",
            "kind": "rng_baseline",
            "description": "Uniform random action on every decision.",
        },
        {
            "policy_id": "random_sticky",
            "kind": "rng_baseline",
            "stick_probability": 0.8,
            "description": "Repeat the previous action with fixed probability, otherwise resample.",
        },
        {
            "policy_id": "one_step_safe",
            "kind": "scripted_baseline",
            "description": "Choose the turn with the largest immediate clearance; ties prefer straight.",
        },
    ]
    for learned_spec in learned_specs:
        model = learned_spec.model
        specs.append(
            {
                "policy_id": learned_spec.policy_id,
                "kind": "learned_checkpoint",
                "checkpoint_path": str(learned_spec.checkpoint_path),
                "checkpoint_metadata": model.checkpoint_metadata,
                "states": learned_spec.checkpoint_state_count,
                "learned_dynamics_edges": learned_spec.checkpoint_dynamics_edge_count,
                "description": "Loaded DummyMuZeroModel checkpoint with deterministic DummyPlanner.",
            }
        )
    return specs


def _run_matchup(
    *,
    config: LineDuelConfig,
    match_id: str,
    pair_group_id: str,
    player_policy_names: dict[str, str],
    episodes: int,
    seed: int,
    policy_seed: int,
    learned_specs_by_id: dict[str, LearnedCheckpointPolicySpec],
) -> list[EpisodeRecord]:
    records = []
    for episode in range(episodes):
        episode_seed = seed + episode
        policies = {
            agent: _make_policy(
                policy_name,
                seed=policy_seed + index * 100,
                config=config,
                learned_specs_by_id=learned_specs_by_id,
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
    config: LineDuelConfig,
    match_id: str,
    pair_group_id: str,
    episode: int,
    episode_seed: int,
    player_policy_by_agent: dict[str, str],
    policies: dict[str, BaselinePolicy],
) -> EpisodeRecord:
    env = LineDuelEnv(config)
    observations = env.reset(seed=episode_seed)
    action_counts = {agent: [0, 0, 0] for agent in AGENTS}

    final_step = None
    while True:
        joint_action = {}
        for agent in AGENTS:
            action = policies[agent].action(observations[agent])
            joint_action[agent] = action
            action_counts[agent][action] += 1
        final_step = env.step(joint_action)
        observations = final_step.observations
        if final_step.terminated or final_step.truncated:
            break

    winner = final_step.infos["winner"] if not final_step.truncated else None
    return EpisodeRecord(
        match_id=match_id,
        pair_group_id=pair_group_id,
        episode=episode,
        seed=episode_seed,
        player_policy_by_agent=dict(player_policy_by_agent),
        steps=int(next(iter(final_step.observations.values())).step),
        winner=winner,
        draw=winner is None and not final_step.truncated,
        truncated=final_step.truncated,
        rewards={agent: float(final_step.rewards[agent]) for agent in AGENTS},
        action_counts_by_agent=action_counts,
        death_causes={
            agent: list(final_step.infos["death_causes"][agent])
            for agent in AGENTS
        },
    )


def _make_policy(
    policy_name: str,
    *,
    seed: int,
    config: LineDuelConfig,
    learned_specs_by_id: dict[str, LearnedCheckpointPolicySpec],
) -> BaselinePolicy:
    if policy_name == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    if policy_name == "random_sticky":
        return RandomStickyPolicy(seed=seed)
    if policy_name == "one_step_safe":
        return OneStepSafePolicy()
    if policy_name in learned_specs_by_id:
        return LearnedCheckpointPolicy(
            spec=learned_specs_by_id[policy_name],
            config=config,
            seed=seed,
        )
    raise ValueError(f"unknown baseline policy {policy_name!r}")


def _summarize_records(records: list[EpisodeRecord]) -> dict[str, object]:
    if not records:
        raise ValueError("cannot summarize an empty matchup")

    match_id = records[0].match_id
    player_policy_by_agent = records[0].player_policy_by_agent
    wins_by_player = {
        agent: sum(record.winner == agent for record in records)
        for agent in AGENTS
    }
    wins_by_policy: Counter[str] = Counter()
    action_histogram_by_player = {agent: [0, 0, 0] for agent in AGENTS}
    action_histogram_by_policy = {
        policy_name: [0, 0, 0]
        for policy_name in sorted(set(player_policy_by_agent.values()))
    }
    rewards_by_policy: dict[str, list[float]] = {
        policy_name: []
        for policy_name in sorted(set(player_policy_by_agent.values()))
    }
    death_causes_by_player = {agent: Counter() for agent in AGENTS}
    death_causes_by_policy: dict[str, Counter[str]] = {
        policy_name: Counter()
        for policy_name in sorted(set(player_policy_by_agent.values()))
    }

    for record in records:
        if record.winner is not None:
            wins_by_policy[player_policy_by_agent[record.winner]] += 1
        for agent in AGENTS:
            policy_name = player_policy_by_agent[agent]
            rewards_by_policy[policy_name].append(record.rewards[agent])
            for action, count in enumerate(record.action_counts_by_agent[agent]):
                action_histogram_by_player[agent][action] += count
                action_histogram_by_policy[policy_name][action] += count
            for cause in record.death_causes[agent]:
                death_causes_by_player[agent][cause] += 1
                death_causes_by_policy[policy_name][cause] += 1

    return {
        "match_id": match_id,
        "pair_group_id": records[0].pair_group_id,
        "episodes": len(records),
        "seed_start": records[0].seed,
        "player_policy_by_agent": dict(player_policy_by_agent),
        "wins_by_player": wins_by_player,
        "wins_by_policy": {
            policy_name: wins_by_policy[policy_name]
            for policy_name in sorted(set(player_policy_by_agent.values()))
        },
        "draws": sum(record.draw for record in records),
        "truncations": sum(record.truncated for record in records),
        "mean_steps": float(np.mean([record.steps for record in records])),
        "mean_reward_by_player": {
            agent: float(np.mean([record.rewards[agent] for record in records]))
            for agent in AGENTS
        },
        "mean_reward_by_policy": {
            policy_name: float(np.mean(rewards))
            for policy_name, rewards in rewards_by_policy.items()
        },
        "action_histogram_by_player": action_histogram_by_player,
        "action_histogram_by_policy": action_histogram_by_policy,
        "death_causes_by_player": {
            agent: dict(sorted(counter.items()))
            for agent, counter in death_causes_by_player.items()
        },
        "death_causes_by_policy": {
            policy_name: dict(sorted(counter.items()))
            for policy_name, counter in death_causes_by_policy.items()
        },
    }


def _summarize_pair_groups(records: list[EpisodeRecord]) -> list[dict[str, object]]:
    groups: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        groups.setdefault(record.pair_group_id, []).append(record)

    summaries = []
    for pair_group_id, group_records in sorted(groups.items()):
        rewards_by_policy: dict[str, list[float]] = {}
        wins_by_policy: Counter[str] = Counter()
        wins_by_seat: Counter[str] = Counter()
        terminal_causes: Counter[str] = Counter()
        for record in group_records:
            if record.winner is not None:
                wins_by_seat[record.winner] += 1
                wins_by_policy[record.player_policy_by_agent[record.winner]] += 1
            terminal_causes.update(
                cause
                for causes in record.death_causes.values()
                for cause in causes
            )
            for agent in AGENTS:
                policy_name = record.player_policy_by_agent[agent]
                rewards_by_policy.setdefault(policy_name, []).append(record.rewards[agent])
        player_0_win_rate = wins_by_seat["player_0"] / len(group_records)
        player_1_win_rate = wins_by_seat["player_1"] / len(group_records)
        summaries.append(
            {
                "pair_group_id": pair_group_id,
                "episodes": len(group_records),
                "unique_env_seeds": len({record.seed for record in group_records}),
                "paired_seat_groups": _paired_seed_count(group_records),
                "wins_by_policy": dict(sorted(wins_by_policy.items())),
                "wins_by_seat": {agent: wins_by_seat[agent] for agent in AGENTS},
                "seat_delta_player_0_minus_player_1": player_0_win_rate - player_1_win_rate,
                "draws": sum(record.draw for record in group_records),
                "truncations": sum(record.truncated for record in group_records),
                "terminal_causes": dict(sorted(terminal_causes.items())),
                "mean_steps": float(np.mean([record.steps for record in group_records])),
                "mean_reward_by_policy": {
                    policy_name: float(np.mean(rewards))
                    for policy_name, rewards in sorted(rewards_by_policy.items())
                },
            }
        )
    return summaries


def _paired_seat_group_count(records: list[EpisodeRecord]) -> int:
    return sum(_paired_seed_count(group) for group in _records_by_pair_group(records).values())


def _paired_seed_count(records: list[EpisodeRecord]) -> int:
    by_seed: dict[int, set[tuple[tuple[str, str], ...]]] = {}
    for record in records:
        seating = tuple(sorted(record.player_policy_by_agent.items()))
        by_seed.setdefault(record.seed, set()).add(seating)
    return len(by_seed)


def _records_by_pair_group(records: list[EpisodeRecord]) -> dict[str, list[EpisodeRecord]]:
    groups: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        groups.setdefault(record.pair_group_id, []).append(record)
    return groups


def _seed_list_hash(seeds: list[int]) -> str:
    payload = json.dumps(seeds, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_learned_checkpoint_policies(
    checkpoint_policies: list[str | Path],
    *,
    config: LineDuelConfig,
) -> list[LearnedCheckpointPolicySpec]:
    specs = []
    used_policy_ids = set(POLICY_NAMES)
    for index, value in enumerate(checkpoint_policies):
        checkpoint_path = _checkpoint_policy_path(value)
        model = DummyMuZeroModel.from_checkpoint(
            checkpoint_path,
            action_count=config.action_count,
            width=config.width,
            height=config.height,
        )
        policy_id = _learned_policy_id(
            checkpoint_path,
            index=index,
            used_policy_ids=used_policy_ids,
        )
        used_policy_ids.add(policy_id)
        specs.append(
            LearnedCheckpointPolicySpec(
                policy_id=policy_id,
                checkpoint_path=checkpoint_path,
                model=model,
                checkpoint_state_count=len(model.q_values),
                checkpoint_dynamics_edge_count=len(model.transition_counts),
            )
        )
    return specs


def _checkpoint_policy_path(value: str | Path) -> Path:
    if isinstance(value, Path):
        return value
    prefix = "learned:"
    if not value.startswith(prefix):
        raise ValueError(
            f"checkpoint policy must use {prefix}<checkpoint.npz> syntax: {value!r}"
        )
    path_text = value[len(prefix) :]
    if not path_text:
        raise ValueError("checkpoint policy path must not be empty")
    return Path(path_text)


def _learned_policy_id(
    checkpoint_path: Path,
    *,
    index: int,
    used_policy_ids: set[str],
) -> str:
    source = (
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


def _agent_seed(agent: str) -> int:
    if agent == "player_0":
        return 17
    if agent == "player_1":
        return 29
    raise ValueError(f"unknown agent {agent!r}")


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
