"""Probe whether dummy Pong angle control can challenge track_ball."""

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
from curvyzero.training.dummy_pong import PongConfig
from curvyzero.training.dummy_pong import PongEnv
from curvyzero.training.dummy_pong_eval import AngleControlPolicy
from curvyzero.training.dummy_pong_eval import BaselinePolicy
from curvyzero.training.dummy_pong_eval import RandomUniformPolicy
from curvyzero.training.dummy_pong_eval import TrackBallPolicy

PROBE_SCHEMA_ID = "dummy_pong_angle_control_probe_v0"
MATCHUPS = (
    ("angle_control", "track_ball"),
    ("track_ball", "angle_control"),
    ("angle_control", "random_uniform"),
    ("random_uniform", "angle_control"),
)


@dataclass(frozen=True, slots=True)
class ContactRecord:
    step: int
    agent: str
    policy: str
    impact_offset: int
    outgoing_ball_vy: int
    hit_y: int
    paddle_center_y: int


@dataclass(frozen=True, slots=True)
class ProbeEpisodeRecord:
    match_id: str
    episode: int
    seed: int
    player_policy_by_agent: dict[str, str]
    steps: int
    winner: str | None
    truncated: bool
    rewards: dict[str, float]
    action_counts_by_agent: dict[str, list[int]]
    contacts: list[ContactRecord]


def run_dummy_pong_angle_control_probe(
    *,
    episodes: int,
    seed: int,
    max_steps: int = 120,
    output_dir: Path | None = None,
) -> dict[str, object]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    config = PongConfig(max_steps=max_steps)
    records: list[ProbeEpisodeRecord] = []
    for matchup_index, (player_0_policy, player_1_policy) in enumerate(MATCHUPS):
        match_seed = seed + matchup_index * 100_000
        match_id = f"{player_0_policy}_p0__{player_1_policy}_p1"
        for episode in range(episodes):
            episode_seed = match_seed + episode
            records.append(
                _run_probe_episode(
                    config=config,
                    match_id=match_id,
                    episode=episode,
                    episode_seed=episode_seed,
                    policy_seed=seed + matchup_index * 10_000 + episode * 100,
                    player_policy_by_agent={
                        "player_0": player_0_policy,
                        "player_1": player_1_policy,
                    },
                )
            )

    summary: dict[str, object] = {
        "kind": "curvyzero_dummy_pong_angle_control_probe",
        "probe_schema_id": PROBE_SCHEMA_ID,
        "seed": seed,
        "episodes_per_match": episodes,
        "total_episodes": len(records),
        "config": asdict(config),
        "action_labels": list(ACTION_LABELS),
        "paddle_bounce_schema_id": PADDLE_BOUNCE_SCHEMA_ID,
        "policies": [
            {
                "policy_id": "angle_control",
                "kind": "scripted_probe",
                "description": (
                    "Tracks normally, but on incoming balls aims for top or bottom "
                    "paddle contact using the predicted contact row."
                ),
            },
            {
                "policy_id": "track_ball",
                "kind": "scripted_baseline",
                "description": "Moves paddle center toward the current ball row.",
            },
            {
                "policy_id": "random_uniform",
                "kind": "rng_baseline",
                "description": "Uniform random up/stay/down action on every decision.",
            },
        ],
        "matchups": _summarize_matchups(records),
        "pair_groups": _summarize_pair_groups(records),
    }
    if output_dir is not None:
        artifacts = {
            "summary_json": output_dir / "summary.json",
            "episodes_jsonl": output_dir / "episodes.jsonl",
        }
        summary["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        _write_artifacts(artifacts=artifacts, summary=summary, records=records)
    return summary


def _run_probe_episode(
    *,
    config: PongConfig,
    match_id: str,
    episode: int,
    episode_seed: int,
    policy_seed: int,
    player_policy_by_agent: dict[str, str],
) -> ProbeEpisodeRecord:
    env = PongEnv(config)
    observations = env.reset(seed=episode_seed)
    policies = {
        agent: _make_policy(
            policy_name,
            seed=policy_seed + index * 100,
        )
        for index, (agent, policy_name) in enumerate(player_policy_by_agent.items())
    }
    for agent, policy in policies.items():
        policy.reset(episode_seed, agent)

    action_counts = {agent: [0, 0, 0] for agent in AGENTS}
    contacts: list[ContactRecord] = []
    final_step = None
    while True:
        previous_vx = observations["player_0"].ball_vx_forward
        raster_grid = env.raster_observation()
        joint_action = {}
        for agent in AGENTS:
            action = policies[agent].action(observations[agent], raster_grid, agent)
            if action < 0 or action >= config.action_count:
                raise ValueError(
                    f"policy {policies[agent].name!r} returned invalid action {action!r}"
                )
            joint_action[agent] = action
            action_counts[agent][action] += 1

        final_step = env.step(joint_action)
        if final_step.infos["ball"]["vx"] != previous_vx:
            impact = final_step.infos["last_hit_impact"]
            if impact is None:
                raise RuntimeError("ball vx changed without a recorded paddle hit")
            agent = str(impact["agent"])
            contacts.append(
                ContactRecord(
                    step=int(next(iter(final_step.observations.values())).step),
                    agent=agent,
                    policy=player_policy_by_agent[agent],
                    impact_offset=int(impact["impact_offset"]),
                    outgoing_ball_vy=int(impact["outgoing_ball_vy"]),
                    hit_y=int(impact["hit_y"]),
                    paddle_center_y=int(impact["paddle_center_y"]),
                )
            )
        observations = final_step.observations
        if final_step.terminated or final_step.truncated:
            break

    return ProbeEpisodeRecord(
        match_id=match_id,
        episode=episode,
        seed=episode_seed,
        player_policy_by_agent=dict(player_policy_by_agent),
        steps=int(next(iter(final_step.observations.values())).step),
        winner=final_step.infos["winner"] if not final_step.truncated else None,
        truncated=final_step.truncated,
        rewards={agent: float(final_step.rewards[agent]) for agent in AGENTS},
        action_counts_by_agent=action_counts,
        contacts=contacts,
    )


def _make_policy(policy_name: str, *, seed: int) -> BaselinePolicy:
    if policy_name == "angle_control":
        return AngleControlPolicy()
    if policy_name == "track_ball":
        return TrackBallPolicy()
    if policy_name == "random_uniform":
        return RandomUniformPolicy(seed=seed)
    raise ValueError(f"unknown policy {policy_name!r}")


def _summarize_matchups(records: list[ProbeEpisodeRecord]) -> list[dict[str, object]]:
    by_match_id: dict[str, list[ProbeEpisodeRecord]] = {}
    for record in records:
        by_match_id.setdefault(record.match_id, []).append(record)
    return [
        _summarize_records(match_records)
        for match_id, match_records in sorted(by_match_id.items())
    ]


def _summarize_pair_groups(records: list[ProbeEpisodeRecord]) -> list[dict[str, object]]:
    by_pair: dict[str, list[ProbeEpisodeRecord]] = {}
    for record in records:
        policies = sorted(set(record.player_policy_by_agent.values()))
        pair_group_id = "_vs_".join(policies)
        by_pair.setdefault(pair_group_id, []).append(record)
    summaries = []
    for pair_group_id, pair_records in sorted(by_pair.items()):
        summary = _summarize_records(pair_records)
        summary.pop("match_id")
        summary.pop("player_policy_by_agent")
        summary["pair_group_id"] = pair_group_id
        summaries.append(summary)
    return summaries


def _summarize_records(records: list[ProbeEpisodeRecord]) -> dict[str, object]:
    if not records:
        raise ValueError("cannot summarize empty records")

    wins_by_policy: Counter[str] = Counter()
    rewards_by_policy: dict[str, list[float]] = {}
    action_histogram_by_policy: dict[str, list[int]] = {}
    for record in records:
        if record.winner is not None:
            wins_by_policy[record.player_policy_by_agent[record.winner]] += 1
        for agent in AGENTS:
            policy = record.player_policy_by_agent[agent]
            rewards_by_policy.setdefault(policy, []).append(record.rewards[agent])
            histogram = action_histogram_by_policy.setdefault(policy, [0, 0, 0])
            for action, count in enumerate(record.action_counts_by_agent[agent]):
                histogram[action] += count

    contacts = [contact for record in records for contact in record.contacts]
    return {
        "match_id": records[0].match_id,
        "episodes": len(records),
        "seed_start": records[0].seed,
        "player_policy_by_agent": dict(records[0].player_policy_by_agent),
        "wins_by_policy": dict(sorted(wins_by_policy.items())),
        "truncations": sum(record.truncated for record in records),
        "mean_steps": float(np.mean([record.steps for record in records])),
        "mean_reward_by_policy": {
            policy: float(np.mean(rewards))
            for policy, rewards in sorted(rewards_by_policy.items())
        },
        "action_histogram_by_policy": dict(sorted(action_histogram_by_policy.items())),
        "contacts": _summarize_contacts(contacts),
    }


def _summarize_contacts(contacts: list[ContactRecord]) -> dict[str, object]:
    by_policy: dict[str, list[ContactRecord]] = {}
    for contact in contacts:
        by_policy.setdefault(contact.policy, []).append(contact)
    return {
        "total": len(contacts),
        "by_policy": {
            policy: _contact_policy_summary(policy_contacts)
            for policy, policy_contacts in sorted(by_policy.items())
        },
    }


def _contact_policy_summary(contacts: list[ContactRecord]) -> dict[str, object]:
    offset_histogram: Counter[str] = Counter(str(contact.impact_offset) for contact in contacts)
    outgoing_vy_histogram: Counter[str] = Counter(
        str(contact.outgoing_ball_vy)
        for contact in contacts
    )
    off_center_contacts = sum(contact.impact_offset != 0 for contact in contacts)
    return {
        "total": len(contacts),
        "off_center_contacts": off_center_contacts,
        "off_center_rate": 0.0 if not contacts else off_center_contacts / len(contacts),
        "impact_offset_histogram": dict(sorted(offset_histogram.items())),
        "outgoing_ball_vy_histogram": dict(sorted(outgoing_vy_histogram.items())),
    }


def _write_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, object],
    records: list[ProbeEpisodeRecord],
) -> None:
    summary_path = artifacts["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with artifacts["episodes_jsonl"].open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_json(record), sort_keys=True) + "\n")


def _record_to_json(record: ProbeEpisodeRecord) -> dict[str, Any]:
    row = asdict(record)
    row["contacts"] = [asdict(contact) for contact in record.contacts]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = run_dummy_pong_angle_control_probe(
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
