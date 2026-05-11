"""Run the smallest Pong checkpoint scoreboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_eval import LEARNED_POLICY_PREFIX
from curvyzero.training.dummy_pong_eval import POLICY_NAMES
from curvyzero.training.dummy_pong_eval import run_dummy_pong_eval

SCOREBOARD_SCHEMA_ID = "dummy_pong_checkpoint_scoreboard_v0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--paired-seats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate checkpoint-vs-baseline rows in both seats. Use "
            "--no-paired-seats for checkpoint player_0 vs baseline player_1 only."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Learned checkpoint to score. Use LABEL=PATH for readable rows, "
            "for example latest=artifacts/local/run/checkpoint.npz."
        ),
    )
    parser.add_argument("--split-id", default=None)
    parser.add_argument("--split-role", default=None)
    args = parser.parse_args()

    checkpoint_policies = [
        _checkpoint_policy_arg(value)
        for value in args.checkpoint
    ]
    summary = run_dummy_pong_eval(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_policies=checkpoint_policies,
        paired_seats=args.paired_seats,
    )
    summary = _as_scoreboard_summary(
        summary,
        checkpoint_args=args.checkpoint,
        output_dir=args.output_dir,
        split_id=args.split_id,
        split_role=args.split_role,
        paired_seats=args.paired_seats,
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _checkpoint_policy_arg(value: str) -> str:
    label, path = _parse_checkpoint_arg(value)
    if label is None:
        return f"{LEARNED_POLICY_PREFIX}{path}"
    return f"{LEARNED_POLICY_PREFIX}{label}={path}"


def _parse_checkpoint_arg(value: str) -> tuple[str | None, Path]:
    if "=" not in value:
        return None, Path(value)
    label, path_text = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("checkpoint label must not be empty")
    if not path_text:
        raise ValueError("checkpoint path must not be empty")
    return label, Path(path_text)


def _as_scoreboard_summary(
    summary: dict[str, object],
    *,
    checkpoint_args: list[str],
    output_dir: Path,
    split_id: str | None,
    split_role: str | None,
    paired_seats: bool = True,
) -> dict[str, object]:
    source_eval_kind = summary.get("kind")
    summary["kind"] = "curvyzero_dummy_pong_checkpoint_scoreboard"
    summary["scoreboard_schema_id"] = SCOREBOARD_SCHEMA_ID
    summary["source_eval_kind"] = source_eval_kind
    summary["note"] = (
        "Checkpoint scoreboard for learned Pong policies versus random_uniform, "
        "lagged_track_ball_1, track_ball, and explicit learned checkpoint peers. "
        "Angle-control and contact-outcome probes are diagnostics outside this "
        "scoreboard."
    )
    summary["checkpoint_specs"] = _checkpoint_specs(summary, checkpoint_args)
    summary["scoreboard_rows"] = _scoreboard_rows(summary)
    summary["eval_setup"] = _eval_setup(summary["scoreboard_rows"])
    summary["artifacts"] = {
        "summary_json": str(output_dir / "summary.json"),
        "episodes_jsonl": str(output_dir / "episodes.jsonl"),
    }
    if split_id is not None or split_role is not None:
        summary["eval_split"] = {
            "split_id": split_id,
            "split_role": split_role,
            "seed_generation": "sequential_pair_seed_plus_episode",
            "base_seed": summary["seed"],
            "paired_seat": paired_seats,
            "eval_seating": summary.get("eval_seating"),
        }
    return summary


def _checkpoint_specs(
    summary: dict[str, object],
    checkpoint_args: list[str],
) -> list[dict[str, object]]:
    specs = []
    for index, policy in enumerate(summary.get("checkpoint_policies", [])):
        if not isinstance(policy, dict):
            continue
        checkpoint_arg = checkpoint_args[index] if index < len(checkpoint_args) else None
        checkpoint_label = None
        if checkpoint_arg is not None:
            checkpoint_label, _ = _parse_checkpoint_arg(checkpoint_arg)
        checkpoint_path = str(policy.get("checkpoint_path", ""))
        specs.append(
            {
                "policy_id": policy.get("policy_id"),
                "checkpoint_label": checkpoint_label,
                "checkpoint_arg": checkpoint_arg,
                "checkpoint_path": checkpoint_path,
                "checkpoint_schema_id": policy.get("checkpoint_schema_id"),
                "feature_encoding_id": policy.get("feature_encoding_id"),
            }
        )
    return specs


def _scoreboard_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    learned_policy_ids = {
        str(policy.get("policy_id"))
        for policy in summary.get("checkpoint_policies", [])
        if isinstance(policy, dict)
    }
    baseline_policy_ids = set(POLICY_NAMES)
    rows = []
    for group in summary.get("pair_groups", []):
        if not isinstance(group, dict):
            continue
        policy_ids = _policy_ids_for_pair_group(str(group["pair_group_id"]))
        rows.append(
            {
                "pair_group_id": group["pair_group_id"],
                "row_kind": _row_kind(policy_ids, baseline_policy_ids, learned_policy_ids),
                "setup": _setup_for_row(policy_ids, baseline_policy_ids, learned_policy_ids),
                "policies": policy_ids,
                "episodes": group["episodes"],
                "wins_by_policy": {
                    policy_id: int(group.get("wins_by_policy", {}).get(policy_id, 0))
                    for policy_id in policy_ids
                },
                "truncations": group["truncations"],
                "truncation_rate": group.get("truncation_rate"),
                "mean_steps": group["mean_steps"],
                "median_steps": group.get("median_steps"),
                "p90_steps": group.get("p90_steps"),
                "std_steps": group.get("std_steps"),
                "survival_steps": group.get("survival_steps"),
                "mean_reward_by_policy": {
                    policy_id: float(group.get("mean_reward_by_policy", {}).get(policy_id, 0.0))
                    for policy_id in policy_ids
                },
                "score_return_stats_by_policy": {
                    policy_id: group.get("score_return_stats_by_policy", {}).get(policy_id)
                    for policy_id in policy_ids
                },
                "mean_shaped_loss_delay_return_by_policy": {
                    policy_id: float(
                        group.get("mean_shaped_loss_delay_return_by_policy", {}).get(
                            policy_id,
                            0.0,
                        )
                    )
                    for policy_id in policy_ids
                },
                "shaped_loss_delay_return_stats_by_policy": {
                    policy_id: group.get("shaped_loss_delay_return_stats_by_policy", {}).get(
                        policy_id
                    )
                    for policy_id in policy_ids
                },
            }
        )
    return rows


def _policy_ids_for_pair_group(pair_group_id: str) -> list[str]:
    left, right = pair_group_id.rsplit("_vs_", 1)
    return [left, right] if left != right else [left]


def _row_kind(
    policy_ids: list[str],
    baseline_policy_ids: set[str],
    learned_policy_ids: set[str],
) -> str:
    policy_set = set(policy_ids)
    if policy_set <= baseline_policy_ids:
        return "baseline_sanity"
    if policy_set <= learned_policy_ids:
        return "learned_vs_learned"
    if policy_set & learned_policy_ids and policy_set & baseline_policy_ids:
        return "learned_vs_baseline"
    return "other"


def _setup_for_row(
    policy_ids: list[str],
    baseline_policy_ids: set[str],
    learned_policy_ids: set[str],
) -> str:
    policy_set = set(policy_ids)
    if policy_set & learned_policy_ids and policy_set <= learned_policy_ids:
        return "frozen_checkpoint_opponent"
    if policy_set & learned_policy_ids and policy_set & baseline_policy_ids:
        return "scripted_opponent"
    if policy_set <= baseline_policy_ids:
        return "scripted_opponent"
    return "unknown"


def _eval_setup(rows: list[dict[str, object]]) -> dict[str, object]:
    setups = sorted(
        {
            str(row.get("setup"))
            for row in rows
            if isinstance(row, dict) and row.get("setup")
        }
    )
    return {
        "schema": "curvyzero_dummy_pong_eval_setup/v0",
        "mode": "scorecard_matrix",
        "setups": setups,
        "true_self_play": "true_self_play" in setups,
        "notes": (
            "Independent scorecard rows use fixed evaluation policies. "
            "checkpoint-vs-checkpoint rows are frozen checkpoint opponents, "
            "not live learner self-play."
        ),
    }


if __name__ == "__main__":
    main()
