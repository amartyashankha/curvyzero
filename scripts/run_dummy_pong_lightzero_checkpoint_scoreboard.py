"""Run dummy Pong scoreboard for LightZero policy-head checkpoints.

This intentionally evaluates only direct policy-head greedy actions:
``MuZeroModelMLP.initial_inference(...).policy_logits.argmax()``. It does not
use LightZero MCTS or claim parity with LightZero's evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_eval import DEFAULT_LIGHTZERO_ENV
from curvyzero.training.dummy_pong_eval import DEFAULT_LIGHTZERO_FEATURE_MODE
from curvyzero.training.dummy_pong_eval import DEFAULT_LIGHTZERO_MAX_ENV_STEP
from curvyzero.training.dummy_pong_eval import DEFAULT_LIGHTZERO_OPPONENT_POLICY
from curvyzero.training.dummy_pong_eval import POLICY_NAMES
from curvyzero.training.dummy_pong_eval import run_dummy_pong_eval
from curvyzero.training.lightzero_dummy_pong_policy import LIGHTZERO_POLICY_HEAD_GREEDY_LABEL
from curvyzero.training.lightzero_dummy_pong_policy import LIGHTZERO_POLICY_HEAD_GREEDY_SCHEMA_ID
from curvyzero.training.lightzero_dummy_pong_policy import LIGHTZERO_POLICY_PREFIX

SCOREBOARD_SCHEMA_ID = "dummy_pong_lightzero_policy_head_checkpoint_scoreboard_v0"


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
            "Evaluate LightZero-vs-baseline rows in both seats. Use "
            "--no-paired-seats for LightZero player_0 vs baseline player_1 only."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        default=[],
        metavar="lightzero:LABEL=PATH",
        help=(
            "LightZero .pth.tar checkpoint to score. Use "
            "lightzero:LABEL=PATH, for example "
            "lightzero:best=/runs/.../ckpt_best.pth.tar."
        ),
    )
    parser.add_argument("--lightzero-env", default=DEFAULT_LIGHTZERO_ENV)
    parser.add_argument("--feature-mode", default=DEFAULT_LIGHTZERO_FEATURE_MODE)
    parser.add_argument("--opponent-policy", default=DEFAULT_LIGHTZERO_OPPONENT_POLICY)
    parser.add_argument("--max-env-step", type=int, default=DEFAULT_LIGHTZERO_MAX_ENV_STEP)
    parser.add_argument("--split-id", default=None)
    parser.add_argument("--split-role", default=None)
    args = parser.parse_args()

    checkpoint_policies = [
        _lightzero_checkpoint_policy_arg(value)
        for value in args.checkpoint
    ]
    summary = run_dummy_pong_eval(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        lightzero_checkpoint_policies=checkpoint_policies,
        lightzero_env=args.lightzero_env,
        lightzero_feature_mode=args.feature_mode,
        lightzero_opponent_policy=args.opponent_policy,
        lightzero_max_env_step=args.max_env_step,
        paired_seats=args.paired_seats,
    )
    summary = _as_lightzero_scoreboard_summary(
        summary,
        checkpoint_args=args.checkpoint,
        output_dir=args.output_dir,
        split_id=args.split_id,
        split_role=args.split_role,
        lightzero_env=args.lightzero_env,
        feature_mode=args.feature_mode,
        opponent_policy=args.opponent_policy,
        max_env_step=args.max_env_step,
        paired_seats=args.paired_seats,
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _lightzero_checkpoint_policy_arg(value: str) -> str:
    _parse_lightzero_checkpoint_arg(value)
    return value


def _parse_lightzero_checkpoint_arg(value: str) -> tuple[str | None, Path]:
    if not value.startswith(LIGHTZERO_POLICY_PREFIX):
        raise ValueError(
            f"checkpoint must use {LIGHTZERO_POLICY_PREFIX}LABEL=PATH syntax: {value!r}"
        )
    spec_text = value[len(LIGHTZERO_POLICY_PREFIX) :]
    if "=" not in spec_text:
        return None, Path(spec_text)
    label, path_text = spec_text.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("checkpoint label must not be empty")
    if not path_text:
        raise ValueError("checkpoint path must not be empty")
    return label, Path(path_text)


def _as_lightzero_scoreboard_summary(
    summary: dict[str, object],
    *,
    checkpoint_args: list[str],
    output_dir: Path,
    split_id: str | None,
    split_role: str | None,
    lightzero_env: str,
    feature_mode: str,
    opponent_policy: str,
    max_env_step: int,
    paired_seats: bool = True,
) -> dict[str, object]:
    source_eval_kind = summary.get("kind")
    summary["kind"] = "curvyzero_dummy_pong_lightzero_checkpoint_scoreboard"
    summary["scoreboard_schema_id"] = SCOREBOARD_SCHEMA_ID
    summary["source_eval_kind"] = source_eval_kind
    summary["adapter_schema_id"] = LIGHTZERO_POLICY_HEAD_GREEDY_SCHEMA_ID
    summary["adapter_label"] = LIGHTZERO_POLICY_HEAD_GREEDY_LABEL
    summary["note"] = (
        f"{LIGHTZERO_POLICY_HEAD_GREEDY_LABEL}. Scores LightZero MuZero "
        "checkpoints through direct policy logits only: no MuZeroPolicy, no "
        "MCTS, and no LightZero evaluator parity claim."
    )
    summary["lightzero_eval_config"] = {
        "env": lightzero_env,
        "feature_mode": feature_mode,
        "opponent_policy_for_config_reconstruction": opponent_policy,
        "max_env_step": max_env_step,
        "action_selection": "argmax(initial_inference.policy_logits)",
        "mcts": False,
    }
    summary["checkpoint_specs"] = _checkpoint_specs(summary, checkpoint_args)
    summary["scoreboard_rows"] = _scoreboard_rows(summary)
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
        if policy.get("kind") != "lightzero_policy_head_greedy_no_mcts":
            continue
        checkpoint_arg = checkpoint_args[index] if index < len(checkpoint_args) else None
        checkpoint_label = None
        if checkpoint_arg is not None:
            checkpoint_label, _ = _parse_lightzero_checkpoint_arg(checkpoint_arg)
        specs.append(
            {
                "policy_id": policy.get("policy_id"),
                "checkpoint_label": checkpoint_label,
                "checkpoint_arg": checkpoint_arg,
                "checkpoint_path": policy.get("checkpoint_path"),
                "checkpoint_schema_id": policy.get("checkpoint_schema_id"),
                "feature_mode": policy.get("feature_mode"),
                "feature_schema_id": policy.get("feature_schema_id"),
                "adapter_schema_id": policy.get("adapter_schema_id"),
                "adapter_label": policy.get("adapter_label"),
                "load_state_dict": dict(policy.get("checkpoint_metadata", {})).get(
                    "load_state_dict"
                ),
            }
        )
    return specs


def _scoreboard_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    lightzero_policy_ids = {
        str(policy.get("policy_id"))
        for policy in summary.get("checkpoint_policies", [])
        if isinstance(policy, dict)
        and policy.get("kind") == "lightzero_policy_head_greedy_no_mcts"
    }
    baseline_policy_ids = set(POLICY_NAMES)
    action_histograms_by_pair_group = _action_histograms_by_pair_group(summary)
    rows = []
    for group in summary.get("pair_groups", []):
        if not isinstance(group, dict):
            continue
        policy_ids = _policy_ids_for_pair_group(str(group["pair_group_id"]))
        action_histograms = action_histograms_by_pair_group.get(str(group["pair_group_id"]), {})
        rows.append(
            {
                "pair_group_id": group["pair_group_id"],
                "row_kind": _row_kind(policy_ids, baseline_policy_ids, lightzero_policy_ids),
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
                "action_histogram_by_policy": {
                    policy_id: action_histograms.get(policy_id, [0, 0, 0])
                    for policy_id in policy_ids
                },
            }
        )
    return rows


def _action_histograms_by_pair_group(
    summary: dict[str, object],
) -> dict[str, dict[str, list[int]]]:
    histograms_by_pair_group: dict[str, dict[str, list[int]]] = {}
    for matchup in summary.get("matchups", []):
        if not isinstance(matchup, dict):
            continue
        pair_group_id = matchup.get("pair_group_id")
        if pair_group_id is None:
            continue
        pair_histograms = histograms_by_pair_group.setdefault(str(pair_group_id), {})
        action_histograms = matchup.get("action_histogram_by_policy", {})
        if not isinstance(action_histograms, dict):
            continue
        for policy_id, counts in action_histograms.items():
            if not isinstance(counts, list):
                continue
            dest = pair_histograms.setdefault(str(policy_id), [0, 0, 0])
            for index, count in enumerate(counts[: len(dest)]):
                dest[index] += int(count)
    return histograms_by_pair_group


def _policy_ids_for_pair_group(pair_group_id: str) -> list[str]:
    left, right = pair_group_id.rsplit("_vs_", 1)
    return [left, right] if left != right else [left]


def _row_kind(
    policy_ids: list[str],
    baseline_policy_ids: set[str],
    lightzero_policy_ids: set[str],
) -> str:
    policy_set = set(policy_ids)
    if policy_set <= baseline_policy_ids:
        return "baseline_sanity"
    if policy_set <= lightzero_policy_ids:
        return "lightzero_vs_lightzero"
    if policy_set & lightzero_policy_ids and policy_set & baseline_policy_ids:
        return "lightzero_vs_baseline"
    return "other"


if __name__ == "__main__":
    main()
