"""Select one dummy Pong checkpoint from a checkpoint scoreboard summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SELECTION_RECORD_SCHEMA_ID = "dummy_pong_checkpoint_selection_record_v0"
SELECTION_RULE_ID = (
    "track_ball_win_rate_then_lower_track_ball_loss_rate_then_higher_track_ball_truncation_rate_then_random_uniform_win_rate_v0"
)
PRESSURE_RULE_ID = (
    "track_ball_win_rate_then_lower_track_ball_loss_rate_then_higher_track_ball_truncation_rate_v0"
)
REQUIRED_BASELINES = ("track_ball", "random_uniform")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing selection record. The input summary is never overwritten.",
    )
    args = parser.parse_args()

    output_path = _record_output_path(
        summary_path=args.summary,
        output_dir=args.output_dir,
        output_path=args.output_path,
    )
    record = select_dummy_pong_checkpoint(
        summary_path=args.summary,
        output_path=output_path,
    )
    _write_record(record=record, output_path=output_path, force=args.force)
    print(json.dumps(record, indent=2, sort_keys=True))


def select_dummy_pong_checkpoint(
    *,
    summary_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    summary_bytes = summary_path.read_bytes()
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    summary = json.loads(summary_bytes.decode("utf-8"))
    candidates = _candidate_records(summary)
    if not candidates:
        raise ValueError("scoreboard summary has no complete learned checkpoint candidates")

    candidates.sort(key=_rank_key, reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    pressure_candidates = sorted(candidates, key=_pressure_rank_key, reverse=True)
    for rank, candidate in enumerate(pressure_candidates, start=1):
        candidate["pressure_rank"] = rank
    selected = candidates[0]
    eval_split = _eval_split(summary)
    return {
        "kind": "curvyzero_dummy_pong_checkpoint_selection",
        "selection_record_schema_id": SELECTION_RECORD_SCHEMA_ID,
        "note": (
            "Selects one Pong checkpoint on one scoreboard split. This is "
            "selection-split bookkeeping only and does not prove final quality."
        ),
        "source_summary": {
            "path": str(summary_path),
            "sha256": summary_sha256,
            "kind": summary.get("kind"),
            "scoreboard_schema_id": summary.get("scoreboard_schema_id"),
            "seed": summary.get("seed"),
            "episodes_per_match": summary.get("episodes_per_match"),
            "total_episodes": summary.get("total_episodes"),
        },
        "selection_split": eval_split,
        "selection_rule": {
            "rule_id": SELECTION_RULE_ID,
            "required_baselines": list(REQUIRED_BASELINES),
            "description": (
                "Rank learned checkpoint candidates by wins against track_ball, "
                "then by fewer losses to track_ball, then by more truncations "
                "against track_ball, then by win rate against random_uniform. "
                "Exact ties keep checkpoint spec order from the scoreboard summary."
            ),
        },
        "pressure_rule": {
            "rule_id": PRESSURE_RULE_ID,
            "description": (
                "Debug ranking for the track_ball gate. Rank learned checkpoints "
                "by wins against track_ball, then by fewer track_ball wins, then "
                "by more truncations against track_ball. This is pressure "
                "bookkeeping, not the quality selector."
            ),
        },
        "candidate_count": len(candidates),
        "selected_checkpoint_label": selected["checkpoint_label"],
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_policy_id": selected["policy_id"],
        "selected_metric_values": selected["metric_values"],
        "selected_candidate": selected,
        "ranked_candidates": candidates,
        "pressure_ranked_candidates": pressure_candidates,
        "candidate_rows_used": [
            {
                "checkpoint_label": candidate["checkpoint_label"],
                "checkpoint_path": candidate["checkpoint_path"],
                "policy_id": candidate["policy_id"],
                "scoreboard_rows": candidate["scoreboard_rows_used"],
            }
            for candidate in candidates
        ],
        "heldout_required_for_quality_claim": True,
        "claim_status": "selected_pending_heldout",
        "artifacts": {
            "selection_record_json": str(output_path),
            "source_summary_json": str(summary_path),
        },
    }


def _record_output_path(
    *,
    summary_path: Path,
    output_dir: Path | None,
    output_path: Path | None,
) -> Path:
    if output_dir is not None and output_path is not None:
        raise ValueError("use either --output-dir or --output-path, not both")
    if output_path is not None:
        return output_path
    if output_dir is not None:
        return output_dir / "selection_record.json"
    return summary_path.parent / "selection_record.json"


def _write_record(
    *,
    record: dict[str, Any],
    output_path: Path,
    force: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_summary = Path(str(record["source_summary"]["path"]))
    if output_path.resolve(strict=False) == source_summary.resolve(strict=False):
        raise ValueError("refusing to overwrite the source summary.json")
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass --force to overwrite it")
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_records(summary: dict[str, Any]) -> list[dict[str, Any]]:
    specs = _checkpoint_specs_by_policy(summary)
    if not specs:
        specs = _fallback_specs_from_rows(summary)
    candidates = []
    for spec_index, (policy_id, spec) in enumerate(specs.items()):
        rows_by_baseline = _required_rows_for_policy(summary, policy_id)
        if set(rows_by_baseline) != set(REQUIRED_BASELINES):
            continue
        metrics_by_baseline = {
            baseline: _metric_for_row(
                row=rows_by_baseline[baseline],
                policy_id=policy_id,
                baseline=baseline,
            )
            for baseline in REQUIRED_BASELINES
        }
        metric_values = _candidate_metric_values(metrics_by_baseline)
        candidates.append(
            {
                "checkpoint_label": _checkpoint_label(spec, policy_id),
                "checkpoint_path": _string_or_none(spec.get("checkpoint_path")),
                "checkpoint_schema_id": _string_or_none(spec.get("checkpoint_schema_id")),
                "feature_encoding_id": _string_or_none(spec.get("feature_encoding_id")),
                "checkpoint_arg": _string_or_none(spec.get("checkpoint_arg")),
                "policy_id": policy_id,
                "spec_index": spec_index,
                "metric_values": metric_values,
                "metrics_by_baseline": metrics_by_baseline,
                "scoreboard_rows_used": [
                    rows_by_baseline[baseline] for baseline in REQUIRED_BASELINES
                ],
            }
        )
    return candidates


def _checkpoint_specs_by_policy(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {}
    for spec in summary.get("checkpoint_specs", []):
        if not isinstance(spec, dict):
            continue
        policy_id = spec.get("policy_id")
        if isinstance(policy_id, str) and policy_id:
            specs[policy_id] = dict(spec)
    return specs


def _fallback_specs_from_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for row in summary.get("scoreboard_rows", []):
        if not isinstance(row, dict):
            continue
        if row.get("row_kind") != "learned_vs_baseline":
            continue
        for policy_id in _policy_ids(row):
            if policy_id not in REQUIRED_BASELINES:
                specs.setdefault(policy_id, {"policy_id": policy_id})
    return specs


def _required_rows_for_policy(
    summary: dict[str, Any],
    policy_id: str,
) -> dict[str, dict[str, Any]]:
    rows_by_baseline = {}
    for row in summary.get("scoreboard_rows", []):
        if not isinstance(row, dict):
            continue
        policies = set(_policy_ids(row))
        if policy_id not in policies:
            continue
        for baseline in REQUIRED_BASELINES:
            if baseline in policies:
                rows_by_baseline[baseline] = dict(row)
    return rows_by_baseline


def _metric_for_row(
    *,
    row: dict[str, Any],
    policy_id: str,
    baseline: str,
) -> dict[str, Any]:
    episodes = _int_value(row.get("episodes"))
    if episodes <= 0:
        raise ValueError(f"{row.get('pair_group_id')} has no episodes")
    wins_by_policy = row.get("wins_by_policy", {})
    mean_reward_by_policy = row.get("mean_reward_by_policy", {})
    if not isinstance(wins_by_policy, dict):
        wins_by_policy = {}
    if not isinstance(mean_reward_by_policy, dict):
        mean_reward_by_policy = {}
    policy_wins = _int_value(wins_by_policy.get(policy_id, 0))
    baseline_wins = _int_value(wins_by_policy.get(baseline, 0))
    truncations = _int_value(row.get("truncations", 0))
    return {
        "baseline": baseline,
        "pair_group_id": row.get("pair_group_id"),
        "episodes": episodes,
        "policy_wins": policy_wins,
        "baseline_wins": baseline_wins,
        "win_rate": policy_wins / episodes,
        "truncations": truncations,
        "truncation_rate": truncations / episodes,
        "mean_steps": _float_or_none(row.get("mean_steps")),
        "mean_reward": _float_or_none(mean_reward_by_policy.get(policy_id)),
    }


def _candidate_metric_values(metrics_by_baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    track_ball = metrics_by_baseline["track_ball"]
    random_uniform = metrics_by_baseline["random_uniform"]
    total_episodes = sum(
        _int_value(metrics["episodes"]) for metrics in metrics_by_baseline.values()
    )
    total_truncations = sum(
        _int_value(metrics["truncations"]) for metrics in metrics_by_baseline.values()
    )
    truncation_rate = total_truncations / total_episodes
    track_ball_loss_rate = track_ball["baseline_wins"] / track_ball["episodes"]
    rank_key = {
        "track_ball_win_rate": track_ball["win_rate"],
        "negative_track_ball_loss_rate": -track_ball_loss_rate,
        "track_ball_truncation_rate": track_ball["truncation_rate"],
        "random_uniform_win_rate": random_uniform["win_rate"],
    }
    pressure_rank_key = {
        "track_ball_win_rate": track_ball["win_rate"],
        "negative_track_ball_loss_rate": -track_ball_loss_rate,
        "track_ball_truncation_rate": track_ball["truncation_rate"],
        "random_uniform_win_rate": random_uniform["win_rate"],
    }
    return {
        "track_ball_win_rate": track_ball["win_rate"],
        "track_ball_wins": track_ball["policy_wins"],
        "track_ball_loss_rate": track_ball_loss_rate,
        "track_ball_losses": track_ball["baseline_wins"],
        "track_ball_truncation_rate": track_ball["truncation_rate"],
        "track_ball_truncations": track_ball["truncations"],
        "track_ball_episodes": track_ball["episodes"],
        "random_uniform_win_rate": random_uniform["win_rate"],
        "random_uniform_wins": random_uniform["policy_wins"],
        "random_uniform_episodes": random_uniform["episodes"],
        "required_baseline_truncations": total_truncations,
        "required_baseline_episodes": total_episodes,
        "required_baseline_truncation_rate": truncation_rate,
        "rank_key": rank_key,
        "pressure_rank_key": pressure_rank_key,
    }


def _rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = candidate["metric_values"]
    return (
        float(metrics["track_ball_win_rate"]),
        -float(metrics["track_ball_loss_rate"]),
        float(metrics["track_ball_truncation_rate"]),
        float(metrics["random_uniform_win_rate"]),
        -float(candidate["spec_index"]),
    )


def _pressure_rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = candidate["metric_values"]
    return (
        float(metrics["track_ball_win_rate"]),
        -float(metrics["track_ball_loss_rate"]),
        float(metrics["track_ball_truncation_rate"]),
        float(metrics["random_uniform_win_rate"]),
        -float(candidate["spec_index"]),
    )


def _eval_split(summary: dict[str, Any]) -> dict[str, Any]:
    eval_split = summary.get("eval_split")
    if isinstance(eval_split, dict):
        return dict(eval_split)
    return {
        "split_id": None,
        "split_role": None,
        "seed_generation": None,
        "base_seed": summary.get("seed"),
        "paired_seat": True,
    }


def _policy_ids(row: dict[str, Any]) -> list[str]:
    policies = row.get("policies", [])
    if not isinstance(policies, list):
        return []
    return [policy for policy in policies if isinstance(policy, str)]


def _checkpoint_label(spec: dict[str, Any], policy_id: str) -> str | None:
    label = spec.get("checkpoint_label")
    if isinstance(label, str) and label:
        return label
    if policy_id.startswith("learned_"):
        return policy_id.removeprefix("learned_")
    return None


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"expected integer-like value, got {value!r}")


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return None


if __name__ == "__main__":
    main()
