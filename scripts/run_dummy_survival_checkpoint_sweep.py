"""Sweep dummy survival checkpoints against fixed eval baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from curvyzero.training.dummy_survival_eval import DEFAULT_POLICIES
from curvyzero.training.dummy_survival_eval import run_dummy_survival_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split-id", default="dummy_survival_monitor_v0")
    parser.add_argument(
        "--split-role",
        default="monitor",
        choices=["monitor", "selection", "heldout", "debug", "train"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
        choices=list(DEFAULT_POLICIES),
    )
    args = parser.parse_args()

    summary = run_dummy_survival_checkpoint_sweep(
        checkpoint_dir=args.checkpoint_dir,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        policies=tuple(args.policies),
        split_id=args.split_id,
        split_role=args.split_role,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_dummy_survival_checkpoint_sweep(
    *,
    checkpoint_dir: Path,
    episodes: int,
    seed: int,
    output_dir: Path,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
    split_id: str = "dummy_survival_monitor_v0",
    split_role: str = "monitor",
) -> dict[str, Any]:
    checkpoint_paths = sorted(checkpoint_dir.glob("*.npz"))
    if not checkpoint_paths:
        raise ValueError(f"no .npz checkpoints found in {checkpoint_dir}")

    eval_summary = run_dummy_survival_eval(
        episodes=episodes,
        seed=seed,
        output_dir=None,
        policies=policies,
        checkpoint_policies=tuple(f"learned:{path}" for path in checkpoint_paths),
        split_id=split_id,
        split_role=split_role,
    )
    table = list(eval_summary["table"])
    baseline_table = [row for row in table if row["policy_kind"] == "baseline"]
    checkpoint_table = [row for row in table if row["policy_kind"] == "learned_checkpoint"]
    checkpoint_table.sort(key=_checkpoint_rank_key, reverse=True)
    for rank, row in enumerate(checkpoint_table, start=1):
        row["rank"] = rank

    artifacts = {
        "summary_json": output_dir / "summary.json",
        "checkpoint_eval_jsonl": output_dir / "checkpoint_eval.jsonl",
        "best_checkpoint_json": output_dir / "best_checkpoint.json",
        "best_checkpoint_path_txt": output_dir / "best_checkpoint_path.txt",
        "selection_record_json": output_dir / "selection_record.json",
    }
    best_checkpoint = checkpoint_table[0] if checkpoint_table else None
    latest_checkpoint = _latest_checkpoint_row(
        checkpoint_paths=checkpoint_paths,
        checkpoint_table=checkpoint_table,
    )
    selection_record = _selection_record(
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        split_id=split_id,
        split_role=split_role,
        policies=policies,
    )
    summary: dict[str, Any] = {
        "kind": "curvyzero_dummy_survival_checkpoint_sweep",
        "note": "Checkpoint selection sweep using the existing dummy survival learned-checkpoint evaluator.",
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_count": len(checkpoint_paths),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "seed": seed,
        "episodes_per_policy": episodes,
        "policies": list(policies),
        "eval_split": eval_summary["eval_split"],
        "selection_metric": "survival_rate_then_mean_steps_then_terminal_reward_then_max_steps",
        "baseline_table": baseline_table,
        "checkpoint_table": checkpoint_table,
        "best_checkpoint": best_checkpoint,
        "selected_checkpoint": best_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "selection_record": selection_record,
        "heldout_required": split_role == "selection",
        "eval": eval_summary["eval"],
        "config": eval_summary["config"],
        "seed_handling": eval_summary["seed_handling"],
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    _write_sweep_artifacts(
        artifacts=artifacts,
        summary=summary,
        checkpoint_table=checkpoint_table,
        best_checkpoint=best_checkpoint,
        selection_record=selection_record,
    )
    return summary


def _checkpoint_rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["survival_rate"]),
        float(row["mean_steps"]),
        float(row["mean_terminal_reward"]),
        int(row["max_steps"]),
    )


def _latest_checkpoint_row(
    *,
    checkpoint_paths: list[Path],
    checkpoint_table: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not checkpoint_paths:
        return None
    latest_path = str(checkpoint_paths[-1])
    for row in checkpoint_table:
        if row.get("checkpoint_path") == latest_path:
            return row
    return None


def _selection_record(
    *,
    best_checkpoint: dict[str, Any] | None,
    latest_checkpoint: dict[str, Any] | None,
    split_id: str,
    split_role: str,
    policies: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "selection_record_schema_id": "dummy_survival_selection_record_v0",
        "selection_split_id": split_id,
        "selection_split_role": split_role,
        "selection_metric": "survival_rate_then_mean_steps_then_terminal_reward_then_max_steps",
        "required_baselines": list(policies),
        "selected_checkpoint_id": None if best_checkpoint is None else best_checkpoint["checkpoint_id"],
        "selected_checkpoint_path": None if best_checkpoint is None else best_checkpoint["checkpoint_path"],
        "selected_metric_values": None if best_checkpoint is None else _metric_values(best_checkpoint),
        "latest_checkpoint_id": None if latest_checkpoint is None else latest_checkpoint["checkpoint_id"],
        "latest_checkpoint_path": None if latest_checkpoint is None else latest_checkpoint["checkpoint_path"],
        "latest_metric_values": None if latest_checkpoint is None else _metric_values(latest_checkpoint),
        "heldout_required": split_role == "selection",
        "claim_status": "selected_pending_heldout" if split_role == "selection" else "monitor_only",
    }


def _metric_values(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "survival_rate": row["survival_rate"],
        "mean_steps": row["mean_steps"],
        "mean_terminal_reward": row["mean_terminal_reward"],
        "max_steps": row["max_steps"],
    }


def _write_sweep_artifacts(
    *,
    artifacts: dict[str, Path],
    summary: dict[str, Any],
    checkpoint_table: list[dict[str, Any]],
    best_checkpoint: dict[str, Any] | None,
    selection_record: dict[str, Any],
) -> None:
    summary_path = artifacts["summary_json"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with artifacts["checkpoint_eval_jsonl"].open("w", encoding="utf-8") as handle:
        for row in checkpoint_table:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    artifacts["best_checkpoint_json"].write_text(
        json.dumps(best_checkpoint or {}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts["best_checkpoint_path_txt"].write_text(
        (str(best_checkpoint["checkpoint_path"]) if best_checkpoint else "") + "\n",
        encoding="utf-8",
    )
    artifacts["selection_record_json"].write_text(
        json.dumps(selection_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
