"""Confirm a dummy survival checkpoint selection on a heldout split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from curvyzero.training.dummy_survival_eval import DEFAULT_POLICIES
from curvyzero.training.dummy_survival_eval import UNTRAINED_PLANNER_POLICY_ID
from curvyzero.training.dummy_survival_eval import run_dummy_survival_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-record", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--split-id", default="dummy_survival_heldout_v0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    confirmation = run_dummy_survival_selection_holdout(
        selection_record_path=args.selection_record,
        episodes=args.episodes,
        seed=args.seed,
        split_id=args.split_id,
        output_dir=args.output_dir,
    )
    print(json.dumps(confirmation, indent=2, sort_keys=True))


def run_dummy_survival_selection_holdout(
    *,
    selection_record_path: Path,
    episodes: int,
    seed: int,
    split_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    selection_record = json.loads(selection_record_path.read_text(encoding="utf-8"))
    checkpoint_paths = _checkpoint_policies_from_selection(selection_record)
    eval_summary = run_dummy_survival_eval(
        episodes=episodes,
        seed=seed,
        output_dir=output_dir,
        policies=DEFAULT_POLICIES,
        checkpoint_policies=tuple(f"learned:{path}" for path in checkpoint_paths),
        split_id=split_id,
        split_role="heldout",
    )

    selected_row = _row_for_checkpoint(
        eval_summary["table"],
        selection_record.get("selected_checkpoint_path"),
    )
    latest_row = _row_for_checkpoint(
        eval_summary["table"],
        selection_record.get("latest_checkpoint_path"),
    )
    planner_row = _row_for_policy(eval_summary["table"], UNTRAINED_PLANNER_POLICY_ID)
    confirmation = _confirmation_summary(
        selection_record_path=selection_record_path,
        selection_record=selection_record,
        eval_summary=eval_summary,
        selected_row=selected_row,
        latest_row=latest_row,
        planner_row=planner_row,
        output_dir=output_dir,
    )
    confirmation_path = output_dir / "holdout_confirmation.json"
    confirmation_path.write_text(
        json.dumps(confirmation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return confirmation


def _checkpoint_policies_from_selection(selection_record: dict[str, Any]) -> list[str]:
    paths = []
    for key in ("selected_checkpoint_path", "latest_checkpoint_path"):
        value = selection_record.get(key)
        if isinstance(value, str) and value and value not in paths:
            paths.append(value)
    if not paths:
        raise ValueError("selection record does not contain selected/latest checkpoint paths")
    return paths


def _row_for_checkpoint(table: list[dict[str, Any]], checkpoint_path: object) -> dict[str, Any] | None:
    if not isinstance(checkpoint_path, str):
        return None
    for row in table:
        if row.get("checkpoint_path") == checkpoint_path:
            return row
    return None


def _row_for_policy(table: list[dict[str, Any]], policy_id: str) -> dict[str, Any] | None:
    for row in table:
        if row.get("policy_id") == policy_id:
            return row
    return None


def _confirmation_summary(
    *,
    selection_record_path: Path,
    selection_record: dict[str, Any],
    eval_summary: dict[str, Any],
    selected_row: dict[str, Any] | None,
    latest_row: dict[str, Any] | None,
    planner_row: dict[str, Any] | None,
    output_dir: Path,
) -> dict[str, Any]:
    selected_vs_latest = _compare_rows(selected_row, latest_row)
    selected_vs_planner = _compare_rows(selected_row, planner_row)
    claim_status = (
        "confirmed"
        if selected_vs_latest == "better" and selected_vs_planner == "better"
        else "inconclusive"
    )
    return {
        "kind": "curvyzero_dummy_survival_selection_holdout",
        "schema_id": "dummy_survival_selection_holdout_v0",
        "note": (
            "Heldout confirmation for a preselected dummy survival checkpoint. "
            "Strictly requires selected to beat both latest and planner-only."
        ),
        "selection_record_path": str(selection_record_path),
        "selection_record": selection_record,
        "eval_split": eval_summary["eval_split"],
        "episodes_per_policy": eval_summary["episodes_per_policy"],
        "required_baselines": list(DEFAULT_POLICIES),
        "selected_checkpoint": selected_row,
        "latest_checkpoint": latest_row,
        "planner_only": planner_row,
        "selected_vs_latest": selected_vs_latest,
        "selected_vs_planner_only": selected_vs_planner,
        "claim_status": claim_status,
        "artifacts": {
            "summary_json": str(output_dir / "summary.json"),
            "episodes_jsonl": str(output_dir / "episodes.jsonl"),
            "holdout_confirmation_json": str(output_dir / "holdout_confirmation.json"),
        },
    }


def _compare_rows(left: dict[str, Any] | None, right: dict[str, Any] | None) -> str:
    if left is None or right is None:
        return "missing"
    left_key = _rank_key(left)
    right_key = _rank_key(right)
    if left_key > right_key:
        return "better"
    if left_key < right_key:
        return "worse"
    return "tied"


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(row["survival_rate"]),
        float(row["mean_steps"]),
        float(row["mean_terminal_reward"]),
        int(row["max_steps"]),
    )


if __name__ == "__main__":
    main()
