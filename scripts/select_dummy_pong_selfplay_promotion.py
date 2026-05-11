"""Select a Pong self-play child checkpoint only if it beats its parent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROMOTION_RECORD_SCHEMA_ID = "dummy_pong_selfplay_promotion_record_v0"
PROMOTION_RULE_ID = "parent_vs_child_win_margin_v0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-child-win-margin", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing promotion record. The input summary is never overwritten.",
    )
    args = parser.parse_args()

    output_path = _record_output_path(
        summary_path=args.summary,
        output_path=args.output_path,
        output_dir=args.output_dir,
    )
    record = select_dummy_pong_selfplay_promotion(
        summary_path=args.summary,
        parent_checkpoint=args.parent_checkpoint,
        min_child_win_margin=args.min_child_win_margin,
        output_path=output_path,
    )
    _write_record(record=record, output_path=output_path, force=args.force)
    print(json.dumps(record, indent=2, sort_keys=True))


def select_dummy_pong_selfplay_promotion(
    *,
    summary_path: Path,
    parent_checkpoint: Path,
    min_child_win_margin: int,
    output_path: Path,
) -> dict[str, Any]:
    if min_child_win_margin < 1:
        raise ValueError("min_child_win_margin must be at least 1")

    summary_bytes = summary_path.read_bytes()
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    summary = json.loads(summary_bytes.decode("utf-8"))
    policies = _checkpoint_policies(summary)
    parent_policy = _policy_for_checkpoint(policies, parent_checkpoint)
    children = _children_for_parent(policies, parent_checkpoint, parent_policy)
    candidates = [
        _candidate_record(
            summary=summary,
            parent_policy=parent_policy,
            child_policy=child_policy,
            min_child_win_margin=min_child_win_margin,
        )
        for child_policy in children
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    candidates.sort(key=_candidate_rank_key, reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
    passing_candidates = [candidate for candidate in candidates if candidate["passes_gate"]]
    selected = passing_candidates[0] if passing_candidates else None
    return {
        "kind": "curvyzero_dummy_pong_selfplay_promotion",
        "promotion_record_schema_id": PROMOTION_RECORD_SCHEMA_ID,
        "promotion_rule": {
            "rule_id": PROMOTION_RULE_ID,
            "description": (
                "Promote a child checkpoint only if it beats the parent checkpoint "
                "in the learned-vs-learned scoreboard row by the required win margin."
            ),
            "min_child_win_margin": min_child_win_margin,
        },
        "source_summary": {
            "path": str(summary_path),
            "sha256": summary_sha256,
            "kind": summary.get("kind"),
            "scoreboard_schema_id": summary.get("scoreboard_schema_id"),
            "seed": summary.get("seed"),
            "episodes_per_match": summary.get("episodes_per_match"),
            "total_episodes": summary.get("total_episodes"),
        },
        "parent_checkpoint_path": str(parent_checkpoint),
        "parent_policy_id": parent_policy["policy_id"],
        "candidate_count": len(candidates),
        "passed_candidate_count": len(passing_candidates),
        "promoted": selected is not None,
        "selected_checkpoint_label": None if selected is None else selected["checkpoint_label"],
        "selected_checkpoint_path": None if selected is None else selected["checkpoint_path"],
        "selected_policy_id": None if selected is None else selected["policy_id"],
        "selected_candidate": selected,
        "ranked_candidates": candidates,
        "claim_status": "promoted_pending_heldout" if selected is not None else "no_promotion",
        "artifacts": {
            "promotion_record_json": str(output_path),
            "source_summary_json": str(summary_path),
        },
    }


def _checkpoint_policies(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    specs_by_policy = {
        str(spec.get("policy_id")): spec
        for spec in summary.get("checkpoint_specs", [])
        if isinstance(spec, dict) and spec.get("policy_id")
    }
    for policy in summary.get("checkpoint_policies", []):
        if not isinstance(policy, dict):
            continue
        policy_id = policy.get("policy_id")
        if not isinstance(policy_id, str) or not policy_id:
            continue
        payload = dict(policy)
        if policy_id in specs_by_policy:
            payload["checkpoint_label"] = specs_by_policy[policy_id].get("checkpoint_label")
            payload["checkpoint_arg"] = specs_by_policy[policy_id].get("checkpoint_arg")
        policies[policy_id] = payload
    return policies


def _policy_for_checkpoint(
    policies: dict[str, dict[str, Any]],
    checkpoint: Path,
) -> dict[str, Any]:
    checkpoint_text = str(checkpoint)
    matches = [
        policy
        for policy in policies.values()
        if str(policy.get("checkpoint_path")) == checkpoint_text
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one policy for checkpoint {checkpoint_text!r}, got {len(matches)}"
        )
    return matches[0]


def _children_for_parent(
    policies: dict[str, dict[str, Any]],
    parent_checkpoint: Path,
    parent_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    parent_path = str(parent_checkpoint)
    children = []
    for policy in policies.values():
        policy_id = policy.get("policy_id")
        if policy_id == parent_policy["policy_id"]:
            continue
        metadata = policy.get("checkpoint_metadata", {})
        if not isinstance(metadata, dict):
            continue
        if metadata.get("initial_checkpoint") == parent_path:
            children.append(policy)
    if not children:
        raise ValueError(f"no child checkpoints point at parent {parent_path!r}")
    return children


def _candidate_record(
    *,
    summary: dict[str, Any],
    parent_policy: dict[str, Any],
    child_policy: dict[str, Any],
    min_child_win_margin: int,
) -> dict[str, Any] | None:
    parent_policy_id = str(parent_policy["policy_id"])
    child_policy_id = str(child_policy["policy_id"])
    row = _learned_vs_learned_row(summary, parent_policy_id, child_policy_id)
    if row is None:
        return None
    wins_by_policy = row.get("wins_by_policy", {})
    if not isinstance(wins_by_policy, dict):
        wins_by_policy = {}
    child_wins = _int_value(wins_by_policy.get(child_policy_id, 0))
    parent_wins = _int_value(wins_by_policy.get(parent_policy_id, 0))
    margin = child_wins - parent_wins
    episodes = _int_value(row.get("episodes", 0))
    passes_gate = margin >= min_child_win_margin
    return {
        "checkpoint_label": _checkpoint_label(child_policy),
        "checkpoint_path": child_policy.get("checkpoint_path"),
        "policy_id": child_policy_id,
        "parent_policy_id": parent_policy_id,
        "parent_checkpoint_path": parent_policy.get("checkpoint_path"),
        "episodes": episodes,
        "child_wins": child_wins,
        "parent_wins": parent_wins,
        "draw_or_truncation_count": max(0, episodes - child_wins - parent_wins),
        "child_win_margin": margin,
        "passes_gate": passes_gate,
        "scoreboard_row": row,
    }


def _learned_vs_learned_row(
    summary: dict[str, Any],
    parent_policy_id: str,
    child_policy_id: str,
) -> dict[str, Any] | None:
    expected = {parent_policy_id, child_policy_id}
    for row in summary.get("scoreboard_rows", []):
        if not isinstance(row, dict):
            continue
        if row.get("row_kind") != "learned_vs_learned":
            continue
        policies = row.get("policies", [])
        if isinstance(policies, list) and set(policies) == expected:
            return dict(row)
    return None


def _candidate_rank_key(candidate: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(candidate["passes_gate"]),
        int(candidate["child_win_margin"]),
        int(candidate["child_wins"]),
    )


def _checkpoint_label(policy: dict[str, Any]) -> str | None:
    label = policy.get("checkpoint_label")
    if isinstance(label, str) and label:
        return label
    policy_id = str(policy.get("policy_id", ""))
    if policy_id.startswith("learned_"):
        return policy_id.removeprefix("learned_")
    return None


def _record_output_path(
    *,
    summary_path: Path,
    output_path: Path | None,
    output_dir: Path | None,
) -> Path:
    if output_path is not None and output_dir is not None:
        raise ValueError("use either --output-path or --output-dir, not both")
    if output_path is not None:
        return output_path
    if output_dir is not None:
        return output_dir / "promotion_record.json"
    return summary_path.parent / "promotion_record.json"


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
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return 0


if __name__ == "__main__":
    main()
