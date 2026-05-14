#!/usr/bin/env python3
"""Materialize local leaderboard and assignment artifacts from rating JSON.

This is a local bridge tool. It does not read or write Modal directly. Use it to
turn an exported tournament rating snapshot/API payload into:

- public leaderboard snapshot JSON;
- live pointer JSON;
- trainer assignment JSON;
- assignment audit JSON.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.training.opponent_leaderboard import (
    build_leaderboard_pointer,
    build_leaderboard_snapshot_from_rating_snapshot,
    select_opponent_assignment_from_leaderboard,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _rating_snapshot_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("ratings"), list):
        return payload
    if isinstance(payload.get("rows"), list):
        return {
            "schema_id": "curvyzero_curvytron_checkpoint_rating_snapshot/v0",
            "formula_version": payload.get("formula_version") or "unknown_from_api_payload",
            "tournament_id": payload.get("selected_tournament_id"),
            "rating_run_id": payload.get("rating_run_id"),
            "ratings_ref": payload.get("ratings_ref"),
            "context_hash": payload.get("context_hash"),
            "roster_hash": payload.get("roster_hash"),
            "created_at": payload.get("created_at"),
            "ratings": payload["rows"],
        }
    raise ValueError("input JSON must contain 'ratings' or API 'rows'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rating_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--leaderboard-id", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--snapshot-ref", required=True)
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument("--assignment-source-ref", default="")
    parser.add_argument("--published-at", default="")
    parser.add_argument("--created-at", default="")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-slots", type=int, default=5)
    parser.add_argument("--no-blank-sentinel", action="store_true")
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args()

    rating_payload = _rating_snapshot_from_payload(_load_json(args.rating_json))
    timestamp = args.created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        rating_payload,
        leaderboard_id=args.leaderboard_id,
        snapshot_id=args.snapshot_id,
        generation=args.generation,
        created_at=timestamp,
    )
    pointer = build_leaderboard_pointer(
        snapshot,
        snapshot_ref=args.snapshot_ref,
        published_at=args.published_at or timestamp,
        writer={"kind": "local_materialize_cli"},
    )
    assignment_source_ref = args.assignment_source_ref or args.snapshot_ref
    assignment, audit = select_opponent_assignment_from_leaderboard(
        snapshot,
        assignment_id=args.assignment_id,
        source_ref=assignment_source_ref,
        seed=args.seed,
        max_slots=args.max_slots,
        include_blank_sentinel=not args.no_blank_sentinel,
        allow_provisional=args.allow_provisional,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "leaderboard_snapshot.json": snapshot,
        "leaderboard_pointer.json": pointer,
        "assignment.json": assignment,
        "audit.json": audit,
    }
    for filename, payload in outputs.items():
        (args.output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({name: (args.output_dir / name).as_posix() for name in outputs}, indent=2))


if __name__ == "__main__":
    main()
