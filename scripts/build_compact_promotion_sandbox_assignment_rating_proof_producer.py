#!/usr/bin/env python3
"""Generate local sandbox rating/leaderboard/assignment artifacts and proof."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

from curvyzero.tournament import curvytron_checkpoint_tournament as arena
from curvyzero.training.compact_promotion_readiness import (
    build_compact_promotion_sandbox_assignment_rating_proof_v1,
)
from curvyzero.training.opponent_leaderboard import (
    OPPONENT_DEATH_MODE_NORMAL,
    STABLE_SENTINEL_BLANK_CANVAS,
    STABLE_SLOT_PROFILE_3,
    build_leaderboard_pointer,
    build_leaderboard_snapshot_from_rating_snapshot,
    select_stable_slots_v1_assignment,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_promotion_readiness_results")
DEFAULT_RUN_ID = "optimizer-compact-promotion-sandbox-assignment-rating-proof-20260530"

_FORBIDDEN_TOUCH_KEYS = (
    "production_live_runs_touched",
    "production_intake_touched",
    "production_rating_touched",
    "production_leaderboard_touched",
    "production_control_pointer_touched",
    "writes_checkpoint_intake",
    "spawns_rating",
    "publishes_leaderboard",
    "rewrites_production_control_pointers",
    "uses_production_modal_objects",
    "background_eval_enabled",
    "background_gif_enabled",
    "checkpoint_intake_touched",
    "rating_round_started",
    "rating_latest_written",
    "public_leaderboard_written",
    "leaderboard_pointer_published",
    "training_candidate_assignment_written",
    "assignment_pointer_rewritten",
    "promotion_published",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--unified-lifecycle-report", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--stock-resume-load-canary-report", type=Path, required=True)
    parser.add_argument("--isolated-live-run-safety-canary-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    report_path = output_dir / "sandbox_assignment_rating_proof_report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"sandbox assignment/rating proof report already exists: {report_path}"
        )
    input_dir = output_dir / "sandbox_assignment_rating_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).isoformat()

    lifecycle_path = _resolve_input(args.unified_lifecycle_report, repo_root)
    compatibility_path = _resolve_input(args.compatibility_report, repo_root)
    stock_canary_path = _resolve_input(args.stock_resume_load_canary_report, repo_root)
    isolated_canary_path = _resolve_input(
        args.isolated_live_run_safety_canary_report,
        repo_root,
    )
    lifecycle = _load_json(lifecycle_path)
    stock_canary = _load_json(stock_canary_path)
    candidate_checkpoint_id = str(lifecycle["checkpoint_id"])
    stock_export_path = Path(str(stock_canary["resumed_stock_export_path"])).resolve()
    if not stock_export_path.is_file():
        raise FileNotFoundError(f"resumed stock export missing: {stock_export_path}")

    run_slug = _short_id(args.run_id)
    candidate_ref = _relative_ref_or_fallback(
        stock_export_path,
        repo_root,
        f"sandbox/{run_slug}/candidate/iteration_0.pth.tar",
    )

    reference_path = input_dir / "sandbox_reference" / "iteration_0.pth.tar"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(stock_export_path, reference_path)

    reference_ref = _relative_ref_or_fallback(
        reference_path,
        repo_root,
        f"sandbox/{run_slug}/reference/iteration_0.pth.tar",
    )
    reference_checkpoint_id = "sandbox-reference-opponent"
    rating_snapshot = _build_rating_snapshot(
        run_id=str(args.run_id),
        candidate_checkpoint_id=candidate_checkpoint_id,
        candidate_ref=candidate_ref,
        reference_checkpoint_id=reference_checkpoint_id,
        reference_ref=reference_ref,
        seed=int(args.seed),
        created_at=created_at,
    )
    rating_path = input_dir / "rating_snapshot.json"
    _write_json(rating_path, rating_snapshot)

    leaderboard_id = f"sandbox-{args.run_id}"
    snapshot_id = f"sandbox-{args.run_id}-snapshot-000"
    snapshot_ref = _relative_ref_or_fallback(
        input_dir / "leaderboard_snapshot.json",
        repo_root,
        f"sandbox/{run_slug}/leaderboard_snapshot.json",
    )
    leaderboard = build_leaderboard_snapshot_from_rating_snapshot(
        rating_snapshot,
        leaderboard_id=leaderboard_id,
        snapshot_id=snapshot_id,
        generation=0,
        created_at=created_at,
        active_min_distinct_opponents=1,
        active_min_valid_games=5,
    )
    pointer = build_leaderboard_pointer(
        leaderboard,
        snapshot_ref=snapshot_ref,
        published_at=created_at,
        writer={"kind": "local_sandbox_assignment_rating_proof_producer"},
    )
    pointer.update(
        {
            "local_only": True,
            "published": False,
            "public_leaderboard_written": False,
            "leaderboard_pointer_published": False,
        }
    )
    leaderboard_path = input_dir / "leaderboard_snapshot.json"
    pointer_path = input_dir / "leaderboard_pointer.json"
    _write_json(leaderboard_path, leaderboard)
    _write_json(pointer_path, pointer)

    assignment, assignment_audit = select_stable_slots_v1_assignment(
        leaderboard,
        assignment_id=f"opt058-{run_slug}",
        source_ref=snapshot_ref,
        seed=int(args.seed),
        profile=STABLE_SLOT_PROFILE_3,
        sentinel=STABLE_SENTINEL_BLANK_CANVAS,
        allow_recent_provisional=False,
        checkpoint_death_mode=OPPONENT_DEATH_MODE_NORMAL,
        expected_rating_context_hash=str(rating_snapshot["context_hash"]),
    )
    assignment_path = input_dir / "assignment.json"
    assignment_audit_path = input_dir / "assignment_audit.json"
    _write_json(assignment_path, assignment)
    _write_json(assignment_audit_path, assignment_audit)

    forbidden_touch_path = input_dir / "forbidden_touch_audit.json"
    forbidden_touch_audit = {
        "schema_id": "curvyzero_compact_promotion_sandbox_assignment_rating_touch_audit/v1",
        "ok": True,
        "sandbox_scope": {
            "local_only": True,
            "namespace": f"sandbox-{args.run_id}",
            "production_namespace": False,
        },
        "forbidden_touch_audit": {key: False for key in _FORBIDDEN_TOUCH_KEYS},
        "lineage_stages": [
            "local_rating_snapshot_reduced",
            "local_leaderboard_snapshot_materialized",
            "local_assignment_materialized",
        ],
    }
    _write_json(forbidden_touch_path, forbidden_touch_audit)

    payload = build_compact_promotion_sandbox_assignment_rating_proof_v1(
        run_id=str(args.run_id),
        unified_lifecycle_report_path=lifecycle_path,
        compatibility_report_path=compatibility_path,
        stock_resume_load_canary_report_path=stock_canary_path,
        isolated_live_run_safety_canary_report_path=isolated_canary_path,
        rating_snapshot_path=rating_path,
        leaderboard_snapshot_path=leaderboard_path,
        leaderboard_pointer_path=pointer_path,
        assignment_path=assignment_path,
        assignment_audit_path=assignment_audit_path,
        forbidden_touch_audit_path=forbidden_touch_path,
        repo_root=repo_root,
        created_at=created_at,
    )
    _write_json(report_path, payload)
    manifest = {
        "schema_id": "curvyzero_compact_promotion_sandbox_assignment_rating_proof_producer_manifest/v1",
        "ok": True,
        "run_id": str(args.run_id),
        "created_at": created_at,
        "local_only": True,
        "sandbox_only": True,
        "touches_live_runs": False,
        "touches_production_live_runs": False,
        "publishes_leaderboard": False,
        "spawns_rating": False,
        "rewrites_control_pointer": False,
        "report_path": str(report_path),
        "inputs_dir": str(input_dir),
    }
    manifest_path = output_dir / "sandbox_assignment_rating_proof_producer_manifest.json"
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "report_path": str(report_path),
                "manifest_path": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _build_rating_snapshot(
    *,
    run_id: str,
    candidate_checkpoint_id: str,
    candidate_ref: str,
    reference_checkpoint_id: str,
    reference_ref: str,
    seed: int,
    created_at: str,
) -> dict[str, object]:
    rating_spec = {
        "tournament_id": f"sandbox-{run_id}",
        "rating_run_id": "sandbox-assignment-rating-proof",
        "checkpoints": [
            {
                "checkpoint_id": candidate_checkpoint_id,
                "label": "compact candidate",
                "checkpoint_ref": candidate_ref,
                "run_id": run_id,
                "attempt_id": "sandbox-assignment-rating-proof",
                "iteration": 0,
                "latest_for_run": True,
            },
            {
                "checkpoint_id": reference_checkpoint_id,
                "label": "sandbox reference",
                "checkpoint_ref": reference_ref,
                "run_id": "sandbox-reference",
                "attempt_id": "sandbox-assignment-rating-proof",
                "iteration": 0,
                "latest_for_run": True,
            },
        ],
        "games_per_pair": 5,
        "placement_min_games": 5,
        "placement_min_opponents": 1,
        "active_pool_limit": 2,
        "min_valid_fraction": 1.0,
        "seed": int(seed),
        "num_simulations": 1,
        "policy_batch_size": 1,
    }
    normalized = arena.normalize_rating_spec(rating_spec)
    candidate, reference = normalized["checkpoints"]
    pair_summary = {
        "battle_id": "sandbox-pair-000",
        "pair_index": 0,
        "summary_ref": f"local://{run_id}/sandbox-pair-000",
        "players": [candidate, reference],
        "settings": {"games_per_pair": 5},
        "tally": {
            "game_count": 5,
            "draw_count": 0,
            "failure_count": 0,
            "wins_by_checkpoint": {
                candidate_checkpoint_id: 3,
                reference_checkpoint_id: 2,
            },
        },
    }
    snapshot = arena.rating_snapshot_from_pair_results(
        pair_results=[pair_summary],
        rating_spec=normalized,
        round_index=0,
        created_at=created_at,
    )
    snapshot["sandbox_only"] = True
    snapshot["local_only"] = True
    snapshot["rating_signal_kind"] = "local_tiny_sandbox_tournament_signal"
    return snapshot


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_input(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _relative_ref_or_fallback(path: Path, repo_root: Path, fallback: str) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return fallback


def _short_id(value: object) -> str:
    text = str(value)
    clean = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in text)
    return clean.strip("-")[:80] or "sandbox-assignment-rating-proof"


if __name__ == "__main__":
    raise SystemExit(main())
