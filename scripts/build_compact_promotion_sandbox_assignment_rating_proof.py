#!/usr/bin/env python3
"""Package a sandbox rating-to-assignment proof from local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.compact_promotion_readiness import (
    build_compact_promotion_sandbox_assignment_rating_proof_v1,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_promotion_readiness_results")
DEFAULT_RUN_ID = "optimizer-compact-promotion-sandbox-assignment-rating-proof-20260530"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--unified-lifecycle-report", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--stock-resume-load-canary-report", type=Path, required=True)
    parser.add_argument("--isolated-live-run-safety-canary-report", type=Path, required=True)
    parser.add_argument("--rating-snapshot", type=Path, required=True)
    parser.add_argument("--leaderboard-snapshot", type=Path, required=True)
    parser.add_argument("--leaderboard-pointer", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-audit", type=Path, required=True)
    parser.add_argument("--forbidden-touch-audit", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "sandbox_assignment_rating_proof_report.json"
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"sandbox assignment/rating proof report already exists: {report_path}"
        )
    payload = build_compact_promotion_sandbox_assignment_rating_proof_v1(
        run_id=str(args.run_id),
        unified_lifecycle_report_path=_resolve_input(
            args.unified_lifecycle_report,
            repo_root,
        ),
        compatibility_report_path=_resolve_input(args.compatibility_report, repo_root),
        stock_resume_load_canary_report_path=_resolve_input(
            args.stock_resume_load_canary_report,
            repo_root,
        ),
        isolated_live_run_safety_canary_report_path=_resolve_input(
            args.isolated_live_run_safety_canary_report,
            repo_root,
        ),
        rating_snapshot_path=_resolve_input(args.rating_snapshot, repo_root),
        leaderboard_snapshot_path=_resolve_input(args.leaderboard_snapshot, repo_root),
        leaderboard_pointer_path=_resolve_input(args.leaderboard_pointer, repo_root),
        assignment_path=_resolve_input(args.assignment, repo_root),
        assignment_audit_path=_resolve_input(args.assignment_audit, repo_root),
        forbidden_touch_audit_path=_resolve_input(args.forbidden_touch_audit, repo_root),
        repo_root=repo_root,
    )
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "report_path": str(report_path)}, sort_keys=True))
    return 0


def _resolve_input(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
