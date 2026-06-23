#!/usr/bin/env python3
"""Build the final no-promotion compact readiness bundle review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_promotion_readiness_bundle_review import (
    MATCHED_QUALITY_CANARY_SCALE_DECISION,
)
from curvyzero.training.compact_promotion_readiness_bundle_review import (
    build_compact_promotion_readiness_bundle_review_v1,
)
from curvyzero.training.compact_promotion_readiness_bundle_review import (
    validate_compact_promotion_readiness_bundle_review_v1,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_promotion_readiness_results")
DEFAULT_RUN_ID = "optimizer-compact-promotion-readiness-bundle-review-20260530a"
DEFAULT_COMPATIBILITY_REPORT = Path(
    "artifacts/local/curvytron_compact_coach_compatibility_results"
    "/optimizer-compact-coach-compatibility-after-speed-row-h100-threshold-20260530"
    "/compatibility_report.json"
)
DEFAULT_UNIFIED_LIFECYCLE_REPORT = Path(
    "artifacts/local/curvytron_compact_current_chain_eval_gif_tournament_results"
    "/optimizer-compact-unified-lifecycle-smoke-20260530"
    "/unified_lifecycle_report.json"
)
DEFAULT_MATCHED_LEARNING_QUALITY_REPORT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-matched-learning-quality-canary-current-env16train2-20260530"
    "/matched_learning_quality_canary_report.json"
)
DEFAULT_MATCHED_PAIR_VERIFICATION_REPORT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-matched-learning-quality-pair-verifier-current-env16train2-20260530"
    "/matched_pair_verification_report.json"
)
DEFAULT_STOCK_RESUME_LOAD_CANARY_REPORT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-promotion-stock-resume-load-canary-20260530"
    "/stock_resume_load_canary_report.json"
)
DEFAULT_ISOLATED_LIVE_RUN_SAFETY_CANARY_REPORT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-promotion-isolated-live-run-safety-canary-20260530d"
    "/isolated_live_run_safety_canary_report.json"
)
DEFAULT_SANDBOX_ASSIGNMENT_RATING_PROOF_REPORT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-promotion-sandbox-assignment-rating-proof-20260530a"
    "/sandbox_assignment_rating_proof_report.json"
)
DEFAULT_LONGER_HORIZON_LEARNING_METRICS_REPORT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-longer-horizon-learning-metrics-local-20260530a"
    "/longer_horizon_learning_metrics_report.json"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    payload = build_compact_promotion_readiness_bundle_review_v1(
        run_id=str(args.run_id),
        compatibility_report_path=_resolve_input(args.compatibility_report, repo_root),
        unified_lifecycle_report_path=_resolve_input(
            args.unified_lifecycle_report,
            repo_root,
        ),
        matched_learning_quality_report_path=_resolve_input(
            args.matched_learning_quality_report,
            repo_root,
        ),
        matched_pair_verification_report_path=_resolve_input(
            args.matched_pair_verification_report,
            repo_root,
        ),
        stock_resume_load_canary_report_path=_resolve_input(
            args.stock_resume_load_canary_report,
            repo_root,
        ),
        isolated_live_run_safety_canary_report_path=_resolve_input(
            args.isolated_live_run_safety_canary_report,
            repo_root,
        ),
        sandbox_assignment_rating_proof_report_path=_resolve_input(
            args.sandbox_assignment_rating_proof_report,
            repo_root,
        ),
        longer_horizon_learning_metrics_report_path=_resolve_input(
            args.longer_horizon_learning_metrics_report,
            repo_root,
        ),
        matched_quality_sufficiency_decision=str(
            args.matched_quality_sufficiency_decision
        ),
        matched_quality_sufficiency_rationale=args.matched_quality_sufficiency_rationale,
    )
    validate_compact_promotion_readiness_bundle_review_v1(payload)

    report_path = output_dir / "readiness_bundle_review_report.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(report_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_id": "curvyzero_compact_promotion_readiness_bundle_review_manifest/v1",
            "ok": True,
            "run_id": str(args.run_id),
            "report_path": str(report_path),
            "candidate_checkpoint_id": payload["candidate_checkpoint_id"],
            "status": payload["status"],
            "ready_for_manual_review": payload["review_decision"][
                "ready_for_manual_review"
            ],
            "promotion_claim": payload["review_decision"]["promotion_claim"],
            "automatic_promotion_allowed": payload["review_decision"][
                "automatic_promotion_allowed"
            ],
            "evidence_ref": payload["evidence_ref"],
            "input_reports": payload["input_reports"],
            "non_claims": payload["non_claims"],
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "run_id": str(args.run_id),
                "report_path": str(report_path),
                "manifest_path": str(manifest_path),
                "status": payload["status"],
                "ready_for_manual_review": payload["review_decision"][
                    "ready_for_manual_review"
                ],
                "promotion_claim": payload["review_decision"]["promotion_claim"],
            },
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--compatibility-report",
        type=Path,
        default=DEFAULT_COMPATIBILITY_REPORT,
    )
    parser.add_argument(
        "--unified-lifecycle-report",
        type=Path,
        default=DEFAULT_UNIFIED_LIFECYCLE_REPORT,
    )
    parser.add_argument(
        "--matched-learning-quality-report",
        type=Path,
        default=DEFAULT_MATCHED_LEARNING_QUALITY_REPORT,
    )
    parser.add_argument(
        "--matched-pair-verification-report",
        type=Path,
        default=DEFAULT_MATCHED_PAIR_VERIFICATION_REPORT,
    )
    parser.add_argument(
        "--stock-resume-load-canary-report",
        type=Path,
        default=DEFAULT_STOCK_RESUME_LOAD_CANARY_REPORT,
    )
    parser.add_argument(
        "--isolated-live-run-safety-canary-report",
        type=Path,
        default=DEFAULT_ISOLATED_LIVE_RUN_SAFETY_CANARY_REPORT,
    )
    parser.add_argument(
        "--sandbox-assignment-rating-proof-report",
        type=Path,
        default=DEFAULT_SANDBOX_ASSIGNMENT_RATING_PROOF_REPORT,
    )
    parser.add_argument(
        "--longer-horizon-learning-metrics-report",
        type=Path,
        default=DEFAULT_LONGER_HORIZON_LEARNING_METRICS_REPORT,
    )
    parser.add_argument(
        "--matched-quality-sufficiency-decision",
        default=MATCHED_QUALITY_CANARY_SCALE_DECISION,
    )
    parser.add_argument(
        "--matched-quality-sufficiency-rationale",
        action="append",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"bundle review output dir is not empty: {output_dir}")
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_input(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
