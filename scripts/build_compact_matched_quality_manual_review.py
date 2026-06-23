#!/usr/bin/env python3
"""Build the OPT-070 larger matched-quality manual-review decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_matched_quality_manual_review import (
    COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_MANIFEST_SCHEMA_ID,
)
from curvyzero.training.compact_matched_quality_manual_review import (
    DECISION_KEEP_COMPACT_CANDIDATE_ONLY,
)
from curvyzero.training.compact_matched_quality_manual_review import (
    DECISION_REQUIRE_LARGER_REPEAT,
)
from curvyzero.training.compact_matched_quality_manual_review import (
    DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL,
)
from curvyzero.training.compact_matched_quality_manual_review import DEFAULT_RUN_ID
from curvyzero.training.compact_matched_quality_manual_review import (
    build_compact_matched_quality_manual_review_v1,
)
from curvyzero.training.compact_matched_quality_manual_review import (
    validate_compact_matched_quality_manual_review_v1,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_promotion_readiness_results")
DEFAULT_SUFFICIENCY_REVIEW = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-matched-quality-sufficiency-review-larger-2048x32-env64train8-20260531"
    "/matched_quality_sufficiency_review_report.json"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    payload = build_compact_matched_quality_manual_review_v1(
        run_id=str(args.run_id),
        sufficiency_review_path=_resolve(args.sufficiency_review, repo_root),
        manual_review_decision=str(args.manual_review_decision),
        next_allowed_step=str(args.next_allowed_step),
        named_non_production_proposal_id=args.named_non_production_proposal_id,
        reviewer_id=str(args.reviewer_id),
        reviewer_mode=str(args.reviewer_mode),
        canonical_stock_failure_run_id=str(args.canonical_stock_failure_run_id),
        accepted_stock_capture_run_id=str(args.accepted_stock_capture_run_id),
        canonical_stock_failure_acknowledged=(
            not bool(args.no_canonical_stock_failure_acknowledgement)
        ),
    )
    validate_compact_matched_quality_manual_review_v1(payload)

    report_path = output_dir / "manual_review_decision_report.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(report_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_id": COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_MANIFEST_SCHEMA_ID,
            "ok": payload["ok"],
            "run_id": str(args.run_id),
            "report_path": str(report_path),
            "status": payload["status"],
            "opt_id": payload["opt_id"],
            "candidate_checkpoint_id": payload["candidate_checkpoint_id"],
            "manual_review_scope": payload["manual_review_scope"],
            "reviewed_packet": payload["reviewed_packet"],
            "manual_decision": payload["manual_decision"],
            "promotion_claim": payload["non_claims"]["promotion_claim"],
            "automatic_promotion_allowed": payload["non_claims"][
                "automatic_promotion_allowed"
            ],
            "evidence_ref": payload["evidence_ref"],
            "source_sufficiency_review": payload["source_sufficiency_review"],
            "input_packet_refs": payload["input_packet_refs"],
            "non_claims": payload["non_claims"],
        },
    )
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "run_id": str(args.run_id),
                "report_path": str(report_path),
                "manifest_path": str(manifest_path),
                "manual_review_decision": payload["manual_decision"][
                    "manual_review_decision"
                ],
                "next_allowed_step": payload["manual_decision"][
                    "next_allowed_step"
                ],
                "promotion_claim": payload["non_claims"]["promotion_claim"],
                "automatic_promotion_allowed": payload["non_claims"][
                    "automatic_promotion_allowed"
                ],
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
        "--sufficiency-review",
        type=Path,
        default=DEFAULT_SUFFICIENCY_REVIEW,
    )
    parser.add_argument(
        "--manual-review-decision",
        choices=(
            DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL,
            DECISION_REQUIRE_LARGER_REPEAT,
            DECISION_KEEP_COMPACT_CANDIDATE_ONLY,
        ),
        default=DECISION_KEEP_COMPACT_CANDIDATE_ONLY,
    )
    parser.add_argument(
        "--next-allowed-step",
        default="return_to_engineering_speed_and_stock_evaluator_hardening",
    )
    parser.add_argument("--named-non-production-proposal-id")
    parser.add_argument("--reviewer-id", default="optimizer-main-thread-manual-review")
    parser.add_argument("--reviewer-mode", default="main_thread_manual_policy_review")
    parser.add_argument(
        "--canonical-stock-failure-run-id",
        default="optimizer-stock-reference-quality-producer-larger-2048x32-20260531",
    )
    parser.add_argument(
        "--accepted-stock-capture-run-id",
        default=(
            "optimizer-stock-reference-quality-producer-larger-2048x32-"
            "20260531-evalenv32"
        ),
    )
    parser.add_argument(
        "--no-canonical-stock-failure-acknowledgement",
        action="store_true",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"manual-review output dir is not empty: {output_dir}"
            )
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
