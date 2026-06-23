#!/usr/bin/env python3
"""Build the matched-quality sufficiency decision after the readiness bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_matched_quality_sufficiency_review import (
    DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    DEFAULT_LARGER_DENOMINATOR_ID,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    DEFAULT_LARGER_QUALITY_HORIZON,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    DEFAULT_STUDY_ID,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    build_compact_matched_quality_sufficiency_review_v1,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    validate_compact_matched_quality_sufficiency_review_v1,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_promotion_readiness_results")
DEFAULT_RUN_ID = "optimizer-compact-matched-quality-sufficiency-review-20260531a"
DEFAULT_READINESS_BUNDLE_REVIEW = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-promotion-readiness-bundle-review-20260530a"
    "/readiness_bundle_review_report.json"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    payload = build_compact_matched_quality_sufficiency_review_v1(
        run_id=str(args.run_id),
        readiness_bundle_review_path=_resolve_input(
            args.readiness_bundle_review,
            repo_root,
        ),
        sufficiency_decision=str(args.sufficiency_decision),
        reviewer_id=str(args.reviewer_id),
        reviewer_mode=str(args.reviewer_mode),
        next_non_production_step=args.next_non_production_step,
        larger_study_id=str(args.larger_study_id),
        min_eval_seed_count=int(args.min_eval_seed_count),
        min_eval_max_steps=int(args.min_eval_max_steps),
        stock_reference_min_max_env_step=int(args.stock_reference_min_max_env_step),
        stock_reference_min_max_train_iter=int(args.stock_reference_min_max_train_iter),
        compact_candidate_min_env_steps=int(args.compact_candidate_min_env_steps),
        compact_candidate_min_train_steps=int(args.compact_candidate_min_train_steps),
        eval_seed_rng_seed=int(args.eval_seed_rng_seed),
        larger_denominator_id=str(args.larger_denominator_id),
        larger_quality_horizon=str(args.larger_quality_horizon),
    )
    validate_compact_matched_quality_sufficiency_review_v1(payload)

    report_path = output_dir / "matched_quality_sufficiency_review_report.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(report_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_id": "curvyzero_compact_matched_quality_sufficiency_review_manifest/v1",
            "ok": True,
            "run_id": str(args.run_id),
            "report_path": str(report_path),
            "candidate_checkpoint_id": payload["candidate_checkpoint_id"],
            "status": payload["status"],
            "decision": payload["decision"],
            "promotion_claim": payload["decision"]["promotion_claim"],
            "automatic_promotion_allowed": payload["decision"][
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
                "decision": payload["decision"][
                    "matched_quality_sufficiency_decision"
                ],
                "promotion_claim": payload["decision"]["promotion_claim"],
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
        "--readiness-bundle-review",
        type=Path,
        default=DEFAULT_READINESS_BUNDLE_REVIEW,
    )
    parser.add_argument(
        "--sufficiency-decision",
        choices=(
            DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP,
            DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY,
        ),
        default=DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY,
    )
    parser.add_argument("--reviewer-id", default="optimizer-main-thread-policy-review")
    parser.add_argument("--reviewer-mode", default="main_thread_policy_review")
    parser.add_argument("--next-non-production-step")
    parser.add_argument(
        "--larger-study-id",
        default=DEFAULT_STUDY_ID,
    )
    parser.add_argument("--min-eval-seed-count", type=int, default=32)
    parser.add_argument("--min-eval-max-steps", type=int, default=2048)
    parser.add_argument("--stock-reference-min-max-env-step", type=int, default=2048)
    parser.add_argument("--stock-reference-min-max-train-iter", type=int, default=4)
    parser.add_argument("--compact-candidate-min-env-steps", type=int, default=64)
    parser.add_argument("--compact-candidate-min-train-steps", type=int, default=8)
    parser.add_argument("--eval-seed-rng-seed", type=int, default=20260833)
    parser.add_argument(
        "--larger-denominator-id",
        default=DEFAULT_LARGER_DENOMINATOR_ID,
    )
    parser.add_argument(
        "--larger-quality-horizon",
        default=DEFAULT_LARGER_QUALITY_HORIZON,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"matched-quality sufficiency output dir is not empty: {output_dir}"
            )
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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
