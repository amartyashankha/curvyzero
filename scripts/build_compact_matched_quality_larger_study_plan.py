#!/usr/bin/env python3
"""Build the larger same-surface matched-quality study launch plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_BUNDLE_REVIEW_REPORT,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_COMPACT_ENV_STEPS,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_COMPACT_REPLAY_PAIR_CAPACITY,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_COMPACT_SAMPLE_BATCH_SIZE,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_COMPACT_TRAIN_STEPS,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_EVAL_SEED_RNG_SEED,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_MIN_EVAL_MAX_STEPS,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_MIN_EVAL_SEED_COUNT,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_STOCK_MAX_ENV_STEP,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_STOCK_MAX_TRAIN_ITER,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    DEFAULT_STUDY_RUN_STAMP,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    build_compact_matched_quality_larger_study_plan_v1,
)
from curvyzero.training.compact_matched_quality_study_plan import (
    validate_compact_matched_quality_larger_study_plan_v1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_matched_quality_study_plan_results"
)
DEFAULT_RUN_ID = "optimizer-compact-matched-quality-larger-study-plan-20260531"
MANIFEST_SCHEMA_ID = "curvyzero_compact_matched_quality_larger_study_plan_manifest/v1"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    payload = build_compact_matched_quality_larger_study_plan_v1(
        run_id=str(args.run_id),
        bundle_review_report_path=_resolve(args.bundle_review_report, repo_root),
        min_eval_seed_count=int(args.min_eval_seed_count),
        min_eval_max_steps=int(args.min_eval_max_steps),
        stock_max_env_step=int(args.stock_max_env_step),
        stock_max_train_iter=int(args.stock_max_train_iter),
        compact_env_steps=int(args.compact_env_steps),
        compact_train_steps=int(args.compact_train_steps),
        compact_sample_batch_size=int(args.compact_sample_batch_size),
        compact_replay_pair_capacity=int(args.compact_replay_pair_capacity),
        eval_seed_rng_seed=int(args.eval_seed_rng_seed),
        study_run_stamp=str(args.study_run_stamp),
    )
    validate_compact_matched_quality_larger_study_plan_v1(payload)

    report_path = output_dir / "larger_study_plan_report.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(report_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_id": MANIFEST_SCHEMA_ID,
            "ok": payload["ok"],
            "run_id": str(args.run_id),
            "report_path": str(report_path),
            "status": payload["status"],
            "candidate_checkpoint_id": payload["candidate_checkpoint_id"],
            "study_run_stamp": payload["study_run_stamp"],
            "study_requirements": payload["study_requirements"],
            "required_output_artifacts": payload["required_output_artifacts"],
            "evidence_ref": payload["evidence_ref"],
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
                "status": payload["status"],
                "study_run_stamp": payload["study_run_stamp"],
                "required_output_artifacts": payload["required_output_artifacts"],
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
        "--bundle-review-report",
        type=Path,
        default=DEFAULT_BUNDLE_REVIEW_REPORT,
    )
    parser.add_argument("--min-eval-seed-count", type=int, default=DEFAULT_MIN_EVAL_SEED_COUNT)
    parser.add_argument("--min-eval-max-steps", type=int, default=DEFAULT_MIN_EVAL_MAX_STEPS)
    parser.add_argument("--stock-max-env-step", type=int, default=DEFAULT_STOCK_MAX_ENV_STEP)
    parser.add_argument(
        "--stock-max-train-iter",
        type=int,
        default=DEFAULT_STOCK_MAX_TRAIN_ITER,
    )
    parser.add_argument("--compact-env-steps", type=int, default=DEFAULT_COMPACT_ENV_STEPS)
    parser.add_argument("--compact-train-steps", type=int, default=DEFAULT_COMPACT_TRAIN_STEPS)
    parser.add_argument(
        "--compact-sample-batch-size",
        type=int,
        default=DEFAULT_COMPACT_SAMPLE_BATCH_SIZE,
    )
    parser.add_argument(
        "--compact-replay-pair-capacity",
        type=int,
        default=DEFAULT_COMPACT_REPLAY_PAIR_CAPACITY,
    )
    parser.add_argument("--eval-seed-rng-seed", type=int, default=DEFAULT_EVAL_SEED_RNG_SEED)
    parser.add_argument("--study-run-stamp", default=DEFAULT_STUDY_RUN_STAMP)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"refusing to overwrite existing output directory: {output_dir}"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
