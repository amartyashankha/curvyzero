#!/usr/bin/env python3
"""Build a reciprocal verifier for an existing matched learning-quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.compact_promotion_readiness_learning_quality import (
    build_compact_matched_learning_quality_pair_verification_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    validate_compact_matched_learning_quality_pair_verification_v1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
)
DEFAULT_RUN_ID = "optimizer-compact-matched-learning-quality-pair-verifier-20260530"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--matched-learning-quality-report", type=Path, required=True)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "matched_pair_verification_report.json"
    manifest_path = output_dir / "manifest.json"

    payload = build_compact_matched_learning_quality_pair_verification_v1(
        matched_learning_quality_report_path=_resolve_input(
            args.matched_learning_quality_report,
            repo_root,
        ),
    )
    validate_compact_matched_learning_quality_pair_verification_v1(payload)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": (
                    "curvyzero_compact_matched_learning_quality_pair_verifier_manifest/v1"
                ),
                "ok": True,
                "run_id": str(args.run_id),
                "builder_scope": "matched_quality_report_to_pair_verification",
                "matched_learning_quality_report": payload[
                    "matched_learning_quality_report"
                ],
                "verification_report_path": str(report_path),
                "non_claims": payload["non_claims"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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


def _resolve_input(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
