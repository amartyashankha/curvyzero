#!/usr/bin/env python3
"""Build a read-only execution preflight for the larger matched-quality study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_matched_quality_larger_study_preflight import (
    DEFAULT_BUNDLE_REPORT,
)
from curvyzero.training.compact_matched_quality_larger_study_preflight import (
    build_compact_matched_quality_larger_study_preflight_v1,
)
from curvyzero.training.compact_matched_quality_larger_study_preflight import (
    validate_compact_matched_quality_larger_study_preflight_v1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_matched_quality_study_preflight_results"
)
DEFAULT_RUN_ID = "optimizer-compact-matched-quality-larger-study-preflight-20260531"
MANIFEST_SCHEMA_ID = (
    "curvyzero_compact_matched_quality_larger_study_preflight_manifest/v1"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    payload = build_compact_matched_quality_larger_study_preflight_v1(
        run_id=str(args.run_id),
        bundle_report_path=_resolve(args.bundle_report, repo_root),
        repo_root=repo_root,
    )
    validate_compact_matched_quality_larger_study_preflight_v1(payload)

    report_path = output_dir / "larger_study_preflight_report.json"
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
            "commands_run_by_preflight": payload["execution_readiness"][
                "commands_run_by_preflight"
            ],
            "quality_evidence_produced": payload["execution_readiness"][
                "quality_evidence_produced"
            ],
            "recommended_first_wave": payload["execution_readiness"][
                "recommended_first_wave"
            ],
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
                "recommended_first_wave": payload["execution_readiness"][
                    "recommended_first_wave"
                ],
                "commands_run_by_preflight": payload["execution_readiness"][
                    "commands_run_by_preflight"
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
    parser.add_argument("--bundle-report", type=Path, default=DEFAULT_BUNDLE_REPORT)
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
