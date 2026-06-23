#!/usr/bin/env python3
"""Build the compact Coach speed-row floor decomposition bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_speed_row_floor_bundle import (
    COMPACT_SPEED_ROW_FLOOR_BUNDLE_MANIFEST_SCHEMA_ID,
)
from curvyzero.training.compact_speed_row_floor_bundle import (
    build_compact_speed_row_floor_bundle_v1,
)
from curvyzero.training.compact_speed_row_floor_bundle import (
    validate_compact_speed_row_floor_bundle_v1,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_speed_row_floor_bundle_results")
DEFAULT_RUN_ID = "optimizer-compact-speed-row-floor-bundle-20260530"
DEFAULT_ACCEPTED_REPORT = Path(
    "artifacts/local/curvytron_compact_coach_speed_row_results"
    "/optimizer-compact-coach-speed-row-h100-b1024a16-threshold-20260530"
    "/compact_coach_speed_row_modal_report.json"
)
DEFAULT_COMPACT_TORCH_REPORT = Path(
    "artifacts/local/curvytron_compact_coach_speed_row_results"
    "/optimizer-speed-row-compact-torch-h100-b1024a16-threshold-20260530"
    "/compact_coach_speed_row_modal_report.json"
)
DEFAULT_FIXED_FLOOR_REPORT = Path(
    "artifacts/local/curvytron_compact_coach_speed_row_results"
    "/optimizer-speed-row-fixed-shape-h100-b1024a16-threshold-20260530"
    "/compact_coach_speed_row_modal_report.json"
)
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    payload = build_compact_speed_row_floor_bundle_v1(
        run_id=str(args.run_id),
        accepted_speed_row_report_path=_resolve(args.accepted_speed_row_report, repo_root),
        compact_torch_sibling_report_path=_resolve(
            args.compact_torch_sibling_report,
            repo_root,
        ),
        fixed_floor_sibling_report_path=_resolve(args.fixed_floor_sibling_report, repo_root),
        compatibility_report_path=_resolve(args.compatibility_report, repo_root)
        if args.compatibility_report
        else None,
        unified_lifecycle_report_path=_resolve(args.unified_lifecycle_report, repo_root)
        if args.unified_lifecycle_report
        else None,
    )
    validate_compact_speed_row_floor_bundle_v1(payload)

    report_path = output_dir / "compact_speed_row_floor_bundle_report.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(report_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_id": COMPACT_SPEED_ROW_FLOOR_BUNDLE_MANIFEST_SCHEMA_ID,
            "ok": payload["ok"],
            "run_id": str(args.run_id),
            "report_path": str(report_path),
            "candidate_checkpoint_id": payload["candidate_checkpoint_id"],
            "status": payload["status"],
            "evidence_ref": payload["evidence_ref"],
            "engineering_read": payload["engineering_read"],
            "input_reports": payload["input_reports"],
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
                "engineering_read": payload["engineering_read"]["classification"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["ok"] is True else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--accepted-speed-row-report",
        type=Path,
        default=DEFAULT_ACCEPTED_REPORT,
    )
    parser.add_argument(
        "--compact-torch-sibling-report",
        type=Path,
        default=DEFAULT_COMPACT_TORCH_REPORT,
    )
    parser.add_argument(
        "--fixed-floor-sibling-report",
        type=Path,
        default=DEFAULT_FIXED_FLOOR_REPORT,
    )
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"speed-row floor output dir is not empty: {output_dir}")
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
