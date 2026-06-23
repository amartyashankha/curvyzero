#!/usr/bin/env python3
"""Build the compact Torch service-bucket target-selection packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_torch_service_bucket_decision import (
    COMPACT_TORCH_SERVICE_BUCKET_DECISION_MANIFEST_SCHEMA_ID,
)
from curvyzero.training.compact_torch_service_bucket_decision import (
    ServiceBucketSupportInput,
)
from curvyzero.training.compact_torch_service_bucket_decision import (
    build_compact_torch_service_bucket_decision_v1,
)
from curvyzero.training.compact_torch_service_bucket_decision import (
    validate_compact_torch_service_bucket_decision_v1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_torch_service_bucket_decision_results"
)
DEFAULT_RUN_ID = "opt074-compact-torch-service-bucket-decision-20260531"
DEFAULT_CANONICAL_FLOOR_BUNDLE = Path(
    "artifacts/local/curvytron_compact_speed_row_floor_bundle_results"
    "/opt073b-compact-speed-row-floor-bundle-h100-b1024a16-sim1-20260531"
    "/compact_speed_row_floor_bundle_report.json"
)
DEFAULT_REPEAT_FLOOR_BUNDLE = Path(
    "artifacts/local/curvytron_compact_speed_row_floor_bundle_results"
    "/opt074-compact-speed-row-floor-bundle-host-phase-cudaevent-sim1-20260531"
    "/compact_speed_row_floor_bundle_report.json"
)
DEFAULT_COMPILE_DECISION_REPORT = Path(
    "artifacts/local/curvytron_compact_model_compile_decision_results"
    "/opt066-model-compile-default-decision-20260531"
    "/model_compile_decision_report.json"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    payload = build_compact_torch_service_bucket_decision_v1(
        run_id=str(args.run_id),
        canonical_floor_bundle_path=_resolve(args.canonical_floor_bundle, repo_root),
        repeat_floor_bundle_paths=[
            _resolve(path, repo_root) for path in args.repeat_floor_bundle
        ],
        support_floor_bundles=[
            ServiceBucketSupportInput(role=role, path=_resolve(Path(path), repo_root))
            for role, path in args.support_floor_bundle
        ],
        compile_decision_report_path=(
            _resolve(args.compile_decision_report, repo_root)
            if args.compile_decision_report is not None
            else None
        ),
        min_repeat_count=int(args.min_repeat_count),
        recurrent_refresh_guard_plan_present=bool(
            args.recurrent_refresh_guard_plan_present
        ),
    )
    validate_compact_torch_service_bucket_decision_v1(payload)

    report_path = output_dir / "compact_torch_service_bucket_decision_report.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(report_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_id": COMPACT_TORCH_SERVICE_BUCKET_DECISION_MANIFEST_SCHEMA_ID,
            "ok": payload["ok"],
            "run_id": str(args.run_id),
            "report_path": str(report_path),
            "status": payload["status"],
            "decision": payload["decision"],
            "selected_next_target": payload["selected_next_target"],
            "evidence_ref": payload["evidence_ref"],
            "input_refs": payload["input_refs"],
            "non_claims": payload["non_claims"],
            "attached_claims": payload["attached_claims"],
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
                "decision": payload["decision"],
                "selected_next_target": payload["selected_next_target"],
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
        "--canonical-floor-bundle",
        type=Path,
        default=DEFAULT_CANONICAL_FLOOR_BUNDLE,
    )
    parser.add_argument(
        "--repeat-floor-bundle",
        type=Path,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--support-floor-bundle",
        type=_support_floor_bundle_arg,
        action="append",
        default=[],
        metavar=("ROLE", "PATH"),
        nargs=2,
    )
    parser.add_argument(
        "--compile-decision-report",
        type=Path,
        default=DEFAULT_COMPILE_DECISION_REPORT,
    )
    parser.add_argument("--no-compile-decision-report", action="store_true")
    parser.add_argument("--min-repeat-count", type=int, default=2)
    parser.add_argument(
        "--recurrent-refresh-guard-plan-present",
        action="store_true",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.repeat_floor_bundle is None:
        args.repeat_floor_bundle = [DEFAULT_REPEAT_FLOOR_BUNDLE]
    if args.no_compile_decision_report:
        args.compile_decision_report = None
    return args


def _support_floor_bundle_arg(value: str) -> str:
    if not value:
        raise argparse.ArgumentTypeError("support floor bundle fields must be non-empty")
    return value


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"service-bucket decision output dir is not empty: {output_dir}"
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
