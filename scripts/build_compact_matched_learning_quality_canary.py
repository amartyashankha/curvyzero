#!/usr/bin/env python3
"""Assemble a matched stock-vs-compact quality report from raw captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from curvyzero.training.compact_promotion_readiness_learning_quality import (
    build_compact_matched_learning_quality_canary_from_captures_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    validate_compact_matched_learning_quality_canary_v1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
)
DEFAULT_RUN_ID = "optimizer-compact-matched-learning-quality-canary-20260530"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--unified-lifecycle-report", type=Path, required=True)
    parser.add_argument("--stock-reference-capture", type=Path, required=True)
    parser.add_argument("--compact-candidate-capture", type=Path, required=True)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "matched_learning_quality_canary_report.json"
    payload = build_compact_matched_learning_quality_canary_payload(
        run_id=str(args.run_id),
        compatibility_report_path=_resolve_input(args.compatibility_report, repo_root),
        unified_lifecycle_report_path=_resolve_input(
            args.unified_lifecycle_report,
            repo_root,
        ),
        stock_reference_capture_path=_resolve_input(
            args.stock_reference_capture,
            repo_root,
        ),
        compact_candidate_capture_path=_resolve_input(
            args.compact_candidate_capture,
            repo_root,
        ),
    )
    stock_arm_path = output_dir / "stock_reference_arm.json"
    compact_arm_path = output_dir / "compact_candidate_arm.json"
    manifest_path = output_dir / "manifest.json"
    stock_arm_path.write_text(
        json.dumps(payload["arms"]["stock_reference"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact_arm_path.write_text(
        json.dumps(payload["arms"]["compact_candidate"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_id": "curvyzero_compact_matched_learning_quality_builder_manifest/v1",
                "ok": True,
                "run_id": str(args.run_id),
                "builder_scope": "capture_files_to_matched_quality_report",
                "input_capture_files": payload["input_capture_files"],
                "derived_arm_files": {
                    "stock_reference_arm_path": str(stock_arm_path),
                    "compact_candidate_arm_path": str(compact_arm_path),
                },
                "report_path": str(report_path),
                "non_claims": payload["non_claims"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
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


def build_compact_matched_learning_quality_canary_payload(
    *,
    run_id: str,
    compatibility_report_path: str | Path,
    unified_lifecycle_report_path: str | Path,
    stock_reference_capture_path: str | Path,
    compact_candidate_capture_path: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a matched quality canary from externally produced raw captures."""

    stock_capture = _read_json_mapping(
        Path(stock_reference_capture_path),
        "stock reference capture",
    )
    compact_capture = _read_json_mapping(
        Path(compact_candidate_capture_path),
        "compact candidate capture",
    )
    input_capture_files = {
        "stock_reference_capture": _file_record(Path(stock_reference_capture_path)),
        "compact_candidate_capture": _file_record(Path(compact_candidate_capture_path)),
    }
    payload = build_compact_matched_learning_quality_canary_from_captures_v1(
        run_id=str(run_id),
        compatibility_report_path=compatibility_report_path,
        unified_lifecycle_report_path=unified_lifecycle_report_path,
        stock_reference_capture=stock_capture,
        compact_candidate_capture=compact_capture,
        input_capture_files=input_capture_files,
        created_at=created_at,
    )
    validate_compact_matched_learning_quality_canary_v1(payload)
    return payload


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_input(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _file_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
