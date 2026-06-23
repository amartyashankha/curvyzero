#!/usr/bin/env python3
"""Assemble a matched learning-quality capture from raw evidence artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_CANDIDATE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    STOCK_REFERENCE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    build_compact_matched_learning_quality_capture_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    compact_matched_learning_quality_arm_from_capture_v1,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_promotion_readiness_results")
CAPTURE_FILENAMES = {
    STOCK_REFERENCE_ROLE: "stock_reference_capture.json",
    COMPACT_CANDIDATE_ROLE: "compact_candidate_capture.json",
}
PREVIEW_CAPTURE_FILENAMES = frozenset(
    {
        "stock_reference_capture_preview.json",
        "compact_candidate_capture_preview.json",
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=(STOCK_REFERENCE_ROLE, COMPACT_CANDIDATE_ROLE), required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--candidate-checkpoint-id", required=True)
    parser.add_argument("--denominator-id", required=True)
    parser.add_argument("--quality-horizon", required=True)
    parser.add_argument("--hardware-class", required=True)
    parser.add_argument("--source-fingerprint-json", type=Path, required=True)
    parser.add_argument("--model-identity-json", type=Path, required=True)
    parser.add_argument("--eval-settings-json", type=Path, required=True)
    parser.add_argument("--denominators-json", type=Path, required=True)
    parser.add_argument("--capture-provenance-json", type=Path, required=True)
    parser.add_argument("--training-artifact", type=Path, required=True)
    parser.add_argument("--pre-eval-summary", type=Path, required=True)
    parser.add_argument("--post-eval-summary", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    output_path = (
        _resolve_path(args.output, repo_root)
        if args.output is not None
        else (
            repo_root / args.output_root / str(args.run_id) / CAPTURE_FILENAMES[str(args.role)]
        ).resolve()
    )
    _validate_output_path(output_path, role=str(args.role))
    if output_path.exists() and not bool(args.overwrite):
        raise FileExistsError(f"capture output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    paths = _resolve_inputs(args, repo_root)
    capture = build_compact_matched_learning_quality_capture_v1(
        role=str(args.role),
        run_id=str(args.run_id),
        capture_id=str(args.capture_id),
        candidate_checkpoint_id=str(args.candidate_checkpoint_id),
        denominator_id=str(args.denominator_id),
        quality_horizon=str(args.quality_horizon),
        hardware_class=str(args.hardware_class),
        source_fingerprint=_read_json_mapping(
            paths["source_fingerprint_json"],
            "source fingerprint",
        ),
        model_identity=_read_json_mapping(
            paths["model_identity_json"],
            "model identity",
        ),
        eval_settings=_read_json_mapping(paths["eval_settings_json"], "eval settings"),
        pre_eval_summary=_read_json_mapping(paths["pre_eval_summary"], "pre eval summary"),
        post_eval_summary=_read_json_mapping(
            paths["post_eval_summary"],
            "post eval summary",
        ),
        denominators=_read_json_mapping(paths["denominators_json"], "denominators"),
        artifact_paths={
            "training_artifact": paths["training_artifact"],
            "pre_eval_summary": paths["pre_eval_summary"],
            "post_eval_summary": paths["post_eval_summary"],
            "initial_checkpoint": paths["initial_checkpoint"],
            "final_checkpoint": paths["final_checkpoint"],
        },
        capture_provenance=_read_json_mapping(
            paths["capture_provenance_json"],
            "capture provenance",
        ),
    )
    compact_matched_learning_quality_arm_from_capture_v1(
        capture,
        expected_role=str(args.role),
    )
    output_path.write_text(
        json.dumps(capture, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "role": str(args.role),
                "capture_path": str(output_path),
                "feeds_builder": True,
            },
            sort_keys=True,
        )
    )
    return 0


def _resolve_inputs(args: argparse.Namespace, repo_root: Path) -> dict[str, Path]:
    return {
        "source_fingerprint_json": _resolve_path(args.source_fingerprint_json, repo_root),
        "model_identity_json": _resolve_path(args.model_identity_json, repo_root),
        "eval_settings_json": _resolve_path(args.eval_settings_json, repo_root),
        "denominators_json": _resolve_path(args.denominators_json, repo_root),
        "capture_provenance_json": _resolve_path(args.capture_provenance_json, repo_root),
        "training_artifact": _resolve_path(args.training_artifact, repo_root),
        "pre_eval_summary": _resolve_path(args.pre_eval_summary, repo_root),
        "post_eval_summary": _resolve_path(args.post_eval_summary, repo_root),
        "initial_checkpoint": _resolve_path(args.initial_checkpoint, repo_root),
        "final_checkpoint": _resolve_path(args.final_checkpoint, repo_root),
    }


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _validate_output_path(path: Path, *, role: str) -> None:
    expected_name = CAPTURE_FILENAMES[role]
    if path.name in PREVIEW_CAPTURE_FILENAMES:
        raise ValueError("capture-from-artifacts output must not use preview capture filenames")
    if path.name != expected_name:
        raise ValueError(f"{role} capture output must be named {expected_name}")


if __name__ == "__main__":
    raise SystemExit(main())
