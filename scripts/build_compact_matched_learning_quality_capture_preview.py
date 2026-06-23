#!/usr/bin/env python3
"""Preview matched learning-quality capture inputs without producing evidence."""

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
    COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PREVIEW_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    STOCK_REFERENCE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    compact_matched_learning_quality_eval_point_from_summary_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    matched_learning_quality_non_claims_v1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=(STOCK_REFERENCE_ROLE, COMPACT_CANDIDATE_ROLE), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-checkpoint-id", required=True)
    parser.add_argument("--denominator-id", required=True)
    parser.add_argument("--quality-horizon", required=True)
    parser.add_argument("--hardware-class", required=True)
    parser.add_argument("--source-fingerprint-json", type=Path, required=True)
    parser.add_argument("--model-identity-json", type=Path, required=True)
    parser.add_argument("--eval-settings-json", type=Path, required=True)
    parser.add_argument("--denominators-json", type=Path, required=True)
    parser.add_argument("--pre-eval-summary", type=Path, required=True)
    parser.add_argument("--post-eval-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    output_path = (
        _resolve_path(args.output, repo_root)
        if args.output is not None
        else (
            repo_root
            / args.output_root
            / str(args.run_id)
            / f"{args.role}_capture_preview.json"
        ).resolve()
    )
    if output_path.name in {
        "stock_reference_capture.json",
        "compact_candidate_capture.json",
    }:
        raise ValueError("preview output must not use builder input capture filenames")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_fingerprint_path = _resolve_path(args.source_fingerprint_json, repo_root)
    model_identity_path = _resolve_path(args.model_identity_json, repo_root)
    eval_settings_path = _resolve_path(args.eval_settings_json, repo_root)
    denominators_path = _resolve_path(args.denominators_json, repo_root)
    pre_eval_path = _resolve_path(args.pre_eval_summary, repo_root)
    post_eval_path = _resolve_path(args.post_eval_summary, repo_root)

    pre_eval = _read_json_mapping(pre_eval_path, "pre eval summary")
    post_eval = _read_json_mapping(post_eval_path, "post eval summary")
    preview = {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PREVIEW_SCHEMA_ID,
        "ok": True,
        "run_id": str(args.run_id),
        "role": str(args.role),
        "route": (
            "stock_train_muzero_reference"
            if str(args.role) == STOCK_REFERENCE_ROLE
            else "compact_owned_trainer"
        ),
        "candidate_checkpoint_id": str(args.candidate_checkpoint_id),
        "denominator_id": str(args.denominator_id),
        "quality_horizon": str(args.quality_horizon),
        "hardware_class": str(args.hardware_class),
        "usable_as_quality_capture": False,
        "feeds_builder": False,
        "support_only": True,
        "would_fail_reasons": [
            "preview_helper_did_not_run_training",
            "preview_helper_did_not_run_matched_pre_post_eval",
            "preview_helper_must_not_feed_matched_quality_builder",
        ],
        "normalized_eval_points_preview": [
            compact_matched_learning_quality_eval_point_from_summary_v1(
                pre_eval,
                point_id="pre_train",
            ),
            compact_matched_learning_quality_eval_point_from_summary_v1(
                post_eval,
                point_id="post_train",
            ),
        ],
        "source_refs": _artifact_refs_from_paths(
            {
                "source_fingerprint": source_fingerprint_path,
                "model_identity": model_identity_path,
                "eval_settings": eval_settings_path,
                "denominators": denominators_path,
                "pre_eval_summary": pre_eval_path,
                "post_eval_summary": post_eval_path,
            }
        ),
        "non_claims": matched_learning_quality_non_claims_v1(),
    }
    output_path.write_text(
        json.dumps(preview, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "preview_path": str(output_path),
                "feeds_builder": False,
                "usable_as_quality_capture": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _artifact_refs_from_paths(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for kind, path in sorted(paths.items()):
        refs.append(
            {
                "kind": kind,
                "path": str(path),
                "sha256": _file_sha256(path),
                "required": True,
            }
        )
    return refs


def _file_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
