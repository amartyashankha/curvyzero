#!/usr/bin/env python3
"""Build the OPT-066 compact Torch model-compile decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_model_compile_decision import (
    COMPACT_MODEL_COMPILE_DECISION_MANIFEST_SCHEMA_ID,
)
from curvyzero.training.compact_model_compile_decision import (
    build_compact_model_compile_decision_v1,
)
from curvyzero.training.compact_model_compile_decision import (
    validate_compact_model_compile_decision_v1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_model_compile_decision_results"
)
DEFAULT_RUN_ID = "opt066-model-compile-default-decision-20260531"
DEFAULT_FIXED_ROOT_TAPE_RESULT = Path(
    "artifacts/local/curvytron_hybrid_observation_profile_results"
    "/opt066-fixed-root-modelcompile-gate-smoke-20260531"
    "/row_001_result.json"
)
DEFAULT_POST_GUARD_PAIR_REPORT = Path(
    "artifacts/local/curvytron_compact_compile_eager_speed_pair_results"
    "/opt066-compile-eager-speed-pair-evalguard-r1r2-20260531"
    "/compile_eager_speed_pair_report.json"
)
DEFAULT_PRIOR_PAIR_REPORT = Path(
    "artifacts/local/curvytron_compact_compile_eager_speed_pair_results"
    "/opt065-compile-eager-speed-pair-review-r1r2-20260531-postkeyfix"
    "/compile_eager_speed_pair_report.json"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    prior_report = None if args.no_prior_report else _resolve(args.prior_pair_report, repo_root)
    payload = build_compact_model_compile_decision_v1(
        run_id=str(args.run_id),
        fixed_root_tape_result_path=_resolve(args.fixed_root_tape_result, repo_root),
        post_guard_speed_pair_report_path=_resolve(
            args.post_guard_pair_report,
            repo_root,
        ),
        prior_speed_pair_report_path=prior_report,
    )
    validate_compact_model_compile_decision_v1(payload)

    report_path = output_dir / "model_compile_decision_report.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(report_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_id": COMPACT_MODEL_COMPILE_DECISION_MANIFEST_SCHEMA_ID,
            "ok": payload["ok"],
            "run_id": str(args.run_id),
            "report_path": str(report_path),
            "status": payload["status"],
            "decision": payload["decision"],
            "model_compile_mode": payload["model_compile_mode"],
            "model_compile_default_speed_default_allowed": payload[
                "attached_claims"
            ]["model_compile_default_speed_default_allowed"],
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
                "decision": payload["decision"],
                "model_compile_default_speed_default_allowed": payload[
                    "attached_claims"
                ]["model_compile_default_speed_default_allowed"],
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
        "--fixed-root-tape-result",
        type=Path,
        default=DEFAULT_FIXED_ROOT_TAPE_RESULT,
    )
    parser.add_argument(
        "--post-guard-pair-report",
        type=Path,
        default=DEFAULT_POST_GUARD_PAIR_REPORT,
    )
    parser.add_argument(
        "--prior-pair-report",
        type=Path,
        default=DEFAULT_PRIOR_PAIR_REPORT,
    )
    parser.add_argument("--no-prior-report", action="store_true")
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
