#!/usr/bin/env python3
"""Build the OPT-065 compact Torch compile/eager speed-pair review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_compile_eager_speed_pair import (
    COMPACT_COMPILE_EAGER_SPEED_PAIR_MANIFEST_SCHEMA_ID,
)
from curvyzero.training.compact_compile_eager_speed_pair import (
    CompileEagerPairInput,
)
from curvyzero.training.compact_compile_eager_speed_pair import (
    build_compact_compile_eager_speed_pair_v1,
)
from curvyzero.training.compact_compile_eager_speed_pair import (
    validate_compact_compile_eager_speed_pair_v1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_compile_eager_speed_pair_results"
)
DEFAULT_RUN_ID = "opt065-compile-eager-speed-pair-review-r1r2-20260531"
DEFAULT_PAIRS = (
    (
        "r1",
        Path(
            "artifacts/local/curvytron_compact_coach_speed_row_results"
            "/opt065-h100-pair-eager-canonical-sim1-r1-20260531"
            "/compact_coach_speed_row_modal_report.json"
        ),
        Path(
            "artifacts/local/curvytron_compact_coach_speed_row_results"
            "/opt065-h100-pair-modelcompile-default-canonical-sim1-r1-20260531"
            "/compact_coach_speed_row_modal_report.json"
        ),
    ),
    (
        "r2",
        Path(
            "artifacts/local/curvytron_compact_coach_speed_row_results"
            "/opt065-h100-pair-eager-canonical-sim1-r2-20260531"
            "/compact_coach_speed_row_modal_report.json"
        ),
        Path(
            "artifacts/local/curvytron_compact_coach_speed_row_results"
            "/opt065-h100-pair-modelcompile-default-canonical-sim1-r2-20260531"
            "/compact_coach_speed_row_modal_report.json"
        ),
    ),
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    pairs = [
        CompileEagerPairInput(
            pair_id=str(pair_id),
            eager_report_path=_resolve(Path(eager), repo_root),
            compile_report_path=_resolve(Path(compiled), repo_root),
        )
        for pair_id, eager, compiled in args.pair
    ]
    payload = build_compact_compile_eager_speed_pair_v1(
        run_id=str(args.run_id),
        pairs=pairs,
        min_pair_count=int(args.min_pair_count),
        min_wall_win_fraction=float(args.min_wall_win_fraction),
        require_action_trajectory_match=not bool(args.allow_trajectory_mismatch),
    )
    validate_compact_compile_eager_speed_pair_v1(payload)

    report_path = output_dir / "compile_eager_speed_pair_report.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(report_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_id": COMPACT_COMPILE_EAGER_SPEED_PAIR_MANIFEST_SCHEMA_ID,
            "ok": payload["ok"],
            "run_id": str(args.run_id),
            "report_path": str(report_path),
            "status": payload["status"],
            "decision": payload["aggregate"]["decision"],
            "speed_claim_allowed": payload["aggregate"]["speed_claim_allowed"],
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
                "decision": payload["aggregate"]["decision"],
                "speed_claim_allowed": payload["aggregate"]["speed_claim_allowed"],
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
        "--pair",
        action="append",
        nargs=3,
        metavar=("PAIR_ID", "EAGER_REPORT", "COMPILE_REPORT"),
        default=None,
        help="Pair id plus eager and compile speed-row report paths.",
    )
    parser.add_argument("--min-pair-count", type=int, default=2)
    parser.add_argument("--min-wall-win-fraction", type=float, default=0.05)
    parser.add_argument(
        "--allow-trajectory-mismatch",
        action="store_true",
        help="Review speed without requiring action/trajectory checksum equality.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(argv)
    if parsed.pair is None:
        parsed.pair = DEFAULT_PAIRS
    return parsed


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
