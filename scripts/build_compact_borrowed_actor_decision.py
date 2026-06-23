#!/usr/bin/env python3
"""Build the OPT-098 borrowed single-actor speed decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from curvyzero.training.compact_borrowed_actor_decision import (
    COMPACT_BORROWED_ACTOR_DECISION_MANIFEST_SCHEMA_ID,
)
from curvyzero.training.compact_borrowed_actor_decision import (
    build_compact_borrowed_actor_decision_v1,
)
from curvyzero.training.compact_borrowed_actor_decision import (
    validate_compact_borrowed_actor_decision_v1,
)


DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/local/curvytron_compact_borrowed_actor_decision_results"
)
DEFAULT_RUN_ID = "opt098-a1-borrowed-operational-speed-decision-20260531"
DEFAULT_BORROWED_SPEED_ROWS = [
    Path(
        "artifacts/local/curvytron_compact_coach_speed_row_results"
        "/opt096-h100-borrow-eager-r1-20260531/row_001_result.json"
    ),
    Path(
        "artifacts/local/curvytron_compact_coach_speed_row_results"
        "/opt096-h100-borrow-eager-r2-20260531/row_001_result.json"
    ),
]
DEFAULT_REFERENCE_SPEED_ROW = Path(
    "artifacts/local/curvytron_compact_coach_speed_row_results"
    "/opt094-h100-refresh-cadence4-r2-20260531/row_001_result.json"
)
DEFAULT_NORMAL_DEATH_PROFILE_RESULT = Path(
    "artifacts/local/curvytron_hybrid_observation_profile_results"
    "/opt097-borrowed-normal-death-compact-owned-20260531/row_001_result.json"
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
    borrowed_rows = args.borrowed_speed_row or DEFAULT_BORROWED_SPEED_ROWS
    payload = build_compact_borrowed_actor_decision_v1(
        run_id=str(args.run_id),
        borrowed_speed_row_paths=[_resolve(path, repo_root) for path in borrowed_rows],
        normal_death_profile_result_path=_resolve(
            args.normal_death_profile_result,
            repo_root,
        ),
        same_shape_reference_speed_row_path=_resolve(
            args.same_shape_reference_speed_row,
            repo_root,
        ),
    )
    validate_compact_borrowed_actor_decision_v1(payload)
    report_path = output_dir / "borrowed_actor_decision_report.json"
    manifest_path = output_dir / "manifest.json"
    _write_json(report_path, payload)
    _write_json(
        manifest_path,
        {
            "schema_id": COMPACT_BORROWED_ACTOR_DECISION_MANIFEST_SCHEMA_ID,
            "ok": payload["ok"],
            "run_id": str(args.run_id),
            "status": payload["status"],
            "decision": payload["decision"],
            "report_path": str(report_path),
            "evidence_ref": payload["evidence_ref"],
            "borrowed_a1_operational_speed_candidate_allowed": payload[
                "attached_claims"
            ]["borrowed_a1_operational_speed_candidate_allowed"],
            "same_shape_baseline_replaced": payload["attached_claims"][
                "same_shape_baseline_replaced"
            ],
            "normal_death_speed_claim": payload["attached_claims"][
                "normal_death_speed_claim"
            ],
            "non_claims": payload["non_claims"],
        },
    )
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "run_id": str(args.run_id),
                "status": payload["status"],
                "decision": payload["decision"],
                "report_path": str(report_path),
                "manifest_path": str(manifest_path),
                "borrowed_mean_steps_per_sec": payload["speed_lane"][
                    "mean_steps_per_sec"
                ],
                "borrowed_mean_wall_sec": payload["speed_lane"]["mean_wall_sec"],
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
        "--borrowed-speed-row",
        type=Path,
        action="append",
        help="Borrowed A1 speed row result. Defaults to OPT-096 r1/r2.",
    )
    parser.add_argument(
        "--normal-death-profile-result",
        type=Path,
        default=DEFAULT_NORMAL_DEATH_PROFILE_RESULT,
    )
    parser.add_argument(
        "--same-shape-reference-speed-row",
        type=Path,
        default=DEFAULT_REFERENCE_SPEED_ROW,
    )
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
