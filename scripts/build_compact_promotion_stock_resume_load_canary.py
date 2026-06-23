#!/usr/bin/env python3
"""Build the first post-compatibility stock resume/load readiness canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.compact_promotion_readiness import (
    build_compact_promotion_stock_resume_load_canary_v1,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_promotion_readiness_results")
DEFAULT_RUN_ID = "optimizer-compact-promotion-stock-resume-load-canary-20260530"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--unified-lifecycle-report", type=Path, required=True)
    parser.add_argument("--compatibility-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-simulations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    payload = build_compact_promotion_stock_resume_load_canary_v1(
        run_id=str(args.run_id),
        unified_lifecycle_report_path=_resolve_input(
            args.unified_lifecycle_report,
            repo_root,
        ),
        compatibility_report_path=_resolve_input(args.compatibility_report, repo_root),
        output_dir=output_dir,
        repo_root=repo_root,
        seed=int(args.seed),
        num_simulations=int(args.num_simulations),
        batch_size=int(args.batch_size),
    )
    report_path = output_dir / "stock_resume_load_canary_report.json"
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "report_path": str(report_path)}, sort_keys=True))
    return 0


def _resolve_input(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
