"""Run a tiny host-only actor/search -> sample/learner transport mock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.compact_split_transport_mock import (
    run_host_only_split_transport_mock,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_split_transport_mock_results")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="opt119-host-only-split-transport-mock-20260601")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sample-interval", type=int, default=2)
    parser.add_argument("--max-pending", type=int, default=2)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=1)
    parser.add_argument("--worker-delay-sec", type=float, default=0.01)
    args = parser.parse_args()

    report = run_host_only_split_transport_mock(
        steps=int(args.steps),
        sample_interval=int(args.sample_interval),
        max_pending=int(args.max_pending),
        sample_batch_size=int(args.sample_batch_size),
        train_steps=int(args.train_steps),
        worker_delay_sec=float(args.worker_delay_sec),
    ).to_dict()
    output_dir = Path(args.output_root) / str(args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mock_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": bool(report["ok"]), "report_path": str(report_path)}))
    return 0 if bool(report["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
