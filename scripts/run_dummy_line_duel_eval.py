"""Run the Tiny Line Duel baseline matrix, optionally with learned checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_line_duel_eval import run_dummy_line_duel_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-id", default="dummy_line_duel_monitor_v0")
    parser.add_argument(
        "--split-role",
        default="monitor",
        choices=["monitor", "selection", "heldout", "debug", "train"],
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-policy",
        action="append",
        default=[],
        help="Learned checkpoint policy, e.g. learned:artifacts/local/run/checkpoint.npz.",
    )
    args = parser.parse_args()

    summary = run_dummy_line_duel_eval(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_policies=args.checkpoint_policy,
        split_id=args.split_id,
        split_role=args.split_role,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
