"""Run the dummy Pong-like baseline/checkpoint matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_eval import run_dummy_pong_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--paired-seats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate distinct matchups in paired seats. Use --no-paired-seats "
            "to keep checkpoint-vs-baseline rows to checkpoint player_0 only."
        ),
    )
    parser.add_argument(
        "--checkpoint-policy",
        action="append",
        default=[],
        help=(
            "Learned checkpoint policy, e.g. "
            "learned:artifacts/local/run/checkpoint.npz."
        ),
    )
    args = parser.parse_args()

    summary = run_dummy_pong_eval(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_policies=args.checkpoint_policy,
        paired_seats=args.paired_seats,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
