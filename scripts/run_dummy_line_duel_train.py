"""Run the dummy two-player line-duel training scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_line_duel import run_dummy_line_duel_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episodes-per-iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = run_dummy_line_duel_training(
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
