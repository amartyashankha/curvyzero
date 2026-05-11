"""Run the first local toy baseline evaluation harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.toy_baseline import run_toy_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = run_toy_baselines(
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
