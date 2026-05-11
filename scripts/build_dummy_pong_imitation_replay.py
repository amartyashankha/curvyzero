"""Build a tiny track-ball imitation replay dataset for dummy Pong."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_replay import build_dummy_pong_imitation_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = build_dummy_pong_imitation_replay(
        games=args.games,
        seed=args.seed,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
