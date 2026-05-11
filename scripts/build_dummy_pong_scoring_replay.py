"""Build track-ball versus random scoring replay for dummy Pong."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_scoring_replay import build_dummy_pong_scoring_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-seat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--row-policy", choices=("track_ball", "all"), default="track_ball")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = build_dummy_pong_scoring_replay(
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        max_steps=args.max_steps,
        row_policy=args.row_policy,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
