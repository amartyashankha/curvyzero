"""Run deterministic dummy Pong observability traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_observability import run_dummy_pong_observability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-match", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = run_dummy_pong_observability(
        games_per_match=args.games_per_match,
        seed=args.seed,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
