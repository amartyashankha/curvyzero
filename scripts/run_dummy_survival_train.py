"""Run the dummy solo turning-survival training scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_survival import run_dummy_survival_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-every-iterations",
        type=int,
        help="Optionally write periodic checkpoints/iteration-*.npz under output dir.",
    )
    parser.add_argument(
        "--safety-filter-epsilon",
        action="store_true",
        help=(
            "Experimental collection variant: during epsilon exploration, "
            "sample only positive-clearance actions when possible."
        ),
    )
    parser.add_argument(
        "--planner-unknown-next-value",
        type=float,
        help=(
            "Experimental planner option: value used for unseen next states. "
            "Use a negative value, such as -1.0, to make unknown moves look unsafe."
        ),
    )
    args = parser.parse_args()

    summary = run_dummy_survival_training(
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        seed=args.seed,
        output_dir=args.output_dir,
        checkpoint_every_iterations=args.checkpoint_every_iterations,
        safety_filter_epsilon=args.safety_filter_epsilon,
        planner_unknown_next_value=args.planner_unknown_next_value,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
