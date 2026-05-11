"""Run baseline/checkpoint evaluation for the dummy solo survival task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_survival_eval import DEFAULT_POLICIES
from curvyzero.training.dummy_survival_eval import run_dummy_survival_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-id", default="dummy_survival_monitor_v0")
    parser.add_argument(
        "--split-role",
        default="monitor",
        choices=["monitor", "selection", "heldout", "debug", "train"],
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
        choices=list(DEFAULT_POLICIES),
    )
    parser.add_argument(
        "--checkpoint-policy",
        action="append",
        nargs="+",
        default=[],
        metavar="learned:PATH",
        help=(
            "Evaluate one or more learned checkpoint policies, e.g. "
            "learned:artifacts/local/run/checkpoint.npz."
        ),
    )
    args = parser.parse_args()

    summary = run_dummy_survival_eval(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        policies=tuple(args.policies),
        checkpoint_policies=tuple(
            policy_spec
            for policy_specs in args.checkpoint_policy
            for policy_spec in policy_specs
        ),
        split_id=args.split_id,
        split_role=args.split_role,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
