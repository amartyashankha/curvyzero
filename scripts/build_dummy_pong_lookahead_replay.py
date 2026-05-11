"""Build short-lookahead action-label replay for dummy Pong."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_lookahead_replay import COLLECTOR_POLICIES
from curvyzero.training.dummy_pong_lookahead_replay import TIE_BREAK_POLICIES
from curvyzero.training.dummy_pong_lookahead_replay import build_dummy_pong_lookahead_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-seat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--lookahead-steps", type=int, default=24)
    parser.add_argument(
        "--ego-sequence-depth",
        type=int,
        default=1,
        choices=(1, 2),
        help="Number of forced ego decisions to search before track_ball rollout.",
    )
    parser.add_argument("--collector-policy", choices=COLLECTOR_POLICIES, default="random_uniform")
    parser.add_argument("--tie-break-policy", choices=TIE_BREAK_POLICIES, default="track_ball")
    parser.add_argument(
        "--include-ties",
        action="store_true",
        help="Emit states where every candidate action has the same score-delta return.",
    )
    parser.add_argument(
        "--loss-delay-alpha",
        type=float,
        default=0.0,
        help=(
            "Training-label-only bonus for losing candidate rollouts, scaled by "
            "steps_run/lookahead_steps. Eval still uses true score outcomes."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = build_dummy_pong_lookahead_replay(
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        max_steps=args.max_steps,
        lookahead_steps=args.lookahead_steps,
        ego_sequence_depth=args.ego_sequence_depth,
        collector_policy=args.collector_policy,
        include_ties=args.include_ties,
        tie_break_policy=args.tie_break_policy,
        loss_delay_alpha=args.loss_delay_alpha,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
