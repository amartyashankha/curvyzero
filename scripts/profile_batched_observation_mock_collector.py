#!/usr/bin/env python3
"""Run the profile-only batched-observation mock collector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.training.source_state_batched_observation_mock_collector import (  # noqa: E402
    MockCollectorConfig,
)
from curvyzero.training.source_state_batched_observation_mock_collector import (  # noqa: E402
    run_mock_collector_profile,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--player-count", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=2000)
    parser.add_argument("--include-rnd-meter", action="store_true")
    parser.add_argument("--rnd-batch-size", type=int, default=64)
    parser.add_argument("--rnd-update-per-collect", type=int, default=100)
    parser.add_argument("--rnd-device", default="cpu")
    parser.add_argument("--no-pickle-payload", action="store_true")
    args = parser.parse_args()

    result = run_mock_collector_profile(
        MockCollectorConfig(
            batch_size=args.batch_size,
            player_count=args.player_count,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            max_ticks=args.max_ticks,
            include_rnd_meter=bool(args.include_rnd_meter),
            rnd_batch_size=args.rnd_batch_size,
            rnd_update_per_collect=args.rnd_update_per_collect,
            rnd_device=str(args.rnd_device),
            pickle_payload=not bool(args.no_pickle_payload),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
