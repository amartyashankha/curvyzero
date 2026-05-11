"""Train the tiny raster value regressor for dummy Pong scoring replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_value_train import train_dummy_pong_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-path",
        type=Path,
        required=True,
        help="Path to replay_rows.jsonl or a scoring replay directory containing it.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--ridge-l2", type=float, default=1e-6)
    args = parser.parse_args()

    summary = train_dummy_pong_value(
        replay_path=args.replay_path,
        output_dir=args.output_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        discount=args.discount,
        ridge_l2=args.ridge_l2,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
