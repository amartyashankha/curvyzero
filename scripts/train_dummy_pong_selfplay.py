"""Train a tiny policy/value model from dummy Pong self-play replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_selfplay_train import train_dummy_pong_selfplay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-path",
        type=Path,
        required=True,
        help="Path to replay_rows.jsonl or a self-play replay directory containing it.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--policy-learning-rate", type=float, default=0.1)
    parser.add_argument("--value-learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--action-diversity-beta",
        type=float,
        default=0.01,
        help="Small logit regularizer that discourages collapsed action distributions.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=None,
        help="Optionally write policy checkpoints under output_dir/checkpoints every N epochs.",
    )
    args = parser.parse_args()

    summary = train_dummy_pong_selfplay(
        replay_path=args.replay_path,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        policy_learning_rate=args.policy_learning_rate,
        value_learning_rate=args.value_learning_rate,
        action_diversity_beta=args.action_diversity_beta,
        validation_fraction=args.validation_fraction,
        l2=args.l2,
        initial_checkpoint=args.initial_checkpoint,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
