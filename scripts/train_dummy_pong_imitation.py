"""Train the tiny supervised raster imitation policy for dummy Pong."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_imitation_train import FEATURE_MODES
from curvyzero.training.dummy_pong_imitation_train import MODEL_TYPES
from curvyzero.training.dummy_pong_imitation_train import train_dummy_pong_imitation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-path",
        type=Path,
        required=True,
        help="Path to replay_rows.jsonl or a replay directory containing it.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "balanced"),
        default="none",
        help=(
            "Optional supervised loss weighting. 'balanced' uses inverse target-action "
            "frequency from the training split, per ego-agent head."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=None,
        help="Optionally write policy checkpoints under output_dir/checkpoints every N epochs.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="raster_plus_geometry",
        help=(
            "Feature encoding for the linear policy. The default preserves existing "
            "one-hot raster plus decoded geometry helpers; raster_only uses only "
            "one-hot raster cells and the per-ego policy head."
        ),
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=1,
        help=(
            "Number of chronological raster frames to concatenate. Use 2 with "
            "lag-1 trace replay built via --frame-stack 2 to expose velocity cues."
        ),
    )
    parser.add_argument(
        "--model-type",
        choices=MODEL_TYPES,
        default="linear",
        help="Policy class to train. 'linear' preserves the old softmax checkpoint path.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden width for --model-type mlp. Ignored by the linear policy.",
    )
    args = parser.parse_args()

    summary = train_dummy_pong_imitation(
        replay_path=args.replay_path,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        l2=args.l2,
        class_weighting=args.class_weighting,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        feature_mode=args.feature_mode,
        frame_stack=args.frame_stack,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
