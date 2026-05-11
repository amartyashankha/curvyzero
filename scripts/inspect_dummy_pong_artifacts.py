"""Inspect dummy Pong replay and observability artifact directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from curvyzero.training.dummy_pong_artifact_inspect import inspect_dummy_pong_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--sample-frames", type=int, default=3)
    args = parser.parse_args()

    summary = inspect_dummy_pong_artifacts(
        args.artifact_dir,
        sample_frames=args.sample_frames,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
