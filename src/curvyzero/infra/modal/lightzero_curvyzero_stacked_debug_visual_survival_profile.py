"""Modal wrapper for the CurvyTron stacked debug visual survival profile.

Run from the repository root:

    uv run --extra modal modal run \
      -m curvyzero.infra.modal.lightzero_curvyzero_stacked_debug_visual_survival_profile \
      --seed 0 --steps 4 --num-simulations 2

This does not train. It installs LightZero, runs the bounded profile, and
reports collect/search/replay/sample/learner-loss plumbing only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from curvyzero.training.curvyzero_stacked_debug_visual_survival_profile import (
    profile_output_payload,
    run_stacked_debug_visual_survival_profile,
)


APP_NAME = "curvyzero-lightzero-stacked-debug-visual-survival-profile"
LIGHTZERO_VERSION = "0.2.0"
REMOTE_ROOT = Path("/repo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}", "numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App(APP_NAME)


@app.function(image=image, timeout=10 * 60)
def lightzero_curvyzero_stacked_debug_visual_survival_profile(
    seed: int = 0,
    steps: int = 4,
    num_simulations: int = 2,
    batch_size: int = 2,
    output: str = "none",
) -> dict[str, Any]:
    result = run_stacked_debug_visual_survival_profile(
        seed=seed,
        steps=steps,
        num_simulations=num_simulations,
        batch_size=batch_size,
        require_installed_lightzero=True,
    )
    payload = profile_output_payload(result, output)
    if payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    seed: int = 0,
    steps: int = 4,
    num_simulations: int = 2,
    batch_size: int = 2,
    output: str = "summary",
    remote_output: str = "none",
) -> None:
    result = lightzero_curvyzero_stacked_debug_visual_survival_profile.remote(
        seed=seed,
        steps=steps,
        num_simulations=num_simulations,
        batch_size=batch_size,
        output=remote_output,
    )
    payload = profile_output_payload(result, output)
    if payload is not None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)
