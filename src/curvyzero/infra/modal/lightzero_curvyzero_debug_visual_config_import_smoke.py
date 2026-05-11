"""Installed-runtime config/import smoke for CurvyZero's debug visual LightZero env.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.lightzero_curvyzero_debug_visual_config_import_smoke

This does not train. It installs LightZero, imports the debug visual CurvyZero
env wrapper, creates/resets/steps it through DI-engine's env registry, and
reports the non-ALE debug visual tensor surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from curvyzero.training.curvyzero_debug_visual_lightzero_runtime_probe import (
    run_curvyzero_debug_visual_lightzero_runtime_probe,
)


APP_NAME = "curvyzero-lightzero-curvyzero-debug-visual-config-import-smoke"
LIGHTZERO_VERSION = "0.2.0"
REMOTE_ROOT = Path("/repo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(f"LightZero=={LIGHTZERO_VERSION}", "numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App(APP_NAME)


@app.function(image=image, timeout=8 * 60)
def lightzero_curvyzero_debug_visual_config_import_smoke(seed: int = 0) -> dict[str, Any]:
    result = run_curvyzero_debug_visual_lightzero_runtime_probe(
        seed=seed,
        include_env_factory=True,
        require_installed_runtime=True,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(seed: int = 0) -> None:
    result = lightzero_curvyzero_debug_visual_config_import_smoke.remote(seed=seed)
    print(json.dumps(result, indent=2, sort_keys=True))
