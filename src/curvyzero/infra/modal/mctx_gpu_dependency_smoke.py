"""Contained Modal GPU dependency smoke for the Mctx/JAX MuZero path.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.mctx_gpu_dependency_smoke

This is not a trainer. It only verifies that a cheap Modal GPU can import JAX
with CUDA support, import Mctx, and execute one tiny synthetic Gumbel MuZero
search.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from curvyzero.infra.modal.mctx_dependency_smoke import JAX_VERSION
from curvyzero.infra.modal.mctx_dependency_smoke import MCTX_VERSION
from curvyzero.infra.modal.mctx_dependency_smoke import _run_mctx_search_smoke

APP_NAME = "curvyzero-mctx-gpu-dependency-smoke"
REMOTE_ROOT = Path("/repo")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        f"mctx=={MCTX_VERSION}",
        f"jax[cuda12]=={JAX_VERSION}",
        "numpy>=1.26",
    )
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App(APP_NAME)


@app.function(image=gpu_image, gpu=["L4", "T4"], timeout=8 * 60)
def gpu_smoke() -> dict:
    return _run_mctx_search_smoke(require_gpu_backend=True)


@app.local_entrypoint()
def main() -> None:
    result = gpu_smoke.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
