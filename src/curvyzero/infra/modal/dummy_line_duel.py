"""Modal wrapper for the dummy two-player line-duel training scaffold.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.dummy_line_duel
    uv run --extra modal modal run -m curvyzero.infra.modal.dummy_line_duel --iterations 1 --episodes-per-iter 2 --seed 0 --eval-episodes 2

This mirrors ``scripts/run_dummy_line_duel_train.py`` on a small CPU Modal
Function. Artifacts are written to the remote ephemeral filesystem only; no
Volume is attached.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "curvyzero-dummy-line-duel"
REMOTE_ROOT = Path("/repo")
REMOTE_ARTIFACT_ROOT = Path("/tmp/artifacts/curvyzero/dummy_line_duel")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)

app = modal.App(APP_NAME)


def _compact_summary(summary: dict[str, Any], *, elapsed_sec: float, output_dir: Path) -> dict[str, Any]:
    artifacts = summary.get("artifacts", {})
    return {
        "schema": "curvyzero_modal_dummy_line_duel_result/v1",
        "kind": summary["kind"],
        "note": summary["note"],
        "seed": summary["seed"],
        "iterations": summary["iterations"],
        "episodes_per_iter": summary["episodes_per_iter"],
        "eval_episodes": summary["eval_episodes"],
        "schemas": summary["schemas"],
        "rollout_policy": summary["rollout_policy"],
        "final_eval": summary["final_eval"],
        "model": summary["model"],
        "artifact_dir": str(output_dir),
        "artifacts": artifacts,
        "remote_elapsed_sec": elapsed_sec,
    }


@app.function(image=image, timeout=10 * 60)
def train_dummy_line_duel(
    iterations: int = 3,
    episodes_per_iter: int = 10,
    seed: int = 0,
    eval_episodes: int = 20,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    from curvyzero.training.dummy_line_duel import run_dummy_line_duel_training

    started = time.perf_counter()
    output_dir = REMOTE_ARTIFACT_ROOT / f"seed-{seed}-iters-{iterations}-episodes-{episodes_per_iter}"
    summary = run_dummy_line_duel_training(
        iterations=iterations,
        episodes_per_iter=episodes_per_iter,
        seed=seed,
        output_dir=output_dir if write_artifacts else None,
        eval_episodes=eval_episodes,
    )
    result = _compact_summary(
        summary,
        elapsed_sec=round(time.perf_counter() - started, 6),
        output_dir=output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    iterations: int = 3,
    episodes_per_iter: int = 10,
    seed: int = 0,
    eval_episodes: int = 20,
    write_artifacts: bool = True,
) -> None:
    started = time.perf_counter()
    result = train_dummy_line_duel.remote(
        iterations=iterations,
        episodes_per_iter=episodes_per_iter,
        seed=seed,
        eval_episodes=eval_episodes,
        write_artifacts=write_artifacts,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
