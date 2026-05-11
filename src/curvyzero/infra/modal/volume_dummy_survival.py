"""Volume-backed Modal smoke for dummy survival training artifacts.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.volume_dummy_survival
    uv run --extra modal modal run -m curvyzero.infra.modal.volume_dummy_survival --iterations 1 --episodes-per-iter 2 --seed 0 --eval-episodes 2

This is intentionally one coarse CPU Modal Function. It proves that the dummy
training stack can write summary/checkpoint/metrics artifacts to a durable
Volume and return compact artifact refs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from curvyzero.infra.modal.run_management import file_summary

APP_NAME = "curvyzero-volume-dummy-survival"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _artifact_ref(
    *,
    seed: int,
    iterations: int,
    episodes_per_iter: int,
    eval_episodes: int,
) -> PurePosixPath:
    return (
        PurePosixPath("training")
        / "dummy-survival"
        / "volume-smoke"
        / f"seed-{seed}"
        / f"iterations-{iterations}"
        / f"episodes-per-iter-{episodes_per_iter}"
        / f"eval-episodes-{eval_episodes}"
    )


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=10 * 60)
def train_volume_dummy_survival(
    iterations: int = 1,
    episodes_per_iter: int = 2,
    seed: int = 0,
    eval_episodes: int = 2,
) -> dict[str, Any]:
    from curvyzero.training.dummy_survival import run_dummy_survival_training

    started = time.perf_counter()
    artifact_ref = _artifact_ref(
        seed=seed,
        iterations=iterations,
        episodes_per_iter=episodes_per_iter,
        eval_episodes=eval_episodes,
    )
    output_dir = RUNS_MOUNT / Path(*artifact_ref.parts)
    summary = run_dummy_survival_training(
        iterations=iterations,
        episodes_per_iter=episodes_per_iter,
        seed=seed,
        output_dir=output_dir,
        eval_episodes=eval_episodes,
    )
    runs_volume.commit()

    artifacts = summary.get("artifacts", {})
    artifact_files = {
        name: file_summary(Path(path), mount=RUNS_MOUNT)
        for name, path in sorted(artifacts.items())
    }
    result = {
        "schema": "curvyzero_modal_volume_dummy_survival_result/v1",
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
        "volume_mount": str(RUNS_MOUNT),
        "volume_path": artifact_ref.as_posix(),
        "seed": seed,
        "iterations": iterations,
        "episodes_per_iter": episodes_per_iter,
        "eval_episodes": eval_episodes,
        "artifact_files": artifact_files,
        "final_eval": summary["final_eval"],
        "model": summary["model"],
        "committed": True,
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main(
    iterations: int = 1,
    episodes_per_iter: int = 2,
    seed: int = 0,
    eval_episodes: int = 2,
) -> None:
    started = time.perf_counter()
    result = train_volume_dummy_survival.remote(
        iterations=iterations,
        episodes_per_iter=episodes_per_iter,
        seed=seed,
        eval_episodes=eval_episodes,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
