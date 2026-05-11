"""Volume-backed Modal train-attempt wrapper for dummy survival.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.dummy_survival_train_attempt
    uv run --extra modal modal run -m curvyzero.infra.modal.dummy_survival_train_attempt --iterations 1 --episodes-per-iter 2 --seed 0 --eval-episodes 2

This is the first run/attempt-shaped wrapper for the dummy survival trainer.
It keeps resume out of scope: each call writes one fresh attempt directory and
updates the latest-attempt pointer for that run.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-dummy-survival-train-attempt"
TASK_ID = "dummy-survival"
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


def _training_config(
    *,
    iterations: int,
    episodes_per_iter: int,
    seed: int,
    eval_episodes: int,
    checkpoint_every_iterations: int | None,
    safety_filter_epsilon: bool,
) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "episodes_per_iter": episodes_per_iter,
        "seed": seed,
        "eval_episodes": eval_episodes,
        "checkpoint_every_iterations": checkpoint_every_iterations,
        "safety_filter_epsilon": safety_filter_epsilon,
    }


def _write_run_manifest_once(*, run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    ref = runs.run_manifest_ref(TASK_ID, run_id)
    path = runs.volume_path(RUNS_MOUNT, ref)
    if path.exists():
        return runs.file_summary(path, mount=RUNS_MOUNT)
    payload = runs.run_manifest(task_id=TASK_ID, run_id=run_id, config=config)
    runs.write_json(path, payload, exclusive=True)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_attempt_state(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str | None,
    summary_ref: str | None,
    config: dict[str, Any],
    exclusive: bool = False,
) -> dict[str, Any]:
    ref = runs.attempt_manifest_ref(TASK_ID, run_id, attempt_id)
    path = runs.volume_path(RUNS_MOUNT, ref)
    payload = runs.attempt_manifest(
        task_id=TASK_ID,
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        summary_ref=summary_ref,
        config=config,
    )
    runs.write_json(path, payload, exclusive=exclusive)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _write_latest_attempt(
    *,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str | None,
    summary_ref: str | None,
) -> dict[str, Any]:
    ref = runs.latest_attempt_ref(TASK_ID, run_id)
    path = runs.volume_path(RUNS_MOUNT, ref)
    payload = runs.latest_attempt_pointer(
        task_id=TASK_ID,
        run_id=run_id,
        attempt_id=attempt_id,
        status=status,
        started_at=started_at,
        ended_at=ended_at,
        summary_ref=summary_ref,
    )
    runs.write_json(path, payload)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _file_summaries_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, raw_path in sorted(summary.get("artifacts", {}).items()):
        path = Path(raw_path)
        if path.is_file():
            summaries[name] = runs.file_summary(path, mount=RUNS_MOUNT)
        elif path.is_dir():
            summaries[name] = {
                "ref": runs.file_ref(path, mount=RUNS_MOUNT),
                "path": str(path),
                "files": [
                    runs.file_summary(child, mount=RUNS_MOUNT)
                    for child in sorted(path.iterdir())
                    if child.is_file()
                ],
            }
    return summaries


def _mirror_periodic_checkpoints(
    *,
    summary: dict[str, Any],
    run_id: str,
    attempt_id: str,
) -> dict[str, Any]:
    mirrored_files: list[dict[str, Any]] = []
    latest_pointer: dict[str, Any] | None = None
    for item in summary.get("periodic_checkpoints", []):
        completed_iterations = int(item["completed_iterations"])
        source = Path(str(item["path"]))
        checkpoint_ref = runs.checkpoint_file_ref(TASK_ID, run_id, completed_iterations)
        metadata_ref = runs.checkpoint_metadata_ref(TASK_ID, run_id, completed_iterations)
        checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
        metadata_path = runs.volume_path(RUNS_MOUNT, metadata_ref)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, checkpoint_path)
        metadata = {
            "schema": "curvyzero_dummy_survival_periodic_checkpoint_metadata/v1",
            "task_id": TASK_ID,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "completed_iterations": completed_iterations,
            "source_checkpoint_ref": runs.file_ref(source, mount=RUNS_MOUNT),
            "checkpoint_ref": checkpoint_ref.as_posix(),
            "eval": item.get("eval", {}),
        }
        runs.write_json(metadata_path, metadata)

        pointer = runs.checkpoint_pointer(
            task_id=TASK_ID,
            run_id=run_id,
            attempt_id=attempt_id,
            completed_iterations=completed_iterations,
            checkpoint_ref=checkpoint_ref.as_posix(),
            metadata_ref=metadata_ref.as_posix(),
            artifact_hashes={"checkpoint_npz": runs.sha256_file(checkpoint_path)},
        )
        latest_path = runs.volume_path(RUNS_MOUNT, runs.latest_checkpoint_ref(TASK_ID, run_id))
        runs.write_json(latest_path, pointer)
        latest_pointer = runs.file_summary(latest_path, mount=RUNS_MOUNT)
        mirrored_files.append(
            {
                "completed_iterations": completed_iterations,
                "checkpoint": runs.file_summary(checkpoint_path, mount=RUNS_MOUNT),
                "metadata": runs.file_summary(metadata_path, mount=RUNS_MOUNT),
            }
        )
    return {
        "files": mirrored_files,
        "latest_pointer": latest_pointer,
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=10 * 60)
def train_dummy_survival_attempt(
    iterations: int = 1,
    episodes_per_iter: int = 2,
    seed: int = 0,
    eval_episodes: int = 2,
    checkpoint_every_iterations: int | None = None,
    safety_filter_epsilon: bool = False,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    from curvyzero.training.dummy_survival import run_dummy_survival_training

    started = time.perf_counter()
    clean_run_id = runs.clean_id(run_id or runs.new_run_id("run"), label="run_id")
    clean_attempt_id = runs.clean_id(
        attempt_id or runs.new_attempt_id("attempt"),
        label="attempt_id",
    )
    config = _training_config(
        iterations=iterations,
        episodes_per_iter=episodes_per_iter,
        seed=seed,
        eval_episodes=eval_episodes,
        checkpoint_every_iterations=checkpoint_every_iterations,
        safety_filter_epsilon=safety_filter_epsilon,
    )
    train_ref = runs.attempt_train_ref(TASK_ID, clean_run_id, clean_attempt_id)
    train_dir = runs.volume_path(RUNS_MOUNT, train_ref)
    started_at = runs.utc_timestamp()
    manifest_files: dict[str, Any] = {}
    summary_ref: str | None = None

    manifest_files["run_json"] = _write_run_manifest_once(
        run_id=clean_run_id,
        config=config,
    )
    manifest_files["attempt_json"] = _write_attempt_state(
        run_id=clean_run_id,
        attempt_id=clean_attempt_id,
        status="running",
        started_at=started_at,
        ended_at=None,
        summary_ref=None,
        config=config,
        exclusive=True,
    )
    manifest_files["latest_attempt_json"] = _write_latest_attempt(
        run_id=clean_run_id,
        attempt_id=clean_attempt_id,
        status="running",
        started_at=started_at,
        ended_at=None,
        summary_ref=None,
    )

    try:
        summary = run_dummy_survival_training(
            iterations=iterations,
            episodes_per_iter=episodes_per_iter,
            seed=seed,
            output_dir=train_dir,
            eval_episodes=eval_episodes,
            checkpoint_every_iterations=checkpoint_every_iterations,
            safety_filter_epsilon=safety_filter_epsilon,
        )
        summary_ref = runs.file_ref(train_dir / "summary.json", mount=RUNS_MOUNT)
        mirrored_checkpoints = _mirror_periodic_checkpoints(
            summary=summary,
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
        )
        ended_at = runs.utc_timestamp()
        manifest_files["attempt_json"] = _write_attempt_state(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            status="completed",
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref,
            config=config,
        )
        manifest_files["latest_attempt_json"] = _write_latest_attempt(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            status="completed",
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref,
        )
        runs_volume.commit()

        result = {
            "schema": "curvyzero_modal_dummy_survival_train_attempt_result/v1",
            "app_name": APP_NAME,
            "volume_name": VOLUME_NAME,
            "run_id": clean_run_id,
            "attempt_id": clean_attempt_id,
            "output_refs": {
                "run_json": runs.run_manifest_ref(TASK_ID, clean_run_id).as_posix(),
                "attempt_json": runs.attempt_manifest_ref(
                    TASK_ID,
                    clean_run_id,
                    clean_attempt_id,
                ).as_posix(),
                "latest_attempt_json": runs.latest_attempt_ref(
                    TASK_ID,
                    clean_run_id,
                ).as_posix(),
                "train_dir": train_ref.as_posix(),
                "summary_json": summary_ref,
            },
            "file_summaries": {
                "manifests": manifest_files,
                "train_outputs": _file_summaries_from_summary(summary),
                "mirrored_checkpoints": mirrored_checkpoints,
            },
            "final_eval": summary["final_eval"],
            "model": summary["model"],
            "committed": True,
            "remote_elapsed_sec": round(time.perf_counter() - started, 6),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result
    except BaseException:
        ended_at = runs.utc_timestamp()
        _write_attempt_state(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            status="failed",
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref,
            config=config,
        )
        _write_latest_attempt(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            status="failed",
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref,
        )
        runs_volume.commit()
        raise


@app.local_entrypoint()
def main(
    iterations: int = 1,
    episodes_per_iter: int = 2,
    seed: int = 0,
    eval_episodes: int = 2,
    checkpoint_every_iterations: int | None = None,
    safety_filter_epsilon: bool = False,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    started = time.perf_counter()
    result = train_dummy_survival_attempt.remote(
        iterations=iterations,
        episodes_per_iter=episodes_per_iter,
        seed=seed,
        eval_episodes=eval_episodes,
        checkpoint_every_iterations=checkpoint_every_iterations,
        safety_filter_epsilon=safety_filter_epsilon,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
