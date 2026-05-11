"""Volume-backed Modal train-attempt wrapper for dummy Pong imitation training.

Run from the repository root after placing the replay on the ``curvyzero-runs``
Volume:

    uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_imitation_train_attempt \
      --replay-path ref:training/dummy-pong/manual-replays/lag1-trace-stack2/replay_rows.jsonl \
      --epochs 800 \
      --learning-rate 0.005 \
      --class-weighting balanced \
      --feature-mode raster_only \
      --frame-stack 2 \
      --model-type mlp \
      --hidden-dim 128 \
      --seed 0

This is intentionally one coarse CPU Modal Function. It runs the existing NumPy
supervised imitation trainer inside one container, writes replay refs,
summary/checkpoint artifacts, manifests, and a latest checkpoint pointer to the
durable ``curvyzero-runs`` Volume, then returns compact refs.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-dummy-pong-imitation-train-attempt"
TASK_ID = "dummy-pong"
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
    replay_path: str,
    seed: int,
    epochs: int,
    learning_rate: float,
    validation_fraction: float,
    l2: float,
    checkpoint_every_epochs: int | None,
    class_weighting: str,
    feature_mode: str,
    frame_stack: int,
    model_type: str,
    hidden_dim: int,
) -> dict[str, Any]:
    return {
        "job_kind": "dummy_pong_imitation_train_attempt",
        "replay_path": replay_path,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "validation_fraction": validation_fraction,
        "l2": l2,
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "class_weighting": class_weighting,
        "feature_mode": feature_mode,
        "frame_stack": frame_stack,
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "plain_language": {
            "proves": (
                "Modal Volume artifact discipline and remote execution for the "
                "dummy Pong supervised imitation learner."
            ),
            "does_not_prove": (
                "MuZero training, self-play improvement, or reward-driven behavior. "
                "The replay rows are still used as supervised action labels."
            ),
        },
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
    modal_task_id: str | None,
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
        modal_task_id=modal_task_id,
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
    modal_task_id: str | None,
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
        modal_task_id=modal_task_id,
        summary_ref=summary_ref,
    )
    runs.write_json(path, payload)
    return runs.file_summary(path, mount=RUNS_MOUNT)


def _attempt_replay_ref(task_id: str, run_id: str, attempt_id: str) -> PurePosixPath:
    return runs.attempt_root_ref(task_id, run_id, attempt_id) / "replay"


def _explicit_volume_ref(path_text: str) -> PurePosixPath | None:
    for prefix in ("ref:", "volume:"):
        if path_text.startswith(prefix):
            return runs.require_relative_ref(path_text[len(prefix) :])
    return None


def _resolve_path_or_ref(path_text: str) -> tuple[Path, dict[str, Any]]:
    explicit_ref = _explicit_volume_ref(path_text)
    if explicit_ref is not None:
        path = runs.volume_path(RUNS_MOUNT, explicit_ref)
        return path, {
            "source_kind": "volume_ref",
            "source_ref": explicit_ref.as_posix(),
        }

    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate, {
            "source_kind": "absolute_path",
            "source_ref": None,
        }
    if candidate.exists():
        return candidate, {
            "source_kind": "relative_path",
            "source_ref": None,
        }

    repo_candidate = REMOTE_ROOT / path_text
    if repo_candidate.exists():
        return repo_candidate, {
            "source_kind": "repo_path",
            "source_ref": None,
        }

    ref = runs.require_relative_ref(path_text)
    path = runs.volume_path(RUNS_MOUNT, ref)
    return path, {
        "source_kind": "volume_ref",
        "source_ref": ref.as_posix(),
    }


def _file_summary_any_mount(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": runs.sha256_file(path),
    }
    try:
        summary["ref"] = runs.file_ref(path, mount=RUNS_MOUNT)
    except ValueError:
        pass
    return summary


def _directory_file_summaries(path: Path) -> dict[str, Any]:
    return {
        "ref": runs.file_ref(path, mount=RUNS_MOUNT),
        "path": str(path),
        "files": [
            runs.file_summary(child, mount=RUNS_MOUNT)
            for child in sorted(path.rglob("*"))
            if child.is_file()
        ],
    }


def _copy_replay_to_attempt(*, replay_path_arg: str, replay_dir: Path) -> dict[str, Any]:
    source_path, source = _resolve_path_or_ref(replay_path_arg)
    replay_rows_source = source_path / "replay_rows.jsonl" if source_path.is_dir() else source_path
    if not replay_rows_source.is_file():
        raise FileNotFoundError(f"replay rows file not found: {replay_rows_source}")

    replay_dir.mkdir(parents=True, exist_ok=True)
    replay_rows_dest = replay_dir / "replay_rows.jsonl"
    shutil.copy2(replay_rows_source, replay_rows_dest)

    summary_source = replay_rows_source.parent / "summary.json"
    copied_summary = None
    if summary_source.is_file():
        summary_dest = replay_dir / "summary.json"
        shutil.copy2(summary_source, summary_dest)
        copied_summary = runs.file_summary(summary_dest, mount=RUNS_MOUNT)

    return {
        "replay_path_arg": replay_path_arg,
        "resolved_replay_rows_path": str(replay_rows_source),
        **source,
        "source_file": _file_summary_any_mount(replay_rows_source),
        "copied_replay_rows": runs.file_summary(replay_rows_dest, mount=RUNS_MOUNT),
        "copied_summary_json": copied_summary,
    }


def _annotate_summary_file(path: Path, metadata: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["modal_train_attempt"] = metadata
    runs.write_json(path, payload)


def _write_latest_checkpoint_pointer(
    *,
    run_id: str,
    attempt_id: str,
    epochs: int,
    final_checkpoint: Path,
    train_summary_ref: str,
) -> dict[str, Any]:
    checkpoint_ref = runs.checkpoint_file_ref(TASK_ID, run_id, epochs)
    metadata_ref = runs.checkpoint_metadata_ref(TASK_ID, run_id, epochs)
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    metadata_path = runs.volume_path(RUNS_MOUNT, metadata_ref)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_checkpoint, checkpoint_path)

    metadata = {
        "schema": "curvyzero_modal_dummy_pong_imitation_checkpoint_metadata/v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "training_axis": "epochs",
        "completed_epochs": epochs,
        "source_summary_ref": train_summary_ref,
        "source_checkpoint_ref": runs.file_ref(final_checkpoint, mount=RUNS_MOUNT),
        "checkpoint_ref": checkpoint_ref.as_posix(),
    }
    runs.write_json(metadata_path, metadata)

    pointer = runs.checkpoint_pointer(
        task_id=TASK_ID,
        run_id=run_id,
        attempt_id=attempt_id,
        completed_iterations=epochs,
        checkpoint_ref=checkpoint_ref.as_posix(),
        metadata_ref=metadata_ref.as_posix(),
        seed_cursor={"training_axis": "epochs", "completed_epochs": epochs},
        artifact_hashes={"checkpoint_npz": runs.sha256_file(checkpoint_path)},
    )
    latest_path = runs.volume_path(RUNS_MOUNT, runs.latest_checkpoint_ref(TASK_ID, run_id))
    runs.write_json(latest_path, pointer)
    return {
        "checkpoint": runs.file_summary(checkpoint_path, mount=RUNS_MOUNT),
        "metadata": runs.file_summary(metadata_path, mount=RUNS_MOUNT),
        "latest_pointer": runs.file_summary(latest_path, mount=RUNS_MOUNT),
    }


def _compact_train_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "epochs": summary.get("epochs"),
        "learning_rate": summary.get("learning_rate"),
        "class_weighting": summary.get("class_weighting"),
        "feature_mode": summary.get("feature_mode"),
        "frame_stack": summary.get("frame_stack"),
        "model": summary.get("model"),
        "data": summary.get("data"),
        "metrics": summary.get("metrics"),
        "checkpoints": summary.get("checkpoints"),
        "plain_language": summary.get("plain_language"),
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=20 * 60)
def train_dummy_pong_imitation_attempt(
    replay_path: str,
    seed: int = 0,
    epochs: int = 800,
    learning_rate: float = 0.005,
    validation_fraction: float = 0.2,
    l2: float = 0.0,
    checkpoint_every_epochs: int | None = None,
    class_weighting: str = "balanced",
    feature_mode: str = "raster_only",
    frame_stack: int = 2,
    model_type: str = "mlp",
    hidden_dim: int = 128,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong_imitation_train import train_dummy_pong_imitation

    runs_volume.reload()
    started = time.perf_counter()
    clean_run_id = runs.clean_id(
        run_id or runs.new_run_id("pong-imitation"),
        label="run_id",
    )
    clean_attempt_id = runs.clean_id(
        attempt_id or runs.new_attempt_id("attempt"),
        label="attempt_id",
    )
    config = _training_config(
        replay_path=replay_path,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        l2=l2,
        checkpoint_every_epochs=checkpoint_every_epochs,
        class_weighting=class_weighting,
        feature_mode=feature_mode,
        frame_stack=frame_stack,
        model_type=model_type,
        hidden_dim=hidden_dim,
    )
    replay_ref = _attempt_replay_ref(TASK_ID, clean_run_id, clean_attempt_id)
    train_ref = runs.attempt_train_ref(TASK_ID, clean_run_id, clean_attempt_id)
    replay_dir = runs.volume_path(RUNS_MOUNT, replay_ref)
    train_dir = runs.volume_path(RUNS_MOUNT, train_ref)
    started_at = runs.utc_timestamp()
    modal_task_id = os.environ.get("MODAL_TASK_ID")
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
        modal_task_id=modal_task_id,
        exclusive=True,
    )
    manifest_files["latest_attempt_json"] = _write_latest_attempt(
        run_id=clean_run_id,
        attempt_id=clean_attempt_id,
        status="running",
        started_at=started_at,
        ended_at=None,
        summary_ref=None,
        modal_task_id=modal_task_id,
    )

    try:
        replay_input = _copy_replay_to_attempt(
            replay_path_arg=replay_path,
            replay_dir=replay_dir,
        )
        train_summary = train_dummy_pong_imitation(
            replay_path=replay_dir,
            output_dir=train_dir,
            seed=seed,
            epochs=epochs,
            learning_rate=learning_rate,
            validation_fraction=validation_fraction,
            l2=l2,
            checkpoint_every_epochs=checkpoint_every_epochs,
            class_weighting=class_weighting,
            feature_mode=feature_mode,
            frame_stack=frame_stack,
            model_type=model_type,
            hidden_dim=hidden_dim,
        )
        summary_path = train_dir / "summary.json"
        summary_ref = runs.file_ref(summary_path, mount=RUNS_MOUNT)
        _annotate_summary_file(
            summary_path,
            {
                "schema": "curvyzero_modal_dummy_pong_imitation_train_attempt/v1",
                "app_name": APP_NAME,
                "volume_name": VOLUME_NAME,
                "task_id": TASK_ID,
                "run_id": clean_run_id,
                "attempt_id": clean_attempt_id,
                "replay_ref": replay_ref.as_posix(),
                "train_ref": train_ref.as_posix(),
                "replay_input": replay_input,
                "plain_language": config["plain_language"],
            },
        )
        latest_checkpoint = _write_latest_checkpoint_pointer(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            epochs=epochs,
            final_checkpoint=train_dir / "checkpoint.npz",
            train_summary_ref=summary_ref,
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
            modal_task_id=modal_task_id,
        )
        manifest_files["latest_attempt_json"] = _write_latest_attempt(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            status="completed",
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref,
            modal_task_id=modal_task_id,
        )
        runs_volume.commit()

        result = {
            "schema": "curvyzero_modal_dummy_pong_imitation_train_attempt_result/v1",
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
                "replay_dir": replay_ref.as_posix(),
                "replay_rows_jsonl": runs.file_ref(
                    replay_dir / "replay_rows.jsonl",
                    mount=RUNS_MOUNT,
                ),
                "train_dir": train_ref.as_posix(),
                "train_summary_json": summary_ref,
                "checkpoint_npz": runs.file_ref(
                    train_dir / "checkpoint.npz",
                    mount=RUNS_MOUNT,
                ),
                "latest_checkpoint_json": runs.latest_checkpoint_ref(
                    TASK_ID,
                    clean_run_id,
                ).as_posix(),
            },
            "file_summaries": {
                "manifests": manifest_files,
                "inputs": {"replay": replay_input},
                "replay_outputs": _directory_file_summaries(replay_dir),
                "train_outputs": _directory_file_summaries(train_dir),
                "canonical_checkpoint": latest_checkpoint,
            },
            "train": _compact_train_summary(train_summary),
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
            modal_task_id=modal_task_id,
        )
        _write_latest_attempt(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            status="failed",
            started_at=started_at,
            ended_at=ended_at,
            summary_ref=summary_ref,
            modal_task_id=modal_task_id,
        )
        runs_volume.commit()
        raise


@app.local_entrypoint()
def main(
    replay_path: str,
    seed: int = 0,
    epochs: int = 800,
    learning_rate: float = 0.005,
    validation_fraction: float = 0.2,
    l2: float = 0.0,
    checkpoint_every_epochs: int | None = None,
    class_weighting: str = "balanced",
    feature_mode: str = "raster_only",
    frame_stack: int = 2,
    model_type: str = "mlp",
    hidden_dim: int = 128,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    started = time.perf_counter()
    result = train_dummy_pong_imitation_attempt.remote(
        replay_path=replay_path,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        validation_fraction=validation_fraction,
        l2=l2,
        checkpoint_every_epochs=checkpoint_every_epochs,
        class_weighting=class_weighting,
        feature_mode=feature_mode,
        frame_stack=frame_stack,
        model_type=model_type,
        hidden_dim=hidden_dim,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
