"""Volume-backed Modal train-attempt wrapper for dummy Pong self-play.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_train_attempt \
      --games 16 \
      --epochs 5 \
      --seed 0

This is a reproduction/diagnostic wrapper around the current tiny NumPy Pong
self-play trainer. It proves Modal Volume artifact discipline and enables
remote Pong reproduction. It does not prove that the current self-play
objective is correct or that the resulting policy is strong.
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

APP_NAME = "curvyzero-dummy-pong-train-attempt"
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
    games: int,
    epochs: int,
    seed: int,
    max_steps: int,
    policy: str,
    epsilon: float,
    policy_learning_rate: float,
    value_learning_rate: float,
    action_diversity_beta: float,
    validation_fraction: float,
    l2: float,
    checkpoint_every_epochs: int | None,
    initial_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "job_kind": "dummy_pong_selfplay_reproduction_train_attempt",
        "games": games,
        "epochs": epochs,
        "seed": seed,
        "max_steps": max_steps,
        "policy": policy,
        "epsilon": epsilon,
        "policy_learning_rate": policy_learning_rate,
        "value_learning_rate": value_learning_rate,
        "action_diversity_beta": action_diversity_beta,
        "validation_fraction": validation_fraction,
        "l2": l2,
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "initial_checkpoint": initial_checkpoint,
        "plain_language": {
            "proves": (
                "Modal Volume artifact discipline and remote reproduction for the "
                "current dummy Pong self-play trainer."
            ),
            "does_not_prove": (
                "The current self-play objective is correct, final, or capable of "
                "learning a strong Pong policy."
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


def _resolve_initial_checkpoint(path_text: str | None) -> tuple[Path | None, dict[str, Any] | None]:
    if path_text is None:
        return None, None
    path, source = _resolve_path_or_ref(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"initial checkpoint file not found: {path}")
    return path, {
        "initial_checkpoint_arg": path_text,
        "resolved_initial_checkpoint_path": str(path),
        **source,
        "file": _file_summary_any_mount(path),
    }


def _resolve_policy_arg(policy: str) -> tuple[str, dict[str, Any]]:
    learned_prefix = "learned:"
    if not policy.startswith(learned_prefix):
        return policy, {
            "policy_arg": policy,
            "resolved_policy_arg": policy,
            "source_kind": "builtin_policy",
        }
    checkpoint_text = policy[len(learned_prefix) :]
    label: str | None = None
    path_text = checkpoint_text
    if "=" in checkpoint_text:
        label_text, path_text = checkpoint_text.split("=", 1)
        label = label_text.strip()
        if not label:
            raise ValueError("learned policy label must not be empty")
    if not path_text:
        raise ValueError("learned policy checkpoint path/ref must not be empty")

    path, source = _resolve_path_or_ref(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"policy checkpoint file not found: {path}")
    resolved_checkpoint = str(path)
    resolved_arg = f"{learned_prefix}{label}={resolved_checkpoint}" if label else (
        f"{learned_prefix}{resolved_checkpoint}"
    )
    return resolved_arg, {
        "policy_arg": policy,
        "resolved_policy_arg": resolved_arg,
        "resolved_checkpoint_path": str(path),
        "label": label,
        **source,
        "file": _file_summary_any_mount(path),
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


def _file_summaries_from_paths(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        name: _file_summary_any_mount(path)
        for name, path in sorted(paths.items())
        if path.is_file()
    }


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


def _annotate_summary_file(path: Path, metadata_key: str, metadata: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[metadata_key] = metadata
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
        "schema": "curvyzero_modal_dummy_pong_checkpoint_metadata/v1",
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
        "data": summary.get("data"),
        "metrics": summary.get("metrics"),
        "checkpoints": summary.get("checkpoints"),
        "plain_language": summary.get("plain_language"),
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=10 * 60)
def train_dummy_pong_attempt(
    games: int = 16,
    epochs: int = 5,
    seed: int = 0,
    max_steps: int = 120,
    policy: str = "random_uniform",
    epsilon: float = 0.0,
    policy_learning_rate: float = 0.1,
    value_learning_rate: float = 0.001,
    action_diversity_beta: float = 0.01,
    validation_fraction: float = 0.2,
    l2: float = 0.0,
    checkpoint_every_epochs: int | None = None,
    initial_checkpoint: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong_selfplay_replay import build_dummy_pong_selfplay_replay
    from curvyzero.training.dummy_pong_selfplay_train import train_dummy_pong_selfplay

    runs_volume.reload()
    started = time.perf_counter()
    clean_run_id = runs.clean_id(run_id or runs.new_run_id("pong-train"), label="run_id")
    clean_attempt_id = runs.clean_id(
        attempt_id or runs.new_attempt_id("attempt"),
        label="attempt_id",
    )
    config = _training_config(
        games=games,
        epochs=epochs,
        seed=seed,
        max_steps=max_steps,
        policy=policy,
        epsilon=epsilon,
        policy_learning_rate=policy_learning_rate,
        value_learning_rate=value_learning_rate,
        action_diversity_beta=action_diversity_beta,
        validation_fraction=validation_fraction,
        l2=l2,
        checkpoint_every_epochs=checkpoint_every_epochs,
        initial_checkpoint=initial_checkpoint,
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
        resolved_policy, policy_input = _resolve_policy_arg(policy)
        resolved_initial_checkpoint, checkpoint_input = _resolve_initial_checkpoint(
            initial_checkpoint
        )
        replay_summary = build_dummy_pong_selfplay_replay(
            games=games,
            seed=seed,
            output_dir=replay_dir,
            max_steps=max_steps,
            policy=resolved_policy,
            epsilon=epsilon,
        )
        _annotate_summary_file(
            replay_dir / "summary.json",
            "modal_train_attempt",
            {
                "schema": "curvyzero_modal_dummy_pong_replay_attempt/v1",
                "app_name": APP_NAME,
                "volume_name": VOLUME_NAME,
                "task_id": TASK_ID,
                "run_id": clean_run_id,
                "attempt_id": clean_attempt_id,
                "replay_ref": replay_ref.as_posix(),
                "policy_input": policy_input,
                "plain_language": config["plain_language"],
            },
        )

        train_summary = train_dummy_pong_selfplay(
            replay_path=replay_dir,
            output_dir=train_dir,
            seed=seed,
            epochs=epochs,
            policy_learning_rate=policy_learning_rate,
            value_learning_rate=value_learning_rate,
            action_diversity_beta=action_diversity_beta,
            validation_fraction=validation_fraction,
            l2=l2,
            initial_checkpoint=resolved_initial_checkpoint,
            checkpoint_every_epochs=checkpoint_every_epochs,
        )
        summary_path = train_dir / "summary.json"
        summary_ref = runs.file_ref(summary_path, mount=RUNS_MOUNT)
        _annotate_summary_file(
            summary_path,
            "modal_train_attempt",
            {
                "schema": "curvyzero_modal_dummy_pong_train_attempt/v1",
                "app_name": APP_NAME,
                "volume_name": VOLUME_NAME,
                "task_id": TASK_ID,
                "run_id": clean_run_id,
                "attempt_id": clean_attempt_id,
                "replay_ref": replay_ref.as_posix(),
                "train_ref": train_ref.as_posix(),
                "policy_input": policy_input,
                "initial_checkpoint_input": checkpoint_input,
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
            "schema": "curvyzero_modal_dummy_pong_train_attempt_result/v1",
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
                "latest_attempt_json": runs.latest_attempt_ref(TASK_ID, clean_run_id).as_posix(),
                "replay_dir": replay_ref.as_posix(),
                "replay_summary_json": runs.file_ref(replay_dir / "summary.json", mount=RUNS_MOUNT),
                "replay_rows_jsonl": runs.file_ref(
                    replay_dir / "replay_rows.jsonl",
                    mount=RUNS_MOUNT,
                ),
                "train_dir": train_ref.as_posix(),
                "train_summary_json": summary_ref,
                "checkpoint_npz": runs.file_ref(train_dir / "checkpoint.npz", mount=RUNS_MOUNT),
                "latest_checkpoint_json": runs.latest_checkpoint_ref(
                    TASK_ID,
                    clean_run_id,
                ).as_posix(),
            },
            "file_summaries": {
                "manifests": manifest_files,
                "inputs": {
                    "policy": policy_input,
                    "initial_checkpoint": checkpoint_input,
                },
                "replay_outputs": _file_summaries_from_paths(
                    {
                        "summary_json": replay_dir / "summary.json",
                        "replay_rows_jsonl": replay_dir / "replay_rows.jsonl",
                    }
                ),
                "train_outputs": _directory_file_summaries(train_dir),
                "canonical_checkpoint": latest_checkpoint,
            },
            "replay": {
                "games": replay_summary.get("games"),
                "total_rows": replay_summary.get("total_rows"),
                "outcome_summary": replay_summary.get("outcome_summary"),
                "action_histogram_by_ego_agent": replay_summary.get(
                    "action_histogram_by_ego_agent"
                ),
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
    games: int = 16,
    epochs: int = 5,
    seed: int = 0,
    max_steps: int = 120,
    policy: str = "random_uniform",
    epsilon: float = 0.0,
    policy_learning_rate: float = 0.1,
    value_learning_rate: float = 0.001,
    action_diversity_beta: float = 0.01,
    validation_fraction: float = 0.2,
    l2: float = 0.0,
    checkpoint_every_epochs: int | None = None,
    initial_checkpoint: str | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    started = time.perf_counter()
    result = train_dummy_pong_attempt.remote(
        games=games,
        epochs=epochs,
        seed=seed,
        max_steps=max_steps,
        policy=policy,
        epsilon=epsilon,
        policy_learning_rate=policy_learning_rate,
        value_learning_rate=value_learning_rate,
        action_diversity_beta=action_diversity_beta,
        validation_fraction=validation_fraction,
        l2=l2,
        checkpoint_every_epochs=checkpoint_every_epochs,
        initial_checkpoint=initial_checkpoint,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
