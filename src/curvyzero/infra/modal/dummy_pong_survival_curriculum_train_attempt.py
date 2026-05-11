"""Volume-backed Modal wrapper for shaped dummy Pong survival training.

Run from the repository root:

    uv run --extra modal modal run \
      -m curvyzero.infra.modal.dummy_pong_survival_curriculum_train_attempt \
      --epochs 8 \
      --games-per-epoch 8 \
      --eval-games 4 \
      --survival-weight 0.5 \
      --truncation-bonus 0.0 \
      --reward-mode loss_delay \
      --attempt-id survival-shaped-loss-delay-alpha0.5-smoke8192-s0

This launches the repo-owned dummy Pong shaped-objective ablation only. It is
not stock LightZero Atari Pong and must not be compared as stock sparse reward.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-dummy-pong-survival-curriculum-train-attempt"
TASK_ID = "dummy-pong-survival-shaped"
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
    seed: int,
    epochs: int,
    games_per_epoch: int,
    eval_games: int,
    max_steps: int,
    learning_rate: float,
    l2: float,
    survival_weight: float,
    truncation_bonus: float,
    reward_mode: str,
    weak_track_ball_epsilon: float,
    random_phase_fraction: float,
    weak_phase_fraction: float,
) -> dict[str, Any]:
    train_step_budget = epochs * games_per_epoch * max_steps
    return {
        "job_kind": "dummy_pong_survival_shaped_curriculum_train_attempt",
        "algorithm": "tiny_numpy_on_policy_policy_gradient",
        "lane": "custom_dummy_pong_shaped_objective_ablation",
        "not_stock_control": True,
        "stock_control_warning": (
            "Do not compare this as stock LightZero Atari Pong or stock sparse "
            "dummy Pong. True score is reported separately from shaped training return."
        ),
        "seed": seed,
        "epochs": epochs,
        "games_per_epoch": games_per_epoch,
        "eval_games": eval_games,
        "max_steps": max_steps,
        "max_train_env_steps_upper_bound": train_step_budget,
        "learning_rate": learning_rate,
        "l2": l2,
        "survival_weight": survival_weight,
        "loss_delay_alpha": survival_weight,
        "truncation_bonus": truncation_bonus,
        "reward_mode": reward_mode,
        "weak_track_ball_epsilon": weak_track_ball_epsilon,
        "random_phase_fraction": random_phase_fraction,
        "weak_phase_fraction": weak_phase_fraction,
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


def _file_summaries(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        name: runs.file_summary(path, mount=RUNS_MOUNT)
        for name, path in sorted(paths.items())
        if path.is_file()
    }


def _write_rows(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in summary.get("history", []):
        rows.append(
            {
                "row_kind": "epoch",
                "epoch": item.get("epoch"),
                "curriculum_opponent": item.get("curriculum_opponent"),
                "train": item.get("train"),
                "eval": item.get("eval"),
            }
        )
    for opponent_id, metrics in sorted(
        summary.get("metrics", {}).get("final_eval", {}).items()
    ):
        rows.append(
            {
                "row_kind": "final_eval_by_opponent",
                "opponent_policy_id": opponent_id,
                "metrics": metrics,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return {"rows": len(rows), "file": runs.file_summary(path, mount=RUNS_MOUNT)}


def _annotate_summary(path: Path, metadata: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["modal_train_attempt"] = metadata
    runs.write_json(path, payload)


def _write_latest_checkpoint(
    *,
    run_id: str,
    attempt_id: str,
    epochs: int,
    checkpoint_path: Path,
    summary_ref: str,
) -> dict[str, Any]:
    checkpoint_ref = runs.checkpoint_file_ref(TASK_ID, run_id, epochs)
    metadata_ref = runs.checkpoint_metadata_ref(TASK_ID, run_id, epochs)
    canonical_checkpoint = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    metadata_path = runs.volume_path(RUNS_MOUNT, metadata_ref)
    canonical_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, canonical_checkpoint)
    metadata = {
        "schema": "curvyzero_modal_dummy_pong_survival_shaped_checkpoint_metadata/v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "training_axis": "epochs",
        "completed_epochs": epochs,
        "source_summary_ref": summary_ref,
        "source_checkpoint_ref": runs.file_ref(checkpoint_path, mount=RUNS_MOUNT),
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
        artifact_hashes={"checkpoint_npz": runs.sha256_file(canonical_checkpoint)},
    )
    latest_path = runs.volume_path(RUNS_MOUNT, runs.latest_checkpoint_ref(TASK_ID, run_id))
    runs.write_json(latest_path, pointer)
    return {
        "checkpoint": runs.file_summary(canonical_checkpoint, mount=RUNS_MOUNT),
        "metadata": runs.file_summary(metadata_path, mount=RUNS_MOUNT),
        "latest_pointer": runs.file_summary(latest_path, mount=RUNS_MOUNT),
    }


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": summary.get("kind"),
        "reward_schema_id": summary.get("reward_schema_id"),
        "training_return": summary.get("training_return"),
        "metrics": summary.get("metrics"),
        "plain_language": summary.get("plain_language"),
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=20 * 60)
def train_dummy_pong_survival_shaped_attempt(
    seed: int = 0,
    epochs: int = 8,
    games_per_epoch: int = 8,
    eval_games: int = 4,
    max_steps: int = 120,
    learning_rate: float = 0.05,
    l2: float = 0.0001,
    survival_weight: float = 0.5,
    truncation_bonus: float = 0.0,
    reward_mode: str = "loss_delay",
    weak_track_ball_epsilon: float = 0.35,
    random_phase_fraction: float = 0.34,
    weak_phase_fraction: float = 0.34,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong_survival_curriculum_train import (
        train_dummy_pong_survival_curriculum,
    )

    runs_volume.reload()
    started = time.perf_counter()
    clean_run_id = runs.clean_id(
        run_id or runs.new_run_id("pong-survival-shaped"),
        label="run_id",
    )
    clean_attempt_id = runs.clean_id(
        attempt_id or runs.new_attempt_id("survival-shaped-loss-delay"),
        label="attempt_id",
    )
    config = _training_config(
        seed=seed,
        epochs=epochs,
        games_per_epoch=games_per_epoch,
        eval_games=eval_games,
        max_steps=max_steps,
        learning_rate=learning_rate,
        l2=l2,
        survival_weight=survival_weight,
        truncation_bonus=truncation_bonus,
        reward_mode=reward_mode,
        weak_track_ball_epsilon=weak_track_ball_epsilon,
        random_phase_fraction=random_phase_fraction,
        weak_phase_fraction=weak_phase_fraction,
    )
    train_ref = runs.attempt_train_ref(TASK_ID, clean_run_id, clean_attempt_id)
    train_dir = runs.volume_path(RUNS_MOUNT, train_ref)
    rows_path = train_dir / "survival_shaped_rows.jsonl"
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
        train_summary = train_dummy_pong_survival_curriculum(
            output_dir=train_dir,
            seed=seed,
            epochs=epochs,
            games_per_epoch=games_per_epoch,
            eval_games=eval_games,
            max_steps=max_steps,
            learning_rate=learning_rate,
            l2=l2,
            survival_weight=survival_weight,
            truncation_bonus=truncation_bonus,
            reward_mode=reward_mode,
            weak_track_ball_epsilon=weak_track_ball_epsilon,
            random_phase_fraction=random_phase_fraction,
            weak_phase_fraction=weak_phase_fraction,
        )
        summary_path = train_dir / "summary.json"
        summary_ref = runs.file_ref(summary_path, mount=RUNS_MOUNT)
        rows_summary = _write_rows(rows_path, train_summary)
        _annotate_summary(
            summary_path,
            {
                "schema": "curvyzero_modal_dummy_pong_survival_shaped_train_attempt/v1",
                "app_name": APP_NAME,
                "volume_name": VOLUME_NAME,
                "task_id": TASK_ID,
                "run_id": clean_run_id,
                "attempt_id": clean_attempt_id,
                "train_ref": train_ref.as_posix(),
                "rows_ref": runs.file_ref(rows_path, mount=RUNS_MOUNT),
                "stock_control_warning": config["stock_control_warning"],
            },
        )
        latest_checkpoint = _write_latest_checkpoint(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            epochs=epochs,
            checkpoint_path=train_dir / "checkpoint.npz",
            summary_ref=summary_ref,
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
            "schema": "curvyzero_modal_dummy_pong_survival_shaped_train_attempt_result/v1",
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
                "train_summary_json": summary_ref,
                "rows_jsonl": runs.file_ref(rows_path, mount=RUNS_MOUNT),
                "checkpoint_npz": runs.file_ref(train_dir / "checkpoint.npz", mount=RUNS_MOUNT),
                "latest_checkpoint_json": runs.latest_checkpoint_ref(
                    TASK_ID,
                    clean_run_id,
                ).as_posix(),
            },
            "file_summaries": {
                "manifests": manifest_files,
                "train_outputs": _file_summaries(
                    {
                        "summary_json": summary_path,
                        "checkpoint_npz": train_dir / "checkpoint.npz",
                        "rows_jsonl": rows_path,
                    }
                ),
                "rows": rows_summary,
                "canonical_checkpoint": latest_checkpoint,
            },
            "train": _compact_summary(train_summary),
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
    seed: int = 0,
    epochs: int = 8,
    games_per_epoch: int = 8,
    eval_games: int = 4,
    max_steps: int = 120,
    learning_rate: float = 0.05,
    l2: float = 0.0001,
    survival_weight: float = 0.5,
    truncation_bonus: float = 0.0,
    reward_mode: str = "loss_delay",
    weak_track_ball_epsilon: float = 0.35,
    random_phase_fraction: float = 0.34,
    weak_phase_fraction: float = 0.34,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    started = time.perf_counter()
    result = train_dummy_pong_survival_shaped_attempt.remote(
        seed=seed,
        epochs=epochs,
        games_per_epoch=games_per_epoch,
        eval_games=eval_games,
        max_steps=max_steps,
        learning_rate=learning_rate,
        l2=l2,
        survival_weight=survival_weight,
        truncation_bonus=truncation_bonus,
        reward_mode=reward_mode,
        weak_track_ball_epsilon=weak_track_ball_epsilon,
        random_phase_fraction=random_phase_fraction,
        weak_phase_fraction=weak_phase_fraction,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
