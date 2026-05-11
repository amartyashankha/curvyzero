"""Volume-backed Modal train-attempt wrapper for dummy Pong geometry CEM.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_cem_train_attempt \
      --generations 2 \
      --population-size 8 \
      --elite-count 3 \
      --eval-games 4 \
      --opponent-weights lagged_track_ball_1=1.0,random_uniform=0.10,track_ball=0.10 \
      --target-opponent-id lagged_track_ball_1 \
      --seed 0

This is intentionally one coarse CPU Modal Function. It runs the existing
NumPy CEM trainer inside one container, writes summary/checkpoint/rows artifacts
to the durable ``curvyzero-runs`` Volume, and returns compact refs. It does not
put Modal calls inside environment steps or candidate rollouts.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-dummy-pong-cem-train-attempt"
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


def _cem_config(
    *,
    seed: int,
    width: int,
    height: int,
    paddle_height: int,
    generations: int,
    population_size: int,
    elite_count: int,
    eval_games: int,
    max_steps: int,
    initial_sigma: float,
    sigma_decay: float,
    min_sigma: float,
    track_ball_prior_strength: float,
    stay_bias: float,
    random_opponent_weight: float,
    track_ball_opponent_weight: float,
    opponent_weights: str | None,
    target_opponent_id: str,
    weak_track_ball_epsilon: float,
    loss_delay_weight: float,
    truncation_value: float,
) -> dict[str, Any]:
    return {
        "job_kind": "dummy_pong_geometry_cem_train_attempt",
        "seed": seed,
        "width": width,
        "height": height,
        "paddle_height": paddle_height,
        "generations": generations,
        "population_size": population_size,
        "elite_count": elite_count,
        "eval_games": eval_games,
        "max_steps": max_steps,
        "initial_sigma": initial_sigma,
        "sigma_decay": sigma_decay,
        "min_sigma": min_sigma,
        "track_ball_prior_strength": track_ball_prior_strength,
        "stay_bias": stay_bias,
        "random_opponent_weight": random_opponent_weight,
        "track_ball_opponent_weight": track_ball_opponent_weight,
        "opponent_weights": opponent_weights,
        "target_opponent_id": target_opponent_id,
        "weak_track_ball_epsilon": weak_track_ball_epsilon,
        "loss_delay_weight": loss_delay_weight,
        "truncation_value": truncation_value,
        "plain_language": {
            "proves": (
                "Modal Volume artifact discipline and remote execution for the "
                "dummy Pong CEM-v2 geometry learner."
            ),
            "does_not_prove": (
                "MuZero training, GPU training, or general Pong skill beyond the "
                "configured monitor target."
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


def _parse_opponent_weights(value: str | None) -> dict[str, float] | None:
    if value is None or not value.strip():
        return None
    weights: dict[str, float] = {}
    parts = [
        item.strip()
        for item in value.replace("\n", ",").split(",")
        if item.strip()
    ]
    for item in parts:
        policy_id, separator, weight_text = item.partition("=")
        if not separator:
            raise ValueError("opponent weights must use POLICY=WEIGHT entries")
        policy_id = policy_id.strip()
        if not policy_id:
            raise ValueError("opponent policy id must not be empty")
        if policy_id in weights:
            raise ValueError(f"duplicate opponent weight for {policy_id!r}")
        weights[policy_id] = float(weight_text)
    return weights


def _annotate_summary_file(path: Path, metadata: dict[str, Any]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["modal_train_attempt"] = metadata
    runs.write_json(path, payload)


def _write_cem_rows(path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for generation in summary.get("history", []):
        rows.append(
            {
                "row_kind": "generation",
                "generation": generation.get("generation"),
                "best": generation.get("best"),
                "elite_mean_selection_score": generation.get(
                    "elite_mean_selection_score"
                ),
                "population_selection_score": generation.get(
                    "population_selection_score"
                ),
                "sigma_mean": generation.get("sigma_mean"),
            }
        )

    best_metrics = (
        summary.get("metrics", {})
        .get("best_search_candidate", {})
        .get("metrics", {})
        .get("by_opponent", {})
    )
    for opponent_policy_id, metrics in sorted(best_metrics.items()):
        rows.append(
            {
                "row_kind": "best_search_candidate_by_opponent",
                "opponent_policy_id": opponent_policy_id,
                "metrics": metrics,
            }
        )

    final_metrics = summary.get("metrics", {}).get("final_eval", {}).get(
        "by_opponent",
        {},
    )
    for opponent_policy_id, metrics in sorted(final_metrics.items()):
        rows.append(
            {
                "row_kind": "final_eval_by_opponent",
                "opponent_policy_id": opponent_policy_id,
                "metrics": metrics,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return {
        "rows": len(rows),
        "file": runs.file_summary(path, mount=RUNS_MOUNT),
    }


def _write_latest_checkpoint_pointer(
    *,
    run_id: str,
    attempt_id: str,
    generations: int,
    final_checkpoint: Path,
    train_summary_ref: str,
) -> dict[str, Any]:
    checkpoint_ref = runs.checkpoint_file_ref(TASK_ID, run_id, generations)
    metadata_ref = runs.checkpoint_metadata_ref(TASK_ID, run_id, generations)
    checkpoint_path = runs.volume_path(RUNS_MOUNT, checkpoint_ref)
    metadata_path = runs.volume_path(RUNS_MOUNT, metadata_ref)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_checkpoint, checkpoint_path)

    metadata = {
        "schema": "curvyzero_modal_dummy_pong_cem_checkpoint_metadata/v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "training_axis": "generations",
        "completed_generations": generations,
        "source_summary_ref": train_summary_ref,
        "source_checkpoint_ref": runs.file_ref(final_checkpoint, mount=RUNS_MOUNT),
        "checkpoint_ref": checkpoint_ref.as_posix(),
    }
    runs.write_json(metadata_path, metadata)

    pointer = runs.checkpoint_pointer(
        task_id=TASK_ID,
        run_id=run_id,
        attempt_id=attempt_id,
        completed_iterations=generations,
        checkpoint_ref=checkpoint_ref.as_posix(),
        metadata_ref=metadata_ref.as_posix(),
        seed_cursor={"training_axis": "generations", "completed_generations": generations},
        artifact_hashes={"checkpoint_npz": runs.sha256_file(checkpoint_path)},
    )
    latest_path = runs.volume_path(RUNS_MOUNT, runs.latest_checkpoint_ref(TASK_ID, run_id))
    runs.write_json(latest_path, pointer)
    return {
        "checkpoint": runs.file_summary(checkpoint_path, mount=RUNS_MOUNT),
        "metadata": runs.file_summary(metadata_path, mount=RUNS_MOUNT),
        "latest_pointer": runs.file_summary(latest_path, mount=RUNS_MOUNT),
    }


def _compact_cem_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_opponent_id": summary.get("target_opponent_id"),
        "generations": summary.get("generations"),
        "population_size": summary.get("population_size"),
        "elite_count": summary.get("elite_count"),
        "eval_games": summary.get("eval_games"),
        "selection_return": summary.get("selection_return"),
        "metrics": summary.get("metrics"),
        "plain_language": summary.get("plain_language"),
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=20 * 60)
def train_dummy_pong_cem_attempt(
    seed: int = 0,
    width: int = 15,
    height: int = 9,
    paddle_height: int = 3,
    generations: int = 3,
    population_size: int = 12,
    elite_count: int = 4,
    eval_games: int = 8,
    max_steps: int = 120,
    initial_sigma: float = 0.75,
    sigma_decay: float = 0.7,
    min_sigma: float = 0.05,
    track_ball_prior_strength: float = 12.0,
    stay_bias: float = 0.75,
    random_opponent_weight: float = 0.25,
    track_ball_opponent_weight: float = 0.75,
    opponent_weights: str | None = None,
    target_opponent_id: str = "track_ball",
    weak_track_ball_epsilon: float = 0.35,
    loss_delay_weight: float = 0.5,
    truncation_value: float = 0.0,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong_cem_train import train_dummy_pong_cem

    runs_volume.reload()
    started = time.perf_counter()
    clean_run_id = runs.clean_id(run_id or runs.new_run_id("pong-cem"), label="run_id")
    clean_attempt_id = runs.clean_id(
        attempt_id or runs.new_attempt_id("attempt"),
        label="attempt_id",
    )
    config = _cem_config(
        seed=seed,
        width=width,
        height=height,
        paddle_height=paddle_height,
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        eval_games=eval_games,
        max_steps=max_steps,
        initial_sigma=initial_sigma,
        sigma_decay=sigma_decay,
        min_sigma=min_sigma,
        track_ball_prior_strength=track_ball_prior_strength,
        stay_bias=stay_bias,
        random_opponent_weight=random_opponent_weight,
        track_ball_opponent_weight=track_ball_opponent_weight,
        opponent_weights=opponent_weights,
        target_opponent_id=target_opponent_id,
        weak_track_ball_epsilon=weak_track_ball_epsilon,
        loss_delay_weight=loss_delay_weight,
        truncation_value=truncation_value,
    )
    train_ref = runs.attempt_train_ref(TASK_ID, clean_run_id, clean_attempt_id)
    train_dir = runs.volume_path(RUNS_MOUNT, train_ref)
    rows_path = train_dir / "cem_rows.jsonl"
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
        parsed_opponent_weights = _parse_opponent_weights(opponent_weights)
        train_summary = train_dummy_pong_cem(
            output_dir=train_dir,
            seed=seed,
            width=width,
            height=height,
            paddle_height=paddle_height,
            generations=generations,
            population_size=population_size,
            elite_count=elite_count,
            eval_games=eval_games,
            max_steps=max_steps,
            initial_sigma=initial_sigma,
            sigma_decay=sigma_decay,
            min_sigma=min_sigma,
            track_ball_prior_strength=track_ball_prior_strength,
            stay_bias=stay_bias,
            random_opponent_weight=random_opponent_weight,
            track_ball_opponent_weight=track_ball_opponent_weight,
            opponent_weights=parsed_opponent_weights,
            target_opponent_id=target_opponent_id,
            weak_track_ball_epsilon=weak_track_ball_epsilon,
            loss_delay_weight=loss_delay_weight,
            truncation_value=truncation_value,
        )
        summary_path = train_dir / "summary.json"
        rows_summary = _write_cem_rows(rows_path, train_summary)
        summary_ref = runs.file_ref(summary_path, mount=RUNS_MOUNT)
        _annotate_summary_file(
            summary_path,
            {
                "schema": "curvyzero_modal_dummy_pong_cem_train_attempt/v1",
                "app_name": APP_NAME,
                "volume_name": VOLUME_NAME,
                "task_id": TASK_ID,
                "run_id": clean_run_id,
                "attempt_id": clean_attempt_id,
                "train_ref": train_ref.as_posix(),
                "rows_ref": runs.file_ref(rows_path, mount=RUNS_MOUNT),
                "opponent_weights_input": opponent_weights,
                "plain_language": config["plain_language"],
            },
        )
        latest_checkpoint = _write_latest_checkpoint_pointer(
            run_id=clean_run_id,
            attempt_id=clean_attempt_id,
            generations=generations,
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
            "schema": "curvyzero_modal_dummy_pong_cem_train_attempt_result/v1",
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
                "cem_rows_jsonl": runs.file_ref(rows_path, mount=RUNS_MOUNT),
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
                "train_outputs": _file_summaries_from_paths(
                    {
                        "summary_json": summary_path,
                        "checkpoint_npz": train_dir / "checkpoint.npz",
                        "cem_rows_jsonl": rows_path,
                    }
                ),
                "rows": rows_summary,
                "canonical_checkpoint": latest_checkpoint,
            },
            "train": _compact_cem_summary(train_summary),
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
    seed: int = 0,
    width: int = 15,
    height: int = 9,
    paddle_height: int = 3,
    generations: int = 3,
    population_size: int = 12,
    elite_count: int = 4,
    eval_games: int = 8,
    max_steps: int = 120,
    initial_sigma: float = 0.75,
    sigma_decay: float = 0.7,
    min_sigma: float = 0.05,
    track_ball_prior_strength: float = 12.0,
    stay_bias: float = 0.75,
    random_opponent_weight: float = 0.25,
    track_ball_opponent_weight: float = 0.75,
    opponent_weights: str | None = None,
    target_opponent_id: str = "track_ball",
    weak_track_ball_epsilon: float = 0.35,
    loss_delay_weight: float = 0.5,
    truncation_value: float = 0.0,
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> None:
    started = time.perf_counter()
    result = train_dummy_pong_cem_attempt.remote(
        seed=seed,
        width=width,
        height=height,
        paddle_height=paddle_height,
        generations=generations,
        population_size=population_size,
        elite_count=elite_count,
        eval_games=eval_games,
        max_steps=max_steps,
        initial_sigma=initial_sigma,
        sigma_decay=sigma_decay,
        min_sigma=min_sigma,
        track_ball_prior_strength=track_ball_prior_strength,
        stay_bias=stay_bias,
        random_opponent_weight=random_opponent_weight,
        track_ball_opponent_weight=track_ball_opponent_weight,
        opponent_weights=opponent_weights,
        target_opponent_id=target_opponent_id,
        weak_track_ball_epsilon=weak_track_ball_epsilon,
        loss_delay_weight=loss_delay_weight,
        truncation_value=truncation_value,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
