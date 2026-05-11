"""Volume-backed Modal eval-attempt wrapper for dummy Pong checkpoint scoreboards.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.dummy_pong_scoreboard_attempt \
      --checkpoints latest=ref:training/dummy-pong/RUN_ID/attempts/ATTEMPT_ID/train/checkpoint.npz \
      --episodes 2 \
      --seed 0 \
      --split-id dummy_pong_monitor_v0 \
      --split-role monitor

This is intentionally one coarse CPU Modal Function. It runs the existing
dummy Pong checkpoint scoreboard logic, writes summary/episodes artifacts to
the durable ``curvyzero-runs`` Volume, and updates run/attempt manifests.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-dummy-pong-scoreboard-attempt"
TASK_ID = "dummy-pong"
DEFAULT_EVAL_ID = "checkpoint-scoreboard"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26")
    .env({"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}"})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(Path.cwd() / "scripts", remote_path=str(REMOTE_ROOT / "scripts"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _checkpoint_args_from_text(checkpoints: str) -> list[str]:
    values = [
        item.strip()
        for item in checkpoints.replace("\n", ",").split(",")
        if item.strip()
    ]
    if not values:
        raise ValueError(
            "checkpoints must include at least one LABEL=PATH_OR_REF entry; "
            "use commas for multiple entries"
        )
    return values


def _scoreboard_config(
    *,
    checkpoint_args: list[str],
    episodes: int,
    seed: int,
    split_id: str | None,
    split_role: str | None,
    eval_id: str,
    output_ref: str | None,
) -> dict[str, Any]:
    return {
        "job_kind": "dummy_pong_checkpoint_scoreboard",
        "checkpoints": checkpoint_args,
        "episodes": episodes,
        "seed": seed,
        "split_id": split_id,
        "split_role": split_role,
        "eval_id": eval_id,
        "output_ref": output_ref,
        "plain_language": {
            "proves": "Remote eval and Modal Volume artifact plumbing.",
            "does_not_prove": "Policy quality.",
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


def _eval_ref(
    *,
    run_id: str,
    attempt_id: str,
    eval_id: str,
    output_ref: str | None,
) -> PurePosixPath:
    if output_ref:
        return _explicit_volume_ref(output_ref) or runs.require_relative_ref(output_ref)
    return runs.attempt_eval_ref(TASK_ID, run_id, attempt_id, eval_id)


def _parse_checkpoint_arg(value: str) -> tuple[str | None, str]:
    if "=" not in value:
        return None, value
    label, path_text = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("checkpoint label must not be empty")
    path_text = path_text.strip()
    if not path_text:
        raise ValueError("checkpoint path/ref must not be empty")
    return label, path_text


def _resolved_checkpoint_args(
    checkpoint_args: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    resolved_args = []
    inputs = []
    for value in checkpoint_args:
        label, path_text = _parse_checkpoint_arg(value)
        path, source = _resolve_checkpoint_path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint file not found: {path}")
        resolved_arg = f"{label}={path}" if label else str(path)
        resolved_args.append(resolved_arg)
        inputs.append(
            {
                "checkpoint_arg": value,
                "checkpoint_label": label,
                "checkpoint_path_or_ref": path_text,
                "resolved_checkpoint_arg": resolved_arg,
                "resolved_checkpoint_path": str(path),
                **source,
                "file": _file_summary_any_mount(path),
            }
        )
    return resolved_args, inputs


def _resolve_checkpoint_path(path_text: str) -> tuple[Path, dict[str, Any]]:
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


def _explicit_volume_ref(path_text: str) -> PurePosixPath | None:
    for prefix in ("ref:", "volume:"):
        if path_text.startswith(prefix):
            return runs.require_relative_ref(path_text[len(prefix) :])
    return None


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


def _compact_scoreboard(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "episodes_per_match": summary.get("episodes_per_match"),
        "total_episodes": summary.get("total_episodes"),
        "checkpoint_specs": summary.get("checkpoint_specs", []),
        "scoreboard_rows": summary.get("scoreboard_rows", []),
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=10 * 60)
def score_dummy_pong_attempt(
    checkpoints: str,
    episodes: int = 2,
    seed: int = 0,
    split_id: str | None = "dummy_pong_monitor_v0",
    split_role: str | None = "monitor",
    run_id: str | None = None,
    attempt_id: str | None = None,
    eval_id: str = DEFAULT_EVAL_ID,
    output_ref: str | None = None,
) -> dict[str, Any]:
    from curvyzero.training.dummy_pong_eval import run_dummy_pong_eval
    from scripts.run_dummy_pong_checkpoint_scoreboard import _as_scoreboard_summary
    from scripts.run_dummy_pong_checkpoint_scoreboard import _checkpoint_policy_arg

    started = time.perf_counter()
    clean_run_id = runs.clean_id(run_id or runs.new_run_id("pong-scoreboard"), label="run_id")
    clean_attempt_id = runs.clean_id(
        attempt_id or runs.new_attempt_id("attempt"),
        label="attempt_id",
    )
    clean_eval_id = runs.clean_id(eval_id, label="eval_id")
    checkpoint_args = _checkpoint_args_from_text(checkpoints)
    config = _scoreboard_config(
        checkpoint_args=checkpoint_args,
        episodes=episodes,
        seed=seed,
        split_id=split_id,
        split_role=split_role,
        eval_id=clean_eval_id,
        output_ref=output_ref,
    )
    eval_ref = _eval_ref(
        run_id=clean_run_id,
        attempt_id=clean_attempt_id,
        eval_id=clean_eval_id,
        output_ref=output_ref,
    )
    eval_dir = runs.volume_path(RUNS_MOUNT, eval_ref)
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
        resolved_checkpoint_args, checkpoint_inputs = _resolved_checkpoint_args(checkpoint_args)
        checkpoint_policies = [
            _checkpoint_policy_arg(value)
            for value in resolved_checkpoint_args
        ]
        summary = run_dummy_pong_eval(
            episodes=episodes,
            seed=seed,
            output_dir=eval_dir,
            checkpoint_policies=checkpoint_policies,
        )
        summary = _as_scoreboard_summary(
            summary,
            checkpoint_args=checkpoint_args,
            output_dir=eval_dir,
            split_id=split_id,
            split_role=split_role,
        )
        summary["modal_eval"] = {
            "schema": "curvyzero_modal_dummy_pong_scoreboard_attempt/v1",
            "app_name": APP_NAME,
            "volume_name": VOLUME_NAME,
            "task_id": TASK_ID,
            "run_id": clean_run_id,
            "attempt_id": clean_attempt_id,
            "eval_id": clean_eval_id,
            "eval_ref": eval_ref.as_posix(),
            "checkpoint_inputs": checkpoint_inputs,
            "plain_language": {
                "proves": "Remote eval and Modal Volume artifact plumbing.",
                "does_not_prove": "Policy quality.",
            },
        }
        summary_path = eval_dir / "summary.json"
        episodes_path = eval_dir / "episodes.jsonl"
        runs.write_json(summary_path, summary)
        summary_ref = runs.file_ref(summary_path, mount=RUNS_MOUNT)

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
            "schema": "curvyzero_modal_dummy_pong_scoreboard_attempt_result/v1",
            "app_name": APP_NAME,
            "volume_name": VOLUME_NAME,
            "run_id": clean_run_id,
            "attempt_id": clean_attempt_id,
            "eval_id": clean_eval_id,
            "output_refs": {
                "run_json": runs.run_manifest_ref(TASK_ID, clean_run_id).as_posix(),
                "attempt_json": runs.attempt_manifest_ref(
                    TASK_ID,
                    clean_run_id,
                    clean_attempt_id,
                ).as_posix(),
                "latest_attempt_json": runs.latest_attempt_ref(TASK_ID, clean_run_id).as_posix(),
                "eval_dir": eval_ref.as_posix(),
                "summary_json": summary_ref,
                "episodes_jsonl": runs.file_ref(episodes_path, mount=RUNS_MOUNT),
            },
            "file_summaries": {
                "manifests": manifest_files,
                "checkpoint_inputs": checkpoint_inputs,
                "eval_outputs": _file_summaries_from_paths(
                    {
                        "summary_json": summary_path,
                        "episodes_jsonl": episodes_path,
                    }
                ),
            },
            "scoreboard": _compact_scoreboard(summary),
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
    checkpoints: str,
    episodes: int = 2,
    seed: int = 0,
    split_id: str | None = "dummy_pong_monitor_v0",
    split_role: str | None = "monitor",
    run_id: str | None = None,
    attempt_id: str | None = None,
    eval_id: str = DEFAULT_EVAL_ID,
    output_ref: str | None = None,
) -> None:
    started = time.perf_counter()
    result = score_dummy_pong_attempt.remote(
        checkpoints=checkpoints,
        episodes=episodes,
        seed=seed,
        split_id=split_id,
        split_role=split_role,
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=eval_id,
        output_ref=output_ref,
    )
    result["client_elapsed_sec"] = round(time.perf_counter() - started, 6)
    print(json.dumps(result, indent=2, sort_keys=True))
