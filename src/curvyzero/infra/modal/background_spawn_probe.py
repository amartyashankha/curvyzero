"""Tiny Modal Function.spawn lifecycle probe.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.background_spawn_probe

The local entrypoint launches a background function with ``Function.spawn`` and
returns after printing the function call id plus Volume refs. The remote
function writes progress markers and a done marker to ``curvyzero-runs``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import modal

from curvyzero.infra.modal import run_management as runs

APP_NAME = "curvyzero-background-spawn-probe"
TASK_ID = "modal-background-spawn-probe"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")

DEFAULT_RUN_ID = "bg-spawn-probe-20260510"
DEFAULT_ATTEMPT_ID = "sleep6-commit2"
DEFAULT_STEPS = 3
DEFAULT_SLEEP_SEC = 2.0

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _attempt_probe_ref(run_id: str, attempt_id: str) -> Path:
    return runs.volume_path(RUNS_MOUNT, runs.attempt_root_ref(TASK_ID, run_id, attempt_id))


def _write_marker(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    runs.write_json(path, payload)
    return runs.file_summary(path, mount=RUNS_MOUNT)


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=5 * 60, cpu=0.25)
def background_sleep_and_mark(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    steps: int = DEFAULT_STEPS,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
) -> dict[str, Any]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if sleep_sec < 0:
        raise ValueError("sleep_sec must be non-negative")

    clean_run_id = runs.clean_id(run_id, label="run_id")
    clean_attempt_id = runs.clean_id(attempt_id, label="attempt_id")
    started = time.perf_counter()
    started_at = runs.utc_timestamp()
    root = _attempt_probe_ref(clean_run_id, clean_attempt_id)
    progress_root = root / "probe_progress"

    common = {
        "app_name": APP_NAME,
        "task_id": TASK_ID,
        "volume_name": VOLUME_NAME,
        "run_id": clean_run_id,
        "attempt_id": clean_attempt_id,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "steps": steps,
        "sleep_sec": sleep_sec,
        "started_at": started_at,
    }

    start_payload = {
        "schema": "curvyzero_modal_background_spawn_probe_progress/v1",
        **common,
        "phase": "starting",
        "step": 0,
        "timestamp": runs.utc_timestamp(),
        "elapsed_sec": 0.0,
    }
    start_summary = _write_marker(progress_root / "latest.json", start_payload)
    _write_marker(progress_root / "progress_000.json", start_payload)
    runs_volume.commit()

    progress_summaries = [start_summary]
    for step in range(1, steps + 1):
        time.sleep(sleep_sec)
        payload = {
            "schema": "curvyzero_modal_background_spawn_probe_progress/v1",
            **common,
            "phase": "running",
            "step": step,
            "timestamp": runs.utc_timestamp(),
            "elapsed_sec": round(time.perf_counter() - started, 6),
        }
        latest_summary = _write_marker(progress_root / "latest.json", payload)
        _write_marker(progress_root / f"progress_{step:03d}.json", payload)
        progress_summaries.append(latest_summary)
        runs_volume.commit()

    ended_at = runs.utc_timestamp()
    done_payload = {
        "schema": "curvyzero_modal_background_spawn_probe_done/v1",
        **common,
        "phase": "completed",
        "ok": True,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "progress_latest_ref": runs.file_ref(progress_root / "latest.json", mount=RUNS_MOUNT),
    }
    done_summary = _write_marker(root / "done.json", done_payload)

    summary_payload = {
        "schema": "curvyzero_modal_background_spawn_probe_summary/v1",
        **common,
        "ok": True,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": round(time.perf_counter() - started, 6),
        "progress_latest": progress_summaries[-1],
        "done": done_summary,
    }
    summary = _write_marker(root / "summary.json", summary_payload)
    runs_volume.commit()

    return {
        "schema": "curvyzero_modal_background_spawn_probe_result/v1",
        "ok": True,
        "run_id": clean_run_id,
        "attempt_id": clean_attempt_id,
        "summary_ref": summary["ref"],
        "done_ref": done_summary["ref"],
        "progress_latest_ref": done_payload["progress_latest_ref"],
        "remote_elapsed_sec": round(time.perf_counter() - started, 6),
    }


@app.local_entrypoint()
def main(
    run_id: str = DEFAULT_RUN_ID,
    attempt_id: str = DEFAULT_ATTEMPT_ID,
    steps: int = DEFAULT_STEPS,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
) -> None:
    clean_run_id = runs.clean_id(run_id, label="run_id")
    clean_attempt_id = runs.clean_id(attempt_id, label="attempt_id")
    function_call = background_sleep_and_mark.spawn(
        run_id=clean_run_id,
        attempt_id=clean_attempt_id,
        steps=steps,
        sleep_sec=sleep_sec,
    )
    call_id = getattr(function_call, "object_id", None) or getattr(function_call, "id", None)
    root_ref = runs.attempt_root_ref(TASK_ID, clean_run_id, clean_attempt_id)
    print(
        json.dumps(
            {
                "schema": "curvyzero_modal_background_spawn_probe_launch/v1",
                "status": "spawned",
                "function_call_id": call_id,
                "volume_name": VOLUME_NAME,
                "run_id": clean_run_id,
                "attempt_id": clean_attempt_id,
                "steps": steps,
                "sleep_sec": sleep_sec,
                "root_ref": root_ref.as_posix(),
                "progress_latest_ref": (root_ref / "probe_progress" / "latest.json").as_posix(),
                "done_ref": (root_ref / "done.json").as_posix(),
                "summary_ref": (root_ref / "summary.json").as_posix(),
                "note": "Local entrypoint returned after Function.spawn; inspect Volume refs.",
            },
            indent=2,
            sort_keys=True,
        )
    )
