"""Tiny Modal artifact smoke for CurvyZero.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.artifact_smoke
    uv run --extra modal modal run -m curvyzero.infra.modal.artifact_smoke --run-id smoke-demo

The remote Function writes one JSON artifact to the ``curvyzero-runs`` Volume,
commits it, and returns the immutable run/attempt path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import modal

APP_NAME = "curvyzero-artifact-smoke"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
ARTIFACT_DIR = "artifact-smoke"
ARTIFACT_FILE = "artifact.json"
SAFE_ID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")

image = modal.Image.debian_slim(python_version="3.11")
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _created_at() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{_utc_stamp()}-{uuid.uuid4().hex[:12]}"


def _clean_id(raw: str, *, label: str) -> str:
    if (
        not raw
        or len(raw) > 96
        or raw in {".", ".."}
        or not raw[0].isalnum()
        or any(char not in SAFE_ID_CHARS for char in raw)
    ):
        raise ValueError(
            f"{label} must be 1-96 chars of letters, numbers, dash, underscore, or dot"
        )
    return raw


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _artifact_ref(run_id: str, attempt_id: str) -> PurePosixPath:
    return (
        PurePosixPath("experiments")
        / run_id
        / "attempts"
        / attempt_id
        / ARTIFACT_DIR
        / ARTIFACT_FILE
    )


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=5 * 60)
def write_artifact(run_id: str | None = None, attempt_id: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    clean_run_id = _clean_id(run_id or _new_id("run"), label="run_id")
    clean_attempt_id = _clean_id(attempt_id or _new_id("attempt"), label="attempt_id")
    ref = _artifact_ref(clean_run_id, clean_attempt_id)
    path = RUNS_MOUNT / Path(*ref.parts)

    payload = {
        "schema": "curvyzero_modal_artifact_smoke/v1",
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
        "run_id": clean_run_id,
        "attempt_id": clean_attempt_id,
        "created_at": _created_at(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        },
        "message": "generic Modal Volume artifact smoke",
    }
    body = _json_bytes(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)

    runs_volume.commit()

    result = {
        "schema": "curvyzero_modal_artifact_smoke_result/v1",
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
        "volume_mount": str(RUNS_MOUNT),
        "run_id": clean_run_id,
        "attempt_id": clean_attempt_id,
        "artifact_ref": ref.as_posix(),
        "artifact_path": str(path),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "committed": True,
        "remote_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    return result


@app.local_entrypoint()
def main(run_id: str | None = None, attempt_id: str | None = None) -> None:
    started = time.perf_counter()
    result = write_artifact.remote(run_id=run_id, attempt_id=attempt_id)
    result["client_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    print(json.dumps(result, indent=2, sort_keys=True))
