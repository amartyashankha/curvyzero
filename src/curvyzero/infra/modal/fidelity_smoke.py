"""Python-only Modal trace/fingerprint smoke for CurvyZero.

Run from the repository root:

    uv run --extra modal modal run -m curvyzero.infra.modal.fidelity_smoke
    uv run --extra modal modal run -m curvyzero.infra.modal.fidelity_smoke --run-id env-fidelity-smoke-20260508

The remote Function runs one scripted ``curvyzero-v0`` toy Python trace, writes
one JSON artifact to the ``curvyzero-runs`` Volume, commits it, and returns the
immutable run/attempt path. This does not call a JS oracle yet.
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

APP_NAME = "curvyzero-env-fidelity-smoke"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
REMOTE_ROOT = Path("/repo")
ARTIFACT_DIR = "fidelity-smoke"
ARTIFACT_FILE = "trace_fingerprint.json"
SAFE_ID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")

TRACE_SEED = 123
SCRIPTED_ACTIONS = (
    {"player_0": 1, "player_1": 1},
    {"player_0": 2, "player_1": 0},
    {"player_0": 2, "player_1": 0},
    {"player_0": 1, "player_1": 1},
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy>=1.26")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
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


def _build_trace_payload(run_id: str, attempt_id: str) -> dict[str, Any]:
    from curvyzero.env.config import CurvyTronConfig
    from curvyzero.env.core import CurvyTronEnv
    from curvyzero.env.tracing import trace_scripted_actions

    env = CurvyTronEnv(CurvyTronConfig(action_repeat=1, max_ticks=64))
    trace = trace_scripted_actions(env, SCRIPTED_ACTIONS, seed=TRACE_SEED)
    trace_payload = trace.to_payload()

    return {
        "schema": "curvyzero_modal_python_trace_smoke/v1",
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "created_at": _created_at(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        },
        "scenario": {
            "id": "python-toy-v0-scripted-smoke",
            "seed": TRACE_SEED,
            "scripted_actions": SCRIPTED_ACTIONS,
        },
        "trace_scope": trace.scope,
        "trace_schema_version": trace.schema_version,
        "ruleset": trace_payload["ruleset"],
        "rules_hash": trace_payload["rules_hash"],
        "trace_fingerprint": trace.fingerprint,
        "trace": trace_payload,
        "message": "Python toy-v0 trace smoke only; JS oracle is intentionally not included.",
    }


@app.function(image=image, volumes={RUNS_MOUNT: runs_volume}, timeout=5 * 60)
def write_trace_fingerprint(
    run_id: str | None = None,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    clean_run_id = _clean_id(run_id or _new_id("run"), label="run_id")
    clean_attempt_id = _clean_id(attempt_id or _new_id("attempt"), label="attempt_id")
    ref = _artifact_ref(clean_run_id, clean_attempt_id)
    path = RUNS_MOUNT / Path(*ref.parts)

    payload = _build_trace_payload(clean_run_id, clean_attempt_id)
    body = _json_bytes(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)

    runs_volume.commit()

    return {
        "schema": "curvyzero_modal_python_trace_smoke_result/v1",
        "app_name": APP_NAME,
        "volume_name": VOLUME_NAME,
        "volume_mount": str(RUNS_MOUNT),
        "run_id": clean_run_id,
        "attempt_id": clean_attempt_id,
        "artifact_ref": ref.as_posix(),
        "artifact_path": str(path),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "trace_fingerprint": payload["trace_fingerprint"],
        "committed": True,
        "remote_elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


@app.local_entrypoint()
def main(run_id: str | None = None, attempt_id: str | None = None) -> None:
    started = time.perf_counter()
    result = write_trace_fingerprint.remote(run_id=run_id, attempt_id=attempt_id)
    result["client_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    print(json.dumps(result, indent=2, sort_keys=True))
