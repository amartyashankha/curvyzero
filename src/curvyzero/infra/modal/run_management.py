"""Small helpers for Modal Volume training run artifacts.

The helpers here are intentionally boring: they validate run/attempt ids,
build relative Volume refs, write stable JSON, and describe pointer manifest
shapes. They do not implement training resume or Modal orchestration.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

SAFE_ID_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")

RUN_SCHEMA = "curvyzero_modal_training_run/v1"
ATTEMPT_SCHEMA = "curvyzero_modal_training_attempt/v1"
LATEST_ATTEMPT_SCHEMA = "curvyzero_modal_training_latest_attempt/v1"
CHECKPOINT_POINTER_SCHEMA = "curvyzero_modal_training_checkpoint_pointer/v1"
BEST_CHECKPOINT_SCHEMA = "curvyzero_modal_training_best_checkpoint/v1"
GIF_BROWSER_RUN_MARKER_SCHEMA = "curvyzero_modal_gif_browser_run_marker/v1"
GIF_BROWSER_RUN_MARKER_FILENAME = "show_in_gif_browser.flag"

ATTEMPT_STATUSES = {"running", "completed", "failed", "superseded"}


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for JSON manifests."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    """Return a compact UTC timestamp suitable for generated ids."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def new_id(prefix: str) -> str:
    """Create a validated id like ``run-20260509T010203Z-deadbeef1234``."""

    clean_prefix = clean_id(prefix, label="prefix")
    return f"{clean_prefix}-{utc_stamp()}-{uuid.uuid4().hex[:12]}"


def new_run_id(prefix: str = "run") -> str:
    return new_id(prefix)


def new_attempt_id(prefix: str = "attempt") -> str:
    return new_id(prefix)


def clean_id(raw: str, *, label: str) -> str:
    """Validate a path segment id used in Modal run-management refs."""

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


def run_root_ref(task_id: str, run_id: str) -> PurePosixPath:
    return (
        PurePosixPath("training")
        / clean_id(task_id, label="task_id")
        / clean_id(run_id, label="run_id")
    )


def run_manifest_ref(task_id: str, run_id: str) -> PurePosixPath:
    return run_root_ref(task_id, run_id) / "run.json"


def gif_browser_run_marker_ref(task_id: str, run_id: str) -> PurePosixPath:
    return run_root_ref(task_id, run_id) / GIF_BROWSER_RUN_MARKER_FILENAME


def latest_attempt_ref(task_id: str, run_id: str) -> PurePosixPath:
    return run_root_ref(task_id, run_id) / "latest_attempt.json"


def attempt_root_ref(task_id: str, run_id: str, attempt_id: str) -> PurePosixPath:
    return run_root_ref(task_id, run_id) / "attempts" / clean_id(attempt_id, label="attempt_id")


def attempt_manifest_ref(task_id: str, run_id: str, attempt_id: str) -> PurePosixPath:
    return attempt_root_ref(task_id, run_id, attempt_id) / "attempt.json"


def attempt_train_ref(task_id: str, run_id: str, attempt_id: str) -> PurePosixPath:
    return attempt_root_ref(task_id, run_id, attempt_id) / "train"


def attempt_eval_ref(
    task_id: str,
    run_id: str,
    attempt_id: str,
    eval_id: str = "final",
) -> PurePosixPath:
    return attempt_root_ref(task_id, run_id, attempt_id) / "eval" / clean_id(
        eval_id, label="eval_id"
    )


def checkpoints_root_ref(task_id: str, run_id: str) -> PurePosixPath:
    return run_root_ref(task_id, run_id) / "checkpoints"


def checkpoint_iteration_ref(
    task_id: str,
    run_id: str,
    completed_iterations: int,
) -> PurePosixPath:
    if completed_iterations < 1:
        raise ValueError("completed_iterations must be at least 1")
    return checkpoints_root_ref(task_id, run_id) / f"iteration-{completed_iterations:06d}"


def checkpoint_file_ref(
    task_id: str,
    run_id: str,
    completed_iterations: int,
) -> PurePosixPath:
    return checkpoint_iteration_ref(task_id, run_id, completed_iterations) / "checkpoint.npz"


def checkpoint_metadata_ref(
    task_id: str,
    run_id: str,
    completed_iterations: int,
) -> PurePosixPath:
    return checkpoint_iteration_ref(task_id, run_id, completed_iterations) / "metadata.json"


def latest_checkpoint_ref(task_id: str, run_id: str) -> PurePosixPath:
    return checkpoints_root_ref(task_id, run_id) / "latest.json"


def best_checkpoint_ref(task_id: str, run_id: str) -> PurePosixPath:
    return checkpoints_root_ref(task_id, run_id) / "best.json"


def eval_root_ref(task_id: str, run_id: str, eval_id: str) -> PurePosixPath:
    return run_root_ref(task_id, run_id) / "eval" / clean_id(eval_id, label="eval_id")


def require_relative_ref(ref: PurePosixPath | str) -> PurePosixPath:
    """Validate a Volume ref so callers cannot escape the mounted run root."""

    path = PurePosixPath(ref)
    if path.is_absolute() or not path.parts:
        raise ValueError("ref must be a non-empty relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("ref must not contain empty, dot, or parent segments")
    return path


def explicit_volume_ref(path_text: str) -> PurePosixPath | None:
    """Return the relative Volume ref from ``ref:``/``volume:`` inputs."""

    for prefix in ("ref:", "volume:"):
        if path_text.startswith(prefix):
            return require_relative_ref(path_text[len(prefix) :])
    return None


def volume_path(mount: Path, ref: PurePosixPath | str) -> Path:
    return mount / Path(*require_relative_ref(ref).parts)


def resolve_mounted_ref_or_path(
    path_text: str,
    *,
    mount: Path,
    remote_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve a Modal Volume ref or in-container path without checking file kind."""

    explicit_ref = explicit_volume_ref(path_text)
    if explicit_ref is not None:
        return volume_path(mount, explicit_ref), {
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

    if remote_root is not None:
        repo_candidate = remote_root / path_text
        if repo_candidate.exists():
            return repo_candidate, {
                "source_kind": "repo_path",
                "source_ref": None,
            }

    ref = require_relative_ref(path_text)
    return volume_path(mount, ref), {
        "source_kind": "volume_ref",
        "source_ref": ref.as_posix(),
    }


def file_ref(path: Path, *, mount: Path) -> str:
    return path.relative_to(mount).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_summary(path: Path, *, mount: Path) -> dict[str, Any]:
    return {
        "ref": file_ref(path, mount=mount),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def file_summary_any_mount(path: Path, *, mount: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    try:
        summary["ref"] = file_ref(path, mount=mount)
    except ValueError:
        pass
    return summary


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def write_json(path: Path, payload: Any, *, exclusive: bool = False) -> dict[str, Any]:
    """Write stable JSON and return byte/hash metadata for the written payload."""

    body = json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as handle:
        handle.write(body)
    return {
        "path": str(path),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def run_manifest(
    *,
    task_id: str,
    run_id: str,
    config: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "task_id": clean_id(task_id, label="task_id"),
        "run_id": clean_id(run_id, label="run_id"),
        "created_at": created_at or utc_timestamp(),
        "config": config or {},
    }


def gif_browser_run_marker(
    *,
    task_id: str,
    run_id: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": GIF_BROWSER_RUN_MARKER_SCHEMA,
        "task_id": clean_id(task_id, label="task_id"),
        "run_id": clean_id(run_id, label="run_id"),
        "created_at": created_at or utc_timestamp(),
        "purpose": "show this run in the CurvyTron GIF browser run picker",
    }


def attempt_manifest(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str | None = None,
    ended_at: str | None = None,
    modal_task_id: str | None = None,
    parent_checkpoint_ref: str | None = None,
    summary_ref: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in ATTEMPT_STATUSES:
        raise ValueError(f"status must be one of {sorted(ATTEMPT_STATUSES)}")
    return {
        "schema": ATTEMPT_SCHEMA,
        "task_id": clean_id(task_id, label="task_id"),
        "run_id": clean_id(run_id, label="run_id"),
        "attempt_id": clean_id(attempt_id, label="attempt_id"),
        "status": status,
        "started_at": started_at or utc_timestamp(),
        "ended_at": ended_at,
        "modal_task_id": modal_task_id,
        "parent_checkpoint_ref": parent_checkpoint_ref,
        "summary_ref": summary_ref,
        "config": config or {},
    }


def latest_attempt_pointer(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    status: str,
    started_at: str,
    ended_at: str | None = None,
    modal_task_id: str | None = None,
    summary_ref: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    if status not in ATTEMPT_STATUSES:
        raise ValueError(f"status must be one of {sorted(ATTEMPT_STATUSES)}")
    return {
        "schema": LATEST_ATTEMPT_SCHEMA,
        "task_id": clean_id(task_id, label="task_id"),
        "run_id": clean_id(run_id, label="run_id"),
        "attempt_id": clean_id(attempt_id, label="attempt_id"),
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "modal_task_id": modal_task_id,
        "summary_ref": summary_ref,
        "updated_at": updated_at or utc_timestamp(),
    }


def checkpoint_pointer(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    completed_iterations: int,
    checkpoint_ref: str,
    metadata_ref: str | None = None,
    seed_cursor: dict[str, Any] | None = None,
    artifact_hashes: dict[str, str] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    if completed_iterations < 1:
        raise ValueError("completed_iterations must be at least 1")
    return {
        "schema": CHECKPOINT_POINTER_SCHEMA,
        "task_id": clean_id(task_id, label="task_id"),
        "run_id": clean_id(run_id, label="run_id"),
        "attempt_id": clean_id(attempt_id, label="attempt_id"),
        "completed_iterations": completed_iterations,
        "checkpoint_ref": require_relative_ref(checkpoint_ref).as_posix(),
        "metadata_ref": require_relative_ref(metadata_ref).as_posix() if metadata_ref else None,
        "seed_cursor": seed_cursor or {},
        "artifact_hashes": artifact_hashes or {},
        "updated_at": updated_at or utc_timestamp(),
    }


def best_checkpoint_pointer(
    *,
    task_id: str,
    run_id: str,
    eval_id: str,
    ranking_metric: str,
    metric_value: int | float,
    checkpoint_ref: str,
    higher_is_better: bool = True,
    metadata_ref: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": BEST_CHECKPOINT_SCHEMA,
        "task_id": clean_id(task_id, label="task_id"),
        "run_id": clean_id(run_id, label="run_id"),
        "eval_id": clean_id(eval_id, label="eval_id"),
        "ranking_metric": ranking_metric,
        "metric_value": metric_value,
        "higher_is_better": higher_is_better,
        "checkpoint_ref": require_relative_ref(checkpoint_ref).as_posix(),
        "metadata_ref": require_relative_ref(metadata_ref).as_posix() if metadata_ref else None,
        "updated_at": updated_at or utc_timestamp(),
    }
