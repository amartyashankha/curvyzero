"""List or purge stale CurvyTron visual-survival runs from the Modal Volume.

Dry-run:
    uv run --extra modal modal run scripts/cleanup_curvytron_modal_runs.py --keep 1

Delete:
    uv run --extra modal modal run scripts/cleanup_curvytron_modal_runs.py --keep 1 --delete --yes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvyzero.infra.modal import run_management as runs  # noqa: E402


APP_NAME = "curvyzero-cleanup-curvytron-modal-runs"
TASK_ID = "lightzero-curvytron-visual-survival"
VOLUME_NAME = "curvyzero-runs"
REMOTE_ROOT = Path("/repo")
RUNS_MOUNT = Path("/runs")
BASE_REF = PurePosixPath("training") / TASK_ID
RUN_PICKER_FLAG_FILENAME = runs.GIF_BROWSER_RUN_MARKER_FILENAME
RUN_RECENCY_PATTERNS = (
    "attempts/*/eval/*/selfplay/summary.json",
    "attempts/*/eval/*/selfplay/raw.gif",
    "attempts/*/train/summary.json",
    "attempts/*/train/status.json",
    "attempts/*/train/status_heartbeat.json",
    "attempts/*/train/progress.jsonl",
    "attempts/*/train/progress_latest.json",
    "attempts/*/train/progress/latest.json",
    "attempts/*/train/checkpoint_eval_poller.json",
    "attempts/*/train/checkpoint.*",
    "attempts/*/train/checkpoints/*",
    "attempts/*/train/checkpoints/*/*",
    "attempts/*/train/lightzero_exp/ckpt/*",
    "checkpoints/latest.json",
    "checkpoints/best.json",
    "checkpoints/lightzero/*",
    "checkpoints/lightzero/*/*",
)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTHONPATH": str(REMOTE_ROOT / "src")})
    .add_local_dir(Path.cwd() / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


def _path_for_ref(mount: Path, ref: PurePosixPath) -> Path:
    return mount.joinpath(*ref.parts)


def _safe_stat(path: Path):
    try:
        return path.stat()
    except OSError:
        return None


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_iterdir(path: Path) -> list[Path]:
    try:
        return list(path.iterdir())
    except OSError:
        return []


def _safe_glob(path: Path, pattern: str) -> list[Path]:
    try:
        return list(path.glob(pattern))
    except OSError:
        return []


def _iso_from_ts(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat().replace("+00:00", "Z")


def _run_recency(run_path: Path) -> tuple[int, float | None]:
    artifact_count = 0
    updated_ts: float | None = None
    seen_artifacts: set[Path] = set()
    for pattern in RUN_RECENCY_PATTERNS:
        for artifact_path in _safe_glob(run_path, pattern):
            if artifact_path in seen_artifacts or not _safe_is_file(artifact_path):
                continue
            artifact_stat = _safe_stat(artifact_path)
            if artifact_stat is None:
                continue
            seen_artifacts.add(artifact_path)
            artifact_count += 1
            updated_ts = (
                artifact_stat.st_mtime
                if updated_ts is None
                else max(updated_ts, artifact_stat.st_mtime)
            )
    if updated_ts is None:
        run_stat = _safe_stat(run_path)
        updated_ts = run_stat.st_mtime if run_stat is not None else None
    return artifact_count, updated_ts


def _list_picker_runs(mount: Path) -> list[dict[str, Any]]:
    base_path = _path_for_ref(mount, BASE_REF)
    if not _safe_is_dir(base_path):
        return []

    rows: list[dict[str, Any]] = []
    for run_path in _safe_iterdir(base_path):
        marker_path = run_path / RUN_PICKER_FLAG_FILENAME
        if not _safe_is_dir(run_path) or not _safe_is_file(marker_path):
            continue
        try:
            run_id = runs.clean_id(run_path.name, label="run_id")
        except ValueError:
            continue
        artifact_count, updated_ts = _run_recency(run_path)
        if updated_ts is None:
            continue
        rows.append(
            {
                "run_id": run_id,
                "run_ref": (BASE_REF / run_id).as_posix(),
                "marker_ref": (BASE_REF / run_id / RUN_PICKER_FLAG_FILENAME).as_posix(),
                "artifact_count": artifact_count,
                "updated_at": _iso_from_ts(updated_ts),
                "updated_ts": updated_ts,
            }
        )
    rows.sort(key=lambda item: (item["updated_ts"], item["run_id"]), reverse=True)
    return rows


def _remove_run_dir(run_path: Path) -> str:
    if not _safe_is_dir(run_path):
        return "missing"
    shutil.rmtree(run_path)
    return "deleted"


def _remove_marker(marker_path: Path) -> str:
    if not _safe_is_file(marker_path):
        return "missing"
    marker_path.unlink()
    return "deleted"


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=20 * 60, cpu=1.0)
def cleanup_curvytron_runs_remote(
    *,
    keep: int = 1,
    delete: bool = False,
    yes: bool = False,
    markers_only: bool = False,
) -> dict[str, Any]:
    if keep < 0:
        raise ValueError("keep must be >= 0")
    if delete and not yes:
        raise ValueError("destructive cleanup requires --delete --yes")

    current_runs = _list_picker_runs(RUNS_MOUNT)
    kept = current_runs[:keep]
    stale = current_runs[keep:]
    actions: list[dict[str, Any]] = []

    for row in stale:
        run_path = _path_for_ref(RUNS_MOUNT, PurePosixPath(row["run_ref"]))
        marker_path = run_path / RUN_PICKER_FLAG_FILENAME
        action = "delete_marker" if markers_only else "delete_run_dir"
        status = "dry_run"
        if delete:
            status = _remove_marker(marker_path) if markers_only else _remove_run_dir(run_path)
        actions.append({**row, "action": action, "status": status})

    if delete and actions and hasattr(runs_volume, "commit"):
        runs_volume.commit()

    return {
        "schema_id": "curvyzero_curvytron_modal_run_cleanup/v1",
        "volume": VOLUME_NAME,
        "base_ref": BASE_REF.as_posix(),
        "dry_run": not delete,
        "delete": delete,
        "markers_only": markers_only,
        "keep": keep,
        "total_picker_runs": len(current_runs),
        "kept_count": len(kept),
        "stale_count": len(stale),
        "action_count": len(actions),
        "kept": kept,
        "actions": actions,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List or purge stale CurvyTron visual-survival Modal Volume runs. "
            "Only runs under training/lightzero-curvytron-visual-survival that have "
            f"{RUN_PICKER_FLAG_FILENAME} are considered."
        )
    )
    parser.add_argument("--keep", type=int, default=1, help="newest picker runs to keep")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="actually delete stale runs; omitted means dry-run",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="required with --delete to confirm destructive cleanup",
    )
    parser.add_argument(
        "--markers-only",
        action="store_true",
        help="only remove show_in_gif_browser.flag instead of deleting run directories",
    )
    return parser


@app.local_entrypoint()
def main(
    keep: int = 1,
    delete: bool = False,
    yes: bool = False,
    markers_only: bool = False,
) -> None:
    result = cleanup_curvytron_runs_remote.remote(
        keep=keep,
        delete=delete,
        yes=yes,
        markers_only=markers_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = cleanup_curvytron_runs_remote.remote(
        keep=args.keep,
        delete=args.delete,
        yes=args.yes,
        markers_only=args.markers_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
