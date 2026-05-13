"""List or purge stale CurvyTron visual-survival runs from the Modal Volume.

Dry-run:
    uv run --extra modal modal run scripts/cleanup_curvytron_modal_runs.py --keep 1

Delete:
    uv run --extra modal modal run scripts/cleanup_curvytron_modal_runs.py --keep 1 --delete --yes

Manifest allowlist dry-run:
    uv run --extra modal modal run scripts/cleanup_curvytron_modal_runs.py \
        --purge-unpreserved \
        --preserve-manifest artifacts/local/curvytron_survivaldiag_manifests/example.json \
        --report-path artifacts/local/example.cleanup_dry_run.json \
        --output-detail compact
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


def _list_direct_run_dirs(mount: Path) -> dict[str, Any]:
    base_path = _path_for_ref(mount, BASE_REF)
    if not _safe_is_dir(base_path):
        return {"runs": [], "skipped": [{"ref": BASE_REF.as_posix(), "reason": "missing_base"}]}

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for child_path in sorted(_safe_iterdir(base_path), key=lambda path: path.name):
        child_ref = (BASE_REF / child_path.name).as_posix()
        if not _safe_is_dir(child_path):
            skipped.append(
                {
                    "name": child_path.name,
                    "ref": child_ref,
                    "kind": "non_dir",
                    "reason": "not_a_run_directory",
                }
            )
            continue
        try:
            run_id = runs.clean_id(child_path.name, label="run_id")
        except ValueError as exc:
            skipped.append(
                {
                    "name": child_path.name,
                    "ref": child_ref,
                    "kind": "dir",
                    "reason": "invalid_run_id",
                    "error": str(exc),
                }
            )
            continue

        artifact_count, updated_ts = _run_recency(child_path)
        rows.append(
            {
                "run_id": run_id,
                "run_ref": child_ref,
                "marker_ref": (BASE_REF / run_id / RUN_PICKER_FLAG_FILENAME).as_posix(),
                "has_picker_marker": _safe_is_file(child_path / RUN_PICKER_FLAG_FILENAME),
                "artifact_count": artifact_count,
                "updated_at": _iso_from_ts(updated_ts) if updated_ts is not None else None,
                "updated_ts": updated_ts,
                "category": _run_category(run_id),
            }
        )

    return {"runs": rows, "skipped": skipped}


def _run_category(run_id: str) -> str:
    if run_id.startswith("curvy-survive-bonus-"):
        return "curvy_survive_bonus"
    if run_id.startswith("survivaldiag-v1b-20260513h-"):
        return "survivaldiag_v1b_20260513h"
    if run_id.startswith("survivaldiag-"):
        return "survivaldiag_other"
    if run_id.startswith("stock-") or "stock" in run_id:
        return "stock_lightzero"
    if "wall" in run_id:
        return "wall_avoidance_or_wall_probe"
    if "curvy" in run_id or "survival" in run_id:
        return "curvytron_visual_survival_other"
    return "other"


def _count_categories(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def _matching_preserve_reasons(
    run_id: str,
    *,
    preserve_run_ids: set[str],
    preserve_prefixes: tuple[str, ...],
) -> list[str]:
    reasons: list[str] = []
    if run_id in preserve_run_ids:
        reasons.append("run_id")
    for prefix in preserve_prefixes:
        if run_id.startswith(prefix):
            reasons.append(f"prefix:{prefix}")
    return reasons


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


def _remove_run_ref_volume_api(run_ref: str) -> str:
    try:
        runs_volume.remove_file(run_ref, recursive=True)
    except FileNotFoundError:
        return "missing"
    except Exception as exc:  # noqa: BLE001
        return f"error:{type(exc).__name__}:{exc}"
    return "deleted"


@app.function(image=image, volumes={str(RUNS_MOUNT): runs_volume}, timeout=60 * 60, cpu=1.0)
def cleanup_curvytron_runs_allowlist_remote(
    *,
    preserve_run_ids: list[str],
    preserve_prefixes: list[str] | None = None,
    delete: bool = False,
    yes: bool = False,
    markers_only: bool = False,
    action_limit: int | None = None,
    delete_method: str = "mounted",
) -> dict[str, Any]:
    if delete and not yes:
        raise ValueError("destructive cleanup requires --delete --yes")
    if action_limit is not None and action_limit < 1:
        raise ValueError("action_limit must be positive when set")
    if delete_method not in {"mounted", "volume-api"}:
        raise ValueError("delete_method must be 'mounted' or 'volume-api'")
    clean_preserve_run_ids = sorted(
        {runs.clean_id(run_id, label="preserve_run_id") for run_id in preserve_run_ids}
    )
    clean_preserve_prefixes = tuple(
        sorted(
            {
                runs.clean_id(prefix, label="preserve_prefix")
                for prefix in (preserve_prefixes or [])
            }
        )
    )
    if delete and not clean_preserve_run_ids and not clean_preserve_prefixes:
        raise ValueError("refusing to delete without at least one preserved run id or prefix")

    listing = _list_direct_run_dirs(RUNS_MOUNT)
    all_runs = listing["runs"]
    preserve_run_id_set = set(clean_preserve_run_ids)
    existing_run_ids = {str(row["run_id"]) for row in all_runs}
    preserved: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    for row in all_runs:
        run_id = str(row["run_id"])
        preserve_reasons = _matching_preserve_reasons(
            run_id,
            preserve_run_ids=preserve_run_id_set,
            preserve_prefixes=clean_preserve_prefixes,
        )
        if preserve_reasons:
            preserved.append({**row, "preserve_reasons": preserve_reasons})
            continue

        run_path = _path_for_ref(RUNS_MOUNT, PurePosixPath(row["run_ref"]))
        marker_path = run_path / RUN_PICKER_FLAG_FILENAME
        action = "delete_marker" if markers_only else "delete_run_dir"
        status = "dry_run"
        if delete:
            if markers_only:
                status = _remove_marker(marker_path)
            elif delete_method == "volume-api":
                status = _remove_run_ref_volume_api(str(row["run_ref"]))
            else:
                status = _remove_run_dir(run_path)
        actions.append({**row, "action": action, "status": status})
        if status.startswith("error:"):
            break
        if action_limit is not None and len(actions) >= action_limit:
            break

    if delete and actions and delete_method == "mounted" and hasattr(runs_volume, "commit"):
        runs_volume.commit()

    missing_preserved_ids = sorted(preserve_run_id_set - existing_run_ids)
    prefix_only_preserved = [
        row
        for row in preserved
        if "run_id" not in row["preserve_reasons"]
    ]
    return {
        "schema_id": "curvyzero_curvytron_modal_run_cleanup_allowlist/v1",
        "volume": VOLUME_NAME,
        "base_ref": BASE_REF.as_posix(),
        "dry_run": not delete,
        "delete": delete,
        "action_limit": action_limit,
        "delete_method": delete_method,
        "preserve_run_id_count": len(clean_preserve_run_ids),
        "preserve_prefixes": list(clean_preserve_prefixes),
        "direct_run_dir_count": len(all_runs),
        "skipped_count": len(listing["skipped"]),
        "preserved_count": len(preserved),
        "prefix_only_preserved_count": len(prefix_only_preserved),
        "missing_preserved_id_count": len(missing_preserved_ids),
        "action_count": len(actions),
        "preserved_categories": _count_categories(preserved),
        "delete_categories": _count_categories(actions),
        "preserve_run_ids": clean_preserve_run_ids,
        "missing_preserved_ids": missing_preserved_ids,
        "prefix_only_preserved": prefix_only_preserved,
        "preserved": preserved,
        "actions": actions,
        "skipped": listing["skipped"],
    }


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
    parser.add_argument(
        "--purge-unpreserved",
        action="store_true",
        help=(
            "delete/dry-run every direct run directory under the CurvyTron task "
            "that is not matched by --preserve-manifest, --preserve-run-id, or "
            "--preserve-prefix"
        ),
    )
    parser.add_argument(
        "--action-limit",
        type=int,
        default=None,
        help="limit unpreserved actions for small cleanup waves",
    )
    parser.add_argument(
        "--delete-method",
        choices=("mounted", "volume-api"),
        default="mounted",
        help="delete run directories through the mounted path or Modal volume API",
    )
    parser.add_argument(
        "--preserve-manifest",
        action="append",
        default=[],
        help="JSON/JSONL manifest path containing rows with run_id values; repeatable",
    )
    parser.add_argument(
        "--preserve-run-id",
        action="append",
        default=[],
        help="specific run_id to preserve; repeatable",
    )
    parser.add_argument(
        "--preserve-prefix",
        action="append",
        default=[],
        help="run_id prefix to preserve; repeatable",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="optional local path for writing the full JSON cleanup report",
    )
    parser.add_argument(
        "--output-detail",
        choices=("full", "compact"),
        default="full",
        help="print the full report or a compact summary to stdout",
    )
    return parser


def _split_items(items: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if items is None:
        return []
    if isinstance(items, str):
        raw_items = [items]
    else:
        raw_items = list(items)
    split: list[str] = []
    for item in raw_items:
        for part in str(item).replace("\n", ",").split(","):
            clean_part = part.strip()
            if clean_part:
                split.append(clean_part)
    return split


def _extract_run_ids_from_manifest_payload(payload: Any) -> list[str]:
    rows: Any
    if isinstance(payload, dict):
        if isinstance(payload.get("rows"), list):
            rows = payload["rows"]
        elif isinstance(payload.get("launches"), list):
            rows = payload["launches"]
        elif isinstance(payload.get("runs"), list):
            rows = payload["runs"]
        elif isinstance(payload.get("run_id"), str):
            rows = [payload]
        else:
            rows = []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    run_ids: list[str] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("run_id"), str):
            run_ids.append(row["run_id"])
    return run_ids


def _load_manifest_run_ids(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        run_ids: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
                run_ids.extend(_extract_run_ids_from_manifest_payload(payload))
        return run_ids

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _extract_run_ids_from_manifest_payload(payload)


def _build_preserve_inputs(
    *,
    manifest_paths: list[str],
    explicit_run_ids: list[str],
    prefixes: list[str],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    run_ids: set[str] = set()
    sources: list[dict[str, Any]] = []
    for manifest_path_text in manifest_paths:
        manifest_path = Path(manifest_path_text)
        manifest_run_ids = _load_manifest_run_ids(manifest_path)
        clean_manifest_run_ids = [
            runs.clean_id(run_id, label=f"run_id from {manifest_path}") for run_id in manifest_run_ids
        ]
        run_ids.update(clean_manifest_run_ids)
        sources.append(
            {
                "path": str(manifest_path),
                "run_id_count": len(clean_manifest_run_ids),
                "unique_run_id_count": len(set(clean_manifest_run_ids)),
                "first_run_id": clean_manifest_run_ids[0] if clean_manifest_run_ids else None,
                "last_run_id": clean_manifest_run_ids[-1] if clean_manifest_run_ids else None,
            }
        )

    for run_id in explicit_run_ids:
        run_ids.add(runs.clean_id(run_id, label="preserve_run_id"))
    clean_prefixes = [runs.clean_id(prefix, label="preserve_prefix") for prefix in prefixes]
    return sorted(run_ids), sorted(set(clean_prefixes)), sources


def _write_report(path_text: str, result: dict[str, Any]) -> None:
    report_path = Path(path_text)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compact_keys = (
        "schema_id",
        "volume",
        "base_ref",
        "dry_run",
        "delete",
        "action_limit",
        "delete_method",
        "preserve_sources",
        "preserve_run_id_count",
        "preserve_prefixes",
        "direct_run_dir_count",
        "skipped_count",
        "preserved_count",
        "prefix_only_preserved_count",
        "missing_preserved_id_count",
        "action_count",
        "preserved_categories",
        "delete_categories",
        "report_path",
    )
    compact = {key: result[key] for key in compact_keys if key in result}
    compact["sample_preserved"] = result.get("preserved", [])[:10]
    compact["sample_actions"] = result.get("actions", [])[:25]
    compact["sample_skipped"] = result.get("skipped", [])[:10]
    compact["sample_missing_preserved_ids"] = result.get("missing_preserved_ids", [])[:25]
    compact["sample_prefix_only_preserved"] = result.get("prefix_only_preserved", [])[:25]
    return compact


@app.local_entrypoint()
def main(
    keep: int = 1,
    delete: bool = False,
    yes: bool = False,
    markers_only: bool = False,
    purge_unpreserved: bool = False,
    preserve_manifest: str = "",
    preserve_run_id: str = "",
    preserve_prefix: str = "",
    report_path: str = "",
    output_detail: str = "full",
    action_limit: int | None = None,
    delete_method: str = "mounted",
) -> None:
    if output_detail not in {"full", "compact"}:
        raise ValueError("output_detail must be 'full' or 'compact'")

    manifest_paths = _split_items(preserve_manifest)
    explicit_run_ids = _split_items(preserve_run_id)
    prefixes = _split_items(preserve_prefix)
    if purge_unpreserved or manifest_paths or explicit_run_ids or prefixes:
        if not purge_unpreserved:
            raise ValueError("allowlist cleanup requires --purge-unpreserved")
        preserve_run_ids, preserve_prefixes, preserve_sources = _build_preserve_inputs(
            manifest_paths=manifest_paths,
            explicit_run_ids=explicit_run_ids,
            prefixes=prefixes,
        )
        result = cleanup_curvytron_runs_allowlist_remote.remote(
            preserve_run_ids=preserve_run_ids,
            preserve_prefixes=preserve_prefixes,
            delete=delete,
            yes=yes,
            markers_only=markers_only,
            action_limit=action_limit,
            delete_method=delete_method,
        )
        result["preserve_sources"] = preserve_sources
    else:
        result = cleanup_curvytron_runs_remote.remote(
            keep=keep,
            delete=delete,
            yes=yes,
            markers_only=markers_only,
        )

    if report_path:
        _write_report(report_path, result)
        result["report_path"] = report_path

    printable = _compact_result(result) if output_detail == "compact" else result
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    args = _build_parser().parse_args()
    manifest_paths = _split_items(args.preserve_manifest)
    explicit_run_ids = _split_items(args.preserve_run_id)
    prefixes = _split_items(args.preserve_prefix)
    if args.purge_unpreserved or manifest_paths or explicit_run_ids or prefixes:
        if not args.purge_unpreserved:
            raise ValueError("allowlist cleanup requires --purge-unpreserved")
        preserve_run_ids, preserve_prefixes, preserve_sources = _build_preserve_inputs(
            manifest_paths=manifest_paths,
            explicit_run_ids=explicit_run_ids,
            prefixes=prefixes,
        )
        result = cleanup_curvytron_runs_allowlist_remote.remote(
            preserve_run_ids=preserve_run_ids,
            preserve_prefixes=preserve_prefixes,
            delete=args.delete,
            yes=args.yes,
            markers_only=args.markers_only,
            action_limit=args.action_limit,
            delete_method=args.delete_method,
        )
        result["preserve_sources"] = preserve_sources
    else:
        result = cleanup_curvytron_runs_remote.remote(
            keep=args.keep,
            delete=args.delete,
            yes=args.yes,
            markers_only=args.markers_only,
        )

    if args.report_path:
        _write_report(args.report_path, result)
        result["report_path"] = args.report_path

    printable = _compact_result(result) if args.output_detail == "compact" else result
    print(json.dumps(printable, indent=2, sort_keys=True))
