"""Tiny Modal ASGI browser for CurvyTron checkpoint self-play GIFs.

Usage:

    uv run --extra modal modal serve -m curvyzero.infra.modal.curvytron_gif_browser
    uv run --extra modal modal deploy -m curvyzero.infra.modal.curvytron_gif_browser
"""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from html import escape
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any
from urllib.parse import quote, urlencode

from curvyzero.infra.modal import run_management as run_mgmt

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - lets helper tests run without modal.
    modal = None  # type: ignore[assignment]


APP_NAME = "curvyzero-curvytron-gif-browser"
TASK_ID = "lightzero-curvytron-visual-survival"
VOLUME_NAME = "curvyzero-runs"
RUNS_MOUNT = Path("/runs")
BASE_REF = PurePosixPath("training") / TASK_ID
DEFAULT_LIMIT = 8
MAX_LIMIT = 500
DEFAULT_OK_FILTER = "ok"
LISTING_CACHE_TTL_SECONDS = 10.0
VOLUME_RELOAD_TTL_SECONDS = 30.0
GIF_CACHE_MAX_AGE_SECONDS = 86_400
DYNAMIC_RESPONSE_HEADERS = {"Cache-Control": "no-store"}
RUN_PICKER_FLAG_FILENAME = run_mgmt.GIF_BROWSER_RUN_MARKER_FILENAME
GIF_VARIANT_EVAL_GREEDY = "eval_greedy"
GIF_VARIANT_COLLECT_T1 = "collect_t1"
GIF_VARIANT_LABELS = {
    GIF_VARIANT_EVAL_GREEDY: "Greedy eval",
    GIF_VARIANT_COLLECT_T1: "Collect T=1",
}
GIF_VARIANT_FILENAMES = {
    GIF_VARIANT_EVAL_GREEDY: "raw.gif",
    GIF_VARIANT_COLLECT_T1: "collect_t1.gif",
}
ALLOWED_SELFPLAY_GIF_FILENAMES = frozenset(GIF_VARIANT_FILENAMES.values())
CHECKPOINT_ITERATION_RE = re.compile(r"iteration[_-](\d+)")
RUN_RECENCY_PATTERNS = (
    RUN_PICKER_FLAG_FILENAME,
    "run.json",
    "latest_attempt.json",
    "attempts/*/attempt.json",
    "attempts/*/command.json",
    "attempts/*/train/summary.json",
    "attempts/*/train/status.json",
    "attempts/*/train/progress_latest.json",
    "attempts/*/train/progress/latest.json",
    "attempts/*/train/checkpoint_eval_poller.json",
    "checkpoints/latest.json",
    "checkpoints/best.json",
)

if modal is not None:
    image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("fastapi>=0.115")
    runs_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    app = modal.App(APP_NAME)
else:  # pragma: no cover - test/import convenience only.
    image = None
    runs_volume = None
    app = None


class RefValidationError(ValueError):
    """Raised when a requested Volume ref is outside the browser's allowed tree."""


_RUNS_CACHE: dict[tuple[Any, ...], tuple[float, Any]] = {}
_SUMMARY_CACHE: dict[tuple[Any, ...], tuple[float, Any]] = {}
_VOLUME_IO_LOCK = threading.Lock()
_LAST_VOLUME_RELOAD_AT = 0.0


def _clone_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _clone_cached_value(value: Any) -> Any:
    if isinstance(value, list):
        return _clone_rows(value)
    if isinstance(value, dict):
        cloned = dict(value)
        if isinstance(cloned.get("rows"), list):
            cloned["rows"] = _clone_rows(cloned["rows"])
        return cloned
    return value


def _cache_get(
    cache: dict[tuple[Any, ...], tuple[float, Any]],
    key: tuple[Any, ...],
) -> Any | None:
    if LISTING_CACHE_TTL_SECONDS <= 0:
        return None
    cached = cache.get(key)
    if cached is None:
        return None
    expires_at, value = cached
    if monotonic() >= expires_at:
        cache.pop(key, None)
        return None
    return _clone_cached_value(value)


def _cache_set(
    cache: dict[tuple[Any, ...], tuple[float, Any]],
    key: tuple[Any, ...],
    value: Any,
) -> Any:
    if LISTING_CACHE_TTL_SECONDS > 0:
        cache[key] = (monotonic() + LISTING_CACHE_TTL_SECONDS, _clone_cached_value(value))
    return value


def _clear_listing_caches() -> None:
    _RUNS_CACHE.clear()
    _SUMMARY_CACHE.clear()


def _maybe_reload_volume(volume: Any, *, force: bool = False) -> str | None:
    global _LAST_VOLUME_RELOAD_AT
    if volume is None or not hasattr(volume, "reload"):
        return None
    now = monotonic()
    if not force and now - _LAST_VOLUME_RELOAD_AT < VOLUME_RELOAD_TTL_SECONDS:
        return None
    if not _VOLUME_IO_LOCK.acquire(blocking=False):
        return None
    try:
        now = monotonic()
        if not force and now - _LAST_VOLUME_RELOAD_AT < VOLUME_RELOAD_TTL_SECONDS:
            return None
        try:
            volume.reload()
        except Exception as exc:  # pragma: no cover - Modal Volume runtime behavior.
            _LAST_VOLUME_RELOAD_AT = monotonic()
            if "open files preventing the operation" in str(exc):
                return None
            return f"{type(exc).__name__}: {exc}"
        _LAST_VOLUME_RELOAD_AT = monotonic()
        _clear_listing_caches()
        return None
    finally:
        _VOLUME_IO_LOCK.release()


def _path_for_ref(mount: Path, ref: PurePosixPath) -> Path:
    return mount.joinpath(*ref.parts)


def _validate_volume_ref(ref_text: str, *, suffix: str) -> PurePosixPath:
    """Validate a user-provided Volume ref before serving it over HTTP."""

    if not ref_text:
        raise RefValidationError("ref is required")
    if "\x00" in ref_text or "\\" in ref_text:
        raise RefValidationError("ref contains an invalid path character")
    if ref_text.startswith("/"):
        raise RefValidationError("ref must be relative")
    ref = PurePosixPath(ref_text)
    if ref.is_absolute() or not ref.parts:
        raise RefValidationError("ref must be a relative path")
    if any(part in {"", ".", ".."} for part in ref.parts):
        raise RefValidationError("ref must not contain dot or parent segments")
    if ref.parts[: len(BASE_REF.parts)] != BASE_REF.parts or len(ref.parts) <= len(
        BASE_REF.parts
    ):
        raise RefValidationError(f"ref must live under {BASE_REF.as_posix()}")
    if ref.suffix.lower() != suffix:
        raise RefValidationError(f"ref must end with {suffix}")
    return ref


def _ref_from_path(mount: Path, path: Path) -> PurePosixPath:
    return PurePosixPath(path.relative_to(mount).as_posix())


def _extract_selfplay_artifact_ids(
    artifact_ref: PurePosixPath, *, filename: str
) -> dict[str, str] | None:
    parts = artifact_ref.parts
    base_len = len(BASE_REF.parts)
    expected_tail = ("attempts", None, "eval", None, "selfplay", filename)
    if len(parts) != base_len + 1 + len(expected_tail):
        return None
    tail = parts[base_len + 1 :]
    if (
        tail[0] != "attempts"
        or tail[2] != "eval"
        or tail[4] != "selfplay"
        or tail[5] != filename
    ):
        return None
    return {
        "run_id": parts[base_len],
        "attempt_id": tail[1],
        "eval_id": tail[3],
    }


def _extract_artifact_ids(summary_ref: PurePosixPath) -> dict[str, str] | None:
    return _extract_selfplay_artifact_ids(summary_ref, filename="summary.json")


def _validate_selfplay_summary_ref(ref_text: str) -> PurePosixPath:
    ref = _validate_volume_ref(ref_text, suffix=".json")
    if _extract_artifact_ids(ref) is None:
        raise RefValidationError("ref must be a selfplay summary.json artifact")
    return ref


def _validate_selfplay_gif_ref(ref_text: str) -> PurePosixPath:
    ref = _validate_volume_ref(ref_text, suffix=".gif")
    if ref.name not in ALLOWED_SELFPLAY_GIF_FILENAMES or _extract_selfplay_artifact_ids(
        ref,
        filename=ref.name,
    ) is None:
        raise RefValidationError("ref must be a known selfplay GIF artifact")
    return ref


def _read_json_object(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        with _VOLUME_IO_LOCK:
            value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return {}, "summary JSON was not an object"
    return value, None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _matches_text(value: str, needle: str) -> bool:
    return not needle or needle.lower() in value.lower()


def _matches_ok(row: dict[str, Any], ok_filter: str) -> bool:
    normalized = ok_filter.lower()
    if normalized in {"", "all", "any"}:
        return True
    is_ok = row.get("ok") is True and row.get("gif_exists") is True
    if normalized in {"1", "true", "ok", "pass", "passed"}:
        return is_ok
    if normalized in {"0", "false", "fail", "failed"}:
        return not is_ok
    return True


def _coerce_limit(limit: int) -> int:
    return min(max(1, int(limit)), MAX_LIMIT)


def _coerce_offset(offset: int) -> int:
    return max(0, int(offset))


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


def _stat_token(path: Path) -> tuple[int, int] | None:
    stat = _safe_stat(path)
    if stat is None:
        return None
    return stat.st_mtime_ns, stat.st_size


def _runs_cache_state_token(mount: Path) -> tuple[Any, ...]:
    base_path = _path_for_ref(mount, BASE_REF)
    run_tokens: list[tuple[str, tuple[int, int] | None]] = []
    for run_path in _safe_iterdir(base_path):
        if not _safe_is_dir(run_path):
            continue
        marker_token = _stat_token(run_path / RUN_PICKER_FLAG_FILENAME)
        if marker_token is None:
            continue
        run_tokens.append((run_path.name, marker_token))
    return (mount.as_posix(), _stat_token(base_path), tuple(sorted(run_tokens)))


def _run_summary_cache_state_token(mount: Path, run_id: str) -> tuple[Any, ...]:
    run_path = _path_for_ref(mount, BASE_REF / run_id)
    eval_dir_tokens: list[tuple[str, tuple[int, int] | None]] = []
    for eval_dir in _safe_glob(run_path, "attempts/*/eval"):
        if not _safe_is_dir(eval_dir):
            continue
        eval_dir_tokens.append(
            (eval_dir.relative_to(run_path).as_posix(), _stat_token(eval_dir))
        )
    return (
        run_id,
        _stat_token(run_path / RUN_PICKER_FLAG_FILENAME),
        tuple(sorted(eval_dir_tokens)),
    )


def _run_has_picker_flag(run_path: Path) -> bool:
    return _safe_is_file(run_path / RUN_PICKER_FLAG_FILENAME)


def _run_picker_flag_path(mount: Path, run_id: str) -> Path:
    clean_run_id = run_mgmt.clean_id(run_id, label="run_id")
    return _path_for_ref(mount, run_mgmt.gif_browser_run_marker_ref(TASK_ID, clean_run_id))


def _hide_run_from_picker_on_mount(
    *,
    mount: Path,
    run_id: str,
    volume: Any = None,
) -> dict[str, Any]:
    clean_run_id = run_mgmt.clean_id(run_id, label="run_id")
    marker_path = _run_picker_flag_path(mount, clean_run_id)
    existed = _safe_is_file(marker_path)
    hidden = False
    if existed:
        try:
            marker_path.unlink()
            hidden = True
        except FileNotFoundError:
            existed = False
    if volume is not None and hasattr(volume, "commit"):
        volume.commit()
    _clear_listing_caches()
    return {
        "ok": True,
        "run_id": clean_run_id,
        "marker_ref": run_mgmt.gif_browser_run_marker_ref(
            TASK_ID,
            clean_run_id,
        ).as_posix(),
        "marker_existed": existed,
        "hidden": hidden,
    }


def _safe_next_url(value: str) -> str:
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/"
    if "\x00" in value or "\\" in value:
        return "/"
    return value


def _gif_etag(path: Path, stat: Any) -> str:
    token = f"{path.name}-{stat.st_mtime_ns:x}-{stat.st_size:x}"
    return f'"{token}"'


def _etag_matches(header_value: str, etag: str) -> bool:
    return any(value.strip() == etag for value in header_value.split(","))


def _list_runs(mount: Path) -> list[dict[str, Any]]:
    base_path = _path_for_ref(mount, BASE_REF)
    if not _safe_is_dir(base_path):
        return []

    cache_key = ("runs", mount.as_posix())
    cached = _cache_get(_RUNS_CACHE, cache_key)
    if cached is not None:
        return cached

    runs: list[dict[str, Any]] = []
    for run_path in _safe_iterdir(base_path):
        if not _safe_is_dir(run_path) or not _run_has_picker_flag(run_path):
            continue
        artifact_count = 0
        updated_ts: float | None = None
        seen_artifacts: set[Path] = set()
        for pattern in RUN_RECENCY_PATTERNS:
            for artifact_path in _safe_glob(run_path, pattern):
                if artifact_path in seen_artifacts:
                    continue
                artifact_stat = _safe_stat(artifact_path)
                if artifact_stat is None:
                    continue
                if not _safe_is_file(artifact_path):
                    continue
                seen_artifacts.add(artifact_path)
                if artifact_path.name != RUN_PICKER_FLAG_FILENAME:
                    artifact_count += 1
                updated_ts = (
                    artifact_stat.st_mtime
                    if updated_ts is None
                    else max(updated_ts, artifact_stat.st_mtime)
                )
        if updated_ts is None:
            run_stat = _safe_stat(run_path)
            if run_stat is None:
                continue
            updated_ts = run_stat.st_mtime
        runs.append(
            {
                "run_id": run_path.name,
                "artifact_count": artifact_count,
                "updated_at": datetime.fromtimestamp(updated_ts, UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                "updated_ts": updated_ts,
            }
        )

    runs.sort(key=lambda item: (item["updated_ts"], item["run_id"]), reverse=True)
    return _cache_set(_RUNS_CACHE, cache_key, runs)


def _checkpoint_iteration(row: dict[str, Any]) -> int | None:
    for key in ("checkpoint_label", "eval_id", "checkpoint_ref"):
        value = row.get(key)
        if not isinstance(value, str):
            continue
        match = CHECKPOINT_ITERATION_RE.search(value)
        if match is not None:
            return int(match.group(1))
    return None


def _summary_sort_key(
    row: dict[str, Any],
    *,
    run_rank_by_id: dict[str, int],
) -> tuple[int, int, float, str]:
    iteration = _checkpoint_iteration(row)
    iteration_key = -iteration if iteration is not None else 1
    updated_ts = float(row.get("updated_ts") or 0.0)
    return (
        run_rank_by_id.get(str(row.get("run_id", "")), len(run_rank_by_id)),
        iteration_key,
        -updated_ts,
        str(row.get("eval_id", "")),
    )


def _summary_path_sort_key(path: Path) -> tuple[int, int, str]:
    ref_text = path.as_posix()
    match = CHECKPOINT_ITERATION_RE.search(ref_text)
    iteration_key = -int(match.group(1)) if match is not None else 1
    return (iteration_key, 0 if match is not None else 1, ref_text)


def _default_selected_run_id(runs: list[dict[str, Any]], requested_run_id: str) -> str:
    if requested_run_id:
        try:
            clean_run_id = run_mgmt.clean_id(requested_run_id, label="run_id")
        except ValueError:
            return ""
        visible_run_ids = {str(run["run_id"]) for run in runs}
        if clean_run_id in visible_run_ids:
            return clean_run_id
    if not runs:
        return ""
    return str(runs[0]["run_id"])


def _gif_variant_rows(
    mount: Path,
    *,
    summary: dict[str, Any],
    summary_ref: PurePosixPath,
    legacy_gif_ref: PurePosixPath,
) -> list[dict[str, Any]]:
    raw_variants = summary.get("gif_variants")
    raw_variants = raw_variants if isinstance(raw_variants, dict) else {}
    variants: list[dict[str, Any]] = []
    for variant_id in (GIF_VARIANT_EVAL_GREEDY, GIF_VARIANT_COLLECT_T1):
        default_ref = summary_ref.parent / GIF_VARIANT_FILENAMES[variant_id]
        if variant_id == GIF_VARIANT_EVAL_GREEDY:
            default_ref = legacy_gif_ref
        variant_summary = raw_variants.get(variant_id)
        variant_summary = variant_summary if isinstance(variant_summary, dict) else {}
        ref_text = variant_summary.get("gif_ref")
        try:
            gif_ref = _validate_selfplay_gif_ref(
                ref_text if isinstance(ref_text, str) else default_ref.as_posix()
            )
            expected_filename = GIF_VARIANT_FILENAMES[variant_id]
            if gif_ref.parent != summary_ref.parent or gif_ref.name != expected_filename:
                gif_ref = default_ref
        except RefValidationError:
            gif_ref = default_ref
        gif_path = _path_for_ref(mount, gif_ref)
        gif_stat = _safe_stat(gif_path)
        gif_exists = gif_stat is not None and _safe_is_file(gif_path)
        variants.append(
            {
                "variant_id": variant_id,
                "label": variant_summary.get("label") or GIF_VARIANT_LABELS[variant_id],
                "policy_mode": variant_summary.get("policy_mode"),
                "temperature": variant_summary.get("temperature"),
                "epsilon": variant_summary.get("epsilon"),
                "gif_ref": gif_ref.as_posix(),
                "gif_exists": gif_exists,
                "gif_bytes": gif_stat.st_size if gif_exists and gif_stat is not None else None,
                "gif_updated_ts": (
                    gif_stat.st_mtime if gif_exists and gif_stat is not None else None
                ),
                "gif_updated_ns": (
                    gif_stat.st_mtime_ns if gif_exists and gif_stat is not None else None
                ),
                "frame_count": _safe_int(
                    variant_summary.get("frame_count", summary.get("frame_count"))
                ),
                "physical_steps": _safe_int(
                    variant_summary.get("physical_steps", summary.get("physical_steps"))
                ),
                "max_steps": _safe_int(
                    variant_summary.get("max_steps", summary.get("max_steps"))
                ),
                "terminal_reason": variant_summary.get(
                    "terminal_reason",
                    summary.get("terminal_reason"),
                ),
                "ok": (
                    variant_summary.get("ok")
                    if variant_summary
                    else summary.get("ok")
                    if variant_id == GIF_VARIANT_EVAL_GREEDY
                    else None
                ),
            }
        )
    return variants


def _summary_row(mount: Path, summary_path: Path) -> dict[str, Any] | None:
    summary_ref = _ref_from_path(mount, summary_path)
    try:
        _validate_selfplay_summary_ref(summary_ref.as_posix())
    except RefValidationError:
        return None
    ids = _extract_artifact_ids(summary_ref)
    if ids is None:
        return None

    summary, load_error = _read_json_object(summary_path)
    sibling_gif_ref = summary_ref.parent / "raw.gif"
    gif_ref_text = summary.get("gif_ref") if isinstance(summary.get("gif_ref"), str) else None
    try:
        gif_ref = _validate_selfplay_gif_ref(gif_ref_text or sibling_gif_ref.as_posix())
        if gif_ref.parent != summary_ref.parent or gif_ref.name != "raw.gif":
            gif_ref = sibling_gif_ref
    except RefValidationError:
        gif_ref = sibling_gif_ref

    gif_path = _path_for_ref(mount, gif_ref)
    stat = _safe_stat(summary_path)
    if stat is None:
        return None
    gif_stat = _safe_stat(gif_path)
    gif_exists = gif_stat is not None and _safe_is_file(gif_path)
    gif_variants = _gif_variant_rows(
        mount,
        summary=summary,
        summary_ref=summary_ref,
        legacy_gif_ref=gif_ref,
    )
    checkpoint_ref = summary.get("checkpoint_ref")
    checkpoint_label = summary.get("checkpoint_label") or (
        PurePosixPath(checkpoint_ref).name if isinstance(checkpoint_ref, str) else None
    )
    return {
        **ids,
        "ok": summary.get("ok") if load_error is None else False,
        "summary_ref": summary_ref.as_posix(),
        "gif_ref": gif_ref.as_posix(),
        "gif_exists": gif_exists,
        "gif_bytes": gif_stat.st_size if gif_exists and gif_stat is not None else None,
        "gif_updated_ts": gif_stat.st_mtime if gif_exists and gif_stat is not None else None,
        "gif_updated_ns": gif_stat.st_mtime_ns if gif_exists and gif_stat is not None else None,
        "gif_variants": gif_variants,
        "updated_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat().replace(
            "+00:00", "Z"
        ),
        "updated_ts": stat.st_mtime,
        "updated_ns": stat.st_mtime_ns,
        "frame_count": _safe_int(summary.get("frame_count")),
        "physical_steps": _safe_int(summary.get("physical_steps")),
        "scalar_steps": _safe_int(summary.get("scalar_steps")),
        "max_steps": _safe_int(summary.get("max_steps")),
        "terminal_reason": summary.get("terminal_reason"),
        "checkpoint_label": checkpoint_label,
        "checkpoint_ref": checkpoint_ref,
        "schema_id": summary.get("schema_id"),
        "load_error": load_error,
    }


def _list_selfplay_summaries(
    mount: Path,
    *,
    run_filter: str = "",
    run_id: str = "",
    attempt_filter: str = "",
    eval_filter: str = "",
    ok_filter: str = DEFAULT_OK_FILTER,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> list[dict[str, Any]]:
    page = _list_selfplay_summary_page(
        mount,
        run_filter=run_filter,
        run_id=run_id,
        attempt_filter=attempt_filter,
        eval_filter=eval_filter,
        ok_filter=ok_filter,
        limit=limit,
        offset=offset,
    )
    return page["rows"]


def _list_selfplay_summary_page(
    mount: Path,
    *,
    run_filter: str = "",
    run_id: str = "",
    attempt_filter: str = "",
    eval_filter: str = "",
    ok_filter: str = DEFAULT_OK_FILTER,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    base_path = _path_for_ref(mount, BASE_REF)
    limit = _coerce_limit(limit)
    offset = _coerce_offset(offset)
    if not _safe_is_dir(base_path):
        return {
            "rows": [],
            "total_rows": 0,
            "total_rows_exact": True,
            "offset": offset,
            "limit": limit,
            "has_newer": False,
            "has_older": False,
        }

    rows: list[dict[str, Any]] = []
    run_rank_by_id: dict[str, int] = {}
    if run_id:
        try:
            clean_run_id = run_mgmt.clean_id(run_id, label="run_id")
        except ValueError:
            return {
                "rows": [],
                "total_rows": 0,
                "total_rows_exact": True,
                "offset": offset,
                "limit": limit,
                "has_newer": False,
                "has_older": False,
            }
        if run_filter and not _matches_text(clean_run_id, run_filter):
            return {
                "rows": [],
                "total_rows": 0,
                "total_rows_exact": True,
                "offset": offset,
                "limit": limit,
                "has_newer": False,
                "has_older": False,
            }
        cache_key = (
            "summaries",
            mount.as_posix(),
            "run",
            clean_run_id,
            run_filter,
            attempt_filter,
            eval_filter,
            ok_filter.lower(),
            limit,
            offset,
        )
        cached = _cache_get(_SUMMARY_CACHE, cache_key)
        if cached is not None:
            return cached
        run_rank_by_id[clean_run_id] = 0
        summary_paths = sorted(
            _safe_glob(
                base_path / clean_run_id,
                "attempts/*/eval/*/selfplay/summary.json",
            ),
            key=_summary_path_sort_key,
        )
        page_rows: list[dict[str, Any]] = []
        matched_count = 0
        has_older = False
        for summary_path in summary_paths:
            summary_ref = _ref_from_path(mount, summary_path)
            ids = _extract_artifact_ids(summary_ref)
            if ids is not None:
                if not _matches_text(ids["attempt_id"], attempt_filter):
                    continue
                if not _matches_text(ids["eval_id"], eval_filter):
                    continue
            row = _summary_row(mount, summary_path)
            if row is None:
                continue
            if not _matches_ok(row, ok_filter):
                continue
            if matched_count >= offset:
                if len(page_rows) < limit:
                    page_rows.append(row)
                else:
                    has_older = True
                    matched_count += 1
                    break
            matched_count += 1
        total_rows = matched_count
        page = {
            "rows": page_rows,
            "total_rows": total_rows,
            "total_rows_exact": not has_older,
            "offset": offset,
            "limit": limit,
            "has_newer": offset > 0,
            "has_older": has_older,
        }
        return _cache_set(_SUMMARY_CACHE, cache_key, page)
    else:
        summary_paths = []
        listed_runs = _list_runs(mount)
        cache_key = (
            "summaries",
            mount.as_posix(),
            "all",
            tuple(
                str(run["run_id"])
                for run in listed_runs
            ),
            run_filter,
            attempt_filter,
            eval_filter,
            ok_filter.lower(),
        )
        cached = _cache_get(_SUMMARY_CACHE, cache_key)
        if cached is not None:
            total_rows = len(cached)
            return {
                "rows": cached[offset : offset + limit],
                "total_rows": total_rows,
                "total_rows_exact": True,
                "offset": offset,
                "limit": limit,
                "has_newer": offset > 0,
                "has_older": offset + limit < total_rows,
            }
        run_rank_by_id = {
            str(run["run_id"]): index for index, run in enumerate(listed_runs)
        }
        for run in listed_runs:
            summary_paths.extend(
                _safe_glob(
                    base_path / str(run["run_id"]),
                    "attempts/*/eval/*/selfplay/summary.json",
                )
            )

    for summary_path in summary_paths:
        row = _summary_row(mount, summary_path)
        if row is None:
            continue
        if run_id and row["run_id"] != clean_run_id:
            continue
        if not _matches_text(row["run_id"], run_filter):
            continue
        if not _matches_text(row["attempt_id"], attempt_filter):
            continue
        if not _matches_text(row["eval_id"], eval_filter):
            continue
        if not _matches_ok(row, ok_filter):
            continue
        rows.append(row)

    rows.sort(key=lambda item: _summary_sort_key(item, run_rank_by_id=run_rank_by_id))
    rows = _cache_set(_SUMMARY_CACHE, cache_key, rows)
    total_rows = len(rows)
    return {
        "rows": rows[offset : offset + limit],
        "total_rows": total_rows,
        "total_rows_exact": True,
        "offset": offset,
        "limit": limit,
        "has_newer": offset > 0,
        "has_older": offset + limit < total_rows,
    }


def _html_attr(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _link(path: str, ref: str, **params: Any) -> str:
    query = [f"ref={quote(ref, safe='')}"]
    for key, value in params.items():
        if value is None:
            continue
        query.append(f"{quote(str(key), safe='')}={quote(str(value), safe='')}")
    return f"{path}?{'&'.join(query)}"


def _next_url_without_selected_run(
    *,
    run_filter: str,
    attempt_filter: str,
    eval_filter: str,
    ok_filter: str,
    limit: int,
    offset: int = 0,
) -> str:
    params: dict[str, Any] = {"limit": int(limit)}
    if offset:
        params["offset"] = int(offset)
    if run_filter:
        params["run"] = run_filter
    if attempt_filter:
        params["attempt"] = attempt_filter
    if eval_filter:
        params["eval"] = eval_filter
    if ok_filter and ok_filter.lower() != DEFAULT_OK_FILTER:
        params["ok"] = ok_filter
    return f"/?{urlencode(params)}"


def _filter_url(
    *,
    run_id: str,
    run_filter: str,
    attempt_filter: str,
    eval_filter: str,
    ok_filter: str,
    limit: int,
    offset: int = 0,
) -> str:
    params: dict[str, Any] = {"limit": int(limit)}
    if offset:
        params["offset"] = int(offset)
    if run_id:
        params["run_id"] = run_id
    if run_filter:
        params["run"] = run_filter
    if attempt_filter:
        params["attempt"] = attempt_filter
    if eval_filter:
        params["eval"] = eval_filter
    if ok_filter and ok_filter.lower() != DEFAULT_OK_FILTER:
        params["ok"] = ok_filter
    return f"/?{urlencode(params)}"


def _fast_gif_url(row: dict[str, Any]) -> str:
    gif_version_ns = int(row.get("gif_updated_ns") or row.get("updated_ns") or 0)
    gif_version = f"{gif_version_ns}-{row.get('gif_bytes') or 0}"
    return _link("/gif", row["gif_ref"], v=gif_version)


def _variant_gif_url(variant: dict[str, Any], row: dict[str, Any]) -> str:
    gif_version_ns = int(variant.get("gif_updated_ns") or row.get("updated_ns") or 0)
    gif_version = f"{gif_version_ns}-{variant.get('gif_bytes') or 0}"
    return _link("/gif", variant["gif_ref"], v=gif_version)


def _head_token(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    row = rows[0]
    gif_version_ns = int(row.get("gif_updated_ns") or row.get("updated_ns") or 0)
    variant_parts = []
    for variant in row.get("gif_variants") or []:
        variant_parts.append(
            ",".join(
                [
                    str(variant.get("variant_id") or ""),
                    str(int(variant.get("gif_updated_ns") or 0)),
                    str(variant.get("gif_bytes") or 0),
                    "1" if variant.get("gif_exists") is True else "0",
                ]
            )
        )
    return ":".join(
        [
            str(row.get("run_id") or ""),
            str(row.get("attempt_id") or ""),
            str(row.get("eval_id") or ""),
            str(row.get("gif_bytes") or 0),
            str(gif_version_ns),
            "|".join(variant_parts),
        ]
    )


def _runs_token(runs: list[dict[str, Any]]) -> str:
    return "|".join(
        f"{run.get('run_id')}:{int(float(run.get('updated_ts') or 0))}:{run.get('artifact_count')}"
        for run in runs[:100]
    )


def _page_token(*, runs: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    return f"runs={_runs_token(runs)};head={_head_token(rows)}"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 1.0:
        return f"{value * 1000:.0f} ms"
    if value < 60.0:
        return f"{value:.2f} s"
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes}m {seconds:02d}s"


def _latest_training_progress(mount: Path, run_id: str) -> dict[str, Any] | None:
    if not run_id:
        return None
    try:
        clean_run_id = run_mgmt.clean_id(run_id, label="run_id")
    except ValueError:
        return None
    run_path = _path_for_ref(mount, BASE_REF / clean_run_id)
    best: tuple[float, dict[str, Any]] | None = None
    for progress_path in _safe_glob(run_path, "attempts/*/train/progress_latest.json"):
        stat = _safe_stat(progress_path)
        if stat is None or not _safe_is_file(progress_path):
            continue
        progress, load_error = _read_json_object(progress_path)
        if load_error is not None:
            continue
        iteration = _safe_int(progress.get("iteration"))
        elapsed_sec = _safe_float(progress.get("elapsed_sec"))
        iteration_count = (iteration + 1) if iteration is not None and iteration >= 0 else None
        sec_per_iteration = (
            elapsed_sec / iteration_count
            if elapsed_sec is not None and iteration_count
            else None
        )
        attempt_id = progress_path.parts[-3] if len(progress_path.parts) >= 3 else ""
        summary = {
            "attempt_id": attempt_id,
            "iteration": iteration,
            "elapsed_sec": elapsed_sec,
            "sec_per_iteration": sec_per_iteration,
            "timestamp": progress.get("timestamp"),
            "updated_at": datetime.fromtimestamp(stat.st_mtime, UTC)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        if best is None or stat.st_mtime > best[0]:
            best = (stat.st_mtime, summary)
    return None if best is None else best[1]


def _render_training_progress(progress: dict[str, Any] | None) -> str:
    if not progress:
        return '<div class="run-metrics"><span class="muted">speed unknown</span></div>'
    iteration = progress.get("iteration")
    sec_per_iteration = _format_seconds(_safe_float(progress.get("sec_per_iteration")))
    elapsed = _format_seconds(_safe_float(progress.get("elapsed_sec")))
    attempt = progress.get("attempt_id") or "unknown attempt"
    return (
        '<div class="run-metrics">'
        f'<span>iter {_html_attr(iteration)}</span>'
        f'<span>{_html_attr(sec_per_iteration)}/iter</span>'
        f'<span>elapsed {_html_attr(elapsed)}</span>'
        f'<span>{_html_attr(attempt)}</span>'
        '</div>'
    )


def _render_filters(
    *,
    runs: list[dict[str, Any]],
    selected_run_id: str,
    training_progress: dict[str, Any] | None,
    run_filter: str,
    attempt_filter: str,
    eval_filter: str,
    ok_filter: str,
    limit: int,
    offset: int,
) -> str:
    status_options = []
    for value, label in (("ok", "OK"), ("all", "All"), ("failed", "Failed")):
        selected = " selected" if ok_filter.lower() == value else ""
        status_options.append(f'<option value="{value}"{selected}>{label}</option>')

    limit_options = []
    for value in (5, 8, 12, 24, 50):
        selected = " selected" if int(limit) == value else ""
        limit_options.append(f'<option value="{value}"{selected}>{value}</option>')

    preserved_filters = []
    for name, value in (
        ("run", run_filter),
        ("attempt", attempt_filter),
        ("eval", eval_filter),
    ):
        if value:
            preserved_filters.append(
                f'<input type="hidden" name="{name}" form="filters-form" '
                f'value="{_html_attr(value)}">'
            )

    run_menu_rows: list[str] = []
    selected_label = "No runs"
    hidden_selected_run = ""
    if selected_run_id:
        selected_label = selected_run_id
        hidden_selected_run = (
            f'<input type="hidden" name="run_id" form="filters-form" '
            f'value="{_html_attr(selected_run_id)}">'
        )
    for run in runs:
        run_id = str(run["run_id"])
        select_url = _filter_url(
            run_id=run_id,
            run_filter=run_filter,
            attempt_filter=attempt_filter,
            eval_filter=eval_filter,
            ok_filter=ok_filter,
            limit=limit,
            offset=0,
        )
        next_run_id = "" if run_id == selected_run_id else selected_run_id
        next_url = _filter_url(
            run_id=next_run_id,
            run_filter=run_filter,
            attempt_filter=attempt_filter,
            eval_filter=eval_filter,
            ok_filter=ok_filter,
            limit=limit,
            offset=0,
        )
        delete_action = (
            f"/api/runs/{quote(run_id, safe='')}/hide?"
            f"next={quote(next_url, safe='')}"
        )
        selected_class = " selected" if run_id == selected_run_id else ""
        run_menu_rows.append(
            f"""
            <div class="run-menu-row{selected_class}">
                <a class="run-menu-link" href="{_html_attr(select_url)}">
                    <span class="run-menu-name">{_html_attr(run_id)}</span>
                    <span class="run-menu-time">{_html_attr(run["updated_at"])}</span>
                </a>
                <form method="post" action="{_html_attr(delete_action)}" class="hide-run-form">
                    <button type="submit" class="danger compact" data-delete-run-id="{_html_attr(run_id)}">
                        <span class="spinner" aria-hidden="true"></span>
                        <span class="button-label">Delete</span>
                    </button>
                </form>
            </div>
            """
        )
    refresh_url = _filter_url(
        run_id=selected_run_id,
        run_filter=run_filter,
        attempt_filter=attempt_filter,
        eval_filter=eval_filter,
        ok_filter=ok_filter,
        limit=limit,
        offset=0,
    )
    refresh_url = f"{refresh_url}&fresh=1"
    return f"""
        <form id="filters-form" method="get"></form>
        <div class="toolbar">
            {hidden_selected_run}
            {''.join(preserved_filters)}
            <input type="hidden" name="offset" form="filters-form" value="0">
            <div class="run-panel">
                <span class="field-label">Run</span>
                <details class="run-picker">
                    <summary>{_html_attr(selected_label)}</summary>
                    <div class="run-menu">{''.join(run_menu_rows)}</div>
                </details>
                {_render_training_progress(training_progress)}
            </div>
            <div class="toolbar-actions">
                <label>Status <select name="ok" form="filters-form">{''.join(status_options)}</select></label>
                <label>Show <select name="limit" form="filters-form">{''.join(limit_options)}</select></label>
                <button type="submit" form="filters-form">Apply</button>
                <a class="button secondary" href="{_html_attr(refresh_url)}">Refresh</a>
            </div>
        </div>
    """


def _render_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<section id="gallery" class="empty-state">No matching GIFs.</section>'

    rendered = []
    for index, row in enumerate(rows):
        primary_gif_url = _fast_gif_url(row)
        meta_url = _link("/meta", row["summary_ref"])
        checkpoint = row.get("checkpoint_label") or row.get("eval_id") or "checkpoint"
        loading = "eager" if index == 0 else "lazy"
        priority = "high" if index == 0 else "low"
        variant_previews = []
        for variant in row.get("gif_variants") or []:
            variant_label = variant.get("label") or variant.get("variant_id") or "GIF"
            if variant.get("gif_exists") is True:
                gif_url = _variant_gif_url(variant, row)
                frame = (
                    f'<a class="gif-frame" href="{_html_attr(gif_url)}">'
                    f'<img class="preview" loading="{loading}" decoding="async" '
                    f'fetchpriority="{priority}" src="{_html_attr(gif_url)}" '
                    f'width="224" height="224" alt=""></a>'
                )
            else:
                frame = (
                    '<a class="gif-frame missing-preview" href="'
                    + _html_attr(meta_url)
                    + '">missing</a>'
                )
            variant_previews.append(
                f'<div class="gif-variant"><div class="variant-label">'
                f'{_html_attr(variant_label)}</div>{frame}</div>'
            )
        if not variant_previews:
            variant_previews.append(
                f'<div class="gif-variant"><div class="variant-label">Greedy eval</div>'
                f'<a class="gif-frame" href="{_html_attr(primary_gif_url)}">'
                f'<img class="preview" loading="{loading}" decoding="async" '
                f'fetchpriority="{priority}" src="{_html_attr(primary_gif_url)}" '
                f'width="224" height="224" alt=""></a></div>'
                if row["gif_exists"]
                else '<div class="gif-variant"><div class="variant-label">Greedy eval</div>'
                '<a class="gif-frame missing-preview" href="' + _html_attr(meta_url) + '">missing</a></div>'
            )
        ok_label = (
            "OK"
            if row["ok"] is True and row["gif_exists"]
            else "missing"
            if not row["gif_exists"]
            else "failed"
            if row["ok"] is False
            else "unknown"
        )
        status_class = "ok" if row["ok"] is True and row["gif_exists"] else "failed"
        steps = row["physical_steps"]
        if row["max_steps"] is not None:
            steps = f"{steps or 0}/{row['max_steps']}"
        rendered.append(
            f"""
            <article class="gif-card" data-run-id="{_html_attr(row["run_id"])}">
                <div class="variant-grid">{''.join(variant_previews)}</div>
                <div class="gif-card-body">
                    <div class="gif-card-topline">
                        <strong>{_html_attr(checkpoint)}</strong>
                        <span class="status-pill {status_class}">{_html_attr(ok_label)}</span>
                    </div>
                    <div class="facts">
                        <span>{_html_attr(row["frame_count"])} frames</span>
                        <span>{_html_attr(steps)} steps</span>
                    </div>
                    <div class="reason">{_html_attr(row["terminal_reason"])}</div>
                    <div class="links">
                        <a href="{_html_attr(primary_gif_url)}">GIF</a>
                        <a href="{_html_attr(meta_url)}">JSON</a>
                    </div>
                </div>
            </article>
            """
        )
    return f'<section id="gallery" class="gallery">{"".join(rendered)}</section>'


def _render_pager(
    *,
    row_count: int,
    total_rows: int,
    total_rows_exact: bool,
    offset: int,
    limit: int,
    selected_run_id: str,
    run_filter: str,
    attempt_filter: str,
    eval_filter: str,
    ok_filter: str,
) -> str:
    if total_rows <= 0:
        return ""
    start = offset + 1
    end = min(offset + row_count, total_rows)
    total_label = f"{total_rows}" if total_rows_exact else f"{total_rows}+"
    links: list[str] = []
    if offset > 0:
        newer_url = _filter_url(
            run_id=selected_run_id,
            run_filter=run_filter,
            attempt_filter=attempt_filter,
            eval_filter=eval_filter,
            ok_filter=ok_filter,
            limit=limit,
            offset=max(0, offset - limit),
        )
        links.append(f'<a href="{_html_attr(newer_url)}">Newer</a>')
    if offset + limit < total_rows:
        older_url = _filter_url(
            run_id=selected_run_id,
            run_filter=run_filter,
            attempt_filter=attempt_filter,
            eval_filter=eval_filter,
            ok_filter=ok_filter,
            limit=limit,
            offset=offset + limit,
        )
        links.append(f'<a href="{_html_attr(older_url)}">Older</a>')
    links_html = " ".join(links)
    if links_html:
        links_html = f'<span class="pager-links">{links_html}</span>'
    return f'<nav class="pager"><span>{start}-{end} of {total_label}</span>{links_html}</nav>'


def _render_page(
    rows: list[dict[str, Any]],
    *,
    runs: list[dict[str, Any]],
    selected_run_id: str,
    run_filter: str,
    attempt_filter: str,
    eval_filter: str,
    ok_filter: str,
    limit: int,
    offset: int,
    total_rows: int,
    total_rows_exact: bool = True,
    page_token: str = "",
    server_elapsed_ms: int | None = None,
    reload_error: str | None = None,
) -> str:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    training_progress = _latest_training_progress(
        RUNS_MOUNT,
        selected_run_id,
    )
    filters = _render_filters(
        runs=runs,
        selected_run_id=selected_run_id,
        training_progress=training_progress,
        run_filter=run_filter,
        attempt_filter=attempt_filter,
        eval_filter=eval_filter,
        ok_filter=ok_filter,
        limit=limit,
        offset=offset,
    )
    pager = _render_pager(
        row_count=len(rows),
        total_rows=total_rows,
        total_rows_exact=total_rows_exact,
        offset=offset,
        limit=limit,
        selected_run_id=selected_run_id,
        run_filter=run_filter,
        attempt_filter=attempt_filter,
        eval_filter=eval_filter,
        ok_filter=ok_filter,
    )
    reload_warning = (
        f'<p class="warning">Volume refresh failed; showing last visible data. '
        f'{_html_attr(reload_error)}</p>'
        if reload_error
        else ""
    )
    load_label = (
        f"server {server_elapsed_ms} ms"
        if server_elapsed_ms is not None
        else "server timing unavailable"
    )
    current_page_token = page_token or _page_token(runs=runs, rows=rows)
    shown_label = (
        "No GIFs"
        if total_rows <= 0
        else f"{len(rows)} shown / {total_rows}{'' if total_rows_exact else '+'} total"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CurvyTron Self-Play GIFs</title>
    <style>
        body {{
            margin: 0;
            color: #202124;
            background: #f6f7f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .shell {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 18px;
        }}
        .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 14px;
        }}
        h1 {{ margin: 0; font-size: 22px; line-height: 1.15; }}
        .subtitle {{ margin: 4px 0 0; color: #5f6368; font-size: 13px; }}
        .top-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            color: #5f6368;
            font-size: 13px;
        }}
        .auto-refresh {{
            display: inline-flex;
            gap: 6px;
            align-items: center;
            white-space: nowrap;
        }}
        .auto-refresh input {{ height: auto; padding: 0; }}
        .toolbar {{
            display: flex;
            flex-wrap: wrap;
            align-items: stretch;
            justify-content: space-between;
            gap: 12px;
            padding: 12px;
            background: #ffffff;
            border: 1px solid #dadce0;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(60, 64, 67, 0.08);
        }}
        label {{ display: grid; gap: 4px; color: #5f6368; font-size: 12px; }}
        .run-panel {{
            display: grid;
            gap: 6px;
            color: #5f6368;
            font-size: 12px;
            min-width: min(620px, 100%);
            flex: 1 1 520px;
        }}
        .field-label {{ display: block; }}
        .run-metrics {{ display: flex; flex-wrap: wrap; gap: 6px; }}
        .run-metrics span, .metric {{
            display: inline-flex;
            align-items: center;
            min-height: 24px;
            padding: 0 8px;
            border-radius: 999px;
            background: #eef3f8;
            color: #5f6368;
            font-size: 12px;
        }}
        .muted {{ color: #80868b; }}
        .run-picker {{ position: relative; color: #202124; }}
        .run-picker summary {{
            display: flex;
            align-items: center;
            min-height: 36px;
            border: 1px solid #dadce0;
            border-radius: 6px;
            padding: 0 10px;
            cursor: pointer;
            user-select: none;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .run-picker[open] summary {{ border-color: #1a73e8; }}
        .run-menu {{
            position: absolute;
            z-index: 20;
            top: calc(100% + 4px);
            left: 0;
            width: min(720px, 90vw);
            max-height: 420px;
            overflow: auto;
            border: 1px solid #dadce0;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(60, 64, 67, 0.18);
        }}
        .run-menu-row {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: center;
            gap: 8px;
            border-top: 1px solid #e8eaed;
        }}
        .run-menu-row.selected {{ background: #f1f5ff; }}
        .run-menu-link {{
            display: grid;
            gap: 2px;
            margin: 0;
            padding: 8px;
            overflow: hidden;
            color: inherit;
            text-decoration: none;
        }}
        .run-menu-name {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .run-menu-time {{ color: #80868b; font-size: 11px; }}
        .hide-run-form {{ margin: 0; }}
        .hide-run-form button {{ margin-right: 6px; }}
        .hide-run-form.is-deleting button {{
            opacity: 0.78;
            cursor: wait;
        }}
        body.action-busy .run-menu-link,
        body.action-busy .toolbar-actions button,
        body.action-busy .toolbar-actions .button {{
            opacity: 0.55;
            pointer-events: none;
        }}
        button:disabled {{
            cursor: wait;
            opacity: 0.65;
        }}
        .spinner {{
            display: none;
            width: 12px;
            height: 12px;
            margin-right: 6px;
            border: 2px solid rgba(255, 255, 255, 0.45);
            border-top-color: #ffffff;
            border-radius: 50%;
            vertical-align: -2px;
            animation: spin 0.8s linear infinite;
        }}
        .hide-run-form.is-deleting .spinner {{ display: inline-block; }}
        @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
        input, select, button, .button {{
            box-sizing: border-box;
            height: 36px;
            border: 1px solid #dadce0;
            border-radius: 6px;
            padding: 0 8px;
            font: inherit;
        }}
        button, .button {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #1a73e8;
            border-color: #1a73e8;
            color: white;
            text-decoration: none;
        }}
        .button.secondary {{ background: #ffffff; color: #1a73e8; }}
        button.danger {{ background: #b3261e; border-color: #b3261e; }}
        button.compact {{ height: 28px; padding: 0 8px; font-size: 12px; }}
        .toolbar-actions {{
            display: flex;
            flex-wrap: wrap;
            align-items: end;
            gap: 8px;
        }}
        code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
        a {{ color: #1a73e8; }}
        .page-summary {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 10px;
            align-items: center;
            margin: 14px 0 10px;
            color: #5f6368;
            font-size: 13px;
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(min(460px, 100%), 1fr));
            gap: 12px;
        }}
        .gif-card {{
            overflow: hidden;
            background: #ffffff;
            border: 1px solid #dadce0;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(60, 64, 67, 0.08);
        }}
        .variant-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            padding: 8px;
            background: #f8fafd;
        }}
        .gif-variant {{ min-width: 0; }}
        .variant-label {{
            margin: 0 0 5px;
            color: #5f6368;
            font-size: 12px;
            font-weight: 650;
            line-height: 1.2;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .gif-frame {{
            display: grid;
            place-items: center;
            aspect-ratio: 1;
            background: #111827;
            text-decoration: none;
        }}
        .preview {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        .missing-preview {{ color: #f8fafd; font-size: 13px; }}
        .gif-card-body {{
            display: grid;
            gap: 8px;
            padding: 10px;
            font-size: 13px;
        }}
        .gif-card-topline {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }}
        .gif-card-topline strong {{
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .status-pill {{
            flex: 0 0 auto;
            padding: 2px 7px;
            border-radius: 999px;
            background: #edf0f3;
            color: #5f6368;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .status-pill.ok {{ background: #e6f4ea; color: #137333; }}
        .status-pill.failed {{ background: #fce8e6; color: #b3261e; }}
        .facts {{ display: flex; flex-wrap: wrap; gap: 6px; color: #5f6368; font-size: 12px; }}
        .facts span {{ padding: 2px 6px; border-radius: 999px; background: #f1f3f4; }}
        .reason {{
            min-height: 18px;
            color: #3c4043;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .links {{ display: flex; gap: 10px; font-size: 12px; }}
        .pager {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 10px;
            align-items: center;
            margin: 0 0 12px;
            color: #5f6368;
            font-size: 13px;
        }}
        .pager-links {{ display: flex; gap: 10px; }}
        .warning {{ color: #b06000; }}
        .empty-state {{
            display: grid;
            min-height: 220px;
            place-items: center;
            color: #80868b;
            background: #ffffff;
            border: 1px dashed #dadce0;
            border-radius: 8px;
        }}
        @media (max-width: 900px) {{
            .shell {{ padding: 12px; }}
            .topbar {{ display: block; }}
            .top-meta {{ margin-top: 10px; }}
            .toolbar-actions {{ width: 100%; }}
            .toolbar-actions label, .toolbar-actions button, .toolbar-actions .button {{ flex: 1 1 120px; }}
        }}
        @media (max-width: 560px) {{
            .variant-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <main class="shell">
        <header class="topbar">
            <div>
                <h1>CurvyTron GIFs</h1>
                <p class="subtitle">Latest checkpoint self-play clips. Visual check only.</p>
            </div>
            <div class="top-meta">
                <span id="load-time">{_html_attr(load_label)}</span>
                <span id="refresh-state">updated {_html_attr(generated_at)}</span>
                <label class="auto-refresh">
                    <input id="auto-refresh" type="checkbox" checked>
                    Live 15s
                </label>
            </div>
        </header>
        {filters}
        {reload_warning}
        <div class="page-summary">
            <span id="match-count">{_html_attr(shown_label)}</span>
            <span class="muted">Run: <span id="selected-run-label">{_html_attr(selected_run_id or "none")}</span></span>
        </div>
        {pager}
        {_render_rows(rows)}
    </main>
    <script>
        const refreshState = document.getElementById("refresh-state");
        const matchCount = document.getElementById("match-count");
        let actionBusy = false;
        const setRefreshState = (text) => {{
            if (refreshState) refreshState.textContent = text;
        }};
        const setRunActionBusy = (busy) => {{
            actionBusy = busy;
            document.body.classList.toggle("action-busy", busy);
            for (const button of document.querySelectorAll(".hide-run-form button")) {{
                button.disabled = busy;
            }}
        }};
        const enterNavigationState = (message) => {{
            if (actionBusy) return false;
            setRunActionBusy(true);
            setRefreshState(message);
            return true;
        }};
        const filtersForm = document.getElementById("filters-form");
        if (filtersForm) {{
            filtersForm.addEventListener("submit", (event) => {{
                if (!enterNavigationState("loading...")) event.preventDefault();
            }});
        }}
        for (const link of document.querySelectorAll(".run-menu-link")) {{
            link.addEventListener("click", (event) => {{
                if (!enterNavigationState("loading run...")) event.preventDefault();
            }});
        }}
        for (const link of document.querySelectorAll(".toolbar-actions .button")) {{
            link.addEventListener("click", (event) => {{
                if (!enterNavigationState("refreshing...")) event.preventDefault();
            }});
        }}
        for (const form of document.querySelectorAll(".hide-run-form")) {{
            form.addEventListener("submit", async (event) => {{
                event.preventDefault();
                if (actionBusy || form.classList.contains("is-deleting")) return;
                form.classList.add("is-deleting");
                const button = form.querySelector("button");
                const label = form.querySelector(".button-label");
                const runName = button?.dataset.deleteRunId || "run";
                setRunActionBusy(true);
                if (label) label.textContent = "Deleting";
                setRefreshState(`deleting ${{runName}}...`);
                try {{
                    const response = await fetch(form.action, {{
                        method: "POST",
                        headers: {{
                            "Accept": "application/json",
                            "X-Requested-With": "fetch"
                        }}
                    }});
                    if (!response.ok) throw new Error(`delete failed: ${{response.status}}`);
                    const payload = await response.json();
                    if (!payload.ok) throw new Error("delete failed");
                    const runId = payload.run_id || runName || "";
                    const row = form.closest(".run-menu-row");
                    const picker = form.closest(".run-picker");
                    const selectedInput = document.querySelector(
                        'input[name="run_id"][form="filters-form"]'
                    );
                    const deletedSelectedRun = selectedInput && selectedInput.value === runId;
                    if (row) row.remove();
                    for (const resultRow of document.querySelectorAll(".gif-card[data-run-id]")) {{
                        if (resultRow.dataset.runId === runId) resultRow.remove();
                    }}
                    if (deletedSelectedRun) {{
                        selectedInput.remove();
                        const summary = picker?.querySelector("summary");
                        if (summary) summary.textContent = "Pick a run";
                        renderRows([]);
                        if (matchCount) matchCount.textContent = "No matching GIFs";
                        const selectedRunLabel = document.getElementById("selected-run-label");
                        if (selectedRunLabel) selectedRunLabel.textContent = "none";
                        currentPageToken = "";
                        headUrl.searchParams.delete("run_id");
                        summariesUrl.searchParams.delete("run_id");
                        if (autoRefresh) {{
                            autoRefresh.checked = false;
                            autoRefresh.disabled = true;
                        }}
                        const url = new URL(window.location.href);
                        url.searchParams.delete("run_id");
                        window.history.replaceState(null, "", url.pathname + url.search + url.hash);
                    }}
                    setRefreshState(`deleted ${{runId}}`);
                }} catch (error) {{
                    form.classList.remove("is-deleting");
                    if (label) label.textContent = "Retry";
                    setRefreshState("delete failed");
                }} finally {{
                    setRunActionBusy(false);
                }}
            }});
        }}
        const escapeHtml = (value) => {{
            const element = document.createElement("span");
            element.textContent = value ?? "";
            return element.innerHTML;
        }};
        const artifactUrl = (path, ref, version = "") => {{
            const url = new URL(path, window.location.href);
            url.searchParams.set("ref", ref || "");
            if (version) url.searchParams.set("v", version);
            return url.pathname + url.search;
        }};
        const rowGifVersion = (row) => {{
            const version = Math.floor(Number(row.gif_updated_ns || row.updated_ns || 0));
            return `${{version}}-${{row.gif_bytes || 0}}`;
        }};
        const variantGifVersion = (variant, row) => {{
            const version = Math.floor(Number(variant.gif_updated_ns || row.updated_ns || 0));
            return `${{version}}-${{variant.gif_bytes || 0}}`;
        }};
        const rowCard = (row, index) => {{
            const primaryGifUrl = artifactUrl("/gif", row.gif_ref, rowGifVersion(row));
            const metaUrl = artifactUrl("/meta", row.summary_ref);
            const checkpoint = row.checkpoint_label || row.eval_id || "checkpoint";
            const ok = row.ok === true && row.gif_exists === true;
            const status = row.ok === true && row.gif_exists === true
                ? "OK"
                : row.gif_exists !== true
                    ? "missing"
                    : row.ok === false
                        ? "failed"
                        : "unknown";
            const statusClass = ok ? "ok" : "failed";
            const steps = row.max_steps != null
                ? `${{row.physical_steps || 0}}/${{row.max_steps}}`
                : String(row.physical_steps ?? "unknown");
            const loading = index === 0 ? "eager" : "lazy";
            const priority = index === 0 ? "high" : "low";
            const variants = Array.isArray(row.gif_variants) && row.gif_variants.length
                ? row.gif_variants
                : [{{
                    variant_id: "eval_greedy",
                    label: "Greedy eval",
                    gif_ref: row.gif_ref,
                    gif_exists: row.gif_exists,
                    gif_bytes: row.gif_bytes,
                    gif_updated_ns: row.gif_updated_ns
                }}];
            const variantGrid = variants.map((variant) => {{
                const label = variant.label || variant.variant_id || "GIF";
                if (variant.gif_exists === true && variant.gif_ref) {{
                    const gifUrl = artifactUrl("/gif", variant.gif_ref, variantGifVersion(variant, row));
                    return `<div class="gif-variant"><div class="variant-label">${{escapeHtml(label)}}</div><a class="gif-frame" href="${{escapeHtml(gifUrl)}}"><img class="preview" loading="${{loading}}" decoding="async" fetchpriority="${{priority}}" src="${{escapeHtml(gifUrl)}}" width="224" height="224" alt=""></a></div>`;
                }}
                return `<div class="gif-variant"><div class="variant-label">${{escapeHtml(label)}}</div><a class="gif-frame missing-preview" href="${{escapeHtml(metaUrl)}}">missing</a></div>`;
            }}).join("");
            return `
                <article class="gif-card" data-run-id="${{escapeHtml(row.run_id)}}">
                    <div class="variant-grid">${{variantGrid}}</div>
                    <div class="gif-card-body">
                        <div class="gif-card-topline">
                            <strong>${{escapeHtml(checkpoint)}}</strong>
                            <span class="status-pill ${{statusClass}}">${{escapeHtml(status)}}</span>
                        </div>
                        <div class="facts">
                            <span>${{escapeHtml(row.frame_count)}} frames</span>
                            <span>${{escapeHtml(steps)}} steps</span>
                        </div>
                        <div class="reason">${{escapeHtml(row.terminal_reason || "")}}</div>
                        <div class="links">
                            <a href="${{primaryGifUrl}}">GIF</a>
                            <a href="${{metaUrl}}">JSON</a>
                        </div>
                    </div>
                </article>`;
        }};
        const renderRows = (rows) => {{
            const gallery = document.getElementById("gallery");
            if (!gallery) return;
            if (!rows || rows.length === 0) {{
                gallery.className = "empty-state";
                gallery.innerHTML = "No matching GIFs.";
                return;
            }}
            gallery.className = "gallery";
            gallery.innerHTML = rows.map(rowCard).join("");
        }};
        const loadTime = document.getElementById("load-time");
        if (loadTime && performance?.timing) {{
            const elapsed = Date.now() - performance.timing.navigationStart;
            if (Number.isFinite(elapsed) && elapsed >= 0) {{
                loadTime.textContent += `, browser ${{elapsed}} ms`;
            }}
        }}
        const autoRefresh = document.getElementById("auto-refresh");
        let currentPageToken = {json.dumps(current_page_token)};
        const selectedRunId = {json.dumps(selected_run_id)};
        const autoRefreshAllowed = {json.dumps(offset == 0)};
        const headUrl = new URL("/api/head", window.location.href);
        const summariesUrl = new URL("/api/summaries", window.location.href);
        const currentParams = new URLSearchParams(window.location.search);
        if (!autoRefreshAllowed && autoRefresh) {{
            autoRefresh.checked = false;
            autoRefresh.disabled = true;
        }}
        if (!currentParams.get("run_id") && selectedRunId) {{
            headUrl.searchParams.set("run_id", selectedRunId);
            summariesUrl.searchParams.set("run_id", selectedRunId);
        }}
        for (const key of ["run_id", "run", "attempt", "eval", "ok", "limit"]) {{
            const value = currentParams.get(key);
            if (value) {{
                headUrl.searchParams.set(key, value);
                summariesUrl.searchParams.set(key, value);
            }}
        }}
        headUrl.searchParams.set("fresh", "1");
        summariesUrl.searchParams.set("offset", "0");
        const checkForNewGif = async () => {{
            if (actionBusy) return;
            if (!autoRefreshAllowed || !autoRefresh?.checked) return;
            try {{
                const response = await fetch(headUrl.toString(), {{
                    cache: "no-store",
                    headers: {{ "Accept": "application/json" }}
                }});
                if (!response.ok) return;
                const payload = await response.json();
                const nextToken = payload.page_token || "";
                if (nextToken && nextToken !== currentPageToken) {{
                    const summaries = await fetch(summariesUrl.toString(), {{
                        cache: "no-store",
                        headers: {{ "Accept": "application/json" }}
                    }});
                    if (!summaries.ok) return;
                    const summaryPayload = await summaries.json();
                    renderRows(summaryPayload.rows || []);
                    if (matchCount) {{
                        const total = summaryPayload.total_rows ?? 0;
                        const suffix = summaryPayload.total_rows_exact === false ? "+" : "";
                        matchCount.textContent = `${{(summaryPayload.rows || []).length}} shown / ${{total}}${{suffix}} total`;
                    }}
                    currentPageToken = nextToken;
                    if (refreshState) {{
                        refreshState.textContent = `updated ${{new Date().toLocaleTimeString()}}`;
                    }}
                }}
            }} catch (error) {{
                // Keep the page quiet if a transient poll fails.
            }}
        }};
        window.setInterval(checkForNewGif, 15000);
    </script>
</body>
</html>
"""


def _build_fastapi_app(volume: Any) -> Any:
    from fastapi import FastAPI, Header, Query
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

    web_app = FastAPI(title="CurvyTron Self-Play GIF Browser")

    @web_app.get("/", response_class=HTMLResponse)
    def index(
        run_id: str = "",
        run: str = "",
        attempt: str = "",
        eval: str = "",
        ok: str = DEFAULT_OK_FILTER,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        fresh: bool = False,
    ) -> HTMLResponse:
        started_at = monotonic()
        reload_error = None
        if fresh:
            reload_error = _maybe_reload_volume(volume, force=True)
        runs = _list_runs(RUNS_MOUNT)
        selected_run_id = _default_selected_run_id(runs, run_id)
        page = _list_selfplay_summary_page(
            RUNS_MOUNT,
            run_id=selected_run_id,
            run_filter=run,
            attempt_filter=attempt,
            eval_filter=eval,
            ok_filter=ok,
            limit=limit,
            offset=offset,
        )
        head_page = (
            page
            if page["offset"] == 0
            else _list_selfplay_summary_page(
                RUNS_MOUNT,
                run_id=selected_run_id,
                run_filter=run,
                attempt_filter=attempt,
                eval_filter=eval,
                ok_filter=ok,
                limit=1,
                offset=0,
            )
        )
        return HTMLResponse(
            _render_page(
                page["rows"],
                runs=runs,
                selected_run_id=selected_run_id,
                run_filter=run,
                attempt_filter=attempt,
                eval_filter=eval,
                ok_filter=ok,
                limit=page["limit"],
                offset=page["offset"],
                total_rows=page["total_rows"],
                total_rows_exact=bool(page.get("total_rows_exact", True)),
                page_token=_head_token(head_page["rows"]),
                server_elapsed_ms=int((monotonic() - started_at) * 1000),
                reload_error=reload_error,
            ),
            headers=DYNAMIC_RESPONSE_HEADERS,
        )

    @web_app.get("/api/summaries")
    def summaries(
        run_id: str = "",
        run: str = "",
        attempt: str = "",
        eval: str = "",
        ok: str = DEFAULT_OK_FILTER,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
        fresh: bool = False,
    ) -> JSONResponse:
        reload_error = None
        if fresh:
            reload_error = _maybe_reload_volume(volume, force=True)
        runs = _list_runs(RUNS_MOUNT)
        selected_run_id = _default_selected_run_id(runs, run_id)
        page = _list_selfplay_summary_page(
            RUNS_MOUNT,
            run_id=selected_run_id,
            run_filter=run,
            attempt_filter=attempt,
            eval_filter=eval,
            ok_filter=ok,
            limit=limit,
            offset=offset,
        )
        return JSONResponse(
            {
                "rows": page["rows"],
                "runs": runs,
                "selected_run_id": selected_run_id,
                "total_rows": page["total_rows"],
                "total_rows_exact": page.get("total_rows_exact", True),
                "offset": page["offset"],
                "limit": page["limit"],
                "has_newer": page["has_newer"],
                "has_older": page["has_older"],
                "head_token": _head_token(page["rows"]),
                "reload_error": reload_error,
            },
            headers=DYNAMIC_RESPONSE_HEADERS,
        )

    @web_app.get("/api/head")
    def head(
        run_id: str = "",
        run: str = "",
        attempt: str = "",
        eval: str = "",
        ok: str = DEFAULT_OK_FILTER,
        fresh: bool = False,
    ) -> JSONResponse:
        reload_error = None
        if fresh:
            reload_error = _maybe_reload_volume(volume)
        runs: list[dict[str, Any]] = []
        if run_id:
            try:
                selected_run_id = run_mgmt.clean_id(run_id, label="run_id")
            except ValueError:
                selected_run_id = ""
        else:
            runs = _list_runs(RUNS_MOUNT)
            selected_run_id = _default_selected_run_id(runs, run_id)
        page = _list_selfplay_summary_page(
            RUNS_MOUNT,
            run_id=selected_run_id,
            run_filter=run,
            attempt_filter=attempt,
            eval_filter=eval,
            ok_filter=ok,
            limit=1,
            offset=0,
        )
        rows = page["rows"]
        head_token = _head_token(rows)
        return JSONResponse(
            {
                "page_token": head_token,
                "head_token": head_token,
                "head": rows[0] if rows else None,
                "runs": runs,
                "selected_run_id": selected_run_id,
                "total_rows": page["total_rows"],
                "total_rows_exact": page.get("total_rows_exact", True),
                "reload_error": reload_error,
            },
            headers=DYNAMIC_RESPONSE_HEADERS,
        )

    @web_app.post("/api/runs/{run_id}/hide")
    def hide_run(
        run_id: str,
        next: str = "/",
        x_requested_with: str = Header(default=""),
    ) -> Response:
        next_url = _safe_next_url(next)
        try:
            clean_run_id = run_mgmt.clean_id(run_id, label="run_id")
        except ValueError as exc:
            return Response(str(exc), status_code=400)

        if (
            modal is not None
            and runs_volume is not None
            and volume is runs_volume
            and "curvytron_gif_browser_hide_run" in globals()
        ):
            delete_result = curvytron_gif_browser_hide_run.remote(run_id=clean_run_id)
            _clear_listing_caches()
            local_reload_error = _maybe_reload_volume(volume, force=True)
            if local_reload_error and not delete_result.get("reload_error"):
                delete_result["reload_error"] = local_reload_error
        else:
            delete_result = _hide_run_from_picker_on_mount(
                mount=RUNS_MOUNT,
                run_id=clean_run_id,
                volume=volume,
            )
        if x_requested_with == "fetch":
            return JSONResponse(
                {**delete_result, "next": next_url},
                headers=DYNAMIC_RESPONSE_HEADERS,
            )
        return RedirectResponse(next_url, status_code=303)

    @web_app.get("/gif")
    def gif(ref: str, if_none_match: str = Header(default="")) -> Response:
        try:
            safe_ref = _validate_selfplay_gif_ref(ref)
        except RefValidationError as exc:
            return Response(str(exc), status_code=400)
        path = _path_for_ref(RUNS_MOUNT, safe_ref)
        gif_stat = _safe_stat(path)
        if gif_stat is None or not _safe_is_file(path):
            return Response("GIF not found", status_code=404)
        etag = _gif_etag(path, gif_stat)
        headers = {
            "Cache-Control": (
                f"public, max-age={GIF_CACHE_MAX_AGE_SECONDS}, immutable"
            ),
            "ETag": etag,
            "Content-Length": str(gif_stat.st_size),
        }
        if _etag_matches(if_none_match, etag):
            return Response(status_code=304, headers=headers)
        try:
            with _VOLUME_IO_LOCK:
                gif_bytes = path.read_bytes()
        except OSError:
            return Response("GIF not found", status_code=404)
        return Response(
            gif_bytes,
            media_type="image/gif",
            headers=headers,
        )

    @web_app.get("/meta")
    def meta(ref: str) -> Response:
        try:
            safe_ref = _validate_selfplay_summary_ref(ref)
        except RefValidationError as exc:
            return Response(str(exc), status_code=400)
        path = _path_for_ref(RUNS_MOUNT, safe_ref)
        if not path.is_file():
            return Response("JSON not found", status_code=404)
        try:
            with _VOLUME_IO_LOCK:
                payload = path.read_bytes()
        except OSError:
            return Response("JSON not found", status_code=404)
        return Response(
            payload,
            media_type="application/json",
            headers={"Cache-Control": "no-cache"},
        )

    return web_app


if modal is not None:

    @app.function(
        image=image,
        volumes={RUNS_MOUNT.as_posix(): runs_volume},
        timeout=60,
        cpu=0.25,
        memory=256,
    )
    def curvytron_gif_browser_hide_run(run_id: str) -> dict[str, Any]:
        reload_error = None
        if hasattr(runs_volume, "reload"):
            try:
                runs_volume.reload()
            except Exception as exc:  # pragma: no cover - remote Volume resilience.
                reload_error = f"{type(exc).__name__}: {exc}"
        result = _hide_run_from_picker_on_mount(
            mount=RUNS_MOUNT,
            run_id=run_id,
            volume=runs_volume,
        )
        result["reload_error"] = reload_error
        print(json.dumps(result, sort_keys=True))
        return result


    @app.function(
        image=image,
        volumes={RUNS_MOUNT.as_posix(): runs_volume},
        timeout=300,
        cpu=4,
        memory=4096,
        max_containers=2,
    )
    @modal.concurrent(max_inputs=50)
    @modal.asgi_app()
    def gif_browser():
        return _build_fastapi_app(runs_volume)

else:  # pragma: no cover - useful only when importing without the modal extra.

    def gif_browser():
        return _build_fastapi_app(None)
