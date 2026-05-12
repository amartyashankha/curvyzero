"""Tiny Modal ASGI browser for CurvyTron checkpoint self-play GIFs.

Usage:

    uv run --extra modal modal serve -m curvyzero.infra.modal.curvytron_gif_browser
    uv run --extra modal modal deploy -m curvyzero.infra.modal.curvytron_gif_browser
"""

from __future__ import annotations

import json
import re
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
DEFAULT_LIMIT = 50
MAX_LIMIT = 500
DEFAULT_OK_FILTER = "ok"
LISTING_CACHE_TTL_SECONDS = 2.0
VOLUME_RELOAD_TTL_SECONDS = 1.0
GIF_CACHE_MAX_AGE_SECONDS = 86_400
RUN_PICKER_FLAG_FILENAME = run_mgmt.GIF_BROWSER_RUN_MARKER_FILENAME
CHECKPOINT_ITERATION_RE = re.compile(r"iteration[_-](\d+)")
RUN_RECENCY_PATTERNS = (
    "attempts/*/eval/*/selfplay/summary.json",
    "attempts/*/eval/*/selfplay/raw.gif",
    "attempts/*/train/summary.json",
    "attempts/*/train/status.json",
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


_RUNS_CACHE: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}
_SUMMARY_CACHE: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]] = {}


def _clone_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _cache_get(
    cache: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]],
    key: tuple[Any, ...],
) -> list[dict[str, Any]] | None:
    if LISTING_CACHE_TTL_SECONDS <= 0:
        return None
    cached = cache.get(key)
    if cached is None:
        return None
    expires_at, rows = cached
    if monotonic() >= expires_at:
        cache.pop(key, None)
        return None
    return _clone_rows(rows)


def _cache_set(
    cache: dict[tuple[Any, ...], tuple[float, list[dict[str, Any]]]],
    key: tuple[Any, ...],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if LISTING_CACHE_TTL_SECONDS > 0:
        cache[key] = (monotonic() + LISTING_CACHE_TTL_SECONDS, _clone_rows(rows))
    return rows


def _clear_listing_caches() -> None:
    _RUNS_CACHE.clear()
    _SUMMARY_CACHE.clear()


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
    if _extract_selfplay_artifact_ids(ref, filename="raw.gif") is None:
        raise RefValidationError("ref must be a selfplay raw.gif artifact")
    return ref


def _read_json_object(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
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
    run_tokens: list[tuple[str, tuple[int, int] | None, tuple[int, int] | None]] = []
    for run_path in _safe_iterdir(base_path):
        if not _safe_is_dir(run_path):
            continue
        marker_token = _stat_token(run_path / RUN_PICKER_FLAG_FILENAME)
        if marker_token is None:
            continue
        run_tokens.append((run_path.name, marker_token, _stat_token(run_path)))
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
        _stat_token(run_path),
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

    cache_key = ("runs", _runs_cache_state_token(mount))
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


def _default_selected_run_id(runs: list[dict[str, Any]], requested_run_id: str) -> str:
    if requested_run_id:
        try:
            return run_mgmt.clean_id(requested_run_id, label="run_id")
        except ValueError:
            return ""
    if not runs:
        return ""
    return str(runs[0]["run_id"])


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
        if gif_ref.parent != summary_ref.parent:
            gif_ref = sibling_gif_ref
    except RefValidationError:
        gif_ref = sibling_gif_ref

    gif_path = _path_for_ref(mount, gif_ref)
    stat = _safe_stat(summary_path)
    if stat is None:
        return None
    gif_stat = _safe_stat(gif_path)
    gif_exists = gif_stat is not None and _safe_is_file(gif_path)
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
        "updated_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat().replace(
            "+00:00", "Z"
        ),
        "updated_ts": stat.st_mtime,
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
            _run_summary_cache_state_token(mount, clean_run_id),
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
                "offset": offset,
                "limit": limit,
                "has_newer": offset > 0,
                "has_older": offset + limit < total_rows,
            }
        run_rank_by_id[clean_run_id] = 0
        summary_paths = _safe_glob(
            base_path / clean_run_id,
            "attempts/*/eval/*/selfplay/summary.json",
        )
    else:
        summary_paths = []
        listed_runs = _list_runs(mount)
        cache_key = (
            "summaries",
            mount.as_posix(),
            "all",
            tuple(
                _run_summary_cache_state_token(mount, str(run["run_id"]))
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
    gif_version_ts = int(row.get("gif_updated_ts") or row["updated_ts"])
    gif_version = f"{gif_version_ts}-{row.get('gif_bytes') or 0}"
    return _link("/gif", row["gif_ref"], v=gif_version)


def _render_filters(
    *,
    runs: list[dict[str, Any]],
    selected_run_id: str,
    run_filter: str,
    attempt_filter: str,
    eval_filter: str,
    ok_filter: str,
    limit: int,
    offset: int,
) -> str:
    status_options = []
    for value, label in (("all", "All"), ("ok", "OK"), ("failed", "Failed")):
        selected = " selected" if ok_filter.lower() == value else ""
        status_options.append(f'<option value="{value}"{selected}>{label}</option>')

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
        run_label = f"{run_id} ({run['updated_at']})"
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
                <a class="run-menu-link" href="{_html_attr(select_url)}">{_html_attr(run_label)}</a>
                <form method="post" action="{_html_attr(delete_action)}" class="hide-run-form">
                    <button type="submit" class="danger" data-delete-run-id="{_html_attr(run_id)}">
                        <span class="spinner" aria-hidden="true"></span>
                        <span class="button-label">Delete</span>
                    </button>
                </form>
            </div>
            """
        )
    return f"""
        <form id="filters-form" method="get"></form>
        <div class="filters">
            {hidden_selected_run}
            <div class="run-control">
                <span class="field-label">Run</span>
                <details class="run-picker">
                    <summary>{_html_attr(selected_label)}</summary>
                    <div class="run-menu">{''.join(run_menu_rows)}</div>
                </details>
            </div>
            <label>Run text <input name="run" form="filters-form" value="{_html_attr(run_filter)}"></label>
            <label>Attempt <input name="attempt" form="filters-form" value="{_html_attr(attempt_filter)}"></label>
            <label>Eval <input name="eval" form="filters-form" value="{_html_attr(eval_filter)}"></label>
            <label>Status <select name="ok" form="filters-form">{''.join(status_options)}</select></label>
            <label>Limit <input name="limit" type="number" min="1" max="{MAX_LIMIT}"
                form="filters-form" value="{_html_attr(limit)}"></label>
            <input type="hidden" name="offset" form="filters-form" value="0">
            <button type="submit" form="filters-form">Apply</button>
        </div>
    """


def _render_strip(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    tiles: list[str] = []
    for row in rows:
        checkpoint = row.get("checkpoint_label") or row.get("eval_id")
        if row.get("gif_exists"):
            gif_url = _fast_gif_url(row)
            preview = (
                f'<img class="strip-preview" loading="lazy" decoding="async" '
                f'fetchpriority="low" src="{_html_attr(gif_url)}" '
                f'width="96" height="96" alt="">'
            )
            href = gif_url
        else:
            preview = '<span class="strip-missing">missing</span>'
            href = _link("/meta", row["summary_ref"])
        tiles.append(
            f"""
            <a class="strip-tile" href="{_html_attr(href)}">
                {preview}
                <span>{_html_attr(checkpoint)}</span>
            </a>
            """
        )
    return f'<div class="strip">{"".join(tiles)}</div>'


def _render_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No self-play GIF summaries found.</p>'

    rendered = []
    for row in rows:
        gif_url = _fast_gif_url(row)
        meta_url = _link("/meta", row["summary_ref"])
        preview = (
            f'<a href="{_html_attr(gif_url)}"><img class="preview" loading="lazy" '
            f'decoding="async" fetchpriority="low" src="{_html_attr(gif_url)}" '
            f'width="128" height="128" alt=""></a>'
            if row["gif_exists"]
            else '<span class="missing">missing raw.gif</span>'
        )
        ok_label = "OK" if row["ok"] is True else "failed" if row["ok"] is False else "unknown"
        steps = row["physical_steps"]
        if row["max_steps"] is not None:
            steps = f"{steps or 0}/{row['max_steps']}"
        rendered.append(
            f"""
            <tr data-run-id="{_html_attr(row["run_id"])}">
                <td>{preview}</td>
                <td><span class="status">{_html_attr(ok_label)}</span></td>
                <td>{_html_attr(row["updated_at"])}</td>
                <td><code>{_html_attr(row["run_id"])}</code></td>
                <td><code>{_html_attr(row["attempt_id"])}</code></td>
                <td><code>{_html_attr(row["eval_id"])}</code></td>
                <td>{_html_attr(row["frame_count"])}</td>
                <td>{_html_attr(steps)}</td>
                <td>{_html_attr(row["terminal_reason"])}</td>
                <td>{_html_attr(row["checkpoint_label"])}</td>
                <td>
                    <a href="{_html_attr(gif_url)}">gif</a>
                    <a href="{_html_attr(meta_url)}">json</a>
                </td>
            </tr>
            """
        )
    return f"""
        <table>
            <thead>
                <tr>
                    <th>Preview</th>
                    <th>Status</th>
                    <th>Updated</th>
                    <th>Run</th>
                    <th>Attempt</th>
                    <th>Eval</th>
                    <th>Frames</th>
                    <th>Steps</th>
                    <th>Reason</th>
                    <th>Checkpoint</th>
                    <th>Links</th>
                </tr>
            </thead>
            <tbody>{''.join(rendered)}</tbody>
        </table>
    """


def _render_pager(
    *,
    row_count: int,
    total_rows: int,
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
    if limit < MAX_LIMIT:
        show_more_url = _filter_url(
            run_id=selected_run_id,
            run_filter=run_filter,
            attempt_filter=attempt_filter,
            eval_filter=eval_filter,
            ok_filter=ok_filter,
            limit=min(MAX_LIMIT, max(200, limit)),
            offset=0,
        )
        links.append(f'<a href="{_html_attr(show_more_url)}">Show more</a>')
    links_html = " ".join(links)
    if links_html:
        links_html = f'<span class="pager-links">{links_html}</span>'
    return (
        f'<div class="pager">Showing {start}-{end} of {total_rows} matching GIFs. '
        f"{links_html}</div>"
    )


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
    reload_error: str | None = None,
) -> str:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    filters = _render_filters(
        runs=runs,
        selected_run_id=selected_run_id,
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
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CurvyTron Self-Play GIFs</title>
    <style>
        body {{
            margin: 24px;
            color: #202124;
            background: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        header {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 16px;
        }}
        h1 {{ margin: 0; font-size: 24px; }}
        .generated {{ color: #5f6368; font-size: 13px; white-space: nowrap; }}
        .filters {{
            display: flex;
            flex-wrap: wrap;
            align-items: end;
            gap: 10px;
            padding: 12px 0 18px;
            border-top: 1px solid #dadce0;
        }}
        label {{ display: grid; gap: 4px; color: #5f6368; font-size: 12px; }}
        .run-control {{
            display: grid;
            gap: 4px;
            color: #5f6368;
            font-size: 12px;
            min-width: 320px;
        }}
        .field-label {{ display: block; }}
        .run-picker {{ position: relative; color: #202124; }}
        .run-picker summary {{
            display: flex;
            align-items: center;
            min-height: 30px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            padding: 0 8px;
            cursor: pointer;
            user-select: none;
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
            border-radius: 4px;
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
            display: block;
            margin: 0;
            padding: 8px;
            overflow: hidden;
            color: #202124;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .hide-run-form {{ margin: 0; }}
        .hide-run-form button {{ margin-right: 6px; }}
        .hide-run-form.is-deleting button {{
            opacity: 0.78;
            cursor: wait;
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
        input, select, button {{
            height: 32px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            padding: 0 8px;
            font: inherit;
        }}
        button {{ background: #1a73e8; border-color: #1a73e8; color: white; }}
        button.danger {{ background: #b3261e; border-color: #b3261e; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 8px; border-top: 1px solid #e8eaed; vertical-align: top; }}
        th {{ text-align: left; color: #5f6368; font-weight: 600; }}
        code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
        a {{ color: #1a73e8; margin-right: 8px; }}
        .preview {{
            width: 128px;
            height: 128px;
            image-rendering: pixelated;
            object-fit: contain;
            border: 1px solid #dadce0;
            background: #f8fafd;
        }}
        .strip {{
            display: flex;
            gap: 10px;
            overflow-x: auto;
            padding: 0 0 14px;
        }}
        .strip-tile {{
            display: grid;
            gap: 4px;
            min-width: 96px;
            color: #202124;
            font-size: 12px;
            text-align: center;
            text-decoration: none;
        }}
        .strip-preview, .strip-missing {{
            width: 96px;
            height: 96px;
            border: 1px solid #dadce0;
            background: #f8fafd;
            object-fit: contain;
            image-rendering: pixelated;
        }}
        .strip-missing {{
            display: grid;
            place-items: center;
            color: #9aa0a6;
        }}
        .status {{ white-space: nowrap; }}
        .pager {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            padding: 0 0 12px;
            color: #5f6368;
            font-size: 13px;
        }}
        .pager-links a {{ margin-right: 10px; }}
        .warning {{ color: #b06000; }}
        .missing, .empty {{ color: #9aa0a6; }}
        @media (max-width: 900px) {{
            body {{ margin: 12px; }}
            header {{ display: block; }}
            table {{ display: block; overflow-x: auto; white-space: nowrap; }}
            .preview {{ width: 96px; height: 96px; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>CurvyTron Self-Play GIFs</h1>
        <span class="generated">{_html_attr(generated_at)}</span>
    </header>
    {filters}
    {reload_warning}
    {pager}
    {_render_strip(rows)}
    {_render_rows(rows)}
    <script>
        for (const form of document.querySelectorAll(".hide-run-form")) {{
            form.addEventListener("submit", async (event) => {{
                event.preventDefault();
                if (form.classList.contains("is-deleting")) return;
                form.classList.add("is-deleting");
                const button = form.querySelector("button");
                const label = form.querySelector(".button-label");
                if (button) button.disabled = true;
                if (label) label.textContent = "Deleting";
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
                    const runId = payload.run_id || button?.dataset.deleteRunId || "";
                    const row = form.closest(".run-menu-row");
                    const picker = form.closest(".run-picker");
                    const selectedInput = document.querySelector(
                        'input[name="run_id"][form="filters-form"]'
                    );
                    const deletedSelectedRun = selectedInput && selectedInput.value === runId;
                    if (row) row.remove();
                    for (const resultRow of document.querySelectorAll("tr[data-run-id]")) {{
                        if (resultRow.dataset.runId === runId) resultRow.remove();
                    }}
                    if (deletedSelectedRun) {{
                        selectedInput.remove();
                        const summary = picker?.querySelector("summary");
                        if (summary) summary.textContent = "Pick a run";
                        const url = new URL(window.location.href);
                        url.searchParams.delete("run_id");
                        window.history.replaceState(null, "", url.pathname + url.search + url.hash);
                    }}
                }} catch (error) {{
                    form.classList.remove("is-deleting");
                    if (button) button.disabled = false;
                    if (label) label.textContent = "Retry";
                }}
            }});
        }}
    </script>
</body>
</html>
"""


def _build_fastapi_app(volume: Any) -> Any:
    from fastapi import FastAPI, Header, Query
    from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

    web_app = FastAPI(title="CurvyTron Self-Play GIF Browser")
    last_reload_at = 0.0
    last_reload_error: str | None = None

    def reload_volume(*, force: bool = False) -> str | None:
        nonlocal last_reload_at, last_reload_error
        if volume is not None and hasattr(volume, "reload"):
            now = monotonic()
            if (
                not force
                and VOLUME_RELOAD_TTL_SECONDS > 0
                and now < last_reload_at + VOLUME_RELOAD_TTL_SECONDS
            ):
                return last_reload_error
            try:
                volume.reload()
                last_reload_error = None
            except Exception as exc:  # pragma: no cover - remote Volume resilience.
                last_reload_error = f"{type(exc).__name__}: {exc}"
            last_reload_at = now
            return last_reload_error
        return None

    @web_app.get("/", response_class=HTMLResponse)
    def index(
        run_id: str = "",
        run: str = "",
        attempt: str = "",
        eval: str = "",
        ok: str = DEFAULT_OK_FILTER,
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
    ) -> HTMLResponse:
        reload_error = reload_volume()
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
                reload_error=reload_error,
            )
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
    ) -> JSONResponse:
        reload_error = reload_volume()
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
                "offset": page["offset"],
                "limit": page["limit"],
                "has_newer": page["has_newer"],
                "has_older": page["has_older"],
                "reload_error": reload_error,
            }
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
            reload_volume(force=True)
            _clear_listing_caches()
        else:
            delete_result = _hide_run_from_picker_on_mount(
                mount=RUNS_MOUNT,
                run_id=clean_run_id,
                volume=volume,
            )
        if x_requested_with == "fetch":
            return JSONResponse({**delete_result, "next": next_url})
        return RedirectResponse(next_url, status_code=303)

    @web_app.get("/gif")
    def gif(ref: str, if_none_match: str = Header(default="")) -> Response:
        try:
            safe_ref = _validate_selfplay_gif_ref(ref)
        except RefValidationError as exc:
            return Response(str(exc), status_code=400)
        reload_volume()
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
        reload_volume()
        path = _path_for_ref(RUNS_MOUNT, safe_ref)
        if not path.is_file():
            return Response("JSON not found", status_code=404)
        return Response(
            path.read_bytes(),
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
        cpu=1,
        memory=512,
        max_containers=2,
    )
    @modal.concurrent(max_inputs=50)
    @modal.asgi_app()
    def gif_browser():
        return _build_fastapi_app(runs_volume)

else:  # pragma: no cover - useful only when importing without the modal extra.

    def gif_browser():
        return _build_fastapi_app(None)
