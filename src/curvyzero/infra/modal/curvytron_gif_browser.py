"""Tiny Modal ASGI browser for CurvyTron checkpoint self-play GIFs.

Usage:

    uv run --extra modal modal serve -m curvyzero.infra.modal.curvytron_gif_browser
    uv run --extra modal modal deploy -m curvyzero.infra.modal.curvytron_gif_browser
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

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
RUN_PICKER_FLAG_FILENAME = run_mgmt.GIF_BROWSER_RUN_MARKER_FILENAME
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
    is_ok = row.get("ok") is True
    if normalized in {"1", "true", "ok", "pass", "passed"}:
        return is_ok
    if normalized in {"0", "false", "fail", "failed"}:
        return not is_ok
    return True


def _coerce_limit(limit: int) -> int:
    return min(max(1, int(limit)), MAX_LIMIT)


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


def _run_has_picker_flag(run_path: Path) -> bool:
    return _safe_is_file(run_path / RUN_PICKER_FLAG_FILENAME)


def _list_runs(mount: Path) -> list[dict[str, Any]]:
    base_path = _path_for_ref(mount, BASE_REF)
    if not _safe_is_dir(base_path):
        return []

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
    return runs


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
    ok_filter: str = "all",
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    base_path = _path_for_ref(mount, BASE_REF)
    if not _safe_is_dir(base_path):
        return []

    rows: list[dict[str, Any]] = []
    if run_id:
        try:
            clean_run_id = run_mgmt.clean_id(run_id, label="run_id")
        except ValueError:
            return []
        summary_paths = _safe_glob(
            base_path / clean_run_id,
            "attempts/*/eval/*/selfplay/summary.json",
        )
    else:
        summary_paths = []
        for run in _list_runs(mount):
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
        if run_id and row["run_id"] != run_id:
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

    rows.sort(key=lambda item: item["updated_ts"], reverse=True)
    return rows[: _coerce_limit(limit)]


def _html_attr(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _link(path: str, ref: str, **params: Any) -> str:
    query = [f"ref={quote(ref, safe='')}"]
    for key, value in params.items():
        if value is None:
            continue
        query.append(f"{quote(str(key), safe='')}={quote(str(value), safe='')}")
    return f"{path}?{'&'.join(query)}"


def _render_filters(
    *,
    runs: list[dict[str, Any]],
    selected_run_id: str,
    run_filter: str,
    attempt_filter: str,
    eval_filter: str,
    ok_filter: str,
    limit: int,
) -> str:
    run_options = ['<option value="">All runs</option>']
    for run in runs:
        run_id = str(run["run_id"])
        selected = " selected" if run_id == selected_run_id else ""
        label = f"{run_id} ({run['updated_at']})"
        run_options.append(
            f'<option value="{_html_attr(run_id)}"{selected}>{_html_attr(label)}</option>'
        )
    status_options = []
    for value, label in (("all", "All"), ("ok", "OK"), ("failed", "Failed")):
        selected = " selected" if ok_filter.lower() == value else ""
        status_options.append(f'<option value="{value}"{selected}>{label}</option>')
    return f"""
        <form method="get" class="filters">
            <label>Run <select name="run_id" onchange="this.form.submit()">{''.join(run_options)}</select></label>
            <label>Run text <input name="run" value="{_html_attr(run_filter)}"></label>
            <label>Attempt <input name="attempt" value="{_html_attr(attempt_filter)}"></label>
            <label>Eval <input name="eval" value="{_html_attr(eval_filter)}"></label>
            <label>Status <select name="ok">{''.join(status_options)}</select></label>
            <label>Limit <input name="limit" type="number" min="1" max="{MAX_LIMIT}"
                value="{_html_attr(limit)}"></label>
            <button type="submit">Apply</button>
        </form>
    """


def _render_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="empty">No self-play GIF summaries found.</p>'

    rendered = []
    for row in rows:
        gif_version = f"{int(row['updated_ts'])}-{row.get('gif_bytes') or 0}"
        gif_url = _link("/gif", row["gif_ref"], v=gif_version)
        meta_url = _link("/meta", row["summary_ref"])
        preview = (
            f'<a href="{_html_attr(gif_url)}"><img class="preview" loading="lazy" '
            f'src="{_html_attr(gif_url)}" alt=""></a>'
            if row["gif_exists"]
            else '<span class="missing">missing raw.gif</span>'
        )
        ok_label = "OK" if row["ok"] is True else "failed" if row["ok"] is False else "unknown"
        steps = row["physical_steps"]
        if row["max_steps"] is not None:
            steps = f"{steps or 0}/{row['max_steps']}"
        rendered.append(
            f"""
            <tr>
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
        input, select, button {{
            height: 32px;
            border: 1px solid #dadce0;
            border-radius: 4px;
            padding: 0 8px;
            font: inherit;
        }}
        button {{ background: #1a73e8; border-color: #1a73e8; color: white; }}
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
        .status {{ white-space: nowrap; }}
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
    {_render_rows(rows)}
</body>
</html>
"""


def _build_fastapi_app(volume: Any) -> Any:
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    web_app = FastAPI(title="CurvyTron Self-Play GIF Browser")

    def reload_volume() -> str | None:
        if volume is not None and hasattr(volume, "reload"):
            try:
                volume.reload()
            except Exception as exc:  # pragma: no cover - remote Volume resilience.
                return f"{type(exc).__name__}: {exc}"
        return None

    @web_app.get("/", response_class=HTMLResponse)
    def index(
        run_id: str = "",
        run: str = "",
        attempt: str = "",
        eval: str = "",
        ok: str = "all",
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> HTMLResponse:
        reload_error = reload_volume()
        runs = _list_runs(RUNS_MOUNT)
        rows = _list_selfplay_summaries(
            RUNS_MOUNT,
            run_id=run_id,
            run_filter=run,
            attempt_filter=attempt,
            eval_filter=eval,
            ok_filter=ok,
            limit=limit,
        )
        return HTMLResponse(
            _render_page(
                rows,
                runs=runs,
                selected_run_id=run_id,
                run_filter=run,
                attempt_filter=attempt,
                eval_filter=eval,
                ok_filter=ok,
                limit=limit,
                reload_error=reload_error,
            )
        )

    @web_app.get("/api/summaries")
    def summaries(
        run_id: str = "",
        run: str = "",
        attempt: str = "",
        eval: str = "",
        ok: str = "all",
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    ) -> JSONResponse:
        reload_error = reload_volume()
        runs = _list_runs(RUNS_MOUNT)
        rows = _list_selfplay_summaries(
            RUNS_MOUNT,
            run_id=run_id,
            run_filter=run,
            attempt_filter=attempt,
            eval_filter=eval,
            ok_filter=ok,
            limit=limit,
        )
        return JSONResponse({"rows": rows, "runs": runs, "reload_error": reload_error})

    @web_app.get("/gif")
    def gif(ref: str) -> Response:
        try:
            safe_ref = _validate_selfplay_gif_ref(ref)
        except RefValidationError as exc:
            return Response(str(exc), status_code=400)
        reload_volume()
        path = _path_for_ref(RUNS_MOUNT, safe_ref)
        if not path.is_file():
            return Response("GIF not found", status_code=404)
        return Response(
            path.read_bytes(),
            media_type="image/gif",
            headers={"Cache-Control": "no-store"},
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
