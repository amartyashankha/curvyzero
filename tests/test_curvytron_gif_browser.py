from __future__ import annotations

import json
import os

import pytest

from curvyzero.infra.modal import curvytron_gif_browser as browser


@pytest.fixture(autouse=True)
def _clear_browser_listing_caches():
    browser._clear_listing_caches()
    browser._LAST_VOLUME_RELOAD_AT = 0.0
    yield
    browser._clear_listing_caches()
    browser._LAST_VOLUME_RELOAD_AT = 0.0


def _selfplay_dir(tmp_path, *, run_id: str, attempt_id: str, eval_id: str):
    return (
        tmp_path
        / "training"
        / browser.TASK_ID
        / run_id
        / "attempts"
        / attempt_id
        / "eval"
        / eval_id
        / "selfplay"
    )


def _write_summary(
    tmp_path,
    *,
    run_id: str,
    attempt_id: str,
    eval_id: str,
    ok: bool,
    frame_count: int,
    mtime: int,
    checkpoint_label: str = "iteration_1",
    terminal_reason: str = "round_all_dead_draw",
    write_gif: bool = True,
) -> None:
    selfplay_dir = _selfplay_dir(
        tmp_path,
        run_id=run_id,
        attempt_id=attempt_id,
        eval_id=eval_id,
    )
    selfplay_dir.mkdir(parents=True)
    gif_ref = (
        f"training/{browser.TASK_ID}/{run_id}/attempts/{attempt_id}/eval/"
        f"{eval_id}/selfplay/raw.gif"
    )
    gif_path = selfplay_dir / "raw.gif"
    if write_gif:
        gif_path.write_bytes(b"GIF89a")
    summary_path = selfplay_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "ok": ok,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "eval_id": eval_id,
                "gif_ref": gif_ref,
                "frame_count": frame_count,
                "physical_steps": frame_count - 1,
                "max_steps": 64,
                "checkpoint_label": checkpoint_label,
                "terminal_reason": terminal_reason,
            }
        ),
        encoding="utf-8",
    )
    if write_gif:
        os.utime(gif_path, (mtime, mtime))
    os.utime(summary_path, (mtime, mtime))


def _write_checkpoint(tmp_path, *, run_id: str, name: str, mtime: int) -> None:
    checkpoint_path = (
        tmp_path
        / "training"
        / browser.TASK_ID
        / run_id
        / "checkpoints"
        / "lightzero"
        / name
    )
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    os.utime(checkpoint_path, (mtime, mtime))


def _write_progress_latest(
    tmp_path,
    *,
    run_id: str,
    attempt_id: str = "attempt-a",
    iteration: int = 1,
    mtime: int,
) -> None:
    progress_path = (
        tmp_path
        / "training"
        / browser.TASK_ID
        / run_id
        / "attempts"
        / attempt_id
        / "train"
        / "progress_latest.json"
    )
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        json.dumps({"iteration": iteration, "elapsed_sec": float(iteration)}),
        encoding="utf-8",
    )
    os.utime(progress_path, (mtime, mtime))


def _write_picker_flag(tmp_path, *, run_id: str, mtime: int = 1) -> None:
    flag_path = (
        tmp_path
        / "training"
        / browser.TASK_ID
        / run_id
        / browser.RUN_PICKER_FLAG_FILENAME
    )
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text("show\n", encoding="utf-8")
    os.utime(flag_path, (mtime, mtime))


def _picker_flag_path(tmp_path, *, run_id: str):
    return (
        tmp_path
        / "training"
        / browser.TASK_ID
        / run_id
        / browser.RUN_PICKER_FLAG_FILENAME
    )


def test_validate_volume_ref_accepts_only_task_relative_gifs_and_json() -> None:
    gif_ref = (
        f"training/{browser.TASK_ID}/run-a/attempts/attempt-a/eval/eval-a/"
        "selfplay/raw.gif"
    )
    json_ref = gif_ref.removesuffix("raw.gif") + "summary.json"

    assert browser._validate_volume_ref(gif_ref, suffix=".gif").as_posix() == gif_ref
    assert browser._validate_volume_ref(json_ref, suffix=".json").as_posix() == json_ref
    assert browser._validate_selfplay_gif_ref(gif_ref).as_posix() == gif_ref
    assert browser._validate_selfplay_summary_ref(json_ref).as_posix() == json_ref


@pytest.mark.parametrize(
    "ref,suffix",
    [
        ("/training/lightzero-curvytron-visual-survival/run/raw.gif", ".gif"),
        ("training/lightzero-curvytron-visual-survival/../run/raw.gif", ".gif"),
        ("training/other-task/run/raw.gif", ".gif"),
        ("training/lightzero-curvytron-visual-survival/run/raw.txt", ".gif"),
        ("training/lightzero-curvytron-visual-survival/run/raw.gif", ".json"),
        ("training/lightzero-curvytron-visual-survival", ".json"),
        ("training/lightzero-curvytron-visual-survival/run\\raw.gif", ".gif"),
    ],
)
def test_validate_volume_ref_rejects_unsafe_refs(ref: str, suffix: str) -> None:
    with pytest.raises(browser.RefValidationError):
        browser._validate_volume_ref(ref, suffix=suffix)


@pytest.mark.parametrize(
    "ref,validator",
    [
        (
            "training/lightzero-curvytron-visual-survival/run-a/summary.json",
            browser._validate_selfplay_summary_ref,
        ),
        (
            "training/lightzero-curvytron-visual-survival/run-a/attempts/attempt-a/eval/"
            "eval-a/manifest.json",
            browser._validate_selfplay_summary_ref,
        ),
        (
            "training/lightzero-curvytron-visual-survival/run-a/attempts/attempt-a/eval/"
            "eval-a/selfplay/other.gif",
            browser._validate_selfplay_gif_ref,
        ),
        (
            "training/lightzero-curvytron-visual-survival/run-a/attempts/attempt-a/"
            "raw.gif",
            browser._validate_selfplay_gif_ref,
        ),
    ],
)
def test_strict_selfplay_refs_reject_other_task_files(ref, validator) -> None:
    with pytest.raises(browser.RefValidationError):
        validator(ref)


def test_list_selfplay_summaries_filters_and_sorts_by_run_recency(tmp_path) -> None:
    _write_picker_flag(tmp_path, run_id="run-old", mtime=100)
    _write_picker_flag(tmp_path, run_id="run-new", mtime=200)
    _write_summary(
        tmp_path,
        run_id="run-old",
        attempt_id="attempt-a",
        eval_id="live_checkpoint_iteration_1",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    _write_summary(
        tmp_path,
        run_id="run-new",
        attempt_id="attempt-b",
        eval_id="live_checkpoint_iteration_2",
        ok=False,
        frame_count=7,
        mtime=200,
    )

    rows = browser._list_selfplay_summaries(tmp_path, ok_filter="all", limit=10)

    assert [row["run_id"] for row in rows] == ["run-new", "run-old"]
    assert rows[0]["gif_exists"] is True
    assert rows[0]["gif_bytes"] == 6
    assert rows[0]["frame_count"] == 7
    assert rows[0]["terminal_reason"] == "round_all_dead_draw"
    assert rows[0]["summary_ref"].endswith("/selfplay/summary.json")
    assert rows[0]["gif_ref"].endswith("/selfplay/raw.gif")

    failed_rows = browser._list_selfplay_summaries(
        tmp_path,
        run_filter="new",
        ok_filter="failed",
        limit=10,
    )

    assert len(failed_rows) == 1
    assert failed_rows[0]["run_id"] == "run-new"


def test_list_selfplay_summaries_defaults_to_successful_gifs(tmp_path) -> None:
    _write_picker_flag(tmp_path, run_id="run-a")
    _write_summary(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="live_checkpoint_iteration_1",
        ok=True,
        frame_count=5,
        mtime=100,
        checkpoint_label="iteration_1",
    )
    _write_summary(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="live_checkpoint_iteration_2",
        ok=True,
        frame_count=5,
        mtime=200,
        checkpoint_label="iteration_2",
        write_gif=False,
    )

    default_rows = browser._list_selfplay_summaries(tmp_path, run_id="run-a")
    failed_rows = browser._list_selfplay_summaries(
        tmp_path,
        run_id="run-a",
        ok_filter="failed",
    )

    assert [row["eval_id"] for row in default_rows] == ["live_checkpoint_iteration_1"]
    assert [row["eval_id"] for row in failed_rows] == ["live_checkpoint_iteration_2"]


def test_list_selfplay_summaries_sorts_selected_run_by_iteration(tmp_path) -> None:
    _write_picker_flag(tmp_path, run_id="run-a")
    for iteration, mtime in ((14, 300), (16, 100), (15, 200)):
        _write_summary(
            tmp_path,
            run_id="run-a",
            attempt_id="attempt-a",
            eval_id=f"live_checkpoint_iteration_{iteration}",
            ok=True,
            frame_count=5,
            mtime=mtime,
            checkpoint_label=f"iteration_{iteration}",
        )

    rows = browser._list_selfplay_summaries(tmp_path, run_id="run-a", limit=10)

    assert [row["eval_id"] for row in rows] == [
        "live_checkpoint_iteration_16",
        "live_checkpoint_iteration_15",
        "live_checkpoint_iteration_14",
    ]


def test_list_selfplay_summaries_reuses_short_ttl_cache_for_same_run_state(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(browser, "LISTING_CACHE_TTL_SECONDS", 60.0)
    _write_picker_flag(tmp_path, run_id="run-a")
    _write_summary(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="live_checkpoint_iteration_1",
        ok=True,
        frame_count=5,
        mtime=100,
    )

    rows = browser._list_selfplay_summaries(tmp_path, run_id="run-a", limit=10)

    def fail_summary_row(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("summary rows should come from cache")

    monkeypatch.setattr(browser, "_summary_row", fail_summary_row)
    cached_rows = browser._list_selfplay_summaries(tmp_path, run_id="run-a", limit=10)

    assert [row["eval_id"] for row in rows] == ["live_checkpoint_iteration_1"]
    assert cached_rows == rows


def test_selected_run_page_cache_preserves_exact_pagination_metadata(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(browser, "LISTING_CACHE_TTL_SECONDS", 60.0)
    _write_picker_flag(tmp_path, run_id="run-a")
    for iteration, mtime in ((2, 200), (1, 100)):
        _write_summary(
            tmp_path,
            run_id="run-a",
            attempt_id="attempt-a",
            eval_id=f"live_checkpoint_iteration_{iteration}",
            ok=True,
            frame_count=5,
            mtime=mtime,
            checkpoint_label=f"iteration_{iteration}",
        )

    first_page = browser._list_selfplay_summary_page(
        tmp_path,
        run_id="run-a",
        limit=2,
    )
    second_page = browser._list_selfplay_summary_page(
        tmp_path,
        run_id="run-a",
        limit=2,
    )

    assert first_page["total_rows"] == 2
    assert first_page["total_rows_exact"] is True
    assert first_page["has_older"] is False
    assert second_page["total_rows"] == 2
    assert second_page["total_rows_exact"] is True
    assert second_page["has_older"] is False


def test_list_runs_sorts_by_recent_run_level_artifacts(tmp_path) -> None:
    _write_picker_flag(tmp_path, run_id="run-summary-newer")
    _write_picker_flag(tmp_path, run_id="run-checkpoint-newest")
    _write_summary(
        tmp_path,
        run_id="run-summary-newer",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=200,
    )
    _write_summary(
        tmp_path,
        run_id="run-checkpoint-newest",
        attempt_id="attempt-b",
        eval_id="eval-b",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    _write_progress_latest(
        tmp_path,
        run_id="run-checkpoint-newest",
        iteration=3,
        mtime=300,
    )

    runs = browser._list_runs(tmp_path)

    assert [run["run_id"] for run in runs] == [
        "run-checkpoint-newest",
        "run-summary-newer",
    ]
    assert runs[0]["artifact_count"] == 1
    assert runs[0]["updated_at"] == "1970-01-01T00:05:00Z"


def test_default_browser_lists_only_runs_with_picker_flag(tmp_path) -> None:
    _write_picker_flag(tmp_path, run_id="run-marked")
    _write_summary(
        tmp_path,
        run_id="run-marked",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=200,
    )
    _write_summary(
        tmp_path,
        run_id="run-unmarked",
        attempt_id="attempt-b",
        eval_id="eval-b",
        ok=True,
        frame_count=7,
        mtime=300,
    )

    runs = browser._list_runs(tmp_path)
    default_rows = browser._list_selfplay_summaries(tmp_path, limit=10)
    exact_unmarked_rows = browser._list_selfplay_summaries(
        tmp_path,
        run_id="run-unmarked",
        limit=10,
    )

    assert [run["run_id"] for run in runs] == ["run-marked"]
    assert [row["run_id"] for row in default_rows] == ["run-marked"]
    assert [row["run_id"] for row in exact_unmarked_rows] == ["run-unmarked"]


def test_list_runs_uses_known_paths_without_recursive_rglob(tmp_path, monkeypatch) -> None:
    _write_picker_flag(tmp_path, run_id="run-known-old", mtime=100)
    _write_picker_flag(tmp_path, run_id="run-known-new", mtime=200)
    _write_summary(
        tmp_path,
        run_id="run-known-old",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    _write_summary(
        tmp_path,
        run_id="run-known-new",
        attempt_id="attempt-b",
        eval_id="eval-b",
        ok=True,
        frame_count=7,
        mtime=200,
    )
    ignored_deep_gif = (
        tmp_path
        / "training"
        / browser.TASK_ID
        / "run-known-old"
        / "deep"
        / "nested"
        / "raw.gif"
    )
    ignored_deep_gif.parent.mkdir(parents=True)
    ignored_deep_gif.write_bytes(b"GIF89a")
    os.utime(ignored_deep_gif, (999, 999))

    def fail_rglob(self, pattern):  # noqa: ANN001
        raise AssertionError(f"unexpected recursive scan: {self!s}.rglob({pattern!r})")

    monkeypatch.setattr(type(tmp_path), "rglob", fail_rglob)

    runs = browser._list_runs(tmp_path)

    assert [run["run_id"] for run in runs[:2]] == ["run-known-new", "run-known-old"]


def test_list_runs_skips_run_level_artifacts_that_disappear_during_volume_scan(
    tmp_path, monkeypatch
) -> None:
    _write_picker_flag(tmp_path, run_id="run-racy")
    _write_progress_latest(
        tmp_path,
        run_id="run-racy",
        mtime=100,
    )
    progress_suffix = ("train", "progress_latest.json")
    real_safe_stat = browser._safe_stat

    def racy_safe_stat(path):  # noqa: ANN001
        if path.parts[-2:] == progress_suffix:
            return None
        return real_safe_stat(path)

    monkeypatch.setattr(browser, "_safe_stat", racy_safe_stat)

    runs = browser._list_runs(tmp_path)

    assert runs[0]["run_id"] == "run-racy"
    assert runs[0]["artifact_count"] == 0


def test_summary_row_treats_disappearing_gif_as_missing(tmp_path, monkeypatch) -> None:
    _write_summary(
        tmp_path,
        run_id="run-racy",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    summary_path = (
        tmp_path
        / "training"
        / browser.TASK_ID
        / "run-racy"
        / "attempts"
        / "attempt-a"
        / "eval"
        / "eval-a"
        / "selfplay"
        / "summary.json"
    )
    raw_gif_suffix = ("selfplay", "raw.gif")
    real_safe_stat = browser._safe_stat

    def racy_safe_stat(path):  # noqa: ANN001
        if path.parts[-2:] == raw_gif_suffix:
            return None
        return real_safe_stat(path)

    monkeypatch.setattr(browser, "_safe_stat", racy_safe_stat)

    row = browser._summary_row(tmp_path, summary_path)

    assert row is not None
    assert row["gif_exists"] is False
    assert row["gif_bytes"] is None


def test_list_selfplay_summaries_falls_back_to_sibling_gif_ref(tmp_path) -> None:
    _write_picker_flag(tmp_path, run_id="run-a")
    selfplay_dir = _selfplay_dir(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="eval-a",
    )
    selfplay_dir.mkdir(parents=True)
    (selfplay_dir / "raw.gif").write_bytes(b"GIF89a")
    (selfplay_dir / "summary.json").write_text('{"ok": true}', encoding="utf-8")

    rows = browser._list_selfplay_summaries(tmp_path)

    assert len(rows) == 1
    assert rows[0]["gif_exists"] is True
    assert rows[0]["gif_ref"] == (
        f"training/{browser.TASK_ID}/run-a/attempts/attempt-a/eval/eval-a/"
        "selfplay/raw.gif"
    )


def test_summary_row_ignores_cross_artifact_gif_ref(tmp_path) -> None:
    _write_summary(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    _write_summary(
        tmp_path,
        run_id="run-b",
        attempt_id="attempt-b",
        eval_id="eval-b",
        ok=True,
        frame_count=5,
        mtime=200,
    )
    summary_path = _selfplay_dir(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="eval-a",
    ) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["gif_ref"] = (
        f"training/{browser.TASK_ID}/run-b/attempts/attempt-b/eval/eval-b/selfplay/raw.gif"
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    row = browser._summary_row(tmp_path, summary_path)

    assert row is not None
    assert row["gif_ref"] == (
        f"training/{browser.TASK_ID}/run-a/attempts/attempt-a/eval/eval-a/"
        "selfplay/raw.gif"
    )


def test_render_rows_includes_terminal_reason_and_versioned_gif_url(tmp_path) -> None:
    _write_picker_flag(tmp_path, run_id="run-a")
    _write_summary(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
        terminal_reason="own_trail",
    )
    rows = browser._list_selfplay_summaries(tmp_path)

    html = browser._render_rows(rows)

    assert "own_trail" in html
    assert "&amp;v=100000000000-6" in html


def test_fastapi_routes_serve_only_selfplay_artifacts(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    class FakeVolume:
        def __init__(self) -> None:
            self.reload_count = 0

        def reload(self) -> None:
            self.reload_count += 1

    volume = FakeVolume()
    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
    _write_picker_flag(tmp_path, run_id="run-a")
    _write_summary(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    stray_json = tmp_path / "training" / browser.TASK_ID / "run-a" / "summary.json"
    stray_json.write_text("{}", encoding="utf-8")
    app = browser._build_fastapi_app(volume)
    client = TestClient(app)
    gif_ref = (
        f"training/{browser.TASK_ID}/run-a/attempts/attempt-a/eval/eval-a/selfplay/raw.gif"
    )
    summary_ref = gif_ref.removesuffix("raw.gif") + "summary.json"

    api_response = client.get("/api/summaries", params={"limit": 5})
    gif_response = client.get("/gif", params={"ref": gif_ref})
    meta_response = client.get("/meta", params={"ref": summary_ref})
    rejected_response = client.get(
        "/meta",
        params={"ref": f"training/{browser.TASK_ID}/run-a/summary.json"},
    )

    assert api_response.status_code == 200
    assert api_response.json()["rows"][0]["gif_ref"] == gif_ref
    assert gif_response.status_code == 200
    assert gif_response.headers["cache-control"].startswith("public, max-age=")
    assert gif_response.headers["etag"]
    assert gif_response.content == b"GIF89a"
    assert meta_response.status_code == 200
    assert rejected_response.status_code == 400
    assert volume.reload_count == 0

    cached_gif_response = client.get(
        "/gif",
        params={"ref": gif_ref},
        headers={"If-None-Match": gif_response.headers["etag"]},
    )

    assert cached_gif_response.status_code == 304


def test_fastapi_routes_reload_only_on_explicit_fresh(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    class FakeVolume:
        def __init__(self) -> None:
            self.reload_count = 0

        def reload(self) -> None:
            self.reload_count += 1

    volume = FakeVolume()
    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
    _write_picker_flag(tmp_path, run_id="run-a")
    _write_summary(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    app = browser._build_fastapi_app(volume)
    client = TestClient(app)

    index_response = client.get("/")
    summaries_response = client.get("/api/summaries")
    head_response = client.get("/api/head")

    assert index_response.status_code == 200
    assert summaries_response.status_code == 200
    assert head_response.status_code == 200
    assert index_response.headers["cache-control"] == "no-store"
    assert summaries_response.headers["cache-control"] == "no-store"
    assert head_response.headers["cache-control"] == "no-store"
    assert volume.reload_count == 0

    assert client.get("/", params={"fresh": "1"}).status_code == 200
    assert volume.reload_count == 1


def test_fastapi_index_and_api_accept_run_id_picker_selection(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
    _write_picker_flag(tmp_path, run_id="run-old", mtime=100)
    _write_picker_flag(tmp_path, run_id="run-new", mtime=200)
    _write_summary(
        tmp_path,
        run_id="run-old",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    _write_summary(
        tmp_path,
        run_id="run-new",
        attempt_id="attempt-b",
        eval_id="eval-b",
        ok=True,
        frame_count=7,
        mtime=200,
    )
    app = browser._build_fastapi_app(None)
    client = TestClient(app)

    page_response = client.get("/", params={"run_id": "run-old"})
    api_response = client.get("/api/summaries", params={"run_id": "run-old"})

    assert page_response.status_code == 200
    assert '<form id="filters-form" method="get"></form>' in page_response.text
    assert '<input type="hidden" name="run_id" form="filters-form" value="run-old">' in (
        page_response.text
    )
    assert '<details class="run-picker">' in page_response.text
    assert "<summary>run-old</summary>" in page_response.text
    assert "confirm(" not in page_response.text
    assert "Deleting" in page_response.text
    assert "actionBusy" in page_response.text
    assert "setRunActionBusy" in page_response.text
    assert "loading run..." in page_response.text
    assert "refreshing..." in page_response.text
    assert "deleting ${runName}" in page_response.text
    assert 'body.action-busy .run-menu-link' in page_response.text
    assert 'headUrl.searchParams.delete("run_id")' in page_response.text
    assert "autoRefresh.disabled = true" in page_response.text
    assert "X-Requested-With" in page_response.text
    assert "row.remove()" in page_response.text
    assert "window.history.replaceState" in page_response.text
    assert "window.location.reload" not in page_response.text
    assert "window.location.assign" not in page_response.text
    assert "window.location.replace" not in page_response.text
    assert 'class="hide-run-form"' in page_response.text
    assert 'class="spinner" aria-hidden="true"' in page_response.text
    assert 'class="gif-card"' in page_response.text
    assert 'id="gallery"' in page_response.text
    assert page_response.text.index('<details class="run-picker">') < page_response.text.index(
        'class="hide-run-form"'
    )
    assert page_response.text.index('class="hide-run-form"') < page_response.text.index(
        "Status"
    )
    assert "Run text" not in page_response.text
    assert "/api/runs/run-old/hide" in page_response.text
    assert page_response.text.index(">run-new</span>") < page_response.text.index(
        ">run-old</span>"
    )
    assert api_response.status_code == 200
    assert [row["run_id"] for row in api_response.json()["rows"]] == ["run-old"]
    assert [run["run_id"] for run in api_response.json()["runs"]] == [
        "run-new",
        "run-old",
    ]


def test_fastapi_defaults_to_newest_run_and_successful_gifs(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
    _write_picker_flag(tmp_path, run_id="run-old", mtime=100)
    _write_picker_flag(tmp_path, run_id="run-new", mtime=200)
    _write_summary(
        tmp_path,
        run_id="run-old",
        attempt_id="attempt-old",
        eval_id="live_checkpoint_iteration_99",
        ok=True,
        frame_count=5,
        mtime=100,
        checkpoint_label="iteration_99",
    )
    _write_summary(
        tmp_path,
        run_id="run-new",
        attempt_id="attempt-new",
        eval_id="live_checkpoint_iteration_16",
        ok=True,
        frame_count=5,
        mtime=200,
        checkpoint_label="iteration_16",
    )
    _write_summary(
        tmp_path,
        run_id="run-new",
        attempt_id="attempt-new",
        eval_id="live_checkpoint_iteration_15",
        ok=True,
        frame_count=5,
        mtime=300,
        checkpoint_label="iteration_15",
        write_gif=False,
    )
    app = browser._build_fastapi_app(None)
    client = TestClient(app)

    page_response = client.get("/")
    api_response = client.get("/api/summaries")
    failed_response = client.get("/api/summaries", params={"run_id": "run-new", "ok": "failed"})

    assert page_response.status_code == 200
    assert "<summary>run-new</summary>" in page_response.text
    assert "live_checkpoint_iteration_15" not in page_response.text
    assert api_response.status_code == 200
    assert api_response.json()["selected_run_id"] == "run-new"
    assert [row["eval_id"] for row in api_response.json()["rows"]] == [
        "live_checkpoint_iteration_16"
    ]
    assert [row["eval_id"] for row in failed_response.json()["rows"]] == [
        "live_checkpoint_iteration_15"
    ]


def test_fastapi_hide_run_removes_only_picker_flag(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
    _write_picker_flag(tmp_path, run_id="run-old")
    _write_summary(
        tmp_path,
        run_id="run-old",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    summary_path = _selfplay_dir(
        tmp_path,
        run_id="run-old",
        attempt_id="attempt-a",
        eval_id="eval-a",
    ) / "summary.json"
    app = browser._build_fastapi_app(None)
    client = TestClient(app)

    response = client.post(
        "/api/runs/run-old/hide",
        params={"next": "/?limit=10"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"] == "/?limit=10"
    assert not _picker_flag_path(tmp_path, run_id="run-old").exists()
    assert summary_path.exists()
    assert browser._list_runs(tmp_path) == []


def test_fastapi_hide_run_fetch_waits_and_returns_next_url(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
    _write_picker_flag(tmp_path, run_id="run-old")
    _write_summary(
        tmp_path,
        run_id="run-old",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    app = browser._build_fastapi_app(None)
    client = TestClient(app)

    response = client.post(
        "/api/runs/run-old/hide",
        params={"next": "/?limit=10"},
        headers={"X-Requested-With": "fetch"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["run_id"] == "run-old"
    assert payload["next"] == "/?limit=10"
    assert payload["marker_existed"] is True
    assert payload["hidden"] is True
    assert not _picker_flag_path(tmp_path, run_id="run-old").exists()


def test_fastapi_hide_run_clears_cached_run_picker_state(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(browser, "LISTING_CACHE_TTL_SECONDS", 60.0)
    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
    _write_picker_flag(tmp_path, run_id="run-old")
    _write_summary(
        tmp_path,
        run_id="run-old",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    app = browser._build_fastapi_app(None)
    client = TestClient(app)

    before = client.get("/api/summaries")
    delete_response = client.post(
        "/api/runs/run-old/hide",
        headers={"X-Requested-With": "fetch"},
    )
    after = client.get("/api/summaries")

    assert before.json()["runs"][0]["run_id"] == "run-old"
    assert delete_response.status_code == 200
    assert after.json()["runs"] == []
    assert after.json()["rows"] == []
    assert after.json()["selected_run_id"] == ""


def test_fastapi_routes_keep_serving_when_volume_reload_fails(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    class FailingVolume:
        def reload(self) -> None:
            raise RuntimeError("temporary volume outage")

    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
    _write_picker_flag(tmp_path, run_id="run-a")
    _write_summary(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    app = browser._build_fastapi_app(FailingVolume())
    client = TestClient(app)

    page_response = client.get("/")
    api_response = client.get("/api/summaries")

    assert page_response.status_code == 200
    assert api_response.status_code == 200
    assert api_response.json()["reload_error"] is None


def test_fastapi_hides_transient_open_file_reload_error(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    class OpenFileVolume:
        def __init__(self) -> None:
            self.reload_count = 0

        def reload(self) -> None:
            self.reload_count += 1
            raise RuntimeError(
                "there are open files preventing the operation: "
                "path training/example/summary.json is open"
            )

    volume = OpenFileVolume()
    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
    _write_picker_flag(tmp_path, run_id="run-a")
    _write_summary(
        tmp_path,
        run_id="run-a",
        attempt_id="attempt-a",
        eval_id="eval-a",
        ok=True,
        frame_count=5,
        mtime=100,
    )
    app = browser._build_fastapi_app(volume)
    client = TestClient(app)

    page_response = client.get("/", params={"fresh": "1"})
    api_response = client.get("/api/head", params={"fresh": "1"})

    assert page_response.status_code == 200
    assert "Volume refresh failed" not in page_response.text
    assert api_response.status_code == 200
    assert api_response.json()["reload_error"] is None
    assert volume.reload_count == 1
