from __future__ import annotations

import json
import os

import pytest

from curvyzero.infra.modal import curvytron_gif_browser as browser


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
    terminal_reason: str = "round_all_dead_draw",
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
    (selfplay_dir / "raw.gif").write_bytes(b"GIF89a")
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
                "checkpoint_label": "iteration_1",
                "terminal_reason": terminal_reason,
            }
        ),
        encoding="utf-8",
    )
    os.utime(summary_path, (mtime, mtime))


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


def test_list_selfplay_summaries_filters_and_sorts_recent(tmp_path) -> None:
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

    rows = browser._list_selfplay_summaries(tmp_path, limit=10)

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


def test_list_selfplay_summaries_falls_back_to_sibling_gif_ref(tmp_path) -> None:
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
    assert "&amp;v=100-6" in html


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
    assert gif_response.headers["cache-control"] == "no-store"
    assert gif_response.content == b"GIF89a"
    assert meta_response.status_code == 200
    assert rejected_response.status_code == 400
    assert volume.reload_count >= 3


def test_fastapi_routes_keep_serving_when_volume_reload_fails(tmp_path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    class FailingVolume:
        def reload(self) -> None:
            raise RuntimeError("temporary volume outage")

    monkeypatch.setattr(browser, "RUNS_MOUNT", tmp_path)
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
    assert "Volume refresh failed" in page_response.text
    assert api_response.status_code == 200
    assert "temporary volume outage" in api_response.json()["reload_error"]
