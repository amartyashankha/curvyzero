import json

import pytest

from curvyzero.infra.modal import run_management as runs


def test_write_json_overwrite_uses_stable_payload_and_cleans_temp(tmp_path):
    path = tmp_path / "progress_latest.json"

    first = runs.write_json(path, {"iteration": 0})
    second = runs.write_json(path, {"iteration": 10, "status": "running"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "iteration": 10,
        "status": "running",
    }
    assert second["bytes"] == path.stat().st_size
    assert second["sha256"] != first["sha256"]
    assert not list(tmp_path.glob(".progress_latest.json.*.tmp"))


def test_write_json_exclusive_still_refuses_existing_file(tmp_path):
    path = tmp_path / "manifest.json"
    runs.write_json(path, {"run_id": "a"}, exclusive=True)

    with pytest.raises(FileExistsError):
        runs.write_json(path, {"run_id": "b"}, exclusive=True)
