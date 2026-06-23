import json

import pytest

from curvyzero.observability.feedback_loop_lineage import (
    LINEAGE_EVENT_SCHEMA_ID,
    LINEAGE_EVENTS_FILENAME,
    append_lineage_event,
    lineage_event,
    lineage_events_path,
)


def test_append_lineage_event_writes_jsonl_boundary_rows(tmp_path):
    path = tmp_path / "lineage_events.jsonl"

    first = append_lineage_event(
        path,
        stage="checkpoint_written",
        run_id="run-a",
        checkpoint_ref="training/task/run-a/checkpoints/iteration_1.pth.tar",
        iteration=1,
        counts_json={"sidecar": True},
    )
    second = append_lineage_event(
        path,
        stage="trainer_assignment_applied",
        assignment_sha256="abc123",
        trainer_modal_task_id="task-1",
    )

    assert first["ok"] is True
    assert second["ok"] is True
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in rows] == [
        "checkpoint_written",
        "trainer_assignment_applied",
    ]
    assert rows[0]["schema_id"] == LINEAGE_EVENT_SCHEMA_ID
    assert rows[0]["checkpoint_ref"].endswith("iteration_1.pth.tar")
    assert rows[0]["counts_json"] == {"sidecar": True}
    assert rows[0]["event_id"] != rows[1]["event_id"]


def test_lineage_event_rejects_unknown_stage():
    with pytest.raises(ValueError, match="stage must be one of"):
        lineage_event(stage="made_up_stage")


def test_append_lineage_event_is_best_effort_by_default(tmp_path):
    result = append_lineage_event(
        tmp_path,
        stage="checkpoint_written",
        checkpoint_ref="training/task/run-a/checkpoints/iteration_1.pth.tar",
    )

    assert result["ok"] is False
    assert result["error_type"]
    assert result["event"]["stage"] == "checkpoint_written"


def test_append_lineage_event_can_raise_write_failures(tmp_path):
    with pytest.raises(OSError):
        append_lineage_event(
            tmp_path,
            best_effort=False,
            stage="checkpoint_written",
        )


def test_lineage_events_path_is_per_owner_artifact_root(tmp_path):
    assert lineage_events_path(tmp_path / "attempt-a") == (
        tmp_path / "attempt-a" / "feedback_loop" / LINEAGE_EVENTS_FILENAME
    )
