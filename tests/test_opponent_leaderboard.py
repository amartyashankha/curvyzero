import pytest

from curvyzero.training.opponent_leaderboard import (
    LEADERBOARD_POINTER_SCHEMA_ID,
    LEADERBOARD_SNAPSHOT_SCHEMA_ID,
    OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID,
    build_leaderboard_pointer,
    build_leaderboard_snapshot_from_rating_snapshot,
    select_opponent_assignment_from_leaderboard,
    validate_assignment_audit,
    validate_leaderboard_pointer,
    validate_leaderboard_snapshot,
)
from curvyzero.training.opponent_registry import (
    OPPONENT_ASSIGNMENT_SCHEMA_ID,
    parse_opponent_assignment_snapshot,
)


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/attempts/attempt-a/train/lightzero_exp/ckpt/"
        f"iteration_{iteration}.pth.tar"
    )


def _rating_row(
    checkpoint_id: str,
    *,
    run_id: str,
    rank: int,
    rating: float,
    iteration: int,
    games: int = 500,
    distinct_opponents: int = 20,
    failure_count: int = 0,
):
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_ref": _checkpoint_ref(run_id, iteration),
        "run_id": run_id,
        "label": f"{run_id} i{iteration}",
        "rank": rank,
        "rating": rating,
        "games": games,
        "wins": int(games * 0.6),
        "losses": int(games * 0.3),
        "draws": games - int(games * 0.6) - int(games * 0.3),
        "battles": distinct_opponents,
        "rated_battles": distinct_opponents,
        "distinct_opponents": distinct_opponents,
        "failure_count": failure_count,
        "win_rate": 0.6,
    }


def _rating_snapshot():
    return {
        "schema_id": "curvyzero_curvytron_checkpoint_rating_snapshot/v0",
        "formula_version": "batch_elo_v0",
        "tournament_id": "arena-a",
        "rating_run_id": "elo-a",
        "ratings_ref": "tournaments/curvytron/arena-a/ratings/elo-a/latest.json",
        "context_hash": "ctx-a",
        "roster_hash": "roster-a",
        "created_at": "2026-05-13T00:00:00Z",
        "ratings": [
            _rating_row("ckpt-a", run_id="run-a", rank=1, rating=1700.0, iteration=100000),
            _rating_row("ckpt-b", run_id="run-b", rank=2, rating=1650.0, iteration=90000),
            _rating_row("ckpt-c", run_id="run-c", rank=3, rating=1600.0, iteration=80000),
            _rating_row("ckpt-d", run_id="run-d", rank=4, rating=1500.0, iteration=70000),
            _rating_row(
                "ckpt-e",
                run_id="run-e",
                rank=5,
                rating=1400.0,
                iteration=60000,
                games=100,
                distinct_opponents=3,
            ),
        ],
    }


def test_build_leaderboard_snapshot_from_rating_snapshot_marks_active_and_provisional():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
        generation=7,
    )

    assert snapshot["schema_id"] == LEADERBOARD_SNAPSHOT_SCHEMA_ID
    assert snapshot["leaderboard_id"] == "curvytron-main"
    assert snapshot["generation"] == 7
    assert snapshot["source"]["tournament_id"] == "arena-a"
    assert snapshot["context"]["rating_context_hash"] == "ctx-a"
    assert snapshot["snapshot_sha256"]
    assert [row["status"] for row in snapshot["rows"]] == [
        "active",
        "active",
        "active",
        "active",
        "provisional",
    ]

    validated = validate_leaderboard_snapshot(snapshot)
    assert validated["snapshot_sha256"] == snapshot["snapshot_sha256"]


def test_build_leaderboard_snapshot_rejects_mutable_checkpoint_refs():
    payload = _rating_snapshot()
    payload["ratings"][0]["checkpoint_ref"] = (
        "training/lightzero-curvytron-visual-survival/run-a/checkpoints/latest.pth.tar"
    )

    with pytest.raises(ValueError, match="immutable exact iteration_N"):
        build_leaderboard_snapshot_from_rating_snapshot(
            payload,
            leaderboard_id="curvytron-main",
            snapshot_id="snapshot-001",
        )


def test_leaderboard_pointer_round_trips_compact_summary():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
        generation=7,
    )

    pointer = build_leaderboard_pointer(
        snapshot,
        snapshot_ref="tournaments/curvytron/leaderboards/main/snapshots/snapshot-001.json",
        published_at="2026-05-13T00:01:00Z",
        writer={"kind": "test"},
    )

    assert pointer["schema_id"] == LEADERBOARD_POINTER_SCHEMA_ID
    assert pointer["generation"] == 7
    assert pointer["compact_summary"]["row_count"] == 5
    assert pointer["compact_summary"]["active_count"] == 4
    assert pointer["compact_summary"]["top_checkpoint_ids"][:2] == ["ckpt-a", "ckpt-b"]
    assert validate_leaderboard_pointer(pointer)["snapshot_id"] == "snapshot-001"


def test_select_opponent_assignment_from_leaderboard_outputs_parser_compatible_assignment():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
        generation=7,
    )

    assignment, audit = select_opponent_assignment_from_leaderboard(
        snapshot,
        assignment_id="run-a-attempt-a-refresh000",
        source_ref="tournaments/curvytron/leaderboards/main/snapshots/snapshot-001.json",
        seed=123,
        max_slots=5,
    )

    assert assignment["schema_id"] == OPPONENT_ASSIGNMENT_SCHEMA_ID
    assert assignment["source_epoch"] == 7
    assert [entry["name"] for entry in assignment["entries"]] == [
        "slot_champion",
        "slot_recent_strong",
        "slot_diverse_challenger",
        "slot_anchor",
        "slot_sentinel_blank_canvas",
    ]
    assert assignment["entries"][0]["opponent_checkpoint_ref"].endswith(
        "iteration_100000.pth.tar"
    )
    assert assignment["entries"][-1]["opponent_runtime_mode"] == "blank_canvas_noop"

    parsed = parse_opponent_assignment_snapshot(assignment)
    assert parsed is not None
    assert parsed["opponent_mixture"]["total_weight"] == 50.0

    assert audit["schema_id"] == OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID
    assert audit["source_leaderboard"]["snapshot_id"] == "snapshot-001"
    assert validate_assignment_audit(audit, assignment=assignment)["assignment_id"] == (
        "run-a-attempt-a-refresh000"
    )


def test_select_opponent_assignment_sorts_unsorted_rows_before_selecting_champion():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
        generation=7,
    )
    rows_by_id = {row["checkpoint_id"]: dict(row) for row in snapshot["rows"]}
    snapshot["rows"] = [
        rows_by_id["ckpt-d"],
        rows_by_id["ckpt-c"],
        rows_by_id["ckpt-b"],
        rows_by_id["ckpt-a"],
        rows_by_id["ckpt-e"],
    ]
    snapshot.pop("snapshot_sha256")

    assignment, audit = select_opponent_assignment_from_leaderboard(
        snapshot,
        assignment_id="assignment-unsorted",
        source_ref="snapshot-ref",
    )

    assert audit["selected_rows"][0]["slot"] == "champion"
    assert audit["selected_rows"][0]["checkpoint_id"] == "ckpt-a"
    assert assignment["entries"][0]["opponent_checkpoint_ref"].endswith(
        "iteration_100000.pth.tar"
    )


def test_select_opponent_assignment_allow_provisional_excludes_retired_rows():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
        generation=7,
    )
    rows_by_id = {row["checkpoint_id"]: dict(row) for row in snapshot["rows"]}
    rows_by_id["ckpt-a"]["status"] = "retired"
    rows_by_id["ckpt-a"]["rank"] = 1
    rows_by_id["ckpt-a"]["rating"] = 3000.0
    rows_by_id["ckpt-e"]["status"] = "provisional"
    rows_by_id["ckpt-e"]["rank"] = 2
    rows_by_id["ckpt-e"]["rating"] = 2000.0
    rows_by_id["ckpt-b"]["status"] = "active"
    rows_by_id["ckpt-b"]["rank"] = 3
    rows_by_id["ckpt-b"]["rating"] = 1900.0
    snapshot["rows"] = [rows_by_id["ckpt-a"], rows_by_id["ckpt-e"], rows_by_id["ckpt-b"]]
    snapshot.pop("snapshot_sha256")

    _assignment, audit = select_opponent_assignment_from_leaderboard(
        snapshot,
        assignment_id="assignment-provisional",
        source_ref="snapshot-ref",
        max_slots=3,
        include_blank_sentinel=False,
        allow_provisional=True,
    )

    assert [row["checkpoint_id"] for row in audit["selected_rows"]] == ["ckpt-e", "ckpt-b"]
    assert [row["leaderboard_status"] for row in audit["selected_rows"]] == [
        "provisional",
        "active",
    ]


def test_select_opponent_assignment_reuses_same_snapshot_deterministically():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
        generation=7,
    )

    first, first_audit = select_opponent_assignment_from_leaderboard(
        snapshot,
        assignment_id="assignment-a",
        source_ref="snapshot-ref",
        seed=123,
    )
    second, second_audit = select_opponent_assignment_from_leaderboard(
        snapshot,
        assignment_id="assignment-a",
        source_ref="snapshot-ref",
        seed=123,
    )

    assert first == second
    assert first_audit == second_audit
