import pytest

from curvyzero.training.opponent_leaderboard import (
    LEADERBOARD_POINTER_SCHEMA_ID,
    LEADERBOARD_SNAPSHOT_SCHEMA_ID,
    OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID,
    build_leaderboard_pointer,
    build_leaderboard_snapshot_from_rating_snapshot,
    select_opponent_assignment_from_leaderboard,
    select_stable_slots_v1_assignment,
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
    latest_for_run: bool = False,
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
        "latest_for_run": latest_for_run,
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


def test_build_leaderboard_snapshot_defaults_to_top_100_active_pool_and_retired_tail():
    payload = _rating_snapshot()
    payload["ratings"] = [
        _rating_row(
            f"ckpt-{index:03d}",
            run_id=f"run-{index:03d}",
            rank=index + 1,
            rating=2000.0 - index,
            iteration=index * 1000,
            games=500,
            distinct_opponents=20,
        )
        for index in range(101)
    ]

    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        payload,
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
    )

    assert len(snapshot["rows"]) == 101
    assert [row["status"] for row in snapshot["rows"][:100]] == ["active"] * 100
    assert snapshot["rows"][100]["status"] == "retired"
    assert snapshot["rows"][100]["rank"] == 101
    assert snapshot["rows"][100]["eligibility"]["eligible_for_training_default"] is False


def test_build_leaderboard_snapshot_keeps_unplaced_tail_rows_provisional():
    payload = _rating_snapshot()
    payload["ratings"] = [
        _rating_row(
            f"ckpt-{index:03d}",
            run_id=f"run-{index:03d}",
            rank=index + 1,
            rating=2000.0 - index,
            iteration=index * 1000,
            games=500 if index < 100 else 0,
            distinct_opponents=20 if index < 100 else 0,
        )
        for index in range(101)
    ]

    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        payload,
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
    )

    assert snapshot["rows"][100]["rank"] == 101
    assert snapshot["rows"][100]["status"] == "provisional"
    assert snapshot["rows"][100]["eligibility"]["reasons"] == [
        "below_active_rank_limit",
        "insufficient_games",
        "insufficient_distinct_opponents",
    ]


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


def test_leaderboard_pointer_compact_summary_does_not_count_retired_rows_as_provisional():
    payload = _rating_snapshot()
    payload["ratings"] = [
        _rating_row(
            f"ckpt-{index:03d}",
            run_id=f"run-{index:03d}",
            rank=index + 1,
            rating=2000.0 - index,
            iteration=index * 1000,
            games=500,
            distinct_opponents=20,
        )
        for index in range(101)
    ]
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        payload,
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
    )

    pointer = build_leaderboard_pointer(
        snapshot,
        snapshot_ref="tournaments/curvytron/leaderboards/main/snapshots/snapshot-001.json",
        published_at="2026-05-13T00:01:00Z",
    )

    assert pointer["compact_summary"]["row_count"] == 101
    assert pointer["compact_summary"]["active_count"] == 100
    assert pointer["compact_summary"]["provisional_count"] == 0
    assert pointer["compact_summary"]["retired_count"] == 1


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


def test_stable_slots_v1_outputs_parser_compatible_stable5_assignment_and_audit():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
        generation=7,
    )

    assignment, audit = select_stable_slots_v1_assignment(
        snapshot,
        assignment_id="stable-slots-001",
        source_ref="tournaments/curvytron/leaderboards/main/snapshots/snapshot-001.json",
        seed=123,
        profile="stable_5",
        sentinel="blank_canvas",
        expected_rating_context_hash="ctx-a",
        checkpoint_death_mode="immortal",
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
    checkpoint_entries = assignment["entries"][:-1]
    assert {
        entry["opponent_policy_kind"] for entry in checkpoint_entries
    } == {"frozen_lightzero_checkpoint"}
    assert {
        entry["opponent_death_mode"] for entry in checkpoint_entries
    } == {"immortal"}
    assert assignment["entries"][-1]["opponent_policy_kind"] == "fixed_straight"
    assert assignment["entries"][-1]["opponent_runtime_mode"] == "blank_canvas_noop"

    parsed = parse_opponent_assignment_snapshot(assignment)
    assert parsed is not None
    assert parsed["opponent_mixture"]["total_weight"] > 0.0

    assert audit["schema_id"] == OPPONENT_ASSIGNMENT_AUDIT_SCHEMA_ID
    assert audit["source_leaderboard"]["snapshot_sha256"] == snapshot["snapshot_sha256"]
    assert audit["selection"]["strategy_id"] == "stable_slots_v1"
    assert audit["selection"]["profile"] == "stable_5"
    assert audit["selection"]["sentinel"] == "blank_canvas"
    assert validate_assignment_audit(audit, assignment=assignment)["assignment_id"] == (
        "stable-slots-001"
    )


def test_stable_slots_v1_uses_nested_recency_latest_for_run():
    rating_snapshot = _rating_snapshot()
    rating_snapshot["ratings"] = [
        _rating_row("ckpt-a", run_id="run-a", rank=1, rating=1700.0, iteration=100000),
        _rating_row("ckpt-b", run_id="run-b", rank=2, rating=1650.0, iteration=90000),
        _rating_row(
            "ckpt-c",
            run_id="run-c",
            rank=3,
            rating=1600.0,
            iteration=80000,
            latest_for_run=True,
        ),
    ]
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        rating_snapshot,
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-recency",
    )

    assignment, audit = select_stable_slots_v1_assignment(
        snapshot,
        assignment_id="stable-slots-recency",
        source_ref="snapshot-ref",
        profile="stable_3",
        sentinel="none",
    )

    entries_by_name = {entry["name"]: entry for entry in assignment["entries"]}
    assert len(assignment["entries"]) == 3
    assert entries_by_name["slot_recent_strong"]["opponent_checkpoint_ref"].endswith(
        "iteration_80000.pth.tar"
    )
    recent_evidence = {
        row["slot"]: row for row in audit["selected_rows"]
    }["recent_strong"]
    assert recent_evidence["checkpoint_id"] == "ckpt-c"


def test_stable_slots_v1_rejects_rating_context_mismatch():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
    )

    with pytest.raises(ValueError, match="rating_context_hash"):
        select_stable_slots_v1_assignment(
            snapshot,
            assignment_id="stable-slots-wrong-context",
            source_ref="snapshot-ref",
            profile="stable_3",
            sentinel="none",
            expected_rating_context_hash="different-context",
        )


def test_stable_slots_v1_allows_provisional_recent_slot_only():
    rating_snapshot = _rating_snapshot()
    rating_snapshot["ratings"] = [
        _rating_row(
            "ckpt-provisional",
            run_id="run-provisional",
            rank=1,
            rating=3000.0,
            iteration=120000,
            games=20,
            distinct_opponents=2,
            latest_for_run=True,
        ),
        _rating_row("ckpt-a", run_id="run-a", rank=2, rating=1700.0, iteration=100000),
        _rating_row("ckpt-b", run_id="run-b", rank=3, rating=1650.0, iteration=90000),
    ]
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        rating_snapshot,
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-provisional-recent",
    )

    assignment, audit = select_stable_slots_v1_assignment(
        snapshot,
        assignment_id="stable-slots-provisional-recent",
        source_ref="snapshot-ref",
        profile="stable_3",
        sentinel="none",
        allow_recent_provisional=True,
    )

    selected_by_slot = {row["slot"]: row for row in audit["selected_rows"]}
    assert selected_by_slot["champion"]["checkpoint_id"] == "ckpt-a"
    assert selected_by_slot["champion"]["leaderboard_status"] == "active"
    assert selected_by_slot["recent_strong"]["checkpoint_id"] == "ckpt-provisional"
    assert selected_by_slot["recent_strong"]["leaderboard_status"] == "provisional"
    assert selected_by_slot["diverse_challenger"]["checkpoint_id"] == "ckpt-b"
    assert selected_by_slot["diverse_challenger"]["leaderboard_status"] == "active"

    assert [
        entry["opponent_checkpoint_ref"].split("/")[-1]
        for entry in assignment["entries"]
    ] == [
        "iteration_100000.pth.tar",
        "iteration_120000.pth.tar",
        "iteration_90000.pth.tar",
    ]


def test_stable_slots_v1_does_not_use_non_recent_provisional_fallback():
    rating_snapshot = _rating_snapshot()
    rating_snapshot["ratings"] = [
        _rating_row(
            "ckpt-provisional",
            run_id="run-provisional",
            rank=1,
            rating=3000.0,
            iteration=120000,
            games=20,
            distinct_opponents=2,
            latest_for_run=False,
        ),
        _rating_row("ckpt-a", run_id="run-a", rank=2, rating=1700.0, iteration=100000),
        _rating_row("ckpt-b", run_id="run-b", rank=3, rating=1650.0, iteration=90000),
        _rating_row("ckpt-c", run_id="run-c", rank=4, rating=1600.0, iteration=80000),
    ]
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        rating_snapshot,
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-provisional-not-recent",
    )

    _assignment, audit = select_stable_slots_v1_assignment(
        snapshot,
        assignment_id="stable-slots-provisional-not-recent",
        source_ref="snapshot-ref",
        profile="stable_3",
        sentinel="none",
        allow_recent_provisional=True,
    )

    selected_by_slot = {row["slot"]: row for row in audit["selected_rows"]}
    assert selected_by_slot["recent_strong"]["checkpoint_id"] == "ckpt-b"
    assert selected_by_slot["recent_strong"]["leaderboard_status"] == "active"
    assert "ckpt-provisional" not in {
        row["checkpoint_id"] for row in audit["selected_rows"]
    }


def test_stable_slots_v1_outputs_wall_avoidant_immortal_sentinel():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-001",
    )

    assignment, audit = select_stable_slots_v1_assignment(
        snapshot,
        assignment_id="stable-slots-wall-sentinel",
        source_ref="snapshot-ref",
        profile="stable_5",
        sentinel="wall_avoidant_immortal",
    )

    sentinel = assignment["entries"][-1]
    assert sentinel["name"] == "slot_sentinel_wall_avoidant_immortal"
    assert sentinel["opponent_policy_kind"] == "proactive_wall_avoidant"
    assert sentinel["opponent_runtime_mode"] == "normal"
    assert sentinel["opponent_death_mode"] == "immortal"
    assert parse_opponent_assignment_snapshot(assignment) is not None
    assert audit["hardcoded_slots"][-1]["slot"] == "sentinel"
    assert audit["hardcoded_slots"][-1]["slot_kind"] == "wall_avoidant_immortal"


@pytest.mark.parametrize(
    ("profile", "sentinel", "expected_names"),
    [
        (
            "stable_3",
            "none",
            ["slot_champion", "slot_recent_strong", "slot_diverse_challenger"],
        ),
        (
            "stable_5",
            "blank_canvas",
            [
                "slot_champion",
                "slot_recent_strong",
                "slot_diverse_challenger",
                "slot_anchor",
                "slot_sentinel_blank_canvas",
            ],
        ),
    ],
)
def test_stable_slots_v1_profile_exactness(profile, sentinel, expected_names):
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id=f"snapshot-{profile}-{sentinel}",
    )

    assignment, _audit = select_stable_slots_v1_assignment(
        snapshot,
        assignment_id=f"stable-slots-{profile}-{sentinel}",
        source_ref="snapshot-ref",
        profile=profile,
        sentinel=sentinel,
    )

    assert [entry["name"] for entry in assignment["entries"]] == expected_names


def test_stable_slots_v1_rejects_stable5_without_sentinel():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-stable5-no-sentinel",
    )

    with pytest.raises(ValueError, match="stable_5 requires a sentinel"):
        select_stable_slots_v1_assignment(
            snapshot,
            assignment_id="stable-slots-stable5-no-sentinel",
            source_ref="snapshot-ref",
            profile="stable_5",
            sentinel="none",
        )


def test_stable_slots_v1_rejects_profile_when_it_would_silently_drop_slots():
    rating_snapshot = _rating_snapshot()
    rating_snapshot["ratings"] = [
        _rating_row("ckpt-a", run_id="run-a", rank=1, rating=1700.0, iteration=100000),
        _rating_row("ckpt-b", run_id="run-b", rank=2, rating=1650.0, iteration=90000),
    ]
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        rating_snapshot,
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-too-small",
    )

    with pytest.raises(ValueError, match="stable_3"):
        select_stable_slots_v1_assignment(
            snapshot,
            assignment_id="stable-slots-too-small",
            source_ref="snapshot-ref",
            profile="stable_3",
            sentinel="none",
        )


def test_stable_slots_v1_audit_validation_checks_slot_plan_shape():
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-audit-shape",
    )
    assignment, audit = select_stable_slots_v1_assignment(
        snapshot,
        assignment_id="stable-slots-audit-shape",
        source_ref="snapshot-ref",
        profile="stable_3",
        sentinel="none",
    )
    audit["slot_plan"] = audit["slot_plan"][:-1]

    with pytest.raises(ValueError, match="slot_plan length"):
        validate_assignment_audit(audit, assignment=assignment)
