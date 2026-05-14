import importlib.util
from pathlib import Path

import pytest

from curvyzero.training.opponent_leaderboard import (
    build_leaderboard_snapshot_from_rating_snapshot,
)


def _load_materializer_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "materialize_curvytron_leaderboard_assignment.py"
    )
    spec = importlib.util.spec_from_file_location(
        "materialize_curvytron_leaderboard_assignment",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/attempts/attempt-a/train/lightzero_exp/ckpt/"
        f"iteration_{iteration}.pth.tar"
    )


def _rating_snapshot():
    return {
        "schema_id": "curvyzero_curvytron_checkpoint_rating_snapshot/v0",
        "formula_version": "batch_elo_v0",
        "tournament_id": "arena-a",
        "rating_run_id": "elo-a",
        "ratings_ref": "tournaments/curvytron/arena-a/ratings/elo-a/latest.json",
        "context_hash": "ctx-a",
        "roster_hash": "roster-a",
        "created_at": "2026-05-14T00:00:00Z",
        "ratings": [
            {
                "checkpoint_id": "ckpt-a",
                "checkpoint_ref": _checkpoint_ref("run-a", 100000),
                "run_id": "run-a",
                "rank": 1,
                "rating": 1700.0,
                "games": 500,
                "wins": 300,
                "losses": 150,
                "draws": 50,
                "battles": 20,
                "rated_battles": 20,
                "distinct_opponents": 20,
                "failure_count": 0,
            }
        ],
    }


def test_materializer_accepts_published_leaderboard_snapshot_without_rebuilding():
    materializer = _load_materializer_module()
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-published",
        generation=12,
        created_at="2026-05-14T01:00:00Z",
    )

    resolved = materializer._leaderboard_snapshot_from_payload(
        snapshot,
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-published",
        generation=999,
        created_at="2099-01-01T00:00:00Z",
    )

    assert resolved == snapshot
    assert resolved["generation"] == 12
    assert resolved["created_at"] == "2026-05-14T01:00:00Z"
    assert resolved["snapshot_sha256"] == snapshot["snapshot_sha256"]


def test_materializer_rejects_published_snapshot_identity_mismatch():
    materializer = _load_materializer_module()
    snapshot = build_leaderboard_snapshot_from_rating_snapshot(
        _rating_snapshot(),
        leaderboard_id="curvytron-main",
        snapshot_id="snapshot-published",
    )

    with pytest.raises(ValueError, match="snapshot_id"):
        materializer._leaderboard_snapshot_from_payload(
            snapshot,
            leaderboard_id="curvytron-main",
            snapshot_id="different-snapshot",
            generation=0,
            created_at="2026-05-14T01:00:00Z",
        )
