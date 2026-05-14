from __future__ import annotations

from pathlib import Path

import pytest

from curvyzero.infra.modal import curvyzero_checkpoint_tournament as modal_arena
from curvyzero.training.opponent_leaderboard import (
    build_leaderboard_pointer,
    build_leaderboard_snapshot_from_rating_snapshot,
)
from curvyzero.tournament import curvytron_checkpoint_tournament as arena


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        "training/lightzero-curvytron-visual-survival/"
        f"{run_id}/checkpoints/lightzero/iteration_{iteration}.pth.tar"
    )


def _leaderboard_snapshot(
    *,
    leaderboard_id: str = "main",
    snapshot_id: str = "snapshot-001",
    generation: int = 1,
) -> dict:
    return build_leaderboard_snapshot_from_rating_snapshot(
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "formula_version": arena.RATING_FORMULA_VERSION,
            "tournament_id": "arena-a",
            "rating_run_id": "elo-a",
            "ratings_ref": "tournaments/curvytron/arena-a/ratings/elo-a/latest.json",
            "context_hash": "ctx-a",
            "roster_hash": "roster-a",
            "created_at": "2026-05-13T00:00:00Z",
            "ratings": [
                {
                    "checkpoint_id": "ckpt-a",
                    "checkpoint_ref": _checkpoint_ref("run-a", 100000),
                    "label": "run-a i100000",
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
        },
        leaderboard_id=leaderboard_id,
        snapshot_id=snapshot_id,
        generation=generation,
    )


def _write_leaderboard_files(
    tmp_path: Path,
    latest_snapshot: dict,
    *,
    immutable_snapshot: dict | None = None,
    write_immutable: bool = True,
) -> tuple[str, str]:
    latest_ref = modal_arena._leaderboard_latest_ref(latest_snapshot["leaderboard_id"])
    snapshot_ref = modal_arena._leaderboard_snapshot_ref(
        latest_snapshot["leaderboard_id"],
        latest_snapshot["snapshot_id"],
    )
    arena.write_json_artifact(tmp_path, latest_ref, latest_snapshot)
    if write_immutable:
        arena.write_json_artifact(
            tmp_path,
            snapshot_ref,
            immutable_snapshot or latest_snapshot,
        )
    return latest_ref.as_posix(), snapshot_ref.as_posix()


class FakeDict:
    def __init__(self, values=None) -> None:
        self.values = dict(values or {})

    def get(self, key, default=None):
        return self.values.get(key, default)

    def put(self, key, value, **_kwargs):
        self.values[key] = value
        return True


def _patch_repair_env(monkeypatch, tmp_path: Path, fake_dict: FakeDict) -> None:
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "opponent_leaderboard_state", fake_dict)


def test_missing_opponent_leaderboard_pointer_is_republished_from_latest(
    tmp_path,
    monkeypatch,
) -> None:
    snapshot = _leaderboard_snapshot()
    latest_ref, snapshot_ref = _write_leaderboard_files(tmp_path, snapshot)
    fake_dict = FakeDict()
    _patch_repair_env(monkeypatch, tmp_path, fake_dict)

    result = modal_arena.curvytron_opponent_leaderboard_pointer_repair.local(
        {"leaderboard_id": "main"}
    )

    pointer = fake_dict.values["current:main"]
    assert result["previous_pointer_status"] == "missing"
    assert result["snapshot_ref"] == snapshot_ref
    assert pointer["snapshot_id"] == "snapshot-001"
    assert pointer["snapshot_ref"] == snapshot_ref
    assert pointer["snapshot_ref"] != latest_ref


def test_stale_opponent_leaderboard_pointer_is_replaced(tmp_path, monkeypatch) -> None:
    snapshot = _leaderboard_snapshot()
    latest_ref, snapshot_ref = _write_leaderboard_files(tmp_path, snapshot)
    stale_pointer = build_leaderboard_pointer(
        snapshot,
        snapshot_ref=latest_ref,
        published_at="2026-05-13T00:00:00Z",
        writer={"kind": "stale-test"},
    )
    fake_dict = FakeDict({"current:main": stale_pointer})
    _patch_repair_env(monkeypatch, tmp_path, fake_dict)

    result = modal_arena.curvytron_opponent_leaderboard_pointer_repair.local(
        {"leaderboard_id": "main"}
    )

    pointer = fake_dict.values["current:main"]
    assert result["previous_pointer_status"] == "stale"
    assert pointer != stale_pointer
    assert pointer["snapshot_id"] == "snapshot-001"
    assert pointer["snapshot_ref"] == snapshot_ref
    assert pointer["snapshot_ref"] != latest_ref


@pytest.mark.parametrize(
    ("case", "error_match"),
    [
        ("missing", "immutable snapshot not found"),
        ("hash_mismatch", "snapshot_sha256 mismatch"),
    ],
)
def test_opponent_leaderboard_pointer_repair_rejects_unproven_latest(
    tmp_path,
    monkeypatch,
    case: str,
    error_match: str,
) -> None:
    latest_snapshot = _leaderboard_snapshot(generation=1)
    immutable_snapshot = (
        _leaderboard_snapshot(generation=2) if case == "hash_mismatch" else None
    )
    _write_leaderboard_files(
        tmp_path,
        latest_snapshot,
        immutable_snapshot=immutable_snapshot,
        write_immutable=case == "hash_mismatch",
    )
    existing = {"current:main": {"sentinel": "unchanged"}}
    fake_dict = FakeDict(existing)
    _patch_repair_env(monkeypatch, tmp_path, fake_dict)

    with pytest.raises(ValueError, match=error_match):
        modal_arena.curvytron_opponent_leaderboard_pointer_repair.local(
            {"leaderboard_id": "main"}
        )

    assert fake_dict.values == existing
