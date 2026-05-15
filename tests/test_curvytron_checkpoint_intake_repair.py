from __future__ import annotations

import pytest

from curvyzero.tournament import curvytron_checkpoint_tournament as arena
from curvyzero.infra.modal import curvyzero_checkpoint_tournament as modal_arena


def _checkpoint_ref(run_id: str, iteration: int) -> str:
    return (
        f"runs/lightzero-curvytron-visual-survival/{run_id}/attempts/attempt-a/"
        f"train/lightzero_exp/ckpt/iteration_{iteration}.pth.tar"
    )


def _queued_manifest() -> dict[str, object]:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": refs},
        rating_defaults={"pairs_per_round": 1, "games_per_pair": 3},
        discovery={"checkpoint_refs": refs},
    )
    return modal_arena._mark_intake_manifest_queued(manifest, refs)


def test_intake_rating_defaults_default_to_all_pairs() -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": refs},
        rating_defaults={"games_per_pair": 3},
        discovery={"checkpoint_refs": refs},
    )

    spec = modal_arena._intake_rating_spec_from_manifest(manifest)

    assert spec["pair_selection"] == arena.RATING_PAIR_SELECTION_ALL_PAIRS
    assert spec["pairs_per_round"] is None


def test_intake_rating_defaults_reject_adaptive_without_pairs_per_round() -> None:
    with pytest.raises(ValueError, match="adaptive_v0 pair selection requires"):
        modal_arena._validate_intake_rating_defaults(
            {"pair_selection": arena.RATING_PAIR_SELECTION_ADAPTIVE_V0},
            tournament_id="arena-a",
            rating_run_id="elo-test",
        )


class _FakeCall:
    object_id = "fc-test"

    def get(self):
        return {"status": "complete", "round_count_completed": 1}


class _FakeRatingLoop:
    def __init__(self) -> None:
        self.specs: list[dict[str, object]] = []

    def spawn(self, spec):
        self.specs.append(dict(spec))
        return _FakeCall()


def test_intake_drain_repairs_missing_queue_events_before_spawn(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _queued_manifest()

    class FakeState:
        def __init__(self) -> None:
            self.values = {manifest["manifest_key"]: manifest}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def put(self, value, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            self.events.append(value)
            return True

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_queue = FakeQueue()
    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", fake_queue)
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: False,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
        }
    )

    assert result["queue_len_before"] == 0
    assert result["queue_len_after_repair"] == 2
    assert result["queue_repair"]["enqueued_count"] == 2
    assert {event["reason"] for event in result["queue_repair"]["events"]} == {
        "repair_missing_queue_events"
    }
    assert result["event_count"] == 2
    assert result["rating_claimed"] is True
    assert result["rating_call_id"] == "fc-test"
    assert fake_queue.events == []
    assert len(fake_rating_loop.specs) == 1


def test_intake_drain_rebuilds_missing_dict_manifest_from_volume(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _queued_manifest()
    modal_arena.runs.write_json(
        tmp_path
        / arena.tournament_intake_manifest_ref("arena-a", "elo-test"),
        manifest,
    )

    class FakeState:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def put(self, value, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            self.events.append(value)
            return True

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_state = FakeState()
    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: False,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
        }
    )

    assert result["manifest_source"] == "volume"
    assert result["manifest_state_repaired"] is True
    assert fake_state.values[manifest["manifest_key"]]["manifest_key"] == manifest[
        "manifest_key"
    ]
    assert manifest["manifest_key"] in fake_state.values[
        modal_arena.CHECKPOINT_INTAKE_ACTIVE_KEYS
    ]
    assert result["queue_repair"]["enqueued_count"] == 2
    assert result["rating_call_id"] == "fc-test"
    assert len(fake_rating_loop.specs) == 1


def test_intake_status_tolerates_missing_volume_manifest(tmp_path, monkeypatch) -> None:
    class FakeState:
        def get(self, key, default=None):
            if key == modal_arena.CHECKPOINT_INTAKE_ACTIVE_KEYS:
                return []
            return default

    class FakeQueue:
        def len(self, **_kwargs):
            raise AssertionError("status must not query Queue without a manifest partition")

    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())

    result = modal_arena.curvytron_checkpoint_intake_status.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
        }
    )

    assert result["manifest_source"] == "missing"
    assert result["queue_len"] is None
    assert result["manifest"] is None


def test_intake_status_repairs_invalid_dict_stub_from_volume(
    tmp_path,
    monkeypatch,
) -> None:
    manifest = _queued_manifest()
    modal_arena.runs.write_json(
        tmp_path
        / arena.tournament_intake_manifest_ref("arena-a", "elo-test"),
        manifest,
    )

    class FakeState:
        def __init__(self) -> None:
            self.values = {manifest["manifest_key"]: {"manifest_key": manifest["manifest_key"]}}

        def get(self, key, default=None):
            if key == modal_arena.CHECKPOINT_INTAKE_ACTIVE_KEYS:
                return []
            return self.values.get(key, default)

        def put(self, key, value):
            self.values[key] = value
            return True

    class FakeQueue:
        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return 0

    fake_state = FakeState()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())

    result = modal_arena.curvytron_checkpoint_intake_status.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
        }
    )

    assert result["manifest_source"] == "volume"
    assert result["manifest_state_repaired"] is True
    assert result["queue_len"] == 0
    assert result["manifest"]["queue_partition"] == manifest["queue_partition"]


def test_intake_drain_reclaims_stale_rating_claim(tmp_path, monkeypatch) -> None:
    manifest = _queued_manifest()
    claim_key = modal_arena._intake_rating_claim_key(manifest)
    refs = list(manifest["queued_checkpoint_refs"])

    class FakeState:
        def __init__(self) -> None:
            self.values = {
                manifest["manifest_key"]: manifest,
                claim_key: {
                    "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                    "created_at": "2000-01-01T00:00:00Z",
                },
            }

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self) -> None:
            self.events = [{"checkpoint_ref": ref} for ref in refs]

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_state = FakeState()
    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: False,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
        }
    )

    assert result["rating_claimed"] is True
    assert result["rating_claim_stale"] is True
    assert result["rating_claim_repaired"] is True
    assert result["event_count"] == 2
    assert result["rating_call_id"] == "fc-test"
    assert fake_state.values[claim_key]["repaired_stale_claim"] is True
    assert len(fake_rating_loop.specs) == 1


def test_rating_round_input_guard_rejects_replaced_input(tmp_path) -> None:
    input_payload = {
        "tournament_id": "arena-a",
        "rating_run_id": "elo-test",
        "round_id": arena.rating_round_id(1),
        "round_index": 1,
        "pool_hash": "pool-a",
        "roster_hash": "pool-a",
        "context_hash": "ctx-a",
        "started_at": "2026-05-14T00:00:00Z",
        "pair_count": 3,
        "game_count": 63,
    }
    ref = arena.rating_round_input_ref("arena-a", "elo-test", arena.rating_round_id(1))
    modal_arena.runs.write_json(
        tmp_path / ref,
        {
            **input_payload,
            "pool_hash": "pool-b",
            "pair_count": 6,
            "game_count": 126,
        },
    )

    with pytest.raises(RuntimeError, match="input was replaced"):
        modal_arena._assert_rating_round_input_still_matches(tmp_path, input_payload)


def test_cli_intake_drain_override_filter_drops_defaults() -> None:
    payload = {
        "allow_rating_overrides": True,
        "continue_from_latest": False,
        "games_per_pair": arena.DEFAULT_GAMES_PER_PAIR,
        "decision_source_frames": None,
        "pair_selection": arena.DEFAULT_RATING_PAIR_SELECTION,
        "active_pool_limit": 10,
    }

    cleaned = modal_arena._drop_default_intake_drain_rating_overrides(payload)

    assert cleaned["allow_rating_overrides"] is True
    assert cleaned["active_pool_limit"] == 10
    assert "games_per_pair" not in cleaned
    assert "decision_source_frames" not in cleaned
    assert "pair_selection" not in cleaned
    assert "continue_from_latest" not in cleaned


def test_cli_intake_drain_filter_drops_default_continue_without_override_mode() -> None:
    cleaned = modal_arena._drop_default_intake_drain_rating_overrides(
        {
            "allow_rating_overrides": False,
            "continue_from_latest": False,
            "spawn_rating": True,
        }
    )

    assert cleaned["allow_rating_overrides"] is False
    assert cleaned["spawn_rating"] is True
    assert "continue_from_latest" not in cleaned


def test_expanded_pool_claim_is_not_blocked_by_fresh_partial_pool_claim(
    tmp_path,
    monkeypatch,
) -> None:
    old_manifest = _queued_manifest()
    old_claim_key = modal_arena._intake_rating_claim_key(
        old_manifest,
        continue_from_latest=True,
    )
    old_refs = list(old_manifest["seen_checkpoint_refs"])
    new_ref = _checkpoint_ref("run-c", 30)
    expanded_manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": [*old_refs, new_ref]},
        rating_defaults={
            "continue_from_latest": True,
            "pairs_per_round": 1,
            "games_per_pair": 3,
        },
        discovery={"checkpoint_refs": [*old_refs, new_ref]},
        existing=old_manifest,
    )
    expanded_manifest = modal_arena._mark_intake_manifest_queued(
        expanded_manifest,
        [new_ref],
    )
    new_claim_key = modal_arena._intake_rating_claim_key(
        expanded_manifest,
        continue_from_latest=True,
    )
    assert new_claim_key != old_claim_key

    class FakeState:
        def __init__(self) -> None:
            self.values = {
                expanded_manifest["manifest_key"]: expanded_manifest,
                old_claim_key: {
                    "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                    "created_at": "2999-01-01T00:00:00Z",
                    "checkpoint_count": len(old_manifest["checkpoint_refs"]),
                    "pool_hash": modal_arena._checkpoint_ref_pool_hash(
                        old_manifest["checkpoint_refs"]
                    ),
                },
            }

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self) -> None:
            self.events = [{"checkpoint_ref": new_ref}]

        def len(self, *, partition):
            assert partition == expanded_manifest["queue_partition"]
            return len(self.events)

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == expanded_manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_state = FakeState()
    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: True,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
            "continue_from_latest": True,
        }
    )

    assert result["rating_claimed"] is True
    assert result["rating_claim_key"] == new_claim_key
    assert result["rating_claim_stale"] is False
    assert result["rating_claim_repaired"] is False
    assert result["spawn_skipped_reason"] == ""
    assert fake_state.values[old_claim_key]["created_at"] == "2999-01-01T00:00:00Z"
    assert fake_state.values[new_claim_key]["pool_hash"] == modal_arena._checkpoint_ref_pool_hash(
        expanded_manifest["seen_checkpoint_refs"]
    )
    assert len(fake_rating_loop.specs) == 1


def test_expanded_pool_claim_ignores_completed_partial_pool_claim(
    tmp_path,
    monkeypatch,
) -> None:
    old_manifest = _queued_manifest()
    claim_key = modal_arena._intake_rating_claim_key(
        old_manifest,
        continue_from_latest=True,
    )
    old_refs = list(old_manifest["seen_checkpoint_refs"])
    new_ref = _checkpoint_ref("run-c", 30)
    expanded_manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": [*old_refs, new_ref]},
        rating_defaults={
            "continue_from_latest": True,
            "pairs_per_round": 1,
            "games_per_pair": 3,
        },
        discovery={"checkpoint_refs": [*old_refs, new_ref]},
        existing=old_manifest,
    )
    expanded_manifest = modal_arena._mark_intake_manifest_queued(
        expanded_manifest,
        [new_ref],
    )
    new_claim_key = modal_arena._intake_rating_claim_key(
        expanded_manifest,
        continue_from_latest=True,
    )
    assert new_claim_key != claim_key
    modal_arena.runs.write_json(
        tmp_path / arena.rating_progress_ref("arena-a", "elo-test"),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "status": "complete",
            "phase": "ratings_written",
            "ratings_written": True,
        },
    )

    class FakeState:
        def __init__(self) -> None:
            self.values = {
                expanded_manifest["manifest_key"]: expanded_manifest,
                claim_key: {
                    "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                    "created_at": "2999-01-01T00:00:00Z",
                    "checkpoint_count": len(old_manifest["checkpoint_refs"]),
                    "pool_hash": modal_arena._checkpoint_ref_pool_hash(
                        old_manifest["checkpoint_refs"]
                    ),
                },
            }

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self) -> None:
            self.events = [{"checkpoint_ref": new_ref}]

        def len(self, *, partition):
            assert partition == expanded_manifest["queue_partition"]
            return len(self.events)

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == expanded_manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_state = FakeState()
    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: True,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
            "continue_from_latest": True,
        }
    )

    assert result["rating_claimed"] is True
    assert result["rating_claim_key"] == new_claim_key
    assert result["rating_claim_stale"] is False
    assert result["rating_claim_repaired"] is False
    assert result["spawn_skipped_reason"] == ""
    assert fake_state.values[claim_key]["created_at"] == "2999-01-01T00:00:00Z"
    assert fake_state.values[new_claim_key]["repaired_stale_claim"] is False
    assert len(fake_rating_loop.specs) == 1


def test_intake_drain_can_wait_for_spawned_rating_result(tmp_path, monkeypatch) -> None:
    manifest = _queued_manifest()

    class FakeState:
        def __init__(self) -> None:
            self.values = {manifest["manifest_key"]: manifest}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self) -> None:
            self.events = [{"checkpoint_ref": ref} for ref in manifest["queued_checkpoint_refs"]]

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: False,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
            "wait_for_rating": True,
        }
    )

    assert result["rating_call_id"] == "fc-test"
    assert result["rating_result"] == {"status": "complete", "round_count_completed": 1}
    assert len(fake_rating_loop.specs) == 1


def test_intake_drain_keeps_fresh_rating_claim_blocking(tmp_path, monkeypatch) -> None:
    manifest = _queued_manifest()
    claim_key = modal_arena._intake_rating_claim_key(manifest)

    class FakeState:
        def __init__(self) -> None:
            self.values = {
                manifest["manifest_key"]: manifest,
                claim_key: {
                    "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                    "created_at": "2999-01-01T00:00:00Z",
                    "checkpoint_count": len(manifest["checkpoint_refs"]),
                    "pool_hash": modal_arena._checkpoint_ref_pool_hash(
                        manifest["checkpoint_refs"]
                    ),
                },
            }

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, *_args, **_kwargs):
            raise AssertionError("fresh claims must not be overwritten")

    class FakeQueue:
        def __init__(self) -> None:
            self.get_many_calls = 0

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return 2

        def get_many(self, *_args, **_kwargs):
            self.get_many_calls += 1
            return [{"checkpoint_ref": ref} for ref in manifest["queued_checkpoint_refs"]]

    fake_queue = FakeQueue()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", fake_queue)
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", _FakeRatingLoop())
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: False,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
        }
    )

    assert result["event_count"] == 0
    assert result["rating_claimed"] is False
    assert result["rating_claim_stale"] is False
    assert result["spawn_skipped_reason"] == "rating_run_claim_exists"
    assert fake_queue.get_many_calls == 0


def test_live_watch_drain_continues_past_partial_existing_rating_and_old_claim(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-a", 20)]
    new_refs = [
        _checkpoint_ref("run-a", 30),
        _checkpoint_ref("run-b", 10),
        _checkpoint_ref("run-b", 20),
    ]
    all_refs = old_refs + new_refs
    scan_spec = {"run_ids": "run-a,run-b", "checkpoint_selection": "all"}
    early_manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec=scan_spec,
        rating_defaults={
            "continue_from_latest": False,
            "pairs_per_round": 10,
            "games_per_pair": 3,
        },
        discovery={"checkpoint_refs": old_refs},
    )
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec=scan_spec,
        rating_defaults=early_manifest["rating_defaults"],
        discovery={"checkpoint_refs": all_refs},
        existing=early_manifest,
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, new_refs)
    old_underdescribed_claim_key = modal_arena._intake_rating_claim_key(
        manifest,
        continue_from_latest=False,
    )

    class FakeState:
        def __init__(self) -> None:
            self.values = {
                manifest["manifest_key"]: manifest,
                old_underdescribed_claim_key: {
                    "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                    "created_at": "2999-01-01T00:00:00Z",
                    "event_count": len(old_refs),
                },
            }

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def put(self, value, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            self.events.append(value)
            return True

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_state = FakeState()
    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: True,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "max_events": 10,
            "spawn_rating": True,
        }
    )

    spawned_refs = [
        str(checkpoint["checkpoint_ref"])
        for checkpoint in fake_rating_loop.specs[0]["checkpoints"]
    ]
    assert result["existing_rating_run"] is True
    assert result["continue_from_latest"] is True
    assert result["requested_continue_from_latest"] is False
    assert result["continuation_reason"] == "live_watch"
    assert result["queue_len_before"] == 0
    assert result["queue_repair"]["enqueued_count"] == len(new_refs)
    assert result["event_count"] == len(new_refs)
    assert result["rating_call_id"] == "fc-test"
    assert result["rating_claim_key"] != old_underdescribed_claim_key
    assert ":mode-continue:" in result["rating_claim_key"]
    assert result["rating_claim_key"].rsplit(":", 1)[-1].startswith("pool-")
    assert result["rating_claimed"] is True
    assert set(spawned_refs) == set(all_refs)


def test_intake_drain_continues_explicit_manifest_when_queue_was_lost(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    new_refs = [_checkpoint_ref("run-a", 20)]
    all_refs = old_refs + new_refs
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec={"checkpoint_refs": all_refs},
        rating_defaults={
            "continue_from_latest": True,
            "games_per_pair": 3,
        },
        discovery={"checkpoint_refs": all_refs},
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, all_refs)
    modal_arena.runs.write_json(
        tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id),
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(0),
            "round_index": 0,
            "ratings": [
                {
                    "checkpoint_id": f"old-{index}",
                    "checkpoint_ref": ref,
                    "rank": index + 1,
                    "rating": 1500.0,
                    "status": "active",
                }
                for index, ref in enumerate(old_refs)
            ],
        },
    )

    class FakeState:
        def __init__(self) -> None:
            self.values = {manifest["manifest_key"]: manifest}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = []

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def put(self, value, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            self.events.append(value)
            return True

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: True,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "max_events": 10,
            "spawn_rating": True,
        }
    )

    spawned_refs = [
        str(checkpoint["checkpoint_ref"])
        for checkpoint in fake_rating_loop.specs[0]["checkpoints"]
    ]
    assert result["existing_rating_run"] is True
    assert result["continue_from_latest"] is True
    assert result["latest_rating_checkpoint_count"] == len(old_refs)
    assert result["desired_pool_new_checkpoint_count"] == len(new_refs)
    assert result["queue_repair"]["enqueued_count"] == len(new_refs)
    assert result["event_count"] == len(new_refs)
    assert result["rating_call_id"] == "fc-test"
    assert set(spawned_refs) == set(all_refs)


def test_intake_drain_continue_from_latest_preserves_retired_history_in_rating_spec(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    active_refs = [_checkpoint_ref(f"active-{index:03d}", index) for index in range(100)]
    retired_refs = [_checkpoint_ref(f"retired-{index:03d}", 1000 + index) for index in range(2)]
    new_ref = _checkpoint_ref("new-run", 2000)
    all_refs = active_refs + retired_refs + [new_ref]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec={"checkpoint_refs": all_refs},
        rating_defaults={
            "continue_from_latest": True,
            "pairs_per_round": 10,
            "games_per_pair": 3,
        },
        discovery={"checkpoint_refs": all_refs},
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, [new_ref])
    latest_path = tmp_path / arena.rating_latest_ref(tournament_id, rating_run_id)
    latest_path.parent.mkdir(parents=True)
    modal_arena.runs.write_json(
        latest_path,
        {
            "schema_id": arena.RATING_SNAPSHOT_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(3),
            "round_index": 3,
            "ratings": [
                {
                    "checkpoint_id": f"active-{index:03d}",
                    "checkpoint_ref": ref,
                    "rank": index + 1,
                    "rating": 2000.0 - index,
                    "status": "active",
                }
                for index, ref in enumerate(active_refs)
            ]
            + [
                {
                    "checkpoint_id": f"retired-{index:03d}",
                    "checkpoint_ref": ref,
                    "rank": 101 + index,
                    "rating": 1000.0 - index,
                    "status": "retired",
                    "retired_reason": "below_active_pool_limit",
                }
                for index, ref in enumerate(retired_refs)
            ],
        },
    )

    class FakeState:
        def __init__(self) -> None:
            self.values = {manifest["manifest_key"]: manifest}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value, skip_if_exists=False):
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    class FakeQueue:
        def __init__(self) -> None:
            self.events = [{"checkpoint_ref": new_ref, "reason": "tick"}]

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", FakeQueue())
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "max_events": 10,
            "spawn_rating": True,
            "continue_from_latest": True,
        }
    )

    spawned_refs = [
        str(checkpoint["checkpoint_ref"])
        for checkpoint in fake_rating_loop.specs[0]["checkpoints"]
    ]
    assert result["rating_call_id"] == "fc-test"
    assert result["continue_from_latest"] is True
    assert set(spawned_refs) == set(all_refs)
