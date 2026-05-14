from __future__ import annotations

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


class _FakeCall:
    object_id = "fc-test"


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
