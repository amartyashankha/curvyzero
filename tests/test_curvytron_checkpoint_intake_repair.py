from __future__ import annotations

import os
import time

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


def test_live_intake_rating_defaults_are_bounded_adaptive() -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": "run-a,run-b", "checkpoint_selection": "all"},
        rating_defaults={
            "continue_from_latest": True,
            "games_per_pair": 3,
        },
        discovery={"checkpoint_refs": refs},
    )

    spec = modal_arena._intake_rating_spec_from_manifest(manifest)

    assert spec["continue_from_latest"] is True
    assert spec["pair_selection"] == arena.RATING_PAIR_SELECTION_ADAPTIVE_V0
    assert spec["pairs_per_round"] == arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND
    assert spec["active_pool_limit"] == arena.DEFAULT_RATING_ACTIVE_POOL_LIMIT
    assert spec["save_gif"] is True
    assert spec["gif_sample_games_per_pair"] == arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR


def test_live_intake_continue_from_latest_uses_current_manifest_pool() -> None:
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    new_refs = [*old_refs, _checkpoint_ref("run-c", 30)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_id_prefix": "run-", "checkpoint_selection": "all"},
        rating_defaults={"continue_from_latest": True, "games_per_pair": 3},
        discovery={"checkpoint_refs": new_refs},
    )
    manifest["seen_checkpoint_refs"] = list(old_refs)

    refs = modal_arena._intake_manifest_rating_checkpoint_refs(
        manifest,
        continue_from_latest=True,
    )

    assert refs == new_refs


def test_live_intake_legacy_all_pairs_default_is_capped() -> None:
    refs = [_checkpoint_ref(f"run-{index:03d}", index) for index in range(261)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": ",".join(f"run-{index:03d}" for index in range(18))},
        rating_defaults={
            "continue_from_latest": True,
            "games_per_pair": 3,
            "pair_selection": arena.RATING_PAIR_SELECTION_ALL_PAIRS,
        },
        discovery={"checkpoint_refs": refs},
    )

    spec = modal_arena._intake_rating_spec_from_manifest(manifest)
    pair_specs = arena.build_rating_round_pair_specs(spec)

    assert spec["pair_selection"] == arena.RATING_PAIR_SELECTION_ADAPTIVE_V0
    assert spec["pairs_per_round"] == arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND
    assert len(pair_specs) == arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND
    assert spec["save_gif"] is True
    assert spec["gif_sample_games_per_pair"] == arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR


def test_live_intake_legacy_gifs_off_is_repaired() -> None:
    refs = [_checkpoint_ref(f"run-{index:03d}", index) for index in range(12)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": ",".join(f"run-{index:03d}" for index in range(12))},
        rating_defaults={
            "continue_from_latest": True,
            "games_per_pair": 21,
            "pair_selection": arena.RATING_PAIR_SELECTION_ADAPTIVE_V0,
            "pairs_per_round": 6,
            "save_gif": False,
            "gif_sample_games_per_pair": 0,
        },
        discovery={"checkpoint_refs": refs},
    )

    spec = modal_arena._intake_rating_spec_from_manifest(manifest)
    repaired = modal_arena._repair_live_intake_rating_defaults(manifest)

    assert spec["save_gif"] is True
    assert spec["gif_sample_games_per_pair"] == arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR
    assert repaired["rating_defaults"]["save_gif"] is True
    assert repaired["rating_defaults"]["gif_sample_games_per_pair"] == (
        arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR
    )
    assert repaired["rating_defaults"]["live_gif_repaired_to_enabled"] is True


def test_live_intake_put_repairs_gifs_off_before_persisting(monkeypatch) -> None:
    refs = [_checkpoint_ref(f"run-{index:03d}", index) for index in range(12)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": ",".join(f"run-{index:03d}" for index in range(12))},
        rating_defaults={
            "continue_from_latest": True,
            "games_per_pair": 21,
            "pair_selection": arena.RATING_PAIR_SELECTION_ADAPTIVE_V0,
            "pairs_per_round": 6,
            "save_gif": False,
            "gif_sample_games_per_pair": 0,
        },
        discovery={"checkpoint_refs": refs},
    )

    class FakeState:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value):
            self.values[key] = value
            return True

    fake_state = FakeState()
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)

    saved = modal_arena._put_intake_manifest(manifest)

    assert saved["manifest"]["rating_defaults"]["save_gif"] is True
    assert saved["manifest"]["rating_defaults"]["gif_sample_games_per_pair"] == (
        arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR
    )
    stored = fake_state.values[manifest["manifest_key"]]
    assert stored["rating_defaults"]["save_gif"] is True
    assert stored["rating_defaults"]["gif_sample_games_per_pair"] == (
        arena.DEFAULT_GIF_SAMPLE_GAMES_PER_PAIR
    )


def test_live_intake_legacy_all_pairs_exact_active_pool_is_capped() -> None:
    refs = [_checkpoint_ref(f"run-{index:03d}", index) for index in range(100)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": ",".join(f"run-{index:03d}" for index in range(18))},
        rating_defaults={
            "continue_from_latest": True,
            "games_per_pair": 3,
            "pair_selection": arena.RATING_PAIR_SELECTION_ALL_PAIRS,
            "pairs_per_round": 4950,
            "active_pool_limit": 100,
        },
        discovery={"checkpoint_refs": refs},
    )

    spec = modal_arena._intake_rating_spec_from_manifest(manifest)
    pair_specs = arena.build_rating_round_pair_specs(spec)

    assert spec["pair_selection"] == arena.RATING_PAIR_SELECTION_ADAPTIVE_V0
    assert spec["pairs_per_round"] == arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND
    assert len(pair_specs) == arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND


def test_live_intake_manifest_merge_repairs_large_all_pairs_state() -> None:
    refs = [_checkpoint_ref(f"run-{index:03d}", index) for index in range(261)]
    current = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": "run-a,run-b", "checkpoint_selection": "all"},
        rating_defaults={
            "continue_from_latest": True,
            "games_per_pair": 3,
            "pair_selection": arena.RATING_PAIR_SELECTION_ALL_PAIRS,
            "pairs_per_round": 4950,
            "active_pool_limit": 100,
        },
        discovery={"checkpoint_refs": refs},
    )
    candidate = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": "run-a,run-b", "checkpoint_selection": "all"},
        rating_defaults=current["rating_defaults"],
        discovery={"checkpoint_refs": refs},
        existing=current,
    )

    merged = modal_arena._repair_live_intake_rating_defaults(
        modal_arena._intake_manifest_with_merged_pool(candidate, current)
    )

    assert merged["rating_defaults"]["pair_selection"] == arena.RATING_PAIR_SELECTION_ADAPTIVE_V0
    assert merged["rating_defaults"]["pairs_per_round"] == (
        arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND
    )
    assert merged["rating_defaults"]["live_all_pairs_repaired_to_bounded"] is True


def test_live_intake_volume_repair_returns_repaired_manifest(tmp_path, monkeypatch) -> None:
    refs = [_checkpoint_ref(f"run-{index:03d}", index) for index in range(261)]
    stale_manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": "run-a,run-b", "checkpoint_selection": "all"},
        rating_defaults={
            "continue_from_latest": True,
            "games_per_pair": 3,
            "pair_selection": arena.RATING_PAIR_SELECTION_ALL_PAIRS,
            "pairs_per_round": 4950,
            "active_pool_limit": 100,
        },
        discovery={"checkpoint_refs": refs},
    )
    modal_arena.runs.write_json(
        tmp_path / arena.tournament_intake_manifest_ref("arena-a", "elo-test"),
        stale_manifest,
    )

    class FakeState:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value):
            self.values[key] = value
            return True

    fake_state = FakeState()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)

    loaded, load_info = modal_arena._load_intake_manifest(
        "arena-a",
        "elo-test",
        repair_state=True,
    )

    assert load_info["manifest_source"] == "volume"
    assert load_info["manifest_state_repaired"] is True
    assert loaded["rating_defaults"]["pair_selection"] == arena.RATING_PAIR_SELECTION_ADAPTIVE_V0
    assert loaded["rating_defaults"]["pairs_per_round"] == (
        arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND
    )
    saved = fake_state.values[stale_manifest["manifest_key"]]
    assert saved["rating_defaults"]["pair_selection"] == arena.RATING_PAIR_SELECTION_ADAPTIVE_V0


def test_live_intake_dict_repair_returns_repaired_manifest(monkeypatch) -> None:
    refs = [_checkpoint_ref(f"run-{index:03d}", index) for index in range(261)]
    stale_manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"run_ids": "run-a,run-b", "checkpoint_selection": "all"},
        rating_defaults={
            "continue_from_latest": True,
            "games_per_pair": 3,
            "pair_selection": arena.RATING_PAIR_SELECTION_ALL_PAIRS,
            "pairs_per_round": 4950,
            "active_pool_limit": 100,
        },
        discovery={"checkpoint_refs": refs},
    )

    class FakeState:
        def __init__(self) -> None:
            self.values: dict[str, object] = {stale_manifest["manifest_key"]: stale_manifest}

        def get(self, key, default=None):
            return self.values.get(key, default)

        def put(self, key, value):
            self.values[key] = value
            return True

    fake_state = FakeState()
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", fake_state)

    loaded, load_info = modal_arena._load_intake_manifest(
        "arena-a",
        "elo-test",
        repair_state=True,
    )

    assert load_info["manifest_source"] == "dict"
    assert load_info["manifest_state_repaired"] is True
    assert loaded["rating_defaults"]["pair_selection"] == arena.RATING_PAIR_SELECTION_ADAPTIVE_V0
    assert loaded["rating_defaults"]["pairs_per_round"] == (
        arena.DEFAULT_LIVE_INTAKE_PAIRS_PER_ROUND
    )
    saved = fake_state.values[stale_manifest["manifest_key"]]
    assert saved["rating_defaults"]["pair_selection"] == arena.RATING_PAIR_SELECTION_ADAPTIVE_V0


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

    class FakeReduceCall:
        object_id = "fc-reduce"

    class FakeRatingReduce:
        def __init__(self) -> None:
            self.specs: list[dict[str, object]] = []

        def spawn(self, spec):
            self.specs.append(dict(spec))
            return FakeReduceCall()

    fake_rating_reduce = FakeRatingReduce()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", fake_queue)
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "curvytron_rating_reduce", fake_rating_reduce)
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
    assert new_claim_key == old_claim_key

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
    monkeypatch.setattr(
        modal_arena,
        "_rating_writer_has_finished",
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
    assert result["rating_claim_stale"] is True
    assert result["rating_claim_repaired"] is True
    assert result["spawn_skipped_reason"] == ""
    assert fake_state.values[new_claim_key]["replaces_stale_claim_created_at"] == (
        "2999-01-01T00:00:00Z"
    )
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
    assert new_claim_key == claim_key
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
    monkeypatch.setattr(
        modal_arena,
        "_rating_writer_has_finished",
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
    assert result["rating_claim_stale"] is True
    assert result["rating_claim_repaired"] is True
    assert result["spawn_skipped_reason"] == ""
    assert fake_state.values[new_claim_key]["replaces_stale_claim_created_at"] == (
        "2999-01-01T00:00:00Z"
    )
    assert fake_state.values[new_claim_key]["repaired_stale_claim"] is True
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


def test_intake_drain_spawn_if_existing_does_not_force_empty_same_pool_continuation(
    tmp_path,
    monkeypatch,
) -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": refs},
        rating_defaults={"continue_from_latest": True, "games_per_pair": 3},
        discovery={"checkpoint_refs": refs},
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, refs)
    claim_key = modal_arena._intake_rating_claim_key(
        manifest,
        continue_from_latest=True,
    )

    class FakeState:
        def __init__(self) -> None:
            self.values = {
                manifest["manifest_key"]: manifest,
                claim_key: {
                    "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                    "created_at": "2999-01-01T00:00:00Z",
                    "checkpoint_count": len(refs),
                    "pool_hash": modal_arena._checkpoint_ref_pool_hash(refs),
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
        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return 0

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            return []

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
    monkeypatch.setattr(
        modal_arena,
        "_rating_writer_has_finished",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        modal_arena,
        "_rating_latest_checkpoint_refs",
        lambda *_args, **_kwargs: list(refs),
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
            "continue_from_latest": True,
            "spawn_if_existing": True,
            "wait_for_rating": True,
        }
    )

    assert result["event_count"] == 0
    assert result["rating_claimed"] is False
    assert result["rating_claim_stale"] is False
    assert result["rating_claim_repaired"] is False
    assert result["rating_call_id"] == ""
    assert result["spawn_skipped_reason"] == "rating_run_claim_exists"
    assert len(fake_rating_loop.specs) == 0


def test_intake_drain_spawn_if_empty_allows_explicit_same_pool_continuation(
    tmp_path,
    monkeypatch,
) -> None:
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 20)]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id="arena-a",
        rating_run_id="elo-test",
        scan_spec={"checkpoint_refs": refs},
        rating_defaults={"continue_from_latest": True, "games_per_pair": 3},
        discovery={"checkpoint_refs": refs},
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, refs)
    claim_key = modal_arena._intake_rating_claim_key(
        manifest,
        continue_from_latest=True,
    )

    class FakeState:
        def __init__(self) -> None:
            self.values = {
                manifest["manifest_key"]: manifest,
                claim_key: {
                    "schema_id": "curvyzero_curvytron_checkpoint_intake_rating_claim/v0",
                    "created_at": "2999-01-01T00:00:00Z",
                    "checkpoint_count": len(refs),
                    "pool_hash": modal_arena._checkpoint_ref_pool_hash(refs),
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
        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return 0

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            return []

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
    monkeypatch.setattr(
        modal_arena,
        "_rating_writer_has_finished",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        modal_arena,
        "_rating_latest_checkpoint_refs",
        lambda *_args, **_kwargs: list(refs),
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": "arena-a",
            "rating_run_id": "elo-test",
            "max_events": 10,
            "spawn_rating": True,
            "continue_from_latest": True,
            "spawn_if_existing": True,
            "spawn_if_empty": True,
            "wait_for_rating": True,
        }
    )

    assert result["event_count"] == 0
    assert result["rating_claimed"] is True
    assert result["rating_claim_stale"] is True
    assert result["rating_claim_repaired"] is True
    assert result["rating_call_id"] == "fc-test"
    assert result["spawn_skipped_reason"] == ""
    assert len(fake_rating_loop.specs) == 1


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
    monkeypatch.setattr(
        modal_arena,
        "_rating_writer_has_finished",
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
    assert result["rating_claim_key"].endswith(":active")
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


def test_intake_drain_waits_for_unfinished_continuation_round(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    new_ref = _checkpoint_ref("run-a", 20)
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec={
            "run_ids": "run-a,run-b",
            "checkpoint_selection": arena.CHECKPOINT_SELECTION_ALL,
        },
        rating_defaults={"continue_from_latest": True, "games_per_pair": 3},
        discovery={"checkpoint_refs": [*old_refs, new_ref]},
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, [new_ref])
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
    modal_arena.runs.write_json(
        tmp_path
        / arena.rating_round_input_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(1),
        ),
        {"round_index": 1},
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
            self.events = [{"checkpoint_ref": new_ref}]

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    fake_queue = FakeQueue()
    fake_rating_loop = _FakeRatingLoop()

    class FakeReduceCall:
        object_id = "fc-reduce"

    class FakeRatingReduce:
        def __init__(self) -> None:
            self.specs: list[dict[str, object]] = []

        def spawn(self, spec):
            self.specs.append(dict(spec))
            return FakeReduceCall()

    fake_rating_reduce = FakeRatingReduce()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", fake_queue)
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "curvytron_rating_reduce", fake_rating_reduce)
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

    assert result["existing_rating_run"] is True
    assert result["continue_from_latest"] is True
    assert result["rating_writer_finished"] is False
    assert result["spawn_skipped_reason"] == "rating_writer_not_finished_round_running"
    assert result["event_count"] == 0
    assert result["rating_call_id"] == ""
    assert result["rating_recovery_claimed"] is False
    assert result["rating_recovery_reduce_ready"] is False
    assert result["rating_recovery_round"]["round_id"] == arena.rating_round_id(1)
    assert fake_rating_reduce.specs == []
    assert fake_queue.events == [{"checkpoint_ref": new_ref}]
    assert fake_rating_loop.specs == []


def test_intake_drain_skips_stale_zero_output_round_and_spawns_fresh_loop(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    new_ref = _checkpoint_ref("run-a", 20)
    all_refs = [*old_refs, new_ref]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec={
            "run_ids": "run-a,run-b",
            "checkpoint_selection": arena.CHECKPOINT_SELECTION_ALL,
        },
        rating_defaults={"continue_from_latest": True, "games_per_pair": 3},
        discovery={"checkpoint_refs": all_refs},
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, [new_ref])
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
    modal_arena.runs.write_json(
        tmp_path
        / arena.rating_round_input_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(1),
        ),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(1),
            "round_index": 1,
            "checkpoint_roster": [
                {"checkpoint_id": f"old-{index}", "checkpoint_ref": ref}
                for index, ref in enumerate(old_refs)
            ],
            "pair_count": 1,
            "game_count": 3,
            "pair_specs": [],
            "rating_spec": {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "checkpoints": old_refs,
                "games_per_pair": 3,
            },
        },
    )
    modal_arena.runs.write_json(
        tmp_path
        / arena.rating_round_progress_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(1),
        ),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(1),
            "round_index": 1,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": 1,
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 0,
            "started_pair_count": 1,
        },
    )
    old_mtime = time.time() - 900.0
    os.utime(
        tmp_path
        / arena.rating_round_input_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(1),
        ),
        (old_mtime, old_mtime),
    )
    os.utime(
        tmp_path
        / arena.rating_round_progress_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(1),
        ),
        (old_mtime, old_mtime),
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
            self.events = [{"checkpoint_ref": new_ref}]

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    class FakeReduceCall:
        object_id = "fc-reduce"

    class FakeRatingReduce:
        def __init__(self) -> None:
            self.specs: list[dict[str, object]] = []

        def spawn(self, spec):
            self.specs.append(dict(spec))
            return FakeReduceCall()

    fake_queue = FakeQueue()
    fake_rating_loop = _FakeRatingLoop()
    fake_rating_reduce = FakeRatingReduce()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", fake_queue)
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "curvytron_rating_reduce", fake_rating_reduce)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: True,
    )
    original_skip_decision = modal_arena._rating_round_skip_decision
    skip_decision_calls: list[dict[str, object]] = []

    def wrapped_skip_decision(*args, **kwargs):
        skip_decision_calls.append(dict(kwargs))
        return original_skip_decision(*args, **kwargs)

    monkeypatch.setattr(
        modal_arena,
        "_rating_round_skip_decision",
        wrapped_skip_decision,
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "max_events": 10,
            "spawn_rating": True,
            "rating_round_stale_after_seconds": 600,
        }
    )
    skipped_progress = modal_arena._read_json(
        tmp_path
        / arena.rating_round_progress_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(1),
        )
    )

    assert result["rating_recovery_skip_decision"]["skip"] is True
    assert (
        result["rating_recovery_skip_decision"]["reason"]
        == "different_spec_zero_output"
    )
    assert result["rating_recovery_skipped_progress"]["status"] == "skipped"
    assert result["rating_writer_finished"] is True
    assert result["claim_stale_after_seconds"] == (
        modal_arena.DEFAULT_CHECKPOINT_INTAKE_CLAIM_STALE_SECONDS
    )
    assert result["rating_round_stale_after_seconds"] == 600
    assert skip_decision_calls
    assert skip_decision_calls[0]["scan_output_progress"] is True
    assert result["spawn_skipped_reason"] == ""
    assert result["event_count"] == 1
    assert fake_queue.events == []
    assert len(fake_rating_loop.specs) == 1
    assert fake_rating_reduce.specs == []
    assert skipped_progress["status"] == "skipped"


def test_intake_drain_spawns_partial_reduce_for_old_incomplete_round(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    new_ref = _checkpoint_ref("run-c", 20)
    all_refs = [*old_refs, new_ref]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec={
            "run_ids": "run-a,run-b,run-c",
            "checkpoint_selection": arena.CHECKPOINT_SELECTION_ALL,
        },
        rating_defaults={"continue_from_latest": True, "games_per_pair": 3},
        discovery={"checkpoint_refs": all_refs},
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, [new_ref])
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
    modal_arena.runs.write_json(
        tmp_path
        / arena.rating_round_input_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(1),
        ),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(1),
            "round_index": 1,
            "checkpoint_roster": [
                {"checkpoint_id": f"ckpt-{index}", "checkpoint_ref": ref}
                for index, ref in enumerate(all_refs)
            ],
            "pair_count": 300,
            "game_count": 6300,
            "pair_specs": [],
            "rating_spec": {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "checkpoints": all_refs,
                "games_per_pair": 3,
            },
        },
    )
    modal_arena.runs.write_json(
        tmp_path
        / arena.rating_round_progress_ref(
            tournament_id,
            rating_run_id,
            arena.rating_round_id(1),
        ),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(1),
            "round_index": 1,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": 300,
            "game_count": 6300,
            "completed_pair_count": 2,
            "completed_game_count": 42,
            "started_pair_count": 2,
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
            self.events = [{"checkpoint_ref": new_ref}]

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

        def get_many(self, max_events, **kwargs):
            assert kwargs["partition"] == manifest["queue_partition"]
            drained = self.events[:max_events]
            del self.events[:max_events]
            return drained

    class FakeReduceCall:
        object_id = "fc-partial-reduce"

    class FakeRatingReduce:
        def __init__(self) -> None:
            self.specs: list[dict[str, object]] = []

        def spawn(self, spec):
            self.specs.append(dict(spec))
            return FakeReduceCall()

    fake_queue = FakeQueue()
    fake_rating_reduce = FakeRatingReduce()
    fake_rating_loop = _FakeRatingLoop()
    monkeypatch.setattr(modal_arena, "TOURNAMENT_MOUNT", tmp_path)
    monkeypatch.setattr(modal_arena, "checkpoint_intake_state", FakeState())
    monkeypatch.setattr(modal_arena, "checkpoint_intake_queue", fake_queue)
    monkeypatch.setattr(modal_arena, "curvytron_rating_loop", fake_rating_loop)
    monkeypatch.setattr(modal_arena, "curvytron_rating_reduce", fake_rating_reduce)
    monkeypatch.setattr(modal_arena, "_reload_volume", lambda _volume: None)
    monkeypatch.setattr(modal_arena, "_commit_volume", lambda _volume: None)
    monkeypatch.setattr(
        modal_arena,
        "_rating_run_has_existing_output",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        modal_arena,
        "_rating_round_skip_decision",
        lambda *_args, **_kwargs: {
            "skip": False,
            "reason": "not_skippable",
            "game_count": 6300,
            "completed_game_count": 42,
            "started_pair_count": 2,
            "partial_reduce_recommended": True,
        },
    )

    result = modal_arena.curvytron_checkpoint_intake_drain.local(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "max_events": 10,
            "spawn_rating": True,
            "rating_round_stale_after_seconds": 600,
        }
    )

    assert result["rating_recovery_partial_reduce_recommended"] is True
    assert result["rating_recovery_claimed"] is True
    assert result["rating_call_id"] == "fc-partial-reduce"
    assert result["spawn_skipped_reason"] == "spawned_unfinished_round_partial_reduce"
    assert result["event_count"] == 0
    assert fake_rating_reduce.specs == [
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_index": 1,
            "allow_partial": True,
        }
    ]
    assert fake_rating_loop.specs == []
    assert fake_queue.events == [{"checkpoint_ref": new_ref}]


def test_fresh_zero_output_smaller_pool_round_is_not_skippable(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(1)
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_input_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "checkpoint_roster": [
                {"checkpoint_id": f"old-{index}", "checkpoint_ref": ref}
                for index, ref in enumerate(old_refs)
            ],
            "pair_count": 1,
            "game_count": 3,
            "pair_specs": [],
            "rating_spec": {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "checkpoints": old_refs,
                "games_per_pair": 3,
            },
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.rating_round_progress_ref(tournament_id, rating_run_id, round_id),
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": 1,
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 0,
            "started_pair_count": 0,
        },
    )

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=3,
        stale_after_seconds=1,
    )

    assert decision["skip"] is False
    assert decision["is_stale"] is False
    assert decision["reason"] == "not_skippable"


def test_already_rated_old_pool_round_is_skippable_even_if_fresh(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(1)
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    new_ref = _checkpoint_ref("run-c", 10)
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    progress_path = tmp_path / arena.rating_round_progress_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "checkpoint_roster": arena.rating_roster_by_checkpoint(
                arena.normalize_checkpoint_specs(old_refs)
            ),
            "pair_count": 1,
            "game_count": 3,
            "pair_specs": [],
            "rating_spec": {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "checkpoints": old_refs,
                "games_per_pair": 3,
            },
        },
    )
    modal_arena.runs.write_json(
        progress_path,
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "status": "running",
            "phase": "game_map_started",
            "pair_count": 1,
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 1,
            "started_pair_count": 1,
        },
    )

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=3,
        latest_checkpoint_count=2,
        desired_rating_spec=arena.normalize_rating_spec(
            {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "checkpoints": [*old_refs, new_ref],
                "games_per_pair": 3,
            }
        ),
        stale_after_seconds=600,
    )

    assert decision["skip"] is True
    assert decision["is_stale"] is False
    assert decision["reason"] == "different_spec_already_rated_pool"
    assert decision["input_is_no_newer_than_latest"] is True


def test_stale_incomplete_smaller_pool_round_is_skippable(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(1)
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    progress_path = tmp_path / arena.rating_round_progress_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "checkpoint_roster": [
                {"checkpoint_id": f"old-{index}", "checkpoint_ref": ref}
                for index, ref in enumerate(old_refs)
            ],
            "pair_count": 100,
            "game_count": 2100,
            "pair_specs": [],
            "rating_spec": {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "checkpoints": old_refs,
                "games_per_pair": 21,
            },
        },
    )
    modal_arena.runs.write_json(
        progress_path,
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "status": "running",
            "phase": "games_running",
            "pair_count": 100,
            "game_count": 2100,
            "completed_pair_count": 10,
            "completed_game_count": 210,
            "started_pair_count": 10,
        },
    )
    old_mtime = time.time() - 120.0
    os.utime(input_path, (old_mtime, old_mtime))
    os.utime(progress_path, (old_mtime, old_mtime))

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=3,
        stale_after_seconds=60,
    )

    assert decision["skip"] is True
    assert decision["is_stale"] is True
    assert decision["reason"] == "stale_incomplete_smaller_pool"
    assert decision["stale_after_seconds"] == 60
    assert decision["completed_game_count"] == 210


def test_stale_zero_output_round_ignores_recent_progress_mtime(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(1)
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    progress_path = tmp_path / arena.rating_round_progress_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "checkpoint_roster": [
                {"checkpoint_id": f"old-{index}", "checkpoint_ref": ref}
                for index, ref in enumerate(refs)
            ],
            "pair_count": 1,
            "game_count": 3,
            "pair_specs": [],
            "rating_spec": {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "checkpoints": refs,
                "games_per_pair": 3,
            },
        },
    )
    modal_arena.runs.write_json(
        progress_path,
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "status": "running",
            "phase": "games_running",
            "pair_count": 1,
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 0,
            "started_pair_count": 0,
        },
    )
    old_mtime = time.time() - 120.0
    os.utime(input_path, (old_mtime, old_mtime))

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=len(refs),
        stale_after_seconds=60,
    )

    assert decision["skip"] is True
    assert decision["is_stale"] is True
    assert decision["reason"] == "zero_progress_orphan_round"
    assert decision["stale_basis"] == "round_input_or_game_output"
    assert decision["progress_updated_ts_ignored_for_stale"] is not None


def test_stale_incomplete_same_pool_round_is_skippable(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(1)
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    progress_path = tmp_path / arena.rating_round_progress_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "checkpoint_roster": [
                {"checkpoint_id": f"old-{index}", "checkpoint_ref": ref}
                for index, ref in enumerate(refs)
            ],
            "pair_count": 1,
            "game_count": 21,
            "pair_specs": [],
            "rating_spec": {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "checkpoints": refs,
                "games_per_pair": 21,
            },
        },
    )
    modal_arena.runs.write_json(
        progress_path,
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "status": "running",
            "phase": "games_running",
            "pair_count": 1,
            "game_count": 21,
            "completed_pair_count": 0,
            "completed_game_count": 4,
            "started_pair_count": 1,
        },
    )
    old_mtime = time.time() - 120.0
    os.utime(input_path, (old_mtime, old_mtime))
    os.utime(progress_path, (old_mtime, old_mtime))

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=len(refs),
        stale_after_seconds=60,
    )

    assert decision["skip"] is True
    assert decision["is_stale"] is True
    assert decision["reason"] == "stale_incomplete_round"
    assert decision["completed_game_count"] == 4


def test_recent_game_output_keeps_incomplete_round_running(
    tmp_path,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(1)
    rating_spec = arena.normalize_rating_spec(
        {
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "checkpoints": [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)],
            "games_per_pair": 3,
            "games_per_shard": 1,
        }
    )
    pairs = arena.build_rating_round_pair_specs(rating_spec, round_index=1)
    pair = pairs[0]
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    progress_path = tmp_path / arena.rating_round_progress_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "checkpoint_roster": arena.rating_roster_by_checkpoint(rating_spec["checkpoints"]),
            "pair_count": len(pairs),
            "game_count": 3,
            "pair_specs": pairs,
            "rating_spec": rating_spec,
        },
    )
    modal_arena.runs.write_json(
        progress_path,
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "status": "running",
            "phase": "games_running",
            "pair_count": len(pairs),
            "game_count": 3,
            "completed_pair_count": 0,
            "completed_game_count": 1,
            "started_pair_count": 1,
        },
    )
    modal_arena.runs.write_json(
        tmp_path / arena.game_summary_ref(tournament_id, pair["battle_id"], "game-000000"),
        {
            "schema_id": arena.GAME_SCHEMA_ID,
            "ok": True,
            "tournament_id": tournament_id,
            "battle_id": pair["battle_id"],
            "game_id": "game-000000",
            "pair_index": pair["pair_index"],
        },
    )
    old_mtime = time.time() - 120.0
    os.utime(input_path, (old_mtime, old_mtime))
    os.utime(progress_path, (old_mtime, old_mtime))

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=len(rating_spec["checkpoints"]),
        stale_after_seconds=60,
        scan_output_progress=True,
    )

    assert decision["skip"] is False
    assert decision["is_stale"] is False
    assert decision["completed_game_count"] == 1
    assert decision["latest_result_ts"] is not None


def test_scan_error_blocks_stale_round_skip(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    round_id = arena.rating_round_id(1)
    refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    input_path = tmp_path / arena.rating_round_input_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    progress_path = tmp_path / arena.rating_round_progress_ref(
        tournament_id,
        rating_run_id,
        round_id,
    )
    modal_arena.runs.write_json(
        input_path,
        {
            "schema_id": arena.RATING_ROUND_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "checkpoint_roster": [
                {"checkpoint_id": f"old-{index}", "checkpoint_ref": ref}
                for index, ref in enumerate(refs)
            ],
            "pair_count": 1,
            "game_count": 21,
            "pair_specs": [],
            "rating_spec": {
                "tournament_id": tournament_id,
                "rating_run_id": rating_run_id,
                "checkpoints": refs,
                "games_per_pair": 21,
            },
        },
    )
    modal_arena.runs.write_json(
        progress_path,
        {
            "schema_id": arena.RATING_PROGRESS_SCHEMA_ID,
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": round_id,
            "round_index": 1,
            "status": "running",
            "phase": "games_running",
            "pair_count": 1,
            "game_count": 21,
            "completed_pair_count": 0,
            "completed_game_count": 0,
            "started_pair_count": 0,
        },
    )
    old_mtime = time.time() - 120.0
    os.utime(input_path, (old_mtime, old_mtime))
    os.utime(progress_path, (old_mtime, old_mtime))

    def boom(*_args, **_kwargs):
        raise RuntimeError("scan broke")

    monkeypatch.setattr(modal_arena, "_rating_round_progress_payload", boom)

    decision = modal_arena._rating_round_skip_decision(
        tmp_path,
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        round_id=round_id,
        desired_checkpoint_count=len(refs),
        stale_after_seconds=60,
        scan_output_progress=True,
    )

    assert decision["skip"] is False
    assert decision["is_stale"] is True
    assert decision["reason"] == "not_skippable"
    assert decision["progress_scan_error"] == "RuntimeError: scan broke"
    assert decision["progress_scan_error_blocks_skip"] is True


def test_intake_drain_ignores_empty_waiting_round_stub(
    tmp_path,
    monkeypatch,
) -> None:
    tournament_id = "arena-a"
    rating_run_id = "elo-test"
    old_refs = [_checkpoint_ref("run-a", 10), _checkpoint_ref("run-b", 10)]
    new_ref = _checkpoint_ref("run-a", 20)
    all_refs = [*old_refs, new_ref]
    manifest = modal_arena._intake_manifest_from_discovery(
        tournament_id=tournament_id,
        rating_run_id=rating_run_id,
        scan_spec={
            "run_ids": "run-a,run-b",
            "checkpoint_selection": arena.CHECKPOINT_SELECTION_ALL,
        },
        rating_defaults={"continue_from_latest": True, "games_per_pair": 3},
        discovery={"checkpoint_refs": all_refs},
    )
    manifest = modal_arena._mark_intake_manifest_queued(manifest, [new_ref])
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
    modal_arena.runs.write_json(
        tmp_path / arena.rating_progress_ref(tournament_id, rating_run_id),
        {
            "schema_id": "curvyzero_curvytron_checkpoint_rating_progress/v0",
            "tournament_id": tournament_id,
            "rating_run_id": rating_run_id,
            "round_id": arena.rating_round_id(1),
            "round_index": 1,
            "phase": "waiting_for_round_input",
            "status": "pending",
            "pair_count": 0,
            "game_count": 0,
            "completed_pair_count": 0,
            "completed_game_count": 0,
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
            self.events = [{"checkpoint_ref": new_ref}]

        def len(self, *, partition):
            assert partition == manifest["queue_partition"]
            return len(self.events)

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

    assert result["existing_rating_run"] is True
    assert result["continue_from_latest"] is True
    assert result["rating_writer_finished"] is True
    assert result["spawn_skipped_reason"] == ""
    assert result["event_count"] == 1
    assert result["rating_call_id"] == "fc-test"
    assert len(fake_rating_loop.specs) == 1


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
