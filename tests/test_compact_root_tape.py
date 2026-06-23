from dataclasses import replace

import numpy as np
import pytest

from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_observation_contract import ResidentObservationBatchV1
from curvyzero.training.compact_policy_row_bridge import build_compact_root_batch_v1
from curvyzero.training.compact_policy_row_bridge import validate_compact_search_result_v1
from curvyzero.training.compact_rollout_slab import CompactRolloutSlab
from curvyzero.training.compact_root_tape import InMemoryCompactRootTapeRecorderV1
from curvyzero.training.compact_root_tape import (
    compact_root_batch_v1_from_tape_record,
)
from curvyzero.training.compact_root_tape import (
    compact_root_tape_record_v1_from_root_batch,
)
from curvyzero.training.compact_root_tape import read_compact_root_tape_npz_v1
from curvyzero.training.compact_root_tape import run_compact_root_tape_comparison_v1
from curvyzero.training.compact_root_tape import write_compact_root_tape_npz_v1
from curvyzero.training.compact_search_service import COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID
from curvyzero.training.compact_search_service import CompactSearchActionStepV1
from curvyzero.training.compact_search_service import compact_search_array_digest_v1
from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT
from curvyzero.training.multiplayer_source_state_target_rows import DEFAULT_TO_PLAY
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.source_state_hybrid_observation_profile import HybridCompactBatch


def test_compact_root_tape_round_trips_and_compares_services(tmp_path):
    root_batch = build_compact_root_batch_v1(
        _compact_batch(),
        search_lane="unit_test_root_tape",
    )
    tape = InMemoryCompactRootTapeRecorderV1(tape_label="unit_test_tape")
    tape.record_root_batch(root_batch, record_index=7)
    artifact = tmp_path / "root_tape.npz"

    write_compact_root_tape_npz_v1(artifact, tape.build_tape())
    loaded = read_compact_root_tape_npz_v1(artifact)

    replayed = compact_root_batch_v1_from_tape_record(loaded.records[0])
    np.testing.assert_array_equal(replayed.observation, root_batch.observation)
    assert replayed.metadata["root_tape_replay"] is True
    assert replayed.metadata["host_observation_authoritative"] is True
    assert replayed.resident_observation is None

    reference = _StaticSearchService(
        search_impl="reference_search",
        selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
        root_value=np.asarray([0.0, 0.25, -0.25, 0.5], dtype=np.float32),
    )
    candidate = _StaticSearchService(
        search_impl="candidate_search",
        selected_action=np.asarray([0, 2, 2, 1], dtype=np.int16),
        root_value=np.asarray([0.0, -0.25, -0.25, 0.0], dtype=np.float32),
        h2d_bytes=128,
        d2h_bytes=64,
    )

    report = run_compact_root_tape_comparison_v1(
        loaded,
        services={"candidate": candidate, "reference": reference},
        reference_label="reference",
    )

    assert report["record_count"] == 1
    assert report["backend"]["candidate"]["h2d_bytes"] == 128
    assert report["backend"]["candidate"]["d2h_bytes"] == 64
    comparison = report["comparison"]["candidate_vs_reference"]
    assert comparison["active_root_count"] == 4
    assert comparison["action_match_fraction"] == pytest.approx(0.5)
    assert comparison["visit_l1_max"] > 0.0
    assert comparison["root_value_abs_diff_max"] == pytest.approx(0.5)


def test_compact_root_tape_compares_three_services():
    root_batch = build_compact_root_batch_v1(
        _compact_batch(),
        search_lane="unit_test_root_tape_three_services",
    )
    tape = InMemoryCompactRootTapeRecorderV1(tape_label="unit_test_tape_three_services")
    tape.record_root_batch(root_batch, record_index=0)

    primary = _StaticSearchService(
        search_impl="primary_search",
        selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
        root_value=np.asarray([0.0, 0.25, -0.25, 0.5], dtype=np.float32),
    )
    floor = _StaticSearchService(
        search_impl="fixed_shape_floor_search",
        selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
        root_value=np.asarray([0.0, 0.25, -0.25, 0.5], dtype=np.float32),
    )
    mctx = _StaticSearchService(
        search_impl="mctx_profile_search",
        selected_action=np.asarray([2, 1, 2, 0], dtype=np.int16),
        root_value=np.asarray([0.5, 0.25, -0.25, 0.25], dtype=np.float32),
        h2d_bytes=1024,
        d2h_bytes=256,
    )

    report = run_compact_root_tape_comparison_v1(
        tape.build_tape(),
        services={
            "primary": primary,
            "fixed_shape_floor": floor,
            "mctx": mctx,
        },
        reference_label="primary",
    )

    assert set(report["backend"]) == {"fixed_shape_floor", "mctx", "primary"}
    assert report["backend"]["mctx"]["h2d_bytes"] == 1024
    assert report["backend"]["mctx"]["d2h_bytes"] == 256
    assert set(report["comparison"]) == {
        "fixed_shape_floor_vs_primary",
        "mctx_vs_primary",
    }
    assert report["comparison"]["fixed_shape_floor_vs_primary"][
        "action_match_fraction"
    ] == pytest.approx(1.0)
    assert report["comparison"]["mctx_vs_primary"]["action_match_fraction"] == (
        pytest.approx(0.75)
    )


def test_compact_root_tape_requires_explicit_snapshot_for_resident_source():
    root_batch = build_compact_root_batch_v1(
        _compact_batch(),
        search_lane="unit_test_root_tape_resident_guard",
    )
    resident_like = replace(
        root_batch,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    )

    with pytest.raises(ReplayCompatibilityError, match="host_observation_snapshot"):
        compact_root_tape_record_v1_from_root_batch(
            resident_like,
            record_index=0,
        )


def test_compact_root_tape_recorder_rejects_resident_source_without_real_readback():
    root_batch = build_compact_root_batch_v1(
        _compact_batch(),
        search_lane="unit_test_root_tape_resident_recorder_guard",
    )
    resident_like = replace(
        root_batch,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    )
    recorder = InMemoryCompactRootTapeRecorderV1(
        tape_label="unit_test_resident_recorder",
        allow_resident_host_snapshot=True,
    )

    with pytest.raises(ReplayCompatibilityError, match="device-to-host snapshot"):
        recorder.record_root_batch(resident_like, record_index=0)


def test_compact_root_tape_compare_requires_two_services():
    root_batch = build_compact_root_batch_v1(
        _compact_batch(),
        search_lane="unit_test_root_tape_one_service_guard",
    )
    tape = InMemoryCompactRootTapeRecorderV1(tape_label="unit_test_tape_one_service")
    tape.record_root_batch(root_batch, record_index=0)

    with pytest.raises(ReplayCompatibilityError, match="at least two services"):
        run_compact_root_tape_comparison_v1(
            tape.build_tape(),
            services={
                "primary": _StaticSearchService(
                    search_impl="primary_search",
                    selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
                    root_value=np.zeros((4,), dtype=np.float32),
                )
            },
            reference_label="primary",
        )


def test_compact_root_tape_compare_validates_each_service_identity():
    root_batch = build_compact_root_batch_v1(
        _compact_batch(),
        search_lane="unit_test_root_tape_identity_guard",
    )
    tape = InMemoryCompactRootTapeRecorderV1(tape_label="unit_test_tape_identity")
    tape.record_root_batch(root_batch, record_index=0)

    with pytest.raises(ReplayCompatibilityError, match="policy_env_id"):
        run_compact_root_tape_comparison_v1(
            tape.build_tape(),
            services={
                "poisoned": _PoisonedIdentitySearchService(),
                "reference": _StaticSearchService(
                    search_impl="reference_search",
                    selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
                    root_value=np.zeros((4,), dtype=np.float32),
                ),
            },
            reference_label="reference",
        )


def test_compact_root_tape_counts_host_to_device_byte_aliases():
    root_batch = build_compact_root_batch_v1(
        _compact_batch(),
        search_lane="unit_test_root_tape_byte_aliases",
    )
    tape = InMemoryCompactRootTapeRecorderV1(tape_label="unit_test_tape_byte_aliases")
    tape.record_root_batch(root_batch, record_index=0)

    report = run_compact_root_tape_comparison_v1(
        tape.build_tape(),
        services={
            "candidate": _AliasTelemetrySearchService(),
            "reference": _StaticSearchService(
                search_impl="reference_search",
                selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
                root_value=np.zeros((4,), dtype=np.float32),
            ),
        },
        reference_label="reference",
    )

    assert report["backend"]["candidate"]["h2d_bytes"] == 128
    assert report["backend"]["candidate"]["d2h_bytes"] == 64


def test_compact_rollout_slab_records_root_tape_before_search():
    compact_batch = _compact_batch()
    events = []

    class Recorder:
        def __init__(self):
            self.records = []

        def record_root_batch(self, root_batch, *, record_index, metadata=None):
            events.append("capture")
            self.records.append(
                compact_root_tape_record_v1_from_root_batch(
                    root_batch,
                    record_index=record_index,
                    capture_metadata=metadata,
                )
            )

    recorder = Recorder()
    service = _StaticSearchService(
        search_impl="unit_test_slab_tape_service",
        selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
        root_value=np.zeros((4,), dtype=np.float32),
        events=events,
    )
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=service,
        search_lane="unit_test_slab_tape",
        policy_source="unit_test_slab_tape",
        root_tape_recorder=recorder,
    )

    step = slab.step(compact_batch)

    assert events == ["capture", "search"]
    assert len(recorder.records) == 1
    assert recorder.records[0].record_index == 0
    assert recorder.records[0].capture_metadata["compact_rollout_slab_capture"] is True
    np.testing.assert_array_equal(
        compact_root_batch_v1_from_tape_record(recorder.records[0]).legal_mask,
        step.root_batch.legal_mask,
    )


def test_compact_rollout_slab_resident_two_phase_requires_device_flush():
    compact_batch = _resident_compact_batch()
    service = _ResidentTwoPhaseHostFlushOnlyService()
    slab = CompactRolloutSlab(
        batch_size=2,
        player_count=2,
        search_service=service,
        search_lane="unit_test_resident_no_host_fallback",
        policy_source="unit_test_resident_no_host_fallback",
    )

    slab.step(compact_batch)
    with pytest.raises(ReplayCompatibilityError, match="flush_device_replay_payload"):
        slab.step(compact_batch)


def _compact_batch() -> HybridCompactBatch:
    batch_size = 2
    player_count = 2
    root_count = batch_size * player_count
    observation = (
        np.arange(
            batch_size * player_count * 4 * 64 * 64,
            dtype=np.uint32,
        )
        % 256
    ).astype(np.uint8).reshape(batch_size, player_count, 4, 64, 64)
    action_mask = np.ones((batch_size, player_count, ACTION_COUNT), dtype=np.bool_)
    return HybridCompactBatch(
        observation=observation,
        action_mask=action_mask,
        reward=np.zeros((batch_size, player_count), dtype=np.float32),
        final_reward_map=np.zeros((batch_size, player_count), dtype=np.float32),
        done=np.zeros((batch_size,), dtype=np.bool_),
        policy_env_id=np.arange(root_count, dtype=np.int64),
        policy_env_row=np.repeat(np.arange(batch_size, dtype=np.int32), player_count),
        policy_player=np.tile(np.arange(player_count, dtype=np.int16), batch_size),
        target_reward=np.zeros((root_count, 1), dtype=np.float32),
        done_root=np.zeros((root_count,), dtype=np.bool_),
        to_play=np.full((root_count,), DEFAULT_TO_PLAY, dtype=np.int64),
        active_root_mask=np.ones((root_count,), dtype=np.bool_),
        final_observation=None,
        final_observation_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        terminal_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        autoreset_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        terminal_global_rows=np.asarray([], dtype=np.int32),
        autoreset_global_rows=np.asarray([], dtype=np.int32),
        episode_step=np.zeros((batch_size,), dtype=np.int32),
        elapsed_ms=np.zeros((batch_size,), dtype=np.float64),
        round_id=np.zeros((batch_size,), dtype=np.int32),
        alive=np.ones((batch_size, player_count), dtype=np.bool_),
        joint_action=np.zeros((batch_size, player_count), dtype=np.int16),
    )


def _resident_compact_batch() -> HybridCompactBatch:
    batch = _compact_batch()
    root_count = int(batch.policy_env_id.shape[0])
    resident = ResidentObservationBatchV1(
        device_observation=batch.observation,
        root_device_observation=batch.observation.reshape(root_count, 4, 64, 64),
        generation_id=11,
        batch_size=2,
        player_count=2,
        stack_shape=(4, 64, 64),
        dtype="uint8",
        device="cuda:0",
        row_major_order=True,
        fresh_for_step_index=3,
        source_backend="unit_test_resident_tape",
    )
    return replace(
        batch,
        resident_observation=resident,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    )


class _StaticSearchService:
    calls_train_muzero = False
    trainer_defaults_changed = False
    touches_live_runs = False
    num_simulations = 8

    def __init__(
        self,
        *,
        search_impl: str,
        selected_action: np.ndarray,
        root_value: np.ndarray,
        h2d_bytes: int = 0,
        d2h_bytes: int = 0,
        events: list[str] | None = None,
    ) -> None:
        self.search_impl = search_impl
        self.selected_action = selected_action
        self.root_value = root_value
        self.h2d_bytes = int(h2d_bytes)
        self.d2h_bytes = int(d2h_bytes)
        self.events = events

    def run(self, root_batch):
        if self.events is not None:
            self.events.append("search")
        visit_policy = np.zeros((self.selected_action.size, ACTION_COUNT), dtype=np.float32)
        visit_policy[np.arange(self.selected_action.size), self.selected_action] = 1.0
        return validate_compact_search_result_v1(
            root_batch,
            selected_action=self.selected_action,
            visit_policy=visit_policy,
            root_value=self.root_value,
            raw_visit_counts=visit_policy * float(self.num_simulations),
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
            metadata={
                "profile_telemetry": {
                    f"{self.search_impl}_h2d_bytes": self.h2d_bytes,
                    f"{self.search_impl}_d2h_bytes": self.d2h_bytes,
                }
            },
        )


class _PoisonedIdentitySearchService(_StaticSearchService):
    def __init__(self) -> None:
        super().__init__(
            search_impl="poisoned_identity_search",
            selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
            root_value=np.zeros((4,), dtype=np.float32),
        )

    def run(self, root_batch):
        result = super().run(root_batch)
        return replace(result, policy_env_id=result.policy_env_id[::-1].copy())


class _AliasTelemetrySearchService(_StaticSearchService):
    def __init__(self) -> None:
        super().__init__(
            search_impl="alias_telemetry_search",
            selected_action=np.asarray([0, 1, 2, 0], dtype=np.int16),
            root_value=np.zeros((4,), dtype=np.float32),
        )

    def run(self, root_batch):
        result = super().run(root_batch)
        telemetry = {
            "alias_host_to_device_bytes": 128,
            "alias_device_to_host_bytes": 64,
        }
        return replace(result, metadata={**result.metadata, "profile_telemetry": telemetry})


class _ResidentTwoPhaseHostFlushOnlyService:
    supports_two_phase_compact_search = True
    search_impl = "unit_test_resident_host_flush_only"
    num_simulations = 1
    calls_train_muzero = False
    trainer_defaults_changed = False
    touches_live_runs = False

    def run_action_step(self, root_batch):
        root_index = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
        selected = np.zeros((root_index.size,), dtype=np.int16)
        return CompactSearchActionStepV1(
            replay_payload_handle="unit-test-host-only",
            root_index=root_index,
            env_row=root_batch.env_row[root_index].astype(np.int32, copy=True),
            player=root_batch.player[root_index].astype(np.int16, copy=True),
            policy_env_id=root_batch.policy_env_id[root_index].astype(
                np.int64,
                copy=True,
            ),
            selected_action=selected,
            metadata={
                "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                "phase": "action_critical",
                "selected_action_digest": compact_search_array_digest_v1(selected),
            },
        )

    def flush_replay_payload(self, replay_payload_handle):
        raise AssertionError("resident path must not fall back to host replay payload")
