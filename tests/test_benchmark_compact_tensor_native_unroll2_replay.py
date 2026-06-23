from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

benchmark = pytest.importorskip("benchmark_compact_tensor_native_unroll2_replay")


def test_tensor_native_unroll2_replay_benchmark_proves_toy_identity():
    config = benchmark.BenchmarkConfig(
        records=6,
        rows_per_record=2,
        sample_rows=0,
        iters=1,
        terminal_rate=0.25,
        seed=7,
        device="cpu",
    )

    result = benchmark.run_benchmark(config)

    assert result["proof"]["required_pass"] is True
    assert result["proof"]["all_equal"] is True
    assert result["proof"]["checksum_match"] is True
    assert result["proof"]["ring_cache_used"] is True
    assert result["proof"]["grouped_cache_used"] is True
    assert result["proof"]["host_fallback_allowed"] is False
    assert result["proof"]["device_replay_index_rows_sample_all"] is True
    assert result["fixture"]["candidate_records"] == 6


def test_real_sample_gate_uses_default_off_tensor_native_replay_path():
    torch = pytest.importorskip("torch")
    from curvyzero.training import source_state_hybrid_observation_profile as hybrid

    config = benchmark.BenchmarkConfig(
        records=6,
        rows_per_record=2,
        sample_rows=0,
        iters=1,
        terminal_rate=0.25,
        seed=7,
        device="cpu",
    )
    fixture = benchmark._build_fixture(config, torch, torch.device("cpu"))
    assert (
        len(dict(fixture.ring.snapshot_for_sample().tensor_native_unroll2_table_by_record_index))
        == 0
    )
    fixture.ring.update_store_metadata(
        {
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_REQUESTED_KEY: True,
        }
    )
    sample_snapshot = fixture.ring.snapshot_for_sample()
    maintained_tables = dict(sample_snapshot.tensor_native_unroll2_table_by_record_index)
    assert len(maintained_tables) == len(fixture.sample_plan.candidate_entries)
    assert sum(int(entry.row_count) for entry in maintained_tables.values()) == 12

    result = fixture.ring.sample_from_snapshot(
        sample_snapshot,
        seed=config.seed,
        sample_batch_size=config.sample_rows,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    tensor_native_batch = result["learner_batch"]
    grouped_batch = hybrid._build_compact_resident_grouped_device_learner_batch_fast(
        group_samples=fixture.group_samples,
        metadata=fixture.metadata,
        sample_position_order=fixture.sample_plan.sample_position_order,
    )

    equalities = benchmark._batch_equalities(torch, tensor_native_batch, grouped_batch)
    assert all(equalities.values()), equalities
    metadata = tensor_native_batch.metadata
    assert (
        metadata[hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_REQUESTED_KEY]
        is True
    )
    assert metadata[hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_USED_KEY] is True
    assert (
        metadata[hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_CALL_COUNT_KEY]
        == 1
    )
    assert metadata["compact_muzero_learner_batch_tensor_native_replay_fallback_count"] == 0
    assert metadata["compact_muzero_learner_batch_tensor_native_replay_table_rows"] == 12
    assert (
        metadata["compact_muzero_learner_batch_tensor_native_replay_impl"]
        == "maintained_unroll2_table_gather_v1"
    )
    assert (
        metadata["compact_muzero_learner_batch_tensor_native_replay_table_source"]
        == "maintained_record_table_v1"
    )
    assert (
        metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_reused_record_count"
        ]
        == len(fixture.sample_plan.candidate_entries)
    )
    assert (
        metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_missing_record_count"
        ]
        == 0
    )
    assert (
        metadata[
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used"
        ]
        is False
    )
    assert (
        metadata["compact_muzero_learner_batch_sample_row_checksum"]
        == grouped_batch.metadata["compact_muzero_learner_batch_sample_row_checksum"]
    )
    telemetry = result["telemetry"]
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
        ]
        is True
    )
    assert (
        telemetry[
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_table_source"
            )
        ]
        == "maintained_record_table_v1"
    )
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used"
        ]
        is False
    )


def test_tensor_native_replay_missing_maintained_table_falls_back_closed():
    torch = pytest.importorskip("torch")
    from curvyzero.training import source_state_hybrid_observation_profile as hybrid

    config = benchmark.BenchmarkConfig(
        records=6,
        rows_per_record=2,
        sample_rows=0,
        iters=1,
        terminal_rate=0.25,
        seed=7,
        device="cpu",
    )
    fixture = benchmark._build_fixture(config, torch, torch.device("cpu"))
    fixture.ring.update_store_metadata(
        {
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_REQUESTED_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_SELECTED_MAINTAINED_GATHER_KEY: True,
        }
    )
    sample_snapshot = fixture.ring.snapshot_for_sample()
    broken_snapshot = replace(
        sample_snapshot,
        tensor_native_unroll2_table_by_record_index={},
    )

    result = fixture.ring.sample_from_snapshot(
        broken_snapshot,
        seed=config.seed,
        sample_batch_size=config.sample_rows,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    fallback_batch = result["learner_batch"]
    grouped_batch = hybrid._build_compact_resident_grouped_device_learner_batch_fast(
        group_samples=fixture.group_samples,
        metadata=fixture.metadata,
        sample_position_order=fixture.sample_plan.sample_position_order,
    )

    equalities = benchmark._batch_equalities(torch, fallback_batch, grouped_batch)
    assert all(equalities.values()), equalities
    metadata = fallback_batch.metadata
    assert (
        metadata[hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_REQUESTED_KEY]
        is True
    )
    assert metadata[hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_USED_KEY] is False
    assert (
        metadata["compact_muzero_learner_batch_tensor_native_replay_fallback_count"]
        == 1
    )
    assert (
        metadata["compact_muzero_learner_batch_tensor_native_replay_fallback_reason"]
        == "tensor-native replay missing maintained table"
    )
    telemetry = result["telemetry"]
    assert (
        telemetry[
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
        ]
        is False
    )
    assert (
        telemetry[
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "tensor_native_replay_fallback_reason"
            )
        ]
        == "tensor-native replay missing maintained table"
    )


def test_tensor_native_replay_maintained_table_survives_eviction_rebuild():
    torch = pytest.importorskip("torch")
    from curvyzero.training import source_state_hybrid_observation_profile as hybrid

    config = benchmark.BenchmarkConfig(
        records=6,
        rows_per_record=2,
        sample_rows=0,
        iters=1,
        terminal_rate=0.25,
        seed=7,
        device="cpu",
    )
    device = torch.device("cpu")
    fixture = benchmark._build_fixture(config, torch, device)
    fixture.ring.update_store_metadata(
        {
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_REQUESTED_KEY: True,
            hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_SELECTED_MAINTAINED_GATHER_KEY: True,
        }
    )
    before_snapshot = fixture.ring.snapshot_for_sample()
    assert 0 in dict(before_snapshot.tensor_native_unroll2_table_by_record_index)

    append_config = benchmark.BenchmarkConfig(
        records=7,
        rows_per_record=2,
        sample_rows=0,
        iters=1,
        terminal_rate=0.25,
        seed=7,
        device="cpu",
    )
    record_index = 6
    done_mask = benchmark._terminal_mask_for_record(append_config, record_index)
    previous_snapshot = benchmark._resident_snapshot(
        torch,
        device=device,
        record_index=record_index,
        rows=int(config.rows_per_record),
        generation_id=record_index * 2,
        final_mask=np.zeros((int(config.rows_per_record),), dtype=np.bool_),
    )
    current_snapshot = benchmark._resident_snapshot(
        torch,
        device=device,
        record_index=record_index,
        rows=int(config.rows_per_record),
        generation_id=record_index * 2 + 1,
        final_mask=done_mask,
    )
    index_rows = benchmark._index_rows_for_record(
        torch,
        config=append_config,
        device=device,
        record_index=record_index,
        done_mask=done_mask,
    )
    fixture.ring.append(
        previous_step=benchmark._step_for_record(
            snapshot=previous_snapshot,
            index_rows=index_rows,
            done_mask=np.zeros((int(config.rows_per_record),), dtype=np.bool_),
        ),
        current_step=benchmark._step_for_record(
            snapshot=current_snapshot,
            index_rows=index_rows,
            done_mask=done_mask,
        ),
        index_rows=index_rows,
    )

    after_snapshot = fixture.ring.snapshot_for_sample()
    maintained_tables = dict(after_snapshot.tensor_native_unroll2_table_by_record_index)
    remaining_records = {
        int(getattr(entry.index_rows, "record_index"))
        for entry in tuple(after_snapshot.entries)
    }
    assert fixture.ring.evicted_entry_count == 1
    assert remaining_records == {1, 2, 3, 4, 5, 6}
    assert 0 not in maintained_tables
    assert set(maintained_tables) == remaining_records

    plan = benchmark._sample_plan(config, after_snapshot)
    group_samples = benchmark._group_samples_for_flat_rows(
        plan=plan,
        snapshot=after_snapshot,
        flat_rows=plan.sampled_flat_rows,
        sampled_group=plan.sampled_group,
    )
    metadata = benchmark._metadata_for_plan(config, plan)
    result = fixture.ring.sample_from_snapshot(
        after_snapshot,
        seed=config.seed,
        sample_batch_size=config.sample_rows,
        require_next_targets=True,
        num_unroll_steps=2,
        build_compact_muzero_learner_batch=True,
        compact_muzero_learner_batch_only=True,
    )
    tensor_native_batch = result["learner_batch"]
    grouped_batch = hybrid._build_compact_resident_grouped_device_learner_batch_fast(
        group_samples=group_samples,
        metadata=metadata,
        sample_position_order=plan.sample_position_order,
    )

    equalities = benchmark._batch_equalities(torch, tensor_native_batch, grouped_batch)
    assert all(equalities.values()), equalities
    metadata = tensor_native_batch.metadata
    assert metadata[hybrid.COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_USED_KEY] is True
    assert (
        metadata["compact_muzero_learner_batch_tensor_native_replay_table_source"]
        == "selected_maintained_record_table_v1"
    )
    assert (
        metadata[
            "compact_muzero_learner_batch_tensor_native_replay_table_reused_record_count"
        ]
        == len(plan.candidate_entries)
    )
    assert metadata["compact_muzero_learner_batch_tensor_native_replay_table_rows"] == 12
