#!/usr/bin/env python3
"""Toy ceiling for tensor-native compact replay unroll-2 batches.

This benchmark is local proof machinery, not speed evidence. It compares:

1. the real compact replay ring sample/build path,
2. the real resident grouped learner builder with learner-ready unroll-2 cache,
3. a toy flat tensor-native gather from a prepacked learner-ready table.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.training import source_state_hybrid_observation_profile as hybrid_profile
from curvyzero.training.compact_muzero_learner import CompactMuZeroLearnerBatchV1
from curvyzero.training.compact_observation_contract import ResidentObservationBatchV1
from curvyzero.training.compact_policy_row_bridge import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    CompactDeviceReplayIndexRowsV1,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_CALL_COUNT_KEY,
    COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_KEY,
    COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY,
    COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_USED_KEY,
    COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_CALL_COUNT_KEY,
    COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_KEY,
    COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_REQUESTED_KEY,
    COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_SELECTED_MAINTAINED_GATHER_KEY,
    COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_USED_KEY,
    PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
    _CompactReplayRingV1,
    _build_compact_resident_grouped_device_learner_batch_fast,
)


LEARNER_TENSOR_FIELDS = (
    "observation",
    "action",
    "action_mask",
    "target_reward",
    "target_value",
    "target_policy",
    "target_reward_mask",
    "target_value_mask",
    "target_policy_mask",
    "action_valid_mask",
    "weights",
    "next_action_mask",
    "done",
    "terminated",
    "truncated",
)


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    records: int = 32
    rows_per_record: int = 8
    sample_rows: int = 0
    iters: int = 5
    terminal_rate: float = 0.05
    seed: int = 123
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class SamplePlan:
    candidate_entries: tuple[Any, ...]
    sampled_flat_rows: np.ndarray
    sampled_group: np.ndarray
    sample_position_order: np.ndarray
    offsets: np.ndarray
    group_counts: np.ndarray
    source_record_pairs: list[list[int]]
    source_record_windows: list[list[int]]
    replace: bool
    sample_row_count: int
    total_index_rows: int
    terminal_row_forced: bool


@dataclass(frozen=True, slots=True)
class Fixture:
    config: BenchmarkConfig
    ring: _CompactReplayRingV1
    snapshot: Any
    sample_plan: SamplePlan
    group_samples: list[dict[str, Any]]
    identity_group_samples: list[dict[str, Any]]
    metadata: dict[str, Any]
    identity_metadata: dict[str, Any]
    identity_position_order: np.ndarray


def _resolve_torch_device(torch: Any, device: str) -> Any:
    requested = torch.device(str(device))
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda is not available")
    return requested


def _sync_device(torch: Any, device: Any) -> float:
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        return 0.0
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    blend = rank - lower
    return ordered[lower] * (1.0 - blend) + ordered[upper] * blend


def _timed_path(
    *,
    name: str,
    rows: int,
    iters: int,
    torch: Any,
    device: Any,
    fn: Callable[[], Any],
) -> tuple[dict[str, Any], Any]:
    wall_sec: list[float] = []
    process_cpu_ns: list[int] = []
    thread_cpu_ns: list[int] = []
    cuda_sync_sec = 0.0
    result = None
    thread_time_ns = getattr(time, "thread_time_ns", None)
    for _ in range(int(iters)):
        cuda_sync_sec += _sync_device(torch, device)
        process_started = int(time.process_time_ns())
        thread_started = int(thread_time_ns()) if callable(thread_time_ns) else 0
        started = time.perf_counter()
        result = fn()
        cuda_sync_sec += _sync_device(torch, device)
        wall_sec.append(time.perf_counter() - started)
        process_cpu_ns.append(int(time.process_time_ns()) - process_started)
        if callable(thread_time_ns):
            thread_cpu_ns.append(int(thread_time_ns()) - thread_started)
        else:
            thread_cpu_ns.append(0)
    median_wall = _median(wall_sec)
    timing = {
        "name": name,
        "iters": int(iters),
        "rows": int(rows),
        "wall_sec": {
            "min": float(min(wall_sec)) if wall_sec else 0.0,
            "median": float(median_wall),
            "p95": float(_percentile(wall_sec, 0.95)),
            "max": float(max(wall_sec)) if wall_sec else 0.0,
            "values": [float(value) for value in wall_sec],
        },
        "rows_per_sec_median": float(int(rows) / median_wall) if median_wall > 0 else 0.0,
        "process_cpu_time_ns_median": int(statistics.median(process_cpu_ns))
        if process_cpu_ns
        else 0,
        "thread_cpu_time_ns_median": int(statistics.median(thread_cpu_ns))
        if thread_cpu_ns
        else 0,
        "cuda_sync_sec": float(cuda_sync_sec),
    }
    return timing, result


def _terminal_mask_for_record(config: BenchmarkConfig, record_index: int) -> np.ndarray:
    rows = int(config.rows_per_record)
    if record_index == int(config.records) - 1:
        return np.ones((rows,), dtype=np.bool_)
    if float(config.terminal_rate) <= 0.0:
        return np.zeros((rows,), dtype=np.bool_)
    rng = np.random.default_rng(int(config.seed) + int(record_index) * 9973)
    return (rng.random(rows) < float(config.terminal_rate)).astype(np.bool_, copy=False)


def _resident_snapshot(
    torch: Any,
    *,
    device: Any,
    record_index: int,
    rows: int,
    generation_id: int,
    final_mask: np.ndarray,
) -> ResidentObservationBatchV1:
    row_index = torch.arange(rows, dtype=torch.int64, device=device)
    observation = torch.zeros((rows, 1, 4, 8, 8), dtype=torch.uint8, device=device)
    observation[:, 0, -1, 0, 0] = (
        (int(record_index) * 17 + row_index) % 251
    ).to(dtype=torch.uint8)
    observation[:, 0, -1, 0, 1] = int(generation_id) % 251
    final_device_observation = None
    root_final_device_observation = None
    if bool(np.asarray(final_mask, dtype=np.bool_).any()):
        final_device_observation = observation.clone()
        final_device_observation[:, 0, -1, 0, 2] = (
            (int(record_index) * 19 + row_index + 37) % 251
        ).to(dtype=torch.uint8)
        root_final_device_observation = final_device_observation.reshape(rows, 4, 8, 8)
    return ResidentObservationBatchV1(
        device_observation=observation,
        root_device_observation=observation.reshape(rows, 4, 8, 8),
        generation_id=int(generation_id),
        batch_size=int(rows),
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device=str(torch.device(device)),
        row_major_order=True,
        fresh_for_step_index=int(generation_id),
        source_backend=PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
        host_fallback_allowed=False,
        metadata={"benchmark_compact_tensor_native_unroll2_replay": True},
        final_device_observation=final_device_observation,
        root_final_device_observation=root_final_device_observation,
        final_observation_row_mask=np.asarray(final_mask, dtype=np.bool_).copy(),
    )


def _index_rows_for_record(
    torch: Any,
    *,
    config: BenchmarkConfig,
    device: Any,
    record_index: int,
    done_mask: np.ndarray,
) -> CompactDeviceReplayIndexRowsV1:
    rows = int(config.rows_per_record)
    row_index = torch.arange(rows, dtype=torch.int64, device=device)
    actions = ((int(record_index) + row_index) % ACTION_COUNT).to(dtype=torch.int16)
    policy = torch.zeros((rows, ACTION_COUNT), dtype=torch.float32, device=device)
    policy.scatter_(1, actions.to(dtype=torch.long).reshape(-1, 1), 1.0)
    done = torch.as_tensor(np.asarray(done_mask, dtype=np.bool_), dtype=torch.bool, device=device)
    done_rows = np.flatnonzero(np.asarray(done_mask, dtype=np.bool_)).astype(
        np.int64,
        copy=False,
    )
    metadata = {
        "device_replay_index_rows": True,
        "index_row_count": int(rows),
        "done_row_count": int(done_rows.size),
        "done_row_indices": [int(row) for row in done_rows.tolist()],
        "terminated_row_count": int(done_rows.size),
        "terminated_row_indices": [int(row) for row in done_rows.tolist()],
        "truncated_row_count": 0,
        "truncated_row_indices": [],
        "next_final_observation_row_count": int(done_rows.size),
        "next_final_observation_row_indices": [int(row) for row in done_rows.tolist()],
        "host_search_payload_fallback_allowed": False,
        "env_player_identity_digest": f"benchmark_rows={rows}:players=1",
        "search_impl": "benchmark_tensor_native_unroll2_replay",
        "num_simulations": 0,
        "active_root_count": int(rows),
    }
    return CompactDeviceReplayIndexRowsV1(
        metadata=metadata,
        record_index=int(record_index),
        next_record_index=int(record_index) + 1,
        compact_root_row=row_index.to(dtype=torch.int32),
        policy_env_id=row_index.to(dtype=torch.int64),
        policy_row=row_index.to(dtype=torch.int32),
        env_row=row_index.to(dtype=torch.int32),
        player=torch.zeros((rows,), dtype=torch.int16, device=device),
        action=actions,
        action_mask=torch.ones((rows, ACTION_COUNT), dtype=torch.bool, device=device),
        policy_target=policy,
        root_value=(float(record_index) + row_index.to(dtype=torch.float32) / 100.0),
        reward=(float(record_index + 1) + row_index.to(dtype=torch.float32) / 10.0),
        final_reward=(float(record_index + 1) + row_index.to(dtype=torch.float32) / 5.0),
        done=done,
        terminated=done.clone(),
        truncated=torch.zeros((rows,), dtype=torch.bool, device=device),
        next_final_observation_row=done.clone(),
        to_play=torch.full((rows,), -1, dtype=torch.int64, device=device),
        policy_source="benchmark_tensor_native_unroll2_replay",
    )


def _step_for_record(
    *,
    snapshot: ResidentObservationBatchV1,
    index_rows: CompactDeviceReplayIndexRowsV1,
    done_mask: np.ndarray,
) -> Any:
    rows = int(snapshot.batch_size)
    return SimpleNamespace(
        observation=np.zeros((rows, 1, 4, 8, 8), dtype=np.float32),
        action_mask=np.ones((rows, 1, ACTION_COUNT), dtype=np.bool_),
        reward=np.zeros((rows, 1), dtype=np.float32),
        final_reward_map=np.zeros((rows, 1), dtype=np.float32),
        done=np.asarray(done_mask, dtype=np.bool_).copy(),
        payload={
            "joint_action": index_rows.action.detach()
            .cpu()
            .numpy()
            .astype(np.int16, copy=False)
            .reshape(rows, 1)
        },
        compact_batch=SimpleNamespace(
            observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        ),
        resident_observation_replay_snapshot=snapshot,
    )


def _build_ring(config: BenchmarkConfig, torch: Any, device: Any) -> _CompactReplayRingV1:
    ring = _CompactReplayRingV1(
        capacity=int(config.records),
        metadata={
            COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_KEY: True,
            COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY: True,
            "benchmark_compact_tensor_native_unroll2_replay": True,
        },
    )
    previous_steps: list[Any] = []
    current_steps: list[Any] = []
    index_rows_by_record: list[CompactDeviceReplayIndexRowsV1] = []
    for record_index in range(int(config.records)):
        done_mask = _terminal_mask_for_record(config, record_index)
        previous_snapshot = _resident_snapshot(
            torch,
            device=device,
            record_index=record_index,
            rows=int(config.rows_per_record),
            generation_id=record_index * 2,
            final_mask=np.zeros((int(config.rows_per_record),), dtype=np.bool_),
        )
        current_snapshot = _resident_snapshot(
            torch,
            device=device,
            record_index=record_index,
            rows=int(config.rows_per_record),
            generation_id=record_index * 2 + 1,
            final_mask=done_mask,
        )
        index_rows = _index_rows_for_record(
            torch,
            config=config,
            device=device,
            record_index=record_index,
            done_mask=done_mask,
        )
        previous_steps.append(
            _step_for_record(
                snapshot=previous_snapshot,
                index_rows=index_rows,
                done_mask=np.zeros((int(config.rows_per_record),), dtype=np.bool_),
            )
        )
        current_steps.append(
            _step_for_record(snapshot=current_snapshot, index_rows=index_rows, done_mask=done_mask)
        )
        index_rows_by_record.append(index_rows)
    for previous_step, current_step, index_rows in zip(
        previous_steps,
        current_steps,
        index_rows_by_record,
        strict=True,
    ):
        ring.append(
            previous_step=previous_step,
            current_step=current_step,
            index_rows=index_rows,
        )
    return ring


def _sample_plan(config: BenchmarkConfig, snapshot: Any) -> SamplePlan:
    entries = list(tuple(snapshot.entries))
    successor_by_record_index = hybrid_profile._compact_replay_successor_by_record_index(entries)
    candidate_entries = tuple(
        hybrid_profile._compact_replay_sample_candidate_entries(
            entries,
            successor_by_record_index=successor_by_record_index,
            num_unroll_steps=2,
            require_successor_window=True,
        )
    )
    group_counts = np.asarray(
        [int(np.asarray(entry.index_rows.action.detach().cpu()).reshape(-1).shape[0]) for entry in candidate_entries],
        dtype=np.int64,
    )
    total_index_rows = int(group_counts.sum())
    requested_sample_rows = int(config.sample_rows)
    if requested_sample_rows < 0:
        raise ValueError("sample_rows must be non-negative")
    sample_row_count = total_index_rows if requested_sample_rows == 0 else requested_sample_rows
    replace = bool(sample_row_count > total_index_rows)
    rng = np.random.default_rng(int(config.seed))
    sampled_flat_rows = rng.choice(
        total_index_rows,
        size=int(sample_row_count),
        replace=replace,
    ).astype(np.int64, copy=False)
    offsets = np.concatenate(
        [np.asarray([0], dtype=np.int64), np.cumsum(group_counts, dtype=np.int64)]
    )
    successor_window_by_record_index: dict[int, tuple[Any, ...] | None] = {}
    for entry in candidate_entries:
        record_index = int(getattr(entry.index_rows, "record_index"))
        successor_window_by_record_index[record_index] = (
            hybrid_profile._successor_index_row_window_for_entry(
                entry,
                successor_by_record_index=successor_by_record_index,
                num_unroll_steps=2,
                allow_terminal_padding=True,
            )
        )
    terminal_flat_rows = hybrid_profile._compact_replay_terminal_flat_rows(
        candidate_entries,
        offsets=offsets,
    )
    terminal_window_flat_rows = hybrid_profile._compact_replay_terminal_window_flat_rows(
        candidate_entries,
        offsets=offsets,
        successor_by_record_index=successor_by_record_index,
        num_unroll_steps=2,
        successor_window_by_record_index=successor_window_by_record_index,
    )
    terminal_row_forced = False
    if requested_sample_rows > 0 and sample_row_count > 0:
        terminal_window_present = bool(
            terminal_window_flat_rows.size > 0
            and np.isin(sampled_flat_rows, terminal_window_flat_rows).any()
        )
        if terminal_flat_rows.size > 0 and not terminal_window_present:
            sampled_flat_rows[0] = int(rng.choice(terminal_flat_rows))
            terminal_row_forced = True
        elif terminal_window_flat_rows.size > 0 and not terminal_window_present:
            sampled_flat_rows[0] = int(rng.choice(terminal_window_flat_rows))
            terminal_row_forced = True
    sampled_group = np.searchsorted(offsets[1:], sampled_flat_rows, side="right")
    sample_position_batches: list[np.ndarray] = []
    source_record_pairs: list[list[int]] = []
    source_record_windows: list[list[int]] = []
    for group_index, entry in enumerate(candidate_entries):
        selected_mask = sampled_group == group_index
        if not bool(selected_mask.any()):
            continue
        sample_position_batches.append(
            np.flatnonzero(selected_mask).astype(np.int64, copy=False)
        )
        source_record_pairs.append(
            [
                int(getattr(entry.index_rows, "record_index")),
                int(getattr(entry.index_rows, "next_record_index")),
            ]
        )
        successor_window = successor_window_by_record_index.get(
            int(getattr(entry.index_rows, "record_index"))
        )
        if successor_window is None:
            source_record_windows.append(source_record_pairs[-1])
        else:
            source_record_windows.append(
                [
                    int(getattr(entry.index_rows, "record_index")),
                    *[
                        int(getattr(successor_rows, "record_index"))
                        for successor_rows in tuple(successor_window)
                    ],
                ]
            )
    sample_position_order = np.concatenate(sample_position_batches).astype(
        np.int64,
        copy=False,
    )
    return SamplePlan(
        candidate_entries=candidate_entries,
        sampled_flat_rows=sampled_flat_rows,
        sampled_group=sampled_group,
        sample_position_order=sample_position_order,
        offsets=offsets,
        group_counts=group_counts,
        source_record_pairs=source_record_pairs,
        source_record_windows=source_record_windows,
        replace=replace,
        sample_row_count=int(sample_row_count),
        total_index_rows=int(total_index_rows),
        terminal_row_forced=bool(terminal_row_forced),
    )


def _group_samples_for_flat_rows(
    *,
    plan: SamplePlan,
    snapshot: Any,
    flat_rows: np.ndarray,
    sampled_group: np.ndarray,
) -> list[dict[str, Any]]:
    successor_by_record_index = hybrid_profile._compact_replay_successor_by_record_index(
        list(tuple(snapshot.entries))
    )
    learner_ready_by_record_index = dict(snapshot.learner_ready_unroll2_by_record_index)
    group_samples: list[dict[str, Any]] = []
    for group_index, entry in enumerate(plan.candidate_entries):
        selected_mask = sampled_group == group_index
        if not bool(selected_mask.any()):
            continue
        record_index = int(getattr(entry.index_rows, "record_index"))
        successor_window = hybrid_profile._successor_index_row_window_for_entry(
            entry,
            successor_by_record_index=successor_by_record_index,
            num_unroll_steps=2,
            allow_terminal_padding=True,
        )
        if successor_window is None:
            raise RuntimeError(f"candidate record {record_index} lost its unroll window")
        local_rows = flat_rows[selected_mask] - plan.offsets[group_index]
        group_samples.append(
            {
                "previous_step": entry.previous_step,
                "current_step": entry.current_step,
                "source_index_row": flat_rows[selected_mask],
                "index_rows": entry.index_rows,
                "local_rows": local_rows.astype(np.int64, copy=False),
                "next_index_rows": successor_window[0],
                "unroll_index_rows_window": successor_window,
                "learner_ready_unroll2_targets": learner_ready_by_record_index.get(
                    record_index
                ),
            }
        )
    return group_samples


def _metadata_for_plan(config: BenchmarkConfig, plan: SamplePlan) -> dict[str, Any]:
    return {
        "source": "benchmark_compact_tensor_native_unroll2_replay",
        "seed": int(config.seed),
        "replace": bool(plan.replace),
        "sample_row_count": int(plan.sample_row_count),
        "stored_pair_count": int(len(plan.candidate_entries)),
        "sampled_pair_count": int(len(plan.source_record_windows)),
        "require_next_targets": True,
        "num_unroll_steps": 2,
        "terminal_unroll_windows_supported": True,
        "source_record_pairs": plan.source_record_pairs,
        "source_record_windows": plan.source_record_windows,
        "sampled_flat_row_checksum": hybrid_profile._int_array_checksum(
            plan.sampled_flat_rows
        ),
        "sample_position_order_checksum": hybrid_profile._int_array_checksum(
            plan.sample_position_order
        ),
        "source_record_pair_checksum": hybrid_profile._ragged_i64_checksum(
            plan.source_record_pairs
        ),
        "source_record_window_checksum": hybrid_profile._ragged_i64_checksum(
            plan.source_record_windows
        ),
        COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_KEY: True,
        COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY: True,
    }


def _build_fixture(config: BenchmarkConfig, torch: Any, device: Any) -> Fixture:
    if int(config.records) < 3:
        raise ValueError("records must be >= 3 for this unroll-2 benchmark")
    if int(config.rows_per_record) <= 0:
        raise ValueError("rows_per_record must be positive")
    if int(config.iters) <= 0:
        raise ValueError("iters must be positive")
    if not 0.0 <= float(config.terminal_rate) <= 1.0:
        raise ValueError("terminal_rate must be in [0, 1]")
    ring = _build_ring(config, torch, device)
    snapshot = ring.snapshot_for_sample()
    plan = _sample_plan(config, snapshot)
    group_samples = _group_samples_for_flat_rows(
        plan=plan,
        snapshot=snapshot,
        flat_rows=plan.sampled_flat_rows,
        sampled_group=plan.sampled_group,
    )
    identity_flat_rows = np.arange(plan.total_index_rows, dtype=np.int64)
    identity_group = np.searchsorted(plan.offsets[1:], identity_flat_rows, side="right")
    identity_group_samples = _group_samples_for_flat_rows(
        plan=plan,
        snapshot=snapshot,
        flat_rows=identity_flat_rows,
        sampled_group=identity_group,
    )
    identity_windows: list[list[int]] = []
    identity_pairs: list[list[int]] = []
    for entry in plan.candidate_entries:
        record_index = int(getattr(entry.index_rows, "record_index"))
        window = hybrid_profile._successor_index_row_window_for_entry(
            entry,
            successor_by_record_index=hybrid_profile._compact_replay_successor_by_record_index(
                list(tuple(snapshot.entries))
            ),
            num_unroll_steps=2,
            allow_terminal_padding=True,
        )
        if window is None:
            raise RuntimeError(f"candidate record {record_index} lost identity window")
        identity_pairs.append(
            [
                int(getattr(entry.index_rows, "record_index")),
                int(getattr(entry.index_rows, "next_record_index")),
            ]
        )
        identity_windows.append(
            [
                int(getattr(entry.index_rows, "record_index")),
                *[int(getattr(successor_rows, "record_index")) for successor_rows in window],
            ]
        )
    metadata = _metadata_for_plan(config, plan)
    identity_metadata = {
        **metadata,
        "sample_row_count": int(plan.total_index_rows),
        "sampled_pair_count": int(len(identity_group_samples)),
        "source_record_pairs": identity_pairs,
        "source_record_windows": identity_windows,
        "sampled_flat_row_checksum": hybrid_profile._int_array_checksum(identity_flat_rows),
        "sample_position_order_checksum": hybrid_profile._int_array_checksum(
            identity_flat_rows
        ),
        "source_record_pair_checksum": hybrid_profile._ragged_i64_checksum(identity_pairs),
        "source_record_window_checksum": hybrid_profile._ragged_i64_checksum(
            identity_windows
        ),
    }
    return Fixture(
        config=config,
        ring=ring,
        snapshot=snapshot,
        sample_plan=plan,
        group_samples=group_samples,
        identity_group_samples=identity_group_samples,
        metadata=metadata,
        identity_metadata=identity_metadata,
        identity_position_order=identity_flat_rows,
    )


def _flat_gather_batch(
    *,
    torch: Any,
    table: CompactMuZeroLearnerBatchV1,
    sampled_flat_rows: np.ndarray,
    metadata: dict[str, Any],
) -> CompactMuZeroLearnerBatchV1:
    device = table.observation.device
    index = torch.as_tensor(sampled_flat_rows, dtype=torch.long, device=device)

    def gather(name: str) -> Any:
        value = getattr(table, name)
        if value is None:
            return None
        return value.index_select(0, index).contiguous()

    sample_weights = np.arange(1, int(sampled_flat_rows.shape[0]) + 1, dtype=np.int64)
    flat_metadata = dict(metadata)
    flat_metadata.update(
        {
            "source": "benchmark_flat_tensor_native_unroll2_gather",
            "compact_muzero_learner_batch_schema_id": table.metadata.get(
                "compact_muzero_learner_batch_schema_id"
            ),
            "sample_schema_id": "curvyzero_compact_muzero_direct_learner_batch/v1",
            "sample_row_count": int(sampled_flat_rows.shape[0]),
            "compact_muzero_learner_batch_rows": int(sampled_flat_rows.shape[0]),
            "compact_muzero_learner_batch_prevalidated": True,
            "compact_muzero_learner_batch_prevalidation_source": (
                "benchmark_flat_tensor_native_unroll2_gather"
            ),
            "compact_muzero_learner_batch_sample_order": "rng",
            "compact_muzero_learner_batch_preserves_sample_order": True,
            "compact_muzero_learner_batch_sample_row_checksum": int(
                np.asarray(sampled_flat_rows, dtype=np.int64).dot(sample_weights)
            ),
            "resident_grouped_device_replay_learner_batch": False,
            "resident_grouped_device_direct_write_learner_batch": False,
            "device_replay_index_rows_sample": True,
            "device_replay_index_rows_sample_all": True,
            "compact_muzero_learner_host_fallback_allowed": False,
            "host_observation_fallback_allowed": False,
            "flat_tensor_native_prepacked_table_rows": int(table.observation.shape[0]),
            "flat_tensor_native_sampled_rows": int(sampled_flat_rows.shape[0]),
            "flat_tensor_native_gather_only": True,
        }
    )
    return CompactMuZeroLearnerBatchV1(
        metadata=flat_metadata,
        observation=gather("observation"),
        action=gather("action"),
        action_mask=gather("action_mask"),
        target_reward=gather("target_reward"),
        target_value=gather("target_value"),
        target_policy=gather("target_policy"),
        target_reward_mask=gather("target_reward_mask"),
        target_value_mask=gather("target_value_mask"),
        target_policy_mask=gather("target_policy_mask"),
        action_valid_mask=gather("action_valid_mask"),
        weights=gather("weights"),
        source_sample_batch=None,
        next_action_mask=gather("next_action_mask"),
        done=gather("done"),
        terminated=gather("terminated"),
        truncated=gather("truncated"),
    )


def _tensor_equal(torch: Any, left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    if left.device != right.device:
        right = right.to(device=left.device)
    return bool(torch.equal(left, right))


def _batch_equalities(torch: Any, left: Any, right: Any) -> dict[str, bool]:
    return {
        field: _tensor_equal(torch, getattr(left, field), getattr(right, field))
        for field in LEARNER_TENSOR_FIELDS
    }


def _metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    return metadata.get(key, default)


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    import torch

    device = _resolve_torch_device(torch, config.device)
    fixture = _build_fixture(config, torch, device)
    plan = fixture.sample_plan

    def ring_path() -> Any:
        result = fixture.ring.sample_from_snapshot(
            fixture.snapshot,
            seed=int(config.seed),
            sample_batch_size=int(config.sample_rows),
            require_next_targets=True,
            num_unroll_steps=2,
            build_compact_muzero_learner_batch=True,
            compact_muzero_learner_batch_only=True,
        )
        learner_batch = result.get("learner_batch")
        if learner_batch is None:
            raise RuntimeError("ring sample did not return a learner batch")
        return learner_batch

    def grouped_path() -> Any:
        return _build_compact_resident_grouped_device_learner_batch_fast(
            group_samples=fixture.group_samples,
            metadata=fixture.metadata,
            sample_position_order=plan.sample_position_order,
        )

    prepack_started = time.perf_counter()
    prepack_process_started = int(time.process_time_ns())
    prepack_table = _build_compact_resident_grouped_device_learner_batch_fast(
        group_samples=fixture.identity_group_samples,
        metadata=fixture.identity_metadata,
        sample_position_order=fixture.identity_position_order,
    )
    prepack_sync_sec = _sync_device(torch, device)
    prepack_sec = time.perf_counter() - prepack_started
    prepack_process_cpu_ns = int(time.process_time_ns()) - prepack_process_started

    def flat_path() -> Any:
        return _flat_gather_batch(
            torch=torch,
            table=prepack_table,
            sampled_flat_rows=plan.sampled_flat_rows,
            metadata=fixture.metadata,
        )

    ring_timing, ring_batch = _timed_path(
        name="current_replay_ring_sample_build",
        rows=plan.sample_row_count,
        iters=int(config.iters),
        torch=torch,
        device=device,
        fn=ring_path,
    )
    fixture.ring.update_store_metadata(
        {
            COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_KEY: True,
            COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY: True,
            COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_KEY: True,
            COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_REQUESTED_KEY: True,
            COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_SELECTED_MAINTAINED_GATHER_KEY: True,
            "benchmark_compact_tensor_native_unroll2_replay": True,
        }
    )
    tensor_native_snapshot = fixture.ring.snapshot_for_sample()
    tensor_native_table_count = len(
        dict(tensor_native_snapshot.tensor_native_unroll2_table_by_record_index)
    )

    def real_tensor_native_path() -> Any:
        result = fixture.ring.sample_from_snapshot(
            tensor_native_snapshot,
            seed=int(config.seed),
            sample_batch_size=int(config.sample_rows),
            require_next_targets=True,
            num_unroll_steps=2,
            build_compact_muzero_learner_batch=True,
            compact_muzero_learner_batch_only=True,
        )
        learner_batch = result.get("learner_batch")
        if learner_batch is None:
            raise RuntimeError("tensor-native ring sample did not return a learner batch")
        return learner_batch

    real_tensor_native_timing, real_tensor_native_batch = _timed_path(
        name="real_tensor_native_replay_sample_build",
        rows=plan.sample_row_count,
        iters=int(config.iters),
        torch=torch,
        device=device,
        fn=real_tensor_native_path,
    )
    grouped_timing, grouped_batch = _timed_path(
        name="resident_grouped_learner_ready_unroll2",
        rows=plan.sample_row_count,
        iters=int(config.iters),
        torch=torch,
        device=device,
        fn=grouped_path,
    )
    flat_timing, flat_batch = _timed_path(
        name="flat_tensor_native_gather",
        rows=plan.sample_row_count,
        iters=int(config.iters),
        torch=torch,
        device=device,
        fn=flat_path,
    )

    ring_vs_grouped = _batch_equalities(torch, ring_batch, grouped_batch)
    real_tensor_native_vs_grouped = _batch_equalities(
        torch,
        real_tensor_native_batch,
        grouped_batch,
    )
    grouped_vs_flat = _batch_equalities(torch, grouped_batch, flat_batch)
    ring_vs_grouped_equal = all(ring_vs_grouped.values())
    real_tensor_native_vs_grouped_equal = all(real_tensor_native_vs_grouped.values())
    grouped_vs_flat_equal = all(grouped_vs_flat.values())
    all_equal = bool(
        ring_vs_grouped_equal
        and real_tensor_native_vs_grouped_equal
        and grouped_vs_flat_equal
    )
    ring_metadata = dict(getattr(ring_batch, "metadata", {}) or {})
    real_tensor_native_metadata = dict(
        getattr(real_tensor_native_batch, "metadata", {}) or {}
    )
    grouped_metadata = dict(getattr(grouped_batch, "metadata", {}) or {})
    flat_metadata = dict(getattr(flat_batch, "metadata", {}) or {})
    ring_median = float(ring_timing["wall_sec"]["median"])
    real_tensor_native_median = float(real_tensor_native_timing["wall_sec"]["median"])
    grouped_median = float(grouped_timing["wall_sec"]["median"])
    flat_median = float(flat_timing["wall_sec"]["median"])
    proof = {
        "all_equal": bool(all_equal),
        "ring_vs_grouped_equal": bool(ring_vs_grouped_equal),
        "real_tensor_native_vs_grouped_equal": bool(real_tensor_native_vs_grouped_equal),
        "grouped_vs_flat_equal": bool(grouped_vs_flat_equal),
        "ring_vs_grouped_fields": ring_vs_grouped,
        "real_tensor_native_vs_grouped_fields": real_tensor_native_vs_grouped,
        "grouped_vs_flat_fields": grouped_vs_flat,
        "sampled_flat_row_checksum": int(
            hybrid_profile._int_array_checksum(plan.sampled_flat_rows)
        ),
        "ring_sampled_flat_row_checksum": int(
            _metadata_value(ring_metadata, "sampled_flat_row_checksum", 0) or 0
        ),
        "source_record_window_checksum": int(
            hybrid_profile._ragged_i64_checksum(plan.source_record_windows)
        ),
        "ring_source_record_window_checksum": int(
            _metadata_value(ring_metadata, "source_record_window_checksum", 0) or 0
        ),
        "learner_batch_sample_row_checksum": int(
            _metadata_value(grouped_metadata, "compact_muzero_learner_batch_sample_row_checksum", 0)
            or 0
        ),
        "ring_learner_batch_sample_row_checksum": int(
            _metadata_value(ring_metadata, "compact_muzero_learner_batch_sample_row_checksum", 0)
            or 0
        ),
        "real_tensor_native_learner_batch_sample_row_checksum": int(
            _metadata_value(
                real_tensor_native_metadata,
                "compact_muzero_learner_batch_sample_row_checksum",
                0,
            )
            or 0
        ),
        "flat_learner_batch_sample_row_checksum": int(
            _metadata_value(flat_metadata, "compact_muzero_learner_batch_sample_row_checksum", 0)
            or 0
        ),
        "num_unroll_steps": int(_metadata_value(grouped_metadata, "num_unroll_steps", 0) or 0),
        "compact_muzero_learner_num_unroll_steps": int(
            _metadata_value(grouped_metadata, "compact_muzero_learner_num_unroll_steps", 0)
            or 0
        ),
        "host_fallback_allowed": bool(
            _metadata_value(grouped_metadata, "compact_muzero_learner_host_fallback_allowed", True)
        ),
        "device_replay_index_rows_sample_all": bool(
            _metadata_value(grouped_metadata, "device_replay_index_rows_sample_all", False)
        ),
        "ring_cache_requested": bool(
            _metadata_value(
                ring_metadata,
                COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY,
                False,
            )
        ),
        "ring_cache_used": bool(
            _metadata_value(
                ring_metadata,
                COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_USED_KEY,
                False,
            )
        ),
        "ring_cache_call_count": int(
            _metadata_value(
                ring_metadata,
                COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_CALL_COUNT_KEY,
                0,
            )
            or 0
        ),
        "grouped_cache_requested": bool(
            _metadata_value(
                grouped_metadata,
                COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY,
                False,
            )
        ),
        "grouped_cache_used": bool(
            _metadata_value(
                grouped_metadata,
                COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_USED_KEY,
                False,
            )
        ),
        "grouped_cache_call_count": int(
            _metadata_value(
                grouped_metadata,
                COMPACT_MUZERO_LEARNER_BATCH_LEARNER_READY_UNROLL2_CACHE_CALL_COUNT_KEY,
                0,
            )
            or 0
        ),
        "real_tensor_native_requested": bool(
            _metadata_value(
                real_tensor_native_metadata,
                COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_REQUESTED_KEY,
                False,
            )
        ),
        "real_tensor_native_used": bool(
            _metadata_value(
                real_tensor_native_metadata,
                COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_USED_KEY,
                False,
            )
        ),
        "real_tensor_native_call_count": int(
            _metadata_value(
                real_tensor_native_metadata,
                COMPACT_MUZERO_LEARNER_BATCH_TENSOR_NATIVE_REPLAY_CALL_COUNT_KEY,
                0,
            )
            or 0
        ),
        "real_tensor_native_table_rows": int(
            _metadata_value(
                real_tensor_native_metadata,
                "compact_muzero_learner_batch_tensor_native_replay_table_rows",
                0,
            )
            or 0
        ),
        "real_tensor_native_impl": str(
            _metadata_value(
                real_tensor_native_metadata,
                "compact_muzero_learner_batch_tensor_native_replay_impl",
                "none",
            )
        ),
        "real_tensor_native_table_source": str(
            _metadata_value(
                real_tensor_native_metadata,
                "compact_muzero_learner_batch_tensor_native_replay_table_source",
                "none",
            )
        ),
        "real_tensor_native_table_reused_record_count": int(
            _metadata_value(
                real_tensor_native_metadata,
                (
                    "compact_muzero_learner_batch_tensor_native_replay_"
                    "table_reused_record_count"
                ),
                0,
            )
            or 0
        ),
        "real_tensor_native_table_missing_record_count": int(
            _metadata_value(
                real_tensor_native_metadata,
                (
                    "compact_muzero_learner_batch_tensor_native_replay_"
                    "table_missing_record_count"
                ),
                0,
            )
            or 0
        ),
        "real_tensor_native_maintained_table_count": int(tensor_native_table_count),
        "real_tensor_native_direct_fast_metadata_path_used": bool(
            _metadata_value(
                real_tensor_native_metadata,
                (
                    "compact_rollout_slab_sample_gate_tensor_native_direct_"
                    "fast_metadata_path_used"
                ),
                False,
            )
        ),
        "sample_order_preserved": bool(
            _metadata_value(
                grouped_metadata,
                "compact_muzero_learner_batch_preserves_sample_order",
                False,
            )
        ),
        "terminal_row_forced": bool(plan.terminal_row_forced),
    }
    proof["checksum_match"] = bool(
        proof["sampled_flat_row_checksum"] == proof["ring_sampled_flat_row_checksum"]
        and proof["source_record_window_checksum"]
        == proof["ring_source_record_window_checksum"]
        and proof["learner_batch_sample_row_checksum"]
        == proof["ring_learner_batch_sample_row_checksum"]
        == proof["real_tensor_native_learner_batch_sample_row_checksum"]
        == proof["flat_learner_batch_sample_row_checksum"]
    )
    proof["required_pass"] = bool(
        proof["all_equal"]
        and proof["checksum_match"]
        and proof["num_unroll_steps"] == 2
        and proof["compact_muzero_learner_num_unroll_steps"] == 2
        and proof["host_fallback_allowed"] is False
        and proof["device_replay_index_rows_sample_all"] is True
        and proof["ring_cache_requested"] is True
        and proof["ring_cache_used"] is True
        and proof["grouped_cache_requested"] is True
        and proof["grouped_cache_used"] is True
        and proof["real_tensor_native_requested"] is True
        and proof["real_tensor_native_used"] is True
        and proof["real_tensor_native_call_count"] == 1
        and proof["real_tensor_native_impl"]
        == "selected_maintained_record_table_gather_v1"
        and proof["real_tensor_native_table_source"]
        == "selected_maintained_record_table_v1"
        and proof["real_tensor_native_direct_fast_metadata_path_used"] is True
        and proof["real_tensor_native_table_missing_record_count"] == 0
        and proof["real_tensor_native_table_reused_record_count"]
        == len(plan.source_record_windows)
        and proof["real_tensor_native_maintained_table_count"] == len(plan.candidate_entries)
        and proof["sample_order_preserved"] is True
    )

    return {
        "schema_id": "curvyzero_benchmark_compact_tensor_native_unroll2_replay/v1",
        "config": {
            "records": int(config.records),
            "rows_per_record": int(config.rows_per_record),
            "sample_rows": int(config.sample_rows),
            "iters": int(config.iters),
            "terminal_rate": float(config.terminal_rate),
            "seed": int(config.seed),
            "device": str(torch.device(device)),
        },
        "fixture": {
            "stored_records": int(fixture.ring.entry_count),
            "candidate_records": int(len(plan.candidate_entries)),
            "total_index_rows": int(plan.total_index_rows),
            "sample_row_count": int(plan.sample_row_count),
            "replace": bool(plan.replace),
            "sampled_pair_count": int(len(plan.source_record_windows)),
            "learner_ready_cache_available_records": int(
                len(dict(fixture.snapshot.learner_ready_unroll2_by_record_index))
            ),
            "tensor_native_maintained_table_records": int(tensor_native_table_count),
        },
        "paths": {
            "current_replay_ring_sample_build": ring_timing,
            "real_tensor_native_replay_sample_build": real_tensor_native_timing,
            "resident_grouped_learner_ready_unroll2": grouped_timing,
            "flat_tensor_native_prepack": {
                "name": "flat_tensor_native_prepack",
                "iters": 1,
                "rows": int(plan.total_index_rows),
                "wall_sec": float(prepack_sec),
                "rows_per_sec": float(plan.total_index_rows / prepack_sec)
                if prepack_sec > 0
                else 0.0,
                "process_cpu_time_ns": int(prepack_process_cpu_ns),
                "cuda_sync_sec": float(prepack_sync_sec),
            },
            "flat_tensor_native_gather": flat_timing,
        },
        "speedup_vs_current_median": {
            "real_tensor_native_replay_sample_build": (
                float(ring_median / real_tensor_native_median)
                if real_tensor_native_median > 0
                else 0.0
            ),
            "resident_grouped_learner_ready_unroll2": (
                float(ring_median / grouped_median) if grouped_median > 0 else 0.0
            ),
            "flat_tensor_native_gather": (
                float(ring_median / flat_median) if flat_median > 0 else 0.0
            ),
        },
        "proof": proof,
    }


def _parse_args() -> tuple[BenchmarkConfig, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=BenchmarkConfig.records)
    parser.add_argument(
        "--rows-per-record",
        type=int,
        default=BenchmarkConfig.rows_per_record,
    )
    parser.add_argument("--sample-rows", type=int, default=BenchmarkConfig.sample_rows)
    parser.add_argument("--iters", type=int, default=BenchmarkConfig.iters)
    parser.add_argument("--terminal-rate", type=float, default=BenchmarkConfig.terminal_rate)
    parser.add_argument("--seed", type=int, default=BenchmarkConfig.seed)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=BenchmarkConfig.device)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    return (
        BenchmarkConfig(
            records=int(args.records),
            rows_per_record=int(args.rows_per_record),
            sample_rows=int(args.sample_rows),
            iters=int(args.iters),
            terminal_rate=float(args.terminal_rate),
            seed=int(args.seed),
            device=str(args.device),
        ),
        args.json_output,
    )


def main() -> int:
    config, json_output = _parse_args()
    result = run_benchmark(config)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if json_output is not None:
        Path(json_output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if bool(result["proof"]["required_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
