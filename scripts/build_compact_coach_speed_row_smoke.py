#!/usr/bin/env python3
"""Build a local compact-owned Coach speed-row smoke artifact.

This is a sibling evidence producer, not the hybrid profile manifest runner.
The raw measured loop is preserved inside the result as support evidence while
the manifest/result pair use the non-profile Coach speed-row schemas.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT,
)
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_MODEL_IDENTITY_SCOPE_SUPPORT_ONLY,
)
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
)
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_SPEED_ROW_MANIFEST_SCHEMA_ID,
)
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_SPEED_ROW_PRODUCER_SCHEMA_ID,
)
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_SPEED_ROW_RESULT_SCHEMA_ID,
)
from curvyzero.training.compact_coach_speed_row import (
    compact_coach_speed_row_evidence_ref,
)
from curvyzero.training.compact_coach_speed_row import (
    save_compact_coach_speed_row_evidence_v1,
)
from curvyzero.training.compact_death_terminal_contract import (
    CompactDeathTerminalContractError,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_normal_collision_death_contract_from_profile_result_v1,
)
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
)
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_model_state_digest_v1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_REPLAY_APPEND_TRANSPORT_KINDS,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1,
)
from curvyzero.training.compact_owned_loop import COMPACT_MODEL_STATE_TRANSPORT_KINDS
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS,
)
from curvyzero.training.compact_owned_loop import COMPACT_SAMPLE_LEARNER_WORKER_KINDS
from curvyzero.training.compact_owned_loop import CompactOwnedLoopV1
from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import _host_only_clone
from curvyzero.training.compact_owned_loop import _load_model_state_snapshot_file
from curvyzero.training.compact_owned_loop import _move_tensors_to_device
from curvyzero.training.compact_owned_loop import _write_model_state_snapshot_file
from curvyzero.training.compact_owned_loop import compact_owned_loop_replay_store_metadata
from curvyzero.training.compact_muzero_learner import (
    COMPACT_MUZERO_LEARNER_BATCH_SCHEMA_ID,
)
from curvyzero.training.compact_owned_trainer import CompactOwnedTrainerConfigV1
from curvyzero.training.compact_owned_trainer import CompactOwnedTrainerV1
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_OWNER_REF_DIGEST_DEFERRED_TO_SEARCH_REFRESH_KEY,
)
from curvyzero.training.compact_owner_search_service import (
    CompactLazyInlineBackgroundOwnerSearchSlabProxyV1,
)
from curvyzero.training.compact_owner_search_service import (
    CompactLazyInlineOwnerSearchSlabProxyV1,
)
from curvyzero.training.compact_owner_search_service import CompactLazyOwnerSearchSlabProxyV1
from curvyzero.training.compact_owner_search_service import (
    CompactLazyThreadedOwnerSearchSlabProxyV1,
)
from curvyzero.training.compact_owner_search_service import (
    _contains_cuda_tensor as _owner_search_contains_cuda_tensor,
)
from curvyzero.training.compact_owner_search_service import (
    _compact_batch_from_root_batch,
)
from curvyzero.training.compact_owner_search_service import (
    _device_replay_payload_from_search_result,
)
from curvyzero.training.compact_owner_search_service import _resident_root_device
from curvyzero.training.compact_owner_search_service import (
    build_compact_resident_shared_memory_root_provider_v1,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_device_replay_index_rows_v1_from_payload,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_replay_index_rows_v1_from_search_result,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_transition_outcome_v1_from_next_root_batch,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION,
)
from curvyzero.training.compact_rollout_slab import (
    CompactOwnerSearchReplayAppendEntryV1,
)
from curvyzero.training.compact_rollout_slab import (
    CompactOwnerSearchReplayAppendIndexEntryV1,
)
from curvyzero.training.compact_rollout_slab import CompactRolloutSlab
from curvyzero.training.compact_rollout_slab import CompactOwnerSearchDirectStepperV1
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import (
    COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import (
    COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import CompactDeviceSearchReplayPayloadV1
from curvyzero.training.compact_search_service import CompactSearchActionStepV1
from curvyzero.training.compact_search_service import (
    compact_search_action_step_v1_from_result,
)
from curvyzero.training.compact_search_service import compact_search_array_digest_v1
from curvyzero.training.compact_search_service import (
    compact_search_deferred_replay_payload_digest_v1,
)
from curvyzero.training.compact_search_service import (
    validate_compact_device_search_two_phase_payload_v1,
)
from curvyzero.training.source_state_batched_observation_profile import (
    SourceStateBatchedRenderResult,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    HYBRID_STACK_STORAGE_DTYPE_UINT8,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    RESIDENT_REPLAY_SNAPSHOT_MODE_FULL_STACK,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    RESIDENT_REPLAY_SNAPSHOT_MODE_LATEST_FRAME_HISTORY,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    HybridObservationProfileConfig,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _CompactReplayRingV1,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _CompactReplayRingEntry,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    CompactReplayColumnarAppendRecordV1,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _build_compact_muzero_process_worker_learner,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _compact_policy_refresh_metadata_seen,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _cpu_clone_model_for_process_worker,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _refresh_compact_rollout_slab_search_from_learner,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    run_hybrid_observation_profile,
)
from curvyzero.training.compact_trainer_checkpoint import (
    load_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    restore_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_torch_search_service import CompactTorchCompileConfig
from curvyzero.training.compact_torch_search_service import (
    COMPACT_TORCH_INITIAL_INFERENCE_MODES,
    COMPACT_TORCH_MODEL_COMPILE_MODES,
    COMPACT_TORCH_MEMORY_FORMATS,
)
from curvyzero.training.compact_torch_search_service import COMPACT_TORCH_TIMING_MODES
from curvyzero.training.compact_torch_search_service import CompactTorchSearchServiceV1
from curvyzero.training.fixed_shape_batched_search_owner import (
    FixedShapeBatchedSearchOwnerV0,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_coach_speed_row_results")
DEFAULT_RUN_ID = "optimizer-compact-coach-speed-row-smoke-20260530"
SPEED_CURRENCY = "compact_trainer_env_steps_per_sec"
OPT104_BASELINE_ENV_STEPS_PER_SEC = 12689.381637
WHOLE_OWNER_BUFFER_REPLAY_CEILING_TARGET_MULTIPLIER = 2.0
WHOLE_OWNER_BUFFER_REPLAY_CEILING_SCHEMA_ID = (
    "curvyzero_compact_whole_owner_buffer_replay_ceiling/v1"
)
WHOLE_OWNER_BUFFER_REPLAY_CEILING_SPEED_CURRENCY = "local_projection_no_speed"
ROW_ID = "001"
UNROLL2_SPECIALIZED_BUILDER_KEY = "compact_muzero_learner_batch_unroll2_specialized_builder"
UNROLL2_SPECIALIZED_BUILDER_REQUESTED_KEY = (
    "compact_muzero_learner_batch_unroll2_specialized_builder_requested"
)
LEARNER_READY_UNROLL2_CACHE_KEY = "compact_muzero_learner_batch_learner_ready_unroll2_cache"
LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY = (
    "compact_muzero_learner_batch_learner_ready_unroll2_cache_requested"
)
TENSOR_NATIVE_REPLAY_KEY = "compact_muzero_learner_batch_tensor_native_replay"
TENSOR_NATIVE_REPLAY_REQUESTED_KEY = "compact_muzero_learner_batch_tensor_native_replay_requested"
FIXED_SOA_REPLAY_REQUESTED_KEY = "compact_replay_fixed_soa_unroll2_buffer_requested"
FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY = (
    "compact_replay_fixed_soa_learner_batch_handle_ring"
)
FIXED_SOA_LEARNER_BATCH_HANDLE_RING_REQUESTED_KEY = (
    "compact_replay_fixed_soa_learner_batch_handle_ring_requested"
)
FIXED_SOA_LOCALITY_SAMPLE_GROUP_SIZE_KEY = "compact_replay_fixed_soa_locality_sample_group_size"
OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX = "compact_owned_loop_learner_resident_batch_handle"
OWNER_SEARCH_RESIDENT_BATCH_HANDLE_PREFIX = (
    "compact_owner_search_learner_resident_batch_handle"
)
SEARCH_SERVICE_DEVICE_TARGET = "device_target"
SEARCH_SERVICE_COMPACT_TORCH = "compact_torch_search_service"
SEARCH_SERVICE_FIXED_SHAPE = "fixed_shape_search_owner"
SEARCH_SERVICE_OWNER_SEARCH_SLAB_PROXY = "owner_search_slab_proxy"
SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY = "owner_search_inline_proxy"
SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY = "owner_search_inline_background_proxy"
SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY = "owner_search_threaded_proxy"
SEARCH_SERVICE_CHOICES = (
    SEARCH_SERVICE_DEVICE_TARGET,
    SEARCH_SERVICE_COMPACT_TORCH,
    SEARCH_SERVICE_FIXED_SHAPE,
    SEARCH_SERVICE_OWNER_SEARCH_SLAB_PROXY,
    SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY,
    SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY,
    SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY,
)
OWNER_SEARCH_INNER_SEARCH_SERVICE_CHOICES = (
    SEARCH_SERVICE_COMPACT_TORCH,
    SEARCH_SERVICE_FIXED_SHAPE,
)
OWNER_SEARCH_LEARNER_COMPACT_MUZERO = "compact_muzero"
OWNER_SEARCH_LEARNER_MOCK_FAST = "mock_fast"
OWNER_SEARCH_LEARNER_CHOICES = (
    OWNER_SEARCH_LEARNER_COMPACT_MUZERO,
    OWNER_SEARCH_LEARNER_MOCK_FAST,
)
OWNER_SEARCH_MODEL_STATE_TRANSPORT_SHARED_MODEL_V1 = "shared_model_state_v1"
COMPACT_MUZERO_DIRECT_LEARNER_BATCH_SCHEMA_ID = "curvyzero_compact_muzero_direct_learner_batch/v1"


class _GpuUtilizationSamplerV1:
    """Small default-off nvidia-smi sampler for hardware bottleneck checks."""

    def __init__(self, *, enabled: bool, interval_sec: float) -> None:
        self.enabled = bool(enabled)
        self.interval_sec = float(interval_sec)
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled or self.interval_sec <= 0.0:
            return
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="compact-coach-speed-row-gpu-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec + 1.0))
        gpu_util = [
            float(sample["gpu_util_percent"])
            for sample in self.samples
            if sample.get("gpu_util_percent") is not None
        ]
        memory_util = [
            float(sample["memory_util_percent"])
            for sample in self.samples
            if sample.get("memory_util_percent") is not None
        ]
        memory_used = [
            float(sample["memory_used_mib"])
            for sample in self.samples
            if sample.get("memory_used_mib") is not None
        ]
        power_draw = [
            float(sample["power_draw_w"])
            for sample in self.samples
            if sample.get("power_draw_w") is not None
        ]
        gpu_util_over_50 = [value for value in gpu_util if value >= 50.0]
        gpu_util_over_80 = [value for value in gpu_util if value >= 80.0]
        return {
            "enabled": self.enabled,
            "interval_sec": self.interval_sec,
            "sample_count": len(self.samples),
            "first_sample": self.samples[0] if self.samples else None,
            "last_sample": self.samples[-1] if self.samples else None,
            "gpu_name": str(self.samples[-1].get("name", "")) if self.samples else "",
            "max_gpu_util_percent": max(gpu_util) if gpu_util else None,
            "mean_gpu_util_percent": (sum(gpu_util) / len(gpu_util)) if gpu_util else None,
            "gpu_util_nonzero_sample_count": sum(1 for value in gpu_util if value > 0.0),
            "gpu_util_over_50_sample_count": len(gpu_util_over_50),
            "gpu_util_over_80_sample_count": len(gpu_util_over_80),
            "max_memory_util_percent": max(memory_util) if memory_util else None,
            "max_memory_used_mib": max(memory_used) if memory_used else None,
            "max_power_draw_w": max(power_draw) if power_draw else None,
            "errors": list(self.errors),
        }

    def _sample_loop(self) -> None:
        self._sample_gpu()
        assert self._stop_event is not None
        while not self._stop_event.wait(self.interval_sec):
            self._sample_gpu()

    def _sample_gpu(self) -> None:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,"
                    "memory.used,memory.total,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as exc:  # pragma: no cover - remote diagnostic only.
            if len(self.errors) < 10:
                self.errors.append(f"{type(exc).__name__}: {exc}")
            return
        if proc.returncode != 0:
            if len(self.errors) < 10:
                self.errors.append(proc.stderr.strip() or f"nvidia-smi rc={proc.returncode}")
            return
        for line in proc.stdout.splitlines():
            sample = _parse_nvidia_smi_utilization_row_v1(line)
            if sample is not None:
                self.samples.append(sample)


def _parse_nvidia_smi_utilization_row_v1(line: str) -> dict[str, Any] | None:
    parts = [part.strip() for part in str(line).split(",")]
    if len(parts) < 7:
        return None
    return {
        "timestamp": parts[0],
        "name": parts[1],
        "gpu_util_percent": _parse_float_or_none(parts[2]),
        "memory_util_percent": _parse_float_or_none(parts[3]),
        "memory_used_mib": _parse_float_or_none(parts[4]),
        "memory_total_mib": _parse_float_or_none(parts[5]),
        "power_draw_w": _parse_float_or_none(parts[6]),
    }


def _parse_float_or_none(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--unified-lifecycle-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--actor-count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--death-mode",
        choices=tuple(vector_runtime.DEATH_MODES),
        default=vector_runtime.DEATH_MODE_PROFILE_NO_DEATH,
        help=(
            "Runtime death surface measured by this speed row. Borrowed A1 "
            "speed rows default to profile_no_death and cannot be used as "
            "normal-death speed evidence."
        ),
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=2,
        help=(
            "Replay sample rows per train request. For owner_search_slab_proxy, "
            "0 means sample every eligible replay row; use that only for proof "
            "rows, not same-work speed comparisons."
        ),
    )
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--replay-pair-capacity", type=int, default=16)
    parser.add_argument("--learner-train-steps", type=int, default=1)
    parser.add_argument(
        "--learner-num-unroll-steps",
        type=int,
        default=1,
        help=(
            "Compact MuZero learner unroll length. Normal-death speed rows use "
            "at least 2 so terminal no-bootstrap windows are actually exercised."
        ),
    )
    parser.add_argument(
        "--policy-refresh-interval",
        type=int,
        default=1,
        help=(
            "Refresh the compact Torch search worker every N learner updates. "
            "Final consumed search/replay metadata is still required."
        ),
    )
    parser.add_argument(
        "--compact-owned-loop-deferred-learner",
        action="store_true",
        help=(
            "Submit compact-owned learner work to a one-worker deferred lane. "
            "Replay sampling stays synchronous and search refresh consumes only "
            "completed learner updates."
        ),
    )
    parser.add_argument(
        "--compact-owned-loop-deferred-sample-learner",
        action="store_true",
        help=(
            "Submit replay sampling and the learner update together to a deferred "
            "lane so actor collection can continue while that work runs."
        ),
    )
    parser.add_argument(
        "--compact-owned-loop-deferred-sample-learner-max-pending",
        type=int,
        default=1,
        help="Maximum queued replay-sample-plus-learner jobs for the staged lane.",
    )
    parser.add_argument(
        "--compact-owned-loop-sample-learner-worker-kind",
        choices=COMPACT_SAMPLE_LEARNER_WORKER_KINDS,
        default=COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD,
        help=(
            "Worker adapter for deferred replay-sample-plus-learner jobs. "
            "local_process is a separate OS process but not a separate GPU."
        ),
    )
    parser.add_argument(
        "--compact-owned-loop-deferred-sample-learner-replay-append-transport-kind",
        choices=COMPACT_REPLAY_APPEND_TRANSPORT_KINDS,
        default=COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1,
        help=(
            "Replay append payload for deferred sample+learner workers. "
            "scalar_ref_v1 strips observation tensors and requires worker-side "
            "observation reconstruction."
        ),
    )
    parser.add_argument(
        "--compact-owned-loop-deferred-sample-learner-model-state-transport-kind",
        choices=COMPACT_MODEL_STATE_TRANSPORT_KINDS,
        default=COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
        help=(
            "How deferred sample+learner workers publish model state for search "
            "refresh. snapshot_file_v1 returns a small file handle instead of "
            "the full state dict in the Python result."
        ),
    )
    parser.add_argument(
        "--compact-owned-loop-fused-learner-batch",
        action="store_true",
        help=(
            "Build a direct resident replay-to-compact-MuZero learner batch "
            "instead of materializing a full sample batch for the learner."
        ),
    )
    parser.add_argument(
        "--compact-muzero-learner-batch-unroll2-specialized-builder",
        action="store_true",
        help=(
            "Default-off diagnostic: use the guarded unroll-2 specialized "
            "builder inside the direct resident fused learner-batch path."
        ),
    )
    parser.add_argument(
        "--compact-muzero-learner-batch-learner-ready-unroll2-cache",
        action="store_true",
        help=(
            "Default-off structural path: select learner-ready unroll-2 targets "
            "cached on resident replay entries inside the fused learner-batch path."
        ),
    )
    parser.add_argument(
        "--compact-muzero-learner-batch-tensor-native-replay",
        action="store_true",
        help=(
            "Default-off structural path: gather learner batches from maintained "
            "tensor-native replay table state inside the fused learner-batch path."
        ),
    )
    parser.add_argument(
        "--compact-owned-lean-trainer-step",
        action="store_true",
        help=(
            "Drive the compact trainer step directly instead of using the "
            "hybrid observation profile runner as the outer measured loop."
        ),
    )
    parser.add_argument(
        "--compact-owned-lean-profile-oracle",
        action="store_true",
        help=(
            "When lean trainer mode is enabled, also run the legacy profile "
            "runner as a correctness oracle. Oracle wall time is not included "
            "in the lean speed denominator."
        ),
    )
    parser.add_argument(
        "--compact-profile-bounded-diagnostics",
        action="store_true",
        help=(
            "Use bounded diagnostic bookkeeping for long stability rows: keep "
            "scalar proof fields and per-step training work, but do not retain "
            "the slab's full committed-row history or embed the full source "
            "profile payload in the final compact row."
        ),
    )
    parser.add_argument(
        "--compact-profile-cuda-sync-timing-diagnostics",
        action="store_true",
        help=(
            "Diagnostic-only: insert CUDA synchronizes around sample/learner "
            "timing blocks so long stability rows can attribute async GPU work."
        ),
    )
    parser.add_argument(
        "--compact-profile-runtime-step-timing-diagnostics",
        action="store_true",
        help=(
            "Diagnostic-only: record measured-step runtime envelope stats "
            "without requiring CUDA synchronization probes."
        ),
    )
    parser.add_argument(
        "--hybrid-persistent-compact-render-state-buffer",
        action="store_true",
        help=(
            "Use compact persistent renderer buffers for actor render-state handoff. "
            "Default-off OPT-084 actor/render-state wall probe."
        ),
    )
    parser.add_argument(
        "--hybrid-borrow-single-actor-render-state",
        action="store_true",
        help=(
            "Borrow the single actor env render state directly instead of copying "
            "parent render-state rows. Requires actor_count=1; terminal snapshot "
            "copies are allowed only on the proven normal-death surface."
        ),
    )
    parser.add_argument(
        "--learner-device",
        default="cpu",
        choices=("cpu", "cuda", "mps", "auto"),
    )
    parser.add_argument(
        "--gpu-utilization-sampling",
        action="store_true",
        help="Sample nvidia-smi during the measured speed-row profile.",
    )
    parser.add_argument(
        "--gpu-utilization-sample-interval-sec",
        type=float,
        default=1.0,
        help="Seconds between nvidia-smi utilization samples when sampling is enabled.",
    )
    parser.add_argument(
        "--load-unified-lifecycle-checkpoint",
        action="store_true",
        help=(
            "Load the compact checkpoint named by the unified lifecycle report "
            "and use its model in the measured compact learner gate."
        ),
    )
    parser.add_argument(
        "--omit-loaded-checkpoint-identity-path",
        action="store_true",
        help=(
            "Omit compact_checkpoint_path from loaded identity. Use this for "
            "remote rows that stage the same checkpoint under a different path."
        ),
    )
    parser.add_argument("--num-simulations", type=int, default=1)
    parser.add_argument(
        "--search-service-kind",
        choices=SEARCH_SERVICE_CHOICES,
        default=SEARCH_SERVICE_DEVICE_TARGET,
        help=(
            "Compact search backend for the measured speed row. The default "
            "preserves the historical device-target trainer smoke; the other "
            "choices produce OPT-062 floor-decomposition sibling rows."
        ),
    )
    parser.add_argument(
        "--owner-search-inner-search-service-kind",
        choices=OWNER_SEARCH_INNER_SEARCH_SERVICE_CHOICES,
        default=SEARCH_SERVICE_COMPACT_TORCH,
        help=(
            "Inner search service for --search-service-kind owner_search_slab_proxy. "
            "compact_torch_search_service uses the interim shared-memory host-root "
            "to owner resident-tensor bridge and reports that bridge H2D cost."
        ),
    )
    parser.add_argument(
        "--owner-search-defer-maintenance",
        action="store_true",
        help=(
            "For owner_search_slab_proxy, return action-critical search first and "
            "drain owner replay append, train, and search refresh as deferred "
            "owner maintenance."
        ),
    )
    parser.add_argument(
        "--owner-search-slab-bypass",
        action="store_true",
        help=(
            "For owner-search rows, bypass the general CompactRolloutSlab "
            "replay-row builder and stage owner transition handles directly."
        ),
    )
    parser.add_argument(
        "--owner-search-transition-batch-size",
        type=int,
        default=1,
        help=(
            "For --owner-search-slab-bypass, coalesce this many logical owner "
            "replay transitions into one parent-to-owner transport object."
        ),
    )
    parser.add_argument(
        "--owner-search-direct-transition-batch-replay",
        action="store_true",
        help=(
            "For owner-search transition-batch rows, let the owner replay store "
            "consume the fixed batch directly instead of expanding it into "
            "per-transition service materialization entries."
        ),
    )
    parser.add_argument(
        "--owner-search-owner-local-transition-derivation",
        action="store_true",
        help=(
            "For owner-search slab-bypass rows, stage compact transition handles "
            "and let owner direct replay derive next reward/done/final facts from "
            "the cached current root instead of transporting parent outcome arrays."
        ),
    )
    parser.add_argument(
        "--owner-search-owner-proxy-transition-closure",
        action="store_true",
        help=(
            "For direct-root owner-local transition rows, let the owner proxy "
            "close previous transitions from cached action frames and the "
            "current root-build request joint action."
        ),
    )
    parser.add_argument(
        "--owner-search-require-resident-root-view",
        action="store_true",
        help=(
            "For direct-root owner-search rows, fail unless the owner resolves "
            "the source resident root view without host fallback."
        ),
    )
    parser.add_argument(
        "--owner-search-resident-root-host-observation-stub",
        action="store_true",
        help=(
            "For direct-root resident owner-search slab-bypass rows, keep only a "
            "shape-only host observation stub in the parent root batch."
        ),
    )
    parser.add_argument(
        "--owner-search-direct-root-build-request",
        action="store_true",
        help=(
            "For direct-root resident owner-search slab-bypass rows, send a "
            "root-build request to the owner instead of building the root batch "
            "in parent Python."
        ),
    )
    parser.add_argument(
        "--compact-owner-action-step-boundary",
        action="store_true",
        help=(
            "Proof-only owner-boundary gate: drive manager.step from the "
            "cached owner search-feedback action and verify the next action "
            "returned by the slab. Requires direct-root owner-search."
        ),
    )
    parser.add_argument(
        "--compact-owner-action-dispatch-step-overlap",
        action="store_true",
        help=(
            "Default-off direct-root owner-boundary probe: submit owner action "
            "search before parent step payload/snapshot work and resolve before "
            "manager.step returns. Requires compact owner action-step boundary."
        ),
    )
    parser.add_argument(
        "--owner-search-fixed-action-result-buffer",
        action="store_true",
        help=(
            "For same-process direct-root owner-search rows, keep the full "
            "action result in a fixed local slot and return only a slot stub "
            "through the parent-facing result path."
        ),
    )
    parser.add_argument(
        "--owner-search-action-result-slot-capacity",
        type=int,
        default=4,
        help="Fixed action-result slot count for the same-process owner result buffer.",
    )
    parser.add_argument(
        "--owner-search-fixed-soa-replay",
        action="store_true",
        help=(
            "For owner-search direct transition-batch replay, append resident "
            "records into the fixed SoA unroll-2 buffer instead of the columnar "
            "entry-view replay path."
        ),
    )
    parser.add_argument(
        "--owner-search-fixed-soa-locality-sample-group-size",
        type=int,
        default=1,
        help=(
            "Experimental fixed-SoA learner sampling locality. Values greater "
            "than 1 sample rows in group-local chunks and are semantic-drift "
            "speed probes, not exact replay-sampling rows."
        ),
    )
    parser.add_argument(
        "--owner-search-learner-kind",
        choices=OWNER_SEARCH_LEARNER_CHOICES,
        default=OWNER_SEARCH_LEARNER_COMPACT_MUZERO,
        help=(
            "Owner-search learner implementation. mock_fast samples/builds the "
            "owner learner batch but skips the neural update to measure the "
            "non-learner owner-search ceiling."
        ),
    )
    parser.add_argument(
        "--owner-search-async-learner-worker",
        action="store_true",
        help=(
            "For deferred owner-search rows, submit owner learner train/update "
            "work to an owner learner worker and apply search refresh when the "
            "future completes."
        ),
    )
    parser.add_argument(
        "--owner-search-async-learner-worker-kind",
        choices=COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS,
        default=COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
        help="Worker adapter kind for deferred owner-search learner jobs.",
    )
    parser.add_argument(
        "--owner-search-async-learner-max-pending",
        type=int,
        default=1,
        help="Maximum queued deferred owner-search learner jobs.",
    )
    parser.add_argument(
        "--owner-search-defer-model-state-digest-to-refresh",
        action="store_true",
        help=(
            "For owner-search same-process model-state transport, skip the "
            "learner-side model hash and use the search worker's post-load "
            "model digest as the authoritative refresh identity."
        ),
    )
    parser.add_argument(
        "--compact-torch-request-compile",
        action="store_true",
        help="Request torch.compile for compact Torch helper kernels.",
    )
    parser.add_argument(
        "--compact-torch-request-model-compile",
        action="store_true",
        help="Request torch.compile for compact Torch model inference.",
    )
    parser.add_argument(
        "--compact-torch-model-compile-mode",
        choices=COMPACT_TORCH_MODEL_COMPILE_MODES,
        default="reduce-overhead",
        help=(
            "torch.compile mode for compact Torch model inference. "
            "Use default or max-autotune-no-cudagraphs for non-CUDAGraph probes."
        ),
    )
    parser.add_argument(
        "--compact-torch-timing-mode",
        choices=COMPACT_TORCH_TIMING_MODES,
        default="host_phase_sync",
        help="Timing/sync mode for compact Torch service diagnostics.",
    )
    parser.add_argument(
        "--compact-torch-initial-inference-mode",
        choices=COMPACT_TORCH_INITIAL_INFERENCE_MODES,
        default="model_method",
        help="Default-off OPT-079 initial inference path probe.",
    )
    parser.add_argument(
        "--compact-torch-observation-memory-format",
        choices=COMPACT_TORCH_MEMORY_FORMATS,
        default="contiguous",
        help="Default-off representation input layout probe.",
    )
    parser.add_argument(
        "--compact-torch-model-memory-format",
        choices=COMPACT_TORCH_MEMORY_FORMATS,
        default="contiguous",
        help="Default-off representation model layout probe.",
    )
    parser.add_argument(
        "--compact-torch-defer-one-simulation-replay-payload",
        action="store_true",
        help=(
            "Default-off owner-search probe: defer compact Torch one-simulation "
            "replay materialization until replay-payload flush."
        ),
    )
    parser.add_argument("--source-max-steps", type=int, default=1048576)
    parser.add_argument("--decision-source-frames", type=int, default=1)
    parser.add_argument(
        "--source-physics-step-ms",
        type=float,
        default=16.666666666666668,
    )
    parser.add_argument("--source-max-steps-semantics", default="source_physics_steps")
    parser.add_argument("--seed", type=int, default=20260530)
    args = parser.parse_args(argv)
    if str(args.compact_torch_model_memory_format) != "contiguous":
        raise ValueError(
            "compact_torch_model_memory_format=channels_last is parked for the "
            "current LightZero MuZero model because recurrent dynamics uses "
            ".view(); use --compact-torch-model-memory-format contiguous"
        )
    if bool(args.compact_owned_loop_deferred_learner) and bool(
        args.compact_owned_loop_deferred_sample_learner
    ):
        raise ValueError(
            "--compact-owned-loop-deferred-sample-learner cannot be combined with "
            "--compact-owned-loop-deferred-learner"
        )
    if int(args.compact_owned_loop_deferred_sample_learner_max_pending) <= 0:
        raise ValueError(
            "--compact-owned-loop-deferred-sample-learner-max-pending must be positive"
        )
    if str(
        args.compact_owned_loop_sample_learner_worker_kind
    ) != COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD and not bool(
        args.compact_owned_loop_deferred_sample_learner
    ):
        raise ValueError(
            "--compact-owned-loop-sample-learner-worker-kind requires "
            "--compact-owned-loop-deferred-sample-learner"
        )
    if (
        str(args.compact_owned_loop_deferred_sample_learner_replay_append_transport_kind)
        == COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
        and str(args.compact_owned_loop_sample_learner_worker_kind)
        != COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS
    ):
        raise ValueError(
            "scalar_ref_v1 replay append transport currently requires "
            "--compact-owned-loop-sample-learner-worker-kind local_process"
        )
    _validate_owner_search_speed_row_args(args)
    if bool(args.compact_muzero_learner_batch_unroll2_specialized_builder):
        if not bool(args.compact_owned_loop_fused_learner_batch):
            raise ValueError(
                "--compact-muzero-learner-batch-unroll2-specialized-builder requires "
                "--compact-owned-loop-fused-learner-batch"
            )
        if _learner_num_unroll_steps(args) != 2:
            raise ValueError(
                "--compact-muzero-learner-batch-unroll2-specialized-builder requires "
                "--learner-num-unroll-steps 2"
            )
    if bool(getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)):
        if not bool(args.compact_owned_loop_fused_learner_batch):
            raise ValueError(
                "--compact-muzero-learner-batch-learner-ready-unroll2-cache requires "
                "--compact-owned-loop-fused-learner-batch"
            )
        if _learner_num_unroll_steps(args) != 2:
            raise ValueError(
                "--compact-muzero-learner-batch-learner-ready-unroll2-cache requires "
                "--learner-num-unroll-steps 2"
            )
    if bool(getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)):
        if not bool(args.compact_owned_loop_fused_learner_batch):
            raise ValueError(
                "--compact-muzero-learner-batch-tensor-native-replay requires "
                "--compact-owned-loop-fused-learner-batch"
            )
        if not bool(
            getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ):
            raise ValueError(
                "--compact-muzero-learner-batch-tensor-native-replay requires "
                "--compact-muzero-learner-batch-learner-ready-unroll2-cache"
            )
        if _learner_num_unroll_steps(args) != 2:
            raise ValueError(
                "--compact-muzero-learner-batch-tensor-native-replay requires "
                "--learner-num-unroll-steps 2"
            )
    if (
        bool(args.compact_owned_loop_deferred_sample_learner)
        and _uses_compact_torch_search_service(args)
        and int(args.sample_interval) > 0
        and (int(args.steps) + int(args.warmup_steps)) % int(args.sample_interval) == 0
    ):
        raise ValueError(
            "compact Torch deferred sample+learner rows need at least one "
            "post-refresh actor/search step; choose steps + warmup_steps that "
            "is not divisible by --sample-interval"
        )
    if float(args.gpu_utilization_sample_interval_sec) < 0.0:
        raise ValueError("--gpu-utilization-sample-interval-sec must be non-negative")

    repo_root = Path.cwd()
    lifecycle_path = _resolve_path(args.unified_lifecycle_report, repo_root)
    lifecycle = _load_json(lifecycle_path)
    candidate_checkpoint_id = str(lifecycle.get("checkpoint_id") or "").strip()
    if not candidate_checkpoint_id:
        raise ValueError("unified lifecycle report must carry checkpoint_id")

    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    result_path = output_dir / f"row_{ROW_ID}_result.json"

    row = _manifest_row(
        args=args,
        candidate_checkpoint_id=candidate_checkpoint_id,
        repo_root=repo_root,
        lifecycle_path=lifecycle_path,
    )
    manifest = {
        "schema_id": COMPACT_COACH_SPEED_ROW_MANIFEST_SCHEMA_ID,
        "created_at": _utc_timestamp(),
        "experiment_id": str(args.run_id),
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "death_mode": str(args.death_mode),
        "learner_num_unroll_steps": _learner_num_unroll_steps(args),
        "compact_owned_loop_deferred_sample_learner": bool(
            args.compact_owned_loop_deferred_sample_learner
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_requested": int(
            args.compact_owned_loop_deferred_sample_learner_max_pending
        ),
        "compact_owned_loop_sample_learner_worker_kind_requested": str(
            args.compact_owned_loop_sample_learner_worker_kind
        ),
        "compact_owned_loop_fused_learner_batch": bool(args.compact_owned_loop_fused_learner_batch),
        UNROLL2_SPECIALIZED_BUILDER_KEY: bool(
            args.compact_muzero_learner_batch_unroll2_specialized_builder
        ),
        LEARNER_READY_UNROLL2_CACHE_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ),
        TENSOR_NATIVE_REPLAY_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
        ),
        "compact_owned_lean_profile_oracle": bool(args.compact_owned_lean_profile_oracle),
        "compact_profile_bounded_diagnostics": bool(
            getattr(args, "compact_profile_bounded_diagnostics", False)
        ),
        "compact_profile_cuda_sync_timing_diagnostics": bool(
            getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
        ),
        "compact_profile_runtime_step_timing_diagnostics": bool(
            getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
            or getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
        ),
        **_compact_owned_runner_fields(args),
        **_render_state_config_fields(args),
        **_compact_torch_memory_format_fields(args),
        **_owner_search_config_fields(args),
        "non_claims": _non_claims(),
        "rows": [row],
    }
    _write_json(manifest_path, manifest)

    learner_model = None
    search_model = None
    loaded_checkpoint_identity: dict[str, Any] = {}
    if bool(args.load_unified_lifecycle_checkpoint):
        loaded = _load_lifecycle_checkpoint_model(
            args=args,
            lifecycle=lifecycle,
            lifecycle_path=lifecycle_path,
            output_dir=output_dir,
        )
        learner_model = loaded["model"]
        search_model = copy.deepcopy(learner_model)
        loaded_checkpoint_identity = loaded["identity"]

    oracle_learner_model = None
    oracle_search_model = None
    if bool(args.compact_owned_lean_profile_oracle) and learner_model is None:
        learner_model = _TinyMuZero()
        search_model = copy.deepcopy(learner_model)
    if bool(args.compact_owned_lean_profile_oracle) and learner_model is not None:
        oracle_learner_model = copy.deepcopy(learner_model)
        oracle_search_model = copy.deepcopy(search_model)

    profile_runner = (
        _run_local_compact_owned_lean_trainer_profile
        if bool(args.compact_owned_lean_trainer_step)
        else _run_local_compact_owned_profile
    )
    gpu_sampler = _GpuUtilizationSamplerV1(
        enabled=bool(args.gpu_utilization_sampling),
        interval_sec=float(args.gpu_utilization_sample_interval_sec),
    )
    gpu_sampler.start()
    gpu_sampling_report: dict[str, Any] | None = None
    try:
        try:
            profile_payload = profile_runner(
                args=args,
                learner_model=learner_model,
                search_model=search_model,
                loaded_checkpoint_identity=loaded_checkpoint_identity,
            )
        except Exception as exc:
            gpu_sampling_report = gpu_sampler.stop()
            return _write_profile_failure_report(
                args=args,
                candidate_checkpoint_id=candidate_checkpoint_id,
                manifest_path=manifest_path,
                result_path=result_path,
                row=row,
                problem=f"{type(exc).__name__}: {exc}",
                failure_traceback=traceback.format_exc(),
                gpu_sampling_report=gpu_sampling_report,
            )
    finally:
        if gpu_sampling_report is None:
            gpu_sampling_report = gpu_sampler.stop()
    profile_payload["speed_row_gpu_utilization_sampling"] = gpu_sampling_report
    if bool(args.compact_owned_lean_profile_oracle):
        if not bool(args.compact_owned_lean_trainer_step):
            raise ValueError(
                "--compact-owned-lean-profile-oracle requires --compact-owned-lean-trainer-step"
            )
        profile_oracle_payload = _run_local_compact_owned_profile(
            args=args,
            learner_model=oracle_learner_model,
            search_model=oracle_search_model,
            loaded_checkpoint_identity=loaded_checkpoint_identity,
        )
        profile_payload["compact_owned_lean_profile_oracle"] = (
            _compact_owned_lean_profile_oracle_report(
                lean_payload=profile_payload,
                profile_payload=profile_oracle_payload,
            )
        )
    try:
        summary, compact_payload = _speed_summary_and_compact_payload(
            args=args,
            candidate_checkpoint_id=candidate_checkpoint_id,
            profile_payload=profile_payload,
            loaded_checkpoint_identity=loaded_checkpoint_identity,
        )
    except Exception as exc:
        return _write_profile_failure_report(
            args=args,
            candidate_checkpoint_id=candidate_checkpoint_id,
            manifest_path=manifest_path,
            result_path=result_path,
            row=row,
            problem=f"{type(exc).__name__}: {exc}",
            failure_traceback=traceback.format_exc(),
            gpu_sampling_report=gpu_sampling_report,
            profile_payload=profile_payload,
        )
    result = {
        "schema_id": COMPACT_COACH_SPEED_ROW_RESULT_SCHEMA_ID,
        "ok": True,
        "status": "complete",
        "problem": None,
        "returncode": 0,
        "run_invocation_id": f"{args.run_id}:{_utc_timestamp()}",
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "row_id": ROW_ID,
        "row": row,
        "producer": {
            "schema_id": COMPACT_COACH_SPEED_ROW_PRODUCER_SCHEMA_ID,
            "producer_id": "scripts/build_compact_coach_speed_row_smoke.py",
            "run_id": str(args.run_id),
            "produced_by": "local_compact_owned_env_search_replay_learner_smoke",
        },
        "summary": summary,
        "compact": compact_payload,
        "non_claims": _non_claims(),
    }
    _write_json(result_path, result)

    try:
        saved = save_compact_coach_speed_row_evidence_v1(
            route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
            candidate_checkpoint_id=candidate_checkpoint_id,
            unified_lifecycle_report_path=lifecycle_path,
            manifest_path=manifest_path,
            row_id=ROW_ID,
            result_json_path=result_path,
            speed_currency=SPEED_CURRENCY,
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )
    except Exception as exc:
        return _write_profile_failure_report(
            args=args,
            candidate_checkpoint_id=candidate_checkpoint_id,
            manifest_path=manifest_path,
            result_path=result_path,
            row=row,
            problem=f"{type(exc).__name__}: {exc}",
            failure_traceback=traceback.format_exc(),
            gpu_sampling_report=gpu_sampling_report,
            profile_payload=profile_payload,
        )
    evidence = saved["evidence"]
    report = {
        "ok": True,
        "schema_id": "curvyzero_compact_coach_speed_row_smoke_report/v1",
        "run_id": str(args.run_id),
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "manifest_path": str(manifest_path),
        "result_path": str(result_path),
        "evidence_path": str(saved["path"]),
        "evidence_ref": compact_coach_speed_row_evidence_ref(evidence),
        "speed_currency": SPEED_CURRENCY,
        "env_steps_collected": summary["env_steps_collected"],
        "training_wall_sec": summary["training_wall_sec"],
        "compact_trainer_env_steps_per_sec": summary["steps_per_sec"],
        "steps_per_sec": summary["steps_per_sec"],
        "compact_profile_bounded_diagnostics": summary["compact_profile_bounded_diagnostics"],
        "compact_profile_cuda_sync_timing_diagnostics": summary[
            "compact_profile_cuda_sync_timing_diagnostics"
        ],
        "source_profile_payload_embedded": summary["source_profile_payload_embedded"],
        "search_service_kind": summary["search_service_kind"],
        "search_service_impl": summary["search_service_impl"],
        "compact_owned_loop_deferred_learner": summary["compact_owned_loop_deferred_learner"],
        "compact_owned_loop_deferred_sample_learner": summary[
            "compact_owned_loop_deferred_sample_learner"
        ],
        "compact_owned_loop_deferred_sample_learner_max_pending_requested": summary[
            "compact_owned_loop_deferred_sample_learner_max_pending_requested"
        ],
        "compact_owned_loop_fused_learner_batch": summary["compact_owned_loop_fused_learner_batch"],
        UNROLL2_SPECIALIZED_BUILDER_KEY: summary.get(UNROLL2_SPECIALIZED_BUILDER_KEY, False),
        LEARNER_READY_UNROLL2_CACHE_KEY: summary.get(LEARNER_READY_UNROLL2_CACHE_KEY, False),
        **_deferred_learner_proof_fields(profile_payload),
        **_deferred_sample_learner_proof_fields(profile_payload),
        **_owner_search_slab_proxy_proof_fields(profile_payload),
        **_compact_loop_counter_fields(profile_payload),
        **_compact_owned_trainer_counter_fields(profile_payload),
        **_policy_refresh_proof_fields(profile_payload),
        "learner_num_unroll_steps": _learner_num_unroll_steps(args),
        **_operational_surface_fields(args, profile_payload),
        **{key: summary.get(key) for key in _SPEED_TIMING_PROJECTION_FIELDS},
        **_compact_rollout_slab_telemetry_total_fields(profile_payload),
        **{key: summary.get(key) for key in _WHOLE_OWNER_BUFFER_REPLAY_CEILING_FIELDS},
        **_sample_learner_fusion_fields(profile_payload),
        **_runtime_step_diagnostic_fields(profile_payload),
        **_normal_death_contract_fields(profile_payload),
        **_render_state_handoff_fields(args, profile_payload),
        **_compact_torch_memory_format_fields(args),
        **_owner_search_config_fields(args),
        **_compact_owned_runner_fields(args),
        **_gpu_utilization_sampling_fields(profile_payload),
        "compact_owned_lean_profile_oracle": profile_payload.get(
            "compact_owned_lean_profile_oracle"
        ),
        "profile_support_profile_only": profile_payload.get("profile_only"),
        "model_identity_scope": compact_payload["model_identity_scope"],
        "real_compact_owned_training_work": compact_payload["real_compact_owned_training_work"],
        "promotion_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    report_path = output_dir / "compact_coach_speed_row_smoke_report.json"
    _write_json(report_path, report)
    print(json.dumps({"ok": True, "report_path": str(report_path)}, sort_keys=True))
    return 0


def _write_profile_failure_report(
    *,
    args: argparse.Namespace,
    candidate_checkpoint_id: str,
    manifest_path: Path,
    result_path: Path,
    row: Mapping[str, Any],
    problem: str,
    gpu_sampling_report: Mapping[str, Any],
    failure_traceback: str = "",
    profile_payload: Mapping[str, Any] | None = None,
) -> int:
    run_invocation_id = f"{args.run_id}:{_utc_timestamp()}"
    common_fields = {
        "search_service_kind": str(args.search_service_kind),
        "search_service_impl": _search_service_impl(str(args.search_service_kind)),
        **_compact_torch_memory_format_fields(args),
        **_owner_search_config_fields(args),
        **_compact_owned_runner_fields(args),
    }
    gpu_payload = {
        "speed_row_gpu_utilization_sampling": dict(gpu_sampling_report),
    }
    profile_payload_dict = dict(profile_payload) if isinstance(profile_payload, Mapping) else {}
    profile_probe_fields = (
        _profile_failure_probe_fields(profile_payload_dict) if profile_payload_dict else {}
    )
    result = {
        "schema_id": COMPACT_COACH_SPEED_ROW_RESULT_SCHEMA_ID,
        "ok": False,
        "status": "failed",
        "problem": str(problem),
        "failure_traceback": str(failure_traceback or ""),
        "returncode": 1,
        "run_invocation_id": run_invocation_id,
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "row_id": ROW_ID,
        "row": dict(row),
        "producer": {
            "schema_id": COMPACT_COACH_SPEED_ROW_PRODUCER_SCHEMA_ID,
            "producer_id": "scripts/build_compact_coach_speed_row_smoke.py",
            "run_id": str(args.run_id),
            "produced_by": "local_compact_owned_env_search_replay_learner_smoke",
        },
        "summary": {
            "ok": False,
            "problem": str(problem),
            "failure_traceback": str(failure_traceback or ""),
            "returncode": 1,
            **common_fields,
            **_gpu_utilization_sampling_fields(gpu_payload),
            **profile_probe_fields,
        },
        "compact": {
            "ok": False,
            "problem": str(problem),
            "failure_traceback": str(failure_traceback or ""),
            "returncode": 1,
            **common_fields,
            **profile_probe_fields,
        },
        "non_claims": _non_claims(),
    }
    _write_json(result_path, result)
    report = {
        "ok": False,
        "schema_id": "curvyzero_compact_coach_speed_row_smoke_report/v1",
        "run_id": str(args.run_id),
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "manifest_path": str(manifest_path),
        "result_path": str(result_path),
        "problem": str(problem),
        "failure_traceback": str(failure_traceback or ""),
        "returncode": 1,
        **common_fields,
        **_gpu_utilization_sampling_fields(gpu_payload),
        **profile_probe_fields,
        "profile_support_profile_only": profile_payload_dict.get("profile_only"),
        "real_compact_owned_training_work": False,
        "promotion_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    report_path = result_path.parent / "compact_coach_speed_row_smoke_report.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "ok": False,
                "problem": str(problem),
                "report_path": str(report_path),
                "result_path": str(result_path),
            },
            sort_keys=True,
        )
    )
    return 1


def _profile_failure_probe_fields(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    last_search_metadata = _latest_compact_rollout_slab_search_metadata(profile_payload)
    env_steps_collected = float(profile_payload.get("steps") or 0.0) * float(
        profile_payload.get("batch_size") or 0.0
    )
    measured_sec = float(profile_payload.get("measured_sec") or 0.0)
    try:
        observed_steps_per_sec = float(profile_payload.get("steps_per_sec") or 0.0)
    except (TypeError, ValueError):
        observed_steps_per_sec = 0.0
    fields: dict[str, Any] = {
        "profile_failure_profile_payload_available": True,
        "profile_failure_steps": int(profile_payload.get("steps") or 0),
        "profile_failure_batch_size": int(profile_payload.get("batch_size") or 0),
        "profile_failure_measured_sec": measured_sec,
        "source_profile_steps_per_sec": profile_payload.get("steps_per_sec"),
        "source_profile_physical_rows_per_sec": profile_payload.get("physical_rows_per_sec"),
        "compact_rollout_slab_last_search_metadata": dict(last_search_metadata),
        **_owner_search_slab_proxy_proof_fields(profile_payload),
        **_sample_learner_fusion_fields(profile_payload),
        **_compact_rollout_slab_telemetry_total_fields(profile_payload),
        **_whole_owner_buffer_replay_ceiling_fields(
            profile_payload,
            training_wall_sec=measured_sec,
            env_steps_collected=env_steps_collected,
            observed_steps_per_sec=observed_steps_per_sec,
        ),
    }
    return fields


def _manifest_row(
    *,
    args: argparse.Namespace,
    candidate_checkpoint_id: str,
    repo_root: Path,
    lifecycle_path: Path,
) -> dict[str, Any]:
    return {
        "schema_id": "curvyzero_compact_coach_speed_row_manifest_row/v1",
        "row_id": ROW_ID,
        "label": _search_service_label(args),
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "unified_lifecycle_report": _relative_ref(lifecycle_path, repo_root),
        "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "row_purpose": "coach_speed_row",
        "speed_currency": SPEED_CURRENCY,
        "promotion_claim": False,
        "batch_size": int(args.batch_size),
        "actor_count": int(args.actor_count),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "death_mode": str(args.death_mode),
        "sample_batch_size": int(args.sample_batch_size),
        "sample_interval": int(args.sample_interval),
        "replay_pair_capacity": int(args.replay_pair_capacity),
        "learner_train_steps": int(args.learner_train_steps),
        "learner_num_unroll_steps": _learner_num_unroll_steps(args),
        "compact_owned_loop_deferred_sample_learner": bool(
            args.compact_owned_loop_deferred_sample_learner
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_requested": int(
            args.compact_owned_loop_deferred_sample_learner_max_pending
        ),
        "compact_owned_loop_sample_learner_worker_kind_requested": str(
            args.compact_owned_loop_sample_learner_worker_kind
        ),
        "compact_owned_loop_fused_learner_batch": bool(args.compact_owned_loop_fused_learner_batch),
        UNROLL2_SPECIALIZED_BUILDER_KEY: bool(
            args.compact_muzero_learner_batch_unroll2_specialized_builder
        ),
        LEARNER_READY_UNROLL2_CACHE_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ),
        TENSOR_NATIVE_REPLAY_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
        ),
        "compact_owned_lean_profile_oracle": bool(args.compact_owned_lean_profile_oracle),
        "compact_profile_bounded_diagnostics": bool(
            getattr(args, "compact_profile_bounded_diagnostics", False)
        ),
        "compact_profile_cuda_sync_timing_diagnostics": bool(
            getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
        ),
        "compact_profile_runtime_step_timing_diagnostics": bool(
            getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
            or getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
        ),
        **_compact_owned_runner_fields(args),
        "policy_refresh_interval": int(args.policy_refresh_interval),
        **_render_state_config_fields(args),
        "search_service_kind": str(args.search_service_kind),
        "search_service_impl": _search_service_impl(str(args.search_service_kind)),
        **_compact_torch_memory_format_fields(args),
        **_owner_search_config_fields(args),
        "num_simulations": int(args.num_simulations),
        "search_service_floor_decomposition_role": _search_service_floor_role(
            str(args.search_service_kind)
        ),
        "learner_device": str(args.learner_device),
        "seed": int(args.seed),
        "non_claims": _non_claims(),
        "command": [str(part) for part in sys.argv],
    }


def _search_service_impl(kind: str) -> str:
    if kind == SEARCH_SERVICE_DEVICE_TARGET:
        return _DeviceTargetSearchService.search_impl
    if kind == SEARCH_SERVICE_COMPACT_TORCH:
        return CompactTorchSearchServiceV1.search_impl
    if kind == SEARCH_SERVICE_FIXED_SHAPE:
        return FixedShapeBatchedSearchOwnerV0.search_impl
    if kind == SEARCH_SERVICE_OWNER_SEARCH_SLAB_PROXY:
        return "compact_owner_search_slab_proxy_v1"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY:
        return "compact_owner_search_inline_proxy_v1"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY:
        return "compact_owner_search_inline_background_proxy_v1"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY:
        return "compact_owner_search_threaded_proxy_v1"
    raise ValueError(f"unknown search service kind {kind!r}")


def _search_service_label(args: argparse.Namespace) -> str:
    kind = str(args.search_service_kind)
    if kind == SEARCH_SERVICE_DEVICE_TARGET:
        return "local compact-owned env/search/replay/learner speed smoke"
    if kind == SEARCH_SERVICE_COMPACT_TORCH:
        return "local compact-owned real compact Torch search speed sibling"
    if kind == SEARCH_SERVICE_FIXED_SHAPE:
        return "local compact-owned fixed-shape no-search floor speed sibling"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_SLAB_PROXY:
        return "local compact-owned owner-search slab-proxy sibling"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY:
        return "local compact-owned inline owner-search sibling"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY:
        return "local compact-owned inline/background owner-search sibling"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY:
        return "local compact-owned threaded owner-search sibling"
    raise ValueError(f"unknown search service kind {kind!r}")


def _search_service_floor_role(kind: str) -> str:
    if kind == SEARCH_SERVICE_DEVICE_TARGET:
        return "accepted_device_target_baseline"
    if kind == SEARCH_SERVICE_COMPACT_TORCH:
        return "compact_torch_search_service_sibling"
    if kind == SEARCH_SERVICE_FIXED_SHAPE:
        return "fixed_no_search_floor_sibling"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_SLAB_PROXY:
        return "owner_search_slab_proxy_sibling"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY:
        return "owner_search_inline_proxy_sibling"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY:
        return "owner_search_inline_background_proxy_sibling"
    if kind == SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY:
        return "owner_search_threaded_proxy_sibling"
    raise ValueError(f"unknown search service kind {kind!r}")


def _owner_search_inner_search_service_kind(args: argparse.Namespace) -> str:
    return str(
        getattr(
            args,
            "owner_search_inner_search_service_kind",
            SEARCH_SERVICE_COMPACT_TORCH,
        )
    )


def _owner_search_owner_train_enabled(args: argparse.Namespace) -> bool:
    return str(getattr(args, "search_service_kind", "")) in {
        SEARCH_SERVICE_OWNER_SEARCH_SLAB_PROXY,
        SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY,
        SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY,
        SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY,
    }


def _owner_search_train_interval(args: argparse.Namespace) -> int:
    return max(int(args.sample_interval), _learner_num_unroll_steps(args) + 1)


def _owner_search_expected_train_request_count(args: argparse.Namespace) -> int:
    train_interval = _owner_search_train_interval(args)
    measured_steps = max(0, int(getattr(args, "steps", 0) or 0))
    if train_interval <= 0 or measured_steps <= 0:
        return 0

    warmup_steps = max(0, int(getattr(args, "warmup_steps", 0) or 0))
    replayable_transition_count = measured_steps
    if warmup_steps <= 0:
        replayable_transition_count = max(0, measured_steps - 1)

    if bool(getattr(args, "owner_search_slab_bypass", False)):
        transition_batch_size = max(
            1,
            int(getattr(args, "owner_search_transition_batch_size", 1) or 1),
        )
        if transition_batch_size > 1:
            replayable_transition_count = (
                replayable_transition_count // transition_batch_size
            ) * transition_batch_size

    return replayable_transition_count // train_interval


def _validate_owner_search_speed_row_args(args: argparse.Namespace) -> None:
    action_result_slot_capacity = int(
        getattr(args, "owner_search_action_result_slot_capacity", 4) or 4
    )
    if action_result_slot_capacity <= 0:
        raise ValueError("--owner-search-action-result-slot-capacity must be positive")
    fixed_soa_locality_group_size = int(
        getattr(args, "owner_search_fixed_soa_locality_sample_group_size", 1) or 1
    )
    if fixed_soa_locality_group_size <= 0:
        raise ValueError("owner_search fixed SoA locality sample group size must be positive")
    if fixed_soa_locality_group_size > 1 and not bool(
        getattr(args, "owner_search_fixed_soa_replay", False)
    ):
        raise ValueError(
            "--owner-search-fixed-soa-locality-sample-group-size > 1 requires "
            "--owner-search-fixed-soa-replay"
        )
    if not _owner_search_owner_train_enabled(args):
        if bool(getattr(args, "owner_search_direct_root_build_request", False)):
            raise ValueError(
                "--owner-search-direct-root-build-request requires an owner-search row"
            )
        if bool(getattr(args, "owner_search_fixed_action_result_buffer", False)):
            raise ValueError(
                "--owner-search-fixed-action-result-buffer requires an owner-search row"
            )
        if bool(getattr(args, "owner_search_owner_local_transition_derivation", False)):
            raise ValueError(
                "--owner-search-owner-local-transition-derivation requires an owner-search row"
            )
        if bool(getattr(args, "owner_search_owner_proxy_transition_closure", False)):
            raise ValueError(
                "--owner-search-owner-proxy-transition-closure requires an owner-search row"
            )
        if bool(getattr(args, "compact_owner_action_step_boundary", False)):
            raise ValueError(
                "--compact-owner-action-step-boundary requires an owner-search row"
            )
        if bool(getattr(args, "compact_owner_action_dispatch_step_overlap", False)):
            raise ValueError(
                "--compact-owner-action-dispatch-step-overlap requires an owner-search row"
            )
        return
    if _owner_search_inner_search_service_kind(args) != SEARCH_SERVICE_COMPACT_TORCH:
        raise ValueError("owner_search_slab_proxy speed rows require compact_torch inner search")
    if bool(args.compact_owned_lean_trainer_step):
        raise ValueError("owner_search_slab_proxy cannot be combined with lean trainer step")
    if bool(args.compact_owned_lean_profile_oracle):
        raise ValueError("owner_search_slab_proxy cannot run the lean profile oracle")
    if bool(args.compact_owned_loop_deferred_learner) or bool(
        args.compact_owned_loop_deferred_sample_learner
    ):
        raise ValueError(
            "owner_search_slab_proxy owns training; parent deferred learner flags are invalid"
        )
    if bool(getattr(args, "owner_search_async_learner_worker", False)) and not bool(
        getattr(args, "owner_search_defer_maintenance", False)
    ):
        raise ValueError("owner_search async learner worker requires deferred owner maintenance")
    if bool(getattr(args, "owner_search_async_learner_worker", False)):
        if int(getattr(args, "owner_search_async_learner_max_pending", 1)) <= 0:
            raise ValueError("owner_search async learner max pending must be positive")
        worker_kind = str(
            getattr(
                args,
                "owner_search_async_learner_worker_kind",
                COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
            )
        )
        if worker_kind not in COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS:
            allowed = ", ".join(COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS)
            raise ValueError(f"owner_search async learner worker kind must be one of {allowed}")
        if (
            bool(
                getattr(
                    args,
                    "owner_search_defer_model_state_digest_to_refresh",
                    False,
                )
            )
            and worker_kind == COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH
        ):
            raise ValueError(
                "--owner-search-defer-model-state-digest-to-refresh requires "
                "same-process owner-search model-state transport"
            )
    if bool(getattr(args, "owner_search_resident_root_host_observation_stub", False)) and not (
        bool(getattr(args, "owner_search_slab_bypass", False))
        and bool(getattr(args, "owner_search_require_resident_root_view", False))
    ):
        raise ValueError(
            "--owner-search-resident-root-host-observation-stub requires "
            "--owner-search-slab-bypass and --owner-search-require-resident-root-view"
        )
    if bool(getattr(args, "owner_search_direct_root_build_request", False)) and not (
        bool(getattr(args, "owner_search_slab_bypass", False))
        and bool(getattr(args, "owner_search_require_resident_root_view", False))
        and bool(getattr(args, "owner_search_resident_root_host_observation_stub", False))
    ):
        raise ValueError(
            "--owner-search-direct-root-build-request requires "
            "--owner-search-slab-bypass, --owner-search-require-resident-root-view, "
            "and --owner-search-resident-root-host-observation-stub"
        )
    if bool(getattr(args, "owner_search_direct_root_build_request", False)) and str(
        getattr(args, "search_service_kind", "")
    ) not in {
        SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY,
        SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY,
        SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY,
    }:
        raise ValueError(
            "--owner-search-direct-root-build-request requires an inline, "
            "inline-background, or threaded owner-search proxy"
        )
    if bool(getattr(args, "compact_owner_action_step_boundary", False)):
        if not bool(getattr(args, "owner_search_direct_root_build_request", False)):
            raise ValueError(
                "--compact-owner-action-step-boundary requires "
                "--owner-search-direct-root-build-request"
            )
        if not bool(getattr(args, "owner_search_slab_bypass", False)):
            raise ValueError(
                "--compact-owner-action-step-boundary requires --owner-search-slab-bypass"
            )
    if bool(getattr(args, "compact_owner_action_dispatch_step_overlap", False)):
        if not bool(getattr(args, "compact_owner_action_step_boundary", False)):
            raise ValueError(
                "--compact-owner-action-dispatch-step-overlap requires "
                "--compact-owner-action-step-boundary"
            )
        if not bool(getattr(args, "owner_search_direct_root_build_request", False)):
            raise ValueError(
                "--compact-owner-action-dispatch-step-overlap requires "
                "--owner-search-direct-root-build-request"
            )
    if bool(getattr(args, "owner_search_fixed_action_result_buffer", False)):
        if not bool(getattr(args, "owner_search_defer_maintenance", False)):
            raise ValueError(
                "--owner-search-fixed-action-result-buffer requires "
                "--owner-search-defer-maintenance"
            )
        if not bool(getattr(args, "owner_search_direct_root_build_request", False)):
            raise ValueError(
                "--owner-search-fixed-action-result-buffer requires "
                "--owner-search-direct-root-build-request"
            )
        if str(getattr(args, "search_service_kind", "")) not in {
            SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY,
            SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY,
            SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY,
        }:
            raise ValueError(
                "--owner-search-fixed-action-result-buffer requires an inline, "
                "inline-background, or threaded owner-search proxy"
            )
    if bool(getattr(args, "owner_search_owner_local_transition_derivation", False)):
        if not bool(getattr(args, "owner_search_slab_bypass", False)):
            raise ValueError(
                "--owner-search-owner-local-transition-derivation requires "
                "--owner-search-slab-bypass"
            )
        if int(getattr(args, "owner_search_transition_batch_size", 1)) <= 1:
            raise ValueError(
                "--owner-search-owner-local-transition-derivation requires "
                "--owner-search-transition-batch-size > 1"
            )
        if not bool(getattr(args, "owner_search_direct_transition_batch_replay", False)):
            raise ValueError(
                "--owner-search-owner-local-transition-derivation requires "
                "--owner-search-direct-transition-batch-replay"
            )
    if bool(getattr(args, "owner_search_owner_proxy_transition_closure", False)):
        if not bool(getattr(args, "owner_search_owner_local_transition_derivation", False)):
            raise ValueError(
                "--owner-search-owner-proxy-transition-closure requires "
                "--owner-search-owner-local-transition-derivation"
            )
        if not bool(getattr(args, "owner_search_direct_root_build_request", False)):
            raise ValueError(
                "--owner-search-owner-proxy-transition-closure requires "
                "--owner-search-direct-root-build-request"
            )
    if int(args.sample_batch_size) < 0:
        raise ValueError("owner_search_slab_proxy requires --sample-batch-size >= 0")
    if int(args.sample_interval) <= 0:
        raise ValueError("owner_search_slab_proxy requires --sample-interval > 0")
    if int(args.learner_train_steps) <= 0:
        raise ValueError("owner_search_slab_proxy requires --learner-train-steps > 0")
    if int(args.replay_pair_capacity) <= _learner_num_unroll_steps(args):
        raise ValueError("owner_search_slab_proxy replay capacity must exceed learner unroll steps")
    train_interval = _owner_search_train_interval(args)
    append_count = int(args.steps) + int(args.warmup_steps) - 1
    if append_count < train_interval:
        raise ValueError(
            "owner_search_slab_proxy run is too short to train from replay: "
            f"needs at least {train_interval + 1} total steps, got "
            f"{int(args.steps) + int(args.warmup_steps)}"
        )


def _uses_compact_torch_search_service(args: argparse.Namespace) -> bool:
    kind = str(args.search_service_kind)
    if kind == SEARCH_SERVICE_COMPACT_TORCH:
        return True
    return (
        kind
        in {
            SEARCH_SERVICE_OWNER_SEARCH_SLAB_PROXY,
            SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY,
            SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY,
            SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY,
        }
        and _owner_search_inner_search_service_kind(args) == SEARCH_SERVICE_COMPACT_TORCH
    )


def _owner_search_config_fields(args: argparse.Namespace) -> dict[str, Any]:
    owner_search = _owner_search_owner_train_enabled(args)
    inline_owner_search = str(args.search_service_kind) == SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY
    inline_background_owner_search = (
        str(args.search_service_kind) == SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY
    )
    threaded_owner_search = (
        str(args.search_service_kind) == SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY
    )
    inner_kind = _owner_search_inner_search_service_kind(args)
    bridge_ready = (
        owner_search
        and not inline_owner_search
        and not inline_background_owner_search
        and not threaded_owner_search
        and inner_kind == SEARCH_SERVICE_COMPACT_TORCH
    )
    return {
        "owner_search_slab_proxy_requested": owner_search,
        "owner_search_inline_proxy_requested": inline_owner_search,
        "owner_search_inline_background_proxy_requested": (inline_background_owner_search),
        "owner_search_threaded_proxy_requested": threaded_owner_search,
        "owner_search_inner_search_service_kind": inner_kind if owner_search else "",
        "owner_search_inner_search_service_impl": (
            _search_service_impl(inner_kind) if owner_search else ""
        ),
        "owner_search_compact_torch_resident_root_bridge_ready": bool(bridge_ready),
        "owner_search_defer_maintenance_requested": bool(
            owner_search and getattr(args, "owner_search_defer_maintenance", False)
        ),
        "owner_search_slab_bypass_requested": bool(
            owner_search and getattr(args, "owner_search_slab_bypass", False)
        ),
        "owner_search_transition_batch_size_requested": int(
            getattr(args, "owner_search_transition_batch_size", 1)
        )
        if owner_search
        else 1,
        "owner_search_transition_batch_transport_requested": bool(
            owner_search
            and getattr(args, "owner_search_slab_bypass", False)
            and int(getattr(args, "owner_search_transition_batch_size", 1)) > 1
        ),
        "owner_search_direct_transition_batch_replay_requested": bool(
            owner_search and getattr(args, "owner_search_direct_transition_batch_replay", False)
        ),
        "owner_search_owner_local_transition_derivation_requested": bool(
            owner_search and getattr(args, "owner_search_owner_local_transition_derivation", False)
        ),
        "owner_search_owner_proxy_transition_closure_requested": bool(
            owner_search and getattr(args, "owner_search_owner_proxy_transition_closure", False)
        ),
        "owner_search_require_resident_root_view_requested": bool(
            owner_search and getattr(args, "owner_search_require_resident_root_view", False)
        ),
        "owner_search_resident_root_host_observation_stub_requested": bool(
            owner_search
            and getattr(
                args,
                "owner_search_resident_root_host_observation_stub",
                False,
            )
        ),
        "owner_search_direct_root_build_request_requested": bool(
            owner_search and getattr(args, "owner_search_direct_root_build_request", False)
        ),
        "compact_owner_action_step_boundary_requested": bool(
            owner_search and getattr(args, "compact_owner_action_step_boundary", False)
        ),
        "compact_owner_action_dispatch_step_overlap_requested": bool(
            owner_search
            and getattr(args, "compact_owner_action_dispatch_step_overlap", False)
        ),
        "owner_search_fixed_action_result_buffer_requested": bool(
            owner_search and getattr(args, "owner_search_fixed_action_result_buffer", False)
        ),
        "owner_search_action_result_slot_capacity_requested": (
            int(getattr(args, "owner_search_action_result_slot_capacity", 4) or 4)
            if owner_search
            else 0
        ),
        "owner_search_async_learner_worker_requested": bool(
            owner_search and getattr(args, "owner_search_async_learner_worker", False)
        ),
        "owner_search_async_learner_worker_kind_requested": (
            str(
                getattr(
                    args,
                    "owner_search_async_learner_worker_kind",
                    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
                )
            )
            if owner_search
            else ""
        ),
        "owner_search_async_learner_max_pending_requested": int(
            getattr(args, "owner_search_async_learner_max_pending", 1)
        )
        if owner_search
        else 0,
    }


def _compact_torch_memory_format_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "compact_torch_initial_inference_mode": str(args.compact_torch_initial_inference_mode),
        "compact_torch_observation_memory_format": str(
            args.compact_torch_observation_memory_format
        ),
        "compact_torch_model_memory_format": str(args.compact_torch_model_memory_format),
        "compact_torch_defer_one_simulation_replay_payload_requested": bool(
            getattr(args, "compact_torch_defer_one_simulation_replay_payload", False)
        ),
        "compact_torch_memory_format_applies_to_search_service": _uses_compact_torch_search_service(
            args
        ),
    }


def _gpu_utilization_sampling_fields(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    sampling = profile_payload.get("speed_row_gpu_utilization_sampling")
    if not isinstance(sampling, Mapping):
        sampling = {}
    return {
        "speed_row_gpu_utilization_sampling_enabled": bool(sampling.get("enabled", False)),
        "speed_row_gpu_utilization_sample_interval_sec": float(sampling.get("interval_sec") or 0.0),
        "speed_row_gpu_utilization_sample_count": int(sampling.get("sample_count") or 0),
        "speed_row_gpu_name": str(sampling.get("gpu_name") or ""),
        "speed_row_gpu_utilization_max_percent": sampling.get("max_gpu_util_percent"),
        "speed_row_gpu_utilization_mean_percent": sampling.get("mean_gpu_util_percent"),
        "speed_row_gpu_utilization_nonzero_sample_count": int(
            sampling.get("gpu_util_nonzero_sample_count") or 0
        ),
        "speed_row_gpu_utilization_over_50_sample_count": int(
            sampling.get("gpu_util_over_50_sample_count") or 0
        ),
        "speed_row_gpu_utilization_over_80_sample_count": int(
            sampling.get("gpu_util_over_80_sample_count") or 0
        ),
        "speed_row_gpu_memory_utilization_max_percent": sampling.get("max_memory_util_percent"),
        "speed_row_gpu_memory_used_max_mib": sampling.get("max_memory_used_mib"),
        "speed_row_gpu_power_draw_max_w": sampling.get("max_power_draw_w"),
        "speed_row_gpu_utilization_sampling_errors": list(sampling.get("errors") or []),
    }


def _runtime_step_diagnostic_fields(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    stats = profile_payload.get("compact_profile_runtime_step_timing_stats")
    if not isinstance(stats, Mapping):
        stats = {}

    def float_stat(name: str) -> float:
        try:
            value = float(stats.get(name) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        return value if math.isfinite(value) else 0.0

    def int_stat(name: str) -> int:
        try:
            return int(stats.get(name) or 0)
        except (TypeError, ValueError):
            return 0

    fields = {
        "compact_profile_runtime_step_timing_diagnostics": bool(
            profile_payload.get("compact_profile_runtime_step_timing_diagnostics")
        ),
        "compact_profile_runtime_step_timing_stats": dict(stats),
        "compact_profile_runtime_step_count": int_stat("count"),
        "compact_profile_runtime_step_sum_sec": float_stat("sum_sec"),
        "compact_profile_runtime_step_min_sec": float_stat("min_sec"),
        "compact_profile_runtime_step_max_sec": float_stat("max_sec"),
        "compact_profile_runtime_step_p50_sec": float_stat("p50_sec"),
        "compact_profile_runtime_step_p95_sec": float_stat("p95_sec"),
        "compact_profile_runtime_step_slowest_iteration": int_stat("slowest_iteration"),
        "compact_profile_runtime_step_slowest_measured_iteration": int_stat(
            "slowest_measured_iteration"
        ),
        "compact_profile_runtime_step_slowest_actor_step_wall_sec": float_stat(
            "slowest_actor_step_wall_sec"
        ),
        "compact_profile_runtime_step_slowest_observation_sec": float_stat(
            "slowest_observation_sec"
        ),
        "compact_profile_runtime_step_slowest_compact_rollout_slab_sec": float_stat(
            "slowest_compact_rollout_slab_sec"
        ),
        "compact_profile_runtime_step_slowest_sample_gate_sec": float_stat(
            "slowest_sample_gate_sec"
        ),
        "compact_profile_runtime_step_slowest_learner_gate_sec": float_stat(
            "slowest_learner_gate_sec"
        ),
        "compact_profile_runtime_step_slowest_policy_refresh_sec": float_stat(
            "slowest_policy_refresh_sec"
        ),
        "compact_profile_runtime_step_slowest_primary_accounted_sec": float_stat(
            "slowest_primary_accounted_sec"
        ),
        "compact_profile_runtime_step_slowest_primary_residual_sec": float_stat(
            "slowest_primary_residual_sec"
        ),
        "compact_profile_runtime_step_slowest_env_trajectory_checksum": int_stat(
            "slowest_env_trajectory_checksum"
        ),
        "compact_profile_runtime_step_top_slowest_records": list(
            stats.get("top_slowest_records") or []
        ),
    }
    for phase_name in (
        "actor_step_wall",
        "actor_env_runtime",
        "actor_autoreset",
        "observation",
        "compact_rollout_slab",
        "sample_gate",
        "sample_gate_residual",
        "sample_gate_cuda_sync",
        "sample_gate_builder_group_loop",
        "sample_gate_builder_cuda_sync",
        "learner_gate",
        "policy_refresh",
        "primary_accounted",
        "primary_residual",
    ):
        for stat_name in ("sum_sec", "min_sec", "max_sec", "p50_sec", "p95_sec"):
            fields[f"compact_profile_runtime_step_{phase_name}_{stat_name}"] = float_stat(
                f"{phase_name}_{stat_name}"
            )
    for bucket_name in (
        "sample_gate_active",
        "sample_gate_inactive",
        "early",
        "mid",
        "late",
    ):
        fields[f"compact_profile_runtime_step_{bucket_name}_count"] = int_stat(
            f"{bucket_name}_count"
        )
        fields[f"compact_profile_runtime_step_{bucket_name}_sample_gate_active_count"] = int_stat(
            f"{bucket_name}_sample_gate_active_count"
        )
        for stat_name in ("sum_sec", "min_sec", "max_sec", "p50_sec", "p95_sec"):
            fields[f"compact_profile_runtime_step_{bucket_name}_{stat_name}"] = float_stat(
                f"{bucket_name}_{stat_name}"
            )
        for phase_name in (
            "actor_step_wall",
            "observation",
            "sample_gate",
            "sample_gate_residual",
            "sample_gate_builder_group_loop",
            "learner_gate",
            "primary_residual",
        ):
            fields[f"compact_profile_runtime_step_{bucket_name}_{phase_name}_sum_sec"] = float_stat(
                f"{bucket_name}_{phase_name}_sum_sec"
            )
        if bucket_name != "sample_gate_inactive":
            active_prefix = (
                "sample_gate_active"
                if bucket_name == "sample_gate_active"
                else f"{bucket_name}_sample_gate_active"
            )
            for phase_name in (
                "sample_gate",
                "sample_gate_residual",
                "sample_gate_builder_group_loop",
                "learner_gate",
                "observation",
                "primary_residual",
            ):
                for stat_name in (
                    "count",
                    "sum_sec",
                    "min_sec",
                    "max_sec",
                    "p50_sec",
                    "p95_sec",
                ):
                    key = f"{active_prefix}_{phase_name}_{stat_name}"
                    if stat_name == "count":
                        fields[f"compact_profile_runtime_step_{key}"] = int_stat(key)
                    else:
                        fields[f"compact_profile_runtime_step_{key}"] = float_stat(key)
    return fields


def _surface_compact_owned_loop_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    loop_telemetry = payload.get("compact_owned_loop_telemetry")
    if not isinstance(loop_telemetry, Mapping):
        return payload
    for key, value in loop_telemetry.items():
        key_str = str(key)
        if key_str.startswith("compact_owned_loop_"):
            payload[key_str] = value
    return payload


def _render_state_config_fields(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hybrid_persistent_compact_render_state_buffer": bool(
            getattr(args, "hybrid_persistent_compact_render_state_buffer", False)
        ),
        "hybrid_borrow_single_actor_render_state": bool(
            getattr(args, "hybrid_borrow_single_actor_render_state", False)
        ),
    }


def _compact_muzero_learner_batch_metadata_flags(
    args: argparse.Namespace,
) -> dict[str, Any]:
    unroll2_specialized = bool(
        getattr(args, "compact_muzero_learner_batch_unroll2_specialized_builder", False)
    )
    learner_ready_unroll2_cache = bool(
        getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
    )
    tensor_native_replay = bool(
        getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
    )
    fixed_soa_replay = bool(getattr(args, "owner_search_fixed_soa_replay", False))
    fixed_soa_locality_group_size = int(
        getattr(args, "owner_search_fixed_soa_locality_sample_group_size", 1) or 1
    )
    return {
        "compact_owned_loop_fused_learner_batch": bool(
            getattr(args, "compact_owned_loop_fused_learner_batch", False)
        ),
        UNROLL2_SPECIALIZED_BUILDER_KEY: unroll2_specialized,
        UNROLL2_SPECIALIZED_BUILDER_REQUESTED_KEY: unroll2_specialized,
        LEARNER_READY_UNROLL2_CACHE_KEY: learner_ready_unroll2_cache,
        LEARNER_READY_UNROLL2_CACHE_REQUESTED_KEY: learner_ready_unroll2_cache,
        TENSOR_NATIVE_REPLAY_KEY: tensor_native_replay,
        TENSOR_NATIVE_REPLAY_REQUESTED_KEY: tensor_native_replay,
        FIXED_SOA_REPLAY_REQUESTED_KEY: fixed_soa_replay,
        FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY: fixed_soa_replay,
        FIXED_SOA_LEARNER_BATCH_HANDLE_RING_REQUESTED_KEY: fixed_soa_replay,
        FIXED_SOA_LOCALITY_SAMPLE_GROUP_SIZE_KEY: int(max(1, fixed_soa_locality_group_size)),
    }


def _resident_batch_handle_fields_from_metadata(
    metadata: Mapping[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    requested = bool(metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_requested", False))
    used = bool(metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_used", False))
    fallback_count = int(
        metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_fallback_count", 0)
        or 0
    )
    fallback_reason = str(
        metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_fallback_reason", "none")
        or "none"
    )
    consumed = bool(requested and used and fallback_count == 0)
    return {
        f"{prefix}_requested": requested,
        f"{prefix}_consumed": consumed,
        f"{prefix}_source": "fixed_soa_learner_batch_handle_ring" if requested else "none",
        f"{prefix}_schema_id": str(
            metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_schema_id", "none")
            or "none"
        ),
        f"{prefix}_handle_id": int(
            metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_handle_id", 0) or 0
        ),
        f"{prefix}_snapshot_version": int(
            metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_snapshot_version", 0) or 0
        ),
        f"{prefix}_request_checksum": int(
            metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_request_checksum", 0) or 0
        ),
        f"{prefix}_sample_row_count": int(
            metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_sample_row_count", 0) or 0
        ),
        f"{prefix}_target_row_count": int(
            metadata.get(f"{FIXED_SOA_LEARNER_BATCH_HANDLE_RING_KEY}_target_row_count", 0) or 0
        ),
        f"{prefix}_fallback_count": fallback_count,
        f"{prefix}_fallback_reason": fallback_reason,
        f"{prefix}_materialized_parent_fallback_count": fallback_count,
        f"{prefix}_materialized_parent_fallback_reason": (
            fallback_reason if fallback_count else "none"
        ),
    }


def _owner_search_learner_resident_batch_handle_fields(
    *,
    learner_batch: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged_metadata = dict(metadata or {})
    batch_metadata = getattr(learner_batch, "metadata", None)
    if isinstance(batch_metadata, Mapping):
        merged_metadata.update(dict(batch_metadata))
    owner_fields = _resident_batch_handle_fields_from_metadata(
        merged_metadata,
        prefix=OWNER_SEARCH_RESIDENT_BATCH_HANDLE_PREFIX,
    )
    owned_loop_alias_fields = _resident_batch_handle_fields_from_metadata(
        merged_metadata,
        prefix=OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX,
    )
    return {**owner_fields, **owned_loop_alias_fields}


def _compact_owned_runner_fields(args: argparse.Namespace) -> dict[str, Any]:
    lean = bool(getattr(args, "compact_owned_lean_trainer_step", False))
    owner_search = _owner_search_owner_train_enabled(args)
    return {
        "compact_owned_lean_trainer_step": lean,
        "compact_owned_lean_profile_oracle_requested": bool(
            getattr(args, "compact_owned_lean_profile_oracle", False)
        ),
        ("compact_owned_loop_deferred_sample_learner_replay_append_transport_kind_requested"): str(
            getattr(
                args,
                "compact_owned_loop_deferred_sample_learner_replay_append_transport_kind",
                COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1,
            )
        ),
        "compact_owned_training_loop_owner": (
            "owner_search_worker"
            if owner_search
            else ("lean_compact_trainer_step" if lean else "hybrid_observation_profile_runner")
        ),
    }


def _compact_owned_trainer_counter_fields(
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "compact_owned_trainer_record_step_calls": int(
            profile_payload.get("compact_owned_trainer_record_step_calls") or 0
        ),
        "compact_owned_trainer_learner_update_count": int(
            profile_payload.get("compact_owned_trainer_learner_update_count") or 0
        ),
        "compact_owned_trainer_sample_batch_count": int(
            profile_payload.get("compact_owned_trainer_sample_batch_count") or 0
        ),
        "compact_owned_trainer_policy_refresh_count": int(
            profile_payload.get("compact_owned_trainer_policy_refresh_count") or 0
        ),
        "compact_owned_trainer_policy_version_ref": str(
            profile_payload.get("compact_owned_trainer_policy_version_ref") or ""
        ),
        "compact_owned_trainer_model_version_ref": str(
            profile_payload.get("compact_owned_trainer_model_version_ref") or ""
        ),
        "compact_owned_trainer_loop_counter_source": str(
            profile_payload.get("compact_owned_trainer_loop_counter_source") or ""
        ),
    }


def _deferred_learner_proof_fields(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compact_owned_loop_deferred_learner_submit_count": int(
            profile_payload.get("compact_owned_loop_deferred_learner_submit_count") or 0
        ),
        "compact_owned_loop_deferred_learner_completed_count": int(
            profile_payload.get("compact_owned_loop_deferred_learner_completed_count") or 0
        ),
        "compact_owned_loop_deferred_learner_pending": bool(
            profile_payload.get("compact_owned_loop_deferred_learner_pending", False)
        ),
        "compact_owned_loop_deferred_learner_pending_count": int(
            profile_payload.get("compact_owned_loop_deferred_learner_pending_count") or 0
        ),
        "compact_owned_loop_deferred_learner_max_pending": int(
            profile_payload.get("compact_owned_loop_deferred_learner_max_pending") or 0
        ),
        "compact_owned_loop_deferred_learner_max_pending_observed": int(
            profile_payload.get("compact_owned_loop_deferred_learner_max_pending_observed") or 0
        ),
        "compact_owned_loop_deferred_learner_actor_steps_while_pending": int(
            profile_payload.get("compact_owned_loop_deferred_learner_actor_steps_while_pending")
            or 0
        ),
        "compact_owned_loop_deferred_learner_policy_lag_current": int(
            profile_payload.get("compact_owned_loop_deferred_learner_policy_lag_current") or 0
        ),
        "compact_owned_loop_deferred_learner_policy_lag_max": int(
            profile_payload.get("compact_owned_loop_deferred_learner_policy_lag_max") or 0
        ),
        "compact_owned_loop_deferred_learner_wait_count": int(
            profile_payload.get("compact_owned_loop_deferred_learner_wait_count") or 0
        ),
        "compact_owned_loop_deferred_learner_wait_sec": float(
            profile_payload.get("compact_owned_loop_deferred_learner_wait_sec") or 0.0
        ),
        "compact_owned_loop_deferred_learner_last_wait_sec": float(
            profile_payload.get("compact_owned_loop_deferred_learner_last_wait_sec") or 0.0
        ),
    }


def _deferred_sample_learner_proof_fields(
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "compact_owned_loop_sample_learner_worker_kind": str(
            profile_payload.get("compact_owned_loop_sample_learner_worker_kind") or "none"
        ),
        "compact_owned_loop_sample_learner_worker_resource_id": str(
            profile_payload.get("compact_owned_loop_sample_learner_worker_resource_id") or "none"
        ),
        "compact_owned_loop_actor_search_resource_id": str(
            profile_payload.get("compact_owned_loop_actor_search_resource_id")
            or "local_process:actor-search"
        ),
        "compact_owned_loop_actor_search_pid": int(
            profile_payload.get("compact_owned_loop_actor_search_pid") or 0
        ),
        "compact_owned_loop_sample_learner_worker_parent_pid": int(
            profile_payload.get("compact_owned_loop_sample_learner_worker_parent_pid") or 0
        ),
        "compact_owned_loop_sample_learner_worker_resource_scope": str(
            profile_payload.get("compact_owned_loop_sample_learner_worker_resource_scope") or "none"
        ),
        "compact_owned_loop_sample_learner_worker_start_method": str(
            profile_payload.get("compact_owned_loop_sample_learner_worker_start_method") or "none"
        ),
        "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings": str(
            profile_payload.get(
                "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings"
            )
            or "none"
        ),
        "compact_owned_loop_sample_learner_worker_bootstrap_source": str(
            profile_payload.get("compact_owned_loop_sample_learner_worker_bootstrap_source")
            or "none"
        ),
        "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": bool(
            profile_payload.get(
                "compact_owned_loop_sample_learner_resource_distinct_from_actor_search",
                False,
            )
        ),
        "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search": bool(
            profile_payload.get(
                "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search",
                False,
            )
        ),
        "compact_owned_loop_deferred_sample_learner_submit_count": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_submit_count") or 0
        ),
        "compact_owned_loop_deferred_sample_learner_completed_count": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_completed_count") or 0
        ),
        "compact_owned_loop_deferred_sample_learner_pending": bool(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_pending", False)
        ),
        "compact_owned_loop_deferred_sample_learner_pending_count": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_pending_count") or 0
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_max_pending") or 0
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_observed": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_max_pending_observed")
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_policy_lag_current": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_policy_lag_current")
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_policy_lag_max": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_policy_lag_max") or 0
        ),
        "compact_owned_loop_deferred_sample_learner_last_submitted_request_id": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_last_submitted_request_id"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_request_id": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_last_completed_request_id"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_last_submitted_snapshot_version": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_last_submitted_snapshot_version"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_snapshot_version": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_last_completed_snapshot_version"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id": str(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id"
            )
            or "none"
        ),
        "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device": str(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device"
            )
            or "none"
        ),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "last_completed_worker_pid_distinct_from_actor_search"
        ): bool(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "last_completed_worker_pid_distinct_from_actor_search"
                ),
                False,
            )
        ),
        "compact_owned_loop_deferred_sample_learner_model_state_apply_count": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_model_state_apply_count"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_last_model_state_applied": bool(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_last_model_state_applied",
                False,
            )
        ),
        "compact_owned_loop_deferred_sample_learner_model_state_interval": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_model_state_interval")
            or 1
        ),
        ("compact_owned_loop_deferred_sample_learner_model_state_transport_kind"): str(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_model_state_transport_kind")
            )
            or COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1
        ),
        ("compact_owned_loop_deferred_sample_learner_replay_append_transport_kind"): str(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_transport_kind")
            )
            or "durable_entry_v1"
        ),
        "compact_owned_loop_deferred_sample_learner_model_state_return_count": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_model_state_return_count"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_model_state_omitted_count": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_model_state_omitted_count"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_last_model_state_returned": bool(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_last_model_state_returned",
                False,
            )
        ),
        ("compact_owned_loop_deferred_sample_learner_model_owner_ref_return_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_model_owner_ref_return_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_returned"): bool(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_returned"),
                False,
            )
        ),
        ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_digest"): str(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_digest"),
                "",
            )
        ),
        ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_worker_pid"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_worker_pid")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_return_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_return_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_publish_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_publish_bytes")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_publish_sec"): float(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_publish_sec")
            )
            or 0.0
        ),
        ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_bytes")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_sec"): float(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_sec")
            )
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_request_host_only": bool(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_request_host_only",
                False,
            )
        ),
        "compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_result_host_only": bool(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_result_host_only",
                False,
            )
        ),
        "compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_snapshot_host_clone_used": bool(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_snapshot_host_clone_used",
                False,
            )
        ),
        "compact_owned_loop_deferred_sample_learner_request_bytes": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_request_bytes") or 0
        ),
        "compact_owned_loop_deferred_sample_learner_result_bytes": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_result_bytes") or 0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_owns_model_state": bool(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_owns_model_state",
                False,
            )
        ),
        "compact_owned_loop_deferred_sample_learner_worker_owns_replay_store": bool(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_owns_replay_store",
                False,
            )
        ),
        "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent": bool(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent",
                False,
            )
        ),
        ("compact_owned_loop_deferred_sample_learner_full_replay_snapshot_submit_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_full_replay_snapshot_submit_count")
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_replay_append_entry_count": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_replay_append_entry_count"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_replay_append_index_row_count": int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_index_row_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_last_replay_append_entry_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_last_replay_append_entry_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_last_replay_append_index_row_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_last_replay_append_index_row_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_replay_append_entry_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_entry_bytes")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_replay_append_host_observation_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_host_observation_bytes")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_resident_snapshot_bytes")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_replay_append_compact_batch_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_compact_batch_bytes")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_replay_append_step_payload_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_step_payload_bytes")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_replay_append_render_state_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_render_state_bytes")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_last_provider_bootstrap_step_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_last_provider_bootstrap_step_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_bytes")
            )
            or 0
        ),
        (
            "compact_owned_loop_deferred_sample_learner_provider_bootstrap_host_observation_bytes"
        ): int(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_host_observation_bytes"
                )
            )
            or 0
        ),
        (
            "compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_count"
        ): int(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_resident_snapshot_count"
                )
            )
            or 0
        ),
        (
            "compact_owned_loop_deferred_sample_learner_provider_bootstrap_resident_snapshot_bytes"
        ): int(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_resident_snapshot_bytes"
                )
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_render_state_bytes"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_render_state_bytes")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_entry_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_entry_count")
            )
            or 0
        ),
        (
            "compact_owned_loop_deferred_sample_learner_provider_bootstrap_replay_index_row_count"
        ): int(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_replay_index_row_count"
                )
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_learner_call_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_learner_call_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_worker_observation_provider_present"): bool(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_worker_observation_provider_present"),
                False,
            )
        ),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_bootstrap_step_count"
        ): int(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_observation_provider_bootstrap_step_count"
                )
            )
            or 0
        ),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_last_observation_provider_bootstrap_step_count"
        ): int(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_last_observation_provider_bootstrap_step_count"
                )
            )
            or 0
        ),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_missing_stack_history_count"
        ): int(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_observation_provider_missing_stack_history_count"
                )
            )
            or 0
        ),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_materialized_entry_count"
        ): int(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_observation_provider_materialized_entry_count"
                )
            )
            or 0
        ),
        (
            "compact_owned_loop_deferred_sample_learner_"
            "worker_last_observation_provider_materialized_entry_count"
        ): int(
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_last_observation_provider_materialized_entry_count"
                )
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_worker_model_initialized_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_worker_model_initialized_count")
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_completed_count": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_worker_completed_count")
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_job_wall_sec": float(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_worker_job_wall_sec")
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_inner_job_wall_sec": float(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_inner_job_wall_sec"
            )
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_replay_prepare_sec": float(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_replay_prepare_sec"
            )
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_sample_sec": float(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_worker_sample_sec")
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_learner_sec": float(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_worker_learner_sec")
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_model_state_prepare_sec": float(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_worker_model_state_prepare_sec")
            )
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_model_state_fn_sec": float(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_model_state_fn_sec"
            )
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_model_state_clone_sec": float(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_worker_model_state_clone_sec")
            )
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_model_state_digest_sec": float(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_worker_model_state_digest_sec")
            )
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_result_public_sec": float(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_result_public_sec"
            )
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_result_pickle_sec": float(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_result_pickle_sec"
            )
            or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_replay_append_count": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_replay_append_count"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_replay_entry_count": int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_replay_entry_count"
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_worker_replay_index_row_count": int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_worker_replay_index_row_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_worker_replay_evicted_entry_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_worker_replay_evicted_entry_count")
            )
            or 0
        ),
        ("compact_owned_loop_deferred_sample_learner_worker_replay_evicted_index_row_count"): int(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_worker_replay_evicted_index_row_count")
            )
            or 0
        ),
        "compact_owned_loop_deferred_sample_learner_wait_count": int(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_wait_count") or 0
        ),
        "compact_owned_loop_deferred_sample_learner_wait_sec": float(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_wait_sec") or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_last_wait_sec": float(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_last_wait_sec") or 0.0
        ),
        "compact_owned_loop_deferred_sample_learner_drained": bool(
            profile_payload.get("compact_owned_loop_deferred_sample_learner_drained", False)
        ),
        "compact_owned_loop_final_deferred_drain_sec": float(
            profile_payload.get("compact_owned_loop_final_deferred_drain_sec") or 0.0
        ),
        "compact_owned_loop_final_deferred_sample_learner_drain_sec": float(
            profile_payload.get("compact_owned_loop_final_deferred_sample_learner_drain_sec") or 0.0
        ),
        "compact_owned_loop_final_deferred_learner_drain_sec": float(
            profile_payload.get("compact_owned_loop_final_deferred_learner_drain_sec") or 0.0
        ),
        "compact_owned_loop_final_deferred_drain_in_measured_sec": bool(
            profile_payload.get("compact_owned_loop_final_deferred_drain_in_measured_sec", False)
        ),
    }


def _owner_search_normalized_owner_sample_telemetry_for_proof(
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    owner_sample_telemetry_raw = profile_payload.get("compact_owner_search_owner_sample_telemetry")
    owner_sample_telemetry = (
        dict(owner_sample_telemetry_raw) if isinstance(owner_sample_telemetry_raw, Mapping) else {}
    )
    owner_learner_telemetry_raw = profile_payload.get(
        "compact_owner_search_owner_learner_telemetry"
    )
    owner_learner_telemetry = (
        dict(owner_learner_telemetry_raw)
        if isinstance(owner_learner_telemetry_raw, Mapping)
        else {}
    )
    def _missing(
        name: str,
        *,
        replace_false: bool = False,
        replace_zero: bool = False,
        replace_none_string: bool = False,
    ) -> bool:
        if name not in owner_sample_telemetry:
            return True
        value = owner_sample_telemetry[name]
        if value is None or value == "":
            return True
        if replace_false and value is False:
            return True
        if replace_zero and value == 0:
            return True
        return bool(replace_none_string and value == "none")

    def _set_from_value(
        name: str,
        value: Any,
        *,
        replace_false: bool = False,
        replace_zero: bool = False,
        replace_none_string: bool = False,
    ) -> None:
        if value is None or value == "":
            return
        if not _missing(
            name,
            replace_false=replace_false,
            replace_zero=replace_zero,
            replace_none_string=replace_none_string,
        ):
            return
        owner_sample_telemetry[name] = value

    def _set_from_learner(
        name: str,
        *learner_names: str,
        replace_false: bool = False,
        replace_zero: bool = False,
        replace_none_string: bool = False,
    ) -> None:
        for learner_name in learner_names:
            if learner_name not in owner_learner_telemetry:
                continue
            _set_from_value(
                name,
                owner_learner_telemetry[learner_name],
                replace_false=replace_false,
                replace_zero=replace_zero,
                replace_none_string=replace_none_string,
            )
            if not _missing(
                name,
                replace_false=replace_false,
                replace_zero=replace_zero,
                replace_none_string=replace_none_string,
            ):
                return

    def _set_positive_int_from_learner(name: str, *learner_names: str) -> None:
        if not _missing(name, replace_zero=True):
            return
        for learner_name in learner_names:
            value = owner_learner_telemetry.get(learner_name)
            try:
                int_value = int(value or 0)
            except (TypeError, ValueError):
                continue
            if int_value > 0:
                owner_sample_telemetry[name] = int_value
                return

    def _owner_local_value(name: str) -> Any:
        if name in owner_sample_telemetry:
            return owner_sample_telemetry[name]
        if name in profile_payload:
            return profile_payload[name]
        return None

    def _positive_owner_local_int(name: str) -> int:
        try:
            return int(_owner_local_value(name) or 0)
        except (TypeError, ValueError):
            return 0

    def _owner_local_bool(name: str) -> bool:
        return bool(_owner_local_value(name))

    def _overlay_owner_local_transition_batch_projection() -> None:
        if not _owner_local_bool("compact_owner_search_owner_local_transition_derivation_used"):
            return
        batch_count = _positive_owner_local_int(
            "compact_owner_search_owner_local_transition_derivation_batch_count"
        )
        transition_count = _positive_owner_local_int(
            "compact_owner_search_owner_local_transition_derivation_transition_count"
        )
        transport_entry_count = _positive_owner_local_int(
            "compact_owner_search_owner_local_transition_derivation_transport_entry_count"
        )
        transport_bytes = _positive_owner_local_int(
            "compact_owner_search_owner_local_transition_derivation_transport_bytes"
        )
        pending_count = _positive_owner_local_int(
            "compact_owner_search_owner_local_transition_derivation_pending_count"
        )
        fallback_count = _positive_owner_local_int(
            "compact_owner_search_owner_local_transition_derivation_fallback_count"
        )
        digest = str(
            _owner_local_value("compact_owner_search_owner_local_transition_derivation_digest")
            or ""
        )
        digest_verified = bool(
            _owner_local_value(
                "compact_owner_search_owner_local_transition_derivation_digest_verified"
            )
        )
        requested_batch_size = _positive_owner_local_int(
            "owner_search_transition_batch_size_requested"
        )
        fixed_capacity = _positive_owner_local_int(
            "compact_owner_search_transition_batch_fixed_capacity"
        )
        if fixed_capacity <= 0:
            fixed_capacity = requested_batch_size
        max_entries = _positive_owner_local_int(
            "compact_owner_search_transition_batch_max_entries_per_batch"
        )
        if max_entries <= 0:
            max_entries = fixed_capacity
        padding_count = _positive_owner_local_int(
            "compact_owner_search_transition_batch_padding_count"
        )
        if fixed_capacity > 0 and transition_count > 0 and batch_count > 0:
            padding_count = max(0, batch_count * fixed_capacity - transition_count)
        owner_sample_telemetry.update(
            {
                "compact_owner_search_owner_replay_transport_kind": (
                    COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
                ),
                "compact_owner_search_owner_replay_transport_entry_count": int(
                    transport_entry_count
                ),
                "compact_owner_search_owner_replay_transition_batch_enabled": True,
                "compact_owner_search_owner_replay_transition_batch_count": int(batch_count),
                "compact_owner_search_owner_replay_transition_batch_transition_count": int(
                    transition_count
                ),
                "compact_owner_search_owner_replay_transition_legacy_entry_count": 0,
                "compact_owner_search_transition_batch_transport_requested": True,
                "compact_owner_search_transition_batch_transport_enabled": True,
                "compact_owner_search_transition_batch_transport_kind": (
                    COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
                ),
                "compact_owner_search_transition_batch_schema_id": (
                    COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
                ),
                "compact_owner_search_transition_batch_count": int(batch_count),
                "compact_owner_search_transition_batch_entry_count": int(transition_count),
                "compact_owner_search_transition_batch_transport_entry_count": int(
                    transport_entry_count
                ),
                "compact_owner_search_transition_batch_max_entries_per_batch": int(max_entries),
                "compact_owner_search_transition_batch_fixed_capacity": int(fixed_capacity),
                "compact_owner_search_transition_batch_padding_count": int(padding_count),
                "compact_owner_search_transition_batch_overflow_count": 0,
                "compact_owner_search_transition_batch_fallback_count": int(fallback_count),
                "compact_owner_search_transition_batch_fallback_reason": str(
                    _owner_local_value(
                        "compact_owner_search_owner_local_transition_derivation_fallback_reason"
                    )
                    or "none"
                ),
                "compact_owner_search_transition_batch_pending_count": int(pending_count),
                "compact_owner_search_transition_batch_transport_bytes": int(transport_bytes),
                "compact_owner_search_transition_batch_digest": digest,
                "compact_owner_search_transition_batch_digest_verified": bool(
                    digest_verified or (digest and transition_count > 0)
                ),
                "compact_owner_search_transition_batch_build_sec": float(
                    _owner_local_value(
                        "compact_owner_search_owner_local_transition_derivation_build_sec"
                    )
                    or 0.0
                ),
                "compact_owner_search_transition_batch_submit_sec": float(
                    _owner_local_value(
                        "compact_owner_search_owner_local_transition_derivation_submit_sec"
                    )
                    or 0.0
                ),
            }
        )

    _overlay_owner_local_transition_batch_projection()
    if not owner_learner_telemetry:
        return owner_sample_telemetry

    for learner_name, value in owner_learner_telemetry.items():
        if learner_name.startswith("compact_rollout_slab_sample_gate_"):
            _set_from_value(learner_name, value)

    _set_positive_int_from_learner(
        "compact_rollout_slab_sample_gate_sample_row_count",
        "sample_row_count",
        "compact_muzero_learner_sample_rows",
        "compact_muzero_learner_batch_rows",
        "terminal_sample_row_count",
    )
    _set_positive_int_from_learner(
        "compact_rollout_slab_sample_gate_target_row_count",
        "target_row_count",
        "terminal_unroll_value_target_row_count",
        "compact_muzero_learner_batch_rows",
        "sample_row_count",
    )
    _set_positive_int_from_learner(
        "compact_rollout_slab_sample_gate_requested_sample_row_count",
        "requested_sample_row_count",
    )
    owner_sample_batch_size = int(
        profile_payload.get("compact_owner_search_owner_sample_batch_size") or 0
    )
    if owner_sample_batch_size > 0 and _missing(
        "compact_rollout_slab_sample_gate_requested_sample_row_count",
        replace_zero=True,
    ):
        owner_sample_telemetry["compact_rollout_slab_sample_gate_requested_sample_row_count"] = (
            owner_sample_batch_size
        )

    _set_from_learner(
        "compact_rollout_slab_sample_gate_require_next_targets",
        "require_next_targets",
        replace_false=True,
    )
    _set_from_learner(
        "compact_rollout_slab_sample_gate_explicit_next_targets",
        "explicit_next_targets",
        "require_next_targets",
        replace_false=True,
    )
    _set_from_learner(
        "compact_rollout_slab_sample_gate_explicit_unroll_targets",
        "explicit_unroll_targets",
        replace_false=True,
    )
    _set_positive_int_from_learner(
        "compact_rollout_slab_sample_gate_explicit_unroll_target_group_count",
        "explicit_unroll_target_group_count",
    )
    _set_positive_int_from_learner(
        "compact_rollout_slab_sample_gate_num_unroll_steps",
        "num_unroll_steps",
        "compact_muzero_learner_num_unroll_steps",
    )
    _set_from_learner(
        "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported",
        "terminal_unroll_windows_supported",
        replace_false=True,
    )
    terminal_sample_gate_aliases = {
        "compact_rollout_slab_sample_gate_terminal_sample_row_count": ("terminal_sample_row_count"),
        "compact_rollout_slab_sample_gate_next_final_observation_row_count": (
            "next_final_observation_row_count"
        ),
        "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count": (
            "terminal_unroll_value_target_row_count"
        ),
        "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode": (
            "terminal_unroll_value_target_mode"
        ),
        "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used": (
            "resident_terminal_final_observation_used"
        ),
        "compact_rollout_slab_sample_gate_host_terminal_final_observation_used": (
            "host_terminal_final_observation_used"
        ),
        "compact_rollout_slab_sample_gate_device_replay_index_rows": (
            "device_replay_index_rows_sample"
        ),
    }
    for sample_name, learner_name in terminal_sample_gate_aliases.items():
        _set_from_learner(
            sample_name,
            learner_name,
            replace_false=True,
            replace_zero=True,
            replace_none_string=True,
        )
    for suffix in (
        "group_count",
        "index_fast_path_count",
        "fallback_count",
        "validate_only_count",
        "materialized_count",
        "final_row_count_sum",
        "final_row_count_max",
        "dense_storage_count",
        "sparse_storage_count",
        "missing_storage_count",
        "sparse_row_count_sum",
        "sparse_row_count_max",
    ):
        _set_from_learner(
            f"compact_rollout_slab_sample_gate_terminal_final_observation_{suffix}",
            f"terminal_final_observation_{suffix}",
            replace_zero=True,
        )

    learner_batch_schema_id = str(
        owner_learner_telemetry.get("compact_muzero_learner_batch_schema_id") or ""
    )
    direct_sample_schema_id = str(owner_learner_telemetry.get("sample_schema_id") or "")
    learner_batch_rows = int(owner_learner_telemetry.get("compact_muzero_learner_batch_rows") or 0)
    if learner_batch_schema_id == COMPACT_MUZERO_LEARNER_BATCH_SCHEMA_ID or learner_batch_rows > 0:
        _set_from_value(
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch",
            True,
            replace_false=True,
        )
    if (
        direct_sample_schema_id == COMPACT_MUZERO_DIRECT_LEARNER_BATCH_SCHEMA_ID
        and learner_batch_rows > 0
    ):
        _set_from_value(
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only",
            True,
            replace_false=True,
        )

    compact_muzero_sample_gate_aliases = {
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source": (
            "compact_muzero_learner_batch_prevalidation_source"
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes": (
            "compact_muzero_learner_input_h2d_bytes"
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order": (
            "compact_muzero_learner_batch_sample_order"
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order": (
            "compact_muzero_learner_batch_preserves_sample_order"
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count": (
            "compact_muzero_learner_batch_order_restore_index_copy_count"
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": (
            "compact_muzero_learner_batch_unroll_builder_path"
        ),
    }
    for sample_name, learner_name in compact_muzero_sample_gate_aliases.items():
        _set_from_learner(
            sample_name,
            learner_name,
            replace_zero=True,
            replace_none_string=True,
        )

    learner_batch_builder_alias_groups = (
        ("learner_ready_unroll2_cache", "learner_ready_unroll2_cache"),
        ("tensor_native_replay", "tensor_native_replay"),
        ("unroll2_specialized_builder", "unroll2_specialized_builder"),
    )
    learner_batch_builder_suffixes = (
        "requested",
        "eligible_count",
        "available_group_count",
        "used",
        "call_count",
        "fallback_count",
        "fallback_reason",
        "impl",
        "table_source",
        "table_reused_record_count",
        "table_missing_record_count",
        "table_rows",
        "table_concat_sec",
        "gather_sec",
    )
    for sample_group, learner_group in learner_batch_builder_alias_groups:
        for suffix in learner_batch_builder_suffixes:
            _set_from_learner(
                (f"compact_rollout_slab_sample_gate_learner_batch_builder_{sample_group}_{suffix}"),
                f"compact_muzero_learner_batch_{learner_group}_{suffix}",
                replace_false=True,
                replace_zero=True,
                replace_none_string=True,
            )

    _overlay_owner_local_transition_batch_projection()
    return owner_sample_telemetry


def _owner_search_slab_proxy_proof_fields(
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    slab_telemetry_raw = profile_payload.get("compact_rollout_slab_last_telemetry")
    slab_telemetry = dict(slab_telemetry_raw) if isinstance(slab_telemetry_raw, Mapping) else {}
    search_metadata = _latest_compact_rollout_slab_search_metadata(profile_payload)
    owner_learner_telemetry_raw = profile_payload.get(
        "compact_owner_search_owner_learner_telemetry"
    )
    owner_learner_telemetry = (
        dict(owner_learner_telemetry_raw)
        if isinstance(owner_learner_telemetry_raw, Mapping)
        else {}
    )
    owner_sample_telemetry = _owner_search_normalized_owner_sample_telemetry_for_proof(
        profile_payload
    )
    owner_sample_claims_owner_local_derivation = bool(
        owner_sample_telemetry.get("compact_owner_search_owner_local_transition_derivation_used")
        or owner_sample_telemetry.get(
            "compact_owner_search_owner_local_transition_derivation_requested"
        )
    )

    def _prefer_owner_sample_for_owner_local(name: str) -> bool:
        if not owner_sample_claims_owner_local_derivation:
            return False
        if str(name).startswith("compact_owner_search_owner_local_transition_derivation_"):
            return True
        return str(name) in {
            "compact_owner_search_owner_replay_transport_kind",
            "compact_owner_search_owner_replay_transport_entry_count",
            "compact_owner_search_owner_replay_transition_batch_enabled",
            "compact_owner_search_owner_replay_transition_batch_count",
            "compact_owner_search_owner_replay_transition_batch_transition_count",
            "compact_owner_search_owner_replay_transition_legacy_entry_count",
            "compact_owner_search_transition_batch_transport_requested",
            "compact_owner_search_transition_batch_transport_enabled",
            "compact_owner_search_transition_batch_transport_kind",
            "compact_owner_search_transition_batch_schema_id",
            "compact_owner_search_transition_batch_count",
            "compact_owner_search_transition_batch_entry_count",
            "compact_owner_search_transition_batch_transport_entry_count",
            "compact_owner_search_transition_batch_max_entries_per_batch",
            "compact_owner_search_transition_batch_fixed_capacity",
            "compact_owner_search_transition_batch_padding_count",
            "compact_owner_search_transition_batch_overflow_count",
            "compact_owner_search_transition_batch_fallback_count",
            "compact_owner_search_transition_batch_fallback_reason",
            "compact_owner_search_transition_batch_pending_count",
            "compact_owner_search_transition_batch_transport_bytes",
            "compact_owner_search_transition_batch_digest",
            "compact_owner_search_transition_batch_digest_verified",
        }

    def _proof_value(name: str, default: Any = None) -> Any:
        if _prefer_owner_sample_for_owner_local(name) and name in owner_sample_telemetry:
            return owner_sample_telemetry[name]
        if name in profile_payload:
            return profile_payload[name]
        if name in slab_telemetry:
            return slab_telemetry[name]
        if name in search_metadata:
            return search_metadata[name]
        if name in owner_sample_telemetry:
            return owner_sample_telemetry[name]
        return default

    def _proof_int(name: str, default: int = 0) -> int:
        value = _proof_value(name, default)
        try:
            return int(value or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"owner-search proof field {name} expected int, got {value!r}"
            ) from exc

    def _proof_float(name: str, default: float = 0.0) -> float:
        value = _proof_value(name, default)
        try:
            return float(value or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"owner-search proof field {name} expected float, got {value!r}"
            ) from exc

    def _proof_bool(name: str, default: bool = False) -> bool:
        return bool(_proof_value(name, default))

    def _proof_str(name: str, default: str = "") -> str:
        return str(_proof_value(name, default) or "")

    resident_handle_fields = _owner_search_learner_resident_batch_handle_fields(
        metadata=owner_learner_telemetry
    )

    return {
        "compact_owner_search_slab_proxy": bool(
            profile_payload.get("compact_owner_search_slab_proxy", False)
        ),
        "compact_owner_search_lazy_slab_proxy": bool(
            profile_payload.get("compact_owner_search_lazy_slab_proxy", False)
        ),
        "compact_owner_search_inline_slab_proxy": bool(
            profile_payload.get("compact_owner_search_inline_slab_proxy", False)
            or search_metadata.get("compact_owner_search_inline_slab_proxy", False)
        ),
        "compact_owner_search_inline_background_slab_proxy": bool(
            profile_payload.get(
                "compact_owner_search_inline_background_slab_proxy",
                False,
            )
            or search_metadata.get(
                "compact_owner_search_inline_background_slab_proxy",
                False,
            )
        ),
        "compact_owner_search_threaded_slab_proxy": bool(
            profile_payload.get("compact_owner_search_threaded_slab_proxy", False)
            or search_metadata.get("compact_owner_search_threaded_slab_proxy", False)
        ),
        "compact_owner_search_slab_bypass": bool(
            profile_payload.get("compact_owner_search_slab_bypass", False)
            or search_metadata.get("compact_owner_search_slab_bypass", False)
        ),
        "compact_owner_search_slab_bypass_kind": str(
            profile_payload.get("compact_owner_search_slab_bypass_kind")
            or search_metadata.get("compact_owner_search_slab_bypass_kind")
            or "none"
        ),
        "compact_rollout_slab_bypassed": bool(
            profile_payload.get("compact_rollout_slab_bypassed", False)
            or search_metadata.get("compact_rollout_slab_bypassed", False)
        ),
        "compact_rollout_slab_general_replay_row_builder_used": bool(
            profile_payload.get("compact_rollout_slab_general_replay_row_builder_used", False)
            or search_metadata.get(
                "compact_rollout_slab_general_replay_row_builder_used",
                False,
            )
        ),
        "compact_rollout_slab_resident_host_observation_stub_requested": _proof_bool(
            "compact_rollout_slab_resident_host_observation_stub_requested"
        ),
        "compact_rollout_slab_resident_host_observation_stubbed": _proof_bool(
            "compact_rollout_slab_resident_host_observation_stubbed"
        ),
        "compact_rollout_slab_resident_host_observation_stub_kind": _proof_str(
            "compact_rollout_slab_resident_host_observation_stub_kind"
        ),
        "compact_rollout_slab_resident_host_observation_stub_materialized_bytes": _proof_int(
            "compact_rollout_slab_resident_host_observation_stub_materialized_bytes"
        ),
        "compact_rollout_slab_resident_host_observation_stub_logical_bytes": _proof_int(
            "compact_rollout_slab_resident_host_observation_stub_logical_bytes"
        ),
        "compact_owner_search_direct_root_build_request_requested": _proof_bool(
            "compact_owner_search_direct_root_build_request_requested"
        ),
        "compact_rollout_slab_parent_root_batch_build_avoided": _proof_bool(
            "compact_rollout_slab_parent_root_batch_build_avoided"
        ),
        "compact_rollout_slab_parent_root_batch_builder_used": _proof_bool(
            "compact_rollout_slab_parent_root_batch_builder_used"
        ),
        "compact_rollout_slab_parent_root_batch_builder_call_count": _proof_int(
            "compact_rollout_slab_parent_root_batch_builder_call_count"
        ),
        "compact_rollout_slab_root_batch_build_sec": _proof_float(
            "compact_rollout_slab_root_batch_build_sec"
        ),
        "compact_rollout_slab_root_build_request_sec": _proof_float(
            "compact_rollout_slab_root_build_request_sec"
        ),
        "compact_owner_action_step_boundary_enabled": _proof_bool(
            "compact_owner_action_step_boundary_enabled"
        ),
        "compact_owner_action_step_boundary_proof_passed": _proof_bool(
            "compact_owner_action_step_boundary_proof_passed"
        ),
        "compact_owner_action_step_boundary_step_count": _proof_int(
            "compact_owner_action_step_boundary_step_count"
        ),
        "compact_owner_action_step_boundary_seeded_action_count": _proof_int(
            "compact_owner_action_step_boundary_seeded_action_count"
        ),
        "compact_owner_action_step_boundary_feedback_action_count": _proof_int(
            "compact_owner_action_step_boundary_feedback_action_count"
        ),
        "compact_owner_action_step_boundary_action_verified_count": _proof_int(
            "compact_owner_action_step_boundary_action_verified_count"
        ),
        "compact_owner_action_step_boundary_next_action_count": _proof_int(
            "compact_owner_action_step_boundary_next_action_count"
        ),
        "compact_owner_action_step_boundary_last_action_source": _proof_str(
            "compact_owner_action_step_boundary_last_action_source"
        ),
        "compact_owner_action_step_boundary_last_applied_action_checksum": _proof_int(
            "compact_owner_action_step_boundary_last_applied_action_checksum"
        ),
        "compact_owner_action_step_boundary_last_next_action_checksum": _proof_int(
            "compact_owner_action_step_boundary_last_next_action_checksum"
        ),
        "compact_owner_action_step_boundary_failure_reason": _proof_str(
            "compact_owner_action_step_boundary_failure_reason"
        ),
        "compact_owner_mechanics_step_boundary_enabled": _proof_bool(
            "compact_owner_mechanics_step_boundary_enabled"
        ),
        "compact_owner_mechanics_step_boundary": _proof_bool(
            "compact_owner_mechanics_step_boundary"
        ),
        "compact_owner_mechanics_step_view_schema_id": _proof_str(
            "compact_owner_mechanics_step_view_schema_id"
        ),
        "compact_owner_mechanics_step_frame_slot_schema_id": _proof_str(
            "compact_owner_mechanics_step_frame_slot_schema_id"
        ),
        "compact_owner_mechanics_step_boundary_count": _proof_int(
            "compact_owner_mechanics_step_boundary_count"
        ),
        "compact_owner_mechanics_parent_compact_batch_builder_call_count": _proof_int(
            "compact_owner_mechanics_parent_compact_batch_builder_call_count"
        ),
        "compact_owner_mechanics_parent_compact_batch_object_count": _proof_int(
            "compact_owner_mechanics_parent_compact_batch_object_count"
        ),
        "compact_owner_mechanics_parent_compact_batch_builder_used": _proof_bool(
            "compact_owner_mechanics_parent_compact_batch_builder_used"
        ),
        "compact_owner_mechanics_step_view_object_count": _proof_int(
            "compact_owner_mechanics_step_view_object_count"
        ),
        "compact_owner_mechanics_host_observation_bytes_sent": _proof_int(
            "compact_owner_mechanics_host_observation_bytes_sent"
        ),
        "compact_owner_mechanics_host_final_observation_bytes_sent": _proof_int(
            "compact_owner_mechanics_host_final_observation_bytes_sent"
        ),
        "compact_owner_mechanics_resident_observation_handle_present": _proof_bool(
            "compact_owner_mechanics_resident_observation_handle_present"
        ),
        "compact_owner_mechanics_step_frame_handle_schema_id": _proof_str(
            "compact_owner_mechanics_step_frame_handle_schema_id"
        ),
        "compact_owner_mechanics_step_frame_handle_ring_used": _proof_bool(
            "compact_owner_mechanics_step_frame_handle_ring_used"
        ),
        "compact_owner_mechanics_step_frame_handle_published": _proof_bool(
            "compact_owner_mechanics_step_frame_handle_published"
        ),
        "compact_owner_mechanics_step_frame_handle_consumed": _proof_bool(
            "compact_owner_mechanics_step_frame_handle_consumed"
        ),
        "compact_owner_mechanics_step_frame_handle_publish_count": _proof_int(
            "compact_owner_mechanics_step_frame_handle_publish_count"
        ),
        "compact_owner_mechanics_step_frame_handle_consume_count": _proof_int(
            "compact_owner_mechanics_step_frame_handle_consume_count"
        ),
        "compact_owner_mechanics_step_frame_handle_ring_slot_count": _proof_int(
            "compact_owner_mechanics_step_frame_handle_ring_slot_count"
        ),
        "compact_owner_mechanics_step_frame_handle_slot_id": _proof_int(
            "compact_owner_mechanics_step_frame_handle_slot_id"
        ),
        "compact_owner_mechanics_step_frame_handle_generation": _proof_int(
            "compact_owner_mechanics_step_frame_handle_generation"
        ),
        "compact_owner_mechanics_step_frame_handle_digest": _proof_str(
            "compact_owner_mechanics_step_frame_handle_digest"
        ),
        "compact_owner_mechanics_step_frame_handle_digest_verified": _proof_bool(
            "compact_owner_mechanics_step_frame_handle_digest_verified"
        ),
        "compact_owner_mechanics_step_frame_handle_owner_digest_verified": _proof_bool(
            "compact_owner_mechanics_step_frame_handle_owner_digest_verified"
        ),
        "compact_owner_mechanics_step_frame_handle_resident_observation_present": (
            _proof_bool(
                "compact_owner_mechanics_step_frame_handle_resident_observation_present"
            )
        ),
        "compact_owner_mechanics_step_frame_slot_write_count": _proof_int(
            "compact_owner_mechanics_step_frame_slot_write_count"
        ),
        "compact_owner_mechanics_parent_step_frame_build_count": _proof_int(
            "compact_owner_mechanics_parent_step_frame_build_count"
        ),
        "compact_owner_action_dispatch_step_overlap_enabled": _proof_bool(
            "compact_owner_action_dispatch_step_overlap_enabled"
        ),
        "compact_owner_action_dispatch_step_overlap_proof_passed": _proof_bool(
            "compact_owner_action_dispatch_step_overlap_proof_passed"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_supported": _proof_bool(
            "compact_rollout_slab_action_dispatch_step_overlap_supported"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_used": _proof_bool(
            "compact_rollout_slab_action_dispatch_step_overlap_used"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait": _proof_bool(
            "compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper": _proof_bool(
            "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count": _proof_int(
            "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_submit_count": _proof_int(
            "compact_rollout_slab_action_dispatch_step_overlap_submit_count"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_resolve_count": _proof_int(
            "compact_rollout_slab_action_dispatch_step_overlap_resolve_count"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_pending_count": _proof_int(
            "compact_rollout_slab_action_dispatch_step_overlap_pending_count"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_max_pending_count": _proof_int(
            "compact_rollout_slab_action_dispatch_step_overlap_max_pending_count"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_submit_wall_sec": _proof_float(
            "compact_rollout_slab_action_dispatch_step_overlap_submit_wall_sec"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_resolve_wall_sec": _proof_float(
            "compact_rollout_slab_action_dispatch_step_overlap_resolve_wall_sec"
        ),
        (
            "compact_rollout_slab_action_dispatch_step_overlap_submit_to_"
            "resolve_elapsed_sec"
        ): _proof_float(
            "compact_rollout_slab_action_dispatch_step_overlap_submit_to_resolve_elapsed_sec"
        ),
        "compact_rollout_slab_action_dispatch_step_overlap_parent_work_sec": _proof_float(
            "compact_rollout_slab_action_dispatch_step_overlap_parent_work_sec"
        ),
        "compact_owner_search_action_dispatch_handle_boundary_supported": _proof_bool(
            "compact_owner_search_action_dispatch_handle_boundary_supported"
        ),
        "compact_owner_search_action_dispatch_handle_used": _proof_bool(
            "compact_owner_search_action_dispatch_handle_used"
        ),
        "compact_owner_search_action_dispatch_handle_schema_id": _proof_str(
            "compact_owner_search_action_dispatch_handle_schema_id"
        ),
        "compact_owner_search_action_dispatch_handle_id": _proof_int(
            "compact_owner_search_action_dispatch_handle_id"
        ),
        "compact_owner_search_action_dispatch_handle_submit_no_wait": _proof_bool(
            "compact_owner_search_action_dispatch_handle_submit_no_wait"
        ),
        "compact_owner_search_action_dispatch_handle_sync_wrapper": _proof_bool(
            "compact_owner_search_action_dispatch_handle_sync_wrapper"
        ),
        "compact_owner_search_action_dispatch_handle_sync_wrapper_count": _proof_int(
            "compact_owner_search_action_dispatch_handle_sync_wrapper_count"
        ),
        "compact_owner_search_action_dispatch_handle_completed_at_submit_count": _proof_int(
            "compact_owner_search_action_dispatch_handle_completed_at_submit_count"
        ),
        "compact_owner_search_action_dispatch_handle_submit_count": _proof_int(
            "compact_owner_search_action_dispatch_handle_submit_count"
        ),
        "compact_owner_search_action_dispatch_handle_resolve_count": _proof_int(
            "compact_owner_search_action_dispatch_handle_resolve_count"
        ),
        "compact_owner_search_action_dispatch_handle_pending_count": _proof_int(
            "compact_owner_search_action_dispatch_handle_pending_count"
        ),
        "compact_owner_search_action_dispatch_handle_max_pending_count": _proof_int(
            "compact_owner_search_action_dispatch_handle_max_pending_count"
        ),
        "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count": _proof_int(
            "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count"
        ),
        "compact_owner_search_action_dispatch_handle_result_wait_sec": _proof_float(
            "compact_owner_search_action_dispatch_handle_result_wait_sec"
        ),
        "compact_owner_root_action_context_handle_used": _proof_bool(
            "compact_owner_root_action_context_handle_used"
        ),
        "compact_owner_root_action_context_handle_schema_id": _proof_str(
            "compact_owner_root_action_context_handle_schema_id"
        ),
        "compact_owner_root_action_context_handle_id": _proof_int(
            "compact_owner_root_action_context_handle_id"
        ),
        "compact_owner_root_action_context_transaction_id": _proof_int(
            "compact_owner_root_action_context_transaction_id"
        ),
        "compact_owner_root_action_context_dispatch_id": _proof_int(
            "compact_owner_root_action_context_dispatch_id"
        ),
        "compact_owner_root_action_context_root_count": _proof_int(
            "compact_owner_root_action_context_root_count"
        ),
        "compact_owner_root_action_context_active_root_count": _proof_int(
            "compact_owner_root_action_context_active_root_count"
        ),
        "compact_owner_root_action_context_context_digest": _proof_str(
            "compact_owner_root_action_context_context_digest"
        ),
        "compact_owner_root_action_context_owner_store_count": _proof_int(
            "compact_owner_root_action_context_owner_store_count"
        ),
        "compact_owner_root_action_context_owner_resolve_count": _proof_int(
            "compact_owner_root_action_context_owner_resolve_count"
        ),
        "compact_owner_root_action_context_owner_release_count": _proof_int(
            "compact_owner_root_action_context_owner_release_count"
        ),
        "compact_owner_root_action_context_owner_pending_count": _proof_int(
            "compact_owner_root_action_context_owner_pending_count"
        ),
        "compact_owner_root_action_context_owner_max_pending_count": _proof_int(
            "compact_owner_root_action_context_owner_max_pending_count"
        ),
        "compact_owner_root_action_context_owner_digest_verified": _proof_bool(
            "compact_owner_root_action_context_owner_digest_verified"
        ),
        "compact_owner_search_pending_root_action_context_stored": _proof_bool(
            "compact_owner_search_pending_root_action_context_stored"
        ),
        "compact_owner_search_action_dispatch_pending_root_action_context_stored": (
            _proof_bool(
                "compact_owner_search_action_dispatch_pending_root_action_context_stored"
            )
        ),
        "compact_owner_search_action_dispatch_pending_root_action_context_store_count": (
            _proof_int(
                "compact_owner_search_action_dispatch_pending_root_action_context_store_count"
            )
        ),
        (
            "compact_owner_search_action_dispatch_pending_root_action_context_"
            "avoided_count"
        ): _proof_int(
            "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count"
        ),
        "compact_owner_search_parent_action_context_validation_count": _proof_int(
            "compact_owner_search_parent_action_context_validation_count"
        ),
        "compact_owner_search_owner_action_context_validation_count": _proof_int(
            "compact_owner_search_owner_action_context_validation_count"
        ),
        "compact_owner_search_slab_bypass_parent_committed_index_rows": int(
            profile_payload.get("compact_owner_search_slab_bypass_parent_committed_index_rows")
            or search_metadata.get("compact_owner_search_slab_bypass_parent_committed_index_rows")
            or 0
        ),
        "compact_owner_search_slab_bypass_parent_stored_index_rows": int(
            profile_payload.get("compact_owner_search_slab_bypass_parent_stored_index_rows")
            or search_metadata.get("compact_owner_search_slab_bypass_parent_stored_index_rows")
            or 0
        ),
        "compact_owner_search_slab_proxy_initialized": bool(
            profile_payload.get("compact_owner_search_slab_proxy_initialized", False)
        ),
        "compact_owner_search_boundary_kind": str(
            profile_payload.get("compact_owner_search_boundary_kind") or "none"
        ),
        "compact_owner_search_parent_slab_commits_replay": bool(
            profile_payload.get("compact_owner_search_parent_slab_commits_replay", False)
        ),
        "compact_owner_search_worker_kind": str(
            profile_payload.get("compact_owner_search_worker_kind") or ""
        ),
        "compact_owner_search_worker_resource_scope": str(
            profile_payload.get("compact_owner_search_worker_resource_scope") or ""
        ),
        "compact_owner_search_worker_resource_distinct_from_actor": bool(
            profile_payload.get(
                "compact_owner_search_worker_resource_distinct_from_actor",
                False,
            )
        ),
        "compact_owner_search_worker_hardware_resource_distinct_from_actor": bool(
            profile_payload.get(
                "compact_owner_search_worker_hardware_resource_distinct_from_actor",
                False,
            )
        ),
        "compact_owner_search_owner_pid": int(
            profile_payload.get("compact_owner_search_owner_pid") or 0
        ),
        "compact_owner_search_root_slot_count": int(
            profile_payload.get("compact_owner_search_root_slot_count") or 0
        ),
        "compact_owner_search_active_root_count": int(
            profile_payload.get("compact_owner_search_active_root_count") or 0
        ),
        "compact_owner_search_request_bytes": int(
            profile_payload.get("compact_owner_search_request_bytes") or 0
        ),
        "compact_owner_search_result_bytes": int(
            profile_payload.get("compact_owner_search_result_bytes") or 0
        ),
        "compact_owner_search_fixed_action_result_buffer_requested": _proof_bool(
            "compact_owner_search_fixed_action_result_buffer_requested"
        ),
        "compact_owner_search_fixed_action_result_buffer_used": _proof_bool(
            "compact_owner_search_fixed_action_result_buffer_used"
        ),
        "compact_owner_search_fixed_action_result_buffer_slot_count": _proof_int(
            "compact_owner_search_fixed_action_result_buffer_slot_count"
        ),
        "compact_owner_search_fixed_action_result_buffer_acquire_count": _proof_int(
            "compact_owner_search_fixed_action_result_buffer_acquire_count"
        ),
        "compact_owner_search_fixed_action_result_buffer_write_count": _proof_int(
            "compact_owner_search_fixed_action_result_buffer_write_count"
        ),
        "compact_owner_search_fixed_action_result_buffer_read_count": _proof_int(
            "compact_owner_search_fixed_action_result_buffer_read_count"
        ),
        "compact_owner_search_fixed_action_result_buffer_slot_id": _proof_int(
            "compact_owner_search_fixed_action_result_buffer_slot_id",
            -1,
        ),
        "compact_owner_search_fixed_action_result_buffer_last_slot_id": _proof_int(
            "compact_owner_search_fixed_action_result_buffer_last_slot_id",
            -1,
        ),
        "compact_owner_search_fixed_action_result_buffer_wire_result_bytes": _proof_int(
            "compact_owner_search_fixed_action_result_buffer_wire_result_bytes"
        ),
        "compact_owner_search_fixed_action_result_buffer_full_result_bytes": _proof_int(
            "compact_owner_search_fixed_action_result_buffer_full_result_bytes"
        ),
        "compact_owner_search_fixed_action_result_buffer_pending_slot_count": _proof_int(
            "compact_owner_search_fixed_action_result_buffer_pending_slot_count"
        ),
        "compact_owner_search_request_cuda_tensor_count": int(
            profile_payload.get("compact_owner_search_request_cuda_tensor_count") or 0
        ),
        "compact_owner_search_result_cuda_tensor_count": int(
            profile_payload.get("compact_owner_search_result_cuda_tensor_count") or 0
        ),
        "compact_owner_search_root_observation_bytes_sent": int(
            profile_payload.get("compact_owner_search_root_observation_bytes_sent") or 0
        ),
        "compact_owner_search_parent_reconstructed_search_result": bool(
            profile_payload.get("compact_owner_search_parent_reconstructed_search_result", False)
        ),
        "compact_owner_search_action_only_result": bool(
            profile_payload.get("compact_owner_search_action_only_result", False)
        ),
        "compact_owner_search_owner_materializes_replay": bool(
            profile_payload.get("compact_owner_search_owner_materializes_replay", False)
        ),
        "compact_owner_search_action_feedback_verified": bool(
            profile_payload.get("compact_owner_search_action_feedback_verified", False)
        ),
        "compact_owner_search_action_feedback_transition_count": int(
            profile_payload.get("compact_owner_search_action_feedback_transition_count") or 0
        ),
        "compact_owner_search_action_feedback_action_count": int(
            profile_payload.get("compact_owner_search_action_feedback_action_count") or 0
        ),
        "compact_owner_search_action_feedback_mismatch_count": int(
            profile_payload.get("compact_owner_search_action_feedback_mismatch_count") or 0
        ),
        "compact_owner_search_expected_joint_action_checksum": int(
            profile_payload.get("compact_owner_search_expected_joint_action_checksum") or 0
        ),
        "compact_owner_search_applied_joint_action_checksum": int(
            profile_payload.get("compact_owner_search_applied_joint_action_checksum") or 0
        ),
        "compact_owner_search_replay_action_checksum": int(
            profile_payload.get("compact_owner_search_replay_action_checksum") or 0
        ),
        "compact_owner_search_inner_two_phase_action_step": bool(
            profile_payload.get("compact_owner_search_inner_two_phase_action_step", False)
        ),
        "compact_owner_search_inner_device_replay_payload_deferred": bool(
            profile_payload.get(
                "compact_owner_search_inner_device_replay_payload_deferred",
                False,
            )
        ),
        "compact_owner_search_use_inner_two_phase_device_replay": bool(
            profile_payload.get(
                "compact_owner_search_use_inner_two_phase_device_replay",
                False,
            )
        ),
        "compact_owner_search_inner_device_replay_payload_flushed_count": _proof_int(
            "compact_owner_search_inner_device_replay_payload_flushed_count"
        ),
        "compact_owner_search_inner_deferred_one_simulation_replay_payload_flush_count": _proof_int(
            "compact_owner_search_inner_deferred_one_simulation_replay_payload_flush_count"
        ),
        "compact_owner_search_inner_deferred_one_simulation_replay_materialized_on_flush_count": _proof_int(
            "compact_owner_search_inner_deferred_one_simulation_replay_materialized_on_flush_count"
        ),
        "compact_owner_search_inner_deferred_one_simulation_replay_recurrent_inference_calls": _proof_float(
            "compact_owner_search_inner_deferred_one_simulation_replay_recurrent_inference_calls"
        ),
        "compact_owner_search_inner_deferred_one_simulation_model_identity_match_count": _proof_int(
            "compact_owner_search_inner_deferred_one_simulation_model_identity_match_count"
        ),
        "compact_owner_search_inner_deferred_one_simulation_model_refresh_crossed_count": _proof_int(
            "compact_owner_search_inner_deferred_one_simulation_model_refresh_crossed_count"
        ),
        "compact_owner_search_inner_pending_deferred_replay_payload_count_max": _proof_int(
            "compact_owner_search_inner_pending_deferred_replay_payload_count_max"
        ),
        "compact_owner_search_inner_pending_deferred_replay_payload_final_count": _proof_int(
            "compact_owner_search_inner_pending_deferred_replay_payload_final_count"
        ),
        "compact_owner_search_inner_deferred_one_simulation_replay_flush_sec": _proof_float(
            "compact_owner_search_inner_deferred_one_simulation_replay_flush_sec"
        ),
        "compact_owner_search_inner_device_replay_payload_flush_sec": _proof_float(
            "compact_owner_search_inner_device_replay_payload_flush_sec"
        ),
        "compact_owner_search_inner_replay_payload_d2h_bytes": _proof_float(
            "compact_owner_search_inner_replay_payload_d2h_bytes"
        ),
        "compact_owner_search_inner_deferred_one_simulation_action_model_state_digest": _proof_str(
            "compact_owner_search_inner_deferred_one_simulation_action_model_state_digest"
        ),
        "compact_owner_search_inner_deferred_one_simulation_flush_model_state_digest": _proof_str(
            "compact_owner_search_inner_deferred_one_simulation_flush_model_state_digest"
        ),
        "compact_owner_search_replay_payload_handle_present": bool(
            profile_payload.get("compact_owner_search_replay_payload_handle_present", False)
        ),
        "compact_owner_search_model_state_bytes": int(
            profile_payload.get("compact_owner_search_model_state_bytes") or 0
        ),
        "compact_owner_search_model_state_return_count": int(
            profile_payload.get("compact_owner_search_model_state_return_count") or 0
        ),
        "compact_owner_search_model_state_snapshot_return_count": int(
            profile_payload.get("compact_owner_search_model_state_snapshot_return_count") or 0
        ),
        "compact_owner_search_search_result_payload_bytes": int(
            profile_payload.get("compact_owner_search_search_result_payload_bytes") or 0
        ),
        "compact_owner_search_search_result_payload_transport_kind": str(
            profile_payload.get("compact_owner_search_search_result_payload_transport_kind") or ""
        ),
        "compact_owner_search_search_result_payload_json_safe": bool(
            profile_payload.get(
                "compact_owner_search_search_result_payload_json_safe",
                True,
            )
        ),
        "compact_owner_search_selected_action_bytes": int(
            profile_payload.get("compact_owner_search_selected_action_bytes") or 0
        ),
        "compact_owner_search_visit_policy_bytes": int(
            profile_payload.get("compact_owner_search_visit_policy_bytes") or 0
        ),
        "compact_owner_search_root_value_bytes": int(
            profile_payload.get("compact_owner_search_root_value_bytes") or 0
        ),
        "compact_owner_search_optional_array_bytes": int(
            profile_payload.get("compact_owner_search_optional_array_bytes") or 0
        ),
        "compact_owner_search_worker_owns_search_state": bool(
            profile_payload.get("compact_owner_search_worker_owns_search_state", False)
        ),
        "compact_owner_search_worker_owns_replay_state": bool(
            profile_payload.get("compact_owner_search_worker_owns_replay_state", False)
        ),
        "compact_owner_search_worker_owns_model_state": bool(
            profile_payload.get("compact_owner_search_worker_owns_model_state", False)
        ),
        "compact_owner_search_consumed_learner_update": bool(
            profile_payload.get("compact_owner_search_consumed_learner_update", False)
        ),
        "compact_owner_search_search_refresh_update_count": int(
            profile_payload.get("compact_owner_search_search_refresh_update_count") or 0
        ),
        "compact_owner_search_model_state_snapshot_load_count": int(
            profile_payload.get("compact_owner_search_model_state_snapshot_load_count") or 0
        ),
        "compact_owner_search_model_state_snapshot_load_bytes": int(
            profile_payload.get("compact_owner_search_model_state_snapshot_load_bytes") or 0
        ),
        "compact_owner_search_model_state_snapshot_load_sec": float(
            profile_payload.get("compact_owner_search_model_state_snapshot_load_sec") or 0.0
        ),
        "compact_owner_search_replay_append_entry_count": int(
            profile_payload.get("compact_owner_search_replay_append_entry_count") or 0
        ),
        "compact_owner_search_replay_append_transport_entry_count": _proof_int(
            "compact_owner_search_replay_append_transport_entry_count"
        ),
        "compact_owner_search_replay_append_transition_batch_count": _proof_int(
            "compact_owner_search_replay_append_transition_batch_count"
        ),
        "compact_owner_search_replay_append_transition_batch_entry_count": _proof_int(
            "compact_owner_search_replay_append_transition_batch_entry_count"
        ),
        "compact_owner_search_replay_append_count": int(
            profile_payload.get("compact_owner_search_replay_append_count") or 0
        ),
        "compact_owner_search_learner_update_count": int(
            profile_payload.get("compact_owner_search_learner_update_count") or 0
        ),
        "compact_owner_search_model_owner_ref_returned": bool(
            profile_payload.get("compact_owner_search_model_owner_ref_returned", False)
        ),
        "compact_owner_search_model_owner_ref_digest": str(
            profile_payload.get("compact_owner_search_model_owner_ref_digest") or ""
        ),
        "compact_owner_search_owner_replay_append_enabled": bool(
            profile_payload.get("compact_owner_search_owner_replay_append_enabled", False)
        ),
        "compact_owner_search_owner_learning_enabled": bool(
            profile_payload.get("compact_owner_search_owner_learning_enabled", False)
        ),
        "compact_owner_search_owner_sample_batch_size": int(
            profile_payload.get("compact_owner_search_owner_sample_batch_size") or 0
        ),
        "compact_owner_search_owner_sample_all_eligible_replay_rows": int(
            profile_payload.get("compact_owner_search_owner_sample_batch_size") or 0
        )
        == 0,
        "compact_owner_search_owner_train_steps": int(
            profile_payload.get("compact_owner_search_owner_train_steps") or 0
        ),
        "compact_owner_search_owner_train_interval": int(
            profile_payload.get("compact_owner_search_owner_train_interval") or 0
        ),
        "compact_owner_search_owner_model_refresh_interval": int(
            profile_payload.get("compact_owner_search_owner_model_refresh_interval") or 0
        ),
        "compact_owner_search_owner_expected_train_request_count": int(
            profile_payload.get("compact_owner_search_owner_expected_train_request_count") or 0
        ),
        "compact_owner_search_owner_defer_maintenance": bool(
            profile_payload.get("compact_owner_search_owner_defer_maintenance", False)
        ),
        "compact_owner_search_owner_loop_schema_id": str(
            profile_payload.get("compact_owner_search_owner_loop_schema_id") or ""
        ),
        "compact_owner_search_owner_loop_kind": str(
            profile_payload.get("compact_owner_search_owner_loop_kind") or ""
        ),
        "compact_owner_search_owner_loop_persistent": bool(
            profile_payload.get("compact_owner_search_owner_loop_persistent", False)
        ),
        "compact_owner_search_owner_action_priority_enabled": bool(
            profile_payload.get("compact_owner_search_owner_action_priority_enabled", False)
        ),
        "compact_owner_search_owner_background_maintenance_thread": bool(
            profile_payload.get(
                "compact_owner_search_owner_background_maintenance_thread",
                False,
            )
        ),
        "compact_owner_search_owner_background_overlap_enabled": bool(
            profile_payload.get(
                "compact_owner_search_owner_background_overlap_enabled",
                False,
            )
        ),
        "compact_owner_search_owner_action_request_count": int(
            profile_payload.get("compact_owner_search_owner_action_request_count") or 0
        ),
        "compact_owner_search_owner_maintenance_request_count": int(
            profile_payload.get("compact_owner_search_owner_maintenance_request_count") or 0
        ),
        "compact_owner_search_owner_run_request_count": int(
            profile_payload.get("compact_owner_search_owner_run_request_count") or 0
        ),
        "compact_owner_search_owner_sample_telemetry": dict(owner_sample_telemetry),
        "compact_owner_search_owner_learner_telemetry": dict(owner_learner_telemetry),
        **resident_handle_fields,
        "compact_owner_search_owner_replay_append_staged_entry_count": int(
            profile_payload.get("compact_owner_search_owner_replay_append_staged_entry_count") or 0
        ),
        "compact_owner_search_owner_replay_append_suppressed_entry_count": int(
            profile_payload.get("compact_owner_search_owner_replay_append_suppressed_entry_count")
            or 0
        ),
        "compact_owner_search_owner_replay_append_submitted_entry_count": int(
            profile_payload.get("compact_owner_search_owner_replay_append_submitted_entry_count")
            or 0
        ),
        "compact_owner_search_owner_replay_append_staged_transport_entry_count": _proof_int(
            "compact_owner_search_owner_replay_append_staged_transport_entry_count"
        ),
        "compact_owner_search_owner_replay_append_suppressed_transport_entry_count": _proof_int(
            "compact_owner_search_owner_replay_append_suppressed_transport_entry_count"
        ),
        "compact_owner_search_owner_replay_append_submitted_transport_entry_count": _proof_int(
            "compact_owner_search_owner_replay_append_submitted_transport_entry_count"
        ),
        "compact_owner_search_owner_replay_transport_entry_count": _proof_int(
            "compact_owner_search_owner_replay_transport_entry_count"
        ),
        "compact_owner_search_owner_replay_transport_kind": _proof_str(
            "compact_owner_search_owner_replay_transport_kind"
        ),
        "compact_owner_search_owner_replay_transition_batch_enabled": _proof_bool(
            "compact_owner_search_owner_replay_transition_batch_enabled"
        ),
        "compact_owner_search_owner_replay_transition_batch_count": _proof_int(
            "compact_owner_search_owner_replay_transition_batch_count"
        ),
        "compact_owner_search_owner_replay_transition_batch_transition_count": _proof_int(
            "compact_owner_search_owner_replay_transition_batch_transition_count"
        ),
        "compact_owner_search_owner_replay_transition_legacy_entry_count": _proof_int(
            "compact_owner_search_owner_replay_transition_legacy_entry_count"
        ),
        "compact_owner_search_transition_batch_transport_requested": _proof_bool(
            "compact_owner_search_transition_batch_transport_requested"
        ),
        "compact_owner_search_transition_batch_transport_enabled": _proof_bool(
            "compact_owner_search_transition_batch_transport_enabled"
        ),
        "compact_owner_search_transition_batch_transport_kind": _proof_str(
            "compact_owner_search_transition_batch_transport_kind"
        ),
        "compact_owner_search_transition_batch_schema_id": _proof_str(
            "compact_owner_search_transition_batch_schema_id"
        ),
        "compact_owner_search_transition_batch_count": _proof_int(
            "compact_owner_search_transition_batch_count"
        ),
        "compact_owner_search_transition_batch_entry_count": _proof_int(
            "compact_owner_search_transition_batch_entry_count"
        ),
        "compact_owner_search_transition_batch_transport_entry_count": _proof_int(
            "compact_owner_search_transition_batch_transport_entry_count"
        ),
        "compact_owner_search_transition_batch_max_entries_per_batch": _proof_int(
            "compact_owner_search_transition_batch_max_entries_per_batch"
        ),
        "compact_owner_search_transition_batch_fixed_capacity": _proof_int(
            "compact_owner_search_transition_batch_fixed_capacity"
        ),
        "compact_owner_search_transition_batch_padding_count": _proof_int(
            "compact_owner_search_transition_batch_padding_count"
        ),
        "compact_owner_search_transition_batch_overflow_count": _proof_int(
            "compact_owner_search_transition_batch_overflow_count"
        ),
        "compact_owner_search_transition_batch_fallback_count": _proof_int(
            "compact_owner_search_transition_batch_fallback_count"
        ),
        "compact_owner_search_transition_batch_fallback_reason": _proof_str(
            "compact_owner_search_transition_batch_fallback_reason"
        ),
        "compact_owner_search_transition_batch_pending_count": _proof_int(
            "compact_owner_search_transition_batch_pending_count"
        ),
        "compact_owner_search_transition_batch_transport_bytes": _proof_int(
            "compact_owner_search_transition_batch_transport_bytes"
        ),
        "compact_owner_search_transition_batch_digest": _proof_str(
            "compact_owner_search_transition_batch_digest"
        ),
        "compact_owner_search_transition_batch_digest_verified": _proof_bool(
            "compact_owner_search_transition_batch_digest_verified"
        ),
        "compact_owner_search_transition_batch_build_sec": _proof_float(
            "compact_owner_search_transition_batch_build_sec"
        ),
        "compact_owner_search_transition_batch_submit_sec": _proof_float(
            "compact_owner_search_transition_batch_submit_sec"
        ),
        "compact_owner_search_owner_local_transition_derivation_requested": _proof_bool(
            "compact_owner_search_owner_local_transition_derivation_requested"
        ),
        "compact_owner_search_owner_local_transition_derivation_used": _proof_bool(
            "compact_owner_search_owner_local_transition_derivation_used"
        ),
        "compact_owner_search_owner_local_transition_derivation_schema_id": _proof_str(
            "compact_owner_search_owner_local_transition_derivation_schema_id"
        ),
        "compact_owner_search_owner_local_transition_derivation_kind": _proof_str(
            "compact_owner_search_owner_local_transition_derivation_kind"
        ),
        "compact_owner_search_owner_local_transition_derivation_batch_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_batch_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_transition_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_transition_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_transport_entry_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_transport_entry_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_pending_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_pending_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_transport_bytes": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_transport_bytes"
        ),
        "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes"
        ),
        "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_digest": _proof_str(
            "compact_owner_search_owner_local_transition_derivation_digest"
        ),
        "compact_owner_search_owner_local_transition_derivation_digest_verified": _proof_bool(
            "compact_owner_search_owner_local_transition_derivation_digest_verified"
        ),
        "compact_owner_search_owner_local_transition_derivation_build_sec": _proof_float(
            "compact_owner_search_owner_local_transition_derivation_build_sec"
        ),
        "compact_owner_search_owner_local_transition_derivation_submit_sec": _proof_float(
            "compact_owner_search_owner_local_transition_derivation_submit_sec"
        ),
        "compact_owner_search_owner_local_transition_derivation_cache_hit_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_cache_hit_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_cache_miss_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_cache_miss_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_action_checksum_verified_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_action_checksum_verified_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_action_checksum_mismatch_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_action_checksum_mismatch_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_fallback_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_fallback_count"
        ),
        "compact_owner_search_owner_local_transition_derivation_fallback_reason": _proof_str(
            "compact_owner_search_owner_local_transition_derivation_fallback_reason"
        ),
        "compact_owner_search_owner_local_transition_derivation_dropped_pending_count": _proof_int(
            "compact_owner_search_owner_local_transition_derivation_dropped_pending_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_requested": _proof_bool(
            "compact_owner_search_owner_proxy_transition_closure_requested"
        ),
        "compact_owner_search_owner_proxy_transition_closure_requested_count": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_requested_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_used": _proof_bool(
            "compact_owner_search_owner_proxy_transition_closure_used"
        ),
        "compact_owner_search_owner_proxy_transition_closure_used_count": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_used_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_source": _proof_str(
            "compact_owner_search_owner_proxy_transition_closure_source"
        ),
        "compact_owner_search_owner_proxy_transition_closure_no_pending_count": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_no_pending_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_closed_count": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_closed_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_batch_count": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_batch_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_transition_count": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_transition_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_transport_entry_count": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_transport_entry_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_pending_count": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_pending_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_transport_bytes": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_transport_bytes"
        ),
        "compact_owner_search_owner_proxy_transition_closure_digest": _proof_str(
            "compact_owner_search_owner_proxy_transition_closure_digest"
        ),
        "compact_owner_search_owner_proxy_transition_closure_digest_verified": _proof_bool(
            "compact_owner_search_owner_proxy_transition_closure_digest_verified"
        ),
        "compact_owner_search_owner_proxy_transition_closure_build_sec": _proof_float(
            "compact_owner_search_owner_proxy_transition_closure_build_sec"
        ),
        "compact_owner_search_owner_proxy_transition_closure_submit_sec": _proof_float(
            "compact_owner_search_owner_proxy_transition_closure_submit_sec"
        ),
        "compact_owner_search_owner_proxy_transition_closure_fallback_count": _proof_int(
            "compact_owner_search_owner_proxy_transition_closure_fallback_count"
        ),
        "compact_owner_search_owner_proxy_transition_closure_fallback_reason": _proof_str(
            "compact_owner_search_owner_proxy_transition_closure_fallback_reason"
        ),
        "compact_owner_search_owner_proxy_applied_action_verification_count": _proof_int(
            "compact_owner_search_owner_proxy_applied_action_verification_count"
        ),
        "compact_owner_search_owner_proxy_applied_action_mismatch_count": _proof_int(
            "compact_owner_search_owner_proxy_applied_action_mismatch_count"
        ),
        "compact_owner_search_owner_proxy_applied_action_count": _proof_int(
            "compact_owner_search_owner_proxy_applied_action_count"
        ),
        "compact_owner_search_owner_proxy_applied_action_checksum": _proof_int(
            "compact_owner_search_owner_proxy_applied_action_checksum"
        ),
        "compact_owner_search_owner_proxy_action_frame_pending": _proof_bool(
            "compact_owner_search_owner_proxy_action_frame_pending"
        ),
        "compact_owner_search_owner_proxy_action_frame_store_count": _proof_int(
            "compact_owner_search_owner_proxy_action_frame_store_count"
        ),
        "compact_owner_search_parent_previous_transition_closure_count": _proof_int(
            "compact_owner_search_parent_previous_transition_closure_count"
        ),
        "compact_owner_search_parent_previous_transition_closure_avoided_count": _proof_int(
            "compact_owner_search_parent_previous_transition_closure_avoided_count"
        ),
        "compact_owner_search_parent_applied_action_validation_count": _proof_int(
            "compact_owner_search_parent_applied_action_validation_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_requested": _proof_bool(
            "compact_owner_search_direct_transition_batch_replay_requested"
        ),
        "compact_owner_search_direct_transition_batch_replay_used": _proof_bool(
            "compact_owner_search_direct_transition_batch_replay_used"
        ),
        "compact_owner_search_direct_transition_batch_replay_batch_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_batch_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_transition_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_transition_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_transport_entry_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_transport_entry_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_legacy_expanded_entry_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_legacy_expanded_entry_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_index_entry_object_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_index_entry_object_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_ring_entry_object_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_ring_entry_object_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_append_used": _proof_bool(
            "compact_owner_search_direct_transition_batch_replay_columnar_append_used"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_record_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_columnar_record_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_entry_view_object_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_columnar_entry_view_object_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_step_view_object_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_columnar_step_view_object_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_fallback_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_fallback_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_fallback_reason": _proof_str(
            "compact_owner_search_direct_transition_batch_replay_fallback_reason"
        ),
        "compact_owner_search_direct_transition_batch_replay_last_append_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_last_append_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_append_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_append_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_accounted_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_accounted_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_array_extract_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_array_extract_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_transition_validate_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_transition_validate_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_device_payload_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_device_payload_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flushed_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flushed_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls"
        ),
        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_count_max": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_count_max"
        ),
        "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_final_count": _proof_int(
            "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_final_count"
        ),
        "compact_owner_search_direct_transition_batch_replay_replay_payload_d2h_bytes": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_replay_payload_d2h_bytes"
        ),
        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_flush_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_flush_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flush_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flush_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_action_model_state_digest": _proof_str(
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_action_model_state_digest"
        ),
        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_model_state_digest": _proof_str(
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_model_state_digest"
        ),
        "compact_owner_search_direct_transition_batch_replay_index_rows_build_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_index_rows_build_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_step_object_build_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_step_object_build_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_ring_append_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_ring_append_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_prepare_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_prepare_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_register_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_register_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_append_store_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_append_store_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_retain_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_retain_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_evict_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_evict_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_evict_release_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_evict_release_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_candidate_indices_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_candidate_indices_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_cache_refresh_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_cache_refresh_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_cache_rebuild_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_cache_rebuild_sec"
        ),
        "compact_owner_search_direct_transition_batch_replay_columnar_total_sec": _proof_float(
            "compact_owner_search_direct_transition_batch_replay_columnar_total_sec"
        ),
        "compact_owner_search_owner_replay_append_request_count": int(
            profile_payload.get("compact_owner_search_owner_replay_append_request_count") or 0
        ),
        "compact_owner_search_owner_replay_append_count": int(
            profile_payload.get("compact_owner_search_owner_replay_append_count") or 0
        ),
        "compact_owner_search_owner_train_request_count": int(
            profile_payload.get("compact_owner_search_owner_train_request_count") or 0
        ),
        "compact_owner_search_owner_model_refresh_request_count": int(
            profile_payload.get("compact_owner_search_owner_model_refresh_request_count") or 0
        ),
        "compact_owner_search_owner_model_refresh_skipped_count": int(
            profile_payload.get("compact_owner_search_owner_model_refresh_skipped_count") or 0
        ),
        "compact_owner_search_owner_submitted_learner_update_count": int(
            profile_payload.get("compact_owner_search_owner_submitted_learner_update_count") or 0
        ),
        "compact_owner_search_owner_learner_update_count": int(
            profile_payload.get("compact_owner_search_owner_learner_update_count") or 0
        ),
        "compact_owner_search_owner_pending_replay_append_entry_count": int(
            profile_payload.get("compact_owner_search_owner_pending_replay_append_entry_count") or 0
        ),
        "compact_owner_search_owner_maintenance_drain_request_count": int(
            profile_payload.get("compact_owner_search_owner_maintenance_drain_request_count") or 0
        ),
        "compact_owner_search_owner_maintenance_staged_work_item_count": int(
            profile_payload.get("compact_owner_search_owner_maintenance_staged_work_item_count")
            or 0
        ),
        "compact_owner_search_owner_maintenance_drained_count": int(
            profile_payload.get("compact_owner_search_owner_maintenance_drained_count") or 0
        ),
        "compact_owner_search_owner_maintenance_drained_work_item_count": int(
            profile_payload.get("compact_owner_search_owner_maintenance_drained_work_item_count")
            or 0
        ),
        "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": int(
            profile_payload.get(
                "compact_owner_search_owner_maintenance_drained_replay_append_entry_count"
            )
            or 0
        ),
        "compact_owner_search_owner_maintenance_drained_replay_append_transport_entry_count": _proof_int(
            "compact_owner_search_owner_maintenance_drained_replay_append_transport_entry_count"
        ),
        "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_count": _proof_int(
            "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_count"
        ),
        "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_entry_count": _proof_int(
            "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_entry_count"
        ),
        "compact_owner_search_owner_maintenance_drained_replay_append_count": int(
            profile_payload.get(
                "compact_owner_search_owner_maintenance_drained_replay_append_count"
            )
            or 0
        ),
        "compact_owner_search_owner_maintenance_pending_work_count": int(
            profile_payload.get("compact_owner_search_owner_maintenance_pending_work_count") or 0
        ),
        "compact_owner_search_owner_maintenance_inflight": bool(
            profile_payload.get("compact_owner_search_owner_maintenance_inflight", False)
        ),
        "compact_owner_search_owner_maintenance_final_drain_sec": float(
            profile_payload.get("compact_owner_search_owner_maintenance_final_drain_sec") or 0.0
        ),
        "compact_owner_search_owner_maintenance_final_drain_in_measured_sec": bool(
            profile_payload.get(
                "compact_owner_search_owner_maintenance_final_drain_in_measured_sec",
                False,
            )
        ),
        "compact_owner_search_owner_maintenance_coalescing_kind": str(
            profile_payload.get("compact_owner_search_owner_maintenance_coalescing_kind") or ""
        ),
        "compact_owner_search_owner_maintenance_coalesced_skip_count": int(
            profile_payload.get("compact_owner_search_owner_maintenance_coalesced_skip_count") or 0
        ),
        "compact_owner_search_owner_maintenance_eager_append_drain_count": int(
            profile_payload.get("compact_owner_search_owner_maintenance_eager_append_drain_count")
            or 0
        ),
        "compact_owner_search_owner_async_learner_worker_enabled": bool(
            profile_payload.get(
                "compact_owner_search_owner_async_learner_worker_enabled",
                False,
            )
        ),
        "compact_owner_search_owner_async_learner_worker_kind": str(
            profile_payload.get("compact_owner_search_owner_async_learner_worker_kind") or "none"
        ),
        "compact_owner_search_owner_async_learner_worker_resource_scope": str(
            profile_payload.get("compact_owner_search_owner_async_learner_worker_resource_scope")
            or ""
        ),
        "compact_owner_search_owner_async_learner_worker_resource_id": str(
            profile_payload.get("compact_owner_search_owner_async_learner_worker_resource_id") or ""
        ),
        "compact_owner_search_owner_async_learner_actor_resource_id": str(
            profile_payload.get("compact_owner_search_owner_async_learner_actor_resource_id") or ""
        ),
        "compact_owner_search_owner_async_learner_worker_parent_pid": int(
            profile_payload.get("compact_owner_search_owner_async_learner_worker_parent_pid") or 0
        ),
        "compact_owner_search_owner_async_learner_resource_distinct_from_owner": bool(
            profile_payload.get(
                "compact_owner_search_owner_async_learner_resource_distinct_from_owner",
                False,
            )
        ),
        ("compact_owner_search_owner_async_learner_hardware_resource_distinct_from_owner"): bool(
            profile_payload.get(
                "compact_owner_search_owner_async_learner_hardware_resource_distinct_from_owner",
                False,
            )
        ),
        "compact_owner_search_owner_async_learner_max_pending": int(
            profile_payload.get("compact_owner_search_owner_async_learner_max_pending") or 0
        ),
        "compact_owner_search_owner_async_learner_submit_count": int(
            profile_payload.get("compact_owner_search_owner_async_learner_submit_count") or 0
        ),
        "compact_owner_search_owner_async_learner_completed_count": int(
            profile_payload.get("compact_owner_search_owner_async_learner_completed_count") or 0
        ),
        "compact_owner_search_owner_async_learner_pending_count": int(
            profile_payload.get("compact_owner_search_owner_async_learner_pending_count") or 0
        ),
        "compact_owner_search_owner_async_learner_max_pending_observed": int(
            profile_payload.get("compact_owner_search_owner_async_learner_max_pending_observed")
            or 0
        ),
        "compact_owner_search_owner_async_learner_wait_count": int(
            profile_payload.get("compact_owner_search_owner_async_learner_wait_count") or 0
        ),
        "compact_owner_search_owner_async_learner_wait_sec": float(
            profile_payload.get("compact_owner_search_owner_async_learner_wait_sec") or 0.0
        ),
        "compact_owner_search_owner_action_while_async_learner_pending_count": int(
            profile_payload.get(
                "compact_owner_search_owner_action_while_async_learner_pending_count"
            )
            or 0
        ),
        "compact_owner_search_owner_async_learner_failed": bool(
            profile_payload.get(
                "compact_owner_search_owner_async_learner_failed",
                False,
            )
        ),
        "compact_owner_search_owner_async_learner_request_host_only": bool(
            profile_payload.get(
                "compact_owner_search_owner_async_learner_request_host_only",
                False,
            )
        ),
        "compact_owner_search_owner_async_learner_request_cuda_tensor_count": int(
            profile_payload.get(
                "compact_owner_search_owner_async_learner_request_cuda_tensor_count"
            )
            or 0
        ),
        "compact_owner_search_owner_async_learner_result_host_only": bool(
            profile_payload.get(
                "compact_owner_search_owner_async_learner_result_host_only",
                False,
            )
        ),
        "compact_owner_search_owner_async_learner_result_cuda_tensor_count": int(
            profile_payload.get("compact_owner_search_owner_async_learner_result_cuda_tensor_count")
            or 0
        ),
        "compact_owner_search_owner_async_learner_request_bytes": int(
            profile_payload.get("compact_owner_search_owner_async_learner_request_bytes") or 0
        ),
        "compact_owner_search_owner_async_learner_result_bytes": int(
            profile_payload.get("compact_owner_search_owner_async_learner_result_bytes") or 0
        ),
        "compact_owner_search_owner_async_learner_worker_pid": int(
            profile_payload.get("compact_owner_search_owner_async_learner_worker_pid") or 0
        ),
        "compact_owner_search_owner_async_learner_worker_job_wall_sec": float(
            profile_payload.get("compact_owner_search_owner_async_learner_worker_job_wall_sec")
            or 0.0
        ),
        "compact_owner_search_owner_async_learner_payload_prepare_sec": float(
            profile_payload.get("compact_owner_search_owner_async_learner_payload_prepare_sec")
            or 0.0
        ),
        ("compact_owner_search_owner_async_learner_worker_pid_distinct_from_owner"): bool(
            profile_payload.get(
                "compact_owner_search_owner_async_learner_worker_pid_distinct_from_owner",
                False,
            )
        ),
        "compact_owner_search_owner_async_learner_worker_owns_model_state": bool(
            profile_payload.get(
                "compact_owner_search_owner_async_learner_worker_owns_model_state",
                False,
            )
        ),
        "compact_owner_search_owner_policy_lag_current": int(
            profile_payload.get("compact_owner_search_owner_policy_lag_current") or 0
        ),
        "compact_owner_search_owner_policy_lag_max": int(
            profile_payload.get("compact_owner_search_owner_policy_lag_max") or 0
        ),
        "compact_owner_search_owner_maintenance_actor_steps_while_pending": int(
            profile_payload.get("compact_owner_search_owner_maintenance_actor_steps_while_pending")
            or 0
        ),
        "compact_owner_search_owner_maintenance_actor_steps_while_policy_lagged": int(
            profile_payload.get(
                "compact_owner_search_owner_maintenance_actor_steps_while_policy_lagged"
            )
            or 0
        ),
        "compact_owner_search_owner_action_while_maintenance_pending_count": int(
            profile_payload.get("compact_owner_search_owner_action_while_maintenance_pending_count")
            or 0
        ),
        "compact_owner_search_owner_action_while_policy_lagged_count": int(
            profile_payload.get("compact_owner_search_owner_action_while_policy_lagged_count") or 0
        ),
        "compact_owner_search_owner_action_served_before_maintenance_count": int(
            profile_payload.get("compact_owner_search_owner_action_served_before_maintenance_count")
            or 0
        ),
        "compact_owner_search_owner_fifo_blocked_action_count": int(
            profile_payload.get("compact_owner_search_owner_fifo_blocked_action_count") or 0
        ),
        "compact_owner_search_owner_maintenance_failed": bool(
            profile_payload.get("compact_owner_search_owner_maintenance_failed", False)
        ),
        "compact_owner_search_parent_publish_sec": float(
            profile_payload.get("compact_owner_search_parent_publish_sec") or 0.0
        ),
        "compact_owner_search_parent_submit_sec": float(
            profile_payload.get("compact_owner_search_parent_submit_sec") or 0.0
        ),
        "compact_owner_search_parent_wait_sec": float(
            profile_payload.get("compact_owner_search_parent_wait_sec") or 0.0
        ),
        "compact_owner_search_parent_wall_sec": float(
            profile_payload.get("compact_owner_search_parent_wall_sec") or 0.0
        ),
        "compact_owner_search_worker_wall_sec": float(
            profile_payload.get("compact_owner_search_worker_wall_sec") or 0.0
        ),
        "compact_owner_search_worker_root_resolve_sec": float(
            profile_payload.get("compact_owner_search_worker_root_resolve_sec") or 0.0
        ),
        "compact_owner_search_worker_search_sec": float(
            profile_payload.get("compact_owner_search_worker_search_sec") or 0.0
        ),
        "compact_owner_search_worker_replay_append_sec": float(
            profile_payload.get("compact_owner_search_worker_replay_append_sec") or 0.0
        ),
        "compact_owner_search_worker_learner_train_sec": float(
            profile_payload.get("compact_owner_search_worker_learner_train_sec") or 0.0
        ),
        "compact_owner_search_owner_train_wall_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_wall_sec",
                owner_learner_telemetry.get("compact_owner_search_owner_train_wall_sec"),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_sample_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_sample_sec",
                owner_learner_telemetry.get("compact_owner_search_owner_train_sample_sec"),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_payload_host_clone_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_payload_host_clone_sec",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_payload_host_clone_sec"
                ),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_payload_device_move_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_payload_device_move_sec",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_payload_device_move_sec"
                ),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_learner_update_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_learner_update_sec",
                owner_learner_telemetry.get("compact_owner_search_owner_train_learner_update_sec"),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_model_state_digest_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_model_state_digest_sec",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_model_state_digest_sec"
                ),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_model_state_digest_deferred_to_refresh": bool(
            profile_payload.get(
                "compact_owner_search_owner_train_model_state_digest_deferred_to_refresh",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_model_state_digest_deferred_to_refresh",
                    False,
                ),
            )
        ),
        "compact_owner_search_owner_train_model_state_dict_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_model_state_dict_sec",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_model_state_dict_sec"
                ),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_owner_ref_build_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_owner_ref_build_sec",
                owner_learner_telemetry.get("compact_owner_search_owner_train_owner_ref_build_sec"),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_model_state_snapshot_returned": bool(
            profile_payload.get(
                "compact_owner_search_owner_train_model_state_snapshot_returned",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_model_state_snapshot_returned",
                    False,
                ),
            )
        ),
        "compact_owner_search_owner_train_model_state_snapshot_bytes": int(
            profile_payload.get(
                "compact_owner_search_owner_train_model_state_snapshot_bytes",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_model_state_snapshot_bytes"
                ),
            )
            or 0
        ),
        "compact_owner_search_owner_train_model_state_snapshot_write_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_model_state_snapshot_write_sec",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_model_state_snapshot_write_sec"
                ),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_accounted_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_accounted_sec",
                owner_learner_telemetry.get("compact_owner_search_owner_train_accounted_sec"),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_residual_sec": float(
            profile_payload.get(
                "compact_owner_search_owner_train_residual_sec",
                owner_learner_telemetry.get("compact_owner_search_owner_train_residual_sec"),
            )
            or 0.0
        ),
        "compact_owner_search_owner_train_timing_aggregate_count": int(
            profile_payload.get(
                "compact_owner_search_owner_train_timing_aggregate_count",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_timing_aggregate_count"
                ),
            )
            or 0
        ),
        "compact_owner_search_worker_search_refresh_sec": float(
            profile_payload.get("compact_owner_search_worker_search_refresh_sec") or 0.0
        ),
        "compact_owner_search_resident_root_bridge_ready": bool(
            search_metadata.get(
                "compact_owner_search_resident_root_bridge_ready",
                profile_payload.get("compact_owner_search_resident_root_bridge_ready", False),
            )
        ),
        "compact_owner_search_resident_root_bridge_kind": str(
            search_metadata.get("compact_owner_search_resident_root_bridge_kind")
            or profile_payload.get("compact_owner_search_resident_root_bridge_kind")
            or ""
        ),
        "compact_owner_search_resident_root_bridge_device": str(
            search_metadata.get("compact_owner_search_resident_root_bridge_device")
            or profile_payload.get("compact_owner_search_resident_root_bridge_device")
            or ""
        ),
        "compact_owner_search_resident_root_bridge_h2d_bytes": float(
            search_metadata.get(
                "compact_owner_search_resident_root_bridge_h2d_bytes",
                profile_payload.get("compact_owner_search_resident_root_bridge_h2d_bytes", 0.0),
            )
            or 0.0
        ),
        "compact_owner_search_resident_root_bridge_host_observation_copied": bool(
            search_metadata.get(
                "compact_owner_search_resident_root_bridge_host_observation_copied",
                profile_payload.get(
                    "compact_owner_search_resident_root_bridge_host_observation_copied",
                    True,
                ),
            )
        ),
        "compact_owner_search_resident_root_bridge_generation_id": int(
            search_metadata.get(
                "compact_owner_search_resident_root_bridge_generation_id",
                profile_payload.get("compact_owner_search_resident_root_bridge_generation_id", 0),
            )
            or 0
        ),
        "compact_owner_search_resident_root_bridge_final_storage": str(
            search_metadata.get("compact_owner_search_resident_root_bridge_final_storage")
            or profile_payload.get("compact_owner_search_resident_root_bridge_final_storage")
            or ""
        ),
        "compact_owner_search_resident_root_bridge_final_sparse_row_count": int(
            search_metadata.get(
                "compact_owner_search_resident_root_bridge_final_sparse_row_count",
                profile_payload.get(
                    "compact_owner_search_resident_root_bridge_final_sparse_row_count",
                    0,
                ),
            )
            or 0
        ),
        "compact_owner_search_resident_root_bridge_final_sparse_bytes": int(
            search_metadata.get(
                "compact_owner_search_resident_root_bridge_final_sparse_bytes",
                profile_payload.get(
                    "compact_owner_search_resident_root_bridge_final_sparse_bytes",
                    0,
                ),
            )
            or 0
        ),
        "compact_owner_search_resident_root_bridge_final_dense_clone_avoided_bytes": int(
            search_metadata.get(
                "compact_owner_search_resident_root_bridge_final_dense_clone_avoided_bytes",
                profile_payload.get(
                    "compact_owner_search_resident_root_bridge_final_dense_clone_avoided_bytes",
                    0,
                ),
            )
            or 0
        ),
        "compact_direct_root_store": bool(profile_payload.get("compact_direct_root_store", False)),
        "compact_direct_root_store_publish_count": int(
            profile_payload.get("compact_direct_root_store_publish_count") or 0
        ),
        "compact_direct_root_store_resolve_count": int(
            profile_payload.get("compact_direct_root_store_resolve_count") or 0
        ),
        "compact_direct_root_store_last_root_slot_count": int(
            profile_payload.get("compact_direct_root_store_last_root_slot_count") or 0
        ),
        "compact_owner_search_direct_root_handoff": bool(
            profile_payload.get("compact_owner_search_direct_root_handoff", False)
        ),
        "compact_owner_search_direct_root_rebuild_avoided": bool(
            profile_payload.get("compact_owner_search_direct_root_rebuild_avoided", False)
        ),
        "compact_owner_search_direct_root_resolved": bool(
            profile_payload.get("compact_owner_search_direct_root_resolved", False)
        ),
        "compact_owner_search_direct_root_observation_bytes_sent": int(
            profile_payload.get("compact_owner_search_direct_root_observation_bytes_sent") or 0
        ),
        "compact_owner_search_direct_root_build_request_handoff": _proof_bool(
            "compact_owner_search_direct_root_build_request_handoff"
        ),
        "compact_owner_step_frame_root_build_request_used": _proof_bool(
            "compact_owner_step_frame_root_build_request_used"
        ),
        "compact_owner_step_frame_root_build_request_from_batch_helper_used": (
            _proof_bool(
                "compact_owner_step_frame_root_build_request_from_batch_helper_used"
            )
        ),
        "compact_owner_step_frame_root_request_sidecar_array_bytes": _proof_int(
            "compact_owner_step_frame_root_request_sidecar_array_bytes"
        ),
        "compact_owner_step_frame_root_request_sidecar_field_count": _proof_int(
            "compact_owner_step_frame_root_request_sidecar_field_count"
        ),
        "compact_owner_root_search_transaction_boundary_supported": _proof_bool(
            "compact_owner_root_search_transaction_boundary_supported"
        ),
        "compact_owner_root_search_transaction_requested": _proof_bool(
            "compact_owner_root_search_transaction_requested"
        ),
        "compact_owner_root_search_transaction_used": _proof_bool(
            "compact_owner_root_search_transaction_used"
        ),
        "compact_owner_root_search_transaction_schema_id": _proof_str(
            "compact_owner_root_search_transaction_schema_id"
        ),
        "compact_owner_root_search_transaction_id": _proof_int(
            "compact_owner_root_search_transaction_id"
        ),
        "compact_owner_root_search_transaction_begin_count": _proof_int(
            "compact_owner_root_search_transaction_begin_count"
        ),
        "compact_owner_root_search_transaction_submit_count": _proof_int(
            "compact_owner_root_search_transaction_submit_count"
        ),
        "compact_owner_root_search_transaction_resolve_count": _proof_int(
            "compact_owner_root_search_transaction_resolve_count"
        ),
        "compact_owner_root_search_transaction_pending_count": _proof_int(
            "compact_owner_root_search_transaction_pending_count"
        ),
        "compact_owner_root_search_transaction_max_pending_count": _proof_int(
            "compact_owner_root_search_transaction_max_pending_count"
        ),
        "compact_owner_root_search_transaction_parent_root_request_build_count": (
            _proof_int(
                "compact_owner_root_search_transaction_parent_root_request_build_count"
            )
        ),
        "compact_owner_root_search_transaction_parent_root_request_stored": (
            _proof_bool(
                "compact_owner_root_search_transaction_parent_root_request_stored"
            )
        ),
        "compact_owner_root_search_transaction_parent_compact_batch_stored": (
            _proof_bool(
                "compact_owner_root_search_transaction_parent_compact_batch_stored"
            )
        ),
        "compact_owner_root_search_transaction_parent_rebuild_count": _proof_int(
            "compact_owner_root_search_transaction_parent_rebuild_count"
        ),
        "compact_owner_root_search_transaction_parent_root_action_context_stored": (
            _proof_bool(
                "compact_owner_root_search_transaction_parent_root_action_context_stored"
            )
        ),
        "compact_owner_root_search_transaction_parent_root_action_context_store_count": (
            _proof_int(
                "compact_owner_root_search_transaction_parent_root_action_context_store_count"
            )
        ),
        "compact_owner_root_search_transaction_parent_root_action_context_array_bytes": (
            _proof_int(
                "compact_owner_root_search_transaction_parent_root_action_context_array_bytes"
            )
        ),
        "compact_owner_root_search_transaction_parent_root_action_context_field_count": (
            _proof_int(
                "compact_owner_root_search_transaction_parent_root_action_context_field_count"
            )
        ),
        "compact_owner_root_search_transaction_owner_root_request_build_count": (
            _proof_int(
                "compact_owner_root_search_transaction_owner_root_request_build_count"
            )
        ),
        "compact_owner_root_search_transaction_owner_root_request_build_sec": (
            _proof_float(
                "compact_owner_root_search_transaction_owner_root_request_build_sec"
            )
        ),
        "compact_owner_root_search_transaction_owner_root_store_publish_count": (
            _proof_int(
                "compact_owner_root_search_transaction_owner_root_store_publish_count"
            )
        ),
        "compact_owner_root_search_transaction_frame_generation_verified": (
            _proof_bool(
                "compact_owner_root_search_transaction_frame_generation_verified"
            )
        ),
        "compact_owner_root_search_transaction_frame_digest_verified": _proof_bool(
            "compact_owner_root_search_transaction_frame_digest_verified"
        ),
        "compact_owner_root_search_transaction_action_identity_verified": _proof_bool(
            "compact_owner_root_search_transaction_action_identity_verified"
        ),
        "compact_owner_root_search_transaction_proxy_transition_closure_used": (
            _proof_bool(
                "compact_owner_root_search_transaction_proxy_transition_closure_used"
            )
        ),
        "compact_owner_root_search_transaction_applied_action_mismatch_count": (
            _proof_int(
                "compact_owner_root_search_transaction_applied_action_mismatch_count"
            )
        ),
        "compact_owner_search_direct_root_build_request_schema_id": _proof_str(
            "compact_owner_search_direct_root_build_request_schema_id"
        ),
        "compact_owner_search_direct_root_build_request_kind": _proof_str(
            "compact_owner_search_direct_root_build_request_kind"
        ),
        "compact_owner_search_direct_root_build_request_publish_count": _proof_int(
            "compact_owner_search_direct_root_build_request_publish_count"
        ),
        "compact_owner_search_direct_root_build_request_resolve_count": _proof_int(
            "compact_owner_search_direct_root_build_request_resolve_count"
        ),
        "compact_owner_search_direct_root_build_request_root_count": _proof_int(
            "compact_owner_search_direct_root_build_request_root_count"
        ),
        "compact_owner_search_direct_root_build_request_active_root_count": _proof_int(
            "compact_owner_search_direct_root_build_request_active_root_count"
        ),
        "compact_owner_search_direct_root_build_request_observation_included": _proof_bool(
            "compact_owner_search_direct_root_build_request_observation_included"
        ),
        "compact_owner_search_direct_root_build_request_observation_bytes_sent": _proof_int(
            "compact_owner_search_direct_root_build_request_observation_bytes_sent"
        ),
        "compact_owner_search_direct_root_build_request_resident_handle_present": _proof_bool(
            "compact_owner_search_direct_root_build_request_resident_handle_present"
        ),
        "compact_owner_search_direct_root_parent_build_avoided": _proof_bool(
            "compact_owner_search_direct_root_parent_build_avoided"
        ),
        "compact_owner_search_direct_root_parent_build_call_count": _proof_int(
            "compact_owner_search_direct_root_parent_build_call_count"
        ),
        "compact_owner_search_direct_root_parent_build_sec": _proof_float(
            "compact_owner_search_direct_root_parent_build_sec"
        ),
        "compact_owner_search_direct_root_build_request_sec": _proof_float(
            "compact_owner_search_direct_root_build_request_sec"
        ),
        "compact_owner_search_direct_root_owner_build_used": _proof_bool(
            "compact_owner_search_direct_root_owner_build_used"
        ),
        "compact_owner_search_direct_root_owner_build_count": _proof_int(
            "compact_owner_search_direct_root_owner_build_count"
        ),
        "compact_owner_search_direct_root_owner_build_sec": _proof_float(
            "compact_owner_search_direct_root_owner_build_sec"
        ),
        "compact_owner_search_parent_compact_root_batch_objects_sent": _proof_int(
            "compact_owner_search_parent_compact_root_batch_objects_sent"
        ),
        "compact_owner_search_root_build_request_host_observation_bytes_sent": _proof_int(
            "compact_owner_search_root_build_request_host_observation_bytes_sent"
        ),
        "compact_owner_search_resident_root_view_required": bool(
            profile_payload.get("compact_owner_search_resident_root_view_required", False)
            or search_metadata.get("compact_owner_search_resident_root_view_required", False)
        ),
        "compact_owner_search_resident_root_view_proved": bool(
            profile_payload.get("compact_owner_search_resident_root_view_proved", False)
            or search_metadata.get("compact_owner_search_resident_root_view_proved", False)
        ),
        "compact_owner_search_resident_root_view_kind": str(
            profile_payload.get("compact_owner_search_resident_root_view_kind")
            or search_metadata.get("compact_owner_search_resident_root_view_kind")
            or ""
        ),
        "compact_owner_search_resident_root_view_generation_id": int(
            profile_payload.get("compact_owner_search_resident_root_view_generation_id")
            or search_metadata.get("compact_owner_search_resident_root_view_generation_id")
            or 0
        ),
        "compact_owner_search_resident_root_view_fresh_for_step_index": int(
            profile_payload.get("compact_owner_search_resident_root_view_fresh_for_step_index")
            or search_metadata.get("compact_owner_search_resident_root_view_fresh_for_step_index")
            or 0
        ),
        "compact_owner_search_resident_root_view_device": str(
            profile_payload.get("compact_owner_search_resident_root_view_device")
            or search_metadata.get("compact_owner_search_resident_root_view_device")
            or ""
        ),
        "compact_owner_search_resident_root_view_source_backend": str(
            profile_payload.get("compact_owner_search_resident_root_view_source_backend")
            or search_metadata.get("compact_owner_search_resident_root_view_source_backend")
            or ""
        ),
        "compact_owner_search_resident_root_view_root_shape": list(
            profile_payload.get("compact_owner_search_resident_root_view_root_shape")
            or search_metadata.get("compact_owner_search_resident_root_view_root_shape")
            or []
        ),
        "compact_owner_search_resident_root_view_stack_shape": list(
            profile_payload.get("compact_owner_search_resident_root_view_stack_shape")
            or search_metadata.get("compact_owner_search_resident_root_view_stack_shape")
            or []
        ),
        "compact_owner_search_resident_root_view_h2d_bytes": float(
            profile_payload.get("compact_owner_search_resident_root_view_h2d_bytes")
            or search_metadata.get("compact_owner_search_resident_root_view_h2d_bytes")
            or 0.0
        ),
        "compact_owner_search_resident_root_view_d2h_bytes": float(
            profile_payload.get("compact_owner_search_resident_root_view_d2h_bytes")
            or search_metadata.get("compact_owner_search_resident_root_view_d2h_bytes")
            or 0.0
        ),
        "compact_owner_search_resident_root_view_host_fallback_allowed": bool(
            profile_payload.get(
                "compact_owner_search_resident_root_view_host_fallback_allowed",
                False,
            )
            or search_metadata.get(
                "compact_owner_search_resident_root_view_host_fallback_allowed",
                False,
            )
        ),
        "compact_owner_search_resident_root_view_row_major_order": bool(
            profile_payload.get(
                "compact_owner_search_resident_root_view_row_major_order",
                False,
            )
            or search_metadata.get(
                "compact_owner_search_resident_root_view_row_major_order",
                False,
            )
        ),
        "compact_rollout_slab_committed_index_row_count": int(
            profile_payload.get("compact_rollout_slab_committed_index_row_count") or 0
        ),
        "compact_rollout_slab_stored_index_row_count": int(
            profile_payload.get("compact_rollout_slab_stored_index_row_count") or 0
        ),
    }


def _latest_compact_rollout_slab_search_metadata(
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    telemetry = profile_payload.get("compact_rollout_slab_last_telemetry")
    if not isinstance(telemetry, Mapping):
        return {}
    search_metadata = telemetry.get("compact_rollout_slab_search_metadata")
    if isinstance(search_metadata, Mapping):
        return dict(search_metadata)
    return {}


def _compact_loop_counter_fields(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    contract = profile_payload.get("contract")
    if not isinstance(contract, Mapping):
        contract = {}
    return {
        "compact_owned_loop_record_step_calls": int(
            profile_payload.get("compact_owned_loop_record_step_calls") or 0
        ),
        "compact_owned_loop_appended_replay_entry_count": int(
            profile_payload.get("compact_owned_loop_appended_replay_entry_count") or 0
        ),
        "compact_rollout_slab_learner_gate_calls": int(
            profile_payload.get("compact_rollout_slab_learner_gate_calls") or 0
        ),
        "compact_rollout_slab_learner_gate_updates": int(
            profile_payload.get("compact_rollout_slab_learner_gate_updates") or 0
        ),
        "compact_rollout_slab_learner_gate_sample_rows": int(
            profile_payload.get("compact_rollout_slab_learner_gate_sample_row_count") or 0
        ),
        "compact_rollout_slab_sample_gate_calls": int(
            profile_payload.get("compact_rollout_slab_sample_gate_calls") or 0
        ),
        "compact_rollout_slab_sample_gate_sample_rows": int(
            profile_payload.get("compact_rollout_slab_sample_gate_sample_row_count") or 0
        ),
        "compact_rollout_slab_sample_gate_opportunities": int(
            profile_payload.get("compact_rollout_slab_sample_gate_opportunities") or 0
        ),
        "compact_rollout_slab_sample_gate_skipped_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_skipped_count") or 0
        ),
        "resident_replay_snapshot_mode": str(
            profile_payload.get("resident_replay_snapshot_mode")
            or contract.get("resident_replay_snapshot_mode")
            or ""
        ),
        "compact_owned_loop_replay_store_retained_resident_snapshot_count": int(
            profile_payload.get("compact_owned_loop_replay_store_retained_resident_snapshot_count")
            or 0
        ),
        "compact_owned_loop_replay_store_retained_resident_snapshot_bytes": int(
            profile_payload.get("compact_owned_loop_replay_store_retained_resident_snapshot_bytes")
            or 0
        ),
        "compact_rollout_slab_sample_gate_replay_ring_entry_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_replay_ring_entry_count") or 0
        ),
        "compact_rollout_slab_sample_gate_replay_ring_index_row_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_replay_ring_index_row_count") or 0
        ),
        "compact_rollout_slab_sample_gate_replay_ring_pair_capacity": int(
            profile_payload.get("compact_rollout_slab_sample_gate_replay_ring_pair_capacity") or 0
        ),
        "compact_rollout_slab_sample_gate_replay_ring_evicted_pair_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_replay_ring_evicted_pair_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_replay_ring_evicted_index_row_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_replay_ring_evicted_index_row_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_bytes": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_replay_ring_retained_resident_snapshot_bytes"
            )
            or 0
        ),
    }


def _nested_mapping_value(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _arg_int(args: argparse.Namespace, name: str, default: int) -> int:
    value = getattr(args, name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be integer-like") from exc


def _repeatability_work_shape_fields(
    *,
    args: argparse.Namespace,
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    sample_gate = _nested_mapping_value(
        profile_payload,
        "compact_owned_loop_sample_gate_last_telemetry",
    )
    if not sample_gate:
        sample_gate = _nested_mapping_value(
            profile_payload,
            "compact_rollout_slab_sample_gate_last_telemetry",
        )
    learner_gate = _nested_mapping_value(
        profile_payload,
        "compact_owned_loop_learner_gate_last_telemetry",
    )
    if not learner_gate:
        learner_gate = _nested_mapping_value(
            profile_payload,
            "compact_rollout_slab_learner_gate_last_telemetry",
        )
    learner_muzero = _nested_mapping_value(
        learner_gate,
        "compact_rollout_slab_learner_gate_compact_muzero_telemetry",
    )
    loop_sample_metadata = _nested_mapping_value(
        profile_payload,
        "compact_owned_loop_sample_gate_last_sample_metadata",
    )
    refresh_sample_metadata = _nested_mapping_value(
        profile_payload,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_metadata",
    )
    seed = _arg_int(args, "seed", 20260530)
    return {
        "seed": seed,
        "sample_seed_base": seed,
        "sample_batch_size": _arg_int(args, "sample_batch_size", 2),
        "sample_interval": _arg_int(args, "sample_interval", 1),
        "replay_pair_capacity": _arg_int(args, "replay_pair_capacity", 16),
        "learner_train_steps": _arg_int(args, "learner_train_steps", 1),
        "policy_refresh_interval": _arg_int(args, "policy_refresh_interval", 1),
        "num_simulations": _arg_int(args, "num_simulations", 1),
        "compact_rollout_slab_sample_gate_last_seed": _optional_int(
            sample_gate.get("compact_rollout_slab_sample_gate_sample_seed")
        ),
        "compact_rollout_slab_learner_gate_last_seed": _optional_int(learner_muzero.get("seed")),
        "compact_owned_loop_sample_gate_last_metadata_seed": _optional_int(
            loop_sample_metadata.get("seed")
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed": (
            _optional_int(refresh_sample_metadata.get("seed"))
        ),
    }


def _require_lean_trainer_counter_proof(
    args: argparse.Namespace,
    profile_payload: Mapping[str, Any],
) -> None:
    if not bool(getattr(args, "compact_owned_lean_trainer_step", False)):
        return
    counter_source = str(profile_payload.get("compact_owned_trainer_loop_counter_source") or "")
    if counter_source != "run_hybrid_observation_profile":
        raise ValueError("lean trainer row must report runner-owned counter source")
    if int(profile_payload.get("compact_owned_trainer_record_step_calls") or 0) <= 0:
        raise ValueError("lean trainer row must report trainer record_step calls")
    if int(profile_payload.get("compact_owned_trainer_sample_batch_count") or 0) != int(
        profile_payload.get("compact_rollout_slab_sample_gate_calls") or 0
    ):
        raise ValueError("lean trainer sample counter must match profile loop")
    if int(profile_payload.get("compact_owned_trainer_learner_update_count") or 0) != int(
        profile_payload.get("compact_rollout_slab_learner_gate_updates") or 0
    ):
        raise ValueError("lean trainer learner counter must match profile loop")


_LEAN_PROFILE_ORACLE_COMPARE_FIELDS = (
    "death_mode",
    "env_action_checksum_total",
    "env_done_checksum_total",
    "env_reward_checksum_total",
    "env_action_mask_checksum_total",
    "env_trajectory_checksum_total",
    "env_trajectory_ordered_checksum_total",
    "env_terminal_row_checksum_total",
    "env_autoreset_row_checksum_total",
    "env_terminal_reason_checksum_total",
    "env_death_count_checksum_total",
    "env_death_cause_checksum_total",
    "env_death_hit_owner_checksum_total",
    "last_env_action_checksum",
    "last_env_trajectory_checksum",
    "last_env_terminal_row_checksum",
    "last_env_autoreset_row_checksum",
    "done_rows",
    "terminal_row_count",
    "death_row_count",
    "terminated_row_count",
    "truncated_row_count",
    "terminal_final_observation_row_count",
    "terminal_final_observation_before_autoreset_verified",
    "terminal_final_reward_map_verified",
    "autoreset_row_count",
    "done_semantics_verified",
    "death_count_total",
    "normal_collision_death_causes",
    "normal_collision_death_hit_owner_present",
    "normal_collision_death_evidence_rows",
    "compact_rollout_slab_committed_index_row_count",
    "compact_rollout_slab_sample_gate_calls",
    "compact_rollout_slab_sample_gate_sample_row_count",
    "compact_rollout_slab_sample_gate_target_row_count",
    "compact_rollout_slab_learner_gate_calls",
    "compact_rollout_slab_learner_gate_updates",
    "compact_rollout_slab_learner_gate_sample_row_count",
    "compact_rollout_slab_policy_refresh_after_learner_gate_calls",
    "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count",
    "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest",
    "resident_observation_host_fallback_count",
)


def _compact_owned_lean_profile_oracle_report(
    *,
    lean_payload: Mapping[str, Any],
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if str(lean_payload.get("compact_owned_training_loop_owner") or "") != (
        "lean_compact_trainer_step"
    ):
        raise ValueError("lean oracle requires lean_compact_trainer_step owner")
    if profile_payload.get("profile_only") is not True:
        raise ValueError("lean oracle profile payload must be profile_only=true")
    if profile_payload.get("calls_train_muzero") is not False:
        raise ValueError("lean oracle profile payload must not call train_muzero")
    if profile_payload.get("touches_live_runs") is not False:
        raise ValueError("lean oracle profile payload must not touch live runs")
    profile_owner = str(
        profile_payload.get("compact_owned_training_loop_owner")
        or "hybrid_observation_profile_runner"
    )
    if profile_owner != "hybrid_observation_profile_runner":
        raise ValueError("lean oracle profile payload owner mismatch")

    mismatches: list[dict[str, Any]] = []
    compared: list[str] = []
    for field in _LEAN_PROFILE_ORACLE_COMPARE_FIELDS:
        lean_value = _compact_owned_lean_oracle_value(lean_payload, field)
        profile_value = _compact_owned_lean_oracle_value(profile_payload, field)
        if lean_value != profile_value:
            mismatches.append(
                {
                    "field": field,
                    "lean": lean_value,
                    "profile": profile_value,
                }
            )
        compared.append(field)
    if mismatches:
        fields = ", ".join(str(item["field"]) for item in mismatches[:8])
        raise ValueError(
            "lean compact trainer profile oracle mismatch: "
            f"{fields}; mismatch_count={len(mismatches)}"
        )
    return {
        "schema_id": "curvyzero_compact_owned_lean_profile_oracle/v1",
        "ok": True,
        "compared_fields": compared,
        "mismatch_count": 0,
        "lean_owner": str(lean_payload.get("compact_owned_training_loop_owner") or ""),
        "profile_owner": profile_owner,
        "profile_only_oracle": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
    }


def _compact_owned_lean_oracle_value(payload: Mapping[str, Any], field: str) -> Any:
    if field == "resident_observation_host_fallback_count":
        timings = payload.get("timings")
        if not isinstance(timings, Mapping):
            timings = {}
        return float(
            payload.get("resident_observation_host_fallback_count")
            or timings.get("resident_observation_host_fallback_count")
            or 0.0
        )
    if field not in payload:
        raise ValueError(f"lean profile oracle payload missing field {field!r}")
    value = payload[field]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return float(value)
    return value


def _learner_num_unroll_steps(args: argparse.Namespace) -> int:
    configured = int(getattr(args, "learner_num_unroll_steps", 1))
    if configured <= 0:
        raise ValueError("learner_num_unroll_steps must be positive")
    if str(getattr(args, "death_mode", "")) == vector_runtime.DEATH_MODE_NORMAL:
        return max(configured, 2)
    return configured


def _operational_surface_fields(
    args: argparse.Namespace,
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    sample_gate = profile_payload.get("compact_rollout_slab_sample_gate_last_telemetry")
    if not isinstance(sample_gate, Mapping):
        sample_gate = {}
    normal_death_evidence = profile_payload.get("normal_death_terminal_contract_evidence")
    if not isinstance(normal_death_evidence, Mapping):
        normal_death_evidence = {}
    timings = profile_payload.get("timings")
    if not isinstance(timings, Mapping):
        timings = {}
    return {
        "actor_count": int(args.actor_count),
        "batch_size": int(getattr(args, "batch_size", profile_payload.get("batch_size", 0))),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "death_mode": str(
            profile_payload.get("death_mode")
            or getattr(args, "death_mode", vector_runtime.DEATH_MODE_PROFILE_NO_DEATH)
        ),
        "compact_owned_trainer_config_death_mode": str(
            profile_payload.get("compact_owned_trainer_config_death_mode") or ""
        ),
        "normal_death_terminal_contract_owner": str(
            profile_payload.get("normal_death_terminal_contract_owner") or "none"
        ),
        "terminal_row_count": int(profile_payload.get("terminal_row_count", 0) or 0),
        "death_row_count": int(profile_payload.get("death_row_count", 0) or 0),
        "terminated_row_count": int(profile_payload.get("terminated_row_count", 0) or 0),
        "truncated_row_count": int(profile_payload.get("truncated_row_count", 0) or 0),
        "env_action_checksum_total": int(profile_payload.get("env_action_checksum_total") or 0),
        "env_done_checksum_total": int(profile_payload.get("env_done_checksum_total") or 0),
        "env_reward_checksum_total": int(profile_payload.get("env_reward_checksum_total") or 0),
        "env_action_mask_checksum_total": int(
            profile_payload.get("env_action_mask_checksum_total") or 0
        ),
        "env_trajectory_checksum_total": int(
            profile_payload.get("env_trajectory_checksum_total") or 0
        ),
        "env_trajectory_ordered_checksum_total": int(
            profile_payload.get("env_trajectory_ordered_checksum_total") or 0
        ),
        "env_terminal_row_checksum_total": int(
            profile_payload.get("env_terminal_row_checksum_total") or 0
        ),
        "env_autoreset_row_checksum_total": int(
            profile_payload.get("env_autoreset_row_checksum_total") or 0
        ),
        "env_terminal_reason_checksum_total": int(
            profile_payload.get("env_terminal_reason_checksum_total") or 0
        ),
        "env_death_count_checksum_total": int(
            profile_payload.get("env_death_count_checksum_total") or 0
        ),
        "env_death_cause_checksum_total": int(
            profile_payload.get("env_death_cause_checksum_total") or 0
        ),
        "env_death_hit_owner_checksum_total": int(
            profile_payload.get("env_death_hit_owner_checksum_total") or 0
        ),
        "last_env_action_checksum": int(profile_payload.get("last_env_action_checksum") or 0),
        "last_env_trajectory_checksum": int(
            profile_payload.get("last_env_trajectory_checksum") or 0
        ),
        "last_env_terminal_row_checksum": int(
            profile_payload.get("last_env_terminal_row_checksum") or 0
        ),
        "last_env_autoreset_row_checksum": int(
            profile_payload.get("last_env_autoreset_row_checksum") or 0
        ),
        "terminal_final_observation_row_count": int(
            profile_payload.get("terminal_final_observation_row_count", 0) or 0
        ),
        "terminal_final_observation_before_autoreset_verified": bool(
            profile_payload.get("terminal_final_observation_before_autoreset_verified") or False
        ),
        "terminal_sample_row_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_terminal_sample_rows")
            or profile_payload.get("terminal_sample_row_count")
            or sample_gate.get("compact_rollout_slab_sample_gate_terminal_sample_row_count")
            or normal_death_evidence.get("terminal_sample_row_count")
            or 0
        ),
        "terminal_unroll_value_target_mode": str(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode"
            )
            or profile_payload.get("terminal_unroll_value_target_mode")
            or sample_gate.get("compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode")
            or normal_death_evidence.get("terminal_unroll_value_target_mode")
            or "none"
        ),
        "terminal_unroll_value_target_row_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_terminal_unroll_value_target_rows"
            )
            or profile_payload.get("terminal_unroll_value_target_row_count")
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count"
            )
            or normal_death_evidence.get("terminal_unroll_value_target_row_count")
            or 0
        ),
        "resident_observation_host_fallback_count": float(
            profile_payload.get("resident_observation_host_fallback_count")
            or timings.get("resident_observation_host_fallback_count")
            or 0.0
        ),
        "normal_death_terminal_contract_promotion_gate_satisfied": bool(
            profile_payload.get("normal_death_terminal_contract_promotion_gate_satisfied") or False
        ),
        "compact_profile_autoreset_direct_count": int(
            profile_payload.get("compact_profile_autoreset_direct_count") or 0
        ),
        "compact_profile_autoreset_template_copy_skipped_count": int(
            profile_payload.get("compact_profile_autoreset_template_copy_skipped_count") or 0
        ),
        "compact_profile_autoreset_direct_row_count": int(
            profile_payload.get("compact_profile_autoreset_direct_row_count") or 0
        ),
    }


_SAMPLE_GATE_PER_CALL_STAT_PROJECTIONS = (
    (
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_per_call_stats",
        "compact_rollout_slab_sample_gate_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_candidate_per_call_stats",
        "compact_rollout_slab_sample_gate_candidate_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_rng_per_call_stats",
        "compact_rollout_slab_sample_gate_rng_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_residual_per_call_stats",
        "compact_rollout_slab_sample_gate_residual_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_accounted_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_accounted_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_residual_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_residual_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_snapshot_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_snapshot_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_index_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_index_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_observation_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_observation_per_call",
    ),
    (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "group_loop_terminal_value_bookkeeping_per_call_stats"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "group_loop_terminal_value_bookkeeping_per_call"
        ),
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_tensor_fallback_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_tensor_fallback_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_validate_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_validate_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_final_observation_per_call",
    ),
    (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_accounted_per_call_stats"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_accounted_per_call"
        ),
    ),
    (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_residual_per_call_stats"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_residual_per_call"
        ),
    ),
    (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_presence_per_call_stats"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_presence_per_call"
        ),
    ),
    (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_select_current_per_call_stats"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_select_current_per_call"
        ),
    ),
    (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_gather_per_call_stats"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_gather_per_call"
        ),
    ),
    (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_storage_per_call_stats"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_storage_per_call"
        ),
    ),
    (
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_validate_per_call_stats"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_validate_per_call"
        ),
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_builder_select_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_builder_select_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_row_index_prepare_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_row_index_prepare_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call_stats",
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "unroll_terminal_window_hint_per_call"
        ),
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_identity_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_identity_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_build_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_build_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_value_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_value_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_apply_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_apply_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_action_stack_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_action_stack_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_write_output_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_write_output_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_per_call",
    ),
    (
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_per_call_stats",
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_per_call",
    ),
)
_BUILDER_CHILD_CPU_TIME_NAMES = (
    "group_loop",
    "group_loop_accounted",
    "group_loop_residual",
    "group_loop_prepare",
    "group_loop_prepare_accounted",
    "group_loop_prepare_residual",
    "group_loop_prepare_snapshot",
    "group_loop_prepare_index",
    "group_loop_prepare_observation",
    "group_loop_terminal_value_bookkeeping",
    "terminal_metadata",
    "terminal_metadata_accounted",
    "terminal_metadata_residual",
    "terminal_metadata_mask",
    "terminal_metadata_tensor_fallback",
    "terminal_metadata_validate",
    "terminal_metadata_final_observation",
    "terminal_metadata_final_observation_accounted",
    "terminal_metadata_final_observation_residual",
    "terminal_metadata_final_observation_presence",
    "terminal_metadata_final_observation_select_current",
    "terminal_metadata_final_observation_gather",
    "terminal_metadata_final_observation_storage",
    "terminal_metadata_final_observation_validate",
    "unroll_terminal_window_hint",
    "unroll_fields",
    "unroll_fields_accounted",
    "unroll_fields_residual",
    "unroll_builder_select",
    "unroll_row_index_prepare",
    "unroll_identity",
    "unroll_stack_fields",
    "unroll_mask_build",
    "unroll_terminal_value",
    "unroll_mask_apply",
    "unroll_action_stack",
    "write_output",
    "order_restore",
    "finalize_outputs",
    "metadata_sync",
    "metadata_build",
)
_BUILDER_CHILD_CPU_TIME_REPORT_FIELDS = tuple(
    (
        f"compact_rollout_slab_sample_gate_learner_batch_builder_{child_name}_"
        f"{scope}_cpu_time_delta_ns"
    )
    for child_name in _BUILDER_CHILD_CPU_TIME_NAMES
    for scope in ("process", "thread")
)
_TERMINAL_FINAL_OBSERVATION_PROOF_FIELDS = (
    "compact_rollout_slab_sample_gate_terminal_final_observation_group_count",
    ("compact_rollout_slab_sample_gate_terminal_final_observation_index_fast_path_count"),
    "compact_rollout_slab_sample_gate_terminal_final_observation_fallback_count",
    ("compact_rollout_slab_sample_gate_terminal_final_observation_validate_only_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_materialized_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_final_row_count_sum"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_final_row_count_max"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_dense_storage_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_sparse_storage_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_missing_storage_count"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_sparse_row_count_sum"),
    ("compact_rollout_slab_sample_gate_terminal_final_observation_sparse_row_count_max"),
)


def _sample_learner_fusion_fields(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    sample_gate = profile_payload.get("compact_rollout_slab_sample_gate_last_telemetry")
    if not isinstance(sample_gate, Mapping):
        sample_gate = {}
    owner_sample_gate = _owner_search_normalized_owner_sample_telemetry_for_proof(profile_payload)
    if not sample_gate and owner_sample_gate:
        sample_gate = owner_sample_gate
    learner_gate = profile_payload.get("compact_rollout_slab_learner_gate_last_telemetry")
    if not isinstance(learner_gate, Mapping):
        learner_gate = {}
    owner_learner_telemetry = profile_payload.get("compact_owner_search_owner_learner_telemetry")
    if not isinstance(owner_learner_telemetry, Mapping):
        owner_learner_telemetry = {}
    if not learner_gate and owner_learner_telemetry:
        learner_gate = owner_learner_telemetry
    learner_telemetry = learner_gate.get(
        "compact_rollout_slab_learner_gate_compact_muzero_telemetry"
    )
    if not isinstance(learner_telemetry, Mapping):
        learner_telemetry = {}
    if not learner_telemetry and owner_learner_telemetry:
        learner_telemetry = owner_learner_telemetry

    def _fixed_soa_handle_value(suffix: str, default: Any = None) -> Any:
        sample_key = (
            "compact_rollout_slab_sample_gate_fixed_soa_"
            f"learner_batch_handle_ring_{suffix}"
        )
        learner_key = f"compact_replay_fixed_soa_learner_batch_handle_ring_{suffix}"
        for source, key in (
            (profile_payload, sample_key),
            (sample_gate, sample_key),
            (owner_learner_telemetry, sample_key),
            (owner_learner_telemetry, learner_key),
            (learner_telemetry, sample_key),
            (learner_telemetry, learner_key),
        ):
            if isinstance(source, Mapping) and source.get(key) is not None:
                return source.get(key)
        return default

    def _sample_gate_string(
        name: str,
        *,
        default: str = "",
        missing_strings: tuple[str, ...] = ("", "none"),
    ) -> str:
        profile_value = profile_payload.get(name)
        sample_value = sample_gate.get(name)
        if profile_value is not None:
            profile_string = str(profile_value)
            if profile_string not in missing_strings:
                return profile_string
            if sample_value is None or str(sample_value) in missing_strings:
                return profile_string or default
        if sample_value is not None:
            sample_string = str(sample_value)
            if sample_string:
                return sample_string
        return default

    owner_sample_sec = float(
        profile_payload.get("compact_owner_search_owner_train_sample_sec")
        or owner_learner_telemetry.get("compact_owner_search_owner_train_sample_sec")
        or 0.0
    )
    owner_learner_sec = float(
        profile_payload.get("compact_owner_search_owner_train_learner_update_sec")
        or owner_learner_telemetry.get("compact_owner_search_owner_train_learner_update_sec")
        or 0.0
    )
    sample_sec = float(profile_payload.get("compact_rollout_slab_sample_gate_sec") or 0.0)
    learner_sec = float(profile_payload.get("compact_rollout_slab_learner_gate_sec") or 0.0)
    if sample_sec <= 0.0 and owner_sample_sec > 0.0:
        sample_sec = owner_sample_sec
    if learner_sec <= 0.0 and owner_learner_sec > 0.0:
        learner_sec = owner_learner_sec
    fields = {
        "compact_rollout_slab_sample_gate_sec": sample_sec,
        "compact_rollout_slab_learner_gate_sec": learner_sec,
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": bool(
            profile_payload.get("compact_rollout_slab_sample_gate_compact_muzero_learner_batch")
            or sample_gate.get("compact_rollout_slab_sample_gate_compact_muzero_learner_batch")
            or False
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"
            )
            or sample_gate.get("compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only")
            or False
        ),
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_host_provider_learner_batch": bool(
            profile_payload.get("compact_rollout_slab_sample_gate_host_provider_learner_batch")
            or sample_gate.get("compact_rollout_slab_sample_gate_host_provider_learner_batch")
            or False
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source": str(
            _sample_gate_string(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source"
            )
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_input_h2d_bytes"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_explicit_next_targets": bool(
            profile_payload.get("compact_rollout_slab_sample_gate_explicit_next_targets")
            or sample_gate.get("compact_rollout_slab_sample_gate_explicit_next_targets")
            or False
        ),
        "compact_rollout_slab_sample_gate_explicit_unroll_targets": bool(
            profile_payload.get("compact_rollout_slab_sample_gate_explicit_unroll_targets")
            or sample_gate.get("compact_rollout_slab_sample_gate_explicit_unroll_targets")
            or False
        ),
        "compact_rollout_slab_sample_gate_explicit_unroll_target_group_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_explicit_unroll_target_group_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_explicit_unroll_target_group_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_num_unroll_steps": int(
            profile_payload.get("compact_rollout_slab_sample_gate_num_unroll_steps")
            or sample_gate.get("compact_rollout_slab_sample_gate_num_unroll_steps")
            or 0
        ),
        "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported"
            )
            or sample_gate.get("compact_rollout_slab_sample_gate_terminal_unroll_windows_supported")
            or False
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order": str(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_sample_order"
            )
            or "unknown"
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order": (
            profile_payload.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order"
            )
            if profile_payload.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order"
            )
            is not None
            else sample_gate.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_preserves_sample_order"
            )
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count"
            )
            if profile_payload.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count"
            )
            is not None
            else sample_gate.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_order_restore_index_copy_count",
                -1,
            )
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason": str(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason"
            )
            or "none"
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl": str(
            _sample_gate_string(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl"
            )
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason": str(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason"
            )
            or "none"
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": str(
            _sample_gate_string(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl"
            )
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason": str(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason"
            )
            or "none"
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": str(
            _sample_gate_string(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"
            )
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": str(
            _sample_gate_string(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source"
            )
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_impl": str(
            _sample_gate_string(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_build_impl"
                )
            )
            or owner_learner_telemetry.get(
                "compact_muzero_learner_batch_tensor_native_replay_table_build_impl",
                "",
            )
            or ""
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_direct_build_used": bool(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_direct_build_used"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_direct_build_used"
                )
            )
            or owner_learner_telemetry.get(
                ("compact_muzero_learner_batch_tensor_native_replay_table_direct_build_used")
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count": int(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_reused_record_count"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_reused_record_count"
                )
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count": int(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_missing_record_count"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_learner_batch_builder_"
                    "tensor_native_replay_table_missing_record_count"
                )
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec": float(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec"
            )
            or 0.0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_sec": float(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_sec"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_sec"
            )
            or owner_learner_telemetry.get(
                "compact_muzero_learner_batch_tensor_native_replay_table_build_sec"
            )
            or 0.0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec": float(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec"
            )
            or 0.0
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested": bool(
            profile_payload.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested")
            )
            or sample_gate.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested")
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible": bool(
            profile_payload.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible")
            )
            or sample_gate.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible")
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used": bool(
            profile_payload.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used")
            )
            or sample_gate.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used")
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count": int(
            profile_payload.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count")
            )
            or sample_gate.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_count")
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason": str(
            profile_payload.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason")
            )
            or sample_gate.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_fallback_reason")
            )
            or "none"
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count": int(
            profile_payload.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count")
            )
            or sample_gate.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count")
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped": bool(
            profile_payload.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped")
            )
            or sample_gate.get(
                ("compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped")
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_requested": bool(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_path_requested"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_path_requested"
                )
            )
            or owner_learner_telemetry.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_path_requested"
                )
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_path_used": bool(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_path_used"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_path_used"
                )
            )
            or owner_learner_telemetry.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_path_used"
                )
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_fast_metadata_selected_group_count": int(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_selected_group_count"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_selected_group_count"
                )
            )
            or owner_learner_telemetry.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_fast_metadata_selected_group_count"
                )
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested": bool(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_requested"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_requested"
                )
            )
            or owner_learner_telemetry.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_requested"
                )
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used": bool(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_used"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_used"
                )
            )
            or owner_learner_telemetry.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_used"
                )
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_record_count": int(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_record_count"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_record_count"
                )
            )
            or owner_learner_telemetry.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_record_count"
                )
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_missing_record_count": int(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_missing_record_count"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_missing_record_count"
                )
            )
            or owner_learner_telemetry.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_missing_record_count"
                )
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_rows": int(
            profile_payload.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_rows"
                )
            )
            or sample_gate.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_rows"
                )
            )
            or owner_learner_telemetry.get(
                (
                    "compact_rollout_slab_sample_gate_tensor_native_"
                    "direct_maintained_table_handle_rows"
                )
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_requested": bool(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_requested")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_requested")
            or profile_payload.get(
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_requested"
            )
            or owner_learner_telemetry.get("compact_muzero_learner_batch_fixed_soa_requested")
            or False
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_used": bool(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_used")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_used")
            or profile_payload.get(
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_used"
            )
            or owner_learner_telemetry.get("compact_muzero_learner_batch_fixed_soa_used")
            or False
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_schema_id": str(
            _fixed_soa_handle_value("schema_id", "none")
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_requested": bool(
            _fixed_soa_handle_value("requested", False)
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_used": bool(
            _fixed_soa_handle_value("used", False)
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_handle_id": int(
            _fixed_soa_handle_value("handle_id", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_snapshot_version": int(
            _fixed_soa_handle_value("snapshot_version", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_request_checksum": int(
            _fixed_soa_handle_value("request_checksum", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_sample_row_count": int(
            _fixed_soa_handle_value("sample_row_count", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_target_row_count": int(
            _fixed_soa_handle_value("target_row_count", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_create_count": int(
            _fixed_soa_handle_value("create_count", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_resolve_count": int(
            _fixed_soa_handle_value("resolve_count", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_inline_resolve_count": int(
            _fixed_soa_handle_value("inline_resolve_count", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_count": int(
            _fixed_soa_handle_value("fallback_count", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_fallback_reason": str(
            _fixed_soa_handle_value("fallback_reason", "none")
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_batch_handle_ring_pending_handle_count": int(
            _fixed_soa_handle_value("pending_handle_count", 0) or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_slot_write_count")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_slot_write_count")
            or profile_payload.get(
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_count"
            )
            or owner_learner_telemetry.get("compact_muzero_learner_batch_fixed_soa_record_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count"
            )
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count")
            or owner_learner_telemetry.get(
                "compact_muzero_learner_batch_fixed_soa_entry_view_object_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count")
            or owner_learner_telemetry.get(
                "compact_muzero_learner_batch_fixed_soa_step_view_object_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count"
            )
            or owner_learner_telemetry.get(
                "compact_muzero_learner_batch_fixed_soa_learner_ready_object_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count"
            )
            or owner_learner_telemetry.get(
                "compact_muzero_learner_batch_fixed_soa_table_entry_object_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_table_concat_count")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_table_concat_count")
            or owner_learner_telemetry.get(
                "compact_muzero_learner_batch_fixed_soa_table_concat_count"
            )
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_record_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_record_count")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_record_count")
            or owner_learner_telemetry.get("compact_muzero_learner_batch_fixed_soa_record_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_selected_record_count")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_selected_record_count")
            or owner_learner_telemetry.get(
                "compact_muzero_learner_batch_fixed_soa_selected_record_count"
            )
            or owner_learner_telemetry.get("sampled_pair_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_table_row_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_table_row_count")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_table_row_count")
            or owner_learner_telemetry.get(
                "compact_muzero_learner_batch_tensor_native_replay_table_rows"
            )
            or owner_learner_telemetry.get("stored_index_row_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size"
            )
            or owner_learner_telemetry.get("fixed_soa_locality_sample_group_size")
            or owner_learner_telemetry.get("compact_replay_fixed_soa_locality_sample_group_size")
            or 1
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used": bool(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used")
            or owner_learner_telemetry.get("fixed_soa_locality_sample_used")
            or False
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift"
            )
            or owner_learner_telemetry.get("fixed_soa_locality_sample_semantic_drift")
            or False
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count"
            )
            or owner_learner_telemetry.get("fixed_soa_locality_selected_group_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count"
            )
            or owner_learner_telemetry.get("fixed_soa_locality_duplicate_group_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count"
            )
            or owner_learner_telemetry.get("fixed_soa_locality_local_replace_group_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_fallback_count")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_fallback_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason": str(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_fallback_reason")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_fallback_reason")
            or "none"
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec": float(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec")
            or profile_payload.get(
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_sec"
            )
            or 0.0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec": float(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec")
            or profile_payload.get(
                (
                    "compact_owner_search_direct_transition_batch_replay_"
                    "fixed_soa_successor_index_sec"
                )
            )
            or 0.0
        ),
        "compact_rollout_slab_sample_gate_fixed_soa_total_sec": float(
            profile_payload.get("compact_rollout_slab_sample_gate_fixed_soa_total_sec")
            or sample_gate.get("compact_rollout_slab_sample_gate_fixed_soa_total_sec")
            or profile_payload.get(
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_total_sec"
            )
            or 0.0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": str(
            _sample_gate_string(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path"
            )
        ),
        **{
            stats_field: dict(
                profile_payload.get(stats_field) or sample_gate.get(stats_field) or {}
            )
            for stats_field, _flat_prefix in _SAMPLE_GATE_PER_CALL_STAT_PROJECTIONS
        },
        "compact_rollout_slab_sample_gate_action_checksum": int(
            profile_payload.get("compact_rollout_slab_sample_gate_action_checksum")
            or sample_gate.get("compact_rollout_slab_sample_gate_action_checksum")
            or 0
        ),
        "compact_rollout_slab_sample_gate_sample_row_checksum": int(
            profile_payload.get("compact_rollout_slab_sample_gate_sample_row_checksum")
            or sample_gate.get("compact_rollout_slab_sample_gate_sample_row_checksum")
            or 0
        ),
        "compact_rollout_slab_sample_gate_sample_action_checksum": int(
            profile_payload.get("compact_rollout_slab_sample_gate_sample_action_checksum")
            or sample_gate.get("compact_rollout_slab_sample_gate_sample_action_checksum")
            or 0
        ),
        "compact_rollout_slab_sample_gate_sampled_flat_row_checksum": int(
            profile_payload.get("compact_rollout_slab_sample_gate_sampled_flat_row_checksum")
            or sample_gate.get("compact_rollout_slab_sample_gate_sampled_flat_row_checksum")
            or 0
        ),
        "compact_rollout_slab_sample_gate_sample_position_order_checksum": int(
            profile_payload.get("compact_rollout_slab_sample_gate_sample_position_order_checksum")
            or sample_gate.get("compact_rollout_slab_sample_gate_sample_position_order_checksum")
            or 0
        ),
        "compact_rollout_slab_sample_gate_source_record_pair_checksum": int(
            profile_payload.get("compact_rollout_slab_sample_gate_source_record_pair_checksum")
            or sample_gate.get("compact_rollout_slab_sample_gate_source_record_pair_checksum")
            or 0
        ),
        "compact_rollout_slab_sample_gate_source_record_window_checksum": int(
            profile_payload.get("compact_rollout_slab_sample_gate_source_record_window_checksum")
            or sample_gate.get("compact_rollout_slab_sample_gate_source_record_window_checksum")
            or 0
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_used": bool(
            profile_payload.get("compact_rollout_slab_learner_gate_prebuilt_batch_used")
            or learner_gate.get("compact_rollout_slab_learner_gate_prebuilt_batch_used")
            or learner_telemetry.get("compact_muzero_learner_prebuilt_batch_used")
            or False
        ),
        "compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled": bool(
            profile_payload.get("compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled")
            or learner_gate.get("compact_rollout_slab_learner_gate_cuda_memory_telemetry_enabled")
            or learner_telemetry.get("compact_muzero_learner_cuda_memory_telemetry_enabled")
            or False
        ),
        "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics": bool(
            profile_payload.get("compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics")
            or sample_gate.get("compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics")
            or False
        ),
        "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled": bool(
            profile_payload.get("compact_rollout_slab_sample_gate_cuda_sync_timing_enabled")
            or sample_gate.get("compact_rollout_slab_sample_gate_cuda_sync_timing_enabled")
            or False
        ),
        "compact_rollout_slab_sample_gate_cuda_sync_count": int(
            profile_payload.get("compact_rollout_slab_sample_gate_cuda_sync_count")
            or sample_gate.get("compact_rollout_slab_sample_gate_cuda_sync_count")
            or 0
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled": bool(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled"
            )
            or False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count": int(
            profile_payload.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count"
            )
            or sample_gate.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count"
            )
            or 0
        ),
        "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics": bool(
            profile_payload.get("compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics")
            or learner_gate.get("compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics")
            or learner_telemetry.get("compact_muzero_learner_cuda_sync_timing_diagnostics")
            or False
        ),
        "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled": bool(
            profile_payload.get("compact_rollout_slab_learner_gate_cuda_sync_timing_enabled")
            or learner_gate.get("compact_rollout_slab_learner_gate_cuda_sync_timing_enabled")
            or learner_telemetry.get("compact_muzero_learner_cuda_sync_timing_enabled")
            or False
        ),
        "compact_rollout_slab_learner_gate_cuda_sync_count": int(
            profile_payload.get("compact_rollout_slab_learner_gate_cuda_sync_count")
            or learner_gate.get("compact_rollout_slab_learner_gate_cuda_sync_count")
            or learner_telemetry.get("compact_muzero_learner_cuda_sync_count")
            or 0
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep": bool(
            profile_payload.get("compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep")
            if profile_payload.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep"
            )
            is not None
            else learner_gate.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_validation_deep",
                learner_telemetry.get(
                    "compact_muzero_learner_prebuilt_batch_validation_deep",
                    True,
                ),
            )
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used": bool(
            profile_payload.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used"
            )
            or learner_gate.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_used"
            )
            or learner_telemetry.get("compact_muzero_learner_prebuilt_batch_fast_validation_used")
            or False
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count": int(
            profile_payload.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count"
            )
            or learner_gate.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_deep_validation_count"
            )
            or learner_telemetry.get("compact_muzero_learner_prebuilt_batch_deep_validation_count")
            or 0
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count": int(
            profile_payload.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count"
            )
            or learner_gate.get(
                "compact_rollout_slab_learner_gate_prebuilt_batch_fast_validation_count"
            )
            or learner_telemetry.get("compact_muzero_learner_prebuilt_batch_fast_validation_count")
            or 0
        ),
        "compact_muzero_learner_prebuilt_batch_used": bool(
            profile_payload.get("compact_muzero_learner_prebuilt_batch_used")
            or learner_gate.get("compact_muzero_learner_prebuilt_batch_used")
            or learner_telemetry.get("compact_muzero_learner_prebuilt_batch_used")
            or False
        ),
    }
    for key in (
        "compact_rollout_slab_sample_gate_candidate_sec",
        "compact_rollout_slab_sample_gate_rng_sec",
        "compact_rollout_slab_sample_gate_resident_check_sec",
        "compact_rollout_slab_sample_gate_group_loop_sec",
        "compact_rollout_slab_sample_gate_metadata_sec",
        "compact_rollout_slab_sample_gate_learner_batch_build_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_accounted_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_residual_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_snapshot_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_index_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_prepare_observation_sec",
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "group_loop_terminal_value_bookkeeping_sec"
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_accounted_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_residual_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_sec",
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_tensor_fallback_sec"
        ),
        ("compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_validate_sec"),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_sec"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_accounted_sec"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_residual_sec"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_presence_sec"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_select_current_sec"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_gather_sec"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_storage_sec"
        ),
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "terminal_metadata_final_observation_validate_sec"
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_accounted_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_fields_residual_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_builder_select_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_row_index_prepare_sec",
        ("compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_sec"),
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_identity_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_build_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_value_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_mask_apply_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_action_stack_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_write_output_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_order_restore_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_finalize_outputs_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_sync_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_metadata_build_sec",
        "compact_rollout_slab_sample_gate_cuda_sync_sec",
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec",
        "compact_rollout_slab_sample_gate_sample_batch_build_sec",
        "compact_rollout_slab_sample_gate_accounted_sec",
        "compact_rollout_slab_sample_gate_residual_sec",
    ):
        fields[key] = float(profile_payload.get(key) or sample_gate.get(key) or 0.0)
    fields["compact_rollout_slab_sample_gate_candidate_universe_source"] = str(
        profile_payload.get("compact_rollout_slab_sample_gate_candidate_universe_source")
        or sample_gate.get("compact_rollout_slab_sample_gate_candidate_universe_source")
        or "none"
    )
    fields["compact_rollout_slab_sample_gate_candidate_universe_cache_hit"] = bool(
        profile_payload.get("compact_rollout_slab_sample_gate_candidate_universe_cache_hit")
        or sample_gate.get("compact_rollout_slab_sample_gate_candidate_universe_cache_hit")
        or False
    )
    fields["compact_rollout_slab_sample_gate_candidate_universe_snapshot_version"] = int(
        profile_payload.get("compact_rollout_slab_sample_gate_candidate_universe_snapshot_version")
        or sample_gate.get("compact_rollout_slab_sample_gate_candidate_universe_snapshot_version")
        or 0
    )
    fields["compact_rollout_slab_sample_gate_candidate_offset_checksum"] = int(
        profile_payload.get("compact_rollout_slab_sample_gate_candidate_offset_checksum")
        or sample_gate.get("compact_rollout_slab_sample_gate_candidate_offset_checksum")
        or 0
    )
    for key in _TERMINAL_FINAL_OBSERVATION_PROOF_FIELDS:
        fields[key] = int(profile_payload.get(key) or sample_gate.get(key) or 0)
    for key in _BUILDER_CHILD_CPU_TIME_REPORT_FIELDS:
        fields[key] = int(profile_payload.get(key) or sample_gate.get(key) or 0)
    for suffix in (
        "validation_sec",
        "zero_grad_sec",
        "target_transform_sec",
        "initial_inference_sec",
        "recurrent_inference_sec",
        "loss_build_sec",
        "backward_sec",
        "grad_clip_sec",
        "optimizer_step_sec",
        "loss_readback_sec",
        "final_sync_sec",
        "cuda_sync_sec",
        "accounted_sec",
        "residual_sec",
    ):
        gate_key = f"compact_rollout_slab_learner_gate_{suffix}"
        learner_key = f"compact_muzero_learner_{suffix}"
        fields[gate_key] = float(
            profile_payload.get(gate_key)
            or learner_gate.get(gate_key)
            or learner_telemetry.get(learner_key)
            or 0.0
        )
    for stats_field, flat_prefix in _SAMPLE_GATE_PER_CALL_STAT_PROJECTIONS:
        stats = fields.get(stats_field)
        if not isinstance(stats, Mapping):
            stats = {}
        fields[f"{flat_prefix}_count"] = int(stats.get("count") or 0)
        for stat_name in ("sum_sec", "min_sec", "max_sec", "p50_sec", "p95_sec"):
            fields[f"{flat_prefix}_{stat_name}"] = float(stats.get(stat_name) or 0.0)
        fields[f"{flat_prefix}_slowest_call_index"] = int(stats.get("slowest_call_index") or 0)
        fields[f"{flat_prefix}_slowest_iteration"] = int(stats.get("slowest_iteration") or 0)
        fields[f"{flat_prefix}_slowest_measured_iteration"] = int(
            stats.get("slowest_measured_iteration") or 0
        )
    trace_records = profile_payload.get("compact_rollout_slab_sample_gate_call_trace_records")
    fields["compact_rollout_slab_sample_gate_call_trace_records"] = (
        list(trace_records) if isinstance(trace_records, list) else []
    )
    return fields


_SPEED_TIMING_PROJECTION_FIELDS = (
    "source_profile_total_sec",
    "source_profile_warmup_sec",
    "source_profile_measured_sec",
    "source_profile_timing_per_timestep_sec",
    "speed_row_actor_step_wall_sec",
    "speed_row_observation_sec",
    "speed_row_renderer_stack_update_sec",
    "speed_row_compact_rollout_slab_sec",
    "speed_row_sample_gate_sec",
    "speed_row_learner_gate_sec",
    "speed_row_policy_refresh_sec",
    "speed_row_primary_accounted_sec",
    "speed_row_primary_residual_sec",
    "speed_row_actor_step_sec",
    "speed_row_actor_idle_wait_sec",
    "speed_row_actor_payload_copy_sec",
    "speed_row_actor_compact_write_sec",
    "speed_row_actor_render_state_write_sec",
    "speed_row_actor_autoreset_sec",
    "speed_row_actor_env_runtime_sec",
    "speed_row_actor_env_runtime_step_many_sec",
    "speed_row_actor_env_runtime_movement_sec",
    "speed_row_actor_env_runtime_collision_sec",
    "speed_row_actor_env_runtime_visual_trail_append_sec",
    "speed_row_actor_env_runtime_body_append_sec",
    "speed_row_actor_env_runtime_phase_accounted_sec",
    "speed_row_actor_env_runtime_phase_residual_sec",
    "speed_row_actor_env_public_prepare_sec",
    "speed_row_actor_env_public_info_sec",
    "speed_row_actor_env_compact_action_mask_sec",
    "speed_row_actor_env_reward_sec",
    "speed_row_actor_env_final_observation_sec",
    "speed_row_actor_env_batch_pack_sec",
    "speed_row_actor_env_post_runtime_bookkeeping_sec",
    "speed_row_actor_step_other_sec",
    "speed_row_renderer_render_sec",
    "speed_row_renderer_device_render_sec",
    "speed_row_renderer_host_to_device_sec",
    "speed_row_renderer_device_to_host_sec",
    "speed_row_renderer_production_to_compact_sec",
    "speed_row_renderer_persistent_compact_state_handoff_sec",
    "speed_row_renderer_persistent_delta_pack_sec",
    "speed_row_renderer_persistent_update_sec",
    "speed_row_stack_shift_sec",
    "speed_row_stack_latest_update_sec",
    "speed_row_resident_observation_stack_update_sec",
    "speed_row_resident_observation_frame_view_sec",
    "speed_row_resident_observation_stack_shift_sec",
    "speed_row_resident_observation_latest_write_sec",
    "speed_row_resident_observation_autoreset_sec",
    "speed_row_resident_observation_autoreset_frame_view_sec",
    "speed_row_resident_observation_autoreset_index_build_sec",
    "speed_row_resident_observation_autoreset_zero_sec",
    "speed_row_resident_observation_autoreset_latest_write_sec",
    "speed_row_scalar_materialization_sec",
    "speed_row_resident_observation_replay_snapshot_sec",
    "speed_row_observation_other_sec",
)
_SPEED_ACTOR_OBSERVATION_TIMING_FIELDS = (
    ("speed_row_actor_step_sec", "actor_step_sec"),
    ("speed_row_actor_idle_wait_sec", "actor_idle_wait_sec"),
    ("speed_row_actor_payload_copy_sec", "actor_payload_copy_sec"),
    ("speed_row_actor_compact_write_sec", "actor_compact_write_sec"),
    ("speed_row_actor_render_state_write_sec", "actor_render_state_write_sec"),
    ("speed_row_actor_autoreset_sec", "actor_autoreset_sec"),
    ("speed_row_actor_env_runtime_sec", "actor_env_runtime_sec"),
    ("speed_row_actor_env_runtime_step_many_sec", "actor_env_runtime_step_many_sec"),
    ("speed_row_actor_env_runtime_movement_sec", "actor_env_runtime_movement_sec"),
    ("speed_row_actor_env_runtime_collision_sec", "actor_env_runtime_collision_sec"),
    (
        "speed_row_actor_env_runtime_visual_trail_append_sec",
        "actor_env_runtime_visual_trail_append_sec",
    ),
    ("speed_row_actor_env_runtime_body_append_sec", "actor_env_runtime_body_append_sec"),
    (
        "speed_row_actor_env_runtime_phase_accounted_sec",
        "actor_env_runtime_phase_accounted_sec",
    ),
    (
        "speed_row_actor_env_runtime_phase_residual_sec",
        "actor_env_runtime_phase_residual_sec",
    ),
    ("speed_row_actor_env_public_prepare_sec", "actor_env_public_prepare_sec"),
    ("speed_row_actor_env_public_info_sec", "actor_env_public_info_sec"),
    ("speed_row_actor_env_compact_action_mask_sec", "actor_env_compact_action_mask_sec"),
    ("speed_row_actor_env_reward_sec", "actor_env_reward_sec"),
    ("speed_row_actor_env_final_observation_sec", "actor_env_final_observation_sec"),
    ("speed_row_actor_env_batch_pack_sec", "actor_env_batch_pack_sec"),
    (
        "speed_row_actor_env_post_runtime_bookkeeping_sec",
        "actor_env_post_runtime_bookkeeping_sec",
    ),
    ("speed_row_renderer_render_sec", "renderer_render_sec"),
    ("speed_row_renderer_device_render_sec", "renderer_device_render_sec"),
    ("speed_row_renderer_host_to_device_sec", "renderer_host_to_device_sec"),
    ("speed_row_renderer_device_to_host_sec", "renderer_device_to_host_sec"),
    ("speed_row_renderer_production_to_compact_sec", "renderer_production_to_compact_sec"),
    (
        "speed_row_renderer_persistent_compact_state_handoff_sec",
        "renderer_persistent_compact_state_handoff_sec",
    ),
    ("speed_row_renderer_persistent_delta_pack_sec", "renderer_persistent_delta_pack_sec"),
    ("speed_row_renderer_persistent_update_sec", "renderer_persistent_update_sec"),
    ("speed_row_stack_shift_sec", "stack_shift_sec"),
    ("speed_row_stack_latest_update_sec", "stack_latest_update_sec"),
    (
        "speed_row_resident_observation_stack_update_sec",
        "resident_observation_stack_update_sec",
    ),
    ("speed_row_resident_observation_frame_view_sec", "resident_observation_frame_view_sec"),
    ("speed_row_resident_observation_stack_shift_sec", "resident_observation_stack_shift_sec"),
    (
        "speed_row_resident_observation_latest_write_sec",
        "resident_observation_latest_write_sec",
    ),
    (
        "speed_row_resident_observation_autoreset_sec",
        "resident_observation_autoreset_sec",
    ),
    (
        "speed_row_resident_observation_autoreset_frame_view_sec",
        "resident_observation_autoreset_frame_view_sec",
    ),
    (
        "speed_row_resident_observation_autoreset_index_build_sec",
        "resident_observation_autoreset_index_build_sec",
    ),
    (
        "speed_row_resident_observation_autoreset_zero_sec",
        "resident_observation_autoreset_zero_sec",
    ),
    (
        "speed_row_resident_observation_autoreset_latest_write_sec",
        "resident_observation_autoreset_latest_write_sec",
    ),
    ("speed_row_scalar_materialization_sec", "scalar_materialization_sec"),
    (
        "speed_row_resident_observation_replay_snapshot_sec",
        "resident_observation_replay_snapshot_sec",
    ),
)
_COMPACT_ROLLOUT_SLAB_TOTAL_HEADLINE_FIELDS = (
    "compact_rollout_slab_commit_previous_sec",
    "compact_rollout_slab_root_batch_build_sec",
    "compact_rollout_slab_root_build_request_sec",
    "compact_rollout_slab_search_dispatch_wall_sec",
    "compact_rollout_slab_search_dispatch_service_envelope_sec",
    "compact_rollout_slab_search_dispatch_positive_residual_sec",
    "compact_rollout_slab_search_identity_validation_sec",
    "compact_rollout_slab_joint_action_assembly_sec",
    "compact_rollout_slab_pending_store_sec",
    "compact_rollout_slab_telemetry_build_sec",
    "compact_rollout_slab_replay_index_rows_build_sec",
    "compact_rollout_slab_replay_index_rows_store_sec",
    "compact_rollout_slab_owner_replay_stage_sec",
    "compact_rollout_slab_owner_search_parent_publish_sec",
    "compact_rollout_slab_owner_search_parent_submit_sec",
    "compact_rollout_slab_owner_search_parent_wait_sec",
    "compact_rollout_slab_owner_search_parent_wall_sec",
    "compact_rollout_slab_owner_search_worker_wall_sec",
    "compact_rollout_slab_owner_search_worker_root_resolve_sec",
    "compact_rollout_slab_owner_search_worker_search_sec",
    "compact_rollout_slab_owner_search_worker_replay_append_sec",
    "compact_rollout_slab_owner_search_worker_learner_train_sec",
    "compact_rollout_slab_owner_search_worker_search_refresh_sec",
    "compact_rollout_slab_resident_host_observation_stub_materialized_bytes",
    "compact_rollout_slab_resident_host_observation_stub_logical_bytes",
)


def _compact_rollout_slab_telemetry_total_fields(
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    raw_totals = profile_payload.get("compact_rollout_slab_telemetry_totals")
    totals: dict[str, float] = {}
    if isinstance(raw_totals, Mapping):
        for raw_key, raw_value in raw_totals.items():
            key = str(raw_key)
            try:
                value = float(raw_value or 0.0)
            except (TypeError, ValueError):
                continue
            totals[key] = value if math.isfinite(value) else 0.0
    fields: dict[str, Any] = {"compact_rollout_slab_telemetry_totals": totals}
    for key in _COMPACT_ROLLOUT_SLAB_TOTAL_HEADLINE_FIELDS:
        headline = key.removeprefix("compact_rollout_slab_")
        fields[f"speed_row_total_{headline}"] = float(totals.get(key, 0.0))
    return fields


_WHOLE_OWNER_BUFFER_REPLAY_CEILING_FIELDS = (
    "compact_whole_owner_buffer_replay_ceiling_schema_id",
    "compact_whole_owner_buffer_replay_ceiling_enabled",
    "compact_whole_owner_buffer_replay_ceiling_projection_only",
    "compact_whole_owner_buffer_replay_ceiling_production_speed_claim",
    "compact_whole_owner_buffer_replay_ceiling_touches_live_training",
    "compact_whole_owner_buffer_replay_ceiling_requires_h100_validation",
    "compact_whole_owner_buffer_replay_ceiling_speed_currency",
    "compact_whole_owner_buffer_replay_ceiling_projection_source",
    "compact_whole_owner_buffer_replay_ceiling_basis",
    "compact_whole_owner_buffer_replay_ceiling_h100_validation_status",
    "compact_whole_owner_buffer_replay_ceiling_variance_interpretation",
    "compact_whole_owner_buffer_replay_ceiling_promotion_eligible",
    "compact_whole_owner_buffer_replay_ceiling_observed_env_steps",
    "compact_whole_owner_buffer_replay_ceiling_observed_wall_sec",
    "compact_whole_owner_buffer_replay_ceiling_observed_env_steps_per_sec",
    "compact_whole_owner_buffer_replay_ceiling_baseline_env_steps_per_sec",
    "compact_whole_owner_buffer_replay_ceiling_baseline_whole_loop_sec",
    "compact_whole_owner_buffer_replay_ceiling_target_multiplier",
    "compact_whole_owner_buffer_replay_ceiling_target_env_steps_per_sec",
    "compact_whole_owner_buffer_replay_ceiling_target_wall_sec",
    "compact_whole_owner_buffer_replay_ceiling_observed_speedup_vs_opt104",
    "compact_whole_owner_buffer_replay_ceiling_observed_replay_append_sec",
    "compact_whole_owner_buffer_replay_ceiling_observed_owner_train_sample_sec",
    "compact_whole_owner_buffer_replay_ceiling_observed_owner_train_wall_sec",
    "compact_whole_owner_buffer_replay_ceiling_observed_learner_update_sec",
    "compact_whole_owner_buffer_replay_ceiling_observed_worker_search_sec",
    "compact_whole_owner_buffer_replay_ceiling_observed_parent_wait_sec",
    "compact_whole_owner_buffer_replay_ceiling_direct_replay_sample_surface_sec",
    "compact_whole_owner_buffer_replay_ceiling_parent_wait_bounded_surface_sec",
    "compact_whole_owner_buffer_replay_ceiling_preserved_search_update_floor_sec",
    "compact_whole_owner_buffer_replay_ceiling_max_removable_sec",
    "compact_whole_owner_buffer_replay_ceiling_projected_removed_sec",
    "compact_whole_owner_buffer_replay_ceiling_projected_wall_sec",
    "compact_whole_owner_buffer_replay_ceiling_projected_env_steps_per_sec",
    "compact_whole_owner_buffer_replay_ceiling_projected_speedup_vs_opt104",
    "compact_whole_owner_buffer_replay_ceiling_projected_delta_sec",
    "compact_whole_owner_buffer_replay_ceiling_projected_reaches_2x",
    "compact_whole_owner_buffer_replay_ceiling_additional_removed_sec_to_2x",
)


def _whole_owner_buffer_replay_ceiling_fields(
    profile_payload: Mapping[str, Any],
    *,
    training_wall_sec: float,
    env_steps_collected: float,
    observed_steps_per_sec: float,
) -> dict[str, Any]:
    proof_fields = _owner_search_slab_proxy_proof_fields(profile_payload)
    slab_total_fields = _compact_rollout_slab_telemetry_total_fields(profile_payload)

    def surface_float(*keys: str) -> float:
        candidates: list[float] = []
        for source in (profile_payload, proof_fields, slab_total_fields):
            for key in keys:
                value = source.get(key)
                try:
                    number = float(value or 0.0)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number) and number >= 0.0:
                    candidates.append(number)
        return max(candidates) if candidates else 0.0

    replay_append_sec = surface_float(
        "compact_owner_search_worker_replay_append_sec",
        "speed_row_total_owner_search_worker_replay_append_sec",
    )
    owner_train_sample_sec = surface_float(
        "compact_owner_search_owner_train_sample_sec",
    )
    owner_train_wall_sec = surface_float("compact_owner_search_owner_train_wall_sec")
    learner_update_sec = surface_float("compact_owner_search_owner_train_learner_update_sec")
    worker_search_sec = surface_float(
        "compact_owner_search_worker_search_sec",
        "speed_row_total_owner_search_worker_search_sec",
    )
    parent_wait_sec = surface_float(
        "compact_owner_search_parent_wait_sec",
        "speed_row_total_owner_search_parent_wait_sec",
    )

    wall_sec = max(0.0, float(training_wall_sec))
    env_steps = max(0.0, float(env_steps_collected))
    observed_speed = max(0.0, float(observed_steps_per_sec))
    direct_surface_sec = replay_append_sec + owner_train_sample_sec
    parent_wait_bounded_surface_sec = min(parent_wait_sec, direct_surface_sec)
    preserved_floor_sec = worker_search_sec + learner_update_sec
    max_removable_sec = max(0.0, wall_sec - preserved_floor_sec)
    projected_removed_sec = min(parent_wait_bounded_surface_sec, max_removable_sec)
    projected_wall_sec = max(0.0, wall_sec - projected_removed_sec)
    projected_speed = env_steps / projected_wall_sec if projected_wall_sec > 0.0 else 0.0
    baseline_speed = OPT104_BASELINE_ENV_STEPS_PER_SEC
    target_multiplier = WHOLE_OWNER_BUFFER_REPLAY_CEILING_TARGET_MULTIPLIER
    target_speed = baseline_speed * target_multiplier
    baseline_wall_sec = env_steps / baseline_speed if env_steps > 0.0 else 0.0
    target_wall_sec = env_steps / target_speed if env_steps > 0.0 else 0.0
    projection_enabled = (
        bool(proof_fields.get("compact_owner_search_slab_proxy"))
        and wall_sec > 0.0
        and env_steps > 0.0
        and (direct_surface_sec > 0.0 or parent_wait_sec > 0.0)
    )

    return {
        "compact_whole_owner_buffer_replay_ceiling_schema_id": (
            WHOLE_OWNER_BUFFER_REPLAY_CEILING_SCHEMA_ID
        ),
        "compact_whole_owner_buffer_replay_ceiling_enabled": projection_enabled,
        "compact_whole_owner_buffer_replay_ceiling_projection_only": True,
        "compact_whole_owner_buffer_replay_ceiling_production_speed_claim": False,
        "compact_whole_owner_buffer_replay_ceiling_touches_live_training": False,
        "compact_whole_owner_buffer_replay_ceiling_requires_h100_validation": True,
        "compact_whole_owner_buffer_replay_ceiling_speed_currency": (
            WHOLE_OWNER_BUFFER_REPLAY_CEILING_SPEED_CURRENCY
        ),
        "compact_whole_owner_buffer_replay_ceiling_projection_source": (
            "measured_owner_search_surface_projection_v1"
        ),
        "compact_whole_owner_buffer_replay_ceiling_basis": (
            "owner_replay_append_train_sample_parent_wait_bound_v1"
        ),
        "compact_whole_owner_buffer_replay_ceiling_h100_validation_status": "not_run",
        "compact_whole_owner_buffer_replay_ceiling_variance_interpretation": (
            "projection_not_measurement"
        ),
        "compact_whole_owner_buffer_replay_ceiling_promotion_eligible": False,
        "compact_whole_owner_buffer_replay_ceiling_observed_env_steps": env_steps,
        "compact_whole_owner_buffer_replay_ceiling_observed_wall_sec": wall_sec,
        "compact_whole_owner_buffer_replay_ceiling_observed_env_steps_per_sec": (observed_speed),
        "compact_whole_owner_buffer_replay_ceiling_baseline_env_steps_per_sec": (baseline_speed),
        "compact_whole_owner_buffer_replay_ceiling_baseline_whole_loop_sec": (baseline_wall_sec),
        "compact_whole_owner_buffer_replay_ceiling_target_multiplier": target_multiplier,
        "compact_whole_owner_buffer_replay_ceiling_target_env_steps_per_sec": target_speed,
        "compact_whole_owner_buffer_replay_ceiling_target_wall_sec": target_wall_sec,
        "compact_whole_owner_buffer_replay_ceiling_observed_speedup_vs_opt104": (
            observed_speed / baseline_speed if baseline_speed > 0.0 else 0.0
        ),
        "compact_whole_owner_buffer_replay_ceiling_observed_replay_append_sec": (replay_append_sec),
        "compact_whole_owner_buffer_replay_ceiling_observed_owner_train_sample_sec": (
            owner_train_sample_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_observed_owner_train_wall_sec": (
            owner_train_wall_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_observed_learner_update_sec": (
            learner_update_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_observed_worker_search_sec": (worker_search_sec),
        "compact_whole_owner_buffer_replay_ceiling_observed_parent_wait_sec": (parent_wait_sec),
        "compact_whole_owner_buffer_replay_ceiling_direct_replay_sample_surface_sec": (
            direct_surface_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_parent_wait_bounded_surface_sec": (
            parent_wait_bounded_surface_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_preserved_search_update_floor_sec": (
            preserved_floor_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_max_removable_sec": max_removable_sec,
        "compact_whole_owner_buffer_replay_ceiling_projected_removed_sec": (projected_removed_sec),
        "compact_whole_owner_buffer_replay_ceiling_projected_wall_sec": (projected_wall_sec),
        "compact_whole_owner_buffer_replay_ceiling_projected_env_steps_per_sec": (projected_speed),
        "compact_whole_owner_buffer_replay_ceiling_projected_speedup_vs_opt104": (
            projected_speed / baseline_speed if baseline_speed > 0.0 else 0.0
        ),
        "compact_whole_owner_buffer_replay_ceiling_projected_delta_sec": (
            wall_sec - projected_wall_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_projected_reaches_2x": (
            projection_enabled and projected_speed >= target_speed
        ),
        "compact_whole_owner_buffer_replay_ceiling_additional_removed_sec_to_2x": max(
            0.0,
            projected_wall_sec - target_wall_sec,
        ),
    }


def _speed_timing_projection_fields(
    profile_payload: Mapping[str, Any],
    *,
    training_wall_sec: float,
) -> dict[str, float]:
    timings = profile_payload.get("timings")
    if not isinstance(timings, Mapping):
        timings = {}

    def timing_value(top_level: str, timing_key: str | None = None) -> float:
        key = timing_key or top_level
        value = profile_payload.get(top_level)
        if value is None:
            value = timings.get(key)
        try:
            number = float(value or 0.0)
        except (TypeError, ValueError):
            number = 0.0
        return number if math.isfinite(number) else 0.0

    actor_step = timing_value("actor_step_wall_sec")
    observation = timing_value("observation_sec")
    renderer_stack = timing_value("renderer_stack_update_sec")
    slab = timing_value("compact_rollout_slab_sec")
    sample = timing_value("compact_rollout_slab_sample_gate_sec")
    learner = timing_value("compact_rollout_slab_learner_gate_sec")
    refresh = timing_value("compact_rollout_slab_policy_refresh_after_learner_gate_sec")
    total = timing_value("total_sec")
    warmup = timing_value("warmup_sec")
    measured = timing_value("measured_sec")
    accounted = actor_step + observation + slab + sample + learner + refresh
    wall = float(training_wall_sec)
    timestep_count = float(profile_payload.get("steps") or 0.0)
    env_step_count = timestep_count * float(profile_payload.get("batch_size") or 0.0)
    fields = {
        "source_profile_total_sec": total,
        "source_profile_warmup_sec": warmup,
        "source_profile_measured_sec": measured,
        "source_profile_timing_per_timestep_sec": (
            wall / env_step_count if env_step_count > 0.0 and wall > 0.0 else 0.0
        ),
        "speed_row_actor_step_wall_sec": actor_step,
        "speed_row_observation_sec": observation,
        "speed_row_renderer_stack_update_sec": renderer_stack,
        "speed_row_compact_rollout_slab_sec": slab,
        "speed_row_sample_gate_sec": sample,
        "speed_row_learner_gate_sec": learner,
        "speed_row_policy_refresh_sec": refresh,
        "speed_row_primary_accounted_sec": accounted,
        "speed_row_primary_residual_sec": wall - accounted,
    }
    for surface_key, timing_key in _SPEED_ACTOR_OBSERVATION_TIMING_FIELDS:
        fields[surface_key] = timing_value(timing_key)
    actor_child_accounted = (
        fields["speed_row_actor_autoreset_sec"]
        + fields["speed_row_actor_env_runtime_sec"]
        + fields["speed_row_actor_env_public_prepare_sec"]
        + fields["speed_row_actor_env_public_info_sec"]
        + fields["speed_row_actor_env_compact_action_mask_sec"]
        + fields["speed_row_actor_env_reward_sec"]
        + fields["speed_row_actor_env_final_observation_sec"]
        + fields["speed_row_actor_env_batch_pack_sec"]
        + fields["speed_row_actor_env_post_runtime_bookkeeping_sec"]
    )
    fields["speed_row_actor_step_other_sec"] = (
        fields["speed_row_actor_step_sec"] - actor_child_accounted
    )
    observation_child_accounted = (
        fields["speed_row_renderer_render_sec"]
        + fields["speed_row_stack_shift_sec"]
        + fields["speed_row_stack_latest_update_sec"]
        + fields["speed_row_resident_observation_stack_update_sec"]
        + fields["speed_row_resident_observation_autoreset_sec"]
        + fields["speed_row_scalar_materialization_sec"]
        + fields["speed_row_resident_observation_replay_snapshot_sec"]
    )
    fields["speed_row_observation_other_sec"] = observation - observation_child_accounted
    return fields


def _require_fused_learner_batch_proof(
    args: argparse.Namespace,
    profile_payload: Mapping[str, Any],
) -> None:
    if not bool(getattr(args, "compact_owned_loop_fused_learner_batch", False)):
        return
    fields = _sample_learner_fusion_fields(profile_payload)
    owner_search_fields = _owner_search_slab_proxy_proof_fields(profile_payload)

    def _owner_search_field(name: str, default: Any = None) -> Any:
        value = profile_payload.get(name)
        if value is not None:
            return value
        return owner_search_fields.get(name, default)

    owner_search_owner_batch = (
        str(
            profile_payload.get("compact_owned_training_loop_owner")
            or ("owner_search_worker" if _owner_search_owner_train_enabled(args) else "")
        )
        == "owner_search_worker"
        and _owner_search_field("compact_owner_search_action_only_result") is True
        and _owner_search_field("compact_owner_search_owner_materializes_replay") is True
        and _owner_search_field("compact_owner_search_parent_slab_commits_replay") is False
        and int(_owner_search_field("compact_owner_search_owner_train_request_count", 0) or 0) > 0
        and int(
            _owner_search_field(
                "compact_owner_search_owner_expected_train_request_count",
                0,
            )
            or 0
        )
        == int(_owner_search_field("compact_owner_search_owner_train_request_count", 0) or 0)
        and max(
            int(_owner_search_field("compact_owner_search_learner_update_count", 0) or 0),
            int(_owner_search_field("compact_owner_search_owner_learner_update_count", 0) or 0),
        )
        > 0
    )
    required_true = [
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only",
    ]
    if not owner_search_owner_batch:
        required_true.append("compact_rollout_slab_learner_gate_prebuilt_batch_used")
    missing = [key for key in required_true if fields.get(key) is not True]
    if (
        owner_search_owner_batch
        and float(_owner_search_field("compact_owner_search_worker_learner_train_sec", 0.0) or 0.0)
        <= 0.0
    ):
        missing.append("compact_owner_search_worker_learner_train_sec")
    resident_fused = bool(
        fields.get("compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch")
        and fields.get(
            "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"
        )
    )
    host_provider_fused = bool(
        fields.get("compact_rollout_slab_sample_gate_host_provider_learner_batch")
    )
    tensor_native_fused = bool(
        fields.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
        )
        is True
    )
    if not resident_fused and not host_provider_fused and not tensor_native_fused:
        missing.append(
            "resident_grouped_device_learner_batch or host_provider_learner_batch "
            "or tensor_native_replay"
        )
    learner_ready_unroll2_cache_requested = bool(
        getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
    )
    tensor_native_replay_requested = bool(
        getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
    )
    tensor_native_impl = str(
        fields.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
            "",
        )
    )
    tensor_native_table_source = str(
        fields.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
            "",
        )
    )
    learner_ready_cache_impl = str(
        fields.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl",
            "",
        )
    )
    learner_batch_builder_path = str(
        fields.get(
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path",
            "",
        )
    )
    fixed_soa_tensor_native = bool(
        tensor_native_impl == "fixed_soa_direct_gather_v1"
        and tensor_native_table_source == "fixed_soa_columns_v1"
    )
    if bool(
        getattr(args, "compact_muzero_learner_batch_unroll2_specialized_builder", False)
    ) and not (learner_ready_unroll2_cache_requested):
        if _learner_num_unroll_steps(args) != 2:
            missing.append("learner_num_unroll_steps == 2")
        if (
            fields.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested"
            )
            is not True
        ):
            missing.append("unroll2_specialized_builder_requested")
        if (
            fields.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used"
            )
            is not True
        ):
            missing.append("unroll2_specialized_builder_used")
        if (
            int(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count",
                    0,
                )
                or 0
            )
            <= 0
        ):
            missing.append("unroll2_specialized_builder_call_count")
        if (
            int(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count",
                    0,
                )
                or 0
            )
            != 0
        ):
            missing.append("unroll2_specialized_builder_fallback_count")
        if (
            str(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason",
                    "",
                )
            )
            != "none"
        ):
            missing.append("unroll2_specialized_builder_fallback_reason")
        if (
            str(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl",
                    "",
                )
            )
            != "unroll2_specialized_v1"
        ):
            missing.append("unroll2_specialized_builder_impl")
        if (
            str(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path",
                    "",
                )
            )
            != "unroll2_specialized"
        ):
            missing.append("unroll2_specialized_builder_path")
    if learner_ready_unroll2_cache_requested:
        if _learner_num_unroll_steps(args) != 2:
            missing.append("learner_num_unroll_steps == 2")
        if (
            fields.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested"
            )
            is not True
        ):
            missing.append("learner_ready_unroll2_cache_requested")
        if (
            int(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count",
                    0,
                )
                or 0
            )
            != 0
        ):
            missing.append("learner_ready_unroll2_cache_fallback_count")
        if (
            str(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason",
                    "",
                )
            )
            != "none"
        ):
            missing.append("learner_ready_unroll2_cache_fallback_reason")
        if fixed_soa_tensor_native:
            if learner_ready_cache_impl != "fixed_soa_columns_v1":
                missing.append("learner_ready_unroll2_cache_impl")
            if learner_batch_builder_path != "fixed_soa_direct_gather":
                missing.append("learner_ready_unroll2_cache_builder_path")
        else:
            if (
                int(
                    fields.get(
                        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count",
                        0,
                    )
                    or 0
                )
                <= 0
            ):
                missing.append("learner_ready_unroll2_cache_available_group_count")
            if (
                int(
                    fields.get(
                        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count",
                        0,
                    )
                    or 0
                )
                <= 0
            ):
                missing.append("learner_ready_unroll2_cache_eligible_count")
            if (
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used"
                )
                is not True
            ):
                missing.append("learner_ready_unroll2_cache_used")
            if (
                int(
                    fields.get(
                        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count",
                        0,
                    )
                    or 0
                )
                <= 0
            ):
                missing.append("learner_ready_unroll2_cache_call_count")
            if learner_ready_cache_impl != "learner_ready_unroll2_cache_v1":
                missing.append("learner_ready_unroll2_cache_impl")
            if learner_batch_builder_path != "learner_ready_unroll2_cache":
                missing.append("learner_ready_unroll2_cache_builder_path")
    if tensor_native_replay_requested:
        if _learner_num_unroll_steps(args) != 2:
            missing.append("learner_num_unroll_steps == 2")
        if (
            fields.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested"
            )
            is not True
        ):
            missing.append("tensor_native_replay_requested")
        if (
            fields.get(
                "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used"
            )
            is not True
        ):
            missing.append("tensor_native_replay_used")
        if (
            int(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count",
                    0,
                )
                or 0
            )
            <= 0
        ):
            missing.append("tensor_native_replay_call_count")
        if (
            int(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
                    0,
                )
                or 0
            )
            != 0
        ):
            missing.append("tensor_native_replay_fallback_count")
        if (
            str(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason",
                    "",
                )
            )
            != "none"
        ):
            missing.append("tensor_native_replay_fallback_reason")
        if tensor_native_impl not in {
            "maintained_unroll2_table_gather_v1",
            "selected_maintained_record_table_gather_v1",
            "selected_direct_record_table_gather_v1",
            "fixed_soa_direct_gather_v1",
        }:
            missing.append("tensor_native_replay_impl")
        if tensor_native_table_source not in {
            "maintained_record_table_v1",
            "selected_maintained_record_table_v1",
            "selected_direct_record_table_v1",
            "fixed_soa_columns_v1",
        }:
            missing.append("tensor_native_replay_table_source")
        if tensor_native_impl == "selected_maintained_record_table_gather_v1":
            if tensor_native_table_source != "selected_maintained_record_table_v1":
                missing.append("selected_maintained_replay_table_source")
            if (
                fields.get(
                    (
                        "compact_rollout_slab_sample_gate_tensor_native_"
                        "direct_fast_metadata_path_requested"
                    )
                )
                is not True
            ):
                missing.append("tensor_native_direct_fast_metadata_path_requested")
            if (
                fields.get(
                    (
                        "compact_rollout_slab_sample_gate_tensor_native_"
                        "direct_fast_metadata_path_used"
                    )
                )
                is not True
            ):
                missing.append("tensor_native_direct_fast_metadata_path_used")
            if (
                int(
                    fields.get(
                        (
                            "compact_rollout_slab_sample_gate_tensor_native_"
                            "direct_fast_metadata_selected_group_count"
                        ),
                        0,
                    )
                    or 0
                )
                <= 0
            ):
                missing.append("tensor_native_direct_fast_metadata_selected_group_count")
        if (
            fields.get(
                "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_requested"
            )
            is True
        ):
            if (
                fields.get(
                    "compact_rollout_slab_sample_gate_tensor_native_direct_maintained_table_handle_used"
                )
                is not True
            ):
                missing.append("tensor_native_direct_maintained_table_handle_used")
            if (
                int(
                    fields.get(
                        (
                            "compact_rollout_slab_sample_gate_tensor_native_"
                            "direct_maintained_table_handle_record_count"
                        ),
                        0,
                    )
                    or 0
                )
                <= 0
            ):
                missing.append("tensor_native_direct_maintained_table_handle_record_count")
            if (
                int(
                    fields.get(
                        (
                            "compact_rollout_slab_sample_gate_tensor_native_"
                            "direct_maintained_table_handle_missing_record_count"
                        ),
                        -1,
                    )
                    or 0
                )
                != 0
            ):
                missing.append(
                    "tensor_native_direct_maintained_table_handle_missing_record_count"
                )
            if (
                int(
                    fields.get(
                        (
                            "compact_rollout_slab_sample_gate_tensor_native_"
                            "direct_maintained_table_handle_rows"
                        ),
                        0,
                    )
                    or 0
                )
                <= 0
            ):
                missing.append("tensor_native_direct_maintained_table_handle_rows")
        if (
            tensor_native_impl == "fixed_soa_direct_gather_v1"
            and fields.get(
                (
                    "compact_rollout_slab_sample_gate_fixed_soa_"
                    "learner_batch_handle_ring_requested"
                )
            )
            is True
            and int(
                fields.get(
                    (
                        "compact_rollout_slab_sample_gate_fixed_soa_"
                        "learner_batch_handle_ring_sample_row_count"
                    ),
                    0,
                )
                or 0
            )
            > 0
        ):
            if (
                fields.get(
                    (
                        "compact_rollout_slab_sample_gate_fixed_soa_"
                        "learner_batch_handle_ring_used"
                    )
                )
                is not True
            ):
                missing.append("fixed_soa_learner_batch_handle_ring_used")
            for field in (
                (
                    "compact_rollout_slab_sample_gate_fixed_soa_"
                    "learner_batch_handle_ring_create_count"
                ),
                (
                    "compact_rollout_slab_sample_gate_fixed_soa_"
                    "learner_batch_handle_ring_resolve_count"
                ),
                (
                    "compact_rollout_slab_sample_gate_fixed_soa_"
                    "learner_batch_handle_ring_inline_resolve_count"
                ),
            ):
                if int(fields.get(field, 0) or 0) != 1:
                    missing.append(field)
            for field in (
                (
                    "compact_rollout_slab_sample_gate_fixed_soa_"
                    "learner_batch_handle_ring_fallback_count"
                ),
                (
                    "compact_rollout_slab_sample_gate_fixed_soa_"
                    "learner_batch_handle_ring_pending_handle_count"
                ),
            ):
                if int(fields.get(field, -1) or 0) != 0:
                    missing.append(field)
            if (
                str(
                    fields.get(
                        (
                            "compact_rollout_slab_sample_gate_fixed_soa_"
                            "learner_batch_handle_ring_fallback_reason"
                        ),
                        "",
                    )
                )
                != "none"
            ):
                missing.append("fixed_soa_learner_batch_handle_ring_fallback_reason")
        if (
            int(
                fields.get(
                    (
                        "compact_rollout_slab_sample_gate_learner_batch_builder_"
                        "tensor_native_replay_table_reused_record_count"
                    ),
                    0,
                )
                or 0
            )
            <= 0
        ):
            missing.append("tensor_native_replay_table_reused_record_count")
        if (
            int(
                fields.get(
                    (
                        "compact_rollout_slab_sample_gate_learner_batch_builder_"
                        "tensor_native_replay_table_missing_record_count"
                    ),
                    -1,
                )
                or 0
            )
            != 0
        ):
            missing.append("tensor_native_replay_table_missing_record_count")
        if (
            int(
                fields.get(
                    "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows",
                    0,
                )
                or 0
            )
            <= 0
        ):
            missing.append("tensor_native_replay_table_rows")
        if (
            fields.get(
                "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested"
            )
            is True
        ):
            if (
                fields.get(
                    "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used"
                )
                is not True
            ):
                missing.append("tensor_native_direct_prebuilt_path_used")
            if (
                int(
                    fields.get(
                        (
                            "compact_rollout_slab_sample_gate_tensor_native_"
                            "direct_prebuilt_fallback_count"
                        ),
                        0,
                    )
                    or 0
                )
                != 0
            ):
                missing.append("tensor_native_direct_prebuilt_fallback_count")
            if (
                str(
                    fields.get(
                        (
                            "compact_rollout_slab_sample_gate_tensor_native_"
                            "direct_prebuilt_fallback_reason"
                        ),
                        "",
                    )
                )
                != "none"
            ):
                missing.append("tensor_native_direct_prebuilt_fallback_reason")
            if (
                int(
                    fields.get(
                        (
                            "compact_rollout_slab_sample_gate_tensor_native_"
                            "direct_group_object_count"
                        ),
                        -1,
                    )
                    or 0
                )
                != 0
            ):
                missing.append("tensor_native_direct_group_object_count")
    if missing:
        raise ValueError(
            "fused learner-batch speed rows require fused sample/learner proof: "
            + ", ".join(missing)
        )


def _attach_normal_death_contract_fields(
    *,
    args: argparse.Namespace,
    candidate_checkpoint_id: str,
    profile_payload: dict[str, Any],
) -> None:
    death_mode = str(
        profile_payload.get("death_mode")
        or getattr(args, "death_mode", vector_runtime.DEATH_MODE_PROFILE_NO_DEATH)
    )
    if death_mode != vector_runtime.DEATH_MODE_NORMAL:
        profile_payload.setdefault("normal_death_terminal_contract_owner", "none")
        profile_payload.setdefault("normal_death_terminal_contract_source", "none")
        profile_payload.setdefault(
            "normal_death_terminal_contract_trainer_config_matches_runtime",
            False,
        )
        return
    lean_owner = bool(getattr(args, "compact_owned_lean_trainer_step", False)) or (
        str(profile_payload.get("compact_owned_training_loop_owner") or "")
        == "lean_compact_trainer_step"
    )
    trainer_config_death_mode = str(
        profile_payload.get("compact_owned_trainer_config_death_mode")
        or ("" if lean_owner else death_mode)
    )
    if lean_owner and trainer_config_death_mode != vector_runtime.DEATH_MODE_NORMAL:
        raise ValueError("normal-death lean speed row requires trainer config death_mode=normal")
    profile_payload["compact_owned_trainer_config_death_mode"] = trainer_config_death_mode
    profile_payload["normal_death_terminal_contract_owner"] = (
        "compact_owned_trainer_config"
        if trainer_config_death_mode == vector_runtime.DEATH_MODE_NORMAL
        else "measured_profile_payload"
    )
    profile_payload["normal_death_terminal_contract_source"] = "measured_speed_row_payload"
    profile_payload["normal_death_terminal_contract_trainer_config_matches_runtime"] = (
        trainer_config_death_mode == death_mode
    )
    search_metadata = _latest_compact_rollout_slab_search_metadata(profile_payload)
    for owner_key in (
        "compact_owner_search_owner_sample_telemetry",
        "compact_owner_search_owner_learner_telemetry",
    ):
        existing_owner_telemetry = profile_payload.get(owner_key)
        latest_owner_telemetry = search_metadata.get(owner_key)
        if (
            isinstance(latest_owner_telemetry, Mapping)
            and latest_owner_telemetry
            and (not isinstance(existing_owner_telemetry, Mapping) or not existing_owner_telemetry)
        ):
            profile_payload[owner_key] = latest_owner_telemetry
    owner_sample_telemetry = _owner_search_normalized_owner_sample_telemetry_for_proof(
        profile_payload
    )
    if owner_sample_telemetry:
        profile_payload["compact_owner_search_owner_sample_telemetry"] = owner_sample_telemetry
    evidence_refs = (
        str(candidate_checkpoint_id),
        str(getattr(args, "run_id", "")),
    )
    try:
        contract = build_normal_collision_death_contract_from_profile_result_v1(
            profile_payload,
            evidence_id=f"{getattr(args, 'run_id', 'speed-row')}:normal_death_speed_row",
            evidence_refs=evidence_refs,
        )
    except CompactDeathTerminalContractError as exc:
        raise ValueError(
            f"normal-death speed row requires terminal/death contract proof: {exc}"
        ) from exc
    evidence = contract.get("normal_collision_death_evidence")
    profile_payload["normal_death_terminal_contract"] = contract
    profile_payload["normal_death_terminal_contract_schema_id"] = contract.get(
        "compact_death_terminal_contract_schema_id"
    )
    profile_payload["normal_death_terminal_contract_evidence"] = evidence
    profile_payload["normal_death_terminal_contract_evidence_id"] = contract.get(
        "normal_collision_death_evidence_id"
    )
    profile_payload["normal_death_terminal_contract_evidence_refs"] = list(
        contract.get("normal_collision_death_evidence_refs") or []
    )
    profile_payload["normal_death_terminal_contract_promotion_gate_satisfied"] = contract.get(
        "compact_death_terminal_contract_promotion_gate_satisfied"
    )
    if isinstance(evidence, Mapping):
        profile_payload.setdefault(
            "terminal_sample_row_count",
            evidence.get("terminal_sample_row_count"),
        )
        profile_payload.setdefault(
            "terminal_unroll_value_target_row_count",
            evidence.get("terminal_unroll_value_target_row_count"),
        )
        profile_payload.setdefault(
            "terminal_unroll_value_target_mode",
            evidence.get("terminal_unroll_value_target_mode"),
        )


def _normal_death_contract_fields(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = {}
    for key in (
        "compact_owned_trainer_config_death_mode",
        "normal_death_terminal_contract_owner",
        "normal_death_terminal_contract_source",
        "normal_death_terminal_contract_trainer_config_matches_runtime",
        "normal_death_terminal_contract",
        "normal_death_terminal_contract_schema_id",
        "normal_death_terminal_contract_evidence",
        "normal_death_terminal_contract_evidence_id",
        "normal_death_terminal_contract_evidence_refs",
        "normal_death_terminal_contract_promotion_gate_satisfied",
    ):
        if key in profile_payload:
            fields[key] = profile_payload[key]
    return fields


def _require_borrowed_normal_death_terminal_snapshot_proof(
    *,
    args: argparse.Namespace,
    profile_payload: Mapping[str, Any],
    copy_steps: int,
) -> None:
    del copy_steps
    surface = _operational_surface_fields(args, profile_payload)
    contract = profile_payload.get("contract")
    if not isinstance(contract, Mapping):
        contract = {}
    row_overlay_rows = int(
        contract.get("render_state_row_overlay_rows")
        or profile_payload.get("render_state_row_overlay_rows")
        or 0
    )
    if int(surface["terminal_row_count"]) <= 0:
        raise ValueError("borrowed normal-death speed rows require terminal rows")
    if row_overlay_rows <= 0:
        raise ValueError("borrowed normal-death speed rows require terminal row overlays")
    if surface["normal_death_terminal_contract_promotion_gate_satisfied"] is not True:
        raise ValueError("borrowed normal-death speed rows require the normal-death contract gate")
    if surface["terminal_final_observation_before_autoreset_verified"] is not True:
        raise ValueError(
            "borrowed normal-death speed rows must verify final observations before autoreset"
        )
    if int(surface["terminal_sample_row_count"]) <= 0:
        raise ValueError("borrowed normal-death speed rows require terminal samples")
    if int(surface["terminal_unroll_value_target_row_count"]) <= 0:
        raise ValueError("borrowed normal-death speed rows require terminal targets")
    if str(surface["terminal_unroll_value_target_mode"]) != (
        "stock_terminal_no_bootstrap_return_discount_1.0"
    ):
        raise ValueError(
            "borrowed normal-death speed rows require stock no-bootstrap terminal targets"
        )
    if float(surface["resident_observation_host_fallback_count"]) != 0.0:
        raise ValueError("borrowed normal-death speed rows must not use resident host fallback")


def _render_state_handoff_fields(
    args: argparse.Namespace,
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    persistent_requested = bool(
        getattr(args, "hybrid_persistent_compact_render_state_buffer", False)
    )
    borrow_requested = bool(getattr(args, "hybrid_borrow_single_actor_render_state", False))
    if persistent_requested and borrow_requested:
        raise ValueError(
            "hybrid borrow_single_actor_render_state cannot be combined with "
            "persistent_compact_render_state_buffer"
        )
    contract_raw = profile_payload.get("contract")
    if not isinstance(contract_raw, Mapping):
        raise ValueError("profile support contract must be present for render-state handoff")
    contract = dict(contract_raw)
    actual_persistent = bool(contract.get("persistent_compact_render_state_buffer", False))
    actual_borrow = bool(contract.get("borrow_single_actor_render_state", False))
    if actual_persistent is not persistent_requested:
        raise ValueError(
            "profile support persistent_compact_render_state_buffer does not match request"
        )
    if actual_borrow is not borrow_requested:
        raise ValueError("profile support borrow_single_actor_render_state does not match request")
    mode = str(contract.get("render_state_handoff_mode") or "").strip()
    if borrow_requested:
        expected_mode = "borrow_single_actor_env_state"
    elif persistent_requested:
        expected_mode = "persistent_compact_render_state_buffer"
    else:
        expected_mode = "copy_actor_state_to_parent_buffers"
    if mode != expected_mode:
        raise ValueError(
            "profile support render_state_handoff_mode mismatch: "
            f"expected {expected_mode!r}, got {mode!r}"
        )
    copy_steps = int(contract.get("render_state_copy_steps") or 0)
    borrowed_steps = int(contract.get("render_state_borrowed_steps") or 0)
    row_overlay_steps = int(contract.get("render_state_row_overlay_steps") or 0)
    row_overlay_rows = int(contract.get("render_state_row_overlay_rows") or 0)
    row_overlay_bytes = int(contract.get("render_state_row_overlay_bytes") or 0)
    if borrow_requested:
        if int(args.actor_count) != 1:
            raise ValueError("borrowed render-state speed rows require actor_count=1")
        expected_borrowed_steps = int(args.steps) + int(args.warmup_steps)
        if borrowed_steps != expected_borrowed_steps:
            raise ValueError(
                "borrowed render-state speed row did not borrow every step: "
                f"expected {expected_borrowed_steps}, got {borrowed_steps}"
            )
        death_mode = str(
            profile_payload.get("death_mode")
            or getattr(args, "death_mode", vector_runtime.DEATH_MODE_PROFILE_NO_DEATH)
        )
        if death_mode == vector_runtime.DEATH_MODE_PROFILE_NO_DEATH:
            if copy_steps != 0:
                raise ValueError("profile_no_death borrowed speed rows must not copy render state")
            if int(profile_payload.get("terminal_row_count", 0) or 0) != 0:
                raise ValueError(
                    "profile_no_death borrowed speed rows require terminal_row_count=0"
                )
        elif death_mode == vector_runtime.DEATH_MODE_NORMAL:
            _require_borrowed_normal_death_terminal_snapshot_proof(
                args=args,
                profile_payload=profile_payload,
                copy_steps=copy_steps,
            )
        else:
            raise ValueError(
                "borrowed render-state speed rows require death_mode profile_no_death or normal"
            )
        timings = profile_payload.get("timings")
        if isinstance(timings, Mapping):
            fallback_count = float(timings.get("resident_observation_host_fallback_count", 0.0))
            if fallback_count != 0.0:
                raise ValueError("borrowed render-state speed row used resident host fallback")
        totals = profile_payload.get("compact_rollout_slab_telemetry_totals")
        if isinstance(totals, Mapping):
            for key in (
                "compact_rollout_slab_replay_payload_d2h_bytes",
                "compact_rollout_slab_committed_replay_payload_d2h_bytes",
            ):
                if float(totals.get(key, 0.0)) != 0.0:
                    raise ValueError(f"borrowed render-state speed row has nonzero {key}")
    return {
        "hybrid_persistent_compact_render_state_buffer": persistent_requested,
        "hybrid_borrow_single_actor_render_state": borrow_requested,
        "render_state_handoff_mode": mode,
        "render_state_copy_steps": copy_steps,
        "render_state_borrowed_steps": borrowed_steps,
        "render_state_row_overlay_steps": row_overlay_steps,
        "render_state_row_overlay_rows": row_overlay_rows,
        "render_state_row_overlay_bytes": row_overlay_bytes,
    }


def _run_local_compact_owned_profile(
    *,
    args: argparse.Namespace,
    loaded_model: Any | None = None,
    learner_model: Any | None = None,
    search_model: Any | None = None,
    loaded_checkpoint_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    learner_device = _resident_renderer_device(str(args.learner_device))
    if learner_model is None:
        learner_model = loaded_model if loaded_model is not None else _TinyMuZero()
    if search_model is None:
        search_model = copy.deepcopy(learner_model)
    search_service = _build_search_service(
        args=args,
        model=search_model,
        device=learner_device,
        loaded_checkpoint_identity=loaded_checkpoint_identity or {},
    )
    policy_version_ref = f"{args.run_id}:policy"
    model_version_ref = f"{args.run_id}:model"
    policy_source = "compact_coach_speed_row_smoke"
    if loaded_checkpoint_identity:
        policy_version_ref = str(
            loaded_checkpoint_identity.get("policy_version_ref") or policy_version_ref
        )
        model_version_ref = str(
            loaded_checkpoint_identity.get("model_version_ref") or model_version_ref
        )
        policy_source = str(loaded_checkpoint_identity.get("policy_source") or policy_source)
    owner_search_train = _owner_search_owner_train_enabled(args)
    owner_search_slab_bypass = bool(
        owner_search_train and getattr(args, "owner_search_slab_bypass", False)
    )
    if owner_search_slab_bypass:
        slab = CompactOwnerSearchDirectStepperV1(
            batch_size=int(args.batch_size),
            player_count=2,
            search_service=search_service,
            search_lane="compact_coach_speed_row_smoke:owner_search_slab_bypass",
            policy_source=policy_source,
            copy_root_observation=False,
            transition_batch_size=int(getattr(args, "owner_search_transition_batch_size", 1)),
            resident_root_host_observation_stub=bool(
                getattr(
                    args,
                    "owner_search_resident_root_host_observation_stub",
                    False,
                )
            ),
            direct_root_build_request=bool(
                getattr(args, "owner_search_direct_root_build_request", False)
            ),
            owner_local_transition_derivation=bool(
                getattr(args, "owner_search_owner_local_transition_derivation", False)
            ),
            owner_proxy_transition_closure=bool(
                getattr(args, "owner_search_owner_proxy_transition_closure", False)
            ),
        )
    else:
        slab = CompactRolloutSlab(
            batch_size=int(args.batch_size),
            player_count=2,
            search_service=search_service,
            search_lane="compact_coach_speed_row_smoke",
            policy_source=policy_source,
            copy_root_observation=False,
            retain_committed_index_rows=not bool(
                getattr(args, "compact_profile_bounded_diagnostics", False)
            ),
        )
    parent_sample_learner_enabled = not owner_search_train
    parent_fused_learner_batch = bool(
        parent_sample_learner_enabled
        and getattr(args, "compact_owned_loop_fused_learner_batch", False)
    )
    parent_unroll2_specialized_builder = bool(
        parent_sample_learner_enabled
        and getattr(args, "compact_muzero_learner_batch_unroll2_specialized_builder", False)
    )
    parent_learner_ready_unroll2_cache = bool(
        parent_sample_learner_enabled
        and getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
    )
    parent_tensor_native_replay = bool(
        parent_sample_learner_enabled
        and getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
    )
    try:
        payload = run_hybrid_observation_profile(
            HybridObservationProfileConfig(
                batch_size=int(args.batch_size),
                actor_count=int(args.actor_count),
                steps=int(args.steps),
                warmup_steps=int(args.warmup_steps),
                seed=int(args.seed),
                death_mode=str(args.death_mode),
                stack_storage_dtype=HYBRID_STACK_STORAGE_DTYPE_UINT8,
                update_host_observation_stack=False,
                resident_observation_search=True,
                resident_replay_snapshot_mode=(
                    RESIDENT_REPLAY_SNAPSHOT_MODE_LATEST_FRAME_HISTORY
                    if bool(getattr(args, "compact_profile_bounded_diagnostics", False))
                    else RESIDENT_REPLAY_SNAPSHOT_MODE_FULL_STACK
                ),
                materialize_scalar_timestep=False,
                native_actor_buffer=True,
                borrow_single_actor_render_state=bool(args.hybrid_borrow_single_actor_render_state),
                persistent_compact_render_state_buffer=bool(
                    args.hybrid_persistent_compact_render_state_buffer
                ),
                compact_rollout_slab_sample_gate=parent_sample_learner_enabled,
                compact_rollout_slab_sample_gate_batch_size=int(args.sample_batch_size),
                compact_rollout_slab_sample_gate_interval=int(args.sample_interval),
                compact_rollout_slab_sample_gate_replay_pair_capacity=int(
                    args.replay_pair_capacity
                ),
                compact_rollout_slab_learner_gate=parent_sample_learner_enabled,
                compact_rollout_slab_learner_gate_impl=(
                    COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO
                ),
                compact_rollout_slab_learner_gate_device=str(args.learner_device),
                compact_rollout_slab_learner_gate_train_steps=int(args.learner_train_steps),
                compact_rollout_slab_learner_gate_support_scale=int(
                    (loaded_checkpoint_identity or {}).get("support_scale") or 1
                ),
                compact_rollout_slab_learner_gate_num_unroll_steps=_learner_num_unroll_steps(args),
                compact_owned_loop_entrypoint=parent_sample_learner_enabled,
                compact_owned_loop_policy_version_ref=policy_version_ref,
                compact_owned_loop_model_version_ref=model_version_ref,
                compact_owned_loop_policy_source=policy_source,
                compact_owned_loop_capture_replay_store_state=parent_sample_learner_enabled,
                compact_owned_loop_defer_learner_gate=bool(
                    args.compact_owned_loop_deferred_learner
                ),
                compact_owned_loop_defer_sample_learner_gate=bool(
                    args.compact_owned_loop_deferred_sample_learner
                ),
                compact_owned_loop_defer_sample_learner_gate_max_pending=int(
                    args.compact_owned_loop_deferred_sample_learner_max_pending
                ),
                compact_owned_loop_defer_sample_learner_model_state_interval=int(
                    args.policy_refresh_interval
                ),
                compact_owned_loop_defer_sample_learner_model_state_transport_kind=str(
                    getattr(
                        args,
                        ("compact_owned_loop_deferred_sample_learner_model_state_transport_kind"),
                        COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
                    )
                ),
                compact_owned_loop_defer_sample_learner_replay_append_transport_kind=str(
                    getattr(
                        args,
                        "compact_owned_loop_deferred_sample_learner_replay_append_transport_kind",
                        COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1,
                    )
                ),
                compact_owned_loop_sample_learner_worker_kind=str(
                    args.compact_owned_loop_sample_learner_worker_kind
                ),
                compact_owned_loop_fused_learner_batch=parent_fused_learner_batch,
                compact_muzero_learner_batch_unroll2_specialized_builder=(
                    parent_unroll2_specialized_builder
                ),
                compact_muzero_learner_batch_learner_ready_unroll2_cache=(
                    parent_learner_ready_unroll2_cache
                ),
                compact_muzero_learner_batch_tensor_native_replay=(parent_tensor_native_replay),
                compact_owner_action_step_boundary=bool(
                    getattr(args, "compact_owner_action_step_boundary", False)
                ),
                compact_owner_action_dispatch_step_overlap=bool(
                    getattr(args, "compact_owner_action_dispatch_step_overlap", False)
                ),
                compact_profile_cuda_sync_timing_diagnostics=bool(
                    getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
                ),
                compact_profile_runtime_step_timing_diagnostics=bool(
                    getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
                ),
            ),
            observation_renderer=_PersistentDeviceRenderer(device=learner_device),
            compact_rollout_slab=slab,
            compact_rollout_slab_learner_model=learner_model,
            compact_rollout_slab_refresh_search_after_learner_gate=(
                str(args.search_service_kind) == SEARCH_SERVICE_COMPACT_TORCH
            ),
            compact_rollout_slab_refresh_search_after_learner_gate_interval=int(
                args.policy_refresh_interval
            ),
        )
        return _surface_compact_owned_loop_telemetry(payload)
    finally:
        close = getattr(search_service, "close", None)
        if callable(close):
            close()


def _run_local_compact_owned_lean_trainer_profile(
    *,
    args: argparse.Namespace,
    loaded_model: Any | None = None,
    learner_model: Any | None = None,
    search_model: Any | None = None,
    loaded_checkpoint_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    learner_device = _resident_renderer_device(str(args.learner_device))
    if learner_model is None:
        learner_model = loaded_model if loaded_model is not None else _TinyMuZero()
    if search_model is None:
        search_model = copy.deepcopy(learner_model)
    loaded_identity = dict(loaded_checkpoint_identity or {})
    search_service = _build_search_service(
        args=args,
        model=search_model,
        device=learner_device,
        loaded_checkpoint_identity=loaded_identity,
    )
    policy_version_ref = f"{args.run_id}:policy"
    model_version_ref = f"{args.run_id}:model"
    policy_source = "compact_coach_speed_row_smoke"
    if loaded_identity:
        policy_version_ref = str(loaded_identity.get("policy_version_ref") or policy_version_ref)
        model_version_ref = str(loaded_identity.get("model_version_ref") or model_version_ref)
        policy_source = str(loaded_identity.get("policy_source") or policy_source)
    slab = CompactRolloutSlab(
        batch_size=int(args.batch_size),
        player_count=2,
        search_service=search_service,
        search_lane="compact_coach_speed_row_smoke:lean_trainer_step",
        policy_source=policy_source,
        copy_root_observation=False,
        retain_committed_index_rows=not bool(
            getattr(args, "compact_profile_bounded_diagnostics", False)
        ),
    )

    trainer_box: dict[str, CompactOwnedTrainerV1] = {}

    def make_trainer_owner(*, compact_owned_loop, learner, policy_version):
        if learner is None:
            raise ValueError("lean trainer owner requires the compact learner probe")
        if str(policy_version.policy_version_ref) != str(policy_version_ref):
            raise ValueError("lean trainer policy version mismatch")
        trainer = CompactOwnedTrainerV1(
            config=CompactOwnedTrainerConfigV1(
                trainer_id=f"{args.run_id}:lean-compact-trainer",
                policy_source=policy_source,
                initial_policy_version_ref=policy_version_ref,
                initial_model_version_ref=model_version_ref,
                death_mode=str(args.death_mode),
                allow_pending_normal_death_contract=(
                    str(args.death_mode) == vector_runtime.DEATH_MODE_NORMAL
                ),
            ),
            learner=learner,
            loop=compact_owned_loop,
        )
        trainer_box["trainer"] = trainer
        return trainer

    payload = run_hybrid_observation_profile(
        HybridObservationProfileConfig(
            batch_size=int(args.batch_size),
            actor_count=int(args.actor_count),
            steps=int(args.steps),
            warmup_steps=int(args.warmup_steps),
            seed=int(args.seed),
            death_mode=str(args.death_mode),
            stack_storage_dtype=HYBRID_STACK_STORAGE_DTYPE_UINT8,
            update_host_observation_stack=False,
            resident_observation_search=True,
            resident_replay_snapshot_mode=(
                RESIDENT_REPLAY_SNAPSHOT_MODE_LATEST_FRAME_HISTORY
                if bool(getattr(args, "compact_profile_bounded_diagnostics", False))
                else RESIDENT_REPLAY_SNAPSHOT_MODE_FULL_STACK
            ),
            materialize_scalar_timestep=False,
            native_actor_buffer=True,
            borrow_single_actor_render_state=bool(args.hybrid_borrow_single_actor_render_state),
            persistent_compact_render_state_buffer=bool(
                args.hybrid_persistent_compact_render_state_buffer
            ),
            compact_rollout_slab_sample_gate=True,
            compact_rollout_slab_sample_gate_batch_size=int(args.sample_batch_size),
            compact_rollout_slab_sample_gate_interval=int(args.sample_interval),
            compact_rollout_slab_sample_gate_replay_pair_capacity=int(args.replay_pair_capacity),
            compact_rollout_slab_learner_gate=True,
            compact_rollout_slab_learner_gate_impl=(
                COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO
            ),
            compact_rollout_slab_learner_gate_device=str(args.learner_device),
            compact_rollout_slab_learner_gate_train_steps=int(args.learner_train_steps),
            compact_rollout_slab_learner_gate_support_scale=int(
                loaded_identity.get("support_scale") or 1
            ),
            compact_rollout_slab_learner_gate_num_unroll_steps=_learner_num_unroll_steps(args),
            compact_owned_loop_entrypoint=True,
            compact_owned_loop_policy_version_ref=policy_version_ref,
            compact_owned_loop_model_version_ref=model_version_ref,
            compact_owned_loop_policy_source=policy_source,
            compact_owned_loop_capture_replay_store_state=True,
            compact_owned_loop_defer_learner_gate=bool(args.compact_owned_loop_deferred_learner),
            compact_owned_loop_defer_sample_learner_gate=bool(
                args.compact_owned_loop_deferred_sample_learner
            ),
            compact_owned_loop_defer_sample_learner_gate_max_pending=int(
                args.compact_owned_loop_deferred_sample_learner_max_pending
            ),
            compact_owned_loop_defer_sample_learner_model_state_interval=int(
                args.policy_refresh_interval
            ),
            compact_owned_loop_defer_sample_learner_model_state_transport_kind=str(
                getattr(
                    args,
                    ("compact_owned_loop_deferred_sample_learner_model_state_transport_kind"),
                    COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
                )
            ),
            compact_owned_loop_defer_sample_learner_replay_append_transport_kind=str(
                getattr(
                    args,
                    "compact_owned_loop_deferred_sample_learner_replay_append_transport_kind",
                    COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1,
                )
            ),
            compact_owned_loop_sample_learner_worker_kind=str(
                args.compact_owned_loop_sample_learner_worker_kind
            ),
            compact_owned_loop_fused_learner_batch=bool(
                args.compact_owned_loop_fused_learner_batch
            ),
            compact_muzero_learner_batch_unroll2_specialized_builder=bool(
                args.compact_muzero_learner_batch_unroll2_specialized_builder
            ),
            compact_muzero_learner_batch_learner_ready_unroll2_cache=bool(
                getattr(
                    args,
                    "compact_muzero_learner_batch_learner_ready_unroll2_cache",
                    False,
                )
            ),
            compact_muzero_learner_batch_tensor_native_replay=bool(
                getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
            ),
            compact_profile_cuda_sync_timing_diagnostics=bool(
                getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
            ),
            compact_profile_runtime_step_timing_diagnostics=bool(
                getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
            ),
        ),
        observation_renderer=_PersistentDeviceRenderer(device=learner_device),
        compact_rollout_slab=slab,
        compact_rollout_slab_learner_model=learner_model,
        compact_rollout_slab_refresh_search_after_learner_gate=(
            str(args.search_service_kind) == SEARCH_SERVICE_COMPACT_TORCH
        ),
        compact_rollout_slab_refresh_search_after_learner_gate_interval=int(
            args.policy_refresh_interval
        ),
        compact_owned_record_step_owner_factory=make_trainer_owner,
        compact_owned_loop_metadata_extra={
            "compact_owned_lean_trainer_step": True,
            "compact_owned_training_loop_owner": "lean_compact_trainer_step",
        },
    )
    _surface_compact_owned_loop_telemetry(payload)
    trainer = trainer_box.get("trainer")
    if trainer is None:
        raise RuntimeError("lean compact trainer owner was not constructed")
    loop_telemetry = dict(payload.get("compact_owned_loop_telemetry") or {})
    payload.update(
        {
            "compact_owned_lean_trainer_step": True,
            "compact_owned_training_loop_owner": "lean_compact_trainer_step",
            "compact_owned_trainer_record_step_calls": int(trainer.record_step_calls),
            "compact_owned_trainer_learner_update_count": int(trainer.learner_update_count),
            "compact_owned_trainer_sample_batch_count": int(trainer.sample_batch_count),
            "compact_owned_trainer_policy_refresh_count": int(trainer.policy_refresh_count),
            "compact_owned_trainer_policy_version_ref": str(trainer.policy_version_ref),
            "compact_owned_trainer_model_version_ref": str(trainer.model_version_ref),
            "compact_owned_trainer_config_death_mode": str(trainer.config.death_mode),
            "compact_owned_trainer_loop_counter_source": "run_hybrid_observation_profile",
        }
    )
    if int(payload.get("compact_rollout_slab_learner_gate_updates") or 0) != int(
        trainer.learner_update_count
    ):
        raise ValueError("lean trainer learner-update counter drift")
    if int(loop_telemetry.get("compact_owned_loop_sample_gate_calls") or 0) != int(
        trainer.sample_batch_count
    ):
        raise ValueError("lean trainer sample counter drift")
    return payload


class _LeanPolicyRefreshState:
    def __init__(self, *, enabled: bool, interval: int) -> None:
        if int(interval) <= 0:
            raise ValueError("policy refresh interval must be positive")
        self.enabled = bool(enabled)
        self.interval = int(interval)
        self.calls = 0
        self.skipped_count = 0
        self.forced_final_count = 0
        self.sec = 0.0
        self.last_state: dict[str, Any] = {}
        self.last_update_count = 0
        self.last_digest = ""
        self.distinct = False
        self.search_metadata_count = 0
        self.replay_metadata_count = 0
        self.sample_metadata_count = 0
        self.service_total_sec = 0.0
        self.state_load_sec = 0.0
        self.model_digest_sec = 0.0
        self.last_service_total_sec = 0.0
        self.last_state_load_sec = 0.0
        self.last_model_digest_sec = 0.0
        self.last_search_metadata: dict[str, Any] = {}
        self.last_replay_metadata: dict[str, Any] = {}
        self.last_sample_metadata: dict[str, Any] = {}
        self._seen_search_metadata: set[tuple[Any, ...]] = set()
        self._seen_replay_metadata: set[tuple[Any, ...]] = set()
        self._seen_sample_metadata: set[tuple[Any, ...]] = set()


def _sum_numeric_fields(
    target: dict[str, float],
    source: Mapping[str, Any],
    *,
    keys: tuple[str, ...] | None = None,
) -> None:
    field_names = keys if keys is not None else tuple(str(key) for key in source.keys())
    for key in field_names:
        value = source.get(key, 0.0)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if math.isfinite(number):
                target[str(key)] = float(target.get(str(key), 0.0)) + number


def _update_lean_terminal_counters(
    *,
    step: Any,
    death_cause_count_by_name: dict[str, int],
    normal_collision_death_evidence_rows: list[dict[str, Any]],
    counters: Mapping[str, int],
    state: Mapping[str, bool],
) -> dict[str, Any]:
    updated: dict[str, Any] = {str(key): int(value) for key, value in counters.items()}
    done_array = np.asarray(step.done, dtype=np.bool_)
    terminated_array = np.asarray(step.terminated, dtype=np.bool_)
    truncated_array = np.asarray(step.truncated, dtype=np.bool_)
    death_count_array = np.asarray(step.death_count, dtype=np.int32)
    death_cause_array = np.asarray(step.death_cause, dtype=np.int16)
    death_hit_owner_array = np.asarray(step.death_hit_owner, dtype=np.int16)
    death_player_array = np.asarray(step.death_player, dtype=np.int16)
    winner_array = np.asarray(step.winner, dtype=np.int16)
    draw_array = np.asarray(step.draw, dtype=np.bool_)
    terminal_reason_array = np.asarray(step.terminal_reason, dtype=np.int16)
    final_observation_row_mask = (
        np.asarray(step.compact_batch.final_observation_row_mask, dtype=np.bool_)
        if step.compact_batch is not None
        else np.zeros_like(done_array, dtype=np.bool_)
    )
    updated["done_semantics_verified"] = bool(
        state.get("done_semantics_verified", True)
        and np.array_equal(done_array, np.logical_or(terminated_array, truncated_array))
    )
    updated["done_rows"] += int(done_array.sum())
    updated["terminal_rows"] += int(np.asarray(step.payload.get("terminal_global_rows", ())).size)
    updated["autoreset_rows"] += int(np.asarray(step.payload.get("autoreset_global_rows", ())).size)
    updated["terminated_rows"] += int(terminated_array.sum())
    updated["truncated_rows"] += int(truncated_array.sum())
    updated["death_rows"] += int((death_count_array > 0).sum())
    updated["death_count_total"] += int(death_count_array.sum())
    for cause_code, cause_name in enumerate(vector_runtime.DEATH_CAUSE_NAMES):
        if cause_code == vector_runtime.DEATH_CAUSE_NONE:
            continue
        death_cause_count_by_name[str(cause_name)] += int((death_cause_array == cause_code).sum())
    hit_owner_present = bool(
        np.any(
            (death_cause_array != vector_runtime.DEATH_CAUSE_NONE) & (death_hit_owner_array >= 0)
        )
    )
    updated["normal_collision_death_hit_owner_present"] = bool(
        state.get("normal_collision_death_hit_owner_present", False) or hit_owner_present
    )
    final_reward_array = np.asarray(step.final_reward_map, dtype=np.float32)
    reward_array = np.asarray(step.reward, dtype=np.float32)
    if bool(done_array.any()):
        updated["terminal_final_reward_map_row_count"] += int(done_array.sum())
        if np.allclose(
            final_reward_array[done_array],
            reward_array[done_array],
            atol=1.0e-6,
        ):
            updated["terminal_final_reward_map_matches_reward_row_count"] += int(done_array.sum())
    if len(normal_collision_death_evidence_rows) < 8:
        global_rows = np.asarray(
            step.payload.get("global_rows", np.arange(done_array.shape[0])),
            dtype=np.int64,
        )
        for row_index in np.flatnonzero(death_count_array > 0):
            cause_names = [
                str(vector_runtime.DEATH_CAUSE_NAMES[int(cause)])
                for cause in death_cause_array[int(row_index)].reshape(-1)
                if 0 <= int(cause) < len(vector_runtime.DEATH_CAUSE_NAMES)
                and int(cause) != vector_runtime.DEATH_CAUSE_NONE
            ]
            if not any(cause in {"opponent_trail", "wall"} for cause in cause_names):
                continue
            normal_collision_death_evidence_rows.append(
                {
                    "global_row": int(global_rows[int(row_index)]),
                    "done": bool(done_array[int(row_index)]),
                    "terminated": bool(terminated_array[int(row_index)]),
                    "truncated": bool(truncated_array[int(row_index)]),
                    "terminal_reason": int(terminal_reason_array[int(row_index)]),
                    "death_count": int(death_count_array[int(row_index)]),
                    "death_player": [
                        int(value) for value in death_player_array[int(row_index)].reshape(-1)
                    ],
                    "death_cause": cause_names,
                    "death_hit_owner": [
                        int(value) for value in death_hit_owner_array[int(row_index)].reshape(-1)
                    ],
                    "winner": int(winner_array[int(row_index)]),
                    "draw": bool(draw_array[int(row_index)]),
                    "reward": [float(value) for value in reward_array[int(row_index)].reshape(-1)],
                    "final_reward_map": [
                        float(value) for value in final_reward_array[int(row_index)].reshape(-1)
                    ],
                    "final_reward_map_matches_reward": bool(
                        np.allclose(
                            final_reward_array[int(row_index)],
                            reward_array[int(row_index)],
                            atol=1.0e-6,
                        )
                    ),
                    "final_observation_row": bool(final_observation_row_mask[int(row_index)]),
                }
            )
            if len(normal_collision_death_evidence_rows) >= 8:
                break
    updated["terminal_final_observation_row_count"] += int(final_observation_row_mask.sum())
    updated["compact_profile_autoreset_direct_count"] += _payload_int(
        step.payload,
        "compact_profile_autoreset_direct_count",
    )
    updated["compact_profile_autoreset_template_copy_skipped_count"] += _payload_int(
        step.payload,
        "compact_profile_autoreset_template_copy_skipped_count",
    )
    updated["compact_profile_autoreset_direct_row_count"] += _payload_int(
        step.payload,
        "compact_profile_autoreset_direct_row_count",
    )
    return updated


def _payload_int(payload: Mapping[str, Any], key: str) -> int:
    return int(np.asarray(payload.get(key, 0), dtype=np.int64).reshape(()).item())


def _update_lean_refresh_metadata_seen(
    refresh: _LeanPolicyRefreshState,
    *,
    slab_step: Any,
    loop: CompactOwnedLoopV1,
) -> None:
    search_metadata = dict(
        getattr(slab_step, "telemetry", {}).get("compact_rollout_slab_search_metadata", {}) or {}
    )
    _maybe_record_refresh_metadata(
        refresh,
        metadata=search_metadata,
        label="search",
        identity=("search", int(getattr(slab_step, "record_index", -1))),
    )
    committed = getattr(slab_step, "committed_index_rows", None)
    replay_metadata = dict(getattr(committed, "metadata", {}) or {})
    _maybe_record_refresh_metadata(
        refresh,
        metadata=replay_metadata,
        label="replay",
        identity=("replay", int(getattr(committed, "record_index", -1))),
    )
    sample_metadata = dict(getattr(loop, "sample_gate_last_sample_metadata", {}) or {})
    _maybe_record_refresh_metadata(
        refresh,
        metadata=sample_metadata,
        label="sample",
        identity=(
            "sample",
            int(getattr(loop, "sample_gate_calls", 0)),
            int(getattr(loop, "sample_gate_sample_rows", 0)),
        ),
    )


def _maybe_record_refresh_metadata(
    refresh: _LeanPolicyRefreshState,
    *,
    metadata: Mapping[str, Any],
    label: str,
    identity: tuple[Any, ...],
) -> None:
    if not _compact_policy_refresh_metadata_seen(metadata):
        return
    signature = (
        *identity,
        int(metadata.get("compact_policy_refresh_learner_update_count") or 0),
        str(metadata.get("compact_policy_refresh_model_state_digest") or ""),
    )
    seen = getattr(refresh, f"_seen_{label}_metadata")
    if signature in seen:
        return
    seen.add(signature)
    if label == "search":
        refresh.search_metadata_count += 1
        refresh.last_search_metadata = dict(metadata)
    elif label == "replay":
        refresh.replay_metadata_count += 1
        refresh.last_replay_metadata = dict(metadata)
    elif label == "sample":
        refresh.sample_metadata_count += 1
        refresh.last_sample_metadata = dict(metadata)
    else:
        raise ValueError(f"unknown refresh metadata label {label!r}")


def _maybe_refresh_lean_search_after_learner_update(
    *,
    args: argparse.Namespace,
    refresh: _LeanPolicyRefreshState,
    slab: CompactRolloutSlab,
    learner_probe: Any,
    trainer: CompactOwnedTrainerV1,
    loop: CompactOwnedLoopV1,
    replay_store: _CompactReplayRingV1,
    policy_source: str,
    total_iterations: int,
    iteration: int,
    timings: dict[str, float],
) -> None:
    if not refresh.enabled:
        return
    update_count = int(trainer.learner_update_count)
    if update_count <= 0:
        return
    if update_count == refresh.last_update_count:
        return
    remaining_iterations = max(0, int(total_iterations) - int(iteration) - 1)
    final_consumable_refresh = remaining_iterations > 0 and remaining_iterations <= int(
        args.sample_interval
    )
    if update_count % int(refresh.interval) != 0 and not final_consumable_refresh:
        refresh.skipped_count += 1
        return
    if update_count % int(refresh.interval) != 0 and final_consumable_refresh:
        refresh.forced_final_count += 1
    started = time.perf_counter()
    refreshed = _refresh_compact_rollout_slab_search_from_learner(
        compact_rollout_slab=slab,
        learner_probe=learner_probe,
        policy_version_ref=str(trainer.policy_version_ref),
        model_version_ref=str(trainer.model_version_ref),
        policy_source=policy_source,
        learner_update_count=update_count,
    )
    refresh_sec = max(0.0, time.perf_counter() - started)
    refresh.calls += 1
    refresh.sec += refresh_sec
    timings["compact_rollout_slab_policy_refresh_after_learner_gate_sec"] += refresh_sec
    refresh.last_state = dict(refreshed["search_worker_state"])
    refresh.last_update_count = int(refreshed["learner_update_count"])
    refresh.last_digest = str(refreshed["model_state_digest"])
    refresh.distinct = bool(refreshed["search_worker_distinct_from_learner"])
    service_total_sec = float(refresh.last_state.get("refresh_total_sec") or 0.0)
    state_load_sec = float(refresh.last_state.get("state_load_sec") or 0.0)
    model_digest_sec = float(refresh.last_state.get("model_state_digest_sec") or 0.0)
    refresh.service_total_sec += service_total_sec
    refresh.state_load_sec += state_load_sec
    refresh.model_digest_sec += model_digest_sec
    refresh.last_service_total_sec = service_total_sec
    refresh.last_state_load_sec = state_load_sec
    refresh.last_model_digest_sec = model_digest_sec
    policy_version = CompactPolicyVersionRefV1(
        policy_version_ref=str(trainer.policy_version_ref),
        policy_source=policy_source,
        model_version_ref=str(trainer.model_version_ref),
    )
    loop.update_policy_version(policy_version)
    replay_store.update_store_metadata(
        compact_owned_loop_replay_store_metadata(
            policy_version,
            extra={
                "compact_owned_lean_trainer_step": True,
                "compact_owned_training_loop_owner": "lean_compact_trainer_step",
            },
        )
    )


def _lean_policy_refresh_fields(refresh: _LeanPolicyRefreshState) -> dict[str, Any]:
    return {
        "compact_rollout_slab_policy_refresh_after_learner_gate_enabled": refresh.enabled,
        "compact_rollout_slab_policy_refresh_after_learner_gate_calls": int(refresh.calls),
        "compact_rollout_slab_policy_refresh_after_learner_gate_interval": int(refresh.interval),
        "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count": int(
            refresh.skipped_count
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count": int(
            refresh.forced_final_count
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_sec": float(refresh.sec),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count": int(
            refresh.last_update_count
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest": str(
            refresh.last_digest
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_search_worker_distinct_from_learner": bool(
            refresh.distinct
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_worker_state": dict(
            refresh.last_state
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_service_total_sec": float(
            refresh.service_total_sec
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_state_load_sec": float(
            refresh.state_load_sec
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_model_digest_sec": float(
            refresh.model_digest_sec
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_service_total_sec": float(
            refresh.last_service_total_sec
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_state_load_sec": float(
            refresh.last_state_load_sec
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_digest_sec": float(
            refresh.last_model_digest_sec
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_count": int(
            refresh.search_metadata_count
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_count": int(
            refresh.replay_metadata_count
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_sample_metadata_count": int(
            refresh.sample_metadata_count
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata": dict(
            refresh.last_search_metadata
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata": dict(
            refresh.last_replay_metadata
        ),
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_metadata": dict(
            refresh.last_sample_metadata
        ),
    }


def _build_search_service(
    *,
    args: argparse.Namespace,
    model: Any | None,
    device: str,
    loaded_checkpoint_identity: Mapping[str, Any],
) -> Any:
    kind = str(args.search_service_kind)
    if kind == SEARCH_SERVICE_DEVICE_TARGET:
        return _DeviceTargetSearchService(
            seed=int(args.seed),
            model=model,
            num_simulations=int(args.num_simulations),
            loaded_checkpoint_identity=loaded_checkpoint_identity,
        )
    if kind == SEARCH_SERVICE_FIXED_SHAPE:
        return FixedShapeBatchedSearchOwnerV0(
            root_count=int(args.batch_size) * 2,
            num_simulations=int(args.num_simulations),
            device=device,
            model=model if model is not None else _TinyMuZero(),
        )
    if kind == SEARCH_SERVICE_COMPACT_TORCH:
        import torch

        torch.manual_seed(int(args.seed))
        target_device = torch.device(device)
        search_model = model if model is not None else _TinyMuZero()
        model_to = getattr(search_model, "to", None)
        if callable(model_to):
            search_model = model_to(target_device)
        service = CompactTorchSearchServiceV1(
            policy=_ModelPolicy(search_model),
            num_simulations=int(args.num_simulations),
            device=target_device,
            root_noise_weight=0.0,
            compile_config=CompactTorchCompileConfig(
                request_compile=bool(args.compact_torch_request_compile),
                request_model_compile=bool(args.compact_torch_request_model_compile),
                model_compile_mode=str(
                    getattr(args, "compact_torch_model_compile_mode", "reduce-overhead")
                ),
                require_cuda_device=False,
                require_torch_compile=False,
                require_model_compile=False,
                require_all_roots_active=False,
                require_all_actions_legal=False,
                recurrent_action_shape_mode="auto",
                timing_mode=str(args.compact_torch_timing_mode),
                initial_inference_mode=str(args.compact_torch_initial_inference_mode),
                observation_memory_format=str(
                    getattr(
                        args,
                        "compact_torch_observation_memory_format",
                        "contiguous",
                    )
                ),
                model_memory_format=str(
                    getattr(args, "compact_torch_model_memory_format", "contiguous")
                ),
                defer_one_simulation_replay_payload=bool(
                    getattr(
                        args,
                        "compact_torch_defer_one_simulation_replay_payload",
                        False,
                    )
                ),
            ),
            require_resident_observation=True,
        )
        if loaded_checkpoint_identity:
            service.refresh_model_state(
                model_state_dict=search_model.state_dict(),
                policy_version_ref=str(
                    loaded_checkpoint_identity.get("policy_version_ref") or f"{args.run_id}:policy"
                ),
                model_version_ref=str(
                    loaded_checkpoint_identity.get("model_version_ref") or f"{args.run_id}:model"
                ),
                policy_source=str(
                    loaded_checkpoint_identity.get("policy_source")
                    or "compact_coach_speed_row_smoke"
                ),
                learner_update_count=max(
                    1,
                    int(loaded_checkpoint_identity.get("learner_update_count") or 1),
                ),
                expected_model_state_digest=compact_model_state_digest_v1(search_model),
            )
        return service
    if kind == SEARCH_SERVICE_OWNER_SEARCH_SLAB_PROXY:
        return _build_owner_search_slab_proxy(
            args=args,
            model=model,
            device=device,
            loaded_checkpoint_identity=loaded_checkpoint_identity,
        )
    if kind == SEARCH_SERVICE_OWNER_SEARCH_INLINE_PROXY:
        return _build_owner_search_slab_proxy(
            args=args,
            model=model,
            device=device,
            loaded_checkpoint_identity=loaded_checkpoint_identity,
            inline=True,
        )
    if kind == SEARCH_SERVICE_OWNER_SEARCH_INLINE_BACKGROUND_PROXY:
        return _build_owner_search_slab_proxy(
            args=args,
            model=model,
            device=device,
            loaded_checkpoint_identity=loaded_checkpoint_identity,
            inline_background=True,
        )
    if kind == SEARCH_SERVICE_OWNER_SEARCH_THREADED_PROXY:
        return _build_owner_search_slab_proxy(
            args=args,
            model=model,
            device=device,
            loaded_checkpoint_identity=loaded_checkpoint_identity,
            threaded=True,
        )
    raise ValueError(f"unknown search_service_kind {kind!r}")


class _OwnerSearchReplayStoreFactorySidecarV1:
    """Owner-worker replay store that appends real compact replay entries."""

    def __init__(
        self,
        *,
        capacity: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._ring = _CompactReplayRingV1(
            capacity=int(capacity),
            metadata=dict(metadata or {}),
        )
        self.append_count = 0

    @property
    def entry_count(self) -> int:
        return int(self._ring.entry_count)

    @property
    def stored_index_row_count(self) -> int:
        return int(self._ring.stored_index_row_count)

    @property
    def evicted_entry_count(self) -> int:
        return int(self._ring.evicted_entry_count)

    @property
    def evicted_index_row_count(self) -> int:
        return int(self._ring.evicted_index_row_count)

    def append_owner_search_replay(
        self,
        *,
        replay_append_entries: tuple[Any, ...],
        root_batch: Any,
        search_result: Any,
        request: Any,
        root_batch_cache: Mapping[int, Any] | None = None,
    ) -> int:
        del search_result
        cache = {int(key): value for key, value in dict(root_batch_cache or {}).items()}
        cache[int(getattr(request, "actor_step"))] = root_batch
        entries: list[_CompactReplayRingEntry] = []
        for entry in tuple(replay_append_entries):
            if isinstance(entry, CompactOwnerSearchReplayAppendEntryV1):
                raise RuntimeError(
                    "owner-search replay append must be index-only; full compact "
                    "batch entries would put previous/current observations back "
                    "in CompactOwnerSearchRequestV1"
                )
            if isinstance(entry, CompactOwnerSearchReplayAppendIndexEntryV1):
                previous_root = cache.get(int(entry.record_index))
                current_root = cache.get(int(entry.next_record_index))
                if previous_root is None or current_root is None:
                    raise RuntimeError(
                        "owner-search index-only replay append requires cached "
                        "previous and current root batches"
                    )
                previous_step = self._step_from_root_batch(
                    previous_root,
                    entry.index_rows,
                    role="previous",
                )
                current_step = self._step_from_root_batch(
                    current_root,
                    entry.index_rows,
                    role="current",
                )
            else:
                raise RuntimeError(
                    "owner-search replay append requires "
                    "CompactOwnerSearchReplayAppendEntryV1 or "
                    "CompactOwnerSearchReplayAppendIndexEntryV1"
                )
            entries.append(
                _CompactReplayRingEntry(
                    previous_step=previous_step,
                    current_step=current_step,
                    index_rows=entry.index_rows,
                )
            )
        append_entries = getattr(self._ring, "append_entries", None)
        if callable(append_entries):
            appended = int(append_entries(tuple(entries)))
        else:
            appended = 0
            for entry in entries:
                self._ring.append_entry(entry)
                appended += 1
        self.append_count += appended
        return appended

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        return self._ring.sample(**kwargs)

    def update_store_metadata(self, metadata: Mapping[str, Any]) -> None:
        self._ring.update_store_metadata(metadata)

    @staticmethod
    def _step_from_root_batch(
        root_batch: Any,
        index_rows: Any,
        *,
        role: str,
    ) -> SimpleNamespace:
        resident = getattr(root_batch, "resident_observation", None)
        legal_mask = np.asarray(root_batch.legal_mask, dtype=np.bool_)
        if legal_mask.ndim != 2 or int(legal_mask.shape[1]) != ACTION_COUNT:
            raise RuntimeError("owner-search cached root legal_mask shape mismatch")
        root_count = int(legal_mask.shape[0])
        metadata = dict(getattr(root_batch, "metadata", {}) or {})
        batch_size = int(metadata.get("batch_size") or max(1, root_count))
        player_count = int(metadata.get("player_count") or 1)
        if batch_size * player_count != root_count:
            raise RuntimeError("owner-search cached root batch shape mismatch")
        if resident is None:
            observation = np.asarray(root_batch.observation)
            if observation.ndim != 4:
                raise RuntimeError("owner-search cached root observation must be rank-4")
            stack_shape = tuple(int(dim) for dim in observation.shape[1:])
            host_observation = observation.reshape(batch_size, player_count, *stack_shape).copy()
            observation_source = COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
        else:
            host_observation = None
            observation_source = COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        action_mask = legal_mask.reshape(
            batch_size,
            player_count,
            ACTION_COUNT,
        )
        reward = np.asarray(root_batch.target_reward, dtype=np.float32).reshape(
            batch_size,
            player_count,
        )
        done_root = np.asarray(root_batch.done_root, dtype=np.bool_).reshape(
            batch_size,
            player_count,
        )
        done = done_root.any(axis=1)
        joint_action = np.zeros((batch_size, player_count), dtype=np.int16)
        if role == "current":

            def host_array(value: Any, *, dtype: Any) -> np.ndarray:
                if hasattr(value, "detach") and hasattr(value, "cpu"):
                    return value.detach().cpu().numpy().astype(dtype, copy=False)
                return np.asarray(value, dtype=dtype)

            env_row = host_array(index_rows.env_row, dtype=np.int64).reshape(-1)
            player = host_array(index_rows.player, dtype=np.int64).reshape(-1)
            action = host_array(index_rows.action, dtype=np.int16).reshape(-1)
            if env_row.size:
                joint_action[env_row, player] = action
            row_reward = host_array(index_rows.reward, dtype=np.float32).reshape(-1)
            row_final_reward = host_array(
                index_rows.final_reward,
                dtype=np.float32,
            ).reshape(-1)
            row_done = host_array(index_rows.done, dtype=np.bool_).reshape(-1)
            if env_row.size:
                reward = reward.copy()
                final_reward_map = reward.copy()
                reward[env_row, player] = row_reward
                final_reward_map[env_row, player] = row_final_reward
                done = done.copy()
                done[env_row] = row_done
            else:
                final_reward_map = reward.copy()
        else:
            final_reward_map = reward.copy()
        compact_batch = SimpleNamespace(
            observation_source=observation_source,
            final_observation=None,
        )
        return SimpleNamespace(
            observation=host_observation,
            action_mask=action_mask.copy(),
            reward=reward.astype(np.float32, copy=True),
            final_reward_map=final_reward_map.astype(np.float32, copy=True),
            done=done.astype(np.bool_, copy=True),
            terminated=done.astype(np.bool_, copy=True),
            truncated=np.zeros((batch_size,), dtype=np.bool_),
            payload={"joint_action": joint_action},
            compact_batch=compact_batch,
            resident_observation_replay_snapshot=resident,
        )


_COMPACT_TERMINAL_ROW_METADATA_KEYS = frozenset(
    {
        "done_row_count",
        "done_row_indices",
        "terminated_row_count",
        "terminated_row_indices",
        "truncated_row_count",
        "truncated_row_indices",
        "next_final_observation_row_count",
        "next_final_observation_row_indices",
    }
)


def _owner_search_transition_row_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(metadata or {}).items()
        if str(key) not in _COMPACT_TERMINAL_ROW_METADATA_KEYS
    }


class _OwnerSearchDirectTransitionBatchReplayStoreFactorySidecarV1(
    _OwnerSearchReplayStoreFactorySidecarV1
):
    """Owner replay store that consumes fixed transition batches directly."""

    def __init__(
        self,
        *,
        capacity: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(capacity=capacity, metadata=metadata)
        self.direct_transition_batch_replay_requested = True
        self.direct_transition_batch_replay_used = False
        self.direct_transition_batch_replay_batch_count = 0
        self.direct_transition_batch_replay_transition_count = 0
        self.direct_transition_batch_replay_transport_entry_count = 0
        self.direct_transition_batch_replay_max_entries_per_batch = 0
        self.direct_transition_batch_replay_fixed_capacity = 0
        self.direct_transition_batch_replay_padding_count = 0
        self.direct_transition_batch_replay_transport_bytes = 0
        self.direct_transition_batch_replay_digest = ""
        self.direct_transition_batch_replay_legacy_expanded_entry_count = 0
        self.direct_transition_batch_replay_index_entry_object_count = 0
        self.direct_transition_batch_replay_ring_entry_object_count = 0
        self.direct_transition_batch_replay_columnar_append_used = False
        self.direct_transition_batch_replay_columnar_slot_write_count = 0
        self.owner_local_transition_derivation_requested = bool(
            dict(metadata or {}).get(
                "compact_owner_search_owner_local_transition_derivation_requested",
                False,
            )
        )
        self.owner_local_transition_derivation_used = False
        self.owner_local_transition_derivation_batch_count = 0
        self.owner_local_transition_derivation_transition_count = 0
        self.owner_local_transition_derivation_transport_entry_count = 0
        self.owner_local_transition_derivation_cache_hit_count = 0
        self.owner_local_transition_derivation_cache_miss_count = 0
        self.owner_local_transition_derivation_action_checksum_verified_count = 0
        self.owner_local_transition_derivation_action_checksum_mismatch_count = 0
        self.owner_local_transition_derivation_pending_count = 0
        self.owner_local_transition_derivation_dropped_pending_count = 0
        self.owner_local_transition_derivation_transport_bytes = 0
        self.owner_local_transition_derivation_digest = ""
        self.owner_local_transition_derivation_digest_verified = True
        self.owner_local_transition_derivation_build_sec = 0.0
        self.owner_local_transition_derivation_submit_sec = 0.0
        self.owner_local_transition_derivation_fallback_count = 0
        self.owner_local_transition_derivation_fallback_reason = "none"
        self.direct_transition_batch_replay_fixed_soa_requested = bool(
            dict(metadata or {}).get(FIXED_SOA_REPLAY_REQUESTED_KEY, False)
        )
        self.direct_transition_batch_replay_fixed_soa_used = False
        self.direct_transition_batch_replay_fixed_soa_slot_write_count = 0
        self.direct_transition_batch_replay_fixed_soa_entry_view_object_count = 0
        self.direct_transition_batch_replay_fixed_soa_step_view_object_count = 0
        self.direct_transition_batch_replay_fixed_soa_learner_ready_object_count = 0
        self.direct_transition_batch_replay_fixed_soa_table_entry_object_count = 0
        self.direct_transition_batch_replay_fixed_soa_table_concat_count = 0
        self.direct_transition_batch_replay_fixed_soa_fallback_count = 0
        self.direct_transition_batch_replay_fixed_soa_fallback_reason = "none"
        self.direct_transition_batch_replay_fixed_soa_slot_write_sec = 0.0
        self.direct_transition_batch_replay_fixed_soa_successor_index_sec = 0.0
        self.direct_transition_batch_replay_fixed_soa_total_sec = 0.0
        self.direct_transition_batch_replay_fallback_count = 0
        self.direct_transition_batch_replay_fallback_reason = "none"
        self.direct_transition_batch_replay_last_append_sec = 0.0
        self.direct_transition_batch_replay_append_sec = 0.0
        self.direct_transition_batch_replay_accounted_sec = 0.0
        self.direct_transition_batch_replay_array_extract_sec = 0.0
        self.direct_transition_batch_replay_transition_validate_sec = 0.0
        self.direct_transition_batch_replay_device_payload_sec = 0.0
        self.direct_transition_batch_replay_device_replay_payload_flushed_count = 0
        self.direct_transition_batch_replay_deferred_one_simulation_flush_count = 0
        (self.direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count) = 0
        (
            self.direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls
        ) = 0.0
        (self.direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count) = 0
        (
            self.direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count
        ) = 0
        self.direct_transition_batch_replay_pending_deferred_replay_payload_count_max = 0
        self.direct_transition_batch_replay_pending_deferred_replay_payload_final_count = 0
        self.direct_transition_batch_replay_replay_payload_d2h_bytes = 0.0
        self.direct_transition_batch_replay_deferred_one_simulation_replay_flush_sec = 0.0
        self.direct_transition_batch_replay_device_replay_payload_flush_sec = 0.0
        (self.direct_transition_batch_replay_deferred_one_simulation_action_model_state_digest) = ""
        (self.direct_transition_batch_replay_deferred_one_simulation_flush_model_state_digest) = ""
        self.direct_transition_batch_replay_index_rows_build_sec = 0.0
        self.direct_transition_batch_replay_step_object_build_sec = 0.0
        self.direct_transition_batch_replay_ring_append_sec = 0.0
        self.direct_transition_batch_replay_columnar_record_count = 0
        self.direct_transition_batch_replay_columnar_entry_view_object_count = 0
        self.direct_transition_batch_replay_columnar_step_view_object_count = 0
        self.direct_transition_batch_replay_columnar_prepare_sec = 0.0
        self.direct_transition_batch_replay_columnar_register_sec = 0.0
        self.direct_transition_batch_replay_columnar_append_store_sec = 0.0
        self.direct_transition_batch_replay_columnar_retain_sec = 0.0
        self.direct_transition_batch_replay_columnar_evict_sec = 0.0
        self.direct_transition_batch_replay_columnar_evict_release_sec = 0.0
        self.direct_transition_batch_replay_columnar_candidate_indices_sec = 0.0
        self.direct_transition_batch_replay_columnar_cache_refresh_sec = 0.0
        self.direct_transition_batch_replay_columnar_cache_rebuild_sec = 0.0
        self.direct_transition_batch_replay_columnar_total_sec = 0.0

    @staticmethod
    def _replay_payload_metadata_view(replay_payload: Any) -> dict[str, Any]:
        metadata = dict(getattr(replay_payload, "metadata", {}) or {})
        profile_telemetry = metadata.get("profile_telemetry")
        if isinstance(profile_telemetry, Mapping):
            metadata.update(dict(profile_telemetry))
        return metadata

    @staticmethod
    def _telemetry_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _telemetry_int(value: Any) -> int:
        try:
            return int(float(value or 0))
        except (TypeError, ValueError):
            return 0

    def _record_deferred_replay_payload_flush(self, replay_payload: Any) -> None:
        metadata = self._replay_payload_metadata_view(replay_payload)
        if not metadata:
            return
        device_payload_flushed = bool(
            metadata.get("compact_torch_search_service_device_replay_payload_flushed")
            or metadata.get("compact_torch_search_service_replay_payload_flushed")
            or metadata.get("device_replay_payload")
        )
        deferred = bool(
            metadata.get("compact_torch_search_one_simulation_replay_materialization_deferred")
            or metadata.get("compact_torch_search_one_simulation_replay_materialized_on_flush")
            or metadata.get("compact_torch_search_service_defer_one_simulation_replay_payload_used")
        )
        materialized_on_flush = bool(
            metadata.get("compact_torch_search_one_simulation_replay_materialized_on_flush")
        )
        identity_match_present = (
            "compact_torch_search_deferred_one_simulation_model_identity_match" in metadata
        )
        identity_match = bool(
            metadata.get("compact_torch_search_deferred_one_simulation_model_identity_match")
        )
        refresh_crossed_count = self._telemetry_int(
            metadata.get("compact_torch_search_deferred_one_simulation_model_refresh_crossed_count")
        )
        if deferred and not materialized_on_flush:
            raise RuntimeError(
                "direct transition-batch replay deferred one-simulation payload "
                "was not materialized during flush"
            )
        if deferred and identity_match_present and not identity_match:
            raise RuntimeError(
                "direct transition-batch replay deferred one-simulation payload "
                "crossed a model refresh"
            )
        if refresh_crossed_count:
            raise RuntimeError(
                "direct transition-batch replay deferred one-simulation payload "
                "reported model-refresh crossing"
            )
        if device_payload_flushed:
            self.direct_transition_batch_replay_device_replay_payload_flushed_count += 1
        if not deferred:
            return
        self.direct_transition_batch_replay_deferred_one_simulation_flush_count += 1
        if materialized_on_flush:
            (
                self.direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count
            ) += 1
        self.direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls += self._telemetry_float(
            metadata.get(
                "compact_torch_search_deferred_one_simulation_replay_recurrent_inference_calls"
            )
        )
        if identity_match:
            (
                self.direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count
            ) += 1
        (
            self.direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count
        ) += int(refresh_crossed_count)
        self.direct_transition_batch_replay_pending_deferred_replay_payload_count_max = max(
            int(self.direct_transition_batch_replay_pending_deferred_replay_payload_count_max),
            self._telemetry_int(
                metadata.get("compact_torch_search_pending_deferred_replay_payload_count")
            ),
        )
        self.direct_transition_batch_replay_pending_deferred_replay_payload_final_count = (
            self._telemetry_int(
                metadata.get("compact_torch_search_pending_deferred_replay_payload_final_count")
            )
        )
        self.direct_transition_batch_replay_replay_payload_d2h_bytes += self._telemetry_float(
            metadata.get("compact_torch_search_service_replay_payload_d2h_bytes")
        )
        self.direct_transition_batch_replay_deferred_one_simulation_replay_flush_sec += (
            self._telemetry_float(
                metadata.get("compact_torch_search_deferred_one_simulation_replay_flush_sec")
            )
        )
        self.direct_transition_batch_replay_device_replay_payload_flush_sec += (
            self._telemetry_float(
                metadata.get("compact_torch_search_service_device_replay_payload_flush_sec")
            )
        )
        (
            self.direct_transition_batch_replay_deferred_one_simulation_action_model_state_digest
        ) = str(
            metadata.get("compact_torch_search_deferred_one_simulation_action_model_state_digest")
            or ""
        )
        (
            self.direct_transition_batch_replay_deferred_one_simulation_flush_model_state_digest
        ) = str(
            metadata.get("compact_torch_search_deferred_one_simulation_flush_model_state_digest")
            or ""
        )

    @staticmethod
    def _is_owner_local_derived_transition_batch(batch: Any) -> bool:
        metadata = dict(getattr(batch, "metadata", {}) or {})
        schema_id = str(getattr(batch, "schema_id", ""))
        kind = str(metadata.get("compact_owner_search_replay_append_transition_batch_kind") or "")
        return (
            schema_id == COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
            or kind == COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
            or bool(
                metadata.get("compact_owner_search_replay_append_owner_derives_transition_outcome")
            )
        )

    @staticmethod
    def _derived_joint_action_from_search(
        search_result: Any,
        *,
        current_root: Any,
    ) -> np.ndarray:
        current_metadata = dict(getattr(current_root, "metadata", {}) or {})
        batch_size = int(
            current_metadata.get("batch_size")
            or np.asarray(getattr(current_root, "final_observation_row_mask")).size
        )
        player_count = int(
            current_metadata.get("player_count")
            or max(1, int(np.asarray(getattr(current_root, "player")).max(initial=0)) + 1)
        )
        joint_action = np.zeros((batch_size, player_count), dtype=np.int16)
        env_row = np.asarray(search_result.env_row, dtype=np.int64).reshape(-1)
        player = np.asarray(search_result.player, dtype=np.int64).reshape(-1)
        selected = np.asarray(search_result.selected_action, dtype=np.int16).reshape(-1)
        if selected.size:
            joint_action[env_row, player] = selected
        return joint_action

    def append_owner_search_transition_batches(
        self,
        *,
        replay_append_transition_batches: tuple[Any, ...],
        root_batch: Any,
        search_result: Any,
        request: Any,
        root_batch_cache: Mapping[int, Any] | None = None,
        search_result_cache: Mapping[str, Any] | None = None,
        flush_device_replay_payload: Any | None = None,
    ) -> dict[str, Any]:
        del search_result
        append_started = time.perf_counter()
        batches = tuple(replay_append_transition_batches)
        if not batches:
            return {
                "appended_count": 0,
                "cache_handles_to_evict": (),
                "owner_action_feedback": self._empty_action_feedback(),
                **self._direct_transition_batch_metadata(),
            }
        if not callable(flush_device_replay_payload):
            raise RuntimeError(
                "direct transition-batch replay requires flush_device_replay_payload"
            )
        cache = {int(key): value for key, value in dict(root_batch_cache or {}).items()}
        cache[int(getattr(request, "actor_step"))] = root_batch
        cached_search_by_handle = {
            str(key): value for key, value in dict(search_result_cache or {}).items()
        }
        if not cached_search_by_handle:
            raise RuntimeError("direct transition-batch replay requires cached searches")
        entries: list[_CompactReplayRingEntry] = []
        columnar_records: list[CompactReplayColumnarAppendRecordV1] = []
        cache_handles_to_evict: list[str] = []
        feedback = self._empty_action_feedback()
        array_extract_sec = 0.0
        transition_validate_sec = 0.0
        device_payload_sec = 0.0
        index_rows_build_sec = 0.0
        step_object_build_sec = 0.0
        ring_append_sec = 0.0
        batch_count = 0
        transition_count = 0
        derived_batch_count = 0
        derived_transition_count = 0
        derived_cache_hit_count = 0
        derived_cache_miss_count = 0
        derived_action_checksum_verified_count = 0
        derived_action_checksum_mismatch_count = 0
        derived_transport_bytes = 0
        derived_digest = ""
        transition_batch_max_entries_per_batch = 0
        transition_batch_fixed_capacity = 0
        transition_batch_padding_count = 0
        transition_batch_transport_bytes = 0
        transition_batch_digest = ""
        seen_handles: set[str] = set()
        seen_record_indices: set[int] = set()
        for batch in batches:
            array_extract_started = time.perf_counter()
            count = self._transition_batch_count(batch)
            if count <= 0:
                raise RuntimeError("direct transition-batch replay requires transitions")
            batch_count += 1
            transition_count += int(count)
            record_indices = self._transition_batch_array(
                batch,
                "record_indices",
                count=count,
                dtype=np.int64,
            )
            next_record_indices = self._transition_batch_array(
                batch,
                "next_record_indices",
                count=count,
                dtype=np.int64,
            )
            batch_metadata = dict(getattr(batch, "metadata", {}) or {})
            transition_batch_max_entries_per_batch = max(
                int(transition_batch_max_entries_per_batch),
                int(batch_metadata.get("compact_owner_search_transition_batch_max_entries_per_batch") or 0),
            )
            transition_batch_fixed_capacity = max(
                int(transition_batch_fixed_capacity),
                int(batch_metadata.get("compact_owner_search_transition_batch_fixed_capacity") or 0),
            )
            transition_batch_padding_count += int(
                batch_metadata.get("compact_owner_search_transition_batch_padding_count") or 0
            )
            transition_batch_transport_bytes += int(
                batch_metadata.get("compact_owner_search_transition_batch_transport_bytes") or 0
            )
            batch_digest_for_generic = str(
                batch_metadata.get("compact_owner_search_transition_batch_digest") or ""
            )
            if batch_digest_for_generic:
                transition_batch_digest = batch_digest_for_generic
            derived_batch = self._is_owner_local_derived_transition_batch(batch)
            if derived_batch:
                derived_batch_count += 1
                derived_transition_count += int(count)
                derived_transport_bytes += int(
                    batch_metadata.get(
                        "compact_owner_search_owner_local_transition_derivation_transport_bytes"
                    )
                    or batch_metadata.get("compact_owner_search_transition_batch_transport_bytes")
                    or 0
                )
                batch_digest = str(
                    batch_metadata.get(
                        "compact_owner_search_owner_local_transition_derivation_digest"
                    )
                    or batch_metadata.get("compact_owner_search_transition_batch_digest")
                    or ""
                )
                if batch_digest:
                    derived_digest = batch_digest
                applied_action_counts = self._transition_batch_array(
                    batch,
                    "applied_action_counts",
                    count=count,
                    dtype=np.int64,
                )
                applied_action_checksums = self._transition_batch_array(
                    batch,
                    "applied_action_checksums",
                    count=count,
                    dtype=np.int64,
                )
                next_joint_action = None
                next_reward = None
                next_done = None
                next_terminated = None
                next_truncated = None
                next_final_reward_map = None
                next_final_observation_row_mask = None
            else:
                applied_action_counts = None
                applied_action_checksums = None
                next_joint_action = self._transition_batch_array(
                    batch,
                    "next_joint_action",
                    count=count,
                    dtype=np.int16,
                )
                next_reward = self._transition_batch_array(
                    batch,
                    "next_reward",
                    count=count,
                    dtype=np.float32,
                )
                next_done = self._transition_batch_array(
                    batch,
                    "next_done",
                    count=count,
                    dtype=np.bool_,
                )
                next_terminated = self._transition_batch_array(
                    batch,
                    "next_terminated",
                    count=count,
                    dtype=np.bool_,
                )
                next_truncated = self._transition_batch_array(
                    batch,
                    "next_truncated",
                    count=count,
                    dtype=np.bool_,
                )
                next_final_reward_map = self._transition_batch_array(
                    batch,
                    "next_final_reward_map",
                    count=count,
                    dtype=np.float32,
                )
                next_final_observation_row_mask = self._transition_batch_array(
                    batch,
                    "next_final_observation_row_mask",
                    count=count,
                    dtype=np.bool_,
                )
            replay_payload_handles = tuple(
                str(value) for value in getattr(batch, "replay_payload_handles", ())
            )
            selected_action_digests = tuple(
                str(value) for value in getattr(batch, "selected_action_digests", ())
            )
            search_replay_payload_digests = tuple(
                str(value) for value in getattr(batch, "search_replay_payload_digests", ())
            )
            if len(replay_payload_handles) != count:
                raise RuntimeError("direct transition-batch replay handle count mismatch")
            if len(set(replay_payload_handles)) != count:
                raise RuntimeError("direct transition-batch replay duplicate handle")
            if len(selected_action_digests) != count:
                raise RuntimeError("direct transition-batch replay selected digest count mismatch")
            if len(search_replay_payload_digests) != count:
                raise RuntimeError("direct transition-batch replay payload digest count mismatch")
            array_extract_sec += time.perf_counter() - array_extract_started
            for offset in range(count):
                transition_validate_started = time.perf_counter()
                handle = str(replay_payload_handles[offset])
                if handle in seen_handles:
                    raise RuntimeError(
                        "direct transition-batch replay duplicate handle across batches"
                    )
                seen_handles.add(handle)
                cached = cached_search_by_handle.get(handle)
                if cached is None:
                    raise RuntimeError(
                        f"direct transition-batch replay missing cached search: {handle}"
                    )
                record_index = int(record_indices[offset])
                next_record_index = int(next_record_indices[offset])
                if record_index in seen_record_indices:
                    raise RuntimeError(
                        "direct transition-batch replay duplicate record across batches"
                    )
                seen_record_indices.add(record_index)
                if int(getattr(cached, "record_index")) != record_index:
                    raise RuntimeError("direct transition-batch replay cached record mismatch")
                previous_root = cache.get(record_index)
                if previous_root is None:
                    previous_root = getattr(cached, "root_batch")
                current_root = cache.get(next_record_index)
                if previous_root is None or current_root is None:
                    if derived_batch:
                        derived_cache_miss_count += 1
                    raise RuntimeError(
                        "direct transition-batch replay requires cached previous "
                        "and current root batches"
                    )
                if derived_batch:
                    derived_cache_hit_count += 1
                previous_search_result = getattr(cached, "search_result")
                selected_digest = str(selected_action_digests[offset])
                if selected_digest and selected_digest != compact_search_array_digest_v1(
                    previous_search_result.selected_action
                ):
                    raise RuntimeError(
                        "direct transition-batch replay selected-action digest mismatch"
                    )
                expected_replay_digest = str(search_replay_payload_digests[offset]).strip()
                if not expected_replay_digest:
                    raise RuntimeError("direct transition-batch replay missing payload digest")
                owner_replay_digest = compact_search_deferred_replay_payload_digest_v1(handle)
                if expected_replay_digest != owner_replay_digest:
                    raise RuntimeError("direct transition-batch replay payload digest mismatch")
                env_row = np.asarray(
                    previous_search_result.env_row,
                    dtype=np.int64,
                ).reshape(-1)
                player = np.asarray(
                    previous_search_result.player,
                    dtype=np.int64,
                ).reshape(-1)
                selected = np.asarray(
                    previous_search_result.selected_action,
                    dtype=np.int16,
                ).reshape(-1)
                if derived_batch:
                    action_facts = self._derived_joint_action_from_search(
                        previous_search_result,
                        current_root=current_root,
                    )
                    applied = selected.copy()
                    expected_action_count = int(selected.size)
                    expected_action_checksum = self._action_checksum(selected)
                    if int(applied_action_counts[offset]) != expected_action_count:
                        derived_action_checksum_mismatch_count += 1
                        raise RuntimeError("direct derived-transition replay action count mismatch")
                    if int(applied_action_checksums[offset]) != expected_action_checksum:
                        derived_action_checksum_mismatch_count += 1
                        raise RuntimeError(
                            "direct derived-transition replay action checksum mismatch"
                        )
                    derived_action_checksum_verified_count += 1
                    outcome = compact_transition_outcome_v1_from_next_root_batch(current_root)
                    next_reward_value = outcome.next_reward
                    next_done_value = outcome.next_done
                    next_terminated_value = outcome.next_terminated
                    next_truncated_value = outcome.next_truncated
                    next_final_reward_map_value = outcome.next_final_reward_map
                    next_final_observation_row_mask_value = outcome.next_final_observation_row_mask
                else:
                    action_facts = np.asarray(next_joint_action[offset], dtype=np.int16)
                    applied = np.asarray(
                        action_facts[env_row, player],
                        dtype=np.int16,
                    ).reshape(-1)
                    next_reward_value = np.asarray(next_reward[offset], dtype=np.float32)
                    next_done_value = np.asarray(next_done[offset], dtype=np.bool_)
                    next_terminated_value = np.asarray(
                        next_terminated[offset],
                        dtype=np.bool_,
                    )
                    next_truncated_value = np.asarray(
                        next_truncated[offset],
                        dtype=np.bool_,
                    )
                    next_final_reward_map_value = np.asarray(
                        next_final_reward_map[offset],
                        dtype=np.float32,
                    )
                    next_final_observation_row_mask_value = np.asarray(
                        next_final_observation_row_mask[offset],
                        dtype=np.bool_,
                    )
                mismatch_count = int(np.count_nonzero(applied != selected))
                if selected.size and mismatch_count:
                    raise RuntimeError(
                        "direct transition-batch replay action facts do not match search"
                    )
                previous_batch = _compact_batch_from_root_batch(previous_root)
                row_metadata = {
                    **_owner_search_transition_row_metadata(getattr(batch, "metadata", {}) or {}),
                    "compact_owner_search_owner_materialized_replay_rows": True,
                    "compact_owner_search_direct_transition_batch_replay_used": True,
                    "compact_owner_search_direct_transition_batch_replay_offset": int(offset),
                    "compact_owner_search_owner_local_transition_derivation_used": bool(
                        derived_batch
                    ),
                }
                transition_validate_sec += time.perf_counter() - transition_validate_started
                if (
                    str(previous_root.observation_source)
                    == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
                ):
                    row_metadata.update(
                        {
                            "compact_owner_search_owner_materialized_replay_resident_observation": True,
                            "compact_owner_search_owner_materialized_replay_host_observation_copy": False,
                        }
                    )
                    action_step = getattr(cached, "action_step", None)
                    device_payload_started = time.perf_counter()
                    if action_step is not None:
                        replay_payload = getattr(
                            cached,
                            "inner_device_replay_payload",
                            None,
                        )
                        if replay_payload is None:
                            inner_handle = str(
                                getattr(cached, "inner_replay_payload_handle", "")
                                or getattr(action_step, "replay_payload_handle", "")
                            )
                            replay_payload = flush_device_replay_payload(inner_handle)
                        self._record_deferred_replay_payload_flush(replay_payload)
                        validate_compact_device_search_two_phase_payload_v1(
                            action_step,
                            replay_payload,
                        )
                        row_metadata.update(
                            {
                                "compact_owner_search_owner_materialized_inner_device_replay_payload": True,
                                "compact_owner_search_inner_two_phase_action_step": True,
                            }
                        )
                    else:
                        if bool(
                            previous_search_result.metadata.get(
                                "compact_owner_search_action_only_compat_result"
                            )
                        ):
                            raise RuntimeError(
                                "direct transition-batch replay action-only "
                                "compatibility result requires an inner action step"
                            )
                        action_step = compact_search_action_step_v1_from_result(
                            previous_search_result,
                            replay_payload_handle=handle,
                            metadata={
                                "compact_owner_search_owner_materialized_action_step": True,
                                **dict(previous_search_result.metadata),
                            },
                        )
                        replay_payload = _device_replay_payload_from_search_result(
                            previous_search_result,
                            replay_payload_handle=handle,
                            device=_resident_root_device(previous_root),
                        )
                    device_payload_sec += time.perf_counter() - device_payload_started
                    index_rows_started = time.perf_counter()
                    index_rows = build_compact_device_replay_index_rows_v1_from_payload(
                        previous_batch,
                        previous_root,
                        action_step,
                        replay_payload,
                        record_index=record_index,
                        next_joint_action=action_facts,
                        next_reward=next_reward_value,
                        next_done=next_done_value,
                        next_terminated=next_terminated_value,
                        next_truncated=next_truncated_value,
                        next_final_reward_map=next_final_reward_map_value,
                        next_final_observation_row_mask=next_final_observation_row_mask_value,
                        policy_source=str(getattr(batch, "policy_source")),
                        metadata={
                            "compact_owner_search_owner_materialized_device_replay_rows": True,
                            **row_metadata,
                        },
                    )
                    index_rows_build_sec += time.perf_counter() - index_rows_started
                else:
                    index_rows_started = time.perf_counter()
                    index_rows = build_compact_replay_index_rows_v1_from_search_result(
                        previous_batch,
                        previous_root,
                        previous_search_result,
                        record_index=record_index,
                        next_joint_action=action_facts,
                        next_reward=next_reward_value,
                        next_done=next_done_value,
                        next_terminated=next_terminated_value,
                        next_truncated=next_truncated_value,
                        next_final_reward_map=next_final_reward_map_value,
                        next_final_observation_row_mask=next_final_observation_row_mask_value,
                        policy_source=str(getattr(batch, "policy_source")),
                        metadata=row_metadata,
                    )
                    index_rows_build_sec += time.perf_counter() - index_rows_started
                step_object_started = time.perf_counter()
                if (
                    str(getattr(previous_root, "observation_source", ""))
                    == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
                ):
                    previous_resident = getattr(previous_root, "resident_observation", None)
                    current_resident = getattr(current_root, "resident_observation", None)
                    if previous_resident is None or current_resident is None:
                        raise RuntimeError(
                            "direct transition-batch columnar replay requires "
                            "resident previous/current root snapshots"
                        )
                    columnar_records.append(
                        CompactReplayColumnarAppendRecordV1(
                            previous_resident_observation_replay_snapshot=(previous_resident),
                            current_resident_observation_replay_snapshot=current_resident,
                            index_rows=index_rows,
                            observation_source=(COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1),
                        )
                    )
                else:
                    previous_step = self._step_from_root_batch(
                        previous_root,
                        index_rows,
                        role="previous",
                    )
                    current_step = self._step_from_root_batch(
                        current_root,
                        index_rows,
                        role="current",
                    )
                    entries.append(
                        _CompactReplayRingEntry(
                            previous_step=previous_step,
                            current_step=current_step,
                            index_rows=index_rows,
                        )
                    )
                step_object_build_sec += time.perf_counter() - step_object_started
                cache_handles_to_evict.append(handle)
                feedback = self._merge_action_feedback(
                    feedback,
                    action_count=int(selected.size),
                    mismatch_count=int(mismatch_count),
                    expected_checksum=self._action_checksum(selected),
                    applied_checksum=self._action_checksum(applied),
                )
        columnar_telemetry_before = self._ring_columnar_append_telemetry_snapshot()
        fixed_soa_telemetry_before = self._ring_fixed_soa_append_telemetry_snapshot()
        ring_append_started = time.perf_counter()
        appended = 0
        if columnar_records:
            if self.direct_transition_batch_replay_fixed_soa_requested:
                append_fixed_soa_records = getattr(
                    self._ring,
                    "append_fixed_soa_columnar_records",
                    None,
                )
                if not callable(append_fixed_soa_records):
                    raise RuntimeError("direct transition-batch replay requires fixed SoA append")
                appended += int(append_fixed_soa_records(tuple(columnar_records)))
            else:
                append_columnar_entries = getattr(
                    self._ring,
                    "append_columnar_entries",
                    None,
                )
                if not callable(append_columnar_entries):
                    raise RuntimeError(
                        "direct transition-batch replay requires columnar ring append"
                    )
                appended += int(append_columnar_entries(tuple(columnar_records)))
        if entries:
            append_entries = getattr(self._ring, "append_entries", None)
            if not callable(append_entries):
                raise RuntimeError("direct transition-batch replay requires batch ring append")
            appended += int(append_entries(tuple(entries)))
        ring_append_sec += time.perf_counter() - ring_append_started
        columnar_telemetry_after = self._ring_columnar_append_telemetry_snapshot()
        fixed_soa_telemetry_after = self._ring_fixed_soa_append_telemetry_snapshot()
        if self.direct_transition_batch_replay_fixed_soa_requested:
            self._accumulate_fixed_soa_append_telemetry_delta(
                before=fixed_soa_telemetry_before,
                after=fixed_soa_telemetry_after,
            )
        else:
            self._accumulate_columnar_append_telemetry_delta(
                before=columnar_telemetry_before,
                after=columnar_telemetry_after,
            )
        expected_appended = int(len(entries) + len(columnar_records))
        if appended != expected_appended:
            raise RuntimeError("direct transition-batch replay append count mismatch")
        self.append_count += int(appended)
        self.direct_transition_batch_replay_used = True
        self.direct_transition_batch_replay_batch_count += int(batch_count)
        self.direct_transition_batch_replay_transition_count += int(transition_count)
        self.direct_transition_batch_replay_transport_entry_count += int(len(batches))
        self.direct_transition_batch_replay_max_entries_per_batch = max(
            int(self.direct_transition_batch_replay_max_entries_per_batch),
            int(transition_batch_max_entries_per_batch),
        )
        self.direct_transition_batch_replay_fixed_capacity = max(
            int(self.direct_transition_batch_replay_fixed_capacity),
            int(transition_batch_fixed_capacity),
        )
        self.direct_transition_batch_replay_padding_count += int(transition_batch_padding_count)
        self.direct_transition_batch_replay_transport_bytes += int(transition_batch_transport_bytes)
        if transition_batch_digest:
            self.direct_transition_batch_replay_digest = str(transition_batch_digest)
        self.direct_transition_batch_replay_ring_entry_object_count += int(len(entries))
        if derived_transition_count:
            self.owner_local_transition_derivation_used = True
            self.owner_local_transition_derivation_batch_count += int(derived_batch_count)
            self.owner_local_transition_derivation_transition_count += int(derived_transition_count)
            self.owner_local_transition_derivation_transport_entry_count += int(derived_batch_count)
            self.owner_local_transition_derivation_pending_count = 0
            self.owner_local_transition_derivation_transport_bytes += int(derived_transport_bytes)
            self.owner_local_transition_derivation_digest = str(derived_digest)
            self.owner_local_transition_derivation_digest_verified = bool(derived_digest)
            self.owner_local_transition_derivation_cache_hit_count += int(derived_cache_hit_count)
            self.owner_local_transition_derivation_cache_miss_count += int(derived_cache_miss_count)
            self.owner_local_transition_derivation_action_checksum_verified_count += int(
                derived_action_checksum_verified_count
            )
            self.owner_local_transition_derivation_action_checksum_mismatch_count += int(
                derived_action_checksum_mismatch_count
            )
        if columnar_records and self.direct_transition_batch_replay_fixed_soa_requested:
            self.direct_transition_batch_replay_fixed_soa_used = True
        elif columnar_records:
            self.direct_transition_batch_replay_columnar_append_used = True
            self.direct_transition_batch_replay_columnar_slot_write_count += int(
                len(columnar_records)
            )
        self.direct_transition_batch_replay_last_append_sec = max(
            0.0,
            time.perf_counter() - append_started,
        )
        self.direct_transition_batch_replay_append_sec += float(
            self.direct_transition_batch_replay_last_append_sec
        )
        accounted_sec = (
            array_extract_sec
            + transition_validate_sec
            + device_payload_sec
            + index_rows_build_sec
            + step_object_build_sec
            + ring_append_sec
        )
        self.direct_transition_batch_replay_accounted_sec += float(accounted_sec)
        self.direct_transition_batch_replay_array_extract_sec += float(array_extract_sec)
        self.direct_transition_batch_replay_transition_validate_sec += float(
            transition_validate_sec
        )
        self.direct_transition_batch_replay_device_payload_sec += float(device_payload_sec)
        self.direct_transition_batch_replay_index_rows_build_sec += float(index_rows_build_sec)
        self.direct_transition_batch_replay_step_object_build_sec += float(step_object_build_sec)
        self.direct_transition_batch_replay_ring_append_sec += float(ring_append_sec)
        return {
            "appended_count": int(appended),
            "cache_handles_to_evict": tuple(cache_handles_to_evict),
            "owner_action_feedback": feedback,
            **self._direct_transition_batch_metadata(),
            "compact_owner_search_owner_local_transition_derivation_requested": bool(
                self.owner_local_transition_derivation_used or derived_transition_count > 0
            ),
            "compact_owner_search_owner_local_transition_derivation_used": bool(
                self.owner_local_transition_derivation_used
            ),
            "compact_owner_search_owner_local_transition_derivation_schema_id": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
                if self.owner_local_transition_derivation_used
                else ""
            ),
            "compact_owner_search_owner_local_transition_derivation_kind": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
                if self.owner_local_transition_derivation_used
                else ""
            ),
            "compact_owner_search_owner_local_transition_derivation_batch_count": int(
                self.owner_local_transition_derivation_batch_count
            ),
            "compact_owner_search_owner_local_transition_derivation_transition_count": int(
                self.owner_local_transition_derivation_transition_count
            ),
            "compact_owner_search_owner_local_transition_derivation_transport_entry_count": int(
                self.owner_local_transition_derivation_transport_entry_count
            ),
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes": 0,
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count": 0,
            "compact_owner_search_owner_local_transition_derivation_cache_hit_count": int(
                self.owner_local_transition_derivation_cache_hit_count
            ),
            "compact_owner_search_owner_local_transition_derivation_cache_miss_count": int(
                self.owner_local_transition_derivation_cache_miss_count
            ),
            "compact_owner_search_owner_local_transition_derivation_action_checksum_verified_count": int(
                self.owner_local_transition_derivation_action_checksum_verified_count
            ),
            "compact_owner_search_owner_local_transition_derivation_action_checksum_mismatch_count": int(
                self.owner_local_transition_derivation_action_checksum_mismatch_count
            ),
            "compact_owner_search_owner_local_transition_derivation_fallback_count": int(
                self.owner_local_transition_derivation_fallback_count
            ),
            "compact_owner_search_owner_local_transition_derivation_fallback_reason": str(
                self.owner_local_transition_derivation_fallback_reason
            ),
        }

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        result = dict(super().sample(**kwargs))
        proof = self._direct_transition_batch_metadata()
        sample_metadata = dict(result.get("sample_metadata") or {})
        sample_metadata.update(proof)
        result["sample_metadata"] = sample_metadata
        telemetry = dict(result.get("telemetry") or {})
        telemetry.update(proof)
        result["telemetry"] = telemetry
        return result

    def _direct_transition_batch_metadata(self) -> dict[str, Any]:
        owner_local_used = bool(self.owner_local_transition_derivation_used)
        transition_batch_kind = (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
            if owner_local_used
            else COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
        )
        transition_batch_schema_id = (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
            if owner_local_used
            else COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
        )
        transition_batch_count = (
            int(self.owner_local_transition_derivation_batch_count)
            if owner_local_used
            else int(self.direct_transition_batch_replay_batch_count)
        )
        transition_batch_entry_count = (
            int(self.owner_local_transition_derivation_transition_count)
            if owner_local_used
            else int(self.direct_transition_batch_replay_transition_count)
        )
        transition_batch_transport_entry_count = (
            int(self.owner_local_transition_derivation_transport_entry_count)
            if owner_local_used
            else int(self.direct_transition_batch_replay_transport_entry_count)
        )
        transition_batch_transport_bytes = (
            int(self.owner_local_transition_derivation_transport_bytes)
            if owner_local_used
            else int(self.direct_transition_batch_replay_transport_bytes)
        )
        transition_batch_digest = (
            str(self.owner_local_transition_derivation_digest)
            if owner_local_used
            else str(self.direct_transition_batch_replay_digest)
        )
        return {
            "compact_owner_search_owner_replay_transport_kind": str(transition_batch_kind),
            "compact_owner_search_owner_replay_transition_batch_enabled": bool(
                self.direct_transition_batch_replay_requested
            ),
            "compact_owner_search_owner_replay_transport_entry_count": int(
                transition_batch_transport_entry_count
            ),
            "compact_owner_search_owner_replay_transition_batch_count": int(
                transition_batch_count
            ),
            "compact_owner_search_owner_replay_transition_batch_transition_count": int(
                transition_batch_entry_count
            ),
            "compact_owner_search_owner_replay_transition_legacy_entry_count": 0,
            "compact_owner_search_transition_batch_transport_requested": bool(
                self.direct_transition_batch_replay_requested
            ),
            "compact_owner_search_transition_batch_transport_enabled": bool(
                self.direct_transition_batch_replay_requested
            ),
            "compact_owner_search_transition_batch_transport_kind": str(transition_batch_kind),
            "compact_owner_search_transition_batch_schema_id": str(transition_batch_schema_id),
            "compact_owner_search_transition_batch_count": int(transition_batch_count),
            "compact_owner_search_transition_batch_entry_count": int(
                transition_batch_entry_count
            ),
            "compact_owner_search_transition_batch_transport_entry_count": int(
                transition_batch_transport_entry_count
            ),
            "compact_owner_search_transition_batch_max_entries_per_batch": int(
                self.direct_transition_batch_replay_max_entries_per_batch
            ),
            "compact_owner_search_transition_batch_fixed_capacity": int(
                self.direct_transition_batch_replay_fixed_capacity
            ),
            "compact_owner_search_transition_batch_padding_count": int(
                self.direct_transition_batch_replay_padding_count
            ),
            "compact_owner_search_transition_batch_overflow_count": 0,
            "compact_owner_search_transition_batch_fallback_count": 0,
            "compact_owner_search_transition_batch_fallback_reason": "none",
            "compact_owner_search_transition_batch_pending_count": 0,
            "compact_owner_search_transition_batch_transport_bytes": int(
                transition_batch_transport_bytes
            ),
            "compact_owner_search_transition_batch_digest": str(transition_batch_digest),
            "compact_owner_search_transition_batch_digest_verified": bool(
                transition_batch_digest or transition_batch_entry_count == 0
            ),
            "compact_owner_search_direct_transition_batch_replay_requested": bool(
                self.direct_transition_batch_replay_requested
            ),
            "compact_owner_search_direct_transition_batch_replay_used": bool(
                self.direct_transition_batch_replay_used
            ),
            "compact_owner_search_direct_transition_batch_replay_batch_count": int(
                self.direct_transition_batch_replay_batch_count
            ),
            "compact_owner_search_direct_transition_batch_replay_transition_count": int(
                self.direct_transition_batch_replay_transition_count
            ),
            "compact_owner_search_direct_transition_batch_replay_transport_entry_count": int(
                self.direct_transition_batch_replay_transport_entry_count
            ),
            "compact_owner_search_direct_transition_batch_replay_legacy_expanded_entry_count": int(
                self.direct_transition_batch_replay_legacy_expanded_entry_count
            ),
            "compact_owner_search_direct_transition_batch_replay_index_entry_object_count": int(
                self.direct_transition_batch_replay_index_entry_object_count
            ),
            "compact_owner_search_direct_transition_batch_replay_ring_entry_object_count": int(
                self.direct_transition_batch_replay_ring_entry_object_count
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_append_used": bool(
                self.direct_transition_batch_replay_columnar_append_used
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count": int(
                self.direct_transition_batch_replay_columnar_slot_write_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_requested": bool(
                self.direct_transition_batch_replay_fixed_soa_requested
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_used": bool(
                self.direct_transition_batch_replay_fixed_soa_used
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_count": int(
                self.direct_transition_batch_replay_fixed_soa_slot_write_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_entry_view_object_count": int(
                self.direct_transition_batch_replay_fixed_soa_entry_view_object_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_step_view_object_count": int(
                self.direct_transition_batch_replay_fixed_soa_step_view_object_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_learner_ready_object_count": int(
                self.direct_transition_batch_replay_fixed_soa_learner_ready_object_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_entry_object_count": int(
                self.direct_transition_batch_replay_fixed_soa_table_entry_object_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_concat_count": int(
                self.direct_transition_batch_replay_fixed_soa_table_concat_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_count": int(
                self.direct_transition_batch_replay_fixed_soa_fallback_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_reason": str(
                self.direct_transition_batch_replay_fixed_soa_fallback_reason
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_sec": float(
                self.direct_transition_batch_replay_fixed_soa_slot_write_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_successor_index_sec": float(
                self.direct_transition_batch_replay_fixed_soa_successor_index_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_fixed_soa_total_sec": float(
                self.direct_transition_batch_replay_fixed_soa_total_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_fallback_count": int(
                self.direct_transition_batch_replay_fallback_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fallback_reason": str(
                self.direct_transition_batch_replay_fallback_reason
            ),
            "compact_owner_search_direct_transition_batch_replay_last_append_sec": float(
                self.direct_transition_batch_replay_last_append_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_append_sec": float(
                self.direct_transition_batch_replay_append_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_accounted_sec": float(
                self.direct_transition_batch_replay_accounted_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_array_extract_sec": float(
                self.direct_transition_batch_replay_array_extract_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_transition_validate_sec": float(
                self.direct_transition_batch_replay_transition_validate_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_device_payload_sec": float(
                self.direct_transition_batch_replay_device_payload_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flushed_count": int(
                self.direct_transition_batch_replay_device_replay_payload_flushed_count
            ),
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_count": int(
                self.direct_transition_batch_replay_deferred_one_simulation_flush_count
            ),
            "compact_owner_search_direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count": int(
                self.direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count
            ),
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls": float(
                self.direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls
            ),
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count": int(
                self.direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count
            ),
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count": int(
                self.direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count
            ),
            "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_count_max": int(
                self.direct_transition_batch_replay_pending_deferred_replay_payload_count_max
            ),
            "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_final_count": int(
                self.direct_transition_batch_replay_pending_deferred_replay_payload_final_count
            ),
            "compact_owner_search_direct_transition_batch_replay_replay_payload_d2h_bytes": float(
                self.direct_transition_batch_replay_replay_payload_d2h_bytes
            ),
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_flush_sec": float(
                self.direct_transition_batch_replay_deferred_one_simulation_replay_flush_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flush_sec": float(
                self.direct_transition_batch_replay_device_replay_payload_flush_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_action_model_state_digest": str(
                self.direct_transition_batch_replay_deferred_one_simulation_action_model_state_digest
            ),
            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_model_state_digest": str(
                self.direct_transition_batch_replay_deferred_one_simulation_flush_model_state_digest
            ),
            "compact_owner_search_direct_transition_batch_replay_index_rows_build_sec": float(
                self.direct_transition_batch_replay_index_rows_build_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_step_object_build_sec": float(
                self.direct_transition_batch_replay_step_object_build_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_ring_append_sec": float(
                self.direct_transition_batch_replay_ring_append_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_record_count": int(
                self.direct_transition_batch_replay_columnar_record_count
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_entry_view_object_count": int(
                self.direct_transition_batch_replay_columnar_entry_view_object_count
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_step_view_object_count": int(
                self.direct_transition_batch_replay_columnar_step_view_object_count
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_prepare_sec": float(
                self.direct_transition_batch_replay_columnar_prepare_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_register_sec": float(
                self.direct_transition_batch_replay_columnar_register_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_append_store_sec": float(
                self.direct_transition_batch_replay_columnar_append_store_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_retain_sec": float(
                self.direct_transition_batch_replay_columnar_retain_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_evict_sec": float(
                self.direct_transition_batch_replay_columnar_evict_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_evict_release_sec": float(
                self.direct_transition_batch_replay_columnar_evict_release_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_candidate_indices_sec": float(
                self.direct_transition_batch_replay_columnar_candidate_indices_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_cache_refresh_sec": float(
                self.direct_transition_batch_replay_columnar_cache_refresh_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_cache_rebuild_sec": float(
                self.direct_transition_batch_replay_columnar_cache_rebuild_sec
            ),
            "compact_owner_search_direct_transition_batch_replay_columnar_total_sec": float(
                self.direct_transition_batch_replay_columnar_total_sec
            ),
            "compact_owner_search_owner_local_transition_derivation_requested": bool(
                self.owner_local_transition_derivation_requested
                or self.owner_local_transition_derivation_used
            ),
            "compact_owner_search_owner_local_transition_derivation_used": bool(
                self.owner_local_transition_derivation_used
            ),
            "compact_owner_search_owner_local_transition_derivation_schema_id": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
                if self.owner_local_transition_derivation_used
                else ""
            ),
            "compact_owner_search_owner_local_transition_derivation_kind": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
                if self.owner_local_transition_derivation_used
                else ""
            ),
            "compact_owner_search_owner_local_transition_derivation_batch_count": int(
                self.owner_local_transition_derivation_batch_count
            ),
            "compact_owner_search_owner_local_transition_derivation_transition_count": int(
                self.owner_local_transition_derivation_transition_count
            ),
            "compact_owner_search_owner_local_transition_derivation_transport_entry_count": int(
                self.owner_local_transition_derivation_transport_entry_count
            ),
            "compact_owner_search_owner_local_transition_derivation_pending_count": int(
                self.owner_local_transition_derivation_pending_count
            ),
            "compact_owner_search_owner_local_transition_derivation_transport_bytes": int(
                self.owner_local_transition_derivation_transport_bytes
            ),
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes": 0,
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count": 0,
            "compact_owner_search_owner_local_transition_derivation_digest": str(
                self.owner_local_transition_derivation_digest
            ),
            "compact_owner_search_owner_local_transition_derivation_digest_verified": bool(
                self.owner_local_transition_derivation_digest_verified
            ),
            "compact_owner_search_owner_local_transition_derivation_build_sec": float(
                self.owner_local_transition_derivation_build_sec
            ),
            "compact_owner_search_owner_local_transition_derivation_submit_sec": float(
                self.owner_local_transition_derivation_submit_sec
            ),
            "compact_owner_search_owner_local_transition_derivation_cache_hit_count": int(
                self.owner_local_transition_derivation_cache_hit_count
            ),
            "compact_owner_search_owner_local_transition_derivation_cache_miss_count": int(
                self.owner_local_transition_derivation_cache_miss_count
            ),
            "compact_owner_search_owner_local_transition_derivation_action_checksum_verified_count": int(
                self.owner_local_transition_derivation_action_checksum_verified_count
            ),
            "compact_owner_search_owner_local_transition_derivation_action_checksum_mismatch_count": int(
                self.owner_local_transition_derivation_action_checksum_mismatch_count
            ),
            "compact_owner_search_owner_local_transition_derivation_fallback_count": int(
                self.owner_local_transition_derivation_fallback_count
            ),
            "compact_owner_search_owner_local_transition_derivation_fallback_reason": str(
                self.owner_local_transition_derivation_fallback_reason
            ),
            "compact_owner_search_owner_local_transition_derivation_dropped_pending_count": int(
                self.owner_local_transition_derivation_dropped_pending_count
            ),
        }

    def _ring_columnar_append_telemetry_snapshot(self) -> dict[str, float]:
        snapshot_fn = getattr(self._ring, "columnar_append_telemetry_snapshot", None)
        if not callable(snapshot_fn):
            return {}
        return {
            str(key): float(value)
            for key, value in dict(snapshot_fn()).items()
            if isinstance(value, (int, float))
        }

    def _ring_fixed_soa_append_telemetry_snapshot(self) -> dict[str, float]:
        snapshot_fn = getattr(self._ring, "fixed_soa_append_telemetry_snapshot", None)
        if not callable(snapshot_fn):
            return {}
        return {
            str(key): float(value)
            for key, value in dict(snapshot_fn()).items()
            if isinstance(value, (int, float))
        }

    @staticmethod
    def _columnar_append_telemetry_delta(
        before: Mapping[str, float],
        after: Mapping[str, float],
        key: str,
    ) -> float:
        if key not in after:
            return 0.0
        return max(0.0, float(after.get(key, 0.0)) - float(before.get(key, 0.0)))

    def _accumulate_columnar_append_telemetry_delta(
        self,
        *,
        before: Mapping[str, float],
        after: Mapping[str, float],
    ) -> None:
        if not after:
            return
        self.direct_transition_batch_replay_columnar_record_count += int(
            round(self._columnar_append_telemetry_delta(before, after, "record_count"))
        )
        self.direct_transition_batch_replay_columnar_entry_view_object_count += int(
            round(
                self._columnar_append_telemetry_delta(
                    before,
                    after,
                    "entry_view_object_count",
                )
            )
        )
        self.direct_transition_batch_replay_columnar_step_view_object_count += int(
            round(
                self._columnar_append_telemetry_delta(
                    before,
                    after,
                    "step_view_object_count",
                )
            )
        )
        for key, attr in (
            ("prepare_sec", "direct_transition_batch_replay_columnar_prepare_sec"),
            ("register_sec", "direct_transition_batch_replay_columnar_register_sec"),
            (
                "append_store_sec",
                "direct_transition_batch_replay_columnar_append_store_sec",
            ),
            ("retain_sec", "direct_transition_batch_replay_columnar_retain_sec"),
            ("evict_sec", "direct_transition_batch_replay_columnar_evict_sec"),
            (
                "evict_release_sec",
                "direct_transition_batch_replay_columnar_evict_release_sec",
            ),
            (
                "candidate_indices_sec",
                "direct_transition_batch_replay_columnar_candidate_indices_sec",
            ),
            (
                "cache_refresh_sec",
                "direct_transition_batch_replay_columnar_cache_refresh_sec",
            ),
            (
                "cache_rebuild_sec",
                "direct_transition_batch_replay_columnar_cache_rebuild_sec",
            ),
            ("total_sec", "direct_transition_batch_replay_columnar_total_sec"),
        ):
            setattr(
                self,
                attr,
                float(getattr(self, attr))
                + self._columnar_append_telemetry_delta(before, after, key),
            )

    def _accumulate_fixed_soa_append_telemetry_delta(
        self,
        *,
        before: Mapping[str, float],
        after: Mapping[str, float],
    ) -> None:
        if not after:
            return
        int_fields = (
            ("slot_write_count", "direct_transition_batch_replay_fixed_soa_slot_write_count"),
            (
                "entry_view_object_count",
                "direct_transition_batch_replay_fixed_soa_entry_view_object_count",
            ),
            (
                "step_view_object_count",
                "direct_transition_batch_replay_fixed_soa_step_view_object_count",
            ),
            (
                "learner_ready_object_count",
                "direct_transition_batch_replay_fixed_soa_learner_ready_object_count",
            ),
            (
                "table_entry_object_count",
                "direct_transition_batch_replay_fixed_soa_table_entry_object_count",
            ),
            (
                "table_concat_count",
                "direct_transition_batch_replay_fixed_soa_table_concat_count",
            ),
            ("fallback_count", "direct_transition_batch_replay_fixed_soa_fallback_count"),
        )
        for key, attr in int_fields:
            setattr(
                self,
                attr,
                int(getattr(self, attr))
                + int(round(self._columnar_append_telemetry_delta(before, after, key))),
            )
        for key, attr in (
            ("slot_write_sec", "direct_transition_batch_replay_fixed_soa_slot_write_sec"),
            (
                "successor_index_sec",
                "direct_transition_batch_replay_fixed_soa_successor_index_sec",
            ),
            ("total_sec", "direct_transition_batch_replay_fixed_soa_total_sec"),
        ):
            setattr(
                self,
                attr,
                float(getattr(self, attr))
                + self._columnar_append_telemetry_delta(before, after, key),
            )

    @staticmethod
    def _transition_batch_count(batch: Any) -> int:
        metadata = dict(getattr(batch, "metadata", {}) or {})
        schema_id = str(getattr(batch, "schema_id", ""))
        if schema_id not in {
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID,
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID,
        }:
            raise RuntimeError("direct transition-batch replay schema mismatch")
        kind = str(metadata.get("compact_owner_search_replay_append_transition_batch_kind") or "")
        expected_kind = (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
            if schema_id == COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
            else COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
        )
        if kind != expected_kind:
            raise RuntimeError("direct transition-batch replay kind mismatch")
        count = int(getattr(batch, "transition_count", 0) or 0)
        if count <= 0:
            raise RuntimeError("direct transition-batch replay count must be positive")
        metadata_count = int(
            metadata.get(
                "compact_owner_search_replay_append_transition_batch_transition_count",
                metadata.get("compact_owner_search_transition_batch_entry_count", count),
            )
            or 0
        )
        if metadata_count != count:
            raise RuntimeError("direct transition-batch replay metadata count mismatch")
        capacity = int(metadata.get("compact_owner_search_transition_batch_fixed_capacity") or 0)
        padding = int(metadata.get("compact_owner_search_transition_batch_padding_count") or 0)
        if capacity <= 1 or count > capacity or padding != capacity - count:
            raise RuntimeError("direct transition-batch replay fixed-capacity metadata mismatch")
        if int(metadata.get("compact_owner_search_transition_batch_overflow_count") or 0):
            raise RuntimeError("direct transition-batch replay overflow must be zero")
        return count

    @staticmethod
    def _transition_batch_array(
        batch: Any,
        name: str,
        *,
        count: int,
        dtype: Any,
    ) -> np.ndarray:
        array = np.asarray(getattr(batch, name), dtype=dtype)
        if array.ndim <= 0 or int(array.shape[0]) != int(count):
            raise RuntimeError(f"direct transition-batch replay {name} shape mismatch")
        return array[: int(count)]

    @staticmethod
    def _empty_action_feedback() -> dict[str, Any]:
        return {
            "compact_owner_search_action_feedback_verified": False,
            "compact_owner_search_action_feedback_transition_count": 0,
            "compact_owner_search_action_feedback_action_count": 0,
            "compact_owner_search_action_feedback_mismatch_count": 0,
            "compact_owner_search_expected_joint_action_checksum": 0,
            "compact_owner_search_applied_joint_action_checksum": 0,
            "compact_owner_search_replay_action_checksum": 0,
        }

    @classmethod
    def _merge_action_feedback(
        cls,
        feedback: Mapping[str, Any],
        *,
        action_count: int,
        mismatch_count: int,
        expected_checksum: int,
        applied_checksum: int,
    ) -> dict[str, Any]:
        merged = dict(feedback)
        merged["compact_owner_search_action_feedback_transition_count"] = (
            int(merged.get("compact_owner_search_action_feedback_transition_count") or 0) + 1
        )
        merged["compact_owner_search_action_feedback_action_count"] = int(
            merged.get("compact_owner_search_action_feedback_action_count") or 0
        ) + int(action_count)
        merged["compact_owner_search_action_feedback_mismatch_count"] = int(
            merged.get("compact_owner_search_action_feedback_mismatch_count") or 0
        ) + int(mismatch_count)
        merged["compact_owner_search_expected_joint_action_checksum"] = int(
            merged.get("compact_owner_search_expected_joint_action_checksum") or 0
        ) + int(expected_checksum)
        merged["compact_owner_search_applied_joint_action_checksum"] = int(
            merged.get("compact_owner_search_applied_joint_action_checksum") or 0
        ) + int(applied_checksum)
        merged["compact_owner_search_replay_action_checksum"] = int(
            merged.get("compact_owner_search_replay_action_checksum") or 0
        ) + int(expected_checksum)
        merged["compact_owner_search_action_feedback_verified"] = (
            int(merged["compact_owner_search_action_feedback_action_count"]) > 0
            and int(merged["compact_owner_search_action_feedback_mismatch_count"]) == 0
        )
        return merged

    @staticmethod
    def _action_checksum(action: Any) -> int:
        flat = np.asarray(action, dtype=np.int64).reshape(-1)
        if flat.size <= 0:
            return 0
        weights = np.arange(1, flat.size + 1, dtype=np.int64)
        return int(np.dot(flat + 1, weights))


def _owner_search_request_with_train_params(
    request: Any,
    *,
    sample_batch_size: int,
    train_steps: int,
) -> Any:
    if hasattr(request, "sample_batch_size") and hasattr(request, "train_steps"):
        return request
    return SimpleNamespace(
        request_id=int(getattr(request, "request_id")),
        sample_batch_size=int(sample_batch_size),
        train_steps=int(train_steps),
        policy_version_ref=str(getattr(request, "policy_version_ref", "")),
        model_version_ref=str(getattr(request, "model_version_ref", "")),
        policy_source=str(getattr(request, "policy_source", "")),
        refresh_model=bool(getattr(request, "refresh_model", True)),
    )


def _owner_search_host_only_model_for_process_worker(model: Any) -> Any:
    cloned = _cpu_clone_model_for_process_worker(model)
    to_device = getattr(cloned, "to", None)
    if callable(to_device):
        cloned = to_device("cpu")
    if not _owner_search_contains_cuda_tensor(cloned):
        return cloned
    if isinstance(cloned, torch.nn.Module):
        for attr_name, attr_value in list(vars(cloned).items()):
            if attr_name in {"_parameters", "_buffers", "_modules"}:
                continue
            setattr(cloned, attr_name, _host_only_clone(attr_value))
        to_device = getattr(cloned, "to", None)
        if callable(to_device):
            cloned = to_device("cpu")
    if _owner_search_contains_cuda_tensor(cloned):
        raise RuntimeError("owner-search process learner model clone still contains CUDA tensors")
    return cloned


class _OwnerSearchMuZeroLearnerFactorySidecarV1:
    """Owner-worker compact MuZero learner with owner-ref publication."""

    def __init__(
        self,
        *,
        model: Any,
        seed: int,
        device: str,
        support_scale: int,
        num_unroll_steps: int,
        model_owner_ref_payload_kind: str = COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
        shared_model_state_with_search: bool = False,
        host_clone_learner_payload: bool = True,
        defer_model_state_digest_to_search_refresh: bool = False,
    ) -> None:
        self._seed = int(seed)
        self._num_unroll_steps = int(num_unroll_steps)
        self._device = str(device)
        self._shared_model_state_with_search = bool(shared_model_state_with_search)
        self._host_clone_learner_payload = bool(host_clone_learner_payload)
        self._defer_model_state_digest_to_search_refresh = bool(
            defer_model_state_digest_to_search_refresh
        )
        payload_kind = str(model_owner_ref_payload_kind)
        allowed_payload_kinds = {
            COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
            COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1,
            OWNER_SEARCH_MODEL_STATE_TRANSPORT_SHARED_MODEL_V1,
        }
        if payload_kind not in allowed_payload_kinds:
            raise ValueError(
                f"model_owner_ref_payload_kind must be one of {sorted(allowed_payload_kinds)}"
            )
        self._model_owner_ref_payload_kind = payload_kind
        self._learner = _build_compact_muzero_process_worker_learner(
            model=model,
            seed=int(seed),
            device=str(device),
            support_scale=int(support_scale),
            num_unroll_steps=int(num_unroll_steps),
            require_resident_sample=False,
            require_device_replay_rows=False,
        )

    def prepare_owner_search_learner_payload(
        self,
        *,
        replay_store: Any,
        root_batch: Any,
        search_result: Any,
        request: Any,
    ) -> dict[str, Any]:
        del root_batch, search_result
        if replay_store is None:
            raise RuntimeError("owner-search learner needs owner replay_store")
        sample_started = time.perf_counter()
        sample_result = replay_store.sample(
            seed=int(self._seed) + int(request.request_id),
            sample_batch_size=int(request.sample_batch_size),
            require_next_targets=True,
            num_unroll_steps=int(self._num_unroll_steps),
            build_compact_muzero_learner_batch=True,
            compact_muzero_learner_batch_only=True,
        )
        sample_sec = max(0.0, time.perf_counter() - sample_started)
        if int(sample_result.get("sample_row_count") or 0) <= 0:
            raise RuntimeError("owner-search learner sampled zero replay rows")
        learner_batch = sample_result.get("learner_batch")
        sample_batch = None if learner_batch is not None else sample_result.get("sample_batch")
        payload = {
            "sample_sec": float(sample_sec),
            "sample_row_count": int(sample_result.get("sample_row_count") or 0),
            "sample_metadata": dict(sample_result.get("sample_metadata") or {}),
            "sample_telemetry": dict(sample_result.get("telemetry") or {}),
            "sample_batch": sample_batch,
            "learner_batch": learner_batch,
        }
        if self._host_clone_learner_payload:
            host_clone_started = time.perf_counter()
            payload = _host_only_clone(payload)
            payload["process_learner_payload_host_clone_sec"] = max(
                0.0,
                time.perf_counter() - host_clone_started,
            )
        else:
            payload["process_learner_payload_host_clone_sec"] = 0.0
        return payload

    def train_owner_search_step(
        self,
        *,
        replay_store: Any,
        root_batch: Any,
        search_result: Any,
        sample_batch_size: int,
        train_steps: int,
        request: Any,
    ) -> dict[str, Any]:
        request = _owner_search_request_with_train_params(
            request,
            sample_batch_size=sample_batch_size,
            train_steps=train_steps,
        )
        payload = self.prepare_owner_search_learner_payload(
            replay_store=replay_store,
            root_batch=root_batch,
            search_result=search_result,
            request=request,
        )
        return self.train_owner_search_learner_payload(
            payload=payload,
            request=request,
        )

    def train_owner_search_learner_payload(
        self,
        *,
        payload: Mapping[str, Any],
        request: Any,
    ) -> dict[str, Any]:
        train_wall_started = time.perf_counter()
        sample_sec = float(payload.get("sample_sec") or 0.0)
        learner_batch = payload.get("learner_batch")
        sample_batch = payload.get("sample_batch")
        device_move_started = time.perf_counter()
        if self._device and str(self._device) != "cpu":
            if learner_batch is not None:
                learner_batch = _move_tensors_to_device(
                    learner_batch,
                    target_device=self._device,
                )
            if sample_batch is not None:
                sample_batch = _move_tensors_to_device(
                    sample_batch,
                    target_device=self._device,
                )
        payload_device_move_sec = max(0.0, time.perf_counter() - device_move_started)
        learner_update_started = time.perf_counter()
        if learner_batch is not None:
            learner_result = self._learner.train_on_learner_batch(
                learner_batch,
                train_steps=int(request.train_steps),
            )
        elif sample_batch is not None:
            learner_result = self._learner.train_on_sample_batch(
                sample_batch,
                train_steps=int(request.train_steps),
            )
        else:
            raise RuntimeError("owner-search learner sample produced no trainable batch")
        learner_update_sec = max(0.0, time.perf_counter() - learner_update_started)
        update_delta = int(
            learner_result.get("compact_rollout_slab_learner_gate_updates")
            or int(request.train_steps)
        )
        refresh_model = bool(getattr(request, "refresh_model", True))
        digest = ""
        digest_sec = 0.0
        model_state_dict: dict[str, Any] | None = None
        model_state_dict_sec = 0.0
        model_state_snapshot: dict[str, Any] | None = None
        model_state_snapshot_write_sec = 0.0
        model_state_snapshot_bytes = 0
        owner_ref: dict[str, Any] | None = None
        owner_ref_build_sec = 0.0
        shared_model_ref = (
            refresh_model
            and self._shared_model_state_with_search
            and self._model_owner_ref_payload_kind
            == OWNER_SEARCH_MODEL_STATE_TRANSPORT_SHARED_MODEL_V1
        )
        if refresh_model and not shared_model_ref:
            state_dict_started = time.perf_counter()
            model_state_dict = self._learner.model_state_dict()
            model_state_dict_sec = max(0.0, time.perf_counter() - state_dict_started)
            if self._defer_model_state_digest_to_search_refresh:
                if (
                    self._model_owner_ref_payload_kind
                    == COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1
                ):
                    raise RuntimeError(
                        "deferred model-state digest requires same-process owner ref"
                    )
                digest = (
                    "search-refresh-deferred-model-state:"
                    f"{int(request.request_id)}:{int(update_delta)}"
                )
            else:
                digest_started = time.perf_counter()
                digest = self._learner.model_state_digest()
                digest_sec = max(0.0, time.perf_counter() - digest_started)
        if (
            refresh_model
            and self._model_owner_ref_payload_kind == COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1
        ):
            assert model_state_dict is not None
            snapshot = _write_model_state_snapshot_file(
                model_state_dict,
                request_id=int(request.request_id),
                digest=digest,
                object_id=None,
            )
            model_state_snapshot = dict(snapshot)
            model_state_snapshot_write_sec = float(
                model_state_snapshot.get("write_sec", 0.0) or 0.0
            )
            model_state_snapshot_bytes = int(model_state_snapshot.get("bytes", 0) or 0)
        if refresh_model:
            owner_ref_started = time.perf_counter()
            if shared_model_ref:
                digest = f"shared-model-state:{int(request.request_id)}:{int(request.train_steps)}"
            owner_ref = {
                "schema_id": "curvyzero_compact_owned_loop_model_owner_ref/v1",
                "transport_kind": COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
                "model_state_digest": digest,
                "model_state_payload_kind": str(self._model_owner_ref_payload_kind),
                "policy_version_ref": str(request.policy_version_ref),
                "model_version_ref": str(request.model_version_ref),
                "policy_source": str(request.policy_source),
                "worker_pid": 0,
            }
            if shared_model_ref:
                owner_ref["shared_model_state"] = True
            elif model_state_snapshot is None:
                assert model_state_dict is not None
                owner_ref["model_state_dict"] = model_state_dict
                if self._defer_model_state_digest_to_search_refresh:
                    owner_ref[
                        COMPACT_OWNER_SEARCH_OWNER_REF_DIGEST_DEFERRED_TO_SEARCH_REFRESH_KEY
                    ] = True
                    owner_ref["model_state_digest_source"] = "deferred_to_search_refresh_after_load"
            else:
                owner_ref["model_state_snapshot"] = model_state_snapshot
            owner_ref_build_sec = max(0.0, time.perf_counter() - owner_ref_started)
        learner_telemetry = getattr(learner_result, "telemetry", None)
        if not isinstance(learner_telemetry, Mapping) and isinstance(learner_result, Mapping):
            nested_telemetry = learner_result.get("telemetry")
            if not isinstance(nested_telemetry, Mapping):
                nested_telemetry = learner_result.get(
                    "compact_rollout_slab_learner_gate_compact_muzero_telemetry"
                )
            learner_telemetry = nested_telemetry if isinstance(nested_telemetry, Mapping) else {}
        if not isinstance(learner_telemetry, Mapping):
            learner_telemetry = {}
        train_wall_sec = sample_sec + max(0.0, time.perf_counter() - train_wall_started)
        train_accounted_sec = (
            sample_sec
            + payload_device_move_sec
            + learner_update_sec
            + digest_sec
            + model_state_dict_sec
            + model_state_snapshot_write_sec
            + owner_ref_build_sec
        )
        train_timing_telemetry = {
            "compact_owner_search_owner_train_refresh_model": bool(refresh_model),
            "compact_owner_search_owner_train_model_state_digest_deferred_to_refresh": (
                bool(self._defer_model_state_digest_to_search_refresh)
                and bool(refresh_model)
                and not bool(shared_model_ref)
            ),
            "compact_owner_search_owner_train_wall_sec": float(train_wall_sec),
            "compact_owner_search_owner_train_sample_sec": float(sample_sec),
            "compact_owner_search_owner_train_payload_host_clone_sec": float(
                payload.get("process_learner_payload_host_clone_sec") or 0.0
            ),
            "compact_owner_search_owner_train_payload_device_move_sec": float(
                payload_device_move_sec
            ),
            "compact_owner_search_owner_train_learner_update_sec": float(learner_update_sec),
            "compact_owner_search_owner_train_model_state_digest_sec": float(digest_sec),
            "compact_owner_search_owner_train_model_state_dict_sec": float(model_state_dict_sec),
            "compact_owner_search_owner_train_owner_ref_build_sec": float(owner_ref_build_sec),
            "compact_owner_search_owner_train_model_state_snapshot_returned": bool(
                model_state_snapshot is not None
            ),
            "compact_owner_search_owner_train_model_state_snapshot_bytes": int(
                model_state_snapshot_bytes
            ),
            "compact_owner_search_owner_train_model_state_snapshot_write_sec": float(
                model_state_snapshot_write_sec
            ),
            "compact_owner_search_owner_train_accounted_sec": float(train_accounted_sec),
            "compact_owner_search_owner_train_residual_sec": float(
                max(0.0, train_wall_sec - train_accounted_sec)
            ),
        }
        merged_learner_telemetry = dict(learner_telemetry)
        merged_learner_telemetry.update(
            _owner_search_learner_resident_batch_handle_fields(
                learner_batch=learner_batch,
                metadata=learner_telemetry,
            )
        )
        merged_learner_telemetry.update(train_timing_telemetry)
        learner_result_payload = (
            dict(learner_result)
            if isinstance(learner_result, Mapping)
            else {"telemetry": dict(merged_learner_telemetry)}
        )
        learner_result_payload["telemetry"] = dict(merged_learner_telemetry)
        return {
            "learner_update_count": update_delta,
            "model_owner_ref": owner_ref,
            "sample_row_count": int(payload.get("sample_row_count") or 0),
            "sample_metadata": dict(payload.get("sample_metadata") or {}),
            "sample_telemetry": dict(payload.get("sample_telemetry") or {}),
            "learner_telemetry": dict(merged_learner_telemetry),
            "learner_result": learner_result_payload,
        }


class _OwnerSearchFastMockLearnerFactorySidecarV1:
    """Owner-search ceiling learner: sample/build batch, skip neural update."""

    _REQUIRED_LEARNER_BATCH_PROOF_KEYS = (
        "compact_muzero_learner_value_valid_count",
        "compact_muzero_learner_done_count",
        "compact_muzero_learner_truncated_count",
    )

    def __init__(
        self,
        *,
        model: Any,
        seed: int,
        device: str,
        support_scale: int,
        num_unroll_steps: int,
        model_owner_ref_payload_kind: str = COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
        shared_model_state_with_search: bool = False,
        host_clone_learner_payload: bool = True,
        defer_model_state_digest_to_search_refresh: bool = False,
    ) -> None:
        del device, support_scale
        self._seed = int(seed)
        self._num_unroll_steps = int(num_unroll_steps)
        self._shared_model_state_with_search = bool(shared_model_state_with_search)
        self._host_clone_learner_payload = bool(host_clone_learner_payload)
        self._defer_model_state_digest_to_search_refresh = bool(
            defer_model_state_digest_to_search_refresh
        )
        payload_kind = str(model_owner_ref_payload_kind)
        allowed_payload_kinds = {
            COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
            COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1,
            OWNER_SEARCH_MODEL_STATE_TRANSPORT_SHARED_MODEL_V1,
        }
        if payload_kind not in allowed_payload_kinds:
            raise ValueError(
                f"model_owner_ref_payload_kind must be one of {sorted(allowed_payload_kinds)}"
            )
        self._model_owner_ref_payload_kind = payload_kind
        self._model = model
        self._model_state_dict = {
            str(key): value.detach().cpu().clone() if hasattr(value, "detach") else value
            for key, value in dict(model.state_dict()).items()
        }
        self._model_state_digest = compact_model_state_digest_v1(model)

    def prepare_owner_search_learner_payload(
        self,
        *,
        replay_store: Any,
        root_batch: Any,
        search_result: Any,
        request: Any,
    ) -> dict[str, Any]:
        del root_batch, search_result
        if replay_store is None:
            raise RuntimeError("owner-search mock learner needs owner replay_store")
        sample_started = time.perf_counter()
        sample_result = replay_store.sample(
            seed=int(self._seed) + int(request.request_id),
            sample_batch_size=int(request.sample_batch_size),
            require_next_targets=True,
            num_unroll_steps=int(self._num_unroll_steps),
            build_compact_muzero_learner_batch=True,
            compact_muzero_learner_batch_only=True,
        )
        sample_sec = max(0.0, time.perf_counter() - sample_started)
        if int(sample_result.get("sample_row_count") or 0) <= 0:
            raise RuntimeError("owner-search mock learner sampled zero replay rows")
        payload = {
            "sample_sec": float(sample_sec),
            "sample_row_count": int(sample_result.get("sample_row_count") or 0),
            "sample_metadata": dict(sample_result.get("sample_metadata") or {}),
            "sample_telemetry": dict(sample_result.get("telemetry") or {}),
            "learner_batch": sample_result.get("learner_batch"),
        }
        if self._host_clone_learner_payload:
            host_clone_started = time.perf_counter()
            payload = _host_only_clone(payload)
            payload["process_learner_payload_host_clone_sec"] = max(
                0.0,
                time.perf_counter() - host_clone_started,
            )
        else:
            payload["process_learner_payload_host_clone_sec"] = 0.0
        return payload

    def train_owner_search_step(
        self,
        *,
        replay_store: Any,
        root_batch: Any,
        search_result: Any,
        sample_batch_size: int,
        train_steps: int,
        request: Any,
    ) -> dict[str, Any]:
        request = _owner_search_request_with_train_params(
            request,
            sample_batch_size=sample_batch_size,
            train_steps=train_steps,
        )
        payload = self.prepare_owner_search_learner_payload(
            replay_store=replay_store,
            root_batch=root_batch,
            search_result=search_result,
            request=request,
        )
        return self.train_owner_search_learner_payload(
            payload=payload,
            request=request,
        )

    def train_owner_search_learner_payload(
        self,
        *,
        payload: Mapping[str, Any],
        request: Any,
    ) -> dict[str, Any]:
        train_wall_started = time.perf_counter()
        sample_sec = float(payload.get("sample_sec") or 0.0)
        learner_batch = payload.get("learner_batch")
        learner_batch_metadata = dict(getattr(learner_batch, "metadata", {}) or {})
        if not learner_batch_metadata:
            learner_batch_metadata = dict(payload.get("sample_metadata") or {})
        if not learner_batch_metadata.get("compact_muzero_learner_batch_schema_id"):
            raise RuntimeError(
                "owner-search mock learner requires compact MuZero learner-batch metadata"
            )
        missing_proof_keys = [
            key
            for key in self._REQUIRED_LEARNER_BATCH_PROOF_KEYS
            if key not in learner_batch_metadata
        ]
        if missing_proof_keys:
            joined = ", ".join(missing_proof_keys)
            raise RuntimeError(
                f"owner-search mock learner missing learner-batch proof keys: {joined}"
            )
        update_delta = int(request.train_steps)
        refresh_model = bool(getattr(request, "refresh_model", True))
        model_state_snapshot: dict[str, Any] | None = None
        model_state_snapshot_write_sec = 0.0
        model_state_snapshot_bytes = 0
        owner_ref: dict[str, Any] | None = None
        owner_ref_build_sec = 0.0
        shared_model_ref = (
            refresh_model
            and self._shared_model_state_with_search
            and self._model_owner_ref_payload_kind
            == OWNER_SEARCH_MODEL_STATE_TRANSPORT_SHARED_MODEL_V1
        )
        if (
            refresh_model
            and not shared_model_ref
            and self._model_owner_ref_payload_kind == COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1
        ):
            snapshot = _write_model_state_snapshot_file(
                self._model_state_dict,
                request_id=int(request.request_id),
                digest=self._model_state_digest,
                object_id=None,
            )
            model_state_snapshot = dict(snapshot)
            model_state_snapshot_write_sec = float(
                model_state_snapshot.get("write_sec", 0.0) or 0.0
            )
            model_state_snapshot_bytes = int(model_state_snapshot.get("bytes", 0) or 0)
        if refresh_model:
            owner_ref_started = time.perf_counter()
            model_state_digest = self._model_state_digest
            if shared_model_ref:
                model_state_digest = (
                    f"shared-model-state:{int(request.request_id)}:{int(request.train_steps)}"
                )
            elif self._defer_model_state_digest_to_search_refresh:
                if (
                    self._model_owner_ref_payload_kind
                    == COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1
                ):
                    raise RuntimeError(
                        "deferred model-state digest requires same-process owner ref"
                    )
                model_state_digest = (
                    "search-refresh-deferred-model-state:"
                    f"{int(request.request_id)}:{int(update_delta)}"
                )
            owner_ref = {
                "schema_id": "curvyzero_compact_owned_loop_model_owner_ref/v1",
                "transport_kind": COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
                "model_state_digest": model_state_digest,
                "model_state_payload_kind": str(self._model_owner_ref_payload_kind),
                "policy_version_ref": str(request.policy_version_ref),
                "model_version_ref": str(request.model_version_ref),
                "policy_source": str(request.policy_source),
                "worker_pid": 0,
            }
            if shared_model_ref:
                owner_ref["shared_model_state"] = True
            elif model_state_snapshot is None:
                owner_ref["model_state_dict"] = self._model_state_dict
                if self._defer_model_state_digest_to_search_refresh:
                    owner_ref[
                        COMPACT_OWNER_SEARCH_OWNER_REF_DIGEST_DEFERRED_TO_SEARCH_REFRESH_KEY
                    ] = True
                    owner_ref["model_state_digest_source"] = "deferred_to_search_refresh_after_load"
            else:
                owner_ref["model_state_snapshot"] = model_state_snapshot
            owner_ref_build_sec = max(0.0, time.perf_counter() - owner_ref_started)
        train_wall_sec = sample_sec + max(0.0, time.perf_counter() - train_wall_started)
        train_accounted_sec = sample_sec + model_state_snapshot_write_sec + owner_ref_build_sec
        learner_telemetry = dict(learner_batch_metadata)
        learner_telemetry.update(
            _owner_search_learner_resident_batch_handle_fields(
                learner_batch=learner_batch,
                metadata=learner_batch_metadata,
            )
        )
        learner_telemetry.update(
            {
                "compact_owner_search_mock_fast_learner": True,
                "compact_owner_search_owner_train_refresh_model": bool(refresh_model),
                "compact_owner_search_owner_train_model_state_digest_deferred_to_refresh": (
                    bool(self._defer_model_state_digest_to_search_refresh)
                    and bool(refresh_model)
                    and not bool(shared_model_ref)
                ),
                "compact_owner_search_owner_train_wall_sec": float(train_wall_sec),
                "compact_owner_search_owner_train_sample_sec": float(sample_sec),
                "compact_owner_search_owner_train_payload_host_clone_sec": float(
                    payload.get("process_learner_payload_host_clone_sec") or 0.0
                ),
                "compact_owner_search_owner_train_payload_device_move_sec": 0.0,
                "compact_owner_search_owner_train_learner_update_sec": 0.0,
                "compact_owner_search_owner_train_model_state_digest_sec": 0.0,
                "compact_owner_search_owner_train_model_state_dict_sec": 0.0,
                "compact_owner_search_owner_train_owner_ref_build_sec": float(owner_ref_build_sec),
                "compact_owner_search_owner_train_model_state_snapshot_returned": bool(
                    model_state_snapshot is not None
                ),
                "compact_owner_search_owner_train_model_state_snapshot_bytes": int(
                    model_state_snapshot_bytes
                ),
                "compact_owner_search_owner_train_model_state_snapshot_write_sec": float(
                    model_state_snapshot_write_sec
                ),
                "compact_owner_search_owner_train_accounted_sec": float(train_accounted_sec),
                "compact_owner_search_owner_train_residual_sec": float(
                    max(0.0, train_wall_sec - train_accounted_sec)
                ),
                "compact_muzero_learner_sec": 0.0,
            }
        )
        return {
            "learner_update_count": update_delta,
            "model_owner_ref": owner_ref,
            "sample_row_count": int(payload.get("sample_row_count") or 0),
            "sample_metadata": dict(payload.get("sample_metadata") or {}),
            "sample_telemetry": dict(payload.get("sample_telemetry") or {}),
            "learner_telemetry": dict(learner_telemetry),
            "learner_result": {"telemetry": dict(learner_telemetry)},
        }


class _OwnerSearchCompactTorchServiceSidecarV1:
    """Compact Torch search wrapper that consumes owner-local model refs."""

    def __init__(self, inner: CompactTorchSearchServiceV1) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def refresh_model_owner_ref(
        self,
        *,
        owner_ref: Mapping[str, Any],
        policy_version_ref: str,
        model_version_ref: str,
        policy_source: str,
        learner_update_count: int,
        expected_model_state_digest: str | None = None,
    ) -> dict[str, Any]:
        payload_kind = str(owner_ref.get("model_state_payload_kind") or "")
        if payload_kind == OWNER_SEARCH_MODEL_STATE_TRANSPORT_SHARED_MODEL_V1:
            refresh_shared = getattr(self._inner, "refresh_shared_model_state", None)
            if not callable(refresh_shared):
                raise RuntimeError(
                    "owner-search shared model ref requires "
                    "search_service.refresh_shared_model_state"
                )
            return refresh_shared(
                policy_version_ref=str(policy_version_ref),
                model_version_ref=str(model_version_ref),
                policy_source=str(policy_source),
                learner_update_count=int(learner_update_count),
                model_state_digest=str(owner_ref.get("model_state_digest") or ""),
            )
        model_state_dict = owner_ref.get("model_state_dict")
        snapshot_load_sec = 0.0
        snapshot_load_bytes = 0
        snapshot_loaded = False
        if not isinstance(model_state_dict, Mapping):
            snapshot = owner_ref.get("model_state_snapshot")
            if not isinstance(snapshot, Mapping):
                raise RuntimeError(
                    "owner-search model owner ref lacks model_state_dict or model_state_snapshot"
                )
            model_state_dict, load_metadata = _load_model_state_snapshot_file(snapshot)
            snapshot_loaded = True
            snapshot_load_sec = float(load_metadata.get("load_sec", 0.0) or 0.0)
            snapshot_load_bytes = int(load_metadata.get("bytes", 0) or 0)
        state = self._inner.refresh_model_state(
            model_state_dict=model_state_dict,
            policy_version_ref=str(policy_version_ref),
            model_version_ref=str(model_version_ref),
            policy_source=str(policy_source),
            learner_update_count=int(learner_update_count),
            expected_model_state_digest=expected_model_state_digest,
        )
        if snapshot_loaded:
            state = dict(state)
            state["model_state_snapshot_loaded"] = True
            state["model_state_snapshot_load_sec"] = float(snapshot_load_sec)
            state["model_state_snapshot_load_bytes"] = int(snapshot_load_bytes)
        return state


def _build_owner_search_slab_proxy(
    *,
    args: argparse.Namespace,
    model: Any | None,
    device: str,
    loaded_checkpoint_identity: Mapping[str, Any],
    inline: bool = False,
    inline_background: bool = False,
    threaded: bool = False,
) -> CompactLazyOwnerSearchSlabProxyV1:
    inner_kind = _owner_search_inner_search_service_kind(args)
    if inner_kind != SEARCH_SERVICE_COMPACT_TORCH:
        raise ValueError(
            "owner-search speed-row replay/train wiring requires compact_torch inner search"
        )
    policy_version_ref = str(
        loaded_checkpoint_identity.get("policy_version_ref") or f"{args.run_id}:policy"
    )
    model_version_ref = str(
        loaded_checkpoint_identity.get("model_version_ref") or f"{args.run_id}:model"
    )
    policy_source = str(
        loaded_checkpoint_identity.get("policy_source") or "compact_coach_speed_row_smoke"
    )
    root_count = int(args.batch_size) * 2
    search_service_factory = _build_compact_torch_owner_search_service
    search_service_factory_kwargs = {
        "policy_model": model,
        "num_simulations": int(args.num_simulations),
        "device": str(device),
        "seed": int(args.seed),
        "compact_torch_request_compile": bool(args.compact_torch_request_compile),
        "compact_torch_request_model_compile": bool(args.compact_torch_request_model_compile),
        "compact_torch_model_compile_mode": str(
            getattr(args, "compact_torch_model_compile_mode", "reduce-overhead")
        ),
        "compact_torch_timing_mode": str(args.compact_torch_timing_mode),
        "compact_torch_initial_inference_mode": str(args.compact_torch_initial_inference_mode),
        "compact_torch_observation_memory_format": str(
            getattr(args, "compact_torch_observation_memory_format", "contiguous")
        ),
        "compact_torch_model_memory_format": str(
            getattr(args, "compact_torch_model_memory_format", "contiguous")
        ),
        "compact_torch_defer_one_simulation_replay_payload": bool(
            getattr(args, "compact_torch_defer_one_simulation_replay_payload", False)
        ),
        "loaded_checkpoint_identity": dict(loaded_checkpoint_identity),
        "run_id": str(args.run_id),
    }
    root_provider_factory = build_compact_resident_shared_memory_root_provider_v1
    root_provider_factory_kwargs = {
        "device": str(device),
        "source_backend": "owner_search_shared_memory_root_to_resident_tensor_v1",
    }
    use_inner_two_phase_device_replay = inner_kind == SEARCH_SERVICE_COMPACT_TORCH
    bridge_ready = not (bool(inline) or bool(threaded))
    direct_transition_batch_replay = bool(
        getattr(args, "owner_search_direct_transition_batch_replay", False)
    )
    fixed_soa_replay = bool(getattr(args, "owner_search_fixed_soa_replay", False))
    if direct_transition_batch_replay and not (
        bool(getattr(args, "owner_search_slab_bypass", False))
        and int(getattr(args, "owner_search_transition_batch_size", 1)) > 1
    ):
        raise ValueError(
            "owner-search direct transition-batch replay requires slab bypass "
            "and transition batch size > 1"
        )
    if fixed_soa_replay:
        if not direct_transition_batch_replay:
            raise ValueError(
                "owner-search fixed SoA replay requires direct transition-batch replay"
            )
        if not bool(getattr(args, "compact_owned_loop_fused_learner_batch", False)):
            raise ValueError("owner-search fixed SoA replay requires fused learner batch")
        if not bool(
            getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ):
            raise ValueError("owner-search fixed SoA replay requires learner-ready unroll2 cache")
        if not bool(getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)):
            raise ValueError("owner-search fixed SoA replay requires tensor-native replay")
        if int(getattr(args, "learner_num_unroll_steps", 1) or 1) != 2:
            raise ValueError("owner-search fixed SoA replay requires learner unroll 2")
    policy_version = CompactPolicyVersionRefV1(
        policy_version_ref=policy_version_ref,
        policy_source=policy_source,
        model_version_ref=model_version_ref,
    )
    replay_store_factory_kwargs = {
        "capacity": int(args.replay_pair_capacity),
        "metadata": compact_owned_loop_replay_store_metadata(
            policy_version,
            extra={
                "compact_owner_search_worker_replay_store": True,
                "compact_owned_training_loop_owner": "owner_search_worker",
                "compact_owner_search_direct_transition_batch_replay_requested": bool(
                    direct_transition_batch_replay
                ),
                "compact_owner_search_owner_local_transition_derivation_requested": bool(
                    getattr(
                        args,
                        "owner_search_owner_local_transition_derivation",
                        False,
                    )
                ),
                "compact_owner_search_owner_proxy_transition_closure_requested": bool(
                    getattr(
                        args,
                        "owner_search_owner_proxy_transition_closure",
                        False,
                    )
                ),
                **_compact_muzero_learner_batch_metadata_flags(args),
            },
        ),
    }
    replay_store_factory = (
        _OwnerSearchDirectTransitionBatchReplayStoreFactorySidecarV1
        if direct_transition_batch_replay
        else _OwnerSearchReplayStoreFactorySidecarV1
    )
    owner_search_learner_kind = str(
        getattr(args, "owner_search_learner_kind", OWNER_SEARCH_LEARNER_COMPACT_MUZERO)
        or OWNER_SEARCH_LEARNER_COMPACT_MUZERO
    )
    if owner_search_learner_kind == OWNER_SEARCH_LEARNER_MOCK_FAST:
        learner_factory = _OwnerSearchFastMockLearnerFactorySidecarV1
    elif owner_search_learner_kind == OWNER_SEARCH_LEARNER_COMPACT_MUZERO:
        learner_factory = _OwnerSearchMuZeroLearnerFactorySidecarV1
    else:
        raise ValueError(f"unknown owner-search learner kind: {owner_search_learner_kind}")
    async_learner_worker_kind = str(
        getattr(
            args,
            "owner_search_async_learner_worker_kind",
            COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
        )
    )
    model_owner_ref_payload_kind = COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1
    if (
        bool(getattr(args, "owner_search_async_learner_worker", False))
        and async_learner_worker_kind
        == COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH
    ):
        model_owner_ref_payload_kind = COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1
    shared_model_state_with_search = (
        bool(inline)
        and not bool(threaded)
        and not bool(getattr(args, "owner_search_async_learner_worker", False))
    )
    if shared_model_state_with_search:
        model_owner_ref_payload_kind = OWNER_SEARCH_MODEL_STATE_TRANSPORT_SHARED_MODEL_V1
        learner_model = model if model is not None else _TinyMuZero()
    else:
        learner_model = _owner_search_host_only_model_for_process_worker(
            model if model is not None else _TinyMuZero()
        )
    learner_factory_kwargs = {
        "model": learner_model,
        "seed": int(args.seed) + 104729,
        "device": str(args.learner_device),
        "support_scale": int(loaded_checkpoint_identity.get("support_scale") or 1),
        "num_unroll_steps": _learner_num_unroll_steps(args),
        "model_owner_ref_payload_kind": str(model_owner_ref_payload_kind),
        "shared_model_state_with_search": bool(shared_model_state_with_search),
        "host_clone_learner_payload": bool(
            getattr(args, "owner_search_async_learner_worker", False)
            and async_learner_worker_kind
            == COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH
        ),
        "defer_model_state_digest_to_search_refresh": bool(
            getattr(args, "owner_search_defer_model_state_digest_to_refresh", False)
        ),
    }
    if sum(bool(value) for value in (inline, inline_background, threaded)) > 1:
        raise ValueError(
            "owner-search proxy cannot combine inline, inline-background, and threaded"
        )
    if bool(threaded):
        proxy_cls = CompactLazyThreadedOwnerSearchSlabProxyV1
    elif bool(inline_background):
        proxy_cls = CompactLazyInlineBackgroundOwnerSearchSlabProxyV1
    elif bool(inline):
        proxy_cls = CompactLazyInlineOwnerSearchSlabProxyV1
    else:
        proxy_cls = CompactLazyOwnerSearchSlabProxyV1
    return proxy_cls(
        search_service_factory=search_service_factory,
        search_service_factory_kwargs=search_service_factory_kwargs,
        root_provider_factory=root_provider_factory,
        root_provider_factory_kwargs=root_provider_factory_kwargs,
        replay_store_factory=replay_store_factory,
        replay_store_factory_kwargs=replay_store_factory_kwargs,
        learner_factory=learner_factory,
        learner_factory_kwargs=learner_factory_kwargs,
        owner_replay_append_enabled=True,
        owner_sample_batch_size=int(args.sample_batch_size),
        owner_train_steps=int(args.learner_train_steps),
        owner_train_interval=_owner_search_train_interval(args),
        owner_model_refresh_interval=max(
            1,
            int(getattr(args, "policy_refresh_interval", 1)),
        ),
        owner_expected_train_request_count=_owner_search_expected_train_request_count(args),
        owner_defer_maintenance=bool(getattr(args, "owner_search_defer_maintenance", False)),
        async_learner_worker=bool(getattr(args, "owner_search_async_learner_worker", False)),
        async_learner_worker_kind=str(async_learner_worker_kind),
        async_learner_max_pending=int(getattr(args, "owner_search_async_learner_max_pending", 1)),
        require_resident_root_view=bool(
            getattr(args, "owner_search_require_resident_root_view", False)
        ),
        fixed_action_result_buffer=bool(
            getattr(args, "owner_search_fixed_action_result_buffer", False)
        ),
        action_result_slot_capacity=int(
            getattr(args, "owner_search_action_result_slot_capacity", 4) or 4
        ),
        use_inner_two_phase_device_replay=bool(use_inner_two_phase_device_replay),
        policy_version_ref=policy_version_ref,
        model_version_ref=model_version_ref,
        policy_source=policy_source,
        root_store_capacity=root_count,
        root_store_metadata={
            "compact_owner_search_inline_slab_proxy": bool(inline),
            "compact_owner_search_inline_background_slab_proxy": bool(inline_background),
            "compact_owner_search_threaded_slab_proxy": bool(threaded),
            "owner_search_inner_search_service_kind": inner_kind,
            "owner_search_inner_search_service_impl": _search_service_impl(inner_kind),
            "owner_search_compact_torch_resident_root_bridge_ready": bridge_ready,
            "owner_search_defer_model_state_digest_to_refresh": bool(
                getattr(args, "owner_search_defer_model_state_digest_to_refresh", False)
            ),
            "owner_search_owner_proxy_transition_closure": bool(
                getattr(args, "owner_search_owner_proxy_transition_closure", False)
            ),
            "owner_search_fixed_action_result_buffer": bool(
                getattr(args, "owner_search_fixed_action_result_buffer", False)
            ),
            "owner_search_action_result_slot_capacity": int(
                getattr(args, "owner_search_action_result_slot_capacity", 4) or 4
            ),
        },
    )


def _build_compact_torch_owner_search_service(
    *,
    policy_model: Any | None,
    num_simulations: int,
    device: str,
    seed: int,
    compact_torch_request_compile: bool,
    compact_torch_request_model_compile: bool,
    compact_torch_model_compile_mode: str,
    compact_torch_timing_mode: str,
    compact_torch_initial_inference_mode: str,
    compact_torch_observation_memory_format: str,
    compact_torch_model_memory_format: str,
    compact_torch_defer_one_simulation_replay_payload: bool = False,
    loaded_checkpoint_identity: Mapping[str, Any] | None = None,
    run_id: str = "owner-search-compact-torch",
) -> _OwnerSearchCompactTorchServiceSidecarV1:
    import torch

    torch.manual_seed(int(seed))
    target_device = torch.device(str(device))
    search_model = policy_model if policy_model is not None else _TinyMuZero()
    model_to = getattr(search_model, "to", None)
    if callable(model_to):
        search_model = model_to(target_device)
    service = CompactTorchSearchServiceV1(
        policy=_ModelPolicy(search_model),
        num_simulations=int(num_simulations),
        device=target_device,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=bool(compact_torch_request_compile),
            request_model_compile=bool(compact_torch_request_model_compile),
            model_compile_mode=str(compact_torch_model_compile_mode),
            require_cuda_device=False,
            require_torch_compile=False,
            require_model_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            recurrent_action_shape_mode="auto",
            timing_mode=str(compact_torch_timing_mode),
            initial_inference_mode=str(compact_torch_initial_inference_mode),
            observation_memory_format=str(compact_torch_observation_memory_format),
            model_memory_format=str(compact_torch_model_memory_format),
            defer_one_simulation_replay_payload=bool(
                compact_torch_defer_one_simulation_replay_payload
            ),
        ),
        require_resident_observation=True,
    )
    identity = dict(loaded_checkpoint_identity or {})
    if identity:
        service.refresh_model_state(
            model_state_dict=search_model.state_dict(),
            policy_version_ref=str(identity.get("policy_version_ref") or f"{run_id}:policy"),
            model_version_ref=str(identity.get("model_version_ref") or f"{run_id}:model"),
            policy_source=str(identity.get("policy_source") or "compact_coach_speed_row_smoke"),
            learner_update_count=max(1, int(identity.get("learner_update_count") or 1)),
            expected_model_state_digest=compact_model_state_digest_v1(search_model),
        )
    return _OwnerSearchCompactTorchServiceSidecarV1(service)


def _build_fixed_shape_owner_search_service(
    *,
    root_count: int,
    num_simulations: int,
    device: str,
) -> FixedShapeBatchedSearchOwnerV0:
    return FixedShapeBatchedSearchOwnerV0(
        root_count=int(root_count),
        num_simulations=int(num_simulations),
        device=str(device),
        model=_TinyMuZero(),
    )


def _speed_summary_and_compact_payload(
    *,
    args: argparse.Namespace,
    candidate_checkpoint_id: str,
    profile_payload: dict[str, Any],
    loaded_checkpoint_identity: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _owner_search_owner_train_enabled(args):
        _require_owner_search_slab_proxy_proof(profile_payload, args=args)
    _require_profile_support(args, profile_payload)
    _attach_normal_death_contract_fields(
        args=args,
        candidate_checkpoint_id=candidate_checkpoint_id,
        profile_payload=profile_payload,
    )
    _require_fused_learner_batch_proof(args, profile_payload)
    if str(args.search_service_kind) == SEARCH_SERVICE_COMPACT_TORCH:
        _require_compact_torch_policy_refresh_proof(profile_payload)
    _require_lean_trainer_counter_proof(args, profile_payload)
    _require_deferred_learner_proof(args, profile_payload)
    _require_deferred_sample_learner_proof(args, profile_payload)
    if not _owner_search_owner_train_enabled(args):
        _require_owner_search_slab_proxy_proof(profile_payload, args=args)
    env_steps_collected = float(profile_payload["steps"]) * float(profile_payload["batch_size"])
    training_wall_sec = float(profile_payload["measured_sec"])
    if env_steps_collected <= 0.0 or training_wall_sec <= 0.0:
        raise ValueError("speed row requires positive env steps and measured wall sec")
    steps_per_sec = env_steps_collected / training_wall_sec
    if not math.isfinite(steps_per_sec) or steps_per_sec <= 0.0:
        raise ValueError("speed row produced invalid steps/sec")
    bounded_diagnostics = bool(getattr(args, "compact_profile_bounded_diagnostics", False))
    cuda_sync_timing_diagnostics = bool(
        getattr(args, "compact_profile_cuda_sync_timing_diagnostics", False)
    )
    runtime_step_timing_diagnostics = bool(
        getattr(args, "compact_profile_runtime_step_timing_diagnostics", False)
        or cuda_sync_timing_diagnostics
    )
    timing_projection = _speed_timing_projection_fields(
        profile_payload,
        training_wall_sec=training_wall_sec,
    )
    slab_total_projection = _compact_rollout_slab_telemetry_total_fields(profile_payload)
    whole_owner_buffer_ceiling_projection = _whole_owner_buffer_replay_ceiling_fields(
        profile_payload,
        training_wall_sec=training_wall_sec,
        env_steps_collected=env_steps_collected,
        observed_steps_per_sec=steps_per_sec,
    )
    summary = {
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "status": "complete",
        "ok": True,
        "row_id": ROW_ID,
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        "row_purpose": "coach_speed_row",
        "promotion_claim": False,
        "speed_currency": SPEED_CURRENCY,
        "compact_profile_bounded_diagnostics": bounded_diagnostics,
        "compact_profile_cuda_sync_timing_diagnostics": cuda_sync_timing_diagnostics,
        "compact_profile_runtime_step_timing_diagnostics": runtime_step_timing_diagnostics,
        "source_profile_payload_embedded": not bounded_diagnostics,
        "search_service_kind": str(args.search_service_kind),
        "search_service_impl": _search_service_impl(str(args.search_service_kind)),
        "compact_owned_loop_deferred_learner": bool(
            getattr(args, "compact_owned_loop_deferred_learner", False)
        ),
        "compact_owned_loop_deferred_sample_learner": bool(
            getattr(args, "compact_owned_loop_deferred_sample_learner", False)
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_requested": int(
            getattr(args, "compact_owned_loop_deferred_sample_learner_max_pending", 1)
        ),
        "compact_owned_loop_sample_learner_worker_kind_requested": str(
            getattr(
                args,
                "compact_owned_loop_sample_learner_worker_kind",
                COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD,
            )
        ),
        "compact_owned_loop_fused_learner_batch": bool(
            getattr(args, "compact_owned_loop_fused_learner_batch", False)
        ),
        UNROLL2_SPECIALIZED_BUILDER_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_unroll2_specialized_builder", False)
        ),
        LEARNER_READY_UNROLL2_CACHE_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ),
        TENSOR_NATIVE_REPLAY_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
        ),
        **_compact_owned_runner_fields(args),
        **_deferred_learner_proof_fields(profile_payload),
        **_deferred_sample_learner_proof_fields(profile_payload),
        **_owner_search_slab_proxy_proof_fields(profile_payload),
        **_repeatability_work_shape_fields(args=args, profile_payload=profile_payload),
        **_compact_loop_counter_fields(profile_payload),
        **_compact_owned_trainer_counter_fields(profile_payload),
        "learner_num_unroll_steps": _learner_num_unroll_steps(args),
        **_operational_surface_fields(args, profile_payload),
        **timing_projection,
        **slab_total_projection,
        **whole_owner_buffer_ceiling_projection,
        **_sample_learner_fusion_fields(profile_payload),
        **_runtime_step_diagnostic_fields(profile_payload),
        **_normal_death_contract_fields(profile_payload),
        **_render_state_handoff_fields(args, profile_payload),
        **_compact_torch_memory_format_fields(args),
        **_owner_search_config_fields(args),
        **_gpu_utilization_sampling_fields(profile_payload),
        "search_service_floor_decomposition_role": _search_service_floor_role(
            str(args.search_service_kind)
        ),
        **_policy_refresh_proof_fields(profile_payload),
        "env_steps_collected": env_steps_collected,
        "training_wall_sec": training_wall_sec,
        "compact_trainer_env_steps_per_sec": steps_per_sec,
        "steps_per_sec": steps_per_sec,
        "source_profile_steps_per_sec": profile_payload.get("steps_per_sec"),
        "source_profile_physical_rows_per_sec": profile_payload.get("physical_rows_per_sec"),
        "source_profile_support_profile_only": profile_payload.get("profile_only"),
        "source_profile_support_schema_id": profile_payload.get("schema_id"),
        "compact_owned_lean_profile_oracle": profile_payload.get(
            "compact_owned_lean_profile_oracle"
        ),
        "non_claims": _non_claims(),
    }
    loaded_identity = dict(loaded_checkpoint_identity or {})
    model_identity_scope = (
        COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT
        if loaded_identity
        else COMPACT_COACH_MODEL_IDENTITY_SCOPE_SUPPORT_ONLY
    )
    compact_payload = {
        "ok": True,
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "route": COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "real_compact_owned_training_work": True,
        "compact_profile_bounded_diagnostics": bounded_diagnostics,
        "compact_profile_cuda_sync_timing_diagnostics": cuda_sync_timing_diagnostics,
        "compact_profile_runtime_step_timing_diagnostics": runtime_step_timing_diagnostics,
        "source_profile_payload_embedded": not bounded_diagnostics,
        "compact_owned_loop_deferred_learner": bool(
            getattr(args, "compact_owned_loop_deferred_learner", False)
        ),
        "compact_owned_loop_deferred_sample_learner": bool(
            getattr(args, "compact_owned_loop_deferred_sample_learner", False)
        ),
        "compact_owned_loop_deferred_sample_learner_max_pending_requested": int(
            getattr(args, "compact_owned_loop_deferred_sample_learner_max_pending", 1)
        ),
        "compact_owned_loop_sample_learner_worker_kind_requested": str(
            getattr(
                args,
                "compact_owned_loop_sample_learner_worker_kind",
                COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD,
            )
        ),
        "compact_owned_loop_fused_learner_batch": bool(
            getattr(args, "compact_owned_loop_fused_learner_batch", False)
        ),
        UNROLL2_SPECIALIZED_BUILDER_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_unroll2_specialized_builder", False)
        ),
        LEARNER_READY_UNROLL2_CACHE_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_learner_ready_unroll2_cache", False)
        ),
        TENSOR_NATIVE_REPLAY_KEY: bool(
            getattr(args, "compact_muzero_learner_batch_tensor_native_replay", False)
        ),
        **_compact_owned_runner_fields(args),
        "compact_owned_lean_profile_oracle": profile_payload.get(
            "compact_owned_lean_profile_oracle"
        ),
        **_deferred_learner_proof_fields(profile_payload),
        **_deferred_sample_learner_proof_fields(profile_payload),
        **_owner_search_slab_proxy_proof_fields(profile_payload),
        **_compact_owned_trainer_counter_fields(profile_payload),
        **_compact_loop_counter_fields(profile_payload),
        **_repeatability_work_shape_fields(args=args, profile_payload=profile_payload),
        "compact_owned_trainer_env_step_source": (
            "local_hybrid_profile_physical_env_rows"
            if not loaded_identity
            else "loaded_lifecycle_checkpoint_hybrid_profile_physical_env_rows"
        ),
        "search_service_kind": str(args.search_service_kind),
        "search_service_impl": _search_service_impl(str(args.search_service_kind)),
        "learner_num_unroll_steps": _learner_num_unroll_steps(args),
        **_operational_surface_fields(args, profile_payload),
        **timing_projection,
        **slab_total_projection,
        **whole_owner_buffer_ceiling_projection,
        **_sample_learner_fusion_fields(profile_payload),
        **_runtime_step_diagnostic_fields(profile_payload),
        **_normal_death_contract_fields(profile_payload),
        **_render_state_handoff_fields(args, profile_payload),
        **_compact_torch_memory_format_fields(args),
        **_owner_search_config_fields(args),
        **_gpu_utilization_sampling_fields(profile_payload),
        "search_service_floor_decomposition_role": _search_service_floor_role(
            str(args.search_service_kind)
        ),
        "model_identity_scope": model_identity_scope,
        "loaded_checkpoint_identity": loaded_identity,
        "compact_owned_loop_entrypoint_enabled": profile_payload[
            "compact_owned_loop_entrypoint_enabled"
        ],
        "compact_owned_loop_profile_only_support": profile_payload[
            "compact_owned_loop_profile_only"
        ],
        "compact_rollout_slab_learner_gate_impl": profile_payload[
            "compact_rollout_slab_learner_gate_impl"
        ],
        "compact_rollout_slab_learner_gate_real_muzero_update": profile_payload[
            "compact_rollout_slab_learner_gate_real_muzero_update"
        ],
        **_compact_loop_counter_fields(profile_payload),
        **_policy_refresh_proof_fields(profile_payload),
        "env_steps_collected": env_steps_collected,
        "training_wall_sec": training_wall_sec,
        "compact_trainer_env_steps_per_sec": steps_per_sec,
        "steps_per_sec": steps_per_sec,
        "non_claims": _non_claims(),
    }
    if bounded_diagnostics:
        compact_payload["source_profile_payload_omitted_reason"] = (
            "compact_profile_bounded_diagnostics"
        )
    else:
        compact_payload["source_profile_payload"] = profile_payload
    return summary, compact_payload


def _load_lifecycle_checkpoint_model(
    *,
    args: argparse.Namespace,
    lifecycle: Mapping[str, Any],
    lifecycle_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    checkpoint_path_raw = str(lifecycle.get("compact_checkpoint_path") or "").strip()
    if not checkpoint_path_raw:
        raise ValueError("unified lifecycle report must carry compact_checkpoint_path")
    checkpoint_path = _resolve_artifact_path(
        checkpoint_path_raw,
        base_dir=lifecycle_path.parent,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"compact checkpoint not found: {checkpoint_path}")
    checkpoint = load_compact_trainer_checkpoint_v1(checkpoint_path)
    model = _build_stock_lightzero_model_for_checkpoint(args=args, output_dir=output_dir)
    restore_compact_trainer_checkpoint_v1(checkpoint, model=model)
    metadata = dict(getattr(checkpoint, "metadata", {}) or {})
    support_scale = _support_scale_from_checkpoint_metadata(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    identity = {
        "scope": COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT,
        "identity_source": "loaded_compact_trainer_checkpoint",
        "candidate_loaded_checkpoint": True,
        "compact_checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_id": str(metadata.get("checkpoint_id") or ""),
        "trainer_id": str(metadata.get("trainer_id") or ""),
        "policy_version_ref": str(metadata.get("policy_version_ref") or ""),
        "model_version_ref": str(metadata.get("model_version_ref") or ""),
        "policy_source": str(metadata.get("policy_source") or ""),
        "learner_update_count": int(metadata.get("learner_update_count") or 0),
        "model_state_digest": compact_model_state_digest_v1(model),
        "support_scale": int(support_scale),
    }
    if not bool(args.omit_loaded_checkpoint_identity_path):
        identity["compact_checkpoint_path"] = str(checkpoint_path)
    if str(identity["checkpoint_id"]) != str(lifecycle.get("checkpoint_id") or ""):
        raise ValueError("loaded checkpoint_id does not match lifecycle checkpoint_id")
    return {"model": model, "identity": identity}


def _build_stock_lightzero_model_for_checkpoint(
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> Any:
    import copy
    from ding.config import compile_config
    from lzero.policy.muzero import MuZeroPolicy

    from curvyzero.infra.modal import lightzero_curvytron_visual_survival_eval as eval_mod
    from curvyzero.infra.modal import (
        lightzero_curvyzero_stacked_debug_visual_survival_train as train_mod,
    )

    decision_ms = float(args.decision_source_frames) * float(args.source_physics_step_ms)
    patched = eval_mod._build_visual_survival_configs(
        seed=int(args.seed),
        exp_name=output_dir / "loaded_checkpoint_model_build_exp",
        telemetry_path=output_dir / "loaded_checkpoint_model_build_telemetry.jsonl",
        cuda=False,
        max_env_step=int(args.source_max_steps),
        source_max_steps=int(args.source_max_steps),
        decision_ms=decision_ms,
        decision_source_frames=int(args.decision_source_frames),
        source_physics_step_ms=float(args.source_physics_step_ms),
        source_max_steps_semantics=str(args.source_max_steps_semantics),
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        n_episode=1,
        num_simulations=int(args.num_simulations),
        batch_size=int(args.batch_size),
        lightzero_eval_freq=0,
        lightzero_multi_gpu=False,
        max_train_iter=1,
        save_ckpt_after_iter=1,
        env_variant=train_mod.ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT,
        reward_variant=train_mod.REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
        reward_outcome_alpha=train_mod.DEFAULT_REWARD_OUTCOME_ALPHA,
        ego_action_straight_override_probability=0.0,
        control_noise_profile_id=train_mod.DEFAULT_CONTROL_NOISE_PROFILE_ID,
        disable_death_for_profile=False,
        env_telemetry_stride=1,
        env_manager_type="base",
        opponent_policy_kind=train_mod.OPPONENT_POLICY_KIND_FIXED_STRAIGHT,
        opponent_use_cuda=False,
        opponent_checkpoint=None,
        opponent_snapshot_ref=None,
        opponent_checkpoint_state_key=None,
        natural_bonus_spawn=True,
        opponent_death_mode=train_mod.DEFAULT_OPPONENT_DEATH_MODE,
        opponent_runtime_mode=train_mod.DEFAULT_OPPONENT_RUNTIME_MODE,
    )
    cfg = compile_config(
        copy.deepcopy(patched["main_config"]),
        seed=int(args.seed),
        auto=True,
        create_cfg=copy.deepcopy(patched["create_config"]),
        save_cfg=False,
    )
    cfg.policy.cuda = False
    cfg.policy.device = "cpu"
    policy = MuZeroPolicy(cfg.policy)
    model = getattr(policy, "_model", None)
    if model is None:
        raise AttributeError("MuZeroPolicy has no _model attribute")
    return model


def _support_scale_from_checkpoint_metadata(
    *,
    checkpoint: Any,
    checkpoint_path: Path,
) -> int:
    trainer_config = getattr(checkpoint, "trainer_config", {}) or {}
    if isinstance(trainer_config, Mapping):
        surface_raw = str(trainer_config.get("model_surface_path") or "").strip()
        if surface_raw:
            surface_path = _resolve_artifact_path(
                surface_raw,
                base_dir=checkpoint_path.parent,
            )
            if surface_path.is_file():
                surface = _load_json(surface_path)
                policy = surface.get("policy")
                if isinstance(policy, Mapping) and policy.get("support_scale") is not None:
                    return int(policy["support_scale"])
    return 300


def _require_profile_support(
    args: argparse.Namespace,
    profile_payload: dict[str, Any],
) -> None:
    owner_search_train = _owner_search_owner_train_enabled(args)
    expected = {
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    if not owner_search_train:
        expected.update(
            {
                "compact_owned_loop_entrypoint_enabled": True,
                "compact_rollout_slab_learner_gate_real_muzero_update": True,
            }
        )
    for key, value in expected.items():
        if profile_payload.get(key) is not value:
            raise ValueError(f"profile support {key} must be {value!r}")
    if profile_payload.get("compact_rollout_slab_learner_gate_impl") != (
        COMPACT_ROLLOUT_SLAB_LEARNER_GATE_IMPL_COMPACT_MUZERO
    ):
        raise ValueError("profile support must use compact_muzero learner gate")
    for key in (
        "measured_sec",
        "steps",
        "batch_size",
    ):
        value = float(profile_payload.get(key) or 0.0)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"profile support {key} must be positive")
    if owner_search_train:
        if profile_payload.get("compact_owned_loop_entrypoint_enabled") is not False:
            raise ValueError("owner-search row must disable parent compact-owned loop")
        if int(profile_payload.get("compact_rollout_slab_learner_gate_updates") or 0) != 0:
            raise ValueError("owner-search row must not run parent learner updates")
        if int(profile_payload.get("compact_rollout_slab_sample_gate_calls") or 0) != 0:
            raise ValueError("owner-search row must not run parent sample gate")
        owner_materializes_replay = bool(
            profile_payload.get("compact_owner_search_owner_materializes_replay", False)
        )
        if (
            not owner_materializes_replay
            and int(profile_payload.get("compact_rollout_slab_committed_index_row_count") or 0) <= 0
        ):
            raise ValueError("owner-search row must still commit slab replay rows")
    else:
        for key in (
            "compact_rollout_slab_learner_gate_calls",
            "compact_rollout_slab_learner_gate_updates",
            "compact_rollout_slab_sample_gate_calls",
        ):
            value = float(profile_payload.get(key) or 0.0)
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"profile support {key} must be positive")
    if bool(profile_payload.get("compact_owned_loop_deferred_learner_pending", False)):
        raise ValueError("profile support must drain deferred learner work")
    if bool(profile_payload.get("compact_owned_loop_deferred_sample_learner_pending", False)):
        raise ValueError("profile support must drain deferred sample+learner work")
    if str(profile_payload.get("death_mode") or "") == vector_runtime.DEATH_MODE_NORMAL:
        num_unroll_steps = int(
            profile_payload.get("compact_rollout_slab_learner_gate_num_unroll_steps") or 0
        )
        if num_unroll_steps <= 1:
            raise ValueError("normal-death speed rows require learner unroll steps greater than 1")


def _require_owner_search_slab_proxy_proof(
    profile_payload: Mapping[str, Any],
    *,
    args: argparse.Namespace | None = None,
) -> None:
    if profile_payload.get("compact_owner_search_slab_proxy") is not True:
        if profile_payload.get("owner_search_slab_proxy_requested") is True:
            raise ValueError("owner-search row requested but proof fields are missing")
        return
    boundary_kind = str(profile_payload.get("compact_owner_search_boundary_kind") or "")
    if boundary_kind not in {
        "worker_search_parent_slab_commit",
        "inline_owner_search_parent_slab_commit",
        "inline_background_owner_search_parent_slab_commit",
        "threaded_owner_search_parent_slab_commit",
    }:
        raise ValueError("owner-search row must report a known owner-search boundary kind")
    inline_boundary = boundary_kind == "inline_owner_search_parent_slab_commit"
    inline_background_boundary = (
        boundary_kind == "inline_background_owner_search_parent_slab_commit"
    )
    threaded_boundary = boundary_kind == "threaded_owner_search_parent_slab_commit"
    if (
        inline_boundary
        and profile_payload.get("compact_owner_search_inline_slab_proxy") is not True
    ):
        raise ValueError("inline owner-search row must prove inline slab proxy mode")
    if (
        inline_background_boundary
        and profile_payload.get("compact_owner_search_inline_background_slab_proxy") is not True
    ):
        raise ValueError(
            "inline-background owner-search row must prove inline-background slab proxy mode"
        )
    if (
        threaded_boundary
        and profile_payload.get("compact_owner_search_threaded_slab_proxy") is not True
    ):
        raise ValueError("threaded owner-search row must prove threaded slab proxy mode")
    if inline_background_boundary:
        if str(profile_payload.get("compact_owner_search_worker_kind") or "") != (
            "inline_background_owner_search_v1"
        ):
            raise ValueError(
                "inline-background owner-search row must prove inline-background worker kind"
            )
        if str(profile_payload.get("compact_owner_search_worker_resource_scope") or "") != (
            "inline_process_background_maintenance_thread"
        ):
            raise ValueError(
                "inline-background owner-search row must prove background maintenance scope"
            )
    if threaded_boundary:
        if str(profile_payload.get("compact_owner_search_worker_kind") or "") != (
            "threaded_owner_search_v1"
        ):
            raise ValueError("threaded owner-search row must prove threaded worker kind")
        if str(profile_payload.get("compact_owner_search_worker_resource_scope") or "") != (
            "colocated_thread"
        ):
            raise ValueError("threaded owner-search row must prove colocated thread scope")
    if profile_payload.get("compact_owner_search_worker_owns_search_state") is not True:
        raise ValueError("owner-search row must prove worker-owned search state")
    action_only_result = bool(profile_payload.get("compact_owner_search_action_only_result", False))
    owner_replay_append_enabled = bool(
        profile_payload.get("compact_owner_search_owner_replay_append_enabled", False)
    )
    compact_torch_owner_search = (
        str(profile_payload.get("owner_search_inner_search_service_kind") or "")
        == SEARCH_SERVICE_COMPACT_TORCH
        or profile_payload.get("owner_search_compact_torch_resident_root_bridge_ready") is True
    )
    if not owner_replay_append_enabled:
        raise ValueError("owner-search row must enable owner replay/train")
    if int(profile_payload.get("compact_rollout_slab_learner_gate_updates") or 0) != 0:
        raise ValueError("owner-search row must not run parent learner updates")
    if int(profile_payload.get("compact_rollout_slab_learner_gate_calls") or 0) != 0:
        raise ValueError("owner-search row must not call parent learner gate")
    if int(profile_payload.get("compact_rollout_slab_learner_gate_sample_row_count") or 0) != 0:
        raise ValueError("owner-search row must not sample parent learner rows")
    if int(profile_payload.get("compact_rollout_slab_sample_gate_calls") or 0) != 0:
        raise ValueError("owner-search row must not run parent sample gate")
    if int(profile_payload.get("compact_rollout_slab_sample_gate_sample_row_count") or 0) != 0:
        raise ValueError("owner-search row must not sample parent replay rows")
    if args is not None and bool(getattr(args, "owner_search_slab_bypass", False)):

        def _require_bypass_field(name: str) -> Any:
            if name not in profile_payload:
                raise ValueError(f"owner-search slab-bypass row must report {name}")
            return profile_payload[name]

        if profile_payload.get("compact_owner_search_slab_bypass") is not True:
            raise ValueError("owner-search slab-bypass row must prove bypass mode")
        if str(profile_payload.get("compact_owner_search_slab_bypass_kind") or "") != (
            COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
        ):
            raise ValueError("owner-search slab-bypass row must prove direct-stepper kind")
        if profile_payload.get("compact_rollout_slab_bypassed") is not True:
            raise ValueError("owner-search slab-bypass row must mark slab as bypassed")
        if (
            _require_bypass_field("compact_rollout_slab_general_replay_row_builder_used")
            is not False
        ):
            raise ValueError(
                "owner-search slab-bypass row must not use the general replay-row builder"
            )
        if int(_require_bypass_field("compact_rollout_slab_committed_index_row_count")) != 0:
            raise ValueError("owner-search slab-bypass row must commit zero parent rows")
        if int(_require_bypass_field("compact_rollout_slab_stored_index_row_count")) != 0:
            raise ValueError("owner-search slab-bypass row must store zero parent rows")
        if _require_bypass_field("compact_rollout_slab_retains_committed_index_rows") is not False:
            raise ValueError("owner-search slab-bypass row must not retain parent rows")
        if (
            int(
                _require_bypass_field(
                    "compact_owner_search_slab_bypass_parent_committed_index_rows"
                )
                or 0
            )
            != 0
        ):
            raise ValueError("owner-search slab-bypass parent committed proof must be zero")
        if (
            int(
                _require_bypass_field("compact_owner_search_slab_bypass_parent_stored_index_rows")
                or 0
            )
            != 0
        ):
            raise ValueError("owner-search slab-bypass parent stored proof must be zero")
    transition_batch_requested = bool(
        args is not None
        and bool(getattr(args, "owner_search_slab_bypass", False))
        and int(getattr(args, "owner_search_transition_batch_size", 1)) > 1
    )
    owner_local_transition_derivation_requested = bool(
        args is not None
        and bool(getattr(args, "owner_search_owner_local_transition_derivation", False))
    )
    owner_proxy_transition_closure_requested = bool(
        args is not None
        and bool(getattr(args, "owner_search_owner_proxy_transition_closure", False))
    )
    if transition_batch_requested:
        owner_sample_telemetry_for_proof = (
            _owner_search_normalized_owner_sample_telemetry_for_proof(profile_payload)
        )

        def _require_transition_batch_field(name: str) -> Any:
            owner_sample_first = bool(
                owner_local_transition_derivation_requested
                and (
                    str(name).startswith(
                        "compact_owner_search_owner_local_transition_derivation_"
                    )
                    or str(name).startswith(
                        "compact_owner_search_owner_proxy_transition_closure_"
                    )
                    or str(name).startswith("compact_owner_search_owner_proxy_")
                    or str(name).startswith(
                        "compact_owner_search_parent_previous_transition_closure"
                    )
                    or str(name)
                    == "compact_owner_search_parent_applied_action_validation_count"
                    or str(name)
                    in {
                        "compact_owner_search_owner_replay_transport_kind",
                        "compact_owner_search_owner_replay_transport_entry_count",
                        "compact_owner_search_owner_replay_transition_batch_enabled",
                        "compact_owner_search_owner_replay_transition_batch_count",
                        "compact_owner_search_owner_replay_transition_batch_transition_count",
                        "compact_owner_search_owner_replay_transition_legacy_entry_count",
                        "compact_owner_search_transition_batch_transport_requested",
                        "compact_owner_search_transition_batch_transport_enabled",
                        "compact_owner_search_transition_batch_transport_kind",
                        "compact_owner_search_transition_batch_schema_id",
                        "compact_owner_search_transition_batch_count",
                        "compact_owner_search_transition_batch_entry_count",
                        "compact_owner_search_transition_batch_transport_entry_count",
                        "compact_owner_search_transition_batch_max_entries_per_batch",
                        "compact_owner_search_transition_batch_fixed_capacity",
                        "compact_owner_search_transition_batch_padding_count",
                        "compact_owner_search_transition_batch_overflow_count",
                        "compact_owner_search_transition_batch_fallback_count",
                        "compact_owner_search_transition_batch_fallback_reason",
                        "compact_owner_search_transition_batch_pending_count",
                        "compact_owner_search_transition_batch_transport_bytes",
                        "compact_owner_search_transition_batch_digest",
                        "compact_owner_search_transition_batch_digest_verified",
                    }
                )
                and name in owner_sample_telemetry_for_proof
            )
            if owner_sample_first:
                return owner_sample_telemetry_for_proof[name]
            if name in profile_payload:
                return profile_payload[name]
            if name in owner_sample_telemetry_for_proof:
                return owner_sample_telemetry_for_proof[name]
            if name not in profile_payload:
                raise ValueError(f"owner-search transition-batch row must report {name}")
            return profile_payload[name]

        if (
            "owner_search_transition_batch_transport_requested" in profile_payload
            and profile_payload.get("owner_search_transition_batch_transport_requested") is not True
        ):
            raise ValueError("owner-search transition-batch row must report request bit")
        if (
            _require_transition_batch_field(
                "compact_owner_search_transition_batch_transport_enabled"
            )
            is not True
        ):
            raise ValueError("owner-search transition-batch row must enable transport")
        expected_transition_batch_kind = (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
            if owner_local_transition_derivation_requested
            else COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
        )
        if (
            str(
                _require_transition_batch_field(
                    "compact_owner_search_transition_batch_transport_kind"
                )
                or ""
            )
            != expected_transition_batch_kind
        ):
            raise ValueError("owner-search transition-batch row must report expected kind")
        expected_transition_batch_schema_id = (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
            if owner_local_transition_derivation_requested
            else COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
        )
        if (
            str(
                _require_transition_batch_field("compact_owner_search_transition_batch_schema_id")
                or ""
            )
            != expected_transition_batch_schema_id
        ):
            raise ValueError("owner-search transition-batch row must report schema id")
        batch_count = int(
            _require_transition_batch_field("compact_owner_search_transition_batch_count") or 0
        )
        entry_count = int(
            _require_transition_batch_field("compact_owner_search_transition_batch_entry_count")
            or 0
        )
        transport_entry_count = int(
            _require_transition_batch_field(
                "compact_owner_search_transition_batch_transport_entry_count"
            )
            or 0
        )
        if batch_count <= 0:
            raise ValueError("owner-search transition-batch row must report batches")
        if entry_count <= 0:
            raise ValueError("owner-search transition-batch row must report entries")
        if transport_entry_count != batch_count:
            raise ValueError("owner-search transition-batch transport count must match batch count")
        if transport_entry_count >= entry_count:
            raise ValueError("owner-search transition-batch row must reduce transport entries")
        if (
            int(
                _require_transition_batch_field(
                    "compact_owner_search_transition_batch_max_entries_per_batch"
                )
                or 0
            )
            <= 1
        ):
            raise ValueError("owner-search transition-batch max size must exceed one")
        if (
            int(
                _require_transition_batch_field(
                    "compact_owner_search_transition_batch_fixed_capacity"
                )
                or 0
            )
            <= 1
        ):
            raise ValueError("owner-search transition-batch capacity must exceed one")
        for name in (
            "compact_owner_search_transition_batch_overflow_count",
            "compact_owner_search_transition_batch_fallback_count",
        ):
            if int(_require_transition_batch_field(name) or 0) != 0:
                raise ValueError(f"owner-search transition-batch row must keep {name} zero")
        pending_transition_count = int(
            _require_transition_batch_field("compact_owner_search_transition_batch_pending_count")
            or 0
        )
        if pending_transition_count < 0:
            raise ValueError(
                "owner-search transition-batch row must report nonnegative pending count"
            )
        if (
            str(
                _require_transition_batch_field(
                    "compact_owner_search_transition_batch_fallback_reason"
                )
                or ""
            )
            != "none"
        ):
            raise ValueError("owner-search transition-batch row must not fallback")
        if not str(
            _require_transition_batch_field("compact_owner_search_transition_batch_digest") or ""
        ):
            raise ValueError("owner-search transition-batch row must report digest")
        if (
            _require_transition_batch_field("compact_owner_search_transition_batch_digest_verified")
            is not True
        ):
            raise ValueError("owner-search transition-batch row must verify digest")
        for name in (
            "compact_owner_search_transition_batch_padding_count",
            "compact_owner_search_transition_batch_transport_bytes",
            "compact_owner_search_transition_batch_build_sec",
            "compact_owner_search_transition_batch_submit_sec",
        ):
            value = float(_require_transition_batch_field(name) or 0.0)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"owner-search transition-batch row must report nonnegative {name}"
                )
        if (
            int(
                _require_transition_batch_field(
                    "compact_owner_search_owner_replay_transition_legacy_entry_count"
                )
                or 0
            )
            != 0
        ):
            raise ValueError("owner-search transition-batch row must avoid legacy entries")
        owner_batch_count = int(
            _require_transition_batch_field(
                "compact_owner_search_owner_replay_transition_batch_count"
            )
            or 0
        )
        owner_batch_entry_count = int(
            _require_transition_batch_field(
                "compact_owner_search_owner_replay_transition_batch_transition_count"
            )
            or 0
        )
        if owner_batch_count != batch_count:
            raise ValueError("owner-search transition-batch owner batch count mismatch")
        if owner_batch_entry_count != entry_count:
            raise ValueError("owner-search transition-batch owner entry count mismatch")
        if (
            int(
                _require_transition_batch_field(
                    "compact_owner_search_owner_replay_append_request_count"
                )
                or 0
            )
            != batch_count
        ):
            raise ValueError("owner-search transition-batch request count mismatch")
        for name in (
            "compact_owner_search_owner_replay_append_staged_entry_count",
            "compact_owner_search_owner_replay_append_submitted_entry_count",
            "compact_owner_search_replay_append_entry_count",
            "compact_owner_search_action_feedback_transition_count",
        ):
            if int(_require_transition_batch_field(name) or 0) != entry_count:
                raise ValueError(f"owner-search transition-batch logical count mismatch for {name}")
        if (
            int(
                _require_transition_batch_field(
                    "compact_owner_search_owner_replay_append_staged_transport_entry_count"
                )
                or 0
            )
            != batch_count
        ):
            raise ValueError("owner-search transition-batch staged transport mismatch")
        if (
            int(
                _require_transition_batch_field(
                    "compact_owner_search_owner_replay_append_submitted_transport_entry_count"
                )
                or 0
            )
            != batch_count
        ):
            raise ValueError("owner-search transition-batch submitted transport mismatch")
        if (
            _require_transition_batch_field(
                "compact_owner_search_transition_batch_transport_requested"
            )
            is not True
        ):
            raise ValueError("owner-search transition-batch row must report compact request bit")

        def _transition_batch_worker_counter(
            primary_name: str,
            deferred_name: str,
        ) -> int:
            deferred = int(_require_transition_batch_field(deferred_name) or 0)
            if deferred > 0:
                return deferred
            primary = int(profile_payload.get(primary_name) or 0)
            if primary > 0:
                return primary
            return deferred

        if (
            _transition_batch_worker_counter(
                "compact_owner_search_replay_append_transition_batch_count",
                "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_count",
            )
            != batch_count
        ):
            raise ValueError("owner-search worker transition-batch count mismatch")
        if (
            _transition_batch_worker_counter(
                "compact_owner_search_replay_append_transition_batch_entry_count",
                "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_entry_count",
            )
            != entry_count
        ):
            raise ValueError("owner-search worker transition-batch entry count mismatch")
        if (
            _transition_batch_worker_counter(
                "compact_owner_search_replay_append_transport_entry_count",
                "compact_owner_search_owner_maintenance_drained_replay_append_transport_entry_count",
            )
            != batch_count
        ):
            raise ValueError("owner-search worker transition-batch transport count mismatch")
        if owner_local_transition_derivation_requested:
            if (
                _require_transition_batch_field(
                    "compact_owner_search_owner_local_transition_derivation_requested"
                )
                is not True
            ):
                raise ValueError("owner-local transition derivation must report request bit")
            if (
                _require_transition_batch_field(
                    "compact_owner_search_owner_local_transition_derivation_used"
                )
                is not True
            ):
                raise ValueError("owner-local transition derivation must be used")
            if (
                str(
                    _require_transition_batch_field(
                        "compact_owner_search_owner_local_transition_derivation_schema_id"
                    )
                    or ""
                )
                != COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
            ):
                raise ValueError("owner-local transition derivation must report derived schema")
            if (
                str(
                    _require_transition_batch_field(
                        "compact_owner_search_owner_local_transition_derivation_kind"
                    )
                    or ""
                )
                != COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
            ):
                raise ValueError("owner-local transition derivation must report derived kind")
            if (
                int(
                    _require_transition_batch_field(
                        "compact_owner_search_owner_local_transition_derivation_batch_count"
                    )
                    or 0
                )
                != batch_count
            ):
                raise ValueError("owner-local transition derivation batch count mismatch")
            if (
                int(
                    _require_transition_batch_field(
                        "compact_owner_search_owner_local_transition_derivation_transition_count"
                    )
                    or 0
                )
                != entry_count
            ):
                raise ValueError("owner-local transition derivation transition count mismatch")
            if (
                int(
                    _require_transition_batch_field(
                        "compact_owner_search_owner_local_transition_derivation_transport_entry_count"
                    )
                    or 0
                )
                != batch_count
            ):
                raise ValueError("owner-local transition derivation transport count mismatch")
            if (
                int(
                    _require_transition_batch_field(
                        "compact_owner_search_owner_local_transition_derivation_cache_hit_count"
                    )
                    or 0
                )
                != entry_count
            ):
                raise ValueError("owner-local transition derivation cache-hit count mismatch")
            for name in (
                "compact_owner_search_owner_local_transition_derivation_pending_count",
                "compact_owner_search_owner_local_transition_derivation_cache_miss_count",
                "compact_owner_search_owner_local_transition_derivation_action_checksum_mismatch_count",
                "compact_owner_search_owner_local_transition_derivation_fallback_count",
                "compact_owner_search_owner_local_transition_derivation_dropped_pending_count",
                "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes",
                "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count",
            ):
                if int(_require_transition_batch_field(name) or 0) != 0:
                    raise ValueError(f"owner-local transition derivation must keep {name} zero")
            if (
                int(
                    _require_transition_batch_field(
                        "compact_owner_search_owner_local_transition_derivation_action_checksum_verified_count"
                    )
                    or 0
                )
                != entry_count
            ):
                raise ValueError("owner-local transition derivation action-check count mismatch")
            if (
                str(
                    _require_transition_batch_field(
                        "compact_owner_search_owner_local_transition_derivation_fallback_reason"
                    )
                    or ""
                )
                != "none"
            ):
                raise ValueError("owner-local transition derivation must not fallback")
            if not str(
                _require_transition_batch_field(
                    "compact_owner_search_owner_local_transition_derivation_digest"
                )
                or ""
            ):
                raise ValueError("owner-local transition derivation must report digest")
            if (
                _require_transition_batch_field(
                    "compact_owner_search_owner_local_transition_derivation_digest_verified"
                )
                is not True
            ):
                raise ValueError("owner-local transition derivation must verify digest")
            for name in (
                "compact_owner_search_owner_local_transition_derivation_transport_bytes",
                "compact_owner_search_owner_local_transition_derivation_build_sec",
                "compact_owner_search_owner_local_transition_derivation_submit_sec",
            ):
                value = float(_require_transition_batch_field(name) or 0.0)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"owner-local transition derivation must report nonnegative {name}"
                    )
            if owner_proxy_transition_closure_requested:
                if (
                    _require_transition_batch_field(
                        "compact_owner_search_owner_proxy_transition_closure_requested"
                    )
                    is not True
                ):
                    raise ValueError("owner-proxy transition closure must report request bit")
                if (
                    _require_transition_batch_field(
                        "compact_owner_search_owner_proxy_transition_closure_used"
                    )
                    is not True
                ):
                    raise ValueError("owner-proxy transition closure must be used")
                if (
                    str(
                        _require_transition_batch_field(
                            "compact_owner_search_owner_proxy_transition_closure_source"
                        )
                        or ""
                    )
                    != "owner_proxy_cached_state_v1"
                ):
                    raise ValueError("owner-proxy transition closure must report owner source")
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_owner_proxy_transition_closure_closed_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError("owner-proxy transition closure closed-count mismatch")
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_owner_proxy_transition_closure_transition_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError("owner-proxy transition closure transition-count mismatch")
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_owner_proxy_transition_closure_batch_count"
                        )
                        or 0
                    )
                    != batch_count
                ):
                    raise ValueError("owner-proxy transition closure batch-count mismatch")
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_owner_proxy_transition_closure_transport_entry_count"
                        )
                        or 0
                    )
                    != batch_count
                ):
                    raise ValueError("owner-proxy transition closure transport-count mismatch")
                for name in (
                    "compact_owner_search_owner_proxy_transition_closure_pending_count",
                    "compact_owner_search_owner_proxy_transition_closure_fallback_count",
                    "compact_owner_search_owner_proxy_applied_action_mismatch_count",
                    "compact_owner_search_parent_previous_transition_closure_count",
                    "compact_owner_search_parent_applied_action_validation_count",
                ):
                    if int(_require_transition_batch_field(name) or 0) != 0:
                        raise ValueError(f"owner-proxy transition closure must keep {name} zero")
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_owner_proxy_applied_action_verification_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError(
                        "owner-proxy transition closure applied-action verification mismatch"
                    )
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_owner_proxy_applied_action_count"
                        )
                        or 0
                    )
                    <= 0
                ):
                    raise ValueError("owner-proxy transition closure applied no actions")
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_owner_proxy_action_frame_store_count"
                        )
                        or 0
                    )
                    <= 0
                ):
                    raise ValueError("owner-proxy transition closure stored no action frames")
                if (
                    str(
                        _require_transition_batch_field(
                            "compact_owner_search_owner_proxy_transition_closure_fallback_reason"
                        )
                        or ""
                    )
                    != "none"
                ):
                    raise ValueError("owner-proxy transition closure must not fallback")
                if not str(
                    _require_transition_batch_field(
                        "compact_owner_search_owner_proxy_transition_closure_digest"
                    )
                    or ""
                ):
                    raise ValueError("owner-proxy transition closure must report digest")
                if (
                    _require_transition_batch_field(
                        "compact_owner_search_owner_proxy_transition_closure_digest_verified"
                    )
                    is not True
                ):
                    raise ValueError("owner-proxy transition closure must verify digest")
                for name in (
                    "compact_owner_search_owner_proxy_transition_closure_transport_bytes",
                    "compact_owner_search_owner_proxy_transition_closure_build_sec",
                    "compact_owner_search_owner_proxy_transition_closure_submit_sec",
                ):
                    value = float(_require_transition_batch_field(name) or 0.0)
                    if not math.isfinite(value) or value < 0.0:
                        raise ValueError(
                            f"owner-proxy transition closure must report nonnegative {name}"
                        )
        direct_transition_batch_replay_requested = bool(
            args is not None
            and bool(getattr(args, "owner_search_direct_transition_batch_replay", False))
        )
        if direct_transition_batch_replay_requested:
            deferred_one_simulation_requested = bool(
                args is not None
                and bool(
                    getattr(
                        args,
                        "compact_torch_defer_one_simulation_replay_payload",
                        False,
                    )
                )
            )
            direct_requested = _require_transition_batch_field(
                "compact_owner_search_direct_transition_batch_replay_requested"
            )
            if direct_requested is not True:
                raise ValueError(
                    "owner-search direct transition-batch replay must report request bit"
                )
            if (
                _require_transition_batch_field(
                    "compact_owner_search_direct_transition_batch_replay_used"
                )
                is not True
            ):
                raise ValueError("owner-search direct transition-batch replay must be used")
            if (
                int(
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_batch_count"
                    )
                    or 0
                )
                != batch_count
            ):
                raise ValueError("owner-search direct transition-batch replay batch count mismatch")
            if (
                int(
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_transition_count"
                    )
                    or 0
                )
                != entry_count
            ):
                raise ValueError(
                    "owner-search direct transition-batch replay transition count mismatch"
                )
            if (
                int(
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_transport_entry_count"
                    )
                    or 0
                )
                != batch_count
            ):
                raise ValueError(
                    "owner-search direct transition-batch replay transport count mismatch"
                )
            for name in (
                "compact_owner_search_direct_transition_batch_replay_legacy_expanded_entry_count",
                "compact_owner_search_direct_transition_batch_replay_index_entry_object_count",
                "compact_owner_search_direct_transition_batch_replay_fallback_count",
            ):
                if int(_require_transition_batch_field(name) or 0) != 0:
                    raise ValueError(
                        f"owner-search direct transition-batch replay must keep {name} zero"
                    )
            if (
                str(
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_fallback_reason"
                    )
                    or ""
                )
                != "none"
            ):
                raise ValueError("owner-search direct transition-batch replay must not fallback")
            ring_entry_objects = int(
                _require_transition_batch_field(
                    "compact_owner_search_direct_transition_batch_replay_ring_entry_object_count"
                )
                or 0
            )
            if ring_entry_objects != 0:
                raise ValueError(
                    "owner-search direct transition-batch replay must remove ring entry objects"
                )
            fixed_soa_replay_requested = bool(
                args is not None and bool(getattr(args, "owner_search_fixed_soa_replay", False))
            )
            if fixed_soa_replay_requested:
                if (
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_fixed_soa_requested"
                    )
                    is not True
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay fixed SoA "
                        "must report request bit"
                    )
                if (
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_fixed_soa_used"
                    )
                    is not True
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay fixed SoA must be used"
                    )
                if (
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_columnar_append_used"
                    )
                    is not False
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay fixed SoA "
                        "must bypass columnar append"
                    )
                for name in (
                    "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count",
                    "compact_owner_search_direct_transition_batch_replay_columnar_record_count",
                    "compact_owner_search_direct_transition_batch_replay_columnar_entry_view_object_count",
                    "compact_owner_search_direct_transition_batch_replay_columnar_step_view_object_count",
                ):
                    if int(_require_transition_batch_field(name) or 0) != 0:
                        raise ValueError(
                            "owner-search direct transition-batch replay fixed SoA "
                            f"must keep {name} zero"
                        )
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay fixed SoA slot count mismatch"
                    )
                for name in (
                    "compact_owner_search_direct_transition_batch_replay_fixed_soa_entry_view_object_count",
                    "compact_owner_search_direct_transition_batch_replay_fixed_soa_step_view_object_count",
                    "compact_owner_search_direct_transition_batch_replay_fixed_soa_learner_ready_object_count",
                    "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_entry_object_count",
                    "compact_owner_search_direct_transition_batch_replay_fixed_soa_table_concat_count",
                    "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_count",
                ):
                    if int(_require_transition_batch_field(name) or 0) != 0:
                        raise ValueError(
                            "owner-search direct transition-batch replay fixed SoA "
                            f"must keep {name} zero"
                        )
                if (
                    str(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_fixed_soa_fallback_reason"
                        )
                        or ""
                    )
                    != "none"
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay fixed SoA must not fallback"
                    )
            else:
                if (
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_columnar_append_used"
                    )
                    is not True
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay must use columnar append"
                    )
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay columnar slot count mismatch"
                    )
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_columnar_record_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay columnar record count mismatch"
                    )
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_columnar_entry_view_object_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay columnar entry-view count mismatch"
                    )
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_columnar_step_view_object_count"
                        )
                        or 0
                    )
                    != entry_count * 2
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay columnar step-view count mismatch"
                    )
            append_sec = float(
                _require_transition_batch_field(
                    "compact_owner_search_direct_transition_batch_replay_last_append_sec"
                )
                or 0.0
            )
            if not math.isfinite(append_sec) or append_sec < 0.0:
                raise ValueError(
                    "owner-search direct transition-batch replay must report nonnegative append sec"
                )
            for timing_name in (
                "compact_owner_search_direct_transition_batch_replay_append_sec",
                "compact_owner_search_direct_transition_batch_replay_accounted_sec",
                "compact_owner_search_direct_transition_batch_replay_array_extract_sec",
                "compact_owner_search_direct_transition_batch_replay_transition_validate_sec",
                "compact_owner_search_direct_transition_batch_replay_device_payload_sec",
                "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_flush_sec",
                "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flush_sec",
                "compact_owner_search_direct_transition_batch_replay_index_rows_build_sec",
                "compact_owner_search_direct_transition_batch_replay_step_object_build_sec",
                "compact_owner_search_direct_transition_batch_replay_ring_append_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_prepare_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_register_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_append_store_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_retain_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_evict_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_evict_release_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_candidate_indices_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_cache_refresh_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_cache_rebuild_sec",
                "compact_owner_search_direct_transition_batch_replay_columnar_total_sec",
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_slot_write_sec",
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_successor_index_sec",
                "compact_owner_search_direct_transition_batch_replay_fixed_soa_total_sec",
            ):
                timing_sec = float(_require_transition_batch_field(timing_name) or 0.0)
                if not math.isfinite(timing_sec) or timing_sec < 0.0:
                    raise ValueError(
                        "owner-search direct transition-batch replay must report "
                        f"nonnegative timing field {timing_name}"
                    )
            for count_name in (
                "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flushed_count",
                "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_count",
                "compact_owner_search_direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count",
                "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls",
                "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count",
                "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_count_max",
                "compact_owner_search_direct_transition_batch_replay_pending_deferred_replay_payload_final_count",
                "compact_owner_search_direct_transition_batch_replay_replay_payload_d2h_bytes",
            ):
                count_value = float(_require_transition_batch_field(count_name) or 0.0)
                if not math.isfinite(count_value) or count_value < 0.0:
                    raise ValueError(
                        "owner-search direct transition-batch replay must report "
                        f"nonnegative deferred flush field {count_name}"
                    )
            if (
                int(
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_refresh_crossed_count"
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError(
                    "owner-search direct transition-batch replay must keep deferred "
                    "one-simulation model-refresh-crossed count zero"
                )
            if deferred_one_simulation_requested:
                deferred_flush_count = int(
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_flush_count"
                    )
                    or 0
                )
                if deferred_flush_count != entry_count:
                    raise ValueError(
                        "owner-search direct transition-batch replay deferred "
                        "one-simulation flush count mismatch"
                    )
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_device_replay_payload_flushed_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay device flush count mismatch"
                    )
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_one_simulation_replay_materialized_on_flush_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay materialized-on-flush "
                        "count mismatch"
                    )
                if (
                    int(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_model_identity_match_count"
                        )
                        or 0
                    )
                    != entry_count
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay model identity "
                        "match count mismatch"
                    )
                recurrent_calls = float(
                    _require_transition_batch_field(
                        "compact_owner_search_direct_transition_batch_replay_deferred_one_simulation_replay_recurrent_inference_calls"
                    )
                    or 0.0
                )
                if int(round(recurrent_calls)) != entry_count:
                    raise ValueError(
                        "owner-search direct transition-batch replay recurrent "
                        "inference call count mismatch"
                    )
                if (
                    int(
                        profile_payload.get(
                            "compact_owner_search_inner_pending_deferred_replay_payload_final_count"
                        )
                        or 0
                    )
                    != 0
                ):
                    raise ValueError(
                        "owner-search deferred one-simulation replay payload final "
                        "pending count must be zero after owner preflush"
                    )
                if (
                    float(
                        _require_transition_batch_field(
                            "compact_owner_search_direct_transition_batch_replay_replay_payload_d2h_bytes"
                        )
                        or 0.0
                    )
                    != 0.0
                ):
                    raise ValueError(
                        "owner-search direct transition-batch replay deferred device "
                        "payload must not copy replay payload to host"
                    )
    if owner_replay_append_enabled:
        expected_owner_sample_batch_size: int | None = None
        expected_owner_train_steps: int | None = None
        expected_owner_train_interval: int | None = None
        if args is not None and hasattr(args, "sample_batch_size"):
            expected_owner_sample_batch_size = int(args.sample_batch_size)
        if args is not None and hasattr(args, "learner_train_steps"):
            expected_owner_train_steps = int(args.learner_train_steps)
        if (
            args is not None
            and hasattr(args, "sample_interval")
            and hasattr(args, "learner_num_unroll_steps")
        ):
            expected_owner_train_interval = _owner_search_train_interval(args)
        if profile_payload.get("compact_owner_search_worker_owns_replay_state") is not True:
            raise ValueError("owner replay/train row must prove worker-owned replay state")
        if profile_payload.get("compact_owner_search_worker_owns_model_state") is not True:
            raise ValueError("owner replay/train row must prove worker-owned model state")
        owner_sample_batch_size = int(
            profile_payload.get("compact_owner_search_owner_sample_batch_size") or 0
        )
        if (
            expected_owner_sample_batch_size is not None
            and owner_sample_batch_size != expected_owner_sample_batch_size
        ):
            raise ValueError(
                "owner replay/train row sample batch size does not match requested args: "
                f"expected {expected_owner_sample_batch_size}, got {owner_sample_batch_size}"
            )
        owner_sample_telemetry = _owner_search_normalized_owner_sample_telemetry_for_proof(
            profile_payload
        )
        if not isinstance(owner_sample_telemetry, Mapping) or not owner_sample_telemetry:
            raise ValueError("owner replay/train row must report owner sample telemetry")
        sample_row_count = int(
            owner_sample_telemetry.get("compact_rollout_slab_sample_gate_sample_row_count") or 0
        )
        target_row_count = int(
            owner_sample_telemetry.get("compact_rollout_slab_sample_gate_target_row_count") or 0
        )
        requested_sample_row_count = int(
            owner_sample_telemetry.get(
                "compact_rollout_slab_sample_gate_requested_sample_row_count"
            )
            or 0
        )
        if sample_row_count <= 0:
            raise ValueError("owner replay/train row must sample owner replay rows")
        if target_row_count <= 0:
            raise ValueError("owner replay/train row must build owner target rows")
        if owner_sample_batch_size > 0 and sample_row_count != owner_sample_batch_size:
            raise ValueError(
                "owner replay/train row sample rows must match owner sample batch size"
            )
        if owner_sample_batch_size > 0 and requested_sample_row_count != owner_sample_batch_size:
            raise ValueError(
                "owner replay/train row requested sample rows must match sample batch size"
            )
        if owner_sample_batch_size == 0 and requested_sample_row_count != 0:
            raise ValueError("owner replay/train row zero-batch samples must request zero rows")
        if owner_sample_batch_size == 0 and sample_row_count != target_row_count:
            raise ValueError("owner replay/train row zero-batch sample rows must match target rows")
        if (
            owner_sample_telemetry.get("compact_rollout_slab_sample_gate_require_next_targets")
            is not True
        ):
            raise ValueError("owner replay/train row must require next targets")
        if (
            owner_sample_telemetry.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch"
            )
            is not True
        ):
            raise ValueError("owner replay/train row must build compact MuZero batch")
        if (
            owner_sample_telemetry.get(
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"
            )
            is not True
        ):
            raise ValueError("owner replay/train row must sample learner-batch only")
        staged_append_entries = int(
            profile_payload.get("compact_owner_search_owner_replay_append_staged_entry_count") or 0
        )
        submitted_append_entries = int(
            profile_payload.get("compact_owner_search_owner_replay_append_submitted_entry_count")
            or 0
        )
        append_request_count = int(
            profile_payload.get("compact_owner_search_owner_replay_append_request_count") or 0
        )
        owner_append_count = int(
            profile_payload.get("compact_owner_search_owner_replay_append_count") or 0
        )
        pending_append_entries = int(
            profile_payload.get("compact_owner_search_owner_pending_replay_append_entry_count") or 0
        )
        owner_defer_maintenance = bool(
            profile_payload.get("compact_owner_search_owner_defer_maintenance", False)
        )
        if staged_append_entries <= 0:
            raise ValueError("owner replay/train row must stage owner replay append entries")
        if submitted_append_entries != staged_append_entries:
            raise ValueError("owner replay/train row must submit every staged replay append entry")
        if append_request_count <= 0 or append_request_count > submitted_append_entries:
            raise ValueError("owner replay/train row must report valid append request count")
        if owner_append_count != submitted_append_entries:
            raise ValueError("owner replay/train row must append every submitted entry")
        if pending_append_entries != 0:
            raise ValueError("owner replay/train row must drain pending owner replay appends")
        if owner_defer_maintenance:
            if inline_boundary:
                expected_loop_kind = "inline_priority_owner_loop_v1"
            elif inline_background_boundary:
                expected_loop_kind = "inline_background_maintenance_owner_loop_v1"
            elif threaded_boundary:
                expected_loop_kind = "threaded_priority_owner_loop_v1"
            else:
                expected_loop_kind = "persistent_priority_owner_loop_v1"
            if str(profile_payload.get("compact_owner_search_owner_loop_kind") or "") != (
                expected_loop_kind
            ):
                raise ValueError(
                    "deferred owner maintenance row must use the expected "
                    "persistent priority owner loop"
                )
            if profile_payload.get("compact_owner_search_owner_loop_persistent") is not True:
                raise ValueError("deferred owner maintenance row must prove persistent owner loop")
            if (
                profile_payload.get("compact_owner_search_owner_action_priority_enabled")
                is not True
            ):
                raise ValueError(
                    "deferred owner maintenance row must prove action priority is enabled"
                )
            if inline_background_boundary or threaded_boundary:
                if (
                    profile_payload.get("compact_owner_search_owner_background_maintenance_thread")
                    is not True
                ):
                    raise ValueError(
                        "background owner-search row must prove background maintenance thread"
                    )
                if (
                    profile_payload.get("compact_owner_search_owner_background_overlap_enabled")
                    is not True
                ):
                    raise ValueError("background owner-search row must prove background overlap")
            drain_request_count = int(
                profile_payload.get("compact_owner_search_owner_maintenance_drain_request_count")
                or 0
            )
            action_request_count = int(
                profile_payload.get("compact_owner_search_owner_action_request_count") or 0
            )
            maintenance_request_count = int(
                profile_payload.get("compact_owner_search_owner_maintenance_request_count") or 0
            )
            action_while_pending_count = int(
                profile_payload.get(
                    "compact_owner_search_owner_action_while_maintenance_pending_count"
                )
                or 0
            )
            action_while_policy_lagged_count = int(
                profile_payload.get("compact_owner_search_owner_action_while_policy_lagged_count")
                or 0
            )
            action_served_before_maintenance_count = int(
                profile_payload.get(
                    "compact_owner_search_owner_action_served_before_maintenance_count"
                )
                or 0
            )
            fifo_blocked_action_count = int(
                profile_payload.get("compact_owner_search_owner_fifo_blocked_action_count") or 0
            )
            drained_count = int(
                profile_payload.get("compact_owner_search_owner_maintenance_drained_count") or 0
            )
            staged_work_item_count = int(
                profile_payload.get("compact_owner_search_owner_maintenance_staged_work_item_count")
                or 0
            )
            drained_work_item_count = int(
                profile_payload.get(
                    "compact_owner_search_owner_maintenance_drained_work_item_count",
                    drained_count,
                )
                or 0
            )
            drained_replay_append_entry_count = int(
                profile_payload.get(
                    "compact_owner_search_owner_maintenance_drained_replay_append_entry_count"
                )
                or 0
            )
            drained_replay_append_count = int(
                profile_payload.get(
                    "compact_owner_search_owner_maintenance_drained_replay_append_count"
                )
                or 0
            )
            pending_work_count = int(
                profile_payload.get("compact_owner_search_owner_maintenance_pending_work_count")
                or 0
            )
            policy_lag_current = int(
                profile_payload.get("compact_owner_search_owner_policy_lag_current") or 0
            )
            policy_lag_max = int(
                profile_payload.get("compact_owner_search_owner_policy_lag_max") or 0
            )
            owner_train_request_count_for_async = int(
                profile_payload.get("compact_owner_search_owner_train_request_count") or 0
            )
            async_requested = bool(
                profile_payload.get(
                    "owner_search_async_learner_worker_requested",
                    False,
                )
            )
            if args is not None:
                async_requested = async_requested or bool(
                    getattr(args, "owner_search_async_learner_worker", False)
                )
            async_enabled = bool(
                profile_payload.get(
                    "compact_owner_search_owner_async_learner_worker_enabled",
                    False,
                )
            )
            async_submit_count = int(
                profile_payload.get("compact_owner_search_owner_async_learner_submit_count") or 0
            )
            async_completed_count = int(
                profile_payload.get("compact_owner_search_owner_async_learner_completed_count") or 0
            )
            async_pending_count = int(
                profile_payload.get("compact_owner_search_owner_async_learner_pending_count") or 0
            )
            async_action_pending_count = int(
                profile_payload.get(
                    "compact_owner_search_owner_action_while_async_learner_pending_count"
                )
                or 0
            )
            async_worker_kind = str(
                profile_payload.get("compact_owner_search_owner_async_learner_worker_kind")
                or "none"
            )
            requested_async_worker_kind = str(
                profile_payload.get("owner_search_async_learner_worker_kind_requested")
                or getattr(
                    args,
                    "owner_search_async_learner_worker_kind",
                    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
                )
            )
            requested_async_max_pending = int(
                profile_payload.get("owner_search_async_learner_max_pending_requested")
                or getattr(args, "owner_search_async_learner_max_pending", 1)
                or 1
            )
            async_max_pending = int(
                profile_payload.get("compact_owner_search_owner_async_learner_max_pending") or 0
            )
            overlap_telemetry = profile_payload.get("compact_owner_search_owner_learner_telemetry")
            mock_fast_overlap_row = bool(
                isinstance(overlap_telemetry, Mapping)
                and overlap_telemetry.get("compact_owner_search_mock_fast_learner", False)
            )
            if drain_request_count <= 0:
                raise ValueError("deferred owner maintenance row must request drains")
            if action_request_count <= 0:
                raise ValueError("deferred owner maintenance row must report action requests")
            if maintenance_request_count <= 0:
                raise ValueError("deferred owner maintenance row must report maintenance requests")
            if staged_work_item_count <= 0:
                raise ValueError("deferred owner maintenance row must stage maintenance work items")
            if fifo_blocked_action_count != 0:
                raise ValueError("deferred owner maintenance row must not report FIFO blocking")
            if not inline_boundary and append_request_count > 1 and action_while_pending_count <= 0:
                raise ValueError(
                    "persistent priority row must prove action while maintenance pending"
                )
            if (
                not inline_boundary
                and append_request_count > 1
                and action_served_before_maintenance_count <= 0
            ):
                raise ValueError(
                    "persistent priority row must prove action served before maintenance"
                )
            if (
                not inline_boundary
                and not mock_fast_overlap_row
                and append_request_count > 1
                and policy_lag_max <= 0
            ):
                raise ValueError(
                    "real deferred owner maintenance row must prove positive policy lag"
                )
            if (
                not inline_boundary
                and not mock_fast_overlap_row
                and append_request_count > 1
                and action_while_policy_lagged_count <= 0
            ):
                raise ValueError(
                    "real deferred owner maintenance row must prove action while policy lagged"
                )
            if drained_count != drained_work_item_count:
                raise ValueError(
                    "deferred owner maintenance legacy drained count must match drained work items"
                )
            if drained_work_item_count != staged_work_item_count:
                raise ValueError(
                    "deferred owner maintenance row must drain every staged work item: "
                    f"expected {staged_work_item_count}, got {drained_work_item_count}"
                )
            if drained_replay_append_entry_count != submitted_append_entries:
                raise ValueError(
                    "deferred owner maintenance row must drain every submitted replay "
                    "append entry: "
                    f"expected {submitted_append_entries}, "
                    f"got {drained_replay_append_entry_count}"
                )
            if drained_replay_append_count != owner_append_count:
                raise ValueError(
                    "deferred owner maintenance row must drain every appended replay row: "
                    f"expected {owner_append_count}, got {drained_replay_append_count}"
                )
            if pending_work_count != 0:
                raise ValueError("deferred owner maintenance row must finish with no pending work")
            if profile_payload.get("compact_owner_search_owner_maintenance_inflight") is True:
                raise ValueError("deferred owner maintenance row must finish with no inflight work")
            if profile_payload.get("compact_owner_search_owner_maintenance_failed") is True:
                raise ValueError("deferred owner maintenance row must not report failure")
            if async_enabled or async_requested:
                if async_enabled is not True:
                    raise ValueError(
                        "async owner learner row must prove async learner worker enabled"
                    )
                if async_worker_kind == "none":
                    raise ValueError(
                        "async owner learner row must report async learner worker kind"
                    )
                if async_worker_kind != requested_async_worker_kind:
                    raise ValueError(
                        "async owner learner worker kind must match request: "
                        f"expected {requested_async_worker_kind}, got {async_worker_kind}"
                    )
                if async_max_pending != requested_async_max_pending:
                    raise ValueError(
                        "async owner learner max pending must match request: "
                        f"expected {requested_async_max_pending}, got {async_max_pending}"
                    )
                if (
                    profile_payload.get(
                        "compact_owner_search_owner_async_learner_failed",
                        False,
                    )
                    is True
                ):
                    raise ValueError("async owner learner row must not report failure")
                if async_pending_count != 0:
                    raise ValueError(
                        "async owner learner row must finish with no pending learner jobs"
                    )
                if async_submit_count <= 0:
                    raise ValueError("async owner learner row must submit learner jobs")
                if async_submit_count != owner_train_request_count_for_async:
                    raise ValueError("async owner learner submit count must match train requests")
                if async_completed_count != async_submit_count:
                    raise ValueError(
                        "async owner learner completed count must match submitted jobs"
                    )
                if (
                    int(
                        profile_payload.get(
                            "compact_owner_search_owner_async_learner_max_pending_observed"
                        )
                        or 0
                    )
                    <= 0
                ):
                    raise ValueError("async owner learner row must observe pending learner work")
                if (
                    not inline_boundary
                    and append_request_count > 1
                    and async_action_pending_count <= 0
                ):
                    raise ValueError(
                        "async owner learner row must prove action while learner pending"
                    )
                if (
                    async_worker_kind
                    == COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH
                ):
                    if (
                        str(
                            profile_payload.get(
                                "compact_owner_search_owner_async_learner_worker_resource_scope"
                            )
                            or ""
                        )
                        != "process"
                    ):
                        raise ValueError(
                            "process async owner learner row must report process scope"
                        )
                    if (
                        profile_payload.get(
                            "compact_owner_search_owner_async_learner_request_host_only"
                        )
                        is not True
                    ):
                        raise ValueError(
                            "process async owner learner row must send host-only requests"
                        )
                    if (
                        int(
                            profile_payload.get(
                                "compact_owner_search_owner_async_learner_request_cuda_tensor_count"
                            )
                            or 0
                        )
                        != 0
                    ):
                        raise ValueError(
                            "process async owner learner row must not send CUDA tensors"
                        )
                    if (
                        profile_payload.get(
                            "compact_owner_search_owner_async_learner_result_host_only"
                        )
                        is not True
                    ):
                        raise ValueError(
                            "process async owner learner row must return host-only results"
                        )
                    if (
                        int(
                            profile_payload.get(
                                "compact_owner_search_owner_async_learner_result_cuda_tensor_count"
                            )
                            or 0
                        )
                        != 0
                    ):
                        raise ValueError(
                            "process async owner learner row must not return CUDA tensors"
                        )
                    if (
                        int(
                            profile_payload.get(
                                "compact_owner_search_owner_async_learner_request_bytes"
                            )
                            or 0
                        )
                        <= 0
                    ):
                        raise ValueError(
                            "process async owner learner row must report request bytes"
                        )
                    if (
                        int(
                            profile_payload.get(
                                "compact_owner_search_owner_async_learner_result_bytes"
                            )
                            or 0
                        )
                        <= 0
                    ):
                        raise ValueError("process async owner learner row must report result bytes")
                    if (
                        int(
                            profile_payload.get(
                                "compact_owner_search_owner_async_learner_worker_pid"
                            )
                            or 0
                        )
                        <= 0
                    ):
                        raise ValueError("process async owner learner row must report worker pid")
                    if (
                        profile_payload.get(
                            "compact_owner_search_owner_async_learner_worker_pid_distinct_"
                            "from_owner"
                        )
                        is not True
                    ):
                        raise ValueError(
                            "process async owner learner row must prove distinct worker pid"
                        )
                    if (
                        profile_payload.get(
                            "compact_owner_search_owner_async_learner_worker_owns_model_state"
                        )
                        is not True
                    ):
                        raise ValueError(
                            "process async owner learner row must prove worker owns model state"
                        )
            if policy_lag_current != 0:
                raise ValueError("deferred owner maintenance row must finish with zero policy lag")
            if (
                profile_payload.get(
                    "compact_owner_search_owner_maintenance_final_drain_in_measured_sec"
                )
                is not True
            ):
                raise ValueError(
                    "deferred owner maintenance row must include final drain in wall time"
                )
            if "compact_owner_search_owner_maintenance_final_drain_sec" not in profile_payload:
                raise ValueError("deferred owner maintenance row must report final drain sec")
            final_drain_sec = float(
                profile_payload.get("compact_owner_search_owner_maintenance_final_drain_sec") or 0.0
            )
            if not math.isfinite(final_drain_sec) or final_drain_sec < 0.0:
                raise ValueError(
                    "deferred owner maintenance row must report finite nonnegative final drain"
                )
        replay_append_entry_count = int(
            profile_payload.get("compact_owner_search_replay_append_entry_count") or 0
        )
        if replay_append_entry_count <= 0:
            raise ValueError("owner replay/train row must send replay append entries")
        if replay_append_entry_count != submitted_append_entries:
            raise ValueError(
                "owner replay/train row replay append entry count must match submitted entries"
            )
        replay_append_count = int(
            profile_payload.get("compact_owner_search_replay_append_count") or 0
        )
        if replay_append_count <= 0:
            raise ValueError("owner replay/train row must append replay in owner")
        if replay_append_count != owner_append_count:
            raise ValueError(
                "owner replay/train row replay append count must match owner append count"
            )
        learner_update_count = int(
            profile_payload.get("compact_owner_search_learner_update_count") or 0
        )
        owner_learner_update_count = int(
            profile_payload.get("compact_owner_search_owner_learner_update_count") or 0
        )
        owner_submitted_learner_update_count = int(
            profile_payload.get(
                "compact_owner_search_owner_submitted_learner_update_count",
                owner_learner_update_count,
            )
            or 0
        )
        if learner_update_count <= 0:
            raise ValueError("owner replay/train row must report owner learner updates")
        train_request_count = int(
            profile_payload.get("compact_owner_search_owner_train_request_count") or 0
        )
        if train_request_count <= 0:
            raise ValueError("owner replay/train row must submit owner train requests")
        train_interval = int(profile_payload.get("compact_owner_search_owner_train_interval") or 0)
        owner_train_steps = int(profile_payload.get("compact_owner_search_owner_train_steps") or 0)
        if train_interval <= 0:
            raise ValueError("owner replay/train row must report positive train interval")
        if owner_train_steps <= 0:
            raise ValueError("owner replay/train row must report positive owner train steps")
        if (
            expected_owner_train_steps is not None
            and owner_train_steps != expected_owner_train_steps
        ):
            raise ValueError(
                "owner replay/train row train steps do not match requested args: "
                f"expected {expected_owner_train_steps}, got {owner_train_steps}"
            )
        if (
            expected_owner_train_interval is not None
            and train_interval != expected_owner_train_interval
        ):
            raise ValueError(
                "owner replay/train row train interval does not match requested args: "
                f"expected {expected_owner_train_interval}, got {train_interval}"
            )
        expected_train_buckets = submitted_append_entries // train_interval
        if train_request_count > expected_train_buckets:
            raise ValueError(
                "owner replay/train row train request count does not match cadence: "
                f"expected at most {expected_train_buckets}, got {train_request_count}"
            )
        model_refresh_interval = int(
            profile_payload.get("compact_owner_search_owner_model_refresh_interval") or 0
        )
        expected_train_request_count = int(
            profile_payload.get("compact_owner_search_owner_expected_train_request_count") or 0
        )
        model_refresh_request_count = int(
            profile_payload.get("compact_owner_search_owner_model_refresh_request_count") or 0
        )
        model_refresh_skipped_count = int(
            profile_payload.get("compact_owner_search_owner_model_refresh_skipped_count") or 0
        )
        if model_refresh_interval <= 0:
            raise ValueError("owner replay/train row must report positive model refresh interval")
        if expected_train_request_count not in {0, train_request_count}:
            raise ValueError(
                "owner replay/train row expected train count must match actual "
                f"train requests: expected {train_request_count}, got "
                f"{expected_train_request_count}"
            )
        expected_model_refresh_requests = train_request_count
        if model_refresh_interval > 1:
            expected_model_refresh_requests = train_request_count // model_refresh_interval
            if train_request_count % model_refresh_interval:
                expected_model_refresh_requests += 1
        if model_refresh_request_count != expected_model_refresh_requests:
            raise ValueError(
                "owner replay/train row model refresh requests do not match cadence: "
                f"expected {expected_model_refresh_requests}, got "
                f"{model_refresh_request_count}"
            )
        expected_model_refresh_skips = train_request_count - expected_model_refresh_requests
        if model_refresh_skipped_count != expected_model_refresh_skips:
            raise ValueError(
                "owner replay/train row model refresh skips do not match cadence: "
                f"expected {expected_model_refresh_skips}, got "
                f"{model_refresh_skipped_count}"
            )
        expected_updates = expected_train_buckets * owner_train_steps
        if owner_learner_update_count != expected_updates:
            raise ValueError(
                "owner replay/train row owner learner updates do not match train requests: "
                f"expected {expected_updates}, got {owner_learner_update_count}"
            )
        if owner_submitted_learner_update_count != expected_updates:
            raise ValueError(
                "owner replay/train row submitted learner updates do not match train "
                "requests: "
                f"expected {expected_updates}, got {owner_submitted_learner_update_count}"
            )
        if learner_update_count != owner_learner_update_count:
            raise ValueError("owner replay/train row learner update counters disagree")
        if profile_payload.get("compact_owner_search_model_owner_ref_returned") is not True:
            raise ValueError("owner replay/train row must return an owner model ref")
        if not str(profile_payload.get("compact_owner_search_model_owner_ref_digest") or ""):
            raise ValueError("owner replay/train row must report owner-ref digest")
        if profile_payload.get("compact_owner_search_consumed_learner_update") is not True:
            raise ValueError("owner replay/train row must refresh search from owner learner")
        if (
            int(profile_payload.get("compact_owner_search_search_refresh_update_count") or 0)
            != owner_learner_update_count
        ):
            raise ValueError(
                "owner replay/train row search refresh count must match owner learner updates"
            )
        owner_learner_telemetry = profile_payload.get(
            "compact_owner_search_owner_learner_telemetry"
        )
        if not isinstance(owner_learner_telemetry, Mapping) or not owner_learner_telemetry:
            raise ValueError("owner replay/train row must report owner learner telemetry")
        timing_aggregate_count = int(
            profile_payload.get(
                "compact_owner_search_owner_train_timing_aggregate_count",
                owner_learner_telemetry.get(
                    "compact_owner_search_owner_train_timing_aggregate_count"
                ),
            )
            or 0
        )
        if timing_aggregate_count != train_request_count:
            raise ValueError("owner replay/train row train timing count must match train requests")
        for key in (
            "compact_owner_search_owner_train_wall_sec",
            "compact_owner_search_owner_train_sample_sec",
            "compact_owner_search_owner_train_learner_update_sec",
            "compact_owner_search_owner_train_model_state_digest_sec",
            "compact_owner_search_owner_train_model_state_dict_sec",
            "compact_owner_search_owner_train_owner_ref_build_sec",
            "compact_owner_search_owner_train_accounted_sec",
            "compact_owner_search_owner_train_residual_sec",
        ):
            if key not in profile_payload and key not in owner_learner_telemetry:
                raise ValueError(f"owner replay/train row must report {key}")
            value = float(profile_payload.get(key, owner_learner_telemetry.get(key)) or 0.0)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"owner replay/train row has invalid {key}")
        if (
            float(
                profile_payload.get(
                    "compact_owner_search_owner_train_wall_sec",
                    owner_learner_telemetry.get("compact_owner_search_owner_train_wall_sec"),
                )
                or 0.0
            )
            <= 0.0
        ):
            raise ValueError("owner replay/train row must report positive train wall")
        mock_fast_learner = bool(
            owner_learner_telemetry.get("compact_owner_search_mock_fast_learner", False)
        )
        learner_update_sec = float(
            profile_payload.get(
                "compact_owner_search_owner_train_learner_update_sec",
                owner_learner_telemetry.get("compact_owner_search_owner_train_learner_update_sec"),
            )
            or 0.0
        )
        if not mock_fast_learner and learner_update_sec <= 0.0:
            raise ValueError("owner replay/train row must report positive learner update")
        fixed_soa_replay_requested = bool(
            args is not None and bool(getattr(args, "owner_search_fixed_soa_replay", False))
        )
        if fixed_soa_replay_requested:
            resident_handle_fields = _owner_search_learner_resident_batch_handle_fields(
                metadata=owner_learner_telemetry
            )

            def _resident_handle_field(suffix: str) -> Any:
                return resident_handle_fields[
                    f"{OWNER_SEARCH_RESIDENT_BATCH_HANDLE_PREFIX}_{suffix}"
                ]

            if _resident_handle_field("requested") is not True:
                raise ValueError(
                    "owner-search fixed SoA row must request resident learner-batch handle"
                )
            if int(_resident_handle_field("fallback_count") or 0) != 0:
                raise ValueError(
                    "owner-search fixed SoA row resident learner-batch handle "
                    "fallback count must be zero"
                )
            if str(_resident_handle_field("fallback_reason") or "") != "none":
                raise ValueError(
                    "owner-search fixed SoA row resident learner-batch handle "
                    "fallback reason must be none"
                )
            if int(_resident_handle_field("materialized_parent_fallback_count") or 0) != 0:
                raise ValueError(
                    "owner-search fixed SoA row materialized-parent fallback count "
                    "must be zero"
                )
            if (
                str(_resident_handle_field("materialized_parent_fallback_reason") or "")
                != "none"
            ):
                raise ValueError(
                    "owner-search fixed SoA row materialized-parent fallback reason "
                    "must be none"
                )
            if _resident_handle_field("consumed") is not True:
                raise ValueError(
                    "owner-search fixed SoA row must consume resident learner-batch handle"
                )
            for name in (
                "handle_id",
                "snapshot_version",
                "request_checksum",
                "sample_row_count",
                "target_row_count",
            ):
                if int(_resident_handle_field(name) or 0) <= 0:
                    raise ValueError(
                        "owner-search fixed SoA row resident learner-batch handle "
                        f"must report positive {name}"
                    )
            if str(_resident_handle_field("schema_id") or "none") == "none":
                raise ValueError(
                    "owner-search fixed SoA row resident learner-batch handle "
                    "must report schema_id"
                )
    if int(profile_payload.get("compact_owner_search_owner_pid") or 0) <= 0:
        raise ValueError("owner-search row must report owner worker pid")
    if int(profile_payload.get("compact_owner_search_root_slot_count") or 0) <= 0:
        raise ValueError("owner-search row must report positive root slot count")
    if int(profile_payload.get("compact_owner_search_active_root_count") or 0) <= 0:
        raise ValueError("owner-search row must report positive active root count")
    if int(profile_payload.get("compact_owner_search_request_bytes") or 0) <= 0:
        raise ValueError("owner-search row must report request bytes")
    if int(profile_payload.get("compact_owner_search_result_bytes") or 0) <= 0:
        raise ValueError("owner-search row must report result bytes")
    if int(profile_payload.get("compact_owner_search_request_cuda_tensor_count") or 0) != 0:
        raise ValueError("owner-search request must contain no CUDA tensors")
    if int(profile_payload.get("compact_owner_search_result_cuda_tensor_count") or 0) != 0:
        raise ValueError("owner-search result must contain no CUDA tensors")
    if int(profile_payload.get("compact_owner_search_root_observation_bytes_sent") or 0) != 0:
        raise ValueError("owner-search row must send zero root-observation bytes")
    if action_only_result:
        if int(profile_payload.get("compact_rollout_slab_committed_index_row_count") or 0) != 0:
            raise ValueError("action-only owner-search row must commit zero parent slab rows")
        if int(profile_payload.get("compact_rollout_slab_stored_index_row_count") or 0) != 0:
            raise ValueError("action-only owner-search row must store zero parent slab rows")
        if profile_payload.get("compact_owner_search_parent_slab_commits_replay") is not False:
            raise ValueError(
                "action-only owner-search row must prove parent slab does not commit replay"
            )
        if profile_payload.get("compact_owner_search_owner_materializes_replay") is not True:
            raise ValueError("action-only owner-search row must prove owner materializes replay")
        if (
            profile_payload.get("compact_owner_search_parent_reconstructed_search_result")
            is not False
        ):
            raise ValueError("action-only owner-search row must not reconstruct full parent result")
        if int(profile_payload.get("compact_owner_search_search_result_payload_bytes") or 0) != 0:
            raise ValueError("action-only owner-search row must return zero search payload bytes")
        if int(profile_payload.get("compact_owner_search_visit_policy_bytes") or 0) != 0:
            raise ValueError("action-only owner-search row must return zero visit-policy bytes")
        if int(profile_payload.get("compact_owner_search_root_value_bytes") or 0) != 0:
            raise ValueError("action-only owner-search row must return zero root-value bytes")
        if (
            str(
                profile_payload.get("compact_owner_search_search_result_payload_transport_kind")
                or ""
            )
            != "action_only_owner_cached_replay_v1"
        ):
            raise ValueError("action-only owner-search row has wrong payload transport kind")
        if profile_payload.get("compact_owner_search_replay_payload_handle_present") is not True:
            raise ValueError("action-only owner-search row must return a replay payload handle")
        for key in (
            "compact_owner_search_inner_two_phase_action_step",
            "compact_owner_search_inner_device_replay_payload_deferred",
        ):
            if key not in profile_payload:
                raise ValueError(f"action-only owner-search row must report {key}")
        if compact_torch_owner_search:
            if (
                profile_payload.get("compact_owner_search_use_inner_two_phase_device_replay")
                is not True
            ):
                raise ValueError(
                    "compact Torch action-only owner-search row must enable inner "
                    "two-phase device replay"
                )
            if profile_payload.get("compact_owner_search_inner_two_phase_action_step") is not True:
                raise ValueError(
                    "compact Torch action-only owner-search row must use inner "
                    "two-phase action step"
                )
            if (
                profile_payload.get("compact_owner_search_inner_device_replay_payload_deferred")
                is not True
            ):
                raise ValueError(
                    "compact Torch action-only owner-search row must defer inner "
                    "device replay payload"
                )
        if profile_payload.get("compact_owner_search_action_feedback_verified") is not True:
            raise ValueError("action-only owner-search row must verify action feedback")
        action_feedback_transition_count = int(
            profile_payload.get("compact_owner_search_action_feedback_transition_count") or 0
        )
        action_only_submitted_append_entries = int(
            profile_payload.get("compact_owner_search_owner_replay_append_submitted_entry_count")
            or 0
        )
        if action_feedback_transition_count != action_only_submitted_append_entries:
            raise ValueError(
                "action-only owner-search row action-feedback transitions "
                "must match submitted entries"
            )
        action_feedback_action_count = int(
            profile_payload.get("compact_owner_search_action_feedback_action_count") or 0
        )
        if action_feedback_action_count <= 0:
            raise ValueError("action-only owner-search row must verify feedback actions")
        action_feedback_mismatch_count = int(
            profile_payload.get("compact_owner_search_action_feedback_mismatch_count") or 0
        )
        if action_feedback_mismatch_count != 0:
            raise ValueError("action-only owner-search row action feedback must not mismatch")
        expected_action_checksum = int(
            profile_payload.get("compact_owner_search_expected_joint_action_checksum") or 0
        )
        applied_action_checksum = int(
            profile_payload.get("compact_owner_search_applied_joint_action_checksum") or 0
        )
        replay_action_checksum = int(
            profile_payload.get("compact_owner_search_replay_action_checksum") or 0
        )
        if expected_action_checksum <= 0:
            raise ValueError("action-only owner-search row must report action checksum")
        if not (expected_action_checksum == applied_action_checksum == replay_action_checksum):
            raise ValueError("action-only owner-search row action-feedback checksums must agree")
    else:
        if profile_payload.get("compact_owner_search_parent_slab_commits_replay") is not True:
            raise ValueError("owner-search row must explicitly report parent slab replay commit")
        if (
            profile_payload.get("compact_owner_search_parent_reconstructed_search_result")
            is not True
        ):
            raise ValueError("owner-search row must prove parent reconstructed search result")
    if int(profile_payload.get("compact_owner_search_model_state_bytes") or 0) != 0:
        raise ValueError("owner-search row must return zero model-state bytes")
    if int(profile_payload.get("compact_owner_search_model_state_return_count") or 0) != 0:
        raise ValueError("owner-search row must return zero parent model states")
    if int(profile_payload.get("compact_owner_search_model_state_snapshot_return_count") or 0) != 0:
        raise ValueError("owner-search row must return zero snapshot model states")
    byte_keys = ["compact_owner_search_selected_action_bytes"]
    if not action_only_result:
        byte_keys.extend(
            [
                "compact_owner_search_search_result_payload_bytes",
                "compact_owner_search_visit_policy_bytes",
                "compact_owner_search_root_value_bytes",
            ]
        )
    for key in byte_keys:
        if int(profile_payload.get(key) or 0) <= 0:
            raise ValueError(f"owner-search row must report positive {key}")
    if not action_only_result and (
        str(profile_payload.get("compact_owner_search_search_result_payload_transport_kind") or "")
        != "numpy_ndarray_ipc_v1"
    ):
        raise ValueError("owner-search row must use numpy ndarray result-payload IPC")
    for key in (
        "compact_owner_search_parent_publish_sec",
        "compact_owner_search_parent_submit_sec",
        "compact_owner_search_parent_wait_sec",
        "compact_owner_search_parent_wall_sec",
        "compact_owner_search_worker_wall_sec",
        "compact_owner_search_worker_root_resolve_sec",
        "compact_owner_search_worker_search_sec",
        "compact_owner_search_worker_replay_append_sec",
        "compact_owner_search_worker_learner_train_sec",
        "compact_owner_search_worker_search_refresh_sec",
    ):
        if key not in profile_payload:
            raise ValueError(f"owner-search row must report {key}")
        value = float(profile_payload.get(key) or 0.0)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"owner-search row has invalid {key}")
    if float(profile_payload.get("compact_owner_search_parent_wall_sec") or 0.0) <= 0.0:
        raise ValueError("owner-search row must report positive parent owner-search wall time")
    if float(profile_payload.get("compact_owner_search_worker_wall_sec") or 0.0) <= 0.0:
        raise ValueError("owner-search row must report positive worker owner-search wall time")
    if owner_replay_append_enabled:
        if (
            float(profile_payload.get("compact_owner_search_worker_replay_append_sec") or 0.0)
            <= 0.0
        ):
            raise ValueError("owner replay/train row must report positive replay append time")
        if (
            float(profile_payload.get("compact_owner_search_worker_learner_train_sec") or 0.0)
            <= 0.0
        ):
            raise ValueError("owner replay/train row must report positive learner train time")
        if (
            float(profile_payload.get("compact_owner_search_worker_search_refresh_sec") or 0.0)
            <= 0.0
        ):
            raise ValueError("owner replay/train row must report positive search refresh time")
    if (
        not action_only_result
        and int(profile_payload.get("compact_rollout_slab_committed_index_row_count") or 0) <= 0
    ):
        raise ValueError("owner-search row must preserve parent slab replay rows")
    if (
        str(profile_payload.get("owner_search_inner_search_service_kind") or "")
        == SEARCH_SERVICE_COMPACT_TORCH
        or profile_payload.get("owner_search_compact_torch_resident_root_bridge_ready") is True
    ):
        if inline_boundary or inline_background_boundary or threaded_boundary:
            _require_inline_owner_search_direct_root_proof(profile_payload)
            if _owner_search_resident_root_view_requested(args, profile_payload):
                _require_direct_resident_root_view_proof(profile_payload)
            if _owner_search_resident_root_host_observation_stub_requested(
                args,
                profile_payload,
            ):
                _require_resident_root_host_observation_stub_proof(profile_payload)
            if _owner_search_direct_root_build_request_requested(args, profile_payload):
                _require_direct_root_build_request_proof(
                    _owner_search_slab_proxy_guard_payload(profile_payload)
                )
            if _owner_search_fixed_action_result_buffer_requested(args, profile_payload):
                _require_fixed_action_result_buffer_proof(
                    {
                        **dict(profile_payload),
                        **_owner_search_slab_proxy_proof_fields(profile_payload),
                    }
                )
            if _owner_search_action_step_boundary_requested(args, profile_payload):
                _require_compact_owner_action_step_boundary_proof(
                    {
                        **dict(profile_payload),
                        **_owner_search_slab_proxy_proof_fields(profile_payload),
                    }
                )
            if _owner_search_action_dispatch_step_overlap_requested(args, profile_payload):
                _require_raw_compact_owner_action_dispatch_step_overlap_fields(
                    profile_payload
                )
                _require_compact_owner_action_dispatch_step_overlap_proof(
                    {
                        **dict(profile_payload),
                        **_owner_search_slab_proxy_proof_fields(profile_payload),
                    }
                )
        else:
            if _owner_search_resident_root_view_requested(args, profile_payload):
                raise ValueError("resident root-view proof requires direct-root owner-search mode")
            if _owner_search_resident_root_host_observation_stub_requested(
                args,
                profile_payload,
            ):
                raise ValueError(
                    "resident root host-observation stub requires direct-root owner-search mode"
                )
            if _owner_search_direct_root_build_request_requested(args, profile_payload):
                raise ValueError(
                    "direct root build-request proof requires direct-root owner-search mode"
                )
            if _owner_search_fixed_action_result_buffer_requested(args, profile_payload):
                raise ValueError(
                    "fixed action-result buffer proof requires direct-root owner-search mode"
                )
            if _owner_search_action_step_boundary_requested(args, profile_payload):
                raise ValueError(
                    "compact owner action-step boundary proof requires "
                    "direct-root owner-search mode"
                )
            if _owner_search_action_dispatch_step_overlap_requested(args, profile_payload):
                raise ValueError(
                    "compact owner action dispatch overlap proof requires "
                    "direct-root owner-search mode"
                )
            _require_compact_torch_owner_search_bridge_proof(profile_payload)


def _owner_search_resident_root_view_requested(
    args: argparse.Namespace | None,
    profile_payload: Mapping[str, Any],
) -> bool:
    return bool(
        (args is not None and getattr(args, "owner_search_require_resident_root_view", False))
        or profile_payload.get("owner_search_require_resident_root_view_requested", False)
        or profile_payload.get("compact_owner_search_resident_root_view_required", False)
    )


def _owner_search_resident_root_host_observation_stub_requested(
    args: argparse.Namespace | None,
    profile_payload: Mapping[str, Any],
) -> bool:
    return bool(
        (
            args is not None
            and getattr(args, "owner_search_resident_root_host_observation_stub", False)
        )
        or profile_payload.get(
            "owner_search_resident_root_host_observation_stub_requested",
            False,
        )
        or profile_payload.get(
            "compact_rollout_slab_resident_host_observation_stub_requested",
            False,
        )
    )


def _owner_search_direct_root_build_request_requested(
    args: argparse.Namespace | None,
    profile_payload: Mapping[str, Any],
) -> bool:
    return bool(
        (args is not None and getattr(args, "owner_search_direct_root_build_request", False))
        or profile_payload.get(
            "owner_search_direct_root_build_request_requested",
            False,
        )
        or profile_payload.get(
            "compact_owner_search_direct_root_build_request_requested",
            False,
        )
    )


def _owner_search_fixed_action_result_buffer_requested(
    args: argparse.Namespace | None,
    profile_payload: Mapping[str, Any],
) -> bool:
    return bool(
        (args is not None and getattr(args, "owner_search_fixed_action_result_buffer", False))
        or profile_payload.get(
            "owner_search_fixed_action_result_buffer_requested",
            False,
        )
        or profile_payload.get(
            "compact_owner_search_fixed_action_result_buffer_requested",
            False,
        )
    )


def _owner_search_action_step_boundary_requested(
    args: argparse.Namespace | None,
    profile_payload: Mapping[str, Any],
) -> bool:
    return bool(
        (args is not None and getattr(args, "compact_owner_action_step_boundary", False))
        or profile_payload.get("compact_owner_action_step_boundary_requested", False)
        or profile_payload.get("compact_owner_action_step_boundary_enabled", False)
    )


def _owner_search_action_dispatch_step_overlap_requested(
    args: argparse.Namespace | None,
    profile_payload: Mapping[str, Any],
) -> bool:
    return bool(
        (
            args is not None
            and getattr(args, "compact_owner_action_dispatch_step_overlap", False)
        )
        or profile_payload.get(
            "compact_owner_action_dispatch_step_overlap_requested",
            False,
        )
        or profile_payload.get(
            "compact_owner_action_dispatch_step_overlap_enabled",
            False,
        )
    )


def _owner_search_mechanics_step_frame_requested(
    profile_payload: Mapping[str, Any],
) -> bool:
    return bool(
        profile_payload.get("compact_owner_mechanics_step_boundary_enabled", False)
        or profile_payload.get("compact_owner_mechanics_step_boundary", False)
        or profile_payload.get("compact_owner_mechanics_step_frame_slot_schema_id")
        or profile_payload.get("compact_owner_mechanics_step_frame_handle_ring_used", False)
    )


def _owner_search_slab_proxy_guard_payload(
    profile_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **dict(profile_payload),
        **_owner_search_slab_proxy_proof_fields(profile_payload),
        "_owner_search_raw_profile_payload_keys": frozenset(
            str(key) for key in profile_payload.keys()
        ),
    }


def _require_compact_owner_action_step_boundary_proof(
    profile_payload: Mapping[str, Any],
) -> None:
    if profile_payload.get("compact_owner_search_direct_root_build_request_requested") is not True:
        raise ValueError(
            "compact owner action-step boundary requires direct root build-request proof"
        )
    if profile_payload.get("compact_owner_search_direct_root_build_request_handoff") is not True:
        raise ValueError("compact owner action-step boundary requires direct root handoff")
    if profile_payload.get("compact_owner_search_action_only_result") is not True:
        raise ValueError("compact owner action-step boundary requires action-only result")
    if profile_payload.get("compact_owner_search_parent_slab_commits_replay") is not False:
        raise ValueError("compact owner action-step boundary requires zero parent replay commits")
    for key in (
        "compact_owner_search_search_result_payload_bytes",
        "compact_owner_search_visit_policy_bytes",
        "compact_owner_search_root_value_bytes",
        "compact_rollout_slab_committed_index_row_count",
        "compact_rollout_slab_stored_index_row_count",
        "compact_owner_search_slab_bypass_parent_committed_index_rows",
        "compact_owner_search_slab_bypass_parent_stored_index_rows",
    ):
        if int(profile_payload.get(key) or 0) != 0:
            raise ValueError(f"compact owner action-step boundary must keep {key} zero")
    if profile_payload.get("compact_owner_action_step_boundary_enabled") is not True:
        raise ValueError("compact owner action-step boundary was requested but not enabled")
    if profile_payload.get("compact_owner_action_step_boundary_proof_passed") is not True:
        raise ValueError("compact owner action-step boundary proof did not pass")
    if str(profile_payload.get("compact_owner_action_step_boundary_failure_reason") or "") != (
        "none"
    ):
        raise ValueError("compact owner action-step boundary reported a failure reason")
    total_iterations = int(profile_payload.get("steps") or 0) + int(
        profile_payload.get("warmup_steps") or 0
    )
    if total_iterations <= 1:
        raise ValueError("compact owner action-step boundary requires feedback iterations")
    expected_counts = {
        "compact_owner_action_step_boundary_step_count": total_iterations,
        "compact_owner_action_step_boundary_action_verified_count": total_iterations,
        "compact_owner_action_step_boundary_next_action_count": total_iterations,
        "compact_owner_action_step_boundary_seeded_action_count": 1,
        "compact_owner_action_step_boundary_feedback_action_count": total_iterations - 1,
    }
    for key, expected in expected_counts.items():
        if int(profile_payload.get(key) or 0) != expected:
            raise ValueError(
                f"compact owner action-step boundary count mismatch for {key}"
            )
    if str(profile_payload.get("compact_owner_action_step_boundary_last_action_source") or "") != (
        "search_feedback"
    ):
        raise ValueError("compact owner action-step boundary must end on search feedback")


def _require_raw_compact_owner_action_dispatch_step_overlap_fields(
    profile_payload: Mapping[str, Any],
) -> None:
    for key in (
        "compact_owner_action_dispatch_step_overlap_enabled",
        "compact_owner_action_dispatch_step_overlap_proof_passed",
        "compact_rollout_slab_action_dispatch_step_overlap_supported",
        "compact_rollout_slab_action_dispatch_step_overlap_used",
        "compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait",
        "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper",
        "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count",
        "compact_rollout_slab_action_dispatch_step_overlap_submit_count",
        "compact_rollout_slab_action_dispatch_step_overlap_resolve_count",
        "compact_rollout_slab_action_dispatch_step_overlap_pending_count",
        "compact_rollout_slab_action_dispatch_step_overlap_max_pending_count",
        "compact_rollout_slab_action_dispatch_step_overlap_parent_work_sec",
        "compact_owner_search_action_dispatch_handle_sync_wrapper_count",
        "compact_owner_search_action_dispatch_handle_completed_at_submit_count",
        "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count",
    ):
        if key not in profile_payload:
            if key == "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count":
                raise ValueError(
                    "compact owner action dispatch overlap must report result wait-in-submit count"
                )
            raise ValueError(f"compact owner action dispatch overlap must report {key}")


def _require_compact_owner_action_dispatch_step_overlap_proof(
    profile_payload: Mapping[str, Any],
) -> None:
    _require_compact_owner_action_step_boundary_proof(profile_payload)
    if profile_payload.get("compact_owner_action_dispatch_step_overlap_enabled") is not True:
        raise ValueError("compact owner action dispatch overlap was requested but not enabled")
    if profile_payload.get("compact_owner_action_dispatch_step_overlap_proof_passed") is not True:
        raise ValueError("compact owner action dispatch overlap proof did not pass")
    if profile_payload.get("compact_rollout_slab_action_dispatch_step_overlap_supported") is not True:
        raise ValueError("compact owner action dispatch overlap was not supported")
    if profile_payload.get("compact_rollout_slab_action_dispatch_step_overlap_used") is not True:
        raise ValueError("compact owner action dispatch overlap was not used")
    if (
        profile_payload.get("compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait")
        is not True
    ):
        raise ValueError("compact owner action dispatch overlap submit waited")
    if profile_payload.get("compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper") is not False:
        raise ValueError("compact owner action dispatch overlap used sync wrapper")
    for key in (
        "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count",
        "compact_owner_search_action_dispatch_handle_sync_wrapper_count",
        "compact_owner_search_action_dispatch_handle_completed_at_submit_count",
    ):
        if key not in profile_payload:
            raise ValueError(f"compact owner action dispatch overlap must report {key}")
        if int(profile_payload.get(key) or 0) != 0:
            raise ValueError(f"compact owner action dispatch overlap must keep {key} zero")
    total_iterations = int(profile_payload.get("steps") or 0) + int(
        profile_payload.get("warmup_steps") or 0
    )
    submit_count = int(
        profile_payload.get("compact_rollout_slab_action_dispatch_step_overlap_submit_count")
        or 0
    )
    resolve_count = int(
        profile_payload.get("compact_rollout_slab_action_dispatch_step_overlap_resolve_count")
        or 0
    )
    if not (submit_count == resolve_count == total_iterations):
        raise ValueError("compact owner action dispatch overlap submit/resolve counts mismatch")
    if (
        int(
            profile_payload.get("compact_rollout_slab_action_dispatch_step_overlap_pending_count")
            or 0
        )
        != 0
    ):
        raise ValueError("compact owner action dispatch overlap left pending work")
    if (
        int(
            profile_payload.get(
                "compact_rollout_slab_action_dispatch_step_overlap_max_pending_count"
            )
            or 0
        )
        <= 0
    ):
        raise ValueError("compact owner action dispatch overlap never opened a pending handle")
    if (
        float(
            profile_payload.get(
                "compact_rollout_slab_action_dispatch_step_overlap_parent_work_sec"
            )
            or 0.0
        )
        <= 0.0
    ):
        raise ValueError("compact owner action dispatch overlap reported no parent work")
    if "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count" not in profile_payload:
        raise ValueError(
            "compact owner action dispatch overlap must report result wait-in-submit count"
        )
    if int(
        profile_payload.get(
            "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count"
        )
        or 0
    ) != 0:
        raise ValueError("compact owner action dispatch overlap waited during submit")


def _require_fixed_action_result_buffer_proof(
    profile_payload: Mapping[str, Any],
) -> None:
    if profile_payload.get("compact_owner_search_fixed_action_result_buffer_requested") is not True:
        raise ValueError("fixed action-result buffer request bit was not reported")
    if profile_payload.get("compact_owner_search_fixed_action_result_buffer_used") is not True:
        raise ValueError("fixed action-result buffer was requested but not used")
    if profile_payload.get("compact_owner_search_action_only_result") is not True:
        raise ValueError("fixed action-result buffer row must keep action-only result")
    if profile_payload.get("compact_owner_search_owner_defer_maintenance") is not True:
        raise ValueError("fixed action-result buffer requires deferred owner maintenance proof")
    if profile_payload.get("compact_owner_search_direct_root_build_request_requested") is not True:
        raise ValueError("fixed action-result buffer requires direct root build-request proof")
    if profile_payload.get("compact_owner_search_direct_root_build_request_handoff") is not True:
        raise ValueError("fixed action-result buffer requires direct root handoff")
    for key in (
        "compact_owner_search_fixed_action_result_buffer_slot_count",
        "compact_owner_search_fixed_action_result_buffer_acquire_count",
        "compact_owner_search_fixed_action_result_buffer_write_count",
        "compact_owner_search_fixed_action_result_buffer_read_count",
    ):
        if int(profile_payload.get(key) or 0) <= 0:
            raise ValueError(f"fixed action-result buffer must report positive {key}")
    acquire_count = int(
        profile_payload.get("compact_owner_search_fixed_action_result_buffer_acquire_count") or 0
    )
    write_count = int(
        profile_payload.get("compact_owner_search_fixed_action_result_buffer_write_count") or 0
    )
    read_count = int(
        profile_payload.get("compact_owner_search_fixed_action_result_buffer_read_count") or 0
    )
    if not (acquire_count == write_count == read_count):
        raise ValueError("fixed action-result buffer counts must agree")
    slot_id_raw = profile_payload.get("compact_owner_search_fixed_action_result_buffer_slot_id")
    last_slot_id_raw = profile_payload.get(
        "compact_owner_search_fixed_action_result_buffer_last_slot_id"
    )
    slot_id = int(-1 if slot_id_raw is None else slot_id_raw)
    last_slot_id = int(-1 if last_slot_id_raw is None else last_slot_id_raw)
    if slot_id < 0 or last_slot_id < 0:
        raise ValueError("fixed action-result buffer must report nonnegative slot ids")
    if (
        int(
            profile_payload.get(
                "compact_owner_search_fixed_action_result_buffer_pending_slot_count"
            )
            or 0
        )
        != 0
    ):
        raise ValueError("fixed action-result buffer must drain all pending slots")
    wire_result_bytes = int(
        profile_payload.get("compact_owner_search_fixed_action_result_buffer_wire_result_bytes")
        or 0
    )
    full_result_bytes = int(
        profile_payload.get("compact_owner_search_fixed_action_result_buffer_full_result_bytes")
        or 0
    )
    if wire_result_bytes <= 0:
        raise ValueError("fixed action-result buffer must report wire result bytes")
    if full_result_bytes <= wire_result_bytes:
        raise ValueError(
            "fixed action-result buffer must prove full result bytes exceed wire bytes"
        )


def _require_direct_root_build_request_proof(
    profile_payload: Mapping[str, Any],
) -> None:
    if profile_payload.get("compact_owner_search_resident_root_view_proved") is not True:
        raise ValueError("direct root build-request requires resident root-view proof")
    if (
        profile_payload.get("compact_rollout_slab_resident_host_observation_stub_requested")
        is not True
    ):
        raise ValueError("direct root build-request requires host-observation stub")
    if profile_payload.get("compact_owner_search_direct_root_build_request_requested") is not True:
        raise ValueError("direct root build-request request bit was not reported")
    if profile_payload.get("compact_owner_search_direct_root_build_request_handoff") is not True:
        raise ValueError("direct root build-request handoff was not proved")
    schema_id = str(
        profile_payload.get("compact_owner_search_direct_root_build_request_schema_id") or ""
    )
    if schema_id != "curvyzero_compact_root_build_request/v1":
        raise ValueError("direct root build-request schema id is invalid")
    request_kind = str(
        profile_payload.get("compact_owner_search_direct_root_build_request_kind") or ""
    )
    if request_kind != "resident_root_view_build_request_v1":
        raise ValueError("direct root build-request kind is invalid")
    publish_count = int(
        profile_payload.get("compact_owner_search_direct_root_build_request_publish_count") or 0
    )
    resolve_count = int(
        profile_payload.get("compact_owner_search_direct_root_build_request_resolve_count") or 0
    )
    owner_build_count = int(
        profile_payload.get("compact_owner_search_direct_root_owner_build_count") or 0
    )
    if publish_count <= 0:
        raise ValueError("direct root build-request row must publish requests")
    if resolve_count != publish_count:
        raise ValueError("direct root build-request publish/resolve count mismatch")
    if owner_build_count != publish_count:
        raise ValueError("direct root build-request owner build count mismatch")
    root_count = int(
        profile_payload.get("compact_owner_search_direct_root_build_request_root_count") or 0
    )
    active_root_count = int(
        profile_payload.get("compact_owner_search_direct_root_build_request_active_root_count") or 0
    )
    if root_count <= 0 or active_root_count <= 0 or active_root_count > root_count:
        raise ValueError("direct root build-request root counts are invalid")
    if (
        profile_payload.get("compact_owner_search_direct_root_build_request_observation_included")
        is not False
    ):
        raise ValueError("direct root build-request must not include observation bytes")
    for key in (
        "compact_owner_search_direct_root_build_request_observation_bytes_sent",
        "compact_owner_search_root_build_request_host_observation_bytes_sent",
        "compact_owner_search_parent_compact_root_batch_objects_sent",
        "compact_owner_search_direct_root_parent_build_call_count",
        "compact_rollout_slab_parent_root_batch_builder_call_count",
    ):
        if int(profile_payload.get(key) or 0) != 0:
            raise ValueError(f"direct root build-request row must keep {key} zero")
    for key in (
        "compact_owner_search_direct_root_parent_build_sec",
        "compact_rollout_slab_root_batch_build_sec",
    ):
        value = float(profile_payload.get(key) or 0.0)
        if not math.isfinite(value) or value != 0.0:
            raise ValueError(f"direct root build-request row must report zero {key}")
    for key in (
        "compact_owner_search_direct_root_build_request_sec",
        "compact_owner_search_direct_root_owner_build_sec",
        "compact_rollout_slab_root_build_request_sec",
    ):
        value = float(profile_payload.get(key) or 0.0)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"direct root build-request row has invalid {key}")
    if (
        profile_payload.get(
            "compact_owner_search_direct_root_build_request_resident_handle_present"
        )
        is not True
    ):
        raise ValueError("direct root build-request must carry resident root handle")
    if profile_payload.get("compact_owner_search_direct_root_parent_build_avoided") is not True:
        raise ValueError("direct root build-request must avoid parent root builder")
    if profile_payload.get("compact_owner_search_direct_root_owner_build_used") is not True:
        raise ValueError("direct root build-request must use owner root builder")
    if profile_payload.get("compact_rollout_slab_parent_root_batch_build_avoided") is not True:
        raise ValueError("direct root build-request must mark parent root build avoided")
    if profile_payload.get("compact_rollout_slab_parent_root_batch_builder_used") is not False:
        raise ValueError("direct root build-request must not use parent root builder")
    if _owner_search_mechanics_step_frame_requested(profile_payload):
        _require_owner_mechanics_step_frame_root_request_proof(profile_payload)


def _require_owner_mechanics_step_frame_root_request_proof(
    profile_payload: Mapping[str, Any],
) -> None:
    raw_keys_value = profile_payload.get("_owner_search_raw_profile_payload_keys")
    if isinstance(raw_keys_value, (frozenset, set, tuple, list)):
        raw_keys = {str(key) for key in raw_keys_value}
    else:
        raw_keys = {str(key) for key in profile_payload.keys()}
    required_fields = (
        "compact_owner_mechanics_step_frame_slot_schema_id",
        "compact_owner_mechanics_step_frame_handle_schema_id",
        "compact_owner_mechanics_step_frame_handle_ring_used",
        "compact_owner_mechanics_step_frame_handle_published",
        "compact_owner_mechanics_step_frame_handle_consumed",
        "compact_owner_mechanics_step_frame_handle_digest_verified",
        "compact_owner_mechanics_step_frame_handle_owner_digest_verified",
        "compact_owner_mechanics_step_frame_handle_resident_observation_present",
        "compact_owner_mechanics_step_frame_slot_write_count",
        "compact_owner_mechanics_parent_step_frame_build_count",
        "compact_owner_mechanics_parent_compact_batch_builder_call_count",
        "compact_owner_mechanics_parent_compact_batch_object_count",
        "compact_owner_mechanics_step_view_object_count",
        "compact_owner_step_frame_root_build_request_used",
        "compact_owner_step_frame_root_build_request_from_batch_helper_used",
        "compact_owner_step_frame_root_request_sidecar_array_bytes",
        "compact_owner_step_frame_root_request_sidecar_field_count",
        "compact_owner_root_action_context_handle_used",
        "compact_owner_root_action_context_handle_schema_id",
        "compact_owner_root_action_context_handle_id",
        "compact_owner_root_action_context_transaction_id",
        "compact_owner_root_action_context_dispatch_id",
        "compact_owner_root_action_context_root_count",
        "compact_owner_root_action_context_active_root_count",
        "compact_owner_root_action_context_context_digest",
        "compact_owner_root_action_context_owner_store_count",
        "compact_owner_root_action_context_owner_resolve_count",
        "compact_owner_root_action_context_owner_release_count",
        "compact_owner_root_action_context_owner_pending_count",
        "compact_owner_root_action_context_owner_max_pending_count",
        "compact_owner_root_action_context_owner_digest_verified",
        "compact_owner_search_pending_root_action_context_stored",
        "compact_owner_search_action_dispatch_pending_root_action_context_stored",
        "compact_owner_search_action_dispatch_pending_root_action_context_store_count",
        "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count",
        "compact_owner_search_parent_action_context_validation_count",
        "compact_owner_search_owner_action_context_validation_count",
        "compact_owner_root_search_transaction_used",
        "compact_owner_root_search_transaction_begin_count",
        "compact_owner_root_search_transaction_submit_count",
        "compact_owner_root_search_transaction_resolve_count",
        "compact_owner_root_search_transaction_pending_count",
        "compact_owner_root_search_transaction_parent_root_request_build_count",
        "compact_owner_root_search_transaction_parent_root_request_stored",
        "compact_owner_root_search_transaction_parent_compact_batch_stored",
        "compact_owner_root_search_transaction_parent_rebuild_count",
        "compact_owner_root_search_transaction_parent_root_action_context_stored",
        "compact_owner_root_search_transaction_parent_root_action_context_store_count",
        "compact_owner_root_search_transaction_parent_root_action_context_array_bytes",
        "compact_owner_root_search_transaction_parent_root_action_context_field_count",
        "compact_owner_root_search_transaction_owner_root_request_build_count",
        "compact_owner_root_search_transaction_owner_root_store_publish_count",
        "compact_owner_root_search_transaction_frame_generation_verified",
        "compact_owner_root_search_transaction_frame_digest_verified",
        "compact_owner_root_search_transaction_action_identity_verified",
        "compact_owner_root_search_transaction_applied_action_mismatch_count",
    )
    for key in required_fields:
        if key not in raw_keys:
            raise ValueError(f"owner mechanics step-frame proof must report {key}")
    slot_schema_id = str(
        profile_payload.get("compact_owner_mechanics_step_frame_slot_schema_id") or ""
    )
    if slot_schema_id != "curvyzero_compact_owner_mechanics_step_frame_slot/v1":
        raise ValueError("owner mechanics step-frame slot schema id is invalid")
    handle_schema_id = str(
        profile_payload.get("compact_owner_mechanics_step_frame_handle_schema_id") or ""
    )
    if handle_schema_id != "curvyzero_compact_owner_mechanics_step_frame_handle/v1":
        raise ValueError("owner mechanics step-frame handle schema id is invalid")
    if str(profile_payload.get("compact_owner_mechanics_step_view_schema_id") or ""):
        raise ValueError("owner mechanics step-frame must not build legacy step-view objects")
    for key in (
        "compact_owner_mechanics_step_frame_handle_ring_used",
        "compact_owner_mechanics_step_frame_handle_published",
        "compact_owner_mechanics_step_frame_handle_consumed",
        "compact_owner_mechanics_step_frame_handle_digest_verified",
        "compact_owner_mechanics_step_frame_handle_owner_digest_verified",
        "compact_owner_mechanics_step_frame_handle_resident_observation_present",
        "compact_owner_step_frame_root_build_request_used",
        "compact_owner_root_action_context_handle_used",
        "compact_owner_root_action_context_owner_digest_verified",
        "compact_owner_root_search_transaction_used",
        "compact_owner_root_search_transaction_frame_generation_verified",
        "compact_owner_root_search_transaction_frame_digest_verified",
        "compact_owner_root_search_transaction_action_identity_verified",
    ):
        if profile_payload.get(key) is not True:
            raise ValueError(f"owner mechanics step-frame proof requires {key}")
    if profile_payload.get("compact_owner_step_frame_root_build_request_from_batch_helper_used") is not False:
        raise ValueError("owner mechanics step-frame must bypass root request from-batch helper")
    for key in (
        "compact_owner_mechanics_step_frame_slot_write_count",
        "compact_owner_mechanics_step_frame_handle_publish_count",
        "compact_owner_mechanics_step_frame_handle_consume_count",
        "compact_owner_root_search_transaction_begin_count",
        "compact_owner_root_search_transaction_submit_count",
        "compact_owner_root_search_transaction_resolve_count",
        "compact_owner_root_search_transaction_owner_root_request_build_count",
        "compact_owner_root_search_transaction_owner_root_store_publish_count",
        "compact_owner_root_action_context_owner_store_count",
        "compact_owner_root_action_context_owner_resolve_count",
        "compact_owner_root_action_context_owner_release_count",
        "compact_owner_search_action_dispatch_pending_root_action_context_avoided_count",
        "compact_owner_search_owner_action_context_validation_count",
    ):
        if int(profile_payload.get(key) or 0) <= 0:
            raise ValueError(f"owner mechanics step-frame must report positive {key}")
    if (
        str(profile_payload.get("compact_owner_root_action_context_handle_schema_id") or "")
        != "curvyzero_compact_owner_root_action_context_handle/v1"
    ):
        raise ValueError("owner mechanics step-frame root-action context handle schema is invalid")
    if not str(profile_payload.get("compact_owner_root_action_context_context_digest") or ""):
        raise ValueError("owner mechanics step-frame must report root-action context digest")
    root_count = int(profile_payload.get("compact_owner_root_action_context_root_count") or 0)
    active_root_count = int(
        profile_payload.get("compact_owner_root_action_context_active_root_count") or 0
    )
    if root_count <= 0 or active_root_count <= 0 or active_root_count > root_count:
        raise ValueError("owner mechanics step-frame root-action context counts are invalid")
    if int(profile_payload.get("compact_owner_mechanics_step_frame_handle_publish_count") or 0) != int(
        profile_payload.get("compact_owner_mechanics_step_frame_handle_consume_count") or 0
    ):
        raise ValueError("owner mechanics step-frame publish/consume count mismatch")
    boundary_count = int(profile_payload.get("compact_owner_mechanics_step_boundary_count") or 0)
    slot_write_count = int(
        profile_payload.get("compact_owner_mechanics_step_frame_slot_write_count") or 0
    )
    if boundary_count > 0 and slot_write_count != boundary_count:
        raise ValueError("owner mechanics step-frame slot write count mismatch")
    for key in (
        "compact_owner_root_search_transaction_begin_count",
        "compact_owner_root_search_transaction_submit_count",
        "compact_owner_root_search_transaction_resolve_count",
        "compact_owner_root_search_transaction_owner_root_request_build_count",
        "compact_owner_root_search_transaction_owner_root_store_publish_count",
        "compact_owner_root_action_context_owner_store_count",
        "compact_owner_root_action_context_owner_resolve_count",
        "compact_owner_root_action_context_owner_release_count",
    ):
        if boundary_count > 0 and int(profile_payload.get(key) or 0) != boundary_count:
            raise ValueError("owner mechanics step-frame transaction count mismatch")
    for key in (
        "compact_owner_mechanics_parent_step_frame_build_count",
        "compact_owner_mechanics_parent_compact_batch_builder_call_count",
        "compact_owner_mechanics_parent_compact_batch_object_count",
        "compact_owner_mechanics_step_view_object_count",
        "compact_owner_step_frame_root_request_sidecar_array_bytes",
        "compact_owner_step_frame_root_request_sidecar_field_count",
        "compact_owner_root_search_transaction_pending_count",
        "compact_owner_root_search_transaction_parent_root_request_build_count",
        "compact_owner_root_search_transaction_parent_rebuild_count",
        "compact_owner_root_action_context_owner_pending_count",
        "compact_owner_search_action_dispatch_pending_root_action_context_store_count",
        "compact_owner_search_parent_action_context_validation_count",
        "compact_owner_root_search_transaction_parent_root_action_context_store_count",
        "compact_owner_root_search_transaction_parent_root_action_context_array_bytes",
        "compact_owner_root_search_transaction_parent_root_action_context_field_count",
        "compact_owner_root_search_transaction_applied_action_mismatch_count",
    ):
        if int(profile_payload.get(key) or 0) != 0:
            raise ValueError(f"owner mechanics step-frame must keep {key} zero")
    for key in (
        "compact_owner_root_search_transaction_parent_root_request_stored",
        "compact_owner_root_search_transaction_parent_compact_batch_stored",
        "compact_owner_search_pending_root_action_context_stored",
        "compact_owner_search_action_dispatch_pending_root_action_context_stored",
        "compact_owner_root_search_transaction_parent_root_action_context_stored",
    ):
        if profile_payload.get(key) is not False:
            raise ValueError(f"owner mechanics step-frame must keep {key} false")


def _require_resident_root_host_observation_stub_proof(
    profile_payload: Mapping[str, Any],
) -> None:
    if profile_payload.get("compact_owner_search_resident_root_view_proved") is not True:
        raise ValueError("resident host-observation stub requires resident root-view proof")
    if (
        profile_payload.get("compact_rollout_slab_resident_host_observation_stub_requested")
        is not True
    ):
        raise ValueError("resident host-observation stub request was not reported")
    if profile_payload.get("compact_rollout_slab_resident_host_observation_stubbed") is not True:
        raise ValueError("resident host-observation stub was requested but not used")
    if (
        str(profile_payload.get("compact_rollout_slab_resident_host_observation_stub_kind") or "")
        != "zero_stride_shape_only_v1"
    ):
        raise ValueError("resident host-observation stub kind is invalid")
    if (
        int(
            profile_payload.get(
                "compact_rollout_slab_resident_host_observation_stub_materialized_bytes"
            )
            or 0
        )
        != 0
    ):
        raise ValueError("resident host-observation stub materialized host bytes")
    if (
        int(
            profile_payload.get("compact_rollout_slab_resident_host_observation_stub_logical_bytes")
            or 0
        )
        <= 0
    ):
        raise ValueError("resident host-observation stub logical bytes must be positive")


def _require_direct_resident_root_view_proof(
    profile_payload: Mapping[str, Any],
) -> None:
    if profile_payload.get("compact_owner_search_resident_root_view_required") is not True:
        raise ValueError("resident root-view row must report proof requirement")
    if profile_payload.get("compact_owner_search_resident_root_view_proved") is not True:
        raise ValueError("resident root-view row must prove owner consumed resident root view")
    kind = str(profile_payload.get("compact_owner_search_resident_root_view_kind") or "")
    if kind != "direct_root_batch_resident_handle_v1":
        raise ValueError("resident root-view row has wrong proof kind")
    if int(profile_payload.get("compact_owner_search_resident_root_view_generation_id") or 0) <= 0:
        raise ValueError("resident root-view row must report positive generation id")
    if not str(profile_payload.get("compact_owner_search_resident_root_view_device") or ""):
        raise ValueError("resident root-view row must report resident device")
    if (
        profile_payload.get("compact_owner_search_resident_root_view_host_fallback_allowed")
        is not False
    ):
        raise ValueError("resident root-view row must forbid host fallback")
    if profile_payload.get("compact_owner_search_resident_root_view_row_major_order") is not True:
        raise ValueError("resident root-view row must prove row-major resident roots")
    for key in (
        "compact_owner_search_resident_root_view_h2d_bytes",
        "compact_owner_search_resident_root_view_d2h_bytes",
    ):
        value = float(profile_payload.get(key) or 0.0)
        if not math.isfinite(value) or value != 0.0:
            raise ValueError(f"resident root-view row must report zero {key}")
    if not profile_payload.get("compact_owner_search_resident_root_view_root_shape"):
        raise ValueError("resident root-view row must report root shape")
    if not profile_payload.get("compact_owner_search_resident_root_view_stack_shape"):
        raise ValueError("resident root-view row must report stack shape")


def _require_compact_torch_owner_search_bridge_proof(
    profile_payload: Mapping[str, Any],
) -> None:
    if profile_payload.get("compact_owner_search_resident_root_bridge_ready") is not True:
        raise ValueError("compact Torch owner-search row must prove runtime resident-root bridge")
    bridge_kind = str(profile_payload.get("compact_owner_search_resident_root_bridge_kind") or "")
    if bridge_kind != "shared_memory_host_root_to_owner_resident_tensor_v1":
        raise ValueError("compact Torch owner-search row has wrong resident-root bridge kind")
    bridge_device = str(
        profile_payload.get("compact_owner_search_resident_root_bridge_device") or ""
    )
    if not bridge_device:
        raise ValueError("compact Torch owner-search row must report bridge device")
    bridge_bytes = float(
        profile_payload.get("compact_owner_search_resident_root_bridge_h2d_bytes") or 0.0
    )
    if not math.isfinite(bridge_bytes) or bridge_bytes <= 0.0:
        raise ValueError("compact Torch owner-search row must report positive bridge bytes")
    if (
        int(profile_payload.get("compact_owner_search_resident_root_bridge_generation_id") or 0)
        <= 0
    ):
        raise ValueError("compact Torch owner-search row must report bridge generation id")
    fallback_count = float(profile_payload.get("resident_observation_host_fallback_count") or 0.0)
    if not math.isfinite(fallback_count) or fallback_count != 0.0:
        raise ValueError("compact Torch owner-search row must not use resident host fallback")


def _require_inline_owner_search_direct_root_proof(
    profile_payload: Mapping[str, Any],
) -> None:
    if profile_payload.get("compact_direct_root_store") is not True:
        raise ValueError("inline owner-search row must prove direct root store")
    if profile_payload.get("compact_owner_search_direct_root_handoff") is not True:
        raise ValueError("inline owner-search row must prove direct root handoff")
    if profile_payload.get("compact_owner_search_direct_root_rebuild_avoided") is not True:
        raise ValueError("inline owner-search row must prove root rebuild is avoided")
    if profile_payload.get("compact_owner_search_direct_root_resolved") is not True:
        raise ValueError("inline owner-search row must prove direct root resolution")
    if int(profile_payload.get("compact_direct_root_store_publish_count") or 0) <= 0:
        raise ValueError("inline owner-search row must publish direct roots")
    if int(profile_payload.get("compact_direct_root_store_resolve_count") or 0) <= 0:
        raise ValueError("inline owner-search row must resolve direct roots")
    if int(profile_payload.get("compact_direct_root_store_last_root_slot_count") or 0) <= 0:
        raise ValueError("inline owner-search row must report direct root slot count")
    if (
        int(profile_payload.get("compact_owner_search_direct_root_observation_bytes_sent") or 0)
        != 0
    ):
        raise ValueError("inline owner-search row must send zero direct root observation bytes")
    fallback_count = float(profile_payload.get("resident_observation_host_fallback_count") or 0.0)
    if not math.isfinite(fallback_count) or fallback_count != 0.0:
        raise ValueError("inline owner-search row must not use resident host fallback")


def _require_deferred_learner_proof(
    args: argparse.Namespace,
    profile_payload: Mapping[str, Any],
) -> None:
    if not bool(getattr(args, "compact_owned_loop_deferred_learner", False)):
        return
    if profile_payload.get("compact_owned_loop_defer_learner_gate") is not True:
        raise ValueError("deferred learner row must enable the profile loop defer gate")
    submit_count = int(profile_payload.get("compact_owned_loop_deferred_learner_submit_count") or 0)
    completed_count = int(
        profile_payload.get("compact_owned_loop_deferred_learner_completed_count") or 0
    )
    if submit_count <= 0:
        raise ValueError("deferred learner row must submit learner work")
    if completed_count != submit_count:
        raise ValueError("deferred learner row must complete every submitted learner job")
    if bool(profile_payload.get("compact_owned_loop_deferred_learner_pending", False)):
        raise ValueError("deferred learner row must drain pending learner work")
    if int(profile_payload.get("compact_owned_loop_deferred_learner_pending_count") or 0) != 0:
        raise ValueError("deferred learner row must end with pending count 0")
    if (
        int(profile_payload.get("compact_owned_loop_deferred_learner_max_pending_observed") or 0)
        <= 0
    ):
        raise ValueError("deferred learner row must observe at least one pending learner job")


def _require_deferred_sample_learner_proof(
    args: argparse.Namespace,
    profile_payload: Mapping[str, Any],
) -> None:
    if not bool(getattr(args, "compact_owned_loop_deferred_sample_learner", False)):
        return
    if profile_payload.get("compact_owned_loop_defer_sample_learner_gate") is not True:
        raise ValueError("deferred sample+learner row must enable the profile loop defer gate")
    requested_worker_kind = str(
        getattr(
            args,
            "compact_owned_loop_sample_learner_worker_kind",
            COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD,
        )
    )
    actual_worker_kind = str(
        profile_payload.get("compact_owned_loop_sample_learner_worker_kind") or "none"
    )
    if actual_worker_kind != requested_worker_kind:
        raise ValueError("deferred sample+learner worker kind mismatch")
    submit_count = int(
        profile_payload.get("compact_owned_loop_deferred_sample_learner_submit_count") or 0
    )
    completed_count = int(
        profile_payload.get("compact_owned_loop_deferred_sample_learner_completed_count") or 0
    )
    sample_calls = int(profile_payload.get("compact_rollout_slab_sample_gate_calls") or 0)
    learner_calls = int(profile_payload.get("compact_rollout_slab_learner_gate_calls") or 0)
    learner_updates = int(profile_payload.get("compact_rollout_slab_learner_gate_updates") or 0)
    max_pending = int(
        profile_payload.get("compact_owned_loop_deferred_sample_learner_max_pending") or 0
    )
    max_pending_observed = int(
        profile_payload.get("compact_owned_loop_deferred_sample_learner_max_pending_observed") or 0
    )
    if submit_count <= 0:
        raise ValueError("deferred sample+learner row must submit work")
    if completed_count != submit_count:
        raise ValueError("deferred sample+learner row must complete every submitted job")
    if bool(profile_payload.get("compact_owned_loop_deferred_sample_learner_pending", False)):
        raise ValueError("deferred sample+learner row must drain pending work")
    if (
        int(profile_payload.get("compact_owned_loop_deferred_sample_learner_pending_count") or 0)
        != 0
    ):
        raise ValueError("deferred sample+learner row must end with pending count 0")
    if max_pending <= 0:
        raise ValueError("deferred sample+learner row must report positive max pending")
    if max_pending_observed <= 0:
        raise ValueError("deferred sample+learner row must observe pending work")
    if max_pending_observed > max_pending:
        raise ValueError("deferred sample+learner row exceeded max pending")
    if profile_payload.get("compact_owned_loop_deferred_sample_learner_drained") is not True:
        raise ValueError("deferred sample+learner row must prove staged work drained")
    if profile_payload.get("compact_owned_loop_final_deferred_drain_in_measured_sec") is not True:
        raise ValueError("deferred sample+learner row must include final drain in wall time")
    if sample_calls != completed_count:
        raise ValueError("deferred sample+learner row must preserve replay sample calls")
    if learner_calls <= 0:
        raise ValueError("deferred sample+learner row must preserve learner calls")
    if learner_calls > completed_count:
        raise ValueError("deferred sample+learner row reported more learner calls than samples")
    if learner_updates <= 0:
        raise ValueError("deferred sample+learner row must preserve learner updates")
    if requested_worker_kind != COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD:
        if (
            str(
                profile_payload.get("compact_owned_loop_sample_learner_worker_resource_scope") or ""
            )
            != "process"
        ):
            raise ValueError("local-process sample+learner row must report process scope")
        if (
            str(profile_payload.get("compact_owned_loop_sample_learner_worker_start_method") or "")
            != "spawn"
        ):
            raise ValueError("local-process sample+learner row must use spawn start method")
        if str(getattr(args, "learner_device", "")).startswith("cuda") and (
            str(
                profile_payload.get(
                    "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings"
                )
                or ""
            )
            != "expandable_segments:False"
        ):
            raise ValueError(
                "local-process CUDA sample+learner row must configure CUDA IPC allocator"
            )
        bootstrap_source = str(
            profile_payload.get("compact_owned_loop_sample_learner_worker_bootstrap_source") or ""
        )
        if bootstrap_source not in {"parent_learner", "factory"}:
            raise ValueError("local-process sample+learner row must report bootstrap source")
        if str(getattr(args, "learner_device", "")).startswith("cuda") and (
            bootstrap_source != "factory"
        ):
            raise ValueError(
                "local-process CUDA sample+learner row must bootstrap from a host-only factory"
            )
        if (
            profile_payload.get(
                "compact_owned_loop_sample_learner_resource_distinct_from_actor_search"
            )
            is not True
        ):
            raise ValueError("local-process sample+learner row must report a distinct resource")
        if (
            profile_payload.get(
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "last_completed_worker_pid_distinct_from_actor_search"
                )
            )
            is not True
        ):
            raise ValueError(
                "local-process sample+learner row must prove completed work ran in "
                "a different process"
            )
        if (
            profile_payload.get(
                "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search"
            )
            is True
        ):
            raise ValueError(
                "local-process sample+learner row must not claim a distinct hardware resource"
            )
        if (
            int(
                profile_payload.get(
                    "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending"
                )
                or 0
            )
            <= 0
        ):
            raise ValueError(
                "local-process sample+learner row must collect actor steps while work is pending"
            )
        if (
            int(
                profile_payload.get("compact_owned_loop_deferred_sample_learner_policy_lag_max")
                or 0
            )
            <= 0
        ):
            raise ValueError("local-process sample+learner row must report policy lag")
        if (
            int(
                profile_payload.get(
                    "compact_owned_loop_deferred_sample_learner_last_submitted_request_id"
                )
                or 0
            )
            != submit_count
        ):
            raise ValueError("local-process sample+learner row must close submitted request ids")
        if (
            int(
                profile_payload.get(
                    "compact_owned_loop_deferred_sample_learner_last_completed_request_id"
                )
                or 0
            )
            != completed_count
        ):
            raise ValueError("local-process sample+learner row must close completed request ids")
        model_state_apply_count = int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_model_state_apply_count"
            )
            or 0
        )
        model_state_return_count = int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_model_state_return_count"
            )
            or model_state_apply_count
        )
        model_state_omitted_count = int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_model_state_omitted_count"
            )
            or 0
        )
        model_state_transport_kind = str(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_model_state_transport_kind",
                COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
            )
        )
        if model_state_transport_kind == COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1:
            model_owner_ref_return_count = int(
                profile_payload.get(
                    ("compact_owned_loop_deferred_sample_learner_model_owner_ref_return_count")
                )
                or 0
            )
            if model_state_apply_count != 0 or model_state_return_count != 0:
                raise ValueError("owner_ref_v1 row must not return or apply parent model state")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "model_state_snapshot_return_count"
                        )
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError("owner_ref_v1 row must not return snapshot model state")
            if learner_calls > 0 and model_owner_ref_return_count <= 0:
                raise ValueError("owner_ref_v1 row must return at least one owner ref")
            if learner_calls > 0 and (
                profile_payload.get(
                    ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_returned")
                )
                is not True
            ):
                raise ValueError("owner_ref_v1 row must return the final owner ref")
            if (
                learner_calls > 0
                and not str(
                    profile_payload.get(
                        ("compact_owned_loop_deferred_sample_learner_last_model_owner_ref_digest"),
                        "",
                    )
                ).strip()
            ):
                raise ValueError("owner_ref_v1 row must report final owner-ref digest")
        else:
            if model_state_apply_count != model_state_return_count:
                raise ValueError(
                    "local-process sample+learner row must apply every returned model state"
                )
            if learner_calls > 0 and model_state_return_count <= 0:
                raise ValueError(
                    "local-process sample+learner row must return at least one model state"
                )
            if model_state_return_count < learner_calls and model_state_omitted_count <= 0:
                raise ValueError(
                    "local-process sample+learner row must report omitted model states"
                )
            if (
                learner_calls > 0
                and model_state_return_count > 0
                and (
                    profile_payload.get(
                        "compact_owned_loop_deferred_sample_learner_last_model_state_applied"
                    )
                    is not True
                )
            ):
                raise ValueError(
                    "local-process sample+learner row must apply the final returned model state"
                )
        if (
            profile_payload.get("compact_owned_loop_deferred_sample_learner_request_host_only")
            is not True
        ):
            raise ValueError("local-process sample+learner row must use host-only requests")
        if (
            int(
                profile_payload.get(
                    "compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count"
                )
                or 0
            )
            != 0
        ):
            raise ValueError("local-process sample+learner request must contain no CUDA tensors")
        if (
            profile_payload.get("compact_owned_loop_deferred_sample_learner_result_host_only")
            is not True
        ):
            raise ValueError("local-process sample+learner row must return host-only results")
        if (
            int(
                profile_payload.get(
                    "compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count"
                )
                or 0
            )
            != 0
        ):
            raise ValueError("local-process sample+learner result must contain no CUDA tensors")
        if (
            int(
                profile_payload.get("compact_owned_loop_deferred_sample_learner_request_bytes") or 0
            )
            <= 0
        ):
            raise ValueError("local-process sample+learner row must report request bytes")
        if (
            int(profile_payload.get("compact_owned_loop_deferred_sample_learner_result_bytes") or 0)
            <= 0
        ):
            raise ValueError("local-process sample+learner row must report result bytes")
        if (
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_owns_model_state"
            )
            is not True
        ):
            raise ValueError("local-process sample+learner worker must own model state")
        if (
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_worker_owns_replay_store"
            )
            is not True
        ):
            raise ValueError("local-process sample+learner worker must own replay store")
        if (
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent"
            )
            is True
        ):
            raise ValueError("local-process sample+learner must not send full replay snapshots")
        if (
            int(
                profile_payload.get(
                    ("compact_owned_loop_deferred_sample_learner_full_replay_snapshot_submit_count")
                )
                or 0
            )
            != 0
        ):
            raise ValueError("local-process sample+learner sent full replay snapshots")
        replay_append_entry_count = int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_replay_append_entry_count"
            )
            or 0
        )
        if replay_append_entry_count <= 0:
            raise ValueError("local-process sample+learner sent no replay append entries")
        append_entry_bytes = int(
            profile_payload.get(
                "compact_owned_loop_deferred_sample_learner_replay_append_entry_bytes"
            )
            or 0
        )
        if append_entry_bytes <= 0:
            raise ValueError("local-process sample+learner must report replay append bytes")
        if (
            int(
                profile_payload.get(
                    (
                        "compact_owned_loop_deferred_sample_learner_"
                        "replay_append_host_observation_bytes"
                    )
                )
                or 0
            )
            > 0
        ):
            raise ValueError("local-process sample+learner still sends host replay observations")
        if (
            int(
                profile_payload.get(
                    (
                        "compact_owned_loop_deferred_sample_learner_"
                        "replay_append_resident_snapshot_count"
                    )
                )
                or 0
            )
            > 0
        ):
            raise ValueError("local-process sample+learner still sends resident replay snapshots")
        transport_kind = str(
            profile_payload.get(
                ("compact_owned_loop_deferred_sample_learner_replay_append_transport_kind")
            )
            or ""
        )
        if transport_kind == COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1:
            if (
                profile_payload.get(
                    (
                        "compact_owned_loop_deferred_sample_learner_"
                        "worker_observation_provider_present"
                    )
                )
                is not True
            ):
                raise ValueError("scalar-ref local-process row must configure a provider")
            if (
                int(
                    profile_payload.get(
                        ("compact_owned_loop_deferred_sample_learner_provider_bootstrap_step_count")
                    )
                    or 0
                )
                <= 0
            ):
                raise ValueError("scalar-ref local-process row must send provider bootstrap")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "worker_observation_provider_bootstrap_step_count"
                        )
                    )
                    or 0
                )
                <= 0
            ):
                raise ValueError("scalar-ref worker provider must apply bootstrap")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "provider_bootstrap_host_observation_bytes"
                        )
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError("scalar-ref provider bootstrap must not send observations")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "provider_bootstrap_resident_snapshot_count"
                        )
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError("scalar-ref provider bootstrap must not send resident snapshots")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "provider_bootstrap_resident_snapshot_bytes"
                        )
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError(
                    "scalar-ref provider bootstrap must not send resident snapshot bytes"
                )
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "provider_bootstrap_replay_entry_count"
                        )
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError("scalar-ref provider bootstrap must not send replay entries")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "provider_bootstrap_replay_index_row_count"
                        )
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError("scalar-ref provider bootstrap must not send replay rows")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "provider_bootstrap_learner_call_count"
                        )
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError("scalar-ref provider bootstrap must not send learner calls")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "provider_bootstrap_render_state_bytes"
                        )
                    )
                    or 0
                )
                <= 0
            ):
                raise ValueError("scalar-ref provider bootstrap must send render-state facts")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "replay_append_render_state_bytes"
                        )
                    )
                    or 0
                )
                <= 0
            ):
                raise ValueError("scalar-ref replay append must send render-state facts")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "worker_observation_provider_missing_stack_history_count"
                        )
                    )
                    or 0
                )
                != 0
            ):
                raise ValueError("scalar-ref provider reported missing stack history")
            if (
                int(
                    profile_payload.get(
                        (
                            "compact_owned_loop_deferred_sample_learner_"
                            "worker_observation_provider_materialized_entry_count"
                        )
                    )
                    or 0
                )
                != replay_append_entry_count
            ):
                raise ValueError("scalar-ref provider must materialize every append entry")
        if (
            int(
                profile_payload.get(
                    "compact_owned_loop_deferred_sample_learner_worker_replay_append_count"
                )
                or 0
            )
            != replay_append_entry_count
        ):
            raise ValueError("local-process worker replay append count mismatch")
        if (
            int(
                profile_payload.get(
                    "compact_owned_loop_deferred_sample_learner_worker_replay_index_row_count"
                )
                or 0
            )
            <= 0
        ):
            raise ValueError("local-process worker replay index row count missing")
        if (
            int(
                profile_payload.get(
                    ("compact_owned_loop_deferred_sample_learner_worker_model_initialized_count")
                )
                or 0
            )
            != 1
        ):
            raise ValueError("local-process sample+learner worker must initialize once")
        if (
            int(
                profile_payload.get(
                    "compact_owned_loop_deferred_sample_learner_worker_completed_count"
                )
                or 0
            )
            != completed_count
        ):
            raise ValueError("local-process sample+learner worker completed count mismatch")
        if str(getattr(args, "learner_device", "")).startswith("cuda"):
            worker_cuda_device = str(
                profile_payload.get(
                    "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device"
                )
                or ""
            )
            if not worker_cuda_device.startswith("cuda"):
                raise ValueError("local-process CUDA sample+learner row must prove worker CUDA use")
            if learner_calls != completed_count:
                raise ValueError(
                    "local-process CUDA sample+learner row must train every completed job"
                )


def _require_compact_torch_policy_refresh_proof(profile_payload: dict[str, Any]) -> None:
    if (
        profile_payload.get("compact_rollout_slab_policy_refresh_after_learner_gate_enabled")
        is not True
    ):
        raise ValueError("compact Torch speed row requires learner-to-search refresh")
    refresh_calls = int(
        profile_payload.get("compact_rollout_slab_policy_refresh_after_learner_gate_calls") or 0
    )
    learner_updates = int(profile_payload.get("compact_rollout_slab_learner_gate_updates") or 0)
    last_update = int(
        profile_payload.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count"
        )
        or 0
    )
    if refresh_calls <= 0:
        raise ValueError("compact Torch speed row requires refresh calls")
    if last_update != learner_updates:
        raise ValueError("compact Torch speed row refresh must reach final learner update")
    if (
        profile_payload.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_search_worker_distinct_from_learner"
        )
        is not True
    ):
        raise ValueError("compact Torch speed row requires distinct learner/search models")
    if not str(
        profile_payload.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest"
        )
        or ""
    ).strip():
        raise ValueError("compact Torch speed row requires refreshed model digest")
    final_digest = str(
        profile_payload.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest"
        )
        or ""
    ).strip()
    if (
        int(
            profile_payload.get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_count"
            )
            or 0
        )
        <= 0
    ):
        raise ValueError("compact Torch speed row requires a post-refresh search step")
    if (
        int(
            profile_payload.get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_count"
            )
            or 0
        )
        <= 0
    ):
        raise ValueError("compact Torch speed row requires post-refresh replay rows")
    refresh_transport_kind = str(
        profile_payload.get("compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind")
        or profile_payload.get(
            "compact_owned_loop_deferred_sample_learner_model_state_transport_kind"
        )
        or ""
    )
    if refresh_transport_kind == COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1:
        owner_ref_refresh_count = int(
            profile_payload.get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count"
            )
            or 0
        )
        model_state_refresh_count = int(
            profile_payload.get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count"
            )
            or 0
        )
        last_transport_kind = str(
            profile_payload.get(
                "compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind"
            )
            or ""
        )
        if owner_ref_refresh_count != refresh_calls:
            raise ValueError("owner_ref_v1 refresh row must use owner refs for every refresh")
        if model_state_refresh_count != 0:
            raise ValueError("owner_ref_v1 refresh row must not use parent model state")
        if last_transport_kind != COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1:
            raise ValueError("owner_ref_v1 refresh row must report final owner-ref transport")
        if (
            profile_payload.get(
                (
                    "compact_rollout_slab_policy_refresh_after_learner_gate_"
                    "parent_model_state_transport_avoided"
                )
            )
            is not True
        ):
            raise ValueError("owner_ref_v1 refresh row must avoid parent model-state transport")
    _require_exact_policy_refresh_metadata(
        profile_payload,
        field="compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata",
        label="search",
        expected_update_count=last_update,
        expected_digest=final_digest,
    )
    replay_metadata_field = (
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata"
    )
    if profile_payload.get("compact_owned_loop_defer_sample_learner_gate") is True:
        _require_lagged_policy_refresh_metadata(
            profile_payload,
            field=replay_metadata_field,
            label="replay",
            final_update_count=last_update,
        )
    else:
        _require_exact_policy_refresh_metadata(
            profile_payload,
            field=replay_metadata_field,
            label="replay",
            expected_update_count=last_update,
            expected_digest=final_digest,
        )


def _require_exact_policy_refresh_metadata(
    profile_payload: Mapping[str, Any],
    *,
    field: str,
    label: str,
    expected_update_count: int,
    expected_digest: str,
) -> None:
    metadata = profile_payload.get(field)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"compact Torch speed row requires final post-refresh {label} metadata")
    if metadata.get("compact_policy_refresh_search_worker_refreshed") is not True:
        raise ValueError(f"compact Torch speed row {label} metadata is not refreshed")
    if int(metadata.get("compact_policy_refresh_learner_update_count") or 0) != int(
        expected_update_count
    ):
        raise ValueError(f"compact Torch speed row final {label} metadata update count mismatch")
    digest = str(metadata.get("compact_policy_refresh_model_state_digest") or "").strip()
    if digest != str(expected_digest):
        raise ValueError(f"compact Torch speed row final {label} metadata digest mismatch")


def _require_lagged_policy_refresh_metadata(
    profile_payload: Mapping[str, Any],
    *,
    field: str,
    label: str,
    final_update_count: int,
) -> None:
    metadata = profile_payload.get(field)
    if not isinstance(metadata, Mapping):
        raise ValueError(f"compact Torch speed row requires post-refresh {label} metadata")
    if metadata.get("compact_policy_refresh_search_worker_refreshed") is not True:
        raise ValueError(f"compact Torch speed row {label} metadata is not refreshed")
    update_count = int(metadata.get("compact_policy_refresh_learner_update_count") or 0)
    if update_count <= 0:
        raise ValueError(f"compact Torch speed row {label} metadata update count missing")
    if update_count > int(final_update_count):
        raise ValueError(f"compact Torch speed row {label} metadata update count exceeds final")
    digest = str(metadata.get("compact_policy_refresh_model_state_digest") or "").strip()
    if not digest:
        raise ValueError(f"compact Torch speed row {label} metadata digest missing")


def _policy_refresh_metadata_update_count(metadata: Any) -> int:
    if not isinstance(metadata, Mapping):
        return 0
    return int(metadata.get("compact_policy_refresh_learner_update_count") or 0)


def _policy_refresh_proof_fields(profile_payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "compact_rollout_slab_policy_refresh_after_learner_gate_enabled",
        "compact_rollout_slab_policy_refresh_after_learner_gate_calls",
        "compact_rollout_slab_policy_refresh_after_learner_gate_interval",
        "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count",
        "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count",
        "compact_rollout_slab_policy_refresh_after_learner_gate_sec",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest",
        "compact_rollout_slab_policy_refresh_after_learner_gate_search_worker_distinct_from_learner",
        "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_count",
        "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_count",
        "compact_rollout_slab_policy_refresh_after_learner_gate_sample_metadata_count",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_worker_state",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_metadata",
        "compact_rollout_slab_policy_refresh_after_learner_gate_service_total_sec",
        "compact_rollout_slab_policy_refresh_after_learner_gate_state_load_sec",
        "compact_rollout_slab_policy_refresh_after_learner_gate_model_digest_sec",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_service_total_sec",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_state_load_sec",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_digest_sec",
        "compact_rollout_slab_policy_refresh_after_learner_gate_transport_kind",
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_transport_kind",
        "compact_rollout_slab_policy_refresh_after_learner_gate_owner_ref_used_count",
        "compact_rollout_slab_policy_refresh_after_learner_gate_model_state_used_count",
        (
            "compact_rollout_slab_policy_refresh_after_learner_gate_"
            "parent_model_state_transport_avoided"
        ),
    )
    fields = {key: profile_payload.get(key) for key in keys}
    final_update = int(
        profile_payload.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_update_count"
        )
        or 0
    )
    search_update = _policy_refresh_metadata_update_count(
        profile_payload.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata"
        )
    )
    replay_update = _policy_refresh_metadata_update_count(
        profile_payload.get(
            "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata"
        )
    )
    fields[
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_search_metadata_update_count"
    ] = search_update
    fields[
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_replay_metadata_update_count"
    ] = replay_update
    fields[
        "compact_rollout_slab_policy_refresh_after_learner_gate_search_metadata_lag_to_final_update"
    ] = max(0, final_update - search_update)
    fields[
        "compact_rollout_slab_policy_refresh_after_learner_gate_replay_metadata_lag_to_final_update"
    ] = max(0, final_update - replay_update)
    return fields


class _TinyMuZero(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 16),
            torch.nn.Tanh(),
        )
        self.action_embedding = torch.nn.Embedding(ACTION_COUNT, 16)
        self.policy_head = torch.nn.Linear(16, ACTION_COUNT)
        self.value_head = torch.nn.Linear(16, 3)
        self.reward_head = torch.nn.Linear(16, 3)

    def initial_inference(self, obs: Any) -> Any:
        from lzero.model.common import MZNetworkOutput

        latent = self.encoder(obs)
        value = self.value_head(latent)
        return MZNetworkOutput(
            value,
            value.new_zeros(value.shape),
            self.policy_head(latent),
            latent,
        )

    def recurrent_inference(self, latent_state: Any, action: Any) -> Any:
        from lzero.model.common import MZNetworkOutput

        next_latent = torch.tanh(latent_state + self.action_embedding(action.reshape(-1).long()))
        return MZNetworkOutput(
            self.value_head(next_latent),
            self.reward_head(next_latent),
            self.policy_head(next_latent),
            next_latent,
        )


class _ModelPolicy:
    def __init__(self, model: Any) -> None:
        self._model = model


class _DeviceTargetSearchService:
    supports_two_phase_compact_search = True
    search_impl = "compact_coach_speed_row_device_target_search"
    num_simulations = 1

    def __init__(
        self,
        *,
        seed: int,
        model: Any | None = None,
        num_simulations: int = 1,
        loaded_checkpoint_identity: Mapping[str, Any] | None = None,
    ) -> None:
        import torch

        torch.manual_seed(int(seed))
        self._model = model if model is not None else _TinyMuZero()
        self.num_simulations = int(num_simulations)
        self.loaded_checkpoint_identity = dict(loaded_checkpoint_identity or {})
        self.counter = 0
        self.pending: dict[str, tuple[Any, ...]] = {}

    def run_action_step(self, root_batch: Any) -> CompactSearchActionStepV1:
        active_roots = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
        selected = np.asarray(
            [self.counter % ACTION_COUNT for _ in range(active_roots.size)],
            dtype=np.int16,
        )
        handle = f"coach-speed-row:{self.counter}"
        self.counter += 1
        self.pending[handle] = (
            active_roots,
            root_batch.env_row[active_roots].astype(np.int32, copy=True),
            root_batch.player[active_roots].astype(np.int16, copy=True),
            root_batch.policy_env_id[active_roots].astype(np.int64, copy=True),
            selected.copy(),
            self.counter,
        )
        return CompactSearchActionStepV1(
            replay_payload_handle=handle,
            root_index=active_roots,
            env_row=self.pending[handle][1],
            player=self.pending[handle][2],
            policy_env_id=self.pending[handle][3],
            selected_action=selected,
            metadata={
                "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                "phase": "action_critical",
                "search_impl": self.search_impl,
                "num_simulations": self.num_simulations,
                "active_root_count": int(active_roots.size),
                "replay_payload_origin": f"{self.search_impl}:{handle}",
                "selected_action_digest": compact_search_array_digest_v1(selected),
                "search_replay_payload_digest": (
                    compact_search_deferred_replay_payload_digest_v1(handle)
                ),
                "search_replay_payload_digest_deferred": True,
                "loaded_checkpoint_identity": self.loaded_checkpoint_identity,
            },
        )

    def flush_replay_payload(self, replay_payload_handle: str) -> Any:
        raise AssertionError(f"unexpected host replay flush: {replay_payload_handle}")

    def flush_device_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactDeviceSearchReplayPayloadV1:
        import torch

        root_index, env_row, player, policy_env_id, selected, search_id = self.pending.pop(
            replay_payload_handle
        )
        visit_policy = torch.zeros((root_index.size, ACTION_COUNT), dtype=torch.float32)
        visit_policy[
            torch.arange(root_index.size),
            torch.as_tensor(selected, dtype=torch.long),
        ] = 1.0
        root_value = torch.full((root_index.size,), float(search_id), dtype=torch.float32)
        return CompactDeviceSearchReplayPayloadV1(
            replay_payload_handle=str(replay_payload_handle),
            root_index=root_index,
            env_row=env_row,
            player=player,
            policy_env_id=policy_env_id,
            visit_policy=visit_policy,
            root_value=root_value,
            raw_visit_counts=visit_policy.clone(),
            predicted_value=None,
            predicted_policy_logits=None,
            metadata={
                "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                "phase": "replay_critical_device",
                "search_impl": self.search_impl,
                "num_simulations": self.num_simulations,
                "active_root_count": int(root_index.size),
                "replay_payload_origin": (f"{self.search_impl}:{str(replay_payload_handle)}"),
                "device_replay_payload": True,
                "host_search_payload_fallback_allowed": False,
                "loaded_checkpoint_identity": self.loaded_checkpoint_identity,
            },
        )


def _resident_renderer_device(learner_device: str) -> str:
    requested = str(learner_device)
    if requested == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        return "cuda"
    if requested == "mps":
        return "mps"
    return "cpu"


class _PersistentDeviceRenderer:
    backend_name = PERSISTENT_GPU_PROFILE_RENDERER_BACKEND_NAME

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = str(device)

    def render(self, request: Any) -> SourceStateBatchedRenderResult:
        import torch

        out = np.asarray(request.out)
        rows = np.asarray(request.row_indices, dtype=np.int64)
        players = np.asarray(request.controlled_players, dtype=np.int64)
        values = ((rows + 1) * 10 + players + 1).astype(np.uint8)
        out.fill(0)
        out[:, 0, :, :] = values[:, None, None]
        player_count = int(players.max(initial=-1) + 1)
        if player_count <= 0 or int(out.shape[0]) % player_count != 0:
            raise ValueError("renderer request rows must contain complete player groups")
        batch_size = int(out.shape[0]) // player_count
        device_frames = np.zeros_like(out)
        device_frames[:, 0, :, :] = values[:, None, None]
        device_grid = device_frames.reshape(batch_size, player_count, 1, 64, 64)
        return SourceStateBatchedRenderResult(
            frames=out,
            telemetry={
                "render_sec": 0.001,
                "device_render_sec": 0.001,
                "device_to_host_sec": 0.0,
            },
            device_frames=torch.as_tensor(device_grid, device=self.device),
        )


def _non_claims() -> dict[str, bool]:
    return {
        "promotion_claim": False,
        "training_speedup_claim": False,
        "live_run_safety_claim": False,
        "stock_resume_claim": False,
        "rating_or_promotion_quality_claim": False,
    }


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _resolve_artifact_path(raw: str, *, base_dir: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = (base_dir / path).resolve()
    if candidate.exists():
        return candidate
    return path.resolve()


def _relative_ref(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    return value


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
