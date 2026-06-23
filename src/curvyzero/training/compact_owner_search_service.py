"""Owner-side search/replay/learner boundary for compact trainer work.

This module is the production-facing bridge after the mock: the parent can send
small root slot references, while an owner resolves roots, runs search, owns
replay ingest, trains/publishes a model owner ref, and refreshes owner-side
search without returning a model state dict.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
import copy
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
import inspect
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import pickle
import queue
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np

from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
)
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_observation_contract import ResidentObservationBatchV1
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
)
from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactRootActionContextV1
from curvyzero.training.compact_policy_row_bridge import CompactRootBuildRequestV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1
from curvyzero.training.compact_policy_row_bridge import (
    COMPACT_ROOT_BUILD_REQUEST_KIND_RESIDENT_ROOT_VIEW,
)
from curvyzero.training.compact_policy_row_bridge import COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID
from curvyzero.training.compact_policy_row_bridge import COMPACT_ROOT_MECHANICS_OUTCOME_SCHEMA_ID
from curvyzero.training.compact_policy_row_bridge import COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_root_batch_v1_from_request,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_device_replay_index_rows_v1_from_payload,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_replay_index_rows_v1_from_search_result,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_transition_outcome_v1_from_root_build_request,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_root_action_context_v1_from_request,
)
from curvyzero.training.compact_policy_row_bridge import validate_compact_search_result_v1
from curvyzero.training.compact_policy_row_bridge import (
    validate_compact_search_result_identity_v1,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_INDEX_ENTRY_SCHEMA_ID,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID,
)
from curvyzero.training.compact_rollout_slab import CompactOwnerSearchReplayAppendIndexEntryV1
from curvyzero.training.compact_rollout_slab import (
    CompactOwnerSearchReplayAppendDerivedTransitionBatchV1,
)
from curvyzero.training.compact_rollout_slab import (
    _owner_mechanics_step_frame_handle_metadata_v1,
)
from curvyzero.training.compact_search_service import (
    COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID
from curvyzero.training.compact_search_service import CompactDeviceSearchReplayPayloadV1
from curvyzero.training.compact_search_service import CompactSearchActionStepV1
from curvyzero.training.compact_search_service import CompactSearchReplayPayloadV1
from curvyzero.training.compact_search_service import (
    compact_search_action_step_v1_from_result,
)
from curvyzero.training.compact_search_service import compact_search_array_digest_v1
from curvyzero.training.compact_search_service import (
    compact_search_deferred_replay_payload_digest_v1,
)
from curvyzero.training.compact_search_service import (
    compact_search_replay_payload_v1_from_result,
)
from curvyzero.training.compact_search_service import (
    validate_compact_device_search_two_phase_payload_v1,
)
from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


COMPACT_OWNER_SEARCH_SERVICE_SCHEMA_ID = "curvyzero_compact_owner_search_service/v1"
COMPACT_OWNER_SEARCH_REQUEST_SCHEMA_ID = "curvyzero_compact_owner_search_request/v1"
COMPACT_OWNER_SEARCH_RESULT_SCHEMA_ID = "curvyzero_compact_owner_search_result/v1"
COMPACT_OWNER_MAINTENANCE_DRAIN_REQUEST_SCHEMA_ID = (
    "curvyzero_compact_owner_maintenance_drain_request/v1"
)
COMPACT_OWNER_MAINTENANCE_DRAIN_RESULT_SCHEMA_ID = (
    "curvyzero_compact_owner_maintenance_drain_result/v1"
)
COMPACT_OWNER_SEARCH_KIND_IN_PROCESS = "in_process_owner_search_v1"
COMPACT_OWNER_SEARCH_KIND_LOCAL_PROCESS = "local_process_owner_search_v1"
COMPACT_OWNER_SEARCH_KIND_INLINE = "inline_owner_search_v1"
COMPACT_OWNER_SEARCH_KIND_INLINE_BACKGROUND = "inline_background_owner_search_v1"
COMPACT_OWNER_SEARCH_KIND_THREADED = "threaded_owner_search_v1"
COMPACT_DIRECT_ROOT_STORE_SCHEMA_ID = "curvyzero_compact_direct_root_store/v1"
COMPACT_SHARED_MEMORY_ROOT_STORE_SCHEMA_ID = "curvyzero_compact_shared_memory_root_store/v1"
COMPACT_SHARED_MEMORY_ARRAY_SCHEMA_ID = "curvyzero_compact_shared_memory_array/v1"
COMPACT_OWNER_ACTION_RESULT_SLOT_TABLE_SCHEMA_ID = (
    "curvyzero_compact_owner_action_result_slot_table/v1"
)
COMPACT_OWNER_ACTION_RESULT_SLOT_STUB_SCHEMA_ID = (
    "curvyzero_compact_owner_action_result_slot_stub/v1"
)
COMPACT_OWNER_ACTION_DISPATCH_HANDLE_SCHEMA_ID = (
    "curvyzero_compact_owner_action_dispatch_handle/v1"
)
COMPACT_OWNER_ROOT_ACTION_CONTEXT_HANDLE_SCHEMA_ID = (
    "curvyzero_compact_owner_root_action_context_handle/v1"
)
COMPACT_OWNER_ROOT_SEARCH_TRANSACTION_SCHEMA_ID = (
    "curvyzero_compact_owner_root_search_transaction/v1"
)
COMPACT_OWNER_SEARCH_PRIORITY_LOOP_SCHEMA_ID = "curvyzero_compact_owner_search_priority_loop/v1"
COMPACT_OWNER_SEARCH_PRIORITY_LOOP_KIND = "persistent_priority_owner_loop_v1"
COMPACT_OWNER_SEARCH_INLINE_BACKGROUND_LOOP_KIND = "inline_background_maintenance_owner_loop_v1"
COMPACT_OWNER_SEARCH_THREADED_LOOP_KIND = "threaded_priority_owner_loop_v1"
COMPACT_OWNER_SEARCH_RESULT_PAYLOAD_TRANSPORT_KIND = "numpy_ndarray_ipc_v1"
COMPACT_OWNER_SEARCH_ROOT_BATCH_CACHE_MAX = 64
COMPACT_OWNER_SEARCH_DIRECT_TRANSITION_BATCH_REPLAY_TELEMETRY_PREFIX = (
    "compact_owner_search_direct_transition_batch_replay_"
)
COMPACT_OWNER_SEARCH_OWNER_LOCAL_TRANSITION_DERIVATION_TELEMETRY_PREFIX = (
    "compact_owner_search_owner_local_transition_derivation_"
)
COMPACT_OWNER_SEARCH_INNER_REPLAY_TELEMETRY_PREFIX = "compact_owner_search_inner_"
COMPACT_OWNER_SEARCH_REPLAY_APPEND_TELEMETRY_PREFIXES = (
    COMPACT_OWNER_SEARCH_DIRECT_TRANSITION_BATCH_REPLAY_TELEMETRY_PREFIX,
    COMPACT_OWNER_SEARCH_OWNER_LOCAL_TRANSITION_DERIVATION_TELEMETRY_PREFIX,
    COMPACT_OWNER_SEARCH_INNER_REPLAY_TELEMETRY_PREFIX,
)
COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_NONE = "none"
COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD = "in_process_thread_v1"
COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH = (
    "local_process_learner_batch_v1"
)
COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS = (
    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD,
    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH,
)
COMPACT_OWNER_SEARCH_OWNER_REF_DIGEST_DEFERRED_TO_SEARCH_REFRESH_KEY = (
    "model_state_digest_deferred_to_search_refresh"
)
_OWNER_PRIORITY_COMMAND_RUN = "run"
_OWNER_PRIORITY_COMMAND_ACTION = "action"
_OWNER_PRIORITY_COMMAND_DRAIN_MAINTENANCE = "drain_maintenance"
_OWNER_PRIORITY_COMMAND_CLOSE = "close"
_OWNER_SEARCH_TRAIN_TIMING_TELEMETRY_KEYS = (
    "compact_owner_search_owner_train_wall_sec",
    "compact_owner_search_owner_train_sample_sec",
    "compact_owner_search_owner_train_payload_host_clone_sec",
    "compact_owner_search_owner_train_payload_device_move_sec",
    "compact_owner_search_owner_train_learner_update_sec",
    "compact_owner_search_owner_train_model_state_digest_sec",
    "compact_owner_search_owner_train_model_state_dict_sec",
    "compact_owner_search_owner_train_owner_ref_build_sec",
    "compact_owner_search_owner_train_model_state_snapshot_bytes",
    "compact_owner_search_owner_train_model_state_snapshot_write_sec",
    "compact_owner_search_owner_train_accounted_sec",
    "compact_owner_search_owner_train_residual_sec",
)
_OWNER_SEARCH_LEARNER_TIMING_TELEMETRY_KEYS = _OWNER_SEARCH_TRAIN_TIMING_TELEMETRY_KEYS + (
    "compact_muzero_learner_sec",
    "compact_muzero_learner_validation_sec",
    "compact_muzero_learner_zero_grad_sec",
    "compact_muzero_learner_target_transform_sec",
    "compact_muzero_learner_initial_inference_sec",
    "compact_muzero_learner_recurrent_inference_sec",
    "compact_muzero_learner_loss_build_sec",
    "compact_muzero_learner_backward_sec",
    "compact_muzero_learner_grad_clip_sec",
    "compact_muzero_learner_optimizer_step_sec",
    "compact_muzero_learner_loss_readback_sec",
    "compact_muzero_learner_final_sync_sec",
    "compact_muzero_learner_cuda_sync_sec",
    "compact_muzero_learner_accounted_sec",
    "compact_muzero_learner_residual_sec",
)
_OWNER_SEARCH_ACTION_FEEDBACK_KEYS = (
    "compact_owner_search_action_feedback_transition_count",
    "compact_owner_search_action_feedback_action_count",
    "compact_owner_search_action_feedback_mismatch_count",
    "compact_owner_search_expected_joint_action_checksum",
    "compact_owner_search_applied_joint_action_checksum",
    "compact_owner_search_replay_action_checksum",
)


def _empty_owner_action_feedback_v1() -> dict[str, Any]:
    return {
        "compact_owner_search_action_feedback_verified": False,
        "compact_owner_search_action_feedback_transition_count": 0,
        "compact_owner_search_action_feedback_action_count": 0,
        "compact_owner_search_action_feedback_mismatch_count": 0,
        "compact_owner_search_expected_joint_action_checksum": 0,
        "compact_owner_search_applied_joint_action_checksum": 0,
        "compact_owner_search_replay_action_checksum": 0,
    }


def _owner_action_checksum_v1(action: Any) -> int:
    flat = np.asarray(action, dtype=np.int64).reshape(-1)
    if flat.size <= 0:
        return 0
    weights = np.arange(1, flat.size + 1, dtype=np.int64)
    return int(np.dot(flat + 1, weights))


def _owner_dense_joint_action_from_search_result_v1(
    root_batch: CompactRootBatchV1,
    search_result: CompactSearchResultV1,
    *,
    inactive_action: int = 0,
) -> np.ndarray:
    terminal_mask = np.asarray(root_batch.terminal_row_mask, dtype=np.bool_).reshape(-1)
    batch_size = int(terminal_mask.shape[0])
    if batch_size <= 0:
        raise ReplayCompatibilityError("dense owner action requires positive batch size")
    root_count = int(np.asarray(root_batch.active_root_mask).reshape(-1).shape[0])
    if root_count % batch_size != 0:
        raise ReplayCompatibilityError("dense owner action root count is not row-major")
    player_count = int(root_count // batch_size)
    inactive = int(inactive_action)
    if inactive < 0 or inactive >= ACTION_COUNT:
        raise ReplayCompatibilityError("dense owner inactive action is out of range")
    dense = np.full((batch_size, player_count), inactive, dtype=np.int16)
    root_index = np.asarray(search_result.root_index, dtype=np.int64).reshape(-1)
    env_row = np.asarray(search_result.env_row, dtype=np.int64).reshape(-1)
    player = np.asarray(search_result.player, dtype=np.int64).reshape(-1)
    selected = np.asarray(search_result.selected_action, dtype=np.int16).reshape(-1)
    if not (root_index.shape == env_row.shape == player.shape == selected.shape):
        raise ReplayCompatibilityError("dense owner action search sidecar shape mismatch")
    legal_mask = np.asarray(root_batch.legal_mask, dtype=np.bool_)
    if legal_mask.shape != (root_count, ACTION_COUNT):
        raise ReplayCompatibilityError("dense owner action legal mask shape mismatch")
    if selected.size:
        if bool((selected < 0).any()) or bool((selected >= ACTION_COUNT).any()):
            raise ReplayCompatibilityError("dense owner selected action is out of range")
        if bool((env_row < 0).any()) or bool((env_row >= batch_size).any()):
            raise ReplayCompatibilityError("dense owner env row is out of range")
        if bool((player < 0).any()) or bool((player >= player_count).any()):
            raise ReplayCompatibilityError("dense owner player is out of range")
        if bool((root_index < 0).any()) or bool((root_index >= root_count).any()):
            raise ReplayCompatibilityError("dense owner root index is out of range")
        if not bool(legal_mask[root_index, selected.astype(np.int64)].all()):
            raise ReplayCompatibilityError("dense owner selected action is illegal")
        dense[env_row, player] = selected
    return dense


def _owner_dense_joint_action_metadata_v1(action: Any) -> dict[str, Any]:
    dense = np.asarray(action, dtype=np.int16)
    return {
        "dense_joint_action": tuple(
            tuple(int(value) for value in row) for row in dense.reshape(dense.shape)
        ),
        "dense_joint_action_shape": tuple(int(dim) for dim in dense.shape),
        "dense_joint_action_checksum": int(_owner_action_checksum_v1(dense)),
        "dense_joint_action_digest": str(compact_search_array_digest_v1(dense.reshape(-1))),
    }


def _owner_root_action_context_digest_v1(
    root_action_context: CompactRootActionContextV1,
) -> str:
    arrays = (
        np.asarray(root_action_context.active_root_index, dtype=np.int64).reshape(-1),
        np.asarray(root_action_context.env_row, dtype=np.int64).reshape(-1),
        np.asarray(root_action_context.player, dtype=np.int64).reshape(-1),
        np.asarray(root_action_context.policy_env_id, dtype=np.int64).reshape(-1),
        np.asarray(root_action_context.active_legal_mask, dtype=np.bool_)
        .astype(np.int64, copy=False)
        .reshape(-1),
    )
    return str(compact_search_array_digest_v1(np.concatenate(arrays).astype(np.int64)))


def _merge_owner_action_feedback_v1(
    base: Mapping[str, Any] | None,
    update: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = _empty_owner_action_feedback_v1()
    for source in (base, update):
        if not isinstance(source, Mapping):
            continue
        for key in _OWNER_SEARCH_ACTION_FEEDBACK_KEYS:
            merged[key] = int(merged[key]) + int(source.get(key) or 0)
    merged["compact_owner_search_action_feedback_verified"] = (
        int(merged["compact_owner_search_action_feedback_action_count"]) > 0
        and int(merged["compact_owner_search_action_feedback_mismatch_count"]) == 0
    )
    return merged


_SHARED_ROOT_ARRAY_FIELDS = (
    "observation",
    "legal_mask",
    "active_root_mask",
    "to_play",
    "env_row",
    "player",
    "policy_env_id",
    "target_reward",
    "done_root",
    "final_observation_row_mask",
    "terminal_row_mask",
    "autoreset_row_mask",
)

_PROCESS_OWNER_SEARCH_SERVICE: CompactOwnerSearchServiceV1 | None = None
_PROCESS_OWNER_SEARCH_ASYNC_LEARNER: Any | None = None
_PROCESS_OWNER_SEARCH_ASYNC_LEARNER_INITIALIZED_COUNT = 0
_PROCESS_OWNER_SEARCH_ASYNC_LEARNER_COMPLETED_COUNT = 0


@dataclass(frozen=True, slots=True)
class CompactSharedMemoryArraySpecV1:
    """One fixed shared-memory array used by the root-slot provider."""

    schema_id: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True, slots=True)
class CompactSharedMemoryRootStoreSpecV1:
    """Picklable handle for owner-side root-slot resolution."""

    schema_id: str
    capacity: int
    arrays: dict[str, CompactSharedMemoryArraySpecV1]
    metadata: dict[str, Any]


class CompactDirectRootStoreV1:
    """Inline root handoff that keeps the current root batch in-process."""

    def __init__(
        self,
        *,
        capacity: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        require_resident_root_view: bool = False,
    ) -> None:
        self.capacity = None if capacity is None else int(capacity)
        if self.capacity is not None and self.capacity <= 0:
            raise ValueError("direct root-store capacity must be positive")
        self.extra_metadata = dict(metadata or {})
        self.require_resident_root_view = bool(require_resident_root_view)
        self.publish_count = 0
        self.resolve_count = 0
        self.root_build_request_publish_count = 0
        self.root_build_request_resolve_count = 0
        self.owner_root_batch_build_count = 0
        self.owner_root_batch_build_sec = 0.0
        self.last_root_slot_ids: tuple[int, ...] = ()
        self._last_root_batch: CompactRootBatchV1 | None = None
        self._last_root_build_request: CompactRootBuildRequestV1 | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        root_count = len(self.last_root_slot_ids)
        return {
            "compact_direct_root_store_schema_id": COMPACT_DIRECT_ROOT_STORE_SCHEMA_ID,
            "compact_direct_root_store": True,
            "compact_direct_root_store_capacity": (
                0 if self.capacity is None else int(self.capacity)
            ),
            "compact_direct_root_store_publish_count": int(self.publish_count),
            "compact_direct_root_store_resolve_count": int(self.resolve_count),
            "compact_owner_search_direct_root_build_request_publish_count": int(
                self.root_build_request_publish_count
            ),
            "compact_owner_search_direct_root_build_request_resolve_count": int(
                self.root_build_request_resolve_count
            ),
            "compact_owner_search_direct_root_owner_build_count": int(
                self.owner_root_batch_build_count
            ),
            "compact_owner_search_direct_root_owner_build_sec": float(
                self.owner_root_batch_build_sec
            ),
            "compact_direct_root_store_last_root_slot_count": int(root_count),
            "compact_owner_search_direct_root_handoff": True,
            "compact_owner_search_direct_root_rebuild_avoided": True,
            "compact_owner_search_direct_root_observation_bytes_sent": 0,
            "compact_owner_search_resident_root_view_required": bool(
                self.require_resident_root_view
            ),
            **self.extra_metadata,
        }

    def publish_root_batch(
        self,
        root_batch: CompactRootBatchV1,
        *,
        slot_offset: int = 0,
    ) -> tuple[int, ...]:
        offset = int(slot_offset)
        if offset < 0:
            raise ValueError("slot_offset must be nonnegative")
        if not isinstance(root_batch, CompactRootBatchV1):
            raise TypeError("direct root store requires CompactRootBatchV1")
        root_count = int(np.asarray(root_batch.active_root_mask).shape[0])
        if root_count <= 0:
            raise ValueError("direct root batch must have positive root count")
        if self.capacity is not None and offset + root_count > int(self.capacity):
            raise ValueError("root batch does not fit in direct root-store slots")
        self._last_root_batch = root_batch
        self._last_root_build_request = None
        self.publish_count += 1
        self.last_root_slot_ids = tuple(range(offset, offset + root_count))
        return self.last_root_slot_ids

    def publish_root_build_request(
        self,
        root_build_request: CompactRootBuildRequestV1,
        *,
        slot_offset: int = 0,
    ) -> tuple[int, ...]:
        offset = int(slot_offset)
        if offset < 0:
            raise ValueError("slot_offset must be nonnegative")
        if not isinstance(root_build_request, CompactRootBuildRequestV1):
            raise TypeError("direct root store requires CompactRootBuildRequestV1")
        root_count = int(root_build_request.root_count)
        if root_count <= 0:
            raise ValueError("direct root build request must have positive root count")
        if self.capacity is not None and offset + root_count > int(self.capacity):
            raise ValueError("root build request does not fit in direct root-store slots")
        self._last_root_batch = None
        self._last_root_build_request = root_build_request
        self.root_build_request_publish_count += 1
        self.last_root_slot_ids = tuple(range(offset, offset + root_count))
        return self.last_root_slot_ids

    def resolve_root_batch(
        self,
        *,
        root_slot_ids: tuple[int, ...],
        request: CompactOwnerSearchRequestV1,
    ) -> CompactRootBatchV1:
        del request
        slots = tuple(int(slot_id) for slot_id in tuple(root_slot_ids))
        if not slots:
            raise ValueError("root_slot_ids must be non-empty")
        root_batch = self._last_root_batch
        root_build_request = self._last_root_build_request
        owner_build_metadata: dict[str, Any] = {}
        if root_batch is None and root_build_request is not None:
            build_started = time.perf_counter()
            root_batch = build_compact_root_batch_v1_from_request(
                root_build_request,
                metadata={
                    "compact_owner_search_direct_root_build_request_handoff": True,
                    "compact_owner_search_direct_root_owner_build_used": True,
                    "compact_owner_search_direct_root_parent_build_avoided": True,
                    "compact_owner_search_direct_root_parent_build_call_count": 0,
                    "compact_owner_search_direct_root_parent_build_sec": 0.0,
                },
            )
            owner_build_sec = _elapsed(build_started)
            self.root_build_request_resolve_count += 1
            self.owner_root_batch_build_count += 1
            self.owner_root_batch_build_sec += float(owner_build_sec)
            owner_build_metadata = {
                "compact_owner_search_direct_root_build_request_handoff": True,
                "compact_owner_search_direct_root_build_request_schema_id": str(
                    root_build_request.schema_id
                ),
                "compact_owner_search_direct_root_build_request_kind": str(
                    root_build_request.request_kind
                ),
                "compact_owner_search_direct_root_build_request_publish_count": int(
                    self.root_build_request_publish_count
                ),
                "compact_owner_search_direct_root_build_request_resolve_count": int(
                    self.root_build_request_resolve_count
                ),
                "compact_owner_search_direct_root_build_request_root_count": int(
                    root_build_request.root_count
                ),
                "compact_owner_search_direct_root_build_request_active_root_count": int(
                    np.count_nonzero(root_build_request.active_root_mask)
                ),
                "compact_owner_search_direct_root_build_request_observation_included": (
                    root_build_request.observation is not None
                ),
                "compact_owner_search_direct_root_build_request_observation_bytes_sent": (
                    0
                    if root_build_request.observation is None
                    else int(np.asarray(root_build_request.observation).nbytes)
                ),
                "compact_owner_search_direct_root_build_request_resident_handle_present": (
                    root_build_request.resident_observation is not None
                ),
                "compact_owner_search_direct_root_parent_build_avoided": True,
                "compact_owner_search_direct_root_parent_build_call_count": 0,
                "compact_owner_search_direct_root_parent_build_sec": 0.0,
                "compact_owner_search_direct_root_owner_build_used": True,
                "compact_owner_search_direct_root_owner_build_count": int(
                    self.owner_root_batch_build_count
                ),
                "compact_owner_search_direct_root_owner_build_sec": float(
                    self.owner_root_batch_build_sec
                ),
                "compact_owner_search_parent_compact_root_batch_objects_sent": 0,
                "compact_owner_search_root_build_request_host_observation_bytes_sent": 0,
            }
        if root_batch is None:
            raise RuntimeError("direct root store has no published root batch")
        if slots != self.last_root_slot_ids:
            raise RuntimeError("direct root store can only resolve the latest root batch")
        resident_root_view_metadata: dict[str, Any] = {}
        if self.require_resident_root_view:
            resident_root_view_metadata = _direct_resident_root_view_metadata_v1(root_batch)
        self.resolve_count += 1
        metadata = {
            **dict(root_batch.metadata),
            **self.metadata,
            **owner_build_metadata,
            **resident_root_view_metadata,
            "compact_direct_root_store_resolved": True,
            "compact_direct_root_store_root_slot_count": int(len(slots)),
            "compact_direct_root_store_root_slot_checksum": int(sum(slots)),
            "compact_owner_search_direct_root_resolved": True,
        }
        return replace(root_batch, metadata=metadata)

    def close(self) -> None:
        self._last_root_batch = None
        self._last_root_build_request = None

    def unlink(self) -> None:
        return None


def _direct_resident_root_view_metadata_v1(
    root_batch: CompactRootBatchV1,
) -> dict[str, Any]:
    if str(root_batch.observation_source) != COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1:
        raise ReplayCompatibilityError(
            "direct resident root-view proof requires resident_device_v1 roots"
        )
    resident = root_batch.resident_observation
    if resident is None:
        raise ReplayCompatibilityError(
            "direct resident root-view proof requires resident_observation"
        )
    if bool(resident.host_fallback_allowed):
        raise ReplayCompatibilityError("direct resident root-view proof forbids host fallback")
    if not bool(resident.row_major_order):
        raise ReplayCompatibilityError(
            "direct resident root-view proof requires row-major resident roots"
        )
    root_device_observation = resident.root_device_observation
    if root_device_observation is None:
        raise ReplayCompatibilityError(
            "direct resident root-view proof requires root_device_observation"
        )
    root_shape = tuple(int(dim) for dim in tuple(getattr(root_device_observation, "shape", ())))
    expected_root_count = int(np.asarray(root_batch.active_root_mask).reshape(-1).shape[0])
    expected_shape = (
        expected_root_count,
        *tuple(int(dim) for dim in tuple(resident.stack_shape)),
    )
    if root_shape != expected_shape:
        raise ReplayCompatibilityError(
            "direct resident root-view proof root_device_observation shape mismatch: "
            f"expected {expected_shape}, got {root_shape}"
        )
    return {
        "compact_owner_search_resident_root_view_proved": True,
        "compact_owner_search_resident_root_view_kind": ("direct_root_batch_resident_handle_v1"),
        "compact_owner_search_resident_root_view_generation_id": int(resident.generation_id),
        "compact_owner_search_resident_root_view_fresh_for_step_index": int(
            resident.fresh_for_step_index
        ),
        "compact_owner_search_resident_root_view_device": str(resident.device),
        "compact_owner_search_resident_root_view_source_backend": str(resident.source_backend),
        "compact_owner_search_resident_root_view_root_shape": list(root_shape),
        "compact_owner_search_resident_root_view_stack_shape": [
            int(dim) for dim in tuple(resident.stack_shape)
        ],
        "compact_owner_search_resident_root_view_h2d_bytes": 0.0,
        "compact_owner_search_resident_root_view_d2h_bytes": 0.0,
        "compact_owner_search_resident_root_view_host_fallback_allowed": False,
        "compact_owner_search_resident_root_view_row_major_order": True,
    }


def _owner_root_search_transaction_request_from_step_frame_slot_v1(
    compact_batch: Any,
    *,
    search_lane: str,
    metadata: Mapping[str, Any],
    copy_observation: bool,
    resident_host_observation_stub: bool,
) -> CompactRootBuildRequestV1:
    frame_metadata = _owner_mechanics_step_frame_handle_metadata_v1(compact_batch)
    resident_observation = getattr(compact_batch, "resident_observation", None)
    if resident_observation is None:
        raise ReplayCompatibilityError(
            "owner root-search transaction requires resident_observation"
        )
    batch_size = int(getattr(resident_observation, "batch_size"))
    player_count = int(getattr(resident_observation, "player_count"))
    root_count = int(batch_size * player_count)
    stack_shape = tuple(int(dim) for dim in getattr(resident_observation, "stack_shape"))
    action_mask = np.asarray(compact_batch.action_mask, dtype=np.bool_)
    if action_mask.shape != (batch_size, player_count, ACTION_COUNT):
        raise ReplayCompatibilityError(
            "owner root-search transaction action_mask shape mismatch"
        )
    reward = np.asarray(compact_batch.reward, dtype=np.float32)
    done = np.asarray(compact_batch.done, dtype=np.bool_)
    policy_env_id = np.asarray(compact_batch.policy_env_id, dtype=np.int32)
    policy_env_row = np.asarray(compact_batch.policy_env_row, dtype=np.int32)
    policy_player = np.asarray(compact_batch.policy_player, dtype=np.int32)
    target_reward = reward.reshape(root_count, 1)
    done_root = np.repeat(done, player_count).astype(np.bool_, copy=False)
    to_play = np.full((root_count,), -1, dtype=np.int64)
    active_root_mask = np.logical_and(
        ~done_root,
        action_mask.reshape(root_count, ACTION_COUNT).any(axis=1),
    )
    request_metadata = {
        str(key): value
        for key, value in dict(metadata).items()
        if value is None or isinstance(value, (bool, int, float, str))
    }
    request_metadata.update(
        {
            **frame_metadata,
            "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
            "schema_id": COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID,
            "root_build_request_schema_id": COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID,
            "root_build_request_kind": COMPACT_ROOT_BUILD_REQUEST_KIND_RESIDENT_ROOT_VIEW,
            "search_lane": str(search_lane),
            "observation_source": COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
            "resident_host_observation_stub_requested": bool(
                resident_host_observation_stub
            ),
            "batch_size": int(batch_size),
            "player_count": int(player_count),
            "root_count": int(root_count),
            "stack_shape": list(stack_shape),
            "mechanics_outcome_schema_id": COMPACT_ROOT_MECHANICS_OUTCOME_SCHEMA_ID,
            "mechanics_outcome_sidecars_present": True,
            "compact_owner_step_frame_root_build_request_used": True,
            "compact_owner_step_frame_root_build_request_from_batch_helper_used": False,
            "compact_owner_step_frame_root_request_sidecar_array_bytes": 0,
            "compact_owner_step_frame_root_request_sidecar_field_count": 0,
            "compact_owner_root_search_transaction_used": True,
            "compact_owner_root_search_transaction_schema_id": (
                COMPACT_OWNER_ROOT_SEARCH_TRANSACTION_SCHEMA_ID
            ),
            "compact_owner_root_search_transaction_parent_root_request_build_count": 0,
            "compact_owner_root_search_transaction_parent_root_request_stored": False,
            "compact_owner_root_search_transaction_parent_compact_batch_stored": False,
            "compact_owner_root_search_transaction_parent_rebuild_count": 0,
            "compact_owner_root_search_transaction_frame_generation_verified": True,
            "compact_owner_root_search_transaction_frame_digest_verified": True,
        }
    )
    return CompactRootBuildRequestV1(
        schema_id=COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID,
        request_kind=COMPACT_ROOT_BUILD_REQUEST_KIND_RESIDENT_ROOT_VIEW,
        search_lane=str(search_lane),
        metadata=request_metadata,
        copy_observation=bool(copy_observation),
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_host_observation_stub=bool(resident_host_observation_stub),
        batch_size=int(batch_size),
        player_count=int(player_count),
        root_count=int(root_count),
        stack_shape=stack_shape,
        observation=None,
        action_mask=action_mask,
        reward=reward,
        done=done,
        policy_env_id=policy_env_id,
        policy_env_row=policy_env_row,
        policy_player=policy_player,
        target_reward=target_reward,
        done_root=done_root,
        to_play=to_play,
        active_root_mask=active_root_mask,
        final_observation=None,
        final_observation_row_mask=np.asarray(
            compact_batch.final_observation_row_mask,
            dtype=np.bool_,
        ),
        terminal_row_mask=np.asarray(compact_batch.terminal_row_mask, dtype=np.bool_),
        autoreset_row_mask=np.asarray(compact_batch.autoreset_row_mask, dtype=np.bool_),
        joint_action=np.asarray(compact_batch.joint_action, dtype=np.int16),
        resident_observation=resident_observation,
        terminated=np.asarray(compact_batch.terminated, dtype=np.bool_),
        truncated=np.asarray(compact_batch.truncated, dtype=np.bool_),
        final_reward_map=np.asarray(compact_batch.final_reward_map, dtype=np.float32),
    )


class CompactOwnerActionResultSlotTableV1:
    """Same-process action-result table that returns a tiny slot stub to parent."""

    def __init__(self, *, capacity: int = 4) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("action-result slot table capacity must be positive")
        self.acquire_count = 0
        self.write_count = 0
        self.read_count = 0
        self.last_slot_id = -1
        self.last_wire_result_bytes = 0
        self.last_full_result_bytes = 0
        self._slots: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    @property
    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return {
                "compact_owner_search_fixed_action_result_buffer_requested": True,
                "compact_owner_search_fixed_action_result_buffer_slot_count": int(self.capacity),
                "compact_owner_search_fixed_action_result_buffer_acquire_count": int(
                    self.acquire_count
                ),
                "compact_owner_search_fixed_action_result_buffer_write_count": int(
                    self.write_count
                ),
                "compact_owner_search_fixed_action_result_buffer_read_count": int(self.read_count),
                "compact_owner_search_fixed_action_result_buffer_last_slot_id": int(
                    self.last_slot_id
                ),
                "compact_owner_search_fixed_action_result_buffer_wire_result_bytes": int(
                    self.last_wire_result_bytes
                ),
                "compact_owner_search_fixed_action_result_buffer_full_result_bytes": int(
                    self.last_full_result_bytes
                ),
                "compact_owner_search_fixed_action_result_buffer_pending_slot_count": int(
                    len(self._slots)
                ),
            }

    def acquire(self) -> int:
        with self._lock:
            slot_id = int(self.acquire_count % self.capacity)
            if slot_id in self._slots:
                raise RuntimeError("action-result slot was reused before read")
            self.acquire_count += 1
            self.last_slot_id = slot_id
            return slot_id

    def write(self, slot_id: int, payload: Mapping[str, Any]) -> dict[str, Any]:
        slot = int(slot_id)
        full_payload = dict(payload)
        full_result_bytes = int(
            full_payload.get("result_bytes") or _pickle_size_bytes(full_payload)
        )
        with self._lock:
            if slot < 0 or slot >= self.capacity:
                raise RuntimeError("action-result slot id out of range")
            if slot in self._slots:
                raise RuntimeError("action-result slot already contains unread payload")
            self._slots[slot] = full_payload
            self.write_count += 1
            self.last_slot_id = slot
            self.last_full_result_bytes = int(full_result_bytes)
            stub = {
                "schema_id": COMPACT_OWNER_ACTION_RESULT_SLOT_STUB_SCHEMA_ID,
                "compact_owner_search_fixed_action_result_buffer_used": True,
                "compact_owner_search_fixed_action_result_buffer_slot_id": int(slot),
                "compact_owner_search_fixed_action_result_buffer_full_result_bytes": int(
                    full_result_bytes
                ),
                "compact_owner_search_fixed_action_result_buffer_slot_count": int(self.capacity),
                "compact_owner_search_fixed_action_result_buffer_write_count": int(
                    self.write_count
                ),
            }
            wire_result_bytes = _pickle_size_bytes(stub)
            stub["result_bytes"] = int(wire_result_bytes)
            wire_result_bytes = _pickle_size_bytes(stub)
            stub["result_bytes"] = int(wire_result_bytes)
            self.last_wire_result_bytes = int(wire_result_bytes)
            return dict(stub)

    def read(self, slot_id: int) -> dict[str, Any]:
        slot = int(slot_id)
        with self._lock:
            payload = self._slots.pop(slot, None)
            if payload is None:
                raise RuntimeError("action-result slot payload is missing")
            self.read_count += 1
            self.last_slot_id = slot
            return dict(payload)


@dataclass(frozen=True, slots=True)
class CompactOwnerSearchRequestV1:
    """Parent-to-owner request.

    Root observations are intentionally absent. The owner must resolve
    ``root_slot_ids`` into a ``CompactRootBatchV1`` inside the owner boundary.
    Owner-search replay append entries must carry small index/transition facts;
    previous/current compact observation batches do not belong in this request.
    """

    request_id: int
    actor_step: int
    root_slot_ids: tuple[int, ...]
    replay_append_entries: tuple[Any, ...] = ()
    sample_batch_size: int = 0
    train_steps: int = 0
    policy_version_ref: str = ""
    model_version_ref: str = ""
    policy_source: str = ""
    refresh_model: bool = True
    action_result_slot_id: int = -1


@dataclass(frozen=True, slots=True)
class CompactOwnerActionDispatchHandleV1:
    """Parent-visible handle for a submitted owner action transaction."""

    schema_id: str
    dispatch_id: int
    request_id: int
    actor_step: int
    root_slot_count: int
    action_result_slot_id: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactOwnerRootActionContextHandleV1:
    """Small handle for a proxy-owned action-critical root context."""

    schema_id: str
    context_id: int
    transaction_id: int
    dispatch_id: int
    batch_size: int
    player_count: int
    root_count: int
    active_root_count: int
    context_digest: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactOwnerRootSearchTransactionDispatchV1:
    """Proxy-owned root/search transaction opened from a mechanics frame slot."""

    schema_id: str
    transaction_id: int
    action_dispatch_handle: CompactOwnerActionDispatchHandleV1
    root_action_context_handle: CompactOwnerRootActionContextHandleV1
    commit_timing: dict[str, float]
    metadata: dict[str, Any]
    root_action_context: CompactRootActionContextV1 | None = None


@dataclass(frozen=True, slots=True)
class _PendingOwnerActionDispatchV1:
    handle: CompactOwnerActionDispatchHandleV1
    root_action_context_handle: CompactOwnerRootActionContextHandleV1
    request: CompactOwnerSearchRequestV1
    worker_handle: Any
    started: float
    publish_sec: float
    submit_sec: float
    had_inflight_maintenance: bool
    replay_append_entries: tuple[Any, ...]
    replay_append_logical_entry_count: int
    replay_append_transport_entry_count: int
    replay_append_transition_batch_count: int
    replay_append_transition_batch_entry_count: int
    replay_append_transition_legacy_entry_count: int
    train_steps: int
    root_search_transaction_id: int = 0


@dataclass(frozen=True, slots=True)
class _CompactOwnerProxyActionFrameV1:
    record_index: int
    replay_payload_handle: str
    selected_action_digest: str
    search_replay_payload_digest: str
    env_row: np.ndarray
    player: np.ndarray
    selected_action: np.ndarray


@dataclass(frozen=True, slots=True)
class _CompactOwnerProxyDerivedTransitionFactsV1:
    record_index: int
    next_record_index: int
    replay_payload_handle: str
    selected_action_digest: str
    search_replay_payload_digest: str
    applied_action_count: int
    applied_action_checksum: int
    policy_source: str


@dataclass(frozen=True, slots=True)
class CompactOwnerMaintenanceDrainRequestV1:
    """Parent request to drain replay/train/search-refresh work already staged in the owner."""

    drain_id: int
    max_items: int = 0
    fail_if_empty: bool = False


@dataclass(frozen=True, slots=True)
class _CompactOwnerMaintenanceWorkV1:
    request: CompactOwnerSearchRequestV1
    root_batch: CompactRootBatchV1
    search_result: CompactSearchResultV1
    root_batch_cache: dict[int, CompactRootBatchV1]


@dataclass(frozen=True, slots=True)
class _CompactOwnerPendingLearnerJobV1:
    request: CompactOwnerSearchRequestV1
    handle: Any
    submitted_at: float


@dataclass(frozen=True, slots=True)
class _CompactOwnerSearchProcessLearnerRequestV1:
    """Host-only learner payload submitted to a child learner process."""

    request: CompactOwnerSearchRequestV1
    learner_payload: Any
    payload_prepare_sec: float
    request_bytes: int
    request_cuda_tensor_count: int


@dataclass(frozen=True, slots=True)
class _CompactOwnerSearchProcessLearnerHandleV1:
    """Process future plus parent-side transport proof."""

    future: Future[Any]
    request: CompactOwnerSearchRequestV1
    payload_prepare_sec: float
    request_bytes: int
    request_cuda_tensor_count: int


@dataclass(frozen=True, slots=True)
class _CompactOwnerCachedSearchV1:
    record_index: int
    root_batch: CompactRootBatchV1
    search_result: CompactSearchResultV1
    action_step: CompactSearchActionStepV1 | None = None
    inner_replay_payload_handle: str = ""
    inner_device_replay_payload: Any | None = None


@dataclass(frozen=True, slots=True)
class CompactOwnerSearchResultV1:
    """Small owner-to-parent result from one owner-side search request."""

    schema_id: str
    owner_kind: str
    request_id: int
    actor_step: int
    owner_pid: int
    root_slot_count: int
    active_root_count: int
    selected_action: tuple[int, ...]
    search_impl: str
    num_simulations: int
    replay_append_entry_count: int
    replay_append_count: int
    learner_update_count: int
    model_owner_ref_returned: bool
    model_owner_ref_digest: str
    model_state_return_count: int
    model_state_bytes: int
    model_state_snapshot_return_count: int
    root_observation_bytes_sent: int
    request_cuda_tensor_count: int
    result_cuda_tensor_count: int
    request_bytes: int
    result_bytes: int
    worker_owns_search_state: bool
    worker_owns_replay_state: bool
    worker_owns_model_state: bool
    search_consumed_learner_update: bool
    search_refresh_update_count: int
    search_worker_state: dict[str, Any] | None
    search_result_payload: dict[str, Any]
    search_result_payload_bytes: int
    search_selected_action_bytes: int
    search_visit_policy_bytes: int
    search_root_value_bytes: int
    search_optional_array_bytes: int
    owner_sample_telemetry: dict[str, Any]
    owner_learner_telemetry: dict[str, Any]
    owner_action_feedback: dict[str, Any]
    timing: dict[str, float]
    search_result_metadata: dict[str, Any] | None = None
    owner_maintenance_deferred: bool = False
    owner_maintenance_staged_work_count: int = 0
    owner_maintenance_pending_work_count: int = 0
    replay_payload_handle: str = ""
    inner_two_phase_action_step: bool = False
    inner_device_replay_payload_deferred: bool = False
    replay_append_transport_entry_count: int = 0
    replay_append_transition_batch_count: int = 0
    replay_append_transition_batch_entry_count: int = 0
    dense_joint_action: tuple[tuple[int, ...], ...] = ()
    dense_joint_action_shape: tuple[int, ...] = ()
    dense_joint_action_checksum: int = 0
    dense_joint_action_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompactOwnerMaintenanceDrainResultV1:
    """Small owner-to-parent result after draining deferred maintenance."""

    schema_id: str
    owner_kind: str
    drain_id: int
    owner_pid: int
    drained_count: int
    drained_work_item_count: int
    drained_replay_append_entry_count: int
    drained_replay_append_count: int
    pending_count: int
    replay_append_entry_count: int
    replay_append_count: int
    learner_update_count: int
    model_owner_ref_returned: bool
    model_owner_ref_digest: str
    search_refresh_update_count: int
    search_consumed_learner_update: bool
    owner_sample_telemetry: dict[str, Any]
    owner_learner_telemetry: dict[str, Any]
    owner_action_feedback: dict[str, Any]
    timing: dict[str, float]
    owner_async_learner_telemetry: dict[str, Any] | None = None
    drained_replay_append_transport_entry_count: int = 0
    replay_append_transport_entry_count: int = 0
    drained_replay_append_transition_batch_count: int = 0
    replay_append_transition_batch_count: int = 0
    drained_replay_append_transition_batch_entry_count: int = 0
    replay_append_transition_batch_entry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _CompactOwnerPriorityCommandV1:
    command_id: int
    command_kind: str
    payload: Any = None


@dataclass(frozen=True, slots=True)
class _CompactOwnerPriorityResultV1:
    command_id: int
    ok: bool
    payload: dict[str, Any] | None = None
    error_message: str = ""
    traceback_text: str = ""


class CompactSharedMemoryRootStoreV1:
    """Fixed root-slot store shared between actor/game and owner-search process."""

    def __init__(
        self,
        *,
        spec: CompactSharedMemoryRootStoreSpecV1,
        owner: bool = False,
    ) -> None:
        self.spec = spec
        self.owner = bool(owner)
        self._shared_memory: dict[str, shared_memory.SharedMemory] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self.publish_count = 0
        self.last_root_slot_ids: tuple[int, ...] = ()
        for field, array_spec in dict(spec.arrays).items():
            shm = shared_memory.SharedMemory(name=array_spec.name)
            array = np.ndarray(
                tuple(array_spec.shape),
                dtype=np.dtype(array_spec.dtype),
                buffer=shm.buf,
            )
            self._shared_memory[str(field)] = shm
            self._arrays[str(field)] = array

    @classmethod
    def create(
        cls,
        root_batch: CompactRootBatchV1,
        *,
        capacity: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> CompactSharedMemoryRootStoreV1:
        root_count = int(np.asarray(root_batch.active_root_mask).shape[0])
        capacity_int = root_count if capacity is None else int(capacity)
        if capacity_int < root_count:
            raise ValueError("shared root-store capacity must fit the root batch")
        arrays: dict[str, CompactSharedMemoryArraySpecV1] = {}
        shared: dict[str, shared_memory.SharedMemory] = {}
        try:
            for field in _SHARED_ROOT_ARRAY_FIELDS:
                source = _shared_root_store_array(root_batch, field)
                shape = (capacity_int, *tuple(source.shape[1:]))
                dtype = np.dtype(source.dtype)
                nbytes = int(dtype.itemsize * int(np.prod(shape)))
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                shared[field] = shm
                arrays[field] = CompactSharedMemoryArraySpecV1(
                    schema_id=COMPACT_SHARED_MEMORY_ARRAY_SCHEMA_ID,
                    name=str(shm.name),
                    shape=tuple(int(dim) for dim in shape),
                    dtype=str(dtype),
                    nbytes=nbytes,
                )
            spec = CompactSharedMemoryRootStoreSpecV1(
                schema_id=COMPACT_SHARED_MEMORY_ROOT_STORE_SCHEMA_ID,
                capacity=capacity_int,
                arrays=arrays,
                metadata={
                    "compact_shared_memory_root_store": True,
                    "compact_shared_memory_root_store_capacity": capacity_int,
                    **dict(getattr(root_batch, "metadata", {}) or {}),
                    **dict(metadata or {}),
                },
            )
            store = cls(spec=spec, owner=True)
            for shm in shared.values():
                shm.close()
            store.publish_root_batch(root_batch)
            return store
        except Exception:
            for shm in shared.values():
                try:
                    shm.close()
                except FileNotFoundError:
                    pass
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            raise

    @classmethod
    def attach(
        cls,
        *,
        spec: CompactSharedMemoryRootStoreSpecV1,
    ) -> CompactSharedMemoryRootStoreV1:
        return cls(spec=spec, owner=False)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "compact_shared_memory_root_store_schema_id": (
                COMPACT_SHARED_MEMORY_ROOT_STORE_SCHEMA_ID
            ),
            "compact_shared_memory_root_store_capacity": int(self.spec.capacity),
            "compact_shared_memory_root_store_publish_count": int(self.publish_count),
            "compact_shared_memory_root_store_last_root_slot_count": len(self.last_root_slot_ids),
            "compact_shared_memory_root_store_array_fields": tuple(sorted(self._arrays.keys())),
            "compact_shared_memory_root_store_total_nbytes": int(
                sum(int(spec.nbytes) for spec in self.spec.arrays.values())
            ),
        }

    def publish_root_batch(
        self,
        root_batch: CompactRootBatchV1,
        *,
        slot_offset: int = 0,
    ) -> tuple[int, ...]:
        offset = int(slot_offset)
        if offset < 0:
            raise ValueError("slot_offset must be nonnegative")
        root_count = int(np.asarray(root_batch.active_root_mask).shape[0])
        if offset + root_count > int(self.spec.capacity):
            raise ValueError("root batch does not fit in shared root-store slots")
        for field in _SHARED_ROOT_ARRAY_FIELDS:
            source = _shared_root_store_array(root_batch, field)
            destination = self._arrays[field]
            expected_shape = destination[offset : offset + root_count].shape
            if tuple(source.shape) != tuple(expected_shape):
                raise ValueError(f"shared root-store field {field} shape mismatch")
            destination[offset : offset + root_count] = source
        self.publish_count += 1
        self.last_root_slot_ids = tuple(range(offset, offset + root_count))
        return self.last_root_slot_ids

    def resolve_root_batch(
        self,
        *,
        root_slot_ids: tuple[int, ...],
        request: CompactOwnerSearchRequestV1,
        copy_observation: bool = True,
    ) -> CompactRootBatchV1:
        del request
        slots = np.asarray(tuple(int(slot_id) for slot_id in root_slot_ids), dtype=np.int64)
        if slots.size == 0:
            raise ValueError("root_slot_ids must be non-empty")
        if int(slots.min()) < 0 or int(slots.max()) >= int(self.spec.capacity):
            raise ValueError("root_slot_ids outside shared root-store capacity")
        arrays: dict[str, np.ndarray] = {}
        for field in _SHARED_ROOT_ARRAY_FIELDS:
            selected = _select_shared_root_slots(self._arrays[field], slots)
            if field == "observation" and not bool(copy_observation):
                arrays[field] = selected
            else:
                arrays[field] = selected.copy()
        return CompactRootBatchV1(
            observation=arrays["observation"],
            legal_mask=arrays["legal_mask"],
            active_root_mask=arrays["active_root_mask"],
            to_play=arrays["to_play"],
            env_row=arrays["env_row"],
            player=arrays["player"],
            policy_env_id=arrays["policy_env_id"],
            target_reward=arrays["target_reward"],
            done_root=arrays["done_root"],
            final_observation=None,
            final_observation_row_mask=arrays["final_observation_row_mask"],
            terminal_row_mask=arrays["terminal_row_mask"],
            autoreset_row_mask=arrays["autoreset_row_mask"],
            metadata={
                **dict(self.spec.metadata),
                "compact_shared_memory_root_store_resolved": True,
                "compact_shared_memory_root_store_observation_copied": bool(copy_observation),
                "compact_shared_memory_root_store_root_slot_count": int(slots.size),
                "compact_shared_memory_root_store_root_slot_checksum": int(slots.sum()),
            },
        )

    def close(self) -> None:
        for shm in self._shared_memory.values():
            try:
                shm.close()
            except FileNotFoundError:
                pass
        self._shared_memory.clear()
        self._arrays.clear()

    def unlink(self) -> None:
        if not self.owner:
            return
        for array_spec in self.spec.arrays.values():
            try:
                shm = shared_memory.SharedMemory(name=array_spec.name)
            except FileNotFoundError:
                continue
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            finally:
                try:
                    shm.close()
                except FileNotFoundError:
                    pass


def build_compact_shared_memory_root_provider_v1(
    *,
    spec: CompactSharedMemoryRootStoreSpecV1,
) -> CompactSharedMemoryRootStoreV1:
    """Factory for process-worker owner-side root providers."""

    return CompactSharedMemoryRootStoreV1.attach(spec=spec)


class CompactResidentSharedMemoryRootProviderV1:
    """Resolve shared host root slots into owner-side resident Torch roots."""

    def __init__(
        self,
        *,
        spec: CompactSharedMemoryRootStoreSpecV1,
        device: str,
        source_backend: str = "owner_shared_memory_root_store_v1",
    ) -> None:
        self.root_store = CompactSharedMemoryRootStoreV1.attach(spec=spec)
        self.device = str(device)
        self.source_backend = str(source_backend)
        self.resolve_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.root_store.metadata,
            "compact_resident_shared_memory_root_provider": True,
            "compact_resident_shared_memory_root_provider_device": self.device,
            "compact_resident_shared_memory_root_provider_resolve_count": int(self.resolve_count),
        }

    def resolve_root_batch(
        self,
        *,
        root_slot_ids: tuple[int, ...],
        request: CompactOwnerSearchRequestV1,
    ) -> CompactRootBatchV1:
        base = self.root_store.resolve_root_batch(
            root_slot_ids=root_slot_ids,
            request=request,
            copy_observation=False,
        )
        resident, metadata = _resident_observation_from_host_root_batch(
            base,
            device=self.device,
            source_backend=self.source_backend,
            generation_id=int(request.actor_step) + 1,
        )
        self.resolve_count += 1
        return CompactRootBatchV1(
            observation=base.observation,
            legal_mask=base.legal_mask,
            active_root_mask=base.active_root_mask,
            to_play=base.to_play,
            env_row=base.env_row,
            player=base.player,
            policy_env_id=base.policy_env_id,
            target_reward=base.target_reward,
            done_root=base.done_root,
            final_observation=None,
            final_observation_row_mask=base.final_observation_row_mask,
            terminal_row_mask=base.terminal_row_mask,
            autoreset_row_mask=base.autoreset_row_mask,
            metadata={
                **dict(base.metadata),
                **metadata,
                "compact_resident_shared_memory_root_provider_resolved": True,
            },
            resident_observation=resident,
            observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        )

    def close(self) -> None:
        self.root_store.close()


def build_compact_resident_shared_memory_root_provider_v1(
    *,
    spec: CompactSharedMemoryRootStoreSpecV1,
    device: str,
    source_backend: str = "owner_shared_memory_root_store_v1",
) -> CompactResidentSharedMemoryRootProviderV1:
    """Factory for owner-side compact Torch resident root providers."""

    return CompactResidentSharedMemoryRootProviderV1(
        spec=spec,
        device=str(device),
        source_backend=str(source_backend),
    )


def _resident_observation_from_host_root_batch(
    root_batch: CompactRootBatchV1,
    *,
    device: str,
    source_backend: str,
    generation_id: int,
) -> tuple[ResidentObservationBatchV1, dict[str, Any]]:
    import torch

    observation = np.asarray(root_batch.observation)
    if observation.ndim != 4:
        raise ReplayCompatibilityError("resident shared root observation must be rank-4")
    root_count = int(observation.shape[0])
    metadata = dict(getattr(root_batch, "metadata", {}) or {})
    batch_size = int(metadata.get("batch_size") or root_count)
    player_count = int(metadata.get("player_count") or 1)
    if batch_size <= 0 or player_count <= 0:
        raise ReplayCompatibilityError("resident shared root batch/player count must be positive")
    if batch_size * player_count != root_count:
        raise ReplayCompatibilityError(
            "resident shared root bridge requires full row-major root batches"
        )
    stack_shape = tuple(int(dim) for dim in observation.shape[1:])
    target_device = torch.device(str(device))
    root_device_observation = torch.as_tensor(
        observation,
        dtype=torch.uint8,
        device=target_device,
    )
    if target_device.type == "cpu":
        root_device_observation = root_device_observation.clone()
    root_device_observation = root_device_observation.contiguous()
    device_observation = root_device_observation.reshape(
        batch_size,
        player_count,
        *stack_shape,
    )
    final_mask = _resident_final_env_mask(
        root_batch,
        batch_size=batch_size,
    )
    final_device_observation = None
    root_final_device_observation = None
    final_device_observation_rows = None
    final_device_observation_row_indices = None
    final_storage = "none"
    final_sparse_bytes = 0
    final_dense_clone_avoided_bytes = 0
    if bool(final_mask.any()):
        final_device_observation_row_indices = np.flatnonzero(final_mask).astype(
            np.int32,
            copy=False,
        )
        final_index_tensor = torch.as_tensor(
            final_device_observation_row_indices,
            dtype=torch.long,
            device=target_device,
        )
        final_device_observation_rows = (
            device_observation.index_select(0, final_index_tensor).clone().contiguous()
        )
        final_storage = "sparse_rows"
        final_sparse_bytes = int(
            final_device_observation_rows.numel() * final_device_observation_rows.element_size()
        )
        final_dense_clone_avoided_bytes = int(
            device_observation.numel() * device_observation.element_size()
        )
    resident_metadata = {
        "resident_observation_owner": "CompactResidentSharedMemoryRootProviderV1",
        "resident_observation_source_backend": str(source_backend),
        "resident_observation_created_in_owner_process": True,
        "resident_observation_h2d_bytes": float(int(observation.nbytes)),
        "resident_observation_d2h_bytes": 0.0,
        "resident_observation_host_fallback_allowed": False,
        "resident_final_observation_row_count": int(final_mask.sum()),
        "resident_final_device_observation_present": bool(final_mask.any()),
        "resident_final_device_observation_storage": final_storage,
        "resident_final_device_observation_sparse_row_count": int(final_mask.sum()),
        "resident_final_device_observation_sparse_bytes": int(final_sparse_bytes),
        "resident_final_device_observation_dense_clone_avoided_bytes": int(
            final_dense_clone_avoided_bytes
        ),
    }
    resident = ResidentObservationBatchV1(
        device_observation=device_observation,
        root_device_observation=root_device_observation,
        generation_id=int(generation_id),
        batch_size=batch_size,
        player_count=player_count,
        stack_shape=stack_shape,
        dtype=str(root_device_observation.dtype),
        device=str(root_device_observation.device),
        row_major_order=True,
        fresh_for_step_index=int(generation_id),
        source_backend=str(source_backend),
        host_fallback_allowed=False,
        metadata=resident_metadata,
        final_device_observation=final_device_observation,
        root_final_device_observation=root_final_device_observation,
        final_observation_row_mask=final_mask,
        final_device_observation_rows=final_device_observation_rows,
        final_device_observation_row_indices=final_device_observation_row_indices,
    )
    bridge_metadata = {
        "compact_owner_search_resident_root_bridge_ready": True,
        "owner_search_compact_torch_resident_root_bridge_ready": True,
        "compact_owner_search_resident_root_bridge_kind": (
            "shared_memory_host_root_to_owner_resident_tensor_v1"
        ),
        "compact_owner_search_resident_root_bridge_device": str(root_device_observation.device),
        "compact_owner_search_resident_root_bridge_h2d_bytes": float(int(observation.nbytes)),
        "compact_owner_search_resident_root_bridge_host_observation_copied": bool(
            metadata.get("compact_shared_memory_root_store_observation_copied", True)
        ),
        "compact_owner_search_resident_root_bridge_generation_id": int(generation_id),
        "compact_owner_search_resident_root_bridge_final_storage": final_storage,
        "compact_owner_search_resident_root_bridge_final_sparse_row_count": int(final_mask.sum()),
        "compact_owner_search_resident_root_bridge_final_sparse_bytes": int(final_sparse_bytes),
        "compact_owner_search_resident_root_bridge_final_dense_clone_avoided_bytes": int(
            final_dense_clone_avoided_bytes
        ),
    }
    return resident, bridge_metadata


def _resident_final_env_mask(
    root_batch: CompactRootBatchV1,
    *,
    batch_size: int,
) -> np.ndarray:
    final_mask = np.asarray(root_batch.final_observation_row_mask, dtype=np.bool_).reshape(-1)
    if final_mask.shape == (int(batch_size),):
        return final_mask.astype(np.bool_, copy=True)
    root_count = int(np.asarray(root_batch.env_row).shape[0])
    if final_mask.shape != (root_count,):
        raise ReplayCompatibilityError("resident shared root final mask shape mismatch")
    env_row = np.asarray(root_batch.env_row, dtype=np.int64).reshape(-1)
    if env_row.shape != (root_count,):
        raise ReplayCompatibilityError("resident shared root env_row shape mismatch")
    if env_row.size and (int(env_row.min()) < 0 or int(env_row.max()) >= int(batch_size)):
        raise ReplayCompatibilityError("resident shared root env_row outside batch")
    env_mask = np.zeros((int(batch_size),), dtype=np.bool_)
    if root_count:
        env_mask[env_row[final_mask]] = True
    return env_mask


def _shared_root_store_array(root_batch: CompactRootBatchV1, field: str) -> np.ndarray:
    source = np.asarray(getattr(root_batch, field))
    root_count = int(np.asarray(root_batch.active_root_mask).shape[0])
    if source.shape[0] == root_count:
        return source
    if field in {
        "final_observation_row_mask",
        "terminal_row_mask",
        "autoreset_row_mask",
    }:
        env_row = np.asarray(root_batch.env_row, dtype=np.int64)
        if env_row.shape == (root_count,) and source.ndim >= 1:
            if env_row.size == 0 or int(env_row.max()) < int(source.shape[0]):
                return source[env_row]
    raise ValueError(f"shared root-store field {field} cannot be aligned to root slots")


def _select_shared_root_slots(array: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Return a view for contiguous slots and a copy for indexed slots."""

    if slots.ndim != 1:
        raise ValueError("root slots must be rank-1")
    if slots.size == 0:
        return array[:0]
    start = int(slots[0])
    expected = np.arange(start, start + int(slots.size), dtype=slots.dtype)
    if np.array_equal(slots, expected):
        return array[start : start + int(slots.size)]
    return array[slots]


class _ThreadOwnerSearchLearnerWorkerV1:
    """Default adapter for the existing same-process owner learner lane."""

    def __init__(self, train_fn: Any) -> None:
        if not callable(train_fn):
            raise ValueError("owner-search learner worker requires callable train_fn")
        self._train_fn = train_fn
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="compact-owner-search-learner",
        )

    @property
    def metadata(self) -> dict[str, Any]:
        pid = os.getpid()
        return {
            "compact_owner_search_owner_async_learner_worker_kind": (
                COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
            ),
            "compact_owner_search_owner_async_learner_worker_resource_scope": "thread",
            "compact_owner_search_owner_async_learner_worker_resource_id": (
                f"local_process:{pid}:compact-owner-search-learner-thread"
            ),
            "compact_owner_search_owner_async_learner_actor_resource_id": (
                f"local_process:{pid}:owner-search"
            ),
            "compact_owner_search_owner_async_learner_worker_parent_pid": pid,
            "compact_owner_search_owner_async_learner_resource_distinct_from_owner": False,
            (
                "compact_owner_search_owner_async_learner_hardware_resource_distinct_from_owner"
            ): False,
        }

    def submit(
        self,
        *,
        request: CompactOwnerSearchRequestV1,
        root_batch: CompactRootBatchV1,
        search_result: CompactSearchResultV1,
        replay_store: Any | None = None,
        learner: Any | None = None,
    ) -> Future[Any]:
        del replay_store, learner
        return self._executor.submit(
            self._train_fn,
            request=request,
            root_batch=root_batch,
            search_result=search_result,
        )

    def done(self, handle: Any) -> bool:
        return bool(handle.done())

    def result(self, handle: Any) -> Any:
        return handle.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class CompactProcessOwnerSearchLearnerWorkerV1:
    """Adapter for owner-search learner updates in a separate local process.

    The owner process still owns replay append/materialization and prepares the
    trainable learner payload. The child owns learner/model/optimizer state and
    returns only public owner-ref plus telemetry.
    """

    def __init__(
        self,
        *,
        learner_factory: Any,
        learner_factory_kwargs: Mapping[str, Any] | None = None,
        max_workers: int = 1,
    ) -> None:
        if not callable(learner_factory):
            raise ValueError("process owner-search learner requires learner_factory")
        self._parent_pid = os.getpid()
        self._learner_factory = learner_factory
        self._learner_factory_kwargs = dict(learner_factory_kwargs or {})
        self._max_workers = int(max_workers)
        self._mp_context = mp.get_context("spawn")
        self._executor: ProcessPoolExecutor | None = None
        self._prepared = False

    @property
    def metadata(self) -> dict[str, Any]:
        parent_pid = int(self._parent_pid)
        return {
            "compact_owner_search_owner_async_learner_worker_kind": (
                COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH
            ),
            "compact_owner_search_owner_async_learner_worker_resource_scope": ("process"),
            "compact_owner_search_owner_async_learner_worker_resource_id": (
                f"local_process_pool:{parent_pid}:compact-owner-search-learner"
            ),
            "compact_owner_search_owner_async_learner_actor_resource_id": (
                f"local_process:{parent_pid}:owner-search"
            ),
            "compact_owner_search_owner_async_learner_worker_parent_pid": parent_pid,
            "compact_owner_search_owner_async_learner_worker_start_method": str(
                self._mp_context.get_start_method()
            ),
            "compact_owner_search_owner_async_learner_resource_distinct_from_owner": True,
            (
                "compact_owner_search_owner_async_learner_hardware_resource_distinct_from_owner"
            ): False,
        }

    def prepare(self) -> None:
        if self._prepared:
            return
        if int(self._max_workers) != 1:
            raise RuntimeError("process owner-search learner requires max_workers=1")
        if _contains_cuda_tensor(self._learner_factory_kwargs):
            self._learner_factory_kwargs = _host_only_clone_for_process(
                self._learner_factory_kwargs
            )
        if _contains_cuda_tensor(self._learner_factory_kwargs):
            cuda_count = int(_count_cuda_tensors(self._learner_factory_kwargs))
            raise RuntimeError(
                "process owner-search learner factory kwargs must be host-only; "
                f"found {cuda_count} CUDA tensor(s)"
            )
        self._executor = ProcessPoolExecutor(
            max_workers=int(self._max_workers),
            mp_context=self._mp_context,
            initializer=_compact_process_owner_search_async_learner_initializer,
            initargs=(self._learner_factory, self._learner_factory_kwargs),
        )
        self._prepared = True

    def submit(
        self,
        *,
        request: CompactOwnerSearchRequestV1,
        root_batch: CompactRootBatchV1,
        search_result: CompactSearchResultV1,
        replay_store: Any | None = None,
        learner: Any | None = None,
    ) -> _CompactOwnerSearchProcessLearnerHandleV1:
        if learner is None:
            raise RuntimeError("process owner-search learner needs owner-side learner")
        prepare_payload = getattr(learner, "prepare_owner_search_learner_payload", None)
        if not callable(prepare_payload):
            raise RuntimeError(
                "process owner-search learner requires prepare_owner_search_learner_payload"
            )
        payload_started = time.perf_counter()
        learner_payload = prepare_payload(
            replay_store=replay_store,
            root_batch=root_batch,
            search_result=search_result,
            request=request,
        )
        payload_prepare_sec = _elapsed(payload_started)
        transport_request = _CompactOwnerSearchProcessLearnerRequestV1(
            request=request,
            learner_payload=learner_payload,
            payload_prepare_sec=float(payload_prepare_sec),
            request_bytes=0,
            request_cuda_tensor_count=0,
        )
        request_cuda_count = int(_count_cuda_tensors(transport_request))
        if request_cuda_count:
            raise RuntimeError("process owner-search learner request contains CUDA tensors")
        request_bytes = _pickle_size_bytes(transport_request)
        transport_request = _CompactOwnerSearchProcessLearnerRequestV1(
            request=request,
            learner_payload=learner_payload,
            payload_prepare_sec=float(payload_prepare_sec),
            request_bytes=int(request_bytes),
            request_cuda_tensor_count=int(request_cuda_count),
        )
        self.prepare()
        assert self._executor is not None
        future = self._executor.submit(
            _run_compact_process_owner_search_async_learner,
            transport_request,
        )
        return _CompactOwnerSearchProcessLearnerHandleV1(
            future=future,
            request=request,
            payload_prepare_sec=float(payload_prepare_sec),
            request_bytes=int(request_bytes),
            request_cuda_tensor_count=int(request_cuda_count),
        )

    def done(self, handle: _CompactOwnerSearchProcessLearnerHandleV1) -> bool:
        return bool(handle.future.done())

    def result(self, handle: _CompactOwnerSearchProcessLearnerHandleV1) -> Any:
        raw = handle.future.result()
        if not isinstance(raw, Mapping):
            raise RuntimeError("process owner-search learner returned non-mapping result")
        owner_ref, updates, sample_telemetry, learner_telemetry = (
            _owner_learner_result_tuple_from_mapping(
                raw.get("learner_result"),
                require_owner_ref=bool(handle.request.refresh_model),
            )
        )
        worker_runtime = raw.get("worker_runtime")
        if isinstance(worker_runtime, Mapping):
            learner_telemetry = dict(learner_telemetry)
            learner_telemetry.update(
                {
                    "compact_owner_search_owner_async_learner_request_host_only": (
                        int(handle.request_cuda_tensor_count) == 0
                    ),
                    "compact_owner_search_owner_async_learner_request_cuda_tensor_count": int(
                        handle.request_cuda_tensor_count
                    ),
                    "compact_owner_search_owner_async_learner_result_host_only": bool(
                        worker_runtime.get("process_result_host_only", False)
                    ),
                    "compact_owner_search_owner_async_learner_result_cuda_tensor_count": int(
                        worker_runtime.get("process_result_cuda_tensor_count", 0) or 0
                    ),
                    "compact_owner_search_owner_async_learner_request_bytes": int(
                        handle.request_bytes
                    ),
                    "compact_owner_search_owner_async_learner_result_bytes": int(
                        worker_runtime.get("process_result_bytes", 0) or 0
                    ),
                    "compact_owner_search_owner_async_learner_worker_pid": int(
                        worker_runtime.get("worker_pid", 0) or 0
                    ),
                    "compact_owner_search_owner_async_learner_worker_resource_id": str(
                        worker_runtime.get("worker_resource_id") or ""
                    ),
                    "compact_owner_search_owner_async_learner_worker_job_wall_sec": float(
                        worker_runtime.get("worker_job_wall_sec", 0.0) or 0.0
                    ),
                    ("compact_owner_search_owner_async_learner_payload_prepare_sec"): float(
                        handle.payload_prepare_sec
                    ),
                    "compact_owner_search_owner_async_learner_worker_completed_count": int(
                        worker_runtime.get("worker_completed_count", 0) or 0
                    ),
                    "compact_owner_search_owner_async_learner_worker_initialized_count": int(
                        worker_runtime.get("worker_initialized_count", 0) or 0
                    ),
                    "compact_owner_search_owner_async_learner_worker_owns_model_state": bool(
                        worker_runtime.get("worker_owns_model_state", False)
                    ),
                    (
                        "compact_owner_search_owner_async_learner_worker_pid_distinct_from_owner"
                    ): bool(
                        int(worker_runtime.get("worker_pid", 0) or 0)
                        not in {0, int(self._parent_pid)}
                    ),
                }
            )
        return owner_ref, updates, sample_telemetry, learner_telemetry

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        self._prepared = False
        if executor is not None:
            executor.shutdown(wait=True)


class CompactOwnerSearchServiceV1:
    """Run compact search, replay ingest, learner publish, and search refresh inside one owner."""

    profile_only = True
    calls_train_muzero = False
    touches_live_runs = False

    def __init__(
        self,
        *,
        root_provider: Any,
        search_service: Any,
        replay_store: Any | None = None,
        learner: Any | None = None,
        owner_kind: str = COMPACT_OWNER_SEARCH_KIND_IN_PROCESS,
        use_inner_two_phase_device_replay: bool = False,
        async_learner_worker: bool = False,
        async_learner_worker_kind: str = (
            COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
        ),
        async_learner_max_pending: int = 1,
        async_learner_worker_adapter: Any | None = None,
        action_result_slot_table: CompactOwnerActionResultSlotTableV1 | None = None,
    ) -> None:
        self.root_provider = root_provider
        self.search_service = search_service
        self.replay_store = replay_store
        self.learner = learner
        self.owner_kind = str(owner_kind)
        self.action_result_slot_table = action_result_slot_table
        self.use_inner_two_phase_device_replay = bool(use_inner_two_phase_device_replay)
        if not self.owner_kind:
            raise ValueError("owner_kind must be non-empty")
        self.async_learner_worker_enabled = bool(async_learner_worker)
        self.async_learner_max_pending = int(async_learner_max_pending)
        if self.async_learner_worker_enabled and self.learner is None:
            raise ValueError("async owner-search learner worker requires a learner")
        if self.async_learner_worker_enabled and self.async_learner_max_pending <= 0:
            raise ValueError("async_learner_max_pending must be positive")
        self.async_learner_worker_kind = str(async_learner_worker_kind)
        if not self.async_learner_worker_kind:
            self.async_learner_worker_kind = (
                COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
            )
        if self.async_learner_worker_kind not in COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS:
            allowed = ", ".join(COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS)
            raise ValueError(f"async owner-search learner worker kind must be one of {allowed}")
        if async_learner_worker_adapter is not None and not self.async_learner_worker_enabled:
            raise ValueError("async_learner_worker_adapter requires async_learner_worker=True")
        if (
            async_learner_worker_adapter is None
            and self.async_learner_worker_enabled
            and self.async_learner_worker_kind
            != COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
        ):
            raise ValueError("non-thread async owner-search learner worker requires an adapter")
        self._learner_worker = None
        self._learner_worker_metadata: dict[str, Any] = {}
        if self.async_learner_worker_enabled:
            self._learner_worker = (
                async_learner_worker_adapter
                if async_learner_worker_adapter is not None
                else _ThreadOwnerSearchLearnerWorkerV1(self._train_and_publish)
            )
            metadata = getattr(self._learner_worker, "metadata", {})
            if isinstance(metadata, Mapping):
                self._learner_worker_metadata = dict(metadata)
        self.request_count = 0
        self.replay_append_count = 0
        self.owner_replay_append_telemetry: dict[str, Any] = {}
        self.learner_update_count = 0
        self.last_model_owner_ref: dict[str, Any] | None = None
        self.last_search_worker_state: dict[str, Any] | None = None
        self._root_batch_cache: dict[int, CompactRootBatchV1] = {}
        self._search_result_cache: dict[str, _CompactOwnerCachedSearchV1] = {}
        self._pending_maintenance: list[_CompactOwnerMaintenanceWorkV1] = []
        self._pending_learner_jobs: list[_CompactOwnerPendingLearnerJobV1] = []
        self._maintenance_failed_message = ""
        self.owner_async_learner_submit_count = 0
        self.owner_async_learner_completed_count = 0
        self.owner_async_learner_max_pending_observed = 0
        self.owner_async_learner_wait_count = 0
        self.owner_async_learner_wait_sec = 0.0
        self.owner_action_while_async_learner_pending_count = 0
        self.owner_async_learner_failed = False
        self.owner_maintenance_root_cache_snapshot_count = 0
        self.owner_maintenance_root_cache_snapshot_full_entry_count = 0
        self.owner_maintenance_root_cache_snapshot_retained_entry_count = 0
        self.owner_maintenance_root_cache_snapshot_required_entry_count = 0
        self.owner_maintenance_root_cache_snapshot_dropped_entry_count = 0
        self.owner_maintenance_root_cache_snapshot_full_fallback_count = 0
        self._state_lock = threading.RLock()
        self._search_lock = threading.RLock()

    @property
    def metadata(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "compact_owner_search_service_schema_id": COMPACT_OWNER_SEARCH_SERVICE_SCHEMA_ID,
                "compact_owner_search_service_kind": self.owner_kind,
                "compact_owner_search_service_pid": os.getpid(),
                "compact_owner_search_service_worker_owns_search_state": True,
                "compact_owner_search_service_worker_owns_replay_state": (
                    self.replay_store is not None
                ),
                "compact_owner_search_service_worker_owns_model_state": (self.learner is not None),
                "compact_owner_search_use_inner_two_phase_device_replay": bool(
                    self.use_inner_two_phase_device_replay
                ),
                "compact_owner_search_service_request_count": int(self.request_count),
                "compact_owner_search_service_replay_append_count": int(self.replay_append_count),
                **dict(self.owner_replay_append_telemetry),
                "compact_owner_search_service_learner_update_count": int(self.learner_update_count),
                "compact_owner_search_service_pending_maintenance_count": int(
                    len(self._pending_maintenance)
                ),
                "compact_owner_search_owner_maintenance_root_cache_snapshot_count": int(
                    self.owner_maintenance_root_cache_snapshot_count
                ),
                "compact_owner_search_owner_maintenance_root_cache_snapshot_full_entry_count": int(
                    self.owner_maintenance_root_cache_snapshot_full_entry_count
                ),
                "compact_owner_search_owner_maintenance_root_cache_snapshot_retained_entry_count": int(
                    self.owner_maintenance_root_cache_snapshot_retained_entry_count
                ),
                "compact_owner_search_owner_maintenance_root_cache_snapshot_required_entry_count": int(
                    self.owner_maintenance_root_cache_snapshot_required_entry_count
                ),
                "compact_owner_search_owner_maintenance_root_cache_snapshot_dropped_entry_count": int(
                    self.owner_maintenance_root_cache_snapshot_dropped_entry_count
                ),
                "compact_owner_search_owner_maintenance_root_cache_snapshot_full_fallback_count": int(
                    self.owner_maintenance_root_cache_snapshot_full_fallback_count
                ),
                **self._owner_async_learner_metadata_locked(),
                "compact_owner_search_service_maintenance_failed": bool(
                    self._maintenance_failed_message
                ),
                **(
                    {}
                    if self.action_result_slot_table is None
                    else self.action_result_slot_table.metadata
                ),
            }

    def _record_owner_replay_append_telemetry(self, value: Any) -> None:
        if not isinstance(value, Mapping) or not value:
            return
        telemetry = {
            str(key): item
            for key, item in value.items()
            if str(key).startswith(COMPACT_OWNER_SEARCH_REPLAY_APPEND_TELEMETRY_PREFIXES)
        }
        if not telemetry:
            return
        with self._state_lock:
            self.owner_replay_append_telemetry.update(telemetry)

    def action_result_payload_for_request(
        self,
        request: CompactOwnerSearchRequestV1,
        result: CompactOwnerSearchResultV1,
    ) -> dict[str, Any]:
        payload = result.to_dict()
        slot_table = self.action_result_slot_table
        slot_id = int(getattr(request, "action_result_slot_id", -1))
        if slot_table is None or slot_id < 0:
            return payload
        return slot_table.write(slot_id, payload)

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

    def _record_inner_device_replay_payload_telemetry(
        self,
        replay_payload: Any,
    ) -> None:
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
                "owner-search deferred one-simulation replay payload was not "
                "materialized during replay flush"
            )
        if deferred and identity_match_present and not identity_match:
            raise RuntimeError(
                "owner-search deferred one-simulation replay payload crossed a model refresh"
            )
        if refresh_crossed_count:
            raise RuntimeError(
                "owner-search deferred one-simulation replay payload reported "
                "model-refresh crossing"
            )
        telemetry: dict[str, Any] = {}
        if device_payload_flushed:
            telemetry["compact_owner_search_inner_device_replay_payload_flushed_count"] = 1
        if deferred:
            telemetry.update(
                {
                    "compact_owner_search_inner_deferred_one_simulation_replay_payload_flush_count": 1,
                    "compact_owner_search_inner_deferred_one_simulation_replay_materialized_on_flush_count": (
                        1 if materialized_on_flush else 0
                    ),
                    "compact_owner_search_inner_deferred_one_simulation_replay_recurrent_inference_calls": (
                        self._telemetry_float(
                            metadata.get(
                                "compact_torch_search_deferred_one_simulation_replay_recurrent_inference_calls"
                            )
                        )
                    ),
                    "compact_owner_search_inner_deferred_one_simulation_model_identity_match_count": (
                        1 if identity_match else 0
                    ),
                    "compact_owner_search_inner_deferred_one_simulation_model_refresh_crossed_count": (
                        refresh_crossed_count
                    ),
                    "compact_owner_search_inner_pending_deferred_replay_payload_count_max": (
                        self._telemetry_int(
                            metadata.get(
                                "compact_torch_search_pending_deferred_replay_payload_count"
                            )
                        )
                    ),
                    "compact_owner_search_inner_pending_deferred_replay_payload_final_count": (
                        self._telemetry_int(
                            metadata.get(
                                "compact_torch_search_pending_deferred_replay_payload_final_count"
                            )
                        )
                    ),
                    "compact_owner_search_inner_deferred_one_simulation_replay_flush_sec": (
                        self._telemetry_float(
                            metadata.get(
                                "compact_torch_search_deferred_one_simulation_replay_flush_sec"
                            )
                        )
                    ),
                    "compact_owner_search_inner_device_replay_payload_flush_sec": (
                        self._telemetry_float(
                            metadata.get(
                                "compact_torch_search_service_device_replay_payload_flush_sec"
                            )
                        )
                    ),
                    "compact_owner_search_inner_replay_payload_d2h_bytes": (
                        self._telemetry_float(
                            metadata.get("compact_torch_search_service_replay_payload_d2h_bytes")
                        )
                    ),
                    "compact_owner_search_inner_deferred_one_simulation_action_model_state_digest": str(
                        metadata.get(
                            "compact_torch_search_deferred_one_simulation_action_model_state_digest"
                        )
                        or ""
                    ),
                    "compact_owner_search_inner_deferred_one_simulation_flush_model_state_digest": str(
                        metadata.get(
                            "compact_torch_search_deferred_one_simulation_flush_model_state_digest"
                        )
                        or ""
                    ),
                }
            )
        if not telemetry:
            return
        additive_keys = {
            key
            for key in telemetry
            if key
            not in {
                "compact_owner_search_inner_pending_deferred_replay_payload_final_count",
                "compact_owner_search_inner_pending_deferred_replay_payload_count_max",
                "compact_owner_search_inner_deferred_one_simulation_action_model_state_digest",
                "compact_owner_search_inner_deferred_one_simulation_flush_model_state_digest",
            }
        }
        with self._state_lock:
            for key, value in telemetry.items():
                if key in additive_keys:
                    self.owner_replay_append_telemetry[key] = self._telemetry_float(
                        self.owner_replay_append_telemetry.get(key)
                    ) + self._telemetry_float(value)
                elif key == (
                    "compact_owner_search_inner_pending_deferred_replay_payload_count_max"
                ):
                    self.owner_replay_append_telemetry[key] = max(
                        self._telemetry_int(self.owner_replay_append_telemetry.get(key)),
                        self._telemetry_int(value),
                    )
                else:
                    self.owner_replay_append_telemetry[key] = value

    def _owner_sample_telemetry_with_replay_append(
        self,
        owner_sample_telemetry: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(owner_sample_telemetry or {})
        with self._state_lock:
            merged.update(self.owner_replay_append_telemetry)
            merged.update(self._owner_root_cache_snapshot_metadata_locked())
        return merged

    def _owner_root_cache_snapshot_metadata_locked(self) -> dict[str, Any]:
        return {
            "compact_owner_search_owner_maintenance_root_cache_snapshot_count": int(
                self.owner_maintenance_root_cache_snapshot_count
            ),
            "compact_owner_search_owner_maintenance_root_cache_snapshot_full_entry_count": int(
                self.owner_maintenance_root_cache_snapshot_full_entry_count
            ),
            "compact_owner_search_owner_maintenance_root_cache_snapshot_retained_entry_count": int(
                self.owner_maintenance_root_cache_snapshot_retained_entry_count
            ),
            "compact_owner_search_owner_maintenance_root_cache_snapshot_required_entry_count": int(
                self.owner_maintenance_root_cache_snapshot_required_entry_count
            ),
            "compact_owner_search_owner_maintenance_root_cache_snapshot_dropped_entry_count": int(
                self.owner_maintenance_root_cache_snapshot_dropped_entry_count
            ),
            "compact_owner_search_owner_maintenance_root_cache_snapshot_full_fallback_count": int(
                self.owner_maintenance_root_cache_snapshot_full_fallback_count
            ),
        }

    def _record_owner_root_cache_snapshot_locked(
        self,
        metadata: Mapping[str, Any],
    ) -> None:
        self.owner_maintenance_root_cache_snapshot_count += 1
        self.owner_maintenance_root_cache_snapshot_full_entry_count += int(
            metadata.get(
                "compact_owner_search_owner_maintenance_root_cache_snapshot_full_entry_count",
                0,
            )
            or 0
        )
        self.owner_maintenance_root_cache_snapshot_retained_entry_count += int(
            metadata.get(
                "compact_owner_search_owner_maintenance_root_cache_snapshot_retained_entry_count",
                0,
            )
            or 0
        )
        self.owner_maintenance_root_cache_snapshot_required_entry_count += int(
            metadata.get(
                "compact_owner_search_owner_maintenance_root_cache_snapshot_required_entry_count",
                0,
            )
            or 0
        )
        self.owner_maintenance_root_cache_snapshot_dropped_entry_count += int(
            metadata.get(
                "compact_owner_search_owner_maintenance_root_cache_snapshot_dropped_entry_count",
                0,
            )
            or 0
        )
        self.owner_maintenance_root_cache_snapshot_full_fallback_count += int(
            metadata.get(
                "compact_owner_search_owner_maintenance_root_cache_snapshot_full_fallback_count",
                0,
            )
            or 0
        )

    def _owner_async_learner_metadata_locked(self) -> dict[str, Any]:
        metadata = dict(self._learner_worker_metadata)
        metadata.update(
            {
                "compact_owner_search_owner_async_learner_worker_enabled": bool(
                    self.async_learner_worker_enabled
                ),
                "compact_owner_search_owner_async_learner_worker_kind": (
                    str(
                        metadata.get("compact_owner_search_owner_async_learner_worker_kind")
                        or COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
                    )
                    if self.async_learner_worker_enabled
                    else COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_NONE
                ),
                "compact_owner_search_owner_async_learner_max_pending": int(
                    self.async_learner_max_pending
                ),
                "compact_owner_search_owner_async_learner_submit_count": int(
                    self.owner_async_learner_submit_count
                ),
                "compact_owner_search_owner_async_learner_completed_count": int(
                    self.owner_async_learner_completed_count
                ),
                "compact_owner_search_owner_async_learner_pending_count": int(
                    len(self._pending_learner_jobs)
                ),
                "compact_owner_search_owner_async_learner_max_pending_observed": int(
                    self.owner_async_learner_max_pending_observed
                ),
                "compact_owner_search_owner_async_learner_wait_count": int(
                    self.owner_async_learner_wait_count
                ),
                "compact_owner_search_owner_async_learner_wait_sec": float(
                    self.owner_async_learner_wait_sec
                ),
                ("compact_owner_search_owner_action_while_async_learner_pending_count"): int(
                    self.owner_action_while_async_learner_pending_count
                ),
                "compact_owner_search_owner_async_learner_failed": bool(
                    self.owner_async_learner_failed
                ),
            }
        )
        return metadata

    def _owner_async_learner_metadata(self) -> dict[str, Any]:
        with self._state_lock:
            return self._owner_async_learner_metadata_locked()

    def close(self) -> None:
        worker = self._learner_worker
        if worker is None:
            return
        try:
            self._poll_owner_learner_jobs(wait=True)
        finally:
            close = getattr(worker, "close", None)
            if callable(close):
                close()
            self._learner_worker = None

    def run(self, request: CompactOwnerSearchRequestV1) -> CompactOwnerSearchResultV1:
        started = time.perf_counter()
        request = _validate_owner_search_request(request)
        if _contains_cuda_tensor(request):
            raise RuntimeError("owner-search parent request must not contain CUDA tensors")
        with self._state_lock:
            if self._pending_learner_jobs:
                self.owner_action_while_async_learner_pending_count += 1
        request_bytes = _pickle_size_bytes(request)
        root_started = time.perf_counter()
        root_batch = _resolve_root_batch(
            self.root_provider,
            request.root_slot_ids,
            request=request,
        )
        self._root_batch_cache[int(request.actor_step)] = root_batch
        root_resolve_sec = _elapsed(root_started)
        search_started = time.perf_counter()
        search_result = self.search_service.run(root_batch)
        validate_compact_search_result_identity_v1(root_batch, search_result)
        search_result = _attach_owner_resident_bridge_metadata(
            search_result,
            root_batch=root_batch,
        )
        search_sec = _elapsed(search_started)
        replay_started = time.perf_counter()
        append_count, owner_action_feedback = self._append_replay(
            request=request,
            root_batch=root_batch,
            search_result=search_result,
        )
        replay_sec = _elapsed(replay_started)
        train_started = time.perf_counter()
        (
            owner_ref,
            learner_updates,
            owner_sample_telemetry,
            owner_learner_telemetry,
        ) = self._train_and_publish(
            request=request,
            root_batch=root_batch,
            search_result=search_result,
        )
        owner_sample_telemetry = self._owner_sample_telemetry_with_replay_append(
            owner_sample_telemetry
        )
        train_sec = _elapsed(train_started)
        public_search_state: dict[str, Any] | None = None
        refresh_sec = 0.0
        if owner_ref is not None:
            refresh_started = time.perf_counter()
            search_state = self._refresh_search_from_owner_ref(
                request=request,
                owner_ref=owner_ref,
                learner_update_count=learner_updates,
            )
            public_search_state = _public_owner_search_worker_state_v1(search_state)
            refresh_sec = _elapsed(refresh_started)
        self.request_count += 1
        self._root_batch_cache = _trim_owner_root_batch_cache_v1(self._root_batch_cache)
        self.replay_append_count += int(append_count)
        self.learner_update_count += int(learner_updates)
        if owner_ref is not None:
            self.last_model_owner_ref = dict(owner_ref)
        if public_search_state is not None:
            self.last_search_worker_state = dict(public_search_state)
        selected_action = tuple(int(value) for value in np.asarray(search_result.selected_action))
        dense_joint_action_metadata = (
            _owner_dense_joint_action_metadata_v1(
                _owner_dense_joint_action_from_search_result_v1(
                    root_batch,
                    search_result,
                )
            )
            if int(request.action_result_slot_id) >= 0
            else {}
        )
        search_result_payload = _compact_search_result_payload_v1(search_result)
        search_result_byte_counts = _compact_search_result_byte_counts_v1(search_result)
        search_result_payload_bytes = _pickle_size_bytes(search_result_payload)
        public_stub = {
            "schema_id": COMPACT_OWNER_SEARCH_RESULT_SCHEMA_ID,
            "request_id": int(request.request_id),
            "selected_action": selected_action,
            **dense_joint_action_metadata,
            "learner_update_count": int(self.learner_update_count),
            "model_owner_ref_digest": _owner_ref_digest(owner_ref),
            "owner_sample_telemetry": dict(owner_sample_telemetry),
            "owner_learner_telemetry": dict(owner_learner_telemetry),
            "owner_action_feedback": dict(owner_action_feedback),
            "search_result_payload": search_result_payload,
        }
        result_bytes = _pickle_size_bytes(public_stub)
        timing = {
            "root_resolve_sec": float(root_resolve_sec),
            "search_sec": float(search_sec),
            "replay_append_sec": float(replay_sec),
            "learner_train_sec": float(train_sec),
            "search_refresh_sec": float(refresh_sec),
            "wall_sec": float(_elapsed(started)),
        }
        search_refresh_update_count = _search_refresh_update_count(public_search_state)
        return CompactOwnerSearchResultV1(
            schema_id=COMPACT_OWNER_SEARCH_RESULT_SCHEMA_ID,
            owner_kind=self.owner_kind,
            request_id=int(request.request_id),
            actor_step=int(request.actor_step),
            owner_pid=os.getpid(),
            root_slot_count=len(tuple(request.root_slot_ids)),
            active_root_count=int(search_result.selected_action.shape[0]),
            selected_action=selected_action,
            search_impl=str(search_result.metadata.get("search_impl") or ""),
            num_simulations=int(search_result.metadata.get("num_simulations") or 0),
            replay_append_entry_count=_owner_replay_append_entry_count_v1(
                request.replay_append_entries
            ),
            replay_append_count=int(append_count),
            learner_update_count=int(self.learner_update_count),
            model_owner_ref_returned=owner_ref is not None,
            model_owner_ref_digest=_owner_ref_digest(owner_ref),
            model_state_return_count=0,
            model_state_bytes=0,
            model_state_snapshot_return_count=0,
            root_observation_bytes_sent=0,
            request_cuda_tensor_count=0,
            result_cuda_tensor_count=0,
            request_bytes=int(request_bytes),
            result_bytes=int(result_bytes),
            worker_owns_search_state=True,
            worker_owns_replay_state=self.replay_store is not None,
            worker_owns_model_state=self.learner is not None,
            search_consumed_learner_update=(
                int(learner_updates) == 0
                or int(search_refresh_update_count) == int(self.learner_update_count)
            ),
            search_refresh_update_count=int(search_refresh_update_count),
            search_worker_state=(
                None if public_search_state is None else dict(public_search_state)
            ),
            search_result_payload=search_result_payload,
            search_result_payload_bytes=int(search_result_payload_bytes),
            search_selected_action_bytes=int(
                search_result_byte_counts["search_selected_action_bytes"]
            ),
            search_visit_policy_bytes=int(search_result_byte_counts["search_visit_policy_bytes"]),
            search_root_value_bytes=int(search_result_byte_counts["search_root_value_bytes"]),
            search_optional_array_bytes=int(
                search_result_byte_counts["search_optional_array_bytes"]
            ),
            owner_sample_telemetry=dict(owner_sample_telemetry),
            owner_learner_telemetry=dict(owner_learner_telemetry),
            owner_action_feedback=dict(owner_action_feedback),
            timing=timing,
            search_result_metadata=dict(search_result.metadata),
            replay_append_transport_entry_count=(
                _owner_replay_append_transport_entry_count_v1(request.replay_append_entries)
            ),
            replay_append_transition_batch_count=(
                _owner_replay_append_transition_batch_count_v1(request.replay_append_entries)
            ),
            replay_append_transition_batch_entry_count=(
                _owner_replay_append_transition_batch_entry_count_total_v1(
                    request.replay_append_entries
                )
            ),
            **dense_joint_action_metadata,
        )

    def run_action(self, request: CompactOwnerSearchRequestV1) -> CompactOwnerSearchResultV1:
        """Run only action-critical root resolve and search, then stage maintenance."""

        self._raise_if_maintenance_failed()
        started = time.perf_counter()
        request = _validate_owner_search_request(request)
        if _contains_cuda_tensor(request):
            raise RuntimeError("owner-search parent request must not contain CUDA tensors")
        with self._state_lock:
            if self._pending_learner_jobs:
                self.owner_action_while_async_learner_pending_count += 1
        request_bytes = _pickle_size_bytes(request)
        root_started = time.perf_counter()
        root_batch = _resolve_root_batch(
            self.root_provider,
            request.root_slot_ids,
            request=request,
        )
        with self._state_lock:
            self._root_batch_cache[int(request.actor_step)] = root_batch
        root_resolve_sec = _elapsed(root_started)
        search_started = time.perf_counter()
        action_step: CompactSearchActionStepV1 | None = None
        inner_replay_payload_handle = ""
        use_inner_two_phase = (
            self.use_inner_two_phase_device_replay
            and str(root_batch.observation_source) == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
            and callable(getattr(self.search_service, "run_action_step", None))
            and callable(getattr(self.search_service, "flush_device_replay_payload", None))
        )
        with self._search_lock:
            if use_inner_two_phase:
                action_step = self.search_service.run_action_step(root_batch)
                _validate_owner_action_step_identity_v1(root_batch, action_step)
                inner_replay_payload_handle = str(action_step.replay_payload_handle)
                search_result = _compat_search_result_from_action_step_v1(
                    root_batch,
                    action_step,
                    metadata={
                        "compact_owner_search_inner_two_phase_action_step": True,
                        "compact_owner_search_inner_replay_payload_handle": (
                            inner_replay_payload_handle
                        ),
                    },
                )
            else:
                search_result = self.search_service.run(root_batch)
                validate_compact_search_result_identity_v1(root_batch, search_result)
        search_result = _attach_owner_resident_bridge_metadata(
            search_result,
            root_batch=root_batch,
        )
        search_sec = _elapsed(search_started)
        with self._state_lock:
            public_search_state = _public_owner_search_worker_state_v1(
                self.last_search_worker_state
            )
            self.request_count += 1
            handle = (
                f"compact-owner-search-service:{int(request.actor_step)}:{int(request.request_id)}"
            )
            self._search_result_cache[handle] = _CompactOwnerCachedSearchV1(
                record_index=int(request.actor_step),
                root_batch=root_batch,
                search_result=search_result,
                action_step=action_step,
                inner_replay_payload_handle=inner_replay_payload_handle,
            )
            if request.replay_append_entries or int(request.train_steps) > 0:
                root_batch_cache, root_batch_cache_metadata = (
                    _owner_root_batch_cache_for_replay_entries_v1(
                        self._root_batch_cache,
                        request.replay_append_entries,
                        current_actor_step=int(request.actor_step),
                        current_root_batch=root_batch,
                    )
                )
                self._record_owner_root_cache_snapshot_locked(root_batch_cache_metadata)
                self._pending_maintenance.append(
                    _CompactOwnerMaintenanceWorkV1(
                        request=request,
                        root_batch=root_batch,
                        search_result=search_result,
                        root_batch_cache=root_batch_cache,
                    )
                )
            self._root_batch_cache = _trim_owner_root_batch_cache_v1(self._root_batch_cache)
            learner_update_count = int(self.learner_update_count)
            model_owner_ref = (
                None if self.last_model_owner_ref is None else dict(self.last_model_owner_ref)
            )
            pending_maintenance_count = len(self._pending_maintenance)
        selected_action = tuple(int(value) for value in np.asarray(search_result.selected_action))
        dense_joint_action_metadata = (
            _owner_dense_joint_action_metadata_v1(
                _owner_dense_joint_action_from_search_result_v1(
                    root_batch,
                    search_result,
                )
            )
            if int(request.action_result_slot_id) >= 0
            else {}
        )
        search_result_byte_counts = _compact_search_result_byte_counts_v1(search_result)
        search_refresh_update_count = _search_refresh_update_count(public_search_state)
        public_stub = {
            "schema_id": COMPACT_OWNER_SEARCH_RESULT_SCHEMA_ID,
            "request_id": int(request.request_id),
            "selected_action": selected_action,
            **dense_joint_action_metadata,
            "learner_update_count": int(learner_update_count),
            "model_owner_ref_digest": _owner_ref_digest(model_owner_ref),
            "owner_sample_telemetry": {},
            "owner_learner_telemetry": {},
            "owner_action_feedback": _empty_owner_action_feedback_v1(),
            "search_result_payload": {},
            "replay_payload_handle": handle,
        }
        result_bytes = _pickle_size_bytes(public_stub)
        timing = {
            "root_resolve_sec": float(root_resolve_sec),
            "search_sec": float(search_sec),
            "replay_append_sec": 0.0,
            "learner_train_sec": 0.0,
            "search_refresh_sec": 0.0,
            "wall_sec": float(_elapsed(started)),
        }
        search_result_metadata = dict(search_result.metadata)
        search_result_metadata["compact_owner_search_use_inner_two_phase_device_replay"] = bool(
            self.use_inner_two_phase_device_replay
        )
        return CompactOwnerSearchResultV1(
            schema_id=COMPACT_OWNER_SEARCH_RESULT_SCHEMA_ID,
            owner_kind=self.owner_kind,
            request_id=int(request.request_id),
            actor_step=int(request.actor_step),
            owner_pid=os.getpid(),
            root_slot_count=len(tuple(request.root_slot_ids)),
            active_root_count=int(search_result.selected_action.shape[0]),
            selected_action=selected_action,
            search_impl=str(search_result.metadata.get("search_impl") or ""),
            num_simulations=int(search_result.metadata.get("num_simulations") or 0),
            replay_append_entry_count=_owner_replay_append_entry_count_v1(
                request.replay_append_entries
            ),
            replay_append_count=0,
            learner_update_count=int(learner_update_count),
            model_owner_ref_returned=model_owner_ref is not None,
            model_owner_ref_digest=_owner_ref_digest(model_owner_ref),
            model_state_return_count=0,
            model_state_bytes=0,
            model_state_snapshot_return_count=0,
            root_observation_bytes_sent=0,
            request_cuda_tensor_count=0,
            result_cuda_tensor_count=0,
            request_bytes=int(request_bytes),
            result_bytes=int(result_bytes),
            worker_owns_search_state=True,
            worker_owns_replay_state=self.replay_store is not None,
            worker_owns_model_state=self.learner is not None,
            search_consumed_learner_update=(
                int(search_refresh_update_count) == int(learner_update_count)
            ),
            search_refresh_update_count=int(search_refresh_update_count),
            search_worker_state=(
                None if public_search_state is None else dict(public_search_state)
            ),
            search_result_payload={},
            search_result_payload_bytes=0,
            search_selected_action_bytes=int(
                search_result_byte_counts["search_selected_action_bytes"]
            ),
            search_visit_policy_bytes=0,
            search_root_value_bytes=0,
            search_optional_array_bytes=0,
            owner_sample_telemetry={},
            owner_learner_telemetry={},
            owner_action_feedback=_empty_owner_action_feedback_v1(),
            timing=timing,
            search_result_metadata=search_result_metadata,
            owner_maintenance_deferred=True,
            owner_maintenance_staged_work_count=(
                1 if request.replay_append_entries or int(request.train_steps) > 0 else 0
            ),
            owner_maintenance_pending_work_count=int(pending_maintenance_count),
            replay_payload_handle=handle,
            inner_two_phase_action_step=bool(action_step is not None),
            inner_device_replay_payload_deferred=bool(inner_replay_payload_handle),
            replay_append_transport_entry_count=(
                _owner_replay_append_transport_entry_count_v1(request.replay_append_entries)
            ),
            replay_append_transition_batch_count=(
                _owner_replay_append_transition_batch_count_v1(request.replay_append_entries)
            ),
            replay_append_transition_batch_entry_count=(
                _owner_replay_append_transition_batch_entry_count_total_v1(
                    request.replay_append_entries
                )
            ),
            **dense_joint_action_metadata,
        )

    def drain_maintenance(
        self,
        request: CompactOwnerMaintenanceDrainRequestV1,
    ) -> CompactOwnerMaintenanceDrainResultV1:
        """Drain deferred replay append, learner train, and search refresh work."""

        self._raise_if_maintenance_failed()
        request = _validate_owner_maintenance_drain_request(request)
        started = time.perf_counter()
        drained_count = 0
        replay_append_entry_count = 0
        replay_append_transport_entry_count = 0
        replay_append_transition_batch_count = 0
        replay_append_transition_batch_entry_count = 0
        replay_append_count = 0
        learner_updates_total = 0
        owner_ref: dict[str, Any] | None = None
        owner_sample_telemetry: dict[str, Any] = {}
        owner_learner_telemetry: dict[str, Any] = {}
        owner_action_feedback: dict[str, Any] = _empty_owner_action_feedback_v1()
        owner_learner_timing_totals: dict[str, float] = {}
        owner_learner_timing_count = 0
        root_resolve_sec = 0.0
        search_sec = 0.0
        replay_sec = 0.0
        train_sec = 0.0
        refresh_sec = 0.0

        def merge_learner_telemetry(value: Any) -> None:
            nonlocal owner_learner_telemetry
            nonlocal owner_learner_timing_count
            if not isinstance(value, Mapping) or not value:
                return
            owner_learner_telemetry = dict(value)
            owner_learner_timing_count = self._merge_owner_learner_timing_telemetry(
                telemetry=value,
                totals=owner_learner_timing_totals,
                current_count=owner_learner_timing_count,
            )

        def merge_async_poll(value: Mapping[str, Any]) -> None:
            nonlocal owner_ref
            nonlocal owner_sample_telemetry
            nonlocal train_sec
            nonlocal refresh_sec
            nonlocal learner_updates_total
            if value.get("owner_ref") is not None:
                owner_ref = dict(value["owner_ref"])
            sample_telemetry = value.get("owner_sample_telemetry")
            if isinstance(sample_telemetry, Mapping) and sample_telemetry:
                owner_sample_telemetry = dict(sample_telemetry)
            merge_learner_telemetry(value.get("owner_learner_telemetry"))
            train_sec += float(value.get("train_sec") or 0.0)
            refresh_sec += float(value.get("refresh_sec") or 0.0)
            learner_updates_total += int(value.get("learner_updates") or 0)

        try:
            merge_async_poll(self._poll_owner_learner_jobs(wait=False))
            with self._state_lock:
                max_items = int(request.max_items)
                target_count = (
                    len(self._pending_maintenance)
                    if max_items <= 0
                    else min(max_items, len(self._pending_maintenance))
                )
                pending_learner_count = len(self._pending_learner_jobs)
                if pending_learner_count > 0 and max_items > 0:
                    target_count = 0
                if target_count <= 0 and bool(request.fail_if_empty) and pending_learner_count <= 0:
                    raise RuntimeError(
                        "owner-search maintenance drain requested with no pending work"
                    )
            if pending_learner_count > 0 and max_items <= 0:
                merge_async_poll(self._poll_owner_learner_jobs(wait=True))
                with self._state_lock:
                    target_count = (
                        len(self._pending_maintenance)
                        if max_items <= 0
                        else min(max_items, len(self._pending_maintenance))
                    )
            for _ in range(target_count):
                with self._state_lock:
                    if not self._pending_maintenance:
                        break
                    work = self._pending_maintenance.pop(0)
                replay_append_entries = tuple(work.request.replay_append_entries)
                replay_append_entry_count += _owner_replay_append_entry_count_v1(
                    replay_append_entries
                )
                replay_append_transport_entry_count += (
                    _owner_replay_append_transport_entry_count_v1(replay_append_entries)
                )
                replay_append_transition_batch_count += (
                    _owner_replay_append_transition_batch_count_v1(replay_append_entries)
                )
                replay_append_transition_batch_entry_count += (
                    _owner_replay_append_transition_batch_entry_count_total_v1(
                        replay_append_entries
                    )
                )
                replay_started = time.perf_counter()
                append_count, step_action_feedback = self._append_replay(
                    request=work.request,
                    root_batch=work.root_batch,
                    search_result=work.search_result,
                    root_batch_cache=work.root_batch_cache,
                )
                owner_action_feedback = _merge_owner_action_feedback_v1(
                    owner_action_feedback,
                    step_action_feedback,
                )
                replay_sec += _elapsed(replay_started)
                train_started = time.perf_counter()
                if self.async_learner_worker_enabled and int(work.request.train_steps) > 0:
                    while True:
                        with self._state_lock:
                            pending_learner_jobs = len(self._pending_learner_jobs)
                        if pending_learner_jobs < int(self.async_learner_max_pending):
                            break
                        merge_async_poll(self._poll_owner_learner_jobs(wait=True))
                    self._submit_owner_learner_job(
                        request=work.request,
                        root_batch=work.root_batch,
                        search_result=work.search_result,
                    )
                    train_sec += _elapsed(train_started)
                    learner_updates = 0
                else:
                    (
                        step_owner_ref,
                        learner_updates,
                        step_sample_telemetry,
                        step_learner_telemetry,
                    ) = self._train_and_publish(
                        request=work.request,
                        root_batch=work.root_batch,
                        search_result=work.search_result,
                    )
                    train_sec += _elapsed(train_started)
                    public_search_state: dict[str, Any] | None = None
                    if step_owner_ref is not None:
                        refresh_started = time.perf_counter()
                        with self._search_lock:
                            search_state = self._refresh_search_from_owner_ref(
                                request=work.request,
                                owner_ref=step_owner_ref,
                                learner_update_count=learner_updates,
                            )
                        public_search_state = _public_owner_search_worker_state_v1(search_state)
                        refresh_sec += _elapsed(refresh_started)
                    with self._state_lock:
                        self.learner_update_count += int(learner_updates)
                        if step_owner_ref is not None:
                            owner_ref = dict(step_owner_ref)
                            self.last_model_owner_ref = dict(step_owner_ref)
                        if public_search_state is not None:
                            self.last_search_worker_state = dict(public_search_state)
                    if step_sample_telemetry:
                        owner_sample_telemetry = dict(step_sample_telemetry)
                    merge_learner_telemetry(step_learner_telemetry)
                with self._state_lock:
                    self.replay_append_count += int(append_count)
                replay_append_count += int(append_count)
                learner_updates_total += int(learner_updates)
                drained_count += 1
            merge_async_poll(self._poll_owner_learner_jobs(wait=False))
        except Exception as exc:
            with self._state_lock:
                self._maintenance_failed_message = f"owner-search maintenance failed closed: {exc}"
            raise RuntimeError(self._maintenance_failed_message) from exc
        with self._state_lock:
            refresh_update_count = _search_refresh_update_count(self.last_search_worker_state)
            learner_update_count = int(self.learner_update_count)
            pending_count = len(self._pending_maintenance) + len(self._pending_learner_jobs)
        timing = {
            "root_resolve_sec": float(root_resolve_sec),
            "search_sec": float(search_sec),
            "replay_append_sec": float(replay_sec),
            "learner_train_sec": float(train_sec),
            "search_refresh_sec": float(refresh_sec),
            "wall_sec": float(_elapsed(started)),
        }
        if owner_learner_timing_totals:
            owner_learner_telemetry = dict(owner_learner_telemetry)
            owner_learner_telemetry.update(
                {key: float(value) for key, value in owner_learner_timing_totals.items()}
            )
            owner_learner_telemetry["compact_owner_search_owner_train_timing_aggregate_count"] = (
                int(owner_learner_timing_count)
            )
        owner_sample_telemetry = self._owner_sample_telemetry_with_replay_append(
            owner_sample_telemetry
        )
        return CompactOwnerMaintenanceDrainResultV1(
            schema_id=COMPACT_OWNER_MAINTENANCE_DRAIN_RESULT_SCHEMA_ID,
            owner_kind=self.owner_kind,
            drain_id=int(request.drain_id),
            owner_pid=os.getpid(),
            drained_count=int(drained_count),
            drained_work_item_count=int(drained_count),
            drained_replay_append_entry_count=int(replay_append_entry_count),
            drained_replay_append_count=int(replay_append_count),
            pending_count=int(pending_count),
            replay_append_entry_count=int(replay_append_entry_count),
            replay_append_count=int(replay_append_count),
            learner_update_count=int(learner_update_count),
            model_owner_ref_returned=owner_ref is not None,
            model_owner_ref_digest=_owner_ref_digest(owner_ref),
            search_refresh_update_count=int(refresh_update_count),
            search_consumed_learner_update=(int(refresh_update_count) == int(learner_update_count)),
            owner_sample_telemetry=dict(owner_sample_telemetry),
            owner_learner_telemetry=dict(owner_learner_telemetry),
            owner_action_feedback=dict(owner_action_feedback),
            timing=timing,
            owner_async_learner_telemetry=self._owner_async_learner_metadata(),
            drained_replay_append_transport_entry_count=int(replay_append_transport_entry_count),
            replay_append_transport_entry_count=int(replay_append_transport_entry_count),
            drained_replay_append_transition_batch_count=int(replay_append_transition_batch_count),
            replay_append_transition_batch_count=int(replay_append_transition_batch_count),
            drained_replay_append_transition_batch_entry_count=int(
                replay_append_transition_batch_entry_count
            ),
            replay_append_transition_batch_entry_count=int(
                replay_append_transition_batch_entry_count
            ),
        )

    def _raise_if_maintenance_failed(self) -> None:
        if self._maintenance_failed_message:
            raise RuntimeError(self._maintenance_failed_message)

    def _append_replay(
        self,
        *,
        request: CompactOwnerSearchRequestV1,
        root_batch: CompactRootBatchV1,
        search_result: CompactSearchResultV1,
        root_batch_cache: Mapping[int, CompactRootBatchV1] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        entries = tuple(request.replay_append_entries)
        if not entries:
            return 0, _empty_owner_action_feedback_v1()
        if self.replay_store is None:
            raise RuntimeError("owner-search replay append requested without owner replay_store")
        direct_append = getattr(
            self.replay_store,
            "append_owner_search_transition_batches",
            None,
        )
        if callable(direct_append) and all(
            _owner_replay_append_is_transition_batch_v1(entry) for entry in entries
        ):
            kwargs: dict[str, Any] = {
                "replay_append_transition_batches": entries,
                "root_batch": root_batch,
                "search_result": search_result,
                "request": request,
            }
            if _callable_accepts_keyword(direct_append, "root_batch_cache"):
                kwargs["root_batch_cache"] = dict(
                    self._root_batch_cache if root_batch_cache is None else root_batch_cache
                )
            if _callable_accepts_keyword(direct_append, "search_result_cache"):
                kwargs["search_result_cache"] = dict(self._search_result_cache)
            if _callable_accepts_keyword(direct_append, "flush_device_replay_payload"):

                def flush_device_replay_payload(handle: str) -> Any:
                    flush_device = getattr(
                        self.search_service,
                        "flush_device_replay_payload",
                        None,
                    )
                    if not callable(flush_device):
                        raise RuntimeError(
                            "owner-search direct transition-batch append requires "
                            "flush_device_replay_payload"
                        )
                    with self._search_lock:
                        replay_payload = flush_device(str(handle))
                    self._record_inner_device_replay_payload_telemetry(replay_payload)
                    return replay_payload

                kwargs["flush_device_replay_payload"] = flush_device_replay_payload
            result = direct_append(**kwargs)
            if isinstance(result, Mapping):
                self._record_owner_replay_append_telemetry(result)
                appended = int(
                    result.get(
                        "appended_count",
                        result.get("replay_append_count", 0),
                    )
                    or 0
                )
                cache_handles_to_evict = tuple(
                    str(handle)
                    for handle in tuple(result.get("cache_handles_to_evict", ()))
                    if str(handle).strip()
                )
                feedback_value = result.get("owner_action_feedback")
                owner_action_feedback = (
                    dict(feedback_value)
                    if isinstance(feedback_value, Mapping)
                    else _empty_owner_action_feedback_v1()
                )
            else:
                appended = int(result or 0)
                cache_handles_to_evict = ()
                owner_action_feedback = _empty_owner_action_feedback_v1()
            for handle in cache_handles_to_evict:
                self._search_result_cache.pop(str(handle), None)
            return int(appended), owner_action_feedback
        append = getattr(self.replay_store, "append_owner_search_replay", None)
        if not callable(append):
            raise RuntimeError("owner-search replay_store must expose append_owner_search_replay")
        materialized_entries, cache_handles_to_evict, owner_action_feedback = (
            self._materialize_owner_transition_entries(entries)
        )
        kwargs: dict[str, Any] = {
            "replay_append_entries": materialized_entries,
            "root_batch": root_batch,
            "search_result": search_result,
            "request": request,
        }
        if _callable_accepts_keyword(append, "root_batch_cache"):
            kwargs["root_batch_cache"] = dict(
                self._root_batch_cache if root_batch_cache is None else root_batch_cache
            )
        result = append(**kwargs)
        for handle in cache_handles_to_evict:
            self._search_result_cache.pop(str(handle), None)
        if result is None:
            return len(materialized_entries), owner_action_feedback
        return int(result), owner_action_feedback

    def _materialize_owner_transition_entries(
        self,
        entries: tuple[Any, ...],
    ) -> tuple[tuple[Any, ...], tuple[str, ...], dict[str, Any]]:
        materialized: list[Any] = []
        cache_handles_to_evict: list[str] = []
        owner_action_feedback = _empty_owner_action_feedback_v1()
        expanded_entries: list[Any] = []
        for entry in entries:
            expanded_entries.extend(_owner_replay_append_materialization_entries_v1(entry))
        for entry in expanded_entries:
            metadata = dict(getattr(entry, "metadata", {}) or {})
            if not bool(metadata.get("compact_owner_search_replay_append_transition_only")):
                materialized.append(entry)
                continue
            handle = str(getattr(entry, "replay_payload_handle", "")).strip()
            if not handle:
                raise RuntimeError("owner-search transition replay entry is missing handle")
            cached = self._search_result_cache.get(handle)
            if cached is None:
                raise RuntimeError(
                    f"owner-search transition replay handle is missing or stale: {handle}"
                )
            if int(cached.record_index) != int(getattr(entry, "record_index")):
                raise RuntimeError("owner-search transition replay handle record mismatch")
            previous_root_batch = cached.root_batch
            previous_search_result = cached.search_result
            selected_digest = str(metadata.get("selected_action_digest") or "")
            if selected_digest and selected_digest != compact_search_array_digest_v1(
                previous_search_result.selected_action
            ):
                raise RuntimeError("owner-search transition selected-action digest mismatch")
            next_joint_action = np.asarray(entry.next_joint_action, dtype=np.int16)
            env_row = np.asarray(previous_search_result.env_row, dtype=np.int64).reshape(-1)
            player = np.asarray(previous_search_result.player, dtype=np.int64).reshape(-1)
            selected = np.asarray(previous_search_result.selected_action, dtype=np.int16).reshape(
                -1
            )
            applied = np.asarray(next_joint_action[env_row, player], dtype=np.int16).reshape(-1)
            mismatch_count = int(np.count_nonzero(applied != selected))
            if selected.size and mismatch_count:
                raise RuntimeError("owner-search transition action facts do not match search")
            expected_checksum = _owner_action_checksum_v1(selected)
            applied_checksum = _owner_action_checksum_v1(applied)
            owner_action_feedback = _merge_owner_action_feedback_v1(
                owner_action_feedback,
                {
                    "compact_owner_search_action_feedback_transition_count": 1,
                    "compact_owner_search_action_feedback_action_count": int(selected.size),
                    "compact_owner_search_action_feedback_mismatch_count": int(mismatch_count),
                    "compact_owner_search_expected_joint_action_checksum": int(expected_checksum),
                    "compact_owner_search_applied_joint_action_checksum": int(applied_checksum),
                    "compact_owner_search_replay_action_checksum": int(expected_checksum),
                },
            )
            previous_batch = _compact_batch_from_root_batch(previous_root_batch)
            row_metadata = {
                "compact_owner_search_owner_materialized_replay_rows": True,
                **metadata,
            }
            if (
                str(previous_root_batch.observation_source)
                == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
            ):
                row_metadata.update(
                    {
                        "compact_owner_search_owner_materialized_replay_resident_observation": True,
                        "compact_owner_search_owner_materialized_replay_host_observation_copy": False,
                    }
                )
                if cached.action_step is not None:
                    action_step = cached.action_step
                    replay_payload = cached.inner_device_replay_payload
                    if replay_payload is None:
                        flush_device = getattr(
                            self.search_service,
                            "flush_device_replay_payload",
                            None,
                        )
                        if not callable(flush_device):
                            raise RuntimeError(
                                "owner-search cached inner action step requires "
                                "flush_device_replay_payload"
                            )
                        inner_handle = str(cached.inner_replay_payload_handle) or str(
                            action_step.replay_payload_handle
                        )
                        with self._search_lock:
                            replay_payload = flush_device(inner_handle)
                        self._record_inner_device_replay_payload_telemetry(replay_payload)
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
                            "owner-search action-only compatibility result cannot "
                            "materialize replay without an inner action step"
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
                        device=_resident_root_device(previous_root_batch),
                    )
                index_rows = build_compact_device_replay_index_rows_v1_from_payload(
                    previous_batch,
                    previous_root_batch,
                    action_step,
                    replay_payload,
                    record_index=int(entry.record_index),
                    next_joint_action=next_joint_action,
                    next_reward=np.asarray(entry.next_reward, dtype=np.float32),
                    next_done=np.asarray(entry.next_done, dtype=np.bool_),
                    next_terminated=np.asarray(entry.next_terminated, dtype=np.bool_),
                    next_truncated=np.asarray(entry.next_truncated, dtype=np.bool_),
                    next_final_reward_map=np.asarray(
                        entry.next_final_reward_map,
                        dtype=np.float32,
                    ),
                    next_final_observation_row_mask=np.asarray(
                        entry.next_final_observation_row_mask,
                        dtype=np.bool_,
                    ),
                    policy_source=str(entry.policy_source),
                    metadata={
                        "compact_owner_search_owner_materialized_device_replay_rows": True,
                        **row_metadata,
                    },
                )
            else:
                index_rows = build_compact_replay_index_rows_v1_from_search_result(
                    previous_batch,
                    previous_root_batch,
                    previous_search_result,
                    record_index=int(entry.record_index),
                    next_joint_action=next_joint_action,
                    next_reward=np.asarray(entry.next_reward, dtype=np.float32),
                    next_done=np.asarray(entry.next_done, dtype=np.bool_),
                    next_terminated=np.asarray(entry.next_terminated, dtype=np.bool_),
                    next_truncated=np.asarray(entry.next_truncated, dtype=np.bool_),
                    next_final_reward_map=np.asarray(
                        entry.next_final_reward_map,
                        dtype=np.float32,
                    ),
                    next_final_observation_row_mask=np.asarray(
                        entry.next_final_observation_row_mask,
                        dtype=np.bool_,
                    ),
                    policy_source=str(entry.policy_source),
                    metadata=row_metadata,
                )
            materialized.append(
                CompactOwnerSearchReplayAppendIndexEntryV1(
                    schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_INDEX_ENTRY_SCHEMA_ID,
                    record_index=int(entry.record_index),
                    next_record_index=int(entry.next_record_index),
                    index_rows=index_rows,
                    metadata={
                        **metadata,
                        "compact_owner_search_replay_append_index_only": True,
                        "compact_owner_search_replay_append_carries_compact_batches": False,
                        "compact_owner_search_replay_append_transition_materialized": True,
                    },
                )
            )
            cache_handles_to_evict.append(handle)
        return tuple(materialized), tuple(cache_handles_to_evict), owner_action_feedback

    def _train_and_publish(
        self,
        *,
        request: CompactOwnerSearchRequestV1,
        root_batch: CompactRootBatchV1,
        search_result: CompactSearchResultV1,
    ) -> tuple[dict[str, Any] | None, int, dict[str, Any], dict[str, Any]]:
        train_steps = int(request.train_steps)
        if train_steps <= 0:
            return None, 0, {}, {}
        if self.learner is None:
            raise RuntimeError("owner-search train requested without owner learner")
        train = getattr(self.learner, "train_owner_search_step", None)
        if not callable(train):
            raise RuntimeError("owner-search learner must expose train_owner_search_step")
        result = train(
            replay_store=self.replay_store,
            root_batch=root_batch,
            search_result=search_result,
            sample_batch_size=int(request.sample_batch_size),
            train_steps=train_steps,
            request=request,
        )
        return _owner_learner_result_tuple_from_mapping(
            result,
            default_train_steps=train_steps,
            require_owner_ref=bool(request.refresh_model),
        )

    @staticmethod
    def _merge_owner_learner_timing_telemetry(
        *,
        telemetry: Mapping[str, Any],
        totals: dict[str, float],
        current_count: int,
    ) -> int:
        timing_seen = False
        for key in _OWNER_SEARCH_LEARNER_TIMING_TELEMETRY_KEYS:
            if key in telemetry:
                timing_seen = True
                totals[key] = float(totals.get(key, 0.0)) + float(telemetry.get(key) or 0.0)
        timing_count = int(
            telemetry.get("compact_owner_search_owner_train_timing_aggregate_count") or 0
        )
        if timing_seen and timing_count <= 0:
            timing_count = 1
        return int(current_count) + max(0, int(timing_count))

    def _submit_owner_learner_job(
        self,
        *,
        request: CompactOwnerSearchRequestV1,
        root_batch: CompactRootBatchV1,
        search_result: CompactSearchResultV1,
    ) -> None:
        worker = self._learner_worker
        if worker is None:
            raise RuntimeError("async owner-search learner requested without worker")
        with self._state_lock:
            pending_count = len(self._pending_learner_jobs)
        if pending_count >= int(self.async_learner_max_pending):
            raise RuntimeError("async owner-search learner queue is full")
        submit = getattr(worker, "submit", None)
        if not callable(submit):
            raise RuntimeError("async owner-search learner worker must expose submit")
        submit_kwargs: dict[str, Any] = {
            "request": request,
            "root_batch": root_batch,
            "search_result": search_result,
        }
        if _callable_accepts_keyword(submit, "replay_store"):
            submit_kwargs["replay_store"] = self.replay_store
        if _callable_accepts_keyword(submit, "learner"):
            submit_kwargs["learner"] = self.learner
        handle = submit(**submit_kwargs)
        with self._state_lock:
            self._pending_learner_jobs.append(
                _CompactOwnerPendingLearnerJobV1(
                    request=request,
                    handle=handle,
                    submitted_at=time.perf_counter(),
                )
            )
            self.owner_async_learner_submit_count += 1
            self.owner_async_learner_max_pending_observed = max(
                int(self.owner_async_learner_max_pending_observed),
                len(self._pending_learner_jobs),
            )

    def _owner_learner_job_done(self, handle: Any) -> bool:
        worker = self._learner_worker
        done = getattr(worker, "done", None)
        if callable(done):
            return bool(done(handle))
        handle_done = getattr(handle, "done", None)
        if callable(handle_done):
            return bool(handle_done())
        return False

    def _owner_learner_job_result(self, handle: Any) -> Any:
        worker = self._learner_worker
        result = getattr(worker, "result", None)
        if callable(result):
            return result(handle)
        handle_result = getattr(handle, "result", None)
        if callable(handle_result):
            return handle_result()
        raise RuntimeError("async owner-search learner handle cannot produce result")

    def _poll_owner_learner_jobs(self, *, wait: bool) -> dict[str, Any]:
        completed_count = 0
        learner_updates_total = 0
        train_sec = 0.0
        refresh_sec = 0.0
        wait_sec = 0.0
        owner_ref: dict[str, Any] | None = None
        owner_sample_telemetry: dict[str, Any] = {}
        owner_learner_telemetry: dict[str, Any] = {}
        owner_learner_timing_totals: dict[str, float] = {}
        owner_learner_timing_count = 0
        while True:
            with self._state_lock:
                job = self._pending_learner_jobs[0] if self._pending_learner_jobs else None
            if job is None:
                break
            job_done = self._owner_learner_job_done(job.handle)
            if not bool(wait) and not job_done:
                break
            wait_started = time.perf_counter()
            try:
                if bool(wait) and not job_done:
                    with self._state_lock:
                        self.owner_async_learner_wait_count += 1
                    result = self._owner_learner_job_result(job.handle)
                    elapsed_wait = _elapsed(wait_started)
                    wait_sec += elapsed_wait
                    with self._state_lock:
                        self.owner_async_learner_wait_sec += float(elapsed_wait)
                else:
                    result = self._owner_learner_job_result(job.handle)
            except Exception as exc:
                with self._state_lock:
                    self.owner_async_learner_failed = True
                    self._maintenance_failed_message = (
                        f"owner-search async learner failed closed: {exc}"
                    )
                raise RuntimeError(self._maintenance_failed_message) from exc
            with self._state_lock:
                if self._pending_learner_jobs and self._pending_learner_jobs[0] is job:
                    self._pending_learner_jobs.pop(0)
                elif job in self._pending_learner_jobs:
                    self._pending_learner_jobs.remove(job)
            if not isinstance(result, tuple) or len(result) != 4:
                raise RuntimeError("owner-search async learner returned invalid result")
            (
                step_owner_ref,
                learner_updates,
                step_sample_telemetry,
                step_learner_telemetry,
            ) = result
            if not isinstance(step_owner_ref, Mapping):
                if bool(job.request.refresh_model):
                    raise RuntimeError("owner-search async learner returned no owner ref")
                step_owner_ref = None
            public_search_state: dict[str, Any] | None = None
            if isinstance(step_owner_ref, Mapping):
                refresh_started = time.perf_counter()
                with self._search_lock:
                    search_state = self._refresh_search_from_owner_ref(
                        request=job.request,
                        owner_ref=step_owner_ref,
                        learner_update_count=int(learner_updates),
                    )
                public_search_state = _public_owner_search_worker_state_v1(search_state)
                refresh_sec += _elapsed(refresh_started)
            if public_search_state is not None and bool(
                public_search_state.get("model_state_snapshot_loaded", False)
            ):
                step_learner_telemetry = dict(step_learner_telemetry)
                step_learner_telemetry.update(
                    {
                        "compact_owner_search_model_state_snapshot_load_count": 1,
                        "compact_owner_search_model_state_snapshot_load_bytes": int(
                            public_search_state.get("model_state_snapshot_load_bytes") or 0
                        ),
                        "compact_owner_search_model_state_snapshot_load_sec": float(
                            public_search_state.get("model_state_snapshot_load_sec") or 0.0
                        ),
                    }
                )
            with self._state_lock:
                self.learner_update_count += int(learner_updates)
                self.owner_async_learner_completed_count += 1
                if isinstance(step_owner_ref, Mapping):
                    owner_ref = dict(step_owner_ref)
                    self.last_model_owner_ref = dict(step_owner_ref)
                if public_search_state is not None:
                    self.last_search_worker_state = dict(public_search_state)
            if isinstance(step_sample_telemetry, Mapping) and step_sample_telemetry:
                owner_sample_telemetry = dict(step_sample_telemetry)
            if isinstance(step_learner_telemetry, Mapping) and step_learner_telemetry:
                owner_learner_telemetry = dict(step_learner_telemetry)
                owner_learner_timing_count = self._merge_owner_learner_timing_telemetry(
                    telemetry=step_learner_telemetry,
                    totals=owner_learner_timing_totals,
                    current_count=owner_learner_timing_count,
                )
                train_sec += float(
                    step_learner_telemetry.get(
                        "compact_owner_search_owner_train_wall_sec",
                        0.0,
                    )
                    or 0.0
                )
            learner_updates_total += int(learner_updates)
            completed_count += 1
        if owner_learner_timing_totals:
            owner_learner_telemetry = dict(owner_learner_telemetry)
            owner_learner_telemetry.update(
                {key: float(value) for key, value in owner_learner_timing_totals.items()}
            )
            owner_learner_telemetry["compact_owner_search_owner_train_timing_aggregate_count"] = (
                int(owner_learner_timing_count)
            )
        return {
            "completed_count": int(completed_count),
            "learner_updates": int(learner_updates_total),
            "owner_ref": owner_ref,
            "owner_sample_telemetry": owner_sample_telemetry,
            "owner_learner_telemetry": owner_learner_telemetry,
            "train_sec": float(train_sec),
            "refresh_sec": float(refresh_sec),
            "wait_sec": float(wait_sec),
        }

    def _refresh_search_from_owner_ref(
        self,
        *,
        request: CompactOwnerSearchRequestV1,
        owner_ref: Mapping[str, Any] | None,
        learner_update_count: int,
    ) -> dict[str, Any] | None:
        if owner_ref is None:
            return None
        self._preflush_cached_inner_device_replay_payloads()
        refresh = getattr(self.search_service, "refresh_model_owner_ref", None)
        if not callable(refresh):
            raise RuntimeError(
                "owner-search requires owner-side search_service.refresh_model_owner_ref"
            )
        total_update_count = int(self.learner_update_count) + int(learner_update_count)
        digest_deferred = _owner_ref_digest_deferred_to_search_refresh(owner_ref)
        if digest_deferred and not isinstance(
            owner_ref.get("model_state_dict"),
            Mapping,
        ):
            raise RuntimeError(
                "owner-search deferred owner-ref digest requires same-process model state"
            )
        expected_digest = None if digest_deferred else _owner_ref_digest(owner_ref)
        state = refresh(
            owner_ref=dict(owner_ref),
            policy_version_ref=(
                str(request.policy_version_ref) or str(owner_ref.get("policy_version_ref") or "")
            ),
            model_version_ref=(
                str(request.model_version_ref) or str(owner_ref.get("model_version_ref") or "")
            ),
            policy_source=str(request.policy_source) or str(owner_ref.get("policy_source") or ""),
            learner_update_count=total_update_count,
            expected_model_state_digest=expected_digest,
        )
        if not isinstance(state, Mapping):
            raise RuntimeError("owner-search refresh returned non-mapping state")
        state_dict = dict(state)
        if _search_refresh_update_count(state_dict) != total_update_count:
            raise RuntimeError("owner-search refresh update count mismatch")
        refreshed_digest = str(state_dict.get("model_state_digest") or "").strip()
        if not refreshed_digest:
            raise RuntimeError("owner-search refresh missing model digest")
        if not digest_deferred and refreshed_digest != _owner_ref_digest(owner_ref):
            raise RuntimeError("owner-search refresh digest mismatch")
        if digest_deferred:
            digest_source = str(state_dict.get("model_state_digest_source") or "")
            if digest_source != "search_worker_after_load":
                raise RuntimeError(
                    "owner-search deferred owner-ref digest requires post-load search digest"
                )
            state_dict["owner_ref_model_state_digest_deferred_to_search_refresh"] = True
            state_dict["owner_ref_model_state_digest_token"] = _owner_ref_digest(owner_ref)
            state_dict["owner_ref_model_state_digest_source"] = str(
                owner_ref.get("model_state_digest_source") or ""
            )
        return state_dict

    def _preflush_cached_inner_device_replay_payloads(self) -> None:
        flush_device = getattr(self.search_service, "flush_device_replay_payload", None)
        if not callable(flush_device):
            return
        updated: dict[str, _CompactOwnerCachedSearchV1] = {}
        for handle, cached in tuple(self._search_result_cache.items()):
            if cached.action_step is None or cached.inner_device_replay_payload is not None:
                continue
            inner_handle = str(cached.inner_replay_payload_handle) or str(
                cached.action_step.replay_payload_handle
            )
            if not inner_handle:
                continue
            with self._search_lock:
                replay_payload = flush_device(inner_handle)
            self._record_inner_device_replay_payload_telemetry(replay_payload)
            validate_compact_device_search_two_phase_payload_v1(
                cached.action_step,
                replay_payload,
            )
            updated[str(handle)] = replace(
                cached,
                inner_device_replay_payload=replay_payload,
            )
        if updated:
            self._search_result_cache.update(updated)


class CompactProcessOwnerSearchWorkerV1:
    """Factory-backed process worker that owns search, replay, learner, and publish state."""

    def __init__(
        self,
        *,
        root_provider_factory: Any,
        search_service_factory: Any,
        replay_store_factory: Any | None = None,
        learner_factory: Any | None = None,
        root_provider_factory_kwargs: Mapping[str, Any] | None = None,
        search_service_factory_kwargs: Mapping[str, Any] | None = None,
        replay_store_factory_kwargs: Mapping[str, Any] | None = None,
        learner_factory_kwargs: Mapping[str, Any] | None = None,
        use_inner_two_phase_device_replay: bool = False,
        async_learner_worker: bool = False,
        async_learner_worker_kind: str = (
            COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
        ),
        async_learner_max_pending: int = 1,
        max_workers: int = 1,
    ) -> None:
        if not callable(root_provider_factory):
            raise ValueError("root_provider_factory must be callable")
        if not callable(search_service_factory):
            raise ValueError("search_service_factory must be callable")
        self._parent_pid = os.getpid()
        self._root_provider_factory = root_provider_factory
        self._search_service_factory = search_service_factory
        self._replay_store_factory = replay_store_factory
        self._learner_factory = learner_factory
        self._root_provider_factory_kwargs = dict(root_provider_factory_kwargs or {})
        self._search_service_factory_kwargs = dict(search_service_factory_kwargs or {})
        self._replay_store_factory_kwargs = dict(replay_store_factory_kwargs or {})
        self._learner_factory_kwargs = dict(learner_factory_kwargs or {})
        self._use_inner_two_phase_device_replay = bool(use_inner_two_phase_device_replay)
        self._async_learner_worker = bool(async_learner_worker)
        self._async_learner_worker_kind = str(async_learner_worker_kind)
        self._async_learner_max_pending = int(async_learner_max_pending)
        self._max_workers = int(max_workers)
        self._mp_context = mp.get_context("spawn")
        self._process: Any | None = None
        self._command_queue: Any | None = None
        self._result_queue: Any | None = None
        self._collector_thread: threading.Thread | None = None
        self._futures: dict[int, Future[dict[str, Any]]] = {}
        self._futures_lock = threading.Lock()
        self._next_command_id = 0
        self._closed = False
        self._action_request_count = 0
        self._maintenance_request_count = 0
        self._run_request_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        parent_pid = int(self._parent_pid)
        return {
            "compact_owner_search_worker_kind": COMPACT_OWNER_SEARCH_KIND_LOCAL_PROCESS,
            "compact_owner_search_worker_resource_scope": "persistent_process",
            "compact_owner_search_worker_parent_pid": parent_pid,
            "compact_owner_search_actor_search_pid": parent_pid,
            "compact_owner_search_worker_resource_id": (
                f"local_process_priority_loop:{parent_pid}:compact-owner-search"
            ),
            "compact_owner_search_actor_resource_id": (f"local_process:{parent_pid}:actor-game"),
            "compact_owner_search_worker_start_method": str(self._mp_context.get_start_method()),
            "compact_owner_search_owner_loop_schema_id": (
                COMPACT_OWNER_SEARCH_PRIORITY_LOOP_SCHEMA_ID
            ),
            "compact_owner_search_owner_loop_kind": COMPACT_OWNER_SEARCH_PRIORITY_LOOP_KIND,
            "compact_owner_search_owner_loop_persistent": True,
            "compact_owner_search_owner_action_priority_enabled": True,
            "compact_owner_search_owner_action_request_count": int(self._action_request_count),
            "compact_owner_search_owner_maintenance_request_count": int(
                self._maintenance_request_count
            ),
            "compact_owner_search_owner_run_request_count": int(self._run_request_count),
            "compact_owner_search_worker_owns_search_state": True,
            "compact_owner_search_worker_owns_replay_state": callable(self._replay_store_factory),
            "compact_owner_search_worker_owns_model_state": callable(self._learner_factory),
            "compact_owner_search_use_inner_two_phase_device_replay": bool(
                self._use_inner_two_phase_device_replay
            ),
            "compact_owner_search_owner_async_learner_worker_enabled": bool(
                self._async_learner_worker
            ),
            "compact_owner_search_owner_async_learner_worker_kind": (
                self._async_learner_worker_kind if self._async_learner_worker else "none"
            ),
            "compact_owner_search_owner_async_learner_max_pending": int(
                self._async_learner_max_pending
            ),
            "compact_owner_search_worker_resource_distinct_from_actor": True,
            "compact_owner_search_worker_hardware_resource_distinct_from_actor": False,
        }

    def prepare(self) -> None:
        if self._process is not None:
            return
        if int(self._max_workers) != 1:
            raise RuntimeError("owner-search process worker requires max_workers=1")
        factory_payload = (
            self._root_provider_factory,
            self._search_service_factory,
            self._replay_store_factory,
            self._learner_factory,
            self._root_provider_factory_kwargs,
            self._search_service_factory_kwargs,
            self._replay_store_factory_kwargs,
            self._learner_factory_kwargs,
            self._use_inner_two_phase_device_replay,
            self._async_learner_worker,
            self._async_learner_worker_kind,
            self._async_learner_max_pending,
        )
        if _contains_cuda_tensor(factory_payload):
            raise RuntimeError("owner-search process factories must be host-only")
        self._command_queue = self._mp_context.Queue()
        self._result_queue = self._mp_context.Queue()
        self._process = self._mp_context.Process(
            target=_compact_process_owner_search_priority_loop,
            args=(
                factory_payload,
                self._command_queue,
                self._result_queue,
            ),
            name="compact-owner-search-priority-loop",
        )
        self._process.start()
        self._collector_thread = threading.Thread(
            target=self._collect_priority_loop_results,
            name="compact-owner-search-result-collector",
            daemon=True,
        )
        self._collector_thread.start()

    def submit(self, request: CompactOwnerSearchRequestV1) -> Future[dict[str, Any]]:
        if _contains_cuda_tensor(request):
            raise RuntimeError("owner-search process request must be host-only")
        self._run_request_count += 1
        return self._submit_priority_command(_OWNER_PRIORITY_COMMAND_RUN, request)

    def submit_action(
        self,
        request: CompactOwnerSearchRequestV1,
    ) -> Future[dict[str, Any]]:
        if _contains_cuda_tensor(request):
            raise RuntimeError("owner-search process action request must be host-only")
        self._action_request_count += 1
        return self._submit_priority_command(_OWNER_PRIORITY_COMMAND_ACTION, request)

    def submit_maintenance_drain(
        self,
        request: CompactOwnerMaintenanceDrainRequestV1,
    ) -> Future[dict[str, Any]]:
        if _contains_cuda_tensor(request):
            raise RuntimeError("owner-search maintenance drain request must be host-only")
        self._maintenance_request_count += 1
        return self._submit_priority_command(
            _OWNER_PRIORITY_COMMAND_DRAIN_MAINTENANCE,
            request,
        )

    def _submit_priority_command(
        self,
        command_kind: str,
        payload: Any,
    ) -> Future[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("owner-search process worker is closed")
        self.prepare()
        assert self._command_queue is not None
        future: Future[dict[str, Any]] = Future()
        with self._futures_lock:
            self._next_command_id += 1
            command_id = int(self._next_command_id)
            self._futures[command_id] = future
        command = _CompactOwnerPriorityCommandV1(
            command_id=command_id,
            command_kind=str(command_kind),
            payload=payload,
        )
        self._command_queue.put(command)
        return future

    def _collect_priority_loop_results(self) -> None:
        result_queue = self._result_queue
        if result_queue is None:
            return
        while True:
            try:
                result = result_queue.get()
            except (EOFError, OSError):
                break
            if result is None:
                break
            if not isinstance(result, _CompactOwnerPriorityResultV1):
                continue
            with self._futures_lock:
                future = self._futures.pop(int(result.command_id), None)
            if future is None:
                continue
            if bool(result.ok):
                future.set_result(dict(result.payload or {}))
            else:
                message = str(result.error_message or "owner-search priority loop failed")
                future.set_exception(RuntimeError(message))

    def done(self, handle: Future[dict[str, Any]]) -> bool:
        self._fail_futures_if_process_dead()
        return bool(handle.done())

    def result(self, handle: Future[dict[str, Any]]) -> dict[str, Any]:
        while True:
            try:
                return dict(handle.result(timeout=0.1))
            except FutureTimeoutError:
                self._fail_futures_if_process_dead()
                continue

    def close(self) -> None:
        if self._closed:
            return
        process = self._process
        command_queue = self._command_queue
        close_future: Future[dict[str, Any]] | None = None
        if process is not None and command_queue is not None and process.is_alive():
            try:
                close_future = self._submit_priority_command(
                    _OWNER_PRIORITY_COMMAND_CLOSE,
                    None,
                )
            except RuntimeError:
                close_future = None
        self._closed = True
        if close_future is not None:
            try:
                close_future.result(timeout=10.0)
            except Exception:
                pass
        if process is not None:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        result_queue = self._result_queue
        if result_queue is not None:
            try:
                result_queue.put(None)
            except (EOFError, OSError):
                pass
        collector = self._collector_thread
        if collector is not None:
            collector.join(timeout=2.0)
        with self._futures_lock:
            remaining = list(self._futures.values())
            self._futures.clear()
        for future in remaining:
            if not future.done():
                future.set_exception(RuntimeError("owner-search process worker closed"))
        for mp_queue in (self._command_queue, self._result_queue):
            if mp_queue is None:
                continue
            close = getattr(mp_queue, "close", None)
            if callable(close):
                close()
        self._process = None
        self._command_queue = None
        self._result_queue = None
        self._collector_thread = None

    def _fail_futures_if_process_dead(self) -> None:
        process = self._process
        if process is None or process.is_alive() or process.exitcode is None:
            return
        message = f"owner-search priority loop process exited with code {int(process.exitcode)}"
        with self._futures_lock:
            remaining = list(self._futures.values())
            self._futures.clear()
        for future in remaining:
            if not future.done():
                future.set_exception(RuntimeError(message))


class CompactInlineOwnerSearchWorkerV1:
    """Same owner service contract as the process worker, without IPC."""

    def __init__(
        self,
        *,
        root_provider: Any,
        search_service: Any,
        replay_store: Any | None = None,
        learner: Any | None = None,
        use_inner_two_phase_device_replay: bool = False,
        async_learner_worker: bool = False,
        async_learner_worker_kind: str = (
            COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
        ),
        async_learner_max_pending: int = 1,
        action_result_slot_table: CompactOwnerActionResultSlotTableV1 | None = None,
    ) -> None:
        self._parent_pid = os.getpid()
        self._service = CompactOwnerSearchServiceV1(
            root_provider=root_provider,
            search_service=search_service,
            replay_store=replay_store,
            learner=learner,
            owner_kind=COMPACT_OWNER_SEARCH_KIND_INLINE,
            use_inner_two_phase_device_replay=bool(use_inner_two_phase_device_replay),
            async_learner_worker=bool(async_learner_worker),
            async_learner_worker_kind=str(async_learner_worker_kind),
            async_learner_max_pending=int(async_learner_max_pending),
            action_result_slot_table=action_result_slot_table,
        )
        self._closed = False
        self._action_request_count = 0
        self._maintenance_request_count = 0
        self._run_request_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        parent_pid = int(self._parent_pid)
        service_metadata = dict(self._service.metadata)
        return {
            **service_metadata,
            "compact_owner_search_worker_kind": COMPACT_OWNER_SEARCH_KIND_INLINE,
            "compact_owner_search_worker_resource_scope": "inline_process",
            "compact_owner_search_worker_parent_pid": parent_pid,
            "compact_owner_search_actor_search_pid": parent_pid,
            "compact_owner_search_worker_resource_id": (
                f"inline_owner:{parent_pid}:compact-owner-search"
            ),
            "compact_owner_search_actor_resource_id": (f"inline_owner:{parent_pid}:actor-game"),
            "compact_owner_search_worker_start_method": "inline",
            "compact_owner_search_owner_loop_schema_id": (
                COMPACT_OWNER_SEARCH_PRIORITY_LOOP_SCHEMA_ID
            ),
            "compact_owner_search_owner_loop_kind": "inline_priority_owner_loop_v1",
            "compact_owner_search_owner_loop_persistent": True,
            "compact_owner_search_owner_action_priority_enabled": True,
            "compact_owner_search_owner_action_request_count": int(self._action_request_count),
            "compact_owner_search_owner_maintenance_request_count": int(
                self._maintenance_request_count
            ),
            "compact_owner_search_owner_run_request_count": int(self._run_request_count),
            "compact_owner_search_worker_owns_search_state": True,
            "compact_owner_search_worker_owns_replay_state": bool(
                service_metadata.get("compact_owner_search_service_worker_owns_replay_state")
            ),
            "compact_owner_search_worker_owns_model_state": bool(
                service_metadata.get("compact_owner_search_service_worker_owns_model_state")
            ),
            "compact_owner_search_worker_resource_distinct_from_actor": False,
            "compact_owner_search_worker_hardware_resource_distinct_from_actor": False,
        }

    def submit(self, request: CompactOwnerSearchRequestV1) -> Future[dict[str, Any]]:
        self._run_request_count += 1
        return self._completed_future(self._service.run(request).to_dict())

    def submit_action(
        self,
        request: CompactOwnerSearchRequestV1,
    ) -> Future[dict[str, Any]]:
        self._action_request_count += 1
        result = self._service.run_action(request)
        return self._completed_future(
            self._service.action_result_payload_for_request(request, result)
        )

    def submit_maintenance_drain(
        self,
        request: CompactOwnerMaintenanceDrainRequestV1,
    ) -> Future[dict[str, Any]]:
        self._maintenance_request_count += 1
        return self._completed_future(self._service.drain_maintenance(request).to_dict())

    def done(self, handle: Future[dict[str, Any]]) -> bool:
        return bool(handle.done())

    def result(self, handle: Future[dict[str, Any]]) -> dict[str, Any]:
        return dict(handle.result())

    def close(self) -> None:
        self._service.close()
        self._closed = True

    @staticmethod
    def _completed_future(payload: dict[str, Any]) -> Future[dict[str, Any]]:
        future: Future[dict[str, Any]] = Future()
        future.set_result(dict(payload))
        return future


class CompactInlineBackgroundOwnerSearchWorkerV1:
    """Inline action path with owner maintenance drained on a background thread."""

    def __init__(
        self,
        *,
        root_provider: Any,
        search_service: Any,
        replay_store: Any | None = None,
        learner: Any | None = None,
        use_inner_two_phase_device_replay: bool = False,
        async_learner_worker: bool = False,
        async_learner_worker_kind: str = (
            COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
        ),
        async_learner_max_pending: int = 1,
        action_result_slot_table: CompactOwnerActionResultSlotTableV1 | None = None,
    ) -> None:
        self._parent_pid = os.getpid()
        self._service = CompactOwnerSearchServiceV1(
            root_provider=root_provider,
            search_service=search_service,
            replay_store=replay_store,
            learner=learner,
            owner_kind=COMPACT_OWNER_SEARCH_KIND_INLINE_BACKGROUND,
            use_inner_two_phase_device_replay=bool(use_inner_two_phase_device_replay),
            async_learner_worker=bool(async_learner_worker),
            async_learner_worker_kind=str(async_learner_worker_kind),
            async_learner_max_pending=int(async_learner_max_pending),
            action_result_slot_table=action_result_slot_table,
        )
        self._maintenance_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="compact-owner-search-inline-maintenance",
        )
        self._closed = False
        self._action_request_count = 0
        self._maintenance_request_count = 0
        self._run_request_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        parent_pid = int(self._parent_pid)
        service_metadata = dict(self._service.metadata)
        return {
            **service_metadata,
            "compact_owner_search_worker_kind": (COMPACT_OWNER_SEARCH_KIND_INLINE_BACKGROUND),
            "compact_owner_search_worker_resource_scope": (
                "inline_process_background_maintenance_thread"
            ),
            "compact_owner_search_worker_parent_pid": parent_pid,
            "compact_owner_search_actor_search_pid": parent_pid,
            "compact_owner_search_worker_resource_id": (
                f"inline_background_owner:{parent_pid}:compact-owner-search"
            ),
            "compact_owner_search_actor_resource_id": (
                f"inline_background_owner:{parent_pid}:actor-game"
            ),
            "compact_owner_search_worker_start_method": "inline_plus_threading",
            "compact_owner_search_owner_loop_schema_id": (
                COMPACT_OWNER_SEARCH_PRIORITY_LOOP_SCHEMA_ID
            ),
            "compact_owner_search_owner_loop_kind": (
                COMPACT_OWNER_SEARCH_INLINE_BACKGROUND_LOOP_KIND
            ),
            "compact_owner_search_owner_loop_persistent": True,
            "compact_owner_search_owner_action_priority_enabled": True,
            "compact_owner_search_owner_background_maintenance_thread": True,
            "compact_owner_search_owner_background_overlap_enabled": True,
            "compact_owner_search_owner_action_request_count": int(self._action_request_count),
            "compact_owner_search_owner_maintenance_request_count": int(
                self._maintenance_request_count
            ),
            "compact_owner_search_owner_run_request_count": int(self._run_request_count),
            "compact_owner_search_worker_owns_search_state": True,
            "compact_owner_search_worker_owns_replay_state": bool(
                service_metadata.get("compact_owner_search_service_worker_owns_replay_state")
            ),
            "compact_owner_search_worker_owns_model_state": bool(
                service_metadata.get("compact_owner_search_service_worker_owns_model_state")
            ),
            "compact_owner_search_worker_resource_distinct_from_actor": False,
            "compact_owner_search_worker_hardware_resource_distinct_from_actor": False,
            "compact_owner_search_maintenance_resource_distinct_from_actor": True,
        }

    def submit(self, request: CompactOwnerSearchRequestV1) -> Future[dict[str, Any]]:
        self._raise_if_closed()
        self._run_request_count += 1
        return self._completed_future(self._service.run(request).to_dict())

    def submit_action(
        self,
        request: CompactOwnerSearchRequestV1,
    ) -> Future[dict[str, Any]]:
        self._raise_if_closed()
        self._action_request_count += 1
        result = self._service.run_action(request)
        return self._completed_future(
            self._service.action_result_payload_for_request(request, result)
        )

    def submit_maintenance_drain(
        self,
        request: CompactOwnerMaintenanceDrainRequestV1,
    ) -> Future[dict[str, Any]]:
        self._raise_if_closed()
        self._maintenance_request_count += 1
        return self._maintenance_executor.submit(
            self._drain_maintenance_to_dict,
            request,
        )

    def _drain_maintenance_to_dict(
        self,
        request: CompactOwnerMaintenanceDrainRequestV1,
    ) -> dict[str, Any]:
        return self._service.drain_maintenance(request).to_dict()

    def done(self, handle: Future[dict[str, Any]]) -> bool:
        return bool(handle.done())

    def result(self, handle: Future[dict[str, Any]]) -> dict[str, Any]:
        return dict(handle.result())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._maintenance_executor.shutdown(wait=True, cancel_futures=False)
        self._service.close()

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("owner-search inline-background worker is closed")

    @staticmethod
    def _completed_future(payload: dict[str, Any]) -> Future[dict[str, Any]]:
        future: Future[dict[str, Any]] = Future()
        future.set_result(dict(payload))
        return future


class CompactThreadedOwnerSearchWorkerV1:
    """Colocated owner worker with direct roots and background maintenance."""

    def __init__(
        self,
        *,
        root_provider: Any,
        search_service: Any,
        replay_store: Any | None = None,
        learner: Any | None = None,
        use_inner_two_phase_device_replay: bool = False,
        async_learner_worker: bool = False,
        async_learner_worker_kind: str = (
            COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
        ),
        async_learner_max_pending: int = 1,
        action_result_slot_table: CompactOwnerActionResultSlotTableV1 | None = None,
    ) -> None:
        self._parent_pid = os.getpid()
        self._service = CompactOwnerSearchServiceV1(
            root_provider=root_provider,
            search_service=search_service,
            replay_store=replay_store,
            learner=learner,
            owner_kind=COMPACT_OWNER_SEARCH_KIND_THREADED,
            use_inner_two_phase_device_replay=bool(use_inner_two_phase_device_replay),
            async_learner_worker=bool(async_learner_worker),
            async_learner_worker_kind=str(async_learner_worker_kind),
            async_learner_max_pending=int(async_learner_max_pending),
            action_result_slot_table=action_result_slot_table,
        )
        self._command_queue: queue.Queue[_CompactOwnerPriorityCommandV1 | None] = queue.Queue()
        self._result_queue: queue.Queue[_CompactOwnerPriorityResultV1 | None] = queue.Queue()
        self._owner_thread: threading.Thread | None = None
        self._collector_thread: threading.Thread | None = None
        self._futures: dict[int, Future[dict[str, Any]]] = {}
        self._futures_lock = threading.Lock()
        self._next_command_id = 0
        self._closed = False
        self._action_request_count = 0
        self._maintenance_request_count = 0
        self._run_request_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        parent_pid = int(self._parent_pid)
        service_metadata = dict(self._service.metadata)
        return {
            **service_metadata,
            "compact_owner_search_worker_kind": COMPACT_OWNER_SEARCH_KIND_THREADED,
            "compact_owner_search_worker_resource_scope": "colocated_thread",
            "compact_owner_search_worker_parent_pid": parent_pid,
            "compact_owner_search_actor_search_pid": parent_pid,
            "compact_owner_search_worker_resource_id": (
                f"threaded_owner:{parent_pid}:compact-owner-search"
            ),
            "compact_owner_search_actor_resource_id": (f"threaded_owner:{parent_pid}:actor-game"),
            "compact_owner_search_worker_start_method": "threading",
            "compact_owner_search_owner_loop_schema_id": (
                COMPACT_OWNER_SEARCH_PRIORITY_LOOP_SCHEMA_ID
            ),
            "compact_owner_search_owner_loop_kind": (COMPACT_OWNER_SEARCH_THREADED_LOOP_KIND),
            "compact_owner_search_owner_loop_persistent": True,
            "compact_owner_search_owner_action_priority_enabled": True,
            "compact_owner_search_owner_background_maintenance_thread": True,
            "compact_owner_search_owner_background_overlap_enabled": True,
            "compact_owner_search_owner_action_request_count": int(self._action_request_count),
            "compact_owner_search_owner_maintenance_request_count": int(
                self._maintenance_request_count
            ),
            "compact_owner_search_owner_run_request_count": int(self._run_request_count),
            "compact_owner_search_worker_owns_search_state": True,
            "compact_owner_search_worker_owns_replay_state": bool(
                service_metadata.get("compact_owner_search_service_worker_owns_replay_state")
            ),
            "compact_owner_search_worker_owns_model_state": bool(
                service_metadata.get("compact_owner_search_service_worker_owns_model_state")
            ),
            "compact_owner_search_worker_resource_distinct_from_actor": True,
            "compact_owner_search_worker_hardware_resource_distinct_from_actor": False,
        }

    def prepare(self) -> None:
        if self._owner_thread is not None:
            return
        self._owner_thread = threading.Thread(
            target=_compact_threaded_owner_search_priority_loop,
            args=(self._service, self._command_queue, self._result_queue),
            name="compact-owner-search-threaded-priority-loop",
            daemon=False,
        )
        self._collector_thread = threading.Thread(
            target=self._collect_priority_loop_results,
            name="compact-owner-search-threaded-result-collector",
            daemon=True,
        )
        self._owner_thread.start()
        self._collector_thread.start()

    def submit(self, request: CompactOwnerSearchRequestV1) -> Future[dict[str, Any]]:
        self._run_request_count += 1
        return self._submit_priority_command(_OWNER_PRIORITY_COMMAND_RUN, request)

    def submit_action(
        self,
        request: CompactOwnerSearchRequestV1,
    ) -> Future[dict[str, Any]]:
        self._action_request_count += 1
        return self._submit_priority_command(_OWNER_PRIORITY_COMMAND_ACTION, request)

    def submit_maintenance_drain(
        self,
        request: CompactOwnerMaintenanceDrainRequestV1,
    ) -> Future[dict[str, Any]]:
        self._maintenance_request_count += 1
        return self._submit_priority_command(
            _OWNER_PRIORITY_COMMAND_DRAIN_MAINTENANCE,
            request,
        )

    def _submit_priority_command(
        self,
        command_kind: str,
        payload: Any,
    ) -> Future[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("owner-search threaded worker is closed")
        self.prepare()
        future: Future[dict[str, Any]] = Future()
        with self._futures_lock:
            self._next_command_id += 1
            command_id = int(self._next_command_id)
            self._futures[command_id] = future
        self._command_queue.put(
            _CompactOwnerPriorityCommandV1(
                command_id=command_id,
                command_kind=str(command_kind),
                payload=payload,
            )
        )
        return future

    def _collect_priority_loop_results(self) -> None:
        while True:
            try:
                result = self._result_queue.get()
            except (EOFError, OSError):
                break
            if result is None:
                break
            if not isinstance(result, _CompactOwnerPriorityResultV1):
                continue
            with self._futures_lock:
                future = self._futures.pop(int(result.command_id), None)
            if future is None:
                continue
            if bool(result.ok):
                future.set_result(dict(result.payload or {}))
            else:
                message = str(result.error_message or "owner-search threaded loop failed")
                future.set_exception(RuntimeError(message))

    def done(self, handle: Future[dict[str, Any]]) -> bool:
        self._fail_futures_if_thread_dead()
        return bool(handle.done())

    def result(self, handle: Future[dict[str, Any]]) -> dict[str, Any]:
        while True:
            try:
                return dict(handle.result(timeout=0.1))
            except FutureTimeoutError:
                self._fail_futures_if_thread_dead()
                continue

    def close(self) -> None:
        if self._closed:
            return
        close_future: Future[dict[str, Any]] | None = None
        owner_thread = self._owner_thread
        if owner_thread is not None and owner_thread.is_alive():
            try:
                close_future = self._submit_priority_command(
                    _OWNER_PRIORITY_COMMAND_CLOSE,
                    None,
                )
            except RuntimeError:
                close_future = None
        self._closed = True
        if close_future is not None:
            try:
                close_future.result(timeout=10.0)
            except Exception:
                pass
        if owner_thread is not None:
            owner_thread.join(timeout=10.0)
        self._result_queue.put(None)
        collector = self._collector_thread
        if collector is not None:
            collector.join(timeout=2.0)
        with self._futures_lock:
            remaining = list(self._futures.values())
            self._futures.clear()
        for future in remaining:
            if not future.done():
                future.set_exception(RuntimeError("owner-search threaded worker closed"))
        self._owner_thread = None
        self._collector_thread = None

    def _fail_futures_if_thread_dead(self) -> None:
        owner_thread = self._owner_thread
        if owner_thread is None or owner_thread.is_alive() or self._closed:
            return
        with self._futures_lock:
            remaining = list(self._futures.values())
            self._futures.clear()
        for future in remaining:
            if not future.done():
                future.set_exception(RuntimeError("owner-search threaded loop exited"))


class CompactOwnerSearchSlabProxyV1:
    """Use an owner-search worker as a ``CompactRolloutSlab`` search service."""

    profile_only = True
    calls_train_muzero = False
    touches_live_runs = False
    supports_two_phase_compact_search = True
    direct_root_build_request_supported = False

    def __init__(
        self,
        *,
        root_store: CompactSharedMemoryRootStoreV1,
        worker: CompactProcessOwnerSearchWorkerV1,
        policy_version_ref: str = "",
        model_version_ref: str = "",
        policy_source: str = "",
        owner_replay_append_enabled: bool = False,
        owner_sample_batch_size: int = 0,
        owner_train_steps: int = 0,
        owner_train_interval: int = 1,
        owner_model_refresh_interval: int = 1,
        owner_expected_train_request_count: int = 0,
        owner_defer_maintenance: bool = False,
        owner_learning_enabled: bool = True,
        boundary_kind: str = "worker_search_parent_slab_commit",
        action_result_slot_table: CompactOwnerActionResultSlotTableV1 | None = None,
    ) -> None:
        self.root_store = root_store
        self.worker = worker
        self.policy_version_ref = str(policy_version_ref)
        self.model_version_ref = str(model_version_ref)
        self.policy_source = str(policy_source)
        self.owner_replay_append_enabled = bool(owner_replay_append_enabled)
        self.owner_sample_batch_size = int(owner_sample_batch_size)
        self.owner_train_steps = int(owner_train_steps)
        self.owner_train_interval = int(owner_train_interval)
        self.owner_model_refresh_interval = int(owner_model_refresh_interval)
        self.owner_expected_train_request_count = int(owner_expected_train_request_count)
        self.owner_defer_maintenance = bool(owner_defer_maintenance)
        self.owner_learning_enabled = bool(owner_learning_enabled)
        self.boundary_kind = str(boundary_kind)
        self.action_result_slot_table = action_result_slot_table
        if not self.boundary_kind:
            raise ValueError("boundary_kind must be non-empty")
        if self.owner_sample_batch_size < 0:
            raise ValueError("owner_sample_batch_size must be nonnegative")
        if self.owner_train_steps < 0:
            raise ValueError("owner_train_steps must be nonnegative")
        if self.owner_train_interval <= 0:
            raise ValueError("owner_train_interval must be positive")
        if self.owner_model_refresh_interval <= 0:
            raise ValueError("owner_model_refresh_interval must be positive")
        if self.owner_expected_train_request_count < 0:
            raise ValueError("owner_expected_train_request_count must be nonnegative")
        self.request_count = 0
        self.last_result_payload: dict[str, Any] = {}
        self.last_search_result_payload_bytes = 0
        self.owner_replay_append_staged_entry_count = 0
        self.owner_replay_append_suppressed_entry_count = 0
        self.owner_replay_append_submitted_entry_count = 0
        self.owner_replay_append_staged_transport_entry_count = 0
        self.owner_replay_append_suppressed_transport_entry_count = 0
        self.owner_replay_append_submitted_transport_entry_count = 0
        self.owner_replay_append_transition_batch_count = 0
        self.owner_replay_append_transition_batch_entry_count = 0
        self.owner_replay_append_transition_legacy_entry_count = 0
        self.owner_replay_append_request_count = 0
        self.owner_replay_append_count = 0
        self.owner_train_request_count = 0
        self.owner_model_refresh_request_count = 0
        self.owner_model_refresh_skipped_count = 0
        self.owner_submitted_learner_update_count = 0
        self.owner_learner_update_count = 0
        self.owner_model_owner_ref_returned = False
        self.owner_model_owner_ref_digest = ""
        self.owner_model_state_snapshot_load_count = 0
        self.owner_model_state_snapshot_load_bytes = 0
        self.owner_model_state_snapshot_load_sec = 0.0
        self.owner_search_consumed_learner_update = False
        self.owner_search_refresh_update_count = 0
        self.owner_sample_telemetry: dict[str, Any] = {}
        self.owner_learner_telemetry: dict[str, Any] = {}
        self.owner_action_feedback: dict[str, Any] = _empty_owner_action_feedback_v1()
        self.owner_learner_timing_totals: dict[str, float] = {}
        self.owner_learner_timing_aggregate_count = 0
        self.owner_maintenance_drain_request_count = 0
        self.owner_maintenance_staged_work_item_count = 0
        self.owner_maintenance_drained_count = 0
        self.owner_maintenance_drained_work_item_count = 0
        self.owner_maintenance_drained_replay_append_entry_count = 0
        self.owner_maintenance_drained_replay_append_transport_entry_count = 0
        self.owner_maintenance_drained_replay_append_transition_batch_count = 0
        self.owner_maintenance_drained_replay_append_transition_batch_entry_count = 0
        self.owner_maintenance_drained_replay_append_count = 0
        self.owner_maintenance_pending_work_count = 0
        self.owner_maintenance_final_drain_sec = 0.0
        self.owner_maintenance_replay_append_sec = 0.0
        self.owner_maintenance_learner_train_sec = 0.0
        self.owner_maintenance_search_refresh_sec = 0.0
        self.owner_maintenance_coalesced_skip_count = 0
        self.owner_maintenance_eager_append_drain_count = 0
        self.owner_async_learner_worker_enabled = False
        self.owner_async_learner_worker_kind = "none"
        self.owner_async_learner_worker_resource_scope = ""
        self.owner_async_learner_worker_resource_id = ""
        self.owner_async_learner_actor_resource_id = ""
        self.owner_async_learner_worker_parent_pid = 0
        self.owner_async_learner_resource_distinct_from_owner = False
        self.owner_async_learner_hardware_resource_distinct_from_owner = False
        self.owner_async_learner_max_pending = 0
        self.owner_async_learner_submit_count = 0
        self.owner_async_learner_completed_count = 0
        self.owner_async_learner_pending_count = 0
        self.owner_async_learner_max_pending_observed = 0
        self.owner_async_learner_wait_count = 0
        self.owner_async_learner_wait_sec = 0.0
        self.owner_action_while_async_learner_pending_count = 0
        self.owner_async_learner_failed = False
        self.owner_async_learner_request_host_only = False
        self.owner_async_learner_request_cuda_tensor_count = -1
        self.owner_async_learner_result_host_only = False
        self.owner_async_learner_result_cuda_tensor_count = -1
        self.owner_async_learner_request_bytes = 0
        self.owner_async_learner_result_bytes = 0
        self.owner_async_learner_worker_pid = 0
        self.owner_async_learner_worker_job_wall_sec = 0.0
        self.owner_async_learner_payload_prepare_sec = 0.0
        self.owner_async_learner_worker_pid_distinct_from_owner = False
        self.owner_async_learner_worker_owns_model_state = False
        self.owner_policy_lag_current = 0
        self.owner_policy_lag_max = 0
        self.owner_maintenance_actor_steps_while_pending = 0
        self.owner_maintenance_actor_steps_while_policy_lagged = 0
        self.owner_action_served_before_maintenance_count = 0
        self.owner_fifo_blocked_action_count = 0
        self._replay_payload_counter = 0
        self._pending_replay_payloads: dict[str, CompactSearchReplayPayloadV1] = {}
        self._pending_owner_replay_append_entries: list[Any] = []
        self._pending_owner_proxy_action_frame: _CompactOwnerProxyActionFrameV1 | None = None
        self._pending_owner_proxy_derived_transition_facts: list[
            _CompactOwnerProxyDerivedTransitionFactsV1
        ] = []
        self.owner_proxy_transition_closure_requested_count = 0
        self.owner_proxy_transition_closure_no_pending_count = 0
        self.owner_proxy_transition_closure_closed_count = 0
        self.owner_proxy_transition_closure_batch_count = 0
        self.owner_proxy_transition_closure_transition_count = 0
        self.owner_proxy_transition_closure_transport_entry_count = 0
        self.owner_proxy_transition_closure_transport_bytes = 0
        self.owner_proxy_transition_closure_build_sec = 0.0
        self.owner_proxy_transition_closure_submit_sec = 0.0
        self.owner_proxy_transition_closure_digest = ""
        self.owner_proxy_transition_closure_fallback_count = 0
        self.owner_proxy_transition_closure_fallback_reason = "none"
        self.owner_proxy_transition_closure_applied_action_verification_count = 0
        self.owner_proxy_transition_closure_applied_action_mismatch_count = 0
        self.owner_proxy_transition_closure_applied_action_count = 0
        self.owner_proxy_transition_closure_applied_action_checksum = 0
        self.owner_proxy_transition_closure_action_frame_store_count = 0
        self.owner_action_dispatch_handle_submit_count = 0
        self.owner_action_dispatch_handle_resolve_count = 0
        self.owner_action_dispatch_handle_sync_wrapper_count = 0
        self.owner_action_dispatch_handle_completed_at_submit_count = 0
        self.owner_action_dispatch_handle_pending_count = 0
        self.owner_action_dispatch_handle_max_pending_count = 0
        self.owner_action_dispatch_handle_result_wait_in_submit_count = 0
        self.owner_action_dispatch_handle_result_wait_sec = 0.0
        self.owner_action_dispatch_pending_root_build_request_store_count = 0
        self.owner_action_dispatch_pending_root_build_request_avoided_count = 0
        self.owner_action_dispatch_pending_root_action_context_store_count = 0
        self.owner_action_dispatch_pending_root_action_context_avoided_count = 0
        self.owner_root_action_context_owner_store_count = 0
        self.owner_root_action_context_owner_resolve_count = 0
        self.owner_root_action_context_owner_release_count = 0
        self.owner_root_action_context_owner_digest_verified_count = 0
        self.owner_root_action_context_owner_max_pending_count = 0
        self.owner_root_search_transaction_begin_count = 0
        self.owner_root_search_transaction_submit_count = 0
        self.owner_root_search_transaction_resolve_count = 0
        self.owner_root_search_transaction_max_pending_count = 0
        self.owner_root_search_transaction_owner_root_request_build_count = 0
        self.owner_root_search_transaction_owner_root_request_build_sec = 0.0
        self.owner_root_search_transaction_frame_generation_verified_count = 0
        self.owner_root_search_transaction_frame_digest_verified_count = 0
        self.owner_root_search_transaction_action_identity_verified_count = 0
        self.owner_root_search_transaction_parent_root_request_build_count = 0
        self.owner_root_search_transaction_parent_rebuild_count = 0
        self._next_action_dispatch_id = 0
        self._next_root_action_context_id = 0
        self._next_owner_root_search_transaction_id = 0
        self._root_action_contexts: dict[int, CompactRootActionContextV1] = {}
        self._pending_action_dispatches: dict[int, _PendingOwnerActionDispatchV1] = {}
        self._inflight_owner_maintenance: Future[dict[str, Any]] | None = None
        self._owner_maintenance_failure = ""

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "compact_owner_search_slab_proxy": True,
            "compact_owner_search_boundary_kind": self.boundary_kind,
            "compact_owner_search_parent_slab_commits_replay": (
                not bool(self.owner_defer_maintenance)
            ),
            "compact_owner_search_slab_proxy_request_count": int(self.request_count),
            "compact_owner_search_slab_proxy_last_search_result_payload_bytes": int(
                self.last_search_result_payload_bytes
            ),
            "compact_owner_search_fixed_action_result_buffer_requested": bool(
                self.action_result_slot_table is not None
            ),
            "compact_owner_search_owner_replay_append_enabled": bool(
                self.owner_replay_append_enabled
            ),
            "compact_owner_search_owner_learning_enabled": bool(self.owner_learning_enabled),
            "compact_owner_search_owner_sample_batch_size": int(self.owner_sample_batch_size),
            "compact_owner_search_owner_train_steps": int(self.owner_train_steps),
            "compact_owner_search_owner_train_interval": int(self.owner_train_interval),
            "compact_owner_search_owner_model_refresh_interval": int(
                self.owner_model_refresh_interval
            ),
            "compact_owner_search_owner_expected_train_request_count": int(
                self.owner_expected_train_request_count
            ),
            "compact_owner_search_owner_defer_maintenance": bool(self.owner_defer_maintenance),
            "compact_owner_search_owner_replay_append_staged_entry_count": int(
                self.owner_replay_append_staged_entry_count
            ),
            "compact_owner_search_owner_replay_append_suppressed_entry_count": int(
                self.owner_replay_append_suppressed_entry_count
            ),
            "compact_owner_search_owner_replay_append_submitted_entry_count": int(
                self.owner_replay_append_submitted_entry_count
            ),
            "compact_owner_search_owner_replay_append_staged_transport_entry_count": int(
                self.owner_replay_append_staged_transport_entry_count
            ),
            "compact_owner_search_owner_replay_append_suppressed_transport_entry_count": int(
                self.owner_replay_append_suppressed_transport_entry_count
            ),
            "compact_owner_search_owner_replay_append_submitted_transport_entry_count": int(
                self.owner_replay_append_submitted_transport_entry_count
            ),
            "compact_owner_search_owner_replay_transport_entry_count": int(
                self.owner_replay_append_submitted_transport_entry_count
            ),
            "compact_owner_search_owner_replay_transport_kind": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
                if int(self.owner_replay_append_transition_batch_count) > 0
                else "per_transition_entry_v1"
            ),
            "compact_owner_search_owner_replay_transition_batch_enabled": (
                int(self.owner_replay_append_transition_batch_count) > 0
            ),
            "compact_owner_search_owner_replay_transition_batch_count": int(
                self.owner_replay_append_transition_batch_count
            ),
            "compact_owner_search_owner_replay_transition_batch_transition_count": int(
                self.owner_replay_append_transition_batch_entry_count
            ),
            "compact_owner_search_owner_replay_transition_legacy_entry_count": int(
                self.owner_replay_append_transition_legacy_entry_count
            ),
            "compact_owner_search_transition_batch_transport_enabled": (
                int(self.owner_replay_append_transition_batch_count) > 0
            ),
            "compact_owner_search_transition_batch_transport_kind": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
                if int(self.owner_replay_append_transition_batch_count) > 0
                else "per_transition_entry_v1"
            ),
            "compact_owner_search_transition_batch_schema_id": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
                if int(self.owner_replay_append_transition_batch_count) > 0
                else ""
            ),
            "compact_owner_search_transition_batch_count": int(
                self.owner_replay_append_transition_batch_count
            ),
            "compact_owner_search_transition_batch_entry_count": int(
                self.owner_replay_append_transition_batch_entry_count
            ),
            "compact_owner_search_transition_batch_transport_entry_count": int(
                self.owner_replay_append_submitted_transport_entry_count
            ),
            "compact_owner_search_transition_batch_fallback_count": 0,
            "compact_owner_search_transition_batch_fallback_reason": "none",
            "compact_owner_search_transition_batch_pending_count": int(
                _owner_replay_append_transition_batch_entry_count_total_v1(
                    self._pending_owner_replay_append_entries
                )
            ),
            "compact_owner_search_owner_replay_append_request_count": int(
                self.owner_replay_append_request_count
            ),
            "compact_owner_search_owner_replay_append_count": int(self.owner_replay_append_count),
            "compact_owner_search_replay_append_entry_count": int(
                self.owner_replay_append_submitted_entry_count
            ),
            "compact_owner_search_replay_append_count": int(self.owner_replay_append_count),
            "compact_owner_search_owner_train_request_count": int(self.owner_train_request_count),
            "compact_owner_search_owner_model_refresh_request_count": int(
                self.owner_model_refresh_request_count
            ),
            "compact_owner_search_owner_model_refresh_skipped_count": int(
                self.owner_model_refresh_skipped_count
            ),
            "compact_owner_search_owner_submitted_learner_update_count": int(
                self.owner_submitted_learner_update_count
            ),
            "compact_owner_search_owner_learner_update_count": int(self.owner_learner_update_count),
            "compact_owner_search_learner_update_count": int(self.owner_learner_update_count),
            "compact_owner_search_model_owner_ref_returned": bool(
                self.owner_model_owner_ref_returned
            ),
            "compact_owner_search_model_owner_ref_digest": str(self.owner_model_owner_ref_digest),
            "compact_owner_search_model_state_snapshot_load_count": int(
                self.owner_model_state_snapshot_load_count
            ),
            "compact_owner_search_model_state_snapshot_load_bytes": int(
                self.owner_model_state_snapshot_load_bytes
            ),
            "compact_owner_search_model_state_snapshot_load_sec": float(
                self.owner_model_state_snapshot_load_sec
            ),
            "compact_owner_search_consumed_learner_update": bool(
                self.owner_search_consumed_learner_update
            ),
            "compact_owner_search_search_refresh_update_count": int(
                self.owner_search_refresh_update_count
            ),
            "compact_owner_search_owner_sample_telemetry": dict(self.owner_sample_telemetry),
            "compact_owner_search_owner_learner_telemetry": dict(self.owner_learner_telemetry),
            **dict(self.owner_action_feedback),
            "compact_owner_search_owner_proxy_transition_closure_requested_count": int(
                self.owner_proxy_transition_closure_requested_count
            ),
            "compact_owner_search_owner_proxy_transition_closure_used": bool(
                self.owner_proxy_transition_closure_closed_count > 0
            ),
            "compact_owner_search_owner_proxy_transition_closure_source": (
                "owner_proxy_cached_state_v1"
                if self.owner_proxy_transition_closure_requested_count > 0
                else ""
            ),
            "compact_owner_search_owner_proxy_transition_closure_no_pending_count": int(
                self.owner_proxy_transition_closure_no_pending_count
            ),
            "compact_owner_search_owner_proxy_transition_closure_closed_count": int(
                self.owner_proxy_transition_closure_closed_count
            ),
            "compact_owner_search_owner_proxy_transition_closure_batch_count": int(
                self.owner_proxy_transition_closure_batch_count
            ),
            "compact_owner_search_owner_proxy_transition_closure_transition_count": int(
                self.owner_proxy_transition_closure_transition_count
            ),
            "compact_owner_search_owner_proxy_transition_closure_transport_entry_count": int(
                self.owner_proxy_transition_closure_transport_entry_count
            ),
            "compact_owner_search_owner_proxy_transition_closure_pending_count": int(
                len(self._pending_owner_proxy_derived_transition_facts)
            ),
            "compact_owner_search_owner_proxy_transition_closure_transport_bytes": int(
                self.owner_proxy_transition_closure_transport_bytes
            ),
            "compact_owner_search_owner_proxy_transition_closure_digest": str(
                self.owner_proxy_transition_closure_digest
            ),
            "compact_owner_search_owner_proxy_transition_closure_digest_verified": bool(
                self.owner_proxy_transition_closure_digest
                or self.owner_proxy_transition_closure_transition_count == 0
            ),
            "compact_owner_search_owner_proxy_transition_closure_build_sec": float(
                self.owner_proxy_transition_closure_build_sec
            ),
            "compact_owner_search_owner_proxy_transition_closure_submit_sec": float(
                self.owner_proxy_transition_closure_submit_sec
            ),
            "compact_owner_search_owner_proxy_transition_closure_fallback_count": int(
                self.owner_proxy_transition_closure_fallback_count
            ),
            "compact_owner_search_owner_proxy_transition_closure_fallback_reason": str(
                self.owner_proxy_transition_closure_fallback_reason
            ),
            "compact_owner_search_owner_proxy_applied_action_verification_count": int(
                self.owner_proxy_transition_closure_applied_action_verification_count
            ),
            "compact_owner_search_owner_proxy_applied_action_mismatch_count": int(
                self.owner_proxy_transition_closure_applied_action_mismatch_count
            ),
            "compact_owner_search_owner_proxy_applied_action_count": int(
                self.owner_proxy_transition_closure_applied_action_count
            ),
            "compact_owner_search_owner_proxy_applied_action_checksum": int(
                self.owner_proxy_transition_closure_applied_action_checksum
            ),
            "compact_owner_search_owner_proxy_action_frame_pending": bool(
                self._pending_owner_proxy_action_frame is not None
            ),
            "compact_owner_search_owner_proxy_action_frame_store_count": int(
                self.owner_proxy_transition_closure_action_frame_store_count
            ),
            "compact_owner_search_action_dispatch_handle_boundary_supported": True,
            "compact_owner_search_action_dispatch_handle_submit_count": int(
                self.owner_action_dispatch_handle_submit_count
            ),
            "compact_owner_search_action_dispatch_handle_resolve_count": int(
                self.owner_action_dispatch_handle_resolve_count
            ),
            "compact_owner_search_action_dispatch_handle_sync_wrapper_count": int(
                self.owner_action_dispatch_handle_sync_wrapper_count
            ),
            "compact_owner_search_action_dispatch_handle_completed_at_submit_count": int(
                self.owner_action_dispatch_handle_completed_at_submit_count
            ),
            "compact_owner_search_action_dispatch_handle_pending_count": int(
                self.owner_action_dispatch_handle_pending_count
            ),
            "compact_owner_search_action_dispatch_handle_max_pending_count": int(
                self.owner_action_dispatch_handle_max_pending_count
            ),
            "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count": int(
                self.owner_action_dispatch_handle_result_wait_in_submit_count
            ),
            "compact_owner_search_action_dispatch_handle_result_wait_sec": float(
                self.owner_action_dispatch_handle_result_wait_sec
            ),
            "compact_owner_search_action_dispatch_pending_root_build_request_stored": False,
            "compact_owner_search_action_dispatch_pending_root_build_request_store_count": int(
                self.owner_action_dispatch_pending_root_build_request_store_count
            ),
            "compact_owner_search_action_dispatch_pending_root_build_request_avoided_count": int(
                self.owner_action_dispatch_pending_root_build_request_avoided_count
            ),
            "compact_owner_search_action_dispatch_pending_root_action_context_stored": False,
            "compact_owner_search_action_dispatch_pending_root_action_context_store_count": int(
                self.owner_action_dispatch_pending_root_action_context_store_count
            ),
            (
                "compact_owner_search_action_dispatch_pending_root_action_context_"
                "avoided_count"
            ): int(self.owner_action_dispatch_pending_root_action_context_avoided_count),
            **self._owner_root_action_context_handle_metadata_v1(None),
            "compact_owner_root_search_transaction_boundary_supported": True,
            "compact_owner_root_search_transaction_used": bool(
                self.owner_root_search_transaction_begin_count > 0
            ),
            "compact_owner_root_search_transaction_requested": bool(
                self.owner_root_search_transaction_begin_count > 0
            ),
            "compact_owner_root_search_transaction_schema_id": (
                COMPACT_OWNER_ROOT_SEARCH_TRANSACTION_SCHEMA_ID
            ),
            "compact_owner_root_search_transaction_begin_count": int(
                self.owner_root_search_transaction_begin_count
            ),
            "compact_owner_root_search_transaction_submit_count": int(
                self.owner_root_search_transaction_submit_count
            ),
            "compact_owner_root_search_transaction_resolve_count": int(
                self.owner_root_search_transaction_resolve_count
            ),
            "compact_owner_root_search_transaction_pending_count": int(
                sum(
                    1
                    for pending in self._pending_action_dispatches.values()
                    if int(pending.root_search_transaction_id) > 0
                )
            ),
            "compact_owner_root_search_transaction_max_pending_count": int(
                self.owner_root_search_transaction_max_pending_count
            ),
            "compact_owner_root_search_transaction_parent_root_request_build_count": int(
                self.owner_root_search_transaction_parent_root_request_build_count
            ),
            "compact_owner_root_search_transaction_parent_root_request_stored": False,
            "compact_owner_root_search_transaction_parent_compact_batch_stored": False,
            "compact_owner_root_search_transaction_parent_rebuild_count": int(
                self.owner_root_search_transaction_parent_rebuild_count
            ),
            "compact_owner_root_search_transaction_parent_root_action_context_stored": False,
            "compact_owner_root_search_transaction_parent_root_action_context_store_count": 0,
            "compact_owner_root_search_transaction_parent_root_action_context_array_bytes": 0,
            "compact_owner_root_search_transaction_parent_root_action_context_field_count": 0,
            "compact_owner_root_search_transaction_owner_root_request_build_count": int(
                self.owner_root_search_transaction_owner_root_request_build_count
            ),
            "compact_owner_root_search_transaction_owner_root_request_build_sec": float(
                self.owner_root_search_transaction_owner_root_request_build_sec
            ),
            "compact_owner_root_search_transaction_owner_root_store_publish_count": int(
                getattr(self.root_store, "root_build_request_publish_count", 0)
            ),
            "compact_owner_root_search_transaction_frame_generation_verified": bool(
                self.owner_root_search_transaction_frame_generation_verified_count > 0
            ),
            "compact_owner_root_search_transaction_frame_digest_verified": bool(
                self.owner_root_search_transaction_frame_digest_verified_count > 0
            ),
            "compact_owner_root_search_transaction_action_identity_verified": bool(
                self.owner_root_search_transaction_action_identity_verified_count > 0
            ),
            "compact_owner_root_search_transaction_proxy_transition_closure_used": bool(
                self.owner_proxy_transition_closure_closed_count > 0
            ),
            "compact_owner_root_search_transaction_applied_action_mismatch_count": int(
                self.owner_proxy_transition_closure_applied_action_mismatch_count
            ),
            "compact_owner_search_parent_previous_transition_closure_count": 0,
            "compact_owner_search_parent_applied_action_validation_count": 0,
            "compact_owner_search_owner_pending_replay_append_entry_count": int(
                _owner_replay_append_entry_count_v1(self._pending_owner_replay_append_entries)
            ),
            "compact_owner_search_owner_pending_replay_append_transport_entry_count": int(
                _owner_replay_append_transport_entry_count_v1(
                    self._pending_owner_replay_append_entries
                )
            ),
            "compact_owner_search_owner_maintenance_drain_request_count": int(
                self.owner_maintenance_drain_request_count
            ),
            "compact_owner_search_owner_maintenance_staged_work_item_count": int(
                self.owner_maintenance_staged_work_item_count
            ),
            "compact_owner_search_owner_maintenance_drained_count": int(
                self.owner_maintenance_drained_count
            ),
            "compact_owner_search_owner_maintenance_drained_work_item_count": int(
                self.owner_maintenance_drained_work_item_count
            ),
            "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": int(
                self.owner_maintenance_drained_replay_append_entry_count
            ),
            "compact_owner_search_owner_maintenance_drained_replay_append_transport_entry_count": int(
                self.owner_maintenance_drained_replay_append_transport_entry_count
            ),
            "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_count": int(
                self.owner_maintenance_drained_replay_append_transition_batch_count
            ),
            "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_entry_count": int(
                self.owner_maintenance_drained_replay_append_transition_batch_entry_count
            ),
            "compact_owner_search_owner_maintenance_drained_replay_append_count": int(
                self.owner_maintenance_drained_replay_append_count
            ),
            "compact_owner_search_owner_maintenance_pending_work_count": int(
                self.owner_maintenance_pending_work_count
            ),
            "compact_owner_search_owner_maintenance_inflight": (
                self._inflight_owner_maintenance is not None
            ),
            "compact_owner_search_owner_maintenance_final_drain_sec": float(
                self.owner_maintenance_final_drain_sec
            ),
            "compact_owner_search_owner_maintenance_coalescing_kind": (
                "eager_append_or_train_boundary_v1" if self.owner_defer_maintenance else ""
            ),
            "compact_owner_search_owner_maintenance_coalesced_skip_count": int(
                self.owner_maintenance_coalesced_skip_count
            ),
            "compact_owner_search_owner_maintenance_eager_append_drain_count": int(
                self.owner_maintenance_eager_append_drain_count
            ),
            "compact_owner_search_owner_async_learner_worker_enabled": bool(
                self.owner_async_learner_worker_enabled
            ),
            "compact_owner_search_owner_async_learner_worker_kind": str(
                self.owner_async_learner_worker_kind
            ),
            "compact_owner_search_owner_async_learner_worker_resource_scope": str(
                self.owner_async_learner_worker_resource_scope
            ),
            "compact_owner_search_owner_async_learner_worker_resource_id": str(
                self.owner_async_learner_worker_resource_id
            ),
            "compact_owner_search_owner_async_learner_actor_resource_id": str(
                self.owner_async_learner_actor_resource_id
            ),
            "compact_owner_search_owner_async_learner_worker_parent_pid": int(
                self.owner_async_learner_worker_parent_pid
            ),
            "compact_owner_search_owner_async_learner_resource_distinct_from_owner": bool(
                self.owner_async_learner_resource_distinct_from_owner
            ),
            (
                "compact_owner_search_owner_async_learner_hardware_resource_distinct_from_owner"
            ): bool(self.owner_async_learner_hardware_resource_distinct_from_owner),
            "compact_owner_search_owner_async_learner_max_pending": int(
                self.owner_async_learner_max_pending
            ),
            "compact_owner_search_owner_async_learner_submit_count": int(
                self.owner_async_learner_submit_count
            ),
            "compact_owner_search_owner_async_learner_completed_count": int(
                self.owner_async_learner_completed_count
            ),
            "compact_owner_search_owner_async_learner_pending_count": int(
                self.owner_async_learner_pending_count
            ),
            "compact_owner_search_owner_async_learner_max_pending_observed": int(
                self.owner_async_learner_max_pending_observed
            ),
            "compact_owner_search_owner_async_learner_wait_count": int(
                self.owner_async_learner_wait_count
            ),
            "compact_owner_search_owner_async_learner_wait_sec": float(
                self.owner_async_learner_wait_sec
            ),
            "compact_owner_search_owner_action_while_async_learner_pending_count": int(
                self.owner_action_while_async_learner_pending_count
            ),
            "compact_owner_search_owner_async_learner_failed": bool(
                self.owner_async_learner_failed
            ),
            "compact_owner_search_owner_async_learner_request_host_only": bool(
                self.owner_async_learner_request_host_only
            ),
            "compact_owner_search_owner_async_learner_request_cuda_tensor_count": int(
                self.owner_async_learner_request_cuda_tensor_count
            ),
            "compact_owner_search_owner_async_learner_result_host_only": bool(
                self.owner_async_learner_result_host_only
            ),
            "compact_owner_search_owner_async_learner_result_cuda_tensor_count": int(
                self.owner_async_learner_result_cuda_tensor_count
            ),
            "compact_owner_search_owner_async_learner_request_bytes": int(
                self.owner_async_learner_request_bytes
            ),
            "compact_owner_search_owner_async_learner_result_bytes": int(
                self.owner_async_learner_result_bytes
            ),
            "compact_owner_search_owner_async_learner_worker_pid": int(
                self.owner_async_learner_worker_pid
            ),
            "compact_owner_search_owner_async_learner_worker_job_wall_sec": float(
                self.owner_async_learner_worker_job_wall_sec
            ),
            "compact_owner_search_owner_async_learner_payload_prepare_sec": float(
                self.owner_async_learner_payload_prepare_sec
            ),
            ("compact_owner_search_owner_async_learner_worker_pid_distinct_from_owner"): bool(
                self.owner_async_learner_worker_pid_distinct_from_owner
            ),
            "compact_owner_search_owner_async_learner_worker_owns_model_state": bool(
                self.owner_async_learner_worker_owns_model_state
            ),
            "compact_owner_search_worker_replay_append_sec": float(
                self.owner_maintenance_replay_append_sec
            ),
            "compact_owner_search_worker_learner_train_sec": float(
                self.owner_maintenance_learner_train_sec
            ),
            "compact_owner_search_worker_search_refresh_sec": float(
                self.owner_maintenance_search_refresh_sec
            ),
            "compact_owner_search_owner_policy_lag_current": int(self.owner_policy_lag_current),
            "compact_owner_search_owner_policy_lag_max": int(self.owner_policy_lag_max),
            "compact_owner_search_owner_maintenance_actor_steps_while_pending": int(
                self.owner_maintenance_actor_steps_while_pending
            ),
            "compact_owner_search_owner_maintenance_actor_steps_while_policy_lagged": int(
                self.owner_maintenance_actor_steps_while_policy_lagged
            ),
            "compact_owner_search_owner_action_while_maintenance_pending_count": int(
                self.owner_maintenance_actor_steps_while_pending
            ),
            "compact_owner_search_owner_action_while_policy_lagged_count": int(
                self.owner_maintenance_actor_steps_while_policy_lagged
            ),
            "compact_owner_search_owner_action_served_before_maintenance_count": int(
                self.owner_action_served_before_maintenance_count
            ),
            "compact_owner_search_owner_fifo_blocked_action_count": int(
                self.owner_fifo_blocked_action_count
            ),
            "compact_owner_search_owner_maintenance_failed": bool(self._owner_maintenance_failure),
            **self.root_store.metadata,
            **self.worker.metadata,
        }

    def set_owner_learning_enabled(self, enabled: bool) -> None:
        self.owner_learning_enabled = bool(enabled)
        if not self.owner_learning_enabled:
            self._pending_owner_replay_append_entries.clear()
            self.owner_maintenance_pending_work_count = 0
            self._pending_owner_proxy_action_frame = None
            self._pending_owner_proxy_derived_transition_facts.clear()

    def stage_replay_append_entries(self, replay_append_entries: Any) -> int:
        """Queue committed replay rows for the next owner request.

        The parent may still build replay rows, but the owner owns the replay
        store and learner state once this path is enabled. CUDA tensors are
        rejected here because they would cross the process boundary.
        """

        if not self.owner_replay_append_enabled:
            return 0
        if replay_append_entries is None:
            return 0
        if not self.owner_learning_enabled:
            self.owner_replay_append_suppressed_entry_count += int(
                _owner_replay_append_entry_count_v1(replay_append_entries)
            )
            self.owner_replay_append_suppressed_transport_entry_count += int(
                _owner_replay_append_transport_entry_count_v1(replay_append_entries)
            )
            return 0
        if isinstance(replay_append_entries, tuple):
            entries = replay_append_entries
        elif isinstance(replay_append_entries, list):
            entries = tuple(replay_append_entries)
        else:
            entries = (replay_append_entries,)
        if not entries:
            return 0
        if _contains_cuda_tensor(entries):
            raise RuntimeError("owner-search replay append entries must not contain CUDA tensors")
        self._pending_owner_replay_append_entries.extend(entries)
        logical_count = _owner_replay_append_entry_count_v1(entries)
        transport_count = _owner_replay_append_transport_entry_count_v1(entries)
        self.owner_replay_append_staged_entry_count += int(logical_count)
        self.owner_replay_append_staged_transport_entry_count += int(transport_count)
        return int(logical_count)

    def stage_owner_proxy_transition_from_root_build_request(
        self,
        root_build_request: CompactRootBuildRequestV1,
        *,
        max_entries_per_batch: int,
        policy_source: str,
    ) -> dict[str, float]:
        """Close the previous transition from proxy-cached action state."""

        if not isinstance(root_build_request, CompactRootBuildRequestV1):
            raise TypeError("root_build_request must be CompactRootBuildRequestV1")
        timing: dict[str, float] = {
            "compact_rollout_slab_replay_index_rows_build_sec": 0.0,
            "compact_rollout_slab_commit_action_check_sec": 0.0,
            "compact_rollout_slab_replay_index_rows_store_sec": 0.0,
            "compact_rollout_slab_owner_replay_transition_stage_sec": 0.0,
            "compact_rollout_slab_owner_replay_transition_stage_count": 0.0,
            "compact_rollout_slab_owner_replay_transition_stage_entry_count": 0.0,
            "compact_rollout_slab_owner_replay_transition_stage_transport_entry_count": 0.0,
            "compact_rollout_slab_owner_replay_transition_batch_submit_sec": 0.0,
        }
        if not self.owner_replay_append_enabled:
            raise ReplayCompatibilityError(
                "owner-proxy transition closure requires owner replay append"
            )
        fixed_capacity = int(max_entries_per_batch)
        if fixed_capacity <= 1:
            raise ReplayCompatibilityError(
                "owner-proxy transition closure requires transition batch capacity > 1"
            )
        self.owner_proxy_transition_closure_requested_count += 1
        if not self.owner_learning_enabled:
            self._pending_owner_proxy_action_frame = None
            self._pending_owner_proxy_derived_transition_facts.clear()
            return timing
        pending = self._pending_owner_proxy_action_frame
        if pending is None:
            self.owner_proxy_transition_closure_no_pending_count += 1
            return timing

        outcome_started = time.perf_counter()
        compact_transition_outcome_v1_from_root_build_request(
            root_build_request,
            batch_size=int(root_build_request.batch_size),
            player_count=int(root_build_request.player_count),
        )
        timing["compact_rollout_slab_replay_index_rows_build_sec"] = _elapsed(
            outcome_started
        )

        joint_action = root_build_request.joint_action
        if joint_action is None:
            self.owner_proxy_transition_closure_fallback_count += 1
            self.owner_proxy_transition_closure_fallback_reason = "missing_joint_action"
            raise ReplayCompatibilityError(
                "owner-proxy transition closure requires root-build joint_action"
            )
        next_joint_action = np.asarray(joint_action, dtype=np.int16)
        expected_shape = (int(root_build_request.batch_size), int(root_build_request.player_count))
        if next_joint_action.shape != expected_shape:
            self.owner_proxy_transition_closure_fallback_count += 1
            self.owner_proxy_transition_closure_fallback_reason = "joint_action_shape_mismatch"
            raise ReplayCompatibilityError(
                "owner-proxy transition closure joint_action shape mismatch"
            )

        action_check_started = time.perf_counter()
        env_row = np.asarray(pending.env_row, dtype=np.int64).reshape(-1)
        player = np.asarray(pending.player, dtype=np.int64).reshape(-1)
        selected = np.asarray(pending.selected_action, dtype=np.int16).reshape(-1)
        if not (env_row.shape == player.shape == selected.shape):
            self.owner_proxy_transition_closure_fallback_count += 1
            self.owner_proxy_transition_closure_fallback_reason = "pending_action_shape_mismatch"
            raise ReplayCompatibilityError(
                "owner-proxy transition closure pending action shape mismatch"
            )
        applied = np.asarray(next_joint_action[env_row, player], dtype=np.int16).reshape(-1)
        self.owner_proxy_transition_closure_applied_action_verification_count += 1
        if selected.size and not np.array_equal(applied, selected):
            self.owner_proxy_transition_closure_applied_action_mismatch_count += 1
            raise ReplayCompatibilityError(
                "owner-proxy transition closure applied action mismatch"
            )
        timing["compact_rollout_slab_commit_action_check_sec"] = _elapsed(
            action_check_started
        )

        applied_checksum = _owner_action_checksum_v1(applied)
        self.owner_proxy_transition_closure_applied_action_count += int(applied.size)
        self.owner_proxy_transition_closure_applied_action_checksum += int(applied_checksum)
        transition_started = time.perf_counter()
        facts = _CompactOwnerProxyDerivedTransitionFactsV1(
            record_index=int(pending.record_index),
            next_record_index=int(pending.record_index + 1),
            replay_payload_handle=str(pending.replay_payload_handle),
            selected_action_digest=str(pending.selected_action_digest),
            search_replay_payload_digest=str(pending.search_replay_payload_digest),
            applied_action_count=int(applied.size),
            applied_action_checksum=int(applied_checksum),
            policy_source=str(policy_source or self.policy_source),
        )
        self._pending_owner_proxy_derived_transition_facts.append(facts)
        self._pending_owner_proxy_action_frame = None
        self.owner_proxy_transition_closure_closed_count += 1
        self.owner_proxy_transition_closure_transition_count += 1
        staged_count = self._flush_owner_proxy_derived_transition_batch(
            max_entries_per_batch=fixed_capacity,
            force=False,
        )
        timing["compact_rollout_slab_replay_index_rows_store_sec"] = _elapsed(
            transition_started
        )
        timing["compact_rollout_slab_owner_replay_transition_stage_sec"] = timing[
            "compact_rollout_slab_replay_index_rows_store_sec"
        ]
        timing["compact_rollout_slab_owner_replay_transition_stage_count"] = float(
            1 if staged_count else 0
        )
        timing["compact_rollout_slab_owner_replay_transition_stage_entry_count"] = float(
            staged_count
        )
        timing[
            "compact_rollout_slab_owner_replay_transition_stage_transport_entry_count"
        ] = float(1 if staged_count else 0)
        timing["compact_rollout_slab_owner_replay_transition_batch_submit_sec"] = float(
            self.owner_proxy_transition_closure_submit_sec
        )
        return timing

    def _flush_owner_proxy_derived_transition_batch(
        self,
        *,
        max_entries_per_batch: int,
        force: bool,
    ) -> int:
        count = len(self._pending_owner_proxy_derived_transition_facts)
        if count <= 0:
            return 0
        fixed_capacity = int(max_entries_per_batch)
        if not bool(force) and count < fixed_capacity:
            return 0
        batch_started = time.perf_counter()
        facts = tuple(self._pending_owner_proxy_derived_transition_facts[:fixed_capacity])
        batch = _owner_proxy_replay_append_derived_transition_batch_v1(
            facts,
            max_entries_per_batch=fixed_capacity,
        )
        build_sec = _elapsed(batch_started)
        submit_started = time.perf_counter()
        staged_count = int(self.stage_replay_append_entries(batch) or 0)
        submit_sec = _elapsed(submit_started)
        del self._pending_owner_proxy_derived_transition_facts[: len(facts)]
        self.owner_proxy_transition_closure_batch_count += 1
        self.owner_proxy_transition_closure_transport_entry_count += 1
        self.owner_proxy_transition_closure_transport_bytes += int(
            batch.metadata.get("compact_owner_search_transition_batch_transport_bytes") or 0
        )
        self.owner_proxy_transition_closure_digest = str(
            batch.metadata.get("compact_owner_search_transition_batch_digest") or ""
        )
        self.owner_proxy_transition_closure_build_sec += float(build_sec)
        self.owner_proxy_transition_closure_submit_sec += float(submit_sec)
        return int(staged_count)

    def _record_owner_proxy_action_frame(
        self,
        *,
        record_index: int,
        action_step: CompactSearchActionStepV1,
    ) -> None:
        if not bool(
            dict(getattr(action_step, "metadata", {}) or {}).get(
                "compact_owner_search_owner_materializes_replay",
                False,
            )
        ):
            raise ReplayCompatibilityError(
                "owner-proxy transition closure requires owner-materialized replay"
            )
        dense_joint_action = getattr(action_step, "dense_joint_action", None)
        if dense_joint_action is None:
            self.owner_proxy_transition_closure_fallback_count += 1
            self.owner_proxy_transition_closure_fallback_reason = "missing_owner_dense_action"
            raise ReplayCompatibilityError(
                "owner-proxy transition closure requires owner dense action"
            )
        selected_action = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
        env_row = np.asarray(action_step.env_row, dtype=np.int32).reshape(-1)
        player = np.asarray(action_step.player, dtype=np.int16).reshape(-1)
        if not (selected_action.shape == env_row.shape == player.shape):
            self.owner_proxy_transition_closure_fallback_count += 1
            self.owner_proxy_transition_closure_fallback_reason = "action_step_shape_mismatch"
            raise ReplayCompatibilityError(
                "owner-proxy transition closure action-step shape mismatch"
            )
        self._pending_owner_proxy_action_frame = _CompactOwnerProxyActionFrameV1(
            record_index=int(record_index),
            replay_payload_handle=str(action_step.replay_payload_handle),
            selected_action_digest=str(action_step.metadata.get("selected_action_digest") or ""),
            search_replay_payload_digest=str(
                action_step.metadata.get("search_replay_payload_digest") or ""
            ),
            env_row=env_row.astype(np.int32, copy=True),
            player=player.astype(np.int16, copy=True),
            selected_action=selected_action.astype(np.int16, copy=True),
        )
        self.owner_proxy_transition_closure_action_frame_store_count += 1

    def drain_owner_maintenance(self, *, wait: bool = True) -> dict[str, Any]:
        """Collect deferred owner maintenance and return current proxy metadata."""

        if bool(wait):
            drain_started = time.perf_counter()
            try:
                while True:
                    self._poll_owner_maintenance(wait=True)
                    if (
                        self._inflight_owner_maintenance is None
                        and int(self.owner_maintenance_pending_work_count) <= 0
                    ):
                        break
                    if self._inflight_owner_maintenance is None:
                        self._submit_owner_maintenance_drain()
            finally:
                self.owner_maintenance_final_drain_sec += _elapsed(drain_started)
        else:
            self._poll_owner_maintenance(wait=False)
        return dict(self.metadata)

    def _raise_if_owner_maintenance_failed(self) -> None:
        if self._owner_maintenance_failure:
            raise RuntimeError(self._owner_maintenance_failure)

    def _poll_owner_maintenance(self, *, wait: bool) -> dict[str, Any] | None:
        handle = self._inflight_owner_maintenance
        if handle is None:
            return None
        if not bool(wait) and not self.worker.done(handle):
            return None
        try:
            payload = self.worker.result(handle)
        except Exception as exc:
            self._inflight_owner_maintenance = None
            self._owner_maintenance_failure = (
                f"owner-search deferred maintenance failed closed: {exc}"
            )
            raise RuntimeError(self._owner_maintenance_failure) from exc
        self._inflight_owner_maintenance = None
        self._apply_owner_maintenance_payload(payload)
        return payload

    def _submit_owner_maintenance_drain(self, *, max_items: int = 0) -> None:
        if self._inflight_owner_maintenance is not None:
            return
        self.owner_maintenance_drain_request_count += 1
        request = CompactOwnerMaintenanceDrainRequestV1(
            drain_id=int(self.owner_maintenance_drain_request_count),
            max_items=max(0, int(max_items)),
        )
        self._inflight_owner_maintenance = self.worker.submit_maintenance_drain(request)

    def _submit_owner_maintenance_drain_if_due(
        self,
        *,
        train_steps: int,
        staged_work_count: int,
    ) -> bool:
        if int(staged_work_count) <= 0:
            return False
        if self._inflight_owner_maintenance is not None:
            return False
        if int(train_steps) <= 0:
            self.owner_maintenance_coalesced_skip_count += 1
            return False
        self._submit_owner_maintenance_drain()
        return True

    def _refresh_owner_policy_lag(self) -> None:
        self.owner_policy_lag_current = max(
            0,
            int(self.owner_submitted_learner_update_count)
            - int(self.owner_search_refresh_update_count),
        )
        self.owner_policy_lag_max = max(
            int(self.owner_policy_lag_max),
            int(self.owner_policy_lag_current),
        )

    def _apply_owner_search_worker_state(self, value: Any) -> None:
        if not isinstance(value, Mapping) or not value:
            return
        state = dict(value)
        if bool(state.get("model_state_snapshot_loaded", False)):
            self.owner_model_state_snapshot_load_count += 1
            self.owner_model_state_snapshot_load_bytes += int(
                state.get("model_state_snapshot_load_bytes") or 0
            )
            self.owner_model_state_snapshot_load_sec += float(
                state.get("model_state_snapshot_load_sec") or 0.0
            )

    def _apply_owner_learner_telemetry(self, value: Any) -> None:
        if not isinstance(value, Mapping) or not value:
            return
        telemetry = dict(value)
        timing_seen = False
        for key in _OWNER_SEARCH_LEARNER_TIMING_TELEMETRY_KEYS:
            if key in telemetry:
                timing_seen = True
                self.owner_learner_timing_totals[key] = float(
                    self.owner_learner_timing_totals.get(key, 0.0)
                ) + float(telemetry.get(key) or 0.0)
        timing_count = int(
            telemetry.get("compact_owner_search_owner_train_timing_aggregate_count") or 0
        )
        if timing_seen and timing_count <= 0:
            timing_count = 1
        self.owner_learner_timing_aggregate_count += max(0, int(timing_count))
        if self.owner_learner_timing_totals:
            telemetry.update(
                {key: float(total) for key, total in self.owner_learner_timing_totals.items()}
            )
            telemetry["compact_owner_search_owner_train_timing_aggregate_count"] = int(
                self.owner_learner_timing_aggregate_count
            )
        if "compact_owner_search_model_state_snapshot_load_count" in telemetry:
            self.owner_model_state_snapshot_load_count += int(
                telemetry.get("compact_owner_search_model_state_snapshot_load_count") or 0
            )
        if "compact_owner_search_model_state_snapshot_load_bytes" in telemetry:
            self.owner_model_state_snapshot_load_bytes += int(
                telemetry.get("compact_owner_search_model_state_snapshot_load_bytes") or 0
            )
        if "compact_owner_search_model_state_snapshot_load_sec" in telemetry:
            self.owner_model_state_snapshot_load_sec += float(
                telemetry.get("compact_owner_search_model_state_snapshot_load_sec") or 0.0
            )
        if "compact_owner_search_owner_async_learner_request_host_only" in telemetry:
            self.owner_async_learner_request_host_only = bool(
                telemetry.get("compact_owner_search_owner_async_learner_request_host_only")
            )
        if "compact_owner_search_owner_async_learner_request_cuda_tensor_count" in telemetry:
            self.owner_async_learner_request_cuda_tensor_count = int(
                telemetry.get("compact_owner_search_owner_async_learner_request_cuda_tensor_count")
                or 0
            )
        if "compact_owner_search_owner_async_learner_result_host_only" in telemetry:
            self.owner_async_learner_result_host_only = bool(
                telemetry.get("compact_owner_search_owner_async_learner_result_host_only")
            )
        if "compact_owner_search_owner_async_learner_result_cuda_tensor_count" in telemetry:
            self.owner_async_learner_result_cuda_tensor_count = int(
                telemetry.get("compact_owner_search_owner_async_learner_result_cuda_tensor_count")
                or 0
            )
        if "compact_owner_search_owner_async_learner_request_bytes" in telemetry:
            self.owner_async_learner_request_bytes = int(
                telemetry.get("compact_owner_search_owner_async_learner_request_bytes") or 0
            )
        if "compact_owner_search_owner_async_learner_result_bytes" in telemetry:
            self.owner_async_learner_result_bytes = int(
                telemetry.get("compact_owner_search_owner_async_learner_result_bytes") or 0
            )
        if "compact_owner_search_owner_async_learner_worker_pid" in telemetry:
            self.owner_async_learner_worker_pid = int(
                telemetry.get("compact_owner_search_owner_async_learner_worker_pid") or 0
            )
        if "compact_owner_search_owner_async_learner_worker_job_wall_sec" in telemetry:
            self.owner_async_learner_worker_job_wall_sec = float(
                telemetry.get("compact_owner_search_owner_async_learner_worker_job_wall_sec") or 0.0
            )
        if "compact_owner_search_owner_async_learner_payload_prepare_sec" in telemetry:
            self.owner_async_learner_payload_prepare_sec = float(
                telemetry.get("compact_owner_search_owner_async_learner_payload_prepare_sec") or 0.0
            )
        pid_distinct_key = "compact_owner_search_owner_async_learner_worker_pid_distinct_from_owner"
        if pid_distinct_key in telemetry:
            self.owner_async_learner_worker_pid_distinct_from_owner = bool(
                telemetry.get(pid_distinct_key)
            )
        if "compact_owner_search_owner_async_learner_worker_owns_model_state" in telemetry:
            self.owner_async_learner_worker_owns_model_state = bool(
                telemetry.get("compact_owner_search_owner_async_learner_worker_owns_model_state")
            )
        self.owner_learner_telemetry = telemetry

    def _apply_owner_action_feedback(self, value: Any) -> None:
        if not isinstance(value, Mapping) or not value:
            return
        self.owner_action_feedback = _merge_owner_action_feedback_v1(
            self.owner_action_feedback,
            value,
        )

    def _apply_owner_maintenance_payload(self, payload: Mapping[str, Any]) -> None:
        drained_work_item_count = int(
            payload.get("drained_work_item_count", payload.get("drained_count")) or 0
        )
        drained_replay_append_entry_count = int(
            payload.get(
                "drained_replay_append_entry_count",
                payload.get("replay_append_entry_count"),
            )
            or 0
        )
        drained_replay_append_count = int(
            payload.get(
                "drained_replay_append_count",
                payload.get("replay_append_count"),
            )
            or 0
        )
        drained_replay_append_transport_entry_count = int(
            payload.get(
                "drained_replay_append_transport_entry_count",
                payload.get("replay_append_transport_entry_count"),
            )
            or 0
        )
        drained_transition_batch_count = int(
            payload.get(
                "drained_replay_append_transition_batch_count",
                payload.get("replay_append_transition_batch_count"),
            )
            or 0
        )
        drained_transition_batch_entry_count = int(
            payload.get(
                "drained_replay_append_transition_batch_entry_count",
                payload.get("replay_append_transition_batch_entry_count"),
            )
            or 0
        )
        self.owner_maintenance_drained_count += int(drained_work_item_count)
        self.owner_maintenance_drained_work_item_count += int(drained_work_item_count)
        self.owner_maintenance_drained_replay_append_entry_count += int(
            drained_replay_append_entry_count
        )
        self.owner_maintenance_drained_replay_append_transport_entry_count += int(
            drained_replay_append_transport_entry_count
        )
        self.owner_maintenance_drained_replay_append_transition_batch_count += int(
            drained_transition_batch_count
        )
        self.owner_maintenance_drained_replay_append_transition_batch_entry_count += int(
            drained_transition_batch_entry_count
        )
        self.owner_maintenance_drained_replay_append_count += int(drained_replay_append_count)
        self.owner_maintenance_pending_work_count = int(payload.get("pending_count") or 0)
        self.owner_replay_append_count += int(payload.get("replay_append_count") or 0)
        self.owner_learner_update_count = int(
            payload.get("learner_update_count") or self.owner_learner_update_count
        )
        if bool(payload.get("model_owner_ref_returned", False)):
            self.owner_model_owner_ref_returned = True
            self.owner_model_owner_ref_digest = str(
                payload.get("model_owner_ref_digest") or self.owner_model_owner_ref_digest
            )
        if bool(payload.get("search_consumed_learner_update", False)):
            self.owner_search_consumed_learner_update = True
        owner_sample_telemetry = payload.get("owner_sample_telemetry")
        if isinstance(owner_sample_telemetry, Mapping) and owner_sample_telemetry:
            self.owner_sample_telemetry = dict(owner_sample_telemetry)
        self._apply_owner_learner_telemetry(payload.get("owner_learner_telemetry"))
        self._apply_owner_search_worker_state(payload.get("search_worker_state"))
        self._apply_owner_action_feedback(payload.get("owner_action_feedback"))
        async_learner_telemetry = payload.get("owner_async_learner_telemetry")
        if isinstance(async_learner_telemetry, Mapping):
            if "compact_owner_search_owner_async_learner_worker_enabled" in async_learner_telemetry:
                self.owner_async_learner_worker_enabled = bool(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_worker_enabled"
                    )
                )
            if "compact_owner_search_owner_async_learner_worker_kind" in async_learner_telemetry:
                self.owner_async_learner_worker_kind = str(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_worker_kind"
                    )
                    or "none"
                )
            if (
                "compact_owner_search_owner_async_learner_worker_resource_scope"
                in async_learner_telemetry
            ):
                self.owner_async_learner_worker_resource_scope = str(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_worker_resource_scope"
                    )
                    or ""
                )
            if (
                "compact_owner_search_owner_async_learner_worker_resource_id"
                in async_learner_telemetry
            ):
                self.owner_async_learner_worker_resource_id = str(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_worker_resource_id"
                    )
                    or ""
                )
            if (
                "compact_owner_search_owner_async_learner_actor_resource_id"
                in async_learner_telemetry
            ):
                self.owner_async_learner_actor_resource_id = str(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_actor_resource_id"
                    )
                    or ""
                )
            if (
                "compact_owner_search_owner_async_learner_worker_parent_pid"
                in async_learner_telemetry
            ):
                self.owner_async_learner_worker_parent_pid = int(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_worker_parent_pid"
                    )
                    or 0
                )
            if (
                "compact_owner_search_owner_async_learner_resource_distinct_from_owner"
                in async_learner_telemetry
            ):
                self.owner_async_learner_resource_distinct_from_owner = bool(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_resource_distinct_from_owner"
                    )
                )
            hardware_distinct_key = (
                "compact_owner_search_owner_async_learner_hardware_resource_distinct_from_owner"
            )
            if hardware_distinct_key in async_learner_telemetry:
                self.owner_async_learner_hardware_resource_distinct_from_owner = bool(
                    async_learner_telemetry.get(hardware_distinct_key)
                )
            self.owner_async_learner_max_pending = max(
                int(self.owner_async_learner_max_pending),
                int(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_max_pending"
                    )
                    or 0
                ),
            )
            self.owner_async_learner_submit_count = max(
                int(self.owner_async_learner_submit_count),
                int(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_submit_count"
                    )
                    or 0
                ),
            )
            self.owner_async_learner_completed_count = max(
                int(self.owner_async_learner_completed_count),
                int(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_completed_count"
                    )
                    or 0
                ),
            )
            self.owner_async_learner_pending_count = int(
                async_learner_telemetry.get(
                    "compact_owner_search_owner_async_learner_pending_count"
                )
                or 0
            )
            self.owner_async_learner_max_pending_observed = max(
                int(self.owner_async_learner_max_pending_observed),
                int(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_max_pending_observed"
                    )
                    or 0
                ),
            )
            self.owner_async_learner_wait_count = max(
                int(self.owner_async_learner_wait_count),
                int(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_async_learner_wait_count"
                    )
                    or 0
                ),
            )
            self.owner_async_learner_wait_sec = max(
                float(self.owner_async_learner_wait_sec),
                float(
                    async_learner_telemetry.get("compact_owner_search_owner_async_learner_wait_sec")
                    or 0.0
                ),
            )
            self.owner_action_while_async_learner_pending_count = max(
                int(self.owner_action_while_async_learner_pending_count),
                int(
                    async_learner_telemetry.get(
                        "compact_owner_search_owner_action_while_async_learner_pending_count"
                    )
                    or 0
                ),
            )
            self.owner_async_learner_failed = bool(
                self.owner_async_learner_failed
                or async_learner_telemetry.get(
                    "compact_owner_search_owner_async_learner_failed",
                    False,
                )
            )
        self.owner_search_refresh_update_count = max(
            int(self.owner_search_refresh_update_count),
            int(payload.get("search_refresh_update_count") or 0),
        )
        timing = payload.get("timing")
        if isinstance(timing, Mapping):
            self.owner_maintenance_replay_append_sec += float(
                timing.get("replay_append_sec") or 0.0
            )
            self.owner_maintenance_learner_train_sec += float(
                timing.get("learner_train_sec") or 0.0
            )
            self.owner_maintenance_search_refresh_sec += float(
                timing.get("search_refresh_sec") or 0.0
            )
        self._refresh_owner_policy_lag()

    def _owner_model_refresh_due(self, next_train_request: int) -> bool:
        request_index = int(next_train_request)
        if request_index <= 0:
            return False
        if int(self.owner_model_refresh_interval) <= 1:
            return True
        if request_index % int(self.owner_model_refresh_interval) == 0:
            return True
        expected_count = int(self.owner_expected_train_request_count)
        return expected_count > 0 and request_index >= expected_count

    def _acquire_action_result_slot_id(self) -> int:
        if self.action_result_slot_table is None:
            return -1
        return self.action_result_slot_table.acquire()

    def _resolve_action_result_payload(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        wire_payload = dict(payload)
        if not bool(wire_payload.get("compact_owner_search_fixed_action_result_buffer_used")):
            return wire_payload, {
                "compact_owner_search_fixed_action_result_buffer_used": False,
                "compact_owner_search_fixed_action_result_buffer_wire_result_bytes": int(
                    wire_payload.get("result_bytes") or _pickle_size_bytes(wire_payload)
                ),
            }
        slot_table = self.action_result_slot_table
        if slot_table is None:
            raise RuntimeError("fixed action-result slot stub has no local slot table")
        slot_id = int(wire_payload.get("compact_owner_search_fixed_action_result_buffer_slot_id"))
        full_payload = slot_table.read(slot_id)
        wire_result_bytes = int(
            wire_payload.get("result_bytes") or _pickle_size_bytes(wire_payload)
        )
        slot_metadata = dict(slot_table.metadata)
        slot_metadata.update(
            {
                "compact_owner_search_fixed_action_result_buffer_used": True,
                "compact_owner_search_fixed_action_result_buffer_slot_id": int(slot_id),
                "compact_owner_search_fixed_action_result_buffer_wire_result_bytes": int(
                    wire_result_bytes
                ),
                "compact_owner_search_fixed_action_result_buffer_full_result_bytes": int(
                    wire_payload.get(
                        "compact_owner_search_fixed_action_result_buffer_full_result_bytes",
                        full_payload.get("result_bytes") or _pickle_size_bytes(full_payload),
                    )
                ),
            }
        )
        full_payload.update(slot_metadata)
        return full_payload, slot_metadata

    def run(
        self,
        root_batch: CompactRootBatchV1,
        *,
        _allow_action_only_result: bool = False,
    ) -> CompactSearchResultV1:
        if not isinstance(root_batch, CompactRootBatchV1):
            raise TypeError("owner-search slab proxy requires CompactRootBatchV1")
        self._raise_if_owner_maintenance_failed()
        if self.owner_defer_maintenance:
            self._refresh_owner_policy_lag()
            had_inflight_maintenance = (
                self._inflight_owner_maintenance is not None
                and not self.worker.done(self._inflight_owner_maintenance)
            )
            if int(self.owner_async_learner_pending_count) > 0:
                self.owner_action_while_async_learner_pending_count += 1
            if had_inflight_maintenance:
                self.owner_maintenance_actor_steps_while_pending += 1
                if int(self.owner_policy_lag_current) > 0:
                    self.owner_maintenance_actor_steps_while_policy_lagged += 1
            self._poll_owner_maintenance(wait=False)
        else:
            had_inflight_maintenance = False
        self.request_count += 1
        started = time.perf_counter()
        publish_started = time.perf_counter()
        root_slot_ids = self.root_store.publish_root_batch(root_batch)
        publish_sec = _elapsed(publish_started)
        replay_append_entries = (
            tuple(self._pending_owner_replay_append_entries) if self.owner_learning_enabled else ()
        )
        replay_append_logical_entry_count = _owner_replay_append_entry_count_v1(
            replay_append_entries
        )
        replay_append_transport_entry_count = _owner_replay_append_transport_entry_count_v1(
            replay_append_entries
        )
        replay_append_transition_batch_count = _owner_replay_append_transition_batch_count_v1(
            replay_append_entries
        )
        replay_append_transition_batch_entry_count = (
            _owner_replay_append_transition_batch_entry_count_total_v1(replay_append_entries)
        )
        replay_append_transition_legacy_entry_count = (
            _owner_replay_append_transition_legacy_entry_count_v1(replay_append_entries)
        )
        train_steps = 0
        if (
            replay_append_logical_entry_count > 0
            and self.owner_sample_batch_size >= 0
            and self.owner_train_steps > 0
        ):
            previous_train_bucket = int(self.owner_replay_append_submitted_entry_count) // int(
                self.owner_train_interval
            )
            next_train_bucket = int(
                self.owner_replay_append_submitted_entry_count + replay_append_logical_entry_count
            ) // int(self.owner_train_interval)
            train_bucket_delta = int(next_train_bucket - previous_train_bucket)
            if train_bucket_delta > 0:
                train_steps = int(self.owner_train_steps) * train_bucket_delta
        refresh_model = True
        if int(train_steps) > 0:
            refresh_model = self._owner_model_refresh_due(self.owner_train_request_count + 1)
        request = CompactOwnerSearchRequestV1(
            request_id=int(self.request_count),
            actor_step=int(self.request_count - 1),
            root_slot_ids=tuple(root_slot_ids),
            replay_append_entries=replay_append_entries,
            sample_batch_size=(int(self.owner_sample_batch_size) if int(train_steps) > 0 else 0),
            train_steps=int(train_steps),
            policy_version_ref=self.policy_version_ref,
            model_version_ref=self.model_version_ref,
            policy_source=self.policy_source,
            refresh_model=bool(refresh_model),
        )
        submit_started = time.perf_counter()
        handle = (
            self.worker.submit_action(request)
            if self.owner_defer_maintenance
            else self.worker.submit(request)
        )
        submit_sec = _elapsed(submit_started)
        wait_started = time.perf_counter()
        payload = self.worker.result(handle)
        parent_wait_sec = _elapsed(wait_started)
        if had_inflight_maintenance:
            self.owner_action_served_before_maintenance_count += 1
        if self.owner_defer_maintenance:
            self._poll_owner_maintenance(wait=False)
        if replay_append_entries:
            del self._pending_owner_replay_append_entries[: len(replay_append_entries)]
            self.owner_replay_append_request_count += 1
            self.owner_replay_append_submitted_entry_count += int(replay_append_logical_entry_count)
            self.owner_replay_append_submitted_transport_entry_count += int(
                replay_append_transport_entry_count
            )
            self.owner_replay_append_transition_batch_count += int(
                replay_append_transition_batch_count
            )
            self.owner_replay_append_transition_batch_entry_count += int(
                replay_append_transition_batch_entry_count
            )
            self.owner_replay_append_transition_legacy_entry_count += int(
                replay_append_transition_legacy_entry_count
            )
        if int(train_steps) > 0:
            self.owner_train_request_count += 1
            if bool(request.refresh_model):
                self.owner_model_refresh_request_count += 1
            else:
                self.owner_model_refresh_skipped_count += 1
            self.owner_submitted_learner_update_count += int(train_steps)
            self._refresh_owner_policy_lag()
        if self.owner_defer_maintenance:
            self.owner_maintenance_pending_work_count = int(
                payload.get("owner_maintenance_pending_work_count")
                or self.owner_maintenance_pending_work_count
            )
            staged_work_count = int(payload.get("owner_maintenance_staged_work_count") or 0)
            self.owner_maintenance_staged_work_item_count += int(staged_work_count)
            self._submit_owner_maintenance_drain_if_due(
                train_steps=int(train_steps),
                staged_work_count=int(staged_work_count),
            )
        else:
            self.owner_replay_append_count += int(payload.get("replay_append_count") or 0)
            self.owner_learner_update_count = int(
                payload.get("learner_update_count") or self.owner_learner_update_count
            )
            if bool(payload.get("model_owner_ref_returned", False)):
                self.owner_model_owner_ref_returned = True
                self.owner_model_owner_ref_digest = str(
                    payload.get("model_owner_ref_digest") or self.owner_model_owner_ref_digest
                )
            if bool(payload.get("search_consumed_learner_update", False)):
                self.owner_search_consumed_learner_update = True
            owner_sample_telemetry = payload.get("owner_sample_telemetry")
            if isinstance(owner_sample_telemetry, Mapping) and owner_sample_telemetry:
                self.owner_sample_telemetry = dict(owner_sample_telemetry)
            self._apply_owner_learner_telemetry(payload.get("owner_learner_telemetry"))
            self._apply_owner_search_worker_state(payload.get("search_worker_state"))
            self._apply_owner_action_feedback(payload.get("owner_action_feedback"))
            self.owner_search_refresh_update_count = max(
                int(self.owner_search_refresh_update_count),
                int(payload.get("search_refresh_update_count") or 0),
            )
            self._refresh_owner_policy_lag()
            payload_timing = payload.get("timing")
            if isinstance(payload_timing, Mapping):
                self.owner_maintenance_replay_append_sec += float(
                    payload_timing.get("replay_append_sec") or 0.0
                )
                self.owner_maintenance_learner_train_sec += float(
                    payload_timing.get("learner_train_sec") or 0.0
                )
                self.owner_maintenance_search_refresh_sec += float(
                    payload_timing.get("search_refresh_sec") or 0.0
                )
        search_result = compact_search_result_v1_from_owner_search_payload(
            root_batch,
            payload,
        )
        metadata = dict(search_result.metadata)
        owner_materializes_replay = bool(
            metadata.get("compact_owner_search_owner_materializes_replay", False)
        )
        action_only_result = bool(metadata.get("compact_owner_search_action_only_result", False))
        parent_slab_commits_replay = not bool(owner_materializes_replay or action_only_result)
        timing = payload.get("timing")
        if not isinstance(timing, Mapping):
            timing = {}
        worker_metadata = dict(self.worker.metadata)
        if (
            self.owner_defer_maintenance
            and not bool(payload.get("search_result_payload"))
            and not bool(_allow_action_only_result)
        ):
            raise RuntimeError(
                "deferred owner-search run() would return an action-only synthetic "
                "search result; use run_action_step() so replay stays owner-side"
            )
        metadata.update(
            {
                "compact_owner_search_parent_publish_sec": float(publish_sec),
                "compact_owner_search_parent_submit_sec": float(submit_sec),
                "compact_owner_search_parent_wait_sec": float(parent_wait_sec),
                "compact_owner_search_parent_wall_sec": float(_elapsed(started)),
                "compact_owner_search_boundary_kind": self.boundary_kind,
                "compact_owner_search_parent_slab_commits_replay": (parent_slab_commits_replay),
                "compact_owner_search_worker_wall_sec": float(timing.get("wall_sec") or 0.0),
                "compact_owner_search_worker_root_resolve_sec": float(
                    timing.get("root_resolve_sec") or 0.0
                ),
                "compact_owner_search_worker_search_sec": float(timing.get("search_sec") or 0.0),
                "compact_owner_search_worker_replay_append_sec": float(
                    timing.get("replay_append_sec") or 0.0
                ),
                "compact_owner_search_worker_learner_train_sec": float(
                    timing.get("learner_train_sec") or 0.0
                ),
                "compact_owner_search_worker_search_refresh_sec": float(
                    timing.get("search_refresh_sec") or 0.0
                ),
                "compact_owner_search_owner_replay_append_enabled": bool(
                    self.owner_replay_append_enabled
                ),
                "compact_owner_search_owner_learning_enabled": bool(self.owner_learning_enabled),
                "compact_owner_search_owner_sample_batch_size": int(self.owner_sample_batch_size),
                "compact_owner_search_owner_train_steps": int(self.owner_train_steps),
                "compact_owner_search_owner_train_interval": int(self.owner_train_interval),
                "compact_owner_search_owner_model_refresh_interval": int(
                    self.owner_model_refresh_interval
                ),
                "compact_owner_search_owner_expected_train_request_count": int(
                    self.owner_expected_train_request_count
                ),
                "compact_owner_search_owner_defer_maintenance": bool(self.owner_defer_maintenance),
                "compact_owner_search_owner_loop_schema_id": (
                    COMPACT_OWNER_SEARCH_PRIORITY_LOOP_SCHEMA_ID
                ),
                "compact_owner_search_owner_loop_kind": str(
                    worker_metadata.get("compact_owner_search_owner_loop_kind")
                    or COMPACT_OWNER_SEARCH_PRIORITY_LOOP_KIND
                ),
                "compact_owner_search_owner_loop_persistent": True,
                "compact_owner_search_owner_action_priority_enabled": bool(
                    worker_metadata.get(
                        "compact_owner_search_owner_action_priority_enabled",
                        True,
                    )
                ),
                "compact_owner_search_owner_background_maintenance_thread": bool(
                    worker_metadata.get(
                        "compact_owner_search_owner_background_maintenance_thread",
                        False,
                    )
                ),
                "compact_owner_search_owner_background_overlap_enabled": bool(
                    worker_metadata.get(
                        "compact_owner_search_owner_background_overlap_enabled",
                        False,
                    )
                ),
                "compact_owner_search_owner_action_request_count": int(
                    worker_metadata.get(
                        "compact_owner_search_owner_action_request_count",
                        0,
                    )
                ),
                "compact_owner_search_owner_maintenance_request_count": int(
                    worker_metadata.get(
                        "compact_owner_search_owner_maintenance_request_count",
                        0,
                    )
                ),
                "compact_owner_search_owner_run_request_count": int(
                    worker_metadata.get(
                        "compact_owner_search_owner_run_request_count",
                        0,
                    )
                ),
                "compact_owner_search_owner_replay_append_staged_entry_count": int(
                    self.owner_replay_append_staged_entry_count
                ),
                "compact_owner_search_owner_replay_append_suppressed_entry_count": int(
                    self.owner_replay_append_suppressed_entry_count
                ),
                "compact_owner_search_owner_replay_append_submitted_entry_count": int(
                    self.owner_replay_append_submitted_entry_count
                ),
                "compact_owner_search_owner_replay_append_staged_transport_entry_count": int(
                    self.owner_replay_append_staged_transport_entry_count
                ),
                "compact_owner_search_owner_replay_append_suppressed_transport_entry_count": int(
                    self.owner_replay_append_suppressed_transport_entry_count
                ),
                "compact_owner_search_owner_replay_append_submitted_transport_entry_count": int(
                    self.owner_replay_append_submitted_transport_entry_count
                ),
                "compact_owner_search_owner_replay_transport_entry_count": int(
                    self.owner_replay_append_submitted_transport_entry_count
                ),
                "compact_owner_search_owner_replay_transport_kind": (
                    COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
                    if int(self.owner_replay_append_transition_batch_count) > 0
                    else "per_transition_entry_v1"
                ),
                "compact_owner_search_owner_replay_transition_batch_enabled": (
                    int(self.owner_replay_append_transition_batch_count) > 0
                ),
                "compact_owner_search_owner_replay_transition_batch_count": int(
                    self.owner_replay_append_transition_batch_count
                ),
                "compact_owner_search_owner_replay_transition_batch_transition_count": int(
                    self.owner_replay_append_transition_batch_entry_count
                ),
                "compact_owner_search_owner_replay_transition_legacy_entry_count": int(
                    self.owner_replay_append_transition_legacy_entry_count
                ),
                "compact_owner_search_transition_batch_transport_enabled": (
                    int(self.owner_replay_append_transition_batch_count) > 0
                ),
                "compact_owner_search_transition_batch_transport_kind": (
                    COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
                    if int(self.owner_replay_append_transition_batch_count) > 0
                    else "per_transition_entry_v1"
                ),
                "compact_owner_search_transition_batch_schema_id": (
                    COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
                    if int(self.owner_replay_append_transition_batch_count) > 0
                    else ""
                ),
                "compact_owner_search_transition_batch_count": int(
                    self.owner_replay_append_transition_batch_count
                ),
                "compact_owner_search_transition_batch_entry_count": int(
                    self.owner_replay_append_transition_batch_entry_count
                ),
                "compact_owner_search_transition_batch_transport_entry_count": int(
                    self.owner_replay_append_submitted_transport_entry_count
                ),
                "compact_owner_search_transition_batch_fallback_count": 0,
                "compact_owner_search_transition_batch_fallback_reason": "none",
                "compact_owner_search_transition_batch_pending_count": int(
                    _owner_replay_append_transition_batch_entry_count_total_v1(
                        self._pending_owner_replay_append_entries
                    )
                ),
                "compact_owner_search_owner_replay_append_request_count": int(
                    self.owner_replay_append_request_count
                ),
                "compact_owner_search_owner_replay_append_count": int(
                    self.owner_replay_append_count
                ),
                "compact_owner_search_replay_append_entry_count": int(
                    self.owner_replay_append_submitted_entry_count
                ),
                "compact_owner_search_replay_append_count": int(self.owner_replay_append_count),
                "compact_owner_search_owner_train_request_count": int(
                    self.owner_train_request_count
                ),
                "compact_owner_search_owner_model_refresh_request_count": int(
                    self.owner_model_refresh_request_count
                ),
                "compact_owner_search_owner_model_refresh_skipped_count": int(
                    self.owner_model_refresh_skipped_count
                ),
                "compact_owner_search_owner_submitted_learner_update_count": int(
                    self.owner_submitted_learner_update_count
                ),
                "compact_owner_search_owner_learner_update_count": int(
                    self.owner_learner_update_count
                ),
                "compact_owner_search_model_owner_ref_returned": bool(
                    self.owner_model_owner_ref_returned
                ),
                "compact_owner_search_model_owner_ref_digest": str(
                    self.owner_model_owner_ref_digest
                ),
                "compact_owner_search_model_state_snapshot_load_count": int(
                    self.owner_model_state_snapshot_load_count
                ),
                "compact_owner_search_model_state_snapshot_load_bytes": int(
                    self.owner_model_state_snapshot_load_bytes
                ),
                "compact_owner_search_model_state_snapshot_load_sec": float(
                    self.owner_model_state_snapshot_load_sec
                ),
                "compact_owner_search_consumed_learner_update": bool(
                    self.owner_search_consumed_learner_update
                ),
                "compact_owner_search_search_refresh_update_count": int(
                    self.owner_search_refresh_update_count
                ),
                "compact_owner_search_owner_pending_replay_append_entry_count": int(
                    _owner_replay_append_entry_count_v1(self._pending_owner_replay_append_entries)
                ),
                "compact_owner_search_owner_pending_replay_append_transport_entry_count": int(
                    _owner_replay_append_transport_entry_count_v1(
                        self._pending_owner_replay_append_entries
                    )
                ),
                "compact_owner_search_owner_maintenance_drain_request_count": int(
                    self.owner_maintenance_drain_request_count
                ),
                "compact_owner_search_owner_maintenance_staged_work_item_count": int(
                    self.owner_maintenance_staged_work_item_count
                ),
                "compact_owner_search_owner_maintenance_drained_count": int(
                    self.owner_maintenance_drained_count
                ),
                "compact_owner_search_owner_maintenance_drained_work_item_count": int(
                    self.owner_maintenance_drained_work_item_count
                ),
                "compact_owner_search_owner_maintenance_drained_replay_append_entry_count": int(
                    self.owner_maintenance_drained_replay_append_entry_count
                ),
                "compact_owner_search_owner_maintenance_drained_replay_append_transport_entry_count": int(
                    self.owner_maintenance_drained_replay_append_transport_entry_count
                ),
                "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_count": int(
                    self.owner_maintenance_drained_replay_append_transition_batch_count
                ),
                "compact_owner_search_owner_maintenance_drained_replay_append_transition_batch_entry_count": int(
                    self.owner_maintenance_drained_replay_append_transition_batch_entry_count
                ),
                "compact_owner_search_owner_maintenance_drained_replay_append_count": int(
                    self.owner_maintenance_drained_replay_append_count
                ),
                "compact_owner_search_owner_maintenance_pending_work_count": int(
                    self.owner_maintenance_pending_work_count
                ),
                "compact_owner_search_owner_maintenance_inflight": (
                    self._inflight_owner_maintenance is not None
                ),
                "compact_owner_search_owner_maintenance_final_drain_sec": float(
                    self.owner_maintenance_final_drain_sec
                ),
                "compact_owner_search_owner_maintenance_coalescing_kind": (
                    "eager_append_or_train_boundary_v1" if self.owner_defer_maintenance else ""
                ),
                "compact_owner_search_owner_maintenance_coalesced_skip_count": int(
                    self.owner_maintenance_coalesced_skip_count
                ),
                "compact_owner_search_owner_maintenance_eager_append_drain_count": int(
                    self.owner_maintenance_eager_append_drain_count
                ),
                "compact_owner_search_owner_policy_lag_current": int(self.owner_policy_lag_current),
                "compact_owner_search_owner_policy_lag_max": int(self.owner_policy_lag_max),
                "compact_owner_search_owner_maintenance_actor_steps_while_pending": int(
                    self.owner_maintenance_actor_steps_while_pending
                ),
                "compact_owner_search_owner_maintenance_actor_steps_while_policy_lagged": int(
                    self.owner_maintenance_actor_steps_while_policy_lagged
                ),
                "compact_owner_search_owner_action_while_maintenance_pending_count": int(
                    self.owner_maintenance_actor_steps_while_pending
                ),
                "compact_owner_search_owner_action_while_policy_lagged_count": int(
                    self.owner_maintenance_actor_steps_while_policy_lagged
                ),
                "compact_owner_search_owner_action_served_before_maintenance_count": int(
                    self.owner_action_served_before_maintenance_count
                ),
                "compact_owner_search_owner_fifo_blocked_action_count": int(
                    self.owner_fifo_blocked_action_count
                ),
                "compact_owner_search_owner_maintenance_failed": bool(
                    self._owner_maintenance_failure
                ),
                **(
                    {}
                    if self.action_result_slot_table is None
                    else self.action_result_slot_table.metadata
                ),
                **dict(self.owner_action_feedback),
            }
        )
        search_result = CompactSearchResultV1(
            root_index=search_result.root_index,
            env_row=search_result.env_row,
            player=search_result.player,
            policy_env_id=search_result.policy_env_id,
            selected_action=search_result.selected_action,
            visit_policy=search_result.visit_policy,
            root_value=search_result.root_value,
            raw_visit_counts=search_result.raw_visit_counts,
            predicted_value=search_result.predicted_value,
            predicted_policy_logits=search_result.predicted_policy_logits,
            metadata=metadata,
        )
        self.last_result_payload = dict(payload)
        self.last_search_result_payload_bytes = int(
            payload.get("search_result_payload_bytes", 0) or 0
        )
        return search_result

    def run_action_step(self, root_batch: CompactRootBatchV1) -> CompactSearchActionStepV1:
        search_result = self.run(root_batch, _allow_action_only_result=True)
        owner_materializes_replay = bool(
            search_result.metadata.get("compact_owner_search_owner_materializes_replay")
        )
        owner_handle = str(
            search_result.metadata.get("compact_owner_search_replay_payload_handle")
            or search_result.metadata.get("replay_payload_handle")
            or ""
        )
        if owner_materializes_replay:
            if not owner_handle:
                raise ReplayCompatibilityError(
                    "owner-search action-only result is missing replay payload handle"
                )
            handle = owner_handle
        else:
            handle = f"compact-owner-search-slab-proxy:{self._replay_payload_counter}"
            self._replay_payload_counter += 1
        action_step = compact_search_action_step_v1_from_result(
            search_result,
            replay_payload_handle=handle,
            metadata={
                "replay_payload_origin": f"compact_owner_search_slab_proxy:{handle}",
                "compact_owner_search_two_phase_action_step": True,
                **dict(search_result.metadata),
            },
        )
        action_step.metadata["search_replay_payload_digest"] = (
            compact_search_deferred_replay_payload_digest_v1(handle)
        )
        action_step.metadata["search_replay_payload_digest_deferred"] = True
        if owner_materializes_replay:
            action_step.metadata["compact_owner_search_owner_materializes_replay"] = True
            action_step.metadata["compact_owner_search_action_only_result"] = True
            return action_step
        replay_payload = compact_search_replay_payload_v1_from_result(
            search_result,
            replay_payload_handle=handle,
            metadata={
                "replay_payload_origin": f"compact_owner_search_slab_proxy:{handle}",
                "compact_owner_search_two_phase_replay_payload": True,
                **dict(search_result.metadata),
            },
        )
        if handle in self._pending_replay_payloads:
            raise ReplayCompatibilityError("duplicate owner-search replay payload handle")
        self._pending_replay_payloads[handle] = replay_payload
        return action_step

    def _owner_root_search_transaction_metadata_v1(
        self,
        *,
        transaction_id: int,
        action_identity_verified: bool = False,
    ) -> dict[str, Any]:
        pending_count = int(
            sum(
                1
                for pending in self._pending_action_dispatches.values()
                if int(pending.root_search_transaction_id) > 0
            )
        )
        return {
            "compact_owner_root_search_transaction_boundary_supported": True,
            "compact_owner_root_search_transaction_requested": True,
            "compact_owner_root_search_transaction_used": True,
            "compact_owner_root_search_transaction_schema_id": (
                COMPACT_OWNER_ROOT_SEARCH_TRANSACTION_SCHEMA_ID
            ),
            "compact_owner_root_search_transaction_id": int(transaction_id),
            "compact_owner_root_search_transaction_begin_count": int(
                self.owner_root_search_transaction_begin_count
            ),
            "compact_owner_root_search_transaction_submit_count": int(
                self.owner_root_search_transaction_submit_count
            ),
            "compact_owner_root_search_transaction_resolve_count": int(
                self.owner_root_search_transaction_resolve_count
            ),
            "compact_owner_root_search_transaction_pending_count": int(pending_count),
            "compact_owner_root_search_transaction_max_pending_count": int(
                self.owner_root_search_transaction_max_pending_count
            ),
            "compact_owner_root_search_transaction_parent_root_request_build_count": int(
                self.owner_root_search_transaction_parent_root_request_build_count
            ),
            "compact_owner_root_search_transaction_parent_root_request_stored": False,
            "compact_owner_root_search_transaction_parent_compact_batch_stored": False,
            "compact_owner_root_search_transaction_parent_rebuild_count": int(
                self.owner_root_search_transaction_parent_rebuild_count
            ),
            "compact_owner_root_search_transaction_owner_root_request_build_count": int(
                self.owner_root_search_transaction_owner_root_request_build_count
            ),
            "compact_owner_root_search_transaction_owner_root_request_build_sec": float(
                self.owner_root_search_transaction_owner_root_request_build_sec
            ),
            "compact_owner_root_search_transaction_owner_root_store_publish_count": int(
                getattr(self.root_store, "root_build_request_publish_count", 0)
            ),
            "compact_owner_root_search_transaction_frame_generation_verified": bool(
                self.owner_root_search_transaction_frame_generation_verified_count > 0
            ),
            "compact_owner_root_search_transaction_frame_digest_verified": bool(
                self.owner_root_search_transaction_frame_digest_verified_count > 0
            ),
            "compact_owner_root_search_transaction_action_identity_verified": bool(
                action_identity_verified
                or self.owner_root_search_transaction_action_identity_verified_count > 0
            ),
            "compact_owner_root_search_transaction_proxy_transition_closure_used": bool(
                self.owner_proxy_transition_closure_closed_count > 0
            ),
            "compact_owner_root_search_transaction_applied_action_mismatch_count": int(
                self.owner_proxy_transition_closure_applied_action_mismatch_count
            ),
        }

    def _owner_root_action_context_handle_metadata_v1(
        self,
        handle: CompactOwnerRootActionContextHandleV1 | None,
    ) -> dict[str, Any]:
        context_digest = "" if handle is None else str(handle.context_digest)
        return {
            "compact_owner_root_action_context_handle_used": handle is not None,
            "compact_owner_root_action_context_handle_schema_id": (
                "" if handle is None else str(handle.schema_id)
            ),
            "compact_owner_root_action_context_handle_id": (
                0 if handle is None else int(handle.context_id)
            ),
            "compact_owner_root_action_context_transaction_id": (
                0 if handle is None else int(handle.transaction_id)
            ),
            "compact_owner_root_action_context_dispatch_id": (
                0 if handle is None else int(handle.dispatch_id)
            ),
            "compact_owner_root_action_context_root_count": (
                0 if handle is None else int(handle.root_count)
            ),
            "compact_owner_root_action_context_active_root_count": (
                0 if handle is None else int(handle.active_root_count)
            ),
            "compact_owner_root_action_context_context_digest": context_digest,
            "compact_owner_root_action_context_owner_store_count": int(
                self.owner_root_action_context_owner_store_count
            ),
            "compact_owner_root_action_context_owner_resolve_count": int(
                self.owner_root_action_context_owner_resolve_count
            ),
            "compact_owner_root_action_context_owner_release_count": int(
                self.owner_root_action_context_owner_release_count
            ),
            "compact_owner_root_action_context_owner_pending_count": int(
                len(self._root_action_contexts)
            ),
            "compact_owner_root_action_context_owner_max_pending_count": int(
                self.owner_root_action_context_owner_max_pending_count
            ),
            "compact_owner_root_action_context_owner_digest_verified": bool(
                self.owner_root_action_context_owner_digest_verified_count > 0
            ),
            "compact_owner_search_pending_root_action_context_stored": False,
            "compact_owner_search_action_dispatch_pending_root_action_context_stored": False,
            "compact_owner_search_action_dispatch_pending_root_action_context_store_count": int(
                self.owner_action_dispatch_pending_root_action_context_store_count
            ),
            (
                "compact_owner_search_action_dispatch_pending_root_action_context_"
                "avoided_count"
            ): int(self.owner_action_dispatch_pending_root_action_context_avoided_count),
            "compact_owner_search_parent_action_context_validation_count": 0,
            "compact_owner_search_owner_action_context_validation_count": int(
                self.owner_root_action_context_owner_resolve_count
            ),
            "compact_owner_root_search_transaction_parent_root_action_context_stored": False,
            "compact_owner_root_search_transaction_parent_root_action_context_store_count": 0,
            "compact_owner_root_search_transaction_parent_root_action_context_array_bytes": 0,
            "compact_owner_root_search_transaction_parent_root_action_context_field_count": 0,
        }

    def _store_owner_root_action_context_handle(
        self,
        root_action_context: CompactRootActionContextV1,
        *,
        transaction_id: int,
        dispatch_id: int,
    ) -> CompactOwnerRootActionContextHandleV1:
        self._next_root_action_context_id += 1
        context_id = int(self._next_root_action_context_id)
        active_root_count = int(
            np.asarray(root_action_context.active_root_index, dtype=np.int64).reshape(-1).size
        )
        context_digest = _owner_root_action_context_digest_v1(root_action_context)
        handle = CompactOwnerRootActionContextHandleV1(
            schema_id=COMPACT_OWNER_ROOT_ACTION_CONTEXT_HANDLE_SCHEMA_ID,
            context_id=context_id,
            transaction_id=int(transaction_id),
            dispatch_id=int(dispatch_id),
            batch_size=int(root_action_context.batch_size),
            player_count=int(root_action_context.player_count),
            root_count=int(root_action_context.root_count),
            active_root_count=int(active_root_count),
            context_digest=str(context_digest),
            metadata={},
        )
        handle.metadata.update(self._owner_root_action_context_handle_metadata_v1(handle))
        self._root_action_contexts[context_id] = root_action_context
        self.owner_root_action_context_owner_store_count += 1
        self.owner_root_action_context_owner_max_pending_count = max(
            int(self.owner_root_action_context_owner_max_pending_count),
            int(len(self._root_action_contexts)),
        )
        handle.metadata.update(self._owner_root_action_context_handle_metadata_v1(handle))
        return handle

    def _pop_owner_root_action_context_for_handle(
        self,
        handle: CompactOwnerRootActionContextHandleV1,
    ) -> CompactRootActionContextV1:
        context = self._root_action_contexts.pop(int(handle.context_id), None)
        if context is None:
            raise ReplayCompatibilityError("owner root-action context handle is missing")
        digest = _owner_root_action_context_digest_v1(context)
        if digest != str(handle.context_digest):
            raise ReplayCompatibilityError("owner root-action context handle digest mismatch")
        self.owner_root_action_context_owner_resolve_count += 1
        self.owner_root_action_context_owner_release_count += 1
        self.owner_root_action_context_owner_digest_verified_count += 1
        return context

    def submit_owner_root_search_transaction_from_step_frame_slot(
        self,
        compact_batch: Any,
        *,
        search_lane: str,
        metadata: Mapping[str, Any] | None = None,
        copy_observation: bool = False,
        resident_host_observation_stub: bool = True,
        close_previous_transition: bool = False,
        max_entries_per_batch: int = 0,
        policy_source: str = "",
    ) -> CompactOwnerRootSearchTransactionDispatchV1:
        """Open an owner/proxy root-search transaction from a step-frame slot."""

        if self._pending_action_dispatches:
            raise ReplayCompatibilityError(
                "owner root-search transaction supports one pending dispatch"
            )
        self.owner_root_search_transaction_begin_count += 1
        self._next_owner_root_search_transaction_id += 1
        transaction_id = int(self._next_owner_root_search_transaction_id)
        root_request_started = time.perf_counter()
        request_metadata = {
            **dict(metadata or {}),
            "compact_owner_root_search_transaction_requested": True,
            "compact_owner_root_search_transaction_used": True,
            "compact_owner_root_search_transaction_schema_id": (
                COMPACT_OWNER_ROOT_SEARCH_TRANSACTION_SCHEMA_ID
            ),
            "compact_owner_root_search_transaction_id": int(transaction_id),
            "compact_owner_root_search_transaction_begin_count": int(
                self.owner_root_search_transaction_begin_count
            ),
            "compact_owner_root_search_transaction_parent_root_request_build_count": 0,
            "compact_owner_root_search_transaction_parent_root_request_stored": False,
            "compact_owner_root_search_transaction_parent_compact_batch_stored": False,
            "compact_owner_root_search_transaction_parent_rebuild_count": 0,
        }
        root_build_request = _owner_root_search_transaction_request_from_step_frame_slot_v1(
            compact_batch,
            search_lane=str(search_lane),
            metadata=request_metadata,
            copy_observation=bool(copy_observation),
            resident_host_observation_stub=bool(resident_host_observation_stub),
        )
        root_request_sec = _elapsed(root_request_started)
        self.owner_root_search_transaction_owner_root_request_build_count += 1
        self.owner_root_search_transaction_owner_root_request_build_sec += float(
            root_request_sec
        )
        self.owner_root_search_transaction_frame_generation_verified_count += 1
        self.owner_root_search_transaction_frame_digest_verified_count += 1
        commit_timing = {
            "compact_rollout_slab_replay_index_rows_build_sec": 0.0,
            "compact_rollout_slab_commit_action_check_sec": 0.0,
            "compact_rollout_slab_replay_index_rows_store_sec": 0.0,
            "compact_rollout_slab_owner_replay_transition_stage_sec": 0.0,
            "compact_rollout_slab_owner_replay_transition_stage_count": 0.0,
            "compact_rollout_slab_owner_replay_transition_stage_entry_count": 0.0,
            "compact_rollout_slab_owner_replay_transition_stage_transport_entry_count": 0.0,
            "compact_rollout_slab_owner_replay_transition_batch_submit_sec": 0.0,
        }
        if bool(close_previous_transition):
            commit_timing = self.stage_owner_proxy_transition_from_root_build_request(
                root_build_request,
                max_entries_per_batch=int(max_entries_per_batch),
                policy_source=str(policy_source),
            )
        dispatch_handle = self.submit_action_step_from_root_build_request(root_build_request)
        self.owner_root_search_transaction_submit_count += 1
        pending = self._pending_action_dispatches.get(int(dispatch_handle.dispatch_id))
        if pending is None:
            raise ReplayCompatibilityError("owner root-search transaction lost pending dispatch")
        root_action_context_handle = pending.root_action_context_handle
        if pending is not None:
            self._pending_action_dispatches[int(dispatch_handle.dispatch_id)] = replace(
                pending,
                root_search_transaction_id=int(transaction_id),
            )
            root_action_context_handle = replace(
                root_action_context_handle,
                transaction_id=int(transaction_id),
                metadata={
                    **dict(root_action_context_handle.metadata),
                    "compact_owner_root_action_context_transaction_id": int(transaction_id),
                },
            )
            self._pending_action_dispatches[int(dispatch_handle.dispatch_id)] = replace(
                self._pending_action_dispatches[int(dispatch_handle.dispatch_id)],
                root_action_context_handle=root_action_context_handle,
            )
        self.owner_root_search_transaction_max_pending_count = max(
            int(self.owner_root_search_transaction_max_pending_count),
            int(
                sum(
                    1
                    for pending_dispatch in self._pending_action_dispatches.values()
                    if int(pending_dispatch.root_search_transaction_id) > 0
                )
            ),
        )
        transaction_metadata = {
            str(key): value
            for key, value in dict(root_build_request.metadata or {}).items()
            if value is None or isinstance(value, (bool, int, float, str))
        }
        transaction_metadata.update(
            self._owner_root_search_transaction_metadata_v1(
                transaction_id=int(transaction_id)
            )
        )
        transaction_metadata.update(
            self._owner_root_action_context_handle_metadata_v1(root_action_context_handle)
        )
        transaction_metadata[
            "compact_owner_root_search_transaction_owner_root_request_build_sec"
        ] = float(root_request_sec)
        dispatch_handle.metadata.update(transaction_metadata)
        return CompactOwnerRootSearchTransactionDispatchV1(
            schema_id=COMPACT_OWNER_ROOT_SEARCH_TRANSACTION_SCHEMA_ID,
            transaction_id=int(transaction_id),
            action_dispatch_handle=dispatch_handle,
            root_action_context_handle=root_action_context_handle,
            commit_timing={str(key): float(value) for key, value in commit_timing.items()},
            metadata=transaction_metadata,
        )

    def submit_action_step_from_root_build_request(
        self,
        root_build_request: CompactRootBuildRequestV1,
    ) -> CompactOwnerActionDispatchHandleV1:
        if not isinstance(root_build_request, CompactRootBuildRequestV1):
            raise TypeError("root_build_request must be CompactRootBuildRequestV1")
        if self._pending_action_dispatches:
            raise ReplayCompatibilityError(
                "owner action dispatch handle supports one pending dispatch"
            )
        publish_root_build_request = getattr(
            self.root_store,
            "publish_root_build_request",
            None,
        )
        if not callable(publish_root_build_request):
            raise ReplayCompatibilityError(
                "owner-search root build request requires a direct root store"
            )
        self._raise_if_owner_maintenance_failed()
        if self.owner_defer_maintenance:
            self._refresh_owner_policy_lag()
            had_inflight_maintenance = (
                self._inflight_owner_maintenance is not None
                and not self.worker.done(self._inflight_owner_maintenance)
            )
            if int(self.owner_async_learner_pending_count) > 0:
                self.owner_action_while_async_learner_pending_count += 1
            if had_inflight_maintenance:
                self.owner_maintenance_actor_steps_while_pending += 1
                if int(self.owner_policy_lag_current) > 0:
                    self.owner_maintenance_actor_steps_while_policy_lagged += 1
            self._poll_owner_maintenance(wait=False)
        else:
            had_inflight_maintenance = False
        self.request_count += 1
        started = time.perf_counter()
        publish_started = time.perf_counter()
        root_slot_ids = publish_root_build_request(root_build_request)
        root_action_context = compact_root_action_context_v1_from_request(
            root_build_request
        )
        publish_sec = _elapsed(publish_started)
        replay_append_entries = (
            tuple(self._pending_owner_replay_append_entries) if self.owner_learning_enabled else ()
        )
        replay_append_logical_entry_count = _owner_replay_append_entry_count_v1(
            replay_append_entries
        )
        replay_append_transport_entry_count = _owner_replay_append_transport_entry_count_v1(
            replay_append_entries
        )
        replay_append_transition_batch_count = _owner_replay_append_transition_batch_count_v1(
            replay_append_entries
        )
        replay_append_transition_batch_entry_count = (
            _owner_replay_append_transition_batch_entry_count_total_v1(replay_append_entries)
        )
        replay_append_transition_legacy_entry_count = (
            _owner_replay_append_transition_legacy_entry_count_v1(replay_append_entries)
        )
        train_steps = 0
        if (
            replay_append_logical_entry_count > 0
            and self.owner_sample_batch_size >= 0
            and self.owner_train_steps > 0
        ):
            previous_train_bucket = int(self.owner_replay_append_submitted_entry_count) // int(
                self.owner_train_interval
            )
            next_train_bucket = int(
                self.owner_replay_append_submitted_entry_count + replay_append_logical_entry_count
            ) // int(self.owner_train_interval)
            train_bucket_delta = int(next_train_bucket - previous_train_bucket)
            if train_bucket_delta > 0:
                train_steps = int(self.owner_train_steps) * train_bucket_delta
        refresh_model = True
        if int(train_steps) > 0:
            refresh_model = self._owner_model_refresh_due(self.owner_train_request_count + 1)
        action_result_slot_id = self._acquire_action_result_slot_id()
        request = CompactOwnerSearchRequestV1(
            request_id=int(self.request_count),
            actor_step=int(self.request_count - 1),
            root_slot_ids=tuple(root_slot_ids),
            replay_append_entries=replay_append_entries,
            sample_batch_size=(int(self.owner_sample_batch_size) if int(train_steps) > 0 else 0),
            train_steps=int(train_steps),
            policy_version_ref=self.policy_version_ref,
            model_version_ref=self.model_version_ref,
            policy_source=self.policy_source,
            refresh_model=bool(refresh_model),
            action_result_slot_id=int(action_result_slot_id),
        )
        submit_started = time.perf_counter()
        worker_handle = (
            self.worker.submit_action(request)
            if self.owner_defer_maintenance
            else self.worker.submit(request)
        )
        completed_at_submit = bool(self.worker.done(worker_handle))
        submit_sec = _elapsed(submit_started)
        if completed_at_submit:
            self.owner_action_dispatch_handle_completed_at_submit_count += 1
        self._next_action_dispatch_id += 1
        dispatch_id = int(self._next_action_dispatch_id)
        dispatch_handle = CompactOwnerActionDispatchHandleV1(
            schema_id=COMPACT_OWNER_ACTION_DISPATCH_HANDLE_SCHEMA_ID,
            dispatch_id=dispatch_id,
            request_id=int(request.request_id),
            actor_step=int(request.actor_step),
            root_slot_count=int(len(root_slot_ids)),
            action_result_slot_id=int(action_result_slot_id),
            metadata={
                "compact_owner_search_action_dispatch_handle_submitted": True,
                "compact_owner_search_action_dispatch_handle_submit_no_wait": True,
                "compact_owner_search_action_dispatch_handle_result_wait_in_submit": False,
                "compact_owner_search_action_dispatch_handle_completed_at_submit": bool(
                    completed_at_submit
                ),
                "compact_owner_search_parent_publish_sec": float(publish_sec),
                "compact_owner_search_parent_submit_sec": float(submit_sec),
            },
        )
        root_action_context_handle = self._store_owner_root_action_context_handle(
            root_action_context,
            transaction_id=0,
            dispatch_id=int(dispatch_id),
        )
        dispatch_handle.metadata.update(
            self._owner_root_action_context_handle_metadata_v1(root_action_context_handle)
        )
        self._pending_action_dispatches[dispatch_id] = _PendingOwnerActionDispatchV1(
            handle=dispatch_handle,
            root_action_context_handle=root_action_context_handle,
            request=request,
            worker_handle=worker_handle,
            started=float(started),
            publish_sec=float(publish_sec),
            submit_sec=float(submit_sec),
            had_inflight_maintenance=bool(had_inflight_maintenance),
            replay_append_entries=replay_append_entries,
            replay_append_logical_entry_count=int(replay_append_logical_entry_count),
            replay_append_transport_entry_count=int(replay_append_transport_entry_count),
            replay_append_transition_batch_count=int(replay_append_transition_batch_count),
            replay_append_transition_batch_entry_count=int(
                replay_append_transition_batch_entry_count
            ),
            replay_append_transition_legacy_entry_count=int(
                replay_append_transition_legacy_entry_count
            ),
            train_steps=int(train_steps),
        )
        self.owner_action_dispatch_pending_root_build_request_avoided_count += 1
        self.owner_action_dispatch_pending_root_action_context_avoided_count += 1
        self.owner_action_dispatch_handle_submit_count += 1
        dispatch_handle.metadata.update(
            self._owner_root_action_context_handle_metadata_v1(root_action_context_handle)
        )
        self.owner_action_dispatch_handle_pending_count = len(self._pending_action_dispatches)
        self.owner_action_dispatch_handle_max_pending_count = max(
            int(self.owner_action_dispatch_handle_max_pending_count),
            int(self.owner_action_dispatch_handle_pending_count),
        )
        return dispatch_handle

    def resolve_action_step_handle(
        self,
        action_dispatch_handle: CompactOwnerActionDispatchHandleV1,
        *,
        sync_wrapper: bool = False,
    ) -> CompactSearchActionStepV1:
        if not isinstance(action_dispatch_handle, CompactOwnerActionDispatchHandleV1):
            raise TypeError("action_dispatch_handle must be CompactOwnerActionDispatchHandleV1")
        dispatch_id = int(action_dispatch_handle.dispatch_id)
        pending = self._pending_action_dispatches.pop(dispatch_id, None)
        if pending is None:
            raise ReplayCompatibilityError("unknown owner action dispatch handle")
        root_action_context = self._pop_owner_root_action_context_for_handle(
            pending.root_action_context_handle
        )
        self.owner_action_dispatch_handle_pending_count = len(self._pending_action_dispatches)
        wait_started = time.perf_counter()
        wire_payload = self.worker.result(pending.worker_handle)
        parent_wait_sec = _elapsed(wait_started)
        self.owner_action_dispatch_handle_result_wait_sec += float(parent_wait_sec)
        payload, action_result_slot_metadata = self._resolve_action_result_payload(wire_payload)
        if pending.had_inflight_maintenance:
            self.owner_action_served_before_maintenance_count += 1
        if self.owner_defer_maintenance:
            self._poll_owner_maintenance(wait=False)
        replay_append_entries = pending.replay_append_entries
        if replay_append_entries:
            del self._pending_owner_replay_append_entries[: len(replay_append_entries)]
            self.owner_replay_append_request_count += 1
            self.owner_replay_append_submitted_entry_count += int(
                pending.replay_append_logical_entry_count
            )
            self.owner_replay_append_submitted_transport_entry_count += int(
                pending.replay_append_transport_entry_count
            )
            self.owner_replay_append_transition_batch_count += int(
                pending.replay_append_transition_batch_count
            )
            self.owner_replay_append_transition_batch_entry_count += int(
                pending.replay_append_transition_batch_entry_count
            )
            self.owner_replay_append_transition_legacy_entry_count += int(
                pending.replay_append_transition_legacy_entry_count
            )
        request = pending.request
        train_steps = int(pending.train_steps)
        if train_steps > 0:
            self.owner_train_request_count += 1
            if bool(request.refresh_model):
                self.owner_model_refresh_request_count += 1
            else:
                self.owner_model_refresh_skipped_count += 1
            self.owner_submitted_learner_update_count += int(train_steps)
            self._refresh_owner_policy_lag()
        if self.owner_defer_maintenance:
            self.owner_maintenance_pending_work_count = int(
                payload.get("owner_maintenance_pending_work_count")
                or self.owner_maintenance_pending_work_count
            )
            staged_work_count = int(payload.get("owner_maintenance_staged_work_count") or 0)
            self.owner_maintenance_staged_work_item_count += int(staged_work_count)
            self._submit_owner_maintenance_drain_if_due(
                train_steps=int(train_steps),
                staged_work_count=int(staged_work_count),
            )
        else:
            self.owner_replay_append_count += int(payload.get("replay_append_count") or 0)
            self.owner_learner_update_count = int(
                payload.get("learner_update_count") or self.owner_learner_update_count
            )
            if bool(payload.get("model_owner_ref_returned", False)):
                self.owner_model_owner_ref_returned = True
                self.owner_model_owner_ref_digest = str(
                    payload.get("model_owner_ref_digest") or self.owner_model_owner_ref_digest
                )
            if bool(payload.get("search_consumed_learner_update", False)):
                self.owner_search_consumed_learner_update = True
            owner_sample_telemetry = payload.get("owner_sample_telemetry")
            if isinstance(owner_sample_telemetry, Mapping) and owner_sample_telemetry:
                self.owner_sample_telemetry = dict(owner_sample_telemetry)
            self._apply_owner_learner_telemetry(payload.get("owner_learner_telemetry"))
            self._apply_owner_search_worker_state(payload.get("search_worker_state"))
            self._apply_owner_action_feedback(payload.get("owner_action_feedback"))
            self.owner_search_refresh_update_count = max(
                int(self.owner_search_refresh_update_count),
                int(payload.get("search_refresh_update_count") or 0),
            )
            self._refresh_owner_policy_lag()
            payload_timing = payload.get("timing")
            if isinstance(payload_timing, Mapping):
                self.owner_maintenance_replay_append_sec += float(
                    payload_timing.get("replay_append_sec") or 0.0
                )
                self.owner_maintenance_learner_train_sec += float(
                    payload_timing.get("learner_train_sec") or 0.0
                )
                self.owner_maintenance_search_refresh_sec += float(
                    payload_timing.get("search_refresh_sec") or 0.0
                )
        timing = payload.get("timing")
        if not isinstance(timing, Mapping):
            timing = {}
        worker_metadata = dict(self.worker.metadata)
        payload_search_metadata = payload.get("search_result_metadata")
        if not isinstance(payload_search_metadata, Mapping):
            payload_search_metadata = {}
        search_worker_state = payload.get("search_worker_state")
        search_worker_state = (
            dict(search_worker_state) if isinstance(search_worker_state, Mapping) else {}
        )
        owner_action_feedback = payload.get("owner_action_feedback")
        owner_action_feedback = (
            dict(owner_action_feedback)
            if isinstance(owner_action_feedback, Mapping)
            else _empty_owner_action_feedback_v1()
        )
        self.owner_action_dispatch_handle_resolve_count += 1
        if bool(sync_wrapper):
            self.owner_action_dispatch_handle_sync_wrapper_count += 1
        metadata = dict(payload_search_metadata)
        metadata.update(
            {
                "compact_owner_search_slab_proxy": True,
                "compact_owner_search_action_dispatch_handle_used": True,
                "compact_owner_search_action_dispatch_handle_schema_id": (
                    COMPACT_OWNER_ACTION_DISPATCH_HANDLE_SCHEMA_ID
                ),
                "compact_owner_search_action_dispatch_handle_id": int(dispatch_id),
                "compact_owner_search_action_dispatch_handle_submit_no_wait": True,
                "compact_owner_search_action_dispatch_handle_sync_wrapper": bool(sync_wrapper),
                "compact_owner_search_action_dispatch_handle_sync_wrapper_count": int(
                    self.owner_action_dispatch_handle_sync_wrapper_count
                ),
                "compact_owner_search_action_dispatch_handle_completed_at_submit_count": int(
                    self.owner_action_dispatch_handle_completed_at_submit_count
                ),
                "compact_owner_search_action_dispatch_handle_submit_count": int(
                    self.owner_action_dispatch_handle_submit_count
                ),
                "compact_owner_search_action_dispatch_handle_resolve_count": int(
                    self.owner_action_dispatch_handle_resolve_count
                ),
                "compact_owner_search_action_dispatch_handle_pending_count": int(
                    self.owner_action_dispatch_handle_pending_count
                ),
                "compact_owner_search_action_dispatch_handle_max_pending_count": int(
                    self.owner_action_dispatch_handle_max_pending_count
                ),
                "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count": int(
                    self.owner_action_dispatch_handle_result_wait_in_submit_count
                ),
                "compact_owner_search_action_dispatch_handle_result_wait_sec": float(
                    self.owner_action_dispatch_handle_result_wait_sec
                ),
                "compact_owner_search_action_dispatch_pending_root_build_request_stored": False,
                "compact_owner_search_action_dispatch_pending_root_build_request_store_count": int(
                    self.owner_action_dispatch_pending_root_build_request_store_count
                ),
                "compact_owner_search_action_dispatch_pending_root_build_request_avoided_count": int(
                    self.owner_action_dispatch_pending_root_build_request_avoided_count
                ),
                "compact_owner_search_action_dispatch_pending_root_action_context_stored": False,
                "compact_owner_search_action_dispatch_pending_root_action_context_store_count": int(
                    self.owner_action_dispatch_pending_root_action_context_store_count
                ),
                (
                    "compact_owner_search_action_dispatch_pending_root_action_context_"
                    "avoided_count"
                ): int(self.owner_action_dispatch_pending_root_action_context_avoided_count),
                "compact_owner_search_parent_publish_sec": float(pending.publish_sec),
                "compact_owner_search_parent_submit_sec": float(pending.submit_sec),
                "compact_owner_search_parent_wait_sec": float(parent_wait_sec),
                "compact_owner_search_parent_wall_sec": float(_elapsed(pending.started)),
                "compact_owner_search_boundary_kind": self.boundary_kind,
                "compact_owner_search_parent_slab_commits_replay": False,
                "compact_owner_search_fixed_action_result_buffer_requested": bool(
                    self.action_result_slot_table is not None
                ),
                "compact_owner_search_result_schema_id": str(payload.get("schema_id") or ""),
                "compact_owner_search_owner_kind": str(payload.get("owner_kind") or ""),
                "compact_owner_search_owner_pid": int(payload.get("owner_pid") or 0),
                "compact_owner_search_request_id": int(payload.get("request_id") or 0),
                "compact_owner_search_root_slot_count": int(payload.get("root_slot_count") or 0),
                "compact_owner_search_active_root_count": int(
                    payload.get("active_root_count") or 0
                ),
                "compact_owner_search_request_bytes": int(payload.get("request_bytes") or 0),
                "compact_owner_search_result_bytes": int(payload.get("result_bytes") or 0),
                "compact_owner_search_request_cuda_tensor_count": int(
                    payload.get("request_cuda_tensor_count") or 0
                ),
                "compact_owner_search_result_cuda_tensor_count": int(
                    payload.get("result_cuda_tensor_count") or 0
                ),
                "compact_owner_search_root_observation_bytes_sent": int(
                    payload.get("root_observation_bytes_sent") or 0
                ),
                "compact_owner_search_worker_wall_sec": float(timing.get("wall_sec") or 0.0),
                "compact_owner_search_worker_root_resolve_sec": float(
                    timing.get("root_resolve_sec") or 0.0
                ),
                "compact_owner_search_worker_search_sec": float(timing.get("search_sec") or 0.0),
                "compact_owner_search_worker_replay_append_sec": float(
                    timing.get("replay_append_sec") or 0.0
                ),
                "compact_owner_search_worker_learner_train_sec": float(
                    timing.get("learner_train_sec") or 0.0
                ),
                "compact_owner_search_worker_search_refresh_sec": float(
                    timing.get("search_refresh_sec") or 0.0
                ),
                "compact_owner_search_action_only_result": True,
                "compact_owner_search_owner_materializes_replay": True,
                "compact_owner_search_parent_reconstructed_search_result": False,
                "compact_owner_search_inner_two_phase_action_step": bool(
                    payload.get("inner_two_phase_action_step", False)
                ),
                "compact_owner_search_inner_device_replay_payload_deferred": bool(
                    payload.get("inner_device_replay_payload_deferred", False)
                ),
                "compact_owner_search_use_inner_two_phase_device_replay": bool(
                    metadata.get(
                        "compact_owner_search_use_inner_two_phase_device_replay",
                        payload.get("use_inner_two_phase_device_replay", False),
                    )
                ),
                "compact_owner_search_replay_payload_handle": str(
                    payload.get("replay_payload_handle")
                    or metadata.get("replay_payload_handle")
                    or ""
                ),
                "compact_owner_search_model_state_bytes": int(
                    payload.get("model_state_bytes") or 0
                ),
                "compact_owner_search_model_state_return_count": int(
                    payload.get("model_state_return_count") or 0
                ),
                "compact_owner_search_model_state_snapshot_return_count": int(
                    payload.get("model_state_snapshot_return_count") or 0
                ),
                "compact_owner_search_search_result_payload_bytes": int(
                    payload.get("search_result_payload_bytes") or 0
                ),
                "compact_owner_search_search_result_payload_transport_kind": "action_only_owner_cached_replay_v1",
                "compact_owner_search_search_result_payload_json_safe": True,
                "compact_owner_search_selected_action_bytes": int(
                    payload.get("search_selected_action_bytes") or 0
                ),
                "compact_owner_search_visit_policy_bytes": int(
                    payload.get("search_visit_policy_bytes") or 0
                ),
                "compact_owner_search_root_value_bytes": int(
                    payload.get("search_root_value_bytes") or 0
                ),
                "compact_owner_search_optional_array_bytes": int(
                    payload.get("search_optional_array_bytes") or 0
                ),
                "compact_owner_search_worker_owns_search_state": bool(
                    payload.get("worker_owns_search_state", False)
                ),
                "compact_owner_search_worker_owns_replay_state": bool(
                    payload.get("worker_owns_replay_state", False)
                ),
                "compact_owner_search_worker_owns_model_state": bool(
                    payload.get("worker_owns_model_state", False)
                ),
                "compact_owner_search_consumed_learner_update": bool(
                    payload.get("search_consumed_learner_update", False)
                ),
                "compact_owner_search_search_refresh_update_count": int(
                    payload.get("search_refresh_update_count") or 0
                ),
                "compact_owner_search_model_state_snapshot_load_count": (
                    1 if bool(search_worker_state.get("model_state_snapshot_loaded", False)) else 0
                ),
                "compact_owner_search_model_state_snapshot_load_bytes": int(
                    search_worker_state.get("model_state_snapshot_load_bytes") or 0
                ),
                "compact_owner_search_model_state_snapshot_load_sec": float(
                    search_worker_state.get("model_state_snapshot_load_sec") or 0.0
                ),
                "compact_owner_search_replay_append_entry_count": int(
                    payload.get("replay_append_entry_count") or 0
                ),
                "compact_owner_search_replay_append_transport_entry_count": int(
                    payload.get("replay_append_transport_entry_count") or 0
                ),
                "compact_owner_search_replay_append_transition_batch_count": int(
                    payload.get("replay_append_transition_batch_count") or 0
                ),
                "compact_owner_search_replay_append_transition_batch_entry_count": int(
                    payload.get("replay_append_transition_batch_entry_count") or 0
                ),
                "compact_owner_search_replay_append_count": int(
                    payload.get("replay_append_count") or 0
                ),
                "compact_owner_search_learner_update_count": int(
                    payload.get("learner_update_count") or 0
                ),
                "compact_owner_search_model_owner_ref_returned": bool(
                    payload.get("model_owner_ref_returned", False)
                ),
                "compact_owner_search_model_owner_ref_digest": str(
                    payload.get("model_owner_ref_digest") or ""
                ),
                "compact_owner_search_owner_sample_telemetry": dict(
                    payload.get("owner_sample_telemetry")
                    if isinstance(payload.get("owner_sample_telemetry"), Mapping)
                    else {}
                ),
                "compact_owner_search_owner_learner_telemetry": dict(
                    payload.get("owner_learner_telemetry")
                    if isinstance(payload.get("owner_learner_telemetry"), Mapping)
                    else {}
                ),
                "compact_owner_search_direct_root_build_request_handoff": True,
                "compact_owner_search_direct_root_parent_build_avoided": True,
                "compact_owner_search_direct_root_parent_build_call_count": 0,
                "compact_owner_search_direct_root_parent_build_sec": 0.0,
                "compact_owner_search_parent_compact_root_batch_objects_sent": 0,
                **self._owner_root_action_context_handle_metadata_v1(
                    pending.root_action_context_handle
                ),
                **action_result_slot_metadata,
                **worker_metadata,
                **owner_action_feedback,
            }
        )
        action_step = compact_action_step_v1_from_owner_search_payload_and_root_request(
            root_action_context,
            payload,
            metadata=metadata,
        )
        if int(pending.root_search_transaction_id) > 0:
            self.owner_root_search_transaction_resolve_count += 1
            self.owner_root_search_transaction_action_identity_verified_count += 1
            action_step.metadata.update(
                self._owner_root_search_transaction_metadata_v1(
                    transaction_id=int(pending.root_search_transaction_id),
                    action_identity_verified=True,
                )
            )
            action_step.metadata.update(
                self._owner_root_action_context_handle_metadata_v1(
                    pending.root_action_context_handle
                )
            )
        if self.owner_proxy_transition_closure_requested_count > 0:
            self._record_owner_proxy_action_frame(
                record_index=int(request.actor_step),
                action_step=action_step,
            )
            action_step.metadata.update(self.metadata)
        self.last_result_payload = dict(payload)
        self.last_search_result_payload_bytes = int(
            payload.get("search_result_payload_bytes", 0) or 0
        )
        return action_step

    def run_action_step_from_root_build_request(
        self,
        root_build_request: CompactRootBuildRequestV1,
    ) -> CompactSearchActionStepV1:
        dispatch_handle = self.submit_action_step_from_root_build_request(root_build_request)
        return self.resolve_action_step_handle(dispatch_handle, sync_wrapper=True)

    def _run_action_step_from_root_build_request_sync_legacy(
        self,
        root_build_request: CompactRootBuildRequestV1,
    ) -> CompactSearchActionStepV1:
        if not isinstance(root_build_request, CompactRootBuildRequestV1):
            raise TypeError("root_build_request must be CompactRootBuildRequestV1")
        publish_root_build_request = getattr(
            self.root_store,
            "publish_root_build_request",
            None,
        )
        if not callable(publish_root_build_request):
            raise ReplayCompatibilityError(
                "owner-search root build request requires a direct root store"
            )
        self._raise_if_owner_maintenance_failed()
        if self.owner_defer_maintenance:
            self._refresh_owner_policy_lag()
            had_inflight_maintenance = (
                self._inflight_owner_maintenance is not None
                and not self.worker.done(self._inflight_owner_maintenance)
            )
            if int(self.owner_async_learner_pending_count) > 0:
                self.owner_action_while_async_learner_pending_count += 1
            if had_inflight_maintenance:
                self.owner_maintenance_actor_steps_while_pending += 1
                if int(self.owner_policy_lag_current) > 0:
                    self.owner_maintenance_actor_steps_while_policy_lagged += 1
            self._poll_owner_maintenance(wait=False)
        else:
            had_inflight_maintenance = False
        self.request_count += 1
        started = time.perf_counter()
        publish_started = time.perf_counter()
        root_slot_ids = publish_root_build_request(root_build_request)
        publish_sec = _elapsed(publish_started)
        replay_append_entries = (
            tuple(self._pending_owner_replay_append_entries) if self.owner_learning_enabled else ()
        )
        replay_append_logical_entry_count = _owner_replay_append_entry_count_v1(
            replay_append_entries
        )
        replay_append_transport_entry_count = _owner_replay_append_transport_entry_count_v1(
            replay_append_entries
        )
        replay_append_transition_batch_count = _owner_replay_append_transition_batch_count_v1(
            replay_append_entries
        )
        replay_append_transition_batch_entry_count = (
            _owner_replay_append_transition_batch_entry_count_total_v1(replay_append_entries)
        )
        replay_append_transition_legacy_entry_count = (
            _owner_replay_append_transition_legacy_entry_count_v1(replay_append_entries)
        )
        train_steps = 0
        if (
            replay_append_logical_entry_count > 0
            and self.owner_sample_batch_size >= 0
            and self.owner_train_steps > 0
        ):
            previous_train_bucket = int(self.owner_replay_append_submitted_entry_count) // int(
                self.owner_train_interval
            )
            next_train_bucket = int(
                self.owner_replay_append_submitted_entry_count + replay_append_logical_entry_count
            ) // int(self.owner_train_interval)
            train_bucket_delta = int(next_train_bucket - previous_train_bucket)
            if train_bucket_delta > 0:
                train_steps = int(self.owner_train_steps) * train_bucket_delta
        refresh_model = True
        if int(train_steps) > 0:
            refresh_model = self._owner_model_refresh_due(self.owner_train_request_count + 1)
        action_result_slot_id = self._acquire_action_result_slot_id()
        request = CompactOwnerSearchRequestV1(
            request_id=int(self.request_count),
            actor_step=int(self.request_count - 1),
            root_slot_ids=tuple(root_slot_ids),
            replay_append_entries=replay_append_entries,
            sample_batch_size=(int(self.owner_sample_batch_size) if int(train_steps) > 0 else 0),
            train_steps=int(train_steps),
            policy_version_ref=self.policy_version_ref,
            model_version_ref=self.model_version_ref,
            policy_source=self.policy_source,
            refresh_model=bool(refresh_model),
            action_result_slot_id=int(action_result_slot_id),
        )
        submit_started = time.perf_counter()
        handle = (
            self.worker.submit_action(request)
            if self.owner_defer_maintenance
            else self.worker.submit(request)
        )
        submit_sec = _elapsed(submit_started)
        wait_started = time.perf_counter()
        wire_payload = self.worker.result(handle)
        parent_wait_sec = _elapsed(wait_started)
        payload, action_result_slot_metadata = self._resolve_action_result_payload(wire_payload)
        if had_inflight_maintenance:
            self.owner_action_served_before_maintenance_count += 1
        if self.owner_defer_maintenance:
            self._poll_owner_maintenance(wait=False)
        if replay_append_entries:
            del self._pending_owner_replay_append_entries[: len(replay_append_entries)]
            self.owner_replay_append_request_count += 1
            self.owner_replay_append_submitted_entry_count += int(replay_append_logical_entry_count)
            self.owner_replay_append_submitted_transport_entry_count += int(
                replay_append_transport_entry_count
            )
            self.owner_replay_append_transition_batch_count += int(
                replay_append_transition_batch_count
            )
            self.owner_replay_append_transition_batch_entry_count += int(
                replay_append_transition_batch_entry_count
            )
            self.owner_replay_append_transition_legacy_entry_count += int(
                replay_append_transition_legacy_entry_count
            )
        if int(train_steps) > 0:
            self.owner_train_request_count += 1
            if bool(request.refresh_model):
                self.owner_model_refresh_request_count += 1
            else:
                self.owner_model_refresh_skipped_count += 1
            self.owner_submitted_learner_update_count += int(train_steps)
            self._refresh_owner_policy_lag()
        if self.owner_defer_maintenance:
            self.owner_maintenance_pending_work_count = int(
                payload.get("owner_maintenance_pending_work_count")
                or self.owner_maintenance_pending_work_count
            )
            staged_work_count = int(payload.get("owner_maintenance_staged_work_count") or 0)
            self.owner_maintenance_staged_work_item_count += int(staged_work_count)
            self._submit_owner_maintenance_drain_if_due(
                train_steps=int(train_steps),
                staged_work_count=int(staged_work_count),
            )
        else:
            self.owner_replay_append_count += int(payload.get("replay_append_count") or 0)
            self.owner_learner_update_count = int(
                payload.get("learner_update_count") or self.owner_learner_update_count
            )
            if bool(payload.get("model_owner_ref_returned", False)):
                self.owner_model_owner_ref_returned = True
                self.owner_model_owner_ref_digest = str(
                    payload.get("model_owner_ref_digest") or self.owner_model_owner_ref_digest
                )
            if bool(payload.get("search_consumed_learner_update", False)):
                self.owner_search_consumed_learner_update = True
            owner_sample_telemetry = payload.get("owner_sample_telemetry")
            if isinstance(owner_sample_telemetry, Mapping) and owner_sample_telemetry:
                self.owner_sample_telemetry = dict(owner_sample_telemetry)
            self._apply_owner_learner_telemetry(payload.get("owner_learner_telemetry"))
            self._apply_owner_search_worker_state(payload.get("search_worker_state"))
            self._apply_owner_action_feedback(payload.get("owner_action_feedback"))
            self.owner_search_refresh_update_count = max(
                int(self.owner_search_refresh_update_count),
                int(payload.get("search_refresh_update_count") or 0),
            )
            self._refresh_owner_policy_lag()
            payload_timing = payload.get("timing")
            if isinstance(payload_timing, Mapping):
                self.owner_maintenance_replay_append_sec += float(
                    payload_timing.get("replay_append_sec") or 0.0
                )
                self.owner_maintenance_learner_train_sec += float(
                    payload_timing.get("learner_train_sec") or 0.0
                )
                self.owner_maintenance_search_refresh_sec += float(
                    payload_timing.get("search_refresh_sec") or 0.0
                )
        timing = payload.get("timing")
        if not isinstance(timing, Mapping):
            timing = {}
        worker_metadata = dict(self.worker.metadata)
        payload_search_metadata = payload.get("search_result_metadata")
        if not isinstance(payload_search_metadata, Mapping):
            payload_search_metadata = {}
        search_worker_state = payload.get("search_worker_state")
        search_worker_state = (
            dict(search_worker_state) if isinstance(search_worker_state, Mapping) else {}
        )
        owner_action_feedback = payload.get("owner_action_feedback")
        owner_action_feedback = (
            dict(owner_action_feedback)
            if isinstance(owner_action_feedback, Mapping)
            else _empty_owner_action_feedback_v1()
        )
        metadata = dict(payload_search_metadata)
        metadata.update(
            {
                "compact_owner_search_slab_proxy": True,
                "compact_owner_search_parent_publish_sec": float(publish_sec),
                "compact_owner_search_parent_submit_sec": float(submit_sec),
                "compact_owner_search_parent_wait_sec": float(parent_wait_sec),
                "compact_owner_search_parent_wall_sec": float(_elapsed(started)),
                "compact_owner_search_boundary_kind": self.boundary_kind,
                "compact_owner_search_parent_slab_commits_replay": False,
                "compact_owner_search_fixed_action_result_buffer_requested": bool(
                    self.action_result_slot_table is not None
                ),
                "compact_owner_search_result_schema_id": str(payload.get("schema_id") or ""),
                "compact_owner_search_owner_kind": str(payload.get("owner_kind") or ""),
                "compact_owner_search_owner_pid": int(payload.get("owner_pid") or 0),
                "compact_owner_search_request_id": int(payload.get("request_id") or 0),
                "compact_owner_search_root_slot_count": int(payload.get("root_slot_count") or 0),
                "compact_owner_search_active_root_count": int(
                    payload.get("active_root_count") or 0
                ),
                "compact_owner_search_request_bytes": int(payload.get("request_bytes") or 0),
                "compact_owner_search_result_bytes": int(payload.get("result_bytes") or 0),
                "compact_owner_search_request_cuda_tensor_count": int(
                    payload.get("request_cuda_tensor_count") or 0
                ),
                "compact_owner_search_result_cuda_tensor_count": int(
                    payload.get("result_cuda_tensor_count") or 0
                ),
                "compact_owner_search_root_observation_bytes_sent": int(
                    payload.get("root_observation_bytes_sent") or 0
                ),
                "compact_owner_search_worker_wall_sec": float(timing.get("wall_sec") or 0.0),
                "compact_owner_search_worker_root_resolve_sec": float(
                    timing.get("root_resolve_sec") or 0.0
                ),
                "compact_owner_search_worker_search_sec": float(timing.get("search_sec") or 0.0),
                "compact_owner_search_worker_replay_append_sec": float(
                    timing.get("replay_append_sec") or 0.0
                ),
                "compact_owner_search_worker_learner_train_sec": float(
                    timing.get("learner_train_sec") or 0.0
                ),
                "compact_owner_search_worker_search_refresh_sec": float(
                    timing.get("search_refresh_sec") or 0.0
                ),
                "compact_owner_search_action_only_result": True,
                "compact_owner_search_owner_materializes_replay": True,
                "compact_owner_search_parent_reconstructed_search_result": False,
                "compact_owner_search_inner_two_phase_action_step": bool(
                    payload.get("inner_two_phase_action_step", False)
                ),
                "compact_owner_search_inner_device_replay_payload_deferred": bool(
                    payload.get("inner_device_replay_payload_deferred", False)
                ),
                "compact_owner_search_use_inner_two_phase_device_replay": bool(
                    metadata.get(
                        "compact_owner_search_use_inner_two_phase_device_replay",
                        payload.get("use_inner_two_phase_device_replay", False),
                    )
                ),
                "compact_owner_search_replay_payload_handle": str(
                    payload.get("replay_payload_handle")
                    or metadata.get("replay_payload_handle")
                    or ""
                ),
                "compact_owner_search_model_state_bytes": int(
                    payload.get("model_state_bytes") or 0
                ),
                "compact_owner_search_model_state_return_count": int(
                    payload.get("model_state_return_count") or 0
                ),
                "compact_owner_search_model_state_snapshot_return_count": int(
                    payload.get("model_state_snapshot_return_count") or 0
                ),
                "compact_owner_search_search_result_payload_bytes": int(
                    payload.get("search_result_payload_bytes") or 0
                ),
                "compact_owner_search_search_result_payload_transport_kind": "action_only_owner_cached_replay_v1",
                "compact_owner_search_search_result_payload_json_safe": True,
                "compact_owner_search_selected_action_bytes": int(
                    payload.get("search_selected_action_bytes") or 0
                ),
                "compact_owner_search_visit_policy_bytes": int(
                    payload.get("search_visit_policy_bytes") or 0
                ),
                "compact_owner_search_root_value_bytes": int(
                    payload.get("search_root_value_bytes") or 0
                ),
                "compact_owner_search_optional_array_bytes": int(
                    payload.get("search_optional_array_bytes") or 0
                ),
                "compact_owner_search_worker_owns_search_state": bool(
                    payload.get("worker_owns_search_state", False)
                ),
                "compact_owner_search_worker_owns_replay_state": bool(
                    payload.get("worker_owns_replay_state", False)
                ),
                "compact_owner_search_worker_owns_model_state": bool(
                    payload.get("worker_owns_model_state", False)
                ),
                "compact_owner_search_consumed_learner_update": bool(
                    payload.get("search_consumed_learner_update", False)
                ),
                "compact_owner_search_search_refresh_update_count": int(
                    payload.get("search_refresh_update_count") or 0
                ),
                "compact_owner_search_model_state_snapshot_load_count": (
                    1 if bool(search_worker_state.get("model_state_snapshot_loaded", False)) else 0
                ),
                "compact_owner_search_model_state_snapshot_load_bytes": int(
                    search_worker_state.get("model_state_snapshot_load_bytes") or 0
                ),
                "compact_owner_search_model_state_snapshot_load_sec": float(
                    search_worker_state.get("model_state_snapshot_load_sec") or 0.0
                ),
                "compact_owner_search_replay_append_entry_count": int(
                    payload.get("replay_append_entry_count") or 0
                ),
                "compact_owner_search_replay_append_transport_entry_count": int(
                    payload.get("replay_append_transport_entry_count") or 0
                ),
                "compact_owner_search_replay_append_transition_batch_count": int(
                    payload.get("replay_append_transition_batch_count") or 0
                ),
                "compact_owner_search_replay_append_transition_batch_entry_count": int(
                    payload.get("replay_append_transition_batch_entry_count") or 0
                ),
                "compact_owner_search_replay_append_count": int(
                    payload.get("replay_append_count") or 0
                ),
                "compact_owner_search_learner_update_count": int(
                    payload.get("learner_update_count") or 0
                ),
                "compact_owner_search_model_owner_ref_returned": bool(
                    payload.get("model_owner_ref_returned", False)
                ),
                "compact_owner_search_model_owner_ref_digest": str(
                    payload.get("model_owner_ref_digest") or ""
                ),
                "compact_owner_search_owner_sample_telemetry": dict(
                    payload.get("owner_sample_telemetry")
                    if isinstance(payload.get("owner_sample_telemetry"), Mapping)
                    else {}
                ),
                "compact_owner_search_owner_learner_telemetry": dict(
                    payload.get("owner_learner_telemetry")
                    if isinstance(payload.get("owner_learner_telemetry"), Mapping)
                    else {}
                ),
                "compact_owner_search_direct_root_build_request_handoff": True,
                "compact_owner_search_direct_root_parent_build_avoided": True,
                "compact_owner_search_direct_root_parent_build_call_count": 0,
                "compact_owner_search_direct_root_parent_build_sec": 0.0,
                "compact_owner_search_parent_compact_root_batch_objects_sent": 0,
                **action_result_slot_metadata,
                **worker_metadata,
                **owner_action_feedback,
            }
        )
        action_step = compact_action_step_v1_from_owner_search_payload_and_root_request(
            root_build_request,
            payload,
            metadata=metadata,
        )
        if self.owner_proxy_transition_closure_requested_count > 0:
            self._record_owner_proxy_action_frame(
                record_index=int(request.actor_step),
                action_step=action_step,
            )
            action_step.metadata.update(self.metadata)
        self.last_result_payload = dict(payload)
        self.last_search_result_payload_bytes = int(
            payload.get("search_result_payload_bytes", 0) or 0
        )
        return action_step

    def flush_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactSearchReplayPayloadV1:
        handle = str(replay_payload_handle)
        if not handle:
            raise ReplayCompatibilityError("replay_payload_handle must be non-empty")
        payload = self._pending_replay_payloads.pop(handle, None)
        if payload is None:
            raise ReplayCompatibilityError("unknown owner-search replay payload handle")
        return payload

    def flush_device_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactDeviceSearchReplayPayloadV1:
        import torch

        payload = self.flush_replay_payload(replay_payload_handle)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        visit_policy = torch.as_tensor(payload.visit_policy, dtype=torch.float32, device=device)
        root_value = torch.as_tensor(payload.root_value, dtype=torch.float32, device=device)
        raw_visit_counts = (
            None
            if payload.raw_visit_counts is None
            else torch.as_tensor(payload.raw_visit_counts, dtype=torch.float32, device=device)
        )
        predicted_value = (
            None
            if payload.predicted_value is None
            else torch.as_tensor(payload.predicted_value, dtype=torch.float32, device=device)
        )
        predicted_policy_logits = (
            None
            if payload.predicted_policy_logits is None
            else torch.as_tensor(
                payload.predicted_policy_logits,
                dtype=torch.float32,
                device=device,
            )
        )
        metadata = dict(payload.metadata)
        metadata.update(
            {
                "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                "phase": "replay_critical_device",
                "device_replay_payload": True,
                "device_replay_payload_device": str(device),
                "host_search_payload_fallback_allowed": False,
                "compact_owner_search_device_replay_payload": True,
            }
        )
        return CompactDeviceSearchReplayPayloadV1(
            replay_payload_handle=str(replay_payload_handle),
            root_index=payload.root_index.astype(np.int32, copy=True),
            env_row=payload.env_row.astype(np.int32, copy=True),
            player=payload.player.astype(np.int16, copy=True),
            policy_env_id=payload.policy_env_id.astype(np.int64, copy=True),
            visit_policy=visit_policy,
            root_value=root_value,
            raw_visit_counts=raw_visit_counts,
            predicted_value=predicted_value,
            predicted_policy_logits=predicted_policy_logits,
            metadata=metadata,
        )

    def close(self) -> None:
        error: BaseException | None = None
        try:
            if self._pending_action_dispatches:
                raise ReplayCompatibilityError("owner action dispatch handle pending at close")
            if self.owner_defer_maintenance:
                self.drain_owner_maintenance(wait=True)
        except BaseException as exc:
            error = exc
        finally:
            self._pending_action_dispatches.clear()
            self.owner_action_dispatch_handle_pending_count = 0
            self._pending_replay_payloads.clear()
            self.worker.close()
        if error is not None:
            raise error


class CompactLazyOwnerSearchSlabProxyV1:
    """Create the owner-search slab proxy after the first root batch reveals shape."""

    profile_only = True
    calls_train_muzero = False
    touches_live_runs = False
    supports_two_phase_compact_search = True

    def __init__(
        self,
        *,
        search_service_factory: Any,
        search_service_factory_kwargs: Mapping[str, Any] | None = None,
        root_provider_factory: Any = build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs: Mapping[str, Any] | None = None,
        replay_store_factory: Any | None = None,
        replay_store_factory_kwargs: Mapping[str, Any] | None = None,
        learner_factory: Any | None = None,
        learner_factory_kwargs: Mapping[str, Any] | None = None,
        owner_replay_append_enabled: bool = False,
        owner_sample_batch_size: int = 0,
        owner_train_steps: int = 0,
        owner_train_interval: int = 1,
        owner_model_refresh_interval: int = 1,
        owner_expected_train_request_count: int = 0,
        owner_defer_maintenance: bool = False,
        owner_learning_enabled: bool = True,
        policy_version_ref: str = "",
        model_version_ref: str = "",
        policy_source: str = "",
        root_store_capacity: int | None = None,
        root_store_metadata: Mapping[str, Any] | None = None,
        use_inner_two_phase_device_replay: bool = False,
        async_learner_worker: bool = False,
        async_learner_worker_kind: str = (
            COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD
        ),
        async_learner_max_pending: int = 1,
        require_resident_root_view: bool = False,
        fixed_action_result_buffer: bool = False,
        action_result_slot_capacity: int = 4,
    ) -> None:
        if not callable(search_service_factory):
            raise ValueError("search_service_factory must be callable")
        if not callable(root_provider_factory):
            raise ValueError("root_provider_factory must be callable")
        self.search_service_factory = search_service_factory
        self.search_service_factory_kwargs = dict(search_service_factory_kwargs or {})
        self.root_provider_factory = root_provider_factory
        self.root_provider_factory_kwargs = dict(root_provider_factory_kwargs or {})
        self.replay_store_factory = replay_store_factory
        self.replay_store_factory_kwargs = dict(replay_store_factory_kwargs or {})
        self.learner_factory = learner_factory
        self.learner_factory_kwargs = dict(learner_factory_kwargs or {})
        self.owner_replay_append_enabled = bool(owner_replay_append_enabled)
        self.owner_sample_batch_size = int(owner_sample_batch_size)
        self.owner_train_steps = int(owner_train_steps)
        self.owner_train_interval = int(owner_train_interval)
        self.owner_model_refresh_interval = int(owner_model_refresh_interval)
        self.owner_expected_train_request_count = int(owner_expected_train_request_count)
        self.owner_defer_maintenance = bool(owner_defer_maintenance)
        self.owner_learning_enabled = bool(owner_learning_enabled)
        if self.owner_sample_batch_size < 0:
            raise ValueError("owner_sample_batch_size must be nonnegative")
        if self.owner_train_steps < 0:
            raise ValueError("owner_train_steps must be nonnegative")
        if self.owner_train_interval <= 0:
            raise ValueError("owner_train_interval must be positive")
        if self.owner_model_refresh_interval <= 0:
            raise ValueError("owner_model_refresh_interval must be positive")
        if self.owner_expected_train_request_count < 0:
            raise ValueError("owner_expected_train_request_count must be nonnegative")
        self.policy_version_ref = str(policy_version_ref)
        self.model_version_ref = str(model_version_ref)
        self.policy_source = str(policy_source)
        self.use_inner_two_phase_device_replay = bool(use_inner_two_phase_device_replay)
        self.async_learner_worker = bool(async_learner_worker)
        self.async_learner_worker_kind = str(async_learner_worker_kind)
        self.async_learner_max_pending = int(async_learner_max_pending)
        self.require_resident_root_view = bool(require_resident_root_view)
        self.fixed_action_result_buffer = bool(fixed_action_result_buffer)
        self.action_result_slot_capacity = int(action_result_slot_capacity)
        if self.action_result_slot_capacity <= 0:
            raise ValueError("action_result_slot_capacity must be positive")
        self.action_result_slot_table = (
            CompactOwnerActionResultSlotTableV1(capacity=int(self.action_result_slot_capacity))
            if self.fixed_action_result_buffer
            else None
        )
        self.root_store_capacity = None if root_store_capacity is None else int(root_store_capacity)
        self.root_store_metadata = dict(root_store_metadata or {})
        self.root_store: CompactSharedMemoryRootStoreV1 | None = None
        self.worker: CompactProcessOwnerSearchWorkerV1 | None = None
        self.proxy: CompactOwnerSearchSlabProxyV1 | None = None
        self.request_count = 0
        self.last_search_result_payload_bytes = 0
        self.owner_replay_append_suppressed_entry_count = 0
        self.owner_replay_append_suppressed_transport_entry_count = 0
        self._last_closed_metadata: dict[str, Any] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        proxy = self.proxy
        base = {
            "compact_owner_search_slab_proxy": True,
            "compact_owner_search_lazy_slab_proxy": True,
            "compact_owner_search_boundary_kind": "worker_search_parent_slab_commit",
            "compact_owner_search_parent_slab_commits_replay": (
                not bool(self.owner_defer_maintenance)
            ),
            "compact_owner_search_slab_proxy_request_count": int(self.request_count),
            "compact_owner_search_slab_proxy_initialized": proxy is not None,
            "compact_owner_search_slab_proxy_last_search_result_payload_bytes": int(
                self.last_search_result_payload_bytes
            ),
            "compact_owner_search_owner_replay_append_enabled": bool(
                self.owner_replay_append_enabled
            ),
            "compact_owner_search_owner_learning_enabled": bool(self.owner_learning_enabled),
            "compact_owner_search_owner_sample_batch_size": int(self.owner_sample_batch_size),
            "compact_owner_search_owner_train_steps": int(self.owner_train_steps),
            "compact_owner_search_owner_train_interval": int(self.owner_train_interval),
            "compact_owner_search_owner_model_refresh_interval": int(
                self.owner_model_refresh_interval
            ),
            "compact_owner_search_owner_expected_train_request_count": int(
                self.owner_expected_train_request_count
            ),
            "compact_owner_search_owner_defer_maintenance": bool(self.owner_defer_maintenance),
            "compact_owner_search_resident_root_view_required": bool(
                self.require_resident_root_view
            ),
            "compact_owner_search_fixed_action_result_buffer_requested": bool(
                self.fixed_action_result_buffer
            ),
            "compact_owner_search_owner_replay_append_suppressed_entry_count": int(
                self.owner_replay_append_suppressed_entry_count
            ),
            "compact_owner_search_owner_replay_append_suppressed_transport_entry_count": int(
                self.owner_replay_append_suppressed_transport_entry_count
            ),
            "compact_owner_search_use_inner_two_phase_device_replay": bool(
                self.use_inner_two_phase_device_replay
            ),
            **(
                {}
                if self.action_result_slot_table is None
                else self.action_result_slot_table.metadata
            ),
        }
        if proxy is not None:
            base.update(proxy.metadata)
            base["compact_owner_search_lazy_slab_proxy"] = True
        elif self._last_closed_metadata:
            base.update(self._last_closed_metadata)
            base["compact_owner_search_lazy_slab_proxy"] = True
        return base

    def set_owner_learning_enabled(self, enabled: bool) -> None:
        self.owner_learning_enabled = bool(enabled)
        proxy = self.proxy
        if proxy is not None:
            proxy.set_owner_learning_enabled(enabled)

    def stage_replay_append_entries(self, replay_append_entries: Any) -> int:
        if self.proxy is not None:
            return self.proxy.stage_replay_append_entries(replay_append_entries)
        if not self.owner_learning_enabled:
            self.owner_replay_append_suppressed_entry_count += int(
                _owner_replay_append_entry_count_v1(replay_append_entries)
            )
            self.owner_replay_append_suppressed_transport_entry_count += int(
                _owner_replay_append_transport_entry_count_v1(replay_append_entries)
            )
            return 0
        if not self.owner_replay_append_enabled:
            return 0
        raise ReplayCompatibilityError(
            "lazy owner-search proxy cannot stage replay before initialization"
        )

    def stage_owner_proxy_transition_from_root_build_request(
        self,
        root_build_request: CompactRootBuildRequestV1,
        *,
        max_entries_per_batch: int,
        policy_source: str,
    ) -> dict[str, float]:
        if not self.direct_root_build_request_supported:
            raise ReplayCompatibilityError(
                "owner-proxy transition closure requires a direct-root owner-search proxy"
            )
        if self.proxy is None:
            self._initialize(None)
        assert self.proxy is not None
        return self.proxy.stage_owner_proxy_transition_from_root_build_request(
            root_build_request,
            max_entries_per_batch=int(max_entries_per_batch),
            policy_source=str(policy_source),
        )

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        if self.proxy is None:
            self._initialize(root_batch)
        assert self.proxy is not None
        result = self.proxy.run(root_batch)
        self.request_count = int(self.proxy.request_count)
        self.last_search_result_payload_bytes = int(self.proxy.last_search_result_payload_bytes)
        metadata = dict(result.metadata)
        metadata.update(
            {
                "compact_owner_search_lazy_slab_proxy": True,
                "compact_owner_search_slab_proxy_initialized": True,
            }
        )
        return CompactSearchResultV1(
            root_index=result.root_index,
            env_row=result.env_row,
            player=result.player,
            policy_env_id=result.policy_env_id,
            selected_action=result.selected_action,
            visit_policy=result.visit_policy,
            root_value=result.root_value,
            raw_visit_counts=result.raw_visit_counts,
            predicted_value=result.predicted_value,
            predicted_policy_logits=result.predicted_policy_logits,
            metadata=metadata,
        )

    def run_action_step(self, root_batch: CompactRootBatchV1) -> CompactSearchActionStepV1:
        if self.proxy is None:
            self._initialize(root_batch)
        assert self.proxy is not None
        action_step = self.proxy.run_action_step(root_batch)
        self.request_count = int(self.proxy.request_count)
        self.last_search_result_payload_bytes = int(self.proxy.last_search_result_payload_bytes)
        action_step.metadata["compact_owner_search_lazy_slab_proxy"] = True
        action_step.metadata["compact_owner_search_slab_proxy_initialized"] = True
        return action_step

    def run_action_step_from_root_build_request(
        self,
        root_build_request: CompactRootBuildRequestV1,
    ) -> CompactSearchActionStepV1:
        if not self.direct_root_build_request_supported:
            raise ReplayCompatibilityError(
                "root build request handoff requires a direct owner root store"
            )
        if self.proxy is None:
            self._initialize(None)
        assert self.proxy is not None
        action_step = self.proxy.run_action_step_from_root_build_request(root_build_request)
        self.request_count = int(self.proxy.request_count)
        self.last_search_result_payload_bytes = int(self.proxy.last_search_result_payload_bytes)
        action_step.metadata["compact_owner_search_lazy_slab_proxy"] = True
        action_step.metadata["compact_owner_search_slab_proxy_initialized"] = True
        return action_step

    def submit_owner_root_search_transaction_from_step_frame_slot(
        self,
        compact_batch: Any,
        *,
        search_lane: str,
        metadata: Mapping[str, Any] | None = None,
        copy_observation: bool = False,
        resident_host_observation_stub: bool = True,
        close_previous_transition: bool = False,
        max_entries_per_batch: int = 0,
        policy_source: str = "",
    ) -> CompactOwnerRootSearchTransactionDispatchV1:
        if not self.direct_root_build_request_supported:
            raise ReplayCompatibilityError(
                "owner root-search transaction requires a direct owner root store"
            )
        if self.proxy is None:
            self._initialize(None)
        assert self.proxy is not None
        submit = getattr(
            self.proxy,
            "submit_owner_root_search_transaction_from_step_frame_slot",
            None,
        )
        if not callable(submit):
            raise ReplayCompatibilityError(
                "direct owner proxy does not support root-search transactions"
            )
        transaction_dispatch = submit(
            compact_batch,
            search_lane=str(search_lane),
            metadata=dict(metadata or {}),
            copy_observation=bool(copy_observation),
            resident_host_observation_stub=bool(resident_host_observation_stub),
            close_previous_transition=bool(close_previous_transition),
            max_entries_per_batch=int(max_entries_per_batch),
            policy_source=str(policy_source),
        )
        self.request_count = int(self.proxy.request_count)
        self.last_search_result_payload_bytes = int(self.proxy.last_search_result_payload_bytes)
        transaction_dispatch.metadata["compact_owner_search_lazy_slab_proxy"] = True
        transaction_dispatch.metadata["compact_owner_search_slab_proxy_initialized"] = True
        return transaction_dispatch

    def submit_action_step_from_root_build_request(
        self,
        root_build_request: CompactRootBuildRequestV1,
    ) -> CompactOwnerActionDispatchHandleV1:
        if not self.direct_root_build_request_supported:
            raise ReplayCompatibilityError(
                "root build request handoff requires a direct owner root store"
            )
        if self.proxy is None:
            self._initialize(None)
        assert self.proxy is not None
        submit = getattr(self.proxy, "submit_action_step_from_root_build_request", None)
        if not callable(submit):
            raise ReplayCompatibilityError(
                "direct owner proxy does not support action dispatch handles"
            )
        dispatch_handle = submit(root_build_request)
        self.request_count = int(self.proxy.request_count)
        self.last_search_result_payload_bytes = int(self.proxy.last_search_result_payload_bytes)
        return dispatch_handle

    def resolve_action_step_handle(
        self,
        action_dispatch_handle: CompactOwnerActionDispatchHandleV1,
        *,
        sync_wrapper: bool = False,
    ) -> CompactSearchActionStepV1:
        if self.proxy is None:
            raise ReplayCompatibilityError("lazy owner-search proxy is not initialized")
        resolve = getattr(self.proxy, "resolve_action_step_handle", None)
        if not callable(resolve):
            raise ReplayCompatibilityError(
                "direct owner proxy does not support action dispatch handles"
            )
        action_step = resolve(action_dispatch_handle, sync_wrapper=bool(sync_wrapper))
        self.request_count = int(self.proxy.request_count)
        self.last_search_result_payload_bytes = int(self.proxy.last_search_result_payload_bytes)
        action_step.metadata["compact_owner_search_lazy_slab_proxy"] = True
        action_step.metadata["compact_owner_search_slab_proxy_initialized"] = True
        return action_step

    def flush_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactSearchReplayPayloadV1:
        if self.proxy is None:
            raise ReplayCompatibilityError("lazy owner-search proxy is not initialized")
        return self.proxy.flush_replay_payload(replay_payload_handle)

    def flush_device_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactDeviceSearchReplayPayloadV1:
        if self.proxy is None:
            raise ReplayCompatibilityError("lazy owner-search proxy is not initialized")
        return self.proxy.flush_device_replay_payload(replay_payload_handle)

    def drain_owner_maintenance(self, *, wait: bool = True) -> dict[str, Any]:
        proxy = self.proxy
        if proxy is None:
            return dict(self.metadata)
        metadata = proxy.drain_owner_maintenance(wait=bool(wait))
        self._last_closed_metadata = dict(metadata)
        return dict(self.metadata)

    def close(self) -> None:
        error: BaseException | None = None
        proxy = self.proxy
        self.proxy = None
        if proxy is not None:
            self._last_closed_metadata = dict(proxy.metadata)
            try:
                proxy.close()
            except BaseException as exc:
                error = exc
            finally:
                self._last_closed_metadata = dict(proxy.metadata)
        root_store = self.root_store
        self.root_store = None
        if root_store is not None:
            try:
                root_store.close()
            finally:
                root_store.unlink()
        if error is not None:
            raise error

    def _initialize(self, root_batch: CompactRootBatchV1) -> None:
        if self.fixed_action_result_buffer:
            raise ReplayCompatibilityError(
                "fixed action-result buffer requires a same-process owner proxy"
            )
        if self.require_resident_root_view:
            raise ReplayCompatibilityError(
                "resident root-view proof requires a direct-root owner-search proxy"
            )
        root_store = CompactSharedMemoryRootStoreV1.create(
            root_batch,
            capacity=self.root_store_capacity,
            metadata={
                "compact_owner_search_lazy_slab_proxy": True,
                **self.root_store_metadata,
            },
        )
        worker = CompactProcessOwnerSearchWorkerV1(
            root_provider_factory=self.root_provider_factory,
            root_provider_factory_kwargs={
                "spec": root_store.spec,
                **self.root_provider_factory_kwargs,
            },
            search_service_factory=self.search_service_factory,
            search_service_factory_kwargs=self.search_service_factory_kwargs,
            replay_store_factory=self.replay_store_factory,
            replay_store_factory_kwargs=self.replay_store_factory_kwargs,
            learner_factory=self.learner_factory,
            learner_factory_kwargs=self.learner_factory_kwargs,
            use_inner_two_phase_device_replay=self.use_inner_two_phase_device_replay,
            async_learner_worker=self.async_learner_worker,
            async_learner_worker_kind=self.async_learner_worker_kind,
            async_learner_max_pending=self.async_learner_max_pending,
        )
        self.root_store = root_store
        self.worker = worker
        self.proxy = CompactOwnerSearchSlabProxyV1(
            root_store=root_store,
            worker=worker,
            policy_version_ref=self.policy_version_ref,
            model_version_ref=self.model_version_ref,
            policy_source=self.policy_source,
            owner_replay_append_enabled=self.owner_replay_append_enabled,
            owner_sample_batch_size=self.owner_sample_batch_size,
            owner_train_steps=self.owner_train_steps,
            owner_train_interval=self.owner_train_interval,
            owner_model_refresh_interval=self.owner_model_refresh_interval,
            owner_expected_train_request_count=self.owner_expected_train_request_count,
            owner_defer_maintenance=self.owner_defer_maintenance,
            owner_learning_enabled=self.owner_learning_enabled,
        )


class CompactLazyInlineOwnerSearchSlabProxyV1(CompactLazyOwnerSearchSlabProxyV1):
    """Lazy owner proxy that keeps owner state colocated with the slab."""

    direct_root_build_request_supported = True

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(super().metadata)
        metadata["compact_owner_search_inline_slab_proxy"] = True
        metadata["compact_owner_search_boundary_kind"] = "inline_owner_search_parent_slab_commit"
        metadata["compact_owner_search_direct_root_handoff"] = True
        return metadata

    def _initialize(self, root_batch: CompactRootBatchV1) -> None:
        root_store = CompactDirectRootStoreV1(
            capacity=self.root_store_capacity,
            metadata={
                "compact_owner_search_lazy_slab_proxy": True,
                "compact_owner_search_inline_slab_proxy": True,
                **self.root_store_metadata,
            },
            require_resident_root_view=self.require_resident_root_view,
        )
        search_service = self.search_service_factory(**self.search_service_factory_kwargs)
        replay_store = (
            None
            if self.replay_store_factory is None
            else self.replay_store_factory(**self.replay_store_factory_kwargs)
        )
        learner = (
            None
            if self.learner_factory is None
            else self.learner_factory(**self.learner_factory_kwargs)
        )
        worker = CompactInlineOwnerSearchWorkerV1(
            root_provider=root_store,
            search_service=search_service,
            replay_store=replay_store,
            learner=learner,
            use_inner_two_phase_device_replay=self.use_inner_two_phase_device_replay,
            async_learner_worker=self.async_learner_worker,
            async_learner_worker_kind=self.async_learner_worker_kind,
            async_learner_max_pending=self.async_learner_max_pending,
        )
        self.root_store = root_store
        self.worker = worker
        self.proxy = CompactOwnerSearchSlabProxyV1(
            root_store=root_store,
            worker=worker,
            policy_version_ref=self.policy_version_ref,
            model_version_ref=self.model_version_ref,
            policy_source=self.policy_source,
            owner_replay_append_enabled=self.owner_replay_append_enabled,
            owner_sample_batch_size=self.owner_sample_batch_size,
            owner_train_steps=self.owner_train_steps,
            owner_train_interval=self.owner_train_interval,
            owner_model_refresh_interval=self.owner_model_refresh_interval,
            owner_expected_train_request_count=self.owner_expected_train_request_count,
            owner_defer_maintenance=self.owner_defer_maintenance,
            owner_learning_enabled=self.owner_learning_enabled,
            boundary_kind="inline_owner_search_parent_slab_commit",
            action_result_slot_table=self.action_result_slot_table,
        )


class CompactLazyInlineBackgroundOwnerSearchSlabProxyV1(CompactLazyOwnerSearchSlabProxyV1):
    """Lazy owner proxy with inline action and background maintenance overlap."""

    direct_root_build_request_supported = True

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(super().metadata)
        metadata["compact_owner_search_inline_background_slab_proxy"] = True
        metadata["compact_owner_search_boundary_kind"] = (
            "inline_background_owner_search_parent_slab_commit"
        )
        metadata["compact_owner_search_direct_root_handoff"] = True
        metadata["compact_owner_search_owner_background_overlap_enabled"] = True
        return metadata

    def _initialize(self, root_batch: CompactRootBatchV1) -> None:
        root_store = CompactDirectRootStoreV1(
            capacity=self.root_store_capacity,
            metadata={
                "compact_owner_search_lazy_slab_proxy": True,
                "compact_owner_search_inline_background_slab_proxy": True,
                **self.root_store_metadata,
            },
            require_resident_root_view=self.require_resident_root_view,
        )
        search_service = self.search_service_factory(**self.search_service_factory_kwargs)
        replay_store = (
            None
            if self.replay_store_factory is None
            else self.replay_store_factory(**self.replay_store_factory_kwargs)
        )
        learner = (
            None
            if self.learner_factory is None
            else self.learner_factory(**self.learner_factory_kwargs)
        )
        worker = CompactInlineBackgroundOwnerSearchWorkerV1(
            root_provider=root_store,
            search_service=search_service,
            replay_store=replay_store,
            learner=learner,
            use_inner_two_phase_device_replay=self.use_inner_two_phase_device_replay,
            async_learner_worker=self.async_learner_worker,
            async_learner_worker_kind=self.async_learner_worker_kind,
            async_learner_max_pending=self.async_learner_max_pending,
            action_result_slot_table=self.action_result_slot_table,
        )
        self.root_store = root_store
        self.worker = worker
        self.proxy = CompactOwnerSearchSlabProxyV1(
            root_store=root_store,
            worker=worker,
            policy_version_ref=self.policy_version_ref,
            model_version_ref=self.model_version_ref,
            policy_source=self.policy_source,
            owner_replay_append_enabled=self.owner_replay_append_enabled,
            owner_sample_batch_size=self.owner_sample_batch_size,
            owner_train_steps=self.owner_train_steps,
            owner_train_interval=self.owner_train_interval,
            owner_model_refresh_interval=self.owner_model_refresh_interval,
            owner_expected_train_request_count=self.owner_expected_train_request_count,
            owner_defer_maintenance=self.owner_defer_maintenance,
            owner_learning_enabled=self.owner_learning_enabled,
            boundary_kind="inline_background_owner_search_parent_slab_commit",
            action_result_slot_table=self.action_result_slot_table,
        )


class CompactLazyThreadedOwnerSearchSlabProxyV1(CompactLazyOwnerSearchSlabProxyV1):
    """Lazy owner proxy with direct roots and threaded owner overlap."""

    direct_root_build_request_supported = True

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = dict(super().metadata)
        metadata["compact_owner_search_threaded_slab_proxy"] = True
        metadata["compact_owner_search_boundary_kind"] = "threaded_owner_search_parent_slab_commit"
        metadata["compact_owner_search_direct_root_handoff"] = True
        metadata["compact_owner_search_owner_background_overlap_enabled"] = True
        return metadata

    def _initialize(self, root_batch: CompactRootBatchV1) -> None:
        root_store = CompactDirectRootStoreV1(
            capacity=self.root_store_capacity,
            metadata={
                "compact_owner_search_lazy_slab_proxy": True,
                "compact_owner_search_threaded_slab_proxy": True,
                **self.root_store_metadata,
            },
            require_resident_root_view=self.require_resident_root_view,
        )
        search_service = self.search_service_factory(**self.search_service_factory_kwargs)
        replay_store = (
            None
            if self.replay_store_factory is None
            else self.replay_store_factory(**self.replay_store_factory_kwargs)
        )
        learner = (
            None
            if self.learner_factory is None
            else self.learner_factory(**self.learner_factory_kwargs)
        )
        worker = CompactThreadedOwnerSearchWorkerV1(
            root_provider=root_store,
            search_service=search_service,
            replay_store=replay_store,
            learner=learner,
            use_inner_two_phase_device_replay=self.use_inner_two_phase_device_replay,
            async_learner_worker=self.async_learner_worker,
            async_learner_worker_kind=self.async_learner_worker_kind,
            async_learner_max_pending=self.async_learner_max_pending,
            action_result_slot_table=self.action_result_slot_table,
        )
        self.root_store = root_store
        self.worker = worker
        self.proxy = CompactOwnerSearchSlabProxyV1(
            root_store=root_store,
            worker=worker,
            policy_version_ref=self.policy_version_ref,
            model_version_ref=self.model_version_ref,
            policy_source=self.policy_source,
            owner_replay_append_enabled=self.owner_replay_append_enabled,
            owner_sample_batch_size=self.owner_sample_batch_size,
            owner_train_steps=self.owner_train_steps,
            owner_train_interval=self.owner_train_interval,
            owner_model_refresh_interval=self.owner_model_refresh_interval,
            owner_expected_train_request_count=self.owner_expected_train_request_count,
            owner_defer_maintenance=self.owner_defer_maintenance,
            owner_learning_enabled=self.owner_learning_enabled,
            boundary_kind="threaded_owner_search_parent_slab_commit",
            action_result_slot_table=self.action_result_slot_table,
        )


def _validate_owner_search_request(
    request: CompactOwnerSearchRequestV1,
) -> CompactOwnerSearchRequestV1:
    if not isinstance(request, CompactOwnerSearchRequestV1):
        raise TypeError("request must be CompactOwnerSearchRequestV1")
    if int(request.request_id) <= 0:
        raise ValueError("request_id must be positive")
    if int(request.actor_step) < 0:
        raise ValueError("actor_step must be nonnegative")
    root_slot_ids = tuple(int(slot_id) for slot_id in tuple(request.root_slot_ids))
    if not root_slot_ids:
        raise ValueError("root_slot_ids must be non-empty")
    if int(request.train_steps) < 0:
        raise ValueError("train_steps must be nonnegative")
    if int(request.sample_batch_size) < 0:
        raise ValueError("sample_batch_size must be nonnegative")
    if int(request.action_result_slot_id) < -1:
        raise ValueError("action_result_slot_id must be -1 or nonnegative")
    return CompactOwnerSearchRequestV1(
        request_id=int(request.request_id),
        actor_step=int(request.actor_step),
        root_slot_ids=root_slot_ids,
        replay_append_entries=tuple(request.replay_append_entries),
        sample_batch_size=int(request.sample_batch_size),
        train_steps=int(request.train_steps),
        policy_version_ref=str(request.policy_version_ref),
        model_version_ref=str(request.model_version_ref),
        policy_source=str(request.policy_source),
        refresh_model=bool(request.refresh_model),
        action_result_slot_id=int(request.action_result_slot_id),
    )


def _validate_owner_maintenance_drain_request(
    request: CompactOwnerMaintenanceDrainRequestV1,
) -> CompactOwnerMaintenanceDrainRequestV1:
    if not isinstance(request, CompactOwnerMaintenanceDrainRequestV1):
        raise TypeError("request must be CompactOwnerMaintenanceDrainRequestV1")
    if int(request.drain_id) <= 0:
        raise ValueError("drain_id must be positive")
    if int(request.max_items) < 0:
        raise ValueError("max_items must be nonnegative")
    return CompactOwnerMaintenanceDrainRequestV1(
        drain_id=int(request.drain_id),
        max_items=int(request.max_items),
        fail_if_empty=bool(request.fail_if_empty),
    )


def _resolve_root_batch(
    root_provider: Any,
    root_slot_ids: tuple[int, ...],
    *,
    request: CompactOwnerSearchRequestV1,
) -> CompactRootBatchV1:
    resolver = getattr(root_provider, "resolve_root_batch", None)
    if callable(resolver):
        root_batch = resolver(root_slot_ids=root_slot_ids, request=request)
    elif callable(root_provider):
        root_batch = root_provider(root_slot_ids=root_slot_ids, request=request)
    else:
        raise RuntimeError("owner-search root_provider must resolve root batches")
    if not isinstance(root_batch, CompactRootBatchV1):
        raise RuntimeError("owner-search root provider returned wrong root batch type")
    return root_batch


def _attach_owner_resident_bridge_metadata(
    search_result: CompactSearchResultV1,
    *,
    root_batch: CompactRootBatchV1,
) -> CompactSearchResultV1:
    root_metadata = dict(getattr(root_batch, "metadata", {}) or {})
    bridge_metadata = {
        key: value
        for key, value in root_metadata.items()
        if str(key).startswith("compact_owner_search_resident_root_bridge_")
        or str(key).startswith("compact_owner_search_resident_root_view_")
        or str(key).startswith("compact_owner_search_direct_root_")
        or str(key).startswith("compact_direct_root_store")
        or str(key) == "compact_owner_search_inline_slab_proxy"
        or str(key) == "compact_owner_search_inline_background_slab_proxy"
        or str(key) == "compact_owner_search_threaded_slab_proxy"
        or str(key) == "owner_search_compact_torch_resident_root_bridge_ready"
    }
    if not bridge_metadata:
        return search_result
    metadata = dict(search_result.metadata)
    for key, value in bridge_metadata.items():
        metadata[str(key)] = value
    return CompactSearchResultV1(
        root_index=search_result.root_index,
        env_row=search_result.env_row,
        player=search_result.player,
        policy_env_id=search_result.policy_env_id,
        selected_action=search_result.selected_action,
        visit_policy=search_result.visit_policy,
        root_value=search_result.root_value,
        raw_visit_counts=search_result.raw_visit_counts,
        predicted_value=search_result.predicted_value,
        predicted_policy_logits=search_result.predicted_policy_logits,
        metadata=metadata,
    )


def _owner_ref_digest(owner_ref: Mapping[str, Any] | None) -> str:
    if owner_ref is None:
        return ""
    return str(owner_ref.get("model_state_digest") or "").strip()


def _owner_ref_digest_deferred_to_search_refresh(
    owner_ref: Mapping[str, Any] | None,
) -> bool:
    if owner_ref is None:
        return False
    return bool(owner_ref.get(COMPACT_OWNER_SEARCH_OWNER_REF_DIGEST_DEFERRED_TO_SEARCH_REFRESH_KEY))


def _owner_learner_result_tuple_from_mapping(
    result: Any,
    *,
    default_train_steps: int = 1,
    require_owner_ref: bool = True,
) -> tuple[dict[str, Any] | None, int, dict[str, Any], dict[str, Any]]:
    if not isinstance(result, Mapping):
        raise RuntimeError("owner-search learner returned non-mapping result")
    result_dict = dict(result)
    owner_ref = result_dict.get("model_owner_ref")
    if owner_ref is None and not bool(require_owner_ref):
        owner_ref_dict = None
    elif not isinstance(owner_ref, Mapping):
        raise RuntimeError("owner-search learner must return model_owner_ref")
    else:
        owner_ref_dict = dict(owner_ref)
    if (
        owner_ref_dict is not None
        and owner_ref_dict.get("transport_kind") != COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1
    ):
        raise RuntimeError("owner-search learner returned wrong owner-ref transport")
    if owner_ref_dict is not None and not _owner_ref_digest(owner_ref_dict):
        raise RuntimeError("owner-search learner owner ref must include a digest")
    updates = int(result_dict.get("learner_update_count", default_train_steps) or 0)
    if updates <= 0:
        raise RuntimeError("owner-search learner must report positive update count")
    sample_metadata = result_dict.get("sample_metadata")
    if not isinstance(sample_metadata, Mapping):
        sample_metadata = {}
    sample_telemetry = result_dict.get("sample_telemetry")
    if not isinstance(sample_telemetry, Mapping):
        sample_telemetry = {}
    sample_telemetry = {
        **dict(sample_metadata),
        **dict(sample_telemetry),
    }
    learner_telemetry = result_dict.get("learner_telemetry")
    if not isinstance(learner_telemetry, Mapping):
        learner_result = result_dict.get("learner_result")
        if hasattr(learner_result, "telemetry"):
            learner_telemetry = getattr(learner_result, "telemetry")
        elif isinstance(learner_result, Mapping):
            nested = learner_result.get("telemetry")
            learner_telemetry = nested if isinstance(nested, Mapping) else {}
        else:
            learner_telemetry = {}
    return (
        owner_ref_dict,
        updates,
        dict(sample_telemetry),
        dict(learner_telemetry),
    )


def _public_owner_search_worker_state_v1(
    search_state: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if search_state is None:
        return None
    public: dict[str, Any] = {}
    allowed_exact = {
        "compact_policy_refresh_search_worker_refreshed",
        "compact_policy_refresh_learner_update_count",
        "learner_update_count",
        "model_state_digest",
        "expected_model_state_digest",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "search_worker_pid",
        "worker_pid",
        "model_state_snapshot_loaded",
        "model_state_snapshot_load_sec",
        "model_state_snapshot_load_bytes",
    }
    allowed_fragments = (
        "digest",
        "ref",
        "count",
        "version",
        "source",
        "pid",
        "refreshed",
    )
    for raw_key, value in dict(search_state).items():
        key = str(raw_key)
        if key not in allowed_exact and not any(fragment in key for fragment in allowed_fragments):
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            public[key] = value
    if _contains_cuda_tensor(public):
        raise RuntimeError("owner-search public worker state must not contain CUDA tensors")
    return public


def _owner_replay_append_entry_count_v1(replay_append_entries: Any) -> int:
    if replay_append_entries is None:
        return 0
    if isinstance(replay_append_entries, (list, tuple)):
        return sum(_owner_replay_append_entry_count_v1(entry) for entry in replay_append_entries)
    if _owner_replay_append_is_transition_batch_v1(replay_append_entries):
        return _owner_replay_append_transition_batch_entry_count_v1(replay_append_entries)
    return 1


def _owner_proxy_replay_append_derived_transition_batch_v1(
    facts: tuple[_CompactOwnerProxyDerivedTransitionFactsV1, ...],
    *,
    max_entries_per_batch: int,
) -> CompactOwnerSearchReplayAppendDerivedTransitionBatchV1:
    if not facts:
        raise ReplayCompatibilityError("owner-proxy derived transition batch requires entries")
    transition_count = len(facts)
    fixed_capacity = int(max_entries_per_batch)
    if fixed_capacity <= 1:
        raise ReplayCompatibilityError("owner-proxy transition batch capacity must exceed one")
    if transition_count > fixed_capacity:
        raise ReplayCompatibilityError("owner-proxy transition batch exceeds fixed capacity")
    record_indices = np.asarray([int(item.record_index) for item in facts], dtype=np.int64)
    next_record_indices = np.asarray(
        [int(item.next_record_index) for item in facts],
        dtype=np.int64,
    )
    applied_action_counts = np.asarray(
        [int(item.applied_action_count) for item in facts],
        dtype=np.int64,
    )
    applied_action_checksums = np.asarray(
        [int(item.applied_action_checksum) for item in facts],
        dtype=np.int64,
    )
    replay_payload_handles = tuple(str(item.replay_payload_handle) for item in facts)
    selected_action_digests = tuple(str(item.selected_action_digest) for item in facts)
    search_replay_payload_digests = tuple(str(item.search_replay_payload_digest) for item in facts)
    policy_sources = tuple(str(item.policy_source) for item in facts)
    if len(set(policy_sources)) != 1:
        raise ReplayCompatibilityError("owner-proxy derived transition policy sources must match")
    digest = (
        f"{compact_search_array_digest_v1(record_indices)}:"
        f"{compact_search_array_digest_v1(next_record_indices)}:"
        f"{compact_search_array_digest_v1(applied_action_checksums)}"
    )
    transport_bytes = int(
        sum(
            int(array.nbytes)
            for array in (
                record_indices,
                next_record_indices,
                applied_action_counts,
                applied_action_checksums,
            )
        )
    )
    metadata = {
        "compact_owner_search_replay_append_entry": True,
        "compact_owner_search_replay_append_entry_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_replay_append_transition_batch": True,
        "compact_owner_search_replay_append_transition_batch_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_replay_append_transition_batch_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
        ),
        "compact_owner_search_replay_append_transition_batch_transition_count": int(
            transition_count
        ),
        "compact_owner_search_replay_append_owner_materializes_rows": True,
        "compact_owner_search_replay_append_owner_derives_transition_outcome": True,
        "compact_owner_search_replay_append_carries_index_rows": False,
        "compact_owner_search_replay_append_carries_compact_batches": False,
        "compact_owner_search_replay_append_carries_parent_transition_outcomes": False,
        "compact_owner_search_owner_replay_transport_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
        ),
        "compact_owner_search_owner_local_transition_derivation_requested": True,
        "compact_owner_search_owner_local_transition_derivation_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_owner_local_transition_derivation_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
        ),
        "compact_owner_search_owner_local_transition_derivation_batch_count": 1,
        "compact_owner_search_owner_local_transition_derivation_transition_count": int(
            transition_count
        ),
        "compact_owner_search_owner_local_transition_derivation_transport_entry_count": 1,
        "compact_owner_search_owner_local_transition_derivation_transport_bytes": int(
            transport_bytes
        ),
        "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes": 0,
        "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count": 0,
        "compact_owner_search_owner_local_transition_derivation_digest": digest,
        "compact_owner_search_owner_local_transition_derivation_digest_verified": True,
        "compact_owner_search_owner_local_transition_derivation_fallback_count": 0,
        "compact_owner_search_owner_local_transition_derivation_fallback_reason": "none",
        "compact_owner_search_owner_proxy_transition_closure_source": (
            "owner_proxy_cached_state_v1"
        ),
        "compact_owner_search_owner_proxy_transition_closure_batch_count": 1,
        "compact_owner_search_owner_proxy_transition_closure_transition_count": int(
            transition_count
        ),
        "compact_owner_search_transition_batch_transport_requested": True,
        "compact_owner_search_transition_batch_transport_enabled": True,
        "compact_owner_search_transition_batch_transport_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
        ),
        "compact_owner_search_transition_batch_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_transition_batch_count": 1,
        "compact_owner_search_transition_batch_entry_count": int(transition_count),
        "compact_owner_search_transition_batch_transport_entry_count": 1,
        "compact_owner_search_transition_batch_max_entries_per_batch": int(fixed_capacity),
        "compact_owner_search_transition_batch_fixed_capacity": int(fixed_capacity),
        "compact_owner_search_transition_batch_padding_count": int(
            max(0, fixed_capacity - transition_count)
        ),
        "compact_owner_search_transition_batch_overflow_count": 0,
        "compact_owner_search_transition_batch_fallback_count": 0,
        "compact_owner_search_transition_batch_fallback_reason": "none",
        "compact_owner_search_transition_batch_pending_count": 0,
        "compact_owner_search_transition_batch_transport_bytes": int(transport_bytes),
        "compact_owner_search_transition_batch_digest": digest,
        "compact_owner_search_transition_batch_digest_verified": True,
        "record_index_first": int(record_indices[0]),
        "record_index_last": int(record_indices[-1]),
        "next_record_index_last": int(next_record_indices[-1]),
    }
    return CompactOwnerSearchReplayAppendDerivedTransitionBatchV1(
        schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID,
        transition_count=int(transition_count),
        record_indices=record_indices,
        next_record_indices=next_record_indices,
        replay_payload_handles=replay_payload_handles,
        selected_action_digests=selected_action_digests,
        search_replay_payload_digests=search_replay_payload_digests,
        applied_action_counts=applied_action_counts,
        applied_action_checksums=applied_action_checksums,
        policy_source=policy_sources[0],
        metadata=metadata,
    )


def _owner_replay_append_transport_entry_count_v1(replay_append_entries: Any) -> int:
    if replay_append_entries is None:
        return 0
    if isinstance(replay_append_entries, (list, tuple)):
        return len(replay_append_entries)
    return 1


def _owner_replay_append_transition_batch_count_v1(replay_append_entries: Any) -> int:
    if replay_append_entries is None:
        return 0
    if isinstance(replay_append_entries, (list, tuple)):
        return sum(
            _owner_replay_append_transition_batch_count_v1(entry) for entry in replay_append_entries
        )
    return 1 if _owner_replay_append_is_transition_batch_v1(replay_append_entries) else 0


def _owner_replay_append_transition_batch_entry_count_total_v1(
    replay_append_entries: Any,
) -> int:
    if replay_append_entries is None:
        return 0
    if isinstance(replay_append_entries, (list, tuple)):
        return sum(
            _owner_replay_append_transition_batch_entry_count_total_v1(entry)
            for entry in replay_append_entries
        )
    if not _owner_replay_append_is_transition_batch_v1(replay_append_entries):
        return 0
    return _owner_replay_append_transition_batch_entry_count_v1(replay_append_entries)


def _owner_replay_append_transition_legacy_entry_count_v1(
    replay_append_entries: Any,
) -> int:
    if replay_append_entries is None:
        return 0
    if isinstance(replay_append_entries, (list, tuple)):
        return sum(
            _owner_replay_append_transition_legacy_entry_count_v1(entry)
            for entry in replay_append_entries
        )
    if _owner_replay_append_is_transition_batch_v1(replay_append_entries):
        return 0
    metadata = dict(getattr(replay_append_entries, "metadata", {}) or {})
    schema_id = str(getattr(replay_append_entries, "schema_id", ""))
    if bool(metadata.get("compact_owner_search_replay_append_transition_only")):
        return 1
    if schema_id == COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID:
        return 1
    return 0


def _owner_replay_append_is_transition_batch_v1(entry: Any) -> bool:
    if entry is None:
        return False
    metadata = dict(getattr(entry, "metadata", {}) or {})
    schema_id = str(getattr(entry, "schema_id", ""))
    return (
        schema_id == COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
        or schema_id == COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        or bool(metadata.get("compact_owner_search_replay_append_transition_batch"))
    )


def _owner_replay_append_is_derived_transition_batch_v1(entry: Any) -> bool:
    if entry is None:
        return False
    metadata = dict(getattr(entry, "metadata", {}) or {})
    schema_id = str(getattr(entry, "schema_id", ""))
    kind = str(metadata.get("compact_owner_search_replay_append_transition_batch_kind") or "")
    return (
        schema_id == COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        or kind == COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
        or bool(metadata.get("compact_owner_search_replay_append_owner_derives_transition_outcome"))
    )


def _owner_replay_append_transition_batch_entry_count_v1(entry: Any) -> int:
    metadata = dict(getattr(entry, "metadata", {}) or {})
    raw_count = getattr(entry, "transition_count", None)
    if raw_count is None:
        raw_count = metadata.get(
            "compact_owner_search_replay_append_transition_batch_transition_count",
            metadata.get("compact_owner_search_transition_batch_entry_count", 0),
        )
    count = int(raw_count or 0)
    if count < 0:
        raise RuntimeError("owner-search transition batch count must be nonnegative")
    return count


def _owner_replay_append_materialization_entries_v1(entry: Any) -> tuple[Any, ...]:
    if not _owner_replay_append_is_transition_batch_v1(entry):
        return (entry,)
    if _owner_replay_append_is_derived_transition_batch_v1(entry):
        raise RuntimeError(
            "owner-local derived transition batches require direct transition-batch replay"
        )
    count = _owner_replay_append_transition_batch_entry_count_v1(entry)
    if count <= 0:
        raise RuntimeError("owner-search transition batch must carry entries")
    record_indices = _owner_transition_batch_array_v1(
        entry,
        "record_indices",
        count=count,
        dtype=np.int64,
    )
    next_record_indices = _owner_transition_batch_array_v1(
        entry,
        "next_record_indices",
        count=count,
        dtype=np.int64,
    )
    replay_payload_handles = tuple(
        str(value) for value in getattr(entry, "replay_payload_handles", ())
    )
    selected_action_digests = tuple(
        str(value) for value in getattr(entry, "selected_action_digests", ())
    )
    search_replay_payload_digests = tuple(
        str(value) for value in getattr(entry, "search_replay_payload_digests", ())
    )
    if len(replay_payload_handles) != count:
        raise RuntimeError("owner-search transition batch handle count mismatch")
    if len(set(replay_payload_handles)) != count:
        raise RuntimeError("owner-search transition batch duplicate handle")
    if len(selected_action_digests) != count:
        raise RuntimeError("owner-search transition batch selected digest count mismatch")
    if len(search_replay_payload_digests) != count:
        raise RuntimeError("owner-search transition batch replay digest count mismatch")
    next_joint_action = _owner_transition_batch_array_v1(
        entry,
        "next_joint_action",
        count=count,
        dtype=np.int16,
    )
    next_reward = _owner_transition_batch_array_v1(
        entry,
        "next_reward",
        count=count,
        dtype=np.float32,
    )
    next_done = _owner_transition_batch_array_v1(
        entry,
        "next_done",
        count=count,
        dtype=np.bool_,
    )
    next_terminated = _owner_transition_batch_array_v1(
        entry,
        "next_terminated",
        count=count,
        dtype=np.bool_,
    )
    next_truncated = _owner_transition_batch_array_v1(
        entry,
        "next_truncated",
        count=count,
        dtype=np.bool_,
    )
    next_final_reward_map = _owner_transition_batch_array_v1(
        entry,
        "next_final_reward_map",
        count=count,
        dtype=np.float32,
    )
    next_final_observation_row_mask = _owner_transition_batch_array_v1(
        entry,
        "next_final_observation_row_mask",
        count=count,
        dtype=np.bool_,
    )
    batch_metadata = dict(getattr(entry, "metadata", {}) or {})
    digest = str(batch_metadata.get("compact_owner_search_transition_batch_digest") or "")
    if not digest:
        raise RuntimeError("owner-search transition batch missing digest")
    materialized: list[Any] = []
    for offset in range(count):
        metadata = {
            "compact_owner_search_replay_append_entry": True,
            "compact_owner_search_replay_append_entry_schema_id": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID
            ),
            "compact_owner_search_replay_append_transition_only": True,
            "compact_owner_search_replay_append_owner_materializes_rows": True,
            "compact_owner_search_replay_append_carries_index_rows": False,
            "compact_owner_search_replay_append_carries_compact_batches": False,
            "compact_owner_search_replay_append_transition_batch_member": True,
            "compact_owner_search_replay_append_transition_batch_schema_id": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
            ),
            "compact_owner_search_replay_append_transition_batch_kind": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
            ),
            "compact_owner_search_replay_append_transition_batch_member_index": int(offset),
            "compact_owner_search_transition_batch_digest": digest,
            "compact_owner_search_transition_batch_digest_verified": True,
            "record_index": int(record_indices[offset]),
            "next_record_index": int(next_record_indices[offset]),
            "replay_payload_handle": replay_payload_handles[offset],
            "selected_action_digest": selected_action_digests[offset],
            "search_replay_payload_digest": search_replay_payload_digests[offset],
        }
        materialized.append(
            SimpleNamespace(
                schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID,
                record_index=int(record_indices[offset]),
                next_record_index=int(next_record_indices[offset]),
                replay_payload_handle=replay_payload_handles[offset],
                next_joint_action=next_joint_action[offset],
                next_reward=next_reward[offset],
                next_done=next_done[offset],
                next_terminated=next_terminated[offset],
                next_truncated=next_truncated[offset],
                next_final_reward_map=next_final_reward_map[offset],
                next_final_observation_row_mask=(next_final_observation_row_mask[offset]),
                policy_source=str(getattr(entry, "policy_source", "")),
                metadata=metadata,
            )
        )
    return tuple(materialized)


def _owner_transition_batch_array_v1(
    entry: Any,
    field: str,
    *,
    count: int,
    dtype: Any,
) -> np.ndarray:
    value = np.asarray(getattr(entry, field), dtype=dtype)
    if value.shape[:1] != (int(count),):
        raise RuntimeError(f"owner-search transition batch {field} shape mismatch")
    return value


def _trim_owner_root_batch_cache_v1(
    cache: Mapping[int, CompactRootBatchV1],
    *,
    max_entries: int = COMPACT_OWNER_SEARCH_ROOT_BATCH_CACHE_MAX,
) -> dict[int, CompactRootBatchV1]:
    retained = dict(cache)
    limit = int(max_entries)
    if limit <= 0 or len(retained) <= limit:
        return retained
    keep = set(sorted(int(key) for key in retained.keys())[-limit:])
    return {int(key): value for key, value in retained.items() if int(key) in keep}


def _owner_root_batch_cache_for_replay_entries_v1(
    cache: Mapping[int, CompactRootBatchV1],
    replay_append_entries: Any,
    *,
    current_actor_step: int,
    current_root_batch: CompactRootBatchV1,
) -> tuple[dict[int, CompactRootBatchV1], dict[str, Any]]:
    available = {int(key): value for key, value in cache.items()}
    available[int(current_actor_step)] = current_root_batch
    required, full_fallback = _owner_replay_required_root_batch_indices_v1(replay_append_entries)
    required.add(int(current_actor_step))
    full_entry_count = len(available)
    if full_fallback:
        retained = dict(available)
    else:
        retained = {
            int(key): available[int(key)] for key in sorted(required) if int(key) in available
        }
    retained_entry_count = len(retained)
    metadata = {
        "compact_owner_search_owner_maintenance_root_cache_snapshot_full_entry_count": int(
            full_entry_count
        ),
        "compact_owner_search_owner_maintenance_root_cache_snapshot_retained_entry_count": int(
            retained_entry_count
        ),
        "compact_owner_search_owner_maintenance_root_cache_snapshot_required_entry_count": int(
            len(required)
        ),
        "compact_owner_search_owner_maintenance_root_cache_snapshot_dropped_entry_count": int(
            max(0, full_entry_count - retained_entry_count)
        ),
        "compact_owner_search_owner_maintenance_root_cache_snapshot_full_fallback_count": int(
            1 if full_fallback else 0
        ),
    }
    return retained, metadata


def _owner_replay_required_root_batch_indices_v1(
    replay_append_entries: Any,
) -> tuple[set[int], bool]:
    if replay_append_entries is None:
        return set(), False
    if isinstance(replay_append_entries, (list, tuple)):
        required: set[int] = set()
        full_fallback = False
        for entry in replay_append_entries:
            entry_required, entry_fallback = _owner_replay_required_root_batch_indices_v1(entry)
            required.update(entry_required)
            full_fallback = full_fallback or entry_fallback
        return required, full_fallback
    entry = replay_append_entries
    if _owner_replay_append_is_transition_batch_v1(entry):
        count = _owner_replay_append_transition_batch_entry_count_v1(entry)
        record_indices = _owner_transition_batch_array_v1(
            entry,
            "record_indices",
            count=count,
            dtype=np.int64,
        )
        next_record_indices = _owner_transition_batch_array_v1(
            entry,
            "next_record_indices",
            count=count,
            dtype=np.int64,
        )
        required = {
            int(value)
            for value in np.concatenate(
                (
                    np.asarray(record_indices, dtype=np.int64).reshape(-1),
                    np.asarray(next_record_indices, dtype=np.int64).reshape(-1),
                )
            )
        }
        return required, False
    if hasattr(entry, "record_index") and hasattr(entry, "next_record_index"):
        return {
            int(getattr(entry, "record_index")),
            int(getattr(entry, "next_record_index")),
        }, False
    metadata = dict(getattr(entry, "metadata", {}) or {})
    if "record_index" in metadata and "next_record_index" in metadata:
        return {
            int(metadata["record_index"]),
            int(metadata["next_record_index"]),
        }, False
    return set(), True


def _search_refresh_update_count(search_state: Mapping[str, Any] | None) -> int:
    if search_state is None:
        return 0
    return int(
        search_state.get(
            "learner_update_count",
            search_state.get("compact_policy_refresh_learner_update_count", 0),
        )
        or 0
    )


def _compact_batch_from_root_batch(root_batch: CompactRootBatchV1) -> SimpleNamespace:
    observation = np.asarray(root_batch.observation)
    if observation.ndim != 4:
        raise RuntimeError("owner-search cached root observation must be rank-4")
    root_count = int(observation.shape[0])
    metadata = dict(root_batch.metadata or {})
    player_count = int(metadata.get("player_count") or 0)
    if player_count <= 0:
        players = np.asarray(root_batch.player, dtype=np.int64).reshape(-1)
        player_count = int(players.max()) + 1 if players.size else 1
    batch_size = int(metadata.get("batch_size") or 0)
    if batch_size <= 0:
        if root_count % player_count != 0:
            raise RuntimeError("owner-search root batch cannot infer batch size")
        batch_size = root_count // player_count
    if batch_size <= 0 or player_count <= 0 or batch_size * player_count != root_count:
        raise RuntimeError("owner-search cached root batch shape mismatch")
    stack_shape = tuple(int(dim) for dim in observation.shape[1:])
    final_observation = None
    if root_batch.final_observation is not None:
        final_root = np.asarray(root_batch.final_observation)
        if tuple(int(dim) for dim in final_root.shape) != (root_count, *stack_shape):
            raise RuntimeError("owner-search cached root final observation shape mismatch")
        final_observation = final_root.reshape(batch_size, player_count, *stack_shape).copy()
    resident = root_batch.resident_observation
    observation_source = (
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        if resident is not None
        else COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
    )
    observation_grid = observation.reshape(batch_size, player_count, *stack_shape)
    if resident is None:
        observation_grid = observation_grid.copy()
    return SimpleNamespace(
        observation=observation_grid,
        action_mask=np.asarray(root_batch.legal_mask, dtype=np.bool_).reshape(
            batch_size,
            player_count,
            -1,
        ),
        policy_env_id=np.asarray(root_batch.policy_env_id, dtype=np.int64).copy(),
        policy_env_row=np.asarray(root_batch.env_row, dtype=np.int64).copy(),
        policy_player=np.asarray(root_batch.player, dtype=np.int64).copy(),
        done=np.asarray(root_batch.done_root, dtype=np.bool_)
        .reshape(batch_size, player_count)
        .any(axis=1),
        done_root=np.asarray(root_batch.done_root, dtype=np.bool_).copy(),
        to_play=np.asarray(root_batch.to_play, dtype=np.int64).copy(),
        target_reward=np.asarray(root_batch.target_reward, dtype=np.float32).copy(),
        reward=np.asarray(root_batch.target_reward, dtype=np.float32).reshape(
            batch_size, player_count
        ),
        final_observation=final_observation,
        final_observation_row_mask=np.asarray(
            root_batch.final_observation_row_mask,
            dtype=np.bool_,
        ).copy(),
        terminal_row_mask=np.asarray(root_batch.terminal_row_mask, dtype=np.bool_).copy(),
        autoreset_row_mask=np.asarray(root_batch.autoreset_row_mask, dtype=np.bool_).copy(),
        observation_source=observation_source,
        resident_observation=resident,
    )


def _resident_root_device(root_batch: CompactRootBatchV1) -> Any:
    resident = root_batch.resident_observation
    device_observation = getattr(resident, "device_observation", None)
    device = getattr(device_observation, "device", None)
    if device is not None:
        return device
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _validate_owner_action_step_identity_v1(
    root_batch: CompactRootBatchV1,
    action_step: CompactSearchActionStepV1,
) -> None:
    root_index = np.asarray(action_step.root_index, dtype=np.int32)
    expected_root_index = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
    if not np.array_equal(root_index, expected_root_index):
        raise ReplayCompatibilityError("owner action step roots do not match root batch")
    if not np.array_equal(action_step.env_row, root_batch.env_row[root_index]):
        raise ReplayCompatibilityError("owner action step env_row does not match root batch")
    if not np.array_equal(action_step.player, root_batch.player[root_index]):
        raise ReplayCompatibilityError("owner action step player does not match root batch")
    if not np.array_equal(action_step.policy_env_id, root_batch.policy_env_id[root_index]):
        raise ReplayCompatibilityError("owner action step policy_env_id does not match root batch")
    selected = np.asarray(action_step.selected_action, dtype=np.int16)
    if selected.shape != (int(root_index.size),):
        raise ReplayCompatibilityError("owner action step selected_action count mismatch")
    legal_mask = np.asarray(root_batch.legal_mask, dtype=np.bool_)
    if selected.size and not bool(legal_mask[root_index, selected.astype(np.int64)].all()):
        raise ReplayCompatibilityError("owner action step selected_action is illegal")
    selected_digest = str(action_step.metadata.get("selected_action_digest") or "")
    if selected_digest and selected_digest != compact_search_array_digest_v1(selected):
        raise ReplayCompatibilityError("owner action step selected-action digest is stale")


def _compat_search_result_from_action_step_v1(
    root_batch: CompactRootBatchV1,
    action_step: CompactSearchActionStepV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CompactSearchResultV1:
    """Build a marked compatibility result without materializing search targets."""

    _validate_owner_action_step_identity_v1(root_batch, action_step)
    selected = np.asarray(action_step.selected_action, dtype=np.int16)
    action_count = int(np.asarray(root_batch.legal_mask).shape[1])
    visit_policy = np.zeros((int(selected.size), action_count), dtype=np.float32)
    if selected.size:
        visit_policy[np.arange(int(selected.size)), selected.astype(np.int64)] = 1.0
    result_metadata = dict(action_step.metadata)
    if metadata:
        result_metadata.update({str(key): value for key, value in metadata.items()})
    result_metadata.update(
        {
            "schema_id": "curvyzero_compact_search_result/v1",
            "search_impl": str(action_step.metadata.get("search_impl") or "owner_action_only"),
            "num_simulations": int(action_step.metadata.get("num_simulations") or 0),
            "compact_owner_search_action_only_compat_result": True,
            "compact_owner_search_action_only_result": True,
            "compact_owner_search_owner_materializes_replay": True,
            "compact_owner_search_parent_slab_commits_replay": False,
            "replay_payload_handle": str(action_step.replay_payload_handle),
            "search_replay_payload_digest": str(
                action_step.metadata.get("search_replay_payload_digest") or ""
            ),
        }
    )
    return validate_compact_search_result_v1(
        root_batch,
        selected_action=selected,
        visit_policy=visit_policy,
        root_value=np.zeros((int(selected.size),), dtype=np.float32),
        raw_visit_counts=None,
        predicted_value=None,
        predicted_policy_logits=None,
        search_impl=str(result_metadata["search_impl"]),
        num_simulations=int(result_metadata["num_simulations"]),
        metadata=result_metadata,
    )


def _device_replay_payload_from_search_result(
    search_result: CompactSearchResultV1,
    *,
    replay_payload_handle: str,
    device: Any | None = None,
) -> CompactDeviceSearchReplayPayloadV1:
    import torch

    handle = str(replay_payload_handle)
    if not handle:
        raise ReplayCompatibilityError("device replay payload handle must be non-empty")
    target_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )

    def tensor_or_none(value: Any | None) -> Any | None:
        if value is None:
            return None
        return torch.as_tensor(value, dtype=torch.float32, device=target_device).contiguous()

    metadata = dict(search_result.metadata)
    metadata.update(
        {
            "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
            "phase": "replay_critical_device",
            "search_result_schema_id": "curvyzero_compact_search_result/v1",
            "search_impl": str(metadata.get("search_impl") or ""),
            "num_simulations": int(metadata.get("num_simulations") or 0),
            "active_root_count": int(np.asarray(search_result.root_index).size),
            "search_replay_payload_digest": (
                compact_search_deferred_replay_payload_digest_v1(handle)
            ),
            "replay_payload_origin": f"compact_owner_search_service:{handle}",
            "device_replay_payload": True,
            "device_replay_payload_device": str(target_device),
            "host_search_payload_fallback_allowed": False,
            "compact_owner_search_owner_materialized_device_replay_payload": True,
        }
    )
    return CompactDeviceSearchReplayPayloadV1(
        replay_payload_handle=handle,
        root_index=np.asarray(search_result.root_index, dtype=np.int32).copy(),
        env_row=np.asarray(search_result.env_row, dtype=np.int32).copy(),
        player=np.asarray(search_result.player, dtype=np.int16).copy(),
        policy_env_id=np.asarray(search_result.policy_env_id, dtype=np.int64).copy(),
        visit_policy=torch.as_tensor(
            search_result.visit_policy,
            dtype=torch.float32,
            device=target_device,
        ).contiguous(),
        root_value=torch.as_tensor(
            search_result.root_value,
            dtype=torch.float32,
            device=target_device,
        ).contiguous(),
        raw_visit_counts=tensor_or_none(search_result.raw_visit_counts),
        predicted_value=tensor_or_none(search_result.predicted_value),
        predicted_policy_logits=tensor_or_none(search_result.predicted_policy_logits),
        metadata=metadata,
    )


def _compact_search_result_payload_v1(
    search_result: CompactSearchResultV1,
) -> dict[str, Any]:
    return {
        "schema_id": "curvyzero_compact_owner_search_result_search_payload/v1",
        "payload_transport_kind": COMPACT_OWNER_SEARCH_RESULT_PAYLOAD_TRANSPORT_KIND,
        "payload_json_safe": False,
        "root_index": _array_to_ipc_payload(search_result.root_index),
        "env_row": _array_to_ipc_payload(search_result.env_row),
        "player": _array_to_ipc_payload(search_result.player),
        "policy_env_id": _array_to_ipc_payload(search_result.policy_env_id),
        "selected_action": _array_to_ipc_payload(search_result.selected_action),
        "visit_policy": _array_to_ipc_payload(search_result.visit_policy),
        "root_value": _array_to_ipc_payload(search_result.root_value),
        "raw_visit_counts": _optional_array_to_ipc_payload(search_result.raw_visit_counts),
        "predicted_value": _optional_array_to_ipc_payload(search_result.predicted_value),
        "predicted_policy_logits": _optional_array_to_ipc_payload(
            search_result.predicted_policy_logits
        ),
        "metadata": dict(search_result.metadata),
    }


def _compact_search_result_byte_counts_v1(
    search_result: CompactSearchResultV1,
) -> dict[str, int]:
    optional_bytes = 0
    for value in (
        search_result.raw_visit_counts,
        search_result.predicted_value,
        search_result.predicted_policy_logits,
    ):
        if value is not None:
            optional_bytes += int(np.asarray(value).nbytes)
    return {
        "search_selected_action_bytes": int(np.asarray(search_result.selected_action).nbytes),
        "search_visit_policy_bytes": int(np.asarray(search_result.visit_policy).nbytes),
        "search_root_value_bytes": int(np.asarray(search_result.root_value).nbytes),
        "search_optional_array_bytes": int(optional_bytes),
    }


def compact_search_result_v1_from_owner_search_payload(
    root_batch: CompactRootBatchV1,
    payload: Mapping[str, Any],
) -> CompactSearchResultV1:
    if not isinstance(root_batch, CompactRootBatchV1):
        raise TypeError("root_batch must be CompactRootBatchV1")
    if not isinstance(payload, Mapping):
        raise TypeError("owner-search payload must be a mapping")
    search_payload = payload.get("search_result_payload")
    if not isinstance(search_payload, Mapping):
        raise RuntimeError("owner-search payload is missing search_result_payload")
    if not search_payload:
        payload_search_metadata = payload.get("search_result_metadata")
        if not isinstance(payload_search_metadata, Mapping):
            payload_search_metadata = {}
        active_root_index = np.flatnonzero(root_batch.active_root_mask).astype(
            np.int32,
            copy=False,
        )
        selected = np.asarray(payload.get("selected_action"), dtype=np.int16).reshape(-1)
        if selected.shape != active_root_index.shape:
            raise RuntimeError("owner-search action-only selected_action shape mismatch")
        action_count = int(np.asarray(root_batch.legal_mask).shape[1])
        visit_policy = np.zeros((int(selected.size), action_count), dtype=np.float32)
        if selected.size:
            visit_policy[np.arange(int(selected.size)), selected.astype(np.int64)] = 1.0
        search_payload = {
            "root_index": active_root_index,
            "env_row": root_batch.env_row[active_root_index],
            "player": root_batch.player[active_root_index],
            "policy_env_id": root_batch.policy_env_id[active_root_index],
            "selected_action": selected,
            "visit_policy": visit_policy,
            "root_value": np.zeros((int(selected.size),), dtype=np.float32),
            "raw_visit_counts": None,
            "predicted_value": None,
            "predicted_policy_logits": None,
            "payload_transport_kind": "action_only_owner_cached_replay_v1",
            "payload_json_safe": True,
            "metadata": {
                "schema_id": "curvyzero_compact_search_result/v1",
                "search_impl": str(payload.get("search_impl") or "owner_action_only"),
                "num_simulations": int(payload.get("num_simulations") or 0),
                **dict(payload_search_metadata),
                "compact_owner_search_action_only_result": True,
                "compact_owner_search_owner_materializes_replay": True,
                "replay_payload_handle": str(payload.get("replay_payload_handle") or ""),
                "search_replay_payload_digest": str(
                    compact_search_deferred_replay_payload_digest_v1(
                        str(payload.get("replay_payload_handle") or "")
                    )
                ),
            },
        }
    _validate_owner_search_payload_identity(root_batch, search_payload)
    search_metadata = dict(search_payload.get("metadata") or {})
    search_worker_state = dict(payload.get("search_worker_state") or {})
    action_only_result = bool(search_metadata.get("compact_owner_search_action_only_result"))
    owner_materializes_replay = action_only_result or bool(
        search_metadata.get("compact_owner_search_owner_materializes_replay", False)
    )
    owner_action_feedback = payload.get("owner_action_feedback")
    owner_action_feedback = (
        dict(owner_action_feedback)
        if isinstance(owner_action_feedback, Mapping)
        else _empty_owner_action_feedback_v1()
    )
    boundary_kind = str(
        search_metadata.get("compact_owner_search_boundary_kind")
        or payload.get("compact_owner_search_boundary_kind")
        or "worker_search_parent_slab_commit"
    )
    search_metadata.update(
        {
            "compact_owner_search_slab_proxy": True,
            "compact_owner_search_boundary_kind": boundary_kind,
            "compact_owner_search_parent_slab_commits_replay": (not owner_materializes_replay),
            "compact_owner_search_result_schema_id": str(payload.get("schema_id") or ""),
            "compact_owner_search_owner_kind": str(payload.get("owner_kind") or ""),
            "compact_owner_search_owner_pid": int(payload.get("owner_pid") or 0),
            "compact_owner_search_request_id": int(payload.get("request_id") or 0),
            "compact_owner_search_root_slot_count": int(payload.get("root_slot_count") or 0),
            "compact_owner_search_active_root_count": int(payload.get("active_root_count") or 0),
            "compact_owner_search_request_bytes": int(payload.get("request_bytes") or 0),
            "compact_owner_search_result_bytes": int(payload.get("result_bytes") or 0),
            "compact_owner_search_request_cuda_tensor_count": int(
                payload.get("request_cuda_tensor_count") or 0
            ),
            "compact_owner_search_result_cuda_tensor_count": int(
                payload.get("result_cuda_tensor_count") or 0
            ),
            "compact_owner_search_root_observation_bytes_sent": int(
                payload.get("root_observation_bytes_sent") or 0
            ),
            "compact_owner_search_parent_reconstructed_search_result": (not action_only_result),
            "compact_owner_search_action_only_result": action_only_result,
            "compact_owner_search_owner_materializes_replay": owner_materializes_replay,
            "compact_owner_search_inner_two_phase_action_step": bool(
                payload.get("inner_two_phase_action_step", False)
            ),
            "compact_owner_search_inner_device_replay_payload_deferred": bool(
                payload.get("inner_device_replay_payload_deferred", False)
            ),
            "compact_owner_search_use_inner_two_phase_device_replay": bool(
                search_metadata.get(
                    "compact_owner_search_use_inner_two_phase_device_replay",
                    payload.get("use_inner_two_phase_device_replay", False),
                )
            ),
            "compact_owner_search_replay_payload_handle": str(
                payload.get("replay_payload_handle")
                or search_metadata.get("replay_payload_handle")
                or ""
            ),
            "compact_owner_search_model_state_bytes": int(payload.get("model_state_bytes") or 0),
            "compact_owner_search_model_state_return_count": int(
                payload.get("model_state_return_count") or 0
            ),
            "compact_owner_search_model_state_snapshot_return_count": int(
                payload.get("model_state_snapshot_return_count") or 0
            ),
            "compact_owner_search_search_result_payload_bytes": int(
                payload.get("search_result_payload_bytes") or 0
            ),
            "compact_owner_search_search_result_payload_transport_kind": str(
                search_payload.get("payload_transport_kind") or ""
            ),
            "compact_owner_search_search_result_payload_json_safe": bool(
                search_payload.get("payload_json_safe", True)
            ),
            "compact_owner_search_selected_action_bytes": int(
                payload.get("search_selected_action_bytes") or 0
            ),
            "compact_owner_search_visit_policy_bytes": int(
                payload.get("search_visit_policy_bytes") or 0
            ),
            "compact_owner_search_root_value_bytes": int(
                payload.get("search_root_value_bytes") or 0
            ),
            "compact_owner_search_optional_array_bytes": int(
                payload.get("search_optional_array_bytes") or 0
            ),
            "compact_owner_search_worker_owns_search_state": bool(
                payload.get("worker_owns_search_state", False)
            ),
            "compact_owner_search_worker_owns_replay_state": bool(
                payload.get("worker_owns_replay_state", False)
            ),
            "compact_owner_search_worker_owns_model_state": bool(
                payload.get("worker_owns_model_state", False)
            ),
            "compact_owner_search_consumed_learner_update": bool(
                payload.get("search_consumed_learner_update", False)
            ),
            "compact_owner_search_search_refresh_update_count": int(
                payload.get("search_refresh_update_count") or 0
            ),
            "compact_owner_search_model_state_snapshot_load_count": (
                1 if bool(search_worker_state.get("model_state_snapshot_loaded", False)) else 0
            ),
            "compact_owner_search_model_state_snapshot_load_bytes": int(
                search_worker_state.get("model_state_snapshot_load_bytes") or 0
            ),
            "compact_owner_search_model_state_snapshot_load_sec": float(
                search_worker_state.get("model_state_snapshot_load_sec") or 0.0
            ),
            "compact_owner_search_replay_append_entry_count": int(
                payload.get("replay_append_entry_count") or 0
            ),
            "compact_owner_search_replay_append_transport_entry_count": int(
                payload.get("replay_append_transport_entry_count") or 0
            ),
            "compact_owner_search_replay_append_transition_batch_count": int(
                payload.get("replay_append_transition_batch_count") or 0
            ),
            "compact_owner_search_replay_append_transition_batch_entry_count": int(
                payload.get("replay_append_transition_batch_entry_count") or 0
            ),
            "compact_owner_search_replay_append_count": int(
                payload.get("replay_append_count") or 0
            ),
            "compact_owner_search_learner_update_count": int(
                payload.get("learner_update_count") or 0
            ),
            "compact_owner_search_model_owner_ref_returned": bool(
                payload.get("model_owner_ref_returned", False)
            ),
            "compact_owner_search_model_owner_ref_digest": str(
                payload.get("model_owner_ref_digest") or ""
            ),
            "compact_owner_search_owner_sample_telemetry": dict(
                payload.get("owner_sample_telemetry")
                if isinstance(payload.get("owner_sample_telemetry"), Mapping)
                else {}
            ),
            "compact_owner_search_owner_learner_telemetry": dict(
                payload.get("owner_learner_telemetry")
                if isinstance(payload.get("owner_learner_telemetry"), Mapping)
                else {}
            ),
            **owner_action_feedback,
        }
    )
    result = validate_compact_search_result_v1(
        root_batch,
        selected_action=np.asarray(search_payload["selected_action"], dtype=np.int16),
        visit_policy=np.asarray(search_payload["visit_policy"], dtype=np.float32),
        root_value=np.asarray(search_payload["root_value"], dtype=np.float32),
        search_impl=str(search_metadata.get("search_impl") or payload.get("search_impl") or ""),
        num_simulations=int(
            search_metadata.get("num_simulations", payload.get("num_simulations", 0)) or 0
        ),
        raw_visit_counts=_optional_payload_array(
            search_payload.get("raw_visit_counts"),
            dtype=np.float32,
        ),
        predicted_value=_optional_payload_array(
            search_payload.get("predicted_value"),
            dtype=np.float32,
        ),
        predicted_policy_logits=_optional_payload_array(
            search_payload.get("predicted_policy_logits"),
            dtype=np.float32,
        ),
        metadata=search_metadata,
    )
    validate_compact_search_result_identity_v1(root_batch, result)
    return result


def compact_action_step_v1_from_owner_search_payload_and_root_request(
    root_request: CompactRootBuildRequestV1 | CompactRootActionContextV1,
    payload: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CompactSearchActionStepV1:
    if isinstance(root_request, CompactRootBuildRequestV1):
        root_action_context = compact_root_action_context_v1_from_request(root_request)
    elif isinstance(root_request, CompactRootActionContextV1):
        root_action_context = root_request
    else:
        raise TypeError("root_request must be CompactRootBuildRequestV1")
    if not isinstance(payload, Mapping):
        raise TypeError("owner-search payload must be a mapping")
    active_root_index = np.asarray(
        root_action_context.active_root_index,
        dtype=np.int32,
    ).reshape(-1)
    selected = np.asarray(payload.get("selected_action"), dtype=np.int16).reshape(-1)
    if selected.shape != active_root_index.shape:
        raise RuntimeError("owner-search action-only selected_action shape mismatch")
    if bool((selected < 0).any()) or bool((selected >= ACTION_COUNT).any()):
        raise RuntimeError("owner-search action-only selected_action is illegal")
    legal_mask = np.asarray(root_action_context.active_legal_mask, dtype=np.bool_)
    if legal_mask.shape != (int(selected.size), ACTION_COUNT):
        raise RuntimeError("owner-search action-only legal mask shape mismatch")
    if selected.size and not bool(legal_mask[np.arange(selected.size), selected].all()):
        raise RuntimeError("owner-search action-only selected_action is illegal")
    dense_joint_action: np.ndarray | None = None
    dense_value = payload.get("dense_joint_action")
    dense_present = dense_value is not None and not (
        isinstance(dense_value, tuple) and len(dense_value) == 0
    )
    if dense_present:
        dense = np.asarray(dense_value, dtype=np.int16)
        expected_shape = (
            int(root_action_context.batch_size),
            int(root_action_context.player_count),
        )
        if dense.shape != expected_shape:
            raise RuntimeError("owner-search dense_joint_action shape mismatch")
        payload_shape = payload.get("dense_joint_action_shape")
        shape_present = payload_shape is not None and not (
            isinstance(payload_shape, tuple | list) and len(payload_shape) == 0
        )
        if shape_present:
            reported_shape = tuple(int(dim) for dim in np.asarray(payload_shape).reshape(-1))
            if reported_shape != expected_shape:
                raise RuntimeError("owner-search dense_joint_action shape metadata mismatch")
        if bool((dense < 0).any()) or bool((dense >= ACTION_COUNT).any()):
            raise RuntimeError("owner-search dense_joint_action is out of range")
        env_row_for_dense = np.asarray(root_action_context.env_row, dtype=np.int64).reshape(-1)
        player_for_dense = np.asarray(root_action_context.player, dtype=np.int64).reshape(-1)
        if selected.size and not np.array_equal(
            dense[
                env_row_for_dense,
                player_for_dense,
            ],
            selected,
        ):
            raise RuntimeError("owner-search dense_joint_action selected-action mismatch")
        expected_checksum = int(payload.get("dense_joint_action_checksum") or 0)
        actual_checksum = int(_owner_action_checksum_v1(dense))
        if expected_checksum and expected_checksum != actual_checksum:
            raise RuntimeError("owner-search dense_joint_action checksum mismatch")
        expected_digest = str(payload.get("dense_joint_action_digest") or "")
        actual_digest = str(compact_search_array_digest_v1(dense.reshape(-1)))
        if expected_digest and expected_digest != actual_digest:
            raise RuntimeError("owner-search dense_joint_action digest mismatch")
        dense_joint_action = dense.astype(np.int16, copy=True)
    payload_search_metadata = payload.get("search_result_metadata")
    if not isinstance(payload_search_metadata, Mapping):
        payload_search_metadata = {}
    handle = str(payload.get("replay_payload_handle") or "")
    dense_action_proof_metadata = {
        "compact_owner_search_dense_joint_action_owner_assembled": (
            dense_joint_action is not None
        ),
        "compact_owner_search_dense_joint_action_parent_assembly_avoided": (
            dense_joint_action is not None
        ),
        "compact_owner_search_dense_joint_action_present": dense_joint_action is not None,
        "compact_owner_search_dense_joint_action_fallback_count": (
            0 if dense_joint_action is not None else 1
        ),
        "compact_owner_search_dense_joint_action_fallback_reason": (
            "none" if dense_joint_action is not None else "missing_dense_joint_action"
        ),
        "compact_owner_search_dense_joint_action_checksum": (
            0 if dense_joint_action is None else int(_owner_action_checksum_v1(dense_joint_action))
        ),
        "compact_owner_search_dense_joint_action_shape": (
            ()
            if dense_joint_action is None
            else tuple(int(dim) for dim in dense_joint_action.shape)
        ),
        "compact_owner_search_dense_joint_action_digest": (
            ""
            if dense_joint_action is None
            else str(compact_search_array_digest_v1(dense_joint_action.reshape(-1)))
        ),
        "compact_owner_search_dense_joint_action_bytes": (
            0 if dense_joint_action is None else int(dense_joint_action.nbytes)
        ),
        "compact_owner_search_dense_joint_action_mismatch_count": 0,
    }
    action_metadata = {
        **dict(payload_search_metadata),
        "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
        "phase": "action_critical",
        "search_impl": str(payload.get("search_impl") or "owner_action_only"),
        "num_simulations": int(payload.get("num_simulations") or 0),
        "compact_owner_search_action_only_result": True,
        "compact_owner_search_owner_materializes_replay": True,
        "compact_owner_search_direct_root_build_request_handoff": True,
        "replay_payload_handle": handle,
        "selected_action_digest": compact_search_array_digest_v1(selected),
        "search_replay_payload_digest": str(
            compact_search_deferred_replay_payload_digest_v1(handle)
        ),
        "search_replay_payload_digest_deferred": True,
        **dense_action_proof_metadata,
    }
    if metadata:
        action_metadata.update({str(key): value for key, value in metadata.items()})
    # Metadata from the inner search service can carry its own replay handle.
    # Direct transition replay validates against the outer owner handle here.
    action_metadata["replay_payload_handle"] = handle
    action_metadata["selected_action_digest"] = compact_search_array_digest_v1(selected)
    action_metadata["search_replay_payload_digest"] = str(
        compact_search_deferred_replay_payload_digest_v1(handle)
    )
    action_metadata["search_replay_payload_digest_deferred"] = True
    action_metadata["schema_id"] = COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID
    action_metadata["phase"] = "action_critical"
    action_metadata.update(dense_action_proof_metadata)
    env_row = np.asarray(root_action_context.env_row, dtype=np.int32).reshape(-1)
    player = np.asarray(root_action_context.player, dtype=np.int16).reshape(-1)
    policy_env_id = np.asarray(root_action_context.policy_env_id, dtype=np.int64).reshape(-1)
    return CompactSearchActionStepV1(
        replay_payload_handle=handle,
        root_index=active_root_index.astype(np.int32, copy=True),
        env_row=env_row.astype(np.int32, copy=True),
        player=player.astype(np.int16, copy=True),
        policy_env_id=policy_env_id.astype(np.int64, copy=True),
        selected_action=selected.astype(np.int16, copy=True),
        metadata=action_metadata,
        dense_joint_action=(
            None if dense_joint_action is None else dense_joint_action.astype(np.int16, copy=True)
        ),
    )


def _validate_owner_search_payload_identity(
    root_batch: CompactRootBatchV1,
    search_payload: Mapping[str, Any],
) -> None:
    root_index = np.asarray(search_payload.get("root_index"), dtype=np.int32)
    expected_root_index = np.flatnonzero(root_batch.active_root_mask).astype(
        np.int32,
        copy=False,
    )
    if not np.array_equal(root_index, expected_root_index):
        raise RuntimeError("owner-search payload root_index does not match root batch")
    env_row = np.asarray(search_payload.get("env_row"), dtype=np.int32)
    player = np.asarray(search_payload.get("player"), dtype=np.int16)
    policy_env_id = np.asarray(search_payload.get("policy_env_id"), dtype=np.int64)
    if not np.array_equal(env_row, root_batch.env_row[root_index]):
        raise RuntimeError("owner-search payload env_row does not match root batch")
    if not np.array_equal(player, root_batch.player[root_index]):
        raise RuntimeError("owner-search payload player does not match root batch")
    if not np.array_equal(policy_env_id, root_batch.policy_env_id[root_index]):
        raise RuntimeError("owner-search payload policy_env_id does not match root batch")


def _array_to_ipc_payload(value: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value)).copy()


def _optional_array_to_ipc_payload(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    return _array_to_ipc_payload(value)


def _optional_payload_array(
    value: Any,
    *,
    dtype: Any,
) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _pickle_size_bytes(value: Any) -> int:
    try:
        return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return 0


def _callable_accepts_keyword(function: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword:
            return True
    return False


def _elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def _contains_cuda_tensor(value: Any, *, _seen: set[int] | None = None) -> bool:
    seen = set() if _seen is None else _seen
    if value is None:
        return False
    object_id = id(value)
    if object_id in seen:
        return False
    seen.add(object_id)
    is_cuda = getattr(value, "is_cuda", None)
    if isinstance(is_cuda, bool):
        return is_cuda
    parameters = getattr(value, "parameters", None)
    if callable(parameters):
        try:
            if any(_contains_cuda_tensor(item, _seen=seen) for item in parameters()):
                return True
        except TypeError:
            pass
    buffers = getattr(value, "buffers", None)
    if callable(buffers):
        try:
            if any(_contains_cuda_tensor(item, _seen=seen) for item in buffers()):
                return True
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return any(_contains_cuda_tensor(item, _seen=seen) for item in value.values())
    if isinstance(value, tuple | list | set | frozenset):
        return any(_contains_cuda_tensor(item, _seen=seen) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return any(
            _contains_cuda_tensor(getattr(value, name), _seen=seen)
            for name in value.__dataclass_fields__
        )
    return False


def _host_only_clone_for_process(value: Any, *, _seen: dict[int, Any] | None = None) -> Any:
    if value is None:
        return None
    seen = {} if _seen is None else _seen
    object_id = id(value)
    if object_id in seen:
        return seen[object_id]
    detach = getattr(value, "detach", None)
    if callable(detach) and hasattr(value, "device"):
        cloned_tensor = detach().to("cpu").clone().contiguous()
        seen[object_id] = cloned_tensor
        return cloned_tensor
    try:
        import torch
    except Exception:  # pragma: no cover - torch is present in supported profiles.
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.nn.Module):
        cloned_module = copy.deepcopy(value)
        to_device = getattr(cloned_module, "to", None)
        if callable(to_device):
            cloned_module = to_device("cpu")
        seen[object_id] = cloned_module
        for attr_name, attr_value in list(vars(cloned_module).items()):
            if attr_name in {"_parameters", "_buffers", "_modules"}:
                continue
            setattr(
                cloned_module,
                attr_name,
                _host_only_clone_for_process(attr_value, _seen=seen),
            )
        to_device = getattr(cloned_module, "to", None)
        if callable(to_device):
            cloned_module = to_device("cpu")
        return cloned_module
    if isinstance(value, np.ndarray):
        cloned_array = value.copy()
        seen[object_id] = cloned_array
        return cloned_array
    if isinstance(value, Mapping):
        cloned_mapping = {
            key: _host_only_clone_for_process(item, _seen=seen) for key, item in value.items()
        }
        seen[object_id] = cloned_mapping
        return cloned_mapping
    if isinstance(value, tuple):
        cloned_tuple = tuple(_host_only_clone_for_process(item, _seen=seen) for item in value)
        seen[object_id] = cloned_tuple
        return cloned_tuple
    if isinstance(value, list):
        cloned_list = [_host_only_clone_for_process(item, _seen=seen) for item in value]
        seen[object_id] = cloned_list
        return cloned_list
    if isinstance(value, set):
        cloned_set = {_host_only_clone_for_process(item, _seen=seen) for item in value}
        seen[object_id] = cloned_set
        return cloned_set
    dataclass_fields = getattr(value, "__dataclass_fields__", None)
    if isinstance(dataclass_fields, Mapping) and not isinstance(value, type):
        updates = {
            str(name): _host_only_clone_for_process(getattr(value, str(name)), _seen=seen)
            for name in dataclass_fields
        }
        cloned_dataclass = replace(value, **updates)
        seen[object_id] = cloned_dataclass
        return cloned_dataclass
    if isinstance(value, SimpleNamespace):
        cloned_namespace = SimpleNamespace(
            **{
                key: _host_only_clone_for_process(item, _seen=seen)
                for key, item in vars(value).items()
            }
        )
        seen[object_id] = cloned_namespace
        return cloned_namespace
    return value


def _count_cuda_tensors(value: Any, *, _seen: set[int] | None = None) -> int:
    seen = set() if _seen is None else _seen
    if value is None:
        return 0
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    is_cuda = getattr(value, "is_cuda", None)
    if isinstance(is_cuda, bool):
        return int(is_cuda)
    parameters = getattr(value, "parameters", None)
    if callable(parameters):
        try:
            return sum(_count_cuda_tensors(item, _seen=seen) for item in parameters())
        except TypeError:
            return 0
    buffers = getattr(value, "buffers", None)
    if callable(buffers):
        try:
            return sum(_count_cuda_tensors(item, _seen=seen) for item in buffers())
        except TypeError:
            return 0
    if isinstance(value, Mapping):
        return sum(_count_cuda_tensors(item, _seen=seen) for item in value.values())
    if isinstance(value, tuple | list | set | frozenset):
        return sum(_count_cuda_tensors(item, _seen=seen) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return sum(
            _count_cuda_tensors(getattr(value, name), _seen=seen)
            for name in value.__dataclass_fields__
        )
    return 0


def _compact_process_owner_search_async_learner_initializer(
    learner_factory: Any,
    learner_factory_kwargs: Mapping[str, Any],
) -> None:
    global _PROCESS_OWNER_SEARCH_ASYNC_LEARNER
    global _PROCESS_OWNER_SEARCH_ASYNC_LEARNER_INITIALIZED_COUNT
    global _PROCESS_OWNER_SEARCH_ASYNC_LEARNER_COMPLETED_COUNT
    if not callable(learner_factory):
        raise RuntimeError("process owner-search learner factory is not callable")
    kwargs = _host_only_clone_for_process(dict(learner_factory_kwargs or {}))
    if _contains_cuda_tensor(kwargs):
        cuda_count = int(_count_cuda_tensors(kwargs))
        raise RuntimeError(
            f"process owner-search learner kwargs contain CUDA tensors; found {cuda_count}"
        )
    _PROCESS_OWNER_SEARCH_ASYNC_LEARNER = learner_factory(**kwargs)
    _PROCESS_OWNER_SEARCH_ASYNC_LEARNER_INITIALIZED_COUNT = 1
    _PROCESS_OWNER_SEARCH_ASYNC_LEARNER_COMPLETED_COUNT = 0


def _run_compact_process_owner_search_async_learner(
    request: _CompactOwnerSearchProcessLearnerRequestV1,
) -> Mapping[str, Any]:
    global _PROCESS_OWNER_SEARCH_ASYNC_LEARNER_COMPLETED_COUNT
    worker_started = time.perf_counter()
    learner = _PROCESS_OWNER_SEARCH_ASYNC_LEARNER
    if learner is None:
        raise RuntimeError("process owner-search learner was not initialized")
    request_cuda_count = int(_count_cuda_tensors(request))
    if request_cuda_count:
        raise RuntimeError("process owner-search learner request contains CUDA tensors")
    train_payload = getattr(learner, "train_owner_search_learner_payload", None)
    if not callable(train_payload):
        raise RuntimeError(
            "process owner-search learner requires train_owner_search_learner_payload"
        )
    learner_started = time.perf_counter()
    learner_result = train_payload(
        payload=request.learner_payload,
        request=request.request,
    )
    learner_sec = _elapsed(learner_started)
    _PROCESS_OWNER_SEARCH_ASYNC_LEARNER_COMPLETED_COUNT += 1
    result_cuda_count = int(_count_cuda_tensors(learner_result))
    result: dict[str, Any] = {
        "learner_result": learner_result,
        "worker_runtime": {
            "worker_pid": int(os.getpid()),
            "worker_resource_id": (
                f"local_process:{int(os.getpid())}:compact-owner-search-learner"
            ),
            "process_request_host_only": request_cuda_count == 0,
            "process_request_cuda_tensor_count": int(request_cuda_count),
            "process_result_host_only": result_cuda_count == 0,
            "process_result_cuda_tensor_count": int(result_cuda_count),
            "process_request_bytes": int(request.request_bytes),
            "process_result_bytes": 0,
            "worker_job_wall_sec": float(_elapsed(worker_started)),
            "worker_learner_sec": float(learner_sec),
            "worker_initialized_count": int(_PROCESS_OWNER_SEARCH_ASYNC_LEARNER_INITIALIZED_COUNT),
            "worker_completed_count": int(_PROCESS_OWNER_SEARCH_ASYNC_LEARNER_COMPLETED_COUNT),
            "worker_owns_model_state": True,
        },
    }
    result["worker_runtime"]["process_result_bytes"] = _pickle_size_bytes(result)
    return result


def _compact_process_owner_search_initializer(
    root_provider_factory: Any,
    search_service_factory: Any,
    replay_store_factory: Any | None,
    learner_factory: Any | None,
    root_provider_factory_kwargs: Mapping[str, Any],
    search_service_factory_kwargs: Mapping[str, Any],
    replay_store_factory_kwargs: Mapping[str, Any],
    learner_factory_kwargs: Mapping[str, Any],
) -> None:
    global _PROCESS_OWNER_SEARCH_SERVICE
    _PROCESS_OWNER_SEARCH_SERVICE = _build_compact_process_owner_search_service(
        (
            root_provider_factory,
            search_service_factory,
            replay_store_factory,
            learner_factory,
            root_provider_factory_kwargs,
            search_service_factory_kwargs,
            replay_store_factory_kwargs,
            learner_factory_kwargs,
        )
    )


def _build_compact_process_owner_search_service(
    factory_payload: tuple[
        Any,
        Any,
        Any | None,
        Any | None,
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        bool,
        bool,
        str,
        int,
    ],
) -> CompactOwnerSearchServiceV1:
    (
        root_provider_factory,
        search_service_factory,
        replay_store_factory,
        learner_factory,
        root_provider_factory_kwargs,
        search_service_factory_kwargs,
        replay_store_factory_kwargs,
        learner_factory_kwargs,
        use_inner_two_phase_device_replay,
        async_learner_worker,
        async_learner_worker_kind,
        async_learner_max_pending,
    ) = factory_payload
    root_provider = root_provider_factory(**dict(root_provider_factory_kwargs))
    search_service = search_service_factory(**dict(search_service_factory_kwargs))
    replay_store = (
        None
        if replay_store_factory is None
        else replay_store_factory(**dict(replay_store_factory_kwargs))
    )
    learner = None if learner_factory is None else learner_factory(**dict(learner_factory_kwargs))
    async_learner_worker_adapter = None
    if (
        bool(async_learner_worker)
        and str(async_learner_worker_kind)
        == COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH
    ):
        async_learner_worker_adapter = CompactProcessOwnerSearchLearnerWorkerV1(
            learner_factory=learner_factory,
            learner_factory_kwargs=dict(learner_factory_kwargs),
        )
    return CompactOwnerSearchServiceV1(
        root_provider=root_provider,
        search_service=search_service,
        replay_store=replay_store,
        learner=learner,
        owner_kind=COMPACT_OWNER_SEARCH_KIND_LOCAL_PROCESS,
        use_inner_two_phase_device_replay=bool(use_inner_two_phase_device_replay),
        async_learner_worker=bool(async_learner_worker),
        async_learner_worker_kind=str(async_learner_worker_kind),
        async_learner_max_pending=int(async_learner_max_pending),
        async_learner_worker_adapter=async_learner_worker_adapter,
    )


def _compact_process_owner_search_priority_loop(
    factory_payload: tuple[
        Any,
        Any,
        Any | None,
        Any | None,
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        bool,
        bool,
        str,
        int,
    ],
    command_queue: Any,
    result_queue: Any,
) -> None:
    service = _build_compact_process_owner_search_service(factory_payload)
    maintenance_queue: queue.Queue[_CompactOwnerPriorityCommandV1 | None] = queue.Queue()
    maintenance_thread = threading.Thread(
        target=_compact_process_owner_search_maintenance_loop,
        args=(service, maintenance_queue, result_queue),
        name="compact-owner-search-maintenance",
        daemon=False,
    )
    maintenance_thread.start()
    try:
        while True:
            command = command_queue.get()
            if not isinstance(command, _CompactOwnerPriorityCommandV1):
                continue
            if command.command_kind == _OWNER_PRIORITY_COMMAND_CLOSE:
                maintenance_queue.put(None)
                maintenance_thread.join()
                service.close()
                _compact_process_owner_search_put_result(
                    result_queue,
                    command_id=int(command.command_id),
                    payload={
                        "schema_id": COMPACT_OWNER_SEARCH_PRIORITY_LOOP_SCHEMA_ID,
                        "owner_loop_closed": True,
                    },
                )
                break
            if command.command_kind == _OWNER_PRIORITY_COMMAND_DRAIN_MAINTENANCE:
                maintenance_queue.put(command)
                continue
            _compact_process_owner_search_handle_command(
                service,
                command,
                result_queue,
            )
    except BaseException as exc:
        traceback_text = traceback.format_exc()
        try:
            while True:
                command = command_queue.get_nowait()
                if isinstance(command, _CompactOwnerPriorityCommandV1):
                    _compact_process_owner_search_put_error(
                        result_queue,
                        command_id=int(command.command_id),
                        exc=exc,
                        traceback_text=traceback_text,
                    )
        except queue.Empty:
            pass
        raise


def _compact_process_owner_search_maintenance_loop(
    service: CompactOwnerSearchServiceV1,
    maintenance_queue: queue.Queue[_CompactOwnerPriorityCommandV1 | None],
    result_queue: Any,
) -> None:
    while True:
        command = maintenance_queue.get()
        if command is None:
            break
        _compact_process_owner_search_handle_command(
            service,
            command,
            result_queue,
        )


def _compact_threaded_owner_search_priority_loop(
    service: CompactOwnerSearchServiceV1,
    command_queue: queue.Queue[_CompactOwnerPriorityCommandV1 | None],
    result_queue: queue.Queue[_CompactOwnerPriorityResultV1 | None],
) -> None:
    maintenance_queue: queue.Queue[_CompactOwnerPriorityCommandV1 | None] = queue.Queue()
    maintenance_thread = threading.Thread(
        target=_compact_process_owner_search_maintenance_loop,
        args=(service, maintenance_queue, result_queue),
        name="compact-owner-search-threaded-maintenance",
        daemon=False,
    )
    maintenance_thread.start()
    try:
        while True:
            command = command_queue.get()
            if command is None:
                maintenance_queue.put(None)
                maintenance_thread.join()
                break
            if not isinstance(command, _CompactOwnerPriorityCommandV1):
                continue
            if command.command_kind == _OWNER_PRIORITY_COMMAND_CLOSE:
                maintenance_queue.put(None)
                maintenance_thread.join()
                service.close()
                _compact_process_owner_search_put_result(
                    result_queue,
                    command_id=int(command.command_id),
                    payload={
                        "schema_id": COMPACT_OWNER_SEARCH_PRIORITY_LOOP_SCHEMA_ID,
                        "owner_loop_closed": True,
                    },
                )
                break
            if command.command_kind == _OWNER_PRIORITY_COMMAND_DRAIN_MAINTENANCE:
                maintenance_queue.put(command)
                continue
            _compact_process_owner_search_handle_command(
                service,
                command,
                result_queue,
            )
    except BaseException as exc:
        traceback_text = traceback.format_exc()
        try:
            while True:
                command = command_queue.get_nowait()
                if isinstance(command, _CompactOwnerPriorityCommandV1):
                    _compact_process_owner_search_put_error(
                        result_queue,
                        command_id=int(command.command_id),
                        exc=exc,
                        traceback_text=traceback_text,
                    )
        except queue.Empty:
            pass
        raise


def _compact_process_owner_search_handle_command(
    service: CompactOwnerSearchServiceV1,
    command: _CompactOwnerPriorityCommandV1,
    result_queue: Any,
) -> None:
    try:
        command_kind = str(command.command_kind)
        if command_kind == _OWNER_PRIORITY_COMMAND_RUN:
            request = command.payload
            if _contains_cuda_tensor(request):
                raise RuntimeError("owner-search process received CUDA tensor payload")
            result = service.run(request)
            payload = result.to_dict()
        elif command_kind == _OWNER_PRIORITY_COMMAND_ACTION:
            request = command.payload
            if _contains_cuda_tensor(request):
                raise RuntimeError("owner-search process received CUDA tensor payload")
            result = service.run_action(request)
            payload = service.action_result_payload_for_request(request, result)
        elif command_kind == _OWNER_PRIORITY_COMMAND_DRAIN_MAINTENANCE:
            request = command.payload
            if _contains_cuda_tensor(request):
                raise RuntimeError("owner-search process received CUDA tensor payload")
            result = service.drain_maintenance(request)
            payload = result.to_dict()
        else:
            raise RuntimeError(f"unknown owner-search priority command: {command_kind}")
        if _contains_cuda_tensor(payload):
            raise RuntimeError("owner-search process result contains CUDA tensor payload")
        _compact_process_owner_search_put_result(
            result_queue,
            command_id=int(command.command_id),
            payload=payload,
        )
    except BaseException as exc:
        _compact_process_owner_search_put_error(
            result_queue,
            command_id=int(command.command_id),
            exc=exc,
            traceback_text=traceback.format_exc(),
        )


def _compact_process_owner_search_put_result(
    result_queue: Any,
    *,
    command_id: int,
    payload: dict[str, Any],
) -> None:
    result_queue.put(
        _CompactOwnerPriorityResultV1(
            command_id=int(command_id),
            ok=True,
            payload=dict(payload),
        )
    )


def _compact_process_owner_search_put_error(
    result_queue: Any,
    *,
    command_id: int,
    exc: BaseException,
    traceback_text: str,
) -> None:
    result_queue.put(
        _CompactOwnerPriorityResultV1(
            command_id=int(command_id),
            ok=False,
            error_message=f"{type(exc).__name__}: {exc}",
            traceback_text=str(traceback_text),
        )
    )


def _compact_process_owner_search_run(
    request: CompactOwnerSearchRequestV1,
) -> dict[str, Any]:
    if _contains_cuda_tensor(request):
        raise RuntimeError("owner-search process received CUDA tensor payload")
    service = _PROCESS_OWNER_SEARCH_SERVICE
    if service is None:
        raise RuntimeError("owner-search process worker was not initialized")
    result = service.run(request)
    payload = result.to_dict()
    if _contains_cuda_tensor(payload):
        raise RuntimeError("owner-search process result contains CUDA tensor payload")
    return payload


def _compact_process_owner_search_run_action(
    request: CompactOwnerSearchRequestV1,
) -> dict[str, Any]:
    if _contains_cuda_tensor(request):
        raise RuntimeError("owner-search process received CUDA tensor payload")
    service = _PROCESS_OWNER_SEARCH_SERVICE
    if service is None:
        raise RuntimeError("owner-search process worker was not initialized")
    result = service.run_action(request)
    payload = service.action_result_payload_for_request(request, result)
    if _contains_cuda_tensor(payload):
        raise RuntimeError("owner-search process action result contains CUDA tensor payload")
    return payload


def _compact_process_owner_search_drain_maintenance(
    request: CompactOwnerMaintenanceDrainRequestV1,
) -> dict[str, Any]:
    if _contains_cuda_tensor(request):
        raise RuntimeError("owner-search process received CUDA tensor payload")
    service = _PROCESS_OWNER_SEARCH_SERVICE
    if service is None:
        raise RuntimeError("owner-search process worker was not initialized")
    result = service.drain_maintenance(request)
    payload = result.to_dict()
    if _contains_cuda_tensor(payload):
        raise RuntimeError("owner-search maintenance result contains CUDA tensor payload")
    return payload


__all__ = [
    "COMPACT_OWNER_MAINTENANCE_DRAIN_REQUEST_SCHEMA_ID",
    "COMPACT_OWNER_MAINTENANCE_DRAIN_RESULT_SCHEMA_ID",
    "COMPACT_OWNER_ACTION_RESULT_SLOT_STUB_SCHEMA_ID",
    "COMPACT_OWNER_ACTION_RESULT_SLOT_TABLE_SCHEMA_ID",
    "COMPACT_OWNER_SEARCH_KIND_IN_PROCESS",
    "COMPACT_OWNER_SEARCH_KIND_INLINE",
    "COMPACT_OWNER_SEARCH_KIND_INLINE_BACKGROUND",
    "COMPACT_OWNER_SEARCH_KIND_LOCAL_PROCESS",
    "COMPACT_OWNER_SEARCH_KIND_THREADED",
    "COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_IN_PROCESS_THREAD",
    "COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_KINDS",
    "COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH",
    "COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_NONE",
    "COMPACT_OWNER_SEARCH_REQUEST_SCHEMA_ID",
    "COMPACT_OWNER_SEARCH_RESULT_SCHEMA_ID",
    "COMPACT_OWNER_SEARCH_SERVICE_SCHEMA_ID",
    "COMPACT_DIRECT_ROOT_STORE_SCHEMA_ID",
    "COMPACT_SHARED_MEMORY_ARRAY_SCHEMA_ID",
    "COMPACT_SHARED_MEMORY_ROOT_STORE_SCHEMA_ID",
    "CompactOwnerSearchRequestV1",
    "CompactOwnerSearchResultV1",
    "CompactDirectRootStoreV1",
    "CompactInlineBackgroundOwnerSearchWorkerV1",
    "CompactInlineOwnerSearchWorkerV1",
    "CompactLazyInlineBackgroundOwnerSearchSlabProxyV1",
    "CompactLazyInlineOwnerSearchSlabProxyV1",
    "CompactLazyOwnerSearchSlabProxyV1",
    "CompactLazyThreadedOwnerSearchSlabProxyV1",
    "CompactOwnerMaintenanceDrainRequestV1",
    "CompactOwnerMaintenanceDrainResultV1",
    "CompactOwnerActionResultSlotTableV1",
    "CompactOwnerSearchSlabProxyV1",
    "CompactOwnerSearchServiceV1",
    "CompactProcessOwnerSearchLearnerWorkerV1",
    "CompactProcessOwnerSearchWorkerV1",
    "CompactThreadedOwnerSearchWorkerV1",
    "CompactResidentSharedMemoryRootProviderV1",
    "CompactSharedMemoryArraySpecV1",
    "CompactSharedMemoryRootStoreSpecV1",
    "CompactSharedMemoryRootStoreV1",
    "build_compact_resident_shared_memory_root_provider_v1",
    "build_compact_shared_memory_root_provider_v1",
    "compact_search_result_v1_from_owner_search_payload",
]
