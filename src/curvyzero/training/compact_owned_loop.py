"""Compact-owned loop orchestration boundary.

This module is intentionally small.  It owns compact replay sampling cadence,
learner-edge calls, and policy-version lineage for the profile-only compact
candidate without claiming to be stock LightZero training.
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
import multiprocessing as mp
import os
import pickle
import tempfile
import time
from types import SimpleNamespace
from typing import Any

from curvyzero.training.compact_coach_compatibility import (
    build_profile_only_compact_coach_report_v1,
)


COMPACT_OWNED_LOOP_SCHEMA_ID = "curvyzero_compact_owned_loop/v1"
COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD = "in_process_thread"
COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS = "local_process"
COMPACT_SAMPLE_LEARNER_WORKER_KINDS = (
    COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD,
    COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS,
)
COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1 = "durable_entry_v1"
COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1 = "scalar_ref_v1"
COMPACT_REPLAY_APPEND_TRANSPORT_KINDS = (
    COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1,
    COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1,
)
COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1 = "result_v1"
COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1 = "snapshot_file_v1"
COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1 = "owner_ref_v1"
COMPACT_MODEL_STATE_TRANSPORT_KINDS = (
    COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
    COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1,
    COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
)
COMPACT_MODEL_STATE_SNAPSHOT_FILE_SCHEMA_ID = (
    "curvyzero_compact_owned_loop_model_state_snapshot_file/v1"
)
COMPACT_PROVIDER_BOOTSTRAP_STACK_DEPTH = 4
_DEFERRED_LEARNER_MODEL_STATE_DICT_KEY = "_compact_owned_loop_deferred_learner_model_state_dict"
_DEFERRED_LEARNER_MODEL_STATE_DIGEST_KEY = "_compact_owned_loop_deferred_learner_model_state_digest"
_DEFERRED_LEARNER_MODEL_OBJECT_ID_KEY = "_compact_owned_loop_deferred_learner_model_object_id"
_DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY = (
    "_compact_owned_loop_deferred_learner_model_state_snapshot"
)
_DEFERRED_LEARNER_MODEL_OWNER_REF_KEY = (
    "_compact_owned_loop_deferred_learner_model_owner_ref"
)
_PRIVATE_LEARNER_RESULT_KEYS = frozenset(
    {
        _DEFERRED_LEARNER_MODEL_STATE_DICT_KEY,
        _DEFERRED_LEARNER_MODEL_STATE_DIGEST_KEY,
        _DEFERRED_LEARNER_MODEL_OBJECT_ID_KEY,
        _DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY,
        _DEFERRED_LEARNER_MODEL_OWNER_REF_KEY,
    }
)
_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX = (
    "compact_replay_fixed_soa_learner_batch_handle_ring"
)
_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX = (
    "compact_owned_loop_learner_resident_batch_handle"
)
_PROCESS_WORKER_REPLAY_STORE_CLASS: Any | None = None
_PROCESS_WORKER_REPLAY_STORE: Any | None = None
_PROCESS_WORKER_LEARNER: Any | None = None
_PROCESS_WORKER_OBSERVATION_PROVIDER: Any | None = None
_PROCESS_WORKER_INITIALIZED_COUNT = 0
_PROCESS_WORKER_COMPLETED_COUNT = 0
_PROCESS_WORKER_REPLAY_APPEND_COUNT = 0


@dataclass(frozen=True, slots=True)
class CompactPolicyVersionRefV1:
    """Lineage for compact-owned search/replay/learner handoff."""

    policy_version_ref: str
    policy_source: str
    model_version_ref: str | None = None


@dataclass(frozen=True, slots=True)
class CompactOwnedLoopConfigV1:
    """Profile-only compact loop ownership knobs."""

    sample_batch_size: int
    sample_interval: int
    replay_capacity: int
    learner_train_steps: int
    num_unroll_steps: int
    sample_seed_base: int
    learner_impl: str
    require_next_targets: bool
    capture_replay_store_state: bool = False
    defer_learner_gate: bool = False
    defer_learner_gate_max_pending: int = 2
    defer_sample_learner_gate: bool = False
    defer_sample_learner_gate_max_pending: int = 1
    defer_sample_learner_model_state_interval: int = 1
    defer_sample_learner_model_state_transport_kind: str = (
        COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1
    )
    defer_sample_learner_replay_append_transport_kind: str = (
        COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1
    )
    fused_learner_batch: bool = False
    profile_only: bool = True
    calls_train_muzero: bool = False
    touches_live_runs: bool = False


@dataclass(frozen=True, slots=True)
class CompactOwnedLoopStepResultV1:
    """Delta from one compact-owned loop handoff."""

    appended_replay_rows: bool
    sampled: bool
    trained: bool
    sample_result: Mapping[str, Any] | None
    learner_result: Mapping[str, Any] | None
    telemetry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactSampleLearnerWorkRequestV1:
    """One replay-sample-plus-learner job submitted outside actor collection."""

    request_id: int
    replay_store: Any
    replay_snapshot: Any
    learner: Any
    seed: int
    sample_batch_size: int
    require_next_targets: bool
    num_unroll_steps: int
    fused_learner_batch: bool
    train_steps: int
    policy_version_ref: str
    model_version_ref: str | None
    policy_source: str
    return_model_state: bool = True
    model_state_transport_kind: str = COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1
    replay_append_entries: tuple[Any, ...] = ()
    provider_bootstrap_steps: tuple[Any, ...] = ()
    replay_store_metadata: Mapping[str, Any] | None = None
    replay_capacity: int | None = None
    replay_snapshot_version: int = 0
    full_replay_snapshot_sent: bool = True


@dataclass(frozen=True, slots=True)
class CompactProcessSampleLearnerTransportRequestV1:
    """Host-side payload actually sent to a process sample+learner worker."""

    request_id: int
    replay_snapshot: Any
    seed: int
    sample_batch_size: int
    require_next_targets: bool
    num_unroll_steps: int
    fused_learner_batch: bool
    train_steps: int
    policy_version_ref: str
    model_version_ref: str | None
    policy_source: str
    return_model_state: bool = True
    model_state_transport_kind: str = COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1
    replay_append_entries: tuple[Any, ...] = ()
    provider_bootstrap_steps: tuple[Any, ...] = ()
    replay_store_metadata: Mapping[str, Any] | None = None
    replay_capacity: int | None = None
    replay_snapshot_version: int = 0
    full_replay_snapshot_sent: bool = True
    replay_append_entry_count: int = 0
    replay_append_index_row_count: int = 0
    replay_append_entry_bytes: int = 0
    replay_append_host_observation_bytes: int = 0
    replay_append_resident_snapshot_count: int = 0
    replay_append_resident_snapshot_bytes: int = 0
    replay_append_compact_batch_bytes: int = 0
    replay_append_step_payload_bytes: int = 0
    replay_append_render_state_bytes: int = 0
    provider_bootstrap_step_count: int = 0
    provider_bootstrap_step_bytes: int = 0
    provider_bootstrap_host_observation_bytes: int = 0
    provider_bootstrap_resident_snapshot_count: int = 0
    provider_bootstrap_resident_snapshot_bytes: int = 0
    provider_bootstrap_render_state_bytes: int = 0
    request_bytes: int = 0
    snapshot_host_clone_used: bool = False


@dataclass(frozen=True, slots=True)
class CompactPendingSampleLearnerWorkV1:
    """Submitted sample+learner job plus proof metadata."""

    request_id: int
    handle: Any
    submitted_at: float
    snapshot_version: int
    return_model_state: bool = False


@dataclass(frozen=True, slots=True)
class CompactOwnedLoopResultV1:
    """Final compact-owned loop telemetry and optional durable metadata."""

    telemetry: dict[str, Any]
    replay_store_state_metadata: dict[str, Any] | None


class CompactOwnedLoopV1:
    """Own compact replay-store sampling and learner-edge calls."""

    profile_only = True
    calls_train_muzero = False
    touches_live_runs = False

    def __init__(
        self,
        *,
        config: CompactOwnedLoopConfigV1,
        policy_version: CompactPolicyVersionRefV1,
        replay_store: Any,
        learner: Any | None = None,
        sample_learner_worker: Any | None = None,
    ) -> None:
        _validate_config(config)
        _validate_policy_version(policy_version)
        self.config = config
        self.policy_version = policy_version
        self.replay_store = replay_store
        self.learner = learner
        capacity = int(getattr(replay_store, "capacity", int(config.replay_capacity)))
        if capacity != int(config.replay_capacity):
            raise ValueError("compact-owned loop replay capacity mismatch")
        self._previous_step: Any | None = None
        self.record_step_calls = 0
        self.appended_replay_entry_count = 0
        self.sample_gate_calls = 0
        self.sample_gate_opportunities = 0
        self.sample_gate_skipped_count = 0
        self.sample_gate_index_rows = 0
        self.sample_gate_target_rows = 0
        self.sample_gate_sample_rows = 0
        self.sample_gate_sec = 0.0
        self.sample_gate_last_telemetry: dict[str, Any] = {}
        self.sample_gate_last_sample_metadata: dict[str, Any] = {}
        self.learner_gate_calls = 0
        self.learner_gate_updates = 0
        self.learner_gate_sample_rows = 0
        self.learner_gate_input_bytes = 0
        self.learner_gate_sec = 0.0
        self.learner_gate_last_telemetry: dict[str, Any] = {}
        self.learner_gate_resident_batch_handle_requested_count = 0
        self.learner_gate_resident_batch_handle_consumed_count = 0
        self.learner_gate_resident_batch_handle_fallback_count = 0
        self.learner_gate_resident_batch_handle_last_consumed = False
        self.learner_gate_resident_batch_handle_last_schema_id = "none"
        self.learner_gate_resident_batch_handle_last_handle_id = 0
        self.learner_gate_resident_batch_handle_last_snapshot_version = 0
        self.learner_gate_resident_batch_handle_last_request_checksum = 0
        self.learner_gate_resident_batch_handle_last_sample_row_count = 0
        self.learner_gate_resident_batch_handle_last_target_row_count = 0
        self.learner_gate_resident_batch_handle_last_fallback_count = 0
        self.learner_gate_resident_batch_handle_last_fallback_reason = "none"
        self.deferred_learner_submit_count = 0
        self.deferred_learner_completed_count = 0
        self.deferred_learner_wait_count = 0
        self.deferred_learner_wait_sec = 0.0
        self.deferred_learner_last_wait_sec = 0.0
        self.deferred_learner_max_pending_observed = 0
        self.deferred_learner_actor_steps_while_pending = 0
        self.deferred_learner_policy_lag_max = 0
        self.deferred_sample_learner_submit_count = 0
        self.deferred_sample_learner_completed_count = 0
        self.deferred_sample_learner_wait_count = 0
        self.deferred_sample_learner_wait_sec = 0.0
        self.deferred_sample_learner_last_wait_sec = 0.0
        self.deferred_sample_learner_max_pending_observed = 0
        self.deferred_sample_learner_actor_steps_while_pending = 0
        self.deferred_sample_learner_policy_lag_max = 0
        self.deferred_sample_learner_last_submitted_request_id = 0
        self.deferred_sample_learner_last_completed_request_id = 0
        self.deferred_sample_learner_last_submitted_snapshot_version = 0
        self.deferred_sample_learner_last_completed_snapshot_version = 0
        self.deferred_sample_learner_last_completed_worker_pid = 0
        self.deferred_sample_learner_last_completed_worker_resource_id = "none"
        self.deferred_sample_learner_last_completed_worker_cuda_device = "none"
        self.deferred_sample_learner_last_completed_worker_pid_distinct = False
        self.deferred_sample_learner_model_state_apply_count = 0
        self.deferred_sample_learner_last_model_state_applied = False
        self.deferred_sample_learner_model_state_return_count = 0
        self.deferred_sample_learner_model_state_omitted_count = 0
        self.deferred_sample_learner_last_model_state_returned = False
        self.deferred_sample_learner_model_owner_ref_return_count = 0
        self.deferred_sample_learner_last_model_owner_ref_returned = False
        self.deferred_sample_learner_last_model_owner_ref_digest = ""
        self.deferred_sample_learner_last_model_owner_ref_worker_pid = 0
        self.deferred_sample_learner_model_state_snapshot_return_count = 0
        self.deferred_sample_learner_model_state_snapshot_publish_bytes = 0
        self.deferred_sample_learner_model_state_snapshot_publish_sec = 0.0
        self.deferred_sample_learner_model_state_snapshot_load_count = 0
        self.deferred_sample_learner_model_state_snapshot_load_bytes = 0
        self.deferred_sample_learner_model_state_snapshot_load_sec = 0.0
        self.deferred_sample_learner_request_host_only = False
        self.deferred_sample_learner_request_cuda_tensor_count = -1
        self.deferred_sample_learner_result_host_only = False
        self.deferred_sample_learner_result_cuda_tensor_count = -1
        self.deferred_sample_learner_snapshot_host_clone_used = False
        self.deferred_sample_learner_request_bytes = 0
        self.deferred_sample_learner_result_bytes = 0
        self.deferred_sample_learner_worker_owns_model_state = False
        self.deferred_sample_learner_worker_owns_replay_store = False
        self.deferred_sample_learner_worker_model_initialized_count = 0
        self.deferred_sample_learner_worker_completed_count = 0
        self.deferred_sample_learner_worker_job_wall_sec = 0.0
        self.deferred_sample_learner_worker_inner_job_wall_sec = 0.0
        self.deferred_sample_learner_worker_replay_prepare_sec = 0.0
        self.deferred_sample_learner_worker_sample_sec = 0.0
        self.deferred_sample_learner_worker_learner_sec = 0.0
        self.deferred_sample_learner_worker_model_state_prepare_sec = 0.0
        self.deferred_sample_learner_worker_model_state_fn_sec = 0.0
        self.deferred_sample_learner_worker_model_state_clone_sec = 0.0
        self.deferred_sample_learner_worker_model_state_digest_sec = 0.0
        self.deferred_sample_learner_worker_result_public_sec = 0.0
        self.deferred_sample_learner_worker_result_pickle_sec = 0.0
        self.deferred_sample_learner_full_replay_snapshot_sent = False
        self.deferred_sample_learner_full_replay_snapshot_submit_count = 0
        self.deferred_sample_learner_replay_append_entry_count = 0
        self.deferred_sample_learner_replay_append_index_row_count = 0
        self.deferred_sample_learner_last_replay_append_entry_count = 0
        self.deferred_sample_learner_last_replay_append_index_row_count = 0
        self.deferred_sample_learner_replay_append_entry_bytes = 0
        self.deferred_sample_learner_replay_append_host_observation_bytes = 0
        self.deferred_sample_learner_replay_append_resident_snapshot_count = 0
        self.deferred_sample_learner_replay_append_resident_snapshot_bytes = 0
        self.deferred_sample_learner_replay_append_compact_batch_bytes = 0
        self.deferred_sample_learner_replay_append_step_payload_bytes = 0
        self.deferred_sample_learner_replay_append_render_state_bytes = 0
        self.deferred_sample_learner_provider_bootstrap_step_count = 0
        self.deferred_sample_learner_last_provider_bootstrap_step_count = 0
        self.deferred_sample_learner_provider_bootstrap_step_bytes = 0
        self.deferred_sample_learner_provider_bootstrap_host_observation_bytes = 0
        self.deferred_sample_learner_provider_bootstrap_resident_snapshot_count = 0
        self.deferred_sample_learner_provider_bootstrap_resident_snapshot_bytes = 0
        self.deferred_sample_learner_provider_bootstrap_render_state_bytes = 0
        self.deferred_sample_learner_provider_bootstrap_replay_entry_count = 0
        self.deferred_sample_learner_provider_bootstrap_replay_index_row_count = 0
        self.deferred_sample_learner_provider_bootstrap_learner_call_count = 0
        self.deferred_sample_learner_worker_observation_provider_present = False
        self.deferred_sample_learner_worker_observation_provider_materialized_entry_count = 0
        self.deferred_sample_learner_worker_last_observation_provider_materialized_entry_count = 0
        self.deferred_sample_learner_worker_observation_provider_bootstrap_step_count = 0
        self.deferred_sample_learner_worker_last_observation_provider_bootstrap_step_count = 0
        self.deferred_sample_learner_worker_observation_provider_missing_stack_history_count = 0
        self.deferred_sample_learner_worker_replay_append_count = 0
        self.deferred_sample_learner_worker_replay_entry_count = 0
        self.deferred_sample_learner_worker_replay_index_row_count = 0
        self.deferred_sample_learner_worker_replay_evicted_entry_count = 0
        self.deferred_sample_learner_worker_replay_evicted_index_row_count = 0
        self._learner_executor: ThreadPoolExecutor | None = None
        self._pending_learner_futures: deque[Future[Mapping[str, Any]]] = deque()
        if bool(config.defer_learner_gate) and learner is not None:
            self._learner_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="compact-owned-learner",
            )
        self._sample_learner_worker: Any | None = None
        self._pending_sample_learner_futures: deque[CompactPendingSampleLearnerWorkV1] = deque()
        self._pending_sample_learner_replay_append_entries: list[Any] = []
        self._pending_sample_learner_replay_append_index_rows = 0
        self._pending_sample_learner_provider_bootstrap_steps: list[Any] = []
        self._force_next_sample_learner_model_state = False
        if bool(config.defer_sample_learner_gate) and learner is not None:
            self._sample_learner_worker = sample_learner_worker or _ThreadSampleLearnerWorkerV1()
            prepare = getattr(self._sample_learner_worker, "prepare", None)
            if callable(prepare):
                prepare(replay_store=self.replay_store, learner=learner)
        self._sample_learner_worker_metadata = _sample_learner_worker_metadata(
            self._sample_learner_worker
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return _compact_owned_loop_metadata(
            config=self.config,
            policy_version=self.policy_version,
        )

    def update_policy_version(self, policy_version: CompactPolicyVersionRefV1) -> None:
        _validate_policy_version(policy_version)
        self.policy_version = policy_version

    def force_next_sample_learner_model_state(self) -> None:
        """Ask the next sample+learner job to publish model state for refresh."""

        if any(
            bool(getattr(pending, "return_model_state", False))
            for pending in self._pending_sample_learner_futures
        ):
            return
        self._force_next_sample_learner_model_state = True

    def prime_previous_step(self, step: Any) -> None:
        self._queue_sample_learner_provider_bootstrap_step(step)
        self._previous_step = step

    def record_step(
        self,
        *,
        current_step: Any,
        index_rows: Any | None,
        defer_learner: bool | None = None,
    ) -> CompactOwnedLoopStepResultV1:
        self.record_step_calls += 1
        if self._pending_sample_learner_futures:
            self.deferred_sample_learner_actor_steps_while_pending += 1
        if self._pending_learner_futures:
            self.deferred_learner_actor_steps_while_pending += 1
        sample_result: Mapping[str, Any] | None = None
        completed_sample = self.consume_completed_sample_learner_result(wait=False)
        if completed_sample is not None:
            sample_result = completed_sample.sample_result
        learner_result: Mapping[str, Any] | None = None
        if completed_sample is not None and completed_sample.learner_result is not None:
            learner_result = completed_sample.learner_result
        completed_learner = self.consume_completed_learner_result(wait=False)
        if completed_learner is not None:
            learner_result = _combine_learner_results(learner_result, completed_learner)
        appended = False
        sampled = completed_sample is not None and completed_sample.sampled
        trained = learner_result is not None
        if index_rows is not None:
            if self._previous_step is None:
                self._queue_sample_learner_provider_bootstrap_step(current_step)
                self._previous_step = current_step
                return CompactOwnedLoopStepResultV1(
                    appended_replay_rows=False,
                    sampled=False,
                    trained=False,
                    sample_result=None,
                    learner_result=None,
                    telemetry=self.telemetry(),
                )
            self.replay_store.append(
                previous_step=self._previous_step,
                current_step=current_step,
                index_rows=index_rows,
            )
            self._queue_sample_learner_replay_append_entry(
                previous_step=self._previous_step,
                current_step=current_step,
                index_rows=index_rows,
            )
            appended = True
            self.appended_replay_entry_count += 1
            self.sample_gate_opportunities += 1
            if self.sample_gate_opportunities % int(self.config.sample_interval) != 0:
                self.sample_gate_skipped_count += 1
            else:
                if self._should_defer_sample_learner():
                    completed = self._submit_deferred_sample_learner()
                    if completed is not None:
                        sample_result = completed.sample_result
                        if completed.learner_result is not None:
                            learner_result = _combine_learner_results(
                                learner_result,
                                completed.learner_result,
                            )
                        sampled = sampled or completed.sampled
                        trained = learner_result is not None
                else:
                    sample_result, sample_learner_result = self._sample_and_train(
                        defer_learner=defer_learner
                    )
                    if sample_learner_result is not None:
                        learner_result = _combine_learner_results(
                            learner_result,
                            sample_learner_result,
                        )
                    sampled = True
                    trained = learner_result is not None
        self._previous_step = current_step
        return CompactOwnedLoopStepResultV1(
            appended_replay_rows=appended,
            sampled=sampled,
            trained=trained,
            sample_result=sample_result,
            learner_result=learner_result,
            telemetry=self.telemetry(),
        )

    def sample_and_train_from_store(self) -> CompactOwnedLoopStepResultV1:
        sample_result, learner_result = self._sample_and_train()
        return CompactOwnedLoopStepResultV1(
            appended_replay_rows=False,
            sampled=True,
            trained=learner_result is not None,
            sample_result=sample_result,
            learner_result=learner_result,
            telemetry=self.telemetry(),
        )

    def snapshot_replay_store_state(self) -> Any:
        snapshot = getattr(self.replay_store, "snapshot_durable_state", None)
        if not callable(snapshot):
            raise ValueError("compact-owned loop replay store must expose snapshot_durable_state")
        return snapshot(
            policy_version_ref=self.policy_version.policy_version_ref,
            policy_source=self.policy_version.policy_source,
            model_version_ref=self.policy_version.model_version_ref,
            metadata=self.metadata,
        )

    def replay_store_state_metadata(self) -> dict[str, Any] | None:
        if not bool(self.config.capture_replay_store_state):
            return None
        state = self.snapshot_replay_store_state()
        return dict(getattr(state, "metadata", {}) or {})

    def result(self) -> CompactOwnedLoopResultV1:
        return CompactOwnedLoopResultV1(
            telemetry=self.telemetry(),
            replay_store_state_metadata=self.replay_store_state_metadata(),
        )

    @property
    def has_pending_learner_result(self) -> bool:
        return bool(self._pending_learner_futures)

    @property
    def has_pending_sample_learner_result(self) -> bool:
        return bool(self._pending_sample_learner_futures)

    def consume_completed_sample_learner_result(
        self,
        *,
        wait: bool = False,
    ) -> CompactOwnedLoopStepResultV1 | None:
        while self._pending_sample_learner_futures:
            pending = self._pending_sample_learner_futures[0]
            worker = self._sample_learner_worker
            if worker is None:
                raise RuntimeError("pending sample+learner work exists without worker")
            if not wait and not bool(worker.done(pending.handle)):
                break
            started = time.perf_counter()
            result = dict(worker.result(pending.handle))
            wait_sec = max(0.0, time.perf_counter() - started)
            self._pending_sample_learner_futures.popleft()
            if wait or wait_sec > 0.0:
                self.deferred_sample_learner_wait_count += 1
                self.deferred_sample_learner_wait_sec += wait_sec
                self.deferred_sample_learner_last_wait_sec = wait_sec
            self.deferred_sample_learner_completed_count += 1
            self.deferred_sample_learner_last_completed_request_id = pending.request_id
            self.deferred_sample_learner_last_completed_snapshot_version = (
                pending.snapshot_version
            )
            worker_runtime = dict(result.get("worker_runtime") or {})
            worker_pid = int(worker_runtime.get("worker_pid") or 0)
            worker_resource_id = str(
                worker_runtime.get("worker_resource_id")
                or self._sample_learner_worker_metadata.get(
                    "compact_owned_loop_sample_learner_worker_resource_id",
                    "none",
                )
            )
            self.deferred_sample_learner_last_completed_worker_pid = worker_pid
            self.deferred_sample_learner_last_completed_worker_resource_id = worker_resource_id
            self.deferred_sample_learner_last_completed_worker_cuda_device = str(
                worker_runtime.get("worker_cuda_device") or "none"
            )
            actor_pid = int(
                self._sample_learner_worker_metadata.get(
                    "compact_owned_loop_actor_search_pid",
                    os.getpid(),
                )
                or os.getpid()
            )
            self.deferred_sample_learner_last_completed_worker_pid_distinct = (
                worker_pid > 0 and worker_pid != actor_pid
            )
            self.deferred_sample_learner_request_host_only = bool(
                worker_runtime.get("process_request_host_only", False)
            )
            self.deferred_sample_learner_request_cuda_tensor_count = int(
                worker_runtime.get("process_request_cuda_tensor_count", -1)
            )
            self.deferred_sample_learner_result_host_only = bool(
                worker_runtime.get("process_result_host_only", False)
            )
            self.deferred_sample_learner_result_cuda_tensor_count = int(
                worker_runtime.get("process_result_cuda_tensor_count", -1)
            )
            self.deferred_sample_learner_snapshot_host_clone_used = bool(
                worker_runtime.get("process_snapshot_host_clone_used", False)
            )
            self.deferred_sample_learner_request_bytes += int(
                worker_runtime.get("process_request_bytes", 0) or 0
            )
            self.deferred_sample_learner_result_bytes += int(
                worker_runtime.get("process_result_bytes", 0) or 0
            )
            self.deferred_sample_learner_replay_append_entry_bytes += int(
                worker_runtime.get("process_replay_append_entry_bytes", 0) or 0
            )
            self.deferred_sample_learner_replay_append_host_observation_bytes += int(
                worker_runtime.get("process_replay_append_host_observation_bytes", 0) or 0
            )
            self.deferred_sample_learner_replay_append_resident_snapshot_count += int(
                worker_runtime.get("process_replay_append_resident_snapshot_count", 0) or 0
            )
            self.deferred_sample_learner_replay_append_resident_snapshot_bytes += int(
                worker_runtime.get("process_replay_append_resident_snapshot_bytes", 0) or 0
            )
            self.deferred_sample_learner_replay_append_compact_batch_bytes += int(
                worker_runtime.get("process_replay_append_compact_batch_bytes", 0) or 0
            )
            self.deferred_sample_learner_replay_append_step_payload_bytes += int(
                worker_runtime.get("process_replay_append_step_payload_bytes", 0) or 0
            )
            self.deferred_sample_learner_replay_append_render_state_bytes += int(
                worker_runtime.get("process_replay_append_render_state_bytes", 0) or 0
            )
            self.deferred_sample_learner_provider_bootstrap_step_bytes += int(
                worker_runtime.get("process_provider_bootstrap_step_bytes", 0) or 0
            )
            self.deferred_sample_learner_provider_bootstrap_host_observation_bytes += int(
                worker_runtime.get(
                    "process_provider_bootstrap_host_observation_bytes",
                    0,
                )
                or 0
            )
            self.deferred_sample_learner_provider_bootstrap_resident_snapshot_count += int(
                worker_runtime.get(
                    "process_provider_bootstrap_resident_snapshot_count",
                    0,
                )
                or 0
            )
            self.deferred_sample_learner_provider_bootstrap_resident_snapshot_bytes += int(
                worker_runtime.get(
                    "process_provider_bootstrap_resident_snapshot_bytes",
                    0,
                )
                or 0
            )
            self.deferred_sample_learner_provider_bootstrap_render_state_bytes += int(
                worker_runtime.get("process_provider_bootstrap_render_state_bytes", 0)
                or 0
            )
            self.deferred_sample_learner_provider_bootstrap_replay_entry_count += int(
                worker_runtime.get("process_provider_bootstrap_replay_entry_count", 0)
                or 0
            )
            self.deferred_sample_learner_provider_bootstrap_replay_index_row_count += int(
                worker_runtime.get("process_provider_bootstrap_replay_index_row_count", 0)
                or 0
            )
            self.deferred_sample_learner_provider_bootstrap_learner_call_count += int(
                worker_runtime.get("process_provider_bootstrap_learner_call_count", 0)
                or 0
            )
            self.deferred_sample_learner_worker_observation_provider_present = bool(
                worker_runtime.get("worker_observation_provider_present", False)
            )
            last_provider_bootstrap = int(
                worker_runtime.get("worker_observation_provider_bootstrap_step_count", 0)
                or 0
            )
            self.deferred_sample_learner_worker_last_observation_provider_bootstrap_step_count = (
                last_provider_bootstrap
            )
            self.deferred_sample_learner_worker_observation_provider_bootstrap_step_count += (
                last_provider_bootstrap
            )
            self.deferred_sample_learner_worker_observation_provider_missing_stack_history_count += int(
                worker_runtime.get(
                    "worker_observation_provider_missing_stack_history_count",
                    0,
                )
                or 0
            )
            last_provider_materialized = int(
                worker_runtime.get(
                    "worker_observation_provider_materialized_entry_count",
                    0,
                )
                or 0
            )
            self.deferred_sample_learner_worker_last_observation_provider_materialized_entry_count = (
                last_provider_materialized
            )
            self.deferred_sample_learner_worker_observation_provider_materialized_entry_count += (
                last_provider_materialized
            )
            self.deferred_sample_learner_worker_owns_model_state = bool(
                worker_runtime.get("worker_owns_model_state", False)
            )
            self.deferred_sample_learner_worker_owns_replay_store = bool(
                worker_runtime.get("worker_owns_replay_store", False)
            )
            self.deferred_sample_learner_worker_model_initialized_count = int(
                worker_runtime.get("worker_model_initialized_count", 0) or 0
            )
            self.deferred_sample_learner_worker_completed_count = int(
                worker_runtime.get("worker_completed_count", 0) or 0
            )
            self.deferred_sample_learner_worker_job_wall_sec += float(
                worker_runtime.get(
                    "process_worker_job_wall_sec",
                    worker_runtime.get("worker_job_wall_sec", 0.0),
                )
                or 0.0
            )
            self.deferred_sample_learner_worker_inner_job_wall_sec += float(
                worker_runtime.get("worker_job_wall_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_replay_prepare_sec += float(
                worker_runtime.get("process_worker_replay_prepare_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_sample_sec += float(
                worker_runtime.get("worker_sample_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_learner_sec += float(
                worker_runtime.get("worker_learner_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_model_state_prepare_sec += float(
                worker_runtime.get("worker_model_state_prepare_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_model_state_fn_sec += float(
                worker_runtime.get("worker_model_state_fn_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_model_state_clone_sec += float(
                worker_runtime.get("worker_model_state_clone_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_model_state_digest_sec += float(
                worker_runtime.get("worker_model_state_digest_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_result_public_sec += float(
                worker_runtime.get("process_worker_result_public_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_result_pickle_sec += float(
                worker_runtime.get("process_result_pickle_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_worker_replay_append_count = int(
                worker_runtime.get("worker_replay_append_count", 0) or 0
            )
            self.deferred_sample_learner_worker_replay_entry_count = int(
                worker_runtime.get("worker_replay_entry_count", 0) or 0
            )
            self.deferred_sample_learner_worker_replay_index_row_count = int(
                worker_runtime.get("worker_replay_index_row_count", 0) or 0
            )
            self.deferred_sample_learner_worker_replay_evicted_entry_count = int(
                worker_runtime.get("worker_replay_evicted_entry_count", 0) or 0
            )
            self.deferred_sample_learner_worker_replay_evicted_index_row_count = int(
                worker_runtime.get("worker_replay_evicted_index_row_count", 0) or 0
            )
            sample_result = result.get("sample_result")
            learner_result = result.get("learner_result")
            model_state_returned = bool(
                worker_runtime.get(
                    "process_model_state_returned",
                    bool(
                        isinstance(learner_result, Mapping)
                        and (
                            _DEFERRED_LEARNER_MODEL_STATE_DICT_KEY in learner_result
                            or _DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY in learner_result
                        )
                    ),
                )
            )
            model_state_omitted = bool(
                worker_runtime.get("process_model_state_omitted", False)
            )
            if model_state_returned:
                self.deferred_sample_learner_model_state_return_count += 1
            if model_state_omitted:
                self.deferred_sample_learner_model_state_omitted_count += 1
            if bool(worker_runtime.get("process_model_state_snapshot_returned", False)):
                self.deferred_sample_learner_model_state_snapshot_return_count += 1
            model_owner_ref = (
                dict(learner_result.get(_DEFERRED_LEARNER_MODEL_OWNER_REF_KEY) or {})
                if isinstance(learner_result, Mapping)
                else {}
            )
            model_owner_ref_returned = bool(
                worker_runtime.get("process_model_owner_ref_returned", False)
                or model_owner_ref
            )
            if model_owner_ref_returned:
                self.deferred_sample_learner_model_owner_ref_return_count += 1
                self.deferred_sample_learner_last_model_owner_ref_digest = str(
                    model_owner_ref.get("model_state_digest") or ""
                )
                self.deferred_sample_learner_last_model_owner_ref_worker_pid = int(
                    model_owner_ref.get("worker_pid", worker_pid) or 0
                )
            self.deferred_sample_learner_last_model_owner_ref_returned = (
                model_owner_ref_returned
            )
            self.deferred_sample_learner_model_state_snapshot_publish_bytes += int(
                worker_runtime.get("process_model_state_snapshot_bytes", 0) or 0
            )
            self.deferred_sample_learner_model_state_snapshot_publish_sec += float(
                worker_runtime.get("process_model_state_snapshot_write_sec", 0.0) or 0.0
            )
            self.deferred_sample_learner_last_model_state_returned = model_state_returned
            self._apply_deferred_learner_model_state(learner_result)
            if sample_result is not None:
                self._record_sample_result(sample_result)
            if learner_result is not None:
                self._record_learner_result(learner_result)
            return CompactOwnedLoopStepResultV1(
                appended_replay_rows=False,
                sampled=sample_result is not None,
                trained=learner_result is not None,
                sample_result=sample_result,
                learner_result=learner_result,
                telemetry=self.telemetry(),
            )
        return None

    def consume_completed_learner_result(
        self,
        *,
        wait: bool = False,
    ) -> Mapping[str, Any] | None:
        combined: Mapping[str, Any] | None = None
        while self._pending_learner_futures:
            future = self._pending_learner_futures[0]
            if not wait and not future.done():
                break
            started = time.perf_counter()
            result = future.result()
            wait_sec = max(0.0, time.perf_counter() - started)
            self._pending_learner_futures.popleft()
            if wait or wait_sec > 0.0:
                self.deferred_learner_wait_count += 1
                self.deferred_learner_wait_sec += wait_sec
                self.deferred_learner_last_wait_sec = wait_sec
            self.deferred_learner_completed_count += 1
            self._record_learner_result(result)
            combined = _combine_learner_results(combined, result)
        return combined

    def close(self) -> None:
        self.consume_completed_sample_learner_result(wait=True)
        self.consume_completed_learner_result(wait=True)
        executor = self._learner_executor
        self._learner_executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        sample_learner_worker = self._sample_learner_worker
        self._sample_learner_worker = None
        if sample_learner_worker is not None:
            close = getattr(sample_learner_worker, "close", None)
            if callable(close):
                close()

    def telemetry(self) -> dict[str, Any]:
        entry_count = int(getattr(self.replay_store, "entry_count", 0))
        stored_rows = int(getattr(self.replay_store, "stored_index_row_count", 0))
        evicted_entries = int(getattr(self.replay_store, "evicted_entry_count", 0))
        evicted_rows = int(getattr(self.replay_store, "evicted_index_row_count", 0))
        retained_resident_snapshot_count = int(
            getattr(self.replay_store, "retained_resident_snapshot_count", 0)
        )
        retained_resident_snapshot_bytes = int(
            getattr(self.replay_store, "retained_resident_snapshot_bytes", 0)
        )
        metadata = self.metadata
        metadata.update(
            {
                "compact_owned_loop_record_step_calls": int(self.record_step_calls),
                "compact_owned_loop_appended_replay_entry_count": int(
                    self.appended_replay_entry_count
                ),
                "compact_owned_loop_sample_gate_calls": int(self.sample_gate_calls),
                "compact_owned_loop_sample_gate_opportunities": int(self.sample_gate_opportunities),
                "compact_owned_loop_sample_gate_skipped_count": int(self.sample_gate_skipped_count),
                "compact_owned_loop_sample_gate_index_row_count": int(self.sample_gate_index_rows),
                "compact_owned_loop_sample_gate_target_row_count": int(
                    self.sample_gate_target_rows
                ),
                "compact_owned_loop_sample_gate_sample_row_count": int(
                    self.sample_gate_sample_rows
                ),
                "compact_owned_loop_sample_gate_sec": float(self.sample_gate_sec),
                "compact_owned_loop_sample_gate_last_telemetry": dict(
                    self.sample_gate_last_telemetry
                ),
                "compact_owned_loop_sample_gate_last_sample_metadata": dict(
                    self.sample_gate_last_sample_metadata
                ),
                "compact_owned_loop_learner_gate_calls": int(self.learner_gate_calls),
                "compact_owned_loop_learner_gate_updates": int(self.learner_gate_updates),
                "compact_owned_loop_learner_gate_sample_row_count": int(
                    self.learner_gate_sample_rows
                ),
                "compact_owned_loop_learner_gate_input_bytes": int(self.learner_gate_input_bytes),
                "compact_owned_loop_learner_gate_sec": float(self.learner_gate_sec),
                "compact_owned_loop_learner_gate_last_telemetry": dict(
                    self.learner_gate_last_telemetry
                ),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "requested_count"
                ): int(self.learner_gate_resident_batch_handle_requested_count),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "consumed_count"
                ): int(self.learner_gate_resident_batch_handle_consumed_count),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "fallback_count"
                ): int(self.learner_gate_resident_batch_handle_fallback_count),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "last_consumed"
                ): bool(self.learner_gate_resident_batch_handle_last_consumed),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "last_schema_id"
                ): str(self.learner_gate_resident_batch_handle_last_schema_id),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "last_handle_id"
                ): int(self.learner_gate_resident_batch_handle_last_handle_id),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "last_snapshot_version"
                ): int(
                    self.learner_gate_resident_batch_handle_last_snapshot_version
                ),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "last_request_checksum"
                ): int(
                    self.learner_gate_resident_batch_handle_last_request_checksum
                ),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "last_sample_row_count"
                ): int(
                    self.learner_gate_resident_batch_handle_last_sample_row_count
                ),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "last_target_row_count"
                ): int(
                    self.learner_gate_resident_batch_handle_last_target_row_count
                ),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "last_fallback_count"
                ): int(
                    self.learner_gate_resident_batch_handle_last_fallback_count
                ),
                (
                    "compact_owned_loop_learner_resident_batch_handle_"
                    "last_fallback_reason"
                ): str(
                    self.learner_gate_resident_batch_handle_last_fallback_reason
                ),
                "compact_owned_loop_defer_learner_gate": bool(self.config.defer_learner_gate),
                "compact_owned_loop_defer_sample_learner_gate": bool(
                    self.config.defer_sample_learner_gate
                ),
                **self._sample_learner_worker_metadata,
                "compact_owned_loop_fused_learner_batch": bool(self.config.fused_learner_batch),
                "compact_owned_loop_deferred_learner_submit_count": int(
                    self.deferred_learner_submit_count
                ),
                "compact_owned_loop_deferred_learner_completed_count": int(
                    self.deferred_learner_completed_count
                ),
                "compact_owned_loop_deferred_learner_pending": bool(self._pending_learner_futures),
                "compact_owned_loop_deferred_learner_pending_count": int(
                    len(self._pending_learner_futures)
                ),
                "compact_owned_loop_deferred_learner_max_pending": int(
                    self.config.defer_learner_gate_max_pending
                ),
                "compact_owned_loop_deferred_learner_max_pending_observed": int(
                    self.deferred_learner_max_pending_observed
                ),
                "compact_owned_loop_deferred_learner_actor_steps_while_pending": int(
                    self.deferred_learner_actor_steps_while_pending
                ),
                "compact_owned_loop_deferred_learner_policy_lag_current": int(
                    self._deferred_learner_policy_lag_current()
                ),
                "compact_owned_loop_deferred_learner_policy_lag_max": int(
                    self.deferred_learner_policy_lag_max
                ),
                "compact_owned_loop_deferred_learner_wait_count": int(
                    self.deferred_learner_wait_count
                ),
                "compact_owned_loop_deferred_learner_wait_sec": float(
                    self.deferred_learner_wait_sec
                ),
                "compact_owned_loop_deferred_learner_last_wait_sec": float(
                    self.deferred_learner_last_wait_sec
                ),
                "compact_owned_loop_deferred_sample_learner_submit_count": int(
                    self.deferred_sample_learner_submit_count
                ),
                "compact_owned_loop_deferred_sample_learner_completed_count": int(
                    self.deferred_sample_learner_completed_count
                ),
                "compact_owned_loop_deferred_sample_learner_pending": bool(
                    self._pending_sample_learner_futures
                ),
                "compact_owned_loop_deferred_sample_learner_pending_count": int(
                    len(self._pending_sample_learner_futures)
                ),
                "compact_owned_loop_deferred_sample_learner_max_pending": int(
                    self.config.defer_sample_learner_gate_max_pending
                ),
                "compact_owned_loop_deferred_sample_learner_model_state_interval": int(
                    self.config.defer_sample_learner_model_state_interval
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "model_state_transport_kind"
                ): str(self.config.defer_sample_learner_model_state_transport_kind),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_transport_kind"
                ): str(self.config.defer_sample_learner_replay_append_transport_kind),
                "compact_owned_loop_deferred_sample_learner_max_pending_observed": int(
                    self.deferred_sample_learner_max_pending_observed
                ),
                "compact_owned_loop_deferred_sample_learner_actor_steps_while_pending": int(
                    self.deferred_sample_learner_actor_steps_while_pending
                ),
                "compact_owned_loop_deferred_sample_learner_policy_lag_current": int(
                    self._deferred_sample_learner_policy_lag_current()
                ),
                "compact_owned_loop_deferred_sample_learner_policy_lag_max": int(
                    self.deferred_sample_learner_policy_lag_max
                ),
                "compact_owned_loop_deferred_sample_learner_last_submitted_request_id": int(
                    self.deferred_sample_learner_last_submitted_request_id
                ),
                "compact_owned_loop_deferred_sample_learner_last_completed_request_id": int(
                    self.deferred_sample_learner_last_completed_request_id
                ),
                "compact_owned_loop_deferred_sample_learner_last_submitted_snapshot_version": int(
                    self.deferred_sample_learner_last_submitted_snapshot_version
                ),
                "compact_owned_loop_deferred_sample_learner_last_completed_snapshot_version": int(
                    self.deferred_sample_learner_last_completed_snapshot_version
                ),
                "compact_owned_loop_deferred_sample_learner_last_completed_worker_pid": int(
                    self.deferred_sample_learner_last_completed_worker_pid
                ),
                "compact_owned_loop_deferred_sample_learner_last_completed_worker_resource_id": str(
                    self.deferred_sample_learner_last_completed_worker_resource_id
                ),
                "compact_owned_loop_deferred_sample_learner_last_completed_worker_cuda_device": str(
                    self.deferred_sample_learner_last_completed_worker_cuda_device
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "last_completed_worker_pid_distinct_from_actor_search"
                ): bool(self.deferred_sample_learner_last_completed_worker_pid_distinct),
                "compact_owned_loop_deferred_sample_learner_model_state_apply_count": int(
                    self.deferred_sample_learner_model_state_apply_count
                ),
                "compact_owned_loop_deferred_sample_learner_last_model_state_applied": bool(
                    self.deferred_sample_learner_last_model_state_applied
                ),
                "compact_owned_loop_deferred_sample_learner_model_state_return_count": int(
                    self.deferred_sample_learner_model_state_return_count
                ),
                "compact_owned_loop_deferred_sample_learner_model_state_omitted_count": int(
                    self.deferred_sample_learner_model_state_omitted_count
                ),
                "compact_owned_loop_deferred_sample_learner_last_model_state_returned": bool(
                    self.deferred_sample_learner_last_model_state_returned
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "model_owner_ref_return_count"
                ): int(self.deferred_sample_learner_model_owner_ref_return_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "last_model_owner_ref_returned"
                ): bool(self.deferred_sample_learner_last_model_owner_ref_returned),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "last_model_owner_ref_digest"
                ): str(self.deferred_sample_learner_last_model_owner_ref_digest),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "last_model_owner_ref_worker_pid"
                ): int(self.deferred_sample_learner_last_model_owner_ref_worker_pid),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "model_state_snapshot_return_count"
                ): int(self.deferred_sample_learner_model_state_snapshot_return_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "model_state_snapshot_publish_bytes"
                ): int(self.deferred_sample_learner_model_state_snapshot_publish_bytes),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "model_state_snapshot_publish_sec"
                ): float(self.deferred_sample_learner_model_state_snapshot_publish_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "model_state_snapshot_load_count"
                ): int(self.deferred_sample_learner_model_state_snapshot_load_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "model_state_snapshot_load_bytes"
                ): int(self.deferred_sample_learner_model_state_snapshot_load_bytes),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "model_state_snapshot_load_sec"
                ): float(self.deferred_sample_learner_model_state_snapshot_load_sec),
                "compact_owned_loop_deferred_sample_learner_request_host_only": bool(
                    self.deferred_sample_learner_request_host_only
                ),
                "compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count": int(
                    self.deferred_sample_learner_request_cuda_tensor_count
                ),
                "compact_owned_loop_deferred_sample_learner_result_host_only": bool(
                    self.deferred_sample_learner_result_host_only
                ),
                "compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count": int(
                    self.deferred_sample_learner_result_cuda_tensor_count
                ),
                "compact_owned_loop_deferred_sample_learner_snapshot_host_clone_used": bool(
                    self.deferred_sample_learner_snapshot_host_clone_used
                ),
                "compact_owned_loop_deferred_sample_learner_request_bytes": int(
                    self.deferred_sample_learner_request_bytes
                ),
                "compact_owned_loop_deferred_sample_learner_result_bytes": int(
                    self.deferred_sample_learner_result_bytes
                ),
                "compact_owned_loop_deferred_sample_learner_worker_owns_model_state": bool(
                    self.deferred_sample_learner_worker_owns_model_state
                ),
                "compact_owned_loop_deferred_sample_learner_worker_owns_replay_store": bool(
                    self.deferred_sample_learner_worker_owns_replay_store
                ),
                "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent": bool(
                    self.deferred_sample_learner_full_replay_snapshot_sent
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "full_replay_snapshot_submit_count"
                ): int(self.deferred_sample_learner_full_replay_snapshot_submit_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_entry_count"
                ): int(self.deferred_sample_learner_replay_append_entry_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_index_row_count"
                ): int(self.deferred_sample_learner_replay_append_index_row_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "last_replay_append_entry_count"
                ): int(self.deferred_sample_learner_last_replay_append_entry_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "last_replay_append_index_row_count"
                ): int(self.deferred_sample_learner_last_replay_append_index_row_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_entry_bytes"
                ): int(self.deferred_sample_learner_replay_append_entry_bytes),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_host_observation_bytes"
                ): int(
                    self.deferred_sample_learner_replay_append_host_observation_bytes
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_resident_snapshot_count"
                ): int(
                    self.deferred_sample_learner_replay_append_resident_snapshot_count
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_resident_snapshot_bytes"
                ): int(
                    self.deferred_sample_learner_replay_append_resident_snapshot_bytes
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_compact_batch_bytes"
                ): int(self.deferred_sample_learner_replay_append_compact_batch_bytes),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_step_payload_bytes"
                ): int(self.deferred_sample_learner_replay_append_step_payload_bytes),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "replay_append_render_state_bytes"
                ): int(self.deferred_sample_learner_replay_append_render_state_bytes),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_step_count"
                ): int(self.deferred_sample_learner_provider_bootstrap_step_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "last_provider_bootstrap_step_count"
                ): int(self.deferred_sample_learner_last_provider_bootstrap_step_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_step_bytes"
                ): int(self.deferred_sample_learner_provider_bootstrap_step_bytes),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_host_observation_bytes"
                ): int(
                    self.deferred_sample_learner_provider_bootstrap_host_observation_bytes
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_resident_snapshot_count"
                ): int(
                    self.deferred_sample_learner_provider_bootstrap_resident_snapshot_count
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_resident_snapshot_bytes"
                ): int(
                    self.deferred_sample_learner_provider_bootstrap_resident_snapshot_bytes
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_render_state_bytes"
                ): int(self.deferred_sample_learner_provider_bootstrap_render_state_bytes),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_replay_entry_count"
                ): int(self.deferred_sample_learner_provider_bootstrap_replay_entry_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_replay_index_row_count"
                ): int(
                    self.deferred_sample_learner_provider_bootstrap_replay_index_row_count
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "provider_bootstrap_learner_call_count"
                ): int(self.deferred_sample_learner_provider_bootstrap_learner_call_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_observation_provider_present"
                ): bool(self.deferred_sample_learner_worker_observation_provider_present),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_observation_provider_bootstrap_step_count"
                ): int(
                    self.deferred_sample_learner_worker_observation_provider_bootstrap_step_count
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_last_observation_provider_bootstrap_step_count"
                ): int(
                    self.deferred_sample_learner_worker_last_observation_provider_bootstrap_step_count
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_observation_provider_missing_stack_history_count"
                ): int(
                    self.deferred_sample_learner_worker_observation_provider_missing_stack_history_count
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_observation_provider_materialized_entry_count"
                ): int(
                    self.deferred_sample_learner_worker_observation_provider_materialized_entry_count
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_last_observation_provider_materialized_entry_count"
                ): int(
                    self.deferred_sample_learner_worker_last_observation_provider_materialized_entry_count
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_model_initialized_count"
                ): int(self.deferred_sample_learner_worker_model_initialized_count),
                "compact_owned_loop_deferred_sample_learner_worker_completed_count": int(
                    self.deferred_sample_learner_worker_completed_count
                ),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_job_wall_sec"
                ): float(self.deferred_sample_learner_worker_job_wall_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_inner_job_wall_sec"
                ): float(self.deferred_sample_learner_worker_inner_job_wall_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_replay_prepare_sec"
                ): float(self.deferred_sample_learner_worker_replay_prepare_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_sample_sec"
                ): float(self.deferred_sample_learner_worker_sample_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_learner_sec"
                ): float(self.deferred_sample_learner_worker_learner_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_model_state_prepare_sec"
                ): float(self.deferred_sample_learner_worker_model_state_prepare_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_model_state_fn_sec"
                ): float(self.deferred_sample_learner_worker_model_state_fn_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_model_state_clone_sec"
                ): float(self.deferred_sample_learner_worker_model_state_clone_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_model_state_digest_sec"
                ): float(self.deferred_sample_learner_worker_model_state_digest_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_result_public_sec"
                ): float(self.deferred_sample_learner_worker_result_public_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_result_pickle_sec"
                ): float(self.deferred_sample_learner_worker_result_pickle_sec),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_replay_append_count"
                ): int(self.deferred_sample_learner_worker_replay_append_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_replay_entry_count"
                ): int(self.deferred_sample_learner_worker_replay_entry_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_replay_index_row_count"
                ): int(self.deferred_sample_learner_worker_replay_index_row_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_replay_evicted_entry_count"
                ): int(self.deferred_sample_learner_worker_replay_evicted_entry_count),
                (
                    "compact_owned_loop_deferred_sample_learner_"
                    "worker_replay_evicted_index_row_count"
                ): int(
                    self.deferred_sample_learner_worker_replay_evicted_index_row_count
                ),
                "compact_owned_loop_deferred_sample_learner_wait_count": int(
                    self.deferred_sample_learner_wait_count
                ),
                "compact_owned_loop_deferred_sample_learner_wait_sec": float(
                    self.deferred_sample_learner_wait_sec
                ),
                "compact_owned_loop_deferred_sample_learner_last_wait_sec": float(
                    self.deferred_sample_learner_last_wait_sec
                ),
                "compact_owned_loop_replay_store_entry_count": entry_count,
                "compact_owned_loop_replay_store_index_row_count": stored_rows,
                "compact_owned_loop_replay_store_capacity": int(self.config.replay_capacity),
                "compact_owned_loop_replay_store_evicted_entry_count": (evicted_entries),
                "compact_owned_loop_replay_store_evicted_index_row_count": (evicted_rows),
                "compact_owned_loop_replay_store_retained_resident_snapshot_count": (
                    retained_resident_snapshot_count
                ),
                "compact_owned_loop_replay_store_retained_resident_snapshot_bytes": (
                    retained_resident_snapshot_bytes
                ),
            }
        )
        return metadata

    def _sample_and_train(
        self,
        *,
        defer_learner: bool | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
        sample_result = self.replay_store.sample(
            seed=int(self.config.sample_seed_base) + self.sample_gate_calls,
            sample_batch_size=int(self.config.sample_batch_size),
            require_next_targets=bool(self.config.require_next_targets),
            num_unroll_steps=int(self.config.num_unroll_steps),
            build_compact_muzero_learner_batch=bool(self.config.fused_learner_batch),
            compact_muzero_learner_batch_only=bool(self.config.fused_learner_batch),
        )
        self._record_sample_result(sample_result)
        sample_batch = sample_result.get("sample_batch")
        learner_batch = sample_result.get("learner_batch")
        learner_result = None
        if self.learner is not None and (sample_batch is not None or learner_batch is not None):
            if self._should_defer_learner(defer_learner):
                learner_result = self._submit_deferred_learner(sample_batch, learner_batch)
            else:
                learner_result = self._train_learner_now(sample_batch, learner_batch)
        return sample_result, learner_result

    def _record_sample_result(self, sample_result: Mapping[str, Any]) -> None:
        self.sample_gate_calls += 1
        self.sample_gate_index_rows += int(sample_result["index_row_count"])
        self.sample_gate_target_rows += int(sample_result["target_row_count"])
        self.sample_gate_sample_rows += int(sample_result["sample_row_count"])
        self.sample_gate_sec += float(sample_result["sec"])
        self.sample_gate_last_telemetry = dict(sample_result["telemetry"])
        sample_batch = sample_result.get("sample_batch")
        learner_batch = sample_result.get("learner_batch")
        self.sample_gate_last_sample_metadata = dict(
            sample_result.get("sample_metadata")
            or getattr(sample_batch, "metadata", {})
            or getattr(learner_batch, "metadata", {})
            or {}
        )

    def _should_defer_sample_learner(self) -> bool:
        if self.learner is None:
            return False
        if self._sample_learner_worker is None:
            return False
        return bool(self.config.defer_sample_learner_gate)

    def _queue_sample_learner_replay_append_entry(
        self,
        *,
        previous_step: Any,
        current_step: Any,
        index_rows: Any,
    ) -> None:
        if not _sample_learner_worker_uses_replay_append_transport(
            self._sample_learner_worker
        ):
            return
        transport_kind = str(
            self.config.defer_sample_learner_replay_append_transport_kind
        )
        make_entry_name = (
            "make_scalar_append_delta_entry"
            if transport_kind == COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
            else "make_append_delta_entry"
        )
        make_entry = getattr(self.replay_store, make_entry_name, None)
        if not callable(make_entry):
            raise RuntimeError(
                "local_process sample+learner replay ownership requires "
                f"{make_entry_name}"
            )
        entry = make_entry(
            previous_step=previous_step,
            current_step=current_step,
            index_rows=index_rows,
        )
        if entry is None:
            return
        row_count = _leading_dim_for_replay_append_entry(entry)
        self._pending_sample_learner_replay_append_entries.append(entry)
        self._pending_sample_learner_replay_append_index_rows += row_count

    def _queue_sample_learner_provider_bootstrap_step(self, step: Any) -> None:
        if not _sample_learner_worker_uses_replay_append_transport(
            self._sample_learner_worker
        ):
            return
        if (
            str(self.config.defer_sample_learner_replay_append_transport_kind)
            != COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
        ):
            return
        bootstrap_step = _clone_provider_bootstrap_step(step)
        if bootstrap_step is None:
            return
        self._pending_sample_learner_provider_bootstrap_steps.append(bootstrap_step)
        max_steps = int(COMPACT_PROVIDER_BOOTSTRAP_STACK_DEPTH)
        if len(self._pending_sample_learner_provider_bootstrap_steps) > max_steps:
            del self._pending_sample_learner_provider_bootstrap_steps[:-max_steps]

    def _should_defer_learner(self, defer_learner: bool | None) -> bool:
        if self.learner is None:
            return False
        if self._learner_executor is None:
            return False
        if defer_learner is None:
            return bool(self.config.defer_learner_gate)
        return bool(defer_learner)

    def _submit_deferred_learner(
        self,
        sample_batch: Any | None,
        learner_batch: Any | None,
    ) -> Mapping[str, Any] | None:
        prior_result = self.consume_completed_learner_result(wait=False)
        executor = self._learner_executor
        if executor is None or self.learner is None:
            raise RuntimeError("deferred learner requested without learner executor")
        max_pending = int(self.config.defer_learner_gate_max_pending)
        if max_pending <= 0:
            raise ValueError("defer_learner_gate_max_pending must be positive")
        while len(self._pending_learner_futures) >= max_pending:
            blocked_result = self.consume_completed_learner_result(wait=True)
            if blocked_result is not None:
                prior_result = _combine_learner_results(prior_result, blocked_result)
        train_input = sample_batch
        if learner_batch is not None:
            train_method = getattr(self.learner, "train_on_learner_batch", None)
            if not callable(train_method):
                raise RuntimeError("fused learner batch requested without learner support")
            train_input = learner_batch
        else:
            train_method = self.learner.train_on_sample_batch
        self._pending_learner_futures.append(
            executor.submit(
                _run_deferred_learner_train,
                learner=self.learner,
                train_method=train_method,
                train_input=train_input,
                train_steps=int(self.config.learner_train_steps),
            )
        )
        self.deferred_learner_max_pending_observed = max(
            self.deferred_learner_max_pending_observed,
            len(self._pending_learner_futures),
        )
        self.deferred_learner_submit_count += 1
        self.deferred_learner_policy_lag_max = max(
            self.deferred_learner_policy_lag_max,
            self._deferred_learner_policy_lag_current(),
        )
        return prior_result

    def _submit_deferred_sample_learner(self) -> CompactOwnedLoopStepResultV1 | None:
        prior_result = self.consume_completed_sample_learner_result(wait=False)
        worker = self._sample_learner_worker
        if worker is None or self.learner is None:
            raise RuntimeError("deferred sample+learner requested without executor")
        max_pending = int(self.config.defer_sample_learner_gate_max_pending)
        if max_pending <= 0:
            raise ValueError("defer_sample_learner_gate_max_pending must be positive")
        while len(self._pending_sample_learner_futures) >= max_pending:
            prior_result = self.consume_completed_sample_learner_result(wait=True)
        append_transport = _sample_learner_worker_uses_replay_append_transport(worker)
        replay_snapshot = None
        replay_append_entries: tuple[Any, ...] = ()
        provider_bootstrap_steps: tuple[Any, ...] = ()
        replay_append_index_rows = 0
        replay_store_metadata: Mapping[str, Any] | None = None
        replay_capacity: int | None = None
        full_replay_snapshot_sent = True
        if append_transport:
            replay_append_entries = tuple(self._pending_sample_learner_replay_append_entries)
            provider_bootstrap_steps = tuple(
                self._pending_sample_learner_provider_bootstrap_steps
            )
            replay_append_index_rows = int(
                self._pending_sample_learner_replay_append_index_rows
            )
            if not replay_append_entries:
                raise RuntimeError(
                    "local_process sample+learner worker has no replay append entries "
                    "to apply; refusing to sample from unsynchronized worker replay"
                )
            replay_capacity = int(
                getattr(self.replay_store, "capacity", int(self.config.replay_capacity))
            )
            replay_store_metadata = _replay_store_metadata_for_append_transport(
                self.replay_store
            )
            replay_snapshot_version = int(getattr(self.replay_store, "snapshot_version", 0))
            full_replay_snapshot_sent = False
        else:
            snapshot_for_sample = getattr(self.replay_store, "snapshot_for_sample", None)
            if not callable(snapshot_for_sample):
                raise RuntimeError("deferred sample+learner requires replay snapshot_for_sample")
            sample_from_snapshot = getattr(self.replay_store, "sample_from_snapshot", None)
            if not callable(sample_from_snapshot):
                raise RuntimeError("deferred sample+learner requires replay sample_from_snapshot")
            replay_snapshot = snapshot_for_sample()
            replay_snapshot_version = int(getattr(replay_snapshot, "snapshot_version", 0))
        request_id = int(self.deferred_sample_learner_submit_count) + 1
        return_model_state = self._deferred_sample_learner_should_return_model_state(request_id)
        request = CompactSampleLearnerWorkRequestV1(
            request_id=request_id,
            replay_store=self.replay_store,
            replay_snapshot=replay_snapshot,
            learner=self.learner,
            seed=(
                int(self.config.sample_seed_base)
                + int(self.deferred_sample_learner_submit_count)
            ),
            sample_batch_size=int(self.config.sample_batch_size),
            require_next_targets=bool(self.config.require_next_targets),
            num_unroll_steps=int(self.config.num_unroll_steps),
            fused_learner_batch=bool(self.config.fused_learner_batch),
            train_steps=int(self.config.learner_train_steps),
            policy_version_ref=str(self.policy_version.policy_version_ref),
            model_version_ref=self.policy_version.model_version_ref,
            policy_source=str(self.policy_version.policy_source),
            return_model_state=return_model_state,
            model_state_transport_kind=str(
                self.config.defer_sample_learner_model_state_transport_kind
            ),
            replay_append_entries=replay_append_entries,
            provider_bootstrap_steps=provider_bootstrap_steps,
            replay_store_metadata=replay_store_metadata,
            replay_capacity=replay_capacity,
            replay_snapshot_version=replay_snapshot_version,
            full_replay_snapshot_sent=full_replay_snapshot_sent,
        )
        handle = worker.submit(request)
        if append_transport:
            self._pending_sample_learner_replay_append_entries.clear()
            self._pending_sample_learner_replay_append_index_rows = 0
            self._pending_sample_learner_provider_bootstrap_steps.clear()
            self.deferred_sample_learner_last_replay_append_entry_count = len(
                replay_append_entries
            )
            self.deferred_sample_learner_last_replay_append_index_row_count = int(
                replay_append_index_rows
            )
            self.deferred_sample_learner_replay_append_entry_count += len(
                replay_append_entries
            )
            self.deferred_sample_learner_replay_append_index_row_count += int(
                replay_append_index_rows
            )
            self.deferred_sample_learner_last_provider_bootstrap_step_count = len(
                provider_bootstrap_steps
            )
            self.deferred_sample_learner_provider_bootstrap_step_count += len(
                provider_bootstrap_steps
            )
        else:
            self.deferred_sample_learner_last_replay_append_entry_count = 0
            self.deferred_sample_learner_last_replay_append_index_row_count = 0
            self.deferred_sample_learner_last_provider_bootstrap_step_count = 0
        if full_replay_snapshot_sent:
            self.deferred_sample_learner_full_replay_snapshot_submit_count += 1
        self.deferred_sample_learner_full_replay_snapshot_sent = bool(
            full_replay_snapshot_sent
        )
        self._pending_sample_learner_futures.append(
            CompactPendingSampleLearnerWorkV1(
                request_id=request_id,
                handle=handle,
                submitted_at=time.perf_counter(),
                snapshot_version=int(replay_snapshot_version),
                return_model_state=bool(return_model_state),
            )
        )
        self.deferred_sample_learner_max_pending_observed = max(
            self.deferred_sample_learner_max_pending_observed,
            len(self._pending_sample_learner_futures),
        )
        self.deferred_sample_learner_submit_count += 1
        self.deferred_sample_learner_last_submitted_request_id = request_id
        self.deferred_sample_learner_last_submitted_snapshot_version = int(
            replay_snapshot_version
        )
        self.deferred_sample_learner_policy_lag_max = max(
            self.deferred_sample_learner_policy_lag_max,
            self._deferred_sample_learner_policy_lag_current(),
        )
        return prior_result

    def _deferred_sample_learner_should_return_model_state(self, request_id: int) -> bool:
        if bool(self._force_next_sample_learner_model_state):
            self._force_next_sample_learner_model_state = False
            return True
        interval = int(self.config.defer_sample_learner_model_state_interval)
        if interval <= 1:
            return True
        train_steps = max(1, int(self.config.learner_train_steps))
        request_index = max(1, int(request_id))
        earliest_possible_update = ((request_index - 1) * train_steps) + 1
        latest_possible_update = request_index * train_steps
        for update_count in range(earliest_possible_update, latest_possible_update + 1):
            if update_count % interval == 0:
                return True
        return False

    def _train_learner_now(
        self,
        sample_batch: Any | None,
        learner_batch: Any | None,
    ) -> Mapping[str, Any]:
        prior_result = self.consume_completed_learner_result(wait=True)
        if self.learner is None:
            raise RuntimeError("learner is required")
        if learner_batch is not None:
            train_method = getattr(self.learner, "train_on_learner_batch", None)
            if not callable(train_method):
                raise RuntimeError("fused learner batch requested without learner support")
            result = train_method(
                learner_batch,
                train_steps=int(self.config.learner_train_steps),
            )
            result = dict(result)
            result.update(
                _learner_batch_resident_handle_consumption_fields(learner_batch)
            )
        else:
            result = self.learner.train_on_sample_batch(
                sample_batch,
                train_steps=int(self.config.learner_train_steps),
            )
            result = dict(result)
        self._record_learner_result(result)
        return _combine_learner_results(prior_result, result)

    def _record_learner_result(self, learner_result: Mapping[str, Any]) -> None:
        self.learner_gate_calls += 1
        self.learner_gate_updates += int(
            learner_result["compact_rollout_slab_learner_gate_updates"]
        )
        self.learner_gate_sample_rows += int(
            learner_result["compact_rollout_slab_learner_gate_sample_rows"]
        )
        self.learner_gate_input_bytes += int(
            learner_result["compact_rollout_slab_learner_gate_input_bytes"]
        )
        self.learner_gate_sec += float(learner_result["compact_rollout_slab_learner_gate_sec"])
        self.learner_gate_last_telemetry = _public_learner_result(learner_result)
        if bool(
            learner_result.get(
                f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_requested",
                False,
            )
        ):
            consumed = bool(
                learner_result.get(
                    f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_consumed",
                    False,
                )
            )
            fallback_count = int(
                learner_result.get(
                    f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_fallback_count",
                    0,
                )
                or 0
            )
            self.learner_gate_resident_batch_handle_requested_count += 1
            self.learner_gate_resident_batch_handle_consumed_count += int(consumed)
            self.learner_gate_resident_batch_handle_fallback_count += fallback_count
            self.learner_gate_resident_batch_handle_last_consumed = consumed
            self.learner_gate_resident_batch_handle_last_schema_id = str(
                learner_result.get(
                    f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_schema_id",
                    "none",
                )
            )
            self.learner_gate_resident_batch_handle_last_handle_id = int(
                learner_result.get(
                    f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_handle_id",
                    0,
                )
                or 0
            )
            self.learner_gate_resident_batch_handle_last_snapshot_version = int(
                learner_result.get(
                    f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_snapshot_version",
                    0,
                )
                or 0
            )
            self.learner_gate_resident_batch_handle_last_request_checksum = int(
                learner_result.get(
                    f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_request_checksum",
                    0,
                )
                or 0
            )
            self.learner_gate_resident_batch_handle_last_sample_row_count = int(
                learner_result.get(
                    f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_sample_row_count",
                    0,
                )
                or 0
            )
            self.learner_gate_resident_batch_handle_last_target_row_count = int(
                learner_result.get(
                    f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_target_row_count",
                    0,
                )
                or 0
            )
            self.learner_gate_resident_batch_handle_last_fallback_count = fallback_count
            self.learner_gate_resident_batch_handle_last_fallback_reason = str(
                learner_result.get(
                    f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_fallback_reason",
                    "none",
                )
            )

    def _apply_deferred_learner_model_state(
        self,
        learner_result: Mapping[str, Any] | None,
    ) -> None:
        self.deferred_sample_learner_last_model_state_applied = False
        if learner_result is None:
            return
        state_dict = learner_result.get(_DEFERRED_LEARNER_MODEL_STATE_DICT_KEY)
        if state_dict is None:
            snapshot = learner_result.get(_DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY)
            if isinstance(snapshot, Mapping):
                state_dict, load_metadata = _load_model_state_snapshot_file(snapshot)
                self.deferred_sample_learner_model_state_snapshot_load_count += 1
                self.deferred_sample_learner_model_state_snapshot_load_bytes += int(
                    load_metadata.get("bytes", 0) or 0
                )
                self.deferred_sample_learner_model_state_snapshot_load_sec += float(
                    load_metadata.get("load_sec", 0.0) or 0.0
                )
        if state_dict is None or self.learner is None:
            return
        load = getattr(self.learner, "load_model_state_dict", None)
        if callable(load):
            load(state_dict)
        else:
            model = getattr(self.learner, "model", None)
            load_state_dict = getattr(model, "load_state_dict", None)
            if not callable(load_state_dict):
                load_state_dict = getattr(self.learner, "load_state_dict", None)
            if not callable(load_state_dict):
                raise RuntimeError(
                    "deferred sample+learner returned model state but learner cannot load it"
                )
            load_state_dict(dict(state_dict))
        self.deferred_sample_learner_model_state_apply_count += 1
        self.deferred_sample_learner_last_model_state_applied = True

    def _deferred_learner_policy_lag_current(self) -> int:
        return max(
            0,
            int(self.deferred_learner_submit_count) - int(self.deferred_learner_completed_count),
        )

    def _deferred_sample_learner_policy_lag_current(self) -> int:
        return max(
            0,
            int(self.deferred_sample_learner_submit_count)
            - int(self.deferred_sample_learner_completed_count),
        )


class _ThreadSampleLearnerWorkerV1:
    """Default adapter for the existing in-process sample+learner lane."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="compact-owned-sample-learner",
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "compact_owned_loop_sample_learner_worker_kind": "in_process_thread",
            "compact_owned_loop_sample_learner_worker_resource_id": (
                "local_process:compact-owned-sample-learner-thread"
            ),
            "compact_owned_loop_actor_search_resource_id": "local_process:actor-search",
            "compact_owned_loop_actor_search_pid": os.getpid(),
            "compact_owned_loop_sample_learner_worker_parent_pid": os.getpid(),
            "compact_owned_loop_sample_learner_worker_resource_scope": "thread",
            "compact_owned_loop_sample_learner_worker_start_method": "thread",
            "compact_owned_loop_sample_learner_worker_bootstrap_source": "in_process",
            "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": False,
            (
                "compact_owned_loop_sample_learner_hardware_resource_distinct_"
                "from_actor_search"
            ): False,
        }

    def submit(self, request: CompactSampleLearnerWorkRequestV1) -> Future[Mapping[str, Any]]:
        return self._executor.submit(
            _run_deferred_sample_learner_train,
            replay_store=request.replay_store,
            replay_snapshot=request.replay_snapshot,
            learner=request.learner,
            seed=int(request.seed),
            sample_batch_size=int(request.sample_batch_size),
            require_next_targets=bool(request.require_next_targets),
            num_unroll_steps=int(request.num_unroll_steps),
            fused_learner_batch=bool(request.fused_learner_batch),
            train_steps=int(request.train_steps),
            return_model_state=bool(request.return_model_state),
            model_state_transport_kind=str(request.model_state_transport_kind),
            model_state_snapshot_request_id=int(request.request_id),
        )

    def done(self, handle: Future[Mapping[str, Any]]) -> bool:
        return bool(handle.done())

    def result(self, handle: Future[Mapping[str, Any]]) -> Mapping[str, Any]:
        return handle.result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)


class CompactProcessSampleLearnerWorkerV1:
    """Default-off adapter for sample+learner work in a separate local process."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        learner_factory: Any | None = None,
        learner_factory_kwargs: Mapping[str, Any] | None = None,
        observation_provider_factory: Any | None = None,
        observation_provider_factory_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._parent_pid = os.getpid()
        self._max_workers = int(max_workers)
        self._mp_context = mp.get_context("spawn")
        self._executor: ProcessPoolExecutor | None = None
        self._prepared = False
        self._learner_factory = learner_factory
        self._learner_factory_kwargs = dict(learner_factory_kwargs or {})
        self._observation_provider_factory = observation_provider_factory
        self._observation_provider_factory_kwargs = dict(
            observation_provider_factory_kwargs or {}
        )
        self._bootstrap_source = "factory" if callable(learner_factory) else "parent_learner"

    def prepare(self, *, replay_store: Any, learner: Any) -> None:
        if self._prepared:
            return
        if int(self._max_workers) != 1:
            raise RuntimeError(
                "local_process sample+learner replay ownership requires max_workers=1"
            )
        if _contains_cuda_tensor(getattr(replay_store, "snapshot_for_sample", None)):
            raise RuntimeError("local_process replay bootstrap unexpectedly contains CUDA tensors")
        learner_for_initializer = learner
        learner_factory = self._learner_factory
        learner_factory_kwargs = dict(self._learner_factory_kwargs)
        if callable(learner_factory):
            learner_for_initializer = None
            if _learner_factory_kwargs_contain_cuda_tensors(learner_factory_kwargs):
                raise RuntimeError(
                    "local_process sample+learner factory kwargs must be host-only"
                )
        elif _learner_model_state_contains_cuda_tensors(learner):
            raise RuntimeError(
                "local_process sample+learner cannot bootstrap from CUDA learner "
                "state; use a host-only learner factory or a persistent learner service"
            )
        observation_provider_factory = self._observation_provider_factory
        observation_provider_factory_kwargs = dict(
            self._observation_provider_factory_kwargs
        )
        if callable(observation_provider_factory) and _contains_cuda_tensor(
            observation_provider_factory_kwargs
        ):
            raise RuntimeError(
                "local_process sample+learner observation provider kwargs must be host-only"
            )
        replay_store_class = replay_store.__class__
        replay_capacity = int(getattr(replay_store, "capacity"))
        replay_store_metadata = _replay_store_metadata_for_append_transport(replay_store)
        self._executor = ProcessPoolExecutor(
            max_workers=int(self._max_workers),
            mp_context=self._mp_context,
            initializer=_compact_process_worker_initializer,
            initargs=(
                replay_store_class,
                replay_capacity,
                replay_store_metadata,
                learner_for_initializer,
                learner_factory,
                learner_factory_kwargs,
                observation_provider_factory,
                observation_provider_factory_kwargs,
            ),
        )
        self._prepared = True

    @property
    def metadata(self) -> dict[str, Any]:
        parent_pid = int(self._parent_pid)
        return {
            "compact_owned_loop_sample_learner_worker_kind": "local_process",
            "compact_owned_loop_sample_learner_worker_resource_id": (
                f"local_process_pool:{parent_pid}:compact-owned-sample-learner"
            ),
            "compact_owned_loop_actor_search_resource_id": (
                f"local_process:{parent_pid}:actor-search"
            ),
            "compact_owned_loop_actor_search_pid": parent_pid,
            "compact_owned_loop_sample_learner_worker_parent_pid": parent_pid,
            "compact_owned_loop_sample_learner_worker_resource_scope": "process",
            "compact_owned_loop_sample_learner_worker_start_method": str(
                self._mp_context.get_start_method()
            ),
            "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings": (
                "expandable_segments:False"
            ),
            "compact_owned_loop_sample_learner_worker_bootstrap_source": str(
                self._bootstrap_source
            ),
            "compact_owned_loop_sample_learner_observation_provider_configured": bool(
                callable(self._observation_provider_factory)
            ),
            "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": True,
            (
                "compact_owned_loop_sample_learner_hardware_resource_distinct_"
                "from_actor_search"
            ): False,
        }

    def submit(self, request: CompactSampleLearnerWorkRequestV1) -> Future[Mapping[str, Any]]:
        replay_snapshot = request.replay_snapshot
        replay_append_entries = tuple(request.replay_append_entries)
        provider_bootstrap_steps = tuple(request.provider_bootstrap_steps)
        snapshot_host_clone_used = False
        if _sample_learner_process_payload_contains_cuda_tensors(request):
            replay_snapshot = _host_only_clone(replay_snapshot)
            replay_append_entries = tuple(_host_only_clone(replay_append_entries))
            provider_bootstrap_steps = tuple(_host_only_clone(provider_bootstrap_steps))
            snapshot_host_clone_used = True
        if _contains_cuda_tensor(replay_snapshot):
            raise RuntimeError(
                "local_process sample+learner cannot receive CUDA tensors through "
                "Python multiprocessing; use a host-only transport or a persistent "
                "learner service"
            )
        if _contains_cuda_tensor(replay_append_entries):
            raise RuntimeError(
                "local_process sample+learner cannot receive CUDA replay append "
                "entries through Python multiprocessing; use a host-only transport "
                "or a persistent learner service"
            )
        if _contains_cuda_tensor(provider_bootstrap_steps):
            raise RuntimeError(
                "local_process sample+learner cannot receive CUDA provider bootstrap "
                "steps through Python multiprocessing; use a host-only transport "
                "or a persistent learner service"
            )
        if self._executor is None:
            self.prepare(replay_store=request.replay_store, learner=request.learner)
        _validate_provider_bootstrap_steps(provider_bootstrap_steps)
        append_payload = _replay_append_transport_payload_summary(replay_append_entries)
        bootstrap_payload = _provider_bootstrap_payload_summary(provider_bootstrap_steps)
        transport_request = CompactProcessSampleLearnerTransportRequestV1(
            request_id=int(request.request_id),
            replay_snapshot=replay_snapshot,
            seed=int(request.seed),
            sample_batch_size=int(request.sample_batch_size),
            require_next_targets=bool(request.require_next_targets),
            num_unroll_steps=int(request.num_unroll_steps),
            fused_learner_batch=bool(request.fused_learner_batch),
            train_steps=int(request.train_steps),
            policy_version_ref=str(request.policy_version_ref),
            model_version_ref=request.model_version_ref,
            policy_source=str(request.policy_source),
            return_model_state=bool(request.return_model_state),
            model_state_transport_kind=str(request.model_state_transport_kind),
            replay_append_entries=replay_append_entries,
            provider_bootstrap_steps=provider_bootstrap_steps,
            replay_store_metadata=request.replay_store_metadata,
            replay_capacity=request.replay_capacity,
            replay_snapshot_version=int(request.replay_snapshot_version),
            full_replay_snapshot_sent=bool(request.full_replay_snapshot_sent),
            replay_append_entry_count=len(replay_append_entries),
            replay_append_index_row_count=_replay_append_index_row_count(
                replay_append_entries
            ),
            replay_append_entry_bytes=int(append_payload["entry_bytes"]),
            replay_append_host_observation_bytes=int(
                append_payload["host_observation_bytes"]
            ),
            replay_append_resident_snapshot_count=int(
                append_payload["resident_snapshot_count"]
            ),
            replay_append_resident_snapshot_bytes=int(
                append_payload["resident_snapshot_bytes"]
            ),
            replay_append_compact_batch_bytes=int(append_payload["compact_batch_bytes"]),
            replay_append_step_payload_bytes=int(append_payload["step_payload_bytes"]),
            replay_append_render_state_bytes=int(append_payload["render_state_bytes"]),
            provider_bootstrap_step_count=len(provider_bootstrap_steps),
            provider_bootstrap_step_bytes=int(bootstrap_payload["step_bytes"]),
            provider_bootstrap_host_observation_bytes=int(
                bootstrap_payload["host_observation_bytes"]
            ),
            provider_bootstrap_resident_snapshot_count=int(
                bootstrap_payload["resident_snapshot_count"]
            ),
            provider_bootstrap_resident_snapshot_bytes=int(
                bootstrap_payload["resident_snapshot_bytes"]
            ),
            provider_bootstrap_render_state_bytes=int(
                bootstrap_payload["render_state_bytes"]
            ),
            snapshot_host_clone_used=snapshot_host_clone_used,
        )
        request_bytes = _pickle_size_bytes(transport_request)
        transport_request = CompactProcessSampleLearnerTransportRequestV1(
            request_id=transport_request.request_id,
            replay_snapshot=transport_request.replay_snapshot,
            seed=transport_request.seed,
            sample_batch_size=transport_request.sample_batch_size,
            require_next_targets=transport_request.require_next_targets,
            num_unroll_steps=transport_request.num_unroll_steps,
            fused_learner_batch=transport_request.fused_learner_batch,
            train_steps=transport_request.train_steps,
            policy_version_ref=transport_request.policy_version_ref,
            model_version_ref=transport_request.model_version_ref,
            policy_source=transport_request.policy_source,
            return_model_state=bool(transport_request.return_model_state),
            model_state_transport_kind=str(transport_request.model_state_transport_kind),
            replay_append_entries=transport_request.replay_append_entries,
            provider_bootstrap_steps=transport_request.provider_bootstrap_steps,
            replay_store_metadata=transport_request.replay_store_metadata,
            replay_capacity=transport_request.replay_capacity,
            replay_snapshot_version=int(transport_request.replay_snapshot_version),
            full_replay_snapshot_sent=bool(transport_request.full_replay_snapshot_sent),
            replay_append_entry_count=int(transport_request.replay_append_entry_count),
            replay_append_index_row_count=int(
                transport_request.replay_append_index_row_count
            ),
            replay_append_entry_bytes=int(transport_request.replay_append_entry_bytes),
            replay_append_host_observation_bytes=int(
                transport_request.replay_append_host_observation_bytes
            ),
            replay_append_resident_snapshot_count=int(
                transport_request.replay_append_resident_snapshot_count
            ),
            replay_append_resident_snapshot_bytes=int(
                transport_request.replay_append_resident_snapshot_bytes
            ),
            replay_append_compact_batch_bytes=int(
                transport_request.replay_append_compact_batch_bytes
            ),
            replay_append_step_payload_bytes=int(
                transport_request.replay_append_step_payload_bytes
            ),
            replay_append_render_state_bytes=int(
                transport_request.replay_append_render_state_bytes
            ),
            provider_bootstrap_step_count=int(
                transport_request.provider_bootstrap_step_count
            ),
            provider_bootstrap_step_bytes=int(
                transport_request.provider_bootstrap_step_bytes
            ),
            provider_bootstrap_host_observation_bytes=int(
                transport_request.provider_bootstrap_host_observation_bytes
            ),
            provider_bootstrap_resident_snapshot_count=int(
                transport_request.provider_bootstrap_resident_snapshot_count
            ),
            provider_bootstrap_resident_snapshot_bytes=int(
                transport_request.provider_bootstrap_resident_snapshot_bytes
            ),
            provider_bootstrap_render_state_bytes=int(
                transport_request.provider_bootstrap_render_state_bytes
            ),
            request_bytes=request_bytes,
            snapshot_host_clone_used=bool(transport_request.snapshot_host_clone_used),
        )
        assert self._executor is not None
        return self._executor.submit(
            _run_deferred_sample_learner_train_from_worker_state,
            transport_request,
        )

    def done(self, handle: Future[Mapping[str, Any]]) -> bool:
        return bool(handle.done())

    def result(self, handle: Future[Mapping[str, Any]]) -> Mapping[str, Any]:
        return handle.result()

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)


def build_compact_sample_learner_worker_v1(kind: str) -> Any:
    kind_str = str(kind).strip()
    if kind_str == COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD:
        return _ThreadSampleLearnerWorkerV1()
    if kind_str == COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS:
        return CompactProcessSampleLearnerWorkerV1()
    allowed = ", ".join(COMPACT_SAMPLE_LEARNER_WORKER_KINDS)
    raise ValueError(f"compact sample+learner worker kind must be one of {allowed}")


def compact_owned_loop_replay_store_metadata(
    policy_version: CompactPolicyVersionRefV1,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_policy_version(policy_version)
    metadata = _policy_metadata(policy_version)
    metadata.update(
        {
            "compact_owned_loop_entrypoint": True,
            "compact_owned_loop_schema_id": COMPACT_OWNED_LOOP_SCHEMA_ID,
            "compact_owned_loop_replay_store_owned": True,
            "compact_owned_loop_policy_version_handoff": True,
            "profile_only": True,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "stock_lightzero_integrated": False,
            "lightzero_training_integration_claim": False,
            "coach_integration_claim": False,
            "training_speed_claim": False,
        }
    )
    metadata.update({str(key): value for key, value in dict(extra or {}).items()})
    return metadata


def _compact_owned_loop_metadata(
    *,
    config: CompactOwnedLoopConfigV1,
    policy_version: CompactPolicyVersionRefV1,
) -> dict[str, Any]:
    metadata = compact_owned_loop_replay_store_metadata(policy_version)
    coach_report = build_profile_only_compact_coach_report_v1(
        speed_currency="local_compact_owned_loop_profile_only",
    )
    metadata.update(
        {
            "schema_id": COMPACT_OWNED_LOOP_SCHEMA_ID,
            "compact_owned_loop_schema_id": COMPACT_OWNED_LOOP_SCHEMA_ID,
            "compact_owned_loop_entrypoint": True,
            "compact_owned_loop_profile_only": True,
            "compact_owned_loop_calls_train_muzero": False,
            "compact_owned_loop_touches_live_runs": False,
            "compact_owned_loop_replay_store_owned": True,
            "compact_owned_loop_policy_version_handoff": True,
            "compact_owned_loop_sample_batch_size": int(config.sample_batch_size),
            "compact_owned_loop_sample_interval": int(config.sample_interval),
            "compact_owned_loop_replay_capacity": int(config.replay_capacity),
            "compact_owned_loop_learner_train_steps": int(config.learner_train_steps),
            "compact_owned_loop_num_unroll_steps": int(config.num_unroll_steps),
            "compact_owned_loop_learner_impl": str(config.learner_impl),
            "compact_owned_loop_require_next_targets": bool(config.require_next_targets),
            "compact_owned_loop_capture_replay_store_state": bool(
                config.capture_replay_store_state
            ),
            "compact_owned_loop_defer_learner_gate": bool(config.defer_learner_gate),
            "compact_owned_loop_defer_learner_gate_max_pending": int(
                config.defer_learner_gate_max_pending
            ),
            "compact_owned_loop_defer_sample_learner_gate": bool(
                config.defer_sample_learner_gate
            ),
            "compact_owned_loop_defer_sample_learner_gate_max_pending": int(
                config.defer_sample_learner_gate_max_pending
            ),
            "compact_owned_loop_defer_sample_learner_model_state_interval": int(
                config.defer_sample_learner_model_state_interval
            ),
            (
                "compact_owned_loop_defer_sample_learner_"
                "model_state_transport_kind"
            ): str(config.defer_sample_learner_model_state_transport_kind),
            (
                "compact_owned_loop_defer_sample_learner_"
                "replay_append_transport_kind"
            ): str(config.defer_sample_learner_replay_append_transport_kind),
            "compact_owned_loop_fused_learner_batch": bool(config.fused_learner_batch),
            "promotion_eligible": False,
            "promotion_blocker": "profile_only_compact_owned_loop_entrypoint",
            "speed_currency": "local_compact_owned_loop_profile_only",
        }
    )
    metadata.update(coach_report.as_metadata())
    return metadata


def _policy_metadata(policy_version: CompactPolicyVersionRefV1) -> dict[str, Any]:
    return {
        "policy_version_ref": str(policy_version.policy_version_ref),
        "compact_owned_loop_policy_version_ref": str(policy_version.policy_version_ref),
        "policy_source": str(policy_version.policy_source),
        "compact_owned_loop_policy_source": str(policy_version.policy_source),
        "model_version_ref": (
            None
            if policy_version.model_version_ref is None
            else str(policy_version.model_version_ref)
        ),
        "compact_owned_loop_model_version_ref": (
            None
            if policy_version.model_version_ref is None
            else str(policy_version.model_version_ref)
        ),
    }


def _sample_learner_worker_metadata(worker: Any | None) -> dict[str, Any]:
    default = {
        "compact_owned_loop_sample_learner_worker_kind": "none",
        "compact_owned_loop_sample_learner_worker_resource_id": "none",
        "compact_owned_loop_actor_search_resource_id": "local_process:actor-search",
        "compact_owned_loop_actor_search_pid": os.getpid(),
        "compact_owned_loop_sample_learner_worker_parent_pid": os.getpid(),
        "compact_owned_loop_sample_learner_worker_resource_scope": "none",
        "compact_owned_loop_sample_learner_worker_start_method": "none",
        "compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings": "none",
        "compact_owned_loop_sample_learner_worker_bootstrap_source": "none",
        "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": False,
        "compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search": False,
    }
    if worker is None:
        return default
    raw = getattr(worker, "metadata", None)
    metadata = raw() if callable(raw) else raw
    if not isinstance(metadata, Mapping):
        metadata = {}
    merged = dict(default)
    merged.update({str(key): value for key, value in dict(metadata).items()})
    return merged


def _sample_learner_worker_uses_replay_append_transport(worker: Any | None) -> bool:
    if worker is None:
        return False
    metadata = _sample_learner_worker_metadata(worker)
    return (
        str(metadata.get("compact_owned_loop_sample_learner_worker_kind"))
        == COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS
    )


def _learner_batch_resident_handle_consumption_fields(
    learner_batch: Any | None,
) -> dict[str, Any]:
    metadata = dict(getattr(learner_batch, "metadata", {}) or {})
    requested = bool(metadata.get(f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_requested", False))
    used = bool(metadata.get(f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_used", False))
    fallback_count = int(
        metadata.get(f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_fallback_count", 0)
        or 0
    )
    fallback_reason = str(
        metadata.get(
            f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_fallback_reason",
            "none",
        )
        or "none"
    )
    consumed = bool(requested and used and fallback_count == 0)
    schema_id = str(
        metadata.get(f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_schema_id", "none")
        or "none"
    )
    handle_id = int(
        metadata.get(f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_handle_id", 0)
        or 0
    )
    snapshot_version = int(
        metadata.get(f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_snapshot_version", 0)
        or 0
    )
    request_checksum = int(
        metadata.get(f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_request_checksum", 0)
        or 0
    )
    sample_row_count = int(
        metadata.get(f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_sample_row_count", 0)
        or 0
    )
    target_row_count = int(
        metadata.get(f"{_FIXED_SOA_LEARNER_BATCH_HANDLE_PREFIX}_target_row_count", 0)
        or 0
    )
    return {
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_requested": requested,
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_consumed": consumed,
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_source": (
            "fixed_soa_learner_batch_handle_ring" if requested else "none"
        ),
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_schema_id": schema_id,
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_handle_id": handle_id,
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_snapshot_version": (
            snapshot_version
        ),
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_request_checksum": (
            request_checksum
        ),
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_sample_row_count": (
            sample_row_count
        ),
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_target_row_count": (
            target_row_count
        ),
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_fallback_count": fallback_count,
        f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_fallback_reason": fallback_reason,
        (
            f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_"
            "materialized_parent_fallback_count"
        ): fallback_count,
        (
            f"{_OWNED_LOOP_RESIDENT_BATCH_HANDLE_PREFIX}_"
            "materialized_parent_fallback_reason"
        ): fallback_reason if fallback_count else "none",
    }


def _replay_store_metadata_for_append_transport(replay_store: Any) -> dict[str, Any]:
    metadata = getattr(replay_store, "store_metadata", None)
    if isinstance(metadata, Mapping):
        return {str(key): value for key, value in dict(metadata).items()}
    snapshot_for_sample = getattr(replay_store, "snapshot_for_sample", None)
    if callable(snapshot_for_sample):
        snapshot = snapshot_for_sample()
        return {
            str(key): value
            for key, value in dict(getattr(snapshot, "store_metadata", {}) or {}).items()
        }
    return {}


def _leading_dim_for_replay_append_entry(entry: Any) -> int:
    return _leading_dim_for_replay_rows(getattr(entry, "index_rows", None))


def _leading_dim_for_replay_rows(index_rows: Any) -> int:
    action = getattr(index_rows, "action", None)
    try:
        return int(len(action))
    except Exception:
        shape = getattr(action, "shape", ())
        if shape:
            return int(shape[0])
    return 0


def _replay_append_index_row_count(entries: tuple[Any, ...]) -> int:
    return sum(_leading_dim_for_replay_append_entry(entry) for entry in entries)


def _replay_append_transport_payload_summary(entries: tuple[Any, ...]) -> dict[str, int]:
    resident_snapshot_count = 0
    resident_snapshot_bytes = 0
    host_observation_bytes = 0
    compact_batch_bytes = 0
    step_payload_bytes = 0
    render_state_bytes = 0
    for entry in entries:
        for step_name in ("previous_step", "current_step"):
            step = getattr(entry, step_name, None)
            if step is None:
                continue
            host_observation_bytes += _array_payload_nbytes(
                getattr(step, "observation", None)
            )
            resident_snapshot = getattr(step, "resident_observation_replay_snapshot", None)
            if resident_snapshot is not None:
                resident_snapshot_count += 1
                resident_snapshot_bytes += _array_payload_nbytes(resident_snapshot)
            compact_batch_bytes += _array_payload_nbytes(
                getattr(step, "compact_batch", None)
            )
            step_payload_bytes += _array_payload_nbytes(getattr(step, "payload", None))
            render_state_bytes += _array_payload_nbytes(
                getattr(step, "render_state_snapshot", None)
            )
            render_state_bytes += _array_payload_nbytes(
                getattr(step, "autoreset_render_state_snapshot", None)
            )
    entry_bytes = _pickle_size_bytes(entries)
    return {
        "entry_bytes": max(0, int(entry_bytes)),
        "host_observation_bytes": int(host_observation_bytes),
        "resident_snapshot_count": int(resident_snapshot_count),
        "resident_snapshot_bytes": int(resident_snapshot_bytes),
        "compact_batch_bytes": int(compact_batch_bytes),
        "step_payload_bytes": int(step_payload_bytes),
        "render_state_bytes": int(render_state_bytes),
    }


def _provider_bootstrap_payload_summary(steps: tuple[Any, ...]) -> dict[str, int]:
    resident_snapshot_count = 0
    resident_snapshot_bytes = 0
    host_observation_bytes = 0
    render_state_bytes = 0
    for step in steps:
        _validate_provider_bootstrap_step(step)
        host_observation_bytes += _array_payload_nbytes(getattr(step, "observation", None))
        resident_snapshot = getattr(step, "resident_observation_replay_snapshot", None)
        if resident_snapshot is not None:
            resident_snapshot_count += 1
            resident_snapshot_bytes += _array_payload_nbytes(resident_snapshot)
        render_state_bytes += _array_payload_nbytes(
            getattr(step, "render_state_snapshot", None)
        )
        render_state_bytes += _array_payload_nbytes(
            getattr(step, "autoreset_render_state_snapshot", None)
        )
    return {
        "step_bytes": max(0, int(_pickle_size_bytes(steps))),
        "host_observation_bytes": int(host_observation_bytes),
        "resident_snapshot_count": int(resident_snapshot_count),
        "resident_snapshot_bytes": int(resident_snapshot_bytes),
        "render_state_bytes": int(render_state_bytes),
    }


def _clone_provider_bootstrap_step(step: Any) -> Any | None:
    if step is None:
        return None
    raw_payload = getattr(step, "payload", None) or {}
    payload: dict[str, Any] = {}
    if isinstance(raw_payload, Mapping) and "joint_action" in raw_payload:
        payload["joint_action"] = _host_only_clone(raw_payload["joint_action"])
    compact_batch = getattr(step, "compact_batch", None)
    compact_batch_snapshot = None
    if compact_batch is not None:
        compact_batch_snapshot = SimpleNamespace(
            final_observation=None,
            observation_source=getattr(compact_batch, "observation_source", None),
        )
    return SimpleNamespace(
        observation=None,
        flat_obs=None,
        target_reward=None,
        timestep=None,
        action_mask=_host_only_clone(getattr(step, "action_mask", None)),
        reward=_host_only_clone(getattr(step, "reward", None)),
        final_reward_map=_host_only_clone(getattr(step, "final_reward_map", None)),
        done=_host_only_clone(getattr(step, "done", None)),
        payload=payload,
        compact_batch=compact_batch_snapshot,
        resident_observation_replay_snapshot=None,
        render_state_snapshot=_host_only_clone(getattr(step, "render_state_snapshot", None)),
        autoreset_render_state_snapshot=_host_only_clone(
            getattr(step, "autoreset_render_state_snapshot", None)
        ),
    )


def _validate_provider_bootstrap_steps(steps: tuple[Any, ...]) -> None:
    for step in steps:
        _validate_provider_bootstrap_step(step)


def _validate_provider_bootstrap_step(step: Any) -> None:
    if getattr(step, "observation", None) is not None:
        raise RuntimeError("provider bootstrap step must not carry observations")
    if getattr(step, "flat_obs", None) is not None:
        raise RuntimeError("provider bootstrap step must not carry flat observations")
    if getattr(step, "target_reward", None) is not None:
        raise RuntimeError("provider bootstrap step must not carry target rewards")
    if getattr(step, "timestep", None) is not None:
        raise RuntimeError("provider bootstrap step must not carry timestep objects")
    if getattr(step, "resident_observation_replay_snapshot", None) is not None:
        raise RuntimeError("provider bootstrap step must not carry resident snapshots")
    compact_batch = getattr(step, "compact_batch", None)
    if compact_batch is not None:
        for field_name in (
            "observation",
            "final_observation",
            "device_observation",
            "root_device_observation",
            "final_device_observation",
            "root_final_device_observation",
        ):
            if getattr(compact_batch, field_name, None) is not None:
                raise RuntimeError(
                    "provider bootstrap compact batch must not carry observations"
                )
    payload = getattr(step, "payload", None)
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise RuntimeError("provider bootstrap payload must be a mapping")
        unexpected = set(str(key) for key in payload) - {"joint_action"}
        if unexpected:
            raise RuntimeError("provider bootstrap payload has unexpected fields")


def _array_payload_nbytes(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> int:
    if value is None or _depth > 12:
        return 0
    seen = set() if _seen is None else _seen
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return max(0, int(nbytes))
    element_size = getattr(value, "element_size", None)
    numel = getattr(value, "numel", None)
    if callable(element_size) and callable(numel):
        try:
            return max(0, int(element_size()) * int(numel()))
        except Exception:
            return 0
    if isinstance(value, Mapping):
        return sum(
            _array_payload_nbytes(item, _seen=seen, _depth=_depth + 1)
            for item in value.values()
        )
    if isinstance(value, tuple | list | set | frozenset):
        return sum(
            _array_payload_nbytes(item, _seen=seen, _depth=_depth + 1)
            for item in value
        )
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _array_payload_nbytes(
                getattr(value, field.name),
                _seen=seen,
                _depth=_depth + 1,
            )
            for field in fields(value)
        )
    if isinstance(value, SimpleNamespace):
        return sum(
            _array_payload_nbytes(item, _seen=seen, _depth=_depth + 1)
            for item in vars(value).values()
        )
    return 0


def _validate_config(config: CompactOwnedLoopConfigV1) -> None:
    if not bool(config.profile_only):
        raise ValueError("compact-owned loop is profile-only in this entrypoint")
    if bool(config.calls_train_muzero):
        raise ValueError("compact-owned loop does not call train_muzero")
    if bool(config.touches_live_runs):
        raise ValueError("compact-owned loop must not touch live runs")
    if int(config.sample_batch_size) < 0:
        raise ValueError("sample_batch_size must be non-negative")
    if int(config.sample_interval) <= 0:
        raise ValueError("sample_interval must be positive")
    if int(config.replay_capacity) <= 0:
        raise ValueError("replay_capacity must be positive")
    if int(config.learner_train_steps) <= 0:
        raise ValueError("learner_train_steps must be positive")
    if int(config.num_unroll_steps) <= 0:
        raise ValueError("num_unroll_steps must be positive")
    if int(config.defer_learner_gate_max_pending) <= 0:
        raise ValueError("defer_learner_gate_max_pending must be positive")
    if int(config.defer_sample_learner_gate_max_pending) <= 0:
        raise ValueError("defer_sample_learner_gate_max_pending must be positive")
    if int(config.defer_sample_learner_model_state_interval) <= 0:
        raise ValueError("defer_sample_learner_model_state_interval must be positive")
    if str(config.defer_sample_learner_model_state_transport_kind) not in (
        COMPACT_MODEL_STATE_TRANSPORT_KINDS
    ):
        raise ValueError(
            "defer_sample_learner_model_state_transport_kind must be one of "
            f"{COMPACT_MODEL_STATE_TRANSPORT_KINDS}"
        )
    if str(config.defer_sample_learner_replay_append_transport_kind) not in (
        COMPACT_REPLAY_APPEND_TRANSPORT_KINDS
    ):
        raise ValueError(
            "defer_sample_learner_replay_append_transport_kind must be one of "
            f"{COMPACT_REPLAY_APPEND_TRANSPORT_KINDS}"
        )
    if bool(config.defer_sample_learner_gate) and bool(config.defer_learner_gate):
        raise ValueError("defer_sample_learner_gate cannot be combined with defer_learner_gate")
    if not str(config.learner_impl).strip():
        raise ValueError("learner_impl must be non-empty")


def _combine_learner_results(
    first: Mapping[str, Any] | None,
    second: Mapping[str, Any],
) -> Mapping[str, Any]:
    if first is None:
        return second
    combined = dict(second)
    int_keys = (
        "compact_rollout_slab_learner_gate_updates",
        "compact_rollout_slab_learner_gate_sample_rows",
        "compact_rollout_slab_learner_gate_input_bytes",
    )
    for key in int_keys:
        combined[key] = int(first[key]) + int(second[key])
    sec_key = "compact_rollout_slab_learner_gate_sec"
    combined[sec_key] = float(first[sec_key]) + float(second[sec_key])
    for key, value in second.items():
        if (
            isinstance(key, str)
            and key.startswith("compact_rollout_slab_learner_gate_")
            and key.endswith("_sec")
            and key != sec_key
        ):
            combined[key] = float(first.get(key, 0.0) or 0.0) + float(value or 0.0)
    first_count = int(first.get("compact_owned_loop_learner_result_aggregate_count", 1))
    second_count = int(second.get("compact_owned_loop_learner_result_aggregate_count", 1))
    combined["compact_owned_loop_learner_result_aggregate_count"] = first_count + second_count
    return combined


def _run_deferred_learner_train(
    *,
    learner: Any,
    train_method: Any,
    train_input: Any,
    train_steps: int,
    model_state_transport_device: str | None = None,
    include_model_state: bool = True,
    model_state_transport_kind: str = COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
    model_state_snapshot_request_id: int = 0,
) -> Mapping[str, Any]:
    train_started = time.perf_counter()
    result = dict(train_method(train_input, train_steps=int(train_steps)))
    result.update(_learner_batch_resident_handle_consumption_fields(train_input))
    result["compact_owned_loop_deferred_learner_train_outer_sec"] = max(
        0.0,
        time.perf_counter() - train_started,
    )
    state_dict_fn = getattr(learner, "model_state_dict", None)
    model_state_fn_sec = 0.0
    model_state_clone_sec = 0.0
    model_state_digest_sec = 0.0
    model_state_prepare_started = time.perf_counter()
    transport_kind = str(model_state_transport_kind)
    if bool(include_model_state) and transport_kind == COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1:
        model_state_digest_started = time.perf_counter()
        digest_fn = getattr(learner, "model_state_digest", None)
        digest = str(digest_fn()) if callable(digest_fn) else None
        object_id_fn = getattr(learner, "model_object_id", None)
        object_id = int(object_id_fn()) if callable(object_id_fn) else None
        model_state_digest_sec = max(
            0.0,
            time.perf_counter() - model_state_digest_started,
        )
        if digest is not None:
            result[_DEFERRED_LEARNER_MODEL_STATE_DIGEST_KEY] = digest
        if object_id is not None:
            result[_DEFERRED_LEARNER_MODEL_OBJECT_ID_KEY] = object_id
        result[_DEFERRED_LEARNER_MODEL_OWNER_REF_KEY] = {
            "schema_id": "curvyzero_compact_owned_loop_model_owner_ref/v1",
            "transport_kind": COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
            "request_id": int(model_state_snapshot_request_id),
            "worker_pid": os.getpid(),
            "model_state_digest": digest,
            "model_object_id": object_id,
        }
    elif bool(include_model_state) and callable(state_dict_fn):
        model_state_fn_started = time.perf_counter()
        raw_state_dict = state_dict_fn()
        model_state_fn_sec = max(0.0, time.perf_counter() - model_state_fn_started)
        model_state_clone_started = time.perf_counter()
        state_dict = _clone_state_dict(
            raw_state_dict,
            target_device=model_state_transport_device,
        )
        model_state_clone_sec = max(
            0.0,
            time.perf_counter() - model_state_clone_started,
        )
        model_state_digest_started = time.perf_counter()
        digest_fn = getattr(learner, "model_state_digest", None)
        digest = str(digest_fn()) if callable(digest_fn) else None
        object_id_fn = getattr(learner, "model_object_id", None)
        object_id = int(object_id_fn()) if callable(object_id_fn) else None
        model_state_digest_sec = max(
            0.0,
            time.perf_counter() - model_state_digest_started,
        )
        if digest is not None:
            result[_DEFERRED_LEARNER_MODEL_STATE_DIGEST_KEY] = digest
        if object_id is not None:
            result[_DEFERRED_LEARNER_MODEL_OBJECT_ID_KEY] = object_id
        if transport_kind == COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1:
            result[_DEFERRED_LEARNER_MODEL_STATE_DICT_KEY] = state_dict
        elif transport_kind == COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1:
            result[_DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY] = (
                _write_model_state_snapshot_file(
                    state_dict,
                    request_id=int(model_state_snapshot_request_id),
                    digest=digest,
                    object_id=object_id,
                )
            )
        else:
            raise ValueError(
                "model_state_transport_kind must be one of "
                f"{COMPACT_MODEL_STATE_TRANSPORT_KINDS}"
            )
    result["compact_owned_loop_deferred_learner_model_state_prepare_sec"] = max(
        0.0,
        time.perf_counter() - model_state_prepare_started,
    )
    result["compact_owned_loop_deferred_learner_model_state_fn_sec"] = (
        model_state_fn_sec
    )
    result["compact_owned_loop_deferred_learner_model_state_clone_sec"] = (
        model_state_clone_sec
    )
    result["compact_owned_loop_deferred_learner_model_state_digest_sec"] = (
        model_state_digest_sec
    )
    return result


def _run_deferred_sample_learner_train(
    *,
    replay_store: Any,
    replay_snapshot: Any,
    learner: Any,
    seed: int,
    sample_batch_size: int,
    require_next_targets: bool,
    num_unroll_steps: int,
    fused_learner_batch: bool,
    train_steps: int,
    model_state_transport_device: str | None = None,
    return_model_state: bool = True,
    model_state_transport_kind: str = COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1,
    model_state_snapshot_request_id: int = 0,
) -> Mapping[str, Any]:
    worker_started = time.perf_counter()
    worker_pid = os.getpid()
    worker_cuda_device = _current_cuda_device_label()
    sample_from_snapshot = getattr(replay_store, "sample_from_snapshot", None)
    if not callable(sample_from_snapshot):
        raise RuntimeError("deferred sample+learner requires replay sample_from_snapshot")
    sample_started = time.perf_counter()
    sample_result = sample_from_snapshot(
        replay_snapshot,
        seed=int(seed),
        sample_batch_size=int(sample_batch_size),
        require_next_targets=bool(require_next_targets),
        num_unroll_steps=int(num_unroll_steps),
        build_compact_muzero_learner_batch=bool(fused_learner_batch),
        compact_muzero_learner_batch_only=bool(fused_learner_batch),
    )
    sample_sec = max(0.0, time.perf_counter() - sample_started)
    sample_batch = sample_result.get("sample_batch")
    learner_batch = sample_result.get("learner_batch")
    learner_result = None
    learner_sec = 0.0
    if sample_batch is not None or learner_batch is not None:
        train_input = sample_batch
        if learner_batch is not None:
            train_method = getattr(learner, "train_on_learner_batch", None)
            if not callable(train_method):
                raise RuntimeError("fused learner batch requested without learner support")
            train_input = learner_batch
        else:
            train_method = learner.train_on_sample_batch
        learner_started = time.perf_counter()
        learner_result = _run_deferred_learner_train(
            learner=learner,
            train_method=train_method,
            train_input=train_input,
            train_steps=int(train_steps),
            model_state_transport_device=model_state_transport_device,
            include_model_state=bool(return_model_state),
            model_state_transport_kind=str(model_state_transport_kind),
            model_state_snapshot_request_id=int(model_state_snapshot_request_id),
        )
        learner_sec = max(0.0, time.perf_counter() - learner_started)
        worker_cuda_device = _current_cuda_device_label()
    model_state_snapshot = (
        learner_result.get(_DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY)
        if isinstance(learner_result, Mapping)
        else None
    )
    model_owner_ref = (
        learner_result.get(_DEFERRED_LEARNER_MODEL_OWNER_REF_KEY)
        if isinstance(learner_result, Mapping)
        else None
    )
    model_state_returned = bool(
        isinstance(learner_result, Mapping)
        and (
            _DEFERRED_LEARNER_MODEL_STATE_DICT_KEY in learner_result
            or _DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY in learner_result
        )
    )
    return {
        "sample_result": sample_result,
        "learner_result": learner_result,
        "worker_runtime": {
            "worker_pid": worker_pid,
            "worker_resource_id": f"local_process:{worker_pid}:sample-learner",
            "worker_cuda_device": worker_cuda_device,
            "process_model_state_transport_kind": str(model_state_transport_kind),
            "process_model_state_returned": bool(model_state_returned),
            "process_model_state_omitted": bool(
                learner_result is not None and not model_state_returned
            ),
            "process_model_state_snapshot_returned": bool(model_state_snapshot),
            "process_model_owner_ref_returned": bool(model_owner_ref),
            "process_model_state_snapshot_bytes": int(
                dict(model_state_snapshot or {}).get("bytes", 0) or 0
            ),
            "process_model_state_snapshot_write_sec": float(
                dict(model_state_snapshot or {}).get("write_sec", 0.0) or 0.0
            ),
            "worker_job_wall_sec": max(0.0, time.perf_counter() - worker_started),
            "worker_sample_sec": float(sample_sec),
            "worker_learner_sec": float(learner_sec),
            "worker_model_state_prepare_sec": float(
                dict(learner_result or {}).get(
                    "compact_owned_loop_deferred_learner_model_state_prepare_sec",
                    0.0,
                )
                or 0.0
            ),
            "worker_model_state_fn_sec": float(
                dict(learner_result or {}).get(
                    "compact_owned_loop_deferred_learner_model_state_fn_sec",
                    0.0,
                )
                or 0.0
            ),
            "worker_model_state_clone_sec": float(
                dict(learner_result or {}).get(
                    "compact_owned_loop_deferred_learner_model_state_clone_sec",
                    0.0,
                )
                or 0.0
            ),
            "worker_model_state_digest_sec": float(
                dict(learner_result or {}).get(
                    "compact_owned_loop_deferred_learner_model_state_digest_sec",
                    0.0,
                )
                or 0.0
            ),
        },
    }


def _run_deferred_sample_learner_train_from_snapshot_class(
    *,
    replay_store_class: Any,
    replay_snapshot: Any,
    learner: Any,
    seed: int,
    sample_batch_size: int,
    require_next_targets: bool,
    num_unroll_steps: int,
    fused_learner_batch: bool,
    train_steps: int,
) -> Mapping[str, Any]:
    replay_store = replay_store_class(
        capacity=int(getattr(replay_snapshot, "capacity")),
        metadata=dict(getattr(replay_snapshot, "store_metadata", {}) or {}),
    )
    return _run_deferred_sample_learner_train(
        replay_store=replay_store,
        replay_snapshot=replay_snapshot,
        learner=learner,
        seed=int(seed),
        sample_batch_size=int(sample_batch_size),
        require_next_targets=bool(require_next_targets),
        num_unroll_steps=int(num_unroll_steps),
        fused_learner_batch=bool(fused_learner_batch),
        train_steps=int(train_steps),
        model_state_transport_device="cpu",
    )


def _run_deferred_sample_learner_train_from_worker_state(
    request: CompactProcessSampleLearnerTransportRequestV1,
) -> Mapping[str, Any]:
    global _PROCESS_WORKER_COMPLETED_COUNT
    worker_job_started = time.perf_counter()
    if _PROCESS_WORKER_REPLAY_STORE_CLASS is None or _PROCESS_WORKER_LEARNER is None:
        raise RuntimeError("process sample+learner worker was not initialized")
    request_cuda_count = _count_cuda_tensors(request)
    if request_cuda_count:
        raise RuntimeError("process sample+learner request contains CUDA tensors")
    replay_prepare_started = time.perf_counter()
    replay_store = _process_worker_replay_store_for_request(request)
    full_replay_snapshot_sent = bool(request.full_replay_snapshot_sent)
    worker_observation_provider_bootstrap_step_count = (
        _apply_process_worker_provider_bootstrap_request(request)
    )
    worker_observation_provider_materialized_entry_count = 0
    if full_replay_snapshot_sent:
        if request.replay_snapshot is None:
            raise RuntimeError("full replay snapshot request missing replay_snapshot")
        replay_store_for_sample = _PROCESS_WORKER_REPLAY_STORE_CLASS(
            capacity=int(getattr(request.replay_snapshot, "capacity")),
            metadata=dict(getattr(request.replay_snapshot, "store_metadata", {}) or {}),
        )
        replay_snapshot = request.replay_snapshot
        worker_owns_replay_store = False
    else:
        worker_observation_provider_materialized_entry_count = (
            _apply_process_worker_replay_append_request(replay_store, request)
        )
        replay_snapshot = replay_store.snapshot_for_sample()
        replay_store_for_sample = replay_store
        worker_owns_replay_store = True
    replay_prepare_sec = max(0.0, time.perf_counter() - replay_prepare_started)
    raw = _run_deferred_sample_learner_train(
        replay_store=replay_store_for_sample,
        replay_snapshot=replay_snapshot,
        learner=_PROCESS_WORKER_LEARNER,
        seed=int(request.seed),
        sample_batch_size=int(request.sample_batch_size),
        require_next_targets=bool(request.require_next_targets),
        num_unroll_steps=int(request.num_unroll_steps),
        fused_learner_batch=bool(request.fused_learner_batch),
        train_steps=int(request.train_steps),
        model_state_transport_device="cpu",
        return_model_state=bool(request.return_model_state),
        model_state_transport_kind=str(request.model_state_transport_kind),
        model_state_snapshot_request_id=int(request.request_id),
    )
    _PROCESS_WORKER_COMPLETED_COUNT += 1
    public_result_started = time.perf_counter()
    sample_result = _public_sample_result(raw.get("sample_result"))
    learner_result = raw.get("learner_result")
    public_result_sec = max(0.0, time.perf_counter() - public_result_started)
    model_state_snapshot = (
        learner_result.get(_DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY)
        if isinstance(learner_result, Mapping)
        else None
    )
    model_owner_ref = (
        learner_result.get(_DEFERRED_LEARNER_MODEL_OWNER_REF_KEY)
        if isinstance(learner_result, Mapping)
        else None
    )
    model_state_returned = bool(
        isinstance(learner_result, Mapping)
        and (
            _DEFERRED_LEARNER_MODEL_STATE_DICT_KEY in learner_result
            or _DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY in learner_result
        )
    )
    model_state_omitted = bool(learner_result is not None and not model_state_returned)
    result_cuda_count = _count_cuda_tensors(
        {
            "sample_result": sample_result,
            "learner_result": learner_result,
        }
    )
    result = {
        "sample_result": sample_result,
        "learner_result": learner_result,
        "worker_runtime": {
            **dict(raw.get("worker_runtime") or {}),
            "process_request_host_only": request_cuda_count == 0,
            "process_request_cuda_tensor_count": int(request_cuda_count),
            "process_result_host_only": result_cuda_count == 0,
            "process_result_cuda_tensor_count": int(result_cuda_count),
            "process_snapshot_host_clone_used": bool(request.snapshot_host_clone_used),
            "process_request_bytes": int(request.request_bytes),
            "process_result_bytes": 0,
            "process_full_replay_snapshot_sent": bool(full_replay_snapshot_sent),
            "process_replay_append_entry_count": int(request.replay_append_entry_count),
            "process_replay_append_index_row_count": int(
                request.replay_append_index_row_count
            ),
            "process_replay_append_entry_bytes": int(request.replay_append_entry_bytes),
            "process_replay_append_host_observation_bytes": int(
                request.replay_append_host_observation_bytes
            ),
            "process_replay_append_resident_snapshot_count": int(
                request.replay_append_resident_snapshot_count
            ),
            "process_replay_append_resident_snapshot_bytes": int(
                request.replay_append_resident_snapshot_bytes
            ),
            "process_replay_append_compact_batch_bytes": int(
                request.replay_append_compact_batch_bytes
            ),
            "process_replay_append_step_payload_bytes": int(
                request.replay_append_step_payload_bytes
            ),
            "process_replay_append_render_state_bytes": int(
                request.replay_append_render_state_bytes
            ),
            "process_provider_bootstrap_step_count": int(
                request.provider_bootstrap_step_count
            ),
            "process_provider_bootstrap_step_bytes": int(
                request.provider_bootstrap_step_bytes
            ),
            "process_provider_bootstrap_host_observation_bytes": int(
                request.provider_bootstrap_host_observation_bytes
            ),
            "process_provider_bootstrap_resident_snapshot_count": int(
                request.provider_bootstrap_resident_snapshot_count
            ),
            "process_provider_bootstrap_resident_snapshot_bytes": int(
                request.provider_bootstrap_resident_snapshot_bytes
            ),
            "process_provider_bootstrap_render_state_bytes": int(
                request.provider_bootstrap_render_state_bytes
            ),
            "process_provider_bootstrap_replay_entry_count": 0,
            "process_provider_bootstrap_replay_index_row_count": 0,
            "process_provider_bootstrap_learner_call_count": 0,
            "worker_observation_provider_present": bool(
                _PROCESS_WORKER_OBSERVATION_PROVIDER is not None
            ),
            "worker_observation_provider_bootstrap_step_count": int(
                worker_observation_provider_bootstrap_step_count
            ),
            "worker_observation_provider_missing_stack_history_count": int(
                getattr(
                    _PROCESS_WORKER_OBSERVATION_PROVIDER,
                    "missing_stack_history_count",
                    0,
                )
                or 0
            ),
            "worker_observation_provider_materialized_entry_count": int(
                worker_observation_provider_materialized_entry_count
            ),
            "process_model_state_transport_kind": str(request.model_state_transport_kind),
            "process_model_state_returned": bool(model_state_returned),
            "process_model_state_omitted": bool(model_state_omitted),
            "process_model_state_snapshot_returned": bool(model_state_snapshot),
            "process_model_owner_ref_returned": bool(model_owner_ref),
            "process_model_state_snapshot_bytes": int(
                dict(model_state_snapshot or {}).get("bytes", 0) or 0
            ),
            "process_model_state_snapshot_write_sec": float(
                dict(model_state_snapshot or {}).get("write_sec", 0.0) or 0.0
            ),
            "worker_owns_model_state": True,
            "worker_owns_replay_store": bool(worker_owns_replay_store),
            "worker_model_initialized_count": int(_PROCESS_WORKER_INITIALIZED_COUNT),
            "worker_completed_count": int(_PROCESS_WORKER_COMPLETED_COUNT),
            "worker_replay_append_count": int(
                _PROCESS_WORKER_REPLAY_APPEND_COUNT if worker_owns_replay_store else 0
            ),
            "worker_replay_entry_count": int(
                getattr(replay_store, "entry_count", 0)
                if worker_owns_replay_store
                else 0
            ),
            "worker_replay_index_row_count": int(
                getattr(replay_store, "stored_index_row_count", 0)
                if worker_owns_replay_store
                else 0
            ),
            "worker_replay_evicted_entry_count": int(
                getattr(replay_store, "evicted_entry_count", 0)
                if worker_owns_replay_store
                else 0
            ),
            "worker_replay_evicted_index_row_count": int(
                getattr(replay_store, "evicted_index_row_count", 0)
                if worker_owns_replay_store
                else 0
            ),
            "process_worker_replay_prepare_sec": float(replay_prepare_sec),
            "process_worker_result_public_sec": float(public_result_sec),
            "process_result_pickle_sec": 0.0,
            "process_worker_job_wall_sec": 0.0,
        },
    }
    result_pickle_started = time.perf_counter()
    result["worker_runtime"]["process_result_bytes"] = _pickle_size_bytes(result)
    result["worker_runtime"]["process_result_pickle_sec"] = max(
        0.0,
        time.perf_counter() - result_pickle_started,
    )
    result["worker_runtime"]["process_worker_job_wall_sec"] = max(
        0.0,
        time.perf_counter() - worker_job_started,
    )
    return result


def _process_worker_replay_store_for_request(
    request: CompactProcessSampleLearnerTransportRequestV1,
) -> Any:
    global _PROCESS_WORKER_REPLAY_STORE
    if bool(request.full_replay_snapshot_sent):
        if _PROCESS_WORKER_REPLAY_STORE is not None:
            return _PROCESS_WORKER_REPLAY_STORE
        if _PROCESS_WORKER_REPLAY_STORE_CLASS is None:
            raise RuntimeError("process sample+learner worker replay store is unavailable")
        capacity = int(
            request.replay_capacity
            if request.replay_capacity is not None
            else getattr(request.replay_snapshot, "capacity")
        )
        _PROCESS_WORKER_REPLAY_STORE = _PROCESS_WORKER_REPLAY_STORE_CLASS(
            capacity=capacity,
            metadata=dict(request.replay_store_metadata or {}),
        )
        return _PROCESS_WORKER_REPLAY_STORE
    if _PROCESS_WORKER_REPLAY_STORE is None:
        if _PROCESS_WORKER_REPLAY_STORE_CLASS is None:
            raise RuntimeError("process sample+learner worker replay store is unavailable")
        capacity = int(
            request.replay_capacity
            if request.replay_capacity is not None
            else getattr(request.replay_snapshot, "capacity", 0)
        )
        if capacity <= 0:
            raise RuntimeError("replay append request missing replay capacity")
        _PROCESS_WORKER_REPLAY_STORE = _PROCESS_WORKER_REPLAY_STORE_CLASS(
            capacity=capacity,
            metadata=dict(request.replay_store_metadata or {}),
        )
    return _PROCESS_WORKER_REPLAY_STORE


def _apply_process_worker_provider_bootstrap_request(
    request: CompactProcessSampleLearnerTransportRequestV1,
) -> int:
    steps = tuple(request.provider_bootstrap_steps)
    if not steps:
        return 0
    provider = _PROCESS_WORKER_OBSERVATION_PROVIDER
    if provider is None:
        raise RuntimeError(
            "process sample+learner provider bootstrap requires an observation provider"
        )
    bootstrap = getattr(provider, "bootstrap_compact_replay_step", None)
    if not callable(bootstrap):
        bootstrap = getattr(provider, "bootstrap_render_state_step", None)
    if not callable(bootstrap):
        bootstrap = getattr(provider, "bootstrap_step", None)
    if not callable(bootstrap):
        raise RuntimeError(
            "process sample+learner observation provider must implement "
            "bootstrap_compact_replay_step(step)"
        )
    for step in steps:
        _validate_provider_bootstrap_step(step)
        bootstrap(step)
    return len(steps)


def _apply_process_worker_replay_append_request(
    replay_store: Any,
    request: CompactProcessSampleLearnerTransportRequestV1,
) -> int:
    global _PROCESS_WORKER_REPLAY_APPEND_COUNT
    metadata = dict(request.replay_store_metadata or {})
    if metadata:
        current_metadata = getattr(replay_store, "store_metadata", None)
        current = current_metadata if isinstance(current_metadata, Mapping) else None
        if current is None or dict(current) != metadata:
            update = getattr(replay_store, "update_store_metadata", None)
            if not callable(update):
                raise RuntimeError("worker replay store cannot update metadata")
            update(metadata)
    append_entry = getattr(replay_store, "append_entry", None)
    if not callable(append_entry):
        raise RuntimeError("worker replay store cannot apply append entries")
    target_device = _process_worker_replay_tensor_target_device()
    materialized_entry_count = 0
    for entry in tuple(request.replay_append_entries):
        if _PROCESS_WORKER_OBSERVATION_PROVIDER is not None:
            materialized = _materialize_process_worker_replay_append_entry(
                entry,
                observation_provider=_PROCESS_WORKER_OBSERVATION_PROVIDER,
            )
            if materialized is not entry:
                materialized_entry_count += 1
            entry = materialized
        if target_device is not None:
            entry = _move_tensors_to_device(entry, target_device=target_device)
        append_entry(entry)
    _PROCESS_WORKER_REPLAY_APPEND_COUNT += len(tuple(request.replay_append_entries))
    return materialized_entry_count


def _materialize_process_worker_replay_append_entry(
    entry: Any,
    *,
    observation_provider: Any,
) -> Any:
    from curvyzero.training.source_state_hybrid_observation_profile import (
        _compact_replay_entry_requires_observation_provider,
        _materialize_compact_replay_observation_entry,
    )

    if not _compact_replay_entry_requires_observation_provider(entry):
        return entry
    return _materialize_compact_replay_observation_entry(
        entry,
        observation_provider=observation_provider,
    )


def _process_worker_replay_tensor_target_device() -> Any | None:
    learner = _PROCESS_WORKER_LEARNER
    if learner is None:
        return None
    try:
        import torch
    except Exception:
        return None
    raw_device = getattr(learner, "device", None)
    if raw_device is not None:
        device = torch.device(raw_device)
        return device if device.type == "cuda" else None
    model = getattr(learner, "model", None)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            for parameter in parameters():
                device = torch.device(parameter.device)
                if device.type == "cuda":
                    return device
                break
        except Exception:
            return None
    return None


def _move_tensors_to_device(
    value: Any,
    *,
    target_device: Any,
    _seen: dict[int, Any] | None = None,
    _depth: int = 0,
) -> Any:
    if value is None:
        return None
    if _depth > 20:
        return value
    seen = {} if _seen is None else _seen
    object_id = id(value)
    if object_id in seen:
        return seen[object_id]
    detach = getattr(value, "detach", None)
    if callable(detach) and hasattr(value, "device"):
        moved = detach().to(target_device, non_blocking=True).contiguous()
        seen[object_id] = moved
        return moved
    if isinstance(value, Mapping):
        moved = {
            key: _move_tensors_to_device(
                item,
                target_device=target_device,
                _seen=seen,
                _depth=_depth + 1,
            )
            for key, item in value.items()
        }
        seen[object_id] = moved
        return moved
    if isinstance(value, tuple):
        moved = tuple(
            _move_tensors_to_device(
                item,
                target_device=target_device,
                _seen=seen,
                _depth=_depth + 1,
            )
            for item in value
        )
        seen[object_id] = moved
        return moved
    if isinstance(value, list):
        moved = [
            _move_tensors_to_device(
                item,
                target_device=target_device,
                _seen=seen,
                _depth=_depth + 1,
            )
            for item in value
        ]
        seen[object_id] = moved
        return moved
    if isinstance(value, set):
        moved = {
            _move_tensors_to_device(
                item,
                target_device=target_device,
                _seen=seen,
                _depth=_depth + 1,
            )
            for item in value
        }
        seen[object_id] = moved
        return moved
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _move_tensors_to_device(
                getattr(value, field.name),
                target_device=target_device,
                _seen=seen,
                _depth=_depth + 1,
            )
            for field in fields(value)
        }
        moved = replace(value, **updates)
        seen[object_id] = moved
        return moved
    if isinstance(value, SimpleNamespace):
        moved = SimpleNamespace(
            **{
                key: _move_tensors_to_device(
                    item,
                    target_device=target_device,
                    _seen=seen,
                    _depth=_depth + 1,
                )
                for key, item in vars(value).items()
            }
        )
        seen[object_id] = moved
        return moved
    return value


def _public_sample_result(sample_result: Any) -> Mapping[str, Any] | None:
    if sample_result is None:
        return None
    sample = dict(sample_result)
    public = {
        key: value
        for key, value in sample.items()
        if str(key) not in {"sample_batch", "resident_sample_batch", "learner_batch"}
    }
    if "sample_metadata" not in public:
        sample_batch = sample.get("sample_batch")
        learner_batch = sample.get("learner_batch")
        metadata = (
            getattr(sample_batch, "metadata", None)
            or getattr(learner_batch, "metadata", None)
            or {}
        )
        public["sample_metadata"] = dict(metadata)
    return public


def _clone_state_dict(
    state_dict: Mapping[str, Any],
    *,
    target_device: str | None = None,
) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in dict(state_dict).items():
        detach = getattr(value, "detach", None)
        if callable(detach):
            detached = detach()
            if target_device:
                to_device = getattr(detached, "to", None)
                if callable(to_device):
                    detached = to_device(target_device)
            clone = getattr(detached, "clone", None)
            if callable(clone):
                cloned[str(key)] = clone()
                continue
        clone = getattr(value, "clone", None)
        if callable(clone):
            cloned[str(key)] = clone()
        else:
            cloned[str(key)] = copy.deepcopy(value)
    return cloned


def _write_model_state_snapshot_file(
    state_dict: Mapping[str, Any],
    *,
    request_id: int,
    digest: str | None = None,
    object_id: int | None = None,
) -> dict[str, Any]:
    snapshot_dir = os.path.join(
        tempfile.gettempdir(),
        "curvyzero_compact_model_state_snapshots",
    )
    os.makedirs(snapshot_dir, exist_ok=True)
    filename = (
        f"model_state_pid{os.getpid()}_request{int(request_id)}_"
        f"{time.time_ns()}.pkl"
    )
    path = os.path.join(snapshot_dir, filename)
    tmp_path = f"{path}.tmp"
    payload = {
        "schema_id": COMPACT_MODEL_STATE_SNAPSHOT_FILE_SCHEMA_ID,
        "model_state_dict": dict(state_dict),
        "model_state_digest": digest,
        "model_object_id": object_id,
        "request_id": int(request_id),
        "worker_pid": os.getpid(),
    }
    started = time.perf_counter()
    try:
        with open(tmp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    write_sec = max(0.0, time.perf_counter() - started)
    byte_count = int(os.path.getsize(path))
    return {
        "schema_id": COMPACT_MODEL_STATE_SNAPSHOT_FILE_SCHEMA_ID,
        "transport_kind": COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1,
        "path": path,
        "bytes": byte_count,
        "write_sec": write_sec,
        "model_state_digest": digest,
        "model_object_id": object_id,
        "request_id": int(request_id),
        "worker_pid": os.getpid(),
    }


def _load_model_state_snapshot_file(
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = dict(snapshot)
    if metadata.get("schema_id") != COMPACT_MODEL_STATE_SNAPSHOT_FILE_SCHEMA_ID:
        raise RuntimeError("model state snapshot file has the wrong schema")
    path = str(metadata.get("path") or "")
    if not path:
        raise RuntimeError("model state snapshot file is missing a path")
    started = time.perf_counter()
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise RuntimeError("model state snapshot file payload is not a mapping")
    if payload.get("schema_id") != COMPACT_MODEL_STATE_SNAPSHOT_FILE_SCHEMA_ID:
        raise RuntimeError("model state snapshot file payload has the wrong schema")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("model state snapshot file payload is missing model_state_dict")
    byte_count = int(metadata.get("bytes", 0) or 0)
    if byte_count <= 0:
        byte_count = int(os.path.getsize(path))
    try:
        os.unlink(path)
        deleted = True
    except OSError:
        deleted = False
    load_sec = max(0.0, time.perf_counter() - started)
    return dict(state_dict), {
        "bytes": byte_count,
        "load_sec": load_sec,
        "path_deleted": deleted,
    }


def _current_cuda_device_label() -> str:
    try:
        import torch
    except Exception:
        return "unavailable"
    try:
        if not bool(torch.cuda.is_available()):
            return "none"
        return f"cuda:{int(torch.cuda.current_device())}"
    except Exception as exc:
        return f"error:{exc.__class__.__name__}"


def _compact_process_worker_initializer(
    replay_store_class: Any | None = None,
    replay_capacity: int | None = None,
    replay_store_metadata: Mapping[str, Any] | None = None,
    learner: Any | None = None,
    learner_factory: Any | None = None,
    learner_factory_kwargs: Mapping[str, Any] | None = None,
    observation_provider_factory: Any | None = None,
    observation_provider_factory_kwargs: Mapping[str, Any] | None = None,
) -> None:
    global _PROCESS_WORKER_REPLAY_STORE_CLASS
    global _PROCESS_WORKER_REPLAY_STORE
    global _PROCESS_WORKER_LEARNER
    global _PROCESS_WORKER_OBSERVATION_PROVIDER
    global _PROCESS_WORKER_INITIALIZED_COUNT
    global _PROCESS_WORKER_COMPLETED_COUNT
    global _PROCESS_WORKER_REPLAY_APPEND_COUNT
    _PROCESS_WORKER_REPLAY_STORE_CLASS = replay_store_class
    if replay_store_class is not None and replay_capacity is not None:
        _PROCESS_WORKER_REPLAY_STORE = replay_store_class(
            capacity=int(replay_capacity),
            metadata=dict(replay_store_metadata or {}),
        )
    else:
        _PROCESS_WORKER_REPLAY_STORE = None
    if learner is None and callable(learner_factory):
        learner = learner_factory(**dict(learner_factory_kwargs or {}))
    _PROCESS_WORKER_LEARNER = learner
    _PROCESS_WORKER_OBSERVATION_PROVIDER = (
        observation_provider_factory(
            **dict(observation_provider_factory_kwargs or {})
        )
        if callable(observation_provider_factory)
        else None
    )
    _PROCESS_WORKER_INITIALIZED_COUNT = 1 if learner is not None else 0
    _PROCESS_WORKER_COMPLETED_COUNT = 0
    _PROCESS_WORKER_REPLAY_APPEND_COUNT = 0
    try:
        import torch
    except Exception:
        return
    memory = getattr(getattr(torch, "cuda", None), "memory", None)
    setter = getattr(memory, "_set_allocator_settings", None)
    if callable(setter):
        try:
            setter("expandable_segments:False")
        except Exception:
            return


def _sample_learner_request_contains_cuda_tensors(
    request: CompactSampleLearnerWorkRequestV1,
) -> bool:
    if _contains_cuda_tensor(request.replay_snapshot):
        return True
    if _contains_cuda_tensor(request.replay_append_entries):
        return True
    if _contains_cuda_tensor(request.provider_bootstrap_steps):
        return True
    return _learner_model_state_contains_cuda_tensors(request.learner)


def _sample_learner_process_payload_contains_cuda_tensors(
    request: CompactSampleLearnerWorkRequestV1,
) -> bool:
    return (
        _contains_cuda_tensor(request.replay_snapshot)
        or _contains_cuda_tensor(request.replay_append_entries)
        or _contains_cuda_tensor(request.provider_bootstrap_steps)
    )


def _learner_model_state_contains_cuda_tensors(learner: Any) -> bool:
    state_dict_fn = getattr(learner, "model_state_dict", None)
    if callable(state_dict_fn):
        try:
            if _contains_cuda_tensor(state_dict_fn()):
                return True
        except Exception:
            return True
    return False


def _learner_factory_kwargs_contain_cuda_tensors(kwargs: Mapping[str, Any]) -> bool:
    values = dict(kwargs)
    if _contains_cuda_tensor(values):
        return True
    model = values.get("model")
    if model is None:
        return False
    state_dict_fn = getattr(model, "state_dict", None)
    if callable(state_dict_fn):
        try:
            return _contains_cuda_tensor(state_dict_fn())
        except Exception:
            return True
    return False


def _pickle_size_bytes(value: Any) -> int:
    try:
        return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return -1


def _contains_cuda_tensor(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> bool:
    if value is None:
        return False
    if _depth > 10:
        return False
    seen = set() if _seen is None else _seen
    object_id = id(value)
    if object_id in seen:
        return False
    seen.add(object_id)
    is_cuda = getattr(value, "is_cuda", None)
    if isinstance(is_cuda, bool):
        return is_cuda
    if callable(is_cuda):
        try:
            return bool(is_cuda())
        except Exception:
            return True
    if isinstance(value, Mapping):
        return any(
            _contains_cuda_tensor(item, _seen=seen, _depth=_depth + 1)
            for item in value.values()
        )
    if isinstance(value, tuple | list | set | frozenset):
        return any(
            _contains_cuda_tensor(item, _seen=seen, _depth=_depth + 1) for item in value
        )
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_cuda_tensor(getattr(value, field.name), _seen=seen, _depth=_depth + 1)
            for field in fields(value)
        )
    if isinstance(value, SimpleNamespace):
        return any(
            _contains_cuda_tensor(item, _seen=seen, _depth=_depth + 1)
            for item in vars(value).values()
        )
    return False


def _host_only_clone(value: Any, *, _seen: dict[int, Any] | None = None, _depth: int = 0) -> Any:
    if value is None:
        return None
    if _depth > 20:
        return value
    seen = {} if _seen is None else _seen
    object_id = id(value)
    if object_id in seen:
        return seen[object_id]
    detach = getattr(value, "detach", None)
    if callable(detach) and hasattr(value, "is_cuda"):
        cloned = detach().to("cpu").clone().contiguous()
        seen[object_id] = cloned
        return cloned
    try:
        import numpy as _np
    except Exception:  # pragma: no cover - numpy is present in supported profiles.
        _np = None
    if _np is not None and isinstance(value, _np.ndarray):
        cloned = value.copy()
        seen[object_id] = cloned
        return cloned
    if isinstance(value, Mapping):
        cloned = {
            key: _host_only_clone(item, _seen=seen, _depth=_depth + 1)
            for key, item in value.items()
        }
        seen[object_id] = cloned
        return cloned
    if isinstance(value, tuple):
        cloned = tuple(_host_only_clone(item, _seen=seen, _depth=_depth + 1) for item in value)
        seen[object_id] = cloned
        return cloned
    if isinstance(value, list):
        cloned = [_host_only_clone(item, _seen=seen, _depth=_depth + 1) for item in value]
        seen[object_id] = cloned
        return cloned
    if isinstance(value, set):
        cloned = {_host_only_clone(item, _seen=seen, _depth=_depth + 1) for item in value}
        seen[object_id] = cloned
        return cloned
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _host_only_clone(
                getattr(value, field.name),
                _seen=seen,
                _depth=_depth + 1,
            )
            for field in fields(value)
        }
        cloned = replace(value, **updates)
        seen[object_id] = cloned
        return cloned
    if isinstance(value, SimpleNamespace):
        cloned = SimpleNamespace(
            **{
                key: _host_only_clone(item, _seen=seen, _depth=_depth + 1)
                for key, item in vars(value).items()
            }
        )
        seen[object_id] = cloned
        return cloned
    return value


def _count_cuda_tensors(value: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> int:
    if value is None:
        return 0
    if _depth > 10:
        return 0
    seen = set() if _seen is None else _seen
    object_id = id(value)
    if object_id in seen:
        return 0
    seen.add(object_id)
    is_cuda = getattr(value, "is_cuda", None)
    if isinstance(is_cuda, bool):
        return int(is_cuda)
    if callable(is_cuda):
        try:
            return int(bool(is_cuda()))
        except Exception:
            return 1
    if isinstance(value, Mapping):
        return sum(
            _count_cuda_tensors(item, _seen=seen, _depth=_depth + 1)
            for item in value.values()
        )
    if isinstance(value, tuple | list | set | frozenset):
        return sum(
            _count_cuda_tensors(item, _seen=seen, _depth=_depth + 1) for item in value
        )
    if is_dataclass(value) and not isinstance(value, type):
        return sum(
            _count_cuda_tensors(getattr(value, field.name), _seen=seen, _depth=_depth + 1)
            for field in fields(value)
        )
    if isinstance(value, SimpleNamespace):
        return sum(
            _count_cuda_tensors(item, _seen=seen, _depth=_depth + 1)
            for item in vars(value).values()
        )
    return 0


def _public_learner_result(learner_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in dict(learner_result).items()
        if str(key) not in _PRIVATE_LEARNER_RESULT_KEYS
    }


def _validate_policy_version(policy_version: CompactPolicyVersionRefV1) -> None:
    if not str(policy_version.policy_version_ref).strip():
        raise ValueError("policy_version_ref must be non-empty")
    if not str(policy_version.policy_source).strip():
        raise ValueError("policy_source must be non-empty")


__all__ = [
    "COMPACT_OWNED_LOOP_SCHEMA_ID",
    "COMPACT_REPLAY_APPEND_TRANSPORT_DURABLE_ENTRY_V1",
    "COMPACT_REPLAY_APPEND_TRANSPORT_KINDS",
    "COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1",
    "COMPACT_MODEL_STATE_SNAPSHOT_FILE_SCHEMA_ID",
    "COMPACT_MODEL_STATE_TRANSPORT_KINDS",
    "COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1",
    "COMPACT_MODEL_STATE_TRANSPORT_RESULT_V1",
    "COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1",
    "COMPACT_SAMPLE_LEARNER_WORKER_IN_PROCESS_THREAD",
    "COMPACT_SAMPLE_LEARNER_WORKER_KINDS",
    "COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS",
    "CompactProcessSampleLearnerWorkerV1",
    "CompactOwnedLoopConfigV1",
    "CompactOwnedLoopResultV1",
    "CompactOwnedLoopStepResultV1",
    "CompactOwnedLoopV1",
    "CompactPolicyVersionRefV1",
    "CompactPendingSampleLearnerWorkV1",
    "CompactProcessSampleLearnerTransportRequestV1",
    "CompactSampleLearnerWorkRequestV1",
    "_DEFERRED_LEARNER_MODEL_STATE_DICT_KEY",
    "_DEFERRED_LEARNER_MODEL_STATE_DIGEST_KEY",
    "_DEFERRED_LEARNER_MODEL_OBJECT_ID_KEY",
    "_DEFERRED_LEARNER_MODEL_OWNER_REF_KEY",
    "_DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY",
    "build_compact_sample_learner_worker_v1",
    "compact_owned_loop_replay_store_metadata",
]
