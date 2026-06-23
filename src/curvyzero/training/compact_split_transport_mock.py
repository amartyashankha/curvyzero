"""Small mock for the compact actor/search -> sample/learner split.

The mock is intentionally simple: it proves dataflow shape, not training
quality. The worker owns replay/model state and receives only host primitives.
"""

from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
import multiprocessing as mp
import os
import sys
import time
from typing import Any


MOCK_TRANSPORT_SCHEMA_ID = "curvyzero_compact_split_transport_mock/v1"
MOCK_TRANSPORT_KIND_HOST_ONLY = "host_only_process_service_mock"
MOCK_TRANSPORT_KIND_OWNER_SEARCH = "owner_search_process_service_mock"

_WORKER_REPLAY: list[tuple[int, int, float]] = []
_WORKER_MODEL_VERSION = 0
_WORKER_SEARCH_MODEL_VERSION = 0
_WORKER_COMPLETED = 0
_WORKER_SEARCH_COMPLETED = 0


@dataclass(frozen=True, slots=True)
class MockReplayEntryV1:
    """One actor/search replay row represented with host-only values."""

    step: int
    action: int
    reward: float


@dataclass(frozen=True, slots=True)
class MockSampleLearnerRequestV1:
    """Host-only message sent to the sample+learner worker."""

    request_id: int
    actor_step: int
    replay_entries: tuple[MockReplayEntryV1, ...]
    sample_batch_size: int
    train_steps: int
    worker_delay_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class MockSampleLearnerResultV1:
    """Result from one worker-owned sample+learner update."""

    request_id: int
    worker_pid: int
    replay_entry_count: int
    sampled_rows: int
    learner_update_count: int
    model_version: int
    worker_completed_count: int
    request_bytes: int
    result_bytes: int
    cuda_tensor_payload_count: int
    worker_owns_replay_state: bool
    worker_owns_model_state: bool


@dataclass(frozen=True, slots=True)
class MockOwnerSearchRequestV1:
    """One owner-side search plus optional replay/learner update request."""

    request_id: int
    actor_step: int
    root_slot_ids: tuple[int, ...]
    replay_entries: tuple[MockReplayEntryV1, ...]
    sample_batch_size: int
    train_steps: int
    worker_delay_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class MockOwnerSearchResultV1:
    """Small result from an owner that owns search, replay, and learner state."""

    request_id: int
    worker_pid: int
    actions: tuple[int, ...]
    root_slot_count: int
    replay_entry_count: int
    sampled_rows: int
    learner_update_count: int
    model_version: int
    search_model_version: int
    model_owner_ref: dict[str, Any]
    request_bytes: int
    result_bytes: int
    model_state_bytes: int
    cuda_tensor_payload_count: int
    root_observation_bytes_sent: int
    worker_owns_replay_state: bool
    worker_owns_model_state: bool
    worker_owns_search_state: bool


@dataclass(frozen=True, slots=True)
class MockSplitTransportReportV1:
    """Summary of the mock split run."""

    schema_id: str
    transport_kind: str
    ok: bool
    steps: int
    sample_interval: int
    max_pending: int
    submit_count: int
    completed_count: int
    actor_steps_while_pending: int
    policy_lag_max: int
    pending_count_at_end: int
    final_drain_in_wall_sec: bool
    final_drain_sec: float
    wall_sec: float
    request_bytes_total: int
    result_bytes_total: int
    cuda_tensor_payload_count: int
    worker_pid_distinct_from_actor: bool
    worker_owns_replay_state: bool
    worker_owns_model_state: bool
    final_model_version: int
    last_completed_request_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MockOwnerSearchTransportReportV1:
    """Summary for search/replay/learner colocated behind one owner boundary."""

    schema_id: str
    transport_kind: str
    ok: bool
    steps: int
    sample_interval: int
    max_pending: int
    search_request_count: int
    search_result_count: int
    final_refresh_request_id: int
    actor_steps_while_pending: int
    policy_lag_max: int
    pending_count_at_end: int
    final_drain_in_wall_sec: bool
    final_drain_sec: float
    wall_sec: float
    request_bytes_total: int
    result_bytes_total: int
    model_state_bytes_total: int
    root_slot_id_count: int
    root_observation_bytes_sent: int
    cuda_tensor_payload_count: int
    worker_pid_distinct_from_actor: bool
    worker_owns_replay_state: bool
    worker_owns_model_state: bool
    worker_owns_search_state: bool
    model_state_return_count: int
    owner_ref_result_count: int
    final_model_version: int
    final_search_model_version: int
    search_consumed_final_update: bool
    last_completed_request_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_host_only_split_transport_mock(
    *,
    steps: int = 12,
    sample_interval: int = 2,
    max_pending: int = 2,
    sample_batch_size: int = 4,
    train_steps: int = 1,
    worker_delay_sec: float = 0.01,
) -> MockSplitTransportReportV1:
    """Run a tiny host-only split mock and return proof counters."""

    steps_int = int(steps)
    sample_interval_int = int(sample_interval)
    max_pending_int = int(max_pending)
    if steps_int <= 0:
        raise ValueError("steps must be positive")
    if sample_interval_int <= 0:
        raise ValueError("sample_interval must be positive")
    if max_pending_int <= 0:
        raise ValueError("max_pending must be positive")

    actor_pid = os.getpid()
    replay_log: list[MockReplayEntryV1] = []
    pending: list[Future[MockSampleLearnerResultV1]] = []
    submit_count = 0
    completed_count = 0
    actor_steps_while_pending = 0
    policy_lag_max = 0
    final_model_version = 0
    last_completed_request_id = 0
    request_bytes_total = 0
    result_bytes_total = 0
    cuda_tensor_payload_count = 0
    worker_pid = 0
    started = time.perf_counter()
    final_drain_sec = 0.0

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        initializer=_mock_worker_init,
    ) as executor:
        for step in range(steps_int):
            if pending:
                actor_steps_while_pending += 1
            replay_log.append(
                MockReplayEntryV1(
                    step=step,
                    action=step % 3,
                    reward=float((step % 5) - 2),
                )
            )
            completed = _consume_completed(pending, wait=False)
            for result in completed:
                completed_count += 1
                last_completed_request_id = int(result.request_id)
                final_model_version = int(result.model_version)
                request_bytes_total += int(result.request_bytes)
                result_bytes_total += int(result.result_bytes)
                cuda_tensor_payload_count += int(result.cuda_tensor_payload_count)
                worker_pid = int(result.worker_pid)
            if (step + 1) % sample_interval_int != 0:
                continue
            while len(pending) >= max_pending_int:
                blocked = _consume_completed(pending, wait=True)
                for result in blocked:
                    completed_count += 1
                    last_completed_request_id = int(result.request_id)
                    final_model_version = int(result.model_version)
                    request_bytes_total += int(result.request_bytes)
                    result_bytes_total += int(result.result_bytes)
                    cuda_tensor_payload_count += int(result.cuda_tensor_payload_count)
                    worker_pid = int(result.worker_pid)
            request = MockSampleLearnerRequestV1(
                request_id=submit_count + 1,
                actor_step=step,
                replay_entries=tuple(replay_log),
                sample_batch_size=int(sample_batch_size),
                train_steps=int(train_steps),
                worker_delay_sec=float(worker_delay_sec),
            )
            if _contains_cuda_tensor_like(request):
                raise RuntimeError("mock host-only request unexpectedly contains CUDA tensors")
            pending.append(executor.submit(_mock_sample_learner_worker_step, request))
            submit_count += 1
            policy_lag_max = max(policy_lag_max, submit_count - completed_count)

        drain_started = time.perf_counter()
        while pending:
            drained = _consume_completed(pending, wait=True)
            for result in drained:
                completed_count += 1
                last_completed_request_id = int(result.request_id)
                final_model_version = int(result.model_version)
                request_bytes_total += int(result.request_bytes)
                result_bytes_total += int(result.result_bytes)
                cuda_tensor_payload_count += int(result.cuda_tensor_payload_count)
                worker_pid = int(result.worker_pid)
        final_drain_sec = max(0.0, time.perf_counter() - drain_started)

    wall_sec = max(0.0, time.perf_counter() - started)
    ok = (
        submit_count > 0
        and completed_count == submit_count
        and last_completed_request_id == completed_count
        and actor_steps_while_pending > 0
        and policy_lag_max > 0
        and cuda_tensor_payload_count == 0
        and worker_pid > 0
        and worker_pid != actor_pid
        and final_model_version == completed_count * int(train_steps)
    )
    return MockSplitTransportReportV1(
        schema_id=MOCK_TRANSPORT_SCHEMA_ID,
        transport_kind=MOCK_TRANSPORT_KIND_HOST_ONLY,
        ok=ok,
        steps=steps_int,
        sample_interval=sample_interval_int,
        max_pending=max_pending_int,
        submit_count=submit_count,
        completed_count=completed_count,
        actor_steps_while_pending=actor_steps_while_pending,
        policy_lag_max=policy_lag_max,
        pending_count_at_end=len(pending),
        final_drain_in_wall_sec=True,
        final_drain_sec=final_drain_sec,
        wall_sec=wall_sec,
        request_bytes_total=request_bytes_total,
        result_bytes_total=result_bytes_total,
        cuda_tensor_payload_count=cuda_tensor_payload_count,
        worker_pid_distinct_from_actor=worker_pid > 0 and worker_pid != actor_pid,
        worker_owns_replay_state=True,
        worker_owns_model_state=True,
        final_model_version=final_model_version,
        last_completed_request_id=last_completed_request_id,
    )


def run_owner_search_split_transport_mock(
    *,
    steps: int = 12,
    sample_interval: int = 2,
    max_pending: int = 2,
    roots_per_step: int = 2,
    sample_batch_size: int = 4,
    train_steps: int = 1,
    worker_delay_sec: float = 0.01,
) -> MockOwnerSearchTransportReportV1:
    """Run a tiny owner-side search/replay/learner transport mock.

    The parent sends root slot ids and replay append deltas. The worker owns
    replay, learner model state, and search model state, then returns actions
    plus an owner ref. No model state bytes are returned.
    """

    steps_int = int(steps)
    sample_interval_int = int(sample_interval)
    max_pending_int = int(max_pending)
    roots_per_step_int = int(roots_per_step)
    if steps_int <= 0:
        raise ValueError("steps must be positive")
    if sample_interval_int <= 0:
        raise ValueError("sample_interval must be positive")
    if max_pending_int <= 0:
        raise ValueError("max_pending must be positive")
    if roots_per_step_int <= 0:
        raise ValueError("roots_per_step must be positive")

    actor_pid = os.getpid()
    pending: list[Future[MockOwnerSearchResultV1]] = []
    search_request_count = 0
    search_result_count = 0
    actor_steps_while_pending = 0
    policy_lag_max = 0
    final_model_version = 0
    final_search_model_version = 0
    final_refresh_request_id = 0
    last_completed_request_id = 0
    request_bytes_total = 0
    result_bytes_total = 0
    model_state_bytes_total = 0
    root_slot_id_count = 0
    root_observation_bytes_sent = 0
    cuda_tensor_payload_count = 0
    worker_pid = 0
    owner_ref_result_count = 0
    started = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context("spawn"),
        initializer=_mock_worker_init,
    ) as executor:
        for step in range(steps_int):
            if pending:
                actor_steps_while_pending += 1
            for result in _consume_completed_owner_search(pending, wait=False):
                search_result_count += 1
                last_completed_request_id = int(result.request_id)
                final_model_version = int(result.model_version)
                final_search_model_version = int(result.search_model_version)
                request_bytes_total += int(result.request_bytes)
                result_bytes_total += int(result.result_bytes)
                model_state_bytes_total += int(result.model_state_bytes)
                root_observation_bytes_sent += int(result.root_observation_bytes_sent)
                cuda_tensor_payload_count += int(result.cuda_tensor_payload_count)
                owner_ref_result_count += int(bool(result.model_owner_ref))
                worker_pid = int(result.worker_pid)
            while len(pending) >= max_pending_int:
                for result in _consume_completed_owner_search(pending, wait=True):
                    search_result_count += 1
                    last_completed_request_id = int(result.request_id)
                    final_model_version = int(result.model_version)
                    final_search_model_version = int(result.search_model_version)
                    request_bytes_total += int(result.request_bytes)
                    result_bytes_total += int(result.result_bytes)
                    model_state_bytes_total += int(result.model_state_bytes)
                    root_observation_bytes_sent += int(result.root_observation_bytes_sent)
                    cuda_tensor_payload_count += int(result.cuda_tensor_payload_count)
                    owner_ref_result_count += int(bool(result.model_owner_ref))
                    worker_pid = int(result.worker_pid)
            train_steps_for_request = (
                int(train_steps) if (step + 1) % sample_interval_int == 0 else 0
            )
            root_slot_ids = tuple(
                range(step * roots_per_step_int, (step + 1) * roots_per_step_int)
            )
            root_slot_id_count += len(root_slot_ids)
            request = MockOwnerSearchRequestV1(
                request_id=search_request_count + 1,
                actor_step=step,
                root_slot_ids=root_slot_ids,
                replay_entries=(
                    MockReplayEntryV1(
                        step=step,
                        action=step % 3,
                        reward=float((step % 5) - 2),
                    ),
                ),
                sample_batch_size=int(sample_batch_size),
                train_steps=train_steps_for_request,
                worker_delay_sec=float(worker_delay_sec),
            )
            if _contains_cuda_tensor_like(request):
                raise RuntimeError("mock owner-search request unexpectedly has CUDA tensors")
            pending.append(executor.submit(_mock_owner_search_worker_step, request))
            search_request_count += 1
            policy_lag_max = max(policy_lag_max, search_request_count - search_result_count)

        drain_started = time.perf_counter()
        while pending:
            for result in _consume_completed_owner_search(pending, wait=True):
                search_result_count += 1
                last_completed_request_id = int(result.request_id)
                final_model_version = int(result.model_version)
                final_search_model_version = int(result.search_model_version)
                request_bytes_total += int(result.request_bytes)
                result_bytes_total += int(result.result_bytes)
                model_state_bytes_total += int(result.model_state_bytes)
                root_observation_bytes_sent += int(result.root_observation_bytes_sent)
                cuda_tensor_payload_count += int(result.cuda_tensor_payload_count)
                owner_ref_result_count += int(bool(result.model_owner_ref))
                worker_pid = int(result.worker_pid)
        final_refresh_request_id = search_request_count + 1
        final_request = MockOwnerSearchRequestV1(
            request_id=final_refresh_request_id,
            actor_step=steps_int,
            root_slot_ids=(steps_int * roots_per_step_int,),
            replay_entries=(),
            sample_batch_size=int(sample_batch_size),
            train_steps=0,
            worker_delay_sec=0.0,
        )
        final_future = executor.submit(_mock_owner_search_worker_step, final_request)
        final_result = final_future.result()
        search_request_count += 1
        search_result_count += 1
        last_completed_request_id = int(final_result.request_id)
        final_model_version = int(final_result.model_version)
        final_search_model_version = int(final_result.search_model_version)
        request_bytes_total += int(final_result.request_bytes)
        result_bytes_total += int(final_result.result_bytes)
        model_state_bytes_total += int(final_result.model_state_bytes)
        root_slot_id_count += int(final_result.root_slot_count)
        root_observation_bytes_sent += int(final_result.root_observation_bytes_sent)
        cuda_tensor_payload_count += int(final_result.cuda_tensor_payload_count)
        owner_ref_result_count += int(bool(final_result.model_owner_ref))
        worker_pid = int(final_result.worker_pid)
        final_drain_sec = max(0.0, time.perf_counter() - drain_started)

    wall_sec = max(0.0, time.perf_counter() - started)
    search_consumed_final_update = (
        final_model_version > 0 and final_search_model_version == final_model_version
    )
    ok = (
        search_request_count == steps_int + 1
        and search_result_count == search_request_count
        and last_completed_request_id == final_refresh_request_id
        and actor_steps_while_pending > 0
        and policy_lag_max > 0
        and cuda_tensor_payload_count == 0
        and root_observation_bytes_sent == 0
        and model_state_bytes_total == 0
        and owner_ref_result_count == search_result_count
        and worker_pid > 0
        and worker_pid != actor_pid
        and search_consumed_final_update
    )
    return MockOwnerSearchTransportReportV1(
        schema_id=MOCK_TRANSPORT_SCHEMA_ID,
        transport_kind=MOCK_TRANSPORT_KIND_OWNER_SEARCH,
        ok=ok,
        steps=steps_int,
        sample_interval=sample_interval_int,
        max_pending=max_pending_int,
        search_request_count=search_request_count,
        search_result_count=search_result_count,
        final_refresh_request_id=final_refresh_request_id,
        actor_steps_while_pending=actor_steps_while_pending,
        policy_lag_max=policy_lag_max,
        pending_count_at_end=len(pending),
        final_drain_in_wall_sec=True,
        final_drain_sec=final_drain_sec,
        wall_sec=wall_sec,
        request_bytes_total=request_bytes_total,
        result_bytes_total=result_bytes_total,
        model_state_bytes_total=model_state_bytes_total,
        root_slot_id_count=root_slot_id_count,
        root_observation_bytes_sent=root_observation_bytes_sent,
        cuda_tensor_payload_count=cuda_tensor_payload_count,
        worker_pid_distinct_from_actor=worker_pid > 0 and worker_pid != actor_pid,
        worker_owns_replay_state=True,
        worker_owns_model_state=True,
        worker_owns_search_state=True,
        model_state_return_count=0,
        owner_ref_result_count=owner_ref_result_count,
        final_model_version=final_model_version,
        final_search_model_version=final_search_model_version,
        search_consumed_final_update=search_consumed_final_update,
        last_completed_request_id=last_completed_request_id,
    )


def _mock_worker_init() -> None:
    global _WORKER_REPLAY
    global _WORKER_MODEL_VERSION
    global _WORKER_SEARCH_MODEL_VERSION
    global _WORKER_COMPLETED
    global _WORKER_SEARCH_COMPLETED
    _WORKER_REPLAY = []
    _WORKER_MODEL_VERSION = 0
    _WORKER_SEARCH_MODEL_VERSION = 0
    _WORKER_COMPLETED = 0
    _WORKER_SEARCH_COMPLETED = 0


def _mock_sample_learner_worker_step(
    request: MockSampleLearnerRequestV1,
) -> MockSampleLearnerResultV1:
    global _WORKER_REPLAY
    global _WORKER_MODEL_VERSION
    global _WORKER_COMPLETED
    if _contains_cuda_tensor_like(request):
        raise RuntimeError("mock worker received CUDA tensor payload")
    if float(request.worker_delay_sec) > 0.0:
        time.sleep(float(request.worker_delay_sec))
    _WORKER_REPLAY = [
        (int(entry.step), int(entry.action), float(entry.reward))
        for entry in tuple(request.replay_entries)
    ]
    sampled_rows = min(int(request.sample_batch_size), len(_WORKER_REPLAY))
    _WORKER_MODEL_VERSION += int(request.train_steps)
    _WORKER_COMPLETED += 1
    result_stub = {
        "request_id": int(request.request_id),
        "model_version": int(_WORKER_MODEL_VERSION),
        "sampled_rows": int(sampled_rows),
    }
    return MockSampleLearnerResultV1(
        request_id=int(request.request_id),
        worker_pid=os.getpid(),
        replay_entry_count=len(_WORKER_REPLAY),
        sampled_rows=sampled_rows,
        learner_update_count=int(request.train_steps),
        model_version=int(_WORKER_MODEL_VERSION),
        worker_completed_count=int(_WORKER_COMPLETED),
        request_bytes=sys.getsizeof(request) + sum(
            sys.getsizeof(entry) for entry in tuple(request.replay_entries)
        ),
        result_bytes=sys.getsizeof(result_stub),
        cuda_tensor_payload_count=0,
        worker_owns_replay_state=True,
        worker_owns_model_state=True,
    )


def _mock_owner_search_worker_step(
    request: MockOwnerSearchRequestV1,
) -> MockOwnerSearchResultV1:
    global _WORKER_REPLAY
    global _WORKER_MODEL_VERSION
    global _WORKER_SEARCH_MODEL_VERSION
    global _WORKER_SEARCH_COMPLETED
    if _contains_cuda_tensor_like(request):
        raise RuntimeError("mock owner-search worker received CUDA tensor payload")
    if float(request.worker_delay_sec) > 0.0:
        time.sleep(float(request.worker_delay_sec))
    search_model_version = int(_WORKER_SEARCH_MODEL_VERSION)
    actions = tuple(
        (int(root_slot_id) + search_model_version) % 3
        for root_slot_id in tuple(request.root_slot_ids)
    )
    for entry in tuple(request.replay_entries):
        _WORKER_REPLAY.append((int(entry.step), int(entry.action), float(entry.reward)))
    sampled_rows = 0
    learner_update_count = 0
    if int(request.train_steps) > 0:
        sampled_rows = min(int(request.sample_batch_size), len(_WORKER_REPLAY))
        learner_update_count = int(request.train_steps)
        _WORKER_MODEL_VERSION += learner_update_count
        _WORKER_SEARCH_MODEL_VERSION = int(_WORKER_MODEL_VERSION)
    _WORKER_SEARCH_COMPLETED += 1
    owner_ref = {
        "schema_id": "curvyzero_compact_owned_loop_model_owner_ref/v1",
        "transport_kind": "owner_ref_v1",
        "model_version": int(_WORKER_MODEL_VERSION),
        "model_state_digest": f"mock-model-v{int(_WORKER_MODEL_VERSION)}",
        "worker_pid": os.getpid(),
    }
    result_stub = {
        "request_id": int(request.request_id),
        "actions": actions,
        "model_owner_ref": owner_ref,
    }
    return MockOwnerSearchResultV1(
        request_id=int(request.request_id),
        worker_pid=os.getpid(),
        actions=actions,
        root_slot_count=len(tuple(request.root_slot_ids)),
        replay_entry_count=len(_WORKER_REPLAY),
        sampled_rows=sampled_rows,
        learner_update_count=learner_update_count,
        model_version=int(_WORKER_MODEL_VERSION),
        search_model_version=search_model_version,
        model_owner_ref=owner_ref,
        request_bytes=sys.getsizeof(request)
        + sys.getsizeof(tuple(request.root_slot_ids))
        + sum(sys.getsizeof(entry) for entry in tuple(request.replay_entries)),
        result_bytes=sys.getsizeof(result_stub)
        + sys.getsizeof(actions)
        + sys.getsizeof(owner_ref),
        model_state_bytes=0,
        cuda_tensor_payload_count=0,
        root_observation_bytes_sent=0,
        worker_owns_replay_state=True,
        worker_owns_model_state=True,
        worker_owns_search_state=True,
    )


def _consume_completed(
    pending: list[Future[MockSampleLearnerResultV1]],
    *,
    wait: bool,
) -> list[MockSampleLearnerResultV1]:
    completed: list[MockSampleLearnerResultV1] = []
    while pending:
        future = pending[0]
        if not wait and not future.done():
            break
        completed.append(future.result())
        pending.pop(0)
        if not wait:
            continue
    return completed


def _consume_completed_owner_search(
    pending: list[Future[MockOwnerSearchResultV1]],
    *,
    wait: bool,
) -> list[MockOwnerSearchResultV1]:
    completed: list[MockOwnerSearchResultV1] = []
    while pending:
        future = pending[0]
        if not wait and not future.done():
            break
        completed.append(future.result())
        pending.pop(0)
        if not wait:
            continue
    return completed


def _contains_cuda_tensor_like(value: Any, *, _seen: set[int] | None = None) -> bool:
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
    if isinstance(value, dict):
        return any(_contains_cuda_tensor_like(item, _seen=seen) for item in value.values())
    if isinstance(value, tuple | list | set | frozenset):
        return any(_contains_cuda_tensor_like(item, _seen=seen) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return any(
            _contains_cuda_tensor_like(getattr(value, name), _seen=seen)
            for name in value.__dataclass_fields__
        )
    return False


__all__ = [
    "MOCK_TRANSPORT_KIND_OWNER_SEARCH",
    "MOCK_TRANSPORT_KIND_HOST_ONLY",
    "MOCK_TRANSPORT_SCHEMA_ID",
    "MockOwnerSearchRequestV1",
    "MockOwnerSearchResultV1",
    "MockOwnerSearchTransportReportV1",
    "MockReplayEntryV1",
    "MockSampleLearnerRequestV1",
    "MockSampleLearnerResultV1",
    "MockSplitTransportReportV1",
    "run_host_only_split_transport_mock",
    "run_owner_search_split_transport_mock",
]
