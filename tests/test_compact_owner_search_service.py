from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
import os
import time

import numpy as np
import pytest

from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_KIND_INLINE,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_KIND_INLINE_BACKGROUND,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_KIND_IN_PROCESS,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_KIND_LOCAL_PROCESS,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_KIND_THREADED,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_OWNER_REF_DIGEST_DEFERRED_TO_SEARCH_REFRESH_KEY,
)
from curvyzero.training.compact_owner_search_service import (
    COMPACT_OWNER_SEARCH_RESULT_SCHEMA_ID,
)
from curvyzero.training.compact_owner_search_service import CompactDirectRootStoreV1
from curvyzero.training.compact_owner_search_service import (
    CompactOwnerActionDispatchHandleV1,
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
    CompactOwnerMaintenanceDrainRequestV1,
)
from curvyzero.training.compact_owner_search_service import CompactOwnerSearchRequestV1
from curvyzero.training.compact_owner_search_service import CompactOwnerSearchServiceV1
from curvyzero.training.compact_owner_search_service import CompactOwnerSearchSlabProxyV1
from curvyzero.training.compact_owner_search_service import (
    CompactProcessOwnerSearchLearnerWorkerV1,
)
from curvyzero.training.compact_owner_search_service import CompactProcessOwnerSearchWorkerV1
from curvyzero.training.compact_owner_search_service import CompactSharedMemoryRootStoreV1
from curvyzero.training.compact_owner_search_service import (
    build_compact_resident_shared_memory_root_provider_v1,
)
from curvyzero.training.compact_owner_search_service import (
    build_compact_shared_memory_root_provider_v1,
)
from curvyzero.training.compact_owner_search_service import (
    compact_action_step_v1_from_owner_search_payload_and_root_request,
)
from curvyzero.training.compact_owner_search_service import _contains_cuda_tensor
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_observation_contract import ResidentObservationBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactDeviceReplayIndexRowsV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_root_batch_v1_from_request,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_root_build_request_v1_from_batch,
)
from curvyzero.training.compact_policy_row_bridge import (
    validate_resident_observation_batch_v1,
)
from curvyzero.training.compact_policy_row_bridge import validate_compact_search_result_v1
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND,
)
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID,
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
from curvyzero.training.compact_rollout_slab import (
    COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION,
)
from curvyzero.training.compact_rollout_slab import (
    CompactOwnerSearchDirectStepDispatchHandleV1,
)
from curvyzero.training.compact_rollout_slab import CompactOwnerSearchDirectStepperV1
from curvyzero.training.compact_rollout_slab import (
    CompactOwnerSearchReplayAppendDerivedTransitionBatchV1,
)
from curvyzero.training.compact_rollout_slab import (
    CompactOwnerSearchReplayAppendIndexEntryV1,
)
from curvyzero.training.compact_rollout_slab import (
    CompactOwnerSearchReplayAppendTransitionBatchV1,
)
from curvyzero.training.compact_rollout_slab import (
    CompactOwnerSearchReplayAppendTransitionEntryV1,
)
from curvyzero.training.compact_rollout_slab import CompactRolloutSlab
from curvyzero.training.compact_search_service import (
    COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID
from curvyzero.training.compact_search_service import CompactDeviceSearchReplayPayloadV1
from curvyzero.training.compact_search_service import CompactSearchActionStepV1
from curvyzero.training.compact_search_service import compact_search_array_digest_v1
from curvyzero.training.compact_search_service import (
    compact_search_deferred_replay_payload_digest_v1,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


def _make_unit_transition_batch() -> CompactOwnerSearchReplayAppendTransitionBatchV1:
    return CompactOwnerSearchReplayAppendTransitionBatchV1(
        schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID,
        transition_count=2,
        record_indices=np.asarray((0, 1), dtype=np.int64),
        next_record_indices=np.asarray((1, 2), dtype=np.int64),
        replay_payload_handles=("handle-0", "handle-1"),
        selected_action_digests=("selected-0", "selected-1"),
        search_replay_payload_digests=("replay-0", "replay-1"),
        next_joint_action=np.asarray(
            (((0,), (1,), (2,)), ((1,), (2,), (0,))),
            dtype=np.int16,
        ),
        next_reward=np.zeros((2, 3, 1), dtype=np.float32),
        next_done=np.zeros((2, 3), dtype=np.bool_),
        next_terminated=np.zeros((2, 3), dtype=np.bool_),
        next_truncated=np.zeros((2, 3), dtype=np.bool_),
        next_final_reward_map=np.zeros((2, 3, 1), dtype=np.float32),
        next_final_observation_row_mask=np.zeros((2, 3), dtype=np.bool_),
        policy_source="unit",
        metadata={
            "compact_owner_search_replay_append_transition_batch": True,
            "compact_owner_search_replay_append_transition_batch_transition_count": 2,
            "compact_owner_search_owner_replay_transition_batch_count": 1,
            "compact_owner_search_owner_replay_transition_batch_transition_count": 2,
            "compact_owner_search_owner_replay_transition_legacy_entry_count": 0,
        },
    )


def test_owner_search_service_keeps_search_replay_and_model_publication_owned():
    root_provider = _FakeRootProvider()
    search_service = _FakeOwnerSearchService()
    replay_store = _FakeOwnerReplayStore()
    learner = _FakeOwnerLearner()
    service = CompactOwnerSearchServiceV1(
        root_provider=root_provider,
        search_service=search_service,
        replay_store=replay_store,
        learner=learner,
    )

    result = service.run(
        CompactOwnerSearchRequestV1(
            request_id=1,
            actor_step=5,
            root_slot_ids=(10, 11, 12),
            replay_append_entries=({"row": 1}, {"row": 2}),
            sample_batch_size=2,
            train_steps=2,
            policy_version_ref="policy-v1",
            model_version_ref="model-v1",
            policy_source="unit",
        )
    )
    payload = result.to_dict()

    assert payload["schema_id"] == COMPACT_OWNER_SEARCH_RESULT_SCHEMA_ID
    assert payload["owner_kind"] == COMPACT_OWNER_SEARCH_KIND_IN_PROCESS
    assert payload["root_slot_count"] == 3
    assert payload["active_root_count"] == 3
    assert payload["selected_action"] == (0, 1, 2)
    assert payload["replay_append_entry_count"] == 2
    assert payload["replay_append_count"] == 2
    assert payload["learner_update_count"] == 2
    assert payload["model_owner_ref_returned"] is True
    assert payload["model_owner_ref_digest"] == "owner-digest-2"
    assert payload["model_state_return_count"] == 0
    assert payload["model_state_bytes"] == 0
    assert payload["model_state_snapshot_return_count"] == 0
    assert payload["root_observation_bytes_sent"] == 0
    assert payload["search_result_payload_bytes"] > 0
    search_payload = payload["search_result_payload"]
    assert search_payload["payload_transport_kind"] == "numpy_ndarray_ipc_v1"
    assert search_payload["payload_json_safe"] is False
    assert isinstance(search_payload["selected_action"], np.ndarray)
    assert tuple(int(value) for value in search_payload["selected_action"]) == (0, 1, 2)
    assert payload["search_selected_action_bytes"] > 0
    assert payload["search_visit_policy_bytes"] > 0
    assert payload["search_root_value_bytes"] > 0
    assert payload["request_cuda_tensor_count"] == 0
    assert payload["result_cuda_tensor_count"] == 0
    assert payload["worker_owns_search_state"] is True
    assert payload["worker_owns_replay_state"] is True
    assert payload["worker_owns_model_state"] is True
    assert payload["search_consumed_learner_update"] is True
    assert payload["search_refresh_update_count"] == 2
    assert root_provider.seen_slot_ids == [(10, 11, 12)]
    assert search_service.run_count == 1
    assert search_service.refresh_count == 1
    assert search_service.last_owner_ref["model_state_digest"] == "owner-digest-2"
    assert replay_store.append_count == 2
    assert learner.train_calls == 1
    assert service.metadata["compact_owner_search_service_request_count"] == 1
    assert service.metadata["compact_owner_search_service_replay_append_count"] == 2
    assert service.metadata["compact_owner_search_service_learner_update_count"] == 2


def test_owner_search_service_can_defer_owner_ref_digest_to_search_refresh():
    class DeferredDigestSearchService(_FakeOwnerSearchService):
        def refresh_model_owner_ref(
            self,
            *,
            owner_ref: dict[str, Any],
            policy_version_ref: str,
            model_version_ref: str,
            policy_source: str,
            learner_update_count: int,
            expected_model_state_digest: str | None = None,
        ) -> dict[str, Any]:
            assert expected_model_state_digest is None
            assert owner_ref["model_state_digest"] == "refresh-token-2"
            assert (
                owner_ref[COMPACT_OWNER_SEARCH_OWNER_REF_DIGEST_DEFERRED_TO_SEARCH_REFRESH_KEY]
                is True
            )
            assert isinstance(owner_ref["model_state_dict"], dict)
            self.refresh_count += 1
            self.last_owner_ref = dict(owner_ref)
            return {
                "schema_id": "curvyzero_compact_policy_refresh_search_worker_state/v1",
                "learner_update_count": int(learner_update_count),
                "model_state_digest": "actual-search-digest-2",
                "model_state_digest_source": "search_worker_after_load",
                "search_worker_model_object_id": id(self),
                "refresh_count": int(self.refresh_count),
            }

    class DeferredDigestLearner(_FakeOwnerLearner):
        def train_owner_search_step(self, **kwargs: Any) -> dict[str, Any]:
            result = dict(super().train_owner_search_step(**kwargs))
            owner_ref = dict(result["model_owner_ref"])
            owner_ref["model_state_digest"] = "refresh-token-2"
            owner_ref["model_state_dict"] = {"weight": np.asarray([1.0], dtype=np.float32)}
            owner_ref[COMPACT_OWNER_SEARCH_OWNER_REF_DIGEST_DEFERRED_TO_SEARCH_REFRESH_KEY] = True
            owner_ref["model_state_digest_source"] = "deferred_to_search_refresh_after_load"
            result["model_owner_ref"] = owner_ref
            return result

    service = CompactOwnerSearchServiceV1(
        root_provider=_FakeRootProvider(),
        search_service=DeferredDigestSearchService(),
        replay_store=_FakeOwnerReplayStore(),
        learner=DeferredDigestLearner(),
    )

    result = service.run(
        CompactOwnerSearchRequestV1(
            request_id=1,
            actor_step=5,
            root_slot_ids=(10, 11, 12),
            replay_append_entries=({"row": 1},),
            sample_batch_size=1,
            train_steps=2,
            policy_version_ref="policy-v1",
            model_version_ref="model-v1",
            policy_source="unit",
        )
    )
    payload = result.to_dict()

    assert payload["model_owner_ref_digest"] == "refresh-token-2"
    assert payload["search_consumed_learner_update"] is True
    assert payload["search_worker_state"]["model_state_digest"] == "actual-search-digest-2"
    assert (
        payload["search_worker_state"]["owner_ref_model_state_digest_deferred_to_search_refresh"]
        is True
    )
    assert payload["search_worker_state"]["owner_ref_model_state_digest_token"] == "refresh-token-2"


def test_owner_search_service_deferred_owner_ref_digest_requires_model_state_dict():
    class BadDeferredDigestLearner(_FakeOwnerLearner):
        def train_owner_search_step(self, **kwargs: Any) -> dict[str, Any]:
            result = dict(super().train_owner_search_step(**kwargs))
            owner_ref = dict(result["model_owner_ref"])
            owner_ref["model_state_digest"] = "refresh-token-1"
            owner_ref[COMPACT_OWNER_SEARCH_OWNER_REF_DIGEST_DEFERRED_TO_SEARCH_REFRESH_KEY] = True
            result["model_owner_ref"] = owner_ref
            return result

    search_service = _FakeOwnerSearchService()
    service = CompactOwnerSearchServiceV1(
        root_provider=_FakeRootProvider(),
        search_service=search_service,
        replay_store=_FakeOwnerReplayStore(),
        learner=BadDeferredDigestLearner(),
    )

    with pytest.raises(RuntimeError, match="same-process model state"):
        service.run(
            CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=5,
                root_slot_ids=(10, 11, 12),
                replay_append_entries=({"row": 1},),
                sample_batch_size=1,
                train_steps=1,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )
    assert search_service.refresh_count == 0


def test_owner_search_service_can_defer_replay_train_and_refresh_until_drain():
    root_provider = _FakeRootProvider()
    search_service = _FakeOwnerSearchService()
    replay_store = _FakeOwnerReplayStore()
    learner = _FakeOwnerLearner()
    service = CompactOwnerSearchServiceV1(
        root_provider=root_provider,
        search_service=search_service,
        replay_store=replay_store,
        learner=learner,
    )

    action_result = service.run_action(
        CompactOwnerSearchRequestV1(
            request_id=1,
            actor_step=5,
            root_slot_ids=(10, 11, 12),
            replay_append_entries=({"row": 1}, {"row": 2}),
            sample_batch_size=2,
            train_steps=2,
            policy_version_ref="policy-v1",
            model_version_ref="model-v1",
            policy_source="unit",
        )
    )

    assert action_result.owner_maintenance_deferred is True
    assert action_result.owner_maintenance_staged_work_count == 1
    assert action_result.replay_append_count == 0
    assert action_result.learner_update_count == 0
    assert replay_store.append_count == 0
    assert learner.train_calls == 0
    assert search_service.refresh_count == 0
    assert service.metadata["compact_owner_search_service_pending_maintenance_count"] == 1

    drain_result = service.drain_maintenance(CompactOwnerMaintenanceDrainRequestV1(drain_id=1))

    assert drain_result.drained_count == 1
    assert drain_result.drained_work_item_count == 1
    assert drain_result.drained_replay_append_entry_count == 2
    assert drain_result.drained_replay_append_count == 2
    assert drain_result.pending_count == 0
    assert drain_result.replay_append_entry_count == 2
    assert drain_result.replay_append_count == 2
    assert drain_result.learner_update_count == 2
    assert drain_result.model_owner_ref_returned is True
    assert drain_result.model_owner_ref_digest == "owner-digest-2"
    assert drain_result.search_consumed_learner_update is True
    assert drain_result.search_refresh_update_count == 2
    assert replay_store.append_count == 2
    assert learner.train_calls == 1
    assert search_service.refresh_count == 1
    assert service.metadata["compact_owner_search_service_pending_maintenance_count"] == 0


def test_owner_search_service_routes_transition_batch_to_direct_replay_store():
    root_provider = _FakeRootProvider()
    search_service = _FakeOwnerSearchService()
    replay_store = _FakeDirectTransitionBatchOwnerReplayStore()
    service = CompactOwnerSearchServiceV1(
        root_provider=root_provider,
        search_service=search_service,
        replay_store=replay_store,
        learner=_FakeOwnerLearner(),
    )
    transition_batch = _make_unit_transition_batch()

    result = service.run(
        CompactOwnerSearchRequestV1(
            request_id=2,
            actor_step=5,
            root_slot_ids=(10, 11, 12),
            replay_append_entries=(transition_batch,),
            policy_version_ref="policy-v1",
            model_version_ref="model-v1",
            policy_source="unit",
        )
    )

    assert replay_store.direct_append_call_count == 1
    assert replay_store.last_transition_batches == (transition_batch,)
    assert isinstance(
        replay_store.last_transition_batches[0],
        CompactOwnerSearchReplayAppendTransitionBatchV1,
    )
    assert replay_store.legacy_append_call_count == 0
    assert replay_store.append_count == 2
    assert result.replay_append_count == 2
    assert result.replay_append_entry_count == 2
    assert result.replay_append_transport_entry_count == 1
    assert result.replay_append_transition_batch_count == 1
    assert result.replay_append_transition_batch_entry_count == 2
    assert (
        result.owner_sample_telemetry["compact_owner_search_direct_transition_batch_replay_used"]
        is True
    )
    assert (
        result.owner_sample_telemetry[
            "compact_owner_search_direct_transition_batch_replay_batch_count"
        ]
        == 1
    )
    assert (
        result.owner_sample_telemetry[
            "compact_owner_search_direct_transition_batch_replay_transition_count"
        ]
        == 2
    )
    metadata = dict(replay_store.last_transition_batches[0].metadata)
    assert metadata["compact_owner_search_owner_replay_transition_batch_count"] == 1
    assert metadata["compact_owner_search_owner_replay_transition_batch_transition_count"] == 2
    assert metadata["compact_owner_search_owner_replay_transition_legacy_entry_count"] == 0
    assert service.metadata["compact_owner_search_service_replay_append_count"] == 2
    assert (
        service.metadata["compact_owner_search_direct_transition_batch_replay_transition_count"]
        == 2
    )


def test_owner_search_service_drain_projects_direct_transition_batch_replay_metadata():
    root_provider = _FakeRootProvider()
    search_service = _FakeOwnerSearchService()
    replay_store = _FakeDirectTransitionBatchOwnerReplayStore()
    service = CompactOwnerSearchServiceV1(
        root_provider=root_provider,
        search_service=search_service,
        replay_store=replay_store,
        learner=_FakeOwnerLearner(),
    )
    transition_batch = _make_unit_transition_batch()
    for actor_step in (0, 1, 2, 6):
        service.run_action(
            CompactOwnerSearchRequestV1(
                request_id=30 + actor_step,
                actor_step=actor_step,
                root_slot_ids=(10, 11, 12),
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )

    action_result = service.run_action(
        CompactOwnerSearchRequestV1(
            request_id=3,
            actor_step=5,
            root_slot_ids=(10, 11, 12),
            replay_append_entries=(transition_batch,),
            sample_batch_size=2,
            train_steps=2,
            policy_version_ref="policy-v1",
            model_version_ref="model-v1",
            policy_source="unit",
        )
    )

    assert action_result.owner_maintenance_deferred is True
    assert replay_store.direct_append_call_count == 0
    metadata = service.metadata
    assert metadata["compact_owner_search_owner_maintenance_root_cache_snapshot_count"] == 1
    assert (
        metadata["compact_owner_search_owner_maintenance_root_cache_snapshot_full_fallback_count"]
        == 0
    )
    drain_result = service.drain_maintenance(CompactOwnerMaintenanceDrainRequestV1(drain_id=1))

    assert replay_store.direct_append_call_count == 1
    assert replay_store.last_root_batch_cache_keys == (0, 1, 2, 5)
    assert drain_result.drained_replay_append_count == 2
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_owner_maintenance_root_cache_snapshot_retained_entry_count"
        ]
        == 4
    )
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_direct_transition_batch_replay_used"
        ]
        is True
    )
    assert (
        drain_result.owner_sample_telemetry["compact_rollout_slab_sample_gate_sample_row_count"]
        == 2
    )
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_direct_transition_batch_replay_batch_count"
        ]
        == 1
    )
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_direct_transition_batch_replay_transition_count"
        ]
        == 2
    )


def test_owner_search_service_accepts_injected_async_learner_worker():
    root_provider = _FakeRootProvider()
    search_service = _FakeOwnerSearchService()
    replay_store = _FakeOwnerReplayStore()
    learner = _FakeOwnerLearner()
    worker = _FakeExternalOwnerLearnerWorker()
    service = CompactOwnerSearchServiceV1(
        root_provider=root_provider,
        search_service=search_service,
        replay_store=replay_store,
        learner=learner,
        async_learner_worker=True,
        async_learner_worker_adapter=worker,
    )

    action_result = service.run_action(
        CompactOwnerSearchRequestV1(
            request_id=7,
            actor_step=5,
            root_slot_ids=(10, 11, 12),
            replay_append_entries=({"row": 1}, {"row": 2}),
            sample_batch_size=2,
            train_steps=2,
            policy_version_ref="policy-v1",
            model_version_ref="model-v1",
            policy_source="unit",
        )
    )

    assert action_result.owner_maintenance_deferred is True
    assert replay_store.append_count == 0
    assert learner.train_calls == 0
    drain_result = service.drain_maintenance(CompactOwnerMaintenanceDrainRequestV1(drain_id=1))

    assert drain_result.drained_count == 1
    assert drain_result.drained_replay_append_count == 2
    assert drain_result.learner_update_count == 2
    assert drain_result.model_owner_ref_digest == "external-owner-digest-2"
    assert drain_result.search_consumed_learner_update is True
    assert search_service.refresh_count == 1
    assert replay_store.append_count == 2
    assert learner.train_calls == 0
    assert len(worker.requests) == 1
    async_telemetry = drain_result.owner_async_learner_telemetry or {}
    assert (
        async_telemetry["compact_owner_search_owner_async_learner_worker_kind"] == "external_fake"
    )
    assert (
        async_telemetry["compact_owner_search_owner_async_learner_resource_distinct_from_owner"]
        is True
    )
    assert async_telemetry["compact_owner_search_owner_async_learner_submit_count"] == 1
    assert async_telemetry["compact_owner_search_owner_async_learner_completed_count"] == 1
    assert async_telemetry["compact_owner_search_owner_async_learner_pending_count"] == 0
    service.close()
    assert worker.closed is True


def test_owner_search_service_process_async_learner_worker_uses_child_process():
    root_provider = _FakeRootProvider()
    search_service = _FakeOwnerSearchService()
    replay_store = _FakeOwnerReplayStore()
    learner = _FakeProcessOwnerBatchLearner()
    worker = CompactProcessOwnerSearchLearnerWorkerV1(
        learner_factory=_fake_process_owner_batch_learner_factory,
    )
    service = CompactOwnerSearchServiceV1(
        root_provider=root_provider,
        search_service=search_service,
        replay_store=replay_store,
        learner=learner,
        async_learner_worker=True,
        async_learner_worker_kind=(
            COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH
        ),
        async_learner_worker_adapter=worker,
    )

    action_result = service.run_action(
        CompactOwnerSearchRequestV1(
            request_id=8,
            actor_step=5,
            root_slot_ids=(10, 11, 12),
            replay_append_entries=({"row": 1}, {"row": 2}),
            sample_batch_size=2,
            train_steps=2,
            policy_version_ref="policy-v1",
            model_version_ref="model-v1",
            policy_source="unit",
        )
    )

    assert action_result.owner_maintenance_deferred is True
    assert replay_store.append_count == 0
    first_drain = service.drain_maintenance(CompactOwnerMaintenanceDrainRequestV1(drain_id=1))

    assert learner.prepare_calls == 1
    assert replay_store.append_count == 2
    assert first_drain.pending_count == 1
    drain_result = service.drain_maintenance(CompactOwnerMaintenanceDrainRequestV1(drain_id=2))

    assert drain_result.learner_update_count == 2
    assert drain_result.model_owner_ref_digest == "process-owner-digest-2"
    assert drain_result.search_consumed_learner_update is True
    assert search_service.refresh_count == 1
    assert drain_result.owner_sample_telemetry["unit_owner_search_process_payload_prepared"] is True
    telemetry = drain_result.owner_learner_telemetry
    assert telemetry["unit_owner_search_process_learner_pid"] != os.getpid()
    assert (
        telemetry["compact_owner_search_owner_async_learner_worker_pid_distinct_from_owner"] is True
    )
    assert telemetry["compact_owner_search_owner_async_learner_request_host_only"] is True
    assert telemetry["compact_owner_search_owner_async_learner_request_cuda_tensor_count"] == 0
    assert telemetry["compact_owner_search_owner_async_learner_result_host_only"] is True
    assert telemetry["compact_owner_search_owner_async_learner_result_cuda_tensor_count"] == 0
    assert telemetry["compact_owner_search_owner_async_learner_request_bytes"] > 0
    assert telemetry["compact_owner_search_owner_async_learner_result_bytes"] > 0
    async_telemetry = drain_result.owner_async_learner_telemetry or {}
    assert (
        async_telemetry["compact_owner_search_owner_async_learner_worker_kind"]
        == COMPACT_OWNER_SEARCH_ASYNC_LEARNER_WORKER_LOCAL_PROCESS_LEARNER_BATCH
    )
    assert (
        async_telemetry["compact_owner_search_owner_async_learner_worker_resource_scope"]
        == "process"
    )
    assert (
        async_telemetry["compact_owner_search_owner_async_learner_resource_distinct_from_owner"]
        is True
    )
    assert async_telemetry["compact_owner_search_owner_async_learner_submit_count"] == 1
    assert async_telemetry["compact_owner_search_owner_async_learner_completed_count"] == 1
    assert async_telemetry["compact_owner_search_owner_async_learner_pending_count"] == 0
    service.close()


def test_process_owner_search_worker_keeps_search_replay_and_model_state_in_worker():
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=_fake_root_provider_factory,
        search_service_factory=_fake_owner_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_fake_owner_learner_factory,
    )
    try:
        handle = worker.submit(
            CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=7,
                root_slot_ids=(20, 21),
                replay_append_entries=({"row": 1},),
                sample_batch_size=1,
                train_steps=1,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )
        payload = worker.result(handle)
    finally:
        worker.close()

    assert payload["schema_id"] == COMPACT_OWNER_SEARCH_RESULT_SCHEMA_ID
    assert payload["owner_kind"] == COMPACT_OWNER_SEARCH_KIND_LOCAL_PROCESS
    assert payload["owner_pid"] != os.getpid()
    assert payload["root_slot_count"] == 2
    assert payload["selected_action"] == (0, 1)
    assert payload["model_state_return_count"] == 0
    assert payload["model_state_bytes"] == 0
    assert payload["model_state_snapshot_return_count"] == 0
    assert payload["root_observation_bytes_sent"] == 0
    assert payload["search_result_payload_bytes"] > 0
    search_payload = payload["search_result_payload"]
    assert search_payload["payload_transport_kind"] == "numpy_ndarray_ipc_v1"
    assert search_payload["payload_json_safe"] is False
    assert isinstance(search_payload["visit_policy"], np.ndarray)
    assert payload["request_cuda_tensor_count"] == 0
    assert payload["result_cuda_tensor_count"] == 0
    assert payload["worker_owns_search_state"] is True
    assert payload["worker_owns_replay_state"] is True
    assert payload["worker_owns_model_state"] is True
    assert payload["model_owner_ref_returned"] is True
    assert payload["model_owner_ref_digest"] == "owner-digest-1"
    assert payload["search_consumed_learner_update"] is True
    assert payload["search_refresh_update_count"] == 1
    assert worker.metadata["compact_owner_search_worker_resource_distinct_from_actor"] is True
    assert worker.metadata["compact_owner_search_worker_owns_search_state"] is True
    assert worker.metadata["compact_owner_search_worker_owns_replay_state"] is True
    assert worker.metadata["compact_owner_search_worker_owns_model_state"] is True


def test_process_owner_search_worker_resolves_roots_from_shared_memory_slots():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((0, 1, 2)),
        capacity=3,
        metadata={"unit_test_shared_root_store": True},
    )
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={"spec": root_store.spec},
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_fake_owner_learner_factory,
    )
    try:
        first = worker.result(
            worker.submit(
                CompactOwnerSearchRequestV1(
                    request_id=1,
                    actor_step=1,
                    root_slot_ids=root_store.last_root_slot_ids,
                    replay_append_entries=({"row": 1},),
                    sample_batch_size=1,
                    train_steps=1,
                    policy_version_ref="policy-v1",
                    model_version_ref="model-v1",
                    policy_source="unit",
                )
            )
        )
        root_store.publish_root_batch(_shared_observation_root_batch((2, 2, 1)))
        second = worker.result(
            worker.submit(
                CompactOwnerSearchRequestV1(
                    request_id=2,
                    actor_step=2,
                    root_slot_ids=root_store.last_root_slot_ids,
                    replay_append_entries=({"row": 2},),
                    sample_batch_size=1,
                    train_steps=1,
                    policy_version_ref="policy-v1",
                    model_version_ref="model-v1",
                    policy_source="unit",
                )
            )
        )
    finally:
        worker.close()
        root_store.close()
        root_store.unlink()

    assert first["selected_action"] == (0, 1, 2)
    assert second["selected_action"] == (2, 2, 1)
    assert second["owner_pid"] != os.getpid()
    assert second["root_observation_bytes_sent"] == 0
    assert second["request_bytes"] < 4096
    assert second["model_state_bytes"] == 0
    assert second["model_state_snapshot_return_count"] == 0
    assert second["worker_owns_search_state"] is True
    assert second["worker_owns_replay_state"] is True
    assert second["worker_owns_model_state"] is True
    assert second["search_consumed_learner_update"] is True
    assert second["search_refresh_update_count"] == 2


def test_owner_search_host_only_guard_inspects_module_parameters_and_buffers():
    class FakeModule:
        def parameters(self):
            return (SimpleNamespace(is_cuda=True),)

        def buffers(self):
            return ()

    class CpuModule:
        def parameters(self):
            return (SimpleNamespace(is_cuda=False),)

        def buffers(self):
            return (SimpleNamespace(is_cuda=False),)

    assert _contains_cuda_tensor({"model": FakeModule()}) is True
    assert _contains_cuda_tensor({"model": CpuModule()}) is False


def test_process_owner_search_worker_rejects_cuda_replay_append_payload():
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=_fake_root_provider_factory,
        search_service_factory=_fake_owner_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_fake_owner_learner_factory,
    )
    try:
        with pytest.raises(RuntimeError, match="host-only"):
            worker.submit(
                CompactOwnerSearchRequestV1(
                    request_id=1,
                    actor_step=1,
                    root_slot_ids=(0,),
                    replay_append_entries=(SimpleNamespace(is_cuda=True),),
                    sample_batch_size=1,
                    train_steps=1,
                    policy_version_ref="policy-v1",
                    model_version_ref="model-v1",
                    policy_source="unit",
                )
            )
    finally:
        worker.close()


def test_resident_shared_memory_root_provider_creates_owner_resident_batch():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((0, 1, 2)),
        capacity=3,
        metadata={"unit_test_shared_root_store": True},
    )
    provider = build_compact_resident_shared_memory_root_provider_v1(
        spec=root_store.spec,
        device="cpu",
        source_backend="unit-resident-provider",
    )
    try:
        resolved = provider.resolve_root_batch(
            root_slot_ids=root_store.last_root_slot_ids,
            request=CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=0,
                root_slot_ids=root_store.last_root_slot_ids,
            ),
        )
        resident = resolved.resident_observation
        assert resident is not None
        resident_before = resident.root_device_observation.detach().cpu().numpy().copy()
        root_store.publish_root_batch(_shared_observation_root_batch((8, 8, 8)))
        resident_after = resident.root_device_observation.detach().cpu().numpy().copy()
    finally:
        provider.close()
        root_store.close()
        root_store.unlink()

    assert resolved.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    assert np.array_equal(resident_before, resident_after)
    assert resident.host_fallback_allowed is False
    assert resident.source_backend == "unit-resident-provider"
    assert tuple(int(dim) for dim in resident.root_device_observation.shape) == (3, 4, 8, 8)
    assert str(resident.root_device_observation.device) == "cpu"
    assert resolved.metadata["compact_owner_search_resident_root_bridge_ready"] is True
    assert resolved.metadata["owner_search_compact_torch_resident_root_bridge_ready"] is True
    assert resolved.metadata["compact_owner_search_resident_root_bridge_h2d_bytes"] > 0
    assert (
        resolved.metadata["compact_owner_search_resident_root_bridge_host_observation_copied"]
        is False
    )


def test_direct_root_store_resolves_latest_root_batch_without_shared_memory():
    root_batch = _shared_observation_root_batch((2, 1, 0))
    root_store = CompactDirectRootStoreV1(
        capacity=3,
        metadata={"unit_test_direct_root_store": True},
    )
    slots = root_store.publish_root_batch(root_batch)
    resolved = root_store.resolve_root_batch(
        root_slot_ids=slots,
        request=CompactOwnerSearchRequestV1(
            request_id=1,
            actor_step=0,
            root_slot_ids=slots,
        ),
    )

    assert np.array_equal(resolved.observation, root_batch.observation)
    assert resolved.metadata["compact_direct_root_store"] is True
    assert resolved.metadata["compact_direct_root_store_publish_count"] == 1
    assert resolved.metadata["compact_direct_root_store_resolve_count"] == 1
    assert resolved.metadata["compact_owner_search_direct_root_handoff"] is True
    assert resolved.metadata["compact_owner_search_direct_root_rebuild_avoided"] is True
    assert resolved.metadata["compact_owner_search_direct_root_resolved"] is True
    assert resolved.metadata["compact_owner_search_direct_root_observation_bytes_sent"] == 0


def test_root_build_request_builds_resident_stub_root_batch():
    batch = _compact_batch((99, 99, 99), joint_action=(0, 0, 0))
    resident_values = np.asarray(batch.observation, dtype=np.uint8).copy()
    resident_values[:, 0, 0, 0, 0] = np.asarray([0, 1, 2], dtype=np.uint8)
    resident = ResidentObservationBatchV1(
        device_observation=resident_values,
        root_device_observation=resident_values.reshape(3, 4, 8, 8),
        generation_id=31,
        batch_size=3,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype=str(resident_values.dtype),
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=31,
        source_backend="unit_root_build_request",
        host_fallback_allowed=False,
        final_observation_row_mask=np.zeros((3,), dtype=np.bool_),
    )
    resident_batch = SimpleNamespace(
        **{
            **vars(batch),
            "observation_source": COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
            "resident_observation": resident,
        }
    )

    request = compact_root_build_request_v1_from_batch(
        resident_batch,
        search_lane="unit-root-build-request",
        copy_observation=False,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=resident,
        resident_host_observation_stub=True,
    )
    root_batch = build_compact_root_batch_v1_from_request(request)

    assert request.observation is None
    assert root_batch.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    assert root_batch.resident_observation is resident
    assert root_batch.metadata["resident_host_observation_stubbed"] is True
    assert root_batch.metadata["resident_host_observation_stub_materialized_bytes"] == 0
    assert root_batch.metadata["resident_host_observation_stub_logical_bytes"] > 0
    assert not np.any(np.asarray(root_batch.observation) == 99)


def test_direct_root_store_builds_root_request_inside_owner():
    batch = _compact_batch((99, 99, 99), joint_action=(0, 0, 0))
    resident_values = np.asarray(batch.observation, dtype=np.uint8).copy()
    resident_values[:, 0, 0, 0, 0] = np.asarray([0, 1, 2], dtype=np.uint8)
    resident = ResidentObservationBatchV1(
        device_observation=resident_values,
        root_device_observation=resident_values.reshape(3, 4, 8, 8),
        generation_id=32,
        batch_size=3,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype=str(resident_values.dtype),
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=32,
        source_backend="unit_direct_root_store_request",
        host_fallback_allowed=False,
        final_observation_row_mask=np.zeros((3,), dtype=np.bool_),
    )
    resident_batch = SimpleNamespace(
        **{
            **vars(batch),
            "observation_source": COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
            "resident_observation": resident,
        }
    )
    request = compact_root_build_request_v1_from_batch(
        resident_batch,
        search_lane="unit-direct-root-store-request",
        copy_observation=False,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=resident,
        resident_host_observation_stub=True,
    )
    root_store = CompactDirectRootStoreV1(
        capacity=3,
        require_resident_root_view=True,
    )
    slots = root_store.publish_root_build_request(request)
    resolved = root_store.resolve_root_batch(
        root_slot_ids=slots,
        request=CompactOwnerSearchRequestV1(
            request_id=1,
            actor_step=32,
            root_slot_ids=slots,
        ),
    )

    assert resolved.metadata["compact_owner_search_direct_root_build_request_handoff"] is True
    assert resolved.metadata["compact_owner_search_direct_root_parent_build_avoided"] is True
    assert resolved.metadata["compact_owner_search_direct_root_owner_build_used"] is True
    assert resolved.metadata["compact_owner_search_direct_root_owner_build_count"] == 1
    assert (
        resolved.metadata["compact_owner_search_direct_root_build_request_observation_included"]
        is False
    )
    assert (
        resolved.metadata["compact_owner_search_direct_root_build_request_observation_bytes_sent"]
        == 0
    )
    assert resolved.metadata["compact_owner_search_resident_root_view_proved"] is True
    assert root_store.metadata["compact_owner_search_direct_root_owner_build_count"] == 1


def test_direct_root_store_proves_resident_root_view_without_host_fallback():
    root_batch = _resident_direct_root_batch(
        host_actions=(0, 0, 0),
        resident_actions=(2, 1, 0),
        generation_id=11,
    )
    root_store = CompactDirectRootStoreV1(
        capacity=3,
        metadata={"unit_test_direct_root_store": True},
        require_resident_root_view=True,
    )
    slots = root_store.publish_root_batch(root_batch)
    resolved = root_store.resolve_root_batch(
        root_slot_ids=slots,
        request=CompactOwnerSearchRequestV1(
            request_id=1,
            actor_step=0,
            root_slot_ids=slots,
        ),
    )

    assert resolved.metadata["compact_owner_search_resident_root_view_required"] is True
    assert resolved.metadata["compact_owner_search_resident_root_view_proved"] is True
    assert resolved.metadata["compact_owner_search_resident_root_view_kind"] == (
        "direct_root_batch_resident_handle_v1"
    )
    assert resolved.metadata["compact_owner_search_resident_root_view_generation_id"] == 11
    assert resolved.metadata["compact_owner_search_resident_root_view_h2d_bytes"] == 0.0
    assert resolved.metadata["compact_owner_search_resident_root_view_d2h_bytes"] == 0.0
    assert (
        resolved.metadata["compact_owner_search_resident_root_view_host_fallback_allowed"] is False
    )
    assert resolved.metadata["compact_owner_search_resident_root_view_row_major_order"] is True
    assert resolved.metadata["compact_owner_search_resident_root_view_root_shape"] == [
        3,
        4,
        8,
        8,
    ]


def test_threaded_owner_search_direct_root_view_consumes_resident_not_host_observation():
    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_fake_resident_observation_search_service_factory,
        owner_replay_append_enabled=False,
        owner_defer_maintenance=False,
        root_store_capacity=3,
        require_resident_root_view=True,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    try:
        action_step = proxy.run_action_step(
            _resident_direct_root_batch(
                host_actions=(0, 0, 0),
                resident_actions=(2, 1, 0),
                generation_id=13,
            )
        )
    finally:
        proxy.close()

    assert tuple(int(value) for value in action_step.selected_action) == (2, 1, 0)
    assert action_step.metadata["compact_owner_search_threaded_slab_proxy"] is True
    assert action_step.metadata["compact_owner_search_resident_root_view_required"] is True
    assert action_step.metadata["compact_owner_search_resident_root_view_proved"] is True
    assert action_step.metadata["compact_owner_search_resident_root_view_kind"] == (
        "direct_root_batch_resident_handle_v1"
    )
    assert action_step.metadata["compact_owner_search_resident_root_view_generation_id"] == 13
    assert action_step.metadata["compact_owner_search_resident_root_view_h2d_bytes"] == 0.0
    assert action_step.metadata["compact_owner_search_resident_root_view_d2h_bytes"] == 0.0
    assert (
        action_step.metadata["compact_owner_search_resident_root_view_host_fallback_allowed"]
        is False
    )


def test_shared_memory_owner_proxy_rejects_resident_root_view_requirement():
    proxy = CompactLazyOwnerSearchSlabProxyV1(
        search_service_factory=_FakeSharedObservationSearchService,
        owner_replay_append_enabled=False,
        owner_defer_maintenance=False,
        root_store_capacity=3,
        require_resident_root_view=True,
    )
    with pytest.raises(ReplayCompatibilityError, match="direct-root owner-search proxy"):
        proxy.run_action_step(_shared_observation_root_batch((2, 1, 0)))


def test_lazy_inline_owner_search_proxy_uses_direct_root_handoff():
    proxy = CompactLazyInlineOwnerSearchSlabProxyV1(
        search_service_factory=_FakeSharedObservationSearchService,
        owner_replay_append_enabled=False,
        owner_defer_maintenance=False,
        root_store_capacity=3,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    try:
        action_step = proxy.run_action_step(_shared_observation_root_batch((2, 1, 0)))
    finally:
        proxy.close()

    assert tuple(int(value) for value in action_step.selected_action) == (2, 1, 0)
    assert action_step.metadata["compact_owner_search_inline_slab_proxy"] is True
    assert action_step.metadata["compact_owner_search_boundary_kind"] == (
        "inline_owner_search_parent_slab_commit"
    )
    assert action_step.metadata["compact_owner_search_owner_kind"] == (
        COMPACT_OWNER_SEARCH_KIND_INLINE
    )
    assert action_step.metadata["compact_direct_root_store"] is True
    assert action_step.metadata["compact_direct_root_store_publish_count"] == 1
    assert action_step.metadata["compact_direct_root_store_resolve_count"] == 1
    assert action_step.metadata["compact_owner_search_direct_root_handoff"] is True
    assert action_step.metadata["compact_owner_search_direct_root_rebuild_avoided"] is True
    assert action_step.metadata["compact_owner_search_direct_root_resolved"] is True


def test_lazy_threaded_owner_search_proxy_uses_direct_root_and_background_loop():
    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_FakeSharedObservationSearchService,
        owner_replay_append_enabled=False,
        owner_defer_maintenance=False,
        root_store_capacity=3,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    try:
        action_step = proxy.run_action_step(_shared_observation_root_batch((2, 1, 0)))
    finally:
        proxy.close()

    assert tuple(int(value) for value in action_step.selected_action) == (2, 1, 0)
    assert action_step.metadata["compact_owner_search_threaded_slab_proxy"] is True
    assert action_step.metadata["compact_owner_search_boundary_kind"] == (
        "threaded_owner_search_parent_slab_commit"
    )
    assert action_step.metadata["compact_owner_search_owner_kind"] == (
        COMPACT_OWNER_SEARCH_KIND_THREADED
    )
    assert action_step.metadata["compact_owner_search_owner_loop_kind"] == (
        "threaded_priority_owner_loop_v1"
    )
    assert action_step.metadata["compact_owner_search_owner_background_maintenance_thread"] is True
    assert action_step.metadata["compact_owner_search_owner_background_overlap_enabled"] is True
    assert action_step.metadata["compact_direct_root_store"] is True
    assert action_step.metadata["compact_direct_root_store_publish_count"] == 1
    assert action_step.metadata["compact_direct_root_store_resolve_count"] == 1
    assert action_step.metadata["compact_owner_search_direct_root_handoff"] is True
    assert action_step.metadata["compact_owner_search_direct_root_rebuild_avoided"] is True
    assert action_step.metadata["compact_owner_search_direct_root_resolved"] is True


def test_lazy_inline_background_owner_search_proxy_uses_direct_root_and_background_loop():
    proxy = CompactLazyInlineBackgroundOwnerSearchSlabProxyV1(
        search_service_factory=_FakeSharedObservationSearchService,
        owner_replay_append_enabled=False,
        owner_defer_maintenance=False,
        root_store_capacity=3,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    try:
        action_step = proxy.run_action_step(_shared_observation_root_batch((2, 1, 0)))
    finally:
        proxy.close()

    assert tuple(int(value) for value in action_step.selected_action) == (2, 1, 0)
    assert action_step.metadata["compact_owner_search_inline_background_slab_proxy"] is True
    assert action_step.metadata["compact_owner_search_boundary_kind"] == (
        "inline_background_owner_search_parent_slab_commit"
    )
    assert action_step.metadata["compact_owner_search_owner_kind"] == (
        COMPACT_OWNER_SEARCH_KIND_INLINE_BACKGROUND
    )
    assert action_step.metadata["compact_owner_search_owner_loop_kind"] == (
        "inline_background_maintenance_owner_loop_v1"
    )
    assert action_step.metadata["compact_owner_search_owner_background_maintenance_thread"] is True
    assert action_step.metadata["compact_owner_search_owner_background_overlap_enabled"] is True
    assert action_step.metadata["compact_direct_root_store"] is True
    assert action_step.metadata["compact_direct_root_store_publish_count"] == 1
    assert action_step.metadata["compact_direct_root_store_resolve_count"] == 1
    assert action_step.metadata["compact_owner_search_direct_root_handoff"] is True
    assert action_step.metadata["compact_owner_search_direct_root_rebuild_avoided"] is True
    assert action_step.metadata["compact_owner_search_direct_root_resolved"] is True


def test_resident_shared_memory_root_provider_uses_sparse_terminal_final_rows():
    root_batch = _shared_observation_root_batch((0, 1, 2))
    root_batch = replace(
        root_batch,
        done_root=np.asarray((False, True, False), dtype=np.bool_),
        final_observation_row_mask=np.asarray((False, True, False), dtype=np.bool_),
        terminal_row_mask=np.asarray((False, True, False), dtype=np.bool_),
    )
    root_store = CompactSharedMemoryRootStoreV1.create(
        root_batch,
        capacity=3,
        metadata={"unit_test_shared_root_store": True},
    )
    provider = build_compact_resident_shared_memory_root_provider_v1(
        spec=root_store.spec,
        device="cpu",
        source_backend="unit-resident-provider",
    )
    try:
        resolved = provider.resolve_root_batch(
            root_slot_ids=root_store.last_root_slot_ids,
            request=CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=0,
                root_slot_ids=root_store.last_root_slot_ids,
            ),
        )
    finally:
        provider.close()
        root_store.close()
        root_store.unlink()

    resident = resolved.resident_observation
    assert resident is not None
    validate_resident_observation_batch_v1(
        resident,
        batch_size=3,
        player_count=1,
        stack_shape=(4, 8, 8),
    )
    assert resident.final_device_observation is None
    assert resident.root_final_device_observation is None
    assert resident.final_device_observation_rows is not None
    assert tuple(int(dim) for dim in resident.final_device_observation_rows.shape) == (
        1,
        1,
        4,
        8,
        8,
    )
    assert tuple(int(value) for value in resident.final_device_observation_row_indices) == (1,)
    sparse_host = resident.final_device_observation_rows.detach().cpu().numpy()
    assert int(sparse_host[0, 0, 0, 0, 0]) == 1
    assert resident.metadata["resident_final_device_observation_storage"] == "sparse_rows"
    assert (
        resolved.metadata["compact_owner_search_resident_root_bridge_final_storage"]
        == "sparse_rows"
    )
    assert (
        resolved.metadata[
            "compact_owner_search_resident_root_bridge_final_dense_clone_avoided_bytes"
        ]
        > 0
    )


def test_process_owner_search_worker_resolves_resident_roots_from_shared_memory_slots():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((2, 1, 0)),
        capacity=3,
        metadata={"unit_test_shared_root_store": True},
    )
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_resident_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={
            "spec": root_store.spec,
            "device": "cpu",
            "source_backend": "unit-resident-process-provider",
        },
        search_service_factory=_fake_resident_observation_search_service_factory,
    )
    try:
        payload = worker.result(
            worker.submit(
                CompactOwnerSearchRequestV1(
                    request_id=1,
                    actor_step=1,
                    root_slot_ids=root_store.last_root_slot_ids,
                    policy_version_ref="policy-v1",
                    model_version_ref="model-v1",
                    policy_source="unit",
                )
            )
        )
    finally:
        worker.close()
        root_store.close()
        root_store.unlink()

    assert payload["selected_action"] == (2, 1, 0)
    metadata = payload["search_result_payload"]["metadata"]
    assert metadata["resident_owner_search_observation_source"] == (
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    )
    assert metadata["resident_owner_search_host_fallback_allowed"] is False
    assert metadata["compact_owner_search_resident_root_bridge_ready"] is True
    assert (
        payload["search_result_metadata"]["compact_owner_search_resident_root_bridge_ready"] is True
    )
    assert (
        payload["search_result_metadata"][
            "compact_owner_search_resident_root_bridge_host_observation_copied"
        ]
        is False
    )
    assert payload["root_observation_bytes_sent"] == 0
    assert payload["request_cuda_tensor_count"] == 0
    assert payload["result_cuda_tensor_count"] == 0


def test_owner_transition_materialization_keeps_resident_replay_rows_device_owned():
    first_batch = _shared_observation_root_batch((2, 1, 0))
    root_store = CompactSharedMemoryRootStoreV1.create(
        first_batch,
        capacity=3,
        metadata={"unit_test_shared_root_store": True},
    )
    provider = build_compact_resident_shared_memory_root_provider_v1(
        spec=root_store.spec,
        device="cpu",
        source_backend="unit-resident-owner-materialization",
    )
    replay_store = _FakeOwnerReplayStore()
    service = CompactOwnerSearchServiceV1(
        root_provider=provider,
        search_service=_FakeResidentObservationSearchService(),
        replay_store=replay_store,
        learner=_FakeOwnerLearner(),
    )
    try:
        first = service.run_action(
            CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=0,
                root_slot_ids=root_store.last_root_slot_ids,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )
        assert first.search_result_metadata is not None
        assert (
            first.search_result_metadata["compact_owner_search_resident_root_bridge_ready"] is True
        )
        assert (
            first.search_result_metadata[
                "compact_owner_search_resident_root_bridge_host_observation_copied"
            ]
            is False
        )
        next_batch = _shared_observation_root_batch((1, 2, 0))
        next_slots = root_store.publish_root_batch(next_batch)
        selected = np.asarray(first.selected_action, dtype=np.int16).reshape(-1, 1)
        transition = CompactOwnerSearchReplayAppendTransitionEntryV1(
            schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID,
            record_index=0,
            next_record_index=1,
            replay_payload_handle=first.replay_payload_handle,
            next_joint_action=selected,
            next_reward=np.asarray(next_batch.target_reward, dtype=np.float32).reshape(3, 1),
            next_done=np.asarray(next_batch.done_root, dtype=np.bool_).reshape(3, 1).any(axis=1),
            next_terminated=np.asarray(next_batch.done_root, dtype=np.bool_)
            .reshape(3, 1)
            .any(axis=1),
            next_truncated=np.zeros((3,), dtype=np.bool_),
            next_final_reward_map=np.asarray(next_batch.target_reward, dtype=np.float32).reshape(
                3, 1
            ),
            next_final_observation_row_mask=np.asarray(
                next_batch.final_observation_row_mask,
                dtype=np.bool_,
            )
            .reshape(3, 1)
            .any(axis=1),
            policy_source="unit",
            metadata={
                "compact_owner_search_replay_append_entry": True,
                "compact_owner_search_replay_append_transition_only": True,
            },
        )
        service.run_action(
            CompactOwnerSearchRequestV1(
                request_id=2,
                actor_step=1,
                root_slot_ids=next_slots,
                replay_append_entries=(transition,),
                sample_batch_size=1,
                train_steps=1,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )
        service.drain_maintenance(
            CompactOwnerMaintenanceDrainRequestV1(drain_id=1, fail_if_empty=True)
        )
    finally:
        provider.close()
        root_store.close()
        root_store.unlink()

    assert replay_store.last_entries
    entry = replay_store.last_entries[0]
    assert isinstance(entry, CompactOwnerSearchReplayAppendIndexEntryV1)
    assert isinstance(entry.index_rows, CompactDeviceReplayIndexRowsV1)
    assert entry.index_rows.metadata["device_replay_index_rows"] is True
    assert (
        entry.index_rows.metadata["compact_owner_search_owner_materialized_device_replay_rows"]
        is True
    )
    assert (
        entry.index_rows.metadata[
            "compact_owner_search_owner_materialized_replay_resident_observation"
        ]
        is True
    )
    assert (
        entry.index_rows.metadata[
            "compact_owner_search_owner_materialized_replay_host_observation_copy"
        ]
        is False
    )


def test_owner_action_uses_inner_two_phase_device_replay_for_resident_roots():
    first_batch = _shared_observation_root_batch((2, 1, 0))
    root_store = CompactSharedMemoryRootStoreV1.create(
        first_batch,
        capacity=3,
        metadata={"unit_test_shared_root_store": True},
    )
    provider = build_compact_resident_shared_memory_root_provider_v1(
        spec=root_store.spec,
        device="cpu",
        source_backend="unit-resident-owner-two-phase",
    )
    search_service = _FakeTwoPhaseResidentObservationSearchService()
    replay_store = _FakeOwnerReplayStore()
    service = CompactOwnerSearchServiceV1(
        root_provider=provider,
        search_service=search_service,
        replay_store=replay_store,
        learner=_FakeOwnerLearner(),
        use_inner_two_phase_device_replay=True,
    )
    try:
        first = service.run_action(
            CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=0,
                root_slot_ids=root_store.last_root_slot_ids,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )
        next_batch = _shared_observation_root_batch((1, 2, 0))
        next_slots = root_store.publish_root_batch(next_batch)
        selected = np.asarray(first.selected_action, dtype=np.int16).reshape(-1, 1)
        transition = CompactOwnerSearchReplayAppendTransitionEntryV1(
            schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID,
            record_index=0,
            next_record_index=1,
            replay_payload_handle=first.replay_payload_handle,
            next_joint_action=selected,
            next_reward=np.asarray(next_batch.target_reward, dtype=np.float32).reshape(3, 1),
            next_done=np.asarray(next_batch.done_root, dtype=np.bool_).reshape(3, 1).any(axis=1),
            next_terminated=np.asarray(next_batch.done_root, dtype=np.bool_)
            .reshape(3, 1)
            .any(axis=1),
            next_truncated=np.zeros((3,), dtype=np.bool_),
            next_final_reward_map=np.asarray(next_batch.target_reward, dtype=np.float32).reshape(
                3, 1
            ),
            next_final_observation_row_mask=np.asarray(
                next_batch.final_observation_row_mask,
                dtype=np.bool_,
            )
            .reshape(3, 1)
            .any(axis=1),
            policy_source="unit",
            metadata={
                "compact_owner_search_replay_append_entry": True,
                "compact_owner_search_replay_append_transition_only": True,
            },
        )
        service.run_action(
            CompactOwnerSearchRequestV1(
                request_id=2,
                actor_step=1,
                root_slot_ids=next_slots,
                replay_append_entries=(transition,),
                sample_batch_size=1,
                train_steps=1,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )
        drain_result = service.drain_maintenance(
            CompactOwnerMaintenanceDrainRequestV1(drain_id=1, fail_if_empty=True)
        )
    finally:
        provider.close()
        root_store.close()
        root_store.unlink()

    assert search_service.run_count == 0
    assert search_service.action_step_count == 2
    assert search_service.device_flush_count == 2
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_inner_device_replay_payload_flushed_count"
        ]
        == 2
    )
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_inner_deferred_one_simulation_replay_payload_flush_count"
        ]
        == 2
    )
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_inner_deferred_one_simulation_replay_materialized_on_flush_count"
        ]
        == 2
    )
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_inner_deferred_one_simulation_replay_recurrent_inference_calls"
        ]
        == 2.0
    )
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_inner_deferred_one_simulation_model_identity_match_count"
        ]
        == 2
    )
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_inner_deferred_one_simulation_model_refresh_crossed_count"
        ]
        == 0
    )
    assert (
        drain_result.owner_sample_telemetry[
            "compact_owner_search_inner_pending_deferred_replay_payload_final_count"
        ]
        == 0
    )
    assert (
        service.metadata["compact_owner_search_inner_pending_deferred_replay_payload_final_count"]
        == 0
    )
    assert replay_store.last_entries
    entry = replay_store.last_entries[0]
    assert isinstance(entry.index_rows, CompactDeviceReplayIndexRowsV1)
    assert (
        entry.index_rows.metadata[
            "compact_owner_search_owner_materialized_inner_device_replay_payload"
        ]
        is True
    )
    assert entry.index_rows.metadata["compact_owner_search_inner_two_phase_action_step"] is True


def test_owner_inner_two_phase_deferred_replay_fails_closed_on_identity_cross():
    first_batch = _shared_observation_root_batch((2, 1, 0))
    root_store = CompactSharedMemoryRootStoreV1.create(
        first_batch,
        capacity=3,
        metadata={"unit_test_shared_root_store": True},
    )
    provider = build_compact_resident_shared_memory_root_provider_v1(
        spec=root_store.spec,
        device="cpu",
        source_backend="unit-resident-owner-two-phase-crossed",
    )
    search_service = _FakeTwoPhaseResidentObservationSearchService()
    search_service.deferred_payload_identity_match = False
    search_service.deferred_payload_model_refresh_crossed_count = 1
    replay_store = _FakeOwnerReplayStore()
    service = CompactOwnerSearchServiceV1(
        root_provider=provider,
        search_service=search_service,
        replay_store=replay_store,
        learner=_FakeOwnerLearner(),
        use_inner_two_phase_device_replay=True,
    )
    try:
        first = service.run_action(
            CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=0,
                root_slot_ids=root_store.last_root_slot_ids,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )
        next_batch = _shared_observation_root_batch((1, 2, 0))
        next_slots = root_store.publish_root_batch(next_batch)
        selected = np.asarray(first.selected_action, dtype=np.int16).reshape(-1, 1)
        transition = CompactOwnerSearchReplayAppendTransitionEntryV1(
            schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID,
            record_index=0,
            next_record_index=1,
            replay_payload_handle=first.replay_payload_handle,
            next_joint_action=selected,
            next_reward=np.asarray(next_batch.target_reward, dtype=np.float32).reshape(3, 1),
            next_done=np.asarray(next_batch.done_root, dtype=np.bool_).reshape(3, 1).any(axis=1),
            next_terminated=np.asarray(next_batch.done_root, dtype=np.bool_)
            .reshape(3, 1)
            .any(axis=1),
            next_truncated=np.zeros((3,), dtype=np.bool_),
            next_final_reward_map=np.asarray(next_batch.target_reward, dtype=np.float32).reshape(
                3, 1
            ),
            next_final_observation_row_mask=np.asarray(
                next_batch.final_observation_row_mask,
                dtype=np.bool_,
            )
            .reshape(3, 1)
            .any(axis=1),
            policy_source="unit",
            metadata={
                "compact_owner_search_replay_append_entry": True,
                "compact_owner_search_replay_append_transition_only": True,
            },
        )
        service.run_action(
            CompactOwnerSearchRequestV1(
                request_id=2,
                actor_step=1,
                root_slot_ids=next_slots,
                replay_append_entries=(transition,),
                sample_batch_size=1,
                train_steps=1,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )
        with pytest.raises(RuntimeError, match="model refresh"):
            service.drain_maintenance(
                CompactOwnerMaintenanceDrainRequestV1(drain_id=1, fail_if_empty=True)
            )
    finally:
        provider.close()
        root_store.close()
        root_store.unlink()

    assert search_service.device_flush_count == 1
    assert replay_store.append_count == 0
    assert service.metadata["compact_owner_search_service_maintenance_failed"] is True


def test_owner_action_does_not_use_inner_two_phase_by_default():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((2, 1, 0)),
        capacity=3,
        metadata={"unit_test_shared_root_store": True},
    )
    provider = build_compact_resident_shared_memory_root_provider_v1(
        spec=root_store.spec,
        device="cpu",
        source_backend="unit-resident-owner-two-phase-default-off",
    )
    search_service = _FakeTwoPhaseResidentObservationSearchService()
    service = CompactOwnerSearchServiceV1(
        root_provider=provider,
        search_service=search_service,
        replay_store=_FakeOwnerReplayStore(),
        learner=_FakeOwnerLearner(),
    )
    try:
        result = service.run_action(
            CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=0,
                root_slot_ids=root_store.last_root_slot_ids,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )
    finally:
        provider.close()
        root_store.close()
        root_store.unlink()

    assert result.inner_two_phase_action_step is False
    assert result.inner_device_replay_payload_deferred is False
    assert search_service.run_count == 1
    assert search_service.action_step_count == 0
    assert search_service.device_flush_count == 0


def test_owner_search_slab_proxy_drives_actions_and_preserves_replay_commit():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((0, 0, 0)),
        capacity=3,
    )
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={"spec": root_store.spec},
        search_service_factory=_fake_shared_observation_search_service_factory,
    )
    proxy = CompactOwnerSearchSlabProxyV1(
        root_store=root_store,
        worker=worker,
        policy_version_ref="policy-v1",
        model_version_ref="model-v1",
        policy_source="unit",
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search",
        policy_source="unit",
    )
    try:
        first = slab.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = slab.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
    finally:
        slab.close()
        proxy.close()
        root_store.close()
        root_store.unlink()

    assert tuple(int(value) for value in first.next_joint_action[:, 0]) == (0, 1, 2)
    assert first.search_result is None
    assert first.action_step is not None
    metadata = first.action_step.metadata
    assert metadata["compact_owner_search_slab_proxy"] is True
    assert metadata["compact_owner_search_two_phase_action_step"] is True
    assert metadata["search_replay_payload_digest_deferred"] is True
    assert metadata["compact_owner_search_root_observation_bytes_sent"] == 0
    assert metadata["compact_owner_search_request_bytes"] > 0
    assert metadata["compact_owner_search_result_bytes"] > 0
    assert metadata["compact_owner_search_request_cuda_tensor_count"] == 0
    assert metadata["compact_owner_search_result_cuda_tensor_count"] == 0
    assert metadata["compact_owner_search_parent_reconstructed_search_result"] is True
    assert (
        metadata["compact_owner_search_search_result_payload_transport_kind"]
        == "numpy_ndarray_ipc_v1"
    )
    assert metadata["compact_owner_search_search_result_payload_json_safe"] is False
    assert metadata["compact_owner_search_boundary_kind"] == ("worker_search_parent_slab_commit")
    assert metadata["compact_owner_search_parent_slab_commits_replay"] is True
    assert metadata["compact_owner_search_worker_owns_search_state"] is True
    assert metadata["compact_owner_search_worker_owns_replay_state"] is False
    assert metadata["compact_owner_search_model_state_bytes"] == 0
    assert metadata["compact_owner_search_model_state_return_count"] == 0
    assert metadata["compact_owner_search_selected_action_bytes"] > 0
    assert metadata["compact_owner_search_visit_policy_bytes"] > 0
    assert metadata["compact_owner_search_root_value_bytes"] > 0
    assert metadata["compact_owner_search_parent_wait_sec"] >= 0.0
    assert metadata["compact_owner_search_worker_wall_sec"] >= 0.0
    assert metadata["compact_owner_search_worker_search_sec"] >= 0.0
    assert proxy.last_search_result_payload_bytes > 0
    assert second.committed_index_rows is not None
    assert tuple(int(value) for value in np.asarray(second.committed_index_rows.action)) == (
        0,
        1,
        2,
    )
    assert second.committed_index_rows.metadata["search_impl"] == (
        "fake_shared_observation_owner_search"
    )


@pytest.mark.parametrize("owner_sample_batch_size", [1, 0])
def test_owner_search_slab_proxy_sends_committed_replay_and_train_to_owner(
    owner_sample_batch_size: int,
):
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((0, 0, 0)),
        capacity=3,
    )
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={"spec": root_store.spec},
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_fake_owner_learner_factory,
    )
    proxy = CompactOwnerSearchSlabProxyV1(
        root_store=root_store,
        worker=worker,
        policy_version_ref="policy-v1",
        model_version_ref="model-v1",
        policy_source="unit",
        owner_replay_append_enabled=True,
        owner_sample_batch_size=owner_sample_batch_size,
        owner_train_steps=1,
        owner_train_interval=1,
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-train",
        policy_source="unit",
    )
    try:
        first = slab.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = slab.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        proxy_metadata = dict(proxy.metadata)
    finally:
        slab.close()
        proxy.close()
        root_store.close()
        root_store.unlink()

    assert second.action_step is not None
    metadata = second.action_step.metadata
    assert metadata["compact_owner_search_worker_owns_search_state"] is True
    assert metadata["compact_owner_search_worker_owns_replay_state"] is True
    assert metadata["compact_owner_search_worker_owns_model_state"] is True
    assert metadata["compact_owner_search_root_observation_bytes_sent"] == 0
    assert metadata["compact_owner_search_request_cuda_tensor_count"] == 0
    assert metadata["compact_owner_search_result_cuda_tensor_count"] == 0
    assert metadata["compact_owner_search_request_bytes"] < 8192
    assert metadata["compact_owner_search_model_state_bytes"] == 0
    assert metadata["compact_owner_search_model_state_return_count"] == 0
    assert metadata["compact_owner_search_model_state_snapshot_return_count"] == 0
    assert metadata["compact_owner_search_replay_append_entry_count"] == 1
    assert metadata["compact_owner_search_replay_append_count"] == 1
    assert metadata["compact_owner_search_learner_update_count"] == 1
    assert metadata["compact_owner_search_model_owner_ref_returned"] is True
    assert metadata["compact_owner_search_model_owner_ref_digest"] == "owner-digest-1"
    assert metadata["compact_owner_search_consumed_learner_update"] is True
    assert metadata["compact_owner_search_search_refresh_update_count"] == 1
    assert metadata["compact_owner_search_owner_replay_append_enabled"] is True
    assert metadata["compact_owner_search_owner_sample_batch_size"] == owner_sample_batch_size
    assert metadata["compact_owner_search_owner_replay_append_submitted_entry_count"] == 1
    assert metadata["compact_owner_search_owner_replay_append_count"] == 1
    assert metadata["compact_owner_search_owner_train_request_count"] == 1
    assert metadata["compact_owner_search_owner_submitted_learner_update_count"] == 1
    assert metadata["compact_owner_search_owner_learner_update_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_pending_replay_append_entry_count"] == 0
    assert proxy_metadata["compact_owner_search_worker_owns_replay_state"] is True
    assert proxy_metadata["compact_owner_search_worker_owns_model_state"] is True
    assert proxy_metadata["compact_owner_search_model_owner_ref_returned"] is True
    assert proxy_metadata["compact_owner_search_model_owner_ref_digest"] == "owner-digest-1"


def test_owner_search_slab_proxy_learning_gate_suppresses_warmup_replay():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((0, 0, 0)),
        capacity=3,
    )
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={"spec": root_store.spec},
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_fake_owner_learner_factory,
    )
    proxy = CompactOwnerSearchSlabProxyV1(
        root_store=root_store,
        worker=worker,
        policy_version_ref="policy-v1",
        model_version_ref="model-v1",
        policy_source="unit",
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-learning-gate",
        policy_source="unit",
    )
    try:
        proxy.set_owner_learning_enabled(False)
        first = slab.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        disabled = slab.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        disabled_metadata = dict(disabled.action_step.metadata)
        proxy.set_owner_learning_enabled(True)
        enabled = slab.step(
            _compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in disabled.next_joint_action[:, 0]),
            )
        )
        enabled_metadata = dict(enabled.action_step.metadata)
        proxy.drain_owner_maintenance(wait=True)
        proxy_metadata = dict(proxy.metadata)
    finally:
        slab.close()
        proxy.close()
        root_store.close()
        root_store.unlink()

    assert disabled_metadata["compact_owner_search_owner_learning_enabled"] is False
    assert disabled_metadata["compact_owner_search_owner_replay_append_staged_entry_count"] == 0
    assert disabled_metadata["compact_owner_search_owner_replay_append_suppressed_entry_count"] == 1
    assert disabled_metadata["compact_owner_search_owner_train_request_count"] == 0
    assert enabled_metadata["compact_owner_search_owner_learning_enabled"] is True
    assert enabled_metadata["compact_owner_search_owner_replay_append_staged_entry_count"] == 1
    assert enabled_metadata["compact_owner_search_owner_replay_append_submitted_entry_count"] == 1
    assert enabled_metadata["compact_owner_search_owner_train_request_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_replay_append_suppressed_entry_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_replay_append_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_learner_update_count"] == 1


def test_inline_owner_search_slab_proxy_refreshes_model_on_cadence_and_final_train():
    proxy = CompactLazyInlineOwnerSearchSlabProxyV1(
        search_service_factory=_FakeSharedObservationSearchService,
        replay_store_factory=_FakeOwnerReplayStore,
        learner_factory=_FakeOwnerLearner,
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_model_refresh_interval=2,
        owner_expected_train_request_count=3,
        owner_defer_maintenance=False,
        root_store_capacity=3,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-refresh-cadence",
        policy_source="unit",
    )
    try:
        first = slab.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = slab.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        third = slab.step(
            _compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
            )
        )
        slab.step(
            _compact_batch(
                (2, 1, 1),
                joint_action=tuple(int(value) for value in third.next_joint_action[:, 0]),
            )
        )
        proxy_metadata = dict(proxy.metadata)
    finally:
        slab.close()
        proxy.close()

    assert proxy_metadata["compact_owner_search_owner_train_request_count"] == 3
    assert proxy_metadata["compact_owner_search_owner_model_refresh_interval"] == 2
    assert proxy_metadata["compact_owner_search_owner_expected_train_request_count"] == 3
    assert proxy_metadata["compact_owner_search_owner_model_refresh_request_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_model_refresh_skipped_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_submitted_learner_update_count"] == 3
    assert proxy_metadata["compact_owner_search_owner_learner_update_count"] == 3
    assert proxy_metadata["compact_owner_search_search_refresh_update_count"] == 3
    assert proxy_metadata["compact_owner_search_model_owner_ref_returned"] is True
    assert proxy_metadata["compact_owner_search_model_owner_ref_digest"] == "owner-digest-3"
    assert proxy_metadata["compact_owner_search_consumed_learner_update"] is True


def test_owner_search_slab_proxy_can_defer_owner_maintenance_until_drain():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((0, 0, 0)),
        capacity=3,
    )
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={"spec": root_store.spec},
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_fake_owner_learner_factory,
    )
    proxy = CompactOwnerSearchSlabProxyV1(
        root_store=root_store,
        worker=worker,
        policy_version_ref="policy-v1",
        model_version_ref="model-v1",
        policy_source="unit",
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-deferred-train",
        policy_source="unit",
    )
    try:
        first = slab.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = slab.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        action_metadata = dict(second.action_step.metadata)
        proxy_metadata = proxy.drain_owner_maintenance(wait=True)
    finally:
        slab.close()
        proxy.close()
        root_store.close()
        root_store.unlink()

    assert action_metadata["compact_owner_search_owner_defer_maintenance"] is True
    assert second.committed_index_rows is None
    assert action_metadata["compact_owner_search_action_only_result"] is True
    assert action_metadata["compact_owner_search_owner_materializes_replay"] is True
    assert action_metadata["compact_owner_search_parent_slab_commits_replay"] is False
    assert action_metadata["compact_owner_search_parent_reconstructed_search_result"] is False
    assert action_metadata["compact_owner_search_search_result_payload_bytes"] == 0
    assert (
        action_metadata["compact_owner_search_search_result_payload_transport_kind"]
        == "action_only_owner_cached_replay_v1"
    )
    assert action_metadata["compact_owner_search_replay_append_entry_count"] == 1
    assert action_metadata["compact_owner_search_replay_append_count"] == 0
    assert action_metadata["compact_owner_search_learner_update_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_replay_append_submitted_entry_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_replay_append_count"] == 1
    assert proxy_metadata["compact_owner_search_replay_append_entry_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_train_request_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_submitted_learner_update_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_learner_update_count"] == 1
    assert proxy_metadata["compact_owner_search_replay_append_count"] == 1
    assert proxy_metadata["compact_owner_search_learner_update_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_maintenance_drain_request_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_maintenance_staged_work_item_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_maintenance_drained_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_maintenance_drained_work_item_count"] == 1
    assert (
        proxy_metadata["compact_owner_search_owner_maintenance_drained_replay_append_entry_count"]
        == 1
    )
    assert proxy_metadata["compact_owner_search_owner_maintenance_drained_replay_append_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_maintenance_pending_work_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_maintenance_inflight"] is False
    assert proxy_metadata["compact_owner_search_parent_slab_commits_replay"] is False
    assert proxy_metadata["compact_owner_search_owner_policy_lag_current"] == 0
    assert proxy_metadata["compact_owner_search_owner_policy_lag_max"] == 1
    assert proxy_metadata["compact_owner_search_model_owner_ref_returned"] is True
    assert proxy_metadata["compact_owner_search_model_owner_ref_digest"] == "owner-digest-1"
    assert proxy_metadata["compact_owner_search_consumed_learner_update"] is True
    assert proxy_metadata["compact_owner_search_action_feedback_verified"] is True
    assert proxy_metadata["compact_owner_search_action_feedback_transition_count"] == 1
    assert proxy_metadata["compact_owner_search_action_feedback_action_count"] > 0
    assert proxy_metadata["compact_owner_search_action_feedback_mismatch_count"] == 0
    assert proxy_metadata["compact_owner_search_expected_joint_action_checksum"] > 0
    assert (
        proxy_metadata["compact_owner_search_expected_joint_action_checksum"]
        == proxy_metadata["compact_owner_search_applied_joint_action_checksum"]
        == proxy_metadata["compact_owner_search_replay_action_checksum"]
    )
    learner_telemetry = proxy_metadata["compact_owner_search_owner_learner_telemetry"]
    assert learner_telemetry["compact_owner_search_owner_train_wall_sec"] == pytest.approx(0.01)
    assert learner_telemetry["compact_owner_search_owner_train_timing_aggregate_count"] == 1
    assert learner_telemetry["compact_muzero_learner_sec"] == pytest.approx(0.003)
    assert learner_telemetry["compact_muzero_learner_backward_sec"] == pytest.approx(0.0004)


def test_owner_search_deferred_direct_run_fails_closed_for_action_only_result():
    root_batch = _shared_observation_root_batch((0, 1, 2))
    root_store = CompactSharedMemoryRootStoreV1.create(root_batch, capacity=3)
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={"spec": root_store.spec},
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_fake_owner_learner_factory,
    )
    proxy = CompactOwnerSearchSlabProxyV1(
        root_store=root_store,
        worker=worker,
        policy_version_ref="policy-v1",
        model_version_ref="model-v1",
        policy_source="unit",
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
    )
    try:
        with pytest.raises(RuntimeError, match="action-only synthetic search result"):
            proxy.run(root_batch)
    finally:
        proxy.close()
        root_store.close()
        root_store.unlink()


def test_owner_search_deferred_maintenance_coalesces_append_until_train_boundary():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((0, 0, 0)),
        capacity=3,
    )
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={"spec": root_store.spec},
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_fake_owner_learner_factory,
    )
    proxy = CompactOwnerSearchSlabProxyV1(
        root_store=root_store,
        worker=worker,
        policy_version_ref="policy-v1",
        model_version_ref="model-v1",
        policy_source="unit",
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=2,
        owner_defer_maintenance=True,
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-coalesced-maintenance",
        policy_source="unit",
    )
    try:
        first = slab.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = slab.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        second_metadata = dict(second.action_step.metadata)
        third = slab.step(
            _compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
            )
        )
        third_metadata = dict(third.action_step.metadata)
        proxy_metadata = proxy.drain_owner_maintenance(wait=True)
    finally:
        slab.close()
        proxy.close()
        root_store.close()
        root_store.unlink()

    assert second_metadata["compact_owner_search_owner_train_request_count"] == 0
    assert second_metadata["compact_owner_search_owner_maintenance_drain_request_count"] == 0
    assert second_metadata["compact_owner_search_owner_maintenance_coalesced_skip_count"] == 1
    assert second_metadata["compact_owner_search_owner_maintenance_eager_append_drain_count"] == 0
    assert (
        second_metadata["compact_owner_search_owner_maintenance_coalescing_kind"]
        == "eager_append_or_train_boundary_v1"
    )
    assert third_metadata["compact_owner_search_owner_train_request_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_replay_append_submitted_entry_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_replay_append_count"] == 2
    assert proxy_metadata["compact_owner_search_replay_append_entry_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_train_request_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_submitted_learner_update_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_learner_update_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_maintenance_drain_request_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_maintenance_eager_append_drain_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_maintenance_coalesced_skip_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_maintenance_staged_work_item_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_maintenance_drained_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_maintenance_drained_work_item_count"] == 2
    assert (
        proxy_metadata["compact_owner_search_owner_maintenance_drained_replay_append_entry_count"]
        == 2
    )
    assert proxy_metadata["compact_owner_search_owner_maintenance_drained_replay_append_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_maintenance_pending_work_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_policy_lag_current"] == 0
    assert proxy_metadata["compact_owner_search_owner_policy_lag_max"] == 1
    assert proxy_metadata["compact_owner_search_action_feedback_verified"] is True
    assert proxy_metadata["compact_owner_search_action_feedback_transition_count"] == 2
    assert proxy_metadata["compact_owner_search_action_feedback_action_count"] > 0
    assert proxy_metadata["compact_owner_search_action_feedback_mismatch_count"] == 0


def test_owner_search_priority_loop_serves_action_while_maintenance_is_pending():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((0, 0, 0)),
        capacity=3,
    )
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={"spec": root_store.spec},
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_slow_owner_learner_factory,
    )
    proxy = CompactOwnerSearchSlabProxyV1(
        root_store=root_store,
        worker=worker,
        policy_version_ref="policy-v1",
        model_version_ref="model-v1",
        policy_source="unit",
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-priority-loop",
        policy_source="unit",
    )
    try:
        first = slab.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = slab.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        started = time.perf_counter()
        third = slab.step(
            _compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
            )
        )
        action_elapsed = time.perf_counter() - started
        proxy.drain_owner_maintenance(wait=True)
        proxy_metadata = dict(proxy.metadata)
    finally:
        slab.close()
        proxy.close()
        root_store.close()
        root_store.unlink()

    assert third.action_step is not None
    assert action_elapsed < 1.0
    assert proxy_metadata["compact_owner_search_owner_loop_kind"] == (
        "persistent_priority_owner_loop_v1"
    )
    assert proxy_metadata["compact_owner_search_owner_loop_persistent"] is True
    assert proxy_metadata["compact_owner_search_owner_action_priority_enabled"] is True
    assert proxy_metadata["compact_owner_search_owner_action_while_maintenance_pending_count"] > 0
    assert proxy_metadata["compact_owner_search_owner_action_served_before_maintenance_count"] > 0
    assert proxy_metadata["compact_owner_search_owner_fifo_blocked_action_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_replay_append_submitted_entry_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_replay_append_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_train_request_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_submitted_learner_update_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_learner_update_count"] == 2
    assert (
        proxy_metadata["compact_owner_search_owner_maintenance_drained_replay_append_entry_count"]
        == 2
    )
    assert proxy_metadata["compact_owner_search_owner_maintenance_drained_replay_append_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_maintenance_pending_work_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_maintenance_inflight"] is False
    assert proxy_metadata["compact_owner_search_owner_policy_lag_current"] == 0
    assert proxy_metadata["compact_owner_search_owner_policy_lag_max"] > 0
    assert proxy_metadata["compact_owner_search_owner_action_while_policy_lagged_count"] > 0
    assert proxy_metadata["compact_owner_search_search_refresh_update_count"] == 2
    assert proxy_metadata["compact_owner_search_action_feedback_verified"] is True
    assert proxy_metadata["compact_owner_search_action_feedback_transition_count"] == 2
    assert proxy_metadata["compact_owner_search_action_feedback_action_count"] > 0
    assert proxy_metadata["compact_owner_search_action_feedback_mismatch_count"] == 0
    learner_telemetry = proxy_metadata["compact_owner_search_owner_learner_telemetry"]
    assert learner_telemetry["compact_owner_search_owner_train_wall_sec"] == pytest.approx(0.02)
    assert learner_telemetry["compact_owner_search_owner_train_sample_sec"] == pytest.approx(0.002)
    assert learner_telemetry["compact_owner_search_owner_train_timing_aggregate_count"] == 2
    assert learner_telemetry["compact_muzero_learner_sec"] == pytest.approx(0.006)
    assert learner_telemetry["compact_muzero_learner_backward_sec"] == pytest.approx(0.0008)
    assert proxy_metadata["compact_owner_search_owner_replay_append_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_train_request_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_submitted_learner_update_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_learner_update_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_maintenance_staged_work_item_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_maintenance_drained_work_item_count"] == 2


def test_inline_background_owner_search_serves_action_while_maintenance_is_pending():
    proxy = CompactLazyInlineBackgroundOwnerSearchSlabProxyV1(
        search_service_factory=_FakeSharedObservationSearchService,
        replay_store_factory=_FakeOwnerReplayStore,
        learner_factory=_SlowOwnerLearner,
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
        root_store_capacity=3,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-inline-background",
        policy_source="unit",
    )
    try:
        first = slab.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = slab.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        started = time.perf_counter()
        third = slab.step(
            _compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
            )
        )
        action_elapsed = time.perf_counter() - started
        proxy.drain_owner_maintenance(wait=True)
        proxy_metadata = dict(proxy.metadata)
    finally:
        slab.close()
        proxy.close()

    assert third.action_step is not None
    assert action_elapsed < 1.0
    assert proxy_metadata["compact_owner_search_service_kind"] == (
        COMPACT_OWNER_SEARCH_KIND_INLINE_BACKGROUND
    )
    assert proxy_metadata["compact_owner_search_worker_kind"] == (
        COMPACT_OWNER_SEARCH_KIND_INLINE_BACKGROUND
    )
    assert proxy_metadata["compact_owner_search_owner_loop_kind"] == (
        "inline_background_maintenance_owner_loop_v1"
    )
    assert proxy_metadata["compact_owner_search_owner_background_maintenance_thread"] is True
    assert proxy_metadata["compact_owner_search_owner_background_overlap_enabled"] is True
    assert proxy_metadata["compact_owner_search_owner_action_while_maintenance_pending_count"] > 0
    assert proxy_metadata["compact_owner_search_owner_action_served_before_maintenance_count"] > 0
    assert proxy_metadata["compact_owner_search_owner_fifo_blocked_action_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_replay_append_submitted_entry_count"] == 2


def test_owner_search_direct_stepper_bypasses_parent_slab_rows():
    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_FakeSharedObservationSearchService,
        replay_store_factory=_FakeOwnerReplayStore,
        learner_factory=_FakeOwnerLearner,
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
        root_store_capacity=3,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-direct-stepper",
        policy_source="unit",
        transition_batch_size=2,
    )
    try:
        first = stepper.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = stepper.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        third = stepper.step(
            _compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
            )
        )
        proxy.drain_owner_maintenance(wait=True)
        proxy_metadata = dict(proxy.metadata)
    finally:
        stepper.close()
        proxy.close()

    assert third.committed_index_rows is None
    assert stepper.committed_index_row_count == 0
    assert stepper.committed_index_group_count == 0
    assert third.telemetry["compact_owner_search_slab_bypass"] is True
    assert third.telemetry["compact_owner_search_slab_bypass_kind"] == (
        COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
    )
    assert third.telemetry["compact_rollout_slab_bypassed"] is True
    assert third.telemetry["compact_rollout_slab_general_replay_row_builder_used"] is False
    assert third.telemetry["compact_owner_search_transition_batch_transport_enabled"] is True
    assert third.telemetry["compact_owner_search_transition_batch_count"] == 1
    assert third.telemetry["compact_owner_search_transition_batch_entry_count"] == 2
    assert third.telemetry["compact_owner_search_transition_batch_transport_entry_count"] == 1
    assert third.telemetry["compact_owner_search_transition_batch_transport_kind"] == (
        COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
    )
    assert third.telemetry["compact_owner_search_transition_batch_schema_id"] == (
        COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
    )
    assert third.telemetry["compact_owner_search_transition_batch_digest"]
    assert third.telemetry["compact_owner_search_transition_batch_digest_verified"] is True
    search_metadata = third.telemetry["compact_rollout_slab_search_metadata"]
    assert search_metadata["compact_owner_search_slab_bypass"] is True
    assert search_metadata["compact_owner_search_slab_bypass_kind"] == (
        COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
    )
    assert search_metadata["compact_rollout_slab_bypassed"] is True
    assert search_metadata["compact_rollout_slab_general_replay_row_builder_used"] is False
    assert search_metadata["compact_owner_search_parent_slab_commits_replay"] is False
    assert proxy_metadata["compact_owner_search_owner_replay_append_staged_entry_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_replay_append_submitted_entry_count"] == 2
    assert (
        proxy_metadata["compact_owner_search_owner_replay_append_staged_transport_entry_count"] == 1
    )
    assert (
        proxy_metadata["compact_owner_search_owner_replay_append_submitted_transport_entry_count"]
        == 1
    )
    assert proxy_metadata["compact_owner_search_owner_replay_transport_entry_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_replay_transport_kind"] == (
        COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
    )
    assert proxy_metadata["compact_owner_search_owner_replay_transition_batch_enabled"] is True
    assert proxy_metadata["compact_owner_search_owner_replay_transition_batch_count"] == 1
    assert (
        proxy_metadata["compact_owner_search_owner_replay_transition_batch_transition_count"] == 2
    )
    assert proxy_metadata["compact_owner_search_owner_replay_transition_legacy_entry_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_replay_append_request_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_replay_append_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_train_request_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_submitted_learner_update_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_learner_update_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_maintenance_pending_work_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_maintenance_inflight"] is False
    assert proxy_metadata["compact_owner_search_owner_policy_lag_current"] == 0
    assert proxy_metadata["compact_owner_search_search_refresh_update_count"] == 2
    assert proxy_metadata["compact_owner_search_action_feedback_verified"] is True
    assert proxy_metadata["compact_owner_search_action_feedback_transition_count"] == 2
    assert proxy_metadata["compact_owner_search_action_feedback_action_count"] > 0
    assert proxy_metadata["compact_owner_search_action_feedback_mismatch_count"] == 0


def test_owner_search_direct_stepper_owner_local_transition_derivation_uses_derived_batch():
    replay_store = _FakeDirectTransitionBatchOwnerReplayStore()
    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_FakeSharedObservationSearchService,
        replay_store_factory=lambda: replay_store,
        learner_factory=_FakeOwnerLearner,
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
        root_store_capacity=4,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-derived-transition-stepper",
        policy_source="unit",
        transition_batch_size=2,
        owner_local_transition_derivation=True,
    )
    try:
        first = stepper.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = stepper.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        third = stepper.step(
            _compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
            )
        )
        proxy.drain_owner_maintenance(wait=True)
        proxy_metadata = dict(proxy.metadata)
        stepper_metadata = dict(stepper.metadata)
    finally:
        stepper.close()
        proxy.close()

    assert third.committed_index_rows is None
    assert replay_store.direct_append_call_count == 1
    assert replay_store.legacy_append_call_count == 0
    assert len(replay_store.last_transition_batches) == 1
    batch = replay_store.last_transition_batches[0]
    assert isinstance(batch, CompactOwnerSearchReplayAppendDerivedTransitionBatchV1)
    assert batch.schema_id == COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
    assert batch.metadata["compact_owner_search_replay_append_transition_batch_kind"] == (
        COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
    )
    assert batch.transition_count == 2
    assert not hasattr(batch, "next_reward")
    assert not hasattr(batch, "next_done")
    assert not hasattr(batch, "next_final_reward_map")
    assert np.array_equal(batch.record_indices, np.asarray((0, 1), dtype=np.int64))
    assert np.array_equal(batch.next_record_indices, np.asarray((1, 2), dtype=np.int64))
    assert np.all(np.asarray(batch.applied_action_counts, dtype=np.int64) > 0)
    assert np.all(np.asarray(batch.applied_action_checksums, dtype=np.int64) > 0)
    assert stepper_metadata["compact_owner_search_transition_batch_count"] == 1
    assert stepper_metadata["compact_owner_search_transition_batch_schema_id"] == (
        COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
    )
    assert stepper_metadata["compact_owner_search_transition_batch_transport_kind"] == (
        COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
    )
    assert (
        stepper_metadata["compact_owner_search_owner_local_transition_derivation_transition_count"]
        == 2
    )
    assert (
        stepper_metadata[
            "compact_owner_search_owner_local_transition_derivation_transport_entry_count"
        ]
        == 1
    )
    assert (
        stepper_metadata["compact_owner_search_owner_local_transition_derivation_fallback_count"]
        == 0
    )
    assert proxy_metadata["compact_owner_search_owner_replay_append_submitted_entry_count"] == 2
    assert (
        proxy_metadata["compact_owner_search_owner_replay_append_submitted_transport_entry_count"]
        == 1
    )
    assert proxy_metadata["compact_owner_search_action_feedback_verified"] is True
    assert proxy_metadata["compact_owner_search_action_feedback_transition_count"] == 2
    assert proxy_metadata["compact_owner_search_action_feedback_mismatch_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_maintenance_pending_work_count"] == 0


def test_owner_search_direct_stepper_stubs_resident_parent_host_observation():
    class FakeResidentActionSearch:
        supports_two_phase_compact_search = True

        def run_action_step(self, root_batch):
            assert root_batch.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
            assert root_batch.metadata["resident_host_observation_stubbed"] is True
            assert not np.any(np.asarray(root_batch.observation) == 99)
            resident = root_batch.resident_observation
            assert resident is not None
            active_roots = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = (
                np.asarray(resident.root_device_observation)[active_roots, 0, 0, 0].astype(
                    np.int16, copy=False
                )
                % 3
            )
            return CompactSearchActionStepV1(
                replay_payload_handle="resident-stub:0",
                root_index=active_roots,
                env_row=root_batch.env_row[active_roots].astype(np.int32, copy=True),
                player=root_batch.player[active_roots].astype(np.int16, copy=True),
                policy_env_id=root_batch.policy_env_id[active_roots].astype(
                    np.int64,
                    copy=True,
                ),
                selected_action=selected.astype(np.int16, copy=True),
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "search_impl": "fake_resident_stub_search",
                    "num_simulations": 1,
                    "active_root_count": int(selected.size),
                    "compact_owner_search_owner_materializes_replay": True,
                    "compact_owner_search_action_only_result": True,
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                    "search_replay_payload_digest": (
                        compact_search_deferred_replay_payload_digest_v1("resident-stub:0")
                    ),
                    "search_replay_payload_digest_deferred": True,
                },
            )

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"unexpected flush {replay_payload_handle}")

    batch = _compact_batch((99, 99, 99), joint_action=(0, 0, 0))
    resident_values = np.asarray(batch.observation, dtype=np.uint8).copy()
    resident_values[:, 0, 0, 0, 0] = np.asarray([0, 1, 2], dtype=np.uint8)
    resident = ResidentObservationBatchV1(
        device_observation=resident_values,
        root_device_observation=resident_values.reshape(3, 4, 8, 8),
        generation_id=21,
        batch_size=3,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype=str(resident_values.dtype),
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=21,
        source_backend="unit_direct_stepper_stub",
        host_fallback_allowed=False,
        final_observation_row_mask=np.zeros((3,), dtype=np.bool_),
    )
    resident_batch = SimpleNamespace(
        **{
            **vars(batch),
            "observation_source": COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
            "resident_observation": resident,
        }
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=FakeResidentActionSearch(),
        search_lane="unit-owner-search-resident-stub-stepper",
        policy_source="unit",
        resident_root_host_observation_stub=True,
    )
    try:
        step = stepper.step(resident_batch)
    finally:
        stepper.close()

    np.testing.assert_array_equal(
        step.next_joint_action.reshape(-1),
        np.asarray([0, 1, 2], dtype=np.int16),
    )
    assert step.telemetry["compact_rollout_slab_resident_host_observation_stub_requested"] is True
    assert step.telemetry["compact_rollout_slab_resident_host_observation_stubbed"] is True
    assert (
        step.telemetry["compact_rollout_slab_resident_host_observation_stub_kind"]
        == "zero_stride_shape_only_v1"
    )
    assert (
        step.telemetry["compact_rollout_slab_resident_host_observation_stub_materialized_bytes"]
        == 0
    )
    assert (
        step.telemetry["compact_rollout_slab_resident_host_observation_stub_logical_bytes"]
        == batch.observation.nbytes
    )


def test_owner_search_direct_stepper_root_build_request_avoids_parent_builder(
    monkeypatch,
):
    import curvyzero.training.compact_rollout_slab as slab_module

    class FakeRootBuildRequestSearch:
        supports_two_phase_compact_search = True

        def __init__(self):
            self.request_count = 0

        def run_action_step(self, root_batch):
            raise AssertionError("direct root build request path must not pass root_batch")

        def run_action_step_from_root_build_request(self, root_build_request):
            self.request_count += 1
            root_batch = build_compact_root_batch_v1_from_request(root_build_request)
            assert root_batch.metadata["resident_host_observation_stubbed"] is True
            resident = root_batch.resident_observation
            assert resident is not None
            active_roots = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
            selected = (
                np.asarray(resident.root_device_observation)[active_roots, 0, 0, 0].astype(
                    np.int16, copy=False
                )
                % 3
            )
            metadata = {
                **dict(root_batch.metadata),
                "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                "phase": "action_critical",
                "search_impl": "fake_root_build_request_search",
                "num_simulations": 1,
                "active_root_count": int(selected.size),
                "compact_owner_search_owner_materializes_replay": True,
                "compact_owner_search_action_only_result": True,
                "compact_owner_search_direct_root_build_request_handoff": True,
                "compact_owner_search_direct_root_owner_build_used": True,
                "selected_action_digest": compact_search_array_digest_v1(selected),
                "search_replay_payload_digest": (
                    compact_search_deferred_replay_payload_digest_v1("root-build-request:0")
                ),
                "search_replay_payload_digest_deferred": True,
            }
            return CompactSearchActionStepV1(
                replay_payload_handle="root-build-request:0",
                root_index=active_roots,
                env_row=root_batch.env_row[active_roots].astype(np.int32, copy=True),
                player=root_batch.player[active_roots].astype(np.int16, copy=True),
                policy_env_id=root_batch.policy_env_id[active_roots].astype(
                    np.int64,
                    copy=True,
                ),
                selected_action=selected.astype(np.int16, copy=True),
                metadata=metadata,
            )

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"unexpected flush {replay_payload_handle}")

    def fail_parent_builder(*args, **kwargs):
        raise AssertionError("parent root batch builder must not be called")

    monkeypatch.setattr(slab_module, "build_compact_root_batch_v1", fail_parent_builder)

    batch = _compact_batch((99, 99, 99), joint_action=(0, 0, 0))
    resident_values = np.asarray(batch.observation, dtype=np.uint8).copy()
    resident_values[:, 0, 0, 0, 0] = np.asarray([0, 1, 2], dtype=np.uint8)
    resident = ResidentObservationBatchV1(
        device_observation=resident_values,
        root_device_observation=resident_values.reshape(3, 4, 8, 8),
        generation_id=33,
        batch_size=3,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype=str(resident_values.dtype),
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=33,
        source_backend="unit_direct_stepper_root_build_request",
        host_fallback_allowed=False,
        final_observation_row_mask=np.zeros((3,), dtype=np.bool_),
    )
    resident_batch = SimpleNamespace(
        **{
            **vars(batch),
            "observation_source": COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
            "resident_observation": resident,
        }
    )
    search = FakeRootBuildRequestSearch()
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=search,
        search_lane="unit-owner-search-root-build-request-stepper",
        policy_source="unit",
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )
    try:
        step = stepper.step(resident_batch)
    finally:
        stepper.close()

    assert search.request_count == 1
    assert step.root_batch is None
    np.testing.assert_array_equal(
        step.next_joint_action.reshape(-1),
        np.asarray([0, 1, 2], dtype=np.int16),
    )
    assert step.telemetry["compact_rollout_slab_root_batch_build_sec"] == 0.0
    assert step.telemetry["compact_rollout_slab_root_build_request_sec"] >= 0.0
    assert step.telemetry["compact_owner_search_direct_root_build_request_requested"] is True
    search_metadata = step.telemetry["compact_rollout_slab_search_metadata"]
    assert search_metadata["compact_owner_search_direct_root_build_request_handoff"] is True
    assert search_metadata["compact_owner_search_direct_root_parent_build_avoided"] is True
    assert search_metadata["compact_rollout_slab_parent_root_batch_builder_used"] is False
    assert search_metadata["compact_rollout_slab_parent_root_batch_builder_call_count"] == 0
    assert search_metadata["compact_rollout_slab_return_root_batch_sidecar_stored"] is False
    assert (
        search_metadata["compact_rollout_slab_return_root_batch_sidecar_storage_avoided"]
        is True
    )
    assert search_metadata["compact_rollout_slab_return_root_batch_sidecar_build_count"] == 0


def test_owner_search_direct_stepper_dispatch_pending_fails_closed():
    class FakeDispatchSearch:
        supports_two_phase_compact_search = True

        def __init__(self) -> None:
            self.submit_count = 0
            self.resolve_count = 0

        def run_action_step(self, root_batch):
            del root_batch
            raise AssertionError("split dispatch test must not pass root_batch")

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"split dispatch test flushed {replay_payload_handle}")

        def submit_action_step_from_root_build_request(self, root_build_request):
            self.submit_count += 1
            return SimpleNamespace(
                dispatch_id=self.submit_count,
                root_build_request=root_build_request,
            )

        def resolve_action_step_handle(self, handle, *, sync_wrapper=False):
            del handle, sync_wrapper
            self.resolve_count += 1
            raise AssertionError("pending-close guard should not resolve")

    batch = _compact_batch((99, 99, 99), joint_action=(0, 0, 0))
    resident_values = np.asarray(batch.observation, dtype=np.uint8).copy()
    resident = ResidentObservationBatchV1(
        device_observation=resident_values,
        root_device_observation=resident_values.reshape(3, 4, 8, 8),
        generation_id=35,
        batch_size=3,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype=str(resident_values.dtype),
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=35,
        source_backend="unit_direct_stepper_dispatch_pending",
        host_fallback_allowed=False,
        final_observation_row_mask=np.zeros((3,), dtype=np.bool_),
    )
    resident_batch = SimpleNamespace(
        **{
            **vars(batch),
            "observation_source": COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
            "resident_observation": resident,
        }
    )
    search = FakeDispatchSearch()
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=search,
        search_lane="unit-owner-search-direct-step-dispatch-pending",
        policy_source="unit",
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )

    handle = stepper.submit_step(resident_batch)

    assert isinstance(handle, CompactOwnerSearchDirectStepDispatchHandleV1)
    assert search.submit_count == 1
    assert stepper._pending_direct_step_dispatch is not None
    assert not hasattr(stepper._pending_direct_step_dispatch, "root_build_request")
    assert stepper._pending_direct_step_dispatch.root_action_context.root_count == 3
    assert (
        stepper.metadata["compact_owner_search_pending_root_build_request_stored"]
        is False
    )
    assert (
        stepper.metadata["compact_owner_search_pending_root_action_context_stored"]
        is True
    )
    with pytest.raises(ReplayCompatibilityError, match="one pending step"):
        stepper.submit_step(resident_batch)
    with pytest.raises(ReplayCompatibilityError, match="dispatch pending at close"):
        stepper.close()
    assert search.resolve_count == 0


def test_root_build_request_action_step_reasserts_outer_replay_digest():
    batch = _compact_batch((0, 1, 2), joint_action=(0, 0, 0))
    request = compact_root_build_request_v1_from_batch(
        batch,
        search_lane="unit-root-build-request-digest",
    )
    selected = np.asarray((0, 1, 2), dtype=np.int16)
    outer_handle = "owner-root-build-request:outer"
    inner_handle = "compact-torch-inner:stale"
    stale_digest = compact_search_deferred_replay_payload_digest_v1(inner_handle)

    action_step = compact_action_step_v1_from_owner_search_payload_and_root_request(
        request,
        {
            "selected_action": selected,
            "replay_payload_handle": outer_handle,
            "search_impl": "owner_action_only",
            "num_simulations": 1,
            "search_result_metadata": {
                "replay_payload_handle": inner_handle,
                "selected_action_digest": "stale-selected-digest",
                "search_replay_payload_digest": stale_digest,
                "search_replay_payload_digest_deferred": False,
                "schema_id": "stale-schema",
                "phase": "stale-phase",
            },
        },
        metadata={
            "replay_payload_handle": inner_handle,
            "selected_action_digest": "stale-selected-digest",
            "search_replay_payload_digest": stale_digest,
            "search_replay_payload_digest_deferred": False,
            "schema_id": "stale-schema",
            "phase": "stale-phase",
            "compact_owner_search_worker_wall_sec": 0.25,
        },
    )

    expected_digest = compact_search_deferred_replay_payload_digest_v1(outer_handle)
    metadata = action_step.metadata
    assert action_step.replay_payload_handle == outer_handle
    assert metadata["replay_payload_handle"] == outer_handle
    assert metadata["selected_action_digest"] == compact_search_array_digest_v1(selected)
    assert metadata["search_replay_payload_digest"] == expected_digest
    assert metadata["search_replay_payload_digest"] != stale_digest
    assert metadata["search_replay_payload_digest_deferred"] is True
    assert metadata["schema_id"] == COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID
    assert metadata["phase"] == "action_critical"
    assert metadata["compact_owner_search_worker_wall_sec"] == 0.25


def test_root_build_request_action_step_validates_dense_joint_action():
    batch = _compact_batch((0, 1, 2), joint_action=(0, 0, 0))
    request = compact_root_build_request_v1_from_batch(
        batch,
        search_lane="unit-root-build-request-dense-action",
    )
    selected = np.asarray((0, 1, 2), dtype=np.int16)

    action_step = compact_action_step_v1_from_owner_search_payload_and_root_request(
        request,
        {
            "selected_action": selected,
            "dense_joint_action": ((0,), (1,), (2,)),
            "dense_joint_action_shape": (3, 1),
            "replay_payload_handle": "owner-root-build-request:dense-action",
            "search_impl": "owner_action_only",
            "num_simulations": 1,
        },
        metadata={
            "compact_owner_search_dense_joint_action_present": False,
            "compact_owner_search_dense_joint_action_checksum": 0,
            "compact_owner_search_dense_joint_action_bytes": 0,
            "compact_owner_search_dense_joint_action_mismatch_count": 99,
        },
    )

    assert action_step.dense_joint_action is not None
    np.testing.assert_array_equal(
        action_step.dense_joint_action,
        np.asarray(((0,), (1,), (2,)), dtype=np.int16),
    )
    assert action_step.metadata["compact_owner_search_dense_joint_action_present"] is True
    assert (
        action_step.metadata["compact_owner_search_dense_joint_action_owner_assembled"]
        is True
    )
    assert (
        action_step.metadata[
            "compact_owner_search_dense_joint_action_parent_assembly_avoided"
        ]
        is True
    )
    assert action_step.metadata["compact_owner_search_dense_joint_action_fallback_count"] == 0
    assert action_step.metadata["compact_owner_search_dense_joint_action_fallback_reason"] == "none"
    assert action_step.metadata["compact_owner_search_dense_joint_action_checksum"] != 0
    assert action_step.metadata["compact_owner_search_dense_joint_action_shape"] == (3, 1)
    assert action_step.metadata["compact_owner_search_dense_joint_action_digest"]
    assert action_step.metadata["compact_owner_search_dense_joint_action_bytes"] == 6
    assert action_step.metadata["compact_owner_search_dense_joint_action_mismatch_count"] == 0

    with pytest.raises(
        RuntimeError,
        match="dense_joint_action selected-action mismatch",
    ):
        compact_action_step_v1_from_owner_search_payload_and_root_request(
            request,
            {
                "selected_action": selected,
                "dense_joint_action": ((1,), (1,), (2,)),
                "replay_payload_handle": "owner-root-build-request:dense-action-stale",
                "search_impl": "owner_action_only",
                "num_simulations": 1,
            },
        )

    with pytest.raises(
        RuntimeError,
        match="dense_joint_action shape metadata mismatch",
    ):
        compact_action_step_v1_from_owner_search_payload_and_root_request(
            request,
            {
                "selected_action": selected,
                "dense_joint_action": ((0,), (1,), (2,)),
                "dense_joint_action_shape": (2, 2),
                "replay_payload_handle": "owner-root-build-request:dense-action-bad-shape",
                "search_impl": "owner_action_only",
                "num_simulations": 1,
            },
        )


def test_threaded_owner_search_direct_stepper_root_build_request_uses_owner_build(
    monkeypatch,
):
    import curvyzero.training.compact_rollout_slab as slab_module

    def fail_parent_builder(*args, **kwargs):
        raise AssertionError("parent root batch builder must not be called")

    monkeypatch.setattr(slab_module, "build_compact_root_batch_v1", fail_parent_builder)

    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_fake_resident_observation_search_service_factory,
        owner_replay_append_enabled=False,
        owner_defer_maintenance=True,
        root_store_capacity=3,
        require_resident_root_view=True,
        fixed_action_result_buffer=True,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    batch = _compact_batch((99, 99, 99), joint_action=(0, 0, 0))
    import torch

    resident_values = torch.zeros((3, 1, 4, 8, 8), dtype=torch.uint8)
    resident_values[:, 0, 0, 0, 0] = torch.as_tensor([2, 1, 0], dtype=torch.uint8)
    root_device_observation = resident_values.reshape(3, 4, 8, 8)
    resident = ResidentObservationBatchV1(
        device_observation=resident_values,
        root_device_observation=root_device_observation,
        generation_id=34,
        batch_size=3,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=34,
        source_backend="unit_threaded_root_build_request",
        host_fallback_allowed=False,
        final_observation_row_mask=np.zeros((3,), dtype=np.bool_),
    )
    resident_batch = SimpleNamespace(
        **{
            **vars(batch),
            "observation_source": COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
            "resident_observation": resident,
        }
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-threaded-root-build-request-stepper",
        policy_source="unit",
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )
    try:
        step = stepper.step(resident_batch)
        proxy_metadata = dict(proxy.metadata)
    finally:
        stepper.close()
        proxy.close()

    np.testing.assert_array_equal(
        step.next_joint_action.reshape(-1),
        np.asarray([2, 1, 0], dtype=np.int16),
    )
    assert step.root_batch is None
    search_metadata = step.telemetry["compact_rollout_slab_search_metadata"]
    assert search_metadata["compact_owner_search_threaded_slab_proxy"] is True
    assert search_metadata["compact_owner_search_direct_root_build_request_handoff"] is True
    assert search_metadata["compact_owner_search_direct_root_parent_build_avoided"] is True
    assert search_metadata["compact_owner_search_direct_root_owner_build_used"] is True
    assert search_metadata["compact_owner_search_direct_root_owner_build_count"] == 1
    assert search_metadata["compact_owner_search_slab_proxy"] is True
    assert search_metadata["compact_owner_search_owner_pid"] > 0
    assert search_metadata["compact_owner_search_root_slot_count"] == 3
    assert search_metadata["compact_owner_search_active_root_count"] == 3
    assert search_metadata["compact_owner_search_request_bytes"] > 0
    assert search_metadata["compact_owner_search_result_bytes"] > 0
    assert search_metadata["compact_owner_search_fixed_action_result_buffer_requested"] is True
    assert search_metadata["compact_owner_search_fixed_action_result_buffer_used"] is True
    assert search_metadata["compact_owner_search_fixed_action_result_buffer_wire_result_bytes"] > 0
    assert (
        search_metadata["compact_owner_search_fixed_action_result_buffer_full_result_bytes"]
        > search_metadata["compact_owner_search_fixed_action_result_buffer_wire_result_bytes"]
    )
    assert search_metadata["compact_owner_search_action_dispatch_handle_used"] is True
    assert search_metadata["compact_owner_search_action_dispatch_handle_submit_no_wait"] is True
    assert search_metadata["compact_owner_search_action_dispatch_handle_sync_wrapper"] is True
    assert search_metadata["compact_owner_search_action_dispatch_handle_submit_count"] == 1
    assert search_metadata["compact_owner_search_action_dispatch_handle_resolve_count"] == 1
    assert search_metadata["compact_owner_search_action_dispatch_handle_pending_count"] == 0
    assert (
        search_metadata[
            "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count"
        ]
        == 0
    )
    assert proxy_metadata["compact_owner_search_fixed_action_result_buffer_pending_slot_count"] == 0
    assert proxy_metadata["compact_owner_search_fixed_action_result_buffer_write_count"] == 1
    assert proxy_metadata["compact_owner_search_fixed_action_result_buffer_read_count"] == 1
    assert search_metadata["compact_owner_search_pending_compact_batch_sidecar_stored"] is False
    assert (
        search_metadata["compact_owner_search_pending_compact_batch_sidecar_storage_avoided"]
        is True
    )
    assert search_metadata["compact_owner_search_pending_compact_batch_sidecar_store_count"] == 0
    assert (
        search_metadata[
            "compact_owner_search_pending_compact_batch_sidecar_store_avoided_count"
        ]
        == 1
    )
    assert search_metadata["compact_owner_search_pending_root_batch_sidecar_stored"] is False
    assert (
        search_metadata["compact_owner_search_pending_root_batch_sidecar_storage_avoided"]
        is True
    )
    assert search_metadata["compact_owner_search_pending_root_batch_sidecar_store_count"] == 0
    assert (
        search_metadata[
            "compact_owner_search_pending_root_batch_sidecar_store_avoided_count"
        ]
        == 1
    )
    assert (
        search_metadata[
            "compact_owner_search_pending_action_step_identity_handle_stored"
        ]
        is True
    )
    assert (
        search_metadata[
            "compact_owner_search_pending_action_step_identity_handle_store_count"
        ]
        == 1
    )
    assert search_metadata["compact_owner_search_pending_root_build_request_stored"] is False
    assert search_metadata["compact_rollout_slab_return_root_batch_sidecar_stored"] is False
    assert (
        search_metadata["compact_rollout_slab_return_root_batch_sidecar_storage_avoided"]
        is True
    )
    assert search_metadata["compact_rollout_slab_return_root_batch_sidecar_build_count"] == 0
    assert search_metadata["compact_owner_search_request_cuda_tensor_count"] == 0
    assert search_metadata["compact_owner_search_result_cuda_tensor_count"] == 0
    assert search_metadata["compact_owner_search_action_only_result"] is True
    assert search_metadata["compact_owner_search_owner_materializes_replay"] is True
    assert search_metadata["compact_owner_search_parent_reconstructed_search_result"] is False
    assert search_metadata["compact_owner_search_search_result_payload_bytes"] == 0
    assert (
        search_metadata["compact_owner_search_direct_root_build_request_observation_included"]
        is False
    )
    assert (
        search_metadata["compact_owner_search_direct_root_build_request_observation_bytes_sent"]
        == 0
    )
    assert search_metadata["compact_owner_search_resident_root_view_proved"] is True
    assert search_metadata["compact_owner_search_resident_root_view_h2d_bytes"] == 0.0
    assert search_metadata["compact_owner_search_resident_root_view_d2h_bytes"] == 0.0
    assert proxy_metadata["compact_owner_search_direct_root_owner_build_count"] == 1


def test_owner_action_dispatch_handle_submit_defers_wait_and_slot_read(monkeypatch):
    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_fake_shared_observation_search_service_factory,
        owner_replay_append_enabled=True,
        owner_defer_maintenance=True,
        root_store_capacity=4,
        require_resident_root_view=True,
        fixed_action_result_buffer=True,
    )
    batch = _resident_compact_batch(
        (0, 1, 2),
        joint_action=(0, 0, 0),
        generation_id=301,
    )
    root_build_request = compact_root_build_request_v1_from_batch(
        batch,
        search_lane="unit-owner-action-dispatch-handle",
        metadata={"compact_rollout_slab": True},
        copy_observation=False,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=batch.resident_observation,
        resident_host_observation_stub=True,
    )
    try:
        proxy._initialize(None)
        assert proxy.proxy is not None
        worker = proxy.proxy.worker
        original_result = worker.result

        def fail_if_submit_waits(handle):
            del handle
            raise AssertionError("submit must not wait for owner action result")

        monkeypatch.setattr(worker, "result", fail_if_submit_waits)
        dispatch_handle = proxy.submit_action_step_from_root_build_request(root_build_request)
        assert isinstance(dispatch_handle, CompactOwnerActionDispatchHandleV1)
        assert proxy.action_result_slot_table is not None
        assert proxy.action_result_slot_table.read_count == 0
        assert proxy.metadata["compact_owner_search_action_dispatch_handle_submit_count"] == 1
        assert proxy.metadata["compact_owner_search_action_dispatch_handle_resolve_count"] == 0
        assert proxy.metadata["compact_owner_search_action_dispatch_handle_pending_count"] == 1
        assert (
            proxy.metadata[
                "compact_owner_search_action_dispatch_pending_root_build_request_stored"
            ]
            is False
        )
        assert (
            proxy.metadata[
                "compact_owner_search_action_dispatch_pending_root_build_request_avoided_count"
            ]
            == 1
        )
        assert (
            proxy.metadata[
                "compact_owner_search_action_dispatch_pending_root_action_context_stored"
            ]
            is False
        )
        assert (
            proxy.metadata[
                "compact_owner_search_action_dispatch_pending_root_action_context_store_count"
            ]
            == 0
        )
        assert (
            proxy.metadata[
                (
                    "compact_owner_search_action_dispatch_pending_root_action_context_"
                    "avoided_count"
                )
            ]
            == 1
        )
        pending_dispatch = next(iter(proxy.proxy._pending_action_dispatches.values()))
        assert not hasattr(pending_dispatch, "root_build_request")
        assert not hasattr(pending_dispatch, "root_action_context")
        assert pending_dispatch.root_action_context_handle.root_count == root_build_request.root_count
        assert (
            pending_dispatch.root_action_context_handle.active_root_count
            == root_build_request.active_root_mask.nonzero()[0].shape[0]
        )
        assert proxy.metadata["compact_owner_root_action_context_handle_used"] is False
        assert proxy.metadata["compact_owner_root_action_context_owner_store_count"] == 1
        assert proxy.metadata["compact_owner_root_action_context_owner_pending_count"] == 1
        assert (
            proxy.metadata[
                "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count"
            ]
            == 0
        )

        monkeypatch.setattr(worker, "result", original_result)
        action_step = proxy.resolve_action_step_handle(dispatch_handle)
        assert action_step.dense_joint_action is not None
        assert proxy.action_result_slot_table.read_count == 1
        assert action_step.metadata["compact_owner_search_action_dispatch_handle_used"] is True
        assert (
            action_step.metadata["compact_owner_search_action_dispatch_handle_submit_no_wait"]
            is True
        )
        assert action_step.metadata["compact_owner_search_action_dispatch_handle_sync_wrapper"] is False
        assert action_step.metadata["compact_owner_search_action_dispatch_handle_submit_count"] == 1
        assert action_step.metadata["compact_owner_search_action_dispatch_handle_resolve_count"] == 1
        assert action_step.metadata["compact_owner_search_action_dispatch_handle_pending_count"] == 0
        assert (
            action_step.metadata[
                "compact_owner_search_action_dispatch_pending_root_build_request_stored"
            ]
            is False
        )
        assert (
            action_step.metadata[
                "compact_owner_search_action_dispatch_pending_root_build_request_avoided_count"
            ]
            == 1
        )
        assert (
            action_step.metadata[
                "compact_owner_search_action_dispatch_pending_root_action_context_stored"
            ]
            is False
        )
        assert (
            action_step.metadata["compact_owner_root_action_context_handle_used"] is True
        )
        assert (
            action_step.metadata["compact_owner_root_action_context_owner_pending_count"] == 0
        )
        assert (
            action_step.metadata["compact_owner_root_action_context_owner_resolve_count"] == 1
        )
        assert (
            action_step.metadata["compact_owner_root_action_context_owner_release_count"] == 1
        )
        assert (
            action_step.metadata[
                "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count"
            ]
            == 0
        )
    finally:
        proxy.close()


def test_owner_action_dispatch_handle_pending_close_fails_closed():
    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_fake_shared_observation_search_service_factory,
        owner_replay_append_enabled=True,
        owner_defer_maintenance=True,
        root_store_capacity=4,
        require_resident_root_view=True,
        fixed_action_result_buffer=True,
    )
    batch = _resident_compact_batch(
        (0, 1, 2),
        joint_action=(0, 0, 0),
        generation_id=302,
    )
    root_build_request = compact_root_build_request_v1_from_batch(
        batch,
        search_lane="unit-owner-action-dispatch-handle-close",
        metadata={"compact_rollout_slab": True},
        copy_observation=False,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=batch.resident_observation,
        resident_host_observation_stub=True,
    )

    proxy.submit_action_step_from_root_build_request(root_build_request)
    assert proxy.metadata["compact_owner_search_action_dispatch_handle_pending_count"] == 1
    with pytest.raises(ReplayCompatibilityError, match="dispatch handle pending at close"):
        proxy.close()
    assert proxy.metadata["compact_owner_search_slab_proxy_initialized"] is False


def test_direct_root_owner_local_transition_derivation_avoids_parent_root_sidecar(
    monkeypatch,
):
    import curvyzero.training.compact_rollout_slab as slab_module

    class FakeDirectRootOwnerLocalTransitionSearch:
        supports_two_phase_compact_search = True

        def __init__(self) -> None:
            self.request_count = 0
            self.staged_entries: list[Any] = []

        def run_action_step(self, root_batch):
            del root_batch
            raise AssertionError("direct root build request path must not pass root_batch")

        def run_action_step_from_root_build_request(self, root_build_request):
            resident = root_build_request.resident_observation
            assert resident is not None
            active_roots = np.flatnonzero(root_build_request.active_root_mask).astype(np.int32)
            selected = (
                np.asarray(resident.root_device_observation)[active_roots, 0, 0, 0].astype(
                    np.int16,
                    copy=False,
                )
                % 3
            )
            env_row = np.asarray(root_build_request.policy_env_row, dtype=np.int32).reshape(-1)
            player = np.asarray(root_build_request.policy_player, dtype=np.int16).reshape(-1)
            policy_env_id = np.asarray(root_build_request.policy_env_id, dtype=np.int64).reshape(
                -1
            )
            dense = np.zeros(
                (int(root_build_request.batch_size), int(root_build_request.player_count)),
                dtype=np.int16,
            )
            dense[
                env_row[active_roots].astype(np.int64),
                player[active_roots].astype(np.int64),
            ] = selected
            handle = f"direct-root-owner-local:{self.request_count}"
            self.request_count += 1
            return CompactSearchActionStepV1(
                replay_payload_handle=handle,
                root_index=active_roots.astype(np.int32, copy=True),
                env_row=env_row[active_roots].astype(np.int32, copy=True),
                player=player[active_roots].astype(np.int16, copy=True),
                policy_env_id=policy_env_id[active_roots].astype(np.int64, copy=True),
                selected_action=selected.astype(np.int16, copy=True),
                dense_joint_action=dense,
                metadata={
                    "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                    "phase": "action_critical",
                    "search_impl": "fake_direct_root_owner_local_transition_search",
                    "num_simulations": 1,
                    "active_root_count": int(selected.size),
                    "compact_owner_search_owner_materializes_replay": True,
                    "compact_owner_search_action_only_result": True,
                    "compact_owner_search_direct_root_build_request_handoff": True,
                    "compact_owner_search_direct_root_parent_build_avoided": True,
                    "selected_action_digest": compact_search_array_digest_v1(selected),
                    "search_replay_payload_digest": (
                        compact_search_deferred_replay_payload_digest_v1(handle)
                    ),
                    "search_replay_payload_digest_deferred": True,
                    "compact_owner_search_dense_joint_action_owner_assembled": True,
                    "compact_owner_search_dense_joint_action_parent_assembly_avoided": True,
                    "compact_owner_search_dense_joint_action_present": True,
                    "compact_owner_search_dense_joint_action_checksum": 0,
                    "compact_owner_search_dense_joint_action_digest": "",
                    "compact_owner_search_dense_joint_action_bytes": int(dense.nbytes),
                    "compact_owner_search_dense_joint_action_mismatch_count": 0,
                },
            )

        def stage_replay_append_entries(self, replay_append_entries):
            self.staged_entries.append(replay_append_entries)
            return int(getattr(replay_append_entries, "transition_count", 1))

        def flush_replay_payload(self, replay_payload_handle):
            raise AssertionError(f"unexpected replay flush {replay_payload_handle}")

    def fail_parent_root_sidecar(*args, **kwargs):
        del args, kwargs
        raise AssertionError("direct-root owner-local path must not build parent root sidecar")

    def fail_parent_root_builder(*args, **kwargs):
        del args, kwargs
        raise AssertionError("direct-root owner-local path must not call parent root builder")

    monkeypatch.setattr(slab_module, "_root_batch_sidecar_from_build_request", fail_parent_root_sidecar)
    monkeypatch.setattr(slab_module, "build_compact_root_batch_v1", fail_parent_root_builder)

    search = FakeDirectRootOwnerLocalTransitionSearch()
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=search,
        search_lane="unit-owner-search-direct-root-derived-transition-stepper",
        policy_source="unit",
        transition_batch_size=2,
        owner_local_transition_derivation=True,
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )
    try:
        first = stepper.step(
            _resident_compact_batch(
                (0, 1, 2),
                joint_action=(0, 0, 0),
                generation_id=101,
            )
        )
        second = stepper.step(
            _resident_compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
                generation_id=102,
            )
        )
        third = stepper.step(
            _resident_compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
                generation_id=103,
            )
        )
        stepper_metadata = dict(stepper.metadata)
    finally:
        stepper.close()

    assert first.root_batch is None
    assert second.root_batch is None
    assert third.root_batch is None
    assert len(search.staged_entries) == 1
    batch = search.staged_entries[0]
    assert isinstance(batch, CompactOwnerSearchReplayAppendDerivedTransitionBatchV1)
    assert batch.transition_count == 2
    assert np.array_equal(batch.record_indices, np.asarray((0, 1), dtype=np.int64))
    assert np.all(np.asarray(batch.applied_action_counts, dtype=np.int64) > 0)
    assert np.all(np.asarray(batch.applied_action_checksums, dtype=np.int64) > 0)
    search_metadata = third.telemetry["compact_rollout_slab_search_metadata"]
    assert search_metadata["compact_rollout_slab_return_root_batch_sidecar_stored"] is False
    assert (
        search_metadata["compact_rollout_slab_return_root_batch_sidecar_storage_avoided"]
        is True
    )
    assert search_metadata["compact_rollout_slab_return_root_batch_sidecar_build_count"] == 0
    assert third.telemetry["compact_rollout_slab_root_batch_build_sec"] == 0.0
    assert third.telemetry["compact_rollout_slab_root_count"] == 3
    assert third.telemetry["compact_rollout_slab_resident_host_observation_stubbed"] is True
    assert stepper_metadata["compact_owner_search_transition_batch_count"] == 1
    assert stepper_metadata["compact_owner_search_transition_batch_schema_id"] == (
        COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
    )


def test_direct_root_owner_proxy_transition_closure_bypasses_parent_stage_previous(
    monkeypatch,
):
    def fail_parent_stage(*args, **kwargs):
        del args, kwargs
        raise AssertionError("parent previous-transition closure must not run")

    monkeypatch.setattr(
        CompactOwnerSearchDirectStepperV1,
        "_stage_previous_derived_transition",
        fail_parent_stage,
    )
    monkeypatch.setattr(
        CompactOwnerSearchDirectStepperV1,
        "_flush_derived_transition_batch",
        fail_parent_stage,
    )
    monkeypatch.setattr(
        CompactOwnerSearchDirectStepperV1,
        "_commit_previous_transition",
        fail_parent_stage,
    )

    replay_store = _FakeDirectTransitionBatchOwnerReplayStore()
    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=lambda: replay_store,
        learner_factory=_FakeOwnerLearner,
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
        root_store_capacity=4,
        require_resident_root_view=True,
        fixed_action_result_buffer=True,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-proxy-transition-closure",
        policy_source="unit",
        transition_batch_size=2,
        owner_local_transition_derivation=True,
        owner_proxy_transition_closure=True,
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )
    try:
        first = stepper.step(
            _resident_compact_batch(
                (0, 1, 2),
                joint_action=(0, 0, 0),
                generation_id=201,
            )
        )
        second = stepper.step(
            _resident_compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
                generation_id=202,
            )
        )
        third = stepper.step(
            _resident_compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
                generation_id=203,
            )
        )
        proxy.drain_owner_maintenance(wait=True)
        proxy_metadata = dict(proxy.metadata)
        stepper_metadata = dict(stepper.metadata)
    finally:
        stepper.close()
        proxy.close()

    assert first.root_batch is None
    assert second.root_batch is None
    assert third.root_batch is None
    assert replay_store.direct_append_call_count == 1
    assert replay_store.legacy_append_call_count == 0
    assert len(replay_store.last_transition_batches) == 1
    batch = replay_store.last_transition_batches[0]
    assert isinstance(batch, CompactOwnerSearchReplayAppendDerivedTransitionBatchV1)
    assert batch.transition_count == 2
    assert np.array_equal(batch.record_indices, np.asarray((0, 1), dtype=np.int64))
    assert np.array_equal(batch.next_record_indices, np.asarray((1, 2), dtype=np.int64))
    assert np.all(np.asarray(batch.applied_action_counts, dtype=np.int64) > 0)
    assert np.all(np.asarray(batch.applied_action_checksums, dtype=np.int64) > 0)
    assert stepper_metadata["compact_owner_search_parent_previous_transition_closure_count"] == 0
    assert (
        stepper_metadata[
            "compact_owner_search_parent_previous_transition_closure_avoided_count"
        ]
        == 3
    )
    assert stepper_metadata["compact_owner_search_owner_proxy_transition_closure_used"] is True
    assert (
        stepper_metadata["compact_owner_search_owner_proxy_transition_closure_used_count"] == 2
    )
    assert (
        stepper_metadata[
            "compact_owner_search_pending_action_step_identity_handle_stored"
        ]
        is False
    )
    assert (
        stepper_metadata[
            "compact_owner_search_pending_action_step_identity_handle_storage_avoided"
        ]
        is True
    )
    assert (
        stepper_metadata[
            "compact_owner_search_pending_action_step_identity_handle_store_count"
        ]
        == 0
    )
    assert (
        stepper_metadata[
            "compact_owner_search_pending_action_step_identity_handle_store_avoided_count"
        ]
        == 3
    )
    assert stepper_metadata["compact_owner_search_transition_batch_count"] == 1
    assert stepper_metadata["compact_owner_search_transition_batch_entry_count"] == 2
    assert stepper_metadata["compact_owner_search_transition_batch_schema_id"] == (
        COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
    )
    assert proxy_metadata["compact_owner_search_owner_proxy_transition_closure_used"] is True
    assert (
        proxy_metadata["compact_owner_search_owner_proxy_transition_closure_closed_count"] == 2
    )
    assert proxy_metadata["compact_owner_search_owner_proxy_transition_closure_batch_count"] == 1
    assert (
        proxy_metadata["compact_owner_search_owner_proxy_transition_closure_transition_count"] == 2
    )
    assert (
        proxy_metadata["compact_owner_search_owner_proxy_transition_closure_pending_count"] == 0
    )
    assert proxy_metadata["compact_owner_search_owner_proxy_applied_action_verification_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_proxy_applied_action_mismatch_count"] == 0
    assert proxy_metadata["compact_owner_search_parent_previous_transition_closure_count"] == 0
    assert proxy_metadata["compact_owner_search_parent_applied_action_validation_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_proxy_action_frame_pending"] is True
    assert proxy_metadata["compact_owner_search_owner_proxy_action_frame_store_count"] == 3
    assert proxy_metadata["compact_owner_search_action_feedback_verified"] is True
    assert proxy_metadata["compact_owner_search_action_feedback_transition_count"] == 2
    assert proxy_metadata["compact_owner_search_action_feedback_mismatch_count"] == 0


def test_direct_root_owner_proxy_transition_closure_suppresses_warmup_frames():
    replay_store = _FakeDirectTransitionBatchOwnerReplayStore()
    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=lambda: replay_store,
        learner_factory=_FakeOwnerLearner,
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
        root_store_capacity=4,
        require_resident_root_view=True,
        fixed_action_result_buffer=True,
        root_store_metadata={
            "owner_search_inner_search_service_kind": "compact_torch_search_service",
            "owner_search_compact_torch_resident_root_bridge_ready": False,
        },
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-proxy-transition-closure-warmup",
        policy_source="unit",
        transition_batch_size=2,
        owner_local_transition_derivation=True,
        owner_proxy_transition_closure=True,
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )
    try:
        proxy.set_owner_learning_enabled(False)
        first = stepper.step(
            _resident_compact_batch(
                (0, 1, 2),
                joint_action=(0, 0, 0),
                generation_id=301,
            )
        )
        second = stepper.step(
            _resident_compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
                generation_id=302,
            )
        )
        proxy.set_owner_learning_enabled(True)
        third = stepper.step(
            _resident_compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
                generation_id=303,
            )
        )
        fourth = stepper.step(
            _resident_compact_batch(
                (0, 2, 1),
                joint_action=tuple(int(value) for value in third.next_joint_action[:, 0]),
                generation_id=304,
            )
        )
        del fourth
        proxy.drain_owner_maintenance(wait=True)
        proxy_metadata = dict(proxy.metadata)
        stepper_metadata = dict(stepper.metadata)
    finally:
        stepper.close()
        proxy.close()

    assert replay_store.direct_append_call_count == 1
    assert replay_store.legacy_append_call_count == 0
    assert len(replay_store.last_transition_batches) == 1
    batch = replay_store.last_transition_batches[0]
    assert batch.transition_count == 2
    assert np.array_equal(batch.record_indices, np.asarray((1, 2), dtype=np.int64))
    assert np.array_equal(batch.next_record_indices, np.asarray((2, 3), dtype=np.int64))
    assert proxy_metadata["compact_owner_search_owner_proxy_transition_closure_used"] is True
    assert proxy_metadata["compact_owner_search_owner_proxy_transition_closure_closed_count"] == 2
    assert proxy_metadata["compact_owner_search_owner_proxy_transition_closure_batch_count"] == 1
    assert (
        proxy_metadata["compact_owner_search_owner_proxy_transition_closure_transition_count"]
        == 2
    )
    assert (
        proxy_metadata["compact_owner_search_owner_proxy_applied_action_verification_count"]
        == 2
    )
    assert proxy_metadata["compact_owner_search_owner_proxy_action_frame_store_count"] == 4
    assert stepper_metadata["compact_owner_search_transition_batch_entry_count"] == 2


def test_direct_root_owner_proxy_transition_closure_rejects_applied_action_mismatch():
    replay_store = _FakeDirectTransitionBatchOwnerReplayStore()
    proxy = CompactLazyThreadedOwnerSearchSlabProxyV1(
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=lambda: replay_store,
        learner_factory=_FakeOwnerLearner,
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
        root_store_capacity=4,
        require_resident_root_view=True,
        fixed_action_result_buffer=True,
    )
    stepper = CompactOwnerSearchDirectStepperV1(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-proxy-transition-closure-mismatch",
        policy_source="unit",
        transition_batch_size=2,
        owner_local_transition_derivation=True,
        owner_proxy_transition_closure=True,
        resident_root_host_observation_stub=True,
        direct_root_build_request=True,
    )
    try:
        first = stepper.step(
            _resident_compact_batch(
                (0, 1, 2),
                joint_action=(0, 0, 0),
                generation_id=211,
            )
        )
        bad_action = tuple((int(value) + 1) % 3 for value in first.next_joint_action[:, 0])
        with pytest.raises(ReplayCompatibilityError, match="applied action mismatch"):
            stepper.step(
                _resident_compact_batch(
                    (2, 2, 1),
                    joint_action=bad_action,
                    generation_id=212,
                )
            )
        proxy_metadata = dict(proxy.metadata)
        stepper_metadata = dict(stepper.metadata)
    finally:
        stepper.close()
        proxy.close()

    assert replay_store.direct_append_call_count == 0
    assert stepper_metadata["compact_owner_search_parent_previous_transition_closure_count"] == 0
    assert (
        stepper_metadata[
            "compact_owner_search_parent_previous_transition_closure_avoided_count"
        ]
        == 2
    )
    assert (
        stepper_metadata[
            "compact_owner_search_pending_action_step_identity_handle_store_count"
        ]
        == 0
    )
    assert (
        stepper_metadata[
            "compact_owner_search_pending_action_step_identity_handle_store_avoided_count"
        ]
        == 1
    )
    assert proxy_metadata["compact_owner_search_owner_proxy_applied_action_mismatch_count"] == 1
    assert proxy_metadata["compact_owner_search_owner_proxy_transition_closure_batch_count"] == 0


def test_owner_search_async_learner_worker_serves_action_while_learner_pending():
    root_store = CompactSharedMemoryRootStoreV1.create(
        _shared_observation_root_batch((0, 0, 0)),
        capacity=3,
    )
    worker = CompactProcessOwnerSearchWorkerV1(
        root_provider_factory=build_compact_shared_memory_root_provider_v1,
        root_provider_factory_kwargs={"spec": root_store.spec},
        search_service_factory=_fake_shared_observation_search_service_factory,
        replay_store_factory=_fake_owner_replay_store_factory,
        learner_factory=_slow_owner_learner_factory,
        async_learner_worker=True,
    )
    proxy = CompactOwnerSearchSlabProxyV1(
        root_store=root_store,
        worker=worker,
        policy_version_ref="policy-v1",
        model_version_ref="model-v1",
        policy_source="unit",
        owner_replay_append_enabled=True,
        owner_sample_batch_size=1,
        owner_train_steps=1,
        owner_train_interval=1,
        owner_defer_maintenance=True,
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-owner-search-async-learner",
        policy_source="unit",
    )
    try:
        first = slab.step(_compact_batch((0, 1, 2), joint_action=(0, 0, 0)))
        second = slab.step(
            _compact_batch(
                (2, 2, 1),
                joint_action=tuple(int(value) for value in first.next_joint_action[:, 0]),
            )
        )
        started = time.perf_counter()
        third = slab.step(
            _compact_batch(
                (1, 0, 2),
                joint_action=tuple(int(value) for value in second.next_joint_action[:, 0]),
            )
        )
        third_action_elapsed = time.perf_counter() - started
        time.sleep(0.2)
        started = time.perf_counter()
        fourth = slab.step(
            _compact_batch(
                (0, 2, 1),
                joint_action=tuple(int(value) for value in third.next_joint_action[:, 0]),
            )
        )
        async_action_elapsed = time.perf_counter() - started
        proxy_metadata = proxy.drain_owner_maintenance(wait=True)
    finally:
        slab.close()
        proxy.close()
        root_store.close()
        root_store.unlink()

    assert third.action_step is not None
    assert fourth.action_step is not None
    assert third_action_elapsed < 1.0
    assert async_action_elapsed < 1.0
    assert proxy_metadata["compact_owner_search_owner_async_learner_worker_enabled"] is True
    assert (
        proxy_metadata["compact_owner_search_owner_async_learner_worker_kind"]
        == "in_process_thread_v1"
    )
    assert proxy_metadata["compact_owner_search_owner_async_learner_submit_count"] == 3
    assert proxy_metadata["compact_owner_search_owner_async_learner_completed_count"] == 3
    assert proxy_metadata["compact_owner_search_owner_async_learner_pending_count"] == 0
    assert proxy_metadata["compact_owner_search_owner_async_learner_max_pending_observed"] > 0
    assert proxy_metadata["compact_owner_search_owner_action_while_async_learner_pending_count"] > 0
    assert proxy_metadata["compact_owner_search_owner_async_learner_failed"] is False
    assert proxy_metadata["compact_owner_search_owner_train_request_count"] == 3
    assert proxy_metadata["compact_owner_search_owner_submitted_learner_update_count"] == 3
    assert proxy_metadata["compact_owner_search_owner_learner_update_count"] == 3
    learner_telemetry = proxy_metadata["compact_owner_search_owner_learner_telemetry"]
    assert learner_telemetry["compact_owner_search_owner_train_timing_aggregate_count"] == 3
    assert proxy_metadata["compact_owner_search_owner_maintenance_pending_work_count"] == 0
    assert proxy_metadata["compact_owner_search_search_refresh_update_count"] == 3
    assert proxy_metadata["compact_owner_search_owner_policy_lag_current"] == 0


def test_lazy_owner_search_slab_proxy_initializes_from_first_root_batch():
    proxy = CompactLazyOwnerSearchSlabProxyV1(
        search_service_factory=_fake_shared_observation_search_service_factory,
        policy_version_ref="policy-v1",
        model_version_ref="model-v1",
        policy_source="unit",
    )
    slab = CompactRolloutSlab(
        batch_size=3,
        player_count=1,
        search_service=proxy,
        search_lane="unit-lazy-owner-search",
        policy_source="unit",
    )
    proxy_metadata: dict[str, Any] = {}
    try:
        first = slab.step(_compact_batch((2, 0, 1), joint_action=(0, 0, 0)))
        proxy_metadata = dict(proxy.metadata)
    finally:
        slab.close()
        proxy.close()

    assert tuple(int(value) for value in first.next_joint_action[:, 0]) == (2, 0, 1)
    assert first.search_result is None
    assert first.action_step is not None
    assert first.action_step.metadata["compact_owner_search_lazy_slab_proxy"] is True
    assert first.action_step.metadata["compact_owner_search_slab_proxy_initialized"] is True
    assert first.action_step.metadata["compact_owner_search_parent_slab_commits_replay"] is True
    assert first.action_step.metadata["compact_owner_search_parent_wait_sec"] >= 0.0
    assert first.action_step.metadata["compact_owner_search_worker_wall_sec"] >= 0.0
    assert proxy_metadata["compact_owner_search_slab_proxy_initialized"] is True
    assert proxy.last_search_result_payload_bytes > 0


def test_shared_memory_root_store_aligns_row_masks_to_root_slots():
    root_batch = _shared_observation_root_batch((0, 1, 2, 0))
    root_batch = CompactRootBatchV1(
        observation=root_batch.observation,
        legal_mask=root_batch.legal_mask,
        active_root_mask=root_batch.active_root_mask,
        to_play=root_batch.to_play,
        env_row=np.asarray((0, 0, 1, 1), dtype=np.int32),
        player=np.asarray((0, 1, 0, 1), dtype=np.int16),
        policy_env_id=root_batch.policy_env_id,
        target_reward=root_batch.target_reward,
        done_root=root_batch.done_root,
        final_observation=None,
        final_observation_row_mask=np.asarray((False, False), dtype=np.bool_),
        terminal_row_mask=np.asarray((False, True), dtype=np.bool_),
        autoreset_row_mask=np.asarray((True, False), dtype=np.bool_),
        metadata=root_batch.metadata,
    )
    root_store = CompactSharedMemoryRootStoreV1.create(root_batch, capacity=4)
    try:
        resolved = root_store.resolve_root_batch(
            root_slot_ids=(0, 1, 2, 3),
            request=CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=0,
                root_slot_ids=(0, 1, 2, 3),
            ),
        )
    finally:
        root_store.close()
        root_store.unlink()

    assert tuple(bool(value) for value in resolved.terminal_row_mask) == (
        False,
        False,
        True,
        True,
    )
    assert tuple(bool(value) for value in resolved.autoreset_row_mask) == (
        True,
        True,
        False,
        False,
    )


def test_owner_search_service_refuses_owner_ref_without_owner_side_search_refresh():
    service = CompactOwnerSearchServiceV1(
        root_provider=_FakeRootProvider(),
        search_service=_FakeSearchWithoutOwnerRefresh(),
        replay_store=_FakeOwnerReplayStore(),
        learner=_FakeOwnerLearner(),
    )

    with pytest.raises(RuntimeError, match="refresh_model_owner_ref"):
        service.run(
            CompactOwnerSearchRequestV1(
                request_id=1,
                actor_step=0,
                root_slot_ids=(1, 2),
                replay_append_entries=({"row": 1},),
                sample_batch_size=1,
                train_steps=1,
                policy_version_ref="policy-v1",
                model_version_ref="model-v1",
                policy_source="unit",
            )
        )


class _FakeRootProvider:
    def __init__(self) -> None:
        self.seen_slot_ids: list[tuple[int, ...]] = []

    def resolve_root_batch(
        self,
        *,
        root_slot_ids: tuple[int, ...],
        request: CompactOwnerSearchRequestV1,
    ) -> CompactRootBatchV1:
        del request
        self.seen_slot_ids.append(tuple(root_slot_ids))
        root_count = len(tuple(root_slot_ids))
        return CompactRootBatchV1(
            observation=np.zeros((root_count, 4, 8, 8), dtype=np.uint8),
            legal_mask=np.ones((root_count, 3), dtype=np.bool_),
            active_root_mask=np.ones((root_count,), dtype=np.bool_),
            to_play=np.full((root_count,), -1, dtype=np.int64),
            env_row=np.asarray(root_slot_ids, dtype=np.int32),
            player=np.zeros((root_count,), dtype=np.int16),
            policy_env_id=np.asarray(root_slot_ids, dtype=np.int64),
            target_reward=np.zeros((root_count, 1), dtype=np.float32),
            done_root=np.zeros((root_count,), dtype=np.bool_),
            final_observation=None,
            final_observation_row_mask=np.zeros((root_count,), dtype=np.bool_),
            terminal_row_mask=np.zeros((root_count,), dtype=np.bool_),
            autoreset_row_mask=np.zeros((root_count,), dtype=np.bool_),
            metadata={
                "schema_id": "curvyzero_compact_root_batch/v1",
                "root_provider": "fake-owner-root-provider",
            },
        )


class _FakeOwnerSearchService:
    search_impl = "fake_owner_search"
    num_simulations = 3

    def __init__(self) -> None:
        self.run_count = 0
        self.refresh_count = 0
        self.last_owner_ref: dict[str, Any] = {}

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        self.run_count += 1
        active_count = int(np.asarray(root_batch.active_root_mask).sum())
        selected_action = (np.arange(active_count, dtype=np.int16) + self.refresh_count) % 3
        visit_policy = np.zeros((active_count, 3), dtype=np.float32)
        visit_policy[np.arange(active_count), selected_action] = 1.0
        return validate_compact_search_result_v1(
            root_batch,
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=np.full((active_count,), float(self.refresh_count), dtype=np.float32),
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
        )

    def refresh_model_owner_ref(
        self,
        *,
        owner_ref: dict[str, Any],
        policy_version_ref: str,
        model_version_ref: str,
        policy_source: str,
        learner_update_count: int,
        expected_model_state_digest: str | None = None,
    ) -> dict[str, Any]:
        assert policy_version_ref == "policy-v1"
        assert model_version_ref == "model-v1"
        assert policy_source == "unit"
        assert expected_model_state_digest == owner_ref["model_state_digest"]
        self.refresh_count += 1
        self.last_owner_ref = dict(owner_ref)
        return {
            "schema_id": "curvyzero_compact_policy_refresh_search_worker_state/v1",
            "learner_update_count": int(learner_update_count),
            "model_state_digest": str(owner_ref["model_state_digest"]),
            "search_worker_model_object_id": id(self),
            "refresh_count": int(self.refresh_count),
        }


class _FakeSearchWithoutOwnerRefresh:
    search_impl = "fake_owner_search_without_refresh"
    num_simulations = 1

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        active_count = int(np.asarray(root_batch.active_root_mask).sum())
        selected_action = np.zeros((active_count,), dtype=np.int16)
        visit_policy = np.zeros((active_count, 3), dtype=np.float32)
        visit_policy[:, 0] = 1.0
        return validate_compact_search_result_v1(
            root_batch,
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=np.zeros((active_count,), dtype=np.float32),
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
        )


class _FakeSharedObservationSearchService(_FakeOwnerSearchService):
    search_impl = "fake_shared_observation_owner_search"

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        self.run_count += 1
        active_count = int(np.asarray(root_batch.active_root_mask).sum())
        selected_action = (
            np.asarray(root_batch.observation, dtype=np.int16)[:active_count, 0, 0, 0] % 3
        ).astype(np.int16, copy=False)
        visit_policy = np.zeros((active_count, 3), dtype=np.float32)
        visit_policy[np.arange(active_count), selected_action] = 1.0
        return validate_compact_search_result_v1(
            root_batch,
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=np.full(
                (active_count,),
                float(np.asarray(root_batch.observation)[:active_count, 0, 0, 0].sum()),
                dtype=np.float32,
            ),
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
        )


class _FakeResidentObservationSearchService(_FakeOwnerSearchService):
    search_impl = "fake_resident_observation_owner_search"

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        self.run_count += 1
        assert root_batch.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        resident = root_batch.resident_observation
        assert resident is not None
        assert resident.host_fallback_allowed is False
        assert resident.root_device_observation is not None
        active_count = int(np.asarray(root_batch.active_root_mask).sum())
        selected_action = (
            resident.root_device_observation[:active_count, 0, 0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int16, copy=False)
            % 3
        )
        visit_policy = np.zeros((active_count, 3), dtype=np.float32)
        visit_policy[np.arange(active_count), selected_action] = 1.0
        return validate_compact_search_result_v1(
            root_batch,
            selected_action=selected_action,
            visit_policy=visit_policy,
            root_value=np.full((active_count,), float(selected_action.sum()), dtype=np.float32),
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
            metadata={
                "resident_owner_search_observation_source": root_batch.observation_source,
                "resident_owner_search_host_fallback_allowed": (resident.host_fallback_allowed),
                "compact_owner_search_resident_root_bridge_ready": False,
                "owner_search_compact_torch_resident_root_bridge_ready": False,
                "compact_owner_search_resident_root_bridge_kind": "",
                "compact_owner_search_resident_root_bridge_h2d_bytes": 0.0,
            },
        )


class _FakeTwoPhaseResidentObservationSearchService(_FakeResidentObservationSearchService):
    search_impl = "fake_two_phase_resident_observation_owner_search"

    def __init__(self) -> None:
        super().__init__()
        self.action_step_count = 0
        self.device_flush_count = 0
        self.deferred_payload_identity_match = True
        self.deferred_payload_model_refresh_crossed_count = 0
        self.deferred_payload_final_pending_count = 0
        self._pending_action_steps: dict[str, CompactSearchActionStepV1] = {}

    def run_action_step(self, root_batch: CompactRootBatchV1) -> CompactSearchActionStepV1:
        assert root_batch.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        resident = root_batch.resident_observation
        assert resident is not None
        assert resident.root_device_observation is not None
        active_root_indices = np.flatnonzero(root_batch.active_root_mask).astype(np.int32)
        selected_action = (
            resident.root_device_observation[: int(active_root_indices.size), 0, 0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int16, copy=False)
            % 3
        )
        handle = f"{self.search_impl}:{self.action_step_count}"
        metadata = {
            "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
            "phase": "action_critical",
            "search_impl": self.search_impl,
            "num_simulations": int(self.num_simulations),
            "active_root_count": int(selected_action.size),
            "replay_payload_origin": f"{self.search_impl}:{handle}",
            "selected_action_digest": compact_search_array_digest_v1(selected_action),
            "search_replay_payload_digest": (
                compact_search_deferred_replay_payload_digest_v1(handle)
            ),
            "search_replay_payload_digest_deferred": True,
        }
        action_step = CompactSearchActionStepV1(
            replay_payload_handle=handle,
            root_index=active_root_indices.astype(np.int32, copy=True),
            env_row=root_batch.env_row[active_root_indices].astype(np.int32, copy=True),
            player=root_batch.player[active_root_indices].astype(np.int16, copy=True),
            policy_env_id=root_batch.policy_env_id[active_root_indices].astype(
                np.int64,
                copy=True,
            ),
            selected_action=selected_action.astype(np.int16, copy=True),
            metadata=metadata,
        )
        self._pending_action_steps[handle] = action_step
        self.action_step_count += 1
        return action_step

    def flush_device_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactDeviceSearchReplayPayloadV1:
        import torch

        handle = str(replay_payload_handle)
        action_step = self._pending_action_steps.pop(handle)
        selected_action = np.asarray(action_step.selected_action, dtype=np.int16)
        visit_policy_np = np.zeros((int(selected_action.size), 3), dtype=np.float32)
        if selected_action.size:
            visit_policy_np[np.arange(int(selected_action.size)), selected_action] = 1.0
        visit_policy = torch.as_tensor(visit_policy_np, dtype=torch.float32)
        root_value = torch.full(
            (int(selected_action.size),),
            float(selected_action.sum()),
            dtype=torch.float32,
        )
        metadata = dict(action_step.metadata)
        metadata.update(
            {
                "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                "phase": "replay_critical_device",
                "search_result_schema_id": "curvyzero_compact_search_result/v1",
                "device_replay_payload": True,
                "device_replay_payload_device": str(visit_policy.device),
                "host_search_payload_fallback_allowed": False,
                "active_root_count": int(selected_action.size),
                "compact_torch_search_one_simulation_replay_materialization_deferred": True,
                "compact_torch_search_one_simulation_replay_materialized_on_flush": True,
                "compact_torch_search_deferred_one_simulation_replay_recurrent_inference_calls": 1.0,
                "compact_torch_search_deferred_one_simulation_model_identity_match": bool(
                    self.deferred_payload_identity_match
                ),
                "compact_torch_search_deferred_one_simulation_model_refresh_crossed_count": int(
                    self.deferred_payload_model_refresh_crossed_count
                ),
                "compact_torch_search_pending_deferred_replay_payload_count": 1.0,
                "compact_torch_search_pending_deferred_replay_payload_final_count": float(
                    self.deferred_payload_final_pending_count
                ),
                "compact_torch_search_deferred_one_simulation_replay_flush_sec": 0.001,
                "compact_torch_search_service_device_replay_payload_flush_sec": 0.001,
                "compact_torch_search_service_replay_payload_d2h_bytes": 0.0,
                "compact_torch_search_service_device_replay_payload_flushed": True,
                "compact_torch_search_deferred_one_simulation_action_model_state_digest": "digest-a",
                "compact_torch_search_deferred_one_simulation_flush_model_state_digest": "digest-a",
            }
        )
        self.device_flush_count += 1
        return CompactDeviceSearchReplayPayloadV1(
            replay_payload_handle=handle,
            root_index=action_step.root_index.astype(np.int32, copy=True),
            env_row=action_step.env_row.astype(np.int32, copy=True),
            player=action_step.player.astype(np.int16, copy=True),
            policy_env_id=action_step.policy_env_id.astype(np.int64, copy=True),
            visit_policy=visit_policy,
            root_value=root_value,
            raw_visit_counts=visit_policy.clone(),
            predicted_value=None,
            predicted_policy_logits=None,
            metadata=metadata,
        )


class _FakeOwnerReplayStore:
    def __init__(self) -> None:
        self.append_count = 0
        self.last_entries: tuple[Any, ...] = ()

    def append_owner_search_replay(
        self,
        *,
        replay_append_entries: tuple[Any, ...],
        root_batch: CompactRootBatchV1,
        search_result: CompactSearchResultV1,
        request: CompactOwnerSearchRequestV1,
    ) -> int:
        assert isinstance(root_batch, CompactRootBatchV1)
        assert isinstance(search_result, CompactSearchResultV1)
        assert isinstance(request, CompactOwnerSearchRequestV1)
        self.last_entries = tuple(replay_append_entries)
        for entry in self.last_entries:
            metadata = dict(getattr(entry, "metadata", {}) or {})
            if metadata.get("compact_owner_search_replay_append_entry") is True:
                assert isinstance(entry, CompactOwnerSearchReplayAppendIndexEntryV1)
                assert metadata["compact_owner_search_replay_append_index_only"] is True
                assert (
                    metadata["compact_owner_search_replay_append_carries_compact_batches"] is False
                )
                assert not hasattr(entry, "previous_compact_batch")
                assert not hasattr(entry, "current_compact_batch")
        self.append_count += len(self.last_entries)
        return len(self.last_entries)


class _FakeDirectTransitionBatchOwnerReplayStore(_FakeOwnerReplayStore):
    def __init__(self) -> None:
        super().__init__()
        self.direct_append_call_count = 0
        self.legacy_append_call_count = 0
        self.last_transition_batches: tuple[Any, ...] = ()
        self.last_root_batch_cache_keys: tuple[int, ...] = ()

    def append_owner_search_transition_batches(
        self,
        *,
        replay_append_transition_batches: tuple[Any, ...],
        root_batch: CompactRootBatchV1,
        search_result: CompactSearchResultV1,
        request: CompactOwnerSearchRequestV1,
        root_batch_cache: Mapping[int, CompactRootBatchV1] | None = None,
    ) -> int:
        assert isinstance(root_batch, CompactRootBatchV1)
        assert isinstance(search_result, CompactSearchResultV1)
        assert isinstance(request, CompactOwnerSearchRequestV1)
        self.last_root_batch_cache_keys = tuple(
            sorted(int(key) for key in dict(root_batch_cache or {}).keys())
        )
        self.direct_append_call_count += 1
        self.last_transition_batches = tuple(replay_append_transition_batches)
        assert self.last_transition_batches
        for batch in self.last_transition_batches:
            assert isinstance(
                batch,
                (
                    CompactOwnerSearchReplayAppendTransitionBatchV1,
                    CompactOwnerSearchReplayAppendDerivedTransitionBatchV1,
                ),
            )
        transition_count = sum(
            int(batch.transition_count) for batch in self.last_transition_batches
        )
        derived_action_count = sum(
            int(np.asarray(getattr(batch, "applied_action_counts", ()), dtype=np.int64).sum())
            for batch in self.last_transition_batches
            if hasattr(batch, "applied_action_counts")
        )
        derived_action_checksum = sum(
            int(
                np.asarray(
                    getattr(batch, "applied_action_checksums", ()),
                    dtype=np.int64,
                ).sum()
            )
            for batch in self.last_transition_batches
            if hasattr(batch, "applied_action_checksums")
        )
        self.append_count += int(transition_count)
        return {
            "appended_count": int(transition_count),
            "owner_action_feedback": {
                "compact_owner_search_action_feedback_verified": bool(derived_action_count > 0),
                "compact_owner_search_action_feedback_transition_count": int(
                    transition_count if derived_action_count > 0 else 0
                ),
                "compact_owner_search_action_feedback_action_count": int(derived_action_count),
                "compact_owner_search_action_feedback_mismatch_count": 0,
                "compact_owner_search_expected_joint_action_checksum": int(derived_action_checksum),
                "compact_owner_search_applied_joint_action_checksum": int(derived_action_checksum),
                "compact_owner_search_replay_action_checksum": int(derived_action_checksum),
            },
            "compact_owner_search_direct_transition_batch_replay_requested": True,
            "compact_owner_search_direct_transition_batch_replay_used": True,
            "compact_owner_search_direct_transition_batch_replay_batch_count": len(
                self.last_transition_batches
            ),
            "compact_owner_search_direct_transition_batch_replay_transition_count": int(
                transition_count
            ),
            "compact_owner_search_direct_transition_batch_replay_transport_entry_count": len(
                self.last_transition_batches
            ),
            "compact_owner_search_direct_transition_batch_replay_legacy_expanded_entry_count": 0,
            "compact_owner_search_direct_transition_batch_replay_index_entry_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_ring_entry_object_count": 0,
            "compact_owner_search_direct_transition_batch_replay_columnar_append_used": True,
            "compact_owner_search_direct_transition_batch_replay_columnar_slot_write_count": int(
                transition_count
            ),
            "compact_owner_search_direct_transition_batch_replay_fallback_count": 0,
            "compact_owner_search_direct_transition_batch_replay_fallback_reason": "none",
            "compact_owner_search_direct_transition_batch_replay_last_append_sec": 0.001,
        }

    def append_owner_search_replay(self, **kwargs: Any) -> int:
        del kwargs
        self.legacy_append_call_count += 1
        raise AssertionError("legacy replay append should not be called")


class _FakeOwnerLearner:
    def __init__(self) -> None:
        self.train_calls = 0
        self.update_count = 0
        self.last_sample_batch_size: int | None = None

    def train_owner_search_step(
        self,
        *,
        replay_store: _FakeOwnerReplayStore,
        root_batch: CompactRootBatchV1,
        search_result: CompactSearchResultV1,
        sample_batch_size: int,
        train_steps: int,
        request: CompactOwnerSearchRequestV1,
    ) -> dict[str, Any]:
        assert replay_store.append_count > 0
        assert isinstance(root_batch, CompactRootBatchV1)
        assert isinstance(search_result, CompactSearchResultV1)
        assert sample_batch_size >= 0
        assert isinstance(request, CompactOwnerSearchRequestV1)
        self.last_sample_batch_size = int(sample_batch_size)
        self.train_calls += 1
        self.update_count += int(train_steps)
        result: dict[str, Any] = {
            "learner_update_count": int(train_steps),
            "sample_metadata": {
                "compact_rollout_slab_sample_gate_sample_row_count": int(sample_batch_size),
                "compact_rollout_slab_sample_gate_target_row_count": int(sample_batch_size),
                "compact_rollout_slab_sample_gate_requested_sample_row_count": int(
                    sample_batch_size
                ),
                "compact_rollout_slab_sample_gate_require_next_targets": True,
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
            },
            "learner_telemetry": {
                "compact_owner_search_owner_train_wall_sec": 0.01,
                "compact_owner_search_owner_train_sample_sec": 0.001,
                "compact_owner_search_owner_train_learner_update_sec": 0.002,
                "compact_owner_search_owner_train_model_state_digest_sec": 0.003,
                "compact_owner_search_owner_train_model_state_dict_sec": 0.004,
                "compact_owner_search_owner_train_owner_ref_build_sec": 0.0005,
                "compact_owner_search_owner_train_accounted_sec": 0.0105,
                "compact_owner_search_owner_train_residual_sec": 0.0001,
                "compact_muzero_learner_sec": 0.003,
                "compact_muzero_learner_validation_sec": 0.0002,
                "compact_muzero_learner_initial_inference_sec": 0.0003,
                "compact_muzero_learner_backward_sec": 0.0004,
                "compact_muzero_learner_optimizer_step_sec": 0.0005,
                "compact_muzero_learner_accounted_sec": 0.0025,
                "compact_muzero_learner_residual_sec": 0.0005,
            },
        }
        if bool(request.refresh_model):
            result["model_owner_ref"] = {
                "schema_id": "curvyzero_compact_owned_loop_model_owner_ref/v1",
                "transport_kind": COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
                "model_state_digest": f"owner-digest-{self.update_count}",
                "model_version_ref": "model-v1",
                "policy_version_ref": "policy-v1",
                "policy_source": "unit",
                "worker_pid": 12345,
            }
        return result


class _SlowOwnerLearner(_FakeOwnerLearner):
    def train_owner_search_step(self, **kwargs: Any) -> dict[str, Any]:
        time.sleep(1.5)
        return super().train_owner_search_step(**kwargs)


class _FakeExternalOwnerLearnerWorker:
    def __init__(self) -> None:
        self.requests: list[CompactOwnerSearchRequestV1] = []
        self.closed = False

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "compact_owner_search_owner_async_learner_worker_kind": "external_fake",
            "compact_owner_search_owner_async_learner_worker_resource_scope": (
                "external_fake_resource"
            ),
            "compact_owner_search_owner_async_learner_resource_distinct_from_owner": True,
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
    ) -> Future[Any]:
        assert isinstance(root_batch, CompactRootBatchV1)
        assert isinstance(search_result, CompactSearchResultV1)
        self.requests.append(request)
        updates = int(request.train_steps)
        future: Future[Any] = Future()
        future.set_result(
            (
                {
                    "schema_id": "curvyzero_compact_owned_loop_model_owner_ref/v1",
                    "transport_kind": COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
                    "model_state_digest": f"external-owner-digest-{updates}",
                    "model_version_ref": "model-v1",
                    "policy_version_ref": "policy-v1",
                    "policy_source": "unit",
                    "worker_pid": 54321,
                },
                updates,
                {},
                {
                    "compact_owner_search_owner_train_wall_sec": 0.01,
                    "compact_owner_search_owner_train_learner_update_sec": 0.002,
                },
            )
        )
        return future

    def done(self, handle: Future[Any]) -> bool:
        return bool(handle.done())

    def result(self, handle: Future[Any]) -> Any:
        return handle.result()

    def close(self) -> None:
        self.closed = True


class _FakeProcessOwnerBatchLearner:
    def __init__(self) -> None:
        self.prepare_calls = 0

    def prepare_owner_search_learner_payload(
        self,
        *,
        replay_store: _FakeOwnerReplayStore,
        root_batch: CompactRootBatchV1,
        search_result: CompactSearchResultV1,
        request: CompactOwnerSearchRequestV1,
    ) -> dict[str, Any]:
        assert replay_store.append_count > 0
        assert isinstance(root_batch, CompactRootBatchV1)
        assert isinstance(search_result, CompactSearchResultV1)
        self.prepare_calls += 1
        return {
            "payload_schema_id": "unit_owner_search_learner_payload/v1",
            "sample_row_count": int(request.sample_batch_size),
            "sample_telemetry": {
                "unit_owner_search_process_payload_prepared": True,
            },
            "learner_batch": {
                "train_steps": int(request.train_steps),
                "request_id": int(request.request_id),
            },
        }

    def train_owner_search_learner_payload(
        self,
        *,
        payload: dict[str, Any],
        request: CompactOwnerSearchRequestV1,
    ) -> dict[str, Any]:
        learner_batch = dict(payload["learner_batch"])
        updates = int(learner_batch["train_steps"])
        worker_pid = os.getpid()
        return {
            "learner_update_count": updates,
            "model_owner_ref": {
                "schema_id": "curvyzero_compact_owned_loop_model_owner_ref/v1",
                "transport_kind": COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
                "model_state_digest": f"process-owner-digest-{updates}",
                "model_version_ref": str(request.model_version_ref),
                "policy_version_ref": str(request.policy_version_ref),
                "policy_source": str(request.policy_source),
                "worker_pid": worker_pid,
            },
            "sample_telemetry": dict(payload.get("sample_telemetry") or {}),
            "learner_telemetry": {
                "compact_owner_search_owner_train_wall_sec": 0.02,
                "compact_owner_search_owner_train_learner_update_sec": 0.01,
                "unit_owner_search_process_learner_pid": worker_pid,
            },
        }


def _fake_root_provider_factory() -> _FakeRootProvider:
    return _FakeRootProvider()


def _fake_owner_search_service_factory() -> _FakeOwnerSearchService:
    return _FakeOwnerSearchService()


def _fake_shared_observation_search_service_factory() -> _FakeSharedObservationSearchService:
    return _FakeSharedObservationSearchService()


def _fake_resident_observation_search_service_factory() -> _FakeResidentObservationSearchService:
    return _FakeResidentObservationSearchService()


def _fake_owner_replay_store_factory() -> _FakeOwnerReplayStore:
    return _FakeOwnerReplayStore()


def _fake_owner_learner_factory() -> _FakeOwnerLearner:
    return _FakeOwnerLearner()


def _slow_owner_learner_factory() -> _SlowOwnerLearner:
    return _SlowOwnerLearner()


def _fake_process_owner_batch_learner_factory() -> _FakeProcessOwnerBatchLearner:
    return _FakeProcessOwnerBatchLearner()


def _shared_observation_root_batch(actions: tuple[int, ...]) -> CompactRootBatchV1:
    root_count = len(tuple(actions))
    observation = np.zeros((root_count, 4, 8, 8), dtype=np.uint8)
    observation[:, 0, 0, 0] = np.asarray(actions, dtype=np.uint8)
    return CompactRootBatchV1(
        observation=observation,
        legal_mask=np.ones((root_count, 3), dtype=np.bool_),
        active_root_mask=np.ones((root_count,), dtype=np.bool_),
        to_play=np.full((root_count,), -1, dtype=np.int64),
        env_row=np.arange(root_count, dtype=np.int32),
        player=np.zeros((root_count,), dtype=np.int16),
        policy_env_id=np.arange(root_count, dtype=np.int64),
        target_reward=np.zeros((root_count, 1), dtype=np.float32),
        done_root=np.zeros((root_count,), dtype=np.bool_),
        final_observation=None,
        final_observation_row_mask=np.zeros((root_count,), dtype=np.bool_),
        terminal_row_mask=np.zeros((root_count,), dtype=np.bool_),
        autoreset_row_mask=np.zeros((root_count,), dtype=np.bool_),
        metadata={
            "schema_id": "curvyzero_compact_root_batch/v1",
            "root_provider": "shared-memory-unit-test",
        },
    )


def _resident_direct_root_batch(
    *,
    host_actions: tuple[int, ...],
    resident_actions: tuple[int, ...],
    generation_id: int = 7,
    host_fallback_allowed: bool = False,
    row_major_order: bool = True,
) -> CompactRootBatchV1:
    import torch

    root_batch = _shared_observation_root_batch(host_actions)
    resident_values = np.asarray(resident_actions, dtype=np.uint8)
    root_device_observation = torch.zeros(
        (len(tuple(resident_actions)), 4, 8, 8),
        dtype=torch.uint8,
    )
    root_device_observation[:, 0, 0, 0] = torch.as_tensor(
        resident_values,
        dtype=torch.uint8,
    )
    resident = ResidentObservationBatchV1(
        device_observation=root_device_observation.reshape(
            len(tuple(resident_actions)),
            1,
            4,
            8,
            8,
        ),
        root_device_observation=root_device_observation,
        generation_id=int(generation_id),
        batch_size=len(tuple(resident_actions)),
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype="torch.uint8",
        device="cpu",
        row_major_order=bool(row_major_order),
        fresh_for_step_index=int(generation_id),
        source_backend="unit-direct-resident-root",
        host_fallback_allowed=bool(host_fallback_allowed),
    )
    return replace(
        root_batch,
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=resident,
        metadata={
            **dict(root_batch.metadata),
            "root_provider": "direct-resident-unit-test",
        },
    )


def _compact_batch(
    actions_from_observation: tuple[int, ...],
    *,
    joint_action: tuple[int, ...],
) -> SimpleNamespace:
    batch_size = len(tuple(actions_from_observation))
    player_count = 1
    root_count = batch_size * player_count
    observation = np.zeros((batch_size, player_count, 4, 8, 8), dtype=np.uint8)
    observation[:, 0, 0, 0, 0] = np.asarray(actions_from_observation, dtype=np.uint8)
    reward = np.zeros((batch_size, player_count), dtype=np.float32)
    done = np.zeros((batch_size,), dtype=np.bool_)
    return SimpleNamespace(
        observation=observation,
        action_mask=np.ones((batch_size, player_count, 3), dtype=np.bool_),
        reward=reward,
        final_reward_map=reward.copy(),
        done=done,
        policy_env_id=np.arange(root_count, dtype=np.int64),
        policy_env_row=np.arange(batch_size, dtype=np.int32),
        policy_player=np.zeros((root_count,), dtype=np.int16),
        target_reward=reward.reshape(root_count, 1).astype(np.float32, copy=True),
        done_root=np.repeat(done, player_count).astype(np.bool_, copy=False),
        to_play=np.full((root_count,), -1, dtype=np.int64),
        active_root_mask=np.ones((root_count,), dtype=np.bool_),
        final_observation=None,
        final_observation_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        terminal_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        autoreset_row_mask=np.zeros((batch_size,), dtype=np.bool_),
        joint_action=np.asarray(joint_action, dtype=np.int16).reshape(
            batch_size,
            player_count,
        ),
        terminated=done.copy(),
        truncated=np.zeros((batch_size,), dtype=np.bool_),
    )


def _resident_compact_batch(
    actions_from_observation: tuple[int, ...],
    *,
    joint_action: tuple[int, ...],
    generation_id: int,
) -> SimpleNamespace:
    batch = _compact_batch(actions_from_observation, joint_action=joint_action)
    resident_values = np.asarray(batch.observation, dtype=np.uint8).copy()
    resident_values[:, 0, 0, 0, 0] = np.asarray(actions_from_observation, dtype=np.uint8)
    batch_size = int(resident_values.shape[0])
    player_count = int(resident_values.shape[1])
    resident = ResidentObservationBatchV1(
        device_observation=resident_values,
        root_device_observation=resident_values.reshape(batch_size * player_count, 4, 8, 8),
        generation_id=int(generation_id),
        batch_size=batch_size,
        player_count=player_count,
        stack_shape=(4, 8, 8),
        dtype=str(resident_values.dtype),
        device="cpu",
        row_major_order=True,
        fresh_for_step_index=int(generation_id),
        source_backend="unit_resident_compact_batch",
        host_fallback_allowed=False,
        final_observation_row_mask=np.zeros((batch_size,), dtype=np.bool_),
    )
    return SimpleNamespace(
        **{
            **vars(batch),
            "observation_source": COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
            "resident_observation": resident,
        }
    )
