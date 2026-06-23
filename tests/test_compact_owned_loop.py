import os
import pickle
import threading
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.training.compact_owned_loop import COMPACT_OWNED_LOOP_SCHEMA_ID
from curvyzero.training.compact_owned_loop import (
    COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1,
)
from curvyzero.training.compact_owned_loop import (
    COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS,
)
from curvyzero.training.compact_owned_loop import CompactOwnedLoopConfigV1
from curvyzero.training.compact_owned_loop import CompactOwnedLoopV1
from curvyzero.training.compact_owned_loop import CompactPendingSampleLearnerWorkV1
from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import CompactProcessSampleLearnerWorkerV1
from curvyzero.training.compact_owned_loop import CompactSampleLearnerWorkRequestV1
from curvyzero.training.compact_owned_loop import (
    _DEFERRED_LEARNER_MODEL_OWNER_REF_KEY,
)
from curvyzero.training.compact_owned_loop import (
    _DEFERRED_LEARNER_MODEL_STATE_DICT_KEY,
)
from curvyzero.training.compact_owned_loop import (
    _DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY,
)
from curvyzero.training.compact_owned_loop import compact_owned_loop_replay_store_metadata
from curvyzero.training.compact_policy_row_bridge import CompactReplayIndexRowsV1
from curvyzero.training.source_state_batched_observation_profile import (
    SourceStateBatchedRenderResult,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    COMPACT_REPLAY_STORE_STATE_SCHEMA_ID,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    CompactReplayRendererBackedObservationProviderV1,
)
from curvyzero.training.source_state_hybrid_observation_profile import _CompactReplayRingV1


def test_compact_owned_loop_split_handoff_samples_loaded_durable_replay_store():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    config = _loop_config(capture_replay_store_state=True)
    ring = _CompactReplayRingV1(
        capacity=4,
        metadata=compact_owned_loop_replay_store_metadata(
            policy,
            extra={"split_phase": "actor_search_append"},
        ),
    )
    loop_a = CompactOwnedLoopV1(
        config=config,
        policy_version=policy,
        replay_store=ring,
    )
    loop_a.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    phase_a = loop_a.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )
    assert phase_a.appended_replay_rows is True
    assert phase_a.sampled is True
    assert phase_a.trained is False

    state = loop_a.snapshot_replay_store_state()
    assert state.metadata["schema_id"] == COMPACT_REPLAY_STORE_STATE_SCHEMA_ID
    assert state.metadata["compact_owned_loop_schema_id"] == COMPACT_OWNED_LOOP_SCHEMA_ID
    assert state.metadata["compact_owned_loop_replay_store_owned"] is True
    assert state.metadata["compact_owned_loop_policy_version_handoff"] is True
    assert state.metadata["profile_only"] is True
    assert state.metadata["calls_train_muzero"] is False
    assert state.metadata["touches_live_runs"] is False
    assert state.metadata["compact_coach_compatibility_profile_only"] is True
    assert state.metadata["compact_coach_compatibility_promotion_eligible"] is False
    assert (
        "checkpoint_save_load"
        in state.metadata["compact_coach_compatibility_missing_required_gates"]
    )
    assert (
        "checkpoint_save_load"
        in state.metadata["compact_coach_compatibility_missing_required_evidence"]
    )

    ring._entries[0].index_rows.action[:] = 0
    ring._entries[0].previous_step.observation[:] = 999.0

    loaded_state = pickle.loads(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    restored_ring = _CompactReplayRingV1.from_durable_state(loaded_state)
    loop_b = CompactOwnedLoopV1(
        config=_loop_config(capture_replay_store_state=False),
        policy_version=policy,
        replay_store=restored_ring,
    )

    phase_b = loop_b.sample_and_train_from_store()
    assert phase_b.appended_replay_rows is False
    assert phase_b.sampled is True
    assert phase_b.trained is False
    assert phase_b.sample_result is not None
    batch = phase_b.sample_result["sample_batch"]

    np.testing.assert_array_equal(np.sort(batch.action), np.asarray([1, 2], dtype=np.int16))
    assert float(np.max(batch.observation)) < 999.0
    metadata = batch.metadata
    assert metadata["compact_replay_store_loaded"] is True
    assert metadata["compact_replay_store_load_strict"] is True
    assert metadata["compact_owned_loop_entrypoint"] is True
    assert metadata["compact_owned_loop_replay_store_owned"] is True
    assert metadata["compact_owned_loop_policy_version_ref"] == "unit-owned-policy-v1"
    assert metadata["compact_owned_loop_model_version_ref"] == "unit-owned-model-v1"
    assert metadata["compact_owned_loop_policy_source"] == "unit_test_compact_owned_loop"
    assert metadata["profile_only"] is True
    assert metadata["calls_train_muzero"] is False
    assert metadata["touches_live_runs"] is False
    assert metadata["compact_coach_compatibility_gate_trainer_entrypoint"] is False
    assert phase_b.telemetry["compact_owned_loop_sample_gate_calls"] == 1
    assert phase_b.telemetry["compact_owned_loop_calls_train_muzero"] is False


def test_compact_owned_loop_real_update_still_promotion_ineligible():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(capture_replay_store_state=True),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=_FakeLearner(),
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    result = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert result.trained is True
    assert result.telemetry["compact_owned_loop_learner_gate_updates"] == 1
    assert result.telemetry["compact_coach_compatibility_promotion_eligible"] is False
    assert result.telemetry["compact_coach_compatibility_gate_checkpoint_save_load"] is False
    assert (
        "checkpoint_save_load"
        in result.telemetry["compact_coach_compatibility_missing_required_gates"]
    )


def test_compact_owned_loop_deferred_learner_submits_then_drains():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _BlockingFakeLearner()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_learner_gate=True,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    result = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert learner.started.wait(timeout=2.0)
    assert result.appended_replay_rows is True
    assert result.sampled is True
    assert result.trained is False
    assert result.learner_result is None
    assert loop.has_pending_learner_result is True
    assert loop.learner_gate_calls == 0
    assert loop.deferred_learner_submit_count == 1
    assert loop.telemetry()["compact_owned_loop_deferred_learner_pending"] is True

    learner.release.set()
    learner_result = loop.consume_completed_learner_result(wait=True)

    assert learner_result is not None
    assert loop.has_pending_learner_result is False
    assert loop.learner_gate_calls == 1
    assert loop.learner_gate_updates == 1
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_defer_learner_gate"] is True
    assert telemetry["compact_owned_loop_deferred_learner_submit_count"] == 1
    assert telemetry["compact_owned_loop_deferred_learner_completed_count"] == 1
    assert telemetry["compact_owned_loop_deferred_learner_pending"] is False
    assert telemetry["compact_owned_loop_deferred_learner_pending_count"] == 0
    loop.close()


def test_compact_owned_loop_deferred_learner_queues_without_sample_wait():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _BlockingFakeLearner()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_learner_gate=True,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    first = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )
    assert learner.started.wait(timeout=2.0)

    second = loop.record_step(
        current_step=_make_step(
            actions=[2, 0],
            rewards=[5.0, 6.0],
            observation_base=30,
        ),
        index_rows=_make_rows(record_index=8, actions=[2, 0], rewards=[5.0, 6.0]),
    )

    assert first.trained is False
    assert second.trained is False
    assert loop.deferred_learner_submit_count == 2
    assert loop.deferred_learner_completed_count == 0
    assert loop.has_pending_learner_result is True
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_record_step_calls"] == 2
    assert telemetry["compact_owned_loop_appended_replay_entry_count"] == 2
    assert telemetry["compact_owned_loop_deferred_learner_actor_steps_while_pending"] == 1
    assert telemetry["compact_owned_loop_deferred_learner_pending"] is True
    assert telemetry["compact_owned_loop_deferred_learner_pending_count"] == 2
    assert telemetry["compact_owned_loop_deferred_learner_max_pending_observed"] == 2
    assert telemetry["compact_owned_loop_deferred_learner_policy_lag_current"] == 2
    assert telemetry["compact_owned_loop_deferred_learner_policy_lag_max"] == 2

    learner.release.set()
    learner_result = loop.consume_completed_learner_result(wait=True)

    assert learner_result is not None
    assert learner_result["compact_rollout_slab_learner_gate_updates"] == 2
    assert learner_result["compact_owned_loop_learner_result_aggregate_count"] == 2
    assert loop.has_pending_learner_result is False
    assert loop.learner_gate_calls == 2
    assert loop.learner_gate_updates == 2
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_deferred_learner_submit_count"] == 2
    assert telemetry["compact_owned_loop_deferred_learner_completed_count"] == 2
    assert telemetry["compact_owned_loop_deferred_learner_pending_count"] == 0
    assert telemetry["compact_owned_loop_deferred_learner_policy_lag_current"] == 0
    assert telemetry["compact_owned_loop_deferred_learner_policy_lag_max"] == 2
    loop.close()


def test_compact_owned_loop_deferred_sample_learner_submits_from_snapshot_then_drains():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _BlockingFakeLearner()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    result = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert learner.started.wait(timeout=2.0)
    assert result.appended_replay_rows is True
    assert result.sampled is False
    assert result.trained is False
    assert loop.has_pending_sample_learner_result is True
    assert loop.sample_gate_calls == 0
    assert loop.learner_gate_calls == 0
    assert loop.deferred_sample_learner_submit_count == 1

    learner.release.set()
    drained = loop.consume_completed_sample_learner_result(wait=True)

    assert drained is not None
    assert drained.sampled is True
    assert drained.trained is True
    assert drained.sample_result is not None
    assert drained.learner_result is not None
    assert loop.has_pending_sample_learner_result is False
    assert loop.sample_gate_calls == 1
    assert loop.learner_gate_calls == 1
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_defer_sample_learner_gate"] is True
    assert telemetry["compact_owned_loop_deferred_sample_learner_submit_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_completed_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_pending"] is False
    assert telemetry["compact_owned_loop_deferred_sample_learner_pending_count"] == 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_max_pending_observed"] == 1
    assert (
        drained.sample_result["telemetry"]["compact_rollout_slab_sample_gate_snapshot_version"]
        == 1
    )
    loop.close()


def test_compact_owned_loop_deferred_sample_learner_reports_collection_while_pending():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _BlockingFakeLearner()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_gate_max_pending=2,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    first = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )
    assert learner.started.wait(timeout=2.0)
    second = loop.record_step(
        current_step=_make_step(
            actions=[2, 0],
            rewards=[5.0, 6.0],
            observation_base=30,
        ),
        index_rows=_make_rows(record_index=8, actions=[2, 0], rewards=[5.0, 6.0]),
    )

    assert first.sampled is False
    assert second.sampled is False
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_record_step_calls"] == 2
    assert telemetry["compact_owned_loop_appended_replay_entry_count"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_submit_count"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_completed_count"] == 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_pending_count"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_actor_steps_while_pending"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_policy_lag_current"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_policy_lag_max"] == 2

    learner.release.set()
    while loop.has_pending_sample_learner_result:
        loop.consume_completed_sample_learner_result(wait=True)

    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_deferred_sample_learner_completed_count"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_pending_count"] == 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_policy_lag_current"] == 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_policy_lag_max"] == 2
    loop.close()


def test_compact_owned_loop_accepts_injected_sample_learner_worker():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _FakeLearner()
    worker = _FakeExternalSampleLearnerWorker()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
        sample_learner_worker=worker,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    result = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert result.sampled is False
    assert result.trained is False
    assert len(worker.requests) == 1
    assert worker.requests[0].request_id == 1
    assert worker.requests[0].policy_version_ref == "unit-owned-policy-v1"
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_sample_learner_worker_kind"] == "external_fake"
    assert telemetry["compact_owned_loop_sample_learner_resource_distinct_from_actor_search"] is True
    assert telemetry["compact_owned_loop_deferred_sample_learner_last_submitted_request_id"] == 1
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_last_submitted_snapshot_version"]
        == 1
    )
    assert loop.sample_gate_calls == 0
    assert loop.learner_gate_calls == 0

    drained = loop.consume_completed_sample_learner_result(wait=True)

    assert drained is not None
    assert drained.sampled is True
    assert drained.trained is True
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_deferred_sample_learner_completed_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_last_completed_request_id"] == 1
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_last_completed_snapshot_version"]
        == 1
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_policy_lag_current"] == 0
    assert worker.closed is False
    loop.close()
    assert worker.closed is True


def test_compact_owned_loop_process_sample_learner_worker_reports_distinct_pid():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _StatefulFakeLearner()
    worker = CompactProcessSampleLearnerWorkerV1()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_gate_max_pending=2,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
        sample_learner_worker=worker,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    result = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert result.sampled is False
    assert result.trained is False
    second = loop.record_step(
        current_step=_make_step(
            actions=[2, 0],
            rewards=[5.0, 6.0],
            observation_base=30,
        ),
        index_rows=_make_rows(record_index=8, actions=[2, 0], rewards=[5.0, 6.0]),
    )
    assert second.sampled is False
    assert second.trained is False
    drained = loop.consume_completed_sample_learner_result(wait=True)
    assert drained is not None
    assert drained.sampled is True
    assert drained.trained is True
    while loop.has_pending_sample_learner_result:
        loop.consume_completed_sample_learner_result(wait=True)
    assert learner.state == 2
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_sample_learner_worker_kind"] == "local_process"
    assert telemetry["compact_owned_loop_sample_learner_worker_resource_scope"] == "process"
    assert telemetry["compact_owned_loop_sample_learner_worker_start_method"] == "spawn"
    assert (
        telemetry["compact_owned_loop_sample_learner_worker_cuda_ipc_allocator_settings"]
        == "expandable_segments:False"
    )
    assert telemetry["compact_owned_loop_sample_learner_resource_distinct_from_actor_search"] is True
    assert (
        telemetry["compact_owned_loop_sample_learner_hardware_resource_distinct_from_actor_search"]
        is False
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_last_completed_worker_pid"] > 0
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "last_completed_worker_pid_distinct_from_actor_search"
        ]
        is True
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_apply_count"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_last_model_state_applied"] is True
    assert telemetry["compact_owned_loop_deferred_sample_learner_submit_count"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_completed_count"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_request_host_only"] is True
    assert telemetry["compact_owned_loop_deferred_sample_learner_request_cuda_tensor_count"] == 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_result_host_only"] is True
    assert telemetry["compact_owned_loop_deferred_sample_learner_result_cuda_tensor_count"] == 0
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_transport_kind"
        ]
        == "durable_entry_v1"
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_request_bytes"] > 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_result_bytes"] > 0
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_worker_owns_model_state"]
        is True
    )
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_worker_owns_replay_store"]
        is True
    )
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent"]
        is False
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_full_replay_snapshot_submit_count"
        ]
        == 0
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_replay_append_entry_count"] == 2
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_replay_append_index_row_count"
        ]
        == 4
    )
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_replay_append_entry_bytes"]
        > 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_host_observation_bytes"
        ]
        > 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_resident_snapshot_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_last_replay_append_entry_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_last_replay_append_index_row_count"
        ]
        == 2
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_worker_model_initialized_count"
        ]
        == 1
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_worker_completed_count"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_worker_replay_append_count"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_worker_replay_entry_count"] == 2
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_worker_replay_index_row_count"]
        == 4
    )
    loop.close()


def test_compact_owned_loop_can_queue_scalar_ref_replay_append_entries():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    worker = _FakeLocalProcessCaptureWorker()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_replay_append_transport_kind=(
                COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
            ),
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=_FakeLearner(),
        sample_learner_worker=worker,
    )

    loop.prime_previous_step(
        _make_step(
            actions=[0, 1],
            rewards=[1.0, 2.0],
            observation_base=0,
            render_state_snapshot={
                "head_x": np.asarray([[0.0], [1.0]], dtype=np.float32),
            },
        )
    )
    loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=10,
        ),
        index_rows=_make_rows(
            record_index=0,
            actions=[1, 2],
            rewards=[3.0, 4.0],
        ),
    )

    assert len(worker.requests) == 1
    request = worker.requests[0]
    assert request.full_replay_snapshot_sent is False
    assert len(request.provider_bootstrap_steps) == 1
    bootstrap_step = request.provider_bootstrap_steps[0]
    assert bootstrap_step.observation is None
    assert bootstrap_step.resident_observation_replay_snapshot is None
    assert bootstrap_step.render_state_snapshot is not None
    assert len(request.replay_append_entries) == 1
    entry = request.replay_append_entries[0]
    assert entry.previous_step.observation is None
    assert entry.current_step.observation is None
    assert entry.current_step.resident_observation_replay_snapshot is None
    assert entry.index_rows.metadata["compact_replay_append_transport_kind"] == (
        COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
    )
    telemetry = loop.telemetry()
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_transport_kind"
        ]
        == COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_replay_append_entry_count"] == 1
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "provider_bootstrap_step_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "last_provider_bootstrap_step_count"
        ]
        == 1
    )
    worker.close()
    assert worker.closed is True


def test_compact_owned_loop_bounds_scalar_ref_provider_bootstrap_history():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    worker = _FakeLocalProcessCaptureWorker()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_replay_append_transport_kind=(
                COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
            ),
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=_FakeLearner(),
        sample_learner_worker=worker,
    )

    for base in range(5):
        loop.prime_previous_step(
            _make_step(
                actions=[0, 1],
                rewards=[1.0, 2.0],
                observation_base=base,
                render_state_snapshot={
                    "head_x": np.asarray(
                        [[float(base)], [float(base + 10)]],
                        dtype=np.float32,
                    ),
                },
            )
        )
    loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=0, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    request = worker.requests[0]
    assert len(request.provider_bootstrap_steps) == 4
    bootstrap_head_x = [
        float(step.render_state_snapshot["head_x"][0, 0])
        for step in request.provider_bootstrap_steps
    ]
    assert bootstrap_head_x == [1.0, 2.0, 3.0, 4.0]
    assert request.provider_bootstrap_steps[0].observation is None
    assert request.provider_bootstrap_steps[-1].resident_observation_replay_snapshot is None
    assert len(request.replay_append_entries) == 1
    worker.close()


def test_compact_process_worker_materializes_scalar_ref_replay_append_entries():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _StatefulFakeLearner()
    worker = CompactProcessSampleLearnerWorkerV1(
        learner_factory=_make_stateful_fake_learner_for_process,
        learner_factory_kwargs={"initial_state": 0},
        observation_provider_factory=(
            _make_scalar_ref_observation_provider_for_process_test
        ),
        observation_provider_factory_kwargs={
            "previous_bases_by_record_index": {7: 10},
            "current_bases_by_record_index": {7: 20},
        },
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_replay_append_transport_kind=(
                COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
            ),
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
        sample_learner_worker=worker,
    )

    loop.prime_previous_step(
        _make_step(
            actions=[0, 1],
            rewards=[1.0, 2.0],
            observation_base=10,
            render_state_snapshot={
                "head_x": np.asarray([[10.0], [11.0]], dtype=np.float32),
            },
        )
    )
    result = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
            render_state_snapshot={
                "head_x": np.asarray([[20.0], [21.0]], dtype=np.float32),
            },
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert result.sampled is False
    drained = loop.consume_completed_sample_learner_result(wait=True)
    assert drained is not None
    assert drained.sampled is True
    assert drained.trained is True
    assert learner.state == 1
    telemetry = loop.telemetry()
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_transport_kind"
        ]
        == COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_request_host_only"] is True
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_host_observation_bytes"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_render_state_bytes"
        ]
        > 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "provider_bootstrap_step_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "last_provider_bootstrap_step_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "provider_bootstrap_host_observation_bytes"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "provider_bootstrap_resident_snapshot_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "provider_bootstrap_render_state_bytes"
        ]
        > 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "provider_bootstrap_replay_entry_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "provider_bootstrap_replay_index_row_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "provider_bootstrap_learner_call_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_resident_snapshot_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_present"
        ]
        is True
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_bootstrap_step_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "worker_last_observation_provider_bootstrap_step_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_missing_stack_history_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_materialized_entry_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "worker_last_observation_provider_materialized_entry_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_worker_replay_append_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_worker_replay_entry_count"
        ]
        == 1
    )
    assert loop.sample_gate_last_sample_metadata["observation_provider_used"] is False
    loop.close()


def test_compact_process_worker_uses_renderer_backed_scalar_ref_provider():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _StatefulFakeLearner()
    worker = CompactProcessSampleLearnerWorkerV1(
        learner_factory=_make_stateful_fake_learner_for_process,
        learner_factory_kwargs={"initial_state": 0},
        observation_provider_factory=_make_renderer_backed_provider_for_process_test,
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_replay_append_transport_kind=(
                COMPACT_REPLAY_APPEND_TRANSPORT_SCALAR_REF_V1
            ),
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
        sample_learner_worker=worker,
    )

    loop.prime_previous_step(
        _make_step(
            actions=[0, 1],
            rewards=[1.0, 2.0],
            observation_base=10,
            render_state_snapshot={
                "frame_value": np.asarray([[10], [11]], dtype=np.uint8),
            },
        )
    )
    result = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
            render_state_snapshot={
                "frame_value": np.asarray([[20], [21]], dtype=np.uint8),
            },
        ),
        index_rows=_make_rows(record_index=8, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert result.sampled is False
    drained = loop.consume_completed_sample_learner_result(wait=True)
    assert drained is not None
    assert drained.sampled is True
    assert drained.trained is True
    telemetry = loop.telemetry()
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_present"
        ]
        is True
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_materialized_entry_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "worker_observation_provider_missing_stack_history_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_host_observation_bytes"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_"
            "replay_append_render_state_bytes"
        ]
        > 0
    )
    loop.close()


def test_compact_owned_loop_process_sample_learner_can_skip_ordinary_model_state():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _StatefulFakeLearner()
    worker = CompactProcessSampleLearnerWorkerV1()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_gate_max_pending=2,
            defer_sample_learner_model_state_interval=2,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
        sample_learner_worker=worker,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )
    loop.record_step(
        current_step=_make_step(
            actions=[2, 0],
            rewards=[5.0, 6.0],
            observation_base=30,
        ),
        index_rows=_make_rows(record_index=8, actions=[2, 0], rewards=[5.0, 6.0]),
    )

    first = loop.consume_completed_sample_learner_result(wait=True)
    assert first is not None
    assert learner.state == 0
    second = loop.consume_completed_sample_learner_result(wait=True)
    assert second is not None
    assert learner.state == 2
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_interval"] == 2
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_return_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_omitted_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_apply_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_last_model_state_returned"] is True
    assert telemetry["compact_owned_loop_deferred_sample_learner_last_model_state_applied"] is True
    loop.close()


def test_compact_owned_loop_process_sample_learner_model_state_snapshot_file_transport():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _StatefulFakeLearner()
    worker = CompactProcessSampleLearnerWorkerV1()
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_gate_max_pending=2,
            defer_sample_learner_model_state_interval=2,
            defer_sample_learner_model_state_transport_kind=(
                COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1
            ),
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
        sample_learner_worker=worker,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )
    loop.record_step(
        current_step=_make_step(
            actions=[2, 0],
            rewards=[5.0, 6.0],
            observation_base=30,
        ),
        index_rows=_make_rows(record_index=8, actions=[2, 0], rewards=[5.0, 6.0]),
    )

    first = loop.consume_completed_sample_learner_result(wait=True)
    assert first is not None
    assert learner.state == 0
    second = loop.consume_completed_sample_learner_result(wait=True)
    assert second is not None
    assert learner.state == 2
    assert second.learner_result is not None
    assert _DEFERRED_LEARNER_MODEL_STATE_DICT_KEY not in second.learner_result
    snapshot = second.learner_result[_DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY]
    assert int(snapshot["bytes"]) > 0
    assert not os.path.exists(str(snapshot["path"]))

    telemetry = loop.telemetry()
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_model_state_transport_kind"
        ]
        == COMPACT_MODEL_STATE_TRANSPORT_SNAPSHOT_FILE_V1
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_return_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_omitted_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_apply_count"] == 1
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_model_state_snapshot_return_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_model_state_snapshot_publish_bytes"
        ]
        > 0
    )
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_count"]
        == 1
    )
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_model_state_snapshot_load_bytes"]
        > 0
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_worker_job_wall_sec"] > 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_worker_sample_sec"] > 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_worker_learner_sec"] > 0
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_worker_model_state_clone_sec"
        ]
        >= 0
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_worker_result_pickle_sec"] > 0
    loop.close()


def test_compact_owned_loop_process_sample_learner_owner_ref_avoids_state_clone():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _OwnerRefFakeLearner()
    worker = CompactProcessSampleLearnerWorkerV1(
        learner_factory=_make_owner_ref_fake_learner_for_process,
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_gate_max_pending=1,
            defer_sample_learner_model_state_interval=1,
            defer_sample_learner_model_state_transport_kind=(
                COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1
            ),
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
        sample_learner_worker=worker,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    result = loop.consume_completed_sample_learner_result(wait=True)
    assert result is not None
    assert result.learner_result is not None
    assert learner.state == 0
    assert _DEFERRED_LEARNER_MODEL_STATE_DICT_KEY not in result.learner_result
    assert _DEFERRED_LEARNER_MODEL_STATE_SNAPSHOT_KEY not in result.learner_result
    owner_ref = result.learner_result[_DEFERRED_LEARNER_MODEL_OWNER_REF_KEY]
    assert owner_ref["transport_kind"] == COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1
    assert owner_ref["model_state_digest"] == "owner-ref-state-1"
    assert int(owner_ref["model_object_id"]) > 0

    telemetry = loop.telemetry()
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_model_state_transport_kind"
        ]
        == COMPACT_MODEL_STATE_TRANSPORT_OWNER_REF_V1
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_return_count"] == 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_omitted_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_state_apply_count"] == 0
    assert telemetry["compact_owned_loop_deferred_sample_learner_last_model_state_applied"] is False
    assert telemetry["compact_owned_loop_deferred_sample_learner_model_owner_ref_return_count"] == 1
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_last_model_owner_ref_returned"
        ]
        is True
    )
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_last_model_owner_ref_digest"]
        == "owner-ref-state-1"
    )
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_last_model_owner_ref_worker_pid"]
        > 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_model_state_snapshot_return_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_worker_model_state_fn_sec"
        ]
        == 0.0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_worker_model_state_clone_sec"
        ]
        == 0.0
    )
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_worker_model_state_digest_sec"
        ]
        > 0.0
    )
    loop.close()


def test_compact_owned_loop_model_state_return_uses_completed_update_window():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_model_state_interval=2,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
    )

    assert loop._deferred_sample_learner_should_return_model_state(1) is False
    loop.learner_gate_updates = 1
    assert loop._deferred_sample_learner_should_return_model_state(2) is True
    loop.close()


def test_compact_owned_loop_model_state_return_skips_pending_update_window():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_model_state_interval=4,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
    )

    loop.learner_gate_updates = 2
    loop._pending_sample_learner_futures.append(
        CompactPendingSampleLearnerWorkV1(
            request_id=3,
            handle=Future(),
            submitted_at=0.0,
            snapshot_version=3,
        )
    )
    assert loop._deferred_sample_learner_should_return_model_state(4) is True

    loop.learner_gate_updates = 3
    loop._pending_sample_learner_futures.clear()
    loop._pending_sample_learner_futures.append(
        CompactPendingSampleLearnerWorkV1(
            request_id=4,
            handle=Future(),
            submitted_at=0.0,
            snapshot_version=4,
        )
    )
    assert loop._deferred_sample_learner_should_return_model_state(5) is False
    loop._pending_sample_learner_futures.clear()
    loop.close()


def test_compact_owned_loop_model_state_force_reuses_pending_return():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
            defer_sample_learner_model_state_interval=4,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
    )

    loop.learner_gate_updates = 4
    loop._pending_sample_learner_futures.append(
        CompactPendingSampleLearnerWorkV1(
            request_id=4,
            handle=Future(),
            submitted_at=0.0,
            snapshot_version=4,
            return_model_state=True,
        )
    )
    loop.force_next_sample_learner_model_state()
    assert loop._deferred_sample_learner_should_return_model_state(5) is False

    loop._pending_sample_learner_futures.clear()
    loop.force_next_sample_learner_model_state()
    assert loop._deferred_sample_learner_should_return_model_state(6) is True
    loop.close()


def test_compact_process_sample_learner_worker_rejects_cuda_learner_bootstrap():
    class _FakeCudaTensor:
        is_cuda = True

    class _FakeCudaStateLearner(_FakeLearner):
        def model_state_dict(self):
            return {"weight": _FakeCudaTensor()}

    worker = CompactProcessSampleLearnerWorkerV1()
    try:
        with pytest.raises(RuntimeError, match="cannot bootstrap from CUDA learner state"):
            worker.prepare(
                replay_store=_CompactReplayRingV1(capacity=4),
                learner=_FakeCudaStateLearner(),
            )
    finally:
        worker.close()


def test_compact_process_sample_learner_worker_factory_bootstrap_owns_state():
    class _FakeCudaTensor:
        is_cuda = True

    class _FakeCudaStateLearner(_StatefulFakeLearner):
        def model_state_dict(self):
            return {"weight": _FakeCudaTensor()}

    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _FakeCudaStateLearner()
    worker = CompactProcessSampleLearnerWorkerV1(
        learner_factory=_make_stateful_fake_learner_for_process,
        learner_factory_kwargs={"initial_state": 0},
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(
            capture_replay_store_state=False,
            defer_sample_learner_gate=True,
        ),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
        learner=learner,
        sample_learner_worker=worker,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0], observation_base=10))

    result = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert result.sampled is False
    drained = loop.consume_completed_sample_learner_result(wait=True)
    assert drained is not None
    assert learner.state == 1
    telemetry = loop.telemetry()
    assert telemetry["compact_owned_loop_sample_learner_worker_bootstrap_source"] == "factory"
    assert telemetry["compact_owned_loop_deferred_sample_learner_request_host_only"] is True
    assert telemetry["compact_owned_loop_deferred_sample_learner_result_host_only"] is True
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_worker_owns_model_state"]
        is True
    )
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_worker_owns_replay_store"]
        is True
    )
    assert (
        telemetry["compact_owned_loop_deferred_sample_learner_full_replay_snapshot_sent"]
        is False
    )
    assert telemetry["compact_owned_loop_deferred_sample_learner_replay_append_entry_count"] == 1
    assert telemetry["compact_owned_loop_deferred_sample_learner_worker_replay_append_count"] == 1
    assert (
        telemetry[
            "compact_owned_loop_deferred_sample_learner_worker_model_initialized_count"
        ]
        == 1
    )
    loop.close()


def test_compact_process_sample_learner_worker_rejects_cuda_tensor_payload():
    class _FakeCudaTensor:
        is_cuda = True

    worker = CompactProcessSampleLearnerWorkerV1()
    request = CompactSampleLearnerWorkRequestV1(
        request_id=1,
        replay_store=SimpleNamespace(),
        replay_snapshot=SimpleNamespace(value=_FakeCudaTensor()),
        learner=_FakeLearner(),
        seed=1,
        sample_batch_size=1,
        require_next_targets=False,
        num_unroll_steps=1,
        fused_learner_batch=False,
        train_steps=1,
        policy_version_ref="unit-policy",
        model_version_ref="unit-model",
        policy_source="unit-test",
    )
    try:
        with pytest.raises(RuntimeError, match="cannot receive CUDA tensors"):
            worker.submit(request)
    finally:
        worker.close()


def test_compact_owned_loop_can_train_from_fused_learner_batch():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _FakeFusedLearner()
    loop = CompactOwnedLoopV1(
        config=CompactOwnedLoopConfigV1(
            sample_batch_size=2,
            sample_interval=1,
            replay_capacity=4,
            learner_train_steps=1,
            num_unroll_steps=2,
            sample_seed_base=31,
            learner_impl="compact_muzero",
            require_next_targets=True,
            fused_learner_batch=True,
        ),
        policy_version=policy,
        replay_store=_FakeFusedReplayStore(),
        learner=learner,
    )

    result = loop.sample_and_train_from_store()

    assert result.trained is True
    assert learner.used_learner_batch is True
    assert (
        learner.last_learner_batch_metadata[
            "compact_replay_fixed_soa_learner_batch_handle_ring_used"
        ]
        is True
    )
    assert result.sample_result is not None
    assert result.sample_result["sample_batch"] is None
    assert (
        loop.sample_gate_last_sample_metadata["resident_grouped_device_direct_write_learner_batch"]
        is True
    )
    assert (
        loop.learner_gate_last_telemetry["compact_rollout_slab_learner_gate_prebuilt_batch_used"]
        is True
    )
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_requested"
        ]
        is True
    )
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_consumed"
        ]
        is True
    )
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_handle_id"
        ]
        == 7
    )
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_fallback_count"
        ]
        == 0
    )
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_materialized_parent_fallback_count"
        ]
        == 0
    )
    assert loop.telemetry()["compact_owned_loop_fused_learner_batch"] is True
    assert (
        loop.telemetry()[
            "compact_owned_loop_learner_resident_batch_handle_requested_count"
        ]
        == 1
    )
    assert (
        loop.telemetry()[
            "compact_owned_loop_learner_resident_batch_handle_consumed_count"
        ]
        == 1
    )
    assert (
        loop.telemetry()[
            "compact_owned_loop_learner_resident_batch_handle_fallback_count"
        ]
        == 0
    )


def test_compact_owned_loop_resident_handle_proof_fails_closed_on_fallback():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
        model_version_ref="unit-owned-model-v1",
    )
    learner = _FakeFusedLearner()
    loop = CompactOwnedLoopV1(
        config=CompactOwnedLoopConfigV1(
            sample_batch_size=2,
            sample_interval=1,
            replay_capacity=4,
            learner_train_steps=1,
            num_unroll_steps=2,
            sample_seed_base=31,
            learner_impl="compact_muzero",
            require_next_targets=True,
            fused_learner_batch=True,
        ),
        policy_version=policy,
        replay_store=_FakeFusedReplayStore(
            handle_used=False,
            fallback_count=1,
            fallback_reason="handle_resolve_failed",
        ),
        learner=learner,
    )

    result = loop.sample_and_train_from_store()

    assert result.trained is True
    assert learner.used_learner_batch is True
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_requested"
        ]
        is True
    )
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_consumed"
        ]
        is False
    )
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_fallback_count"
        ]
        == 1
    )
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_fallback_reason"
        ]
        == "handle_resolve_failed"
    )
    assert (
        loop.learner_gate_last_telemetry[
            "compact_owned_loop_learner_resident_batch_handle_materialized_parent_fallback_count"
        ]
        == 1
    )
    telemetry = loop.telemetry()
    assert (
        telemetry[
            "compact_owned_loop_learner_resident_batch_handle_requested_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_learner_resident_batch_handle_consumed_count"
        ]
        == 0
    )
    assert (
        telemetry[
            "compact_owned_loop_learner_resident_batch_handle_fallback_count"
        ]
        == 1
    )
    assert (
        telemetry[
            "compact_owned_loop_learner_resident_batch_handle_last_consumed"
        ]
        is False
    )
    assert (
        telemetry[
            "compact_owned_loop_learner_resident_batch_handle_last_fallback_reason"
        ]
        == "handle_resolve_failed"
    )


def test_compact_owned_loop_requires_explicit_policy_lineage():
    with pytest.raises(ValueError, match="policy_version_ref"):
        CompactOwnedLoopV1(
            config=_loop_config(capture_replay_store_state=False),
            policy_version=CompactPolicyVersionRefV1(
                policy_version_ref="",
                policy_source="unit_test_compact_owned_loop",
            ),
            replay_store=_CompactReplayRingV1(capacity=2),
        )

    with pytest.raises(ValueError, match="policy_source"):
        CompactOwnedLoopV1(
            config=_loop_config(capture_replay_store_state=False),
            policy_version=CompactPolicyVersionRefV1(
                policy_version_ref="unit-owned-policy-v1",
                policy_source="",
            ),
            replay_store=_CompactReplayRingV1(capacity=2),
        )


def test_compact_owned_loop_rejects_unknown_replay_append_transport_kind():
    with pytest.raises(ValueError, match="replay_append_transport_kind"):
        CompactOwnedLoopV1(
            config=_loop_config(
                capture_replay_store_state=False,
                defer_sample_learner_replay_append_transport_kind="mystery",
            ),
            policy_version=CompactPolicyVersionRefV1(
                policy_version_ref="unit-owned-policy-v1",
                policy_source="unit_test_compact_owned_loop",
            ),
            replay_store=_CompactReplayRingV1(capacity=2),
        )


def test_compact_owned_loop_primes_first_no_warmup_step_without_append():
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-owned-policy-v1",
        policy_source="unit_test_compact_owned_loop",
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(capture_replay_store_state=False),
        policy_version=policy,
        replay_store=_CompactReplayRingV1(
            capacity=4,
            metadata=compact_owned_loop_replay_store_metadata(policy),
        ),
    )

    first = loop.record_step(
        current_step=_make_step(
            actions=[1, 2],
            rewards=[3.0, 4.0],
            observation_base=20,
        ),
        index_rows=_make_rows(record_index=0, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert first.appended_replay_rows is False
    assert first.sampled is False
    assert first.trained is False
    assert loop.replay_store.entry_count == 0

    second = loop.record_step(
        current_step=_make_step(
            actions=[2, 0],
            rewards=[5.0, 6.0],
            observation_base=30,
        ),
        index_rows=_make_rows(record_index=1, actions=[2, 0], rewards=[5.0, 6.0]),
    )

    assert second.appended_replay_rows is True
    assert second.sampled is True
    assert loop.replay_store.entry_count == 1


def _loop_config(
    *,
    capture_replay_store_state: bool,
    defer_learner_gate: bool = False,
    defer_sample_learner_gate: bool = False,
    defer_sample_learner_gate_max_pending: int = 1,
    defer_sample_learner_model_state_interval: int = 1,
    defer_sample_learner_model_state_transport_kind: str = "result_v1",
    defer_sample_learner_replay_append_transport_kind: str = "durable_entry_v1",
) -> CompactOwnedLoopConfigV1:
    return CompactOwnedLoopConfigV1(
        sample_batch_size=2,
        sample_interval=1,
        replay_capacity=4,
        learner_train_steps=1,
        num_unroll_steps=1,
        sample_seed_base=31,
        learner_impl="unit_test",
        require_next_targets=False,
        capture_replay_store_state=bool(capture_replay_store_state),
        defer_learner_gate=bool(defer_learner_gate),
        defer_sample_learner_gate=bool(defer_sample_learner_gate),
        defer_sample_learner_gate_max_pending=int(defer_sample_learner_gate_max_pending),
        defer_sample_learner_model_state_interval=int(
            defer_sample_learner_model_state_interval
        ),
        defer_sample_learner_model_state_transport_kind=str(
            defer_sample_learner_model_state_transport_kind
        ),
        defer_sample_learner_replay_append_transport_kind=str(
            defer_sample_learner_replay_append_transport_kind
        ),
    )


def _make_step(
    *,
    actions: list[int],
    rewards: list[float],
    observation_base: int,
    render_state_snapshot: dict[str, np.ndarray] | None = None,
    autoreset_render_state_snapshot: dict[str, np.ndarray] | None = None,
) -> SimpleNamespace:
    observation = np.zeros((2, 1, 4, 64, 64), dtype=np.float32)
    observation[0, 0] = float(observation_base)
    observation[1, 0] = float(observation_base + 1)
    return SimpleNamespace(
        observation=observation,
        action_mask=np.ones((2, 1, ACTION_COUNT), dtype=np.bool_),
        reward=np.asarray([[rewards[0]], [rewards[1]]], dtype=np.float32),
        final_reward_map=np.asarray([[rewards[0]], [rewards[1]]], dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        payload={
            "joint_action": np.asarray(
                [[actions[0]], [actions[1]]],
                dtype=np.int16,
            ),
        },
        compact_batch=None,
        render_state_snapshot=render_state_snapshot,
        autoreset_render_state_snapshot=autoreset_render_state_snapshot,
    )


def _make_rows(
    *,
    record_index: int,
    actions: list[int],
    rewards: list[float],
) -> CompactReplayIndexRowsV1:
    return CompactReplayIndexRowsV1(
        metadata={"policy_version_ref": "unit-owned-policy-v1"},
        record_index=record_index,
        next_record_index=record_index + 1,
        compact_root_row=np.asarray([0, 1], dtype=np.int32),
        policy_env_id=np.asarray([record_index * 10, record_index * 10 + 1]),
        policy_row=np.asarray([0, 1], dtype=np.int32),
        env_row=np.asarray([0, 1], dtype=np.int32),
        player=np.asarray([0, 0], dtype=np.int16),
        action=np.asarray(actions, dtype=np.int16),
        action_mask=np.ones((2, ACTION_COUNT), dtype=np.bool_),
        policy_target=np.eye(ACTION_COUNT, dtype=np.float32)[np.asarray(actions)],
        root_value=np.asarray([0.0, 0.0], dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        final_reward=np.asarray(rewards, dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        terminated=np.asarray([False, False], dtype=np.bool_),
        truncated=np.asarray([False, False], dtype=np.bool_),
        next_final_observation_row=np.asarray([False, False], dtype=np.bool_),
        to_play=np.asarray([-1, -1], dtype=np.int64),
        policy_source="unit_test_compact_owned_loop",
    )


def _observation_for_process_provider_test(observation_base: int) -> np.ndarray:
    observation = np.zeros((2, 1, 4, 64, 64), dtype=np.float32)
    observation[0, 0] = float(observation_base)
    observation[1, 0] = float(observation_base + 1)
    return observation


class _ScalarRefObservationProviderForProcessTest:
    def __init__(
        self,
        *,
        previous_bases_by_record_index: dict[int, int],
        current_bases_by_record_index: dict[int, int],
    ) -> None:
        self.previous_bases_by_record_index = {
            int(key): int(value)
            for key, value in dict(previous_bases_by_record_index).items()
        }
        self.current_bases_by_record_index = {
            int(key): int(value)
            for key, value in dict(current_bases_by_record_index).items()
        }
        self.bootstrap_step_count = 0
        self.missing_stack_history_count = 0

    def bootstrap_compact_replay_step(self, step) -> None:
        if getattr(step, "observation", None) is not None:
            raise RuntimeError("bootstrap step must not carry observation tensors")
        if getattr(step, "resident_observation_replay_snapshot", None) is not None:
            raise RuntimeError("bootstrap step must not carry resident snapshots")
        self.bootstrap_step_count += 1

    def materialize_compact_replay_entry(self, entry):
        if self.bootstrap_step_count <= 0:
            self.missing_stack_history_count += 1
            raise RuntimeError("missing provider bootstrap history")
        record_index = int(getattr(entry.index_rows, "record_index"))
        previous_step = replace(
            entry.previous_step,
            observation=_observation_for_process_provider_test(
                self.previous_bases_by_record_index[record_index]
            ),
        )
        current_step = replace(
            entry.current_step,
            observation=_observation_for_process_provider_test(
                self.current_bases_by_record_index[record_index]
            ),
        )
        return replace(
            entry,
            previous_step=previous_step,
            current_step=current_step,
        )


def _make_scalar_ref_observation_provider_for_process_test(
    *,
    previous_bases_by_record_index: dict[int, int],
    current_bases_by_record_index: dict[int, int],
) -> _ScalarRefObservationProviderForProcessTest:
    return _ScalarRefObservationProviderForProcessTest(
        previous_bases_by_record_index=previous_bases_by_record_index,
        current_bases_by_record_index=current_bases_by_record_index,
    )


class _FrameValueRendererForProcessTest:
    backend_name = "unit_test_process_frame_value_renderer"

    def render(self, request):
        values = np.asarray(request.state["frame_value"], dtype=np.uint8)
        out = np.asarray(request.out)
        for output_row, (state_row, player) in enumerate(
            zip(request.row_indices, request.controlled_players, strict=True)
        ):
            out[output_row, 0].fill(values[int(state_row), int(player)])
        return SourceStateBatchedRenderResult(
            frames=out,
            telemetry={"unit_test_rendered_frame_count": float(out.shape[0])},
        )


def _make_renderer_backed_provider_for_process_test():
    return CompactReplayRendererBackedObservationProviderV1(
        batch_size=2,
        player_count=1,
        renderer=_FrameValueRendererForProcessTest(),
    )


class _FakeLearner:
    def train_on_sample_batch(
        self,
        sample_batch: object,
        *,
        train_steps: int,
    ) -> dict[str, object]:
        del sample_batch
        return {
            "compact_rollout_slab_learner_gate_updates": int(train_steps),
            "compact_rollout_slab_learner_gate_sample_rows": 2,
            "compact_rollout_slab_learner_gate_input_bytes": 0,
            "compact_rollout_slab_learner_gate_sec": 0.01,
            "compact_muzero_learner_update_claim": True,
        }


class _StatefulFakeLearner(_FakeLearner):
    def __init__(self) -> None:
        self.state = 0

    def train_on_sample_batch(
        self,
        sample_batch: object,
        *,
        train_steps: int,
    ) -> dict[str, object]:
        self.state += int(train_steps)
        return super().train_on_sample_batch(sample_batch, train_steps=train_steps)

    def model_state_dict(self) -> dict[str, int]:
        return {"state": int(self.state)}

    def load_model_state_dict(self, state_dict: object) -> None:
        self.state = int(dict(state_dict)["state"])


class _OwnerRefFakeLearner(_StatefulFakeLearner):
    def model_state_dict(self) -> dict[str, int]:
        raise AssertionError("owner_ref_v1 must not clone model_state_dict")

    def load_model_state_dict(self, state_dict: object) -> None:
        raise AssertionError("owner_ref_v1 must not apply state in the parent")

    def model_state_digest(self) -> str:
        return f"owner-ref-state-{int(self.state)}"

    def model_object_id(self) -> int:
        return id(self)


def _make_stateful_fake_learner_for_process(initial_state: int = 0) -> _StatefulFakeLearner:
    learner = _StatefulFakeLearner()
    learner.state = int(initial_state)
    return learner


def _make_owner_ref_fake_learner_for_process() -> _OwnerRefFakeLearner:
    return _OwnerRefFakeLearner()


class _FakeFusedLearner(_FakeLearner):
    def __init__(self) -> None:
        self.used_learner_batch = False
        self.last_learner_batch_metadata = {}

    def train_on_learner_batch(
        self,
        learner_batch: object,
        *,
        train_steps: int,
    ) -> dict[str, object]:
        self.used_learner_batch = True
        self.last_learner_batch_metadata = dict(
            getattr(learner_batch, "metadata", {}) or {}
        )
        result = super().train_on_sample_batch(object(), train_steps=train_steps)
        result["compact_rollout_slab_learner_gate_prebuilt_batch_used"] = True
        return result


class _FakeFusedReplayStore:
    capacity = 4
    entry_count = 0
    stored_index_row_count = 0
    evicted_entry_count = 0
    evicted_index_row_count = 0

    def __init__(
        self,
        *,
        handle_used: bool = True,
        fallback_count: int = 0,
        fallback_reason: str = "none",
    ) -> None:
        self.handle_used = bool(handle_used)
        self.fallback_count = int(fallback_count)
        self.fallback_reason = str(fallback_reason)

    def sample(self, **kwargs):
        assert kwargs["build_compact_muzero_learner_batch"] is True
        assert kwargs["compact_muzero_learner_batch_only"] is True
        handle_metadata = {
            "compact_replay_fixed_soa_learner_batch_handle_ring_schema_id": (
                "curvyzero_compact_replay_fixed_soa_learner_batch_handle/v1"
            ),
            "compact_replay_fixed_soa_learner_batch_handle_ring_requested": True,
            "compact_replay_fixed_soa_learner_batch_handle_ring_used": self.handle_used,
            "compact_replay_fixed_soa_learner_batch_handle_ring_handle_id": 7,
            "compact_replay_fixed_soa_learner_batch_handle_ring_snapshot_version": 3,
            "compact_replay_fixed_soa_learner_batch_handle_ring_request_checksum": 12345,
            "compact_replay_fixed_soa_learner_batch_handle_ring_sample_row_count": 2,
            "compact_replay_fixed_soa_learner_batch_handle_ring_target_row_count": 2,
            "compact_replay_fixed_soa_learner_batch_handle_ring_fallback_count": (
                self.fallback_count
            ),
            "compact_replay_fixed_soa_learner_batch_handle_ring_fallback_reason": (
                self.fallback_reason
            ),
        }
        return {
            "sec": 0.01,
            "index_row_count": 2,
            "target_row_count": 2,
            "sample_row_count": 2,
            "sample_batch": None,
            "resident_sample_batch": None,
            "learner_batch": SimpleNamespace(
                metadata={
                    "resident_grouped_device_direct_write_learner_batch": True,
                    **handle_metadata,
                }
            ),
            "sample_metadata": {
                "resident_grouped_device_direct_write_learner_batch": True,
                **handle_metadata,
            },
            "telemetry": {
                "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
            },
        }


class _FakeExternalSampleLearnerWorker:
    metadata = {
        "compact_owned_loop_sample_learner_worker_kind": "external_fake",
        "compact_owned_loop_sample_learner_worker_resource_id": "fake-worker",
        "compact_owned_loop_actor_search_resource_id": "fake-actor-search",
        "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": True,
    }

    def __init__(self) -> None:
        self.requests = []
        self.closed = False

    def submit(self, request):
        self.requests.append(request)
        return len(self.requests) - 1

    def done(self, handle) -> bool:
        return False

    def result(self, handle):
        request = self.requests[int(handle)]
        return {
            "sample_result": request.replay_store.sample_from_snapshot(
                request.replay_snapshot,
                seed=request.seed,
                sample_batch_size=request.sample_batch_size,
                require_next_targets=request.require_next_targets,
                num_unroll_steps=request.num_unroll_steps,
                build_compact_muzero_learner_batch=request.fused_learner_batch,
                compact_muzero_learner_batch_only=request.fused_learner_batch,
            ),
            "learner_result": request.learner.train_on_sample_batch(
                object(),
                train_steps=request.train_steps,
            ),
        }

    def close(self) -> None:
        self.closed = True


class _FakeLocalProcessCaptureWorker:
    metadata = {
        "compact_owned_loop_sample_learner_worker_kind": (
            COMPACT_SAMPLE_LEARNER_WORKER_LOCAL_PROCESS
        ),
        "compact_owned_loop_sample_learner_worker_resource_id": "fake-process-worker",
        "compact_owned_loop_actor_search_resource_id": "fake-actor-search",
        "compact_owned_loop_sample_learner_resource_distinct_from_actor_search": True,
    }

    def __init__(self) -> None:
        self.requests = []
        self.closed = False

    def submit(self, request):
        self.requests.append(request)
        return len(self.requests) - 1

    def done(self, handle) -> bool:
        del handle
        return False

    def close(self) -> None:
        self.closed = True


class _BlockingFakeLearner(_FakeLearner):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def train_on_sample_batch(
        self,
        sample_batch: object,
        *,
        train_steps: int,
    ) -> dict[str, object]:
        self.started.set()
        assert self.release.wait(timeout=2.0)
        return super().train_on_sample_batch(sample_batch, train_steps=train_steps)
