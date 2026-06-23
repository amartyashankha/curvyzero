from curvyzero.training.compact_split_transport_mock import (
    MOCK_TRANSPORT_KIND_HOST_ONLY,
)
from curvyzero.training.compact_split_transport_mock import (
    MOCK_TRANSPORT_KIND_OWNER_SEARCH,
)
from curvyzero.training.compact_split_transport_mock import MOCK_TRANSPORT_SCHEMA_ID
from curvyzero.training.compact_split_transport_mock import (
    run_host_only_split_transport_mock,
)
from curvyzero.training.compact_split_transport_mock import (
    run_owner_search_split_transport_mock,
)


def test_host_only_split_transport_mock_proves_run_ahead_and_final_drain():
    report = run_host_only_split_transport_mock(
        steps=8,
        sample_interval=1,
        max_pending=2,
        sample_batch_size=3,
        train_steps=1,
        worker_delay_sec=0.005,
    ).to_dict()

    assert report["schema_id"] == MOCK_TRANSPORT_SCHEMA_ID
    assert report["transport_kind"] == MOCK_TRANSPORT_KIND_HOST_ONLY
    assert report["ok"] is True
    assert report["submit_count"] == 8
    assert report["completed_count"] == report["submit_count"]
    assert report["last_completed_request_id"] == report["completed_count"]
    assert report["actor_steps_while_pending"] > 0
    assert report["policy_lag_max"] > 0
    assert report["pending_count_at_end"] == 0
    assert report["final_drain_in_wall_sec"] is True
    assert report["cuda_tensor_payload_count"] == 0
    assert report["worker_pid_distinct_from_actor"] is True
    assert report["worker_owns_replay_state"] is True
    assert report["worker_owns_model_state"] is True
    assert report["final_model_version"] == report["completed_count"]


def test_owner_search_split_transport_mock_keeps_model_state_with_owner():
    report = run_owner_search_split_transport_mock(
        steps=8,
        sample_interval=2,
        max_pending=2,
        roots_per_step=3,
        sample_batch_size=3,
        train_steps=1,
        worker_delay_sec=0.005,
    ).to_dict()

    assert report["schema_id"] == MOCK_TRANSPORT_SCHEMA_ID
    assert report["transport_kind"] == MOCK_TRANSPORT_KIND_OWNER_SEARCH
    assert report["ok"] is True
    assert report["search_request_count"] == 9
    assert report["search_result_count"] == report["search_request_count"]
    assert report["last_completed_request_id"] == report["final_refresh_request_id"]
    assert report["actor_steps_while_pending"] > 0
    assert report["policy_lag_max"] > 0
    assert report["pending_count_at_end"] == 0
    assert report["final_drain_in_wall_sec"] is True
    assert report["cuda_tensor_payload_count"] == 0
    assert report["root_observation_bytes_sent"] == 0
    assert report["model_state_bytes_total"] == 0
    assert report["model_state_return_count"] == 0
    assert report["owner_ref_result_count"] == report["search_result_count"]
    assert report["worker_pid_distinct_from_actor"] is True
    assert report["worker_owns_replay_state"] is True
    assert report["worker_owns_model_state"] is True
    assert report["worker_owns_search_state"] is True
    assert report["final_model_version"] == 4
    assert report["final_search_model_version"] == report["final_model_version"]
    assert report["search_consumed_final_update"] is True
