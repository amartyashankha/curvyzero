from __future__ import annotations

import numpy as np
import pytest

from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import (
    validate_compact_search_result_identity_v1,
)
from curvyzero.training.compact_search_service import CompactSearchServiceV1
from curvyzero.training.compact_search_service import CompactSearchTwoPhaseServiceV1
from curvyzero.training.compact_search_service import (
    compact_search_result_v1_from_two_phase_payloads,
)
from curvyzero.training.compact_search_service import (
    validate_compact_device_search_two_phase_payload_v1,
)
from curvyzero.training.compact_search_service import (
    validate_compact_search_two_phase_payload_v1,
)
from curvyzero.training.fixed_shape_batched_search_owner import (
    FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL,
)
from curvyzero.training.fixed_shape_batched_search_owner import (
    FIXED_SHAPE_BATCHED_SEARCH_OWNER_SEMANTICS,
)
from curvyzero.training.fixed_shape_batched_search_owner import (
    FixedShapeBatchedSearchOwnerV0,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


def test_fixed_shape_batched_search_owner_returns_valid_compact_result() -> None:
    root_batch = _root_batch(
        legal_mask=np.asarray(
            [
                [False, True, True],
                [True, True, True],
                [True, False, True],
                [False, False, False],
            ],
            dtype=np.bool_,
        ),
        active_root_mask=np.asarray([True, False, True, False], dtype=np.bool_),
    )
    service = FixedShapeBatchedSearchOwnerV0(root_count=4, num_simulations=7)

    result = service.run(root_batch)

    assert isinstance(service, CompactSearchServiceV1)
    validate_compact_search_result_identity_v1(root_batch, result)
    assert result.metadata["search_impl"] == FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL
    assert result.metadata["profile_backend"] == FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL
    assert result.metadata["profile_only"] is True
    assert result.metadata["profile_semantics"] == FIXED_SHAPE_BATCHED_SEARCH_OWNER_SEMANTICS
    np.testing.assert_array_equal(result.root_index, np.asarray([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(result.env_row, np.asarray([10, 12], dtype=np.int32))
    np.testing.assert_array_equal(result.player, np.asarray([0, 2], dtype=np.int16))
    np.testing.assert_array_equal(result.policy_env_id, np.asarray([100, 102], dtype=np.int64))
    np.testing.assert_array_equal(result.selected_action, np.asarray([1, 0], dtype=np.int16))
    np.testing.assert_allclose(
        result.visit_policy,
        np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result.raw_visit_counts,
        np.asarray([[0.0, 7.0, 0.0], [7.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(result.root_value, np.zeros((2,), dtype=np.float32))


def test_fixed_shape_batched_search_owner_masks_inactive_roots_and_emits_hard_telemetry() -> None:
    root_batch = _root_batch(
        legal_mask=np.asarray(
            [
                [False, True, False],
                [False, False, True],
                [False, False, False],
            ],
            dtype=np.bool_,
        ),
        active_root_mask=np.asarray([False, True, False], dtype=np.bool_),
    )

    result = FixedShapeBatchedSearchOwnerV0(root_count=3, num_simulations=5).run(
        root_batch
    )

    np.testing.assert_array_equal(result.root_index, np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(result.selected_action, np.asarray([2], dtype=np.int16))
    assert result.metadata["ctree_calls"] == 0
    assert result.metadata["tolist_calls"] == 0
    assert result.metadata["per_sim_d2h_bytes"] == 0
    telemetry = result.metadata["profile_telemetry"]
    assert telemetry["fixed_shape_batched_search_owner_ctree_calls"] == 0
    assert telemetry["fixed_shape_batched_search_owner_tolist_calls"] == 0
    assert telemetry["fixed_shape_batched_search_owner_per_sim_d2h_bytes"] == 0
    assert telemetry["fixed_shape_batched_search_owner_padded_root_count"] == 3
    assert telemetry["fixed_shape_batched_search_owner_active_root_count"] == 1
    assert telemetry["fixed_shape_batched_search_owner_masked_inactive_root_count"] == 2
    assert telemetry["fixed_shape_batched_search_owner_requested_simulations"] == 5
    assert telemetry["fixed_shape_batched_search_owner_actual_search_simulations"] == 0
    assert telemetry["fixed_shape_batched_search_owner_d2h_bytes"] == (
        result.selected_action.nbytes
        + result.visit_policy.nbytes
        + result.raw_visit_counts.nbytes
        + result.root_value.nbytes
    )


def test_fixed_shape_batched_search_owner_zero_active_roots_returns_empty_result() -> None:
    root_batch = _root_batch(
        legal_mask=np.zeros((2, 3), dtype=np.bool_),
        active_root_mask=np.zeros((2,), dtype=np.bool_),
    )

    result = FixedShapeBatchedSearchOwnerV0(root_count=2, num_simulations=3).run(
        root_batch
    )

    assert result.selected_action.shape == (0,)
    assert result.visit_policy.shape == (0, 3)
    assert result.raw_visit_counts is not None
    assert result.raw_visit_counts.shape == (0, 3)
    assert result.metadata["profile_telemetry"][
        "fixed_shape_batched_search_owner_active_root_count"
    ] == 0
    assert result.metadata["ctree_calls"] == 0
    assert result.metadata["tolist_calls"] == 0
    assert result.metadata["per_sim_d2h_bytes"] == 0


def test_fixed_shape_batched_search_owner_rejects_shape_and_active_legality_errors() -> None:
    service = FixedShapeBatchedSearchOwnerV0(root_count=2, num_simulations=3)

    with pytest.raises(ReplayCompatibilityError, match="expected legal_mask shape"):
        service.run(
            _root_batch(
                legal_mask=np.ones((3, 3), dtype=np.bool_),
                active_root_mask=np.ones((3,), dtype=np.bool_),
            )
        )

    with pytest.raises(ReplayCompatibilityError, match="active root with no legal action"):
        service.run(
            _root_batch(
                legal_mask=np.asarray(
                    [[False, False, False], [True, False, False]],
                    dtype=np.bool_,
                ),
                active_root_mask=np.asarray([True, False], dtype=np.bool_),
            )
        )


def test_fixed_shape_batched_search_owner_reuses_preallocated_buffers() -> None:
    service = FixedShapeBatchedSearchOwnerV0(root_count=1, num_simulations=2)
    root_batch = _root_batch(
        legal_mask=np.asarray([[True, False, False]], dtype=np.bool_),
        active_root_mask=np.asarray([True], dtype=np.bool_),
    )

    first = service.run(root_batch)
    second = service.run(root_batch)

    assert first.metadata["profile_telemetry"][
        "fixed_shape_batched_search_owner_buffer_reused"
    ] is False
    assert second.metadata["profile_telemetry"][
        "fixed_shape_batched_search_owner_buffer_reused"
    ] is True
    assert service.last_profile_telemetry[
        "fixed_shape_batched_search_owner_buffer_reused"
    ] is True


def test_fixed_shape_batched_search_owner_two_phase_handle_lifecycle() -> None:
    root_batch = _root_batch(
        legal_mask=np.asarray(
            [
                [False, True, True],
                [True, True, True],
                [True, False, True],
                [False, False, False],
            ],
            dtype=np.bool_,
        ),
        active_root_mask=np.asarray([True, False, True, False], dtype=np.bool_),
    )
    service = FixedShapeBatchedSearchOwnerV0(root_count=4, num_simulations=7)

    action_step = service.run_action_step(root_batch)

    assert isinstance(service, CompactSearchTwoPhaseServiceV1)
    assert action_step.replay_payload_handle.startswith(FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL)
    np.testing.assert_array_equal(action_step.selected_action, np.asarray([1, 0]))
    payload = service.flush_replay_payload(action_step.replay_payload_handle)
    validate_compact_search_two_phase_payload_v1(action_step, payload)
    result = compact_search_result_v1_from_two_phase_payloads(
        root_batch,
        action_step,
        payload,
    )
    validate_compact_search_result_identity_v1(root_batch, result)
    np.testing.assert_array_equal(result.selected_action, action_step.selected_action)
    np.testing.assert_allclose(result.visit_policy, payload.visit_policy)
    np.testing.assert_allclose(result.root_value, payload.root_value)

    with pytest.raises(ReplayCompatibilityError, match="already-flushed|unknown"):
        service.flush_replay_payload(action_step.replay_payload_handle)


def test_fixed_shape_batched_search_owner_flushes_device_payload_without_host_fallback() -> None:
    root_batch = _root_batch(
        legal_mask=np.asarray(
            [
                [False, True, True],
                [True, True, True],
            ],
            dtype=np.bool_,
        ),
        active_root_mask=np.asarray([True, True], dtype=np.bool_),
    )
    service = FixedShapeBatchedSearchOwnerV0(
        root_count=2,
        num_simulations=7,
        device="cpu",
    )

    action_step = service.run_action_step(root_batch)
    payload = service.flush_device_replay_payload(action_step.replay_payload_handle)

    validate_compact_device_search_two_phase_payload_v1(action_step, payload)
    assert payload.metadata["device_replay_payload"] is True
    assert payload.metadata["host_search_payload_fallback_allowed"] is False
    assert str(payload.visit_policy.device) == "cpu"
    np.testing.assert_array_equal(action_step.selected_action, np.asarray([1, 0]))

    with pytest.raises(ReplayCompatibilityError, match="already-flushed|unknown"):
        service.flush_device_replay_payload(action_step.replay_payload_handle)


def _root_batch(
    *,
    legal_mask: np.ndarray,
    active_root_mask: np.ndarray,
) -> CompactRootBatchV1:
    root_count = int(np.asarray(legal_mask).shape[0])
    return CompactRootBatchV1(
        observation=np.zeros((root_count, 4, 8, 8), dtype=np.uint8),
        legal_mask=np.asarray(legal_mask, dtype=np.bool_),
        active_root_mask=np.asarray(active_root_mask, dtype=np.bool_),
        to_play=np.full((root_count,), -1, dtype=np.int64),
        env_row=np.arange(10, 10 + root_count, dtype=np.int32),
        player=np.arange(root_count, dtype=np.int16),
        policy_env_id=np.arange(100, 100 + root_count, dtype=np.int64),
        target_reward=np.zeros((root_count, 1), dtype=np.float32),
        done_root=~np.asarray(active_root_mask, dtype=np.bool_),
        final_observation=None,
        final_observation_row_mask=np.zeros((root_count,), dtype=np.bool_),
        terminal_row_mask=np.zeros((root_count,), dtype=np.bool_),
        autoreset_row_mask=np.zeros((root_count,), dtype=np.bool_),
        metadata={
            "contract_id": "curvyzero_compact_search_replay_service/v1",
            "schema_id": "curvyzero_compact_root_batch/v1",
        },
    )
