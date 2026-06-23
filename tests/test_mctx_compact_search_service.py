from __future__ import annotations

import numpy as np
import pytest

from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.mctx_compact_search_service import MCTX_COMPACT_SEARCH_SERVICE_IMPL
from curvyzero.training.mctx_compact_search_service import MCTX_COMPACT_SEARCH_SERVICE_SEMANTICS
from curvyzero.training.mctx_compact_search_service import MctxCompactSearchConfig
from curvyzero.training.mctx_compact_search_service import MctxCompactSearchServiceV1
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


def test_mctx_compact_search_service_zero_active_roots_is_profile_labeled() -> None:
    root_batch = _root_batch(
        legal_mask=np.zeros((2, 3), dtype=np.bool_),
        active_root_mask=np.zeros((2,), dtype=np.bool_),
        done_root=np.ones((2,), dtype=np.bool_),
    )

    result = MctxCompactSearchServiceV1(num_simulations=4).run(root_batch)

    assert result.metadata["search_impl"] == MCTX_COMPACT_SEARCH_SERVICE_IMPL
    assert result.metadata["profile_backend"] == MCTX_COMPACT_SEARCH_SERVICE_IMPL
    assert result.metadata["profile_only"] is True
    assert result.metadata["profile_semantics"] == MCTX_COMPACT_SEARCH_SERVICE_SEMANTICS
    assert result.metadata["compact_search_service_adapter"] is True
    assert result.metadata["not_lightzero_ctree"] is True
    assert result.metadata["not_train_muzero"] is True
    assert result.metadata["profile_telemetry"]["mctx_compact_search_service_zero"] is True
    assert result.selected_action.shape == (0,)
    assert result.visit_policy.shape == (0, 3)


def test_mctx_compact_search_service_rejects_active_root_with_no_legal_action_before_jax() -> None:
    root_batch = _root_batch(
        legal_mask=np.zeros((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
        done_root=np.zeros((2,), dtype=np.bool_),
    )

    with pytest.raises(ReplayCompatibilityError, match="no legal action"):
        MctxCompactSearchServiceV1(num_simulations=4).run(root_batch)


def test_mctx_compact_search_service_rejects_inactive_roots_in_fixed_shape_mode_before_jax() -> None:
    root_batch = _root_batch(
        legal_mask=np.asarray([[False, True, False], [False, True, False]], dtype=np.bool_),
        active_root_mask=np.asarray([True, False], dtype=np.bool_),
        done_root=np.asarray([False, True], dtype=np.bool_),
    )

    with pytest.raises(ReplayCompatibilityError, match="requires all roots active"):
        MctxCompactSearchServiceV1(num_simulations=4).run(root_batch)


def test_mctx_compact_search_service_can_use_shadow_model_backend() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mctx")
    root_batch = _root_batch(
        legal_mask=np.asarray([[True, True, True], [True, True, True]], dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
        done_root=np.zeros((2,), dtype=np.bool_),
    )

    result = MctxCompactSearchServiceV1(
        num_simulations=2,
        config=MctxCompactSearchConfig(require_gpu_backend=False),
        shadow_model=_FakeShadowModel(),
        model_metadata={"checkpoint_ref": "unit-test/iteration_1.pth.tar"},
    ).run(root_batch)

    telemetry = result.metadata["profile_telemetry"]
    assert telemetry["mctx_compact_search_service_model_backend"] == (
        "lightzero_jax_shadow_model"
    )
    assert telemetry["mctx_compact_search_service_model_metadata"][
        "checkpoint_ref"
    ] == "unit-test/iteration_1.pth.tar"
    assert result.selected_action.shape == (2,)
    assert result.visit_policy.shape == (2, 3)
    assert result.root_value.shape == (2,)
    assert result.predicted_value is not None
    assert result.predicted_value.shape == (2,)
    assert result.predicted_policy_logits is not None
    assert result.predicted_policy_logits.shape == (2, 3)
    assert telemetry["mctx_compact_search_service_predicted_value_shape"] == [2]
    assert telemetry["mctx_compact_search_service_predicted_policy_logits_shape"] == [2, 3]
    np.testing.assert_allclose(result.visit_policy.sum(axis=1), 1.0, atol=1e-6)


class _FakeShadowModel:
    action_space_size = 3
    reward_support_size = 3
    value_support_size = 3

    def representation(self, obs):
        import jax.numpy as jnp

        mean = jnp.mean(obs, axis=(1, 2, 3))
        return jnp.stack([mean, mean * 0.0 + 0.25], axis=1)

    def prediction(self, latent_state):
        import jax.numpy as jnp

        policy_logits = jnp.stack(
            [latent_state[:, 0], latent_state[:, 1], -latent_state[:, 0]],
            axis=1,
        )
        value_logits = jnp.stack(
            [-latent_state[:, 0], latent_state[:, 0] * 0.0, latent_state[:, 0]],
            axis=1,
        )
        return policy_logits, value_logits

    def recurrent_inference(self, latent_state, action):
        import jax
        import jax.numpy as jnp

        one_hot = jax.nn.one_hot(action.astype(jnp.int32), 3, dtype=jnp.float32)
        next_latent = latent_state + jnp.stack(
            [one_hot[:, 1] * 0.05, one_hot[:, 2] * 0.03],
            axis=1,
        )
        reward_logits = jnp.stack(
            [-next_latent[:, 1], next_latent[:, 1] * 0.0, next_latent[:, 1]],
            axis=1,
        )
        policy_logits, value_logits = self.prediction(next_latent)
        return {
            "value": value_logits,
            "reward": reward_logits,
            "policy_logits": policy_logits,
            "latent_state": next_latent,
        }


def _root_batch(
    *,
    legal_mask: np.ndarray,
    active_root_mask: np.ndarray,
    done_root: np.ndarray,
) -> CompactRootBatchV1:
    root_count = int(np.asarray(legal_mask).shape[0])
    return CompactRootBatchV1(
        observation=np.zeros((root_count, 4, 64, 64), dtype=np.uint8),
        legal_mask=np.asarray(legal_mask, dtype=np.bool_),
        active_root_mask=np.asarray(active_root_mask, dtype=np.bool_),
        to_play=np.full((root_count,), -1, dtype=np.int64),
        env_row=np.zeros((root_count,), dtype=np.int32),
        player=np.arange(root_count, dtype=np.int16),
        policy_env_id=np.arange(10, 10 + root_count, dtype=np.int64),
        target_reward=np.zeros((root_count, 1), dtype=np.float32),
        done_root=np.asarray(done_root, dtype=np.bool_),
        final_observation=None,
        final_observation_row_mask=np.zeros((1,), dtype=np.bool_),
        terminal_row_mask=np.zeros((1,), dtype=np.bool_),
        autoreset_row_mask=np.zeros((1,), dtype=np.bool_),
        metadata={
            "contract_id": "curvyzero_compact_search_replay_service/v1",
            "schema_id": "curvyzero_compact_root_batch/v1",
        },
    )
