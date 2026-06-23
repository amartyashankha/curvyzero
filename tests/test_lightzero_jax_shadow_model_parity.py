from __future__ import annotations

import numpy as np
import pytest

from curvyzero.training.lightzero_jax_shadow_model_parity import (
    PROFILE_ONLY_LABELS,
    compare_arrays,
    deterministic_observation_batch,
    inverse_scalar_transform_logits,
    profile_only_report_base,
    require_immutable_checkpoint_ref,
    summarize_state_dict_coverage,
)


def test_profile_only_report_labels_do_not_claim_training() -> None:
    labels = profile_only_report_base()

    assert labels == PROFILE_ONLY_LABELS
    assert labels["profile_only"] is True
    assert labels["not_train_muzero"] is True
    assert labels["not_mctx"] is True
    assert labels["touches_live_runs"] is False
    assert labels["trainer_defaults_changed"] is False


def test_mutable_checkpoint_refs_are_rejected() -> None:
    with pytest.raises(ValueError, match="mutable"):
        require_immutable_checkpoint_ref("training/run/checkpoints/latest.pth.tar")

    with pytest.raises(ValueError, match="mutable"):
        require_immutable_checkpoint_ref("training/run/checkpoints/ckpt_best.pth.tar")

    assert (
        require_immutable_checkpoint_ref("training/run/checkpoints/iteration_42.pth.tar")
        == "training/run/checkpoints/iteration_42.pth.tar"
    )


def test_compare_arrays_reports_shape_and_numeric_mismatch() -> None:
    shape_mismatch = compare_arrays(
        "shape", np.zeros((2, 3)), np.zeros((3, 2)), atol=0.0, rtol=0.0
    )
    assert shape_mismatch.allclose is False
    assert shape_mismatch.max_abs is None

    numeric_mismatch = compare_arrays(
        "numeric",
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 2.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert numeric_mismatch.allclose is False
    assert numeric_mismatch.max_abs == 1.0
    assert numeric_mismatch.max_rel == 0.5


def test_state_dict_coverage_ignores_non_inference_ssl_and_bn_counters() -> None:
    state_dict = {
        "representation_network.downsample_net.conv1.weight": np.zeros((1,)),
        "representation_network.downsample_net.norm1.weight": np.zeros((1,)),
        "representation_network.downsample_net.norm1.num_batches_tracked": np.zeros(()),
        "projection.0.weight": np.zeros((1,)),
        "prediction_head.0.weight": np.zeros((1,)),
    }

    summary = summarize_state_dict_coverage(
        state_dict,
        {"representation_network.downsample_net.conv1.weight"},
    )

    assert summary["ok"] is False
    assert summary["missing_required_keys"] == [
        "representation_network.downsample_net.norm1.weight"
    ]
    assert summary["ignored_key_count"] == 3


def test_deterministic_observation_batch_shapes_and_reproducibility() -> None:
    first = deterministic_observation_batch(batch_size=2, kind="random", seed=7)
    second = deterministic_observation_batch(batch_size=2, kind="random", seed=7)

    assert first.shape == (2, 4, 64, 64)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_inverse_scalar_transform_logits_preserves_zero_support_mean() -> None:
    logits = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    value = inverse_scalar_transform_logits(logits, support_scale=1)

    assert value.shape == (1, 1)
    assert value[0, 0] == pytest.approx(0.0, abs=1e-12)
