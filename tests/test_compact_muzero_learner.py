from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.training.compact_muzero_learner import (
    COMPACT_MUZERO_TARGET_MODE_REPEAT_CURRENT_PROFILE_ONLY,
    CompactMuZeroLearnerConfigV1,
    CompactMuZeroLearnerEdgeV1,
    build_compact_muzero_learner_batch_v1,
    _resolve_device,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


def _torch_module():
    return pytest.importorskip("torch")


def test_compact_muzero_learner_resolves_mps_explicitly_and_fail_closed() -> None:
    resolved = _resolve_device(_FakeTorchDevices(mps_available=True), "mps")
    assert str(resolved) == "mps"

    with pytest.raises(ReplayCompatibilityError, match="requested MPS"):
        _resolve_device(_FakeTorchDevices(mps_available=False), "mps")

    with pytest.raises(ReplayCompatibilityError, match="'mps'"):
        _resolve_device(_FakeTorchDevices(), "tpu")


class _FakeResolvedDevice:
    def __init__(self, value: str) -> None:
        self._value = str(value)
        self.type = self._value.split(":", 1)[0]
        self.index = None

    def __str__(self) -> str:
        return self._value


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self._available = bool(available)

    def is_available(self) -> bool:
        return self._available


class _FakeMpsBackend:
    def __init__(self, *, available: bool) -> None:
        self._available = bool(available)

    def is_available(self) -> bool:
        return self._available


class _FakeTorchDevices:
    def __init__(self, *, cuda_available: bool = False, mps_available: bool = False) -> None:
        self.cuda = _FakeCuda(available=cuda_available)
        self.backends = SimpleNamespace(
            mps=_FakeMpsBackend(available=mps_available),
        )

    def device(self, value: str) -> _FakeResolvedDevice:
        return _FakeResolvedDevice(str(value))


class _TinyMuZeroModel:
    def __new__(cls, *args, **kwargs):
        torch = _torch_module()

        class Tiny(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(4 * 8 * 8, 16),
                    torch.nn.Tanh(),
                )
                self.action_embedding = torch.nn.Embedding(ACTION_COUNT, 16)
                self.policy_head = torch.nn.Linear(16, ACTION_COUNT)
                self.value_head = torch.nn.Linear(16, 3)
                self.reward_head = torch.nn.Linear(16, 3)
                self.recurrent_actions = []

            def initial_inference(self, obs):
                from lzero.model.common import MZNetworkOutput

                latent = self.encoder(obs)
                policy_logits = self.policy_head(latent)
                value = self.value_head(latent)
                reward = torch.zeros_like(value)
                return MZNetworkOutput(value, reward, policy_logits, latent)

            def recurrent_inference(self, latent_state, action):
                from lzero.model.common import MZNetworkOutput

                self.recurrent_actions.append(action.detach().cpu().clone())
                next_latent = torch.tanh(
                    latent_state + self.action_embedding(action.reshape(-1).long())
                )
                policy_logits = self.policy_head(next_latent)
                value = self.value_head(next_latent)
                reward = self.reward_head(next_latent)
                return MZNetworkOutput(value, reward, policy_logits, next_latent)

        return Tiny()


def _one_hot(indices, *, torch_module=None):
    if torch_module is None:
        out = np.zeros((len(indices), ACTION_COUNT), dtype=np.float32)
        out[np.arange(len(indices)), indices] = 1.0
        return out
    out = torch_module.zeros((len(indices), ACTION_COUNT), dtype=torch_module.float32)
    out[torch_module.arange(len(indices)), torch_module.as_tensor(indices)] = 1.0
    return out


def _resident_sample(*, include_next: bool = True):
    torch = _torch_module()
    actions = torch.tensor([0, 1, 2], dtype=torch.int16)
    sample = SimpleNamespace(
        metadata={
            "resident_device_sample_batch": True,
            "device_replay_index_rows_sample": True,
        },
        observation=torch.arange(3 * 4 * 8 * 8, dtype=torch.uint8).reshape(3, 4, 8, 8),
        action=actions,
        action_mask=torch.ones((3, ACTION_COUNT), dtype=torch.bool),
        policy_target=_one_hot([0, 1, 2], torch_module=torch),
        root_value=torch.tensor([0.0, 0.5, -0.5], dtype=torch.float32),
        reward=torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32),
        weights=torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32),
    )
    if include_next:
        sample.next_action_mask = torch.ones((3, ACTION_COUNT), dtype=torch.bool)
        sample.next_policy_target = _one_hot([1, 2, 0], torch_module=torch)
        sample.next_root_value = torch.tensor([0.25, 0.0, -0.25], dtype=torch.float32)
    return sample


def _resident_unroll_sample(*, num_unroll_steps: int = 3):
    torch = _torch_module()
    sample = _resident_sample(include_next=False)
    row_count = int(sample.observation.shape[0])
    step_index = torch.arange(num_unroll_steps, dtype=torch.long)
    state_index = torch.arange(num_unroll_steps + 1, dtype=torch.long)
    row_index = torch.arange(row_count, dtype=torch.long).reshape(-1, 1)
    unroll_action = ((row_index + step_index.reshape(1, -1)) % ACTION_COUNT).to(
        torch.int16
    )
    policy_index = (row_index + state_index.reshape(1, -1)) % ACTION_COUNT
    unroll_policy_target = torch.zeros(
        (row_count, num_unroll_steps + 1, ACTION_COUNT),
        dtype=torch.float32,
    )
    unroll_policy_target.scatter_(2, policy_index.unsqueeze(-1), 1.0)
    row_value = torch.arange(row_count, dtype=torch.float32).reshape(-1, 1)
    unroll_reward = (row_value - step_index.to(torch.float32).reshape(1, -1)) / 4.0
    unroll_root_value = (
        row_value - state_index.to(torch.float32).reshape(1, -1)
    ) / 8.0

    sample.action = unroll_action[:, 0].contiguous()
    sample.action_mask = torch.ones((row_count, ACTION_COUNT), dtype=torch.bool)
    sample.policy_target = unroll_policy_target[:, 0, :].contiguous()
    sample.root_value = unroll_root_value[:, 0].contiguous()
    sample.reward = unroll_reward[:, 0].contiguous()
    sample.unroll_action = unroll_action.contiguous()
    sample.unroll_reward = unroll_reward.contiguous()
    sample.unroll_policy_target = unroll_policy_target.contiguous()
    sample.unroll_root_value = unroll_root_value.contiguous()
    sample.unroll_action_mask = torch.ones(
        (row_count, num_unroll_steps + 1, ACTION_COUNT),
        dtype=torch.bool,
    )
    return sample


def _resident_terminal_unroll_sample(*, num_unroll_steps: int = 3, include_masks: bool = True):
    torch = _torch_module()
    sample = _resident_unroll_sample(num_unroll_steps=num_unroll_steps)
    row_count = int(sample.observation.shape[0])
    done = torch.zeros((row_count, num_unroll_steps), dtype=torch.bool)
    done[0, 1] = True
    sample.unroll_done = done
    sample.unroll_terminated = done.clone()
    sample.unroll_truncated = torch.zeros_like(done)
    if not include_masks:
        return sample

    action_valid = torch.ones((row_count, num_unroll_steps), dtype=torch.bool)
    reward_valid = torch.ones_like(action_valid)
    policy_valid = torch.ones((row_count, num_unroll_steps + 1), dtype=torch.bool)
    value_valid = torch.ones_like(policy_valid)
    action_valid[0, 2:] = False
    reward_valid[0, 2:] = False
    policy_valid[0, 2:] = False
    value_valid[0, 2:] = False

    sample.unroll_action_valid_mask = action_valid
    sample.unroll_reward_valid_mask = reward_valid
    sample.unroll_policy_valid_mask = policy_valid
    sample.unroll_value_valid_mask = value_valid
    sample.unroll_reward = sample.unroll_reward.masked_fill(~reward_valid, 0.0)
    sample.unroll_root_value = sample.unroll_root_value.masked_fill(~value_valid, 0.0)
    sample.unroll_root_value[0, 1] = sample.unroll_reward[0, 1]
    sample.unroll_root_value[0, 0] = (
        sample.unroll_reward[0, 0] + sample.unroll_reward[0, 1]
    )
    sample.unroll_policy_target = sample.unroll_policy_target.masked_fill(
        ~policy_valid.unsqueeze(-1),
        0.0,
    )
    sample.unroll_action_mask[0, 2:, :] = False
    return sample


def _resident_one_step_terminal_sample(*, include_masks: bool = True):
    torch = _torch_module()
    sample = _resident_sample(include_next=True)
    sample.done = torch.tensor([True, False, False], dtype=torch.bool)
    sample.terminated = sample.done.clone()
    sample.truncated = torch.zeros_like(sample.done)
    if not include_masks:
        return sample

    sample.unroll_action_valid_mask = torch.ones((3, 1), dtype=torch.bool)
    sample.unroll_reward_valid_mask = torch.ones((3, 1), dtype=torch.bool)
    sample.unroll_policy_valid_mask = torch.ones((3, 2), dtype=torch.bool)
    sample.unroll_value_valid_mask = torch.ones((3, 2), dtype=torch.bool)
    sample.unroll_policy_valid_mask[0, 1] = False
    sample.unroll_value_valid_mask[0, 1] = False
    sample.next_policy_target[0, :] = 0.0
    sample.next_root_value[0] = 0.0
    sample.next_action_mask[0, :] = False
    return sample


def test_compact_muzero_learner_batch_requires_explicit_next_targets() -> None:
    sample = _resident_sample(include_next=False)

    with pytest.raises(ReplayCompatibilityError, match="next_policy_target"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu"),
        )


def test_compact_muzero_learner_batch_requires_next_action_mask() -> None:
    sample = _resident_sample(include_next=True)
    delattr(sample, "next_action_mask")

    with pytest.raises(ReplayCompatibilityError, match="next_action_mask"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu"),
        )


def test_compact_muzero_learner_batch_can_use_named_profile_only_repeat_mode() -> None:
    sample = _resident_sample(include_next=False)

    batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=CompactMuZeroLearnerConfigV1(
            device="cpu",
            target_mode=COMPACT_MUZERO_TARGET_MODE_REPEAT_CURRENT_PROFILE_ONLY,
        ),
    )

    assert batch.metadata["compact_muzero_learner_target_mode"] == (
        COMPACT_MUZERO_TARGET_MODE_REPEAT_CURRENT_PROFILE_ONLY
    )
    assert batch.metadata["compact_muzero_learner_observation_h2d_bytes"] == 0
    assert batch.metadata["compact_muzero_learner_input_h2d_bytes"] == 0
    assert list(batch.target_reward.shape) == [3, 1]
    assert list(batch.target_value.shape) == [3, 2]
    assert list(batch.target_policy.shape) == [3, 2, ACTION_COUNT]


def test_compact_muzero_learner_batch_rejects_one_step_terminal_without_masks() -> None:
    sample = _resident_one_step_terminal_sample(include_masks=False)

    with pytest.raises(ReplayCompatibilityError, match="complete validity masks"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu"),
        )


def test_compact_muzero_learner_batch_accepts_one_step_terminal_masks() -> None:
    sample = _resident_one_step_terminal_sample()

    batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=CompactMuZeroLearnerConfigV1(device="cpu"),
    )

    assert batch.metadata["compact_muzero_learner_done_count"] == 1
    assert batch.metadata["compact_muzero_learner_policy_valid_count"] == 5
    assert batch.metadata["compact_muzero_learner_value_valid_count"] == 5
    assert not bool(batch.target_policy_mask[0, 1])
    assert not bool(batch.target_value_mask[0, 1])
    assert float(batch.target_policy[0, 1].sum().item()) == 0.0


def test_compact_muzero_learner_batch_accepts_explicit_multi_step_targets() -> None:
    sample = _resident_unroll_sample(num_unroll_steps=3)

    batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
    )

    assert batch.metadata["compact_muzero_learner_num_unroll_steps"] == 3
    assert batch.metadata["compact_muzero_learner_next_action_mask_present"] is False
    assert batch.metadata["compact_muzero_learner_observation_h2d_bytes"] == 0
    assert batch.metadata["compact_muzero_learner_input_h2d_bytes"] == 0
    assert list(batch.action.shape) == [3, 3]
    assert list(batch.action_mask.shape) == [3, 4, ACTION_COUNT]
    assert list(batch.target_reward.shape) == [3, 3]
    assert list(batch.target_value.shape) == [3, 4]
    assert list(batch.target_policy.shape) == [3, 4, ACTION_COUNT]
    assert batch.next_action_mask is None


def test_compact_muzero_learner_batch_rejects_terminal_unroll_without_validity_masks() -> None:
    sample = _resident_terminal_unroll_sample(num_unroll_steps=3, include_masks=False)

    with pytest.raises(ReplayCompatibilityError, match="complete validity masks"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
        )


def test_compact_muzero_learner_batch_accepts_terminal_unroll_masks() -> None:
    sample = _resident_terminal_unroll_sample(num_unroll_steps=3)

    batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
    )

    assert batch.metadata["compact_muzero_learner_done_count"] == 1
    assert batch.metadata["compact_muzero_learner_action_valid_count"] == 8
    assert batch.metadata["compact_muzero_learner_reward_valid_count"] == 8
    assert batch.metadata["compact_muzero_learner_policy_valid_count"] == 10
    assert batch.metadata["compact_muzero_learner_value_valid_count"] == 10
    assert not bool(batch.action_valid_mask[0, 2])
    assert not bool(batch.target_reward_mask[0, 2])
    assert not bool(batch.target_policy_mask[0, 2:].any())
    assert not bool(batch.target_value_mask[0, 2:].any())


def test_compact_muzero_learner_batch_rejects_post_terminal_policy_target() -> None:
    sample = _resident_terminal_unroll_sample(num_unroll_steps=3)
    sample.unroll_policy_target[0, 3, 0] = 1.0

    with pytest.raises(ReplayCompatibilityError, match="validity mask"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
        )


def test_compact_muzero_learner_batch_rejects_post_terminal_reward_or_value() -> None:
    sample = _resident_terminal_unroll_sample(num_unroll_steps=3)
    sample.unroll_reward[0, 2] = 0.25

    with pytest.raises(ReplayCompatibilityError, match="invalid unroll_reward"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
        )

    sample = _resident_terminal_unroll_sample(num_unroll_steps=3)
    sample.unroll_root_value[0, 3] = 0.25

    with pytest.raises(ReplayCompatibilityError, match="invalid unroll_root_value"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
        )


def test_compact_muzero_learner_batch_rejects_truncated_unrolls() -> None:
    sample = _resident_terminal_unroll_sample(num_unroll_steps=3)
    sample.unroll_done[0, 1] = True
    sample.unroll_terminated[0, 1] = False
    sample.unroll_truncated[0, 1] = True

    with pytest.raises(ReplayCompatibilityError, match="truncated unroll"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
        )


def test_compact_muzero_learner_batch_rejects_multi_step_legacy_next_targets() -> None:
    sample = _resident_sample(include_next=True)

    with pytest.raises(ReplayCompatibilityError, match="unroll_action"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=2),
        )


def test_compact_muzero_learner_batch_rejects_repeat_current_multi_step_mode() -> None:
    sample = _resident_unroll_sample(num_unroll_steps=2)

    with pytest.raises(ReplayCompatibilityError, match="only supports num_unroll_steps=1"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(
                device="cpu",
                num_unroll_steps=2,
                target_mode=COMPACT_MUZERO_TARGET_MODE_REPEAT_CURRENT_PROFILE_ONLY,
            ),
        )


def test_compact_muzero_learner_batch_checks_multi_step_action_legality() -> None:
    sample = _resident_unroll_sample(num_unroll_steps=3)
    illegal_action = int(sample.unroll_action[0, 2].item())
    sample.unroll_action_mask[0, 2, illegal_action] = False

    with pytest.raises(ReplayCompatibilityError, match="legal under unroll_action_mask"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
        )


def test_compact_muzero_learner_batch_checks_multi_step_policy_legality() -> None:
    sample = _resident_unroll_sample(num_unroll_steps=3)
    illegal_policy_action = int(sample.unroll_policy_target[0, 3].argmax().item())
    sample.unroll_action_mask[0, 3, illegal_policy_action] = False

    with pytest.raises(ReplayCompatibilityError, match="assigns mass to illegal actions"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
        )


def test_compact_muzero_learner_edge_runs_each_unroll_step() -> None:
    torch = _torch_module()
    torch.manual_seed(20260528)
    sample = _resident_unroll_sample(num_unroll_steps=3)
    edge = CompactMuZeroLearnerEdgeV1(
        model=_TinyMuZeroModel(),
        config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
    )

    result = edge.train_on_sample_batch(sample, train_steps=1)

    assert result.telemetry["compact_muzero_learner_num_unroll_steps"] == 3
    assert [action.tolist() for action in edge.model.recurrent_actions] == [
        sample.unroll_action[:, step].to(torch.long).tolist() for step in range(3)
    ]


def test_compact_muzero_learner_edge_trains_on_prebuilt_batch_equivalently() -> None:
    torch = _torch_module()
    torch.manual_seed(20260531)
    sample = _resident_unroll_sample(num_unroll_steps=2)
    config = CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=2)
    model_a = _TinyMuZeroModel()
    model_b = _TinyMuZeroModel()
    model_b.load_state_dict(model_a.state_dict())
    edge_a = CompactMuZeroLearnerEdgeV1(model=model_a, config=config)
    edge_b = CompactMuZeroLearnerEdgeV1(model=model_b, config=config)
    learner_batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=config,
    )

    result_a = edge_a.train_on_sample_batch(sample, train_steps=1)
    result_b = edge_b.train_on_learner_batch(learner_batch, train_steps=1)

    assert result_a.telemetry["compact_muzero_learner_prebuilt_batch_used"] is False
    assert result_b.telemetry["compact_muzero_learner_prebuilt_batch_used"] is True
    for telemetry in (result_a.telemetry, result_b.telemetry):
        for key in (
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
            "compact_muzero_learner_cuda_sync_timing_diagnostics",
            "compact_muzero_learner_cuda_sync_timing_enabled",
            "compact_muzero_learner_cuda_sync_count",
            "compact_muzero_learner_cuda_sync_sec",
            "compact_muzero_learner_accounted_sec",
            "compact_muzero_learner_residual_sec",
        ):
            assert key in telemetry
        assert telemetry["compact_muzero_learner_accounted_sec"] > 0.0
        assert telemetry["compact_muzero_learner_cuda_sync_timing_diagnostics"] is False
        assert telemetry["compact_muzero_learner_cuda_sync_timing_enabled"] is False
        assert telemetry["compact_muzero_learner_cuda_sync_count"] == 0
        assert telemetry["compact_muzero_learner_cuda_sync_sec"] == 0.0
    assert result_a.telemetry["compact_muzero_learner_sample_rows"] == (
        result_b.telemetry["compact_muzero_learner_sample_rows"]
    )
    assert result_a.telemetry["compact_muzero_learner_input_h2d_bytes"] == (
        result_b.telemetry["compact_muzero_learner_input_h2d_bytes"]
    )
    assert result_a.telemetry["compact_muzero_learner_loss"] == pytest.approx(
        result_b.telemetry["compact_muzero_learner_loss"],
        rel=1e-6,
        abs=1e-6,
    )
    for left, right in zip(edge_a.model.parameters(), edge_b.model.parameters(), strict=True):
        assert torch.allclose(left.detach(), right.detach(), atol=1e-6, rtol=1e-6)


def test_compact_muzero_learner_edge_fast_validates_prevalidated_prebuilt_batches() -> None:
    torch = _torch_module()
    torch.manual_seed(20260531)
    sample = _resident_unroll_sample(num_unroll_steps=2)
    config = CompactMuZeroLearnerConfigV1(
        device="cpu",
        num_unroll_steps=2,
        collect_cuda_memory_telemetry=False,
        fast_prebuilt_batch_validation_after_first_full=True,
    )
    learner_batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=config,
    )
    edge = CompactMuZeroLearnerEdgeV1(model=_TinyMuZeroModel(), config=config)

    first = edge.train_on_learner_batch(learner_batch, train_steps=1)
    second = edge.train_on_learner_batch(learner_batch, train_steps=1)

    assert learner_batch.metadata["compact_muzero_learner_batch_prevalidated"] is True
    assert (
        learner_batch.metadata["compact_muzero_learner_batch_prevalidation_source"]
        == "build_compact_muzero_learner_batch_v1"
    )
    assert first.telemetry["compact_muzero_learner_prebuilt_batch_validation_deep"] is True
    assert first.telemetry["compact_muzero_learner_prebuilt_batch_fast_validation_candidate"] is True
    assert first.telemetry["compact_muzero_learner_prebuilt_batch_fast_validation_used"] is False
    assert first.telemetry["compact_muzero_learner_prebuilt_batch_deep_validation_count"] == 1
    assert first.telemetry["compact_muzero_learner_prebuilt_batch_fast_validation_count"] == 0
    assert second.telemetry["compact_muzero_learner_prebuilt_batch_validation_deep"] is False
    assert second.telemetry["compact_muzero_learner_prebuilt_batch_fast_validation_used"] is True
    assert second.telemetry["compact_muzero_learner_prebuilt_batch_deep_validation_count"] == 1
    assert second.telemetry["compact_muzero_learner_prebuilt_batch_fast_validation_count"] == 1
    assert second.telemetry["compact_muzero_learner_cuda_memory_telemetry_enabled"] is False
    assert (
        second.telemetry[
            "compact_muzero_learner_cuda_before_batch_telemetry_enabled"
        ]
        is False
    )


def test_compact_muzero_learner_edge_keeps_deep_validation_for_unproven_batches() -> None:
    torch = _torch_module()
    torch.manual_seed(20260531)
    sample = _resident_unroll_sample(num_unroll_steps=2)
    config = CompactMuZeroLearnerConfigV1(
        device="cpu",
        num_unroll_steps=2,
        fast_prebuilt_batch_validation_after_first_full=True,
    )
    learner_batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=config,
    )
    unproven_batch = replace(
        learner_batch,
        metadata={
            **learner_batch.metadata,
            "compact_muzero_learner_batch_prevalidated": False,
        },
    )
    edge = CompactMuZeroLearnerEdgeV1(model=_TinyMuZeroModel(), config=config)

    first = edge.train_on_learner_batch(unproven_batch, train_steps=1)
    second = edge.train_on_learner_batch(unproven_batch, train_steps=1)

    assert first.telemetry["compact_muzero_learner_prebuilt_batch_validation_deep"] is True
    assert first.telemetry["compact_muzero_learner_prebuilt_batch_fast_validation_candidate"] is False
    assert second.telemetry["compact_muzero_learner_prebuilt_batch_validation_deep"] is True
    assert second.telemetry["compact_muzero_learner_prebuilt_batch_fast_validation_used"] is False
    assert second.telemetry["compact_muzero_learner_prebuilt_batch_deep_validation_count"] == 2
    assert second.telemetry["compact_muzero_learner_prebuilt_batch_fast_validation_count"] == 0


def test_compact_muzero_learner_edge_rejects_bad_prebuilt_batch_schema() -> None:
    sample = _resident_unroll_sample(num_unroll_steps=2)
    config = CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=2)
    learner_batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=config,
    )
    bad_batch = replace(
        learner_batch,
        metadata={
            **learner_batch.metadata,
            "compact_muzero_learner_batch_schema_id": "bad",
        },
    )
    edge = CompactMuZeroLearnerEdgeV1(model=_TinyMuZeroModel(), config=config)

    with pytest.raises(ReplayCompatibilityError, match="schema mismatch"):
        edge.train_on_learner_batch(bad_batch, train_steps=1)


def test_compact_muzero_learner_edge_rejects_prebuilt_batch_config_mismatch() -> None:
    sample = _resident_unroll_sample(num_unroll_steps=2)
    config = CompactMuZeroLearnerConfigV1(
        device="cpu",
        num_unroll_steps=2,
        require_device_replay_rows=True,
    )
    learner_batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=config,
    )
    edge = CompactMuZeroLearnerEdgeV1(model=_TinyMuZeroModel(), config=config)

    bad_unroll = replace(
        learner_batch,
        metadata={
            **learner_batch.metadata,
            "compact_muzero_learner_num_unroll_steps": 3,
        },
    )
    with pytest.raises(ReplayCompatibilityError, match="unroll steps"):
        edge.train_on_learner_batch(bad_unroll, train_steps=1)

    bad_target_mode = replace(
        learner_batch,
        metadata={
            **learner_batch.metadata,
            "compact_muzero_learner_target_mode": (
                COMPACT_MUZERO_TARGET_MODE_REPEAT_CURRENT_PROFILE_ONLY
            ),
        },
    )
    with pytest.raises(ReplayCompatibilityError, match="target mode"):
        edge.train_on_learner_batch(bad_target_mode, train_steps=1)

    bad_resident = replace(
        learner_batch,
        metadata={
            **learner_batch.metadata,
            "compact_muzero_learner_resident_sample_used": False,
            "resident_device_sample_batch": False,
        },
    )
    with pytest.raises(ReplayCompatibilityError, match="resident sample"):
        edge.train_on_learner_batch(bad_resident, train_steps=1)

    bad_device_rows = replace(
        learner_batch,
        metadata={
            **learner_batch.metadata,
            "compact_muzero_learner_device_replay_index_rows_sample": False,
            "device_replay_index_rows_sample": False,
        },
    )
    with pytest.raises(ReplayCompatibilityError, match="device replay rows"):
        edge.train_on_learner_batch(bad_device_rows, train_steps=1)


def test_compact_muzero_learner_edge_prebuilt_metadata_cannot_override_telemetry() -> None:
    sample = _resident_unroll_sample(num_unroll_steps=2)
    config = CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=2)
    learner_batch = build_compact_muzero_learner_batch_v1(
        sample,
        config=config,
    )
    polluted_batch = replace(
        learner_batch,
        metadata={
            **learner_batch.metadata,
            "compact_muzero_learner_prebuilt_batch_used": False,
            "compact_muzero_learner_train_steps": 999,
        },
    )
    edge = CompactMuZeroLearnerEdgeV1(model=_TinyMuZeroModel(), config=config)

    result = edge.train_on_learner_batch(polluted_batch, train_steps=1)

    assert result.telemetry["compact_muzero_learner_prebuilt_batch_used"] is True
    assert result.telemetry["compact_muzero_learner_train_steps"] == 1


def test_compact_muzero_learner_edge_masks_terminal_unroll_losses() -> None:
    torch = _torch_module()
    torch.manual_seed(20260528)
    sample = _resident_terminal_unroll_sample(num_unroll_steps=3)
    edge = CompactMuZeroLearnerEdgeV1(
        model=_TinyMuZeroModel(),
        config=CompactMuZeroLearnerConfigV1(device="cpu", num_unroll_steps=3),
    )

    result = edge.train_on_sample_batch(sample, train_steps=1)

    assert result.telemetry["compact_muzero_learner_done_count"] == 1
    assert result.telemetry["compact_muzero_learner_policy_valid_count"] == 10
    assert result.telemetry["compact_muzero_learner_reward_valid_count"] == 8
    assert result.telemetry["compact_muzero_learner_loss"] > 0.0


def test_compact_muzero_learner_edge_updates_model_without_stock_forward_learn() -> None:
    torch = _torch_module()
    torch.manual_seed(20260526)
    sample = _resident_sample(include_next=True)
    model = _TinyMuZeroModel()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    edge = CompactMuZeroLearnerEdgeV1(
        model=model,
        config=CompactMuZeroLearnerConfigV1(device="cpu"),
    )

    result = edge.train_on_sample_batch(sample, train_steps=2)

    after = list(edge.model.parameters())
    assert any(
        not torch.allclose(left, right.detach()) for left, right in zip(before, after, strict=True)
    )
    telemetry = result.telemetry
    assert telemetry["compact_muzero_learner_calls_train_muzero"] is False
    assert telemetry["compact_muzero_learner_update_claim"] is True
    assert telemetry["compact_muzero_learner_stock_lightzero_parity_claim"] is False
    assert telemetry["compact_muzero_learner_sample_rows"] == 3
    assert telemetry["compact_muzero_learner_train_steps"] == 2
    assert telemetry["compact_muzero_learner_loss"] > 0.0
    assert telemetry["compact_muzero_learner_resident_sample_used"] is True
    assert telemetry["compact_muzero_learner_device_replay_index_rows_sample"] is True
    assert telemetry["compact_muzero_learner_input_h2d_bytes"] == 0
    assert telemetry["compact_muzero_learner_cuda_before_batch_memory_allocated_bytes"] is None
    assert telemetry["compact_muzero_learner_cuda_before_backward_mem_get_info_free_bytes"] is None
    assert telemetry["compact_muzero_learner_cuda_after_backward_memory_peak_allocated_bytes"] is None
    assert telemetry["compact_muzero_learner_cuda_after_train_memory_reserved_bytes"] is None


def test_compact_muzero_learner_rejects_support_scale_head_mismatch() -> None:
    sample = _resident_sample(include_next=True)
    edge = CompactMuZeroLearnerEdgeV1(
        model=_TinyMuZeroModel(),
        config=CompactMuZeroLearnerConfigV1(device="cpu", support_scale=2),
    )

    with pytest.raises(ReplayCompatibilityError, match="support_scale"):
        edge.train_on_sample_batch(sample, train_steps=1)


def test_compact_muzero_learner_rejects_hidden_host_sample_fallback() -> None:
    sample = _resident_sample(include_next=True)
    sample.metadata = {"resident_device_sample_batch": False}

    with pytest.raises(ReplayCompatibilityError, match="resident sample"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu"),
        )


def test_compact_muzero_learner_rejects_fake_resident_host_arrays() -> None:
    sample = SimpleNamespace(
        metadata={
            "resident_device_sample_batch": True,
            "device_replay_index_rows_sample": True,
        },
        observation=np.zeros((2, 4, 8, 8), dtype=np.uint8),
        action=np.zeros((2,), dtype=np.int16),
        action_mask=np.ones((2, ACTION_COUNT), dtype=np.bool_),
        policy_target=_one_hot([0, 0]),
        root_value=np.zeros((2,), dtype=np.float32),
        reward=np.zeros((2,), dtype=np.float32),
        next_action_mask=np.ones((2, ACTION_COUNT), dtype=np.bool_),
        next_policy_target=_one_hot([0, 0]),
        next_root_value=np.zeros((2,), dtype=np.float32),
    )

    with pytest.raises(ReplayCompatibilityError, match="must be a torch tensor"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu"),
        )


def test_compact_muzero_learner_uses_next_action_mask_for_next_policy_legality() -> None:
    torch = _torch_module()
    sample = _resident_sample(include_next=True)
    sample.action_mask = torch.ones((3, ACTION_COUNT), dtype=torch.bool)
    sample.next_action_mask = torch.ones((3, ACTION_COUNT), dtype=torch.bool)
    sample.next_action_mask[0, 1] = False

    with pytest.raises(ReplayCompatibilityError, match="illegal next actions"):
        build_compact_muzero_learner_batch_v1(
            sample,
            config=CompactMuZeroLearnerConfigV1(device="cpu"),
        )
