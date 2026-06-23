"""Fail-closed tensor-native MuZero learner edge for compact profile work.

This module is not a stock LightZero trainer and is not wired into Coach runs.
It exists to price and validate the learner boundary once compact replay samples
are already device-owned.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


COMPACT_MUZERO_LEARNER_BATCH_SCHEMA_ID = "curvyzero_compact_muzero_learner_batch/v1"
COMPACT_MUZERO_LEARNER_EDGE_IMPL_ID = "curvyzero_compact_muzero_learner_edge/v1"
COMPACT_MUZERO_TARGET_MODE_EXPLICIT_NEXT = "explicit_next_targets"
COMPACT_MUZERO_TARGET_MODE_REPEAT_CURRENT_PROFILE_ONLY = (
    "repeat_current_targets_profile_only"
)
COMPACT_MUZERO_TARGET_MODES = (
    COMPACT_MUZERO_TARGET_MODE_EXPLICIT_NEXT,
    COMPACT_MUZERO_TARGET_MODE_REPEAT_CURRENT_PROFILE_ONLY,
)


@dataclass(frozen=True, slots=True)
class CompactMuZeroLearnerConfigV1:
    """Strict config for a compact MuZero learner update."""

    device: str = "auto"
    support_scale: int = 1
    num_unroll_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_value: float = 10.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    normalize_uint8_observation: bool = True
    require_resident_sample: bool = True
    require_device_replay_rows: bool = False
    target_mode: str = COMPACT_MUZERO_TARGET_MODE_EXPLICIT_NEXT
    collect_cuda_memory_telemetry: bool = True
    fast_prebuilt_batch_validation_after_first_full: bool = False
    cuda_sync_timing_diagnostics: bool = False


@dataclass(frozen=True, slots=True)
class CompactMuZeroLearnerBatchV1:
    """Torch tensors in the exact small shape consumed by the compact edge."""

    metadata: dict[str, Any]
    observation: Any
    action: Any
    action_mask: Any
    target_reward: Any
    target_value: Any
    target_policy: Any
    target_reward_mask: Any
    target_value_mask: Any
    target_policy_mask: Any
    action_valid_mask: Any
    weights: Any
    source_sample_batch: Any
    next_action_mask: Any | None = None
    done: Any | None = None
    terminated: Any | None = None
    truncated: Any | None = None


@dataclass(frozen=True, slots=True)
class CompactMuZeroLearnerStepResultV1:
    """Structured output for one compact learner edge call."""

    telemetry: dict[str, Any]


class CompactMuZeroLearnerEdgeV1:
    """Run a real MuZero-style optimizer update on compact tensors."""

    def __init__(
        self,
        *,
        model: Any,
        config: CompactMuZeroLearnerConfigV1 | None = None,
        optimizer: Any | None = None,
    ) -> None:
        import torch
        from lzero.policy import (
            DiscreteSupport,
            cross_entropy_loss,
            mz_network_output_unpack,
            phi_transform,
            scalar_transform,
        )

        self.config = config or CompactMuZeroLearnerConfigV1()
        _validate_config(self.config)
        self.device = _resolve_device(torch, self.config.device)
        self.model = model.to(self.device)
        self.model.train()
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(self.config.learning_rate),
                weight_decay=float(self.config.weight_decay),
            )
        self._prebuilt_batch_full_validation_signatures: set[tuple[Any, ...]] = set()
        self._prebuilt_batch_deep_validation_count = 0
        self._prebuilt_batch_fast_validation_count = 0
        self._discrete_support = DiscreteSupport
        self._cross_entropy_loss = cross_entropy_loss
        self._mz_network_output_unpack = mz_network_output_unpack
        self._phi_transform = phi_transform
        self._scalar_transform = scalar_transform

    def train_on_sample_batch(
        self,
        sample_batch: Any,
        *,
        train_steps: int = 1,
    ) -> CompactMuZeroLearnerStepResultV1:
        """Build tensors from a compact sample and run optimizer updates."""

        import torch

        started = time.perf_counter()
        collect_cuda_memory = bool(self.config.collect_cuda_memory_telemetry)
        if self.device.type == "cuda" and collect_cuda_memory:
            torch.cuda.reset_peak_memory_stats(self.device)
        cuda_memory_before_batch = _cuda_memory_telemetry(
            torch,
            self.device,
            "before_batch",
            enabled=collect_cuda_memory,
        )
        batch = build_compact_muzero_learner_batch_v1(
            sample_batch,
            config=self.config,
            device=self.device,
        )
        cuda_memory_after_batch = _cuda_memory_telemetry(
            torch,
            self.device,
            "after_batch",
            enabled=collect_cuda_memory,
        )
        return self.train_on_learner_batch(
            batch,
            train_steps=train_steps,
            _started=started,
            _cuda_memory_before_batch=cuda_memory_before_batch,
            _cuda_memory_after_batch=cuda_memory_after_batch,
            _prebuilt_batch_used=False,
        )

    def train_on_learner_batch(
        self,
        learner_batch: CompactMuZeroLearnerBatchV1,
        *,
        train_steps: int = 1,
        _started: float | None = None,
        _cuda_memory_before_batch: dict[str, Any] | None = None,
        _cuda_memory_after_batch: dict[str, Any] | None = None,
        _prebuilt_batch_used: bool = True,
    ) -> CompactMuZeroLearnerStepResultV1:
        """Run optimizer updates from an already-built compact learner batch."""

        import torch

        started = time.perf_counter() if _started is None else float(_started)
        collect_cuda_memory = bool(self.config.collect_cuda_memory_telemetry)
        cuda_sync_timing_requested = bool(self.config.cuda_sync_timing_diagnostics)
        cuda_sync_timing_enabled = bool(
            cuda_sync_timing_requested
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        cuda_sync_count = 0
        cuda_sync_sec = 0.0

        def sync_for_timing() -> None:
            nonlocal cuda_sync_count, cuda_sync_sec
            if not cuda_sync_timing_enabled:
                return
            sync_started = time.perf_counter()
            torch.cuda.synchronize(self.device)
            cuda_sync_sec += time.perf_counter() - sync_started
            cuda_sync_count += 1

        def start_timed_block() -> float:
            sync_for_timing()
            return time.perf_counter()

        def finish_timed_block(block_started: float) -> float:
            sync_for_timing()
            return time.perf_counter() - block_started

        if _started is None and self.device.type == "cuda" and collect_cuda_memory:
            torch.cuda.reset_peak_memory_stats(self.device)
        validation_started = start_timed_block()
        cuda_memory_before_batch = (
            _cuda_memory_telemetry(
                torch,
                self.device,
                "before_batch",
                enabled=collect_cuda_memory,
            )
            if _cuda_memory_before_batch is None
            else dict(_cuda_memory_before_batch)
        )
        learner_batch_metadata = dict(getattr(learner_batch, "metadata", {}) or {})
        prevalidated_batch = bool(
            learner_batch_metadata.get("compact_muzero_learner_batch_prevalidated")
        )
        validation_signature = _prebuilt_batch_validation_signature(
            learner_batch,
            metadata=learner_batch_metadata,
        )
        fast_validation_candidate = bool(
            _prebuilt_batch_used
            and self.config.fast_prebuilt_batch_validation_after_first_full
            and prevalidated_batch
        )
        fast_validation_used = bool(
            fast_validation_candidate
            and validation_signature in self._prebuilt_batch_full_validation_signatures
        )
        deep_validation = not fast_validation_used
        batch = _validate_prebuilt_learner_batch(
            learner_batch,
            config=self.config,
            torch=torch,
            device=self.device,
            deep=deep_validation,
        )
        if _prebuilt_batch_used:
            if deep_validation:
                self._prebuilt_batch_deep_validation_count += 1
            else:
                self._prebuilt_batch_fast_validation_count += 1
        if fast_validation_candidate and deep_validation:
            self._prebuilt_batch_full_validation_signatures.add(validation_signature)
        cuda_memory_after_batch = (
            _cuda_memory_telemetry(
                torch,
                self.device,
                "after_batch",
                enabled=collect_cuda_memory,
            )
            if _cuda_memory_after_batch is None
            else dict(_cuda_memory_after_batch)
        )
        validation_sec = finish_timed_block(validation_started)
        support = self._discrete_support(
            -int(self.config.support_scale),
            int(self.config.support_scale),
            delta=1,
        )
        train_steps_int = int(train_steps)
        if train_steps_int <= 0:
            raise ReplayCompatibilityError("train_steps must be positive")

        last_loss = None
        last_policy_loss = None
        last_value_loss = None
        last_reward_loss = None
        last_grad_norm = None
        zero_grad_sec = 0.0
        target_transform_sec = 0.0
        initial_inference_sec = 0.0
        recurrent_inference_sec = 0.0
        loss_build_sec = 0.0
        backward_sec = 0.0
        grad_clip_sec = 0.0
        optimizer_step_sec = 0.0
        loss_readback_sec = 0.0
        cuda_memory_before_backward: dict[str, Any] = _cuda_memory_telemetry(
            torch,
            self.device,
            "before_backward",
            enabled=collect_cuda_memory,
        )
        cuda_memory_after_backward: dict[str, Any] = _cuda_memory_telemetry(
            torch,
            self.device,
            "after_backward",
            enabled=collect_cuda_memory,
        )
        for _ in range(train_steps_int):
            zero_grad_started = start_timed_block()
            self.optimizer.zero_grad(set_to_none=True)
            zero_grad_sec += finish_timed_block(zero_grad_started)
            target_transform_started = start_timed_block()
            target_reward = self._phi_transform(
                support,
                self._scalar_transform(batch.target_reward.clone()),
            )
            target_value = self._phi_transform(
                support,
                self._scalar_transform(batch.target_value.clone()),
            )
            target_transform_sec += finish_timed_block(target_transform_started)

            initial_inference_started = start_timed_block()
            network_output = self.model.initial_inference(batch.observation)
            latent_state, _initial_reward, value, policy_logits = (
                self._mz_network_output_unpack(network_output)
            )
            _validate_network_output_shapes(
                policy_logits=policy_logits,
                value=value,
                reward=None,
                support_scale=int(self.config.support_scale),
            )
            initial_inference_sec += finish_timed_block(initial_inference_started)
            loss_build_started = start_timed_block()
            policy_loss = self._cross_entropy_loss(
                policy_logits,
                batch.target_policy[:, 0],
            ) * batch.target_policy_mask[:, 0].to(dtype=policy_logits.dtype)
            value_loss = self._cross_entropy_loss(
                value,
                target_value[:, 0],
            ) * batch.target_value_mask[:, 0].to(dtype=value.dtype)
            reward_loss = None

            current_latent = latent_state
            loss_build_sec += finish_timed_block(loss_build_started)
            for unroll_index in range(int(self.config.num_unroll_steps)):
                recurrent_inference_started = start_timed_block()
                recurrent_output = self.model.recurrent_inference(
                    current_latent,
                    batch.action[:, unroll_index],
                )
                current_latent, reward, next_value, next_policy_logits = (
                    self._mz_network_output_unpack(recurrent_output)
                )
                _validate_network_output_shapes(
                    policy_logits=next_policy_logits,
                    value=next_value,
                    reward=reward,
                    support_scale=int(self.config.support_scale),
                )
                recurrent_inference_sec += finish_timed_block(recurrent_inference_started)
                loss_build_started = start_timed_block()
                policy_loss = policy_loss + self._cross_entropy_loss(
                    next_policy_logits,
                    batch.target_policy[:, unroll_index + 1],
                ) * batch.target_policy_mask[:, unroll_index + 1].to(
                    dtype=next_policy_logits.dtype
                )
                value_loss = value_loss + self._cross_entropy_loss(
                    next_value,
                    target_value[:, unroll_index + 1],
                ) * batch.target_value_mask[:, unroll_index + 1].to(
                    dtype=next_value.dtype
                )
                step_reward_loss = self._cross_entropy_loss(
                    reward,
                    target_reward[:, unroll_index],
                ) * batch.target_reward_mask[:, unroll_index].to(dtype=reward.dtype)
                reward_loss = (
                    step_reward_loss
                    if reward_loss is None
                    else reward_loss + step_reward_loss
                )
                loss_build_sec += finish_timed_block(loss_build_started)
            if reward_loss is None:
                raise ReplayCompatibilityError("compact MuZero unroll produced no reward loss")

            loss_build_started = start_timed_block()
            row_loss = (
                float(self.config.policy_loss_weight) * policy_loss
                + float(self.config.value_loss_weight) * value_loss
                + float(self.config.reward_loss_weight) * reward_loss
            )
            weighted_loss = (batch.weights * row_loss).mean()
            loss_build_sec += finish_timed_block(loss_build_started)
            backward_started = start_timed_block()
            cuda_memory_before_backward = _cuda_memory_telemetry(
                torch,
                self.device,
                "before_backward",
                enabled=collect_cuda_memory,
            )
            weighted_loss.backward()
            cuda_memory_after_backward = _cuda_memory_telemetry(
                torch,
                self.device,
                "after_backward",
                enabled=collect_cuda_memory,
            )
            backward_sec += finish_timed_block(backward_started)
            grad_clip_started = start_timed_block()
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.config.grad_clip_value),
                )
                .detach()
                .cpu()
                .item()
            )
            grad_clip_sec += finish_timed_block(grad_clip_started)
            optimizer_step_started = start_timed_block()
            self.optimizer.step()
            optimizer_step_sec += finish_timed_block(optimizer_step_started)
            loss_readback_started = start_timed_block()
            last_loss = float(weighted_loss.detach().cpu().item())
            last_policy_loss = float(policy_loss.mean().detach().cpu().item())
            last_value_loss = float(value_loss.mean().detach().cpu().item())
            last_reward_loss = float(reward_loss.mean().detach().cpu().item())
            loss_readback_sec += finish_timed_block(loss_readback_started)

        final_sync_sec = _synchronize_device_for_timing(torch, self.device)
        elapsed = time.perf_counter() - started
        accounted_sec = (
            validation_sec
            + zero_grad_sec
            + target_transform_sec
            + initial_inference_sec
            + recurrent_inference_sec
            + loss_build_sec
            + backward_sec
            + grad_clip_sec
            + optimizer_step_sec
            + loss_readback_sec
            + final_sync_sec
        )
        cuda_memory_after_train = _cuda_memory_telemetry(
            torch,
            self.device,
            "after_train",
            enabled=collect_cuda_memory,
        )
        telemetry = {
            "compact_muzero_learner_edge_impl_id": COMPACT_MUZERO_LEARNER_EDGE_IMPL_ID,
            "compact_muzero_learner_profile_only": True,
            "compact_muzero_learner_calls_train_muzero": False,
            "compact_muzero_learner_update_claim": True,
            "compact_muzero_learner_stock_lightzero_parity_claim": False,
            "compact_muzero_learner_target_mode": str(self.config.target_mode),
            "compact_muzero_learner_num_unroll_steps": int(
                self.config.num_unroll_steps
            ),
            "compact_muzero_learner_prebuilt_batch_used": bool(_prebuilt_batch_used),
            "compact_muzero_learner_device": str(self.device),
            "compact_muzero_learner_sample_rows": int(batch.observation.shape[0]),
            "compact_muzero_learner_train_steps": train_steps_int,
            "compact_muzero_learner_sec": float(elapsed),
            "compact_muzero_learner_validation_sec": float(validation_sec),
            "compact_muzero_learner_zero_grad_sec": float(zero_grad_sec),
            "compact_muzero_learner_target_transform_sec": float(
                target_transform_sec
            ),
            "compact_muzero_learner_initial_inference_sec": float(
                initial_inference_sec
            ),
            "compact_muzero_learner_recurrent_inference_sec": float(
                recurrent_inference_sec
            ),
            "compact_muzero_learner_loss_build_sec": float(loss_build_sec),
            "compact_muzero_learner_backward_sec": float(backward_sec),
            "compact_muzero_learner_grad_clip_sec": float(grad_clip_sec),
            "compact_muzero_learner_optimizer_step_sec": float(optimizer_step_sec),
            "compact_muzero_learner_loss_readback_sec": float(loss_readback_sec),
            "compact_muzero_learner_final_sync_sec": float(final_sync_sec),
            "compact_muzero_learner_cuda_sync_timing_diagnostics": bool(
                cuda_sync_timing_requested
            ),
            "compact_muzero_learner_cuda_sync_timing_enabled": bool(
                cuda_sync_timing_enabled
            ),
            "compact_muzero_learner_cuda_sync_count": int(cuda_sync_count),
            "compact_muzero_learner_cuda_sync_sec": float(cuda_sync_sec),
            "compact_muzero_learner_accounted_sec": float(accounted_sec),
            "compact_muzero_learner_residual_sec": float(elapsed - accounted_sec),
            "compact_muzero_learner_loss": last_loss,
            "compact_muzero_learner_policy_loss": last_policy_loss,
            "compact_muzero_learner_value_loss": last_value_loss,
            "compact_muzero_learner_reward_loss": last_reward_loss,
            "compact_muzero_learner_grad_norm_before_clip": last_grad_norm,
            **cuda_memory_before_batch,
            **cuda_memory_after_batch,
            **cuda_memory_before_backward,
            **cuda_memory_after_backward,
            **cuda_memory_after_train,
            **batch.metadata,
        }
        telemetry.update(
            {
                "compact_muzero_learner_edge_impl_id": COMPACT_MUZERO_LEARNER_EDGE_IMPL_ID,
                "compact_muzero_learner_profile_only": True,
                "compact_muzero_learner_calls_train_muzero": False,
                "compact_muzero_learner_update_claim": True,
                "compact_muzero_learner_stock_lightzero_parity_claim": False,
                "compact_muzero_learner_target_mode": str(self.config.target_mode),
                "compact_muzero_learner_num_unroll_steps": int(
                    self.config.num_unroll_steps
                ),
                "compact_muzero_learner_prebuilt_batch_used": bool(
                    _prebuilt_batch_used
                ),
                "compact_muzero_learner_device": str(self.device),
                "compact_muzero_learner_sample_rows": int(batch.observation.shape[0]),
                "compact_muzero_learner_train_steps": train_steps_int,
                "compact_muzero_learner_sec": float(elapsed),
                "compact_muzero_learner_validation_sec": float(validation_sec),
                "compact_muzero_learner_zero_grad_sec": float(zero_grad_sec),
                "compact_muzero_learner_target_transform_sec": float(
                    target_transform_sec
                ),
                "compact_muzero_learner_initial_inference_sec": float(
                    initial_inference_sec
                ),
                "compact_muzero_learner_recurrent_inference_sec": float(
                    recurrent_inference_sec
                ),
                "compact_muzero_learner_loss_build_sec": float(loss_build_sec),
                "compact_muzero_learner_backward_sec": float(backward_sec),
                "compact_muzero_learner_grad_clip_sec": float(grad_clip_sec),
                "compact_muzero_learner_optimizer_step_sec": float(
                    optimizer_step_sec
                ),
                "compact_muzero_learner_loss_readback_sec": float(
                    loss_readback_sec
                ),
                "compact_muzero_learner_final_sync_sec": float(final_sync_sec),
                "compact_muzero_learner_cuda_sync_timing_diagnostics": bool(
                    cuda_sync_timing_requested
                ),
                "compact_muzero_learner_cuda_sync_timing_enabled": bool(
                    cuda_sync_timing_enabled
                ),
                "compact_muzero_learner_cuda_sync_count": int(cuda_sync_count),
                "compact_muzero_learner_cuda_sync_sec": float(cuda_sync_sec),
                "compact_muzero_learner_accounted_sec": float(accounted_sec),
                "compact_muzero_learner_residual_sec": float(
                    elapsed - accounted_sec
                ),
                "compact_muzero_learner_loss": last_loss,
                "compact_muzero_learner_policy_loss": last_policy_loss,
                "compact_muzero_learner_value_loss": last_value_loss,
                "compact_muzero_learner_reward_loss": last_reward_loss,
                "compact_muzero_learner_grad_norm_before_clip": last_grad_norm,
            }
        )
        telemetry.update(
            {
                "compact_muzero_learner_cuda_memory_telemetry_enabled": bool(
                    collect_cuda_memory
                    and self.device.type == "cuda"
                    and torch.cuda.is_available()
                ),
                "compact_muzero_learner_prebuilt_batch_validation_deep": bool(
                    deep_validation
                ),
                "compact_muzero_learner_prebuilt_batch_fast_validation_candidate": bool(
                    fast_validation_candidate
                ),
                "compact_muzero_learner_prebuilt_batch_fast_validation_used": bool(
                    fast_validation_used
                ),
                "compact_muzero_learner_prebuilt_batch_prevalidated": bool(
                    prevalidated_batch
                ),
                "compact_muzero_learner_prebuilt_batch_deep_validation_count": int(
                    self._prebuilt_batch_deep_validation_count
                ),
                "compact_muzero_learner_prebuilt_batch_fast_validation_count": int(
                    self._prebuilt_batch_fast_validation_count
                ),
            }
        )
        return CompactMuZeroLearnerStepResultV1(telemetry=telemetry)


def build_compact_muzero_learner_batch_v1(
    sample_batch: Any,
    *,
    config: CompactMuZeroLearnerConfigV1 | None = None,
    device: Any | None = None,
) -> CompactMuZeroLearnerBatchV1:
    """Convert a compact resident sample into MuZero learner tensors."""

    import torch

    cfg = config or CompactMuZeroLearnerConfigV1()
    _validate_config(cfg)
    resolved_device = _resolve_device(torch, cfg.device) if device is None else torch.device(device)
    sample_metadata = dict(getattr(sample_batch, "metadata", {}) or {})
    if bool(cfg.require_resident_sample) and not bool(
        sample_metadata.get("resident_device_sample_batch", False)
    ):
        raise ReplayCompatibilityError("compact MuZero learner requires resident sample batch")
    if bool(cfg.require_device_replay_rows) and not bool(
        sample_metadata.get("device_replay_index_rows_sample", False)
    ):
        raise ReplayCompatibilityError("compact MuZero learner requires device replay rows")

    num_unroll_steps = int(cfg.num_unroll_steps)
    next_policy_source = getattr(sample_batch, "next_policy_target", None)
    next_value_source = getattr(sample_batch, "next_root_value", None)
    next_action_mask_source = getattr(sample_batch, "next_action_mask", None)
    unroll_action_source = getattr(sample_batch, "unroll_action", None)
    unroll_reward_source = getattr(sample_batch, "unroll_reward", None)
    unroll_policy_source = getattr(sample_batch, "unroll_policy_target", None)
    unroll_value_source = getattr(sample_batch, "unroll_root_value", None)
    unroll_action_mask_source = getattr(sample_batch, "unroll_action_mask", None)
    action_valid_mask_source = getattr(sample_batch, "unroll_action_valid_mask", None)
    reward_valid_mask_source = getattr(sample_batch, "unroll_reward_valid_mask", None)
    policy_valid_mask_source = getattr(sample_batch, "unroll_policy_valid_mask", None)
    value_valid_mask_source = getattr(sample_batch, "unroll_value_valid_mask", None)
    done_source = getattr(sample_batch, "unroll_done", None)
    terminated_source = getattr(sample_batch, "unroll_terminated", None)
    truncated_source = getattr(sample_batch, "unroll_truncated", None)
    if num_unroll_steps == 1:
        if done_source is None:
            done_source = getattr(sample_batch, "done", None)
        if terminated_source is None:
            terminated_source = getattr(sample_batch, "terminated", None)
        if truncated_source is None:
            truncated_source = getattr(sample_batch, "truncated", None)
    target_mode = str(cfg.target_mode)
    if target_mode == COMPACT_MUZERO_TARGET_MODE_EXPLICIT_NEXT:
        if num_unroll_steps == 1 and (
            next_policy_source is None
            or next_value_source is None
            or next_action_mask_source is None
        ):
            raise ReplayCompatibilityError(
                "explicit_next_targets mode requires next_policy_target, "
                "next_root_value, and next_action_mask"
            )
        if num_unroll_steps > 1 and (
            unroll_action_source is None
            or unroll_reward_source is None
            or unroll_policy_source is None
            or unroll_value_source is None
            or unroll_action_mask_source is None
        ):
            raise ReplayCompatibilityError(
                "explicit_next_targets mode with num_unroll_steps>1 requires "
                "unroll_action, unroll_reward, unroll_policy_target, "
                "unroll_root_value, and unroll_action_mask"
            )
    elif target_mode == COMPACT_MUZERO_TARGET_MODE_REPEAT_CURRENT_PROFILE_ONLY:
        if num_unroll_steps != 1:
            raise ReplayCompatibilityError(
                "repeat_current_targets_profile_only only supports num_unroll_steps=1"
            )
        next_policy_source = sample_batch.policy_target
        next_value_source = sample_batch.root_value
    else:
        raise ReplayCompatibilityError(f"unknown compact MuZero target_mode {target_mode!r}")
    weights_source = getattr(sample_batch, "weights", None)
    if bool(cfg.require_resident_sample):
        _require_tensor_inputs_on_device(
            torch,
            {
                "observation": sample_batch.observation,
                "action": sample_batch.action,
                "action_mask": sample_batch.action_mask,
                "policy_target": sample_batch.policy_target,
                "root_value": sample_batch.root_value,
                "reward": sample_batch.reward,
                "next_action_mask": next_action_mask_source,
                "next_policy_target": next_policy_source,
                "next_root_value": next_value_source,
                "unroll_action": unroll_action_source,
                "unroll_reward": unroll_reward_source,
                "unroll_policy_target": unroll_policy_source,
                "unroll_root_value": unroll_value_source,
                "unroll_action_mask": unroll_action_mask_source,
                "unroll_action_valid_mask": action_valid_mask_source,
                "unroll_reward_valid_mask": reward_valid_mask_source,
                "unroll_policy_valid_mask": policy_valid_mask_source,
                "unroll_value_valid_mask": value_valid_mask_source,
                "unroll_done": done_source,
                "unroll_terminated": terminated_source,
                "unroll_truncated": truncated_source,
                "weights": weights_source,
            },
            device=resolved_device,
        )

    observation, observation_h2d = _as_device_tensor_with_h2d_bytes(
        sample_batch.observation,
        dtype=torch.float32,
        device=resolved_device,
    )
    if bool(cfg.normalize_uint8_observation) and _source_dtype_text(
        sample_batch.observation
    ).startswith("torch.uint8"):
        observation = observation.div(255.0)
    elif bool(cfg.normalize_uint8_observation) and _source_dtype_text(
        sample_batch.observation
    ) == "uint8":
        observation = observation.div(255.0)
    action_source = sample_batch.action if num_unroll_steps == 1 else unroll_action_source
    action, action_h2d = _as_device_tensor_with_h2d_bytes(
        action_source,
        dtype=torch.long,
        device=resolved_device,
    )
    if num_unroll_steps == 1:
        action = action.reshape(-1, 1)
    action_mask_source = (
        sample_batch.action_mask
        if num_unroll_steps == 1
        else unroll_action_mask_source
    )
    action_mask, mask_h2d = _as_device_tensor_with_h2d_bytes(
        action_mask_source,
        dtype=torch.bool,
        device=resolved_device,
    )
    if num_unroll_steps > 1:
        next_action_mask = None
        next_mask_h2d = 0
    elif next_action_mask_source is None:
        next_action_mask = None
        next_mask_h2d = 0
    else:
        next_action_mask, next_mask_h2d = _as_device_tensor_with_h2d_bytes(
            next_action_mask_source,
            dtype=torch.bool,
            device=resolved_device,
        )
    policy_source = (
        sample_batch.policy_target
        if num_unroll_steps == 1
        else unroll_policy_source
    )
    policy_target, policy_h2d = _as_device_tensor_with_h2d_bytes(
        policy_source,
        dtype=torch.float32,
        device=resolved_device,
    )
    root_value_source = (
        sample_batch.root_value
        if num_unroll_steps == 1
        else unroll_value_source
    )
    root_value, root_value_h2d = _as_device_tensor_with_h2d_bytes(
        root_value_source,
        dtype=torch.float32,
        device=resolved_device,
    )
    reward_source = sample_batch.reward if num_unroll_steps == 1 else unroll_reward_source
    reward, reward_h2d = _as_device_tensor_with_h2d_bytes(
        reward_source,
        dtype=torch.float32,
        device=resolved_device,
    )

    if num_unroll_steps == 1:
        next_policy_target, next_policy_h2d = _as_device_tensor_with_h2d_bytes(
            next_policy_source,
            dtype=torch.float32,
            device=resolved_device,
        )
        next_root_value, next_root_value_h2d = _as_device_tensor_with_h2d_bytes(
            next_value_source,
            dtype=torch.float32,
            device=resolved_device,
        )
    else:
        next_policy_target = None
        next_root_value = None
        next_policy_h2d = 0
        next_root_value_h2d = 0
    if weights_source is None:
        weights = torch.ones((int(observation.shape[0]),), dtype=torch.float32, device=resolved_device)
        weights_h2d = 0
    else:
        weights, weights_h2d = _as_device_tensor_with_h2d_bytes(
            weights_source,
            dtype=torch.float32,
            device=resolved_device,
        )
    row_count = int(observation.shape[0])
    action_valid_mask, action_valid_h2d = _optional_bool_tensor_with_h2d_bytes(
        action_valid_mask_source,
        device=resolved_device,
    )
    reward_valid_mask, reward_valid_h2d = _optional_bool_tensor_with_h2d_bytes(
        reward_valid_mask_source,
        device=resolved_device,
    )
    policy_valid_mask, policy_valid_h2d = _optional_bool_tensor_with_h2d_bytes(
        policy_valid_mask_source,
        device=resolved_device,
    )
    value_valid_mask, value_valid_h2d = _optional_bool_tensor_with_h2d_bytes(
        value_valid_mask_source,
        device=resolved_device,
    )
    done, done_h2d = _optional_bool_tensor_with_h2d_bytes(
        done_source,
        device=resolved_device,
    )
    terminated, terminated_h2d = _optional_bool_tensor_with_h2d_bytes(
        terminated_source,
        device=resolved_device,
    )
    truncated, truncated_h2d = _optional_bool_tensor_with_h2d_bytes(
        truncated_source,
        device=resolved_device,
    )
    if done is not None:
        done = _reshape_unroll_bool_tensor(
            done,
            name="unroll_done",
            row_count=row_count,
            unroll_steps=num_unroll_steps,
        )
    if terminated is not None:
        terminated = _reshape_unroll_bool_tensor(
            terminated,
            name="unroll_terminated",
            row_count=row_count,
            unroll_steps=num_unroll_steps,
        )
    if truncated is not None:
        truncated = _reshape_unroll_bool_tensor(
            truncated,
            name="unroll_truncated",
            row_count=row_count,
            unroll_steps=num_unroll_steps,
        )
    validity_mask_sources = (
        action_valid_mask,
        reward_valid_mask,
        policy_valid_mask,
        value_valid_mask,
    )
    any_validity_mask_source = any(mask is not None for mask in validity_mask_sources)
    all_validity_mask_sources = all(mask is not None for mask in validity_mask_sources)
    has_terminal_provenance = bool(
        (done is not None and bool(done.any()))
        or (terminated is not None and bool(terminated.any()))
        or (truncated is not None and bool(truncated.any()))
    )
    if (any_validity_mask_source or has_terminal_provenance) and not all_validity_mask_sources:
        raise ReplayCompatibilityError(
            "terminal compact MuZero unrolls require complete validity masks"
        )
    if action_valid_mask is None:
        action_valid_mask = torch.ones(
            (row_count, num_unroll_steps),
            dtype=torch.bool,
            device=resolved_device,
        )
        reward_valid_mask = torch.ones_like(action_valid_mask)
        policy_valid_mask = torch.ones(
            (row_count, num_unroll_steps + 1),
            dtype=torch.bool,
            device=resolved_device,
        )
        value_valid_mask = torch.ones_like(policy_valid_mask)
    else:
        action_valid_mask = _reshape_unroll_bool_tensor(
            action_valid_mask,
            name="unroll_action_valid_mask",
            row_count=row_count,
            unroll_steps=num_unroll_steps,
        )
        reward_valid_mask = _reshape_unroll_bool_tensor(
            reward_valid_mask,
            name="unroll_reward_valid_mask",
            row_count=row_count,
            unroll_steps=num_unroll_steps,
        )
        policy_valid_mask = _reshape_unroll_state_bool_tensor(
            policy_valid_mask,
            name="unroll_policy_valid_mask",
            row_count=row_count,
            unroll_steps=num_unroll_steps,
        )
        value_valid_mask = _reshape_unroll_state_bool_tensor(
            value_valid_mask,
            name="unroll_value_valid_mask",
            row_count=row_count,
            unroll_steps=num_unroll_steps,
        )

    _validate_batch_tensors(
        observation=observation,
        action=action,
        action_mask=action_mask,
        policy_target=policy_target,
        next_policy_target=next_policy_target,
        root_value=root_value,
        next_root_value=next_root_value,
        reward=reward,
        weights=weights,
        next_action_mask=next_action_mask,
        action_valid_mask=action_valid_mask,
        reward_valid_mask=reward_valid_mask,
        policy_valid_mask=policy_valid_mask,
        value_valid_mask=value_valid_mask,
        done=done,
        terminated=terminated,
        truncated=truncated,
        num_unroll_steps=num_unroll_steps,
    )
    if num_unroll_steps == 1:
        target_policy = torch.stack((policy_target, next_policy_target), dim=1)
        target_value = torch.stack(
            (
                root_value.reshape(-1),
                next_root_value.reshape(-1),
            ),
            dim=1,
        )
        target_reward = reward.reshape(-1, 1)
    else:
        target_policy = policy_target
        target_value = root_value
        target_reward = reward
    target_policy = target_policy.contiguous()
    target_value = target_value.contiguous()
    target_reward = target_reward.contiguous()
    input_bytes = int(
        _tensor_nbytes(observation)
        + _tensor_nbytes(action)
        + _tensor_nbytes(action_mask)
        + (0 if next_action_mask is None else _tensor_nbytes(next_action_mask))
        + _tensor_nbytes(policy_target)
        + _tensor_nbytes(root_value)
        + _tensor_nbytes(reward)
        + _tensor_nbytes(action_valid_mask)
        + _tensor_nbytes(reward_valid_mask)
        + _tensor_nbytes(policy_valid_mask)
        + _tensor_nbytes(value_valid_mask)
        + _optional_tensor_nbytes(done)
        + _optional_tensor_nbytes(terminated)
        + _optional_tensor_nbytes(truncated)
        + _optional_tensor_nbytes(next_policy_target)
        + _optional_tensor_nbytes(next_root_value)
        + _tensor_nbytes(weights)
    )
    metadata = {
        "compact_muzero_learner_batch_schema_id": COMPACT_MUZERO_LEARNER_BATCH_SCHEMA_ID,
        "compact_muzero_learner_num_unroll_steps": int(cfg.num_unroll_steps),
        "compact_muzero_learner_target_mode": target_mode,
        "compact_muzero_learner_batch_device": str(resolved_device),
        "compact_muzero_learner_batch_rows": int(observation.shape[0]),
        "compact_muzero_learner_observation_shape": list(observation.shape),
        "compact_muzero_learner_action_shape": list(action.shape),
        "compact_muzero_learner_target_reward_shape": list(target_reward.shape),
        "compact_muzero_learner_target_value_shape": list(target_value.shape),
        "compact_muzero_learner_target_policy_shape": list(target_policy.shape),
        "compact_muzero_learner_action_valid_count": int(
            action_valid_mask.sum().detach().cpu().item()
        ),
        "compact_muzero_learner_reward_valid_count": int(
            reward_valid_mask.sum().detach().cpu().item()
        ),
        "compact_muzero_learner_policy_valid_count": int(
            policy_valid_mask.sum().detach().cpu().item()
        ),
        "compact_muzero_learner_value_valid_count": int(
            value_valid_mask.sum().detach().cpu().item()
        ),
        "compact_muzero_learner_done_count": 0
        if done is None
        else int(done.sum().detach().cpu().item()),
        "compact_muzero_learner_truncated_count": 0
        if truncated is None
        else int(truncated.sum().detach().cpu().item()),
        "compact_muzero_learner_next_action_mask_present": next_action_mask is not None,
        "compact_muzero_learner_resident_sample_used": bool(
            sample_metadata.get("resident_device_sample_batch", False)
        ),
        "compact_muzero_learner_device_replay_index_rows_sample": bool(
            sample_metadata.get("device_replay_index_rows_sample", False)
        ),
        "compact_muzero_learner_input_bytes": input_bytes,
        "compact_muzero_learner_observation_h2d_bytes": int(observation_h2d),
        "compact_muzero_learner_input_h2d_bytes": int(
            observation_h2d
            + action_h2d
            + mask_h2d
            + policy_h2d
            + root_value_h2d
            + reward_h2d
            + action_valid_h2d
            + reward_valid_h2d
            + policy_valid_h2d
            + value_valid_h2d
            + done_h2d
            + terminated_h2d
            + truncated_h2d
            + next_mask_h2d
            + next_policy_h2d
            + next_root_value_h2d
            + weights_h2d
        ),
        "compact_muzero_learner_host_fallback_allowed": False,
        "compact_muzero_learner_batch_prevalidated": True,
        "compact_muzero_learner_batch_prevalidation_source": (
            "build_compact_muzero_learner_batch_v1"
        ),
    }
    return CompactMuZeroLearnerBatchV1(
        metadata=metadata,
        observation=observation.contiguous(),
        action=action.contiguous(),
        action_mask=action_mask.contiguous(),
        target_reward=target_reward,
        target_value=target_value,
        target_policy=target_policy,
        target_reward_mask=reward_valid_mask.contiguous(),
        target_value_mask=value_valid_mask.contiguous(),
        target_policy_mask=policy_valid_mask.contiguous(),
        action_valid_mask=action_valid_mask.contiguous(),
        weights=weights.reshape(-1).contiguous(),
        source_sample_batch=sample_batch,
        next_action_mask=None if next_action_mask is None else next_action_mask.contiguous(),
        done=None if done is None else done.contiguous(),
        terminated=None if terminated is None else terminated.contiguous(),
        truncated=None if truncated is None else truncated.contiguous(),
    )


def _validate_prebuilt_learner_batch(
    batch: CompactMuZeroLearnerBatchV1,
    *,
    config: CompactMuZeroLearnerConfigV1,
    torch: Any,
    device: Any,
    deep: bool = True,
) -> CompactMuZeroLearnerBatchV1:
    _validate_config(config)
    metadata = dict(getattr(batch, "metadata", {}) or {})
    if metadata.get("compact_muzero_learner_batch_schema_id") != (
        COMPACT_MUZERO_LEARNER_BATCH_SCHEMA_ID
    ):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner batch schema mismatch")
    _require_tensor_inputs_on_device(
        torch,
        {
            "observation": getattr(batch, "observation", None),
            "action": getattr(batch, "action", None),
            "action_mask": getattr(batch, "action_mask", None),
            "target_reward": getattr(batch, "target_reward", None),
            "target_value": getattr(batch, "target_value", None),
            "target_policy": getattr(batch, "target_policy", None),
            "target_reward_mask": getattr(batch, "target_reward_mask", None),
            "target_value_mask": getattr(batch, "target_value_mask", None),
            "target_policy_mask": getattr(batch, "target_policy_mask", None),
            "action_valid_mask": getattr(batch, "action_valid_mask", None),
            "weights": getattr(batch, "weights", None),
            "next_action_mask": getattr(batch, "next_action_mask", None),
            "done": getattr(batch, "done", None),
            "terminated": getattr(batch, "terminated", None),
            "truncated": getattr(batch, "truncated", None),
        },
        device=device,
    )
    row_count = int(batch.observation.shape[0])
    if row_count <= 0:
        raise ReplayCompatibilityError("prebuilt compact MuZero learner batch is empty")
    if getattr(batch.observation, "ndim", None) != 4:
        raise ReplayCompatibilityError(
            "prebuilt compact MuZero learner observation shape mismatch"
        )
    if int(metadata.get("compact_muzero_learner_batch_rows", row_count)) != row_count:
        raise ReplayCompatibilityError("prebuilt compact MuZero learner batch row mismatch")
    num_unroll_steps = int(metadata.get("compact_muzero_learner_num_unroll_steps", 0))
    if num_unroll_steps <= 0:
        raise ReplayCompatibilityError("prebuilt compact MuZero learner batch missing unroll steps")
    if num_unroll_steps != int(config.num_unroll_steps):
        raise ReplayCompatibilityError(
            "prebuilt compact MuZero learner batch unroll steps do not match config"
        )
    target_mode = str(metadata.get("compact_muzero_learner_target_mode") or "")
    if target_mode != str(config.target_mode):
        raise ReplayCompatibilityError(
            "prebuilt compact MuZero learner batch target mode does not match config"
        )
    if bool(config.require_resident_sample) and not bool(
        metadata.get("compact_muzero_learner_resident_sample_used")
        or metadata.get("resident_device_sample_batch")
    ):
        raise ReplayCompatibilityError(
            "prebuilt compact MuZero learner batch must prove resident sample use"
        )
    if bool(config.require_device_replay_rows) and not bool(
        metadata.get("compact_muzero_learner_device_replay_index_rows_sample")
        or metadata.get("device_replay_index_rows_sample")
    ):
        raise ReplayCompatibilityError(
            "prebuilt compact MuZero learner batch must prove device replay rows"
        )
    if metadata.get("compact_muzero_learner_host_fallback_allowed") is True:
        raise ReplayCompatibilityError(
            "prebuilt compact MuZero learner batch must not allow host fallback"
        )
    if tuple(batch.action.shape) != (row_count, num_unroll_steps):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner action shape mismatch")
    action_mask_shape = tuple(batch.action_mask.shape)
    expected_unroll_action_mask_shape = (row_count, num_unroll_steps + 1, ACTION_COUNT)
    if num_unroll_steps == 1:
        allowed_action_mask_shapes = (
            (row_count, ACTION_COUNT),
            expected_unroll_action_mask_shape,
        )
    else:
        allowed_action_mask_shapes = (expected_unroll_action_mask_shape,)
    if action_mask_shape not in allowed_action_mask_shapes:
        raise ReplayCompatibilityError("prebuilt compact MuZero learner action mask shape mismatch")
    if tuple(batch.target_reward.shape) != (row_count, num_unroll_steps):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner reward shape mismatch")
    if tuple(batch.target_value.shape) != (row_count, num_unroll_steps + 1):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner value shape mismatch")
    if tuple(batch.target_policy.shape) != (row_count, num_unroll_steps + 1, ACTION_COUNT):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner policy shape mismatch")
    if tuple(batch.target_reward_mask.shape) != (row_count, num_unroll_steps):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner reward mask shape mismatch")
    if tuple(batch.target_value_mask.shape) != (row_count, num_unroll_steps + 1):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner value mask shape mismatch")
    if tuple(batch.target_policy_mask.shape) != (row_count, num_unroll_steps + 1):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner policy mask shape mismatch")
    if tuple(batch.action_valid_mask.shape) != (row_count, num_unroll_steps):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner action mask shape mismatch")
    if tuple(batch.weights.reshape(-1).shape) != (row_count,):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner weights shape mismatch")
    if batch.next_action_mask is not None and tuple(batch.next_action_mask.shape) != (
        row_count,
        ACTION_COUNT,
    ):
        raise ReplayCompatibilityError(
            "prebuilt compact MuZero learner next action mask shape mismatch"
        )
    if batch.done is not None and tuple(batch.done.shape) != (row_count, num_unroll_steps):
        raise ReplayCompatibilityError("prebuilt compact MuZero learner done shape mismatch")
    if batch.terminated is not None and tuple(batch.terminated.shape) != (
        row_count,
        num_unroll_steps,
    ):
        raise ReplayCompatibilityError(
            "prebuilt compact MuZero learner terminated shape mismatch"
        )
    if batch.truncated is not None and tuple(batch.truncated.shape) != (
        row_count,
        num_unroll_steps,
    ):
        raise ReplayCompatibilityError(
            "prebuilt compact MuZero learner truncated shape mismatch"
        )
    if not bool(deep):
        return batch
    for name, value in (
        ("target_reward", batch.target_reward),
        ("target_value", batch.target_value),
        ("target_policy", batch.target_policy),
        ("weights", batch.weights),
    ):
        if not torch.isfinite(value).all():
            raise ReplayCompatibilityError(f"prebuilt compact MuZero learner {name} must be finite")
    if num_unroll_steps == 1:
        policy_target = batch.target_policy[:, 0, :]
        next_policy_target = batch.target_policy[:, 1, :]
        root_value = batch.target_value[:, 0]
        next_root_value = batch.target_value[:, 1]
        reward = batch.target_reward[:, 0]
    else:
        policy_target = batch.target_policy
        next_policy_target = None
        root_value = batch.target_value
        next_root_value = None
        reward = batch.target_reward
    _validate_batch_tensors(
        observation=batch.observation,
        action=batch.action,
        action_mask=batch.action_mask,
        policy_target=policy_target,
        next_policy_target=next_policy_target,
        root_value=root_value,
        next_root_value=next_root_value,
        reward=reward,
        weights=batch.weights,
        next_action_mask=batch.next_action_mask,
        action_valid_mask=batch.action_valid_mask,
        reward_valid_mask=batch.target_reward_mask,
        policy_valid_mask=batch.target_policy_mask,
        value_valid_mask=batch.target_value_mask,
        done=batch.done,
        terminated=batch.terminated,
        truncated=batch.truncated,
        num_unroll_steps=num_unroll_steps,
    )
    return batch


def _validate_config(config: CompactMuZeroLearnerConfigV1) -> None:
    if int(config.num_unroll_steps) < 1:
        raise ReplayCompatibilityError("num_unroll_steps must be at least 1")
    if int(config.support_scale) < 1:
        raise ReplayCompatibilityError("support_scale must be at least 1")
    if float(config.learning_rate) <= 0.0:
        raise ReplayCompatibilityError("learning_rate must be positive")
    if float(config.grad_clip_value) <= 0.0:
        raise ReplayCompatibilityError("grad_clip_value must be positive")
    if str(config.target_mode) not in COMPACT_MUZERO_TARGET_MODES:
        allowed = ", ".join(COMPACT_MUZERO_TARGET_MODES)
        raise ReplayCompatibilityError(f"target_mode must be one of {allowed}")


def _tensor_signature(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return (
        tuple(getattr(value, "shape", ())),
        str(getattr(value, "dtype", "")),
        str(getattr(value, "device", "")),
    )


def _prebuilt_batch_validation_signature(
    batch: CompactMuZeroLearnerBatchV1,
    *,
    metadata: dict[str, Any],
) -> tuple[Any, ...]:
    return (
        str(metadata.get("compact_muzero_learner_batch_schema_id", "")),
        int(metadata.get("compact_muzero_learner_num_unroll_steps", -1)),
        str(metadata.get("compact_muzero_learner_target_mode", "")),
        str(metadata.get("compact_muzero_learner_batch_device", "")),
        bool(metadata.get("compact_muzero_learner_resident_sample_used", False)),
        bool(
            metadata.get(
                "compact_muzero_learner_device_replay_index_rows_sample",
                False,
            )
        ),
        bool(metadata.get("resident_grouped_device_replay_learner_batch", False)),
        bool(metadata.get("resident_grouped_device_direct_write_learner_batch", False)),
        str(metadata.get("compact_muzero_learner_batch_prevalidation_source", "")),
        str(metadata.get("source", "")),
        _tensor_signature(getattr(batch, "observation", None)),
        _tensor_signature(getattr(batch, "action", None)),
        _tensor_signature(getattr(batch, "action_mask", None)),
        _tensor_signature(getattr(batch, "target_reward", None)),
        _tensor_signature(getattr(batch, "target_value", None)),
        _tensor_signature(getattr(batch, "target_policy", None)),
        _tensor_signature(getattr(batch, "target_reward_mask", None)),
        _tensor_signature(getattr(batch, "target_value_mask", None)),
        _tensor_signature(getattr(batch, "target_policy_mask", None)),
        _tensor_signature(getattr(batch, "action_valid_mask", None)),
        _tensor_signature(getattr(batch, "weights", None)),
        _tensor_signature(getattr(batch, "next_action_mask", None)),
        _tensor_signature(getattr(batch, "done", None)),
        _tensor_signature(getattr(batch, "terminated", None)),
        _tensor_signature(getattr(batch, "truncated", None)),
    )


def _resolve_device(torch: Any, requested: str) -> Any:
    requested_text = str(requested)
    if requested_text == "auto":
        requested_text = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_text == "cuda" and not torch.cuda.is_available():
        raise ReplayCompatibilityError("compact MuZero learner requested CUDA but CUDA is unavailable")
    if requested_text == "mps" and not _mps_is_available(torch):
        raise ReplayCompatibilityError("compact MuZero learner requested MPS but MPS is unavailable")
    if requested_text not in {"cpu", "cuda", "mps"}:
        raise ReplayCompatibilityError("device must be 'auto', 'cpu', 'cuda', or 'mps'")
    return torch.device(requested_text)


def _mps_is_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    return bool(callable(is_available) and is_available())


def _synchronize_device_for_timing(torch: Any, device: Any) -> float:
    resolved_device = torch.device(device)
    sync_started = time.perf_counter()
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elif resolved_device.type == "mps" and _mps_is_available(torch):
        mps_module = getattr(torch, "mps", None)
        synchronize = getattr(mps_module, "synchronize", None)
        if callable(synchronize):
            synchronize()
        else:
            return 0.0
    else:
        return 0.0
    return time.perf_counter() - sync_started


def _cuda_memory_telemetry(
    torch: Any,
    device: Any,
    phase: str,
    *,
    enabled: bool = True,
) -> dict[str, Any]:
    prefix = f"compact_muzero_learner_cuda_{phase}"
    keys = {
        f"{prefix}_memory_allocated_bytes": None,
        f"{prefix}_memory_reserved_bytes": None,
        f"{prefix}_memory_peak_allocated_bytes": None,
        f"{prefix}_memory_peak_reserved_bytes": None,
        f"{prefix}_mem_get_info_free_bytes": None,
        f"{prefix}_mem_get_info_total_bytes": None,
        f"{prefix}_telemetry_enabled": False,
    }
    resolved_device = torch.device(device)
    if not bool(enabled) or resolved_device.type != "cuda" or not torch.cuda.is_available():
        return keys
    free_bytes, total_bytes = torch.cuda.mem_get_info(resolved_device)
    keys.update(
        {
            f"{prefix}_telemetry_enabled": True,
            f"{prefix}_memory_allocated_bytes": int(
                torch.cuda.memory_allocated(resolved_device)
            ),
            f"{prefix}_memory_reserved_bytes": int(
                torch.cuda.memory_reserved(resolved_device)
            ),
            f"{prefix}_memory_peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(resolved_device)
            ),
            f"{prefix}_memory_peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(resolved_device)
            ),
            f"{prefix}_mem_get_info_free_bytes": int(free_bytes),
            f"{prefix}_mem_get_info_total_bytes": int(total_bytes),
        }
    )
    return keys


def _as_device_tensor_with_h2d_bytes(
    value: Any,
    *,
    dtype: Any,
    device: Any,
) -> tuple[Any, int]:
    import torch

    target_device = torch.device(device)
    if isinstance(value, torch.Tensor):
        h2d_bytes = (
            int(value.numel() * value.element_size())
            if value.device.type == "cpu" and target_device.type != "cpu"
            else 0
        )
        return value.to(device=device, dtype=dtype, non_blocking=True), h2d_bytes
    array = np.asarray(value)
    h2d_bytes = int(array.nbytes) if target_device.type != "cpu" else 0
    return torch.as_tensor(array, dtype=dtype, device=device), h2d_bytes


def _require_tensor_inputs_on_device(
    torch: Any,
    values: dict[str, Any | None],
    *,
    device: Any,
) -> None:
    requested_device = torch.device(device)
    device_text = str(requested_device)
    for name, value in values.items():
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise ReplayCompatibilityError(
                f"compact MuZero learner resident input {name} must be a torch tensor"
            )
        actual_device = value.device
        on_requested_device = actual_device.type == requested_device.type and (
            requested_device.index is None or actual_device.index == requested_device.index
        )
        if not on_requested_device:
            raise ReplayCompatibilityError(
                f"compact MuZero learner resident input {name} must be on {device_text}"
            )


def _tensor_nbytes(value: Any) -> int:
    return int(value.numel() * value.element_size())


def _optional_tensor_nbytes(value: Any | None) -> int:
    return 0 if value is None else _tensor_nbytes(value)


def _optional_bool_tensor_with_h2d_bytes(
    value: Any | None,
    *,
    device: Any,
) -> tuple[Any | None, int]:
    import torch

    if value is None:
        return None, 0
    return _as_device_tensor_with_h2d_bytes(
        value,
        dtype=torch.bool,
        device=device,
    )


def _reshape_unroll_bool_tensor(
    value: Any,
    *,
    name: str,
    row_count: int,
    unroll_steps: int,
) -> Any:
    shape = (int(row_count), int(unroll_steps))
    if tuple(value.shape) == shape:
        return value.contiguous()
    if int(unroll_steps) == 1 and tuple(value.reshape(-1).shape) == (int(row_count),):
        return value.reshape(shape).contiguous()
    raise ReplayCompatibilityError(f"learner {name} must have shape [B,N]")


def _reshape_unroll_state_bool_tensor(
    value: Any,
    *,
    name: str,
    row_count: int,
    unroll_steps: int,
) -> Any:
    shape = (int(row_count), int(unroll_steps) + 1)
    if tuple(value.shape) == shape:
        return value.contiguous()
    raise ReplayCompatibilityError(f"learner {name} must have shape [B,N+1]")


def _validate_terminal_unroll_masks(
    *,
    done: Any,
    action_valid_mask: Any,
    reward_valid_mask: Any,
    policy_valid_mask: Any,
    value_valid_mask: Any,
) -> None:
    unroll_steps = int(action_valid_mask.shape[1])
    for step_index in range(unroll_steps):
        done_at_step = done[:, step_index]
        if not bool(done_at_step.any()):
            continue
        if not bool(action_valid_mask[:, step_index][done_at_step].all()):
            raise ReplayCompatibilityError(
                "learner terminal unroll transition requires a valid action"
            )
        if not bool(reward_valid_mask[:, step_index][done_at_step].all()):
            raise ReplayCompatibilityError(
                "learner terminal unroll transition requires a valid reward"
            )
        if step_index + 1 < unroll_steps:
            if bool(action_valid_mask[:, step_index + 1 :][done_at_step].any()):
                raise ReplayCompatibilityError(
                    "learner terminal unroll masks must truncate post-done actions"
                )
            if bool(reward_valid_mask[:, step_index + 1 :][done_at_step].any()):
                raise ReplayCompatibilityError(
                    "learner terminal unroll masks must truncate post-done rewards"
                )
        if step_index + 1 <= unroll_steps:
            if bool(policy_valid_mask[:, step_index + 1 :][done_at_step].any()):
                raise ReplayCompatibilityError(
                    "learner terminal unroll masks must truncate post-done policies"
                )
            if bool(value_valid_mask[:, step_index + 1 :][done_at_step].any()):
                raise ReplayCompatibilityError(
                    "learner terminal unroll masks must truncate post-done values"
                )


def _terminal_no_bootstrap_unroll_value_targets(
    *,
    root_value: Any,
    reward: Any,
    done: Any,
    reward_valid_mask: Any,
    value_valid_mask: Any,
) -> Any:
    import torch

    unroll_steps = int(reward.shape[1])
    if unroll_steps <= 0 or not bool(done.any()):
        return root_value.masked_fill(~value_valid_mask, 0.0)
    terminal_seen = torch.flip(
        torch.cumsum(torch.flip(done.to(dtype=torch.int64), dims=(1,)), dim=1),
        dims=(1,),
    ) > 0
    reward_for_return = reward.to(dtype=root_value.dtype).masked_fill(
        ~reward_valid_mask,
        0.0,
    )
    no_bootstrap_returns = torch.flip(
        torch.cumsum(torch.flip(reward_for_return, dims=(1,)), dim=1),
        dims=(1,),
    )
    expected = root_value.clone()
    terminal_state_mask = terminal_seen & value_valid_mask[:, :unroll_steps]
    expected[:, :unroll_steps] = torch.where(
        terminal_state_mask,
        no_bootstrap_returns,
        expected[:, :unroll_steps],
    )
    return expected.masked_fill(~value_valid_mask, 0.0)


def _validate_network_output_shapes(
    *,
    policy_logits: Any,
    value: Any,
    reward: Any | None,
    support_scale: int,
) -> None:
    expected_support_width = int(2 * support_scale + 1)
    if int(policy_logits.shape[-1]) != ACTION_COUNT:
        raise ReplayCompatibilityError(
            "compact MuZero learner policy head width does not match ACTION_COUNT"
        )
    if int(value.shape[-1]) != expected_support_width:
        raise ReplayCompatibilityError(
            "compact MuZero learner value head width does not match support_scale"
        )
    if reward is not None and int(reward.shape[-1]) != expected_support_width:
        raise ReplayCompatibilityError(
            "compact MuZero learner reward head width does not match support_scale"
        )


def _source_dtype_text(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return "" if dtype is None else str(dtype)


def _validate_batch_tensors(
    *,
    observation: Any,
    action: Any,
    action_mask: Any,
    policy_target: Any,
    next_policy_target: Any,
    root_value: Any,
    next_root_value: Any,
    reward: Any,
    weights: Any,
    next_action_mask: Any | None,
    action_valid_mask: Any,
    reward_valid_mask: Any,
    policy_valid_mask: Any,
    value_valid_mask: Any,
    done: Any | None,
    terminated: Any | None,
    truncated: Any | None,
    num_unroll_steps: int,
) -> None:
    import torch

    unroll_steps = int(num_unroll_steps)
    if observation.ndim != 4:
        raise ReplayCompatibilityError("learner observation must have shape [B,C,H,W]")
    row_count = int(observation.shape[0])
    if row_count <= 0:
        raise ReplayCompatibilityError("learner batch must contain at least one row")
    if unroll_steps <= 0:
        raise ReplayCompatibilityError("learner num_unroll_steps must be positive")
    if tuple(weights.reshape(-1).shape) != (row_count,):
        raise ReplayCompatibilityError("learner weights must have shape [B]")
    if not torch.isfinite(weights.reshape(-1)).all():
        raise ReplayCompatibilityError("learner weights must be finite")
    if bool((weights.reshape(-1) < 0.0).any()):
        raise ReplayCompatibilityError("learner weights must be nonnegative")
    if tuple(action_valid_mask.shape) != (row_count, unroll_steps):
        raise ReplayCompatibilityError("learner action_valid_mask must have shape [B,N]")
    if tuple(reward_valid_mask.shape) != (row_count, unroll_steps):
        raise ReplayCompatibilityError("learner reward_valid_mask must have shape [B,N]")
    if tuple(policy_valid_mask.shape) != (row_count, unroll_steps + 1):
        raise ReplayCompatibilityError(
            "learner policy_valid_mask must have shape [B,N+1]"
        )
    if tuple(value_valid_mask.shape) != (row_count, unroll_steps + 1):
        raise ReplayCompatibilityError("learner value_valid_mask must have shape [B,N+1]")
    if not bool(reward_valid_mask.any()):
        raise ReplayCompatibilityError("learner reward_valid_mask has no valid targets")
    if not bool(policy_valid_mask.any()):
        raise ReplayCompatibilityError("learner policy_valid_mask has no valid targets")
    if not bool(value_valid_mask.any()):
        raise ReplayCompatibilityError("learner value_valid_mask has no valid targets")
    if bool((reward_valid_mask & ~action_valid_mask).any()):
        raise ReplayCompatibilityError("learner reward_valid_mask requires valid actions")
    invalid_transition = ~action_valid_mask
    if bool(policy_valid_mask[:, 1:][invalid_transition].any()):
        raise ReplayCompatibilityError(
            "learner policy_valid_mask requires preceding valid actions"
        )
    if bool(value_valid_mask[:, 1:][invalid_transition].any()):
        raise ReplayCompatibilityError(
            "learner value_valid_mask requires preceding valid actions"
        )
    if done is not None and tuple(done.shape) != (row_count, unroll_steps):
        raise ReplayCompatibilityError("learner unroll_done must have shape [B,N]")
    if terminated is not None and tuple(terminated.shape) != (row_count, unroll_steps):
        raise ReplayCompatibilityError("learner unroll_terminated must have shape [B,N]")
    if truncated is not None and tuple(truncated.shape) != (row_count, unroll_steps):
        raise ReplayCompatibilityError("learner unroll_truncated must have shape [B,N]")
    if done is not None and terminated is not None:
        expected_done = terminated if truncated is None else (terminated | truncated)
        if bool((done != expected_done).any()):
            raise ReplayCompatibilityError(
                "learner unroll_done must equal unroll_terminated | unroll_truncated"
            )
    if truncated is not None and bool(truncated.any()):
        raise ReplayCompatibilityError(
            "learner truncated unroll bootstrap semantics are not supported"
        )
    if done is not None and bool(done.any()):
        _validate_terminal_unroll_masks(
            done=done,
            action_valid_mask=action_valid_mask,
            reward_valid_mask=reward_valid_mask,
            policy_valid_mask=policy_valid_mask,
            value_valid_mask=value_valid_mask,
        )
    if unroll_steps == 1:
        if tuple(action.shape) != (row_count, 1):
            raise ReplayCompatibilityError("learner action must have shape [B,1]")
        if tuple(action_mask.shape) != (row_count, ACTION_COUNT):
            raise ReplayCompatibilityError(
                "learner action_mask must have shape [B,ACTION_COUNT]"
            )
        if next_action_mask is not None and tuple(next_action_mask.shape) != (
            row_count,
            ACTION_COUNT,
        ):
            raise ReplayCompatibilityError(
                "learner next_action_mask must have shape [B,ACTION_COUNT]"
            )
        if tuple(policy_target.shape) != (row_count, ACTION_COUNT):
            raise ReplayCompatibilityError(
                "learner policy_target must have shape [B,ACTION_COUNT]"
            )
        if next_policy_target is None or tuple(next_policy_target.shape) != (
            row_count,
            ACTION_COUNT,
        ):
            raise ReplayCompatibilityError(
                "learner next_policy_target must have shape [B,ACTION_COUNT]"
            )
        for name, value in (
            ("root_value", root_value),
            ("next_root_value", next_root_value),
            ("reward", reward),
        ):
            if value is None or tuple(value.reshape(-1).shape) != (row_count,):
                raise ReplayCompatibilityError(f"learner {name} must have shape [B]")
            if not torch.isfinite(value.reshape(-1)).all():
                raise ReplayCompatibilityError(f"learner {name} must be finite")
        if bool((action < 0).any()) or bool((action >= ACTION_COUNT).any()):
            raise ReplayCompatibilityError("learner action out of range")
        action_legal = action_mask.gather(1, action)
        if bool(action_valid_mask.any()) and not bool(
            action_legal[action_valid_mask].all()
        ):
            raise ReplayCompatibilityError("learner action must be legal under action_mask")
        if (
            not torch.isfinite(policy_target).all()
            or not torch.isfinite(next_policy_target).all()
        ):
            raise ReplayCompatibilityError("learner policy targets must be finite")
        if bool((policy_target < 0.0).any()) or bool(
            (next_policy_target < 0.0).any()
        ):
            raise ReplayCompatibilityError("learner policy targets must be nonnegative")
        if not torch.allclose(
            reward.reshape(-1, 1).masked_fill(reward_valid_mask, 0.0),
            torch.zeros_like(reward.reshape(-1, 1)),
            atol=1e-7,
        ):
            raise ReplayCompatibilityError("learner invalid reward targets must be zero")
        stacked_values = torch.stack(
            (
                root_value.reshape(-1),
                next_root_value.reshape(-1),
            ),
            dim=1,
        )
        if not torch.allclose(
            stacked_values.masked_fill(value_valid_mask, 0.0),
            torch.zeros_like(stacked_values),
            atol=1e-7,
        ):
            raise ReplayCompatibilityError("learner invalid value targets must be zero")
        if not torch.allclose(
            policy_target.sum(dim=1),
            policy_valid_mask[:, 0].to(dtype=policy_target.dtype),
            atol=1e-5,
        ):
            raise ReplayCompatibilityError(
                "learner policy_target rows must sum to validity mask"
            )
        if not torch.allclose(
            next_policy_target.sum(dim=1),
            policy_valid_mask[:, 1].to(dtype=next_policy_target.dtype),
            atol=1e-5,
        ):
            raise ReplayCompatibilityError(
                "learner next_policy_target rows must sum to validity mask"
            )
        if bool((policy_target[~action_mask] > 1e-7).any()):
            raise ReplayCompatibilityError(
                "learner policy_target assigns mass to illegal actions"
            )
        if next_action_mask is not None and bool(
            (next_policy_target[~next_action_mask] > 1e-7).any()
        ):
            raise ReplayCompatibilityError(
                "learner next_policy_target assigns mass to illegal next actions"
            )
        return

    if tuple(action.shape) != (row_count, unroll_steps):
        raise ReplayCompatibilityError("learner unroll_action must have shape [B,N]")
    if tuple(action_mask.shape) != (row_count, unroll_steps + 1, ACTION_COUNT):
        raise ReplayCompatibilityError(
            "learner unroll_action_mask must have shape [B,N+1,ACTION_COUNT]"
        )
    if tuple(policy_target.shape) != (row_count, unroll_steps + 1, ACTION_COUNT):
        raise ReplayCompatibilityError(
            "learner unroll_policy_target must have shape [B,N+1,ACTION_COUNT]"
        )
    if tuple(root_value.shape) != (row_count, unroll_steps + 1):
        raise ReplayCompatibilityError(
            "learner unroll_root_value must have shape [B,N+1]"
        )
    if tuple(reward.shape) != (row_count, unroll_steps):
        raise ReplayCompatibilityError("learner unroll_reward must have shape [B,N]")
    if not torch.isfinite(root_value).all() or not torch.isfinite(reward).all():
        raise ReplayCompatibilityError("learner unroll value/reward targets must be finite")
    if bool((action < 0).any()) or bool((action >= ACTION_COUNT).any()):
        raise ReplayCompatibilityError("learner unroll_action out of range")
    action_legal = action_mask[:, :unroll_steps, :].gather(2, action.unsqueeze(-1))
    if bool(action_valid_mask.any()) and not bool(
        action_legal.squeeze(-1)[action_valid_mask].all()
    ):
        raise ReplayCompatibilityError(
            "learner unroll_action must be legal under unroll_action_mask"
        )
    if not torch.allclose(
        reward.masked_fill(reward_valid_mask, 0.0),
        torch.zeros_like(reward),
        atol=1e-7,
    ):
        raise ReplayCompatibilityError("learner invalid unroll_reward targets must be zero")
    if not torch.allclose(
        root_value.masked_fill(value_valid_mask, 0.0),
        torch.zeros_like(root_value),
        atol=1e-7,
    ):
        raise ReplayCompatibilityError("learner invalid unroll_root_value targets must be zero")
    if done is not None and bool(done.any()):
        expected_terminal_values = _terminal_no_bootstrap_unroll_value_targets(
            root_value=root_value,
            reward=reward,
            done=done,
            reward_valid_mask=reward_valid_mask,
            value_valid_mask=value_valid_mask,
        )
        if not torch.allclose(root_value, expected_terminal_values, atol=1e-5):
            raise ReplayCompatibilityError(
                "learner terminal unroll_root_value targets must match "
                "no-bootstrap reward returns"
            )
    if not torch.isfinite(policy_target).all():
        raise ReplayCompatibilityError("learner unroll_policy_target must be finite")
    if bool((policy_target < 0.0).any()):
        raise ReplayCompatibilityError("learner unroll_policy_target must be nonnegative")
    if not torch.allclose(
        policy_target.sum(dim=2),
        policy_valid_mask.to(dtype=policy_target.dtype),
        atol=1e-5,
    ):
        raise ReplayCompatibilityError(
            "learner unroll_policy_target rows must sum to validity mask"
        )
    if bool((policy_target[~action_mask] > 1e-7).any()):
        raise ReplayCompatibilityError(
            "learner unroll_policy_target assigns mass to illegal actions"
        )


__all__ = [
    "COMPACT_MUZERO_LEARNER_BATCH_SCHEMA_ID",
    "COMPACT_MUZERO_LEARNER_EDGE_IMPL_ID",
    "COMPACT_MUZERO_TARGET_MODE_EXPLICIT_NEXT",
    "COMPACT_MUZERO_TARGET_MODE_REPEAT_CURRENT_PROFILE_ONLY",
    "COMPACT_MUZERO_TARGET_MODES",
    "CompactMuZeroLearnerBatchV1",
    "CompactMuZeroLearnerConfigV1",
    "CompactMuZeroLearnerEdgeV1",
    "CompactMuZeroLearnerStepResultV1",
    "build_compact_muzero_learner_batch_v1",
]
