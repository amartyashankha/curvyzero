"""Profile-only Torch helpers for a future compact search backend.

This module is deliberately not wired into trainers, launchers, or LightZero
CTree.  It gives the next optimizer backend a small fixed-shape surface that
can be unit-tested without importing the large Modal profile module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
)
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_refresh_handoff import (
    COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_model_state_digest_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_policy_refresh_metadata_from_state_v1,
)
from curvyzero.training.compact_search_service import COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID
from curvyzero.training.compact_search_service import (
    COMPACT_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import (
    COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import CompactSearchActionStepV1
from curvyzero.training.compact_search_service import CompactDeviceSearchReplayPayloadV1
from curvyzero.training.compact_search_service import CompactSearchReplayPayloadV1
from curvyzero.training.compact_search_service import compact_search_array_digest_v1
from curvyzero.training.compact_search_service import (
    compact_search_deferred_replay_payload_digest_v1,
)
from curvyzero.training.compact_search_service import (
    compact_search_replay_payload_digest_v1,
)
from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


COMPACT_TORCH_SEARCH_IMPL = "profile_only_compact_torch_fixed_shape_v0"
COMPACT_TORCH_SEARCH_SERVICE_IMPL = "compact_torch_device_tree_fixed_shape_v0"
COMPACT_TORCH_SEARCH_LABEL = "compact_torch_search_profile_only"
COMPACT_TORCH_SEARCH_SEMANTICS = (
    "profile-only fixed-shape Torch compact-search helper; "
    "not trainer-ready and not LightZero CTree"
)
COMPACT_TORCH_SEARCH_BACKEND_KIND = "not_lightzero_ctree"
COMPACT_TORCH_SEARCH_HELPER = "select_leaf+expand_backup"
COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC = "host_phase_sync"
COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC_CUDA_EVENT = "host_phase_sync_cuda_event"
COMPACT_TORCH_TIMING_MODE_HOST_FINAL_SYNC_ONLY = "host_final_sync_only"
COMPACT_TORCH_TIMING_MODE_CUDA_EVENT_FINAL_SYNC = "cuda_event_final_sync"
COMPACT_TORCH_TIMING_MODES = (
    COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC,
    COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC_CUDA_EVENT,
    COMPACT_TORCH_TIMING_MODE_HOST_FINAL_SYNC_ONLY,
    COMPACT_TORCH_TIMING_MODE_CUDA_EVENT_FINAL_SYNC,
)
COMPACT_TORCH_MODEL_COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune-no-cudagraphs",
    "max-autotune",
)
COMPACT_TORCH_INITIAL_INFERENCE_MODE_MODEL_METHOD = "model_method"
COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE = "direct_core"
COMPACT_TORCH_INITIAL_INFERENCE_MODES = (
    COMPACT_TORCH_INITIAL_INFERENCE_MODE_MODEL_METHOD,
    COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
)
COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS = "contiguous"
COMPACT_TORCH_MEMORY_FORMAT_CHANNELS_LAST = "channels_last"
COMPACT_TORCH_MEMORY_FORMATS = (
    COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS,
    COMPACT_TORCH_MEMORY_FORMAT_CHANNELS_LAST,
)


def _elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def _set_profile_metric(metadata: dict[str, Any], key: str, value: float) -> None:
    metric = float(value)
    metadata[key] = metric
    profile = metadata.get("profile_telemetry")
    if isinstance(profile, dict):
        profile[key] = metric


class _CompactTorchModelInferenceGuard:
    def __init__(self, *, torch: Any, model: Any) -> None:
        self._torch = torch
        self._model = model
        self._context: Any | None = None
        self._training_before: bool | None = None
        self._training_after: bool | None = None
        self._eval_applied = False
        self._inference_mode_used = False
        self._enter_sec = 0.0
        self._exit_sec = 0.0

    def __enter__(self) -> "_CompactTorchModelInferenceGuard":
        started = time.perf_counter()
        training = getattr(self._model, "training", None)
        self._training_before = None if training is None else bool(training)
        eval_fn = getattr(self._model, "eval", None)
        if callable(eval_fn):
            eval_fn()
            self._eval_applied = True
        inference_mode = getattr(self._torch, "inference_mode", None)
        if callable(inference_mode):
            self._context = inference_mode()
            self._inference_mode_used = True
        else:
            self._context = self._torch.no_grad()
        self._context.__enter__()
        self._enter_sec = _elapsed(started)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        assert self._context is not None
        started = time.perf_counter()
        try:
            self._context.__exit__(exc_type, exc, tb)
        finally:
            train_fn = getattr(self._model, "train", None)
            if self._training_before is not None and callable(train_fn):
                train_fn(bool(self._training_before))
            training = getattr(self._model, "training", None)
            self._training_after = None if training is None else bool(training)
            self._exit_sec = _elapsed(started)

    def telemetry(self) -> dict[str, Any]:
        return {
            "compact_torch_search_model_training_before_inference": (self._training_before),
            "compact_torch_search_model_training_after_inference": (self._training_after),
            "compact_torch_search_model_eval_applied_for_inference": (self._eval_applied),
            "compact_torch_search_model_inference_mode_used": (self._inference_mode_used),
            "compact_torch_search_service_inference_guard_enter_sec": float(self._enter_sec),
            "compact_torch_search_service_inference_guard_exit_sec": float(self._exit_sec),
            "compact_torch_search_service_inference_guard_total_sec": float(
                self._enter_sec + self._exit_sec
            ),
        }


def _finalize_action_step_profile_timing(
    pending_metadata: dict[str, Any],
    action_step: CompactSearchActionStepV1,
    *,
    action_service_started: float,
    accounted_sec: float,
    postprocess_base_sec: float,
    action_step_build_sec: float,
) -> CompactSearchActionStepV1:
    action_wall_sec = _elapsed(action_service_started)
    action_postprocess_sec = float(postprocess_base_sec) + float(action_step_build_sec)
    action_accounted_sec = float(accounted_sec)
    action_residual_sec = action_wall_sec - action_accounted_sec
    action_unaccounted_sec = max(0.0, action_residual_sec)
    action_overaccounted_sec = max(0.0, -action_residual_sec)
    for metadata in (pending_metadata, action_step.metadata):
        _set_profile_metric(
            metadata,
            "compact_torch_search_service_action_accounted_sec",
            action_accounted_sec,
        )
        _set_profile_metric(
            metadata,
            "compact_torch_search_service_action_step_build_sec",
            action_step_build_sec,
        )
        _set_profile_metric(
            metadata,
            "compact_torch_search_service_action_postprocess_sec",
            action_postprocess_sec,
        )
        _set_profile_metric(
            metadata,
            "compact_torch_search_service_action_wall_sec",
            action_wall_sec,
        )
        _set_profile_metric(
            metadata,
            "compact_torch_search_service_action_unaccounted_sec",
            action_unaccounted_sec,
        )
        _set_profile_metric(
            metadata,
            "compact_torch_search_service_action_residual_sec",
            action_residual_sec,
        )
        _set_profile_metric(
            metadata,
            "compact_torch_search_service_action_overaccounted_sec",
            action_overaccounted_sec,
        )
    return action_step


@dataclass(frozen=True, slots=True)
class _CompactTorchDecodedInitialOutput:
    policy_logits: Any
    value: Any
    latent_state: Any


class _CompactTorchInitialInferenceRuntime:
    def __init__(
        self,
        *,
        model: Any,
        initial_inference_fn: Any,
        requested_mode: str,
        direct_core_representation_fn: Any | None = None,
        direct_core_prediction_fn: Any | None = None,
    ) -> None:
        self._model = model
        self._initial_inference_fn = initial_inference_fn
        self._requested_mode = str(requested_mode)
        self._effective_mode = COMPACT_TORCH_INITIAL_INFERENCE_MODE_MODEL_METHOD
        self._runtime_status = "model_method_ready"
        self._fallback_count = 0.0
        self._direct_used = False
        self._direct_decoded_output = False
        self._direct_reward_materialized = False
        self._representation_sec = 0.0
        self._prediction_sec = 0.0
        self._pack_sec = 0.0
        self._representation_cuda_event: dict[str, Any] = _disabled_cuda_event_state()
        self._prediction_cuda_event: dict[str, Any] = _disabled_cuda_event_state()
        self._representation: Any | None = None
        self._prediction: Any | None = None
        if self._requested_mode == COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE:
            representation = direct_core_representation_fn
            if representation is None:
                representation = getattr(model, "_representation", None)
            prediction = direct_core_prediction_fn
            if prediction is None:
                prediction = getattr(model, "_prediction", None)
            if not callable(representation) or not callable(prediction):
                raise ValueError(
                    "direct_core initial_inference_mode requires callable "
                    "model._representation and model._prediction"
                )
            self._representation = representation
            self._prediction = prediction
            self._effective_mode = COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE
            self._runtime_status = "direct_core_ready"

    def run(
        self,
        obs_tensor: Any,
        *,
        torch: Any | None = None,
        device: Any | None = None,
        timing_mode: str = "",
    ) -> Any:
        if self._effective_mode != COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE:
            self._runtime_status = "model_method_used"
            self._direct_decoded_output = False
            self._direct_reward_materialized = False
            self._representation_cuda_event = _disabled_cuda_event_state()
            self._prediction_cuda_event = _disabled_cuda_event_state()
            return self._initial_inference_fn(obs_tensor)
        assert self._representation is not None
        assert self._prediction is not None
        representation_started = time.perf_counter()
        self._representation_cuda_event = _start_optional_cuda_event_timing(
            torch=torch,
            device=device,
            timing_mode=timing_mode,
        )
        latent_state = self._representation(obs_tensor)
        _finish_cuda_event_timing(self._representation_cuda_event)
        self._representation_sec = _elapsed(representation_started)
        prediction_started = time.perf_counter()
        self._prediction_cuda_event = _start_optional_cuda_event_timing(
            torch=torch,
            device=device,
            timing_mode=timing_mode,
        )
        prediction_output = self._prediction(latent_state)
        _finish_cuda_event_timing(self._prediction_cuda_event)
        self._prediction_sec = _elapsed(prediction_started)
        if not isinstance(prediction_output, (list, tuple)) or len(prediction_output) < 2:
            raise ValueError(
                "direct_core initial_inference_mode requires _prediction to return "
                "(policy_logits, value)"
            )
        policy_logits, value = prediction_output[0], prediction_output[1]
        pack_started = time.perf_counter()
        output = _CompactTorchDecodedInitialOutput(
            policy_logits=policy_logits,
            value=value,
            latent_state=latent_state,
        )
        self._pack_sec = _elapsed(pack_started)
        self._direct_used = True
        self._direct_decoded_output = True
        self._direct_reward_materialized = False
        self._runtime_status = "direct_core_used"
        return output

    def telemetry(self) -> dict[str, Any]:
        representation_cuda_event_sec, representation_cuda_event_status = _cuda_event_elapsed_sec(
            self._representation_cuda_event
        )
        prediction_cuda_event_sec, prediction_cuda_event_status = _cuda_event_elapsed_sec(
            self._prediction_cuda_event
        )
        direct_core_cuda_event_sec, direct_core_cuda_event_status = _cuda_event_elapsed_total_sec(
            [self._representation_cuda_event, self._prediction_cuda_event]
        )
        return {
            "compact_torch_search_initial_inference_mode_requested": (self._requested_mode),
            "compact_torch_search_initial_inference_mode_effective": (self._effective_mode),
            "compact_torch_search_initial_inference_direct_requested": (
                self._requested_mode == COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE
            ),
            "compact_torch_search_initial_inference_direct_used": self._direct_used,
            "compact_torch_search_initial_inference_direct_decoded_output": (
                self._direct_decoded_output
            ),
            "compact_torch_search_initial_inference_direct_reward_materialized": (
                self._direct_reward_materialized
            ),
            "compact_torch_search_initial_inference_runtime_status": (self._runtime_status),
            "compact_torch_search_initial_inference_fallback_count": float(self._fallback_count),
            "compact_torch_search_service_initial_inference_representation_sec": float(
                self._representation_sec
            ),
            "compact_torch_search_service_initial_inference_prediction_sec": float(
                self._prediction_sec
            ),
            "compact_torch_search_service_initial_inference_pack_sec": float(self._pack_sec),
            "compact_torch_search_service_initial_inference_representation_cuda_event_sec": float(
                representation_cuda_event_sec
            ),
            "compact_torch_search_service_initial_inference_representation_cuda_event_status": str(
                representation_cuda_event_status
            ),
            "compact_torch_search_service_initial_inference_prediction_cuda_event_sec": float(
                prediction_cuda_event_sec
            ),
            "compact_torch_search_service_initial_inference_prediction_cuda_event_status": str(
                prediction_cuda_event_status
            ),
            "compact_torch_search_service_initial_inference_direct_core_cuda_event_sec": float(
                direct_core_cuda_event_sec
            ),
            "compact_torch_search_service_initial_inference_direct_core_cuda_event_status": str(
                direct_core_cuda_event_status
            ),
        }


@dataclass(frozen=True, slots=True)
class CompactTorchCompileConfig:
    """Fixed-shape compile preconditions for the profile-only Torch lane."""

    request_compile: bool = True
    request_model_compile: bool = False
    require_cuda_device: bool = True
    require_torch_compile: bool = True
    require_model_compile: bool = False
    require_all_roots_active: bool = True
    require_all_actions_legal: bool = True
    require_root_noise_zero: bool = False
    compile_mode: str = "reduce-overhead"
    model_compile_mode: str = "reduce-overhead"
    fullgraph: bool = True
    model_fullgraph: bool = False
    recurrent_action_shape_mode: str = "auto"
    timing_mode: str = COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC
    initial_inference_mode: str = COMPACT_TORCH_INITIAL_INFERENCE_MODE_MODEL_METHOD
    observation_memory_format: str = COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS
    model_memory_format: str = COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS
    defer_one_simulation_replay_payload: bool = False
    action_count: int = ACTION_COUNT
    expected_root_count: int | None = None
    expected_observation_shape: tuple[int, ...] | None = None
    expected_device: str | None = None


@dataclass(frozen=True, slots=True)
class CompactTorchFixedShapeMasks:
    """Legal/active masks normalized for fixed-shape precondition checks.

    Forced masks are profile-only compile probes. They must not be treated as a
    legal-action contract for replay, training, or parity claims.
    """

    legal_mask: np.ndarray
    active_root_mask: np.ndarray
    root_count: int
    active_root_count: int
    action_count: int
    all_roots_active: bool
    all_actions_legal: bool
    forced_all_roots_active: bool
    forced_all_actions_legal: bool


@dataclass(frozen=True, slots=True)
class CompactTorchCompileEligibility:
    """Structured result for deterministic compile/fixed-shape telemetry."""

    eligible: bool
    status: str
    reason: str
    telemetry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CompactTorchTensorSearchRun:
    """Device-owned compact search output before replay arrays are materialized."""

    active_root_indices: np.ndarray
    selected_action: Any
    root_value: Any
    visit_policy: Any
    raw_visit_counts: Any
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PendingCompactTorchReplayPayload:
    """Delayed replay payload whose large tensors are still device-owned."""

    root_index: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    policy_env_id: np.ndarray
    visit_policy: Any
    root_value: Any
    raw_visit_counts: Any
    metadata: dict[str, Any]
    deferred_one_simulation_replay: bool = False
    selected_action_tensor: Any | None = None
    root_latent_state: Any | None = None
    recurrent_inference_fn: Any | None = None
    recurrent_action_shape_mode: str = "auto"
    timing_mode: str = COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC
    discount_factor: float = 0.997
    device: Any | None = None


@dataclass(frozen=True, slots=True)
class _ActiveRowSelection:
    """Active root rows plus the host copy bytes paid to gather them."""

    rows: np.ndarray
    copy_bytes: int


def _select_active_rows_without_copy_when_contiguous(
    rows: Any,
    active_root_indices: np.ndarray,
    *,
    name: str,
) -> _ActiveRowSelection:
    """Select active root rows, preserving a view when active rows are contiguous."""

    array = np.asarray(rows)
    if array.ndim == 0:
        raise ValueError(f"{name} must have a leading root dimension")
    active_indices = np.asarray(active_root_indices, dtype=np.int64)
    active_count = int(active_indices.size)
    if active_count == 0:
        return _ActiveRowSelection(rows=array[:0], copy_bytes=0)
    if int(array.shape[0]) <= int(active_indices[-1]):
        raise ValueError(
            f"{name} has {array.shape[0]} root rows, but active root index "
            f"{int(active_indices[-1])} was requested"
        )
    start = int(active_indices[0])
    stop = int(active_indices[-1]) + 1
    if stop - start == active_count and np.array_equal(
        active_indices,
        np.arange(start, stop, dtype=np.int64),
    ):
        return _ActiveRowSelection(rows=array[start:stop], copy_bytes=0)
    gathered = array[active_indices]
    return _ActiveRowSelection(rows=gathered, copy_bytes=int(gathered.nbytes))


class CompactTorchSearchServiceV1:
    """Profile-only Torch search service behind the compact search contract.

    This owns one model/search pass for a ``CompactRootBatchV1`` and returns a
    validated compact result. It is a candidate backend harness, not a trainer
    default and not a LightZero CTree replacement.
    """

    search_impl = COMPACT_TORCH_SEARCH_SERVICE_IMPL
    supports_two_phase_compact_search = True

    def __init__(
        self,
        *,
        policy: Any,
        num_simulations: int,
        device: Any | None = None,
        compile_config: CompactTorchCompileConfig | None = None,
        root_noise_weight: float | None = None,
        normalize_uint8_observation: bool = True,
        require_resident_observation: bool = False,
    ) -> None:
        simulations = int(num_simulations)
        if simulations <= 0:
            raise ValueError("num_simulations must be positive")
        self._policy = policy
        self._model = _model_from_policy(policy)
        self.num_simulations = simulations
        self._device = device
        self._compile_config = compile_config or CompactTorchCompileConfig()
        self._root_noise_weight = root_noise_weight
        self._normalize_uint8_observation = bool(normalize_uint8_observation)
        self._require_resident_observation = bool(require_resident_observation)
        self._replay_payload_counter = 0
        self._pending_replay_payloads: dict[
            str,
            _PendingCompactTorchReplayPayload,
        ] = {}
        self._compiled_helper_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self._compiled_model_cache: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        self._policy_version_ref = ""
        self._model_version_ref = ""
        self._policy_source = ""
        self._policy_refresh_learner_update_count = 0
        self._policy_refresh_count = 0
        self._policy_refresh_cache_cleared = False
        self._policy_refresh_model_state_digest = ""
        self._policy_refresh_metadata_cache: dict[str, Any] = {}
        self._policy_refresh_last_total_sec = 0.0
        self._policy_refresh_last_state_load_sec = 0.0
        self._policy_refresh_last_digest_sec = 0.0
        self._policy_refresh_total_state_load_sec = 0.0
        self._policy_refresh_total_digest_sec = 0.0
        self._policy_refresh_digest_source = ""
        self._model_memory_format_active = COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS
        mode = str(self._compile_config.recurrent_action_shape_mode)
        if mode not in {"auto", "flat", "column"}:
            raise ValueError(
                f"recurrent_action_shape_mode must be auto, flat, or column; got {mode!r}"
            )
        timing_mode = str(self._compile_config.timing_mode)
        if timing_mode not in COMPACT_TORCH_TIMING_MODES:
            allowed = ", ".join(COMPACT_TORCH_TIMING_MODES)
            raise ValueError(f"timing_mode must be one of {allowed}; got {timing_mode!r}")
        model_compile_mode = str(self._compile_config.model_compile_mode)
        if model_compile_mode not in COMPACT_TORCH_MODEL_COMPILE_MODES:
            allowed = ", ".join(COMPACT_TORCH_MODEL_COMPILE_MODES)
            raise ValueError(
                f"model_compile_mode must be one of {allowed}; got {model_compile_mode!r}"
            )
        initial_inference_mode = str(self._compile_config.initial_inference_mode)
        if initial_inference_mode not in COMPACT_TORCH_INITIAL_INFERENCE_MODES:
            allowed = ", ".join(COMPACT_TORCH_INITIAL_INFERENCE_MODES)
            raise ValueError(
                f"initial_inference_mode must be one of {allowed}; got {initial_inference_mode!r}"
            )
        observation_memory_format = str(self._compile_config.observation_memory_format)
        if observation_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
            allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
            raise ValueError(
                "observation_memory_format must be one of "
                f"{allowed}; got {observation_memory_format!r}"
            )
        model_memory_format = str(self._compile_config.model_memory_format)
        if model_memory_format not in COMPACT_TORCH_MEMORY_FORMATS:
            allowed = ", ".join(COMPACT_TORCH_MEMORY_FORMATS)
            raise ValueError(
                f"model_memory_format must be one of {allowed}; got {model_memory_format!r}"
            )
        if model_memory_format != COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS:
            raise ValueError(
                "model_memory_format=channels_last is parked for the current "
                "LightZero model because the dynamics reward head uses .view(); "
                "use model_memory_format='contiguous'"
            )

    def _pending_deferred_one_simulation_replay_count(self) -> int:
        return sum(
            1
            for pending in self._pending_replay_payloads.values()
            if bool(pending.deferred_one_simulation_replay)
        )

    def _raise_if_pending_deferred_one_simulation_replay(self, *, phase: str) -> None:
        count = self._pending_deferred_one_simulation_replay_count()
        if count > 0:
            raise ReplayCompatibilityError(
                f"cannot {phase} with {count} pending deferred one-simulation "
                "replay payload(s); flush replay before model refresh"
            )

    def _deferred_one_simulation_model_identity(self) -> dict[str, Any]:
        state = self.policy_refresh_search_worker_state()
        return {
            "policy_version_ref": str(state.get("policy_version_ref") or ""),
            "model_version_ref": str(state.get("model_version_ref") or ""),
            "policy_source": str(state.get("policy_source") or ""),
            "learner_update_count": int(state.get("learner_update_count") or 0),
            "model_state_digest": str(state.get("model_state_digest") or ""),
            "policy_refresh_count": int(state.get("refresh_count") or 0),
        }

    def refresh_model_state(
        self,
        *,
        model_state_dict: Mapping[str, Any],
        policy_version_ref: str,
        model_version_ref: str,
        policy_source: str,
        learner_update_count: int,
        expected_model_state_digest: str | None = None,
    ) -> dict[str, Any]:
        """Refresh this search worker to learner-produced model weights."""

        self._raise_if_pending_deferred_one_simulation_replay(phase="refresh model state")
        refresh_started = time.perf_counter()
        policy_ref = str(policy_version_ref).strip()
        model_ref = str(model_version_ref).strip()
        source = str(policy_source).strip()
        update_count = int(learner_update_count)
        if not policy_ref:
            raise ValueError("policy_version_ref must be non-empty")
        if not model_ref:
            raise ValueError("model_version_ref must be non-empty")
        if not source:
            raise ValueError("policy_source must be non-empty")
        if update_count <= 0:
            raise ValueError("learner_update_count must be positive")
        load_state = getattr(self._model, "load_state_dict", None)
        if not callable(load_state):
            raise ValueError("compact Torch search model cannot load state_dict")
        load_started = time.perf_counter()
        load_state(dict(model_state_dict))
        state_load_sec = _elapsed(load_started)
        digest_started = time.perf_counter()
        model_digest = compact_model_state_digest_v1(self._model)
        digest_sec = _elapsed(digest_started)
        if (
            expected_model_state_digest is not None
            and str(expected_model_state_digest).strip()
            and str(expected_model_state_digest).strip() != model_digest
        ):
            raise ValueError("refreshed search worker model digest mismatch")
        self._policy_version_ref = policy_ref
        self._model_version_ref = model_ref
        self._policy_source = source
        self._policy_refresh_learner_update_count = update_count
        self._policy_refresh_count += 1
        self._compiled_helper_cache.clear()
        self._retain_current_model_compile_cache()
        self._model_memory_format_active = COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS
        self._policy_refresh_cache_cleared = True
        self._policy_refresh_model_state_digest = model_digest
        self._policy_refresh_last_total_sec = _elapsed(refresh_started)
        self._policy_refresh_last_state_load_sec = state_load_sec
        self._policy_refresh_last_digest_sec = digest_sec
        self._policy_refresh_total_state_load_sec += state_load_sec
        self._policy_refresh_total_digest_sec += digest_sec
        self._policy_refresh_digest_source = "search_worker_after_load"
        self._policy_refresh_metadata_cache = {}
        state = self.policy_refresh_search_worker_state()
        self._policy_refresh_metadata_cache = compact_policy_refresh_metadata_from_state_v1(state)
        return state

    def refresh_shared_model_state(
        self,
        *,
        policy_version_ref: str,
        model_version_ref: str,
        policy_source: str,
        learner_update_count: int,
        model_state_digest: str,
    ) -> dict[str, Any]:
        """Mark an already-shared model object as refreshed by the learner."""

        self._raise_if_pending_deferred_one_simulation_replay(
            phase="refresh shared model state"
        )
        refresh_started = time.perf_counter()
        policy_ref = str(policy_version_ref).strip()
        model_ref = str(model_version_ref).strip()
        source = str(policy_source).strip()
        update_count = int(learner_update_count)
        digest = str(model_state_digest).strip()
        if not policy_ref:
            raise ValueError("policy_version_ref must be non-empty")
        if not model_ref:
            raise ValueError("model_version_ref must be non-empty")
        if not source:
            raise ValueError("policy_source must be non-empty")
        if update_count <= 0:
            raise ValueError("learner_update_count must be positive")
        if not digest:
            raise ValueError("model_state_digest must be non-empty")
        self._policy_version_ref = policy_ref
        self._model_version_ref = model_ref
        self._policy_source = source
        self._policy_refresh_learner_update_count = update_count
        self._policy_refresh_count += 1
        self._compiled_helper_cache.clear()
        self._retain_current_model_compile_cache()
        self._model_memory_format_active = COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS
        self._policy_refresh_cache_cleared = True
        self._policy_refresh_model_state_digest = digest
        self._policy_refresh_last_total_sec = _elapsed(refresh_started)
        self._policy_refresh_last_state_load_sec = 0.0
        self._policy_refresh_last_digest_sec = 0.0
        self._policy_refresh_digest_source = "shared_model_version_token"
        self._policy_refresh_metadata_cache = {}
        state = self.policy_refresh_search_worker_state()
        self._policy_refresh_metadata_cache = compact_policy_refresh_metadata_from_state_v1(state)
        state = dict(state)
        state["shared_model_state_refreshed"] = True
        return state

    def policy_refresh_search_worker_state(self) -> dict[str, Any]:
        """Return validated search-worker refresh state for checkpoint evidence."""

        model_digest = self._policy_refresh_model_state_digest
        if not model_digest:
            model_digest = compact_model_state_digest_v1(self._model)
        return {
            "schema_id": COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID,
            "search_impl": self.search_impl,
            "policy_version_ref": self._policy_version_ref,
            "model_version_ref": self._model_version_ref,
            "policy_source": self._policy_source,
            "learner_update_count": int(self._policy_refresh_learner_update_count),
            "model_state_digest": model_digest,
            "search_worker_model_object_id": int(id(self._model)),
            "search_worker_object_id": int(id(self)),
            "refresh_count": int(self._policy_refresh_count),
            "refresh_applied": bool(self._policy_refresh_count > 0),
            "cache_cleared": bool(self._policy_refresh_cache_cleared),
            "compiled_helper_cache_size": int(len(self._compiled_helper_cache)),
            "compiled_model_cache_size": int(len(self._compiled_model_cache)),
            "refresh_total_sec": float(self._policy_refresh_last_total_sec),
            "state_load_sec": float(self._policy_refresh_last_state_load_sec),
            "model_state_digest_sec": float(self._policy_refresh_last_digest_sec),
            "total_state_load_sec": float(self._policy_refresh_total_state_load_sec),
            "total_model_state_digest_sec": float(self._policy_refresh_total_digest_sec),
            "model_state_digest_source": str(self._policy_refresh_digest_source),
            "calls_train_muzero": False,
            "touches_live_runs": False,
        }

    def _policy_refresh_metadata(self) -> dict[str, Any]:
        if self._policy_refresh_count <= 0:
            return {}
        if not self._policy_refresh_metadata_cache:
            self._policy_refresh_metadata_cache = compact_policy_refresh_metadata_from_state_v1(
                self.policy_refresh_search_worker_state()
            )
        return dict(self._policy_refresh_metadata_cache)

    def _retain_current_model_compile_cache(self) -> None:
        model_id = id(self._model)
        self._compiled_model_cache = {
            key: value
            for key, value in self._compiled_model_cache.items()
            if isinstance(key, tuple) and key and key[0] == model_id
        }

    def _initial_inference_runtime(
        self,
        initial_inference_fn: Any,
        *,
        direct_core_representation_fn: Any | None = None,
        direct_core_prediction_fn: Any | None = None,
    ) -> _CompactTorchInitialInferenceRuntime:
        return _CompactTorchInitialInferenceRuntime(
            model=self._model,
            initial_inference_fn=initial_inference_fn,
            requested_mode=str(self._compile_config.initial_inference_mode),
            direct_core_representation_fn=direct_core_representation_fn,
            direct_core_prediction_fn=direct_core_prediction_fn,
        )

    def run(self, root_batch: CompactRootBatchV1) -> Any:
        """Run compact Torch search once and return ``CompactSearchResultV1``."""

        import torch

        masks = compact_torch_fixed_shape_masks(root_batch)
        active_root_indices = np.flatnonzero(masks.active_root_mask).astype(np.int32)
        active_count = int(active_root_indices.size)
        device = self._resolve_device(torch=torch)
        if active_count == 0:
            _, _, resident_telemetry = self._root_observation_tensor(
                torch=torch,
                root_batch=root_batch,
                active_root_indices=active_root_indices,
                device=device,
            )
            return _compact_torch_validated_result(
                root_batch,
                selected_action=np.zeros((0,), dtype=np.int16),
                visit_policy=np.zeros((0, ACTION_COUNT), dtype=np.float32),
                root_value=np.zeros((0,), dtype=np.float32),
                metadata={
                    **resident_telemetry,
                    **self._policy_refresh_metadata(),
                    "profile_semantics": COMPACT_TORCH_SEARCH_SEMANTICS,
                    "compact_torch_search_semantics": COMPACT_TORCH_SEARCH_SEMANTICS,
                    "compact_torch_search_service_profile_only": True,
                    "compact_torch_search_service_zero_active_roots": True,
                    "compact_torch_search_service_resident_obs_reused": _resident_obs_reused_flag(
                        resident_telemetry
                    ),
                },
                num_simulations=self.num_simulations,
            )

        root_noise_weight = self._resolved_root_noise_weight()
        eligibility = compact_torch_compile_eligibility(
            root_batch,
            device=device,
            root_noise_weight=root_noise_weight,
            config=self._compile_config,
            torch_module=torch,
            fixed_shape_masks=masks,
        )
        select_helper, backup_helper, compile_runtime = self._compiled_helpers_for_eligibility(
            torch=torch,
            eligibility=eligibility,
        )
        (
            initial_inference_fn,
            recurrent_inference_fn,
            direct_core_representation_fn,
            direct_core_prediction_fn,
            model_compile_runtime,
        ) = self._compiled_model_for_runtime(torch=torch, device=device)
        initial_runtime = self._initial_inference_runtime(
            initial_inference_fn,
            direct_core_representation_fn=direct_core_representation_fn,
            direct_core_prediction_fn=direct_core_prediction_fn,
        )

        total_started = time.perf_counter()
        tensor_started = time.perf_counter()
        obs_selection, obs_tensor, resident_telemetry = self._root_observation_tensor(
            torch=torch,
            root_batch=root_batch,
            active_root_indices=active_root_indices,
            device=device,
        )
        legal_selection = _select_active_rows_without_copy_when_contiguous(
            root_batch.legal_mask,
            active_root_indices,
            name="legal_mask",
        )
        legal_mask_np = legal_selection.rows
        mask_h2d_bytes = int(getattr(legal_mask_np, "nbytes", 0))
        timing_mode = str(self._compile_config.timing_mode)
        tensor_sync_sec = _timed_profile_sync_torch_device_if_cuda(
            torch=torch,
            device=device,
            timing_mode=timing_mode,
            phase="tensor_prepare",
        )
        tensor_prepare_sec = time.perf_counter() - tensor_started

        with _CompactTorchModelInferenceGuard(
            torch=torch,
            model=self._model,
        ) as inference_guard:
            initial_started = time.perf_counter()
            initial_cuda_event = _start_cuda_event_timing(
                torch=torch,
                device=device,
                timing_mode=timing_mode,
            )
            root_output = initial_runtime.run(
                obs_tensor,
                torch=torch,
                device=device,
                timing_mode=timing_mode,
            )
            _finish_cuda_event_timing(initial_cuda_event)
            initial_inference_enqueue_sec = time.perf_counter() - initial_started
            initial_inference_sync_sec = _timed_profile_sync_torch_device_if_cuda(
                torch=torch,
                device=device,
                timing_mode=timing_mode,
                phase="initial_inference",
            )
            initial_inference_sec = time.perf_counter() - initial_started

            root_output_decode_started = time.perf_counter()
            (
                policy_logits,
                root_value,
                root_latent_state,
                root_output_direct_decoded,
            ) = _compact_torch_initial_output_fields(root_output)
            root_output_decode_sec = _elapsed(root_output_decode_started)
            del root_output

            root_latent_prepare_started = time.perf_counter()
            root_latent_state, root_latent_telemetry = _compact_torch_root_latent_for_recurrent(
                torch=torch,
                root_latent_state=root_latent_state,
            )
            root_latent_prepare_sec = _elapsed(root_latent_prepare_started)
            model_compile_runtime.update(root_latent_telemetry)
            model_compile_runtime["compact_torch_search_root_latent_prepare_sec"] = float(
                root_latent_prepare_sec
            )

            search_started = time.perf_counter()
            selected, root_values, visit_policy, raw_counts = _run_compact_torch_tree_search(
                torch=torch,
                policy=self._policy,
                model=self._model,
                root_policy_logits=policy_logits,
                root_value=root_value,
                root_latent_state=root_latent_state,
                legal_mask=legal_mask_np,
                num_simulations=self.num_simulations,
                root_noise_weight=root_noise_weight,
                device=device,
                select_helper=select_helper,
                backup_helper=backup_helper,
                recurrent_inference_fn=recurrent_inference_fn,
                recurrent_action_shape_mode=(self._compile_config.recurrent_action_shape_mode),
                timing_mode=timing_mode,
                runtime_metadata=model_compile_runtime,
            )
            tree_search_sec = time.perf_counter() - search_started
            initial_inference_cuda_event_sec, initial_inference_cuda_event_status = (
                _cuda_event_elapsed_sec(initial_cuda_event)
            )
            initial_runtime_telemetry = initial_runtime.telemetry()
            initial_split_residual_sec, initial_split_residual_status = (
                _initial_inference_split_cuda_event_residual(
                    outer_sec=initial_inference_cuda_event_sec,
                    outer_status=initial_inference_cuda_event_status,
                    runtime_telemetry=initial_runtime_telemetry,
                )
            )
        inference_guard_telemetry = inference_guard.telemetry()

        readback_started = time.perf_counter()
        selected_for_readback = _selected_actions_for_readback(
            selected,
            torch_module=torch,
        )
        selected_np = _array_to_numpy(selected_for_readback).astype(np.int16, copy=False)
        visit_policy_np = _array_to_numpy(visit_policy).astype(np.float32, copy=False)
        root_values_np = _array_to_numpy(root_values).astype(np.float32, copy=False)
        raw_counts_np = _array_to_numpy(raw_counts).astype(np.float32, copy=False)
        readback_sec = time.perf_counter() - readback_started
        total_sec = time.perf_counter() - total_started
        action_d2h_bytes = int(selected_np.nbytes)
        replay_payload_d2h_bytes = int(
            visit_policy_np.nbytes + root_values_np.nbytes + raw_counts_np.nbytes
        )

        metadata = {
            **eligibility.telemetry,
            **compile_runtime,
            **model_compile_runtime,
            **initial_runtime_telemetry,
            **inference_guard_telemetry,
            **resident_telemetry,
            **self._policy_refresh_metadata(),
            "profile_semantics": COMPACT_TORCH_SEARCH_SEMANTICS,
            "compact_torch_search_semantics": COMPACT_TORCH_SEARCH_SEMANTICS,
            "compact_torch_search_service_profile_only": True,
            "compact_torch_search_service_not_lightzero_ctree": True,
            "compact_torch_search_service_trainer_ready": False,
            "compact_torch_search_service_active_root_count": float(active_count),
            "compact_torch_search_service_model_class": type(self._model).__name__,
            "compact_torch_search_service_device": str(device),
            "compact_torch_search_service_compile_status": eligibility.status,
            "compact_torch_search_service_compile_reason": eligibility.reason,
            "compact_torch_search_service_timing_mode": timing_mode,
            "compact_torch_search_service_cuda_event_timing_enabled": _cuda_event_timing_enabled(
                timing_mode
            ),
            "compact_torch_search_service_initial_sync_enabled": _phase_sync_enabled(
                timing_mode,
                "initial_inference",
            ),
            "compact_torch_search_service_tensor_prepare_sec": float(tensor_prepare_sec),
            "compact_torch_search_service_tensor_prepare_sync_sec": float(tensor_sync_sec),
            "compact_torch_search_service_initial_inference_sec": float(initial_inference_sec),
            "compact_torch_search_service_initial_inference_enqueue_sec": float(
                initial_inference_enqueue_sec
            ),
            "compact_torch_search_service_initial_inference_sync_sec": float(
                initial_inference_sync_sec
            ),
            "compact_torch_search_service_initial_inference_cuda_event_sec": float(
                initial_inference_cuda_event_sec
            ),
            "compact_torch_search_service_initial_inference_cuda_event_status": str(
                initial_inference_cuda_event_status
            ),
            "compact_torch_search_service_initial_inference_direct_core_cuda_event_residual_sec": float(
                initial_split_residual_sec
            ),
            "compact_torch_search_service_initial_inference_direct_core_cuda_event_residual_status": str(
                initial_split_residual_status
            ),
            "compact_torch_search_service_initial_output_decode_sec": float(root_output_decode_sec),
            "compact_torch_search_service_root_output_decode_sec": float(root_output_decode_sec),
            "compact_torch_search_service_root_latent_prepare_sec": float(root_latent_prepare_sec),
            "compact_torch_search_service_root_output_direct_decoded": bool(
                root_output_direct_decoded
            ),
            "compact_torch_search_service_tree_search_sec": float(tree_search_sec),
            "compact_torch_search_service_readback_sec": float(readback_sec),
            "compact_torch_search_service_total_sec": float(total_sec),
            "compact_torch_search_service_obs_h2d_bytes": float(
                resident_telemetry["resident_observation_h2d_bytes"]
            ),
            "compact_torch_search_service_mask_h2d_bytes": float(mask_h2d_bytes),
            "compact_torch_search_service_action_d2h_bytes": float(action_d2h_bytes),
            "compact_torch_search_service_replay_payload_d2h_bytes": float(
                replay_payload_d2h_bytes
            ),
            "compact_torch_search_service_root_observation_copy_bytes": float(
                obs_selection.copy_bytes
            ),
            "compact_torch_search_service_root_mask_copy_bytes": float(legal_selection.copy_bytes),
            "compact_torch_search_service_python_rows_materialized": 0.0,
            "compact_torch_search_service_rnd_materialized_rows": 0.0,
            "compact_torch_search_service_resident_obs_reused": _resident_obs_reused_flag(
                resident_telemetry
            ),
        }
        metadata["profile_telemetry"] = {
            key: value for key, value in metadata.items() if key != "profile_telemetry"
        }
        return _compact_torch_validated_result(
            root_batch,
            selected_action=selected_np,
            visit_policy=visit_policy_np,
            root_value=root_values_np,
            raw_visit_counts=raw_counts_np,
            metadata=metadata,
            num_simulations=self.num_simulations,
        )

    def run_action_step(
        self,
        root_batch: CompactRootBatchV1,
    ) -> CompactSearchActionStepV1:
        """Return action-critical output while keeping replay tensors deferred."""

        action_service_started = time.perf_counter()
        import torch

        action_preamble_started = action_service_started
        mask_started = time.perf_counter()
        masks = compact_torch_fixed_shape_masks(root_batch)
        active_root_indices = np.flatnonzero(masks.active_root_mask).astype(np.int32)
        active_count = int(active_root_indices.size)
        masks_sec = time.perf_counter() - mask_started
        handle = f"{self.search_impl}:{self._replay_payload_counter}"
        self._replay_payload_counter += 1
        device = self._resolve_device(torch=torch)
        action_preamble_sec = time.perf_counter() - action_preamble_started

        if active_count == 0:
            _, _, resident_telemetry = self._root_observation_tensor(
                torch=torch,
                root_batch=root_batch,
                active_root_indices=active_root_indices,
                device=device,
            )
            selected_np = np.zeros((0,), dtype=np.int16)
            empty_policy = np.zeros((0, ACTION_COUNT), dtype=np.float32)
            empty_value = np.zeros((0,), dtype=np.float32)
            metadata_build_started = time.perf_counter()
            metadata = {
                **resident_telemetry,
                **self._policy_refresh_metadata(),
                "profile_semantics": COMPACT_TORCH_SEARCH_SEMANTICS,
                "compact_torch_search_semantics": COMPACT_TORCH_SEARCH_SEMANTICS,
                "compact_torch_search_service_profile_only": True,
                "compact_torch_search_service_zero_active_roots": True,
                "compact_torch_search_service_two_phase_action_only": True,
                "compact_torch_search_service_active_root_count": 0.0,
                "compact_torch_search_service_timing_mode": str(self._compile_config.timing_mode),
                "compact_torch_search_service_cuda_event_timing_enabled": _cuda_event_timing_enabled(
                    str(self._compile_config.timing_mode)
                ),
                "compact_torch_search_service_initial_sync_enabled": _phase_sync_enabled(
                    str(self._compile_config.timing_mode),
                    "initial_inference",
                ),
                "compact_torch_search_service_action_preamble_sec": float(action_preamble_sec),
                "compact_torch_search_service_fixed_shape_masks_sec": float(masks_sec),
                "compact_torch_search_service_compile_eligibility_sec": 0.0,
                "compact_torch_search_service_helper_cache_sec": 0.0,
                "compact_torch_search_service_model_cache_sec": 0.0,
                "compact_torch_search_service_action_d2h_bytes": 0.0,
                "compact_torch_search_service_replay_payload_d2h_bytes": 0.0,
                "compact_torch_search_service_deferred_replay_payload_d2h_bytes": 0.0,
                "compact_torch_search_service_resident_obs_reused": _resident_obs_reused_flag(
                    resident_telemetry
                ),
            }
            metadata_build_sec = _elapsed(metadata_build_started)
            metadata["compact_torch_search_service_metadata_build_sec"] = float(metadata_build_sec)
            metadata["profile_telemetry"] = dict(metadata)
            pending_store_started = time.perf_counter()
            self._pending_replay_payloads[handle] = _PendingCompactTorchReplayPayload(
                root_index=active_root_indices.astype(np.int32, copy=True),
                env_row=root_batch.env_row[active_root_indices].astype(np.int32, copy=True),
                player=root_batch.player[active_root_indices].astype(np.int16, copy=True),
                policy_env_id=root_batch.policy_env_id[active_root_indices].astype(
                    np.int64,
                    copy=True,
                ),
                visit_policy=empty_policy,
                root_value=empty_value,
                raw_visit_counts=empty_policy.copy(),
                metadata=metadata,
            )
            pending_store_sec = _elapsed(pending_store_started)
            _set_profile_metric(
                metadata,
                "compact_torch_search_service_pending_replay_store_sec",
                pending_store_sec,
            )
            postprocess_base_sec = metadata_build_sec + pending_store_sec
            action_step_build_started = time.perf_counter()
            action_step = self._action_step_from_selected(
                handle,
                root_batch,
                active_root_indices,
                selected_np,
                metadata,
            )
            action_step_build_sec = _elapsed(action_step_build_started)
            return _finalize_action_step_profile_timing(
                metadata,
                action_step,
                action_service_started=action_service_started,
                accounted_sec=action_preamble_sec + postprocess_base_sec + action_step_build_sec,
                postprocess_base_sec=postprocess_base_sec,
                action_step_build_sec=action_step_build_sec,
            )

        root_noise_weight = self._resolved_root_noise_weight()
        eligibility_started = time.perf_counter()
        eligibility = compact_torch_compile_eligibility(
            root_batch,
            device=device,
            root_noise_weight=root_noise_weight,
            config=self._compile_config,
            torch_module=torch,
            fixed_shape_masks=masks,
        )
        eligibility_sec = time.perf_counter() - eligibility_started
        helper_cache_started = time.perf_counter()
        select_helper, backup_helper, compile_runtime = self._compiled_helpers_for_eligibility(
            torch=torch,
            eligibility=eligibility,
        )
        helper_cache_sec = time.perf_counter() - helper_cache_started
        model_cache_started = time.perf_counter()
        (
            initial_inference_fn,
            recurrent_inference_fn,
            direct_core_representation_fn,
            direct_core_prediction_fn,
            model_compile_runtime,
        ) = self._compiled_model_for_runtime(torch=torch, device=device)
        model_cache_sec = time.perf_counter() - model_cache_started
        initial_runtime = self._initial_inference_runtime(
            initial_inference_fn,
            direct_core_representation_fn=direct_core_representation_fn,
            direct_core_prediction_fn=direct_core_prediction_fn,
        )
        action_preamble_sec = time.perf_counter() - action_preamble_started

        total_started = time.perf_counter()
        tensor_started = time.perf_counter()
        obs_selection, obs_tensor, resident_telemetry = self._root_observation_tensor(
            torch=torch,
            root_batch=root_batch,
            active_root_indices=active_root_indices,
            device=device,
        )
        legal_selection = _select_active_rows_without_copy_when_contiguous(
            root_batch.legal_mask,
            active_root_indices,
            name="legal_mask",
        )
        legal_mask_np = legal_selection.rows
        mask_h2d_bytes = int(getattr(legal_mask_np, "nbytes", 0))
        timing_mode = str(self._compile_config.timing_mode)
        tensor_sync_sec = _timed_profile_sync_torch_device_if_cuda(
            torch=torch,
            device=device,
            timing_mode=timing_mode,
            phase="tensor_prepare",
        )
        tensor_prepare_sec = time.perf_counter() - tensor_started

        with _CompactTorchModelInferenceGuard(
            torch=torch,
            model=self._model,
        ) as inference_guard:
            initial_started = time.perf_counter()
            initial_cuda_event = _start_cuda_event_timing(
                torch=torch,
                device=device,
                timing_mode=timing_mode,
            )
            root_output = initial_runtime.run(
                obs_tensor,
                torch=torch,
                device=device,
                timing_mode=timing_mode,
            )
            _finish_cuda_event_timing(initial_cuda_event)
            initial_inference_enqueue_sec = time.perf_counter() - initial_started
            initial_inference_sync_sec = _timed_profile_sync_torch_device_if_cuda(
                torch=torch,
                device=device,
                timing_mode=timing_mode,
                phase="initial_inference",
            )
            initial_inference_sec = time.perf_counter() - initial_started

            initial_output_decode_started = time.perf_counter()
            (
                policy_logits,
                root_value,
                root_latent_state,
                root_output_direct_decoded,
            ) = _compact_torch_initial_output_fields(root_output)
            initial_output_decode_sec = _elapsed(initial_output_decode_started)
            del root_output

            root_latent_prepare_started = time.perf_counter()
            root_latent_state, root_latent_telemetry = _compact_torch_root_latent_for_recurrent(
                torch=torch,
                root_latent_state=root_latent_state,
            )
            root_latent_prepare_sec = _elapsed(root_latent_prepare_started)
            model_compile_runtime.update(root_latent_telemetry)
            model_compile_runtime["compact_torch_search_root_latent_prepare_sec"] = float(
                root_latent_prepare_sec
            )

            search_started = time.perf_counter()
            defer_one_simulation_replay = (
                bool(self._compile_config.defer_one_simulation_replay_payload)
                and int(self.num_simulations) == 1
                and float(root_noise_weight) <= 0.0
            )
            if defer_one_simulation_replay:
                selected = _select_compact_torch_zero_noise_one_simulation_action(
                    torch=torch,
                    root_policy_logits=policy_logits,
                    legal_mask=legal_mask_np,
                    device=device,
                    timing_mode=timing_mode,
                    runtime_metadata=model_compile_runtime,
                )
                root_values = None
                visit_policy = None
                raw_counts = None
            else:
                selected, root_values, visit_policy, raw_counts = _run_compact_torch_tree_search(
                    torch=torch,
                    policy=self._policy,
                    model=self._model,
                    root_policy_logits=policy_logits,
                    root_value=root_value,
                    root_latent_state=root_latent_state,
                    legal_mask=legal_mask_np,
                    num_simulations=self.num_simulations,
                    root_noise_weight=root_noise_weight,
                    device=device,
                    select_helper=select_helper,
                    backup_helper=backup_helper,
                    recurrent_inference_fn=recurrent_inference_fn,
                    recurrent_action_shape_mode=(
                        self._compile_config.recurrent_action_shape_mode
                    ),
                    timing_mode=timing_mode,
                    runtime_metadata=model_compile_runtime,
                )
            tree_search_sec = time.perf_counter() - search_started
            initial_inference_cuda_event_sec, initial_inference_cuda_event_status = (
                _cuda_event_elapsed_sec(initial_cuda_event)
            )
            initial_runtime_telemetry = initial_runtime.telemetry()
            initial_split_residual_sec, initial_split_residual_status = (
                _initial_inference_split_cuda_event_residual(
                    outer_sec=initial_inference_cuda_event_sec,
                    outer_status=initial_inference_cuda_event_status,
                    runtime_telemetry=initial_runtime_telemetry,
                )
            )
        inference_guard_telemetry = inference_guard.telemetry()

        readback_started = time.perf_counter()
        selected_for_readback = _selected_actions_for_readback(
            selected,
            torch_module=torch,
        )
        selected_np = _array_to_numpy(selected_for_readback).astype(np.int16, copy=False)
        action_readback_sec = time.perf_counter() - readback_started
        total_sec = time.perf_counter() - total_started
        action_d2h_bytes = int(selected_np.nbytes)
        deferred_replay_payload_d2h_bytes = (
            0
            if defer_one_simulation_replay
            else int(
                _tensor_nbytes(visit_policy)
                + _tensor_nbytes(root_values)
                + _tensor_nbytes(raw_counts)
            )
        )
        core_accounted_sec = (
            tensor_prepare_sec
            + initial_inference_sec
            + initial_output_decode_sec
            + root_latent_prepare_sec
            + tree_search_sec
            + action_readback_sec
        )
        core_residual_sec = total_sec - core_accounted_sec

        metadata_build_started = time.perf_counter()
        deferred_action_identity = (
            self._deferred_one_simulation_model_identity()
            if defer_one_simulation_replay
            else {}
        )
        metadata = {
            **eligibility.telemetry,
            **compile_runtime,
            **model_compile_runtime,
            **initial_runtime_telemetry,
            **inference_guard_telemetry,
            **resident_telemetry,
            **self._policy_refresh_metadata(),
            "profile_semantics": COMPACT_TORCH_SEARCH_SEMANTICS,
            "compact_torch_search_semantics": COMPACT_TORCH_SEARCH_SEMANTICS,
            "compact_torch_search_service_profile_only": True,
            "compact_torch_search_service_not_lightzero_ctree": True,
            "compact_torch_search_service_trainer_ready": False,
            "compact_torch_search_service_two_phase_action_only": True,
            "compact_torch_search_service_defer_one_simulation_replay_payload_requested": bool(
                self._compile_config.defer_one_simulation_replay_payload
            ),
            "compact_torch_search_service_defer_one_simulation_replay_payload_used": bool(
                defer_one_simulation_replay
            ),
            "compact_torch_search_service_deferred_one_simulation_replay_flush_pending": bool(
                defer_one_simulation_replay
            ),
            "compact_torch_search_deferred_one_simulation_action_model_state_digest": str(
                deferred_action_identity.get("model_state_digest") or ""
            ),
            "compact_torch_search_deferred_one_simulation_action_policy_refresh_count": int(
                deferred_action_identity.get("policy_refresh_count") or 0
            ),
            "compact_torch_search_deferred_one_simulation_action_policy_version_ref": str(
                deferred_action_identity.get("policy_version_ref") or ""
            ),
            "compact_torch_search_deferred_one_simulation_action_model_version_ref": str(
                deferred_action_identity.get("model_version_ref") or ""
            ),
            "compact_torch_search_deferred_one_simulation_action_policy_source": str(
                deferred_action_identity.get("policy_source") or ""
            ),
            "compact_torch_search_deferred_one_simulation_action_learner_update_count": int(
                deferred_action_identity.get("learner_update_count") or 0
            ),
            "compact_torch_search_deferred_one_simulation_model_identity_match": (
                True if defer_one_simulation_replay else False
            ),
            "compact_torch_search_deferred_one_simulation_model_refresh_crossed_count": 0,
            "compact_torch_search_service_active_root_count": float(active_count),
            "compact_torch_search_service_model_class": type(self._model).__name__,
            "compact_torch_search_service_device": str(device),
            "compact_torch_search_service_compile_status": eligibility.status,
            "compact_torch_search_service_compile_reason": eligibility.reason,
            "compact_torch_search_service_timing_mode": timing_mode,
            "compact_torch_search_service_cuda_event_timing_enabled": _cuda_event_timing_enabled(
                timing_mode
            ),
            "compact_torch_search_service_initial_sync_enabled": _phase_sync_enabled(
                timing_mode,
                "initial_inference",
            ),
            "compact_torch_search_service_action_preamble_sec": float(action_preamble_sec),
            "compact_torch_search_service_fixed_shape_masks_sec": float(masks_sec),
            "compact_torch_search_service_compile_eligibility_sec": float(eligibility_sec),
            "compact_torch_search_service_helper_cache_sec": float(helper_cache_sec),
            "compact_torch_search_service_model_cache_sec": float(model_cache_sec),
            "compact_torch_search_service_tensor_prepare_sec": float(tensor_prepare_sec),
            "compact_torch_search_service_tensor_prepare_sync_sec": float(tensor_sync_sec),
            "compact_torch_search_service_initial_inference_sec": float(initial_inference_sec),
            "compact_torch_search_service_initial_inference_enqueue_sec": float(
                initial_inference_enqueue_sec
            ),
            "compact_torch_search_service_initial_inference_sync_sec": float(
                initial_inference_sync_sec
            ),
            "compact_torch_search_service_initial_inference_cuda_event_sec": float(
                initial_inference_cuda_event_sec
            ),
            "compact_torch_search_service_initial_inference_cuda_event_status": str(
                initial_inference_cuda_event_status
            ),
            "compact_torch_search_service_initial_inference_direct_core_cuda_event_residual_sec": float(
                initial_split_residual_sec
            ),
            "compact_torch_search_service_initial_inference_direct_core_cuda_event_residual_status": str(
                initial_split_residual_status
            ),
            "compact_torch_search_service_initial_output_decode_sec": float(
                initial_output_decode_sec
            ),
            "compact_torch_search_service_root_output_decode_sec": float(initial_output_decode_sec),
            "compact_torch_search_service_root_latent_prepare_sec": float(root_latent_prepare_sec),
            "compact_torch_search_service_root_output_direct_decoded": bool(
                root_output_direct_decoded
            ),
            "compact_torch_search_service_tree_search_sec": float(tree_search_sec),
            "compact_torch_search_service_readback_sec": float(action_readback_sec),
            "compact_torch_search_service_action_readback_sec": float(action_readback_sec),
            "compact_torch_search_service_total_sec": float(total_sec),
            "compact_torch_search_service_core_accounted_sec": float(core_accounted_sec),
            "compact_torch_search_service_core_residual_sec": float(core_residual_sec),
            "compact_torch_search_service_core_unaccounted_sec": float(max(0.0, core_residual_sec)),
            "compact_torch_search_service_core_overaccounted_sec": float(
                max(0.0, -core_residual_sec)
            ),
            "compact_torch_search_service_obs_h2d_bytes": float(
                resident_telemetry["resident_observation_h2d_bytes"]
            ),
            "compact_torch_search_service_mask_h2d_bytes": float(mask_h2d_bytes),
            "compact_torch_search_service_action_d2h_bytes": float(action_d2h_bytes),
            "compact_torch_search_service_replay_payload_d2h_bytes": 0.0,
            "compact_torch_search_service_deferred_replay_payload_d2h_bytes": float(
                deferred_replay_payload_d2h_bytes
            ),
            "compact_torch_search_service_root_observation_copy_bytes": float(
                obs_selection.copy_bytes
            ),
            "compact_torch_search_service_root_mask_copy_bytes": float(legal_selection.copy_bytes),
            "compact_torch_search_service_python_rows_materialized": 0.0,
            "compact_torch_search_service_rnd_materialized_rows": 0.0,
            "compact_torch_search_service_resident_obs_reused": _resident_obs_reused_flag(
                resident_telemetry
            ),
        }
        metadata_build_sec = _elapsed(metadata_build_started)
        metadata["compact_torch_search_service_metadata_build_sec"] = float(metadata_build_sec)
        metadata["profile_telemetry"] = {
            key: value for key, value in metadata.items() if key != "profile_telemetry"
        }

        if handle in self._pending_replay_payloads:
            raise ReplayCompatibilityError("duplicate compact Torch replay payload handle")
        pending_store_started = time.perf_counter()
        self._pending_replay_payloads[handle] = _PendingCompactTorchReplayPayload(
            root_index=active_root_indices.astype(np.int32, copy=True),
            env_row=root_batch.env_row[active_root_indices].astype(np.int32, copy=True),
            player=root_batch.player[active_root_indices].astype(np.int16, copy=True),
            policy_env_id=root_batch.policy_env_id[active_root_indices].astype(
                np.int64,
                copy=True,
            ),
            visit_policy=visit_policy,
            root_value=root_values,
            raw_visit_counts=raw_counts,
            metadata=metadata,
            deferred_one_simulation_replay=bool(defer_one_simulation_replay),
            selected_action_tensor=selected if defer_one_simulation_replay else None,
            root_latent_state=root_latent_state if defer_one_simulation_replay else None,
            recurrent_inference_fn=(
                recurrent_inference_fn if defer_one_simulation_replay else None
            ),
            recurrent_action_shape_mode=str(self._compile_config.recurrent_action_shape_mode),
            timing_mode=timing_mode,
            discount_factor=float(
                getattr(getattr(self._policy, "_cfg", None), "discount_factor", 0.997)
            ),
            device=device if defer_one_simulation_replay else None,
        )
        _set_profile_metric(
            metadata,
            "compact_torch_search_pending_deferred_replay_payload_count",
            self._pending_deferred_one_simulation_replay_count(),
        )
        pending_store_sec = _elapsed(pending_store_started)
        _set_profile_metric(
            metadata,
            "compact_torch_search_service_pending_replay_store_sec",
            pending_store_sec,
        )
        postprocess_base_sec = metadata_build_sec + pending_store_sec
        action_step_build_started = time.perf_counter()
        action_step = self._action_step_from_selected(
            handle,
            root_batch,
            active_root_indices,
            selected_np,
            metadata,
        )
        action_step_build_sec = _elapsed(action_step_build_started)
        return _finalize_action_step_profile_timing(
            metadata,
            action_step,
            action_service_started=action_service_started,
            accounted_sec=action_preamble_sec
            + total_sec
            + postprocess_base_sec
            + action_step_build_sec,
            postprocess_base_sec=postprocess_base_sec,
            action_step_build_sec=action_step_build_sec,
        )

    def _materialize_pending_replay_tensors(
        self,
        pending: _PendingCompactTorchReplayPayload,
        *,
        torch: Any,
        device: Any,
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        if not bool(pending.deferred_one_simulation_replay):
            return (
                pending.visit_policy,
                pending.root_value,
                pending.raw_visit_counts,
                {},
            )
        if pending.selected_action_tensor is None or pending.root_latent_state is None:
            raise ReplayCompatibilityError(
                "deferred one-simulation replay payload is missing action tensors"
            )
        action_identity = {
            "policy_version_ref": str(
                pending.metadata.get(
                    "compact_torch_search_deferred_one_simulation_action_policy_version_ref"
                )
                or ""
            ),
            "model_version_ref": str(
                pending.metadata.get(
                    "compact_torch_search_deferred_one_simulation_action_model_version_ref"
                )
                or ""
            ),
            "policy_source": str(
                pending.metadata.get(
                    "compact_torch_search_deferred_one_simulation_action_policy_source"
                )
                or ""
            ),
            "learner_update_count": int(
                pending.metadata.get(
                    "compact_torch_search_deferred_one_simulation_action_learner_update_count"
                )
                or 0
            ),
            "model_state_digest": str(
                pending.metadata.get(
                    "compact_torch_search_deferred_one_simulation_action_model_state_digest"
                )
                or ""
            ),
            "policy_refresh_count": int(
                pending.metadata.get(
                    "compact_torch_search_deferred_one_simulation_action_policy_refresh_count"
                )
                or 0
            ),
        }
        flush_identity = self._deferred_one_simulation_model_identity()
        identity_match = action_identity == flush_identity
        if not identity_match:
            raise ReplayCompatibilityError(
                "deferred one-simulation replay payload crossed a model refresh"
            )
        with _CompactTorchModelInferenceGuard(torch=torch, model=self._model):
            root_values, visit_policy, raw_counts, telemetry = (
                _materialize_compact_torch_one_simulation_replay_from_selected(
                    torch=torch,
                    policy=self._policy,
                    model=self._model,
                    selected=pending.selected_action_tensor,
                    root_latent_state=pending.root_latent_state,
                    discount_factor=float(pending.discount_factor),
                    device=device,
                    recurrent_inference_fn=pending.recurrent_inference_fn,
                    recurrent_action_shape_mode=str(pending.recurrent_action_shape_mode),
                    timing_mode=str(pending.timing_mode),
                )
        )
        telemetry.update(
            {
                "compact_torch_search_deferred_one_simulation_flush_model_state_digest": str(
                    flush_identity.get("model_state_digest") or ""
                ),
                "compact_torch_search_deferred_one_simulation_flush_policy_refresh_count": int(
                    flush_identity.get("policy_refresh_count") or 0
                ),
                "compact_torch_search_deferred_one_simulation_flush_policy_version_ref": str(
                    flush_identity.get("policy_version_ref") or ""
                ),
                "compact_torch_search_deferred_one_simulation_flush_model_version_ref": str(
                    flush_identity.get("model_version_ref") or ""
                ),
                "compact_torch_search_deferred_one_simulation_flush_policy_source": str(
                    flush_identity.get("policy_source") or ""
                ),
                "compact_torch_search_deferred_one_simulation_flush_learner_update_count": int(
                    flush_identity.get("learner_update_count") or 0
                ),
                "compact_torch_search_deferred_one_simulation_model_identity_match": True,
                "compact_torch_search_deferred_one_simulation_model_refresh_crossed_count": 0,
            }
        )
        return visit_policy, root_values, raw_counts, telemetry

    def flush_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactSearchReplayPayloadV1:
        """Materialize and return the delayed replay-critical search payload."""

        handle = str(replay_payload_handle)
        if not handle:
            raise ReplayCompatibilityError("replay_payload_handle must be non-empty")
        pending = self._pending_replay_payloads.get(handle)
        if pending is None:
            raise ReplayCompatibilityError("unknown or already-flushed replay payload handle")
        pending_deferred_before = self._pending_deferred_one_simulation_replay_count()
        pending_deferred_after = max(
            0,
            pending_deferred_before - (1 if bool(pending.deferred_one_simulation_replay) else 0),
        )

        import torch

        device = self._resolve_device(torch=torch)
        visit_policy_value, root_value_value, raw_counts_value, deferred_telemetry = (
            self._materialize_pending_replay_tensors(
                pending,
                torch=torch,
                device=device,
            )
        )
        readback_started = time.perf_counter()
        visit_policy_np = _array_to_numpy(visit_policy_value).astype(
            np.float32,
            copy=False,
        )
        root_value_np = _array_to_numpy(root_value_value).astype(np.float32, copy=False)
        raw_counts_np = _array_to_numpy(raw_counts_value).astype(
            np.float32,
            copy=False,
        )
        readback_sec = time.perf_counter() - readback_started
        replay_payload_d2h_bytes = int(
            visit_policy_np.nbytes + root_value_np.nbytes + raw_counts_np.nbytes
        )

        metadata = dict(pending.metadata)
        profile_telemetry = metadata.get("profile_telemetry")
        profile = dict(profile_telemetry) if isinstance(profile_telemetry, Mapping) else {}
        profile.update(deferred_telemetry)
        profile.update(
            {
                "compact_torch_search_pending_deferred_replay_payload_count": float(
                    pending_deferred_before
                ),
                "compact_torch_search_pending_deferred_replay_payload_final_count": float(
                    pending_deferred_after
                ),
                "compact_torch_search_service_replay_payload_readback_sec": float(readback_sec),
                "compact_torch_search_service_replay_payload_d2h_bytes": float(
                    replay_payload_d2h_bytes
                ),
                "compact_torch_search_service_deferred_replay_payload_d2h_bytes": 0.0,
                "compact_torch_search_service_replay_payload_flushed": True,
            }
        )
        metadata.update(profile)
        metadata.update(
            {
                "schema_id": COMPACT_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                "phase": "replay_critical",
                "search_impl": self.search_impl,
                "num_simulations": int(self.num_simulations),
                "active_root_count": int(pending.root_index.size),
                "replay_payload_origin": f"{self.search_impl}:{handle}",
                "profile_telemetry": profile,
            }
        )
        payload = CompactSearchReplayPayloadV1(
            replay_payload_handle=handle,
            root_index=pending.root_index.astype(np.int32, copy=True),
            env_row=pending.env_row.astype(np.int32, copy=True),
            player=pending.player.astype(np.int16, copy=True),
            policy_env_id=pending.policy_env_id.astype(np.int64, copy=True),
            visit_policy=visit_policy_np,
            root_value=root_value_np,
            raw_visit_counts=raw_counts_np,
            predicted_value=None,
            predicted_policy_logits=None,
            metadata=metadata,
        )
        payload.metadata["search_replay_payload_digest"] = compact_search_replay_payload_digest_v1(
            payload
        )
        self._pending_replay_payloads.pop(handle, None)
        return payload

    def flush_device_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactDeviceSearchReplayPayloadV1:
        """Return the delayed replay payload without copying large tensors to host."""

        import torch

        handle = str(replay_payload_handle)
        if not handle:
            raise ReplayCompatibilityError("replay_payload_handle must be non-empty")
        pending = self._pending_replay_payloads.get(handle)
        if pending is None:
            raise ReplayCompatibilityError("unknown or already-flushed replay payload handle")

        started = time.perf_counter()
        device = self._resolve_device(torch=torch)
        active_count = int(pending.root_index.size)
        pending_deferred_before = self._pending_deferred_one_simulation_replay_count()
        pending_deferred_after = max(
            0,
            pending_deferred_before - (1 if bool(pending.deferred_one_simulation_replay) else 0),
        )
        visit_policy_value, root_value_value, raw_counts_value, deferred_telemetry = (
            self._materialize_pending_replay_tensors(
                pending,
                torch=torch,
                device=device,
            )
        )
        visit_policy = _device_payload_tensor(
            torch=torch,
            value=visit_policy_value,
            device=device,
            dtype=torch.float32,
            shape=(active_count, ACTION_COUNT),
            name="visit_policy",
        )
        root_value = _device_payload_tensor(
            torch=torch,
            value=root_value_value,
            device=device,
            dtype=torch.float32,
            shape=(active_count,),
            name="root_value",
        )
        raw_counts = _device_payload_tensor(
            torch=torch,
            value=raw_counts_value,
            device=device,
            dtype=torch.float32,
            shape=(active_count, ACTION_COUNT),
            name="raw_visit_counts",
        )
        flush_sec = time.perf_counter() - started

        metadata = dict(pending.metadata)
        profile_telemetry = metadata.get("profile_telemetry")
        profile = dict(profile_telemetry) if isinstance(profile_telemetry, Mapping) else {}
        profile.update(deferred_telemetry)
        profile.update(
            {
                "compact_torch_search_pending_deferred_replay_payload_count": float(
                    pending_deferred_before
                ),
                "compact_torch_search_pending_deferred_replay_payload_final_count": float(
                    pending_deferred_after
                ),
                "compact_torch_search_service_replay_payload_readback_sec": 0.0,
                "compact_torch_search_service_device_replay_payload_flush_sec": float(flush_sec),
                "compact_torch_search_service_replay_payload_d2h_bytes": 0.0,
                "compact_torch_search_service_deferred_replay_payload_d2h_bytes": 0.0,
                "compact_torch_search_service_replay_payload_flushed": True,
                "compact_torch_search_service_device_replay_payload_flushed": True,
                "compact_torch_search_service_replay_payload_output": "device_torch",
            }
        )
        metadata.update(profile)
        metadata.update(
            {
                "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                "phase": "replay_critical_device",
                "search_result_schema_id": "curvyzero_compact_search_result/v1",
                "search_impl": self.search_impl,
                "num_simulations": int(self.num_simulations),
                "active_root_count": active_count,
                "search_replay_payload_digest": (
                    compact_search_deferred_replay_payload_digest_v1(handle)
                ),
                "replay_payload_origin": f"{self.search_impl}:{handle}",
                "device_replay_payload": True,
                "device_replay_payload_device": str(visit_policy.device),
                "host_search_payload_fallback_allowed": False,
                "profile_telemetry": profile,
            }
        )
        payload = CompactDeviceSearchReplayPayloadV1(
            replay_payload_handle=handle,
            root_index=pending.root_index.astype(np.int32, copy=True),
            env_row=pending.env_row.astype(np.int32, copy=True),
            player=pending.player.astype(np.int16, copy=True),
            policy_env_id=pending.policy_env_id.astype(np.int64, copy=True),
            visit_policy=visit_policy,
            root_value=root_value,
            raw_visit_counts=raw_counts,
            predicted_value=None,
            predicted_policy_logits=None,
            metadata=metadata,
        )
        self._pending_replay_payloads.pop(handle, None)
        return payload

    def _root_observation_tensor(
        self,
        *,
        torch: Any,
        root_batch: CompactRootBatchV1,
        active_root_indices: np.ndarray,
        device: Any,
    ) -> tuple[_ActiveRowSelection, Any, dict[str, Any]]:
        source = str(
            getattr(
                root_batch,
                "observation_source",
                COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
            )
        )
        telemetry: dict[str, Any] = {
            "resident_observation_required": self._require_resident_observation,
            "resident_observation_used": False,
            "resident_observation_host_fallback_allowed": False,
            "resident_observation_host_fallback_used": False,
            "resident_observation_host_fallback_count": 0.0,
            "resident_observation_h2d_bytes": 0.0,
            "resident_observation_d2h_bytes": 0.0,
            "resident_observation_device_gather_bytes": 0.0,
            "resident_observation_device_index_select_used": False,
            "resident_observation_device_slice_used": False,
            "resident_observation_source": source,
        }
        if source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1:
            resident = root_batch.resident_observation
            if resident is None:
                raise ReplayCompatibilityError(
                    "resident_device_v1 root batch is missing resident_observation"
                )
            if bool(resident.host_fallback_allowed):
                raise ReplayCompatibilityError("resident observation allowed host fallback")
            if not bool(resident.row_major_order):
                raise ReplayCompatibilityError("resident observation is not row-major")
            root_device_observation = resident.root_device_observation
            if root_device_observation is None:
                device_observation = resident.device_observation
                if not _is_torch_like_tensor(device_observation):
                    raise ReplayCompatibilityError(
                        "resident_device_v1 requires torch resident observations"
                    )
                root_device_observation = device_observation.reshape(
                    int(resident.batch_size) * int(resident.player_count),
                    *tuple(int(dim) for dim in resident.stack_shape),
                )
            if not _is_torch_like_tensor(root_device_observation):
                raise ReplayCompatibilityError(
                    "resident_device_v1 requires torch resident observations"
                )
            if not _torch_devices_match(
                torch=torch,
                lhs=root_device_observation.device,
                rhs=device,
            ):
                raise ReplayCompatibilityError(
                    "resident observation device does not match compact Torch device"
                )
            if active_root_indices.size:
                start = int(active_root_indices[0])
                stop = int(active_root_indices[-1]) + 1
                active_count = int(active_root_indices.size)
                if stop - start == active_count:
                    obs_tensor = root_device_observation[start:stop]
                    telemetry["resident_observation_device_slice_used"] = True
                else:
                    index_tensor = torch.as_tensor(
                        active_root_indices,
                        dtype=torch.long,
                        device=device,
                    )
                    obs_tensor = root_device_observation.index_select(0, index_tensor)
                    telemetry["resident_observation_device_index_select_used"] = True
                    telemetry["resident_observation_device_gather_bytes"] = float(
                        _tensor_nbytes(obs_tensor)
                    )
            else:
                obs_tensor = root_device_observation[:0]
                telemetry["resident_observation_device_slice_used"] = True
            obs_tensor = self._prepare_observation_for_model(
                torch=torch,
                obs_tensor=obs_tensor,
                telemetry=telemetry,
            )
            telemetry.update(
                {
                    "resident_observation_used": True,
                    "resident_observation_generation_id": int(resident.generation_id),
                    "resident_observation_fresh_for_step_index": int(resident.fresh_for_step_index),
                    "resident_observation_device": str(resident.device),
                    "resident_observation_dtype": str(resident.dtype),
                    "resident_observation_search_consumed_generation": int(resident.generation_id),
                }
            )
            return (
                _ActiveRowSelection(rows=np.empty((0,), dtype=np.uint8), copy_bytes=0),
                obs_tensor,
                telemetry,
            )

        if source != COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1:
            raise ReplayCompatibilityError(f"unknown compact observation_source {source!r}")
        if self._require_resident_observation:
            raise ReplayCompatibilityError(
                "CompactTorchSearchServiceV1 requires resident observation but received host batch"
            )
        np_observation = np.asarray(root_batch.observation)
        obs_selection = _select_active_rows_without_copy_when_contiguous(
            np_observation,
            active_root_indices,
            name="observation",
        )
        obs_np = obs_selection.rows
        obs_tensor = torch.as_tensor(obs_np, device=device)
        obs_tensor = self._prepare_observation_for_model(
            torch=torch,
            obs_tensor=obs_tensor,
            telemetry=telemetry,
        )
        telemetry["resident_observation_h2d_bytes"] = float(int(getattr(obs_np, "nbytes", 0)))
        return obs_selection, obs_tensor, telemetry

    def _prepare_observation_for_model(
        self,
        *,
        torch: Any,
        obs_tensor: Any,
        telemetry: dict[str, Any],
    ) -> Any:
        requested = str(self._compile_config.observation_memory_format)
        telemetry["compact_torch_search_observation_memory_format_requested"] = requested
        telemetry["compact_torch_search_observation_normalized_uint8"] = False
        telemetry["compact_torch_search_observation_dtype_before_model"] = str(
            getattr(obs_tensor, "dtype", "")
        )
        layout_copy_bytes = 0.0
        if obs_tensor.dtype == torch.uint8 and self._normalize_uint8_observation:
            obs_tensor = obs_tensor.float().div_(255.0)
            telemetry["compact_torch_search_observation_normalized_uint8"] = True
        else:
            previous = obs_tensor
            obs_tensor = obs_tensor.to(dtype=torch.float32, copy=False)
            if obs_tensor is not previous:
                layout_copy_bytes += float(_tensor_nbytes(obs_tensor))

        if requested == COMPACT_TORCH_MEMORY_FORMAT_CHANNELS_LAST:
            if int(getattr(obs_tensor, "ndim", 0)) != 4:
                raise ReplayCompatibilityError(
                    "channels_last observation_memory_format requires rank-4 observation tensor"
                )
            previous = obs_tensor
            obs_tensor = obs_tensor.contiguous(memory_format=torch.channels_last)
            if obs_tensor is not previous:
                layout_copy_bytes += float(_tensor_nbytes(obs_tensor))
            effective = COMPACT_TORCH_MEMORY_FORMAT_CHANNELS_LAST
        else:
            previous = obs_tensor
            obs_tensor = obs_tensor.contiguous()
            if obs_tensor is not previous:
                layout_copy_bytes += float(_tensor_nbytes(obs_tensor))
            effective = COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS

        telemetry["compact_torch_search_observation_memory_format_effective"] = effective
        telemetry["compact_torch_search_observation_layout_copy_bytes"] = float(layout_copy_bytes)
        telemetry["compact_torch_search_observation_dtype_model_input"] = str(
            getattr(obs_tensor, "dtype", "")
        )
        telemetry["compact_torch_search_observation_is_contiguous"] = bool(
            obs_tensor.is_contiguous()
        )
        if int(getattr(obs_tensor, "ndim", 0)) == 4:
            telemetry["compact_torch_search_observation_is_channels_last"] = bool(
                obs_tensor.is_contiguous(memory_format=torch.channels_last)
            )
        else:
            telemetry["compact_torch_search_observation_is_channels_last"] = False
        return obs_tensor

    def _action_step_from_selected(
        self,
        replay_payload_handle: str,
        root_batch: CompactRootBatchV1,
        active_root_indices: np.ndarray,
        selected_action: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> CompactSearchActionStepV1:
        step_metadata = dict(metadata)
        profile_telemetry = step_metadata.get("profile_telemetry")
        if isinstance(profile_telemetry, Mapping):
            step_metadata["profile_telemetry"] = dict(profile_telemetry)
        step_metadata.update(
            {
                "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
                "phase": "action_critical",
                "search_impl": self.search_impl,
                "num_simulations": int(self.num_simulations),
                "active_root_count": int(selected_action.size),
                "two_phase_compact_search": True,
                "compact_torch_search_service_two_phase": True,
                "replay_payload_origin": f"{self.search_impl}:{replay_payload_handle}",
                "selected_action_digest": compact_search_array_digest_v1(
                    selected_action.astype(np.int16, copy=False)
                ),
                "search_replay_payload_digest": (
                    compact_search_deferred_replay_payload_digest_v1(replay_payload_handle)
                ),
                "search_replay_payload_digest_deferred": True,
            }
        )
        return CompactSearchActionStepV1(
            replay_payload_handle=str(replay_payload_handle),
            root_index=active_root_indices.astype(np.int32, copy=True),
            env_row=root_batch.env_row[active_root_indices].astype(np.int32, copy=True),
            player=root_batch.player[active_root_indices].astype(np.int16, copy=True),
            policy_env_id=root_batch.policy_env_id[active_root_indices].astype(
                np.int64,
                copy=True,
            ),
            selected_action=selected_action.astype(np.int16, copy=True),
            metadata=step_metadata,
        )

    def _resolve_device(self, *, torch: Any) -> Any:
        if self._device is not None:
            return self._device
        try:
            return next(self._model.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _resolved_root_noise_weight(self) -> float:
        if self._root_noise_weight is not None:
            return float(self._root_noise_weight)
        cfg = getattr(self._policy, "_cfg", None)
        return float(getattr(cfg, "root_noise_weight", 0.0))

    def _compiled_helpers_for_eligibility(
        self,
        *,
        torch: Any,
        eligibility: CompactTorchCompileEligibility,
    ) -> tuple[Any | None, Any | None, dict[str, Any]]:
        cfg = self._compile_config
        telemetry: dict[str, Any] = {
            "compact_torch_search_compile_attempted": 0.0,
            "compact_torch_search_compile_cache_hit": False,
            "compact_torch_search_compile_used": False,
            "compact_torch_search_compile_runtime_status": "not_eligible",
        }
        if not bool(cfg.request_compile):
            telemetry["compact_torch_search_compile_runtime_status"] = "not_requested"
            return None, None, telemetry
        if not eligibility.eligible:
            telemetry["compact_torch_search_compile_runtime_status"] = "not_eligible"
            return None, None, telemetry
        if not _torch_compile_available(torch):
            telemetry["compact_torch_search_compile_runtime_status"] = "unavailable"
            if bool(cfg.require_torch_compile):
                raise RuntimeError("torch.compile was requested but is unavailable")
            return None, None, telemetry

        key = _compact_torch_compile_cache_key(eligibility.telemetry, cfg)
        cached = self._compiled_helper_cache.get(key)
        if cached is not None:
            telemetry.update(
                {
                    "compact_torch_search_compile_cache_hit": True,
                    "compact_torch_search_compile_used": True,
                    "compact_torch_search_compile_runtime_status": "cache_hit",
                    "compact_torch_search_compiled_helper_cache_size": float(
                        len(self._compiled_helper_cache)
                    ),
                }
            )
            return cached[0], cached[1], telemetry

        telemetry["compact_torch_search_compile_attempted"] = 1.0
        try:
            select_helper = torch.compile(
                make_compact_torch_select_leaf_fixed(torch=torch),
                mode=cfg.compile_mode,
                fullgraph=bool(cfg.fullgraph),
            )
            backup_helper = torch.compile(
                make_compact_torch_expand_and_backup_fixed(torch=torch),
                mode=cfg.compile_mode,
                fullgraph=bool(cfg.fullgraph),
            )
        except Exception as exc:
            telemetry.update(
                {
                    "compact_torch_search_compile_runtime_status": "compile_error",
                    "compact_torch_search_compile_error": str(exc),
                }
            )
            raise RuntimeError("compact Torch helper compile failed") from exc

        self._compiled_helper_cache[key] = (select_helper, backup_helper)
        telemetry.update(
            {
                "compact_torch_search_compile_used": True,
                "compact_torch_search_compile_runtime_status": "compiled",
                "compact_torch_search_compiled_helper_cache_size": float(
                    len(self._compiled_helper_cache)
                ),
            }
        )
        return select_helper, backup_helper, telemetry

    def _ensure_model_memory_format(self, *, torch: Any) -> dict[str, Any]:
        requested = str(self._compile_config.model_memory_format)
        telemetry: dict[str, Any] = {
            "compact_torch_search_model_memory_format_requested": requested,
            "compact_torch_search_model_memory_format_active": (self._model_memory_format_active),
            "compact_torch_search_model_memory_format_applied": False,
        }
        if requested == COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS:
            telemetry["compact_torch_search_model_memory_format_active"] = (
                self._model_memory_format_active
            )
            return telemetry
        if requested == COMPACT_TORCH_MEMORY_FORMAT_CHANNELS_LAST:
            raise ValueError(
                "model_memory_format=channels_last is parked for the current "
                "LightZero model because the dynamics reward head uses .view(); "
                "use model_memory_format='contiguous'"
            )
        raise ValueError(f"unsupported model_memory_format {requested!r}")

    def _compiled_model_for_runtime(
        self,
        *,
        torch: Any,
        device: Any,
    ) -> tuple[Any, Any, Any | None, Any | None, dict[str, Any]]:
        cfg = self._compile_config
        request_model_compile = bool(cfg.request_model_compile or cfg.require_model_compile)
        model_compile_mode = str(cfg.model_compile_mode or cfg.compile_mode)
        initial_inference_mode = str(cfg.initial_inference_mode)
        direct_core_mode = (
            initial_inference_mode == COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE
        )
        telemetry: dict[str, Any] = {
            "compact_torch_search_model_compile_requested": request_model_compile,
            "compact_torch_search_model_compile_attempted": 0.0,
            "compact_torch_search_model_compile_cache_hit": False,
            "compact_torch_search_model_compile_used": False,
            "compact_torch_search_model_compile_initial_path": (
                "direct_core" if direct_core_mode else "model_method"
            ),
            "compact_torch_search_model_compile_direct_core_representation_used": False,
            "compact_torch_search_model_compile_direct_core_prediction_used": False,
            "compact_torch_search_model_compile_runtime_status": (
                "not_requested" if not request_model_compile else "pending"
            ),
            "compact_torch_search_model_compile_mode": (
                model_compile_mode if request_model_compile else "none"
            ),
            "compact_torch_search_model_compile_fullgraph": bool(
                cfg.model_fullgraph and request_model_compile
            ),
            "compact_torch_search_recurrent_action_shape_mode_requested": str(
                cfg.recurrent_action_shape_mode
            ),
        }
        telemetry.update(self._ensure_model_memory_format(torch=torch))
        eager_initial = self._model.initial_inference
        eager_recurrent = self._model.recurrent_inference
        eager_representation = None
        eager_prediction = None
        if direct_core_mode:
            eager_representation = getattr(self._model, "_representation", None)
            eager_prediction = getattr(self._model, "_prediction", None)
            if not callable(eager_representation) or not callable(eager_prediction):
                raise ValueError(
                    "direct_core initial_inference_mode requires callable "
                    "model._representation and model._prediction"
                )
        if not request_model_compile:
            return eager_initial, eager_recurrent, None, None, telemetry
        if bool(cfg.require_cuda_device) and "cuda" not in str(device):
            telemetry["compact_torch_search_model_compile_runtime_status"] = "requires_cuda_device"
            if bool(cfg.require_model_compile):
                raise RuntimeError("model torch.compile requires a CUDA device")
            return eager_initial, eager_recurrent, None, None, telemetry
        if not _torch_compile_available(torch):
            telemetry["compact_torch_search_model_compile_runtime_status"] = "unavailable"
            if bool(cfg.require_model_compile):
                raise RuntimeError("model torch.compile was requested but is unavailable")
            return eager_initial, eager_recurrent, None, None, telemetry

        key = (
            id(self._model),
            str(device),
            model_compile_mode,
            bool(cfg.model_fullgraph),
            initial_inference_mode,
            str(cfg.observation_memory_format),
            str(cfg.model_memory_format),
        )
        cached = self._compiled_model_cache.get(key)
        if cached is not None:
            cached_initial, cached_recurrent, cached_representation, cached_prediction = cached
            telemetry.update(
                {
                    "compact_torch_search_model_compile_cache_hit": True,
                    "compact_torch_search_model_compile_used": True,
                    "compact_torch_search_model_compile_direct_core_representation_used": (
                        cached_representation is not None
                    ),
                    "compact_torch_search_model_compile_direct_core_prediction_used": (
                        cached_prediction is not None
                    ),
                    "compact_torch_search_model_compile_runtime_status": "cache_hit",
                    "compact_torch_search_model_compiled_cache_size": float(
                        len(self._compiled_model_cache)
                    ),
                }
            )
            return (
                cached_initial,
                cached_recurrent,
                cached_representation,
                cached_prediction,
                telemetry,
            )

        telemetry["compact_torch_search_model_compile_attempted"] = 1.0
        try:
            if direct_core_mode:
                assert eager_representation is not None
                assert eager_prediction is not None
                compiled_initial = eager_initial
                compiled_representation = torch.compile(
                    eager_representation,
                    mode=model_compile_mode,
                    fullgraph=bool(cfg.model_fullgraph),
                )
                compiled_prediction = torch.compile(
                    eager_prediction,
                    mode=model_compile_mode,
                    fullgraph=bool(cfg.model_fullgraph),
                )
            else:
                compiled_initial = torch.compile(
                    eager_initial,
                    mode=model_compile_mode,
                    fullgraph=bool(cfg.model_fullgraph),
                )
                compiled_representation = None
                compiled_prediction = None
            compiled_recurrent = torch.compile(
                eager_recurrent,
                mode=model_compile_mode,
                fullgraph=bool(cfg.model_fullgraph),
            )
        except Exception as exc:
            telemetry.update(
                {
                    "compact_torch_search_model_compile_runtime_status": "compile_error",
                    "compact_torch_search_model_compile_error": str(exc),
                }
            )
            if bool(cfg.require_model_compile):
                raise RuntimeError("compact Torch model compile failed") from exc
            return eager_initial, eager_recurrent, None, None, telemetry

        self._compiled_model_cache[key] = (
            compiled_initial,
            compiled_recurrent,
            compiled_representation,
            compiled_prediction,
        )
        telemetry.update(
            {
                "compact_torch_search_model_compile_used": True,
                "compact_torch_search_model_compile_direct_core_representation_used": (
                    compiled_representation is not None
                ),
                "compact_torch_search_model_compile_direct_core_prediction_used": (
                    compiled_prediction is not None
                ),
                "compact_torch_search_model_compile_runtime_status": "compiled",
                "compact_torch_search_model_compiled_cache_size": float(
                    len(self._compiled_model_cache)
                ),
            }
        )
        return (
            compiled_initial,
            compiled_recurrent,
            compiled_representation,
            compiled_prediction,
            telemetry,
        )


def compact_torch_fixed_shape_masks(
    root_batch: CompactRootBatchV1,
    *,
    force_all_roots_active: bool = False,
    force_all_actions_legal: bool = False,
) -> CompactTorchFixedShapeMasks:
    """Return normalized profile-only masks for fixed-shape helper checks."""

    legal_mask = _binary_mask(root_batch.legal_mask, "legal_mask")
    if legal_mask.ndim != 2:
        raise ValueError(f"legal_mask must have rank 2, got shape {legal_mask.shape}")
    root_count, action_count = (int(legal_mask.shape[0]), int(legal_mask.shape[1]))
    if action_count != ACTION_COUNT:
        raise ValueError(f"legal_mask action dimension must be {ACTION_COUNT}, got {action_count}")

    active_root_mask = _binary_mask(root_batch.active_root_mask, "active_root_mask")
    if active_root_mask.shape != (root_count,):
        raise ValueError(
            f"active_root_mask must have shape ({root_count},), got {active_root_mask.shape}"
        )

    forced_roots = False
    forced_actions = False
    if force_all_roots_active:
        active_root_mask = np.ones((root_count,), dtype=np.bool_)
        forced_roots = True
    if force_all_actions_legal:
        legal_mask = np.ones((root_count, action_count), dtype=np.bool_)
        forced_actions = True

    active_root_count = int(np.count_nonzero(active_root_mask))
    active_legal = legal_mask[active_root_mask]
    all_roots_active = bool(active_root_count == root_count)
    all_actions_legal = bool(active_legal.size == 0 or active_legal.all())
    return CompactTorchFixedShapeMasks(
        legal_mask=legal_mask.astype(np.bool_, copy=True),
        active_root_mask=active_root_mask.astype(np.bool_, copy=True),
        root_count=root_count,
        active_root_count=active_root_count,
        action_count=action_count,
        all_roots_active=all_roots_active,
        all_actions_legal=all_actions_legal,
        forced_all_roots_active=forced_roots,
        forced_all_actions_legal=forced_actions,
    )


def compact_torch_compile_eligibility(
    root_batch: CompactRootBatchV1,
    *,
    device: Any,
    root_noise_weight: float,
    config: CompactTorchCompileConfig | None = None,
    torch_module: Any | None = None,
    force_all_roots_active: bool = False,
    force_all_actions_legal: bool = False,
    extra_telemetry: Mapping[str, Any] | None = None,
    fixed_shape_masks: CompactTorchFixedShapeMasks | None = None,
) -> CompactTorchCompileEligibility:
    """Return deterministic compile eligibility and telemetry for a root batch.

    This function checks only shape/device/mask/noise preconditions. It does not
    compile, run search, call the model, or claim parity with LightZero CTree.
    """

    cfg = config or CompactTorchCompileConfig()
    masks = fixed_shape_masks
    if masks is None:
        masks = compact_torch_fixed_shape_masks(
            root_batch,
            force_all_roots_active=force_all_roots_active,
            force_all_actions_legal=force_all_actions_legal,
        )
    device_text = str(device)
    noise = float(root_noise_weight)
    observation_shape, observation_dtype = _compile_observation_shape_dtype(root_batch)
    telemetry: dict[str, Any] = {
        "compact_torch_search_label": COMPACT_TORCH_SEARCH_LABEL,
        "compact_torch_search_impl": COMPACT_TORCH_SEARCH_IMPL,
        "compact_torch_search_semantics": COMPACT_TORCH_SEARCH_SEMANTICS,
        "compact_torch_search_backend_kind": COMPACT_TORCH_SEARCH_BACKEND_KIND,
        "compact_torch_search_helper": COMPACT_TORCH_SEARCH_HELPER,
        "compact_torch_search_profile_only": True,
        "compact_torch_search_not_lightzero_ctree": True,
        "compact_torch_search_trainer_ready": False,
        "compact_torch_search_device": device_text,
        "compact_torch_search_observation_shape": observation_shape,
        "compact_torch_search_root_count": float(masks.root_count),
        "compact_torch_search_active_root_count": float(masks.active_root_count),
        "compact_torch_search_action_count": float(masks.action_count),
        "compact_torch_search_all_roots_active": masks.all_roots_active,
        "compact_torch_search_all_actions_legal": masks.all_actions_legal,
        "compact_torch_search_forced_all_roots_active": masks.forced_all_roots_active,
        "compact_torch_search_forced_all_actions_legal": masks.forced_all_actions_legal,
        "compact_torch_search_root_noise_weight": noise,
        "compact_torch_search_compile_requested": bool(cfg.request_compile),
        "compact_torch_search_compile_attempted": 0.0,
        "compact_torch_search_compile_enabled": 0.0,
        "compact_torch_search_compile_mode": cfg.compile_mode if cfg.request_compile else "none",
        "compact_torch_search_compile_fullgraph": bool(cfg.fullgraph and cfg.request_compile),
        "compact_torch_search_compile_helper": (
            COMPACT_TORCH_SEARCH_HELPER if cfg.request_compile else "none"
        ),
        "compact_torch_search_defer_one_simulation_replay_payload_requested": bool(
            cfg.defer_one_simulation_replay_payload
        ),
        "compact_torch_search_compile_signature": [
            masks.root_count,
            masks.active_root_count,
            observation_shape,
            observation_dtype,
            device_text,
            masks.action_count,
        ],
    }
    if extra_telemetry:
        telemetry.update(
            {str(key): _plain_telemetry_value(value) for key, value in extra_telemetry.items()}
        )

    status = "eligible"
    reason = "preconditions_satisfied"
    if not cfg.request_compile:
        status = "not_requested"
        reason = "compile_not_requested"
    elif cfg.expected_root_count is not None and masks.root_count != int(cfg.expected_root_count):
        status = "fallback_precondition"
        reason = "requires_expected_root_count"
    elif cfg.expected_observation_shape is not None and tuple(observation_shape) != tuple(
        int(dim) for dim in cfg.expected_observation_shape
    ):
        status = "fallback_precondition"
        reason = "requires_expected_observation_shape"
    elif cfg.expected_device is not None and device_text != str(cfg.expected_device):
        status = "fallback_precondition"
        reason = "requires_expected_device"
    elif masks.action_count != int(cfg.action_count):
        status = "fallback_precondition"
        reason = "requires_fixed_action_count"
    elif cfg.require_cuda_device and not device_text.startswith("cuda"):
        status = "fallback_precondition"
        reason = "requires_cuda_device"
    elif cfg.require_torch_compile and not _torch_compile_available(torch_module):
        status = "fallback_precondition"
        reason = "torch_compile_unavailable"
    elif cfg.require_all_roots_active and not masks.all_roots_active:
        status = "fallback_precondition"
        reason = "requires_all_roots_active"
    elif cfg.require_all_actions_legal and not masks.all_actions_legal:
        status = "fallback_precondition"
        reason = "requires_all_actions_legal"
    elif cfg.require_root_noise_zero and noise != 0.0:
        status = "fallback_precondition"
        reason = "requires_root_noise_zero"

    eligible = status == "eligible"
    telemetry.update(
        {
            "compact_torch_search_compile_status": status,
            "compact_torch_search_compile_reason": reason,
            "compact_torch_search_compile_enabled": 1.0 if eligible else 0.0,
        }
    )
    return CompactTorchCompileEligibility(
        eligible=eligible,
        status=status,
        reason=reason,
        telemetry=telemetry,
    )


def _compile_observation_shape_dtype(root_batch: CompactRootBatchV1) -> tuple[list[int], str]:
    if (
        str(getattr(root_batch, "observation_source", COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1))
        == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        and root_batch.resident_observation is not None
    ):
        resident = root_batch.resident_observation
        return (
            [
                int(resident.batch_size),
                int(resident.player_count),
                *[int(dim) for dim in resident.stack_shape],
            ],
            str(resident.dtype),
        )
    observation = np.asarray(root_batch.observation)
    return [int(dim) for dim in observation.shape], str(observation.dtype)


def _compact_torch_compile_cache_key(
    telemetry: Mapping[str, Any],
    cfg: CompactTorchCompileConfig,
) -> tuple[Any, ...]:
    signature = telemetry.get("compact_torch_search_compile_signature", ())
    return (
        _hashable_plain_value(signature),
        bool(cfg.fullgraph),
        str(cfg.compile_mode),
        int(cfg.action_count),
    )


def _hashable_plain_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _hashable_plain_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable_plain_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_plain_value(item) for item in value)
    return value


def make_compact_torch_select_leaf_fixed(*, torch: Any) -> Any:
    """Build the fixed-shape PUCT select helper copied out of profile code."""

    def select_leaf(
        edge_child: Any,
        edge_visit: Any,
        edge_value_sum: Any,
        edge_prior: Any,
        path_node_history: Any,
        path_action_history: Any,
        path_active_history: Any,
        flat_mask_tensor: Any,
        row_index: Any,
        min_value: Any,
        max_value: Any,
        simulation_index: int,
        pb_c_base: float,
        pb_c_init: float,
        value_delta_max: float,
    ) -> tuple[Any, Any]:
        current_node = torch.zeros_like(row_index)
        active = torch.ones(
            row_index.shape,
            dtype=torch.bool,
            device=row_index.device,
        )
        leaf_parent = torch.zeros_like(row_index)
        leaf_action = torch.zeros_like(row_index)

        for depth_index in range(simulation_index + 1):
            visits = edge_visit[row_index, current_node, :]
            values = edge_value_sum[row_index, current_node, :]
            priors = edge_prior[row_index, current_node, :]
            parent_visit = visits.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean_value = values / visits.clamp_min(1.0)
            unvisited = visits <= 0.0
            normalized_value = (mean_value - min_value[:, None]) / (
                (max_value[:, None] - min_value[:, None]).clamp_min(value_delta_max)
            )
            normalized_value = torch.where(
                unvisited,
                torch.zeros_like(normalized_value),
                normalized_value,
            )
            pb_c = (
                (torch.log((parent_visit + pb_c_base + 1.0) / pb_c_base) + pb_c_init)
                * torch.sqrt(parent_visit)
                / (visits + 1.0)
            )
            score = normalized_value + priors * pb_c
            if depth_index == 0:
                score = score.masked_fill(~flat_mask_tensor, -1.0e9)
            selected_action = torch.argmax(score, dim=1)

            path_node_history[depth_index].copy_(current_node)
            path_action_history[depth_index].copy_(selected_action)
            path_active_history[depth_index].copy_(active)

            selected_child = edge_child[row_index, current_node, selected_action]
            needs_expand = active & (selected_child < 0)
            leaf_parent = torch.where(needs_expand, current_node, leaf_parent)
            leaf_action = torch.where(needs_expand, selected_action, leaf_action)
            continue_active = active & (selected_child >= 0)
            current_node = torch.where(continue_active, selected_child, current_node)
            active = continue_active

        return leaf_parent, leaf_action

    return select_leaf


def make_compact_torch_expand_and_backup_fixed(*, torch: Any) -> Any:
    """Build the fixed-shape expand/backup helper copied out of profile code."""

    def expand_and_backup(
        edge_child: Any,
        edge_visit: Any,
        edge_value_sum: Any,
        edge_reward: Any,
        edge_prior: Any,
        latent_pool: Any,
        node_latent_slot: Any,
        next_node_index: Any,
        min_value: Any,
        max_value: Any,
        path_node_history: Any,
        path_action_history: Any,
        path_active_history: Any,
        row_index: Any,
        leaf_parent: Any,
        leaf_action: Any,
        next_latent_state: Any,
        reward: Any,
        value: Any,
        recurrent_priors: Any,
        simulation_index: int,
        discount_factor: float,
    ) -> tuple[Any, Any, Any]:
        new_node = next_node_index
        edge_child[row_index, leaf_parent, leaf_action] = new_node
        edge_reward[row_index, leaf_parent, leaf_action] = reward
        edge_prior[row_index, new_node, :] = recurrent_priors
        latent_pool[simulation_index + 1].copy_(next_latent_state)
        node_latent_slot[row_index, new_node] = simulation_index + 1
        next_node_index = next_node_index + 1

        bootstrap = value
        for depth_index in range(simulation_index, -1, -1):
            nodes = path_node_history[depth_index]
            actions = path_action_history[depth_index]
            mask = path_active_history[depth_index]
            mask_f = mask.float()
            backed_value = edge_reward[row_index, nodes, actions] + discount_factor * bootstrap
            edge_visit[row_index, nodes, actions] += mask_f
            edge_value_sum[row_index, nodes, actions] += backed_value * mask_f
            min_value = torch.where(mask, torch.minimum(min_value, backed_value), min_value)
            max_value = torch.where(mask, torch.maximum(max_value, backed_value), max_value)
            bootstrap = torch.where(mask, backed_value, bootstrap)
        return next_node_index, min_value, max_value

    return expand_and_backup


def _compact_torch_validated_result(
    root_batch: CompactRootBatchV1,
    *,
    selected_action: np.ndarray,
    visit_policy: np.ndarray,
    root_value: np.ndarray,
    num_simulations: int,
    raw_visit_counts: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    from curvyzero.training.compact_search_service import compact_search_result_v1_from_arrays

    arrays: dict[str, Any] = {
        "selected_action": selected_action,
        "visit_policy": visit_policy,
        "root_value": root_value,
        "raw_visit_counts": raw_visit_counts,
        "search_impl": COMPACT_TORCH_SEARCH_SERVICE_IMPL,
        "actual_search_simulations": int(num_simulations),
    }
    return compact_search_result_v1_from_arrays(
        root_batch,
        arrays,
        default_search_impl=COMPACT_TORCH_SEARCH_SERVICE_IMPL,
        default_num_simulations=int(num_simulations),
        metadata=metadata,
    )


def _run_compact_torch_tree_search(
    *,
    torch: Any,
    policy: Any,
    model: Any,
    root_policy_logits: Any,
    root_value: Any,
    root_latent_state: Any,
    legal_mask: np.ndarray,
    num_simulations: int,
    root_noise_weight: float,
    device: Any,
    select_helper: Any | None = None,
    backup_helper: Any | None = None,
    recurrent_inference_fn: Any | None = None,
    recurrent_action_shape_mode: str = "auto",
    timing_mode: str = COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC,
    runtime_metadata: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any]:
    root_count = int(np.asarray(legal_mask).shape[0])
    if root_count <= 0:
        raise ValueError("compact Torch search requires active roots")
    cfg = getattr(policy, "_cfg", None)
    pb_c_base = float(getattr(cfg, "pb_c_base", 19652))
    pb_c_init = float(getattr(cfg, "pb_c_init", 1.25))
    discount_factor = float(getattr(cfg, "discount_factor", 0.997))
    root_dirichlet_alpha = float(getattr(cfg, "root_dirichlet_alpha", 0.3))
    value_delta_max = float(getattr(cfg, "value_delta_max", 0.01))
    if int(num_simulations) == 1:
        return _run_compact_torch_one_simulation_search(
            torch=torch,
            policy=policy,
            model=model,
            root_policy_logits=root_policy_logits,
            root_value=root_value,
            root_latent_state=root_latent_state,
            legal_mask=legal_mask,
            root_noise_weight=root_noise_weight,
            root_dirichlet_alpha=root_dirichlet_alpha,
            discount_factor=discount_factor,
            device=device,
            recurrent_inference_fn=recurrent_inference_fn,
            recurrent_action_shape_mode=recurrent_action_shape_mode,
            timing_mode=timing_mode,
            runtime_metadata=runtime_metadata,
        )
    tree_total_started = time.perf_counter()
    tree_cuda_event = _start_cuda_event_timing(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
    )
    tree_setup_started = time.perf_counter()
    max_nodes = int(num_simulations) + 1
    flat_mask_tensor = torch.as_tensor(
        np.asarray(legal_mask, dtype=np.bool_),
        dtype=torch.bool,
        device=device,
    )
    row_index = torch.arange(root_count, dtype=torch.long, device=device)
    edge_child = torch.full(
        (root_count, max_nodes, ACTION_COUNT),
        -1,
        dtype=torch.long,
        device=device,
    )
    edge_visit = torch.zeros(
        (root_count, max_nodes, ACTION_COUNT),
        dtype=torch.float32,
        device=device,
    )
    edge_value_sum = torch.zeros_like(edge_visit)
    edge_reward = torch.zeros_like(edge_visit)
    edge_prior = torch.zeros_like(edge_visit)
    latent_pool = torch.empty(
        (max_nodes, root_count) + tuple(int(dim) for dim in root_latent_state.shape[1:]),
        dtype=root_latent_state.dtype,
        device=device,
    )
    node_latent_slot = torch.zeros(
        (root_count, max_nodes),
        dtype=torch.long,
        device=device,
    )
    next_node_index = torch.ones((root_count,), dtype=torch.long, device=device)
    min_value = torch.full((root_count,), float("inf"), dtype=torch.float32, device=device)
    max_value = torch.full((root_count,), -float("inf"), dtype=torch.float32, device=device)
    path_node_history = torch.empty(
        (int(num_simulations), root_count),
        dtype=torch.long,
        device=device,
    )
    path_action_history = torch.empty_like(path_node_history)
    path_active_history = torch.empty(
        (int(num_simulations), root_count),
        dtype=torch.bool,
        device=device,
    )

    root_prior_build_started = time.perf_counter()
    root_logits = root_policy_logits.reshape(root_count, -1)[:, :ACTION_COUNT].float()
    root_priors = torch.softmax(
        root_logits.masked_fill(~flat_mask_tensor, -1.0e9),
        dim=1,
    )
    if float(root_noise_weight) > 0.0:
        alpha = torch.full(
            (ACTION_COUNT,),
            root_dirichlet_alpha,
            dtype=torch.float32,
            device=device,
        )
        noise = torch.distributions.Dirichlet(alpha).sample((root_count,))
        noise = noise * flat_mask_tensor.float()
        noise = noise / noise.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        root_priors = root_priors * (1.0 - root_noise_weight) + noise * root_noise_weight
        root_priors = root_priors / root_priors.sum(dim=1, keepdim=True).clamp_min(1.0e-12)

    edge_prior[:, 0, :] = root_priors
    tree_root_prior_build_sec = _elapsed(root_prior_build_started)

    latent_pool[0].copy_(root_latent_state)
    root_value_tensor = _policy_inverse_scalar_value(
        policy=policy,
        value=root_value,
        torch=torch,
        root_count=root_count,
        device=device,
    )
    if select_helper is None:
        select_helper = make_compact_torch_select_leaf_fixed(torch=torch)
    if backup_helper is None:
        backup_helper = make_compact_torch_expand_and_backup_fixed(torch=torch)
    if recurrent_action_shape_mode not in {"auto", "flat", "column"}:
        raise ValueError(
            "recurrent_action_shape_mode must be auto, flat, or column; "
            f"got {recurrent_action_shape_mode!r}"
        )
    recurrent_inference = recurrent_inference_fn or model.recurrent_inference
    action_shape_mode = (
        "none" if recurrent_action_shape_mode == "auto" else recurrent_action_shape_mode
    )
    recurrent_exception_fallback_count = 0
    tree_setup_sec = _elapsed(tree_setup_started)
    tree_select_enqueue_sec = 0.0
    tree_recurrent_action_build_sec = 0.0
    tree_recurrent_inference_enqueue_sec = 0.0
    recurrent_cuda_events: list[dict[str, Any]] = []
    tree_recurrent_output_decode_sec = 0.0
    tree_backup_enqueue_sec = 0.0

    for simulation_index in range(int(num_simulations)):
        select_started = time.perf_counter()
        leaf_parent, leaf_action = select_helper(
            edge_child,
            edge_visit,
            edge_value_sum,
            edge_prior,
            path_node_history,
            path_action_history,
            path_active_history,
            flat_mask_tensor,
            row_index,
            min_value,
            max_value,
            simulation_index,
            pb_c_base,
            pb_c_init,
            value_delta_max,
        )
        tree_select_enqueue_sec += _elapsed(select_started)
        action_build_started = time.perf_counter()
        parent_latent_slots = node_latent_slot[row_index, leaf_parent]
        parent_latents = latent_pool[parent_latent_slots, row_index]
        action_input = _recurrent_action_input(
            np_module=np,
            torch=torch,
            actions=leaf_action,
            device=device,
            mode=action_shape_mode,
        )
        tree_recurrent_action_build_sec += _elapsed(action_build_started)
        recurrent_started = time.perf_counter()
        recurrent_cuda_event = _start_cuda_event_timing(
            torch=torch,
            device=device,
            timing_mode=timing_mode,
        )
        if recurrent_action_shape_mode == "auto":
            try:
                recurrent_output = recurrent_inference(parent_latents, action_input)
                if action_shape_mode == "none":
                    action_shape_mode = "flat"
            except Exception:
                recurrent_exception_fallback_count += 1
                action_input = _recurrent_action_input(
                    np_module=np,
                    torch=torch,
                    actions=leaf_action,
                    device=device,
                    mode="column",
                )
                recurrent_output = recurrent_inference(parent_latents, action_input)
                action_shape_mode = "column"
        else:
            recurrent_output = recurrent_inference(parent_latents, action_input)
        _finish_cuda_event_timing(recurrent_cuda_event)
        recurrent_cuda_events.append(recurrent_cuda_event)
        tree_recurrent_inference_enqueue_sec += _elapsed(recurrent_started)

        output_decode_started = time.perf_counter()
        next_latent_state = _network_output_field(
            recurrent_output,
            ("latent_state", "hidden_state"),
        )
        if next_latent_state is None:
            raise ValueError("CompactTorchSearchServiceV1 requires recurrent latent_state")
        reward = _policy_inverse_scalar_value(
            policy=policy,
            value=_network_output_field(recurrent_output, ("reward", "value_prefix")),
            torch=torch,
            root_count=root_count,
            device=device,
        )
        value = _policy_inverse_scalar_value(
            policy=policy,
            value=_network_output_field(recurrent_output, ("value", "predicted_value")),
            torch=torch,
            root_count=root_count,
            device=device,
        )
        recurrent_logits = _network_output_field(
            recurrent_output,
            ("policy_logits", "policy"),
        )
        if recurrent_logits is None:
            raise ValueError("CompactTorchSearchServiceV1 requires recurrent policy_logits")
        recurrent_priors = torch.softmax(
            recurrent_logits.reshape(root_count, -1)[:, :ACTION_COUNT].float(),
            dim=1,
        )
        tree_recurrent_output_decode_sec += _elapsed(output_decode_started)
        backup_started = time.perf_counter()
        next_node_index, min_value, max_value = backup_helper(
            edge_child,
            edge_visit,
            edge_value_sum,
            edge_reward,
            edge_prior,
            latent_pool,
            node_latent_slot,
            next_node_index,
            min_value,
            max_value,
            path_node_history,
            path_action_history,
            path_active_history,
            row_index,
            leaf_parent,
            leaf_action,
            next_latent_state,
            reward,
            value,
            recurrent_priors,
            simulation_index,
            discount_factor,
        )
        tree_backup_enqueue_sec += _elapsed(backup_started)

    policy_build_started = time.perf_counter()
    root_visits = edge_visit[:, 0, :] * flat_mask_tensor.float()
    raw_visit_totals = root_visits.sum(dim=1, keepdim=True)
    visit_totals = raw_visit_totals.clamp_min(1.0)
    visit_policy = root_visits / visit_totals
    actions = torch.argmax(root_visits.masked_fill(~flat_mask_tensor, -1.0e9), dim=1)
    root_values = edge_value_sum[:, 0, :].sum(dim=1) / visit_totals.reshape(-1)
    root_values = torch.where(
        raw_visit_totals.reshape(-1) > 0.0,
        root_values,
        root_value_tensor,
    )
    tree_policy_build_sec = _elapsed(policy_build_started)
    _finish_cuda_event_timing(tree_cuda_event)
    tree_sync_sec = _timed_profile_sync_torch_device_if_cuda(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
        phase="tree",
    )
    tree_total_sec = _elapsed(tree_total_started)
    tree_cuda_event_sec, tree_cuda_event_status = _cuda_event_elapsed_sec(tree_cuda_event)
    tree_recurrent_inference_cuda_event_sec, recurrent_cuda_event_status = (
        _cuda_event_elapsed_total_sec(recurrent_cuda_events)
    )
    tree_accounted_sec = (
        tree_setup_sec
        + tree_select_enqueue_sec
        + tree_recurrent_action_build_sec
        + tree_recurrent_inference_enqueue_sec
        + tree_recurrent_output_decode_sec
        + tree_backup_enqueue_sec
        + tree_policy_build_sec
        + tree_sync_sec
    )
    tree_residual_sec = tree_total_sec - tree_accounted_sec
    if runtime_metadata is not None:
        runtime_metadata.update(
            {
                "compact_torch_search_one_simulation_fast_path": False,
                "compact_torch_search_recurrent_action_shape_mode_effective": (
                    "flat" if action_shape_mode == "none" else action_shape_mode
                ),
                "compact_torch_search_recurrent_action_shape_exception_fallback_count": float(
                    recurrent_exception_fallback_count
                ),
                "compact_torch_search_recurrent_inference_calls": float(num_simulations),
                "compact_torch_search_service_tree_setup_sec": float(tree_setup_sec),
                "compact_torch_search_service_tree_root_prior_build_sec": float(
                    tree_root_prior_build_sec
                ),
                "compact_torch_search_service_tree_root_prior_select_sec": 0.0,
                "compact_torch_search_service_tree_select_enqueue_sec": float(
                    tree_select_enqueue_sec
                ),
                "compact_torch_search_service_tree_recurrent_action_build_sec": float(
                    tree_recurrent_action_build_sec
                ),
                "compact_torch_search_service_tree_recurrent_inference_enqueue_sec": float(
                    tree_recurrent_inference_enqueue_sec
                ),
                "compact_torch_search_service_tree_recurrent_inference_cuda_event_sec": float(
                    tree_recurrent_inference_cuda_event_sec
                ),
                "compact_torch_search_service_tree_recurrent_inference_cuda_event_status": str(
                    recurrent_cuda_event_status
                ),
                "compact_torch_search_service_tree_recurrent_output_decode_sec": float(
                    tree_recurrent_output_decode_sec
                ),
                "compact_torch_search_service_tree_backup_enqueue_sec": float(
                    tree_backup_enqueue_sec
                ),
                "compact_torch_search_service_tree_policy_build_sec": float(tree_policy_build_sec),
                "compact_torch_search_service_tree_sync_sec": float(tree_sync_sec),
                "compact_torch_search_service_tree_cuda_event_sec": float(tree_cuda_event_sec),
                "compact_torch_search_service_tree_cuda_event_status": str(tree_cuda_event_status),
                "compact_torch_search_service_timing_mode": str(timing_mode),
                "compact_torch_search_service_tree_total_sec": float(tree_total_sec),
                "compact_torch_search_service_tree_accounted_sec": float(tree_accounted_sec),
                "compact_torch_search_service_tree_residual_sec": float(tree_residual_sec),
                "compact_torch_search_service_tree_unaccounted_sec": float(
                    max(0.0, tree_residual_sec)
                ),
                "compact_torch_search_service_tree_overaccounted_sec": float(
                    max(0.0, -tree_residual_sec)
                ),
            }
        )
    return actions, root_values, visit_policy, root_visits


def _run_compact_torch_one_simulation_search(
    *,
    torch: Any,
    policy: Any,
    model: Any,
    root_policy_logits: Any,
    root_value: Any,
    root_latent_state: Any,
    legal_mask: np.ndarray,
    root_noise_weight: float,
    root_dirichlet_alpha: float,
    discount_factor: float,
    device: Any,
    recurrent_inference_fn: Any | None = None,
    recurrent_action_shape_mode: str = "auto",
    timing_mode: str = COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC,
    runtime_metadata: dict[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any]:
    tree_total_started = time.perf_counter()
    tree_cuda_event = _start_cuda_event_timing(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
    )
    root_prior_started = time.perf_counter()
    root_count = int(np.asarray(legal_mask).shape[0])
    flat_mask_tensor = torch.as_tensor(
        np.asarray(legal_mask, dtype=np.bool_),
        dtype=torch.bool,
        device=device,
    )
    root_logits = root_policy_logits.reshape(root_count, -1)[:, :ACTION_COUNT].float()
    masked_root_logits = root_logits.masked_fill(~flat_mask_tensor, -1.0e9)
    root_prior_softmax_skipped = float(root_noise_weight) <= 0.0
    if root_prior_softmax_skipped:
        selected = torch.argmax(masked_root_logits, dim=1)
    else:
        root_priors = torch.softmax(masked_root_logits, dim=1)
        alpha = torch.full(
            (ACTION_COUNT,),
            float(root_dirichlet_alpha),
            dtype=torch.float32,
            device=device,
        )
        noise = torch.distributions.Dirichlet(alpha).sample((root_count,))
        noise = noise * flat_mask_tensor.float()
        noise = noise / noise.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        root_priors = root_priors * (1.0 - root_noise_weight) + noise * root_noise_weight
        root_priors = root_priors / root_priors.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        selected = torch.argmax(root_priors.masked_fill(~flat_mask_tensor, -1.0e9), dim=1)
    tree_root_prior_select_sec = _elapsed(root_prior_started)
    tree_root_prior_build_sec = tree_root_prior_select_sec
    recurrent_inference = recurrent_inference_fn or model.recurrent_inference
    if recurrent_action_shape_mode not in {"auto", "flat", "column"}:
        raise ValueError(
            "recurrent_action_shape_mode must be auto, flat, or column; "
            f"got {recurrent_action_shape_mode!r}"
        )
    action_shape_mode = (
        "none" if recurrent_action_shape_mode == "auto" else recurrent_action_shape_mode
    )
    recurrent_exception_fallback_count = 0
    action_build_started = time.perf_counter()
    action_input = _recurrent_action_input(
        np_module=np,
        torch=torch,
        actions=selected,
        device=device,
        mode=action_shape_mode,
    )
    tree_recurrent_action_build_sec = _elapsed(action_build_started)
    recurrent_started = time.perf_counter()
    recurrent_cuda_event = _start_cuda_event_timing(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
    )
    if recurrent_action_shape_mode == "auto":
        try:
            recurrent_output = recurrent_inference(root_latent_state, action_input)
            if action_shape_mode == "none":
                action_shape_mode = "flat"
        except Exception:
            recurrent_exception_fallback_count += 1
            action_input = _recurrent_action_input(
                np_module=np,
                torch=torch,
                actions=selected,
                device=device,
                mode="column",
            )
            recurrent_output = recurrent_inference(root_latent_state, action_input)
            action_shape_mode = "column"
    else:
        recurrent_output = recurrent_inference(root_latent_state, action_input)
    _finish_cuda_event_timing(recurrent_cuda_event)
    tree_recurrent_inference_enqueue_sec = _elapsed(recurrent_started)
    output_decode_started = time.perf_counter()
    next_latent_state = _network_output_field(
        recurrent_output,
        ("latent_state", "hidden_state"),
    )
    if next_latent_state is None:
        raise ValueError("CompactTorchSearchServiceV1 requires recurrent latent_state")
    reward = _policy_inverse_scalar_value(
        policy=policy,
        value=_network_output_field(recurrent_output, ("reward", "value_prefix")),
        torch=torch,
        root_count=root_count,
        device=device,
    )
    value = _policy_inverse_scalar_value(
        policy=policy,
        value=_network_output_field(recurrent_output, ("value", "predicted_value")),
        torch=torch,
        root_count=root_count,
        device=device,
    )
    recurrent_logits = _network_output_field(
        recurrent_output,
        ("policy_logits", "policy"),
    )
    if recurrent_logits is None:
        raise ValueError("CompactTorchSearchServiceV1 requires recurrent policy_logits")
    tree_recurrent_output_decode_sec = _elapsed(output_decode_started)
    del next_latent_state, recurrent_logits, root_value
    policy_build_started = time.perf_counter()
    raw_counts = torch.zeros(
        (root_count, ACTION_COUNT),
        dtype=torch.float32,
        device=device,
    )
    row_index = torch.arange(root_count, dtype=torch.long, device=device)
    raw_counts[row_index, selected] = 1.0
    visit_policy = raw_counts
    root_values = reward + float(discount_factor) * value
    tree_policy_build_sec = _elapsed(policy_build_started)
    _finish_cuda_event_timing(tree_cuda_event)
    tree_sync_sec = _timed_profile_sync_torch_device_if_cuda(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
        phase="tree",
    )
    tree_total_sec = _elapsed(tree_total_started)
    tree_cuda_event_sec, tree_cuda_event_status = _cuda_event_elapsed_sec(tree_cuda_event)
    tree_recurrent_inference_cuda_event_sec, recurrent_cuda_event_status = (
        _cuda_event_elapsed_total_sec([recurrent_cuda_event])
    )
    tree_accounted_sec = (
        tree_root_prior_select_sec
        + tree_recurrent_action_build_sec
        + tree_recurrent_inference_enqueue_sec
        + tree_recurrent_output_decode_sec
        + tree_policy_build_sec
        + tree_sync_sec
    )
    tree_residual_sec = tree_total_sec - tree_accounted_sec
    if runtime_metadata is not None:
        runtime_metadata.update(
            {
                "compact_torch_search_one_simulation_fast_path": True,
                "compact_torch_search_one_simulation_root_prior_softmax_skipped": (
                    root_prior_softmax_skipped
                ),
                "compact_torch_search_one_simulation_selection_mode": (
                    "masked_logits_argmax" if root_prior_softmax_skipped else "noisy_prior_argmax"
                ),
                "compact_torch_search_recurrent_action_shape_mode_effective": (
                    "flat" if action_shape_mode == "none" else action_shape_mode
                ),
                "compact_torch_search_recurrent_action_shape_exception_fallback_count": float(
                    recurrent_exception_fallback_count
                ),
                "compact_torch_search_recurrent_inference_calls": 1.0,
                "compact_torch_search_service_tree_setup_sec": 0.0,
                "compact_torch_search_service_tree_root_prior_build_sec": float(
                    tree_root_prior_build_sec
                ),
                "compact_torch_search_service_tree_root_prior_select_sec": float(
                    tree_root_prior_select_sec
                ),
                "compact_torch_search_service_tree_select_enqueue_sec": 0.0,
                "compact_torch_search_service_tree_recurrent_action_build_sec": float(
                    tree_recurrent_action_build_sec
                ),
                "compact_torch_search_service_tree_recurrent_inference_enqueue_sec": float(
                    tree_recurrent_inference_enqueue_sec
                ),
                "compact_torch_search_service_tree_recurrent_inference_cuda_event_sec": float(
                    tree_recurrent_inference_cuda_event_sec
                ),
                "compact_torch_search_service_tree_recurrent_inference_cuda_event_status": str(
                    recurrent_cuda_event_status
                ),
                "compact_torch_search_service_tree_recurrent_output_decode_sec": float(
                    tree_recurrent_output_decode_sec
                ),
                "compact_torch_search_service_tree_backup_enqueue_sec": 0.0,
                "compact_torch_search_service_tree_policy_build_sec": float(tree_policy_build_sec),
                "compact_torch_search_service_tree_sync_sec": float(tree_sync_sec),
                "compact_torch_search_service_tree_cuda_event_sec": float(tree_cuda_event_sec),
                "compact_torch_search_service_tree_cuda_event_status": str(tree_cuda_event_status),
                "compact_torch_search_service_timing_mode": str(timing_mode),
                "compact_torch_search_service_tree_total_sec": float(tree_total_sec),
                "compact_torch_search_service_tree_accounted_sec": float(tree_accounted_sec),
                "compact_torch_search_service_tree_residual_sec": float(tree_residual_sec),
                "compact_torch_search_service_tree_unaccounted_sec": float(
                    max(0.0, tree_residual_sec)
                ),
                "compact_torch_search_service_tree_overaccounted_sec": float(
                    max(0.0, -tree_residual_sec)
                ),
            }
        )
    return selected, root_values, visit_policy, raw_counts


def _select_compact_torch_zero_noise_one_simulation_action(
    *,
    torch: Any,
    root_policy_logits: Any,
    legal_mask: np.ndarray,
    device: Any,
    timing_mode: str = COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC,
    runtime_metadata: dict[str, Any] | None = None,
) -> Any:
    """Select one-simulation actions without replay-critical recurrent work."""

    tree_total_started = time.perf_counter()
    tree_cuda_event = _start_cuda_event_timing(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
    )
    root_prior_started = time.perf_counter()
    root_count = int(np.asarray(legal_mask).shape[0])
    flat_mask_tensor = torch.as_tensor(
        np.asarray(legal_mask, dtype=np.bool_),
        dtype=torch.bool,
        device=device,
    )
    root_logits = root_policy_logits.reshape(root_count, -1)[:, :ACTION_COUNT].float()
    masked_root_logits = root_logits.masked_fill(~flat_mask_tensor, -1.0e9)
    selected = torch.argmax(masked_root_logits, dim=1)
    tree_root_prior_select_sec = _elapsed(root_prior_started)
    _finish_cuda_event_timing(tree_cuda_event)
    tree_total_sec = _elapsed(tree_total_started)
    tree_cuda_event_sec, tree_cuda_event_status = _cuda_event_elapsed_sec(tree_cuda_event)
    tree_accounted_sec = tree_root_prior_select_sec
    tree_residual_sec = tree_total_sec - tree_accounted_sec
    if runtime_metadata is not None:
        runtime_metadata.update(
            {
                "compact_torch_search_one_simulation_fast_path": True,
                "compact_torch_search_one_simulation_root_prior_softmax_skipped": True,
                "compact_torch_search_one_simulation_selection_mode": "masked_logits_argmax",
                "compact_torch_search_one_simulation_replay_materialization_deferred": True,
                "compact_torch_search_recurrent_action_shape_mode_effective": "deferred",
                "compact_torch_search_recurrent_action_shape_exception_fallback_count": 0.0,
                "compact_torch_search_recurrent_inference_calls": 0.0,
                "compact_torch_search_service_tree_setup_sec": 0.0,
                "compact_torch_search_service_tree_root_prior_build_sec": float(
                    tree_root_prior_select_sec
                ),
                "compact_torch_search_service_tree_root_prior_select_sec": float(
                    tree_root_prior_select_sec
                ),
                "compact_torch_search_service_tree_select_enqueue_sec": 0.0,
                "compact_torch_search_service_tree_recurrent_action_build_sec": 0.0,
                "compact_torch_search_service_tree_recurrent_inference_enqueue_sec": 0.0,
                "compact_torch_search_service_tree_recurrent_inference_cuda_event_sec": 0.0,
                "compact_torch_search_service_tree_recurrent_inference_cuda_event_status": (
                    "deferred"
                ),
                "compact_torch_search_service_tree_recurrent_output_decode_sec": 0.0,
                "compact_torch_search_service_tree_backup_enqueue_sec": 0.0,
                "compact_torch_search_service_tree_policy_build_sec": 0.0,
                "compact_torch_search_service_tree_sync_sec": 0.0,
                "compact_torch_search_service_tree_cuda_event_sec": float(tree_cuda_event_sec),
                "compact_torch_search_service_tree_cuda_event_status": str(
                    tree_cuda_event_status
                ),
                "compact_torch_search_service_timing_mode": str(timing_mode),
                "compact_torch_search_service_tree_total_sec": float(tree_total_sec),
                "compact_torch_search_service_tree_accounted_sec": float(tree_accounted_sec),
                "compact_torch_search_service_tree_residual_sec": float(tree_residual_sec),
                "compact_torch_search_service_tree_unaccounted_sec": float(
                    max(0.0, tree_residual_sec)
                ),
                "compact_torch_search_service_tree_overaccounted_sec": float(
                    max(0.0, -tree_residual_sec)
                ),
            }
        )
    return selected


def _materialize_compact_torch_one_simulation_replay_from_selected(
    *,
    torch: Any,
    policy: Any,
    model: Any,
    selected: Any,
    root_latent_state: Any,
    discount_factor: float,
    device: Any,
    recurrent_inference_fn: Any | None = None,
    recurrent_action_shape_mode: str = "auto",
    timing_mode: str = COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Build one-simulation replay tensors after the action-critical phase."""

    started = time.perf_counter()
    replay_cuda_event = _start_cuda_event_timing(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
    )
    recurrent_inference = recurrent_inference_fn or model.recurrent_inference
    if recurrent_action_shape_mode not in {"auto", "flat", "column"}:
        raise ValueError(
            "recurrent_action_shape_mode must be auto, flat, or column; "
            f"got {recurrent_action_shape_mode!r}"
        )
    action_shape_mode = (
        "none" if recurrent_action_shape_mode == "auto" else recurrent_action_shape_mode
    )
    recurrent_exception_fallback_count = 0
    action_build_started = time.perf_counter()
    action_input = _recurrent_action_input(
        np_module=np,
        torch=torch,
        actions=selected,
        device=device,
        mode=action_shape_mode,
    )
    action_build_sec = _elapsed(action_build_started)
    recurrent_started = time.perf_counter()
    recurrent_cuda_event = _start_cuda_event_timing(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
    )
    if recurrent_action_shape_mode == "auto":
        try:
            recurrent_output = recurrent_inference(root_latent_state, action_input)
            if action_shape_mode == "none":
                action_shape_mode = "flat"
        except Exception:
            recurrent_exception_fallback_count += 1
            action_input = _recurrent_action_input(
                np_module=np,
                torch=torch,
                actions=selected,
                device=device,
                mode="column",
            )
            recurrent_output = recurrent_inference(root_latent_state, action_input)
            action_shape_mode = "column"
    else:
        recurrent_output = recurrent_inference(root_latent_state, action_input)
    _finish_cuda_event_timing(recurrent_cuda_event)
    recurrent_enqueue_sec = _elapsed(recurrent_started)
    output_decode_started = time.perf_counter()
    next_latent_state = _network_output_field(
        recurrent_output,
        ("latent_state", "hidden_state"),
    )
    if next_latent_state is None:
        raise ValueError("CompactTorchSearchServiceV1 requires recurrent latent_state")
    reward = _policy_inverse_scalar_value(
        policy=policy,
        value=_network_output_field(recurrent_output, ("reward", "value_prefix")),
        torch=torch,
        root_count=int(selected.reshape(-1).shape[0]),
        device=device,
    )
    value = _policy_inverse_scalar_value(
        policy=policy,
        value=_network_output_field(recurrent_output, ("value", "predicted_value")),
        torch=torch,
        root_count=int(selected.reshape(-1).shape[0]),
        device=device,
    )
    recurrent_logits = _network_output_field(
        recurrent_output,
        ("policy_logits", "policy"),
    )
    if recurrent_logits is None:
        raise ValueError("CompactTorchSearchServiceV1 requires recurrent policy_logits")
    output_decode_sec = _elapsed(output_decode_started)
    del next_latent_state, recurrent_logits
    policy_build_started = time.perf_counter()
    selected_flat = selected.reshape(-1).long()
    root_count = int(selected_flat.shape[0])
    raw_counts = torch.zeros(
        (root_count, ACTION_COUNT),
        dtype=torch.float32,
        device=device,
    )
    row_index = torch.arange(root_count, dtype=torch.long, device=device)
    raw_counts[row_index, selected_flat] = 1.0
    visit_policy = raw_counts
    root_values = reward + float(discount_factor) * value
    policy_build_sec = _elapsed(policy_build_started)
    _finish_cuda_event_timing(replay_cuda_event)
    sync_sec = _timed_profile_sync_torch_device_if_cuda(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
        phase="deferred_one_simulation_replay",
    )
    total_sec = _elapsed(started)
    replay_cuda_event_sec, replay_cuda_event_status = _cuda_event_elapsed_sec(
        replay_cuda_event
    )
    recurrent_cuda_event_sec, recurrent_cuda_event_status = _cuda_event_elapsed_total_sec(
        [recurrent_cuda_event]
    )
    accounted_sec = (
        action_build_sec
        + recurrent_enqueue_sec
        + output_decode_sec
        + policy_build_sec
        + sync_sec
    )
    residual_sec = total_sec - accounted_sec
    telemetry = {
        "compact_torch_search_one_simulation_replay_materialization_deferred": True,
        "compact_torch_search_one_simulation_replay_materialized_on_flush": True,
        "compact_torch_search_deferred_one_simulation_replay_flush_sec": float(total_sec),
        "compact_torch_search_deferred_one_simulation_replay_action_build_sec": float(
            action_build_sec
        ),
        "compact_torch_search_deferred_one_simulation_replay_recurrent_inference_enqueue_sec": float(
            recurrent_enqueue_sec
        ),
        "compact_torch_search_deferred_one_simulation_replay_recurrent_inference_cuda_event_sec": float(
            recurrent_cuda_event_sec
        ),
        "compact_torch_search_deferred_one_simulation_replay_recurrent_inference_cuda_event_status": str(
            recurrent_cuda_event_status
        ),
        "compact_torch_search_deferred_one_simulation_replay_output_decode_sec": float(
            output_decode_sec
        ),
        "compact_torch_search_deferred_one_simulation_replay_policy_build_sec": float(
            policy_build_sec
        ),
        "compact_torch_search_deferred_one_simulation_replay_sync_sec": float(sync_sec),
        "compact_torch_search_deferred_one_simulation_replay_cuda_event_sec": float(
            replay_cuda_event_sec
        ),
        "compact_torch_search_deferred_one_simulation_replay_cuda_event_status": str(
            replay_cuda_event_status
        ),
        "compact_torch_search_deferred_one_simulation_replay_accounted_sec": float(
            accounted_sec
        ),
        "compact_torch_search_deferred_one_simulation_replay_residual_sec": float(
            residual_sec
        ),
        "compact_torch_search_deferred_one_simulation_replay_unaccounted_sec": float(
            max(0.0, residual_sec)
        ),
        "compact_torch_search_deferred_one_simulation_replay_overaccounted_sec": float(
            max(0.0, -residual_sec)
        ),
        "compact_torch_search_deferred_one_simulation_replay_recurrent_action_shape_mode_effective": (
            "flat" if action_shape_mode == "none" else action_shape_mode
        ),
        "compact_torch_search_deferred_one_simulation_replay_recurrent_action_shape_exception_fallback_count": float(
            recurrent_exception_fallback_count
        ),
        "compact_torch_search_deferred_one_simulation_replay_recurrent_inference_calls": 1.0,
    }
    return root_values, visit_policy, raw_counts, telemetry


def _model_from_policy(policy: Any) -> Any:
    model = getattr(policy, "_model", policy)
    if model is None or not hasattr(model, "initial_inference"):
        raise ValueError("CompactTorchSearchServiceV1 requires policy._model")
    if not hasattr(model, "recurrent_inference"):
        raise ValueError("CompactTorchSearchServiceV1 requires model.recurrent_inference")
    return model


def _binary_mask(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return array.astype(np.bool_, copy=False)
    numeric = np.asarray(value, dtype=np.float32)
    if not bool(np.logical_or(numeric == 0.0, numeric == 1.0).all()):
        raise ValueError(f"{name} must be binary")
    return numeric.astype(np.bool_, copy=False)


def _torch_compile_available(torch_module: Any | None) -> bool:
    if torch_module is None:
        return False
    return hasattr(torch_module, "compile")


def _plain_telemetry_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_telemetry_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_telemetry_value(item) for item in value]
    return str(value)


def _resident_obs_reused_flag(resident_telemetry: Mapping[str, Any]) -> float:
    return 1.0 if bool(resident_telemetry.get("resident_observation_used", False)) else 0.0


def _sync_torch_device_if_cuda(*, torch: Any, device: Any) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        torch.cuda.synchronize(device)
    except Exception:
        return


def _timed_sync_torch_device_if_cuda(*, torch: Any, device: Any) -> float:
    started = time.perf_counter()
    _sync_torch_device_if_cuda(torch=torch, device=device)
    return _elapsed(started)


def _cuda_event_timing_enabled(timing_mode: str) -> bool:
    return str(timing_mode) in {
        COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC_CUDA_EVENT,
        COMPACT_TORCH_TIMING_MODE_CUDA_EVENT_FINAL_SYNC,
    }


def _phase_sync_enabled(timing_mode: str, phase: str) -> bool:
    if str(phase) == "tree":
        return True
    return str(timing_mode) in {
        COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC,
        COMPACT_TORCH_TIMING_MODE_HOST_PHASE_SYNC_CUDA_EVENT,
    }


def _timed_profile_sync_torch_device_if_cuda(
    *,
    torch: Any,
    device: Any,
    timing_mode: str,
    phase: str,
) -> float:
    if not _phase_sync_enabled(timing_mode, phase):
        return 0.0
    return _timed_sync_torch_device_if_cuda(torch=torch, device=device)


def _disabled_cuda_event_state() -> dict[str, Any]:
    return {"status": "disabled", "start": None, "end": None}


def _start_optional_cuda_event_timing(
    *,
    torch: Any | None,
    device: Any | None,
    timing_mode: str,
) -> dict[str, Any]:
    if torch is None or device is None:
        return _disabled_cuda_event_state()
    return _start_cuda_event_timing(
        torch=torch,
        device=device,
        timing_mode=timing_mode,
    )


def _start_cuda_event_timing(
    *,
    torch: Any,
    device: Any,
    timing_mode: str,
) -> dict[str, Any]:
    if not _cuda_event_timing_enabled(timing_mode):
        return _disabled_cuda_event_state()
    if not str(device).startswith("cuda"):
        return {"status": "non_cuda", "start": None, "end": None}
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    except Exception as exc:
        return {
            "status": "start_error",
            "start": None,
            "end": None,
            "error": str(exc),
        }
    return {"status": "recording", "start": start, "end": end}


def _finish_cuda_event_timing(event_state: dict[str, Any]) -> None:
    if event_state.get("status") != "recording":
        return
    end = event_state.get("end")
    try:
        end.record()
    except Exception as exc:
        event_state["status"] = "end_error"
        event_state["error"] = str(exc)
        return
    event_state["status"] = "recorded"


def _cuda_event_elapsed_sec(event_state: Mapping[str, Any]) -> tuple[float, str]:
    status = str(event_state.get("status", "disabled"))
    if status != "recorded":
        return 0.0, status
    start = event_state.get("start")
    end = event_state.get("end")
    try:
        elapsed_ms = float(start.elapsed_time(end))
    except Exception as exc:
        return 0.0, f"elapsed_error:{type(exc).__name__}"
    return max(0.0, elapsed_ms / 1000.0), "recorded"


def _cuda_event_elapsed_total_sec(
    event_states: list[dict[str, Any]],
) -> tuple[float, str]:
    if not event_states:
        return 0.0, "disabled"
    total = 0.0
    statuses: list[str] = []
    for state in event_states:
        elapsed_sec, status = _cuda_event_elapsed_sec(state)
        total += elapsed_sec
        statuses.append(status)
    unique_statuses = sorted(set(statuses))
    if len(unique_statuses) == 1:
        return total, unique_statuses[0]
    return total, "mixed:" + ",".join(unique_statuses)


def _initial_inference_split_cuda_event_residual(
    *,
    outer_sec: float,
    outer_status: str,
    runtime_telemetry: Mapping[str, Any],
) -> tuple[float, str]:
    split_sec = float(
        runtime_telemetry.get(
            "compact_torch_search_service_initial_inference_direct_core_cuda_event_sec",
            0.0,
        )
    )
    split_status = str(
        runtime_telemetry.get(
            "compact_torch_search_service_initial_inference_direct_core_cuda_event_status",
            "disabled",
        )
    )
    if str(outer_status) != "recorded" or split_status != "recorded":
        return 0.0, f"outer:{outer_status};split:{split_status}"
    return max(0.0, float(outer_sec) - split_sec), "recorded"


def _torch_devices_match(*, torch: Any, lhs: Any, rhs: Any) -> bool:
    left = torch.device(lhs)
    right = torch.device(rhs)
    if left.type != right.type:
        return False
    if left.type != "cuda":
        return True
    left_index = 0 if left.index is None else int(left.index)
    right_index = 0 if right.index is None else int(right.index)
    return left_index == right_index


def _network_output_field(output: Any, names: tuple[str, ...]) -> Any:
    if isinstance(output, Mapping):
        for name in names:
            if name in output:
                return output[name]
    for name in names:
        if hasattr(output, name):
            return getattr(output, name)
    return None


def _compact_torch_initial_output_fields(output: Any) -> tuple[Any, Any, Any, bool]:
    if isinstance(output, _CompactTorchDecodedInitialOutput):
        policy_logits = output.policy_logits
        root_value = output.value
        root_latent_state = output.latent_state
        direct_decoded = True
    else:
        policy_logits = _network_output_field(output, ("policy_logits", "policy"))
        root_value = _network_output_field(output, ("value", "predicted_value"))
        root_latent_state = _network_output_field(
            output,
            ("latent_state", "hidden_state"),
        )
        direct_decoded = False
    if policy_logits is None:
        raise ValueError("CompactTorchSearchServiceV1 requires root policy logits")
    if root_latent_state is None:
        raise ValueError("CompactTorchSearchServiceV1 requires root latent_state")
    if not _is_torch_like_tensor(root_latent_state):
        raise ValueError("CompactTorchSearchServiceV1 requires torch latent_state")
    return policy_logits, root_value, root_latent_state, direct_decoded


def _compact_torch_root_latent_for_recurrent(
    *,
    torch: Any,
    root_latent_state: Any,
) -> tuple[Any, dict[str, Any]]:
    before_contiguous = bool(root_latent_state.is_contiguous())
    before_channels_last = False
    if int(getattr(root_latent_state, "ndim", 0)) == 4:
        before_channels_last = bool(
            root_latent_state.is_contiguous(memory_format=torch.channels_last)
        )
    prepared = root_latent_state.contiguous()
    copied = prepared is not root_latent_state
    after_contiguous = bool(prepared.is_contiguous())
    after_channels_last = False
    if int(getattr(prepared, "ndim", 0)) == 4:
        after_channels_last = bool(prepared.is_contiguous(memory_format=torch.channels_last))
    telemetry = {
        "compact_torch_search_root_latent_dtype": str(getattr(root_latent_state, "dtype", "")),
        "compact_torch_search_root_latent_ndim": float(int(getattr(root_latent_state, "ndim", 0))),
        "compact_torch_search_root_latent_is_contiguous_before_recurrent": before_contiguous,
        "compact_torch_search_root_latent_is_channels_last_before_recurrent": before_channels_last,
        "compact_torch_search_root_latent_contiguous_for_recurrent": after_contiguous,
        "compact_torch_search_root_latent_is_channels_last_for_recurrent": after_channels_last,
        "compact_torch_search_root_latent_contiguous_copy_bytes": (
            float(_tensor_nbytes(prepared)) if copied else 0.0
        ),
    }
    return prepared, telemetry


def _is_torch_like_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "cpu")


def _policy_inverse_scalar_value(
    *,
    policy: Any,
    value: Any,
    torch: Any,
    root_count: int,
    device: Any,
) -> Any:
    if value is None:
        return torch.zeros((int(root_count),), dtype=torch.float32, device=device)
    transformed = value
    inverse = getattr(policy, "inverse_scalar_transform_handle", None)
    if inverse is not None:
        try:
            transformed = inverse(value)
        except Exception:
            transformed = value
    if _is_torch_like_tensor(transformed):
        return transformed.reshape(int(root_count), -1)[:, 0].float().to(device=device)
    try:
        return torch.as_tensor(transformed, dtype=torch.float32, device=device).reshape(
            int(root_count),
            -1,
        )[:, 0]
    except Exception:
        return torch.zeros((int(root_count),), dtype=torch.float32, device=device)


def _recurrent_action_input(
    *,
    np_module: Any,
    torch: Any,
    actions: Any,
    device: Any,
    mode: str,
) -> Any:
    if _is_torch_like_tensor(actions):
        action_tensor = actions.long()
        if mode == "column":
            return action_tensor.reshape(-1, 1)
        return action_tensor.reshape(-1)
    action_np = np_module.asarray(actions, dtype=np.int64)
    if mode == "column":
        action_np = action_np.reshape(-1, 1)
    else:
        action_np = action_np.reshape(-1)
    try:
        return torch.as_tensor(action_np, device=device)
    except Exception:
        return action_np


def _array_to_numpy(value: Any) -> np.ndarray:
    if _is_torch_like_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _selected_actions_for_readback(value: Any, *, torch_module: Any) -> Any:
    if _is_torch_like_tensor(value):
        return value.to(dtype=torch_module.int16)
    return np.asarray(value, dtype=np.int16)


def _device_payload_tensor(
    *,
    torch: Any,
    value: Any,
    device: Any,
    dtype: Any,
    shape: tuple[int, ...],
    name: str,
) -> Any:
    if _is_torch_like_tensor(value):
        tensor = value.detach().to(device=device, dtype=dtype, non_blocking=True)
        is_inference = getattr(tensor, "is_inference", None)
        if callable(is_inference) and bool(is_inference()):
            tensor = tensor.clone()
    else:
        array = np.asarray(value)
        if array.size != 0:
            raise ReplayCompatibilityError(f"device replay payload requires tensor {name}")
        tensor = torch.empty(shape, dtype=dtype, device=device)
    if tuple(int(dim) for dim in tensor.shape) != tuple(int(dim) for dim in shape):
        raise ReplayCompatibilityError(f"device replay payload {name} shape mismatch")
    return tensor.contiguous()


def _tensor_nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel() * element_size())
    return int(np.asarray(value).nbytes)


__all__ = [
    "COMPACT_TORCH_SEARCH_BACKEND_KIND",
    "COMPACT_TORCH_SEARCH_HELPER",
    "COMPACT_TORCH_SEARCH_IMPL",
    "COMPACT_TORCH_SEARCH_LABEL",
    "COMPACT_TORCH_SEARCH_SEMANTICS",
    "COMPACT_TORCH_SEARCH_SERVICE_IMPL",
    "COMPACT_TORCH_MEMORY_FORMAT_CHANNELS_LAST",
    "COMPACT_TORCH_MEMORY_FORMAT_CONTIGUOUS",
    "COMPACT_TORCH_MEMORY_FORMATS",
    "COMPACT_TORCH_MODEL_COMPILE_MODES",
    "COMPACT_TORCH_TIMING_MODES",
    "CompactTorchCompileConfig",
    "CompactTorchCompileEligibility",
    "CompactTorchFixedShapeMasks",
    "CompactTorchSearchServiceV1",
    "compact_torch_compile_eligibility",
    "compact_torch_fixed_shape_masks",
    "make_compact_torch_expand_and_backup_fixed",
    "make_compact_torch_select_leaf_fixed",
]
