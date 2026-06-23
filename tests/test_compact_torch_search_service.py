from __future__ import annotations

import numpy as np
import pytest

import curvyzero.training.compact_torch_search_service as compact_torch_module
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_model_state_digest_v1,
)
from curvyzero.training.compact_policy_row_bridge import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import ResidentObservationBatchV1
from curvyzero.training.compact_search_service import (
    compact_search_result_v1_from_two_phase_payloads,
)
from curvyzero.training.compact_search_service import (
    validate_compact_search_two_phase_payload_v1,
)
from curvyzero.training.compact_search_service import (
    validate_compact_device_search_two_phase_payload_v1,
)
from curvyzero.training.compact_torch_search_service import (
    COMPACT_TORCH_SEARCH_BACKEND_KIND,
    COMPACT_TORCH_SEARCH_HELPER,
    COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
    COMPACT_TORCH_INITIAL_INFERENCE_MODE_MODEL_METHOD,
    COMPACT_TORCH_SEARCH_SEMANTICS,
    COMPACT_TORCH_SEARCH_SERVICE_IMPL,
    CompactTorchCompileConfig,
    CompactTorchSearchServiceV1,
    compact_torch_compile_eligibility,
    compact_torch_fixed_shape_masks,
    make_compact_torch_expand_and_backup_fixed,
    make_compact_torch_select_leaf_fixed,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


def _torch_module():
    return pytest.importorskip("torch")


def _root_batch(
    *,
    legal_mask: np.ndarray,
    active_root_mask: np.ndarray,
    observation_shape: tuple[int, ...] | None = None,
    observation_dtype: np.dtype | type[np.generic] = np.uint8,
    observation_source: str = "host_array_v1",
    resident_observation: ResidentObservationBatchV1 | None = None,
) -> CompactRootBatchV1:
    root_count = int(legal_mask.shape[0])
    observation_shape = observation_shape or (root_count, 4, 8, 8)
    return CompactRootBatchV1(
        observation=np.zeros(observation_shape, dtype=observation_dtype),
        legal_mask=np.asarray(legal_mask),
        active_root_mask=np.asarray(active_root_mask),
        to_play=np.full((root_count,), -1, dtype=np.int64),
        env_row=np.arange(root_count, dtype=np.int32),
        player=np.zeros((root_count,), dtype=np.int16),
        policy_env_id=np.arange(root_count, dtype=np.int64),
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
        resident_observation=resident_observation,
        observation_source=observation_source,
    )


def _core_capable_policy(torch, *, fail_representation: bool = False):
    class FakeOutput:
        def __init__(self, logits: torch.Tensor, value: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = value
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class CoreCapableFakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))
            self.initial_calls = 0
            self.representation_calls = 0
            self.prediction_calls = 0
            self.recurrent_calls = 0
            self.core_training_seen: list[bool] = []
            self.core_inference_mode_seen: list[bool] = []
            self.core_grad_enabled_seen: list[bool] = []
            self.representation_input_signatures: list[dict[str, object]] = []
            self.to_memory_formats: list[str] = []

        def to(self, *args, **kwargs):
            memory_format = kwargs.get("memory_format")
            if memory_format is not None:
                self.to_memory_formats.append(str(memory_format))
            return super().to(*args, **kwargs)

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            latent = self._representation(obs_tensor)
            policy_logits, value = self._prediction(latent)
            return FakeOutput(policy_logits, value, latent)

        def _representation(self, obs_tensor):
            if fail_representation:
                raise RuntimeError("representation failed")
            self.representation_calls += 1
            self.core_training_seen.append(bool(self.training))
            self.core_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.core_grad_enabled_seen.append(torch.is_grad_enabled())
            self.representation_input_signatures.append(
                {
                    "shape": tuple(int(dim) for dim in obs_tensor.shape),
                    "dtype": str(obs_tensor.dtype),
                    "device": str(obs_tensor.device),
                    "is_contiguous": bool(obs_tensor.is_contiguous()),
                    "is_channels_last": (
                        bool(obs_tensor.is_contiguous(memory_format=torch.channels_last))
                        if int(obs_tensor.ndim) == 4
                        else False
                    ),
                }
            )
            flattened = obs_tensor.float().reshape(int(obs_tensor.shape[0]), -1)
            signal = flattened.mean(dim=1)
            return torch.stack((signal + self.bias, signal * 0.0 + self.bias), dim=1)

        def _prediction(self, latent_state):
            self.prediction_calls += 1
            self.core_training_seen.append(bool(self.training))
            self.core_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.core_grad_enabled_seen.append(torch.is_grad_enabled())
            policy_logits = torch.stack(
                (
                    latent_state[:, 0] + 0.1,
                    latent_state[:, 0] + 3.0,
                    latent_state[:, 0] + 0.5,
                ),
                dim=1,
            )
            value = latent_state[:, :1] + 0.25
            return policy_logits, value

        def recurrent_inference(self, latent_state, _actions):
            self.recurrent_calls += 1
            batch = int(latent_state.shape[0])
            logits = torch.zeros((batch, 3), dtype=torch.float32)
            value = latent_state[:, :1] + 1.0
            return FakeOutput(logits, value, latent_state + 1.0)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = CoreCapableFakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    return FakePolicy()


def test_compile_eligibility_reports_profile_only_not_lightzero_ctree_labels() -> None:
    torch_module = _torch_module()
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )

    eligibility = compact_torch_compile_eligibility(
        root_batch,
        device="cpu",
        root_noise_weight=0.0,
        torch_module=torch_module,
    )

    assert eligibility.eligible is False
    assert eligibility.status == "fallback_precondition"
    assert eligibility.reason == "requires_cuda_device"
    telemetry = eligibility.telemetry
    assert telemetry["compact_torch_search_profile_only"] is True
    assert telemetry["compact_torch_search_not_lightzero_ctree"] is True
    assert telemetry["compact_torch_search_backend_kind"] == COMPACT_TORCH_SEARCH_BACKEND_KIND
    assert telemetry["compact_torch_search_backend_kind"] == "not_lightzero_ctree"
    assert telemetry["compact_torch_search_helper"] == COMPACT_TORCH_SEARCH_HELPER
    assert telemetry["compact_torch_search_trainer_ready"] is False
    assert telemetry["compact_torch_search_compile_attempted"] == 0.0


def test_compile_signature_changes_for_batch_shape_dtype_and_device() -> None:
    torch_module = _torch_module()
    config = CompactTorchCompileConfig(
        require_cuda_device=False,
        require_torch_compile=False,
        require_all_roots_active=False,
        require_all_actions_legal=False,
    )

    def signature(
        *,
        legal_mask: np.ndarray,
        active_root_mask: np.ndarray,
        observation_shape: tuple[int, ...] | None = None,
        observation_dtype: np.dtype | type[np.generic] = np.uint8,
        device: str = "cpu",
    ) -> list[object]:
        root_batch = _root_batch(
            legal_mask=legal_mask,
            active_root_mask=active_root_mask,
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
        )
        eligibility = compact_torch_compile_eligibility(
            root_batch,
            device=device,
            root_noise_weight=0.0,
            config=config,
            torch_module=torch_module,
        )
        return eligibility.telemetry["compact_torch_search_compile_signature"]

    baseline = signature(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )

    assert baseline == [2, 2, [2, 4, 8, 8], "uint8", "cpu", 3]
    assert (
        signature(
            legal_mask=np.ones((3, 3), dtype=np.bool_),
            active_root_mask=np.ones((3,), dtype=np.bool_),
        )
        != baseline
    )
    assert (
        signature(
            legal_mask=np.ones((2, 3), dtype=np.bool_),
            active_root_mask=np.asarray([True, False], dtype=np.bool_),
        )
        != baseline
    )
    assert (
        signature(
            legal_mask=np.ones((2, 3), dtype=np.bool_),
            active_root_mask=np.ones((2,), dtype=np.bool_),
            observation_shape=(2, 2, 8, 8),
        )
        != baseline
    )
    assert (
        signature(
            legal_mask=np.ones((2, 3), dtype=np.bool_),
            active_root_mask=np.ones((2,), dtype=np.bool_),
            observation_dtype=np.float32,
        )
        != baseline
    )
    assert (
        signature(
            legal_mask=np.ones((2, 3), dtype=np.bool_),
            active_root_mask=np.ones((2,), dtype=np.bool_),
            device="cuda:0",
        )
        != baseline
    )


@pytest.mark.parametrize(
    ("legal_mask", "active_root_mask", "match"),
    [
        (
            np.asarray([[1.0, 0.5, 1.0]], dtype=np.float32),
            np.asarray([1], dtype=np.int8),
            "legal_mask must be binary",
        ),
        (
            np.asarray([[1, 0, 1]], dtype=np.int8),
            np.asarray([2], dtype=np.int8),
            "active_root_mask must be binary",
        ),
    ],
)
def test_fixed_shape_masks_reject_non_binary_masks(
    legal_mask: np.ndarray,
    active_root_mask: np.ndarray,
    match: str,
) -> None:
    root_batch = _root_batch(
        legal_mask=legal_mask,
        active_root_mask=active_root_mask,
    )

    with pytest.raises(ValueError, match=match):
        compact_torch_fixed_shape_masks(root_batch)


def test_inactive_roots_do_not_make_all_actions_illegal_unless_forced_active() -> None:
    torch_module = _torch_module()
    root_batch = _root_batch(
        legal_mask=np.asarray(
            [
                [True, True, True],
                [True, False, True],
            ],
            dtype=np.bool_,
        ),
        active_root_mask=np.asarray([True, False], dtype=np.bool_),
    )

    masks = compact_torch_fixed_shape_masks(root_batch)
    assert masks.all_roots_active is False
    assert masks.all_actions_legal is True

    forced_masks = compact_torch_fixed_shape_masks(root_batch, force_all_roots_active=True)
    assert forced_masks.all_roots_active is True
    assert forced_masks.all_actions_legal is False

    config = CompactTorchCompileConfig(
        require_cuda_device=False,
        require_torch_compile=False,
        require_all_roots_active=False,
    )
    inactive_ignored = compact_torch_compile_eligibility(
        root_batch,
        device="cpu",
        root_noise_weight=0.0,
        config=config,
        torch_module=torch_module,
    )
    assert inactive_ignored.eligible is True
    assert inactive_ignored.reason == "preconditions_satisfied"

    forced_active = compact_torch_compile_eligibility(
        root_batch,
        device="cpu",
        root_noise_weight=0.0,
        config=config,
        torch_module=torch_module,
        force_all_roots_active=True,
    )
    assert forced_active.eligible is False
    assert forced_active.reason == "requires_all_actions_legal"


def test_compact_torch_search_service_refresh_stamps_action_and_replay_metadata(
    monkeypatch,
) -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class RefreshableModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            logits = torch.stack(
                (
                    torch.ones((batch,), dtype=torch.float32) + self.bias,
                    torch.zeros((batch,), dtype=torch.float32),
                    torch.zeros((batch,), dtype=torch.float32),
                ),
                dim=1,
            )
            latent = torch.zeros((batch, 2), dtype=torch.float32) + self.bias
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                latent_state + 1.0,
            )

    class FakePolicy:
        def __init__(self, model) -> None:
            self._model = model
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    learner_model = RefreshableModel()
    with torch.no_grad():
        learner_model.bias.fill_(2.0)
    learner_digest = compact_model_state_digest_v1(learner_model)
    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(RefreshableModel()),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )
    service._compiled_helper_cache[("unit",)] = ("stale-select", "stale-backup")
    service._compiled_model_cache[("unit",)] = (
        "stale-initial",
        "stale-recurrent",
    )

    state = service.refresh_model_state(
        model_state_dict=learner_model.state_dict(),
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
        policy_source="unit_test_policy_refresh",
        learner_update_count=1,
        expected_model_state_digest=learner_digest,
    )

    def fail_digest(_model):
        raise AssertionError("policy-refresh digest should be cached")

    monkeypatch.setattr(
        compact_torch_module,
        "compact_model_state_digest_v1",
        fail_digest,
    )
    root_batch = _root_batch(
        legal_mask=np.ones((1, 3), dtype=np.bool_),
        active_root_mask=np.ones((1,), dtype=np.bool_),
        observation_shape=(1, 4, 8, 8),
    )

    action_step = service.run_action_step(root_batch)
    replay_payload = service.flush_replay_payload(action_step.replay_payload_handle)

    assert state["model_state_digest"] == learner_digest
    assert state["cache_cleared"] is True
    assert state["refresh_total_sec"] >= 0.0
    assert state["state_load_sec"] >= 0.0
    assert state["model_state_digest_sec"] >= 0.0
    assert state["total_state_load_sec"] >= state["state_load_sec"]
    assert state["total_model_state_digest_sec"] >= state["model_state_digest_sec"]
    assert state["model_state_digest_source"] == "search_worker_after_load"
    assert service._compiled_helper_cache == {}
    assert service._compiled_model_cache == {}
    for metadata in (action_step.metadata, replay_payload.metadata):
        assert metadata["policy_version_ref"] == "unit-policy:update-1"
        assert metadata["model_version_ref"] == "unit-model:update-1"
        assert metadata["policy_source"] == "unit_test_policy_refresh"
        assert metadata["compact_policy_refresh_model_state_digest"] == learner_digest
        assert metadata["compact_policy_refresh_learner_update_count"] == 1
        assert metadata["compact_policy_refresh_search_worker_refreshed"] is True
        assert metadata["compact_policy_refresh_count"] == 1


def test_compact_torch_search_service_shared_refresh_uses_version_token(
    monkeypatch,
) -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class SharedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(()))

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            logits = torch.stack(
                (
                    torch.ones((batch,), dtype=torch.float32) + self.bias,
                    torch.zeros((batch,), dtype=torch.float32),
                    torch.zeros((batch,), dtype=torch.float32),
                ),
                dim=1,
            )
            latent = torch.zeros((batch, 2), dtype=torch.float32) + self.bias
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                latent_state + 1.0,
            )

    class FakePolicy:
        def __init__(self, model) -> None:
            self._model = model
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(SharedModel()),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )
    service._compiled_helper_cache[("unit",)] = ("stale-select", "stale-backup")
    service._compiled_model_cache[("unit",)] = (
        "stale-initial",
        "stale-recurrent",
    )
    token = "shared-model-state:unit:1"

    state = service.refresh_shared_model_state(
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
        policy_source="unit_test_shared_refresh",
        learner_update_count=1,
        model_state_digest=token,
    )

    def fail_digest(_model):
        raise AssertionError("shared refresh should use the version token")

    monkeypatch.setattr(
        compact_torch_module,
        "compact_model_state_digest_v1",
        fail_digest,
    )
    root_batch = _root_batch(
        legal_mask=np.ones((1, 3), dtype=np.bool_),
        active_root_mask=np.ones((1,), dtype=np.bool_),
        observation_shape=(1, 4, 8, 8),
    )

    action_step = service.run_action_step(root_batch)
    replay_payload = service.flush_replay_payload(action_step.replay_payload_handle)

    assert state["shared_model_state_refreshed"] is True
    assert state["model_state_digest"] == token
    assert state["model_state_digest_source"] == "shared_model_version_token"
    assert state["state_load_sec"] == 0.0
    assert state["model_state_digest_sec"] == 0.0
    assert state["total_state_load_sec"] == 0.0
    assert state["total_model_state_digest_sec"] == 0.0
    assert state["cache_cleared"] is True
    assert service._compiled_helper_cache == {}
    assert service._compiled_model_cache == {}
    for metadata in (action_step.metadata, replay_payload.metadata):
        assert metadata["policy_version_ref"] == "unit-policy:update-1"
        assert metadata["model_version_ref"] == "unit-model:update-1"
        assert metadata["policy_source"] == "unit_test_shared_refresh"
        assert metadata["compact_policy_refresh_model_state_digest"] == token
        assert (
            metadata["compact_policy_refresh_model_state_digest_source"]
            == "shared_model_version_token"
        )
        assert metadata["compact_policy_refresh_learner_update_count"] == 1
        assert metadata["compact_policy_refresh_search_worker_refreshed"] is True
        assert metadata["compact_policy_refresh_count"] == 1

    with pytest.raises(ValueError, match="model_state_digest must be non-empty"):
        service.refresh_shared_model_state(
            policy_version_ref="unit-policy:update-2",
            model_version_ref="unit-model:update-2",
            policy_source="unit_test_shared_refresh",
            learner_update_count=2,
            model_state_digest="",
        )


def test_compact_torch_search_service_preserves_active_root_order_and_legal_masks() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.initial_calls = 0
            self.recurrent_calls = 0

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            batch = int(obs_tensor.shape[0])
            logits = torch.tensor(
                [
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 4.0],
                ],
                dtype=torch.float32,
            )[:batch]
            latent = torch.zeros((batch, 2), dtype=torch.float32)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            self.recurrent_calls += 1
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state + 1.0)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    policy = FakePolicy()
    root_batch = _root_batch(
        legal_mask=np.asarray(
            [
                [True, True, True],
                [False, True, False],
                [True, True, True],
            ],
            dtype=np.bool_,
        ),
        active_root_mask=np.asarray([False, True, True], dtype=np.bool_),
        observation_shape=(3, 4, 8, 8),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )

    result = service.run(root_batch)

    assert policy._model.initial_calls == 1
    assert policy._model.recurrent_calls == 1
    assert result.metadata["search_impl"] == COMPACT_TORCH_SEARCH_SERVICE_IMPL
    assert result.metadata["profile_semantics"] == COMPACT_TORCH_SEARCH_SEMANTICS
    assert result.metadata["compact_torch_search_semantics"] == (COMPACT_TORCH_SEARCH_SEMANTICS)
    assert result.metadata["compact_torch_search_service_profile_only"] is True
    assert result.metadata["compact_torch_search_service_obs_h2d_bytes"] == float(
        root_batch.observation[root_batch.active_root_mask].nbytes
    )
    assert result.metadata["compact_torch_search_service_mask_h2d_bytes"] == float(
        root_batch.legal_mask[root_batch.active_root_mask].nbytes
    )
    assert result.metadata["compact_torch_search_service_action_d2h_bytes"] == float(
        result.selected_action.nbytes
    )
    assert result.metadata["compact_torch_search_service_replay_payload_d2h_bytes"] == float(
        result.visit_policy.nbytes + result.root_value.nbytes + result.raw_visit_counts.nbytes
    )
    assert result.metadata["compact_torch_search_service_root_observation_copy_bytes"] == float(0.0)
    assert result.metadata["compact_torch_search_service_root_mask_copy_bytes"] == 0.0
    assert result.metadata["compact_torch_search_service_python_rows_materialized"] == 0.0
    assert result.metadata["compact_torch_search_one_simulation_fast_path"] is True
    assert result.metadata["compact_torch_search_recurrent_inference_calls"] == 1.0
    assert result.metadata["compact_torch_search_service_timing_mode"] == "host_phase_sync"
    assert result.metadata["compact_torch_search_service_cuda_event_timing_enabled"] is False
    assert result.metadata["compact_torch_search_service_initial_sync_enabled"] is True
    assert (
        result.metadata["compact_torch_search_service_initial_inference_cuda_event_status"]
        == "disabled"
    )
    for key in (
        "compact_torch_search_service_tensor_prepare_sync_sec",
        "compact_torch_search_service_initial_inference_enqueue_sec",
        "compact_torch_search_service_initial_inference_sync_sec",
        "compact_torch_search_service_initial_inference_cuda_event_sec",
        "compact_torch_search_service_initial_inference_representation_cuda_event_sec",
        "compact_torch_search_service_initial_inference_prediction_cuda_event_sec",
        "compact_torch_search_service_initial_inference_direct_core_cuda_event_sec",
        "compact_torch_search_service_initial_inference_direct_core_cuda_event_residual_sec",
        "compact_torch_search_service_tree_root_prior_select_sec",
        "compact_torch_search_service_tree_recurrent_action_build_sec",
        "compact_torch_search_service_tree_recurrent_inference_enqueue_sec",
        "compact_torch_search_service_tree_recurrent_inference_cuda_event_sec",
        "compact_torch_search_service_tree_recurrent_output_decode_sec",
        "compact_torch_search_service_tree_policy_build_sec",
        "compact_torch_search_service_tree_sync_sec",
        "compact_torch_search_service_tree_cuda_event_sec",
        "compact_torch_search_service_tree_unaccounted_sec",
    ):
        assert key in result.metadata
        assert result.metadata[key] >= 0.0
    assert (
        result.metadata["compact_torch_search_service_tree_recurrent_inference_cuda_event_status"]
        == "disabled"
    )
    assert result.metadata["compact_torch_search_service_tree_cuda_event_status"] == "disabled"
    assert (
        result.metadata[
            "compact_torch_search_service_initial_inference_representation_cuda_event_status"
        ]
        == "disabled"
    )
    assert (
        result.metadata[
            "compact_torch_search_service_initial_inference_prediction_cuda_event_status"
        ]
        == "disabled"
    )
    np.testing.assert_array_equal(result.root_index, np.asarray([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(result.env_row, np.asarray([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(result.selected_action, np.asarray([1, 2], dtype=np.int16))
    np.testing.assert_allclose(result.visit_policy.sum(axis=1), 1.0)
    assert result.visit_policy[0, 0] == pytest.approx(0.0)
    assert result.visit_policy[0, 2] == pytest.approx(0.0)


def test_one_simulation_zero_noise_selects_from_logits_without_softmax(
    monkeypatch,
) -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            logits = torch.tensor(
                [
                    [0.0, 4.0, 2.0],
                    [5.0, 1.0, 3.0],
                ],
                dtype=torch.float32,
            )[:batch]
            latent = torch.zeros((batch, 2), dtype=torch.float32)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    def fail_softmax(*_args, **_kwargs):
        raise AssertionError("zero-noise one-simulation path should skip softmax")

    monkeypatch.setattr(torch, "softmax", fail_softmax)
    root_batch = _root_batch(
        legal_mask=np.asarray(
            [
                [True, True, True],
                [False, True, True],
            ],
            dtype=np.bool_,
        ),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    policy = FakePolicy()
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )

    action_step = service.run_action_step(root_batch)

    np.testing.assert_array_equal(
        action_step.selected_action,
        np.asarray([1, 2], dtype=np.int16),
    )
    telemetry = action_step.metadata["profile_telemetry"]
    assert telemetry["compact_torch_search_one_simulation_fast_path"] is True
    assert telemetry["compact_torch_search_one_simulation_root_prior_softmax_skipped"] is True
    assert telemetry["compact_torch_search_one_simulation_selection_mode"] == "masked_logits_argmax"
    assert (
        telemetry["compact_torch_search_service_tree_root_prior_build_sec"]
        == (telemetry["compact_torch_search_service_tree_root_prior_select_sec"])
    )


def test_one_simulation_deferred_replay_payload_moves_recurrent_to_flush() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(
            self,
            logits: torch.Tensor,
            latent: torch.Tensor,
            *,
            reward: torch.Tensor | None = None,
            value: torch.Tensor | None = None,
        ) -> None:
            self.policy_logits = logits
            self.value = (
                value
                if value is not None
                else torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            )
            self.reward = (
                reward
                if reward is not None
                else torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            )
            self.latent_state = latent

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.value_head = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.value_head.weight.fill_(1.0)
            self.recurrent_calls = 0
            self.recurrent_grad_enabled_seen: list[bool] = []
            self.recurrent_inference_mode_seen: list[bool] = []

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            logits = torch.tensor(
                [
                    [0.0, 4.0, 2.0],
                    [5.0, 1.0, 3.0],
                ],
                dtype=torch.float32,
            )[:batch]
            latent = torch.arange(batch * 2, dtype=torch.float32).reshape(batch, 2)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, actions):
            self.recurrent_calls += 1
            self.recurrent_grad_enabled_seen.append(bool(torch.is_grad_enabled()))
            self.recurrent_inference_mode_seen.append(
                bool(torch.is_inference_mode_enabled())
            )
            flat_actions = actions.reshape(-1).to(dtype=torch.float32)
            logits = torch.zeros((int(flat_actions.shape[0]), 3), dtype=torch.float32)
            reward = (flat_actions + 1.0).reshape(-1, 1)
            value = self.value_head(latent_state) + 0.5
            return FakeOutput(logits, latent_state, reward=reward, value=value)

    class FakePolicy:
        def __init__(self, model: FakeModel) -> None:
            self._model = model
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.5,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    immediate_model = FakeModel()
    immediate_service = CompactTorchSearchServiceV1(
        policy=FakePolicy(immediate_model),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )
    immediate_step = immediate_service.run_action_step(root_batch)
    assert immediate_model.recurrent_calls == 1
    immediate_payload = immediate_service.flush_replay_payload(
        immediate_step.replay_payload_handle
    )

    deferred_model = FakeModel()
    deferred_service = CompactTorchSearchServiceV1(
        policy=FakePolicy(deferred_model),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            defer_one_simulation_replay_payload=True,
        ),
    )
    deferred_step = deferred_service.run_action_step(root_batch)
    assert deferred_model.recurrent_calls == 0
    action_telemetry = deferred_step.metadata["profile_telemetry"]
    assert (
        action_telemetry[
            "compact_torch_search_service_defer_one_simulation_replay_payload_used"
        ]
        is True
    )
    assert action_telemetry["compact_torch_search_recurrent_inference_calls"] == 0.0
    assert (
        action_telemetry[
            "compact_torch_search_one_simulation_replay_materialization_deferred"
        ]
        is True
    )
    assert (
        action_telemetry[
            "compact_torch_search_deferred_one_simulation_action_model_state_digest"
        ]
    )
    assert (
        action_telemetry[
            "compact_torch_search_deferred_one_simulation_model_identity_match"
        ]
        is True
    )
    assert action_telemetry["compact_torch_search_pending_deferred_replay_payload_count"] == 1.0

    with pytest.raises(ReplayCompatibilityError, match="pending deferred one-simulation"):
        deferred_service.refresh_shared_model_state(
            policy_version_ref="policy:v1",
            model_version_ref="model:v1",
            policy_source="unit",
            learner_update_count=1,
            model_state_digest="digest",
        )
    with pytest.raises(ReplayCompatibilityError, match="pending deferred one-simulation"):
        deferred_service.refresh_model_state(
            model_state_dict=deferred_model.state_dict(),
            policy_version_ref="policy:v1",
            model_version_ref="model:v1",
            policy_source="unit",
            learner_update_count=1,
        )

    deferred_payload = deferred_service.flush_replay_payload(
        deferred_step.replay_payload_handle
    )
    assert deferred_model.recurrent_calls == 1
    assert deferred_model.recurrent_grad_enabled_seen == [False]
    assert deferred_model.recurrent_inference_mode_seen == [True]

    np.testing.assert_array_equal(deferred_step.selected_action, immediate_step.selected_action)
    np.testing.assert_allclose(deferred_payload.visit_policy, immediate_payload.visit_policy)
    np.testing.assert_allclose(deferred_payload.root_value, immediate_payload.root_value)
    np.testing.assert_allclose(deferred_payload.raw_visit_counts, immediate_payload.raw_visit_counts)
    flush_telemetry = deferred_payload.metadata["profile_telemetry"]
    assert (
        flush_telemetry[
            "compact_torch_search_one_simulation_replay_materialized_on_flush"
        ]
        is True
    )
    assert (
        flush_telemetry[
            "compact_torch_search_deferred_one_simulation_replay_recurrent_inference_calls"
        ]
        == 1.0
    )
    assert (
        flush_telemetry[
            "compact_torch_search_deferred_one_simulation_flush_model_state_digest"
        ]
        == action_telemetry[
            "compact_torch_search_deferred_one_simulation_action_model_state_digest"
        ]
    )
    assert (
        flush_telemetry[
            "compact_torch_search_deferred_one_simulation_model_identity_match"
        ]
        is True
    )
    assert (
        flush_telemetry[
            "compact_torch_search_deferred_one_simulation_model_refresh_crossed_count"
        ]
        == 0
    )
    assert flush_telemetry["compact_torch_search_pending_deferred_replay_payload_count"] == 1.0
    assert (
        flush_telemetry["compact_torch_search_pending_deferred_replay_payload_final_count"]
        == 0.0
    )

    device_deferred_model = FakeModel()
    device_deferred_service = CompactTorchSearchServiceV1(
        policy=FakePolicy(device_deferred_model),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            defer_one_simulation_replay_payload=True,
        ),
    )
    device_deferred_step = device_deferred_service.run_action_step(root_batch)
    device_deferred_payload = device_deferred_service.flush_device_replay_payload(
        device_deferred_step.replay_payload_handle
    )
    assert device_deferred_model.recurrent_calls == 1
    assert device_deferred_model.recurrent_grad_enabled_seen == [False]
    assert device_deferred_model.recurrent_inference_mode_seen == [True]
    for tensor in (
        device_deferred_payload.visit_policy,
        device_deferred_payload.root_value,
        device_deferred_payload.raw_visit_counts,
    ):
        is_inference = getattr(tensor, "is_inference", None)
        assert not (callable(is_inference) and bool(is_inference()))


def test_deferred_one_simulation_replay_flush_fails_after_model_identity_drift() -> None:
    torch = _torch_module()
    policy = _core_capable_policy(torch)
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            defer_one_simulation_replay_payload=True,
        ),
    )

    action_step = service.run_action_step(root_batch)
    service._policy_refresh_model_state_digest = "changed-after-action"

    with pytest.raises(ReplayCompatibilityError, match="crossed a model refresh"):
        service.flush_device_replay_payload(action_step.replay_payload_handle)


def test_action_step_reads_back_selected_actions_as_int16(monkeypatch) -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            logits = torch.tensor(
                [
                    [0.0, 4.0, 2.0],
                    [5.0, 1.0, 3.0],
                ],
                dtype=torch.float32,
            )[:batch]
            latent = torch.zeros((batch, 2), dtype=torch.float32)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    original_array_to_numpy = compact_torch_module._array_to_numpy
    selected_readback_dtypes: list[torch.dtype] = []

    def record_array_to_numpy(value):
        if torch.is_tensor(value):
            selected_readback_dtypes.append(value.dtype)
        return original_array_to_numpy(value)

    monkeypatch.setattr(
        compact_torch_module,
        "_array_to_numpy",
        record_array_to_numpy,
    )
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )

    action_step = service.run_action_step(root_batch)

    assert selected_readback_dtypes == [torch.int16]
    assert action_step.metadata["profile_telemetry"][
        "compact_torch_search_service_action_d2h_bytes"
    ] == float(action_step.selected_action.nbytes)
    np.testing.assert_array_equal(
        action_step.selected_action,
        np.asarray([1, 0], dtype=np.int16),
    )


def test_action_step_reuses_fixed_shape_masks_for_compile_eligibility(
    monkeypatch,
) -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            logits = torch.tensor(
                [
                    [0.0, 4.0, 2.0],
                    [5.0, 1.0, 3.0],
                ],
                dtype=torch.float32,
            )[:batch]
            latent = torch.zeros((batch, 2), dtype=torch.float32)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    original_masks = compact_torch_module.compact_torch_fixed_shape_masks
    mask_call_count = 0

    def count_masks(*args, **kwargs):
        nonlocal mask_call_count
        mask_call_count += 1
        return original_masks(*args, **kwargs)

    monkeypatch.setattr(
        compact_torch_module,
        "compact_torch_fixed_shape_masks",
        count_masks,
    )
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )

    action_step = service.run_action_step(root_batch)

    assert mask_call_count == 1
    telemetry = action_step.metadata["profile_telemetry"]
    assert telemetry["compact_torch_search_compile_runtime_status"] == "not_requested"
    assert telemetry["compact_torch_search_compile_status"] == "not_requested"


def test_compact_torch_search_service_two_phase_defers_replay_payload() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.initial_calls = 0
            self.recurrent_calls = 0

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            self.initial_calls += 1
            batch = int(obs_tensor.shape[0])
            logits = torch.tensor(
                [
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 4.0],
                ],
                dtype=torch.float32,
            )[:batch]
            latent = torch.zeros((batch, 2), dtype=torch.float32)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            self.recurrent_calls += 1
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    policy = FakePolicy()
    root_batch = _root_batch(
        legal_mask=np.asarray(
            [
                [True, True, True],
                [False, True, False],
                [True, True, True],
            ],
            dtype=np.bool_,
        ),
        active_root_mask=np.asarray([False, True, True], dtype=np.bool_),
        observation_shape=(3, 4, 8, 8),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=2,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )

    action_step = service.run_action_step(root_batch)

    assert policy._model.initial_calls == 1
    assert policy._model.recurrent_calls == 2
    assert action_step.metadata["search_impl"] == COMPACT_TORCH_SEARCH_SERVICE_IMPL
    assert action_step.metadata["profile_semantics"] == COMPACT_TORCH_SEARCH_SEMANTICS
    assert action_step.metadata["compact_torch_search_semantics"] == (
        COMPACT_TORCH_SEARCH_SEMANTICS
    )
    assert action_step.metadata["search_replay_payload_digest_deferred"] is True
    assert (
        action_step.metadata["profile_telemetry"][
            "compact_torch_search_service_replay_payload_d2h_bytes"
        ]
        == 0.0
    )
    assert (
        action_step.metadata["profile_telemetry"][
            "compact_torch_search_service_deferred_replay_payload_d2h_bytes"
        ]
        > 0.0
    )
    telemetry = action_step.metadata["profile_telemetry"]
    for key in (
        "compact_torch_search_service_action_preamble_sec",
        "compact_torch_search_service_fixed_shape_masks_sec",
        "compact_torch_search_service_compile_eligibility_sec",
        "compact_torch_search_service_helper_cache_sec",
        "compact_torch_search_service_model_cache_sec",
        "compact_torch_search_service_inference_guard_enter_sec",
        "compact_torch_search_service_inference_guard_exit_sec",
        "compact_torch_search_service_inference_guard_total_sec",
        "compact_torch_search_service_metadata_build_sec",
        "compact_torch_search_service_pending_replay_store_sec",
        "compact_torch_search_service_action_step_build_sec",
        "compact_torch_search_service_action_postprocess_sec",
        "compact_torch_search_service_action_wall_sec",
        "compact_torch_search_service_action_accounted_sec",
        "compact_torch_search_service_action_unaccounted_sec",
        "compact_torch_search_service_action_overaccounted_sec",
        "compact_torch_search_service_tensor_prepare_sync_sec",
        "compact_torch_search_service_initial_output_decode_sec",
        "compact_torch_search_service_root_output_decode_sec",
        "compact_torch_search_service_initial_inference_enqueue_sec",
        "compact_torch_search_service_initial_inference_sync_sec",
        "compact_torch_search_service_initial_inference_cuda_event_sec",
        "compact_torch_search_service_tree_setup_sec",
        "compact_torch_search_service_tree_root_prior_build_sec",
        "compact_torch_search_service_tree_root_prior_select_sec",
        "compact_torch_search_service_tree_select_enqueue_sec",
        "compact_torch_search_service_tree_recurrent_action_build_sec",
        "compact_torch_search_service_tree_recurrent_inference_enqueue_sec",
        "compact_torch_search_service_tree_recurrent_inference_cuda_event_sec",
        "compact_torch_search_service_tree_recurrent_output_decode_sec",
        "compact_torch_search_service_tree_backup_enqueue_sec",
        "compact_torch_search_service_tree_policy_build_sec",
        "compact_torch_search_service_tree_sync_sec",
        "compact_torch_search_service_tree_cuda_event_sec",
        "compact_torch_search_service_tree_total_sec",
        "compact_torch_search_service_tree_accounted_sec",
        "compact_torch_search_service_tree_unaccounted_sec",
        "compact_torch_search_service_tree_overaccounted_sec",
        "compact_torch_search_service_action_readback_sec",
        "compact_torch_search_service_core_accounted_sec",
        "compact_torch_search_service_core_unaccounted_sec",
        "compact_torch_search_service_core_overaccounted_sec",
    ):
        assert key in telemetry
        assert telemetry[key] >= 0.0
        assert action_step.metadata[key] == telemetry[key]
    for key in (
        "compact_torch_search_service_action_residual_sec",
        "compact_torch_search_service_tree_residual_sec",
        "compact_torch_search_service_core_residual_sec",
    ):
        assert key in telemetry
        assert action_step.metadata[key] == telemetry[key]
    assert (
        telemetry["compact_torch_search_service_initial_output_decode_sec"]
        == (telemetry["compact_torch_search_service_root_output_decode_sec"])
    )
    assert telemetry["compact_torch_search_service_inference_guard_total_sec"] == (
        telemetry["compact_torch_search_service_inference_guard_enter_sec"]
        + telemetry["compact_torch_search_service_inference_guard_exit_sec"]
    )
    assert (
        telemetry["compact_torch_search_service_action_readback_sec"]
        == (telemetry["compact_torch_search_service_readback_sec"])
    )
    assert telemetry["compact_torch_search_service_action_residual_sec"] == pytest.approx(
        telemetry["compact_torch_search_service_action_wall_sec"]
        - telemetry["compact_torch_search_service_action_accounted_sec"]
    )
    assert telemetry["compact_torch_search_service_timing_mode"] == "host_phase_sync"
    assert telemetry["compact_torch_search_service_cuda_event_timing_enabled"] is False
    assert telemetry["compact_torch_search_service_initial_sync_enabled"] is True
    assert (
        telemetry["compact_torch_search_service_initial_inference_cuda_event_status"] == "disabled"
    )
    assert (
        telemetry["compact_torch_search_service_tree_recurrent_inference_cuda_event_status"]
        == "disabled"
    )
    assert telemetry["compact_torch_search_service_tree_cuda_event_status"] == "disabled"
    assert (
        telemetry["compact_torch_search_service_action_wall_sec"]
        >= telemetry["compact_torch_search_service_total_sec"]
    )
    assert telemetry["compact_torch_search_one_simulation_fast_path"] is False
    assert telemetry["compact_torch_search_recurrent_inference_calls"] == 2.0
    assert (
        telemetry["compact_torch_search_service_action_postprocess_sec"]
        >= telemetry["compact_torch_search_service_action_step_build_sec"]
    )
    np.testing.assert_array_equal(action_step.root_index, np.asarray([1, 2], dtype=np.int32))
    np.testing.assert_array_equal(action_step.selected_action, np.asarray([1, 2], dtype=np.int16))

    payload = service.flush_replay_payload(action_step.replay_payload_handle)
    validate_compact_search_two_phase_payload_v1(action_step, payload)
    result = compact_search_result_v1_from_two_phase_payloads(
        root_batch,
        action_step,
        payload,
    )
    np.testing.assert_array_equal(result.selected_action, action_step.selected_action)
    np.testing.assert_allclose(result.visit_policy.sum(axis=1), 1.0)
    assert (
        payload.metadata["profile_telemetry"]["compact_torch_search_service_replay_payload_flushed"]
        is True
    )
    assert (
        payload.metadata["profile_telemetry"]["compact_torch_search_service_action_wall_sec"]
        == telemetry["compact_torch_search_service_action_wall_sec"]
    )

    with pytest.raises(ReplayCompatibilityError, match="already-flushed|unknown"):
        service.flush_replay_payload(action_step.replay_payload_handle)

    device_action_step = service.run_action_step(root_batch)
    device_payload = service.flush_device_replay_payload(device_action_step.replay_payload_handle)
    validate_compact_device_search_two_phase_payload_v1(
        device_action_step,
        device_payload,
    )
    assert isinstance(device_payload.visit_policy, torch.Tensor)
    assert isinstance(device_payload.root_value, torch.Tensor)
    assert isinstance(device_payload.raw_visit_counts, torch.Tensor)
    assert device_payload.metadata["device_replay_payload"] is True
    assert (
        device_payload.metadata["profile_telemetry"][
            "compact_torch_search_service_replay_payload_d2h_bytes"
        ]
        == 0.0
    )
    assert (
        device_payload.metadata["profile_telemetry"][
            "compact_torch_search_service_replay_payload_output"
        ]
        == "device_torch"
    )
    with pytest.raises(ReplayCompatibilityError, match="already-flushed|unknown"):
        service.flush_device_replay_payload(device_action_step.replay_payload_handle)


def test_compact_torch_search_service_host_final_sync_mode_disables_phase_sync() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            logits = torch.tensor(
                [
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 4.0],
                ],
                dtype=torch.float32,
            )[:batch]
            latent = torch.zeros((batch, 2), dtype=torch.float32)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=True,
            require_all_actions_legal=True,
            timing_mode="host_final_sync_only",
        ),
    )

    action_step = service.run_action_step(root_batch)
    telemetry = action_step.metadata["profile_telemetry"]

    assert telemetry["compact_torch_search_service_timing_mode"] == "host_final_sync_only"
    assert telemetry["compact_torch_search_service_cuda_event_timing_enabled"] is False
    assert telemetry["compact_torch_search_service_initial_sync_enabled"] is False
    assert telemetry["compact_torch_search_service_tensor_prepare_sync_sec"] == 0.0
    assert telemetry["compact_torch_search_service_initial_inference_sync_sec"] == 0.0
    assert telemetry["compact_torch_search_service_inference_guard_total_sec"] >= 0.0
    assert telemetry["compact_torch_search_service_tree_sync_sec"] >= 0.0
    np.testing.assert_array_equal(
        action_step.selected_action,
        np.asarray([1, 2], dtype=np.int16),
    )

    with pytest.raises(ValueError, match="timing_mode must be one of"):
        CompactTorchSearchServiceV1(
            policy=FakePolicy(),
            num_simulations=1,
            compile_config=CompactTorchCompileConfig(timing_mode="not_a_mode"),
        )
    with pytest.raises(ValueError, match="observation_memory_format must be one of"):
        CompactTorchSearchServiceV1(
            policy=FakePolicy(),
            num_simulations=1,
            compile_config=CompactTorchCompileConfig(observation_memory_format="nhwc_maybe"),
        )
    with pytest.raises(ValueError, match="model_memory_format must be one of"):
        CompactTorchSearchServiceV1(
            policy=FakePolicy(),
            num_simulations=1,
            compile_config=CompactTorchCompileConfig(model_memory_format="nhwc_maybe"),
        )
    with pytest.raises(ValueError, match="model_memory_format=channels_last is parked"):
        CompactTorchSearchServiceV1(
            policy=FakePolicy(),
            num_simulations=1,
            compile_config=CompactTorchCompileConfig(model_memory_format="channels_last"),
        )


def test_compact_torch_search_service_compiles_helpers_once_when_eligible(monkeypatch) -> None:
    torch = _torch_module()
    compile_calls: list[str] = []

    def fake_compile(fn, **_kwargs):
        compile_calls.append(fn.__name__)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.training = True
            self.initial_inference_mode_seen: list[bool] = []
            self.initial_grad_enabled_seen: list[bool] = []
            self.recurrent_inference_mode_seen: list[bool] = []
            self.recurrent_grad_enabled_seen: list[bool] = []

        def parameters(self):
            return iter([self.param])

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = bool(mode)
            return self

        def initial_inference(self, obs_tensor):
            assert self.training is False
            self.initial_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.initial_grad_enabled_seen.append(torch.is_grad_enabled())
            batch = int(obs_tensor.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.zeros((batch, 2), dtype=torch.float32),
            )

        def recurrent_inference(self, latent_state, _actions):
            assert self.training is False
            self.recurrent_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.recurrent_grad_enabled_seen.append(torch.is_grad_enabled())
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    policy = FakePolicy()
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=True,
            require_cuda_device=False,
            require_torch_compile=True,
            require_all_roots_active=True,
            require_all_actions_legal=True,
        ),
    )

    first = service.run_action_step(root_batch)
    first_telemetry = first.metadata["profile_telemetry"]
    assert first_telemetry["compact_torch_search_compile_runtime_status"] == "compiled"
    assert first_telemetry["compact_torch_search_compile_attempted"] == 1.0
    assert first_telemetry["compact_torch_search_compile_used"] is True
    assert first_telemetry["compact_torch_search_compile_cache_hit"] is False
    assert compile_calls == ["select_leaf", "expand_and_backup"]

    second = service.run_action_step(root_batch)
    second_telemetry = second.metadata["profile_telemetry"]
    assert second_telemetry["compact_torch_search_compile_runtime_status"] == "cache_hit"
    assert second_telemetry["compact_torch_search_compile_attempted"] == 0.0
    assert second_telemetry["compact_torch_search_compile_used"] is True
    assert second_telemetry["compact_torch_search_compile_cache_hit"] is True
    assert compile_calls == ["select_leaf", "expand_and_backup"]


def test_compact_torch_search_service_can_compile_helpers_with_root_noise(monkeypatch) -> None:
    torch = _torch_module()
    compile_calls: list[str] = []

    def fake_compile(fn, **_kwargs):
        compile_calls.append(fn.__name__)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.training = True
            self.initial_inference_mode_seen: list[bool] = []
            self.initial_grad_enabled_seen: list[bool] = []
            self.recurrent_inference_mode_seen: list[bool] = []
            self.recurrent_grad_enabled_seen: list[bool] = []

        def parameters(self):
            return iter([self.param])

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = bool(mode)
            return self

        def initial_inference(self, obs_tensor):
            assert self.training is False
            self.initial_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.initial_grad_enabled_seen.append(torch.is_grad_enabled())
            batch = int(obs_tensor.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.zeros((batch, 2), dtype=torch.float32),
            )

        def recurrent_inference(self, latent_state, _actions):
            assert self.training is False
            self.recurrent_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.recurrent_grad_enabled_seen.append(torch.is_grad_enabled())
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.25,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        compile_config=CompactTorchCompileConfig(
            request_compile=True,
            require_cuda_device=False,
            require_torch_compile=True,
            require_all_roots_active=True,
            require_all_actions_legal=True,
        ),
    )

    action_step = service.run_action_step(root_batch)
    telemetry = action_step.metadata["profile_telemetry"]

    assert telemetry["compact_torch_search_root_noise_weight"] == 0.25
    assert telemetry["compact_torch_search_compile_runtime_status"] == "compiled"
    assert telemetry["compact_torch_search_compile_used"] is True
    assert compile_calls == ["select_leaf", "expand_and_backup"]


def test_compact_torch_search_service_can_compile_model_inference(monkeypatch) -> None:
    torch = _torch_module()
    compile_calls: list[tuple[str, str | None]] = []

    def fake_compile(fn, **kwargs):
        compile_calls.append((fn.__name__, kwargs.get("mode")))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.training = True
            self.initial_inference_mode_seen: list[bool] = []
            self.initial_grad_enabled_seen: list[bool] = []
            self.recurrent_inference_mode_seen: list[bool] = []
            self.recurrent_grad_enabled_seen: list[bool] = []

        def parameters(self):
            return iter([self.param])

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = bool(mode)
            return self

        def initial_inference(self, obs_tensor):
            assert self.training is False
            self.initial_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.initial_grad_enabled_seen.append(torch.is_grad_enabled())
            batch = int(obs_tensor.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.zeros((batch, 2), dtype=torch.float32),
            )

        def recurrent_inference(self, latent_state, _actions):
            assert self.training is False
            self.recurrent_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.recurrent_grad_enabled_seen.append(torch.is_grad_enabled())
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    policy = FakePolicy()
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=True,
            request_model_compile=True,
            model_compile_mode="default",
            require_cuda_device=False,
            require_torch_compile=True,
            require_all_roots_active=True,
            require_all_actions_legal=True,
            recurrent_action_shape_mode="flat",
        ),
    )

    first = service.run_action_step(root_batch)
    first_telemetry = first.metadata["profile_telemetry"]
    assert first_telemetry["compact_torch_search_model_compile_runtime_status"] == "compiled"
    assert first_telemetry["compact_torch_search_model_compile_attempted"] == 1.0
    assert first_telemetry["compact_torch_search_model_compile_used"] is True
    assert first_telemetry["compact_torch_search_model_compile_mode"] == "default"
    assert first_telemetry["compact_torch_search_recurrent_action_shape_mode_effective"] == "flat"
    assert first_telemetry["compact_torch_search_model_training_after_inference"] is True
    assert first_telemetry["compact_torch_search_model_inference_mode_used"] is True
    assert policy._model.training is True
    assert policy._model.initial_inference_mode_seen == [True]
    assert policy._model.initial_grad_enabled_seen == [False]
    assert policy._model.recurrent_inference_mode_seen == [True]
    assert policy._model.recurrent_grad_enabled_seen == [False]
    assert compile_calls == [
        ("select_leaf", "reduce-overhead"),
        ("expand_and_backup", "reduce-overhead"),
        ("initial_inference", "default"),
        ("recurrent_inference", "default"),
    ]

    second = service.run_action_step(root_batch)
    second_telemetry = second.metadata["profile_telemetry"]
    assert second_telemetry["compact_torch_search_model_compile_runtime_status"] == "cache_hit"
    assert second_telemetry["compact_torch_search_model_compile_attempted"] == 0.0
    assert second_telemetry["compact_torch_search_model_compile_used"] is True
    assert second_telemetry["compact_torch_search_model_compile_mode"] == "default"
    assert second_telemetry["compact_torch_search_model_training_after_inference"] is True
    assert second_telemetry["compact_torch_search_model_inference_mode_used"] is True
    assert policy._model.training is True
    assert policy._model.initial_inference_mode_seen == [True, True]
    assert policy._model.initial_grad_enabled_seen == [False, False]
    assert policy._model.recurrent_inference_mode_seen == [True, True]
    assert policy._model.recurrent_grad_enabled_seen == [False, False]
    assert compile_calls == [
        ("select_leaf", "reduce-overhead"),
        ("expand_and_backup", "reduce-overhead"),
        ("initial_inference", "default"),
        ("recurrent_inference", "default"),
    ]

    with pytest.raises(ValueError, match="model_compile_mode must be one of"):
        CompactTorchSearchServiceV1(
            policy=FakePolicy(),
            num_simulations=1,
            compile_config=CompactTorchCompileConfig(model_compile_mode="cudagraph-maybe"),
        )


def test_compact_torch_search_service_direct_initial_core_matches_public_contract() -> None:
    torch = _torch_module()
    policy = _core_capable_policy(torch)
    model = policy._model
    obs = torch.arange(2 * 4 * 8 * 8, dtype=torch.float32).reshape(2, 4, 8, 8)

    public = model.initial_inference(obs)
    runtime = compact_torch_module._CompactTorchInitialInferenceRuntime(
        model=model,
        initial_inference_fn=model.initial_inference,
        requested_mode=COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
    )
    direct = runtime.run(obs)

    torch.testing.assert_close(direct.policy_logits, public.policy_logits)
    torch.testing.assert_close(direct.value, public.value)
    torch.testing.assert_close(direct.latent_state, public.latent_state)
    assert not hasattr(direct, "reward")
    assert model.initial_calls == 1
    assert model.representation_calls == 2
    assert model.prediction_calls == 2
    telemetry = runtime.telemetry()
    assert (
        telemetry["compact_torch_search_initial_inference_mode_effective"]
        == COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE
    )
    assert telemetry["compact_torch_search_initial_inference_direct_used"] is True
    assert telemetry["compact_torch_search_initial_inference_direct_decoded_output"] is True
    assert telemetry["compact_torch_search_initial_inference_direct_reward_materialized"] is False
    assert telemetry["compact_torch_search_initial_inference_fallback_count"] == 0.0
    assert (
        telemetry["compact_torch_search_service_initial_inference_representation_cuda_event_status"]
        == "disabled"
    )
    assert (
        telemetry["compact_torch_search_service_initial_inference_prediction_cuda_event_status"]
        == "disabled"
    )
    assert (
        telemetry["compact_torch_search_service_initial_inference_direct_core_cuda_event_sec"]
        == 0.0
    )


@pytest.mark.parametrize("entrypoint", ["run", "run_action_step"])
def test_compact_torch_search_service_direct_initial_core_matches_public_path(
    entrypoint,
) -> None:
    torch = _torch_module()
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )

    def service_for(mode: str):
        policy = _core_capable_policy(torch)
        service = CompactTorchSearchServiceV1(
            policy=policy,
            num_simulations=1,
            root_noise_weight=0.0,
            compile_config=CompactTorchCompileConfig(
                request_compile=False,
                require_cuda_device=False,
                require_torch_compile=False,
                require_all_roots_active=False,
                require_all_actions_legal=False,
                initial_inference_mode=mode,
            ),
        )
        return policy, service

    def materialize(service):
        if entrypoint == "run":
            result = service.run(root_batch)
            return (
                result.selected_action,
                result.visit_policy,
                result.root_value,
                result.raw_visit_counts,
                result.metadata,
            )
        action_step = service.run_action_step(root_batch)
        replay_payload = service.flush_replay_payload(action_step.replay_payload_handle)
        return (
            action_step.selected_action,
            replay_payload.visit_policy,
            replay_payload.root_value,
            replay_payload.raw_visit_counts,
            action_step.metadata,
        )

    public_policy, public_service = service_for(COMPACT_TORCH_INITIAL_INFERENCE_MODE_MODEL_METHOD)
    direct_policy, direct_service = service_for(COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE)
    public = materialize(public_service)
    direct = materialize(direct_service)

    for public_array, direct_array in zip(public[:4], direct[:4], strict=True):
        np.testing.assert_allclose(direct_array, public_array)
    telemetry = direct[4]["profile_telemetry"]
    assert (
        telemetry["compact_torch_search_initial_inference_mode_requested"]
        == COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE
    )
    assert (
        telemetry["compact_torch_search_initial_inference_mode_effective"]
        == COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE
    )
    assert telemetry["compact_torch_search_initial_inference_direct_used"] is True
    assert telemetry["compact_torch_search_service_root_output_direct_decoded"] is True
    assert telemetry["compact_torch_search_initial_inference_direct_decoded_output"] is True
    assert telemetry["compact_torch_search_initial_inference_direct_reward_materialized"] is False
    assert telemetry["compact_torch_search_initial_inference_fallback_count"] == 0.0
    assert (
        telemetry["compact_torch_search_service_initial_inference_direct_core_cuda_event_status"]
        == "disabled"
    )
    assert (
        telemetry[
            "compact_torch_search_service_initial_inference_direct_core_cuda_event_residual_status"
        ]
        == "outer:disabled;split:disabled"
    )
    assert direct_policy._model.initial_calls == 0
    assert direct_policy._model.representation_calls == 1
    assert direct_policy._model.prediction_calls == 1
    assert public_policy._model.initial_calls == 1
    assert (
        public[4]["profile_telemetry"]["compact_torch_search_service_root_output_direct_decoded"]
        is False
    )


def test_compact_torch_search_service_direct_core_channels_last_observation_reaches_model() -> None:
    torch = _torch_module()
    policy = _core_capable_policy(torch)
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            initial_inference_mode=COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
            observation_memory_format="channels_last",
            model_memory_format="contiguous",
        ),
    )

    action_step = service.run_action_step(root_batch)
    telemetry = action_step.metadata["profile_telemetry"]

    assert telemetry["compact_torch_search_observation_memory_format_requested"] == "channels_last"
    assert telemetry["compact_torch_search_observation_memory_format_effective"] == "channels_last"
    assert telemetry["compact_torch_search_observation_normalized_uint8"] is True
    assert telemetry["compact_torch_search_observation_is_channels_last"] is True
    assert telemetry["compact_torch_search_model_memory_format_requested"] == "contiguous"
    assert telemetry["compact_torch_search_model_memory_format_active"] == "contiguous"
    assert telemetry["compact_torch_search_model_memory_format_applied"] is False
    assert telemetry["compact_torch_search_initial_inference_direct_used"] is True
    assert telemetry["compact_torch_search_initial_inference_fallback_count"] == 0.0
    assert policy._model.to_memory_formats == []
    assert policy._model.representation_input_signatures
    signature = policy._model.representation_input_signatures[0]
    assert signature["shape"] == (2, 4, 8, 8)
    assert signature["dtype"] == "torch.float32"
    assert signature["is_channels_last"] is True

    second = service.run_action_step(root_batch)
    second_telemetry = second.metadata["profile_telemetry"]
    assert second_telemetry["compact_torch_search_model_memory_format_applied"] is False
    assert policy._model.to_memory_formats == []


def test_compact_torch_search_service_channels_last_root_latent_recurrent_safe() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, value: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = value
            self.reward = value
            self.latent_state = latent

    class ChannelsLastLatentModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent_latent_contiguous_seen: list[bool] = []
            self.recurrent_latent_channels_last_seen: list[bool] = []

        def initial_inference(self, obs_tensor):
            latent = self._representation(obs_tensor)
            policy_logits, value = self._prediction(latent)
            return FakeOutput(policy_logits, value, latent)

        def _representation(self, obs_tensor):
            return obs_tensor[:, :2, :, :].contiguous(memory_format=torch.channels_last)

        def _prediction(self, latent_state):
            batch = int(latent_state.shape[0])
            logits = torch.zeros((batch, 3), dtype=torch.float32)
            logits[:, 1] = 3.0
            value = latent_state.reshape(batch, -1).mean(dim=1, keepdim=True)
            return logits, value

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            self.recurrent_latent_contiguous_seen.append(bool(latent_state.is_contiguous()))
            self.recurrent_latent_channels_last_seen.append(
                bool(latent_state.is_contiguous(memory_format=torch.channels_last))
            )
            value = latent_state.view(batch, -1)[:, :1]
            logits = torch.zeros((batch, 3), dtype=torch.float32)
            return FakeOutput(logits, value, latent_state + 1.0)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = ChannelsLastLatentModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    policy = FakePolicy()
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            initial_inference_mode=COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
            observation_memory_format="channels_last",
            model_memory_format="contiguous",
        ),
    )

    action_step = service.run_action_step(root_batch)
    telemetry = action_step.metadata["profile_telemetry"]

    assert policy._model.recurrent_latent_contiguous_seen == [True]
    assert policy._model.recurrent_latent_channels_last_seen == [False]
    assert telemetry["compact_torch_search_root_latent_is_channels_last_before_recurrent"] is True
    assert telemetry["compact_torch_search_root_latent_contiguous_for_recurrent"] is True
    assert telemetry["compact_torch_search_root_latent_contiguous_copy_bytes"] > 0.0
    assert telemetry["compact_torch_search_service_root_latent_prepare_sec"] >= 0.0


def test_compact_torch_search_service_channels_last_observation_requires_rank4() -> None:
    torch = _torch_module()
    policy = _core_capable_policy(torch)
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
        observation_shape=(2, 8, 8),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            observation_memory_format="channels_last",
        ),
    )

    with pytest.raises(ReplayCompatibilityError, match="requires rank-4"):
        service.run_action_step(root_batch)


def test_compact_torch_search_service_direct_initial_core_uses_inference_guard() -> None:
    torch = _torch_module()
    policy = _core_capable_policy(torch)
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            initial_inference_mode=COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
        ),
    )

    action_step = service.run_action_step(root_batch)

    telemetry = action_step.metadata["profile_telemetry"]
    assert telemetry["compact_torch_search_model_training_after_inference"] is True
    assert telemetry["compact_torch_search_model_inference_mode_used"] is True
    assert policy._model.training is True
    assert policy._model.core_training_seen == [False, False]
    assert policy._model.core_inference_mode_seen == [True, True]
    assert policy._model.core_grad_enabled_seen == [False, False]


def test_compact_torch_search_service_restores_training_on_direct_core_exception() -> None:
    torch = _torch_module()
    policy = _core_capable_policy(torch, fail_representation=True)
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            initial_inference_mode=COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
        ),
    )

    with pytest.raises(RuntimeError, match="representation failed"):
        service.run_action_step(root_batch)

    assert policy._model.training is True


def test_compact_torch_search_service_direct_initial_core_requires_core_surface() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class PublicOnlyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.zeros((batch, 2), dtype=torch.float32),
            )

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = PublicOnlyModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
            initial_inference_mode=COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
        ),
    )

    with pytest.raises(ValueError, match="requires callable model._representation"):
        service.run_action_step(root_batch)


def test_compact_torch_search_service_direct_initial_core_can_compile_core_model_path(
    monkeypatch,
) -> None:
    torch = _torch_module()
    compile_calls: list[tuple[str, str]] = []

    def fake_compile(fn, **kwargs):
        compile_calls.append((fn.__name__, str(kwargs.get("mode"))))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    policy = _core_capable_policy(torch)
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=True,
            request_model_compile=True,
            model_compile_mode="default",
            require_cuda_device=False,
            require_torch_compile=True,
            require_all_roots_active=True,
            require_all_actions_legal=True,
            recurrent_action_shape_mode="flat",
            initial_inference_mode=COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
        ),
    )

    first = service.run_action_step(root_batch)
    first_telemetry = first.metadata["profile_telemetry"]
    assert first_telemetry["compact_torch_search_model_compile_runtime_status"] == "compiled"
    assert first_telemetry["compact_torch_search_model_compile_attempted"] == 1.0
    assert first_telemetry["compact_torch_search_model_compile_used"] is True
    assert first_telemetry["compact_torch_search_model_compile_initial_path"] == "direct_core"
    assert (
        first_telemetry["compact_torch_search_model_compile_direct_core_representation_used"]
        is True
    )
    assert first_telemetry["compact_torch_search_model_compile_direct_core_prediction_used"] is True
    assert (
        first_telemetry["compact_torch_search_initial_inference_mode_effective"]
        == COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE
    )
    assert first_telemetry["compact_torch_search_initial_inference_direct_used"] is True
    assert first_telemetry["compact_torch_search_model_training_after_inference"] is True
    assert first_telemetry["compact_torch_search_model_inference_mode_used"] is True
    assert policy._model.training is True
    assert policy._model.initial_calls == 0
    assert policy._model.representation_calls == 1
    assert policy._model.prediction_calls == 1
    assert policy._model.recurrent_calls == 1
    assert compile_calls == [
        ("select_leaf", "reduce-overhead"),
        ("expand_and_backup", "reduce-overhead"),
        ("_representation", "default"),
        ("_prediction", "default"),
        ("recurrent_inference", "default"),
    ]

    second = service.run_action_step(root_batch)
    second_telemetry = second.metadata["profile_telemetry"]
    assert second_telemetry["compact_torch_search_model_compile_runtime_status"] == "cache_hit"
    assert second_telemetry["compact_torch_search_model_compile_attempted"] == 0.0
    assert second_telemetry["compact_torch_search_model_compile_used"] is True
    assert second_telemetry["compact_torch_search_model_compile_initial_path"] == "direct_core"
    assert (
        second_telemetry["compact_torch_search_model_compile_direct_core_representation_used"]
        is True
    )
    assert (
        second_telemetry["compact_torch_search_model_compile_direct_core_prediction_used"] is True
    )
    assert second_telemetry["compact_torch_search_initial_inference_direct_used"] is True
    assert policy._model.training is True
    assert policy._model.initial_calls == 0
    assert policy._model.representation_calls == 2
    assert policy._model.prediction_calls == 2
    assert policy._model.recurrent_calls == 2
    assert compile_calls == [
        ("select_leaf", "reduce-overhead"),
        ("expand_and_backup", "reduce-overhead"),
        ("_representation", "default"),
        ("_prediction", "default"),
        ("recurrent_inference", "default"),
    ]


def test_compact_torch_search_service_keeps_model_compile_cache_across_refresh(
    monkeypatch,
) -> None:
    torch = _torch_module()
    compile_calls: list[tuple[str, str]] = []

    def fake_compile(fn, **kwargs):
        compile_calls.append((fn.__name__, str(kwargs.get("mode"))))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    policy = _core_capable_policy(torch)
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=True,
            request_model_compile=True,
            model_compile_mode="default",
            require_cuda_device=False,
            require_torch_compile=True,
            require_all_roots_active=True,
            require_all_actions_legal=True,
            recurrent_action_shape_mode="flat",
            initial_inference_mode=COMPACT_TORCH_INITIAL_INFERENCE_MODE_DIRECT_CORE,
        ),
    )

    first = service.run_action_step(root_batch)
    first_payload = service.flush_replay_payload(first.replay_payload_handle)
    first_root_value = np.asarray(first_payload.root_value, dtype=np.float32).copy()
    assert first.metadata["profile_telemetry"][
        "compact_torch_search_model_compile_runtime_status"
    ] == "compiled"
    assert len(service._compiled_model_cache) == 1

    learner_policy = _core_capable_policy(torch)
    with torch.no_grad():
        learner_policy._model.bias.fill_(2.0)
    learner_digest = compact_model_state_digest_v1(learner_policy._model)
    stale_key = ("stale-model-id", "wrong-shape")
    service._compiled_model_cache[stale_key] = ("stale-representation",)

    state = service.refresh_model_state(
        model_state_dict=learner_policy._model.state_dict(),
        policy_version_ref="unit-policy:update-2",
        model_version_ref="unit-model:update-2",
        policy_source="unit_test_compile_refresh",
        learner_update_count=2,
        expected_model_state_digest=learner_digest,
    )

    assert state["cache_cleared"] is True
    assert state["compiled_helper_cache_size"] == 0
    assert state["compiled_model_cache_size"] == 1
    assert stale_key not in service._compiled_model_cache

    second = service.run_action_step(root_batch)
    second_payload = service.flush_replay_payload(second.replay_payload_handle)
    second_telemetry = second.metadata["profile_telemetry"]

    assert second_telemetry["compact_torch_search_model_compile_runtime_status"] == "cache_hit"
    assert second_telemetry["compact_torch_search_model_compile_attempted"] == 0.0
    assert second_telemetry["compact_torch_search_model_compile_used"] is True
    assert second.metadata["compact_policy_refresh_model_state_digest"] == learner_digest
    assert second.metadata["compact_policy_refresh_learner_update_count"] == 2
    assert np.all(np.asarray(second_payload.root_value) > first_root_value + 1.0)
    assert compile_calls.count(("_representation", "default")) == 1
    assert compile_calls.count(("_prediction", "default")) == 1
    assert compile_calls.count(("recurrent_inference", "default")) == 1
    assert compile_calls.count(("select_leaf", "reduce-overhead")) == 2
    assert compile_calls.count(("expand_and_backup", "reduce-overhead")) == 2


def test_compact_torch_search_service_forces_eval_during_search_and_restores() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.training = True
            self.eval_seen = 0
            self.train_restore_values: list[bool] = []
            self.initial_inference_mode_seen: list[bool] = []
            self.initial_grad_enabled_seen: list[bool] = []
            self.recurrent_inference_mode_seen: list[bool] = []
            self.recurrent_grad_enabled_seen: list[bool] = []

        def parameters(self):
            return iter([self.param])

        def eval(self):
            self.eval_seen += 1
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.train_restore_values.append(bool(mode))
            self.training = bool(mode)
            return self

        def initial_inference(self, obs_tensor):
            assert self.training is False
            self.initial_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.initial_grad_enabled_seen.append(torch.is_grad_enabled())
            batch = int(obs_tensor.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.zeros((batch, 2), dtype=torch.float32),
            )

        def recurrent_inference(self, latent_state, _actions):
            assert self.training is False
            self.recurrent_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.recurrent_grad_enabled_seen.append(torch.is_grad_enabled())
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    policy = FakePolicy()
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            recurrent_action_shape_mode="flat",
        ),
    )

    action_step = service.run_action_step(root_batch)

    assert policy._model.training is True
    assert policy._model.eval_seen == 1
    assert policy._model.train_restore_values == [True]
    assert policy._model.initial_inference_mode_seen == [True]
    assert policy._model.initial_grad_enabled_seen == [False]
    assert policy._model.recurrent_inference_mode_seen == [True]
    assert policy._model.recurrent_grad_enabled_seen == [False]
    telemetry = action_step.metadata["profile_telemetry"]
    assert telemetry["compact_torch_search_model_training_before_inference"] is True
    assert telemetry["compact_torch_search_model_training_after_inference"] is True
    assert telemetry["compact_torch_search_model_eval_applied_for_inference"] is True
    assert telemetry["compact_torch_search_model_inference_mode_used"] is True
    for key in (
        "compact_torch_search_service_inference_guard_enter_sec",
        "compact_torch_search_service_inference_guard_exit_sec",
        "compact_torch_search_service_inference_guard_total_sec",
    ):
        assert key in telemetry
        assert key in action_step.metadata
        assert action_step.metadata[key] == telemetry[key]
        assert telemetry[key] >= 0.0
    assert telemetry["compact_torch_search_service_inference_guard_total_sec"] == (
        telemetry["compact_torch_search_service_inference_guard_enter_sec"]
        + telemetry["compact_torch_search_service_inference_guard_exit_sec"]
    )


def test_compact_torch_search_service_recurrent_uses_inference_guard_without_nested_no_grad(
    monkeypatch,
) -> None:
    torch = _torch_module()
    if not callable(getattr(torch, "inference_mode", None)):
        pytest.skip("test requires torch.inference_mode")

    def fail_no_grad():
        raise AssertionError("recurrent search should rely on the service inference guard")

    monkeypatch.setattr(torch, "no_grad", fail_no_grad)

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.training = True
            self.recurrent_inference_mode_seen: list[bool] = []
            self.recurrent_grad_enabled_seen: list[bool] = []

        def parameters(self):
            return iter([self.param])

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = bool(mode)
            return self

        def initial_inference(self, obs_tensor):
            assert self.training is False
            batch = int(obs_tensor.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.zeros((batch, 2), dtype=torch.float32),
            )

        def recurrent_inference(self, latent_state, _actions):
            assert self.training is False
            self.recurrent_inference_mode_seen.append(torch.is_inference_mode_enabled())
            self.recurrent_grad_enabled_seen.append(torch.is_grad_enabled())
            batch = int(latent_state.shape[0])
            logits = torch.zeros((batch, 3), dtype=torch.float32)
            logits[:, 1] = 1.0
            return FakeOutput(logits, latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    policy = FakePolicy()
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=2,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            recurrent_action_shape_mode="flat",
        ),
    )

    action_step = service.run_action_step(root_batch)

    assert (
        action_step.metadata["profile_telemetry"]["compact_torch_search_model_inference_mode_used"]
        is True
    )
    for key in (
        "compact_torch_search_service_inference_guard_enter_sec",
        "compact_torch_search_service_inference_guard_exit_sec",
        "compact_torch_search_service_inference_guard_total_sec",
    ):
        assert key in action_step.metadata["profile_telemetry"]
        assert action_step.metadata["profile_telemetry"][key] >= 0.0
    assert policy._model.training is True
    assert policy._model.recurrent_inference_mode_seen == [True, True]
    assert policy._model.recurrent_grad_enabled_seen == [False, False]


def test_compact_torch_search_service_restores_training_state_on_initial_exception() -> None:
    torch = _torch_module()

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.training = True
            self.eval_seen = 0
            self.train_restore_values: list[bool] = []
            self.initial_inference_mode_seen: list[bool] = []

        def parameters(self):
            return iter([self.param])

        def eval(self):
            self.eval_seen += 1
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.train_restore_values.append(bool(mode))
            self.training = bool(mode)
            return self

        def initial_inference(self, _obs_tensor):
            assert self.training is False
            self.initial_inference_mode_seen.append(torch.is_inference_mode_enabled())
            raise RuntimeError("initial boom")

        def recurrent_inference(self, _latent_state, _actions):  # pragma: no cover
            raise AssertionError("recurrent inference should not run")

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    policy = FakePolicy()
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            recurrent_action_shape_mode="flat",
        ),
    )

    with pytest.raises(RuntimeError, match="initial boom"):
        service.run_action_step(root_batch)

    assert policy._model.training is True
    assert policy._model.eval_seen == 1
    assert policy._model.train_restore_values == [True]
    assert policy._model.initial_inference_mode_seen == [True]


def test_compact_torch_search_service_strict_flat_recurrent_actions_do_not_retry() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.action_shapes: list[tuple[int, ...]] = []

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.zeros((batch, 2), dtype=torch.float32),
            )

        def recurrent_inference(self, latent_state, actions):
            self.action_shapes.append(tuple(int(dim) for dim in actions.shape))
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    policy = FakePolicy()
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=2,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            recurrent_action_shape_mode="flat",
        ),
    )

    result = service.run_action_step(root_batch)
    telemetry = result.metadata["profile_telemetry"]
    assert policy._model.action_shapes == [(2,), (2,)]
    assert telemetry["compact_torch_search_recurrent_action_shape_mode_effective"] == "flat"
    assert telemetry["compact_torch_search_recurrent_action_shape_exception_fallback_count"] == 0.0


def test_compact_torch_search_service_strict_flat_recurrent_error_surfaces() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.recurrent_calls = 0
            self.training = True
            self.train_restore_values: list[bool] = []

        def parameters(self):
            return iter([self.param])

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.train_restore_values.append(bool(mode))
            self.training = bool(mode)
            return self

        def initial_inference(self, obs_tensor):
            assert self.training is False
            batch = int(obs_tensor.shape[0])
            return FakeOutput(
                torch.zeros((batch, 3), dtype=torch.float32),
                torch.zeros((batch, 2), dtype=torch.float32),
            )

        def recurrent_inference(self, _latent_state, _actions):
            assert self.training is False
            self.recurrent_calls += 1
            raise RuntimeError("flat recurrent failed")

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    policy = FakePolicy()
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=2,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            recurrent_action_shape_mode="flat",
        ),
    )

    with pytest.raises(RuntimeError, match="flat recurrent failed"):
        service.run_action_step(root_batch)
    assert policy._model.training is True
    assert policy._model.train_restore_values == [True]
    assert policy._model.recurrent_calls == 1


def test_compact_torch_search_service_rejects_host_batch_when_resident_required() -> None:
    torch = _torch_module()

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, _obs_tensor):  # pragma: no cover - fail before model
            raise AssertionError("resident-required host batch reached model")

        def recurrent_inference(self, _latent_state, _actions):  # pragma: no cover
            raise AssertionError("resident-required host batch reached model")

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type("Cfg", (), {"root_noise_weight": 0.0})()

    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        require_resident_observation=True,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
    )

    with pytest.raises(ReplayCompatibilityError, match="requires resident observation"):
        service.run_action_step(root_batch)


def test_compact_torch_search_service_rejects_zero_active_host_when_resident_required() -> None:
    torch = _torch_module()

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, _obs_tensor):  # pragma: no cover
            raise AssertionError("resident-required host batch reached model")

        def recurrent_inference(self, _latent_state, _actions):  # pragma: no cover
            raise AssertionError("resident-required host batch reached model")

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type("Cfg", (), {"root_noise_weight": 0.0})()

    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        require_resident_observation=True,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.zeros((2,), dtype=np.bool_),
    )

    with pytest.raises(ReplayCompatibilityError, match="requires resident observation"):
        service.run(root_batch)
    with pytest.raises(ReplayCompatibilityError, match="requires resident observation"):
        service.run_action_step(root_batch)


def test_compact_torch_search_service_consumes_resident_observation_without_h2d() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.last_input = None

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            self.last_input = obs_tensor.detach().clone()
            batch = int(obs_tensor.shape[0])
            logits = torch.zeros((batch, 3), dtype=torch.float32)
            logits[:, 1] = 4.0
            latent = torch.zeros((batch, 2), dtype=torch.float32)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    policy = FakePolicy()
    device_observation = torch.full((2, 1, 4, 8, 8), 255, dtype=torch.uint8)
    resident = ResidentObservationBatchV1(
        device_observation=device_observation,
        root_device_observation=device_observation.reshape(2, 4, 8, 8),
        generation_id=7,
        batch_size=2,
        player_count=1,
        stack_shape=(4, 8, 8),
        dtype=str(device_observation.dtype),
        device=str(device_observation.device),
        row_major_order=True,
        fresh_for_step_index=11,
        source_backend="unit_test_resident_torch",
    )
    root_batch = _root_batch(
        legal_mask=np.ones((2, 3), dtype=np.bool_),
        active_root_mask=np.ones((2,), dtype=np.bool_),
        observation_shape=(2, 4, 8, 8),
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_observation=resident,
    )
    service = CompactTorchSearchServiceV1(
        policy=policy,
        num_simulations=1,
        require_resident_observation=True,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )

    action_step = service.run_action_step(root_batch)
    telemetry = action_step.metadata["profile_telemetry"]

    assert telemetry["resident_observation_used"] is True
    assert telemetry["resident_observation_generation_id"] == 7
    assert telemetry["resident_observation_search_consumed_generation"] == 7
    assert telemetry["resident_observation_h2d_bytes"] == 0.0
    assert telemetry["resident_observation_host_fallback_used"] is False
    assert telemetry["compact_torch_search_service_resident_obs_reused"] == 1.0
    assert telemetry["compact_torch_search_service_obs_h2d_bytes"] == 0.0
    assert policy._model.last_input is not None
    assert float(policy._model.last_input.max()) == pytest.approx(1.0)


def test_compact_torch_search_service_reports_noncontiguous_active_root_copy() -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class FakeModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            batch = int(obs_tensor.shape[0])
            logits = torch.zeros((batch, 3), dtype=torch.float32)
            latent = torch.zeros((batch, 2), dtype=torch.float32)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = FakeModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    root_batch = _root_batch(
        legal_mask=np.ones((3, 3), dtype=np.bool_),
        active_root_mask=np.asarray([True, False, True], dtype=np.bool_),
        observation_shape=(3, 4, 8, 8),
    )
    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            require_all_roots_active=False,
            require_all_actions_legal=False,
        ),
    )

    action_step = service.run_action_step(root_batch)

    assert action_step.metadata["profile_telemetry"][
        "compact_torch_search_service_root_observation_copy_bytes"
    ] == float(root_batch.observation[[0, 2]].nbytes)
    assert action_step.metadata["profile_telemetry"][
        "compact_torch_search_service_root_mask_copy_bytes"
    ] == float(root_batch.legal_mask[[0, 2]].nbytes)


@pytest.mark.parametrize("observation_memory_format", ["contiguous", "channels_last"])
def test_compact_torch_search_service_uses_fresh_observations_for_same_shape_calls(
    observation_memory_format,
) -> None:
    torch = _torch_module()

    class FakeOutput:
        def __init__(self, logits: torch.Tensor, latent: torch.Tensor) -> None:
            self.policy_logits = logits
            self.value = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.reward = torch.zeros((int(logits.shape[0]), 1), dtype=torch.float32)
            self.latent_state = latent

    class PixelModel:
        def __init__(self) -> None:
            self.param = torch.nn.Parameter(torch.zeros(()))

        def parameters(self):
            return iter([self.param])

        def initial_inference(self, obs_tensor):
            signal = obs_tensor.reshape(int(obs_tensor.shape[0]), -1)[:, 0]
            logits = torch.stack(
                [
                    signal * 4.0,
                    (1.0 - signal) * 4.0,
                    torch.zeros_like(signal),
                ],
                dim=1,
            )
            latent = signal.reshape(-1, 1).repeat(1, 2)
            return FakeOutput(logits, latent)

        def recurrent_inference(self, latent_state, _actions):
            batch = int(latent_state.shape[0])
            return FakeOutput(torch.zeros((batch, 3), dtype=torch.float32), latent_state)

    class FakePolicy:
        def __init__(self) -> None:
            self._model = PixelModel()
            self._cfg = type(
                "Cfg",
                (),
                {
                    "pb_c_base": 19652,
                    "pb_c_init": 1.25,
                    "discount_factor": 0.997,
                    "root_noise_weight": 0.0,
                    "root_dirichlet_alpha": 0.3,
                    "value_delta_max": 0.01,
                },
            )()

    def batch_with_first_pixel(value: int) -> CompactRootBatchV1:
        batch = _root_batch(
            legal_mask=np.ones((1, 3), dtype=np.bool_),
            active_root_mask=np.ones((1,), dtype=np.bool_),
            observation_shape=(1, 4, 8, 8),
        )
        batch.observation[:, :, :, :] = 0
        batch.observation[0, 0, 0, 0] = value
        return batch

    service = CompactTorchSearchServiceV1(
        policy=FakePolicy(),
        num_simulations=1,
        root_noise_weight=0.0,
        compile_config=CompactTorchCompileConfig(
            request_compile=False,
            require_cuda_device=False,
            require_torch_compile=False,
            observation_memory_format=observation_memory_format,
        ),
    )

    low_signal = service.run(batch_with_first_pixel(0))
    high_signal = service.run(batch_with_first_pixel(255))

    assert int(low_signal.selected_action[0]) == 1
    assert int(high_signal.selected_action[0]) == 0


def test_fixed_shape_forced_masks_and_noise_preconditions_are_deterministic() -> None:
    torch_module = _torch_module()
    root_batch = _root_batch(
        legal_mask=np.asarray(
            [
                [True, False, True],
                [True, True, True],
            ],
            dtype=np.bool_,
        ),
        active_root_mask=np.asarray([True, False], dtype=np.bool_),
    )
    config = CompactTorchCompileConfig(
        require_cuda_device=False,
        require_torch_compile=False,
    )

    masks = compact_torch_fixed_shape_masks(root_batch)
    assert masks.all_roots_active is False
    assert masks.all_actions_legal is False
    assert masks.forced_all_roots_active is False
    assert masks.forced_all_actions_legal is False

    inactive = compact_torch_compile_eligibility(
        root_batch,
        device="cpu",
        root_noise_weight=0.0,
        config=config,
        torch_module=torch_module,
    )
    assert inactive.reason == "requires_all_roots_active"

    partial = compact_torch_compile_eligibility(
        root_batch,
        device="cpu",
        root_noise_weight=0.0,
        config=config,
        torch_module=torch_module,
        force_all_roots_active=True,
    )
    assert partial.reason == "requires_all_actions_legal"
    assert partial.telemetry["compact_torch_search_forced_all_roots_active"] is True

    forced = compact_torch_compile_eligibility(
        root_batch,
        device="cpu",
        root_noise_weight=0.0,
        config=config,
        torch_module=torch_module,
        force_all_roots_active=True,
        force_all_actions_legal=True,
    )
    assert forced.eligible is True
    assert forced.reason == "preconditions_satisfied"
    assert forced.telemetry["compact_torch_search_forced_all_actions_legal"] is True
    assert forced.telemetry["compact_torch_search_compile_enabled"] == 1.0

    noisy_default = compact_torch_compile_eligibility(
        root_batch,
        device="cpu",
        root_noise_weight=0.25,
        config=config,
        torch_module=torch_module,
        force_all_roots_active=True,
        force_all_actions_legal=True,
    )
    assert noisy_default.eligible is True
    assert noisy_default.reason == "preconditions_satisfied"
    assert noisy_default.telemetry["compact_torch_search_root_noise_weight"] == 0.25

    strict_noise_config = CompactTorchCompileConfig(
        require_cuda_device=False,
        require_torch_compile=False,
        require_root_noise_zero=True,
    )
    noisy_strict = compact_torch_compile_eligibility(
        root_batch,
        device="cpu",
        root_noise_weight=0.25,
        config=strict_noise_config,
        torch_module=torch_module,
        force_all_roots_active=True,
        force_all_actions_legal=True,
    )
    assert noisy_strict.eligible is False
    assert noisy_strict.reason == "requires_root_noise_zero"


def test_select_and_backup_helpers_operate_on_tiny_tensors_without_lightzero() -> None:
    torch = _torch_module()
    device = torch.device("cpu")
    root_count = 2
    action_count = 3
    max_nodes = 3
    row_index = torch.arange(root_count, dtype=torch.long, device=device)
    edge_child = torch.full(
        (root_count, max_nodes, action_count),
        -1,
        dtype=torch.long,
        device=device,
    )
    edge_visit = torch.zeros((root_count, max_nodes, action_count), dtype=torch.float32)
    edge_value_sum = torch.zeros_like(edge_visit)
    edge_reward = torch.zeros_like(edge_visit)
    edge_prior = torch.zeros_like(edge_visit)
    edge_prior[:, 0, :] = torch.tensor(
        [
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ],
        dtype=torch.float32,
    )
    path_node_history = torch.empty((1, root_count), dtype=torch.long)
    path_action_history = torch.empty_like(path_node_history)
    path_active_history = torch.empty((1, root_count), dtype=torch.bool)
    flat_mask_tensor = torch.tensor(
        [
            [True, True, False],
            [False, True, True],
        ],
        dtype=torch.bool,
    )
    min_value = torch.full((root_count,), float("inf"), dtype=torch.float32)
    max_value = torch.full((root_count,), -float("inf"), dtype=torch.float32)

    select_leaf = make_compact_torch_select_leaf_fixed(torch=torch)
    leaf_parent, leaf_action = select_leaf(
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
        0,
        19652.0,
        1.25,
        0.01,
    )

    assert leaf_parent.tolist() == [0, 0]
    assert leaf_action.tolist() == [1, 2]
    assert path_node_history[0].tolist() == [0, 0]
    assert path_action_history[0].tolist() == [1, 2]
    assert path_active_history[0].tolist() == [True, True]

    latent_pool = torch.zeros((max_nodes, root_count, 2), dtype=torch.float32)
    node_latent_slot = torch.zeros((root_count, max_nodes), dtype=torch.long)
    next_node_index = torch.ones((root_count,), dtype=torch.long)
    next_latent_state = torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    reward = torch.tensor([1.0, -1.0], dtype=torch.float32)
    value = torch.tensor([0.5, 0.25], dtype=torch.float32)
    recurrent_priors = torch.full((root_count, action_count), 1.0 / action_count)

    expand_and_backup = make_compact_torch_expand_and_backup_fixed(torch=torch)
    next_node_index, min_value, max_value = expand_and_backup(
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
        0,
        0.5,
    )

    assert next_node_index.tolist() == [2, 2]
    assert edge_child[0, 0, 1].item() == 1
    assert edge_child[1, 0, 2].item() == 1
    assert edge_visit[0, 0, 1].item() == pytest.approx(1.0)
    assert edge_visit[1, 0, 2].item() == pytest.approx(1.0)
    assert edge_value_sum[0, 0, 1].item() == pytest.approx(1.25)
    assert edge_value_sum[1, 0, 2].item() == pytest.approx(-0.875)
    assert min_value.tolist() == pytest.approx([1.25, -0.875])
    assert max_value.tolist() == pytest.approx([1.25, -0.875])
    np.testing.assert_allclose(
        latent_pool[1].numpy(),
        next_latent_state.numpy(),
    )
    assert node_latent_slot[:, 1].tolist() == [1, 1]
