from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.env.vector_runtime import DEATH_MODE_NORMAL
from curvyzero.training.compact_death_terminal_contract import (
    COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_normal_collision_death_evidence_from_profile_result_v1,
)
from curvyzero.training.compact_muzero_learner import CompactMuZeroLearnerConfigV1
from curvyzero.training.compact_muzero_learner import CompactMuZeroLearnerEdgeV1
from curvyzero.training.compact_muzero_learner import build_compact_muzero_learner_batch_v1
from curvyzero.training.compact_owned_loop import CompactOwnedLoopConfigV1
from curvyzero.training.compact_owned_loop import CompactOwnedLoopV1
from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import compact_owned_loop_replay_store_metadata
from curvyzero.training.compact_owned_trainer import CompactOwnedTrainerConfigV1
from curvyzero.training.compact_owned_trainer import CompactOwnedTrainerV1
from curvyzero.training.compact_policy_row_bridge import CompactReplayIndexRowsV1
from curvyzero.training.compact_policy_refresh_handoff import (
    build_compact_policy_refresh_handoff_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_model_state_digest_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_policy_refresh_metadata_from_state_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    build_current_policy_observation_sidecar_v1,
)
from curvyzero.training.compact_torch_search_service import (
    CompactTorchCompileConfig,
)
from curvyzero.training.compact_torch_search_service import (
    CompactTorchSearchServiceV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    load_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    restore_compact_owned_loop_runtime_state_v1,
)
from curvyzero.training.exploration_bonus import normalize_exploration_bonus_spec
from curvyzero.training.source_state_hybrid_observation_profile import (
    _CompactReplayRingV1,
)


def _torch_module():
    return pytest.importorskip("torch")


def test_compact_owned_trainer_real_update_and_checkpoint_remain_non_promoted(tmp_path):
    torch = _torch_module()
    torch.manual_seed(20260528)
    model = _TinyMuZeroModel()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    learner = CompactMuZeroLearnerEdgeV1(
        model=model,
        config=CompactMuZeroLearnerConfigV1(device="cpu"),
    )
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-compact-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
        ),
        learner=learner,
    )

    result = trainer.train_on_sample_batch(_resident_sample(), train_steps=2)

    assert result.policy_version_ref == "unit-policy:update-2"
    assert result.model_version_ref == "unit-model:update-2"
    assert result.telemetry["compact_owned_trainer_update_claim"] is True
    assert result.telemetry["compact_owned_trainer_policy_refresh_consumed_by_search"] is False
    assert any(
        not torch.allclose(left, right.detach())
        for left, right in zip(before, trainer.model.parameters(), strict=True)
    )

    replay_state = _owned_replay_state(
        policy_version_ref=trainer.policy_version_ref,
        model_version_ref=trainer.model_version_ref,
    )
    checkpoint_path = trainer.save_checkpoint(
        checkpoint_id="unit-compact-owned-ckpt-0002",
        replay_store_state=replay_state,
        path=tmp_path / "compact-owned.pt",
        metrics={"unit_metric": 1.0},
    )

    assert checkpoint_path.is_file()
    checkpoint = trainer.checkpoint(
        checkpoint_id="unit-compact-owned-ckpt-0002b",
        replay_store_state=replay_state,
        metrics={"unit_metric": 1.0},
    )
    assert checkpoint.metadata["compact_coach_compatibility_gate_checkpoint_save_load"] is True
    assert checkpoint.metadata["compact_coach_compatibility_gate_resume_metadata"] is True
    assert checkpoint.metadata["compact_coach_compatibility_gate_reward_rnd_contract"] is True
    assert checkpoint.metadata["compact_coach_compatibility_gate_eval_gif_tournament_load"] is False
    assert (
        checkpoint.metadata["compact_coach_compatibility_gate_training_metrics_lineage"]
        is False
    )
    assert checkpoint.metadata["compact_coach_compatibility_promotion_eligible"] is False
    assert checkpoint.metadata["stock_eval_tournament_loadable"] is False
    assert checkpoint.metadata["reward_variant"] == "survival_plus_bonus_no_outcome"
    assert checkpoint.metadata["exploration_bonus_mode"] == "none"
    assert checkpoint.metadata["rnd_enabled"] is False
    assert (
        checkpoint.metadata["compact_coach_compatibility_gate_death_terminal_contract"]
        is False
    )
    assert checkpoint.metadata["death_terminal_contract"] is False
    assert checkpoint.metadata["death_terminal_contract_status"] == (
        "profile_no_death_terminal_nstep_only"
    )
    assert (
        checkpoint.metadata["compact_death_terminal_contract_promotion_gate_satisfied"]
        is False
    )
    assert checkpoint.metadata["death_mode"] == "profile_no_death"
    assert checkpoint.metadata["profile_only_terminal_contract"] is True
    assert checkpoint.metadata["normal_collision_death_supported"] is False
    assert checkpoint.metadata["truncated_supported"] is False

    export = trainer.save_stock_eval_export(
        checkpoint_id="unit-compact-owned-ckpt-export-0002",
        replay_store_state=replay_state,
        path=tmp_path / "iteration_2.pth.tar",
        policy_metadata=build_current_policy_observation_sidecar_v1(),
    )
    assert export["checkpoint_path"].is_file()
    assert export["sidecar_path"].is_file()
    assert export["payload"]["metadata"]["source_compact_checkpoint_id"] == (
        "unit-compact-owned-ckpt-export-0002"
    )
    assert export["payload"]["metadata"]["stock_payload_mapping"] is True
    assert export["payload"]["metadata"]["stock_eval_tournament_loadable"] is False


def test_compact_owned_trainer_accepts_prebuilt_learner_batch() -> None:
    torch = _torch_module()
    torch.manual_seed(20260531)
    sample = _resident_sample()
    config = CompactMuZeroLearnerConfigV1(device="cpu")
    learner = CompactMuZeroLearnerEdgeV1(
        model=_TinyMuZeroModel(),
        config=config,
    )
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-compact-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
        ),
        learner=learner,
    )
    learner_batch = build_compact_muzero_learner_batch_v1(sample, config=config)

    result = trainer.train_on_learner_batch(learner_batch, train_steps=1)

    assert result.policy_version_ref == "unit-policy:update-1"
    assert result.telemetry["compact_muzero_learner_prebuilt_batch_used"] is True
    assert trainer.learner_update_count == 1
    assert trainer.sample_batch_count == 1


def test_compact_owned_trainer_record_step_updates_policy_and_rejects_stale_rows():
    torch = _torch_module()
    learner = _FakeCheckpointLearner(torch)
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-policy",
        policy_source="unit_test_compact_owned_trainer",
        model_version_ref="unit-model",
    )
    ring = _CompactReplayRingV1(
        capacity=4,
        metadata=compact_owned_loop_replay_store_metadata(
            policy,
            extra={
                "resident_device_sample_batch": True,
                "device_replay_index_rows_sample": True,
                "search_impl": "unit_compact_torch_search",
                "num_simulations": 8,
                "active_root_count": 2,
                "root_batch_schema_id": "curvyzero_compact_root_batch/v1",
                "search_result_schema_id": "curvyzero_compact_search_result/v1",
                "replay_payload_schema_id": (
                    "curvyzero_compact_device_search_replay_payload/v1"
                ),
                "search_replay_payload_digest": "unit-digest-abc123",
            },
        ),
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(),
        policy_version=policy,
        replay_store=ring,
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0]))
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-loop-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
        ),
        learner=learner,
        loop=loop,
    )

    trainer.record_step(
        current_step=_make_step(actions=[1, 2], rewards=[3.0, 4.0]),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert trainer.learner_update_count == 1
    assert trainer.policy_refresh_count == 1
    assert trainer.policy_version_ref == "unit-policy:update-1"
    assert loop.policy_version.policy_version_ref == "unit-policy:update-1"
    assert trainer.last_record_index == 7

    with pytest.raises(ValueError, match="stale record_index"):
        trainer.record_step(
            current_step=_make_step(actions=[2, 0], rewards=[5.0, 6.0]),
            index_rows=_make_rows(record_index=7, actions=[2, 0], rewards=[5.0, 6.0]),
        )


def test_compact_owned_trainer_consumes_deferred_loop_learner_result():
    torch = _torch_module()
    learner = _FakeCheckpointLearner(torch)
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-policy",
        policy_source="unit_test_compact_owned_trainer",
        model_version_ref="unit-model",
    )
    ring = _CompactReplayRingV1(
        capacity=4,
        metadata=compact_owned_loop_replay_store_metadata(policy),
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(defer_learner_gate=True),
        policy_version=policy,
        replay_store=ring,
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0]))
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-loop-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
        ),
        learner=learner,
        loop=loop,
    )

    result = trainer.record_step(
        current_step=_make_step(actions=[1, 2], rewards=[3.0, 4.0]),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert result.sampled is True
    assert result.trained is False
    assert loop.has_pending_learner_result is True
    assert trainer.sample_batch_count == 1
    assert trainer.learner_update_count == 0

    learner_result = trainer.consume_completed_learner_result(wait=True)

    assert learner_result is not None
    assert loop.has_pending_learner_result is False
    assert loop.deferred_learner_completed_count == 1
    assert trainer.learner_update_count == 1
    assert trainer.policy_refresh_count == 1
    assert trainer.policy_version_ref == "unit-policy:update-1"
    assert loop.policy_version.policy_version_ref == "unit-policy:update-1"


def test_compact_owned_trainer_consumes_deferred_sample_learner_result():
    torch = _torch_module()
    learner = _FakeCheckpointLearner(torch)
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-policy",
        policy_source="unit_test_compact_owned_trainer",
        model_version_ref="unit-model",
    )
    ring = _CompactReplayRingV1(
        capacity=4,
        metadata=compact_owned_loop_replay_store_metadata(policy),
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(defer_sample_learner_gate=True),
        policy_version=policy,
        replay_store=ring,
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0]))
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-loop-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
        ),
        learner=learner,
        loop=loop,
    )

    result = trainer.record_step(
        current_step=_make_step(actions=[1, 2], rewards=[3.0, 4.0]),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    assert result.sampled is False
    assert result.trained is False
    assert trainer.sample_batch_count == 0
    assert trainer.learner_update_count == 0

    drained = trainer.consume_completed_sample_learner_result(wait=True)

    assert drained is not None
    assert drained.sampled is True
    assert drained.trained is True
    assert trainer.sample_batch_count == 1
    assert trainer.learner_update_count == 1
    assert trainer.policy_refresh_count == 1
    assert trainer.policy_version_ref == "unit-policy:update-1"
    assert loop.policy_version.policy_version_ref == "unit-policy:update-1"
    assert loop.deferred_sample_learner_completed_count == 1


def test_compact_owned_trainer_checkpoint_preserves_loop_runtime_state():
    torch = _torch_module()
    learner = _FakeCheckpointLearner(torch)
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-policy",
        policy_source="unit_test_compact_owned_trainer",
        model_version_ref="unit-model",
    )
    ring = _CompactReplayRingV1(
        capacity=4,
        metadata=compact_owned_loop_replay_store_metadata(
            policy,
            extra={
                "resident_device_sample_batch": True,
                "device_replay_index_rows_sample": True,
            },
        ),
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(),
        policy_version=policy,
        replay_store=ring,
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0]))
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-loop-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
        ),
        learner=learner,
        loop=loop,
    )
    trainer.record_step(
        current_step=_make_step(actions=[1, 2], rewards=[3.0, 4.0]),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )
    replay_state = loop.snapshot_replay_store_state()

    checkpoint = trainer.checkpoint(
        checkpoint_id="unit-loop-ckpt-0001",
        replay_store_state=replay_state,
    )

    assert checkpoint.loop_runtime_state is not None
    assert checkpoint.loop_runtime_state.previous_step is not None
    assert checkpoint.loop_runtime_state.counters.learner_gate_updates == 1

    restored_loop = CompactOwnedLoopV1(
        config=_loop_config(),
        policy_version=CompactPolicyVersionRefV1(
            policy_version_ref=trainer.policy_version_ref,
            policy_source="unit_test_compact_owned_trainer",
            model_version_ref=trainer.model_version_ref,
        ),
        replay_store=_CompactReplayRingV1.from_durable_state(
            checkpoint.replay_store_state
        ),
        learner=learner,
    )
    restore_compact_owned_loop_runtime_state_v1(
        restored_loop,
        checkpoint.loop_runtime_state,
    )
    restored_result = restored_loop.record_step(
        current_step=_make_step(actions=[2, 0], rewards=[5.0, 6.0]),
        index_rows=_make_rows(record_index=8, actions=[2, 0], rewards=[5.0, 6.0]),
    )

    assert restored_result.appended_replay_rows is True
    assert restored_loop.sample_gate_calls == 2
    assert restored_loop.learner_gate_updates == 2


def test_compact_owned_trainer_checkpoint_can_carry_metrics_lineage_contract():
    torch = _torch_module()
    learner = _FakeCheckpointLearner(torch)
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-policy",
        policy_source="unit_test_compact_owned_trainer",
        model_version_ref="unit-model",
    )
    ring = _CompactReplayRingV1(
        capacity=4,
        metadata=compact_owned_loop_replay_store_metadata(
            policy,
            extra={
                "resident_device_sample_batch": True,
                "device_replay_index_rows_sample": True,
                "search_impl": "unit_compact_torch_search",
                "num_simulations": 8,
                "active_root_count": 2,
                "root_batch_schema_id": "curvyzero_compact_root_batch/v1",
                "search_result_schema_id": "curvyzero_compact_search_result/v1",
                "replay_payload_schema_id": (
                    "curvyzero_compact_device_search_replay_payload/v1"
                ),
                "search_replay_payload_digest": "unit-digest-abc123",
            },
        ),
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(),
        policy_version=policy,
        replay_store=ring,
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0]))
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-metrics-lineage-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
        ),
        learner=learner,
        loop=loop,
    )
    trainer.record_step(
        current_step=_make_step(actions=[1, 2], rewards=[3.0, 4.0]),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )

    checkpoint = trainer.checkpoint(
        checkpoint_id="unit-metrics-lineage-ckpt",
        replay_store_state=loop.snapshot_replay_store_state(),
        training_metrics_lineage_evidence_refs=(
            "tests/test_compact_owned_trainer.py::"
            "test_compact_owned_trainer_checkpoint_can_carry_metrics_lineage_contract",
        ),
    )

    assert checkpoint.metadata["training_metrics_lineage"] is True
    assert (
        checkpoint.metadata[
            "compact_coach_compatibility_gate_training_metrics_lineage"
        ]
        is True
    )
    assert (
        checkpoint.metadata[
            "compact_coach_compatibility_gate_policy_refresh_handoff"
        ]
        is False
    )
    assert checkpoint.metadata["compact_coach_compatibility_promotion_eligible"] is False
    lineage = checkpoint.metadata["compact_training_metrics_lineage"]
    assert lineage["policy_version_ref"] == trainer.policy_version_ref
    assert lineage["model_version_ref"] == trainer.model_version_ref
    assert lineage["learner_gate_updates"] == 1
    assert lineage["search_provenance"]["search_impl"] == "unit_compact_torch_search"
    assert lineage["search_provenance"]["num_simulations"] == 8
    assert (
        "training_metrics_lineage"
        in checkpoint.metadata["compact_coach_compatibility_evidence"]
    )


def test_compact_owned_trainer_checkpoint_can_carry_policy_refresh_handoff_contract():
    torch = _torch_module()
    torch.manual_seed(20260530)
    learner_model = _TinyMuZeroModel()
    learner = CompactMuZeroLearnerEdgeV1(
        model=learner_model,
        config=CompactMuZeroLearnerConfigV1(device="cpu"),
    )
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-policy-refresh-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
        ),
        learner=learner,
    )
    before_digest = compact_model_state_digest_v1(trainer.model)
    trainer.train_on_sample_batch(_resident_sample(), train_steps=1)
    learner_digest = compact_model_state_digest_v1(trainer.model)
    assert learner_digest != before_digest

    search_worker_model = _TinyMuZeroModel()
    search_service = CompactTorchSearchServiceV1(
        policy=_FakePolicy(search_worker_model),
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
    search_worker_state = search_service.refresh_model_state(
        model_state_dict=trainer.model.state_dict(),
        policy_version_ref=trainer.policy_version_ref,
        model_version_ref=trainer.model_version_ref,
        policy_source=trainer.config.policy_source,
        learner_update_count=trainer.learner_update_count,
        expected_model_state_digest=learner_digest,
    )
    row_metadata = compact_policy_refresh_metadata_from_state_v1(
        search_worker_state
    )
    policy_rows = _make_rows(
        record_index=21,
        actions=[1, 2],
        rewards=[3.0, 4.0],
        extra_metadata=row_metadata,
    )
    ring = _CompactReplayRingV1(capacity=2)
    ring.append(
        previous_step=_make_step(actions=[0, 1], rewards=[1.0, 2.0]),
        current_step=_make_step(actions=[1, 2], rewards=[3.0, 4.0]),
        index_rows=policy_rows,
    )
    sample_metadata = ring.sample(
        seed=11,
        sample_batch_size=1,
        require_next_targets=False,
        num_unroll_steps=1,
    )["sample_batch"].metadata
    contract = build_compact_policy_refresh_handoff_v1(
        checkpoint_id="unit-policy-refresh-ckpt",
        resume_state=trainer.resume_state(),
        learner_model=trainer.model,
        search_worker_state=search_worker_state,
        root_metadata=row_metadata,
        action_metadata=row_metadata,
        replay_metadata=policy_rows.metadata,
        sample_metadata=sample_metadata,
        evidence_refs=(
            "tests/test_compact_owned_trainer.py::"
            "test_compact_owned_trainer_checkpoint_can_carry_policy_refresh_handoff_contract",
        ),
    )
    checkpoint = trainer.checkpoint(
        checkpoint_id="unit-policy-refresh-ckpt",
        replay_store_state=_owned_replay_state(
            policy_version_ref=trainer.policy_version_ref,
            model_version_ref=trainer.model_version_ref,
        ),
        policy_refresh_handoff=contract,
    )

    assert checkpoint.metadata["policy_refresh_handoff"] is True
    assert (
        checkpoint.metadata["compact_coach_compatibility_gate_policy_refresh_handoff"]
        is True
    )
    assert (
        checkpoint.metadata[
            "compact_coach_compatibility_gate_training_metrics_lineage"
        ]
        is False
    )
    assert checkpoint.metadata["compact_coach_compatibility_promotion_eligible"] is False
    lineage = checkpoint.metadata["compact_policy_refresh_handoff"]
    assert lineage["learner_model_state_digest"] == learner_digest
    assert lineage["policy_version_ref"] == trainer.policy_version_ref
    assert lineage["model_version_ref"] == trainer.model_version_ref
    assert lineage["search_worker_distinct_from_learner"] is True
    assert (
        "policy_refresh_handoff"
        in checkpoint.metadata["compact_coach_compatibility_evidence"]
    )


def test_compact_owned_trainer_rejects_rnd_enabled_config():
    torch = _torch_module()
    learner = _FakeCheckpointLearner(torch)

    with pytest.raises(ValueError, match="does not support RND"):
        CompactOwnedTrainerV1(
            config=CompactOwnedTrainerConfigV1(
                trainer_id="unit-rnd-overclaim-trainer",
                policy_source="unit_test_compact_owned_trainer",
                initial_policy_version_ref="unit-policy",
                initial_model_version_ref="unit-model",
                exploration_bonus_config=normalize_exploration_bonus_spec(
                    mode="rnd_meter_v0",
                ).as_dict(),
            ),
            learner=learner,
        )


def test_compact_owned_trainer_rejects_normal_death_until_proven():
    torch = _torch_module()
    learner = _FakeCheckpointLearner(torch)

    with pytest.raises(ValueError, match="normal_collision_evidence"):
        CompactOwnedTrainerV1(
            config=CompactOwnedTrainerConfigV1(
                trainer_id="unit-normal-death-overclaim-trainer",
                policy_source="unit_test_compact_owned_trainer",
                initial_policy_version_ref="unit-policy",
                initial_model_version_ref="unit-model",
                death_mode=DEATH_MODE_NORMAL,
            ),
            learner=learner,
        )


def test_compact_owned_trainer_checkpoint_consumes_payload_derived_normal_death():
    torch = _torch_module()
    learner = _FakeCheckpointLearner(torch)
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-normal-death-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
            death_mode=DEATH_MODE_NORMAL,
            normal_collision_death_profile_result=_normal_death_profile_payload(),
            normal_collision_death_evidence_id="unit-owned-trainer-normal-death-profile",
            normal_collision_death_evidence_refs=(
                "tests/test_source_state_hybrid_observation_profile.py::"
                "test_hybrid_compact_native_path_preserves_normal_collision_death_fixture",
                "tests/test_compact_owned_trainer.py::"
                "test_compact_owned_trainer_checkpoint_consumes_payload_derived_normal_death",
            ),
        ),
        learner=learner,
    )
    replay_state = _owned_replay_state(
        policy_version_ref=trainer.policy_version_ref,
        model_version_ref=trainer.model_version_ref,
    )

    checkpoint = trainer.checkpoint(
        checkpoint_id="unit-normal-death-compact-owned-ckpt",
        replay_store_state=replay_state,
        metrics={"unit_metric": 1.0},
    )

    assert trainer.metadata["death_terminal_contract"] is True
    assert checkpoint.metadata["death_terminal_contract"] is True
    assert (
        checkpoint.metadata["compact_coach_compatibility_gate_death_terminal_contract"]
        is True
    )
    assert checkpoint.metadata["compact_coach_compatibility_promotion_eligible"] is False
    assert checkpoint.metadata["promotion_claim"] is False
    assert checkpoint.metadata["normal_collision_death_supported"] is True
    assert checkpoint.metadata["death_mode"] == DEATH_MODE_NORMAL
    assert (
        checkpoint.metadata["compact_death_terminal_contract"][
            "normal_collision_death_evidence_id"
        ]
        == "unit-owned-trainer-normal-death-profile"
    )


def test_compact_owned_trainer_normal_death_loop_checkpoint_roundtrip_restores_runtime_state(
    tmp_path,
):
    torch = _torch_module()
    learner = _FakeCheckpointLearner(torch)
    policy = CompactPolicyVersionRefV1(
        policy_version_ref="unit-policy",
        policy_source="unit_test_compact_owned_trainer",
        model_version_ref="unit-model",
    )
    ring = _CompactReplayRingV1(
        capacity=4,
        metadata=compact_owned_loop_replay_store_metadata(policy),
    )
    loop = CompactOwnedLoopV1(
        config=_loop_config(),
        policy_version=policy,
        replay_store=ring,
        learner=learner,
    )
    loop.prime_previous_step(_make_step(actions=[0, 1], rewards=[1.0, 2.0]))
    trainer = CompactOwnedTrainerV1(
        config=CompactOwnedTrainerConfigV1(
            trainer_id="unit-normal-death-loop-trainer",
            policy_source="unit_test_compact_owned_trainer",
            initial_policy_version_ref="unit-policy",
            initial_model_version_ref="unit-model",
            death_mode=DEATH_MODE_NORMAL,
            normal_collision_death_profile_result=_normal_death_profile_payload(),
            normal_collision_death_evidence_id=(
                "unit-normal-death-loop-checkpoint-profile"
            ),
            normal_collision_death_evidence_refs=(
                "source_collision_head_head_reverse_order_single_death_step.json",
                "tests/test_compact_owned_trainer.py::"
                "test_compact_owned_trainer_normal_death_loop_checkpoint_roundtrip_restores_runtime_state",
            ),
        ),
        learner=learner,
        loop=loop,
    )
    trainer.record_step(
        current_step=_make_step(actions=[1, 2], rewards=[3.0, 4.0]),
        index_rows=_make_rows(record_index=7, actions=[1, 2], rewards=[3.0, 4.0]),
    )
    checkpoint_path = trainer.save_checkpoint(
        checkpoint_id="unit-normal-death-loop-ckpt",
        replay_store_state=loop.snapshot_replay_store_state(),
        path=tmp_path / "normal-death-loop.pt",
    )

    loaded = load_compact_trainer_checkpoint_v1(checkpoint_path)

    assert loaded.metadata["death_terminal_contract"] is True
    assert loaded.metadata["compact_coach_compatibility_gate_death_terminal_contract"] is True
    assert loaded.metadata["normal_collision_death_supported"] is True
    assert loaded.metadata["death_mode"] == DEATH_MODE_NORMAL
    assert loaded.metadata["promotion_claim"] is False
    assert loaded.metadata["compact_coach_compatibility_promotion_eligible"] is False
    assert loaded.loop_runtime_state is not None

    restored_loop = CompactOwnedLoopV1(
        config=_loop_config(),
        policy_version=CompactPolicyVersionRefV1(
            policy_version_ref=trainer.policy_version_ref,
            policy_source="unit_test_compact_owned_trainer",
            model_version_ref=trainer.model_version_ref,
        ),
        replay_store=_CompactReplayRingV1.from_durable_state(loaded.replay_store_state),
        learner=learner,
    )
    restore_compact_owned_loop_runtime_state_v1(
        restored_loop,
        loaded.loop_runtime_state,
    )
    restored_result = restored_loop.record_step(
        current_step=_make_step(actions=[2, 0], rewards=[5.0, 6.0]),
        index_rows=_make_rows(record_index=8, actions=[2, 0], rewards=[5.0, 6.0]),
    )

    assert restored_result.appended_replay_rows is True
    assert restored_loop.sample_gate_calls == 2
    assert restored_loop.learner_gate_updates == 2


def _loop_config(
    *,
    defer_learner_gate: bool = False,
    defer_sample_learner_gate: bool = False,
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
        capture_replay_store_state=True,
        defer_learner_gate=bool(defer_learner_gate),
        defer_sample_learner_gate=bool(defer_sample_learner_gate),
    )


def _owned_replay_state(*, policy_version_ref: str, model_version_ref: str):
    policy = CompactPolicyVersionRefV1(
        policy_version_ref=policy_version_ref,
        policy_source="unit_test_compact_owned_trainer",
        model_version_ref=model_version_ref,
    )
    metadata = compact_owned_loop_replay_store_metadata(policy)
    ring = _CompactReplayRingV1(capacity=2, metadata=metadata)
    return ring.snapshot_durable_state(
        policy_version_ref=policy.policy_version_ref,
        policy_source=policy.policy_source,
        model_version_ref=policy.model_version_ref,
        metadata=metadata,
    )


def _normal_death_profile_evidence():
    return build_normal_collision_death_evidence_from_profile_result_v1(
        _normal_death_profile_payload(),
        evidence_id="unit-owned-trainer-normal-death-profile",
        evidence_refs=[
            "tests/test_source_state_hybrid_observation_profile.py::"
            "test_hybrid_compact_native_path_preserves_normal_collision_death_fixture",
            "tests/test_compact_owned_trainer.py::"
            "test_compact_owned_trainer_checkpoint_consumes_payload_derived_normal_death",
        ],
    )


def _normal_death_profile_payload():
    return {
        "death_mode": DEATH_MODE_NORMAL,
        "terminal_row_count": 4,
        "done_semantics_verified": True,
        "terminated_row_count": 4,
        "truncated_row_count": 0,
        "death_row_count": 4,
        "death_count_total": 4,
        "death_cause_count_by_name": {
            "wall": 0,
            "own_trail": 0,
            "opponent_trail": 4,
            "body_unknown": 0,
        },
        "normal_collision_death_causes": ["opponent_trail"],
        "normal_collision_death_hit_owner_present": True,
        "normal_collision_death_evidence_rows": [
            {
                "global_row": 3,
                "done": True,
                "terminated": True,
                "truncated": False,
                "terminal_reason": 1,
                "death_count": 1,
                "death_player": [0, -1],
                "death_cause": ["opponent_trail"],
                "death_hit_owner": [1, -1],
                "winner": 1,
                "draw": False,
                "reward": [-1.0, 1.0],
                "final_reward_map": [-1.0, 1.0],
                "final_reward_map_matches_reward": True,
                "final_observation_row": True,
            }
        ],
        "terminal_final_observation_row_count": 4,
        "terminal_final_observation_before_autoreset_verified": True,
        "terminal_final_reward_map_row_count": 4,
        "terminal_final_reward_map_matches_reward_row_count": 4,
        "terminal_final_reward_map_verified": True,
        "compact_rollout_slab_sample_gate_last_telemetry": {
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            "compact_rollout_slab_sample_gate_terminal_sample_row_count": 4,
            "compact_rollout_slab_sample_gate_next_final_observation_row_count": 4,
            "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used": True,
            "compact_rollout_slab_sample_gate_device_replay_index_rows": True,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count": 4,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode": (
                COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP
            ),
        },
        "compact_rollout_slab_learner_gate_last_telemetry": {
            "compact_rollout_slab_learner_gate_compact_muzero_telemetry": {
                "compact_muzero_learner_done_count": 4,
                "compact_muzero_learner_truncated_count": 0,
                "compact_muzero_learner_value_valid_count": 8,
            },
        },
    }


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

            def initial_inference(self, obs):
                from lzero.model.common import MZNetworkOutput

                latent = self.encoder(obs)
                return MZNetworkOutput(
                    self.value_head(latent),
                    torch.zeros_like(self.value_head(latent)),
                    self.policy_head(latent),
                    latent,
                )

            def recurrent_inference(self, latent_state, action):
                from lzero.model.common import MZNetworkOutput

                next_latent = torch.tanh(
                    latent_state + self.action_embedding(action.reshape(-1).long())
                )
                return MZNetworkOutput(
                    self.value_head(next_latent),
                    self.reward_head(next_latent),
                    self.policy_head(next_latent),
                    next_latent,
                )

        return Tiny()


class _FakePolicy:
    def __init__(self, model) -> None:
        self._model = model
        self._cfg = SimpleNamespace(
            pb_c_base=19652,
            pb_c_init=1.25,
            discount_factor=0.997,
            root_noise_weight=0.0,
            root_dirichlet_alpha=0.3,
            value_delta_max=0.01,
        )


def _resident_sample():
    torch = _torch_module()
    actions = torch.tensor([0, 1, 2], dtype=torch.int16)
    policy_target = torch.zeros((3, ACTION_COUNT), dtype=torch.float32)
    next_policy_target = torch.zeros((3, ACTION_COUNT), dtype=torch.float32)
    policy_target[torch.arange(3), torch.tensor([0, 1, 2])] = 1.0
    next_policy_target[torch.arange(3), torch.tensor([1, 2, 0])] = 1.0
    return SimpleNamespace(
        metadata={
            "resident_device_sample_batch": True,
            "device_replay_index_rows_sample": True,
        },
        observation=torch.arange(3 * 4 * 8 * 8, dtype=torch.uint8).reshape(3, 4, 8, 8),
        action=actions,
        action_mask=torch.ones((3, ACTION_COUNT), dtype=torch.bool),
        policy_target=policy_target,
        root_value=torch.tensor([0.0, 0.5, -0.5], dtype=torch.float32),
        reward=torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32),
        weights=torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32),
        next_action_mask=torch.ones((3, ACTION_COUNT), dtype=torch.bool),
        next_policy_target=next_policy_target,
        next_root_value=torch.tensor([0.25, 0.0, -0.25], dtype=torch.float32),
    )


class _FakeCheckpointLearner:
    def __init__(self, torch) -> None:
        self.model = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)

    def train_on_sample_batch(self, sample_batch, *, train_steps: int):
        del sample_batch
        return {
            "compact_rollout_slab_learner_gate_updates": int(train_steps),
            "compact_rollout_slab_learner_gate_sample_rows": 2,
            "compact_rollout_slab_learner_gate_input_bytes": 0,
            "compact_rollout_slab_learner_gate_sec": 0.01,
            "compact_muzero_learner_calls_train_muzero": False,
            "compact_muzero_learner_update_claim": True,
            "compact_muzero_learner_loss": 1.25,
            "compact_muzero_learner_policy_loss": 0.25,
            "compact_muzero_learner_value_loss": 0.5,
            "compact_muzero_learner_reward_loss": 0.75,
            "compact_muzero_learner_grad_norm_before_clip": 0.125,
            "compact_muzero_learner_sample_rows": 2,
            "compact_muzero_learner_train_steps": int(train_steps),
            "compact_muzero_learner_input_bytes": 0,
            "compact_muzero_learner_resident_sample_used": True,
            "compact_muzero_learner_device_replay_index_rows_sample": True,
        }


def _make_step(*, actions: list[int], rewards: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        observation=np.zeros((2, 1, 4, 64, 64), dtype=np.float32),
        action_mask=np.ones((2, 1, ACTION_COUNT), dtype=np.bool_),
        reward=np.asarray([[rewards[0]], [rewards[1]]], dtype=np.float32),
        final_reward_map=np.asarray([[rewards[0]], [rewards[1]]], dtype=np.float32),
        done=np.asarray([False, False], dtype=np.bool_),
        payload={
            "joint_action": np.asarray([[actions[0]], [actions[1]]], dtype=np.int16),
        },
        compact_batch=None,
    )


def _make_rows(
    *,
    record_index: int,
    actions: list[int],
    rewards: list[float],
    extra_metadata: dict[str, object] | None = None,
) -> CompactReplayIndexRowsV1:
    metadata = {"policy_version_ref": "unit-policy"}
    metadata.update(dict(extra_metadata or {}))
    return CompactReplayIndexRowsV1(
        metadata=metadata,
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
        policy_source="unit_test_compact_owned_trainer",
    )
