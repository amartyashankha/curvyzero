from __future__ import annotations

from collections.abc import Mapping

import pytest

from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import compact_owned_loop_replay_store_metadata
from curvyzero.training.compact_trainer_checkpoint import (
    CompactTrainerResumeStateV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    build_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    load_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    restore_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    save_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    validate_compact_trainer_checkpoint_v1,
)
from curvyzero.env.vector_runtime import DEATH_MODE_NORMAL
from curvyzero.training.compact_death_terminal_contract import (
    COMPACT_TERMINAL_VALUE_TARGET_MODE_STOCK_NO_BOOTSTRAP,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_normal_collision_death_evidence_from_profile_result_v1,
)
from curvyzero.training.exploration_bonus import normalize_exploration_bonus_spec
from curvyzero.training.source_state_hybrid_observation_profile import (
    _CompactReplayRingV1,
)


def _torch_module():
    return pytest.importorskip("torch")


def test_compact_trainer_checkpoint_roundtrip_required_fields(tmp_path):
    torch = _torch_module()
    torch.manual_seed(20260528)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.ones((2, 3))).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    replay_state = _owned_replay_state(
        policy_version_ref="unit-policy:update-3",
        model_version_ref="unit-model:update-3",
    )
    resume = CompactTrainerResumeStateV1(
        trainer_id="unit-compact-trainer",
        train_step=2,
        learner_update_count=3,
        sample_batch_count=2,
        policy_version_ref="unit-policy:update-3",
        model_version_ref="unit-model:update-3",
        policy_source="unit_test_compact_trainer_checkpoint",
        loop_counters={"sample_gate_calls": 2},
    )

    checkpoint = build_compact_trainer_checkpoint_v1(
        checkpoint_id="unit-compact-ckpt-0003",
        trainer_config={"schema_id": "unit_config"},
        resume_state=resume,
        model=model,
        optimizer=optimizer,
        replay_store_state=replay_state,
        metrics={"loss": 1.25},
    )

    metadata = checkpoint.metadata
    assert metadata["compact_trainer_checkpoint"] is True
    assert metadata["checkpoint_save_load"] is True
    assert metadata["resume_metadata"] is True
    assert metadata["stock_eval_tournament_loadable"] is False
    assert metadata["compact_eval_adapter_required"] is True
    assert metadata["calls_train_muzero"] is False
    assert metadata["promotion_claim"] is False
    assert metadata["compact_coach_compatibility_gate_checkpoint_save_load"] is True
    assert metadata["compact_coach_compatibility_gate_resume_metadata"] is True
    assert metadata["compact_coach_compatibility_gate_reward_rnd_contract"] is True
    assert metadata["compact_coach_compatibility_gate_death_terminal_contract"] is False
    assert metadata["compact_coach_compatibility_gate_eval_gif_tournament_load"] is False
    assert metadata["compact_coach_compatibility_gate_policy_refresh_handoff"] is False
    assert metadata["policy_refresh_handoff"] is False
    assert metadata["compact_policy_refresh_handoff_verified"] is False
    assert metadata["compact_policy_refresh_handoff"] is None
    assert metadata["compact_coach_compatibility_gate_training_metrics_lineage"] is False
    assert metadata["training_metrics_lineage"] is False
    assert metadata["compact_training_metrics_lineage_verified"] is False
    assert metadata["compact_training_metrics_lineage"] is None
    assert metadata["compact_coach_compatibility_promotion_eligible"] is False
    assert metadata["reward_rnd_contract"] is True
    assert metadata["reward_variant"] == "survival_plus_bonus_no_outcome"
    assert metadata["reward_target_effect"] == "extrinsic_reward_only"
    assert metadata["exploration_bonus_mode"] == "none"
    assert metadata["exploration_bonus_enabled"] is False
    assert metadata["rnd_enabled"] is False
    assert metadata["rnd_state_required"] is False
    assert metadata["rnd_state_dict_present"] is False
    assert metadata["death_terminal_contract"] is False
    assert metadata["death_terminal_contract_status"] == (
        "profile_no_death_terminal_nstep_only"
    )
    assert (
        metadata["compact_death_terminal_contract_promotion_gate_satisfied"]
        is False
    )
    assert metadata["compact_death_terminal_contract_blocker"] == (
        "normal_collision_death_not_proven"
    )
    assert metadata["death_mode"] == "profile_no_death"
    assert metadata["profile_only_terminal_contract"] is True
    assert metadata["normal_collision_death_supported"] is False
    assert metadata["profile_no_death_supported"] is True
    assert metadata["max_ticks_terminal_supported"] is True
    assert metadata["truncated_supported"] is False
    assert metadata["terminal_unroll_value_target_mode"] == (
        "stock_terminal_no_bootstrap_return_discount_1.0"
    )
    assert (
        metadata["compact_coach_compatibility_evidence"]["reward_rnd_contract"]
    )
    assert "death_terminal_contract" not in metadata["compact_coach_compatibility_evidence"]
    assert (
        "training_metrics_lineage"
        not in metadata["compact_coach_compatibility_evidence"]
    )
    assert (
        metadata["compact_coach_compatibility_evidence"][
            "training_metrics_lineage_partial_metrics_present"
        ]
        == "metrics_present:unit-compact-ckpt-0003"
    )

    path = save_compact_trainer_checkpoint_v1(checkpoint, tmp_path / "compact.pt")
    loaded = load_compact_trainer_checkpoint_v1(path)
    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    restored_replay = restore_compact_trainer_checkpoint_v1(
        loaded,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    for left, right in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.allclose(left.detach(), right.detach())
    assert restored_optimizer.state_dict()["state"]
    assert restored_replay.metadata["policy_version_ref"] == "unit-policy:update-3"
    assert restored_replay.metadata["compact_owned_loop_replay_store_owned"] is True


def test_compact_trainer_checkpoint_is_not_stock_lightzero_mapping_payload():
    torch = _torch_module()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    replay_state = _owned_replay_state(
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
    )
    resume = CompactTrainerResumeStateV1(
        trainer_id="unit-compact-trainer",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
        policy_source="unit_test_compact_trainer_checkpoint",
        loop_counters={},
    )

    checkpoint = build_compact_trainer_checkpoint_v1(
        checkpoint_id="unit-compact-ckpt-0001",
        trainer_config={"schema_id": "unit_config"},
        resume_state=resume,
        model=model,
        optimizer=optimizer,
        replay_store_state=replay_state,
    )

    assert not isinstance(checkpoint, Mapping)
    assert checkpoint.metadata["stock_eval_tournament_loadable"] is False
    assert checkpoint.metadata["stock_eval_tournament_load_status"] == "adapter_missing"


def test_compact_trainer_checkpoint_rejects_rnd_state_without_rnd_contract():
    torch = _torch_module()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    replay_state = _owned_replay_state(
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
    )
    resume = CompactTrainerResumeStateV1(
        trainer_id="unit-compact-trainer",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
        policy_source="unit_test_compact_trainer_checkpoint",
        loop_counters={},
    )

    with pytest.raises(ValueError, match="rnd_state_dict"):
        build_compact_trainer_checkpoint_v1(
            checkpoint_id="unit-compact-ckpt-rnd-overclaim",
            trainer_config={"schema_id": "unit_config"},
            resume_state=resume,
            model=model,
            optimizer=optimizer,
            replay_store_state=replay_state,
            rnd_state_dict={"predictor": {}},
        )
    with pytest.raises(ValueError, match="does not support RND"):
        build_compact_trainer_checkpoint_v1(
            checkpoint_id="unit-compact-ckpt-rnd-enabled",
            trainer_config={"schema_id": "unit_config"},
            resume_state=resume,
            model=model,
            optimizer=optimizer,
            replay_store_state=replay_state,
            exploration_bonus_config=normalize_exploration_bonus_spec(
                mode="rnd_meter_v0",
            ).as_dict(),
        )


def test_compact_trainer_checkpoint_rejects_normal_death_until_proven():
    torch = _torch_module()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    replay_state = _owned_replay_state(
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
    )
    resume = CompactTrainerResumeStateV1(
        trainer_id="unit-compact-trainer",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
        policy_source="unit_test_compact_trainer_checkpoint",
        loop_counters={},
    )

    with pytest.raises(ValueError, match="normal_collision_evidence"):
        build_compact_trainer_checkpoint_v1(
            checkpoint_id="unit-compact-ckpt-normal-death-overclaim",
            trainer_config={"schema_id": "unit_config"},
            resume_state=resume,
            model=model,
            optimizer=optimizer,
            replay_store_state=replay_state,
            death_mode=DEATH_MODE_NORMAL,
        )


def test_compact_trainer_checkpoint_accepts_payload_derived_normal_death_evidence(tmp_path):
    torch = _torch_module()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    replay_state = _owned_replay_state(
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
    )
    resume = CompactTrainerResumeStateV1(
        trainer_id="unit-compact-trainer-normal-death",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
        policy_source="unit_test_compact_trainer_checkpoint",
        loop_counters={},
    )
    evidence = _normal_death_profile_evidence()

    checkpoint = build_compact_trainer_checkpoint_v1(
        checkpoint_id="unit-compact-ckpt-normal-death",
        trainer_config={"schema_id": "unit_config"},
        resume_state=resume,
        model=model,
        optimizer=optimizer,
        replay_store_state=replay_state,
        metrics={"loss": 1.25},
        death_mode=DEATH_MODE_NORMAL,
        normal_collision_death_profile_result=_normal_death_profile_payload(),
        normal_collision_death_evidence_id="unit-checkpoint-normal-death-profile",
        normal_collision_death_evidence_refs=[
            "tests/test_source_state_hybrid_observation_profile.py::"
            "test_hybrid_compact_native_path_preserves_normal_collision_death_fixture",
            "tests/test_compact_trainer_checkpoint.py::"
            "test_compact_trainer_checkpoint_accepts_payload_derived_normal_death_evidence",
        ],
    )

    metadata = checkpoint.metadata
    assert metadata["death_terminal_contract"] is True
    assert metadata["compact_coach_compatibility_gate_death_terminal_contract"] is True
    assert metadata["compact_coach_compatibility_promotion_eligible"] is False
    assert metadata["promotion_claim"] is False
    assert metadata["death_mode"] == DEATH_MODE_NORMAL
    assert metadata["death_terminal_contract_status"] == (
        "normal_collision_death_terminal_nstep_v1"
    )
    assert metadata["normal_collision_death_supported"] is True
    assert metadata["compact_death_terminal_contract_blocker"] == ""
    assert metadata["compact_death_terminal_contract"]["normal_collision_death_evidence"] == evidence
    assert (
        "death_terminal_contract"
        in metadata["compact_coach_compatibility_evidence"]
    )
    assert "normal_death=true" in metadata["compact_coach_compatibility_evidence"][
        "death_terminal_contract"
    ]

    path = save_compact_trainer_checkpoint_v1(
        checkpoint,
        tmp_path / "compact-normal-death.pt",
    )
    loaded = load_compact_trainer_checkpoint_v1(path)
    assert loaded.metadata["death_terminal_contract"] is True
    validate_compact_trainer_checkpoint_v1(loaded)


def test_compact_trainer_checkpoint_rejects_protected_extra_metadata_overrides():
    torch = _torch_module()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    replay_state = _owned_replay_state(
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
    )
    resume = CompactTrainerResumeStateV1(
        trainer_id="unit-compact-trainer",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
        policy_source="unit_test_compact_trainer_checkpoint",
        loop_counters={},
    )

    with pytest.raises(ValueError, match="protected metadata keys"):
        build_compact_trainer_checkpoint_v1(
            checkpoint_id="unit-compact-ckpt-extra-overclaim",
            trainer_config={"schema_id": "unit_config"},
            resume_state=resume,
            model=model,
            optimizer=optimizer,
            replay_store_state=replay_state,
            extra_metadata={
                "compact_coach_compatibility_gate_death_terminal_contract": True,
            },
        )

    with pytest.raises(ValueError, match="protected metadata keys"):
        build_compact_trainer_checkpoint_v1(
            checkpoint_id="unit-compact-ckpt-extra-death-overclaim",
            trainer_config={"schema_id": "unit_config"},
            resume_state=resume,
            model=model,
            optimizer=optimizer,
            replay_store_state=replay_state,
            extra_metadata={"death_terminal_contract": True},
        )

    with pytest.raises(ValueError, match="protected metadata keys"):
        build_compact_trainer_checkpoint_v1(
            checkpoint_id="unit-compact-ckpt-extra-metrics-overclaim",
            trainer_config={"schema_id": "unit_config"},
            resume_state=resume,
            model=model,
            optimizer=optimizer,
            replay_store_state=replay_state,
            extra_metadata={"training_metrics_lineage": True},
        )

    with pytest.raises(ValueError, match="protected metadata keys"):
        build_compact_trainer_checkpoint_v1(
            checkpoint_id="unit-compact-ckpt-extra-policy-overclaim",
            trainer_config={"schema_id": "unit_config"},
            resume_state=resume,
            model=model,
            optimizer=optimizer,
            replay_store_state=replay_state,
            extra_metadata={"policy_refresh_handoff": True},
        )

    with pytest.raises(ValueError, match="either normal_collision_death_evidence"):
        build_compact_trainer_checkpoint_v1(
            checkpoint_id="unit-compact-ckpt-duplicate-normal-death-inputs",
            trainer_config={"schema_id": "unit_config"},
            resume_state=resume,
            model=model,
            optimizer=optimizer,
            replay_store_state=replay_state,
            death_mode=DEATH_MODE_NORMAL,
            normal_collision_death_evidence=_normal_death_profile_evidence(),
            normal_collision_death_profile_result=_normal_death_profile_payload(),
        )


def test_replay_store_snapshot_is_not_trainer_checkpoint():
    replay_state = _owned_replay_state(
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        validate_compact_trainer_checkpoint_v1(replay_state)


def test_compact_trainer_checkpoint_rejects_replay_without_owned_loop_metadata():
    torch = _torch_module()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    replay_state = _CompactReplayRingV1(capacity=2).snapshot_durable_state(
        policy_version_ref="unit-policy:update-1",
        policy_source="unit_test_compact_trainer_checkpoint",
        model_version_ref="unit-model:update-1",
    )
    resume = CompactTrainerResumeStateV1(
        trainer_id="unit-compact-trainer",
        train_step=1,
        learner_update_count=1,
        sample_batch_count=1,
        policy_version_ref="unit-policy:update-1",
        model_version_ref="unit-model:update-1",
        policy_source="unit_test_compact_trainer_checkpoint",
        loop_counters={},
    )

    with pytest.raises(ValueError, match="owned by compact loop"):
        build_compact_trainer_checkpoint_v1(
            checkpoint_id="unit-compact-ckpt-0001",
            trainer_config={"schema_id": "unit_config"},
            resume_state=resume,
            model=model,
            optimizer=optimizer,
            replay_store_state=replay_state,
        )


def _owned_replay_state(*, policy_version_ref: str, model_version_ref: str):
    policy = CompactPolicyVersionRefV1(
        policy_version_ref=policy_version_ref,
        policy_source="unit_test_compact_trainer_checkpoint",
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
        evidence_id="unit-checkpoint-normal-death-profile",
        evidence_refs=[
            "tests/test_source_state_hybrid_observation_profile.py::"
            "test_hybrid_compact_native_path_preserves_normal_collision_death_fixture",
            "tests/test_compact_trainer_checkpoint.py::"
            "test_compact_trainer_checkpoint_accepts_payload_derived_normal_death_evidence",
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
