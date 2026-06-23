from __future__ import annotations

import json
import math

import pytest

from curvyzero.training.compact_owned_loop import CompactPolicyVersionRefV1
from curvyzero.training.compact_owned_loop import compact_owned_loop_replay_store_metadata
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_model_state_digest_v1,
)
from curvyzero.training.compact_coach_speed_row import (
    CompactCoachSpeedRowEvidenceError,
)
from curvyzero.training.compact_coach_speed_row import (
    build_compact_coach_speed_row_evidence_v1,
)
from curvyzero.training.compact_coach_speed_row import (
    compact_coach_speed_row_evidence_ref,
)
from curvyzero.training.compact_coach_speed_row import (
    validate_compact_coach_speed_row_evidence_v1,
)
from curvyzero.training.compact_death_terminal_contract import (
    build_compact_death_terminal_contract_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    CompactTrainerResumeStateV1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    build_compact_trainer_checkpoint_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    save_compact_trainer_checkpoint_v1,
)
from curvyzero.training.source_state_hybrid_observation_profile import (
    _CompactReplayRingV1,
)


def test_compact_coach_speed_row_evidence_binds_manifest_result_and_lifecycle(
    tmp_path,
):
    paths = _speed_row_paths(tmp_path)

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )

    assert evidence["status"] == "compact_coach_speed_row_verified"
    assert evidence["candidate_checkpoint_id"] == "unit-compact-ckpt"
    assert evidence["denominator"]["reported_steps_per_sec"] == pytest.approx(60.0)
    assert evidence["search_config"] == _search_config()
    assert evidence["actor_handoff_config"] == _actor_handoff_config()
    assert evidence["operational_surface"] == _operational_surface()
    assert evidence["result_summary"]["compact_torch_observation_memory_format"] == (
        "channels_last"
    )
    assert evidence["result_summary"]["compact_torch_model_memory_format"] == "contiguous"
    assert compact_coach_speed_row_evidence_ref(evidence).startswith(
        "compact_coach_speed_row:unit-compact-ckpt:"
    )
    validate_compact_coach_speed_row_evidence_v1(evidence)


def test_compact_coach_speed_row_projection_fields_remain_nonproduction(
    tmp_path,
):
    paths = _speed_row_paths(tmp_path)
    result = _read_json(paths["result"])
    projection = _whole_owner_buffer_replay_projection()
    result["summary"].update(projection)
    result["compact"].update(projection)
    _write_json(paths["result"], result)

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )

    assert evidence["denominator"]["speed_currency"] == (
        "compact_trainer_env_steps_per_sec"
    )
    assert evidence["denominator"]["reported_steps_per_sec"] == pytest.approx(60.0)
    validate_compact_coach_speed_row_evidence_v1(evidence)

    bad_dir = tmp_path / "bad_projection_claim"
    bad_dir.mkdir()
    bad_paths = _speed_row_paths(bad_dir)
    bad_result = _read_json(bad_paths["result"])
    bad_projection = _whole_owner_buffer_replay_projection()
    bad_projection[
        "compact_whole_owner_buffer_replay_ceiling_production_speed_claim"
    ] = True
    bad_result["summary"].update(bad_projection)
    bad_result["compact"].update(_whole_owner_buffer_replay_projection())
    _write_json(bad_paths["result"], bad_result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="production_speed_claim"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=bad_paths["lifecycle"],
            manifest_path=bad_paths["manifest"],
            row_id="001",
            result_json_path=bad_paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )

    bad_currency_dir = tmp_path / "bad_projection_currency"
    bad_currency_dir.mkdir()
    bad_currency_paths = _speed_row_paths(bad_currency_dir)
    bad_currency_result = _read_json(bad_currency_paths["result"])
    bad_currency_projection = _whole_owner_buffer_replay_projection()
    bad_currency_projection[
        "compact_whole_owner_buffer_replay_ceiling_speed_currency"
    ] = "compact_trainer_env_steps_per_sec"
    bad_currency_result["summary"].update(bad_currency_projection)
    bad_currency_result["compact"].update(_whole_owner_buffer_replay_projection())
    _write_json(bad_currency_paths["result"], bad_currency_result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="speed_currency"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=bad_currency_paths["lifecycle"],
            manifest_path=bad_currency_paths["manifest"],
            row_id="001",
            result_json_path=bad_currency_paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_requires_unroll2_specialized_builder_proof(
    tmp_path,
):
    surface = _fused_unroll2_operational_surface()
    paths = _speed_row_paths(tmp_path, operational_surface=surface)

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )

    assert evidence["operational_surface"][
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl"
    ] == "unroll2_specialized_v1"
    validate_compact_coach_speed_row_evidence_v1(evidence)

    bad_cases = (
        (
            "used",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used",
            False,
            "unroll2_specialized_builder_used",
        ),
        (
            "call_count",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count",
            0,
            "unroll2_specialized_builder_call_count",
        ),
        (
            "fallback_count",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count",
            1,
            "unroll2_specialized_builder_fallback_count",
        ),
        (
            "fallback_reason",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason",
            "guard_failed",
            "unroll2_specialized_builder_fallback_reason",
        ),
        (
            "impl",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl",
            "generic",
            "unroll2_specialized_builder_impl",
        ),
        (
            "path",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path",
            "generic",
            "unroll2_specialized_builder_path",
        ),
    )
    for label, field, value, error in bad_cases:
        case_dir = tmp_path / label
        case_dir.mkdir()
        bad_surface = dict(surface)
        bad_surface[field] = value
        bad_paths = _speed_row_paths(case_dir, operational_surface=bad_surface)
        with pytest.raises(CompactCoachSpeedRowEvidenceError, match=error):
            build_compact_coach_speed_row_evidence_v1(
                route="compact_owned_trainer",
                candidate_checkpoint_id="unit-compact-ckpt",
                unified_lifecycle_report_path=bad_paths["lifecycle"],
                manifest_path=bad_paths["manifest"],
                row_id="001",
                result_json_path=bad_paths["result"],
                speed_currency="compact_trainer_env_steps_per_sec",
                numerator_field="env_steps_collected",
                denominator_field="training_wall_sec",
            )


def test_compact_coach_speed_row_requires_tensor_native_replay_proof(
    tmp_path,
):
    surface = _fused_tensor_native_operational_surface()
    paths = _speed_row_paths(tmp_path, operational_surface=surface)

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )

    assert evidence["operational_surface"][
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source"
    ] == "maintained_record_table_v1"
    assert evidence["operational_surface"][
        (
            "compact_rollout_slab_sample_gate_learner_batch_builder_"
            "tensor_native_replay_table_build_impl"
        )
    ] == "direct_record_table_v1"
    validate_compact_coach_speed_row_evidence_v1(evidence)

    tensor_native_only_dir = tmp_path / "tensor_native_only"
    tensor_native_only_dir.mkdir()
    tensor_native_only_surface = dict(surface)
    tensor_native_only_surface[
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch"
    ] = False
    tensor_native_only_surface[
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"
    ] = False
    tensor_native_only_surface["compact_rollout_slab_sample_gate_host_provider_learner_batch"] = (
        False
    )
    tensor_native_only_surface[
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source"
    ] = "tensor_native_replay_unroll2_table_gather_v1"
    tensor_native_only_paths = _speed_row_paths(
        tensor_native_only_dir,
        operational_surface=tensor_native_only_surface,
    )
    tensor_native_only_evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=tensor_native_only_paths["lifecycle"],
        manifest_path=tensor_native_only_paths["manifest"],
        row_id="001",
        result_json_path=tensor_native_only_paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    validate_compact_coach_speed_row_evidence_v1(tensor_native_only_evidence)

    fixed_soa_dir = tmp_path / "fixed_soa_tensor_native"
    fixed_soa_dir.mkdir()
    fixed_soa_paths = _speed_row_paths(
        fixed_soa_dir,
        operational_surface=_fixed_soa_tensor_native_operational_surface(),
    )
    fixed_soa_evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=fixed_soa_paths["lifecycle"],
        manifest_path=fixed_soa_paths["manifest"],
        row_id="001",
        result_json_path=fixed_soa_paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    assert fixed_soa_evidence["operational_surface"][
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count"
    ] == 0
    assert fixed_soa_evidence["operational_surface"][
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl"
    ] == "fixed_soa_direct_gather_v1"
    assert fixed_soa_evidence["operational_surface"][
        "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count"
    ] == 0
    assert fixed_soa_evidence["operational_surface"][
        "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count"
    ] == 62
    assert fixed_soa_evidence["operational_surface"][
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift"
    ] is True
    validate_compact_coach_speed_row_evidence_v1(fixed_soa_evidence)

    learner_ready_supersedes_unroll2_dir = tmp_path / "learner_ready_supersedes_unroll2"
    learner_ready_supersedes_unroll2_dir.mkdir()
    learner_ready_supersedes_unroll2_surface = dict(surface)
    learner_ready_supersedes_unroll2_surface.update(
        {
            "compact_muzero_learner_batch_unroll2_specialized_builder": True,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "unroll2_specialized_builder_requested"
            ): True,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "unroll2_specialized_builder_eligible_count"
            ): 0,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "unroll2_specialized_builder_used"
            ): False,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "unroll2_specialized_builder_call_count"
            ): 0,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "unroll2_specialized_builder_fallback_count"
            ): 0,
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "unroll2_specialized_builder_fallback_reason"
            ): "none",
            (
                "compact_rollout_slab_sample_gate_learner_batch_builder_"
                "unroll2_specialized_builder_impl"
            ): "none",
            "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": (
                "learner_ready_unroll2_cache"
            ),
        }
    )
    learner_ready_supersedes_unroll2_paths = _speed_row_paths(
        learner_ready_supersedes_unroll2_dir,
        operational_surface=learner_ready_supersedes_unroll2_surface,
    )
    learner_ready_supersedes_unroll2_evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=learner_ready_supersedes_unroll2_paths[
            "lifecycle"
        ],
        manifest_path=learner_ready_supersedes_unroll2_paths["manifest"],
        row_id="001",
        result_json_path=learner_ready_supersedes_unroll2_paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    validate_compact_coach_speed_row_evidence_v1(
        learner_ready_supersedes_unroll2_evidence
    )

    bad_cases = (
        (
            "learner_ready_config",
            "compact_muzero_learner_batch_learner_ready_unroll2_cache",
            False,
            "compact_muzero_learner_batch_learner_ready_unroll2_cache",
        ),
        (
            "learner_ready_used",
            "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used",
            False,
            "learner_ready_unroll2_cache_used",
        ),
        (
            "fused_config",
            "compact_owned_loop_fused_learner_batch",
            False,
            "fused learner-batch proof requires fused learner-batch config",
        ),
        (
            "used",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used",
            False,
            "tensor_native_replay_used",
        ),
        (
            "call_count",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count",
            0,
            "tensor_native_replay_call_count",
        ),
        (
            "fallback_count",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count",
            1,
            "tensor_native_replay_fallback_count",
        ),
        (
            "fallback_reason",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason",
            "guard_failed",
            "tensor_native_replay_fallback_reason",
        ),
        (
            "impl",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl",
            "generic",
            "tensor_native_replay_impl",
        ),
        (
            "table_source",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source",
            "rebuilt_on_sample_v0",
            "tensor_native_replay_table_source",
        ),
        (
            "reused_records",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count",
            0,
            "tensor_native_replay_table_reused_record_count",
        ),
        (
            "missing_records",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count",
            1,
            "tensor_native_replay_table_missing_record_count",
        ),
        (
            "table_rows",
            "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows",
            0,
            "tensor_native_replay_table_rows",
        ),
    )
    for label, field, value, error in bad_cases:
        case_dir = tmp_path / label
        case_dir.mkdir()
        bad_surface = dict(surface)
        bad_surface[field] = value
        bad_paths = _speed_row_paths(case_dir, operational_surface=bad_surface)
        with pytest.raises(CompactCoachSpeedRowEvidenceError, match=error):
            build_compact_coach_speed_row_evidence_v1(
                route="compact_owned_trainer",
                candidate_checkpoint_id="unit-compact-ckpt",
                unified_lifecycle_report_path=bad_paths["lifecycle"],
                manifest_path=bad_paths["manifest"],
                row_id="001",
                result_json_path=bad_paths["result"],
                speed_currency="compact_trainer_env_steps_per_sec",
                numerator_field="env_steps_collected",
                denominator_field="training_wall_sec",
            )


def test_compact_coach_speed_row_allows_owner_search_prebuilt_batch_without_parent_fused(
    tmp_path,
):
    surface = {
        **_operational_surface(),
        "compact_owned_training_loop_owner": "owner_search_worker",
        "compact_owned_loop_fused_learner_batch": False,
        "compact_owner_search_action_only_result": True,
        "compact_owner_search_owner_materializes_replay": True,
        "compact_owner_search_parent_slab_commits_replay": False,
        "compact_rollout_slab_sample_gate_calls": 0,
        "compact_rollout_slab_learner_gate_calls": 0,
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": True,
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": True,
        "compact_rollout_slab_learner_gate_prebuilt_batch_used": True,
    }
    paths = _speed_row_paths(tmp_path, operational_surface=surface)

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    validate_compact_coach_speed_row_evidence_v1(evidence)

    bad_surface = dict(surface)
    bad_surface["compact_rollout_slab_sample_gate_calls"] = 1
    bad_dir = tmp_path / "bad-owner-search"
    bad_dir.mkdir()
    bad_paths = _speed_row_paths(bad_dir, operational_surface=bad_surface)
    with pytest.raises(
        CompactCoachSpeedRowEvidenceError,
        match="fused learner-batch proof requires fused learner-batch config",
    ):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=bad_paths["lifecycle"],
            manifest_path=bad_paths["manifest"],
            row_id="001",
            result_json_path=bad_paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_unroll2_proof_surface_mismatch(
    tmp_path,
):
    surface = _fused_unroll2_operational_surface()
    paths = _speed_row_paths(tmp_path, operational_surface=surface)
    result = _read_json(paths["result"])
    result["compact"][
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl"
    ] = "generic"
    _write_json(paths["result"], result)

    with pytest.raises(
        CompactCoachSpeedRowEvidenceError,
        match="operational_surface .*impl mismatch",
    ):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_evidence_binds_repeatability_fields(tmp_path):
    repeatability = {
        "seed": 123,
        "sample_seed_base": 123,
        "sample_batch_size": 512,
        "sample_interval": 8,
        "replay_pair_capacity": 4096,
        "learner_train_steps": 1,
        "policy_refresh_interval": 4,
        "num_simulations": 1,
        "compact_rollout_slab_sample_gate_last_seed": 131,
        "compact_rollout_slab_learner_gate_last_seed": 137,
        "compact_owned_loop_sample_gate_last_metadata_seed": 131,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_sample_seed": 131,
        "env_trajectory_ordered_checksum_total": 777,
        "env_terminal_row_checksum_total": 33,
        "env_autoreset_row_checksum_total": 44,
        "env_death_count_checksum_total": 55,
        "last_env_terminal_row_checksum": 6,
        "compact_rollout_slab_sample_gate_sample_position_order_checksum": 888,
        "compact_rollout_slab_sample_gate_source_record_window_checksum": 999,
        "compact_owned_loop_record_step_calls": 225,
        "compact_owned_loop_appended_replay_entry_count": 224,
        "compact_rollout_slab_sample_gate_sample_rows": 11264,
        "compact_rollout_slab_learner_gate_sample_rows": 11264,
        "compact_rollout_slab_sample_gate_opportunities": 225,
        "compact_rollout_slab_sample_gate_skipped_count": 197,
        "compact_rollout_slab_sample_gate_calls": 28,
        "compact_rollout_slab_learner_gate_updates": 28,
        "compact_owned_trainer_policy_refresh_count": 7,
        "compact_rollout_slab_committed_index_row_count": 224,
        "compact_rollout_slab_policy_refresh_after_learner_gate_calls": 7,
        "compact_rollout_slab_policy_refresh_after_learner_gate_interval": 4,
        "compact_rollout_slab_policy_refresh_after_learner_gate_skipped_count": 21,
        "compact_rollout_slab_policy_refresh_after_learner_gate_forced_final_count": 1,
        "compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest": (
            "sha256:unit"
        ),
    }
    paths = _speed_row_paths(
        tmp_path,
        operational_surface={**_operational_surface(), **repeatability},
    )

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )

    for key, value in repeatability.items():
        assert evidence["operational_surface"][key] == value
        assert evidence["result_summary"][key] == value
    validate_compact_coach_speed_row_evidence_v1(evidence)

    evidence["result_summary"]["seed"] = 999
    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="result summary seed"):
        validate_compact_coach_speed_row_evidence_v1(evidence)

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    evidence["result_summary"]["compact_rollout_slab_sample_gate_sample_rows"] = 999
    with pytest.raises(
        CompactCoachSpeedRowEvidenceError,
        match="result summary compact_rollout_slab_sample_gate_sample_rows",
    ):
        validate_compact_coach_speed_row_evidence_v1(evidence)


def test_compact_coach_speed_row_rejects_prefixed_ref_mismatch(tmp_path):
    paths = _speed_row_paths(tmp_path)
    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    evidence["evidence_ref"] = "compact_coach_speed_row:fake"

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="evidence_ref"):
        validate_compact_coach_speed_row_evidence_v1(evidence)


def test_compact_coach_speed_row_rejects_operational_surface_mismatch(tmp_path):
    paths = _speed_row_paths(tmp_path)
    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    evidence["operational_surface"]["death_mode"] = "normal"

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="operational_surface"):
        validate_compact_coach_speed_row_evidence_v1(evidence)


def test_compact_coach_speed_row_carries_normal_death_terminal_proof(tmp_path):
    paths = _speed_row_paths(
        tmp_path,
        actor_handoff_config=_borrowed_normal_death_handoff_config(),
        operational_surface=_normal_death_operational_surface(),
        normal_death_contract=_normal_death_contract(),
    )

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )

    proof = evidence["terminal_death_proof"]
    assert proof["death_mode"] == "normal"
    assert proof["normal_death_terminal_contract_promotion_gate_satisfied"] is True
    assert proof["terminal_row_count"] == 3
    assert proof["normal_collision_death_evidence_id"] == "unit-normal-death"
    validate_compact_coach_speed_row_evidence_v1(evidence)


def test_compact_coach_speed_row_rejects_lean_normal_death_config_mismatch(tmp_path):
    surface = _normal_death_operational_surface()
    surface["compact_owned_training_loop_owner"] = "lean_compact_trainer_step"
    surface["compact_owned_trainer_config_death_mode"] = "profile_no_death"
    paths = _speed_row_paths(
        tmp_path,
        actor_handoff_config=_borrowed_normal_death_handoff_config(),
        operational_surface=surface,
        normal_death_contract=_normal_death_contract(),
    )

    with pytest.raises(
        CompactCoachSpeedRowEvidenceError,
        match="trainer config death_mode=normal",
    ):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_normal_death_without_contract(tmp_path):
    paths = _speed_row_paths(
        tmp_path,
        actor_handoff_config=_borrowed_normal_death_handoff_config(),
        operational_surface=_normal_death_operational_surface(),
    )

    with pytest.raises(
        CompactCoachSpeedRowEvidenceError,
        match="terminal/death contract",
    ):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_tampered_result_json(tmp_path):
    paths = _speed_row_paths(tmp_path)
    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    result = _read_json(paths["result"])
    result["summary"]["steps_per_sec"] = 61.0
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="result sha256"):
        validate_compact_coach_speed_row_evidence_v1(evidence)


def test_compact_coach_speed_row_rejects_profile_only_manifest(tmp_path):
    paths = _speed_row_paths(tmp_path)
    manifest = _read_json(paths["manifest"])
    manifest["profile_only"] = True
    _write_json(paths["manifest"], manifest)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="profile_only=false"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_fallback_denominator(tmp_path):
    paths = _speed_row_paths(tmp_path)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="fallback denominator"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
            uses_fallback_denominator=True,
        )


def test_compact_coach_speed_row_rejects_profile_speed_currency(tmp_path):
    paths = _speed_row_paths(
        tmp_path,
        speed_currency="compact_profile_active_roots_per_sec",
    )

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="profile-only"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_profile_active_roots_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_profile_stock_speed_currency(tmp_path):
    paths = _speed_row_paths(
        tmp_path,
        speed_currency="stock_train_muzero_profile_env_steps_per_sec",
    )

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="profile-only"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="stock_train_muzero_profile_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_denominator_override_mismatch(tmp_path):
    paths = _speed_row_paths(tmp_path)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="numerator"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
            numerator_value=240.0,
            reported_steps_per_sec=120.0,
        )


def test_compact_coach_speed_row_rejects_arbitrary_manifest_schema(tmp_path):
    paths = _speed_row_paths(tmp_path)
    manifest = _read_json(paths["manifest"])
    manifest["schema_id"] = "curvyzero_hybrid_observation_profile_manifest/v0"
    _write_json(paths["manifest"], manifest)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="manifest schema"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_arbitrary_result_schema(tmp_path):
    paths = _speed_row_paths(tmp_path)
    result = _read_json(paths["result"])
    result["schema_id"] = "curvyzero_hybrid_observation_profile_collected_result/v0"
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="result schema"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_missing_run_provenance(tmp_path):
    paths = _speed_row_paths(tmp_path)
    result = _read_json(paths["result"])
    result["run_invocation_id"] = ""
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="run_invocation_id"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_compact_payload_not_ok(tmp_path):
    paths = _speed_row_paths(tmp_path)
    result = _read_json(paths["result"])
    result["compact"]["ok"] = False
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="compact payload ok"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_result_row_snapshot_mismatch(tmp_path):
    paths = _speed_row_paths(tmp_path)
    result = _read_json(paths["result"])
    result["row"]["speed_currency"] = "compact_trainer_env_steps_per_sec_v2"
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="row snapshot"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_actor_handoff_fallback(tmp_path):
    paths = _speed_row_paths(tmp_path)
    result = _read_json(paths["result"])
    result["summary"]["render_state_handoff_mode"] = "copy_actor_state_to_parent_buffers"
    result["compact"]["render_state_handoff_mode"] = "copy_actor_state_to_parent_buffers"
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="actor_handoff_config"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_persistent_handoff_copy_steps(tmp_path):
    paths = _speed_row_paths(tmp_path)
    result = _read_json(paths["result"])
    result["summary"]["render_state_copy_steps"] = 7
    result["compact"]["render_state_copy_steps"] = 7
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="must not copy"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_borrowed_actor_count_not_one(tmp_path):
    paths = _speed_row_paths(
        tmp_path,
        actor_handoff_config={
            "hybrid_persistent_compact_render_state_buffer": False,
            "hybrid_borrow_single_actor_render_state": True,
            "render_state_handoff_mode": "borrow_single_actor_env_state",
            "render_state_copy_steps": 0,
            "render_state_borrowed_steps": 5,
            "render_state_row_overlay_steps": 0,
            "render_state_row_overlay_rows": 0,
            "render_state_row_overlay_bytes": 0,
        },
        operational_surface={**_operational_surface(), "actor_count": 16},
    )

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="actor_count=1"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_nodeath_terminal_surface(tmp_path):
    paths = _speed_row_paths(
        tmp_path,
        operational_surface={**_operational_surface(), "terminal_row_count": 1},
    )

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="profile_no_death"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_malformed_actor_handoff_counts(tmp_path):
    paths = _speed_row_paths(tmp_path)
    result = _read_json(paths["result"])
    result["summary"]["render_state_copy_steps"] = "nope"
    result["compact"]["render_state_copy_steps"] = "nope"
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="non-negative integer"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_rejects_manifest_claim_on_direct_validation(tmp_path):
    paths = _speed_row_paths(tmp_path)
    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    manifest = _read_json(paths["manifest"])
    manifest["training_speedup_claim"] = True
    _write_json(paths["manifest"], manifest)
    evidence["manifest"]["sha256"] = _sha256(paths["manifest"])

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="training_speedup_claim"):
        validate_compact_coach_speed_row_evidence_v1(evidence)


def test_compact_coach_speed_row_rejects_nan_denominator(tmp_path):
    paths = _speed_row_paths(tmp_path)
    result = _read_json(paths["result"])
    result["summary"]["steps_per_sec"] = math.nan
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="finite"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_loaded_checkpoint_identity_can_validate(tmp_path):
    paths = _speed_row_paths(
        tmp_path,
        model_identity_scope="candidate_loaded_checkpoint",
    )

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )

    assert evidence["model_identity"]["scope"] == "candidate_loaded_checkpoint"
    validate_compact_coach_speed_row_evidence_v1(evidence)


def test_compact_coach_speed_row_rejects_loaded_model_version_mismatch(tmp_path):
    paths = _speed_row_paths(
        tmp_path,
        model_identity_scope="candidate_loaded_checkpoint",
    )
    result = _read_json(paths["result"])
    result["compact"]["loaded_checkpoint_identity"]["model_version_ref"] = (
        "unit-compact-ckpt:different-model"
    )
    _write_json(paths["result"], result)

    with pytest.raises(CompactCoachSpeedRowEvidenceError, match="model_version_ref"):
        build_compact_coach_speed_row_evidence_v1(
            route="compact_owned_trainer",
            candidate_checkpoint_id="unit-compact-ckpt",
            unified_lifecycle_report_path=paths["lifecycle"],
            manifest_path=paths["manifest"],
            row_id="001",
            result_json_path=paths["result"],
            speed_currency="compact_trainer_env_steps_per_sec",
            numerator_field="env_steps_collected",
            denominator_field="training_wall_sec",
        )


def test_compact_coach_speed_row_lifecycle_identity_can_come_from_checkpoint_path(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint = build_compact_trainer_checkpoint_v1(
        checkpoint_id="unit-compact-ckpt",
        trainer_config={"schema_id": "unit_config"},
        resume_state=CompactTrainerResumeStateV1(
            trainer_id="unit-compact-ckpt:trainer",
            train_step=1,
            learner_update_count=1,
            sample_batch_count=1,
            policy_version_ref="unit-compact-ckpt:policy-update-1",
            model_version_ref="unit-compact-ckpt:model-update-1",
            policy_source="unit_policy_source",
            loop_counters={},
        ),
        model=model,
        optimizer=optimizer,
        replay_store_state=_owned_replay_state(
            policy_version_ref="unit-compact-ckpt:policy-update-1",
            model_version_ref="unit-compact-ckpt:model-update-1",
        ),
        metrics={"loss": 0.0},
    )
    checkpoint_path = save_compact_trainer_checkpoint_v1(
        checkpoint,
        tmp_path / "compact_checkpoint.pt",
    )
    loaded_identity = {
        "scope": "candidate_loaded_checkpoint",
        "identity_source": "unit_loaded_checkpoint",
        "candidate_loaded_checkpoint": True,
        "checkpoint_id": "unit-compact-ckpt",
        "trainer_id": "unit-compact-ckpt:trainer",
        "policy_version_ref": "unit-compact-ckpt:policy-update-1",
        "model_version_ref": "unit-compact-ckpt:model-update-1",
        "policy_source": "unit_policy_source",
        "learner_update_count": 1,
        "model_state_digest": compact_model_state_digest_v1(checkpoint.model_state_dict),
        "compact_checkpoint_path": str(checkpoint_path),
        "compact_checkpoint_sha256": _sha256(checkpoint_path),
    }
    paths = _speed_row_paths(
        tmp_path,
        model_identity_scope="candidate_loaded_checkpoint",
    )
    lifecycle = _read_json(paths["lifecycle"])
    lifecycle.pop("current_chain_identity", None)
    lifecycle["compact_checkpoint_path"] = str(checkpoint_path)
    _write_json(paths["lifecycle"], lifecycle)
    result = _read_json(paths["result"])
    result["compact"]["loaded_checkpoint_identity"] = loaded_identity
    _write_json(paths["result"], result)

    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id="unit-compact-ckpt",
        unified_lifecycle_report_path=paths["lifecycle"],
        manifest_path=paths["manifest"],
        row_id="001",
        result_json_path=paths["result"],
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )

    assert evidence["model_identity"]["scope"] == "candidate_loaded_checkpoint"
    assert (
        evidence["model_identity"]["lifecycle_identity"]["identity_source"]
        == "compact_trainer_checkpoint"
    )
    assert (
        evidence["model_identity"]["result_loaded_checkpoint_identity"]["model_state_digest"]
        == loaded_identity["model_state_digest"]
    )


def _speed_row_paths(
    tmp_path,
    *,
    speed_currency="compact_trainer_env_steps_per_sec",
    model_identity_scope="candidate_named_support_only",
    actor_handoff_config=None,
    operational_surface=None,
    normal_death_contract=None,
):
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    manifest_path = tmp_path / "manifest.json"
    result_path = tmp_path / "row_001_result.json"
    lifecycle = {
        "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
        "ok": True,
        "checkpoint_id": "unit-compact-ckpt",
        "lifecycle_gates_complete": True,
        "missing_required_gates": ["coach_speed_row"],
        "promotion_eligible": False,
    }
    loaded_identity = {}
    if model_identity_scope == "candidate_loaded_checkpoint":
        loaded_identity = _loaded_identity()
        lifecycle["current_chain_identity"] = dict(loaded_identity)
    search_config = _search_config()
    actor_handoff_config = actor_handoff_config or _actor_handoff_config()
    operational_surface = operational_surface or _operational_surface()
    normal_death_fields = (
        {"normal_death_terminal_contract": normal_death_contract}
        if normal_death_contract is not None
        else {}
    )
    _write_json(
        lifecycle_path,
        lifecycle,
    )
    _write_json(
        manifest_path,
        {
            "schema_id": "curvyzero_compact_coach_speed_row_manifest/v1",
            "experiment_id": "unit-speed-row",
            "candidate_checkpoint_id": "unit-compact-ckpt",
            "route": "compact_owned_trainer",
            "profile_only": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "death_mode": operational_surface["death_mode"],
            **search_config,
            "hybrid_persistent_compact_render_state_buffer": (
                actor_handoff_config["hybrid_persistent_compact_render_state_buffer"]
            ),
            "non_claims": _non_claims(),
            "rows": [
                {
                    "row_id": "001",
                    "candidate_checkpoint_id": "unit-compact-ckpt",
                    "route": "compact_owned_trainer",
                    "profile_only": False,
                    "calls_train_muzero": False,
                    "touches_live_runs": False,
                    "row_purpose": "coach_speed_row",
                    "speed_currency": speed_currency,
                    "promotion_claim": False,
                    "actor_count": operational_surface["actor_count"],
                    "batch_size": operational_surface["batch_size"],
                    "steps": operational_surface["steps"],
                    "warmup_steps": operational_surface["warmup_steps"],
                    "death_mode": operational_surface["death_mode"],
                    **search_config,
                    "hybrid_persistent_compact_render_state_buffer": (
                        actor_handoff_config["hybrid_persistent_compact_render_state_buffer"]
                    ),
                    "non_claims": _non_claims(),
                    "command": ["unit", "coach-speed-row"],
                }
            ],
        },
    )
    row = _read_json(manifest_path)["rows"][0]
    _write_json(
        result_path,
        {
            "schema_id": "curvyzero_compact_coach_speed_row_result/v1",
            "ok": True,
            "status": "complete",
            "problem": None,
            "returncode": 0,
            "run_invocation_id": "unit-run-invocation",
            "candidate_checkpoint_id": "unit-compact-ckpt",
            "row_id": "001",
            "row": row,
            "producer": {
                "schema_id": "curvyzero_compact_coach_speed_row_producer/v1",
                "producer_id": "unit-speed-row-producer",
                "run_id": "unit-speed-row",
                "produced_by": "tests/test_compact_coach_speed_row.py",
            },
            "summary": {
                "profile_only": False,
                "calls_train_muzero": False,
                "touches_live_runs": False,
                "status": "complete",
                "ok": True,
                "row_id": "001",
                "candidate_checkpoint_id": "unit-compact-ckpt",
                "route": "compact_owned_trainer",
                "row_purpose": "coach_speed_row",
                "promotion_claim": False,
                "speed_currency": speed_currency,
                **operational_surface,
                **normal_death_fields,
                **search_config,
                **actor_handoff_config,
                "env_steps_collected": 120.0,
                "training_wall_sec": 2.0,
                "steps_per_sec": 60.0,
                "non_claims": _non_claims(),
            },
            "compact": {
                "ok": True,
                "candidate_checkpoint_id": "unit-compact-ckpt",
                "route": "compact_owned_trainer",
                "profile_only": False,
                "calls_train_muzero": False,
                "touches_live_runs": False,
                "real_compact_owned_training_work": True,
                "compact_owned_trainer_update_count": 1,
                "compact_owned_trainer_env_step_source": "unit_fixture",
                "model_identity_scope": model_identity_scope,
                "loaded_checkpoint_identity": loaded_identity,
                **operational_surface,
                **normal_death_fields,
                **search_config,
                **actor_handoff_config,
                "non_claims": _non_claims(),
            },
            "non_claims": _non_claims(),
        },
    )
    return {
        "lifecycle": lifecycle_path,
        "manifest": manifest_path,
        "result": result_path,
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _search_config():
    return {
        "search_service_kind": "compact_torch_search_service",
        "search_service_impl": "compact_torch_device_tree_fixed_shape_v0",
        "compact_torch_initial_inference_mode": "direct_core",
        "compact_torch_observation_memory_format": "channels_last",
        "compact_torch_model_memory_format": "contiguous",
        "compact_torch_defer_one_simulation_replay_payload_requested": False,
        "compact_torch_memory_format_applies_to_search_service": True,
    }


def _actor_handoff_config():
    return {
        "hybrid_persistent_compact_render_state_buffer": True,
        "hybrid_borrow_single_actor_render_state": False,
        "render_state_handoff_mode": "persistent_compact_render_state_buffer",
        "render_state_copy_steps": 0,
        "render_state_borrowed_steps": 0,
        "render_state_row_overlay_steps": 0,
        "render_state_row_overlay_rows": 0,
        "render_state_row_overlay_bytes": 0,
    }


def _whole_owner_buffer_replay_projection():
    env_steps = 120.0
    wall_sec = 2.0
    baseline_speed = 12689.381637
    target_speed = baseline_speed * 2.0
    replay_append_sec = 0.4
    sample_sec = 0.2
    parent_wait_sec = 0.5
    projected_removed_sec = 0.5
    projected_wall_sec = wall_sec - projected_removed_sec
    return {
        "compact_whole_owner_buffer_replay_ceiling_schema_id": (
            "curvyzero_compact_whole_owner_buffer_replay_ceiling/v1"
        ),
        "compact_whole_owner_buffer_replay_ceiling_enabled": True,
        "compact_whole_owner_buffer_replay_ceiling_projection_only": True,
        "compact_whole_owner_buffer_replay_ceiling_production_speed_claim": False,
        "compact_whole_owner_buffer_replay_ceiling_touches_live_training": False,
        "compact_whole_owner_buffer_replay_ceiling_requires_h100_validation": True,
        "compact_whole_owner_buffer_replay_ceiling_speed_currency": (
            "local_projection_no_speed"
        ),
        "compact_whole_owner_buffer_replay_ceiling_projection_source": (
            "measured_owner_search_surface_projection_v1"
        ),
        "compact_whole_owner_buffer_replay_ceiling_basis": (
            "owner_replay_append_train_sample_parent_wait_bound_v1"
        ),
        "compact_whole_owner_buffer_replay_ceiling_h100_validation_status": "not_run",
        "compact_whole_owner_buffer_replay_ceiling_variance_interpretation": (
            "projection_not_measurement"
        ),
        "compact_whole_owner_buffer_replay_ceiling_promotion_eligible": False,
        "compact_whole_owner_buffer_replay_ceiling_observed_env_steps": env_steps,
        "compact_whole_owner_buffer_replay_ceiling_observed_wall_sec": wall_sec,
        "compact_whole_owner_buffer_replay_ceiling_observed_env_steps_per_sec": 60.0,
        "compact_whole_owner_buffer_replay_ceiling_baseline_env_steps_per_sec": (
            baseline_speed
        ),
        "compact_whole_owner_buffer_replay_ceiling_baseline_whole_loop_sec": (
            env_steps / baseline_speed
        ),
        "compact_whole_owner_buffer_replay_ceiling_target_multiplier": 2.0,
        "compact_whole_owner_buffer_replay_ceiling_target_env_steps_per_sec": (
            target_speed
        ),
        "compact_whole_owner_buffer_replay_ceiling_target_wall_sec": (
            env_steps / target_speed
        ),
        "compact_whole_owner_buffer_replay_ceiling_observed_speedup_vs_opt104": (
            60.0 / baseline_speed
        ),
        "compact_whole_owner_buffer_replay_ceiling_observed_replay_append_sec": (
            replay_append_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_observed_owner_train_sample_sec": (
            sample_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_observed_owner_train_wall_sec": 0.7,
        "compact_whole_owner_buffer_replay_ceiling_observed_learner_update_sec": 0.1,
        "compact_whole_owner_buffer_replay_ceiling_observed_worker_search_sec": 0.3,
        "compact_whole_owner_buffer_replay_ceiling_observed_parent_wait_sec": (
            parent_wait_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_direct_replay_sample_surface_sec": (
            replay_append_sec + sample_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_parent_wait_bounded_surface_sec": (
            parent_wait_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_preserved_search_update_floor_sec": 0.4,
        "compact_whole_owner_buffer_replay_ceiling_max_removable_sec": 1.6,
        "compact_whole_owner_buffer_replay_ceiling_projected_removed_sec": (
            projected_removed_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_projected_wall_sec": (
            projected_wall_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_projected_env_steps_per_sec": (
            env_steps / projected_wall_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_projected_speedup_vs_opt104": (
            (env_steps / projected_wall_sec) / baseline_speed
        ),
        "compact_whole_owner_buffer_replay_ceiling_projected_delta_sec": (
            projected_removed_sec
        ),
        "compact_whole_owner_buffer_replay_ceiling_projected_reaches_2x": False,
        "compact_whole_owner_buffer_replay_ceiling_additional_removed_sec_to_2x": (
            projected_wall_sec - (env_steps / target_speed)
        ),
    }


def _borrowed_normal_death_handoff_config():
    return {
        "hybrid_persistent_compact_render_state_buffer": False,
        "hybrid_borrow_single_actor_render_state": True,
        "render_state_handoff_mode": "borrow_single_actor_env_state",
        "render_state_copy_steps": 0,
        "render_state_borrowed_steps": 5,
        "render_state_row_overlay_steps": 1,
        "render_state_row_overlay_rows": 3,
        "render_state_row_overlay_bytes": 1024,
    }


def _operational_surface():
    return {
        "actor_count": 1,
        "batch_size": 2,
        "steps": 4,
        "warmup_steps": 1,
        "death_mode": "profile_no_death",
        "compact_owned_training_loop_owner": "",
        "compact_owned_trainer_config_death_mode": "",
        "normal_death_terminal_contract_owner": "none",
        "terminal_row_count": 0,
        "death_row_count": 0,
        "terminated_row_count": 0,
        "truncated_row_count": 0,
        "terminal_final_observation_row_count": 0,
        "terminal_final_observation_before_autoreset_verified": False,
        "terminal_sample_row_count": 0,
        "terminal_unroll_value_target_mode": "none",
        "terminal_unroll_value_target_row_count": 0,
        "resident_observation_host_fallback_count": 0.0,
        "normal_death_terminal_contract_promotion_gate_satisfied": False,
        "source_profile_total_sec": 0.0,
        "source_profile_warmup_sec": 0.0,
        "source_profile_measured_sec": 0.0,
        "source_profile_timing_per_timestep_sec": 0.0,
        "speed_row_actor_step_wall_sec": 0.0,
        "speed_row_observation_sec": 0.0,
        "speed_row_renderer_stack_update_sec": 0.0,
        "speed_row_compact_rollout_slab_sec": 0.0,
        "speed_row_sample_gate_sec": 0.0,
        "speed_row_learner_gate_sec": 0.0,
        "speed_row_policy_refresh_sec": 0.0,
        "speed_row_primary_accounted_sec": 0.0,
        "speed_row_primary_residual_sec": 0.0,
    }


def _fused_unroll2_operational_surface():
    return {
        **_operational_surface(),
        "learner_num_unroll_steps": 2,
        "compact_owned_loop_fused_learner_batch": True,
        "compact_muzero_learner_batch_unroll2_specialized_builder": True,
        "compact_rollout_slab_sample_gate_sec": 0.11,
        "compact_rollout_slab_learner_gate_sec": 0.12,
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": True,
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": True,
        "compact_rollout_slab_sample_gate_host_provider_learner_batch": False,
        "compact_rollout_slab_learner_gate_prebuilt_batch_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count": 3,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count": 3,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl": "unroll2_specialized_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "unroll2_specialized",
    }


def _fused_tensor_native_operational_surface():
    return {
        **_operational_surface(),
        "learner_num_unroll_steps": 2,
        "compact_owned_loop_fused_learner_batch": True,
        "compact_muzero_learner_batch_unroll2_specialized_builder": False,
        "compact_muzero_learner_batch_learner_ready_unroll2_cache": True,
        "compact_muzero_learner_batch_tensor_native_replay": True,
        "compact_rollout_slab_sample_gate_sec": 0.11,
        "compact_rollout_slab_learner_gate_sec": 0.12,
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": True,
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": True,
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": True,
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": True,
        "compact_rollout_slab_sample_gate_host_provider_learner_batch": False,
        "compact_rollout_slab_learner_gate_prebuilt_batch_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_requested": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": 3,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": 3,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": 3,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": "learner_ready_unroll2_cache_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_requested": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": 3,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": "maintained_unroll2_table_gather_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": "maintained_record_table_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_impl": "direct_record_table_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_direct_build_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_reused_record_count": 3,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_missing_record_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_rows": 192,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec": 0.001,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_gather_sec": 0.002,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "learner_ready_unroll2_cache",
    }


def _fixed_soa_tensor_native_operational_surface():
    return {
        **_fused_tensor_native_operational_surface(),
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": False,
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": False,
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_prevalidation_source": (
            "fixed_soa_direct_gather_v1"
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_available_group_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_eligible_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_used": False,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_call_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_learner_ready_unroll2_cache_impl": "fixed_soa_columns_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_call_count": 1,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_impl": "fixed_soa_direct_gather_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_source": "fixed_soa_columns_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_impl": "fixed_soa_columns_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_direct_build_used": False,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_concat_sec": 0.0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_tensor_native_replay_table_build_sec": 0.0,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_requested": False,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_eligible": False,
        "compact_rollout_slab_sample_gate_tensor_native_direct_prebuilt_path_used": False,
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_count": 0,
        "compact_rollout_slab_sample_gate_tensor_native_direct_group_object_build_skipped": True,
        "compact_rollout_slab_sample_gate_fixed_soa_requested": True,
        "compact_rollout_slab_sample_gate_fixed_soa_used": True,
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_count": 3,
        "compact_rollout_slab_sample_gate_fixed_soa_entry_view_object_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_step_view_object_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_learner_ready_object_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_table_entry_object_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_table_concat_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_record_count": 2155,
        "compact_rollout_slab_sample_gate_fixed_soa_selected_record_count": 62,
        "compact_rollout_slab_sample_gate_fixed_soa_table_row_count": 9770,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_group_size": 8,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_used": True,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_sample_semantic_drift": True,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_selected_group_count": 64,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_duplicate_group_count": 2,
        "compact_rollout_slab_sample_gate_fixed_soa_locality_local_replace_group_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_count": 0,
        "compact_rollout_slab_sample_gate_fixed_soa_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_fixed_soa_slot_write_sec": 0.001,
        "compact_rollout_slab_sample_gate_fixed_soa_successor_index_sec": 0.001,
        "compact_rollout_slab_sample_gate_fixed_soa_total_sec": 0.002,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": (
            "fixed_soa_direct_gather"
        ),
    }


def _normal_death_operational_surface():
    return {
        "actor_count": 1,
        "batch_size": 2,
        "steps": 4,
        "warmup_steps": 1,
        "death_mode": "normal",
        "compact_owned_training_loop_owner": "hybrid_observation_profile_runner",
        "compact_owned_trainer_config_death_mode": "normal",
        "normal_death_terminal_contract_owner": "compact_owned_trainer_config",
        "terminal_row_count": 3,
        "death_row_count": 3,
        "terminated_row_count": 3,
        "truncated_row_count": 0,
        "terminal_final_observation_row_count": 3,
        "terminal_final_observation_before_autoreset_verified": True,
        "terminal_sample_row_count": 1,
        "terminal_unroll_value_target_mode": ("stock_terminal_no_bootstrap_return_discount_1.0"),
        "terminal_unroll_value_target_row_count": 1,
        "resident_observation_host_fallback_count": 0.0,
        "normal_death_terminal_contract_promotion_gate_satisfied": True,
        "source_profile_total_sec": 0.0,
        "source_profile_warmup_sec": 0.0,
        "source_profile_measured_sec": 0.0,
        "source_profile_timing_per_timestep_sec": 0.0,
        "speed_row_actor_step_wall_sec": 0.0,
        "speed_row_observation_sec": 0.0,
        "speed_row_renderer_stack_update_sec": 0.0,
        "speed_row_compact_rollout_slab_sec": 0.0,
        "speed_row_sample_gate_sec": 0.0,
        "speed_row_learner_gate_sec": 0.0,
        "speed_row_policy_refresh_sec": 0.0,
        "speed_row_primary_accounted_sec": 0.0,
        "speed_row_primary_residual_sec": 0.0,
    }


def _normal_death_contract():
    return build_compact_death_terminal_contract_v1(
        death_mode="normal",
        normal_collision_evidence={
            "schema_id": "curvyzero_compact_death_terminal_contract_evidence/v1",
            "evidence_id": "unit-normal-death",
            "evidence_refs": ["unit-normal-death-profile"],
            "death_mode": "normal",
            "trainer_config_death_mode": "normal",
            "normal_death_terminal_contract_owner": "compact_owned_trainer_config",
            "terminal_row_count": 3,
            "terminated_row_count": 3,
            "truncated_row_count": 0,
            "death_row_count": 3,
            "death_count_total": 3,
            "normal_collision_death_causes": ["opponent_trail"],
            "normal_collision_death_hit_owner_present": True,
            "normal_collision_death_evidence_rows": [
                {
                    "death_cause": ["opponent_trail"],
                    "death_count": 1,
                    "death_hit_owner": [1, -1],
                    "death_player": [0, -1],
                    "done": True,
                    "draw": False,
                    "final_observation_row": True,
                    "final_reward_map": [-1.0, 1.0],
                    "final_reward_map_matches_reward": True,
                    "global_row": 1,
                    "reward": [-1.0, 1.0],
                    "terminal_reason": 1,
                    "terminated": True,
                    "truncated": False,
                    "winner": 1,
                }
            ],
            "done_semantics_verified": True,
            "terminal_final_observation_before_autoreset": True,
            "terminal_autoreset_observation_forbidden": True,
            "terminal_final_reward_map_verified": True,
            "terminal_unroll_value_target_mode": (
                "stock_terminal_no_bootstrap_return_discount_1.0"
            ),
            "terminal_unroll_bootstrap_after_done": False,
            "terminal_validity_masks_verified": True,
            "post_terminal_masks_zero": True,
            "resident_terminal_final_observation_used": True,
            "device_replay_terminal_rows_verified": True,
            "terminal_sample_row_count": 1,
            "next_final_observation_row_count": 1,
            "terminal_unroll_value_target_row_count": 1,
            "compact_muzero_learner_done_count": 1,
            "compact_muzero_learner_truncated_count": 0,
        },
    )


def _non_claims():
    return {
        "promotion_claim": False,
        "training_speedup_claim": False,
        "live_run_safety_claim": False,
        "stock_resume_claim": False,
        "rating_or_promotion_quality_claim": False,
    }


def _loaded_identity():
    return {
        "scope": "candidate_loaded_checkpoint",
        "identity_source": "unit_fixture",
        "candidate_loaded_checkpoint": True,
        "checkpoint_id": "unit-compact-ckpt",
        "trainer_id": "unit-compact-ckpt:trainer",
        "policy_version_ref": "unit-compact-ckpt:policy-update-1",
        "model_version_ref": "unit-compact-ckpt:model-update-1",
        "policy_source": "unit_policy_source",
        "learner_update_count": 1,
        "model_state_digest": "a" * 64,
        "compact_checkpoint_sha256": "b" * 64,
    }


def _owned_replay_state(*, policy_version_ref: str, model_version_ref: str):
    policy = CompactPolicyVersionRefV1(
        policy_version_ref=policy_version_ref,
        policy_source="unit_policy_source",
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


def _sha256(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
