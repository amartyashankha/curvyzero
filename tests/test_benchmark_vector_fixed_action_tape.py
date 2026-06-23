import importlib.util
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "benchmark_vector_fixed_action_tape.py"
)
_SPEC = importlib.util.spec_from_file_location("benchmark_vector_fixed_action_tape", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
fixed_action_benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fixed_action_benchmark
_SPEC.loader.exec_module(fixed_action_benchmark)


def _assert_nonempty_digest(value: str) -> None:
    assert isinstance(value, str)
    assert value


def test_fixed_action_tape_direct_loop_matches_compact_profile_borderless_body_skip():
    result = fixed_action_benchmark.run_benchmark(
        fixed_action_benchmark.BenchmarkConfig(
            batch_size=4,
            warmup_steps=1,
            measured_steps=3,
            body_capacity=8,
            require_pass=True,
        )
    )

    assert result["schema"] == fixed_action_benchmark.SCHEMA_VERSION
    assert result["status"] == "pass"
    assert result["tape"]["scenario_id"] == (
        "source_borderless_wrap_skips_destination_body_then_next_frame_kills"
    )
    assert result["comparison"]["passed"] is True
    assert all(result["comparison"]["field_matches"].values())
    assert all(result["comparison"]["output_matches"].values())
    assert all(result["comparison"]["scalar_matches"].values())

    proof = result["proof"]
    assert proof["required_pass"] is True
    assert proof["passed"] is True
    assert proof["speed_claim_scope"] == "local_architecture_only_not_h100_speed_evidence"
    assert proof["state_checksum_match"] is True
    assert proof["full_state_checksum_match"] is True
    assert proof["full_state_field_count"] >= 100
    assert proof["unhashed_state_fields"] == []
    assert proof["body_checksum_match"] is True
    assert proof["trajectory_checksum_match"] is True
    assert proof["per_step_state_checksum_match"] is True
    assert proof["per_step_body_checksum_match"] is True
    assert proof["per_step_death_checksum_match"] is True
    assert proof["observation_checksum_match"] is True
    assert proof["uncompared_output_fields"] == []
    assert {
        "final_reward_map",
        "terminated",
        "truncated",
        "terminal_reason",
        "winner",
        "draw",
        "source_physics_substeps_executed",
        "source_physics_elapsed_ms",
    }.issubset(set(proof["output_compared_fields"]))
    assert proof["done_equals_terminated_or_truncated"] is True
    assert proof["death_row_count"] > 0
    assert proof["new_death_row_count"] == 4
    assert proof["measured_new_death_row_count"] == 4
    assert "opponent_trail" in proof["death_cause_names"]
    assert proof["new_death_cause_names"] == ["opponent_trail"]
    assert proof["measured_new_death_cause_names"] == ["opponent_trail"]
    assert proof["expected_death_cause_names_present"] is True
    assert proof["expected_death_evidence_present"] is True
    assert proof["death_transition_step_indices"] == [1]
    assert proof["measured_death_transition_step_indices"] == [1]
    assert proof["measured_tape_indices"] == [1, 0, 1]
    assert proof["first_measured_tape_index"] == 1
    assert proof["measured_initial_fixture_transition_exercised"] is True
    assert proof["terminal_row_count"] == 0
    assert proof["autoreset_row_count"] == 0
    assert proof["reward_shape"] == [4, 3]
    assert proof["done_shape"] == [4]
    assert proof["terminated_shape"] == [4]
    assert proof["truncated_shape"] == [4]
    assert proof["action_mask_shape"] == [4, 3, 3]
    assert proof["zero_observation_stub"] is True
    assert proof["replay_index_rows_exposed"] is False
    assert proof["env_action_checksum_total"] == result["tape"]["action_checksum"]
    assert result["tape"]["action_checksum"] != result["tape"]["action_array_checksum"]
    assert proof["env_done_checksum_total"]
    assert proof["env_reward_checksum_total"]
    assert proof["env_action_mask_checksum_total"]
    assert proof["env_trajectory_checksum_total"]
    assert proof["env_death_cause_checksum_total"]

    assert result["fixed_buffer_direct"]["measured_env_rows"] == 12
    assert result["compact_profile"]["measured_env_rows"] == 12
    assert result["fixed_buffer_direct"]["env_rows_per_sec"] > 0.0
    assert result["comparison"]["fixed_vs_compact_speedup"] > 0.0
    assert "zero-observation checksum is a stub until the observation variant is added" in (
        result["known_limits"]
    )


def test_fixed_action_tape_direct_loop_matches_terminal_autoreset_wall_death():
    result = fixed_action_benchmark.run_benchmark(
        fixed_action_benchmark.BenchmarkConfig(
            scenario="scenarios/environment/source_normal_wall_death_step.json",
            batch_size=4,
            warmup_steps=0,
            measured_steps=1,
            body_capacity=8,
            random_tape_capacity_min=256,
            render_observation=True,
            run_search=True,
            require_pass=True,
        )
    )

    assert result["status"] == "pass"
    assert result["comparison"]["passed"] is True
    assert all(result["comparison"]["field_matches"].values())
    assert all(result["comparison"]["output_matches"].values())
    assert all(result["comparison"]["scalar_matches"].values())

    proof = result["proof"]
    assert proof["passed"] is True
    assert proof["expects_terminal_rows"] is True
    assert proof["expected_terminal_autoreset_evidence_present"] is True
    assert proof["terminal_row_count"] == 4
    assert proof["autoreset_call_count"] == 1
    assert proof["autoreset_row_count"] == 4
    assert proof["terminal_rows_equal_autoreset_rows"] is True
    assert proof["autoreset_rows_checksum_match"] is True
    assert proof["new_death_row_count"] == 4
    assert proof["measured_new_death_row_count"] == 4
    assert proof["new_death_cause_names"] == ["wall"]
    assert proof["measured_new_death_cause_names"] == ["wall"]
    assert proof["expected_death_cause_names"] == ["wall"]
    assert proof["expected_death_evidence_present"] is True
    assert proof["expected_measured_death_evidence_present"] is True
    assert proof["death_transition_step_indices"] == [0]
    assert proof["measured_death_transition_step_indices"] == [0]
    assert proof["measured_tape_indices"] == [0]
    assert proof["measured_initial_fixture_transition_exercised"] is True
    assert proof["reward_shape"] == [4, 2]
    assert proof["done_shape"] == [4]
    assert proof["action_mask_shape"] == [4, 2, 3]
    assert proof["zero_observation_stub"] is False
    assert proof["observation_checksum_match"] is True
    assert proof["observation_schema_id"] == "curvyzero_source_state_canvas_gray64/v0"
    assert proof["observation_shape"] == [4, 2, 4, 64, 64]
    assert proof["latest_frame_shape"] == [8, 1, 64, 64]
    assert proof["root_observation_shape"] == [8, 4, 64, 64]
    assert proof["render_call_count"] == 1
    assert proof["render_row_count"] == 8
    assert proof["observation_nonzero_count"] > 0
    assert proof["observation_nonzero_checksum_present"] is True
    assert proof["resident_host_fallback_allowed"] is False
    assert proof["renderer_backend"] == "cpu_oracle"
    assert proof["search_enabled"] is True
    assert proof["search_metadata_match"] is True
    assert proof["search_call_count"] == 1
    assert proof["search_root_count"] == 8
    assert proof["search_active_root_count"] == 0
    assert proof["search_inactive_root_count"] == 8
    assert proof["search_selected_action_count"] == 0
    assert proof["search_action_d2h_bytes"] == 0
    assert proof["search_deferred_replay_payload_d2h_bytes"] == 0
    assert proof["search_preallocated_buffer_bytes"] == 240
    assert proof["search_root_observation_copy_bytes"] == 0
    assert proof["search_ctree_calls"] == 0
    assert proof["search_tolist_calls"] == 0
    assert proof["search_action_step_identity_checked"] is True
    assert proof["search_action_step_root_index_matches_active"] is True
    assert proof["search_action_step_env_row_matches_root"] is True
    assert proof["search_action_step_player_matches_root"] is True
    assert proof["search_action_step_policy_env_id_matches_root"] is True
    assert proof["search_selected_action_shape_matches"] is True
    assert proof["search_selected_action_legal"] is True
    assert proof["search_replay_payload_digest_deferred"] is True
    assert proof["search_replay_payload_digest_matches_handle"] is True
    assert proof["search_selected_action_digest_matches_payload"] is True
    assert proof["search_root_batch_observation_source"] == (
        fixed_action_benchmark.COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
    )
    assert proof["search_root_batch_observation_copied"] is False
    assert proof["search_root_batch_observation_shape"] == [8, 4, 64, 64]
    assert proof["search_root_batch_observation_dtype"] == "uint8"
    assert proof["search_root_batch_row_major_sidecars_checked"] is True
    assert proof["search_done_root_matches_repeat_done"] is True
    assert proof["search_active_root_mask_matches_non_done_legal"] is True
    assert proof["search_to_play_all_default"] is True
    assert proof["search_target_reward_matches_reward"] is True
    assert proof["search_root_observation_shares_stack"] is True
    assert proof["full_state_checksum_match"] is True
    assert proof["per_step_state_checksum_match"] is True
    assert proof["per_step_body_checksum_match"] is True
    assert proof["per_step_death_checksum_match"] is True
    assert proof["unhashed_state_fields"] == []
    assert proof["uncompared_output_fields"] == []


def test_fixed_action_tape_rendered_observation_proof_has_schema_and_content():
    result = fixed_action_benchmark.run_benchmark(
        fixed_action_benchmark.BenchmarkConfig(
            batch_size=2,
            warmup_steps=1,
            measured_steps=2,
            body_capacity=8,
            render_observation=True,
            require_pass=True,
        )
    )

    assert result["status"] == "pass"
    assert result["comparison"]["passed"] is True
    assert all(result["comparison"]["field_matches"].values())
    assert all(result["comparison"]["output_matches"].values())
    assert all(result["comparison"]["scalar_matches"].values())

    proof = result["proof"]
    assert proof["passed"] is True
    assert proof["required_pass"] is True
    assert proof["zero_observation_stub"] is False
    assert proof["observation_checksum_match"] is True
    assert proof["observation_schema_id"] == "curvyzero_source_state_canvas_gray64/v0"
    assert proof["observation_schema_hash"] == (
        fixed_action_benchmark.SOURCE_STATE_CANVAS_GRAY64_SCHEMA_HASH
    )
    assert proof["observation_dtype"] == "uint8"
    assert proof["observation_shape"] == [2, 3, 4, 64, 64]
    assert proof["latest_frame_shape"] == [6, 1, 64, 64]
    assert proof["root_observation_shape"] == [6, 4, 64, 64]
    assert proof["resident_device_observation_shape"] == [2, 3, 4, 64, 64]
    assert proof["resident_root_device_observation_shape"] == [6, 4, 64, 64]
    assert proof["render_call_count"] == 2
    assert proof["render_row_count"] == 12
    assert proof["observation_nonzero_count"] > 0
    assert proof["observation_nonzero_checksum_present"] is True
    assert proof["resident_row_major_order"] is True
    assert proof["resident_host_fallback_allowed"] is False
    assert proof["renderer_backend"] == "cpu_oracle"
    assert "zero-observation checksum is a stub until the observation variant is added" not in (
        result["known_limits"]
    )
    assert (
        "rendered observation uses the CPU oracle renderer; not optimized H100 resident "
        "observation speed evidence"
    ) in result["known_limits"]


def test_fixed_action_tape_search_root_proof_has_fixed_shape_metadata():
    result = fixed_action_benchmark.run_benchmark(
        fixed_action_benchmark.BenchmarkConfig(
            batch_size=2,
            warmup_steps=1,
            measured_steps=2,
            body_capacity=8,
            render_observation=True,
            run_search=True,
            require_pass=True,
        )
    )

    assert result["status"] == "pass"
    assert result["comparison"]["passed"] is True
    assert all(result["comparison"]["field_matches"].values())
    assert all(result["comparison"]["output_matches"].values())
    assert all(result["comparison"]["scalar_matches"].values())

    proof = result["proof"]
    assert proof["passed"] is True
    assert proof["zero_observation_stub"] is False
    assert proof["search_enabled"] is True
    assert proof["search_metadata_match"] is True
    assert proof["search_impl"] == (
        fixed_action_benchmark.FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL
    )
    assert proof["search_schema_id"] == "curvyzero_compact_search_action_step/v1"
    assert proof["root_batch_schema_id"] == "curvyzero_compact_root_batch/v1"
    assert proof["search_call_count"] == 2
    assert proof["search_root_count"] == 12
    assert proof["search_active_root_count"] == 8
    assert proof["search_inactive_root_count"] == 4
    assert proof["search_selected_action_count"] == 8
    assert proof["search_max_active_root_count"] == 4
    assert proof["search_action_count"] == 3
    assert proof["search_num_simulations"] == 1
    assert proof["search_first_legal_policy"] is True
    assert proof["search_two_phase_action_only"] is True
    assert proof["search_ctree_calls"] == 0
    assert proof["search_tolist_calls"] == 0
    assert proof["search_per_sim_d2h_bytes"] == 0
    assert proof["search_root_observation_copy_bytes"] == 0
    assert proof["search_action_d2h_bytes"] == 16
    assert proof["search_deferred_replay_payload_d2h_bytes"] == 224
    assert proof["search_preallocated_buffer_bytes"] == 180
    assert proof["search_buffer_reused"] is True
    assert proof["search_action_step_identity_checked"] is True
    assert proof["search_action_step_root_index_matches_active"] is True
    assert proof["search_action_step_env_row_matches_root"] is True
    assert proof["search_action_step_player_matches_root"] is True
    assert proof["search_action_step_policy_env_id_matches_root"] is True
    assert proof["search_selected_action_shape_matches"] is True
    assert proof["search_selected_action_legal"] is True
    assert proof["search_replay_payload_digest_deferred"] is True
    assert proof["search_replay_payload_digest_matches_handle"] is True
    assert proof["search_selected_action_digest_matches_payload"] is True
    assert proof["search_root_batch_observation_source"] == (
        fixed_action_benchmark.COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
    )
    assert proof["search_root_batch_observation_copied"] is False
    assert proof["search_root_batch_observation_shape"] == [6, 4, 64, 64]
    assert proof["search_root_batch_observation_dtype"] == "uint8"
    assert proof["search_root_batch_row_major_sidecars_checked"] is True
    assert proof["search_done_root_matches_repeat_done"] is True
    assert proof["search_active_root_mask_matches_non_done_legal"] is True
    assert proof["search_to_play_all_default"] is True
    assert proof["search_target_reward_matches_reward"] is True
    assert proof["search_root_observation_shares_stack"] is True
    _assert_nonempty_digest(proof["search_selected_action_digest"])
    _assert_nonempty_digest(proof["search_replay_payload_digest"])
    _assert_nonempty_digest(proof["search_root_batch_checksum"])
    _assert_nonempty_digest(proof["search_action_step_checksum"])
    _assert_nonempty_digest(proof["search_root_observation_checksum"])
    _assert_nonempty_digest(proof["search_selected_action_checksum"])
    _assert_nonempty_digest(proof["search_joint_action_checksum"])
    assert (
        "fixed-shape search uses profile-only first-legal action owner; not compact "
        "Torch or MCTS speed evidence"
    ) in result["known_limits"]


def test_fixed_action_tape_slab_replay_sample_proof_closes_search_feedback_loop():
    result = fixed_action_benchmark.run_benchmark(
        fixed_action_benchmark.BenchmarkConfig(
            batch_size=2,
            warmup_steps=1,
            measured_steps=2,
            body_capacity=8,
            render_observation=True,
            run_slab_replay=True,
            require_pass=True,
        )
    )

    assert result["status"] == "pass"
    assert result["comparison"]["passed"] is True
    assert all(result["comparison"]["field_matches"].values())
    assert all(result["comparison"]["output_matches"].values())
    assert all(result["comparison"]["scalar_matches"].values())
    assert result["comparison"]["fixed_vs_compact_whole_loop_speedup"] > 0.0
    assert result["compact_profile"]["whole_loop_wall_sec"]["sum"] >= (
        result["compact_profile"]["step_wall_sec"]["sum"]
    )
    assert result["fixed_buffer_direct"]["whole_loop_wall_sec"]["sum"] >= (
        result["fixed_buffer_direct"]["step_wall_sec"]["sum"]
    )
    assert result["compact_profile"]["observation_wall_sec"]["sum"] > 0.0
    assert result["fixed_buffer_direct"]["observation_wall_sec"]["sum"] > 0.0
    assert result["compact_profile"]["slab_replay_wall_sec"]["sum"] > 0.0
    assert result["fixed_buffer_direct"]["slab_replay_wall_sec"]["sum"] > 0.0

    proof = result["proof"]
    total_steps = result["config"]["warmup_steps"] + result["config"]["measured_steps"]
    roots_per_step = result["config"]["batch_size"] * result["tape"]["player_count"]
    assert proof["passed"] is True
    assert proof["whole_loop_wall_sec_includes_observation_search_replay_sample"] is True
    assert proof["fixed_vs_compact_whole_loop_speedup"] > 0.0
    assert proof["zero_observation_stub"] is False
    assert proof["render_call_count"] == total_steps
    assert proof["render_row_count"] == total_steps * roots_per_step
    assert proof["search_enabled"] is False
    assert proof["search_proof_passed"] is True
    assert proof["slab_replay_enabled"] is True
    assert proof["slab_replay_metadata_match"] is True
    assert proof["slab_replay_proof_passed"] is True
    assert proof["slab_replay_failure_reasons"] == []
    assert proof["slab_replay_expected_total_steps"] == total_steps
    assert proof["slab_replay_expected_root_count"] == total_steps * roots_per_step
    assert proof["slab_replay_expected_measured_feedback_action_count"] == (
        result["config"]["measured_steps"]
    )
    assert proof["slab_replay_expected_append_count"] == total_steps - 1
    assert proof["slab_search_feedback_closed_loop"] is True
    assert proof["slab_replay_tape_bootstrap_action_count"] == 1
    assert proof["slab_replay_feedback_action_count"] == total_steps - 1
    assert proof["slab_replay_measured_feedback_action_count"] == (
        result["config"]["measured_steps"]
    )
    assert proof["slab_replay_prev_next_joint_action_match_count"] == (
        proof["slab_replay_feedback_action_count"]
    )
    assert proof["slab_replay_prev_next_joint_action_mismatch_count"] == 0
    assert proof["slab_replay_feedback_differs_from_tape_count"] >= 0
    _assert_nonempty_digest(proof["slab_replay_action_source_sequence_checksum"])
    assert proof["slab_step_count"] == total_steps
    assert proof["slab_root_count"] == total_steps * roots_per_step
    assert proof["slab_active_root_count"] + proof["slab_inactive_root_count"] == (
        proof["slab_root_count"]
    )
    assert proof["slab_selected_action_count"] == proof["slab_active_root_count"]
    assert proof["slab_action_count"] == fixed_action_benchmark.ACTION_COUNT
    assert proof["slab_num_simulations"] == 1
    assert proof["slab_ctree_calls"] == 0
    assert proof["slab_tolist_calls"] == 0
    assert proof["slab_per_sim_d2h_bytes"] == 0
    assert proof["slab_root_observation_copy_bytes"] == 0
    assert proof["slab_replay_action_check_enforced"] is True
    assert proof["slab_replay_root_observation_copied"] is False
    assert proof["slab_replay_index_rows_observation_materialized"] is False
    assert proof["slab_replay_index_rows_next_observation_materialized"] is False
    assert proof["slab_committed_index_group_count"] == total_steps - 1
    assert proof["slab_committed_index_row_count"] > 0
    assert proof["slab_replay_payload_flush_count"] == total_steps - 1
    assert proof["slab_replay_pending_uncommitted_count"] == 1
    assert proof["slab_retains_committed_index_rows"] is False
    assert proof["replay_append_count"] == total_steps - 1
    assert proof["replay_ring_entry_count"] == total_steps - 1
    assert proof["replay_ring_stored_index_row_count"] == (
        proof["slab_committed_index_row_count"]
    )
    assert proof["sample_gate_calls"] == total_steps - 1
    assert proof["sample_row_count"] > 0
    assert proof["sample_target_row_count"] >= proof["sample_row_count"]
    assert proof["slab_replay_sample_batch_built"] is True
    assert proof["slab_replay_sample_batch_size"] == proof["sample_row_count"]
    _assert_nonempty_digest(proof["slab_replay_index_rows_checksum"])
    _assert_nonempty_digest(proof["slab_replay_joint_action_feedback_checksum"])
    _assert_nonempty_digest(proof["slab_replay_root_batch_checksum"])
    _assert_nonempty_digest(proof["slab_replay_action_step_checksum"])
    _assert_nonempty_digest(proof["slab_next_joint_action_checksum"])
    _assert_nonempty_digest(proof["slab_replay_sample_row_id_checksum"])
    _assert_nonempty_digest(proof["slab_replay_sample_action_checksum"])
    _assert_nonempty_digest(proof["slab_replay_sample_observation_checksum"])
    _assert_nonempty_digest(proof["slab_replay_sample_next_observation_checksum"])
    assert (
        "slab replay uses profile-only first-legal action owner and host replay-ring "
        "sample snapshots; not compact Torch, MCTS, learner, or H100 speed evidence"
    ) in result["known_limits"]
    ceiling = result["owner_buffer_ceiling"]
    assert ceiling["schema_id"] == (
        fixed_action_benchmark.OWNER_BUFFER_CEILING_SCHEMA_VERSION
    )
    assert ceiling["enabled"] is True
    assert ceiling["production_speed_claim"] is False
    assert ceiling["touches_live_training"] is False
    assert ceiling["profile_scope"] == (
        "local_fixed_action_tape_whole_loop_not_h100_speed_evidence"
    )
    assert ceiling["fixed_whole_loop_wall_sec"] == (
        result["fixed_buffer_direct"]["whole_loop_wall_sec"]["sum"]
    )
    assert ceiling["compact_whole_loop_wall_sec"] == (
        result["compact_profile"]["whole_loop_wall_sec"]["sum"]
    )
    assert ceiling["fixed_vs_compact_whole_loop_speedup"] == (
        result["comparison"]["fixed_vs_compact_whole_loop_speedup"]
    )
    assert ceiling["needed_removed_fraction_for_2x"] == 0.5
    assert ceiling["owner_transport_candidate_sec"] > 0.0
    assert ceiling["owner_transport_candidate_share"] > 0.0
    assert ceiling["owner_transport_removed_speedup_ceiling"] > 1.0
    assert ceiling["fixed_slab_replay_sample_sec"] > 0.0
    assert ceiling["fixed_observation_sec"] > 0.0


def test_fixed_action_tape_owner_slot_ceiling_closes_mechanics_root_action_buffers():
    result = fixed_action_benchmark.run_benchmark(
        fixed_action_benchmark.BenchmarkConfig(
            scenario="scenarios/environment/source_kinematics_straight_multistep.json",
            batch_size=2,
            warmup_steps=1,
            measured_steps=3,
            body_capacity=8,
            render_observation=True,
            run_owner_slot_ceiling=True,
            require_pass=True,
        )
    )

    assert result["status"] == "pass"
    assert result["comparison"]["passed"] is True
    assert result["comparison"]["scalar_matches"]["owner_slot_metadata"] is True
    assert all(result["comparison"]["field_matches"].values())
    assert all(result["comparison"]["output_matches"].values())

    proof = result["proof"]
    total_steps = result["config"]["warmup_steps"] + result["config"]["measured_steps"]
    assert proof["passed"] is True
    assert proof["owner_slot_ceiling_enabled"] is True
    assert proof["owner_slot_ceiling_proof_passed"] is True
    assert proof["owner_slot_ceiling_failure_reasons"] == []
    assert proof["owner_slot_ceiling_step_count"] == total_steps
    assert proof["owner_slot_ceiling_tape_bootstrap_action_count"] == 1
    assert proof["owner_slot_ceiling_feedback_action_count"] == total_steps - 1
    assert proof["owner_slot_ceiling_measured_feedback_action_count"] == (
        result["config"]["measured_steps"]
    )
    assert proof["owner_slot_ceiling_prev_next_joint_action_match_count"] == (
        proof["owner_slot_ceiling_feedback_action_count"]
    )
    assert proof["owner_slot_ceiling_prev_next_joint_action_mismatch_count"] == 0
    assert proof["owner_slot_ceiling_mechanics_slot_write_count"] == total_steps
    assert (
        proof["owner_slot_ceiling_mechanics_slot_generation_verified_count"]
        == total_steps
    )
    assert proof["owner_slot_ceiling_mechanics_slot_digest_verified_count"] == total_steps
    assert proof["owner_slot_ceiling_root_request_from_slot_count"] == total_steps
    assert proof["owner_slot_ceiling_root_request_from_batch_count"] == 0
    assert proof["owner_slot_ceiling_hybrid_compact_batch_object_count"] == 0
    assert proof["owner_slot_ceiling_action_result_write_count"] == total_steps
    assert proof["owner_slot_ceiling_action_result_read_count"] == total_steps
    assert proof["owner_slot_ceiling_next_action_count"] == total_steps
    assert proof["owner_slot_ceiling_root_observation_copy_bytes"] == 0
    assert proof["owner_slot_ceiling_ctree_calls"] == 0
    assert proof["owner_slot_ceiling_tolist_calls"] == 0
    assert proof["owner_slot_ceiling_replay_slot_append_count"] == total_steps - 1
    assert proof["owner_slot_ceiling_replay_slot_append_row_count"] > 0
    assert proof["owner_slot_ceiling_replay_slot_object_entry_count"] == 0
    assert proof["owner_slot_ceiling_parent_replay_object_count"] == 0
    assert proof["owner_slot_ceiling_selected_group_object_count"] == 0
    assert proof["owner_slot_ceiling_sample_batch_built"] is True
    assert proof["owner_slot_ceiling_sample_gate_calls"] > 0
    assert proof["owner_slot_ceiling_sample_handle_create_count"] > 0
    assert proof["owner_slot_ceiling_sample_handle_resolve_count"] == (
        proof["owner_slot_ceiling_sample_handle_create_count"]
    )
    assert proof["owner_slot_ceiling_sample_handle_inline_resolve_count"] == (
        proof["owner_slot_ceiling_sample_handle_resolve_count"]
    )
    assert proof["owner_slot_ceiling_sample_handle_pending_count"] == 0
    assert proof["owner_slot_ceiling_sample_row_count"] > 0
    assert proof["owner_slot_ceiling_sample_target_row_count"] == (
        proof["owner_slot_ceiling_sample_row_count"]
    )
    assert proof["owner_slot_ceiling_stage_replay_transport_entry_count"] == (
        total_steps - 1
    )
    assert proof["owner_slot_ceiling_stage_replay_transition_entry_count"] == (
        total_steps - 1
    )
    assert proof["owner_slot_ceiling_stage_replay_payload_cache_hit_count"] == (
        total_steps - 1
    )
    assert proof["owner_slot_ceiling_stage_replay_payload_cache_miss_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_payload_release_count"] == (
        total_steps - 1
    )
    assert proof["owner_slot_ceiling_stage_replay_payload_pending_count"] == 1
    assert proof["owner_slot_ceiling_stage_replay_pending_record_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_ready_record_count"] == (
        total_steps - 1
    )
    assert proof["owner_slot_ceiling_stage_replay_drained_record_count"] == (
        total_steps - 1
    )
    assert proof["owner_slot_ceiling_stage_replay_index_rows_build_count"] == (
        total_steps - 1
    )
    assert proof["owner_slot_ceiling_stage_replay_index_rows_row_count"] == (
        proof["owner_slot_ceiling_replay_slot_append_row_count"]
    )
    assert proof["owner_slot_ceiling_stage_replay_device_index_rows_build_count"] == (
        total_steps - 1
    )
    assert proof["owner_slot_ceiling_stage_replay_device_index_rows_row_count"] == (
        proof["owner_slot_ceiling_stage_replay_index_rows_row_count"]
    )
    assert proof["owner_slot_ceiling_stage_replay_slot_append_count"] == (
        proof["owner_slot_ceiling_replay_slot_append_count"]
    )
    assert proof["owner_slot_ceiling_stage_replay_slot_append_row_count"] == (
        proof["owner_slot_ceiling_replay_slot_append_row_count"]
    )
    assert proof["owner_slot_ceiling_stage_sample_batch_built"] is True
    assert proof["owner_slot_ceiling_stage_sample_gate_calls"] == (
        proof["owner_slot_ceiling_sample_gate_calls"]
    )
    assert proof["owner_slot_ceiling_stage_sample_handle_create_count"] == (
        proof["owner_slot_ceiling_sample_handle_create_count"]
    )
    assert proof["owner_slot_ceiling_stage_sample_handle_resolve_count"] == (
        proof["owner_slot_ceiling_stage_sample_handle_create_count"]
    )
    assert proof["owner_slot_ceiling_stage_sample_handle_inline_resolve_count"] == (
        proof["owner_slot_ceiling_stage_sample_handle_resolve_count"]
    )
    assert proof["owner_slot_ceiling_stage_sample_handle_pending_count"] == 0
    assert proof["owner_slot_ceiling_stage_sample_row_count"] == (
        proof["owner_slot_ceiling_sample_row_count"]
    )
    assert proof["owner_slot_ceiling_stage_sample_target_row_count"] == (
        proof["owner_slot_ceiling_stage_sample_row_count"]
    )
    assert proof["owner_slot_ceiling_replay_ring_append_record_count"] == (
        total_steps - 1
    )
    assert proof["owner_slot_ceiling_replay_ring_append_call_count"] > 0
    assert proof["owner_slot_ceiling_replay_ring_appended_row_count"] == (
        proof["owner_slot_ceiling_stage_replay_index_rows_row_count"]
    )
    assert proof["owner_slot_ceiling_replay_ring_entry_count"] == total_steps - 1
    assert proof["owner_slot_ceiling_replay_ring_stored_index_row_count"] == (
        proof["owner_slot_ceiling_replay_ring_appended_row_count"]
    )
    assert proof["owner_slot_ceiling_replay_ring_evicted_entry_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_evicted_index_row_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_sample_batch_built"] is True
    assert proof["owner_slot_ceiling_replay_ring_sample_gate_calls"] > 0
    assert proof["owner_slot_ceiling_replay_ring_sample_row_count"] > 0
    assert proof["owner_slot_ceiling_replay_ring_sample_target_row_count"] == (
        proof["owner_slot_ceiling_replay_ring_sample_row_count"]
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_sample_source"]
        == "compact_replay_ring_resident_sample_gate"
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample"]
        is True
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample_all"]
        is True
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_sample_resident_device_sample_batch"]
        is True
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_sample_host_observation_fallback_allowed"]
        is False
    )
    assert proof["owner_slot_ceiling_replay_ring_sample_observation_provider_used_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_batch_built"] is True
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls"] > 0
    unroll2_rows = proof["owner_slot_ceiling_replay_ring_learner_unroll2_sample_row_count"]
    assert unroll2_rows > 0
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_row_count"] == (
        unroll2_rows
    )
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_num_unroll_steps"] == 2
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_require_next_targets"]
        is True
    )
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_batch_only"] is True
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_source"]
        == "compact_rollout_slab_resident_device_replay_grouped_learner_batch"
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source"]
        != "none"
    )
    assert (
        proof[
            "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count"
        ]
        >= 0
    )
    assert (
        proof[
            "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count"
        ]
        > 0
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_observation_provider_used_count"]
        == 0
    )
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_schema_id"] != "none"
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source"]
        == "resident_grouped_device_learner_batch_builder_v1"
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_host_fallback_allowed"]
        is False
    )
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_action_shape"] == [
        unroll2_rows,
        2,
    ]
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_shape"] == [
        unroll2_rows,
        2,
    ]
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_value_shape"] == [
        unroll2_rows,
        3,
    ]
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_shape"] == [
        unroll2_rows,
        3,
        fixed_action_benchmark.ACTION_COUNT,
    ]
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_action_mask_shape"] == [
        unroll2_rows,
        3,
        fixed_action_benchmark.ACTION_COUNT,
    ]
    assert proof["owner_slot_ceiling_replay_ring_columnar_append_call_count"] > 0.0
    assert proof["owner_slot_ceiling_replay_ring_columnar_append_record_count"] == (
        total_steps - 1
    )
    assert proof[
        "owner_slot_ceiling_replay_ring_columnar_append_entry_view_object_count"
    ] == (total_steps - 1)
    assert proof[
        "owner_slot_ceiling_replay_ring_columnar_append_step_view_object_count"
    ] == ((total_steps - 1) * 2)
    _assert_nonempty_digest(proof["owner_slot_ceiling_action_source_sequence_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_slot_digest_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_next_joint_action_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_replay_slot_window_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_sample_handle_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_sample_row_id_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_sample_action_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_sample_reward_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_sample_done_checksum"])
    _assert_nonempty_digest(
        proof["owner_slot_ceiling_stage_replay_slot_window_checksum"]
    )
    _assert_nonempty_digest(proof["owner_slot_ceiling_stage_sample_handle_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_stage_sample_row_id_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_stage_sample_action_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_stage_sample_reward_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_stage_sample_done_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_replay_ring_sample_row_id_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_replay_ring_sample_action_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_replay_ring_sample_reward_checksum"])
    _assert_nonempty_digest(proof["owner_slot_ceiling_replay_ring_sample_done_checksum"])
    _assert_nonempty_digest(
        proof["owner_slot_ceiling_replay_ring_sample_observation_checksum"]
    )
    _assert_nonempty_digest(
        proof["owner_slot_ceiling_replay_ring_sample_next_observation_checksum"]
    )
    _assert_nonempty_digest(
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_action_checksum"]
    )
    _assert_nonempty_digest(
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_checksum"]
    )
    _assert_nonempty_digest(
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_value_checksum"]
    )
    _assert_nonempty_digest(
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_checksum"]
    )
    _assert_nonempty_digest(
        proof[
            "owner_slot_ceiling_replay_ring_learner_unroll2_source_record_window_checksum"
        ]
    )
    assert proof["replay_index_rows_exposed"] is True

    assert result["fixed_buffer_direct"]["owner_slot_wall_sec"]["sum"] > 0.0
    ceiling = result["owner_buffer_ceiling"]
    assert ceiling["enabled"] is True
    assert ceiling["production_speed_claim"] is False
    assert ceiling["fixed_owner_slot_root_action_sec"] > 0.0
    assert ceiling["fixed_slab_replay_sample_sec"] == 0.0
    assert (
        "owner-slot ceiling drains staged owner rows into the real compact replay ring "
        "and builds a resident device learner unroll-2 batch locally; fixed resident "
        "row/window slots or handle-ring sampling are the next rung"
    ) in result["known_limits"]


def test_fixed_action_tape_owner_slot_ceiling_terminal_roots_do_not_fake_samples():
    result = fixed_action_benchmark.run_benchmark(
        fixed_action_benchmark.BenchmarkConfig(
            scenario="scenarios/environment/source_normal_wall_death_step.json",
            batch_size=4,
            warmup_steps=0,
            measured_steps=1,
            body_capacity=8,
            random_tape_capacity_min=256,
            render_observation=True,
            run_owner_slot_ceiling=True,
            require_pass=True,
        )
    )

    assert result["status"] == "pass"
    proof = result["proof"]
    assert proof["expects_terminal_rows"] is True
    assert proof["terminal_row_count"] == 4
    assert proof["owner_slot_ceiling_enabled"] is True
    assert proof["owner_slot_ceiling_proof_passed"] is True
    assert proof["owner_slot_ceiling_failure_reasons"] == []
    assert proof["owner_slot_ceiling_step_count"] == 1
    assert proof["owner_slot_ceiling_feedback_action_count"] == 0
    assert proof["owner_slot_ceiling_measured_feedback_action_count"] == 0
    assert proof["owner_slot_ceiling_selected_action_count"] == 0
    assert proof["owner_slot_ceiling_replay_slot_append_count"] == 0
    assert proof["owner_slot_ceiling_replay_slot_append_row_count"] == 0
    assert proof["owner_slot_ceiling_sample_batch_built"] is False
    assert proof["owner_slot_ceiling_sample_gate_calls"] == 0
    assert proof["owner_slot_ceiling_sample_handle_create_count"] == 0
    assert proof["owner_slot_ceiling_sample_handle_pending_count"] == 0
    assert proof["owner_slot_ceiling_sample_row_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_transport_entry_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_transition_entry_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_payload_cache_hit_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_payload_cache_miss_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_payload_release_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_payload_pending_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_pending_record_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_ready_record_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_drained_record_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_index_rows_build_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_index_rows_row_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_device_index_rows_build_count"] == 0
    assert proof["owner_slot_ceiling_stage_replay_device_index_rows_row_count"] == 0
    assert proof["owner_slot_ceiling_stage_sample_batch_built"] is False
    assert proof["owner_slot_ceiling_stage_sample_handle_create_count"] == 0
    assert proof["owner_slot_ceiling_stage_sample_handle_pending_count"] == 0
    assert proof["owner_slot_ceiling_stage_sample_row_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_append_record_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_append_call_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_appended_row_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_entry_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_stored_index_row_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_sample_batch_built"] is False
    assert proof["owner_slot_ceiling_replay_ring_sample_gate_calls"] == 0
    assert proof["owner_slot_ceiling_replay_ring_sample_row_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_sample_source"] == "none"
    assert (
        proof["owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample"]
        is False
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_sample_device_replay_index_rows_sample_all"]
        is False
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_sample_resident_device_sample_batch"]
        is False
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_sample_host_observation_fallback_allowed"]
        is False
    )
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_batch_built"] is False
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_sample_gate_calls"] == 0
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_sample_row_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_row_count"] == 0
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_num_unroll_steps"] == 0
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_require_next_targets"]
        is False
    )
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_batch_only"] is False
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_source"] == "none"
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_candidate_universe_source"]
        == "none"
    )
    assert (
        proof[
            "owner_slot_ceiling_replay_ring_learner_unroll2_explicit_unroll_target_group_count"
        ]
        == 0
    )
    assert (
        proof[
            "owner_slot_ceiling_replay_ring_learner_unroll2_next_target_eligible_pair_count"
        ]
        == 0
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_observation_provider_used_count"]
        == 0
    )
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_schema_id"] == "none"
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_prevalidation_source"]
        == "none"
    )
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_host_fallback_allowed"]
        is False
    )
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_action_shape"] == []
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_shape"] == []
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_value_shape"] == []
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_shape"] == []
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_action_mask_shape"] == []
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_action_checksum"] == ""
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_reward_checksum"] == ""
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_value_checksum"] == ""
    assert proof["owner_slot_ceiling_replay_ring_learner_unroll2_target_policy_checksum"] == ""
    assert (
        proof["owner_slot_ceiling_replay_ring_learner_unroll2_source_record_window_checksum"]
        == ""
    )
    assert proof["owner_slot_ceiling_replay_ring_columnar_append_call_count"] == 0.0
    assert proof["owner_slot_ceiling_replay_ring_columnar_append_record_count"] == 0.0
    assert proof["replay_index_rows_exposed"] is False


def test_fixed_action_tape_slab_replay_terminal_roots_do_not_fake_samples():
    result = fixed_action_benchmark.run_benchmark(
        fixed_action_benchmark.BenchmarkConfig(
            scenario="scenarios/environment/source_normal_wall_death_step.json",
            batch_size=4,
            warmup_steps=0,
            measured_steps=1,
            body_capacity=8,
            random_tape_capacity_min=256,
            render_observation=True,
            run_slab_replay=True,
            require_pass=True,
        )
    )

    assert result["status"] == "pass"
    assert result["comparison"]["passed"] is True
    assert all(result["comparison"]["field_matches"].values())
    assert all(result["comparison"]["output_matches"].values())
    assert all(result["comparison"]["scalar_matches"].values())

    proof = result["proof"]
    assert proof["passed"] is True
    assert proof["expects_terminal_rows"] is True
    assert proof["expected_terminal_autoreset_evidence_present"] is True
    assert proof["terminal_row_count"] == 4
    assert proof["autoreset_call_count"] == 1
    assert proof["autoreset_row_count"] == 4
    assert proof["terminal_rows_equal_autoreset_rows"] is True
    assert proof["zero_observation_stub"] is False
    assert proof["search_enabled"] is False
    assert proof["search_proof_passed"] is True
    assert proof["slab_replay_enabled"] is True
    assert proof["slab_replay_metadata_match"] is True
    assert proof["slab_replay_proof_passed"] is True
    assert proof["slab_replay_failure_reasons"] == []
    assert proof["slab_replay_expected_total_steps"] == 1
    assert proof["slab_replay_expected_root_count"] == 8
    assert proof["slab_replay_expected_measured_feedback_action_count"] == 0
    assert proof["slab_replay_expected_append_count"] == 0
    assert proof["slab_search_feedback_closed_loop"] is True
    assert proof["slab_replay_tape_bootstrap_action_count"] == 1
    assert proof["slab_replay_feedback_action_count"] == 0
    assert proof["slab_replay_measured_feedback_action_count"] == 0
    assert proof["slab_replay_prev_next_joint_action_match_count"] == 0
    assert proof["slab_replay_prev_next_joint_action_mismatch_count"] == 0
    assert proof["slab_replay_feedback_differs_from_tape_count"] == 0
    _assert_nonempty_digest(proof["slab_replay_action_source_sequence_checksum"])
    assert proof["slab_step_count"] == 1
    assert proof["slab_root_count"] == 8
    assert proof["slab_active_root_count"] == 0
    assert proof["slab_inactive_root_count"] == 8
    assert proof["slab_selected_action_count"] == 0
    assert proof["slab_action_d2h_bytes"] == 0
    assert proof["slab_deferred_replay_payload_d2h_bytes"] == 0
    assert proof["slab_replay_payload_d2h_bytes"] == 0
    assert proof["slab_replay_payload_flush_count"] == 0
    assert proof["slab_committed_index_group_count"] == 0
    assert proof["slab_committed_index_row_count"] == 0
    assert proof["slab_replay_pending_uncommitted_count"] == 1
    assert proof["slab_retains_committed_index_rows"] is False
    assert proof["replay_append_count"] == 0
    assert proof["replay_ring_entry_count"] == 0
    assert proof["replay_ring_stored_index_row_count"] == 0
    assert proof["sample_gate_calls"] == 0
    assert proof["sample_row_count"] == 0
    assert proof["slab_replay_sample_batch_built"] is False
    assert proof["replay_index_rows_exposed"] is False
    assert proof["slab_replay_action_check_enforced"] is True
    assert proof["slab_replay_root_observation_copied"] is False
    assert proof["slab_replay_index_rows_observation_materialized"] is False
    assert proof["slab_replay_index_rows_next_observation_materialized"] is False
