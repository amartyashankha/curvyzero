import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / (
    "run_curvytron_hybrid_observation_profile_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("hybrid_observation_profile_runner", SCRIPT_PATH)
assert SPEC is not None
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def _normal_death_payload_fields():
    return {
        "death_mode": "normal",
        "terminal_row_count": 8,
        "done_semantics_verified": True,
        "terminated_row_count": 8,
        "truncated_row_count": 0,
        "death_row_count": 8,
        "death_count_total": 8,
        "death_cause_count_by_name": {
            "none": 0,
            "wall": 0,
            "own_trail": 0,
            "opponent_trail": 8,
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
        "terminal_final_observation_row_count": 8,
        "terminal_final_observation_before_autoreset_verified": True,
        "terminal_final_reward_map_row_count": 8,
        "terminal_final_reward_map_matches_reward_row_count": 8,
        "terminal_final_reward_map_verified": True,
    }


def _valid_row(command_extra=None, **overrides):
    command_extra = command_extra or []
    row = {
        "row_id": "001",
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "death_mode": "profile_no_death",
        "command": [
            "uv",
            "run",
            "--extra",
            "modal",
            "modal",
            "run",
            "-m",
            runner.BOUNDARY_MODULE,
            "--hybrid-observation-canary",
            "--death-mode",
            "profile_no_death",
            *command_extra,
        ],
    }
    row.update(overrides)
    return row


def _root_tape_row(command_extra=None, **overrides):
    command_extra = command_extra or []
    row = _valid_row(
        [
            "--hybrid-compact-rollout-slab-probe",
            "--hybrid-compact-root-tape-compare",
            *command_extra,
        ],
        compute="gpu-h100",
        compact_rollout_slab_probe=True,
        compact_rollout_slab_action_mode="search_feedback",
        compact_root_tape_compare=True,
        compact_root_tape_max_records=4,
        compact_root_tape_compare_fixed_shape_floor=True,
        compact_root_tape_compare_mctx=False,
        compact_root_tape_compare_model_compile=False,
        compact_root_tape_compare_direct_core=False,
        compact_root_tape_model_compile_mode="default",
        compact_root_tape_require_model_compile=True,
        compact_root_tape_reference_label="primary",
        compact_root_tape_allow_resident_host_snapshot=False,
        hybrid_resident_observation_search=False,
        lightzero_array_ceiling_probe=True,
        lightzero_array_ceiling_mode="compact_torch_search_service",
        lightzero_consumer_root_noise_weight=0.0,
        probe_simulations=8,
        require_payload_label_triad=True,
    )
    row.update(overrides)
    return row


def _root_tape_payload(
    *,
    record_count=4,
    max_records=4,
    include_fixed_shape=True,
    include_mctx=False,
    include_model_compile=False,
    include_direct_core=False,
):
    backend = {
        "primary": {
            "run_count": record_count,
            "active_root_count": 16,
            "run_sec": 1.0,
            "run_sec_per_active_root": 0.0625,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
        }
    }
    service_labels = ["primary"]
    comparison = {}
    if include_fixed_shape:
        backend["fixed_shape_floor"] = {
            "run_count": record_count,
            "active_root_count": 16,
            "run_sec": 1.25,
            "run_sec_per_active_root": 0.078125,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
        }
        service_labels.insert(0, "fixed_shape_floor")
        comparison["fixed_shape_floor_vs_primary"] = {
            "record_count": record_count,
            "active_root_count": 16,
            "action_match_count": 16.0,
            "action_match_fraction": 1.0,
            "visit_l1_mean": 0.25,
            "visit_l1_max": 0.5,
            "root_value_abs_diff_mean": 0.00001,
            "root_value_abs_diff_max": 0.00002,
        }
    if include_mctx:
        backend["mctx"] = {
            "run_count": record_count,
            "active_root_count": 16,
            "run_sec": 2.0,
            "run_sec_per_active_root": 0.125,
            "h2d_bytes": 1024,
            "d2h_bytes": 256,
        }
        comparison["mctx_vs_primary"] = {
            "record_count": record_count,
            "active_root_count": 16,
            "action_match_count": 12.0,
            "action_match_fraction": 0.75,
            "visit_l1_mean": 0.5,
            "visit_l1_max": 1.0,
            "root_value_abs_diff_mean": 0.1,
            "root_value_abs_diff_max": 0.2,
        }
        service_labels.insert(1, "mctx")
    if include_model_compile:
        backend["model_compile_default"] = {
            "run_count": record_count,
            "active_root_count": 16,
            "run_sec": 0.9,
            "run_sec_per_active_root": 0.05625,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "model_compile_requested_count": record_count,
            "model_compile_used_count": record_count,
            "model_compile_cache_hit_count": record_count - 1,
            "model_compile_runtime_status_counts": {
                "compiled": 1,
                "cache_hit": record_count - 1,
            },
        }
        comparison["model_compile_default_vs_primary"] = {
            "record_count": record_count,
            "active_root_count": 16,
            "action_match_count": 16.0,
            "action_match_fraction": 1.0,
            "visit_l1_mean": 0.0,
            "visit_l1_max": 0.0,
            "root_value_abs_diff_mean": 0.0,
            "root_value_abs_diff_max": 0.0,
        }
        service_labels.insert(1, "model_compile_default")
    if include_direct_core:
        backend["initial_inference_direct_core"] = {
            "run_count": record_count,
            "active_root_count": 16,
            "run_sec": 0.85,
            "run_sec_per_active_root": 0.053125,
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "initial_inference_direct_requested_count": record_count,
            "initial_inference_direct_used_count": record_count,
            "initial_inference_fallback_count": 0,
            "initial_inference_runtime_status_counts": {
                "direct_core_used": record_count,
            },
        }
        comparison["initial_inference_direct_core_vs_primary"] = {
            "record_count": record_count,
            "active_root_count": 16,
            "action_match_count": 16.0,
            "action_match_fraction": 1.0,
            "visit_l1_mean": 0.0,
            "visit_l1_max": 0.0,
            "root_value_abs_diff_mean": 0.0,
            "root_value_abs_diff_max": 0.0,
        }
        service_labels.insert(1, "initial_inference_direct_core")
    return {
        "ok": True,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "steps_per_sec": 1000.0,
        "measured_sec": 2.0,
        "compact_rollout_slab_enabled": True,
        "compact_rollout_slab_total_roots": 128,
        "timings": {"compact_rollout_slab_sec": 0.5},
        "compact_rollout_slab_last_telemetry": {
            "compact_rollout_slab_search_impl": "compact_torch_device_tree_fixed_shape_v0"
        },
        "compact_root_tape_compare_enabled": True,
        "compact_root_tape_record_count": record_count,
        "compact_root_tape_skipped_record_count": 26,
        "compact_root_tape_reference_label": "primary",
        "compact_root_tape_service_labels": service_labels,
        "compact_root_tape_error": "",
        "compact_root_tape_comparison": {
            "schema_id": runner.COMPACT_ROOT_TAPE_COMPARISON_SCHEMA_ID,
            "record_count": record_count,
            "reference_label": "primary",
            "tape_metadata": {
                "record_count": record_count,
                "skipped_record_count": 26,
                "max_records": max_records,
                "profile_only": True,
                "calls_train_muzero": False,
                "action_mode": "search_feedback",
                "root_noise_weight": 0.0,
            },
            "backend": backend,
            "comparison": comparison,
            "per_record": [],
        },
    }


def _compact_owned_candidate_row(command_extra=None, **overrides):
    command_extra = command_extra or []
    num_unroll_steps = int(overrides.get("compact_rollout_slab_learner_gate_num_unroll_steps", 1))
    row = _valid_row(
        [
            "--no-hybrid-materialize-scalar-timestep",
            "--hybrid-device-only-stack",
            "--hybrid-resident-observation-search",
            "--hybrid-compact-rollout-slab-probe",
            "--hybrid-compact-rollout-slab-sample-gate",
            "--hybrid-compact-rollout-slab-learner-gate",
            "--hybrid-compact-rollout-slab-learner-gate-impl",
            "compact_muzero",
            "--hybrid-compact-rollout-slab-learner-gate-num-unroll-steps",
            str(num_unroll_steps),
            *command_extra,
        ],
        compute="gpu-h100",
        materialize_scalar_timestep=False,
        compact_rollout_slab_probe=True,
        compact_rollout_slab_action_mode="search_feedback",
        compact_rollout_slab_sample_gate=True,
        compact_rollout_slab_sample_gate_batch_size=32,
        compact_rollout_slab_sample_gate_interval=1,
        compact_rollout_slab_learner_gate=True,
        compact_rollout_slab_learner_gate_impl="compact_muzero",
        compact_rollout_slab_learner_gate_train_steps=1,
        compact_rollout_slab_learner_gate_num_unroll_steps=num_unroll_steps,
        compact_rollout_slab_learner_gate_include_rnd=False,
        hybrid_device_only_stack=True,
        hybrid_resident_observation_search=True,
        require_payload_label_triad=True,
    )
    row.update(overrides)
    return row


def _compact_owned_candidate_payload(**overrides):
    payload = {
        "ok": True,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "death_mode": "profile_no_death",
        "steps_per_sec": 20000.0,
        "measured_sec": 2.0,
        "resident_observation_used": True,
        "resident_observation_host_fallback_count": 0.0,
        "resident_observation_h2d_bytes": 0.0,
        "resident_observation_d2h_bytes": 0.0,
        "compact_rollout_slab_enabled": True,
        "compact_rollout_slab_calls": 2,
        "compact_rollout_slab_total_roots": 64,
        "compact_rollout_slab_roots_per_call": 32.0,
        "compact_rollout_slab_committed_index_row_count": 64,
        "compact_rollout_slab_action_mode": "search_feedback",
        "compact_rollout_slab_sample_gate_enabled": True,
        "compact_rollout_slab_sample_gate_calls": 2,
        "compact_rollout_slab_sample_gate_opportunities": 3,
        "compact_rollout_slab_sample_gate_skipped_count": 1,
        "compact_rollout_slab_sample_gate_index_row_count": 128,
        "compact_rollout_slab_sample_gate_target_row_count": 64,
        "compact_rollout_slab_sample_gate_sample_row_count": 64,
        "compact_rollout_slab_sample_gate_batch_size": 32,
        "compact_rollout_slab_sample_gate_interval": 1,
        "compact_rollout_slab_sample_gate_mock_base_env_timestep_rows": 0,
        "compact_rollout_slab_learner_gate_enabled": True,
        "compact_rollout_slab_learner_gate_calls": 2,
        "compact_rollout_slab_learner_gate_updates": 2,
        "compact_rollout_slab_learner_gate_sample_row_count": 64,
        "compact_rollout_slab_learner_gate_input_bytes": 4096,
        "compact_rollout_slab_learner_gate_train_steps": 1,
        "compact_rollout_slab_learner_gate_include_rnd": False,
        "compact_rollout_slab_learner_gate_impl": "compact_muzero",
        "compact_rollout_slab_learner_gate_toy_probe": False,
        "compact_rollout_slab_learner_gate_real_muzero_update": True,
        "compact_rollout_slab_learner_gate_support_scale": 300,
        "compact_rollout_slab_learner_gate_num_unroll_steps": 1,
        "timings": {
            "compact_rollout_slab_sec": 0.5,
            "compact_rollout_slab_sample_gate_sec": 0.01,
            "compact_rollout_slab_learner_gate_sec": 0.02,
            "scalar_materialization_sec": 0.0,
        },
        "compact_rollout_slab_telemetry_totals": {
            "compact_rollout_slab_model_sec": 0.1,
            "compact_rollout_slab_search_sec": 0.2,
            "compact_rollout_slab_h2d_sec": 0.0,
            "compact_rollout_slab_obs_h2d_bytes": 0.0,
            "compact_rollout_slab_replay_payload_d2h_bytes": 0.0,
            "compact_rollout_slab_committed_replay_payload_d2h_bytes": 0.0,
            "compact_rollout_slab_python_rows_materialized": 0.0,
        },
        "compact_rollout_slab_last_telemetry": {
            "compact_rollout_slab_search_impl": (
                "compact_torch_device_tree_fixed_shape_v0"
            ),
            "compact_rollout_slab_profile_telemetry": {
                "resident_observation_used": True,
                "resident_observation_host_fallback_count": 0.0,
                "resident_observation_h2d_bytes": 0.0,
                "resident_observation_d2h_bytes": 0.0,
            },
        },
    }
    payload.update(overrides)
    return payload


def _matched_denominator_compact_row(command_extra=None, **overrides):
    command_extra = command_extra or []
    row = _compact_owned_candidate_row(
        [
            "--batch-size",
            "1024",
            "--actor-count",
            "16",
            "--steps",
            "60",
            "--warmup-steps",
            "15",
            "--max-ticks",
            "2000",
            "--hybrid-lightzero-array-ceiling-probe",
            "--hybrid-lightzero-array-ceiling-mode",
            "compact_torch_search_service",
            "--hybrid-lightzero-array-ceiling-input-mode",
            "host_uint8",
            "--hybrid-lightzero-consumer-root-noise-weight",
            "0.0",
            "--hybrid-compact-rollout-slab-sample-gate-batch-size",
            "512",
            "--hybrid-compact-rollout-slab-sample-gate-interval",
            "8",
            "--hybrid-compact-rollout-slab-sample-gate-replay-pair-capacity",
            "4096",
            "--hybrid-compact-rollout-slab-learner-gate-support-scale",
            "300",
            "--hybrid-compact-owned-loop-entrypoint",
            "--hybrid-compact-owned-loop-policy-version-ref",
            "matched-denominator-compact-owned-policy-v1",
            "--hybrid-compact-owned-loop-policy-source",
            "matched_denominator_compact_owned_profile",
            "--hybrid-compact-owned-loop-model-version-ref",
            "matched-denominator-compact-owned-model-v1",
            "--hybrid-compact-owned-loop-capture-replay-store-state",
            "--hybrid-native-actor-buffer",
            "--detach",
            "--hybrid-profile-spawn-result",
            *command_extra,
        ],
        actor_count=16,
        batch_size=1024,
        compact_rollout_slab_sample_gate_batch_size=512,
        compact_rollout_slab_sample_gate_interval=8,
        compact_rollout_slab_sample_gate_replay_pair_capacity=4096,
        compact_rollout_slab_learner_gate_device="cuda",
        compact_rollout_slab_learner_gate_support_scale=300,
        compact_owned_loop_entrypoint=True,
        compact_owned_loop_policy_version_ref="matched-denominator-compact-owned-policy-v1",
        compact_owned_loop_model_version_ref="matched-denominator-compact-owned-model-v1",
        compact_owned_loop_policy_source="matched_denominator_compact_owned_profile",
        compact_owned_loop_capture_replay_store_state=True,
        counterpart_manifest_ref="stock.json",
        counterpart_row_id="001",
        device_latest=False,
        fixed_denominator={
            "matched_denominator_id": "unit-denominator",
            "matched_pair_role": "compact_candidate",
            "row_purpose": runner.MATCHED_DENOMINATOR_ROW_PURPOSE,
            "speed_currency": runner.MATCHED_COMPACT_SPEED_CURRENCY,
        },
        hybrid_native_actor_buffer=True,
        launch_mode=runner.LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT,
        lightzero_array_ceiling_probe=True,
        lightzero_array_ceiling_input_mode="host_uint8",
        lightzero_array_ceiling_mode="compact_torch_search_service",
        lightzero_consumer_root_noise_weight=0.0,
        matched_denominator_id="unit-denominator",
        matched_pair_role="compact_candidate",
        max_ticks=2000,
        promotion_claim=False,
        probe_simulations=8,
        require_terminal_compact_owned_nstep=False,
        result_capture=runner.RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET,
        row_purpose=runner.MATCHED_DENOMINATOR_ROW_PURPOSE,
        speed_currency=runner.MATCHED_COMPACT_SPEED_CURRENCY,
        steps=60,
        warmup_steps=15,
    )
    row.update(overrides)
    return row


def _matched_denominator_compact_payload(**overrides):
    payload = _compact_owned_candidate_payload(
        actor_count=16,
        batch_size=1024,
        compact_rollout_slab_sample_gate_batch_size=512,
        compact_rollout_slab_sample_gate_interval=8,
        compact_rollout_slab_learner_gate_support_scale=300,
        compact_rollout_slab_learner_gate_num_unroll_steps=1,
        compact_owned_loop_entrypoint_enabled=True,
        compact_owned_loop_schema_id=runner.COMPACT_OWNED_LOOP_SCHEMA_ID,
        compact_owned_loop_profile_only=True,
        compact_owned_loop_calls_train_muzero=False,
        compact_owned_loop_touches_live_runs=False,
        compact_owned_loop_replay_store_owned=True,
        compact_owned_loop_policy_version_handoff=True,
        compact_owned_loop_policy_version_ref="matched-denominator-compact-owned-policy-v1",
        compact_owned_loop_model_version_ref="matched-denominator-compact-owned-model-v1",
        compact_owned_loop_policy_source="matched_denominator_compact_owned_profile",
        compact_owned_loop_telemetry={
            "compact_owned_loop_policy_version_ref": (
                "matched-denominator-compact-owned-policy-v1"
            ),
            "compact_owned_loop_sample_gate_last_sample_metadata": {
                "compact_owned_loop_replay_store_owned": True,
                "compact_owned_loop_policy_version_ref": (
                    "matched-denominator-compact-owned-policy-v1"
                ),
            },
        },
        compact_owned_loop_replay_store_state_metadata={
            "schema_id": runner.COMPACT_REPLAY_STORE_STATE_SCHEMA_ID,
            "compact_owned_loop_replay_store_owned": True,
        },
        max_ticks=2000,
        physical_rows_per_sec=10000.0,
        rows_per_step=2048,
        steps=60,
        terminal_row_count=0,
        warmup_steps=15,
    )
    payload.update(overrides)
    return payload


def _install_detached_hybrid_result(monkeypatch, payload):
    class Completed:
        returncode = 0
        stdout = json.dumps(
            {
                "schema_id": runner.SPAWN_SCHEMA_ID,
                "status": "spawned",
                "function_call_id": "fc-unit",
                "profile_only": True,
                "calls_train_muzero": False,
                "touches_live_runs": False,
            }
        )

    class FakeCall:
        def get(self, *, timeout=None):
            return payload

    class FakeFunctionCall:
        @staticmethod
        def from_id(function_call_id):
            assert function_call_id == "fc-unit"
            return FakeCall()

    class FakeModal:
        FunctionCall = FakeFunctionCall

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())
    monkeypatch.setattr(runner, "modal", FakeModal)


def test_hybrid_profile_runner_row_selection_accepts_numeric_aliases():
    manifest = {
        "rows": [
            {"row_id": "001", "value": "first"},
            {"row_id": "002", "value": "second"},
        ]
    }

    selected = runner._selected_rows(manifest, runner._parse_rows(["2"]))

    assert [row["value"] for row in selected] == ["second"]


def test_hybrid_profile_runner_preflight_rejects_detached_rows():
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [_valid_row(["--detach"])],
    }

    try:
        runner._validate_manifest(manifest)
    except SystemExit as exc:
        assert "remove --detach" in str(exc)
    else:
        raise AssertionError("expected detached preflight failure")


def test_hybrid_profile_runner_preflight_accepts_explicit_detached_result_capture():
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [
            _valid_row(
                ["--detach", "--hybrid-profile-spawn-result"],
                launch_mode=runner.LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT,
                result_capture=runner.RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET,
                captured_result_required=True,
                require_payload_label_triad=True,
            )
        ],
    }

    runner._validate_manifest(manifest)


def test_hybrid_profile_runner_parser_prefers_profile_result_json():
    stdout = """
warning line
{"nested": {"ok": false}}
{
  "ok": true,
  "profile_only": true,
  "steps_per_sec": 123.0,
  "batched_stack_probe_last_telemetry": {
    "lightzero_array_ceiling_semantics": "mock_search_service"
  }
}
"""

    payload = runner._extract_last_json_object(stdout)

    assert payload["ok"] is True
    assert payload["profile_only"] is True
    assert payload["steps_per_sec"] == 123.0


def test_hybrid_profile_runner_parser_prefers_spawn_json_when_present():
    stdout = """
{
  "ok": true,
  "profile_only": true
}
{
  "schema_id": "curvyzero_hybrid_observation_profile_spawn/v0",
  "status": "spawned",
  "function_call_id": "fc-test",
  "profile_only": true,
  "calls_train_muzero": false,
  "touches_live_runs": false
}
"""

    payload = runner._extract_last_json_object(stdout)

    assert payload["schema_id"] == runner.SPAWN_SCHEMA_ID
    assert payload["function_call_id"] == "fc-test"


def test_hybrid_profile_runner_summary_uses_aggregate_timings_not_last_telemetry():
    row = {
        "row_id": "001",
        "compute": "gpu-h100",
        "lightzero_array_ceiling_probe": True,
        "lightzero_array_ceiling_mode": "service_tax_probe",
        "probe_simulations": 16,
    }
    payload = {
        "ok": True,
        "steps_per_sec": 100.0,
        "measured_sec": 20.0,
        "batched_stack_probe_total_roots": 2000,
        "compact_service_replay_proof_calls": 19,
        "compact_service_replay_proof_warmup_seeded_calls": 1,
        "compact_service_replay_proof_target_row_count": 1900,
        "compact_service_replay_proof_sec": 0.2,
        "timings": {
            "actor_step_wall_sec": 5.0,
            "gather_merge_sec": 0.5,
            "observation_sec": 6.0,
            "compact_batch_build_sec": 1.0,
            "lightzero_array_ceiling_total_sec": 4.0,
            "lightzero_array_ceiling_initial_inference_sec": 1.0,
            "lightzero_array_ceiling_recurrent_inference_sec": 2.0,
            "lightzero_array_ceiling_search_update_sec": 0.5,
            "lightzero_array_ceiling_h2d_sec": 0.25,
            "scalar_materialization_sec": 0.0,
            "compact_payload_pickle_sec": 1.5,
        },
        "batched_stack_probe_ledger_totals": {
            "lightzero_array_ceiling_obs_h2d_bytes": 1024.0,
            "lightzero_array_ceiling_mask_h2d_bytes": 24.0,
            "lightzero_array_ceiling_action_d2h_bytes": 8.0,
            "lightzero_array_ceiling_replay_payload_d2h_bytes": 64.0,
            "lightzero_array_ceiling_root_observation_copy_bytes": 0.0,
            "lightzero_array_ceiling_python_rows_materialized": 0.0,
            "lightzero_array_ceiling_rnd_materialized_rows": 0.0,
            "lightzero_array_ceiling_resident_obs_reused": 0.0,
        },
        "batched_stack_probe_last_telemetry": {
            "lightzero_array_ceiling_total_sec": 999.0,
            "lightzero_array_ceiling_semantics": "service_tax_probe",
            "lightzero_array_ceiling_root_noise_weight": 0.0,
            "lightzero_array_ceiling_compile_status": "not_requested",
            "lightzero_array_ceiling_compile_reason": "compile_not_requested",
        },
    }

    summary = runner._compact_line(row, payload)

    assert summary["probe_total_sec"] == 4.0
    assert summary["profile_only"] is True
    assert summary["calls_train_muzero"] is False
    assert summary["touches_live_runs"] is False
    assert summary["promotion_eligible"] is False
    assert summary["promotion_blocker"] == "profile_only_boundary_probe"
    assert summary["probe_roots_per_sec"] == 500.0
    assert summary["model_sec"] == 3.0
    assert summary["search_sec"] == 0.5
    assert summary["h2d_sec"] == 0.25
    assert summary["accounting_actor_wall_sec"] == 5.0
    assert summary["accounting_observation_sec"] == 6.0
    assert summary["accounting_compact_batch_build_sec"] == 1.0
    assert summary["accounting_probe_sec"] == 4.0
    assert summary["accounting_compact_payload_pickle_sec"] == 1.5
    assert summary["accounting_known_sec"] == 18.0
    assert summary["accounting_other_sec"] == 2.0
    assert summary["accounting_other_fraction"] == 0.1
    assert summary["obs_h2d_bytes"] == 1024.0
    assert summary["mask_h2d_bytes"] == 24.0
    assert summary["action_d2h_bytes"] == 8.0
    assert summary["replay_payload_d2h_bytes"] == 64.0
    assert summary["root_observation_copy_bytes"] == 0.0
    assert summary["python_rows_materialized"] == 0.0
    assert summary["rnd_materialized_rows"] == 0.0
    assert summary["resident_obs_reused"] == 0.0
    assert summary["root_noise_weight"] == 0.0
    assert summary["compile_status"] == "not_requested"
    assert summary["compile_reason"] == "compile_not_requested"
    assert summary["compact_service_replay_proof_warmup_seeded_calls"] == 1


def test_hybrid_profile_runner_rejects_manifest_payload_label_mismatch(tmp_path, monkeypatch):
    class Completed:
        returncode = 0
        stdout = """
{
  "ok": true,
  "profile_only": true,
  "calls_train_muzero": true,
  "touches_live_runs": false
}
"""

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(_valid_row(), tmp_path)

    assert record["status"] == "profile_label_mismatch"
    assert "calls_train_muzero" in record["problem"]
    assert "summary" not in record


def test_hybrid_profile_runner_requires_payload_label_triad_when_requested(
    tmp_path,
    monkeypatch,
):
    class Completed:
        returncode = 0
        stdout = """
{
  "ok": true,
  "profile_only": true,
  "touches_live_runs": false
}
"""

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(
        _valid_row(require_payload_label_triad=True),
        tmp_path,
    )

    assert record["status"] == "profile_label_mismatch"
    assert "calls_train_muzero" in record["problem"]
    assert "missing from payload" in record["problem"]


def test_hybrid_profile_runner_records_command_timeout(tmp_path, monkeypatch):
    def fake_run(*_args, **kwargs):
        raise runner.subprocess.TimeoutExpired(
            cmd=["modal"],
            timeout=kwargs["timeout"],
            output="partial launch output",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    record = runner._run_row(_valid_row(row_timeout_sec=12.0), tmp_path)

    assert record["status"] == "command_timeout"
    assert "12.0" in record["problem"]
    assert Path(record["stdout_path"]).read_text(encoding="utf-8") == (
        "partial launch output"
    )


def test_hybrid_profile_runner_collects_detached_function_call_result(
    tmp_path,
    monkeypatch,
):
    class Completed:
        returncode = 0
        stdout = """
{
  "schema_id": "curvyzero_hybrid_observation_profile_spawn/v0",
  "status": "spawned",
  "function_call_id": "fc-test",
  "profile_only": true,
  "calls_train_muzero": false,
  "touches_live_runs": false
}
"""

    captured = {}
    result_payload = {
        "ok": True,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "steps_per_sec": 123.0,
        "timings": {},
    }

    class FakeCall:
        def get(self, *, timeout=None):
            captured["timeout"] = timeout
            return result_payload

    class FakeFunctionCall:
        @staticmethod
        def from_id(function_call_id):
            captured["function_call_id"] = function_call_id
            return FakeCall()

    class FakeModal:
        FunctionCall = FakeFunctionCall

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())
    monkeypatch.setattr(runner, "modal", FakeModal)

    row = _valid_row(
        ["--detach", "--hybrid-profile-spawn-result"],
        launch_mode=runner.LAUNCH_MODE_DETACHED_FUNCTION_CALL_RESULT,
        result_capture=runner.RESULT_CAPTURE_MODAL_FUNCTION_CALL_GET,
        result_timeout_sec=99.0,
        captured_result_required=True,
        require_payload_label_triad=True,
    )
    record = runner._run_row(row, tmp_path)

    assert record["status"] == "complete"
    assert record["function_call_id"] == "fc-test"
    assert captured == {"function_call_id": "fc-test", "timeout": 99.0}
    assert record["launch"]["schema_id"] == runner.SPAWN_SCHEMA_ID
    assert record["compact"] is result_payload
    assert record["summary"]["steps_per_sec"] == 123.0


def test_hybrid_profile_runner_flattened_root_tape_summary(tmp_path, monkeypatch):
    class Completed:
        returncode = 0
        stdout = json.dumps(_root_tape_payload())

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(_root_tape_row(), tmp_path)

    assert record["status"] == "complete"
    summary = record["summary"]
    assert summary["compact_root_tape_record_count"] == 4
    assert summary["compact_root_tape_metadata_max_records"] == 4
    assert summary["compact_root_tape_backend_primary_run_count"] == 4
    assert summary[
        "compact_root_tape_fixed_shape_floor_vs_primary_action_match_fraction"
    ] == 1.0


def test_hybrid_profile_runner_accepts_and_flattens_root_tape_mctx_service(
    tmp_path,
    monkeypatch,
):
    class Completed:
        returncode = 0
        stdout = json.dumps(_root_tape_payload(include_mctx=True))

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    row = _root_tape_row(
        ["--hybrid-compact-root-tape-compare-mctx"],
        compact_root_tape_compare_mctx=True,
    )
    record = runner._run_row(row, tmp_path)

    assert record["status"] == "complete"
    summary = record["summary"]
    assert summary["compact_root_tape_backend_mctx_run_count"] == 4
    assert summary["compact_root_tape_backend_mctx_h2d_bytes"] == 1024
    assert summary["compact_root_tape_mctx_vs_primary_action_match_fraction"] == 0.75
    assert summary["compact_root_tape_mctx_vs_primary_root_value_abs_diff_max"] == 0.2


def test_hybrid_profile_runner_accepts_root_tape_model_compile_service(
    tmp_path,
    monkeypatch,
):
    class Completed:
        returncode = 0
        stdout = json.dumps(
            _root_tape_payload(
                include_fixed_shape=False,
                include_model_compile=True,
            )
        )

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    row = _root_tape_row(
        [
            "--no-hybrid-compact-root-tape-compare-fixed-shape-floor",
            "--hybrid-compact-root-tape-compare-model-compile",
            "--hybrid-compact-root-tape-model-compile-mode",
            "default",
        ],
        compact_root_tape_compare_fixed_shape_floor=False,
        compact_root_tape_compare_model_compile=True,
        compact_root_tape_model_compile_mode="default",
    )
    record = runner._run_row(row, tmp_path)

    assert record["status"] == "complete"
    summary = record["summary"]
    assert summary["compact_root_tape_backend_model_compile_default_run_count"] == 4
    assert (
        summary["compact_root_tape_backend_model_compile_default_model_compile_used_count"]
        == 4
    )
    assert (
        summary[
            "compact_root_tape_model_compile_default_vs_primary_action_match_fraction"
        ]
        == 1.0
    )


def test_hybrid_profile_runner_accepts_root_tape_direct_core_service(
    tmp_path,
    monkeypatch,
):
    class Completed:
        returncode = 0
        stdout = json.dumps(
            _root_tape_payload(
                include_fixed_shape=False,
                include_direct_core=True,
            )
        )

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    row = _root_tape_row(
        [
            "--no-hybrid-compact-root-tape-compare-fixed-shape-floor",
            "--hybrid-compact-root-tape-compare-direct-core",
        ],
        compact_root_tape_compare_fixed_shape_floor=False,
        compact_root_tape_compare_direct_core=True,
        compact_torch_initial_inference_mode="model_method",
    )
    record = runner._run_row(row, tmp_path)

    assert record["status"] == "complete"
    summary = record["summary"]
    assert (
        summary["compact_root_tape_backend_initial_inference_direct_core_run_count"]
        == 4
    )
    assert (
        summary[
            "compact_root_tape_initial_inference_direct_core_vs_primary_action_match_fraction"
        ]
        == 1.0
    )


def test_hybrid_profile_runner_fails_closed_on_root_tape_invariant(tmp_path, monkeypatch):
    class Completed:
        returncode = 0
        stdout = json.dumps(_root_tape_payload(record_count=3))

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(_root_tape_row(), tmp_path)

    assert record["status"] == "fixed_root_tape_invariant_failed"
    assert "compact_root_tape_record_count" in record["problem"]
    assert "summary" not in record


def test_hybrid_profile_runner_preflight_rejects_compact_owned_scalar_materialization():
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [_compact_owned_candidate_row(materialize_scalar_timestep=True)],
    }

    with pytest.raises(SystemExit, match="materialize_scalar_timestep"):
        runner._validate_manifest(manifest)


def test_hybrid_profile_runner_preflight_rejects_death_mode_command_mismatch():
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [_valid_row(death_mode="normal")],
    }

    with pytest.raises(SystemExit, match="death_mode"):
        runner._validate_manifest(manifest)


def test_hybrid_profile_runner_preflight_rejects_zero_root_terminal_nstep_shape():
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [
            _compact_owned_candidate_row(
                ["--max-ticks", "1"],
                compact_rollout_slab_learner_gate_num_unroll_steps=2,
                max_ticks=1,
                steps=4,
                require_terminal_compact_owned_nstep=True,
            )
        ],
    }

    with pytest.raises(SystemExit, match="active root"):
        runner._validate_manifest(manifest)


def test_hybrid_profile_runner_preflight_rejects_normal_death_without_compact_owned_shape():
    row = _valid_row(
        death_mode="normal",
        require_normal_death_terminal_contract=True,
    )
    death_mode_index = row["command"].index("--death-mode")
    row["command"][death_mode_index + 1] = "normal"
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [row],
    }

    with pytest.raises(SystemExit, match="compact-owned candidate"):
        runner._validate_manifest(manifest)


def test_hybrid_profile_runner_preflight_accepts_normal_death_nontruncating_horizon():
    row = _compact_owned_candidate_row(
        [
            "--max-ticks",
            "2000",
            "--hybrid-compact-owned-loop-entrypoint",
            "--hybrid-compact-owned-loop-policy-version-ref",
            "unit-normal-death-policy",
            "--hybrid-compact-owned-loop-policy-source",
            "unit_normal_death_profile",
            "--hybrid-compact-owned-loop-model-version-ref",
            "unit-normal-death-model",
            "--hybrid-compact-owned-loop-capture-replay-store-state",
        ],
        compact_rollout_slab_learner_gate_num_unroll_steps=2,
        compact_owned_loop_entrypoint=True,
        compact_owned_loop_policy_version_ref="unit-normal-death-policy",
        compact_owned_loop_model_version_ref="unit-normal-death-model",
        compact_owned_loop_policy_source="unit_normal_death_profile",
        compact_owned_loop_capture_replay_store_state=True,
        death_mode="normal",
        max_ticks=2000,
        steps=64,
        require_terminal_compact_owned_nstep=True,
        require_normal_death_terminal_contract=True,
    )
    death_mode_index = row["command"].index("--death-mode")
    row["command"][death_mode_index + 1] = "normal"
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [row],
    }

    runner._validate_manifest(manifest)


def test_hybrid_profile_runner_preflight_rejects_normal_death_sample_all_gate():
    row = _compact_owned_candidate_row(
        [
            "--max-ticks",
            "2000",
            "--hybrid-compact-owned-loop-entrypoint",
            "--hybrid-compact-owned-loop-policy-version-ref",
            "unit-normal-death-policy",
            "--hybrid-compact-owned-loop-policy-source",
            "unit_normal_death_profile",
            "--hybrid-compact-owned-loop-model-version-ref",
            "unit-normal-death-model",
            "--hybrid-compact-owned-loop-capture-replay-store-state",
        ],
        compact_rollout_slab_learner_gate_num_unroll_steps=2,
        compact_rollout_slab_sample_gate_batch_size=0,
        compact_owned_loop_entrypoint=True,
        compact_owned_loop_policy_version_ref="unit-normal-death-policy",
        compact_owned_loop_model_version_ref="unit-normal-death-model",
        compact_owned_loop_policy_source="unit_normal_death_profile",
        compact_owned_loop_capture_replay_store_state=True,
        death_mode="normal",
        max_ticks=2000,
        steps=64,
        require_terminal_compact_owned_nstep=True,
        require_normal_death_terminal_contract=True,
    )
    death_mode_index = row["command"].index("--death-mode")
    row["command"][death_mode_index + 1] = "normal"
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [row],
    }

    with pytest.raises(SystemExit, match="bounded"):
        runner._validate_manifest(manifest)


def test_hybrid_profile_runner_preflight_rejects_normal_death_truncating_horizon():
    row = _compact_owned_candidate_row(
        [
            "--max-ticks",
            "3",
            "--hybrid-compact-owned-loop-entrypoint",
            "--hybrid-compact-owned-loop-policy-version-ref",
            "unit-normal-death-policy",
            "--hybrid-compact-owned-loop-policy-source",
            "unit_normal_death_profile",
        ],
        compact_rollout_slab_learner_gate_num_unroll_steps=2,
        compact_owned_loop_entrypoint=True,
        compact_owned_loop_policy_version_ref="unit-normal-death-policy",
        compact_owned_loop_policy_source="unit_normal_death_profile",
        death_mode="normal",
        max_ticks=3,
        steps=64,
        require_terminal_compact_owned_nstep=True,
        require_normal_death_terminal_contract=True,
    )
    death_mode_index = row["command"].index("--death-mode")
    row["command"][death_mode_index + 1] = "normal"
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [row],
    }

    with pytest.raises(SystemExit, match="max_ticks > steps"):
        runner._validate_manifest(manifest)


def test_hybrid_profile_runner_accepts_compact_owned_candidate_invariants(
    tmp_path,
    monkeypatch,
):
    class Completed:
        returncode = 0
        stdout = json.dumps(_compact_owned_candidate_payload())

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(_compact_owned_candidate_row(), tmp_path)

    assert record["status"] == "complete"
    summary = record["summary"]
    assert summary["obs_h2d_bytes"] == 0.0
    assert summary["resident_observation_used"] is True
    assert summary["death_mode"] == "profile_no_death"
    assert summary["committed_replay_payload_d2h_bytes"] == 0.0
    assert summary["compact_rollout_slab_learner_gate_real_muzero_update"] is True


def test_hybrid_profile_runner_accepts_explicit_compact_owned_loop_invariants(
    tmp_path,
    monkeypatch,
):
    row = _compact_owned_candidate_row(
        [
            "--hybrid-compact-owned-loop-entrypoint",
            "--hybrid-compact-owned-loop-policy-version-ref",
            "unit-owned-policy-v1",
            "--hybrid-compact-owned-loop-policy-source",
            "unit_test_runner_owned_loop",
            "--hybrid-compact-owned-loop-model-version-ref",
            "unit-owned-model-v1",
            "--hybrid-compact-owned-loop-capture-replay-store-state",
        ],
        compact_owned_loop_entrypoint=True,
        compact_owned_loop_policy_version_ref="unit-owned-policy-v1",
        compact_owned_loop_model_version_ref="unit-owned-model-v1",
        compact_owned_loop_policy_source="unit_test_runner_owned_loop",
        compact_owned_loop_capture_replay_store_state=True,
    )
    payload = _compact_owned_candidate_payload(
        compact_owned_loop_entrypoint_enabled=True,
        compact_owned_loop_schema_id=runner.COMPACT_OWNED_LOOP_SCHEMA_ID,
        compact_owned_loop_profile_only=True,
        compact_owned_loop_calls_train_muzero=False,
        compact_owned_loop_touches_live_runs=False,
        compact_owned_loop_replay_store_owned=True,
        compact_owned_loop_policy_version_handoff=True,
        compact_owned_loop_policy_version_ref="unit-owned-policy-v1",
        compact_owned_loop_model_version_ref="unit-owned-model-v1",
        compact_owned_loop_policy_source="unit_test_runner_owned_loop",
        compact_owned_loop_telemetry={
            "compact_owned_loop_policy_version_ref": "unit-owned-policy-v1",
            "compact_owned_loop_sample_gate_last_sample_metadata": {
                "compact_owned_loop_replay_store_owned": True,
                "compact_owned_loop_policy_version_ref": "unit-owned-policy-v1",
            },
        },
        compact_owned_loop_replay_store_state_metadata={
            "schema_id": runner.COMPACT_REPLAY_STORE_STATE_SCHEMA_ID,
            "compact_owned_loop_replay_store_owned": True,
        },
    )

    class Completed:
        returncode = 0
        stdout = json.dumps(payload)

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(row, tmp_path)

    assert record["status"] == "complete"
    summary = record["summary"]
    assert summary["compact_owned_loop_entrypoint_enabled"] is True
    assert summary["compact_owned_loop_policy_version_ref"] == "unit-owned-policy-v1"
    assert summary["compact_owned_loop_replay_store_owned"] is True


def test_hybrid_profile_runner_accepts_matched_denominator_compact_row(
    tmp_path,
    monkeypatch,
):
    payload = _matched_denominator_compact_payload()
    _install_detached_hybrid_result(monkeypatch, payload)

    record = runner._run_row(_matched_denominator_compact_row(), tmp_path)

    assert record["status"] == "complete"
    summary = record["summary"]
    assert summary["matched_denominator_id"] == "unit-denominator"
    assert summary["matched_pair_role"] == "compact_candidate"
    assert summary["speed_currency"] == runner.MATCHED_COMPACT_SPEED_CURRENCY
    assert summary["promotion_claim"] is False
    assert summary["batch_size"] == 1024
    assert summary["actor_count"] == 16
    assert summary["physical_rows_per_sec"] == 10000.0


def test_hybrid_profile_runner_rejects_matched_denominator_currency_preflight():
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [
            _matched_denominator_compact_row(
                speed_currency="stock_train_muzero_profile_env_steps_per_sec",
                fixed_denominator={
                    "matched_denominator_id": "unit-denominator",
                    "matched_pair_role": "compact_candidate",
                    "row_purpose": runner.MATCHED_DENOMINATOR_ROW_PURPOSE,
                    "speed_currency": "stock_train_muzero_profile_env_steps_per_sec",
                },
            )
        ],
    }

    with pytest.raises(SystemExit, match="speed_currency"):
        runner._validate_manifest(manifest)


def test_hybrid_profile_runner_fails_matched_denominator_terminal_result(
    tmp_path,
    monkeypatch,
):
    payload = _matched_denominator_compact_payload(terminal_row_count=4)
    _install_detached_hybrid_result(monkeypatch, payload)

    record = runner._run_row(_matched_denominator_compact_row(), tmp_path)

    assert record["status"] == "matched_denominator_invariant_failed"
    assert "non-terminal" in record["problem"]
    assert "summary" not in record


def test_hybrid_profile_runner_rejects_explicit_compact_owned_loop_without_lineage():
    manifest = {
        "schema_id": runner.MANIFEST_SCHEMA_ID,
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "rows": [
            _compact_owned_candidate_row(
                ["--hybrid-compact-owned-loop-entrypoint"],
                compact_owned_loop_entrypoint=True,
                compact_owned_loop_policy_version_ref="",
                compact_owned_loop_policy_source="",
            )
        ],
    }

    with pytest.raises(SystemExit, match="policy version"):
        runner._validate_manifest(manifest)


def test_hybrid_profile_runner_accepts_compact_owned_candidate_unroll_invariants(
    tmp_path,
    monkeypatch,
):
    payload = _compact_owned_candidate_payload(
        compact_rollout_slab_learner_gate_num_unroll_steps=2,
        compact_rollout_slab_sample_gate_last_telemetry={
            "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
            "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
        },
        compact_rollout_slab_learner_gate_last_telemetry={
            "compact_rollout_slab_learner_gate_num_unroll_steps": 2,
        },
    )

    class Completed:
        returncode = 0
        stdout = json.dumps(payload)

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(
        _compact_owned_candidate_row(compact_rollout_slab_learner_gate_num_unroll_steps=2),
        tmp_path,
    )

    assert record["status"] == "complete"
    assert record["summary"]["compact_rollout_slab_learner_gate_num_unroll_steps"] == 2


def test_hybrid_profile_runner_accepts_terminal_compact_owned_nstep_invariants(
    tmp_path,
    monkeypatch,
):
    payload = _compact_owned_candidate_payload(
        **_normal_death_payload_fields(),
        compact_rollout_slab_learner_gate_num_unroll_steps=2,
        compact_rollout_slab_sample_gate_last_telemetry={
            "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
            "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            "compact_rollout_slab_sample_gate_terminal_sample_row_count": 8,
            "compact_rollout_slab_sample_gate_next_final_observation_row_count": 8,
            "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used": True,
            "compact_rollout_slab_sample_gate_device_replay_index_rows": True,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count": 8,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode": (
                "stock_terminal_no_bootstrap_return_discount_1.0"
            ),
        },
        compact_rollout_slab_learner_gate_last_telemetry={
            "compact_rollout_slab_learner_gate_num_unroll_steps": 2,
            "compact_rollout_slab_learner_gate_compact_muzero_telemetry": {
                "compact_muzero_learner_done_count": 8,
                "compact_muzero_learner_truncated_count": 0,
                "compact_muzero_learner_value_valid_count": 16,
            },
        },
    )

    class Completed:
        returncode = 0
        stdout = json.dumps(payload)

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(
        _compact_owned_candidate_row(
            ["--max-ticks", "3"],
            compact_rollout_slab_learner_gate_num_unroll_steps=2,
            death_mode="normal",
            max_ticks=3,
            steps=6,
            require_terminal_compact_owned_nstep=True,
            require_normal_death_terminal_contract=True,
        ),
        tmp_path,
    )

    assert record["status"] == "complete"
    summary = record["summary"]
    assert summary["terminal_row_count"] == 8
    assert summary["compact_rollout_slab_sample_gate_terminal_sample_rows"] == 8
    assert summary["death_mode"] == "normal"
    assert summary["normal_collision_death_causes"] == ["opponent_trail"]
    assert summary["normal_collision_death_hit_owner_present"] is True
    assert summary["normal_collision_death_evidence_rows"][0]["death_cause"] == [
        "opponent_trail"
    ]
    assert summary["normal_death_terminal_contract_schema_id"] == (
        "curvyzero_compact_death_terminal_contract/v1"
    )
    assert summary["normal_death_terminal_contract_evidence_id"] == "001:normal_death"
    assert summary["normal_death_terminal_contract_evidence_refs"] == ["001"]
    assert (
        summary["normal_death_terminal_contract_promotion_gate_satisfied"]
        is True
    )
    assert summary["normal_death_terminal_contract_evidence"][
        "normal_collision_death_causes"
    ] == ["opponent_trail"]
    assert summary["terminal_final_observation_before_autoreset_verified"] is True
    assert summary["terminal_final_reward_map_verified"] is True
    assert summary["promotion_eligible"] is False
    assert summary["promotion_blocker"] == "profile_only_boundary_probe"
    assert summary["promotion_claim"] is not True
    assert summary["compact_rollout_slab_sample_gate_next_final_observation_rows"] == 8
    assert (
        summary[
            "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used"
        ]
        is True
    )
    assert summary["compact_rollout_slab_sample_gate_terminal_unroll_value_target_rows"] == 8


def test_hybrid_profile_runner_fails_terminal_nstep_without_normal_death_evidence(
    tmp_path,
    monkeypatch,
):
    weak_fields = _normal_death_payload_fields()
    weak_fields["truncated_row_count"] = 1
    payload = _compact_owned_candidate_payload(
        **weak_fields,
        compact_rollout_slab_learner_gate_num_unroll_steps=2,
        compact_rollout_slab_sample_gate_last_telemetry={
            "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
            "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            "compact_rollout_slab_sample_gate_terminal_sample_row_count": 8,
            "compact_rollout_slab_sample_gate_next_final_observation_row_count": 8,
            "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used": True,
            "compact_rollout_slab_sample_gate_device_replay_index_rows": True,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count": 8,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode": (
                "stock_terminal_no_bootstrap_return_discount_1.0"
            ),
        },
        compact_rollout_slab_learner_gate_last_telemetry={
            "compact_rollout_slab_learner_gate_num_unroll_steps": 2,
            "compact_rollout_slab_learner_gate_compact_muzero_telemetry": {
                "compact_muzero_learner_done_count": 8,
                "compact_muzero_learner_truncated_count": 0,
                "compact_muzero_learner_value_valid_count": 16,
            },
        },
    )

    class Completed:
        returncode = 0
        stdout = json.dumps(payload)

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(
        _compact_owned_candidate_row(
            ["--max-ticks", "3"],
            compact_rollout_slab_learner_gate_num_unroll_steps=2,
            death_mode="normal",
            max_ticks=3,
            steps=6,
            require_terminal_compact_owned_nstep=True,
            require_normal_death_terminal_contract=True,
        ),
        tmp_path,
    )

    assert record["status"] == "compact_owned_candidate_invariant_failed"
    assert "truncated counts" in record["problem"]
    assert "summary" not in record


def test_hybrid_profile_runner_fails_terminal_nstep_without_terminal_sample(
    tmp_path,
    monkeypatch,
):
    payload = _compact_owned_candidate_payload(
        terminal_row_count=8,
        compact_rollout_slab_learner_gate_num_unroll_steps=2,
        compact_rollout_slab_sample_gate_last_telemetry={
            "compact_rollout_slab_sample_gate_explicit_unroll_targets": True,
            "compact_rollout_slab_sample_gate_num_unroll_steps": 2,
            "compact_rollout_slab_sample_gate_terminal_unroll_windows_supported": True,
            "compact_rollout_slab_sample_gate_terminal_sample_row_count": 0,
            "compact_rollout_slab_sample_gate_next_final_observation_row_count": 8,
            "compact_rollout_slab_sample_gate_resident_terminal_final_observation_used": True,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_row_count": 8,
            "compact_rollout_slab_sample_gate_terminal_unroll_value_target_mode": (
                "stock_terminal_no_bootstrap_return_discount_1.0"
            ),
        },
        compact_rollout_slab_learner_gate_last_telemetry={
            "compact_rollout_slab_learner_gate_num_unroll_steps": 2,
            "compact_rollout_slab_learner_gate_compact_muzero_telemetry": {
                "compact_muzero_learner_done_count": 8,
                "compact_muzero_learner_value_valid_count": 16,
            },
        },
    )

    class Completed:
        returncode = 0
        stdout = json.dumps(payload)

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(
        _compact_owned_candidate_row(
            ["--max-ticks", "3"],
            compact_rollout_slab_learner_gate_num_unroll_steps=2,
            max_ticks=3,
            steps=6,
            require_terminal_compact_owned_nstep=True,
        ),
        tmp_path,
    )

    assert record["status"] == "compact_owned_candidate_invariant_failed"
    assert "terminal_sample_row_count" in record["problem"]


def test_hybrid_profile_runner_requires_label_triad_for_compact_owned_candidate(
    tmp_path,
    monkeypatch,
):
    payload = _compact_owned_candidate_payload()
    del payload["calls_train_muzero"]

    class Completed:
        returncode = 0
        stdout = json.dumps(payload)

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(_compact_owned_candidate_row(), tmp_path)

    assert record["status"] == "profile_label_mismatch"
    assert "calls_train_muzero" in record["problem"]


@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        ("resident_not_used", "resident_observation_used"),
        ("resident_host_fallback", "resident_observation_host_fallback_count"),
        ("resident_h2d_nonzero", "resident_observation_h2d_bytes"),
        ("obs_h2d_nonzero", "obs_h2d_bytes"),
        ("committed_replay_d2h_nonzero", "committed_replay_payload_d2h_bytes"),
        ("replay_d2h_nonzero", "replay_payload_d2h_bytes"),
        ("scalar_materialization_nonzero", "accounting_scalar_materialization_sec"),
        ("python_rows_materialized", "python_rows_materialized"),
        ("sample_gate_zero_rows", "compact_rollout_slab_sample_gate_sample_rows"),
        ("sample_gate_mock_rows", "mock_base_env_timestep_rows"),
        ("learner_toy_probe", "compact_rollout_slab_learner_gate_toy_probe"),
        ("missing_real_muzero_update", "real_muzero_update"),
        ("learner_sample_mismatch", "learner_gate_sample_rows"),
    ],
)
def test_hybrid_profile_runner_fails_closed_on_compact_owned_candidate_invariant(
    tmp_path,
    monkeypatch,
    mutation,
    needle,
):
    payload = _compact_owned_candidate_payload()
    profile = payload["compact_rollout_slab_last_telemetry"][
        "compact_rollout_slab_profile_telemetry"
    ]
    totals = payload["compact_rollout_slab_telemetry_totals"]
    if mutation == "resident_not_used":
        payload["resident_observation_used"] = False
        profile["resident_observation_used"] = False
    elif mutation == "resident_host_fallback":
        payload["resident_observation_host_fallback_count"] = 1
        profile["resident_observation_host_fallback_count"] = 1
    elif mutation == "resident_h2d_nonzero":
        payload["resident_observation_h2d_bytes"] = 128
        profile["resident_observation_h2d_bytes"] = 128
    elif mutation == "obs_h2d_nonzero":
        totals["compact_rollout_slab_obs_h2d_bytes"] = 128
    elif mutation == "committed_replay_d2h_nonzero":
        totals["compact_rollout_slab_committed_replay_payload_d2h_bytes"] = 128
    elif mutation == "replay_d2h_nonzero":
        totals["compact_rollout_slab_replay_payload_d2h_bytes"] = 128
    elif mutation == "scalar_materialization_nonzero":
        payload["timings"]["scalar_materialization_sec"] = 0.25
    elif mutation == "python_rows_materialized":
        totals["compact_rollout_slab_python_rows_materialized"] = 32
    elif mutation == "sample_gate_zero_rows":
        payload["compact_rollout_slab_sample_gate_sample_row_count"] = 0
    elif mutation == "sample_gate_mock_rows":
        payload["compact_rollout_slab_sample_gate_mock_base_env_timestep_rows"] = 16
    elif mutation == "learner_toy_probe":
        payload["compact_rollout_slab_learner_gate_toy_probe"] = True
    elif mutation == "missing_real_muzero_update":
        payload["compact_rollout_slab_learner_gate_real_muzero_update"] = False
    elif mutation == "learner_sample_mismatch":
        payload["compact_rollout_slab_learner_gate_sample_row_count"] = 32
    else:
        raise AssertionError(f"unknown mutation {mutation}")

    class Completed:
        returncode = 0
        stdout = json.dumps(payload)

    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: Completed())

    record = runner._run_row(_compact_owned_candidate_row(), tmp_path)

    assert record["status"] == "compact_owned_candidate_invariant_failed"
    assert needle in record["problem"]
    assert "summary" not in record


def test_hybrid_profile_runner_summary_reports_compact_rollout_slab_rows():
    row = {
        "row_id": "003",
        "compute": "gpu-h100",
        "lightzero_array_ceiling_probe": True,
        "lightzero_array_ceiling_mode": "service_tax_probe",
        "compact_rollout_slab_probe": True,
        "probe_simulations": 16,
    }
    payload = {
        "ok": True,
        "steps_per_sec": 250.0,
        "measured_sec": 8.0,
        "compact_rollout_slab_enabled": True,
        "compact_rollout_slab_calls": 20,
        "compact_rollout_slab_total_roots": 4000,
        "compact_rollout_slab_roots_per_call": 200.0,
        "compact_rollout_slab_committed_index_row_count": 3800,
        "compact_rollout_slab_sample_gate_enabled": True,
        "compact_rollout_slab_sample_gate_calls": 19,
        "compact_rollout_slab_sample_gate_opportunities": 20,
        "compact_rollout_slab_sample_gate_skipped_count": 1,
        "compact_rollout_slab_sample_gate_index_row_count": 3800,
        "compact_rollout_slab_sample_gate_target_row_count": 3800,
        "compact_rollout_slab_sample_gate_sample_row_count": 256,
        "compact_rollout_slab_sample_gate_batch_size": 128,
        "compact_rollout_slab_sample_gate_interval": 2,
        "compact_rollout_slab_sample_gate_sec": 0.3,
        "compact_rollout_slab_sample_gate_mock_base_env_timestep_rows": 0,
        "compact_rollout_slab_learner_gate_impl": "compact_muzero",
        "compact_rollout_slab_learner_gate_toy_probe": False,
        "compact_rollout_slab_learner_gate_real_muzero_update": True,
        "compact_rollout_slab_learner_gate_support_scale": 300,
        "compact_rollout_slab_learner_gate_num_unroll_steps": 2,
        "compact_rollout_slab_telemetry_totals": {
            "compact_rollout_slab_model_sec": 0.9,
            "compact_rollout_slab_search_sec": 1.2,
            "compact_rollout_slab_h2d_sec": 0.2,
            "compact_rollout_slab_obs_h2d_bytes": 4096.0,
        },
        "timings": {
            "compact_rollout_slab_sec": 2.0,
            "lightzero_array_ceiling_total_sec": 999.0,
        },
        "compact_rollout_slab_last_telemetry": {
            "compact_rollout_slab_search_impl": "service_tax_probe",
            "compact_rollout_slab_num_simulations": 16,
            "compact_rollout_slab_search_service_total_sec": 1.5,
            "compact_rollout_slab_model_sec": 0.4,
            "compact_rollout_slab_search_sec": 0.8,
            "compact_rollout_slab_h2d_sec": 0.1,
            "compact_rollout_slab_obs_h2d_bytes": 2048.0,
            "compact_rollout_slab_profile_telemetry": {
                "lightzero_array_ceiling_semantics": "service_tax_probe",
                "lightzero_array_ceiling_root_noise_weight": 0.0,
                "lightzero_array_ceiling_compile_status": "not_requested",
                "lightzero_array_ceiling_compile_reason": "compile_not_requested",
            },
        },
        "compact_rollout_slab_learner_gate_last_telemetry": {
            "compact_rollout_slab_learner_gate_compact_muzero_telemetry": {
                "compact_muzero_learner_cuda_before_backward_mem_get_info_free_bytes": (
                    72_000_000_000
                ),
                "compact_muzero_learner_cuda_after_backward_mem_get_info_free_bytes": (
                    70_000_000_000
                ),
                "compact_muzero_learner_cuda_after_train_memory_peak_allocated_bytes": (
                    2_000_000_000
                ),
                "compact_muzero_learner_cuda_after_train_memory_peak_reserved_bytes": (
                    3_000_000_000
                ),
            }
        },
    }

    summary = runner._compact_line(row, payload)

    assert summary["mode"] == "compact_rollout_slab:service_tax_probe"
    assert summary["compact_rollout_slab_enabled"] is True
    assert summary["compact_rollout_slab_calls"] == 20
    assert summary["compact_rollout_slab_roots_per_call"] == 200.0
    assert summary["compact_rollout_slab_committed_index_rows"] == 3800
    assert summary["compact_rollout_slab_search_impl"] == "service_tax_probe"
    assert summary["compact_rollout_slab_num_simulations"] == 16
    assert summary["compact_rollout_slab_sample_gate_enabled"] is True
    assert summary["compact_rollout_slab_sample_gate_calls"] == 19
    assert summary["compact_rollout_slab_sample_gate_opportunities"] == 20
    assert summary["compact_rollout_slab_sample_gate_skipped_count"] == 1
    assert summary["compact_rollout_slab_sample_gate_index_rows"] == 3800
    assert summary["compact_rollout_slab_sample_gate_target_rows"] == 3800
    assert summary["compact_rollout_slab_sample_gate_sample_rows"] == 256
    assert summary["compact_rollout_slab_sample_gate_batch_size"] == 128
    assert summary["compact_rollout_slab_sample_gate_interval"] == 2
    assert summary["compact_rollout_slab_sample_gate_sec"] == 0.3
    assert summary["compact_rollout_slab_sample_gate_mock_base_env_timestep_rows"] == 0
    assert summary["compact_rollout_slab_learner_gate_impl"] == "compact_muzero"
    assert summary["compact_rollout_slab_learner_gate_toy_probe"] is False
    assert summary["compact_rollout_slab_learner_gate_real_muzero_update"] is True
    assert summary["compact_rollout_slab_learner_gate_support_scale"] == 300
    assert summary["compact_rollout_slab_learner_gate_num_unroll_steps"] == 2
    assert summary["compact_muzero_cuda_before_backward_free_bytes"] == 72_000_000_000
    assert summary["compact_muzero_cuda_after_backward_free_bytes"] == 70_000_000_000
    assert (
        summary["compact_muzero_cuda_after_train_peak_allocated_bytes"]
        == 2_000_000_000
    )
    assert summary["compact_muzero_cuda_after_train_peak_reserved_bytes"] == 3_000_000_000
    assert summary["total_roots"] == 4000
    assert summary["probe_total_sec"] == 2.0
    assert summary["probe_roots_per_sec"] == pytest.approx(2000.0)
    assert summary["model_sec"] == 0.9
    assert summary["search_sec"] == 1.2
    assert summary["h2d_sec"] == 0.2
    assert summary["obs_h2d_bytes"] == 4096.0
    assert summary["compact_rollout_slab_last_search_service_total_sec"] == 1.5
    assert summary["compact_rollout_slab_last_model_sec"] == 0.4
    assert summary["compact_rollout_slab_last_search_sec"] == 0.8
    assert summary["compact_rollout_slab_last_h2d_sec"] == 0.1
    assert summary["semantics"] == "service_tax_probe"
    assert summary["root_noise_weight"] == 0.0
    assert summary["compile_status"] == "not_requested"
    assert summary["compile_reason"] == "compile_not_requested"


def test_hybrid_profile_runner_summary_reports_compact_torch_compile_runtime():
    row = {
        "row_id": "003",
        "compute": "gpu-h100",
        "lightzero_array_ceiling_probe": True,
        "lightzero_array_ceiling_mode": "compact_torch_search_service",
        "compact_rollout_slab_probe": True,
        "probe_simulations": 8,
    }
    payload = {
        "ok": True,
        "steps_per_sec": 100.0,
        "measured_sec": 10.0,
        "compact_rollout_slab_enabled": True,
        "compact_rollout_slab_calls": 10,
        "compact_rollout_slab_total_roots": 1000,
        "compact_rollout_slab_roots_per_call": 100.0,
        "timings": {"compact_rollout_slab_sec": 2.0},
        "compact_rollout_slab_last_telemetry": {
            "compact_rollout_slab_search_impl": "compact_torch_device_tree_fixed_shape_v0",
            "compact_rollout_slab_profile_telemetry": {
                "compact_torch_search_service_compile_status": "eligible",
                "compact_torch_search_service_compile_reason": "preconditions_satisfied",
                "compact_torch_search_compile_attempted": 1.0,
                "compact_torch_search_compile_used": True,
                "compact_torch_search_compile_cache_hit": False,
                "compact_torch_search_compile_runtime_status": "compiled",
                "compact_torch_search_service_resident_obs_reused": 1.0,
            },
        },
    }

    summary = runner._compact_line(row, payload)

    assert summary["compile_status"] == "eligible"
    assert summary["compile_reason"] == "preconditions_satisfied"
    assert summary["compile_attempted"] == 1.0
    assert summary["compile_used"] is True
    assert summary["compile_cache_hit"] is False
    assert summary["compile_runtime_status"] == "compiled"
    assert summary["resident_obs_reused"] == 1.0


def test_hybrid_profile_runner_summary_preserves_compact_torch_semantics():
    row = {
        "row_id": "003",
        "compute": "gpu-h100",
        "lightzero_array_ceiling_probe": True,
        "lightzero_array_ceiling_mode": "compact_torch_search_service",
        "compact_rollout_slab_probe": True,
        "probe_simulations": 8,
    }
    payload = {
        "ok": True,
        "steps_per_sec": 100.0,
        "measured_sec": 10.0,
        "compact_rollout_slab_enabled": True,
        "compact_rollout_slab_total_roots": 1000,
        "timings": {"compact_rollout_slab_sec": 2.0},
        "compact_rollout_slab_last_telemetry": {
            "compact_rollout_slab_search_impl": "compact_torch_device_tree_fixed_shape_v0",
            "compact_rollout_slab_profile_telemetry": {
                "compact_torch_search_semantics": (
                    "profile-only fixed-shape Torch compact-search helper; "
                    "not trainer-ready and not LightZero CTree"
                ),
            },
        },
    }

    summary = runner._compact_line(row, payload)

    assert summary["semantics"] == (
        "profile-only fixed-shape Torch compact-search helper; "
        "not trainer-ready and not LightZero CTree"
    )


def test_hybrid_profile_runner_summary_reports_mctx_compact_slab_semantics():
    row = {
        "row_id": "004",
        "compute": "gpu-h100",
        "compact_rollout_slab_probe": True,
        "mctx_compact_search_probe": True,
        "probe_simulations": 32,
    }
    payload = {
        "ok": True,
        "steps_per_sec": 100.0,
        "compact_rollout_slab_enabled": True,
        "compact_rollout_slab_total_roots": 1000,
        "compact_rollout_slab_telemetry_totals": {
            "compact_rollout_slab_search_sec": 1.0,
        },
        "timings": {"compact_rollout_slab_sec": 2.0},
        "compact_rollout_slab_last_telemetry": {
            "compact_rollout_slab_search_impl": "mctx_compact_search_service_profile_only_v0",
            "compact_rollout_slab_num_simulations": 32,
            "compact_rollout_slab_semantics": (
                "profile_only_jax_mctx_gumbel_muzero_search_not_lightzero_ctree"
            ),
        },
    }

    summary = runner._compact_line(row, payload)

    assert summary["mode"] == (
        "compact_rollout_slab:mctx_compact_search_service_profile_only_v0"
    )
    assert summary["semantics"] == (
        "profile_only_jax_mctx_gumbel_muzero_search_not_lightzero_ctree"
    )
    assert summary["promotion_eligible"] is False


def test_hybrid_profile_runner_summary_falls_back_to_row_root_noise_and_input_bytes():
    row = {
        "row_id": "001",
        "compute": "gpu-h100",
        "lightzero_array_ceiling_probe": True,
        "lightzero_array_ceiling_mode": "service_tax_probe",
        "lightzero_consumer_root_noise_weight": 0.0,
        "probe_simulations": 16,
    }
    payload = {
        "ok": True,
        "batched_stack_probe_total_roots": 2000,
        "timings": {"lightzero_array_ceiling_total_sec": 4.0},
        "batched_stack_probe_last_telemetry": {
            "host_to_device_bytes": 4096.0,
            "lightzero_array_ceiling_semantics": "service_tax_probe",
        },
    }

    summary = runner._compact_line(row, payload)

    assert summary["obs_h2d_bytes"] == 4096.0
    assert summary["root_noise_weight"] == 0.0


def test_hybrid_profile_runner_summary_uses_mcts_ledger_keys_for_direct_probe():
    row = {
        "row_id": "002",
        "compute": "gpu-h100",
        "lightzero_mcts_arrays_boundary_probe": True,
        "lightzero_mcts_arrays_boundary_impl": "direct_ctree_gpu_latent",
        "probe_simulations": 16,
    }
    payload = {
        "ok": True,
        "steps_per_sec": 100.0,
        "measured_sec": 20.0,
        "batched_stack_probe_total_roots": 2000,
        "timings": {
            "lightzero_mcts_arrays_boundary_total_sec": 4.0,
            "lightzero_mcts_arrays_boundary_initial_inference_sec": 1.0,
            "lightzero_mcts_arrays_boundary_recurrent_inference_sec": 2.0,
            "lightzero_mcts_arrays_boundary_search_sec": 0.5,
            "lightzero_mcts_arrays_boundary_input_prepare_sec": 0.25,
        },
        "batched_stack_probe_ledger_totals": {
            "lightzero_mcts_arrays_boundary_obs_h2d_bytes": 2048.0,
            "lightzero_mcts_arrays_boundary_mask_h2d_bytes": 0.0,
            "lightzero_mcts_arrays_boundary_action_d2h_bytes": 0.0,
            "lightzero_mcts_arrays_boundary_replay_payload_d2h_bytes": 0.0,
            "lightzero_mcts_arrays_boundary_root_observation_copy_bytes": 0.0,
            "lightzero_mcts_arrays_boundary_python_rows_materialized": 0.0,
            "lightzero_mcts_arrays_boundary_rnd_materialized_rows": 0.0,
            "lightzero_mcts_arrays_boundary_resident_reused": 2.0,
        },
        "batched_stack_probe_last_telemetry": {
            "lightzero_mcts_arrays_boundary_semantics": "direct_ctree",
            "lightzero_mcts_arrays_boundary_model_output_d2h_bytes": 512.0,
        },
    }

    summary = runner._compact_line(row, payload)

    assert summary["obs_h2d_bytes"] == 2048.0
    assert summary["mask_h2d_bytes"] == 0.0
    assert summary["resident_obs_reused"] == 2.0
    assert summary["model_output_d2h_bytes"] == 512.0
