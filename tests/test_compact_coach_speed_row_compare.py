from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_compact_coach_speed_rows.py"
SPEC = importlib.util.spec_from_file_location("compare_compact_coach_speed_rows", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)
launcher = compare.launcher


def _accepted_summary(**overrides):
    summary = {
        "steps_per_sec": 13_308.0,
        "training_wall_sec": 13.85,
        "env_steps_collected": 184320.0,
        "steps": 180,
        "warmup_steps": 45,
        "speed_row_actor_step_wall_sec": 4.6,
        "speed_row_actor_autoreset_sec": 3.1,
        "speed_row_sample_gate_sec": 2.6,
        "compact_rollout_slab_sample_gate_candidate_sec": 0.4,
        "compact_rollout_slab_sample_gate_rng_sec": 0.09,
        "compact_rollout_slab_sample_gate_learner_batch_build_sec": 1.4,
        "compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics": False,
        "compact_rollout_slab_sample_gate_cuda_sync_timing_enabled": False,
        "compact_rollout_slab_sample_gate_cuda_sync_count": 0,
        "compact_rollout_slab_sample_gate_cuda_sync_sec": 0.0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics": (
            False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled": (
            False
        ),
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec": 0.0,
        "speed_row_learner_gate_sec": 1.16,
        "compact_rollout_slab_learner_gate_backward_sec": 0.6,
        "compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics": False,
        "compact_rollout_slab_learner_gate_cuda_sync_timing_enabled": False,
        "compact_rollout_slab_learner_gate_cuda_sync_count": 0,
        "compact_rollout_slab_learner_gate_cuda_sync_sec": 0.0,
        "compact_profile_cuda_sync_timing_diagnostics": False,
        "compact_profile_runtime_step_timing_diagnostics": False,
        "compact_profile_cpu_perf_stat_diagnostics": False,
        "compact_profile_runtime_step_count": 0,
        "compact_profile_runtime_step_sum_sec": 0.0,
        "compact_profile_runtime_step_min_sec": 0.0,
        "compact_profile_runtime_step_max_sec": 0.0,
        "compact_profile_runtime_step_p50_sec": 0.0,
        "compact_profile_runtime_step_p95_sec": 0.0,
        "compact_profile_runtime_step_slowest_actor_step_wall_sec": 0.0,
        "compact_profile_runtime_step_slowest_observation_sec": 0.0,
        "compact_profile_runtime_step_slowest_compact_rollout_slab_sec": 0.0,
        "compact_profile_runtime_step_slowest_sample_gate_sec": 0.0,
        "compact_profile_runtime_step_slowest_learner_gate_sec": 0.0,
        "compact_profile_runtime_step_slowest_policy_refresh_sec": 0.0,
        "compact_profile_runtime_step_slowest_primary_accounted_sec": 0.0,
        "compact_profile_runtime_step_slowest_primary_residual_sec": 0.0,
        "render_state_borrowed_steps": 225,
        "terminal_sample_row_count": 167,
        "terminal_unroll_value_target_row_count": 167,
    }
    for _label, field, expected in launcher._ACCEPTED_FAST_PATH_RESULT_REQUIREMENTS:
        summary[field] = expected
    for field in launcher._ACCEPTED_FAST_PATH_REPEATABILITY_REQUIRED_FIELDS:
        summary.setdefault(field, 1)
    for field in launcher._ACCEPTED_FAST_PATH_REPEATABILITY_NONZERO_FIELDS:
        summary[field] = -123 if field == "env_trajectory_ordered_checksum_total" else 123
    for field in launcher._ACCEPTED_FAST_PATH_REPEATABILITY_POSITIVE_FIELDS:
        summary[field] = 7
    summary["compact_rollout_slab_policy_refresh_after_learner_gate_last_model_state_digest"] = "abc"
    summary.update(overrides)
    return summary


def _accepted_unroll2_summary(**overrides):
    proof = {
        "compact_muzero_learner_batch_unroll2_specialized_builder": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_requested": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_eligible_count": 7,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_used": True,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_call_count": 7,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_count": 0,
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason": "none",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl": "unroll2_specialized_v1",
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_path": "unroll2_specialized",
    }
    proof.update(overrides)
    return _accepted_summary(**proof)


def _write_row(path: Path, summary: dict) -> Path:
    path.write_text(json.dumps({"summary": summary}))
    return path


def test_compare_speed_rows_accepts_signed_checksums_and_reports_timing_deltas(tmp_path):
    baseline = _write_row(tmp_path / "i.json", _accepted_summary())
    candidate = _write_row(
        tmp_path / "j.json",
        _accepted_summary(
            steps_per_sec=10_058.0,
            training_wall_sec=18.33,
            speed_row_actor_step_wall_sec=6.0,
            speed_row_actor_autoreset_sec=4.2,
            speed_row_sample_gate_sec=4.45,
            compact_rollout_slab_sample_gate_rng_sec=0.5,
            compact_rollout_slab_sample_gate_learner_batch_build_sec=2.4,
            speed_row_learner_gate_sec=1.82,
            compact_rollout_slab_learner_gate_backward_sec=1.04,
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("I", baseline),
            compare.RowInput("J", candidate),
        ],
        baseline_label="I",
    )

    assert report["rows"][0]["accepted_fast_path_violations"] == []
    assert report["rows"][1]["accepted_fast_path_violations"] == []
    assert report["stable_speed_claim_allowed"] is False
    assert report["exact_repeat_stability"]["exact_row_count"] == 2
    assert report["exact_repeat_stability"]["training_wall_sec"]["range"] == pytest.approx(4.48)
    assert report["largest_exact_wall_swing"]["candidate_label"] == "J"
    comparison = report["comparisons"][0]
    assert comparison["identity_status"] == "exact"
    assert comparison["identity_matches"] is True
    deltas = {item["field"]: item["delta"] for item in comparison["largest_timing_deltas"]}
    assert deltas["training_wall_sec"] == pytest.approx(4.48)
    assert deltas["speed_row_sample_gate_sec"] == pytest.approx(1.85)
    assert deltas["speed_row_actor_autoreset_sec"] == pytest.approx(1.1)
    assert deltas["compact_rollout_slab_learner_gate_backward_sec"] == pytest.approx(0.44)


def test_compare_speed_rows_derives_whole_owner_buffer_replay_ceiling(tmp_path):
    baseline = _write_row(
        tmp_path / "baseline.json",
        _accepted_summary(
            env_steps_collected=741_376.0,
            steps_per_sec=12_689.38,
            training_wall_sec=58.42491910398818,
        ),
    )
    candidate = _write_row(
        tmp_path / "columnar.json",
        _accepted_summary(
            env_steps_collected=741_376.0,
            steps_per_sec=15_852.6672239789,
            training_wall_sec=46.766641192,
            compact_owner_search_worker_replay_append_sec=14.219468255000084,
            speed_row_total_owner_search_worker_replay_append_sec=14.219468255000084,
            compact_owner_search_owner_train_sample_sec=2.7769241389999664,
            compact_owner_search_owner_train_wall_sec=9.756279996999996,
            compact_owner_search_owner_train_learner_update_sec=5.254527383999999,
            compact_owner_search_worker_search_sec=0.012870255999999358,
            speed_row_total_owner_search_worker_search_sec=13.294650892000028,
            compact_owner_search_parent_wait_sec=0.018655968000004464,
            speed_row_total_owner_search_parent_wait_sec=17.654943611000064,
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("baseline", baseline),
            compare.RowInput("columnar-r2", candidate),
        ],
        baseline_label="baseline",
    )

    ceiling = report["rows"][1]["whole_owner_buffer_replay_ceiling"]
    direct_surface = 14.219468255000084 + 2.7769241389999664
    projected_wall = 46.766641192 - direct_surface
    target_wall = 741_376.0 / (compare.ACCEPTED_BASELINE_STEPS_PER_SEC * 2.0)
    assert ceiling["enabled"] is True
    assert ceiling["projection_only"] is True
    assert ceiling["production_speed_claim"] is False
    assert ceiling["speed_currency"] == "local_projection_no_speed"
    assert ceiling["observed_worker_search_sec"] == pytest.approx(13.294650892000028)
    assert ceiling["observed_parent_wait_sec"] == pytest.approx(17.654943611000064)
    assert ceiling["direct_replay_sample_surface_sec"] == pytest.approx(direct_surface)
    assert ceiling["parent_wait_bounded_surface_sec"] == pytest.approx(direct_surface)
    assert ceiling["preserved_search_update_floor_sec"] == pytest.approx(
        13.294650892000028 + 5.254527383999999
    )
    assert ceiling["projected_wall_sec"] == pytest.approx(projected_wall)
    assert ceiling["projected_env_steps_per_sec"] == pytest.approx(
        741_376.0 / projected_wall
    )
    assert ceiling["projected_speedup_vs_accepted_baseline"] == pytest.approx(
        (741_376.0 / projected_wall) / compare.ACCEPTED_BASELINE_STEPS_PER_SEC
    )
    assert ceiling["projected_reaches_2x"] is False
    assert ceiling["additional_removed_sec_to_2x"] == pytest.approx(
        projected_wall - target_wall
    )
    assert report["whole_owner_buffer_replay_ceiling_rank"][0]["label"] == "columnar-r2"


def test_compare_speed_rows_reports_identity_mismatch(tmp_path):
    baseline = _write_row(tmp_path / "i.json", _accepted_summary())
    drifted = _write_row(tmp_path / "drifted.json", _accepted_summary(sample_batch_size=256))

    report = compare.compare_rows(
        [
            compare.RowInput("I", baseline),
            compare.RowInput("drifted", drifted),
        ],
        baseline_label="I",
    )

    comparison = report["comparisons"][0]
    assert comparison["identity_status"] == "mismatch"
    assert comparison["identity_matches"] is False
    assert any(item["field"] == "sample_batch_size" for item in comparison["identity_mismatches"])
    assert report["stable_speed_claim_allowed"] is False


def test_compare_speed_rows_treats_present_terminal_final_observation_proof_as_identity(
    tmp_path,
):
    proof_key = (
        "compact_rollout_slab_sample_gate_"
        "terminal_final_observation_validate_only_count"
    )
    legacy_baseline = _write_row(tmp_path / "legacy_a.json", _accepted_summary())
    legacy_candidate = _write_row(tmp_path / "legacy_b.json", _accepted_summary())

    legacy_report = compare.compare_rows(
        [
            compare.RowInput("legacy-a", legacy_baseline),
            compare.RowInput("legacy-b", legacy_candidate),
        ],
        baseline_label="legacy-a",
    )

    assert legacy_report["comparisons"][0]["identity_status"] == "exact"

    baseline = _write_row(
        tmp_path / "proof_a.json",
        _accepted_summary(
            **{
                proof_key: 399,
                "compact_rollout_slab_sample_gate_terminal_final_observation_materialized_count": 0,
            }
        ),
    )
    drifted = _write_row(
        tmp_path / "proof_b.json",
        _accepted_summary(
            **{
                proof_key: 398,
                "compact_rollout_slab_sample_gate_terminal_final_observation_materialized_count": 0,
            }
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("proof-a", baseline),
            compare.RowInput("proof-b", drifted),
        ],
        baseline_label="proof-a",
    )

    comparison = report["comparisons"][0]
    assert comparison["identity_status"] == "mismatch"
    assert any(item["field"] == proof_key for item in comparison["identity_mismatches"])


def test_compare_speed_rows_requires_exact_candidate_for_stable_claim(tmp_path):
    baseline = _write_row(
        tmp_path / "fast_a.json",
        _accepted_summary(steps_per_sec=13_500.0, training_wall_sec=13.6),
    )
    drifted = _write_row(
        tmp_path / "fast_drifted.json",
        _accepted_summary(
            steps_per_sec=13_400.0,
            training_wall_sec=13.7,
            sample_batch_size=256,
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("fast-a", baseline),
            compare.RowInput("fast-drifted", drifted),
        ],
        baseline_label="fast-a",
    )

    assert report["comparisons"][0]["identity_status"] == "mismatch"
    assert report["stable_speed_claim_allowed"] is False


def test_compare_speed_rows_requires_no_violations_for_stable_claim(tmp_path):
    baseline = _write_row(
        tmp_path / "fast_a.json",
        _accepted_summary(steps_per_sec=13_500.0, training_wall_sec=13.6),
    )
    candidate = _write_row(
        tmp_path / "fast_b.json",
        _accepted_summary(
            steps_per_sec=13_400.0,
            training_wall_sec=13.7,
            hybrid_persistent_compact_render_state_buffer=True,
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("fast-a", baseline),
            compare.RowInput("fast-b", candidate),
        ],
        baseline_label="fast-a",
    )

    assert report["comparisons"][0]["identity_status"] == "exact"
    assert report["rows"][1]["accepted_fast_path_violations"]
    assert report["stable_speed_claim_allowed"] is False


def test_compare_speed_rows_allows_stable_claim_for_clean_exact_repeats(tmp_path):
    baseline = _write_row(
        tmp_path / "fast_a.json",
        _accepted_summary(steps_per_sec=13_500.0, training_wall_sec=13.6),
    )
    candidate = _write_row(
        tmp_path / "fast_b.json",
        _accepted_summary(steps_per_sec=13_400.0, training_wall_sec=13.7),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("fast-a", baseline),
            compare.RowInput("fast-b", candidate),
        ],
        baseline_label="fast-a",
    )

    assert report["comparisons"][0]["identity_status"] == "exact"
    assert report["rows"][0]["accepted_fast_path_violations"] == []
    assert report["rows"][1]["accepted_fast_path_violations"] == []
    assert report["stable_speed_claim_allowed"] is True


def test_compare_speed_rows_requires_clean_unroll2_specialized_builder_proof(
    tmp_path,
):
    baseline = _write_row(
        tmp_path / "unroll2_a.json",
        _accepted_unroll2_summary(steps_per_sec=13_500.0, training_wall_sec=13.6),
    )
    candidate = _write_row(
        tmp_path / "unroll2_b.json",
        _accepted_unroll2_summary(steps_per_sec=13_400.0, training_wall_sec=13.7),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("unroll2-a", baseline),
            compare.RowInput("unroll2-b", candidate),
        ],
        baseline_label="unroll2-a",
    )

    assert report["rows"][0]["unroll2_specialized_builder_violations"] == []
    assert report["rows"][1]["unroll2_specialized_builder_violations"] == []
    assert report["stable_speed_claim_allowed"] is True

    stale_reason = _write_row(
        tmp_path / "unroll2_stale_reason.json",
        _accepted_unroll2_summary(
            steps_per_sec=13_400.0,
            training_wall_sec=13.7,
            compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_fallback_reason="guard_failed",
        ),
    )
    stale_report = compare.compare_rows(
        [
            compare.RowInput("unroll2-a", baseline),
            compare.RowInput("stale-reason", stale_reason),
        ],
        baseline_label="unroll2-a",
    )

    assert any(
        "fallback_reason" in item
        for item in stale_report["rows"][1]["unroll2_specialized_builder_violations"]
    )
    assert stale_report["stable_speed_claim_allowed"] is False

    wrong_impl = _write_row(
        tmp_path / "unroll2_wrong_impl.json",
        _accepted_unroll2_summary(
            steps_per_sec=13_400.0,
            training_wall_sec=13.7,
            compact_rollout_slab_sample_gate_learner_batch_builder_unroll2_specialized_builder_impl="generic",
        ),
    )
    impl_report = compare.compare_rows(
        [
            compare.RowInput("unroll2-a", baseline),
            compare.RowInput("wrong-impl", wrong_impl),
        ],
        baseline_label="unroll2-a",
    )

    assert any(
        "impl" in item
        for item in impl_report["rows"][1]["unroll2_specialized_builder_violations"]
    )
    assert impl_report["stable_speed_claim_allowed"] is False


def test_compare_speed_rows_reports_exact_repeat_stability_ranges(tmp_path):
    row_a = _write_row(
        tmp_path / "a.json",
        _accepted_summary(steps_per_sec=13_500.0, training_wall_sec=13.5),
    )
    row_b = _write_row(
        tmp_path / "b.json",
        _accepted_summary(steps_per_sec=13_300.0, training_wall_sec=13.7),
    )
    row_c = _write_row(
        tmp_path / "c.json",
        _accepted_summary(steps_per_sec=13_100.0, training_wall_sec=14.0),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("A", row_a),
            compare.RowInput("B", row_b),
            compare.RowInput("C", row_c),
        ],
        baseline_label="A",
    )

    stability = report["exact_repeat_stability"]
    assert stability["exact_row_count"] == 3
    assert stability["exact_candidate_count"] == 2
    assert stability["labels"] == ["A", "B", "C"]
    assert stability["training_wall_sec"]["min_label"] == "A"
    assert stability["training_wall_sec"]["max_label"] == "C"
    assert stability["training_wall_sec"]["range"] == pytest.approx(0.5)
    assert stability["training_wall_sec"]["median"] == pytest.approx(13.7)
    assert stability["training_wall_sec"]["spread_pct_of_median"] == pytest.approx(
        (0.5 / 13.7) * 100.0
    )


def test_compare_speed_rows_reports_wall_swing_attribution(tmp_path):
    row_fast = _write_row(
        tmp_path / "fast.json",
        _accepted_summary(
            steps_per_sec=14_000.0,
            training_wall_sec=10.0,
            compact_profile_runtime_step_sum_sec=10.0,
            compact_profile_runtime_step_primary_accounted_sum_sec=9.0,
            speed_row_sample_gate_sec=4.0,
            compact_rollout_slab_sample_gate_learner_batch_build_sec=2.5,
            speed_row_actor_step_wall_sec=2.0,
            speed_row_observation_sec=1.0,
        ),
    )
    row_mid = _write_row(
        tmp_path / "mid.json",
        _accepted_summary(
            steps_per_sec=12_000.0,
            training_wall_sec=13.0,
            compact_profile_runtime_step_sum_sec=13.0,
            compact_profile_runtime_step_primary_accounted_sum_sec=12.0,
            speed_row_sample_gate_sec=5.0,
            compact_rollout_slab_sample_gate_learner_batch_build_sec=3.0,
            speed_row_actor_step_wall_sec=2.4,
            speed_row_observation_sec=8.0,
        ),
    )
    row_slow = _write_row(
        tmp_path / "slow.json",
        _accepted_summary(
            steps_per_sec=10_000.0,
            training_wall_sec=16.0,
            compact_profile_runtime_step_sum_sec=16.0,
            compact_profile_runtime_step_primary_accounted_sum_sec=14.0,
            speed_row_sample_gate_sec=7.0,
            compact_rollout_slab_sample_gate_learner_batch_build_sec=4.0,
            speed_row_actor_step_wall_sec=3.0,
            speed_row_observation_sec=1.0,
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("fast", row_fast),
            compare.RowInput("mid", row_mid),
            compare.RowInput("slow", row_slow),
        ],
        baseline_label="fast",
    )

    attribution = report["exact_repeat_stability"]["wall_swing_attribution"]
    assert attribution["fastest_label"] == "fast"
    assert attribution["slowest_label"] == "slow"
    assert attribution["wall_delta_sec"] == pytest.approx(6.0)
    positive_deltas = {
        item["field"]: item for item in attribution["largest_positive_deltas"]
    }
    assert positive_deltas["compact_profile_runtime_step_sum_sec"]["delta"] == (
        pytest.approx(6.0)
    )
    assert positive_deltas["compact_profile_runtime_step_sum_sec"][
        "pct_of_wall_delta"
    ] == pytest.approx(100.0)
    assert positive_deltas["speed_row_sample_gate_sec"]["delta"] == pytest.approx(3.0)
    assert positive_deltas["speed_row_sample_gate_sec"]["pct_of_wall_delta"] == (
        pytest.approx(50.0)
    )
    assert "speed_row_observation_sec" not in positive_deltas
    absolute_deltas = {
        item["field"]: item for item in attribution["largest_absolute_deltas"]
    }
    assert absolute_deltas["speed_row_observation_sec"]["delta"] == pytest.approx(0.0)


def test_compare_speed_rows_labels_long_window_as_diagnostic(tmp_path):
    baseline = _write_row(
        tmp_path / "long_a.json",
        _accepted_summary(
            steps=724,
            warmup_steps=180,
            env_steps_collected=741376.0,
            render_state_borrowed_steps=904,
            terminal_sample_row_count=660,
            terminal_unroll_value_target_row_count=660,
            steps_per_sec=14_000.0,
            training_wall_sec=52.0,
        ),
    )
    candidate = _write_row(
        tmp_path / "long_b.json",
        _accepted_summary(
            steps=724,
            warmup_steps=180,
            env_steps_collected=741376.0,
            render_state_borrowed_steps=904,
            terminal_sample_row_count=660,
            terminal_unroll_value_target_row_count=660,
            steps_per_sec=13_900.0,
            training_wall_sec=52.4,
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("long-a", baseline),
            compare.RowInput("long-b", candidate),
        ],
        baseline_label="long-a",
    )

    assert report["rows"][0]["accepted_fast_path_violations"] == []
    assert report["rows"][0]["compact_owned_accepted_fast_path_step_window"] == (
        "stability_724_180"
    )
    assert report["rows"][0]["compact_owned_accepted_fast_path_stability_diagnostic"] is True
    assert report["rows"][0]["speed_row_comparison_role"] == (
        "long_window_stability_diagnostic"
    )
    assert report["comparisons"][0]["identity_status"] == "exact"
    assert report["stable_speed_claim_allowed"] is False


def test_compare_speed_rows_includes_cuda_sync_diagnostic_fields(tmp_path):
    baseline = _write_row(
        tmp_path / "sync_a.json",
        _accepted_summary(
            compact_profile_cuda_sync_timing_diagnostics=True,
            compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics=True,
            compact_rollout_slab_sample_gate_cuda_sync_timing_enabled=True,
            compact_rollout_slab_sample_gate_cuda_sync_count=12,
            compact_rollout_slab_sample_gate_cuda_sync_sec=2.03,
            compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics=True,
            compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled=True,
            compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count=11977,
            compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec=6.9,
            compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics=True,
            compact_rollout_slab_learner_gate_cuda_sync_timing_enabled=True,
            compact_rollout_slab_learner_gate_cuda_sync_count=28,
            compact_rollout_slab_learner_gate_cuda_sync_sec=2.84,
            compact_profile_runtime_step_timing_diagnostics=True,
            compact_profile_runtime_step_count=1084,
            compact_profile_runtime_step_sum_sec=151.0,
            compact_profile_runtime_step_min_sec=0.12,
            compact_profile_runtime_step_max_sec=0.31,
            compact_profile_runtime_step_p50_sec=0.14,
            compact_profile_runtime_step_p95_sec=0.2,
            compact_profile_runtime_step_slowest_actor_step_wall_sec=0.09,
            compact_profile_runtime_step_slowest_observation_sec=0.07,
            compact_profile_runtime_step_slowest_compact_rollout_slab_sec=0.03,
            compact_profile_runtime_step_slowest_sample_gate_sec=0.08,
            compact_profile_runtime_step_slowest_learner_gate_sec=0.02,
            compact_profile_runtime_step_slowest_policy_refresh_sec=0.01,
            compact_profile_runtime_step_slowest_primary_accounted_sec=0.3,
            compact_profile_runtime_step_slowest_primary_residual_sec=0.01,
            compact_profile_runtime_step_actor_step_wall_sum_sec=31.0,
            compact_profile_runtime_step_actor_step_wall_p95_sec=0.09,
            compact_profile_runtime_step_actor_env_runtime_sum_sec=22.0,
            compact_profile_runtime_step_actor_autoreset_sum_sec=14.0,
            compact_profile_runtime_step_sample_gate_sum_sec=81.0,
            compact_profile_runtime_step_sample_gate_p95_sec=1.15,
            compact_profile_runtime_step_sample_gate_residual_sum_sec=24.0,
            compact_profile_runtime_step_sample_gate_cuda_sync_sum_sec=2.0,
            compact_profile_runtime_step_sample_gate_builder_group_loop_sum_sec=64.0,
            compact_profile_runtime_step_sample_gate_builder_cuda_sync_sum_sec=6.9,
            compact_profile_runtime_step_primary_residual_sum_sec=4.0,
            compact_profile_runtime_step_primary_residual_p95_sec=0.04,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec=50.0,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec=14.0,
            compact_rollout_slab_sample_gate_learner_batch_build_per_call_p95_sec=0.8,
            compact_rollout_slab_sample_gate_per_call_p95_sec=1.15,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_p95_sec=0.41,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_p95_sec=0.34,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_p95_sec=0.07,
            compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_per_call_p95_sec=0.21,
            compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call_p95_sec=0.05,
            compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call_p95_sec=0.11,
        ),
    )
    candidate = _write_row(
        tmp_path / "sync_b.json",
        _accepted_summary(
            compact_profile_cuda_sync_timing_diagnostics=True,
            compact_rollout_slab_sample_gate_cuda_sync_timing_diagnostics=True,
            compact_rollout_slab_sample_gate_cuda_sync_timing_enabled=True,
            compact_rollout_slab_sample_gate_cuda_sync_count=12,
            compact_rollout_slab_sample_gate_cuda_sync_sec=0.02,
            compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_diagnostics=True,
            compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_timing_enabled=True,
            compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count=11977,
            compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec=5.5,
            compact_rollout_slab_learner_gate_cuda_sync_timing_diagnostics=True,
            compact_rollout_slab_learner_gate_cuda_sync_timing_enabled=True,
            compact_rollout_slab_learner_gate_cuda_sync_count=28,
            compact_rollout_slab_learner_gate_cuda_sync_sec=0.83,
            compact_profile_runtime_step_timing_diagnostics=True,
            compact_profile_runtime_step_count=1084,
            compact_profile_runtime_step_sum_sec=191.0,
            compact_profile_runtime_step_min_sec=0.12,
            compact_profile_runtime_step_max_sec=0.31,
            compact_profile_runtime_step_p50_sec=0.14,
            compact_profile_runtime_step_p95_sec=0.2,
            compact_profile_runtime_step_slowest_actor_step_wall_sec=0.09,
            compact_profile_runtime_step_slowest_observation_sec=0.07,
            compact_profile_runtime_step_slowest_compact_rollout_slab_sec=0.03,
            compact_profile_runtime_step_slowest_sample_gate_sec=0.08,
            compact_profile_runtime_step_slowest_learner_gate_sec=0.02,
            compact_profile_runtime_step_slowest_policy_refresh_sec=0.01,
            compact_profile_runtime_step_slowest_primary_accounted_sec=0.3,
            compact_profile_runtime_step_slowest_primary_residual_sec=0.01,
            compact_profile_runtime_step_actor_step_wall_sum_sec=41.0,
            compact_profile_runtime_step_actor_step_wall_p95_sec=0.19,
            compact_profile_runtime_step_actor_env_runtime_sum_sec=22.0,
            compact_profile_runtime_step_actor_autoreset_sum_sec=14.0,
            compact_profile_runtime_step_sample_gate_sum_sec=111.0,
            compact_profile_runtime_step_sample_gate_p95_sec=1.45,
            compact_profile_runtime_step_sample_gate_residual_sum_sec=24.0,
            compact_profile_runtime_step_sample_gate_cuda_sync_sum_sec=2.0,
            compact_profile_runtime_step_sample_gate_builder_group_loop_sum_sec=64.0,
            compact_profile_runtime_step_sample_gate_builder_cuda_sync_sum_sec=6.9,
            compact_profile_runtime_step_primary_residual_sum_sec=9.0,
            compact_profile_runtime_step_primary_residual_p95_sec=0.11,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec=50.0,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec=14.0,
            compact_rollout_slab_sample_gate_learner_batch_build_per_call_p95_sec=1.4,
            compact_rollout_slab_sample_gate_per_call_p95_sec=1.78,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_p95_sec=1.11,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_p95_sec=0.34,
            compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_p95_sec=0.07,
            compact_rollout_slab_sample_gate_learner_batch_builder_unroll_stack_fields_per_call_p95_sec=0.61,
            compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call_p95_sec=1.15,
            compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call_p95_sec=1.31,
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("sync-a", baseline),
            compare.RowInput("sync-b", candidate),
        ],
        baseline_label="sync-a",
    )

    comparison = report["comparisons"][0]
    baseline_timings = report["rows"][0]["timings"]
    candidate_timings = report["rows"][1]["timings"]
    for timings in (baseline_timings, candidate_timings):
        assert timings["compact_profile_runtime_step_actor_env_runtime_sum_sec"] == 22.0
        assert timings["compact_profile_runtime_step_actor_autoreset_sum_sec"] == 14.0
        assert timings["compact_profile_runtime_step_sample_gate_residual_sum_sec"] == 24.0
        assert timings["compact_profile_runtime_step_sample_gate_cuda_sync_sum_sec"] == 2.0
        assert (
            timings["compact_profile_runtime_step_sample_gate_builder_group_loop_sum_sec"]
            == 64.0
        )
        assert (
            timings["compact_profile_runtime_step_sample_gate_builder_cuda_sync_sum_sec"]
            == 6.9
        )
        assert (
            timings[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_sec"
            ]
            == 50.0
        )
        assert (
            timings[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_sec"
            ]
            == 14.0
        )
        assert (
            timings[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_accounted_per_call_p95_sec"
            ]
            == 0.34
        )
        assert (
            timings[
                "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_residual_per_call_p95_sec"
            ]
            == 0.07
        )
    assert comparison["identity_status"] == "exact"
    assert comparison["identity_matches"] is True
    identity = report["rows"][0]["identity"]
    assert identity["compact_profile_cuda_sync_timing_diagnostics"] is True
    assert identity["compact_profile_runtime_step_timing_diagnostics"] is True
    assert identity["compact_profile_runtime_step_count"] == 1084
    assert identity["compact_rollout_slab_sample_gate_cuda_sync_count"] == 12
    assert (
        identity["compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_count"]
        == 11977
    )
    assert identity["compact_rollout_slab_learner_gate_cuda_sync_count"] == 28
    deltas = {item["field"]: item["delta"] for item in comparison["largest_timing_deltas"]}
    assert deltas["compact_rollout_slab_sample_gate_cuda_sync_sec"] == pytest.approx(-2.01)
    assert deltas[
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec"
    ] == pytest.approx(-1.4)
    assert deltas["compact_rollout_slab_learner_gate_cuda_sync_sec"] == pytest.approx(-2.01)
    assert deltas["compact_profile_runtime_step_sum_sec"] == pytest.approx(40.0)
    assert deltas["compact_profile_runtime_step_actor_step_wall_sum_sec"] == pytest.approx(
        10.0
    )
    assert deltas["compact_profile_runtime_step_sample_gate_sum_sec"] == pytest.approx(30.0)
    assert deltas["compact_profile_runtime_step_primary_residual_sum_sec"] == pytest.approx(
        5.0
    )
    assert deltas[
        "compact_rollout_slab_sample_gate_learner_batch_build_per_call_p95_sec"
    ] == pytest.approx(0.6)
    assert deltas["compact_rollout_slab_sample_gate_per_call_p95_sec"] == pytest.approx(0.63)
    assert deltas[
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_p95_sec"
    ] == pytest.approx(0.7)
    assert deltas[
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call_p95_sec"
    ] == pytest.approx(1.1)
    assert deltas[
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call_p95_sec"
    ] == pytest.approx(1.2)
    ranges = {
        item["field"]: item
        for item in report["exact_repeat_stability"]["largest_timing_ranges"]
    }
    assert ranges[
        "compact_rollout_slab_sample_gate_learner_batch_builder_cuda_sync_sec"
    ]["range"] == pytest.approx(1.4)
    assert ranges["compact_profile_runtime_step_sum_sec"]["range"] == pytest.approx(40.0)
    assert ranges["compact_profile_runtime_step_actor_step_wall_sum_sec"][
        "range"
    ] == pytest.approx(10.0)
    assert ranges["compact_profile_runtime_step_sample_gate_sum_sec"][
        "range"
    ] == pytest.approx(30.0)
    assert ranges["compact_profile_runtime_step_primary_residual_sum_sec"][
        "range"
    ] == pytest.approx(5.0)
    assert ranges["compact_rollout_slab_sample_gate_per_call_p95_sec"][
        "range"
    ] == pytest.approx(0.63)
    assert ranges[
        "compact_rollout_slab_sample_gate_learner_batch_builder_group_loop_per_call_p95_sec"
    ]["range"] == pytest.approx(0.7)
    assert ranges[
        "compact_rollout_slab_sample_gate_learner_batch_builder_unroll_terminal_window_hint_per_call_p95_sec"
    ]["range"] == pytest.approx(1.1)
    assert ranges[
        "compact_rollout_slab_sample_gate_learner_batch_builder_terminal_metadata_mask_per_call_p95_sec"
    ]["range"] == pytest.approx(1.2)


def test_compare_speed_rows_includes_cpu_perf_stat_fields(tmp_path):
    baseline = _write_row(
        tmp_path / "perf_a.json",
        _accepted_summary(
            compact_profile_cpu_perf_stat_diagnostics=True,
            compact_profile_cpu_perf_stat_task_clock_sec=120.0,
            compact_profile_cpu_perf_stat_cycles=200_000_000_000.0,
            compact_profile_cpu_perf_stat_ref_cycles=240_000_000_000.0,
            compact_profile_cpu_perf_stat_instructions=500_000_000_000.0,
            compact_profile_cpu_perf_stat_instructions_per_cycle=2.5,
            compact_profile_cpu_perf_stat_cache_references=10_000_000.0,
            compact_profile_cpu_perf_stat_cache_misses=1_000_000.0,
            compact_profile_cpu_perf_stat_cache_miss_rate=0.1,
            compact_profile_cpu_perf_stat_context_switches=4.0,
        ),
    )
    candidate = _write_row(
        tmp_path / "perf_b.json",
        _accepted_summary(
            compact_profile_cpu_perf_stat_diagnostics=True,
            compact_profile_cpu_perf_stat_task_clock_sec=135.0,
            compact_profile_cpu_perf_stat_cycles=230_000_000_000.0,
            compact_profile_cpu_perf_stat_ref_cycles=240_000_000_000.0,
            compact_profile_cpu_perf_stat_instructions=500_000_000_000.0,
            compact_profile_cpu_perf_stat_instructions_per_cycle=2.1739130434782608,
            compact_profile_cpu_perf_stat_cache_references=14_000_000.0,
            compact_profile_cpu_perf_stat_cache_misses=2_800_000.0,
            compact_profile_cpu_perf_stat_cache_miss_rate=0.2,
            compact_profile_cpu_perf_stat_context_switches=4.0,
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("perf-a", baseline),
            compare.RowInput("perf-b", candidate),
        ],
        baseline_label="perf-a",
    )

    assert report["rows"][0]["identity"][
        "compact_profile_cpu_perf_stat_diagnostics"
    ] is True
    timings = report["rows"][1]["timings"]
    assert timings["compact_profile_cpu_perf_stat_task_clock_sec"] == 135.0
    assert timings["compact_profile_cpu_perf_stat_cycles"] == 230_000_000_000.0
    assert timings["compact_profile_cpu_perf_stat_cache_miss_rate"] == 0.2
    deltas = {
        item["field"]: item["delta"]
        for item in report["comparisons"][0]["largest_timing_deltas"]
    }
    assert deltas["compact_profile_cpu_perf_stat_cycles"] == pytest.approx(
        30_000_000_000.0
    )
    assert deltas["compact_profile_cpu_perf_stat_task_clock_sec"] == pytest.approx(
        15.0
    )
    assert deltas[
        "compact_profile_cpu_perf_stat_instructions_per_cycle"
    ] == pytest.approx(-0.32608695652173925)
    ranges = {
        item["field"]: item
        for item in report["exact_repeat_stability"]["largest_timing_ranges"]
    }
    assert ranges["compact_profile_cpu_perf_stat_task_clock_sec"]["range"] == (
        pytest.approx(15.0)
    )


def test_compare_speed_rows_includes_builder_child_cpu_time_fields(tmp_path):
    group_loop_process_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "group_loop_process_cpu_time_delta_ns"
    )
    group_loop_thread_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "group_loop_thread_cpu_time_delta_ns"
    )
    unroll_process_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "unroll_fields_process_cpu_time_delta_ns"
    )
    unroll_stack_process_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "unroll_stack_fields_process_cpu_time_delta_ns"
    )
    final_gather_process_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_gather_process_cpu_time_delta_ns"
    )
    final_validate_process_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "terminal_metadata_final_observation_validate_process_cpu_time_delta_ns"
    )
    unroll_residual_process_key = (
        "compact_rollout_slab_sample_gate_learner_batch_builder_"
        "unroll_fields_residual_process_cpu_time_delta_ns"
    )
    baseline = _write_row(
        tmp_path / "builder_cpu_a.json",
        _accepted_summary(
            **{
                group_loop_process_key: 120_000_000,
                group_loop_thread_key: 110_000_000,
                unroll_process_key: 70_000_000,
                unroll_stack_process_key: 30_000_000,
                unroll_residual_process_key: 10_000_000,
                final_gather_process_key: 20_000_000,
                final_validate_process_key: 12_000_000,
            }
        ),
    )
    candidate = _write_row(
        tmp_path / "builder_cpu_b.json",
        _accepted_summary(
            **{
                group_loop_process_key: 180_000_000,
                group_loop_thread_key: 155_000_000,
                unroll_process_key: 90_000_000,
                unroll_stack_process_key: 65_000_000,
                unroll_residual_process_key: 50_000_000,
                final_gather_process_key: 45_000_000,
                final_validate_process_key: 30_000_000,
            }
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("builder-cpu-a", baseline),
            compare.RowInput("builder-cpu-b", candidate),
        ],
        baseline_label="builder-cpu-a",
    )

    candidate_timings = report["rows"][1]["timings"]
    assert candidate_timings[group_loop_process_key] == 180_000_000.0
    assert candidate_timings[group_loop_thread_key] == 155_000_000.0
    assert candidate_timings[unroll_stack_process_key] == 65_000_000.0
    assert candidate_timings[unroll_residual_process_key] == 50_000_000.0
    assert candidate_timings[final_gather_process_key] == 45_000_000.0
    assert candidate_timings[final_validate_process_key] == 30_000_000.0
    deltas = {
        item["field"]: item["delta"]
        for item in report["comparisons"][0]["largest_timing_deltas"]
    }
    assert deltas[group_loop_process_key] == pytest.approx(60_000_000.0)
    assert deltas[group_loop_thread_key] == pytest.approx(45_000_000.0)
    assert deltas[unroll_residual_process_key] == pytest.approx(40_000_000.0)
    assert deltas[unroll_stack_process_key] == pytest.approx(35_000_000.0)
    assert deltas[final_gather_process_key] == pytest.approx(25_000_000.0)
    assert deltas[unroll_process_key] == pytest.approx(20_000_000.0)
    assert deltas[final_validate_process_key] == pytest.approx(18_000_000.0)


def test_compare_speed_rows_includes_runtime_step_cadence_fields(tmp_path):
    baseline = _write_row(
        tmp_path / "cadence_a.json",
        _accepted_summary(
            training_wall_sec=20.0,
            steps_per_sec=9000.0,
            compact_profile_runtime_step_timing_diagnostics=True,
            compact_profile_runtime_step_count=1084,
            compact_profile_runtime_step_sum_sec=20.0,
            compact_profile_runtime_step_sample_gate_active_count=12,
            compact_profile_runtime_step_sample_gate_active_sum_sec=10.0,
            compact_profile_runtime_step_sample_gate_inactive_count=1072,
            compact_profile_runtime_step_early_sum_sec=6.0,
            compact_profile_runtime_step_mid_sum_sec=7.0,
            compact_profile_runtime_step_late_sum_sec=7.0,
            compact_profile_runtime_step_late_sample_gate_sum_sec=3.0,
            compact_profile_runtime_step_late_sample_gate_builder_group_loop_sum_sec=2.0,
            compact_profile_runtime_step_late_sample_gate_active_sample_gate_count=4,
            compact_profile_runtime_step_late_sample_gate_active_sample_gate_p95_sec=0.75,
            compact_profile_runtime_step_late_sample_gate_active_sample_gate_builder_group_loop_p95_sec=0.55,
        ),
    )
    candidate = _write_row(
        tmp_path / "cadence_b.json",
        _accepted_summary(
            training_wall_sec=30.0,
            steps_per_sec=6000.0,
            compact_profile_runtime_step_timing_diagnostics=True,
            compact_profile_runtime_step_count=1084,
            compact_profile_runtime_step_sum_sec=30.0,
            compact_profile_runtime_step_sample_gate_active_count=12,
            compact_profile_runtime_step_sample_gate_active_sum_sec=18.0,
            compact_profile_runtime_step_sample_gate_inactive_count=1072,
            compact_profile_runtime_step_early_sum_sec=8.0,
            compact_profile_runtime_step_mid_sum_sec=9.0,
            compact_profile_runtime_step_late_sum_sec=13.0,
            compact_profile_runtime_step_late_sample_gate_sum_sec=8.0,
            compact_profile_runtime_step_late_sample_gate_builder_group_loop_sum_sec=5.0,
            compact_profile_runtime_step_late_sample_gate_active_sample_gate_count=4,
            compact_profile_runtime_step_late_sample_gate_active_sample_gate_p95_sec=1.25,
            compact_profile_runtime_step_late_sample_gate_active_sample_gate_builder_group_loop_p95_sec=0.95,
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("cadence-a", baseline),
            compare.RowInput("cadence-b", candidate),
        ],
        baseline_label="cadence-a",
    )

    timings = report["rows"][1]["timings"]
    assert timings["compact_profile_runtime_step_sample_gate_active_count"] == 12.0
    assert timings["compact_profile_runtime_step_sample_gate_inactive_count"] == 1072.0
    deltas = {item["field"]: item["delta"] for item in report["comparisons"][0]["largest_timing_deltas"]}
    assert deltas["compact_profile_runtime_step_sample_gate_active_sum_sec"] == (
        pytest.approx(8.0)
    )
    assert deltas["compact_profile_runtime_step_late_sample_gate_sum_sec"] == (
        pytest.approx(5.0)
    )
    assert deltas[
        "compact_profile_runtime_step_late_sample_gate_builder_group_loop_sum_sec"
    ] == pytest.approx(3.0)
    assert deltas[
        "compact_profile_runtime_step_late_sample_gate_active_sample_gate_p95_sec"
    ] == pytest.approx(0.5)
    assert deltas[
        "compact_profile_runtime_step_late_sample_gate_active_sample_gate_builder_group_loop_p95_sec"
    ] == pytest.approx(0.4)
    ranges = {
        item["field"]: item
        for item in report["exact_repeat_stability"]["largest_timing_ranges"]
    }
    assert ranges["compact_profile_runtime_step_sample_gate_active_sum_sec"][
        "range"
    ] == pytest.approx(8.0)
    assert ranges["compact_profile_runtime_step_late_sample_gate_sum_sec"][
        "range"
    ] == pytest.approx(5.0)
    assert ranges[
        "compact_profile_runtime_step_late_sample_gate_active_sample_gate_p95_sec"
    ]["range"] == pytest.approx(0.5)


def test_compare_speed_rows_projects_gpu_utilization_fields(tmp_path):
    baseline = _write_row(
        tmp_path / "gpu_a.json",
        _accepted_summary(
            speed_row_gpu_utilization_sampling_enabled=True,
            speed_row_gpu_utilization_sample_interval_sec=0.5,
            speed_row_gpu_utilization_sample_count=420,
            speed_row_gpu_name="NVIDIA H100 80GB HBM3",
            speed_row_gpu_utilization_max_percent=72.0,
            speed_row_gpu_utilization_mean_percent=31.5,
            speed_row_gpu_utilization_nonzero_sample_count=300,
            speed_row_gpu_utilization_over_50_sample_count=40,
            speed_row_gpu_utilization_over_80_sample_count=0,
            speed_row_gpu_memory_utilization_max_percent=44.0,
            speed_row_gpu_memory_used_max_mib=45_000.0,
            speed_row_gpu_power_draw_max_w=420.0,
            speed_row_gpu_utilization_sampling_errors=[],
        ),
    )
    candidate = _write_row(
        tmp_path / "gpu_b.json",
        _accepted_summary(
            speed_row_gpu_utilization_sampling_enabled=True,
            speed_row_gpu_utilization_sample_interval_sec=0.5,
            speed_row_gpu_utilization_sample_count=460,
            speed_row_gpu_name="NVIDIA H100 80GB HBM3",
            speed_row_gpu_utilization_max_percent=84.0,
            speed_row_gpu_utilization_mean_percent=39.5,
            speed_row_gpu_utilization_nonzero_sample_count=350,
            speed_row_gpu_utilization_over_50_sample_count=55,
            speed_row_gpu_utilization_over_80_sample_count=3,
            speed_row_gpu_memory_utilization_max_percent=48.0,
            speed_row_gpu_memory_used_max_mib=47_500.0,
            speed_row_gpu_power_draw_max_w=465.0,
            speed_row_gpu_utilization_sampling_errors=["unit"],
        ),
    )

    report = compare.compare_rows(
        [
            compare.RowInput("gpu-a", baseline),
            compare.RowInput("gpu-b", candidate),
        ],
        baseline_label="gpu-a",
    )

    comparison = report["comparisons"][0]
    assert comparison["identity_status"] == "exact"
    assert report["rows"][0]["gpu_utilization"] == {
        "speed_row_gpu_utilization_sampling_enabled": True,
        "speed_row_gpu_utilization_sample_interval_sec": 0.5,
        "speed_row_gpu_utilization_sample_count": 420,
        "speed_row_gpu_name": "NVIDIA H100 80GB HBM3",
        "speed_row_gpu_utilization_max_percent": 72.0,
        "speed_row_gpu_utilization_mean_percent": 31.5,
        "speed_row_gpu_utilization_nonzero_sample_count": 300,
        "speed_row_gpu_utilization_over_50_sample_count": 40,
        "speed_row_gpu_utilization_over_80_sample_count": 0,
        "speed_row_gpu_memory_utilization_max_percent": 44.0,
        "speed_row_gpu_memory_used_max_mib": 45_000.0,
        "speed_row_gpu_power_draw_max_w": 420.0,
        "speed_row_gpu_utilization_sampling_errors": [],
    }
    assert report["rows"][1]["gpu_utilization"][
        "speed_row_gpu_utilization_sampling_errors"
    ] == ["unit"]
    candidate_timings = report["rows"][1]["timings"]
    assert candidate_timings["speed_row_gpu_utilization_max_percent"] == 84.0
    assert candidate_timings["speed_row_gpu_memory_used_max_mib"] == 47_500.0
    deltas = {item["field"]: item["delta"] for item in comparison["largest_timing_deltas"]}
    assert deltas["speed_row_gpu_memory_used_max_mib"] == pytest.approx(2_500.0)
    assert deltas["speed_row_gpu_power_draw_max_w"] == pytest.approx(45.0)
    assert deltas["speed_row_gpu_utilization_over_50_sample_count"] == pytest.approx(
        15.0
    )
