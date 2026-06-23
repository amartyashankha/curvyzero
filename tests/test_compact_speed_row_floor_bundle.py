from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from curvyzero.training import compact_speed_row_floor_bundle as floor
from curvyzero.training import compact_torch_service_bucket_decision as service_decision
from curvyzero.training.compact_compile_eager_speed_pair import (
    CompileEagerPairInput,
)
from curvyzero.training.compact_compile_eager_speed_pair import (
    build_compact_compile_eager_speed_pair_v1,
)
from curvyzero.training.compact_compile_eager_speed_pair import (
    validate_compact_compile_eager_speed_pair_v1,
)
from curvyzero.training.compact_coach_speed_row import (
    build_compact_coach_speed_row_evidence_v1,
)
from curvyzero.training.compact_model_compile_decision import (
    DECISION_PARK_SPEED_UNAPPROVED,
)
from curvyzero.training.compact_model_compile_decision import (
    build_compact_model_compile_decision_v1,
)
from curvyzero.training.compact_model_compile_decision import (
    validate_compact_model_compile_decision_v1,
)


def test_speed_row_floor_bundle_binds_same_denominator_rows(tmp_path):
    inputs = _write_floor_inputs(tmp_path)

    payload = floor.build_compact_speed_row_floor_bundle_v1(
        run_id="unit-floor-bundle",
        accepted_speed_row_report_path=inputs["accepted"],
        compact_torch_sibling_report_path=inputs["compact_torch"],
        fixed_floor_sibling_report_path=inputs["fixed_floor"],
        created_at="2026-05-30T00:00:00+00:00",
    )

    assert payload["schema_id"] == floor.COMPACT_SPEED_ROW_FLOOR_BUNDLE_SCHEMA_ID
    assert payload["ok"] is True
    assert payload["status"] == floor.STATUS_COMPLETE
    assert payload["denominator_check"]["same_denominator"] is True
    assert (
        payload["rows"]["compact_torch_search_service_sibling"]["search_impl_kind"]
        == "compact_torch_search_service"
    )
    assert (
        payload["rows"]["fixed_no_search_floor_sibling"]["search_impl_kind"]
        == "fixed_shape_search_owner"
    )
    comparisons = payload["comparisons"]
    assert comparisons["search_delta_sec"] > 0.0
    assert comparisons["compact_rollout_slab_delta_sec"] > comparisons["search_delta_sec"]
    compact_timing = payload["rows"]["compact_torch_search_service_sibling"][
        "timing_buckets"
    ]
    assert compact_timing["search_service_action_wall_sec"] > compact_timing[
        "search_service_total_sec"
    ]
    assert compact_timing["search_service_action_accounted_sec"] == pytest.approx(
        3.521
    )
    assert compact_timing["search_service_action_residual_sec"] == pytest.approx(0.2)
    assert compact_timing["search_service_initial_output_decode_sec"] == pytest.approx(
        0.17
    )
    assert compact_timing["search_service_root_output_decode_sec"] == pytest.approx(
        0.17
    )
    assert compact_timing["search_service_tree_root_prior_build_sec"] == pytest.approx(
        0.29
    )
    assert compact_timing["search_service_tree_total_sec"] == pytest.approx(2.65)
    assert compact_timing["search_service_tree_accounted_sec"] == pytest.approx(2.6)
    assert compact_timing["search_service_tree_residual_sec"] == pytest.approx(0.05)
    assert compact_timing["search_service_action_readback_sec"] == pytest.approx(0.08)
    assert compact_timing["search_service_core_accounted_sec"] == pytest.approx(3.43)
    assert compact_timing["search_service_core_residual_sec"] == pytest.approx(0.07)
    assert compact_timing["search_service_inference_guard_enter_sec"] == pytest.approx(
        0.031
    )
    assert compact_timing["search_service_inference_guard_exit_sec"] == pytest.approx(
        0.032
    )
    assert compact_timing["search_service_inference_guard_total_sec"] == pytest.approx(
        0.063
    )
    compact_accounting = payload["rows"]["compact_torch_search_service_sibling"][
        "timing_accounting"
    ]
    assert compact_accounting["search_service_action_wall_over_total_sec"] > 0.0
    assert compact_accounting["compact_rollout_slab_non_action_service_sec"] > 0.0
    assert compact_timing["search_service_one_simulation_fast_path_count"] == 1.0
    assert (
        compact_timing[
            "search_service_one_simulation_root_prior_softmax_skipped_count"
        ]
        == 1.0
    )
    assert compact_timing["search_service_recurrent_inference_calls"] == 1.0
    compact_flags = payload["rows"]["compact_torch_search_service_sibling"][
        "search_service_flags"
    ]
    assert compact_flags["one_simulation_root_prior_softmax_skipped_last"] is True
    assert compact_flags["one_simulation_selection_mode_last"] == "masked_logits_argmax"
    assert compact_flags["action_readback_bytes_match_int16_selected_actions"] is True
    assert compact_timing["search_service_initial_inference_sync_sec"] == pytest.approx(
        0.2
    )
    assert compact_timing[
        "search_service_initial_inference_cuda_event_sec"
    ] == pytest.approx(0.21)
    assert compact_timing[
        "search_service_tree_recurrent_inference_enqueue_sec"
    ] == pytest.approx(0.4)
    assert compact_timing[
        "search_service_tree_recurrent_inference_cuda_event_sec"
    ] == pytest.approx(0.41)
    assert compact_timing["search_service_tree_sync_sec"] == pytest.approx(0.6)
    assert compact_timing["search_service_tree_cuda_event_sec"] == pytest.approx(0.61)
    assert comparisons["high_level_deltas"][
        "slab_search_service_action_wall_delta_sec"
    ] > 0.0
    assert comparisons["high_level_deltas"][
        "slab_search_service_action_postprocess_delta_sec"
    ] > 0.0
    assert comparisons["high_level_deltas"][
        "slab_search_service_action_accounted_delta_sec"
    ] == pytest.approx(3.521)
    assert comparisons["high_level_deltas"][
        "slab_search_service_action_residual_delta_sec"
    ] == pytest.approx(0.2)
    assert comparisons["high_level_deltas"][
        "slab_search_service_inference_guard_total_delta_sec"
    ] == pytest.approx(0.063)
    assert comparisons["high_level_deltas"][
        "slab_search_service_initial_output_decode_delta_sec"
    ] == pytest.approx(0.17)
    assert comparisons["high_level_deltas"][
        "slab_search_service_tree_root_prior_build_delta_sec"
    ] == pytest.approx(0.29)
    assert comparisons["high_level_deltas"][
        "slab_search_service_tree_total_delta_sec"
    ] == pytest.approx(2.65)
    assert comparisons["high_level_deltas"][
        "slab_search_service_core_residual_delta_sec"
    ] == pytest.approx(0.07)
    assert comparisons["high_level_deltas"][
        "slab_internal_accounted_delta_sec"
    ] > comparisons["compact_rollout_slab_delta_sec"]
    assert comparisons["dominant_high_level_delta_key"] == "compact_rollout_slab_delta_sec"
    assert (
        comparisons["compact_rollout_slab_non_service_delta_sec"]
        > comparisons["search_delta_sec"]
    )
    assert "compact_rollout_slab_non_dispatch_delta_sec" in comparisons
    assert "slab_search_dispatch_residual_delta_sec" in comparisons
    assert (
        "search_service_action_wall_over_total_delta_sec"
        in comparisons["high_level_deltas"]
    )
    assert (
        payload["engineering_read"]["classification"]
        == "compact_rollout_slab_non_service_dominant"
    )
    assert payload["engineering_read"]["search_dominance_claim"] is False
    assert payload["engineering_read"]["residual_accounting_required"] is True
    assert (
        payload["engineering_read"]["next_target"]
        == "decompose_compact_rollout_slab_commit_flush_materialization"
    )
    assert payload["non_claims"]["promotion_claim"] is False
    floor.validate_compact_speed_row_floor_bundle_v1(payload)


def test_speed_row_floor_bundle_splits_dispatch_envelope_from_service_wall(tmp_path):
    inputs = _write_floor_inputs(
        tmp_path,
        compact_search_dispatch_wall_sec=3.73,
        floor_search_dispatch_wall_sec=0.31,
    )

    payload = floor.build_compact_speed_row_floor_bundle_v1(
        run_id="unit-floor-bundle-dispatch-split",
        accepted_speed_row_report_path=inputs["accepted"],
        compact_torch_sibling_report_path=inputs["compact_torch"],
        fixed_floor_sibling_report_path=inputs["fixed_floor"],
        created_at="2026-05-30T00:00:00+00:00",
    )

    comparisons = payload["comparisons"]
    deltas = comparisons["high_level_deltas"]
    assert deltas["slab_search_dispatch_wall_delta_sec"] > comparisons[
        "search_delta_sec"
    ]
    assert abs(comparisons["slab_search_dispatch_residual_delta_sec"]) < 0.01
    assert comparisons["compact_rollout_slab_non_dispatch_delta_sec"] > 0.0
    assert comparisons["compact_rollout_slab_non_action_service_delta_sec"] > 0.0
    assert comparisons["search_service_action_wall_over_total_delta_sec"] > 0.0
    assert comparisons["slab_search_dispatch_residual_abs_share_of_measured_gap"] < 0.01
    assert (
        payload["engineering_read"]["classification"]
        == "compact_rollout_slab_search_dispatch_wall_dominant"
    )
    assert (
        payload["engineering_read"]["next_target"]
        == "decompose_compact_torch_search_dispatch_envelope"
    )
    floor.validate_compact_speed_row_floor_bundle_v1(payload)


def test_compile_eager_pair_review_blocks_speed_claim_on_trajectory_mismatch(
    tmp_path,
):
    pairs = _write_compile_eager_pair_inputs(tmp_path, trajectory_match=False)

    payload = build_compact_compile_eager_speed_pair_v1(
        run_id="unit-compile-eager-review",
        pairs=pairs,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["ok"] is True
    assert (
        payload["aggregate"]["decision"]
        == "not_approved_action_trajectory_mismatch"
    )
    assert payload["aggregate"]["all_compile_faster"] is True
    assert payload["aggregate"]["all_safety_checks_passed"] is True
    assert payload["aggregate"]["all_action_trajectory_match"] is False
    assert payload["aggregate"]["speed_claim_allowed"] is False
    assert payload["attached_claims"]["narrow_compile_speed_claim_allowed"] is False
    assert payload["non_claims"]["training_speedup_claim"] is False
    validate_compact_compile_eager_speed_pair_v1(payload)


def test_compile_eager_pair_review_approves_only_when_trajectories_match(tmp_path):
    pairs = _write_compile_eager_pair_inputs(tmp_path, trajectory_match=True)

    payload = build_compact_compile_eager_speed_pair_v1(
        run_id="unit-compile-eager-review",
        pairs=pairs,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["aggregate"]["decision"] == "approved_compile_faster_same_trajectory"
    assert payload["aggregate"]["speed_claim_allowed"] is True
    assert payload["aggregate"]["minimum_wall_win_fraction"] == pytest.approx(0.2)
    validate_compact_compile_eager_speed_pair_v1(payload)


def test_compile_eager_pair_review_blocks_when_compile_not_faster(tmp_path):
    pairs = _write_compile_eager_pair_inputs(
        tmp_path,
        trajectory_match=True,
        compile_wall_sec=12.0,
        eager_wall_sec=10.0,
    )

    payload = build_compact_compile_eager_speed_pair_v1(
        run_id="unit-compile-eager-review",
        pairs=pairs,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["aggregate"]["decision"] == "not_approved_compile_not_faster"
    assert payload["aggregate"]["all_action_trajectory_match"] is True
    assert payload["aggregate"]["all_safety_checks_passed"] is True
    assert payload["aggregate"]["all_compile_faster"] is False
    assert payload["aggregate"]["speed_claim_allowed"] is False
    validate_compact_compile_eager_speed_pair_v1(payload)


def test_compile_eager_pair_review_blocks_pre_inference_guard_rows(tmp_path):
    pairs = _write_compile_eager_pair_inputs(
        tmp_path,
        trajectory_match=True,
        inference_guard_present=False,
    )

    payload = build_compact_compile_eager_speed_pair_v1(
        run_id="unit-compile-eager-review",
        pairs=pairs,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["aggregate"]["decision"] == "not_approved_safety_failed"
    assert payload["aggregate"]["all_safety_checks_passed"] is False
    failed = payload["pairs"][0]["safety_check"]["eager"]["failed_checks"]
    assert "model_eval_applied_for_inference" in failed
    assert "model_inference_mode_used" in failed
    assert payload["aggregate"]["speed_claim_allowed"] is False
    validate_compact_compile_eager_speed_pair_v1(payload)


def test_model_compile_decision_parks_speed_unapproved_default(tmp_path):
    root_tape_result = _write_fixed_root_tape_result(tmp_path / "root-tape")
    post_guard_report = _write_compile_eager_pair_report(
        tmp_path / "post-guard",
        run_id="unit-post-guard",
        trajectory_match=True,
        compile_wall_sec=12.0,
        eager_wall_sec=10.0,
    )
    prior_report = _write_compile_eager_pair_report(
        tmp_path / "prior",
        run_id="unit-prior",
        trajectory_match=False,
        compile_wall_sec=8.0,
        eager_wall_sec=10.0,
    )

    payload = build_compact_model_compile_decision_v1(
        run_id="unit-model-compile-decision",
        fixed_root_tape_result_path=root_tape_result,
        post_guard_speed_pair_report_path=post_guard_report,
        prior_speed_pair_report_path=prior_report,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["decision"] == DECISION_PARK_SPEED_UNAPPROVED
    assert payload["fixed_root_tape_gate"]["root_tape_parity_passed"] is True
    assert payload["post_guard_closed_loop_pair"][
        "all_action_trajectory_match"
    ] is True
    assert payload["post_guard_closed_loop_pair"]["all_compile_faster"] is False
    assert payload["prior_closed_loop_pair"]["all_action_trajectory_match"] is False
    assert payload["attached_claims"][
        "model_compile_default_speed_default_allowed"
    ] is False
    assert (
        payload["interpretation"]["speed_read"]
        == "service_model_buckets_improve_but_wall_speed_not_repeatable"
    )
    assert payload["non_claims"]["training_speedup_claim"] is False
    validate_compact_model_compile_decision_v1(payload)


def test_model_compile_decision_rejects_source_hash_drift(tmp_path):
    root_tape_result = _write_fixed_root_tape_result(tmp_path / "root-tape")
    post_guard_report = _write_compile_eager_pair_report(
        tmp_path / "post-guard",
        run_id="unit-post-guard",
        trajectory_match=True,
        compile_wall_sec=12.0,
        eager_wall_sec=10.0,
    )
    payload = build_compact_model_compile_decision_v1(
        run_id="unit-model-compile-decision",
        fixed_root_tape_result_path=root_tape_result,
        post_guard_speed_pair_report_path=post_guard_report,
    )
    report = _read_json(post_guard_report)
    report["aggregate"]["decision"] = "tampered"
    _write_json(post_guard_report, report)

    with pytest.raises(
        ValueError,
        match="post_guard_closed_loop_pair.input_ref sha mismatch",
    ):
        validate_compact_model_compile_decision_v1(payload)


def test_model_compile_decision_cli_writes_report_and_manifest(tmp_path):
    module = _load_model_compile_decision_cli_module()
    root_tape_result = _write_fixed_root_tape_result(tmp_path / "root-tape")
    post_guard_report = _write_compile_eager_pair_report(
        tmp_path / "post-guard",
        run_id="unit-post-guard",
        trajectory_match=True,
        compile_wall_sec=12.0,
        eager_wall_sec=10.0,
    )
    output_root = tmp_path / "out"
    run_id = "unit-model-compile-decision-cli"
    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--fixed-root-tape-result",
        str(root_tape_result),
        "--post-guard-pair-report",
        str(post_guard_report),
        "--no-prior-report",
    ]

    assert module.main(argv) == 0
    report_path = output_root / run_id / "model_compile_decision_report.json"
    manifest_path = output_root / run_id / "manifest.json"
    payload = _read_json(report_path)
    manifest = _read_json(manifest_path)
    assert payload["decision"] == DECISION_PARK_SPEED_UNAPPROVED
    assert manifest["decision"] == DECISION_PARK_SPEED_UNAPPROVED
    assert manifest["model_compile_default_speed_default_allowed"] is False
    validate_compact_model_compile_decision_v1(payload)

    with pytest.raises(FileExistsError):
        module.main(argv)


def test_service_bucket_decision_selects_initial_model_path_from_event_repeat(tmp_path):
    canonical_inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    repeat_inputs = _write_floor_inputs(
        tmp_path / "event-inputs",
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_initial_cuda_event_sec=1.2,
        compact_initial_representation_cuda_event_sec=0.7,
        compact_initial_prediction_cuda_event_sec=0.4,
        compact_initial_direct_core_cuda_event_sec=1.1,
        compact_initial_direct_core_cuda_event_residual_sec=0.1,
        compact_initial_direct_requested_count=180.0,
        compact_initial_direct_used_count=180.0,
        compact_initial_inference_mode_requested="direct_core",
        compact_initial_inference_mode_effective="direct_core",
        compact_initial_inference_runtime_status="direct_core_used",
        compact_recurrent_cuda_event_sec=0.2,
        compact_tree_cuda_event_sec=0.3,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        canonical_inputs,
        run_id="unit-service-bucket-canonical",
    )
    repeat = _write_floor_bundle_report(
        tmp_path / "repeat-report",
        repeat_inputs,
        run_id="unit-service-bucket-repeat",
    )

    payload = service_decision.build_compact_torch_service_bucket_decision_v1(
        run_id="unit-service-bucket-decision",
        canonical_floor_bundle_path=canonical,
        repeat_floor_bundle_paths=[repeat],
        min_repeat_count=2,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["decision"] == service_decision.DECISION_SELECT_INITIAL_MODEL_FORWARD
    assert payload["selected_next_target"] == service_decision.TARGET_INITIAL_MODEL_FORWARD
    assert payload["input_refs"]["repeat_floor_bundles"][0]["path"] == str(
        repeat.resolve()
    )
    assert payload["repeat_read"]["same_denominator_evidence_count"] == 2
    assert payload["bucket_summary"]["event_timing_present"] is True
    assert payload["bucket_summary"]["initial_model_bucket_dominant"] is True
    assert payload["bucket_summary"][
        "initial_representation_cuda_event_sec"
    ] == pytest.approx(0.7)
    assert payload["bucket_summary"][
        "initial_prediction_cuda_event_sec"
    ] == pytest.approx(0.4)
    assert payload["guard_checks"]["canonical_checks"][
        "direct_core_cuda_event_timing_complete"
    ] is True
    assert payload["bucket_summary"]["service_inference_guard_total_sec"] == pytest.approx(
        0.063
    )
    assert payload["attached_claims"]["next_optimization_code_allowed"] is True
    assert payload["attached_claims"]["training_speedup_claim"] is False
    assert payload["parked_options"]["gpu_mechanics"]["status"] == "parked"
    assert payload["parked_options"]["recurrent_deferral"]["status"] == "parked"
    service_decision.validate_compact_torch_service_bucket_decision_v1(payload)


@pytest.mark.parametrize(
    ("omitted_field", "expected_bucket"),
    (
        (
            "compact_rollout_slab_search_service_initial_inference_cuda_event_sec",
            "search_service_initial_inference_cuda_event_sec",
        ),
        (
            (
                "compact_rollout_slab_search_service_tree_recurrent_inference_"
                "cuda_event_sec"
            ),
            "search_service_tree_recurrent_inference_cuda_event_sec",
        ),
        (
            "compact_rollout_slab_search_service_tree_cuda_event_sec",
            "search_service_tree_cuda_event_sec",
        ),
    ),
)
def test_speed_row_floor_bundle_rejects_missing_cuda_event_timing(
    tmp_path,
    omitted_field,
    expected_bucket,
):
    inputs = _write_floor_inputs(
        tmp_path,
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_omit_timing_fields={omitted_field},
    )

    with pytest.raises(
        floor.CompactSpeedRowFloorBundleError,
        match=f"cuda-event timing incomplete: {expected_bucket}",
    ):
        floor.build_compact_speed_row_floor_bundle_v1(
            run_id="unit-floor-bundle-missing-event-timing",
            accepted_speed_row_report_path=inputs["accepted"],
            compact_torch_sibling_report_path=inputs["compact_torch"],
            fixed_floor_sibling_report_path=inputs["fixed_floor"],
        )


def test_speed_row_floor_bundle_carries_direct_core_cuda_event_split(tmp_path):
    inputs = _write_floor_inputs(
        tmp_path,
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_initial_representation_cuda_event_sec=0.12,
        compact_initial_prediction_cuda_event_sec=0.07,
        compact_initial_direct_core_cuda_event_sec=0.19,
        compact_initial_direct_core_cuda_event_residual_sec=0.02,
        compact_initial_direct_requested_count=180.0,
        compact_initial_direct_used_count=180.0,
        compact_initial_inference_mode_requested="direct_core",
        compact_initial_inference_mode_effective="direct_core",
        compact_initial_inference_runtime_status="direct_core_used",
    )

    payload = floor.build_compact_speed_row_floor_bundle_v1(
        run_id="unit-floor-bundle-direct-core-event-split",
        accepted_speed_row_report_path=inputs["accepted"],
        compact_torch_sibling_report_path=inputs["compact_torch"],
        fixed_floor_sibling_report_path=inputs["fixed_floor"],
    )

    compact_timing = payload["rows"]["compact_torch_search_service_sibling"][
        "timing_buckets"
    ]
    assert compact_timing[
        "search_service_initial_inference_representation_cuda_event_sec"
    ] == pytest.approx(0.12)
    assert compact_timing[
        "search_service_initial_inference_prediction_cuda_event_sec"
    ] == pytest.approx(0.07)
    assert compact_timing[
        "search_service_initial_inference_direct_core_cuda_event_sec"
    ] == pytest.approx(0.19)
    assert compact_timing[
        "search_service_initial_inference_direct_core_cuda_event_residual_sec"
    ] == pytest.approx(0.02)
    assert payload["comparisons"]["high_level_deltas"][
        "slab_search_service_initial_inference_representation_cuda_event_delta_sec"
    ] == pytest.approx(0.12)
    assert payload["rows"]["compact_torch_search_service_sibling"][
        "search_service_flags"
    ]["initial_inference_direct_used"] is True


@pytest.mark.parametrize(
    ("omitted_field", "expected_bucket"),
    (
        (
            "compact_rollout_slab_search_service_initial_inference_representation_cuda_event_sec",
            "search_service_initial_inference_representation_cuda_event_sec",
        ),
        (
            "compact_rollout_slab_search_service_initial_inference_prediction_cuda_event_sec",
            "search_service_initial_inference_prediction_cuda_event_sec",
        ),
        (
            "compact_rollout_slab_search_service_initial_inference_direct_core_cuda_event_sec",
            "search_service_initial_inference_direct_core_cuda_event_sec",
        ),
    ),
)
def test_speed_row_floor_bundle_rejects_missing_direct_core_cuda_event_split(
    tmp_path,
    omitted_field,
    expected_bucket,
):
    inputs = _write_floor_inputs(
        tmp_path,
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_initial_representation_cuda_event_sec=0.12,
        compact_initial_prediction_cuda_event_sec=0.07,
        compact_initial_direct_core_cuda_event_sec=0.19,
        compact_initial_direct_core_cuda_event_residual_sec=0.02,
        compact_initial_direct_requested_count=180.0,
        compact_initial_direct_used_count=180.0,
        compact_initial_inference_mode_requested="direct_core",
        compact_initial_inference_mode_effective="direct_core",
        compact_initial_inference_runtime_status="direct_core_used",
        compact_omit_timing_fields={omitted_field},
    )

    with pytest.raises(
        floor.CompactSpeedRowFloorBundleError,
        match=f"direct-core cuda-event timing incomplete: {expected_bucket}",
    ):
        floor.build_compact_speed_row_floor_bundle_v1(
            run_id="unit-floor-bundle-missing-direct-core-event-split",
            accepted_speed_row_report_path=inputs["accepted"],
            compact_torch_sibling_report_path=inputs["compact_torch"],
            fixed_floor_sibling_report_path=inputs["fixed_floor"],
        )


def test_speed_row_floor_bundle_rejects_compact_torch_resident_host_fallback(
    tmp_path,
):
    inputs = _write_floor_inputs(
        tmp_path,
        compact_resident_observation_host_fallback_count=1.0,
    )

    with pytest.raises(
        floor.CompactSpeedRowFloorBundleError,
        match="resident_observation_host_fallback_count must be zero",
    ):
        floor.build_compact_speed_row_floor_bundle_v1(
            run_id="unit-floor-bundle-host-fallback",
            accepted_speed_row_report_path=inputs["accepted"],
            compact_torch_sibling_report_path=inputs["compact_torch"],
            fixed_floor_sibling_report_path=inputs["fixed_floor"],
        )


@pytest.mark.parametrize(
    ("input_overrides", "expected_check"),
    (
        (
            {"compact_action_d2h_bytes": 737282.0},
            "action_readback_int16_bytes",
        ),
        (
            {"compact_replay_payload_d2h_bytes": 1.0},
            "replay_payload_d2h_zero",
        ),
        (
            {"compact_committed_replay_payload_d2h_bytes": 1.0},
            "committed_replay_payload_d2h_zero",
        ),
    ),
)
def test_service_bucket_decision_rejects_canonical_transfer_budget_regression(
    tmp_path,
    input_overrides,
    expected_check,
):
    floor_input_kwargs = {
        "compact_fast_path_count": 180.0,
        "compact_recurrent_calls": 180.0,
        "compact_action_d2h_bytes": 737280.0,
        **input_overrides,
    }
    inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        **floor_input_kwargs,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        inputs,
        run_id="unit-service-bucket-canonical-transfer-regression",
    )

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match=expected_check,
    ):
        service_decision.build_compact_torch_service_bucket_decision_v1(
            run_id="unit-service-bucket-transfer-regression",
            canonical_floor_bundle_path=canonical,
        )


def test_service_bucket_decision_rejects_repeat_transfer_budget_regression(
    tmp_path,
):
    canonical_inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    repeat_inputs = _write_floor_inputs(
        tmp_path / "repeat-inputs",
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_replay_payload_d2h_bytes=1.0,
        compact_initial_cuda_event_sec=1.2,
        compact_recurrent_cuda_event_sec=0.2,
        compact_tree_cuda_event_sec=0.3,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        canonical_inputs,
        run_id="unit-service-bucket-canonical-clean",
    )
    repeat = _write_floor_bundle_report(
        tmp_path / "repeat-report",
        repeat_inputs,
        run_id="unit-service-bucket-repeat-transfer-regression",
    )

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match="repeat_1 service-bucket guard failed: replay_payload_d2h_zero",
    ):
        service_decision.build_compact_torch_service_bucket_decision_v1(
            run_id="unit-service-bucket-repeat-transfer-regression",
            canonical_floor_bundle_path=canonical,
            repeat_floor_bundle_paths=[repeat],
            min_repeat_count=2,
        )


def test_service_bucket_decision_rejects_recurrent_call_count_mismatch(tmp_path):
    inputs = _write_floor_inputs(
        tmp_path / "inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=179.0,
        compact_action_d2h_bytes=737280.0,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        inputs,
        run_id="unit-service-bucket-canonical-recurrent-mismatch",
    )

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match="recurrent_call_count_matches_steps",
    ):
        service_decision.build_compact_torch_service_bucket_decision_v1(
            run_id="unit-service-bucket-recurrent-mismatch",
            canonical_floor_bundle_path=canonical,
        )


def test_service_bucket_decision_rejects_missing_initial_inference_mode_evidence(
    tmp_path,
):
    inputs = _write_floor_inputs(
        tmp_path / "inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        inputs,
        run_id="unit-service-bucket-canonical-missing-initial-mode",
    )
    payload = _read_json(canonical)
    flags = payload["rows"]["compact_torch_search_service_sibling"][
        "search_service_flags"
    ]
    flags["initial_inference_mode_effective"] = ""
    payload["evidence_ref"] = floor.compact_speed_row_floor_bundle_evidence_ref(payload)
    _write_json(canonical, payload)

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match="initial_inference_mode_recorded",
    ):
        service_decision.build_compact_torch_service_bucket_decision_v1(
            run_id="unit-service-bucket-missing-initial-mode",
            canonical_floor_bundle_path=canonical,
        )


def test_service_bucket_decision_rejects_initial_inference_fallback(tmp_path):
    inputs = _write_floor_inputs(
        tmp_path / "inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        inputs,
        run_id="unit-service-bucket-canonical-initial-fallback",
    )
    payload = _read_json(canonical)
    flags = payload["rows"]["compact_torch_search_service_sibling"][
        "search_service_flags"
    ]
    flags["initial_inference_fallback_count"] = 1.0
    payload["evidence_ref"] = floor.compact_speed_row_floor_bundle_evidence_ref(payload)
    _write_json(canonical, payload)

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match="initial_inference_no_fallback",
    ):
        service_decision.build_compact_torch_service_bucket_decision_v1(
            run_id="unit-service-bucket-initial-fallback",
            canonical_floor_bundle_path=canonical,
        )


@pytest.mark.parametrize(
    ("flag_name", "flag_value", "expected_check"),
    (
        (
            "initial_inference_mode_effective",
            "",
            "initial_inference_mode_recorded",
        ),
        (
            "initial_inference_fallback_count",
            1.0,
            "initial_inference_no_fallback",
        ),
    ),
)
def test_service_bucket_decision_rejects_repeat_initial_inference_regression(
    tmp_path,
    flag_name,
    flag_value,
    expected_check,
):
    canonical_inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    repeat_inputs = _write_floor_inputs(
        tmp_path / "repeat-inputs",
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_initial_cuda_event_sec=1.2,
        compact_recurrent_cuda_event_sec=0.2,
        compact_tree_cuda_event_sec=0.3,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        canonical_inputs,
        run_id="unit-service-bucket-canonical-clean",
    )
    repeat = _write_floor_bundle_report(
        tmp_path / "repeat-report",
        repeat_inputs,
        run_id="unit-service-bucket-repeat-initial-regression",
    )
    payload = _read_json(repeat)
    flags = payload["rows"]["compact_torch_search_service_sibling"][
        "search_service_flags"
    ]
    flags[flag_name] = flag_value
    payload["evidence_ref"] = floor.compact_speed_row_floor_bundle_evidence_ref(payload)
    _write_json(repeat, payload)

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match=f"repeat_1 service-bucket guard failed: {expected_check}",
    ):
        service_decision.build_compact_torch_service_bucket_decision_v1(
            run_id="unit-service-bucket-repeat-initial-regression",
            canonical_floor_bundle_path=canonical,
            repeat_floor_bundle_paths=[repeat],
            min_repeat_count=2,
        )


def test_speed_row_floor_bundle_rejects_source_non_claim_flip(tmp_path):
    inputs = _write_floor_inputs(
        tmp_path,
        compact_source_non_claim_overrides={"automatic_promotion_allowed": True},
    )

    with pytest.raises(
        floor.CompactSpeedRowFloorBundleError,
        match="non_claims.automatic_promotion_allowed must be false",
    ):
        floor.build_compact_speed_row_floor_bundle_v1(
            run_id="unit-floor-bundle-source-non-claim-flip",
            accepted_speed_row_report_path=inputs["accepted"],
            compact_torch_sibling_report_path=inputs["compact_torch"],
            fixed_floor_sibling_report_path=inputs["fixed_floor"],
        )


def test_speed_row_floor_bundle_rejects_payload_non_claim_flip(tmp_path):
    inputs = _write_floor_inputs(tmp_path)
    payload = floor.build_compact_speed_row_floor_bundle_v1(
        run_id="unit-floor-bundle-payload-non-claim-flip",
        accepted_speed_row_report_path=inputs["accepted"],
        compact_torch_sibling_report_path=inputs["compact_torch"],
        fixed_floor_sibling_report_path=inputs["fixed_floor"],
    )
    payload["non_claims"]["training_speedup_claim"] = True

    with pytest.raises(
        floor.CompactSpeedRowFloorBundleError,
        match="non_claims.training_speedup_claim must be false",
    ):
        floor.validate_compact_speed_row_floor_bundle_v1(payload)


def test_service_bucket_decision_rejects_floor_bundle_non_claim_flip(tmp_path):
    inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        inputs,
        run_id="unit-service-bucket-canonical-non-claim-flip",
    )
    bundle = _read_json(canonical)
    bundle["attached_claims"]["training_speedup_claim"] = True
    _write_json(canonical, bundle)

    with pytest.raises(
        floor.CompactSpeedRowFloorBundleError,
        match="attached_claims.training_speedup_claim must be false",
    ):
        service_decision.build_compact_torch_service_bucket_decision_v1(
            run_id="unit-service-bucket-floor-non-claim-flip",
            canonical_floor_bundle_path=canonical,
        )


def test_service_bucket_decision_rejects_payload_non_claim_flip(tmp_path):
    canonical_inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    repeat_inputs = _write_floor_inputs(
        tmp_path / "repeat-inputs",
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_initial_cuda_event_sec=1.2,
        compact_recurrent_cuda_event_sec=0.2,
        compact_tree_cuda_event_sec=0.3,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        canonical_inputs,
        run_id="unit-service-bucket-canonical-clean",
    )
    repeat = _write_floor_bundle_report(
        tmp_path / "repeat-report",
        repeat_inputs,
        run_id="unit-service-bucket-repeat-clean",
    )
    payload = service_decision.build_compact_torch_service_bucket_decision_v1(
        run_id="unit-service-bucket-payload-non-claim-flip",
        canonical_floor_bundle_path=canonical,
        repeat_floor_bundle_paths=[repeat],
        min_repeat_count=2,
    )
    payload["attached_claims"]["training_speedup_claim"] = True

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match="attached_claims.training_speedup_claim must be false",
    ):
        service_decision.validate_compact_torch_service_bucket_decision_v1(payload)


def test_service_bucket_decision_selects_replay_gate_when_replay_commit_dominates(
    tmp_path,
):
    canonical_inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_initial_cuda_event_sec=1.5,
        compact_recurrent_cuda_event_sec=0.3,
        compact_tree_cuda_event_sec=0.36,
    )
    repeat_inputs = _write_floor_inputs(
        tmp_path / "repeat-inputs",
        compact_search_dispatch_wall_sec=3.4204,
        floor_search_dispatch_wall_sec=0.0,
        compact_slab_commit_previous_sec=4.2,
        compact_slab_replay_index_rows_build_sec=4.1,
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_initial_cuda_event_sec=1.5,
        compact_recurrent_cuda_event_sec=0.3,
        compact_tree_cuda_event_sec=0.36,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        canonical_inputs,
        run_id="unit-service-bucket-replay-canonical",
    )
    repeat = _write_floor_bundle_report(
        tmp_path / "repeat-report",
        repeat_inputs,
        run_id="unit-service-bucket-replay-repeat",
    )

    payload = service_decision.build_compact_torch_service_bucket_decision_v1(
        run_id="unit-service-bucket-replay-decision",
        canonical_floor_bundle_path=canonical,
        repeat_floor_bundle_paths=[repeat],
        min_repeat_count=2,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["decision"] == service_decision.DECISION_SELECT_REPLAY_INDEX_OR_SAMPLE
    assert payload["selected_next_target"] == service_decision.TARGET_REPLAY_INDEX_OR_SAMPLE
    summary = payload["bucket_summary"]
    assert summary["source_role"] == "repeat_1"
    assert summary["dispatch_residual_delta_sec"] == pytest.approx(0.0)
    assert summary["service_action_residual_sec"] == pytest.approx(0.2)
    assert summary["service_action_readback_sec"] == pytest.approx(0.08)
    assert summary["service_core_residual_sec"] == pytest.approx(0.07)
    assert summary["tree_policy_build_sec"] == pytest.approx(0.5)
    assert summary["tree_root_prior_build_sec"] == pytest.approx(0.29)
    assert summary["replay_or_commit_delta_sec"] > summary["search_delta_sec"]
    assert summary["replay_or_sample_dominant"] is True
    service_decision.validate_compact_torch_service_bucket_decision_v1(payload)


def test_service_bucket_decision_needs_more_measurement_without_repeat(tmp_path):
    canonical_inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        canonical_inputs,
        run_id="unit-service-bucket-canonical",
    )

    payload = service_decision.build_compact_torch_service_bucket_decision_v1(
        run_id="unit-service-bucket-decision",
        canonical_floor_bundle_path=canonical,
        min_repeat_count=2,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["decision"] == service_decision.DECISION_NEEDS_MORE_MEASUREMENT
    assert payload["selected_next_target"] == service_decision.TARGET_NONE
    assert payload["attached_claims"]["next_optimization_code_allowed"] is False
    assert payload["guard_checks"]["repeat_requirement_met"] is False
    service_decision.validate_compact_torch_service_bucket_decision_v1(payload)


def test_service_bucket_decision_rejects_missing_opt073_telemetry(tmp_path):
    inputs = _write_floor_inputs(
        tmp_path / "inputs",
        compact_fast_path_count=179.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        inputs,
        run_id="unit-service-bucket-canonical",
    )

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match="canonical service-bucket guard failed",
    ):
        service_decision.build_compact_torch_service_bucket_decision_v1(
            run_id="unit-service-bucket-decision",
            canonical_floor_bundle_path=canonical,
        )


def test_service_bucket_decision_rejects_missing_inference_guard_telemetry(tmp_path):
    inputs = _write_floor_inputs(
        tmp_path / "inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_inference_guard_present=False,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        inputs,
        run_id="unit-service-bucket-canonical",
    )

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match="model_eval_applied_for_inference",
    ):
        service_decision.build_compact_torch_service_bucket_decision_v1(
            run_id="unit-service-bucket-decision",
            canonical_floor_bundle_path=canonical,
        )


def test_service_bucket_decision_rejects_source_hash_drift(tmp_path):
    canonical_inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    repeat_inputs = _write_floor_inputs(
        tmp_path / "repeat-inputs",
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_initial_cuda_event_sec=1.2,
        compact_recurrent_cuda_event_sec=0.2,
        compact_tree_cuda_event_sec=0.3,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        canonical_inputs,
        run_id="unit-service-bucket-canonical",
    )
    repeat = _write_floor_bundle_report(
        tmp_path / "repeat-report",
        repeat_inputs,
        run_id="unit-service-bucket-repeat",
    )
    payload = service_decision.build_compact_torch_service_bucket_decision_v1(
        run_id="unit-service-bucket-decision",
        canonical_floor_bundle_path=canonical,
        repeat_floor_bundle_paths=[repeat],
    )
    bundle = _read_json(canonical)
    bundle["status"] = "tampered"
    _write_json(canonical, bundle)

    with pytest.raises(
        service_decision.CompactTorchServiceBucketDecisionError,
        match="input_refs.canonical_floor_bundle sha mismatch",
    ):
        service_decision.validate_compact_torch_service_bucket_decision_v1(payload)


def test_service_bucket_decision_cli_writes_report_and_rejects_stale_output(tmp_path):
    module = _load_service_bucket_decision_cli_module()
    canonical_inputs = _write_floor_inputs(
        tmp_path / "canonical-inputs",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
    )
    repeat_inputs = _write_floor_inputs(
        tmp_path / "repeat-inputs",
        compact_timing_mode="host_phase_sync_cuda_event",
        compact_fast_path_count=180.0,
        compact_recurrent_calls=180.0,
        compact_action_d2h_bytes=737280.0,
        compact_initial_cuda_event_sec=1.2,
        compact_recurrent_cuda_event_sec=0.2,
        compact_tree_cuda_event_sec=0.3,
    )
    canonical = _write_floor_bundle_report(
        tmp_path / "canonical-report",
        canonical_inputs,
        run_id="unit-service-bucket-canonical",
    )
    repeat = _write_floor_bundle_report(
        tmp_path / "repeat-report",
        repeat_inputs,
        run_id="unit-service-bucket-repeat",
    )
    output_root = tmp_path / "out"
    run_id = "unit-service-bucket-decision-cli"
    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--canonical-floor-bundle",
        str(canonical),
        "--repeat-floor-bundle",
        str(repeat),
        "--no-compile-decision-report",
    ]

    assert module.main(argv) == 0
    report_path = (
        output_root
        / run_id
        / "compact_torch_service_bucket_decision_report.json"
    )
    manifest_path = output_root / run_id / "manifest.json"
    payload = _read_json(report_path)
    manifest = _read_json(manifest_path)
    assert payload["decision"] == service_decision.DECISION_SELECT_INITIAL_MODEL_FORWARD
    assert manifest["decision"] == service_decision.DECISION_SELECT_INITIAL_MODEL_FORWARD
    service_decision.validate_compact_torch_service_bucket_decision_v1(payload)

    with pytest.raises(FileExistsError):
        module.main(argv)


def test_speed_row_floor_bundle_rejects_wrong_real_search_role(tmp_path):
    inputs = _write_floor_inputs(
        tmp_path,
        compact_kind="device_target",
        compact_impl="compact_coach_speed_row_device_target_search",
    )

    with pytest.raises(
        floor.CompactSpeedRowFloorBundleError,
        match="compact_torch_search_service_sibling search_impl_kind",
    ):
        floor.build_compact_speed_row_floor_bundle_v1(
            run_id="unit-floor-bundle",
            accepted_speed_row_report_path=inputs["accepted"],
            compact_torch_sibling_report_path=inputs["compact_torch"],
            fixed_floor_sibling_report_path=inputs["fixed_floor"],
        )


def test_speed_row_floor_bundle_rejects_input_hash_drift(tmp_path):
    inputs = _write_floor_inputs(tmp_path)
    payload = floor.build_compact_speed_row_floor_bundle_v1(
        run_id="unit-floor-bundle",
        accepted_speed_row_report_path=inputs["accepted"],
        compact_torch_sibling_report_path=inputs["compact_torch"],
        fixed_floor_sibling_report_path=inputs["fixed_floor"],
    )
    report_path = Path(
        payload["rows"]["accepted_coach_speed_row"]["report_ref"]["path"]
    )
    report = _read_json(report_path)
    report["steps_per_sec"] = 123.0
    _write_json(report_path, report)

    with pytest.raises(
        floor.CompactSpeedRowFloorBundleError,
        match="accepted_coach_speed_row.report_ref sha256 mismatch",
    ):
        floor.validate_compact_speed_row_floor_bundle_v1(payload)


def test_speed_row_floor_bundle_cli_writes_report_and_rejects_stale_output(tmp_path):
    module = _load_cli_module()
    inputs = _write_floor_inputs(tmp_path / "inputs")
    output_root = tmp_path / "out"
    run_id = "unit-floor-bundle-cli"
    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--accepted-speed-row-report",
        str(inputs["accepted"]),
        "--compact-torch-sibling-report",
        str(inputs["compact_torch"]),
        "--fixed-floor-sibling-report",
        str(inputs["fixed_floor"]),
    ]

    assert module.main(argv) == 0
    report_path = output_root / run_id / "compact_speed_row_floor_bundle_report.json"
    payload = _read_json(report_path)
    assert payload["ok"] is True
    assert payload["status"] == floor.STATUS_COMPLETE
    floor.validate_compact_speed_row_floor_bundle_v1(payload)

    with pytest.raises(FileExistsError):
        module.main(argv)


def _write_floor_bundle_report(
    directory: Path,
    inputs: dict[str, Path],
    *,
    run_id: str,
) -> Path:
    payload = floor.build_compact_speed_row_floor_bundle_v1(
        run_id=run_id,
        accepted_speed_row_report_path=inputs["accepted"],
        compact_torch_sibling_report_path=inputs["compact_torch"],
        fixed_floor_sibling_report_path=inputs["fixed_floor"],
        created_at="2026-05-31T00:00:00+00:00",
    )
    report_path = directory / "compact_speed_row_floor_bundle_report.json"
    _write_json(report_path, payload)
    return report_path


def _write_floor_inputs(
    tmp_path: Path,
    *,
    compact_kind: str = "compact_torch_search_service",
    compact_impl: str = "compact_torch_device_tree_fixed_shape_v0",
    compact_search_dispatch_wall_sec: float = 0.0,
    floor_search_dispatch_wall_sec: float = 0.0,
    compact_slab_commit_previous_sec: float = 0.0,
    floor_slab_commit_previous_sec: float = 0.0,
    compact_slab_replay_index_rows_build_sec: float = 0.0,
    floor_slab_replay_index_rows_build_sec: float = 0.0,
    compact_timing_mode: str = "host_phase_sync",
    compact_fast_path_count: float = 1.0,
    compact_recurrent_calls: float = 1.0,
    compact_action_d2h_bytes: float = 4096.0,
    compact_replay_payload_d2h_bytes: float = 0.0,
    compact_committed_replay_payload_d2h_bytes: float = 0.0,
    compact_resident_observation_host_fallback_count: float = 0.0,
    compact_initial_cuda_event_sec: float = 0.21,
    compact_initial_representation_cuda_event_sec: float = 0.0,
    compact_initial_prediction_cuda_event_sec: float = 0.0,
    compact_initial_direct_core_cuda_event_sec: float = 0.0,
    compact_initial_direct_core_cuda_event_residual_sec: float = 0.0,
    compact_initial_direct_requested_count: float = 0.0,
    compact_initial_direct_used_count: float = 0.0,
    compact_initial_inference_mode_requested: str = "model_method",
    compact_initial_inference_mode_effective: str = "model_method",
    compact_initial_inference_runtime_status: str = "model_method_used",
    compact_recurrent_cuda_event_sec: float = 0.41,
    compact_tree_cuda_event_sec: float = 0.61,
    compact_inference_guard_present: bool = True,
    compact_omit_timing_fields: set[str] | None = None,
    compact_source_non_claim_overrides: dict[str, bool] | None = None,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return {
        "accepted": _write_speed_row(
            tmp_path / "accepted-h100",
            run_id="unit-accepted-h100",
            search_kind="device_target",
            search_impl="compact_coach_speed_row_device_target_search",
            wall_sec=10.0,
            search_service_total_sec=0.0,
            compact_rollout_slab_sec=1.0,
            actor_step_wall_sec=4.0,
        ),
        "compact_torch": _write_speed_row(
            tmp_path / "compact-torch-h100",
            run_id="unit-compact-torch-h100",
            search_kind=compact_kind,
            search_impl=compact_impl,
            wall_sec=12.0,
            search_service_total_sec=3.5,
            compact_rollout_slab_sec=8.5,
            actor_step_wall_sec=2.0,
            search_dispatch_wall_sec=compact_search_dispatch_wall_sec,
            slab_commit_previous_sec=compact_slab_commit_previous_sec,
            slab_replay_index_rows_build_sec=compact_slab_replay_index_rows_build_sec,
            timing_mode=compact_timing_mode,
            fast_path_count=compact_fast_path_count,
            recurrent_calls=compact_recurrent_calls,
            action_d2h_bytes=compact_action_d2h_bytes,
            replay_payload_d2h_bytes=compact_replay_payload_d2h_bytes,
            committed_replay_payload_d2h_bytes=(
                compact_committed_replay_payload_d2h_bytes
            ),
            resident_observation_host_fallback_count=(
                compact_resident_observation_host_fallback_count
            ),
            initial_cuda_event_sec=compact_initial_cuda_event_sec,
            initial_representation_cuda_event_sec=(
                compact_initial_representation_cuda_event_sec
            ),
            initial_prediction_cuda_event_sec=(
                compact_initial_prediction_cuda_event_sec
            ),
            initial_direct_core_cuda_event_sec=(
                compact_initial_direct_core_cuda_event_sec
            ),
            initial_direct_core_cuda_event_residual_sec=(
                compact_initial_direct_core_cuda_event_residual_sec
            ),
            initial_direct_requested_count=compact_initial_direct_requested_count,
            initial_direct_used_count=compact_initial_direct_used_count,
            initial_inference_mode_requested=(
                compact_initial_inference_mode_requested
            ),
            initial_inference_mode_effective=(
                compact_initial_inference_mode_effective
            ),
            initial_inference_runtime_status=(
                compact_initial_inference_runtime_status
            ),
            recurrent_cuda_event_sec=compact_recurrent_cuda_event_sec,
            tree_cuda_event_sec=compact_tree_cuda_event_sec,
            inference_guard_present=compact_inference_guard_present,
            omit_compact_rollout_timing_fields=compact_omit_timing_fields,
            source_non_claim_overrides=compact_source_non_claim_overrides,
        ),
        "fixed_floor": _write_speed_row(
            tmp_path / "fixed-floor-h100",
            run_id="unit-fixed-floor-h100",
            search_kind="fixed_shape_search_owner",
            search_impl="fixed_shape_batched_search_owner_profile_only_v0",
            wall_sec=8.0,
            search_service_total_sec=0.1,
            compact_rollout_slab_sec=0.6,
            actor_step_wall_sec=2.0,
            search_dispatch_wall_sec=floor_search_dispatch_wall_sec,
            slab_commit_previous_sec=floor_slab_commit_previous_sec,
            slab_replay_index_rows_build_sec=floor_slab_replay_index_rows_build_sec,
        ),
    }


def _write_compile_eager_pair_inputs(
    tmp_path: Path,
    *,
    trajectory_match: bool,
    eager_wall_sec: float = 10.0,
    compile_wall_sec: float = 8.0,
    eager_model_sec: float = 1.0,
    compile_model_sec: float = 0.5,
    inference_guard_present: bool = True,
) -> list[CompileEagerPairInput]:
    pairs: list[CompileEagerPairInput] = []
    for index in range(2):
        eager_action_checksum = 1000 + index
        compile_action_checksum = (
            eager_action_checksum if trajectory_match else eager_action_checksum + 50
        )
        eager_trajectory_checksum = 2000 + index
        compile_trajectory_checksum = (
            eager_trajectory_checksum
            if trajectory_match
            else eager_trajectory_checksum + 50
        )
        eager = _write_speed_row(
            tmp_path / f"pair-{index}-eager",
            run_id=f"unit-pair-{index}-eager",
            search_kind="compact_torch_search_service",
            search_impl="compact_torch_device_tree_fixed_shape_v0",
            wall_sec=eager_wall_sec,
            search_service_total_sec=2.0,
            compact_rollout_slab_sec=3.0,
            actor_step_wall_sec=1.0,
            model_sec=eager_model_sec,
            fast_path_count=180.0,
            recurrent_calls=180.0,
            action_d2h_bytes=737280.0,
            model_compile_requested=False,
            model_compile_used=False,
            model_compile_cache_hit=False,
            model_compile_mode="none",
            model_compile_runtime_status="not_requested",
            env_action_checksum_total=eager_action_checksum,
            env_trajectory_checksum_total=eager_trajectory_checksum,
            inference_guard_present=inference_guard_present,
        )
        compiled = _write_speed_row(
            tmp_path / f"pair-{index}-compile",
            run_id=f"unit-pair-{index}-compile",
            search_kind="compact_torch_search_service",
            search_impl="compact_torch_device_tree_fixed_shape_v0",
            wall_sec=compile_wall_sec,
            search_service_total_sec=1.2,
            compact_rollout_slab_sec=2.0,
            actor_step_wall_sec=1.0,
            model_sec=compile_model_sec,
            fast_path_count=180.0,
            recurrent_calls=180.0,
            action_d2h_bytes=737280.0,
            model_compile_requested=True,
            model_compile_used=True,
            model_compile_cache_hit=True,
            model_compile_mode="default",
            model_compile_runtime_status="cache_hit",
            env_action_checksum_total=compile_action_checksum,
            env_trajectory_checksum_total=compile_trajectory_checksum,
            inference_guard_present=inference_guard_present,
        )
        pairs.append(
            CompileEagerPairInput(
                pair_id=f"r{index + 1}",
                eager_report_path=eager,
                compile_report_path=compiled,
            )
        )
    return pairs


def _write_speed_row(
    directory: Path,
    *,
    run_id: str,
    search_kind: str,
    search_impl: str,
    wall_sec: float,
    search_service_total_sec: float,
    compact_rollout_slab_sec: float,
    actor_step_wall_sec: float,
    search_dispatch_wall_sec: float = 0.0,
    slab_commit_previous_sec: float = 0.0,
    slab_replay_index_rows_build_sec: float = 0.0,
    timing_mode: str = "host_phase_sync",
    fast_path_count: float = 1.0,
    recurrent_calls: float = 1.0,
    action_d2h_bytes: float = 0.0,
    replay_payload_d2h_bytes: float = 0.0,
    committed_replay_payload_d2h_bytes: float = 0.0,
    resident_observation_host_fallback_count: float = 0.0,
    action_override_drop_count: float = 0.0,
    model_sec: float = 0.5,
    model_compile_requested: bool = False,
    model_compile_used: bool = False,
    model_compile_cache_hit: bool = False,
    model_compile_mode: str = "none",
    model_compile_runtime_status: str = "not_requested",
    env_action_checksum_total: int = 11,
    env_trajectory_checksum_total: int = 22,
    inference_guard_present: bool = True,
    initial_cuda_event_sec: float = 0.21,
    initial_representation_cuda_event_sec: float = 0.0,
    initial_prediction_cuda_event_sec: float = 0.0,
    initial_direct_core_cuda_event_sec: float = 0.0,
    initial_direct_core_cuda_event_residual_sec: float = 0.0,
    initial_direct_requested_count: float = 0.0,
    initial_direct_used_count: float = 0.0,
    initial_inference_mode_requested: str = "model_method",
    initial_inference_mode_effective: str = "model_method",
    initial_inference_runtime_status: str = "model_method_used",
    recurrent_cuda_event_sec: float = 0.41,
    tree_cuda_event_sec: float = 0.61,
    omit_compact_rollout_timing_fields: set[str] | None = None,
    source_non_claim_overrides: dict[str, bool] | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    candidate = "unit-compact-ckpt"
    lifecycle_path = directory / "unified_lifecycle_report.json"
    manifest_path = directory / "manifest.json"
    result_path = directory / "row_001_result.json"
    loaded_identity = _loaded_identity(candidate)
    source_non_claims = _non_claims(source_non_claim_overrides)
    _write_json(
        lifecycle_path,
        {
            "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
            "ok": True,
            "checkpoint_id": candidate,
            "lifecycle_gates_complete": True,
            "missing_required_gates": ["coach_speed_row"],
            "promotion_eligible": False,
            "current_chain_identity": dict(loaded_identity),
        },
    )
    row = {
        "schema_id": "curvyzero_compact_coach_speed_row_manifest_row/v1",
        "row_id": "001",
        "candidate_checkpoint_id": candidate,
        "route": "compact_owned_trainer",
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "row_purpose": "coach_speed_row",
        "speed_currency": "compact_trainer_env_steps_per_sec",
        "promotion_claim": False,
        "batch_size": 1024,
        "actor_count": 16,
        "steps": 180,
        "warmup_steps": 45,
        "sample_batch_size": 512,
        "sample_interval": 8,
        "replay_pair_capacity": 4096,
        "learner_train_steps": 1,
        "learner_device": "cuda",
        "num_simulations": 1,
        "search_service_kind": search_kind,
        "search_service_impl": search_impl,
        "non_claims": source_non_claims,
        "command": ["unit", "speed-row"],
    }
    _write_json(
        manifest_path,
        {
            "schema_id": "curvyzero_compact_coach_speed_row_manifest/v1",
            "experiment_id": run_id,
            "candidate_checkpoint_id": candidate,
            "route": "compact_owned_trainer",
            "profile_only": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "non_claims": source_non_claims,
            "rows": [row],
        },
    )
    env_steps = 184320.0
    action_step_build_sec = search_service_total_sec * 0.001
    metadata_build_sec = search_service_total_sec * 0.002
    pending_store_sec = search_service_total_sec * 0.003
    action_postprocess_sec = (
        action_step_build_sec + metadata_build_sec + pending_store_sec
    )
    action_wall_sec = (
        search_service_total_sec + action_postprocess_sec + 0.2
        if search_service_total_sec
        else 0.0
    )
    is_compact_torch_service = search_kind == "compact_torch_search_service"
    action_accounted_sec = search_service_total_sec + action_postprocess_sec
    action_residual_sec = action_wall_sec - action_accounted_sec
    initial_output_decode_sec = 0.17 if is_compact_torch_service else 0.0
    tree_root_prior_build_sec = 0.29 if is_compact_torch_service else 0.0
    tree_accounted_sec = (
        0.3 + 0.35 + 0.4 + 0.45 + 0.5 + 0.6
        if is_compact_torch_service
        else 0.0
    )
    tree_total_sec = tree_accounted_sec + (0.05 if is_compact_torch_service else 0.0)
    tree_residual_sec = tree_total_sec - tree_accounted_sec
    action_readback_sec = 0.08 if is_compact_torch_service else 0.0
    core_residual_sec = 0.07 if is_compact_torch_service else 0.0
    core_accounted_sec = (
        max(search_service_total_sec - core_residual_sec, 0.0)
        if is_compact_torch_service
        else 0.0
    )
    inference_guard_enter_sec = 0.031 if is_compact_torch_service else 0.0
    inference_guard_exit_sec = 0.032 if is_compact_torch_service else 0.0
    inference_guard_total_sec = inference_guard_enter_sec + inference_guard_exit_sec
    profile_telemetry = {
        "compact_policy_refresh_model_state_digest": "unit-model-digest",
        "compact_torch_search_service_timing_mode": timing_mode,
        "compact_torch_search_model_compile_requested": (
            model_compile_requested
        ),
        "compact_torch_search_model_compile_used": model_compile_used,
        "compact_torch_search_model_compile_cache_hit": (
            model_compile_cache_hit
        ),
        "compact_torch_search_model_compile_mode": model_compile_mode,
        "compact_torch_search_model_compile_runtime_status": (
            model_compile_runtime_status
        ),
        "compact_torch_search_initial_inference_mode_requested": (
            initial_inference_mode_requested
        ),
        "compact_torch_search_initial_inference_mode_effective": (
            initial_inference_mode_effective
        ),
        "compact_torch_search_initial_inference_direct_requested": (
            initial_direct_requested_count > 0.0
        ),
        "compact_torch_search_initial_inference_direct_used": (
            initial_direct_used_count > 0.0
        ),
        "compact_torch_search_initial_inference_runtime_status": (
            initial_inference_runtime_status
        ),
        "compact_torch_search_initial_inference_fallback_count": 0.0,
        "compact_torch_search_compile_requested": False,
        "compact_torch_search_compile_used": False,
        "compact_torch_search_compile_mode": "none",
        "compact_torch_search_compile_runtime_status": "not_requested",
    }
    if inference_guard_present:
        profile_telemetry.update(
            {
                "compact_torch_search_model_training_before_inference": True,
                "compact_torch_search_model_training_after_inference": True,
                "compact_torch_search_model_eval_applied_for_inference": True,
                "compact_torch_search_model_inference_mode_used": True,
                "compact_torch_search_service_inference_guard_enter_sec": (
                    inference_guard_enter_sec
                ),
                "compact_torch_search_service_inference_guard_exit_sec": (
                    inference_guard_exit_sec
                ),
                "compact_torch_search_service_inference_guard_total_sec": (
                    inference_guard_total_sec
                ),
            }
        )
    source_profile = {
        "batch_size": 1024,
        "actor_count": 16,
        "steps": 180,
        "warmup_steps": 45,
        "measured_sec": wall_sec,
        "total_sec": wall_sec,
        "physical_rows_per_sec": env_steps / wall_sec,
        "steps_per_sec": (env_steps * 2.0) / wall_sec,
        "compact_rollout_slab_last_telemetry": {
            "compact_rollout_slab_active_root_count": 2048,
            "compact_rollout_slab_search_impl": search_impl,
            "compact_rollout_slab_num_simulations": 1,
            "compact_rollout_slab_search_service_one_simulation_root_prior_softmax_skipped": (
                search_kind == "compact_torch_search_service"
            ),
            "compact_rollout_slab_search_service_one_simulation_selection_mode": (
                "masked_logits_argmax"
                if search_kind == "compact_torch_search_service"
                else ""
            ),
            "compact_rollout_slab_search_service_timing_mode": timing_mode,
            "compact_rollout_slab_profile_telemetry": profile_telemetry,
        },
        "compact_rollout_slab_action_mode": "search_feedback",
        "compact_rollout_slab_action_override_drop_count": action_override_drop_count,
        "compact_rollout_slab_telemetry_totals": {
            "compact_rollout_slab_internal_accounted_sec": (
                compact_rollout_slab_sec * 10.0
            ),
            "compact_rollout_slab_commit_previous_sec": slab_commit_previous_sec,
            "compact_rollout_slab_replay_index_rows_build_sec": (
                slab_replay_index_rows_build_sec
            ),
            "compact_rollout_slab_search_dispatch_wall_sec": search_dispatch_wall_sec,
            "compact_rollout_slab_search_service_total_sec": search_service_total_sec,
            "compact_rollout_slab_search_service_action_preamble_sec": (
                search_service_total_sec * 0.1
            ),
            "compact_rollout_slab_search_service_fixed_shape_masks_sec": (
                search_service_total_sec * 0.01
            ),
            "compact_rollout_slab_search_service_compile_eligibility_sec": (
                search_service_total_sec * 0.02
            ),
            "compact_rollout_slab_search_service_helper_cache_sec": (
                search_service_total_sec * 0.03
            ),
            "compact_rollout_slab_search_service_model_cache_sec": (
                search_service_total_sec * 0.04
            ),
            "compact_rollout_slab_search_service_inference_guard_enter_sec": (
                inference_guard_enter_sec
            ),
            "compact_rollout_slab_search_service_inference_guard_exit_sec": (
                inference_guard_exit_sec
            ),
            "compact_rollout_slab_search_service_inference_guard_total_sec": (
                inference_guard_total_sec
            ),
            "compact_rollout_slab_search_service_metadata_build_sec": (
                metadata_build_sec
            ),
            "compact_rollout_slab_search_service_pending_replay_store_sec": (
                pending_store_sec
            ),
            "compact_rollout_slab_search_service_action_step_build_sec": (
                action_step_build_sec
            ),
            "compact_rollout_slab_search_service_action_postprocess_sec": (
                action_postprocess_sec
            ),
            "compact_rollout_slab_search_service_action_wall_sec": action_wall_sec,
            "compact_rollout_slab_search_service_action_accounted_sec": (
                action_accounted_sec if is_compact_torch_service else 0.0
            ),
            "compact_rollout_slab_search_service_action_residual_sec": (
                action_residual_sec if is_compact_torch_service else 0.0
            ),
            "compact_rollout_slab_search_service_action_unaccounted_sec": (
                0.2 if search_service_total_sec else 0.0
            ),
            "compact_rollout_slab_search_service_action_overaccounted_sec": 0.0,
            "compact_rollout_slab_search_service_tensor_prepare_sync_sec": (
                0.1 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_enqueue_sec": (
                0.15 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_sync_sec": (
                0.2 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_cuda_event_sec": (
                initial_cuda_event_sec
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_representation_sec": (
                0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_prediction_sec": (
                0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_pack_sec": 0.0,
            "compact_rollout_slab_search_service_initial_inference_representation_cuda_event_sec": (
                initial_representation_cuda_event_sec
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_prediction_cuda_event_sec": (
                initial_prediction_cuda_event_sec
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_direct_core_cuda_event_sec": (
                initial_direct_core_cuda_event_sec
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_direct_core_cuda_event_residual_sec": (
                initial_direct_core_cuda_event_residual_sec
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_direct_requested": (
                initial_direct_requested_count
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_direct_used": (
                initial_direct_used_count
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_initial_inference_fallback_count": (
                0.0
            ),
            "compact_rollout_slab_search_service_initial_output_decode_sec": (
                initial_output_decode_sec
            ),
            "compact_rollout_slab_search_service_root_output_decode_sec": (
                initial_output_decode_sec
            ),
            "compact_rollout_slab_search_service_tree_setup_sec": (
                0.0 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_tree_root_prior_select_sec": (
                0.3 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_tree_root_prior_build_sec": (
                tree_root_prior_build_sec
            ),
            "compact_rollout_slab_search_service_tree_select_enqueue_sec": (
                0.0 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_tree_recurrent_action_build_sec": (
                0.35 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_tree_recurrent_inference_enqueue_sec": (
                0.4 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_tree_recurrent_inference_cuda_event_sec": (
                recurrent_cuda_event_sec
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_tree_recurrent_output_decode_sec": (
                0.45 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_tree_backup_enqueue_sec": (
                0.0 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_tree_policy_build_sec": (
                0.5 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_tree_sync_sec": (
                0.6 if search_kind == "compact_torch_search_service" else 0.0
            ),
            "compact_rollout_slab_search_service_tree_cuda_event_sec": (
                tree_cuda_event_sec
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_cuda_event_timing_enabled": (
                fast_path_count
                if search_kind == "compact_torch_search_service"
                and "cuda_event" in timing_mode
                else 0.0
            ),
            "compact_rollout_slab_search_service_initial_sync_enabled": (
                fast_path_count
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_tree_total_sec": tree_total_sec,
            "compact_rollout_slab_search_service_tree_accounted_sec": (
                tree_accounted_sec
            ),
            "compact_rollout_slab_search_service_tree_residual_sec": tree_residual_sec,
            "compact_rollout_slab_search_service_tree_unaccounted_sec": (
                tree_residual_sec if is_compact_torch_service else 0.0
            ),
            "compact_rollout_slab_search_service_tree_overaccounted_sec": 0.0,
            "compact_rollout_slab_search_service_action_readback_sec": (
                action_readback_sec
            ),
            "compact_rollout_slab_search_service_core_accounted_sec": core_accounted_sec,
            "compact_rollout_slab_search_service_core_residual_sec": core_residual_sec,
            "compact_rollout_slab_search_service_core_unaccounted_sec": (
                core_residual_sec
            ),
            "compact_rollout_slab_search_service_core_overaccounted_sec": 0.0,
            "compact_rollout_slab_search_service_one_simulation_fast_path_count": (
                fast_path_count
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_one_simulation_root_prior_softmax_skipped": (
                fast_path_count
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_search_service_recurrent_inference_calls": (
                recurrent_calls
                if search_kind == "compact_torch_search_service"
                else 0.0
            ),
            "compact_rollout_slab_action_d2h_bytes": action_d2h_bytes,
            "compact_rollout_slab_replay_payload_d2h_bytes": (
                replay_payload_d2h_bytes
            ),
            "compact_rollout_slab_committed_replay_payload_d2h_bytes": (
                committed_replay_payload_d2h_bytes
            ),
            "compact_rollout_slab_resident_observation_host_fallback_count": (
                resident_observation_host_fallback_count
            ),
            "compact_rollout_slab_search_sec": search_service_total_sec,
            "compact_rollout_slab_model_sec": model_sec,
            "compact_rollout_slab_h2d_sec": 0.05,
        },
        "compact_rollout_slab_sample_gate_replay_ring_pair_capacity": 4096,
        "compact_rollout_slab_sample_gate_batch_size": 512,
        "compact_rollout_slab_sample_gate_interval": 8,
        "compact_rollout_slab_sample_gate_sec": 0.2,
        "compact_rollout_slab_learner_gate_device": "cuda",
        "compact_rollout_slab_learner_gate_train_steps": 1,
        "compact_rollout_slab_learner_gate_sec": 0.3,
        "timings": {
            "actor_step_wall_sec": actor_step_wall_sec,
            "compact_batch_build_sec": 0.05,
            "compact_payload_pickle_sec": 0.05,
            "compact_rollout_slab_sec": compact_rollout_slab_sec,
            "gather_merge_sec": 0.01,
            "observation_sec": 0.4,
            "actor_step_sec": 0.6,
        },
        "env_action_checksum_total": env_action_checksum_total,
        "env_trajectory_checksum_total": env_trajectory_checksum_total,
        "last_env_action_checksum": 33,
        "last_env_trajectory_checksum": 44,
        "terminal_row_count": 0,
    }
    if omit_compact_rollout_timing_fields:
        totals = source_profile["compact_rollout_slab_telemetry_totals"]
        for key in omit_compact_rollout_timing_fields:
            totals.pop(key, None)
    summary = {
        "profile_only": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "status": "complete",
        "ok": True,
        "row_id": "001",
        "candidate_checkpoint_id": candidate,
        "route": "compact_owned_trainer",
        "row_purpose": "coach_speed_row",
        "promotion_claim": False,
        "speed_currency": "compact_trainer_env_steps_per_sec",
        "search_service_kind": search_kind,
        "search_service_impl": search_impl,
        "env_steps_collected": env_steps,
        "training_wall_sec": wall_sec,
        "steps_per_sec": env_steps / wall_sec,
        "non_claims": source_non_claims,
    }
    _write_json(
        result_path,
        {
            "schema_id": "curvyzero_compact_coach_speed_row_result/v1",
            "ok": True,
            "status": "complete",
            "problem": None,
            "returncode": 0,
            "run_invocation_id": f"{run_id}:unit",
            "candidate_checkpoint_id": candidate,
            "row_id": "001",
            "row": row,
            "producer": {
                "schema_id": "curvyzero_compact_coach_speed_row_producer/v1",
                "producer_id": "unit-speed-row-producer",
                "run_id": run_id,
                "produced_by": "tests/test_compact_speed_row_floor_bundle.py",
            },
            "summary": summary,
            "compact": {
                "ok": True,
                "candidate_checkpoint_id": candidate,
                "route": "compact_owned_trainer",
                "profile_only": False,
                "calls_train_muzero": False,
                "touches_live_runs": False,
                "real_compact_owned_training_work": True,
                "compact_owned_trainer_env_step_source": "unit_fixture",
                "model_identity_scope": "candidate_loaded_checkpoint",
                "loaded_checkpoint_identity": loaded_identity,
                "search_service_kind": search_kind,
                "search_service_impl": search_impl,
                "source_profile_payload": source_profile,
                "non_claims": source_non_claims,
            },
            "non_claims": source_non_claims,
        },
    )
    evidence = build_compact_coach_speed_row_evidence_v1(
        route="compact_owned_trainer",
        candidate_checkpoint_id=candidate,
        unified_lifecycle_report_path=lifecycle_path,
        manifest_path=manifest_path,
        row_id="001",
        result_json_path=result_path,
        speed_currency="compact_trainer_env_steps_per_sec",
        numerator_field="env_steps_collected",
        denominator_field="training_wall_sec",
    )
    evidence_path = Path(f"{result_path}.compact_coach_speed_row.evidence.json")
    _write_json(evidence_path, evidence)
    report_path = directory / "compact_coach_speed_row_modal_report.json"
    _write_json(
        report_path,
        {
            "schema_id": "curvyzero_compact_coach_speed_row_modal_report/v1",
            "ok": True,
            "run_id": run_id,
            "candidate_checkpoint_id": candidate,
            "manifest_path": str(manifest_path),
            "result_path": str(result_path),
            "evidence_path": str(evidence_path),
            "evidence_ref": evidence["evidence_ref"],
            "speed_currency": "compact_trainer_env_steps_per_sec",
            "env_steps_collected": env_steps,
            "training_wall_sec": wall_sec,
            "steps_per_sec": env_steps / wall_sec,
            "search_service_kind": search_kind,
            "search_service_impl": search_impl,
            "model_identity_scope": "candidate_loaded_checkpoint",
            "real_compact_owned_training_work": True,
            "promotion_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
        },
    )
    return report_path


def _write_compile_eager_pair_report(
    tmp_path: Path,
    *,
    run_id: str,
    trajectory_match: bool,
    eager_wall_sec: float,
    compile_wall_sec: float,
) -> Path:
    pairs = _write_compile_eager_pair_inputs(
        tmp_path,
        trajectory_match=trajectory_match,
        eager_wall_sec=eager_wall_sec,
        compile_wall_sec=compile_wall_sec,
    )
    payload = build_compact_compile_eager_speed_pair_v1(
        run_id=run_id,
        pairs=pairs,
        created_at="2026-05-31T00:00:00+00:00",
    )
    report_path = tmp_path / "compile_eager_speed_pair_report.json"
    _write_json(report_path, payload)
    return report_path


def _write_fixed_root_tape_result(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "row_001_result.json"
    summary = {
        "profile_only": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "compact_root_tape_compare_enabled": True,
        "compact_root_tape_metadata_profile_only": True,
        "compact_root_tape_metadata_calls_train_muzero": False,
        "compact_root_tape_metadata_root_noise_weight": 0.0,
        "compact_root_tape_service_labels": ["model_compile_default", "primary"],
        "compact_root_tape_skipped_record_count": 26,
        "compact_root_tape_backend_primary_run_sec": 0.03,
        "compact_root_tape_backend_model_compile_default_model_compile_requested_count": 4,
        "compact_root_tape_backend_model_compile_default_model_compile_used_count": 4,
        "compact_root_tape_backend_model_compile_default_model_compile_cache_hit_count": 3,
        "compact_root_tape_backend_model_compile_default_model_compile_runtime_status_counts": {
            "compiled": 1,
            "cache_hit": 3,
        },
        "compact_root_tape_backend_model_compile_default_run_sec": 23.4,
        "compact_root_tape_model_compile_default_vs_primary_record_count": 4,
        "compact_root_tape_model_compile_default_vs_primary_active_root_count": 1024,
        "compact_root_tape_model_compile_default_vs_primary_action_match_fraction": 1.0,
        "compact_root_tape_model_compile_default_vs_primary_visit_l1_mean": 0.0,
        "compact_root_tape_model_compile_default_vs_primary_visit_l1_max": 0.0,
        "compact_root_tape_model_compile_default_vs_primary_root_value_abs_diff_mean": 0.0,
        "compact_root_tape_model_compile_default_vs_primary_root_value_abs_diff_max": 0.0,
    }
    _write_json(
        path,
        {
            "schema_id": "curvyzero_hybrid_observation_profile_collected_result/v0",
            "status": "complete",
            "returncode": 0,
            "summary": summary,
        },
    )
    return path


def _loaded_identity(candidate: str) -> dict[str, object]:
    return {
        "scope": "candidate_loaded_checkpoint",
        "identity_source": "unit_fixture",
        "candidate_loaded_checkpoint": True,
        "checkpoint_id": candidate,
        "trainer_id": f"{candidate}:trainer",
        "policy_version_ref": f"{candidate}:policy-update-1",
        "model_version_ref": f"{candidate}:model-update-1",
        "policy_source": "unit_policy_source",
        "learner_update_count": 1,
        "model_state_digest": "a" * 64,
        "compact_checkpoint_sha256": "b" * 64,
    }


def _non_claims(overrides: dict[str, bool] | None = None) -> dict[str, bool]:
    claims = {
        "promotion_claim": False,
        "training_speedup_claim": False,
        "live_run_safety_claim": False,
        "stock_resume_claim": False,
        "rating_or_promotion_quality_claim": False,
        "automatic_promotion_allowed": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    if overrides:
        claims.update(overrides)
    return claims


def _load_cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / (
        "build_compact_speed_row_floor_bundle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_speed_row_floor_bundle_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model_compile_decision_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_model_compile_decision.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_model_compile_decision_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_service_bucket_decision_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_torch_service_bucket_decision.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_torch_service_bucket_decision_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
