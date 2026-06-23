from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from curvyzero.training import compact_matched_quality_study_plan as plan


def test_larger_study_plan_builds_fresh_runnable_commands(monkeypatch, tmp_path):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_bundle_review(tmp_path)

    payload = plan.build_compact_matched_quality_larger_study_plan_v1(
        run_id="unit-larger-study-plan",
        bundle_review_report_path=bundle_path,
        created_at="2026-05-31T00:00:00+00:00",
        study_run_stamp="20260531",
    )

    assert payload["schema_id"] == plan.COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_SCHEMA_ID
    assert payload["status"] == plan.COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_STATUS
    assert payload["study_run_stamp"] == "20260531"
    assert payload["decision"]["larger_matched_quality_study_required"] is True
    assert payload["decision"]["promotion_claim"] is False
    assert payload["study_requirements"]["min_eval_seed_count"] == 32
    assert payload["study_requirements"]["min_eval_max_steps"] == 2048
    stock_requirement = payload["study_requirements"]["stock_reference"][
        "stock_evaluator_requirement"
    ]
    assert stock_requirement == {
        "policy": "evalenv_full_episode_surface",
        "eval_seed_count": 32,
        "evaluator_env_num": 32,
        "n_evaluator_episode": 32,
        "env_num_matches_episode_count_required": True,
        "empty_ready_set_workaround_formalized": True,
        "canonical_stock_evaluator_required": False,
    }
    assert payload["non_claims"]["promotion_claim"] is False

    planned = payload["planned_runs"]
    for run in planned.values():
        assert "20260531" in run["run_id"]
    assert planned["stock_reference_capture"]["run_id"].endswith("-evalenv32")
    assert (
        planned["matched_learning_quality_canary"]["run_id"]
        == "optimizer-compact-matched-learning-quality-larger-2048x32-env64train8-20260531-canary"
    )
    assert (
        planned["matched_pair_verification"]["run_id"]
        == "optimizer-compact-matched-learning-quality-larger-2048x32-env64train8-20260531-pair-verifier"
    )
    stock_argv = planned["stock_reference_capture"]["argv"]
    compact_argv = planned["compact_candidate_capture"]["argv"]
    assert _argv_value(stock_argv, "--eval-seed-count") == "32"
    assert _argv_value(stock_argv, "--eval-steps") == "2048"
    assert _argv_value(stock_argv, "--evaluator-env-num") == "32"
    assert _argv_value(stock_argv, "--n-evaluator-episode") == "32"
    assert _argv_value(stock_argv, "--max-env-step") == "2048"
    assert _argv_value(compact_argv, "--compact-training-mode") == "env_search_replay"
    assert _argv_value(compact_argv, "--compact-env-steps") == "64"
    bundle_argv = planned["refreshed_bundle_review"]["argv"]
    assert "--stock-resume-load-canary-report" in bundle_argv
    assert "--isolated-live-run-safety-canary-report" in bundle_argv
    assert "--sandbox-assignment-rating-proof-report" in bundle_argv
    assert "--longer-horizon-learning-metrics-report" in bundle_argv
    plan.validate_compact_matched_quality_larger_study_plan_v1(payload)


def test_larger_study_plan_rejects_not_larger_than_current(monkeypatch, tmp_path):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_bundle_review(tmp_path)

    with pytest.raises(
        plan.CompactMatchedQualityStudyPlanError,
        match="eval seed count must exceed current evidence",
    ):
        plan.build_compact_matched_quality_larger_study_plan_v1(
            run_id="unit-larger-study-plan",
            bundle_review_report_path=bundle_path,
            min_eval_seed_count=8,
        )


def test_larger_study_plan_rejects_source_bundle_hash_drift(monkeypatch, tmp_path):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_bundle_review(tmp_path)
    payload = plan.build_compact_matched_quality_larger_study_plan_v1(
        run_id="unit-larger-study-plan",
        bundle_review_report_path=bundle_path,
    )
    bundle = _read_json(bundle_path)
    bundle["candidate_checkpoint_id"] = "tampered"
    _write_json(bundle_path, bundle)

    with pytest.raises(
        plan.CompactMatchedQualityStudyPlanError,
        match="source bundle sha256 mismatch",
    ):
        plan.validate_compact_matched_quality_larger_study_plan_v1(payload)


def test_larger_study_plan_rejects_missing_stock_evaluator_surface(
    monkeypatch,
    tmp_path,
):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_bundle_review(tmp_path)
    payload = plan.build_compact_matched_quality_larger_study_plan_v1(
        run_id="unit-larger-study-plan",
        bundle_review_report_path=bundle_path,
    )
    argv = payload["planned_runs"]["stock_reference_capture"]["argv"]
    index = argv.index("--evaluator-env-num")
    del argv[index : index + 2]

    with pytest.raises(
        plan.CompactMatchedQualityStudyPlanError,
        match="argv missing --evaluator-env-num",
    ):
        plan.validate_compact_matched_quality_larger_study_plan_v1(payload)


def test_larger_study_plan_cli_writes_report_and_manifest(monkeypatch, tmp_path):
    _patch_bundle_validator(monkeypatch)
    module = _load_cli_module()
    bundle_path = _write_bundle_review(tmp_path / "inputs")
    output_root = tmp_path / "out"
    run_id = "unit-larger-study-plan-cli"
    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--bundle-review-report",
        str(bundle_path),
        "--study-run-stamp",
        "20260531",
    ]

    assert module.main(argv) == 0
    report_path = output_root / run_id / "larger_study_plan_report.json"
    manifest_path = output_root / run_id / "manifest.json"
    payload = _read_json(report_path)
    manifest = _read_json(manifest_path)
    assert payload["study_run_stamp"] == "20260531"
    assert manifest["schema_id"] == module.MANIFEST_SCHEMA_ID
    assert manifest["required_output_artifacts"] == payload["required_output_artifacts"]
    plan.validate_compact_matched_quality_larger_study_plan_v1(payload)

    with pytest.raises(FileExistsError):
        module.main(argv)


def _patch_bundle_validator(monkeypatch) -> None:
    monkeypatch.setattr(
        plan,
        "validate_compact_promotion_readiness_bundle_review_v1",
        lambda payload: None,
    )


def _write_bundle_review(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "readiness_bundle_review_report.json"
    candidate = "unit-compact-ckpt"
    bundle = {
        "schema_id": plan.COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID,
        "ok": True,
        "status": "local_bundle_reviewed_no_promotion",
        "candidate_checkpoint_id": candidate,
        "evidence_ref": "unit-bundle-ref",
        "input_reports": {
            "compatibility_refresh": {
                "path": str(tmp_path / "compatibility_report.json"),
                "sha256": "0" * 64,
            },
            "unified_lifecycle": {
                "path": str(tmp_path / "unified_lifecycle_report.json"),
                "sha256": "1" * 64,
            },
            "stock_resume_load_canary": {
                "path": str(tmp_path / "stock_resume_load_canary_report.json"),
                "sha256": "2" * 64,
            },
            "isolated_live_run_safety_canary": {
                "path": str(tmp_path / "isolated_live_run_safety_canary_report.json"),
                "sha256": "3" * 64,
            },
            "sandbox_assignment_rating_proof": {
                "path": str(tmp_path / "sandbox_assignment_rating_proof_report.json"),
                "sha256": "4" * 64,
            },
            "longer_horizon_compact_learning_metrics": {
                "path": str(tmp_path / "longer_horizon_learning_metrics_report.json"),
                "sha256": "5" * 64,
            },
        },
        "review_decision": {
            "ready_for_manual_review": True,
            "promotion_claim": False,
            "automatic_promotion_allowed": False,
            "matched_quality_sufficiency_decision": (
                "canary_scale_manual_acceptance_or_larger_study_required_before_promotion"
            ),
        },
        "quality_strength_review": {
            "current_matched_canary_sufficient_for_promotion_claim": False,
            "manual_acceptance_or_larger_study_required_before_promotion": True,
            "matched_pair_summary": {
                "denominator_id": "matched_learning_quality_current_1024x8_denominator_v1",
                "quality_horizon": "matched_learning_quality_pre_post_eval_1024x8_current",
                "hardware_class": "mixed",
                "stock_hardware_class": "modal-gpu-l4-t4-cpu40",
                "compact_hardware_class": "local-cpu-producer-smoke",
                "eval_seed_count": 8,
                "eval_max_steps": 1024,
                "stock_calls_train_muzero": True,
                "compact_calls_train_muzero": False,
            },
            "longer_horizon_summary": {
                "checkpoint_count": 3,
                "learner_update_count_delta": 2,
            },
        },
    }
    _write_json(path, bundle)
    return path


def _argv_value(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


def _load_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_matched_quality_larger_study_plan.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_quality_larger_study_plan_for_test",
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
