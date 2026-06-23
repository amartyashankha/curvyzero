from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from curvyzero.training import compact_matched_quality_sufficiency_review as review


def test_sufficiency_review_requires_larger_same_surface_study(monkeypatch, tmp_path):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_inputs(tmp_path)

    payload = review.build_compact_matched_quality_sufficiency_review_v1(
        run_id="unit-sufficiency-review",
        readiness_bundle_review_path=bundle_path,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["schema_id"] == review.COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_SCHEMA_ID
    assert payload["status"] == review.STATUS_LARGER_SAME_SURFACE_STUDY_REQUIRED
    assert payload["decision"]["promotion_claim"] is False
    assert payload["decision"]["larger_same_surface_study_required"] is True
    assert payload["decision"]["current_evidence_sufficient_for_promotion"] is False
    assert payload["larger_study_plan"]["same_surface_required"] is True
    assert payload["larger_study_plan"]["study_id"].endswith("-20260531")
    assert (
        payload["larger_study_plan"]["minimum_scale_over_current"][
            "min_eval_seed_count"
        ]
        > payload["current_evidence_summary"]["eval_seed_count"]
    )
    plan = payload["larger_study_plan"]
    assert plan["stock_evaluator_requirement"] == {
        "policy": "evalenv_full_episode_surface",
        "eval_seed_count": 32,
        "evaluator_env_num": 32,
        "n_evaluator_episode": 32,
        "env_num_matches_episode_count_required": True,
        "empty_ready_set_workaround_formalized": True,
        "canonical_stock_evaluator_required": False,
    }
    assert (
        plan["planned_runs"]["stock_reference_capture_producer"]["run_id"]
        == "optimizer-stock-reference-quality-producer-larger-2048x32-20260531-evalenv32"
    )
    stock_argv = plan["planned_runs"]["stock_reference_capture_producer"]["argv"]
    assert _argv_value(stock_argv, "--evaluator-env-num") == "32"
    assert _argv_value(stock_argv, "--n-evaluator-episode") == "32"
    compact_argv = plan["planned_runs"]["compact_candidate_capture_producer"]["argv"]
    assert _argv_value(compact_argv, "--candidate-checkpoint-id") == "unit-compact-ckpt"
    assert _argv_value(compact_argv, "--compact-training-mode") == "env_search_replay"
    assert (
        _argv_value(compact_argv, "--denominator-id")
        == "matched_quality_larger_2048x32-env64train8_denominator_v1"
    )
    bundle_argv = plan["planned_runs"]["readiness_bundle_refresh"]["argv"]
    assert "--compatibility-report" in bundle_argv
    assert "--unified-lifecycle-report" in bundle_argv
    assert "--stock-resume-load-canary-report" in bundle_argv
    assert "--isolated-live-run-safety-canary-report" in bundle_argv
    assert "--sandbox-assignment-rating-proof-report" in bundle_argv
    assert "--longer-horizon-learning-metrics-report" in bundle_argv
    review_argv = plan["planned_runs"]["sufficiency_review_update"]["argv"]
    assert _argv_value(review_argv, "--larger-study-id") == plan["study_id"]
    assert _argv_value(review_argv, "--larger-denominator-id") == (
        "matched_quality_larger_2048x32-env64train8_denominator_v1"
    )
    assert payload["non_claims"]["automatic_promotion_allowed"] is False
    review.validate_compact_matched_quality_sufficiency_review_v1(payload)


def test_sufficiency_review_can_accept_current_only_for_named_nonproduction_step(
    monkeypatch,
    tmp_path,
):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_inputs(tmp_path)

    payload = review.build_compact_matched_quality_sufficiency_review_v1(
        run_id="unit-sufficiency-review",
        readiness_bundle_review_path=bundle_path,
        sufficiency_decision=review.DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP,
        next_non_production_step="build local speed-row floor bundle",
    )

    assert (
        payload["status"]
        == review.STATUS_CURRENT_CANARY_ACCEPTED_FOR_NEXT_NON_PRODUCTION_STEP
    )
    assert payload["larger_study_plan"] is None
    assert (
        payload["decision"]["current_evidence_accepted_for_next_non_production_step"]
        is True
    )
    assert payload["decision"]["promotion_claim"] is False
    review.validate_compact_matched_quality_sufficiency_review_v1(payload)


def test_sufficiency_review_rejects_missing_stock_evaluator_surface(
    monkeypatch,
    tmp_path,
):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_inputs(tmp_path)
    payload = review.build_compact_matched_quality_sufficiency_review_v1(
        run_id="unit-sufficiency-review",
        readiness_bundle_review_path=bundle_path,
    )
    stock_argv = payload["larger_study_plan"]["planned_runs"][
        "stock_reference_capture_producer"
    ]["argv"]
    index = stock_argv.index("--evaluator-env-num")
    del stock_argv[index : index + 2]

    with pytest.raises(
        review.CompactMatchedQualitySufficiencyReviewError,
        match="argv missing --evaluator-env-num",
    ):
        review.validate_compact_matched_quality_sufficiency_review_v1(payload)


def test_sufficiency_review_rejects_accept_without_named_nonproduction_step(
    monkeypatch,
    tmp_path,
):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_inputs(tmp_path)

    with pytest.raises(
        review.CompactMatchedQualitySufficiencyReviewError,
        match="named non-production next step",
    ):
        review.build_compact_matched_quality_sufficiency_review_v1(
            run_id="unit-sufficiency-review",
            readiness_bundle_review_path=bundle_path,
            sufficiency_decision=(
                review.DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP
            ),
        )


def test_sufficiency_review_rejects_promotion_claim_flip(monkeypatch, tmp_path):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_inputs(tmp_path)
    payload = review.build_compact_matched_quality_sufficiency_review_v1(
        run_id="unit-sufficiency-review",
        readiness_bundle_review_path=bundle_path,
    )
    payload["decision"]["promotion_claim"] = True

    with pytest.raises(
        review.CompactMatchedQualitySufficiencyReviewError,
        match="promotion_claim must be false",
    ):
        review.validate_compact_matched_quality_sufficiency_review_v1(payload)


def test_sufficiency_review_rejects_not_larger_study(monkeypatch, tmp_path):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_inputs(tmp_path)

    with pytest.raises(
        review.CompactMatchedQualitySufficiencyReviewError,
        match="eval seed count must exceed current evidence",
    ):
        review.build_compact_matched_quality_sufficiency_review_v1(
            run_id="unit-sufficiency-review",
            readiness_bundle_review_path=bundle_path,
            min_eval_seed_count=8,
        )


def test_sufficiency_review_rejects_input_hash_drift(monkeypatch, tmp_path):
    _patch_bundle_validator(monkeypatch)
    bundle_path = _write_inputs(tmp_path)
    payload = review.build_compact_matched_quality_sufficiency_review_v1(
        run_id="unit-sufficiency-review",
        readiness_bundle_review_path=bundle_path,
    )
    matched_path = Path(
        payload["input_reports"]["matched_learning_quality_canary"]["path"]
    )
    _write_json(matched_path, {"ok": True})

    with pytest.raises(
        review.CompactMatchedQualitySufficiencyReviewError,
        match="matched_learning_quality_canary sha256 mismatch",
    ):
        review.validate_compact_matched_quality_sufficiency_review_v1(payload)


def test_sufficiency_review_cli_writes_report_and_rejects_stale_output(
    monkeypatch,
    tmp_path,
):
    _patch_bundle_validator(monkeypatch)
    module = _load_cli_module()
    bundle_path = _write_inputs(tmp_path)
    output_root = tmp_path / "out"
    run_id = "unit-sufficiency-review-main"

    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--readiness-bundle-review",
        str(bundle_path),
    ]

    assert module.main(argv) == 0
    report_path = (
        output_root / run_id / "matched_quality_sufficiency_review_report.json"
    )
    payload = _read_json(report_path)
    assert payload["status"] == review.STATUS_LARGER_SAME_SURFACE_STUDY_REQUIRED
    assert payload["decision"]["promotion_claim"] is False
    review.validate_compact_matched_quality_sufficiency_review_v1(payload)

    with pytest.raises(FileExistsError):
        module.main(argv)


def _patch_bundle_validator(monkeypatch):
    monkeypatch.setattr(
        review,
        "validate_compact_promotion_readiness_bundle_review_v1",
        lambda payload: None,
    )


def _argv_value(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


def _write_inputs(tmp_path: Path) -> Path:
    candidate = "unit-compact-ckpt"
    stock_capture_path = tmp_path / "stock_reference_capture.json"
    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    _write_json(stock_capture_path, {"schema_id": "capture", "role": "stock_reference"})
    _write_json(
        compact_capture_path,
        {"schema_id": "capture", "role": "compact_candidate"},
    )
    matched_path = tmp_path / "matched_learning_quality_canary_report.json"
    pair_path = tmp_path / "matched_pair_verification_report.json"
    longer_path = tmp_path / "longer_horizon_learning_metrics_report.json"
    stock_resume_path = tmp_path / "stock_resume_load_canary_report.json"
    isolated_live_path = tmp_path / "isolated_live_run_safety_canary_report.json"
    sandbox_rating_path = tmp_path / "sandbox_assignment_rating_proof_report.json"
    matched = _matched_report(
        candidate,
        stock_capture_path=stock_capture_path,
        compact_capture_path=compact_capture_path,
    )
    pair = {
        "schema_id": "curvyzero_compact_matched_learning_quality_pair_verification/v1",
        "ok": True,
        "status": "matched_pair_verified",
        "candidate_checkpoint_id": candidate,
        "evidence_ref": "unit-pair-ref",
    }
    longer = {
        "schema_id": "curvyzero_compact_longer_horizon_learning_metrics/v1",
        "ok": True,
        "status": "compact_longer_horizon_learning_metrics_observed",
        "candidate_checkpoint_id": candidate,
        "checkpoint_series": [
            {"candidate_checkpoint_id": candidate},
            {"candidate_checkpoint_id": candidate},
            {"candidate_checkpoint_id": candidate},
        ],
        "cumulative_denominators": {
            "learner_update_count_delta": 2,
            "sample_batch_count_delta": 3,
        },
        "evidence_ref": "unit-longer-ref",
    }
    _write_json(matched_path, matched)
    _write_json(pair_path, pair)
    _write_json(longer_path, longer)
    stock_resume = {
        "schema_id": "curvyzero_compact_promotion_stock_resume_load_canary/v1",
        "ok": True,
        "status": "stock_resume_load_canary_verified",
        "candidate_checkpoint_id": candidate,
        "evidence_ref": "unit-stock-resume-ref",
    }
    isolated_live = {
        "schema_id": "curvyzero_compact_promotion_isolated_live_run_safety_canary/v1",
        "ok": True,
        "status": "isolated_live_run_safety_canary_verified",
        "candidate_checkpoint_id": candidate,
        "evidence_ref": "unit-isolated-live-ref",
    }
    sandbox_rating = {
        "schema_id": "curvyzero_compact_promotion_sandbox_assignment_rating_proof/v1",
        "ok": True,
        "status": "sandbox_assignment_rating_proof_verified",
        "candidate_checkpoint_id": candidate,
        "evidence_ref": "unit-sandbox-rating-ref",
    }
    _write_json(stock_resume_path, stock_resume)
    _write_json(isolated_live_path, isolated_live)
    _write_json(sandbox_rating_path, sandbox_rating)
    bundle_path = tmp_path / "readiness_bundle_review_report.json"
    bundle = {
        "schema_id": review.COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID,
        "ok": True,
        "status": "local_bundle_reviewed_no_promotion",
        "candidate_checkpoint_id": candidate,
        "evidence_ref": "unit-bundle-ref",
        "input_reports": {
            "compatibility_refresh": {
                "path": str(tmp_path / "compatibility.json"),
                "sha256": "0" * 64,
            },
            "unified_lifecycle": {
                "path": str(tmp_path / "lifecycle.json"),
                "sha256": "1" * 64,
            },
            "matched_learning_quality_canary": _report_ref(matched_path, matched),
            "matched_pair_verification": _report_ref(pair_path, pair),
            "stock_resume_load_canary": _report_ref(
                stock_resume_path,
                stock_resume,
            ),
            "isolated_live_run_safety_canary": _report_ref(
                isolated_live_path,
                isolated_live,
            ),
            "sandbox_assignment_rating_proof": _report_ref(
                sandbox_rating_path,
                sandbox_rating,
            ),
            "longer_horizon_compact_learning_metrics": _report_ref(
                longer_path,
                longer,
            ),
        },
        "review_decision": {
            "ready_for_manual_review": True,
            "promotion_claim": False,
            "automatic_promotion_allowed": False,
            "manual_review_required_before_any_promotion": True,
            "matched_quality_sufficiency_decision": (
                "canary_scale_manual_acceptance_or_larger_study_required_before_promotion"
            ),
        },
        "quality_strength_review": {
            "current_matched_canary_sufficient_for_promotion_claim": False,
            "manual_acceptance_or_larger_study_required_before_promotion": True,
            "promotion_quality_claim": False,
            "compact_quality_superiority_claim": False,
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
                "sample_batch_count_delta": 3,
            },
        },
    }
    _write_json(bundle_path, bundle)
    return bundle_path


def _matched_report(
    candidate: str,
    *,
    stock_capture_path: Path,
    compact_capture_path: Path,
) -> dict:
    compact_eval = {
        "eval_seed_set": list(range(8)),
        "eval_seed_rng_seed": 20260833,
        "eval_max_steps": 1024,
        "reward_variant": "survival_plus_bonus_no_outcome",
        "death_mode": "normal",
        "terminal_target_mode": "stock_terminal_no_bootstrap_return_discount_1.0",
        "rnd_enabled": False,
        "policy_observation_backend": "cpu_oracle",
        "opponent_policy_kind": "fixed_straight",
        "opponent_runtime_mode": "normal",
        "opponent_death_mode": "normal",
        "natural_bonus_spawn": True,
        "num_simulations": 1,
        "batch_size": 2,
        "root_noise": 0.0,
        "policy_noise": 0.0,
        "source_max_steps": 1_048_576,
    }
    stock_eval = dict(compact_eval)
    return {
        "schema_id": "curvyzero_compact_promotion_matched_learning_quality_canary/v1",
        "ok": True,
        "status": "matched_learning_quality_movement_observed",
        "candidate_checkpoint_id": candidate,
        "evidence_ref": "unit-matched-ref",
        "input_capture_files": {
            "stock_reference_capture": {
                "path": str(stock_capture_path.resolve()),
                "sha256": _sha256(stock_capture_path),
            },
            "compact_candidate_capture": {
                "path": str(compact_capture_path.resolve()),
                "sha256": _sha256(compact_capture_path),
            },
        },
        "matched_pair": {
            "denominator_id": "matched_learning_quality_current_1024x8_denominator_v1",
            "quality_horizon": "matched_learning_quality_pre_post_eval_1024x8_current",
            "hardware_class": "mixed",
            "stock_hardware_class": "modal-gpu-l4-t4-cpu40",
            "compact_hardware_class": "local-cpu-producer-smoke",
        },
        "quality_movement": {
            "stock_reference_delta": -0.875,
            "compact_candidate_delta": 4.75,
            "compact_minus_stock_delta": 5.625,
            "quality_currency": "mean_survival_delta",
        },
        "arms": {
            review.STOCK_REFERENCE_ROLE: {
                "calls_train_muzero": True,
                "eval_settings": stock_eval,
                "denominators": {
                    "collector_envstep_delta": 533,
                    "learner_train_calls": 133,
                    "replay_sample_calls": 133,
                    "uses_fallback_denominator": False,
                },
            },
            review.COMPACT_CANDIDATE_ROLE: {
                "calls_train_muzero": False,
                "eval_settings": compact_eval,
                "source_fingerprint": {
                    "matched_surface": {
                        "env_variant": "source_state_fixed_opponent",
                    },
                },
                "denominators": {
                    "compact_rollout_rows": 64,
                    "compact_sample_rows": 30,
                    "learner_update_count_delta": 30,
                    "sample_batch_count_delta": 16,
                    "uses_fallback_denominator": False,
                },
            },
        },
    }


def _report_ref(path: Path, payload: dict) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "schema_id": payload["schema_id"],
        "status": payload["status"],
        "candidate_checkpoint_id": payload["candidate_checkpoint_id"],
        "evidence_ref": payload.get("evidence_ref", ""),
    }


def _load_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_matched_quality_sufficiency_review.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_quality_sufficiency_review_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch_target = "validate_compact_promotion_readiness_bundle_review_v1"
    setattr(module, monkeypatch_target, lambda payload: None)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
