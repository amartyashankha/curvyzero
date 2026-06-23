from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from curvyzero.training import compact_promotion_readiness_bundle_review as review


def test_bundle_review_builds_hash_bound_manual_review(monkeypatch, tmp_path):
    calls = _patch_lane_validators(monkeypatch)
    inputs = _write_bundle_inputs(tmp_path)

    payload = review.build_compact_promotion_readiness_bundle_review_v1(
        run_id="unit-bundle-review",
        **inputs,
        created_at="2026-05-30T00:00:00+00:00",
    )

    assert payload["schema_id"] == review.COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID
    assert payload["status"] == review.COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_STATUS
    assert payload["candidate_checkpoint_id"] == "unit-compact-ckpt"
    assert payload["review_decision"]["ready_for_manual_review"] is True
    assert payload["review_decision"]["promotion_claim"] is False
    assert payload["review_decision"]["automatic_promotion_allowed"] is False
    assert payload["attached_claims"]["readiness_bundle_hash_bound"] is True
    assert payload["non_claims"]["compact_quality_superiority_claim"] is False
    assert payload["quality_strength_review"][
        "current_matched_canary_sufficient_for_promotion_claim"
    ] is False
    assert set(payload["evidence_lanes"]) == set(review.EVIDENCE_LANE_KEYS)
    assert len(calls) == 12
    review.validate_compact_promotion_readiness_bundle_review_v1(payload)


def test_bundle_review_rejects_candidate_mismatch(monkeypatch, tmp_path):
    _patch_lane_validators(monkeypatch)

    def mutate(reports):
        reports["longer_horizon_compact_learning_metrics"][
            "candidate_checkpoint_id"
        ] = "other-candidate"

    inputs = _write_bundle_inputs(tmp_path, mutator=mutate)

    with pytest.raises(
        review.CompactPromotionReadinessBundleReviewError,
        match="candidate_checkpoint_id mismatch",
    ):
        review.build_compact_promotion_readiness_bundle_review_v1(
            run_id="unit-bundle-review",
            **inputs,
        )


def test_bundle_review_rejects_input_hash_drift(monkeypatch, tmp_path):
    _patch_lane_validators(monkeypatch)
    inputs = _write_bundle_inputs(tmp_path)
    payload = review.build_compact_promotion_readiness_bundle_review_v1(
        run_id="unit-bundle-review",
        **inputs,
    )
    Path(inputs["matched_learning_quality_report_path"]).write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        review.CompactPromotionReadinessBundleReviewError,
        match="matched_learning_quality_canary input report sha256 mismatch",
    ):
        review.validate_compact_promotion_readiness_bundle_review_v1(payload)


def test_bundle_review_rejects_input_claim_flip(monkeypatch, tmp_path):
    _patch_lane_validators(monkeypatch)

    def mutate(reports):
        reports["stock_resume_load_canary"]["non_claims"]["promotion_claim"] = True

    inputs = _write_bundle_inputs(tmp_path, mutator=mutate)

    with pytest.raises(
        review.CompactPromotionReadinessBundleReviewError,
        match="promotion_claim must be false",
    ):
        review.build_compact_promotion_readiness_bundle_review_v1(
            run_id="unit-bundle-review",
            **inputs,
        )


def test_bundle_review_rejects_output_promotion_claim(monkeypatch, tmp_path):
    _patch_lane_validators(monkeypatch)
    inputs = _write_bundle_inputs(tmp_path)
    payload = review.build_compact_promotion_readiness_bundle_review_v1(
        run_id="unit-bundle-review",
        **inputs,
    )
    payload["review_decision"]["promotion_claim"] = True

    with pytest.raises(
        review.CompactPromotionReadinessBundleReviewError,
        match="promotion_claim must be false",
    ):
        review.validate_compact_promotion_readiness_bundle_review_v1(payload)


def test_bundle_review_cli_writes_report_and_rejects_stale_output(
    monkeypatch,
    tmp_path,
):
    _patch_lane_validators(monkeypatch)
    module = _load_cli_module()
    inputs = _write_bundle_inputs(tmp_path)
    output_root = tmp_path / "out"
    run_id = "unit-bundle-review-main"

    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--compatibility-report",
        str(inputs["compatibility_report_path"]),
        "--unified-lifecycle-report",
        str(inputs["unified_lifecycle_report_path"]),
        "--matched-learning-quality-report",
        str(inputs["matched_learning_quality_report_path"]),
        "--matched-pair-verification-report",
        str(inputs["matched_pair_verification_report_path"]),
        "--stock-resume-load-canary-report",
        str(inputs["stock_resume_load_canary_report_path"]),
        "--isolated-live-run-safety-canary-report",
        str(inputs["isolated_live_run_safety_canary_report_path"]),
        "--sandbox-assignment-rating-proof-report",
        str(inputs["sandbox_assignment_rating_proof_report_path"]),
        "--longer-horizon-learning-metrics-report",
        str(inputs["longer_horizon_learning_metrics_report_path"]),
    ]

    assert module.main(argv) == 0
    report_path = output_root / run_id / "readiness_bundle_review_report.json"
    payload = _read_json(report_path)
    assert payload["status"] == review.COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_STATUS
    assert payload["review_decision"]["promotion_claim"] is False
    review.validate_compact_promotion_readiness_bundle_review_v1(payload)

    with pytest.raises(FileExistsError):
        module.main(argv)


def _patch_lane_validators(monkeypatch):
    calls: list[str] = []

    def fake_validator(payload):
        calls.append(str(payload.get("schema_id", "")))
        if payload.get("ok") is not True:
            raise AssertionError("test fixture report must be ok=true")

    for name in (
        "validate_compact_matched_learning_quality_canary_v1",
        "validate_compact_matched_learning_quality_pair_verification_v1",
        "validate_compact_promotion_stock_resume_load_canary_v1",
        "validate_compact_promotion_isolated_live_run_safety_canary_v1",
        "validate_compact_promotion_sandbox_assignment_rating_proof_v1",
        "validate_compact_longer_horizon_learning_metrics_v1",
    ):
        monkeypatch.setattr(review, name, fake_validator)
    return calls


def _write_bundle_inputs(tmp_path: Path, *, mutator=None) -> dict[str, Path]:
    candidate = "unit-compact-ckpt"
    shared_checkpoint_sha = "a" * 64
    false_claims = {key: False for key in review.FALSE_CLAIM_KEYS}
    compatibility_path = tmp_path / "compatibility_report.json"
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"

    compatibility = {
        "schema_id": review.COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID,
        "ok": True,
        "candidate_checkpoint_id": candidate,
        "promotion_eligible": True,
        "coach_speed_row_gate": True,
        "promotion_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "loaded_checkpoint_identity": {
            "compact_checkpoint_sha256": shared_checkpoint_sha,
            "model_state_digest": "digest-initial",
        },
        "non_claims": {
            "promotion_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "rating_or_promotion_quality_claim": False,
            "stock_resume_claim": False,
        },
    }
    lifecycle = {
        "schema_id": review.COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID,
        "ok": True,
        "checkpoint_id": candidate,
        "lifecycle_gates_complete": True,
        "promotion_claim": False,
    }
    _write_json(compatibility_path, compatibility)
    _write_json(lifecycle_path, lifecycle)
    compatibility_sha = _sha256(compatibility_path)
    lifecycle_sha = _sha256(lifecycle_path)

    reports = {
        "matched_learning_quality_canary": _matched_quality_report(
            candidate,
            compatibility_sha=compatibility_sha,
            lifecycle_sha=lifecycle_sha,
            shared_checkpoint_sha=shared_checkpoint_sha,
            false_claims=false_claims,
        ),
        "stock_resume_load_canary": _lane_report(
            review.COMPACT_PROMOTION_STOCK_RESUME_LOAD_CANARY_SCHEMA_ID,
            candidate,
            "stock_resume_load_canary",
            compatibility_sha=compatibility_sha,
            lifecycle_sha=lifecycle_sha,
            false_claims=false_claims,
            extra={
                "source_compact_checkpoint_sha256": shared_checkpoint_sha,
                "readiness": {
                    "stock_resume_load_canary": True,
                    "promotion_readiness_complete": False,
                },
            },
        ),
        "isolated_live_run_safety_canary": _lane_report(
            review.COMPACT_PROMOTION_ISOLATED_LIVE_RUN_SAFETY_CANARY_SCHEMA_ID,
            candidate,
            "isolated_live_run_safety_canary",
            compatibility_sha=compatibility_sha,
            lifecycle_sha=lifecycle_sha,
            false_claims=false_claims,
            extra={
                "forbidden_touch_audit": {
                    "touches_live_runs": False,
                    "touches_production_live_runs": False,
                },
                "readiness": {
                    "isolated_live_run_safety_canary": True,
                    "promotion_readiness_complete": False,
                },
            },
        ),
        "sandbox_assignment_rating_proof": _lane_report(
            review.COMPACT_PROMOTION_SANDBOX_ASSIGNMENT_RATING_PROOF_SCHEMA_ID,
            candidate,
            "sandbox_assignment_rating_proof",
            compatibility_sha=compatibility_sha,
            lifecycle_sha=lifecycle_sha,
            false_claims=false_claims,
            extra={
                "source_compact_checkpoint_sha256": shared_checkpoint_sha,
                "forbidden_touch_audit": {
                    "public_leaderboard_publish_claim": False,
                    "promotion_published": False,
                },
                "sandbox_scope": {
                    "local_only": True,
                    "production_namespace": False,
                },
                "readiness": {
                    "sandbox_assignment_rating_proof": True,
                    "promotion_readiness_complete": False,
                },
            },
        ),
        "longer_horizon_compact_learning_metrics": _lane_report(
            review.COMPACT_LONGER_HORIZON_LEARNING_METRICS_SCHEMA_ID,
            candidate,
            "longer_horizon_compact_learning_metrics",
            compatibility_sha=compatibility_sha,
            lifecycle_sha=lifecycle_sha,
            false_claims=false_claims,
            extra={
                "loaded_compact_checkpoint_sha256": shared_checkpoint_sha,
                "checkpoint_series": [
                    {"candidate_checkpoint_id": candidate},
                    {"candidate_checkpoint_id": candidate},
                    {"candidate_checkpoint_id": candidate},
                ],
                "cumulative_denominators": {
                    "learner_update_count_delta": 2,
                    "sample_batch_count_delta": 3,
                },
                "readiness": {
                    "longer_horizon_compact_learning_metrics": True,
                    "promotion_readiness_complete": False,
                },
            },
        ),
    }
    if mutator is not None:
        mutator(reports)

    matched_path = tmp_path / "matched_learning_quality_canary_report.json"
    stock_path = tmp_path / "stock_resume_load_canary_report.json"
    isolated_path = tmp_path / "isolated_live_run_safety_canary_report.json"
    sandbox_path = tmp_path / "sandbox_assignment_rating_proof_report.json"
    longer_path = tmp_path / "longer_horizon_learning_metrics_report.json"
    _write_json(matched_path, reports["matched_learning_quality_canary"])
    matched_sha = _sha256(matched_path)
    reports["matched_pair_verification"] = {
        "schema_id": review.COMPACT_MATCHED_LEARNING_QUALITY_PAIR_VERIFICATION_SCHEMA_ID,
        "ok": True,
        "status": "matched_pair_verified",
        "candidate_checkpoint_id": candidate,
        "readiness_lane": "matched_stock_vs_compact_learning_quality",
        "matched_learning_quality_report": {
            "path": str(matched_path.resolve()),
            "sha256": matched_sha,
            "schema_id": review.COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID,
        },
        "non_claims": false_claims,
    }
    _write_json(stock_path, reports["stock_resume_load_canary"])
    stock_sha = _sha256(stock_path)
    reports["isolated_live_run_safety_canary"][
        "stock_resume_load_canary_report_sha256"
    ] = stock_sha
    reports["sandbox_assignment_rating_proof"][
        "stock_resume_load_canary_report_sha256"
    ] = stock_sha
    _write_json(isolated_path, reports["isolated_live_run_safety_canary"])
    isolated_sha = _sha256(isolated_path)
    reports["sandbox_assignment_rating_proof"][
        "isolated_live_run_safety_canary_report_sha256"
    ] = isolated_sha
    _write_json(sandbox_path, reports["sandbox_assignment_rating_proof"])
    _write_json(longer_path, reports["longer_horizon_compact_learning_metrics"])
    pair_path = tmp_path / "matched_pair_verification_report.json"
    _write_json(pair_path, reports["matched_pair_verification"])

    return {
        "compatibility_report_path": compatibility_path,
        "unified_lifecycle_report_path": lifecycle_path,
        "matched_learning_quality_report_path": matched_path,
        "matched_pair_verification_report_path": pair_path,
        "stock_resume_load_canary_report_path": stock_path,
        "isolated_live_run_safety_canary_report_path": isolated_path,
        "sandbox_assignment_rating_proof_report_path": sandbox_path,
        "longer_horizon_learning_metrics_report_path": longer_path,
    }


def _lane_report(
    schema_id: str,
    candidate: str,
    readiness_lane: str,
    *,
    compatibility_sha: str,
    lifecycle_sha: str,
    false_claims: dict[str, bool],
    extra: dict,
) -> dict:
    payload = {
        "schema_id": schema_id,
        "ok": True,
        "status": f"{readiness_lane}_verified",
        "candidate_checkpoint_id": candidate,
        "readiness_lane": readiness_lane,
        "compatibility_report_sha256": compatibility_sha,
        "unified_lifecycle_report_sha256": lifecycle_sha,
        "attached_claims": false_claims | {readiness_lane: True},
        "non_claims": false_claims,
    }
    payload.update(copy.deepcopy(extra))
    return payload


def _matched_quality_report(
    candidate: str,
    *,
    compatibility_sha: str,
    lifecycle_sha: str,
    shared_checkpoint_sha: str,
    false_claims: dict[str, bool],
) -> dict:
    compact_arm = {
        "role": review.COMPACT_CANDIDATE_ROLE,
        "candidate_checkpoint_id": candidate,
        "calls_train_muzero": False,
        "eval_settings": {
            "eval_seed_set": [101, 102],
            "eval_max_steps": 128,
        },
        "source_fingerprint": {
            "loaded_compact_checkpoint_sha256": shared_checkpoint_sha,
        },
    }
    stock_arm = {
        "role": review.STOCK_REFERENCE_ROLE,
        "candidate_checkpoint_id": candidate,
        "calls_train_muzero": True,
    }
    return {
        "schema_id": review.COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID,
        "ok": True,
        "status": "matched_learning_quality_movement_observed",
        "candidate_checkpoint_id": candidate,
        "readiness_lane": "matched_stock_vs_compact_learning_quality",
        "compatibility_report_sha256": compatibility_sha,
        "unified_lifecycle_report_sha256": lifecycle_sha,
        "arms": {
            review.COMPACT_CANDIDATE_ROLE: compact_arm,
            review.STOCK_REFERENCE_ROLE: stock_arm,
        },
        "arm_call_contract": {
            "stock_reference_calls_train_muzero": True,
            "compact_candidate_calls_train_muzero": False,
            "both_touch_live_runs": False,
        },
        "matched_pair": {
            "denominator_id": "unit-denominator",
            "quality_horizon": "unit-horizon",
            "hardware_class": "mixed",
            "stock_hardware_class": "unit-stock",
            "compact_hardware_class": "unit-compact",
        },
        "readiness": {
            "matched_learning_quality_canary": True,
            "promotion_readiness_complete": False,
        },
        "attached_claims": false_claims | {"matched_learning_quality_canary": True},
        "non_claims": false_claims,
        "evidence_ref": "unit-matched-evidence-ref",
    }


def _load_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_promotion_readiness_bundle_review.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_promotion_readiness_bundle_review_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
