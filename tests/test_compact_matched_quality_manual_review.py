from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from curvyzero.training import compact_matched_quality_manual_review as review


def test_manual_review_builds_all_allowed_decisions(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    source = _write_sufficiency_fixture(tmp_path)

    decisions = (
        (
            review.DECISION_KEEP_COMPACT_CANDIDATE_ONLY,
            None,
        ),
        (
            review.DECISION_REQUIRE_LARGER_REPEAT,
            None,
        ),
        (
            review.DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL,
            "compact_nonprod_shadow_assignment_001",
        ),
    )
    for decision, proposal_id in decisions:
        payload = review.build_compact_matched_quality_manual_review_v1(
            run_id=f"unit-manual-review-{decision}",
            sufficiency_review_path=source,
            manual_review_decision=decision,
            next_allowed_step="unit_next_step",
            named_non_production_proposal_id=proposal_id,
            created_at="2026-05-31T00:00:00+00:00",
        )

        assert payload["schema_id"] == review.COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_SCHEMA_ID
        assert payload["opt_id"] == "OPT-070"
        assert payload["reviewed_packet"] == review.REVIEWED_PACKET
        assert payload["manual_decision"]["manual_review_decision"] == decision
        assert payload["non_claims"]["promotion_claim"] is False
        assert payload["non_claims"]["automatic_promotion_allowed"] is False
        assert payload["non_claims"]["stock_train_muzero_speedup_claim"] is False
        review.validate_compact_matched_quality_manual_review_v1(payload)


def test_manual_review_requires_named_proposal_id(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    source = _write_sufficiency_fixture(tmp_path)

    with pytest.raises(
        review.CompactMatchedQualityManualReviewError,
        match="named non-production proposal id is required",
    ):
        review.build_compact_matched_quality_manual_review_v1(
            sufficiency_review_path=source,
            manual_review_decision=(
                review.DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL
            ),
        )


def test_manual_review_rejects_wrong_source_status_or_step(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    source = _write_sufficiency_fixture(tmp_path)
    payload = _read_json(source)
    payload["status"] = "larger_same_surface_study_required"
    _write_json(source, payload)

    with pytest.raises(
        review.CompactMatchedQualityManualReviewError,
        match="source sufficiency must be accepted",
    ):
        review.build_compact_matched_quality_manual_review_v1(
            sufficiency_review_path=source,
        )

    source = _write_sufficiency_fixture(tmp_path, suffix="wrong-step")
    payload = _read_json(source)
    payload["decision"]["next_non_production_step"] = "some_other_step"
    _write_json(source, payload)

    with pytest.raises(
        review.CompactMatchedQualityManualReviewError,
        match="next_non_production_step mismatch",
    ):
        review.build_compact_matched_quality_manual_review_v1(
            sufficiency_review_path=source,
        )


def test_manual_review_rejects_claim_flips(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    source = _write_sufficiency_fixture(tmp_path)
    payload = review.build_compact_matched_quality_manual_review_v1(
        sufficiency_review_path=source,
    )
    payload["non_claims"]["stock_train_muzero_speedup_claim"] = True

    with pytest.raises(
        review.CompactMatchedQualityManualReviewError,
        match="stock_train_muzero_speedup_claim must be false",
    ):
        review.validate_compact_matched_quality_manual_review_v1(payload)

    payload = review.build_compact_matched_quality_manual_review_v1(
        sufficiency_review_path=source,
    )
    payload["manual_decision"]["automatic_promotion_allowed"] = True

    with pytest.raises(
        review.CompactMatchedQualityManualReviewError,
        match="automatic_promotion_allowed must be false",
    ):
        review.validate_compact_matched_quality_manual_review_v1(payload)


def test_manual_review_rejects_source_and_transitive_hash_drift(
    monkeypatch,
    tmp_path,
):
    _patch_sufficiency_validator(monkeypatch)
    source = _write_sufficiency_fixture(tmp_path)
    payload = review.build_compact_matched_quality_manual_review_v1(
        sufficiency_review_path=source,
    )
    _write_json(source, {"schema_id": "changed"})

    with pytest.raises(
        review.CompactMatchedQualityManualReviewError,
        match="source_sufficiency_review sha256 mismatch",
    ):
        review.validate_compact_matched_quality_manual_review_v1(payload)

    source = _write_sufficiency_fixture(tmp_path, suffix="transitive")
    payload = review.build_compact_matched_quality_manual_review_v1(
        sufficiency_review_path=source,
    )
    stock_path = Path(payload["input_packet_refs"]["stock_reference_capture"]["path"])
    _write_json(stock_path, {"schema_id": "changed"})

    with pytest.raises(
        review.CompactMatchedQualityManualReviewError,
        match="stock_reference_capture sha256 mismatch",
    ):
        review.validate_compact_matched_quality_manual_review_v1(payload)


def test_manual_review_derives_missing_capture_schema_ids(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    source = _write_sufficiency_fixture(tmp_path)
    source_payload = _read_json(source)
    del source_payload["input_reports"]["stock_reference_capture"]["schema_id"]
    del source_payload["input_reports"]["compact_candidate_capture"]["schema_id"]
    _write_json(source, source_payload)

    payload = review.build_compact_matched_quality_manual_review_v1(
        sufficiency_review_path=source,
    )

    assert payload["input_packet_refs"]["stock_reference_capture"]["schema_id"] == (
        "curvyzero_compact_matched_learning_quality_capture/v1"
    )
    assert payload["input_packet_refs"]["compact_candidate_capture"]["schema_id"] == (
        "curvyzero_compact_matched_learning_quality_capture/v1"
    )
    review.validate_compact_matched_quality_manual_review_v1(payload)


def test_manual_review_packet_summary_carries_larger_packet_guards(
    monkeypatch,
    tmp_path,
):
    _patch_sufficiency_validator(monkeypatch)
    source = _write_sufficiency_fixture(tmp_path)
    payload = review.build_compact_matched_quality_manual_review_v1(
        sufficiency_review_path=source,
    )
    summary = payload["packet_summary"]

    assert summary["eval_seed_count"] == 32
    assert summary["eval_max_steps"] == 2048
    assert summary["denominator_id"] == (
        "matched_quality_larger_2048x32-env64train8_denominator_v1"
    )
    assert summary["stock_reference_delta"] == 5.0625
    assert summary["compact_candidate_delta"] == 5.4375
    assert summary["compact_minus_stock_delta"] == 0.375
    assert summary["hardware_class"] == "mixed"
    assert summary["compact_hardware_class"] == "local-cpu-producer-smoke"
    assert summary["accepted_stock_capture_is_evalenv32"] is True
    assert (
        summary["compact_denominator_contract"]["learner_updates_per_sample_batch"]
        == 7.875
    )
    assert payload["canonical_stock_failure"]["acknowledged"] is True
    assert payload["limitation_acknowledgements"]["dirty_source_fingerprints"] is True


def test_manual_review_rejects_missing_evalenv32_ack(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    source = _write_sufficiency_fixture(tmp_path)

    with pytest.raises(
        review.CompactMatchedQualityManualReviewError,
        match="stock workaround acknowledgement missing: acknowledged",
    ):
        review.build_compact_matched_quality_manual_review_v1(
            sufficiency_review_path=source,
            canonical_stock_failure_acknowledged=False,
        )

    with pytest.raises(
        review.CompactMatchedQualityManualReviewError,
        match="accepted stock capture run id must name evalenv32",
    ):
        review.build_compact_matched_quality_manual_review_v1(
            sufficiency_review_path=source,
            accepted_stock_capture_run_id=(
                "optimizer-stock-reference-quality-producer-larger-2048x32-20260531"
            ),
        )


def test_manual_review_cli_writes_report_and_rejects_stale_output(
    monkeypatch,
    tmp_path,
):
    _patch_sufficiency_validator(monkeypatch)
    module = _load_cli_module()
    source = _write_sufficiency_fixture(tmp_path)
    output_root = tmp_path / "out"
    run_id = "unit-manual-review-cli"
    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--sufficiency-review",
        str(source),
    ]

    assert module.main(argv) == 0
    report_path = output_root / run_id / "manual_review_decision_report.json"
    manifest_path = output_root / run_id / "manifest.json"
    payload = _read_json(report_path)
    manifest = _read_json(manifest_path)
    assert payload["manual_decision"]["manual_review_decision"] == (
        review.DECISION_KEEP_COMPACT_CANDIDATE_ONLY
    )
    assert manifest["schema_id"] == (
        review.COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_MANIFEST_SCHEMA_ID
    )
    review.validate_compact_matched_quality_manual_review_v1(payload)

    with pytest.raises(FileExistsError):
        module.main(argv)


def _patch_sufficiency_validator(monkeypatch):
    monkeypatch.setattr(
        review,
        "validate_compact_matched_quality_sufficiency_review_v1",
        lambda payload: None,
    )


def _write_sufficiency_fixture(tmp_path: Path, *, suffix: str = "base") -> Path:
    root = tmp_path / suffix
    root.mkdir(exist_ok=True)
    refs = {
        "readiness_bundle_review": _write_input(
            root,
            "readiness_bundle_review_report.json",
            "curvyzero_compact_promotion_readiness_bundle_review/v1",
            evidence_ref="bundle-ref",
        ),
        "matched_learning_quality_canary": _write_input(
            root,
            "matched_learning_quality_canary_report.json",
            "curvyzero_compact_promotion_matched_learning_quality_canary/v1",
            evidence_ref="canary-ref",
        ),
        "matched_pair_verification": _write_input(
            root,
            "matched_pair_verification_report.json",
            "curvyzero_compact_matched_learning_quality_pair_verification/v1",
            evidence_ref="pair-ref",
        ),
        "longer_horizon_compact_learning_metrics": _write_input(
            root,
            "longer_horizon_learning_metrics_report.json",
            "curvyzero_compact_longer_horizon_learning_metrics/v1",
            evidence_ref="longer-ref",
        ),
        "stock_reference_capture": _write_input(
            root,
            "optimizer-stock-reference-quality-producer-larger-2048x32-20260531-evalenv32/stock_reference_capture.json",
            "curvyzero_compact_matched_learning_quality_capture/v1",
        ),
        "compact_candidate_capture": _write_input(
            root,
            "optimizer-compact-candidate-env-search-replay-larger-2048x32-env64train8-20260531/compact_candidate_capture.json",
            "curvyzero_compact_matched_learning_quality_capture/v1",
        ),
    }
    sufficiency = {
        "schema_id": "curvyzero_compact_matched_quality_sufficiency_review/v1",
        "ok": True,
        "status": "current_canary_accepted_for_next_non_production_step",
        "run_id": "unit-sufficiency",
        "candidate_checkpoint_id": "unit-compact-ckpt",
        "decision": {
            "matched_quality_sufficiency_decision": (
                "accept_current_for_next_non_production_step"
            ),
            "next_non_production_step": (
                "larger_32x2048_matched_quality_manual_review_not_promotion"
            ),
            "promotion_claim": False,
            "automatic_promotion_allowed": False,
            "current_evidence_sufficient_for_promotion": False,
            "limitation_acknowledgements": {
                "current_canary_scale": True,
                "mixed_hardware": True,
                "compact_local_cpu": True,
                "tiny_longer_horizon_trace": True,
                "promotion_requires_separate_policy_decision": True,
                "manual_review_required_before_any_promotion": True,
            },
        },
        "current_evidence_summary": {
            "quality_currency": "mean_survival_delta",
            "denominator_id": (
                "matched_quality_larger_2048x32-env64train8_denominator_v1"
            ),
            "quality_horizon": (
                "matched_learning_quality_pre_post_eval_larger_2048x32-env64train8"
            ),
            "eval_seed_count": 32,
            "eval_max_steps": 2048,
            "stock_reference_delta": 5.0625,
            "compact_candidate_delta": 5.4375,
            "compact_minus_stock_delta": 0.375,
            "hardware_class": "mixed",
            "stock_hardware_class": "modal-gpu-l4-t4-cpu40",
            "compact_hardware_class": "local-cpu-producer-smoke",
            "bundle_matched_pair_summary": {
                "stock_calls_train_muzero": True,
                "compact_calls_train_muzero": False,
            },
            "stock_denominators": {
                "uses_fallback_denominator": False,
                "wall_sec_used_for_speed_claim": False,
            },
            "compact_denominators": {
                "compact_rollout_rows": 256,
                "compact_sample_rows": 252,
                "sample_batch_count_delta": 64,
                "learner_update_count_delta": 504,
                "uses_fallback_denominator": False,
                "wall_sec_used_for_speed_claim": False,
            },
            "known_limitations": {
                "compact_quality_superiority_claim": False,
                "promotion_quality_claim": False,
            },
        },
        "input_reports": refs,
        "evidence_ref": "unit-sufficiency-ref",
    }
    path = root / "matched_quality_sufficiency_review_report.json"
    _write_json(path, sufficiency)
    return path


def _write_input(
    root: Path,
    relative: str,
    schema_id: str,
    *,
    evidence_ref: str | None = None,
) -> dict[str, str]:
    path = root / relative
    payload = {"schema_id": schema_id, "ok": True}
    if evidence_ref is not None:
        payload["evidence_ref"] = evidence_ref
    _write_json(path, payload)
    ref = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "schema_id": schema_id,
    }
    if evidence_ref is not None:
        ref["evidence_ref"] = evidence_ref
    return ref


def _load_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_matched_quality_manual_review.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_quality_manual_review_for_test",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
