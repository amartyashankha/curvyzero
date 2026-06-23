from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from curvyzero.training import compact_promotion_readiness_learning_quality as quality


def test_matched_learning_quality_canary_validates_two_arm_quality_movement(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["hardware_class"] = "local-compact-cpu"
    compact_arm["eval_settings"]["training_seed_policy"] = "fixed_eval_seed"
    compact_arm["eval_settings"]["initialization_source"] = "loaded_compact_checkpoint"

    payload = _build_compact_matched_learning_quality_canary_v1(
        tmp_path,
        run_id="unit-quality",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_arm=stock_arm,
        compact_candidate_arm=compact_arm,
        created_at="2026-05-30T00:00:00+00:00",
    )

    assert payload["schema_id"] == quality.COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID
    assert payload["ok"] is True
    assert payload["readiness_lane"] == (quality.COMPACT_MATCHED_LEARNING_QUALITY_READINESS_LANE)
    assert payload["arm_call_contract"]["stock_reference_calls_train_muzero"] is True
    assert payload["arm_call_contract"]["compact_candidate_calls_train_muzero"] is False
    assert payload["matched_pair"]["hardware_class"] == "mixed"
    assert payload["matched_pair"]["stock_hardware_class"] == "local-test"
    assert payload["matched_pair"]["compact_hardware_class"] == "local-compact-cpu"
    assert payload["quality_movement"]["stock_reference_delta"] == pytest.approx(2.0)
    assert payload["quality_movement"]["compact_candidate_delta"] == pytest.approx(3.0)
    assert payload["readiness"]["promotion_readiness_complete"] is False
    assert payload["non_claims"]["promotion_claim"] is False
    assert payload["non_claims"]["rating_or_promotion_quality_claim"] is False
    assert payload["evidence_ref"].startswith(
        quality.COMPACT_MATCHED_LEARNING_QUALITY_EVIDENCE_REF_PREFIX
    )
    quality.validate_compact_matched_learning_quality_canary_v1(payload)


def test_matched_learning_quality_rejects_one_point_curve(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["quality_curve"] = compact_arm["quality_curve"][:1]

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="at least two eval points",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_allows_eval_seed_order_mismatch(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["eval_settings"]["eval_seed_set"] = [102, 101]
    compact_arm["source_fingerprint"]["matched_surface"]["eval_seed_set"] = [102, 101]

    payload = _build_compact_matched_learning_quality_canary_v1(
        tmp_path,
        run_id="unit-quality",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_arm=stock_arm,
        compact_candidate_arm=compact_arm,
    )

    quality.validate_compact_matched_learning_quality_canary_v1(payload)


def test_matched_learning_quality_rejects_eval_seed_set_mismatch(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["eval_settings"]["eval_seed_set"] = [101, 999]
    compact_arm["source_fingerprint"]["matched_surface"]["eval_seed_set"] = [101, 999]

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="eval_settings eval_seed_set mismatch",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_single_eval_seed(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["eval_settings"]["eval_seed_set"] = [101]
    compact_arm["eval_settings"]["eval_episode_count"] = 1
    compact_arm["source_fingerprint"]["matched_surface"]["eval_seed_set"] = [101]
    for point in compact_arm["quality_curve"]:
        point["eval_episode_count"] = 1

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="eval_seed_set must contain at least",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_saturated_eval_curve(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["quality_curve"][0]["cap_rate"] = 1.0

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="cap_rate exceeds saturation ceiling",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_allows_route_specific_source_fingerprints(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    stock_arm["source_fingerprint"]["producer_route"] = "stock-route"
    compact_arm["source_fingerprint"]["producer_route"] = "compact-route"

    payload = _build_compact_matched_learning_quality_canary_v1(
        tmp_path,
        run_id="unit-quality",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_arm=stock_arm,
        compact_candidate_arm=compact_arm,
    )

    assert payload["matched_pair"]["stock_source_fingerprint_sha256"] != (
        payload["matched_pair"]["compact_source_fingerprint_sha256"]
    )


def test_matched_learning_quality_rejects_reward_death_or_rnd_mismatch(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["eval_settings"]["death_mode"] = "profile_no_death"

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="death_mode must be normal",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_compact_train_muzero_claim(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["calls_train_muzero"] = True

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="compact_candidate calls_train_muzero must be false",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_stock_without_train_muzero(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    stock_arm["calls_train_muzero"] = False

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="stock_reference calls_train_muzero must be true",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_profile_currency(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["denominators"]["speed_currency"] = "compact_profile_active_roots_per_sec"

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="profile-only speed currency",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_fallback_denominator(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    stock_arm["denominators"]["uses_fallback_denominator"] = True

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="uses_fallback_denominator",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_non_finite_metric(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["quality_curve"][-1]["mean_survival"] = math.nan

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="non-finite mean_survival",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_unchanged_model_digest(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["final_model_state_digest"] = compact_arm["initial_model_state_digest"]

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="model state digest did not change",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_no_observed_movement(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["quality_curve"][-1].update(
        {
            "mean_survival": compact_arm["quality_curve"][0]["mean_survival"],
            "median_survival": compact_arm["quality_curve"][0]["median_survival"],
            "min_survival": compact_arm["quality_curve"][0]["min_survival"],
            "max_survival": compact_arm["quality_curve"][0]["max_survival"],
        }
    )

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="observed mean_survival movement",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_support_only_compact_identity(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["model_identity_scope"] = "candidate_named_support_only"

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="candidate_loaded_checkpoint",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_non_claim_flip(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["non_claims"]["rating_or_promotion_quality_claim"] = True

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="rating_or_promotion_quality_claim",
    ):
        _build_compact_matched_learning_quality_canary_v1(
            tmp_path,
            run_id="unit-quality",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_arm=stock_arm,
            compact_candidate_arm=compact_arm,
        )


def test_matched_learning_quality_rejects_artifact_hash_drift(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    payload = _build_compact_matched_learning_quality_canary_v1(
        tmp_path,
        run_id="unit-quality",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_arm=stock_arm,
        compact_candidate_arm=compact_arm,
    )
    drift_ref = payload["arms"]["compact_candidate"]["artifact_refs"][0]
    with open(drift_ref["path"], "ab") as fh:
        fh.write(b"drift")

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="sha256 mismatch",
    ):
        quality.validate_compact_matched_learning_quality_canary_v1(payload)


def test_matched_pair_verifier_builds_reciprocal_report(tmp_path):
    paths = _input_paths(tmp_path)
    payload = _build_compact_matched_learning_quality_canary_v1(
        tmp_path,
        run_id="unit-quality",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_arm=_quality_arm(tmp_path, role="stock_reference"),
        compact_candidate_arm=_quality_arm(tmp_path, role="compact_candidate"),
    )
    report_path = tmp_path / "matched_learning_quality_canary_report.json"
    _write_json(report_path, payload)

    verification = quality.build_compact_matched_learning_quality_pair_verification_v1(
        matched_learning_quality_report_path=report_path,
        created_at="2026-05-30T00:00:00+00:00",
    )

    assert verification["schema_id"] == (
        quality.COMPACT_MATCHED_LEARNING_QUALITY_PAIR_VERIFICATION_SCHEMA_ID
    )
    assert verification["ok"] is True
    assert verification["status"] == "matched_pair_verified"
    assert verification["matched_pair"]["denominator_id"] == (
        "unit-matched-quality-denominator"
    )
    assert verification["denominator_count_checks"]["stock_reference"][
        "learner_train_calls_equal_replay_sample_calls"
    ] is True
    assert verification["denominator_count_checks"]["compact_candidate"][
        "rollout_rows_cover_sample_rows"
    ] is True
    assert verification["denominator_count_checks"]["compact_candidate"][
        "sample_rows_cover_sample_batches"
    ] is True
    assert verification["fingerprint_inventory"]["same_git_commit"] is True
    assert verification["non_claims"]["promotion_claim"] is False
    quality.validate_compact_matched_learning_quality_pair_verification_v1(verification)


def test_matched_pair_verifier_rejects_stock_count_drift(tmp_path):
    paths = _input_paths(tmp_path)
    stock_arm = _quality_arm(tmp_path, role="stock_reference")
    stock_arm["denominators"]["replay_sample_calls"] = 3
    payload = _build_compact_matched_learning_quality_canary_v1(
        tmp_path,
        run_id="unit-quality",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_arm=stock_arm,
        compact_candidate_arm=_quality_arm(tmp_path, role="compact_candidate"),
    )
    report_path = tmp_path / "matched_learning_quality_canary_report.json"
    _write_json(report_path, payload)

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="learner_train_calls must equal replay_sample_calls",
    ):
        quality.build_compact_matched_learning_quality_pair_verification_v1(
            matched_learning_quality_report_path=report_path,
        )


def test_matched_pair_verifier_rejects_compact_count_drift(tmp_path):
    paths = _input_paths(tmp_path)
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["denominators"]["compact_sample_rows"] = 3
    payload = _build_compact_matched_learning_quality_canary_v1(
        tmp_path,
        run_id="unit-quality",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_arm=_quality_arm(tmp_path, role="stock_reference"),
        compact_candidate_arm=compact_arm,
    )
    report_path = tmp_path / "matched_learning_quality_canary_report.json"
    _write_json(report_path, payload)

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="compact_sample_rows must cover sample_batch_count_delta",
    ):
        quality.build_compact_matched_learning_quality_pair_verification_v1(
            matched_learning_quality_report_path=report_path,
        )


def test_matched_pair_verifier_allows_multiple_compact_updates_per_sample_batch(tmp_path):
    paths = _input_paths(tmp_path)
    compact_arm = _quality_arm(tmp_path, role="compact_candidate")
    compact_arm["denominators"]["compact_sample_rows"] = 4
    compact_arm["denominators"]["sample_batch_count_delta"] = 4
    compact_arm["denominators"]["learner_update_count_delta"] = 8
    payload = _build_compact_matched_learning_quality_canary_v1(
        tmp_path,
        run_id="unit-quality",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_arm=_quality_arm(tmp_path, role="stock_reference"),
        compact_candidate_arm=compact_arm,
    )
    report_path = tmp_path / "matched_learning_quality_canary_report.json"
    _write_json(report_path, payload)

    verification = quality.build_compact_matched_learning_quality_pair_verification_v1(
        matched_learning_quality_report_path=report_path,
    )

    compact_checks = verification["denominator_count_checks"]["compact_candidate"]
    assert compact_checks["sample_rows_cover_sample_batches"] is True
    assert compact_checks["learner_updates_cover_sample_batches"] is True
    assert compact_checks["learner_updates_per_sample_batch"] == 2.0


def test_matched_pair_verifier_main_writes_report(tmp_path):
    module = _load_pair_verifier_module()
    paths = _input_paths(tmp_path)
    payload = _build_compact_matched_learning_quality_canary_v1(
        tmp_path,
        run_id="unit-quality",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_arm=_quality_arm(tmp_path, role="stock_reference"),
        compact_candidate_arm=_quality_arm(tmp_path, role="compact_candidate"),
    )
    matched_report_path = tmp_path / "matched_learning_quality_canary_report.json"
    output_root = tmp_path / "out"
    _write_json(matched_report_path, payload)

    assert (
        module.main(
            [
                "--run-id",
                "unit-pair-verifier-main",
                "--output-root",
                str(output_root),
                "--matched-learning-quality-report",
                str(matched_report_path),
            ]
        )
        == 0
    )

    verification_path = (
        output_root / "unit-pair-verifier-main" / "matched_pair_verification_report.json"
    )
    manifest_path = output_root / "unit-pair-verifier-main" / "manifest.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert manifest_path.is_file()
    assert verification["status"] == "matched_pair_verified"
    quality.validate_compact_matched_learning_quality_pair_verification_v1(verification)


def _load_pair_verifier_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_matched_learning_quality_pair_verifier.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_learning_quality_pair_verifier_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_compact_matched_learning_quality_canary_v1(
    tmp_path,
    *,
    run_id,
    compatibility_report_path,
    unified_lifecycle_report_path,
    stock_reference_arm,
    compact_candidate_arm,
    created_at=None,
):
    return quality.build_compact_matched_learning_quality_canary_v1(
        run_id=run_id,
        compatibility_report_path=compatibility_report_path,
        unified_lifecycle_report_path=unified_lifecycle_report_path,
        stock_reference_arm=stock_reference_arm,
        compact_candidate_arm=compact_candidate_arm,
        input_capture_files=_input_capture_files(
            tmp_path,
            stock_reference_arm=stock_reference_arm,
            compact_candidate_arm=compact_candidate_arm,
        ),
        created_at=created_at,
    )


def _input_capture_files(
    tmp_path,
    *,
    stock_reference_arm,
    compact_candidate_arm,
):
    stock_path = tmp_path / "input_captures" / "stock_reference_capture.json"
    compact_path = tmp_path / "input_captures" / "compact_candidate_capture.json"
    _write_json(stock_path, _capture_from_arm(stock_reference_arm))
    _write_json(compact_path, _capture_from_arm(compact_candidate_arm))
    return {
        "stock_reference_capture": {
            "path": str(stock_path),
            "sha256": quality._file_sha256(stock_path),
        },
        "compact_candidate_capture": {
            "path": str(compact_path),
            "sha256": quality._file_sha256(compact_path),
        },
    }


def _capture_from_arm(arm):
    is_stock = arm["role"] == "stock_reference"
    model_identity = {
        "initial_model_state_digest": arm["initial_model_state_digest"],
        "final_model_state_digest": arm["final_model_state_digest"],
        "model_state_digest_changed": arm["model_state_digest_changed"],
    }
    if not is_stock:
        model_identity["model_identity_scope"] = arm["model_identity_scope"]
    capture = {
        "schema_id": "curvyzero_compact_matched_learning_quality_capture/v1",
        "ok": True,
        "status": arm["status"],
        "role": arm["role"],
        "route": arm["route"],
        "profile_only": arm["profile_only"],
        "touches_live_runs": arm["touches_live_runs"],
        "candidate_checkpoint_id": arm["candidate_checkpoint_id"],
        "capture_id": arm["arm_id"],
        "run_id": arm["run_id"],
        "denominator_id": arm["denominator_id"],
        "quality_horizon": arm["quality_horizon"],
        "hardware_class": arm["hardware_class"],
        "source_fingerprint": arm["source_fingerprint"],
        "model_identity": model_identity,
        "eval_settings": arm["eval_settings"],
        "eval_points": arm["quality_curve"],
        "denominators": arm["denominators"],
        "artifact_refs": arm["artifact_refs"],
        "capture_provenance": _capture_provenance(arm["role"]),
        "non_claims": arm["non_claims"],
    }
    if is_stock:
        capture["called_train_muzero"] = bool(arm["calls_train_muzero"])
    else:
        capture["calls_train_muzero"] = bool(arm["calls_train_muzero"])
        capture["real_compact_owned_training_work"] = bool(arm["real_compact_owned_training_work"])
    return copy.deepcopy(capture)


def _capture_provenance(role):
    is_stock = role == "stock_reference"
    provenance = {
        "schema_id": "curvyzero_compact_matched_learning_quality_capture_provenance/v1",
        "role": role,
        "producer_id": f"unit-{role}-producer",
        "producer_ran_training": True,
        "producer_ran_pre_eval": True,
        "producer_ran_post_eval": True,
        "feeds_builder": True,
        "support_only": False,
        "training_artifact_ref": "training_artifact",
        "pre_eval_artifact_ref": "pre_eval_summary",
        "post_eval_artifact_ref": "post_eval_summary",
    }
    if is_stock:
        provenance["stock_train_muzero"] = {
            "called_train_muzero": True,
            "train_muzero_entrypoint": "lzero.entry.train_muzero",
        }
    else:
        provenance["compact_owned_training"] = {
            "calls_train_muzero": False,
            "real_compact_owned_training_work": True,
            "compact_training_entrypoint": (
                "curvyzero.training.compact_owned_trainer."
                "CompactOwnedTrainerV1.record_step"
            ),
        }
    return provenance


def _input_paths(tmp_path):
    compatibility_path = tmp_path / "compatibility_report.json"
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    _write_json(
        compatibility_path,
        {
            "schema_id": "curvyzero_compact_coach_compatibility_refresh/v1",
            "ok": True,
            "candidate_checkpoint_id": "unit-compact-ckpt",
            "promotion_eligible": True,
            "promotion_claim": False,
            "touches_live_runs": False,
            "calls_train_muzero": False,
            "coach_speed_row_gate": True,
        },
    )
    _write_json(
        lifecycle_path,
        {
            "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
            "ok": True,
            "checkpoint_id": "unit-compact-ckpt",
            "promotion_claim": False,
        },
    )
    return {"compatibility": compatibility_path, "lifecycle": lifecycle_path}


def _quality_arm(tmp_path, *, role):
    artifact_paths = _arm_artifact_paths(tmp_path, role=role)
    is_stock = role == "stock_reference"
    base = {
        "schema_id": "curvyzero_compact_matched_learning_quality_arm/v1",
        "ok": True,
        "status": "complete",
        "role": role,
        "route": ("stock_train_muzero_reference" if is_stock else "compact_owned_trainer"),
        "row_purpose": "matched_learning_quality_canary",
        "profile_only": False,
        "quality_currency": "mean_survival_delta",
        "calls_train_muzero": bool(is_stock),
        "touches_live_runs": False,
        "candidate_checkpoint_id": "unit-compact-ckpt",
        "arm_id": f"unit-{role}",
        "run_id": f"unit-{role}-run",
        "denominator_id": "unit-matched-quality-denominator",
        "quality_horizon": "pre_post_100_env_steps",
        "hardware_class": "local-test",
        "source_fingerprint": _source_fingerprint(role),
        "initial_model_state_digest": f"{role}-initial",
        "final_model_state_digest": f"{role}-final",
        "model_state_digest_changed": True,
        "eval_settings": {
            "observation_schema_id": "curvytron_source_state_v1",
            "policy_observation_backend": "cpu_oracle",
            "eval_seed_set": [101, 102],
            "eval_seed_rng_seed": 20260833,
            "eval_episode_count": 2,
            "source_max_steps": 2000,
            "eval_max_steps": 2000,
            "num_simulations": 8,
            "batch_size": 2,
            "reward_variant": "survival_plus_bonus_no_outcome",
            "reward_target_effect": "extrinsic_reward_only",
            "death_mode": "normal",
            "terminal_target_mode": "stock_terminal_no_bootstrap_return_discount_1.0",
            "root_noise": 0.0,
            "dirichlet_alpha": 0.3,
            "policy_noise": 0.0,
            "rnd_enabled": False,
            "exploration_bonus_mode": "none",
            "opponent_policy_ref": "fixed-anchor",
            "opponent_policy_kind": "fixed_anchor",
            "opponent_runtime_mode": "source_state",
            "opponent_death_mode": "normal",
            "natural_bonus_spawn": False,
            "training_seed_policy": "fixed",
            "initialization_source": "same_repo_state",
            "num_unroll_steps": 2,
            "td_steps": 2,
            "discount": 1.0,
            "support_scale": 300,
        },
        "quality_curve": [
            {
                "point_id": "pre_train",
                "checkpoint_ref": f"{role}-pre",
                "checkpoint_step": 0,
                "eval_episode_count": 2,
                "evaluator_eval_calls": 1,
                "mean_survival": 10.0,
                "median_survival": 10.0,
                "min_survival": 9.0,
                "max_survival": 11.0,
                "terminal_rate": 1.0,
                "cap_rate": 0.0,
                "death_rate": 1.0,
            },
            {
                "point_id": "post_train",
                "checkpoint_ref": f"{role}-post",
                "checkpoint_step": 100,
                "eval_episode_count": 2,
                "evaluator_eval_calls": 1,
                "mean_survival": 12.0 if is_stock else 13.0,
                "median_survival": 12.0 if is_stock else 13.0,
                "min_survival": 11.0 if is_stock else 12.0,
                "max_survival": 13.0 if is_stock else 14.0,
                "terminal_rate": 1.0,
                "cap_rate": 0.0,
                "death_rate": 1.0,
            },
        ],
        "artifact_refs": [
            _artifact_ref(kind, path) for kind, path in sorted(artifact_paths.items())
        ],
        "non_claims": _non_claims(),
        **_non_claims(),
    }
    if is_stock:
        base["denominators"] = {
            "denominator_currency": "stock_train_muzero_learning_quality",
            "env_step_currency": "stock_train_muzero_raw_env_steps",
            "speed_currency": "stock_train_muzero_raw_env_steps_per_sec",
            "collector_envstep_delta": 100,
            "learner_train_calls": 4,
            "replay_sample_calls": 4,
            "checkpoint_iteration_delta": 1,
            "training_wall_sec": 10.0,
            "uses_fallback_denominator": False,
            "wall_sec_used_for_speed_claim": False,
        }
    else:
        base.update(
            {
                "real_compact_owned_training_work": True,
                "model_identity_scope": "candidate_loaded_checkpoint",
            }
        )
        base["denominators"] = {
            "denominator_currency": "compact_owned_learning_quality",
            "env_step_currency": "compact_owned_trainer_env_steps",
            "speed_currency": "compact_trainer_env_steps_per_sec",
            "learner_update_count_delta": 4,
            "sample_batch_count_delta": 4,
            "compact_rollout_rows": 100,
            "compact_sample_rows": 64,
            "training_wall_sec": 8.0,
            "learner_loss_mean": 0.5,
            "uses_fallback_denominator": False,
            "wall_sec_used_for_speed_claim": False,
        }
    return copy.deepcopy(base)


def _non_claims():
    return {
        "promotion_claim": False,
        "training_speedup_claim": False,
        "live_run_safety_claim": False,
        "stock_resume_claim": False,
        "stock_training_resume_claim": False,
        "rating_or_promotion_quality_claim": False,
        "compact_quality_superiority_claim": False,
        "leaderboard_claim": False,
        "touches_live_runs": False,
    }


def _source_fingerprint(role):
    return {
        "git_commit": "unit-commit",
        "git_status_dirty": True,
        "producer_script": f"unit-{role}-producer.py",
        "producer_route": f"unit-{role}-route",
        "image_digest": "unit-image",
        "matched_surface": {
            "env_variant": "source_state_fixed_opponent",
            "reward_variant": "survival_plus_bonus_no_outcome",
            "policy_observation_backend": "cpu_oracle",
            "opponent_policy_kind": "fixed_anchor",
            "eval_seed_set": [101, 102],
            "eval_seed_rng_seed": 20260833,
        },
    }


def _arm_artifact_paths(tmp_path, *, role):
    root = tmp_path / f"{role}_arm_artifacts"
    paths = {
        "training_artifact": root / "training_artifact.json",
        "pre_eval_summary": root / "pre_eval_summary.json",
        "post_eval_summary": root / "post_eval_summary.json",
        "initial_checkpoint": root / "initial_checkpoint.pt",
        "final_checkpoint": root / "final_checkpoint.pt",
    }
    _write_json(paths["initial_checkpoint"], {"kind": "initial_checkpoint", "role": role})
    _write_json(paths["final_checkpoint"], {"kind": "final_checkpoint", "role": role})
    _write_json(paths["training_artifact"], _training_artifact_for_arm(role, paths))
    _write_json(paths["pre_eval_summary"], _eval_summary_for_arm(role, point="pre"))
    _write_json(paths["post_eval_summary"], _eval_summary_for_arm(role, point="post"))
    return paths


def _training_artifact_for_arm(role, paths):
    is_stock = role == "stock_reference"
    base = {
        "schema_id": (
            quality.COMPACT_MATCHED_LEARNING_QUALITY_STOCK_TRAINING_ARTIFACT_SCHEMA_ID
            if is_stock
            else quality.COMPACT_MATCHED_LEARNING_QUALITY_COMPACT_TRAINING_ARTIFACT_SCHEMA_ID
        ),
        "ok": True,
        "role": role,
        "route": "stock_train_muzero_reference" if is_stock else "compact_owned_trainer",
        "run_id": f"unit-{role}-run",
        "profile_only": False,
        "support_only": False,
        "touches_live_runs": False,
        "model_identity": {
            "initial_model_state_digest": f"{role}-initial",
            "final_model_state_digest": f"{role}-final",
            "model_state_digest_changed": True,
        },
    }
    if is_stock:
        base.update(
            {
                "called_train_muzero": True,
                "train_muzero_entrypoint": "lzero.entry.train_muzero",
                "checkpoint_selection": {
                    "initial_ref": f"{role}-pre",
                    "final_ref": f"{role}-post",
                },
                "downloaded_checkpoints": {
                    "initial_path": str(paths["initial_checkpoint"]),
                    "initial_sha256": quality._file_sha256(paths["initial_checkpoint"]),
                    "final_path": str(paths["final_checkpoint"]),
                    "final_sha256": quality._file_sha256(paths["final_checkpoint"]),
                },
            }
        )
    else:
        base["model_identity"]["model_identity_scope"] = "candidate_loaded_checkpoint"
        base.update(
            {
                "calls_train_muzero": False,
                "real_compact_owned_training_work": True,
                "compact_training_entrypoint": (
                    "curvyzero.training.compact_owned_trainer."
                    "CompactOwnedTrainerV1.record_step"
                ),
                "sample_source": "compact_env_search_replay_rows",
                "synthetic_resident_sample": False,
                "real_env_search_replay_rows": True,
                "initial_checkpoint_path": str(paths["initial_checkpoint"]),
                "final_checkpoint_path": str(paths["final_checkpoint"]),
                "pre_eval_checkpoint_ref": f"{role}-pre",
                "post_eval_checkpoint_ref": f"{role}-post",
            }
        )
    return base


def _eval_summary_for_arm(role, *, point):
    is_stock = role == "stock_reference"
    is_pre = point == "pre"
    mean = 10.0 if is_pre else (12.0 if is_stock else 13.0)
    return {
        "checkpoint": f"{role}-{point}",
        "checkpoint_step": 0 if is_pre else 100,
        "eval_seed_set": [101, 102],
        "eval_max_steps": 2000,
        "eval_episode_count": 2,
        "evaluator_eval_calls": 1,
        "mean_steps": mean,
        "median_steps": mean,
        "min_steps": mean - 1.0,
        "max_steps": mean + 1.0,
        "ok_count": 2,
        "capped_count": 0,
        "failure_count": 0,
        "outcome_histogram": {"loss": 2},
    }


def _artifact_ref(kind, path):
    return {
        "kind": kind,
        "path": str(path),
        "sha256": quality._file_sha256(path),
        "required": True,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
