from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

from curvyzero.training import compact_matched_quality_larger_study_bundle as bundle
from curvyzero.training import compact_matched_quality_larger_study_preflight as preflight
from curvyzero.training import compact_matched_quality_sufficiency_review as review


def test_larger_study_bundle_builds_ordered_plan_only_manifest(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    sufficiency_path = _write_sufficiency_review(tmp_path)

    payload = bundle.build_compact_matched_quality_larger_study_bundle_v1(
        run_id="unit-larger-study-bundle",
        sufficiency_review_report_path=sufficiency_path,
        repo_root=tmp_path,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert (
        payload["schema_id"]
        == bundle.COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_SCHEMA_ID
    )
    assert payload["execution_contract"]["plan_only_not_evidence"] is True
    assert payload["execution_contract"]["commands_run_by_manifest"] is False
    assert payload["execution_contract"]["fresh_outputs_produced"] is False
    assert payload["non_claims"]["promotion_claim"] is False
    assert payload["non_claims"]["stock_train_muzero_speedup_claim"] is False
    assert [step["step_key"] for step in payload["ordered_steps"]] == list(
        bundle.STEP_ORDER
    )
    assert (
        payload["ordered_steps"][0]["post_step_validator"]["expected_role"]
        == "stock_reference"
    )
    stock_step = payload["ordered_steps"][0]
    assert (
        stock_step["preflight_checks"]["stock_evaluator_requirement_explicit"]
        is True
    )
    assert stock_step["stock_evaluator_requirement"] == {
        "policy": "evalenv_full_episode_surface",
        "eval_seed_count": 32,
        "evaluator_env_num": 32,
        "n_evaluator_episode": 32,
        "env_num_matches_episode_count_required": True,
        "empty_ready_set_workaround_formalized": True,
        "canonical_stock_evaluator_required": False,
    }
    assert (
        payload["ordered_steps"][2]["post_step_validator"]["validator"]
        == "validate_compact_matched_learning_quality_canary_v1"
    )
    bundle.validate_compact_matched_quality_larger_study_bundle_v1(payload)


def test_larger_study_bundle_rejects_source_promotion_claim(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    sufficiency_path = _write_sufficiency_review(tmp_path)
    sufficiency = _read_json(sufficiency_path)
    sufficiency["decision"]["promotion_claim"] = True
    _write_json(sufficiency_path, sufficiency)

    with pytest.raises(
        bundle.CompactMatchedQualityLargerStudyBundleError,
        match="promotion_claim must be false",
    ):
        bundle.build_compact_matched_quality_larger_study_bundle_v1(
            run_id="unit-larger-study-bundle",
            sufficiency_review_report_path=sufficiency_path,
            repo_root=tmp_path,
        )


def test_larger_study_bundle_rejects_existing_future_output(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    sufficiency_path = _write_sufficiency_review(tmp_path)
    sufficiency = _read_json(sufficiency_path)
    output = (
        tmp_path
        / sufficiency["larger_study_plan"]["required_outputs"][
            "stock_reference_capture"
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("already here\n", encoding="utf-8")

    with pytest.raises(
        bundle.CompactMatchedQualityLargerStudyBundleError,
        match="expected output already exists",
    ):
        bundle.build_compact_matched_quality_larger_study_bundle_v1(
            run_id="unit-larger-study-bundle",
            sufficiency_review_report_path=sufficiency_path,
            repo_root=tmp_path,
        )


def test_larger_study_bundle_rejects_missing_stock_evaluator_surface(
    monkeypatch,
    tmp_path,
):
    _patch_sufficiency_validator(monkeypatch)
    sufficiency_path = _write_sufficiency_review(tmp_path)
    sufficiency = _read_json(sufficiency_path)
    argv = sufficiency["larger_study_plan"]["planned_runs"][
        "stock_reference_capture_producer"
    ]["argv"]
    index = argv.index("--evaluator-env-num")
    del argv[index : index + 2]
    _write_json(sufficiency_path, sufficiency)

    with pytest.raises(
        bundle.CompactMatchedQualityLargerStudyBundleError,
        match="argv missing --evaluator-env-num",
    ):
        bundle.build_compact_matched_quality_larger_study_bundle_v1(
            run_id="unit-larger-study-bundle",
            sufficiency_review_report_path=sufficiency_path,
            repo_root=tmp_path,
        )


def test_larger_study_bundle_rejects_source_hash_drift(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    sufficiency_path = _write_sufficiency_review(tmp_path)
    payload = bundle.build_compact_matched_quality_larger_study_bundle_v1(
        run_id="unit-larger-study-bundle",
        sufficiency_review_report_path=sufficiency_path,
        repo_root=tmp_path,
    )
    sufficiency = _read_json(sufficiency_path)
    sufficiency["run_id"] = "tampered"
    _write_json(sufficiency_path, sufficiency)

    with pytest.raises(
        bundle.CompactMatchedQualityLargerStudyBundleError,
        match="source sufficiency review sha256 mismatch",
    ):
        bundle.validate_compact_matched_quality_larger_study_bundle_v1(payload)


def test_larger_study_bundle_cli_writes_report_and_manifest(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    module = _load_cli_module()
    sufficiency_path = _write_sufficiency_review(tmp_path / "inputs")
    output_root = tmp_path / "out"
    run_id = "unit-larger-study-bundle-cli"
    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--sufficiency-review-report",
        str(sufficiency_path),
    ]

    assert module.main(argv) == 0
    report_path = output_root / run_id / "larger_study_bundle_report.json"
    manifest_path = output_root / run_id / "manifest.json"
    payload = _read_json(report_path)
    manifest = _read_json(manifest_path)
    assert manifest["schema_id"] == module.MANIFEST_SCHEMA_ID
    assert manifest["plan_only_not_evidence"] is True
    assert manifest["future_required_outputs"] == payload["future_required_outputs"]
    bundle.validate_compact_matched_quality_larger_study_bundle_v1(payload)

    with pytest.raises(FileExistsError):
        module.main(argv)


def test_larger_study_preflight_marks_initial_producers_ready(monkeypatch, tmp_path):
    _patch_sufficiency_validator(monkeypatch)
    sufficiency_path = _write_sufficiency_review(tmp_path)
    bundle_payload = bundle.build_compact_matched_quality_larger_study_bundle_v1(
        run_id="unit-larger-study-bundle",
        sufficiency_review_report_path=sufficiency_path,
        repo_root=tmp_path,
    )
    bundle_path = tmp_path / "larger_study_bundle_report.json"
    _write_json(bundle_path, bundle_payload)

    payload = preflight.build_compact_matched_quality_larger_study_preflight_v1(
        run_id="unit-larger-study-preflight",
        bundle_report_path=bundle_path,
        repo_root=tmp_path,
        created_at="2026-05-31T00:00:00+00:00",
    )

    assert payload["execution_readiness"]["commands_run_by_preflight"] is False
    assert payload["execution_readiness"]["quality_evidence_produced"] is False
    assert payload["execution_readiness"]["recommended_first_wave"] == [
        "stock_reference_capture_producer",
        "compact_candidate_capture_producer",
    ]
    by_key = {row["step_key"]: row for row in payload["step_preflights"]}
    assert by_key["stock_reference_capture_producer"]["executable_now"] is True
    assert (
        payload["execution_readiness"]["stock_evaluator_requirement_explicit"]
        is True
    )
    assert by_key["stock_reference_capture_producer"][
        "stock_evaluator_requirement_status"
    ]["matches_source_requirement"] is True
    assert by_key["compact_candidate_capture_producer"]["executable_now"] is True
    assert by_key["matched_canary_builder"]["executable_now"] is False
    assert by_key["matched_canary_builder"]["missing_required_prior_outputs"] == [
        "stock_reference_capture",
        "compact_candidate_capture",
    ]
    assert payload["non_claims"]["promotion_claim"] is False
    assert payload["non_claims"]["quality_evidence_produced"] is False
    preflight.validate_compact_matched_quality_larger_study_preflight_v1(payload)


def test_larger_study_preflight_rejects_source_bundle_hash_drift(
    monkeypatch,
    tmp_path,
):
    _patch_sufficiency_validator(monkeypatch)
    sufficiency_path = _write_sufficiency_review(tmp_path)
    bundle_payload = bundle.build_compact_matched_quality_larger_study_bundle_v1(
        run_id="unit-larger-study-bundle",
        sufficiency_review_report_path=sufficiency_path,
        repo_root=tmp_path,
    )
    bundle_path = tmp_path / "larger_study_bundle_report.json"
    _write_json(bundle_path, bundle_payload)
    payload = preflight.build_compact_matched_quality_larger_study_preflight_v1(
        run_id="unit-larger-study-preflight",
        bundle_report_path=bundle_path,
        repo_root=tmp_path,
    )
    bundle_payload["run_id"] = "tampered"
    _write_json(bundle_path, bundle_payload)

    with pytest.raises(
        preflight.CompactMatchedQualityLargerStudyPreflightError,
        match="source bundle report sha256 mismatch",
    ):
        preflight.validate_compact_matched_quality_larger_study_preflight_v1(payload)


def test_larger_study_preflight_rejects_implicit_readiness_defaults(
    monkeypatch,
    tmp_path,
):
    _patch_sufficiency_validator(monkeypatch)
    sufficiency_path = _write_sufficiency_review(tmp_path)
    sufficiency = _read_json(sufficiency_path)
    argv = sufficiency["larger_study_plan"]["planned_runs"][
        "readiness_bundle_refresh"
    ]["argv"]
    for flag in (
        "--stock-resume-load-canary-report",
        "--isolated-live-run-safety-canary-report",
        "--sandbox-assignment-rating-proof-report",
        "--longer-horizon-learning-metrics-report",
    ):
        index = argv.index(flag)
        del argv[index : index + 2]
    _write_json(sufficiency_path, sufficiency)
    bundle_payload = bundle.build_compact_matched_quality_larger_study_bundle_v1(
        run_id="unit-larger-study-bundle",
        sufficiency_review_report_path=sufficiency_path,
        repo_root=tmp_path,
    )
    bundle_path = tmp_path / "larger_study_bundle_report.json"
    _write_json(bundle_path, bundle_payload)

    with pytest.raises(
        preflight.CompactMatchedQualityLargerStudyPreflightError,
        match="implicit default inputs",
    ):
        preflight.build_compact_matched_quality_larger_study_preflight_v1(
            run_id="unit-larger-study-preflight",
            bundle_report_path=bundle_path,
            repo_root=tmp_path,
        )


def test_larger_study_preflight_cli_writes_report_and_manifest(
    monkeypatch,
    tmp_path,
):
    _patch_sufficiency_validator(monkeypatch)
    module = _load_preflight_cli_module()
    sufficiency_path = _write_sufficiency_review(tmp_path / "inputs")
    bundle_payload = bundle.build_compact_matched_quality_larger_study_bundle_v1(
        run_id="unit-larger-study-bundle",
        sufficiency_review_report_path=sufficiency_path,
        repo_root=tmp_path,
    )
    bundle_path = tmp_path / "larger_study_bundle_report.json"
    _write_json(bundle_path, bundle_payload)
    output_root = tmp_path / "out"
    run_id = "unit-larger-study-preflight-cli"
    argv = [
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--bundle-report",
        str(bundle_path),
    ]

    assert module.main(argv) == 0
    report_path = output_root / run_id / "larger_study_preflight_report.json"
    manifest_path = output_root / run_id / "manifest.json"
    payload = _read_json(report_path)
    manifest = _read_json(manifest_path)
    assert manifest["schema_id"] == module.MANIFEST_SCHEMA_ID
    assert manifest["commands_run_by_preflight"] is False
    assert manifest["recommended_first_wave"] == [
        "stock_reference_capture_producer",
        "compact_candidate_capture_producer",
    ]
    preflight.validate_compact_matched_quality_larger_study_preflight_v1(payload)

    with pytest.raises(FileExistsError):
        module.main(argv)


def _patch_sufficiency_validator(monkeypatch) -> None:
    monkeypatch.setattr(
        bundle,
        "validate_compact_matched_quality_sufficiency_review_v1",
        lambda payload: None,
    )


def _write_sufficiency_review(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate = "unit-compact-ckpt"
    readiness_path = tmp_path / "readiness_bundle_review_report.json"
    current_reports = {
        "matched_learning_quality_canary": _write_ref_file(
            tmp_path,
            "matched_learning_quality_canary_report.json",
        ),
        "matched_pair_verification": _write_ref_file(
            tmp_path,
            "matched_pair_verification_report.json",
        ),
        "longer_horizon_compact_learning_metrics": _write_ref_file(
            tmp_path,
            "longer_horizon_learning_metrics_report.json",
        ),
        "stock_reference_capture": _write_ref_file(
            tmp_path,
            "current_stock_reference_capture.json",
        ),
        "compact_candidate_capture": _write_ref_file(
            tmp_path,
            "current_compact_candidate_capture.json",
        ),
    }
    readiness_inputs = {
        "compatibility_refresh": _write_ref_file(tmp_path, "compatibility_report.json"),
        "unified_lifecycle": _write_ref_file(tmp_path, "unified_lifecycle_report.json"),
        "stock_resume_load_canary": _write_ref_file(
            tmp_path,
            "stock_resume_load_canary_report.json",
        ),
        "isolated_live_run_safety_canary": _write_ref_file(
            tmp_path,
            "isolated_live_run_safety_canary_report.json",
        ),
        "sandbox_assignment_rating_proof": _write_ref_file(
            tmp_path,
            "sandbox_assignment_rating_proof_report.json",
        ),
        "longer_horizon_compact_learning_metrics": current_reports[
            "longer_horizon_compact_learning_metrics"
        ],
        "matched_learning_quality_canary": current_reports[
            "matched_learning_quality_canary"
        ],
        "matched_pair_verification": current_reports["matched_pair_verification"],
    }
    readiness = {
        "schema_id": bundle.COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID,
        "ok": True,
        "status": "local_bundle_reviewed_no_promotion",
        "candidate_checkpoint_id": candidate,
        "evidence_ref": "unit-readiness-ref",
        "input_reports": readiness_inputs,
    }
    _write_json(readiness_path, readiness)
    readiness_ref = _report_ref(readiness_path, readiness)
    larger_plan = _larger_study_plan(
        candidate=candidate,
        compatibility_path=readiness_inputs["compatibility_refresh"]["path"],
        lifecycle_path=readiness_inputs["unified_lifecycle"]["path"],
    )
    sufficiency = {
        "schema_id": bundle.COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_SCHEMA_ID,
        "ok": True,
        "status": review.STATUS_LARGER_SAME_SURFACE_STUDY_REQUIRED,
        "run_id": "unit-sufficiency-review",
        "candidate_checkpoint_id": candidate,
        "evidence_ref": "unit-sufficiency-ref",
        "decision": {
            "matched_quality_sufficiency_decision": (
                review.DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY
            ),
            "larger_same_surface_study_required": True,
            "current_evidence_sufficient_for_promotion": False,
            "promotion_claim": False,
            "automatic_promotion_allowed": False,
        },
        "input_reports": {
            "readiness_bundle_review": readiness_ref,
            **current_reports,
        },
        "current_evidence_summary": {
            "eval_seed_count": 8,
            "eval_max_steps": 1024,
            "matched_surface": {"num_simulations": 1, "batch_size": 2},
        },
        "larger_study_plan": larger_plan,
    }
    path = tmp_path / "matched_quality_sufficiency_review_report.json"
    _write_json(path, sufficiency)
    return path


def _larger_study_plan(
    *,
    candidate: str,
    compatibility_path: str,
    lifecycle_path: str,
) -> dict[str, object]:
    outputs = {
        "stock_reference_capture": (
            "artifacts/local/unit-larger-stock/stock_reference_capture.json"
        ),
        "compact_candidate_capture": (
            "artifacts/local/unit-larger-compact/compact_candidate_capture.json"
        ),
        "matched_learning_quality_canary_report": (
            "artifacts/local/unit-larger-canary/matched_learning_quality_canary_report.json"
        ),
        "matched_pair_verification_report": (
            "artifacts/local/unit-larger-pair/matched_pair_verification_report.json"
        ),
        "refreshed_readiness_bundle_review": (
            "artifacts/local/unit-larger-bundle/readiness_bundle_review_report.json"
        ),
        "updated_sufficiency_review": (
            "artifacts/local/unit-larger-review/"
            "matched_quality_sufficiency_review_report.json"
        ),
    }
    planned = {
        "stock_reference_capture_producer": _run(
            "unit-stock-larger",
            [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_stock_reference_producer.py",
                "--run-id",
                "unit-stock-larger",
                "--candidate-checkpoint-id",
                candidate,
                "--eval-seed-count",
                "32",
                "--eval-steps",
                "2048",
                "--eval-seed-rng-seed",
                "20260833",
                "--evaluator-env-num",
                "32",
                "--n-evaluator-episode",
                "32",
                "--max-env-step",
                "2048",
                "--max-train-iter",
                "4",
                "--denominator-id",
                "matched_quality_larger_2048x32-env64train8_denominator_v1",
                "--quality-horizon",
                "matched_learning_quality_pre_post_eval_larger_2048x32-env64train8",
            ],
        ),
        "compact_candidate_capture_producer": _run(
            "unit-compact-larger",
            [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_compact_candidate_producer.py",
                "--run-id",
                "unit-compact-larger",
                "--unified-lifecycle-report",
                lifecycle_path,
                "--candidate-checkpoint-id",
                candidate,
                "--eval-seed-count",
                "32",
                "--eval-steps",
                "2048",
                "--eval-seed-rng-seed",
                "20260833",
                "--compact-env-steps",
                "64",
                "--train-steps",
                "8",
                "--compact-sample-batch-size",
                "4",
                "--compact-replay-pair-capacity",
                "128",
                "--compact-training-mode",
                "env_search_replay",
                "--learner-device",
                "auto",
                "--denominator-id",
                "matched_quality_larger_2048x32-env64train8_denominator_v1",
                "--quality-horizon",
                "matched_learning_quality_pre_post_eval_larger_2048x32-env64train8",
            ],
        ),
        "matched_canary_builder": _run(
            "unit-canary-larger",
            [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_canary.py",
                "--run-id",
                "unit-canary-larger",
                "--compatibility-report",
                compatibility_path,
                "--unified-lifecycle-report",
                lifecycle_path,
                "--stock-reference-capture",
                outputs["stock_reference_capture"],
                "--compact-candidate-capture",
                outputs["compact_candidate_capture"],
            ],
        ),
        "matched_pair_verifier": _run(
            "unit-pair-larger",
            [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_pair_verifier.py",
                "--run-id",
                "unit-pair-larger",
                "--matched-learning-quality-report",
                outputs["matched_learning_quality_canary_report"],
            ],
        ),
        "readiness_bundle_refresh": _run(
            "unit-bundle-larger",
            [
                "uv",
                "run",
                "python",
                "scripts/build_compact_promotion_readiness_bundle_review.py",
                "--run-id",
                "unit-bundle-larger",
                "--compatibility-report",
                compatibility_path,
                "--unified-lifecycle-report",
                lifecycle_path,
                "--matched-learning-quality-report",
                outputs["matched_learning_quality_canary_report"],
                "--matched-pair-verification-report",
                outputs["matched_pair_verification_report"],
                "--stock-resume-load-canary-report",
                str(Path(compatibility_path).with_name("stock_resume_load_canary_report.json")),
                "--isolated-live-run-safety-canary-report",
                str(Path(compatibility_path).with_name("isolated_live_run_safety_canary_report.json")),
                "--sandbox-assignment-rating-proof-report",
                str(Path(compatibility_path).with_name("sandbox_assignment_rating_proof_report.json")),
                "--longer-horizon-learning-metrics-report",
                str(Path(compatibility_path).with_name("longer_horizon_learning_metrics_report.json")),
            ],
        ),
        "sufficiency_review_update": _run(
            "unit-review-larger",
            [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_quality_sufficiency_review.py",
                "--run-id",
                "unit-review-larger",
                "--readiness-bundle-review",
                outputs["refreshed_readiness_bundle_review"],
                "--sufficiency-decision",
                review.DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY,
                "--larger-study-id",
                "unit-larger-study",
                "--larger-denominator-id",
                "matched_quality_larger_2048x32-env64train8_denominator_v1",
                "--larger-quality-horizon",
                "matched_learning_quality_pre_post_eval_larger_2048x32-env64train8",
            ],
        ),
    }
    return {
        "schema_id": "curvyzero_compact_matched_learning_quality_study_plan/v1",
        "study_id": "unit-larger-study",
        "candidate_checkpoint_id": candidate,
        "same_surface_required": True,
        "promotion_claim": False,
        "automatic_promotion_allowed": False,
        "matched_surface": {"num_simulations": 1, "batch_size": 2},
        "minimum_scale_over_current": {
            "min_eval_seed_count": 32,
            "min_eval_max_steps": 2048,
            "stock_reference_min_max_env_step": 2048,
            "stock_reference_min_max_train_iter": 4,
            "compact_candidate_min_env_steps": 64,
            "compact_candidate_min_train_steps": 8,
        },
        "stock_evaluator_requirement": {
            "policy": "evalenv_full_episode_surface",
            "eval_seed_count": 32,
            "evaluator_env_num": 32,
            "n_evaluator_episode": 32,
            "env_num_matches_episode_count_required": True,
            "empty_ready_set_workaround_formalized": True,
            "canonical_stock_evaluator_required": False,
        },
        "disallowed_shortcuts": {
            key: True for key in review.REQUIRED_DISALLOWED_SHORTCUT_KEYS
        },
        "planned_runs": planned,
        "required_outputs": outputs,
    }


def _run(run_id: str, argv: list[str]) -> dict[str, object]:
    return {"run_id": run_id, "argv": argv}


def _write_ref_file(tmp_path: Path, name: str) -> dict[str, str]:
    path = tmp_path / name
    payload = {"ok": True, "name": name}
    _write_json(path, payload)
    return _report_ref(path, payload)


def _report_ref(path: Path, payload: object) -> dict[str, str]:
    return {"path": str(path), "sha256": _file_sha256(path)}


def _load_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_matched_quality_larger_study_bundle.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_quality_larger_study_bundle_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_preflight_cli_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_matched_quality_larger_study_preflight.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_quality_larger_study_preflight_for_test",
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


def _json_sha256(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
