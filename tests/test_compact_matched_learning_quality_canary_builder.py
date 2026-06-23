from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from curvyzero.training import compact_promotion_readiness_learning_quality as quality


def test_matched_learning_quality_builder_writes_valid_report(tmp_path):
    module = _load_builder_module()
    paths = _input_paths(tmp_path)
    stock_capture_path = tmp_path / "stock_reference_capture.json"
    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    _write_json(stock_capture_path, _quality_capture(tmp_path, role="stock_reference"))
    _write_json(compact_capture_path, _quality_capture(tmp_path, role="compact_candidate"))

    payload = module.build_compact_matched_learning_quality_canary_payload(
        run_id="unit-quality-builder",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_capture_path=stock_capture_path,
        compact_candidate_capture_path=compact_capture_path,
        created_at="2026-05-30T00:00:00+00:00",
    )

    assert payload["schema_id"] == quality.COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID
    assert payload["ok"] is True
    assert payload["input_capture_files"]["stock_reference_capture"]["path"] == str(
        stock_capture_path.resolve()
    )
    assert payload["input_capture_files"]["compact_candidate_capture"]["path"] == str(
        compact_capture_path.resolve()
    )
    assert payload["input_capture_files"]["stock_reference_capture"]["sha256"] == (
        _file_sha256(stock_capture_path)
    )
    assert payload["arms"]["stock_reference"]["schema_id"] == (
        quality.COMPACT_MATCHED_LEARNING_QUALITY_ARM_SCHEMA_ID
    )
    assert payload["quality_movement"]["compact_candidate_delta"] == pytest.approx(3.0)
    assert payload["non_claims"]["promotion_claim"] is False
    quality.validate_compact_matched_learning_quality_canary_v1(payload)


def test_matched_learning_quality_builder_main_writes_report(tmp_path):
    module = _load_builder_module()
    paths = _input_paths(tmp_path)
    stock_capture_path = tmp_path / "stock_reference_capture.json"
    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    output_root = tmp_path / "out"
    _write_json(stock_capture_path, _quality_capture(tmp_path, role="stock_reference"))
    _write_json(compact_capture_path, _quality_capture(tmp_path, role="compact_candidate"))

    assert (
        module.main(
            [
                "--run-id",
                "unit-quality-builder-main",
                "--output-root",
                str(output_root),
                "--compatibility-report",
                str(paths["compatibility"]),
                "--unified-lifecycle-report",
                str(paths["lifecycle"]),
                "--stock-reference-capture",
                str(stock_capture_path),
                "--compact-candidate-capture",
                str(compact_capture_path),
            ]
        )
        == 0
    )

    report_path = (
        output_root / "unit-quality-builder-main" / "matched_learning_quality_canary_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["readiness"]["matched_learning_quality_canary"] is True
    assert payload["readiness"]["promotion_readiness_complete"] is False
    assert (output_root / "unit-quality-builder-main" / "stock_reference_arm.json").is_file()
    assert (output_root / "unit-quality-builder-main" / "compact_candidate_arm.json").is_file()
    assert (output_root / "unit-quality-builder-main" / "manifest.json").is_file()
    quality.validate_compact_matched_learning_quality_canary_v1(payload)


def test_matched_learning_quality_builder_rejects_bad_capture_file(tmp_path):
    module = _load_builder_module()
    paths = _input_paths(tmp_path)
    stock_capture_path = tmp_path / "stock_reference_capture.json"
    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    stock_capture = _quality_capture(tmp_path, role="stock_reference")
    stock_capture["called_train_muzero"] = False
    _write_json(stock_capture_path, stock_capture)
    _write_json(compact_capture_path, _quality_capture(tmp_path, role="compact_candidate"))

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="stock_reference capture must call train_muzero",
    ):
        module.build_compact_matched_learning_quality_canary_payload(
            run_id="unit-quality-builder",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_capture_path=stock_capture_path,
            compact_candidate_capture_path=compact_capture_path,
        )


def test_matched_learning_quality_builder_rejects_capture_missing_provenance(tmp_path):
    module = _load_builder_module()
    paths = _input_paths(tmp_path)
    stock_capture_path = tmp_path / "stock_reference_capture.json"
    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    stock_capture = _quality_capture(tmp_path, role="stock_reference")
    del stock_capture["capture_provenance"]
    _write_json(stock_capture_path, stock_capture)
    _write_json(compact_capture_path, _quality_capture(tmp_path, role="compact_candidate"))

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="stock_reference capture_provenance",
    ):
        module.build_compact_matched_learning_quality_canary_payload(
            run_id="unit-quality-builder",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_capture_path=stock_capture_path,
            compact_candidate_capture_path=compact_capture_path,
        )


def test_matched_learning_quality_builder_rejects_support_only_capture(tmp_path):
    module = _load_builder_module()
    paths = _input_paths(tmp_path)
    stock_capture_path = tmp_path / "stock_reference_capture.json"
    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    compact_capture = _quality_capture(tmp_path, role="compact_candidate")
    compact_capture["capture_provenance"]["support_only"] = True
    _write_json(stock_capture_path, _quality_capture(tmp_path, role="stock_reference"))
    _write_json(compact_capture_path, compact_capture)

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="compact_candidate capture_provenance support_only",
    ):
        module.build_compact_matched_learning_quality_canary_payload(
            run_id="unit-quality-builder",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_capture_path=stock_capture_path,
            compact_candidate_capture_path=compact_capture_path,
        )


def test_matched_learning_quality_builder_rejects_missing_capture_artifact_kind(tmp_path):
    module = _load_builder_module()
    paths = _input_paths(tmp_path)
    stock_capture_path = tmp_path / "stock_reference_capture.json"
    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    stock_capture = _quality_capture(tmp_path, role="stock_reference")
    stock_capture["artifact_refs"] = [
        ref for ref in stock_capture["artifact_refs"] if ref["kind"] != "training_artifact"
    ]
    _write_json(stock_capture_path, stock_capture)
    _write_json(compact_capture_path, _quality_capture(tmp_path, role="compact_candidate"))

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="artifact_refs missing training_artifact",
    ):
        module.build_compact_matched_learning_quality_canary_payload(
            run_id="unit-quality-builder",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_capture_path=stock_capture_path,
            compact_candidate_capture_path=compact_capture_path,
        )


def test_matched_learning_quality_builder_rejects_unbound_provenance_ref(tmp_path):
    module = _load_builder_module()
    paths = _input_paths(tmp_path)
    stock_capture_path = tmp_path / "stock_reference_capture.json"
    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    compact_capture = _quality_capture(tmp_path, role="compact_candidate")
    compact_capture["capture_provenance"]["pre_eval_artifact_ref"] = "unbound-pre-eval"
    _write_json(stock_capture_path, _quality_capture(tmp_path, role="stock_reference"))
    _write_json(compact_capture_path, compact_capture)

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="pre_eval_artifact_ref must match artifact kind or path",
    ):
        module.build_compact_matched_learning_quality_canary_payload(
            run_id="unit-quality-builder",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_capture_path=stock_capture_path,
            compact_candidate_capture_path=compact_capture_path,
        )


def test_matched_learning_quality_report_rejects_capture_file_drift(tmp_path):
    module = _load_builder_module()
    paths = _input_paths(tmp_path)
    stock_capture_path = tmp_path / "stock_reference_capture.json"
    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    _write_json(stock_capture_path, _quality_capture(tmp_path, role="stock_reference"))
    _write_json(compact_capture_path, _quality_capture(tmp_path, role="compact_candidate"))

    payload = module.build_compact_matched_learning_quality_canary_payload(
        run_id="unit-quality-builder",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_capture_path=stock_capture_path,
        compact_candidate_capture_path=compact_capture_path,
    )
    with compact_capture_path.open("ab") as fh:
        fh.write(b"drift")

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="compact_candidate_capture sha256 mismatch",
    ):
        quality.validate_compact_matched_learning_quality_canary_v1(payload)


def test_capture_from_artifacts_writes_builder_input_captures_and_report(tmp_path):
    capture_module = _load_capture_from_artifacts_module()
    builder_module = _load_builder_module()
    paths = _input_paths(tmp_path)
    output_root = tmp_path / "out"
    stock_inputs = _capture_from_artifacts_inputs(tmp_path, role="stock_reference")
    compact_inputs = _capture_from_artifacts_inputs(tmp_path, role="compact_candidate")

    assert (
        capture_module.main(
            [
                "--role",
                "stock_reference",
                "--run-id",
                "unit-capture-from-artifacts",
                "--capture-id",
                "unit-stock-reference-capture",
                "--candidate-checkpoint-id",
                "unit-compact-ckpt",
                "--denominator-id",
                "unit-matched-quality-denominator",
                "--quality-horizon",
                "pre_post_100_env_steps",
                "--hardware-class",
                "local-test",
                "--source-fingerprint-json",
                str(stock_inputs["source_fingerprint"]),
                "--model-identity-json",
                str(stock_inputs["model_identity"]),
                "--eval-settings-json",
                str(stock_inputs["eval_settings"]),
                "--denominators-json",
                str(stock_inputs["denominators"]),
                "--capture-provenance-json",
                str(stock_inputs["capture_provenance"]),
                "--training-artifact",
                str(stock_inputs["training_artifact"]),
                "--pre-eval-summary",
                str(stock_inputs["pre_eval_summary"]),
                "--post-eval-summary",
                str(stock_inputs["post_eval_summary"]),
                "--initial-checkpoint",
                str(stock_inputs["initial_checkpoint"]),
                "--final-checkpoint",
                str(stock_inputs["final_checkpoint"]),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )
    assert (
        capture_module.main(
            [
                "--role",
                "compact_candidate",
                "--run-id",
                "unit-capture-from-artifacts",
                "--capture-id",
                "unit-compact-candidate-capture",
                "--candidate-checkpoint-id",
                "unit-compact-ckpt",
                "--denominator-id",
                "unit-matched-quality-denominator",
                "--quality-horizon",
                "pre_post_100_env_steps",
                "--hardware-class",
                "local-test",
                "--source-fingerprint-json",
                str(compact_inputs["source_fingerprint"]),
                "--model-identity-json",
                str(compact_inputs["model_identity"]),
                "--eval-settings-json",
                str(compact_inputs["eval_settings"]),
                "--denominators-json",
                str(compact_inputs["denominators"]),
                "--capture-provenance-json",
                str(compact_inputs["capture_provenance"]),
                "--training-artifact",
                str(compact_inputs["training_artifact"]),
                "--pre-eval-summary",
                str(compact_inputs["pre_eval_summary"]),
                "--post-eval-summary",
                str(compact_inputs["post_eval_summary"]),
                "--initial-checkpoint",
                str(compact_inputs["initial_checkpoint"]),
                "--final-checkpoint",
                str(compact_inputs["final_checkpoint"]),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    stock_capture_path = (
        output_root / "unit-capture-from-artifacts" / "stock_reference_capture.json"
    )
    compact_capture_path = (
        output_root / "unit-capture-from-artifacts" / "compact_candidate_capture.json"
    )
    stock_capture = json.loads(stock_capture_path.read_text(encoding="utf-8"))
    compact_capture = json.loads(compact_capture_path.read_text(encoding="utf-8"))
    assert stock_capture["capture_provenance"]["feeds_builder"] is True
    assert compact_capture["artifact_refs"][0]["required"] is True
    quality.compact_matched_learning_quality_arm_from_capture_v1(
        stock_capture,
        expected_role="stock_reference",
    )
    quality.compact_matched_learning_quality_arm_from_capture_v1(
        compact_capture,
        expected_role="compact_candidate",
    )

    payload = builder_module.build_compact_matched_learning_quality_canary_payload(
        run_id="unit-quality-builder",
        compatibility_report_path=paths["compatibility"],
        unified_lifecycle_report_path=paths["lifecycle"],
        stock_reference_capture_path=stock_capture_path,
        compact_candidate_capture_path=compact_capture_path,
    )
    assert payload["readiness"]["matched_learning_quality_canary"] is True
    quality.validate_compact_matched_learning_quality_canary_v1(payload)


def test_capture_from_artifacts_rejects_unbound_provenance(tmp_path):
    capture_module = _load_capture_from_artifacts_module()
    output_root = tmp_path / "out"
    inputs = _capture_from_artifacts_inputs(tmp_path, role="stock_reference")
    provenance = json.loads(inputs["capture_provenance"].read_text(encoding="utf-8"))
    provenance["training_artifact_ref"] = "not-bound-to-artifacts"
    _write_json(inputs["capture_provenance"], provenance)

    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="training_artifact_ref must match artifact kind or path",
    ):
        capture_module.main(
            [
                "--role",
                "stock_reference",
                "--run-id",
                "unit-capture-from-artifacts",
                "--capture-id",
                "unit-stock-reference-capture",
                "--candidate-checkpoint-id",
                "unit-compact-ckpt",
                "--denominator-id",
                "unit-matched-quality-denominator",
                "--quality-horizon",
                "pre_post_100_env_steps",
                "--hardware-class",
                "local-test",
                "--source-fingerprint-json",
                str(inputs["source_fingerprint"]),
                "--model-identity-json",
                str(inputs["model_identity"]),
                "--eval-settings-json",
                str(inputs["eval_settings"]),
                "--denominators-json",
                str(inputs["denominators"]),
                "--capture-provenance-json",
                str(inputs["capture_provenance"]),
                "--training-artifact",
                str(inputs["training_artifact"]),
                "--pre-eval-summary",
                str(inputs["pre_eval_summary"]),
                "--post-eval-summary",
                str(inputs["post_eval_summary"]),
                "--initial-checkpoint",
                str(inputs["initial_checkpoint"]),
                "--final-checkpoint",
                str(inputs["final_checkpoint"]),
                "--output-root",
                str(output_root),
            ]
        )


def test_capture_preview_does_not_feed_builder_or_use_capture_schema(tmp_path):
    preview_module = _load_preview_module()
    builder_module = _load_builder_module()
    paths = _input_paths(tmp_path)
    preview_inputs = _preview_inputs(tmp_path)
    output_root = tmp_path / "out"

    assert (
        preview_module.main(
            [
                "--role",
                "stock_reference",
                "--run-id",
                "unit-preview",
                "--candidate-checkpoint-id",
                "unit-compact-ckpt",
                "--denominator-id",
                "unit-matched-quality-denominator",
                "--quality-horizon",
                "pre_post_100_env_steps",
                "--hardware-class",
                "local-test",
                "--source-fingerprint-json",
                str(preview_inputs["source_fingerprint"]),
                "--model-identity-json",
                str(preview_inputs["model_identity"]),
                "--eval-settings-json",
                str(preview_inputs["eval_settings"]),
                "--denominators-json",
                str(preview_inputs["denominators"]),
                "--pre-eval-summary",
                str(preview_inputs["pre_eval"]),
                "--post-eval-summary",
                str(preview_inputs["post_eval"]),
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    preview_path = output_root / "unit-preview" / "stock_reference_capture_preview.json"
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    assert preview["schema_id"] == (
        quality.COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PREVIEW_SCHEMA_ID
    )
    assert preview["usable_as_quality_capture"] is False
    assert preview["feeds_builder"] is False
    assert preview["support_only"] is True
    assert not (
        output_root / "unit-preview" / "matched_learning_quality_canary_report.json"
    ).exists()

    compact_capture_path = tmp_path / "compact_candidate_capture.json"
    _write_json(compact_capture_path, _quality_capture(tmp_path, role="compact_candidate"))
    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="stock_reference capture schema mismatch",
    ):
        builder_module.build_compact_matched_learning_quality_canary_payload(
            run_id="unit-quality-builder",
            compatibility_report_path=paths["compatibility"],
            unified_lifecycle_report_path=paths["lifecycle"],
            stock_reference_capture_path=preview_path,
            compact_candidate_capture_path=compact_capture_path,
        )


def test_capture_preview_rejects_builder_input_filename(tmp_path):
    preview_module = _load_preview_module()
    preview_inputs = _preview_inputs(tmp_path)

    with pytest.raises(ValueError, match="preview output must not use"):
        preview_module.main(
            [
                "--role",
                "stock_reference",
                "--run-id",
                "unit-preview",
                "--candidate-checkpoint-id",
                "unit-compact-ckpt",
                "--denominator-id",
                "unit-matched-quality-denominator",
                "--quality-horizon",
                "pre_post_100_env_steps",
                "--hardware-class",
                "local-test",
                "--source-fingerprint-json",
                str(preview_inputs["source_fingerprint"]),
                "--model-identity-json",
                str(preview_inputs["model_identity"]),
                "--eval-settings-json",
                str(preview_inputs["eval_settings"]),
                "--denominators-json",
                str(preview_inputs["denominators"]),
                "--pre-eval-summary",
                str(preview_inputs["pre_eval"]),
                "--post-eval-summary",
                str(preview_inputs["post_eval"]),
                "--output",
                str(tmp_path / "stock_reference_capture.json"),
            ]
        )


def test_real_capture_builder_requires_real_training_provenance(tmp_path):
    preview_inputs = _preview_inputs(tmp_path)
    provenance = _capture_provenance("stock_reference")

    capture = quality.build_compact_matched_learning_quality_capture_v1(
        role="stock_reference",
        run_id="unit-real-capture",
        capture_id="unit-stock-capture",
        candidate_checkpoint_id="unit-compact-ckpt",
        denominator_id="unit-matched-quality-denominator",
        quality_horizon="pre_post_100_env_steps",
        hardware_class="local-test",
        source_fingerprint=json.loads(
            preview_inputs["source_fingerprint"].read_text(encoding="utf-8")
        ),
        model_identity=json.loads(preview_inputs["model_identity"].read_text(encoding="utf-8")),
        eval_settings=json.loads(preview_inputs["eval_settings"].read_text(encoding="utf-8")),
        pre_eval_summary=json.loads(preview_inputs["pre_eval"].read_text(encoding="utf-8")),
        post_eval_summary=json.loads(preview_inputs["post_eval"].read_text(encoding="utf-8")),
        denominators=json.loads(preview_inputs["denominators"].read_text(encoding="utf-8")),
        artifact_paths=_capture_artifact_paths(
            tmp_path,
            role="stock_reference",
            run_id="unit-real-capture",
        ),
        capture_provenance=provenance,
    )
    assert capture["schema_id"] == quality.COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_SCHEMA_ID
    assert capture["capture_provenance"]["producer_ran_training"] is True
    arm = quality.compact_matched_learning_quality_arm_from_capture_v1(
        capture,
        expected_role="stock_reference",
    )
    assert arm["quality_curve"][-1]["mean_survival"] == pytest.approx(12.0)

    provenance["producer_ran_training"] = False
    with pytest.raises(
        quality.CompactMatchedLearningQualityError,
        match="producer_ran_training=true",
    ):
        quality.build_compact_matched_learning_quality_capture_v1(
            role="stock_reference",
            run_id="unit-real-capture",
            capture_id="unit-stock-capture",
            candidate_checkpoint_id="unit-compact-ckpt",
            denominator_id="unit-matched-quality-denominator",
            quality_horizon="pre_post_100_env_steps",
            hardware_class="local-test",
            source_fingerprint=json.loads(
                preview_inputs["source_fingerprint"].read_text(encoding="utf-8")
            ),
            model_identity=json.loads(preview_inputs["model_identity"].read_text(encoding="utf-8")),
            eval_settings=json.loads(preview_inputs["eval_settings"].read_text(encoding="utf-8")),
            pre_eval_summary=json.loads(preview_inputs["pre_eval"].read_text(encoding="utf-8")),
            post_eval_summary=json.loads(preview_inputs["post_eval"].read_text(encoding="utf-8")),
            denominators=json.loads(preview_inputs["denominators"].read_text(encoding="utf-8")),
            artifact_paths=_capture_artifact_paths(
                tmp_path,
                role="stock_reference",
                run_id="unit-real-capture",
            ),
            capture_provenance=provenance,
        )


def _load_builder_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / ("build_compact_matched_learning_quality_canary.py")
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_learning_quality_canary_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_preview_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / ("build_compact_matched_learning_quality_capture_preview.py")
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_learning_quality_capture_preview_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_capture_from_artifacts_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / ("build_compact_matched_learning_quality_capture_from_artifacts.py")
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_matched_learning_quality_capture_from_artifacts_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _quality_capture(tmp_path, *, role):
    run_id = f"unit-{role}-run"
    artifact_paths = _capture_artifact_paths(tmp_path, role=role, run_id=run_id)
    is_stock = role == "stock_reference"
    base = {
        "schema_id": "curvyzero_compact_matched_learning_quality_capture/v1",
        "ok": True,
        "status": "complete",
        "role": role,
        "route": ("stock_train_muzero_reference" if is_stock else "compact_owned_trainer"),
        "profile_only": False,
        **({"called_train_muzero": True} if is_stock else {"calls_train_muzero": False}),
        "touches_live_runs": False,
        "candidate_checkpoint_id": "unit-compact-ckpt",
        "capture_id": f"unit-{role}",
        "run_id": run_id,
        "denominator_id": "unit-matched-quality-denominator",
        "quality_horizon": "pre_post_100_env_steps",
        "hardware_class": "local-test",
        "source_fingerprint": _source_fingerprint(role),
        "model_identity": {
            "initial_model_state_digest": f"{role}-initial",
            "final_model_state_digest": f"{role}-final",
            "model_state_digest_changed": True,
        },
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
        "eval_points": [
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
        "capture_provenance": _capture_provenance(role),
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
            }
        )
        base["model_identity"]["model_identity_scope"] = "candidate_loaded_checkpoint"
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
    return base


def _capture_provenance(role):
    is_stock = role == "stock_reference"
    base = {
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
        base["stock_train_muzero"] = {
            "called_train_muzero": True,
            "train_muzero_entrypoint": "lzero.entry.train_muzero",
        }
    else:
        base["compact_owned_training"] = {
            "calls_train_muzero": False,
            "real_compact_owned_training_work": True,
            "compact_training_entrypoint": (
                "curvyzero.training.compact_owned_trainer."
                "CompactOwnedTrainerV1.record_step"
            ),
        }
    return base


def _preview_inputs(tmp_path):
    paths = {
        "source_fingerprint": tmp_path / "source_fingerprint.json",
        "model_identity": tmp_path / "model_identity.json",
        "eval_settings": tmp_path / "eval_settings.json",
        "denominators": tmp_path / "denominators.json",
        "pre_eval": tmp_path / "pre_eval_summary.json",
        "post_eval": tmp_path / "post_eval_summary.json",
    }
    capture = _quality_capture(tmp_path, role="stock_reference")
    _write_json(paths["source_fingerprint"], capture["source_fingerprint"])
    _write_json(paths["model_identity"], capture["model_identity"])
    _write_json(paths["eval_settings"], capture["eval_settings"])
    _write_json(paths["denominators"], capture["denominators"])
    _write_json(
        paths["pre_eval"],
        {
            "checkpoint": "stock_reference-pre",
            "checkpoint_step": 0,
            "eval_seed_set": [101, 102],
            "eval_max_steps": 2000,
            "seeds": 2,
            "mean_steps": 10.0,
            "median_steps": 10.0,
            "min_steps": 9.0,
            "max_steps": 11.0,
            "ok_count": 2,
            "capped_count": 0,
            "failure_count": 0,
            "outcome_histogram": {"loss": 2},
        },
    )
    _write_json(
        paths["post_eval"],
        {
            "checkpoint": "stock_reference-post",
            "checkpoint_step": 100,
            "eval_seed_set": [101, 102],
            "eval_max_steps": 2000,
            "seeds": 2,
            "mean_steps": 12.0,
            "median_steps": 12.0,
            "min_steps": 11.0,
            "max_steps": 13.0,
            "ok_count": 2,
            "capped_count": 0,
            "failure_count": 0,
            "outcome_histogram": {"loss": 2},
        },
    )
    return paths


def _capture_from_artifacts_inputs(tmp_path, *, role):
    capture = _quality_capture(tmp_path, role=role)
    run_id = "unit-capture-from-artifacts"
    root = tmp_path / f"{role}_capture_from_artifacts"
    paths = {
        "source_fingerprint": root / "source_fingerprint.json",
        "model_identity": root / "model_identity.json",
        "eval_settings": root / "eval_settings.json",
        "denominators": root / "denominators.json",
        "capture_provenance": root / "capture_provenance.json",
        "training_artifact": root / "training_artifact.json",
        "pre_eval_summary": root / "pre_eval_summary.json",
        "post_eval_summary": root / "post_eval_summary.json",
        "initial_checkpoint": root / "initial_checkpoint.pt",
        "final_checkpoint": root / "final_checkpoint.pt",
    }
    _write_json(paths["source_fingerprint"], capture["source_fingerprint"])
    _write_json(paths["model_identity"], capture["model_identity"])
    _write_json(paths["eval_settings"], capture["eval_settings"])
    _write_json(paths["denominators"], capture["denominators"])
    _write_json(paths["capture_provenance"], capture["capture_provenance"])
    _write_json(paths["initial_checkpoint"], {"role": role, "kind": "initial"})
    _write_json(paths["final_checkpoint"], {"role": role, "kind": "final"})
    _write_json(paths["training_artifact"], _training_artifact_for_capture(role, paths, run_id))
    _write_json(paths["pre_eval_summary"], _eval_summary_for_capture(role, point="pre"))
    _write_json(paths["post_eval_summary"], _eval_summary_for_capture(role, point="post"))
    return paths


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


def _capture_artifact_paths(tmp_path, *, role, run_id=None):
    root = tmp_path / f"{role}_capture_artifacts"
    paths = {
        "training_artifact": root / "training_artifact.json",
        "pre_eval_summary": root / "pre_eval_summary.json",
        "post_eval_summary": root / "post_eval_summary.json",
        "initial_checkpoint": root / "initial_checkpoint.pt",
        "final_checkpoint": root / "final_checkpoint.pt",
    }
    actual_run_id = run_id or f"unit-{role}-run"
    _write_json(paths["initial_checkpoint"], {"kind": "initial_checkpoint", "role": role})
    _write_json(paths["final_checkpoint"], {"kind": "final_checkpoint", "role": role})
    _write_json(
        paths["training_artifact"],
        _training_artifact_for_capture(role, paths, actual_run_id),
    )
    _write_json(paths["pre_eval_summary"], _eval_summary_for_capture(role, point="pre"))
    _write_json(paths["post_eval_summary"], _eval_summary_for_capture(role, point="post"))
    return paths


def _training_artifact_for_capture(role, paths, run_id):
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
        "run_id": run_id,
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
                    "initial_sha256": _file_sha256(paths["initial_checkpoint"]),
                    "final_path": str(paths["final_checkpoint"]),
                    "final_sha256": _file_sha256(paths["final_checkpoint"]),
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


def _eval_summary_for_capture(role, *, point):
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
        "sha256": _file_sha256(path),
        "required": True,
    }


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
