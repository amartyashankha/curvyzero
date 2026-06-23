from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from curvyzero.training import compact_promotion_readiness as readiness
from curvyzero.training.opponent_leaderboard import (
    OPPONENT_DEATH_MODE_NORMAL,
    STABLE_SENTINEL_BLANK_CANVAS,
    STABLE_SLOT_PROFILE_3,
    build_leaderboard_pointer,
    build_leaderboard_snapshot_from_rating_snapshot,
    select_stable_slots_v1_assignment,
)


def test_stock_resume_load_canary_reexports_loaded_checkpoint(monkeypatch, tmp_path):
    paths = _input_paths(tmp_path)
    checkpoint = _checkpoint()
    _patch_canary_dependencies(monkeypatch, checkpoint)

    payload = readiness.build_compact_promotion_stock_resume_load_canary_v1(
        run_id="unit-canary",
        unified_lifecycle_report_path=paths["lifecycle"],
        compatibility_report_path=paths["compatibility"],
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        created_at="2026-05-30T00:00:00+00:00",
    )

    assert payload["schema_id"] == (
        readiness.COMPACT_PROMOTION_STOCK_RESUME_LOAD_CANARY_SCHEMA_ID
    )
    assert payload["ok"] is True
    assert payload["readiness"]["stock_resume_load_canary"] is True
    assert payload["readiness"]["promotion_readiness_complete"] is False
    assert payload["attached_claims"]["stock_resume_load_canary"] is True
    assert payload["attached_claims"]["promotion_claim"] is False
    assert payload["non_claims"]["stock_resume_claim"] is False
    assert payload["stock_resume_selection"]["ok"] is True
    assert payload["stock_resume_selection"]["selected_checkpoint_name"] == (
        "iteration_0.pth.tar"
    )
    assert payload["stock_resume_selection"]["load_ckpt_before_run_patch"]["new"] == (
        payload["stock_resume_selection"]["selected_checkpoint_path"]
    )
    assert payload["strict_stock_model_reload"]["ok"] is True
    assert payload["post_resume_loader_behavior"]["tournament_loader_ok"] is True
    assert payload["loaded_compact_checkpoint_identity"]["checkpoint_id"] == (
        "unit-compact-ckpt"
    )
    assert payload["resumed_stock_export_identity"]["stock_resume_claim"] is False
    readiness.validate_compact_promotion_stock_resume_load_canary_v1(payload)


def test_stock_resume_load_canary_rejects_claiming_compatibility_input(
    monkeypatch,
    tmp_path,
):
    paths = _input_paths(tmp_path, compatibility_overrides={"promotion_claim": True})
    _patch_canary_dependencies(monkeypatch, _checkpoint())

    with pytest.raises(readiness.CompactPromotionReadinessError, match="promotion_claim"):
        readiness.build_compact_promotion_stock_resume_load_canary_v1(
            run_id="unit-canary",
            unified_lifecycle_report_path=paths["lifecycle"],
            compatibility_report_path=paths["compatibility"],
            output_dir=tmp_path / "out",
            repo_root=tmp_path,
        )


def test_stock_resume_load_canary_rejects_loaded_checkpoint_identity_mismatch(
    monkeypatch,
    tmp_path,
):
    paths = _input_paths(tmp_path)
    checkpoint = _checkpoint({"checkpoint_id": "wrong-ckpt"})
    _patch_canary_dependencies(monkeypatch, checkpoint)

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="loaded compact checkpoint checkpoint_id mismatch",
    ):
        readiness.build_compact_promotion_stock_resume_load_canary_v1(
            run_id="unit-canary",
            unified_lifecycle_report_path=paths["lifecycle"],
            compatibility_report_path=paths["compatibility"],
            output_dir=tmp_path / "out",
            repo_root=tmp_path,
        )


def test_stock_resume_load_canary_validation_rejects_claim_flip(monkeypatch, tmp_path):
    paths = _input_paths(tmp_path)
    _patch_canary_dependencies(monkeypatch, _checkpoint())
    payload = readiness.build_compact_promotion_stock_resume_load_canary_v1(
        run_id="unit-canary",
        unified_lifecycle_report_path=paths["lifecycle"],
        compatibility_report_path=paths["compatibility"],
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
    )
    payload["non_claims"]["promotion_claim"] = True

    with pytest.raises(readiness.CompactPromotionReadinessError, match="promotion_claim"):
        readiness.validate_compact_promotion_stock_resume_load_canary_v1(payload)


def test_isolated_live_run_safety_canary_validates_sandbox_trainer_artifacts(
    monkeypatch,
    tmp_path,
):
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)

    payload = readiness.build_compact_promotion_isolated_live_run_safety_canary_v1(
        run_id="unit-isolated-live",
        unified_lifecycle_report_path=inputs["lifecycle"],
        compatibility_report_path=inputs["compatibility"],
        stock_resume_load_canary_report_path=inputs["stock_canary"],
        assignment_path=inputs["assignment"],
        assignment_audit_path=inputs["assignment_audit"],
        trainer_result_path=inputs["trainer_result"],
        metrics_path=inputs["metrics"],
        forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
        initial_checkpoint_path=inputs["initial_checkpoint"],
        final_checkpoint_path=inputs["final_checkpoint"],
        repo_root=tmp_path,
        created_at="2026-05-30T00:00:00+00:00",
    )

    assert payload["schema_id"] == (
        readiness.COMPACT_PROMOTION_ISOLATED_LIVE_RUN_SAFETY_CANARY_SCHEMA_ID
    )
    assert payload["ok"] is True
    assert payload["readiness"]["isolated_live_run_safety_canary"] is True
    assert payload["readiness"]["promotion_readiness_complete"] is False
    assert payload["trainer_consumption"]["canary_stock_train_muzero_called"] is True
    assert payload["trainer_consumption"]["compact_candidate_calls_train_muzero"] is False
    assert payload["attached_claims"]["isolated_live_run_safety_canary"] is True
    assert payload["attached_claims"]["live_run_safety_claim"] is False
    assert payload["non_claims"]["sandbox_assignment_rating_proof"] is False
    assert payload["assignment_plumbing"]["assignment_applied"] is True
    assert payload["checkpoint_and_metrics"]["collector_envstep_delta"] == 8
    readiness.validate_compact_promotion_isolated_live_run_safety_canary_v1(payload)


@pytest.mark.parametrize(
    "flag",
    [
        "production_live_runs_touched",
        "production_intake_touched",
        "production_rating_touched",
        "production_leaderboard_touched",
        "production_control_pointer_touched",
        "background_eval_enabled",
        "background_gif_enabled",
    ],
)
def test_isolated_live_run_safety_rejects_forbidden_production_touch(
    monkeypatch,
    tmp_path,
    flag,
):
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    audit = json.loads(inputs["forbidden_touch_audit"].read_text(encoding="utf-8"))
    audit["forbidden_touch_audit"][flag] = True
    _write_json(inputs["forbidden_touch_audit"], audit)

    with pytest.raises(readiness.CompactPromotionReadinessError, match=flag):
        readiness.build_compact_promotion_isolated_live_run_safety_canary_v1(
            run_id="unit-isolated-live",
            unified_lifecycle_report_path=inputs["lifecycle"],
            compatibility_report_path=inputs["compatibility"],
            stock_resume_load_canary_report_path=inputs["stock_canary"],
            assignment_path=inputs["assignment"],
            assignment_audit_path=inputs["assignment_audit"],
            trainer_result_path=inputs["trainer_result"],
            metrics_path=inputs["metrics"],
            forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
            initial_checkpoint_path=inputs["initial_checkpoint"],
            final_checkpoint_path=inputs["final_checkpoint"],
            repo_root=tmp_path,
        )


def test_isolated_live_run_safety_allows_stock_trainer_but_rejects_compact_train_muzero(
    monkeypatch,
    tmp_path,
):
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    trainer = json.loads(inputs["trainer_result"].read_text(encoding="utf-8"))
    assert trainer["canary_stock_train_muzero_called"] is True
    trainer["compact_candidate_calls_train_muzero"] = True
    _write_json(inputs["trainer_result"], trainer)

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="compact candidate must not call train_muzero",
    ):
        readiness.build_compact_promotion_isolated_live_run_safety_canary_v1(
            run_id="unit-isolated-live",
            unified_lifecycle_report_path=inputs["lifecycle"],
            compatibility_report_path=inputs["compatibility"],
            stock_resume_load_canary_report_path=inputs["stock_canary"],
            assignment_path=inputs["assignment"],
            assignment_audit_path=inputs["assignment_audit"],
            trainer_result_path=inputs["trainer_result"],
            metrics_path=inputs["metrics"],
            forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
            initial_checkpoint_path=inputs["initial_checkpoint"],
            final_checkpoint_path=inputs["final_checkpoint"],
            repo_root=tmp_path,
        )


def test_isolated_live_run_safety_rejects_missing_assignment_apply(
    monkeypatch,
    tmp_path,
):
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    trainer = json.loads(inputs["trainer_result"].read_text(encoding="utf-8"))
    trainer["assignment_consumption"]["assignment_applied"] = False
    _write_json(inputs["trainer_result"], trainer)

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="assignment not applied",
    ):
        readiness.build_compact_promotion_isolated_live_run_safety_canary_v1(
            run_id="unit-isolated-live",
            unified_lifecycle_report_path=inputs["lifecycle"],
            compatibility_report_path=inputs["compatibility"],
            stock_resume_load_canary_report_path=inputs["stock_canary"],
            assignment_path=inputs["assignment"],
            assignment_audit_path=inputs["assignment_audit"],
            trainer_result_path=inputs["trainer_result"],
            metrics_path=inputs["metrics"],
            forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
            initial_checkpoint_path=inputs["initial_checkpoint"],
            final_checkpoint_path=inputs["final_checkpoint"],
            repo_root=tmp_path,
        )


def test_isolated_live_run_safety_allows_initial_assignment_apply_without_refresh(
    monkeypatch,
    tmp_path,
):
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    trainer = json.loads(inputs["trainer_result"].read_text(encoding="utf-8"))
    trainer["assignment_consumption"][
        "assignment_refresh_latest_decision"
    ] = "initial_assignment_applied"
    _write_json(inputs["trainer_result"], trainer)

    payload = readiness.build_compact_promotion_isolated_live_run_safety_canary_v1(
        run_id="unit-isolated-live-initial-assignment",
        unified_lifecycle_report_path=inputs["lifecycle"],
        compatibility_report_path=inputs["compatibility"],
        stock_resume_load_canary_report_path=inputs["stock_canary"],
        assignment_path=inputs["assignment"],
        assignment_audit_path=inputs["assignment_audit"],
        trainer_result_path=inputs["trainer_result"],
        metrics_path=inputs["metrics"],
        forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
        initial_checkpoint_path=inputs["initial_checkpoint"],
        final_checkpoint_path=inputs["final_checkpoint"],
        repo_root=tmp_path,
    )

    assert (
        payload["assignment_plumbing"]["assignment_refresh_latest_decision"]
        == "initial_assignment_applied"
    )
    readiness.validate_compact_promotion_isolated_live_run_safety_canary_v1(payload)


def test_isolated_live_run_safety_rejects_no_metrics_movement(monkeypatch, tmp_path):
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    metrics = json.loads(inputs["metrics"].read_text(encoding="utf-8"))
    metrics["collector_envstep_delta"] = 0
    _write_json(inputs["metrics"], metrics)

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="collector_envstep_delta",
    ):
        readiness.build_compact_promotion_isolated_live_run_safety_canary_v1(
            run_id="unit-isolated-live",
            unified_lifecycle_report_path=inputs["lifecycle"],
            compatibility_report_path=inputs["compatibility"],
            stock_resume_load_canary_report_path=inputs["stock_canary"],
            assignment_path=inputs["assignment"],
            assignment_audit_path=inputs["assignment_audit"],
            trainer_result_path=inputs["trainer_result"],
            metrics_path=inputs["metrics"],
            forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
            initial_checkpoint_path=inputs["initial_checkpoint"],
            final_checkpoint_path=inputs["final_checkpoint"],
            repo_root=tmp_path,
        )


def test_isolated_live_run_safety_rejects_unchanged_final_checkpoint(
    monkeypatch,
    tmp_path,
):
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    inputs["final_checkpoint"].write_bytes(inputs["initial_checkpoint"].read_bytes())

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="final checkpoint must differ",
    ):
        readiness.build_compact_promotion_isolated_live_run_safety_canary_v1(
            run_id="unit-isolated-live",
            unified_lifecycle_report_path=inputs["lifecycle"],
            compatibility_report_path=inputs["compatibility"],
            stock_resume_load_canary_report_path=inputs["stock_canary"],
            assignment_path=inputs["assignment"],
            assignment_audit_path=inputs["assignment_audit"],
            trainer_result_path=inputs["trainer_result"],
            metrics_path=inputs["metrics"],
            forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
            initial_checkpoint_path=inputs["initial_checkpoint"],
            final_checkpoint_path=inputs["final_checkpoint"],
            repo_root=tmp_path,
        )


def test_isolated_live_run_safety_rejects_stock_resume_claim_flip(
    monkeypatch,
    tmp_path,
):
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    stock_canary = json.loads(inputs["stock_canary"].read_text(encoding="utf-8"))
    stock_canary["non_claims"]["stock_resume_claim"] = True
    _write_json(inputs["stock_canary"], stock_canary)

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="stock.*stock_resume_claim",
    ):
        readiness.build_compact_promotion_isolated_live_run_safety_canary_v1(
            run_id="unit-isolated-live",
            unified_lifecycle_report_path=inputs["lifecycle"],
            compatibility_report_path=inputs["compatibility"],
            stock_resume_load_canary_report_path=inputs["stock_canary"],
            assignment_path=inputs["assignment"],
            assignment_audit_path=inputs["assignment_audit"],
            trainer_result_path=inputs["trainer_result"],
            metrics_path=inputs["metrics"],
            forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
            initial_checkpoint_path=inputs["initial_checkpoint"],
            final_checkpoint_path=inputs["final_checkpoint"],
            repo_root=tmp_path,
        )


def test_isolated_live_run_safety_cli_writes_report_and_rejects_stale_output(
    monkeypatch,
    tmp_path,
):
    module = _load_isolated_live_run_module()
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    output_root = tmp_path / "out"
    args = [
        "--run-id",
        "unit-isolated-live-main",
        "--output-root",
        str(output_root),
        "--unified-lifecycle-report",
        str(inputs["lifecycle"]),
        "--compatibility-report",
        str(inputs["compatibility"]),
        "--stock-resume-load-canary-report",
        str(inputs["stock_canary"]),
        "--assignment",
        str(inputs["assignment"]),
        "--assignment-audit",
        str(inputs["assignment_audit"]),
        "--trainer-result",
        str(inputs["trainer_result"]),
        "--metrics",
        str(inputs["metrics"]),
        "--forbidden-touch-audit",
        str(inputs["forbidden_touch_audit"]),
        "--initial-checkpoint",
        str(inputs["initial_checkpoint"]),
        "--final-checkpoint",
        str(inputs["final_checkpoint"]),
    ]

    assert module.main(args) == 0
    report_path = (
        output_root
        / "unit-isolated-live-main"
        / "isolated_live_run_safety_canary_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "isolated_live_run_safety_canary_verified"
    readiness.validate_compact_promotion_isolated_live_run_safety_canary_v1(payload)

    with pytest.raises(FileExistsError, match="already exists"):
        module.main(args)


def test_sandbox_assignment_rating_proof_builds_local_report_and_validates(
    monkeypatch,
    tmp_path,
):
    inputs = _sandbox_assignment_rating_inputs(monkeypatch, tmp_path)

    payload = readiness.build_compact_promotion_sandbox_assignment_rating_proof_v1(
        run_id="unit-sandbox-assignment-rating",
        unified_lifecycle_report_path=inputs["lifecycle"],
        compatibility_report_path=inputs["compatibility"],
        stock_resume_load_canary_report_path=inputs["stock_canary"],
        isolated_live_run_safety_canary_report_path=inputs["isolated_canary"],
        rating_snapshot_path=inputs["rating_snapshot"],
        leaderboard_snapshot_path=inputs["leaderboard_snapshot"],
        leaderboard_pointer_path=inputs["leaderboard_pointer"],
        assignment_path=inputs["assignment"],
        assignment_audit_path=inputs["assignment_audit"],
        forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
        repo_root=tmp_path,
        created_at="2026-05-30T00:00:00+00:00",
    )

    assert payload["schema_id"] == (
        readiness.COMPACT_PROMOTION_SANDBOX_ASSIGNMENT_RATING_PROOF_SCHEMA_ID
    )
    assert payload["readiness"]["sandbox_assignment_rating_proof"] is True
    assert payload["readiness"]["promotion_readiness_complete"] is False
    assert payload["rating_signal"]["rated_pair_count"] == 1
    assert payload["leaderboard_materialization"]["candidate_status"] == "active"
    assert payload["assignment_materialization"]["candidate_selected"] is True
    assert payload["attached_claims"]["sandbox_assignment_rating_proof"] is True
    assert payload["attached_claims"]["rating_or_promotion_quality_claim"] is False
    assert payload["attached_claims"]["leaderboard_claim"] is False
    assert payload["non_claims"]["public_leaderboard_publish_claim"] is False
    readiness.validate_compact_promotion_sandbox_assignment_rating_proof_v1(payload)


def test_sandbox_assignment_rating_proof_rejects_candidate_checkpoint_mismatch(
    monkeypatch,
    tmp_path,
):
    inputs = _sandbox_assignment_rating_inputs(monkeypatch, tmp_path)
    rating = json.loads(inputs["rating_snapshot"].read_text(encoding="utf-8"))
    rating["ratings"][0]["checkpoint_id"] = "wrong-candidate"
    _write_json(inputs["rating_snapshot"], rating)

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="candidate checkpoint",
    ):
        readiness.build_compact_promotion_sandbox_assignment_rating_proof_v1(
            run_id="unit-sandbox-assignment-rating",
            unified_lifecycle_report_path=inputs["lifecycle"],
            compatibility_report_path=inputs["compatibility"],
            stock_resume_load_canary_report_path=inputs["stock_canary"],
            isolated_live_run_safety_canary_report_path=inputs["isolated_canary"],
            rating_snapshot_path=inputs["rating_snapshot"],
            leaderboard_snapshot_path=inputs["leaderboard_snapshot"],
            leaderboard_pointer_path=inputs["leaderboard_pointer"],
            assignment_path=inputs["assignment"],
            assignment_audit_path=inputs["assignment_audit"],
            forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
            repo_root=tmp_path,
        )


def test_sandbox_assignment_rating_proof_validation_rejects_rating_hash_drift(
    monkeypatch,
    tmp_path,
):
    inputs = _sandbox_assignment_rating_inputs(monkeypatch, tmp_path)
    payload = readiness.build_compact_promotion_sandbox_assignment_rating_proof_v1(
        run_id="unit-sandbox-assignment-rating",
        unified_lifecycle_report_path=inputs["lifecycle"],
        compatibility_report_path=inputs["compatibility"],
        stock_resume_load_canary_report_path=inputs["stock_canary"],
        isolated_live_run_safety_canary_report_path=inputs["isolated_canary"],
        rating_snapshot_path=inputs["rating_snapshot"],
        leaderboard_snapshot_path=inputs["leaderboard_snapshot"],
        leaderboard_pointer_path=inputs["leaderboard_pointer"],
        assignment_path=inputs["assignment"],
        assignment_audit_path=inputs["assignment_audit"],
        forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
        repo_root=tmp_path,
    )
    rating = json.loads(inputs["rating_snapshot"].read_text(encoding="utf-8"))
    rating["pair_count"] = 99
    _write_json(inputs["rating_snapshot"], rating)

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="rating_snapshot sha256 mismatch",
    ):
        readiness.validate_compact_promotion_sandbox_assignment_rating_proof_v1(
            payload
        )


def test_sandbox_assignment_rating_proof_rejects_assignment_parser_failure(
    monkeypatch,
    tmp_path,
):
    inputs = _sandbox_assignment_rating_inputs(monkeypatch, tmp_path)
    assignment = json.loads(inputs["assignment"].read_text(encoding="utf-8"))
    assignment["entries"][0]["opponent_checkpoint_ref"] = "runs:sandbox/latest.pth.tar"
    _write_json(inputs["assignment"], assignment)

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="immutable exact",
    ):
        readiness.build_compact_promotion_sandbox_assignment_rating_proof_v1(
            run_id="unit-sandbox-assignment-rating",
            unified_lifecycle_report_path=inputs["lifecycle"],
            compatibility_report_path=inputs["compatibility"],
            stock_resume_load_canary_report_path=inputs["stock_canary"],
            isolated_live_run_safety_canary_report_path=inputs["isolated_canary"],
            rating_snapshot_path=inputs["rating_snapshot"],
            leaderboard_snapshot_path=inputs["leaderboard_snapshot"],
            leaderboard_pointer_path=inputs["leaderboard_pointer"],
            assignment_path=inputs["assignment"],
            assignment_audit_path=inputs["assignment_audit"],
            forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "flag",
    [
        "public_leaderboard_written",
        "leaderboard_pointer_published",
        "production_control_pointer_touched",
        "production_rating_touched",
        "promotion_published",
        "checkpoint_intake_touched",
    ],
)
def test_sandbox_assignment_rating_proof_rejects_public_or_production_touch(
    monkeypatch,
    tmp_path,
    flag,
):
    inputs = _sandbox_assignment_rating_inputs(monkeypatch, tmp_path)
    audit = json.loads(inputs["forbidden_touch_audit"].read_text(encoding="utf-8"))
    audit["forbidden_touch_audit"][flag] = True
    _write_json(inputs["forbidden_touch_audit"], audit)

    with pytest.raises(readiness.CompactPromotionReadinessError, match=flag):
        readiness.build_compact_promotion_sandbox_assignment_rating_proof_v1(
            run_id="unit-sandbox-assignment-rating",
            unified_lifecycle_report_path=inputs["lifecycle"],
            compatibility_report_path=inputs["compatibility"],
            stock_resume_load_canary_report_path=inputs["stock_canary"],
            isolated_live_run_safety_canary_report_path=inputs["isolated_canary"],
            rating_snapshot_path=inputs["rating_snapshot"],
            leaderboard_snapshot_path=inputs["leaderboard_snapshot"],
            leaderboard_pointer_path=inputs["leaderboard_pointer"],
            assignment_path=inputs["assignment"],
            assignment_audit_path=inputs["assignment_audit"],
            forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
            repo_root=tmp_path,
        )


def test_sandbox_assignment_rating_proof_rejects_claim_flip(monkeypatch, tmp_path):
    inputs = _sandbox_assignment_rating_inputs(monkeypatch, tmp_path)
    payload = readiness.build_compact_promotion_sandbox_assignment_rating_proof_v1(
        run_id="unit-sandbox-assignment-rating",
        unified_lifecycle_report_path=inputs["lifecycle"],
        compatibility_report_path=inputs["compatibility"],
        stock_resume_load_canary_report_path=inputs["stock_canary"],
        isolated_live_run_safety_canary_report_path=inputs["isolated_canary"],
        rating_snapshot_path=inputs["rating_snapshot"],
        leaderboard_snapshot_path=inputs["leaderboard_snapshot"],
        leaderboard_pointer_path=inputs["leaderboard_pointer"],
        assignment_path=inputs["assignment"],
        assignment_audit_path=inputs["assignment_audit"],
        forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
        repo_root=tmp_path,
    )
    payload["attached_claims"]["rating_or_promotion_quality_claim"] = True

    with pytest.raises(
        readiness.CompactPromotionReadinessError,
        match="rating_or_promotion_quality_claim",
    ):
        readiness.validate_compact_promotion_sandbox_assignment_rating_proof_v1(
            payload
        )


def test_sandbox_assignment_rating_proof_cli_writes_report_and_rejects_stale_output(
    monkeypatch,
    tmp_path,
):
    module = _load_sandbox_assignment_rating_module()
    inputs = _sandbox_assignment_rating_inputs(monkeypatch, tmp_path)
    output_root = tmp_path / "out"
    args = [
        "--run-id",
        "unit-sandbox-assignment-rating-main",
        "--output-root",
        str(output_root),
        "--unified-lifecycle-report",
        str(inputs["lifecycle"]),
        "--compatibility-report",
        str(inputs["compatibility"]),
        "--stock-resume-load-canary-report",
        str(inputs["stock_canary"]),
        "--isolated-live-run-safety-canary-report",
        str(inputs["isolated_canary"]),
        "--rating-snapshot",
        str(inputs["rating_snapshot"]),
        "--leaderboard-snapshot",
        str(inputs["leaderboard_snapshot"]),
        "--leaderboard-pointer",
        str(inputs["leaderboard_pointer"]),
        "--assignment",
        str(inputs["assignment"]),
        "--assignment-audit",
        str(inputs["assignment_audit"]),
        "--forbidden-touch-audit",
        str(inputs["forbidden_touch_audit"]),
    ]

    assert module.main(args) == 0
    report_path = (
        output_root
        / "unit-sandbox-assignment-rating-main"
        / "sandbox_assignment_rating_proof_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "sandbox_assignment_rating_proof_verified"
    readiness.validate_compact_promotion_sandbox_assignment_rating_proof_v1(payload)

    with pytest.raises(FileExistsError, match="already exists"):
        module.main(args)


def test_sandbox_assignment_rating_proof_producer_packages_local_shape(
    monkeypatch,
    tmp_path,
):
    module = _load_sandbox_assignment_rating_producer_module()
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    output_root = tmp_path / "producer_out"

    assert module.main(
        [
            "--run-id",
            "unit-sandbox-assignment-rating-producer",
            "--output-root",
            str(output_root),
            "--unified-lifecycle-report",
            str(inputs["lifecycle"]),
            "--compatibility-report",
            str(inputs["compatibility"]),
            "--stock-resume-load-canary-report",
            str(inputs["stock_canary"]),
            "--isolated-live-run-safety-canary-report",
            str(_write_isolated_canary_report(inputs, tmp_path)),
        ]
    ) == 0

    report_path = (
        output_root
        / "unit-sandbox-assignment-rating-producer"
        / "sandbox_assignment_rating_proof_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["rating_signal"]["rating_signal_kind"] == (
        "local_tiny_sandbox_tournament_signal"
    )
    assert payload["leaderboard_materialization"]["pointer_published"] is False
    assert payload["assignment_materialization"]["strategy_id"] == "stable_slots_v1"
    readiness.validate_compact_promotion_sandbox_assignment_rating_proof_v1(payload)


def test_isolated_live_run_safety_producer_packages_real_sandbox_shape(
    monkeypatch,
    tmp_path,
):
    module = _load_isolated_live_run_producer_module()
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    output_root = tmp_path / "producer_out"
    assignment = json.loads(inputs["assignment"].read_text(encoding="utf-8"))
    assignment_sha = readiness.canonical_assignment_json_sha256(assignment)
    uploaded = []

    def fake_upload_volume_bytes(**kwargs):
        uploaded.append(kwargs)

    def fake_write_remote_assignment(*, args, assignment, assignment_audit):
        del args, assignment_audit
        return {
            "schema_id": "curvyzero_opponent_assignment_artifact_write/v0",
            "ok": True,
            "assignment_ref": "training/unit/assignment.json",
            "assignment_sha256": readiness.canonical_assignment_json_sha256(assignment),
        }

    def fake_train(*, args, initial_checkpoint_ref, assignment_ref):
        return _isolated_live_producer_train_result(
            args=args,
            initial_checkpoint_ref=initial_checkpoint_ref,
            assignment_ref=assignment_ref,
            assignment_sha=assignment_sha,
        )

    def fake_status(*, args):
        del args
        return {
            "learner_metrics_latest_exists": True,
            "learner_train_call_index": 2,
            "learner_collector_envstep": 16,
        }

    def fake_assignment_proof(*, args, assignment_sha256):
        del args
        assert assignment_sha256 == assignment_sha
        return {
            "assignment_env_proof_target_row_count": 2,
            "assignment_env_proof_target_provider_ok_count": 2,
            "assignment_env_proof_target_provider_false_count": 0,
        }

    def fake_download(ref, output_path, **kwargs):
        del ref, kwargs
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"producer-final-checkpoint")

    monkeypatch.setattr(module, "_upload_volume_bytes", fake_upload_volume_bytes)
    monkeypatch.setattr(module, "_write_remote_assignment", fake_write_remote_assignment)
    monkeypatch.setattr(module, "_run_sandbox_stock_train", fake_train)
    monkeypatch.setattr(module, "_load_run_status", fake_status)
    monkeypatch.setattr(module, "_load_assignment_proof", fake_assignment_proof)
    monkeypatch.setattr(module, "_download_volume_ref", fake_download)

    assert module.main(
        [
            "--run-id",
            "unit-isolated-live-producer",
            "--attempt-id",
            "unit-attempt",
            "--output-root",
            str(output_root),
            "--unified-lifecycle-report",
            str(inputs["lifecycle"]),
            "--compatibility-report",
            str(inputs["compatibility"]),
            "--stock-resume-load-canary-report",
            str(inputs["stock_canary"]),
            "--assignment",
            str(inputs["assignment"]),
            "--assignment-audit",
            str(inputs["assignment_audit"]),
        ]
    ) == 0

    assert uploaded
    report_path = (
        output_root
        / "unit-isolated-live-producer"
        / "isolated_live_run_safety_canary_report.json"
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "isolated_live_run_safety_canary_verified"
    assert payload["trainer_consumption"]["canary_stock_train_muzero_called"] is True
    assert payload["assignment_plumbing"]["provider_ok_count"] == 2
    assert payload["forbidden_touch_audit"]["uses_production_modal_objects"] is False
    readiness.validate_compact_promotion_isolated_live_run_safety_canary_v1(payload)


def test_isolated_live_run_safety_producer_generates_frozen_initial_assignment(
    monkeypatch,
    tmp_path,
):
    module = _load_isolated_live_run_producer_module()
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    output_root = tmp_path / "producer_out"
    written = {}

    def fake_upload_volume_bytes(**kwargs):
        del kwargs

    def fake_write_remote_assignment(*, args, assignment, assignment_audit):
        del args
        module._validate_assignment_inputs(assignment, assignment_audit)
        assignment_sha = readiness.canonical_assignment_json_sha256(assignment)
        entry = assignment["entries"][0]
        assert len(assignment["assignment_id"]) <= 96
        assert assignment["assignment_id"].startswith("opt057-frozen-initial-")
        assert entry["opponent_policy_kind"] == "frozen_lightzero_checkpoint"
        assert entry["opponent_checkpoint_state_key"] == "model"
        assert entry["opponent_checkpoint_ref"].endswith(
            "/isolated_live_seed/iteration_0.pth.tar"
        )
        written["sha"] = assignment_sha
        return {
            "schema_id": "curvyzero_opponent_assignment_artifact_write/v0",
            "ok": True,
            "assignment_ref": "training/unit/generated-assignment.json",
            "assignment_sha256": assignment_sha,
        }

    def fake_train(*, args, initial_checkpoint_ref, assignment_ref):
        return _isolated_live_producer_train_result(
            args=args,
            initial_checkpoint_ref=initial_checkpoint_ref,
            assignment_ref=assignment_ref,
            assignment_sha=written["sha"],
        )

    def fake_status(*, args):
        del args
        return {
            "learner_metrics_latest_exists": True,
            "learner_train_call_index": 2,
            "learner_collector_envstep": 16,
        }

    def fake_assignment_proof(*, args, assignment_sha256):
        del args
        assert assignment_sha256 == written["sha"]
        return {
            "assignment_env_proof_target_row_count": 2,
            "assignment_env_proof_target_provider_ok_count": 2,
            "assignment_env_proof_target_provider_false_count": 0,
        }

    def fake_download(ref, output_path, **kwargs):
        del ref, kwargs
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"producer-final-checkpoint")

    monkeypatch.setattr(module, "_upload_volume_bytes", fake_upload_volume_bytes)
    monkeypatch.setattr(module, "_write_remote_assignment", fake_write_remote_assignment)
    monkeypatch.setattr(module, "_run_sandbox_stock_train", fake_train)
    monkeypatch.setattr(module, "_load_run_status", fake_status)
    monkeypatch.setattr(module, "_load_assignment_proof", fake_assignment_proof)
    monkeypatch.setattr(module, "_download_volume_ref", fake_download)

    assert module.main(
        [
            "--run-id",
            "unit-isolated-live-producer-generated-assignment",
            "--attempt-id",
            "unit-attempt",
            "--output-root",
            str(output_root),
            "--unified-lifecycle-report",
            str(inputs["lifecycle"]),
            "--compatibility-report",
            str(inputs["compatibility"]),
            "--stock-resume-load-canary-report",
            str(inputs["stock_canary"]),
        ]
    ) == 0

    generated_assignment_path = (
        output_root
        / "unit-isolated-live-producer-generated-assignment"
        / "canary_inputs"
        / "sandbox_assignment.generated.json"
    )
    assert generated_assignment_path.is_file()
    generated_assignment = json.loads(generated_assignment_path.read_text(encoding="utf-8"))
    assert (
        readiness.canonical_assignment_json_sha256(generated_assignment)
        == written["sha"]
    )
    report_path = (
        output_root
        / "unit-isolated-live-producer-generated-assignment"
        / "isolated_live_run_safety_canary_report.json"
    )
    readiness.validate_compact_promotion_isolated_live_run_safety_canary_v1(
        json.loads(report_path.read_text(encoding="utf-8"))
    )


def test_isolated_live_run_safety_producer_rejects_missing_provider_proof():
    module = _load_isolated_live_run_producer_module()

    class FakeProofFn:
        def remote(self, *args):
            del args
            return [
                {
                    "assignment_env_proof_target_row_count": 2,
                    "assignment_env_proof_target_provider_ok_count": 0,
                    "assignment_env_proof_target_provider_false_count": 0,
                }
            ]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        module,
        "_deployed_modal_function",
        lambda **kwargs: FakeProofFn(),
    )
    try:
        with pytest.raises(ValueError, match="provider_ok"):
            module._load_assignment_proof(
                args=SimpleNamespace(
                    run_id="unit",
                    attempt_id="attempt",
                    status_app_name="status",
                    modal_env=None,
                ),
                assignment_sha256="a" * 64,
            )
    finally:
        monkeypatch.undo()


def test_isolated_live_run_safety_producer_accepts_modal_slim_command_shape():
    module = _load_isolated_live_run_producer_module()
    args = SimpleNamespace(run_id="unit", attempt_id="attempt")
    train_result = _isolated_live_producer_train_result(
        args=args,
        initial_checkpoint_ref="training/unit/isolated_live_seed/iteration_0.pth.tar",
        assignment_ref="training/unit/assignment.json",
        assignment_sha="a" * 64,
    )
    train_result["command"]["initial_policy_checkpoint_ref"] = None
    train_result["command"]["initial_policy_checkpoint_state_key"] = None
    train_result["command"]["initial_policy_checkpoint_load_mode"] = None
    train_result["command"]["opponent_assignment_refresh_ref"] = None
    train_result["command"]["opponent_assignment_refresh_interval_train_iter"] = None
    train_result["opponent_assignment_refresh"] = None
    train_result["checkpoint_mirror"] = train_result["checkpoint_mirror"][
        "copied_checkpoints"
    ]

    module._validate_sandbox_train_result(
        train_result,
        args=args,
        remote_initial={
            "checkpoint_ref": "training/unit/isolated_live_seed/iteration_0.pth.tar",
        },
        remote_assignment={"assignment_ref": "training/unit/assignment.json"},
    )
    assert module._select_final_checkpoint(train_result) == {
        "iteration": 1,
        "ref": "training/unit/checkpoints/iteration_1.pth.tar",
    }


def _load_isolated_live_run_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_promotion_isolated_live_run_safety_canary.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_promotion_isolated_live_run_safety_canary_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_isolated_live_run_producer_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_promotion_isolated_live_run_safety_canary_producer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_promotion_isolated_live_run_safety_canary_producer_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sandbox_assignment_rating_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_promotion_sandbox_assignment_rating_proof.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_promotion_sandbox_assignment_rating_proof_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sandbox_assignment_rating_producer_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_compact_promotion_sandbox_assignment_rating_proof_producer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_compact_promotion_sandbox_assignment_rating_proof_producer_for_test",
        path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _isolated_live_inputs(monkeypatch, tmp_path):
    paths = _input_paths(tmp_path)
    _patch_canary_dependencies(monkeypatch, _checkpoint())
    stock_payload = readiness.build_compact_promotion_stock_resume_load_canary_v1(
        run_id="unit-stock-canary",
        unified_lifecycle_report_path=paths["lifecycle"],
        compatibility_report_path=paths["compatibility"],
        output_dir=tmp_path / "stock_canary_out",
        repo_root=tmp_path,
        created_at="2026-05-30T00:00:00+00:00",
    )
    stock_canary_path = tmp_path / "stock_resume_load_canary_report.json"
    _write_json(stock_canary_path, stock_payload)
    initial_checkpoint = Path(stock_payload["resumed_stock_export_path"])
    final_checkpoint = tmp_path / "sandbox_train" / "iteration_1.pth.tar"
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    final_checkpoint.write_bytes(b"sandbox-final-checkpoint")

    assignment = {
        "schema_id": "curvyzero_opponent_assignment/v0",
        "assignment_id": "isolated-live-assignment",
        "source_epoch": 1,
        "source_ref": "sandbox/isolated-live/assignment.json",
        "seed": 17,
        "entries": [
            {
                "name": "sandbox_blank_canvas",
                "weight": 1,
                "opponent_policy_kind": "fixed_straight",
                "opponent_runtime_mode": "blank_canvas_noop",
                "opponent_immortal": True,
            }
        ],
    }
    assignment_sha = readiness.canonical_assignment_json_sha256(assignment)
    assignment_path = tmp_path / "sandbox_assignment.json"
    assignment_audit_path = tmp_path / "sandbox_assignment_audit.json"
    _write_json(assignment_path, assignment)
    _write_json(
        assignment_audit_path,
        {
            "schema_id": "curvyzero_opponent_assignment_audit/v0",
            "assignment_id": "isolated-live-assignment",
            "assignment_sha256": assignment_sha,
            "selection": {"strategy_id": "sandbox_fixture_v1"},
        },
    )
    trainer_result_path = tmp_path / "sandbox_trainer_result.json"
    _write_json(
        trainer_result_path,
        {
            "schema_id": "curvyzero_compact_isolated_live_run_trainer_result/v1",
            "ok": True,
            "trainer_mode": "sandbox_canary",
            "trainer_entrypoint": "lzero.entry.train_muzero",
            "canary_stock_train_muzero_called": True,
            "compact_candidate_calls_train_muzero": False,
            "load_ckpt_before_run_target": str(initial_checkpoint),
            "initial_policy_checkpoint": {
                "enabled": True,
                "checkpoint_ref": "runs:sandbox/iteration_0.pth.tar",
                "source_path": str(initial_checkpoint),
                "source_sha256": readiness._file_sha256(initial_checkpoint),
                "load_mode": "matching_shape",
                "state_key": "model",
                "applied": True,
                "prepared": {
                    "kind": "model_only_checkpoint",
                    "fresh_optimizer_intent": True,
                },
                "load_result": {
                    "loaded": True,
                    "fresh_optimizer_preserved": True,
                    "errors": [],
                },
            },
            "assignment_consumption": {
                "assignment_loaded": True,
                "assignment_applied": True,
                "trainer_loaded_assignment_ref": "sandbox:assignment.json",
                "trainer_loaded_assignment_sha256": assignment_sha,
                "trainer_applied_assignment_sha256": assignment_sha,
                "assignment_refresh_latest_decision": "applied",
                "env_telemetry_assignment_sha256": assignment_sha,
                "env_telemetry_row_count": 1,
                "provider_ok_count": 1,
                "provider_false_count": 0,
            },
            "lineage_stages": [
                "trainer_assignment_loaded",
                "trainer_assignment_applied",
                "checkpoint_written",
            ],
        },
    )
    metrics_path = tmp_path / "sandbox_metrics.json"
    _write_json(
        metrics_path,
        {
            "schema_id": "curvyzero_compact_isolated_live_run_metrics/v1",
            "ok": True,
            "checkpoint_write_ok": True,
            "checkpoint_read_ok": True,
            "progress_moved": True,
            "learner_metrics_moved": True,
            "collector_envstep_delta": 8,
            "learner_train_calls_delta": 1,
            "checkpoint_iteration_delta": 1,
            "env_telemetry_row_count": 1,
            "training_wall_sec": 1.25,
        },
    )
    forbidden_touch_audit_path = tmp_path / "forbidden_touch_audit.json"
    _write_json(
        forbidden_touch_audit_path,
        {
            "schema_id": "curvyzero_compact_isolated_live_run_touch_audit/v1",
            "ok": True,
            "sandbox_scope": {
                "isolated": True,
                "namespace": "isolated-opt057-unit",
                "production_namespace": False,
            },
            "forbidden_touch_audit": {
                "production_live_runs_touched": False,
                "production_intake_touched": False,
                "production_rating_touched": False,
                "production_leaderboard_touched": False,
                "production_control_pointer_touched": False,
                "writes_checkpoint_intake": False,
                "spawns_rating": False,
                "publishes_leaderboard": False,
                "rewrites_production_control_pointers": False,
                "uses_production_modal_objects": False,
                "background_eval_enabled": False,
                "background_gif_enabled": False,
            },
        },
    )
    return {
        "lifecycle": paths["lifecycle"],
        "compatibility": paths["compatibility"],
        "stock_canary": stock_canary_path,
        "assignment": assignment_path,
        "assignment_audit": assignment_audit_path,
        "trainer_result": trainer_result_path,
        "metrics": metrics_path,
        "forbidden_touch_audit": forbidden_touch_audit_path,
        "initial_checkpoint": initial_checkpoint,
        "final_checkpoint": final_checkpoint,
    }


def _write_isolated_canary_report(inputs, tmp_path):
    payload = readiness.build_compact_promotion_isolated_live_run_safety_canary_v1(
        run_id="unit-isolated-live-for-sandbox-rating",
        unified_lifecycle_report_path=inputs["lifecycle"],
        compatibility_report_path=inputs["compatibility"],
        stock_resume_load_canary_report_path=inputs["stock_canary"],
        assignment_path=inputs["assignment"],
        assignment_audit_path=inputs["assignment_audit"],
        trainer_result_path=inputs["trainer_result"],
        metrics_path=inputs["metrics"],
        forbidden_touch_audit_path=inputs["forbidden_touch_audit"],
        initial_checkpoint_path=inputs["initial_checkpoint"],
        final_checkpoint_path=inputs["final_checkpoint"],
        repo_root=tmp_path,
        created_at="2026-05-30T00:00:00+00:00",
    )
    path = tmp_path / "isolated_live_run_safety_canary_report.json"
    _write_json(path, payload)
    return path


def _sandbox_assignment_rating_inputs(monkeypatch, tmp_path):
    inputs = _isolated_live_inputs(monkeypatch, tmp_path)
    isolated_canary_path = _write_isolated_canary_report(inputs, tmp_path)
    sandbox_dir = tmp_path / "sandbox_assignment_rating"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    reference_checkpoint = sandbox_dir / "reference" / "iteration_0.pth.tar"
    reference_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    reference_checkpoint.write_bytes(b"sandbox-reference-checkpoint")

    candidate_ref = _relative_test_ref(inputs["initial_checkpoint"], tmp_path)
    reference_ref = _relative_test_ref(reference_checkpoint, tmp_path)
    rating_spec = {
        "tournament_id": "sandbox-unit-assignment-rating",
        "rating_run_id": "sandbox-proof",
        "checkpoints": [
            {
                "checkpoint_id": "unit-compact-ckpt",
                "label": "unit compact candidate",
                "checkpoint_ref": candidate_ref,
                "run_id": "unit-compact",
                "attempt_id": "unit-sandbox",
                "iteration": 0,
                "latest_for_run": True,
            },
            {
                "checkpoint_id": "sandbox-reference-opponent",
                "label": "sandbox reference",
                "checkpoint_ref": reference_ref,
                "run_id": "unit-reference",
                "attempt_id": "unit-sandbox",
                "iteration": 0,
                "latest_for_run": True,
            },
        ],
        "games_per_pair": 5,
        "placement_min_games": 5,
        "placement_min_opponents": 1,
        "active_pool_limit": 2,
        "min_valid_fraction": 1.0,
        "seed": 58,
        "num_simulations": 1,
        "policy_batch_size": 1,
    }
    normalized = readiness.arena.normalize_rating_spec(rating_spec)
    candidate, reference = normalized["checkpoints"]
    rating_snapshot = readiness.arena.rating_snapshot_from_pair_results(
        pair_results=[
            {
                "battle_id": "sandbox-pair-000",
                "pair_index": 0,
                "summary_ref": "local://unit-sandbox/sandbox-pair-000",
                "players": [candidate, reference],
                "settings": {"games_per_pair": 5},
                "tally": {
                    "game_count": 5,
                    "draw_count": 0,
                    "failure_count": 0,
                    "wins_by_checkpoint": {
                        "unit-compact-ckpt": 3,
                        "sandbox-reference-opponent": 2,
                    },
                },
            }
        ],
        rating_spec=normalized,
        round_index=0,
        created_at="2026-05-30T00:00:00+00:00",
    )
    rating_snapshot["sandbox_only"] = True
    rating_snapshot["local_only"] = True
    rating_snapshot["rating_signal_kind"] = "local_tiny_sandbox_tournament_signal"
    rating_path = sandbox_dir / "rating_snapshot.json"
    _write_json(rating_path, rating_snapshot)

    leaderboard = build_leaderboard_snapshot_from_rating_snapshot(
        rating_snapshot,
        leaderboard_id="sandbox-unit-leaderboard",
        snapshot_id="sandbox-unit-snapshot-000",
        generation=0,
        created_at="2026-05-30T00:00:00+00:00",
        active_min_distinct_opponents=1,
        active_min_valid_games=5,
    )
    snapshot_ref = _relative_test_ref(sandbox_dir / "leaderboard_snapshot.json", tmp_path)
    pointer = build_leaderboard_pointer(
        leaderboard,
        snapshot_ref=snapshot_ref,
        published_at="2026-05-30T00:00:00+00:00",
        writer={"kind": "unit-local-materializer"},
    )
    pointer.update(
        {
            "local_only": True,
            "published": False,
            "public_leaderboard_written": False,
            "leaderboard_pointer_published": False,
        }
    )
    leaderboard_path = sandbox_dir / "leaderboard_snapshot.json"
    pointer_path = sandbox_dir / "leaderboard_pointer.json"
    _write_json(leaderboard_path, leaderboard)
    _write_json(pointer_path, pointer)

    assignment, assignment_audit = select_stable_slots_v1_assignment(
        leaderboard,
        assignment_id="opt058-unit-assignment",
        source_ref=snapshot_ref,
        seed=58,
        profile=STABLE_SLOT_PROFILE_3,
        sentinel=STABLE_SENTINEL_BLANK_CANVAS,
        allow_recent_provisional=False,
        checkpoint_death_mode=OPPONENT_DEATH_MODE_NORMAL,
        expected_rating_context_hash=str(rating_snapshot["context_hash"]),
    )
    assignment_path = sandbox_dir / "assignment.json"
    assignment_audit_path = sandbox_dir / "assignment_audit.json"
    _write_json(assignment_path, assignment)
    _write_json(assignment_audit_path, assignment_audit)

    forbidden_touch_path = sandbox_dir / "forbidden_touch_audit.json"
    _write_json(
        forbidden_touch_path,
        {
            "schema_id": "curvyzero_compact_promotion_sandbox_assignment_rating_touch_audit/v1",
            "ok": True,
            "sandbox_scope": {
                "local_only": True,
                "namespace": "sandbox-unit-assignment-rating",
                "production_namespace": False,
            },
            "forbidden_touch_audit": {
                "production_live_runs_touched": False,
                "production_intake_touched": False,
                "production_rating_touched": False,
                "production_leaderboard_touched": False,
                "production_control_pointer_touched": False,
                "writes_checkpoint_intake": False,
                "spawns_rating": False,
                "publishes_leaderboard": False,
                "rewrites_production_control_pointers": False,
                "uses_production_modal_objects": False,
                "background_eval_enabled": False,
                "background_gif_enabled": False,
                "checkpoint_intake_touched": False,
                "rating_round_started": False,
                "rating_latest_written": False,
                "public_leaderboard_written": False,
                "leaderboard_pointer_published": False,
                "training_candidate_assignment_written": False,
                "assignment_pointer_rewritten": False,
                "promotion_published": False,
            },
            "lineage_stages": [
                "local_rating_snapshot_reduced",
                "local_leaderboard_snapshot_materialized",
                "local_assignment_materialized",
            ],
        },
    )
    return {
        **inputs,
        "isolated_canary": isolated_canary_path,
        "rating_snapshot": rating_path,
        "leaderboard_snapshot": leaderboard_path,
        "leaderboard_pointer": pointer_path,
        "assignment": assignment_path,
        "assignment_audit": assignment_audit_path,
        "forbidden_touch_audit": forbidden_touch_path,
    }


def _isolated_live_producer_train_result(
    *,
    args,
    initial_checkpoint_ref,
    assignment_ref,
    assignment_sha,
):
    return {
        "ok": True,
        "status": "completed",
        "mode": "train",
        "compute": "cpu",
        "run_id": str(args.run_id),
        "attempt_id": str(args.attempt_id),
        "called_train_muzero": True,
        "trainer_entrypoint": "lzero.entry.train_muzero",
        "command": {
            "background_eval_enabled": False,
            "background_gif_enabled": False,
            "opponent_assignment_ref": assignment_ref,
            "opponent_assignment_refresh_ref": assignment_ref,
            "initial_policy_checkpoint_ref": initial_checkpoint_ref,
            "initial_policy_checkpoint_state_key": "model",
            "initial_policy_checkpoint_load_mode": "matching_shape",
        },
        "auto_resume": {"found": False},
        "initial_policy_checkpoint": {
            "enabled": True,
            "checkpoint_ref": initial_checkpoint_ref,
            "source_path": initial_checkpoint_ref,
            "load_path": "training/unit/attempt/initial_policy_checkpoint/iteration_0.pth.tar",
            "load_mode": "matching_shape",
            "state_key": "model",
            "applied": True,
            "prepared": {
                "kind": "model_only_checkpoint",
                "fresh_optimizer_intent": True,
            },
            "load_result": {
                "loaded": True,
                "fresh_optimizer_preserved": True,
                "errors": [],
            },
        },
        "opponent_assignment_refresh": {
            "enabled": True,
            "event_count": 1,
            "events": [
                {
                    "decision": "applied",
                    "assignment_ref": assignment_ref,
                    "assignment_sha256": assignment_sha,
                    "env_ready_report": {
                        "ok": True,
                        "assignment_ref": assignment_ref,
                        "assignment_sha256": assignment_sha,
                    },
                }
            ],
        },
        "checkpoint_mirror": {
            "copied_checkpoints": [
                {"ref": "training/unit/checkpoints/iteration_0.pth.tar"},
                {"ref": "training/unit/checkpoints/iteration_1.pth.tar"},
            ]
        },
        "action_observability": {"row_count": 2},
        "target_audit": {"counts": {"replay_sample_calls": 2}},
        "train_result": {"elapsed_sec": 1.5},
        "phase_profile": {"timers_sec": {"train_muzero_wall_sec": 1.5}},
    }


def _input_paths(tmp_path, *, compatibility_overrides=None):
    checkpoint_path = tmp_path / "compact_checkpoint.pt"
    stock_export_path = tmp_path / "iteration_0.pth.tar"
    sidecar_path = tmp_path / "iteration_0.pth.tar.metadata.json"
    current_chain_path = tmp_path / "current_chain.evidence.json"
    lifecycle_path = tmp_path / "unified_lifecycle_report.json"
    compatibility_path = tmp_path / "compatibility_report.json"
    checkpoint_path.write_bytes(b"compact-checkpoint")
    stock_export_path.write_bytes(b"stock-export")
    _write_json(
        sidecar_path,
        {
            "schema_id": "curvyzero_checkpoint_policy_metadata/v0",
            "source_max_steps": 8,
            "policy_observation_backend": "cpu_oracle",
        },
    )
    identity = {
        "checkpoint_id": "unit-compact-ckpt",
        "trainer_id": "unit-compact-trainer",
        "policy_version_ref": "unit-policy-v1",
        "model_version_ref": "unit-model-v1",
        "policy_source": "unit-policy-source",
    }
    _write_json(
        current_chain_path,
        {
            "schema_id": "curvyzero_compact_current_chain_eval_gif_tournament_load_evidence/v1",
            "ok": True,
            "current_chain_identity": identity,
            "files": {
                "policy_metadata_sidecar": {
                    "path": str(sidecar_path),
                    "required": True,
                }
            },
        },
    )
    _write_json(
        lifecycle_path,
        {
            "schema_id": "curvyzero_compact_unified_lifecycle_smoke/v1",
            "ok": True,
            "checkpoint_id": "unit-compact-ckpt",
            "compact_checkpoint_path": str(checkpoint_path),
            "stock_export_path": str(stock_export_path),
            "current_chain_evidence_path": str(current_chain_path),
            "promotion_claim": False,
        },
    )
    compatibility = {
        "schema_id": "curvyzero_compact_coach_compatibility_refresh/v1",
        "ok": True,
        "candidate_checkpoint_id": "unit-compact-ckpt",
        "promotion_eligible": True,
        "promotion_claim": False,
        "touches_live_runs": False,
        "calls_train_muzero": False,
        "coach_speed_row_gate": True,
    }
    compatibility.update(compatibility_overrides or {})
    _write_json(compatibility_path, compatibility)
    return {
        "lifecycle": lifecycle_path,
        "compatibility": compatibility_path,
    }


def _checkpoint(metadata_overrides=None):
    metadata = {
        "checkpoint_id": "unit-compact-ckpt",
        "trainer_id": "unit-compact-trainer",
        "policy_version_ref": "unit-policy-v1",
        "model_version_ref": "unit-model-v1",
        "policy_source": "unit-policy-source",
        "train_step": 1,
        "learner_update_count": 1,
        "sample_batch_count": 1,
        "promotion_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "checkpoint_save_load": True,
        "resume_metadata": True,
    }
    metadata.update(metadata_overrides or {})
    return SimpleNamespace(
        metadata=metadata,
        model_state_dict={"linear.weight": np.asarray([1.0], dtype=np.float32)},
    )


def _patch_canary_dependencies(monkeypatch, checkpoint):
    monkeypatch.setattr(
        readiness,
        "validate_compact_current_chain_eval_gif_tournament_load_matches_checkpoint_v1",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        readiness,
        "load_compact_trainer_checkpoint_v1",
        lambda _path: checkpoint,
    )

    def fake_save_stock_export(_checkpoint, path, *, policy_metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"resumed-stock-export")
        sidecar_path = readiness.checkpoint_policy_metadata_sidecar_path(path)
        _write_json(sidecar_path, dict(policy_metadata))
        return {
            "checkpoint_path": path,
            "sidecar_path": sidecar_path,
            "payload": {
                "metadata": {
                    "source_compact_checkpoint_id": "unit-compact-ckpt",
                    "source_trainer_id": "unit-compact-trainer",
                    "policy_version_ref": "unit-policy-v1",
                    "model_version_ref": "unit-model-v1",
                    "policy_source": "unit-policy-source",
                    "stock_model_state_key": "model",
                    "stock_resume_claim": False,
                    "optimizer_resume_supported": False,
                    "promotion_claim": False,
                }
            },
        }

    monkeypatch.setattr(readiness, "save_compact_stock_export_v1", fake_save_stock_export)
    monkeypatch.setattr(
        readiness,
        "verify_compact_stock_export_model_contract_v1",
        lambda path, **kwargs: {
            "schema_id": "curvyzero_compact_stock_model_contract_verification/v1",
            "ok": True,
            "strict_load": True,
            "strict_stock_model_load_verified": True,
            "state_key": "model",
            "checkpoint_path": str(path),
        },
    )
    monkeypatch.setattr(
        readiness,
        "_tournament_loader_report",
        lambda *, stock_export_path, **kwargs: {
            "schema_id": "curvyzero_compact_stock_export_tournament_loader_smoke/v1",
            "ok": True,
            "checkpoint_state_key": "model",
            "checkpoint_path": str(stock_export_path),
        },
    )

    def fake_save_bundle(checkpoint_path, **kwargs):
        del kwargs
        path = readiness.Path(f"{checkpoint_path}.evidence.json")
        _write_json(path, {"schema_id": "unit-bundle", "ok": True})
        return {"bundle": {"schema_id": "unit-bundle", "ok": True}, "path": path}

    monkeypatch.setattr(
        readiness,
        "save_compact_stock_export_evidence_bundle_v1",
        fake_save_bundle,
    )
    monkeypatch.setattr(
        readiness,
        "validate_compact_stock_export_evidence_bundle_v1",
        lambda _bundle: None,
    )


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative_test_ref(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
