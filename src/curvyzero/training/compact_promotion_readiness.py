"""Post-compatibility readiness evidence for compact-owned candidates."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from curvyzero.tournament import curvytron_checkpoint_tournament as arena
from curvyzero.training import lightzero_checkpoints as lz_checkpoints
from curvyzero.training import lightzero_config_builder as lz_config
from curvyzero.training.compact_eval_gif_tournament_load import (
    validate_compact_current_chain_eval_gif_tournament_load_matches_checkpoint_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_model_state_digest_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    save_compact_stock_export_evidence_bundle_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import save_compact_stock_export_v1
from curvyzero.training.compact_stock_checkpoint_export import (
    validate_compact_stock_export_evidence_bundle_v1,
)
from curvyzero.training.compact_stock_checkpoint_export import (
    verify_compact_stock_export_model_contract_v1,
)
from curvyzero.training.compact_trainer_checkpoint import (
    load_compact_trainer_checkpoint_v1,
)
from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    checkpoint_policy_metadata_sidecar_path,
)
from curvyzero.training.opponent_leaderboard import validate_assignment_audit
from curvyzero.training.opponent_leaderboard import validate_leaderboard_pointer
from curvyzero.training.opponent_leaderboard import validate_leaderboard_snapshot
from curvyzero.training.opponent_leaderboard import validate_rating_snapshot_source
from curvyzero.training.opponent_registry import canonical_assignment_json_sha256
from curvyzero.training.opponent_registry import parse_opponent_assignment_snapshot


COMPACT_PROMOTION_STOCK_RESUME_LOAD_CANARY_SCHEMA_ID = (
    "curvyzero_compact_promotion_stock_resume_load_canary/v1"
)
COMPACT_PROMOTION_ISOLATED_LIVE_RUN_SAFETY_CANARY_SCHEMA_ID = (
    "curvyzero_compact_promotion_isolated_live_run_safety_canary/v1"
)
COMPACT_PROMOTION_SANDBOX_ASSIGNMENT_RATING_PROOF_SCHEMA_ID = (
    "curvyzero_compact_promotion_sandbox_assignment_rating_proof/v1"
)
COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID = (
    "curvyzero_compact_coach_compatibility_refresh/v1"
)
COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID = "curvyzero_compact_unified_lifecycle_smoke/v1"
COMPACT_PROMOTION_READINESS_REQUIRED_REMAINING_LANES = (
    "matched_stock_vs_compact_learning_quality",
    "isolated_live_run_safety_canary",
    "sandbox_assignment_rating_proof",
    "longer_horizon_compact_learning_metrics",
)


class CompactPromotionReadinessError(ValueError):
    """Raised when post-compatibility readiness evidence would overclaim."""


def build_compact_promotion_stock_resume_load_canary_v1(
    *,
    run_id: str,
    unified_lifecycle_report_path: str | Path,
    compatibility_report_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    seed: int = 0,
    num_simulations: int = 1,
    batch_size: int = 1,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a stock resume/load canary from a compatibility-eligible candidate."""

    repo = Path.cwd().resolve() if repo_root is None else Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lifecycle_path = _resolve_path(unified_lifecycle_report_path, repo)
    compatibility_path = _resolve_path(compatibility_report_path, repo)
    lifecycle = _read_json_mapping(lifecycle_path, "unified lifecycle report")
    compatibility = _read_json_mapping(compatibility_path, "compatibility report")
    _validate_lifecycle_and_compatibility_inputs(
        lifecycle,
        compatibility,
        lifecycle_path=lifecycle_path,
        compatibility_path=compatibility_path,
    )

    checkpoint_id = str(lifecycle["checkpoint_id"])
    compact_checkpoint_path = _resolve_path(
        _require_non_empty(lifecycle.get("compact_checkpoint_path"), "compact_checkpoint_path"),
        lifecycle_path.parent,
    )
    current_chain_evidence_path = _resolve_path(
        _require_non_empty(
            lifecycle.get("current_chain_evidence_path"),
            "current_chain_evidence_path",
        ),
        lifecycle_path.parent,
    )
    current_chain_evidence = _read_json_mapping(
        current_chain_evidence_path,
        "current-chain evidence",
    )
    identity = _required_mapping(
        current_chain_evidence.get("current_chain_identity"),
        "current_chain_identity",
    )
    validate_compact_current_chain_eval_gif_tournament_load_matches_checkpoint_v1(
        current_chain_evidence,
        checkpoint_id=checkpoint_id,
        trainer_id=str(identity["trainer_id"]),
        policy_version_ref=str(identity["policy_version_ref"]),
        model_version_ref=str(identity["model_version_ref"]),
        policy_source=str(identity["policy_source"]),
    )

    compact_checkpoint = load_compact_trainer_checkpoint_v1(compact_checkpoint_path)
    checkpoint_metadata = _required_mapping(
        getattr(compact_checkpoint, "metadata", None),
        "compact checkpoint metadata",
    )
    _validate_loaded_checkpoint_identity(
        checkpoint_metadata,
        checkpoint_id=checkpoint_id,
        identity=identity,
    )

    sidecar_path = _policy_sidecar_path_from_evidence(
        current_chain_evidence,
        fallback_stock_export_path=lifecycle.get("stock_export_path"),
        base_dir=lifecycle_path.parent,
    )
    sidecar = _read_json_mapping(sidecar_path, "policy sidecar")
    stock_resume_exp_dir = output / "train" / "lightzero_exp"
    resumed_export_path = stock_resume_exp_dir / "ckpt" / "iteration_0.pth.tar"
    verification_report_path = output / "verification_report.json"
    tournament_loader_report_path = output / "tournament_loader_report.json"

    export = save_compact_stock_export_v1(
        compact_checkpoint,
        resumed_export_path,
        policy_metadata=sidecar,
    )
    stock_resume_selection = _stock_resume_selection(
        exp_dir=stock_resume_exp_dir,
        expected_checkpoint_path=resumed_export_path,
    )
    selected_checkpoint_path = Path(stock_resume_selection["selected_checkpoint_path"])
    verification_report = verify_compact_stock_export_model_contract_v1(
        selected_checkpoint_path,
        seed=int(seed),
        num_simulations=int(num_simulations),
        batch_size=int(batch_size),
        use_cuda=False,
        state_key=COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        raise_on_failure=True,
    )
    _write_json(verification_report_path, verification_report)
    loader_report = _tournament_loader_report(
        stock_export_ref=_relative_ref(selected_checkpoint_path, repo),
        stock_export_path=selected_checkpoint_path,
        seed=int(seed) + 101,
        source_max_steps=int(sidecar["source_max_steps"]),
        num_simulations=int(num_simulations),
        batch_size=int(batch_size),
        telemetry_path=output / "tournament_loader_telemetry.jsonl",
        mount=repo,
    )
    _write_json(tournament_loader_report_path, loader_report)
    stock_bundle = save_compact_stock_export_evidence_bundle_v1(
        selected_checkpoint_path,
        verification_report_path=verification_report_path,
        tournament_loader_report_path=tournament_loader_report_path,
    )
    validate_compact_stock_export_evidence_bundle_v1(stock_bundle["bundle"])

    export_metadata = _required_mapping(export["payload"].get("metadata"), "export metadata")
    loaded_identity = _checkpoint_identity(checkpoint_metadata)
    loaded_identity["model_state_digest"] = compact_model_state_digest_v1(
        getattr(compact_checkpoint, "model_state_dict", {})
    )
    payload = {
        "schema_id": COMPACT_PROMOTION_STOCK_RESUME_LOAD_CANARY_SCHEMA_ID,
        "ok": True,
        "status": "stock_resume_load_canary_verified",
        "run_id": str(run_id),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "readiness_lane": "stock_resume_load_canary",
        "candidate_checkpoint_id": checkpoint_id,
        "input_compatibility": {
            "promotion_eligible": bool(compatibility["promotion_eligible"]),
            "promotion_claim": bool(compatibility["promotion_claim"]),
            "calls_train_muzero": bool(compatibility["calls_train_muzero"]),
            "touches_live_runs": bool(compatibility["touches_live_runs"]),
            "coach_speed_row_gate": bool(compatibility.get("coach_speed_row_gate")),
        },
        "compatibility_report_path": str(compatibility_path),
        "compatibility_report_sha256": _file_sha256(compatibility_path),
        "unified_lifecycle_report_path": str(lifecycle_path),
        "unified_lifecycle_report_sha256": _file_sha256(lifecycle_path),
        "current_chain_evidence_path": str(current_chain_evidence_path),
        "current_chain_evidence_sha256": _file_sha256(current_chain_evidence_path),
        "source_compact_checkpoint_path": str(compact_checkpoint_path),
        "source_compact_checkpoint_sha256": _file_sha256(compact_checkpoint_path),
        "source_policy_sidecar_path": str(sidecar_path),
        "source_policy_sidecar_sha256": _file_sha256(sidecar_path),
        "resumed_stock_export_path": str(selected_checkpoint_path),
        "resumed_stock_export_sha256": _file_sha256(selected_checkpoint_path),
        "resumed_stock_export_sidecar_path": str(export["sidecar_path"]),
        "resumed_stock_export_sidecar_sha256": _file_sha256(export["sidecar_path"]),
        "verification_report_path": str(verification_report_path),
        "verification_report_sha256": _file_sha256(verification_report_path),
        "tournament_loader_report_path": str(tournament_loader_report_path),
        "tournament_loader_report_sha256": _file_sha256(tournament_loader_report_path),
        "stock_export_evidence_bundle_path": str(stock_bundle["path"]),
        "stock_export_evidence_bundle_sha256": _file_sha256(stock_bundle["path"]),
        "loaded_compact_checkpoint_identity": loaded_identity,
        "resumed_stock_export_identity": _stock_export_identity(export_metadata),
        "metadata_acceptance": {
            "ok": True,
            "policy_observation_metadata_required": True,
            "policy_sidecar_path": str(sidecar_path),
            "resumed_policy_sidecar_path": str(export["sidecar_path"]),
        },
        "stock_resume_selection": stock_resume_selection,
        "verification_summary": {
            "ok": bool(verification_report.get("ok")),
            "strict_stock_model_load_verified": bool(
                verification_report.get("strict_stock_model_load_verified")
            ),
            "state_key": str(verification_report.get("state_key", "")),
            "strict": bool(verification_report.get("strict_load")),
        },
        "strict_stock_model_reload": {
            "ok": bool(verification_report.get("ok")),
            "state_key": str(verification_report.get("state_key", "")),
            "strict": bool(verification_report.get("strict_load")),
        },
        "tournament_loader_summary": {
            "ok": bool(loader_report.get("ok")),
            "checkpoint_state_key": str(loader_report.get("checkpoint_state_key", "")),
            "checkpoint_path": str(loader_report.get("checkpoint_path", "")),
        },
        "post_resume_loader_behavior": {
            "tournament_loader_ok": bool(loader_report.get("ok")),
            "eval_loader_ok": bool(verification_report.get("ok")),
        },
        "readiness": {
            "stock_resume_load_canary": True,
            "promotion_readiness_complete": False,
            "remaining_lanes": list(COMPACT_PROMOTION_READINESS_REQUIRED_REMAINING_LANES),
        },
        "attached_claims": {
            "stock_resume_load_canary": True,
            "strict_stock_model_load_verified_after_compact_checkpoint_reload": True,
            "tournament_loader_constructed_after_compact_checkpoint_reload": True,
            "promotion_claim": False,
            "stock_resume_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "rating_or_promotion_quality_claim": False,
            "touches_live_runs": False,
            "calls_train_muzero": False,
        },
        "claims": {
            "metadata_acceptance": True,
            "checkpoint_reload": True,
            "model_identity_and_provenance_preserved": True,
            "eval_loader_behavior_after_resume_selection": True,
            "tournament_loader_behavior_after_resume_selection": True,
            "stock_full_resume_supported": False,
        },
        "non_claims": {
            "promotion_claim": False,
            "stock_resume_claim": False,
            "stock_training_resume_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "rating_or_promotion_quality_claim": False,
            "touches_live_runs": False,
            "calls_train_muzero": False,
        },
    }
    validate_compact_promotion_stock_resume_load_canary_v1(payload)
    return payload


def validate_compact_promotion_stock_resume_load_canary_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate the stock resume/load canary and all referenced file hashes."""

    if payload.get("schema_id") != COMPACT_PROMOTION_STOCK_RESUME_LOAD_CANARY_SCHEMA_ID:
        raise CompactPromotionReadinessError("stock resume/load canary schema mismatch")
    if payload.get("ok") is not True:
        raise CompactPromotionReadinessError("stock resume/load canary must be ok=true")
    if payload.get("readiness_lane") != "stock_resume_load_canary":
        raise CompactPromotionReadinessError("stock resume/load canary lane mismatch")
    if payload.get("status") != "stock_resume_load_canary_verified":
        raise CompactPromotionReadinessError("stock resume/load canary status mismatch")
    input_compatibility = _required_mapping(
        payload.get("input_compatibility"),
        "input_compatibility",
    )
    if input_compatibility.get("promotion_eligible") is not True:
        raise CompactPromotionReadinessError("stock canary input must be eligible")
    for key in ("promotion_claim", "calls_train_muzero", "touches_live_runs"):
        if input_compatibility.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"stock canary input {key} must be false"
            )
    if input_compatibility.get("coach_speed_row_gate") is not True:
        raise CompactPromotionReadinessError("stock canary input speed row missing")
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    for key in (
        "stock_resume_load_canary",
        "strict_stock_model_load_verified_after_compact_checkpoint_reload",
        "tournament_loader_constructed_after_compact_checkpoint_reload",
    ):
        if claims.get(key) is not True:
            raise CompactPromotionReadinessError(f"stock canary {key} not true")
    for key in (
        "promotion_claim",
        "stock_resume_claim",
        "training_speedup_claim",
        "live_run_safety_claim",
        "rating_or_promotion_quality_claim",
        "touches_live_runs",
        "calls_train_muzero",
    ):
        if claims.get(key) is not False:
            raise CompactPromotionReadinessError(f"stock canary claim {key} must be false")
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for key in (
        "promotion_claim",
        "stock_resume_claim",
        "stock_training_resume_claim",
        "training_speedup_claim",
        "live_run_safety_claim",
        "rating_or_promotion_quality_claim",
        "touches_live_runs",
        "calls_train_muzero",
    ):
        if non_claims.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"stock canary non-claim {key} must be false"
            )
    readiness = _required_mapping(payload.get("readiness"), "readiness")
    if readiness.get("stock_resume_load_canary") is not True:
        raise CompactPromotionReadinessError("stock resume/load readiness missing")
    if readiness.get("promotion_readiness_complete") is not False:
        raise CompactPromotionReadinessError(
            "stock resume/load canary must not complete promotion readiness"
        )
    for key in (
        "compatibility_report",
        "unified_lifecycle_report",
        "current_chain_evidence",
        "source_compact_checkpoint",
        "source_policy_sidecar",
        "resumed_stock_export",
        "resumed_stock_export_sidecar",
        "verification_report",
        "tournament_loader_report",
        "stock_export_evidence_bundle",
    ):
        _validate_payload_file(payload, key)
    if not str(payload.get("candidate_checkpoint_id", "")).strip():
        raise CompactPromotionReadinessError("stock canary missing candidate id")
    stock_resume_selection = _required_mapping(
        payload.get("stock_resume_selection"),
        "stock_resume_selection",
    )
    if stock_resume_selection.get("ok") is not True:
        raise CompactPromotionReadinessError("stock resume selection must be ok=true")
    if int(stock_resume_selection.get("selected_iteration", -1)) < 0:
        raise CompactPromotionReadinessError("stock resume selected iteration missing")
    patch = _required_mapping(
        stock_resume_selection.get("load_ckpt_before_run_patch"),
        "load_ckpt_before_run_patch",
    )
    if patch.get("path") != "policy.learn.learner.hook.load_ckpt_before_run":
        raise CompactPromotionReadinessError("stock resume load patch path mismatch")
    if str(patch.get("new", "")) != str(stock_resume_selection.get("selected_checkpoint_path", "")):
        raise CompactPromotionReadinessError("stock resume load patch target mismatch")
    metadata_acceptance = _required_mapping(
        payload.get("metadata_acceptance"),
        "metadata_acceptance",
    )
    if metadata_acceptance.get("ok") is not True:
        raise CompactPromotionReadinessError("metadata acceptance must be ok=true")
    strict_reload = _required_mapping(
        payload.get("strict_stock_model_reload"),
        "strict_stock_model_reload",
    )
    if strict_reload.get("ok") is not True or strict_reload.get("strict") is not True:
        raise CompactPromotionReadinessError("strict stock model reload not proven")
    loader_behavior = _required_mapping(
        payload.get("post_resume_loader_behavior"),
        "post_resume_loader_behavior",
    )
    if loader_behavior.get("tournament_loader_ok") is not True:
        raise CompactPromotionReadinessError("post-resume tournament loader not proven")
    if loader_behavior.get("eval_loader_ok") is not True:
        raise CompactPromotionReadinessError("post-resume eval loader not proven")
    narrow_claims = _required_mapping(payload.get("claims"), "claims")
    for key in (
        "metadata_acceptance",
        "checkpoint_reload",
        "model_identity_and_provenance_preserved",
        "eval_loader_behavior_after_resume_selection",
        "tournament_loader_behavior_after_resume_selection",
    ):
        if narrow_claims.get(key) is not True:
            raise CompactPromotionReadinessError(f"stock canary claim {key} not true")
    if narrow_claims.get("stock_full_resume_supported") is not False:
        raise CompactPromotionReadinessError(
            "stock canary must not claim stock full resume support"
        )


def build_compact_promotion_isolated_live_run_safety_canary_v1(
    *,
    run_id: str,
    unified_lifecycle_report_path: str | Path,
    compatibility_report_path: str | Path,
    stock_resume_load_canary_report_path: str | Path,
    assignment_path: str | Path,
    assignment_audit_path: str | Path,
    trainer_result_path: str | Path,
    metrics_path: str | Path,
    forbidden_touch_audit_path: str | Path,
    initial_checkpoint_path: str | Path,
    final_checkpoint_path: str | Path,
    repo_root: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the isolated live-run safety canary from sandbox run artifacts."""

    repo = Path.cwd().resolve() if repo_root is None else Path(repo_root).resolve()
    lifecycle_path = _resolve_path(unified_lifecycle_report_path, repo)
    compatibility_path = _resolve_path(compatibility_report_path, repo)
    stock_canary_path = _resolve_path(stock_resume_load_canary_report_path, repo)
    assignment_file = _resolve_path(assignment_path, repo)
    assignment_audit_file = _resolve_path(assignment_audit_path, repo)
    trainer_result_file = _resolve_path(trainer_result_path, repo)
    metrics_file = _resolve_path(metrics_path, repo)
    forbidden_touch_file = _resolve_path(forbidden_touch_audit_path, repo)
    initial_checkpoint = _resolve_path(initial_checkpoint_path, repo)
    final_checkpoint = _resolve_path(final_checkpoint_path, repo)

    lifecycle = _read_json_mapping(lifecycle_path, "unified lifecycle report")
    compatibility = _read_json_mapping(compatibility_path, "compatibility report")
    stock_canary = _read_json_mapping(stock_canary_path, "stock resume/load canary")
    assignment = _read_json_mapping(assignment_file, "sandbox assignment")
    assignment_audit = _read_json_mapping(assignment_audit_file, "assignment audit")
    trainer_result = _read_json_mapping(trainer_result_file, "sandbox trainer result")
    metrics = _read_json_mapping(metrics_file, "sandbox metrics")
    forbidden_touch_audit = _read_json_mapping(
        forbidden_touch_file,
        "sandbox forbidden-touch audit",
    )

    _validate_lifecycle_and_compatibility_inputs(
        lifecycle,
        compatibility,
        lifecycle_path=lifecycle_path,
        compatibility_path=compatibility_path,
    )
    validate_compact_promotion_stock_resume_load_canary_v1(stock_canary)
    candidate_checkpoint_id = str(lifecycle["checkpoint_id"])
    _validate_stock_resume_canary_input_for_isolated_live_run(
        stock_canary,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )

    assignment_plumbing = _isolated_live_assignment_plumbing_summary(
        assignment,
        assignment_audit,
        trainer_result=trainer_result,
    )
    trainer_consumption = _isolated_live_trainer_consumption_summary(
        trainer_result,
        stock_canary=stock_canary,
        initial_checkpoint_path=initial_checkpoint,
    )
    checkpoint_and_metrics = _isolated_live_checkpoint_metrics_summary(
        metrics,
        initial_checkpoint_path=initial_checkpoint,
        final_checkpoint_path=final_checkpoint,
    )
    forbidden_touch_summary = _isolated_live_forbidden_touch_summary(
        forbidden_touch_audit,
        trainer_result=trainer_result,
    )

    payload = {
        "schema_id": COMPACT_PROMOTION_ISOLATED_LIVE_RUN_SAFETY_CANARY_SCHEMA_ID,
        "ok": True,
        "status": "isolated_live_run_safety_canary_verified",
        "run_id": str(run_id),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "readiness_lane": "isolated_live_run_safety_canary",
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "input_compatibility": {
            "promotion_eligible": bool(compatibility["promotion_eligible"]),
            "promotion_claim": bool(compatibility["promotion_claim"]),
            "calls_train_muzero": bool(compatibility["calls_train_muzero"]),
            "touches_live_runs": bool(compatibility["touches_live_runs"]),
            "coach_speed_row_gate": bool(compatibility.get("coach_speed_row_gate")),
        },
        "input_stock_resume_load_canary": _isolated_live_stock_resume_summary(
            stock_canary
        ),
        "compatibility_report_path": str(compatibility_path),
        "compatibility_report_sha256": _file_sha256(compatibility_path),
        "unified_lifecycle_report_path": str(lifecycle_path),
        "unified_lifecycle_report_sha256": _file_sha256(lifecycle_path),
        "stock_resume_load_canary_report_path": str(stock_canary_path),
        "stock_resume_load_canary_report_sha256": _file_sha256(stock_canary_path),
        "assignment_path": str(assignment_file),
        "assignment_sha256": _file_sha256(assignment_file),
        "assignment_audit_path": str(assignment_audit_file),
        "assignment_audit_sha256": _file_sha256(assignment_audit_file),
        "trainer_result_path": str(trainer_result_file),
        "trainer_result_sha256": _file_sha256(trainer_result_file),
        "metrics_path": str(metrics_file),
        "metrics_sha256": _file_sha256(metrics_file),
        "forbidden_touch_audit_path": str(forbidden_touch_file),
        "forbidden_touch_audit_sha256": _file_sha256(forbidden_touch_file),
        "initial_checkpoint_path": str(initial_checkpoint),
        "initial_checkpoint_sha256": _file_sha256(initial_checkpoint),
        "final_checkpoint_path": str(final_checkpoint),
        "final_checkpoint_sha256": _file_sha256(final_checkpoint),
        "sandbox_scope": forbidden_touch_summary["sandbox_scope"],
        "forbidden_touch_audit": forbidden_touch_summary["forbidden_touch_audit"],
        "assignment_plumbing": assignment_plumbing,
        "trainer_consumption": trainer_consumption,
        "checkpoint_and_metrics": checkpoint_and_metrics,
        "readiness": {
            "isolated_live_run_safety_canary": True,
            "promotion_readiness_complete": False,
            "other_required_lanes_not_proven_by_this_artifact": [
                lane
                for lane in COMPACT_PROMOTION_READINESS_REQUIRED_REMAINING_LANES
                if lane != "isolated_live_run_safety_canary"
            ],
        },
        "attached_claims": {
            "isolated_live_run_safety_canary": True,
            "sandbox_trainer_launch_boundary_verified": True,
            "assignment_load_apply_boundary_verified": True,
            "checkpoint_write_read_boundary_verified": True,
            "metrics_progress_moved": True,
            "promotion_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "production_live_run_safety_claim": False,
            "stock_resume_claim": False,
            "stock_training_resume_claim": False,
            "rating_or_promotion_quality_claim": False,
            "leaderboard_claim": False,
            "touches_live_runs": False,
            "touches_production_live_runs": False,
            "compact_candidate_calls_train_muzero": False,
        },
        "non_claims": {
            "promotion_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "production_live_run_safety_claim": False,
            "stock_resume_claim": False,
            "stock_training_resume_claim": False,
            "rating_or_promotion_quality_claim": False,
            "leaderboard_claim": False,
            "touches_live_runs": False,
            "touches_production_live_runs": False,
            "compact_candidate_calls_train_muzero": False,
            "sandbox_assignment_rating_proof": False,
        },
    }
    validate_compact_promotion_isolated_live_run_safety_canary_v1(payload)
    return payload


def validate_compact_promotion_isolated_live_run_safety_canary_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate the isolated live-run safety canary and referenced hashes."""

    if (
        payload.get("schema_id")
        != COMPACT_PROMOTION_ISOLATED_LIVE_RUN_SAFETY_CANARY_SCHEMA_ID
    ):
        raise CompactPromotionReadinessError(
            "isolated live-run safety canary schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactPromotionReadinessError(
            "isolated live-run safety canary must be ok=true"
        )
    if payload.get("status") != "isolated_live_run_safety_canary_verified":
        raise CompactPromotionReadinessError(
            "isolated live-run safety canary status mismatch"
        )
    if payload.get("readiness_lane") != "isolated_live_run_safety_canary":
        raise CompactPromotionReadinessError(
            "isolated live-run safety canary lane mismatch"
        )
    candidate_checkpoint_id = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "isolated live-run candidate_checkpoint_id",
    )
    for key in (
        "compatibility_report",
        "unified_lifecycle_report",
        "stock_resume_load_canary_report",
        "assignment",
        "assignment_audit",
        "trainer_result",
        "metrics",
        "forbidden_touch_audit",
        "initial_checkpoint",
        "final_checkpoint",
    ):
        _validate_payload_file_for(payload, key, label="isolated live-run canary")

    compatibility = _read_json_mapping(
        Path(str(payload["compatibility_report_path"])),
        "compatibility report",
    )
    lifecycle = _read_json_mapping(
        Path(str(payload["unified_lifecycle_report_path"])),
        "unified lifecycle report",
    )
    stock_canary = _read_json_mapping(
        Path(str(payload["stock_resume_load_canary_report_path"])),
        "stock resume/load canary",
    )
    _validate_lifecycle_and_compatibility_inputs(
        lifecycle,
        compatibility,
        lifecycle_path=Path(str(payload["unified_lifecycle_report_path"])),
        compatibility_path=Path(str(payload["compatibility_report_path"])),
    )
    if str(lifecycle.get("checkpoint_id")) != candidate_checkpoint_id:
        raise CompactPromotionReadinessError(
            "isolated live-run candidate checkpoint mismatch"
        )
    validate_compact_promotion_stock_resume_load_canary_v1(stock_canary)
    _validate_stock_resume_canary_input_for_isolated_live_run(
        stock_canary,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    _validate_isolated_live_input_compatibility(payload)

    assignment = _read_json_mapping(
        Path(str(payload["assignment_path"])),
        "sandbox assignment",
    )
    assignment_audit = _read_json_mapping(
        Path(str(payload["assignment_audit_path"])),
        "assignment audit",
    )
    trainer_result = _read_json_mapping(
        Path(str(payload["trainer_result_path"])),
        "sandbox trainer result",
    )
    metrics = _read_json_mapping(Path(str(payload["metrics_path"])), "sandbox metrics")
    forbidden_touch_audit = _read_json_mapping(
        Path(str(payload["forbidden_touch_audit_path"])),
        "sandbox forbidden-touch audit",
    )
    initial_checkpoint_path = Path(str(payload["initial_checkpoint_path"]))
    final_checkpoint_path = Path(str(payload["final_checkpoint_path"]))

    expected_assignment = _isolated_live_assignment_plumbing_summary(
        assignment,
        assignment_audit,
        trainer_result=trainer_result,
    )
    expected_trainer = _isolated_live_trainer_consumption_summary(
        trainer_result,
        stock_canary=stock_canary,
        initial_checkpoint_path=initial_checkpoint_path,
    )
    expected_metrics = _isolated_live_checkpoint_metrics_summary(
        metrics,
        initial_checkpoint_path=initial_checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
    )
    expected_forbidden = _isolated_live_forbidden_touch_summary(
        forbidden_touch_audit,
        trainer_result=trainer_result,
    )
    if _json_sha256(payload.get("assignment_plumbing")) != _json_sha256(
        expected_assignment
    ):
        raise CompactPromotionReadinessError(
            "isolated live-run assignment plumbing drift"
        )
    if _json_sha256(payload.get("trainer_consumption")) != _json_sha256(
        expected_trainer
    ):
        raise CompactPromotionReadinessError(
            "isolated live-run trainer consumption drift"
        )
    if _json_sha256(payload.get("checkpoint_and_metrics")) != _json_sha256(
        expected_metrics
    ):
        raise CompactPromotionReadinessError(
            "isolated live-run checkpoint/metrics drift"
        )
    if _json_sha256(payload.get("sandbox_scope")) != _json_sha256(
        expected_forbidden["sandbox_scope"]
    ):
        raise CompactPromotionReadinessError("isolated live-run sandbox scope drift")
    if _json_sha256(payload.get("forbidden_touch_audit")) != _json_sha256(
        expected_forbidden["forbidden_touch_audit"]
    ):
        raise CompactPromotionReadinessError(
            "isolated live-run forbidden-touch drift"
        )

    readiness = _required_mapping(payload.get("readiness"), "readiness")
    if readiness.get("isolated_live_run_safety_canary") is not True:
        raise CompactPromotionReadinessError(
            "isolated live-run readiness flag missing"
        )
    if readiness.get("promotion_readiness_complete") is not False:
        raise CompactPromotionReadinessError(
            "isolated live-run canary must not complete promotion readiness"
        )
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    for key in (
        "isolated_live_run_safety_canary",
        "sandbox_trainer_launch_boundary_verified",
        "assignment_load_apply_boundary_verified",
        "checkpoint_write_read_boundary_verified",
        "metrics_progress_moved",
    ):
        if claims.get(key) is not True:
            raise CompactPromotionReadinessError(f"isolated live-run {key} not true")
    for key in _ISOLATED_LIVE_RUN_FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"isolated live-run claim {key} must be false"
            )
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for key in (*_ISOLATED_LIVE_RUN_FALSE_CLAIM_KEYS, "sandbox_assignment_rating_proof"):
        if non_claims.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"isolated live-run non-claim {key} must be false"
            )


_ISOLATED_LIVE_RUN_FALSE_CLAIM_KEYS = (
    "promotion_claim",
    "training_speedup_claim",
    "live_run_safety_claim",
    "production_live_run_safety_claim",
    "stock_resume_claim",
    "stock_training_resume_claim",
    "rating_or_promotion_quality_claim",
    "leaderboard_claim",
    "touches_live_runs",
    "touches_production_live_runs",
    "compact_candidate_calls_train_muzero",
)

_ISOLATED_LIVE_RUN_FORBIDDEN_TOUCH_KEYS = (
    "production_live_runs_touched",
    "production_intake_touched",
    "production_rating_touched",
    "production_leaderboard_touched",
    "production_control_pointer_touched",
    "writes_checkpoint_intake",
    "spawns_rating",
    "publishes_leaderboard",
    "rewrites_production_control_pointers",
    "uses_production_modal_objects",
    "background_eval_enabled",
    "background_gif_enabled",
)

_ISOLATED_LIVE_RUN_FORBIDDEN_LINEAGE_STAGES = frozenset(
    {
        "checkpoint_intake_seen",
        "checkpoint_intake_enqueued",
        "rating_spawn_claimed",
        "rating_round_started",
        "rating_round_reduced",
        "rating_latest_written",
        "leaderboard_published",
        "training_candidate_assignment_written",
        "assignment_pointer_rewritten",
    }
)

_SANDBOX_ASSIGNMENT_RATING_FALSE_CLAIM_KEYS = (
    "promotion_claim",
    "training_speedup_claim",
    "live_run_safety_claim",
    "production_live_run_safety_claim",
    "stock_resume_claim",
    "stock_training_resume_claim",
    "rating_or_promotion_quality_claim",
    "leaderboard_claim",
    "public_leaderboard_publish_claim",
    "touches_live_runs",
    "touches_production_live_runs",
    "compact_candidate_calls_train_muzero",
)

_SANDBOX_ASSIGNMENT_RATING_FORBIDDEN_TOUCH_KEYS = (
    "production_live_runs_touched",
    "production_intake_touched",
    "production_rating_touched",
    "production_leaderboard_touched",
    "production_control_pointer_touched",
    "writes_checkpoint_intake",
    "spawns_rating",
    "publishes_leaderboard",
    "rewrites_production_control_pointers",
    "uses_production_modal_objects",
    "background_eval_enabled",
    "background_gif_enabled",
    "checkpoint_intake_touched",
    "rating_round_started",
    "rating_latest_written",
    "public_leaderboard_written",
    "leaderboard_pointer_published",
    "training_candidate_assignment_written",
    "assignment_pointer_rewritten",
    "promotion_published",
)

_SANDBOX_ASSIGNMENT_RATING_FORBIDDEN_LINEAGE_STAGES = (
    _ISOLATED_LIVE_RUN_FORBIDDEN_LINEAGE_STAGES
    | frozenset(
        {
            "public_leaderboard_written",
            "leaderboard_pointer_published",
            "production_rating_touched",
            "production_leaderboard_touched",
            "promotion_published",
        }
    )
)


def _validate_stock_resume_canary_input_for_isolated_live_run(
    stock_canary: Mapping[str, Any],
    *,
    candidate_checkpoint_id: str,
) -> None:
    if stock_canary.get("candidate_checkpoint_id") != candidate_checkpoint_id:
        raise CompactPromotionReadinessError(
            "stock resume canary candidate checkpoint mismatch"
        )
    readiness = _required_mapping(stock_canary.get("readiness"), "stock readiness")
    if readiness.get("stock_resume_load_canary") is not True:
        raise CompactPromotionReadinessError("stock resume/load canary input missing")
    if readiness.get("promotion_readiness_complete") is not False:
        raise CompactPromotionReadinessError(
            "stock resume/load canary must not complete promotion readiness"
        )
    non_claims = _required_mapping(stock_canary.get("non_claims"), "stock non_claims")
    for key in (
        "promotion_claim",
        "stock_resume_claim",
        "training_speedup_claim",
        "live_run_safety_claim",
        "rating_or_promotion_quality_claim",
        "touches_live_runs",
    ):
        if non_claims.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"stock resume/load canary non-claim {key} must be false"
            )
    strict_reload = _required_mapping(
        stock_canary.get("strict_stock_model_reload"),
        "stock strict_stock_model_reload",
    )
    loader = _required_mapping(
        stock_canary.get("post_resume_loader_behavior"),
        "stock post_resume_loader_behavior",
    )
    if strict_reload.get("ok") is not True:
        raise CompactPromotionReadinessError("stock resume strict reload missing")
    if loader.get("tournament_loader_ok") is not True or loader.get("eval_loader_ok") is not True:
        raise CompactPromotionReadinessError("stock resume loader proof missing")
    _require_non_empty(
        stock_canary.get("resumed_stock_export_path"),
        "stock resumed_stock_export_path",
    )
    _require_non_empty(
        stock_canary.get("resumed_stock_export_sha256"),
        "stock resumed_stock_export_sha256",
    )


def _isolated_live_stock_resume_summary(
    stock_canary: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_stock_resume_canary_input_for_isolated_live_run(
        stock_canary,
        candidate_checkpoint_id=str(stock_canary.get("candidate_checkpoint_id")),
    )
    strict_reload = _required_mapping(
        stock_canary.get("strict_stock_model_reload"),
        "stock strict_stock_model_reload",
    )
    loader = _required_mapping(
        stock_canary.get("post_resume_loader_behavior"),
        "stock post_resume_loader_behavior",
    )
    return {
        "stock_resume_load_canary": True,
        "promotion_readiness_complete": False,
        "stock_resume_claim": False,
        "strict_stock_model_reload_ok": bool(strict_reload.get("ok")),
        "tournament_loader_ok": bool(loader.get("tournament_loader_ok")),
        "eval_loader_ok": bool(loader.get("eval_loader_ok")),
        "resumed_stock_export_path": str(stock_canary["resumed_stock_export_path"]),
        "resumed_stock_export_sha256": str(stock_canary["resumed_stock_export_sha256"]),
    }


def _isolated_live_assignment_plumbing_summary(
    assignment: Mapping[str, Any],
    assignment_audit: Mapping[str, Any],
    *,
    trainer_result: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        parsed_assignment = parse_opponent_assignment_snapshot(dict(assignment))
        validate_assignment_audit(assignment_audit, assignment=assignment)
    except ValueError as exc:
        raise CompactPromotionReadinessError(
            f"isolated live-run assignment invalid: {exc}"
        ) from exc
    if parsed_assignment is None:
        raise CompactPromotionReadinessError("isolated live-run assignment missing")
    assignment_sha = canonical_assignment_json_sha256(assignment)
    entries = assignment.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CompactPromotionReadinessError(
            "isolated live-run assignment entries missing"
        )
    consumption = _required_mapping(
        trainer_result.get("assignment_consumption"),
        "assignment_consumption",
    )
    if consumption.get("assignment_loaded") is not True:
        raise CompactPromotionReadinessError("isolated live-run assignment not loaded")
    if consumption.get("assignment_applied") is not True:
        raise CompactPromotionReadinessError("isolated live-run assignment not applied")
    if str(consumption.get("trainer_loaded_assignment_sha256", "")) != assignment_sha:
        raise CompactPromotionReadinessError(
            "isolated live-run loaded assignment sha mismatch"
        )
    if str(consumption.get("trainer_applied_assignment_sha256", "")) != assignment_sha:
        raise CompactPromotionReadinessError(
            "isolated live-run applied assignment sha mismatch"
        )
    assignment_application_decision = str(
        consumption.get("assignment_refresh_latest_decision", "")
    )
    if assignment_application_decision not in (
        "applied",
        "initial_assignment_applied",
    ):
        raise CompactPromotionReadinessError(
            "isolated live-run assignment was not applied"
        )
    if int(consumption.get("env_telemetry_row_count", 0)) <= 0:
        raise CompactPromotionReadinessError(
            "isolated live-run assignment env telemetry missing"
        )
    if int(consumption.get("provider_ok_count", 0)) <= 0:
        raise CompactPromotionReadinessError(
            "isolated live-run assignment provider ok count missing"
        )
    if int(consumption.get("provider_false_count", -1)) != 0:
        raise CompactPromotionReadinessError(
            "isolated live-run assignment provider failures present"
        )
    if str(consumption.get("env_telemetry_assignment_sha256", "")) != assignment_sha:
        raise CompactPromotionReadinessError(
            "isolated live-run env telemetry assignment sha mismatch"
        )
    return {
        "assignment_loaded": True,
        "assignment_applied": True,
        "assignment_id": str(assignment["assignment_id"]),
        "assignment_schema_id": str(assignment["schema_id"]),
        "assignment_sha256": assignment_sha,
        "parser_ok": True,
        "parser": "parse_opponent_assignment_snapshot",
        "audit_ok": True,
        "entry_count": len(entries),
        "trainer_loaded_assignment_ref": str(
            consumption.get("trainer_loaded_assignment_ref", "")
        ),
        "trainer_loaded_assignment_sha256": assignment_sha,
        "trainer_applied_assignment_sha256": assignment_sha,
        "assignment_refresh_latest_decision": assignment_application_decision,
        "env_telemetry_row_count": int(consumption["env_telemetry_row_count"]),
        "provider_ok_count": int(consumption["provider_ok_count"]),
        "provider_false_count": int(consumption["provider_false_count"]),
    }


def _isolated_live_trainer_consumption_summary(
    trainer_result: Mapping[str, Any],
    *,
    stock_canary: Mapping[str, Any],
    initial_checkpoint_path: Path,
) -> dict[str, Any]:
    if trainer_result.get("ok") is not True:
        raise CompactPromotionReadinessError("isolated live-run trainer result failed")
    if trainer_result.get("canary_stock_train_muzero_called") is not True:
        raise CompactPromotionReadinessError(
            "isolated live-run stock trainer call missing"
        )
    if trainer_result.get("compact_candidate_calls_train_muzero") is not False:
        raise CompactPromotionReadinessError(
            "compact candidate must not call train_muzero"
        )
    mode = _require_non_empty(trainer_result.get("trainer_mode"), "trainer_mode")
    if mode != "sandbox_canary":
        raise CompactPromotionReadinessError(
            "isolated live-run trainer_mode must be sandbox_canary"
        )
    entrypoint = _require_non_empty(
        trainer_result.get("trainer_entrypoint"),
        "trainer_entrypoint",
    )
    stock_export_path = Path(str(stock_canary["resumed_stock_export_path"]))
    stock_export_sha = str(stock_canary["resumed_stock_export_sha256"])
    if initial_checkpoint_path.resolve() != stock_export_path.resolve():
        raise CompactPromotionReadinessError(
            "isolated live-run initial checkpoint must be the stock-resume export"
        )
    if _file_sha256(initial_checkpoint_path) != stock_export_sha:
        raise CompactPromotionReadinessError(
            "isolated live-run initial checkpoint sha mismatch"
        )
    policy = _required_mapping(
        trainer_result.get("initial_policy_checkpoint"),
        "initial_policy_checkpoint",
    )
    if policy.get("enabled") is not True or policy.get("applied") is not True:
        raise CompactPromotionReadinessError(
            "isolated live-run initial policy checkpoint not applied"
        )
    if policy.get("load_mode") != "matching_shape":
        raise CompactPromotionReadinessError(
            "isolated live-run initial policy load_mode must be matching_shape"
        )
    if policy.get("state_key") != COMPACT_STOCK_EXPORT_MODEL_STATE_KEY:
        raise CompactPromotionReadinessError(
            "isolated live-run initial policy state_key mismatch"
        )
    if str(policy.get("source_sha256", "")) != stock_export_sha:
        raise CompactPromotionReadinessError(
            "isolated live-run initial policy checkpoint sha mismatch"
        )
    prepared = _required_mapping(policy.get("prepared"), "initial policy prepared")
    if prepared.get("kind") != "model_only_checkpoint":
        raise CompactPromotionReadinessError(
            "isolated live-run initial policy must be model-only prepared"
        )
    if prepared.get("fresh_optimizer_intent") is not True:
        raise CompactPromotionReadinessError(
            "isolated live-run fresh optimizer intent missing"
        )
    load_result = _required_mapping(policy.get("load_result"), "initial policy load_result")
    if load_result.get("loaded") is not True:
        raise CompactPromotionReadinessError("isolated live-run initial policy not loaded")
    if load_result.get("fresh_optimizer_preserved") is not True:
        raise CompactPromotionReadinessError(
            "isolated live-run fresh optimizer not preserved"
        )
    errors = load_result.get("errors", [])
    if errors not in (None, []) and len(errors) > 0:
        raise CompactPromotionReadinessError(
            "isolated live-run initial policy load errors present"
        )
    return {
        "canary_stock_train_muzero_called": True,
        "compact_candidate_calls_train_muzero": False,
        "trainer_mode": mode,
        "trainer_entrypoint": entrypoint,
        "load_ckpt_before_run_target": str(
            trainer_result.get("load_ckpt_before_run_target", "")
        ),
        "initial_policy_checkpoint_ref": str(policy.get("checkpoint_ref", "")),
        "initial_policy_checkpoint_source_path": str(policy.get("source_path", "")),
        "initial_policy_checkpoint_sha256": stock_export_sha,
        "load_mode": "matching_shape",
        "state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        "initial_policy_loaded": True,
        "fresh_optimizer_preserved": True,
    }


def _isolated_live_checkpoint_metrics_summary(
    metrics: Mapping[str, Any],
    *,
    initial_checkpoint_path: Path,
    final_checkpoint_path: Path,
) -> dict[str, Any]:
    if metrics.get("ok") is not True:
        raise CompactPromotionReadinessError("isolated live-run metrics failed")
    if metrics.get("checkpoint_write_ok") is not True:
        raise CompactPromotionReadinessError("isolated live-run checkpoint write missing")
    if metrics.get("checkpoint_read_ok") is not True:
        raise CompactPromotionReadinessError("isolated live-run checkpoint read missing")
    if metrics.get("progress_moved") is not True:
        raise CompactPromotionReadinessError("isolated live-run progress did not move")
    if metrics.get("learner_metrics_moved") is not True:
        raise CompactPromotionReadinessError(
            "isolated live-run learner metrics did not move"
        )
    collector_envstep_delta = _positive_int(
        metrics.get("collector_envstep_delta"),
        "collector_envstep_delta",
    )
    learner_train_calls_delta = _positive_int(
        metrics.get("learner_train_calls_delta"),
        "learner_train_calls_delta",
    )
    checkpoint_iteration_delta = _positive_int(
        metrics.get("checkpoint_iteration_delta"),
        "checkpoint_iteration_delta",
    )
    env_telemetry_row_count = _positive_int(
        metrics.get("env_telemetry_row_count"),
        "env_telemetry_row_count",
    )
    training_wall_sec = _positive_float(
        metrics.get("training_wall_sec"),
        "training_wall_sec",
    )
    initial_sha = _file_sha256(initial_checkpoint_path)
    final_sha = _file_sha256(final_checkpoint_path)
    if initial_sha == final_sha:
        raise CompactPromotionReadinessError(
            "isolated live-run final checkpoint must differ from initial"
        )
    return {
        "checkpoint_write_ok": True,
        "checkpoint_read_ok": True,
        "progress_moved": True,
        "learner_metrics_moved": True,
        "collector_envstep_delta": collector_envstep_delta,
        "learner_train_calls_delta": learner_train_calls_delta,
        "checkpoint_iteration_delta": checkpoint_iteration_delta,
        "env_telemetry_row_count": env_telemetry_row_count,
        "training_wall_sec": training_wall_sec,
        "initial_checkpoint_sha256": initial_sha,
        "final_checkpoint_sha256": final_sha,
        "final_checkpoint_differs_from_initial": True,
    }


def _isolated_live_forbidden_touch_summary(
    forbidden_touch_audit: Mapping[str, Any],
    *,
    trainer_result: Mapping[str, Any],
) -> dict[str, Any]:
    if forbidden_touch_audit.get("ok") is not True:
        raise CompactPromotionReadinessError("isolated live-run touch audit failed")
    scope = _required_mapping(forbidden_touch_audit.get("sandbox_scope"), "sandbox_scope")
    namespace = _require_non_empty(scope.get("namespace"), "sandbox namespace")
    if not (namespace.startswith("isolated-") or namespace.startswith("sandbox-")):
        raise CompactPromotionReadinessError(
            "isolated live-run namespace must be isolated or sandbox prefixed"
        )
    if scope.get("isolated") is not True:
        raise CompactPromotionReadinessError("isolated live-run sandbox not isolated")
    if scope.get("production_namespace") is not False:
        raise CompactPromotionReadinessError(
            "isolated live-run production_namespace must be false"
        )
    flags = _required_mapping(
        forbidden_touch_audit.get("forbidden_touch_audit"),
        "forbidden_touch_audit",
    )
    for key in _ISOLATED_LIVE_RUN_FORBIDDEN_TOUCH_KEYS:
        if flags.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"isolated live-run forbidden touch {key} must be false"
            )
    stages = trainer_result.get("lineage_stages", [])
    if stages is None:
        stages = []
    if not isinstance(stages, list):
        raise CompactPromotionReadinessError(
            "isolated live-run lineage_stages must be a list"
        )
    forbidden = sorted(set(str(stage) for stage in stages) & _ISOLATED_LIVE_RUN_FORBIDDEN_LINEAGE_STAGES)
    if forbidden:
        raise CompactPromotionReadinessError(
            f"isolated live-run forbidden lineage stages present: {forbidden!r}"
        )
    return {
        "sandbox_scope": {
            "isolated": True,
            "namespace": namespace,
            "production_namespace": False,
        },
        "forbidden_touch_audit": {key: False for key in _ISOLATED_LIVE_RUN_FORBIDDEN_TOUCH_KEYS},
    }


def _validate_isolated_live_input_compatibility(payload: Mapping[str, Any]) -> None:
    input_compatibility = _required_mapping(
        payload.get("input_compatibility"),
        "input_compatibility",
    )
    if input_compatibility.get("promotion_eligible") is not True:
        raise CompactPromotionReadinessError(
            "isolated live-run input must be compatibility eligible"
        )
    if input_compatibility.get("coach_speed_row_gate") is not True:
        raise CompactPromotionReadinessError(
            "isolated live-run input speed row missing"
        )
    for key in ("promotion_claim", "calls_train_muzero", "touches_live_runs"):
        if input_compatibility.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"isolated live-run input {key} must be false"
            )


def build_compact_promotion_sandbox_assignment_rating_proof_v1(
    *,
    run_id: str,
    unified_lifecycle_report_path: str | Path,
    compatibility_report_path: str | Path,
    stock_resume_load_canary_report_path: str | Path,
    isolated_live_run_safety_canary_report_path: str | Path,
    rating_snapshot_path: str | Path,
    leaderboard_snapshot_path: str | Path,
    leaderboard_pointer_path: str | Path,
    assignment_path: str | Path,
    assignment_audit_path: str | Path,
    forbidden_touch_audit_path: str | Path,
    repo_root: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Bind a local sandbox rating-to-assignment proof without promotion claims."""

    repo = Path.cwd().resolve() if repo_root is None else Path(repo_root).resolve()
    lifecycle_path = _resolve_path(unified_lifecycle_report_path, repo)
    compatibility_path = _resolve_path(compatibility_report_path, repo)
    stock_canary_path = _resolve_path(stock_resume_load_canary_report_path, repo)
    isolated_canary_path = _resolve_path(
        isolated_live_run_safety_canary_report_path,
        repo,
    )
    rating_file = _resolve_path(rating_snapshot_path, repo)
    leaderboard_file = _resolve_path(leaderboard_snapshot_path, repo)
    pointer_file = _resolve_path(leaderboard_pointer_path, repo)
    assignment_file = _resolve_path(assignment_path, repo)
    assignment_audit_file = _resolve_path(assignment_audit_path, repo)
    forbidden_touch_file = _resolve_path(forbidden_touch_audit_path, repo)

    lifecycle = _read_json_mapping(lifecycle_path, "unified lifecycle report")
    compatibility = _read_json_mapping(compatibility_path, "compatibility report")
    stock_canary = _read_json_mapping(stock_canary_path, "stock resume/load canary")
    isolated_canary = _read_json_mapping(
        isolated_canary_path,
        "isolated live-run safety canary",
    )
    rating_snapshot = _read_json_mapping(rating_file, "sandbox rating snapshot")
    leaderboard_snapshot = _read_json_mapping(
        leaderboard_file,
        "sandbox leaderboard snapshot",
    )
    leaderboard_pointer = _read_json_mapping(
        pointer_file,
        "sandbox leaderboard pointer",
    )
    assignment = _read_json_mapping(assignment_file, "sandbox assignment")
    assignment_audit = _read_json_mapping(
        assignment_audit_file,
        "sandbox assignment audit",
    )
    forbidden_touch_audit = _read_json_mapping(
        forbidden_touch_file,
        "sandbox assignment/rating forbidden-touch audit",
    )

    _validate_lifecycle_and_compatibility_inputs(
        lifecycle,
        compatibility,
        lifecycle_path=lifecycle_path,
        compatibility_path=compatibility_path,
    )
    validate_compact_promotion_stock_resume_load_canary_v1(stock_canary)
    validate_compact_promotion_isolated_live_run_safety_canary_v1(isolated_canary)
    candidate_checkpoint_id = str(lifecycle["checkpoint_id"])
    _validate_stock_resume_canary_input_for_isolated_live_run(
        stock_canary,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    _validate_isolated_live_canary_input_for_sandbox_assignment_rating(
        isolated_canary,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )

    rating_summary = _sandbox_assignment_rating_signal_summary(
        rating_snapshot,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    leaderboard_summary = _sandbox_assignment_leaderboard_summary(
        leaderboard_snapshot,
        leaderboard_pointer,
        rating_summary=rating_summary,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    assignment_summary = _sandbox_assignment_materialization_summary(
        assignment,
        assignment_audit,
        leaderboard_summary=leaderboard_summary,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    forbidden_touch_summary = _sandbox_assignment_rating_forbidden_touch_summary(
        forbidden_touch_audit
    )

    compact_checkpoint_path = _resolve_path(
        _require_non_empty(lifecycle.get("compact_checkpoint_path"), "compact_checkpoint_path"),
        lifecycle_path.parent,
    )
    resumed_stock_export_path = Path(
        _require_non_empty(
            stock_canary.get("resumed_stock_export_path"),
            "resumed_stock_export_path",
        )
    )
    if not resumed_stock_export_path.is_absolute():
        resumed_stock_export_path = (stock_canary_path.parent / resumed_stock_export_path).resolve()

    payload = {
        "schema_id": COMPACT_PROMOTION_SANDBOX_ASSIGNMENT_RATING_PROOF_SCHEMA_ID,
        "ok": True,
        "status": "sandbox_assignment_rating_proof_verified",
        "run_id": str(run_id),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "readiness_lane": "sandbox_assignment_rating_proof",
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "input_compatibility": {
            "promotion_eligible": bool(compatibility["promotion_eligible"]),
            "promotion_claim": bool(compatibility["promotion_claim"]),
            "calls_train_muzero": bool(compatibility["calls_train_muzero"]),
            "touches_live_runs": bool(compatibility["touches_live_runs"]),
            "coach_speed_row_gate": bool(compatibility.get("coach_speed_row_gate")),
        },
        "input_stock_resume_load_canary": _isolated_live_stock_resume_summary(
            stock_canary
        ),
        "input_isolated_live_run_safety_canary": {
            "isolated_live_run_safety_canary": True,
            "promotion_readiness_complete": False,
            "candidate_checkpoint_id": str(isolated_canary["candidate_checkpoint_id"]),
        },
        "compatibility_report_path": str(compatibility_path),
        "compatibility_report_sha256": _file_sha256(compatibility_path),
        "unified_lifecycle_report_path": str(lifecycle_path),
        "unified_lifecycle_report_sha256": _file_sha256(lifecycle_path),
        "stock_resume_load_canary_report_path": str(stock_canary_path),
        "stock_resume_load_canary_report_sha256": _file_sha256(stock_canary_path),
        "isolated_live_run_safety_canary_report_path": str(isolated_canary_path),
        "isolated_live_run_safety_canary_report_sha256": _file_sha256(
            isolated_canary_path
        ),
        "source_compact_checkpoint_path": str(compact_checkpoint_path),
        "source_compact_checkpoint_sha256": _file_sha256(compact_checkpoint_path),
        "resumed_stock_export_path": str(resumed_stock_export_path),
        "resumed_stock_export_sha256": _file_sha256(resumed_stock_export_path),
        "rating_snapshot_path": str(rating_file),
        "rating_snapshot_sha256": _file_sha256(rating_file),
        "leaderboard_snapshot_path": str(leaderboard_file),
        "leaderboard_snapshot_sha256": _file_sha256(leaderboard_file),
        "leaderboard_pointer_path": str(pointer_file),
        "leaderboard_pointer_sha256": _file_sha256(pointer_file),
        "assignment_path": str(assignment_file),
        "assignment_sha256": _file_sha256(assignment_file),
        "assignment_audit_path": str(assignment_audit_file),
        "assignment_audit_sha256": _file_sha256(assignment_audit_file),
        "forbidden_touch_audit_path": str(forbidden_touch_file),
        "forbidden_touch_audit_sha256": _file_sha256(forbidden_touch_file),
        "rating_signal": rating_summary,
        "leaderboard_materialization": leaderboard_summary,
        "assignment_materialization": assignment_summary,
        "sandbox_scope": forbidden_touch_summary["sandbox_scope"],
        "forbidden_touch_audit": forbidden_touch_summary["forbidden_touch_audit"],
        "readiness": {
            "sandbox_assignment_rating_proof": True,
            "promotion_readiness_complete": False,
            "input_evidence_lanes_bound": [
                "stock_resume_load_canary",
                "isolated_live_run_safety_canary",
            ],
            "remaining_promotion_readiness_blockers_after_this_artifact": [
                "longer_horizon_compact_learning_metrics"
            ],
            "other_required_lanes_not_proven_by_this_artifact": [
                lane
                for lane in COMPACT_PROMOTION_READINESS_REQUIRED_REMAINING_LANES
                if lane != "sandbox_assignment_rating_proof"
            ],
        },
        "attached_claims": {
            "sandbox_assignment_rating_proof": True,
            "local_rating_snapshot_bound": True,
            "local_leaderboard_snapshot_materialized": True,
            "stable_assignment_parser_verified": True,
            "assignment_audit_verified": True,
            "stock_resume_load_canary_bound": True,
            "isolated_live_run_safety_canary_bound": True,
            "promotion_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "production_live_run_safety_claim": False,
            "stock_resume_claim": False,
            "stock_training_resume_claim": False,
            "rating_or_promotion_quality_claim": False,
            "leaderboard_claim": False,
            "public_leaderboard_publish_claim": False,
            "touches_live_runs": False,
            "touches_production_live_runs": False,
            "compact_candidate_calls_train_muzero": False,
        },
        "non_claims": {
            "promotion_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "production_live_run_safety_claim": False,
            "stock_resume_claim": False,
            "stock_training_resume_claim": False,
            "rating_or_promotion_quality_claim": False,
            "leaderboard_claim": False,
            "public_leaderboard_publish_claim": False,
            "touches_live_runs": False,
            "touches_production_live_runs": False,
            "compact_candidate_calls_train_muzero": False,
        },
    }
    validate_compact_promotion_sandbox_assignment_rating_proof_v1(payload)
    return payload


def validate_compact_promotion_sandbox_assignment_rating_proof_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate a sandbox-only rating-to-assignment proof and bound files."""

    if (
        payload.get("schema_id")
        != COMPACT_PROMOTION_SANDBOX_ASSIGNMENT_RATING_PROOF_SCHEMA_ID
    ):
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating proof schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating proof must be ok=true"
        )
    if payload.get("status") != "sandbox_assignment_rating_proof_verified":
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating proof status mismatch"
        )
    if payload.get("readiness_lane") != "sandbox_assignment_rating_proof":
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating proof lane mismatch"
        )
    candidate_checkpoint_id = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "sandbox assignment/rating candidate_checkpoint_id",
    )
    for key in (
        "compatibility_report",
        "unified_lifecycle_report",
        "stock_resume_load_canary_report",
        "isolated_live_run_safety_canary_report",
        "source_compact_checkpoint",
        "resumed_stock_export",
        "rating_snapshot",
        "leaderboard_snapshot",
        "leaderboard_pointer",
        "assignment",
        "assignment_audit",
        "forbidden_touch_audit",
    ):
        _validate_payload_file_for(
            payload,
            key,
            label="sandbox assignment/rating proof",
        )

    lifecycle = _read_json_mapping(
        Path(str(payload["unified_lifecycle_report_path"])),
        "unified lifecycle report",
    )
    compatibility = _read_json_mapping(
        Path(str(payload["compatibility_report_path"])),
        "compatibility report",
    )
    stock_canary = _read_json_mapping(
        Path(str(payload["stock_resume_load_canary_report_path"])),
        "stock resume/load canary",
    )
    isolated_canary = _read_json_mapping(
        Path(str(payload["isolated_live_run_safety_canary_report_path"])),
        "isolated live-run safety canary",
    )
    _validate_lifecycle_and_compatibility_inputs(
        lifecycle,
        compatibility,
        lifecycle_path=Path(str(payload["unified_lifecycle_report_path"])),
        compatibility_path=Path(str(payload["compatibility_report_path"])),
    )
    if str(lifecycle.get("checkpoint_id")) != candidate_checkpoint_id:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating candidate checkpoint mismatch"
        )
    validate_compact_promotion_stock_resume_load_canary_v1(stock_canary)
    validate_compact_promotion_isolated_live_run_safety_canary_v1(isolated_canary)
    _validate_stock_resume_canary_input_for_isolated_live_run(
        stock_canary,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    _validate_isolated_live_canary_input_for_sandbox_assignment_rating(
        isolated_canary,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    _validate_sandbox_assignment_rating_input_compatibility(payload)

    rating_snapshot = _read_json_mapping(
        Path(str(payload["rating_snapshot_path"])),
        "sandbox rating snapshot",
    )
    leaderboard_snapshot = _read_json_mapping(
        Path(str(payload["leaderboard_snapshot_path"])),
        "sandbox leaderboard snapshot",
    )
    leaderboard_pointer = _read_json_mapping(
        Path(str(payload["leaderboard_pointer_path"])),
        "sandbox leaderboard pointer",
    )
    assignment = _read_json_mapping(
        Path(str(payload["assignment_path"])),
        "sandbox assignment",
    )
    assignment_audit = _read_json_mapping(
        Path(str(payload["assignment_audit_path"])),
        "sandbox assignment audit",
    )
    forbidden_touch_audit = _read_json_mapping(
        Path(str(payload["forbidden_touch_audit_path"])),
        "sandbox assignment/rating forbidden-touch audit",
    )
    expected_rating = _sandbox_assignment_rating_signal_summary(
        rating_snapshot,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    expected_leaderboard = _sandbox_assignment_leaderboard_summary(
        leaderboard_snapshot,
        leaderboard_pointer,
        rating_summary=expected_rating,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    expected_assignment = _sandbox_assignment_materialization_summary(
        assignment,
        assignment_audit,
        leaderboard_summary=expected_leaderboard,
        candidate_checkpoint_id=candidate_checkpoint_id,
    )
    expected_forbidden = _sandbox_assignment_rating_forbidden_touch_summary(
        forbidden_touch_audit
    )
    if _json_sha256(payload.get("rating_signal")) != _json_sha256(expected_rating):
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating rating signal drift"
        )
    if _json_sha256(payload.get("leaderboard_materialization")) != _json_sha256(
        expected_leaderboard
    ):
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating leaderboard materialization drift"
        )
    if _json_sha256(payload.get("assignment_materialization")) != _json_sha256(
        expected_assignment
    ):
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating assignment materialization drift"
        )
    if _json_sha256(payload.get("sandbox_scope")) != _json_sha256(
        expected_forbidden["sandbox_scope"]
    ):
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating sandbox scope drift"
        )
    if _json_sha256(payload.get("forbidden_touch_audit")) != _json_sha256(
        expected_forbidden["forbidden_touch_audit"]
    ):
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating forbidden-touch drift"
        )

    readiness = _required_mapping(payload.get("readiness"), "readiness")
    if readiness.get("sandbox_assignment_rating_proof") is not True:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating readiness flag missing"
        )
    if readiness.get("promotion_readiness_complete") is not False:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating proof must not complete promotion readiness"
        )
    if list(readiness.get("input_evidence_lanes_bound") or []) != [
        "stock_resume_load_canary",
        "isolated_live_run_safety_canary",
    ]:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating input evidence lanes mismatch"
        )
    if list(readiness.get("remaining_promotion_readiness_blockers_after_this_artifact") or []) != [
        "longer_horizon_compact_learning_metrics"
    ]:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating remaining blocker list mismatch"
        )
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    for key in (
        "sandbox_assignment_rating_proof",
        "local_rating_snapshot_bound",
        "local_leaderboard_snapshot_materialized",
        "stable_assignment_parser_verified",
        "assignment_audit_verified",
        "stock_resume_load_canary_bound",
        "isolated_live_run_safety_canary_bound",
    ):
        if claims.get(key) is not True:
            raise CompactPromotionReadinessError(
                f"sandbox assignment/rating claim {key} not true"
            )
    for key in _SANDBOX_ASSIGNMENT_RATING_FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"sandbox assignment/rating claim {key} must be false"
            )
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for key in _SANDBOX_ASSIGNMENT_RATING_FALSE_CLAIM_KEYS:
        if non_claims.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"sandbox assignment/rating non-claim {key} must be false"
            )


def _validate_isolated_live_canary_input_for_sandbox_assignment_rating(
    isolated_canary: Mapping[str, Any],
    *,
    candidate_checkpoint_id: str,
) -> None:
    if isolated_canary.get("candidate_checkpoint_id") != candidate_checkpoint_id:
        raise CompactPromotionReadinessError(
            "isolated live-run canary candidate checkpoint mismatch"
        )
    readiness = _required_mapping(
        isolated_canary.get("readiness"),
        "isolated live-run readiness",
    )
    if readiness.get("isolated_live_run_safety_canary") is not True:
        raise CompactPromotionReadinessError("isolated live-run canary input missing")
    if readiness.get("promotion_readiness_complete") is not False:
        raise CompactPromotionReadinessError(
            "isolated live-run canary must not complete promotion readiness"
        )
    non_claims = _required_mapping(
        isolated_canary.get("non_claims"),
        "isolated live-run non_claims",
    )
    for key in (*_ISOLATED_LIVE_RUN_FALSE_CLAIM_KEYS, "sandbox_assignment_rating_proof"):
        if non_claims.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"isolated live-run canary non-claim {key} must be false"
            )


def _sandbox_assignment_rating_signal_summary(
    rating_snapshot: Mapping[str, Any],
    *,
    candidate_checkpoint_id: str,
) -> dict[str, Any]:
    if rating_snapshot.get("schema_id") != arena.RATING_SNAPSHOT_SCHEMA_ID:
        raise CompactPromotionReadinessError("sandbox rating snapshot schema mismatch")
    if rating_snapshot.get("sandbox_only") is not True:
        raise CompactPromotionReadinessError("sandbox rating snapshot must be sandbox_only")
    if rating_snapshot.get("local_only") is not True:
        raise CompactPromotionReadinessError("sandbox rating snapshot must be local_only")
    if rating_snapshot.get("rating_signal_kind") != "local_tiny_sandbox_tournament_signal":
        raise CompactPromotionReadinessError("sandbox rating signal kind mismatch")
    source = validate_rating_snapshot_source(rating_snapshot)
    pair_count = _positive_int(rating_snapshot.get("pair_count"), "rating pair_count")
    rated_pair_count = _positive_int(
        rating_snapshot.get("rated_pair_count"),
        "rating rated_pair_count",
    )
    if rated_pair_count > pair_count:
        raise CompactPromotionReadinessError("rated_pair_count exceeds pair_count")
    max_abs_delta = _finite_float(
        rating_snapshot.get("max_abs_delta"),
        "rating max_abs_delta",
    )
    pair_results = rating_snapshot.get("pair_rating_results")
    if not isinstance(pair_results, list) or not pair_results:
        raise CompactPromotionReadinessError("rating pair results missing")
    valid_game_count = sum(
        int(result.get("valid_games", 0) or 0)
        for result in pair_results
        if isinstance(result, Mapping)
    )
    if valid_game_count <= 0:
        raise CompactPromotionReadinessError("rating valid games missing")
    rows = rating_snapshot.get("ratings")
    if not isinstance(rows, list) or not rows:
        raise CompactPromotionReadinessError("rating rows missing")
    candidate_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("checkpoint_id", "")) == str(candidate_checkpoint_id)
    ]
    if len(candidate_rows) != 1:
        raise CompactPromotionReadinessError(
            "rating candidate checkpoint missing or duplicated"
        )
    candidate = candidate_rows[0]
    candidate_games = _positive_int(
        candidate.get("games"),
        "rating candidate games",
    )
    candidate_ref = _require_non_empty(
        candidate.get("checkpoint_ref"),
        "rating candidate checkpoint_ref",
    )
    return {
        "rating_signal_kind": "local_tiny_sandbox_tournament_signal",
        "sandbox_only": True,
        "local_only": True,
        "rating_schema_id": str(rating_snapshot["schema_id"]),
        "tournament_id": str(rating_snapshot.get("tournament_id", "")),
        "rating_run_id": str(rating_snapshot.get("rating_run_id", "")),
        "round_id": str(source["round_id"]),
        "round_index": source["round_index"],
        "rating_context_hash": str(source["rating_context_hash"]),
        "roster_hash": str(source["roster_hash"]),
        "rating_snapshot_canonical_sha256": str(source["rating_snapshot_sha256"]),
        "pair_count": pair_count,
        "rated_pair_count": rated_pair_count,
        "valid_game_count": int(valid_game_count),
        "max_abs_delta": max_abs_delta,
        "candidate_checkpoint_id": str(candidate_checkpoint_id),
        "candidate_checkpoint_ref": candidate_ref,
        "candidate_games": candidate_games,
        "candidate_rating": _finite_float(
            candidate.get("rating"),
            "rating candidate rating",
        ),
        "candidate_rank": int(candidate.get("rank", 0) or 0),
        "candidate_status": str(candidate.get("status", "")),
    }


def _sandbox_assignment_leaderboard_summary(
    leaderboard_snapshot: Mapping[str, Any],
    leaderboard_pointer: Mapping[str, Any],
    *,
    rating_summary: Mapping[str, Any],
    candidate_checkpoint_id: str,
) -> dict[str, Any]:
    normalized = validate_leaderboard_snapshot(leaderboard_snapshot)
    pointer = validate_leaderboard_pointer(leaderboard_pointer)
    leaderboard_id = str(normalized["leaderboard_id"])
    _require_sandbox_like_id(leaderboard_id, "leaderboard_id")
    source = _required_mapping(normalized.get("source"), "leaderboard source")
    context = _required_mapping(normalized.get("context"), "leaderboard context")
    if str(source.get("rating_snapshot_sha256", "")) != str(
        rating_summary["rating_snapshot_canonical_sha256"]
    ):
        raise CompactPromotionReadinessError(
            "leaderboard source rating snapshot hash mismatch"
        )
    if str(context.get("rating_context_hash", "")) != str(
        rating_summary["rating_context_hash"]
    ):
        raise CompactPromotionReadinessError(
            "leaderboard rating context hash mismatch"
        )
    if pointer["leaderboard_id"] != leaderboard_id:
        raise CompactPromotionReadinessError("leaderboard pointer id mismatch")
    if pointer["snapshot_id"] != normalized["snapshot_id"]:
        raise CompactPromotionReadinessError("leaderboard pointer snapshot mismatch")
    if pointer["snapshot_sha256"] != normalized["snapshot_sha256"]:
        raise CompactPromotionReadinessError("leaderboard pointer sha mismatch")
    if leaderboard_pointer.get("local_only") is not True:
        raise CompactPromotionReadinessError("leaderboard pointer must be local_only")
    for key in (
        "published",
        "public_leaderboard_written",
        "leaderboard_pointer_published",
    ):
        if leaderboard_pointer.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"leaderboard pointer {key} must be false"
            )
    rows = normalized["rows"]
    active_count = sum(1 for row in rows if row.get("status") == "active")
    provisional_count = sum(1 for row in rows if row.get("status") == "provisional")
    retired_count = sum(1 for row in rows if row.get("status") == "retired")
    candidate_rows = [
        row for row in rows if str(row.get("checkpoint_id", "")) == candidate_checkpoint_id
    ]
    if len(candidate_rows) != 1:
        raise CompactPromotionReadinessError(
            "leaderboard candidate checkpoint missing or duplicated"
        )
    candidate = candidate_rows[0]
    if candidate.get("status") != "active":
        raise CompactPromotionReadinessError(
            "leaderboard candidate checkpoint must be active"
        )
    return {
        "leaderboard_id": leaderboard_id,
        "snapshot_id": str(normalized["snapshot_id"]),
        "snapshot_sha256": str(normalized["snapshot_sha256"]),
        "snapshot_ref": str(pointer["snapshot_ref"]),
        "rating_snapshot_sha256": str(source["rating_snapshot_sha256"]),
        "rating_context_hash": str(context["rating_context_hash"]),
        "row_count": len(rows),
        "active_count": active_count,
        "provisional_count": provisional_count,
        "retired_count": retired_count,
        "candidate_checkpoint_id": str(candidate_checkpoint_id),
        "candidate_checkpoint_ref": str(candidate["checkpoint_ref"]),
        "candidate_rank": int(candidate.get("rank", 0) or 0),
        "candidate_status": "active",
        "local_only": True,
        "pointer_published": False,
        "public_leaderboard_written": False,
    }


def _sandbox_assignment_materialization_summary(
    assignment: Mapping[str, Any],
    assignment_audit: Mapping[str, Any],
    *,
    leaderboard_summary: Mapping[str, Any],
    candidate_checkpoint_id: str,
) -> dict[str, Any]:
    try:
        parsed_assignment = parse_opponent_assignment_snapshot(dict(assignment))
        validate_assignment_audit(assignment_audit, assignment=assignment)
    except ValueError as exc:
        raise CompactPromotionReadinessError(
            f"sandbox assignment invalid: {exc}"
        ) from exc
    if parsed_assignment is None:
        raise CompactPromotionReadinessError("sandbox assignment missing")
    assignment_sha = canonical_assignment_json_sha256(assignment)
    entries = assignment.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CompactPromotionReadinessError("sandbox assignment entries missing")
    source_leaderboard = _required_mapping(
        assignment_audit.get("source_leaderboard"),
        "assignment audit source_leaderboard",
    )
    if str(source_leaderboard.get("leaderboard_id", "")) != str(
        leaderboard_summary["leaderboard_id"]
    ):
        raise CompactPromotionReadinessError(
            "assignment audit leaderboard_id mismatch"
        )
    if str(source_leaderboard.get("snapshot_id", "")) != str(
        leaderboard_summary["snapshot_id"]
    ):
        raise CompactPromotionReadinessError("assignment audit snapshot_id mismatch")
    if str(source_leaderboard.get("snapshot_sha256", "")) != str(
        leaderboard_summary["snapshot_sha256"]
    ):
        raise CompactPromotionReadinessError(
            "assignment audit leaderboard sha mismatch"
        )
    selection = _required_mapping(
        assignment_audit.get("selection"),
        "assignment audit selection",
    )
    if selection.get("strategy_id") != "stable_slots_v1":
        raise CompactPromotionReadinessError(
            "sandbox assignment must use stable_slots_v1"
        )
    selected_rows = assignment_audit.get("selected_rows")
    if not isinstance(selected_rows, list) or not selected_rows:
        raise CompactPromotionReadinessError(
            "sandbox assignment selected rows missing"
        )
    candidate_selected = any(
        isinstance(row, Mapping)
        and str(row.get("checkpoint_id", "")) == candidate_checkpoint_id
        for row in selected_rows
    )
    if not candidate_selected:
        raise CompactPromotionReadinessError(
            "sandbox assignment did not select candidate checkpoint"
        )
    return {
        "assignment_id": str(assignment["assignment_id"]),
        "assignment_schema_id": str(assignment["schema_id"]),
        "assignment_sha256": assignment_sha,
        "parser_ok": True,
        "parser": "parse_opponent_assignment_snapshot",
        "audit_ok": True,
        "strategy_id": "stable_slots_v1",
        "profile": str(selection.get("profile", "")),
        "sentinel": str(selection.get("sentinel", "")),
        "checkpoint_death_mode": str(selection.get("checkpoint_death_mode", "")),
        "entry_count": len(entries),
        "selected_row_count": len(selected_rows),
        "candidate_checkpoint_id": str(candidate_checkpoint_id),
        "candidate_selected": True,
        "source_leaderboard_id": str(source_leaderboard["leaderboard_id"]),
        "source_snapshot_id": str(source_leaderboard["snapshot_id"]),
        "source_snapshot_sha256": str(source_leaderboard["snapshot_sha256"]),
    }


def _sandbox_assignment_rating_forbidden_touch_summary(
    forbidden_touch_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if forbidden_touch_audit.get("ok") is not True:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating touch audit failed"
        )
    scope = _required_mapping(forbidden_touch_audit.get("sandbox_scope"), "sandbox_scope")
    namespace = _require_non_empty(scope.get("namespace"), "sandbox namespace")
    _require_sandbox_like_id(namespace, "sandbox namespace")
    if scope.get("production_namespace") is not False:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating production_namespace must be false"
        )
    if scope.get("local_only") is not True:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating scope must be local_only"
        )
    flags = _required_mapping(
        forbidden_touch_audit.get("forbidden_touch_audit"),
        "forbidden_touch_audit",
    )
    for key in _SANDBOX_ASSIGNMENT_RATING_FORBIDDEN_TOUCH_KEYS:
        if flags.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"sandbox assignment/rating forbidden touch {key} must be false"
            )
    stages = forbidden_touch_audit.get("lineage_stages", [])
    if stages is None:
        stages = []
    if not isinstance(stages, list):
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating lineage_stages must be a list"
        )
    forbidden = sorted(
        set(str(stage) for stage in stages)
        & _SANDBOX_ASSIGNMENT_RATING_FORBIDDEN_LINEAGE_STAGES
    )
    if forbidden:
        raise CompactPromotionReadinessError(
            f"sandbox assignment/rating forbidden lineage stages present: {forbidden!r}"
        )
    return {
        "sandbox_scope": {
            "local_only": True,
            "namespace": namespace,
            "production_namespace": False,
        },
        "forbidden_touch_audit": {
            key: False for key in _SANDBOX_ASSIGNMENT_RATING_FORBIDDEN_TOUCH_KEYS
        },
    }


def _validate_sandbox_assignment_rating_input_compatibility(
    payload: Mapping[str, Any],
) -> None:
    input_compatibility = _required_mapping(
        payload.get("input_compatibility"),
        "input_compatibility",
    )
    if input_compatibility.get("promotion_eligible") is not True:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating input must be compatibility eligible"
        )
    if input_compatibility.get("coach_speed_row_gate") is not True:
        raise CompactPromotionReadinessError(
            "sandbox assignment/rating input speed row missing"
        )
    for key in ("promotion_claim", "calls_train_muzero", "touches_live_runs"):
        if input_compatibility.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"sandbox assignment/rating input {key} must be false"
            )


def _positive_int(value: Any, label: str) -> int:
    number = _positive_float(value, label)
    if not number.is_integer():
        raise CompactPromotionReadinessError(f"{label} must be an integer")
    return int(number)


def _positive_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CompactPromotionReadinessError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise CompactPromotionReadinessError(f"{label} must be finite")
    if number <= 0.0:
        raise CompactPromotionReadinessError(f"{label} must be > 0")
    return number


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CompactPromotionReadinessError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise CompactPromotionReadinessError(f"{label} must be finite")
    return number


def _require_sandbox_like_id(value: str, label: str) -> None:
    text = str(value or "").strip()
    if not text:
        raise CompactPromotionReadinessError(f"{label} must be non-empty")
    allowed_prefixes = ("sandbox-", "local-", "optimizer-", "opt058-")
    if not text.startswith(allowed_prefixes):
        raise CompactPromotionReadinessError(
            f"{label} must be sandbox/local prefixed"
        )


def _validate_lifecycle_and_compatibility_inputs(
    lifecycle: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    *,
    lifecycle_path: Path,
    compatibility_path: Path,
) -> None:
    if lifecycle.get("schema_id") != COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID:
        raise CompactPromotionReadinessError("unified lifecycle schema mismatch")
    if lifecycle.get("ok") is not True:
        raise CompactPromotionReadinessError("unified lifecycle must be ok=true")
    if lifecycle.get("promotion_claim") is not False:
        raise CompactPromotionReadinessError("unified lifecycle promotion_claim must be false")
    if compatibility.get("schema_id") != COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID:
        raise CompactPromotionReadinessError("compatibility refresh schema mismatch")
    if compatibility.get("ok") is not True:
        raise CompactPromotionReadinessError("compatibility refresh must be ok=true")
    if compatibility.get("promotion_eligible") is not True:
        raise CompactPromotionReadinessError(
            "stock resume/load canary requires local compatibility eligibility"
        )
    if compatibility.get("promotion_claim") is not False:
        raise CompactPromotionReadinessError("compatibility promotion_claim must be false")
    if compatibility.get("touches_live_runs") is not False:
        raise CompactPromotionReadinessError("compatibility must not touch live runs")
    if compatibility.get("calls_train_muzero") is not False:
        raise CompactPromotionReadinessError("compatibility must not call train_muzero")
    lifecycle_id = str(lifecycle.get("checkpoint_id", ""))
    compatibility_id = str(compatibility.get("candidate_checkpoint_id", ""))
    if lifecycle_id != compatibility_id:
        raise CompactPromotionReadinessError(
            "compatibility candidate does not match unified lifecycle checkpoint"
        )
    if not lifecycle_path.is_file() or not compatibility_path.is_file():
        raise CompactPromotionReadinessError("stock resume/load input file missing")


def _validate_loaded_checkpoint_identity(
    metadata: Mapping[str, Any],
    *,
    checkpoint_id: str,
    identity: Mapping[str, Any],
) -> None:
    expected = {
        "checkpoint_id": checkpoint_id,
        "trainer_id": identity["trainer_id"],
        "policy_version_ref": identity["policy_version_ref"],
        "model_version_ref": identity["model_version_ref"],
        "policy_source": identity["policy_source"],
    }
    for key, value in expected.items():
        if str(metadata.get(key, "")) != str(value):
            raise CompactPromotionReadinessError(
                f"loaded compact checkpoint {key} mismatch"
            )
    for key in ("promotion_claim", "calls_train_muzero", "touches_live_runs"):
        if metadata.get(key) is not False:
            raise CompactPromotionReadinessError(
                f"loaded compact checkpoint {key} must be false"
            )
    if metadata.get("checkpoint_save_load") is not True:
        raise CompactPromotionReadinessError("loaded checkpoint save/load missing")
    if metadata.get("resume_metadata") is not True:
        raise CompactPromotionReadinessError("loaded checkpoint resume metadata missing")


def _tournament_loader_report(
    *,
    stock_export_ref: str,
    stock_export_path: Path,
    seed: int,
    source_max_steps: int,
    num_simulations: int,
    batch_size: int,
    telemetry_path: Path,
    mount: Path,
) -> dict[str, Any]:
    loaded = arena._load_policy_from_checkpoint(
        checkpoint_ref=stock_export_ref,
        checkpoint_state_key=COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        seed=int(seed),
        source_max_steps=int(source_max_steps),
        num_simulations=int(num_simulations),
        batch_size=int(batch_size),
        telemetry_path=telemetry_path,
        mount=mount,
        remote_root=None,
        model_env_variant=None,
        model_reward_variant=None,
    )
    policy = loaded.pop("policy", None)
    del policy
    return {
        "schema_id": "curvyzero_compact_stock_export_tournament_loader_smoke/v1",
        "ok": True,
        "smoke_scope": (
            "post_compatibility_stock_resume_load_canary_loader_construction"
        ),
        "checkpoint_ref": stock_export_ref,
        "checkpoint_path": str(stock_export_path),
        **loaded,
    }


def _stock_resume_selection(
    *,
    exp_dir: Path,
    expected_checkpoint_path: Path,
) -> dict[str, Any]:
    candidate = lz_checkpoints.latest_lightzero_iteration_checkpoint_from_dirs(
        lz_checkpoints.lightzero_exp_checkpoint_dirs(exp_dir),
        require_non_empty=True,
    )
    if candidate is None:
        raise CompactPromotionReadinessError("stock resume selection found no checkpoint")
    if candidate.path.resolve() != expected_checkpoint_path.resolve():
        raise CompactPromotionReadinessError(
            "stock resume selection did not choose the canary checkpoint"
        )
    main_config: dict[str, Any] = {"policy": {}}
    patch = lz_config.set_load_ckpt_before_run(
        main_config,
        str(candidate.path),
        reason="post-compatibility stock resume/load canary selected checkpoint",
    )
    return {
        "ok": True,
        "selection_policy": "latest_lightzero_iteration_checkpoint_from_dirs",
        "selected_checkpoint_path": str(candidate.path),
        "selected_checkpoint_name": candidate.checkpoint_name,
        "selected_iteration": int(candidate.iteration),
        "selected_size_bytes": int(candidate.size_bytes),
        "candidate_count": len(
            lz_checkpoints.collect_lightzero_iteration_checkpoints(
                lz_checkpoints.lightzero_exp_checkpoint_dirs(exp_dir),
                require_non_empty=True,
            )
        ),
        "resume_state_found": False,
        "load_ckpt_before_run_patch": patch,
        "patched_config_hook": main_config["policy"]["learn"]["learner"]["hook"],
    }


def _policy_sidecar_path_from_evidence(
    evidence: Mapping[str, Any],
    *,
    fallback_stock_export_path: Any,
    base_dir: Path,
) -> Path:
    files = evidence.get("files")
    if isinstance(files, Mapping):
        sidecar = files.get("policy_metadata_sidecar")
        if isinstance(sidecar, Mapping) and sidecar.get("path"):
            return _resolve_path(sidecar["path"], base_dir)
    if fallback_stock_export_path:
        return checkpoint_policy_metadata_sidecar_path(
            _resolve_path(fallback_stock_export_path, base_dir)
        )
    raise CompactPromotionReadinessError("policy sidecar path missing")


def _checkpoint_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "checkpoint_id",
        "trainer_id",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "train_step",
        "learner_update_count",
        "sample_batch_count",
    )
    return {key: _jsonable(metadata.get(key)) for key in keys}


def _stock_export_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "source_compact_checkpoint_id",
        "source_trainer_id",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "stock_model_state_key",
        "stock_resume_claim",
        "optimizer_resume_supported",
        "promotion_claim",
    )
    return {key: _jsonable(metadata.get(key)) for key in keys}


def _validate_payload_file(payload: Mapping[str, Any], key: str) -> None:
    path_text = str(payload.get(f"{key}_path", "")).strip()
    sha = str(payload.get(f"{key}_sha256", "")).strip()
    if not path_text or not sha:
        raise CompactPromotionReadinessError(f"stock canary {key} file record missing")
    path = Path(path_text)
    if not path.is_file():
        raise CompactPromotionReadinessError(f"stock canary {key} file missing")
    if _file_sha256(path) != sha:
        raise CompactPromotionReadinessError(f"stock canary {key} sha256 mismatch")


def _validate_payload_file_for(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str,
) -> None:
    path_text = str(payload.get(f"{key}_path", "")).strip()
    sha = str(payload.get(f"{key}_sha256", "")).strip()
    if not path_text or not sha:
        raise CompactPromotionReadinessError(f"{label} {key} file record missing")
    path = Path(path_text)
    if not path.is_file():
        raise CompactPromotionReadinessError(f"{label} {key} file missing")
    if _file_sha256(path) != sha:
        raise CompactPromotionReadinessError(f"{label} {key} sha256 mismatch")


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompactPromotionReadinessError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompactPromotionReadinessError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, Mapping):
        raise CompactPromotionReadinessError(f"{label} must be a mapping")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactPromotionReadinessError(f"{label} must be a mapping")
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactPromotionReadinessError(f"{label} must be non-empty")
    return text


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _relative_ref(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    data = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "COMPACT_PROMOTION_ISOLATED_LIVE_RUN_SAFETY_CANARY_SCHEMA_ID",
    "COMPACT_PROMOTION_READINESS_REQUIRED_REMAINING_LANES",
    "COMPACT_PROMOTION_SANDBOX_ASSIGNMENT_RATING_PROOF_SCHEMA_ID",
    "COMPACT_PROMOTION_STOCK_RESUME_LOAD_CANARY_SCHEMA_ID",
    "CompactPromotionReadinessError",
    "build_compact_promotion_isolated_live_run_safety_canary_v1",
    "build_compact_promotion_sandbox_assignment_rating_proof_v1",
    "build_compact_promotion_stock_resume_load_canary_v1",
    "validate_compact_promotion_isolated_live_run_safety_canary_v1",
    "validate_compact_promotion_sandbox_assignment_rating_proof_v1",
    "validate_compact_promotion_stock_resume_load_canary_v1",
]
