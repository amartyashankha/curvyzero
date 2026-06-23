"""Final no-promotion review for compact post-compatibility readiness evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from curvyzero.training.compact_longer_horizon_learning_metrics import (
    COMPACT_LONGER_HORIZON_LEARNING_METRICS_SCHEMA_ID,
)
from curvyzero.training.compact_longer_horizon_learning_metrics import (
    validate_compact_longer_horizon_learning_metrics_v1,
)
from curvyzero.training.compact_promotion_readiness import (
    COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness import (
    COMPACT_PROMOTION_ISOLATED_LIVE_RUN_SAFETY_CANARY_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness import (
    COMPACT_PROMOTION_SANDBOX_ASSIGNMENT_RATING_PROOF_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness import (
    COMPACT_PROMOTION_STOCK_RESUME_LOAD_CANARY_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness import (
    COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness import (
    validate_compact_promotion_isolated_live_run_safety_canary_v1,
)
from curvyzero.training.compact_promotion_readiness import (
    validate_compact_promotion_sandbox_assignment_rating_proof_v1,
)
from curvyzero.training.compact_promotion_readiness import (
    validate_compact_promotion_stock_resume_load_canary_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_CANDIDATE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_MATCHED_LEARNING_QUALITY_PAIR_VERIFICATION_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    STOCK_REFERENCE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    validate_compact_matched_learning_quality_canary_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    validate_compact_matched_learning_quality_pair_verification_v1,
)


COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID = (
    "curvyzero_compact_promotion_readiness_bundle_review/v1"
)
COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_STATUS = (
    "local_bundle_reviewed_no_promotion"
)
COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_EVIDENCE_REF_PREFIX = (
    "compact_promotion_readiness_bundle_review:"
)
MATCHED_QUALITY_CANARY_SCALE_DECISION = (
    "canary_scale_manual_acceptance_or_larger_study_required_before_promotion"
)

INPUT_REPORT_KEYS = (
    "compatibility_refresh",
    "unified_lifecycle",
    "matched_learning_quality_canary",
    "matched_pair_verification",
    "stock_resume_load_canary",
    "isolated_live_run_safety_canary",
    "sandbox_assignment_rating_proof",
    "longer_horizon_compact_learning_metrics",
)
EVIDENCE_LANE_KEYS = (
    "matched_learning_quality_canary",
    "matched_pair_verification",
    "stock_resume_load_canary",
    "isolated_live_run_safety_canary",
    "sandbox_assignment_rating_proof",
    "longer_horizon_compact_learning_metrics",
)
FALSE_CLAIM_KEYS = (
    "promotion_claim",
    "training_speedup_claim",
    "live_run_safety_claim",
    "production_live_run_safety_claim",
    "stock_resume_claim",
    "stock_training_resume_claim",
    "rating_or_promotion_quality_claim",
    "compact_quality_superiority_claim",
    "leaderboard_claim",
    "public_leaderboard_claim",
    "public_leaderboard_publish_claim",
    "touches_live_runs",
    "touches_production_live_runs",
    "calls_train_muzero",
    "compact_calls_train_muzero",
    "compact_candidate_calls_train_muzero",
    "promotion_published",
    "automatic_promotion_allowed",
)
TRUE_ATTACHED_CLAIM_KEYS = (
    "post_compatibility_evidence_bundle_reviewed",
    "post_compatibility_evidence_lanes_complete",
    "readiness_bundle_hash_bound",
    "ready_for_manual_review",
)


class CompactPromotionReadinessBundleReviewError(ValueError):
    """Raised when the final compact readiness bundle would overclaim."""


def build_compact_promotion_readiness_bundle_review_v1(
    *,
    run_id: str,
    compatibility_report_path: str | Path,
    unified_lifecycle_report_path: str | Path,
    matched_learning_quality_report_path: str | Path,
    matched_pair_verification_report_path: str | Path,
    stock_resume_load_canary_report_path: str | Path,
    isolated_live_run_safety_canary_report_path: str | Path,
    sandbox_assignment_rating_proof_report_path: str | Path,
    longer_horizon_learning_metrics_report_path: str | Path,
    matched_quality_sufficiency_decision: str = MATCHED_QUALITY_CANARY_SCALE_DECISION,
    matched_quality_sufficiency_rationale: Sequence[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a hash-bound post-compatibility review without claiming promotion."""

    reports = _load_input_reports(
        {
            "compatibility_refresh": compatibility_report_path,
            "unified_lifecycle": unified_lifecycle_report_path,
            "matched_learning_quality_canary": matched_learning_quality_report_path,
            "matched_pair_verification": matched_pair_verification_report_path,
            "stock_resume_load_canary": stock_resume_load_canary_report_path,
            "isolated_live_run_safety_canary": isolated_live_run_safety_canary_report_path,
            "sandbox_assignment_rating_proof": sandbox_assignment_rating_proof_report_path,
            "longer_horizon_compact_learning_metrics": (
                longer_horizon_learning_metrics_report_path
            ),
        }
    )
    validation = _validate_input_reports_and_cross_lanes(reports)
    decision = _review_decision(
        matched_quality_sufficiency_decision=matched_quality_sufficiency_decision,
        matched_quality_sufficiency_rationale=matched_quality_sufficiency_rationale,
    )
    quality = _quality_strength_review(
        reports["matched_learning_quality_canary"]["payload"],
        reports["longer_horizon_compact_learning_metrics"]["payload"],
        matched_quality_sufficiency_decision=matched_quality_sufficiency_decision,
        matched_quality_sufficiency_rationale=matched_quality_sufficiency_rationale,
    )
    non_claims = _bundle_non_claims()
    payload = {
        "schema_id": COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID,
        "ok": True,
        "status": COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_STATUS,
        "run_id": str(run_id),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "candidate_checkpoint_id": validation["candidate_checkpoint_id"],
        "bundle_scope": "post_compatibility_local_readiness_evidence_review",
        "input_reports": _input_report_refs(reports),
        "evidence_lanes": _evidence_lanes(reports),
        "cross_lane_checks": validation["cross_lane_checks"],
        "shared_source_compact_checkpoint": validation[
            "shared_source_compact_checkpoint"
        ],
        "quality_strength_review": quality,
        "forbidden_touch_summary": validation["forbidden_touch_summary"],
        "review_decision": decision,
        "attached_claims": {
            key: True for key in TRUE_ATTACHED_CLAIM_KEYS
        }
        | non_claims,
        "non_claims": non_claims,
    }
    payload["evidence_ref"] = compact_promotion_readiness_bundle_review_evidence_ref(
        payload
    )
    validate_compact_promotion_readiness_bundle_review_v1(payload)
    return payload


def validate_compact_promotion_readiness_bundle_review_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate a final post-compatibility bundle review and bound reports."""

    if payload.get("schema_id") != COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID:
        raise CompactPromotionReadinessBundleReviewError(
            "promotion readiness bundle review schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "promotion readiness bundle review must be ok=true"
        )
    if payload.get("status") != COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_STATUS:
        raise CompactPromotionReadinessBundleReviewError(
            "promotion readiness bundle review status mismatch"
        )
    reports = _load_input_reports_from_review(payload)
    validation = _validate_input_reports_and_cross_lanes(reports)
    if payload.get("candidate_checkpoint_id") != validation["candidate_checkpoint_id"]:
        raise CompactPromotionReadinessBundleReviewError(
            "promotion readiness bundle candidate checkpoint mismatch"
        )
    if _json_sha256(payload.get("cross_lane_checks")) != _json_sha256(
        validation["cross_lane_checks"]
    ):
        raise CompactPromotionReadinessBundleReviewError(
            "promotion readiness bundle cross-lane check drift"
        )
    if _json_sha256(payload.get("forbidden_touch_summary")) != _json_sha256(
        validation["forbidden_touch_summary"]
    ):
        raise CompactPromotionReadinessBundleReviewError(
            "promotion readiness bundle forbidden-touch summary drift"
        )
    if _json_sha256(payload.get("shared_source_compact_checkpoint")) != _json_sha256(
        validation["shared_source_compact_checkpoint"]
    ):
        raise CompactPromotionReadinessBundleReviewError(
            "promotion readiness bundle compact checkpoint binding drift"
        )
    _validate_input_report_refs(payload, reports)
    _validate_evidence_lanes(payload)
    _validate_review_decision(payload)
    _validate_quality_strength_review(payload)
    _validate_bundle_claims(payload)
    expected_ref = compact_promotion_readiness_bundle_review_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactPromotionReadinessBundleReviewError(
            "promotion readiness bundle evidence_ref mismatch"
        )


def compact_promotion_readiness_bundle_review_evidence_ref(
    payload: Mapping[str, Any],
) -> str:
    """Return a stable evidence ref for a validated no-promotion bundle review."""

    candidate = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    input_reports = _required_mapping(payload.get("input_reports"), "input_reports")
    digest_source = {
        key: {
            "schema_id": _required_mapping(input_reports.get(key), key).get(
                "schema_id"
            ),
            "sha256": _required_mapping(input_reports.get(key), key).get("sha256"),
        }
        for key in INPUT_REPORT_KEYS
    }
    return (
        f"{COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_EVIDENCE_REF_PREFIX}"
        f"{candidate}:{_json_sha256(digest_source)[:16]}"
    )


def _load_input_reports(
    paths: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for key in INPUT_REPORT_KEYS:
        path = Path(paths[key]).resolve()
        payload = _read_json_mapping(path, key)
        reports[key] = {
            "path": path,
            "sha256": _file_sha256(path),
            "payload": payload,
        }
    return reports


def _load_input_reports_from_review(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    input_reports = _required_mapping(payload.get("input_reports"), "input_reports")
    paths = {}
    for key in INPUT_REPORT_KEYS:
        ref = _required_mapping(input_reports.get(key), key)
        paths[key] = _require_non_empty(ref.get("path"), f"{key} path")
    reports = _load_input_reports(paths)
    for key in INPUT_REPORT_KEYS:
        ref = _required_mapping(input_reports.get(key), key)
        if reports[key]["sha256"] != ref.get("sha256"):
            raise CompactPromotionReadinessBundleReviewError(
                f"{key} input report sha256 mismatch"
            )
    return reports


def _validate_input_reports_and_cross_lanes(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    compatibility = _payload(reports, "compatibility_refresh")
    lifecycle = _payload(reports, "unified_lifecycle")
    matched = _payload(reports, "matched_learning_quality_canary")
    pair = _payload(reports, "matched_pair_verification")
    stock = _payload(reports, "stock_resume_load_canary")
    isolated = _payload(reports, "isolated_live_run_safety_canary")
    sandbox = _payload(reports, "sandbox_assignment_rating_proof")
    longer = _payload(reports, "longer_horizon_compact_learning_metrics")

    _validate_compatibility_refresh(compatibility)
    _validate_unified_lifecycle(lifecycle)
    validate_compact_matched_learning_quality_canary_v1(matched)
    validate_compact_matched_learning_quality_pair_verification_v1(pair)
    validate_compact_promotion_stock_resume_load_canary_v1(stock)
    validate_compact_promotion_isolated_live_run_safety_canary_v1(isolated)
    validate_compact_promotion_sandbox_assignment_rating_proof_v1(sandbox)
    validate_compact_longer_horizon_learning_metrics_v1(longer)

    candidate = _require_non_empty(lifecycle.get("checkpoint_id"), "checkpoint_id")
    candidate_fields = {
        "compatibility_refresh": compatibility.get("candidate_checkpoint_id"),
        "matched_learning_quality_canary": matched.get("candidate_checkpoint_id"),
        "matched_pair_verification": pair.get("candidate_checkpoint_id"),
        "stock_resume_load_canary": stock.get("candidate_checkpoint_id"),
        "isolated_live_run_safety_canary": isolated.get("candidate_checkpoint_id"),
        "sandbox_assignment_rating_proof": sandbox.get("candidate_checkpoint_id"),
        "longer_horizon_compact_learning_metrics": longer.get("candidate_checkpoint_id"),
    }
    for label, value in candidate_fields.items():
        if str(value) != candidate:
            raise CompactPromotionReadinessBundleReviewError(
                f"{label} candidate_checkpoint_id mismatch"
            )
    _validate_report_schema_and_ok(reports)
    _validate_input_hash_bindings(reports)
    _validate_false_claims_across_inputs(
        (compatibility, lifecycle, matched, pair, stock, isolated, sandbox, longer)
    )
    _validate_matched_compact_call_contract(matched)
    checkpoint_binding = _shared_source_compact_checkpoint(
        compatibility,
        matched,
        stock,
        sandbox,
        longer,
    )
    cross_lane_checks = {
        "same_candidate_checkpoint_id": True,
        "compatibility_promotion_eligible": compatibility.get("promotion_eligible")
        is True,
        "compatibility_promotion_claim_false": compatibility.get("promotion_claim")
        is False,
        "compatibility_coach_speed_row_gate": compatibility.get("coach_speed_row_gate")
        is True,
        "unified_lifecycle_gates_complete": lifecycle.get("lifecycle_gates_complete")
        is True,
        "all_input_reports_hash_bound": True,
        "all_lane_validators_passed": True,
        "matched_pair_verifier_binds_canary": True,
        "stock_resume_report_bound_into_isolated_live": True,
        "stock_resume_report_bound_into_sandbox_assignment_rating": True,
        "isolated_live_report_bound_into_sandbox_assignment_rating": True,
        "shared_source_compact_checkpoint_hash_bound": bool(
            checkpoint_binding.get("sha256")
        ),
        "no_production_live_run_touch": True,
        "no_public_leaderboard_publish": True,
        "compact_candidate_calls_train_muzero_false": True,
    }
    for key, value in cross_lane_checks.items():
        if value is not True:
            raise CompactPromotionReadinessBundleReviewError(
                f"cross-lane check failed: {key}"
            )
    return {
        "candidate_checkpoint_id": candidate,
        "cross_lane_checks": cross_lane_checks,
        "shared_source_compact_checkpoint": checkpoint_binding,
        "forbidden_touch_summary": _forbidden_touch_summary(isolated, sandbox),
    }


def _validate_compatibility_refresh(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_id") != COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID:
        raise CompactPromotionReadinessBundleReviewError(
            "compatibility refresh schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "compatibility refresh must be ok=true"
        )
    if payload.get("promotion_eligible") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "compatibility refresh must be locally eligible"
        )
    if payload.get("coach_speed_row_gate") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "compatibility refresh speed row gate missing"
        )
    for key in ("promotion_claim", "calls_train_muzero", "touches_live_runs"):
        if payload.get(key) is not False:
            raise CompactPromotionReadinessBundleReviewError(
                f"compatibility refresh {key} must be false"
            )
    _require_non_empty(payload.get("candidate_checkpoint_id"), "candidate_checkpoint_id")


def _validate_unified_lifecycle(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_id") != COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID:
        raise CompactPromotionReadinessBundleReviewError(
            "unified lifecycle schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "unified lifecycle must be ok=true"
        )
    if payload.get("lifecycle_gates_complete") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "unified lifecycle gates must be complete"
        )
    if payload.get("promotion_claim") is not False:
        raise CompactPromotionReadinessBundleReviewError(
            "unified lifecycle promotion_claim must be false"
        )
    _require_non_empty(payload.get("checkpoint_id"), "checkpoint_id")


def _validate_report_schema_and_ok(
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {
        "compatibility_refresh": COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID,
        "unified_lifecycle": COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID,
        "matched_learning_quality_canary": (
            COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID
        ),
        "matched_pair_verification": (
            COMPACT_MATCHED_LEARNING_QUALITY_PAIR_VERIFICATION_SCHEMA_ID
        ),
        "stock_resume_load_canary": (
            COMPACT_PROMOTION_STOCK_RESUME_LOAD_CANARY_SCHEMA_ID
        ),
        "isolated_live_run_safety_canary": (
            COMPACT_PROMOTION_ISOLATED_LIVE_RUN_SAFETY_CANARY_SCHEMA_ID
        ),
        "sandbox_assignment_rating_proof": (
            COMPACT_PROMOTION_SANDBOX_ASSIGNMENT_RATING_PROOF_SCHEMA_ID
        ),
        "longer_horizon_compact_learning_metrics": (
            COMPACT_LONGER_HORIZON_LEARNING_METRICS_SCHEMA_ID
        ),
    }
    for key, schema_id in expected.items():
        payload = _payload(reports, key)
        if payload.get("schema_id") != schema_id:
            raise CompactPromotionReadinessBundleReviewError(
                f"{key} schema mismatch"
            )
        if payload.get("ok") is not True:
            raise CompactPromotionReadinessBundleReviewError(
                f"{key} must be ok=true"
            )


def _validate_input_hash_bindings(
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    compatibility_sha = str(reports["compatibility_refresh"]["sha256"])
    lifecycle_sha = str(reports["unified_lifecycle"]["sha256"])
    matched_sha = str(reports["matched_learning_quality_canary"]["sha256"])
    stock_sha = str(reports["stock_resume_load_canary"]["sha256"])
    isolated_sha = str(reports["isolated_live_run_safety_canary"]["sha256"])

    for key in (
        "matched_learning_quality_canary",
        "stock_resume_load_canary",
        "isolated_live_run_safety_canary",
        "sandbox_assignment_rating_proof",
        "longer_horizon_compact_learning_metrics",
    ):
        payload = _payload(reports, key)
        _require_equal(
            payload.get("compatibility_report_sha256"),
            compatibility_sha,
            f"{key} compatibility_report_sha256",
        )
        _require_equal(
            payload.get("unified_lifecycle_report_sha256"),
            lifecycle_sha,
            f"{key} unified_lifecycle_report_sha256",
        )
    pair_report = _required_mapping(
        _payload(reports, "matched_pair_verification").get(
            "matched_learning_quality_report"
        ),
        "matched_learning_quality_report",
    )
    _require_equal(
        pair_report.get("sha256"),
        matched_sha,
        "matched pair report sha256",
    )
    pair_path = Path(_require_non_empty(pair_report.get("path"), "matched pair path"))
    if pair_path.resolve() != Path(reports["matched_learning_quality_canary"]["path"]):
        raise CompactPromotionReadinessBundleReviewError(
            "matched pair verifier path does not point at matched canary"
        )
    for key in ("isolated_live_run_safety_canary", "sandbox_assignment_rating_proof"):
        _require_equal(
            _payload(reports, key).get("stock_resume_load_canary_report_sha256"),
            stock_sha,
            f"{key} stock_resume_load_canary_report_sha256",
        )
    _require_equal(
        _payload(reports, "sandbox_assignment_rating_proof").get(
            "isolated_live_run_safety_canary_report_sha256"
        ),
        isolated_sha,
        "sandbox isolated_live_run_safety_canary_report_sha256",
    )


def _validate_false_claims_across_inputs(
    payloads: Sequence[Mapping[str, Any]],
) -> None:
    for payload in payloads:
        label = str(payload.get("schema_id", "input"))
        for key in FALSE_CLAIM_KEYS:
            if key in payload and payload.get(key) is not False:
                raise CompactPromotionReadinessBundleReviewError(
                    f"{label} {key} must be false"
                )
        for section_name in ("attached_claims", "non_claims"):
            section = payload.get(section_name)
            if not isinstance(section, Mapping):
                continue
            for key in FALSE_CLAIM_KEYS:
                if key in section and section.get(key) is not False:
                    raise CompactPromotionReadinessBundleReviewError(
                        f"{label} {section_name}.{key} must be false"
                    )


def _validate_matched_compact_call_contract(matched: Mapping[str, Any]) -> None:
    call_contract = _required_mapping(
        matched.get("arm_call_contract"),
        "arm_call_contract",
    )
    if call_contract.get("stock_reference_calls_train_muzero") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "matched quality stock reference must call train_muzero"
        )
    if call_contract.get("compact_candidate_calls_train_muzero") is not False:
        raise CompactPromotionReadinessBundleReviewError(
            "matched quality compact candidate must not call train_muzero"
        )
    if call_contract.get("both_touch_live_runs") is not False:
        raise CompactPromotionReadinessBundleReviewError(
            "matched quality must not touch live runs"
        )
    arms = _required_mapping(matched.get("arms"), "matched quality arms")
    compact = _required_mapping(arms.get(COMPACT_CANDIDATE_ROLE), COMPACT_CANDIDATE_ROLE)
    if compact.get("calls_train_muzero") is not False:
        raise CompactPromotionReadinessBundleReviewError(
            "compact candidate arm calls_train_muzero must be false"
        )


def _shared_source_compact_checkpoint(
    compatibility: Mapping[str, Any],
    matched: Mapping[str, Any],
    stock: Mapping[str, Any],
    sandbox: Mapping[str, Any],
    longer: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _required_mapping(
        compatibility.get("loaded_checkpoint_identity"),
        "loaded_checkpoint_identity",
    )
    expected = _require_non_empty(
        identity.get("compact_checkpoint_sha256"),
        "compatibility compact checkpoint sha256",
    )
    checks: dict[str, str] = {
        "compatibility_loaded_checkpoint": expected,
        "stock_resume_source_compact_checkpoint": _require_non_empty(
            stock.get("source_compact_checkpoint_sha256"),
            "stock source compact checkpoint sha256",
        ),
        "sandbox_source_compact_checkpoint": _require_non_empty(
            sandbox.get("source_compact_checkpoint_sha256"),
            "sandbox source compact checkpoint sha256",
        ),
        "longer_horizon_loaded_compact_checkpoint": _require_non_empty(
            longer.get("loaded_compact_checkpoint_sha256"),
            "longer loaded compact checkpoint sha256",
        ),
    }
    arms = _required_mapping(matched.get("arms"), "matched quality arms")
    compact = _required_mapping(arms.get(COMPACT_CANDIDATE_ROLE), COMPACT_CANDIDATE_ROLE)
    fingerprint = _required_mapping(
        compact.get("source_fingerprint"),
        "compact source_fingerprint",
    )
    checks["matched_compact_loaded_checkpoint"] = _require_non_empty(
        fingerprint.get("loaded_compact_checkpoint_sha256"),
        "matched compact loaded checkpoint sha256",
    )
    for label, value in checks.items():
        if value != expected:
            raise CompactPromotionReadinessBundleReviewError(
                f"{label} does not match shared compact checkpoint sha256"
            )
    return {
        "sha256": expected,
        "model_state_digest": str(identity.get("model_state_digest", "")),
        "bindings": checks,
    }


def _forbidden_touch_summary(
    isolated: Mapping[str, Any],
    sandbox: Mapping[str, Any],
) -> dict[str, Any]:
    for label, payload in (("isolated", isolated), ("sandbox", sandbox)):
        section = _required_mapping(
            payload.get("forbidden_touch_audit"),
            f"{label} forbidden_touch_audit",
        )
        for key, value in section.items():
            if value is not False:
                raise CompactPromotionReadinessBundleReviewError(
                    f"{label} forbidden touch {key} must be false"
                )
    sandbox_scope = _required_mapping(sandbox.get("sandbox_scope"), "sandbox_scope")
    if sandbox_scope.get("local_only") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "sandbox scope must be local_only=true"
        )
    if sandbox_scope.get("production_namespace") is not False:
        raise CompactPromotionReadinessBundleReviewError(
            "sandbox scope must not be production"
        )
    return {
        "touches_live_runs": False,
        "touches_production_live_runs": False,
        "production_live_run_safety_claim": False,
        "public_leaderboard_publish_claim": False,
        "promotion_published": False,
        "sandbox_only": True,
        "local_only": True,
    }


def _input_report_refs(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for key in INPUT_REPORT_KEYS:
        payload = _payload(reports, key)
        refs[key] = {
            "path": str(reports[key]["path"]),
            "sha256": str(reports[key]["sha256"]),
            "schema_id": str(payload.get("schema_id", "")),
            "ok": bool(payload.get("ok")),
            "candidate_checkpoint_id": _candidate_for_input(key, payload),
            "status": str(payload.get("status", "")),
            "readiness_lane": str(payload.get("readiness_lane", "")),
            "evidence_ref": str(payload.get("evidence_ref", "")),
        }
    return refs


def _evidence_lanes(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    lanes: dict[str, dict[str, Any]] = {}
    for key in EVIDENCE_LANE_KEYS:
        payload = _payload(reports, key)
        readiness = payload.get("readiness")
        lanes[key] = {
            "present": True,
            "validated": True,
            "schema_id": str(payload.get("schema_id", "")),
            "report_sha256": str(reports[key]["sha256"]),
            "readiness_lane": str(payload.get("readiness_lane", "")),
            "promotion_readiness_complete": False
            if key == "matched_pair_verification"
            else (
                _required_mapping(readiness, f"{key} readiness").get(
                    "promotion_readiness_complete"
                )
                if isinstance(readiness, Mapping)
                else None
            ),
        }
    return lanes


def _quality_strength_review(
    matched: Mapping[str, Any],
    longer: Mapping[str, Any],
    *,
    matched_quality_sufficiency_decision: str,
    matched_quality_sufficiency_rationale: Sequence[str] | None,
) -> dict[str, Any]:
    matched_pair = _required_mapping(matched.get("matched_pair"), "matched_pair")
    arms = _required_mapping(matched.get("arms"), "matched arms")
    compact = _required_mapping(arms.get(COMPACT_CANDIDATE_ROLE), COMPACT_CANDIDATE_ROLE)
    stock = _required_mapping(arms.get(STOCK_REFERENCE_ROLE), STOCK_REFERENCE_ROLE)
    eval_settings = _required_mapping(compact.get("eval_settings"), "compact eval_settings")
    eval_seed_set = eval_settings.get("eval_seed_set")
    if not isinstance(eval_seed_set, Sequence) or isinstance(eval_seed_set, str):
        eval_seed_count = 0
    else:
        eval_seed_count = len(eval_seed_set)
    series = longer.get("checkpoint_series")
    checkpoint_count = len(series) if isinstance(series, list) else 0
    denominators = _required_mapping(
        longer.get("cumulative_denominators"),
        "longer cumulative_denominators",
    )
    rationale = list(matched_quality_sufficiency_rationale or _default_quality_rationale())
    return {
        "matched_quality_sufficiency_decision": str(
            matched_quality_sufficiency_decision
        ),
        "matched_quality_sufficiency_rationale": rationale,
        "current_matched_canary_sufficient_for_promotion_claim": False,
        "manual_acceptance_or_larger_study_required_before_promotion": True,
        "promotion_quality_claim": False,
        "compact_quality_superiority_claim": False,
        "matched_pair_summary": {
            "denominator_id": str(matched_pair.get("denominator_id", "")),
            "quality_horizon": str(matched_pair.get("quality_horizon", "")),
            "hardware_class": str(matched_pair.get("hardware_class", "")),
            "stock_hardware_class": str(matched_pair.get("stock_hardware_class", "")),
            "compact_hardware_class": str(
                matched_pair.get("compact_hardware_class", "")
            ),
            "eval_seed_count": int(eval_seed_count),
            "eval_max_steps": int(eval_settings.get("eval_max_steps", 0) or 0),
            "stock_calls_train_muzero": stock.get("calls_train_muzero") is True,
            "compact_calls_train_muzero": compact.get("calls_train_muzero") is True,
        },
        "longer_horizon_summary": {
            "checkpoint_count": int(checkpoint_count),
            "learner_update_count_delta": int(
                denominators.get("learner_update_count_delta", 0) or 0
            ),
            "sample_batch_count_delta": int(
                denominators.get("sample_batch_count_delta", 0) or 0
            ),
            "promotion_quality_claim": False,
        },
        "residual_risks": [
            "matched quality is canary-scale and does not prove superiority",
            "sandbox rating proves plumbing, not public rating quality",
            "isolated live-run safety is sandbox-only",
            "longer-horizon compact metrics are a minimum-shape local trace",
        ],
    }


def _review_decision(
    *,
    matched_quality_sufficiency_decision: str,
    matched_quality_sufficiency_rationale: Sequence[str] | None,
) -> dict[str, Any]:
    if str(matched_quality_sufficiency_decision).strip() in {
        "",
        "sufficient_for_promotion",
        "promotion_ready",
    }:
        raise CompactPromotionReadinessBundleReviewError(
            "matched quality sufficiency decision would overclaim"
        )
    return {
        "ready_for_manual_review": True,
        "post_compatibility_evidence_lanes_complete": True,
        "promotion_claim": False,
        "automatic_promotion_allowed": False,
        "manual_review_required_before_any_promotion": True,
        "matched_quality_sufficiency_decision": str(
            matched_quality_sufficiency_decision
        ),
        "matched_quality_sufficiency_rationale": list(
            matched_quality_sufficiency_rationale or _default_quality_rationale()
        ),
    }


def _default_quality_rationale() -> tuple[str, ...]:
    return (
        "matched quality is a canary and does not claim compact superiority",
        "assignment/rating evidence is local sandbox plumbing",
        "promotion remains a separate manual or future automated decision",
    )


def _bundle_non_claims() -> dict[str, bool]:
    return {key: False for key in FALSE_CLAIM_KEYS}


def _validate_input_report_refs(
    payload: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    refs = _required_mapping(payload.get("input_reports"), "input_reports")
    for key in INPUT_REPORT_KEYS:
        ref = _required_mapping(refs.get(key), key)
        if Path(str(ref.get("path", ""))).resolve() != reports[key]["path"]:
            raise CompactPromotionReadinessBundleReviewError(
                f"{key} input report path drift"
            )
        if ref.get("sha256") != reports[key]["sha256"]:
            raise CompactPromotionReadinessBundleReviewError(
                f"{key} input report sha drift"
            )
        source = _payload(reports, key)
        if ref.get("schema_id") != source.get("schema_id"):
            raise CompactPromotionReadinessBundleReviewError(
                f"{key} input schema drift"
            )
        if ref.get("ok") is not True:
            raise CompactPromotionReadinessBundleReviewError(
                f"{key} input ok flag drift"
            )


def _validate_evidence_lanes(payload: Mapping[str, Any]) -> None:
    lanes = _required_mapping(payload.get("evidence_lanes"), "evidence_lanes")
    for key in EVIDENCE_LANE_KEYS:
        lane = _required_mapping(lanes.get(key), key)
        if lane.get("present") is not True or lane.get("validated") is not True:
            raise CompactPromotionReadinessBundleReviewError(
                f"{key} evidence lane missing"
            )
        if lane.get("promotion_readiness_complete") is not False:
            raise CompactPromotionReadinessBundleReviewError(
                f"{key} must not independently complete promotion readiness"
            )


def _validate_review_decision(payload: Mapping[str, Any]) -> None:
    decision = _required_mapping(payload.get("review_decision"), "review_decision")
    if decision.get("ready_for_manual_review") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "bundle review must be ready for manual review"
        )
    if decision.get("post_compatibility_evidence_lanes_complete") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "bundle review lanes must be complete"
        )
    for key in ("promotion_claim", "automatic_promotion_allowed"):
        if decision.get(key) is not False:
            raise CompactPromotionReadinessBundleReviewError(
                f"bundle review decision {key} must be false"
            )
    if decision.get("manual_review_required_before_any_promotion") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "bundle review must require manual review before promotion"
        )
    if not str(decision.get("matched_quality_sufficiency_decision", "")).strip():
        raise CompactPromotionReadinessBundleReviewError(
            "matched quality sufficiency decision missing"
        )


def _validate_quality_strength_review(payload: Mapping[str, Any]) -> None:
    quality = _required_mapping(
        payload.get("quality_strength_review"),
        "quality_strength_review",
    )
    if quality.get("current_matched_canary_sufficient_for_promotion_claim") is not False:
        raise CompactPromotionReadinessBundleReviewError(
            "matched canary must not be sufficient for promotion claim"
        )
    if quality.get("manual_acceptance_or_larger_study_required_before_promotion") is not True:
        raise CompactPromotionReadinessBundleReviewError(
            "matched quality must require manual acceptance or larger study"
        )
    for key in ("promotion_quality_claim", "compact_quality_superiority_claim"):
        if quality.get(key) is not False:
            raise CompactPromotionReadinessBundleReviewError(
                f"quality review {key} must be false"
            )


def _validate_bundle_claims(payload: Mapping[str, Any]) -> None:
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    for key in TRUE_ATTACHED_CLAIM_KEYS:
        if claims.get(key) is not True:
            raise CompactPromotionReadinessBundleReviewError(
                f"bundle attached claim {key} missing"
            )
    for key in FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactPromotionReadinessBundleReviewError(
                f"bundle attached claim {key} must be false"
            )
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for key in FALSE_CLAIM_KEYS:
        if non_claims.get(key) is not False:
            raise CompactPromotionReadinessBundleReviewError(
                f"bundle non-claim {key} must be false"
            )


def _candidate_for_input(key: str, payload: Mapping[str, Any]) -> str:
    if key == "unified_lifecycle":
        return str(payload.get("checkpoint_id", ""))
    return str(payload.get("candidate_checkpoint_id", ""))


def _payload(
    reports: Mapping[str, Mapping[str, Any]],
    key: str,
) -> Mapping[str, Any]:
    value = reports[key].get("payload")
    if not isinstance(value, Mapping):
        raise CompactPromotionReadinessBundleReviewError(f"{key} payload missing")
    return value


def _require_equal(left: Any, right: Any, label: str) -> None:
    if str(left) != str(right):
        raise CompactPromotionReadinessBundleReviewError(f"{label} mismatch")


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompactPromotionReadinessBundleReviewError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompactPromotionReadinessBundleReviewError(
            f"{label} is not readable JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CompactPromotionReadinessBundleReviewError(f"{label} must be a mapping")
    return dict(payload)


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactPromotionReadinessBundleReviewError(f"{label} must be a mapping")
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactPromotionReadinessBundleReviewError(
            f"{label} must be non-empty"
        )
    return text


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
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
