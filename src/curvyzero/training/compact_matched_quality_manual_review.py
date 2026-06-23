"""Manual-review decision for larger compact matched-quality evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from curvyzero.training.compact_matched_quality_sufficiency_review import (
    DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    STATUS_CURRENT_CANARY_ACCEPTED_FOR_NEXT_NON_PRODUCTION_STEP,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    validate_compact_matched_quality_sufficiency_review_v1,
)


COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_SCHEMA_ID = (
    "curvyzero_compact_matched_quality_manual_review/v1"
)
COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_MANIFEST_SCHEMA_ID = (
    "curvyzero_compact_matched_quality_manual_review_manifest/v1"
)
COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_EVIDENCE_REF_PREFIX = (
    "compact_matched_quality_manual_review:"
)

STATUS_MANUAL_REVIEW_RECORDED_NO_PROMOTION = (
    "larger_32x2048_matched_quality_manual_review_recorded_no_promotion"
)
MANUAL_REVIEW_SCOPE = "manual_review_of_larger_32x2048_matched_quality_packet"
REVIEWED_PACKET = "larger_32x2048_matched_quality_manual_review_not_promotion"

DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL = (
    "support_named_non_production_promotion_proposal"
)
DECISION_REQUIRE_LARGER_REPEAT = "require_larger_repeat_before_proposal"
DECISION_KEEP_COMPACT_CANDIDATE_ONLY = "keep_compact_candidate_only"
ALLOWED_MANUAL_DECISIONS = (
    DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL,
    DECISION_REQUIRE_LARGER_REPEAT,
    DECISION_KEEP_COMPACT_CANDIDATE_ONLY,
)

DEFAULT_RUN_ID = (
    "optimizer-compact-matched-quality-manual-review-larger-2048x32-"
    "env64train8-20260531"
)
DEFAULT_NEXT_ALLOWED_STEP = "return_to_engineering_speed_and_stock_evaluator_hardening"
DEFAULT_CANONICAL_STOCK_FAILURE_RUN_ID = (
    "optimizer-stock-reference-quality-producer-larger-2048x32-20260531"
)
DEFAULT_ACCEPTED_STOCK_CAPTURE_RUN_ID = (
    "optimizer-stock-reference-quality-producer-larger-2048x32-20260531-evalenv32"
)

FALSE_CLAIM_KEYS = (
    "promotion_claim",
    "automatic_promotion_allowed",
    "production_promotion_allowed",
    "promotion_published",
    "training_speedup_claim",
    "stock_train_muzero_speedup_claim",
    "compact_quality_superiority_claim",
    "live_run_safety_claim",
    "production_live_run_safety_claim",
    "touches_live_runs",
    "touches_production_live_runs",
    "rating_or_promotion_quality_claim",
    "leaderboard_claim",
    "public_leaderboard_claim",
    "public_leaderboard_publish_claim",
    "rating_or_leaderboard_claim",
    "stock_resume_claim",
    "stock_training_resume_claim",
    "production_route_change_claim",
    "automatic_policy_change_allowed",
)
REQUIRED_LIMITATION_ACKS = (
    "manual_review_scale_only",
    "mixed_hardware",
    "compact_local_cpu",
    "tiny_longer_horizon_trace",
    "dirty_source_fingerprints",
    "promotion_requires_separate_policy_decision",
    "manual_review_required_before_any_promotion",
)
REQUIRED_PACKET_REF_KEYS = (
    "readiness_bundle_review",
    "matched_learning_quality_canary",
    "matched_pair_verification",
    "longer_horizon_compact_learning_metrics",
    "stock_reference_capture",
    "compact_candidate_capture",
)


class CompactMatchedQualityManualReviewError(ValueError):
    """Raised when manual-review evidence would overclaim."""


def build_compact_matched_quality_manual_review_v1(
    *,
    run_id: str = DEFAULT_RUN_ID,
    sufficiency_review_path: str | Path,
    manual_review_decision: str = DECISION_KEEP_COMPACT_CANDIDATE_ONLY,
    next_allowed_step: str = DEFAULT_NEXT_ALLOWED_STEP,
    named_non_production_proposal_id: str | None = None,
    reviewer_id: str = "optimizer-main-thread-manual-review",
    reviewer_mode: str = "main_thread_manual_policy_review",
    canonical_stock_failure_run_id: str = DEFAULT_CANONICAL_STOCK_FAILURE_RUN_ID,
    accepted_stock_capture_run_id: str = DEFAULT_ACCEPTED_STOCK_CAPTURE_RUN_ID,
    canonical_stock_failure_acknowledged: bool = True,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a no-promotion manual review decision for OPT-070."""

    sufficiency_path = Path(sufficiency_review_path).resolve()
    sufficiency = _read_json_mapping(sufficiency_path, "sufficiency review")
    validate_compact_matched_quality_sufficiency_review_v1(sufficiency)
    _validate_source_sufficiency(sufficiency)

    packet_refs = _packet_refs_from_sufficiency(sufficiency)
    packet_summary = _packet_summary(sufficiency, packet_refs)
    limitation_acks = _limitation_acknowledgements(sufficiency)
    stock_workaround = _stock_workaround(
        canonical_stock_failure_run_id=canonical_stock_failure_run_id,
        accepted_stock_capture_run_id=accepted_stock_capture_run_id,
        acknowledged=canonical_stock_failure_acknowledged,
    )
    decision = _manual_decision_payload(
        manual_review_decision=manual_review_decision,
        next_allowed_step=next_allowed_step,
        named_non_production_proposal_id=named_non_production_proposal_id,
    )
    non_claims = _false_claims()
    payload: dict[str, Any] = {
        "schema_id": COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_SCHEMA_ID,
        "ok": True,
        "status": STATUS_MANUAL_REVIEW_RECORDED_NO_PROMOTION,
        "opt_id": "OPT-070",
        "run_id": _require_non_empty(run_id, "run_id"),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "candidate_checkpoint_id": _require_non_empty(
            sufficiency.get("candidate_checkpoint_id"),
            "candidate_checkpoint_id",
        ),
        "reviewer": {
            "reviewer_id": _require_non_empty(reviewer_id, "reviewer_id"),
            "reviewer_mode": _require_non_empty(reviewer_mode, "reviewer_mode"),
        },
        "manual_review_scope": MANUAL_REVIEW_SCOPE,
        "reviewed_packet": REVIEWED_PACKET,
        "source_sufficiency_review": _input_ref(sufficiency_path, sufficiency),
        "input_packet_refs": packet_refs,
        "packet_summary": packet_summary,
        "limitation_acknowledgements": limitation_acks,
        "canonical_stock_failure": stock_workaround,
        "manual_decision": decision,
        "attached_claims": {
            "manual_review_recorded": True,
            "source_sufficiency_review_hash_bound": True,
            "larger_32x2048_packet_hash_bound": True,
            "canonical_stock_failure_acknowledged": True,
            "proposal_supported": (
                manual_review_decision
                == DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL
            ),
            "larger_repeat_required": (
                manual_review_decision == DECISION_REQUIRE_LARGER_REPEAT
            ),
            "compact_remains_candidate_only": (
                manual_review_decision == DECISION_KEEP_COMPACT_CANDIDATE_ONLY
            ),
            **non_claims,
        },
        "non_claims": non_claims,
    }
    payload["evidence_ref"] = compact_matched_quality_manual_review_evidence_ref(
        payload
    )
    validate_compact_matched_quality_manual_review_v1(payload)
    return payload


def validate_compact_matched_quality_manual_review_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate an OPT-070 larger matched-quality manual-review decision."""

    if payload.get("schema_id") != COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_SCHEMA_ID:
        raise CompactMatchedQualityManualReviewError(
            "manual-review schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactMatchedQualityManualReviewError("manual-review ok must be true")
    if payload.get("status") != STATUS_MANUAL_REVIEW_RECORDED_NO_PROMOTION:
        raise CompactMatchedQualityManualReviewError(
            "manual-review status mismatch"
        )
    if payload.get("opt_id") != "OPT-070":
        raise CompactMatchedQualityManualReviewError("manual-review opt_id mismatch")
    if payload.get("manual_review_scope") != MANUAL_REVIEW_SCOPE:
        raise CompactMatchedQualityManualReviewError(
            "manual-review scope mismatch"
        )
    if payload.get("reviewed_packet") != REVIEWED_PACKET:
        raise CompactMatchedQualityManualReviewError(
            "manual-review reviewed_packet mismatch"
        )
    source_ref = _required_mapping(
        payload.get("source_sufficiency_review"),
        "source_sufficiency_review",
    )
    source = _validate_input_ref(source_ref, "source_sufficiency_review")
    validate_compact_matched_quality_sufficiency_review_v1(source)
    _validate_source_sufficiency(source)
    if source.get("candidate_checkpoint_id") != payload.get("candidate_checkpoint_id"):
        raise CompactMatchedQualityManualReviewError(
            "manual-review candidate mismatch"
        )
    _validate_packet_refs(
        _required_mapping(payload.get("input_packet_refs"), "input_packet_refs"),
        source,
    )
    _validate_packet_summary(
        _required_mapping(payload.get("packet_summary"), "packet_summary"),
    )
    _validate_limitations(
        _required_mapping(
            payload.get("limitation_acknowledgements"),
            "limitation_acknowledgements",
        )
    )
    _validate_stock_workaround(
        _required_mapping(
            payload.get("canonical_stock_failure"),
            "canonical_stock_failure",
        )
    )
    decision = _required_mapping(payload.get("manual_decision"), "manual_decision")
    _validate_manual_decision(decision)
    _validate_false_claims(payload.get("non_claims"), "non_claims")
    _validate_attached_claims(
        _required_mapping(payload.get("attached_claims"), "attached_claims"),
        str(decision.get("manual_review_decision")),
    )
    expected_ref = compact_matched_quality_manual_review_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactMatchedQualityManualReviewError(
            "manual-review evidence_ref mismatch"
        )


def compact_matched_quality_manual_review_evidence_ref(
    payload: Mapping[str, Any],
) -> str:
    body = {key: value for key, value in payload.items() if key != "evidence_ref"}
    run_id = _require_non_empty(payload.get("run_id"), "run_id")
    digest = _json_sha256(body)
    return (
        f"{COMPACT_MATCHED_QUALITY_MANUAL_REVIEW_EVIDENCE_REF_PREFIX}"
        f"{run_id}:sha256={digest}"
    )


def _validate_source_sufficiency(sufficiency: Mapping[str, Any]) -> None:
    if sufficiency.get("status") != STATUS_CURRENT_CANARY_ACCEPTED_FOR_NEXT_NON_PRODUCTION_STEP:
        raise CompactMatchedQualityManualReviewError(
            "source sufficiency must be accepted for manual review"
        )
    decision = _required_mapping(sufficiency.get("decision"), "source decision")
    if (
        decision.get("matched_quality_sufficiency_decision")
        != DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP
    ):
        raise CompactMatchedQualityManualReviewError(
            "source sufficiency decision mismatch"
        )
    if decision.get("next_non_production_step") != REVIEWED_PACKET:
        raise CompactMatchedQualityManualReviewError(
            "source next_non_production_step mismatch"
        )
    for key in (
        "promotion_claim",
        "automatic_promotion_allowed",
        "current_evidence_sufficient_for_promotion",
    ):
        if decision.get(key) is not False:
            raise CompactMatchedQualityManualReviewError(
                f"source sufficiency {key} must be false"
            )
    summary = _required_mapping(
        sufficiency.get("current_evidence_summary"),
        "current_evidence_summary",
    )
    if int(summary.get("eval_seed_count", 0)) != 32:
        raise CompactMatchedQualityManualReviewError(
            "source eval_seed_count must be 32"
        )
    if int(summary.get("eval_max_steps", 0)) != 2048:
        raise CompactMatchedQualityManualReviewError(
            "source eval_max_steps must be 2048"
        )
    if (
        summary.get("denominator_id")
        != "matched_quality_larger_2048x32-env64train8_denominator_v1"
    ):
        raise CompactMatchedQualityManualReviewError(
            "source denominator_id mismatch"
        )


def _packet_refs_from_sufficiency(
    sufficiency: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    input_reports = _required_mapping(sufficiency.get("input_reports"), "input_reports")
    refs: dict[str, dict[str, str]] = {}
    for key in REQUIRED_PACKET_REF_KEYS:
        ref = _required_mapping(input_reports.get(key), f"input_reports.{key}")
        path = Path(_require_non_empty(ref.get("path"), f"{key} path"))
        input_payload = _read_json_mapping(path, key)
        refs[key] = {
            "path": str(path),
            "sha256": _require_non_empty(ref.get("sha256"), f"{key} sha256"),
            "schema_id": _require_non_empty(
                ref.get("schema_id") or input_payload.get("schema_id"),
                f"{key} schema_id",
            ),
        }
        if str(ref.get("evidence_ref") or "").strip():
            refs[key]["evidence_ref"] = _require_non_empty(
                ref.get("evidence_ref"),
                f"{key} evidence_ref",
            )
    _validate_packet_refs(refs, sufficiency)
    return refs


def _validate_packet_refs(
    refs: Mapping[str, Any],
    sufficiency: Mapping[str, Any],
) -> None:
    source_refs = _required_mapping(sufficiency.get("input_reports"), "input_reports")
    for key in REQUIRED_PACKET_REF_KEYS:
        ref = _required_mapping(refs.get(key), f"input_packet_refs.{key}")
        source_ref = _required_mapping(source_refs.get(key), f"source {key}")
        if ref.get("path") != source_ref.get("path"):
            raise CompactMatchedQualityManualReviewError(f"{key} path mismatch")
        _validate_input_ref(ref, key)
    stock_ref = _required_mapping(refs.get("stock_reference_capture"), "stock ref")
    if "evalenv32" not in str(stock_ref.get("path")):
        raise CompactMatchedQualityManualReviewError(
            "stock reference capture must be the evalenv32 rerun"
        )
    compact_ref = _required_mapping(
        refs.get("compact_candidate_capture"),
        "compact ref",
    )
    if "larger-2048x32-env64train8" not in str(compact_ref.get("path")):
        raise CompactMatchedQualityManualReviewError(
            "compact capture must be the larger 2048x32 env64train8 capture"
        )


def _packet_summary(
    sufficiency: Mapping[str, Any],
    packet_refs: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _required_mapping(
        sufficiency.get("current_evidence_summary"),
        "current_evidence_summary",
    )
    bundle_summary = _required_mapping(
        summary.get("bundle_matched_pair_summary"),
        "bundle_matched_pair_summary",
    )
    stock_denominators = _required_mapping(
        summary.get("stock_denominators"),
        "stock_denominators",
    )
    compact_denominators = _required_mapping(
        summary.get("compact_denominators"),
        "compact_denominators",
    )
    compact_sample_rows = _int(
        compact_denominators.get("compact_sample_rows"),
        "compact_sample_rows",
    )
    sample_batches = _int(
        compact_denominators.get("sample_batch_count_delta"),
        "sample_batch_count_delta",
    )
    learner_updates = _int(
        compact_denominators.get("learner_update_count_delta"),
        "learner_update_count_delta",
    )
    return {
        "quality_currency": _require_non_empty(
            summary.get("quality_currency"),
            "quality_currency",
        ),
        "denominator_id": _require_non_empty(
            summary.get("denominator_id"),
            "denominator_id",
        ),
        "quality_horizon": _require_non_empty(
            summary.get("quality_horizon"),
            "quality_horizon",
        ),
        "eval_seed_count": _int(summary.get("eval_seed_count"), "eval_seed_count"),
        "eval_max_steps": _int(summary.get("eval_max_steps"), "eval_max_steps"),
        "stock_reference_delta": _float(
            summary.get("stock_reference_delta"),
            "stock_reference_delta",
        ),
        "compact_candidate_delta": _float(
            summary.get("compact_candidate_delta"),
            "compact_candidate_delta",
        ),
        "compact_minus_stock_delta": _float(
            summary.get("compact_minus_stock_delta"),
            "compact_minus_stock_delta",
        ),
        "hardware_class": _require_non_empty(
            summary.get("hardware_class"),
            "hardware_class",
        ),
        "stock_hardware_class": _require_non_empty(
            summary.get("stock_hardware_class"),
            "stock_hardware_class",
        ),
        "compact_hardware_class": _require_non_empty(
            summary.get("compact_hardware_class"),
            "compact_hardware_class",
        ),
        "stock_calls_train_muzero": bundle_summary.get("stock_calls_train_muzero"),
        "compact_calls_train_muzero": bundle_summary.get("compact_calls_train_muzero"),
        "stock_wall_sec_used_for_speed_claim": stock_denominators.get(
            "wall_sec_used_for_speed_claim"
        ),
        "compact_wall_sec_used_for_speed_claim": compact_denominators.get(
            "wall_sec_used_for_speed_claim"
        ),
        "uses_fallback_denominator": (
            stock_denominators.get("uses_fallback_denominator") is True
            or compact_denominators.get("uses_fallback_denominator") is True
        ),
        "compact_denominator_contract": {
            "compact_rollout_rows": _int(
                compact_denominators.get("compact_rollout_rows"),
                "compact_rollout_rows",
            ),
            "compact_sample_rows": compact_sample_rows,
            "sample_batch_count_delta": sample_batches,
            "learner_update_count_delta": learner_updates,
            "sample_rows_cover_sample_batches": compact_sample_rows >= sample_batches,
            "learner_updates_cover_sample_batches": learner_updates >= sample_batches,
            "learner_updates_per_sample_batch": (
                learner_updates / sample_batches if sample_batches else 0.0
            ),
        },
        "stock_reference_capture_path": packet_refs["stock_reference_capture"]["path"],
        "compact_candidate_capture_path": packet_refs["compact_candidate_capture"][
            "path"
        ],
        "accepted_stock_capture_is_evalenv32": True,
    }


def _validate_packet_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("eval_seed_count") != 32:
        raise CompactMatchedQualityManualReviewError(
            "packet summary eval_seed_count must be 32"
        )
    if summary.get("eval_max_steps") != 2048:
        raise CompactMatchedQualityManualReviewError(
            "packet summary eval_max_steps must be 2048"
        )
    if (
        summary.get("denominator_id")
        != "matched_quality_larger_2048x32-env64train8_denominator_v1"
    ):
        raise CompactMatchedQualityManualReviewError(
            "packet summary denominator mismatch"
        )
    if summary.get("stock_calls_train_muzero") is not True:
        raise CompactMatchedQualityManualReviewError(
            "stock arm must call train_muzero"
        )
    if summary.get("compact_calls_train_muzero") is not False:
        raise CompactMatchedQualityManualReviewError(
            "compact arm must not call train_muzero"
        )
    if summary.get("uses_fallback_denominator") is not False:
        raise CompactMatchedQualityManualReviewError(
            "manual review rejects fallback denominators"
        )
    if summary.get("stock_wall_sec_used_for_speed_claim") is not False:
        raise CompactMatchedQualityManualReviewError(
            "stock wall time must not be used for speed claim"
        )
    if summary.get("compact_wall_sec_used_for_speed_claim") is not False:
        raise CompactMatchedQualityManualReviewError(
            "compact wall time must not be used for speed claim"
        )
    contract = _required_mapping(
        summary.get("compact_denominator_contract"),
        "compact_denominator_contract",
    )
    for key in (
        "sample_rows_cover_sample_batches",
        "learner_updates_cover_sample_batches",
    ):
        if contract.get(key) is not True:
            raise CompactMatchedQualityManualReviewError(
                f"compact denominator contract failed: {key}"
            )
    if abs(float(contract.get("learner_updates_per_sample_batch", 0.0)) - 7.875) > 1e-9:
        raise CompactMatchedQualityManualReviewError(
            "compact learner update ratio mismatch"
        )
    if "evalenv32" not in str(summary.get("stock_reference_capture_path")):
        raise CompactMatchedQualityManualReviewError(
            "packet summary must use evalenv32 stock capture"
        )


def _limitation_acknowledgements(sufficiency: Mapping[str, Any]) -> dict[str, bool]:
    decision = _required_mapping(sufficiency.get("decision"), "source decision")
    source_acks = _required_mapping(
        decision.get("limitation_acknowledgements"),
        "source limitation acknowledgements",
    )
    summary = _required_mapping(
        sufficiency.get("current_evidence_summary"),
        "current_evidence_summary",
    )
    known = _required_mapping(summary.get("known_limitations"), "known_limitations")
    return {
        "manual_review_scale_only": source_acks.get("current_canary_scale") is True,
        "mixed_hardware": source_acks.get("mixed_hardware") is True,
        "compact_local_cpu": source_acks.get("compact_local_cpu") is True,
        "tiny_longer_horizon_trace": (
            source_acks.get("tiny_longer_horizon_trace") is True
        ),
        "dirty_source_fingerprints": True,
        "promotion_requires_separate_policy_decision": (
            source_acks.get("promotion_requires_separate_policy_decision") is True
        ),
        "manual_review_required_before_any_promotion": (
            source_acks.get("manual_review_required_before_any_promotion") is True
        ),
        "compact_quality_superiority_claim_false": (
            known.get("compact_quality_superiority_claim") is False
        ),
        "promotion_quality_claim_false": known.get("promotion_quality_claim") is False,
    }


def _validate_limitations(limitations: Mapping[str, Any]) -> None:
    for key in (
        *REQUIRED_LIMITATION_ACKS,
        "compact_quality_superiority_claim_false",
        "promotion_quality_claim_false",
    ):
        if limitations.get(key) is not True:
            raise CompactMatchedQualityManualReviewError(
                f"manual-review limitation acknowledgement missing: {key}"
            )


def _stock_workaround(
    *,
    canonical_stock_failure_run_id: str,
    accepted_stock_capture_run_id: str,
    acknowledged: bool,
) -> dict[str, Any]:
    return {
        "canonical_stock_failure_run_id": _require_non_empty(
            canonical_stock_failure_run_id,
            "canonical_stock_failure_run_id",
        ),
        "canonical_stock_failed_closed_before_capture": True,
        "failure_mode": "lightzero_evaluator_empty_ready_set_index_error",
        "accepted_stock_capture_run_id": _require_non_empty(
            accepted_stock_capture_run_id,
            "accepted_stock_capture_run_id",
        ),
        "accepted_stock_capture_uses_evalenv32_workaround": True,
        "evalenv32_is_not_canonical_evaluator_health_proof": True,
        "acknowledged": bool(acknowledged),
    }


def _validate_stock_workaround(workaround: Mapping[str, Any]) -> None:
    _require_non_empty(
        workaround.get("canonical_stock_failure_run_id"),
        "canonical_stock_failure_run_id",
    )
    accepted = _require_non_empty(
        workaround.get("accepted_stock_capture_run_id"),
        "accepted_stock_capture_run_id",
    )
    if "evalenv32" not in accepted:
        raise CompactMatchedQualityManualReviewError(
            "accepted stock capture run id must name evalenv32"
        )
    for key in (
        "canonical_stock_failed_closed_before_capture",
        "accepted_stock_capture_uses_evalenv32_workaround",
        "evalenv32_is_not_canonical_evaluator_health_proof",
        "acknowledged",
    ):
        if workaround.get(key) is not True:
            raise CompactMatchedQualityManualReviewError(
                f"stock workaround acknowledgement missing: {key}"
            )


def _manual_decision_payload(
    *,
    manual_review_decision: str,
    next_allowed_step: str,
    named_non_production_proposal_id: str | None,
) -> dict[str, Any]:
    if manual_review_decision not in ALLOWED_MANUAL_DECISIONS:
        raise CompactMatchedQualityManualReviewError(
            "manual review decision is not allowed"
        )
    if (
        manual_review_decision == DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL
        and not str(named_non_production_proposal_id or "").strip()
    ):
        raise CompactMatchedQualityManualReviewError(
            "named non-production proposal id is required"
        )
    if (
        manual_review_decision != DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL
        and named_non_production_proposal_id is not None
    ):
        raise CompactMatchedQualityManualReviewError(
            "named proposal id is only allowed for proposal-support decision"
        )
    rationale = {
        DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL: [
            "larger packet can support a named non-production proposal artifact",
            "the proposal remains isolated from production promotion surfaces",
            "production promotion remains a separate policy and code decision",
        ],
        DECISION_REQUIRE_LARGER_REPEAT: [
            "larger packet is not enough for a proposal",
            "repeat evidence is required before any proposal artifact",
            "production promotion remains a separate policy and code decision",
        ],
        DECISION_KEEP_COMPACT_CANDIDATE_ONLY: [
            "larger packet is positive but manual-review-only evidence",
            "compact-minus-stock movement is small",
            "mixed hardware and compact-local-CPU limitations remain",
            "production promotion remains a separate policy and code decision",
        ],
    }[manual_review_decision]
    return {
        "manual_review_decision": manual_review_decision,
        "named_non_production_proposal_id": named_non_production_proposal_id,
        "decision_rationale": rationale,
        "next_allowed_step": _require_non_empty(
            next_allowed_step,
            "next_allowed_step",
        ),
        "promotion_claim": False,
        "automatic_promotion_allowed": False,
        "production_live_run_allowed": False,
        "rating_or_leaderboard_publication_allowed": False,
        "stock_train_muzero_speedup_claim": False,
    }


def _validate_manual_decision(decision: Mapping[str, Any]) -> None:
    value = str(decision.get("manual_review_decision") or "")
    if value not in ALLOWED_MANUAL_DECISIONS:
        raise CompactMatchedQualityManualReviewError(
            "manual review decision is not allowed"
        )
    _require_non_empty(decision.get("next_allowed_step"), "next_allowed_step")
    proposal = decision.get("named_non_production_proposal_id")
    if value == DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL:
        _require_non_empty(proposal, "named_non_production_proposal_id")
    elif proposal is not None:
        raise CompactMatchedQualityManualReviewError(
            "named proposal id is only allowed for proposal-support decision"
        )
    rationale = decision.get("decision_rationale")
    if not isinstance(rationale, Sequence) or isinstance(rationale, (str, bytes)):
        raise CompactMatchedQualityManualReviewError(
            "manual-review rationale must be a sequence"
        )
    if not rationale:
        raise CompactMatchedQualityManualReviewError(
            "manual-review rationale must be non-empty"
        )
    for key in (
        "promotion_claim",
        "automatic_promotion_allowed",
        "production_live_run_allowed",
        "rating_or_leaderboard_publication_allowed",
        "stock_train_muzero_speedup_claim",
    ):
        if decision.get(key) is not False:
            raise CompactMatchedQualityManualReviewError(
                f"manual-review decision {key} must be false"
            )


def _validate_attached_claims(claims: Mapping[str, Any], decision: str) -> None:
    for key in (
        "manual_review_recorded",
        "source_sufficiency_review_hash_bound",
        "larger_32x2048_packet_hash_bound",
        "canonical_stock_failure_acknowledged",
    ):
        if claims.get(key) is not True:
            raise CompactMatchedQualityManualReviewError(
                f"manual-review attached claim missing: {key}"
            )
    expected = {
        "proposal_supported": decision == DECISION_SUPPORT_NAMED_NON_PRODUCTION_PROPOSAL,
        "larger_repeat_required": decision == DECISION_REQUIRE_LARGER_REPEAT,
        "compact_remains_candidate_only": decision == DECISION_KEEP_COMPACT_CANDIDATE_ONLY,
    }
    for key, value in expected.items():
        if claims.get(key) is not value:
            raise CompactMatchedQualityManualReviewError(
                f"manual-review attached claim mismatch: {key}"
            )
    _validate_false_claims(claims, "attached_claims")


def _input_ref(path: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    ref = {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "schema_id": _require_non_empty(payload.get("schema_id"), "schema_id"),
    }
    if str(payload.get("evidence_ref") or "").strip():
        ref["evidence_ref"] = _require_non_empty(
            payload.get("evidence_ref"),
            "evidence_ref",
        )
    return ref


def _validate_input_ref(ref: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(_require_non_empty(ref.get("path"), f"{label} path"))
    if not path.is_file():
        raise CompactMatchedQualityManualReviewError(f"{label} path missing")
    if _file_sha256(path) != ref.get("sha256"):
        raise CompactMatchedQualityManualReviewError(f"{label} sha256 mismatch")
    payload = _read_json_mapping(path, label)
    if payload.get("schema_id") != ref.get("schema_id"):
        raise CompactMatchedQualityManualReviewError(f"{label} schema mismatch")
    if "evidence_ref" in ref and payload.get("evidence_ref") != ref.get("evidence_ref"):
        raise CompactMatchedQualityManualReviewError(
            f"{label} evidence_ref mismatch"
        )
    return payload


def _validate_false_claims(value: Any, label: str) -> None:
    claims = _required_mapping(value, label)
    for key in FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactMatchedQualityManualReviewError(
                f"{label} {key} must be false"
            )


def _false_claims() -> dict[str, bool]:
    return {key: False for key in FALSE_CLAIM_KEYS}


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompactMatchedQualityManualReviewError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompactMatchedQualityManualReviewError(
            f"{label} is not readable JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CompactMatchedQualityManualReviewError(f"{label} must be a mapping")
    return dict(payload)


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactMatchedQualityManualReviewError(f"{label} must be a mapping")
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactMatchedQualityManualReviewError(
            f"{label} must be non-empty"
        )
    return text


def _int(value: Any, label: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise CompactMatchedQualityManualReviewError(
            f"{label} must be an integer"
        ) from exc


def _float(value: Any, label: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise CompactMatchedQualityManualReviewError(f"{label} must be numeric") from exc


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
    return value
