"""Fail-closed sufficiency review for compact matched-quality evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from curvyzero.training.compact_promotion_readiness_bundle_review import (
    COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_bundle_review import (
    validate_compact_promotion_readiness_bundle_review_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_CANDIDATE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    STOCK_REFERENCE_ROLE,
)


COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_SCHEMA_ID = (
    "curvyzero_compact_matched_quality_sufficiency_review/v1"
)
COMPACT_MATCHED_LEARNING_QUALITY_STUDY_PLAN_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_study_plan/v1"
)
COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_EVIDENCE_REF_PREFIX = (
    "compact_matched_quality_sufficiency_review:"
)

DECISION_SCOPE_NEXT_NON_PRODUCTION_ONLY = (
    "matched_quality_sufficiency_for_next_non_production_step_only"
)
DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP = (
    "accept_current_for_next_non_production_step"
)
DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY = (
    "require_larger_same_surface_study"
)
STATUS_CURRENT_CANARY_ACCEPTED_FOR_NEXT_NON_PRODUCTION_STEP = (
    "current_canary_accepted_for_next_non_production_step"
)
STATUS_LARGER_SAME_SURFACE_STUDY_REQUIRED = (
    "larger_same_surface_study_required"
)

DEFAULT_MIN_EVAL_SEED_COUNT = 32
DEFAULT_MIN_EVAL_MAX_STEPS = 2048
DEFAULT_STOCK_REFERENCE_MIN_MAX_ENV_STEP = 2048
DEFAULT_STOCK_REFERENCE_MIN_MAX_TRAIN_ITER = 4
DEFAULT_COMPACT_CANDIDATE_MIN_ENV_STEPS = 64
DEFAULT_COMPACT_CANDIDATE_MIN_TRAIN_STEPS = 8
DEFAULT_EVAL_SEED_RNG_SEED = 20260833
DEFAULT_STUDY_RUN_STAMP = "20260531"
DEFAULT_STUDY_SUFFIX = "2048x32"
DEFAULT_COMPACT_STUDY_SUFFIX = "2048x32-env64train8"
STOCK_EVALUATOR_REQUIREMENT_POLICY = "evalenv_full_episode_surface"
DEFAULT_STUDY_ID = (
    "optimizer-compact-matched-learning-quality-larger-"
    f"{DEFAULT_COMPACT_STUDY_SUFFIX}-{DEFAULT_STUDY_RUN_STAMP}"
)
DEFAULT_LARGER_DENOMINATOR_ID = (
    f"matched_quality_larger_{DEFAULT_COMPACT_STUDY_SUFFIX}_denominator_v1"
)
DEFAULT_LARGER_QUALITY_HORIZON = (
    f"matched_learning_quality_pre_post_eval_larger_{DEFAULT_COMPACT_STUDY_SUFFIX}"
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
    "matched_quality_sufficiency_reviewed",
    "readiness_bundle_hash_bound",
    "current_canary_hash_bound",
)
INPUT_REPORT_KEYS = (
    "readiness_bundle_review",
    "matched_learning_quality_canary",
    "matched_pair_verification",
    "longer_horizon_compact_learning_metrics",
    "stock_reference_capture",
    "compact_candidate_capture",
)
REQUIRED_LIMITATION_ACKNOWLEDGEMENTS = (
    "current_canary_scale",
    "mixed_hardware",
    "compact_local_cpu",
    "tiny_longer_horizon_trace",
    "promotion_requires_separate_policy_decision",
    "manual_review_required_before_any_promotion",
)
REQUIRED_STUDY_RUN_KEYS = (
    "stock_reference_capture_producer",
    "compact_candidate_capture_producer",
    "matched_canary_builder",
    "matched_pair_verifier",
    "readiness_bundle_refresh",
    "sufficiency_review_update",
)
REQUIRED_STUDY_OUTPUT_KEYS = (
    "stock_reference_capture",
    "compact_candidate_capture",
    "matched_learning_quality_canary_report",
    "matched_pair_verification_report",
    "refreshed_readiness_bundle_review",
    "updated_sufficiency_review",
)
REQUIRED_DISALLOWED_SHORTCUT_KEYS = (
    "same_eval_surface_required",
    "fresh_outputs_required",
    "no_fallback_denominators",
    "no_preview_captures",
    "no_profile_only_speed_currency",
    "no_live_runs",
    "compact_calls_train_muzero_false",
    "final_bundle_review_refresh_required",
)


class CompactMatchedQualitySufficiencyReviewError(ValueError):
    """Raised when matched-quality sufficiency review evidence would overclaim."""


def build_compact_matched_quality_sufficiency_review_v1(
    *,
    run_id: str,
    readiness_bundle_review_path: str | Path,
    sufficiency_decision: str = DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY,
    reviewer_id: str = "optimizer-main-thread-policy-review",
    reviewer_mode: str = "main_thread_policy_review",
    next_non_production_step: str | None = None,
    larger_study_id: str = DEFAULT_STUDY_ID,
    min_eval_seed_count: int = DEFAULT_MIN_EVAL_SEED_COUNT,
    min_eval_max_steps: int = DEFAULT_MIN_EVAL_MAX_STEPS,
    stock_reference_min_max_env_step: int = DEFAULT_STOCK_REFERENCE_MIN_MAX_ENV_STEP,
    stock_reference_min_max_train_iter: int = DEFAULT_STOCK_REFERENCE_MIN_MAX_TRAIN_ITER,
    compact_candidate_min_env_steps: int = DEFAULT_COMPACT_CANDIDATE_MIN_ENV_STEPS,
    compact_candidate_min_train_steps: int = DEFAULT_COMPACT_CANDIDATE_MIN_TRAIN_STEPS,
    eval_seed_rng_seed: int = DEFAULT_EVAL_SEED_RNG_SEED,
    larger_denominator_id: str = DEFAULT_LARGER_DENOMINATOR_ID,
    larger_quality_horizon: str = DEFAULT_LARGER_QUALITY_HORIZON,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a hash-bound review that chooses the next matched-quality step."""

    bundle_path = Path(readiness_bundle_review_path).resolve()
    bundle = _read_json_mapping(bundle_path, "readiness bundle review")
    validate_compact_promotion_readiness_bundle_review_v1(bundle)

    matched, pair, longer = _load_bound_bundle_inputs(bundle)
    candidate = _require_non_empty(
        bundle.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    current_summary = _current_evidence_summary(
        bundle=bundle,
        matched=matched["payload"],
        longer=longer["payload"],
    )
    decision = _decision_payload(
        sufficiency_decision=sufficiency_decision,
        current_summary=current_summary,
        next_non_production_step=next_non_production_step,
    )
    larger_plan = None
    if sufficiency_decision == DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY:
        larger_plan = _larger_same_surface_study_plan(
            study_id=larger_study_id,
            current_summary=current_summary,
            matched=matched["payload"],
            bundle=bundle,
            min_eval_seed_count=min_eval_seed_count,
            min_eval_max_steps=min_eval_max_steps,
            stock_reference_min_max_env_step=stock_reference_min_max_env_step,
            stock_reference_min_max_train_iter=stock_reference_min_max_train_iter,
            compact_candidate_min_env_steps=compact_candidate_min_env_steps,
            compact_candidate_min_train_steps=compact_candidate_min_train_steps,
            eval_seed_rng_seed=eval_seed_rng_seed,
            larger_denominator_id=larger_denominator_id,
            larger_quality_horizon=larger_quality_horizon,
        )

    non_claims = _false_claims()
    input_reports = _input_report_refs(
        bundle_path=bundle_path,
        bundle=bundle,
        matched=matched,
        pair=pair,
        longer=longer,
    )
    payload = {
        "schema_id": COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_SCHEMA_ID,
        "ok": True,
        "status": _status_for_decision(sufficiency_decision),
        "run_id": str(run_id),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "candidate_checkpoint_id": candidate,
        "decision_scope": DECISION_SCOPE_NEXT_NON_PRODUCTION_ONLY,
        "reviewer": {
            "reviewer_id": _require_non_empty(reviewer_id, "reviewer_id"),
            "review_mode": _require_non_empty(reviewer_mode, "review_mode"),
        },
        "input_reports": input_reports,
        "current_evidence_summary": current_summary,
        "decision": decision,
        "larger_study_plan": larger_plan,
        "attached_claims": {
            **{key: True for key in TRUE_ATTACHED_CLAIM_KEYS},
            "larger_same_surface_study_required": (
                sufficiency_decision
                == DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY
            ),
            "current_canary_accepted_for_next_non_production_step": (
                sufficiency_decision
                == DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP
            ),
            **non_claims,
        },
        "non_claims": non_claims,
    }
    payload["evidence_ref"] = compact_matched_quality_sufficiency_review_evidence_ref(
        payload
    )
    validate_compact_matched_quality_sufficiency_review_v1(payload)
    return payload


def validate_compact_matched_quality_sufficiency_review_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate a matched-quality sufficiency decision and its bound evidence."""

    if payload.get("schema_id") != COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_SCHEMA_ID:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency review schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency review must be ok=true"
        )
    if payload.get("decision_scope") != DECISION_SCOPE_NEXT_NON_PRODUCTION_ONLY:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency review scope mismatch"
        )

    bundle_path, bundle = _load_bound_source_bundle(payload)
    validate_compact_promotion_readiness_bundle_review_v1(bundle)
    matched, pair, longer = _load_bound_bundle_inputs(bundle)
    _validate_input_report_refs(
        payload=payload,
        bundle_path=bundle_path,
        bundle=bundle,
        matched=matched,
        pair=pair,
        longer=longer,
    )

    candidate = _require_non_empty(
        bundle.get("candidate_checkpoint_id"),
        "source bundle candidate_checkpoint_id",
    )
    if payload.get("candidate_checkpoint_id") != candidate:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency candidate mismatch"
        )
    current_summary = _current_evidence_summary(
        bundle=bundle,
        matched=matched["payload"],
        longer=longer["payload"],
    )
    if _json_sha256(payload.get("current_evidence_summary")) != _json_sha256(
        current_summary
    ):
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality current evidence summary drift"
        )

    decision_value = _validate_decision(payload)
    if payload.get("status") != _status_for_decision(decision_value):
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency status mismatch"
        )
    if decision_value == DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY:
        _validate_larger_study_plan(
            payload.get("larger_study_plan"),
            current_summary=current_summary,
            matched=matched["payload"],
        )
    elif payload.get("larger_study_plan") is not None:
        raise CompactMatchedQualitySufficiencyReviewError(
            "accepted current canary review must not attach a larger study plan"
        )

    _validate_claims(payload, decision_value=decision_value)
    expected_ref = compact_matched_quality_sufficiency_review_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency evidence_ref mismatch"
        )


def compact_matched_quality_sufficiency_review_evidence_ref(
    payload: Mapping[str, Any],
) -> str:
    """Return a stable evidence ref for a matched-quality sufficiency review."""

    candidate = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    input_reports = _required_mapping(payload.get("input_reports"), "input_reports")
    bundle = _required_mapping(
        input_reports.get("readiness_bundle_review"),
        "readiness_bundle_review",
    )
    decision = _required_mapping(payload.get("decision"), "decision")
    digest_source = {
        "readiness_bundle_review_sha256": bundle.get("sha256"),
        "decision": decision.get("matched_quality_sufficiency_decision"),
        "status": payload.get("status"),
        "larger_study_plan": payload.get("larger_study_plan"),
    }
    return (
        f"{COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_EVIDENCE_REF_PREFIX}"
        f"{candidate}:{_json_sha256(digest_source)[:16]}"
    )


def _load_bound_source_bundle(
    payload: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any]]:
    input_reports = _required_mapping(payload.get("input_reports"), "input_reports")
    bundle_ref = _required_mapping(
        input_reports.get("readiness_bundle_review"),
        "readiness_bundle_review",
    )
    bundle_path = Path(
        _require_non_empty(bundle_ref.get("path"), "readiness bundle review path")
    ).resolve()
    bundle = _read_json_mapping(bundle_path, "readiness bundle review")
    if _file_sha256(bundle_path) != bundle_ref.get("sha256"):
        raise CompactMatchedQualitySufficiencyReviewError(
            "readiness bundle review sha256 mismatch"
        )
    if bundle_ref.get("schema_id") != COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID:
        raise CompactMatchedQualitySufficiencyReviewError(
            "readiness bundle review schema mismatch"
        )
    return bundle_path, bundle


def _load_bound_bundle_inputs(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    refs = _required_mapping(bundle.get("input_reports"), "bundle input_reports")
    matched = _load_ref_payload(
        refs,
        "matched_learning_quality_canary",
    )
    pair = _load_ref_payload(refs, "matched_pair_verification")
    longer = _load_ref_payload(refs, "longer_horizon_compact_learning_metrics")
    _validate_canary_capture_refs(matched["payload"])
    return matched, pair, longer


def _load_ref_payload(
    refs: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    ref = _required_mapping(refs.get(key), key)
    path = Path(_require_non_empty(ref.get("path"), f"{key} path")).resolve()
    payload = _read_json_mapping(path, key)
    sha256 = _file_sha256(path)
    if sha256 != ref.get("sha256"):
        raise CompactMatchedQualitySufficiencyReviewError(
            f"{key} sha256 mismatch"
        )
    if payload.get("schema_id") != ref.get("schema_id"):
        raise CompactMatchedQualitySufficiencyReviewError(f"{key} schema drift")
    if payload.get("ok") is not True:
        raise CompactMatchedQualitySufficiencyReviewError(f"{key} must be ok=true")
    return {
        "path": path,
        "sha256": sha256,
        "schema_id": str(payload.get("schema_id", "")),
        "status": str(payload.get("status", "")),
        "candidate_checkpoint_id": str(payload.get("candidate_checkpoint_id", "")),
        "evidence_ref": str(payload.get("evidence_ref", "")),
        "payload": payload,
    }


def _validate_canary_capture_refs(matched: Mapping[str, Any]) -> None:
    files = _required_mapping(
        matched.get("input_capture_files"),
        "matched quality input_capture_files",
    )
    for key in ("stock_reference_capture", "compact_candidate_capture"):
        ref = _required_mapping(files.get(key), key)
        path = Path(_require_non_empty(ref.get("path"), f"{key} path")).resolve()
        if not path.is_file():
            raise CompactMatchedQualitySufficiencyReviewError(f"{key} missing")
        if _file_sha256(path) != ref.get("sha256"):
            raise CompactMatchedQualitySufficiencyReviewError(
                f"{key} sha256 mismatch"
            )


def _input_report_refs(
    *,
    bundle_path: Path,
    bundle: Mapping[str, Any],
    matched: Mapping[str, Any],
    pair: Mapping[str, Any],
    longer: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    captures = _required_mapping(
        _required_mapping(matched["payload"].get("input_capture_files"), "captures"),
        "captures",
    )
    stock_capture = _required_mapping(
        captures.get("stock_reference_capture"),
        "stock_reference_capture",
    )
    compact_capture = _required_mapping(
        captures.get("compact_candidate_capture"),
        "compact_candidate_capture",
    )
    refs = {
        "readiness_bundle_review": {
            "path": str(bundle_path),
            "sha256": _file_sha256(bundle_path),
            "schema_id": str(bundle.get("schema_id", "")),
            "status": str(bundle.get("status", "")),
            "candidate_checkpoint_id": str(bundle.get("candidate_checkpoint_id", "")),
            "evidence_ref": str(bundle.get("evidence_ref", "")),
        },
        "matched_learning_quality_canary": _loaded_report_ref(matched),
        "matched_pair_verification": _loaded_report_ref(pair),
        "longer_horizon_compact_learning_metrics": _loaded_report_ref(longer),
        "stock_reference_capture": _capture_ref(stock_capture),
        "compact_candidate_capture": _capture_ref(compact_capture),
    }
    return refs


def _validate_input_report_refs(
    *,
    payload: Mapping[str, Any],
    bundle_path: Path,
    bundle: Mapping[str, Any],
    matched: Mapping[str, Any],
    pair: Mapping[str, Any],
    longer: Mapping[str, Any],
) -> None:
    expected = _input_report_refs(
        bundle_path=bundle_path,
        bundle=bundle,
        matched=matched,
        pair=pair,
        longer=longer,
    )
    refs = _required_mapping(payload.get("input_reports"), "input_reports")
    for key in INPUT_REPORT_KEYS:
        if _json_sha256(refs.get(key)) != _json_sha256(expected[key]):
            raise CompactMatchedQualitySufficiencyReviewError(
                f"{key} input report ref drift"
            )


def _loaded_report_ref(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(report["path"]),
        "sha256": str(report["sha256"]),
        "schema_id": str(report["schema_id"]),
        "status": str(report["status"]),
        "candidate_checkpoint_id": str(report["candidate_checkpoint_id"]),
        "evidence_ref": str(report["evidence_ref"]),
    }


def _capture_ref(ref: Mapping[str, Any]) -> dict[str, str]:
    path = Path(_require_non_empty(ref.get("path"), "capture path")).resolve()
    sha256 = str(ref.get("sha256", ""))
    if _file_sha256(path) != sha256:
        raise CompactMatchedQualitySufficiencyReviewError("capture sha256 mismatch")
    return {
        "path": str(path),
        "sha256": sha256,
    }


def _current_evidence_summary(
    *,
    bundle: Mapping[str, Any],
    matched: Mapping[str, Any],
    longer: Mapping[str, Any],
) -> dict[str, Any]:
    quality = _required_mapping(
        bundle.get("quality_strength_review"),
        "bundle quality_strength_review",
    )
    matched_pair = _required_mapping(matched.get("matched_pair"), "matched_pair")
    movement = _required_mapping(matched.get("quality_movement"), "quality_movement")
    arms = _required_mapping(matched.get("arms"), "matched arms")
    stock = _required_mapping(arms.get(STOCK_REFERENCE_ROLE), STOCK_REFERENCE_ROLE)
    compact = _required_mapping(arms.get(COMPACT_CANDIDATE_ROLE), COMPACT_CANDIDATE_ROLE)
    compact_eval = _required_mapping(compact.get("eval_settings"), "compact eval_settings")
    stock_eval = _required_mapping(stock.get("eval_settings"), "stock eval_settings")
    eval_seed_set = compact_eval.get("eval_seed_set")
    eval_seed_count = (
        len(eval_seed_set)
        if isinstance(eval_seed_set, Sequence) and not isinstance(eval_seed_set, str)
        else 0
    )
    longer_series = longer.get("checkpoint_series")
    denominators = _required_mapping(
        longer.get("cumulative_denominators"),
        "longer cumulative_denominators",
    )
    bundle_pair = _required_mapping(
        quality.get("matched_pair_summary"),
        "bundle matched_pair_summary",
    )
    bundle_longer = _required_mapping(
        quality.get("longer_horizon_summary"),
        "bundle longer_horizon_summary",
    )
    return {
        "denominator_id": str(matched_pair.get("denominator_id", "")),
        "quality_horizon": str(matched_pair.get("quality_horizon", "")),
        "hardware_class": str(matched_pair.get("hardware_class", "")),
        "stock_hardware_class": str(matched_pair.get("stock_hardware_class", "")),
        "compact_hardware_class": str(matched_pair.get("compact_hardware_class", "")),
        "eval_seed_count": int(eval_seed_count),
        "eval_max_steps": int(compact_eval.get("eval_max_steps", 0) or 0),
        "eval_seed_rng_seed": int(compact_eval.get("eval_seed_rng_seed", 0) or 0),
        "stock_reference_delta": float(movement.get("stock_reference_delta", 0.0)),
        "compact_candidate_delta": float(movement.get("compact_candidate_delta", 0.0)),
        "compact_minus_stock_delta": float(
            movement.get("compact_minus_stock_delta", 0.0)
        ),
        "quality_currency": str(movement.get("quality_currency", "")),
        "stock_denominators": _plain_mapping(
            _required_mapping(stock.get("denominators"), "stock denominators")
        ),
        "compact_denominators": _plain_mapping(
            _required_mapping(compact.get("denominators"), "compact denominators")
        ),
        "longer_horizon_summary": {
            "checkpoint_count": (
                len(longer_series) if isinstance(longer_series, list) else 0
            ),
            "learner_update_count_delta": int(
                denominators.get("learner_update_count_delta", 0) or 0
            ),
            "sample_batch_count_delta": int(
                denominators.get("sample_batch_count_delta", 0) or 0
            ),
            "bundle_checkpoint_count": int(
                bundle_longer.get("checkpoint_count", 0) or 0
            ),
            "bundle_learner_update_count_delta": int(
                bundle_longer.get("learner_update_count_delta", 0) or 0
            ),
        },
        "matched_surface": _matched_surface_summary(stock_eval, compact_eval, matched),
        "known_limitations": {
            "current_canary_scale": True,
            "mixed_hardware": str(matched_pair.get("hardware_class", "")) == "mixed",
            "compact_local_cpu": "local-cpu" in str(
                matched_pair.get("compact_hardware_class", "")
            ),
            "tiny_longer_horizon_trace": (
                int(bundle_longer.get("learner_update_count_delta", 0) or 0) <= 2
            ),
            "promotion_quality_claim": False,
            "compact_quality_superiority_claim": False,
            "bundle_current_matched_canary_sufficient_for_promotion_claim": (
                quality.get("current_matched_canary_sufficient_for_promotion_claim")
                is True
            ),
            "bundle_manual_acceptance_or_larger_study_required_before_promotion": (
                quality.get(
                    "manual_acceptance_or_larger_study_required_before_promotion"
                )
                is True
            ),
        },
        "bundle_matched_pair_summary": _plain_mapping(bundle_pair),
    }


def _matched_surface_summary(
    stock_eval: Mapping[str, Any],
    compact_eval: Mapping[str, Any],
    matched: Mapping[str, Any],
) -> dict[str, Any]:
    arms = _required_mapping(matched.get("arms"), "matched arms")
    compact = _required_mapping(arms.get(COMPACT_CANDIDATE_ROLE), COMPACT_CANDIDATE_ROLE)
    fingerprint = _required_mapping(
        compact.get("source_fingerprint"),
        "compact source_fingerprint",
    )
    source_surface = fingerprint.get("matched_surface")
    if not isinstance(source_surface, Mapping):
        source_surface = {}
    return {
        "env_variant": str(source_surface.get("env_variant", "")),
        "reward_variant": str(compact_eval.get("reward_variant", "")),
        "death_mode": str(compact_eval.get("death_mode", "")),
        "terminal_target_mode": str(compact_eval.get("terminal_target_mode", "")),
        "rnd_enabled": compact_eval.get("rnd_enabled") is True,
        "policy_observation_backend": str(
            compact_eval.get("policy_observation_backend", "")
        ),
        "opponent_policy_kind": str(compact_eval.get("opponent_policy_kind", "")),
        "opponent_runtime_mode": str(compact_eval.get("opponent_runtime_mode", "")),
        "opponent_death_mode": str(compact_eval.get("opponent_death_mode", "")),
        "natural_bonus_spawn": compact_eval.get("natural_bonus_spawn") is True,
        "num_simulations": int(compact_eval.get("num_simulations", 0) or 0),
        "batch_size": int(compact_eval.get("batch_size", 0) or 0),
        "root_noise": float(compact_eval.get("root_noise", 0.0) or 0.0),
        "policy_noise": float(compact_eval.get("policy_noise", 0.0) or 0.0),
        "source_max_steps": int(compact_eval.get("source_max_steps", 0) or 0),
        "stock_eval_settings_sha256": _json_sha256(stock_eval),
        "compact_eval_settings_sha256": _json_sha256(compact_eval),
    }


def _decision_payload(
    *,
    sufficiency_decision: str,
    current_summary: Mapping[str, Any],
    next_non_production_step: str | None,
) -> dict[str, Any]:
    if sufficiency_decision not in {
        DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP,
        DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY,
    }:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency decision is not allowed"
        )
    accepted_current = (
        sufficiency_decision == DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP
    )
    if accepted_current and not str(next_non_production_step or "").strip():
        raise CompactMatchedQualitySufficiencyReviewError(
            "accepted current canary review requires a named non-production next step"
        )
    limitations = _required_mapping(
        current_summary.get("known_limitations"),
        "known_limitations",
    )
    rationale = [
        "current matched-quality evidence is canary-scale, mixed-hardware, and not promotion proof",
        "the compact side was produced by the local CPU producer smoke path",
        "the longer-horizon compact trace is intentionally tiny",
        "promotion remains a separate policy and code decision",
    ]
    if not accepted_current:
        rationale.append(
            "no explicit external human acceptance is attached, so a larger same-surface study is required"
        )
    return {
        "matched_quality_sufficiency_decision": sufficiency_decision,
        "decision_scope": DECISION_SCOPE_NEXT_NON_PRODUCTION_ONLY,
        "decision_rationale": rationale,
        "next_non_production_step": str(next_non_production_step or ""),
        "current_evidence_accepted_for_next_non_production_step": accepted_current,
        "larger_same_surface_study_required": not accepted_current,
        "current_evidence_sufficient_for_promotion": False,
        "manual_review_required_before_any_promotion": True,
        "automatic_promotion_allowed": False,
        "promotion_claim": False,
        "limitation_acknowledgements": {
            "current_canary_scale": limitations.get("current_canary_scale") is True,
            "mixed_hardware": limitations.get("mixed_hardware") is True,
            "compact_local_cpu": limitations.get("compact_local_cpu") is True,
            "tiny_longer_horizon_trace": (
                limitations.get("tiny_longer_horizon_trace") is True
            ),
            "promotion_requires_separate_policy_decision": True,
            "manual_review_required_before_any_promotion": True,
        },
    }


def _larger_same_surface_study_plan(
    *,
    study_id: str,
    current_summary: Mapping[str, Any],
    matched: Mapping[str, Any],
    bundle: Mapping[str, Any],
    min_eval_seed_count: int,
    min_eval_max_steps: int,
    stock_reference_min_max_env_step: int,
    stock_reference_min_max_train_iter: int,
    compact_candidate_min_env_steps: int,
    compact_candidate_min_train_steps: int,
    eval_seed_rng_seed: int,
    larger_denominator_id: str,
    larger_quality_horizon: str,
) -> dict[str, Any]:
    current_eval_seed_count = int(current_summary.get("eval_seed_count", 0) or 0)
    current_eval_max_steps = int(current_summary.get("eval_max_steps", 0) or 0)
    if int(min_eval_seed_count) <= current_eval_seed_count:
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study eval seed count must exceed current evidence"
        )
    if int(min_eval_max_steps) <= current_eval_max_steps:
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study eval max steps must exceed current evidence"
        )
    candidate = _require_non_empty(
        bundle.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    study_id = _require_non_empty(study_id, "study_id")
    output_paths = _planned_output_paths(
        study_id=study_id,
        min_eval_seed_count=int(min_eval_seed_count),
        min_eval_max_steps=int(min_eval_max_steps),
        compact_candidate_min_env_steps=int(compact_candidate_min_env_steps),
        compact_candidate_min_train_steps=int(compact_candidate_min_train_steps),
    )
    plan = {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_STUDY_PLAN_SCHEMA_ID,
        "study_id": study_id,
        "plan_scope": "larger_same_surface_matched_learning_quality_before_promotion",
        "candidate_checkpoint_id": candidate,
        "same_surface_required": True,
        "promotion_claim": False,
        "automatic_promotion_allowed": False,
        "baseline_current_evidence": {
            "denominator_id": str(current_summary.get("denominator_id", "")),
            "quality_horizon": str(current_summary.get("quality_horizon", "")),
            "hardware_class": str(current_summary.get("hardware_class", "")),
            "eval_seed_count": current_eval_seed_count,
            "eval_max_steps": current_eval_max_steps,
        },
        "matched_surface": _plain_mapping(
            _required_mapping(current_summary.get("matched_surface"), "matched_surface")
        ),
        "minimum_scale_over_current": {
            "min_eval_seed_count": int(min_eval_seed_count),
            "min_eval_max_steps": int(min_eval_max_steps),
            "stock_reference_min_max_env_step": int(stock_reference_min_max_env_step),
            "stock_reference_min_max_train_iter": int(
                stock_reference_min_max_train_iter
            ),
            "compact_candidate_min_env_steps": int(compact_candidate_min_env_steps),
            "compact_candidate_min_train_steps": int(compact_candidate_min_train_steps),
            "current_eval_seed_count": current_eval_seed_count,
            "current_eval_max_steps": current_eval_max_steps,
            "current_longer_horizon_learner_update_count_delta": int(
                _required_mapping(
                    current_summary.get("longer_horizon_summary"),
                    "longer_horizon_summary",
                ).get("learner_update_count_delta", 0)
                or 0
            ),
        },
        "stock_evaluator_requirement": _stock_evaluator_requirement(
            eval_seed_count=int(min_eval_seed_count)
        ),
        "disallowed_shortcuts": {
            key: True for key in REQUIRED_DISALLOWED_SHORTCUT_KEYS
        },
        "producer_surfaces": {
            "stock": (
                "scripts/build_compact_matched_learning_quality_stock_reference_producer.py"
            ),
            "compact": (
                "scripts/build_compact_matched_learning_quality_compact_candidate_producer.py"
            ),
            "builder": "scripts/build_compact_matched_learning_quality_canary.py",
            "verifier": (
                "scripts/build_compact_matched_learning_quality_pair_verifier.py"
            ),
            "bundle_review": (
                "scripts/build_compact_promotion_readiness_bundle_review.py"
            ),
            "sufficiency_review": (
                "scripts/build_compact_matched_quality_sufficiency_review.py"
            ),
        },
        "planned_runs": _planned_runs(
            study_id=study_id,
            candidate=candidate,
            bundle=bundle,
            matched=matched,
            output_paths=output_paths,
            min_eval_seed_count=int(min_eval_seed_count),
            min_eval_max_steps=int(min_eval_max_steps),
            stock_reference_min_max_env_step=int(stock_reference_min_max_env_step),
            stock_reference_min_max_train_iter=int(
                stock_reference_min_max_train_iter
            ),
            compact_candidate_min_env_steps=int(compact_candidate_min_env_steps),
            compact_candidate_min_train_steps=int(compact_candidate_min_train_steps),
            eval_seed_rng_seed=int(eval_seed_rng_seed),
            larger_denominator_id=_require_non_empty(
                larger_denominator_id,
                "larger_denominator_id",
            ),
            larger_quality_horizon=_require_non_empty(
                larger_quality_horizon,
                "larger_quality_horizon",
            ),
        ),
        "required_outputs": output_paths,
        "decision_after_study": {
            "refresh_readiness_bundle_before_review": True,
            "rerun_sufficiency_review": True,
            "promotion_still_requires_separate_policy_decision": True,
            "automatic_promotion_allowed": False,
            "promotion_claim": False,
        },
    }
    _validate_larger_study_plan(plan, current_summary=current_summary, matched=matched)
    return plan


def _planned_output_paths(
    *,
    study_id: str,
    min_eval_seed_count: int,
    min_eval_max_steps: int,
    compact_candidate_min_env_steps: int,
    compact_candidate_min_train_steps: int,
) -> dict[str, str]:
    ids = _planned_run_ids(
        study_id=study_id,
        min_eval_seed_count=min_eval_seed_count,
        min_eval_max_steps=min_eval_max_steps,
        compact_candidate_min_env_steps=compact_candidate_min_env_steps,
        compact_candidate_min_train_steps=compact_candidate_min_train_steps,
    )
    stock_run_id = ids["stock_run_id"]
    compact_run_id = ids["compact_run_id"]
    canary_run_id = f"{study_id}-canary"
    pair_run_id = f"{study_id}-pair-verifier"
    bundle_run_id = ids["bundle_run_id"]
    review_run_id = ids["review_run_id"]
    matched_root = Path(
        "artifacts/local/curvytron_compact_matched_learning_quality_results"
    )
    readiness_root = Path(
        "artifacts/local/curvytron_compact_promotion_readiness_results"
    )
    return {
        "stock_reference_capture": str(
            matched_root / stock_run_id / "stock_reference_capture.json"
        ),
        "compact_candidate_capture": str(
            matched_root / compact_run_id / "compact_candidate_capture.json"
        ),
        "matched_learning_quality_canary_report": str(
            readiness_root / canary_run_id / "matched_learning_quality_canary_report.json"
        ),
        "matched_pair_verification_report": str(
            readiness_root / pair_run_id / "matched_pair_verification_report.json"
        ),
        "refreshed_readiness_bundle_review": str(
            readiness_root / bundle_run_id / "readiness_bundle_review_report.json"
        ),
        "updated_sufficiency_review": str(
            readiness_root
            / review_run_id
            / "matched_quality_sufficiency_review_report.json"
        ),
    }


def _planned_run_ids(
    *,
    study_id: str,
    min_eval_seed_count: int,
    min_eval_max_steps: int,
    compact_candidate_min_env_steps: int,
    compact_candidate_min_train_steps: int,
) -> dict[str, str]:
    study_suffix = f"{int(min_eval_max_steps)}x{int(min_eval_seed_count)}"
    compact_suffix = (
        f"{study_suffix}-env{int(compact_candidate_min_env_steps)}"
        f"train{int(compact_candidate_min_train_steps)}"
    )
    default_study_prefix = (
        f"optimizer-compact-matched-learning-quality-larger-{compact_suffix}-"
    )
    if not str(study_id).startswith(default_study_prefix):
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study id must match canonical scale/run-stamp pattern"
        )
    stamp = str(study_id)[len(default_study_prefix) :]
    if not stamp:
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study id must include a run stamp"
        )
    return {
        "study_suffix": study_suffix,
        "compact_suffix": compact_suffix,
        "study_run_stamp": stamp,
        "stock_run_id": (
            f"optimizer-stock-reference-quality-producer-larger-"
            f"{study_suffix}-{stamp}-evalenv{int(min_eval_seed_count)}"
        ),
        "compact_run_id": (
            f"optimizer-compact-candidate-env-search-replay-larger-"
            f"{compact_suffix}-{stamp}"
        ),
        "bundle_run_id": (
            f"optimizer-compact-promotion-readiness-bundle-review-larger-"
            f"{compact_suffix}-{stamp}"
        ),
        "review_run_id": (
            f"optimizer-compact-matched-quality-sufficiency-review-larger-"
            f"{compact_suffix}-{stamp}"
        ),
    }


def _planned_runs(
    *,
    study_id: str,
    candidate: str,
    bundle: Mapping[str, Any],
    matched: Mapping[str, Any],
    output_paths: Mapping[str, str],
    min_eval_seed_count: int,
    min_eval_max_steps: int,
    stock_reference_min_max_env_step: int,
    stock_reference_min_max_train_iter: int,
    compact_candidate_min_env_steps: int,
    compact_candidate_min_train_steps: int,
    eval_seed_rng_seed: int,
    larger_denominator_id: str,
    larger_quality_horizon: str,
) -> dict[str, dict[str, Any]]:
    bundle_inputs = _required_mapping(bundle.get("input_reports"), "bundle input_reports")
    lifecycle = _required_mapping(bundle_inputs.get("unified_lifecycle"), "unified_lifecycle")
    compatibility = _required_mapping(
        bundle_inputs.get("compatibility_refresh"),
        "compatibility_refresh",
    )
    stock_resume = _required_mapping(
        bundle_inputs.get("stock_resume_load_canary"),
        "stock_resume_load_canary",
    )
    isolated_live = _required_mapping(
        bundle_inputs.get("isolated_live_run_safety_canary"),
        "isolated_live_run_safety_canary",
    )
    sandbox_rating = _required_mapping(
        bundle_inputs.get("sandbox_assignment_rating_proof"),
        "sandbox_assignment_rating_proof",
    )
    longer_horizon = _required_mapping(
        bundle_inputs.get("longer_horizon_compact_learning_metrics"),
        "longer_horizon_compact_learning_metrics",
    )
    ids = _planned_run_ids(
        study_id=study_id,
        min_eval_seed_count=min_eval_seed_count,
        min_eval_max_steps=min_eval_max_steps,
        compact_candidate_min_env_steps=compact_candidate_min_env_steps,
        compact_candidate_min_train_steps=compact_candidate_min_train_steps,
    )
    stock_run_id = ids["stock_run_id"]
    compact_run_id = ids["compact_run_id"]
    canary_run_id = f"{study_id}-canary"
    pair_run_id = f"{study_id}-pair-verifier"
    bundle_run_id = ids["bundle_run_id"]
    review_run_id = ids["review_run_id"]
    return {
        "stock_reference_capture_producer": {
            "run_id": stock_run_id,
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_stock_reference_producer.py",
                "--run-id",
                stock_run_id,
                "--candidate-checkpoint-id",
                candidate,
                "--eval-seed-count",
                str(min_eval_seed_count),
                "--eval-steps",
                str(min_eval_max_steps),
                "--eval-seed-rng-seed",
                str(eval_seed_rng_seed),
                "--evaluator-env-num",
                str(min_eval_seed_count),
                "--n-evaluator-episode",
                str(min_eval_seed_count),
                "--max-env-step",
                str(stock_reference_min_max_env_step),
                "--max-train-iter",
                str(stock_reference_min_max_train_iter),
                "--denominator-id",
                larger_denominator_id,
                "--quality-horizon",
                larger_quality_horizon,
            ],
        },
        "compact_candidate_capture_producer": {
            "run_id": compact_run_id,
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_compact_candidate_producer.py",
                "--run-id",
                compact_run_id,
                "--unified-lifecycle-report",
                str(lifecycle.get("path", "")),
                "--candidate-checkpoint-id",
                candidate,
                "--eval-seed-count",
                str(min_eval_seed_count),
                "--eval-steps",
                str(min_eval_max_steps),
                "--eval-seed-rng-seed",
                str(eval_seed_rng_seed),
                "--compact-env-steps",
                str(compact_candidate_min_env_steps),
                "--train-steps",
                str(compact_candidate_min_train_steps),
                "--compact-sample-batch-size",
                "4",
                "--compact-replay-pair-capacity",
                "128",
                "--compact-training-mode",
                "env_search_replay",
                "--learner-device",
                "auto",
                "--denominator-id",
                larger_denominator_id,
                "--quality-horizon",
                larger_quality_horizon,
            ],
        },
        "matched_canary_builder": {
            "run_id": canary_run_id,
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_canary.py",
                "--run-id",
                canary_run_id,
                "--compatibility-report",
                str(compatibility.get("path", "")),
                "--unified-lifecycle-report",
                str(lifecycle.get("path", "")),
                "--stock-reference-capture",
                output_paths["stock_reference_capture"],
                "--compact-candidate-capture",
                output_paths["compact_candidate_capture"],
            ],
        },
        "matched_pair_verifier": {
            "run_id": pair_run_id,
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_pair_verifier.py",
                "--run-id",
                pair_run_id,
                "--matched-learning-quality-report",
                output_paths["matched_learning_quality_canary_report"],
            ],
        },
        "readiness_bundle_refresh": {
            "run_id": bundle_run_id,
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_promotion_readiness_bundle_review.py",
                "--run-id",
                bundle_run_id,
                "--compatibility-report",
                str(compatibility.get("path", "")),
                "--unified-lifecycle-report",
                str(lifecycle.get("path", "")),
                "--matched-learning-quality-report",
                output_paths["matched_learning_quality_canary_report"],
                "--matched-pair-verification-report",
                output_paths["matched_pair_verification_report"],
                "--stock-resume-load-canary-report",
                str(stock_resume.get("path", "")),
                "--isolated-live-run-safety-canary-report",
                str(isolated_live.get("path", "")),
                "--sandbox-assignment-rating-proof-report",
                str(sandbox_rating.get("path", "")),
                "--longer-horizon-learning-metrics-report",
                str(longer_horizon.get("path", "")),
            ],
        },
        "sufficiency_review_update": {
            "run_id": review_run_id,
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_quality_sufficiency_review.py",
                "--run-id",
                review_run_id,
                "--readiness-bundle-review",
                output_paths["refreshed_readiness_bundle_review"],
                "--sufficiency-decision",
                DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY,
                "--larger-study-id",
                study_id,
                "--min-eval-seed-count",
                str(min_eval_seed_count),
                "--min-eval-max-steps",
                str(min_eval_max_steps),
                "--stock-reference-min-max-env-step",
                str(stock_reference_min_max_env_step),
                "--stock-reference-min-max-train-iter",
                str(stock_reference_min_max_train_iter),
                "--compact-candidate-min-env-steps",
                str(compact_candidate_min_env_steps),
                "--compact-candidate-min-train-steps",
                str(compact_candidate_min_train_steps),
                "--eval-seed-rng-seed",
                str(eval_seed_rng_seed),
                "--larger-denominator-id",
                larger_denominator_id,
                "--larger-quality-horizon",
                larger_quality_horizon,
            ],
        },
    }


def _validate_decision(payload: Mapping[str, Any]) -> str:
    decision = _required_mapping(payload.get("decision"), "decision")
    decision_value = str(decision.get("matched_quality_sufficiency_decision", ""))
    if decision_value not in {
        DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP,
        DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY,
    }:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency decision is not allowed"
        )
    if decision.get("decision_scope") != DECISION_SCOPE_NEXT_NON_PRODUCTION_ONLY:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency decision scope mismatch"
        )
    forbidden_fragments = ("sufficient_for_promotion", "promotion_ready")
    if any(fragment in decision_value for fragment in forbidden_fragments):
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality sufficiency decision would overclaim"
        )
    for key in (
        "current_evidence_sufficient_for_promotion",
        "automatic_promotion_allowed",
        "promotion_claim",
    ):
        if decision.get(key) is not False:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"matched-quality sufficiency decision {key} must be false"
            )
    if decision.get("manual_review_required_before_any_promotion") is not True:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality review must require manual review before promotion"
        )
    acknowledgements = _required_mapping(
        decision.get("limitation_acknowledgements"),
        "limitation_acknowledgements",
    )
    for key in REQUIRED_LIMITATION_ACKNOWLEDGEMENTS:
        if acknowledgements.get(key) is not True:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"matched-quality limitation acknowledgement missing: {key}"
            )
    if decision_value == DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP:
        if decision.get("current_evidence_accepted_for_next_non_production_step") is not True:
            raise CompactMatchedQualitySufficiencyReviewError(
                "accepted current canary decision must mark non-production acceptance"
            )
        if decision.get("larger_same_surface_study_required") is not False:
            raise CompactMatchedQualitySufficiencyReviewError(
                "accepted current canary decision must not require a larger study"
            )
        _require_non_empty(
            decision.get("next_non_production_step"),
            "next_non_production_step",
        )
    else:
        if decision.get("current_evidence_accepted_for_next_non_production_step") is not False:
            raise CompactMatchedQualitySufficiencyReviewError(
                "larger study decision must not accept current canary"
            )
        if decision.get("larger_same_surface_study_required") is not True:
            raise CompactMatchedQualitySufficiencyReviewError(
                "larger study decision must require a larger same-surface study"
            )
    return decision_value


def _validate_larger_study_plan(
    plan: Any,
    *,
    current_summary: Mapping[str, Any],
    matched: Mapping[str, Any],
) -> None:
    study_plan = _required_mapping(plan, "larger_study_plan")
    if study_plan.get("schema_id") != COMPACT_MATCHED_LEARNING_QUALITY_STUDY_PLAN_SCHEMA_ID:
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study plan schema mismatch"
        )
    if study_plan.get("same_surface_required") is not True:
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study must require the same matched eval surface"
        )
    for key in ("promotion_claim", "automatic_promotion_allowed"):
        if study_plan.get(key) is not False:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"larger study plan {key} must be false"
            )
    surface = _required_mapping(study_plan.get("matched_surface"), "matched_surface")
    expected_surface = _required_mapping(
        current_summary.get("matched_surface"),
        "current matched_surface",
    )
    if _json_sha256(surface) != _json_sha256(expected_surface):
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study plan matched surface drift"
        )
    scale = _required_mapping(
        study_plan.get("minimum_scale_over_current"),
        "minimum_scale_over_current",
    )
    if int(scale.get("min_eval_seed_count", 0) or 0) <= int(
        current_summary.get("eval_seed_count", 0) or 0
    ):
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study eval seed count must exceed current evidence"
        )
    if int(scale.get("min_eval_max_steps", 0) or 0) <= int(
        current_summary.get("eval_max_steps", 0) or 0
    ):
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study eval max steps must exceed current evidence"
        )
    for key in (
        "stock_reference_min_max_env_step",
        "stock_reference_min_max_train_iter",
        "compact_candidate_min_env_steps",
        "compact_candidate_min_train_steps",
    ):
        if int(scale.get(key, 0) or 0) <= 0:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"larger study scale {key} must be positive"
            )
    shortcuts = _required_mapping(
        study_plan.get("disallowed_shortcuts"),
        "disallowed_shortcuts",
    )
    for key in REQUIRED_DISALLOWED_SHORTCUT_KEYS:
        if shortcuts.get(key) is not True:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"larger study disallowed shortcut missing: {key}"
            )
    runs = _required_mapping(study_plan.get("planned_runs"), "planned_runs")
    for key in REQUIRED_STUDY_RUN_KEYS:
        run = _required_mapping(runs.get(key), key)
        argv = run.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(item, str) and item for item in argv
        ):
            raise CompactMatchedQualitySufficiencyReviewError(
                f"larger study planned run {key} argv must be string array"
            )
    stock_requirement = _validate_stock_evaluator_requirement(
        study_plan.get("stock_evaluator_requirement"),
        eval_seed_count=int(scale["min_eval_seed_count"]),
    )
    stock_run = _required_mapping(
        runs.get("stock_reference_capture_producer"),
        "stock_reference_capture_producer",
    )
    stock_argv = _required_sequence(
        stock_run.get("argv"),
        "stock_reference_capture_producer argv",
    )
    _require_argv_pair(
        stock_argv,
        "--evaluator-env-num",
        str(stock_requirement["evaluator_env_num"]),
        label="stock_reference_capture_producer",
    )
    _require_argv_pair(
        stock_argv,
        "--n-evaluator-episode",
        str(stock_requirement["n_evaluator_episode"]),
        label="stock_reference_capture_producer",
    )
    outputs = _required_mapping(study_plan.get("required_outputs"), "required_outputs")
    for key in REQUIRED_STUDY_OUTPUT_KEYS:
        output = _require_non_empty(outputs.get(key), f"{key} output")
        if "preview" in output:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"larger study output must not be preview capture: {key}"
            )
    _validate_study_uses_fresh_outputs(study_plan, matched=matched)
    after = _required_mapping(
        study_plan.get("decision_after_study"),
        "decision_after_study",
    )
    if after.get("refresh_readiness_bundle_before_review") is not True:
        raise CompactMatchedQualitySufficiencyReviewError(
            "larger study must refresh readiness bundle before review"
        )
    for key in ("automatic_promotion_allowed", "promotion_claim"):
        if after.get(key) is not False:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"larger study decision_after_study {key} must be false"
            )


def _validate_study_uses_fresh_outputs(
    study_plan: Mapping[str, Any],
    *,
    matched: Mapping[str, Any],
) -> None:
    outputs = _required_mapping(study_plan.get("required_outputs"), "required_outputs")
    captures = _required_mapping(
        matched.get("input_capture_files"),
        "matched input_capture_files",
    )
    current_paths = {
        str(_required_mapping(captures.get(key), key).get("path", ""))
        for key in ("stock_reference_capture", "compact_candidate_capture")
    }
    for output in outputs.values():
        if str(output) in current_paths:
            raise CompactMatchedQualitySufficiencyReviewError(
                "larger study output path must be fresh"
            )


def _stock_evaluator_requirement(*, eval_seed_count: int) -> dict[str, Any]:
    count = int(eval_seed_count)
    return {
        "policy": STOCK_EVALUATOR_REQUIREMENT_POLICY,
        "eval_seed_count": count,
        "evaluator_env_num": count,
        "n_evaluator_episode": count,
        "env_num_matches_episode_count_required": True,
        "empty_ready_set_workaround_formalized": True,
        "canonical_stock_evaluator_required": False,
    }


def _validate_stock_evaluator_requirement(
    value: Any,
    *,
    eval_seed_count: int,
) -> Mapping[str, Any]:
    requirement = _required_mapping(value, "stock_evaluator_requirement")
    expected = _stock_evaluator_requirement(eval_seed_count=int(eval_seed_count))
    for key, expected_value in expected.items():
        if requirement.get(key) != expected_value:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"stock evaluator requirement {key} mismatch"
            )
    return requirement


def _require_argv_pair(
    argv: Sequence[Any],
    flag: str,
    expected: str,
    *,
    label: str,
) -> None:
    values = [str(item) for item in argv]
    try:
        index = values.index(flag)
    except ValueError as exc:
        raise CompactMatchedQualitySufficiencyReviewError(
            f"{label} argv missing {flag}"
        ) from exc
    if index + 1 >= len(values) or values[index + 1] != str(expected):
        raise CompactMatchedQualitySufficiencyReviewError(
            f"{label} argv {flag} mismatch"
        )


def _validate_claims(payload: Mapping[str, Any], *, decision_value: str) -> None:
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    for key in TRUE_ATTACHED_CLAIM_KEYS:
        if claims.get(key) is not True:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"matched-quality attached claim missing: {key}"
            )
    expected_larger = decision_value == DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY
    if claims.get("larger_same_surface_study_required") is not expected_larger:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality larger-study attached claim mismatch"
        )
    if claims.get("current_canary_accepted_for_next_non_production_step") is expected_larger:
        raise CompactMatchedQualitySufficiencyReviewError(
            "matched-quality non-production acceptance attached claim mismatch"
        )
    for key in FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"matched-quality attached claim {key} must be false"
            )
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for key in FALSE_CLAIM_KEYS:
        if non_claims.get(key) is not False:
            raise CompactMatchedQualitySufficiencyReviewError(
                f"matched-quality non-claim {key} must be false"
            )


def _status_for_decision(sufficiency_decision: str) -> str:
    if sufficiency_decision == DECISION_ACCEPT_CURRENT_FOR_NEXT_NON_PRODUCTION_STEP:
        return STATUS_CURRENT_CANARY_ACCEPTED_FOR_NEXT_NON_PRODUCTION_STEP
    if sufficiency_decision == DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY:
        return STATUS_LARGER_SAME_SURFACE_STUDY_REQUIRED
    raise CompactMatchedQualitySufficiencyReviewError(
        "matched-quality sufficiency decision is not allowed"
    )


def _false_claims() -> dict[str, bool]:
    return {key: False for key in FALSE_CLAIM_KEYS}


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompactMatchedQualitySufficiencyReviewError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompactMatchedQualitySufficiencyReviewError(
            f"{label} is not readable JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CompactMatchedQualitySufficiencyReviewError(f"{label} must be a mapping")
    return dict(payload)


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactMatchedQualitySufficiencyReviewError(f"{label} must be a mapping")
    return value


def _required_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CompactMatchedQualitySufficiencyReviewError(
            f"{label} must be a sequence"
        )
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactMatchedQualitySufficiencyReviewError(
            f"{label} must be non-empty"
        )
    return text


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(val) for key, val in value.items()}


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
