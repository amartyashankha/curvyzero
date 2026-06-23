"""Plan the larger matched-quality study requested by the bundle review."""

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
    FALSE_CLAIM_KEYS,
)
from curvyzero.training.compact_promotion_readiness_bundle_review import (
    validate_compact_promotion_readiness_bundle_review_v1,
)


COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_SCHEMA_ID = (
    "curvyzero_compact_matched_quality_larger_study_plan/v1"
)
COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_STATUS = (
    "larger_matched_quality_study_required"
)
COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_EVIDENCE_REF_PREFIX = (
    "compact_matched_quality_larger_study_plan:"
)

DEFAULT_BUNDLE_REVIEW_REPORT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-promotion-readiness-bundle-review-20260530a"
    "/readiness_bundle_review_report.json"
)
DEFAULT_MIN_EVAL_SEED_COUNT = 32
DEFAULT_MIN_EVAL_MAX_STEPS = 2048
DEFAULT_STOCK_MAX_ENV_STEP = 2048
DEFAULT_STOCK_MAX_TRAIN_ITER = 4
DEFAULT_COMPACT_ENV_STEPS = 64
DEFAULT_COMPACT_TRAIN_STEPS = 8
DEFAULT_COMPACT_SAMPLE_BATCH_SIZE = 4
DEFAULT_COMPACT_REPLAY_PAIR_CAPACITY = 128
DEFAULT_EVAL_SEED_RNG_SEED = 20260833
DEFAULT_STUDY_RUN_STAMP = "20260531"
STOCK_EVALUATOR_REQUIREMENT_POLICY = "evalenv_full_episode_surface"

PLANNED_RUN_KEYS = (
    "stock_reference_capture",
    "compact_candidate_capture",
    "matched_learning_quality_canary",
    "matched_pair_verification",
    "refreshed_bundle_review",
)
TRUE_PLAN_CLAIM_KEYS = (
    "larger_matched_quality_study_plan",
    "fresh_outputs_required",
    "same_eval_surface_required",
    "final_bundle_review_refresh_required",
)


class CompactMatchedQualityStudyPlanError(ValueError):
    """Raised when a larger matched-quality study plan would overclaim."""


def build_compact_matched_quality_larger_study_plan_v1(
    *,
    run_id: str,
    bundle_review_report_path: str | Path = DEFAULT_BUNDLE_REVIEW_REPORT,
    min_eval_seed_count: int = DEFAULT_MIN_EVAL_SEED_COUNT,
    min_eval_max_steps: int = DEFAULT_MIN_EVAL_MAX_STEPS,
    stock_max_env_step: int = DEFAULT_STOCK_MAX_ENV_STEP,
    stock_max_train_iter: int = DEFAULT_STOCK_MAX_TRAIN_ITER,
    compact_env_steps: int = DEFAULT_COMPACT_ENV_STEPS,
    compact_train_steps: int = DEFAULT_COMPACT_TRAIN_STEPS,
    compact_sample_batch_size: int = DEFAULT_COMPACT_SAMPLE_BATCH_SIZE,
    compact_replay_pair_capacity: int = DEFAULT_COMPACT_REPLAY_PAIR_CAPACITY,
    eval_seed_rng_seed: int = DEFAULT_EVAL_SEED_RNG_SEED,
    study_run_stamp: str = DEFAULT_STUDY_RUN_STAMP,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a no-claim plan for the next larger matched-quality study."""

    bundle_path = Path(bundle_review_report_path).resolve()
    bundle = _read_json_mapping(bundle_path, "bundle review report")
    validate_compact_promotion_readiness_bundle_review_v1(bundle)
    current = _current_evidence_summary(bundle)
    candidate = _require_non_empty(
        bundle.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    requirements = _study_requirements(
        current,
        min_eval_seed_count=int(min_eval_seed_count),
        min_eval_max_steps=int(min_eval_max_steps),
        stock_max_env_step=int(stock_max_env_step),
        stock_max_train_iter=int(stock_max_train_iter),
        compact_env_steps=int(compact_env_steps),
        compact_train_steps=int(compact_train_steps),
        compact_sample_batch_size=int(compact_sample_batch_size),
        compact_replay_pair_capacity=int(compact_replay_pair_capacity),
        eval_seed_rng_seed=int(eval_seed_rng_seed),
    )
    non_claims = {key: False for key in FALSE_CLAIM_KEYS}
    planned_runs = _planned_runs(
        bundle,
        candidate_checkpoint_id=candidate,
        requirements=requirements,
        study_run_stamp=str(study_run_stamp),
    )
    payload = {
        "schema_id": COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_SCHEMA_ID,
        "ok": True,
        "status": COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_STATUS,
        "run_id": str(run_id),
        "study_run_stamp": _require_non_empty(study_run_stamp, "study_run_stamp"),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "candidate_checkpoint_id": candidate,
        "source_bundle_review": {
            "path": str(bundle_path),
            "sha256": _file_sha256(bundle_path),
            "schema_id": bundle.get("schema_id"),
            "evidence_ref": bundle.get("evidence_ref"),
            "status": bundle.get("status"),
        },
        "current_evidence_summary": current,
        "decision": {
            "current_evidence_sufficient_for_promotion": False,
            "larger_matched_quality_study_required": True,
            "manual_acceptance_alternative_requires_external_decision": True,
            "promotion_claim": False,
            "automatic_promotion_allowed": False,
        },
        "study_requirements": requirements,
        "planned_runs": planned_runs,
        "required_output_artifacts": _required_output_artifacts(planned_runs),
        "attached_claims": {
            key: True for key in TRUE_PLAN_CLAIM_KEYS
        }
        | non_claims,
        "non_claims": non_claims,
    }
    payload["evidence_ref"] = compact_matched_quality_larger_study_plan_evidence_ref(
        payload
    )
    validate_compact_matched_quality_larger_study_plan_v1(payload)
    return payload


def validate_compact_matched_quality_larger_study_plan_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate a no-claim larger matched-quality study plan."""

    if payload.get("schema_id") != COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_SCHEMA_ID:
        raise CompactMatchedQualityStudyPlanError(
            "matched-quality larger study plan schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactMatchedQualityStudyPlanError(
            "matched-quality larger study plan must be ok=true"
        )
    if payload.get("status") != COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_STATUS:
        raise CompactMatchedQualityStudyPlanError(
            "matched-quality larger study plan status mismatch"
        )
    source = _required_mapping(payload.get("source_bundle_review"), "source bundle")
    bundle_path = Path(_require_non_empty(source.get("path"), "source bundle path"))
    if not bundle_path.is_file():
        raise CompactMatchedQualityStudyPlanError("source bundle report missing")
    if _file_sha256(bundle_path) != source.get("sha256"):
        raise CompactMatchedQualityStudyPlanError("source bundle sha256 mismatch")
    bundle = _read_json_mapping(bundle_path, "source bundle report")
    validate_compact_promotion_readiness_bundle_review_v1(bundle)
    if bundle.get("schema_id") != COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID:
        raise CompactMatchedQualityStudyPlanError("source bundle schema mismatch")
    if source.get("evidence_ref") != bundle.get("evidence_ref"):
        raise CompactMatchedQualityStudyPlanError("source bundle evidence_ref mismatch")
    if payload.get("candidate_checkpoint_id") != bundle.get("candidate_checkpoint_id"):
        raise CompactMatchedQualityStudyPlanError(
            "matched-quality study plan candidate mismatch"
        )
    expected_current = _current_evidence_summary(bundle)
    if _json_sha256(payload.get("current_evidence_summary")) != _json_sha256(
        expected_current
    ):
        raise CompactMatchedQualityStudyPlanError(
            "matched-quality study plan current evidence summary drift"
        )
    requirements = _required_mapping(
        payload.get("study_requirements"),
        "study_requirements",
    )
    _validate_requirements(requirements, current=expected_current)
    _validate_decision(payload)
    _validate_planned_runs(
        payload,
        candidate_checkpoint_id=str(bundle["candidate_checkpoint_id"]),
        requirements=requirements,
    )
    _validate_claims(payload)
    expected_ref = compact_matched_quality_larger_study_plan_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactMatchedQualityStudyPlanError(
            "matched-quality larger study plan evidence_ref mismatch"
        )


def compact_matched_quality_larger_study_plan_evidence_ref(
    payload: Mapping[str, Any],
) -> str:
    """Return a stable evidence ref for the larger-study plan."""

    candidate = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    source = _required_mapping(payload.get("source_bundle_review"), "source bundle")
    digest_source = {
        "source_bundle_sha256": source.get("sha256"),
        "study_requirements": payload.get("study_requirements"),
        "planned_run_ids": {
            key: _required_mapping(
                _required_mapping(payload.get("planned_runs"), "planned_runs").get(key),
                key,
            ).get("run_id")
            for key in PLANNED_RUN_KEYS
        },
    }
    return (
        f"{COMPACT_MATCHED_QUALITY_LARGER_STUDY_PLAN_EVIDENCE_REF_PREFIX}"
        f"{candidate}:{_json_sha256(digest_source)[:16]}"
    )


def _current_evidence_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    decision = _required_mapping(bundle.get("review_decision"), "review_decision")
    quality = _required_mapping(
        bundle.get("quality_strength_review"),
        "quality_strength_review",
    )
    matched = _required_mapping(
        quality.get("matched_pair_summary"),
        "matched_pair_summary",
    )
    longer = _required_mapping(
        quality.get("longer_horizon_summary"),
        "longer_horizon_summary",
    )
    if decision.get("promotion_claim") is not False:
        raise CompactMatchedQualityStudyPlanError("source bundle promotion claim")
    if decision.get("automatic_promotion_allowed") is not False:
        raise CompactMatchedQualityStudyPlanError(
            "source bundle allows automatic promotion"
        )
    if quality.get("current_matched_canary_sufficient_for_promotion_claim") is not False:
        raise CompactMatchedQualityStudyPlanError(
            "source bundle says canary is promotion-sufficient"
        )
    if quality.get("manual_acceptance_or_larger_study_required_before_promotion") is not True:
        raise CompactMatchedQualityStudyPlanError(
            "source bundle does not require manual acceptance or larger study"
        )
    return {
        "source_bundle_status": str(bundle.get("status", "")),
        "source_ready_for_manual_review": decision.get("ready_for_manual_review")
        is True,
        "source_promotion_claim": False,
        "source_automatic_promotion_allowed": False,
        "matched_quality_sufficiency_decision": str(
            decision.get("matched_quality_sufficiency_decision", "")
        ),
        "matched_eval_seed_count": int(matched.get("eval_seed_count", 0) or 0),
        "matched_eval_max_steps": int(matched.get("eval_max_steps", 0) or 0),
        "matched_hardware_class": str(matched.get("hardware_class", "")),
        "stock_hardware_class": str(matched.get("stock_hardware_class", "")),
        "compact_hardware_class": str(matched.get("compact_hardware_class", "")),
        "stock_calls_train_muzero": matched.get("stock_calls_train_muzero") is True,
        "compact_calls_train_muzero": matched.get("compact_calls_train_muzero") is True,
        "longer_horizon_checkpoint_count": int(
            longer.get("checkpoint_count", 0) or 0
        ),
        "longer_horizon_learner_update_count_delta": int(
            longer.get("learner_update_count_delta", 0) or 0
        ),
    }


def _study_requirements(
    current: Mapping[str, Any],
    *,
    min_eval_seed_count: int,
    min_eval_max_steps: int,
    stock_max_env_step: int,
    stock_max_train_iter: int,
    compact_env_steps: int,
    compact_train_steps: int,
    compact_sample_batch_size: int,
    compact_replay_pair_capacity: int,
    eval_seed_rng_seed: int,
) -> dict[str, Any]:
    requirements = {
        "min_eval_seed_count": int(min_eval_seed_count),
        "min_eval_max_steps": int(min_eval_max_steps),
        "eval_seed_rng_seed": int(eval_seed_rng_seed),
        "stock_reference": {
            "producer": "scripts/build_compact_matched_learning_quality_stock_reference_producer.py",
            "min_max_env_step": int(stock_max_env_step),
            "min_max_train_iter": int(stock_max_train_iter),
            "requires_real_train_muzero": True,
            "forbids_fallback_denominator": True,
            "stock_evaluator_requirement": _stock_evaluator_requirement(
                eval_seed_count=min_eval_seed_count
            ),
        },
        "compact_candidate": {
            "producer": "scripts/build_compact_matched_learning_quality_compact_candidate_producer.py",
            "min_compact_env_steps": int(compact_env_steps),
            "min_train_steps": int(compact_train_steps),
            "min_compact_sample_batch_size": int(compact_sample_batch_size),
            "min_compact_replay_pair_capacity": int(compact_replay_pair_capacity),
            "requires_env_search_replay_rows": True,
            "forbids_resident_sample_scaffold": True,
            "forbids_train_muzero": True,
        },
        "same_eval_surface_required": True,
        "fresh_outputs_required": True,
        "no_preview_captures": True,
        "no_profile_rows_as_quality_evidence": True,
        "final_bundle_review_refresh_required": True,
    }
    _validate_requirements(requirements, current=current)
    return requirements


def _planned_runs(
    bundle: Mapping[str, Any],
    *,
    candidate_checkpoint_id: str,
    requirements: Mapping[str, Any],
    study_run_stamp: str,
) -> dict[str, Any]:
    stamp = _require_non_empty(study_run_stamp, "study_run_stamp")
    eval_seed_count = int(requirements["min_eval_seed_count"])
    eval_steps = int(requirements["min_eval_max_steps"])
    study_suffix = f"{eval_steps}x{eval_seed_count}"
    compact_suffix = (
        f"{study_suffix}-env{requirements['compact_candidate']['min_compact_env_steps']}"
        f"train{requirements['compact_candidate']['min_train_steps']}"
    )
    denominator_id = f"matched_quality_larger_{compact_suffix}_denominator_v1"
    quality_horizon = f"matched_learning_quality_pre_post_eval_larger_{compact_suffix}"
    stock_evaluator = _required_mapping(
        requirements["stock_reference"].get("stock_evaluator_requirement"),
        "stock_evaluator_requirement",
    )
    stock_run_id = (
        f"optimizer-stock-reference-quality-producer-larger-{study_suffix}-{stamp}"
        f"-evalenv{stock_evaluator['evaluator_env_num']}"
    )
    compact_run_id = (
        f"optimizer-compact-candidate-env-search-replay-larger-{compact_suffix}-{stamp}"
    )
    canary_run_id = (
        f"optimizer-compact-matched-learning-quality-larger-"
        f"{compact_suffix}-{stamp}-canary"
    )
    verifier_run_id = (
        f"optimizer-compact-matched-learning-quality-larger-"
        f"{compact_suffix}-{stamp}-pair-verifier"
    )
    bundle_run_id = (
        f"optimizer-compact-promotion-readiness-bundle-review-larger-"
        f"{compact_suffix}-{stamp}"
    )

    inputs = _required_mapping(bundle.get("input_reports"), "input_reports")
    compatibility_path = _required_mapping(inputs.get("compatibility_refresh"), "compatibility").get(
        "path"
    )
    lifecycle_path = _required_mapping(inputs.get("unified_lifecycle"), "lifecycle").get(
        "path"
    )
    stock_resume_path = _required_mapping(
        inputs.get("stock_resume_load_canary"),
        "stock_resume_load_canary",
    ).get("path")
    isolated_live_path = _required_mapping(
        inputs.get("isolated_live_run_safety_canary"),
        "isolated_live_run_safety_canary",
    ).get("path")
    sandbox_rating_path = _required_mapping(
        inputs.get("sandbox_assignment_rating_proof"),
        "sandbox_assignment_rating_proof",
    ).get("path")
    longer_horizon_path = _required_mapping(
        inputs.get("longer_horizon_compact_learning_metrics"),
        "longer_horizon_compact_learning_metrics",
    ).get("path")
    output_root = "artifacts/local/curvytron_compact_matched_learning_quality_results"
    readiness_root = "artifacts/local/curvytron_compact_promotion_readiness_results"
    stock_capture = (
        f"{output_root}/{stock_run_id}/stock_reference_capture.json"
    )
    compact_capture = (
        f"{output_root}/{compact_run_id}/compact_candidate_capture.json"
    )
    matched_report = (
        f"{readiness_root}/{canary_run_id}/matched_learning_quality_canary_report.json"
    )
    pair_report = (
        f"{readiness_root}/{verifier_run_id}/matched_pair_verification_report.json"
    )
    return {
        "stock_reference_capture": {
            "run_id": stock_run_id,
            "role": "stock_reference",
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_stock_reference_producer.py",
                "--run-id",
                stock_run_id,
                "--candidate-checkpoint-id",
                candidate_checkpoint_id,
                "--eval-seed-count",
                str(eval_seed_count),
                "--eval-seed-rng-seed",
                str(requirements["eval_seed_rng_seed"]),
                "--eval-steps",
                str(eval_steps),
                "--evaluator-env-num",
                str(stock_evaluator["evaluator_env_num"]),
                "--n-evaluator-episode",
                str(stock_evaluator["n_evaluator_episode"]),
                "--max-env-step",
                str(requirements["stock_reference"]["min_max_env_step"]),
                "--max-train-iter",
                str(requirements["stock_reference"]["min_max_train_iter"]),
                "--denominator-id",
                denominator_id,
                "--quality-horizon",
                quality_horizon,
            ],
            "expected_capture_path": stock_capture,
        },
        "compact_candidate_capture": {
            "run_id": compact_run_id,
            "role": "compact_candidate",
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_compact_candidate_producer.py",
                "--run-id",
                compact_run_id,
                "--unified-lifecycle-report",
                str(lifecycle_path),
                "--candidate-checkpoint-id",
                candidate_checkpoint_id,
                "--eval-seed-count",
                str(eval_seed_count),
                "--eval-seed-rng-seed",
                str(requirements["eval_seed_rng_seed"]),
                "--eval-steps",
                str(eval_steps),
                "--compact-env-steps",
                str(requirements["compact_candidate"]["min_compact_env_steps"]),
                "--train-steps",
                str(requirements["compact_candidate"]["min_train_steps"]),
                "--compact-sample-batch-size",
                str(requirements["compact_candidate"]["min_compact_sample_batch_size"]),
                "--compact-replay-pair-capacity",
                str(requirements["compact_candidate"]["min_compact_replay_pair_capacity"]),
                "--compact-training-mode",
                "env_search_replay",
                "--learner-device",
                "auto",
                "--denominator-id",
                denominator_id,
                "--quality-horizon",
                quality_horizon,
            ],
            "expected_capture_path": compact_capture,
        },
        "matched_learning_quality_canary": {
            "run_id": canary_run_id,
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_canary.py",
                "--run-id",
                canary_run_id,
                "--compatibility-report",
                str(compatibility_path),
                "--unified-lifecycle-report",
                str(lifecycle_path),
                "--stock-reference-capture",
                stock_capture,
                "--compact-candidate-capture",
                compact_capture,
            ],
            "expected_report_path": matched_report,
        },
        "matched_pair_verification": {
            "run_id": verifier_run_id,
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_matched_learning_quality_pair_verifier.py",
                "--run-id",
                verifier_run_id,
                "--matched-learning-quality-report",
                matched_report,
            ],
            "expected_report_path": pair_report,
        },
        "refreshed_bundle_review": {
            "run_id": bundle_run_id,
            "argv": [
                "uv",
                "run",
                "python",
                "scripts/build_compact_promotion_readiness_bundle_review.py",
                "--run-id",
                bundle_run_id,
                "--compatibility-report",
                str(compatibility_path),
                "--unified-lifecycle-report",
                str(lifecycle_path),
                "--matched-learning-quality-report",
                matched_report,
                "--matched-pair-verification-report",
                pair_report,
                "--stock-resume-load-canary-report",
                str(stock_resume_path),
                "--isolated-live-run-safety-canary-report",
                str(isolated_live_path),
                "--sandbox-assignment-rating-proof-report",
                str(sandbox_rating_path),
                "--longer-horizon-learning-metrics-report",
                str(longer_horizon_path),
            ],
            "expected_report_path": (
                f"{readiness_root}/{bundle_run_id}/readiness_bundle_review_report.json"
            ),
        },
    }


def _required_output_artifacts(planned_runs: Mapping[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for key in PLANNED_RUN_KEYS:
        run = _required_mapping(planned_runs.get(key), key)
        path = run.get("expected_report_path") or run.get("expected_capture_path")
        outputs[key] = _require_non_empty(path, f"{key} expected output")
    return outputs


def _validate_requirements(
    requirements: Mapping[str, Any],
    *,
    current: Mapping[str, Any],
) -> None:
    current_seed_count = int(current.get("matched_eval_seed_count", 0) or 0)
    current_eval_steps = int(current.get("matched_eval_max_steps", 0) or 0)
    if int(requirements.get("min_eval_seed_count", 0) or 0) <= current_seed_count:
        raise CompactMatchedQualityStudyPlanError(
            "larger study eval seed count must exceed current evidence"
        )
    if int(requirements.get("min_eval_max_steps", 0) or 0) <= current_eval_steps:
        raise CompactMatchedQualityStudyPlanError(
            "larger study eval max steps must exceed current evidence"
        )
    if int(requirements.get("min_eval_seed_count", 0) or 0) < 16:
        raise CompactMatchedQualityStudyPlanError(
            "larger study must require at least 16 eval seeds"
        )
    if int(requirements.get("min_eval_max_steps", 0) or 0) < 1024:
        raise CompactMatchedQualityStudyPlanError(
            "larger study must require at least 1024 eval steps"
        )
    stock = _required_mapping(requirements.get("stock_reference"), "stock_reference")
    compact = _required_mapping(
        requirements.get("compact_candidate"),
        "compact_candidate",
    )
    if int(stock.get("min_max_env_step", 0) or 0) < 1024:
        raise CompactMatchedQualityStudyPlanError(
            "stock larger study env-step budget too small"
        )
    if int(stock.get("min_max_train_iter", 0) or 0) < 2:
        raise CompactMatchedQualityStudyPlanError(
            "stock larger study train-iter budget too small"
        )
    if int(compact.get("min_compact_env_steps", 0) or 0) < 32:
        raise CompactMatchedQualityStudyPlanError(
            "compact larger study env-search rows too small"
        )
    if int(compact.get("min_train_steps", 0) or 0) < 4:
        raise CompactMatchedQualityStudyPlanError(
            "compact larger study train steps too small"
        )
    for key in (
        "same_eval_surface_required",
        "fresh_outputs_required",
        "no_preview_captures",
        "no_profile_rows_as_quality_evidence",
        "final_bundle_review_refresh_required",
    ):
        if requirements.get(key) is not True:
            raise CompactMatchedQualityStudyPlanError(
                f"study requirement {key} must be true"
            )
    if stock.get("requires_real_train_muzero") is not True:
        raise CompactMatchedQualityStudyPlanError(
            "stock reference must require real train_muzero"
        )
    _validate_stock_evaluator_requirement(
        stock.get("stock_evaluator_requirement"),
        eval_seed_count=int(requirements.get("min_eval_seed_count", 0) or 0),
    )
    if compact.get("requires_env_search_replay_rows") is not True:
        raise CompactMatchedQualityStudyPlanError(
            "compact candidate must require env/search/replay rows"
        )
    if compact.get("forbids_train_muzero") is not True:
        raise CompactMatchedQualityStudyPlanError(
            "compact candidate must forbid train_muzero"
        )


def _validate_decision(payload: Mapping[str, Any]) -> None:
    decision = _required_mapping(payload.get("decision"), "decision")
    if decision.get("current_evidence_sufficient_for_promotion") is not False:
        raise CompactMatchedQualityStudyPlanError(
            "larger study plan must not accept current evidence for promotion"
        )
    if decision.get("larger_matched_quality_study_required") is not True:
        raise CompactMatchedQualityStudyPlanError(
            "larger study plan must require larger matched-quality evidence"
        )
    for key in ("promotion_claim", "automatic_promotion_allowed"):
        if decision.get(key) is not False:
            raise CompactMatchedQualityStudyPlanError(
                f"larger study decision {key} must be false"
            )


def _validate_planned_runs(
    payload: Mapping[str, Any],
    *,
    candidate_checkpoint_id: str,
    requirements: Mapping[str, Any],
) -> None:
    planned = _required_mapping(payload.get("planned_runs"), "planned_runs")
    stamp = _require_non_empty(payload.get("study_run_stamp"), "study_run_stamp")
    for key in PLANNED_RUN_KEYS:
        run = _required_mapping(planned.get(key), key)
        run_id = _require_non_empty(run.get("run_id"), f"{key} run_id")
        if stamp not in run_id:
            raise CompactMatchedQualityStudyPlanError(
                f"{key} run_id must include study_run_stamp"
            )
        argv = run.get("argv")
        if not isinstance(argv, Sequence) or isinstance(argv, str) or not argv:
            raise CompactMatchedQualityStudyPlanError(f"{key} argv must be a sequence")
        for arg in argv:
            _require_non_empty(arg, f"{key} argv item")
    stock_argv = list(_required_mapping(planned["stock_reference_capture"], "stock").get("argv"))
    compact_argv = list(
        _required_mapping(planned["compact_candidate_capture"], "compact").get("argv")
    )
    _require_argv_pair(stock_argv, "--candidate-checkpoint-id", candidate_checkpoint_id)
    _require_argv_pair(compact_argv, "--candidate-checkpoint-id", candidate_checkpoint_id)
    _require_argv_pair(
        stock_argv,
        "--eval-seed-count",
        str(requirements["min_eval_seed_count"]),
    )
    _require_argv_pair(
        compact_argv,
        "--eval-seed-count",
        str(requirements["min_eval_seed_count"]),
    )
    _require_argv_pair(
        stock_argv,
        "--eval-steps",
        str(requirements["min_eval_max_steps"]),
    )
    stock_evaluator = _required_mapping(
        _required_mapping(requirements.get("stock_reference"), "stock_reference").get(
            "stock_evaluator_requirement"
        ),
        "stock_evaluator_requirement",
    )
    _require_argv_pair(
        stock_argv,
        "--evaluator-env-num",
        str(stock_evaluator["evaluator_env_num"]),
    )
    _require_argv_pair(
        stock_argv,
        "--n-evaluator-episode",
        str(stock_evaluator["n_evaluator_episode"]),
    )
    _require_argv_pair(
        compact_argv,
        "--eval-steps",
        str(requirements["min_eval_max_steps"]),
    )
    _require_argv_pair(compact_argv, "--compact-training-mode", "env_search_replay")


def _validate_claims(payload: Mapping[str, Any]) -> None:
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    for key in TRUE_PLAN_CLAIM_KEYS:
        if claims.get(key) is not True:
            raise CompactMatchedQualityStudyPlanError(
                f"larger study plan claim {key} missing"
            )
    for key in FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactMatchedQualityStudyPlanError(
                f"larger study plan claim {key} must be false"
            )
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for key in FALSE_CLAIM_KEYS:
        if non_claims.get(key) is not False:
            raise CompactMatchedQualityStudyPlanError(
                f"larger study plan non-claim {key} must be false"
            )


def _require_argv_pair(argv: Sequence[Any], flag: str, expected: str) -> None:
    values = [str(item) for item in argv]
    try:
        index = values.index(flag)
    except ValueError as exc:
        raise CompactMatchedQualityStudyPlanError(f"argv missing {flag}") from exc
    if index + 1 >= len(values) or values[index + 1] != str(expected):
        raise CompactMatchedQualityStudyPlanError(f"argv {flag} mismatch")


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
) -> None:
    requirement = _required_mapping(value, "stock_evaluator_requirement")
    expected = _stock_evaluator_requirement(eval_seed_count=int(eval_seed_count))
    for key, expected_value in expected.items():
        if requirement.get(key) != expected_value:
            raise CompactMatchedQualityStudyPlanError(
                f"stock evaluator requirement {key} mismatch"
            )


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompactMatchedQualityStudyPlanError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompactMatchedQualityStudyPlanError(f"{label} is not JSON") from exc
    if not isinstance(payload, Mapping):
        raise CompactMatchedQualityStudyPlanError(f"{label} must be a mapping")
    return dict(payload)


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactMatchedQualityStudyPlanError(f"{label} must be a mapping")
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactMatchedQualityStudyPlanError(f"{label} must be non-empty")
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
