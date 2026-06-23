"""Matched learning-quality readiness evidence for compact-owned candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any


COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID = (
    "curvyzero_compact_promotion_matched_learning_quality_canary/v1"
)
COMPACT_MATCHED_LEARNING_QUALITY_ARM_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_arm/v1"
)
COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_capture/v1"
)
COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PREVIEW_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_capture_preview/v1"
)
COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PROVENANCE_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_capture_provenance/v1"
)
COMPACT_MATCHED_LEARNING_QUALITY_PAIR_VERIFICATION_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_pair_verification/v1"
)
COMPACT_MATCHED_LEARNING_QUALITY_STOCK_TRAINING_ARTIFACT_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_stock_reference_training_artifact/v1"
)
COMPACT_MATCHED_LEARNING_QUALITY_COMPACT_TRAINING_ARTIFACT_SCHEMA_ID = (
    "curvyzero_compact_matched_learning_quality_compact_candidate_training_artifact/v1"
)
COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID = (
    "curvyzero_compact_coach_compatibility_refresh/v1"
)
COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID = "curvyzero_compact_unified_lifecycle_smoke/v1"
COMPACT_MATCHED_LEARNING_QUALITY_EVIDENCE_REF_PREFIX = (
    "compact_matched_learning_quality:"
)
COMPACT_MATCHED_LEARNING_QUALITY_READINESS_LANE = (
    "matched_stock_vs_compact_learning_quality"
)
COMPACT_MATCHED_LEARNING_QUALITY_STATUS_OBSERVED = (
    "matched_learning_quality_movement_observed"
)

COMPACT_MATCHED_LEARNING_QUALITY_REMAINING_LANES = (
    "stock_resume_load_canary",
    "isolated_live_run_safety_canary",
    "sandbox_assignment_rating_proof",
    "longer_horizon_compact_learning_metrics",
)

STOCK_REFERENCE_ROLE = "stock_reference"
COMPACT_CANDIDATE_ROLE = "compact_candidate"
STOCK_TRAIN_MUZERO_ROUTE = "stock_train_muzero_reference"
COMPACT_OWNED_TRAINER_ROUTE = "compact_owned_trainer"

REQUIRED_MATCHED_EVAL_SETTING_KEYS = (
    "observation_schema_id",
    "policy_observation_backend",
    "eval_seed_set",
    "eval_episode_count",
    "source_max_steps",
    "eval_max_steps",
    "num_simulations",
    "batch_size",
    "reward_variant",
    "reward_target_effect",
    "death_mode",
    "terminal_target_mode",
    "root_noise",
    "dirichlet_alpha",
    "policy_noise",
    "rnd_enabled",
    "exploration_bonus_mode",
    "opponent_policy_ref",
    "opponent_policy_kind",
    "opponent_runtime_mode",
    "opponent_death_mode",
    "natural_bonus_spawn",
    "num_unroll_steps",
    "td_steps",
    "discount",
    "support_scale",
)
REQUIRED_ARM_EVAL_SETTING_KEYS = REQUIRED_MATCHED_EVAL_SETTING_KEYS + (
    "training_seed_policy",
    "initialization_source",
)

NONCLAIM_FALSE_KEYS = (
    "promotion_claim",
    "training_speedup_claim",
    "live_run_safety_claim",
    "stock_resume_claim",
    "stock_training_resume_claim",
    "rating_or_promotion_quality_claim",
    "compact_quality_superiority_claim",
    "leaderboard_claim",
)

PROFILE_ONLY_SPEED_CURRENCIES = frozenset(
    {
        "compact_profile_active_roots_per_sec",
        "compact_profile_physical_rows_per_sec",
        "stock_train_muzero_profile_env_steps_per_sec",
        "local_compact_owned_loop_profile_only",
        "compact_trainer_lifecycle_evidence_no_speed",
        "compact_trainer_checkpoint_no_speed",
    }
)

REQUIRED_CAPTURE_ARTIFACT_KINDS = (
    "training_artifact",
    "pre_eval_summary",
    "post_eval_summary",
    "initial_checkpoint",
    "final_checkpoint",
)

REQUIRED_SOURCE_FINGERPRINT_KEYS = (
    "git_commit",
    "git_status_dirty",
    "producer_script",
    "producer_route",
    "matched_surface",
)

REQUIRED_MATCHED_SURFACE_KEYS = (
    "env_variant",
    "reward_variant",
    "policy_observation_backend",
    "opponent_policy_kind",
    "eval_seed_set",
)

MIN_MATCHED_EVAL_SEED_COUNT = 2
MIN_MATCHED_EVAL_MAX_STEPS = 128
MAX_MATCHED_EVAL_CAP_RATE = 0.5
COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE = "compact_env_search_replay_rows"


class CompactMatchedLearningQualityError(ValueError):
    """Raised when matched learning-quality evidence would overclaim."""


def build_compact_matched_learning_quality_capture_v1(
    *,
    role: str,
    run_id: str,
    capture_id: str,
    candidate_checkpoint_id: str,
    denominator_id: str,
    quality_horizon: str,
    hardware_class: str,
    source_fingerprint: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    eval_settings: Mapping[str, Any],
    pre_eval_summary: Mapping[str, Any],
    post_eval_summary: Mapping[str, Any],
    denominators: Mapping[str, Any],
    artifact_paths: Mapping[str, str | Path],
    capture_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate a raw matched-quality capture from source artifacts."""

    expected_role = _validate_capture_role(role)
    model_identity_out = _plain_mapping(model_identity)
    if "model_state_digest_changed" not in model_identity_out:
        model_identity_out["model_state_digest_changed"] = (
            model_identity_out.get("initial_model_state_digest")
            != model_identity_out.get("final_model_state_digest")
        )
    capture = {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_SCHEMA_ID,
        "ok": True,
        "status": "complete",
        "role": expected_role,
        "route": _route_for_role(expected_role),
        "profile_only": False,
        "touches_live_runs": False,
        "candidate_checkpoint_id": str(candidate_checkpoint_id),
        "capture_id": str(capture_id),
        "run_id": str(run_id),
        "denominator_id": str(denominator_id),
        "quality_horizon": str(quality_horizon),
        "hardware_class": str(hardware_class),
        "source_fingerprint": _plain_mapping(source_fingerprint),
        "model_identity": model_identity_out,
        "eval_settings": _plain_mapping(eval_settings),
        "eval_points": [
            compact_matched_learning_quality_eval_point_from_summary_v1(
                pre_eval_summary,
                point_id="pre_train",
            ),
            compact_matched_learning_quality_eval_point_from_summary_v1(
                post_eval_summary,
                point_id="post_train",
            ),
        ],
        "denominators": _plain_mapping(denominators),
        "artifact_refs": _artifact_refs_from_paths(artifact_paths),
        "capture_provenance": _plain_mapping(capture_provenance),
        "non_claims": matched_learning_quality_non_claims_v1(),
    }
    if expected_role == STOCK_REFERENCE_ROLE:
        capture["called_train_muzero"] = True
    else:
        capture["calls_train_muzero"] = False
        capture["real_compact_owned_training_work"] = True
    compact_matched_learning_quality_arm_from_capture_v1(
        capture,
        expected_role=expected_role,
    )
    return capture


def compact_matched_learning_quality_eval_point_from_summary_v1(
    summary: Mapping[str, Any],
    *,
    point_id: str,
    checkpoint_ref: str | None = None,
    checkpoint_step: int | None = None,
    evaluator_eval_calls: int | None = None,
) -> dict[str, Any]:
    """Normalize a stock/compact eval summary row into a quality-curve point."""

    source = _plain_mapping(summary)
    row = _select_eval_summary_row(source, checkpoint_ref=checkpoint_ref)
    ref = _require_non_empty(
        checkpoint_ref
        or row.get("checkpoint_ref")
        or row.get("checkpoint")
        or row.get("checkpoint_label")
        or row.get("checkpoint_path")
        or source.get("checkpoint_ref")
        or source.get("checkpoint")
        or source.get("checkpoint_label")
        or source.get("checkpoint_path"),
        f"{point_id} checkpoint_ref",
    )
    step = (
        int(checkpoint_step)
        if checkpoint_step is not None
        else _checkpoint_step_from_summary(row, fallback_ref=ref)
    )
    eval_episode_count = int(
        _first_present_number(
            row,
            (
                "eval_episode_count",
                "episode_count",
                "episodes",
                "seeds",
                "ok_count",
            ),
            label=f"{point_id} eval_episode_count",
        )
    )
    eval_calls = int(
        evaluator_eval_calls
        if evaluator_eval_calls is not None
        else row.get("evaluator_eval_calls", row.get("eval_call_count", 1))
    )
    if eval_calls <= 0:
        raise CompactMatchedLearningQualityError(
            f"{point_id} evaluator_eval_calls must be > 0"
        )
    mean = _survival_number(row, ("mean_survival", "mean_steps", "steps_survived"))
    median = _survival_number(
        row,
        ("median_survival", "median_steps", "steps_survived"),
    )
    minimum = _survival_number(row, ("min_survival", "min_steps", "steps_survived"))
    maximum = _survival_number(row, ("max_survival", "max_steps", "steps_survived"))
    cap_rate = _rate_from_summary(
        row,
        direct_keys=("cap_rate",),
        count_keys=("capped_count", "cap_count"),
        histogram_keys=("cap", "capped"),
        denominator=eval_episode_count,
    )
    terminal_rate = _terminal_rate_from_summary(
        row,
        denominator=eval_episode_count,
        cap_rate=cap_rate,
    )
    point = {
        "point_id": str(point_id),
        "checkpoint_ref": ref,
        "checkpoint_step": step,
        "eval_episode_count": eval_episode_count,
        "evaluator_eval_calls": eval_calls,
        "mean_survival": mean,
        "median_survival": median,
        "min_survival": minimum,
        "max_survival": maximum,
        "terminal_rate": terminal_rate,
        "cap_rate": cap_rate,
    }
    death_rate = _optional_death_rate_from_summary(
        row,
        denominator=eval_episode_count,
    )
    if death_rate is not None:
        point["death_rate"] = death_rate
    return point


def matched_learning_quality_non_claims_v1() -> dict[str, bool]:
    """Return the standard non-claims for matched quality captures."""

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


def build_compact_matched_learning_quality_canary_from_captures_v1(
    *,
    run_id: str,
    compatibility_report_path: str | Path,
    unified_lifecycle_report_path: str | Path,
    stock_reference_capture: Mapping[str, Any],
    compact_candidate_capture: Mapping[str, Any],
    input_capture_files: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a canary from raw arm captures, deriving arm evidence objects."""

    stock_arm = compact_matched_learning_quality_arm_from_capture_v1(
        stock_reference_capture,
        expected_role=STOCK_REFERENCE_ROLE,
    )
    compact_arm = compact_matched_learning_quality_arm_from_capture_v1(
        compact_candidate_capture,
        expected_role=COMPACT_CANDIDATE_ROLE,
    )
    return build_compact_matched_learning_quality_canary_v1(
        run_id=run_id,
        compatibility_report_path=compatibility_report_path,
        unified_lifecycle_report_path=unified_lifecycle_report_path,
        stock_reference_arm=stock_arm,
        compact_candidate_arm=compact_arm,
        input_capture_files=input_capture_files,
        created_at=created_at,
    )


def compact_matched_learning_quality_arm_from_capture_v1(
    capture: Mapping[str, Any],
    *,
    expected_role: str,
) -> dict[str, Any]:
    """Derive a validator arm from a raw matched-quality capture."""

    cap = _plain_mapping(capture)
    _validate_quality_capture_common(cap, expected_role=expected_role)
    model_identity = _required_mapping(
        cap.get("model_identity"),
        f"{expected_role} model_identity",
    )
    non_claims = _plain_mapping(
        _required_mapping(cap.get("non_claims"), f"{expected_role} non_claims")
    )
    arm = {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_ARM_SCHEMA_ID,
        "ok": True,
        "status": "complete",
        "role": expected_role,
        "route": cap["route"],
        "row_purpose": "matched_learning_quality_canary",
        "profile_only": False,
        "quality_currency": "mean_survival_delta",
        "calls_train_muzero": _capture_calls_train_muzero(cap, role=expected_role),
        "touches_live_runs": False,
        "candidate_checkpoint_id": cap["candidate_checkpoint_id"],
        "arm_id": cap["capture_id"],
        "run_id": cap["run_id"],
        "denominator_id": cap["denominator_id"],
        "quality_horizon": cap["quality_horizon"],
        "hardware_class": cap["hardware_class"],
        "source_fingerprint": _plain_mapping(
            _required_mapping(cap.get("source_fingerprint"), "source_fingerprint")
        ),
        "initial_model_state_digest": model_identity["initial_model_state_digest"],
        "final_model_state_digest": model_identity["final_model_state_digest"],
        "model_state_digest_changed": model_identity["model_state_digest_changed"],
        "eval_settings": _plain_mapping(
            _required_mapping(cap.get("eval_settings"), "eval_settings")
        ),
        "quality_curve": [
            _plain_mapping(point)
            for point in _required_sequence(cap.get("eval_points"), "eval_points")
        ],
        "denominators": _plain_mapping(
            _required_mapping(cap.get("denominators"), "denominators")
        ),
        "artifact_refs": [
            _plain_mapping(ref)
            for ref in _required_sequence(cap.get("artifact_refs"), "artifact_refs")
        ],
        "non_claims": non_claims,
        **non_claims,
    }
    if expected_role == COMPACT_CANDIDATE_ROLE:
        arm["real_compact_owned_training_work"] = cap["real_compact_owned_training_work"]
        arm["model_identity_scope"] = model_identity["model_identity_scope"]
    _validate_quality_arm(
        arm,
        expected_role=expected_role,
        expected_route=(
            STOCK_TRAIN_MUZERO_ROUTE
            if expected_role == STOCK_REFERENCE_ROLE
            else COMPACT_OWNED_TRAINER_ROUTE
        ),
        expected_calls_train_muzero=(expected_role == STOCK_REFERENCE_ROLE),
        expected_candidate_checkpoint_id=str(cap["candidate_checkpoint_id"]),
    )
    return arm


def build_compact_matched_learning_quality_canary_v1(
    *,
    run_id: str,
    compatibility_report_path: str | Path,
    unified_lifecycle_report_path: str | Path,
    stock_reference_arm: Mapping[str, Any],
    compact_candidate_arm: Mapping[str, Any],
    input_capture_files: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a two-arm quality-movement canary from already observed arm data."""

    compatibility_path = Path(compatibility_report_path).resolve()
    lifecycle_path = Path(unified_lifecycle_report_path).resolve()
    compatibility = _read_json_mapping(compatibility_path, "compatibility report")
    lifecycle = _read_json_mapping(lifecycle_path, "unified lifecycle report")
    _validate_compatibility_input(compatibility)
    candidate_checkpoint_id = _validate_lifecycle_input(lifecycle, compatibility)

    stock_arm = _plain_mapping(stock_reference_arm)
    compact_arm = _plain_mapping(compact_candidate_arm)
    _validate_quality_arm(
        stock_arm,
        expected_role=STOCK_REFERENCE_ROLE,
        expected_route=STOCK_TRAIN_MUZERO_ROUTE,
        expected_calls_train_muzero=True,
        expected_candidate_checkpoint_id=candidate_checkpoint_id,
    )
    _validate_quality_arm(
        compact_arm,
        expected_role=COMPACT_CANDIDATE_ROLE,
        expected_route=COMPACT_OWNED_TRAINER_ROUTE,
        expected_calls_train_muzero=False,
        expected_candidate_checkpoint_id=candidate_checkpoint_id,
    )
    _validate_matched_pair(stock_arm, compact_arm)

    stock_delta = _mean_survival_delta(stock_arm)
    compact_delta = _mean_survival_delta(compact_arm)
    quality_movement = {
        "metric": "mean_survival",
        "quality_currency": "mean_survival_delta",
        "stock_reference_delta": stock_delta,
        "compact_candidate_delta": compact_delta,
        "compact_minus_stock_delta": compact_delta - stock_delta,
        "stock_reference_initial_mean_survival": _quality_curve(stock_arm)[0][
            "mean_survival"
        ],
        "stock_reference_final_mean_survival": _quality_curve(stock_arm)[-1][
            "mean_survival"
        ],
        "compact_candidate_initial_mean_survival": _quality_curve(compact_arm)[0][
            "mean_survival"
        ],
        "compact_candidate_final_mean_survival": _quality_curve(compact_arm)[-1][
            "mean_survival"
        ],
        "movement_thresholds": {
            "min_abs_primary_delta": 0.0,
            "allowed_compact_regression": 0.0,
        },
        "stock_movement_observed": not math.isclose(stock_delta, 0.0, abs_tol=0.0),
        "compact_movement_observed": not math.isclose(
            compact_delta,
            0.0,
            abs_tol=0.0,
        ),
        "matched_quality_canary": True,
    }
    denominator_id = str(stock_arm["denominator_id"])
    payload = {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID,
        "ok": True,
        "status": COMPACT_MATCHED_LEARNING_QUALITY_STATUS_OBSERVED,
        "run_id": str(run_id),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "readiness_lane": COMPACT_MATCHED_LEARNING_QUALITY_READINESS_LANE,
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "compatibility_report_path": str(compatibility_path),
        "compatibility_report_sha256": _file_sha256(compatibility_path),
        "unified_lifecycle_report_path": str(lifecycle_path),
        "unified_lifecycle_report_sha256": _file_sha256(lifecycle_path),
        "input_compatibility": {
            "promotion_eligible": bool(compatibility["promotion_eligible"]),
            "promotion_claim": bool(compatibility["promotion_claim"]),
            "calls_train_muzero": bool(compatibility["calls_train_muzero"]),
            "touches_live_runs": bool(compatibility["touches_live_runs"]),
            "coach_speed_row_gate": bool(compatibility.get("coach_speed_row_gate")),
        },
        "matched_pair": {
            "denominator_id": denominator_id,
            "quality_horizon": str(stock_arm["quality_horizon"]),
            "hardware_class": (
                str(stock_arm["hardware_class"])
                if stock_arm.get("hardware_class") == compact_arm.get("hardware_class")
                else "mixed"
            ),
            "stock_hardware_class": str(stock_arm["hardware_class"]),
            "compact_hardware_class": str(compact_arm["hardware_class"]),
            "source_fingerprint_sha256": _json_sha256(
                _matched_source_surface(stock_arm)
            ),
            "stock_source_fingerprint_sha256": _json_sha256(
                stock_arm["source_fingerprint"]
            ),
            "compact_source_fingerprint_sha256": _json_sha256(
                compact_arm["source_fingerprint"]
            ),
            "matched_eval_setting_keys": list(REQUIRED_MATCHED_EVAL_SETTING_KEYS),
            "eval_settings_sha256": _json_sha256(
                _matched_eval_settings_subset(
                    _required_mapping(stock_arm["eval_settings"], "eval_settings")
                )
            ),
        },
        "arms": {
            STOCK_REFERENCE_ROLE: stock_arm,
            COMPACT_CANDIDATE_ROLE: compact_arm,
        },
        "input_capture_files": _plain_mapping(input_capture_files),
        "arm_call_contract": {
            "stock_reference_calls_train_muzero": True,
            "compact_candidate_calls_train_muzero": False,
            "both_touch_live_runs": False,
        },
        "quality_movement": quality_movement,
        "readiness": {
            "matched_learning_quality_canary": True,
            "promotion_readiness_complete": False,
            "other_required_lanes_not_proven_by_this_artifact": list(
                COMPACT_MATCHED_LEARNING_QUALITY_REMAINING_LANES
            ),
        },
        "attached_claims": {
            "matched_learning_quality_canary": True,
            "quality_movement_observed_under_fixed_eval_settings": True,
            "promotion_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "stock_resume_claim": False,
            "stock_training_resume_claim": False,
            "rating_or_promotion_quality_claim": False,
            "compact_quality_superiority_claim": False,
            "leaderboard_claim": False,
            "touches_live_runs": False,
            "compact_calls_train_muzero": False,
        },
        "non_claims": {
            "promotion_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "stock_resume_claim": False,
            "stock_training_resume_claim": False,
            "rating_or_promotion_quality_claim": False,
            "compact_quality_superiority_claim": False,
            "leaderboard_claim": False,
            "touches_live_runs": False,
            "compact_calls_train_muzero": False,
        },
    }
    payload["evidence_ref"] = compact_matched_learning_quality_evidence_ref(payload)
    validate_compact_matched_learning_quality_canary_v1(payload)
    return payload


def validate_compact_matched_learning_quality_canary_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate a matched learning-quality canary and referenced artifacts."""

    if payload.get("schema_id") != COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID:
        raise CompactMatchedLearningQualityError("matched quality schema mismatch")
    if payload.get("ok") is not True:
        raise CompactMatchedLearningQualityError("matched quality canary must be ok=true")
    if payload.get("status") != COMPACT_MATCHED_LEARNING_QUALITY_STATUS_OBSERVED:
        raise CompactMatchedLearningQualityError("matched quality status mismatch")
    if payload.get("readiness_lane") != COMPACT_MATCHED_LEARNING_QUALITY_READINESS_LANE:
        raise CompactMatchedLearningQualityError("matched quality readiness lane mismatch")
    candidate_checkpoint_id = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    _validate_payload_file(payload, "compatibility_report")
    _validate_payload_file(payload, "unified_lifecycle_report")
    input_compatibility = _required_mapping(
        payload.get("input_compatibility"),
        "input_compatibility",
    )
    if input_compatibility.get("promotion_eligible") is not True:
        raise CompactMatchedLearningQualityError(
            "matched quality input must be compatibility eligible"
        )
    if input_compatibility.get("coach_speed_row_gate") is not True:
        raise CompactMatchedLearningQualityError("matched quality input speed row missing")
    for key in ("promotion_claim", "calls_train_muzero", "touches_live_runs"):
        if input_compatibility.get(key) is not False:
            raise CompactMatchedLearningQualityError(
                f"matched quality input {key} must be false"
            )

    arms = _required_mapping(payload.get("arms"), "arms")
    stock_arm = _required_mapping(arms.get(STOCK_REFERENCE_ROLE), STOCK_REFERENCE_ROLE)
    compact_arm = _required_mapping(arms.get(COMPACT_CANDIDATE_ROLE), COMPACT_CANDIDATE_ROLE)
    _validate_quality_arm(
        stock_arm,
        expected_role=STOCK_REFERENCE_ROLE,
        expected_route=STOCK_TRAIN_MUZERO_ROUTE,
        expected_calls_train_muzero=True,
        expected_candidate_checkpoint_id=candidate_checkpoint_id,
    )
    _validate_quality_arm(
        compact_arm,
        expected_role=COMPACT_CANDIDATE_ROLE,
        expected_route=COMPACT_OWNED_TRAINER_ROUTE,
        expected_calls_train_muzero=False,
        expected_candidate_checkpoint_id=candidate_checkpoint_id,
    )
    _validate_input_capture_files(payload, stock_arm=stock_arm, compact_arm=compact_arm)
    _validate_matched_pair(stock_arm, compact_arm)
    _validate_matched_pair_summary(payload, stock_arm, compact_arm)

    call_contract = _required_mapping(
        payload.get("arm_call_contract"),
        "arm_call_contract",
    )
    if call_contract.get("stock_reference_calls_train_muzero") is not True:
        raise CompactMatchedLearningQualityError("stock arm must call train_muzero")
    if call_contract.get("compact_candidate_calls_train_muzero") is not False:
        raise CompactMatchedLearningQualityError("compact arm must not call train_muzero")
    if call_contract.get("both_touch_live_runs") is not False:
        raise CompactMatchedLearningQualityError("matched quality must not touch live runs")

    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    for key in (
        "matched_learning_quality_canary",
        "quality_movement_observed_under_fixed_eval_settings",
    ):
        if claims.get(key) is not True:
            raise CompactMatchedLearningQualityError(f"matched quality {key} not true")
    for key in (*NONCLAIM_FALSE_KEYS, "touches_live_runs", "compact_calls_train_muzero"):
        if claims.get(key) is not False:
            raise CompactMatchedLearningQualityError(
                f"matched quality claim {key} must be false"
            )
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for key in (*NONCLAIM_FALSE_KEYS, "touches_live_runs", "compact_calls_train_muzero"):
        if non_claims.get(key) is not False:
            raise CompactMatchedLearningQualityError(
                f"matched quality non-claim {key} must be false"
            )

    readiness = _required_mapping(payload.get("readiness"), "readiness")
    if readiness.get("matched_learning_quality_canary") is not True:
        raise CompactMatchedLearningQualityError("matched quality readiness missing")
    if readiness.get("promotion_readiness_complete") is not False:
        raise CompactMatchedLearningQualityError(
            "matched quality must not complete promotion readiness"
        )
    evidence_ref = str(payload.get("evidence_ref", "")).strip()
    if evidence_ref != compact_matched_learning_quality_evidence_ref(payload):
        raise CompactMatchedLearningQualityError("matched quality evidence_ref mismatch")


def build_compact_matched_learning_quality_pair_verification_v1(
    *,
    matched_learning_quality_report_path: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a focused reciprocal verifier for an existing matched canary report."""

    report_path = Path(matched_learning_quality_report_path).resolve()
    payload = _read_json_mapping(report_path, "matched learning-quality report")
    validate_compact_matched_learning_quality_canary_v1(payload)

    arms = _required_mapping(payload.get("arms"), "arms")
    stock_arm = _required_mapping(arms.get(STOCK_REFERENCE_ROLE), STOCK_REFERENCE_ROLE)
    compact_arm = _required_mapping(arms.get(COMPACT_CANDIDATE_ROLE), COMPACT_CANDIDATE_ROLE)
    stock_denominators = _required_mapping(
        stock_arm.get("denominators"),
        "stock_reference denominators",
    )
    compact_denominators = _required_mapping(
        compact_arm.get("denominators"),
        "compact_candidate denominators",
    )

    stock_denominator_checks = _stock_denominator_count_check_report(
        stock_denominators
    )
    compact_denominator_checks = _compact_denominator_count_check_report(
        compact_denominators
    )

    checks = {
        "canary_report_validates": True,
        "role_pair_complete": True,
        "same_denominator_id": stock_arm.get("denominator_id")
        == compact_arm.get("denominator_id"),
        "same_quality_horizon": stock_arm.get("quality_horizon")
        == compact_arm.get("quality_horizon"),
        "stock_calls_train_muzero": stock_arm.get("calls_train_muzero") is True,
        "compact_does_not_call_train_muzero": (
            compact_arm.get("calls_train_muzero") is False
        ),
        "non_claims_false": _all_non_claims_false(payload.get("non_claims")),
        "stock_denominator_counts_coherent": True,
        "compact_denominator_counts_coherent": True,
        "input_capture_hashes_bound": True,
        "matched_pair_hashes_bound": True,
    }
    for key, value in checks.items():
        if value is not True:
            raise CompactMatchedLearningQualityError(
                f"matched quality pair verifier check failed: {key}"
            )

    matched_pair = _required_mapping(payload.get("matched_pair"), "matched_pair")
    verification = {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_PAIR_VERIFICATION_SCHEMA_ID,
        "ok": True,
        "status": "matched_pair_verified",
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "run_id": str(payload.get("run_id")),
        "candidate_checkpoint_id": str(payload.get("candidate_checkpoint_id")),
        "readiness_lane": COMPACT_MATCHED_LEARNING_QUALITY_READINESS_LANE,
        "matched_learning_quality_report": {
            "path": str(report_path),
            "sha256": _file_sha256(report_path),
            "schema_id": payload.get("schema_id"),
            "evidence_ref": payload.get("evidence_ref"),
        },
        "matched_pair": _plain_mapping(matched_pair),
        "quality_movement": _plain_mapping(
            _required_mapping(payload.get("quality_movement"), "quality_movement")
        ),
        "reciprocal_roles": _matched_pair_reciprocal_roles(stock_arm, compact_arm),
        "denominator_count_checks": {
            "stock_reference": stock_denominator_checks,
            "compact_candidate": compact_denominator_checks,
        },
        "fingerprint_inventory": _matched_pair_fingerprint_inventory(
            stock_arm,
            compact_arm,
        ),
        "input_capture_files": _plain_mapping(
            _required_mapping(payload.get("input_capture_files"), "input_capture_files")
        ),
        "non_claims": _plain_mapping(
            _required_mapping(payload.get("non_claims"), "non_claims")
        ),
    }
    validate_compact_matched_learning_quality_pair_verification_v1(verification)
    return verification


def validate_compact_matched_learning_quality_pair_verification_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate the reciprocal matched-pair verification report."""

    if (
        payload.get("schema_id")
        != COMPACT_MATCHED_LEARNING_QUALITY_PAIR_VERIFICATION_SCHEMA_ID
    ):
        raise CompactMatchedLearningQualityError(
            "matched quality pair verification schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactMatchedLearningQualityError(
            "matched quality pair verification must be ok=true"
        )
    if payload.get("status") != "matched_pair_verified":
        raise CompactMatchedLearningQualityError(
            "matched quality pair verification status mismatch"
        )
    report = _required_mapping(
        payload.get("matched_learning_quality_report"),
        "matched_learning_quality_report",
    )
    report_path = Path(str(report.get("path", "")))
    if not report_path.is_file():
        raise CompactMatchedLearningQualityError("matched pair report path missing")
    if _file_sha256(report_path) != report.get("sha256"):
        raise CompactMatchedLearningQualityError("matched pair report sha256 mismatch")
    canary = _read_json_mapping(report_path, "matched learning-quality report")
    validate_compact_matched_learning_quality_canary_v1(canary)
    if report.get("evidence_ref") != canary.get("evidence_ref"):
        raise CompactMatchedLearningQualityError("matched pair evidence_ref mismatch")
    if payload.get("run_id") != canary.get("run_id"):
        raise CompactMatchedLearningQualityError("matched pair run_id mismatch")
    if payload.get("candidate_checkpoint_id") != canary.get("candidate_checkpoint_id"):
        raise CompactMatchedLearningQualityError(
            "matched pair candidate_checkpoint_id mismatch"
        )
    if _json_sha256(payload.get("matched_pair")) != _json_sha256(canary["matched_pair"]):
        raise CompactMatchedLearningQualityError("matched pair summary drift")
    if _json_sha256(payload.get("quality_movement")) != _json_sha256(
        canary["quality_movement"]
    ):
        raise CompactMatchedLearningQualityError("matched pair movement drift")
    if payload.get("non_claims") != canary.get("non_claims"):
        raise CompactMatchedLearningQualityError("matched pair non_claims drift")
    if not _all_non_claims_false(payload.get("non_claims")):
        raise CompactMatchedLearningQualityError("matched pair non_claims flipped")
    if payload.get("readiness_lane") != COMPACT_MATCHED_LEARNING_QUALITY_READINESS_LANE:
        raise CompactMatchedLearningQualityError("matched pair readiness lane mismatch")

    arms = _required_mapping(canary.get("arms"), "canary arms")
    stock_arm = _required_mapping(arms.get(STOCK_REFERENCE_ROLE), STOCK_REFERENCE_ROLE)
    compact_arm = _required_mapping(arms.get(COMPACT_CANDIDATE_ROLE), COMPACT_CANDIDATE_ROLE)
    expected_roles = _matched_pair_reciprocal_roles(stock_arm, compact_arm)
    if _json_sha256(payload.get("reciprocal_roles")) != _json_sha256(expected_roles):
        raise CompactMatchedLearningQualityError("matched pair reciprocal role drift")

    checks = _required_mapping(
        payload.get("denominator_count_checks"),
        "denominator_count_checks",
    )
    expected_stock_checks = _stock_denominator_count_check_report(
        _required_mapping(stock_arm.get("denominators"), "stock_reference denominators")
    )
    expected_compact_checks = _compact_denominator_count_check_report(
        _required_mapping(compact_arm.get("denominators"), "compact_candidate denominators")
    )
    if _json_sha256(checks.get(STOCK_REFERENCE_ROLE)) != _json_sha256(
        expected_stock_checks
    ):
        raise CompactMatchedLearningQualityError(
            "matched pair stock denominator check drift"
        )
    if _json_sha256(checks.get(COMPACT_CANDIDATE_ROLE)) != _json_sha256(
        expected_compact_checks
    ):
        raise CompactMatchedLearningQualityError(
            "matched pair compact denominator check drift"
        )

    expected_fingerprints = _matched_pair_fingerprint_inventory(stock_arm, compact_arm)
    if _json_sha256(payload.get("fingerprint_inventory")) != _json_sha256(
        expected_fingerprints
    ):
        raise CompactMatchedLearningQualityError("matched pair fingerprint drift")
    if _json_sha256(payload.get("input_capture_files")) != _json_sha256(
        canary.get("input_capture_files")
    ):
        raise CompactMatchedLearningQualityError("matched pair input capture drift")


def compact_matched_learning_quality_evidence_ref(
    payload: Mapping[str, Any],
) -> str:
    """Return the compact Coach evidence ref for a validated quality canary."""

    candidate = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    matched_pair = _required_mapping(payload.get("matched_pair"), "matched_pair")
    denominator_id = _require_non_empty(
        matched_pair.get("denominator_id"),
        "denominator_id",
    )
    movement = _required_mapping(payload.get("quality_movement"), "quality_movement")
    return (
        f"{COMPACT_MATCHED_LEARNING_QUALITY_EVIDENCE_REF_PREFIX}"
        f"{candidate}:{denominator_id}:{_json_sha256(movement)[:16]}"
    )


def _validate_compatibility_input(compatibility: Mapping[str, Any]) -> None:
    if compatibility.get("schema_id") != COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID:
        raise CompactMatchedLearningQualityError("compatibility refresh schema mismatch")
    if compatibility.get("ok") is not True:
        raise CompactMatchedLearningQualityError("compatibility refresh must be ok=true")
    if compatibility.get("promotion_eligible") is not True:
        raise CompactMatchedLearningQualityError(
            "matched quality requires local compatibility eligibility"
        )
    if compatibility.get("coach_speed_row_gate") is not True:
        raise CompactMatchedLearningQualityError("compatibility speed row gate missing")
    for key in ("promotion_claim", "calls_train_muzero", "touches_live_runs"):
        if compatibility.get(key) is not False:
            raise CompactMatchedLearningQualityError(
                f"compatibility {key} must be false"
            )
    _require_non_empty(compatibility.get("candidate_checkpoint_id"), "candidate_checkpoint_id")


def _validate_quality_capture_common(
    capture: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    if capture.get("schema_id") != COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_SCHEMA_ID:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture schema mismatch"
        )
    if capture.get("ok") is not True:
        raise CompactMatchedLearningQualityError(f"{expected_role} capture must be ok=true")
    if capture.get("status") != "complete":
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture status must be complete"
        )
    if capture.get("role") != expected_role:
        raise CompactMatchedLearningQualityError(f"{expected_role} capture role mismatch")
    if capture.get("profile_only") is not False:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture requires profile_only=false"
        )
    if capture.get("touches_live_runs") is not False:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture must not touch live runs"
        )
    expected_route = (
        STOCK_TRAIN_MUZERO_ROUTE
        if expected_role == STOCK_REFERENCE_ROLE
        else COMPACT_OWNED_TRAINER_ROUTE
    )
    if capture.get("route") != expected_route:
        raise CompactMatchedLearningQualityError(f"{expected_role} capture route mismatch")
    _validate_capture_provenance(capture, expected_role=expected_role)
    artifact_refs_by_kind = _validate_capture_artifact_binding(
        capture,
        expected_role=expected_role,
    )
    _validate_capture_artifact_contents(
        capture,
        expected_role=expected_role,
        artifact_refs_by_kind=artifact_refs_by_kind,
    )
    for key in (
        "capture_id",
        "run_id",
        "candidate_checkpoint_id",
        "denominator_id",
        "quality_horizon",
        "hardware_class",
    ):
        _require_non_empty(capture.get(key), f"{expected_role} capture {key}")
    if expected_role == STOCK_REFERENCE_ROLE:
        if capture.get("called_train_muzero") is not True:
            raise CompactMatchedLearningQualityError(
                "stock_reference capture must call train_muzero"
            )
    else:
        if capture.get("calls_train_muzero") is not False:
            raise CompactMatchedLearningQualityError(
                "compact_candidate capture calls_train_muzero must be false"
            )
        if capture.get("real_compact_owned_training_work") is not True:
            raise CompactMatchedLearningQualityError(
                "compact_candidate capture real compact-owned training work missing"
            )


def _validate_capture_provenance(
    capture: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    provenance = _required_mapping(
        capture.get("capture_provenance"),
        f"{expected_role} capture_provenance",
    )
    if provenance.get("schema_id") != COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_PROVENANCE_SCHEMA_ID:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture_provenance schema mismatch"
        )
    if provenance.get("role") != expected_role:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture_provenance role mismatch"
        )
    if provenance.get("producer_ran_training") is not True:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture requires producer_ran_training=true"
        )
    if provenance.get("producer_ran_pre_eval") is not True:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture requires producer_ran_pre_eval=true"
        )
    if provenance.get("producer_ran_post_eval") is not True:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture requires producer_ran_post_eval=true"
        )
    if provenance.get("feeds_builder") is not True:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture_provenance feeds_builder must be true"
        )
    if provenance.get("support_only") is not False:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture_provenance support_only must be false"
        )
    for key in (
        "producer_id",
        "training_artifact_ref",
        "pre_eval_artifact_ref",
        "post_eval_artifact_ref",
    ):
        _require_non_empty(provenance.get(key), f"{expected_role} capture_provenance {key}")
    if expected_role == STOCK_REFERENCE_ROLE:
        stock = _required_mapping(
            provenance.get("stock_train_muzero"),
            "stock_train_muzero provenance",
        )
        if stock.get("called_train_muzero") is not True:
            raise CompactMatchedLearningQualityError(
                "stock_reference capture provenance must call train_muzero"
            )
        _require_non_empty(
            stock.get("train_muzero_entrypoint"),
            "stock_train_muzero train_muzero_entrypoint",
        )
    else:
        compact = _required_mapping(
            provenance.get("compact_owned_training"),
            "compact_owned_training provenance",
        )
        if compact.get("calls_train_muzero") is not False:
            raise CompactMatchedLearningQualityError(
                "compact_candidate capture provenance calls_train_muzero must be false"
            )
        if compact.get("real_compact_owned_training_work") is not True:
            raise CompactMatchedLearningQualityError(
                "compact_candidate capture provenance real compact-owned work missing"
            )
        _require_non_empty(
            compact.get("compact_training_entrypoint"),
            "compact_owned_training compact_training_entrypoint",
        )


def _validate_capture_artifact_binding(
    capture: Mapping[str, Any],
    *,
    expected_role: str,
) -> dict[str, Mapping[str, Any]]:
    refs = _required_sequence(
        capture.get("artifact_refs"),
        f"{expected_role} artifact_refs",
    )
    by_kind: dict[str, Mapping[str, Any]] = {}
    for index, ref_obj in enumerate(refs):
        ref = _required_mapping(ref_obj, f"{expected_role} artifact_refs[{index}]")
        kind = _require_non_empty(
            ref.get("kind"),
            f"{expected_role} artifact_refs[{index}] kind",
        )
        if kind in by_kind:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} duplicate artifact_ref kind {kind}"
            )
        by_kind[kind] = ref

    for kind in REQUIRED_CAPTURE_ARTIFACT_KINDS:
        if kind not in by_kind:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} capture artifact_refs missing {kind}"
            )

    provenance = _required_mapping(
        capture.get("capture_provenance"),
        f"{expected_role} capture_provenance",
    )
    _validate_provenance_ref_matches_artifact(
        provenance.get("training_artifact_ref"),
        by_kind["training_artifact"],
        label=f"{expected_role} capture_provenance training_artifact_ref",
    )
    _validate_provenance_ref_matches_artifact(
        provenance.get("pre_eval_artifact_ref"),
        by_kind["pre_eval_summary"],
        label=f"{expected_role} capture_provenance pre_eval_artifact_ref",
    )
    _validate_provenance_ref_matches_artifact(
        provenance.get("post_eval_artifact_ref"),
        by_kind["post_eval_summary"],
        label=f"{expected_role} capture_provenance post_eval_artifact_ref",
    )
    return by_kind


def _validate_capture_artifact_contents(
    capture: Mapping[str, Any],
    *,
    expected_role: str,
    artifact_refs_by_kind: Mapping[str, Mapping[str, Any]],
) -> None:
    training_artifact = _read_artifact_json(
        artifact_refs_by_kind["training_artifact"],
        label=f"{expected_role} training_artifact",
    )
    _validate_training_artifact_content(
        training_artifact,
        capture=capture,
        expected_role=expected_role,
        artifact_refs_by_kind=artifact_refs_by_kind,
    )
    _validate_eval_summary_artifact_content(
        _read_artifact_json(
            artifact_refs_by_kind["pre_eval_summary"],
            label=f"{expected_role} pre_eval_summary",
        ),
        capture=capture,
        expected_role=expected_role,
        point_id="pre_train",
        point_index=0,
    )
    _validate_eval_summary_artifact_content(
        _read_artifact_json(
            artifact_refs_by_kind["post_eval_summary"],
            label=f"{expected_role} post_eval_summary",
        ),
        capture=capture,
        expected_role=expected_role,
        point_id="post_train",
        point_index=-1,
    )
    initial_ref = artifact_refs_by_kind["initial_checkpoint"]
    final_ref = artifact_refs_by_kind["final_checkpoint"]
    initial_sha = _require_non_empty(
        initial_ref.get("sha256"),
        f"{expected_role} initial_checkpoint sha256",
    )
    final_sha = _require_non_empty(
        final_ref.get("sha256"),
        f"{expected_role} final_checkpoint sha256",
    )
    if initial_sha == final_sha:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} initial/final checkpoint artifacts must differ"
        )


def _validate_training_artifact_content(
    artifact: Mapping[str, Any],
    *,
    capture: Mapping[str, Any],
    expected_role: str,
    artifact_refs_by_kind: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_schema = (
        COMPACT_MATCHED_LEARNING_QUALITY_STOCK_TRAINING_ARTIFACT_SCHEMA_ID
        if expected_role == STOCK_REFERENCE_ROLE
        else COMPACT_MATCHED_LEARNING_QUALITY_COMPACT_TRAINING_ARTIFACT_SCHEMA_ID
    )
    if artifact.get("schema_id") != expected_schema:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} training_artifact schema mismatch"
        )
    if artifact.get("ok") is not True:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} training_artifact must be ok=true"
        )
    if artifact.get("role") != expected_role:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} training_artifact role mismatch"
        )
    if artifact.get("route") != _route_for_role(expected_role):
        raise CompactMatchedLearningQualityError(
            f"{expected_role} training_artifact route mismatch"
        )
    if artifact.get("run_id") != capture.get("run_id"):
        raise CompactMatchedLearningQualityError(
            f"{expected_role} training_artifact run_id mismatch"
        )
    for key, expected in (
        ("profile_only", False),
        ("support_only", False),
        ("touches_live_runs", False),
    ):
        if artifact.get(key) is not expected:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} training_artifact {key} must be {str(expected).lower()}"
            )

    artifact_identity = _required_mapping(
        artifact.get("model_identity"),
        f"{expected_role} training_artifact model_identity",
    )
    capture_identity = _required_mapping(
        capture.get("model_identity"),
        f"{expected_role} capture model_identity",
    )
    for key in (
        "initial_model_state_digest",
        "final_model_state_digest",
        "model_state_digest_changed",
    ):
        if artifact_identity.get(key) != capture_identity.get(key):
            raise CompactMatchedLearningQualityError(
                f"{expected_role} training_artifact model_identity {key} mismatch"
            )

    eval_points = [
        _plain_mapping(point)
        for point in _required_sequence(
            capture.get("eval_points"),
            f"{expected_role} capture eval_points",
        )
    ]
    if len(eval_points) < 2:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} capture requires at least two eval points"
        )
    if expected_role == STOCK_REFERENCE_ROLE:
        _validate_stock_training_artifact_content(
            artifact,
            eval_points=eval_points,
            artifact_refs_by_kind=artifact_refs_by_kind,
        )
    else:
        _validate_compact_training_artifact_content(
            artifact,
            eval_points=eval_points,
            artifact_refs_by_kind=artifact_refs_by_kind,
        )


def _validate_stock_training_artifact_content(
    artifact: Mapping[str, Any],
    *,
    eval_points: Sequence[Mapping[str, Any]],
    artifact_refs_by_kind: Mapping[str, Mapping[str, Any]],
) -> None:
    if artifact.get("called_train_muzero") is not True:
        raise CompactMatchedLearningQualityError(
            "stock_reference training_artifact must call train_muzero"
        )
    _require_non_empty(
        artifact.get("train_muzero_entrypoint"),
        "stock_reference training_artifact train_muzero_entrypoint",
    )
    selection = _required_mapping(
        artifact.get("checkpoint_selection"),
        "stock_reference training_artifact checkpoint_selection",
    )
    if selection.get("initial_ref") != eval_points[0].get("checkpoint_ref"):
        raise CompactMatchedLearningQualityError(
            "stock_reference training_artifact initial_ref mismatch"
        )
    if selection.get("final_ref") != eval_points[-1].get("checkpoint_ref"):
        raise CompactMatchedLearningQualityError(
            "stock_reference training_artifact final_ref mismatch"
        )
    downloaded = _required_mapping(
        artifact.get("downloaded_checkpoints"),
        "stock_reference training_artifact downloaded_checkpoints",
    )
    _validate_checkpoint_artifact_record(
        downloaded,
        ref_path_key="initial_path",
        ref_sha_key="initial_sha256",
        artifact_ref=artifact_refs_by_kind["initial_checkpoint"],
        label="stock_reference training_artifact initial checkpoint",
    )
    _validate_checkpoint_artifact_record(
        downloaded,
        ref_path_key="final_path",
        ref_sha_key="final_sha256",
        artifact_ref=artifact_refs_by_kind["final_checkpoint"],
        label="stock_reference training_artifact final checkpoint",
    )


def _validate_compact_training_artifact_content(
    artifact: Mapping[str, Any],
    *,
    eval_points: Sequence[Mapping[str, Any]],
    artifact_refs_by_kind: Mapping[str, Mapping[str, Any]],
) -> None:
    if artifact.get("calls_train_muzero") is not False:
        raise CompactMatchedLearningQualityError(
            "compact_candidate training_artifact calls_train_muzero must be false"
        )
    if artifact.get("real_compact_owned_training_work") is not True:
        raise CompactMatchedLearningQualityError(
            "compact_candidate training_artifact real compact-owned work missing"
        )
    _require_non_empty(
        artifact.get("compact_training_entrypoint"),
        "compact_candidate training_artifact compact_training_entrypoint",
    )
    entrypoint = str(artifact.get("compact_training_entrypoint") or "")
    if "CompactOwnedTrainerV1.record_step" not in entrypoint:
        raise CompactMatchedLearningQualityError(
            "compact_candidate training_artifact must use CompactOwnedTrainerV1.record_step"
        )
    sample_source = _require_non_empty(
        artifact.get("sample_source"),
        "compact_candidate training_artifact sample_source",
    )
    if sample_source != COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE:
        raise CompactMatchedLearningQualityError(
            "compact_candidate training_artifact sample_source must be "
            f"{COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE}"
        )
    if artifact.get("synthetic_resident_sample") is True:
        raise CompactMatchedLearningQualityError(
            "compact_candidate training_artifact must not use synthetic resident samples"
        )
    if artifact.get("real_env_search_replay_rows") is not True:
        raise CompactMatchedLearningQualityError(
            "compact_candidate training_artifact missing real env/search/replay rows"
        )
    if artifact.get("pre_eval_checkpoint_ref") != eval_points[0].get("checkpoint_ref"):
        raise CompactMatchedLearningQualityError(
            "compact_candidate training_artifact pre_eval_checkpoint_ref mismatch"
        )
    if artifact.get("post_eval_checkpoint_ref") != eval_points[-1].get("checkpoint_ref"):
        raise CompactMatchedLearningQualityError(
            "compact_candidate training_artifact post_eval_checkpoint_ref mismatch"
        )
    for artifact_key, kind in (
        ("initial_checkpoint_path", "initial_checkpoint"),
        ("final_checkpoint_path", "final_checkpoint"),
    ):
        if artifact.get(artifact_key) is not None and not _ref_text_matches(
            artifact.get(artifact_key),
            artifact_refs_by_kind[kind].get("path"),
        ):
            raise CompactMatchedLearningQualityError(
                f"compact_candidate training_artifact {artifact_key} mismatch"
            )


def _validate_checkpoint_artifact_record(
    record: Mapping[str, Any],
    *,
    ref_path_key: str,
    ref_sha_key: str,
    artifact_ref: Mapping[str, Any],
    label: str,
) -> None:
    if not _ref_text_matches(record.get(ref_path_key), artifact_ref.get("path")):
        raise CompactMatchedLearningQualityError(f"{label} path mismatch")
    if record.get(ref_sha_key) != artifact_ref.get("sha256"):
        raise CompactMatchedLearningQualityError(f"{label} sha256 mismatch")


def _validate_eval_summary_artifact_content(
    summary: Mapping[str, Any],
    *,
    capture: Mapping[str, Any],
    expected_role: str,
    point_id: str,
    point_index: int,
) -> None:
    points = [
        _plain_mapping(point)
        for point in _required_sequence(
            capture.get("eval_points"),
            f"{expected_role} capture eval_points",
        )
    ]
    point = points[point_index]
    normalized = compact_matched_learning_quality_eval_point_from_summary_v1(
        summary,
        point_id=point_id,
    )
    for key in (
        "point_id",
        "checkpoint_ref",
        "checkpoint_step",
        "eval_episode_count",
        "evaluator_eval_calls",
    ):
        if normalized.get(key) != point.get(key):
            raise CompactMatchedLearningQualityError(
                f"{expected_role} {point_id} eval summary {key} mismatch"
            )
    for key in (
        "mean_survival",
        "median_survival",
        "min_survival",
        "max_survival",
        "terminal_rate",
        "cap_rate",
    ):
        if not math.isclose(
            float(normalized[key]),
            float(point[key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CompactMatchedLearningQualityError(
                f"{expected_role} {point_id} eval summary {key} mismatch"
            )
    settings = _required_mapping(
        capture.get("eval_settings"),
        f"{expected_role} capture eval_settings",
    )
    _validate_eval_summary_settings(
        summary,
        settings=settings,
        label=f"{expected_role} {point_id} eval summary",
    )


def _validate_eval_summary_settings(
    summary: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    label: str,
) -> None:
    observed_seed_set = sorted(_eval_summary_seed_set(summary))
    expected_seed_set = sorted(
        _eval_seed_set_from_settings(settings, label=f"{label} settings")
    )
    if observed_seed_set != expected_seed_set:
        raise CompactMatchedLearningQualityError(f"{label} eval_seed_set mismatch")
    observed_max_steps = _eval_summary_max_steps(summary)
    if observed_max_steps is None:
        raise CompactMatchedLearningQualityError(f"{label} eval_max_steps missing")
    if int(observed_max_steps) != int(settings["eval_max_steps"]):
        raise CompactMatchedLearningQualityError(f"{label} eval_max_steps mismatch")
    config = summary.get("config")
    if isinstance(config, Mapping):
        for summary_key, settings_key in (
            ("num_simulations", "num_simulations"),
            ("batch_size", "batch_size"),
            ("source_max_steps", "source_max_steps"),
            ("reward_variant", "reward_variant"),
            ("eval_reward_variant", "reward_variant"),
            ("opponent_policy_kind", "opponent_policy_kind"),
            ("opponent_runtime_mode", "opponent_runtime_mode"),
            ("opponent_death_mode", "opponent_death_mode"),
            ("natural_bonus_spawn", "natural_bonus_spawn"),
        ):
            if summary_key in config and config.get(summary_key) != settings.get(settings_key):
                raise CompactMatchedLearningQualityError(
                    f"{label} {settings_key} mismatch"
                )


def _read_artifact_json(
    artifact_ref: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    path_text = _require_non_empty(artifact_ref.get("path"), f"{label} path")
    return _read_json_mapping(Path(path_text), label)


def _eval_seed_set_from_settings(
    settings: Mapping[str, Any],
    *,
    label: str,
) -> list[int]:
    seed_set = settings.get("eval_seed_set")
    if (
        not isinstance(seed_set, Sequence)
        or isinstance(seed_set, (str, bytes))
        or not seed_set
    ):
        raise CompactMatchedLearningQualityError(
            f"{label} eval_seed_set must be a non-empty sequence"
        )
    result: list[int] = []
    for index, seed in enumerate(seed_set):
        if isinstance(seed, bool):
            raise CompactMatchedLearningQualityError(
                f"{label} eval_seed_set[{index}] must be an integer seed"
            )
        try:
            result.append(int(seed))
        except (TypeError, ValueError) as exc:
            raise CompactMatchedLearningQualityError(
                f"{label} eval_seed_set[{index}] must be an integer seed"
            ) from exc
    if len(set(result)) != len(result):
        raise CompactMatchedLearningQualityError(f"{label} eval_seed_set must be unique")
    return result


def _eval_summary_seed_set(summary: Mapping[str, Any]) -> list[int]:
    for container in (summary, summary.get("selection"), summary.get("config")):
        if not isinstance(container, Mapping):
            continue
        for key in ("eval_seed_set", "eval_seed_values", "eval_seeds"):
            value = container.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [int(seed) for seed in value]
        for key in ("eval_seed", "seed"):
            if container.get(key) is not None:
                return [int(container[key])]
    seeds = {
        int(row["seed"])
        for row in summary.get("survival_table", [])
        if isinstance(row, Mapping) and row.get("seed") is not None
    }
    if seeds:
        return sorted(seeds)
    episode = summary.get("episode")
    if isinstance(episode, Mapping) and episode.get("seed") is not None:
        return [int(episode["seed"])]
    raise CompactMatchedLearningQualityError("eval summary seed set missing")


def _eval_summary_max_steps(summary: Mapping[str, Any]) -> int | None:
    for container in (summary, summary.get("config")):
        if not isinstance(container, Mapping):
            continue
        for key in ("eval_max_steps", "max_eval_steps", "eval_steps"):
            value = container.get(key)
            if value is not None and _is_finite_number(value):
                return int(value)
    caps = {
        int(row["cap"])
        for row in summary.get("survival_table", [])
        if isinstance(row, Mapping)
        and row.get("cap") is not None
        and _is_finite_number(row.get("cap"))
    }
    if len(caps) == 1:
        return caps.pop()
    episode = summary.get("episode")
    if isinstance(episode, Mapping):
        for key in ("cap", "steps_run"):
            value = episode.get(key)
            if value is not None and _is_finite_number(value):
                return int(value)
    return None


def _ref_text_matches(value: Any, expected: Any) -> bool:
    value_text = str(value or "").strip()
    expected_text = str(expected or "").strip()
    if not value_text or not expected_text:
        return False
    if value_text == expected_text:
        return True
    value_path = Path(value_text)
    expected_path = Path(expected_text)
    if value_path.is_absolute() and expected_path.is_absolute():
        return value_path.resolve() == expected_path.resolve()
    if value_path.is_absolute() and len(expected_path.parts) > 1:
        return value_path.as_posix().endswith(expected_path.as_posix())
    if expected_path.is_absolute() and len(value_path.parts) > 1:
        return expected_path.as_posix().endswith(value_path.as_posix())
    return False


def _validate_provenance_ref_matches_artifact(
    value: Any,
    artifact_ref: Mapping[str, Any],
    *,
    label: str,
) -> None:
    ref_text = _require_non_empty(value, label)
    kind = _require_non_empty(artifact_ref.get("kind"), f"{label} artifact kind")
    path = _require_non_empty(artifact_ref.get("path"), f"{label} artifact path")
    if ref_text not in {kind, path}:
        raise CompactMatchedLearningQualityError(
            f"{label} must match artifact kind or path"
        )


def _validate_capture_role(role: str) -> str:
    role_text = str(role)
    if role_text not in {STOCK_REFERENCE_ROLE, COMPACT_CANDIDATE_ROLE}:
        raise CompactMatchedLearningQualityError("matched quality capture role invalid")
    return role_text


def _route_for_role(role: str) -> str:
    if role == STOCK_REFERENCE_ROLE:
        return STOCK_TRAIN_MUZERO_ROUTE
    if role == COMPACT_CANDIDATE_ROLE:
        return COMPACT_OWNED_TRAINER_ROUTE
    raise CompactMatchedLearningQualityError("matched quality capture role invalid")


def _capture_calls_train_muzero(
    capture: Mapping[str, Any],
    *,
    role: str,
) -> bool:
    if role == STOCK_REFERENCE_ROLE:
        return bool(capture["called_train_muzero"])
    return bool(capture["calls_train_muzero"])


def _validate_lifecycle_input(
    lifecycle: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> str:
    if lifecycle.get("schema_id") != COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID:
        raise CompactMatchedLearningQualityError("unified lifecycle schema mismatch")
    if lifecycle.get("ok") is not True:
        raise CompactMatchedLearningQualityError("unified lifecycle must be ok=true")
    if lifecycle.get("promotion_claim") is not False:
        raise CompactMatchedLearningQualityError(
            "unified lifecycle promotion_claim must be false"
        )
    lifecycle_checkpoint_id = _require_non_empty(
        lifecycle.get("checkpoint_id"),
        "unified lifecycle checkpoint_id",
    )
    compatibility_checkpoint_id = _require_non_empty(
        compatibility.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    if lifecycle_checkpoint_id != compatibility_checkpoint_id:
        raise CompactMatchedLearningQualityError(
            "compatibility candidate does not match unified lifecycle checkpoint"
        )
    return lifecycle_checkpoint_id


def _validate_quality_arm(
    arm: Mapping[str, Any],
    *,
    expected_role: str,
    expected_route: str,
    expected_calls_train_muzero: bool,
    expected_candidate_checkpoint_id: str | None,
) -> None:
    if arm.get("schema_id") != COMPACT_MATCHED_LEARNING_QUALITY_ARM_SCHEMA_ID:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} matched quality arm schema mismatch"
        )
    if arm.get("ok") is not True:
        raise CompactMatchedLearningQualityError(f"{expected_role} must be ok=true")
    if arm.get("status") != "complete":
        raise CompactMatchedLearningQualityError(f"{expected_role} status must be complete")
    if arm.get("role") != expected_role:
        raise CompactMatchedLearningQualityError(f"{expected_role} role mismatch")
    if arm.get("route") != expected_route:
        raise CompactMatchedLearningQualityError(f"{expected_role} route mismatch")
    if arm.get("profile_only") is not False:
        raise CompactMatchedLearningQualityError(f"{expected_role} requires profile_only=false")
    if arm.get("row_purpose") != "matched_learning_quality_canary":
        raise CompactMatchedLearningQualityError(
            f"{expected_role} row_purpose must be matched_learning_quality_canary"
        )
    if arm.get("quality_currency") != "mean_survival_delta":
        raise CompactMatchedLearningQualityError(
            f"{expected_role} quality_currency must be mean_survival_delta"
        )
    if arm.get("calls_train_muzero") is not expected_calls_train_muzero:
        expected = "true" if expected_calls_train_muzero else "false"
        raise CompactMatchedLearningQualityError(
            f"{expected_role} calls_train_muzero must be {expected}"
        )
    if arm.get("touches_live_runs") is not False:
        raise CompactMatchedLearningQualityError(f"{expected_role} must not touch live runs")
    if expected_candidate_checkpoint_id is not None:
        candidate = _require_non_empty(
            arm.get("candidate_checkpoint_id"),
            f"{expected_role} candidate_checkpoint_id",
        )
        if candidate != expected_candidate_checkpoint_id:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} candidate checkpoint mismatch"
            )
    initial_digest = _require_non_empty(
        arm.get("initial_model_state_digest"),
        f"{expected_role} initial_model_state_digest",
    )
    final_digest = _require_non_empty(
        arm.get("final_model_state_digest"),
        f"{expected_role} final_model_state_digest",
    )
    if initial_digest == final_digest:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} model state digest did not change"
        )
    if arm.get("model_state_digest_changed") is not True:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} model_state_digest_changed must be true"
        )
    if expected_role == COMPACT_CANDIDATE_ROLE:
        if arm.get("real_compact_owned_training_work") is not True:
            raise CompactMatchedLearningQualityError(
                "compact_candidate real compact-owned training work missing"
            )
        if arm.get("model_identity_scope") != "candidate_loaded_checkpoint":
            raise CompactMatchedLearningQualityError(
                "compact_candidate requires candidate_loaded_checkpoint identity"
            )
    for key in (
        "arm_id",
        "run_id",
        "denominator_id",
        "quality_horizon",
        "hardware_class",
    ):
        _require_non_empty(arm.get(key), f"{expected_role} {key}")
    _validate_source_fingerprint(arm, expected_role=expected_role)
    _validate_eval_settings(arm, expected_role=expected_role)
    _validate_quality_curve(arm, expected_role=expected_role)
    _validate_denominators(arm, expected_role=expected_role)
    _validate_artifact_refs(arm, expected_role=expected_role)
    _validate_required_quality_artifact_kinds(arm, expected_role=expected_role)
    _validate_arm_non_claims(arm, expected_role=expected_role)


def _validate_eval_settings(
    arm: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    settings = _required_mapping(arm.get("eval_settings"), f"{expected_role} eval_settings")
    for key in REQUIRED_ARM_EVAL_SETTING_KEYS:
        if key not in settings:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} eval_settings missing {key}"
            )
    seed_set = _eval_seed_set_from_settings(
        settings,
        label=f"{expected_role} eval_settings",
    )
    if len(seed_set) < MIN_MATCHED_EVAL_SEED_COUNT:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} eval_seed_set must contain at least "
            f"{MIN_MATCHED_EVAL_SEED_COUNT} seeds"
        )
    if int(settings.get("eval_episode_count", 0)) != len(seed_set):
        raise CompactMatchedLearningQualityError(
            f"{expected_role} eval_episode_count must match eval_seed_set"
        )
    for key in (
        "eval_episode_count",
        "source_max_steps",
        "eval_max_steps",
        "num_simulations",
        "batch_size",
        "num_unroll_steps",
        "td_steps",
    ):
        if int(settings.get(key, 0)) <= 0:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} eval_settings requires {key} > 0"
            )
    if int(settings["eval_max_steps"]) < MIN_MATCHED_EVAL_MAX_STEPS:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} eval_max_steps must be >= {MIN_MATCHED_EVAL_MAX_STEPS}"
        )
    if settings.get("death_mode") != "normal":
        raise CompactMatchedLearningQualityError(
            f"{expected_role} death_mode must be normal"
        )
    if settings.get("rnd_enabled") is not False:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} rnd_enabled must be false for this canary"
        )
    for key in ("root_noise", "dirichlet_alpha", "policy_noise", "discount", "support_scale"):
        if not _is_finite_number(settings.get(key)):
            raise CompactMatchedLearningQualityError(
                f"{expected_role} eval_settings non-finite {key}"
            )
    if float(settings["discount"]) <= 0.0:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} discount must be > 0"
        )
    if float(settings["support_scale"]) <= 0.0:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} support_scale must be > 0"
        )


def _validate_quality_curve(
    arm: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    curve = _quality_curve(arm)
    settings = _required_mapping(arm.get("eval_settings"), f"{expected_role} eval_settings")
    expected_episodes = int(settings["eval_episode_count"])
    if str(curve[0].get("point_id", "")) != "pre_train":
        raise CompactMatchedLearningQualityError(
            f"{expected_role} first eval point must be pre_train"
        )
    if str(curve[-1].get("point_id", "")) != "post_train":
        raise CompactMatchedLearningQualityError(
            f"{expected_role} final eval point must be post_train"
        )
    previous_step = -1
    for index, point in enumerate(curve):
        label = f"{expected_role} quality_curve[{index}]"
        _require_non_empty(point.get("checkpoint_ref"), f"{label} checkpoint_ref")
        step = int(point.get("checkpoint_step", -1))
        if step < 0:
            raise CompactMatchedLearningQualityError(f"{label} checkpoint_step must be >= 0")
        if step < previous_step:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} quality curve checkpoint_step must be monotonic"
            )
        previous_step = step
        if int(point.get("eval_episode_count", 0)) != expected_episodes:
            raise CompactMatchedLearningQualityError(
                f"{label} eval_episode_count must match eval_settings"
            )
        if int(point.get("evaluator_eval_calls", 0)) <= 0:
            raise CompactMatchedLearningQualityError(
                f"{label} evaluator_eval_calls must be > 0"
            )
        for key in (
            "mean_survival",
            "median_survival",
            "min_survival",
            "max_survival",
            "terminal_rate",
            "cap_rate",
        ):
            if not _is_finite_number(point.get(key)):
                raise CompactMatchedLearningQualityError(f"{label} non-finite {key}")
        minimum = float(point["min_survival"])
        maximum = float(point["max_survival"])
        mean = float(point["mean_survival"])
        median = float(point["median_survival"])
        if minimum > mean or mean > maximum or minimum > median or median > maximum:
            raise CompactMatchedLearningQualityError(
                f"{label} survival summary is internally inconsistent"
            )
        for key in ("terminal_rate", "cap_rate", "death_rate"):
            if key in point:
                value = float(point[key])
                if value < 0.0 or value > 1.0:
                    raise CompactMatchedLearningQualityError(
                        f"{label} {key} must be in [0, 1]"
                    )
                if key == "cap_rate" and value > MAX_MATCHED_EVAL_CAP_RATE:
                    raise CompactMatchedLearningQualityError(
                        f"{label} cap_rate exceeds saturation ceiling"
                    )
    if int(curve[-1]["checkpoint_step"]) <= int(curve[0]["checkpoint_step"]):
        raise CompactMatchedLearningQualityError(
            f"{expected_role} quality curve requires a later final checkpoint"
        )
    if math.isclose(
        float(curve[-1]["mean_survival"]),
        float(curve[0]["mean_survival"]),
        abs_tol=0.0,
    ):
        raise CompactMatchedLearningQualityError(
            f"{expected_role} requires observed mean_survival movement"
        )


def _validate_denominators(
    arm: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    denominators = _required_mapping(
        arm.get("denominators"),
        f"{expected_role} denominators",
    )
    speed_currency = str(denominators.get("speed_currency", "")).strip()
    if speed_currency in PROFILE_ONLY_SPEED_CURRENCIES:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} profile-only speed currency is not quality evidence"
        )
    if denominators.get("uses_fallback_denominator") is not False:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} uses_fallback_denominator must be false"
        )
    if denominators.get("wall_sec_used_for_speed_claim") is not False:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} wall_sec_used_for_speed_claim must be false"
        )
    if expected_role == STOCK_REFERENCE_ROLE:
        if denominators.get("denominator_currency") != "stock_train_muzero_learning_quality":
            raise CompactMatchedLearningQualityError(
                "stock_reference denominator currency mismatch"
            )
        if denominators.get("env_step_currency") != "stock_train_muzero_raw_env_steps":
            raise CompactMatchedLearningQualityError(
                "stock_reference env_step_currency mismatch"
            )
        for key in (
            "collector_envstep_delta",
            "learner_train_calls",
            "replay_sample_calls",
            "checkpoint_iteration_delta",
            "training_wall_sec",
        ):
            _require_positive_number(
                denominators.get(key),
                f"stock_reference denominators {key}",
            )
    elif expected_role == COMPACT_CANDIDATE_ROLE:
        if denominators.get("denominator_currency") != "compact_owned_learning_quality":
            raise CompactMatchedLearningQualityError(
                "compact_candidate denominator currency mismatch"
            )
        if denominators.get("env_step_currency") != "compact_owned_trainer_env_steps":
            raise CompactMatchedLearningQualityError(
                "compact_candidate env_step_currency mismatch"
            )
        for key in (
            "learner_update_count_delta",
            "sample_batch_count_delta",
            "compact_rollout_rows",
            "compact_sample_rows",
            "training_wall_sec",
        ):
            _require_positive_number(
                denominators.get(key),
                f"compact_candidate denominators {key}",
            )
        if not _is_finite_number(denominators.get("learner_loss_mean")):
            raise CompactMatchedLearningQualityError(
                "compact_candidate denominators learner_loss_mean must be finite"
            )


def _validate_artifact_refs(
    arm: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    refs = arm.get("artifact_refs")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)) or not refs:
        raise CompactMatchedLearningQualityError(
            f"{expected_role} artifact_refs must be a non-empty list"
        )
    for index, ref_obj in enumerate(refs):
        ref = _required_mapping(ref_obj, f"{expected_role} artifact_refs[{index}]")
        _require_non_empty(ref.get("kind"), f"{expected_role} artifact_refs[{index}] kind")
        path_text = _require_non_empty(
            ref.get("path"),
            f"{expected_role} artifact_refs[{index}] path",
        )
        expected_sha = _require_non_empty(
            ref.get("sha256"),
            f"{expected_role} artifact_refs[{index}] sha256",
        )
        if ref.get("required") is not True:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} artifact_refs[{index}] required must be true"
            )
        path = Path(path_text)
        if not path.is_file():
            raise CompactMatchedLearningQualityError(
                f"{expected_role} artifact_refs[{index}] path missing"
            )
        actual_sha = _file_sha256(path)
        if actual_sha != expected_sha:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} artifact_refs[{index}] sha256 mismatch"
            )


def _validate_required_quality_artifact_kinds(
    arm: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    refs = _required_sequence(arm.get("artifact_refs"), f"{expected_role} artifact_refs")
    kinds = {
        _require_non_empty(
            _required_mapping(ref_obj, f"{expected_role} artifact_ref").get("kind"),
            f"{expected_role} artifact_ref kind",
        )
        for ref_obj in refs
    }
    for kind in REQUIRED_CAPTURE_ARTIFACT_KINDS:
        if kind not in kinds:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} artifact_refs missing {kind}"
            )


def _validate_arm_non_claims(
    arm: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    non_claims = _required_mapping(arm.get("non_claims"), f"{expected_role} non_claims")
    for key in (*NONCLAIM_FALSE_KEYS, "touches_live_runs"):
        if arm.get(key) is not False:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} top-level {key} must be false"
            )
        if non_claims.get(key) is not False:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} non_claim {key} must be false"
            )


def _matched_pair_reciprocal_roles(
    stock_arm: Mapping[str, Any],
    compact_arm: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        STOCK_REFERENCE_ROLE: {
            "role": stock_arm.get("role"),
            "route": stock_arm.get("route"),
            "calls_train_muzero": stock_arm.get("calls_train_muzero"),
            "touches_live_runs": stock_arm.get("touches_live_runs"),
        },
        COMPACT_CANDIDATE_ROLE: {
            "role": compact_arm.get("role"),
            "route": compact_arm.get("route"),
            "calls_train_muzero": compact_arm.get("calls_train_muzero"),
            "touches_live_runs": compact_arm.get("touches_live_runs"),
        },
    }


def _stock_denominator_count_check_report(
    denominators: Mapping[str, Any],
) -> dict[str, Any]:
    collector_envstep_delta = _require_positive_integral_count(
        denominators.get("collector_envstep_delta"),
        "stock_reference denominators collector_envstep_delta",
    )
    learner_train_calls = _require_positive_integral_count(
        denominators.get("learner_train_calls"),
        "stock_reference denominators learner_train_calls",
    )
    replay_sample_calls = _require_positive_integral_count(
        denominators.get("replay_sample_calls"),
        "stock_reference denominators replay_sample_calls",
    )
    checkpoint_iteration_delta = _require_positive_integral_count(
        denominators.get("checkpoint_iteration_delta"),
        "stock_reference denominators checkpoint_iteration_delta",
    )
    if learner_train_calls != replay_sample_calls:
        raise CompactMatchedLearningQualityError(
            "stock_reference learner_train_calls must equal replay_sample_calls"
        )
    if checkpoint_iteration_delta > learner_train_calls:
        raise CompactMatchedLearningQualityError(
            "stock_reference checkpoint_iteration_delta must be <= learner_train_calls"
        )
    if collector_envstep_delta < learner_train_calls:
        raise CompactMatchedLearningQualityError(
            "stock_reference collector_envstep_delta must cover learner_train_calls"
        )
    return {
        "collector_envstep_delta": collector_envstep_delta,
        "learner_train_calls": learner_train_calls,
        "replay_sample_calls": replay_sample_calls,
        "checkpoint_iteration_delta": checkpoint_iteration_delta,
        "learner_train_calls_equal_replay_sample_calls": True,
        "checkpoint_iterations_within_learner_calls": True,
        "collector_envsteps_cover_learner_calls": True,
    }


def _compact_denominator_count_check_report(
    denominators: Mapping[str, Any],
) -> dict[str, Any]:
    compact_rollout_rows = _require_positive_integral_count(
        denominators.get("compact_rollout_rows"),
        "compact_candidate denominators compact_rollout_rows",
    )
    compact_sample_rows = _require_positive_integral_count(
        denominators.get("compact_sample_rows"),
        "compact_candidate denominators compact_sample_rows",
    )
    learner_update_count_delta = _require_positive_integral_count(
        denominators.get("learner_update_count_delta"),
        "compact_candidate denominators learner_update_count_delta",
    )
    sample_batch_count_delta = _require_positive_integral_count(
        denominators.get("sample_batch_count_delta"),
        "compact_candidate denominators sample_batch_count_delta",
    )
    if compact_rollout_rows < compact_sample_rows:
        raise CompactMatchedLearningQualityError(
            "compact_candidate compact_rollout_rows must cover compact_sample_rows"
        )
    if compact_sample_rows < sample_batch_count_delta:
        raise CompactMatchedLearningQualityError(
            "compact_candidate compact_sample_rows must cover sample_batch_count_delta"
        )
    if learner_update_count_delta < sample_batch_count_delta:
        raise CompactMatchedLearningQualityError(
            "compact_candidate learner_update_count_delta must cover sample_batch_count_delta"
        )
    return {
        "compact_rollout_rows": compact_rollout_rows,
        "compact_sample_rows": compact_sample_rows,
        "learner_update_count_delta": learner_update_count_delta,
        "sample_batch_count_delta": sample_batch_count_delta,
        "rollout_rows_cover_sample_rows": True,
        "sample_rows_cover_sample_batches": True,
        "learner_updates_cover_sample_batches": True,
        "learner_updates_per_sample_batch": (
            float(learner_update_count_delta) / float(sample_batch_count_delta)
        ),
        "count_contract": (
            "compact_sample_rows counts replay rows; learner_update_count_delta "
            "counts gradient steps and may exceed sample rows when train_steps>1"
        ),
    }


def _validate_stock_denominator_count_coherence(
    denominators: Mapping[str, Any],
) -> None:
    _stock_denominator_count_check_report(denominators)


def _validate_compact_denominator_count_coherence(
    denominators: Mapping[str, Any],
) -> None:
    _compact_denominator_count_check_report(denominators)


def _matched_pair_fingerprint_inventory(
    stock_arm: Mapping[str, Any],
    compact_arm: Mapping[str, Any],
) -> dict[str, Any]:
    stock_fingerprint = _required_mapping(
        stock_arm.get("source_fingerprint"),
        "stock source_fingerprint",
    )
    compact_fingerprint = _required_mapping(
        compact_arm.get("source_fingerprint"),
        "compact source_fingerprint",
    )
    return {
        STOCK_REFERENCE_ROLE: _fingerprint_inventory(stock_arm),
        COMPACT_CANDIDATE_ROLE: _fingerprint_inventory(compact_arm),
        "same_git_commit": (
            stock_fingerprint.get("git_commit") == compact_fingerprint.get("git_commit")
        ),
    }


def _fingerprint_inventory(arm: Mapping[str, Any]) -> dict[str, Any]:
    role = _require_non_empty(arm.get("role"), "fingerprint inventory role")
    fingerprint = _required_mapping(
        arm.get("source_fingerprint"),
        f"{role} source_fingerprint",
    )
    inventory: dict[str, Any] = {
        "arm_id": _require_non_empty(arm.get("arm_id"), f"{role} arm_id"),
        "run_id": _require_non_empty(arm.get("run_id"), f"{role} run_id"),
        "hardware_class": _require_non_empty(
            arm.get("hardware_class"),
            f"{role} hardware_class",
        ),
        "git_commit": _require_non_empty(
            fingerprint.get("git_commit"),
            f"{role} source_fingerprint git_commit",
        ),
        "git_status_dirty": bool(fingerprint.get("git_status_dirty")),
        "producer_script": _require_non_empty(
            fingerprint.get("producer_script"),
            f"{role} source_fingerprint producer_script",
        ),
        "producer_route": _require_non_empty(
            fingerprint.get("producer_route"),
            f"{role} source_fingerprint producer_route",
        ),
        "matched_surface_sha256": _json_sha256(
            _required_mapping(
                fingerprint.get("matched_surface"),
                f"{role} source_fingerprint matched_surface",
            )
        ),
    }
    for key in (
        "image_digest",
        "loaded_compact_checkpoint_sha256",
        "unified_lifecycle_report_path",
        "train_summary_ref",
    ):
        value = fingerprint.get(key)
        if value is not None and str(value).strip():
            inventory[key] = value
    runtime_compute = fingerprint.get("runtime_compute")
    if isinstance(runtime_compute, Mapping):
        inventory["runtime_compute_sha256"] = _json_sha256(runtime_compute)
        for key in (
            "requested_compute",
            "modal_task_id",
            "torch_cuda_available",
            "torch_cuda_device_name",
            "torch_cuda_device_count",
        ):
            if key in runtime_compute:
                inventory[f"runtime_compute_{key}"] = runtime_compute[key]
    return inventory


def _all_non_claims_false(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key in (*NONCLAIM_FALSE_KEYS, "touches_live_runs", "compact_calls_train_muzero"):
        if value.get(key) is not False:
            return False
    return True


def _validate_source_fingerprint(
    arm: Mapping[str, Any],
    *,
    expected_role: str,
) -> None:
    fingerprint = _required_mapping(
        arm.get("source_fingerprint"),
        f"{expected_role} source_fingerprint",
    )
    for key in REQUIRED_SOURCE_FINGERPRINT_KEYS:
        if key == "git_status_dirty":
            if not isinstance(fingerprint.get(key), bool):
                raise CompactMatchedLearningQualityError(
                    f"{expected_role} source_fingerprint {key} must be boolean"
                )
        elif key == "matched_surface":
            _required_mapping(
                fingerprint.get(key),
                f"{expected_role} source_fingerprint matched_surface",
            )
        else:
            _require_non_empty(
                fingerprint.get(key),
                f"{expected_role} source_fingerprint {key}",
            )
    surface = _required_mapping(
        fingerprint.get("matched_surface"),
        f"{expected_role} source_fingerprint matched_surface",
    )
    for key in REQUIRED_MATCHED_SURFACE_KEYS:
        if key == "eval_seed_set":
            _eval_seed_set_from_settings(
                surface,
                label=f"{expected_role} source_fingerprint matched_surface",
            )
        else:
            _require_non_empty(
                surface.get(key),
                f"{expected_role} source_fingerprint matched_surface {key}",
            )
    settings = _required_mapping(arm.get("eval_settings"), f"{expected_role} eval_settings")
    for key in (
        "reward_variant",
        "policy_observation_backend",
        "opponent_policy_kind",
        "eval_seed_set",
    ):
        if key == "eval_seed_set":
            surface_value = sorted(
                _eval_seed_set_from_settings(
                    surface,
                    label=f"{expected_role} source_fingerprint matched_surface",
                )
            )
            settings_value = sorted(
                _eval_seed_set_from_settings(
                    settings,
                    label=f"{expected_role} eval_settings",
                )
            )
        else:
            surface_value = surface.get(key)
            settings_value = settings.get(key)
        if surface_value != settings_value:
            raise CompactMatchedLearningQualityError(
                f"{expected_role} source_fingerprint matched_surface {key} mismatch"
            )
    if (
        "eval_seed_rng_seed" in surface
        and "eval_seed_rng_seed" in settings
        and surface.get("eval_seed_rng_seed") != settings.get("eval_seed_rng_seed")
    ):
        raise CompactMatchedLearningQualityError(
            f"{expected_role} source_fingerprint matched_surface eval_seed_rng_seed mismatch"
        )


def _matched_source_surface(arm: Mapping[str, Any]) -> dict[str, Any]:
    fingerprint = _required_mapping(arm.get("source_fingerprint"), "source_fingerprint")
    matched_surface = _plain_mapping(
        _required_mapping(fingerprint.get("matched_surface"), "matched_surface")
    )
    matched_surface["eval_seed_set"] = sorted(
        _eval_seed_set_from_settings(matched_surface, label="matched_surface")
    )
    return {
        "git_commit": fingerprint["git_commit"],
        "matched_surface": matched_surface,
    }


def _matched_eval_settings_subset(settings: Mapping[str, Any]) -> dict[str, Any]:
    subset = {key: settings[key] for key in REQUIRED_MATCHED_EVAL_SETTING_KEYS}
    subset["eval_seed_set"] = sorted(
        _eval_seed_set_from_settings(subset, label="matched eval_settings")
    )
    return subset


def _validate_matched_pair(
    stock_arm: Mapping[str, Any],
    compact_arm: Mapping[str, Any],
) -> None:
    for key in ("denominator_id", "quality_horizon"):
        if stock_arm.get(key) != compact_arm.get(key):
            raise CompactMatchedLearningQualityError(f"matched quality {key} mismatch")
    stock_settings = _required_mapping(stock_arm.get("eval_settings"), "stock eval_settings")
    compact_settings = _required_mapping(
        compact_arm.get("eval_settings"),
        "compact eval_settings",
    )
    for key in REQUIRED_MATCHED_EVAL_SETTING_KEYS:
        if key == "eval_seed_set":
            stock_value = sorted(
                _eval_seed_set_from_settings(stock_settings, label="stock eval_settings")
            )
            compact_value = sorted(
                _eval_seed_set_from_settings(
                    compact_settings,
                    label="compact eval_settings",
                )
            )
        else:
            stock_value = stock_settings.get(key)
            compact_value = compact_settings.get(key)
        if stock_value != compact_value:
            raise CompactMatchedLearningQualityError(
                f"matched quality eval_settings {key} mismatch"
            )
    if _matched_source_surface(stock_arm) != _matched_source_surface(compact_arm):
        raise CompactMatchedLearningQualityError(
            "matched quality source matched_surface mismatch"
        )


def _validate_matched_pair_summary(
    payload: Mapping[str, Any],
    stock_arm: Mapping[str, Any],
    compact_arm: Mapping[str, Any],
) -> None:
    matched_pair = _required_mapping(payload.get("matched_pair"), "matched_pair")
    expected_pair = {
        "denominator_id": stock_arm["denominator_id"],
        "quality_horizon": stock_arm["quality_horizon"],
        "hardware_class": (
            stock_arm["hardware_class"]
            if stock_arm.get("hardware_class") == compact_arm.get("hardware_class")
            else "mixed"
        ),
        "stock_hardware_class": stock_arm["hardware_class"],
        "compact_hardware_class": compact_arm["hardware_class"],
        "source_fingerprint_sha256": _json_sha256(
            _matched_source_surface(stock_arm)
        ),
        "stock_source_fingerprint_sha256": _json_sha256(
            stock_arm["source_fingerprint"]
        ),
        "compact_source_fingerprint_sha256": _json_sha256(
            compact_arm["source_fingerprint"]
        ),
        "eval_settings_sha256": _json_sha256(
            _matched_eval_settings_subset(
                _required_mapping(stock_arm["eval_settings"], "eval_settings")
            )
        ),
    }
    for key, expected in expected_pair.items():
        if matched_pair.get(key) != expected:
            raise CompactMatchedLearningQualityError(
                f"matched quality summary {key} mismatch"
            )
    movement = _required_mapping(payload.get("quality_movement"), "quality_movement")
    expected_movement = {
        "stock_reference_delta": _mean_survival_delta(stock_arm),
        "compact_candidate_delta": _mean_survival_delta(compact_arm),
        "compact_minus_stock_delta": (
            _mean_survival_delta(compact_arm) - _mean_survival_delta(stock_arm)
        ),
    }
    if movement.get("metric") != "mean_survival":
        raise CompactMatchedLearningQualityError("matched quality movement metric mismatch")
    if movement.get("quality_currency") != "mean_survival_delta":
        raise CompactMatchedLearningQualityError(
            "matched quality movement currency mismatch"
        )
    for key, expected in expected_movement.items():
        if not math.isclose(float(movement.get(key)), float(expected), abs_tol=1e-12):
            raise CompactMatchedLearningQualityError(
                f"matched quality movement {key} mismatch"
            )
    thresholds = _required_mapping(
        movement.get("movement_thresholds"),
        "movement_thresholds",
    )
    min_abs_delta = float(
        _require_positive_or_zero_number(
            thresholds.get("min_abs_primary_delta"),
            "movement_thresholds min_abs_primary_delta",
        )
    )
    allowed_regression = float(
        _require_positive_or_zero_number(
            thresholds.get("allowed_compact_regression"),
            "movement_thresholds allowed_compact_regression",
        )
    )
    if abs(float(movement["stock_reference_delta"])) <= min_abs_delta:
        raise CompactMatchedLearningQualityError("stock quality movement below threshold")
    if abs(float(movement["compact_candidate_delta"])) <= min_abs_delta:
        raise CompactMatchedLearningQualityError("compact quality movement below threshold")
    if float(movement["compact_candidate_delta"]) < -allowed_regression:
        raise CompactMatchedLearningQualityError("compact quality regression exceeds threshold")
    if movement.get("stock_movement_observed") is not True:
        raise CompactMatchedLearningQualityError("stock quality movement not observed")
    if movement.get("compact_movement_observed") is not True:
        raise CompactMatchedLearningQualityError("compact quality movement not observed")
    if movement.get("matched_quality_canary") is not True:
        raise CompactMatchedLearningQualityError("matched quality canary marker missing")


def _quality_curve(arm: Mapping[str, Any]) -> list[dict[str, Any]]:
    curve = arm.get("quality_curve")
    if not isinstance(curve, Sequence) or isinstance(curve, (str, bytes)):
        raise CompactMatchedLearningQualityError("quality_curve must be a sequence")
    if len(curve) < 2:
        raise CompactMatchedLearningQualityError(
            "quality_curve requires at least two eval points"
        )
    result: list[dict[str, Any]] = []
    for point in curve:
        result.append(_plain_mapping(point))
    return result


def _mean_survival_delta(arm: Mapping[str, Any]) -> float:
    curve = _quality_curve(arm)
    return float(curve[-1]["mean_survival"]) - float(curve[0]["mean_survival"])


def _validate_payload_file(payload: Mapping[str, Any], key: str) -> None:
    path_text = str(payload.get(f"{key}_path", "")).strip()
    sha = str(payload.get(f"{key}_sha256", "")).strip()
    if not path_text or not sha:
        raise CompactMatchedLearningQualityError(f"matched quality {key} file missing")
    path = Path(path_text)
    if not path.is_file():
        raise CompactMatchedLearningQualityError(f"matched quality {key} file not found")
    if _file_sha256(path) != sha:
        raise CompactMatchedLearningQualityError(f"matched quality {key} sha256 mismatch")


def _validate_input_capture_files(
    payload: Mapping[str, Any],
    *,
    stock_arm: Mapping[str, Any],
    compact_arm: Mapping[str, Any],
) -> None:
    files = payload.get("input_capture_files")
    mapping = _required_mapping(files, "input_capture_files")
    expected_arms = {
        "stock_reference_capture": (STOCK_REFERENCE_ROLE, stock_arm),
        "compact_candidate_capture": (COMPACT_CANDIDATE_ROLE, compact_arm),
    }
    for key, (role, expected_arm) in expected_arms.items():
        record = _required_mapping(mapping.get(key), f"input_capture_files {key}")
        path_text = _require_non_empty(record.get("path"), f"{key} path")
        sha = _require_non_empty(record.get("sha256"), f"{key} sha256")
        path = Path(path_text)
        if not path.is_file():
            raise CompactMatchedLearningQualityError(f"{key} path missing")
        if _file_sha256(path) != sha:
            raise CompactMatchedLearningQualityError(f"{key} sha256 mismatch")
        capture = _read_json_mapping(path, key)
        derived_arm = compact_matched_learning_quality_arm_from_capture_v1(
            capture,
            expected_role=role,
        )
        if _json_sha256(derived_arm) != _json_sha256(expected_arm):
            raise CompactMatchedLearningQualityError(
                f"{key} derived arm mismatch"
            )


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise CompactMatchedLearningQualityError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompactMatchedLearningQualityError(
            f"{label} is not valid JSON: {path}"
        ) from exc
    return _required_mapping(payload, label)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactMatchedLearningQualityError("expected mapping")
    return {str(key): item for key, item in value.items()}


def _select_eval_summary_row(
    summary: Mapping[str, Any],
    *,
    checkpoint_ref: str | None,
) -> Mapping[str, Any]:
    aggregate_table = summary.get("survival_aggregate_table")
    if isinstance(aggregate_table, Sequence) and not isinstance(
        aggregate_table,
        (str, bytes),
    ):
        rows = [row for row in aggregate_table if isinstance(row, Mapping)]
        if checkpoint_ref is not None:
            for row in rows:
                if str(row.get("checkpoint") or row.get("checkpoint_ref") or "") == str(
                    checkpoint_ref
                ):
                    return row
            raise CompactMatchedLearningQualityError(
                f"eval summary missing checkpoint {checkpoint_ref}"
            )
        if len(rows) != 1:
            raise CompactMatchedLearningQualityError(
                "eval summary survival_aggregate_table must have exactly one row "
                "when checkpoint_ref is omitted"
            )
        return rows[0]
    episode = summary.get("episode")
    if isinstance(episode, Mapping):
        merged = dict(episode)
        checkpoint = summary.get("checkpoint")
        if isinstance(checkpoint, Mapping):
            if checkpoint.get("path") is not None:
                merged.setdefault("checkpoint_ref", checkpoint.get("path"))
        config = summary.get("config")
        if isinstance(config, Mapping):
            if config.get("checkpoint_ref") is not None:
                merged.setdefault("checkpoint_ref", config.get("checkpoint_ref"))
        merged.setdefault("eval_episode_count", 1)
        merged.setdefault("evaluator_eval_calls", 1)
        return merged
    return summary


def _checkpoint_step_from_summary(
    summary: Mapping[str, Any],
    *,
    fallback_ref: str,
) -> int:
    for key in ("checkpoint_step", "iteration", "checkpoint_iteration"):
        value = summary.get(key)
        if value is not None:
            step = int(value)
            if step < 0:
                raise CompactMatchedLearningQualityError(
                    "checkpoint_step must be >= 0"
                )
            return step
    name = Path(fallback_ref).name
    if name.endswith(".pth.tar"):
        name = name[: -len(".pth.tar")]
    if name.startswith("iteration_"):
        suffix = name.removeprefix("iteration_")
        if suffix.isdigit():
            return int(suffix)
    raise CompactMatchedLearningQualityError(
        "eval summary checkpoint_step missing and checkpoint_ref is not iteration_N"
    )


def _survival_number(summary: Mapping[str, Any], keys: Sequence[str]) -> float:
    value = _first_present_number(summary, keys, label=f"eval summary {keys[0]}")
    return float(value)


def _first_present_number(
    mapping: Mapping[str, Any],
    keys: Sequence[str],
    *,
    label: str,
) -> float:
    for key in keys:
        value = mapping.get(key)
        if value is not None and _is_finite_number(value):
            return float(value)
    raise CompactMatchedLearningQualityError(f"{label} missing or non-finite")


def _rate_from_summary(
    summary: Mapping[str, Any],
    *,
    direct_keys: Sequence[str],
    count_keys: Sequence[str],
    histogram_keys: Sequence[str],
    denominator: int,
) -> float:
    for key in direct_keys:
        value = summary.get(key)
        if value is not None and _is_finite_number(value):
            return _bounded_rate(float(value), key)
    for key in count_keys:
        value = summary.get(key)
        if value is not None and _is_finite_number(value):
            return _bounded_rate(float(value) / float(denominator), key)
    histogram = summary.get("outcome_histogram")
    if isinstance(histogram, Mapping):
        total = 0.0
        for key in histogram_keys:
            value = histogram.get(key)
            if value is not None and _is_finite_number(value):
                total += float(value)
        if total > 0.0:
            return _bounded_rate(total / float(denominator), "outcome_histogram")
    for key in ("terminal", "terminal_reason", "outcome"):
        value = str(summary.get(key, "")).strip().lower()
        if value in {"cap", "capped", "timeout"}:
            return _bounded_rate(1.0 / float(denominator), key)
    if summary.get("steps_truncated") is True:
        return _bounded_rate(1.0 / float(denominator), "steps_truncated")
    return 0.0


def _terminal_rate_from_summary(
    summary: Mapping[str, Any],
    *,
    denominator: int,
    cap_rate: float,
) -> float:
    for key in ("terminal_rate", "done_rate"):
        value = summary.get(key)
        if value is not None and _is_finite_number(value):
            return _bounded_rate(float(value), key)
    failure_count = 0.0
    for key in ("failure_count", "error_count"):
        value = summary.get(key)
        if value is not None and _is_finite_number(value):
            failure_count += float(value)
    return _bounded_rate(
        1.0 - cap_rate - (failure_count / float(denominator)),
        "derived_terminal_rate",
    )


def _optional_death_rate_from_summary(
    summary: Mapping[str, Any],
    *,
    denominator: int,
) -> float | None:
    for key in ("death_rate",):
        value = summary.get(key)
        if value is not None and _is_finite_number(value):
            return _bounded_rate(float(value), key)
    histogram = summary.get("terminal_cause_histogram")
    if not isinstance(histogram, Mapping):
        histogram = summary.get("terminal_reason_histogram")
    if isinstance(histogram, Mapping):
        death_count = 0.0
        for key, value in histogram.items():
            key_text = str(key)
            if (
                key_text not in {"cap", "capped", "timeout", "error", "none"}
                and value is not None
                and _is_finite_number(value)
            ):
                death_count += float(value)
        if death_count > 0.0:
            return _bounded_rate(death_count / float(denominator), "death_rate")
    return None


def _bounded_rate(value: float, label: str) -> float:
    if value < 0.0 or value > 1.0:
        raise CompactMatchedLearningQualityError(f"{label} must be in [0, 1]")
    return value


def _artifact_refs_from_paths(
    paths: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    if not paths:
        raise CompactMatchedLearningQualityError("capture artifact paths missing")
    refs: list[dict[str, Any]] = []
    for kind, raw_path in sorted(paths.items()):
        kind_text = _require_non_empty(kind, "artifact kind")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise CompactMatchedLearningQualityError(
                f"capture artifact path missing: {path}"
            )
        refs.append(
            {
                "kind": kind_text,
                "path": str(path),
                "sha256": _file_sha256(path),
                "required": True,
            }
        )
    return refs


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactMatchedLearningQualityError(f"{label} must be a mapping")
    return value


def _required_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CompactMatchedLearningQualityError(f"{label} must be a sequence")
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactMatchedLearningQualityError(f"{label} must be non-empty")
    return text


def _require_positive_number(value: Any, label: str) -> float:
    if not _is_finite_number(value):
        raise CompactMatchedLearningQualityError(f"{label} must be finite")
    number = float(value)
    if number <= 0.0:
        raise CompactMatchedLearningQualityError(f"{label} must be > 0")
    return number


def _require_positive_integral_count(value: Any, label: str) -> int:
    number = _require_positive_number(value, label)
    if not number.is_integer():
        raise CompactMatchedLearningQualityError(f"{label} must be an integer count")
    return int(number)


def _require_positive_or_zero_number(value: Any, label: str) -> float:
    if not _is_finite_number(value):
        raise CompactMatchedLearningQualityError(f"{label} must be finite")
    number = float(value)
    if number < 0.0:
        raise CompactMatchedLearningQualityError(f"{label} must be >= 0")
    return number


def _is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
