"""Compact-only longer-horizon learning metrics evidence.

This report is intentionally narrower than the matched stock-vs-compact
learning-quality canary.  It proves that the compact-owned trainer can emit a
multi-checkpoint learning trace with bound evals and monotonic training
denominators; it does not claim promotion, speedup, live-run safety, rating
quality, or compact superiority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_CANDIDATE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_OWNED_TRAINER_ROUTE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    compact_matched_learning_quality_eval_point_from_summary_v1,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    matched_learning_quality_non_claims_v1,
)
from curvyzero.training.compact_training_metrics_lineage import (
    compact_training_metrics_lineage_evidence_ref,
)
from curvyzero.training.compact_training_metrics_lineage import (
    validate_compact_training_metrics_lineage_v1,
)


COMPACT_LONGER_HORIZON_LEARNING_METRICS_SCHEMA_ID = (
    "curvyzero_compact_longer_horizon_learning_metrics/v1"
)
COMPACT_LONGER_HORIZON_LEARNING_METRICS_STATUS = (
    "compact_longer_horizon_learning_metrics_observed"
)
COMPACT_LONGER_HORIZON_LEARNING_METRICS_EVIDENCE_REF_PREFIX = (
    "compact_longer_horizon_learning_metrics:"
)
COMPACT_LONGER_HORIZON_LEARNING_METRICS_READINESS_LANE = (
    "longer_horizon_compact_learning_metrics"
)
COMPACT_RECORD_STEP_ENTRYPOINT = (
    "curvyzero.training.compact_owned_trainer.CompactOwnedTrainerV1.record_step"
)
MAX_LONGER_HORIZON_EVAL_CAP_RATE = 0.5
MIN_LONGER_HORIZON_CHECKPOINT_COUNT = 3
MIN_LONGER_HORIZON_EVAL_SEED_COUNT = 2
MIN_LONGER_HORIZON_EVAL_MAX_STEPS = 128

REQUIRED_EVAL_SETTING_KEYS = (
    "eval_seed_set",
    "eval_episode_count",
    "eval_max_steps",
    "num_simulations",
    "batch_size",
    "reward_variant",
    "death_mode",
    "root_noise",
    "policy_noise",
    "rnd_enabled",
    "opponent_policy_kind",
)
REQUIRED_CUMULATIVE_DENOMINATOR_KEYS = (
    "train_step_delta",
    "learner_update_count_delta",
    "sample_batch_count_delta",
    "record_step_calls",
    "appended_replay_entry_count",
    "sampled_count",
    "trained_count",
    "compact_rollout_rows",
    "compact_sample_rows",
    "replay_store_entry_count",
    "replay_store_index_row_count",
)
REQUIRED_INTERVAL_POSITIVE_KEYS = (
    "learner_update_count_delta",
    "sample_batch_count_delta",
    "record_step_calls",
    "compact_rollout_rows",
    "compact_sample_rows",
)
REQUIRED_LEARNER_METRIC_KEYS = (
    "learner_loss",
    "learner_policy_loss",
    "learner_value_loss",
    "learner_reward_loss",
    "learner_grad_norm_before_clip",
    "learner_sample_rows",
    "learner_train_steps",
)
REQUIRED_SEARCH_PROVENANCE_KEYS = (
    "search_impl",
    "root_batch_schema_id",
    "search_result_schema_id",
    "replay_payload_schema_id",
    "search_replay_payload_digest",
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
    "touches_live_runs",
    "calls_train_muzero",
    "compact_calls_train_muzero",
    "public_leaderboard_claim",
    "promotion_published",
)


class CompactLongerHorizonLearningMetricsError(ValueError):
    """Raised when compact longer-horizon metrics evidence would overclaim."""


def build_compact_longer_horizon_learning_metrics_v1(
    *,
    run_id: str,
    compatibility_report_path: str | Path,
    unified_lifecycle_report_path: str | Path,
    checkpoint_series: Sequence[Mapping[str, Any]],
    source_fingerprint: Mapping[str, Any],
    eval_settings: Mapping[str, Any],
    training_settings: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build and validate a compact-only multi-checkpoint metrics report."""

    compatibility_path = Path(compatibility_report_path).resolve()
    lifecycle_path = Path(unified_lifecycle_report_path).resolve()
    compatibility = _read_json_mapping(compatibility_path, "compatibility report")
    lifecycle = _read_json_mapping(lifecycle_path, "unified lifecycle report")
    _validate_compatibility_input(compatibility)
    candidate_checkpoint_id = _validate_lifecycle_input(lifecycle, compatibility)

    settings = _plain_mapping(eval_settings)
    _validate_eval_settings(settings)
    training = _plain_mapping(training_settings)
    _validate_training_settings(training)

    series = [
        _normalize_checkpoint_point(
            point,
            expected_index=index,
            eval_settings=settings,
        )
        for index, point in enumerate(checkpoint_series)
    ]
    trend_summary = _trend_summary(series)
    cumulative_denominators = dict(series[-1]["cumulative_denominators"]) if series else {}
    non_claims = _longer_horizon_non_claims()
    payload = {
        "schema_id": COMPACT_LONGER_HORIZON_LEARNING_METRICS_SCHEMA_ID,
        "ok": True,
        "status": COMPACT_LONGER_HORIZON_LEARNING_METRICS_STATUS,
        "run_id": str(run_id),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "readiness_lane": COMPACT_LONGER_HORIZON_LEARNING_METRICS_READINESS_LANE,
        "role": COMPACT_CANDIDATE_ROLE,
        "route": COMPACT_OWNED_TRAINER_ROUTE,
        "candidate_checkpoint_id": candidate_checkpoint_id,
        "compatibility_report_path": str(compatibility_path),
        "compatibility_report_sha256": _file_sha256(compatibility_path),
        "unified_lifecycle_report_path": str(lifecycle_path),
        "unified_lifecycle_report_sha256": _file_sha256(lifecycle_path),
        "loaded_compact_checkpoint_path": str(
            lifecycle.get("compact_checkpoint_path") or ""
        ),
        "loaded_compact_checkpoint_sha256": (
            _file_sha256(Path(str(lifecycle["compact_checkpoint_path"])))
            if lifecycle.get("compact_checkpoint_path")
            and Path(str(lifecycle["compact_checkpoint_path"])).is_file()
            else ""
        ),
        "input_compatibility": {
            "promotion_eligible": bool(compatibility["promotion_eligible"]),
            "promotion_claim": bool(compatibility["promotion_claim"]),
            "calls_train_muzero": bool(compatibility["calls_train_muzero"]),
            "touches_live_runs": bool(compatibility["touches_live_runs"]),
            "coach_speed_row_gate": bool(compatibility.get("coach_speed_row_gate")),
        },
        "source_fingerprint": _plain_mapping(source_fingerprint),
        "eval_settings": settings,
        "training_settings": training,
        "checkpoint_series": series,
        "trend_summary": trend_summary,
        "cumulative_denominators": cumulative_denominators,
        "artifact_refs": _artifact_refs(series),
        "readiness": {
            "longer_horizon_compact_learning_metrics": True,
            "promotion_readiness_complete": False,
            "remaining_promotion_readiness_blockers_after_this_artifact": [],
            "final_promotion_bundle_still_required": True,
        },
        "attached_claims": {
            "longer_horizon_compact_learning_metrics": True,
            "compact_multi_checkpoint_metrics_observed": True,
            **non_claims,
        },
        "non_claims": non_claims,
    }
    payload["evidence_ref"] = compact_longer_horizon_learning_metrics_evidence_ref(
        payload
    )
    validate_compact_longer_horizon_learning_metrics_v1(payload)
    return payload


def validate_compact_longer_horizon_learning_metrics_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate compact-only longer-horizon learning metrics evidence."""

    if payload.get("schema_id") != COMPACT_LONGER_HORIZON_LEARNING_METRICS_SCHEMA_ID:
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics must be ok=true"
        )
    if payload.get("status") != COMPACT_LONGER_HORIZON_LEARNING_METRICS_STATUS:
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics status mismatch"
        )
    if payload.get("readiness_lane") != (
        COMPACT_LONGER_HORIZON_LEARNING_METRICS_READINESS_LANE
    ):
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics readiness lane mismatch"
        )
    if payload.get("role") != COMPACT_CANDIDATE_ROLE:
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics role mismatch"
        )
    if payload.get("route") != COMPACT_OWNED_TRAINER_ROUTE:
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics route mismatch"
        )
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
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics input must be compatibility eligible"
        )
    if input_compatibility.get("coach_speed_row_gate") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics input speed row missing"
        )
    for key in ("promotion_claim", "calls_train_muzero", "touches_live_runs"):
        if input_compatibility.get(key) is not False:
            raise CompactLongerHorizonLearningMetricsError(
                f"longer-horizon metrics input {key} must be false"
            )

    eval_settings = _required_mapping(payload.get("eval_settings"), "eval_settings")
    _validate_eval_settings(eval_settings)
    training_settings = _required_mapping(
        payload.get("training_settings"),
        "training_settings",
    )
    _validate_training_settings(training_settings)
    _validate_claims(payload)

    series = _required_sequence(payload.get("checkpoint_series"), "checkpoint_series")
    if len(series) < MIN_LONGER_HORIZON_CHECKPOINT_COUNT:
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics requires at least three checkpoint points"
        )
    seen_compact_hashes: set[str] = set()
    seen_stock_hashes: set[str] = set()
    previous_step = -1
    previous_digest = ""
    previous_cumulative: dict[str, int] | None = None
    for index, raw_point in enumerate(series):
        point = _required_mapping(raw_point, f"checkpoint_series[{index}]")
        if point.get("candidate_checkpoint_id") != candidate_checkpoint_id:
            raise CompactLongerHorizonLearningMetricsError(
                f"checkpoint_series[{index}] candidate checkpoint mismatch"
            )
        if int(point.get("checkpoint_index", -1)) != index:
            raise CompactLongerHorizonLearningMetricsError(
                f"checkpoint_series[{index}] checkpoint_index mismatch"
            )
        step = int(point.get("checkpoint_step", -1))
        if step <= previous_step:
            raise CompactLongerHorizonLearningMetricsError(
                "longer-horizon checkpoint_step must be strictly increasing"
            )
        previous_step = step
        digest = _require_non_empty(
            point.get("model_state_digest"),
            f"checkpoint_series[{index}] model_state_digest",
        )
        if index == 0:
            if point.get("digest_changed_from_previous") is not False:
                raise CompactLongerHorizonLearningMetricsError(
                    "initial checkpoint digest_changed_from_previous must be false"
                )
        else:
            if digest == previous_digest:
                raise CompactLongerHorizonLearningMetricsError(
                    "longer-horizon adjacent model digest did not change"
                )
            if point.get("digest_changed_from_previous") is not True:
                raise CompactLongerHorizonLearningMetricsError(
                    "trained checkpoint digest_changed_from_previous must be true"
                )
        previous_digest = digest

        _validate_point_file(point, "compact_checkpoint")
        _validate_point_file(point, "stock_export")
        _validate_point_file(point, "eval_summary")
        compact_hash = str(point["compact_checkpoint_sha256"])
        stock_hash = str(point["stock_export_sha256"])
        if compact_hash in seen_compact_hashes:
            raise CompactLongerHorizonLearningMetricsError(
                "longer-horizon duplicate compact checkpoint hash"
            )
        if stock_hash in seen_stock_hashes:
            raise CompactLongerHorizonLearningMetricsError(
                "longer-horizon duplicate stock export hash"
            )
        seen_compact_hashes.add(compact_hash)
        seen_stock_hashes.add(stock_hash)

        if point.get("sample_source") != COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE:
            raise CompactLongerHorizonLearningMetricsError(
                "longer-horizon metrics requires compact_env_search_replay_rows"
            )
        if point.get("compact_training_entrypoint") != COMPACT_RECORD_STEP_ENTRYPOINT:
            raise CompactLongerHorizonLearningMetricsError(
                "longer-horizon metrics requires CompactOwnedTrainerV1.record_step"
            )
        _validate_eval_point(point, eval_settings, index=index)
        cumulative = _validate_denominator_curve_point(
            point,
            previous_cumulative=previous_cumulative,
            index=index,
        )
        previous_cumulative = cumulative
        if index > 0:
            _validate_trained_point(point, index=index)

    top_cumulative = _required_mapping(
        payload.get("cumulative_denominators"),
        "cumulative_denominators",
    )
    for key in REQUIRED_CUMULATIVE_DENOMINATOR_KEYS:
        if int(top_cumulative.get(key, -1)) != int(previous_cumulative[key]):
            raise CompactLongerHorizonLearningMetricsError(
                f"top-level cumulative denominator mismatch for {key}"
            )
        if int(top_cumulative[key]) <= 0:
            raise CompactLongerHorizonLearningMetricsError(
                f"final cumulative denominator {key} must be > 0"
            )
    _validate_trend_summary(payload)
    evidence_ref = str(payload.get("evidence_ref", "")).strip()
    if evidence_ref != compact_longer_horizon_learning_metrics_evidence_ref(payload):
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics evidence_ref mismatch"
        )


def compact_longer_horizon_learning_metrics_evidence_ref(
    payload: Mapping[str, Any],
) -> str:
    """Return a compact Coach evidence ref for a validated metrics trace."""

    candidate = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    cumulative = _required_mapping(
        payload.get("cumulative_denominators"),
        "cumulative_denominators",
    )
    update_count = int(cumulative.get("learner_update_count_delta", 0))
    trend = _required_mapping(payload.get("trend_summary"), "trend_summary")
    return (
        f"{COMPACT_LONGER_HORIZON_LEARNING_METRICS_EVIDENCE_REF_PREFIX}"
        f"{candidate}:updates={update_count}:{_json_sha256(trend)[:16]}"
    )


def _normalize_checkpoint_point(
    raw_point: Mapping[str, Any],
    *,
    expected_index: int,
    eval_settings: Mapping[str, Any],
) -> dict[str, Any]:
    point = _plain_mapping(raw_point)
    point["checkpoint_index"] = int(point.get("checkpoint_index", expected_index))
    point.setdefault("checkpoint_step", point["checkpoint_index"])
    point.setdefault(
        "point_id",
        "pre_train"
        if point["checkpoint_index"] == 0
        else f"checkpoint_{point['checkpoint_index']}",
    )
    for prefix in ("compact_checkpoint", "stock_export", "eval_summary"):
        path = Path(_require_non_empty(point.get(f"{prefix}_path"), f"{prefix}_path"))
        if not path.is_file():
            raise CompactLongerHorizonLearningMetricsError(
                f"{prefix} path missing: {path}"
            )
        actual_hash = _file_sha256(path)
        provided_hash = str(point.get(f"{prefix}_sha256") or "").strip()
        if provided_hash and provided_hash != actual_hash:
            raise CompactLongerHorizonLearningMetricsError(
                f"{prefix} sha256 drift"
            )
        point[f"{prefix}_path"] = str(path.resolve())
        point[f"{prefix}_sha256"] = actual_hash

    eval_summary = _read_json_mapping(point["eval_summary_path"], "eval summary")
    eval_point = compact_matched_learning_quality_eval_point_from_summary_v1(
        eval_summary,
        point_id=str(point["point_id"]),
        checkpoint_step=int(point["checkpoint_step"]),
    )
    point["eval_point"] = eval_point
    point["eval_seed_set"] = _eval_seed_set_from_summary(
        eval_summary,
        fallback=eval_settings.get("eval_seed_set"),
    )
    point["eval_max_steps"] = _eval_max_steps_from_summary(
        eval_summary,
        fallback=eval_settings.get("eval_max_steps"),
    )
    point.setdefault("sample_source", COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE)
    point.setdefault("compact_training_entrypoint", COMPACT_RECORD_STEP_ENTRYPOINT)
    if point.get("training_metrics_lineage"):
        lineage = _plain_mapping(
            _required_mapping(
                point.get("training_metrics_lineage"),
                "training_metrics_lineage",
            )
        )
        validate_compact_training_metrics_lineage_v1(lineage)
        point["training_metrics_lineage"] = lineage
        point["training_metrics_lineage_ref"] = (
            compact_training_metrics_lineage_evidence_ref(lineage)
        )
        point.setdefault("search_provenance", lineage["search_provenance"])
    else:
        point.setdefault("training_metrics_lineage_ref", "")
    point.setdefault("learner_metrics", {})
    point.setdefault("loop_metrics", {})
    point.setdefault("search_provenance", {})
    point.setdefault("interval_denominators", {})
    point.setdefault("cumulative_denominators", {})
    point.setdefault("digest_changed_from_previous", point["checkpoint_index"] > 0)
    return point


def _validate_compatibility_input(compatibility: Mapping[str, Any]) -> None:
    if compatibility.get("schema_id") != COMPACT_COACH_COMPATIBILITY_REFRESH_SCHEMA_ID:
        raise CompactLongerHorizonLearningMetricsError(
            "compatibility refresh schema mismatch"
        )
    if compatibility.get("ok") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "compatibility refresh must be ok=true"
        )
    if compatibility.get("promotion_eligible") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics requires local compatibility eligibility"
        )
    if compatibility.get("coach_speed_row_gate") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "compatibility speed row gate missing"
        )
    for key in ("promotion_claim", "calls_train_muzero", "touches_live_runs"):
        if compatibility.get(key) is not False:
            raise CompactLongerHorizonLearningMetricsError(
                f"compatibility {key} must be false"
            )
    _require_non_empty(
        compatibility.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )


def _validate_lifecycle_input(
    lifecycle: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> str:
    if lifecycle.get("schema_id") != COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID:
        raise CompactLongerHorizonLearningMetricsError(
            "unified lifecycle schema mismatch"
        )
    if lifecycle.get("ok") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "unified lifecycle must be ok=true"
        )
    if lifecycle.get("lifecycle_gates_complete") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "unified lifecycle gates must be complete"
        )
    if lifecycle.get("promotion_claim") is not False:
        raise CompactLongerHorizonLearningMetricsError(
            "unified lifecycle must not claim promotion"
        )
    candidate = _require_non_empty(
        compatibility.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    lifecycle_candidate = _require_non_empty(
        lifecycle.get("checkpoint_id"),
        "unified lifecycle checkpoint_id",
    )
    if candidate != lifecycle_candidate:
        raise CompactLongerHorizonLearningMetricsError(
            "compatibility/lifecycle candidate checkpoint mismatch"
        )
    return candidate


def _validate_eval_settings(settings: Mapping[str, Any]) -> None:
    for key in REQUIRED_EVAL_SETTING_KEYS:
        if key not in settings:
            raise CompactLongerHorizonLearningMetricsError(
                f"eval_settings missing {key}"
            )
    seed_set = _seed_list(settings["eval_seed_set"], label="eval_settings eval_seed_set")
    if len(seed_set) < MIN_LONGER_HORIZON_EVAL_SEED_COUNT:
        raise CompactLongerHorizonLearningMetricsError(
            "eval_seed_set must contain at least two seeds"
        )
    if int(settings.get("eval_episode_count", 0)) != len(seed_set):
        raise CompactLongerHorizonLearningMetricsError(
            "eval_episode_count must match eval_seed_set"
        )
    if int(settings.get("eval_max_steps", 0)) < MIN_LONGER_HORIZON_EVAL_MAX_STEPS:
        raise CompactLongerHorizonLearningMetricsError(
            "eval_max_steps must be at least 128"
        )
    for key in ("num_simulations", "batch_size"):
        if int(settings.get(key, 0)) <= 0:
            raise CompactLongerHorizonLearningMetricsError(
                f"eval_settings {key} must be > 0"
            )
    if settings.get("death_mode") != "normal":
        raise CompactLongerHorizonLearningMetricsError(
            "eval_settings death_mode must be normal"
        )
    if settings.get("rnd_enabled") is not False:
        raise CompactLongerHorizonLearningMetricsError(
            "eval_settings rnd_enabled must be false"
        )
    for key in ("root_noise", "policy_noise"):
        if not _is_finite_number(settings.get(key)):
            raise CompactLongerHorizonLearningMetricsError(
                f"eval_settings non-finite {key}"
            )


def _validate_training_settings(settings: Mapping[str, Any]) -> None:
    if settings.get("route") != COMPACT_OWNED_TRAINER_ROUTE:
        raise CompactLongerHorizonLearningMetricsError(
            "training_settings route mismatch"
        )
    if settings.get("sample_source") != COMPACT_ENV_SEARCH_REPLAY_SAMPLE_SOURCE:
        raise CompactLongerHorizonLearningMetricsError(
            "training_settings requires compact_env_search_replay_rows"
        )
    if settings.get("compact_training_entrypoint") != COMPACT_RECORD_STEP_ENTRYPOINT:
        raise CompactLongerHorizonLearningMetricsError(
            "training_settings requires CompactOwnedTrainerV1.record_step"
        )
    for key in ("calls_train_muzero", "touches_live_runs", "training_speedup_claim"):
        if settings.get(key) is not False:
            raise CompactLongerHorizonLearningMetricsError(
                f"training_settings {key} must be false"
            )
    if settings.get("wall_sec_used_for_speed_claim") is not False:
        raise CompactLongerHorizonLearningMetricsError(
            "wall_sec_used_for_speed_claim must be false"
        )


def _validate_claims(payload: Mapping[str, Any]) -> None:
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    if claims.get("longer_horizon_compact_learning_metrics") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "attached_claims missing longer horizon metrics claim"
        )
    if claims.get("compact_multi_checkpoint_metrics_observed") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "attached_claims missing multi-checkpoint metrics claim"
        )
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for source_name, source in (("attached_claims", claims), ("non_claims", non_claims)):
        for key in NONCLAIM_FALSE_KEYS:
            if source.get(key) is not False:
                raise CompactLongerHorizonLearningMetricsError(
                    f"{source_name} {key} must be false"
                )
    readiness = _required_mapping(payload.get("readiness"), "readiness")
    if readiness.get("longer_horizon_compact_learning_metrics") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "readiness missing longer-horizon lane"
        )
    if readiness.get("promotion_readiness_complete") is not False:
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon metrics must not complete promotion readiness"
        )
    if readiness.get("final_promotion_bundle_still_required") is not True:
        raise CompactLongerHorizonLearningMetricsError(
            "final promotion bundle must remain required"
        )


def _validate_eval_point(
    point: Mapping[str, Any],
    eval_settings: Mapping[str, Any],
    *,
    index: int,
) -> None:
    label = f"checkpoint_series[{index}]"
    eval_point = _required_mapping(point.get("eval_point"), f"{label} eval_point")
    if int(eval_point.get("checkpoint_step", -1)) != int(point["checkpoint_step"]):
        raise CompactLongerHorizonLearningMetricsError(
            f"{label} eval checkpoint_step mismatch"
        )
    _validate_eval_checkpoint_binding(point, index=index)
    expected_seeds = sorted(
        _seed_list(eval_settings["eval_seed_set"], label="eval_settings eval_seed_set")
    )
    point_seeds = sorted(
        _seed_list(point.get("eval_seed_set"), label=f"{label} eval_seed_set")
    )
    if point_seeds != expected_seeds:
        raise CompactLongerHorizonLearningMetricsError(
            f"{label} eval_seed_set mismatch"
        )
    if int(point.get("eval_max_steps", -1)) != int(eval_settings["eval_max_steps"]):
        raise CompactLongerHorizonLearningMetricsError(
            f"{label} eval_max_steps mismatch"
        )
    if int(eval_point.get("eval_episode_count", 0)) != len(expected_seeds):
        raise CompactLongerHorizonLearningMetricsError(
            f"{label} eval_episode_count mismatch"
        )
    for key in (
        "mean_survival",
        "median_survival",
        "min_survival",
        "max_survival",
        "terminal_rate",
        "cap_rate",
    ):
        if not _is_finite_number(eval_point.get(key)):
            raise CompactLongerHorizonLearningMetricsError(
                f"{label} non-finite eval {key}"
            )
    if float(eval_point["cap_rate"]) > MAX_LONGER_HORIZON_EVAL_CAP_RATE:
        raise CompactLongerHorizonLearningMetricsError(
            f"{label} cap_rate exceeds saturation ceiling"
        )


def _validate_eval_checkpoint_binding(
    point: Mapping[str, Any],
    *,
    index: int,
) -> None:
    expected = Path(str(point["stock_export_path"])).resolve()
    refs = [str(_required_mapping(point["eval_point"], "eval_point")["checkpoint_ref"])]
    summary = _read_json_mapping(point["eval_summary_path"], "eval summary")
    if summary.get("checkpoint_ref") is not None:
        refs.append(str(summary["checkpoint_ref"]))
    config = summary.get("config")
    if isinstance(config, Mapping) and config.get("checkpoint_ref") is not None:
        refs.append(str(config["checkpoint_ref"]))
    for ref in refs:
        ref_path = Path(ref)
        resolved = ref_path.resolve() if ref_path.is_absolute() else (Path.cwd() / ref_path).resolve()
        if resolved == expected:
            return
    raise CompactLongerHorizonLearningMetricsError(
        f"checkpoint_series[{index}] eval checkpoint reference mismatch"
    )


def _validate_denominator_curve_point(
    point: Mapping[str, Any],
    *,
    previous_cumulative: Mapping[str, int] | None,
    index: int,
) -> dict[str, int]:
    label = f"checkpoint_series[{index}]"
    cumulative = _required_mapping(
        point.get("cumulative_denominators"),
        f"{label} cumulative_denominators",
    )
    interval = _required_mapping(
        point.get("interval_denominators"),
        f"{label} interval_denominators",
    )
    normalized: dict[str, int] = {}
    for key in REQUIRED_CUMULATIVE_DENOMINATOR_KEYS:
        value = int(cumulative.get(key, -1))
        if value < 0:
            raise CompactLongerHorizonLearningMetricsError(
                f"{label} cumulative {key} must be >= 0"
            )
        normalized[key] = value
        previous_value = 0 if previous_cumulative is None else int(previous_cumulative[key])
        if value < previous_value:
            raise CompactLongerHorizonLearningMetricsError(
                f"{label} cumulative {key} is non-monotonic"
            )
        interval_value = int(interval.get(key, value - previous_value))
        if interval_value != value - previous_value:
            raise CompactLongerHorizonLearningMetricsError(
                f"{label} interval {key} does not match cumulative delta"
            )
        if index > 0 and key in REQUIRED_INTERVAL_POSITIVE_KEYS and interval_value <= 0:
            raise CompactLongerHorizonLearningMetricsError(
                f"{label} interval {key} must be > 0"
            )
        if index == 0 and interval_value != 0:
            raise CompactLongerHorizonLearningMetricsError(
                f"{label} initial interval {key} must be zero"
            )
    return normalized


def _validate_trained_point(point: Mapping[str, Any], *, index: int) -> None:
    label = f"checkpoint_series[{index}]"
    lineage = _required_mapping(
        point.get("training_metrics_lineage"),
        f"{label} training_metrics_lineage",
    )
    validate_compact_training_metrics_lineage_v1(lineage)
    lineage_ref = _require_non_empty(
        point.get("training_metrics_lineage_ref"),
        f"{label} training_metrics_lineage_ref",
    )
    if lineage_ref != compact_training_metrics_lineage_evidence_ref(lineage):
        raise CompactLongerHorizonLearningMetricsError(
            f"{label} training metrics lineage ref mismatch"
        )
    learner_metrics = _required_mapping(
        point.get("learner_metrics"),
        f"{label} learner_metrics",
    )
    for key in REQUIRED_LEARNER_METRIC_KEYS:
        if not _is_finite_number(learner_metrics.get(key)):
            raise CompactLongerHorizonLearningMetricsError(
                f"{label} non-finite learner metric {key}"
            )
    for key in ("learner_sample_rows", "learner_train_steps"):
        if int(learner_metrics.get(key, 0)) <= 0:
            raise CompactLongerHorizonLearningMetricsError(
                f"{label} learner metric {key} must be > 0"
            )
    provenance = _required_mapping(
        point.get("search_provenance"),
        f"{label} search_provenance",
    )
    for key in REQUIRED_SEARCH_PROVENANCE_KEYS:
        _require_non_empty(provenance.get(key), f"{label} search_provenance {key}")
    if int(provenance.get("active_root_count", 0)) <= 0:
        raise CompactLongerHorizonLearningMetricsError(
            f"{label} search_provenance active_root_count must be > 0"
        )


def _validate_trend_summary(payload: Mapping[str, Any]) -> None:
    expected = _trend_summary(
        [
            _required_mapping(point, "checkpoint point")
            for point in _required_sequence(
                payload.get("checkpoint_series"),
                "checkpoint_series",
            )
        ]
    )
    if _json_sha256(payload.get("trend_summary")) != _json_sha256(expected):
        raise CompactLongerHorizonLearningMetricsError(
            "longer-horizon trend_summary mismatch"
        )


def _trend_summary(series: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not series:
        return {}
    means = [
        float(_required_mapping(point["eval_point"], "eval_point")["mean_survival"])
        for point in series
    ]
    capped = [
        float(_required_mapping(point["eval_point"], "eval_point")["cap_rate"])
        for point in series
    ]
    best_index = max(range(len(means)), key=lambda index: means[index])
    final_delta = means[-1] - means[0]
    return {
        "metric": "mean_survival",
        "checkpoint_count": len(series),
        "first_checkpoint_step": int(series[0]["checkpoint_step"]),
        "latest_checkpoint_step": int(series[-1]["checkpoint_step"]),
        "best_checkpoint_index": int(best_index),
        "first_mean_survival": means[0],
        "latest_mean_survival": means[-1],
        "best_mean_survival": means[best_index],
        "final_minus_first_mean_survival": final_delta,
        "movement_observed": not math.isclose(final_delta, 0.0, abs_tol=0.0),
        "max_cap_rate": max(capped),
        "saturation_ceiling": MAX_LONGER_HORIZON_EVAL_CAP_RATE,
        "saturated": max(capped) > MAX_LONGER_HORIZON_EVAL_CAP_RATE,
    }


def _artifact_refs(series: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for point in series:
        for prefix, kind in (
            ("compact_checkpoint", "compact_checkpoint"),
            ("stock_export", "stock_eval_export"),
            ("eval_summary", "eval_summary"),
        ):
            refs.append(
                {
                    "kind": kind,
                    "checkpoint_index": int(point["checkpoint_index"]),
                    "path": str(point[f"{prefix}_path"]),
                    "sha256": str(point[f"{prefix}_sha256"]),
                    "required": True,
                }
            )
    return refs


def _longer_horizon_non_claims() -> dict[str, bool]:
    non_claims = matched_learning_quality_non_claims_v1()
    non_claims.update(
        {
            "calls_train_muzero": False,
            "compact_calls_train_muzero": False,
            "public_leaderboard_claim": False,
            "promotion_published": False,
        }
    )
    return {key: False for key in NONCLAIM_FALSE_KEYS} | non_claims


def _validate_payload_file(payload: Mapping[str, Any], prefix: str) -> None:
    path = Path(_require_non_empty(payload.get(f"{prefix}_path"), f"{prefix}_path"))
    if not path.is_file():
        raise CompactLongerHorizonLearningMetricsError(
            f"{prefix} path missing: {path}"
        )
    if str(payload.get(f"{prefix}_sha256", "")) != _file_sha256(path):
        raise CompactLongerHorizonLearningMetricsError(f"{prefix} file drift")


def _validate_point_file(point: Mapping[str, Any], prefix: str) -> None:
    path = Path(_require_non_empty(point.get(f"{prefix}_path"), f"{prefix}_path"))
    if not path.is_file():
        raise CompactLongerHorizonLearningMetricsError(
            f"{prefix} path missing: {path}"
        )
    if str(point.get(f"{prefix}_sha256", "")) != _file_sha256(path):
        raise CompactLongerHorizonLearningMetricsError(f"{prefix} file drift")


def _eval_seed_set_from_summary(
    summary: Mapping[str, Any],
    *,
    fallback: Any,
) -> list[int]:
    direct = summary.get("eval_seed_set")
    if direct is not None:
        return _seed_list(direct, label="eval summary eval_seed_set")
    selection = summary.get("selection")
    if isinstance(selection, Mapping):
        values = selection.get("eval_seed_values") or selection.get("eval_seed_set")
        if values is not None:
            return _seed_list(values, label="eval summary selection eval seeds")
    table = summary.get("survival_table")
    if isinstance(table, Sequence) and not isinstance(table, (str, bytes)):
        seeds = [
            int(row["seed"])
            for row in table
            if isinstance(row, Mapping) and row.get("seed") is not None
        ]
        if seeds:
            return seeds
    return _seed_list(fallback, label="eval summary fallback eval_seed_set")


def _eval_max_steps_from_summary(summary: Mapping[str, Any], *, fallback: Any) -> int:
    for key in ("eval_max_steps", "max_eval_steps"):
        if summary.get(key) is not None:
            return int(summary[key])
    config = summary.get("config")
    if isinstance(config, Mapping):
        for key in ("eval_max_steps", "max_eval_steps"):
            if config.get(key) is not None:
                return int(config[key])
    return int(fallback)


def _seed_list(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CompactLongerHorizonLearningMetricsError(f"{label} must be a sequence")
    seeds = [int(seed) for seed in value]
    if len(seeds) != len(set(seeds)):
        raise CompactLongerHorizonLearningMetricsError(f"{label} must be unique")
    return seeds


def _read_json_mapping(path: str | Path, label: str) -> Mapping[str, Any]:
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"{label} not found: {path_obj}")
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise CompactLongerHorizonLearningMetricsError(
            f"{label} must be a JSON object: {path_obj}"
        )
    return payload


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactLongerHorizonLearningMetricsError(f"{label} must be a mapping")
    return value


def _required_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CompactLongerHorizonLearningMetricsError(f"{label} must be a sequence")
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactLongerHorizonLearningMetricsError(f"{label} must be non-empty")
    return text


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _plain_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(value) for key, value in dict(metadata).items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


__all__ = [
    "COMPACT_LONGER_HORIZON_LEARNING_METRICS_SCHEMA_ID",
    "COMPACT_LONGER_HORIZON_LEARNING_METRICS_STATUS",
    "CompactLongerHorizonLearningMetricsError",
    "build_compact_longer_horizon_learning_metrics_v1",
    "compact_longer_horizon_learning_metrics_evidence_ref",
    "validate_compact_longer_horizon_learning_metrics_v1",
]
