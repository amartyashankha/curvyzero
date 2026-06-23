"""Hash-bound compile/eager speed-pair review for compact Torch rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.training.compact_speed_row_floor_bundle import DENOMINATOR_KEYS
from curvyzero.training.compact_speed_row_floor_bundle import FALSE_CLAIM_KEYS
from curvyzero.training.compact_speed_row_floor_bundle import SEARCH_KIND_COMPACT_TORCH
from curvyzero.training.compact_speed_row_floor_bundle import SPEED_CURRENCY
from curvyzero.training.compact_speed_row_floor_bundle import _denominator_value
from curvyzero.training.compact_speed_row_floor_bundle import _json_sha256
from curvyzero.training.compact_speed_row_floor_bundle import _load_speed_row
from curvyzero.training.compact_speed_row_floor_bundle import _read_json_mapping
from curvyzero.training.compact_speed_row_floor_bundle import _require_non_empty
from curvyzero.training.compact_speed_row_floor_bundle import _required_mapping
from curvyzero.training.compact_speed_row_floor_bundle import _sha256
from curvyzero.training.compact_speed_row_floor_bundle import _stable_json


COMPACT_COMPILE_EAGER_SPEED_PAIR_SCHEMA_ID = (
    "curvyzero_compact_compile_eager_speed_pair/v1"
)
COMPACT_COMPILE_EAGER_SPEED_PAIR_MANIFEST_SCHEMA_ID = (
    "curvyzero_compact_compile_eager_speed_pair_manifest/v1"
)
COMPACT_COMPILE_EAGER_SPEED_PAIR_EVIDENCE_REF_PREFIX = (
    "compact_compile_eager_speed_pair:"
)

STATUS_REVIEW_COMPLETE = "compact_compile_eager_speed_pair_review_complete"

DECISION_APPROVED = "approved_compile_faster_same_trajectory"
DECISION_INSUFFICIENT_PAIRS = "not_approved_insufficient_pairs"
DECISION_DENOMINATOR_MISMATCH = "not_approved_denominator_mismatch"
DECISION_SAFETY_FAILED = "not_approved_safety_failed"
DECISION_COMPILE_NOT_FASTER = "not_approved_compile_not_faster"
DECISION_TRAJECTORY_MISMATCH = "not_approved_action_trajectory_mismatch"

PAIR_DENOMINATOR_KEYS = (
    *DENOMINATOR_KEYS,
    "seed",
    "search_service_kind",
    "search_impl",
    "timing_mode",
    "action_feedback_mode",
    "model_state_digest",
)


@dataclass(frozen=True)
class CompileEagerPairInput:
    pair_id: str
    eager_report_path: str | Path
    compile_report_path: str | Path


class CompactCompileEagerSpeedPairError(ValueError):
    """Raised when a compile/eager speed-pair report is malformed."""


def build_compact_compile_eager_speed_pair_v1(
    *,
    run_id: str,
    pairs: Iterable[CompileEagerPairInput],
    created_at: str | None = None,
    min_pair_count: int = 2,
    min_wall_win_fraction: float = 0.05,
    require_action_trajectory_match: bool = True,
) -> dict[str, Any]:
    pair_reports = [
        _pair_report(pair, min_wall_win_fraction=float(min_wall_win_fraction))
        for pair in pairs
    ]
    aggregate = _aggregate(
        pair_reports,
        min_pair_count=int(min_pair_count),
        require_action_trajectory_match=bool(require_action_trajectory_match),
    )
    non_claims = _false_claims()
    payload: dict[str, Any] = {
        "schema_id": COMPACT_COMPILE_EAGER_SPEED_PAIR_SCHEMA_ID,
        "ok": True,
        "status": STATUS_REVIEW_COMPLETE,
        "run_id": _require_non_empty(run_id, "run_id"),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "min_pair_count": int(min_pair_count),
        "min_wall_win_fraction": float(min_wall_win_fraction),
        "require_action_trajectory_match": bool(require_action_trajectory_match),
        "pairs": pair_reports,
        "aggregate": aggregate,
        "non_claims": non_claims,
        "attached_claims": {
            "compile_eager_pair_review_hash_bound": True,
            "narrow_compile_speed_claim_allowed": bool(
                aggregate["speed_claim_allowed"]
            ),
            **non_claims,
        },
    }
    payload["evidence_ref"] = compact_compile_eager_speed_pair_evidence_ref(payload)
    validate_compact_compile_eager_speed_pair_v1(payload)
    return payload


def validate_compact_compile_eager_speed_pair_v1(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_id") != COMPACT_COMPILE_EAGER_SPEED_PAIR_SCHEMA_ID:
        raise CompactCompileEagerSpeedPairError("compile/eager schema mismatch")
    if payload.get("status") != STATUS_REVIEW_COMPLETE:
        raise CompactCompileEagerSpeedPairError("compile/eager status mismatch")
    _validate_false_claims(payload.get("non_claims"), "non_claims")
    _validate_false_claims(payload.get("attached_claims"), "attached_claims")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise CompactCompileEagerSpeedPairError("pairs must be a list")
    for index, pair in enumerate(pairs):
        _validate_pair(_required_mapping(pair, f"pairs[{index}]"))
    aggregate = _required_mapping(payload.get("aggregate"), "aggregate")
    decision = str(aggregate.get("decision") or "")
    if decision not in {
        DECISION_APPROVED,
        DECISION_INSUFFICIENT_PAIRS,
        DECISION_DENOMINATOR_MISMATCH,
        DECISION_SAFETY_FAILED,
        DECISION_COMPILE_NOT_FASTER,
        DECISION_TRAJECTORY_MISMATCH,
    }:
        raise CompactCompileEagerSpeedPairError("unknown compile/eager decision")
    speed_claim_allowed = bool(aggregate.get("speed_claim_allowed"))
    if speed_claim_allowed != (decision == DECISION_APPROVED):
        raise CompactCompileEagerSpeedPairError(
            "speed_claim_allowed must match approved decision"
        )
    if speed_claim_allowed:
        required_true = (
            "all_pairs_present",
            "all_same_denominator",
            "all_safety_checks_passed",
            "all_compile_faster",
            "all_action_trajectory_match",
        )
        for key in required_true:
            if aggregate.get(key) is not True:
                raise CompactCompileEagerSpeedPairError(
                    f"approved report requires aggregate.{key}"
                )
    expected_ref = compact_compile_eager_speed_pair_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactCompileEagerSpeedPairError("compile/eager evidence_ref mismatch")


def compact_compile_eager_speed_pair_evidence_ref(payload: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key != "evidence_ref"
    }
    run_id = _require_non_empty(payload.get("run_id"), "run_id")
    digest = _json_sha256(body)
    return f"{COMPACT_COMPILE_EAGER_SPEED_PAIR_EVIDENCE_REF_PREFIX}{run_id}:sha256={digest}"


def _pair_report(
    pair: CompileEagerPairInput,
    *,
    min_wall_win_fraction: float,
) -> dict[str, Any]:
    pair_id = _require_non_empty(pair.pair_id, "pair_id")
    eager = _row_entry("eager", pair.eager_report_path)
    compiled = _row_entry("compile", pair.compile_report_path)
    denominator_check = _pair_denominator_check(eager, compiled)
    eager_safety = _row_safety(eager, expect_compile=False)
    compile_safety = _row_safety(compiled, expect_compile=True)
    trajectory_check = _trajectory_check(eager, compiled)
    speed = _speed_check(
        eager,
        compiled,
        min_wall_win_fraction=float(min_wall_win_fraction),
    )
    return {
        "pair_id": pair_id,
        "eager": eager,
        "compile": compiled,
        "denominator_check": denominator_check,
        "safety_check": {
            "passed": bool(eager_safety["passed"] and compile_safety["passed"]),
            "eager": eager_safety,
            "compile": compile_safety,
        },
        "trajectory_check": trajectory_check,
        "speed_check": speed,
        "pair_passed_for_speed_claim": bool(
            denominator_check["same_denominator"]
            and eager_safety["passed"]
            and compile_safety["passed"]
            and speed["compile_faster_by_margin"]
            and trajectory_check["action_trajectory_match"]
        ),
    }


def _row_entry(role: str, report_path_raw: str | Path) -> dict[str, Any]:
    row = _load_speed_row(role, report_path_raw)
    result_ref = _required_mapping(row.get("result_ref"), f"{role}.result_ref")
    result_path = Path(str(result_ref["path"]))
    result = _read_json_mapping(result_path, f"{role} result")
    compact = _required_mapping(result.get("compact"), f"{role} compact")
    source_profile = _required_mapping(
        compact.get("source_profile_payload"),
        f"{role} source_profile_payload",
    )
    last_telemetry = _required_mapping(
        source_profile.get("compact_rollout_slab_last_telemetry"),
        f"{role} compact_rollout_slab_last_telemetry",
    )
    profile_telemetry = last_telemetry.get("compact_rollout_slab_profile_telemetry")
    profile_telemetry = profile_telemetry if isinstance(profile_telemetry, Mapping) else {}
    totals = source_profile.get("compact_rollout_slab_telemetry_totals")
    totals = totals if isinstance(totals, Mapping) else {}
    return {
        **row,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "seed": _nested_int(result, ("row", "seed")),
        "search_service_kind": str(
            row.get("search_service_kind") or row.get("search_impl_kind") or ""
        ),
        "timing_mode": str(
            profile_telemetry.get("compact_torch_search_service_timing_mode")
            or last_telemetry.get("compact_rollout_slab_search_service_timing_mode")
            or ""
        ),
        "action_feedback_mode": str(
            source_profile.get("compact_rollout_slab_action_mode") or ""
        ),
        "model_state_digest": str(
            profile_telemetry.get("compact_policy_refresh_model_state_digest") or ""
        ),
        "compile_telemetry": _compile_telemetry(profile_telemetry),
        "inference_guard_telemetry": _inference_guard_telemetry(profile_telemetry),
        "hotpath_budgets": _hotpath_budgets(row, source_profile, totals),
        "trajectory": _trajectory(source_profile),
        "source_profile_fingerprints": {
            "env_action_mask_checksum_total": source_profile.get(
                "env_action_mask_checksum_total"
            ),
            "env_done_checksum_total": source_profile.get("env_done_checksum_total"),
            "env_reward_checksum_total": source_profile.get(
                "env_reward_checksum_total"
            ),
        },
    }


def _compile_telemetry(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_compile_requested": bool(
            profile.get("compact_torch_search_model_compile_requested")
        ),
        "model_compile_used": bool(
            profile.get("compact_torch_search_model_compile_used")
        ),
        "model_compile_cache_hit": bool(
            profile.get("compact_torch_search_model_compile_cache_hit")
        ),
        "model_compile_mode": str(
            profile.get("compact_torch_search_model_compile_mode") or ""
        ),
        "model_compile_runtime_status": str(
            profile.get("compact_torch_search_model_compile_runtime_status") or ""
        ),
        "helper_compile_requested": bool(
            profile.get("compact_torch_search_compile_requested")
        ),
        "helper_compile_used": bool(profile.get("compact_torch_search_compile_used")),
        "helper_compile_mode": str(profile.get("compact_torch_search_compile_mode") or ""),
        "helper_compile_runtime_status": str(
            profile.get("compact_torch_search_compile_runtime_status") or ""
        ),
    }


def _inference_guard_telemetry(profile: Mapping[str, Any]) -> dict[str, Any]:
    before = profile.get("compact_torch_search_model_training_before_inference")
    after = profile.get("compact_torch_search_model_training_after_inference")
    observed = before is not None and after is not None
    return {
        "model_training_before_inference": before,
        "model_training_after_inference": after,
        "model_training_state_observed": observed,
        "model_training_state_restored": observed and bool(before) == bool(after),
        "model_eval_applied_for_inference": (
            profile.get("compact_torch_search_model_eval_applied_for_inference")
            is True
        ),
        "model_inference_mode_used": (
            profile.get("compact_torch_search_model_inference_mode_used") is True
        ),
    }


def _hotpath_budgets(
    row: Mapping[str, Any],
    source_profile: Mapping[str, Any],
    totals: Mapping[str, Any],
) -> dict[str, Any]:
    shape = _required_mapping(row.get("shape"), "shape")
    expected_action_d2h = int(shape["batch_size"]) * 2 * int(shape["steps"]) * 2
    return {
        "expected_action_d2h_bytes": expected_action_d2h,
        "action_d2h_bytes": _number(totals, "compact_rollout_slab_action_d2h_bytes"),
        "replay_payload_d2h_bytes": _number(
            totals,
            "compact_rollout_slab_replay_payload_d2h_bytes",
        ),
        "committed_replay_payload_d2h_bytes": _number(
            totals,
            "compact_rollout_slab_committed_replay_payload_d2h_bytes",
        ),
        "one_simulation_fast_path_count": _number(
            totals,
            "compact_rollout_slab_search_service_one_simulation_fast_path_count",
        ),
        "recurrent_inference_calls": _number(
            totals,
            "compact_rollout_slab_search_service_recurrent_inference_calls",
        ),
        "action_override_drop_count": _number(
            source_profile,
            "compact_rollout_slab_action_override_drop_count",
        ),
    }


def _trajectory(source_profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "env_action_checksum_total": source_profile.get("env_action_checksum_total"),
        "env_trajectory_checksum_total": source_profile.get(
            "env_trajectory_checksum_total"
        ),
        "last_env_action_checksum": source_profile.get("last_env_action_checksum"),
        "last_env_trajectory_checksum": source_profile.get(
            "last_env_trajectory_checksum"
        ),
        "terminal_row_count": source_profile.get("terminal_row_count"),
    }


def _pair_denominator_check(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> dict[str, Any]:
    values = {
        key: {
            "eager": _pair_denominator_value(eager, key),
            "compile": _pair_denominator_value(compiled, key),
        }
        for key in PAIR_DENOMINATOR_KEYS
    }
    mismatches = {
        key: value
        for key, value in values.items()
        if _stable_json(value["eager"]) != _stable_json(value["compile"])
    }
    return {
        "same_denominator": not mismatches,
        "checked_keys": list(PAIR_DENOMINATOR_KEYS),
        "values": values,
        "mismatches": mismatches,
    }


def _pair_denominator_value(row: Mapping[str, Any], key: str) -> Any:
    if key in DENOMINATOR_KEYS:
        return _denominator_value(row, key)
    return row.get(key)


def _row_safety(row: Mapping[str, Any], *, expect_compile: bool) -> dict[str, Any]:
    shape = _required_mapping(row.get("shape"), "shape")
    expected_steps = float(shape["steps"])
    compile_telemetry = _required_mapping(row.get("compile_telemetry"), "compile")
    inference_guard = _required_mapping(
        row.get("inference_guard_telemetry"),
        "inference_guard",
    )
    hotpath = _required_mapping(row.get("hotpath_budgets"), "hotpath")
    checks = {
        "speed_currency": row.get("speed_currency") == SPEED_CURRENCY,
        "search_kind": row.get("search_impl_kind") == SEARCH_KIND_COMPACT_TORCH,
        "timing_mode": row.get("timing_mode") == "host_phase_sync",
        "profile_only_false": row.get("profile_only") is False,
        "calls_train_muzero_false": row.get("calls_train_muzero") is False,
        "touches_live_runs_false": row.get("touches_live_runs") is False,
        "promotion_claim_false": row.get("promotion_claim") is False,
        "replay_d2h_zero": float(hotpath["replay_payload_d2h_bytes"]) == 0.0,
        "committed_replay_d2h_zero": (
            float(hotpath["committed_replay_payload_d2h_bytes"]) == 0.0
        ),
        "action_d2h_expected": float(hotpath["action_d2h_bytes"])
        == float(hotpath["expected_action_d2h_bytes"]),
        "fast_path_count_expected": (
            float(hotpath["one_simulation_fast_path_count"]) == expected_steps
        ),
        "recurrent_calls_expected": (
            float(hotpath["recurrent_inference_calls"]) == expected_steps
        ),
        "action_override_drop_zero": (
            float(hotpath["action_override_drop_count"]) == 0.0
        ),
        "model_training_state_observed": (
            inference_guard["model_training_state_observed"] is True
        ),
        "model_training_state_restored": (
            inference_guard["model_training_state_restored"] is True
        ),
        "model_eval_applied_for_inference": (
            inference_guard["model_eval_applied_for_inference"] is True
        ),
        "model_inference_mode_used": (
            inference_guard["model_inference_mode_used"] is True
        ),
    }
    if expect_compile:
        checks.update(
            {
                "model_compile_requested": (
                    compile_telemetry["model_compile_requested"] is True
                ),
                "model_compile_mode_default": (
                    compile_telemetry["model_compile_mode"] == "default"
                ),
                "model_compile_used": compile_telemetry["model_compile_used"] is True,
                "model_compile_cache_hit": (
                    compile_telemetry["model_compile_cache_hit"] is True
                ),
                "model_compile_runtime_cache_hit": (
                    compile_telemetry["model_compile_runtime_status"] == "cache_hit"
                ),
                "helper_compile_not_used": (
                    compile_telemetry["helper_compile_used"] is False
                ),
            }
        )
    else:
        checks.update(
            {
                "model_compile_not_requested": (
                    compile_telemetry["model_compile_requested"] is False
                ),
                "model_compile_not_used": (
                    compile_telemetry["model_compile_used"] is False
                ),
                "model_compile_runtime_not_requested": (
                    compile_telemetry["model_compile_runtime_status"]
                    == "not_requested"
                ),
            }
        )
    failed = [key for key, value in checks.items() if value is not True]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }


def _trajectory_check(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> dict[str, Any]:
    eager_trajectory = _required_mapping(eager.get("trajectory"), "eager trajectory")
    compile_trajectory = _required_mapping(
        compiled.get("trajectory"),
        "compile trajectory",
    )
    values = {
        key: {
            "eager": eager_trajectory.get(key),
            "compile": compile_trajectory.get(key),
        }
        for key in (
            "env_action_checksum_total",
            "env_trajectory_checksum_total",
            "last_env_action_checksum",
            "last_env_trajectory_checksum",
            "terminal_row_count",
        )
    }
    mismatches = {
        key: value
        for key, value in values.items()
        if _stable_json(value["eager"]) != _stable_json(value["compile"])
    }
    return {
        "action_trajectory_match": not mismatches,
        "checked_keys": list(values),
        "values": values,
        "mismatches": mismatches,
    }


def _speed_check(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
    *,
    min_wall_win_fraction: float,
) -> dict[str, Any]:
    eager_wall = float(eager["denominator_value_sec"])
    compile_wall = float(compiled["denominator_value_sec"])
    wall_win_fraction = (eager_wall - compile_wall) / eager_wall if eager_wall else 0.0
    eager_steps = float(eager["steps_per_sec"])
    compile_steps = float(compiled["steps_per_sec"])
    steps_win_fraction = (
        (compile_steps - eager_steps) / eager_steps if eager_steps else 0.0
    )
    return {
        "eager_wall_sec": eager_wall,
        "compile_wall_sec": compile_wall,
        "wall_delta_sec": compile_wall - eager_wall,
        "wall_win_fraction": wall_win_fraction,
        "eager_steps_per_sec": eager_steps,
        "compile_steps_per_sec": compile_steps,
        "steps_win_fraction": steps_win_fraction,
        "min_wall_win_fraction": float(min_wall_win_fraction),
        "compile_faster_by_margin": wall_win_fraction >= float(min_wall_win_fraction),
    }


def _aggregate(
    pairs: list[Mapping[str, Any]],
    *,
    min_pair_count: int,
    require_action_trajectory_match: bool,
) -> dict[str, Any]:
    all_pairs_present = len(pairs) >= int(min_pair_count)
    all_same_denominator = bool(pairs) and all(
        bool(pair["denominator_check"]["same_denominator"]) for pair in pairs
    )
    all_safety = bool(pairs) and all(
        bool(pair["safety_check"]["passed"]) for pair in pairs
    )
    all_compile_faster = bool(pairs) and all(
        bool(pair["speed_check"]["compile_faster_by_margin"]) for pair in pairs
    )
    all_trajectory = bool(pairs) and all(
        bool(pair["trajectory_check"]["action_trajectory_match"]) for pair in pairs
    )
    if not all_pairs_present:
        decision = DECISION_INSUFFICIENT_PAIRS
    elif not all_same_denominator:
        decision = DECISION_DENOMINATOR_MISMATCH
    elif not all_safety:
        decision = DECISION_SAFETY_FAILED
    elif not all_compile_faster:
        decision = DECISION_COMPILE_NOT_FASTER
    elif require_action_trajectory_match and not all_trajectory:
        decision = DECISION_TRAJECTORY_MISMATCH
    else:
        decision = DECISION_APPROVED
    wall_win_fractions = [
        float(pair["speed_check"]["wall_win_fraction"])
        for pair in pairs
    ]
    return {
        "decision": decision,
        "speed_claim_allowed": decision == DECISION_APPROVED,
        "pair_count": len(pairs),
        "all_pairs_present": all_pairs_present,
        "all_same_denominator": all_same_denominator,
        "all_safety_checks_passed": all_safety,
        "all_compile_faster": all_compile_faster,
        "all_action_trajectory_match": all_trajectory,
        "minimum_wall_win_fraction": min(wall_win_fractions)
        if wall_win_fractions
        else 0.0,
        "maximum_wall_win_fraction": max(wall_win_fractions)
        if wall_win_fractions
        else 0.0,
        "promotion_claim": False,
        "training_speedup_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }


def _validate_pair(pair: Mapping[str, Any]) -> None:
    _require_non_empty(pair.get("pair_id"), "pair_id")
    for row_key in ("eager", "compile"):
        row = _required_mapping(pair.get(row_key), row_key)
        for ref_key in ("report_ref", "result_ref", "evidence_ref"):
            ref = _required_mapping(row.get(ref_key), f"{row_key}.{ref_key}")
            path = Path(str(ref.get("path") or ""))
            if not path.is_file():
                raise CompactCompileEagerSpeedPairError(f"{row_key}.{ref_key} missing")
            if _sha256(path) != ref.get("sha256"):
                raise CompactCompileEagerSpeedPairError(
                    f"{row_key}.{ref_key} sha mismatch"
                )
    for section in ("denominator_check", "safety_check", "trajectory_check", "speed_check"):
        _required_mapping(pair.get(section), section)


def _validate_false_claims(value: Any, label: str) -> None:
    claims = _required_mapping(value, label)
    for key in FALSE_CLAIM_KEYS:
        if claims.get(key) is not False:
            raise CompactCompileEagerSpeedPairError(f"{label}.{key} must be false")


def _false_claims() -> dict[str, bool]:
    return {key: False for key in FALSE_CLAIM_KEYS}


def _number(mapping: Mapping[str, Any], key: str) -> float:
    try:
        return float(mapping.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _nested_int(mapping: Mapping[str, Any], path: tuple[str, ...]) -> int:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return 0
        value = value.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
