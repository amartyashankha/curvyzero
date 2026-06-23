"""Hash-bound execution manifest for the larger matched-quality study."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from curvyzero.training.compact_matched_quality_sufficiency_review import (
    COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_SCHEMA_ID,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    FALSE_CLAIM_KEYS,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    REQUIRED_DISALLOWED_SHORTCUT_KEYS,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    REQUIRED_STUDY_OUTPUT_KEYS,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    STATUS_LARGER_SAME_SURFACE_STUDY_REQUIRED,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    validate_compact_matched_quality_sufficiency_review_v1,
)
from curvyzero.training.compact_promotion_readiness_bundle_review import (
    COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_CANDIDATE_ROLE,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    COMPACT_MATCHED_LEARNING_QUALITY_PAIR_VERIFICATION_SCHEMA_ID,
)
from curvyzero.training.compact_promotion_readiness_learning_quality import (
    STOCK_REFERENCE_ROLE,
)


COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_SCHEMA_ID = (
    "curvyzero_compact_matched_quality_larger_study_bundle/v1"
)
COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_STATUS = (
    "larger_matched_quality_study_bundle_ready"
)
COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_EVIDENCE_REF_PREFIX = (
    "compact_matched_quality_larger_study_bundle:"
)

DEFAULT_SUFFICIENCY_REVIEW_REPORT = Path(
    "artifacts/local/curvytron_compact_promotion_readiness_results"
    "/optimizer-compact-matched-quality-sufficiency-review-20260531a"
    "/matched_quality_sufficiency_review_report.json"
)

STEP_ORDER = (
    "stock_reference_capture_producer",
    "compact_candidate_capture_producer",
    "matched_canary_builder",
    "matched_pair_verifier",
    "readiness_bundle_refresh",
    "sufficiency_review_update",
)
STEP_OUTPUT_KEYS = {
    "stock_reference_capture_producer": "stock_reference_capture",
    "compact_candidate_capture_producer": "compact_candidate_capture",
    "matched_canary_builder": "matched_learning_quality_canary_report",
    "matched_pair_verifier": "matched_pair_verification_report",
    "readiness_bundle_refresh": "refreshed_readiness_bundle_review",
    "sufficiency_review_update": "updated_sufficiency_review",
}
STEP_VALIDATORS = {
    "stock_reference_capture_producer": {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_SCHEMA_ID,
        "validator": (
            "compact_matched_learning_quality_arm_from_capture_v1"
            "(expected_role='stock_reference')"
        ),
        "expected_role": STOCK_REFERENCE_ROLE,
    },
    "compact_candidate_capture_producer": {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_CAPTURE_SCHEMA_ID,
        "validator": (
            "compact_matched_learning_quality_arm_from_capture_v1"
            "(expected_role='compact_candidate')"
        ),
        "expected_role": COMPACT_CANDIDATE_ROLE,
    },
    "matched_canary_builder": {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_CANARY_SCHEMA_ID,
        "validator": "validate_compact_matched_learning_quality_canary_v1",
    },
    "matched_pair_verifier": {
        "schema_id": COMPACT_MATCHED_LEARNING_QUALITY_PAIR_VERIFICATION_SCHEMA_ID,
        "validator": "validate_compact_matched_learning_quality_pair_verification_v1",
    },
    "readiness_bundle_refresh": {
        "schema_id": COMPACT_PROMOTION_READINESS_BUNDLE_REVIEW_SCHEMA_ID,
        "validator": "validate_compact_promotion_readiness_bundle_review_v1",
    },
    "sufficiency_review_update": {
        "schema_id": COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_SCHEMA_ID,
        "validator": "validate_compact_matched_quality_sufficiency_review_v1",
    },
}
TRUE_BUNDLE_CLAIM_KEYS = (
    "larger_matched_quality_study_bundle_manifest",
    "read_only_orchestration_plan",
    "source_sufficiency_review_hash_bound",
    "source_readiness_bundle_hash_bound",
    "same_surface_required",
    "fresh_outputs_required",
)
EXTRA_FALSE_CLAIM_KEYS = (
    "stock_train_muzero_speedup_claim",
    "quality_evidence_produced",
    "fresh_outputs_produced",
    "captures_produced",
    "matched_canary_refreshed",
    "matched_pair_verification_refreshed",
    "readiness_bundle_refreshed",
    "sufficiency_review_rerun",
)


class CompactMatchedQualityLargerStudyBundleError(ValueError):
    """Raised when a larger-study execution manifest would overclaim."""


def build_compact_matched_quality_larger_study_bundle_v1(
    *,
    run_id: str,
    sufficiency_review_report_path: str | Path = DEFAULT_SUFFICIENCY_REVIEW_REPORT,
    repo_root: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a read-only, hash-bound execution manifest for the larger study."""

    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    sufficiency_path = Path(sufficiency_review_report_path).resolve()
    sufficiency = _read_json_mapping(
        sufficiency_path,
        "matched-quality sufficiency review",
    )
    validate_compact_matched_quality_sufficiency_review_v1(sufficiency)
    _validate_source_sufficiency_decision(sufficiency)

    larger_plan = _required_mapping(sufficiency.get("larger_study_plan"), "larger_study_plan")
    input_reports = _validated_report_refs(
        _required_mapping(sufficiency.get("input_reports"), "sufficiency input_reports")
    )
    readiness_ref = _required_mapping(
        input_reports.get("readiness_bundle_review"),
        "readiness_bundle_review",
    )
    readiness_path = Path(_require_non_empty(readiness_ref.get("path"), "readiness path"))
    readiness = _read_json_mapping(readiness_path, "readiness bundle review")
    readiness_inputs = _validated_report_refs(
        _required_mapping(readiness.get("input_reports"), "readiness input_reports")
    )

    future_outputs = _required_mapping(larger_plan.get("required_outputs"), "required_outputs")
    output_status = _future_output_status(future_outputs, root=root)
    for key, status in output_status.items():
        if status["exists_at_manifest_time"]:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"larger-study expected output already exists: {key}"
            )

    ordered_steps = _ordered_steps(
        larger_plan,
        output_status=output_status,
    )
    non_claims = _false_claims()
    payload = {
        "schema_id": COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_SCHEMA_ID,
        "ok": True,
        "status": COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_STATUS,
        "run_id": _require_non_empty(run_id, "run_id"),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "repo_root_at_manifest_time": str(root),
        "candidate_checkpoint_id": sufficiency.get("candidate_checkpoint_id"),
        "source_sufficiency_review": {
            "path": str(sufficiency_path),
            "sha256": _file_sha256(sufficiency_path),
            "schema_id": sufficiency.get("schema_id"),
            "status": sufficiency.get("status"),
            "evidence_ref": sufficiency.get("evidence_ref"),
        },
        "source_readiness_bundle_review": readiness_ref,
        "source_input_reports": input_reports,
        "source_readiness_bundle_input_reports": readiness_inputs,
        "source_larger_study_plan_sha256": _json_sha256(larger_plan),
        "source_larger_study_plan": _jsonable(larger_plan),
        "execution_contract": {
            "plan_only_not_evidence": True,
            "commands_run_by_manifest": False,
            "fresh_outputs_produced": False,
            "captures_produced": False,
            "same_surface_required": True,
            "fresh_outputs_required": True,
            "promotion_claim": False,
            "automatic_promotion_allowed": False,
            "stock_train_muzero_speedup_claim": False,
            "ordered_step_keys": list(STEP_ORDER),
            "disallowed_shortcuts": _jsonable(
                _required_mapping(
                    larger_plan.get("disallowed_shortcuts"),
                    "disallowed_shortcuts",
                )
            ),
        },
        "current_evidence_summary": _jsonable(
            _required_mapping(
                sufficiency.get("current_evidence_summary"),
                "current_evidence_summary",
            )
        ),
        "minimum_scale_over_current": _jsonable(
            _required_mapping(
                larger_plan.get("minimum_scale_over_current"),
                "minimum_scale_over_current",
            )
        ),
        "matched_surface": _jsonable(
            _required_mapping(larger_plan.get("matched_surface"), "matched_surface")
        ),
        "future_required_outputs": _jsonable(future_outputs),
        "future_output_status": output_status,
        "ordered_steps": ordered_steps,
        "attached_claims": {key: True for key in TRUE_BUNDLE_CLAIM_KEYS} | non_claims,
        "non_claims": non_claims,
    }
    payload["evidence_ref"] = compact_matched_quality_larger_study_bundle_evidence_ref(
        payload
    )
    validate_compact_matched_quality_larger_study_bundle_v1(payload)
    return payload


def validate_compact_matched_quality_larger_study_bundle_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate a read-only larger-study execution manifest."""

    if payload.get("schema_id") != COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_SCHEMA_ID:
        raise CompactMatchedQualityLargerStudyBundleError(
            "larger-study bundle schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactMatchedQualityLargerStudyBundleError(
            "larger-study bundle must be ok=true"
        )
    if payload.get("status") != COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_STATUS:
        raise CompactMatchedQualityLargerStudyBundleError(
            "larger-study bundle status mismatch"
        )

    source = _required_mapping(
        payload.get("source_sufficiency_review"),
        "source_sufficiency_review",
    )
    source_path = Path(_require_non_empty(source.get("path"), "source path"))
    if not source_path.is_file():
        raise CompactMatchedQualityLargerStudyBundleError(
            "source sufficiency review missing"
        )
    if _file_sha256(source_path) != source.get("sha256"):
        raise CompactMatchedQualityLargerStudyBundleError(
            "source sufficiency review sha256 mismatch"
        )
    sufficiency = _read_json_mapping(source_path, "source sufficiency review")
    validate_compact_matched_quality_sufficiency_review_v1(sufficiency)
    _validate_source_sufficiency_decision(sufficiency)
    if source.get("schema_id") != sufficiency.get("schema_id"):
        raise CompactMatchedQualityLargerStudyBundleError(
            "source sufficiency schema mismatch"
        )
    if source.get("evidence_ref") != sufficiency.get("evidence_ref"):
        raise CompactMatchedQualityLargerStudyBundleError(
            "source sufficiency evidence_ref mismatch"
        )
    if payload.get("candidate_checkpoint_id") != sufficiency.get(
        "candidate_checkpoint_id"
    ):
        raise CompactMatchedQualityLargerStudyBundleError(
            "larger-study bundle candidate mismatch"
        )

    larger_plan = _required_mapping(sufficiency.get("larger_study_plan"), "larger_study_plan")
    if payload.get("source_larger_study_plan_sha256") != _json_sha256(larger_plan):
        raise CompactMatchedQualityLargerStudyBundleError(
            "source larger-study plan sha256 mismatch"
        )
    if _json_sha256(payload.get("source_larger_study_plan")) != _json_sha256(larger_plan):
        raise CompactMatchedQualityLargerStudyBundleError(
            "source larger-study plan payload drift"
        )

    _validate_report_refs(
        _required_mapping(payload.get("source_input_reports"), "source_input_reports")
    )
    _validate_report_refs(
        _required_mapping(
            payload.get("source_readiness_bundle_input_reports"),
            "source_readiness_bundle_input_reports",
        )
    )
    _validate_execution_contract(payload, larger_plan=larger_plan)
    _validate_steps(payload, larger_plan=larger_plan)
    _validate_claims(payload)

    expected_ref = compact_matched_quality_larger_study_bundle_evidence_ref(payload)
    if payload.get("evidence_ref") != expected_ref:
        raise CompactMatchedQualityLargerStudyBundleError(
            "larger-study bundle evidence_ref mismatch"
        )


def compact_matched_quality_larger_study_bundle_evidence_ref(
    payload: Mapping[str, Any],
) -> str:
    """Return a stable evidence ref for a larger-study execution manifest."""

    candidate = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    source = _required_mapping(
        payload.get("source_sufficiency_review"),
        "source_sufficiency_review",
    )
    digest_source = {
        "source_sufficiency_sha256": source.get("sha256"),
        "source_larger_study_plan_sha256": payload.get(
            "source_larger_study_plan_sha256"
        ),
        "ordered_steps": [
            {
                "step_key": step.get("step_key"),
                "run_id": step.get("run_id"),
                "argv": step.get("argv"),
                "expected_output_key": step.get("expected_output_key"),
                "expected_output_path": step.get("expected_output_path"),
            }
            for step in _required_sequence(payload.get("ordered_steps"), "ordered_steps")
        ],
        "future_required_outputs": payload.get("future_required_outputs"),
    }
    return (
        f"{COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_EVIDENCE_REF_PREFIX}"
        f"{candidate}:{_json_sha256(digest_source)[:16]}"
    )


def _validate_source_sufficiency_decision(sufficiency: Mapping[str, Any]) -> None:
    if sufficiency.get("schema_id") != COMPACT_MATCHED_QUALITY_SUFFICIENCY_REVIEW_SCHEMA_ID:
        raise CompactMatchedQualityLargerStudyBundleError(
            "source sufficiency schema mismatch"
        )
    if sufficiency.get("status") != STATUS_LARGER_SAME_SURFACE_STUDY_REQUIRED:
        raise CompactMatchedQualityLargerStudyBundleError(
            "source sufficiency status must require larger same-surface study"
        )
    decision = _required_mapping(sufficiency.get("decision"), "decision")
    if (
        decision.get("matched_quality_sufficiency_decision")
        != DECISION_REQUIRE_LARGER_SAME_SURFACE_STUDY
    ):
        raise CompactMatchedQualityLargerStudyBundleError(
            "source sufficiency decision must require larger same-surface study"
        )
    for key in (
        "current_evidence_sufficient_for_promotion",
        "promotion_claim",
        "automatic_promotion_allowed",
    ):
        if decision.get(key) is not False:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"source sufficiency decision {key} must be false"
            )
    if decision.get("larger_same_surface_study_required") is not True:
        raise CompactMatchedQualityLargerStudyBundleError(
            "source sufficiency decision must require larger study"
        )


def _ordered_steps(
    larger_plan: Mapping[str, Any],
    *,
    output_status: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    planned_runs = _required_mapping(larger_plan.get("planned_runs"), "planned_runs")
    outputs = _required_mapping(larger_plan.get("required_outputs"), "required_outputs")
    stock_evaluator_requirement = _stock_evaluator_requirement_from_plan(larger_plan)
    steps = []
    for index, key in enumerate(STEP_ORDER, start=1):
        run = _required_mapping(planned_runs.get(key), key)
        argv = _required_string_sequence(run.get("argv"), f"{key} argv")
        run_id = _require_non_empty(run.get("run_id"), f"{key} run_id")
        _require_argv_pair(argv, "--run-id", run_id, label=key)
        output_key = STEP_OUTPUT_KEYS[key]
        output_path = _require_non_empty(outputs.get(output_key), f"{output_key} output")
        status = _required_mapping(output_status.get(output_key), output_key)
        preflight_checks = {
            "exact_argv_materialized": True,
            "command_not_run_by_manifest": True,
            "same_surface_required": True,
            "fresh_output_required": True,
            "expected_output_absent_at_manifest_time": (
                status["exists_at_manifest_time"] is False
            ),
            "expected_output_under_artifacts_local": _is_artifacts_local_path(
                output_path
            ),
            "no_preview_output_path": "preview" not in output_path,
            "no_profile_only_speed_currency": True,
        }
        row = {
            "step_index": index,
            "step_key": key,
            "run_id": run_id,
            "argv": list(argv),
            "expected_output_key": output_key,
            "expected_output_path": output_path,
            "expected_output_abs_path_at_manifest_time": status[
                "abs_path_at_manifest_time"
            ],
            "preflight_checks": preflight_checks,
            "post_step_validator": STEP_VALIDATORS[key],
        }
        if key == "stock_reference_capture_producer":
            _require_argv_pair(
                argv,
                "--evaluator-env-num",
                str(stock_evaluator_requirement["evaluator_env_num"]),
                label=key,
            )
            _require_argv_pair(
                argv,
                "--n-evaluator-episode",
                str(stock_evaluator_requirement["n_evaluator_episode"]),
                label=key,
            )
            preflight_checks["stock_evaluator_requirement_explicit"] = True
            row["stock_evaluator_requirement"] = _jsonable(
                stock_evaluator_requirement
            )
        steps.append(row)
    return steps


def _future_output_status(
    outputs: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for key in REQUIRED_STUDY_OUTPUT_KEYS:
        output = _require_non_empty(outputs.get(key), f"{key} output")
        path = Path(output)
        abs_path = path if path.is_absolute() else root / path
        statuses[key] = {
            "path": output,
            "abs_path_at_manifest_time": str(abs_path.resolve()),
            "exists_at_manifest_time": abs_path.exists(),
            "requires_absent_before_execution": True,
        }
    return statuses


def _validated_report_refs(refs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, value in refs.items():
        ref = _required_mapping(value, str(key))
        path = Path(_require_non_empty(ref.get("path"), f"{key} path"))
        sha = _require_non_empty(ref.get("sha256"), f"{key} sha256")
        if not path.is_file():
            raise CompactMatchedQualityLargerStudyBundleError(
                f"{key} referenced report missing"
            )
        if _file_sha256(path) != sha:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"{key} referenced report sha256 mismatch"
            )
        row = {
            "path": str(path),
            "sha256": sha,
        }
        for optional in ("schema_id", "status", "evidence_ref", "candidate_checkpoint_id"):
            if optional in ref:
                row[optional] = ref[optional]
        out[str(key)] = row
    return out


def _validate_report_refs(refs: Mapping[str, Any]) -> None:
    _validated_report_refs(refs)


def _validate_execution_contract(
    payload: Mapping[str, Any],
    *,
    larger_plan: Mapping[str, Any],
) -> None:
    contract = _required_mapping(payload.get("execution_contract"), "execution_contract")
    required_true = (
        "plan_only_not_evidence",
        "same_surface_required",
        "fresh_outputs_required",
    )
    for key in required_true:
        if contract.get(key) is not True:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"execution contract {key} must be true"
            )
    for key in (
        "commands_run_by_manifest",
        "fresh_outputs_produced",
        "captures_produced",
        "promotion_claim",
        "automatic_promotion_allowed",
        "stock_train_muzero_speedup_claim",
    ):
        if contract.get(key) is not False:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"execution contract {key} must be false"
            )
    if tuple(contract.get("ordered_step_keys", ())) != STEP_ORDER:
        raise CompactMatchedQualityLargerStudyBundleError(
            "execution contract step order mismatch"
        )
    shortcuts = _required_mapping(contract.get("disallowed_shortcuts"), "shortcuts")
    source_shortcuts = _required_mapping(
        larger_plan.get("disallowed_shortcuts"),
        "source shortcuts",
    )
    if _json_sha256(shortcuts) != _json_sha256(source_shortcuts):
        raise CompactMatchedQualityLargerStudyBundleError(
            "execution contract shortcut drift"
        )
    for key in REQUIRED_DISALLOWED_SHORTCUT_KEYS:
        if shortcuts.get(key) is not True:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"execution contract shortcut missing: {key}"
            )


def _validate_steps(
    payload: Mapping[str, Any],
    *,
    larger_plan: Mapping[str, Any],
) -> None:
    ordered_steps = _required_sequence(payload.get("ordered_steps"), "ordered_steps")
    if len(ordered_steps) != len(STEP_ORDER):
        raise CompactMatchedQualityLargerStudyBundleError(
            "larger-study bundle step count mismatch"
        )
    planned_runs = _required_mapping(larger_plan.get("planned_runs"), "planned_runs")
    outputs = _required_mapping(larger_plan.get("required_outputs"), "required_outputs")
    for index, step in enumerate(ordered_steps, start=1):
        row = _required_mapping(step, f"step {index}")
        key = STEP_ORDER[index - 1]
        if row.get("step_index") != index or row.get("step_key") != key:
            raise CompactMatchedQualityLargerStudyBundleError(
                "larger-study bundle step order mismatch"
            )
        source_run = _required_mapping(planned_runs.get(key), key)
        if row.get("run_id") != source_run.get("run_id"):
            raise CompactMatchedQualityLargerStudyBundleError(
                f"{key} run_id drift"
            )
        if list(_required_sequence(row.get("argv"), f"{key} argv")) != list(
            _required_sequence(source_run.get("argv"), f"{key} source argv")
        ):
            raise CompactMatchedQualityLargerStudyBundleError(f"{key} argv drift")
        output_key = STEP_OUTPUT_KEYS[key]
        if row.get("expected_output_key") != output_key:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"{key} expected output key mismatch"
            )
        if row.get("expected_output_path") != outputs.get(output_key):
            raise CompactMatchedQualityLargerStudyBundleError(
                f"{key} expected output path mismatch"
            )
        preflight = _required_mapping(row.get("preflight_checks"), f"{key} preflight")
        for preflight_key in (
            "exact_argv_materialized",
            "command_not_run_by_manifest",
            "same_surface_required",
            "fresh_output_required",
            "expected_output_absent_at_manifest_time",
            "expected_output_under_artifacts_local",
            "no_preview_output_path",
            "no_profile_only_speed_currency",
        ):
            if preflight.get(preflight_key) is not True:
                raise CompactMatchedQualityLargerStudyBundleError(
                    f"{key} preflight {preflight_key} must be true"
                )
        if key == "stock_reference_capture_producer":
            stock_evaluator_requirement = _stock_evaluator_requirement_from_plan(
                larger_plan
            )
            if preflight.get("stock_evaluator_requirement_explicit") is not True:
                raise CompactMatchedQualityLargerStudyBundleError(
                    "stock evaluator requirement must be explicit"
                )
            row_requirement = _required_mapping(
                row.get("stock_evaluator_requirement"),
                "stock_evaluator_requirement",
            )
            if _json_sha256(row_requirement) != _json_sha256(
                stock_evaluator_requirement
            ):
                raise CompactMatchedQualityLargerStudyBundleError(
                    "stock evaluator requirement drift"
                )
            argv = _required_sequence(row.get("argv"), f"{key} argv")
            _require_argv_pair(
                argv,
                "--evaluator-env-num",
                str(stock_evaluator_requirement["evaluator_env_num"]),
                label=key,
            )
            _require_argv_pair(
                argv,
                "--n-evaluator-episode",
                str(stock_evaluator_requirement["n_evaluator_episode"]),
                label=key,
            )
        validator = _required_mapping(
            row.get("post_step_validator"),
            f"{key} post_step_validator",
        )
        if _json_sha256(validator) != _json_sha256(STEP_VALIDATORS[key]):
            raise CompactMatchedQualityLargerStudyBundleError(
                f"{key} post-step validator mismatch"
            )


def _validate_claims(payload: Mapping[str, Any]) -> None:
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    for key in TRUE_BUNDLE_CLAIM_KEYS:
        if claims.get(key) is not True:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"larger-study bundle claim {key} missing"
            )
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for key, value in _false_claims().items():
        if claims.get(key) is not False:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"larger-study bundle claim {key} must be false"
            )
        if non_claims.get(key) is not False:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"larger-study bundle non-claim {key} must be false"
            )
        if value is not False:
            raise CompactMatchedQualityLargerStudyBundleError(
                f"larger-study bundle internal non-claim {key} drift"
            )


def _false_claims() -> dict[str, bool]:
    keys = tuple(FALSE_CLAIM_KEYS) + EXTRA_FALSE_CLAIM_KEYS
    return {key: False for key in dict.fromkeys(keys)}


def _stock_evaluator_requirement_from_plan(
    larger_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    scale = _required_mapping(
        larger_plan.get("minimum_scale_over_current"),
        "minimum_scale_over_current",
    )
    eval_seed_count = int(scale.get("min_eval_seed_count", 0) or 0)
    requirement = _required_mapping(
        larger_plan.get("stock_evaluator_requirement"),
        "stock_evaluator_requirement",
    )
    expected = {
        "policy": "evalenv_full_episode_surface",
        "eval_seed_count": eval_seed_count,
        "evaluator_env_num": eval_seed_count,
        "n_evaluator_episode": eval_seed_count,
        "env_num_matches_episode_count_required": True,
        "empty_ready_set_workaround_formalized": True,
        "canonical_stock_evaluator_required": False,
    }
    for key, expected_value in expected.items():
        if requirement.get(key) != expected_value:
            raise CompactMatchedQualityLargerStudyBundleError(
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
        raise CompactMatchedQualityLargerStudyBundleError(
            f"{label} argv missing {flag}"
        ) from exc
    if index + 1 >= len(values) or values[index + 1] != str(expected):
        raise CompactMatchedQualityLargerStudyBundleError(
            f"{label} argv {flag} mismatch"
        )


def _is_artifacts_local_path(path: str) -> bool:
    text = str(path)
    return text.startswith("artifacts/local/") or "/artifacts/local/" in text


def _required_string_sequence(value: Any, label: str) -> tuple[str, ...]:
    seq = _required_sequence(value, label)
    out = tuple(_require_non_empty(item, f"{label} item") for item in seq)
    if not out:
        raise CompactMatchedQualityLargerStudyBundleError(f"{label} must not be empty")
    return out


def _required_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CompactMatchedQualityLargerStudyBundleError(
            f"{label} must be a sequence"
        )
    return value


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompactMatchedQualityLargerStudyBundleError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompactMatchedQualityLargerStudyBundleError(
            f"{label} is not JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CompactMatchedQualityLargerStudyBundleError(
            f"{label} must be a mapping"
        )
    return dict(payload)


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactMatchedQualityLargerStudyBundleError(
            f"{label} must be a mapping"
        )
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactMatchedQualityLargerStudyBundleError(
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
