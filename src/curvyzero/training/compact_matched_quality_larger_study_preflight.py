"""Execution preflight for the larger matched-quality study bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from curvyzero.training.compact_matched_quality_larger_study_bundle import (
    COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_SCHEMA_ID,
)
from curvyzero.training.compact_matched_quality_larger_study_bundle import (
    EXTRA_FALSE_CLAIM_KEYS,
)
from curvyzero.training.compact_matched_quality_larger_study_bundle import (
    STEP_ORDER,
)
from curvyzero.training.compact_matched_quality_larger_study_bundle import (
    validate_compact_matched_quality_larger_study_bundle_v1,
)
from curvyzero.training.compact_matched_quality_sufficiency_review import (
    FALSE_CLAIM_KEYS,
)


COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_SCHEMA_ID = (
    "curvyzero_compact_matched_quality_larger_study_preflight/v1"
)
COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_STATUS_READY = (
    "larger_matched_quality_study_initial_steps_ready"
)
COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_STATUS_WAITING = (
    "larger_matched_quality_study_waiting_for_prior_outputs"
)
COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_EVIDENCE_REF_PREFIX = (
    "compact_matched_quality_larger_study_preflight:"
)

DEFAULT_BUNDLE_REPORT = Path(
    "artifacts/local/curvytron_compact_matched_quality_study_bundle_results"
    "/optimizer-compact-matched-quality-larger-study-bundle-canonical-20260531"
    "/larger_study_bundle_report.json"
)

PATH_FLAGS = frozenset(
    {
        "--compatibility-report",
        "--compact-candidate-capture",
        "--matched-learning-quality-report",
        "--matched-pair-verification-report",
        "--readiness-bundle-review",
        "--stock-resume-load-canary-report",
        "--isolated-live-run-safety-canary-report",
        "--sandbox-assignment-rating-proof-report",
        "--longer-horizon-learning-metrics-report",
        "--stock-reference-capture",
        "--unified-lifecycle-report",
    }
)
READINESS_REFRESH_REQUIRED_SOURCE_FLAGS = {
    "compatibility_refresh": "--compatibility-report",
    "unified_lifecycle": "--unified-lifecycle-report",
    "stock_resume_load_canary": "--stock-resume-load-canary-report",
    "isolated_live_run_safety_canary": (
        "--isolated-live-run-safety-canary-report"
    ),
    "sandbox_assignment_rating_proof": "--sandbox-assignment-rating-proof-report",
    "longer_horizon_compact_learning_metrics": (
        "--longer-horizon-learning-metrics-report"
    ),
}
STEP_PRIOR_OUTPUT_KEYS = {
    "stock_reference_capture_producer": (),
    "compact_candidate_capture_producer": (),
    "matched_canary_builder": (
        "stock_reference_capture",
        "compact_candidate_capture",
    ),
    "matched_pair_verifier": ("matched_learning_quality_canary_report",),
    "readiness_bundle_refresh": (
        "matched_learning_quality_canary_report",
        "matched_pair_verification_report",
    ),
    "sufficiency_review_update": ("refreshed_readiness_bundle_review",),
}
READINESS_REFRESH_DEFAULT_INPUT_KEYS = (
    "compatibility_refresh",
    "unified_lifecycle",
    "stock_resume_load_canary",
    "isolated_live_run_safety_canary",
    "sandbox_assignment_rating_proof",
    "longer_horizon_compact_learning_metrics",
)
TRUE_PREFLIGHT_CLAIM_KEYS = (
    "larger_study_execution_preflight",
    "source_bundle_hash_bound",
    "initial_producer_steps_identified",
    "plan_only_not_evidence",
)


class CompactMatchedQualityLargerStudyPreflightError(ValueError):
    """Raised when larger-study preflight input is malformed."""


def build_compact_matched_quality_larger_study_preflight_v1(
    *,
    run_id: str,
    bundle_report_path: str | Path = DEFAULT_BUNDLE_REPORT,
    repo_root: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a current-state preflight for a larger-study execution bundle."""

    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    bundle_path = Path(bundle_report_path).resolve()
    bundle = _read_json_mapping(bundle_path, "larger-study bundle report")
    validate_compact_matched_quality_larger_study_bundle_v1(bundle)
    if bundle.get("schema_id") != COMPACT_MATCHED_QUALITY_LARGER_STUDY_BUNDLE_SCHEMA_ID:
        raise CompactMatchedQualityLargerStudyPreflightError(
            "larger-study preflight bundle schema mismatch"
        )

    required_outputs = _required_mapping(
        bundle.get("future_required_outputs"),
        "future_required_outputs",
    )
    current_output_status = _current_output_status(required_outputs, root=root)
    steps = _step_preflights(
        bundle,
        current_output_status=current_output_status,
        root=root,
    )
    stock_evaluator_requirement_explicit = any(
        step.get("step_key") == "stock_reference_capture_producer"
        and _required_mapping(
            step.get("stock_evaluator_requirement_status"),
            "stock_evaluator_requirement_status",
        ).get("explicit") is True
        for step in steps
    )
    executable_now = [
        str(step["step_key"]) for step in steps if step["executable_now"] is True
    ]
    status = (
        COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_STATUS_READY
        if executable_now
        else COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_STATUS_WAITING
    )
    non_claims = _false_claims()
    payload = {
        "schema_id": COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_SCHEMA_ID,
        "ok": True,
        "status": status,
        "run_id": _require_non_empty(run_id, "run_id"),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "repo_root_at_preflight_time": str(root),
        "candidate_checkpoint_id": bundle.get("candidate_checkpoint_id"),
        "source_bundle_report": {
            "path": str(bundle_path),
            "sha256": _file_sha256(bundle_path),
            "schema_id": bundle.get("schema_id"),
            "status": bundle.get("status"),
            "evidence_ref": bundle.get("evidence_ref"),
        },
        "source_sufficiency_review": bundle.get("source_sufficiency_review"),
        "source_larger_study_plan_sha256": bundle.get(
            "source_larger_study_plan_sha256"
        ),
        "execution_readiness": {
            "commands_run_by_preflight": False,
            "quality_evidence_produced": False,
            "promotion_claim": False,
            "automatic_promotion_allowed": False,
            "stock_train_muzero_speedup_claim": False,
            "stock_evaluator_requirement_explicit": (
                stock_evaluator_requirement_explicit
            ),
            "initial_steps_executable_now": executable_now,
            "blocked_step_keys": [
                str(step["step_key"])
                for step in steps
                if step["executable_now"] is not True
            ],
            "recommended_first_wave": [
                key
                for key in (
                    "stock_reference_capture_producer",
                    "compact_candidate_capture_producer",
                )
                if key in executable_now
            ],
            "freeze_code_between_stock_and_compact_captures": True,
        },
        "current_output_status": current_output_status,
        "step_preflights": steps,
        "attached_claims": {key: True for key in TRUE_PREFLIGHT_CLAIM_KEYS}
        | non_claims,
        "non_claims": non_claims,
    }
    payload["evidence_ref"] = (
        compact_matched_quality_larger_study_preflight_evidence_ref(payload)
    )
    validate_compact_matched_quality_larger_study_preflight_v1(payload)
    return payload


def validate_compact_matched_quality_larger_study_preflight_v1(
    payload: Mapping[str, Any],
) -> None:
    """Validate a larger-study execution preflight report."""

    if payload.get("schema_id") != COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_SCHEMA_ID:
        raise CompactMatchedQualityLargerStudyPreflightError(
            "larger-study preflight schema mismatch"
        )
    if payload.get("ok") is not True:
        raise CompactMatchedQualityLargerStudyPreflightError(
            "larger-study preflight must be ok=true"
        )
    if payload.get("status") not in {
        COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_STATUS_READY,
        COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_STATUS_WAITING,
    }:
        raise CompactMatchedQualityLargerStudyPreflightError(
            "larger-study preflight status mismatch"
        )
    source = _required_mapping(payload.get("source_bundle_report"), "source bundle")
    bundle_path = Path(_require_non_empty(source.get("path"), "source bundle path"))
    if not bundle_path.is_file():
        raise CompactMatchedQualityLargerStudyPreflightError(
            "source bundle report missing"
        )
    if _file_sha256(bundle_path) != source.get("sha256"):
        raise CompactMatchedQualityLargerStudyPreflightError(
            "source bundle report sha256 mismatch"
        )
    bundle = _read_json_mapping(bundle_path, "source bundle report")
    validate_compact_matched_quality_larger_study_bundle_v1(bundle)
    if source.get("evidence_ref") != bundle.get("evidence_ref"):
        raise CompactMatchedQualityLargerStudyPreflightError(
            "source bundle evidence_ref mismatch"
        )
    if payload.get("candidate_checkpoint_id") != bundle.get("candidate_checkpoint_id"):
        raise CompactMatchedQualityLargerStudyPreflightError(
            "larger-study preflight candidate mismatch"
        )
    if payload.get("source_larger_study_plan_sha256") != bundle.get(
        "source_larger_study_plan_sha256"
    ):
        raise CompactMatchedQualityLargerStudyPreflightError(
            "larger-study preflight source plan sha mismatch"
        )
    _validate_execution_readiness(payload)
    _validate_step_preflights(payload, bundle=bundle)
    _validate_claims(payload)
    expected_ref = compact_matched_quality_larger_study_preflight_evidence_ref(
        payload
    )
    if payload.get("evidence_ref") != expected_ref:
        raise CompactMatchedQualityLargerStudyPreflightError(
            "larger-study preflight evidence_ref mismatch"
        )


def compact_matched_quality_larger_study_preflight_evidence_ref(
    payload: Mapping[str, Any],
) -> str:
    """Return a stable evidence ref for a larger-study preflight."""

    candidate = _require_non_empty(
        payload.get("candidate_checkpoint_id"),
        "candidate_checkpoint_id",
    )
    source = _required_mapping(payload.get("source_bundle_report"), "source bundle")
    digest_source = {
        "source_bundle_sha256": source.get("sha256"),
        "source_larger_study_plan_sha256": payload.get(
            "source_larger_study_plan_sha256"
        ),
        "execution_readiness": payload.get("execution_readiness"),
        "step_preflights": [
            {
                "step_key": step.get("step_key"),
                "run_id": step.get("run_id"),
                "executable_now": step.get("executable_now"),
                "expected_output_exists": step.get("expected_output_exists"),
                "missing_required_prior_outputs": step.get(
                    "missing_required_prior_outputs"
                ),
                "missing_existing_input_paths": step.get(
                    "missing_existing_input_paths"
                ),
            }
            for step in _required_sequence(
                payload.get("step_preflights"),
                "step_preflights",
            )
        ],
    }
    return (
        f"{COMPACT_MATCHED_QUALITY_LARGER_STUDY_PREFLIGHT_EVIDENCE_REF_PREFIX}"
        f"{candidate}:{_json_sha256(digest_source)[:16]}"
    )


def _step_preflights(
    bundle: Mapping[str, Any],
    *,
    current_output_status: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    bundle_steps = {
        str(step.get("step_key")): _required_mapping(step, "ordered step")
        for step in _required_sequence(bundle.get("ordered_steps"), "ordered_steps")
    }
    rows = []
    for index, key in enumerate(STEP_ORDER, start=1):
        step = _required_mapping(bundle_steps.get(key), key)
        argv = tuple(str(item) for item in _required_sequence(step.get("argv"), "argv"))
        output_key = _require_non_empty(
            step.get("expected_output_key"),
            f"{key} expected_output_key",
        )
        output_status = _required_mapping(current_output_status.get(output_key), output_key)
        required_prior = tuple(STEP_PRIOR_OUTPUT_KEYS[key])
        prior_status = {
            prior: _required_mapping(current_output_status.get(prior), prior)[
                "exists_now"
            ]
            for prior in required_prior
        }
        missing_prior = [prior for prior, exists in prior_status.items() if not exists]
        path_inputs = _argv_path_inputs(argv, root=root)
        future_paths = {
            str(_required_mapping(current_output_status.get(prior), prior)["path"])
            for prior in required_prior
        }
        missing_existing_inputs = [
            row
            for row in path_inputs
            if not row["exists_now"] and row["path"] not in future_paths
        ]
        implicit_inputs = (
            _readiness_refresh_implicit_inputs(bundle)
            if key == "readiness_bundle_refresh"
            else {}
        )
        missing_explicit_input_keys = (
            _readiness_refresh_missing_explicit_input_keys(argv)
            if key == "readiness_bundle_refresh"
            else []
        )
        if missing_explicit_input_keys:
            raise CompactMatchedQualityLargerStudyPreflightError(
                "readiness refresh would rely on implicit default inputs: "
                + ", ".join(missing_explicit_input_keys)
            )
        expected_output_exists = output_status["exists_now"] is True
        executable = (
            not expected_output_exists
            and not missing_prior
            and not missing_existing_inputs
            and _bool_from_mapping(step.get("preflight_checks"), "same_surface_required")
            and _bool_from_mapping(step.get("preflight_checks"), "fresh_output_required")
        )
        row = {
            "step_index": index,
            "step_key": key,
            "run_id": step.get("run_id"),
            "argv": list(argv),
            "expected_output_key": output_key,
            "expected_output_path": output_status["path"],
            "expected_output_abs_path": output_status["abs_path"],
            "expected_output_exists": expected_output_exists,
            "required_prior_output_keys": list(required_prior),
            "prior_output_status": prior_status,
            "missing_required_prior_outputs": missing_prior,
            "path_input_status": path_inputs,
            "missing_existing_input_paths": missing_existing_inputs,
            "implicit_default_input_refs": implicit_inputs,
            "missing_explicit_input_keys": missing_explicit_input_keys,
            "executable_now": executable,
            "run_mode": "ready_now" if executable else "wait_or_resolve_inputs",
        }
        if key == "stock_reference_capture_producer":
            row["stock_evaluator_requirement"] = _jsonable(
                _required_mapping(
                    step.get("stock_evaluator_requirement"),
                    "stock_evaluator_requirement",
                )
            )
            row["stock_evaluator_requirement_status"] = (
                _stock_evaluator_requirement_status(step)
            )
        rows.append(row)
    return rows


def _current_output_status(
    outputs: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for key, value in outputs.items():
        text = _require_non_empty(value, f"{key} output")
        path = Path(text)
        abs_path = path if path.is_absolute() else root / path
        statuses[str(key)] = {
            "path": text,
            "abs_path": str(abs_path.resolve()),
            "exists_now": abs_path.exists(),
            "requires_absent_before_execution": True,
        }
    return statuses


def _argv_path_inputs(argv: Sequence[str], *, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(argv[:-1]):
        if item not in PATH_FLAGS:
            continue
        text = str(argv[index + 1])
        path = Path(text)
        abs_path = path if path.is_absolute() else root / path
        rows.append(
            {
                "flag": item,
                "path": text,
                "abs_path": str(abs_path.resolve()),
                "exists_now": abs_path.exists(),
            }
        )
    return rows


def _readiness_refresh_implicit_inputs(bundle: Mapping[str, Any]) -> dict[str, Any]:
    refs = _required_mapping(
        bundle.get("source_readiness_bundle_input_reports"),
        "source_readiness_bundle_input_reports",
    )
    out: dict[str, Any] = {}
    for key in READINESS_REFRESH_DEFAULT_INPUT_KEYS:
        if key in refs:
            out[key] = refs[key]
    return out


def _readiness_refresh_missing_explicit_input_keys(argv: Sequence[str]) -> list[str]:
    flags = set(argv)
    return [
        key
        for key, flag in READINESS_REFRESH_REQUIRED_SOURCE_FLAGS.items()
        if flag not in flags
    ]


def _validate_execution_readiness(payload: Mapping[str, Any]) -> None:
    readiness = _required_mapping(
        payload.get("execution_readiness"),
        "execution_readiness",
    )
    for key in (
        "commands_run_by_preflight",
        "quality_evidence_produced",
        "promotion_claim",
        "automatic_promotion_allowed",
        "stock_train_muzero_speedup_claim",
    ):
        if readiness.get(key) is not False:
            raise CompactMatchedQualityLargerStudyPreflightError(
                f"execution readiness {key} must be false"
            )
    if readiness.get("freeze_code_between_stock_and_compact_captures") is not True:
        raise CompactMatchedQualityLargerStudyPreflightError(
            "execution readiness must require code freeze between captures"
        )
    if readiness.get("stock_evaluator_requirement_explicit") is not True:
        raise CompactMatchedQualityLargerStudyPreflightError(
            "execution readiness must require explicit stock evaluator surface"
        )
    recommended = _required_sequence(
        readiness.get("recommended_first_wave"),
        "recommended_first_wave",
    )
    for key in recommended:
        if key not in {
            "stock_reference_capture_producer",
            "compact_candidate_capture_producer",
        }:
            raise CompactMatchedQualityLargerStudyPreflightError(
                "recommended first wave contains non-producer step"
            )


def _validate_step_preflights(
    payload: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
) -> None:
    rows = _required_sequence(payload.get("step_preflights"), "step_preflights")
    if len(rows) != len(STEP_ORDER):
        raise CompactMatchedQualityLargerStudyPreflightError(
            "larger-study preflight step count mismatch"
        )
    source_steps = {
        str(step.get("step_key")): _required_mapping(step, "ordered step")
        for step in _required_sequence(bundle.get("ordered_steps"), "ordered_steps")
    }
    for index, row_any in enumerate(rows, start=1):
        row = _required_mapping(row_any, f"step preflight {index}")
        key = STEP_ORDER[index - 1]
        if row.get("step_index") != index or row.get("step_key") != key:
            raise CompactMatchedQualityLargerStudyPreflightError(
                "larger-study preflight step order mismatch"
            )
        source = _required_mapping(source_steps.get(key), key)
        if row.get("run_id") != source.get("run_id"):
            raise CompactMatchedQualityLargerStudyPreflightError(
                f"{key} run_id mismatch"
            )
        if list(_required_sequence(row.get("argv"), f"{key} argv")) != list(
            _required_sequence(source.get("argv"), f"{key} source argv")
        ):
            raise CompactMatchedQualityLargerStudyPreflightError(f"{key} argv drift")
        if row.get("required_prior_output_keys") != list(STEP_PRIOR_OUTPUT_KEYS[key]):
            raise CompactMatchedQualityLargerStudyPreflightError(
                f"{key} prior output keys mismatch"
            )
        if not isinstance(row.get("executable_now"), bool):
            raise CompactMatchedQualityLargerStudyPreflightError(
                f"{key} executable_now must be bool"
            )
        if row.get("expected_output_exists") is True and row.get("executable_now"):
            raise CompactMatchedQualityLargerStudyPreflightError(
                f"{key} cannot be executable when expected output exists"
            )
        if key == "readiness_bundle_refresh":
            missing_explicit = row.get("missing_explicit_input_keys")
            if missing_explicit not in ([], ()):
                raise CompactMatchedQualityLargerStudyPreflightError(
                    "readiness refresh would rely on implicit default inputs"
                )
        if key == "stock_reference_capture_producer":
            source_requirement = _required_mapping(
                source.get("stock_evaluator_requirement"),
                "source stock_evaluator_requirement",
            )
            row_requirement = _required_mapping(
                row.get("stock_evaluator_requirement"),
                "stock_evaluator_requirement",
            )
            if _json_sha256(row_requirement) != _json_sha256(source_requirement):
                raise CompactMatchedQualityLargerStudyPreflightError(
                    "stock evaluator requirement drift"
                )
            status = _required_mapping(
                row.get("stock_evaluator_requirement_status"),
                "stock_evaluator_requirement_status",
            )
            if status.get("explicit") is not True:
                raise CompactMatchedQualityLargerStudyPreflightError(
                    "stock evaluator requirement must be explicit"
                )
            if status.get("matches_source_requirement") is not True:
                raise CompactMatchedQualityLargerStudyPreflightError(
                    "stock evaluator requirement mismatch"
                )


def _validate_claims(payload: Mapping[str, Any]) -> None:
    claims = _required_mapping(payload.get("attached_claims"), "attached_claims")
    non_claims = _required_mapping(payload.get("non_claims"), "non_claims")
    for key in TRUE_PREFLIGHT_CLAIM_KEYS:
        if claims.get(key) is not True:
            raise CompactMatchedQualityLargerStudyPreflightError(
                f"larger-study preflight claim {key} missing"
            )
    for key in _false_claims():
        if claims.get(key) is not False:
            raise CompactMatchedQualityLargerStudyPreflightError(
                f"larger-study preflight claim {key} must be false"
            )
        if non_claims.get(key) is not False:
            raise CompactMatchedQualityLargerStudyPreflightError(
                f"larger-study preflight non-claim {key} must be false"
            )


def _false_claims() -> dict[str, bool]:
    return {key: False for key in dict.fromkeys(tuple(FALSE_CLAIM_KEYS) + EXTRA_FALSE_CLAIM_KEYS)}


def _stock_evaluator_requirement_status(step: Mapping[str, Any]) -> dict[str, Any]:
    key = str(step.get("step_key"))
    argv = tuple(str(item) for item in _required_sequence(step.get("argv"), "argv"))
    requirement = _required_mapping(
        step.get("stock_evaluator_requirement"),
        "stock_evaluator_requirement",
    )
    preflight = _required_mapping(step.get("preflight_checks"), f"{key} preflight")
    evaluator_env_num = int(
        _require_argv_value(argv, "--evaluator-env-num", label=key)
    )
    n_evaluator_episode = int(
        _require_argv_value(argv, "--n-evaluator-episode", label=key)
    )
    matches = (
        evaluator_env_num == int(requirement.get("evaluator_env_num", -1))
        and n_evaluator_episode == int(requirement.get("n_evaluator_episode", -1))
        and evaluator_env_num == n_evaluator_episode
    )
    if preflight.get("stock_evaluator_requirement_explicit") is not True:
        matches = False
    return {
        "explicit": preflight.get("stock_evaluator_requirement_explicit") is True,
        "policy": requirement.get("policy"),
        "eval_seed_count": int(requirement.get("eval_seed_count", 0) or 0),
        "evaluator_env_num": evaluator_env_num,
        "n_evaluator_episode": n_evaluator_episode,
        "matches_source_requirement": matches,
        "empty_ready_set_workaround_formalized": (
            requirement.get("empty_ready_set_workaround_formalized") is True
        ),
        "canonical_stock_evaluator_required": (
            requirement.get("canonical_stock_evaluator_required") is True
        ),
    }


def _require_argv_value(argv: Sequence[Any], flag: str, *, label: str) -> str:
    values = [str(item) for item in argv]
    try:
        index = values.index(flag)
    except ValueError as exc:
        raise CompactMatchedQualityLargerStudyPreflightError(
            f"{label} argv missing {flag}"
        ) from exc
    if index + 1 >= len(values):
        raise CompactMatchedQualityLargerStudyPreflightError(
            f"{label} argv {flag} missing value"
        )
    return values[index + 1]


def _bool_from_mapping(value: Any, key: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    return value.get(key) is True


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompactMatchedQualityLargerStudyPreflightError(
            f"{label} missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CompactMatchedQualityLargerStudyPreflightError(
            f"{label} is not JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CompactMatchedQualityLargerStudyPreflightError(
            f"{label} must be a mapping"
        )
    return dict(payload)


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactMatchedQualityLargerStudyPreflightError(
            f"{label} must be a mapping"
        )
    return value


def _required_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CompactMatchedQualityLargerStudyPreflightError(
            f"{label} must be a sequence"
        )
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactMatchedQualityLargerStudyPreflightError(
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
