#!/usr/bin/env python3
"""Attach an accepted compact Coach speed row to compatibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_GATE_COACH_SPEED_ROW,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_LIFECYCLE_GATES,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_REQUIRED_PROMOTION_GATES,
)
from curvyzero.training.compact_coach_compatibility import (
    COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
)
from curvyzero.training.compact_coach_compatibility import (
    build_compact_coach_compatibility_report_v1,
)
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT,
)
from curvyzero.training.compact_coach_speed_row import (
    COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID,
)
from curvyzero.training.compact_coach_speed_row import (
    compact_coach_speed_row_evidence_ref,
)
from curvyzero.training.compact_coach_speed_row import (
    validate_compact_coach_speed_row_evidence_v1,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/local/curvytron_compact_coach_compatibility_results")
DEFAULT_RUN_ID = "optimizer-compact-coach-compatibility-after-speed-row-h100-20260530"
REFRESH_SCHEMA_ID = "curvyzero_compact_coach_compatibility_refresh/v1"
SPEED_ROW_MODAL_SCHEMA_ID = "curvyzero_compact_coach_speed_row_modal_report/v1"
SPEED_CURRENCY = "compact_trainer_env_steps_per_sec"
DEFAULT_MIN_ENV_STEPS = 61_440.0
DEFAULT_MIN_TRAINING_WALL_SEC = 10.0
DEFAULT_MIN_STEPS_PER_SEC = 804.0


class CompactCoachSpeedRowRefreshError(ValueError):
    """Raised when speed-row evidence cannot safely close the compatibility gate."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--unified-lifecycle-report", type=Path, required=True)
    parser.add_argument("--speed-row-modal-report", type=Path, required=True)
    parser.add_argument("--min-env-steps", type=float, default=DEFAULT_MIN_ENV_STEPS)
    parser.add_argument(
        "--min-training-wall-sec",
        type=float,
        default=DEFAULT_MIN_TRAINING_WALL_SEC,
    )
    parser.add_argument("--min-steps-per-sec", type=float, default=DEFAULT_MIN_STEPS_PER_SEC)
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    output_dir = (repo_root / args.output_root / str(args.run_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_compact_coach_compatibility_speed_row_refresh_payload(
        run_id=str(args.run_id),
        unified_lifecycle_report_path=_resolve_path(
            args.unified_lifecycle_report,
            repo_root,
        ),
        speed_row_modal_report_path=_resolve_path(
            args.speed_row_modal_report,
            repo_root,
        ),
        min_env_steps=args.min_env_steps,
        min_training_wall_sec=args.min_training_wall_sec,
        min_steps_per_sec=args.min_steps_per_sec,
    )
    path = output_dir / "compatibility_report.json"
    _write_json(path, payload)
    print(json.dumps({"ok": True, "report_path": str(path)}, sort_keys=True))
    return 0


def build_compact_coach_compatibility_speed_row_refresh_payload(
    *,
    run_id: str,
    unified_lifecycle_report_path: str | Path,
    speed_row_modal_report_path: str | Path,
    min_env_steps: float = DEFAULT_MIN_ENV_STEPS,
    min_training_wall_sec: float = DEFAULT_MIN_TRAINING_WALL_SEC,
    min_steps_per_sec: float = DEFAULT_MIN_STEPS_PER_SEC,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed compatibility refresh payload from accepted speed evidence."""

    lifecycle_path = Path(unified_lifecycle_report_path).resolve()
    modal_report_path = Path(speed_row_modal_report_path).resolve()
    lifecycle = _read_json_mapping(lifecycle_path, "unified lifecycle report")
    modal_report = _read_json_mapping(modal_report_path, "speed-row modal report")
    evidence_path = _resolve_path(
        Path(str(_require_non_empty(modal_report.get("evidence_path"), "evidence_path"))),
        modal_report_path.parent,
    )
    speed_evidence = _read_json_mapping(evidence_path, "speed-row evidence")
    validate_compact_coach_speed_row_evidence_v1(speed_evidence)
    speed_evidence_ref = compact_coach_speed_row_evidence_ref(speed_evidence)

    _validate_modal_report(
        modal_report,
        modal_report_path=modal_report_path,
        evidence=speed_evidence,
        evidence_ref=speed_evidence_ref,
    )
    _validate_lifecycle_report(lifecycle, lifecycle_path=lifecycle_path)
    _validate_speed_evidence_targets_lifecycle(
        speed_evidence,
        lifecycle=lifecycle,
        lifecycle_path=lifecycle_path,
    )
    threshold_summary = _validate_thresholds(
        speed_evidence,
        min_env_steps=min_env_steps,
        min_training_wall_sec=min_training_wall_sec,
        min_steps_per_sec=min_steps_per_sec,
    )

    metadata = _required_mapping(
        lifecycle.get("compatibility_metadata"),
        "lifecycle compatibility_metadata",
    )
    gates = _gates_from_lifecycle_metadata(metadata)
    evidence_refs = _evidence_refs_from_lifecycle_metadata(metadata)
    gates[COMPACT_COACH_GATE_COACH_SPEED_ROW] = True
    evidence_refs[COMPACT_COACH_GATE_COACH_SPEED_ROW] = speed_evidence_ref

    report = build_compact_coach_compatibility_report_v1(
        route=COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER,
        profile_only=False,
        calls_train_muzero=False,
        touches_live_runs=False,
        speed_currency=SPEED_CURRENCY,
        gates=gates,
        evidence=evidence_refs,
        coach_speed_row_evidence=speed_evidence,
        promotion_claim=False,
    )
    compatibility_metadata = report.as_metadata()
    passed = [
        gate
        for gate in report.required_gates
        if gate not in report.missing_required_gates
        and gate not in report.missing_required_evidence
    ]
    model_identity = _required_mapping(speed_evidence.get("model_identity"), "model_identity")
    result_summary = _required_mapping(
        speed_evidence.get("result_summary"),
        "result_summary",
    )
    search_config = _required_mapping(speed_evidence.get("search_config"), "search_config")
    actor_handoff_config = _required_mapping(
        speed_evidence.get("actor_handoff_config"),
        "actor_handoff_config",
    )
    operational_surface = _required_mapping(
        speed_evidence.get("operational_surface"),
        "operational_surface",
    )

    return {
        "schema_id": REFRESH_SCHEMA_ID,
        "ok": True,
        "run_id": str(run_id),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "refresh_scope": "post_accepted_compact_coach_speed_row_h100",
        "route": report.route,
        "profile_only": report.profile_only,
        "calls_train_muzero": report.calls_train_muzero,
        "touches_live_runs": report.touches_live_runs,
        "candidate_checkpoint_id": str(speed_evidence["candidate_checkpoint_id"]),
        "unified_lifecycle_report_path": str(lifecycle_path),
        "unified_lifecycle_report_sha256": _file_sha256(lifecycle_path),
        "speed_row_modal_report_path": str(modal_report_path),
        "speed_row_modal_report_sha256": _file_sha256(modal_report_path),
        "speed_row_evidence_path": str(evidence_path),
        "speed_row_evidence_ref": speed_evidence_ref,
        "speed_currency": SPEED_CURRENCY,
        "env_steps_collected": threshold_summary["env_steps_collected"],
        "training_wall_sec": threshold_summary["training_wall_sec"],
        "steps_per_sec": threshold_summary["steps_per_sec"],
        "search_config": dict(search_config),
        "actor_handoff_config": dict(actor_handoff_config),
        "operational_surface": dict(operational_surface),
        "death_mode": operational_surface.get("death_mode"),
        "terminal_row_count": operational_surface.get("terminal_row_count"),
        "death_row_count": operational_surface.get("death_row_count"),
        "compact_torch_initial_inference_mode": search_config.get(
            "compact_torch_initial_inference_mode"
        ),
        "compact_torch_observation_memory_format": search_config.get(
            "compact_torch_observation_memory_format"
        ),
        "compact_torch_model_memory_format": search_config.get("compact_torch_model_memory_format"),
        "compact_torch_defer_one_simulation_replay_payload_requested": search_config.get(
            "compact_torch_defer_one_simulation_replay_payload_requested"
        ),
        "compact_torch_memory_format_applies_to_search_service": search_config.get(
            "compact_torch_memory_format_applies_to_search_service"
        ),
        "hybrid_persistent_compact_render_state_buffer": actor_handoff_config.get(
            "hybrid_persistent_compact_render_state_buffer"
        ),
        "hybrid_borrow_single_actor_render_state": actor_handoff_config.get(
            "hybrid_borrow_single_actor_render_state"
        ),
        "render_state_handoff_mode": actor_handoff_config.get("render_state_handoff_mode"),
        "render_state_copy_steps": actor_handoff_config.get("render_state_copy_steps"),
        "render_state_borrowed_steps": actor_handoff_config.get("render_state_borrowed_steps"),
        "learner_num_unroll_steps": operational_surface.get("learner_num_unroll_steps"),
        "compact_owned_loop_fused_learner_batch": operational_surface.get(
            "compact_owned_loop_fused_learner_batch"
        ),
        "compact_rollout_slab_sample_gate_sec": operational_surface.get(
            "compact_rollout_slab_sample_gate_sec"
        ),
        "compact_rollout_slab_learner_gate_sec": operational_surface.get(
            "compact_rollout_slab_learner_gate_sec"
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch": operational_surface.get(
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch"
        ),
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only": operational_surface.get(
            "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only"
        ),
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch": operational_surface.get(
            "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch"
        ),
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch": operational_surface.get(
            "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch"
        ),
        "compact_rollout_slab_learner_gate_prebuilt_batch_used": operational_surface.get(
            "compact_rollout_slab_learner_gate_prebuilt_batch_used"
        ),
        "compact_muzero_learner_prebuilt_batch_used": operational_surface.get(
            "compact_muzero_learner_prebuilt_batch_used"
        ),
        "thresholds": threshold_summary,
        "model_identity_scope": str(model_identity.get("scope")),
        "loaded_checkpoint_identity": model_identity.get(
            "result_loaded_checkpoint_identity",
            {},
        ),
        "result_summary": dict(result_summary),
        "compatibility_metadata": compatibility_metadata,
        "coach_speed_row_gate": True,
        "passed_required_gates": passed,
        "missing_required_gates": list(report.missing_required_gates),
        "missing_required_evidence": list(report.missing_required_evidence),
        "promotion_eligible": report.promotion_eligible,
        "promotion_blocker": report.promotion_blocker,
        "promotion_claim": report.promotion_claim,
        "selected_next_missing_gate": (
            report.missing_required_gates[0] if report.missing_required_gates else None
        ),
        "selected_next_missing_gate_reason": (
            "All required compact Coach compatibility gates are hash-bound to this "
            "candidate and the accepted H100 speed-row evidence. This is still not "
            "a promotion claim, live-run claim, stock-resume claim, or rating claim."
            if report.promotion_eligible
            else "Compatibility remains blocked by the listed missing gates/evidence."
        ),
        "evidence_refs": dict(evidence_refs),
        "non_claims": {
            "promotion_claim": False,
            "training_speedup_claim": False,
            "live_run_safety_claim": False,
            "stock_resume_claim": False,
            "rating_or_promotion_quality_claim": False,
            "base_lifecycle_report_mutated": False,
        },
    }


def _validate_modal_report(
    modal_report: Mapping[str, Any],
    *,
    modal_report_path: Path,
    evidence: Mapping[str, Any],
    evidence_ref: str,
) -> None:
    if modal_report.get("schema_id") != SPEED_ROW_MODAL_SCHEMA_ID:
        raise CompactCoachSpeedRowRefreshError("speed-row modal report schema mismatch")
    if modal_report.get("ok") is not True:
        raise CompactCoachSpeedRowRefreshError("speed-row modal report must be ok=true")
    for key, expected in (
        ("speed_currency", SPEED_CURRENCY),
        ("model_identity_scope", COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT),
    ):
        if str(modal_report.get(key, "")) != expected:
            if key == "model_identity_scope":
                raise CompactCoachSpeedRowRefreshError(
                    "speed-row modal report must prove loaded checkpoint identity"
                )
            raise CompactCoachSpeedRowRefreshError(f"speed-row modal report {key} mismatch")
    for key in ("calls_train_muzero", "touches_live_runs", "promotion_claim"):
        if modal_report.get(key) is not False:
            raise CompactCoachSpeedRowRefreshError(f"speed-row modal report {key} must be false")
    if modal_report.get("real_compact_owned_training_work") is not True:
        raise CompactCoachSpeedRowRefreshError(
            "speed-row modal report must prove real compact-owned training work"
        )
    if str(modal_report.get("evidence_ref", "")) != evidence_ref:
        raise CompactCoachSpeedRowRefreshError("speed-row modal report evidence_ref mismatch")
    if str(modal_report.get("candidate_checkpoint_id", "")) != str(
        evidence.get("candidate_checkpoint_id", "")
    ):
        raise CompactCoachSpeedRowRefreshError(
            "speed-row modal report candidate_checkpoint_id mismatch"
        )
    denominator = _required_mapping(evidence.get("denominator"), "denominator")
    _require_same_number(
        modal_report.get("env_steps_collected"),
        denominator.get("numerator_value"),
        "modal env_steps_collected",
    )
    _require_same_number(
        modal_report.get("training_wall_sec"),
        denominator.get("denominator_value_sec"),
        "modal training_wall_sec",
    )
    _require_same_number(
        modal_report.get("steps_per_sec"),
        denominator.get("reported_steps_per_sec"),
        "modal steps_per_sec",
    )
    search_config = _required_mapping(evidence.get("search_config"), "search_config")
    for key in (
        "compact_torch_initial_inference_mode",
        "compact_torch_observation_memory_format",
        "compact_torch_model_memory_format",
        "compact_torch_memory_format_applies_to_search_service",
    ):
        if modal_report.get(key) != search_config.get(key):
            raise CompactCoachSpeedRowRefreshError(f"speed-row modal report {key} mismatch")
    actor_handoff_config = _required_mapping(
        evidence.get("actor_handoff_config"),
        "actor_handoff_config",
    )
    for key in (
        "hybrid_persistent_compact_render_state_buffer",
        "hybrid_borrow_single_actor_render_state",
        "render_state_handoff_mode",
        "render_state_copy_steps",
        "render_state_borrowed_steps",
    ):
        if modal_report.get(key) != actor_handoff_config.get(key):
            raise CompactCoachSpeedRowRefreshError(f"speed-row modal report {key} mismatch")
    operational_surface = _required_mapping(
        evidence.get("operational_surface"),
        "operational_surface",
    )
    for key in (
        "learner_num_unroll_steps",
        "compact_owned_loop_fused_learner_batch",
        "compact_rollout_slab_sample_gate_sec",
        "compact_rollout_slab_learner_gate_sec",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch",
        "compact_rollout_slab_sample_gate_compact_muzero_learner_batch_only",
        "compact_rollout_slab_sample_gate_resident_grouped_device_learner_batch",
        "compact_rollout_slab_sample_gate_resident_grouped_device_direct_learner_batch",
        "compact_rollout_slab_learner_gate_prebuilt_batch_used",
        "compact_muzero_learner_prebuilt_batch_used",
    ):
        if key in operational_surface and modal_report.get(key) != operational_surface.get(key):
            raise CompactCoachSpeedRowRefreshError(
                f"speed-row modal report {key} mismatch"
            )


def _validate_lifecycle_report(
    lifecycle: Mapping[str, Any],
    *,
    lifecycle_path: Path,
) -> None:
    if lifecycle.get("schema_id") != COMPACT_UNIFIED_LIFECYCLE_SCHEMA_ID:
        raise CompactCoachSpeedRowRefreshError("unified lifecycle schema mismatch")
    if lifecycle.get("ok") is not True:
        raise CompactCoachSpeedRowRefreshError("unified lifecycle must be ok=true")
    if lifecycle.get("lifecycle_gates_complete") is not True:
        raise CompactCoachSpeedRowRefreshError(
            "unified lifecycle gates must be complete before speed-row refresh"
        )
    if list(lifecycle.get("missing_required_gates") or []) != [COMPACT_COACH_GATE_COACH_SPEED_ROW]:
        raise CompactCoachSpeedRowRefreshError(
            "unified lifecycle must be missing only coach_speed_row"
        )
    if list(lifecycle.get("missing_required_evidence") or []) != [
        COMPACT_COACH_GATE_COACH_SPEED_ROW
    ]:
        raise CompactCoachSpeedRowRefreshError(
            "unified lifecycle evidence must be missing only coach_speed_row"
        )
    if lifecycle.get("promotion_eligible") is not False:
        raise CompactCoachSpeedRowRefreshError(
            "base unified lifecycle must not already be promotion eligible"
        )
    metadata = _required_mapping(
        lifecycle.get("compatibility_metadata"),
        "lifecycle compatibility_metadata",
    )
    if str(metadata.get("compact_coach_compatibility_route", "")) != (
        COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER
    ):
        raise CompactCoachSpeedRowRefreshError("lifecycle compatibility route mismatch")
    if metadata.get("compact_coach_compatibility_profile_only") is not False:
        raise CompactCoachSpeedRowRefreshError("lifecycle compatibility must be non-profile")
    for key in (
        "compact_coach_compatibility_calls_train_muzero",
        "compact_coach_compatibility_touches_live_runs",
        "compact_coach_compatibility_promotion_claim",
        f"compact_coach_compatibility_gate_{COMPACT_COACH_GATE_COACH_SPEED_ROW}",
    ):
        if metadata.get(key) is not False:
            raise CompactCoachSpeedRowRefreshError(f"lifecycle {key} must be false")
    for gate in COMPACT_COACH_LIFECYCLE_GATES:
        gate_key = f"compact_coach_compatibility_gate_{gate}"
        if metadata.get(gate_key) is not True:
            raise CompactCoachSpeedRowRefreshError(f"lifecycle gate {gate} must already be true")
    if not lifecycle_path.is_file():
        raise CompactCoachSpeedRowRefreshError("unified lifecycle path missing")


def _validate_speed_evidence_targets_lifecycle(
    evidence: Mapping[str, Any],
    *,
    lifecycle: Mapping[str, Any],
    lifecycle_path: Path,
) -> None:
    if str(evidence.get("route", "")) != COMPACT_COACH_ROUTE_COMPACT_OWNED_TRAINER:
        raise CompactCoachSpeedRowRefreshError("speed-row route mismatch")
    if str(evidence.get("candidate_checkpoint_id", "")) != str(lifecycle.get("checkpoint_id", "")):
        raise CompactCoachSpeedRowRefreshError(
            "speed-row candidate does not match unified lifecycle checkpoint"
        )
    lifecycle_info = _required_mapping(
        evidence.get("unified_lifecycle"),
        "speed evidence unified_lifecycle",
    )
    if str(lifecycle_info.get("checkpoint_id", "")) != str(lifecycle.get("checkpoint_id", "")):
        raise CompactCoachSpeedRowRefreshError("speed-row evidence lifecycle checkpoint mismatch")
    if str(lifecycle_info.get("sha256", "")) != _file_sha256(lifecycle_path):
        raise CompactCoachSpeedRowRefreshError(
            "speed-row evidence lifecycle sha256 does not match provided lifecycle"
        )
    denominator = _required_mapping(evidence.get("denominator"), "denominator")
    if str(denominator.get("speed_currency", "")) != SPEED_CURRENCY:
        raise CompactCoachSpeedRowRefreshError("speed-row speed currency mismatch")
    model_identity = _required_mapping(evidence.get("model_identity"), "model_identity")
    if str(model_identity.get("scope", "")) != (
        COMPACT_COACH_MODEL_IDENTITY_SCOPE_LOADED_CHECKPOINT
    ):
        raise CompactCoachSpeedRowRefreshError(
            "speed-row evidence must prove loaded checkpoint identity"
        )
    metadata = _required_mapping(
        lifecycle.get("compatibility_metadata"),
        "lifecycle compatibility_metadata",
    )
    death_ref = str(
        _required_mapping(
            metadata.get("compact_coach_compatibility_evidence"),
            "lifecycle compatibility evidence",
        ).get("death_terminal_contract")
        or ""
    )
    if "normal_death=true" in death_ref:
        surface = _required_mapping(
            evidence.get("operational_surface"),
            "speed evidence operational_surface",
        )
        if str(surface.get("death_mode") or "") != "normal":
            raise CompactCoachSpeedRowRefreshError(
                "normal-death lifecycle requires normal-death speed-row evidence"
            )
        if surface.get("normal_death_terminal_contract_promotion_gate_satisfied") is not True:
            raise CompactCoachSpeedRowRefreshError(
                "normal-death speed-row evidence must carry the contract gate"
            )


def _validate_thresholds(
    evidence: Mapping[str, Any],
    *,
    min_env_steps: float,
    min_training_wall_sec: float,
    min_steps_per_sec: float,
) -> dict[str, float | bool]:
    denominator = _required_mapping(evidence.get("denominator"), "denominator")
    env_steps = _positive_float(denominator.get("numerator_value"), "env_steps_collected")
    wall_sec = _positive_float(
        denominator.get("denominator_value_sec"),
        "training_wall_sec",
    )
    steps_per_sec = _positive_float(
        denominator.get("reported_steps_per_sec"),
        "steps_per_sec",
    )
    thresholds = {
        "min_env_steps_collected": float(min_env_steps),
        "min_training_wall_sec": float(min_training_wall_sec),
        "min_steps_per_sec": float(min_steps_per_sec),
        "env_steps_collected": env_steps,
        "training_wall_sec": wall_sec,
        "steps_per_sec": steps_per_sec,
        "passed": True,
    }
    for key, value, minimum in (
        ("env_steps_collected", env_steps, min_env_steps),
        ("training_wall_sec", wall_sec, min_training_wall_sec),
        ("steps_per_sec", steps_per_sec, min_steps_per_sec),
    ):
        if value < float(minimum):
            raise CompactCoachSpeedRowRefreshError(
                f"speed-row threshold failed: {key}={value} < {float(minimum)}"
            )
    return thresholds


def _gates_from_lifecycle_metadata(metadata: Mapping[str, Any]) -> dict[str, bool]:
    gates = {
        gate: bool(metadata.get(f"compact_coach_compatibility_gate_{gate}", False))
        for gate in COMPACT_COACH_REQUIRED_PROMOTION_GATES
    }
    for gate in COMPACT_COACH_LIFECYCLE_GATES:
        if not gates[gate]:
            raise CompactCoachSpeedRowRefreshError(f"lifecycle metadata gate {gate} is not closed")
    if gates[COMPACT_COACH_GATE_COACH_SPEED_ROW]:
        raise CompactCoachSpeedRowRefreshError(
            "base lifecycle metadata already has coach_speed_row=true"
        )
    return gates


def _evidence_refs_from_lifecycle_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    evidence = {
        str(key): str(value)
        for key, value in dict(
            _required_mapping(
                metadata.get("compact_coach_compatibility_evidence"),
                "compact_coach_compatibility_evidence",
            )
        ).items()
    }
    for gate in COMPACT_COACH_LIFECYCLE_GATES:
        if not evidence.get(gate, "").strip():
            raise CompactCoachSpeedRowRefreshError(
                f"lifecycle metadata evidence for {gate} is missing"
            )
    if evidence.get(COMPACT_COACH_GATE_COACH_SPEED_ROW, "").strip():
        raise CompactCoachSpeedRowRefreshError(
            "base lifecycle metadata already has coach_speed_row evidence"
        )
    return evidence


def _resolve_path(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompactCoachSpeedRowRefreshError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CompactCoachSpeedRowRefreshError(f"{label} must be a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompactCoachSpeedRowRefreshError(f"{label} must be a mapping")
    return value


def _require_non_empty(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CompactCoachSpeedRowRefreshError(f"{label} must be non-empty")
    return text


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CompactCoachSpeedRowRefreshError(f"{label} must be numeric") from exc
    if not result > 0:
        raise CompactCoachSpeedRowRefreshError(f"{label} must be positive")
    return result


def _require_same_number(left: Any, right: Any, label: str) -> None:
    left_value = _positive_float(left, label)
    right_value = _positive_float(right, label)
    tolerance = max(1.0e-6, abs(right_value) * 1.0e-6)
    if abs(left_value - right_value) > tolerance:
        raise CompactCoachSpeedRowRefreshError(f"{label} mismatch")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
