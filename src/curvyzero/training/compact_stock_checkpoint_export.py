"""Stock-facing export payload for compact-owned trainer checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from curvyzero.training import lightzero_checkpoint_opponent_provider as provider
from curvyzero.env.observation_surface_contract import (
    DEFAULT_POLICY_OBSERVATION_BACKEND,
)
from curvyzero.env.observation_surface_contract import POLICY_BONUS_RENDER_MODE
from curvyzero.env.observation_surface_contract import POLICY_OBSERVATION_CONTRACT_ID
from curvyzero.env.observation_surface_contract import POLICY_OBSERVATION_PERSPECTIVE
from curvyzero.env.observation_surface_contract import (
    POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
)
from curvyzero.env.observation_surface_contract import POLICY_OBSERVATION_SEAT_MAPPING
from curvyzero.env.observation_surface_contract import POLICY_SINGLE_FRAME_SHAPE
from curvyzero.env.observation_surface_contract import POLICY_STACK_SHAPE
from curvyzero.env.observation_surface_contract import POLICY_TRAIL_RENDER_MODE
from curvyzero.env.observation_surface_contract import policy_observation_surface
from curvyzero.contracts.curvytron import CURVYTRON_DECISION_MS
from curvyzero.contracts.curvytron import CURVYTRON_DECISION_SOURCE_FRAMES
from curvyzero.contracts.curvytron import CURVYTRON_SOURCE_MAX_STEPS
from curvyzero.contracts.curvytron import DEFAULT_LEARNER_SEAT_MODE
from curvyzero.contracts.curvytron import (
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME,
)
from curvyzero.training.compact_trainer_checkpoint import CompactTrainerCheckpointV1
from curvyzero.training.compact_trainer_checkpoint import (
    validate_compact_trainer_checkpoint_v1,
)
from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    checkpoint_policy_metadata_sidecar_path,
)
from curvyzero.training.lightzero_checkpoint_opponent_provider import (
    require_checkpoint_policy_observation_metadata,
)
from curvyzero.training.reward_contracts import ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT


COMPACT_STOCK_CHECKPOINT_EXPORT_SCHEMA_ID = (
    "curvyzero_compact_stock_checkpoint_export/v1"
)
COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID = (
    "curvyzero_compact_stock_model_contract_verification/v1"
)
COMPACT_STOCK_EXPORT_EVIDENCE_BUNDLE_SCHEMA_ID = (
    "curvyzero_compact_stock_export_evidence_bundle/v1"
)
COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID = (
    "curvyzero_compact_stock_export_tournament_loader_smoke/v1"
)
CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID = "curvyzero_checkpoint_policy_metadata/v0"
COMPACT_STOCK_EXPORT_MODEL_STATE_KEY = "model"
REQUIRED_POLICY_OBSERVATION_METADATA_FIELDS = (
    "policy_trail_render_mode",
    "policy_bonus_render_mode",
    "policy_observation_backend",
    "policy_observation_contract_id",
    "policy_observation_perspective_schema_id",
)
REQUIRED_POLICY_OBSERVATION_CONTRACT_FIELDS = (
    "contract_id",
    "perspective_schema_id",
    "perspective",
    "seat_mapping",
    "trail_render_mode",
    "bonus_render_mode",
    "backend",
    "stack_shape",
    "single_frame_shape",
)
REQUIRED_STOCK_EXPORT_RUNTIME_METADATA_FIELDS = (
    "model_env_variant",
    "model_reward_variant",
    "env_variant",
    "reward_variant",
    "decision_source_frames",
    "source_physics_step_ms",
    "decision_ms",
    "source_max_steps",
    "source_max_steps_semantics",
    "learner_seat_mode",
)
DEFAULT_COMPACT_STOCK_EXPORT_ENV_VARIANT = ENV_VARIANT_SOURCE_STATE_FIXED_OPPONENT
DEFAULT_COMPACT_STOCK_EXPORT_REWARD_VARIANT = (
    REWARD_VARIANT_SURVIVAL_PLUS_BONUS_NO_OUTCOME
)
_PROTECTED_EXPORT_EXTRA_METADATA_PREFIXES = (
    "compact_coach_compatibility_",
    "compact_current_chain_eval_gif_tournament_load",
    "eval_gif_tournament_load",
)
_PROTECTED_EXPORT_EXTRA_METADATA_KEYS = frozenset(
    (
        "calls_train_muzero",
        "coach_integration_claim",
        "compact_eval_adapter_required",
        "lightzero_training_integration_claim",
        "optimizer_resume_supported",
        "promotion_claim",
        "stock_eval_tournament_load_status",
        "stock_eval_tournament_loadable",
        "stock_model_contract_verification_required",
        "stock_model_contract_verified",
        "stock_resume_claim",
        "touches_live_runs",
        "training_speed_claim",
    )
)


def build_current_policy_observation_sidecar_v1(
    *,
    model_env_variant: str = DEFAULT_COMPACT_STOCK_EXPORT_ENV_VARIANT,
    model_reward_variant: str = DEFAULT_COMPACT_STOCK_EXPORT_REWARD_VARIANT,
    env_variant: str | None = None,
    reward_variant: str | None = None,
    decision_source_frames: int = CURVYTRON_DECISION_SOURCE_FRAMES,
    source_physics_step_ms: float | None = None,
    source_max_steps: int = CURVYTRON_SOURCE_MAX_STEPS,
    source_max_steps_semantics: str = "source_physics_steps",
    learner_seat_mode: str = DEFAULT_LEARNER_SEAT_MODE,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stock policy-observation sidecar for current CurvyTron policy."""

    frames = int(decision_source_frames)
    if frames <= 0:
        raise ValueError("decision_source_frames must be positive")
    physics_ms = (
        float(CURVYTRON_DECISION_MS) / float(CURVYTRON_DECISION_SOURCE_FRAMES)
        if source_physics_step_ms is None
        else float(source_physics_step_ms)
    )
    if physics_ms <= 0.0:
        raise ValueError("source_physics_step_ms must be positive")
    decision_ms = float(frames) * float(physics_ms)
    model_env = str(model_env_variant)
    model_reward = str(model_reward_variant)
    env = str(env_variant or model_env)
    reward = str(reward_variant or model_reward)
    observation_contract = policy_observation_surface(
        trail_render_mode=POLICY_TRAIL_RENDER_MODE,
        bonus_render_mode=POLICY_BONUS_RENDER_MODE,
        backend=DEFAULT_POLICY_OBSERVATION_BACKEND,
    )
    sidecar: dict[str, Any] = {
        "schema_id": CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID,
        "policy_trail_render_mode": POLICY_TRAIL_RENDER_MODE,
        "policy_bonus_render_mode": POLICY_BONUS_RENDER_MODE,
        "policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
        "policy_observation_perspective_schema_id": (
            POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
        ),
        "observation_contract": observation_contract,
        "model_env_variant": model_env,
        "model_reward_variant": model_reward,
        "env_variant": env,
        "reward_variant": reward,
        "decision_source_frames": frames,
        "source_physics_step_ms": physics_ms,
        "decision_ms": decision_ms,
        "source_max_steps": int(source_max_steps),
        "source_max_steps_semantics": str(source_max_steps_semantics),
        "learner_seat_mode": str(learner_seat_mode),
    }
    sidecar.update(_plain_mapping(extra_metadata or {}))
    validate_policy_observation_sidecar_v1(sidecar)
    return sidecar


def build_compact_stock_export_payload_v1(
    checkpoint: CompactTrainerCheckpointV1,
    *,
    policy_metadata: Mapping[str, Any],
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a mapping payload visible to stock state-dict discovery.

    This exposes the compact model under the stock ``model`` key but does not
    claim stock eval/tournament loadability unless a separate model-contract
    verifier has been run.
    """

    validate_compact_trainer_checkpoint_v1(checkpoint)
    _validate_tensor_state_dict(checkpoint.model_state_dict)
    sidecar = dict(policy_metadata)
    validate_policy_observation_sidecar_v1(sidecar)
    metadata: dict[str, Any] = {
        "schema_id": COMPACT_STOCK_CHECKPOINT_EXPORT_SCHEMA_ID,
        "compact_stock_checkpoint_export_schema_id": (
            COMPACT_STOCK_CHECKPOINT_EXPORT_SCHEMA_ID
        ),
        "compact_stock_checkpoint_export": True,
        "eval_only_export": True,
        "source_compact_checkpoint_schema_id": checkpoint.metadata[
            "compact_trainer_checkpoint_schema_id"
        ],
        "source_compact_checkpoint_id": checkpoint.metadata["checkpoint_id"],
        "source_trainer_id": checkpoint.metadata["trainer_id"],
        "policy_version_ref": checkpoint.metadata["policy_version_ref"],
        "model_version_ref": checkpoint.metadata["model_version_ref"],
        "policy_source": checkpoint.metadata["policy_source"],
        "stock_model_state_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        "stock_payload_mapping": True,
        "stock_state_dict_discovery_key": COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
        "stock_state_dict_discovery_verified": True,
        "stock_policy_observation_sidecar_required": True,
        "stock_policy_observation_sidecar_schema_id": (
            CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID
        ),
        "stock_model_contract_verified": False,
        "stock_model_contract_verification_required": True,
        "stock_eval_tournament_loadable": False,
        "stock_eval_tournament_load_status": "strict_stock_model_load_not_run",
        "stock_resume_claim": False,
        "optimizer_resume_supported": False,
        "stock_optimizer_state_in_payload": False,
        "compact_replay_state_in_payload": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
        "training_speed_claim": False,
        **{key: sidecar[key] for key in REQUIRED_POLICY_OBSERVATION_METADATA_FIELDS},
        "observation_contract": sidecar["observation_contract"],
    }
    for key in (
        "model_env_variant",
        "model_reward_variant",
        "env_variant",
        "reward_variant",
        "decision_source_frames",
        "source_physics_step_ms",
        "decision_ms",
        "source_max_steps",
        "source_max_steps_semantics",
        "learner_seat_mode",
    ):
        if key in sidecar:
            metadata[key] = sidecar[key]
    extra_metadata_dict = _plain_mapping(extra_metadata or {})
    _validate_export_extra_metadata(extra_metadata_dict)
    metadata.update(extra_metadata_dict)
    return {
        COMPACT_STOCK_EXPORT_MODEL_STATE_KEY: checkpoint.model_state_dict,
        "metadata": metadata,
        "compact_trainer_checkpoint_metadata": dict(checkpoint.metadata),
    }


def save_compact_stock_export_v1(
    checkpoint: CompactTrainerCheckpointV1,
    path: str | Path,
    *,
    policy_metadata: Mapping[str, Any],
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write stock-facing payload and adjacent policy-observation sidecar."""

    import torch

    payload = build_compact_stock_export_payload_v1(
        checkpoint,
        policy_metadata=policy_metadata,
        extra_metadata=extra_metadata,
    )
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path_obj)
    sidecar_path = checkpoint_policy_metadata_sidecar_path(path_obj)
    sidecar_path.write_text(
        json.dumps(_plain_mapping(policy_metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    require_checkpoint_policy_observation_metadata(
        checkpoint_path=path_obj,
        checkpoint_payload=payload,
    )
    return {
        "payload": payload,
        "checkpoint_path": path_obj,
        "sidecar_path": sidecar_path,
    }


def compact_stock_export_evidence_bundle_path(path: str | Path) -> Path:
    """Return the sibling evidence-bundle path for a stock-facing export."""

    return Path(f"{Path(path)}.evidence.json")


def build_compact_stock_export_evidence_bundle_v1(
    checkpoint_path: str | Path,
    *,
    verification_report_path: str | Path,
    tournament_loader_report_path: str | Path | None = None,
    require_tournament_loader_report: bool = True,
) -> dict[str, Any]:
    """Build an attachment-only evidence bundle for one compact stock export."""

    checkpoint = Path(checkpoint_path)
    sidecar = checkpoint_policy_metadata_sidecar_path(checkpoint)
    verifier = Path(verification_report_path)
    loader = Path(tournament_loader_report_path) if tournament_loader_report_path else None
    if require_tournament_loader_report and loader is None:
        raise ValueError("compact stock export evidence bundle requires loader report")

    payload = _torch_load_export_payload(checkpoint)
    metadata = _validate_export_payload_for_evidence(
        checkpoint_path=checkpoint,
        payload=payload,
    )
    sidecar_payload = _read_json_mapping(sidecar, label="policy metadata sidecar")
    validate_policy_observation_sidecar_v1(sidecar_payload)
    _require_export_sidecar_consistent(metadata=metadata, sidecar=sidecar_payload)

    verifier_payload = _read_json_mapping(verifier, label="verification report")
    _validate_verification_report_for_evidence(
        report=verifier_payload,
        checkpoint_path=checkpoint,
        sidecar_path=sidecar,
        export_metadata=metadata,
    )

    loader_payload: Mapping[str, Any] | None = None
    if loader is not None:
        loader_payload = _read_json_mapping(loader, label="tournament loader report")
        _validate_loader_report_for_evidence(
            report=loader_payload,
            checkpoint_path=checkpoint,
            export_metadata=metadata,
            sidecar=sidecar_payload,
        )

    files = {
        "checkpoint": _file_record(checkpoint, required=True),
        "policy_metadata_sidecar": {
            **_file_record(sidecar, required=True),
            "schema_id": CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID,
        },
        "verification_report": {
            **_file_record(verifier, required=True),
            "schema_id": COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID,
        },
        "tournament_loader_report": (
            {
                **_file_record(loader, required=True),
                "schema_id": COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID,
            }
            if loader is not None
            else {"path": None, "required": False}
        ),
    }
    state_key = str(metadata["stock_model_state_key"])
    bundle = {
        "schema_id": COMPACT_STOCK_EXPORT_EVIDENCE_BUNDLE_SCHEMA_ID,
        "compact_stock_export_evidence_bundle": True,
        "ok": True,
        "evidence_scope": "attachment_only_no_base_export_mutation",
        "base_export_claims_mutated": False,
        "export_identity": _export_identity_for_evidence(metadata),
        "files": files,
        "attached_claims": {
            "stock_model_contract_verified_by_report": True,
            "strict_stock_model_load_verified": True,
            "tournament_loader_constructed": loader_payload is not None,
            "stock_eval_tournament_loadable_by_evidence": False,
            "eval_gif_tournament_loadable_by_evidence": False,
            "promotion_claim": False,
            "training_speed_claim": False,
            "optimizer_resume_claim": False,
            "base_export_claims_mutated": False,
        },
        "verification_report_summary": {
            "state_key": state_key,
            "strict_load": True,
            "candidate": _plain_value(
                dict(verifier_payload.get("load_summary", {})).get("candidate")
            ),
            "checkpoint_inferred_model_support_config": _plain_mapping(
                verifier_payload.get("checkpoint_inferred_model_support_config") or {}
            ),
        },
        "tournament_loader_report_summary": (
            _loader_report_summary(loader_payload) if loader_payload is not None else None
        ),
        "non_claims": {
            "base_export_stock_eval_tournament_loadable": False,
            "base_export_stock_model_contract_verified": False,
            "stock_resume_claim": False,
            "coach_training_speed_claim": False,
            "touches_live_runs": False,
            "calls_train_muzero": False,
        },
    }
    validate_compact_stock_export_evidence_bundle_v1(bundle)
    return bundle


def save_compact_stock_export_evidence_bundle_v1(
    checkpoint_path: str | Path,
    *,
    verification_report_path: str | Path,
    tournament_loader_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
    require_tournament_loader_report: bool = True,
) -> dict[str, Any]:
    """Write a sibling evidence bundle for a verified compact stock export."""

    bundle = build_compact_stock_export_evidence_bundle_v1(
        checkpoint_path,
        verification_report_path=verification_report_path,
        tournament_loader_report_path=tournament_loader_report_path,
        require_tournament_loader_report=require_tournament_loader_report,
    )
    path = (
        compact_stock_export_evidence_bundle_path(checkpoint_path)
        if output_path is None
        else Path(output_path)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain_mapping(bundle), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"bundle": bundle, "path": path}


def validate_compact_stock_export_evidence_bundle_v1(bundle: Mapping[str, Any]) -> None:
    """Validate a compact stock export evidence bundle and its file hashes."""

    if bundle.get("schema_id") != COMPACT_STOCK_EXPORT_EVIDENCE_BUNDLE_SCHEMA_ID:
        raise ValueError("compact stock export evidence bundle schema mismatch")
    if bundle.get("compact_stock_export_evidence_bundle") is not True:
        raise ValueError("compact stock export evidence bundle marker missing")
    if bundle.get("ok") is not True:
        raise ValueError("compact stock export evidence bundle must be ok=true")
    if bundle.get("base_export_claims_mutated") is not False:
        raise ValueError("compact stock export evidence bundle mutated base export claims")
    claims = bundle.get("attached_claims")
    if not isinstance(claims, Mapping):
        raise ValueError("compact stock export evidence bundle missing attached_claims")
    for key in (
        "promotion_claim",
        "training_speed_claim",
        "optimizer_resume_claim",
        "base_export_claims_mutated",
    ):
        if claims.get(key) is not False:
            raise ValueError(f"compact stock export evidence bundle {key} must be false")
    files = bundle.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("compact stock export evidence bundle missing files")
    for name in (
        "checkpoint",
        "policy_metadata_sidecar",
        "verification_report",
        "tournament_loader_report",
    ):
        entry = files.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"compact stock export evidence bundle missing {name}")
        if entry.get("required") is False:
            continue
        _validate_file_record(name, entry)


def verify_compact_stock_export_model_contract_v1(
    path: str | Path,
    *,
    seed: int = 0,
    num_simulations: int = 8,
    batch_size: int = 16,
    use_cuda: bool = False,
    state_key: str | None = COMPACT_STOCK_EXPORT_MODEL_STATE_KEY,
    require_sidecar: bool = True,
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """Run the existing stock LightZero strict-load path against an export."""

    path_obj = Path(path)
    sidecar_path = checkpoint_policy_metadata_sidecar_path(path_obj)
    report: dict[str, Any] = {
        "schema_id": COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID,
        "compact_stock_model_contract_verification": True,
        "checkpoint_path": str(path_obj),
        "sidecar_path": str(sidecar_path),
        "sidecar_present": sidecar_path.is_file(),
        "sidecar_required": bool(require_sidecar),
        "state_key_requested": state_key,
        "seed": int(seed),
        "num_simulations": int(num_simulations),
        "batch_size": int(batch_size),
        "use_cuda": bool(use_cuda),
        "ok": False,
        "strict_load": True,
        "stock_model_contract_verified": False,
        "stock_eval_tournament_loadable": False,
        "strict_stock_model_load_verified": False,
    }
    if not path_obj.is_file():
        return _verification_failure(
            report,
            reason="checkpoint_missing",
            raise_on_failure=raise_on_failure,
        )
    if require_sidecar and not sidecar_path.is_file():
        return _verification_failure(
            report,
            reason="policy_metadata_sidecar_missing",
            raise_on_failure=raise_on_failure,
        )

    try:
        payload = _torch_load_export_payload(path_obj)
        if not isinstance(payload, Mapping):
            raise ValueError("compact stock export payload must be a mapping")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("compact stock export payload missing metadata mapping")
        if metadata.get("compact_stock_checkpoint_export") is not True:
            raise ValueError("payload is not a compact stock checkpoint export")
        report.update(_source_export_report_metadata(metadata))
        policy_metadata = require_checkpoint_policy_observation_metadata(
            checkpoint_path=path_obj,
            checkpoint_payload=payload,
        )
        found_key, state_dict = provider._state_dict_from_payload(
            payload,
            state_key=state_key,
        )
        report.update(
            {
                "policy_observation_metadata": _plain_mapping(policy_metadata),
                "state_key": found_key,
                "tensor_count": sum(
                    1 for value in state_dict.values() if _is_tensor_like(value)
                ),
                "checkpoint_inferred_model_support_config": _plain_mapping(
                    provider._infer_model_support_config_from_state_dict(state_dict)
                ),
            }
        )
        _policy, _device, load_summary = (
            provider.load_lightzero_curvytron_visual_survival_policy(
                checkpoint_path=path_obj,
                seed=seed,
                num_simulations=num_simulations,
                batch_size=batch_size,
                use_cuda=use_cuda,
                state_key=found_key,
            )
        )
        del _policy, _device
        load_summary_plain = _plain_mapping(load_summary)
        ok = bool(load_summary_plain.get("ok"))
        report.update(
            {
                "ok": ok,
                "load_summary": load_summary_plain,
                "strict_load": bool(load_summary_plain.get("strict", True)),
                "stock_model_contract_verified": ok,
                "strict_stock_model_load_verified": ok,
                "stock_eval_tournament_loadable": False,
                "stock_eval_tournament_load_status": (
                    "strict_stock_model_load_verified_gameplay_not_run"
                    if ok
                    else "strict_load_summary_not_ok"
                ),
                "failure_reason": None if ok else "strict_load_summary_not_ok",
            }
        )
        if not ok:
            return _verification_failure(
                report,
                reason="strict_load_summary_not_ok",
                raise_on_failure=raise_on_failure,
            )
        return report
    except Exception as exc:
        report.update(
            {
                "ok": False,
                "stock_model_contract_verified": False,
                "stock_eval_tournament_loadable": False,
                "strict_stock_model_load_verified": False,
                "stock_eval_tournament_load_status": (
                    "strict_stock_model_load_failed"
                ),
                "failure_reason": "strict_stock_model_load_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        if raise_on_failure:
            raise RuntimeError(
                f"compact stock export strict stock model load failed for {path_obj}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return report


def validate_policy_observation_sidecar_v1(sidecar: Mapping[str, Any]) -> None:
    missing = [
        field
        for field in REQUIRED_POLICY_OBSERVATION_METADATA_FIELDS
        if not sidecar.get(field)
    ]
    if missing:
        raise ValueError(
            "compact stock export sidecar missing required policy observation "
            f"metadata: {', '.join(missing)}"
        )
    if sidecar.get("schema_id") != CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID:
        raise ValueError("compact stock export sidecar schema mismatch")
    observation_contract = sidecar.get("observation_contract")
    if not isinstance(observation_contract, Mapping):
        raise ValueError("compact stock export sidecar missing observation_contract")
    missing_contract = [
        field
        for field in REQUIRED_POLICY_OBSERVATION_CONTRACT_FIELDS
        if observation_contract.get(field) is None
    ]
    if missing_contract:
        raise ValueError(
            "compact stock export sidecar observation_contract missing fields: "
            f"{', '.join(missing_contract)}"
        )
    missing_runtime = [
        field
        for field in REQUIRED_STOCK_EXPORT_RUNTIME_METADATA_FIELDS
        if sidecar.get(field) is None
    ]
    if missing_runtime:
        raise ValueError(
            "compact stock export sidecar missing stock runtime/model metadata: "
            f"{', '.join(missing_runtime)}"
        )
    expected = {
        "policy_trail_render_mode": POLICY_TRAIL_RENDER_MODE,
        "policy_bonus_render_mode": POLICY_BONUS_RENDER_MODE,
        "policy_observation_backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "policy_observation_contract_id": POLICY_OBSERVATION_CONTRACT_ID,
        "policy_observation_perspective_schema_id": (
            POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID
        ),
    }
    for key, value in expected.items():
        if str(sidecar.get(key)) != str(value):
            raise ValueError(f"compact stock export sidecar {key} mismatch")
    contract_expected = {
        "contract_id": POLICY_OBSERVATION_CONTRACT_ID,
        "perspective_schema_id": POLICY_OBSERVATION_PERSPECTIVE_SCHEMA_ID,
        "trail_render_mode": POLICY_TRAIL_RENDER_MODE,
        "bonus_render_mode": POLICY_BONUS_RENDER_MODE,
        "backend": DEFAULT_POLICY_OBSERVATION_BACKEND,
        "perspective": POLICY_OBSERVATION_PERSPECTIVE,
        "seat_mapping": POLICY_OBSERVATION_SEAT_MAPPING,
        "stack_shape": list(POLICY_STACK_SHAPE),
        "single_frame_shape": list(POLICY_SINGLE_FRAME_SHAPE),
    }
    for key, value in contract_expected.items():
        actual = observation_contract.get(key)
        if isinstance(value, list):
            try:
                matches = list(actual) == value
            except TypeError:
                matches = False
            if not matches:
                raise ValueError(
                    "compact stock export sidecar observation_contract."
                    f"{key} mismatch"
                )
            continue
        if str(actual) != str(value):
            raise ValueError(
                f"compact stock export sidecar observation_contract.{key} mismatch"
            )
    _validate_positive_int(sidecar, "decision_source_frames")
    _validate_positive_int(sidecar, "source_max_steps")
    _validate_positive_float(sidecar, "source_physics_step_ms")
    _validate_positive_float(sidecar, "decision_ms")


def _torch_load_export_payload(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _verification_failure(
    report: dict[str, Any],
    *,
    reason: str,
    raise_on_failure: bool,
) -> dict[str, Any]:
    report.update(
        {
            "ok": False,
            "failure_reason": reason,
            "stock_model_contract_verified": False,
            "stock_eval_tournament_loadable": False,
            "strict_stock_model_load_verified": False,
        }
    )
    if raise_on_failure:
        raise RuntimeError(
            f"compact stock export model contract verification failed: {reason}"
        )
    return report


def _source_export_report_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "source_compact_checkpoint_id",
        "source_trainer_id",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "stock_model_state_key",
        "model_env_variant",
        "model_reward_variant",
        "env_variant",
        "reward_variant",
        "decision_source_frames",
        "source_physics_step_ms",
        "decision_ms",
        "source_max_steps",
        "source_max_steps_semantics",
        "learner_seat_mode",
    )
    report = {key: _plain_value(metadata[key]) for key in keys if key in metadata}
    if "schema_id" in metadata:
        report["source_export_schema_id"] = _plain_value(metadata["schema_id"])
    return report


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"compact stock export evidence {label} missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ValueError(
            f"compact stock export evidence {label} is not readable JSON: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"compact stock export evidence {label} must be a mapping")
    return dict(payload)


def _file_record(path: Path, *, required: bool) -> dict[str, Any]:
    if required and not path.is_file():
        raise ValueError(f"compact stock export evidence file missing: {path}")
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "required": bool(required),
    }


def _validate_file_record(name: str, entry: Mapping[str, Any]) -> None:
    path_text = entry.get("path")
    if not path_text:
        raise ValueError(f"compact stock export evidence {name} path missing")
    path = Path(str(path_text))
    if not path.is_file():
        raise ValueError(f"compact stock export evidence {name} file missing")
    expected_sha = str(entry.get("sha256") or "")
    if not expected_sha:
        raise ValueError(f"compact stock export evidence {name} sha256 missing")
    actual_sha = _file_sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(f"compact stock export evidence {name} sha256 mismatch")
    expected_size = int(entry.get("size_bytes"))
    if int(path.stat().st_size) != expected_size:
        raise ValueError(f"compact stock export evidence {name} size mismatch")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_export_payload_for_evidence(
    *,
    checkpoint_path: Path,
    payload: Any,
) -> Mapping[str, Any]:
    if not checkpoint_path.is_file():
        raise ValueError("compact stock export evidence checkpoint missing")
    if not isinstance(payload, Mapping):
        raise ValueError("compact stock export evidence checkpoint must be a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("compact stock export evidence checkpoint missing metadata")
    if payload.get(COMPACT_STOCK_EXPORT_MODEL_STATE_KEY) is None:
        raise ValueError("compact stock export evidence checkpoint missing model")
    if payload.get("compact_trainer_checkpoint_metadata") is None:
        raise ValueError(
            "compact stock export evidence checkpoint missing compact trainer metadata"
        )
    if metadata.get("compact_stock_checkpoint_export") is not True:
        raise ValueError("compact stock export evidence checkpoint is not stock export")
    _require_base_export_non_claims(metadata)
    return metadata


def _require_base_export_non_claims(metadata: Mapping[str, Any]) -> None:
    expected = {
        "stock_payload_mapping": True,
        "stock_state_dict_discovery_verified": True,
        "stock_model_contract_verified": False,
        "stock_model_contract_verification_required": True,
        "stock_eval_tournament_loadable": False,
        "stock_resume_claim": False,
        "optimizer_resume_supported": False,
        "promotion_claim": False,
        "training_speed_claim": False,
        "calls_train_muzero": False,
        "touches_live_runs": False,
    }
    for key, value in expected.items():
        if metadata.get(key) is not value:
            raise ValueError(
                f"compact stock export evidence base export {key} must be {value!r}"
            )
    if metadata.get("stock_state_dict_discovery_key") != COMPACT_STOCK_EXPORT_MODEL_STATE_KEY:
        raise ValueError("compact stock export evidence base export state key mismatch")
    if metadata.get("stock_model_state_key") != COMPACT_STOCK_EXPORT_MODEL_STATE_KEY:
        raise ValueError("compact stock export evidence base export model key mismatch")


def _require_export_sidecar_consistent(
    *,
    metadata: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    for key in (
        *REQUIRED_POLICY_OBSERVATION_METADATA_FIELDS,
        "observation_contract",
        *REQUIRED_STOCK_EXPORT_RUNTIME_METADATA_FIELDS,
    ):
        if _plain_value(metadata.get(key)) != _plain_value(sidecar.get(key)):
            raise ValueError(
                f"compact stock export evidence sidecar metadata drift: {key}"
            )


def _validate_verification_report_for_evidence(
    *,
    report: Mapping[str, Any],
    checkpoint_path: Path,
    sidecar_path: Path,
    export_metadata: Mapping[str, Any],
) -> None:
    if report.get("schema_id") != COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID:
        raise ValueError("compact stock export evidence verifier schema mismatch")
    required_true = (
        "ok",
        "sidecar_required",
        "sidecar_present",
        "strict_load",
        "stock_model_contract_verified",
        "strict_stock_model_load_verified",
    )
    for key in required_true:
        if report.get(key) is not True:
            raise ValueError(f"compact stock export evidence verifier {key} not true")
    if report.get("failure_reason") is not None:
        raise ValueError("compact stock export evidence verifier has failure_reason")
    _require_same_path(
        report.get("checkpoint_path"),
        checkpoint_path,
        context="verifier checkpoint_path",
    )
    _require_same_path(
        report.get("sidecar_path"),
        sidecar_path,
        context="verifier sidecar_path",
    )
    _require_report_identity(report=report, export_metadata=export_metadata)
    if report.get("state_key") != export_metadata.get("stock_model_state_key"):
        raise ValueError("compact stock export evidence verifier state key mismatch")
    if int(report.get("tensor_count") or 0) <= 0:
        raise ValueError("compact stock export evidence verifier tensor_count missing")
    load = report.get("load_summary")
    if not isinstance(load, Mapping):
        raise ValueError("compact stock export evidence verifier missing load_summary")
    _validate_strict_load_summary(load, context="verifier load_summary")


def _validate_export_extra_metadata(metadata: Mapping[str, Any]) -> None:
    bad_keys = [
        key
        for key in metadata
        if key in _PROTECTED_EXPORT_EXTRA_METADATA_KEYS
        or any(
            key.startswith(prefix)
            for prefix in _PROTECTED_EXPORT_EXTRA_METADATA_PREFIXES
        )
    ]
    if bad_keys:
        raise ValueError(
            "compact stock export extra metadata cannot override protected "
            f"metadata keys: {', '.join(sorted(bad_keys))}"
        )


def _validate_loader_report_for_evidence(
    *,
    report: Mapping[str, Any],
    checkpoint_path: Path,
    export_metadata: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    if report.get("schema_id") != COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID:
        raise ValueError("compact stock export evidence loader schema mismatch")
    if report.get("ok") is not True:
        raise ValueError("compact stock export evidence loader ok not true")
    _require_same_path(
        report.get("checkpoint_path"),
        checkpoint_path,
        context="loader checkpoint_path",
    )
    if report.get("checkpoint_state_key") != export_metadata.get("stock_model_state_key"):
        raise ValueError("compact stock export evidence loader state key mismatch")
    for key in (
        "policy_observation_backend",
        "policy_observation_contract_id",
        "policy_observation_perspective_schema_id",
        "policy_trail_render_mode",
        "policy_bonus_render_mode",
        "model_env_variant",
        "model_reward_variant",
    ):
        if _plain_value(report.get(key)) != _plain_value(sidecar.get(key)):
            raise ValueError(f"compact stock export evidence loader {key} mismatch")
    runtime = report.get("runtime_settings")
    if not isinstance(runtime, Mapping):
        raise ValueError("compact stock export evidence loader missing runtime_settings")
    for key in (
        "decision_source_frames",
        "source_physics_step_ms",
        "decision_ms",
        "source_max_steps",
        "source_max_steps_semantics",
    ):
        if _plain_value(runtime.get(key)) != _plain_value(sidecar.get(key)):
            raise ValueError(f"compact stock export evidence loader runtime {key} mismatch")
    _validate_strict_load_summary(
        _loader_load_summary(report),
        context="loader load_state_dict",
    )


def _require_report_identity(
    *,
    report: Mapping[str, Any],
    export_metadata: Mapping[str, Any],
) -> None:
    for key in (
        "source_compact_checkpoint_id",
        "source_trainer_id",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "stock_model_state_key",
        "model_env_variant",
        "model_reward_variant",
        "env_variant",
        "reward_variant",
        "decision_source_frames",
        "source_physics_step_ms",
        "decision_ms",
        "source_max_steps",
        "source_max_steps_semantics",
        "learner_seat_mode",
    ):
        if _plain_value(report.get(key)) != _plain_value(export_metadata.get(key)):
            raise ValueError(f"compact stock export evidence report {key} mismatch")


def _validate_strict_load_summary(summary: Mapping[str, Any], *, context: str) -> None:
    if summary.get("ok") is not True:
        raise ValueError(f"compact stock export evidence {context} ok not true")
    if summary.get("strict") is not True:
        raise ValueError(f"compact stock export evidence {context} strict not true")
    if list(summary.get("missing_keys") or []) != []:
        raise ValueError(f"compact stock export evidence {context} missing_keys not empty")
    if list(summary.get("unexpected_keys") or []) != []:
        raise ValueError(
            f"compact stock export evidence {context} unexpected_keys not empty"
        )


def _loader_load_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    surface = report.get("surface")
    if not isinstance(surface, Mapping):
        raise ValueError("compact stock export evidence loader missing surface")
    load = surface.get("load_state_dict")
    if not isinstance(load, Mapping):
        raise ValueError("compact stock export evidence loader missing load_state_dict")
    return load


def _loader_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    load = _loader_load_summary(report)
    return {
        "checkpoint_state_key": _plain_value(report.get("checkpoint_state_key")),
        "strict_load": True,
        "candidate": _plain_value(load.get("candidate")),
        "smoke_scope": _plain_value(report.get("smoke_scope")),
    }


def _export_identity_for_evidence(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_export_schema_id": COMPACT_STOCK_CHECKPOINT_EXPORT_SCHEMA_ID,
        "source_compact_checkpoint_id": _plain_value(
            metadata.get("source_compact_checkpoint_id")
        ),
        "source_trainer_id": _plain_value(metadata.get("source_trainer_id")),
        "policy_version_ref": _plain_value(metadata.get("policy_version_ref")),
        "model_version_ref": _plain_value(metadata.get("model_version_ref")),
        "policy_source": _plain_value(metadata.get("policy_source")),
        "stock_model_state_key": _plain_value(metadata.get("stock_model_state_key")),
    }


def _require_same_path(value: Any, expected: Path, *, context: str) -> None:
    if value is None:
        raise ValueError(f"compact stock export evidence {context} missing")
    try:
        matches = Path(str(value)).resolve() == expected.resolve()
    except Exception:
        matches = False
    if not matches:
        raise ValueError(f"compact stock export evidence {context} mismatch")


def _validate_tensor_state_dict(state_dict: Mapping[str, Any]) -> None:
    if not isinstance(state_dict, Mapping):
        raise ValueError("compact stock export requires model_state_dict mapping")
    tensor_count = sum(1 for value in state_dict.values() if _is_tensor_like(value))
    if tensor_count <= 0:
        raise ValueError("compact stock export model_state_dict has no tensor values")


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _validate_positive_int(sidecar: Mapping[str, Any], key: str) -> None:
    try:
        value = int(sidecar[key])
    except Exception as exc:
        raise ValueError(f"compact stock export sidecar {key} must be an int") from exc
    if value <= 0:
        raise ValueError(f"compact stock export sidecar {key} must be positive")


def _validate_positive_float(sidecar: Mapping[str, Any], key: str) -> None:
    try:
        value = float(sidecar[key])
    except Exception as exc:
        raise ValueError(f"compact stock export sidecar {key} must be a float") from exc
    if value <= 0.0:
        raise ValueError(f"compact stock export sidecar {key} must be positive")


def _plain_mapping(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(value) for key, value in dict(metadata).items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "CHECKPOINT_POLICY_METADATA_SIDECAR_SCHEMA_ID",
    "COMPACT_STOCK_EXPORT_EVIDENCE_BUNDLE_SCHEMA_ID",
    "COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID",
    "COMPACT_STOCK_CHECKPOINT_EXPORT_SCHEMA_ID",
    "COMPACT_STOCK_EXPORT_MODEL_STATE_KEY",
    "COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID",
    "DEFAULT_COMPACT_STOCK_EXPORT_ENV_VARIANT",
    "DEFAULT_COMPACT_STOCK_EXPORT_REWARD_VARIANT",
    "REQUIRED_POLICY_OBSERVATION_METADATA_FIELDS",
    "REQUIRED_POLICY_OBSERVATION_CONTRACT_FIELDS",
    "REQUIRED_STOCK_EXPORT_RUNTIME_METADATA_FIELDS",
    "build_compact_stock_export_payload_v1",
    "build_compact_stock_export_evidence_bundle_v1",
    "build_current_policy_observation_sidecar_v1",
    "compact_stock_export_evidence_bundle_path",
    "save_compact_stock_export_v1",
    "save_compact_stock_export_evidence_bundle_v1",
    "validate_compact_stock_export_evidence_bundle_v1",
    "validate_policy_observation_sidecar_v1",
    "verify_compact_stock_export_model_contract_v1",
]
