"""Current-chain eval/GIF/tournament-load evidence for compact checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_SCHEMA_ID = (
    "curvyzero_compact_current_chain_eval_gif_tournament_load_evidence/v1"
)
COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_STATUS_VERIFIED = (
    "current_chain_eval_gif_tournament_load_verified"
)
COMPACT_STOCK_CHECKPOINT_EXPORT_SCHEMA_ID = (
    "curvyzero_compact_stock_checkpoint_export/v1"
)
COMPACT_STOCK_EXPORT_EVIDENCE_BUNDLE_SCHEMA_ID = (
    "curvyzero_compact_stock_export_evidence_bundle/v1"
)
COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID = (
    "curvyzero_compact_stock_model_contract_verification/v1"
)
COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID = (
    "curvyzero_compact_stock_export_tournament_loader_smoke/v1"
)
COMPACT_STOCK_ONE_GAME_GIF_SMOKE_SCHEMA_ID = (
    "curvyzero_compact_stock_export_one_game_gif_smoke/v1"
)
COMPACT_TRAINER_CHECKPOINT_SCHEMA_ID = "curvyzero_compact_trainer_checkpoint/v1"
COMPACT_STOCK_EXPORT_MODEL_STATE_KEY = "model"

_IDENTITY_KEYS = (
    "checkpoint_id",
    "trainer_id",
    "policy_version_ref",
    "model_version_ref",
    "policy_source",
)
_EXPORT_IDENTITY_KEYS = {
    "checkpoint_id": "source_compact_checkpoint_id",
    "trainer_id": "source_trainer_id",
    "policy_version_ref": "policy_version_ref",
    "model_version_ref": "model_version_ref",
    "policy_source": "policy_source",
}
_RUNTIME_KEYS = (
    "decision_source_frames",
    "source_physics_step_ms",
    "decision_ms",
    "source_max_steps",
    "source_max_steps_semantics",
)
_POLICY_OBSERVATION_KEYS = (
    "policy_observation_backend",
    "policy_observation_contract_id",
    "policy_observation_perspective_schema_id",
    "policy_trail_render_mode",
    "policy_bonus_render_mode",
)


def build_compact_current_chain_eval_gif_tournament_load_evidence_v1(
    *,
    compact_checkpoint_path: str | Path,
    stock_export_path: str | Path,
    stock_export_evidence_bundle_path: str | Path,
    gameplay_smoke_report_path: str | Path,
    game_summary_path: str | Path | None = None,
    gif_path: str | Path | None = None,
    frames_path: str | Path | None = None,
    standalone_eval_summary_path: str | Path | None = None,
    compact_checkpoint_smoke_report_path: str | Path | None = None,
    profile_result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a hash-bound evidence object for the compact Coach eval/GIF gate."""

    paths = _resolve_current_chain_paths(
        compact_checkpoint_path=compact_checkpoint_path,
        stock_export_path=stock_export_path,
        stock_export_evidence_bundle_path=stock_export_evidence_bundle_path,
        gameplay_smoke_report_path=gameplay_smoke_report_path,
        game_summary_path=game_summary_path,
        gif_path=gif_path,
        frames_path=frames_path,
        standalone_eval_summary_path=standalone_eval_summary_path,
        compact_checkpoint_smoke_report_path=compact_checkpoint_smoke_report_path,
        profile_result_path=profile_result_path,
    )
    compact_metadata = _load_compact_checkpoint_metadata(paths["compact_checkpoint"])
    stock_payload = _torch_load_mapping(paths["stock_export"], label="stock export")
    stock_metadata = _stock_export_metadata(stock_payload)
    _require_stock_export_matches_compact(
        compact_metadata=compact_metadata,
        stock_metadata=stock_metadata,
    )
    stock_bundle = _read_json_mapping(
        paths["stock_export_evidence_bundle"],
        label="stock export evidence bundle",
    )
    bundle_paths = _validate_stock_export_bundle_for_current_chain(
        bundle=stock_bundle,
        stock_export_path=paths["stock_export"],
        stock_metadata=stock_metadata,
    )
    gameplay_report = _read_json_mapping(
        paths["gameplay_smoke_report"],
        label="gameplay smoke report",
    )
    report_paths = _validate_gameplay_smoke_report_for_current_chain(
        report=gameplay_report,
        report_path=paths["gameplay_smoke_report"],
        stock_export_path=paths["stock_export"],
        stock_metadata=stock_metadata,
        stock_bundle_path=paths["stock_export_evidence_bundle"],
        bundle_paths=bundle_paths,
        game_summary_path=paths.get("game_summary"),
        gif_path=paths.get("game_gif"),
        frames_path=paths.get("frames_npz"),
        standalone_eval_summary_path=paths.get("standalone_eval_summary"),
    )
    paths.update(report_paths)

    files = {
        "compact_checkpoint": _file_record(paths["compact_checkpoint"], required=True),
        "stock_export": _file_record(paths["stock_export"], required=True),
        "stock_export_evidence_bundle": _file_record(
            paths["stock_export_evidence_bundle"],
            required=True,
        ),
        "policy_metadata_sidecar": _file_record(
            bundle_paths["policy_metadata_sidecar"],
            required=True,
        ),
        "verification_report": _file_record(
            bundle_paths["verification_report"],
            required=True,
        ),
        "tournament_loader_report": _file_record(
            bundle_paths["tournament_loader_report"],
            required=True,
        ),
        "gameplay_smoke_report": _file_record(
            paths["gameplay_smoke_report"],
            required=True,
        ),
        "game_summary": _file_record(paths["game_summary"], required=True),
        "game_gif": _file_record(paths["game_gif"], required=True),
        "frames_npz": _file_record(paths["frames_npz"], required=True),
        "standalone_eval_summary": _file_record(
            paths["standalone_eval_summary"],
            required=True,
        ),
        "compact_checkpoint_smoke_report": (
            _file_record(paths["compact_checkpoint_smoke_report"], required=True)
            if paths.get("compact_checkpoint_smoke_report") is not None
            else {"path": None, "required": False}
        ),
        "profile_result": (
            _file_record(paths["profile_result"], required=True)
            if paths.get("profile_result") is not None
            else {"path": None, "required": False}
        ),
    }
    first_joint_action = _first_joint_action(gameplay_report)
    standalone_eval = _standalone_eval_status(gameplay_report)
    identity = _current_chain_identity(compact_metadata)
    evidence_id = (
        f"current-chain-eval-gif-tournament:"
        f"{identity['checkpoint_id']}:"
        f"{files['gameplay_smoke_report']['sha256'][:12]}"
    )
    evidence = {
        "schema_id": (
            COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_SCHEMA_ID
        ),
        "compact_current_chain_eval_gif_tournament_load_evidence": True,
        "ok": True,
        "evidence_id": evidence_id,
        "evidence_scope": (
            "current_compact_checkpoint_to_stock_export_to_game_gif_eval_chain"
        ),
        "status": COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_STATUS_VERIFIED,
        "current_chain_identity": identity,
        "stock_export_identity": _stock_export_identity(stock_metadata),
        "files": files,
        "attached_claims": {
            "eval_gif_tournament_load": True,
            "strict_stock_model_load_verified": True,
            "tournament_loader_constructed": True,
            "tournament_gameplay_verified": True,
            "gif_artifact_verified": True,
            "frames_npz_verified": True,
            "standalone_eval_verified": True,
            "promotion_claim": False,
            "training_speed_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "base_export_claims_mutated": False,
        },
        "gameplay_summary": {
            "schema_id": COMPACT_STOCK_ONE_GAME_GIF_SMOKE_SCHEMA_ID,
            "physical_steps": int(gameplay_report["physical_steps"]),
            "first_joint_action": first_joint_action,
            "gif_ref": str(gameplay_report["gif_ref"]),
            "frames_ref": str(gameplay_report["frames_ref"]),
            "game_summary_ref": str(gameplay_report["game_summary_ref"]),
        },
        "standalone_eval_summary": {
            "steps_survived": int(standalone_eval["steps_survived"]),
            "cap": int(standalone_eval.get("cap") or 0),
            "strict_policy_model_load_ok": True,
            "env_reset_ok": True,
            "policy_could_act_in_real_env": True,
        },
        "non_claims": {
            "promotion_claim": False,
            "training_speed_claim": False,
            "calls_train_muzero": False,
            "touches_live_runs": False,
            "stock_resume_claim": False,
            "optimizer_resume_claim": False,
        },
    }
    validate_compact_current_chain_eval_gif_tournament_load_evidence_v1(evidence)
    return evidence


def compact_current_chain_eval_gif_tournament_load_evidence_path(
    path: str | Path,
) -> Path:
    """Return the sibling current-chain evidence path for a stock export."""

    return Path(f"{Path(path)}.current_chain_eval_gif_tournament_load.evidence.json")


def save_compact_current_chain_eval_gif_tournament_load_evidence_v1(
    *,
    compact_checkpoint_path: str | Path,
    stock_export_path: str | Path,
    stock_export_evidence_bundle_path: str | Path,
    gameplay_smoke_report_path: str | Path,
    game_summary_path: str | Path | None = None,
    gif_path: str | Path | None = None,
    frames_path: str | Path | None = None,
    standalone_eval_summary_path: str | Path | None = None,
    compact_checkpoint_smoke_report_path: str | Path | None = None,
    profile_result_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write the hash-bound current-chain eval/GIF/tournament evidence."""

    evidence = build_compact_current_chain_eval_gif_tournament_load_evidence_v1(
        compact_checkpoint_path=compact_checkpoint_path,
        stock_export_path=stock_export_path,
        stock_export_evidence_bundle_path=stock_export_evidence_bundle_path,
        gameplay_smoke_report_path=gameplay_smoke_report_path,
        game_summary_path=game_summary_path,
        gif_path=gif_path,
        frames_path=frames_path,
        standalone_eval_summary_path=standalone_eval_summary_path,
        compact_checkpoint_smoke_report_path=compact_checkpoint_smoke_report_path,
        profile_result_path=profile_result_path,
    )
    path = (
        compact_current_chain_eval_gif_tournament_load_evidence_path(
            stock_export_path,
        )
        if output_path is None
        else Path(output_path)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_plain_value(evidence), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"evidence": evidence, "path": path}


def validate_compact_current_chain_eval_gif_tournament_load_evidence_v1(
    evidence: Mapping[str, Any],
) -> None:
    """Validate current-chain eval/GIF evidence and the referenced file hashes."""

    if (
        evidence.get("schema_id")
        != COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_SCHEMA_ID
    ):
        raise ValueError("current-chain eval/GIF/tournament evidence schema mismatch")
    if evidence.get("compact_current_chain_eval_gif_tournament_load_evidence") is not True:
        raise ValueError("current-chain eval/GIF/tournament evidence marker missing")
    if evidence.get("ok") is not True:
        raise ValueError("current-chain eval/GIF/tournament evidence must be ok=true")
    if evidence.get("status") != (
        COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_STATUS_VERIFIED
    ):
        raise ValueError("current-chain eval/GIF/tournament evidence status mismatch")
    identity = evidence.get("current_chain_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("current-chain eval/GIF/tournament evidence missing identity")
    for key in _IDENTITY_KEYS:
        if not str(identity.get(key, "")).strip():
            raise ValueError(
                f"current-chain eval/GIF/tournament evidence missing {key}"
            )
    claims = evidence.get("attached_claims")
    if not isinstance(claims, Mapping):
        raise ValueError("current-chain eval/GIF/tournament evidence missing claims")
    for key in (
        "eval_gif_tournament_load",
        "strict_stock_model_load_verified",
        "tournament_loader_constructed",
        "tournament_gameplay_verified",
        "gif_artifact_verified",
        "frames_npz_verified",
        "standalone_eval_verified",
    ):
        if claims.get(key) is not True:
            raise ValueError(
                f"current-chain eval/GIF/tournament evidence {key} not true"
            )
    for key in (
        "promotion_claim",
        "training_speed_claim",
        "calls_train_muzero",
        "touches_live_runs",
        "base_export_claims_mutated",
    ):
        if claims.get(key) is not False:
            raise ValueError(
                f"current-chain eval/GIF/tournament evidence {key} must be false"
            )
    non_claims = evidence.get("non_claims")
    if not isinstance(non_claims, Mapping):
        raise ValueError("current-chain eval/GIF/tournament evidence missing non_claims")
    for key in (
        "promotion_claim",
        "training_speed_claim",
        "calls_train_muzero",
        "touches_live_runs",
        "stock_resume_claim",
        "optimizer_resume_claim",
    ):
        if non_claims.get(key) is not False:
            raise ValueError(
                f"current-chain eval/GIF/tournament evidence non-claim {key} "
                "must be false"
            )
    files = evidence.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("current-chain eval/GIF/tournament evidence missing files")
    for name in (
        "compact_checkpoint",
        "stock_export",
        "stock_export_evidence_bundle",
        "policy_metadata_sidecar",
        "verification_report",
        "tournament_loader_report",
        "gameplay_smoke_report",
        "game_summary",
        "game_gif",
        "frames_npz",
        "standalone_eval_summary",
        "compact_checkpoint_smoke_report",
        "profile_result",
    ):
        entry = files.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"current-chain eval/GIF/tournament evidence missing file {name}"
            )
        if entry.get("required") is False:
            continue
        _validate_file_record(name, entry)

    compact_checkpoint = Path(str(files["compact_checkpoint"]["path"]))
    stock_export = Path(str(files["stock_export"]["path"]))
    stock_bundle_path = Path(str(files["stock_export_evidence_bundle"]["path"]))
    gameplay_report_path = Path(str(files["gameplay_smoke_report"]["path"]))
    compact_metadata = _load_compact_checkpoint_metadata(compact_checkpoint)
    stock_payload = _torch_load_mapping(stock_export, label="stock export")
    stock_metadata = _stock_export_metadata(stock_payload)
    _require_stock_export_matches_compact(
        compact_metadata=compact_metadata,
        stock_metadata=stock_metadata,
    )
    if _current_chain_identity(compact_metadata) != {
        key: _plain_value(identity[key]) for key in _IDENTITY_KEYS
    }:
        raise ValueError("current-chain eval/GIF/tournament identity mismatch")
    stock_bundle = _read_json_mapping(
        stock_bundle_path,
        label="stock export evidence bundle",
    )
    bundle_paths = _validate_stock_export_bundle_for_current_chain(
        bundle=stock_bundle,
        stock_export_path=stock_export,
        stock_metadata=stock_metadata,
    )
    gameplay_report = _read_json_mapping(
        gameplay_report_path,
        label="gameplay smoke report",
    )
    _validate_gameplay_smoke_report_for_current_chain(
        report=gameplay_report,
        report_path=gameplay_report_path,
        stock_export_path=stock_export,
        stock_metadata=stock_metadata,
        stock_bundle_path=stock_bundle_path,
        bundle_paths=bundle_paths,
        game_summary_path=Path(str(files["game_summary"]["path"])),
        gif_path=Path(str(files["game_gif"]["path"])),
        frames_path=Path(str(files["frames_npz"]["path"])),
        standalone_eval_summary_path=Path(
            str(files["standalone_eval_summary"]["path"])
        ),
    )


def validate_compact_current_chain_eval_gif_tournament_load_matches_checkpoint_v1(
    evidence: Mapping[str, Any],
    *,
    checkpoint_id: str,
    trainer_id: str,
    policy_version_ref: str,
    model_version_ref: str,
    policy_source: str,
) -> None:
    """Validate evidence and require identity to match a compact checkpoint."""

    validate_compact_current_chain_eval_gif_tournament_load_evidence_v1(evidence)
    identity = evidence["current_chain_identity"]
    expected = {
        "checkpoint_id": checkpoint_id,
        "trainer_id": trainer_id,
        "policy_version_ref": policy_version_ref,
        "model_version_ref": model_version_ref,
        "policy_source": policy_source,
    }
    for key, value in expected.items():
        if str(identity.get(key, "")) != str(value):
            raise ValueError(
                f"current-chain eval/GIF/tournament evidence {key} mismatch"
            )


def compact_current_chain_eval_gif_tournament_load_evidence_ref(
    evidence: Mapping[str, Any],
) -> str:
    """Return a stable human-readable evidence ref for Coach metadata."""

    validate_compact_current_chain_eval_gif_tournament_load_evidence_v1(evidence)
    identity = evidence["current_chain_identity"]
    return (
        "compact_current_chain_eval_gif_tournament_load:"
        f"{identity['checkpoint_id']}:"
        f"{evidence['evidence_id']}"
    )


def _resolve_current_chain_paths(
    *,
    compact_checkpoint_path: str | Path,
    stock_export_path: str | Path,
    stock_export_evidence_bundle_path: str | Path,
    gameplay_smoke_report_path: str | Path,
    game_summary_path: str | Path | None,
    gif_path: str | Path | None,
    frames_path: str | Path | None,
    standalone_eval_summary_path: str | Path | None,
    compact_checkpoint_smoke_report_path: str | Path | None,
    profile_result_path: str | Path | None,
) -> dict[str, Path]:
    paths = {
        "compact_checkpoint": Path(compact_checkpoint_path),
        "stock_export": Path(stock_export_path),
        "stock_export_evidence_bundle": Path(stock_export_evidence_bundle_path),
        "gameplay_smoke_report": Path(gameplay_smoke_report_path),
    }
    optional = {
        "game_summary": game_summary_path,
        "game_gif": gif_path,
        "frames_npz": frames_path,
        "standalone_eval_summary": standalone_eval_summary_path,
        "compact_checkpoint_smoke_report": compact_checkpoint_smoke_report_path,
        "profile_result": profile_result_path,
    }
    paths.update(
        {name: Path(path) for name, path in optional.items() if path is not None}
    )
    return paths


def _load_compact_checkpoint_metadata(path: Path) -> Mapping[str, Any]:
    payload = _torch_load_any(path, label="compact checkpoint")
    metadata = getattr(payload, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("current-chain evidence compact checkpoint missing metadata")
    schema = metadata.get("schema_id") or metadata.get(
        "compact_trainer_checkpoint_schema_id"
    )
    if schema != COMPACT_TRAINER_CHECKPOINT_SCHEMA_ID:
        raise ValueError("current-chain evidence compact checkpoint schema mismatch")
    for key in _IDENTITY_KEYS:
        if not str(metadata.get(key, "")).strip():
            raise ValueError(
                f"current-chain evidence compact checkpoint missing {key}"
            )
    if metadata.get("promotion_claim") is not False:
        raise ValueError("current-chain evidence compact checkpoint overclaims promotion")
    return metadata


def _torch_load_any(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise ValueError(f"current-chain evidence {label} missing: {path}")
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_load_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    payload = _torch_load_any(path, label=label)
    if not isinstance(payload, Mapping):
        raise ValueError(f"current-chain evidence {label} must be a mapping")
    return payload


def _stock_export_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("current-chain evidence stock export missing metadata")
    if metadata.get("schema_id") != COMPACT_STOCK_CHECKPOINT_EXPORT_SCHEMA_ID:
        raise ValueError("current-chain evidence stock export schema mismatch")
    if metadata.get("compact_stock_checkpoint_export") is not True:
        raise ValueError("current-chain evidence stock export marker missing")
    if metadata.get("stock_model_state_key") != COMPACT_STOCK_EXPORT_MODEL_STATE_KEY:
        raise ValueError("current-chain evidence stock export state key mismatch")
    if payload.get(COMPACT_STOCK_EXPORT_MODEL_STATE_KEY) is None:
        raise ValueError("current-chain evidence stock export missing model")
    if metadata.get("stock_eval_tournament_loadable") is not False:
        raise ValueError(
            "current-chain evidence stock export base loadable claim must be false"
        )
    for key in (
        "stock_model_contract_verified",
        "promotion_claim",
        "training_speed_claim",
        "calls_train_muzero",
        "touches_live_runs",
    ):
        if metadata.get(key) is not False:
            raise ValueError(f"current-chain evidence stock export {key} overclaim")
    return metadata


def _require_stock_export_matches_compact(
    *,
    compact_metadata: Mapping[str, Any],
    stock_metadata: Mapping[str, Any],
) -> None:
    for compact_key, export_key in _EXPORT_IDENTITY_KEYS.items():
        if _plain_value(compact_metadata.get(compact_key)) != _plain_value(
            stock_metadata.get(export_key)
        ):
            raise ValueError(
                f"current-chain evidence stock export {export_key} mismatch"
            )


def _validate_stock_export_bundle_for_current_chain(
    *,
    bundle: Mapping[str, Any],
    stock_export_path: Path,
    stock_metadata: Mapping[str, Any],
) -> dict[str, Path]:
    if bundle.get("schema_id") != COMPACT_STOCK_EXPORT_EVIDENCE_BUNDLE_SCHEMA_ID:
        raise ValueError("current-chain evidence stock bundle schema mismatch")
    if bundle.get("compact_stock_export_evidence_bundle") is not True:
        raise ValueError("current-chain evidence stock bundle marker missing")
    if bundle.get("ok") is not True:
        raise ValueError("current-chain evidence stock bundle ok not true")
    if bundle.get("base_export_claims_mutated") is not False:
        raise ValueError("current-chain evidence stock bundle mutated base claims")
    files = bundle.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("current-chain evidence stock bundle missing files")
    for name in (
        "checkpoint",
        "policy_metadata_sidecar",
        "verification_report",
        "tournament_loader_report",
    ):
        entry = files.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"current-chain evidence stock bundle missing {name}")
        _validate_file_record(f"stock bundle {name}", entry)
    _require_same_path_entry(
        files["checkpoint"],
        stock_export_path,
        context="stock bundle checkpoint",
    )
    claims = bundle.get("attached_claims")
    if not isinstance(claims, Mapping):
        raise ValueError("current-chain evidence stock bundle missing claims")
    for key in (
        "stock_model_contract_verified_by_report",
        "strict_stock_model_load_verified",
        "tournament_loader_constructed",
    ):
        if claims.get(key) is not True:
            raise ValueError(f"current-chain evidence stock bundle {key} not true")
    for key in (
        "promotion_claim",
        "training_speed_claim",
        "optimizer_resume_claim",
        "base_export_claims_mutated",
    ):
        if claims.get(key) is not False:
            raise ValueError(f"current-chain evidence stock bundle {key} overclaim")
    identity = bundle.get("export_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("current-chain evidence stock bundle missing export identity")
    for key in (
        "source_compact_checkpoint_id",
        "source_trainer_id",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "stock_model_state_key",
    ):
        if _plain_value(identity.get(key)) != _plain_value(stock_metadata.get(key)):
            raise ValueError(
                f"current-chain evidence stock bundle identity {key} mismatch"
            )
    paths = {
        "policy_metadata_sidecar": Path(str(files["policy_metadata_sidecar"]["path"])),
        "verification_report": Path(str(files["verification_report"]["path"])),
        "tournament_loader_report": Path(
            str(files["tournament_loader_report"]["path"])
        ),
    }
    verifier = _read_json_mapping(paths["verification_report"], label="verification report")
    _validate_verification_report(
        verifier,
        stock_export_path=stock_export_path,
        stock_metadata=stock_metadata,
    )
    loader = _read_json_mapping(paths["tournament_loader_report"], label="loader report")
    _validate_loader_report(
        loader,
        stock_export_path=stock_export_path,
        stock_metadata=stock_metadata,
    )
    return paths


def _validate_verification_report(
    report: Mapping[str, Any],
    *,
    stock_export_path: Path,
    stock_metadata: Mapping[str, Any],
) -> None:
    if report.get("schema_id") != COMPACT_STOCK_MODEL_CONTRACT_VERIFICATION_SCHEMA_ID:
        raise ValueError("current-chain evidence verifier schema mismatch")
    for key in (
        "ok",
        "sidecar_required",
        "sidecar_present",
        "strict_load",
        "stock_model_contract_verified",
    ):
        if report.get(key) is not True:
            raise ValueError(f"current-chain evidence verifier {key} not true")
    if report.get("strict_stock_model_load_verified") is not True:
        raise ValueError(
            "current-chain evidence verifier strict_stock_model_load_verified "
            "not true"
        )
    if report.get("failure_reason") is not None:
        raise ValueError("current-chain evidence verifier has failure_reason")
    _require_same_path_value(
        report.get("checkpoint_path"),
        stock_export_path,
        base=Path.cwd(),
        context="verifier checkpoint_path",
    )
    if report.get("state_key") != COMPACT_STOCK_EXPORT_MODEL_STATE_KEY:
        raise ValueError("current-chain evidence verifier state key mismatch")
    _require_report_identity(report=report, stock_metadata=stock_metadata)
    load = report.get("load_summary")
    if not isinstance(load, Mapping):
        raise ValueError("current-chain evidence verifier missing load summary")
    _validate_strict_load_summary(load, context="verifier load_summary")


def _validate_loader_report(
    report: Mapping[str, Any],
    *,
    stock_export_path: Path,
    stock_metadata: Mapping[str, Any],
) -> None:
    if report.get("schema_id") != COMPACT_STOCK_TOURNAMENT_LOADER_SMOKE_SCHEMA_ID:
        raise ValueError("current-chain evidence loader schema mismatch")
    if report.get("ok") is not True:
        raise ValueError("current-chain evidence loader ok not true")
    _require_same_path_value(
        report.get("checkpoint_path"),
        stock_export_path,
        base=Path.cwd(),
        context="loader checkpoint_path",
    )
    if report.get("checkpoint_state_key") != COMPACT_STOCK_EXPORT_MODEL_STATE_KEY:
        raise ValueError("current-chain evidence loader state key mismatch")
    _require_policy_and_runtime_match(
        report=report,
        stock_metadata=stock_metadata,
        context="loader",
    )
    surface = report.get("surface")
    if not isinstance(surface, Mapping):
        raise ValueError("current-chain evidence loader missing surface")
    load = surface.get("load_state_dict")
    if not isinstance(load, Mapping):
        raise ValueError("current-chain evidence loader missing load_state_dict")
    _validate_strict_load_summary(load, context="loader load_state_dict")


def _validate_gameplay_smoke_report_for_current_chain(
    *,
    report: Mapping[str, Any],
    report_path: Path,
    stock_export_path: Path,
    stock_metadata: Mapping[str, Any],
    stock_bundle_path: Path,
    bundle_paths: Mapping[str, Path],
    game_summary_path: Path | None,
    gif_path: Path | None,
    frames_path: Path | None,
    standalone_eval_summary_path: Path | None,
) -> dict[str, Path]:
    if report.get("schema_id") != COMPACT_STOCK_ONE_GAME_GIF_SMOKE_SCHEMA_ID:
        raise ValueError("current-chain evidence gameplay report schema mismatch")
    if report.get("ok") is not True:
        raise ValueError("current-chain evidence gameplay report ok not true")
    for key in (
        "promotion_claim",
        "training_speed_claim",
        "calls_train_muzero",
        "touches_live_runs",
    ):
        if report.get(key) is not False:
            raise ValueError(f"current-chain evidence gameplay {key} overclaim")
    _require_same_path_value(
        report.get("checkpoint_ref"),
        stock_export_path,
        base=report_path.parent,
        context="gameplay checkpoint_ref",
    )
    _require_same_path_value(
        report.get("evidence_bundle_ref"),
        stock_bundle_path,
        base=report_path.parent,
        context="gameplay evidence_bundle_ref",
    )
    _require_same_path_value(
        report.get("verification_report_ref"),
        bundle_paths["verification_report"],
        base=report_path.parent,
        context="gameplay verification_report_ref",
    )
    _require_same_path_value(
        report.get("tournament_loader_report_ref"),
        bundle_paths["tournament_loader_report"],
        base=report_path.parent,
        context="gameplay tournament_loader_report_ref",
    )
    _require_same_path_value(
        report.get("sidecar_ref"),
        bundle_paths["policy_metadata_sidecar"],
        base=report_path.parent,
        context="gameplay sidecar_ref",
    )
    if int(report.get("physical_steps") or 0) <= 0:
        raise ValueError("current-chain evidence gameplay physical_steps missing")
    first_joint = _first_joint_action(report)
    if not first_joint:
        raise ValueError("current-chain evidence gameplay first joint action missing")
    for action in first_joint:
        if int(action) not in (0, 1, 2):
            raise ValueError("current-chain evidence gameplay illegal action")
    policy_loads = report.get("policy_loads")
    if not isinstance(policy_loads, list) or not policy_loads:
        raise ValueError("current-chain evidence gameplay missing policy loads")
    for load_report in policy_loads:
        if not isinstance(load_report, Mapping):
            raise ValueError("current-chain evidence gameplay policy load malformed")
        _require_same_path_value(
            load_report.get("checkpoint_path") or load_report.get("checkpoint_ref"),
            stock_export_path,
            base=report_path.parent,
            context="gameplay policy checkpoint",
        )
        if load_report.get("checkpoint_state_key") != COMPACT_STOCK_EXPORT_MODEL_STATE_KEY:
            raise ValueError("current-chain evidence gameplay policy state key mismatch")
        _require_policy_and_runtime_match(
            report=load_report,
            stock_metadata=stock_metadata,
            context="gameplay policy load",
        )
        surface = load_report.get("surface")
        if not isinstance(surface, Mapping):
            raise ValueError("current-chain evidence gameplay policy missing surface")
        state = surface.get("load_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError(
                "current-chain evidence gameplay policy missing load_state_dict"
            )
        _validate_strict_load_summary(
            state,
            context="gameplay policy load_state_dict",
        )
    game_summary = _resolve_required_report_artifact(
        report_path=report_path,
        ref=report.get("game_summary_ref"),
        explicit_path=game_summary_path,
        context="game summary",
    )
    gif = _resolve_required_report_artifact(
        report_path=report_path,
        ref=report.get("gif_ref"),
        explicit_path=gif_path,
        context="game gif",
    )
    frames = _resolve_required_report_artifact(
        report_path=report_path,
        ref=report.get("frames_ref"),
        explicit_path=frames_path,
        context="frames npz",
    )
    summary_payload = _read_json_mapping(game_summary, label="game summary")
    if summary_payload.get("ok") is not True:
        raise ValueError("current-chain evidence game summary ok not true")
    for key, path in (
        ("summary_ref", game_summary),
        ("gif_ref", gif),
        ("frames_ref", frames),
    ):
        if summary_payload.get(key):
            _require_same_path_value(
                summary_payload.get(key),
                path,
                base=report_path.parent,
                context=f"game summary {key}",
            )
    standalone_eval_path = _resolve_standalone_eval_summary_path(
        report=report,
        report_path=report_path,
        explicit_path=standalone_eval_summary_path,
    )
    _validate_standalone_eval(report)
    return {
        "game_summary": game_summary,
        "game_gif": gif,
        "frames_npz": frames,
        "standalone_eval_summary": standalone_eval_path,
    }


def _validate_standalone_eval(report: Mapping[str, Any]) -> None:
    standalone = report.get("standalone_eval")
    if not isinstance(standalone, Mapping):
        raise ValueError("current-chain evidence missing standalone eval")
    if standalone.get("ok") is not True:
        raise ValueError("current-chain evidence standalone eval ok not true")
    status = _standalone_eval_status(report)
    for key in (
        "checkpoint_load_ok",
        "strict_policy_model_load_ok",
        "env_reset_ok",
        "policy_could_act_in_real_env",
    ):
        if status.get(key) is not True:
            raise ValueError(f"current-chain evidence standalone eval {key} not true")
    if int(status.get("steps_survived") or 0) <= 0:
        raise ValueError("current-chain evidence standalone eval steps missing")
    load = standalone.get("load_state_dict")
    if not isinstance(load, Mapping):
        raise ValueError("current-chain evidence standalone eval missing load summary")
    _validate_strict_load_summary(load, context="standalone eval load_state_dict")


def _resolve_standalone_eval_summary_path(
    *,
    report: Mapping[str, Any],
    report_path: Path,
    explicit_path: Path | None,
) -> Path:
    standalone = report.get("standalone_eval")
    if not isinstance(standalone, Mapping):
        raise ValueError("current-chain evidence missing standalone eval")
    artifact = standalone.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("current-chain evidence standalone eval missing artifact")
    path_value = artifact.get("path") or artifact.get("ref")
    return _resolve_required_report_artifact(
        report_path=report_path,
        ref=path_value,
        explicit_path=explicit_path,
        context="standalone eval summary",
    )


def _resolve_required_report_artifact(
    *,
    report_path: Path,
    ref: Any,
    explicit_path: Path | None,
    context: str,
) -> Path:
    if ref is None and explicit_path is None:
        raise ValueError(f"current-chain evidence {context} ref missing")
    resolved = explicit_path if explicit_path is not None else _resolve_ref_path(
        ref,
        base=report_path.parent,
    )
    if not resolved.is_file():
        raise ValueError(f"current-chain evidence {context} missing: {resolved}")
    if ref is not None:
        _require_same_path_value(ref, resolved, base=report_path.parent, context=context)
    return resolved


def _require_policy_and_runtime_match(
    *,
    report: Mapping[str, Any],
    stock_metadata: Mapping[str, Any],
    context: str,
) -> None:
    for key in _POLICY_OBSERVATION_KEYS:
        if _plain_value(report.get(key)) != _plain_value(stock_metadata.get(key)):
            raise ValueError(f"current-chain evidence {context} {key} mismatch")
    for key in ("model_env_variant", "model_reward_variant"):
        if _plain_value(report.get(key)) != _plain_value(stock_metadata.get(key)):
            raise ValueError(f"current-chain evidence {context} {key} mismatch")
    runtime = report.get("runtime_settings")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"current-chain evidence {context} missing runtime settings")
    for key in _RUNTIME_KEYS:
        if _plain_value(runtime.get(key)) != _plain_value(stock_metadata.get(key)):
            raise ValueError(f"current-chain evidence {context} runtime {key} mismatch")


def _require_report_identity(
    *,
    report: Mapping[str, Any],
    stock_metadata: Mapping[str, Any],
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
        if _plain_value(report.get(key)) != _plain_value(stock_metadata.get(key)):
            raise ValueError(f"current-chain evidence verifier {key} mismatch")


def _validate_strict_load_summary(summary: Mapping[str, Any], *, context: str) -> None:
    if summary.get("ok") is not True:
        raise ValueError(f"current-chain evidence {context} ok not true")
    if summary.get("strict") is not True:
        raise ValueError(f"current-chain evidence {context} strict not true")
    if list(summary.get("missing_keys") or []) != []:
        raise ValueError(f"current-chain evidence {context} missing keys")
    if list(summary.get("unexpected_keys") or []) != []:
        raise ValueError(f"current-chain evidence {context} unexpected keys")


def _first_joint_action(report: Mapping[str, Any]) -> list[int]:
    trace = report.get("first_action_trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("current-chain evidence gameplay missing first_action_trace")
    first = trace[0]
    if not isinstance(first, Mapping):
        raise ValueError("current-chain evidence first_action_trace malformed")
    action = first.get("joint_action")
    if not isinstance(action, list):
        raise ValueError("current-chain evidence first joint action malformed")
    return [int(value) for value in action]


def _standalone_eval_status(report: Mapping[str, Any]) -> Mapping[str, Any]:
    standalone = report.get("standalone_eval")
    if not isinstance(standalone, Mapping):
        raise ValueError("current-chain evidence missing standalone eval")
    status = standalone.get("status")
    if not isinstance(status, Mapping):
        raise ValueError("current-chain evidence standalone eval missing status")
    return status


def _current_chain_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _plain_value(metadata[key]) for key in _IDENTITY_KEYS}


def _stock_export_identity(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_compact_checkpoint_id": _plain_value(
            metadata["source_compact_checkpoint_id"]
        ),
        "source_trainer_id": _plain_value(metadata["source_trainer_id"]),
        "policy_version_ref": _plain_value(metadata["policy_version_ref"]),
        "model_version_ref": _plain_value(metadata["model_version_ref"]),
        "policy_source": _plain_value(metadata["policy_source"]),
        "stock_model_state_key": _plain_value(metadata["stock_model_state_key"]),
    }


def _read_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"current-chain evidence {label} missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise ValueError(f"current-chain evidence {label} unreadable JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"current-chain evidence {label} must be a mapping")
    return dict(payload)


def _file_record(path: Path, *, required: bool) -> dict[str, Any]:
    if required and not path.is_file():
        raise ValueError(f"current-chain evidence file missing: {path}")
    return {
        "path": str(path),
        "sha256": _file_sha256(path),
        "size_bytes": int(path.stat().st_size),
        "required": bool(required),
    }


def _validate_file_record(name: str, entry: Mapping[str, Any]) -> None:
    path_value = entry.get("path")
    if not path_value:
        raise ValueError(f"current-chain evidence {name} path missing")
    path = Path(str(path_value))
    if not path.is_file():
        raise ValueError(f"current-chain evidence {name} file missing")
    expected_sha = str(entry.get("sha256") or "")
    if not expected_sha:
        raise ValueError(f"current-chain evidence {name} sha256 missing")
    if _file_sha256(path) != expected_sha:
        raise ValueError(f"current-chain evidence {name} sha256 mismatch")
    if int(path.stat().st_size) != int(entry.get("size_bytes")):
        raise ValueError(f"current-chain evidence {name} size mismatch")


def _require_same_path_entry(
    entry: Mapping[str, Any],
    expected: Path,
    *,
    context: str,
) -> None:
    _require_same_path_value(
        entry.get("path"),
        expected,
        base=Path.cwd(),
        context=context,
    )


def _require_same_path_value(
    value: Any,
    expected: Path,
    *,
    base: Path,
    context: str,
) -> None:
    if value is None:
        raise ValueError(f"current-chain evidence {context} missing")
    actual = _resolve_ref_path(value, base=base)
    try:
        matches = actual.resolve() == expected.resolve()
    except Exception:
        matches = False
    if not matches:
        raise ValueError(f"current-chain evidence {context} mismatch")


def _resolve_ref_path(value: Any, *, base: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_EVIDENCE_SCHEMA_ID",
    "COMPACT_CURRENT_CHAIN_EVAL_GIF_TOURNAMENT_LOAD_STATUS_VERIFIED",
    "build_compact_current_chain_eval_gif_tournament_load_evidence_v1",
    "compact_current_chain_eval_gif_tournament_load_evidence_path",
    "compact_current_chain_eval_gif_tournament_load_evidence_ref",
    "save_compact_current_chain_eval_gif_tournament_load_evidence_v1",
    "validate_compact_current_chain_eval_gif_tournament_load_evidence_v1",
    "validate_compact_current_chain_eval_gif_tournament_load_matches_checkpoint_v1",
]
