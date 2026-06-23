"""Fail-closed metrics-lineage contract for compact-owned training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


COMPACT_TRAINING_METRICS_LINEAGE_SCHEMA_ID = (
    "curvyzero_compact_training_metrics_lineage/v1"
)

REQUIRED_LEARNER_METRIC_KEYS = (
    "compact_muzero_learner_loss",
    "compact_muzero_learner_policy_loss",
    "compact_muzero_learner_value_loss",
    "compact_muzero_learner_reward_loss",
    "compact_muzero_learner_grad_norm_before_clip",
)


def build_compact_training_metrics_lineage_v1(
    *,
    checkpoint_id: str,
    resume_state: Any,
    replay_store_state: Any,
    metrics: Mapping[str, Any],
    loop_runtime_state: Any,
    evidence_refs: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Build a metrics lineage proof from checkpoint-visible training state."""

    checkpoint_id_value = str(checkpoint_id).strip()
    if not checkpoint_id_value:
        raise ValueError("checkpoint_id must be non-empty")
    metrics_dict = _plain_mapping(metrics)
    learner_telemetry = _learner_telemetry_from_metrics(metrics_dict)
    loop_counters = _loop_counters(loop_runtime_state)
    replay_metadata = _replay_metadata(replay_store_state)
    refs = tuple(str(ref).strip() for ref in evidence_refs if str(ref).strip())
    if not refs:
        raise ValueError("training metrics lineage evidence_refs must be non-empty")

    policy_ref = str(getattr(resume_state, "policy_version_ref", "")).strip()
    model_ref = str(getattr(resume_state, "model_version_ref", "")).strip()
    trainer_id = str(getattr(resume_state, "trainer_id", "")).strip()
    if not policy_ref:
        raise ValueError("training metrics lineage policy_version_ref is required")
    if not model_ref:
        raise ValueError("training metrics lineage model_version_ref is required")
    if not trainer_id:
        raise ValueError("training metrics lineage trainer_id is required")
    if int(getattr(resume_state, "train_step", -1)) <= 0:
        raise ValueError("training metrics lineage requires train_step > 0")
    if int(getattr(resume_state, "learner_update_count", -1)) <= 0:
        raise ValueError(
            "training metrics lineage requires learner_update_count > 0"
        )
    if int(getattr(resume_state, "sample_batch_count", -1)) <= 0:
        raise ValueError("training metrics lineage requires sample_batch_count > 0")

    _validate_replay_metadata(
        replay_metadata,
        expected_policy_version_ref=policy_ref,
        expected_model_version_ref=model_ref,
    )
    _validate_learner_telemetry(learner_telemetry)
    _validate_loop_counters(loop_counters, learner_telemetry)
    search_provenance = _search_provenance_from_sample_metadata(
        loop_counters["sample_gate_last_sample_metadata"]
    )

    lineage = {
        "compact_training_metrics_lineage_schema_id": (
            COMPACT_TRAINING_METRICS_LINEAGE_SCHEMA_ID
        ),
        "training_metrics_lineage_status": "compact_training_metrics_lineage_v1",
        "checkpoint_id": checkpoint_id_value,
        "trainer_id": trainer_id,
        "policy_version_ref": policy_ref,
        "model_version_ref": model_ref,
        "policy_source": str(getattr(resume_state, "policy_source", "")).strip(),
        "train_step": int(getattr(resume_state, "train_step")),
        "learner_update_count": int(getattr(resume_state, "learner_update_count")),
        "sample_batch_count": int(getattr(resume_state, "sample_batch_count")),
        "learner_metric_keys": list(REQUIRED_LEARNER_METRIC_KEYS),
        "learner_loss": float(learner_telemetry["compact_muzero_learner_loss"]),
        "learner_policy_loss": float(
            learner_telemetry["compact_muzero_learner_policy_loss"]
        ),
        "learner_value_loss": float(
            learner_telemetry["compact_muzero_learner_value_loss"]
        ),
        "learner_reward_loss": float(
            learner_telemetry["compact_muzero_learner_reward_loss"]
        ),
        "learner_grad_norm_before_clip": float(
            learner_telemetry["compact_muzero_learner_grad_norm_before_clip"]
        ),
        "learner_sample_rows": int(
            learner_telemetry["compact_muzero_learner_sample_rows"]
        ),
        "learner_train_steps": int(
            learner_telemetry["compact_muzero_learner_train_steps"]
        ),
        "learner_input_bytes": int(
            learner_telemetry.get("compact_muzero_learner_input_bytes", 0)
        ),
        "resident_sample_used": bool(
            learner_telemetry["compact_muzero_learner_resident_sample_used"]
        ),
        "device_replay_index_rows_sample": bool(
            learner_telemetry[
                "compact_muzero_learner_device_replay_index_rows_sample"
            ]
        ),
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
        "training_speed_claim": False,
        "replay_store_state_schema_id": str(
            replay_metadata["compact_replay_store_state_schema_id"]
        ),
        "replay_store_entry_count": int(
            replay_metadata.get("compact_replay_store_entry_count", 0)
        ),
        "replay_store_index_row_count": int(
            replay_metadata.get(
                "compact_replay_store_index_row_count",
                replay_metadata.get("compact_replay_store_stored_index_row_count", 0),
            )
        ),
        "compact_owned_loop_schema_id": str(
            replay_metadata["compact_owned_loop_schema_id"]
        ),
        "sample_gate_calls": int(loop_counters["sample_gate_calls"]),
        "sample_gate_sample_rows": int(loop_counters["sample_gate_sample_rows"]),
        "learner_gate_updates": int(loop_counters["learner_gate_updates"]),
        "learner_gate_sample_rows": int(loop_counters["learner_gate_sample_rows"]),
        "learner_gate_input_bytes": int(loop_counters["learner_gate_input_bytes"]),
        "search_provenance": search_provenance,
        "evidence_refs": list(refs),
    }
    validate_compact_training_metrics_lineage_v1(lineage)
    return lineage


def validate_compact_training_metrics_lineage_v1(lineage: Any) -> None:
    """Validate a compact training metrics lineage proof."""

    if not isinstance(lineage, Mapping):
        raise ValueError("training metrics lineage must be a mapping")
    schema_id = lineage.get("compact_training_metrics_lineage_schema_id")
    if schema_id != COMPACT_TRAINING_METRICS_LINEAGE_SCHEMA_ID:
        raise ValueError("training metrics lineage schema mismatch")
    for key in (
        "checkpoint_id",
        "trainer_id",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "training_metrics_lineage_status",
        "replay_store_state_schema_id",
        "compact_owned_loop_schema_id",
    ):
        if not str(lineage.get(key, "")).strip():
            raise ValueError(f"training metrics lineage missing {key}")
    if lineage.get("training_metrics_lineage_status") != (
        "compact_training_metrics_lineage_v1"
    ):
        raise ValueError("training metrics lineage status mismatch")
    for key in (
        "learner_loss",
        "learner_policy_loss",
        "learner_value_loss",
        "learner_reward_loss",
        "learner_grad_norm_before_clip",
    ):
        if not _is_finite_number(lineage.get(key)):
            raise ValueError(f"training metrics lineage non-finite {key}")
    for key in (
        "train_step",
        "learner_update_count",
        "sample_batch_count",
        "learner_sample_rows",
        "learner_train_steps",
        "sample_gate_calls",
        "sample_gate_sample_rows",
        "learner_gate_updates",
        "learner_gate_sample_rows",
    ):
        if int(lineage.get(key, 0)) <= 0:
            raise ValueError(f"training metrics lineage requires {key} > 0")
    for key in (
        "learner_input_bytes",
        "learner_gate_input_bytes",
    ):
        if int(lineage.get(key, -1)) < 0:
            raise ValueError(f"training metrics lineage requires {key} >= 0")
    for key in ("replay_store_entry_count", "replay_store_index_row_count"):
        if int(lineage.get(key, 0)) <= 0:
            raise ValueError(f"training metrics lineage requires {key} > 0")
    if lineage.get("resident_sample_used") is not True:
        raise ValueError("training metrics lineage requires resident samples")
    if lineage.get("device_replay_index_rows_sample") is not True:
        raise ValueError(
            "training metrics lineage requires device replay index sample rows"
        )
    for key in (
        "calls_train_muzero",
        "touches_live_runs",
        "promotion_claim",
        "training_speed_claim",
    ):
        if lineage.get(key) is not False:
            raise ValueError(f"training metrics lineage {key} must be false")
    evidence_refs = lineage.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not any(
        str(ref).strip() for ref in evidence_refs
    ):
        raise ValueError("training metrics lineage evidence_refs must be non-empty")
    _validate_search_provenance(lineage.get("search_provenance"))


def compact_training_metrics_lineage_evidence_ref(
    lineage: Mapping[str, Any],
) -> str:
    """Return the compact Coach evidence ref for a validated metrics lineage."""

    validate_compact_training_metrics_lineage_v1(lineage)
    return (
        "compact_training_metrics_lineage:"
        f"{lineage['checkpoint_id']}:"
        f"updates={lineage['learner_update_count']}:"
        f"samples={lineage['sample_batch_count']}"
    )


def _learner_telemetry_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = metrics.get("last_learner_telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError(
            "training metrics lineage requires metrics['last_learner_telemetry']"
        )
    return _plain_mapping(telemetry)


def _loop_counters(loop_runtime_state: Any) -> dict[str, Any]:
    if loop_runtime_state is None:
        raise ValueError("training metrics lineage requires loop_runtime_state")
    counters = getattr(loop_runtime_state, "counters", None)
    if counters is None:
        raise ValueError("training metrics lineage missing loop counters")
    return {
        "sample_gate_calls": int(getattr(counters, "sample_gate_calls", 0)),
        "sample_gate_sample_rows": int(
            getattr(counters, "sample_gate_sample_rows", 0)
        ),
        "sample_gate_last_sample_metadata": _plain_mapping(
            getattr(counters, "sample_gate_last_sample_metadata", {}) or {}
        ),
        "learner_gate_updates": int(getattr(counters, "learner_gate_updates", 0)),
        "learner_gate_sample_rows": int(
            getattr(counters, "learner_gate_sample_rows", 0)
        ),
        "learner_gate_input_bytes": int(
            getattr(counters, "learner_gate_input_bytes", 0)
        ),
        "learner_gate_last_telemetry": _plain_mapping(
            getattr(counters, "learner_gate_last_telemetry", {}) or {}
        ),
    }


def _replay_metadata(replay_store_state: Any) -> dict[str, Any]:
    metadata = getattr(replay_store_state, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("training metrics lineage requires replay-store metadata")
    return _plain_mapping(metadata)


def _validate_replay_metadata(
    metadata: dict[str, Any],
    *,
    expected_policy_version_ref: str,
    expected_model_version_ref: str,
) -> None:
    schema_id = metadata.get("schema_id") or metadata.get(
        "compact_replay_store_state_schema_id"
    )
    if not str(schema_id).strip():
        raise ValueError("training metrics lineage missing replay-store schema")
    metadata["compact_replay_store_state_schema_id"] = str(schema_id)
    if metadata.get("compact_owned_loop_replay_store_owned") is not True:
        raise ValueError("training metrics lineage requires compact-owned replay")
    if metadata.get("compact_owned_loop_policy_version_handoff") is not True:
        raise ValueError("training metrics lineage missing policy handoff metadata")
    if not str(metadata.get("compact_owned_loop_schema_id", "")).strip():
        raise ValueError("training metrics lineage missing compact-owned loop schema")
    policy_ref = str(
        metadata.get("policy_version_ref")
        or metadata.get("compact_replay_store_policy_version_ref")
        or ""
    )
    if policy_ref != str(expected_policy_version_ref):
        raise ValueError("training metrics lineage policy ref mismatch")
    model_ref = str(metadata.get("model_version_ref") or "")
    if model_ref != str(expected_model_version_ref):
        raise ValueError("training metrics lineage model ref mismatch")
    if metadata.get("calls_train_muzero") is not False:
        raise ValueError("training metrics lineage replay claims train_muzero")


def _validate_learner_telemetry(telemetry: Mapping[str, Any]) -> None:
    if telemetry.get("compact_muzero_learner_calls_train_muzero") is not False:
        raise ValueError("training metrics lineage learner called train_muzero")
    if telemetry.get("compact_muzero_learner_update_claim") is not True:
        raise ValueError("training metrics lineage requires learner update claim")
    for key in REQUIRED_LEARNER_METRIC_KEYS:
        if not _is_finite_number(telemetry.get(key)):
            raise ValueError(f"training metrics lineage non-finite {key}")
    for key in (
        "compact_muzero_learner_sample_rows",
        "compact_muzero_learner_train_steps",
    ):
        if int(telemetry.get(key, 0)) <= 0:
            raise ValueError(f"training metrics lineage requires {key} > 0")
    if int(telemetry.get("compact_muzero_learner_input_bytes", 0)) < 0:
        raise ValueError("training metrics lineage input bytes must be non-negative")
    if telemetry.get("compact_muzero_learner_resident_sample_used") is not True:
        raise ValueError("training metrics lineage requires resident learner sample")
    if telemetry.get("compact_muzero_learner_device_replay_index_rows_sample") is not True:
        raise ValueError(
            "training metrics lineage requires device replay index sample telemetry"
        )


def _validate_loop_counters(
    counters: Mapping[str, Any],
    learner_telemetry: Mapping[str, Any],
) -> None:
    learner_sample_rows = int(
        learner_telemetry.get("compact_muzero_learner_sample_rows", 0)
    )
    for key in (
        "sample_gate_calls",
        "sample_gate_sample_rows",
        "learner_gate_updates",
        "learner_gate_sample_rows",
    ):
        if int(counters.get(key, 0)) <= 0:
            raise ValueError(f"training metrics lineage loop counter {key} <= 0")
    if int(counters["sample_gate_sample_rows"]) < learner_sample_rows:
        raise ValueError("training metrics lineage sample rows exceed sample gate rows")
    if int(counters["learner_gate_sample_rows"]) < learner_sample_rows:
        raise ValueError("training metrics lineage sample rows exceed learner gate rows")
    sample_metadata = counters.get("sample_gate_last_sample_metadata", {})
    if sample_metadata.get("resident_device_sample_batch") is not True:
        raise ValueError("training metrics lineage sample metadata is not resident")
    if sample_metadata.get("device_replay_index_rows_sample") is not True:
        raise ValueError(
            "training metrics lineage sample metadata lacks device replay index rows"
        )
    if int(counters.get("learner_gate_input_bytes", 0)) < 0:
        raise ValueError("training metrics lineage learner input bytes negative")


def _search_provenance_from_sample_metadata(
    sample_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    search_impl = _first_nonempty(
        sample_metadata.get("search_impl"),
        _single_value(sample_metadata.get("search_impls")),
    )
    root_schema = _first_nonempty(
        sample_metadata.get("root_batch_schema_id"),
        _single_value(sample_metadata.get("root_batch_schema_ids")),
    )
    search_result_schema = _first_nonempty(
        sample_metadata.get("search_result_schema_id"),
        _single_value(sample_metadata.get("search_result_schema_ids")),
    )
    replay_payload_schema = _first_nonempty(
        sample_metadata.get("replay_payload_schema_id"),
        _single_value(sample_metadata.get("replay_payload_schema_ids")),
    )
    digest = _first_nonempty(
        sample_metadata.get("search_replay_payload_digest"),
        _single_value(sample_metadata.get("search_replay_payload_digests")),
    )
    provenance = {
        "search_impl": search_impl,
        "num_simulations": int(
            sample_metadata.get(
                "num_simulations",
                _single_value(sample_metadata.get("num_simulations_values"), default=0),
            )
        ),
        "active_root_count": int(sample_metadata.get("active_root_count", 0)),
        "root_batch_schema_id": root_schema,
        "search_result_schema_id": search_result_schema,
        "replay_payload_schema_id": replay_payload_schema,
        "search_replay_payload_digest": digest,
    }
    _validate_search_provenance(provenance)
    return provenance


def _validate_search_provenance(provenance: Any) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError("training metrics lineage requires search provenance")
    for key in (
        "search_impl",
        "root_batch_schema_id",
        "search_result_schema_id",
        "replay_payload_schema_id",
        "search_replay_payload_digest",
    ):
        if not str(provenance.get(key, "")).strip():
            raise ValueError(f"training metrics lineage missing search {key}")
    if int(provenance.get("num_simulations", -1)) < 0:
        raise ValueError("training metrics lineage search num_simulations negative")
    if int(provenance.get("active_root_count", 0)) <= 0:
        raise ValueError("training metrics lineage search active_root_count <= 0")


def _single_value(value: Any, *, default: Any = "") -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return default


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


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


__all__ = [
    "COMPACT_TRAINING_METRICS_LINEAGE_SCHEMA_ID",
    "REQUIRED_LEARNER_METRIC_KEYS",
    "build_compact_training_metrics_lineage_v1",
    "compact_training_metrics_lineage_evidence_ref",
    "validate_compact_training_metrics_lineage_v1",
]
