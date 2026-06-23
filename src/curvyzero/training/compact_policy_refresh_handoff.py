"""Fail-closed policy-refresh handoff contract for compact-owned training."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any

import numpy as np


COMPACT_POLICY_REFRESH_HANDOFF_SCHEMA_ID = (
    "curvyzero_compact_policy_refresh_handoff/v1"
)
COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID = (
    "curvyzero_compact_policy_refresh_search_worker_state/v1"
)

COMPACT_POLICY_REFRESH_METADATA_KEYS = (
    "policy_version_ref",
    "model_version_ref",
    "policy_source",
    "compact_policy_refresh_handoff_state_schema_id",
    "compact_policy_refresh_model_state_digest",
    "compact_policy_refresh_model_state_digest_source",
    "compact_policy_refresh_learner_update_count",
    "compact_policy_refresh_search_worker_refreshed",
    "compact_policy_refresh_count",
)


def compact_model_state_digest_v1(model_or_state: Any) -> str:
    """Return a stable digest for a model/state-dict visible to the checkpoint."""

    state = _state_mapping(model_or_state)
    digest = hashlib.sha256()
    digest.update(b"curvyzero_compact_model_state_digest/v1")
    for key in sorted(str(key) for key in state):
        digest.update(key.encode("utf-8"))
        _update_digest_with_value(digest, state[key])
    return digest.hexdigest()


def compact_policy_refresh_metadata_from_state_v1(
    search_worker_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return row metadata that stamps refreshed search-worker identity."""

    validate_compact_policy_refresh_search_worker_state_v1(search_worker_state)
    metadata = {
        "policy_version_ref": str(search_worker_state["policy_version_ref"]),
        "model_version_ref": str(search_worker_state["model_version_ref"]),
        "policy_source": str(search_worker_state["policy_source"]),
        "compact_policy_refresh_handoff_state_schema_id": (
            COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID
        ),
        "compact_policy_refresh_model_state_digest": str(
            search_worker_state["model_state_digest"]
        ),
        "compact_policy_refresh_learner_update_count": int(
            search_worker_state["learner_update_count"]
        ),
        "compact_policy_refresh_search_worker_refreshed": True,
        "compact_policy_refresh_count": int(search_worker_state["refresh_count"]),
    }
    digest_source = str(search_worker_state.get("model_state_digest_source") or "").strip()
    if digest_source:
        metadata["compact_policy_refresh_model_state_digest_source"] = digest_source
    return metadata


def compact_policy_refresh_metadata_subset_v1(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy only policy-refresh row-stamp metadata from a wider payload."""

    return {
        key: metadata[key]
        for key in COMPACT_POLICY_REFRESH_METADATA_KEYS
        if key in metadata
    }


def build_compact_policy_refresh_handoff_v1(
    *,
    checkpoint_id: str,
    resume_state: Any,
    learner_model: Any,
    search_worker_state: Mapping[str, Any],
    root_metadata: Mapping[str, Any],
    action_metadata: Mapping[str, Any],
    replay_metadata: Mapping[str, Any],
    sample_metadata: Mapping[str, Any],
    evidence_refs: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Build a policy-refresh proof from learner/search/checkpoint-visible state."""

    checkpoint_id_value = str(checkpoint_id).strip()
    if not checkpoint_id_value:
        raise ValueError("policy refresh checkpoint_id must be non-empty")
    validate_compact_policy_refresh_search_worker_state_v1(search_worker_state)
    refs = tuple(str(ref).strip() for ref in evidence_refs if str(ref).strip())
    if not refs:
        raise ValueError("policy refresh evidence_refs must be non-empty")

    policy_ref = str(getattr(resume_state, "policy_version_ref", "")).strip()
    model_ref = str(getattr(resume_state, "model_version_ref", "")).strip()
    policy_source = str(getattr(resume_state, "policy_source", "")).strip()
    trainer_id = str(getattr(resume_state, "trainer_id", "")).strip()
    learner_update_count = int(getattr(resume_state, "learner_update_count", -1))
    if not trainer_id:
        raise ValueError("policy refresh trainer_id is required")
    if not policy_ref:
        raise ValueError("policy refresh policy_version_ref is required")
    if not model_ref:
        raise ValueError("policy refresh model_version_ref is required")
    if not policy_source:
        raise ValueError("policy refresh policy_source is required")
    if learner_update_count <= 0:
        raise ValueError("policy refresh requires learner_update_count > 0")

    learner_digest = compact_model_state_digest_v1(learner_model)
    search_digest = str(search_worker_state["model_state_digest"])
    if learner_digest != search_digest:
        raise ValueError("policy refresh learner/search model digest mismatch")
    if int(search_worker_state["learner_update_count"]) != learner_update_count:
        raise ValueError("policy refresh learner update count mismatch")
    for key, expected in (
        ("policy_version_ref", policy_ref),
        ("model_version_ref", model_ref),
        ("policy_source", policy_source),
    ):
        if str(search_worker_state.get(key, "")).strip() != expected:
            raise ValueError(f"policy refresh search worker {key} mismatch")
    learner_object_id = id(learner_model)
    search_object_id = int(search_worker_state["search_worker_model_object_id"])
    if search_object_id == learner_object_id:
        raise ValueError("policy refresh requires a distinct search worker model")

    for label, metadata in (
        ("root", root_metadata),
        ("action", action_metadata),
        ("replay", replay_metadata),
        ("sample", sample_metadata),
    ):
        _validate_policy_refresh_row_metadata(
            metadata,
            label=label,
            expected_policy_version_ref=policy_ref,
            expected_model_version_ref=model_ref,
            expected_policy_source=policy_source,
            expected_model_state_digest=learner_digest,
            expected_learner_update_count=learner_update_count,
        )

    lineage = {
        "compact_policy_refresh_handoff_schema_id": (
            COMPACT_POLICY_REFRESH_HANDOFF_SCHEMA_ID
        ),
        "policy_refresh_handoff_status": "compact_policy_refresh_handoff_v1",
        "checkpoint_id": checkpoint_id_value,
        "trainer_id": trainer_id,
        "policy_version_ref": policy_ref,
        "model_version_ref": model_ref,
        "policy_source": policy_source,
        "learner_update_count": learner_update_count,
        "learner_model_state_digest": learner_digest,
        "search_worker_model_state_digest": search_digest,
        "search_worker_model_object_id": search_object_id,
        "learner_model_object_id": learner_object_id,
        "search_worker_distinct_from_learner": True,
        "search_worker_refresh_count": int(search_worker_state["refresh_count"]),
        "search_worker_cache_cleared": bool(search_worker_state["cache_cleared"]),
        "root_rows_stamped": True,
        "action_rows_stamped": True,
        "replay_rows_stamped": True,
        "sample_rows_stamped": True,
        "calls_train_muzero": False,
        "touches_live_runs": False,
        "promotion_claim": False,
        "training_speed_claim": False,
        "evidence_refs": list(refs),
    }
    validate_compact_policy_refresh_handoff_v1(lineage)
    return lineage


def validate_compact_policy_refresh_search_worker_state_v1(state: Any) -> None:
    """Validate search-worker refresh state."""

    if not isinstance(state, Mapping):
        raise ValueError("policy refresh search worker state must be a mapping")
    if state.get("schema_id") != COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID:
        raise ValueError("policy refresh search worker state schema mismatch")
    for key in (
        "search_impl",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "model_state_digest",
    ):
        if not str(state.get(key, "")).strip():
            raise ValueError(f"policy refresh search worker missing {key}")
    for key in ("learner_update_count", "refresh_count", "search_worker_model_object_id"):
        if int(state.get(key, 0)) <= 0:
            raise ValueError(f"policy refresh search worker requires {key} > 0")
    if state.get("refresh_applied") is not True:
        raise ValueError("policy refresh search worker refresh_applied must be true")
    if state.get("cache_cleared") is not True:
        raise ValueError("policy refresh search worker cache_cleared must be true")
    if state.get("calls_train_muzero") is not False:
        raise ValueError("policy refresh search worker must not call train_muzero")
    if state.get("touches_live_runs") is not False:
        raise ValueError("policy refresh search worker must not touch live runs")


def validate_compact_policy_refresh_handoff_v1(lineage: Any) -> None:
    """Validate a compact policy-refresh handoff proof."""

    if not isinstance(lineage, Mapping):
        raise ValueError("policy refresh handoff must be a mapping")
    if (
        lineage.get("compact_policy_refresh_handoff_schema_id")
        != COMPACT_POLICY_REFRESH_HANDOFF_SCHEMA_ID
    ):
        raise ValueError("policy refresh handoff schema mismatch")
    if lineage.get("policy_refresh_handoff_status") != (
        "compact_policy_refresh_handoff_v1"
    ):
        raise ValueError("policy refresh handoff status mismatch")
    for key in (
        "checkpoint_id",
        "trainer_id",
        "policy_version_ref",
        "model_version_ref",
        "policy_source",
        "learner_model_state_digest",
        "search_worker_model_state_digest",
    ):
        if not str(lineage.get(key, "")).strip():
            raise ValueError(f"policy refresh handoff missing {key}")
    if lineage.get("learner_model_state_digest") != lineage.get(
        "search_worker_model_state_digest"
    ):
        raise ValueError("policy refresh handoff digest mismatch")
    for key in ("learner_update_count", "search_worker_refresh_count"):
        if int(lineage.get(key, 0)) <= 0:
            raise ValueError(f"policy refresh handoff requires {key} > 0")
    if int(lineage.get("search_worker_model_object_id", 0)) <= 0:
        raise ValueError("policy refresh handoff missing search worker object id")
    if int(lineage.get("learner_model_object_id", 0)) <= 0:
        raise ValueError("policy refresh handoff missing learner model object id")
    if lineage.get("search_worker_distinct_from_learner") is not True:
        raise ValueError("policy refresh handoff requires distinct search worker")
    if int(lineage["search_worker_model_object_id"]) == int(
        lineage["learner_model_object_id"]
    ):
        raise ValueError("policy refresh handoff search worker is learner model")
    for key in (
        "search_worker_cache_cleared",
        "root_rows_stamped",
        "action_rows_stamped",
        "replay_rows_stamped",
        "sample_rows_stamped",
    ):
        if lineage.get(key) is not True:
            raise ValueError(f"policy refresh handoff requires {key}")
    for key in (
        "calls_train_muzero",
        "touches_live_runs",
        "promotion_claim",
        "training_speed_claim",
    ):
        if lineage.get(key) is not False:
            raise ValueError(f"policy refresh handoff {key} must be false")
    evidence_refs = lineage.get("evidence_refs")
    if not isinstance(evidence_refs, list) or not any(
        str(ref).strip() for ref in evidence_refs
    ):
        raise ValueError("policy refresh handoff evidence_refs must be non-empty")


def compact_policy_refresh_handoff_evidence_ref(
    lineage: Mapping[str, Any],
) -> str:
    """Return the compact Coach evidence ref for a validated policy handoff."""

    validate_compact_policy_refresh_handoff_v1(lineage)
    return (
        "compact_policy_refresh_handoff:"
        f"{lineage['checkpoint_id']}:"
        f"updates={lineage['learner_update_count']}:"
        f"digest={str(lineage['learner_model_state_digest'])[:16]}"
    )


def _validate_policy_refresh_row_metadata(
    metadata: Any,
    *,
    label: str,
    expected_policy_version_ref: str,
    expected_model_version_ref: str,
    expected_policy_source: str,
    expected_model_state_digest: str,
    expected_learner_update_count: int,
) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"policy refresh {label} metadata must be a mapping")
    checks = {
        "policy_version_ref": expected_policy_version_ref,
        "model_version_ref": expected_model_version_ref,
        "policy_source": expected_policy_source,
        "compact_policy_refresh_model_state_digest": expected_model_state_digest,
    }
    for key, expected in checks.items():
        actual = str(metadata.get(key, "")).strip()
        if actual != str(expected):
            raise ValueError(f"policy refresh {label} metadata {key} mismatch")
    if metadata.get("compact_policy_refresh_search_worker_refreshed") is not True:
        raise ValueError(f"policy refresh {label} metadata missing refreshed marker")
    if int(metadata.get("compact_policy_refresh_learner_update_count", 0)) != int(
        expected_learner_update_count
    ):
        raise ValueError(f"policy refresh {label} metadata update count mismatch")
    if int(metadata.get("compact_policy_refresh_count", 0)) <= 0:
        raise ValueError(f"policy refresh {label} metadata refresh count missing")


def _state_mapping(model_or_state: Any) -> dict[str, Any]:
    if isinstance(model_or_state, Mapping):
        return {str(key): value for key, value in model_or_state.items()}
    state_dict = getattr(model_or_state, "state_dict", None)
    if callable(state_dict):
        return {str(key): value for key, value in state_dict().items()}
    parameters = getattr(model_or_state, "parameters", None)
    if callable(parameters):
        return {f"parameter_{index}": value for index, value in enumerate(parameters())}
    raise ValueError("model state digest requires state_dict or parameters")


def _update_digest_with_value(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"<none>")
        return
    if hasattr(value, "detach"):
        tensor = value.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()
        if hasattr(tensor, "contiguous"):
            tensor = tensor.contiguous()
        array = tensor.numpy()
    else:
        array = np.asarray(value)
    array = np.ascontiguousarray(array)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(int(dim) for dim in array.shape)).encode("utf-8"))
    if array.dtype == object:
        digest.update(repr(array.tolist()).encode("utf-8"))
        return
    if not _array_values_are_finite(array):
        raise ValueError("model state digest requires finite numeric values")
    digest.update(array.view(np.uint8))


def _array_values_are_finite(array: np.ndarray) -> bool:
    dtype = array.dtype
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        return True
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating):
        return bool(np.isfinite(array).all())
    try:
        return bool(np.isfinite(array).all())
    except TypeError as exc:
        raise ValueError("model state digest requires numeric values") from exc


__all__ = [
    "COMPACT_POLICY_REFRESH_HANDOFF_SCHEMA_ID",
    "COMPACT_POLICY_REFRESH_METADATA_KEYS",
    "COMPACT_POLICY_REFRESH_SEARCH_WORKER_STATE_SCHEMA_ID",
    "build_compact_policy_refresh_handoff_v1",
    "compact_model_state_digest_v1",
    "compact_policy_refresh_handoff_evidence_ref",
    "compact_policy_refresh_metadata_from_state_v1",
    "compact_policy_refresh_metadata_subset_v1",
    "validate_compact_policy_refresh_handoff_v1",
    "validate_compact_policy_refresh_search_worker_state_v1",
]
