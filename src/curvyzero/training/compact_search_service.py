"""Shared compact search-service contract for profile-only optimizer lanes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
import hashlib
from typing import Any, Protocol
from typing import runtime_checkable

import numpy as np

from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1
from curvyzero.training.compact_policy_row_bridge import validate_compact_search_result_v1
from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID = "curvyzero_compact_search_action_step/v1"
COMPACT_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID = "curvyzero_compact_search_replay_payload/v1"
COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID = (
    "curvyzero_compact_device_search_replay_payload/v1"
)


@runtime_checkable
class CompactSearchServiceV1(Protocol):
    """Minimal boundary between compact roots and compact search results."""

    search_impl: str
    num_simulations: int

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        """Search active roots and return a validated compact result."""


@runtime_checkable
class CompactSearchTwoPhaseServiceV1(Protocol):
    """Search service that splits env-action sync from delayed replay payloads."""

    search_impl: str
    num_simulations: int

    def run_action_step(
        self,
        root_batch: CompactRootBatchV1,
    ) -> CompactSearchActionStepV1:
        """Return only the action-critical payload for the env hot step."""

    def flush_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactSearchReplayPayloadV1:
        """Return the delayed replay-critical payload for a prior action step."""


@dataclass(frozen=True, slots=True)
class CompactSearchActionStepV1:
    """Action-critical half of a two-phase compact search result."""

    replay_payload_handle: str
    root_index: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    policy_env_id: np.ndarray
    selected_action: np.ndarray
    metadata: dict[str, Any]
    dense_joint_action: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class CompactSearchReplayPayloadV1:
    """Replay-critical half of a two-phase compact search result."""

    replay_payload_handle: str
    root_index: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    policy_env_id: np.ndarray
    visit_policy: np.ndarray
    root_value: np.ndarray
    raw_visit_counts: np.ndarray | None
    predicted_value: np.ndarray | None
    predicted_policy_logits: np.ndarray | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactDeviceSearchReplayPayloadV1:
    """Replay-critical two-phase payload whose large arrays stay device-owned."""

    replay_payload_handle: str
    root_index: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    policy_env_id: np.ndarray
    visit_policy: Any
    root_value: Any
    raw_visit_counts: Any | None
    predicted_value: Any | None
    predicted_policy_logits: Any | None
    metadata: dict[str, Any]


def compact_search_result_v1_from_arrays(
    root_batch: CompactRootBatchV1,
    search_arrays: Mapping[str, Any],
    *,
    default_search_impl: str,
    default_num_simulations: int,
    metadata: Mapping[str, Any] | None = None,
) -> CompactSearchResultV1:
    """Validate compact search arrays without owning or re-running the search."""

    search_impl = str(search_arrays.get("search_impl", default_search_impl))
    num_simulations = int(
        search_arrays.get("actual_search_simulations", default_num_simulations)
    )
    return validate_compact_search_result_v1(
        root_batch,
        selected_action=search_arrays["selected_action"],
        visit_policy=search_arrays["visit_policy"],
        root_value=search_arrays["root_value"],
        search_impl=search_impl,
        num_simulations=num_simulations,
        raw_visit_counts=search_arrays.get("raw_visit_counts"),
        predicted_value=search_arrays.get("predicted_value"),
        predicted_policy_logits=search_arrays.get("predicted_policy_logits"),
        metadata=metadata,
    )


def compact_search_result_v1_from_two_phase_payloads(
    root_batch: CompactRootBatchV1,
    action_step: CompactSearchActionStepV1,
    replay_payload: CompactSearchReplayPayloadV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CompactSearchResultV1:
    """Reassemble a full compact result only when replay rows need it."""

    validate_compact_search_two_phase_payload_v1(action_step, replay_payload)
    search_impl = str(
        replay_payload.metadata.get("search_impl")
        or action_step.metadata.get("search_impl")
        or "two_phase_compact_search"
    )
    simulations = int(
        replay_payload.metadata.get(
            "num_simulations",
            action_step.metadata.get("num_simulations", 0),
        )
    )
    result_metadata = {
        key: value
        for key, value in replay_payload.metadata.items()
        if key not in {"schema_id", "phase"}
    }
    result_metadata.update(
        {
            "two_phase_compact_search": True,
            "action_step_schema_id": action_step.metadata.get("schema_id"),
            "replay_payload_schema_id": replay_payload.metadata.get("schema_id"),
            "replay_payload_handle": replay_payload.replay_payload_handle,
        }
    )
    if metadata:
        result_metadata.update({str(key): value for key, value in metadata.items()})
    return validate_compact_search_result_v1(
        root_batch,
        selected_action=action_step.selected_action,
        visit_policy=replay_payload.visit_policy,
        root_value=replay_payload.root_value,
        raw_visit_counts=replay_payload.raw_visit_counts,
        predicted_value=replay_payload.predicted_value,
        predicted_policy_logits=replay_payload.predicted_policy_logits,
        search_impl=search_impl,
        num_simulations=simulations,
        metadata=result_metadata,
    )


def compact_search_action_step_v1_from_result(
    search_result: CompactSearchResultV1,
    *,
    replay_payload_handle: str,
    metadata: Mapping[str, Any] | None = None,
) -> CompactSearchActionStepV1:
    """Split a compact search result into the env-action critical payload."""

    handle = str(replay_payload_handle)
    if not handle:
        raise ReplayCompatibilityError("replay_payload_handle must be a non-empty string")
    step_metadata = {}
    if metadata:
        step_metadata.update({str(key): value for key, value in metadata.items()})
    step_metadata.update(
        {
            "schema_id": COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID,
            "phase": "action_critical",
            "search_result_schema_id": search_result.metadata.get("schema_id"),
            "search_impl": search_result.metadata.get("search_impl"),
            "num_simulations": search_result.metadata.get("num_simulations"),
            "active_root_count": int(search_result.selected_action.shape[0]),
            "selected_action_digest": _array_digest(search_result.selected_action),
            "search_replay_payload_digest": _search_result_replay_payload_digest(
                search_result
            ),
        }
    )
    return CompactSearchActionStepV1(
        replay_payload_handle=handle,
        root_index=search_result.root_index.astype(np.int32, copy=True),
        env_row=search_result.env_row.astype(np.int32, copy=True),
        player=search_result.player.astype(np.int16, copy=True),
        policy_env_id=search_result.policy_env_id.astype(np.int64, copy=True),
        selected_action=search_result.selected_action.astype(np.int16, copy=True),
        metadata=step_metadata,
    )


def compact_search_replay_payload_v1_from_result(
    search_result: CompactSearchResultV1,
    *,
    replay_payload_handle: str,
    metadata: Mapping[str, Any] | None = None,
) -> CompactSearchReplayPayloadV1:
    """Split a compact search result into the replay-critical payload."""

    handle = str(replay_payload_handle)
    if not handle:
        raise ReplayCompatibilityError("replay_payload_handle must be a non-empty string")
    payload_metadata = {}
    if metadata:
        payload_metadata.update({str(key): value for key, value in metadata.items()})
    payload_metadata.update(
        {
            "schema_id": COMPACT_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
            "phase": "replay_critical",
            "search_result_schema_id": search_result.metadata.get("schema_id"),
            "search_impl": search_result.metadata.get("search_impl"),
            "num_simulations": search_result.metadata.get("num_simulations"),
            "active_root_count": int(search_result.selected_action.shape[0]),
            "search_replay_payload_digest": _search_result_replay_payload_digest(
                search_result
            ),
        }
    )
    return CompactSearchReplayPayloadV1(
        replay_payload_handle=handle,
        root_index=search_result.root_index.astype(np.int32, copy=True),
        env_row=search_result.env_row.astype(np.int32, copy=True),
        player=search_result.player.astype(np.int16, copy=True),
        policy_env_id=search_result.policy_env_id.astype(np.int64, copy=True),
        visit_policy=search_result.visit_policy.astype(np.float32, copy=True),
        root_value=search_result.root_value.astype(np.float32, copy=True),
        raw_visit_counts=None
        if search_result.raw_visit_counts is None
        else search_result.raw_visit_counts.astype(np.float32, copy=True),
        predicted_value=None
        if search_result.predicted_value is None
        else search_result.predicted_value.astype(np.float32, copy=True),
        predicted_policy_logits=None
        if search_result.predicted_policy_logits is None
        else search_result.predicted_policy_logits.astype(np.float32, copy=True),
        metadata=payload_metadata,
    )


def validate_compact_search_two_phase_payload_v1(
    action_step: CompactSearchActionStepV1,
    replay_payload: CompactSearchReplayPayloadV1,
) -> None:
    """Fail closed if delayed replay payload identity no longer matches actions."""

    if action_step.metadata.get("schema_id") != COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID:
        raise ReplayCompatibilityError("action step schema id is invalid")
    if action_step.metadata.get("phase") != "action_critical":
        raise ReplayCompatibilityError("action step phase is invalid")
    if replay_payload.metadata.get("schema_id") != COMPACT_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID:
        raise ReplayCompatibilityError("replay payload schema id is invalid")
    if replay_payload.metadata.get("phase") != "replay_critical":
        raise ReplayCompatibilityError("replay payload phase is invalid")
    action_search_impl = action_step.metadata.get("search_impl")
    payload_search_impl = replay_payload.metadata.get("search_impl")
    if action_search_impl != payload_search_impl:
        raise ReplayCompatibilityError("replay payload search_impl does not match action step")
    action_origin = action_step.metadata.get("replay_payload_origin")
    if action_origin is not None and replay_payload.metadata.get(
        "replay_payload_origin"
    ) != action_origin:
        raise ReplayCompatibilityError("replay payload origin does not match action step")
    action_simulations = int(action_step.metadata.get("num_simulations", -1))
    payload_simulations = int(replay_payload.metadata.get("num_simulations", -2))
    if action_simulations != payload_simulations:
        raise ReplayCompatibilityError(
            "replay payload num_simulations does not match action step"
        )
    expected_count = int(action_step.selected_action.shape[0])
    payload_count = int(replay_payload.metadata.get("active_root_count", -1))
    if payload_count != expected_count:
        raise ReplayCompatibilityError(
            "replay payload active_root_count does not match action step"
        )
    _validate_replay_payload_shapes(replay_payload, active_count=expected_count)
    action_selected_digest = action_step.metadata.get("selected_action_digest")
    if action_selected_digest != _array_digest(action_step.selected_action):
        raise ReplayCompatibilityError("action step selected-action digest is stale")
    if action_step.replay_payload_handle != replay_payload.replay_payload_handle:
        raise ReplayCompatibilityError("replay payload handle does not match action step")
    for field_name in ("root_index", "env_row", "player", "policy_env_id"):
        action_values = getattr(action_step, field_name)
        payload_values = getattr(replay_payload, field_name)
        if not np.array_equal(action_values, payload_values):
            raise ReplayCompatibilityError(
                f"replay payload {field_name} does not match action step"
            )
    action_digest = action_step.metadata.get("search_replay_payload_digest")
    if not isinstance(action_digest, str) or not action_digest:
        raise ReplayCompatibilityError("action step missing replay payload digest")
    replay_digest = compact_search_replay_payload_digest_v1(replay_payload)
    deferred_digest = compact_search_deferred_replay_payload_digest_v1(
        action_step.replay_payload_handle
    )
    if action_digest == deferred_digest:
        if action_step.metadata.get("search_replay_payload_digest_deferred") is not True:
            raise ReplayCompatibilityError(
                "action step deferred replay digest marker is invalid"
            )
    elif action_digest != replay_digest:
        raise ReplayCompatibilityError("replay payload digest does not match action step")
    replay_metadata_digest = replay_payload.metadata.get("search_replay_payload_digest")
    if replay_metadata_digest != replay_digest:
        raise ReplayCompatibilityError("replay payload metadata digest is stale")


def validate_compact_device_search_two_phase_payload_v1(
    action_step: CompactSearchActionStepV1,
    replay_payload: CompactDeviceSearchReplayPayloadV1,
) -> None:
    """Validate the device replay payload without materializing device tensors."""

    if action_step.metadata.get("schema_id") != COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID:
        raise ReplayCompatibilityError("action step schema id is invalid")
    if action_step.metadata.get("phase") != "action_critical":
        raise ReplayCompatibilityError("action step phase is invalid")
    if (
        replay_payload.metadata.get("schema_id")
        != COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID
    ):
        raise ReplayCompatibilityError("device replay payload schema id is invalid")
    if replay_payload.metadata.get("phase") != "replay_critical_device":
        raise ReplayCompatibilityError("device replay payload phase is invalid")
    if replay_payload.metadata.get("device_replay_payload") is not True:
        raise ReplayCompatibilityError("device replay payload marker is missing")
    if replay_payload.metadata.get("host_search_payload_fallback_allowed") is not False:
        raise ReplayCompatibilityError("device replay payload allowed host fallback")
    action_search_impl = action_step.metadata.get("search_impl")
    payload_search_impl = replay_payload.metadata.get("search_impl")
    if action_search_impl != payload_search_impl:
        raise ReplayCompatibilityError("device replay payload search_impl mismatch")
    action_origin = action_step.metadata.get("replay_payload_origin")
    if action_origin is not None and replay_payload.metadata.get(
        "replay_payload_origin"
    ) != action_origin:
        raise ReplayCompatibilityError("device replay payload origin mismatch")
    action_simulations = int(action_step.metadata.get("num_simulations", -1))
    payload_simulations = int(replay_payload.metadata.get("num_simulations", -2))
    if action_simulations != payload_simulations:
        raise ReplayCompatibilityError("device replay payload num_simulations mismatch")
    expected_count = int(action_step.selected_action.shape[0])
    payload_count = int(replay_payload.metadata.get("active_root_count", -1))
    if payload_count != expected_count:
        raise ReplayCompatibilityError("device replay payload active_root_count mismatch")
    if action_step.replay_payload_handle != replay_payload.replay_payload_handle:
        raise ReplayCompatibilityError("device replay payload handle mismatch")
    for field_name in ("root_index", "env_row", "player", "policy_env_id"):
        action_values = getattr(action_step, field_name)
        payload_values = getattr(replay_payload, field_name)
        if not np.array_equal(action_values, payload_values):
            raise ReplayCompatibilityError(
                f"device replay payload {field_name} does not match action step"
            )
    action_selected_digest = action_step.metadata.get("selected_action_digest")
    if action_selected_digest != _array_digest(action_step.selected_action):
        raise ReplayCompatibilityError("action step selected-action digest is stale")
    action_digest = action_step.metadata.get("search_replay_payload_digest")
    deferred_digest = compact_search_deferred_replay_payload_digest_v1(
        action_step.replay_payload_handle
    )
    if action_digest != deferred_digest:
        raise ReplayCompatibilityError("device replay payload must use deferred digest")
    if action_step.metadata.get("search_replay_payload_digest_deferred") is not True:
        raise ReplayCompatibilityError("action step deferred replay marker is invalid")
    _validate_device_replay_payload_shapes(replay_payload, active_count=expected_count)


def _validate_replay_payload_shapes(
    replay_payload: CompactSearchReplayPayloadV1,
    *,
    active_count: int,
) -> None:
    visit_policy = np.asarray(replay_payload.visit_policy)
    if visit_policy.shape != (active_count, ACTION_COUNT):
        raise ReplayCompatibilityError("replay payload visit_policy shape is invalid")
    root_value = np.asarray(replay_payload.root_value)
    if root_value.shape != (active_count,):
        raise ReplayCompatibilityError("replay payload root_value shape is invalid")
    if replay_payload.raw_visit_counts is not None:
        raw_visit_counts = np.asarray(replay_payload.raw_visit_counts)
        if raw_visit_counts.shape != (active_count, ACTION_COUNT):
            raise ReplayCompatibilityError(
                "replay payload raw_visit_counts shape is invalid"
            )
    if replay_payload.predicted_value is not None:
        predicted_value = np.asarray(replay_payload.predicted_value)
        if predicted_value.shape != (active_count,):
            raise ReplayCompatibilityError(
                "replay payload predicted_value shape is invalid"
            )
    if replay_payload.predicted_policy_logits is not None:
        logits = np.asarray(replay_payload.predicted_policy_logits)
        if logits.shape != (active_count, ACTION_COUNT):
            raise ReplayCompatibilityError(
                "replay payload predicted_policy_logits shape is invalid"
            )


def _validate_device_replay_payload_shapes(
    replay_payload: CompactDeviceSearchReplayPayloadV1,
    *,
    active_count: int,
) -> None:
    if _value_shape(replay_payload.visit_policy) != (active_count, ACTION_COUNT):
        raise ReplayCompatibilityError("device replay payload visit_policy shape is invalid")
    if _value_shape(replay_payload.root_value) != (active_count,):
        raise ReplayCompatibilityError("device replay payload root_value shape is invalid")
    if replay_payload.raw_visit_counts is not None and _value_shape(
        replay_payload.raw_visit_counts
    ) != (active_count, ACTION_COUNT):
        raise ReplayCompatibilityError(
            "device replay payload raw_visit_counts shape is invalid"
        )
    if replay_payload.predicted_value is not None and _value_shape(
        replay_payload.predicted_value
    ) != (active_count,):
        raise ReplayCompatibilityError(
            "device replay payload predicted_value shape is invalid"
        )
    if replay_payload.predicted_policy_logits is not None and _value_shape(
        replay_payload.predicted_policy_logits
    ) != (active_count, ACTION_COUNT):
        raise ReplayCompatibilityError(
            "device replay payload predicted_policy_logits shape is invalid"
        )


def _value_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(value, "shape", ()))


def _search_result_replay_payload_digest(search_result: CompactSearchResultV1) -> str:
    digest = hashlib.sha256()
    for field_name in (
        "root_index",
        "env_row",
        "player",
        "policy_env_id",
        "visit_policy",
        "root_value",
        "raw_visit_counts",
        "predicted_value",
        "predicted_policy_logits",
    ):
        _update_digest_with_array(digest, getattr(search_result, field_name))
    return digest.hexdigest()


def compact_search_replay_payload_digest_v1(
    replay_payload: CompactSearchReplayPayloadV1,
) -> str:
    """Return the stable digest for a replay-critical compact payload."""

    digest = hashlib.sha256()
    for field_name in (
        "root_index",
        "env_row",
        "player",
        "policy_env_id",
        "visit_policy",
        "root_value",
        "raw_visit_counts",
        "predicted_value",
        "predicted_policy_logits",
    ):
        _update_digest_with_array(digest, getattr(replay_payload, field_name))
    return digest.hexdigest()


def compact_search_deferred_replay_payload_digest_v1(
    replay_payload_handle: str,
) -> str:
    """Return the action-step marker used when replay arrays stay deferred."""

    return f"deferred:{str(replay_payload_handle)}"


def _update_digest_with_array(digest: Any, value: np.ndarray | None) -> None:
    if value is None:
        digest.update(b"<none>")
        return
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.view(np.uint8))


def _array_digest(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    _update_digest_with_array(digest, value)
    return digest.hexdigest()


def compact_search_array_digest_v1(value: np.ndarray) -> str:
    """Return the stable digest for a compact action-side array."""

    return _array_digest(value)


class CompactSearchPayloadGateV1:
    """Profile-only visibility gate for delayed compact search payloads."""

    def __init__(self) -> None:
        self._action_steps: dict[str, CompactSearchActionStepV1] = {}
        self._replay_payloads: dict[str, CompactSearchReplayPayloadV1] = {}

    @property
    def pending_count(self) -> int:
        """Number of action steps whose replay payload is not yet visible."""

        return len(self._action_steps) - len(self._replay_payloads)

    @property
    def complete_count(self) -> int:
        """Number of action steps with checked replay payloads attached."""

        return len(self._replay_payloads)

    def register_action_step(self, action_step: CompactSearchActionStepV1) -> None:
        """Register selected actions before the replay payload is available."""

        handle = action_step.replay_payload_handle
        if handle in self._action_steps:
            raise ReplayCompatibilityError("duplicate replay payload handle")
        self._action_steps[handle] = action_step

    def attach_replay_payload(self, replay_payload: CompactSearchReplayPayloadV1) -> None:
        """Attach a delayed payload and make it sample-visible if identities match."""

        handle = replay_payload.replay_payload_handle
        action_step = self._action_steps.get(handle)
        if action_step is None:
            raise ReplayCompatibilityError("replay payload arrived before action step")
        if handle in self._replay_payloads:
            raise ReplayCompatibilityError("duplicate replay payload attachment")
        validate_compact_search_two_phase_payload_v1(action_step, replay_payload)
        self._replay_payloads[handle] = replay_payload

    def is_sample_visible(self, replay_payload_handle: str) -> bool:
        """Return whether the handle has a complete, checked replay payload."""

        return str(replay_payload_handle) in self._replay_payloads

    def require_replay_payload(self, replay_payload_handle: str) -> CompactSearchReplayPayloadV1:
        """Return a checked payload or fail before a sampler can see the row."""

        handle = str(replay_payload_handle)
        payload = self._replay_payloads.get(handle)
        if payload is None:
            raise ReplayCompatibilityError("replay payload is not sample-visible")
        return payload


class CompactSearchComparatorServiceV1:
    """Run two compact search services on the same roots and return the primary result."""

    profile_only = True
    calls_train_muzero = False
    trainer_defaults_changed = False
    touches_live_runs = False

    def __init__(
        self,
        *,
        primary: CompactSearchServiceV1,
        reference: CompactSearchServiceV1,
        comparison_label: str,
        fail_on_identity_mismatch: bool = True,
    ) -> None:
        self.primary = primary
        self.reference = reference
        self.search_impl = str(primary.search_impl)
        self.num_simulations = int(primary.num_simulations)
        self.comparison_label = str(comparison_label)
        if not self.comparison_label:
            raise ReplayCompatibilityError("comparison_label must be non-empty")
        self.fail_on_identity_mismatch = bool(fail_on_identity_mismatch)

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        """Return primary search output with reference-comparison telemetry attached."""

        primary_result = self.primary.run(root_batch)
        reference_result = self.reference.run(root_batch)
        comparison = compact_search_comparison_telemetry_v1(
            primary_result,
            reference_result,
            comparison_label=self.comparison_label,
            fail_on_identity_mismatch=self.fail_on_identity_mismatch,
        )
        metadata = dict(primary_result.metadata)
        profile_telemetry = metadata.get("profile_telemetry", {})
        if not isinstance(profile_telemetry, Mapping):
            profile_telemetry = {}
        profile_telemetry = dict(profile_telemetry)
        profile_telemetry.update(comparison)
        metadata.update(
            {
                "compact_search_comparator_enabled": True,
                "compact_search_comparator_label": self.comparison_label,
                "compact_search_comparator_primary_impl": comparison[
                    "compact_search_comparator_primary_impl"
                ],
                "compact_search_comparator_reference_impl": comparison[
                    "compact_search_comparator_reference_impl"
                ],
                "profile_telemetry": profile_telemetry,
            }
        )
        return replace(primary_result, metadata=metadata)


def compact_search_comparison_telemetry_v1(
    primary: CompactSearchResultV1,
    reference: CompactSearchResultV1,
    *,
    comparison_label: str,
    fail_on_identity_mismatch: bool = True,
) -> dict[str, Any]:
    """Compare two compact search outputs that were run on the same roots."""

    identity_match = True
    mismatched_fields: list[str] = []
    for field_name in ("root_index", "env_row", "player", "policy_env_id"):
        if not np.array_equal(getattr(primary, field_name), getattr(reference, field_name)):
            identity_match = False
            mismatched_fields.append(field_name)
    if not identity_match and fail_on_identity_mismatch:
        raise ReplayCompatibilityError(
            "compact search comparator identity mismatch: "
            + ", ".join(mismatched_fields)
        )

    primary_actions = np.asarray(primary.selected_action, dtype=np.int16).reshape(-1)
    reference_actions = np.asarray(reference.selected_action, dtype=np.int16).reshape(-1)
    if primary_actions.shape != reference_actions.shape:
        raise ReplayCompatibilityError("compact search comparator action shape mismatch")
    active_count = int(primary_actions.size)
    action_match_count = int(np.count_nonzero(primary_actions == reference_actions))
    action_match_fraction = (
        1.0 if active_count == 0 else float(action_match_count / active_count)
    )

    primary_visit = np.asarray(primary.visit_policy, dtype=np.float32)
    reference_visit = np.asarray(reference.visit_policy, dtype=np.float32)
    if primary_visit.shape != reference_visit.shape:
        raise ReplayCompatibilityError("compact search comparator visit shape mismatch")
    visit_l1 = np.abs(primary_visit - reference_visit).sum(axis=1)

    primary_value = np.asarray(primary.root_value, dtype=np.float32).reshape(-1)
    reference_value = np.asarray(reference.root_value, dtype=np.float32).reshape(-1)
    if primary_value.shape != reference_value.shape:
        raise ReplayCompatibilityError("compact search comparator value shape mismatch")
    value_abs = np.abs(primary_value - reference_value)

    telemetry: dict[str, Any] = {
        "compact_search_comparator_enabled": True,
        "compact_search_comparator_label": str(comparison_label),
        "compact_search_comparator_identity_match": bool(identity_match),
        "compact_search_comparator_identity_mismatched_fields": list(mismatched_fields),
        "compact_search_comparator_primary_impl": str(
            primary.metadata.get("search_impl", "")
        ),
        "compact_search_comparator_reference_impl": str(
            reference.metadata.get("search_impl", "")
        ),
        "compact_search_comparator_active_root_count": float(active_count),
        "compact_search_comparator_action_match_count": float(action_match_count),
        "compact_search_comparator_action_match_fraction": float(action_match_fraction),
        "compact_search_comparator_primary_action_checksum": float(
            _action_checksum(primary_actions)
        ),
        "compact_search_comparator_reference_action_checksum": float(
            _action_checksum(reference_actions)
        ),
        "compact_search_comparator_visit_l1_mean": _safe_mean(visit_l1),
        "compact_search_comparator_visit_l1_max": _safe_max(visit_l1),
        "compact_search_comparator_root_value_abs_diff_mean": _safe_mean(value_abs),
        "compact_search_comparator_root_value_abs_diff_max": _safe_max(value_abs),
    }
    predicted_value_metrics = _optional_abs_diff_metrics(
        primary.predicted_value,
        reference.predicted_value,
        prefix="compact_search_comparator_predicted_value_abs_diff",
    )
    telemetry.update(predicted_value_metrics)
    predicted_policy_logits_metrics = _optional_abs_diff_metrics(
        primary.predicted_policy_logits,
        reference.predicted_policy_logits,
        prefix="compact_search_comparator_predicted_policy_logits_abs_diff",
    )
    telemetry.update(predicted_policy_logits_metrics)
    raw_visit_metrics = _optional_abs_diff_metrics(
        primary.raw_visit_counts,
        reference.raw_visit_counts,
        prefix="compact_search_comparator_raw_visit_counts_abs_diff",
    )
    telemetry.update(raw_visit_metrics)
    return telemetry


def _action_checksum(actions: np.ndarray) -> int:
    if actions.size == 0:
        return 0
    weights = np.arange(1, actions.size + 1, dtype=np.int64)
    return int((actions.astype(np.int64, copy=False) + 1).dot(weights))


def _safe_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(array.mean())


def _safe_max(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(array.max())


def _optional_abs_diff_metrics(
    primary: np.ndarray | None,
    reference: np.ndarray | None,
    *,
    prefix: str,
) -> dict[str, Any]:
    present = primary is not None and reference is not None
    if not present:
        return {
            f"{prefix}_present": False,
            f"{prefix}_mean": 0.0,
            f"{prefix}_max": 0.0,
        }
    primary_array = np.asarray(primary, dtype=np.float32).reshape(-1)
    reference_array = np.asarray(reference, dtype=np.float32).reshape(-1)
    if primary_array.shape != reference_array.shape:
        raise ReplayCompatibilityError(
            f"compact search comparator optional field shape mismatch for {prefix}"
        )
    diff = np.abs(primary_array - reference_array)
    return {
        f"{prefix}_present": True,
        f"{prefix}_mean": _safe_mean(diff),
        f"{prefix}_max": _safe_max(diff),
    }


__all__ = [
    "COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID",
    "COMPACT_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID",
    "CompactSearchActionStepV1",
    "CompactSearchComparatorServiceV1",
    "CompactSearchPayloadGateV1",
    "CompactSearchReplayPayloadV1",
    "CompactSearchServiceV1",
    "CompactSearchTwoPhaseServiceV1",
    "compact_search_action_step_v1_from_result",
    "compact_search_comparison_telemetry_v1",
    "compact_search_replay_payload_v1_from_result",
    "compact_search_result_v1_from_arrays",
    "compact_search_result_v1_from_two_phase_payloads",
    "validate_compact_search_two_phase_payload_v1",
]
