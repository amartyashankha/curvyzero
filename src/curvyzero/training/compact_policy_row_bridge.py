"""Profile-only bridge from compact search arrays to target-row records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT
from curvyzero.training.multiplayer_source_state_target_rows import DEFAULT_TO_PLAY
from curvyzero.training.multiplayer_source_state_target_rows import PolicyRowRecordV0
from curvyzero.training.multiplayer_source_state_target_rows import (
    SourceStateMultiplayerSampleBatchV0,
)
from curvyzero.training.multiplayer_source_state_target_rows import (
    SourceStateMultiplayerTargetRowsV0,
)
from curvyzero.training.multiplayer_source_state_target_rows import _metadata
from curvyzero.training.multiplayer_source_state_target_rows import _target_rows_from_dicts
from curvyzero.training.multiplayer_source_state_target_rows import _validate_chunk
from curvyzero.training.multiplayer_source_state_target_rows import _validate_policy_row_arrays
from curvyzero.training.multiplayer_source_state_target_rows import (
    build_source_state_multiplayer_sample_batch_v0,
)
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
)
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_observation_contract import (
    RESIDENT_OBSERVATION_BATCH_SCHEMA_ID,
)
from curvyzero.training.compact_observation_contract import RESIDENT_OBSERVATION_OWNER
from curvyzero.training.compact_observation_contract import ResidentObservationBatchV1
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_policy_refresh_metadata_subset_v1,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.source_state_hybrid_observation_profile import HybridCompactBatch


COMPACT_POLICY_ROW_RECORDS_CONTRACT_ID = "curvyzero_compact_policy_row_records_from_search/v0"
COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID = "curvyzero_compact_search_replay_service/v1"
COMPACT_REPLAY_ENV_PLAYER_IDENTITY_DIGEST_KEY = "env_player_identity_digest"
# OPT-106 proved this metadata producer correct but slower on H100; keep it off
# unless a future larger replay ownership change explicitly reselects it.
COMPACT_REPLAY_TERMINAL_METADATA_ROW_INDICES_ENABLED = False


@dataclass(frozen=True, slots=True)
class CompactPolicySearchArraysV0:
    """Checked active-root search arrays without per-root Python records."""

    record_index: int
    policy_row: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    action: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    root_value: np.ndarray
    policy_source: str


@dataclass(frozen=True, slots=True)
class CompactRootBatchV1:
    """All compact row/player roots before a batched search service filters active roots."""

    observation: np.ndarray
    legal_mask: np.ndarray
    active_root_mask: np.ndarray
    to_play: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    policy_env_id: np.ndarray
    target_reward: np.ndarray
    done_root: np.ndarray
    final_observation: np.ndarray | None
    final_observation_row_mask: np.ndarray
    terminal_row_mask: np.ndarray
    autoreset_row_mask: np.ndarray
    metadata: dict[str, Any]
    resident_observation: ResidentObservationBatchV1 | None = None
    observation_source: str = COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
    terminated: np.ndarray | None = None
    truncated: np.ndarray | None = None
    final_reward_map: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class CompactRootBuildRequestV1:
    """Lightweight root request that lets an owner build the root batch."""

    schema_id: str
    request_kind: str
    search_lane: str
    metadata: dict[str, Any]
    copy_observation: bool
    observation_source: str
    resident_host_observation_stub: bool
    batch_size: int
    player_count: int
    root_count: int
    stack_shape: tuple[int, ...]
    observation: Any | None
    action_mask: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    policy_env_id: np.ndarray
    policy_env_row: np.ndarray
    policy_player: np.ndarray
    target_reward: np.ndarray
    done_root: np.ndarray
    to_play: np.ndarray
    active_root_mask: np.ndarray
    final_observation: Any | None
    final_observation_row_mask: np.ndarray
    terminal_row_mask: np.ndarray
    autoreset_row_mask: np.ndarray
    joint_action: np.ndarray | None = None
    resident_observation: ResidentObservationBatchV1 | None = None
    terminated: np.ndarray | None = None
    truncated: np.ndarray | None = None
    final_reward_map: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class CompactRootActionContextV1:
    """Action-critical root sidecars kept by the parent after owner submit."""

    schema_id: str
    request_schema_id: str
    request_kind: str
    search_lane: str
    metadata: dict[str, Any]
    copy_observation: bool
    resident_host_observation_stub: bool
    observation_dtype: str
    batch_size: int
    player_count: int
    root_count: int
    stack_shape: tuple[int, ...]
    active_root_index: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    policy_env_id: np.ndarray
    active_legal_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class CompactSearchResultV1:
    """Checked active-root search result from a compact root batch."""

    root_index: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    policy_env_id: np.ndarray
    selected_action: np.ndarray
    visit_policy: np.ndarray
    root_value: np.ndarray
    raw_visit_counts: np.ndarray | None
    predicted_value: np.ndarray | None
    predicted_policy_logits: np.ndarray | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactReplayChunkV1:
    """Profile-only compact service chunk with a target-row validation edge."""

    record_index: int
    next_record_index: int
    root_batch: CompactRootBatchV1
    search_result: CompactSearchResultV1
    target_rows: SourceStateMultiplayerTargetRowsV0
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactReplayIndexRowsV1:
    """Index-only compact replay rows for the collection hot path.

    These rows deliberately do not copy ``observation`` or ``next_observation``.
    They are the compact replay-writer shape: keep the record indices and search
    arrays now, materialize learner tensors only when a sampler asks for them.
    """

    metadata: dict[str, Any]
    record_index: int
    next_record_index: int
    compact_root_row: np.ndarray
    policy_env_id: np.ndarray
    policy_row: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    action: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    root_value: np.ndarray
    reward: np.ndarray
    final_reward: np.ndarray
    done: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_final_observation_row: np.ndarray
    to_play: np.ndarray
    policy_source: str


@dataclass(frozen=True, slots=True)
class CompactDeviceReplayIndexRowsV1:
    """Compact replay rows whose learner targets are already device tensors."""

    metadata: dict[str, Any]
    record_index: int
    next_record_index: int
    compact_root_row: Any
    policy_env_id: Any
    policy_row: Any
    env_row: Any
    player: Any
    action: Any
    action_mask: Any
    policy_target: Any
    root_value: Any
    reward: Any
    final_reward: Any
    done: Any
    terminated: Any
    truncated: Any
    next_final_observation_row: Any
    to_play: Any
    policy_source: str


COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID = "curvyzero_compact_root_build_request/v1"
COMPACT_ROOT_BUILD_REQUEST_KIND_RESIDENT_ROOT_VIEW = "resident_root_view_build_request_v1"
COMPACT_ROOT_BUILD_REQUEST_KIND_HOST_ARRAY = "host_array_build_request_v1"
COMPACT_ROOT_MECHANICS_OUTCOME_SCHEMA_ID = "curvyzero_compact_root_mechanics_outcome/v1"


@dataclass(frozen=True, slots=True)
class CompactRootTransitionOutcomeV1:
    """Replay transition outcome derived from an owner-visible next root."""

    next_reward: np.ndarray
    next_done: np.ndarray
    next_terminated: np.ndarray
    next_truncated: np.ndarray
    next_final_reward_map: np.ndarray
    next_final_observation_row_mask: np.ndarray


def compact_root_build_request_v1_from_batch(
    batch: HybridCompactBatch,
    *,
    search_lane: str,
    metadata: Mapping[str, Any] | None = None,
    copy_observation: bool = True,
    observation_source: str | None = None,
    resident_observation: ResidentObservationBatchV1 | None = None,
    resident_host_observation_stub: bool = False,
) -> CompactRootBuildRequestV1:
    """Capture root-build inputs without constructing ``CompactRootBatchV1``."""

    search_lane_value = str(search_lane)
    if not search_lane_value:
        raise ReplayCompatibilityError("search_lane must be a non-empty string")
    if observation_source is None:
        observation_source = getattr(
            batch,
            "observation_source",
            COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
        )
    if resident_observation is None:
        resident_observation = getattr(batch, "resident_observation", None)
    observation_source_value = str(observation_source)
    if observation_source_value not in {
        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    }:
        raise ReplayCompatibilityError(f"unknown compact observation_source {observation_source!r}")

    observation: Any | None
    if observation_source_value == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1:
        if resident_observation is None:
            raise ReplayCompatibilityError(
                "resident_device_v1 root build request requires resident_observation"
            )
        batch_size = int(resident_observation.batch_size)
        player_count = int(resident_observation.player_count)
        stack_shape = tuple(int(dim) for dim in resident_observation.stack_shape)
        if bool(resident_host_observation_stub):
            observation = None
        else:
            observation = getattr(batch, "observation", None)
            if observation is None:
                raise ReplayCompatibilityError(
                    "resident root build request without host stub requires observation"
                )
            observation_array = np.asarray(observation)
            if observation_array.ndim != 5:
                raise ReplayCompatibilityError("observation must have shape [B,P,C,H,W]")
            if observation_array.shape[:2] != (batch_size, player_count):
                raise ReplayCompatibilityError(
                    "resident root build request observation shape mismatch"
                )
            if tuple(int(dim) for dim in observation_array.shape[2:]) != stack_shape:
                raise ReplayCompatibilityError("resident root build request stack shape mismatch")
    else:
        observation = getattr(batch, "observation", None)
        if observation is None:
            raise ReplayCompatibilityError("host root build request requires observation")
        observation_array = np.asarray(observation)
        if observation_array.ndim != 5:
            raise ReplayCompatibilityError("observation must have shape [B,P,C,H,W]")
        batch_size, player_count = (
            int(observation_array.shape[0]),
            int(observation_array.shape[1]),
        )
        stack_shape = tuple(int(dim) for dim in observation_array.shape[2:])
        if resident_observation is not None:
            raise ReplayCompatibilityError(
                "resident_observation was provided but observation_source is host_array_v1"
            )
    if batch_size <= 0 or player_count <= 0:
        raise ReplayCompatibilityError("observation batch/player dimensions must be positive")
    root_count = int(batch_size) * int(player_count)
    joint_action = None
    if hasattr(batch, "joint_action"):
        joint_action = np.asarray(getattr(batch, "joint_action"))
        if joint_action.shape != (int(batch_size), int(player_count)):
            raise ReplayCompatibilityError("root build request joint_action shape mismatch")
    request_metadata = {
        "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
        "schema_id": COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID,
        "root_build_request_schema_id": COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID,
        "root_build_request_kind": (
            COMPACT_ROOT_BUILD_REQUEST_KIND_RESIDENT_ROOT_VIEW
            if observation_source_value == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
            else COMPACT_ROOT_BUILD_REQUEST_KIND_HOST_ARRAY
        ),
        "search_lane": search_lane_value,
        "observation_source": observation_source_value,
        "resident_host_observation_stub_requested": bool(resident_host_observation_stub),
        "batch_size": int(batch_size),
        "player_count": int(player_count),
        "root_count": int(root_count),
        "stack_shape": list(stack_shape),
        "mechanics_outcome_schema_id": COMPACT_ROOT_MECHANICS_OUTCOME_SCHEMA_ID,
        "mechanics_outcome_sidecars_present": True,
    }
    if metadata:
        request_metadata.update(
            {str(key): _plain_metadata_value(value) for key, value in metadata.items()}
        )
    return CompactRootBuildRequestV1(
        schema_id=COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID,
        request_kind=str(request_metadata["root_build_request_kind"]),
        search_lane=search_lane_value,
        metadata=request_metadata,
        copy_observation=bool(copy_observation),
        observation_source=observation_source_value,
        resident_host_observation_stub=bool(resident_host_observation_stub),
        batch_size=int(batch_size),
        player_count=int(player_count),
        root_count=int(root_count),
        stack_shape=stack_shape,
        observation=observation,
        action_mask=np.asarray(batch.action_mask),
        reward=np.asarray(batch.reward),
        done=np.asarray(batch.done),
        policy_env_id=np.asarray(batch.policy_env_id),
        policy_env_row=np.asarray(batch.policy_env_row),
        policy_player=np.asarray(batch.policy_player),
        target_reward=np.asarray(batch.target_reward),
        done_root=np.asarray(batch.done_root),
        to_play=np.asarray(batch.to_play),
        active_root_mask=np.asarray(batch.active_root_mask),
        final_observation=(
            None if batch.final_observation is None else np.asarray(batch.final_observation)
        ),
        final_observation_row_mask=np.asarray(batch.final_observation_row_mask),
        terminal_row_mask=np.asarray(batch.terminal_row_mask),
        autoreset_row_mask=np.asarray(batch.autoreset_row_mask),
        joint_action=joint_action,
        resident_observation=resident_observation,
        terminated=np.asarray(batch.terminated),
        truncated=np.asarray(batch.truncated),
        final_reward_map=np.asarray(batch.final_reward_map),
    )


def compact_root_action_context_v1_from_request(
    request: CompactRootBuildRequestV1,
) -> CompactRootActionContextV1:
    """Keep only action-critical root sidecars after owner root submission."""

    if not isinstance(request, CompactRootBuildRequestV1):
        raise TypeError("request must be CompactRootBuildRequestV1")
    root_count = int(request.root_count)
    batch_size = int(request.batch_size)
    player_count = int(request.player_count)
    if root_count <= 0 or root_count != batch_size * player_count:
        raise ReplayCompatibilityError("root action context root shape mismatch")
    action_mask = np.asarray(request.action_mask, dtype=np.bool_).reshape(
        root_count,
        ACTION_COUNT,
    )
    active_root_index = np.flatnonzero(
        np.asarray(request.active_root_mask, dtype=np.bool_).reshape(root_count)
    ).astype(np.int32, copy=False)
    env_row = np.asarray(request.policy_env_row, dtype=np.int32).reshape(root_count)
    player = np.asarray(request.policy_player, dtype=np.int16).reshape(root_count)
    policy_env_id = np.asarray(request.policy_env_id, dtype=np.int64).reshape(root_count)
    observation_dtype = "uint8"
    resident = request.resident_observation
    if resident is not None:
        dtype_text = str(getattr(resident, "dtype", "uint8"))
        observation_dtype = (
            dtype_text.split(".", 1)[1] if dtype_text.startswith("torch.") else dtype_text
        )
    elif request.observation is not None:
        observation_dtype = str(np.asarray(request.observation).dtype)
    return CompactRootActionContextV1(
        schema_id="curvyzero_compact_root_action_context/v1",
        request_schema_id=str(request.schema_id),
        request_kind=str(request.request_kind),
        search_lane=str(request.search_lane),
        metadata=dict(request.metadata or {}),
        copy_observation=bool(request.copy_observation),
        resident_host_observation_stub=bool(request.resident_host_observation_stub),
        observation_dtype=str(observation_dtype),
        batch_size=batch_size,
        player_count=player_count,
        root_count=root_count,
        stack_shape=tuple(int(dim) for dim in request.stack_shape),
        active_root_index=active_root_index.astype(np.int32, copy=True),
        env_row=env_row[active_root_index].astype(np.int32, copy=True),
        player=player[active_root_index].astype(np.int16, copy=True),
        policy_env_id=policy_env_id[active_root_index].astype(np.int64, copy=True),
        active_legal_mask=action_mask[active_root_index].astype(np.bool_, copy=True),
    )


def build_compact_root_batch_v1_from_request(
    request: CompactRootBuildRequestV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CompactRootBatchV1:
    """Build a root batch from an owner-side build request."""

    if not isinstance(request, CompactRootBuildRequestV1):
        raise TypeError("request must be CompactRootBuildRequestV1")
    request_metadata = dict(request.metadata)
    if metadata:
        request_metadata.update(
            {str(key): _plain_metadata_value(value) for key, value in metadata.items()}
        )
    observation = request.observation
    if (
        observation is None
        and request.observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        and bool(request.resident_host_observation_stub)
    ):
        if request.resident_observation is None:
            raise ReplayCompatibilityError(
                "resident root build request requires resident_observation"
            )
        observation = _resident_host_observation_stub_array_v1(
            request.resident_observation,
            batch_size=int(request.batch_size),
            player_count=int(request.player_count),
            stack_shape=tuple(int(dim) for dim in request.stack_shape),
        )
    batch = SimpleNamespace(
        observation=observation,
        action_mask=request.action_mask,
        reward=request.reward,
        done=request.done,
        policy_env_id=request.policy_env_id,
        policy_env_row=request.policy_env_row,
        policy_player=request.policy_player,
        target_reward=request.target_reward,
        done_root=request.done_root,
        to_play=request.to_play,
        active_root_mask=request.active_root_mask,
        final_observation=request.final_observation,
        final_observation_row_mask=request.final_observation_row_mask,
        terminal_row_mask=request.terminal_row_mask,
        autoreset_row_mask=request.autoreset_row_mask,
        joint_action=request.joint_action,
        terminated=request.terminated,
        truncated=request.truncated,
        final_reward_map=request.final_reward_map,
        observation_source=request.observation_source,
        resident_observation=request.resident_observation,
    )
    return build_compact_root_batch_v1(
        batch,
        search_lane=request.search_lane,
        metadata=request_metadata,
        copy_observation=bool(request.copy_observation),
        observation_source=request.observation_source,
        resident_observation=request.resident_observation,
        resident_host_observation_stub=bool(request.resident_host_observation_stub),
    )


def build_compact_root_batch_v1(
    batch: HybridCompactBatch,
    *,
    search_lane: str,
    metadata: Mapping[str, Any] | None = None,
    copy_observation: bool = True,
    observation_source: str | None = None,
    resident_observation: ResidentObservationBatchV1 | None = None,
    resident_host_observation_stub: bool = False,
) -> CompactRootBatchV1:
    """Validate ``HybridCompactBatch`` as the root-batch contract for service tests."""

    search_lane_value = str(search_lane)
    if not search_lane_value:
        raise ReplayCompatibilityError("search_lane must be a non-empty string")

    if observation_source is None:
        observation_source = getattr(
            batch,
            "observation_source",
            COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
        )
    if resident_observation is None:
        resident_observation = getattr(batch, "resident_observation", None)
    observation_source_value = str(observation_source)
    resident_stubbed = False
    if (
        bool(resident_host_observation_stub)
        and observation_source_value == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    ):
        if resident_observation is None:
            raise ReplayCompatibilityError(
                "resident host-observation stub requires resident_observation"
            )
        if bool(copy_observation):
            raise ReplayCompatibilityError("resident host-observation stub cannot be copied")
        batch_size = int(resident_observation.batch_size)
        player_count = int(resident_observation.player_count)
        stack_shape = tuple(int(dim) for dim in resident_observation.stack_shape)
        observation = _resident_host_observation_stub_array_v1(
            resident_observation,
            batch_size=batch_size,
            player_count=player_count,
            stack_shape=stack_shape,
        )
        resident_stubbed = True
    else:
        observation = np.asarray(batch.observation)
        if observation.ndim != 5:
            raise ReplayCompatibilityError("observation must have shape [B,P,C,H,W]")
        batch_size, player_count = observation.shape[:2]
    if batch_size <= 0 or player_count <= 0:
        raise ReplayCompatibilityError("observation batch/player dimensions must be positive")
    root_count = int(batch_size) * int(player_count)

    legal_mask_grid = _binary_bool_array(batch.action_mask, "action_mask")
    if legal_mask_grid.shape != (batch_size, player_count, ACTION_COUNT):
        raise ReplayCompatibilityError("action_mask must have shape [B,P,3]")
    legal_mask = legal_mask_grid.reshape(root_count, ACTION_COUNT)

    policy_env_id = _int_array(batch.policy_env_id, "policy_env_id", (root_count,))
    env_row = _int_array(batch.policy_env_row, "policy_env_row", (root_count,))
    player = _int_array(batch.policy_player, "policy_player", (root_count,))
    _validate_row_major_sidecar(
        policy_env_id=policy_env_id,
        policy_env_row=env_row,
        policy_player=player,
        batch_size=batch_size,
        player_count=player_count,
    )

    done = _bool_array(batch.done, "done", (batch_size,))
    done_root = _bool_array(batch.done_root, "done_root", (root_count,))
    expected_done_root = np.repeat(done, player_count)
    if not np.array_equal(done_root, expected_done_root):
        raise ReplayCompatibilityError("done_root must equal repeat(done, player_count)")

    to_play = _int_array(batch.to_play, "to_play", (root_count,))
    if bool((to_play != DEFAULT_TO_PLAY).any()):
        raise ReplayCompatibilityError("compact fixed-opponent records require to_play=-1")

    active_root_mask = _bool_array(batch.active_root_mask, "active_root_mask", (root_count,))
    expected_active = np.logical_and(~done_root, legal_mask.any(axis=1))
    if not np.array_equal(active_root_mask, expected_active):
        raise ReplayCompatibilityError(
            "active_root_mask must equal non-done roots with any legal action"
        )

    target_reward = np.asarray(batch.target_reward, dtype=np.float32)
    if target_reward.shape != (root_count, 1):
        raise ReplayCompatibilityError("target_reward must have shape [B*P,1]")
    reward = np.asarray(batch.reward, dtype=np.float32)
    if reward.shape != (batch_size, player_count):
        raise ReplayCompatibilityError("reward must have shape [B,P]")
    if not np.allclose(target_reward[:, 0], reward.reshape(root_count), atol=1e-6):
        raise ReplayCompatibilityError("target_reward must equal reward reshaped to [B*P,1]")
    terminated_value = getattr(batch, "terminated", None)
    terminated = (
        done.copy()
        if terminated_value is None
        else _bool_array(terminated_value, "terminated", (batch_size,))
    )
    truncated_value = getattr(batch, "truncated", None)
    truncated = (
        np.zeros((batch_size,), dtype=np.bool_)
        if truncated_value is None
        else _bool_array(truncated_value, "truncated", (batch_size,))
    )
    if not np.array_equal(done, np.logical_or(terminated, truncated)):
        raise ReplayCompatibilityError("done must equal terminated | truncated")
    final_reward_map_value = getattr(batch, "final_reward_map", None)
    final_reward_map = (
        reward.copy()
        if final_reward_map_value is None
        else _float_array(final_reward_map_value, "final_reward_map", (batch_size, player_count))
    )

    final_observation = (
        None if batch.final_observation is None else np.asarray(batch.final_observation)
    )
    if final_observation is not None and final_observation.shape != observation.shape:
        raise ReplayCompatibilityError("final_observation must match observation shape")
    final_observation_row_mask = _bool_array(
        batch.final_observation_row_mask,
        "final_observation_row_mask",
        (batch_size,),
    )
    terminal_row_mask = _bool_array(batch.terminal_row_mask, "terminal_row_mask", (batch_size,))
    autoreset_row_mask = _bool_array(batch.autoreset_row_mask, "autoreset_row_mask", (batch_size,))
    if (
        bool(final_observation_row_mask.any())
        and final_observation is None
        and observation_source_value != COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
    ):
        raise ReplayCompatibilityError("final_observation rows require final_observation")
    if bool(terminal_row_mask.any()) and not bool(
        final_observation_row_mask[terminal_row_mask].all()
    ):
        raise ReplayCompatibilityError(
            "terminal rows require matching final_observation_row_mask entries"
        )
    if observation_source_value not in {
        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
        COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
    }:
        raise ReplayCompatibilityError(f"unknown compact observation_source {observation_source!r}")
    if (
        observation_source_value == COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
        and resident_observation is not None
    ):
        raise ReplayCompatibilityError(
            "resident_observation was provided but observation_source is host_array_v1"
        )
    if observation_source_value == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1:
        if resident_observation is None:
            raise ReplayCompatibilityError(
                "resident_device_v1 observation_source requires resident_observation"
            )
        validate_resident_observation_batch_v1(
            resident_observation,
            batch_size=batch_size,
            player_count=player_count,
            stack_shape=tuple(int(dim) for dim in observation.shape[2:]),
        )
        if final_observation is not None:
            raise ReplayCompatibilityError(
                "resident_device_v1 terminal final_observation rows must be resident-owned"
            )
        resident_final_mask = getattr(resident_observation, "final_observation_row_mask", None)
        if resident_final_mask is not None:
            resident_final_mask_array = _bool_array(
                resident_final_mask,
                "resident final_observation_row_mask",
                (batch_size,),
            )
            if not np.array_equal(resident_final_mask_array, final_observation_row_mask):
                raise ReplayCompatibilityError(
                    "resident final_observation_row_mask must match compact batch"
                )
        if bool(final_observation_row_mask.any()):
            if not _resident_final_storage_present(resident_observation):
                raise ReplayCompatibilityError(
                    "resident_device_v1 terminal final_observation rows require "
                    "resident final_device_observation"
                )
            if resident_final_mask is None:
                raise ReplayCompatibilityError(
                    "resident_device_v1 terminal final_observation rows require "
                    "resident final_observation_row_mask"
                )

    root_metadata = {
        "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
        "schema_id": "curvyzero_compact_root_batch/v1",
        "search_lane": search_lane_value,
        "observation_source": observation_source_value,
        "observation_copied": bool(copy_observation),
        "resident_host_observation_stub_requested": bool(resident_host_observation_stub),
        "resident_host_observation_stubbed": bool(resident_stubbed),
        "resident_host_observation_stub_kind": (
            "zero_stride_shape_only_v1" if resident_stubbed else ""
        ),
        "resident_host_observation_stub_materialized_bytes": 0 if resident_stubbed else 0,
        "resident_host_observation_stub_logical_bytes": int(observation.nbytes)
        if resident_stubbed
        else 0,
        "host_observation_authoritative": (
            observation_source_value == COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1
        ),
        "resident_device_observation_authoritative": (
            observation_source_value == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        ),
        "batch_size": int(batch_size),
        "player_count": int(player_count),
        "root_count": int(root_count),
        "active_root_count": int(np.count_nonzero(active_root_mask)),
        "observation_shape": list(observation.shape),
        "observation_dtype": str(observation.dtype),
        "fixed_opponent_to_play": int(DEFAULT_TO_PLAY),
        "policy_env_id_semantics": "opaque_unique_root_id",
        "env_row_player_semantics": "row_major_root_position",
        "mechanics_outcome_schema_id": COMPACT_ROOT_MECHANICS_OUTCOME_SCHEMA_ID,
        "mechanics_outcome_sidecars_present": True,
        "mechanics_outcome_reward_source": "target_reward",
        "mechanics_outcome_done_source": "done_root",
    }
    if resident_observation is not None:
        resident_metadata = dict(getattr(resident_observation, "metadata", {}) or {})
        root_metadata.update(
            {
                "resident_observation_schema_id": RESIDENT_OBSERVATION_BATCH_SCHEMA_ID,
                "resident_observation_owner": RESIDENT_OBSERVATION_OWNER,
                "resident_observation_generation_id": int(resident_observation.generation_id),
                "resident_observation_record_index": int(resident_observation.generation_id),
                "resident_observation_freshness": "fresh_current_step",
                "resident_observation_fresh_for_step_index": int(
                    resident_observation.fresh_for_step_index
                ),
                "resident_observation_source_backend": str(resident_observation.source_backend),
                "resident_observation_host_fallback_allowed": bool(
                    resident_observation.host_fallback_allowed
                ),
                "resident_observation_host_fallback_used": False,
                "resident_observation_h2d_bytes": 0.0,
                "resident_observation_d2h_bytes": 0.0,
                "resident_observation_device": str(resident_observation.device),
                "resident_final_device_observation_present": (
                    _resident_final_storage_present(resident_observation)
                ),
                "resident_final_observation_row_count": int(
                    np.count_nonzero(final_observation_row_mask)
                ),
                "resident_final_device_observation_storage": str(
                    resident_metadata.get(
                        "resident_final_device_observation_storage",
                        "dense"
                        if getattr(resident_observation, "final_device_observation", None)
                        is not None
                        else "sparse_rows"
                        if getattr(
                            resident_observation,
                            "final_device_observation_rows",
                            None,
                        )
                        is not None
                        else "none",
                    )
                ),
                "resident_final_device_observation_sparse_row_count": int(
                    resident_metadata.get(
                        "resident_final_device_observation_sparse_row_count",
                        0,
                    )
                ),
            }
        )
    if metadata:
        root_metadata.update(
            {str(key): _plain_metadata_value(value) for key, value in metadata.items()}
        )

    root_observation = observation.reshape(root_count, *observation.shape[2:])
    if bool(copy_observation):
        root_observation = root_observation.copy()
    root_final_observation = None
    if final_observation is not None:
        root_final_observation = final_observation.reshape(
            root_count,
            *final_observation.shape[2:],
        )
        if bool(copy_observation):
            root_final_observation = root_final_observation.copy()

    return CompactRootBatchV1(
        observation=root_observation,
        legal_mask=legal_mask.astype(bool, copy=True),
        active_root_mask=active_root_mask.astype(bool, copy=True),
        to_play=to_play.astype(np.int64, copy=True),
        env_row=env_row.astype(np.int32, copy=True),
        player=player.astype(np.int16, copy=True),
        policy_env_id=policy_env_id.astype(np.int64, copy=True),
        target_reward=target_reward.astype(np.float32, copy=True),
        done_root=done_root.astype(bool, copy=True),
        final_observation=root_final_observation,
        final_observation_row_mask=final_observation_row_mask.astype(bool, copy=True),
        terminal_row_mask=terminal_row_mask.astype(bool, copy=True),
        autoreset_row_mask=autoreset_row_mask.astype(bool, copy=True),
        metadata=root_metadata,
        resident_observation=resident_observation,
        observation_source=observation_source_value,
        terminated=terminated.astype(bool, copy=True),
        truncated=truncated.astype(bool, copy=True),
        final_reward_map=final_reward_map.astype(np.float32, copy=True),
    )


def compact_transition_outcome_v1_from_next_root_batch(
    root_batch: CompactRootBatchV1,
    *,
    batch_size: int | None = None,
    player_count: int | None = None,
) -> CompactRootTransitionOutcomeV1:
    """Derive next-transition outcome arrays from a complete next root batch.

    This is the fail-closed contract the owner-local transition path needs:
    reward/done come from the root target sidecars, while terminal flavor and
    final rewards must be explicit mechanics sidecars on the next root.
    """

    if not isinstance(root_batch, CompactRootBatchV1):
        raise TypeError("root_batch must be CompactRootBatchV1")
    done_root = _bool_array(
        root_batch.done_root,
        "root_batch.done_root",
        (int(np.asarray(root_batch.done_root).size),),
    )
    root_count = int(done_root.size)
    if root_count <= 0:
        raise ReplayCompatibilityError("next root batch must contain at least one root")
    final_row_mask = np.asarray(root_batch.final_observation_row_mask, dtype=np.bool_)
    if final_row_mask.ndim != 1 or final_row_mask.size <= 0:
        raise ReplayCompatibilityError("next root final_observation_row_mask must be [B]")

    metadata = dict(root_batch.metadata or {})
    if batch_size is None:
        batch_size = (
            _int(metadata["batch_size"], "batch_size")
            if "batch_size" in metadata
            else int(final_row_mask.size)
        )
    if player_count is None:
        player_count = (
            _int(metadata["player_count"], "player_count")
            if "player_count" in metadata
            else int(root_count // int(batch_size))
        )
    batch_size_int = int(batch_size)
    player_count_int = int(player_count)
    if batch_size_int <= 0 or player_count_int <= 0:
        raise ReplayCompatibilityError("batch_size and player_count must be positive")
    if batch_size_int * player_count_int != root_count:
        raise ReplayCompatibilityError("next root dimensions do not match root count")

    target_reward = np.asarray(root_batch.target_reward, dtype=np.float32)
    if target_reward.shape != (root_count, 1):
        raise ReplayCompatibilityError("next root target_reward must have shape [B*P,1]")
    reward_grid = target_reward[:, 0].reshape(batch_size_int, player_count_int)
    done_grid = done_root.reshape(batch_size_int, player_count_int)
    if not np.array_equal(done_grid, np.repeat(done_grid[:, :1], player_count_int, axis=1)):
        raise ReplayCompatibilityError("next root done_root must be repeated per player")
    next_done = done_grid[:, 0]

    missing = [
        name
        for name, value in (
            ("terminated", root_batch.terminated),
            ("truncated", root_batch.truncated),
            ("final_reward_map", root_batch.final_reward_map),
        )
        if value is None
    ]
    if missing:
        raise ReplayCompatibilityError(
            "next root missing mechanics outcome sidecars: " + ", ".join(missing)
        )
    terminated = _bool_array(root_batch.terminated, "next root terminated", (batch_size_int,))
    truncated = _bool_array(root_batch.truncated, "next root truncated", (batch_size_int,))
    if not np.array_equal(next_done, np.logical_or(terminated, truncated)):
        raise ReplayCompatibilityError("next root done must equal terminated | truncated")
    final_reward_map = _float_array(
        root_batch.final_reward_map,
        "next root final_reward_map",
        (batch_size_int, player_count_int),
    )
    final_row_mask = _bool_array(
        final_row_mask,
        "next root final_observation_row_mask",
        (batch_size_int,),
    )
    if bool(next_done.any()) and not bool(final_row_mask[next_done].all()):
        raise ReplayCompatibilityError(
            "next root done rows require matching final_observation_row_mask entries"
        )

    return CompactRootTransitionOutcomeV1(
        next_reward=reward_grid.astype(np.float32, copy=True),
        next_done=next_done.astype(np.bool_, copy=True),
        next_terminated=terminated.astype(np.bool_, copy=True),
        next_truncated=truncated.astype(np.bool_, copy=True),
        next_final_reward_map=final_reward_map.astype(np.float32, copy=True),
        next_final_observation_row_mask=final_row_mask.astype(np.bool_, copy=True),
    )


def compact_transition_outcome_v1_from_root_build_request(
    request: CompactRootBuildRequestV1,
    *,
    batch_size: int | None = None,
    player_count: int | None = None,
) -> CompactRootTransitionOutcomeV1:
    """Derive next-transition outcome arrays without materializing a root batch."""

    if not isinstance(request, CompactRootBuildRequestV1):
        raise TypeError("request must be CompactRootBuildRequestV1")
    root_count = int(request.root_count)
    if root_count <= 0:
        raise ReplayCompatibilityError("next root build request must contain at least one root")
    final_row_mask = np.asarray(request.final_observation_row_mask, dtype=np.bool_)
    if final_row_mask.ndim != 1 or final_row_mask.size <= 0:
        raise ReplayCompatibilityError(
            "next root build request final_observation_row_mask must be [B]"
        )
    if batch_size is None:
        batch_size = int(request.batch_size)
    if player_count is None:
        player_count = int(request.player_count)
    batch_size_int = int(batch_size)
    player_count_int = int(player_count)
    if batch_size_int <= 0 or player_count_int <= 0:
        raise ReplayCompatibilityError("batch_size and player_count must be positive")
    if batch_size_int * player_count_int != root_count:
        raise ReplayCompatibilityError("next root build request dimensions do not match root count")

    target_reward = np.asarray(request.target_reward, dtype=np.float32)
    if target_reward.shape != (root_count, 1):
        raise ReplayCompatibilityError(
            "next root build request target_reward must have shape [B*P,1]"
        )
    done_root = _bool_array(
        request.done_root,
        "root build request done_root",
        (root_count,),
    )
    done_grid = done_root.reshape(batch_size_int, player_count_int)
    if not np.array_equal(done_grid, np.repeat(done_grid[:, :1], player_count_int, axis=1)):
        raise ReplayCompatibilityError(
            "next root build request done_root must be repeated per player"
        )
    next_done = done_grid[:, 0]

    missing = [
        name
        for name, value in (
            ("terminated", request.terminated),
            ("truncated", request.truncated),
            ("final_reward_map", request.final_reward_map),
        )
        if value is None
    ]
    if missing:
        raise ReplayCompatibilityError(
            "next root build request missing mechanics outcome sidecars: "
            + ", ".join(missing)
        )
    terminated = _bool_array(request.terminated, "next root terminated", (batch_size_int,))
    truncated = _bool_array(request.truncated, "next root truncated", (batch_size_int,))
    if not np.array_equal(next_done, np.logical_or(terminated, truncated)):
        raise ReplayCompatibilityError(
            "next root build request done must equal terminated | truncated"
        )
    final_reward_map = _float_array(
        request.final_reward_map,
        "next root final_reward_map",
        (batch_size_int, player_count_int),
    )
    final_row_mask = _bool_array(
        final_row_mask,
        "next root final_observation_row_mask",
        (batch_size_int,),
    )
    if bool(next_done.any()) and not bool(final_row_mask[next_done].all()):
        raise ReplayCompatibilityError(
            "next root build request done rows require matching final_observation_row_mask entries"
        )

    return CompactRootTransitionOutcomeV1(
        next_reward=target_reward[:, 0]
        .reshape(batch_size_int, player_count_int)
        .astype(np.float32, copy=True),
        next_done=next_done.astype(np.bool_, copy=True),
        next_terminated=terminated.astype(np.bool_, copy=True),
        next_truncated=truncated.astype(np.bool_, copy=True),
        next_final_reward_map=final_reward_map.astype(np.float32, copy=True),
        next_final_observation_row_mask=final_row_mask.astype(np.bool_, copy=True),
    )


def _resident_host_observation_stub_array_v1(
    resident: ResidentObservationBatchV1,
    *,
    batch_size: int,
    player_count: int,
    stack_shape: tuple[int, ...],
) -> np.ndarray:
    dtype = _numpy_dtype_from_resident_dtype(str(resident.dtype))
    scalar = np.zeros((), dtype=dtype)
    return np.broadcast_to(
        scalar,
        (int(batch_size), int(player_count), *tuple(int(dim) for dim in stack_shape)),
    )


def _numpy_dtype_from_resident_dtype(dtype_text: str) -> np.dtype:
    text = str(dtype_text)
    if text.startswith("torch."):
        text = text.split(".", 1)[1]
    aliases = {
        "bool": np.bool_,
        "uint8": np.uint8,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    if text in aliases:
        return np.dtype(aliases[text])
    try:
        return np.dtype(text)
    except TypeError as exc:
        raise ReplayCompatibilityError(
            f"unsupported resident observation dtype for host stub: {dtype_text!r}"
        ) from exc


def validate_resident_observation_batch_v1(
    resident: ResidentObservationBatchV1,
    *,
    batch_size: int,
    player_count: int,
    stack_shape: tuple[int, int, int],
) -> None:
    """Validate a resident observation handle without importing a GPU framework."""

    if int(resident.generation_id) < 0:
        raise ReplayCompatibilityError("resident generation_id must be nonnegative")
    if int(resident.fresh_for_step_index) < 0:
        raise ReplayCompatibilityError("resident fresh_for_step_index must be nonnegative")
    if int(resident.batch_size) != int(batch_size):
        raise ReplayCompatibilityError("resident batch_size mismatch")
    if int(resident.player_count) != int(player_count):
        raise ReplayCompatibilityError("resident player_count mismatch")
    if tuple(int(dim) for dim in resident.stack_shape) != tuple(int(dim) for dim in stack_shape):
        raise ReplayCompatibilityError("resident stack_shape mismatch")
    if not bool(resident.row_major_order):
        raise ReplayCompatibilityError("resident observation must be row-major")
    if bool(resident.host_fallback_allowed):
        raise ReplayCompatibilityError("resident observation must not allow host fallback")
    if not str(resident.source_backend):
        raise ReplayCompatibilityError("resident source_backend must be non-empty")
    device_shape = _shape_tuple(resident.device_observation)
    expected_shape = (int(batch_size), int(player_count), *tuple(int(dim) for dim in stack_shape))
    if device_shape != expected_shape:
        raise ReplayCompatibilityError(
            f"resident device_observation shape mismatch: expected {expected_shape}, "
            f"got {device_shape}"
        )
    dtype_text = _dtype_text(resident.device_observation)
    if str(resident.dtype) != dtype_text:
        raise ReplayCompatibilityError(
            f"resident dtype mismatch: expected {resident.dtype!r}, got {dtype_text!r}"
        )
    if resident.root_device_observation is not None:
        root_shape = _shape_tuple(resident.root_device_observation)
        expected_root_shape = (
            int(batch_size) * int(player_count),
            *tuple(int(dim) for dim in stack_shape),
        )
        if root_shape != expected_root_shape:
            raise ReplayCompatibilityError(
                "resident root_device_observation shape mismatch: "
                f"expected {expected_root_shape}, got {root_shape}"
            )
    final_mask = getattr(resident, "final_observation_row_mask", None)
    final_device_observation = getattr(resident, "final_device_observation", None)
    root_final_device_observation = getattr(
        resident,
        "root_final_device_observation",
        None,
    )
    sparse_final_device_observation = getattr(
        resident,
        "final_device_observation_rows",
        None,
    )
    sparse_final_row_indices = getattr(
        resident,
        "final_device_observation_row_indices",
        None,
    )
    if final_mask is None:
        if (
            final_device_observation is not None
            or root_final_device_observation is not None
            or sparse_final_device_observation is not None
            or sparse_final_row_indices is not None
        ):
            raise ReplayCompatibilityError(
                "resident final device observations require final_observation_row_mask"
            )
        return
    final_mask_array = _bool_array(
        final_mask,
        "resident final_observation_row_mask",
        (int(batch_size),),
    )
    if final_device_observation is not None and sparse_final_device_observation is not None:
        raise ReplayCompatibilityError(
            "resident final observations must be dense or sparse, not both"
        )
    if (
        bool(final_mask_array.any())
        and final_device_observation is None
        and sparse_final_device_observation is None
    ):
        raise ReplayCompatibilityError(
            "resident final_observation_row_mask rows require final_device_observation"
        )
    if root_final_device_observation is not None and final_device_observation is None:
        raise ReplayCompatibilityError(
            "resident root_final_device_observation requires final_device_observation"
        )
    if final_device_observation is not None:
        final_shape = _shape_tuple(final_device_observation)
        if final_shape != expected_shape:
            raise ReplayCompatibilityError(
                "resident final_device_observation shape mismatch: "
                f"expected {expected_shape}, got {final_shape}"
            )
        final_dtype_text = _dtype_text(final_device_observation)
        if final_dtype_text != dtype_text:
            raise ReplayCompatibilityError(
                "resident final_device_observation dtype mismatch: "
                f"expected {dtype_text!r}, got {final_dtype_text!r}"
            )
    if sparse_final_device_observation is not None:
        if sparse_final_row_indices is None:
            raise ReplayCompatibilityError("resident sparse final observations require row indices")
        sparse_indices = np.asarray(sparse_final_row_indices, dtype=np.int64).reshape(-1)
        expected_sparse_indices = np.flatnonzero(final_mask_array).astype(
            np.int64,
            copy=False,
        )
        if not np.array_equal(np.sort(sparse_indices), expected_sparse_indices):
            raise ReplayCompatibilityError(
                "resident sparse final observation rows must match final mask"
            )
        if np.unique(sparse_indices).shape[0] != sparse_indices.shape[0]:
            raise ReplayCompatibilityError("resident sparse final observation rows must be unique")
        sparse_shape = _shape_tuple(sparse_final_device_observation)
        expected_sparse_shape = (
            int(sparse_indices.shape[0]),
            int(player_count),
            *tuple(int(dim) for dim in stack_shape),
        )
        if sparse_shape != expected_sparse_shape:
            raise ReplayCompatibilityError(
                "resident sparse final_device_observation shape mismatch: "
                f"expected {expected_sparse_shape}, got {sparse_shape}"
            )
        sparse_dtype_text = _dtype_text(sparse_final_device_observation)
        if sparse_dtype_text != dtype_text:
            raise ReplayCompatibilityError(
                "resident sparse final_device_observation dtype mismatch: "
                f"expected {dtype_text!r}, got {sparse_dtype_text!r}"
            )
    if root_final_device_observation is not None:
        root_final_shape = _shape_tuple(root_final_device_observation)
        expected_root_shape = (
            int(batch_size) * int(player_count),
            *tuple(int(dim) for dim in stack_shape),
        )
        if root_final_shape != expected_root_shape:
            raise ReplayCompatibilityError(
                "resident root_final_device_observation shape mismatch: "
                f"expected {expected_root_shape}, got {root_final_shape}"
            )


def _resident_final_storage_present(resident: Any) -> bool:
    return (
        getattr(resident, "final_device_observation", None) is not None
        or getattr(resident, "final_device_observation_rows", None) is not None
    )


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ReplayCompatibilityError("resident observation must expose shape")
    return tuple(int(dim) for dim in shape)


def _dtype_text(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise ReplayCompatibilityError("resident observation must expose dtype")
    return str(dtype)


def validate_compact_search_result_v1(
    root_batch: CompactRootBatchV1,
    *,
    selected_action: np.ndarray | Sequence[int],
    visit_policy: np.ndarray | Sequence[Sequence[float]],
    root_value: np.ndarray | Sequence[float],
    search_impl: str,
    num_simulations: int,
    raw_visit_counts: np.ndarray | Sequence[Sequence[int]] | None = None,
    predicted_value: np.ndarray | Sequence[float] | None = None,
    predicted_policy_logits: np.ndarray | Sequence[Sequence[float]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CompactSearchResultV1:
    """Validate compact active-root search output without per-root record objects."""

    search_impl_value = str(search_impl)
    if not search_impl_value:
        raise ReplayCompatibilityError("search_impl must be a non-empty string")
    simulations = _int(num_simulations, "num_simulations")
    if simulations < 0:
        raise ReplayCompatibilityError("num_simulations must be nonnegative")

    root_count = int(root_batch.legal_mask.shape[0])
    _expect_root_batch_shapes(root_batch, root_count=root_count)
    active_root_mask = _bool_array(root_batch.active_root_mask, "active_root_mask", (root_count,))
    active_root_indices = np.flatnonzero(active_root_mask).astype(np.int32, copy=False)
    active_count = int(active_root_indices.size)
    selected = _int_array(selected_action, "selected_action", (active_count,))
    visits = _float_array(visit_policy, "visit_policy", (active_count, ACTION_COUNT))
    values = _float_array(root_value, "root_value", (active_count,))
    legal_mask = _binary_bool_array(root_batch.legal_mask, "legal_mask")
    active_legal_mask = legal_mask[active_root_indices]

    if bool(((selected < 0) | (selected >= ACTION_COUNT)).any()):
        raise ReplayCompatibilityError("selected_action is out of range")
    if selected.size and not bool(active_legal_mask[np.arange(selected.size), selected].all()):
        raise ReplayCompatibilityError("selected_action is illegal")
    if not np.isfinite(visits).all():
        raise ReplayCompatibilityError("visit_policy must be finite")
    if bool((visits < 0.0).any()):
        raise ReplayCompatibilityError("visit_policy must be nonnegative")
    if not np.allclose(visits.sum(axis=1), 1.0, atol=1e-6):
        raise ReplayCompatibilityError("visit_policy must sum to 1")
    if bool((visits[~active_legal_mask] > 1e-7).any()):
        raise ReplayCompatibilityError("visit_policy assigns mass to illegal actions")
    if not np.isfinite(values).all():
        raise ReplayCompatibilityError("root_value must be finite")

    raw_counts = None
    if raw_visit_counts is not None:
        raw_counts = _float_array(
            raw_visit_counts,
            "raw_visit_counts",
            (active_count, ACTION_COUNT),
        )
        if bool((raw_counts < 0.0).any()):
            raise ReplayCompatibilityError("raw_visit_counts must be nonnegative")
        if bool((raw_counts[~active_legal_mask] > 1e-7).any()):
            raise ReplayCompatibilityError("raw_visit_counts assigns mass to illegal actions")

    predicted_values = None
    if predicted_value is not None:
        predicted_values = _float_array(predicted_value, "predicted_value", (active_count,))
        if not np.isfinite(predicted_values).all():
            raise ReplayCompatibilityError("predicted_value must be finite")

    predicted_logits = None
    if predicted_policy_logits is not None:
        predicted_logits = _float_array(
            predicted_policy_logits,
            "predicted_policy_logits",
            (active_count, ACTION_COUNT),
        )
        if not np.isfinite(predicted_logits).all():
            raise ReplayCompatibilityError("predicted_policy_logits must be finite")

    result_metadata = {
        "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
        "schema_id": "curvyzero_compact_search_result/v1",
        "search_impl": search_impl_value,
        "num_simulations": int(simulations),
        "root_count": int(root_count),
        "active_root_count": int(active_count),
        "illegal_action_count": 0,
    }
    if metadata:
        result_metadata.update(
            {str(key): _plain_metadata_value(value) for key, value in metadata.items()}
        )

    return CompactSearchResultV1(
        root_index=active_root_indices.copy(),
        env_row=root_batch.env_row[active_root_indices].astype(np.int32, copy=True),
        player=root_batch.player[active_root_indices].astype(np.int16, copy=True),
        policy_env_id=root_batch.policy_env_id[active_root_indices].astype(
            np.int64,
            copy=True,
        ),
        selected_action=selected.astype(np.int16, copy=True),
        visit_policy=visits.astype(np.float32, copy=True),
        root_value=values.astype(np.float32, copy=True),
        raw_visit_counts=None if raw_counts is None else raw_counts.astype(np.float32, copy=True),
        predicted_value=(
            None if predicted_values is None else predicted_values.astype(np.float32, copy=True)
        ),
        predicted_policy_logits=(
            None if predicted_logits is None else predicted_logits.astype(np.float32, copy=True)
        ),
        metadata=result_metadata,
    )


def build_compact_replay_chunk_v1_from_search_result(
    chunk: Any,
    batch: HybridCompactBatch,
    root_batch: CompactRootBatchV1,
    search_result: CompactSearchResultV1,
    *,
    record_index: int,
    policy_source: str,
    metadata: Mapping[str, Any] | None = None,
) -> CompactReplayChunkV1:
    """Build the first compact service chunk and materialize target rows for parity."""

    record_index_int = _int(record_index, "record_index")
    if record_index_int < 0:
        raise ReplayCompatibilityError("record_index must be nonnegative")
    if not np.array_equal(
        search_result.root_index,
        np.flatnonzero(root_batch.active_root_mask).astype(np.int32, copy=False),
    ):
        raise ReplayCompatibilityError("search_result roots must match root_batch active roots")
    if not np.array_equal(search_result.env_row, root_batch.env_row[search_result.root_index]):
        raise ReplayCompatibilityError("search_result env_row does not match root_batch")
    if not np.array_equal(search_result.player, root_batch.player[search_result.root_index]):
        raise ReplayCompatibilityError("search_result player does not match root_batch")

    target_rows = build_compact_target_rows_from_search_arrays_v0(
        chunk,
        batch,
        selected_action=search_result.selected_action,
        visit_policy=search_result.visit_policy,
        root_value=search_result.root_value,
        record_index=record_index_int,
        policy_source=policy_source,
    )
    replay_metadata = {
        "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
        "schema_id": "curvyzero_compact_replay_chunk/v1",
        "record_index": int(record_index_int),
        "next_record_index": int(record_index_int + 1),
        "policy_source": str(policy_source),
        "target_row_count": int(target_rows.action.size),
        "root_batch_schema_id": str(root_batch.metadata.get("schema_id")),
        "search_result_schema_id": str(search_result.metadata.get("schema_id")),
    }
    if metadata:
        replay_metadata.update(
            {str(key): _plain_metadata_value(value) for key, value in metadata.items()}
        )
    return CompactReplayChunkV1(
        record_index=record_index_int,
        next_record_index=record_index_int + 1,
        root_batch=root_batch,
        search_result=search_result,
        target_rows=target_rows,
        metadata=replay_metadata,
    )


def build_compact_replay_index_rows_v1_from_search_result(
    batch: HybridCompactBatch,
    root_batch: CompactRootBatchV1,
    search_result: CompactSearchResultV1,
    *,
    record_index: int,
    next_joint_action: np.ndarray,
    next_reward: np.ndarray,
    next_done: np.ndarray,
    policy_source: str,
    next_terminated: np.ndarray | None = None,
    next_truncated: np.ndarray | None = None,
    next_final_reward_map: np.ndarray | None = None,
    next_final_observation_row_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CompactReplayIndexRowsV1:
    """Build compact replay rows without materializing observation tensors."""

    record_index_int = _int(record_index, "record_index")
    if record_index_int < 0:
        raise ReplayCompatibilityError("record_index must be nonnegative")
    if not str(policy_source):
        raise ReplayCompatibilityError("policy_source must be a non-empty string")
    validate_compact_search_result_identity_v1(root_batch, search_result)

    legal_mask = _binary_bool_array(root_batch.legal_mask, "legal_mask")
    root_count = int(legal_mask.shape[0])
    if legal_mask.shape != (root_count, ACTION_COUNT):
        raise ReplayCompatibilityError("legal_mask must have shape [root,3]")
    action_mask = legal_mask[search_result.root_index].astype(bool, copy=True)
    selected = np.asarray(search_result.selected_action, dtype=np.int16)
    if selected.shape != (int(search_result.root_index.size),):
        raise ReplayCompatibilityError("selected_action count mismatch")
    if selected.size and not bool(action_mask[np.arange(selected.size), selected].all()):
        raise ReplayCompatibilityError("selected_action is illegal")

    observation = np.asarray(batch.observation)
    if observation.ndim != 5:
        raise ReplayCompatibilityError("compact batch observation must have shape [B,P,C,H,W]")
    batch_size, player_count = int(observation.shape[0]), int(observation.shape[1])
    expected_root_count = batch_size * player_count
    if root_count != expected_root_count:
        raise ReplayCompatibilityError("root_batch root count does not match compact batch")
    expected_legal_mask = _binary_bool_array(batch.action_mask, "batch.action_mask").reshape(
        expected_root_count,
        ACTION_COUNT,
    )
    if not np.array_equal(root_batch.legal_mask, expected_legal_mask):
        raise ReplayCompatibilityError("root_batch legal_mask does not match compact batch")
    expected_policy_env_id = _int_array(
        batch.policy_env_id,
        "batch.policy_env_id",
        (expected_root_count,),
    )
    expected_env_row = _int_array(
        batch.policy_env_row, "batch.policy_env_row", (expected_root_count,)
    )
    expected_player = _int_array(batch.policy_player, "batch.policy_player", (expected_root_count,))
    if not np.array_equal(root_batch.policy_env_id, expected_policy_env_id):
        raise ReplayCompatibilityError("root_batch policy_env_id does not match compact batch")
    if not np.array_equal(root_batch.env_row, expected_env_row):
        raise ReplayCompatibilityError("root_batch env_row does not match compact batch")
    if not np.array_equal(root_batch.player, expected_player):
        raise ReplayCompatibilityError("root_batch player does not match compact batch")
    env_row = np.asarray(search_result.env_row, dtype=np.int32)
    player = np.asarray(search_result.player, dtype=np.int16)
    joint_action = _int_array(next_joint_action, "next_joint_action", (batch_size, player_count))
    reward_grid = _float_array(next_reward, "next_reward", (batch_size, player_count))
    done_grid = _bool_array(next_done, "next_done", (batch_size,))
    terminated_grid = (
        done_grid
        if next_terminated is None
        else _bool_array(next_terminated, "next_terminated", (batch_size,))
    )
    truncated_grid = (
        np.zeros((batch_size,), dtype=np.bool_)
        if next_truncated is None
        else _bool_array(next_truncated, "next_truncated", (batch_size,))
    )
    if not np.array_equal(done_grid, np.logical_or(terminated_grid, truncated_grid)):
        raise ReplayCompatibilityError("next_done must equal next_terminated | next_truncated")
    final_row_mask = (
        np.zeros((batch_size,), dtype=np.bool_)
        if next_final_observation_row_mask is None
        else _bool_array(
            next_final_observation_row_mask,
            "next_final_observation_row_mask",
            (batch_size,),
        )
    )
    if bool(done_grid.any()) and not bool(final_row_mask[done_grid].all()):
        raise ReplayCompatibilityError(
            "next_done rows require matching next_final_observation_row_mask entries"
        )
    if bool(final_row_mask.any()) and next_final_reward_map is None:
        raise ReplayCompatibilityError("next_final_observation rows require next_final_reward_map")
    final_reward_grid = (
        reward_grid
        if next_final_reward_map is None
        else _float_array(
            next_final_reward_map, "next_final_reward_map", (batch_size, player_count)
        )
    )

    env_row_i64 = env_row.astype(np.int64, copy=False)
    player_i64 = player.astype(np.int64, copy=False)
    if selected.size and not np.array_equal(joint_action[env_row_i64, player_i64], selected):
        raise ReplayCompatibilityError("selected_action does not match next_joint_action")

    policy_row = np.arange(int(search_result.root_index.size), dtype=np.int32)
    final_reward = reward_grid[env_row_i64, player_i64].astype(np.float32, copy=True)
    final_rows_for_targets = final_row_mask[env_row_i64]
    if bool(final_rows_for_targets.any()):
        final_reward[final_rows_for_targets] = final_reward_grid[
            env_row_i64[final_rows_for_targets],
            player_i64[final_rows_for_targets],
        ].astype(np.float32, copy=False)

    replay_metadata = {
        "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
        "schema_id": "curvyzero_compact_replay_index_rows/v1",
        "record_index": int(record_index_int),
        "next_record_index": int(record_index_int + 1),
        "policy_source": str(policy_source),
        "index_row_count": int(search_result.root_index.size),
        COMPACT_REPLAY_ENV_PLAYER_IDENTITY_DIGEST_KEY: _env_player_identity_digest(
            env_row,
            player,
        ),
        "root_batch_schema_id": str(root_batch.metadata.get("schema_id")),
        "search_result_schema_id": str(search_result.metadata.get("schema_id")),
        "search_impl": str(search_result.metadata.get("search_impl", "")),
        "num_simulations": int(search_result.metadata.get("num_simulations", 0)),
        "active_root_count": int(search_result.selected_action.size),
        "search_replay_payload_digest": str(
            search_result.metadata.get("search_replay_payload_digest", "")
        ),
        "done_row_count": int(done_grid[env_row_i64].sum()),
        "terminated_row_count": int(terminated_grid[env_row_i64].sum()),
        "truncated_row_count": int(truncated_grid[env_row_i64].sum()),
        "next_final_observation_row_count": int(final_rows_for_targets.sum()),
        "observation_materialized": False,
        "next_observation_materialized": False,
    }
    if COMPACT_REPLAY_TERMINAL_METADATA_ROW_INDICES_ENABLED:
        replay_metadata.update(
            {
                "done_row_indices": np.flatnonzero(done_grid[env_row_i64])
                .astype(np.int64, copy=False)
                .tolist(),
                "terminated_row_indices": np.flatnonzero(terminated_grid[env_row_i64])
                .astype(np.int64, copy=False)
                .tolist(),
                "truncated_row_indices": np.flatnonzero(truncated_grid[env_row_i64])
                .astype(np.int64, copy=False)
                .tolist(),
                "next_final_observation_row_indices": np.flatnonzero(final_rows_for_targets)
                .astype(np.int64, copy=False)
                .tolist(),
            }
        )
    replay_metadata.update(compact_policy_refresh_metadata_subset_v1(search_result.metadata))
    if metadata:
        replay_metadata.update(
            {str(key): _plain_metadata_value(value) for key, value in metadata.items()}
        )
    return CompactReplayIndexRowsV1(
        metadata=replay_metadata,
        record_index=record_index_int,
        next_record_index=record_index_int + 1,
        compact_root_row=np.asarray(search_result.root_index, dtype=np.int32).copy(),
        policy_env_id=np.asarray(search_result.policy_env_id, dtype=np.int64).copy(),
        policy_row=policy_row,
        env_row=env_row.astype(np.int32, copy=True),
        player=player.astype(np.int16, copy=True),
        action=selected.astype(np.int16, copy=True),
        action_mask=action_mask,
        policy_target=np.asarray(search_result.visit_policy, dtype=np.float32).copy(),
        root_value=np.asarray(search_result.root_value, dtype=np.float32).copy(),
        reward=reward_grid[env_row_i64, player_i64].astype(np.float32, copy=True),
        final_reward=final_reward,
        done=done_grid[env_row_i64].astype(np.bool_, copy=True),
        terminated=terminated_grid[env_row_i64].astype(np.bool_, copy=True),
        truncated=truncated_grid[env_row_i64].astype(np.bool_, copy=True),
        next_final_observation_row=final_rows_for_targets.astype(np.bool_, copy=True),
        to_play=np.full((int(search_result.root_index.size),), DEFAULT_TO_PLAY, dtype=np.int64),
        policy_source=str(policy_source),
    )


def build_compact_replay_index_rows_v1_from_owner_action_context_payload(
    root_action_context: CompactRootActionContextV1,
    action_step: Any,
    replay_payload: Any,
    *,
    record_index: int,
    next_joint_action: np.ndarray,
    next_reward: np.ndarray,
    next_done: np.ndarray,
    policy_source: str,
    next_terminated: np.ndarray | None = None,
    next_truncated: np.ndarray | None = None,
    next_final_reward_map: np.ndarray | None = None,
    next_final_observation_row_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CompactReplayIndexRowsV1:
    """Build compact replay rows from owner-retained action context handles."""

    if not isinstance(root_action_context, CompactRootActionContextV1):
        raise TypeError("root_action_context must be CompactRootActionContextV1")
    record_index_int = _int(record_index, "record_index")
    if record_index_int < 0:
        raise ReplayCompatibilityError("record_index must be nonnegative")
    if not str(policy_source):
        raise ReplayCompatibilityError("policy_source must be a non-empty string")

    batch_size = _int(root_action_context.batch_size, "batch_size")
    player_count = _int(root_action_context.player_count, "player_count")
    root_count = _int(root_action_context.root_count, "root_count")
    if batch_size <= 0 or player_count <= 0 or root_count != batch_size * player_count:
        raise ReplayCompatibilityError("root action context root shape mismatch")

    root_index = np.asarray(root_action_context.active_root_index, dtype=np.int32).reshape(-1)
    action_root_index = np.asarray(action_step.root_index, dtype=np.int32).reshape(-1)
    payload_root_index = np.asarray(replay_payload.root_index, dtype=np.int32).reshape(-1)
    if not np.array_equal(action_root_index, root_index):
        raise ReplayCompatibilityError("action step roots do not match owner context")
    if not np.array_equal(payload_root_index, root_index):
        raise ReplayCompatibilityError("replay payload roots do not match owner context")

    env_row = np.asarray(root_action_context.env_row, dtype=np.int32).reshape(-1)
    player = np.asarray(root_action_context.player, dtype=np.int16).reshape(-1)
    policy_env_id = np.asarray(root_action_context.policy_env_id, dtype=np.int64).reshape(-1)
    for name, expected, actual in (
        ("env_row", env_row, np.asarray(action_step.env_row, dtype=np.int32).reshape(-1)),
        ("player", player, np.asarray(action_step.player, dtype=np.int16).reshape(-1)),
        (
            "policy_env_id",
            policy_env_id,
            np.asarray(action_step.policy_env_id, dtype=np.int64).reshape(-1),
        ),
    ):
        if not np.array_equal(actual, expected):
            raise ReplayCompatibilityError(f"action step {name} does not match owner context")
    for name, expected, actual in (
        ("env_row", env_row, np.asarray(replay_payload.env_row, dtype=np.int32).reshape(-1)),
        ("player", player, np.asarray(replay_payload.player, dtype=np.int16).reshape(-1)),
        (
            "policy_env_id",
            policy_env_id,
            np.asarray(replay_payload.policy_env_id, dtype=np.int64).reshape(-1),
        ),
    ):
        if not np.array_equal(actual, expected):
            raise ReplayCompatibilityError(f"replay payload {name} does not match owner context")
    if str(action_step.replay_payload_handle) != str(replay_payload.replay_payload_handle):
        raise ReplayCompatibilityError("replay payload handle does not match action step")

    selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
    active_count = int(root_index.size)
    if selected.shape != (active_count,):
        raise ReplayCompatibilityError("selected_action count mismatch")
    expected_digest = str(action_step.metadata.get("selected_action_digest", ""))
    if expected_digest and expected_digest != _compact_array_digest_v1(selected):
        raise ReplayCompatibilityError("action step selected-action digest is stale")
    action_mask = _binary_bool_array(
        root_action_context.active_legal_mask,
        "root_action_context.active_legal_mask",
    )
    if action_mask.shape != (active_count, ACTION_COUNT):
        raise ReplayCompatibilityError("root action context legal mask shape mismatch")
    if selected.size and not bool(action_mask[np.arange(selected.size), selected].all()):
        raise ReplayCompatibilityError("selected_action is illegal")

    visit_policy = _float_array(replay_payload.visit_policy, "visit_policy", (active_count, ACTION_COUNT))
    root_value = _float_array(replay_payload.root_value, "root_value", (active_count,))
    joint_action = _int_array(next_joint_action, "next_joint_action", (batch_size, player_count))
    reward_grid = _float_array(next_reward, "next_reward", (batch_size, player_count))
    done_grid = _bool_array(next_done, "next_done", (batch_size,))
    terminated_grid = (
        done_grid
        if next_terminated is None
        else _bool_array(next_terminated, "next_terminated", (batch_size,))
    )
    truncated_grid = (
        np.zeros((batch_size,), dtype=np.bool_)
        if next_truncated is None
        else _bool_array(next_truncated, "next_truncated", (batch_size,))
    )
    if not np.array_equal(done_grid, np.logical_or(terminated_grid, truncated_grid)):
        raise ReplayCompatibilityError("next_done must equal next_terminated | next_truncated")
    final_row_mask = (
        np.zeros((batch_size,), dtype=np.bool_)
        if next_final_observation_row_mask is None
        else _bool_array(
            next_final_observation_row_mask,
            "next_final_observation_row_mask",
            (batch_size,),
        )
    )
    if bool(done_grid.any()) and not bool(final_row_mask[done_grid].all()):
        raise ReplayCompatibilityError(
            "next_done rows require matching next_final_observation_row_mask entries"
        )
    if bool(final_row_mask.any()) and next_final_reward_map is None:
        raise ReplayCompatibilityError("next_final_observation rows require next_final_reward_map")
    final_reward_grid = (
        reward_grid
        if next_final_reward_map is None
        else _float_array(
            next_final_reward_map, "next_final_reward_map", (batch_size, player_count)
        )
    )

    env_row_i64 = env_row.astype(np.int64, copy=False)
    player_i64 = player.astype(np.int64, copy=False)
    if selected.size and not np.array_equal(joint_action[env_row_i64, player_i64], selected):
        raise ReplayCompatibilityError("selected_action does not match next_joint_action")

    policy_row = np.arange(active_count, dtype=np.int32)
    final_reward = reward_grid[env_row_i64, player_i64].astype(np.float32, copy=True)
    final_rows_for_targets = final_row_mask[env_row_i64]
    if bool(final_rows_for_targets.any()):
        final_reward[final_rows_for_targets] = final_reward_grid[
            env_row_i64[final_rows_for_targets],
            player_i64[final_rows_for_targets],
        ].astype(np.float32, copy=False)

    replay_metadata = {
        "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
        "schema_id": "curvyzero_compact_replay_index_rows/v1",
        "record_index": int(record_index_int),
        "next_record_index": int(record_index_int + 1),
        "policy_source": str(policy_source),
        "index_row_count": int(active_count),
        COMPACT_REPLAY_ENV_PLAYER_IDENTITY_DIGEST_KEY: _env_player_identity_digest(
            env_row,
            player,
        ),
        "root_action_context_schema_id": str(root_action_context.schema_id),
        "root_action_context_request_schema_id": str(
            root_action_context.request_schema_id
        ),
        "action_step_schema_id": str(action_step.metadata.get("schema_id", "")),
        "replay_payload_schema_id": str(replay_payload.metadata.get("schema_id", "")),
        "search_impl": str(replay_payload.metadata.get("search_impl", "")),
        "num_simulations": int(replay_payload.metadata.get("num_simulations", 0)),
        "active_root_count": int(active_count),
        "search_replay_payload_digest": str(
            replay_payload.metadata.get("search_replay_payload_digest", "")
        ),
        "owner_action_context_replay_index_rows": True,
        "done_row_count": int(done_grid[env_row_i64].sum()),
        "terminated_row_count": int(terminated_grid[env_row_i64].sum()),
        "truncated_row_count": int(truncated_grid[env_row_i64].sum()),
        "next_final_observation_row_count": int(final_rows_for_targets.sum()),
        "observation_materialized": False,
        "next_observation_materialized": False,
    }
    if COMPACT_REPLAY_TERMINAL_METADATA_ROW_INDICES_ENABLED:
        replay_metadata.update(
            {
                "done_row_indices": np.flatnonzero(done_grid[env_row_i64])
                .astype(np.int64, copy=False)
                .tolist(),
                "terminated_row_indices": np.flatnonzero(terminated_grid[env_row_i64])
                .astype(np.int64, copy=False)
                .tolist(),
                "truncated_row_indices": np.flatnonzero(truncated_grid[env_row_i64])
                .astype(np.int64, copy=False)
                .tolist(),
                "next_final_observation_row_indices": np.flatnonzero(final_rows_for_targets)
                .astype(np.int64, copy=False)
                .tolist(),
            }
        )
    replay_metadata.update(compact_policy_refresh_metadata_subset_v1(replay_payload.metadata))
    if metadata:
        replay_metadata.update(
            {str(key): _plain_metadata_value(value) for key, value in metadata.items()}
        )
    return CompactReplayIndexRowsV1(
        metadata=replay_metadata,
        record_index=record_index_int,
        next_record_index=record_index_int + 1,
        compact_root_row=root_index.astype(np.int32, copy=True),
        policy_env_id=policy_env_id.astype(np.int64, copy=True),
        policy_row=policy_row,
        env_row=env_row.astype(np.int32, copy=True),
        player=player.astype(np.int16, copy=True),
        action=selected.astype(np.int16, copy=True),
        action_mask=action_mask.astype(np.bool_, copy=True),
        policy_target=visit_policy.astype(np.float32, copy=True),
        root_value=root_value.astype(np.float32, copy=True),
        reward=reward_grid[env_row_i64, player_i64].astype(np.float32, copy=True),
        final_reward=final_reward,
        done=done_grid[env_row_i64].astype(np.bool_, copy=True),
        terminated=terminated_grid[env_row_i64].astype(np.bool_, copy=True),
        truncated=truncated_grid[env_row_i64].astype(np.bool_, copy=True),
        next_final_observation_row=final_rows_for_targets.astype(np.bool_, copy=True),
        to_play=np.full((active_count,), DEFAULT_TO_PLAY, dtype=np.int64),
        policy_source=str(policy_source),
    )


def build_compact_device_replay_index_rows_v1_from_owner_action_context_payload(
    root_action_context: CompactRootActionContextV1,
    action_step: Any,
    replay_payload: Any,
    *,
    record_index: int,
    next_joint_action: np.ndarray,
    next_reward: np.ndarray,
    next_done: np.ndarray,
    policy_source: str,
    next_terminated: np.ndarray | None = None,
    next_truncated: np.ndarray | None = None,
    next_final_reward_map: np.ndarray | None = None,
    next_final_observation_row_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
    device: Any | None = None,
) -> CompactDeviceReplayIndexRowsV1:
    """Build owner-context replay rows with tensor targets and scalar sidecars."""

    import torch

    identity_started = time.perf_counter()
    if not isinstance(root_action_context, CompactRootActionContextV1):
        raise TypeError("root_action_context must be CompactRootActionContextV1")
    record_index_int = _int(record_index, "record_index")
    if record_index_int < 0:
        raise ReplayCompatibilityError("record_index must be nonnegative")
    if not str(policy_source):
        raise ReplayCompatibilityError("policy_source must be a non-empty string")
    replay_payload_metadata = dict(getattr(replay_payload, "metadata", {}) or {})
    if (
        replay_payload_metadata.get("schema_id")
        != "curvyzero_compact_device_search_replay_payload/v1"
    ):
        raise ReplayCompatibilityError("device replay payload schema id is invalid")
    if replay_payload_metadata.get("phase") != "replay_critical_device":
        raise ReplayCompatibilityError("device replay payload phase is invalid")
    if replay_payload_metadata.get("device_replay_payload") is not True:
        raise ReplayCompatibilityError("device replay payload marker is missing")
    if replay_payload_metadata.get("host_search_payload_fallback_allowed") is not False:
        raise ReplayCompatibilityError("device replay payload allowed host fallback")

    batch_size = _int(root_action_context.batch_size, "batch_size")
    player_count = _int(root_action_context.player_count, "player_count")
    root_count = _int(root_action_context.root_count, "root_count")
    if batch_size <= 0 or player_count <= 0 or root_count != batch_size * player_count:
        raise ReplayCompatibilityError("root action context root shape mismatch")

    root_index = np.asarray(root_action_context.active_root_index, dtype=np.int32).reshape(-1)
    action_root_index = np.asarray(action_step.root_index, dtype=np.int32).reshape(-1)
    payload_root_index = np.asarray(replay_payload.root_index, dtype=np.int32).reshape(-1)
    if not np.array_equal(action_root_index, root_index):
        raise ReplayCompatibilityError("action step roots do not match owner context")
    if not np.array_equal(payload_root_index, root_index):
        raise ReplayCompatibilityError("device replay payload roots do not match owner context")

    env_row = np.asarray(root_action_context.env_row, dtype=np.int32).reshape(-1)
    player = np.asarray(root_action_context.player, dtype=np.int16).reshape(-1)
    policy_env_id = np.asarray(root_action_context.policy_env_id, dtype=np.int64).reshape(-1)
    for name, expected, actual in (
        ("env_row", env_row, np.asarray(action_step.env_row, dtype=np.int32).reshape(-1)),
        ("player", player, np.asarray(action_step.player, dtype=np.int16).reshape(-1)),
        (
            "policy_env_id",
            policy_env_id,
            np.asarray(action_step.policy_env_id, dtype=np.int64).reshape(-1),
        ),
    ):
        if not np.array_equal(actual, expected):
            raise ReplayCompatibilityError(f"action step {name} does not match owner context")
    for name, expected, actual in (
        ("env_row", env_row, np.asarray(replay_payload.env_row, dtype=np.int32).reshape(-1)),
        ("player", player, np.asarray(replay_payload.player, dtype=np.int16).reshape(-1)),
        (
            "policy_env_id",
            policy_env_id,
            np.asarray(replay_payload.policy_env_id, dtype=np.int64).reshape(-1),
        ),
    ):
        if not np.array_equal(actual, expected):
            raise ReplayCompatibilityError(
                f"device replay payload {name} does not match owner context"
            )
    if str(action_step.replay_payload_handle) != str(replay_payload.replay_payload_handle):
        raise ReplayCompatibilityError("device replay payload handle does not match action step")
    action_digest = str(action_step.metadata.get("search_replay_payload_digest", ""))
    payload_digest = str(replay_payload_metadata.get("search_replay_payload_digest", ""))
    if action_digest and payload_digest and action_digest != payload_digest:
        raise ReplayCompatibilityError("device replay payload digest does not match action step")

    selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
    active_count = int(root_index.size)
    if selected.shape != (active_count,):
        raise ReplayCompatibilityError("selected_action count mismatch")
    expected_digest = str(action_step.metadata.get("selected_action_digest", ""))
    if expected_digest and expected_digest != _compact_array_digest_v1(selected):
        raise ReplayCompatibilityError("action step selected-action digest is stale")
    action_mask = _binary_bool_array(
        root_action_context.active_legal_mask,
        "root_action_context.active_legal_mask",
    )
    if action_mask.shape != (active_count, ACTION_COUNT):
        raise ReplayCompatibilityError("root action context legal mask shape mismatch")
    if selected.size and not bool(action_mask[np.arange(selected.size), selected].all()):
        raise ReplayCompatibilityError("selected_action is illegal")

    def value_shape(value: Any) -> tuple[int, ...]:
        shape = getattr(value, "shape", None)
        if shape is None:
            shape = np.asarray(value).shape
        return tuple(int(dim) for dim in shape)

    if value_shape(replay_payload.visit_policy) != (active_count, ACTION_COUNT):
        raise ReplayCompatibilityError("device replay payload visit_policy shape mismatch")
    if value_shape(replay_payload.root_value) != (active_count,):
        raise ReplayCompatibilityError("device replay payload root_value shape mismatch")

    joint_action = _int_array(next_joint_action, "next_joint_action", (batch_size, player_count))
    reward_grid = _float_array(next_reward, "next_reward", (batch_size, player_count))
    done_grid = _bool_array(next_done, "next_done", (batch_size,))
    terminated_grid = (
        done_grid
        if next_terminated is None
        else _bool_array(next_terminated, "next_terminated", (batch_size,))
    )
    truncated_grid = (
        np.zeros((batch_size,), dtype=np.bool_)
        if next_truncated is None
        else _bool_array(next_truncated, "next_truncated", (batch_size,))
    )
    if not np.array_equal(done_grid, np.logical_or(terminated_grid, truncated_grid)):
        raise ReplayCompatibilityError("next_done must equal next_terminated | next_truncated")
    final_row_mask = (
        np.zeros((batch_size,), dtype=np.bool_)
        if next_final_observation_row_mask is None
        else _bool_array(
            next_final_observation_row_mask,
            "next_final_observation_row_mask",
            (batch_size,),
        )
    )
    if bool(done_grid.any()) and not bool(final_row_mask[done_grid].all()):
        raise ReplayCompatibilityError(
            "next_done rows require matching next_final_observation_row_mask entries"
        )
    if bool(final_row_mask.any()) and next_final_reward_map is None:
        raise ReplayCompatibilityError("next_final_observation rows require next_final_reward_map")
    final_reward_grid = (
        reward_grid
        if next_final_reward_map is None
        else _float_array(
            next_final_reward_map, "next_final_reward_map", (batch_size, player_count)
        )
    )

    env_row_i64 = env_row.astype(np.int64, copy=False)
    player_i64 = player.astype(np.int64, copy=False)
    if selected.size and not np.array_equal(joint_action[env_row_i64, player_i64], selected):
        raise ReplayCompatibilityError("selected_action does not match next_joint_action")
    row_count = int(active_count)
    policy_row = np.arange(row_count, dtype=np.int32)
    reward_values = reward_grid[env_row_i64, player_i64].astype(np.float32, copy=True)
    done_values = done_grid[env_row_i64].astype(np.bool_, copy=True)
    terminated_values = terminated_grid[env_row_i64].astype(np.bool_, copy=True)
    truncated_values = truncated_grid[env_row_i64].astype(np.bool_, copy=True)
    final_reward = reward_values.copy()
    final_rows_for_targets = final_row_mask[env_row_i64]
    if bool(final_rows_for_targets.any()):
        final_reward[final_rows_for_targets] = final_reward_grid[
            env_row_i64[final_rows_for_targets],
            player_i64[final_rows_for_targets],
        ].astype(np.float32, copy=False)
    identity_validate_sec = _elapsed_sec(identity_started)

    target_started = time.perf_counter()
    if device is None:
        if isinstance(replay_payload.visit_policy, torch.Tensor):
            device = replay_payload.visit_policy.device
        elif isinstance(replay_payload.root_value, torch.Tensor):
            device = replay_payload.root_value.device
        else:
            device = torch.device("cpu")

    def as_device_tensor(
        value: Any,
        *,
        dtype: Any,
        shape: tuple[int, ...],
        name: str,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )
        else:
            tensor = torch.as_tensor(value, dtype=dtype, device=device)
        tensor = tensor.contiguous()
        if tuple(int(dim) for dim in tensor.shape) != shape:
            raise ReplayCompatibilityError(f"{name} shape mismatch")
        return tensor

    policy_target = as_device_tensor(
        replay_payload.visit_policy,
        dtype=torch.float32,
        shape=(row_count, ACTION_COUNT),
        name="device replay policy_target",
    )
    root_value = as_device_tensor(
        replay_payload.root_value,
        dtype=torch.float32,
        shape=(row_count,),
        name="device replay root_value",
    )
    target_tensor_sec = _elapsed_sec(target_started)

    scalar_host_started = time.perf_counter()
    scalar_i64_host = np.column_stack(
        (
            policy_env_id.astype(np.int64, copy=False),
            np.full((row_count,), DEFAULT_TO_PLAY, dtype=np.int64),
        )
    ).astype(np.int64, copy=False)
    scalar_i32_host = np.column_stack(
        (
            root_index.astype(np.int32, copy=False),
            policy_row.astype(np.int32, copy=False),
            env_row.astype(np.int32, copy=False),
        )
    ).astype(np.int32, copy=False)
    scalar_i16_host = np.column_stack(
        (
            player.astype(np.int16, copy=False),
            selected.astype(np.int16, copy=False),
        )
    ).astype(np.int16, copy=False)
    scalar_bool_host = np.concatenate(
        (
            action_mask.astype(np.bool_, copy=False),
            done_values.reshape(row_count, 1),
            terminated_values.reshape(row_count, 1),
            truncated_values.reshape(row_count, 1),
            final_rows_for_targets.astype(np.bool_, copy=False).reshape(row_count, 1),
        ),
        axis=1,
    )
    scalar_float_host = np.column_stack((reward_values, final_reward)).astype(
        np.float32,
        copy=False,
    )
    scalar_host_pack_sec = _elapsed_sec(scalar_host_started)

    scalar_transfer_started = time.perf_counter()
    scalar_i64 = torch.as_tensor(scalar_i64_host, dtype=torch.int64, device=device)
    scalar_i32 = torch.as_tensor(scalar_i32_host, dtype=torch.int32, device=device)
    scalar_i16 = torch.as_tensor(scalar_i16_host, dtype=torch.int16, device=device)
    scalar_bool = torch.as_tensor(scalar_bool_host, dtype=torch.bool, device=device)
    scalar_float = torch.as_tensor(scalar_float_host, dtype=torch.float32, device=device)
    scalar_device_transfer_sec = _elapsed_sec(scalar_transfer_started)
    scalar_packed_h2d_bytes = int(
        scalar_i64_host.nbytes
        + scalar_i32_host.nbytes
        + scalar_i16_host.nbytes
        + scalar_bool_host.nbytes
        + scalar_float_host.nbytes
    )

    metadata_started = time.perf_counter()
    replay_metadata = {
        "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
        "schema_id": "curvyzero_compact_device_replay_index_rows/v1",
        "record_index": int(record_index_int),
        "next_record_index": int(record_index_int + 1),
        "policy_source": str(policy_source),
        "index_row_count": int(row_count),
        COMPACT_REPLAY_ENV_PLAYER_IDENTITY_DIGEST_KEY: _env_player_identity_digest(
            env_row,
            player,
        ),
        "root_action_context_schema_id": str(root_action_context.schema_id),
        "root_action_context_request_schema_id": str(
            root_action_context.request_schema_id
        ),
        "action_step_schema_id": str(action_step.metadata.get("schema_id", "")),
        "replay_payload_schema_id": str(replay_payload_metadata.get("schema_id", "")),
        "search_impl": str(replay_payload_metadata.get("search_impl", "")),
        "num_simulations": int(replay_payload_metadata.get("num_simulations", 0)),
        "active_root_count": int(row_count),
        "search_replay_payload_digest": str(
            replay_payload_metadata.get("search_replay_payload_digest", "")
        ),
        "replay_index_rows_builder_variant": "owner_context_device_packed_scalar_v1",
        "owner_action_context_replay_index_rows": True,
        "owner_action_context_device_replay_index_rows": True,
        "device_replay_index_rows": True,
        "device_replay_index_rows_device": str(device),
        "host_search_payload_fallback_allowed": False,
        "done_row_count": int(done_values.sum()),
        "terminated_row_count": int(terminated_values.sum()),
        "truncated_row_count": int(truncated_values.sum()),
        "next_final_observation_row_count": int(final_rows_for_targets.sum()),
        "observation_materialized": False,
        "next_observation_materialized": False,
        "replay_index_rows_identity_validate_sec": float(identity_validate_sec),
        "replay_index_rows_target_tensor_sec": float(target_tensor_sec),
        "replay_index_rows_scalar_host_pack_sec": float(scalar_host_pack_sec),
        "replay_index_rows_scalar_device_transfer_sec": float(scalar_device_transfer_sec),
        "replay_index_rows_scalar_packed_h2d_bytes": scalar_packed_h2d_bytes,
        "replay_index_rows_scalar_tensor_count": 5,
    }
    if COMPACT_REPLAY_TERMINAL_METADATA_ROW_INDICES_ENABLED:
        replay_metadata.update(
            {
                "done_row_indices": np.flatnonzero(done_values)
                .astype(np.int64, copy=False)
                .tolist(),
                "terminated_row_indices": np.flatnonzero(terminated_values)
                .astype(np.int64, copy=False)
                .tolist(),
                "truncated_row_indices": np.flatnonzero(truncated_values)
                .astype(np.int64, copy=False)
                .tolist(),
                "next_final_observation_row_indices": np.flatnonzero(final_rows_for_targets)
                .astype(np.int64, copy=False)
                .tolist(),
            }
        )
    replay_metadata.update(compact_policy_refresh_metadata_subset_v1(action_step.metadata))
    replay_metadata.update(compact_policy_refresh_metadata_subset_v1(replay_payload_metadata))
    if metadata:
        replay_metadata.update(
            {str(key): _plain_metadata_value(value) for key, value in metadata.items()}
        )
    replay_metadata["replay_index_rows_metadata_sec"] = float(_elapsed_sec(metadata_started))
    return CompactDeviceReplayIndexRowsV1(
        metadata=replay_metadata,
        record_index=record_index_int,
        next_record_index=record_index_int + 1,
        compact_root_row=scalar_i32[:, 0].contiguous(),
        policy_env_id=scalar_i64[:, 0].contiguous(),
        policy_row=scalar_i32[:, 1].contiguous(),
        env_row=scalar_i32[:, 2].contiguous(),
        player=scalar_i16[:, 0].contiguous(),
        action=scalar_i16[:, 1].contiguous(),
        action_mask=scalar_bool[:, :ACTION_COUNT].contiguous(),
        policy_target=policy_target,
        root_value=root_value,
        reward=scalar_float[:, 0].contiguous(),
        final_reward=scalar_float[:, 1].contiguous(),
        done=scalar_bool[:, ACTION_COUNT].contiguous(),
        terminated=scalar_bool[:, ACTION_COUNT + 1].contiguous(),
        truncated=scalar_bool[:, ACTION_COUNT + 2].contiguous(),
        next_final_observation_row=scalar_bool[:, ACTION_COUNT + 3].contiguous(),
        to_play=scalar_i64[:, 1].contiguous(),
        policy_source=str(policy_source),
    )


def build_compact_device_replay_index_rows_v1_from_payload(
    batch: HybridCompactBatch,
    root_batch: CompactRootBatchV1,
    action_step: Any,
    replay_payload: Any,
    *,
    record_index: int,
    next_joint_action: np.ndarray,
    next_reward: np.ndarray,
    next_done: np.ndarray,
    policy_source: str,
    next_terminated: np.ndarray | None = None,
    next_truncated: np.ndarray | None = None,
    next_final_reward_map: np.ndarray | None = None,
    next_final_observation_row_mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CompactDeviceReplayIndexRowsV1:
    """Build replay rows while keeping search targets on the search device."""

    import torch

    identity_started = time.perf_counter()
    record_index_int = _int(record_index, "record_index")
    if record_index_int < 0:
        raise ReplayCompatibilityError("record_index must be nonnegative")
    if not str(policy_source):
        raise ReplayCompatibilityError("policy_source must be a non-empty string")
    if replay_payload.metadata.get("device_replay_payload") is not True:
        raise ReplayCompatibilityError("device replay index rows require device payload")
    if replay_payload.metadata.get("host_search_payload_fallback_allowed") is not False:
        raise ReplayCompatibilityError("device replay payload allowed host fallback")

    legal_mask = _binary_bool_array(root_batch.legal_mask, "legal_mask")
    root_count = int(legal_mask.shape[0])
    if legal_mask.shape != (root_count, ACTION_COUNT):
        raise ReplayCompatibilityError("legal_mask must have shape [root,3]")
    root_index = np.asarray(replay_payload.root_index, dtype=np.int32)
    if root_index.shape != tuple(int(dim) for dim in replay_payload.visit_policy.shape[:1]):
        raise ReplayCompatibilityError("device replay payload root_index count mismatch")
    if not np.array_equal(root_index, np.asarray(action_step.root_index, dtype=np.int32)):
        raise ReplayCompatibilityError("device replay payload root_index does not match action")
    action_mask = legal_mask[root_index].astype(bool, copy=True)
    selected = np.asarray(action_step.selected_action, dtype=np.int16)
    if selected.shape != (int(root_index.size),):
        raise ReplayCompatibilityError("selected_action count mismatch")
    if selected.size and not bool(action_mask[np.arange(selected.size), selected].all()):
        raise ReplayCompatibilityError("selected_action is illegal")

    observation = np.asarray(batch.observation)
    if observation.ndim != 5:
        raise ReplayCompatibilityError("compact batch observation must have shape [B,P,C,H,W]")
    batch_size, player_count = int(observation.shape[0]), int(observation.shape[1])
    expected_root_count = batch_size * player_count
    if root_count != expected_root_count:
        raise ReplayCompatibilityError("root_batch root count does not match compact batch")
    expected_legal_mask = _binary_bool_array(batch.action_mask, "batch.action_mask").reshape(
        expected_root_count,
        ACTION_COUNT,
    )
    if not np.array_equal(root_batch.legal_mask, expected_legal_mask):
        raise ReplayCompatibilityError("root_batch legal_mask does not match compact batch")
    expected_policy_env_id = _int_array(
        batch.policy_env_id,
        "batch.policy_env_id",
        (expected_root_count,),
    )
    expected_env_row = _int_array(
        batch.policy_env_row,
        "batch.policy_env_row",
        (expected_root_count,),
    )
    expected_player = _int_array(
        batch.policy_player,
        "batch.policy_player",
        (expected_root_count,),
    )
    if not np.array_equal(root_batch.policy_env_id, expected_policy_env_id):
        raise ReplayCompatibilityError("root_batch policy_env_id does not match compact batch")
    if not np.array_equal(root_batch.env_row, expected_env_row):
        raise ReplayCompatibilityError("root_batch env_row does not match compact batch")
    if not np.array_equal(root_batch.player, expected_player):
        raise ReplayCompatibilityError("root_batch player does not match compact batch")

    env_row = np.asarray(replay_payload.env_row, dtype=np.int32)
    player = np.asarray(replay_payload.player, dtype=np.int16)
    policy_env_id = np.asarray(replay_payload.policy_env_id, dtype=np.int64)
    if not np.array_equal(env_row, np.asarray(action_step.env_row, dtype=np.int32)):
        raise ReplayCompatibilityError("device replay env_row does not match action step")
    if not np.array_equal(player, np.asarray(action_step.player, dtype=np.int16)):
        raise ReplayCompatibilityError("device replay player does not match action step")
    if not np.array_equal(
        policy_env_id,
        np.asarray(action_step.policy_env_id, dtype=np.int64),
    ):
        raise ReplayCompatibilityError("device replay policy_env_id does not match action step")
    identity_validate_sec = _elapsed_sec(identity_started)

    terminal_started = time.perf_counter()
    joint_action = _int_array(next_joint_action, "next_joint_action", (batch_size, player_count))
    reward_grid = _float_array(next_reward, "next_reward", (batch_size, player_count))
    done_grid = _bool_array(next_done, "next_done", (batch_size,))
    terminated_grid = (
        done_grid
        if next_terminated is None
        else _bool_array(next_terminated, "next_terminated", (batch_size,))
    )
    truncated_grid = (
        np.zeros((batch_size,), dtype=np.bool_)
        if next_truncated is None
        else _bool_array(next_truncated, "next_truncated", (batch_size,))
    )
    if not np.array_equal(done_grid, np.logical_or(terminated_grid, truncated_grid)):
        raise ReplayCompatibilityError("next_done must equal next_terminated | next_truncated")
    final_row_mask = (
        np.zeros((batch_size,), dtype=np.bool_)
        if next_final_observation_row_mask is None
        else _bool_array(
            next_final_observation_row_mask,
            "next_final_observation_row_mask",
            (batch_size,),
        )
    )
    if bool(done_grid.any()) and not bool(final_row_mask[done_grid].all()):
        raise ReplayCompatibilityError(
            "next_done rows require matching next_final_observation_row_mask entries"
        )
    if bool(final_row_mask.any()) and next_final_reward_map is None:
        raise ReplayCompatibilityError("next_final_observation rows require next_final_reward_map")
    final_reward_grid = (
        reward_grid
        if next_final_reward_map is None
        else _float_array(
            next_final_reward_map, "next_final_reward_map", (batch_size, player_count)
        )
    )

    env_row_i64 = env_row.astype(np.int64, copy=False)
    player_i64 = player.astype(np.int64, copy=False)
    if selected.size and not np.array_equal(joint_action[env_row_i64, player_i64], selected):
        raise ReplayCompatibilityError("selected_action does not match next_joint_action")
    row_count = int(root_index.size)
    policy_row = np.arange(row_count, dtype=np.int64)
    reward_values = reward_grid[env_row_i64, player_i64].astype(np.float32, copy=True)
    done_values = done_grid[env_row_i64].astype(np.bool_, copy=True)
    terminated_values = terminated_grid[env_row_i64].astype(np.bool_, copy=True)
    truncated_values = truncated_grid[env_row_i64].astype(np.bool_, copy=True)
    final_reward = reward_values.copy()
    final_rows_for_targets = final_row_mask[env_row_i64]
    if bool(final_rows_for_targets.any()):
        final_reward[final_rows_for_targets] = final_reward_grid[
            env_row_i64[final_rows_for_targets],
            player_i64[final_rows_for_targets],
        ].astype(np.float32, copy=False)
    terminal_prepare_sec = _elapsed_sec(terminal_started)

    target_started = time.perf_counter()
    policy_target = replay_payload.visit_policy.detach().to(dtype=torch.float32).contiguous()
    root_value = (
        replay_payload.root_value.detach()
        .to(
            device=policy_target.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        .contiguous()
    )
    if tuple(int(dim) for dim in policy_target.shape) != (int(root_index.size), ACTION_COUNT):
        raise ReplayCompatibilityError("device replay policy_target shape mismatch")
    if tuple(int(dim) for dim in root_value.shape) != (int(root_index.size),):
        raise ReplayCompatibilityError("device replay root_value shape mismatch")
    device = policy_target.device
    target_tensor_sec = _elapsed_sec(target_started)

    scalar_host_started = time.perf_counter()
    scalar_i64_host = np.column_stack(
        (
            policy_env_id.astype(np.int64, copy=False),
            np.full((row_count,), DEFAULT_TO_PLAY, dtype=np.int64),
        )
    ).astype(np.int64, copy=False)
    scalar_i32_host = np.column_stack(
        (
            root_index.astype(np.int32, copy=False),
            policy_row.astype(np.int32, copy=False),
            env_row.astype(np.int32, copy=False),
        )
    ).astype(np.int32, copy=False)
    scalar_i16_host = np.column_stack(
        (
            player.astype(np.int16, copy=False),
            selected.astype(np.int16, copy=False),
        )
    ).astype(np.int16, copy=False)
    scalar_bool_host = np.concatenate(
        (
            action_mask.astype(np.bool_, copy=False),
            done_values.reshape(row_count, 1),
            terminated_values.reshape(row_count, 1),
            truncated_values.reshape(row_count, 1),
            final_rows_for_targets.astype(np.bool_, copy=False).reshape(row_count, 1),
        ),
        axis=1,
    )
    scalar_float_host = np.column_stack((reward_values, final_reward)).astype(
        np.float32,
        copy=False,
    )
    scalar_host_pack_sec = _elapsed_sec(scalar_host_started)

    scalar_transfer_started = time.perf_counter()
    scalar_i64 = torch.as_tensor(
        scalar_i64_host,
        dtype=torch.int64,
        device=device,
    )
    scalar_i32 = torch.as_tensor(
        scalar_i32_host,
        dtype=torch.int32,
        device=device,
    )
    scalar_i16 = torch.as_tensor(
        scalar_i16_host,
        dtype=torch.int16,
        device=device,
    )
    scalar_bool = torch.as_tensor(
        scalar_bool_host,
        dtype=torch.bool,
        device=device,
    )
    scalar_float = torch.as_tensor(
        scalar_float_host,
        dtype=torch.float32,
        device=device,
    )
    compact_root_row_tensor = scalar_i32[:, 0].contiguous()
    policy_env_id_tensor = scalar_i64[:, 0].contiguous()
    policy_row_tensor = scalar_i32[:, 1].contiguous()
    env_row_tensor = scalar_i32[:, 2].contiguous()
    player_tensor = scalar_i16[:, 0].contiguous()
    action_tensor = scalar_i16[:, 1].contiguous()
    to_play_tensor = scalar_i64[:, 1].contiguous()
    action_mask_tensor = scalar_bool[:, :ACTION_COUNT].contiguous()
    done_tensor = scalar_bool[:, ACTION_COUNT].contiguous()
    terminated_tensor = scalar_bool[:, ACTION_COUNT + 1].contiguous()
    truncated_tensor = scalar_bool[:, ACTION_COUNT + 2].contiguous()
    next_final_observation_row_tensor = scalar_bool[:, ACTION_COUNT + 3].contiguous()
    reward_tensor = scalar_float[:, 0].contiguous()
    final_reward_tensor = scalar_float[:, 1].contiguous()
    scalar_device_transfer_sec = _elapsed_sec(scalar_transfer_started)
    scalar_packed_h2d_bytes = int(
        scalar_i64_host.nbytes
        + scalar_i32_host.nbytes
        + scalar_i16_host.nbytes
        + scalar_bool_host.nbytes
        + scalar_float_host.nbytes
    )

    metadata_started = time.perf_counter()
    replay_metadata = {
        "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
        "schema_id": "curvyzero_compact_device_replay_index_rows/v1",
        "record_index": int(record_index_int),
        "next_record_index": int(record_index_int + 1),
        "policy_source": str(policy_source),
        "index_row_count": int(root_index.size),
        COMPACT_REPLAY_ENV_PLAYER_IDENTITY_DIGEST_KEY: _env_player_identity_digest(
            env_row,
            player,
        ),
        "root_batch_schema_id": str(root_batch.metadata.get("schema_id")),
        "replay_payload_schema_id": str(replay_payload.metadata.get("schema_id")),
        "search_result_schema_id": str(replay_payload.metadata.get("search_result_schema_id", "")),
        "search_impl": str(replay_payload.metadata.get("search_impl", "")),
        "num_simulations": int(replay_payload.metadata.get("num_simulations", 0)),
        "active_root_count": int(replay_payload.metadata.get("active_root_count", root_index.size)),
        "search_replay_payload_digest": str(
            replay_payload.metadata.get("search_replay_payload_digest", "")
        ),
        "replay_index_rows_builder_variant": "device_packed_scalar_v1",
        "device_replay_index_rows": True,
        "device_replay_index_rows_device": str(device),
        "host_search_payload_fallback_allowed": False,
        "done_row_count": int(done_values.sum()),
        "terminated_row_count": int(terminated_values.sum()),
        "truncated_row_count": int(truncated_values.sum()),
        "next_final_observation_row_count": int(final_rows_for_targets.sum()),
        "observation_materialized": False,
        "next_observation_materialized": False,
        "replay_index_rows_identity_validate_sec": float(identity_validate_sec),
        "replay_index_rows_terminal_prepare_sec": float(terminal_prepare_sec),
        "replay_index_rows_target_tensor_sec": float(target_tensor_sec),
        "replay_index_rows_scalar_host_pack_sec": float(scalar_host_pack_sec),
        "replay_index_rows_scalar_device_transfer_sec": float(scalar_device_transfer_sec),
        "replay_index_rows_scalar_packed_h2d_bytes": scalar_packed_h2d_bytes,
        "replay_index_rows_scalar_tensor_count": 5,
    }
    if COMPACT_REPLAY_TERMINAL_METADATA_ROW_INDICES_ENABLED:
        replay_metadata.update(
            {
                "done_row_indices": np.flatnonzero(done_values)
                .astype(np.int64, copy=False)
                .tolist(),
                "terminated_row_indices": np.flatnonzero(terminated_values)
                .astype(np.int64, copy=False)
                .tolist(),
                "truncated_row_indices": np.flatnonzero(truncated_values)
                .astype(np.int64, copy=False)
                .tolist(),
                "next_final_observation_row_indices": np.flatnonzero(final_rows_for_targets)
                .astype(np.int64, copy=False)
                .tolist(),
            }
        )
    policy_refresh_metadata = compact_policy_refresh_metadata_subset_v1(action_step.metadata)
    policy_refresh_metadata.update(
        compact_policy_refresh_metadata_subset_v1(replay_payload.metadata)
    )
    replay_metadata.update(policy_refresh_metadata)
    if metadata:
        replay_metadata.update(
            {str(key): _plain_metadata_value(value) for key, value in metadata.items()}
        )
    replay_metadata["replay_index_rows_metadata_sec"] = float(_elapsed_sec(metadata_started))
    return CompactDeviceReplayIndexRowsV1(
        metadata=replay_metadata,
        record_index=record_index_int,
        next_record_index=record_index_int + 1,
        compact_root_row=compact_root_row_tensor,
        policy_env_id=policy_env_id_tensor,
        policy_row=policy_row_tensor,
        env_row=env_row_tensor,
        player=player_tensor,
        action=action_tensor,
        action_mask=action_mask_tensor,
        policy_target=policy_target,
        root_value=root_value,
        reward=reward_tensor,
        final_reward=final_reward_tensor,
        done=done_tensor,
        terminated=terminated_tensor,
        truncated=truncated_tensor,
        next_final_observation_row=next_final_observation_row_tensor,
        to_play=to_play_tensor,
        policy_source=str(policy_source),
    )


def validate_compact_search_result_identity_v1(
    root_batch: CompactRootBatchV1,
    search_result: CompactSearchResultV1,
) -> None:
    """Validate search-result identity before its actions feed the env."""

    if not np.array_equal(
        search_result.root_index,
        np.flatnonzero(root_batch.active_root_mask).astype(np.int32, copy=False),
    ):
        raise ReplayCompatibilityError("search_result roots must match root_batch active roots")
    if not np.array_equal(search_result.env_row, root_batch.env_row[search_result.root_index]):
        raise ReplayCompatibilityError("search_result env_row does not match root_batch")
    if not np.array_equal(search_result.player, root_batch.player[search_result.root_index]):
        raise ReplayCompatibilityError("search_result player does not match root_batch")
    if not np.array_equal(
        search_result.policy_env_id,
        root_batch.policy_env_id[search_result.root_index],
    ):
        raise ReplayCompatibilityError("search_result policy_env_id does not match root_batch")


def materialize_compact_target_rows_from_index_rows_v1(
    chunk: Any,
    index_rows: CompactReplayIndexRowsV1,
) -> SourceStateMultiplayerTargetRowsV0:
    """Materialize checked target rows from index-only compact replay rows.

    This is a validation/sampler edge, not the collection hot path. It proves
    ``CompactReplayIndexRowsV1`` kept enough information to rebuild the same
    learner-shaped rows without copying observations during collection.
    """

    arrays = _validate_chunk(chunk)
    observation = arrays["observation"]
    time_steps, batch_size, player_count = observation.shape[:3]
    record_index_int = int(index_rows.record_index)
    next_record_index = int(index_rows.next_record_index)
    if record_index_int < 0 or next_record_index != record_index_int + 1:
        raise ReplayCompatibilityError("compact index rows must reference adjacent records")
    if next_record_index >= time_steps:
        raise ReplayCompatibilityError("compact index rows require a following result record")
    if len(chunk.policy_rows) <= record_index_int:
        raise ReplayCompatibilityError("compact index rows record_index is missing policy rows")

    policy = chunk.policy_rows[record_index_int]
    expected_policy_count = _validate_policy_row_arrays(
        policy,
        batch_size=batch_size,
        player_count=player_count,
    )
    row_count = int(np.asarray(index_rows.action).size)
    fields = {
        "compact_root_row": np.asarray(index_rows.compact_root_row, dtype=np.int32),
        "policy_env_id": np.asarray(index_rows.policy_env_id, dtype=np.int64),
        "policy_row": np.asarray(index_rows.policy_row, dtype=np.int32),
        "env_row": np.asarray(index_rows.env_row, dtype=np.int32),
        "player": np.asarray(index_rows.player, dtype=np.int16),
        "action": np.asarray(index_rows.action, dtype=np.int16),
        "action_mask": np.asarray(index_rows.action_mask, dtype=bool),
        "policy_target": np.asarray(index_rows.policy_target, dtype=np.float32),
        "root_value": np.asarray(index_rows.root_value, dtype=np.float32),
        "reward": np.asarray(index_rows.reward, dtype=np.float32),
        "final_reward": np.asarray(index_rows.final_reward, dtype=np.float32),
        "done": np.asarray(index_rows.done, dtype=np.bool_),
        "terminated": np.asarray(index_rows.terminated, dtype=np.bool_),
        "truncated": np.asarray(index_rows.truncated, dtype=np.bool_),
        "next_final_observation_row": np.asarray(
            index_rows.next_final_observation_row,
            dtype=np.bool_,
        ),
        "to_play": np.asarray(index_rows.to_play, dtype=np.int64),
    }
    for name, value in fields.items():
        expected_shape = (
            (row_count, ACTION_COUNT) if name in {"action_mask", "policy_target"} else (row_count,)
        )
        if value.shape != expected_shape:
            raise ReplayCompatibilityError(f"{name} shape does not match compact index rows")
    if bool((fields["to_play"] != DEFAULT_TO_PLAY).any()):
        raise ReplayCompatibilityError("compact index rows require to_play=-1")
    if not np.array_equal(fields["done"], np.logical_or(fields["terminated"], fields["truncated"])):
        raise ReplayCompatibilityError("compact index done must equal terminated | truncated")
    if bool(fields["done"].any()) and not bool(
        fields["next_final_observation_row"][fields["done"]].all()
    ):
        raise ReplayCompatibilityError("terminal compact index rows require final observations")

    rows: list[dict[str, Any]] = []
    for output_row in range(row_count):
        compact_root_row = int(fields["compact_root_row"][output_row])
        policy_env_id = int(fields["policy_env_id"][output_row])
        env_row = int(fields["env_row"][output_row])
        player = int(fields["player"][output_row])
        policy_row = int(fields["policy_row"][output_row])
        action = int(fields["action"][output_row])
        if env_row < 0 or env_row >= batch_size:
            raise ReplayCompatibilityError("compact index env_row is out of range")
        if player < 0 or player >= player_count:
            raise ReplayCompatibilityError("compact index player is out of range")
        if policy_row < 0 or policy_row >= expected_policy_count:
            raise ReplayCompatibilityError("compact index policy_row is out of range")
        if int(policy["policy_env_row"][policy_row]) != env_row:
            raise ReplayCompatibilityError("compact index env_row does not match replay map")
        if int(policy["policy_player"][policy_row]) != player:
            raise ReplayCompatibilityError("compact index player does not match replay map")
        if not bool(arrays["live_mask"][record_index_int, env_row, player]):
            raise ReplayCompatibilityError("compact index row must point to a live decision seat")
        action_mask = fields["action_mask"][output_row].astype(bool, copy=True)
        if not np.array_equal(action_mask, policy["policy_action_mask"][policy_row]):
            raise ReplayCompatibilityError("compact index action_mask does not match replay")
        if not np.array_equal(
            action_mask,
            arrays["legal_action_mask"][record_index_int, env_row, player],
        ):
            raise ReplayCompatibilityError("compact index action_mask does not match legal mask")
        if int(arrays["joint_action"][next_record_index, env_row, player]) != action:
            raise ReplayCompatibilityError(
                "compact index action does not match next replay joint_action"
            )

        expected_reward = np.float32(arrays["reward"][next_record_index, env_row, player])
        final_row = bool(arrays["final_observation_row_mask"][next_record_index, env_row])
        expected_final_reward = (
            np.float32(arrays["final_reward_map"][next_record_index, env_row, player])
            if final_row
            else expected_reward
        )
        if not np.isclose(fields["reward"][output_row], expected_reward, atol=1e-6):
            raise ReplayCompatibilityError("compact index reward does not match replay")
        if not np.isclose(fields["final_reward"][output_row], expected_final_reward, atol=1e-6):
            raise ReplayCompatibilityError("compact index final_reward does not match replay")
        if bool(fields["done"][output_row]) != bool(arrays["done"][next_record_index, env_row]):
            raise ReplayCompatibilityError("compact index done does not match replay")
        if bool(fields["terminated"][output_row]) != bool(
            arrays["terminated"][next_record_index, env_row]
        ):
            raise ReplayCompatibilityError("compact index terminated does not match replay")
        if bool(fields["truncated"][output_row]) != bool(
            arrays["truncated"][next_record_index, env_row]
        ):
            raise ReplayCompatibilityError("compact index truncated does not match replay")
        if bool(fields["next_final_observation_row"][output_row]) != final_row:
            raise ReplayCompatibilityError(
                "compact index final-observation marker does not match replay"
            )
        next_observation = (
            arrays["final_observation"][next_record_index, env_row, player]
            if final_row
            else observation[next_record_index, env_row, player]
        )
        rows.append(
            {
                "observation": observation[record_index_int, env_row, player].copy(),
                "action": np.int16(action),
                "action_mask": action_mask,
                "policy_target": fields["policy_target"][output_row].copy(),
                "root_value": np.float32(fields["root_value"][output_row]),
                "reward": np.float32(fields["reward"][output_row]),
                "final_reward": np.float32(fields["final_reward"][output_row]),
                "done": bool(fields["done"][output_row]),
                "terminated": bool(fields["terminated"][output_row]),
                "truncated": bool(fields["truncated"][output_row]),
                "next_observation": np.asarray(next_observation, dtype=np.float32).copy(),
                "to_play": np.int64(DEFAULT_TO_PLAY),
                "env_row": np.int32(env_row),
                "player": np.int16(player),
                "record_index": np.int32(record_index_int),
                "next_record_index": np.int32(next_record_index),
                "policy_row": np.int32(policy_row),
                "policy_source": str(index_rows.policy_source),
                "source_record_ref": {
                    "contract_id": COMPACT_POLICY_ROW_RECORDS_CONTRACT_ID,
                    "policy_env_id": policy_env_id,
                    "policy_row": policy_row,
                    "compact_root_row": compact_root_row,
                    "active_output_row": output_row,
                },
            }
        )

    return _target_rows_from_dicts(
        rows,
        metadata=_metadata(chunk=chunk, target_row_count=len(rows)),
    )


def materialize_compact_target_rows_from_index_row_groups_v1(
    chunk: Any,
    index_row_groups: Sequence[CompactReplayIndexRowsV1],
) -> SourceStateMultiplayerTargetRowsV0:
    """Materialize learner rows from multiple compact replay-index groups."""

    groups = tuple(index_row_groups)
    if not groups:
        return _target_rows_from_dicts(
            [],
            metadata=_compact_index_group_metadata(
                chunk,
                group_count=0,
                target_row_count=0,
            ),
        )
    materialized = tuple(
        materialize_compact_target_rows_from_index_rows_v1(chunk, group) for group in groups
    )
    target_row_count = int(sum(rows.action.size for rows in materialized))
    metadata = _compact_index_group_metadata(
        chunk,
        group_count=len(groups),
        target_row_count=target_row_count,
    )
    metadata["source_record_pairs"] = [
        [int(group.record_index), int(group.next_record_index)] for group in groups
    ]
    return SourceStateMultiplayerTargetRowsV0(
        metadata=metadata,
        observation=_concat_target_arrays(materialized, "observation", np.float32),
        action=_concat_target_arrays(materialized, "action", np.int16),
        action_mask=_concat_target_arrays(materialized, "action_mask", bool),
        policy_target=_concat_target_arrays(materialized, "policy_target", np.float32),
        root_value=_concat_target_arrays(materialized, "root_value", np.float32),
        reward=_concat_target_arrays(materialized, "reward", np.float32),
        final_reward=_concat_target_arrays(materialized, "final_reward", np.float32),
        done=_concat_target_arrays(materialized, "done", bool),
        terminated=_concat_target_arrays(materialized, "terminated", bool),
        truncated=_concat_target_arrays(materialized, "truncated", bool),
        next_observation=_concat_target_arrays(materialized, "next_observation", np.float32),
        to_play=_concat_target_arrays(materialized, "to_play", np.int64),
        env_row=_concat_target_arrays(materialized, "env_row", np.int32),
        player=_concat_target_arrays(materialized, "player", np.int16),
        record_index=_concat_target_arrays(materialized, "record_index", np.int32),
        next_record_index=_concat_target_arrays(materialized, "next_record_index", np.int32),
        policy_row=_concat_target_arrays(materialized, "policy_row", np.int32),
        policy_source=tuple(
            policy_source for rows in materialized for policy_source in rows.policy_source
        ),
        source_record_ref=tuple(
            source_ref for rows in materialized for source_ref in rows.source_record_ref
        ),
    )


def sample_compact_target_rows_from_index_row_groups_v1(
    chunk: Any,
    index_row_groups: Sequence[CompactReplayIndexRowsV1],
    *,
    batch_size: int,
    seed: int = 0,
    replace: bool = False,
) -> SourceStateMultiplayerSampleBatchV0:
    """Sample from multiple compact replay-index groups at the sampler edge."""

    target_rows = materialize_compact_target_rows_from_index_row_groups_v1(
        chunk,
        index_row_groups,
    )
    return build_source_state_multiplayer_sample_batch_v0(
        target_rows,
        batch_size=batch_size,
        seed=seed,
        replace=replace,
    )


def build_policy_row_records_from_compact_search_v0(
    batch: HybridCompactBatch,
    *,
    selected_action: np.ndarray | Sequence[int],
    visit_policy: np.ndarray | Sequence[Sequence[float]],
    root_value: np.ndarray | Sequence[float],
    record_index: int,
    policy_source: str,
) -> list[PolicyRowRecordV0]:
    """Build checked policy records from compact row/player search output.

    This proves the compact sidecar can feed the repo-owned target-row bridge.
    It does not claim native LightZero replay, learner updates, or a full
    trainer integration.
    """

    arrays = validate_compact_policy_search_arrays_v0(
        batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=record_index,
        policy_source=policy_source,
    )

    records: list[PolicyRowRecordV0] = []
    policy_env_id = np.asarray(batch.policy_env_id, dtype=np.int64)
    replay_policy_row_by_root = _replay_policy_row_by_compact_root(batch)
    for output_row, compact_root_row in enumerate(arrays.policy_row.astype(np.int64, copy=False)):
        policy_row = replay_policy_row_by_root.get(int(compact_root_row))
        if policy_row is None:
            raise ReplayCompatibilityError("compact search root row is missing replay policy row")
        records.append(
            PolicyRowRecordV0(
                record_index=int(arrays.record_index),
                policy_row=int(policy_row),
                env_row=int(arrays.env_row[output_row]),
                player=int(arrays.player[output_row]),
                action=int(arrays.action[output_row]),
                action_mask=arrays.action_mask[output_row].copy(),
                policy_target=arrays.policy_target[output_row].copy(),
                root_value=float(arrays.root_value[output_row]),
                policy_source=str(arrays.policy_source),
                source_record_ref={
                    "contract_id": COMPACT_POLICY_ROW_RECORDS_CONTRACT_ID,
                    "policy_env_id": int(policy_env_id[int(compact_root_row)]),
                    "policy_row": int(policy_row),
                    "compact_root_row": int(compact_root_row),
                    "active_output_row": int(output_row),
                },
            )
        )
    return records


def build_compact_target_rows_from_search_arrays_v0(
    chunk: Any,
    batch: HybridCompactBatch,
    *,
    selected_action: np.ndarray | Sequence[int],
    visit_policy: np.ndarray | Sequence[Sequence[float]],
    root_value: np.ndarray | Sequence[float],
    record_index: int,
    policy_source: str,
) -> SourceStateMultiplayerTargetRowsV0:
    """Build target rows directly from compact search arrays.

    This is the first local proof for the future compact search/replay lane:
    it avoids allocating one ``PolicyRowRecordV0`` per root while preserving the
    same target-row semantics as the current object bridge.
    """

    search = validate_compact_policy_search_arrays_v0(
        batch,
        selected_action=selected_action,
        visit_policy=visit_policy,
        root_value=root_value,
        record_index=record_index,
        policy_source=policy_source,
    )
    arrays = _validate_chunk(chunk)
    observation = arrays["observation"]
    time_steps, batch_size, player_count = observation.shape[:3]
    record_index_int = int(search.record_index)
    next_record_index = record_index_int + 1
    if record_index_int < 0 or next_record_index >= time_steps:
        raise ReplayCompatibilityError(
            "compact search record_index must have a following result record"
        )
    if len(chunk.policy_rows) <= record_index_int:
        raise ReplayCompatibilityError("compact search record_index is missing policy rows")

    policy = chunk.policy_rows[record_index_int]
    expected_policy_count = _validate_policy_row_arrays(
        policy,
        batch_size=batch_size,
        player_count=player_count,
    )
    _validate_compact_batch_matches_chunk_record(
        batch=batch,
        arrays=arrays,
        record_index=record_index_int,
    )
    replay_policy_row_by_seat = _replay_policy_row_by_seat(policy)
    rows: list[dict[str, Any]] = []
    policy_env_id = np.asarray(batch.policy_env_id, dtype=np.int64)
    for output_row in range(int(search.policy_row.size)):
        compact_root_row = int(search.policy_row[output_row])
        env_row = int(search.env_row[output_row])
        player = int(search.player[output_row])
        action = int(search.action[output_row])
        policy_row = replay_policy_row_by_seat.get((env_row, player))
        if policy_row is None:
            raise ReplayCompatibilityError("compact search row is missing replay policy row")
        if policy_row < 0 or policy_row >= expected_policy_count:
            raise ReplayCompatibilityError("compact replay policy_row is out of range")
        if compact_root_row < 0 or compact_root_row >= int(policy_env_id.size):
            raise ReplayCompatibilityError("compact search root row is out of range")
        if env_row < 0 or env_row >= batch_size:
            raise ReplayCompatibilityError("compact search env_row is out of range")
        if player < 0 or player >= player_count:
            raise ReplayCompatibilityError("compact search player is out of range")
        if int(policy["policy_env_row"][policy_row]) != env_row:
            raise ReplayCompatibilityError("compact search env_row does not match replay map")
        if int(policy["policy_player"][policy_row]) != player:
            raise ReplayCompatibilityError("compact search player does not match replay map")
        if not bool(arrays["live_mask"][record_index_int, env_row, player]):
            raise ReplayCompatibilityError("compact search row must point to a live decision seat")

        action_mask = search.action_mask[output_row].astype(bool, copy=True)
        expected_action_mask = np.asarray(policy["policy_action_mask"][policy_row], dtype=bool)
        if not np.array_equal(action_mask, expected_action_mask):
            raise ReplayCompatibilityError("compact search action_mask does not match replay")
        if not np.array_equal(
            action_mask,
            arrays["legal_action_mask"][record_index_int, env_row, player],
        ):
            raise ReplayCompatibilityError("compact search action_mask does not match legal mask")
        if int(arrays["joint_action"][next_record_index, env_row, player]) != action:
            raise ReplayCompatibilityError(
                "compact search action does not match next replay joint_action"
            )

        final_row = bool(arrays["final_observation_row_mask"][next_record_index, env_row])
        if final_row:
            next_observation = arrays["final_observation"][next_record_index, env_row, player]
            final_reward = arrays["final_reward_map"][next_record_index, env_row, player]
        else:
            next_observation = observation[next_record_index, env_row, player]
            final_reward = arrays["reward"][next_record_index, env_row, player]

        rows.append(
            {
                "observation": observation[record_index_int, env_row, player].copy(),
                "action": np.int16(action),
                "action_mask": action_mask,
                "policy_target": search.policy_target[output_row].copy(),
                "root_value": np.float32(search.root_value[output_row]),
                "reward": np.float32(arrays["reward"][next_record_index, env_row, player]),
                "final_reward": np.float32(final_reward),
                "done": bool(arrays["done"][next_record_index, env_row]),
                "terminated": bool(arrays["terminated"][next_record_index, env_row]),
                "truncated": bool(arrays["truncated"][next_record_index, env_row]),
                "next_observation": np.asarray(next_observation, dtype=np.float32).copy(),
                "to_play": np.int64(DEFAULT_TO_PLAY),
                "env_row": np.int32(env_row),
                "player": np.int16(player),
                "record_index": np.int32(record_index_int),
                "next_record_index": np.int32(next_record_index),
                "policy_row": np.int32(policy_row),
                "policy_source": str(search.policy_source),
                "source_record_ref": {
                    "contract_id": COMPACT_POLICY_ROW_RECORDS_CONTRACT_ID,
                    "policy_env_id": int(policy_env_id[compact_root_row]),
                    "policy_row": policy_row,
                    "compact_root_row": compact_root_row,
                    "active_output_row": output_row,
                },
            }
        )

    return _target_rows_from_dicts(
        rows,
        metadata=_metadata(chunk=chunk, target_row_count=len(rows)),
    )


def validate_compact_policy_search_arrays_v0(
    batch: HybridCompactBatch,
    *,
    selected_action: np.ndarray | Sequence[int],
    visit_policy: np.ndarray | Sequence[Sequence[float]],
    root_value: np.ndarray | Sequence[float],
    record_index: int,
    policy_source: str,
) -> CompactPolicySearchArraysV0:
    """Validate compact search output and keep it as arrays."""

    if not str(policy_source):
        raise ReplayCompatibilityError("policy_source must be a non-empty string")
    record_index_int = _int(record_index, "record_index")
    if record_index_int < 0:
        raise ReplayCompatibilityError("record_index must be nonnegative")

    action_mask_grid = _binary_bool_array(batch.action_mask, "action_mask")
    if action_mask_grid.ndim != 3 or action_mask_grid.shape[2] != ACTION_COUNT:
        raise ReplayCompatibilityError("action_mask must have shape [B,P,3]")
    batch_size, player_count = action_mask_grid.shape[:2]
    root_count = batch_size * player_count
    action_mask = action_mask_grid.reshape(root_count, ACTION_COUNT)

    policy_env_id = _int_array(batch.policy_env_id, "policy_env_id", (root_count,))
    policy_env_row = _int_array(batch.policy_env_row, "policy_env_row", (root_count,))
    policy_player = _int_array(batch.policy_player, "policy_player", (root_count,))
    _validate_row_major_sidecar(
        policy_env_id=policy_env_id,
        policy_env_row=policy_env_row,
        policy_player=policy_player,
        batch_size=batch_size,
        player_count=player_count,
    )

    done = _bool_array(batch.done, "done", (batch_size,))
    done_root = _bool_array(batch.done_root, "done_root", (root_count,))
    expected_done_root = np.repeat(done, player_count)
    if not np.array_equal(done_root, expected_done_root):
        raise ReplayCompatibilityError("done_root must equal repeat(done, player_count)")

    to_play = _int_array(batch.to_play, "to_play", (root_count,))
    if bool((to_play != DEFAULT_TO_PLAY).any()):
        raise ReplayCompatibilityError("compact fixed-opponent records require to_play=-1")

    active_root_mask = _bool_array(batch.active_root_mask, "active_root_mask", (root_count,))
    expected_active = np.logical_and(~done_root, action_mask.any(axis=1))
    if not np.array_equal(active_root_mask, expected_active):
        raise ReplayCompatibilityError(
            "active_root_mask must equal non-done roots with any legal action"
        )

    target_reward = np.asarray(batch.target_reward, dtype=np.float32)
    if target_reward.shape != (root_count, 1):
        raise ReplayCompatibilityError("target_reward must have shape [B*P,1]")
    reward = np.asarray(batch.reward, dtype=np.float32)
    if reward.shape != (batch_size, player_count):
        raise ReplayCompatibilityError("reward must have shape [B,P]")
    if not np.allclose(target_reward[:, 0], reward.reshape(root_count), atol=1e-6):
        raise ReplayCompatibilityError("target_reward must equal reward reshaped to [B*P,1]")

    active_policy_rows = np.flatnonzero(active_root_mask)
    active_root_count = int(active_policy_rows.size)
    selected = _int_array(selected_action, "selected_action", (active_root_count,))
    visits = _float_array(visit_policy, "visit_policy", (active_root_count, ACTION_COUNT))
    values = _float_array(root_value, "root_value", (active_root_count,))

    if bool(((selected < 0) | (selected >= ACTION_COUNT)).any()):
        raise ReplayCompatibilityError("selected_action is out of range")
    active_action_mask = action_mask[active_policy_rows]
    if selected.size and not bool(active_action_mask[np.arange(selected.size), selected].all()):
        raise ReplayCompatibilityError("selected_action is illegal")
    if not np.isfinite(visits).all():
        raise ReplayCompatibilityError("visit_policy must be finite")
    if bool((visits < 0.0).any()):
        raise ReplayCompatibilityError("visit_policy must be nonnegative")
    if not np.allclose(visits.sum(axis=1), 1.0, atol=1e-6):
        raise ReplayCompatibilityError("visit_policy must sum to 1")
    if bool((visits[~active_action_mask] > 1e-7).any()):
        raise ReplayCompatibilityError("visit_policy assigns mass to illegal actions")
    if not np.isfinite(values).all():
        raise ReplayCompatibilityError("root_value must be finite")

    return CompactPolicySearchArraysV0(
        record_index=record_index_int,
        policy_row=active_policy_rows.astype(np.int32, copy=True),
        env_row=policy_env_row[active_policy_rows].astype(np.int32, copy=True),
        player=policy_player[active_policy_rows].astype(np.int16, copy=True),
        action=selected.astype(np.int16, copy=True),
        action_mask=active_action_mask.astype(bool, copy=True),
        policy_target=visits.astype(np.float32, copy=True),
        root_value=values.astype(np.float32, copy=True),
        policy_source=str(policy_source),
    )


def _validate_row_major_sidecar(
    *,
    policy_env_id: np.ndarray,
    policy_env_row: np.ndarray,
    policy_player: np.ndarray,
    batch_size: int,
    player_count: int,
) -> None:
    expected_env_id = np.arange(batch_size * player_count, dtype=np.int64)
    expected_env_row = np.repeat(np.arange(batch_size, dtype=np.int64), player_count)
    expected_player = np.tile(np.arange(player_count, dtype=np.int64), batch_size)
    if policy_env_id.shape != expected_env_id.shape:
        raise ReplayCompatibilityError("policy_env_id must have one id per compact root")
    if np.unique(policy_env_id).size != policy_env_id.size:
        raise ReplayCompatibilityError("policy_env_id must be unique per compact root")
    if not np.array_equal(policy_env_row, expected_env_row):
        raise ReplayCompatibilityError("policy_env_row must match row-major ids")
    if not np.array_equal(policy_player, expected_player):
        raise ReplayCompatibilityError("policy_player must match row-major ids")


def _validate_compact_batch_matches_chunk_record(
    *,
    batch: HybridCompactBatch,
    arrays: dict[str, np.ndarray],
    record_index: int,
) -> None:
    batch_observation = np.asarray(batch.observation)
    expected_observation = arrays["observation"][record_index]
    if batch_observation.shape != expected_observation.shape or not np.array_equal(
        batch_observation,
        expected_observation,
    ):
        raise ReplayCompatibilityError("compact batch observation does not match replay record")

    batch_reward = np.asarray(batch.reward, dtype=np.float32)
    expected_reward = arrays["reward"][record_index]
    if batch_reward.shape != expected_reward.shape or not np.allclose(
        batch_reward,
        expected_reward,
        atol=1e-6,
    ):
        raise ReplayCompatibilityError("compact batch reward does not match replay record")

    batch_done = np.asarray(batch.done, dtype=np.bool_)
    expected_done = arrays["done"][record_index]
    if batch_done.shape != expected_done.shape or not np.array_equal(batch_done, expected_done):
        raise ReplayCompatibilityError("compact batch done does not match replay record")


def _replay_policy_row_by_seat(policy: dict[str, np.ndarray]) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    for policy_row, (env_row, player) in enumerate(
        zip(policy["policy_env_row"], policy["policy_player"], strict=True)
    ):
        key = (int(env_row), int(player))
        if key in mapping:
            raise ReplayCompatibilityError("replay policy rows contain a duplicate seat")
        mapping[key] = int(policy_row)
    return mapping


def _replay_policy_row_by_compact_root(batch: HybridCompactBatch) -> dict[int, int]:
    action_mask = _binary_bool_array(batch.action_mask, "action_mask")
    if action_mask.ndim != 3 or action_mask.shape[2] != ACTION_COUNT:
        raise ReplayCompatibilityError("action_mask must have shape [B,P,3]")
    batch_size, player_count = action_mask.shape[:2]
    active_root_mask = _bool_array(
        batch.active_root_mask,
        "active_root_mask",
        (batch_size * player_count,),
    )
    return {
        int(compact_root_row): int(policy_row)
        for policy_row, compact_root_row in enumerate(np.flatnonzero(active_root_mask))
    }


def _binary_bool_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.bool_:
        return array.astype(bool, copy=False)
    if not np.isin(array, (0, 1)).all():
        raise ReplayCompatibilityError(f"{name} must be binary 0/1 before bool coercion")
    return array.astype(bool, copy=False)


def _bool_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise ReplayCompatibilityError(f"{name} must be bool")
    if array.shape != shape:
        raise ReplayCompatibilityError(f"{name} must have shape {shape}, got {array.shape}")
    return array.astype(bool, copy=False)


def _int_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        raise ReplayCompatibilityError(f"{name} must be integer")
    if array.shape != shape:
        raise ReplayCompatibilityError(f"{name} must have shape {shape}, got {array.shape}")
    return array.astype(np.int64, copy=False)


def _float_array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ReplayCompatibilityError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _expect_root_batch_shapes(root_batch: CompactRootBatchV1, *, root_count: int) -> None:
    if root_count <= 0:
        raise ReplayCompatibilityError("root_batch must contain at least one root")
    _binary_bool_array(root_batch.legal_mask, "legal_mask")
    _bool_array(root_batch.active_root_mask, "active_root_mask", (root_count,))
    _int_array(root_batch.to_play, "to_play", (root_count,))
    _int_array(root_batch.env_row, "env_row", (root_count,))
    _int_array(root_batch.player, "player", (root_count,))
    _int_array(root_batch.policy_env_id, "policy_env_id", (root_count,))
    _bool_array(root_batch.done_root, "done_root", (root_count,))
    target_reward = np.asarray(root_batch.target_reward, dtype=np.float32)
    if target_reward.shape != (root_count, 1):
        raise ReplayCompatibilityError("target_reward must have shape [root,1]")
    observation = np.asarray(root_batch.observation)
    if observation.shape[0] != root_count:
        raise ReplayCompatibilityError("observation must have one row per compact root")
    final_observation = root_batch.final_observation
    if final_observation is not None and np.asarray(final_observation).shape != observation.shape:
        raise ReplayCompatibilityError("final_observation must match observation shape")


def _compact_index_group_metadata(
    chunk: Any,
    *,
    group_count: int,
    target_row_count: int,
) -> dict[str, Any]:
    metadata = _metadata(chunk=chunk, target_row_count=target_row_count)
    metadata.update(
        {
            "compact_index_group_schema_id": ("curvyzero_compact_replay_index_row_groups/v1"),
            "compact_index_group_count": int(group_count),
            "target_row_count": int(target_row_count),
            "observation_materialized_at": "sampler_edge",
        }
    )
    return metadata


def _concat_target_arrays(
    rows: Sequence[SourceStateMultiplayerTargetRowsV0],
    field_name: str,
    dtype: Any,
) -> np.ndarray:
    arrays = [np.asarray(getattr(row, field_name), dtype=dtype) for row in rows]
    if not arrays:
        raise ReplayCompatibilityError("cannot concatenate empty target row groups")
    return np.concatenate(arrays, axis=0).astype(dtype, copy=True)


def _env_player_identity_digest(env_row: Any, player: Any) -> str:
    digest = hashlib.sha256()
    for label, value, dtype in (
        ("env_row", env_row, np.int32),
        ("player", player, np.int16),
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype=dtype).reshape(-1))
        digest.update(label.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _compact_array_digest_v1(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _plain_metadata_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain_metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _elapsed_sec(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def _int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReplayCompatibilityError(f"{name} must be an integer") from exc


__all__ = [
    "COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1",
    "COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1",
    "COMPACT_POLICY_ROW_RECORDS_CONTRACT_ID",
    "COMPACT_ROOT_BUILD_REQUEST_KIND_HOST_ARRAY",
    "COMPACT_ROOT_BUILD_REQUEST_KIND_RESIDENT_ROOT_VIEW",
    "COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID",
    "COMPACT_ROOT_MECHANICS_OUTCOME_SCHEMA_ID",
    "COMPACT_REPLAY_ENV_PLAYER_IDENTITY_DIGEST_KEY",
    "COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID",
    "RESIDENT_OBSERVATION_BATCH_SCHEMA_ID",
    "RESIDENT_OBSERVATION_OWNER",
    "CompactPolicySearchArraysV0",
    "CompactReplayIndexRowsV1",
    "CompactReplayChunkV1",
    "CompactRootActionContextV1",
    "CompactRootBuildRequestV1",
    "CompactRootBatchV1",
    "CompactRootTransitionOutcomeV1",
    "CompactSearchResultV1",
    "ResidentObservationBatchV1",
    "build_compact_root_batch_v1_from_request",
    "build_compact_device_replay_index_rows_v1_from_owner_action_context_payload",
    "build_compact_replay_index_rows_v1_from_owner_action_context_payload",
    "build_compact_replay_index_rows_v1_from_search_result",
    "build_compact_replay_chunk_v1_from_search_result",
    "build_compact_root_batch_v1",
    "compact_transition_outcome_v1_from_root_build_request",
    "compact_transition_outcome_v1_from_next_root_batch",
    "compact_root_action_context_v1_from_request",
    "compact_root_build_request_v1_from_batch",
    "build_compact_target_rows_from_search_arrays_v0",
    "build_policy_row_records_from_compact_search_v0",
    "materialize_compact_target_rows_from_index_rows_v1",
    "materialize_compact_target_rows_from_index_row_groups_v1",
    "sample_compact_target_rows_from_index_row_groups_v1",
    "validate_resident_observation_batch_v1",
    "validate_compact_search_result_v1",
    "validate_compact_search_result_identity_v1",
    "validate_compact_policy_search_arrays_v0",
]
