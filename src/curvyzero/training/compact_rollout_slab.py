"""Profile-only compact rollout slab for search-service dataflow probes."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
import time
from types import SimpleNamespace
from typing import Any
from typing import Mapping

import numpy as np

from curvyzero.env.vector_multiplayer_env import ACTION_COUNT
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
)
from curvyzero.training.compact_observation_contract import (
    COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
)
from curvyzero.training.compact_policy_row_bridge import CompactDeviceReplayIndexRowsV1
from curvyzero.training.compact_policy_row_bridge import CompactReplayIndexRowsV1
from curvyzero.training.compact_policy_row_bridge import CompactRootActionContextV1
from curvyzero.training.compact_policy_row_bridge import CompactRootBuildRequestV1
from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1
from curvyzero.training.compact_policy_row_bridge import (
    COMPACT_ROOT_BUILD_REQUEST_KIND_RESIDENT_ROOT_VIEW,
)
from curvyzero.training.compact_policy_row_bridge import COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID
from curvyzero.training.compact_policy_row_bridge import (
    COMPACT_ROOT_MECHANICS_OUTCOME_SCHEMA_ID,
)
from curvyzero.training.compact_policy_row_bridge import (
    COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_device_replay_index_rows_v1_from_payload,
)
from curvyzero.training.compact_policy_row_bridge import (
    build_compact_replay_index_rows_v1_from_search_result,
)
from curvyzero.training.compact_policy_row_bridge import build_compact_root_batch_v1
from curvyzero.training.compact_policy_row_bridge import (
    compact_root_build_request_v1_from_batch,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_root_action_context_v1_from_request,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_transition_outcome_v1_from_root_build_request,
)
from curvyzero.training.compact_policy_row_bridge import (
    compact_transition_outcome_v1_from_next_root_batch,
)
from curvyzero.training.compact_policy_row_bridge import (
    validate_compact_search_result_identity_v1,
)
from curvyzero.training.compact_policy_refresh_handoff import (
    compact_policy_refresh_metadata_from_state_v1,
)
from curvyzero.training.compact_rollout_modes import COMPACT_ROLLOUT_SLAB_ACTION_MODES
from curvyzero.training.compact_rollout_modes import (
    COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM,
)
from curvyzero.training.compact_rollout_modes import (
    COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK,
)
from curvyzero.training.compact_search_service import CompactSearchServiceV1
from curvyzero.training.compact_search_service import CompactSearchActionStepV1
from curvyzero.training.compact_search_service import COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID
from curvyzero.training.compact_search_service import (
    compact_search_result_v1_from_two_phase_payloads,
)
from curvyzero.training.compact_search_service import compact_search_array_digest_v1
from curvyzero.training.compact_search_service import (
    validate_compact_search_two_phase_payload_v1,
)
from curvyzero.training.compact_search_service import (
    validate_compact_device_search_two_phase_payload_v1,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


COMPACT_ROLLOUT_SLAB_STEP_SCHEMA_ID = "curvyzero_compact_rollout_slab_step/v1"
COMPACT_OWNER_SEARCH_REPLAY_APPEND_ENTRY_SCHEMA_ID = (
    "curvyzero_compact_owner_search_replay_append_entry/v1"
)
COMPACT_OWNER_SEARCH_REPLAY_APPEND_INDEX_ENTRY_SCHEMA_ID = (
    "curvyzero_compact_owner_search_replay_append_index_entry/v1"
)
COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID = (
    "curvyzero_compact_owner_search_replay_append_transition_entry/v1"
)
COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID = (
    "curvyzero_compact_owner_search_replay_append_transition_batch/v1"
)
COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID = (
    "curvyzero_compact_owner_search_replay_append_derived_transition_batch/v1"
)
COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED = (
    "fixed_coalesced_transition_batch_v1"
)
COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND = (
    "owner_local_next_root_derived_transition_batch_v1"
)
COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION = (
    "owner_search_direct_transition_stepper_v1"
)
COMPACT_OWNER_SEARCH_DIRECT_STEP_DISPATCH_HANDLE_SCHEMA_ID = (
    "curvyzero_compact_owner_search_direct_step_dispatch_handle/v1"
)
COMPACT_OWNER_MECHANICS_STEP_FRAME_HANDLE_SCHEMA_ID = (
    "curvyzero_compact_owner_mechanics_step_frame_handle/v1"
)

_COMPACT_ROLLOUT_SLAB_COMMIT_TIMING_FIELDS = (
    "compact_rollout_slab_commit_action_check_sec",
    "compact_rollout_slab_replay_payload_flush_sec",
    "compact_rollout_slab_replay_payload_validate_sec",
    "compact_rollout_slab_replay_payload_materialize_sec",
    "compact_rollout_slab_replay_result_validate_sec",
    "compact_rollout_slab_replay_index_rows_build_sec",
    "compact_rollout_slab_replay_index_rows_store_sec",
)
_COMPACT_ROLLOUT_SLAB_REPLAY_INDEX_DETAIL_FIELDS = (
    "compact_rollout_slab_commit_child_accounted_sec",
    "compact_rollout_slab_commit_residual_sec",
    "compact_rollout_slab_replay_index_rows_identity_validate_sec",
    "compact_rollout_slab_replay_index_rows_terminal_prepare_sec",
    "compact_rollout_slab_replay_index_rows_target_tensor_sec",
    "compact_rollout_slab_replay_index_rows_scalar_host_pack_sec",
    "compact_rollout_slab_replay_index_rows_scalar_device_transfer_sec",
    "compact_rollout_slab_replay_index_rows_metadata_sec",
    "compact_rollout_slab_replay_index_rows_scalar_packed_h2d_bytes",
    "compact_rollout_slab_replay_index_rows_scalar_tensor_count",
)
_COMPACT_ROLLOUT_SLAB_ALL_COMMIT_FIELDS = (
    *_COMPACT_ROLLOUT_SLAB_COMMIT_TIMING_FIELDS,
    *_COMPACT_ROLLOUT_SLAB_REPLAY_INDEX_DETAIL_FIELDS,
)
_COMPACT_ROLLOUT_SLAB_INTERNAL_TIMING_FIELDS = (
    "compact_rollout_slab_commit_previous_sec",
    "compact_rollout_slab_root_batch_build_sec",
    "compact_rollout_slab_root_build_request_sec",
    "compact_rollout_slab_root_tape_record_sec",
    "compact_rollout_slab_search_dispatch_wall_sec",
    "compact_rollout_slab_search_identity_validation_sec",
    "compact_rollout_slab_joint_action_assembly_sec",
    "compact_rollout_slab_pending_store_sec",
    "compact_rollout_slab_action_dispatch_step_overlap_submit_wall_sec",
    "compact_rollout_slab_action_dispatch_step_overlap_resolve_wall_sec",
    "compact_rollout_slab_action_dispatch_step_overlap_submit_to_resolve_elapsed_sec",
    "compact_rollout_slab_action_dispatch_step_overlap_parent_work_sec",
    "compact_rollout_slab_telemetry_build_sec",
    *_COMPACT_ROLLOUT_SLAB_COMMIT_TIMING_FIELDS,
)
_COMPACT_ROLLOUT_SLAB_OWNER_SEARCH_TIMING_FIELDS = (
    (
        "compact_owner_search_parent_publish_sec",
        "compact_rollout_slab_owner_search_parent_publish_sec",
    ),
    (
        "compact_owner_search_parent_submit_sec",
        "compact_rollout_slab_owner_search_parent_submit_sec",
    ),
    (
        "compact_owner_search_parent_wait_sec",
        "compact_rollout_slab_owner_search_parent_wait_sec",
    ),
    (
        "compact_owner_search_parent_wall_sec",
        "compact_rollout_slab_owner_search_parent_wall_sec",
    ),
    (
        "compact_owner_search_worker_wall_sec",
        "compact_rollout_slab_owner_search_worker_wall_sec",
    ),
    (
        "compact_owner_search_worker_root_resolve_sec",
        "compact_rollout_slab_owner_search_worker_root_resolve_sec",
    ),
    (
        "compact_owner_search_worker_search_sec",
        "compact_rollout_slab_owner_search_worker_search_sec",
    ),
    (
        "compact_owner_search_worker_replay_append_sec",
        "compact_rollout_slab_owner_search_worker_replay_append_sec",
    ),
    (
        "compact_owner_search_worker_learner_train_sec",
        "compact_rollout_slab_owner_search_worker_learner_train_sec",
    ),
    (
        "compact_owner_search_worker_search_refresh_sec",
        "compact_rollout_slab_owner_search_worker_search_refresh_sec",
    ),
)


@dataclass(frozen=True, slots=True)
class CompactRolloutSlabStepV1:
    """One profile-only compact slab step."""

    schema_id: str
    record_index: int
    compact_batch: Any
    root_batch: CompactRootBatchV1 | None
    search_result: CompactSearchResultV1 | None
    action_step: CompactSearchActionStepV1 | None
    next_joint_action: np.ndarray
    committed_index_rows: CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1 | None
    telemetry: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactOwnerSearchDirectStepDispatchHandleV1:
    """Parent-local handle for an owner action search submitted before resolve."""

    schema_id: str
    dispatch_id: int
    record_index: int
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PendingOwnerSearchDirectStepDispatchV1:
    handle: CompactOwnerSearchDirectStepDispatchHandleV1
    action_dispatch_handle: Any
    compact_batch: Any | None
    root_action_context: CompactRootActionContextV1 | None
    root_action_context_handle: Any | None
    root_metadata: dict[str, Any]
    commit_timing: dict[str, float]
    commit_previous_sec: float
    root_build_request_sec: float
    submit_wall_sec: float
    opened: float


@dataclass(frozen=True, slots=True)
class CompactOwnerSearchReplayAppendEntryV1:
    """Host-only replay append entry staged for the owner-search worker."""

    schema_id: str
    record_index: int
    previous_compact_batch: Any
    current_compact_batch: Any
    index_rows: CompactReplayIndexRowsV1
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactOwnerSearchReplayAppendIndexEntryV1:
    """Index-only replay append entry for owner-cached root batches."""

    schema_id: str
    record_index: int
    next_record_index: int
    index_rows: CompactReplayIndexRowsV1
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactOwnerSearchReplayAppendTransitionEntryV1:
    """Transition facts for owner-side replay-row materialization."""

    schema_id: str
    record_index: int
    next_record_index: int
    replay_payload_handle: str
    next_joint_action: np.ndarray
    next_reward: np.ndarray
    next_done: np.ndarray
    next_terminated: np.ndarray
    next_truncated: np.ndarray
    next_final_reward_map: np.ndarray
    next_final_observation_row_mask: np.ndarray
    policy_source: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactOwnerSearchReplayAppendTransitionBatchV1:
    """Fixed-shaped transition batch for owner-side replay materialization."""

    schema_id: str
    transition_count: int
    record_indices: np.ndarray
    next_record_indices: np.ndarray
    replay_payload_handles: tuple[str, ...]
    selected_action_digests: tuple[str, ...]
    search_replay_payload_digests: tuple[str, ...]
    next_joint_action: np.ndarray
    next_reward: np.ndarray
    next_done: np.ndarray
    next_terminated: np.ndarray
    next_truncated: np.ndarray
    next_final_reward_map: np.ndarray
    next_final_observation_row_mask: np.ndarray
    policy_source: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompactOwnerSearchReplayAppendDerivedTransitionBatchV1:
    """Transition batch whose env outcomes are derived from owner-cached next roots."""

    schema_id: str
    transition_count: int
    record_indices: np.ndarray
    next_record_indices: np.ndarray
    replay_payload_handles: tuple[str, ...]
    selected_action_digests: tuple[str, ...]
    search_replay_payload_digests: tuple[str, ...]
    applied_action_counts: np.ndarray
    applied_action_checksums: np.ndarray
    policy_source: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PendingCompactSearchV1:
    record_index: int
    compact_batch: Any | None
    root_batch: CompactRootBatchV1 | None
    search_result: CompactSearchResultV1 | None
    action_step: CompactSearchActionStepV1 | None
    next_joint_action: np.ndarray


@dataclass(frozen=True, slots=True)
class _OwnerSearchTransitionFactsV1:
    record_index: int
    next_record_index: int
    replay_payload_handle: str
    selected_action_digest: str
    search_replay_payload_digest: str
    next_joint_action: np.ndarray
    next_reward: np.ndarray
    next_done: np.ndarray
    next_terminated: np.ndarray
    next_truncated: np.ndarray
    next_final_reward_map: np.ndarray
    next_final_observation_row_mask: np.ndarray
    policy_source: str
    terminal_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _OwnerSearchDerivedTransitionFactsV1:
    record_index: int
    next_record_index: int
    replay_payload_handle: str
    selected_action_digest: str
    search_replay_payload_digest: str
    applied_action_count: int
    applied_action_checksum: int
    policy_source: str


class CompactRolloutSlab:
    """Small profile-only owner for compact root/search/action/replay flow."""

    profile_only = True
    calls_train_muzero = False
    trainer_defaults_changed = False
    touches_live_runs = False

    def __init__(
        self,
        *,
        batch_size: int,
        player_count: int,
        search_service: CompactSearchServiceV1,
        search_lane: str,
        policy_source: str,
        copy_root_observation: bool = False,
        action_feedback_mode: str = COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK,
        root_tape_recorder: Any | None = None,
        retain_committed_index_rows: bool = True,
    ) -> None:
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.player_count = _positive_int(player_count, "player_count")
        self.search_service = search_service
        self.search_lane = str(search_lane)
        if not self.search_lane:
            raise ReplayCompatibilityError("search_lane must be a non-empty string")
        self.policy_source = str(policy_source)
        if not self.policy_source:
            raise ReplayCompatibilityError("policy_source must be a non-empty string")
        self.copy_root_observation = bool(copy_root_observation)
        self.root_tape_recorder = root_tape_recorder
        self.action_feedback_mode = str(action_feedback_mode)
        if self.action_feedback_mode not in COMPACT_ROLLOUT_SLAB_ACTION_MODES:
            allowed = ", ".join(COMPACT_ROLLOUT_SLAB_ACTION_MODES)
            raise ReplayCompatibilityError(
                f"action_feedback_mode must be one of {allowed}; got {self.action_feedback_mode!r}"
            )
        self.retain_committed_index_rows = bool(retain_committed_index_rows)
        self._record_index = 0
        self._pending: _PendingCompactSearchV1 | None = None
        self._committed_index_rows: list[
            CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1
        ] = []
        self._committed_index_group_count = 0
        self._committed_index_row_count = 0
        self._dropped_pending_searches = 0
        self._action_override_drop_count = 0
        self._replay_payload_flush_count = 0
        self._last_commit_replay_payload_flushed = False
        self._last_commit_replay_payload_d2h_bytes = 0
        self._last_commit_timing = _empty_commit_timing()
        self._closed = False

    @property
    def committed_index_rows(
        self,
    ) -> tuple[CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1, ...]:
        """Committed compact replay-index rows, returned as copies."""

        return tuple(_copy_compact_replay_index_rows(rows) for rows in self._committed_index_rows)

    @property
    def committed_index_row_count(self) -> int:
        """Total committed compact replay-index row count."""

        return int(self._committed_index_row_count)

    @property
    def committed_index_group_count(self) -> int:
        """Total committed compact replay-index groups."""

        return int(self._committed_index_group_count)

    @property
    def dropped_pending_search_count(self) -> int:
        """Pending searches dropped because no following env result existed."""

        return int(self._dropped_pending_searches)

    @property
    def action_override_drop_count(self) -> int:
        """Search results dropped because controlled profile actions overrode them."""

        return int(self._action_override_drop_count)

    def step(self, compact_batch: Any) -> CompactRolloutSlabStepV1:
        """Commit previous search rows, then search the current compact batch."""

        if self._closed:
            raise ReplayCompatibilityError("compact rollout slab is closed")
        self._validate_batch_shape(compact_batch)
        commit_started = time.perf_counter()
        committed = self._commit_previous(compact_batch)
        commit_previous_sec = _elapsed(commit_started)
        commit_timing = dict(self._last_commit_timing)
        root_metadata = {"compact_rollout_slab": True}
        root_metadata.update(
            _compact_policy_refresh_metadata_from_search_service(self.search_service)
        )
        root_batch_started = time.perf_counter()
        root_batch = build_compact_root_batch_v1(
            compact_batch,
            search_lane=self.search_lane,
            metadata=root_metadata,
            copy_observation=self.copy_root_observation,
            observation_source=str(
                getattr(
                    compact_batch,
                    "observation_source",
                    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
                )
            ),
            resident_observation=getattr(compact_batch, "resident_observation", None),
        )
        root_batch_build_sec = _elapsed(root_batch_started)
        record_index = self._record_index
        root_tape_record_sec = 0.0
        if self.root_tape_recorder is not None:
            record_root_batch = getattr(
                self.root_tape_recorder,
                "record_root_batch",
                None,
            )
            if not callable(record_root_batch):
                raise ReplayCompatibilityError("root_tape_recorder must expose record_root_batch")
            root_tape_started = time.perf_counter()
            record_root_batch(
                root_batch,
                record_index=record_index,
                metadata={
                    "compact_rollout_slab_capture": True,
                    "search_lane": self.search_lane,
                    "action_feedback_mode": self.action_feedback_mode,
                },
            )
            root_tape_record_sec = _elapsed(root_tape_started)
        search_result: CompactSearchResultV1 | None = None
        action_step: CompactSearchActionStepV1 | None = None
        two_phase_search = _search_service_supports_two_phase(self.search_service)
        identity_validation_sec = 0.0
        joint_action_assembly_sec = 0.0
        search_dispatch_started = time.perf_counter()
        if two_phase_search:
            action_step = self.search_service.run_action_step(root_batch)
            search_dispatch_sec = _elapsed(search_dispatch_started)
            identity_started = time.perf_counter()
            _validate_compact_search_action_step_identity(root_batch, action_step)
            identity_validation_sec = _elapsed(identity_started)
            joint_action_started = time.perf_counter()
            next_joint_action = selected_joint_action_from_action_step(
                root_batch,
                action_step,
                batch_size=self.batch_size,
                player_count=self.player_count,
            )
            joint_action_assembly_sec = _elapsed(joint_action_started)
        else:
            search_result = self.search_service.run(root_batch)
            search_dispatch_sec = _elapsed(search_dispatch_started)
            identity_started = time.perf_counter()
            validate_compact_search_result_identity_v1(root_batch, search_result)
            identity_validation_sec = _elapsed(identity_started)
            joint_action_started = time.perf_counter()
            next_joint_action = selected_joint_action_from_search_result(
                root_batch,
                search_result,
                batch_size=self.batch_size,
                player_count=self.player_count,
            )
            joint_action_assembly_sec = _elapsed(joint_action_started)
        pending_store_started = time.perf_counter()
        self._pending = _PendingCompactSearchV1(
            record_index=record_index,
            compact_batch=compact_batch,
            root_batch=root_batch,
            search_result=search_result,
            action_step=action_step,
            next_joint_action=next_joint_action,
        )
        self._record_index += 1
        pending_store_sec = _elapsed(pending_store_started)
        commit_child_accounted_sec = sum(
            float(commit_timing.get(key, 0.0)) for key in _COMPACT_ROLLOUT_SLAB_COMMIT_TIMING_FIELDS
        )
        internal_timing = {
            **commit_timing,
            "compact_rollout_slab_commit_child_accounted_sec": (commit_child_accounted_sec),
            "compact_rollout_slab_commit_residual_sec": (
                commit_previous_sec - commit_child_accounted_sec
            ),
            "compact_rollout_slab_commit_previous_sec": commit_previous_sec,
            "compact_rollout_slab_root_batch_build_sec": root_batch_build_sec,
            "compact_rollout_slab_root_tape_record_sec": root_tape_record_sec,
            "compact_rollout_slab_search_dispatch_wall_sec": search_dispatch_sec,
            "compact_rollout_slab_search_identity_validation_sec": (identity_validation_sec),
            "compact_rollout_slab_joint_action_assembly_sec": joint_action_assembly_sec,
            "compact_rollout_slab_pending_store_sec": pending_store_sec,
        }
        telemetry_started = time.perf_counter()
        telemetry = _slab_telemetry(
            root_batch,
            search_result,
            action_step,
            committed,
            action_feedback_mode=self.action_feedback_mode,
            action_override_drop_count=self._action_override_drop_count,
            two_phase_search=two_phase_search,
            replay_payload_flush_count=self._replay_payload_flush_count,
            committed_replay_payload_flushed=self._last_commit_replay_payload_flushed,
            committed_replay_payload_d2h_bytes=(self._last_commit_replay_payload_d2h_bytes),
            internal_timing=internal_timing,
        )
        telemetry["compact_rollout_slab_telemetry_build_sec"] = _elapsed(telemetry_started)
        telemetry["compact_rollout_slab_internal_accounted_sec"] = sum(
            float(telemetry.get(key, 0.0)) for key in _COMPACT_ROLLOUT_SLAB_INTERNAL_TIMING_FIELDS
        )
        return CompactRolloutSlabStepV1(
            schema_id=COMPACT_ROLLOUT_SLAB_STEP_SCHEMA_ID,
            record_index=record_index,
            compact_batch=compact_batch,
            root_batch=root_batch,
            search_result=search_result,
            action_step=action_step,
            next_joint_action=next_joint_action,
            committed_index_rows=committed,
            telemetry=telemetry,
        )

    def flush_final(
        self,
        final_batch: Any | None = None,
    ) -> CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1 | None:
        """Finish the pending search without starting another search."""

        if self._pending is None:
            return None
        if final_batch is None:
            self._pending = None
            self._dropped_pending_searches += 1
            return None
        self._validate_batch_shape(final_batch)
        return self._commit_previous(final_batch)

    def close(
        self,
        final_batch: Any | None = None,
    ) -> CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1 | None:
        """Close this profile slab and reject future ``step`` calls."""

        committed = self.flush_final(final_batch)
        self._closed = True
        return committed

    def _commit_previous(
        self,
        next_batch: Any,
    ) -> CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1 | None:
        self._last_commit_timing = _empty_commit_timing()
        commit_timing = self._last_commit_timing
        pending = self._pending
        if pending is None:
            return None
        self._last_commit_replay_payload_flushed = False
        self._last_commit_replay_payload_d2h_bytes = 0
        next_joint_action = np.asarray(next_batch.joint_action, dtype=np.int16)
        if self.action_feedback_mode == COMPACT_ROLLOUT_SLAB_ACTION_MODE_SCRIPTED_RANDOM:
            self._pending = None
            self._action_override_drop_count += 1
            return None
        action_check_started = time.perf_counter()
        active_env_row, active_player, active_selected = _pending_action_arrays(pending)
        if active_selected.size and not np.array_equal(
            next_joint_action[active_env_row, active_player],
            active_selected,
        ):
            raise ReplayCompatibilityError(
                "compact slab next batch did not apply staged selected actions"
            )
        commit_timing["compact_rollout_slab_commit_action_check_sec"] = _elapsed(
            action_check_started
        )
        search_result = pending.search_result
        if search_result is None:
            action_step = pending.action_step
            if action_step is None:
                raise ReplayCompatibilityError("pending compact search has no action step")
            if _action_step_owner_materializes_replay(action_step):
                transition_started = time.perf_counter()
                entry = _owner_search_replay_append_transition_entry(
                    pending,
                    action_step=action_step,
                    next_joint_action=next_joint_action,
                    next_batch=next_batch,
                    policy_source=self.policy_source,
                )
                self._stage_committed_replay_for_search_service(entry)
                self._pending = None
                commit_timing["compact_rollout_slab_replay_index_rows_build_sec"] = 0.0
                commit_timing["compact_rollout_slab_replay_index_rows_store_sec"] = _elapsed(
                    transition_started
                )
                commit_timing["compact_rollout_slab_owner_replay_transition_stage_sec"] = (
                    commit_timing["compact_rollout_slab_replay_index_rows_store_sec"]
                )
                commit_timing["compact_rollout_slab_owner_replay_transition_stage_count"] = 1.0
                self._last_commit_replay_payload_flushed = False
                return None
            flush = getattr(self.search_service, "flush_replay_payload", None)
            if not callable(flush):
                raise ReplayCompatibilityError(
                    "two-phase compact search service cannot flush replay payload"
                )
            use_device_payload = (
                str(pending.root_batch.observation_source)
                == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
            )
            flush_device = getattr(self.search_service, "flush_device_replay_payload", None)
            if use_device_payload and callable(flush_device):
                flush_started = time.perf_counter()
                replay_payload = flush_device(action_step.replay_payload_handle)
                commit_timing["compact_rollout_slab_replay_payload_flush_sec"] = _elapsed(
                    flush_started
                )
                validate_started = time.perf_counter()
                validate_compact_device_search_two_phase_payload_v1(
                    action_step,
                    replay_payload,
                )
                commit_timing["compact_rollout_slab_replay_payload_validate_sec"] = _elapsed(
                    validate_started
                )
                self._replay_payload_flush_count += 1
                self._last_commit_replay_payload_flushed = True
                build_started = time.perf_counter()
                committed = build_compact_device_replay_index_rows_v1_from_payload(
                    pending.compact_batch,
                    pending.root_batch,
                    action_step,
                    replay_payload,
                    record_index=pending.record_index,
                    next_joint_action=next_joint_action,
                    next_reward=np.asarray(next_batch.reward, dtype=np.float32),
                    next_done=np.asarray(next_batch.done, dtype=np.bool_),
                    next_terminated=np.asarray(
                        getattr(next_batch, "terminated", next_batch.done),
                        dtype=np.bool_,
                    ),
                    next_truncated=np.asarray(
                        getattr(
                            next_batch,
                            "truncated",
                            np.zeros_like(next_batch.done, dtype=np.bool_),
                        ),
                        dtype=np.bool_,
                    ),
                    next_final_reward_map=np.asarray(
                        getattr(next_batch, "final_reward_map", next_batch.reward),
                        dtype=np.float32,
                    ),
                    next_final_observation_row_mask=np.asarray(
                        next_batch.final_observation_row_mask,
                        dtype=np.bool_,
                    ),
                    policy_source=self.policy_source,
                    metadata={
                        "compact_rollout_slab": True,
                        "compact_rollout_slab_device_replay_payload": True,
                        **_compact_terminal_metadata_from_batch(next_batch),
                    },
                )
                commit_timing["compact_rollout_slab_replay_index_rows_build_sec"] = _elapsed(
                    build_started
                )
                _add_replay_index_detail_timing(commit_timing, committed)
                store_started = time.perf_counter()
                stored = self._store_committed_index_rows(
                    committed,
                    previous_compact_batch=pending.compact_batch,
                    current_compact_batch=next_batch,
                    record_index=pending.record_index,
                )
                commit_timing["compact_rollout_slab_replay_index_rows_store_sec"] = _elapsed(
                    store_started
                )
                return stored
            if use_device_payload:
                raise ReplayCompatibilityError(
                    "resident two-phase compact search requires "
                    "flush_device_replay_payload; refusing host replay fallback"
                )
            flush_started = time.perf_counter()
            replay_payload = flush(action_step.replay_payload_handle)
            commit_timing["compact_rollout_slab_replay_payload_flush_sec"] = _elapsed(flush_started)
            validate_started = time.perf_counter()
            validate_compact_search_two_phase_payload_v1(action_step, replay_payload)
            commit_timing["compact_rollout_slab_replay_payload_validate_sec"] = _elapsed(
                validate_started
            )
            self._last_commit_replay_payload_d2h_bytes = _compact_search_replay_payload_d2h_bytes(
                replay_payload
            )
            materialize_started = time.perf_counter()
            search_result = compact_search_result_v1_from_two_phase_payloads(
                pending.root_batch,
                action_step,
                replay_payload,
                metadata={
                    "compact_rollout_slab_delayed_replay_payload": True,
                    "compact_rollout_slab_record_index": int(pending.record_index),
                },
            )
            commit_timing["compact_rollout_slab_replay_payload_materialize_sec"] = _elapsed(
                materialize_started
            )
            result_validate_started = time.perf_counter()
            validate_compact_search_result_identity_v1(pending.root_batch, search_result)
            commit_timing["compact_rollout_slab_replay_result_validate_sec"] = _elapsed(
                result_validate_started
            )
            self._replay_payload_flush_count += 1
            self._last_commit_replay_payload_flushed = True
        next_done = np.asarray(next_batch.done, dtype=np.bool_)
        next_terminated = np.asarray(
            getattr(next_batch, "terminated", next_done),
            dtype=np.bool_,
        )
        next_truncated = np.asarray(
            getattr(next_batch, "truncated", np.zeros_like(next_done, dtype=np.bool_)),
            dtype=np.bool_,
        )
        build_started = time.perf_counter()
        committed = build_compact_replay_index_rows_v1_from_search_result(
            pending.compact_batch,
            pending.root_batch,
            search_result,
            record_index=pending.record_index,
            next_joint_action=next_joint_action,
            next_reward=np.asarray(next_batch.reward, dtype=np.float32),
            next_done=next_done,
            next_terminated=next_terminated,
            next_truncated=next_truncated,
            next_final_reward_map=np.asarray(
                getattr(next_batch, "final_reward_map", next_batch.reward),
                dtype=np.float32,
            ),
            next_final_observation_row_mask=np.asarray(
                next_batch.final_observation_row_mask,
                dtype=np.bool_,
            ),
            policy_source=self.policy_source,
            metadata={
                "compact_rollout_slab": True,
                **_compact_terminal_metadata_from_batch(next_batch),
            },
        )
        commit_timing["compact_rollout_slab_replay_index_rows_build_sec"] = _elapsed(build_started)
        _add_replay_index_detail_timing(commit_timing, committed)
        store_started = time.perf_counter()
        stored = self._store_committed_index_rows(
            committed,
            previous_compact_batch=pending.compact_batch,
            current_compact_batch=next_batch,
            record_index=pending.record_index,
        )
        commit_timing["compact_rollout_slab_replay_index_rows_store_sec"] = _elapsed(store_started)
        return stored

    def _validate_batch_shape(self, compact_batch: Any) -> None:
        observation = np.asarray(compact_batch.observation)
        expected = (self.batch_size, self.player_count)
        if observation.ndim != 5 or observation.shape[:2] != expected:
            raise ReplayCompatibilityError("compact slab observation shape mismatch")
        if np.asarray(compact_batch.joint_action).shape != expected:
            raise ReplayCompatibilityError("compact slab joint_action shape mismatch")

    def _store_committed_index_rows(
        self,
        committed: CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1,
        *,
        previous_compact_batch: Any | None = None,
        current_compact_batch: Any | None = None,
        record_index: int | None = None,
    ) -> CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1:
        self._pending = None
        if isinstance(committed, CompactDeviceReplayIndexRowsV1):
            self._record_committed_index_rows(committed)
            self._stage_committed_replay_for_search_service(
                _owner_search_replay_append_entry(
                    committed,
                    previous_compact_batch=previous_compact_batch,
                    current_compact_batch=current_compact_batch,
                    record_index=record_index,
                )
            )
            return committed
        stored = _copy_compact_replay_index_rows(committed)
        self._record_committed_index_rows(stored)
        self._stage_committed_replay_for_search_service(
            _owner_search_replay_append_entry(
                stored,
                previous_compact_batch=previous_compact_batch,
                current_compact_batch=current_compact_batch,
                record_index=record_index,
            )
        )
        return _copy_compact_replay_index_rows(stored)

    def _record_committed_index_rows(
        self,
        rows: CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1,
    ) -> None:
        self._committed_index_group_count += 1
        self._committed_index_row_count += _compact_index_row_count(rows)
        if self.retain_committed_index_rows:
            self._committed_index_rows.append(rows)

    def _stage_committed_replay_for_search_service(
        self,
        committed: Any,
    ) -> None:
        stage = getattr(self.search_service, "stage_replay_append_entries", None)
        if not callable(stage):
            return
        started = time.perf_counter()
        staged_count = int(stage(committed) or 0)
        self._last_commit_timing["compact_rollout_slab_owner_replay_stage_sec"] = _elapsed(started)
        self._last_commit_timing["compact_rollout_slab_owner_replay_stage_entry_count"] = float(
            staged_count
        )


class CompactOwnerSearchDirectStepperV1:
    """Owner-search-only stepper that bypasses parent slab replay rows."""

    profile_only = True
    calls_train_muzero = False
    trainer_defaults_changed = False
    touches_live_runs = False
    supports_two_phase_compact_search = True

    def __init__(
        self,
        *,
        batch_size: int,
        player_count: int,
        search_service: Any,
        search_lane: str,
        policy_source: str,
        copy_root_observation: bool = False,
        transition_batch_size: int = 1,
        resident_root_host_observation_stub: bool = False,
        direct_root_build_request: bool = False,
        owner_local_transition_derivation: bool = False,
        owner_proxy_transition_closure: bool = False,
    ) -> None:
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.player_count = _positive_int(player_count, "player_count")
        self.search_service = search_service
        self.search_lane = str(search_lane)
        if not self.search_lane:
            raise ReplayCompatibilityError("search_lane must be a non-empty string")
        self.policy_source = str(policy_source)
        if not self.policy_source:
            raise ReplayCompatibilityError("policy_source must be a non-empty string")
        self.copy_root_observation = bool(copy_root_observation)
        self.resident_root_host_observation_stub = bool(resident_root_host_observation_stub)
        self.direct_root_build_request = bool(direct_root_build_request)
        if self.direct_root_build_request and not self.resident_root_host_observation_stub:
            raise ValueError(
                "direct_root_build_request requires resident_root_host_observation_stub"
            )
        self.owner_local_transition_derivation = bool(owner_local_transition_derivation)
        self.owner_proxy_transition_closure = bool(owner_proxy_transition_closure)
        if self.owner_proxy_transition_closure and not self.owner_local_transition_derivation:
            raise ValueError("owner_proxy_transition_closure requires owner_local_transition_derivation")
        if self.owner_proxy_transition_closure and not self.direct_root_build_request:
            raise ValueError("owner_proxy_transition_closure requires direct_root_build_request")
        self.transition_batch_size = _positive_int(
            transition_batch_size,
            "transition_batch_size",
        )
        if self.owner_local_transition_derivation and self.transition_batch_size <= 1:
            raise ValueError("owner_local_transition_derivation requires transition_batch_size > 1")
        self.retain_committed_index_rows = False
        self._record_index = 0
        self._pending: _PendingCompactSearchV1 | None = None
        self._pending_transition_facts: list[_OwnerSearchTransitionFactsV1] = []
        self._pending_derived_transition_facts: list[_OwnerSearchDerivedTransitionFactsV1] = []
        self._direct_transition_stage_count = 0
        self._direct_transition_stage_entry_count = 0
        self._direct_transition_stage_transport_entry_count = 0
        self._direct_transition_batch_count = 0
        self._direct_transition_batch_entry_count = 0
        self._direct_transition_batch_transport_bytes = 0
        self._direct_transition_batch_build_sec = 0.0
        self._direct_transition_batch_submit_sec = 0.0
        self._direct_transition_batch_digest = ""
        self._direct_transition_legacy_entry_count = 0
        self._direct_transition_batch_dropped_pending_count = 0
        self._derived_transition_batch_count = 0
        self._derived_transition_batch_entry_count = 0
        self._derived_transition_batch_transport_bytes = 0
        self._derived_transition_batch_build_sec = 0.0
        self._derived_transition_batch_submit_sec = 0.0
        self._derived_transition_batch_digest = ""
        self._derived_transition_batch_dropped_pending_count = 0
        self._pending_compact_batch_sidecar_store_count = 0
        self._pending_compact_batch_sidecar_store_avoided_count = 0
        self._pending_root_batch_sidecar_store_count = 0
        self._pending_root_batch_sidecar_store_avoided_count = 0
        self._pending_action_step_identity_handle_store_count = 0
        self._pending_action_step_identity_handle_store_avoided_count = 0
        self._parent_previous_derived_transition_closure_count = 0
        self._parent_previous_derived_transition_avoided_count = 0
        self._owner_proxy_transition_closure_requested_count = 0
        self._owner_proxy_transition_closure_used_count = 0
        self._owner_proxy_transition_closure_batch_count = 0
        self._owner_proxy_transition_closure_entry_count = 0
        self._owner_proxy_transition_closure_transport_entry_count = 0
        self._owner_proxy_transition_closure_transport_bytes = 0
        self._owner_proxy_transition_closure_digest = ""
        self._owner_proxy_transition_closure_build_sec = 0.0
        self._owner_proxy_transition_closure_submit_sec = 0.0
        self._owner_proxy_transition_closure_pending_count = 0
        self._pending_direct_step_dispatch: _PendingOwnerSearchDirectStepDispatchV1 | None = None
        self._next_direct_step_dispatch_id = 0
        self._direct_step_dispatch_submit_count = 0
        self._direct_step_dispatch_resolve_count = 0
        self._direct_step_dispatch_sync_wrapper_count = 0
        self._direct_step_dispatch_max_pending_count = 0
        self._direct_step_dispatch_submit_wall_sec = 0.0
        self._direct_step_dispatch_resolve_wall_sec = 0.0
        self._direct_step_dispatch_submit_to_resolve_elapsed_sec = 0.0
        self._direct_step_dispatch_overlapped_parent_work_sec = 0.0
        self._owner_mechanics_last_step_frame_generation_consumed = -1
        self._owner_mechanics_last_step_frame_generation_by_slot: dict[int, int] = {}
        self._dropped_pending_searches = 0
        self._action_override_drop_count = 0
        self._closed = False

    @property
    def committed_index_rows(self) -> tuple[Any, ...]:
        return ()

    @property
    def committed_index_row_count(self) -> int:
        return 0

    @property
    def committed_index_group_count(self) -> int:
        return 0

    @property
    def dropped_pending_search_count(self) -> int:
        return int(self._dropped_pending_searches)

    @property
    def action_override_drop_count(self) -> int:
        return int(self._action_override_drop_count)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "compact_owner_search_slab_bypass": True,
            "compact_owner_search_slab_bypass_kind": (
                COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
            ),
            "compact_owner_search_resident_root_host_observation_stub_requested": bool(
                self.resident_root_host_observation_stub
            ),
            "compact_owner_search_direct_root_build_request_requested": bool(
                self.direct_root_build_request
            ),
            "compact_owner_search_owner_local_transition_derivation_requested": bool(
                self.owner_local_transition_derivation
            ),
            "compact_owner_search_owner_local_transition_derivation_used": bool(
                self._derived_transition_batch_entry_count > 0
                or self._owner_proxy_transition_closure_used_count > 0
            ),
            "compact_owner_search_owner_proxy_transition_closure_requested": bool(
                self.owner_proxy_transition_closure
            ),
            "compact_owner_search_owner_proxy_transition_closure_requested_count": int(
                self._owner_proxy_transition_closure_requested_count
            ),
            "compact_owner_search_owner_proxy_transition_closure_used": bool(
                self._owner_proxy_transition_closure_used_count > 0
            ),
            "compact_owner_search_owner_proxy_transition_closure_used_count": int(
                self._owner_proxy_transition_closure_used_count
            ),
            "compact_owner_search_parent_previous_transition_closure_count": int(
                self._parent_previous_derived_transition_closure_count
            ),
            "compact_owner_search_parent_previous_transition_closure_avoided_count": int(
                self._parent_previous_derived_transition_avoided_count
            ),
            "compact_owner_search_owner_local_transition_derivation_schema_id": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
                if self.owner_local_transition_derivation
                else ""
            ),
            "compact_owner_search_owner_local_transition_derivation_kind": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
                if self.owner_local_transition_derivation
                else ""
            ),
            "compact_owner_search_owner_local_transition_derivation_batch_count": int(
                self._owner_proxy_transition_closure_batch_count
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_count
            ),
            "compact_owner_search_owner_local_transition_derivation_transition_count": int(
                self._owner_proxy_transition_closure_entry_count
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_entry_count
            ),
            "compact_owner_search_owner_local_transition_derivation_transport_entry_count": int(
                self._owner_proxy_transition_closure_transport_entry_count
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_count
            ),
            "compact_owner_search_owner_local_transition_derivation_pending_count": int(
                self._owner_proxy_transition_closure_pending_count
                if self.owner_proxy_transition_closure
                else len(self._pending_derived_transition_facts)
            ),
            "compact_owner_search_owner_local_transition_derivation_transport_bytes": int(
                self._owner_proxy_transition_closure_transport_bytes
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_transport_bytes
            ),
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes": 0,
            "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count": 0,
            "compact_owner_search_owner_local_transition_derivation_digest": str(
                self._owner_proxy_transition_closure_digest
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_digest
            ),
            "compact_owner_search_owner_local_transition_derivation_digest_verified": bool(
                (
                    self._owner_proxy_transition_closure_digest
                    or self._owner_proxy_transition_closure_entry_count == 0
                )
                if self.owner_proxy_transition_closure
                else (
                    self._derived_transition_batch_digest
                    or self._derived_transition_batch_entry_count == 0
                )
            ),
            "compact_owner_search_owner_local_transition_derivation_build_sec": float(
                self._owner_proxy_transition_closure_build_sec
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_build_sec
            ),
            "compact_owner_search_owner_local_transition_derivation_submit_sec": float(
                self._owner_proxy_transition_closure_submit_sec
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_submit_sec
            ),
            "compact_owner_search_owner_local_transition_derivation_fallback_count": 0,
            "compact_owner_search_owner_local_transition_derivation_fallback_reason": "none",
            "compact_owner_search_owner_local_transition_derivation_dropped_pending_count": int(
                self._derived_transition_batch_dropped_pending_count
            ),
            "compact_owner_search_slab_bypass_stage_count": int(
                self._direct_transition_stage_count
            ),
            "compact_owner_search_slab_bypass_stage_entry_count": int(
                self._direct_transition_stage_entry_count
            ),
            "compact_owner_search_owner_replay_transport_kind": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
                if self.owner_local_transition_derivation
                else (
                    COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
                    if self.transition_batch_size > 1
                    else "per_transition_entry_v1"
                )
            ),
            "compact_owner_search_owner_replay_transition_batch_enabled": (
                self.transition_batch_size > 1
            ),
            "compact_owner_search_owner_replay_transport_entry_count": int(
                self._owner_proxy_transition_closure_transport_entry_count
                if self.owner_proxy_transition_closure
                else self._direct_transition_stage_transport_entry_count
            ),
            "compact_owner_search_owner_replay_transition_batch_count": int(
                self._owner_proxy_transition_closure_batch_count
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_count
                if self.owner_local_transition_derivation
                else self._direct_transition_batch_count
            ),
            "compact_owner_search_owner_replay_transition_batch_transition_count": int(
                self._owner_proxy_transition_closure_entry_count
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_entry_count
                if self.owner_local_transition_derivation
                else self._direct_transition_batch_entry_count
            ),
            "compact_owner_search_owner_replay_transition_legacy_entry_count": int(
                self._direct_transition_legacy_entry_count
            ),
            "compact_owner_search_transition_batch_transport_requested": (
                self.transition_batch_size > 1
            ),
            "compact_owner_search_transition_batch_transport_enabled": (
                self.transition_batch_size > 1
            ),
            "compact_owner_search_transition_batch_transport_kind": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
                if self.owner_local_transition_derivation
                else COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
                if self.transition_batch_size > 1
                else "per_transition_entry_v1"
            ),
            "compact_owner_search_transition_batch_schema_id": (
                COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
                if self.owner_local_transition_derivation
                else COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
                if self.transition_batch_size > 1
                else ""
            ),
            "compact_owner_search_transition_batch_count": int(
                self._owner_proxy_transition_closure_batch_count
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_count
                if self.owner_local_transition_derivation
                else self._direct_transition_batch_count
            ),
            "compact_owner_search_transition_batch_entry_count": int(
                self._owner_proxy_transition_closure_entry_count
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_entry_count
                if self.owner_local_transition_derivation
                else self._direct_transition_batch_entry_count
            ),
            "compact_owner_search_transition_batch_transport_entry_count": int(
                self._owner_proxy_transition_closure_transport_entry_count
                if self.owner_proxy_transition_closure
                else self._direct_transition_stage_transport_entry_count
            ),
            "compact_owner_search_transition_batch_max_entries_per_batch": int(
                self.transition_batch_size
            ),
            "compact_owner_search_transition_batch_fixed_capacity": int(
                self.transition_batch_size if self.transition_batch_size > 1 else 0
            ),
            "compact_owner_search_transition_batch_padding_count": 0,
            "compact_owner_search_transition_batch_overflow_count": 0,
            "compact_owner_search_transition_batch_fallback_count": 0,
            "compact_owner_search_transition_batch_fallback_reason": "none",
            "compact_owner_search_transition_batch_pending_count": int(
                self._owner_proxy_transition_closure_pending_count
                if self.owner_proxy_transition_closure
                else len(self._pending_derived_transition_facts)
                if self.owner_local_transition_derivation
                else len(self._pending_transition_facts)
            ),
            "compact_owner_search_transition_batch_dropped_pending_count": int(
                self._derived_transition_batch_dropped_pending_count
                if self.owner_local_transition_derivation
                else self._direct_transition_batch_dropped_pending_count
            ),
            "compact_owner_search_transition_batch_transport_bytes": int(
                self._owner_proxy_transition_closure_transport_bytes
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_transport_bytes
                if self.owner_local_transition_derivation
                else self._direct_transition_batch_transport_bytes
            ),
            "compact_owner_search_transition_batch_digest": str(
                self._owner_proxy_transition_closure_digest
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_digest
                if self.owner_local_transition_derivation
                else self._direct_transition_batch_digest
            ),
            "compact_owner_search_transition_batch_build_sec": float(
                self._owner_proxy_transition_closure_build_sec
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_build_sec
                if self.owner_local_transition_derivation
                else self._direct_transition_batch_build_sec
            ),
            "compact_owner_search_transition_batch_submit_sec": float(
                self._owner_proxy_transition_closure_submit_sec
                if self.owner_proxy_transition_closure
                else self._derived_transition_batch_submit_sec
                if self.owner_local_transition_derivation
                else self._direct_transition_batch_submit_sec
            ),
            "compact_owner_search_transition_batch_digest_verified": True,
            "compact_owner_search_pending_compact_batch_sidecar_stored": bool(
                self._pending is not None and self._pending.compact_batch is not None
            ),
            "compact_owner_search_pending_compact_batch_sidecar_storage_avoided": bool(
                self._pending_compact_batch_sidecar_store_avoided_count > 0
            ),
            "compact_owner_search_pending_compact_batch_sidecar_store_count": int(
                self._pending_compact_batch_sidecar_store_count
            ),
            "compact_owner_search_pending_compact_batch_sidecar_store_avoided_count": int(
                self._pending_compact_batch_sidecar_store_avoided_count
            ),
            "compact_owner_search_pending_root_batch_sidecar_stored": bool(
                self._pending is not None and self._pending.root_batch is not None
            ),
            "compact_owner_search_pending_root_batch_sidecar_storage_avoided": bool(
                self._pending_root_batch_sidecar_store_avoided_count > 0
            ),
            "compact_owner_search_pending_root_batch_sidecar_store_count": int(
                self._pending_root_batch_sidecar_store_count
            ),
            "compact_owner_search_pending_root_batch_sidecar_store_avoided_count": int(
                self._pending_root_batch_sidecar_store_avoided_count
            ),
            "compact_owner_search_pending_action_step_identity_handle_stored": bool(
                self._pending is not None and self._pending.action_step is not None
            ),
            "compact_owner_search_pending_action_step_identity_handle_storage_avoided": bool(
                self._pending_action_step_identity_handle_store_avoided_count > 0
            ),
            "compact_owner_search_pending_action_step_identity_handle_store_count": int(
                self._pending_action_step_identity_handle_store_count
            ),
            (
                "compact_owner_search_pending_action_step_identity_handle_store_"
                "avoided_count"
            ): int(self._pending_action_step_identity_handle_store_avoided_count),
            "compact_owner_search_pending_root_build_request_stored": False,
            "compact_owner_search_pending_root_action_context_stored": bool(
                self._pending_direct_step_dispatch is not None
                and self._pending_direct_step_dispatch.root_action_context is not None
            ),
            "compact_rollout_slab_action_dispatch_step_overlap_supported": bool(
                self.direct_root_build_request
                and callable(
                    getattr(
                        self.search_service,
                        "submit_action_step_from_root_build_request",
                        None,
                    )
                )
                and callable(getattr(self.search_service, "resolve_action_step_handle", None))
            ),
            "compact_rollout_slab_action_dispatch_step_overlap_pending_count": (
                1 if self._pending_direct_step_dispatch is not None else 0
            ),
            "compact_rollout_slab_action_dispatch_step_overlap_submit_count": int(
                self._direct_step_dispatch_submit_count
            ),
            "compact_rollout_slab_action_dispatch_step_overlap_resolve_count": int(
                self._direct_step_dispatch_resolve_count
            ),
            "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count": int(
                self._direct_step_dispatch_sync_wrapper_count
            ),
            "compact_rollout_slab_action_dispatch_step_overlap_max_pending_count": int(
                self._direct_step_dispatch_max_pending_count
            ),
            "compact_rollout_slab_action_dispatch_step_overlap_submit_wall_sec": float(
                self._direct_step_dispatch_submit_wall_sec
            ),
            "compact_rollout_slab_action_dispatch_step_overlap_resolve_wall_sec": float(
                self._direct_step_dispatch_resolve_wall_sec
            ),
            (
                "compact_rollout_slab_action_dispatch_step_overlap_submit_to_"
                "resolve_elapsed_sec"
            ): float(self._direct_step_dispatch_submit_to_resolve_elapsed_sec),
            (
                "compact_rollout_slab_action_dispatch_step_overlap_parent_work_"
                "sec"
            ): float(self._direct_step_dispatch_overlapped_parent_work_sec),
            "compact_owner_search_slab_bypass_parent_committed_index_rows": 0,
            "compact_owner_search_slab_bypass_parent_stored_index_rows": 0,
            "compact_rollout_slab_retains_committed_index_rows": False,
        }

    def submit_step(self, compact_batch: Any) -> CompactOwnerSearchDirectStepDispatchHandleV1:
        """Submit direct-root owner search while deferring the result wait."""

        if self._closed:
            raise ReplayCompatibilityError("compact owner-search direct stepper is closed")
        if not self.direct_root_build_request:
            raise ReplayCompatibilityError(
                "owner-search direct step dispatch requires direct_root_build_request"
            )
        if self._pending_direct_step_dispatch is not None:
            raise ReplayCompatibilityError(
                "owner-search direct step dispatch supports one pending step"
            )
        root_request_submitter = getattr(
            self.search_service,
            "submit_action_step_from_root_build_request",
            None,
        )
        transaction_submitter = getattr(
            self.search_service,
            "submit_owner_root_search_transaction_from_step_frame_slot",
            None,
        )
        resolver = getattr(self.search_service, "resolve_action_step_handle", None)
        if not callable(resolver) or not (
            callable(root_request_submitter) or callable(transaction_submitter)
        ):
            raise ReplayCompatibilityError(
                "owner-search direct step dispatch requires submit/resolve support"
            )
        self._validate_batch_shape(compact_batch)
        root_metadata = self._root_metadata_from_compact_batch(compact_batch)
        if not _search_service_supports_two_phase(self.search_service):
            raise ReplayCompatibilityError(
                "owner-search direct stepper requires two-phase action search"
            )
        observation_source = str(
            getattr(
                compact_batch,
                "observation_source",
                COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
            )
        )
        if observation_source != COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1:
            raise ReplayCompatibilityError(
                "direct root build request requires resident_device_v1 roots"
            )
        record_index = self._record_index
        request_started = time.perf_counter()
        root_request_metadata = {
            **root_metadata,
            "compact_owner_search_direct_root_build_request_handoff": True,
            "compact_owner_search_direct_root_parent_build_avoided": True,
        }
        ring_frame_used = bool(
            root_metadata.get("compact_owner_mechanics_step_frame_handle_ring_used", False)
        )
        use_owner_root_search_transaction = bool(
            ring_frame_used
            and callable(transaction_submitter)
            and (
                not self.owner_local_transition_derivation
                or self.owner_proxy_transition_closure
            )
        )
        if use_owner_root_search_transaction:
            root_build_request_sec = 0.0
            pending_compact_batch = None
            if self.owner_local_transition_derivation:
                commit_timing = _empty_commit_timing()
                commit_previous_sec = 0.0
            else:
                commit_started = time.perf_counter()
                commit_timing = self._commit_previous_transition(compact_batch)
                commit_previous_sec = _elapsed(commit_started)
            submit_started = time.perf_counter()
            transaction_dispatch = transaction_submitter(
                compact_batch,
                search_lane=self.search_lane,
                metadata=root_request_metadata,
                copy_observation=self.copy_root_observation,
                resident_host_observation_stub=True,
                close_previous_transition=bool(
                    self.owner_local_transition_derivation
                    and self.owner_proxy_transition_closure
                ),
                max_entries_per_batch=int(self.transition_batch_size),
                policy_source=self.policy_source,
            )
            action_dispatch_handle = transaction_dispatch.action_dispatch_handle
            root_action_context = None
            root_action_context_handle = getattr(
                transaction_dispatch,
                "root_action_context_handle",
                None,
            )
            if root_action_context_handle is None:
                raise ReplayCompatibilityError(
                    "owner root-search transaction requires root action context handle"
                )
            root_metadata = {
                **root_metadata,
                **dict(getattr(transaction_dispatch, "metadata", {}) or {}),
            }
            if self.owner_local_transition_derivation and self.owner_proxy_transition_closure:
                self._owner_proxy_transition_closure_requested_count += 1
                self._parent_previous_derived_transition_avoided_count += 1
                commit_timing = dict(transaction_dispatch.commit_timing)
                commit_previous_sec = sum(
                    float(commit_timing.get(key, 0.0))
                    for key in _COMPACT_ROLLOUT_SLAB_COMMIT_TIMING_FIELDS
                )
                self._refresh_owner_proxy_transition_closure_metadata()
            submit_wall_sec = _elapsed(submit_started)
        else:
            pending_compact_batch = compact_batch
            if ring_frame_used:
                root_build_request = _root_build_request_from_owner_step_frame_slot_v1(
                    compact_batch,
                    search_lane=self.search_lane,
                    metadata=root_request_metadata,
                    copy_observation=self.copy_root_observation,
                    resident_host_observation_stub=True,
                )
            else:
                root_build_request = compact_root_build_request_v1_from_batch(
                    compact_batch,
                    search_lane=self.search_lane,
                    metadata=root_request_metadata,
                    copy_observation=self.copy_root_observation,
                    observation_source=observation_source,
                    resident_observation=getattr(compact_batch, "resident_observation", None),
                    resident_host_observation_stub=True,
                )
            root_action_context = compact_root_action_context_v1_from_request(
                root_build_request
            )
            root_action_context_handle = None
            root_build_request_sec = _elapsed(request_started)
            if self.owner_local_transition_derivation:
                commit_timing = _empty_commit_timing()
                commit_previous_sec = 0.0
            else:
                commit_started = time.perf_counter()
                commit_timing = self._commit_previous_transition(compact_batch)
                commit_previous_sec = _elapsed(commit_started)
            if self.owner_local_transition_derivation:
                commit_started = time.perf_counter()
                if self.owner_proxy_transition_closure:
                    closure = getattr(
                        self.search_service,
                        "stage_owner_proxy_transition_from_root_build_request",
                        None,
                    )
                    if not callable(closure):
                        raise ReplayCompatibilityError(
                            "owner proxy transition closure requires proxy closure hook"
                        )
                    self._owner_proxy_transition_closure_requested_count += 1
                    self._parent_previous_derived_transition_avoided_count += 1
                    commit_timing = closure(
                        root_build_request,
                        max_entries_per_batch=self.transition_batch_size,
                        policy_source=self.policy_source,
                    )
                    self._refresh_owner_proxy_transition_closure_metadata()
                else:
                    commit_timing = self._stage_previous_derived_transition(
                        compact_batch,
                        next_root_build_request=root_build_request,
                    )
                commit_previous_sec = _elapsed(commit_started)
            if not callable(root_request_submitter):
                raise ReplayCompatibilityError(
                    "owner-search direct step dispatch requires root-request submit support"
                )
            submit_started = time.perf_counter()
            action_dispatch_handle = root_request_submitter(root_build_request)
            submit_wall_sec = _elapsed(submit_started)
        self._mark_owner_mechanics_step_frame_consumed(root_metadata)
        self._next_direct_step_dispatch_id += 1
        dispatch_id = int(self._next_direct_step_dispatch_id)
        handle = CompactOwnerSearchDirectStepDispatchHandleV1(
            schema_id=COMPACT_OWNER_SEARCH_DIRECT_STEP_DISPATCH_HANDLE_SCHEMA_ID,
            dispatch_id=dispatch_id,
            record_index=int(record_index),
            metadata={
                "compact_rollout_slab_action_dispatch_step_overlap_submitted": True,
                "compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait": True,
                "compact_rollout_slab_action_dispatch_step_overlap_record_index": int(
                    record_index
                ),
            },
        )
        self._pending_direct_step_dispatch = _PendingOwnerSearchDirectStepDispatchV1(
            handle=handle,
            action_dispatch_handle=action_dispatch_handle,
            compact_batch=pending_compact_batch,
            root_action_context=root_action_context,
            root_action_context_handle=root_action_context_handle,
            root_metadata=root_metadata,
            commit_timing=dict(commit_timing),
            commit_previous_sec=float(commit_previous_sec),
            root_build_request_sec=float(root_build_request_sec),
            submit_wall_sec=float(submit_wall_sec),
            opened=float(submit_started),
        )
        self._direct_step_dispatch_submit_count += 1
        self._direct_step_dispatch_submit_wall_sec += float(submit_wall_sec)
        self._direct_step_dispatch_max_pending_count = max(
            int(self._direct_step_dispatch_max_pending_count),
            1,
        )
        return handle

    def resolve_step(
        self,
        handle: CompactOwnerSearchDirectStepDispatchHandleV1,
        *,
        overlapped_parent_work_sec: float = 0.0,
        sync_wrapper: bool = False,
    ) -> CompactRolloutSlabStepV1:
        """Resolve a submitted direct-root owner action search."""

        if not isinstance(handle, CompactOwnerSearchDirectStepDispatchHandleV1):
            raise TypeError("handle must be CompactOwnerSearchDirectStepDispatchHandleV1")
        pending = self._pending_direct_step_dispatch
        if pending is None or int(pending.handle.dispatch_id) != int(handle.dispatch_id):
            raise ReplayCompatibilityError("unknown owner-search direct step dispatch handle")
        resolver = getattr(self.search_service, "resolve_action_step_handle", None)
        if not callable(resolver):
            raise ReplayCompatibilityError(
                "owner-search direct step dispatch requires resolve support"
            )
        resolve_started = time.perf_counter()
        action_step = resolver(
            pending.action_dispatch_handle,
            sync_wrapper=bool(sync_wrapper),
        )
        resolve_wall_sec = _elapsed(resolve_started)
        self._pending_direct_step_dispatch = None
        submit_to_resolve_elapsed_sec = _elapsed(pending.opened)
        self._direct_step_dispatch_resolve_count += 1
        if bool(sync_wrapper):
            self._direct_step_dispatch_sync_wrapper_count += 1
        self._direct_step_dispatch_resolve_wall_sec += float(resolve_wall_sec)
        self._direct_step_dispatch_submit_to_resolve_elapsed_sec += float(
            submit_to_resolve_elapsed_sec
        )
        self._direct_step_dispatch_overlapped_parent_work_sec += float(
            max(0.0, overlapped_parent_work_sec)
        )
        return self._finish_direct_root_action_step(
            compact_batch=pending.compact_batch,
            root_action_context=pending.root_action_context,
            root_action_context_handle=pending.root_action_context_handle,
            root_metadata=pending.root_metadata,
            action_step=action_step,
            record_index=int(pending.handle.record_index),
            commit_timing=pending.commit_timing,
            commit_previous_sec=float(pending.commit_previous_sec),
            root_build_request_sec=float(pending.root_build_request_sec),
            search_dispatch_sec=float(pending.submit_wall_sec + resolve_wall_sec),
            root_batch_build_sec=0.0,
            action_dispatch_step_overlap_used=not bool(sync_wrapper),
            action_dispatch_step_overlap_sync_wrapper=bool(sync_wrapper),
            action_dispatch_step_submit_sec=float(pending.submit_wall_sec),
            action_dispatch_step_resolve_sec=float(resolve_wall_sec),
            action_dispatch_step_submit_to_resolve_elapsed_sec=float(
                submit_to_resolve_elapsed_sec
            ),
            action_dispatch_step_overlapped_parent_work_sec=float(
                max(0.0, overlapped_parent_work_sec)
            ),
        )

    def _finish_direct_root_action_step(
        self,
        *,
        compact_batch: Any,
        root_action_context: CompactRootActionContextV1 | None,
        root_action_context_handle: Any | None = None,
        root_metadata: Mapping[str, Any],
        action_step: CompactSearchActionStepV1,
        record_index: int,
        commit_timing: Mapping[str, float],
        commit_previous_sec: float,
        root_build_request_sec: float,
        search_dispatch_sec: float,
        root_batch_build_sec: float,
        action_dispatch_step_overlap_used: bool = False,
        action_dispatch_step_overlap_sync_wrapper: bool = False,
        action_dispatch_step_submit_sec: float = 0.0,
        action_dispatch_step_resolve_sec: float = 0.0,
        action_dispatch_step_submit_to_resolve_elapsed_sec: float = 0.0,
        action_dispatch_step_overlapped_parent_work_sec: float = 0.0,
    ) -> CompactRolloutSlabStepV1:
        if not _action_step_owner_materializes_replay(action_step):
            raise ReplayCompatibilityError(
                "owner-search direct stepper requires owner-materialized replay"
            )
        owner_transaction_action_identity_verified = (
            action_step.metadata.get(
                "compact_owner_root_search_transaction_action_identity_verified"
            )
            is True
        )
        existing_action_metadata = dict(action_step.metadata)
        action_step.metadata.update(
            {
                **self.metadata,
                **dict(root_metadata),
                **existing_action_metadata,
                **(
                    {}
                    if root_action_context is None
                    else _root_build_request_scalar_metadata_v1(root_action_context)
                ),
                "compact_owner_search_direct_root_build_request_handoff": True,
                "compact_owner_search_direct_root_parent_build_avoided": True,
                "compact_owner_search_direct_root_parent_build_call_count": 0,
                "compact_owner_search_direct_root_parent_build_sec": 0.0,
                "compact_owner_search_direct_root_build_request_sec": float(
                    root_build_request_sec
                ),
                "compact_rollout_slab_parent_root_batch_build_avoided": True,
                "compact_rollout_slab_parent_root_batch_builder_used": False,
                "compact_rollout_slab_parent_root_batch_builder_call_count": 0,
                "compact_rollout_slab_return_root_batch_sidecar_stored": False,
                "compact_rollout_slab_return_root_batch_sidecar_storage_avoided": True,
                "compact_rollout_slab_return_root_batch_sidecar_build_count": 0,
                "compact_rollout_slab_action_dispatch_step_overlap_used": bool(
                    action_dispatch_step_overlap_used
                ),
                "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper": bool(
                    action_dispatch_step_overlap_sync_wrapper
                ),
                "compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper_count": int(
                    self._direct_step_dispatch_sync_wrapper_count
                ),
                "compact_rollout_slab_action_dispatch_step_overlap_submit_no_wait": bool(
                    action_dispatch_step_overlap_used
                ),
                "compact_rollout_slab_action_dispatch_step_overlap_submit_count": int(
                    self._direct_step_dispatch_submit_count
                ),
                "compact_rollout_slab_action_dispatch_step_overlap_resolve_count": int(
                    self._direct_step_dispatch_resolve_count
                ),
                "compact_rollout_slab_action_dispatch_step_overlap_pending_count": (
                    1 if self._pending_direct_step_dispatch is not None else 0
                ),
                "compact_rollout_slab_action_dispatch_step_overlap_max_pending_count": int(
                    self._direct_step_dispatch_max_pending_count
                ),
                "compact_rollout_slab_action_dispatch_step_overlap_submit_wall_sec": float(
                    action_dispatch_step_submit_sec
                ),
                "compact_rollout_slab_action_dispatch_step_overlap_resolve_wall_sec": float(
                    action_dispatch_step_resolve_sec
                ),
                (
                    "compact_rollout_slab_action_dispatch_step_overlap_submit_to_"
                    "resolve_elapsed_sec"
                ): float(action_dispatch_step_submit_to_resolve_elapsed_sec),
                (
                    "compact_rollout_slab_action_dispatch_step_overlap_parent_work_"
                    "sec"
                ): float(action_dispatch_step_overlapped_parent_work_sec),
            }
        )
        action_step.metadata["compact_rollout_slab_bypassed"] = True
        action_step.metadata["compact_rollout_slab_general_replay_row_builder_used"] = False
        action_step.metadata["compact_owner_search_action_only_result"] = True
        action_step.metadata["compact_owner_search_owner_materializes_replay"] = True
        identity_started = time.perf_counter()
        root_action_context_handle_used = root_action_context is None
        if root_action_context is None:
            if root_action_context_handle is None:
                raise ReplayCompatibilityError(
                    "owner-search direct step missing root action context handle"
                )
            if int(getattr(root_action_context_handle, "context_id", 0) or 0) <= 0:
                raise ReplayCompatibilityError(
                    "owner-search direct step root action context handle id is invalid"
                )
            if int(getattr(root_action_context_handle, "root_count", 0) or 0) <= 0:
                raise ReplayCompatibilityError(
                    "owner-search direct step root action context handle root count is invalid"
                )
            if (
                int(getattr(root_action_context_handle, "active_root_count", 0) or 0)
                <= 0
            ):
                raise ReplayCompatibilityError(
                    "owner-search direct step root action context handle active count is invalid"
                )
            if not str(getattr(root_action_context_handle, "context_digest", "") or ""):
                raise ReplayCompatibilityError(
                    "owner-search direct step root action context handle digest is missing"
                )
            if (
                not owner_transaction_action_identity_verified
                and action_step.metadata.get(
                    "compact_owner_root_search_transaction_action_identity_verified"
                )
                is not True
            ):
                raise ReplayCompatibilityError(
                    "owner-search direct step requires owner-verified action identity"
                )
            if owner_transaction_action_identity_verified:
                action_step.metadata[
                    "compact_owner_root_search_transaction_action_identity_verified"
                ] = True
            action_step.metadata["compact_owner_search_parent_action_context_validation_count"] = 0
            action_step.metadata["compact_owner_search_owner_action_context_validation_count"] = max(
                1,
                int(
                    action_step.metadata.get(
                        "compact_owner_search_owner_action_context_validation_count"
                    )
                    or 0
                ),
            )
        else:
            _validate_compact_search_action_step_identity_from_root_action_context(
                root_action_context,
                action_step,
            )
            action_step.metadata["compact_owner_search_parent_action_context_validation_count"] = 1
            action_step.metadata["compact_owner_search_owner_action_context_validation_count"] = int(
                action_step.metadata.get("compact_owner_search_owner_action_context_validation_count")
                or 0
            )
        identity_validation_sec = _elapsed(identity_started)
        joint_action_started = time.perf_counter()
        dense_joint_action_present = getattr(action_step, "dense_joint_action", None) is not None
        if root_action_context is None:
            next_joint_action = _validated_owner_verified_dense_joint_action_from_action_step(
                action_step,
                batch_size=self.batch_size,
                player_count=self.player_count,
            )
        else:
            next_joint_action = selected_joint_action_from_root_action_context_action_step(
                root_action_context,
                action_step,
                batch_size=self.batch_size,
                player_count=self.player_count,
            )
        joint_action_assembly_sec = _elapsed(joint_action_started)
        action_step.metadata.update(
            {
                "compact_owner_search_dense_joint_action_used": bool(
                    dense_joint_action_present
                ),
                "compact_owner_search_next_joint_action_published": bool(
                    dense_joint_action_present
                ),
                "compact_rollout_slab_parent_dense_action_reconstruction_count": (
                    0 if dense_joint_action_present else 1
                ),
                "compact_rollout_slab_parent_dense_action_reconstruction_used": (
                    not bool(dense_joint_action_present)
                ),
                "compact_rollout_slab_dense_joint_action_validation_count": (
                    1 if dense_joint_action_present else 0
                ),
                "compact_rollout_slab_next_joint_action_checksum": (
                    _owner_action_checksum_v1(next_joint_action)
                ),
                "compact_owner_search_pending_root_action_context_stored": bool(
                    root_action_context is not None
                ),
                "compact_owner_search_action_dispatch_pending_root_action_context_stored": bool(
                    root_action_context is not None
                ),
                "compact_owner_root_action_context_handle_used": bool(
                    root_action_context_handle_used
                ),
            }
        )
        pending_store_started = time.perf_counter()
        self._pending_compact_batch_sidecar_store_avoided_count += 1
        self._pending_root_batch_sidecar_store_avoided_count += 1
        if self.owner_proxy_transition_closure:
            self._pending = None
            self._pending_action_step_identity_handle_store_avoided_count += 1
            pending_action_step_identity_stored = False
        else:
            self._pending = _PendingCompactSearchV1(
                record_index=int(record_index),
                compact_batch=None,
                root_batch=None,
                search_result=None,
                action_step=action_step,
                next_joint_action=next_joint_action,
            )
            self._pending_action_step_identity_handle_store_count += 1
            pending_action_step_identity_stored = True
        action_step.metadata.update(
            {
                "compact_owner_search_pending_compact_batch_sidecar_stored": False,
                "compact_owner_search_pending_compact_batch_sidecar_storage_avoided": True,
                "compact_owner_search_pending_compact_batch_sidecar_store_count": int(
                    self._pending_compact_batch_sidecar_store_count
                ),
                "compact_owner_search_pending_compact_batch_sidecar_store_avoided_count": int(
                    self._pending_compact_batch_sidecar_store_avoided_count
                ),
                "compact_owner_search_pending_root_batch_sidecar_stored": False,
                "compact_owner_search_pending_root_batch_sidecar_storage_avoided": True,
                "compact_owner_search_pending_root_batch_sidecar_store_count": int(
                    self._pending_root_batch_sidecar_store_count
                ),
                "compact_owner_search_pending_root_batch_sidecar_store_avoided_count": int(
                    self._pending_root_batch_sidecar_store_avoided_count
                ),
                "compact_owner_search_pending_action_step_identity_handle_stored": (
                    pending_action_step_identity_stored
                ),
                "compact_owner_search_pending_action_step_identity_handle_storage_avoided": (
                    not pending_action_step_identity_stored
                ),
                "compact_owner_search_pending_action_step_identity_handle_store_count": int(
                    self._pending_action_step_identity_handle_store_count
                ),
                (
                    "compact_owner_search_pending_action_step_identity_handle_store_"
                    "avoided_count"
                ): int(self._pending_action_step_identity_handle_store_avoided_count),
                "compact_owner_search_pending_root_build_request_stored": False,
                "compact_owner_search_pending_root_action_context_stored": bool(
                    root_action_context is not None
                ),
            }
        )
        self._record_index += 1
        pending_store_sec = _elapsed(pending_store_started)
        commit_timing_dict = dict(commit_timing)
        internal_timing = {
            **commit_timing_dict,
            "compact_rollout_slab_commit_child_accounted_sec": sum(
                float(commit_timing_dict.get(key, 0.0))
                for key in _COMPACT_ROLLOUT_SLAB_COMMIT_TIMING_FIELDS
            ),
            "compact_rollout_slab_commit_residual_sec": (
                commit_previous_sec
                - sum(
                    float(commit_timing_dict.get(key, 0.0))
                    for key in _COMPACT_ROLLOUT_SLAB_COMMIT_TIMING_FIELDS
                )
            ),
            "compact_rollout_slab_commit_previous_sec": float(commit_previous_sec),
            "compact_rollout_slab_root_batch_build_sec": float(root_batch_build_sec),
            "compact_rollout_slab_root_build_request_sec": float(root_build_request_sec),
            "compact_rollout_slab_root_tape_record_sec": 0.0,
            "compact_rollout_slab_search_dispatch_wall_sec": float(search_dispatch_sec),
            "compact_rollout_slab_search_identity_validation_sec": identity_validation_sec,
            "compact_rollout_slab_joint_action_assembly_sec": joint_action_assembly_sec,
            "compact_rollout_slab_pending_store_sec": pending_store_sec,
            "compact_rollout_slab_action_dispatch_step_overlap_submit_wall_sec": float(
                action_dispatch_step_submit_sec
            ),
            "compact_rollout_slab_action_dispatch_step_overlap_resolve_wall_sec": float(
                action_dispatch_step_resolve_sec
            ),
            (
                "compact_rollout_slab_action_dispatch_step_overlap_submit_to_"
                "resolve_elapsed_sec"
            ): float(action_dispatch_step_submit_to_resolve_elapsed_sec),
            (
                "compact_rollout_slab_action_dispatch_step_overlap_parent_work_"
                "sec"
            ): float(action_dispatch_step_overlapped_parent_work_sec),
        }
        telemetry_started = time.perf_counter()
        telemetry = _slab_telemetry(
            None,
            None,
            action_step,
            None,
            root_action_context=root_action_context,
            action_feedback_mode=COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK,
            action_override_drop_count=self._action_override_drop_count,
            two_phase_search=True,
            replay_payload_flush_count=0,
            committed_replay_payload_flushed=False,
            committed_replay_payload_d2h_bytes=0,
            internal_timing=internal_timing,
        )
        telemetry.update(self.metadata)
        telemetry["compact_rollout_slab_bypassed"] = True
        telemetry["compact_rollout_slab_general_replay_row_builder_used"] = False
        telemetry["compact_rollout_slab_action_dispatch_step_overlap_used"] = bool(
            action_dispatch_step_overlap_used
        )
        telemetry["compact_rollout_slab_action_dispatch_step_overlap_sync_wrapper"] = bool(
            action_dispatch_step_overlap_sync_wrapper
        )
        for key in (
            "compact_owner_search_action_dispatch_handle_boundary_supported",
            "compact_owner_search_action_dispatch_handle_used",
            "compact_owner_search_action_dispatch_handle_schema_id",
            "compact_owner_search_action_dispatch_handle_id",
            "compact_owner_search_action_dispatch_handle_submit_no_wait",
            "compact_owner_search_action_dispatch_handle_sync_wrapper",
            "compact_owner_search_action_dispatch_handle_sync_wrapper_count",
            "compact_owner_search_action_dispatch_handle_completed_at_submit_count",
            "compact_owner_search_action_dispatch_handle_submit_count",
            "compact_owner_search_action_dispatch_handle_resolve_count",
            "compact_owner_search_action_dispatch_handle_pending_count",
            "compact_owner_search_action_dispatch_handle_max_pending_count",
            "compact_owner_search_action_dispatch_handle_result_wait_in_submit_count",
            "compact_owner_search_action_dispatch_handle_result_wait_sec",
            "compact_owner_root_action_context_handle_used",
            "compact_owner_root_action_context_handle_schema_id",
            "compact_owner_root_action_context_handle_id",
            "compact_owner_root_action_context_context_digest",
            "compact_owner_root_action_context_owner_store_count",
            "compact_owner_root_action_context_owner_resolve_count",
            "compact_owner_root_action_context_owner_release_count",
            "compact_owner_root_action_context_owner_pending_count",
            "compact_owner_root_action_context_owner_max_pending_count",
            "compact_owner_root_action_context_owner_digest_verified",
            "compact_owner_search_pending_root_action_context_stored",
            "compact_owner_search_action_dispatch_pending_root_action_context_stored",
            "compact_owner_search_action_dispatch_pending_root_action_context_store_count",
            (
                "compact_owner_search_action_dispatch_pending_root_action_context_"
                "avoided_count"
            ),
            "compact_owner_search_parent_action_context_validation_count",
            "compact_owner_search_owner_action_context_validation_count",
            "compact_owner_root_search_transaction_parent_root_action_context_stored",
            "compact_owner_root_search_transaction_parent_root_action_context_store_count",
            "compact_owner_root_search_transaction_parent_root_action_context_array_bytes",
            "compact_owner_root_search_transaction_parent_root_action_context_field_count",
        ):
            if key in action_step.metadata:
                telemetry[key] = action_step.metadata[key]
                search_metadata = telemetry.get("compact_rollout_slab_search_metadata")
                if isinstance(search_metadata, dict):
                    search_metadata[key] = action_step.metadata[key]
        telemetry["compact_rollout_slab_telemetry_build_sec"] = _elapsed(telemetry_started)
        telemetry["compact_rollout_slab_internal_accounted_sec"] = sum(
            float(telemetry.get(key, 0.0)) for key in _COMPACT_ROLLOUT_SLAB_INTERNAL_TIMING_FIELDS
        )
        return CompactRolloutSlabStepV1(
            schema_id=COMPACT_ROLLOUT_SLAB_STEP_SCHEMA_ID,
            record_index=int(record_index),
            compact_batch=compact_batch,
            root_batch=None,
            search_result=None,
            action_step=action_step,
            next_joint_action=next_joint_action,
            committed_index_rows=None,
            telemetry=telemetry,
        )

    def step(self, compact_batch: Any) -> CompactRolloutSlabStepV1:
        if self._closed:
            raise ReplayCompatibilityError("compact owner-search direct stepper is closed")
        if self.direct_root_build_request and callable(
            getattr(
                self.search_service,
                "submit_action_step_from_root_build_request",
                None,
            )
        ) and callable(getattr(self.search_service, "resolve_action_step_handle", None)):
            handle = self.submit_step(compact_batch)
            return self.resolve_step(handle, sync_wrapper=True)
        self._validate_batch_shape(compact_batch)
        root_metadata = self._root_metadata_from_compact_batch(compact_batch)
        if not _search_service_supports_two_phase(self.search_service):
            raise ReplayCompatibilityError(
                "owner-search direct stepper requires two-phase action search"
            )
        record_index = self._record_index
        root_build_request_sec = 0.0
        if self.owner_local_transition_derivation:
            commit_timing = _empty_commit_timing()
            commit_previous_sec = 0.0
        else:
            commit_started = time.perf_counter()
            commit_timing = self._commit_previous_transition(compact_batch)
            commit_previous_sec = _elapsed(commit_started)
        if self.direct_root_build_request:
            observation_source = str(
                getattr(
                    compact_batch,
                    "observation_source",
                    COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
                )
            )
            if observation_source != COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1:
                raise ReplayCompatibilityError(
                    "direct root build request requires resident_device_v1 roots"
                )
            request_runner = getattr(
                self.search_service,
                "run_action_step_from_root_build_request",
                None,
            )
            if not callable(request_runner):
                raise ReplayCompatibilityError(
                    "direct root build request requires run_action_step_from_root_build_request"
                )
            request_started = time.perf_counter()
            root_request_metadata = {
                **root_metadata,
                "compact_owner_search_direct_root_build_request_handoff": True,
                "compact_owner_search_direct_root_parent_build_avoided": True,
            }
            if bool(root_metadata.get("compact_owner_mechanics_step_frame_handle_ring_used", False)):
                root_build_request = _root_build_request_from_owner_step_frame_slot_v1(
                    compact_batch,
                    search_lane=self.search_lane,
                    metadata=root_request_metadata,
                    copy_observation=self.copy_root_observation,
                    resident_host_observation_stub=True,
                )
            else:
                root_build_request = compact_root_build_request_v1_from_batch(
                    compact_batch,
                    search_lane=self.search_lane,
                    metadata=root_request_metadata,
                    copy_observation=self.copy_root_observation,
                    observation_source=observation_source,
                    resident_observation=getattr(
                        compact_batch,
                        "resident_observation",
                        None,
                    ),
                    resident_host_observation_stub=True,
                )
            root_build_request_sec = _elapsed(request_started)
            if self.owner_local_transition_derivation:
                commit_started = time.perf_counter()
                if self.owner_proxy_transition_closure:
                    closure = getattr(
                        self.search_service,
                        "stage_owner_proxy_transition_from_root_build_request",
                        None,
                    )
                    if not callable(closure):
                        raise ReplayCompatibilityError(
                            "owner proxy transition closure requires proxy closure hook"
                        )
                    self._owner_proxy_transition_closure_requested_count += 1
                    self._parent_previous_derived_transition_avoided_count += 1
                    commit_timing = closure(
                        root_build_request,
                        max_entries_per_batch=self.transition_batch_size,
                        policy_source=self.policy_source,
                    )
                    self._refresh_owner_proxy_transition_closure_metadata()
                else:
                    commit_timing = self._stage_previous_derived_transition(
                        compact_batch,
                        next_root_build_request=root_build_request,
                    )
                commit_previous_sec = _elapsed(commit_started)
            search_dispatch_started = time.perf_counter()
            action_step = request_runner(root_build_request)
            search_dispatch_sec = _elapsed(search_dispatch_started)
            if not _action_step_owner_materializes_replay(action_step):
                raise ReplayCompatibilityError(
                    "owner-search direct stepper requires owner-materialized replay"
                )
            action_step.metadata.update(
                {
                    **root_metadata,
                    **self.metadata,
                    **_root_build_request_scalar_metadata_v1(root_build_request),
                    "compact_owner_search_direct_root_build_request_handoff": True,
                    "compact_owner_search_direct_root_parent_build_avoided": True,
                    "compact_owner_search_direct_root_parent_build_call_count": 0,
                    "compact_owner_search_direct_root_parent_build_sec": 0.0,
                    "compact_owner_search_direct_root_build_request_sec": float(
                        root_build_request_sec
                    ),
                    "compact_rollout_slab_parent_root_batch_build_avoided": True,
                    "compact_rollout_slab_parent_root_batch_builder_used": False,
                    "compact_rollout_slab_parent_root_batch_builder_call_count": 0,
                    "compact_rollout_slab_return_root_batch_sidecar_stored": False,
                    "compact_rollout_slab_return_root_batch_sidecar_storage_avoided": True,
                    "compact_rollout_slab_return_root_batch_sidecar_build_count": 0,
                }
            )
            action_step.metadata["compact_rollout_slab_bypassed"] = True
            action_step.metadata["compact_rollout_slab_general_replay_row_builder_used"] = False
            action_step.metadata["compact_owner_search_action_only_result"] = True
            action_step.metadata["compact_owner_search_owner_materializes_replay"] = True
            identity_started = time.perf_counter()
            _validate_compact_search_action_step_identity_from_root_build_request(
                root_build_request,
                action_step,
            )
            identity_validation_sec = _elapsed(identity_started)
            joint_action_started = time.perf_counter()
            dense_joint_action_present = (
                getattr(action_step, "dense_joint_action", None) is not None
            )
            next_joint_action = selected_joint_action_from_root_build_request_action_step(
                root_build_request,
                action_step,
                batch_size=self.batch_size,
                player_count=self.player_count,
            )
            joint_action_assembly_sec = _elapsed(joint_action_started)
            action_step.metadata.update(
                {
                    "compact_owner_search_dense_joint_action_used": bool(
                        dense_joint_action_present
                    ),
                    "compact_owner_search_next_joint_action_published": bool(
                        dense_joint_action_present
                    ),
                    "compact_rollout_slab_parent_dense_action_reconstruction_count": (
                        0 if dense_joint_action_present else 1
                    ),
                    "compact_rollout_slab_parent_dense_action_reconstruction_used": (
                        not bool(dense_joint_action_present)
                    ),
                    "compact_rollout_slab_dense_joint_action_validation_count": (
                        1 if dense_joint_action_present else 0
                    ),
                    "compact_rollout_slab_next_joint_action_checksum": (
                        _owner_action_checksum_v1(next_joint_action)
                    ),
                }
            )
            root_batch = None
            root_batch_build_sec = 0.0
        else:
            root_batch_started = time.perf_counter()
            root_batch = build_compact_root_batch_v1(
                compact_batch,
                search_lane=self.search_lane,
                metadata=root_metadata,
                copy_observation=self.copy_root_observation,
                observation_source=str(
                    getattr(
                        compact_batch,
                        "observation_source",
                        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
                    )
                ),
                resident_observation=getattr(
                    compact_batch,
                    "resident_observation",
                    None,
                ),
                resident_host_observation_stub=bool(self.resident_root_host_observation_stub),
            )
            root_batch_build_sec = _elapsed(root_batch_started)
            if self.owner_local_transition_derivation:
                commit_started = time.perf_counter()
                commit_timing = self._stage_previous_derived_transition(
                    compact_batch,
                    next_root_batch=root_batch,
                )
                commit_previous_sec = _elapsed(commit_started)
            search_dispatch_started = time.perf_counter()
            action_step = self.search_service.run_action_step(root_batch)
            search_dispatch_sec = _elapsed(search_dispatch_started)
            if not _action_step_owner_materializes_replay(action_step):
                raise ReplayCompatibilityError(
                    "owner-search direct stepper requires owner-materialized replay"
                )
            action_step.metadata.update(self.metadata)
            action_step.metadata["compact_rollout_slab_bypassed"] = True
            action_step.metadata["compact_rollout_slab_general_replay_row_builder_used"] = False
            action_step.metadata["compact_owner_search_action_only_result"] = True
            action_step.metadata["compact_owner_search_owner_materializes_replay"] = True
            identity_started = time.perf_counter()
            _validate_compact_search_action_step_identity(root_batch, action_step)
            identity_validation_sec = _elapsed(identity_started)
            joint_action_started = time.perf_counter()
            next_joint_action = selected_joint_action_from_action_step(
                root_batch,
                action_step,
                batch_size=self.batch_size,
                player_count=self.player_count,
            )
            joint_action_assembly_sec = _elapsed(joint_action_started)
        self._mark_owner_mechanics_step_frame_consumed(root_metadata)
        pending_store_started = time.perf_counter()
        self._pending_compact_batch_sidecar_store_avoided_count += 1
        self._pending_root_batch_sidecar_store_avoided_count += 1
        if self.owner_proxy_transition_closure:
            self._pending = None
            self._pending_action_step_identity_handle_store_avoided_count += 1
            pending_action_step_identity_stored = False
        else:
            self._pending = _PendingCompactSearchV1(
                record_index=record_index,
                compact_batch=None,
                root_batch=None,
                search_result=None,
                action_step=action_step,
                next_joint_action=next_joint_action,
            )
            self._pending_action_step_identity_handle_store_count += 1
            pending_action_step_identity_stored = True
        action_step.metadata.update(
            {
                "compact_owner_search_pending_compact_batch_sidecar_stored": False,
                "compact_owner_search_pending_compact_batch_sidecar_storage_avoided": True,
                "compact_owner_search_pending_compact_batch_sidecar_store_count": int(
                    self._pending_compact_batch_sidecar_store_count
                ),
                "compact_owner_search_pending_compact_batch_sidecar_store_avoided_count": int(
                    self._pending_compact_batch_sidecar_store_avoided_count
                ),
                "compact_owner_search_pending_root_batch_sidecar_stored": False,
                "compact_owner_search_pending_root_batch_sidecar_storage_avoided": True,
                "compact_owner_search_pending_root_batch_sidecar_store_count": int(
                    self._pending_root_batch_sidecar_store_count
                ),
                "compact_owner_search_pending_root_batch_sidecar_store_avoided_count": int(
                    self._pending_root_batch_sidecar_store_avoided_count
                ),
                "compact_owner_search_pending_action_step_identity_handle_stored": (
                    pending_action_step_identity_stored
                ),
                "compact_owner_search_pending_action_step_identity_handle_storage_avoided": (
                    not pending_action_step_identity_stored
                ),
                "compact_owner_search_pending_action_step_identity_handle_store_count": int(
                    self._pending_action_step_identity_handle_store_count
                ),
                (
                    "compact_owner_search_pending_action_step_identity_handle_store_"
                    "avoided_count"
                ): int(self._pending_action_step_identity_handle_store_avoided_count),
                "compact_owner_search_pending_root_build_request_stored": False,
            }
        )
        self._record_index += 1
        pending_store_sec = _elapsed(pending_store_started)
        internal_timing = {
            **commit_timing,
            "compact_rollout_slab_commit_child_accounted_sec": sum(
                float(commit_timing.get(key, 0.0))
                for key in _COMPACT_ROLLOUT_SLAB_COMMIT_TIMING_FIELDS
            ),
            "compact_rollout_slab_commit_residual_sec": (
                commit_previous_sec
                - sum(
                    float(commit_timing.get(key, 0.0))
                    for key in _COMPACT_ROLLOUT_SLAB_COMMIT_TIMING_FIELDS
                )
            ),
            "compact_rollout_slab_commit_previous_sec": commit_previous_sec,
            "compact_rollout_slab_root_batch_build_sec": root_batch_build_sec,
            "compact_rollout_slab_root_build_request_sec": root_build_request_sec,
            "compact_rollout_slab_root_tape_record_sec": 0.0,
            "compact_rollout_slab_search_dispatch_wall_sec": search_dispatch_sec,
            "compact_rollout_slab_search_identity_validation_sec": (identity_validation_sec),
            "compact_rollout_slab_joint_action_assembly_sec": joint_action_assembly_sec,
            "compact_rollout_slab_pending_store_sec": pending_store_sec,
        }
        telemetry_started = time.perf_counter()
        telemetry = _slab_telemetry(
            root_batch,
            None,
            action_step,
            None,
            root_build_request=root_build_request if self.direct_root_build_request else None,
            action_feedback_mode=COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK,
            action_override_drop_count=self._action_override_drop_count,
            two_phase_search=True,
            replay_payload_flush_count=0,
            committed_replay_payload_flushed=False,
            committed_replay_payload_d2h_bytes=0,
            internal_timing=internal_timing,
        )
        telemetry.update(self.metadata)
        telemetry["compact_rollout_slab_bypassed"] = True
        telemetry["compact_rollout_slab_general_replay_row_builder_used"] = False
        telemetry["compact_rollout_slab_telemetry_build_sec"] = _elapsed(telemetry_started)
        telemetry["compact_rollout_slab_internal_accounted_sec"] = sum(
            float(telemetry.get(key, 0.0)) for key in _COMPACT_ROLLOUT_SLAB_INTERNAL_TIMING_FIELDS
        )
        return CompactRolloutSlabStepV1(
            schema_id=COMPACT_ROLLOUT_SLAB_STEP_SCHEMA_ID,
            record_index=record_index,
            compact_batch=compact_batch,
            root_batch=root_batch,
            search_result=None,
            action_step=action_step,
            next_joint_action=next_joint_action,
            committed_index_rows=None,
            telemetry=telemetry,
        )

    def flush_final(
        self,
        final_batch: Any | None = None,
    ) -> None:
        if self._pending_direct_step_dispatch is not None:
            raise ReplayCompatibilityError("owner-search direct step dispatch pending at close")
        if self.owner_proxy_transition_closure and final_batch is not None:
            raise ReplayCompatibilityError(
                "owner proxy transition closure does not support parent final flush"
            )
        if self._pending is None:
            if final_batch is None:
                self._drop_pending_transition_facts()
                self._drop_pending_derived_transition_facts()
            else:
                if self.owner_local_transition_derivation:
                    self._flush_derived_transition_batch(force=True)
                else:
                    self._flush_transition_batch(force=True)
            return None
        if final_batch is None:
            self._pending = None
            self._dropped_pending_searches += 1
            self._drop_pending_transition_facts()
            self._drop_pending_derived_transition_facts()
            return None
        self._validate_batch_shape(final_batch)
        if self.owner_local_transition_derivation:
            root_batch = build_compact_root_batch_v1(
                final_batch,
                search_lane=self.search_lane,
                metadata={
                    "compact_rollout_slab": True,
                    "compact_owner_search_slab_bypass": True,
                    "compact_owner_search_slab_bypass_kind": (
                        COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
                    ),
                    "compact_owner_search_owner_local_transition_derivation_final_flush": True,
                },
                copy_observation=self.copy_root_observation,
                observation_source=str(
                    getattr(
                        final_batch,
                        "observation_source",
                        COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
                    )
                ),
                resident_observation=getattr(final_batch, "resident_observation", None),
                resident_host_observation_stub=bool(self.resident_root_host_observation_stub),
            )
            self._stage_previous_derived_transition(final_batch, next_root_batch=root_batch)
            self._flush_derived_transition_batch(force=True)
        else:
            self._commit_previous_transition(final_batch)
            self._flush_transition_batch(force=True)
        return None

    def close(self, final_batch: Any | None = None) -> None:
        try:
            self.flush_final(final_batch)
        finally:
            self._closed = True
        return None

    def _validate_batch_shape(self, compact_batch: Any) -> None:
        expected = (self.batch_size, self.player_count)
        observation_source = str(
            getattr(
                compact_batch,
                "observation_source",
                COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
            )
        )
        if (
            self.resident_root_host_observation_stub
            and observation_source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1
        ):
            resident = getattr(compact_batch, "resident_observation", None)
            if resident is None:
                raise ReplayCompatibilityError(
                    "resident root host-observation stub requires resident_observation"
                )
            if (
                int(getattr(resident, "batch_size")) != self.batch_size
                or int(getattr(resident, "player_count")) != self.player_count
            ):
                raise ReplayCompatibilityError(
                    "owner-search direct stepper resident shape mismatch"
                )
        else:
            observation = np.asarray(compact_batch.observation)
            if observation.ndim != 5 or observation.shape[:2] != expected:
                raise ReplayCompatibilityError(
                    "owner-search direct stepper observation shape mismatch"
                )
        if np.asarray(compact_batch.joint_action).shape != expected:
            raise ReplayCompatibilityError(
                "owner-search direct stepper joint_action shape mismatch"
            )

    def _root_metadata_from_compact_batch(self, compact_batch: Any) -> dict[str, Any]:
        root_metadata: dict[str, Any] = {
            "compact_rollout_slab": True,
            "compact_owner_search_slab_bypass": True,
            "compact_owner_search_slab_bypass_kind": (
                COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION
            ),
        }
        root_metadata.update(
            _compact_policy_refresh_metadata_from_search_service(self.search_service)
        )
        batch_metadata = getattr(compact_batch, "metadata", None)
        if isinstance(batch_metadata, Mapping):
            root_metadata.update(
                {
                    str(key): value
                    for key, value in batch_metadata.items()
                    if value is None or isinstance(value, (bool, int, float, str))
                }
            )
        root_metadata.update(_owner_mechanics_step_frame_handle_metadata_v1(compact_batch))
        self._validate_owner_mechanics_step_frame_unconsumed(root_metadata)
        return root_metadata

    def _validate_owner_mechanics_step_frame_unconsumed(
        self,
        root_metadata: Mapping[str, Any],
    ) -> None:
        if not bool(
            root_metadata.get("compact_owner_mechanics_step_frame_handle_ring_used", False)
        ):
            return
        slot_id = int(root_metadata.get("compact_owner_mechanics_step_frame_handle_slot_id", -1))
        generation = int(
            root_metadata.get("compact_owner_mechanics_step_frame_handle_generation", -1)
        )
        ring_slot_count = int(
            root_metadata.get(
                "compact_owner_mechanics_step_frame_handle_ring_slot_count",
                0,
            )
            or 0
        )
        if ring_slot_count <= 0 or slot_id < 0 or slot_id >= ring_slot_count:
            raise ReplayCompatibilityError("owner mechanics step frame handle slot mismatch")
        if generation < 0:
            raise ReplayCompatibilityError(
                "owner mechanics step frame handle generation mismatch"
            )
        if generation % ring_slot_count != slot_id:
            raise ReplayCompatibilityError(
                "owner mechanics step frame generation slot mismatch"
            )
        if generation <= int(self._owner_mechanics_last_step_frame_generation_consumed):
            raise ReplayCompatibilityError("owner mechanics step frame stale generation")
        slot_last_generation = int(
            self._owner_mechanics_last_step_frame_generation_by_slot.get(slot_id, -1)
        )
        if generation <= slot_last_generation:
            raise ReplayCompatibilityError("owner mechanics step frame slot reuse")

    def _mark_owner_mechanics_step_frame_consumed(
        self,
        root_metadata: Mapping[str, Any],
    ) -> None:
        if not bool(
            root_metadata.get("compact_owner_mechanics_step_frame_handle_ring_used", False)
        ):
            return
        slot_id = int(root_metadata["compact_owner_mechanics_step_frame_handle_slot_id"])
        generation = int(
            root_metadata["compact_owner_mechanics_step_frame_handle_generation"]
        )
        self._owner_mechanics_last_step_frame_generation_consumed = max(
            int(self._owner_mechanics_last_step_frame_generation_consumed),
            generation,
        )
        self._owner_mechanics_last_step_frame_generation_by_slot[slot_id] = max(
            int(self._owner_mechanics_last_step_frame_generation_by_slot.get(slot_id, -1)),
            generation,
        )

    def _commit_previous_transition(self, next_batch: Any) -> dict[str, float]:
        commit_timing = _empty_commit_timing()
        pending = self._pending
        if pending is None:
            return commit_timing
        next_joint_action = np.asarray(next_batch.joint_action, dtype=np.int16)
        action_check_started = time.perf_counter()
        active_env_row, active_player, active_selected = _pending_action_arrays(pending)
        if active_selected.size and not np.array_equal(
            next_joint_action[active_env_row, active_player],
            active_selected,
        ):
            raise ReplayCompatibilityError(
                "owner-search direct stepper next batch did not apply staged actions"
            )
        commit_timing["compact_rollout_slab_commit_action_check_sec"] = _elapsed(
            action_check_started
        )
        action_step = pending.action_step
        if action_step is None:
            raise ReplayCompatibilityError("pending owner-search direct step has no action step")
        if not _action_step_owner_materializes_replay(action_step):
            raise ReplayCompatibilityError(
                "owner-search direct stepper cannot materialize parent replay rows"
            )
        transition_started = time.perf_counter()
        facts = _owner_search_replay_append_transition_facts(
            pending,
            action_step=action_step,
            next_joint_action=next_joint_action,
            next_batch=next_batch,
            policy_source=self.policy_source,
        )
        stage = getattr(self.search_service, "stage_replay_append_entries", None)
        if not callable(stage):
            raise ReplayCompatibilityError(
                "owner-search direct stepper requires stage_replay_append_entries"
            )
        if self.transition_batch_size <= 1:
            entry = _owner_search_replay_append_transition_entry_from_facts(facts)
            submit_started = time.perf_counter()
            staged_count = int(stage(entry) or 0)
            self._direct_transition_stage_count += 1
            self._direct_transition_stage_transport_entry_count += 1
            self._direct_transition_legacy_entry_count += int(staged_count)
            self._direct_transition_stage_entry_count += int(staged_count)
            submit_sec = _elapsed(submit_started)
        else:
            self._pending_transition_facts.append(facts)
            staged_count = self._flush_transition_batch(force=False)
            submit_sec = float(
                commit_timing.get(
                    "compact_rollout_slab_owner_replay_transition_batch_submit_sec",
                    0.0,
                )
            )
        self._pending = None
        commit_timing["compact_rollout_slab_replay_index_rows_build_sec"] = 0.0
        commit_timing["compact_rollout_slab_replay_index_rows_store_sec"] = _elapsed(
            transition_started
        )
        commit_timing["compact_rollout_slab_owner_replay_transition_stage_sec"] = commit_timing[
            "compact_rollout_slab_replay_index_rows_store_sec"
        ]
        commit_timing["compact_rollout_slab_owner_replay_transition_stage_count"] = float(
            1 if staged_count else 0
        )
        commit_timing["compact_rollout_slab_owner_replay_transition_stage_entry_count"] = float(
            staged_count
        )
        commit_timing[
            "compact_rollout_slab_owner_replay_transition_stage_transport_entry_count"
        ] = float(1 if staged_count else 0)
        commit_timing["compact_rollout_slab_owner_replay_transition_batch_submit_sec"] = float(
            submit_sec
        )
        return commit_timing

    def _refresh_owner_proxy_transition_closure_metadata(self) -> None:
        metadata_obj = getattr(self.search_service, "metadata", None)
        if not isinstance(metadata_obj, Mapping):
            return
        metadata = dict(metadata_obj)
        self._owner_proxy_transition_closure_used_count = int(
            metadata.get(
                "compact_owner_search_owner_proxy_transition_closure_closed_count",
                self._owner_proxy_transition_closure_used_count,
            )
            or 0
        )
        self._owner_proxy_transition_closure_batch_count = int(
            metadata.get(
                "compact_owner_search_owner_proxy_transition_closure_batch_count",
                self._owner_proxy_transition_closure_batch_count,
            )
            or 0
        )
        self._owner_proxy_transition_closure_entry_count = int(
            metadata.get(
                "compact_owner_search_owner_proxy_transition_closure_transition_count",
                self._owner_proxy_transition_closure_entry_count,
            )
            or 0
        )
        self._owner_proxy_transition_closure_transport_entry_count = int(
            metadata.get(
                "compact_owner_search_owner_proxy_transition_closure_transport_entry_count",
                self._owner_proxy_transition_closure_transport_entry_count,
            )
            or 0
        )
        self._owner_proxy_transition_closure_transport_bytes = int(
            metadata.get(
                "compact_owner_search_owner_proxy_transition_closure_transport_bytes",
                self._owner_proxy_transition_closure_transport_bytes,
            )
            or 0
        )
        self._owner_proxy_transition_closure_digest = str(
            metadata.get(
                "compact_owner_search_owner_proxy_transition_closure_digest",
                self._owner_proxy_transition_closure_digest,
            )
            or ""
        )
        self._owner_proxy_transition_closure_build_sec = float(
            metadata.get(
                "compact_owner_search_owner_proxy_transition_closure_build_sec",
                self._owner_proxy_transition_closure_build_sec,
            )
            or 0.0
        )
        self._owner_proxy_transition_closure_submit_sec = float(
            metadata.get(
                "compact_owner_search_owner_proxy_transition_closure_submit_sec",
                self._owner_proxy_transition_closure_submit_sec,
            )
            or 0.0
        )
        self._owner_proxy_transition_closure_pending_count = int(
            metadata.get(
                "compact_owner_search_owner_proxy_transition_closure_pending_count",
                self._owner_proxy_transition_closure_pending_count,
            )
            or 0
        )

    def _stage_previous_derived_transition(
        self,
        next_batch: Any,
        *,
        next_root_batch: CompactRootBatchV1 | None = None,
        next_root_build_request: CompactRootBuildRequestV1 | None = None,
    ) -> dict[str, float]:
        self._parent_previous_derived_transition_closure_count += 1
        commit_timing = _empty_commit_timing()
        pending = self._pending
        if pending is None:
            return commit_timing
        if (next_root_batch is None) == (next_root_build_request is None):
            raise ReplayCompatibilityError(
                "owner-search derived transition requires exactly one next root source"
            )
        outcome_started = time.perf_counter()
        if next_root_build_request is not None:
            compact_transition_outcome_v1_from_root_build_request(
                next_root_build_request,
                batch_size=self.batch_size,
                player_count=self.player_count,
            )
        else:
            compact_transition_outcome_v1_from_next_root_batch(
                next_root_batch,
                batch_size=self.batch_size,
                player_count=self.player_count,
            )
        commit_timing["compact_rollout_slab_replay_index_rows_build_sec"] = _elapsed(
            outcome_started
        )
        next_joint_action = np.asarray(next_batch.joint_action, dtype=np.int16)
        action_check_started = time.perf_counter()
        active_env_row, active_player, active_selected = _pending_action_arrays(pending)
        applied = np.asarray(
            next_joint_action[active_env_row, active_player],
            dtype=np.int16,
        ).reshape(-1)
        if active_selected.size and not np.array_equal(applied, active_selected):
            raise ReplayCompatibilityError(
                "owner-search derived transition next batch did not apply staged actions"
            )
        commit_timing["compact_rollout_slab_commit_action_check_sec"] = _elapsed(
            action_check_started
        )
        action_step = pending.action_step
        if action_step is None:
            raise ReplayCompatibilityError("pending owner-search direct step has no action step")
        if not _action_step_owner_materializes_replay(action_step):
            raise ReplayCompatibilityError(
                "owner-search derived transition requires owner-materialized replay"
            )
        transition_started = time.perf_counter()
        facts = _owner_search_replay_append_derived_transition_facts(
            pending,
            action_step=action_step,
            applied_action=applied,
            policy_source=self.policy_source,
        )
        self._pending_derived_transition_facts.append(facts)
        staged_count = self._flush_derived_transition_batch(force=False)
        self._pending = None
        commit_timing["compact_rollout_slab_replay_index_rows_store_sec"] = _elapsed(
            transition_started
        )
        commit_timing["compact_rollout_slab_owner_replay_transition_stage_sec"] = commit_timing[
            "compact_rollout_slab_replay_index_rows_store_sec"
        ]
        commit_timing["compact_rollout_slab_owner_replay_transition_stage_count"] = float(
            1 if staged_count else 0
        )
        commit_timing["compact_rollout_slab_owner_replay_transition_stage_entry_count"] = float(
            staged_count
        )
        commit_timing[
            "compact_rollout_slab_owner_replay_transition_stage_transport_entry_count"
        ] = float(1 if staged_count else 0)
        commit_timing["compact_rollout_slab_owner_replay_transition_batch_submit_sec"] = float(
            self._derived_transition_batch_submit_sec
        )
        return commit_timing

    def _flush_transition_batch(self, *, force: bool) -> int:
        if self.transition_batch_size <= 1:
            return 0
        count = len(self._pending_transition_facts)
        if count <= 0:
            return 0
        if not bool(force) and count < self.transition_batch_size:
            return 0
        stage = getattr(self.search_service, "stage_replay_append_entries", None)
        if not callable(stage):
            raise ReplayCompatibilityError(
                "owner-search direct stepper requires stage_replay_append_entries"
            )
        batch_started = time.perf_counter()
        facts = tuple(self._pending_transition_facts[: self.transition_batch_size])
        batch = _owner_search_replay_append_transition_batch(
            facts,
            max_entries_per_batch=self.transition_batch_size,
        )
        build_sec = _elapsed(batch_started)
        submit_started = time.perf_counter()
        staged_count = int(stage(batch) or 0)
        submit_sec = _elapsed(submit_started)
        del self._pending_transition_facts[: len(facts)]
        self._direct_transition_stage_count += 1
        self._direct_transition_stage_entry_count += int(staged_count)
        self._direct_transition_stage_transport_entry_count += 1
        self._direct_transition_batch_count += 1
        self._direct_transition_batch_entry_count += int(staged_count)
        self._direct_transition_batch_transport_bytes += int(
            batch.metadata.get("compact_owner_search_transition_batch_transport_bytes") or 0
        )
        self._direct_transition_batch_digest = str(
            batch.metadata.get("compact_owner_search_transition_batch_digest") or ""
        )
        self._direct_transition_batch_build_sec += float(build_sec)
        self._direct_transition_batch_submit_sec += float(submit_sec)
        return int(staged_count)

    def _flush_derived_transition_batch(self, *, force: bool) -> int:
        count = len(self._pending_derived_transition_facts)
        if count <= 0:
            return 0
        if not bool(force) and count < self.transition_batch_size:
            return 0
        stage = getattr(self.search_service, "stage_replay_append_entries", None)
        if not callable(stage):
            raise ReplayCompatibilityError(
                "owner-search derived transition requires stage_replay_append_entries"
            )
        batch_started = time.perf_counter()
        facts = tuple(self._pending_derived_transition_facts[: self.transition_batch_size])
        batch = _owner_search_replay_append_derived_transition_batch(
            facts,
            max_entries_per_batch=self.transition_batch_size,
        )
        build_sec = _elapsed(batch_started)
        submit_started = time.perf_counter()
        staged_count = int(stage(batch) or 0)
        submit_sec = _elapsed(submit_started)
        del self._pending_derived_transition_facts[: len(facts)]
        self._direct_transition_stage_count += 1
        self._direct_transition_stage_entry_count += int(staged_count)
        self._direct_transition_stage_transport_entry_count += 1
        self._derived_transition_batch_count += 1
        self._derived_transition_batch_entry_count += int(staged_count)
        self._derived_transition_batch_transport_bytes += int(
            batch.metadata.get(
                "compact_owner_search_owner_local_transition_derivation_transport_bytes"
            )
            or 0
        )
        self._derived_transition_batch_digest = str(
            batch.metadata.get("compact_owner_search_owner_local_transition_derivation_digest")
            or ""
        )
        self._derived_transition_batch_build_sec += float(build_sec)
        self._derived_transition_batch_submit_sec += float(submit_sec)
        return int(staged_count)

    def _drop_pending_transition_facts(self) -> None:
        dropped = len(self._pending_transition_facts)
        if dropped <= 0:
            return
        self._direct_transition_batch_dropped_pending_count += int(dropped)
        self._pending_transition_facts.clear()

    def _drop_pending_derived_transition_facts(self) -> None:
        dropped = len(self._pending_derived_transition_facts)
        if dropped <= 0:
            return
        self._derived_transition_batch_dropped_pending_count += int(dropped)
        self._pending_derived_transition_facts.clear()


def _compact_terminal_metadata_from_batch(batch: Any) -> dict[str, Any]:
    done = np.asarray(getattr(batch, "done"), dtype=np.bool_)
    terminated = np.asarray(getattr(batch, "terminated", done), dtype=np.bool_)
    truncated = np.asarray(
        getattr(batch, "truncated", np.zeros_like(done, dtype=np.bool_)),
        dtype=np.bool_,
    )
    metadata: dict[str, Any] = {
        "compact_terminal_metadata_variant": "counts_checksums_v1",
        "next_done_row_count": int(done.sum()),
        "next_terminated_row_count": int(terminated.sum()),
        "next_truncated_row_count": int(truncated.sum()),
    }
    for field, dtype in (
        ("terminal_reason", np.int16),
        ("death_count", np.int32),
        ("death_player", np.int16),
        ("death_cause", np.int16),
        ("death_hit_owner", np.int16),
        ("winner", np.int16),
        ("draw", np.bool_),
    ):
        if hasattr(batch, field):
            value = np.asarray(getattr(batch, field), dtype=dtype)
            metadata[f"next_{field}_shape"] = [int(dim) for dim in value.shape]
            metadata[f"next_{field}_checksum"] = _terminal_sidecar_checksum(value)
            metadata[f"next_{field}_nonzero_count"] = int(np.count_nonzero(value))
            if field == "death_count":
                metadata["next_death_row_count"] = int((value > 0).sum())
                metadata["next_death_count_total"] = int(value.sum())
    return metadata


def _terminal_sidecar_checksum(value: np.ndarray) -> int:
    if value.dtype == np.bool_:
        return int(value.astype(np.int64, copy=False).sum())
    return int(np.asarray(value, dtype=np.int64).sum())


def selected_joint_action_from_search_result(
    root_batch: CompactRootBatchV1,
    search_result: CompactSearchResultV1,
    *,
    batch_size: int,
    player_count: int,
    inactive_action: int = 0,
) -> np.ndarray:
    """Map active-root selected actions back to dense joint actions."""

    return _selected_joint_action_from_compact_sidecars(
        root_batch,
        root_index=search_result.root_index,
        env_row=search_result.env_row,
        player=search_result.player,
        selected_action=search_result.selected_action,
        batch_size=batch_size,
        player_count=player_count,
        inactive_action=inactive_action,
    )


def selected_joint_action_from_action_step(
    root_batch: CompactRootBatchV1,
    action_step: CompactSearchActionStepV1,
    *,
    batch_size: int,
    player_count: int,
    inactive_action: int = 0,
) -> np.ndarray:
    """Map action-critical two-phase search output back to dense joint actions."""

    return _selected_joint_action_from_compact_sidecars(
        root_batch,
        root_index=action_step.root_index,
        env_row=action_step.env_row,
        player=action_step.player,
        selected_action=action_step.selected_action,
        batch_size=batch_size,
        player_count=player_count,
        inactive_action=inactive_action,
    )


def selected_joint_action_from_root_build_request_action_step(
    root_request: CompactRootBuildRequestV1,
    action_step: CompactSearchActionStepV1,
    *,
    batch_size: int,
    player_count: int,
    inactive_action: int = 0,
) -> np.ndarray:
    """Map action-critical output using only parent-side root request sidecars."""

    dense_joint_action = getattr(action_step, "dense_joint_action", None)
    if dense_joint_action is not None:
        return _validated_dense_joint_action_from_root_build_request_action_step(
            root_request,
            action_step,
            batch_size=batch_size,
            player_count=player_count,
        )
    return _selected_joint_action_from_compact_sidecars(
        _root_batch_sidecar_from_build_request(root_request),
        root_index=action_step.root_index,
        env_row=action_step.env_row,
        player=action_step.player,
        selected_action=action_step.selected_action,
        batch_size=batch_size,
        player_count=player_count,
        inactive_action=inactive_action,
    )


def selected_joint_action_from_root_action_context_action_step(
    root_action_context: CompactRootActionContextV1,
    action_step: CompactSearchActionStepV1,
    *,
    batch_size: int,
    player_count: int,
    inactive_action: int = 0,
) -> np.ndarray:
    """Map action-critical output using retained action-only root sidecars."""

    _validate_compact_search_action_step_identity_from_root_action_context(
        root_action_context,
        action_step,
    )
    dense_joint_action = getattr(action_step, "dense_joint_action", None)
    if dense_joint_action is not None:
        dense = np.asarray(dense_joint_action, dtype=np.int16)
        expected = (int(batch_size), int(player_count))
        if dense.shape != expected:
            raise ReplayCompatibilityError(
                "dense joint action shape does not match root action context"
            )
        if bool((dense < 0).any()) or bool((dense >= ACTION_COUNT).any()):
            raise ReplayCompatibilityError("dense joint action contains illegal action")
        env_row = np.asarray(action_step.env_row, dtype=np.int64).reshape(-1)
        player = np.asarray(action_step.player, dtype=np.int64).reshape(-1)
        selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
        if selected.size and not np.array_equal(dense[env_row, player], selected):
            raise ReplayCompatibilityError("dense joint action does not match selected actions")
        return dense.astype(np.int16, copy=True)
    joint_action = np.full(
        (int(batch_size), int(player_count)),
        int(inactive_action),
        dtype=np.int16,
    )
    env_row = np.asarray(action_step.env_row, dtype=np.int64).reshape(-1)
    player = np.asarray(action_step.player, dtype=np.int64).reshape(-1)
    selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
    if selected.size:
        joint_action[env_row, player] = selected
    return joint_action


def _validated_owner_verified_dense_joint_action_from_action_step(
    action_step: CompactSearchActionStepV1,
    *,
    batch_size: int,
    player_count: int,
) -> np.ndarray:
    dense_joint_action = getattr(action_step, "dense_joint_action", None)
    if dense_joint_action is None:
        raise ReplayCompatibilityError(
            "owner-verified handle action step requires dense joint action"
        )
    if action_step.metadata.get("compact_owner_search_dense_joint_action_owner_assembled") is not True:
        raise ReplayCompatibilityError("owner dense joint_action was not owner assembled")
    if int(action_step.metadata.get("compact_owner_search_dense_joint_action_fallback_count") or 0):
        raise ReplayCompatibilityError("owner dense joint_action used fallback")
    if int(action_step.metadata.get("compact_owner_search_dense_joint_action_mismatch_count") or 0):
        raise ReplayCompatibilityError("owner dense joint_action reported mismatch")
    dense = np.asarray(dense_joint_action, dtype=np.int16)
    expected = (int(batch_size), int(player_count))
    if dense.shape != expected:
        raise ReplayCompatibilityError("owner dense joint_action shape mismatch")
    if bool((dense < 0).any()) or bool((dense >= ACTION_COUNT).any()):
        raise ReplayCompatibilityError("owner dense joint_action out of range")
    env_row = np.asarray(action_step.env_row, dtype=np.int64).reshape(-1)
    player = np.asarray(action_step.player, dtype=np.int64).reshape(-1)
    selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
    if not (env_row.shape == player.shape == selected.shape):
        raise ReplayCompatibilityError("owner dense joint_action action sidecar shape mismatch")
    if selected.size and not np.array_equal(dense[env_row, player], selected):
        action_step.metadata["compact_owner_search_dense_joint_action_mismatch_count"] = 1
        raise ReplayCompatibilityError("owner dense joint_action selected-action mismatch")
    expected_checksum = int(
        action_step.metadata.get("compact_owner_search_dense_joint_action_checksum") or 0
    )
    actual_checksum = _owner_action_checksum_v1(dense)
    if expected_checksum and expected_checksum != actual_checksum:
        raise ReplayCompatibilityError("owner dense joint_action checksum mismatch")
    expected_digest = str(
        action_step.metadata.get("compact_owner_search_dense_joint_action_digest") or ""
    )
    actual_digest = str(compact_search_array_digest_v1(dense.reshape(-1)))
    if expected_digest and expected_digest != actual_digest:
        raise ReplayCompatibilityError("owner dense joint_action digest mismatch")
    action_step.metadata.update(
        {
            "compact_owner_search_dense_joint_action_owner_assembled": True,
            "compact_owner_search_dense_joint_action_parent_assembly_avoided": True,
            "compact_owner_search_dense_joint_action_present": True,
            "compact_owner_search_dense_joint_action_fallback_count": 0,
            "compact_owner_search_dense_joint_action_fallback_reason": "none",
            "compact_owner_search_dense_joint_action_checksum": int(actual_checksum),
            "compact_owner_search_dense_joint_action_digest": str(actual_digest),
            "compact_owner_search_dense_joint_action_bytes": int(dense.nbytes),
            "compact_owner_search_dense_joint_action_mismatch_count": 0,
            "compact_rollout_slab_parent_dense_action_reconstruction_count": 0,
        }
    )
    return dense.astype(np.int16, copy=True)


def _validated_dense_joint_action_from_root_build_request_action_step(
    root_request: CompactRootBuildRequestV1,
    action_step: CompactSearchActionStepV1,
    *,
    batch_size: int,
    player_count: int,
) -> np.ndarray:
    dense = np.asarray(action_step.dense_joint_action, dtype=np.int16)
    expected = (int(batch_size), int(player_count))
    if dense.shape != expected:
        raise ReplayCompatibilityError("owner dense joint_action shape mismatch")
    if bool((dense < 0).any()) or bool((dense >= ACTION_COUNT).any()):
        raise ReplayCompatibilityError("owner dense joint_action out of range")
    root_index = np.asarray(action_step.root_index, dtype=np.int64).reshape(-1)
    selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
    env_row = np.asarray(root_request.policy_env_row, dtype=np.int64).reshape(-1)
    player = np.asarray(root_request.policy_player, dtype=np.int64).reshape(-1)
    if root_index.shape != selected.shape:
        raise ReplayCompatibilityError("owner dense action selected-action shape mismatch")
    if selected.size and not np.array_equal(
        dense[env_row[root_index], player[root_index]],
        selected,
    ):
        action_step.metadata["compact_owner_search_dense_joint_action_mismatch_count"] = 1
        raise ReplayCompatibilityError("owner dense joint_action selected-action mismatch")
    legal_mask = np.asarray(root_request.action_mask, dtype=np.bool_).reshape(
        int(root_request.root_count),
        ACTION_COUNT,
    )
    if selected.size and not bool(legal_mask[root_index, selected.astype(np.int64)].all()):
        raise ReplayCompatibilityError("owner dense joint_action selected action is illegal")
    expected_checksum = int(
        action_step.metadata.get("compact_owner_search_dense_joint_action_checksum") or 0
    )
    actual_checksum = _owner_action_checksum_v1(dense)
    if expected_checksum and expected_checksum != actual_checksum:
        raise ReplayCompatibilityError("owner dense joint_action checksum mismatch")
    expected_digest = str(
        action_step.metadata.get("compact_owner_search_dense_joint_action_digest") or ""
    )
    actual_digest = str(compact_search_array_digest_v1(dense.reshape(-1)))
    if expected_digest and expected_digest != actual_digest:
        raise ReplayCompatibilityError("owner dense joint_action digest mismatch")
    action_step.metadata.update(
        {
            "compact_owner_search_dense_joint_action_owner_assembled": True,
            "compact_owner_search_dense_joint_action_parent_assembly_avoided": True,
            "compact_owner_search_dense_joint_action_present": True,
            "compact_owner_search_dense_joint_action_fallback_count": 0,
            "compact_owner_search_dense_joint_action_fallback_reason": "none",
            "compact_owner_search_dense_joint_action_checksum": int(actual_checksum),
            "compact_owner_search_dense_joint_action_digest": str(actual_digest),
            "compact_owner_search_dense_joint_action_bytes": int(dense.nbytes),
            "compact_owner_search_dense_joint_action_mismatch_count": 0,
            "compact_rollout_slab_parent_dense_action_reconstruction_count": 0,
        }
    )
    return dense.astype(np.int16, copy=True)


def _selected_joint_action_from_compact_sidecars(
    root_batch: CompactRootBatchV1,
    *,
    root_index: np.ndarray,
    env_row: np.ndarray,
    player: np.ndarray,
    selected_action: np.ndarray,
    batch_size: int,
    player_count: int,
    inactive_action: int,
) -> np.ndarray:
    """Map compact active-root sidecars back to dense joint actions."""

    batch_size = _positive_int(batch_size, "batch_size")
    player_count = _positive_int(player_count, "player_count")
    inactive = int(inactive_action)
    if inactive < 0 or inactive >= ACTION_COUNT:
        raise ReplayCompatibilityError("inactive_action is out of range")
    root_index = np.asarray(root_index, dtype=np.int64).reshape(-1)
    env_row = np.asarray(env_row, dtype=np.int64).reshape(-1)
    player = np.asarray(player, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected_action, dtype=np.int16).reshape(-1)
    if not (root_index.shape == env_row.shape == player.shape == selected.shape):
        raise ReplayCompatibilityError("compact slab search sidecar shape mismatch")
    legal_mask = np.asarray(root_batch.legal_mask, dtype=np.bool_)
    total_roots = int(batch_size) * int(player_count)
    if int(selected.size) == total_roots:
        expected_root = np.arange(total_roots, dtype=np.int64)
        active_root_mask = np.asarray(root_batch.active_root_mask, dtype=np.bool_).reshape(-1)
        if (
            active_root_mask.shape == (total_roots,)
            and bool(active_root_mask.all())
            and legal_mask.shape[0] == total_roots
            and np.array_equal(root_index, expected_root)
            and np.array_equal(env_row, expected_root // int(player_count))
            and np.array_equal(player, expected_root % int(player_count))
        ):
            if bool(((selected < 0) | (selected >= ACTION_COUNT)).any()):
                raise ReplayCompatibilityError("compact slab selected action out of range")
            if not bool(legal_mask[expected_root, selected.astype(np.int64)].all()):
                raise ReplayCompatibilityError("compact slab selected action is illegal")
            return selected.reshape(batch_size, player_count).copy()
    joint_action = np.full((batch_size, player_count), inactive, dtype=np.int16)
    for output_row, compact_root in enumerate(root_index):
        action = int(selected[output_row])
        if action < 0 or action >= ACTION_COUNT:
            raise ReplayCompatibilityError("compact slab selected action out of range")
        if not bool(legal_mask[int(compact_root), action]):
            raise ReplayCompatibilityError("compact slab selected action is illegal")
        row = int(env_row[output_row])
        seat = int(player[output_row])
        if row < 0 or row >= batch_size or seat < 0 or seat >= player_count:
            raise ReplayCompatibilityError("compact slab env/player sidecar out of range")
        joint_action[row, seat] = np.int16(action)
    return joint_action


def _validate_compact_search_action_step_identity(
    root_batch: CompactRootBatchV1,
    action_step: CompactSearchActionStepV1,
) -> None:
    metadata = action_step.metadata
    if metadata.get("schema_id") != COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID:
        raise ReplayCompatibilityError("action step schema id is invalid")
    if metadata.get("phase") != "action_critical":
        raise ReplayCompatibilityError("action step phase is invalid")
    if not str(action_step.replay_payload_handle):
        raise ReplayCompatibilityError("action step replay payload handle is empty")
    root_index = np.asarray(action_step.root_index, dtype=np.int64).reshape(-1)
    active_root_index = np.flatnonzero(root_batch.active_root_mask).astype(
        np.int64,
        copy=False,
    )
    if not np.array_equal(root_index, active_root_index):
        raise ReplayCompatibilityError("action step roots must match root batch active roots")
    if not np.array_equal(action_step.env_row, root_batch.env_row[root_index]):
        raise ReplayCompatibilityError("action step env_row does not match root batch")
    if not np.array_equal(action_step.player, root_batch.player[root_index]):
        raise ReplayCompatibilityError("action step player does not match root batch")
    if not np.array_equal(
        action_step.policy_env_id,
        root_batch.policy_env_id[root_index],
    ):
        raise ReplayCompatibilityError("action step policy_env_id does not match root batch")
    selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
    if selected.shape != root_index.shape:
        raise ReplayCompatibilityError("action step selected_action shape mismatch")
    if metadata.get("selected_action_digest") != compact_search_array_digest_v1(
        selected.astype(np.int16, copy=False)
    ):
        raise ReplayCompatibilityError("action step selected-action digest is stale")


def _validate_compact_search_action_step_identity_from_root_build_request(
    root_request: CompactRootBuildRequestV1,
    action_step: CompactSearchActionStepV1,
) -> None:
    metadata = action_step.metadata
    if metadata.get("schema_id") != COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID:
        raise ReplayCompatibilityError("action step schema id is invalid")
    if metadata.get("phase") != "action_critical":
        raise ReplayCompatibilityError("action step phase is invalid")
    if not str(action_step.replay_payload_handle):
        raise ReplayCompatibilityError("action step replay payload handle is empty")
    root_index = np.asarray(action_step.root_index, dtype=np.int64).reshape(-1)
    active_root_index = np.flatnonzero(
        np.asarray(root_request.active_root_mask, dtype=np.bool_).reshape(-1)
    ).astype(np.int64, copy=False)
    if not np.array_equal(root_index, active_root_index):
        raise ReplayCompatibilityError(
            "action step roots must match root build request active roots"
        )
    env_row = np.asarray(root_request.policy_env_row, dtype=np.int32).reshape(-1)
    player = np.asarray(root_request.policy_player, dtype=np.int16).reshape(-1)
    policy_env_id = np.asarray(root_request.policy_env_id, dtype=np.int64).reshape(-1)
    if not np.array_equal(action_step.env_row, env_row[root_index]):
        raise ReplayCompatibilityError("action step env_row does not match root build request")
    if not np.array_equal(action_step.player, player[root_index]):
        raise ReplayCompatibilityError("action step player does not match root build request")
    if not np.array_equal(action_step.policy_env_id, policy_env_id[root_index]):
        raise ReplayCompatibilityError(
            "action step policy_env_id does not match root build request"
        )
    selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
    if selected.shape != root_index.shape:
        raise ReplayCompatibilityError("action step selected_action shape mismatch")
    legal_mask = np.asarray(root_request.action_mask, dtype=np.bool_).reshape(
        int(root_request.root_count),
        ACTION_COUNT,
    )
    if selected.size and not bool(legal_mask[root_index, selected.astype(np.int64)].all()):
        raise ReplayCompatibilityError("action step selected_action is illegal")


def _validate_compact_search_action_step_identity_from_root_action_context(
    root_action_context: CompactRootActionContextV1,
    action_step: CompactSearchActionStepV1,
) -> None:
    metadata = action_step.metadata
    if metadata.get("schema_id") != COMPACT_SEARCH_ACTION_STEP_SCHEMA_ID:
        raise ReplayCompatibilityError("action step schema id is invalid")
    if metadata.get("phase") != "action_critical":
        raise ReplayCompatibilityError("action step phase is invalid")
    if not str(action_step.replay_payload_handle):
        raise ReplayCompatibilityError("action step replay payload handle is empty")
    root_index = np.asarray(action_step.root_index, dtype=np.int32).reshape(-1)
    expected_root_index = np.asarray(
        root_action_context.active_root_index,
        dtype=np.int32,
    ).reshape(-1)
    if not np.array_equal(root_index, expected_root_index):
        raise ReplayCompatibilityError(
            "action step roots do not match root action context"
        )
    env_row = np.asarray(root_action_context.env_row, dtype=np.int32).reshape(-1)
    player = np.asarray(root_action_context.player, dtype=np.int16).reshape(-1)
    policy_env_id = np.asarray(root_action_context.policy_env_id, dtype=np.int64).reshape(-1)
    if not np.array_equal(action_step.env_row, env_row):
        raise ReplayCompatibilityError(
            "action step env_row does not match root action context"
        )
    if not np.array_equal(action_step.player, player):
        raise ReplayCompatibilityError(
            "action step player does not match root action context"
        )
    if not np.array_equal(action_step.policy_env_id, policy_env_id):
        raise ReplayCompatibilityError(
            "action step policy_env_id does not match root action context"
        )
    selected = np.asarray(action_step.selected_action, dtype=np.int16).reshape(-1)
    if selected.shape != root_index.shape:
        raise ReplayCompatibilityError("action step selected_action shape mismatch")
    if bool((selected < 0).any()) or bool((selected >= ACTION_COUNT).any()):
        raise ReplayCompatibilityError("action step selected_action is illegal")
    legal_mask = np.asarray(root_action_context.active_legal_mask, dtype=np.bool_)
    if legal_mask.shape != (int(selected.size), ACTION_COUNT):
        raise ReplayCompatibilityError("root action context legal mask shape mismatch")
    if selected.size and not bool(legal_mask[np.arange(selected.size), selected].all()):
        raise ReplayCompatibilityError("action step selected_action is illegal")
    if metadata.get("selected_action_digest") != compact_search_array_digest_v1(
        selected.astype(np.int16, copy=False)
    ):
        raise ReplayCompatibilityError("action step selected-action digest is stale")
    if metadata.get("selected_action_digest") != compact_search_array_digest_v1(
        selected.astype(np.int16, copy=False)
    ):
        raise ReplayCompatibilityError("action step selected-action digest is stale")


def _root_batch_sidecar_from_build_request(
    root_request: CompactRootBuildRequestV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> CompactRootBatchV1:
    """Build a parent-side sidecar batch without using the root-batch builder."""

    batch_size = int(root_request.batch_size)
    root_count = int(root_request.root_count)
    stack_shape = tuple(int(dim) for dim in root_request.stack_shape)
    observation_dtype = np.uint8
    resident = root_request.resident_observation
    if resident is not None:
        dtype_text = str(getattr(resident, "dtype", "uint8"))
        if dtype_text.startswith("torch."):
            dtype_text = dtype_text.split(".", 1)[1]
        try:
            observation_dtype = np.dtype(dtype_text)
        except TypeError:
            observation_dtype = np.uint8
    observation = np.broadcast_to(
        np.zeros((), dtype=observation_dtype),
        (root_count, *stack_shape),
    )
    request_metadata = dict(root_request.metadata)
    request_metadata.update(
        {
            "compact_owner_search_direct_root_build_request_handoff": True,
            "compact_owner_search_direct_root_parent_build_avoided": True,
            "compact_owner_search_direct_root_parent_build_call_count": 0,
            "compact_owner_search_direct_root_parent_build_sec": 0.0,
            "compact_rollout_slab_parent_root_batch_build_avoided": True,
            "compact_rollout_slab_parent_root_batch_builder_used": False,
            "compact_rollout_slab_parent_root_batch_builder_call_count": 0,
            "observation_copied": False,
            "resident_host_observation_stub_requested": bool(
                root_request.resident_host_observation_stub
            ),
            "resident_host_observation_stubbed": bool(root_request.resident_host_observation_stub),
            "resident_host_observation_stub_kind": (
                "zero_stride_shape_only_v1" if root_request.resident_host_observation_stub else ""
            ),
            "resident_host_observation_stub_materialized_bytes": 0,
            "resident_host_observation_stub_logical_bytes": int(
                root_count * int(np.prod(stack_shape)) * np.dtype(observation_dtype).itemsize
            ),
        }
    )
    if metadata:
        request_metadata.update({str(key): value for key, value in metadata.items()})
    return CompactRootBatchV1(
        observation=observation,
        legal_mask=np.asarray(root_request.action_mask, dtype=np.bool_).reshape(
            root_count,
            ACTION_COUNT,
        ),
        active_root_mask=np.asarray(
            root_request.active_root_mask,
            dtype=np.bool_,
        ).reshape(root_count),
        to_play=np.asarray(root_request.to_play, dtype=np.int64).reshape(root_count),
        env_row=np.asarray(root_request.policy_env_row, dtype=np.int32).reshape(root_count),
        player=np.asarray(root_request.policy_player, dtype=np.int16).reshape(root_count),
        policy_env_id=np.asarray(root_request.policy_env_id, dtype=np.int64).reshape(root_count),
        target_reward=np.asarray(root_request.target_reward, dtype=np.float32).reshape(
            root_count,
            1,
        ),
        done_root=np.asarray(root_request.done_root, dtype=np.bool_).reshape(root_count),
        final_observation=None,
        final_observation_row_mask=np.asarray(
            root_request.final_observation_row_mask,
            dtype=np.bool_,
        ).reshape(batch_size),
        terminal_row_mask=np.asarray(
            root_request.terminal_row_mask,
            dtype=np.bool_,
        ).reshape(batch_size),
        autoreset_row_mask=np.asarray(
            root_request.autoreset_row_mask,
            dtype=np.bool_,
        ).reshape(batch_size),
        metadata=request_metadata,
        resident_observation=resident,
        observation_source=str(root_request.observation_source),
        terminated=None
        if root_request.terminated is None
        else np.asarray(root_request.terminated, dtype=np.bool_).reshape(batch_size),
        truncated=None
        if root_request.truncated is None
        else np.asarray(root_request.truncated, dtype=np.bool_).reshape(batch_size),
        final_reward_map=None
        if root_request.final_reward_map is None
        else np.asarray(root_request.final_reward_map, dtype=np.float32).reshape(
            batch_size,
            int(root_request.player_count),
        ),
    )


def _pending_action_arrays(
    pending: _PendingCompactSearchV1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pending.action_step is not None:
        return (
            np.asarray(pending.action_step.env_row, dtype=np.int64),
            np.asarray(pending.action_step.player, dtype=np.int64),
            np.asarray(pending.action_step.selected_action, dtype=np.int16),
        )
    if pending.search_result is None:
        raise ReplayCompatibilityError("pending compact search has no action payload")
    return (
        np.asarray(pending.search_result.env_row, dtype=np.int64),
        np.asarray(pending.search_result.player, dtype=np.int64),
        np.asarray(pending.search_result.selected_action, dtype=np.int16),
    )


def _owner_action_checksum_v1(action: Any) -> int:
    flat = np.asarray(action, dtype=np.int64).reshape(-1)
    if flat.size <= 0:
        return 0
    weights = np.arange(1, flat.size + 1, dtype=np.int64)
    return int(np.dot(flat + 1, weights))


def _root_build_request_sidecar_metadata_for_telemetry(
    root_request: CompactRootBuildRequestV1,
) -> dict[str, Any]:
    metadata = dict(root_request.metadata or {})
    observation = root_request.observation
    resident = root_request.resident_observation
    observation_dtype = np.uint8
    if resident is not None:
        dtype_text = str(getattr(resident, "dtype", "uint8"))
        if dtype_text.startswith("torch."):
            dtype_text = dtype_text.split(".", 1)[1]
        try:
            observation_dtype = np.dtype(dtype_text)
        except TypeError:
            observation_dtype = np.uint8
    elif observation is not None:
        observation_dtype = np.asarray(observation).dtype
    stack_shape = tuple(int(dim) for dim in root_request.stack_shape)
    root_count = int(root_request.root_count)
    resident_stubbed = bool(root_request.resident_host_observation_stub)
    metadata.setdefault(
        "observation_copied",
        bool(root_request.copy_observation and observation is not None),
    )
    metadata.setdefault(
        "resident_host_observation_stub_requested",
        bool(root_request.resident_host_observation_stub),
    )
    metadata.setdefault("resident_host_observation_stubbed", resident_stubbed)
    metadata.setdefault(
        "resident_host_observation_stub_kind",
        "zero_stride_shape_only_v1" if resident_stubbed else "",
    )
    metadata.setdefault("resident_host_observation_stub_materialized_bytes", 0)
    metadata.setdefault(
        "resident_host_observation_stub_logical_bytes",
        int(root_count * int(np.prod(stack_shape)) * np.dtype(observation_dtype).itemsize),
    )
    return metadata


def _root_action_context_sidecar_metadata_for_telemetry(
    root_action_context: CompactRootActionContextV1,
) -> dict[str, Any]:
    metadata = dict(root_action_context.metadata or {})
    observation_dtype = np.uint8
    try:
        observation_dtype = np.dtype(str(root_action_context.observation_dtype))
    except TypeError:
        observation_dtype = np.uint8
    stack_shape = tuple(int(dim) for dim in root_action_context.stack_shape)
    root_count = int(root_action_context.root_count)
    resident_stubbed = bool(root_action_context.resident_host_observation_stub)
    metadata.setdefault(
        "observation_copied",
        False,
    )
    metadata.setdefault(
        "resident_host_observation_stub_requested",
        resident_stubbed,
    )
    metadata.setdefault("resident_host_observation_stubbed", resident_stubbed)
    metadata.setdefault(
        "resident_host_observation_stub_kind",
        "zero_stride_shape_only_v1" if resident_stubbed else "",
    )
    metadata.setdefault("resident_host_observation_stub_materialized_bytes", 0)
    metadata.setdefault(
        "resident_host_observation_stub_logical_bytes",
        int(root_count * int(np.prod(stack_shape)) * np.dtype(observation_dtype).itemsize),
    )
    metadata.setdefault("compact_owner_search_pending_root_build_request_stored", False)
    metadata.setdefault(
        "compact_owner_search_pending_root_action_context_stored",
        True,
    )
    return metadata


def _slab_telemetry(
    root_batch: CompactRootBatchV1 | None,
    search_result: CompactSearchResultV1 | None,
    action_step: CompactSearchActionStepV1 | None,
    committed: CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1 | None,
    *,
    root_build_request: CompactRootBuildRequestV1 | None = None,
    root_action_context: CompactRootActionContextV1 | None = None,
    action_feedback_mode: str = COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK,
    action_override_drop_count: int = 0,
    two_phase_search: bool = False,
    replay_payload_flush_count: int = 0,
    committed_replay_payload_flushed: bool = False,
    committed_replay_payload_d2h_bytes: int = 0,
    internal_timing: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if search_result is not None:
        selected_source = search_result.selected_action
        search_metadata = dict(search_result.metadata)
    elif action_step is not None:
        selected_source = action_step.selected_action
        search_metadata = dict(action_step.metadata)
    else:
        raise ReplayCompatibilityError("compact slab telemetry requires action payload")
    selected = np.asarray(selected_source, dtype=np.int64).reshape(-1)
    weights = np.arange(1, selected.size + 1, dtype=np.int64)
    profile_telemetry = search_metadata.get("profile_telemetry", {})
    if not isinstance(profile_telemetry, Mapping):
        profile_telemetry = {}
    if root_batch is not None:
        root_count = int(root_batch.legal_mask.shape[0])
        root_metadata = dict(root_batch.metadata or {})
    elif root_build_request is not None:
        root_count = int(root_build_request.root_count)
        root_metadata = _root_build_request_sidecar_metadata_for_telemetry(
            root_build_request
        )
    elif root_action_context is not None:
        root_count = int(root_action_context.root_count)
        root_metadata = _root_action_context_sidecar_metadata_for_telemetry(
            root_action_context
        )
    else:
        root_count = int(
            search_metadata.get("compact_owner_root_action_context_root_count")
            or search_metadata.get("compact_rollout_slab_root_count")
            or search_metadata.get("active_root_count")
            or selected.size
        )
        if root_count <= 0:
            raise ReplayCompatibilityError("compact slab telemetry requires root provenance")
        root_metadata = {}
    telemetry: dict[str, Any] = {
        "compact_rollout_slab_schema_id": COMPACT_ROLLOUT_SLAB_STEP_SCHEMA_ID,
        "compact_rollout_slab_profile_only": True,
        "compact_rollout_slab_calls_train_muzero": False,
        "compact_rollout_slab_root_count": int(root_count),
        "compact_rollout_slab_active_root_count": int(selected.size),
        "compact_rollout_slab_search_impl": str(search_metadata.get("search_impl", "")),
        "compact_rollout_slab_num_simulations": int(search_metadata.get("num_simulations", 0)),
        "compact_rollout_slab_profile_telemetry": dict(profile_telemetry),
        "compact_rollout_slab_search_metadata": search_metadata,
        "compact_rollout_slab_two_phase_search": bool(two_phase_search),
        "compact_rollout_slab_action_step_only": bool(two_phase_search and search_result is None),
        "compact_rollout_slab_replay_payload_flush_count": int(replay_payload_flush_count),
        "compact_rollout_slab_committed_replay_payload_flushed": bool(
            committed_replay_payload_flushed
        ),
        "compact_rollout_slab_committed_replay_payload_d2h_bytes": int(
            committed_replay_payload_d2h_bytes
        ),
        "compact_rollout_slab_observation_copied": bool(
            root_metadata.get("observation_copied", False)
        ),
        "compact_rollout_slab_resident_host_observation_stub_requested": bool(
            root_metadata.get("resident_host_observation_stub_requested", False)
        ),
        "compact_rollout_slab_resident_host_observation_stubbed": bool(
            root_metadata.get("resident_host_observation_stubbed", False)
        ),
        "compact_rollout_slab_resident_host_observation_stub_kind": str(
            root_metadata.get("resident_host_observation_stub_kind") or ""
        ),
        "compact_rollout_slab_resident_host_observation_stub_materialized_bytes": int(
            root_metadata.get(
                "resident_host_observation_stub_materialized_bytes",
                0,
            )
            or 0
        ),
        "compact_rollout_slab_resident_host_observation_stub_logical_bytes": int(
            root_metadata.get(
                "resident_host_observation_stub_logical_bytes",
                0,
            )
            or 0
        ),
        "compact_rollout_slab_action_feedback_mode": str(action_feedback_mode),
        "compact_rollout_slab_replay_commit_requires_search_action": (
            str(action_feedback_mode) == COMPACT_ROLLOUT_SLAB_ACTION_MODE_SEARCH_FEEDBACK
        ),
        "compact_rollout_slab_action_override_drop_count": int(action_override_drop_count),
        "compact_rollout_slab_selected_action_checksum": int(selected.dot(weights))
        if selected.size
        else 0,
        "compact_rollout_slab_committed_index_row_count": 0
        if committed is None
        else _compact_index_row_count(committed),
    }
    if committed is not None:
        committed_metadata = dict(getattr(committed, "metadata", {}) or {})
        telemetry["compact_rollout_slab_device_replay_index_rows"] = bool(
            committed_metadata.get("device_replay_index_rows", False)
        )
        builder_variant = committed_metadata.get("replay_index_rows_builder_variant")
        if builder_variant is not None:
            telemetry["compact_rollout_slab_replay_index_rows_builder_variant"] = str(
                builder_variant
            )
        for metadata_key, telemetry_key in _REPLAY_INDEX_METADATA_NUMERIC_FIELDS:
            if metadata_key in committed_metadata:
                telemetry[telemetry_key] = float(committed_metadata[metadata_key])
    for key, value in (internal_timing or {}).items():
        telemetry[str(key)] = float(value)
    for metadata_key, telemetry_key in _COMPACT_ROLLOUT_SLAB_OWNER_SEARCH_TIMING_FIELDS:
        if metadata_key in search_metadata:
            telemetry[telemetry_key] = _metadata_float(search_metadata, metadata_key)
    semantics = (
        search_metadata.get("profile_semantics")
        or search_metadata.get("compact_torch_search_semantics")
        or profile_telemetry.get("compact_torch_search_semantics")
    )
    if semantics is not None:
        telemetry["compact_rollout_slab_semantics"] = str(semantics)
    _add_profile_timing_summary(telemetry, profile_telemetry)
    _add_search_dispatch_accounting(telemetry)
    return telemetry


def _empty_commit_timing() -> dict[str, float]:
    return {key: 0.0 for key in _COMPACT_ROLLOUT_SLAB_ALL_COMMIT_FIELDS}


_REPLAY_INDEX_METADATA_TIMING_FIELDS = (
    (
        "replay_index_rows_identity_validate_sec",
        "compact_rollout_slab_replay_index_rows_identity_validate_sec",
    ),
    (
        "replay_index_rows_terminal_prepare_sec",
        "compact_rollout_slab_replay_index_rows_terminal_prepare_sec",
    ),
    (
        "replay_index_rows_target_tensor_sec",
        "compact_rollout_slab_replay_index_rows_target_tensor_sec",
    ),
    (
        "replay_index_rows_scalar_host_pack_sec",
        "compact_rollout_slab_replay_index_rows_scalar_host_pack_sec",
    ),
    (
        "replay_index_rows_scalar_device_transfer_sec",
        "compact_rollout_slab_replay_index_rows_scalar_device_transfer_sec",
    ),
    (
        "replay_index_rows_metadata_sec",
        "compact_rollout_slab_replay_index_rows_metadata_sec",
    ),
)
_REPLAY_INDEX_METADATA_NUMERIC_FIELDS = (
    (
        "replay_index_rows_scalar_packed_h2d_bytes",
        "compact_rollout_slab_replay_index_rows_scalar_packed_h2d_bytes",
    ),
    (
        "replay_index_rows_scalar_tensor_count",
        "compact_rollout_slab_replay_index_rows_scalar_tensor_count",
    ),
)


def _add_replay_index_detail_timing(
    out: dict[str, float],
    committed: CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1,
) -> None:
    metadata = dict(getattr(committed, "metadata", {}) or {})
    for metadata_key, timing_key in _REPLAY_INDEX_METADATA_TIMING_FIELDS:
        out[timing_key] = _metadata_float(metadata, metadata_key)
    for metadata_key, timing_key in _REPLAY_INDEX_METADATA_NUMERIC_FIELDS:
        out[timing_key] = _metadata_float(metadata, metadata_key)


def _elapsed(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def _metadata_float(metadata: Mapping[str, Any], key: str) -> float:
    try:
        return float(metadata.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _profile_float(telemetry: Mapping[str, Any], key: str) -> float:
    try:
        return float(telemetry.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _add_search_dispatch_accounting(out: dict[str, Any]) -> None:
    dispatch_sec = _profile_float(out, "compact_rollout_slab_search_dispatch_wall_sec")
    action_wall_sec = _profile_float(
        out,
        "compact_rollout_slab_search_service_action_wall_sec",
    )
    service_total_sec = _profile_float(
        out,
        "compact_rollout_slab_search_service_total_sec",
    )
    service_envelope_sec = action_wall_sec if action_wall_sec > 0.0 else service_total_sec
    residual_sec = dispatch_sec - service_envelope_sec
    out["compact_rollout_slab_search_dispatch_service_envelope_sec"] = service_envelope_sec
    out["compact_rollout_slab_search_dispatch_residual_sec"] = residual_sec
    out["compact_rollout_slab_search_dispatch_positive_residual_sec"] = max(
        0.0,
        residual_sec,
    )
    out["compact_rollout_slab_search_dispatch_overaccounted_sec"] = max(
        0.0,
        -residual_sec,
    )


def _copy_first_profile_field(
    out: dict[str, Any],
    profile_telemetry: Mapping[str, Any],
    target: str,
    *sources: str,
) -> None:
    for source in sources:
        if source in profile_telemetry:
            value = profile_telemetry[source]
            if isinstance(value, (int, float, bool)) or value is None:
                out[target] = value
            else:
                out[target] = value
            return


def _add_profile_timing_summary(
    out: dict[str, Any],
    profile_telemetry: Mapping[str, Any],
) -> None:
    _copy_first_profile_field(
        out,
        profile_telemetry,
        "compact_rollout_slab_search_service_total_sec",
        "lightzero_mcts_arrays_boundary_total_sec",
        "lightzero_array_ceiling_total_sec",
        "mctx_compact_search_service_total_sec",
        "compact_torch_search_service_total_sec",
        "fixed_shape_batched_search_owner_total_sec",
    )
    consumer_model_sec = _profile_float(profile_telemetry, "lightzero_consumer_model_total_sec")
    initial_sec = _profile_float(
        profile_telemetry,
        "lightzero_mcts_arrays_boundary_initial_inference_sec",
    ) + _profile_float(profile_telemetry, "lightzero_array_ceiling_initial_inference_sec")
    recurrent_sec = _profile_float(
        profile_telemetry,
        "lightzero_mcts_arrays_boundary_recurrent_inference_sec",
    ) + _profile_float(profile_telemetry, "lightzero_array_ceiling_recurrent_inference_sec")
    compact_torch_model_sec = _profile_float(
        profile_telemetry,
        "compact_torch_search_service_initial_inference_sec",
    )
    if consumer_model_sec:
        out["compact_rollout_slab_model_sec"] = consumer_model_sec
    elif compact_torch_model_sec:
        out["compact_rollout_slab_model_sec"] = compact_torch_model_sec
    elif initial_sec or recurrent_sec:
        out["compact_rollout_slab_model_sec"] = initial_sec + recurrent_sec
    _copy_first_profile_field(
        out,
        profile_telemetry,
        "compact_rollout_slab_search_sec",
        "lightzero_mcts_arrays_boundary_search_sec",
        "lightzero_array_ceiling_search_update_sec",
        "mctx_compact_search_service_search_sec",
        "compact_torch_search_service_tree_search_sec",
        "fixed_shape_batched_search_owner_total_sec",
    )
    _copy_first_profile_field(
        out,
        profile_telemetry,
        "compact_rollout_slab_h2d_sec",
        "lightzero_consumer_h2d_sec",
        "host_to_device_sec",
        "lightzero_mcts_arrays_boundary_h2d_sec",
        "lightzero_array_ceiling_h2d_sec",
        "mctx_compact_search_service_h2d_sec",
        "lightzero_mcts_arrays_boundary_input_prepare_sec",
        "compact_torch_search_service_tensor_prepare_sec",
        "fixed_shape_batched_search_owner_h2d_sec",
    )
    for suffix in (
        "action_preamble_sec",
        "fixed_shape_masks_sec",
        "compile_eligibility_sec",
        "helper_cache_sec",
        "model_cache_sec",
        "inference_guard_enter_sec",
        "inference_guard_exit_sec",
        "inference_guard_total_sec",
        "metadata_build_sec",
        "pending_replay_store_sec",
        "action_step_build_sec",
        "action_postprocess_sec",
        "action_wall_sec",
        "action_accounted_sec",
        "action_residual_sec",
        "action_unaccounted_sec",
        "action_overaccounted_sec",
        "tensor_prepare_sync_sec",
        "initial_output_decode_sec",
        "root_output_decode_sec",
        "root_latent_prepare_sec",
        "initial_inference_enqueue_sec",
        "initial_inference_sync_sec",
        "initial_inference_cuda_event_sec",
        "initial_inference_representation_sec",
        "initial_inference_prediction_sec",
        "initial_inference_pack_sec",
        "initial_inference_representation_cuda_event_sec",
        "initial_inference_prediction_cuda_event_sec",
        "initial_inference_direct_core_cuda_event_sec",
        "initial_inference_direct_core_cuda_event_residual_sec",
        "tree_setup_sec",
        "tree_root_prior_build_sec",
        "tree_root_prior_select_sec",
        "tree_select_enqueue_sec",
        "tree_recurrent_action_build_sec",
        "tree_recurrent_inference_enqueue_sec",
        "tree_recurrent_inference_cuda_event_sec",
        "tree_recurrent_output_decode_sec",
        "tree_backup_enqueue_sec",
        "tree_policy_build_sec",
        "tree_sync_sec",
        "tree_cuda_event_sec",
        "tree_total_sec",
        "tree_accounted_sec",
        "tree_residual_sec",
        "tree_unaccounted_sec",
        "tree_overaccounted_sec",
        "action_readback_sec",
        "core_accounted_sec",
        "core_residual_sec",
        "core_unaccounted_sec",
        "core_overaccounted_sec",
        "cuda_event_timing_enabled",
        "initial_sync_enabled",
    ):
        _copy_first_profile_field(
            out,
            profile_telemetry,
            f"compact_rollout_slab_search_service_{suffix}",
            f"compact_torch_search_service_{suffix}",
        )
    _copy_first_profile_field(
        out,
        profile_telemetry,
        "compact_rollout_slab_search_service_timing_mode",
        "compact_torch_search_service_timing_mode",
    )
    for suffix in (
        "initial_inference_direct_requested",
        "initial_inference_direct_used",
        "initial_inference_fallback_count",
        "initial_inference_mode_requested",
        "initial_inference_mode_effective",
        "initial_inference_runtime_status",
    ):
        _copy_first_profile_field(
            out,
            profile_telemetry,
            f"compact_rollout_slab_search_service_{suffix}",
            f"compact_torch_search_{suffix}",
        )
    for suffix in (
        "observation_memory_format_requested",
        "observation_memory_format_effective",
        "observation_normalized_uint8",
        "observation_dtype_before_model",
        "observation_dtype_model_input",
        "observation_layout_copy_bytes",
        "observation_is_contiguous",
        "observation_is_channels_last",
        "model_memory_format_requested",
        "model_memory_format_active",
        "model_memory_format_applied",
        "root_latent_dtype",
        "root_latent_ndim",
        "root_latent_is_contiguous_before_recurrent",
        "root_latent_is_channels_last_before_recurrent",
        "root_latent_contiguous_for_recurrent",
        "root_latent_is_channels_last_for_recurrent",
        "root_latent_contiguous_copy_bytes",
    ):
        _copy_first_profile_field(
            out,
            profile_telemetry,
            f"compact_rollout_slab_search_service_{suffix}",
            f"compact_torch_search_{suffix}",
        )
    _copy_first_profile_field(
        out,
        profile_telemetry,
        "compact_rollout_slab_search_service_one_simulation_fast_path",
        "compact_torch_search_one_simulation_fast_path",
    )
    _copy_first_profile_field(
        out,
        profile_telemetry,
        "compact_rollout_slab_search_service_one_simulation_root_prior_softmax_skipped",
        "compact_torch_search_one_simulation_root_prior_softmax_skipped",
    )
    _copy_first_profile_field(
        out,
        profile_telemetry,
        "compact_rollout_slab_search_service_one_simulation_selection_mode",
        "compact_torch_search_one_simulation_selection_mode",
    )
    if "compact_rollout_slab_search_service_one_simulation_fast_path" in out:
        out["compact_rollout_slab_search_service_one_simulation_fast_path_count"] = (
            1.0
            if bool(out["compact_rollout_slab_search_service_one_simulation_fast_path"])
            else 0.0
        )
    _copy_first_profile_field(
        out,
        profile_telemetry,
        "compact_rollout_slab_search_service_recurrent_inference_calls",
        "compact_torch_search_recurrent_inference_calls",
    )
    for suffix in (
        "defer_one_simulation_replay_payload_requested",
        "defer_one_simulation_replay_payload_used",
        "deferred_one_simulation_replay_flush_pending",
    ):
        _copy_first_profile_field(
            out,
            profile_telemetry,
            f"compact_rollout_slab_search_service_{suffix}",
            f"compact_torch_search_service_{suffix}",
        )
    for suffix in (
        "one_simulation_replay_materialization_deferred",
        "deferred_one_simulation_action_model_state_digest",
        "deferred_one_simulation_action_policy_refresh_count",
        "deferred_one_simulation_action_policy_version_ref",
        "deferred_one_simulation_action_model_version_ref",
        "deferred_one_simulation_action_policy_source",
        "deferred_one_simulation_action_learner_update_count",
        "deferred_one_simulation_model_identity_match",
        "deferred_one_simulation_model_refresh_crossed_count",
        "pending_deferred_replay_payload_count",
    ):
        _copy_first_profile_field(
            out,
            profile_telemetry,
            f"compact_rollout_slab_search_service_{suffix}",
            f"compact_torch_search_{suffix}",
        )
    for suffix in (
        "obs_h2d_bytes",
        "mask_h2d_bytes",
        "action_d2h_bytes",
        "replay_payload_d2h_bytes",
        "root_observation_copy_bytes",
        "python_rows_materialized",
        "rnd_materialized_rows",
        "resident_observation_h2d_bytes",
        "resident_observation_d2h_bytes",
        "resident_observation_host_fallback_count",
    ):
        _copy_first_profile_field(
            out,
            profile_telemetry,
            f"compact_rollout_slab_{suffix}",
            f"lightzero_mcts_arrays_boundary_{suffix}",
            f"lightzero_array_ceiling_{suffix}",
            f"mctx_compact_search_service_{suffix}",
            f"compact_torch_search_service_{suffix}",
            f"fixed_shape_batched_search_owner_{suffix}",
        )


def _search_service_supports_two_phase(search_service: Any) -> bool:
    supported = getattr(search_service, "supports_two_phase_compact_search", None)
    has_methods = callable(getattr(search_service, "run_action_step", None)) and callable(
        getattr(search_service, "flush_replay_payload", None)
    )
    if supported is not None:
        return bool(supported) and has_methods
    return has_methods


def _compact_policy_refresh_metadata_from_search_service(
    search_service: Any,
) -> dict[str, Any]:
    state_fn = getattr(search_service, "policy_refresh_search_worker_state", None)
    if not callable(state_fn):
        return {}
    try:
        state = state_fn()
    except ValueError as exc:
        raise ReplayCompatibilityError(
            "compact rollout slab policy-refresh worker state is invalid"
        ) from exc
    state_get = getattr(state, "get", None)
    if not callable(state_get):
        raise ReplayCompatibilityError(
            "compact rollout slab policy-refresh worker state must be a mapping"
        )
    if not bool(state_get("refresh_applied", False)):
        return {}
    try:
        return compact_policy_refresh_metadata_from_state_v1(state)
    except ValueError as exc:
        raise ReplayCompatibilityError(
            "compact rollout slab policy-refresh metadata is invalid"
        ) from exc


def _owner_mechanics_step_frame_digest_from_batch_v1(compact_batch: Any) -> str:
    parts = (
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "action_mask"), dtype=np.bool_)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "reward"), dtype=np.float32)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "done"), dtype=np.bool_)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "policy_env_row"), dtype=np.int32)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "policy_player"), dtype=np.int32)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "joint_action"), dtype=np.int16)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "terminated"), dtype=np.bool_)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "truncated"), dtype=np.bool_)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "terminal_reason"), dtype=np.int16)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "episode_step"), dtype=np.int32)
        ),
        compact_search_array_digest_v1(
            np.asarray(getattr(compact_batch, "round_id"), dtype=np.int32)
        ),
    )
    return ":".join(parts)


def _owner_mechanics_step_frame_handle_metadata_v1(
    compact_batch: Any,
) -> dict[str, Any]:
    metadata = getattr(compact_batch, "metadata", None)
    if not isinstance(metadata, Mapping) or not bool(
        metadata.get("compact_owner_mechanics_step_boundary", False)
    ):
        return {}
    handle = getattr(compact_batch, "step_frame_handle", None)
    if handle is None:
        raise ReplayCompatibilityError("owner mechanics step frame handle is missing")
    schema_id = str(getattr(handle, "schema_id", ""))
    if schema_id != COMPACT_OWNER_MECHANICS_STEP_FRAME_HANDLE_SCHEMA_ID:
        raise ReplayCompatibilityError("owner mechanics step frame handle schema mismatch")
    slot_id = int(getattr(handle, "slot_id", -1))
    generation = int(getattr(handle, "generation", -1))
    ring_slot_count = int(
        metadata.get("compact_owner_mechanics_step_frame_handle_ring_slot_count", 0)
        or 0
    )
    if ring_slot_count <= 0 or slot_id < 0 or slot_id >= ring_slot_count:
        raise ReplayCompatibilityError("owner mechanics step frame handle slot mismatch")
    if generation < 0:
        raise ReplayCompatibilityError(
            "owner mechanics step frame handle generation mismatch"
        )
    if generation % ring_slot_count != slot_id:
        raise ReplayCompatibilityError(
            "owner mechanics step frame generation slot mismatch"
        )
    handle_batch_size = int(getattr(handle, "batch_size", -1))
    handle_player_count = int(getattr(handle, "player_count", -1))
    if handle_batch_size <= 0 or handle_player_count <= 0:
        raise ReplayCompatibilityError(
            "owner mechanics step frame handle shape mismatch"
        )
    expected_shape = (handle_batch_size, handle_player_count)
    if np.asarray(getattr(compact_batch, "joint_action"), dtype=np.int16).shape != expected_shape:
        raise ReplayCompatibilityError(
            "owner mechanics step frame handle shape mismatch"
        )
    if (
        np.asarray(getattr(compact_batch, "action_mask"), dtype=np.bool_).shape[:2]
        != expected_shape
    ):
        raise ReplayCompatibilityError(
            "owner mechanics step frame handle shape mismatch"
        )
    resident_observation = getattr(compact_batch, "resident_observation", None)
    if resident_observation is not None and (
        int(getattr(resident_observation, "batch_size", -1)) != handle_batch_size
        or int(getattr(resident_observation, "player_count", -1)) != handle_player_count
    ):
        raise ReplayCompatibilityError(
            "owner mechanics step frame handle shape mismatch"
        )
    metadata_slot_id = int(
        metadata.get("compact_owner_mechanics_step_frame_handle_slot_id", -1)
        if metadata.get("compact_owner_mechanics_step_frame_handle_slot_id") is not None
        else -1
    )
    if metadata_slot_id != slot_id:
        raise ReplayCompatibilityError(
            "owner mechanics step frame metadata slot mismatch"
        )
    metadata_generation = int(
        metadata.get("compact_owner_mechanics_step_frame_handle_generation", -1)
        if metadata.get("compact_owner_mechanics_step_frame_handle_generation") is not None
        else -1
    )
    if metadata_generation != generation:
        raise ReplayCompatibilityError(
            "owner mechanics step frame metadata generation mismatch"
        )
    if bool(metadata.get("compact_owner_mechanics_step_frame_handle_ring_used", False)):
        slot_generation = getattr(compact_batch, "slot_generation", None)
        if slot_generation is None:
            raise ReplayCompatibilityError(
                "owner mechanics step frame current generation is missing"
            )
        slot_generation_array = np.asarray(slot_generation, dtype=np.int64).reshape(-1)
        if slot_generation_array.shape != (1,):
            raise ReplayCompatibilityError(
                "owner mechanics step frame current generation shape mismatch"
            )
        if int(slot_generation_array[0]) != generation:
            raise ReplayCompatibilityError(
                "owner mechanics step frame stale generation"
            )
    digest = str(getattr(handle, "digest", ""))
    if not digest:
        raise ReplayCompatibilityError("owner mechanics step frame handle missing digest")
    metadata_digest = str(
        metadata.get("compact_owner_mechanics_step_frame_handle_digest", "")
    )
    if metadata_digest and metadata_digest != digest:
        raise ReplayCompatibilityError("owner mechanics step frame metadata digest mismatch")
    expected_digest = _owner_mechanics_step_frame_digest_from_batch_v1(compact_batch)
    if expected_digest != digest:
        raise ReplayCompatibilityError("owner mechanics step frame digest mismatch")
    return {
        "compact_owner_mechanics_step_frame_handle_schema_id": schema_id,
        "compact_owner_mechanics_step_frame_handle_published": True,
        "compact_owner_mechanics_step_frame_handle_consumed": True,
        "compact_owner_mechanics_step_frame_handle_publish_count": int(
            metadata.get("compact_owner_mechanics_step_frame_handle_publish_count", 1)
            or 1
        ),
        "compact_owner_mechanics_step_frame_handle_consume_count": 1,
        "compact_owner_mechanics_step_frame_handle_ring_used": bool(
            metadata.get("compact_owner_mechanics_step_frame_handle_ring_used", False)
        ),
        "compact_owner_mechanics_step_frame_handle_ring_slot_count": int(
            ring_slot_count
        ),
        "compact_owner_mechanics_step_frame_handle_slot_id": int(slot_id),
        "compact_owner_mechanics_step_frame_handle_generation": int(generation),
        "compact_owner_mechanics_step_frame_handle_digest": digest,
        "compact_owner_mechanics_step_frame_handle_digest_verified": True,
        "compact_owner_mechanics_step_frame_handle_owner_digest_verified": True,
        "compact_owner_mechanics_step_frame_handle_resident_observation_present": bool(
            getattr(handle, "resident_observation_handle_present", False)
        ),
    }


def _root_build_request_from_owner_step_frame_slot_v1(
    compact_batch: Any,
    *,
    search_lane: str,
    metadata: Mapping[str, Any],
    copy_observation: bool,
    resident_host_observation_stub: bool,
) -> CompactRootBuildRequestV1:
    resident_observation = getattr(compact_batch, "resident_observation", None)
    if resident_observation is None:
        raise ReplayCompatibilityError(
            "owner step-frame root request requires resident_observation"
        )
    batch_size = int(getattr(resident_observation, "batch_size"))
    player_count = int(getattr(resident_observation, "player_count"))
    root_count = batch_size * player_count
    stack_shape = tuple(int(dim) for dim in getattr(resident_observation, "stack_shape"))
    action_mask = np.asarray(compact_batch.action_mask, dtype=np.bool_)
    if action_mask.shape != (batch_size, player_count, ACTION_COUNT):
        raise ReplayCompatibilityError("owner step-frame action_mask shape mismatch")
    reward = np.asarray(compact_batch.reward, dtype=np.float32)
    done = np.asarray(compact_batch.done, dtype=np.bool_)
    policy_env_id = np.asarray(compact_batch.policy_env_id, dtype=np.int32)
    policy_env_row = np.asarray(compact_batch.policy_env_row, dtype=np.int32)
    policy_player = np.asarray(compact_batch.policy_player, dtype=np.int32)
    target_reward = reward.reshape(root_count, 1)
    done_root = np.repeat(done, player_count).astype(np.bool_, copy=False)
    to_play = np.full((root_count,), -1, dtype=np.int64)
    active_root_mask = np.logical_and(
        ~done_root,
        action_mask.reshape(root_count, ACTION_COUNT).any(axis=1),
    )
    request_metadata = {
        str(key): value
        for key, value in dict(metadata).items()
        if value is None or isinstance(value, (bool, int, float, str))
    }
    request_metadata.update(
        {
            "contract_id": COMPACT_SEARCH_REPLAY_SERVICE_CONTRACT_ID,
            "schema_id": COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID,
            "root_build_request_schema_id": COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID,
            "root_build_request_kind": COMPACT_ROOT_BUILD_REQUEST_KIND_RESIDENT_ROOT_VIEW,
            "search_lane": str(search_lane),
            "observation_source": COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
            "resident_host_observation_stub_requested": bool(
                resident_host_observation_stub
            ),
            "batch_size": int(batch_size),
            "player_count": int(player_count),
            "root_count": int(root_count),
            "stack_shape": list(stack_shape),
            "mechanics_outcome_schema_id": COMPACT_ROOT_MECHANICS_OUTCOME_SCHEMA_ID,
            "mechanics_outcome_sidecars_present": True,
            "compact_owner_step_frame_root_build_request_used": True,
            "compact_owner_step_frame_root_build_request_from_batch_helper_used": False,
            "compact_owner_step_frame_root_request_sidecar_array_bytes": 0,
            "compact_owner_step_frame_root_request_sidecar_field_count": 0,
        }
    )
    return CompactRootBuildRequestV1(
        schema_id=COMPACT_ROOT_BUILD_REQUEST_SCHEMA_ID,
        request_kind=COMPACT_ROOT_BUILD_REQUEST_KIND_RESIDENT_ROOT_VIEW,
        search_lane=str(search_lane),
        metadata=request_metadata,
        copy_observation=bool(copy_observation),
        observation_source=COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1,
        resident_host_observation_stub=bool(resident_host_observation_stub),
        batch_size=int(batch_size),
        player_count=int(player_count),
        root_count=int(root_count),
        stack_shape=stack_shape,
        observation=None,
        action_mask=action_mask,
        reward=reward,
        done=done,
        policy_env_id=policy_env_id,
        policy_env_row=policy_env_row,
        policy_player=policy_player,
        target_reward=target_reward,
        done_root=done_root,
        to_play=to_play,
        active_root_mask=active_root_mask,
        final_observation=None,
        final_observation_row_mask=np.asarray(
            compact_batch.final_observation_row_mask,
            dtype=np.bool_,
        ),
        terminal_row_mask=np.asarray(compact_batch.terminal_row_mask, dtype=np.bool_),
        autoreset_row_mask=np.asarray(compact_batch.autoreset_row_mask, dtype=np.bool_),
        joint_action=np.asarray(compact_batch.joint_action, dtype=np.int16),
        resident_observation=resident_observation,
        terminated=np.asarray(compact_batch.terminated, dtype=np.bool_),
        truncated=np.asarray(compact_batch.truncated, dtype=np.bool_),
        final_reward_map=np.asarray(compact_batch.final_reward_map, dtype=np.float32),
    )


def _root_build_request_scalar_metadata_v1(
    root_build_request: CompactRootBuildRequestV1 | CompactRootActionContextV1,
) -> dict[str, Any]:
    metadata = getattr(root_build_request, "metadata", None)
    if not isinstance(metadata, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key)
        if not (
            key_text.startswith("compact_owner_")
            or key_text.startswith("compact_rollout_slab_")
            or key_text in {"root_build_request_schema_id", "root_build_request_kind"}
        ):
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            result[key_text] = value
    return result


def _compact_search_replay_payload_d2h_bytes(replay_payload: Any) -> int:
    total = int(np.asarray(replay_payload.visit_policy).nbytes)
    total += int(np.asarray(replay_payload.root_value).nbytes)
    if replay_payload.raw_visit_counts is not None:
        total += int(np.asarray(replay_payload.raw_visit_counts).nbytes)
    if replay_payload.predicted_value is not None:
        total += int(np.asarray(replay_payload.predicted_value).nbytes)
    if replay_payload.predicted_policy_logits is not None:
        total += int(np.asarray(replay_payload.predicted_policy_logits).nbytes)
    return total


def _compact_index_row_count(rows: Any) -> int:
    shape = tuple(int(dim) for dim in getattr(rows.action, "shape", ()))
    if not shape:
        raise ReplayCompatibilityError("compact replay index rows have no row axis")
    return int(shape[0])


def _copy_replay_field(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    return np.asarray(value).copy()


def _copy_compact_replay_index_rows(
    rows: CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1,
) -> CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1:
    return replace(
        rows,
        metadata=dict(rows.metadata),
        compact_root_row=_copy_replay_field(rows.compact_root_row),
        policy_env_id=_copy_replay_field(rows.policy_env_id),
        policy_row=_copy_replay_field(rows.policy_row),
        env_row=_copy_replay_field(rows.env_row),
        player=_copy_replay_field(rows.player),
        action=_copy_replay_field(rows.action),
        action_mask=_copy_replay_field(rows.action_mask),
        policy_target=_copy_replay_field(rows.policy_target),
        root_value=_copy_replay_field(rows.root_value),
        reward=_copy_replay_field(rows.reward),
        final_reward=_copy_replay_field(rows.final_reward),
        done=_copy_replay_field(rows.done),
        terminated=_copy_replay_field(rows.terminated),
        truncated=_copy_replay_field(rows.truncated),
        next_final_observation_row=_copy_replay_field(rows.next_final_observation_row),
        to_play=_copy_replay_field(rows.to_play),
    )


def _owner_search_replay_append_entry(
    rows: CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1,
    *,
    previous_compact_batch: Any | None,
    current_compact_batch: Any | None,
    record_index: int | None,
) -> Any:
    if previous_compact_batch is None or current_compact_batch is None:
        return rows
    host_rows = _owner_search_host_index_rows(rows)
    metadata = {
        "compact_owner_search_replay_append_entry": True,
        "compact_owner_search_replay_append_entry_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_INDEX_ENTRY_SCHEMA_ID
        ),
        "record_index": int(
            record_index if record_index is not None else getattr(rows, "record_index")
        ),
        "next_record_index": int(getattr(rows, "next_record_index")),
        "index_row_count": _compact_index_row_count(host_rows),
        "host_only": True,
        "compact_owner_search_replay_append_index_only": True,
        "compact_owner_search_replay_append_carries_compact_batches": False,
    }
    return CompactOwnerSearchReplayAppendIndexEntryV1(
        schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_INDEX_ENTRY_SCHEMA_ID,
        record_index=int(metadata["record_index"]),
        next_record_index=int(metadata["next_record_index"]),
        index_rows=host_rows,
        metadata=metadata,
    )


def _action_step_owner_materializes_replay(action_step: CompactSearchActionStepV1) -> bool:
    metadata = dict(action_step.metadata or {})
    return bool(metadata.get("compact_owner_search_owner_materializes_replay"))


def _owner_search_replay_append_transition_entry(
    pending: _PendingCompactSearchV1,
    *,
    action_step: CompactSearchActionStepV1,
    next_joint_action: np.ndarray,
    next_batch: Any,
    policy_source: str,
) -> CompactOwnerSearchReplayAppendTransitionEntryV1:
    return _owner_search_replay_append_transition_entry_from_facts(
        _owner_search_replay_append_transition_facts(
            pending,
            action_step=action_step,
            next_joint_action=next_joint_action,
            next_batch=next_batch,
            policy_source=policy_source,
        )
    )


def _owner_search_replay_append_transition_facts(
    pending: _PendingCompactSearchV1,
    *,
    action_step: CompactSearchActionStepV1,
    next_joint_action: np.ndarray,
    next_batch: Any,
    policy_source: str,
) -> _OwnerSearchTransitionFactsV1:
    terminal_metadata = _compact_terminal_metadata_from_batch(next_batch)
    return _OwnerSearchTransitionFactsV1(
        record_index=int(pending.record_index),
        next_record_index=int(pending.record_index + 1),
        replay_payload_handle=str(action_step.replay_payload_handle),
        selected_action_digest=str(action_step.metadata.get("selected_action_digest") or ""),
        search_replay_payload_digest=str(
            action_step.metadata.get("search_replay_payload_digest") or ""
        ),
        next_joint_action=np.asarray(next_joint_action, dtype=np.int16).copy(),
        next_reward=np.asarray(next_batch.reward, dtype=np.float32).copy(),
        next_done=np.asarray(next_batch.done, dtype=np.bool_).copy(),
        next_terminated=np.asarray(
            getattr(next_batch, "terminated", next_batch.done),
            dtype=np.bool_,
        ).copy(),
        next_truncated=np.asarray(
            getattr(
                next_batch,
                "truncated",
                np.zeros_like(next_batch.done, dtype=np.bool_),
            ),
            dtype=np.bool_,
        ).copy(),
        next_final_reward_map=np.asarray(
            getattr(next_batch, "final_reward_map", next_batch.reward),
            dtype=np.float32,
        ).copy(),
        next_final_observation_row_mask=np.asarray(
            next_batch.final_observation_row_mask,
            dtype=np.bool_,
        ).copy(),
        policy_source=str(policy_source),
        terminal_metadata=terminal_metadata,
    )


def _owner_search_replay_append_derived_transition_facts(
    pending: _PendingCompactSearchV1,
    *,
    action_step: CompactSearchActionStepV1,
    applied_action: np.ndarray,
    policy_source: str,
) -> _OwnerSearchDerivedTransitionFactsV1:
    return _OwnerSearchDerivedTransitionFactsV1(
        record_index=int(pending.record_index),
        next_record_index=int(pending.record_index + 1),
        replay_payload_handle=str(action_step.replay_payload_handle),
        selected_action_digest=str(action_step.metadata.get("selected_action_digest") or ""),
        search_replay_payload_digest=str(
            action_step.metadata.get("search_replay_payload_digest") or ""
        ),
        applied_action_count=int(np.asarray(applied_action, dtype=np.int16).size),
        applied_action_checksum=_owner_action_checksum_v1(applied_action),
        policy_source=str(policy_source),
    )


def _owner_search_replay_append_transition_entry_from_facts(
    facts: _OwnerSearchTransitionFactsV1,
) -> CompactOwnerSearchReplayAppendTransitionEntryV1:
    metadata = {
        "compact_owner_search_replay_append_entry": True,
        "compact_owner_search_replay_append_entry_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID
        ),
        "compact_owner_search_replay_append_transition_only": True,
        "compact_owner_search_replay_append_owner_materializes_rows": True,
        "compact_owner_search_replay_append_carries_index_rows": False,
        "compact_owner_search_replay_append_carries_compact_batches": False,
        "record_index": int(facts.record_index),
        "next_record_index": int(facts.next_record_index),
        "replay_payload_handle": str(facts.replay_payload_handle),
        "selected_action_digest": str(facts.selected_action_digest),
        "search_replay_payload_digest": str(facts.search_replay_payload_digest),
        **dict(facts.terminal_metadata),
    }
    return CompactOwnerSearchReplayAppendTransitionEntryV1(
        schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID,
        record_index=int(facts.record_index),
        next_record_index=int(facts.next_record_index),
        replay_payload_handle=str(facts.replay_payload_handle),
        next_joint_action=facts.next_joint_action.copy(),
        next_reward=facts.next_reward.copy(),
        next_done=facts.next_done.copy(),
        next_terminated=facts.next_terminated.copy(),
        next_truncated=facts.next_truncated.copy(),
        next_final_reward_map=facts.next_final_reward_map.copy(),
        next_final_observation_row_mask=facts.next_final_observation_row_mask.copy(),
        policy_source=str(facts.policy_source),
        metadata=metadata,
    )


def _owner_search_replay_append_transition_batch(
    facts: tuple[_OwnerSearchTransitionFactsV1, ...],
    *,
    max_entries_per_batch: int,
) -> CompactOwnerSearchReplayAppendTransitionBatchV1:
    if not facts:
        raise ReplayCompatibilityError("transition batch requires at least one entry")
    transition_count = len(facts)
    fixed_capacity = int(max_entries_per_batch)
    if fixed_capacity <= 1:
        raise ReplayCompatibilityError("transition batch capacity must exceed one")
    if transition_count > fixed_capacity:
        raise ReplayCompatibilityError("transition batch exceeds fixed capacity")
    record_indices = np.asarray(
        [int(item.record_index) for item in facts],
        dtype=np.int64,
    )
    next_record_indices = np.asarray(
        [int(item.next_record_index) for item in facts],
        dtype=np.int64,
    )
    next_joint_action = np.stack(
        [np.asarray(item.next_joint_action, dtype=np.int16) for item in facts],
        axis=0,
    )
    next_reward = np.stack(
        [np.asarray(item.next_reward, dtype=np.float32) for item in facts],
        axis=0,
    )
    next_done = np.stack(
        [np.asarray(item.next_done, dtype=np.bool_) for item in facts],
        axis=0,
    )
    next_terminated = np.stack(
        [np.asarray(item.next_terminated, dtype=np.bool_) for item in facts],
        axis=0,
    )
    next_truncated = np.stack(
        [np.asarray(item.next_truncated, dtype=np.bool_) for item in facts],
        axis=0,
    )
    next_final_reward_map = np.stack(
        [np.asarray(item.next_final_reward_map, dtype=np.float32) for item in facts],
        axis=0,
    )
    next_final_observation_row_mask = np.stack(
        [np.asarray(item.next_final_observation_row_mask, dtype=np.bool_) for item in facts],
        axis=0,
    )
    replay_payload_handles = tuple(str(item.replay_payload_handle) for item in facts)
    selected_action_digests = tuple(str(item.selected_action_digest) for item in facts)
    search_replay_payload_digests = tuple(str(item.search_replay_payload_digest) for item in facts)
    policy_sources = tuple(str(item.policy_source) for item in facts)
    if len(set(policy_sources)) != 1:
        raise ReplayCompatibilityError("transition batch policy sources must match")
    digest = (
        f"{compact_search_array_digest_v1(record_indices)}:"
        f"{compact_search_array_digest_v1(next_joint_action)}"
    )
    transport_bytes = int(
        sum(
            int(array.nbytes)
            for array in (
                record_indices,
                next_record_indices,
                next_joint_action,
                next_reward,
                next_done,
                next_terminated,
                next_truncated,
                next_final_reward_map,
                next_final_observation_row_mask,
            )
        )
    )
    metadata = {
        "compact_owner_search_replay_append_entry": True,
        "compact_owner_search_replay_append_entry_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_replay_append_transition_batch": True,
        "compact_owner_search_replay_append_transition_batch_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_replay_append_transition_batch_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
        ),
        "compact_owner_search_replay_append_transition_batch_transition_count": int(
            transition_count
        ),
        "compact_owner_search_replay_append_owner_materializes_rows": True,
        "compact_owner_search_replay_append_carries_index_rows": False,
        "compact_owner_search_replay_append_carries_compact_batches": False,
        "compact_owner_search_owner_replay_transport_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
        ),
        "compact_owner_search_owner_replay_transport_entry_count": 1,
        "compact_owner_search_owner_replay_transition_batch_enabled": True,
        "compact_owner_search_owner_replay_transition_batch_count": 1,
        "compact_owner_search_owner_replay_transition_batch_transition_count": int(
            transition_count
        ),
        "compact_owner_search_owner_replay_transition_legacy_entry_count": 0,
        "compact_owner_search_transition_batch_transport_requested": True,
        "compact_owner_search_transition_batch_transport_enabled": True,
        "compact_owner_search_transition_batch_transport_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED
        ),
        "compact_owner_search_transition_batch_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_transition_batch_count": 1,
        "compact_owner_search_transition_batch_entry_count": int(transition_count),
        "compact_owner_search_transition_batch_transport_entry_count": 1,
        "compact_owner_search_transition_batch_max_entries_per_batch": int(fixed_capacity),
        "compact_owner_search_transition_batch_fixed_capacity": int(fixed_capacity),
        "compact_owner_search_transition_batch_padding_count": int(
            fixed_capacity - transition_count
        ),
        "compact_owner_search_transition_batch_overflow_count": 0,
        "compact_owner_search_transition_batch_fallback_count": 0,
        "compact_owner_search_transition_batch_fallback_reason": "none",
        "compact_owner_search_transition_batch_pending_count": 0,
        "compact_owner_search_transition_batch_transport_bytes": int(transport_bytes),
        "compact_owner_search_transition_batch_digest": digest,
        "compact_owner_search_transition_batch_digest_verified": True,
        "record_index_first": int(record_indices[0]),
        "record_index_last": int(record_indices[-1]),
        "next_record_index_last": int(next_record_indices[-1]),
    }
    return CompactOwnerSearchReplayAppendTransitionBatchV1(
        schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID,
        transition_count=int(transition_count),
        record_indices=record_indices,
        next_record_indices=next_record_indices,
        replay_payload_handles=replay_payload_handles,
        selected_action_digests=selected_action_digests,
        search_replay_payload_digests=search_replay_payload_digests,
        next_joint_action=next_joint_action,
        next_reward=next_reward,
        next_done=next_done,
        next_terminated=next_terminated,
        next_truncated=next_truncated,
        next_final_reward_map=next_final_reward_map,
        next_final_observation_row_mask=next_final_observation_row_mask,
        policy_source=policy_sources[0],
        metadata=metadata,
    )


def _owner_search_replay_append_derived_transition_batch(
    facts: tuple[_OwnerSearchDerivedTransitionFactsV1, ...],
    *,
    max_entries_per_batch: int,
) -> CompactOwnerSearchReplayAppendDerivedTransitionBatchV1:
    if not facts:
        raise ReplayCompatibilityError("derived transition batch requires at least one entry")
    transition_count = len(facts)
    fixed_capacity = int(max_entries_per_batch)
    if fixed_capacity <= 1:
        raise ReplayCompatibilityError("derived transition batch capacity must exceed one")
    if transition_count > fixed_capacity:
        raise ReplayCompatibilityError("derived transition batch exceeds fixed capacity")
    record_indices = np.asarray(
        [int(item.record_index) for item in facts],
        dtype=np.int64,
    )
    next_record_indices = np.asarray(
        [int(item.next_record_index) for item in facts],
        dtype=np.int64,
    )
    applied_action_counts = np.asarray(
        [int(item.applied_action_count) for item in facts],
        dtype=np.int64,
    )
    applied_action_checksums = np.asarray(
        [int(item.applied_action_checksum) for item in facts],
        dtype=np.int64,
    )
    replay_payload_handles = tuple(str(item.replay_payload_handle) for item in facts)
    selected_action_digests = tuple(str(item.selected_action_digest) for item in facts)
    search_replay_payload_digests = tuple(str(item.search_replay_payload_digest) for item in facts)
    policy_sources = tuple(str(item.policy_source) for item in facts)
    if len(set(policy_sources)) != 1:
        raise ReplayCompatibilityError("derived transition batch policy sources must match")
    digest = (
        f"{compact_search_array_digest_v1(record_indices)}:"
        f"{compact_search_array_digest_v1(next_record_indices)}:"
        f"{compact_search_array_digest_v1(applied_action_checksums)}"
    )
    transport_bytes = int(
        sum(
            int(array.nbytes)
            for array in (
                record_indices,
                next_record_indices,
                applied_action_counts,
                applied_action_checksums,
            )
        )
    )
    metadata = {
        "compact_owner_search_replay_append_entry": True,
        "compact_owner_search_replay_append_entry_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_replay_append_transition_batch": True,
        "compact_owner_search_replay_append_transition_batch_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_replay_append_transition_batch_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
        ),
        "compact_owner_search_replay_append_transition_batch_transition_count": int(
            transition_count
        ),
        "compact_owner_search_replay_append_owner_materializes_rows": True,
        "compact_owner_search_replay_append_owner_derives_transition_outcome": True,
        "compact_owner_search_replay_append_carries_index_rows": False,
        "compact_owner_search_replay_append_carries_compact_batches": False,
        "compact_owner_search_replay_append_carries_parent_transition_outcomes": False,
        "compact_owner_search_owner_replay_transport_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
        ),
        "compact_owner_search_owner_local_transition_derivation_requested": True,
        "compact_owner_search_owner_local_transition_derivation_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_owner_local_transition_derivation_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
        ),
        "compact_owner_search_owner_local_transition_derivation_batch_count": 1,
        "compact_owner_search_owner_local_transition_derivation_transition_count": int(
            transition_count
        ),
        "compact_owner_search_owner_local_transition_derivation_transport_entry_count": 1,
        "compact_owner_search_owner_local_transition_derivation_transport_bytes": int(
            transport_bytes
        ),
        "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_bytes": 0,
        "compact_owner_search_owner_local_transition_derivation_parent_outcome_array_transport_field_count": 0,
        "compact_owner_search_owner_local_transition_derivation_digest": digest,
        "compact_owner_search_owner_local_transition_derivation_digest_verified": True,
        "compact_owner_search_owner_local_transition_derivation_fallback_count": 0,
        "compact_owner_search_owner_local_transition_derivation_fallback_reason": "none",
        "compact_owner_search_transition_batch_transport_requested": True,
        "compact_owner_search_transition_batch_transport_enabled": True,
        "compact_owner_search_transition_batch_transport_kind": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND
        ),
        "compact_owner_search_transition_batch_schema_id": (
            COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID
        ),
        "compact_owner_search_transition_batch_count": 1,
        "compact_owner_search_transition_batch_entry_count": int(transition_count),
        "compact_owner_search_transition_batch_transport_entry_count": 1,
        "compact_owner_search_transition_batch_max_entries_per_batch": int(max_entries_per_batch),
        "compact_owner_search_transition_batch_fixed_capacity": int(max_entries_per_batch),
        "compact_owner_search_transition_batch_padding_count": int(
            max(0, max_entries_per_batch - transition_count)
        ),
        "compact_owner_search_transition_batch_overflow_count": 0,
        "compact_owner_search_transition_batch_fallback_count": 0,
        "compact_owner_search_transition_batch_fallback_reason": "none",
        "compact_owner_search_transition_batch_pending_count": 0,
        "compact_owner_search_transition_batch_transport_bytes": int(transport_bytes),
        "compact_owner_search_transition_batch_digest": digest,
        "compact_owner_search_transition_batch_digest_verified": True,
        "record_index_first": int(record_indices[0]),
        "record_index_last": int(record_indices[-1]),
        "next_record_index_last": int(next_record_indices[-1]),
    }
    return CompactOwnerSearchReplayAppendDerivedTransitionBatchV1(
        schema_id=COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID,
        transition_count=int(transition_count),
        record_indices=record_indices,
        next_record_indices=next_record_indices,
        replay_payload_handles=replay_payload_handles,
        selected_action_digests=selected_action_digests,
        search_replay_payload_digests=search_replay_payload_digests,
        applied_action_counts=applied_action_counts,
        applied_action_checksums=applied_action_checksums,
        policy_source=policy_sources[0],
        metadata=metadata,
    )


def _owner_search_host_compact_batch(compact_batch: Any) -> Any:
    source = str(
        getattr(
            compact_batch,
            "observation_source",
            COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
        )
    )
    host_batch = _owner_search_host_value(compact_batch)
    if source == COMPACT_OBSERVATION_SOURCE_RESIDENT_DEVICE_V1:
        observation = _owner_search_host_observation_from_resident(compact_batch)
        final_observation = _owner_search_host_final_observation_from_resident(
            compact_batch,
            observation=observation,
        )
    elif source == COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1:
        observation = _owner_search_host_value(getattr(compact_batch, "observation"))
        final_observation = _owner_search_host_value(
            getattr(compact_batch, "final_observation", None)
        )
    else:
        raise ReplayCompatibilityError(
            f"unknown compact observation source for owner replay append: {source!r}"
        )
    updates = {
        "observation": observation,
        "final_observation": final_observation,
        "resident_observation": None,
        "observation_source": COMPACT_OBSERVATION_SOURCE_HOST_ARRAY_V1,
    }
    if is_dataclass(host_batch):
        return replace(host_batch, **updates)
    data = dict(vars(host_batch)) if hasattr(host_batch, "__dict__") else {}
    data.update(updates)
    return SimpleNamespace(**data)


def _owner_search_host_index_rows(
    rows: CompactReplayIndexRowsV1 | CompactDeviceReplayIndexRowsV1,
) -> CompactReplayIndexRowsV1:
    metadata = dict(_owner_search_host_value(getattr(rows, "metadata", {})))
    if isinstance(rows, CompactDeviceReplayIndexRowsV1):
        metadata["owner_search_host_clone_from_device_replay_index_rows"] = True
    return CompactReplayIndexRowsV1(
        metadata=metadata,
        record_index=int(getattr(rows, "record_index")),
        next_record_index=int(getattr(rows, "next_record_index")),
        compact_root_row=_owner_search_host_value(rows.compact_root_row),
        policy_env_id=_owner_search_host_value(rows.policy_env_id),
        policy_row=_owner_search_host_value(rows.policy_row),
        env_row=_owner_search_host_value(rows.env_row),
        player=_owner_search_host_value(rows.player),
        action=_owner_search_host_value(rows.action),
        action_mask=_owner_search_host_value(rows.action_mask),
        policy_target=_owner_search_host_value(rows.policy_target),
        root_value=_owner_search_host_value(rows.root_value),
        reward=_owner_search_host_value(rows.reward),
        final_reward=_owner_search_host_value(rows.final_reward),
        done=_owner_search_host_value(rows.done),
        terminated=_owner_search_host_value(rows.terminated),
        truncated=_owner_search_host_value(rows.truncated),
        next_final_observation_row=_owner_search_host_value(rows.next_final_observation_row),
        to_play=_owner_search_host_value(rows.to_play),
        policy_source=str(rows.policy_source),
    )


def _owner_search_host_observation_from_resident(compact_batch: Any) -> np.ndarray:
    resident = getattr(compact_batch, "resident_observation", None)
    if resident is None:
        raise ReplayCompatibilityError(
            "owner replay append cannot materialize resident observation without "
            "resident_observation"
        )
    batch_size = int(getattr(resident, "batch_size"))
    player_count = int(getattr(resident, "player_count"))
    stack_shape = tuple(int(dim) for dim in getattr(resident, "stack_shape"))
    root_observation = getattr(resident, "root_device_observation", None)
    if root_observation is not None:
        host_root = np.asarray(_owner_search_host_value(root_observation)).copy()
        expected_root_shape = (batch_size * player_count, *stack_shape)
        if tuple(int(dim) for dim in host_root.shape) != expected_root_shape:
            raise ReplayCompatibilityError(
                "owner replay append resident root observation shape mismatch"
            )
        return host_root.reshape(batch_size, player_count, *stack_shape).copy()
    device_observation = getattr(resident, "device_observation", None)
    if device_observation is None:
        raise ReplayCompatibilityError(
            "owner replay append resident observation has no device tensor"
        )
    host = np.asarray(_owner_search_host_value(device_observation)).copy()
    expected_shape = (batch_size, player_count, *stack_shape)
    if tuple(int(dim) for dim in host.shape) != expected_shape:
        raise ReplayCompatibilityError("owner replay append resident observation shape mismatch")
    return host


def _owner_search_host_final_observation_from_resident(
    compact_batch: Any,
    *,
    observation: np.ndarray,
) -> np.ndarray | None:
    mask = np.asarray(
        getattr(
            compact_batch,
            "final_observation_row_mask",
            np.zeros((observation.shape[0],), dtype=np.bool_),
        ),
        dtype=np.bool_,
    ).reshape(-1)
    if not bool(mask.any()):
        return None
    resident = getattr(compact_batch, "resident_observation", None)
    if resident is None:
        raise ReplayCompatibilityError(
            "owner replay append cannot materialize resident final observation without "
            "resident_observation"
        )
    batch_size = int(getattr(resident, "batch_size"))
    player_count = int(getattr(resident, "player_count"))
    stack_shape = tuple(int(dim) for dim in getattr(resident, "stack_shape"))
    root_final = getattr(resident, "root_final_device_observation", None)
    if root_final is not None:
        host_root = np.asarray(_owner_search_host_value(root_final)).copy()
        expected_root_shape = (batch_size * player_count, *stack_shape)
        if tuple(int(dim) for dim in host_root.shape) != expected_root_shape:
            raise ReplayCompatibilityError(
                "owner replay append resident root final observation shape mismatch"
            )
        return host_root.reshape(batch_size, player_count, *stack_shape).copy()
    dense_final = getattr(resident, "final_device_observation", None)
    if dense_final is not None:
        host = np.asarray(_owner_search_host_value(dense_final)).copy()
        expected_shape = (batch_size, player_count, *stack_shape)
        if tuple(int(dim) for dim in host.shape) != expected_shape:
            raise ReplayCompatibilityError(
                "owner replay append resident final observation shape mismatch"
            )
        return host
    sparse_final = getattr(resident, "final_device_observation_rows", None)
    sparse_indices = getattr(resident, "final_device_observation_row_indices", None)
    if sparse_final is not None and sparse_indices is not None:
        host = observation.copy()
        rows = np.asarray(_owner_search_host_value(sparse_final)).copy()
        indices = np.asarray(sparse_indices, dtype=np.int64).reshape(-1)
        expected_sparse_shape = (indices.shape[0], player_count, *stack_shape)
        if tuple(int(dim) for dim in rows.shape) != expected_sparse_shape:
            raise ReplayCompatibilityError(
                "owner replay append resident sparse final observation shape mismatch"
            )
        if bool((indices < 0).any()) or bool((indices >= batch_size).any()):
            raise ReplayCompatibilityError(
                "owner replay append resident sparse final observation row out of range"
            )
        host[indices] = rows
        return host
    host_final = getattr(compact_batch, "final_observation", None)
    if host_final is not None:
        host = np.asarray(_owner_search_host_value(host_final)).copy()
        if tuple(int(dim) for dim in host.shape) == tuple(int(dim) for dim in observation.shape):
            return host
    raise ReplayCompatibilityError(
        "owner replay append resident terminal rows require resident final observation"
    )


def _owner_search_host_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return np.asarray(numpy()).copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {str(key): _owner_search_host_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_owner_search_host_value(item) for item in value)
    if isinstance(value, list):
        return [_owner_search_host_value(item) for item in value]
    if is_dataclass(value):
        return replace(
            value,
            **{
                field.name: _owner_search_host_value(getattr(value, field.name))
                for field in fields(value)
            },
        )
    if hasattr(value, "__dict__"):
        return SimpleNamespace(
            **{str(key): _owner_search_host_value(item) for key, item in vars(value).items()}
        )
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    return value


def _positive_int(value: int, name: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise ReplayCompatibilityError(f"{name} must be positive")
    return integer


__all__ = [
    "COMPACT_OWNER_SEARCH_REPLAY_APPEND_ENTRY_SCHEMA_ID",
    "COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_KIND",
    "COMPACT_OWNER_SEARCH_REPLAY_APPEND_DERIVED_TRANSITION_BATCH_SCHEMA_ID",
    "COMPACT_OWNER_SEARCH_REPLAY_APPEND_INDEX_ENTRY_SCHEMA_ID",
    "COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_KIND_FIXED",
    "COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_BATCH_SCHEMA_ID",
    "COMPACT_OWNER_SEARCH_REPLAY_APPEND_TRANSITION_ENTRY_SCHEMA_ID",
    "COMPACT_OWNER_SEARCH_SLAB_BYPASS_KIND_DIRECT_TRANSITION",
    "COMPACT_OWNER_SEARCH_DIRECT_STEP_DISPATCH_HANDLE_SCHEMA_ID",
    "COMPACT_ROLLOUT_SLAB_STEP_SCHEMA_ID",
    "CompactOwnerSearchDirectStepDispatchHandleV1",
    "CompactOwnerSearchDirectStepperV1",
    "CompactOwnerSearchReplayAppendDerivedTransitionBatchV1",
    "CompactOwnerSearchReplayAppendEntryV1",
    "CompactOwnerSearchReplayAppendIndexEntryV1",
    "CompactOwnerSearchReplayAppendTransitionBatchV1",
    "CompactOwnerSearchReplayAppendTransitionEntryV1",
    "CompactRolloutSlab",
    "CompactRolloutSlabStepV1",
    "selected_joint_action_from_action_step",
    "selected_joint_action_from_search_result",
]
