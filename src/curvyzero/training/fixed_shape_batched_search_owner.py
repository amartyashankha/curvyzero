"""Profile-only fixed-shape batched search owner.

This is a deliberately small service object behind ``CompactSearchServiceV1``:
it accepts a fixed-size ``CompactRootBatchV1`` and returns a checked compact
search result without touching LightZero CTree, row lists, trainers, or Modal.
The current policy is deterministic first-legal action selection; the point of
this first owner is the fixed-shape boundary and telemetry, not MCTS parity.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from curvyzero.training.compact_policy_row_bridge import CompactRootBatchV1
from curvyzero.training.compact_policy_row_bridge import CompactSearchResultV1
from curvyzero.training.compact_policy_row_bridge import validate_compact_search_result_v1
from curvyzero.training.compact_search_service import CompactSearchActionStepV1
from curvyzero.training.compact_search_service import CompactDeviceSearchReplayPayloadV1
from curvyzero.training.compact_search_service import CompactSearchReplayPayloadV1
from curvyzero.training.compact_search_service import (
    COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
)
from curvyzero.training.compact_search_service import (
    compact_search_action_step_v1_from_result,
)
from curvyzero.training.compact_search_service import (
    compact_search_deferred_replay_payload_digest_v1,
)
from curvyzero.training.compact_search_service import (
    compact_search_replay_payload_v1_from_result,
)
from curvyzero.training.multiplayer_source_state_target_rows import ACTION_COUNT
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError


FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL = "fixed_shape_batched_search_owner_profile_only_v0"
FIXED_SHAPE_BATCHED_SEARCH_OWNER_SEMANTICS = (
    "profile_only_fixed_R_A3_padded_numpy_no_ctree_no_tolist_no_per_sim_d2h"
)


class FixedShapeBatchedSearchOwnerV0:
    """Minimal fixed-shape compact search owner for profile probes.

    The owner keeps reusable padded buffers sized ``[root_count, 3]`` and emits
    only active roots through the public compact-search contract.
    """

    search_impl = FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL
    profile_only = True
    trainer_defaults_changed = False
    touches_live_runs = False
    supports_two_phase_compact_search = True

    def __init__(
        self,
        *,
        root_count: int,
        num_simulations: int,
        device: Any | None = None,
        model: Any | None = None,
    ) -> None:
        fixed_roots = int(root_count)
        if fixed_roots <= 0:
            raise ValueError("root_count must be positive")
        simulations = int(num_simulations)
        if simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if ACTION_COUNT != 3:
            raise ValueError("FixedShapeBatchedSearchOwnerV0 requires A=3")

        self.root_count = fixed_roots
        self.action_count = ACTION_COUNT
        self.num_simulations = simulations
        self._device = device
        self._model = model
        self._run_count = 0
        self._last_profile_telemetry: dict[str, Any] = {}
        self._selected_action = np.zeros((fixed_roots,), dtype=np.int16)
        self._visit_policy = np.zeros((fixed_roots, ACTION_COUNT), dtype=np.float32)
        self._raw_visit_counts = np.zeros((fixed_roots, ACTION_COUNT), dtype=np.float32)
        self._root_value = np.zeros((fixed_roots,), dtype=np.float32)
        self._replay_payload_counter = 0
        self._pending_replay_payloads: dict[str, CompactSearchReplayPayloadV1] = {}

    @property
    def last_profile_telemetry(self) -> dict[str, Any]:
        """Return the most recent run telemetry as plain Python values."""

        return dict(self._last_profile_telemetry)

    def run(self, root_batch: CompactRootBatchV1) -> CompactSearchResultV1:
        """Run deterministic fixed-shape selection for active compact roots."""

        started = time.perf_counter()
        legal_mask = _binary_bool_array(root_batch.legal_mask, "legal_mask")
        if legal_mask.ndim != 2:
            raise ReplayCompatibilityError("legal_mask must have shape [R,3]")
        if legal_mask.shape != (self.root_count, ACTION_COUNT):
            raise ReplayCompatibilityError(
                "fixed-shape owner expected legal_mask shape "
                f"({self.root_count}, {ACTION_COUNT}), got {legal_mask.shape}"
            )
        active_mask = _binary_bool_array(root_batch.active_root_mask, "active_root_mask")
        if active_mask.shape != (self.root_count,):
            raise ReplayCompatibilityError(
                "fixed-shape owner expected active_root_mask shape "
                f"({self.root_count},), got {active_mask.shape}"
            )

        active_indices = np.flatnonzero(active_mask).astype(np.int32, copy=False)
        active_count = int(active_indices.size)
        if active_count and not bool(legal_mask[active_indices].any(axis=1).all()):
            raise ReplayCompatibilityError(
                "fixed-shape owner got an active root with no legal action"
            )

        self._selected_action.fill(0)
        self._visit_policy.fill(0.0)
        self._raw_visit_counts.fill(0.0)
        self._root_value.fill(0.0)

        if active_count:
            active_legal = legal_mask[active_indices]
            selected_active = np.argmax(active_legal, axis=1).astype(np.int64, copy=False)
            self._selected_action[active_indices] = selected_active.astype(
                np.int16,
                copy=False,
            )
            self._visit_policy[active_indices, selected_active] = 1.0
            self._raw_visit_counts[active_indices, selected_active] = float(
                self.num_simulations
            )

        selected = self._selected_action[active_indices].astype(np.int16, copy=True)
        visit_policy = self._visit_policy[active_indices].astype(np.float32, copy=True)
        raw_visit_counts = self._raw_visit_counts[active_indices].astype(
            np.float32,
            copy=True,
        )
        root_value = self._root_value[active_indices].astype(np.float32, copy=True)
        total_sec = time.perf_counter() - started
        inactive_count = int(self.root_count - active_count)
        action_d2h_bytes = int(selected.nbytes)
        replay_payload_d2h_bytes = int(
            visit_policy.nbytes + raw_visit_counts.nbytes + root_value.nbytes
        )

        profile_telemetry = {
            "fixed_shape_batched_search_owner_profile_only": True,
            "fixed_shape_batched_search_owner_not_lightzero_ctree": True,
            "fixed_shape_batched_search_owner_trainer_ready": False,
            "fixed_shape_batched_search_owner_root_count": int(self.root_count),
            "fixed_shape_batched_search_owner_padded_root_count": int(self.root_count),
            "fixed_shape_batched_search_owner_active_root_count": int(active_count),
            "fixed_shape_batched_search_owner_inactive_root_count": int(inactive_count),
            "fixed_shape_batched_search_owner_masked_inactive_root_count": int(
                inactive_count
            ),
            "fixed_shape_batched_search_owner_action_count": int(ACTION_COUNT),
            "fixed_shape_batched_search_owner_requested_simulations": int(
                self.num_simulations
            ),
            "fixed_shape_batched_search_owner_actual_search_simulations": 0,
            "fixed_shape_batched_search_owner_first_legal_policy": True,
            "fixed_shape_batched_search_owner_buffer_reused": bool(self._run_count > 0),
            "fixed_shape_batched_search_owner_preallocated_buffer_bytes": int(
                self._selected_action.nbytes
                + self._visit_policy.nbytes
                + self._raw_visit_counts.nbytes
                + self._root_value.nbytes
            ),
            "fixed_shape_batched_search_owner_total_sec": float(total_sec),
            "fixed_shape_batched_search_owner_h2d_sec": 0.0,
            "fixed_shape_batched_search_owner_ctree_calls": 0,
            "fixed_shape_batched_search_owner_tolist_calls": 0,
            "fixed_shape_batched_search_owner_per_sim_d2h_bytes": 0,
            "fixed_shape_batched_search_owner_h2d_bytes": 0,
            "fixed_shape_batched_search_owner_d2h_bytes": int(
                action_d2h_bytes + replay_payload_d2h_bytes
            ),
            "fixed_shape_batched_search_owner_obs_h2d_bytes": 0,
            "fixed_shape_batched_search_owner_mask_h2d_bytes": 0,
            "fixed_shape_batched_search_owner_action_d2h_bytes": action_d2h_bytes,
            "fixed_shape_batched_search_owner_replay_payload_d2h_bytes": (
                replay_payload_d2h_bytes
            ),
            "fixed_shape_batched_search_owner_root_observation_copy_bytes": 0,
            "fixed_shape_batched_search_owner_python_rows_materialized": 0,
            "fixed_shape_batched_search_owner_rnd_materialized_rows": 0,
            "fixed_shape_batched_search_owner_python_root_objects": 0,
        }
        metadata = {
            "profile_backend": self.search_impl,
            "profile_only": True,
            "profile_semantics": FIXED_SHAPE_BATCHED_SEARCH_OWNER_SEMANTICS,
            "compact_search_service_adapter": True,
            "not_lightzero_ctree": True,
            "trainer_ready": False,
            "first_legal_policy": True,
            "ctree_calls": 0,
            "tolist_calls": 0,
            "per_sim_d2h_bytes": 0,
            "profile_telemetry": profile_telemetry,
        }

        self._run_count += 1
        self._last_profile_telemetry = dict(profile_telemetry)
        return validate_compact_search_result_v1(
            root_batch,
            selected_action=selected,
            visit_policy=visit_policy,
            root_value=root_value,
            raw_visit_counts=raw_visit_counts,
            search_impl=self.search_impl,
            num_simulations=self.num_simulations,
            metadata=metadata,
        )

    def run_action_step(
        self,
        root_batch: CompactRootBatchV1,
    ) -> CompactSearchActionStepV1:
        """Return only selected actions and keep replay arrays behind a handle."""

        result = self.run(root_batch)
        handle = f"{self.search_impl}:{self._replay_payload_counter}"
        self._replay_payload_counter += 1
        profile_telemetry = result.metadata.get("profile_telemetry", {})
        action_profile_telemetry = (
            dict(profile_telemetry) if isinstance(profile_telemetry, dict) else {}
        )
        action_d2h_bytes = int(
            action_profile_telemetry.get(
                "fixed_shape_batched_search_owner_action_d2h_bytes",
                result.selected_action.nbytes,
            )
        )
        replay_payload_d2h_bytes = int(
            action_profile_telemetry.get(
                "fixed_shape_batched_search_owner_replay_payload_d2h_bytes",
                result.visit_policy.nbytes
                + result.root_value.nbytes
                + (
                    0
                    if result.raw_visit_counts is None
                    else result.raw_visit_counts.nbytes
                ),
            )
        )
        action_profile_telemetry.update(
            {
                "fixed_shape_batched_search_owner_two_phase_action_only": True,
                "fixed_shape_batched_search_owner_d2h_bytes": action_d2h_bytes,
                "fixed_shape_batched_search_owner_replay_payload_d2h_bytes": 0,
                "fixed_shape_batched_search_owner_deferred_replay_payload_d2h_bytes": (
                    replay_payload_d2h_bytes
                ),
            }
        )
        metadata = {
            "two_phase_compact_search": True,
            "fixed_shape_batched_search_owner_two_phase": True,
            "profile_telemetry": action_profile_telemetry,
        }
        replay_metadata = dict(metadata)
        replay_metadata["profile_telemetry"] = (
            dict(profile_telemetry) if isinstance(profile_telemetry, dict) else {}
        )
        action_step = compact_search_action_step_v1_from_result(
            result,
            replay_payload_handle=handle,
            metadata=metadata,
        )
        action_step.metadata["search_replay_payload_digest"] = (
            compact_search_deferred_replay_payload_digest_v1(handle)
        )
        action_step.metadata["search_replay_payload_digest_deferred"] = True
        replay_payload = compact_search_replay_payload_v1_from_result(
            result,
            replay_payload_handle=handle,
            metadata=replay_metadata,
        )
        if handle in self._pending_replay_payloads:
            raise ReplayCompatibilityError("duplicate fixed-shape replay payload handle")
        self._pending_replay_payloads[handle] = replay_payload
        return action_step

    def flush_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactSearchReplayPayloadV1:
        """Return and remove a delayed replay payload for a prior action step."""

        handle = str(replay_payload_handle)
        if not handle:
            raise ReplayCompatibilityError("replay_payload_handle must be non-empty")
        payload = self._pending_replay_payloads.pop(handle, None)
        if payload is None:
            raise ReplayCompatibilityError("unknown or already-flushed replay payload handle")
        return payload

    def flush_device_replay_payload(
        self,
        replay_payload_handle: str,
    ) -> CompactDeviceSearchReplayPayloadV1:
        """Return the delayed replay payload as device tensors.

        This is still a profile-only fixed-shape floor: it does not add model
        inference or MCTS. The method exists so resident-observation compact
        trainer profiles can compare against the floor without using host replay
        fallback.
        """

        import torch

        payload = self.flush_replay_payload(replay_payload_handle)
        device = self._resolve_device(torch_module=torch)
        visit_policy = torch.as_tensor(
            payload.visit_policy,
            dtype=torch.float32,
            device=device,
        )
        root_value = torch.as_tensor(
            payload.root_value,
            dtype=torch.float32,
            device=device,
        )
        raw_visit_counts = torch.as_tensor(
            (
                np.zeros_like(payload.visit_policy, dtype=np.float32)
                if payload.raw_visit_counts is None
                else payload.raw_visit_counts
            ),
            dtype=torch.float32,
            device=device,
        )
        metadata = dict(payload.metadata)
        profile = dict(metadata.get("profile_telemetry", {}))
        profile.update(
            {
                "fixed_shape_batched_search_owner_device_replay_payload": True,
                "fixed_shape_batched_search_owner_device_replay_payload_device": str(
                    visit_policy.device
                ),
                "fixed_shape_batched_search_owner_replay_payload_d2h_bytes": 0,
                "fixed_shape_batched_search_owner_host_search_payload_fallback_allowed": (
                    False
                ),
            }
        )
        metadata.update(profile)
        metadata.update(
            {
                "schema_id": COMPACT_DEVICE_SEARCH_REPLAY_PAYLOAD_SCHEMA_ID,
                "phase": "replay_critical_device",
                "search_impl": self.search_impl,
                "num_simulations": int(self.num_simulations),
                "active_root_count": int(payload.root_index.size),
                "replay_payload_origin": f"{self.search_impl}:{replay_payload_handle}",
                "device_replay_payload": True,
                "device_replay_payload_device": str(visit_policy.device),
                "host_search_payload_fallback_allowed": False,
                "profile_telemetry": profile,
            }
        )
        return CompactDeviceSearchReplayPayloadV1(
            replay_payload_handle=str(replay_payload_handle),
            root_index=payload.root_index.astype(np.int32, copy=True),
            env_row=payload.env_row.astype(np.int32, copy=True),
            player=payload.player.astype(np.int16, copy=True),
            policy_env_id=payload.policy_env_id.astype(np.int64, copy=True),
            visit_policy=visit_policy,
            root_value=root_value,
            raw_visit_counts=raw_visit_counts,
            predicted_value=None,
            predicted_policy_logits=None,
            metadata=metadata,
        )

    def _resolve_device(self, *, torch_module: Any) -> Any:
        if self._device is not None:
            return self._device
        model = self._model
        if model is not None:
            try:
                return next(model.parameters()).device
            except Exception:
                pass
        return torch_module.device("cpu")


def _binary_bool_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ and not np.isin(array, (0, 1)).all():
        raise ReplayCompatibilityError(f"{name} must be binary 0/1 before bool coercion")
    return array.astype(np.bool_, copy=False)


__all__ = [
    "FIXED_SHAPE_BATCHED_SEARCH_OWNER_IMPL",
    "FIXED_SHAPE_BATCHED_SEARCH_OWNER_SEMANTICS",
    "FixedShapeBatchedSearchOwnerV0",
]
