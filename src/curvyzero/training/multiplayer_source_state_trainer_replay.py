"""Array replay for the source-state multiplayer trainer surface.

This is deliberately narrower than the public multiplayer metadata replay. It
stores the arrays emitted by ``SourceStateMultiplayerTrainerSurface`` and makes
no LightZero/GameSegment integration claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from curvyzero.training.multiplayer_source_state_trainer_surface import (
    MULTIPLAYER_TRAINER_SURFACE_SCHEMA_HASH,
    MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID,
)
from curvyzero.training.multiplayer_source_state_trainer_surface import (
    MultiplayerTrainerStepV0,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.replay_chunk_v0 import stable_contract_hash


SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID = (
    "curvyzero_source_state_multiplayer_trainer_replay_arrays/v0"
)
SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_METADATA_SCHEMA_ID = (
    "curvyzero_source_state_multiplayer_trainer_replay_metadata/v0"
)
SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_RECORD_SCHEMA_ID = (
    "curvyzero_source_state_multiplayer_trainer_replay_record/v0"
)
SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID = (
    "curvyzero_source_state_multiplayer_trainer_replay_schema/v0"
)
SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_KIND = (
    "source_state_multiplayer_trainer_replay_arrays"
)
SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CLAIM_ID = (
    "source_state_multiplayer_trainer_surface_arrays_only/v0"
)
POLICY_ROW_STORAGE = "per_record_variable_length_arrays/v0"
ACTION_COUNT = 3
FRAME_STACK_SHAPE = (4, 64, 64)

SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_NON_CLAIMS = (
    "not_metadata_only_replay",
    "not_lightzero_training_integration",
    "not_lightzero_native_game_segment",
    "not_native_game_segment",
    "not_search_targets",
    "not_value_targets",
    "not_policy_targets",
)
SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_ARRAY_KEYS = (
    "observation",
    "legal_action_mask",
    "lightzero_action_mask",
    "live_mask",
    "joint_action",
    "reward",
    "done",
    "terminated",
    "truncated",
    "final_observation",
    "final_observation_row_mask",
    "final_reward_map",
)
SOURCE_STATE_MULTIPLAYER_TRAINER_POLICY_ROW_ARRAY_KEYS = (
    "policy_observation",
    "policy_action_mask",
    "policy_env_row",
    "policy_player",
)
PROJECT_TRAINING_HELPER_METADATA_KEYS = (
    "death_mode",
    "death_suppression_for_profile",
    "death_suppression_claim",
    "disable_death_for_profile",
    "death_immunity_player_ids",
    "death_immunity_mask",
    "death_immunity_diagnostic",
    "death_immunity_claim",
    "opponent_death_mode",
    "opponent_death_mode_diagnostic",
    "opponent_death_mode_claim",
    "opponent_runtime_mode",
    "blank_canvas_noop",
    "source_fidelity_claim",
    "original_curvytron_behavior_claim",
)
SOURCE_STATE_MULTIPLAYER_TRAINER_AUDIT_METADATA_KEYS = (
    "action_sidecar",
    "alive",
    "borderless",
    "bonus_catch_count_step",
    "bonus_support",
    "bonus_support_mode",
    "bonus_support_mode_by_row",
    "death_cause",
    "death_cause_name",
    "death_count",
    "death_hit_owner",
    "death_hit_old",
    "death_player",
    "final_observation_policy",
    "final_reward_policy",
    "loser_ids",
    "present",
    "score",
    "step_counters",
    "terminal_reason",
    "terminal_reason_name",
    "winner",
    "winner_ids",
)
PROJECT_TRAINING_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM = (
    "restricted_by_project_training_helper"
)
SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA = {
    "schema_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID,
    "metadata_schema_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_METADATA_SCHEMA_ID,
    "record_schema_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_RECORD_SCHEMA_ID,
    "contract_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID,
    "surface_schema_id": MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID,
    "surface_schema_hash": MULTIPLAYER_TRAINER_SURFACE_SCHEMA_HASH,
    "arrays": {
        "observation": ("float32", ("time", "batch", "player", *FRAME_STACK_SHAPE)),
        "legal_action_mask": ("bool", ("time", "batch", "player", ACTION_COUNT)),
        "lightzero_action_mask": ("bool", ("time", "batch", "player", ACTION_COUNT)),
        "live_mask": ("bool", ("time", "batch", "player")),
        "joint_action": ("int16", ("time", "batch", "player")),
        "reward": ("float32", ("time", "batch", "player")),
        "done": ("bool", ("time", "batch")),
        "terminated": ("bool", ("time", "batch")),
        "truncated": ("bool", ("time", "batch")),
        "final_observation": (
            "float32",
            ("time", "batch", "player", *FRAME_STACK_SHAPE),
        ),
        "final_observation_row_mask": ("bool", ("time", "batch")),
        "final_reward_map": ("float32", ("time", "batch", "player")),
    },
    "policy_row_storage": POLICY_ROW_STORAGE,
    "scope": "trainer-surface arrays only; no LightZero/native target integration",
    "non_claims": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_NON_CLAIMS,
}
SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_HASH = stable_contract_hash(
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA
)


@dataclass(frozen=True, slots=True)
class SourceStateMultiplayerTrainerReplayChunkV0:
    """A small in-memory array replay chunk for trainer-surface batches."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    policy_rows: tuple[dict[str, np.ndarray], ...]
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class _SourceStateMultiplayerTrainerReplayRecord:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    policy_rows: dict[str, np.ndarray]


@dataclass(slots=True)
class SourceStateMultiplayerTrainerReplayRecorder:
    """Collect ``MultiplayerTrainerStepV0`` batches into one array replay chunk."""

    _records: list[_SourceStateMultiplayerTrainerReplayRecord] = field(
        default_factory=list
    )
    _batch_size: int | None = None
    _player_count: int | None = None
    _action_count: int | None = None
    _closed_by_terminal: bool = False

    @property
    def record_count(self) -> int:
        """Number of trainer-surface batches recorded."""

        return len(self._records)

    @property
    def batch_size(self) -> int | None:
        """Batch size once the first record has been accepted."""

        return self._batch_size

    @property
    def player_count(self) -> int | None:
        """Player count once the first record has been accepted."""

        return self._player_count

    @property
    def closed_by_terminal(self) -> bool:
        """Whether a terminal or final-observation row closed this recorder."""

        return self._closed_by_terminal

    def reset(self) -> None:
        """Clear recorded batches so this recorder can be reused."""

        self._records.clear()
        self._batch_size = None
        self._player_count = None
        self._action_count = None
        self._closed_by_terminal = False

    def record(
        self,
        step: MultiplayerTrainerStepV0,
        *,
        rng_history_ref: Any = None,
        source_ref: Any = None,
    ) -> dict[str, Any]:
        """Append one trainer-surface batch, copying every trainer-critical array."""

        if self._closed_by_terminal:
            raise ReplayCompatibilityError(
                "terminal source-state trainer replay batch must be the final "
                "recorded batch before reset or a new recorder"
            )

        arrays, policy_rows, batch_size, player_count, action_count = (
            _copy_trainer_step_arrays(step)
        )
        if self._batch_size is None:
            self._batch_size = batch_size
            self._player_count = player_count
            self._action_count = action_count
        elif batch_size != self._batch_size:
            raise ReplayCompatibilityError(
                f"B row mismatch: expected {self._batch_size}, got {batch_size}"
            )
        elif player_count != self._player_count:
            raise ReplayCompatibilityError(
                f"player_count mismatch: expected {self._player_count}, got {player_count}"
            )
        elif action_count != self._action_count:
            raise ReplayCompatibilityError(
                f"action_count mismatch: expected {self._action_count}, got {action_count}"
            )

        metadata = _record_metadata(
            step=step,
            arrays=arrays,
            policy_rows=policy_rows,
            sequence_index=len(self._records),
            rng_history_ref=rng_history_ref,
            source_ref=source_ref,
        )
        self._records.append(
            _SourceStateMultiplayerTrainerReplayRecord(
                metadata=metadata,
                arrays=arrays,
                policy_rows=policy_rows,
            )
        )
        self._closed_by_terminal = _has_terminal_or_final_rows(arrays)
        return dict(metadata)

    def build_chunk(self) -> SourceStateMultiplayerTrainerReplayChunkV0:
        """Stack recorded arrays over time and return a replay chunk."""

        if not self._records:
            raise ReplayCompatibilityError("at least one trainer replay record is required")

        arrays = {
            key: np.stack([record.arrays[key] for record in self._records], axis=0)
            for key in SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_ARRAY_KEYS
        }
        policy_rows = tuple(
            {key: value.copy() for key, value in record.policy_rows.items()}
            for record in self._records
        )
        records = tuple(dict(record.metadata) for record in self._records)
        metadata = _chunk_metadata(
            arrays=arrays,
            records=records,
            closed_by_terminal=self._closed_by_terminal,
            batch_size=self._batch_size,
            player_count=self._player_count,
            action_count=self._action_count,
        )
        return SourceStateMultiplayerTrainerReplayChunkV0(
            metadata=metadata,
            arrays=arrays,
            policy_rows=policy_rows,
            records=records,
        )


def _copy_trainer_step_arrays(
    step: MultiplayerTrainerStepV0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int, int, int]:
    observation = _array(step.observation, "observation", np.float32)
    if observation.ndim != 5 or observation.shape[2:] != FRAME_STACK_SHAPE:
        raise ReplayCompatibilityError(
            "observation must have shape [B,P,4,64,64]"
        )
    batch_size = int(observation.shape[0])
    player_count = int(observation.shape[1])

    legal_action_mask = _array(step.legal_action_mask, "legal_action_mask", bool)
    _expect_shape(
        legal_action_mask,
        "legal_action_mask",
        (batch_size, player_count, ACTION_COUNT),
    )
    action_count = int(legal_action_mask.shape[2])
    lightzero_action_mask = _array(
        step.lightzero_action_mask,
        "lightzero_action_mask",
        bool,
    )
    _expect_shape(
        lightzero_action_mask,
        "lightzero_action_mask",
        (batch_size, player_count, action_count),
    )
    live_mask = _array(step.live_mask, "live_mask", bool)
    _expect_shape(live_mask, "live_mask", (batch_size, player_count))
    joint_action = _array(step.joint_action, "joint_action", np.int16)
    _expect_shape(joint_action, "joint_action", (batch_size, player_count))
    reward = _array(step.reward, "reward", np.float32)
    _expect_shape(reward, "reward", (batch_size, player_count))
    done = _array(step.done, "done", bool)
    _expect_shape(done, "done", (batch_size,))
    terminated = _array(step.terminated, "terminated", bool)
    _expect_shape(terminated, "terminated", (batch_size,))
    truncated = _array(step.truncated, "truncated", bool)
    _expect_shape(truncated, "truncated", (batch_size,))
    if not np.array_equal(done, terminated | truncated):
        raise ReplayCompatibilityError("done must equal terminated | truncated")

    final_observation = _array(step.final_observation, "final_observation", np.float32)
    _expect_shape(
        final_observation,
        "final_observation",
        (batch_size, player_count, *FRAME_STACK_SHAPE),
    )
    final_observation_row_mask = _array(
        step.final_observation_row_mask,
        "final_observation_row_mask",
        bool,
    )
    _expect_shape(
        final_observation_row_mask,
        "final_observation_row_mask",
        (batch_size,),
    )
    final_reward_map = _array(step.final_reward_map, "final_reward_map", np.float32)
    _expect_shape(final_reward_map, "final_reward_map", (batch_size, player_count))

    policy_observation = _array(
        step.policy_observation,
        "policy_observation",
        np.float32,
    )
    if policy_observation.ndim != 4 or policy_observation.shape[1:] != FRAME_STACK_SHAPE:
        raise ReplayCompatibilityError(
            "policy_observation must have shape [R,4,64,64]"
        )
    policy_row_count = int(policy_observation.shape[0])
    policy_action_mask = _array(step.policy_action_mask, "policy_action_mask", bool)
    _expect_shape(policy_action_mask, "policy_action_mask", (policy_row_count, action_count))
    policy_env_row = _array(step.policy_env_row, "policy_env_row", np.int32)
    _expect_shape(policy_env_row, "policy_env_row", (policy_row_count,))
    policy_player = _array(step.policy_player, "policy_player", np.int16)
    _expect_shape(policy_player, "policy_player", (policy_row_count,))
    if policy_row_count:
        if int(policy_env_row.min()) < 0 or int(policy_env_row.max()) >= batch_size:
            raise ReplayCompatibilityError("policy_env_row contains out-of-range rows")
        if int(policy_player.min()) < 0 or int(policy_player.max()) >= player_count:
            raise ReplayCompatibilityError("policy_player contains out-of-range players")
    expected_env_row, expected_player = np.nonzero(live_mask)
    expected_env_row = expected_env_row.astype(np.int32, copy=False)
    expected_player = expected_player.astype(np.int16, copy=False)
    if not np.array_equal(policy_env_row, expected_env_row):
        raise ReplayCompatibilityError("policy_env_row must match live_mask order")
    if not np.array_equal(policy_player, expected_player):
        raise ReplayCompatibilityError("policy_player must match live_mask order")
    if policy_row_count:
        expected_observation = observation[policy_env_row, policy_player]
        if not np.array_equal(policy_observation, expected_observation):
            raise ReplayCompatibilityError(
                "policy_observation must match observation[policy_env_row, policy_player]"
            )
        expected_action_mask = legal_action_mask[policy_env_row, policy_player]
        if not np.array_equal(policy_action_mask, expected_action_mask):
            raise ReplayCompatibilityError(
                "policy_action_mask must match legal_action_mask policy rows"
            )

    arrays = {
        "observation": observation,
        "legal_action_mask": legal_action_mask,
        "lightzero_action_mask": lightzero_action_mask,
        "live_mask": live_mask,
        "joint_action": joint_action,
        "reward": reward,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "final_observation": final_observation,
        "final_observation_row_mask": final_observation_row_mask,
        "final_reward_map": final_reward_map,
    }
    policy_rows = {
        "policy_observation": policy_observation,
        "policy_action_mask": policy_action_mask,
        "policy_env_row": policy_env_row,
        "policy_player": policy_player,
    }
    return arrays, policy_rows, batch_size, player_count, action_count


def _array(value: Any, name: str, dtype: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ReplayCompatibilityError(f"{name} must be convertible to {dtype}") from exc
    return array.copy()


def _expect_shape(array: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ReplayCompatibilityError(
            f"{name} must have shape {shape}, got {array.shape}"
        )


def _record_metadata(
    *,
    step: MultiplayerTrainerStepV0,
    arrays: Mapping[str, np.ndarray],
    policy_rows: Mapping[str, np.ndarray],
    sequence_index: int,
    rng_history_ref: Any,
    source_ref: Any,
) -> dict[str, Any]:
    info = step.info if isinstance(step.info, Mapping) else {}
    final_row_mask = arrays["final_observation_row_mask"]
    done = arrays["done"]
    metadata: dict[str, Any] = {
        "record_schema_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_RECORD_SCHEMA_ID,
        "sequence_index": int(sequence_index),
        "surface_schema_id": _metadata_value(
            info,
            "trainer_surface_schema_id",
            MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID,
        ),
        "surface_schema_hash": _metadata_value(
            info,
            "trainer_surface_schema_hash",
            MULTIPLAYER_TRAINER_SURFACE_SCHEMA_HASH,
        ),
        "trainer_surface_api": _metadata_value(info, "trainer_surface_api", None),
        "observation_schema_id": _metadata_value(
            info,
            "trainer_observation_schema_id",
            _metadata_value(info, "observation_schema_id", None),
        ),
        "single_frame_schema_id": _metadata_value(info, "single_frame_schema_id", None),
        "reward_schema_id": _metadata_value(info, "reward_schema_id", None),
        "reward_schema_hash": _metadata_value(info, "reward_schema_hash", None),
        "reward_variant": _metadata_value(info, "reward_variant", None),
        "joint_action_schema_id": _metadata_value(info, "joint_action_schema_id", None),
        "render_mode": _metadata_value(info, "trail_render_mode", None),
        "trail_render_mode": _metadata_value(info, "trail_render_mode", None),
        "bonus_render_mode": _metadata_value(info, "bonus_render_mode", None),
        "policy_row_count": int(policy_rows["policy_env_row"].size),
        "policy_env_row": policy_rows["policy_env_row"].astype(np.int32).tolist(),
        "policy_player": policy_rows["policy_player"].astype(np.int16).tolist(),
        "done_rows": np.flatnonzero(done).astype(np.int32).tolist(),
        "final_observation_rows": np.flatnonzero(final_row_mask)
        .astype(np.int32)
        .tolist(),
        "terminal_or_final": _has_terminal_or_final_rows(arrays),
        "metadata_only": False,
        "trainer_replay_claim": True,
        "trainer_replay_claim_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CLAIM_ID,
        "lightzero_training_claim": False,
        "native_game_segment_claim": False,
    }
    _copy_source_state_audit_metadata(metadata, info)
    _copy_project_training_helper_metadata(metadata, info)
    _label_restricted_project_training_helper_metadata(metadata)
    if rng_history_ref is not None:
        metadata["rng_history_ref"] = _compact_ref(rng_history_ref)
    if source_ref is not None:
        metadata["source_ref"] = _compact_ref(source_ref)
    return metadata


def _chunk_metadata(
    *,
    arrays: Mapping[str, np.ndarray],
    records: Sequence[Mapping[str, Any]],
    closed_by_terminal: bool,
    batch_size: int | None,
    player_count: int | None,
    action_count: int | None,
) -> dict[str, Any]:
    first_record = records[0]
    metadata = {
        "metadata_schema_id": (
            SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_METADATA_SCHEMA_ID
        ),
        "schema_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID,
        "schema_hash": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_HASH,
        "replay_contract_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID,
        "replay_schema_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID,
        "replay_schema_hash": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_HASH,
        "replay_kind": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_KIND,
        "record_schema_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_RECORD_SCHEMA_ID,
        "surface_schema_id": first_record.get(
            "surface_schema_id",
            MULTIPLAYER_TRAINER_SURFACE_SCHEMA_ID,
        ),
        "surface_schema_hash": first_record.get(
            "surface_schema_hash",
            MULTIPLAYER_TRAINER_SURFACE_SCHEMA_HASH,
        ),
        "observation_schema_id": first_record.get("observation_schema_id"),
        "single_frame_schema_id": first_record.get("single_frame_schema_id"),
        "reward_schema_id": first_record.get("reward_schema_id"),
        "reward_schema_hash": first_record.get("reward_schema_hash"),
        "joint_action_schema_id": first_record.get("joint_action_schema_id"),
        "render_mode": first_record.get("render_mode"),
        "trail_render_mode": first_record.get("trail_render_mode"),
        "bonus_render_mode": first_record.get("bonus_render_mode"),
        "record_count": len(records),
        "chunk_steps": len(records),
        "closed_by_terminal": bool(closed_by_terminal),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "player_count": int(player_count) if player_count is not None else None,
        "action_count": int(action_count) if action_count is not None else None,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "policy_row_storage": POLICY_ROW_STORAGE,
        "policy_row_record_count": len(records),
        "record_audit_metadata_keys": SOURCE_STATE_MULTIPLAYER_TRAINER_AUDIT_METADATA_KEYS,
        "metadata_only": False,
        "metadata_only_replay_claim": False,
        "multiplayer_metadata_replay_claim": False,
        "trainer_replay_claim": True,
        "trainer_replay_claim_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CLAIM_ID,
        "lightzero_training_claim": False,
        "lightzero_native_game_segment_claim": False,
        "native_game_segment_claim": False,
        "native_game_segment_integration_claim": False,
        "game_segment_integration_claim": False,
        "lightzero_training_integration": "not_claimed",
        "native_game_segment_integration": "not_claimed",
        "non_claims": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_NON_CLAIMS,
    }
    _copy_project_training_helper_metadata(metadata, first_record)
    _label_restricted_project_training_helper_metadata(metadata)
    return metadata


def _copy_source_state_audit_metadata(
    metadata: dict[str, Any],
    source: Mapping[str, Any],
) -> None:
    for key in SOURCE_STATE_MULTIPLAYER_TRAINER_AUDIT_METADATA_KEYS:
        if key in source:
            metadata[key] = _metadata_value(source, key, None)


def _copy_project_training_helper_metadata(
    metadata: dict[str, Any],
    source: Mapping[str, Any],
) -> None:
    for key in PROJECT_TRAINING_HELPER_METADATA_KEYS:
        if key in source:
            metadata[key] = _metadata_value(source, key, None)


def _label_restricted_project_training_helper_metadata(
    metadata: dict[str, Any],
) -> None:
    if not _project_training_helper_restricts_source_rules(metadata):
        return
    metadata["original_curvytron_behavior_claim"] = False
    metadata["source_fidelity_claim"] = (
        PROJECT_TRAINING_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM
    )


def _project_training_helper_restricts_source_rules(
    metadata: Mapping[str, Any],
) -> bool:
    return (
        metadata.get("death_suppression_for_profile") is True
        or metadata.get("disable_death_for_profile") is True
        or metadata.get("death_immunity_diagnostic") is True
        or metadata.get("opponent_death_mode_diagnostic") is True
        or metadata.get("opponent_runtime_mode") == "blank_canvas_noop"
        or metadata.get("blank_canvas_noop") is True
    )


def _metadata_value(info: Mapping[str, Any], key: str, default: Any) -> Any:
    value = info.get(key, default)
    return _compact_ref(value)


def _compact_ref(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _compact_ref(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_compact_ref(item) for item in value]
    return str(value)


def _has_terminal_or_final_rows(arrays: Mapping[str, np.ndarray]) -> bool:
    return bool(
        np.asarray(arrays["done"], dtype=bool).any()
        or np.asarray(arrays["final_observation_row_mask"], dtype=bool).any()
    )


__all__ = [
    "POLICY_ROW_STORAGE",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_AUDIT_METADATA_KEYS",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_POLICY_ROW_ARRAY_KEYS",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_ARRAY_KEYS",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CLAIM_ID",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_KIND",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_METADATA_SCHEMA_ID",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_NON_CLAIMS",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_RECORD_SCHEMA_ID",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_HASH",
    "SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID",
    "SourceStateMultiplayerTrainerReplayChunkV0",
    "SourceStateMultiplayerTrainerReplayRecorder",
]
