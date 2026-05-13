"""Target rows for source-state multiplayer trainer replay.

This module is the small checked bridge after
``SourceStateMultiplayerTrainerReplayRecorder``. It builds repo-owned target
rows and deliberately does not claim native LightZero ``GameSegment`` support.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from curvyzero.training.multiplayer_source_state_trainer_replay import (
    FRAME_STACK_SHAPE,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_HASH,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID,
)
from curvyzero.training.multiplayer_source_state_trainer_replay import (
    SourceStateMultiplayerTrainerReplayChunkV0,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.replay_chunk_v0 import stable_contract_hash


ACTION_COUNT = 3
DEFAULT_TO_PLAY = -1
SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID = (
    "curvyzero_source_state_multiplayer_muzero_target_rows/v0"
)
SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID = (
    "curvyzero_source_state_multiplayer_target_rows_schema/v0"
)
SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_METADATA_SCHEMA_ID = (
    "curvyzero_source_state_multiplayer_target_rows_metadata/v0"
)
SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_CONTRACT_ID = (
    "curvyzero_source_state_multiplayer_sample_batch/v0"
)
SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_SCHEMA_ID = (
    "curvyzero_source_state_multiplayer_sample_batch_schema/v0"
)
SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_METADATA_SCHEMA_ID = (
    "curvyzero_source_state_multiplayer_sample_batch_metadata/v0"
)
POLICY_ROW_RECORD_SCHEMA_ID = "curvyzero_source_state_policy_row_record/v0"
TARGET_ROW_ALIGNMENT = "record_k_decision_record_k_plus_1_result/v0"
TARGET_ROW_TO_PLAY_POLICY = "always_-1_non_board_game_seat_perspective/v0"
PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM = (
    "restricted_by_project_training_helper"
)

PROJECT_HELPER_METADATA_KEYS = (
    "death_mode",
    "disable_death_for_profile",
    "death_suppression_for_profile",
    "death_suppression_claim",
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
    "profile_mode_enabled",
    "profile_mode_claim",
    "training_only_mode_enabled",
    "training_only_mode_claim",
)

SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_NON_CLAIMS = (
    "not_lightzero_training_integration",
    "not_lightzero_native_game_segment",
    "not_native_game_segment",
    "not_muzero_game_buffer",
    "not_learner_update",
    "not_policy_improvement_claim",
)

SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA = {
    "schema_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
    "metadata_schema_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_METADATA_SCHEMA_ID,
    "contract_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
    "source_replay_contract_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID,
    "source_replay_schema_id": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID,
    "source_replay_schema_hash": SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_HASH,
    "policy_row_record_schema_id": POLICY_ROW_RECORD_SCHEMA_ID,
    "alignment": TARGET_ROW_ALIGNMENT,
    "to_play_policy": TARGET_ROW_TO_PLAY_POLICY,
    "fields": {
        "observation": ("float32", ("target_row", *FRAME_STACK_SHAPE)),
        "action": ("int16", ("target_row",)),
        "action_mask": ("bool", ("target_row", ACTION_COUNT)),
        "policy_target": ("float32", ("target_row", ACTION_COUNT)),
        "root_value": ("float32", ("target_row",)),
        "reward": ("float32", ("target_row",)),
        "final_reward": ("float32", ("target_row",)),
        "done": ("bool", ("target_row",)),
        "terminated": ("bool", ("target_row",)),
        "truncated": ("bool", ("target_row",)),
        "next_observation": ("float32", ("target_row", *FRAME_STACK_SHAPE)),
        "to_play": ("int64", ("target_row",)),
        "env_row": ("int32", ("target_row",)),
        "player": ("int16", ("target_row",)),
        "record_index": ("int32", ("target_row",)),
        "next_record_index": ("int32", ("target_row",)),
        "policy_row": ("int32", ("target_row",)),
    },
    "non_claims": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_NON_CLAIMS,
}
SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH = stable_contract_hash(
    SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA
)


@dataclass(frozen=True, slots=True)
class PolicyRowRecordV0:
    """Policy/search output for one live source-state policy row."""

    record_index: int
    policy_row: int
    env_row: int
    player: int
    action: int
    action_mask: np.ndarray
    policy_target: np.ndarray
    root_value: float
    policy_source: str
    source_record_ref: Any = None


@dataclass(frozen=True, slots=True)
class SourceStateMultiplayerTargetRowsV0:
    """Validated one-step target rows from source-state multiplayer replay."""

    metadata: dict[str, Any]
    observation: np.ndarray
    action: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    root_value: np.ndarray
    reward: np.ndarray
    final_reward: np.ndarray
    done: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_observation: np.ndarray
    to_play: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    record_index: np.ndarray
    next_record_index: np.ndarray
    policy_row: np.ndarray
    policy_source: tuple[str, ...]
    source_record_ref: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SourceStateMultiplayerSampleBatchV0:
    """Deterministic repo-owned sample batch from target rows."""

    metadata: dict[str, Any]
    row_id: np.ndarray
    observation: np.ndarray
    action: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    root_value: np.ndarray
    reward: np.ndarray
    final_reward: np.ndarray
    done: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_observation: np.ndarray
    to_play: np.ndarray
    env_row: np.ndarray
    player: np.ndarray
    record_index: np.ndarray
    next_record_index: np.ndarray
    policy_row: np.ndarray


def build_source_state_multiplayer_target_rows_v0(
    chunk: SourceStateMultiplayerTrainerReplayChunkV0,
    policy_row_records: Sequence[PolicyRowRecordV0 | Mapping[str, Any]],
) -> SourceStateMultiplayerTargetRowsV0:
    """Build checked per-seat transition target rows.

    Replay record ``k`` is the decision state. Replay record ``k + 1`` is the
    result of executing the player-major action in that next record.
    """

    arrays = _validate_chunk(chunk)
    target_records = [_policy_record(record) for record in policy_row_records]
    _reject_duplicate_policy_records(target_records)

    rows: list[dict[str, Any]] = []
    for record in target_records:
        rows.append(_target_row_from_record(arrays, chunk.policy_rows, record))

    return _target_rows_from_dicts(
        rows,
        metadata=_metadata(chunk=chunk, target_row_count=len(rows)),
    )


def build_source_state_multiplayer_sample_batch_v0(
    target_rows: SourceStateMultiplayerTargetRowsV0,
    *,
    batch_size: int,
    seed: int = 0,
    replace: bool = False,
) -> SourceStateMultiplayerSampleBatchV0:
    """Sample target rows with NumPy ``default_rng(seed)``.

    No-replacement full batches use the same seeded choice path, so ``row_id``
    records a deterministic permutation rather than preserving source order.
    """

    try:
        batch_size_int = int(batch_size)
    except (TypeError, ValueError) as exc:
        raise ReplayCompatibilityError("batch_size must be an integer") from exc
    if batch_size_int <= 0:
        raise ReplayCompatibilityError("batch_size must be positive")

    try:
        seed_int = int(seed)
    except (TypeError, ValueError) as exc:
        raise ReplayCompatibilityError("seed must be an integer") from exc

    target_row_count = int(np.asarray(target_rows.action).shape[0])
    if target_row_count <= 0:
        raise ReplayCompatibilityError("sample batch requires at least one target row")
    replace_bool = bool(replace)
    if batch_size_int > target_row_count and not replace_bool:
        raise ReplayCompatibilityError(
            "batch_size cannot exceed target row count without replacement"
        )

    rng = np.random.default_rng(seed_int)
    row_id = rng.choice(
        target_row_count,
        size=batch_size_int,
        replace=replace_bool,
    ).astype(np.int64, copy=True)

    return SourceStateMultiplayerSampleBatchV0(
        metadata=_sample_metadata(
            target_rows=target_rows,
            sample_row_count=batch_size_int,
            seed=seed_int,
            replace=replace_bool,
        ),
        row_id=row_id,
        observation=np.asarray(target_rows.observation[row_id], dtype=np.float32).copy(),
        action=np.asarray(target_rows.action[row_id], dtype=np.int16).copy(),
        action_mask=np.asarray(target_rows.action_mask[row_id], dtype=bool).copy(),
        policy_target=np.asarray(
            target_rows.policy_target[row_id],
            dtype=np.float32,
        ).copy(),
        root_value=np.asarray(target_rows.root_value[row_id], dtype=np.float32).copy(),
        reward=np.asarray(target_rows.reward[row_id], dtype=np.float32).copy(),
        final_reward=np.asarray(target_rows.final_reward[row_id], dtype=np.float32).copy(),
        done=np.asarray(target_rows.done[row_id], dtype=bool).copy(),
        terminated=np.asarray(target_rows.terminated[row_id], dtype=bool).copy(),
        truncated=np.asarray(target_rows.truncated[row_id], dtype=bool).copy(),
        next_observation=np.asarray(
            target_rows.next_observation[row_id],
            dtype=np.float32,
        ).copy(),
        to_play=np.asarray(target_rows.to_play[row_id], dtype=np.int64).copy(),
        env_row=np.asarray(target_rows.env_row[row_id], dtype=np.int32).copy(),
        player=np.asarray(target_rows.player[row_id], dtype=np.int16).copy(),
        record_index=np.asarray(target_rows.record_index[row_id], dtype=np.int32).copy(),
        next_record_index=np.asarray(
            target_rows.next_record_index[row_id],
            dtype=np.int32,
        ).copy(),
        policy_row=np.asarray(target_rows.policy_row[row_id], dtype=np.int32).copy(),
    )


def _validate_chunk(
    chunk: SourceStateMultiplayerTrainerReplayChunkV0,
) -> dict[str, np.ndarray]:
    arrays = {key: np.asarray(value) for key, value in chunk.arrays.items()}
    required = (
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
    missing = sorted(key for key in required if key not in arrays)
    if missing:
        raise ReplayCompatibilityError(
            "target-row chunk is missing arrays: " + ", ".join(missing)
        )

    observation = _array(arrays["observation"], "observation", np.float32)
    if observation.ndim != 6 or observation.shape[3:] != FRAME_STACK_SHAPE:
        raise ReplayCompatibilityError(
            "observation must have shape [T,B,P,4,64,64]"
        )
    time_steps, batch_size, player_count = observation.shape[:3]
    if time_steps < 2:
        raise ReplayCompatibilityError("target rows require at least two records")

    _expect_shape(
        _array(arrays["legal_action_mask"], "legal_action_mask", bool),
        "legal_action_mask",
        (time_steps, batch_size, player_count, ACTION_COUNT),
    )
    _expect_shape(
        _array(arrays["lightzero_action_mask"], "lightzero_action_mask", bool),
        "lightzero_action_mask",
        (time_steps, batch_size, player_count, ACTION_COUNT),
    )
    _expect_shape(
        _array(arrays["live_mask"], "live_mask", bool),
        "live_mask",
        (time_steps, batch_size, player_count),
    )
    _expect_shape(
        _array(arrays["joint_action"], "joint_action", np.int16),
        "joint_action",
        (time_steps, batch_size, player_count),
    )
    _expect_shape(
        _array(arrays["reward"], "reward", np.float32),
        "reward",
        (time_steps, batch_size, player_count),
    )
    done = _array(arrays["done"], "done", bool)
    terminated = _array(arrays["terminated"], "terminated", bool)
    truncated = _array(arrays["truncated"], "truncated", bool)
    _expect_shape(done, "done", (time_steps, batch_size))
    _expect_shape(terminated, "terminated", (time_steps, batch_size))
    _expect_shape(truncated, "truncated", (time_steps, batch_size))
    if not np.array_equal(done, terminated | truncated):
        raise ReplayCompatibilityError("done must equal terminated | truncated")
    _expect_shape(
        _array(arrays["final_observation"], "final_observation", np.float32),
        "final_observation",
        (time_steps, batch_size, player_count, *FRAME_STACK_SHAPE),
    )
    _expect_shape(
        _array(arrays["final_observation_row_mask"], "final_observation_row_mask", bool),
        "final_observation_row_mask",
        (time_steps, batch_size),
    )
    _expect_shape(
        _array(arrays["final_reward_map"], "final_reward_map", np.float32),
        "final_reward_map",
        (time_steps, batch_size, player_count),
    )
    if len(chunk.policy_rows) != time_steps:
        raise ReplayCompatibilityError("policy_rows length must match replay time")

    return {
        "observation": _array(arrays["observation"], "observation", np.float32),
        "legal_action_mask": _array(
            arrays["legal_action_mask"], "legal_action_mask", bool
        ),
        "lightzero_action_mask": _array(
            arrays["lightzero_action_mask"], "lightzero_action_mask", bool
        ),
        "live_mask": _array(arrays["live_mask"], "live_mask", bool),
        "joint_action": _array(arrays["joint_action"], "joint_action", np.int16),
        "reward": _array(arrays["reward"], "reward", np.float32),
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "final_observation": _array(
            arrays["final_observation"], "final_observation", np.float32
        ),
        "final_observation_row_mask": _array(
            arrays["final_observation_row_mask"],
            "final_observation_row_mask",
            bool,
        ),
        "final_reward_map": _array(
            arrays["final_reward_map"], "final_reward_map", np.float32
        ),
    }


def _target_row_from_record(
    arrays: Mapping[str, np.ndarray],
    policy_rows: Sequence[Mapping[str, np.ndarray]],
    record: PolicyRowRecordV0,
) -> dict[str, Any]:
    observation = arrays["observation"]
    time_steps, batch_size, player_count = observation.shape[:3]
    record_index = int(record.record_index)
    next_record_index = record_index + 1
    env_row = int(record.env_row)
    player = int(record.player)
    policy_row = int(record.policy_row)
    action = int(record.action)

    if record_index < 0 or next_record_index >= time_steps:
        raise ReplayCompatibilityError(
            "policy row record_index must have a following result record"
        )
    if env_row < 0 or env_row >= batch_size:
        raise ReplayCompatibilityError("policy row env_row is out of range")
    if player < 0 or player >= player_count:
        raise ReplayCompatibilityError("policy row player is out of range")
    if not bool(arrays["live_mask"][record_index, env_row, player]):
        raise ReplayCompatibilityError("policy row must point to a live decision seat")

    policy = policy_rows[record_index]
    expected_policy_count = _validate_policy_row_arrays(
        policy,
        batch_size=batch_size,
        player_count=player_count,
    )
    if policy_row < 0 or policy_row >= expected_policy_count:
        raise ReplayCompatibilityError("policy_row is out of range")
    if int(policy["policy_env_row"][policy_row]) != env_row:
        raise ReplayCompatibilityError("policy row env_row does not match replay map")
    if int(policy["policy_player"][policy_row]) != player:
        raise ReplayCompatibilityError("policy row player does not match replay map")

    action_mask = _array(record.action_mask, "action_mask", bool)
    _expect_shape(action_mask, "action_mask", (ACTION_COUNT,))
    expected_action_mask = np.asarray(policy["policy_action_mask"][policy_row], dtype=bool)
    if not np.array_equal(action_mask, expected_action_mask):
        raise ReplayCompatibilityError("policy row action_mask does not match replay")
    if not np.array_equal(
        action_mask,
        arrays["legal_action_mask"][record_index, env_row, player],
    ):
        raise ReplayCompatibilityError("policy row action_mask does not match legal mask")

    policy_target = _array(record.policy_target, "policy_target", np.float32)
    _expect_shape(policy_target, "policy_target", (ACTION_COUNT,))
    if not np.isfinite(policy_target).all():
        raise ReplayCompatibilityError("policy_target must be finite")
    if bool((policy_target < 0.0).any()):
        raise ReplayCompatibilityError("policy_target must be nonnegative")
    if not np.isclose(float(policy_target.sum()), 1.0, atol=1e-6):
        raise ReplayCompatibilityError("policy_target must sum to 1")
    if bool((policy_target[~action_mask] > 1e-7).any()):
        raise ReplayCompatibilityError("policy_target assigns mass to illegal actions")
    if action < 0 or action >= ACTION_COUNT:
        raise ReplayCompatibilityError("policy row action is out of range")
    if not bool(action_mask[action]):
        raise ReplayCompatibilityError("policy row action is illegal")
    if int(arrays["joint_action"][next_record_index, env_row, player]) != action:
        raise ReplayCompatibilityError(
            "policy row action does not match next replay joint_action"
        )
    if not np.isfinite(float(record.root_value)):
        raise ReplayCompatibilityError("root_value must be finite")
    if not str(record.policy_source):
        raise ReplayCompatibilityError("policy_source must be a non-empty string")

    final_row = bool(arrays["final_observation_row_mask"][next_record_index, env_row])
    if final_row:
        next_observation = arrays["final_observation"][
            next_record_index,
            env_row,
            player,
        ]
        final_reward = arrays["final_reward_map"][next_record_index, env_row, player]
    else:
        next_observation = observation[next_record_index, env_row, player]
        final_reward = arrays["reward"][next_record_index, env_row, player]

    return {
        "observation": observation[record_index, env_row, player].copy(),
        "action": np.int16(action),
        "action_mask": action_mask.copy(),
        "policy_target": policy_target.copy(),
        "root_value": np.float32(record.root_value),
        "reward": np.float32(arrays["reward"][next_record_index, env_row, player]),
        "final_reward": np.float32(final_reward),
        "done": bool(arrays["done"][next_record_index, env_row]),
        "terminated": bool(arrays["terminated"][next_record_index, env_row]),
        "truncated": bool(arrays["truncated"][next_record_index, env_row]),
        "next_observation": np.asarray(next_observation, dtype=np.float32).copy(),
        "to_play": np.int64(DEFAULT_TO_PLAY),
        "env_row": np.int32(env_row),
        "player": np.int16(player),
        "record_index": np.int32(record_index),
        "next_record_index": np.int32(next_record_index),
        "policy_row": np.int32(policy_row),
        "policy_source": str(record.policy_source),
        "source_record_ref": _compact_ref(record.source_record_ref),
    }


def _target_rows_from_dicts(
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata: dict[str, Any],
) -> SourceStateMultiplayerTargetRowsV0:
    if not rows:
        return SourceStateMultiplayerTargetRowsV0(
            metadata=metadata,
            observation=np.zeros((0, *FRAME_STACK_SHAPE), dtype=np.float32),
            action=np.zeros((0,), dtype=np.int16),
            action_mask=np.zeros((0, ACTION_COUNT), dtype=bool),
            policy_target=np.zeros((0, ACTION_COUNT), dtype=np.float32),
            root_value=np.zeros((0,), dtype=np.float32),
            reward=np.zeros((0,), dtype=np.float32),
            final_reward=np.zeros((0,), dtype=np.float32),
            done=np.zeros((0,), dtype=bool),
            terminated=np.zeros((0,), dtype=bool),
            truncated=np.zeros((0,), dtype=bool),
            next_observation=np.zeros((0, *FRAME_STACK_SHAPE), dtype=np.float32),
            to_play=np.zeros((0,), dtype=np.int64),
            env_row=np.zeros((0,), dtype=np.int32),
            player=np.zeros((0,), dtype=np.int16),
            record_index=np.zeros((0,), dtype=np.int32),
            next_record_index=np.zeros((0,), dtype=np.int32),
            policy_row=np.zeros((0,), dtype=np.int32),
            policy_source=(),
            source_record_ref=(),
        )

    return SourceStateMultiplayerTargetRowsV0(
        metadata=metadata,
        observation=np.stack([row["observation"] for row in rows], axis=0).astype(
            np.float32,
            copy=True,
        ),
        action=np.asarray([row["action"] for row in rows], dtype=np.int16),
        action_mask=np.stack([row["action_mask"] for row in rows], axis=0).astype(
            bool,
            copy=True,
        ),
        policy_target=np.stack([row["policy_target"] for row in rows], axis=0).astype(
            np.float32,
            copy=True,
        ),
        root_value=np.asarray([row["root_value"] for row in rows], dtype=np.float32),
        reward=np.asarray([row["reward"] for row in rows], dtype=np.float32),
        final_reward=np.asarray([row["final_reward"] for row in rows], dtype=np.float32),
        done=np.asarray([row["done"] for row in rows], dtype=bool),
        terminated=np.asarray([row["terminated"] for row in rows], dtype=bool),
        truncated=np.asarray([row["truncated"] for row in rows], dtype=bool),
        next_observation=np.stack(
            [row["next_observation"] for row in rows],
            axis=0,
        ).astype(np.float32, copy=True),
        to_play=np.asarray([row["to_play"] for row in rows], dtype=np.int64),
        env_row=np.asarray([row["env_row"] for row in rows], dtype=np.int32),
        player=np.asarray([row["player"] for row in rows], dtype=np.int16),
        record_index=np.asarray([row["record_index"] for row in rows], dtype=np.int32),
        next_record_index=np.asarray(
            [row["next_record_index"] for row in rows],
            dtype=np.int32,
        ),
        policy_row=np.asarray([row["policy_row"] for row in rows], dtype=np.int32),
        policy_source=tuple(str(row["policy_source"]) for row in rows),
        source_record_ref=tuple(row["source_record_ref"] for row in rows),
    )


def _metadata(
    *,
    chunk: SourceStateMultiplayerTrainerReplayChunkV0,
    target_row_count: int,
) -> dict[str, Any]:
    chunk_metadata = chunk.metadata
    helper_metadata = _project_helper_metadata(chunk_metadata)
    helper_active = _project_helper_active(helper_metadata)
    source_fidelity_claim = chunk_metadata.get(
        "source_fidelity_claim",
        "source_state_trainer_replay_consumption",
    )
    original_source_claim = chunk_metadata.get(
        "original_curvytron_behavior_claim",
        True,
    )
    if helper_active:
        source_fidelity_claim = PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM
        original_source_claim = False

    metadata: dict[str, Any] = {
        "metadata_schema_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_METADATA_SCHEMA_ID,
        "target_contract_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
        "target_schema_id": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
        "target_schema_hash": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
        "policy_row_record_schema_id": POLICY_ROW_RECORD_SCHEMA_ID,
        "source_replay_contract_id": chunk_metadata.get(
            "replay_contract_id",
            SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_CONTRACT_ID,
        ),
        "source_replay_schema_id": chunk_metadata.get(
            "replay_schema_id",
            SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_ID,
        ),
        "source_replay_schema_hash": chunk_metadata.get(
            "replay_schema_hash",
            SOURCE_STATE_MULTIPLAYER_TRAINER_REPLAY_SCHEMA_HASH,
        ),
        "surface_schema_id": chunk_metadata.get("surface_schema_id"),
        "target_row_count": int(target_row_count),
        "alignment": TARGET_ROW_ALIGNMENT,
        "to_play_policy": TARGET_ROW_TO_PLAY_POLICY,
        "native_game_segment_claim": False,
        "lightzero_native_game_segment_claim": False,
        "lightzero_training_integration_claim": False,
        "lightzero_training_claim": False,
        "muzero_game_buffer_claim": False,
        "learner_update_claim": False,
        "policy_improvement_claim": False,
        "source_fidelity_claim": source_fidelity_claim,
        "original_curvytron_behavior_claim": bool(original_source_claim),
        "project_training_helper_active": bool(helper_active),
        "project_training_helper_metadata": helper_metadata,
        "non_claims": SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_NON_CLAIMS,
    }
    for key, value in helper_metadata.items():
        metadata[key] = value
    return metadata


def _sample_metadata(
    *,
    target_rows: SourceStateMultiplayerTargetRowsV0,
    sample_row_count: int,
    seed: int,
    replace: bool,
) -> dict[str, Any]:
    metadata = dict(target_rows.metadata)
    metadata.update(
        {
            "sample_metadata_schema_id": (
                SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_METADATA_SCHEMA_ID
            ),
            "sample_contract_id": SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_CONTRACT_ID,
            "sample_schema_id": SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_SCHEMA_ID,
            "sample_row_count": int(sample_row_count),
            "source_target_contract_id": target_rows.metadata.get(
                "target_contract_id",
                SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID,
            ),
            "source_target_schema_id": target_rows.metadata.get(
                "target_schema_id",
                SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID,
            ),
            "source_target_schema_hash": target_rows.metadata.get(
                "target_schema_hash",
                SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH,
            ),
            "seed": int(seed),
            "replace": bool(replace),
            "native_game_segment_claim": False,
            "lightzero_native_game_segment_claim": False,
            "lightzero_training_integration_claim": False,
            "lightzero_training_claim": False,
            "muzero_game_buffer_claim": False,
            "learner_update_claim": False,
            "policy_improvement_claim": False,
        }
    )
    return metadata


def _policy_record(record: PolicyRowRecordV0 | Mapping[str, Any]) -> PolicyRowRecordV0:
    if isinstance(record, PolicyRowRecordV0):
        return record
    return PolicyRowRecordV0(
        record_index=int(record["record_index"]),
        policy_row=int(record["policy_row"]),
        env_row=int(record["env_row"]),
        player=int(record["player"]),
        action=int(record["action"]),
        action_mask=np.asarray(record["action_mask"], dtype=bool),
        policy_target=np.asarray(record["policy_target"], dtype=np.float32),
        root_value=float(record["root_value"]),
        policy_source=str(record["policy_source"]),
        source_record_ref=record.get("source_record_ref"),
    )


def _reject_duplicate_policy_records(records: Sequence[PolicyRowRecordV0]) -> None:
    seen: set[tuple[int, int]] = set()
    for record in records:
        key = (int(record.record_index), int(record.policy_row))
        if key in seen:
            raise ReplayCompatibilityError(
                "duplicate policy row records for one replay record/policy row"
            )
        seen.add(key)


def _validate_policy_row_arrays(
    policy: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    player_count: int,
) -> int:
    required = (
        "policy_observation",
        "policy_action_mask",
        "policy_env_row",
        "policy_player",
    )
    missing = sorted(key for key in required if key not in policy)
    if missing:
        raise ReplayCompatibilityError(
            "policy row record is missing arrays: " + ", ".join(missing)
        )
    policy_observation = _array(
        policy["policy_observation"],
        "policy_observation",
        np.float32,
    )
    if policy_observation.ndim != 4 or policy_observation.shape[1:] != FRAME_STACK_SHAPE:
        raise ReplayCompatibilityError(
            "policy_observation must have shape [R,4,64,64]"
        )
    row_count = int(policy_observation.shape[0])
    _expect_shape(
        _array(policy["policy_action_mask"], "policy_action_mask", bool),
        "policy_action_mask",
        (row_count, ACTION_COUNT),
    )
    policy_env_row = _array(policy["policy_env_row"], "policy_env_row", np.int32)
    policy_player = _array(policy["policy_player"], "policy_player", np.int16)
    _expect_shape(policy_env_row, "policy_env_row", (row_count,))
    _expect_shape(policy_player, "policy_player", (row_count,))
    if row_count:
        if int(policy_env_row.min()) < 0 or int(policy_env_row.max()) >= batch_size:
            raise ReplayCompatibilityError("policy_env_row contains out-of-range rows")
        if int(policy_player.min()) < 0 or int(policy_player.max()) >= player_count:
            raise ReplayCompatibilityError("policy_player contains out-of-range players")
    return row_count


def _project_helper_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _compact_ref(metadata[key])
        for key in PROJECT_HELPER_METADATA_KEYS
        if key in metadata and metadata[key] is not None
    }


def _project_helper_active(metadata: Mapping[str, Any]) -> bool:
    if metadata.get("death_mode") == "profile_no_death":
        return True
    if bool(metadata.get("disable_death_for_profile", False)):
        return True
    if bool(metadata.get("death_suppression_for_profile", False)):
        return True
    if bool(metadata.get("death_immunity_diagnostic", False)):
        return True
    if _nonempty_sequence(metadata.get("death_immunity_player_ids")):
        return True
    if metadata.get("death_immunity_claim") not in (None, "", "none"):
        return True
    if metadata.get("opponent_death_mode") == "immortal":
        return True
    if bool(metadata.get("opponent_death_mode_diagnostic", False)):
        return True
    if metadata.get("opponent_death_mode_claim") not in (None, "", "none"):
        return True
    if metadata.get("opponent_runtime_mode") == "blank_canvas_noop":
        return True
    if bool(metadata.get("blank_canvas_noop", False)):
        return True
    if bool(metadata.get("profile_mode_enabled", False)):
        return True
    return bool(metadata.get("training_only_mode_enabled", False))


def _nonempty_sequence(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return bool(value.size)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return bool(value)
    return False


def _array(value: Any, name: str, dtype: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ReplayCompatibilityError(f"{name} must be convertible to {dtype}") from exc


def _expect_shape(array: np.ndarray, name: str, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ReplayCompatibilityError(
            f"{name} must have shape {shape}, got {array.shape}"
        )


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


__all__ = [
    "ACTION_COUNT",
    "DEFAULT_TO_PLAY",
    "POLICY_ROW_RECORD_SCHEMA_ID",
    "PROJECT_HELPER_METADATA_KEYS",
    "PROJECT_HELPER_RESTRICTED_SOURCE_FIDELITY_CLAIM",
    "SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_CONTRACT_ID",
    "SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_METADATA_SCHEMA_ID",
    "SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_NON_CLAIMS",
    "SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_HASH",
    "SOURCE_STATE_MULTIPLAYER_TARGET_ROWS_SCHEMA_ID",
    "SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_CONTRACT_ID",
    "SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_METADATA_SCHEMA_ID",
    "SOURCE_STATE_MULTIPLAYER_SAMPLE_BATCH_SCHEMA_ID",
    "SourceStateMultiplayerSampleBatchV0",
    "SourceStateMultiplayerTargetRowsV0",
    "PolicyRowRecordV0",
    "build_source_state_multiplayer_sample_batch_v0",
    "build_source_state_multiplayer_target_rows_v0",
]
