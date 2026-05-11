"""Bridge trainer observation batches to replay chunk v0."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from curvyzero.env.trainer_contract import ACTION_NAMES
from curvyzero.env.trainer_contract import ACTION_SPACE_HASH
from curvyzero.env.trainer_contract import ACTION_SPACE_ID
from curvyzero.env.trainer_contract import LIGHTZERO_ACTION_MASK_DTYPE
from curvyzero.env.trainer_contract import LIGHTZERO_FLAT_OBSERVATION_SHAPE
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_HASH
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_ID
from curvyzero.env.trainer_contract import REWARD_SCHEMA_HASH
from curvyzero.env.trainer_contract import REWARD_SCHEMA_ID
from curvyzero.env.trainer_observation import TrainerObservationBatch1v1
from curvyzero.training import replay_chunk_v0


PRODUCER = "curvyzero.training.trainer_replay_v0_builder"
PLAYER_COUNT = replay_chunk_v0.PLAYER_COUNT
ACTION_COUNT = len(ACTION_NAMES)


def build_replay_v0_arrays_from_trainer_batches(
    batches: Sequence[TrainerObservationBatch1v1]
    | Sequence[Sequence[TrainerObservationBatch1v1]],
    *,
    actions: ArrayLike,
    action_weights: ArrayLike,
    root_value: ArrayLike,
    episode_id: str | Sequence[str],
    reset_seed: ArrayLike,
    reset_source: str | Sequence[str],
) -> dict[str, np.ndarray]:
    """Pack trainer batches into replay-v0 arrays.

    Inputs may use either the original B=1 shape or the vector-row shape:

    - ``batches`` is ``[T]`` of ``TrainerObservationBatch1v1`` for B=1, or
      ``[T][B]`` for vector rows.
    - ``actions`` is ``int[T,2]`` for B=1, or ``int[T,B,2]``.
    - ``action_weights`` is ``float[T,2,ACTION_COUNT]`` for B=1, or
      ``float[T,B,2,ACTION_COUNT]``.
    - ``root_value`` is ``float[T,2]`` for B=1, or ``float[T,B,2]``.

    If the row is terminal, the terminal batch must be the final supplied step.
    ``final_observation`` is always the last trainer observation; for terminal
    rows this is the terminal observation. ``final_reward_map`` is the terminal
    reward map when done, otherwise zeros because replay-v0 has no absent
    sentinel for nonterminal chunks.
    """

    rows = _validate_batch_grid(tuple(batches))
    chunk_steps = len(rows)
    batch_size = len(rows[0])

    observation = np.stack(
        [np.stack([batch.observation for batch in step_rows]) for step_rows in rows],
    )
    reward = np.stack(
        [np.stack([batch.rewards for batch in step_rows]) for step_rows in rows],
    )
    terminated = np.asarray(
        [[batch.terminated for batch in step_rows] for step_rows in rows],
        dtype=np.bool_,
    )
    truncated = np.asarray(
        [[batch.truncated for batch in step_rows] for step_rows in rows],
        dtype=np.bool_,
    )
    done = np.logical_or(terminated, truncated)

    final_reward_map = np.zeros((batch_size, PLAYER_COUNT), dtype=np.float32)
    for row_index in range(batch_size):
        terminal_indices = np.flatnonzero(done[:, row_index])
        if not terminal_indices.size:
            continue
        terminal_index = int(terminal_indices[0])
        if terminal_index != chunk_steps - 1 or terminal_indices.size != 1:
            raise replay_chunk_v0.ReplayCompatibilityError(
                "terminal trainer batch must be the final supplied step"
            )
        terminal_batch = rows[-1][row_index]
        final_reward_map[row_index] = _player_ordered_reward_map(
            terminal_batch.final_reward_map,
            player_ids=terminal_batch.player_ids,
            where=f"batches[-1][{row_index}].final_reward_map",
        )

    return {
        "observation": observation.astype(np.float32, copy=True),
        "reward": reward.astype(np.float32, copy=True),
        "action": _normalize_actions(actions, chunk_steps, batch_size),
        "action_weights": _normalize_action_weights(action_weights, chunk_steps, batch_size),
        "root_value": _normalize_root_value(root_value, chunk_steps, batch_size),
        "done": done.astype(np.bool_, copy=True),
        "terminated": terminated.astype(np.bool_, copy=True),
        "truncated": truncated.astype(np.bool_, copy=True),
        "episode_id": _string_array(episode_id, "episode_id", batch_size),
        "reset_seed": _seed_array(reset_seed, batch_size),
        "reset_source": _string_array(reset_source, "reset_source", batch_size),
        "final_observation": observation[-1].astype(np.float32, copy=True),
        "final_reward_map": final_reward_map,
    }


def build_trainer_replay_v0_metadata(
    arrays: Mapping[str, Any],
    *,
    rules_hash: str,
    ruleset_id: str | None = None,
    producer: str | None = PRODUCER,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build replay-v0 metadata for trainer observation/reward payloads."""

    return replay_chunk_v0.build_replay_chunk_v0_metadata(
        arrays,
        rules_hash=rules_hash,
        observation_schema_hash=OBSERVATION_SCHEMA_HASH,
        action_space_hash=ACTION_SPACE_HASH,
        reward_schema_hash=REWARD_SCHEMA_HASH,
        ruleset_id=ruleset_id,
        observation_schema_id=OBSERVATION_SCHEMA_ID,
        action_space_id=ACTION_SPACE_ID,
        reward_schema_id=REWARD_SCHEMA_ID,
        producer=producer,
        created_at=created_at,
    )


def build_trainer_replay_chunk_v0(
    batches: Sequence[TrainerObservationBatch1v1]
    | Sequence[Sequence[TrainerObservationBatch1v1]],
    *,
    actions: ArrayLike,
    action_weights: ArrayLike,
    root_value: ArrayLike,
    episode_id: str | Sequence[str],
    reset_seed: ArrayLike,
    reset_source: str | Sequence[str],
    rules_hash: str,
    ruleset_id: str | None = None,
    producer: str | None = PRODUCER,
    created_at: str | None = None,
) -> replay_chunk_v0.ReplayChunkV0:
    """Build and validate one replay-v0 chunk from trainer batches."""

    arrays = build_replay_v0_arrays_from_trainer_batches(
        batches,
        actions=actions,
        action_weights=action_weights,
        root_value=root_value,
        episode_id=episode_id,
        reset_seed=reset_seed,
        reset_source=reset_source,
    )
    metadata = build_trainer_replay_v0_metadata(
        arrays,
        rules_hash=rules_hash,
        ruleset_id=ruleset_id,
        producer=producer,
        created_at=created_at,
    )
    normalized = replay_chunk_v0.validate_replay_chunk_v0(
        arrays=arrays,
        metadata=metadata,
    )
    return replay_chunk_v0.ReplayChunkV0(metadata=metadata, arrays=normalized)


def write_trainer_replay_chunk_v0(
    path: str | Path,
    chunk: replay_chunk_v0.ReplayChunkV0,
) -> None:
    """Write a chunk returned by ``build_trainer_replay_chunk_v0``."""

    replay_chunk_v0.write_replay_chunk_v0(
        path,
        arrays=chunk.arrays,
        metadata=chunk.metadata,
    )


def _validate_batch_grid(
    batches: (
        tuple[TrainerObservationBatch1v1, ...]
        | tuple[Sequence[TrainerObservationBatch1v1], ...]
    ),
) -> tuple[tuple[TrainerObservationBatch1v1, ...], ...]:
    if not batches:
        raise replay_chunk_v0.ReplayCompatibilityError("batches must be non-empty")

    if isinstance(batches[0], TrainerObservationBatch1v1):
        rows = tuple((batch,) for batch in batches)
    else:
        rows = tuple(
            _batch_row_tuple(step_rows, where=f"batches[{time_index}]")
            for time_index, step_rows in enumerate(batches)
        )

    batch_size = len(rows[0])
    for time_index, step_rows in enumerate(rows):
        if len(step_rows) != batch_size:
            raise replay_chunk_v0.ReplayCompatibilityError(
                "batches must have a consistent batch size: expected "
                f"{batch_size}, got {len(step_rows)} at timestep {time_index}"
            )

    player_ids = rows[0][0].player_ids
    first_observation = np.asarray(rows[0][0].observation)
    if first_observation.ndim != 2:
        raise replay_chunk_v0.ReplayCompatibilityError(
            "batches[0][0].observation must have shape [2,D]"
        )
    obs_dim = int(first_observation.shape[1])
    if obs_dim != LIGHTZERO_FLAT_OBSERVATION_SHAPE[0]:
        raise replay_chunk_v0.ReplayCompatibilityError(
            "trainer observation width mismatch: expected "
            f"{LIGHTZERO_FLAT_OBSERVATION_SHAPE[0]}, got {obs_dim}"
        )

    for time_index, step_rows in enumerate(rows):
        for row_index, batch in enumerate(step_rows):
            _validate_batch(
                batch,
                player_ids=player_ids,
                obs_dim=obs_dim,
                where=f"batches[{time_index}][{row_index}]",
            )
            if time_index < len(rows) - 1 and bool(batch.done):
                raise replay_chunk_v0.ReplayCompatibilityError(
                    "terminal trainer batch must be the final supplied step"
                )
    return rows


def _batch_row_tuple(
    value: Sequence[TrainerObservationBatch1v1],
    *,
    where: str,
) -> tuple[TrainerObservationBatch1v1, ...]:
    if isinstance(value, TrainerObservationBatch1v1):
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{where} must be a sequence of TrainerObservationBatch1v1 rows"
        )
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{where} must be a sequence of TrainerObservationBatch1v1 rows"
        ) from exc
    if not rows:
        raise replay_chunk_v0.ReplayCompatibilityError(f"{where} must be non-empty")
    for row_index, batch in enumerate(rows):
        if not isinstance(batch, TrainerObservationBatch1v1):
            raise replay_chunk_v0.ReplayCompatibilityError(
                f"{where}[{row_index}] must be TrainerObservationBatch1v1"
            )
    return rows


def _validate_batch(
    batch: TrainerObservationBatch1v1,
    *,
    player_ids: tuple[str, str],
    obs_dim: int,
    where: str,
) -> None:
    if batch.player_ids != player_ids:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{where}.player_ids mismatch: expected {player_ids!r}, got {batch.player_ids!r}"
        )
    if len(batch.player_ids) != PLAYER_COUNT:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{where}.player_ids must contain exactly {PLAYER_COUNT} players"
        )

    _expect_array(batch.observation, f"{where}.observation", (PLAYER_COUNT, obs_dim), np.float32)
    _expect_array(batch.rewards, f"{where}.rewards", (PLAYER_COUNT,), np.float32)
    _expect_array(batch.action_mask, f"{where}.action_mask", (PLAYER_COUNT, ACTION_COUNT), np.bool_)
    _expect_array(
        batch.lightzero_action_mask,
        f"{where}.lightzero_action_mask",
        (PLAYER_COUNT, ACTION_COUNT),
        np.dtype(LIGHTZERO_ACTION_MASK_DTYPE),
    )

    if bool(batch.done) != (bool(batch.terminated) or bool(batch.truncated)):
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{where}.done must equal terminated | truncated"
        )

    reward_map = _player_ordered_reward_map(
        batch.reward_map,
        player_ids=batch.player_ids,
        where=f"{where}.reward_map",
    )
    if not np.array_equal(reward_map, batch.rewards):
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{where}.reward_map must match rewards in player order"
        )

    if bool(batch.done):
        final_reward_map = _player_ordered_reward_map(
            batch.final_reward_map,
            player_ids=batch.player_ids,
            where=f"{where}.final_reward_map",
        )
        if not np.array_equal(final_reward_map, batch.rewards):
            raise replay_chunk_v0.ReplayCompatibilityError(
                f"{where}.final_reward_map must match terminal rewards"
            )
    elif batch.final_reward_map is not None:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{where}.final_reward_map must be None for nonterminal rows"
        )


def _player_ordered_reward_map(
    reward_map: Mapping[str, float] | None,
    *,
    player_ids: tuple[str, str],
    where: str,
) -> np.ndarray:
    if reward_map is None:
        raise replay_chunk_v0.ReplayCompatibilityError(f"{where} is required")
    if set(reward_map) != set(player_ids):
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{where} keys must match player_ids {player_ids!r}"
        )
    return np.asarray([float(reward_map[player_id]) for player_id in player_ids], dtype=np.float32)


def _normalize_actions(value: ArrayLike, chunk_steps: int, batch_size: int) -> np.ndarray:
    raw = np.asarray(value)
    if batch_size == 1 and raw.shape == (chunk_steps, PLAYER_COUNT):
        raw = raw.reshape(chunk_steps, 1, PLAYER_COUNT)
    expected = (chunk_steps, batch_size, PLAYER_COUNT)
    if raw.shape != expected:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"actions shape mismatch: expected {list(expected)}, got {list(raw.shape)}"
        )
    if not np.issubdtype(raw.dtype, np.integer):
        raise replay_chunk_v0.ReplayCompatibilityError("actions must contain integer ids")
    if np.any(raw < 0) or np.any(raw >= ACTION_COUNT):
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"actions values must be in [0, {ACTION_COUNT})"
        )
    return raw.astype(np.int16, copy=True)


def _normalize_action_weights(
    value: ArrayLike,
    chunk_steps: int,
    batch_size: int,
) -> np.ndarray:
    raw = np.asarray(value)
    if batch_size == 1 and raw.shape == (chunk_steps, PLAYER_COUNT, ACTION_COUNT):
        raw = raw.reshape(chunk_steps, 1, PLAYER_COUNT, ACTION_COUNT)
    expected = (chunk_steps, batch_size, PLAYER_COUNT, ACTION_COUNT)
    if raw.shape != expected:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"action_weights shape mismatch: expected {list(expected)}, got {list(raw.shape)}"
        )
    return _astype_float32(raw, "action_weights")


def _normalize_root_value(value: ArrayLike, chunk_steps: int, batch_size: int) -> np.ndarray:
    raw = np.asarray(value)
    if batch_size == 1 and raw.shape == (chunk_steps, PLAYER_COUNT):
        raw = raw.reshape(chunk_steps, 1, PLAYER_COUNT)
    expected = (chunk_steps, batch_size, PLAYER_COUNT)
    if raw.shape != expected:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"root_value shape mismatch: expected {list(expected)}, got {list(raw.shape)}"
        )
    return _astype_float32(raw, "root_value")


def _string_array(value: str | Sequence[str], key: str, batch_size: int) -> np.ndarray:
    if isinstance(value, str):
        if batch_size != 1:
            raise replay_chunk_v0.ReplayCompatibilityError(
                f"{key} must have one entry per batch row"
            )
        entries = (value,)
    else:
        raw = np.asarray(value)
        if raw.shape != (batch_size,):
            raise replay_chunk_v0.ReplayCompatibilityError(
                f"{key} shape mismatch: expected {[batch_size]}, got {list(raw.shape)}"
            )
        entries = tuple(raw.tolist())
    normalized = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, (str, np.str_)) or not str(entry):
            raise replay_chunk_v0.ReplayCompatibilityError(
                f"{key}[{index}] must be a non-empty string"
            )
        normalized.append(str(entry))
    max_len = max(len(entry) for entry in normalized)
    return np.asarray(normalized, dtype=f"<U{max_len}")


def _seed_array(value: ArrayLike, batch_size: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == 0:
        if batch_size != 1:
            raise replay_chunk_v0.ReplayCompatibilityError(
                "reset_seed must have one entry per batch row"
            )
        raw = raw.reshape(1)
    if raw.shape != (batch_size,):
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"reset_seed shape mismatch: expected {[batch_size]}, got {list(raw.shape)}"
        )
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
        raise replay_chunk_v0.ReplayCompatibilityError("reset_seed must contain integers")
    if np.any(raw < 0):
        raise replay_chunk_v0.ReplayCompatibilityError("reset_seed must be non-negative")
    return raw.astype(np.int64, copy=True)


def _expect_array(
    value: np.ndarray,
    key: str,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any] | type[np.generic],
) -> None:
    array = np.asarray(value)
    dtype = np.dtype(expected_dtype)
    if array.shape != expected_shape:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{key} shape mismatch: expected {list(expected_shape)}, got {list(array.shape)}"
        )
    if array.dtype != dtype:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{key} dtype mismatch: expected {dtype}, got {array.dtype}"
        )


def _astype_float32(value: np.ndarray, key: str) -> np.ndarray:
    try:
        return value.astype(np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise replay_chunk_v0.ReplayCompatibilityError(
            f"{key} must be convertible to float32"
        ) from exc


__all__ = [
    "ACTION_COUNT",
    "PRODUCER",
    "build_replay_v0_arrays_from_trainer_batches",
    "build_trainer_replay_chunk_v0",
    "build_trainer_replay_v0_metadata",
    "write_trainer_replay_chunk_v0",
]
