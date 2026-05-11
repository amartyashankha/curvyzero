"""Replay recorder for live ``VectorTrainerEnv1v1NoBonus`` step batches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from curvyzero.env import vector_reset
from curvyzero.env.trainer_contract import LIGHTZERO_FLAT_OBSERVATION_SHAPE
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_HASH
from curvyzero.env.trainer_contract import OBSERVATION_SCHEMA_ID
from curvyzero.env.trainer_contract import RAY_ANGLES_DEGREES
from curvyzero.env.trainer_contract import RAY_CHANNEL_NAMES
from curvyzero.env.trainer_contract import REWARD_SCHEMA_HASH
from curvyzero.env.trainer_contract import REWARD_SCHEMA_ID
from curvyzero.env.trainer_contract import SCALAR_NAMES
from curvyzero.env.trainer_observation import TrainerObservationBatch1v1
from curvyzero.env.vector_trainer_env import ACTION_COUNT
from curvyzero.env.vector_trainer_env import DEFAULT_PLAYER_IDS
from curvyzero.env.vector_trainer_env import PLAYER_COUNT
from curvyzero.env.vector_trainer_env import VectorTrainerBatch1v1NoBonus
from curvyzero.env.vector_trainer_env import VectorTrainerEnv1v1NoBonus
from curvyzero.training import replay_chunk_v0
from curvyzero.training import trainer_replay_v0_builder


PRODUCER = "curvyzero.training.vector_env_replay_recorder"
VECTOR_ENV_REPLAY_MANIFEST_SCHEMA_ID = (
    "curvyzero_vector_trainer_1v1_no_bonus_replay_manifest/v0"
)
VECTOR_ENV_IMPL_ID = "VectorTrainerEnv1v1NoBonus"
STRICT_VECTOR_FEATURE_FLAGS = ("strict_1v1", "no_bonus", "P=2")
FINAL_OBSERVATION_POLICY = (
    "terminal rows use batch.final_observation; nonterminal rows use the last "
    "returned observation because replay-v0 has no absent final_observation sentinel"
)
VECTOR_ENV_ROW_METADATA_SCHEMA_ID = "curvyzero_vector_env_replay_row_metadata/v0"
VECTOR_ENV_ROW_METADATA_ENV_ID = "VectorTrainerEnv1v1NoBonus"
VECTOR_ENV_RNG_STATE_POLICY = "not_recorded_reset_seed_only"


class VectorEnvReplayRecorderError(ValueError):
    """Raised when live vector env batches cannot be packed into replay-v0."""


@dataclass(slots=True)
class VectorEnvReplayRecorder:
    """Collect live vector-env step batches and build one trainer replay chunk.

    replay-v0 stores one episode id/reset tuple per B row, not a per-timestep
    episode boundary. To avoid mixing episodes after vector autoreset, any
    recorded batch with ``done.any()`` is a terminal barrier: the batch may be
    built into the current chunk, but subsequent env steps belong in a fresh
    recorder with the next chunk's reset metadata.
    """

    player_ids: tuple[str, str] = DEFAULT_PLAYER_IDS
    _batches: list[VectorTrainerBatch1v1NoBonus] = field(default_factory=list)
    _actions: list[np.ndarray] = field(default_factory=list)
    _action_weights: list[np.ndarray] = field(default_factory=list)
    _root_value: list[np.ndarray] = field(default_factory=list)
    _batch_size: int | None = None
    _closed_by_terminal: bool = False

    @property
    def step_count(self) -> int:
        """Number of recorded env steps."""

        return len(self._batches)

    @property
    def batch_size(self) -> int | None:
        """Vector row count once the first step has been recorded."""

        return self._batch_size

    def record_step(
        self,
        batch: VectorTrainerBatch1v1NoBonus,
        *,
        actions: ArrayLike,
        action_weights: ArrayLike,
        root_value: ArrayLike,
    ) -> None:
        """Append one returned env step batch and close on terminal rows."""

        if self._closed_by_terminal:
            raise VectorEnvReplayRecorderError(
                "terminal vector env batch must be the final recorded timestep"
            )
        batch_size = _validate_vector_step_batch(batch)
        if self._batch_size is None:
            self._batch_size = batch_size
        elif batch_size != self._batch_size:
            raise VectorEnvReplayRecorderError(
                f"B row mismatch: expected {self._batch_size}, got {batch_size}"
            )

        self._batches.append(batch)
        self._actions.append(_step_actions(actions, batch_size))
        self._action_weights.append(_step_action_weights(action_weights, batch_size))
        self._root_value.append(_step_root_value(root_value, batch_size))
        self._closed_by_terminal = bool(np.asarray(batch.done, dtype=bool).any())

    def build_chunk(
        self,
        *,
        episode_id: str | Sequence[str] | ArrayLike | None = None,
        reset_seed: ArrayLike | None = None,
        reset_source: str | Sequence[str] | ArrayLike | None = None,
        rules_hash: str,
        ruleset_id: str | None = None,
        producer: str | None = PRODUCER,
        created_at: str | None = None,
    ) -> replay_chunk_v0.ReplayChunkV0:
        """Build a replay-v0 chunk through ``trainer_replay_v0_builder``.

        When episode/reset metadata is omitted, the recorder reads the public
        ``batch.info`` arrays from the recorded env step batches.
        """

        if not self._batches or self._batch_size is None:
            raise VectorEnvReplayRecorderError("at least one step batch is required")

        batches = vector_step_batches_to_trainer_rows(
            self._batches,
            player_ids=self.player_ids,
        )
        chunk = trainer_replay_v0_builder.build_trainer_replay_chunk_v0(
            batches,
            actions=np.stack(self._actions),
            action_weights=np.stack(self._action_weights),
            root_value=np.stack(self._root_value),
            episode_id=_metadata_episode_id(
                episode_id,
                self._batches,
                batch_size=self._batch_size,
            ),
            reset_seed=_metadata_reset_seed(
                reset_seed,
                self._batches,
                batch_size=self._batch_size,
            ),
            reset_source=_metadata_reset_source(
                reset_source,
                self._batches,
                batch_size=self._batch_size,
            ),
            rules_hash=rules_hash,
            ruleset_id=ruleset_id,
            producer=producer,
            created_at=created_at,
        )
        metadata = dict(chunk.metadata)
        metadata["vector_env_row_metadata"] = _vector_env_row_metadata(
            self._batches,
            batch_size=self._batch_size,
            chunk_episode_id=chunk.arrays["episode_id"],
            chunk_reset_seed=chunk.arrays["reset_seed"],
            chunk_reset_source=chunk.arrays["reset_source"],
        )
        normalized = replay_chunk_v0.validate_replay_chunk_v0(
            arrays=chunk.arrays,
            metadata=metadata,
        )
        return replay_chunk_v0.ReplayChunkV0(metadata=metadata, arrays=normalized)


def build_vector_env_replay_chunk_v0(
    step_batches: Sequence[VectorTrainerBatch1v1NoBonus],
    *,
    actions: ArrayLike,
    action_weights: ArrayLike,
    root_value: ArrayLike,
    episode_id: str | Sequence[str] | ArrayLike | None = None,
    reset_seed: ArrayLike | None = None,
    reset_source: str | Sequence[str] | ArrayLike | None = None,
    rules_hash: str,
    ruleset_id: str | None = None,
    producer: str | None = PRODUCER,
    created_at: str | None = None,
    player_ids: tuple[str, str] = DEFAULT_PLAYER_IDS,
) -> replay_chunk_v0.ReplayChunkV0:
    """Build a replay-v0 chunk from live env step batches.

    A terminal vector batch is accepted only as the final supplied timestep;
    callers should start a new recorder/chunk for any later env steps.
    Omitted episode/reset metadata is read from public ``batch.info`` arrays.
    """

    if len(step_batches) == 0:
        raise VectorEnvReplayRecorderError("at least one step batch is required")
    recorder = VectorEnvReplayRecorder(player_ids=player_ids)
    action_array = np.asarray(actions)
    action_weights_array = np.asarray(action_weights)
    root_value_array = np.asarray(root_value)
    if action_array.ndim == 0 or action_array.shape[0] != len(step_batches):
        raise VectorEnvReplayRecorderError("actions must have one timestep per step batch")
    if (
        action_weights_array.ndim == 0
        or action_weights_array.shape[0] != len(step_batches)
    ):
        raise VectorEnvReplayRecorderError(
            "action_weights must have one timestep per step batch"
        )
    if root_value_array.ndim == 0 or root_value_array.shape[0] != len(step_batches):
        raise VectorEnvReplayRecorderError("root_value must have one timestep per step batch")
    for index, batch in enumerate(step_batches):
        recorder.record_step(
            batch,
            actions=action_array[index],
            action_weights=action_weights_array[index],
            root_value=root_value_array[index],
        )
    return recorder.build_chunk(
        episode_id=episode_id,
        reset_seed=reset_seed,
        reset_source=reset_source,
        rules_hash=rules_hash,
        ruleset_id=ruleset_id,
        producer=producer,
        created_at=created_at,
    )


def build_vector_env_replay_manifest_v0(
    *,
    chunk: replay_chunk_v0.ReplayChunkV0,
    env: VectorTrainerEnv1v1NoBonus,
    included_stages: Sequence[str],
    step_batches: Sequence[VectorTrainerBatch1v1NoBonus] = (),
    source_claim_id: str | None = None,
) -> dict[str, Any]:
    """Build a minimal sidecar contract for one strict vector replay chunk.

    This does not alter replay-v0 storage. It records the strict env/profile
    claim that should sit beside a chunk when optimizer or throughput reports
    need to state exactly which vector path produced it.
    """

    if not isinstance(chunk, replay_chunk_v0.ReplayChunkV0):
        raise VectorEnvReplayRecorderError("chunk must be ReplayChunkV0")
    if not isinstance(env, VectorTrainerEnv1v1NoBonus):
        raise VectorEnvReplayRecorderError("env must be VectorTrainerEnv1v1NoBonus")
    _validate_strict_chunk_for_manifest(chunk, env)

    metadata = chunk.metadata
    manifest: dict[str, Any] = {
        "manifest_schema_id": VECTOR_ENV_REPLAY_MANIFEST_SCHEMA_ID,
        "env_impl_id": VECTOR_ENV_IMPL_ID,
        "feature_flags": list(STRICT_VECTOR_FEATURE_FLAGS),
        "event_mode": env.event_mode,
        "decision_ms": float(env.decision_ms),
        "capacities": {
            "batch_size": int(env.batch_size),
            "player_count": PLAYER_COUNT,
            "obs_dim": int(metadata["obs_dim"]),
            "action_count": int(metadata["action_count"]),
            "chunk_steps": int(metadata["chunk_steps"]),
            "body_capacity": int(env.body_capacity),
            "event_capacity": int(env.event_capacity),
            "timer_capacity": int(env.timer_capacity),
            "random_tape_capacity": int(env.random_tape_capacity),
        },
        "reset_seed": [int(value) for value in chunk.arrays["reset_seed"].tolist()],
        "reset_source": [str(value) for value in chunk.arrays["reset_source"].tolist()],
        "final_observation_policy": _manifest_final_observation_policy(
            step_batches,
            batch_size=int(env.batch_size),
        ),
        "included_stages": _manifest_string_list(included_stages, "included_stages"),
        "replay_contract_id": str(metadata["replay_contract_id"]),
        "replay_schema_id": str(metadata["replay_schema_id"]),
        "replay_schema_hash": str(metadata["replay_schema_hash"]),
    }
    for key in ("ruleset_id", "rules_hash"):
        if key in metadata:
            manifest[key] = _manifest_non_empty_string(metadata[key], key)
    if source_claim_id is not None:
        manifest["source_claim_id"] = _manifest_non_empty_string(
            source_claim_id,
            "source_claim_id",
        )
    return manifest


def vector_step_batches_to_trainer_rows(
    step_batches: Sequence[VectorTrainerBatch1v1NoBonus],
    *,
    player_ids: tuple[str, str] = DEFAULT_PLAYER_IDS,
) -> tuple[tuple[TrainerObservationBatch1v1, ...], ...]:
    """Convert returned vector env step batches to per-row trainer batches."""

    rows: list[tuple[TrainerObservationBatch1v1, ...]] = []
    expected_batch_size: int | None = None
    for time_index, batch in enumerate(step_batches):
        batch_size = _validate_vector_step_batch(batch)
        if expected_batch_size is None:
            expected_batch_size = batch_size
        elif batch_size != expected_batch_size:
            raise VectorEnvReplayRecorderError(
                f"B row mismatch: expected {expected_batch_size}, got {batch_size}"
            )
        if time_index < len(step_batches) - 1 and bool(np.asarray(batch.done).any()):
            raise VectorEnvReplayRecorderError(
                "terminal vector env batch must be the final supplied timestep"
            )
        rows.append(
            tuple(
                _vector_row_to_trainer_batch(batch, row_index, player_ids=player_ids)
                for row_index in range(batch_size)
            )
        )
    if expected_batch_size is None:
        raise VectorEnvReplayRecorderError("at least one step batch is required")
    return tuple(rows)


def _validate_strict_chunk_for_manifest(
    chunk: replay_chunk_v0.ReplayChunkV0,
    env: VectorTrainerEnv1v1NoBonus,
) -> None:
    replay_chunk_v0.validate_replay_chunk_v0(
        arrays=chunk.arrays,
        metadata=chunk.metadata,
    )
    metadata = chunk.metadata
    if int(metadata["player_count"]) != PLAYER_COUNT:
        raise VectorEnvReplayRecorderError(
            f"manifest requires {PLAYER_COUNT} players"
        )
    if int(metadata["obs_dim"]) != LIGHTZERO_FLAT_OBSERVATION_SHAPE[0]:
        raise VectorEnvReplayRecorderError(
            "manifest requires VectorTrainerEnv1v1NoBonus observation width"
        )
    if int(metadata["action_count"]) != ACTION_COUNT:
        raise VectorEnvReplayRecorderError(
            "manifest requires VectorTrainerEnv1v1NoBonus action count"
        )
    if int(metadata["batch_size"]) != int(env.batch_size):
        raise VectorEnvReplayRecorderError(
            "manifest chunk batch_size must match env.batch_size"
        )


def _manifest_final_observation_policy(
    step_batches: Sequence[VectorTrainerBatch1v1NoBonus],
    *,
    batch_size: int,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "array": "final_observation",
        "policy": FINAL_OBSERVATION_POLICY,
    }
    if not step_batches:
        return {**base, "source": "static_vector_replay_policy"}

    for batch in step_batches:
        observed_batch_size = _validate_vector_step_batch(batch)
        if observed_batch_size != batch_size:
            raise VectorEnvReplayRecorderError(
                "manifest step batch size must match env.batch_size"
            )

    last_batch = step_batches[-1]
    info_policy = last_batch.info.get("final_observation_policy")
    if isinstance(info_policy, dict):
        row_mask = _manifest_bool_list(
            info_policy.get("row_mask"),
            "final_observation_policy.row_mask",
            batch_size=batch_size,
        )
        rows = _manifest_int_list(
            info_policy.get("rows"),
            "final_observation_policy.rows",
        )
        return {
            **base,
            "source": "last_step_batch.info.final_observation_policy",
            "present": bool(info_policy.get("present", False)),
            "terminal_rows": rows,
            "row_mask": row_mask,
            "terminal_rows_only": bool(info_policy.get("terminal_rows_only", True)),
            "nonterminal_rows_zero_filled": bool(
                info_policy.get("nonterminal_rows_zero_filled", False)
            ),
        }

    row_mask = [bool(value) for value in np.asarray(last_batch.done, dtype=bool).tolist()]
    return {
        **base,
        "source": "last_step_batch.done",
        "present": bool(np.asarray(last_batch.done, dtype=bool).any()),
        "terminal_rows": [index for index, done in enumerate(row_mask) if done],
        "row_mask": row_mask,
        "terminal_rows_only": True,
        "nonterminal_rows_zero_filled": False,
    }


def _vector_row_to_trainer_batch(
    batch: VectorTrainerBatch1v1NoBonus,
    row_index: int,
    *,
    player_ids: tuple[str, str],
) -> TrainerObservationBatch1v1:
    done = bool(batch.done[row_index])
    terminated = bool(batch.terminated[row_index])
    truncated = bool(batch.truncated[row_index])

    if done:
        if batch.final_observation is None:
            raise VectorEnvReplayRecorderError(
                f"batch row {row_index} is terminal but final_observation is missing"
            )
        if batch.final_reward is None:
            raise VectorEnvReplayRecorderError(
                f"batch row {row_index} is terminal but final_reward is missing"
            )
        observation = np.asarray(batch.final_observation[row_index], dtype=np.float32).copy()
        rewards = np.asarray(batch.final_reward[row_index], dtype=np.float32).copy()
        if not np.array_equal(rewards, np.asarray(batch.reward[row_index], dtype=np.float32)):
            raise VectorEnvReplayRecorderError(
                f"batch.final_reward[{row_index}] must match batch.reward[{row_index}]"
            )
        action_mask = np.zeros((PLAYER_COUNT, ACTION_COUNT), dtype=np.bool_)
        lightzero_action_mask = np.zeros((PLAYER_COUNT, ACTION_COUNT), dtype=np.int8)
        final_reward_map = _reward_map(rewards, player_ids=player_ids)
    else:
        observation = np.asarray(batch.observation[row_index], dtype=np.float32).copy()
        rewards = np.asarray(batch.reward[row_index], dtype=np.float32).copy()
        action_mask = np.asarray(batch.action_mask[row_index], dtype=np.bool_).copy()
        lightzero_action_mask = np.asarray(
            batch.lightzero_action_mask[row_index],
            dtype=np.int8,
        ).copy()
        final_reward_map = None

    rays, scalars = _split_flat_observation(observation)
    reward_map = _reward_map(rewards, player_ids=player_ids)
    return TrainerObservationBatch1v1(
        player_ids=player_ids,
        rays=rays,
        scalars=scalars,
        observation=observation,
        action_mask=action_mask,
        lightzero_action_mask=lightzero_action_mask,
        to_play=np.asarray(batch.to_play[row_index], dtype=np.int64).copy(),
        rewards=rewards,
        reward_info={
            player_id: {
                "reward_schema_id": REWARD_SCHEMA_ID,
                "reward_schema_hash": REWARD_SCHEMA_HASH,
                "observation_schema_id": OBSERVATION_SCHEMA_ID,
                "observation_schema_hash": OBSERVATION_SCHEMA_HASH,
                "ego_player_id": player_id,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            }
            for player_id in player_ids
        },
        reward_map=reward_map,
        final_reward_map=final_reward_map,
        done=done,
        terminated=terminated,
        truncated=truncated,
    )


def _split_flat_observation(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ray_count = len(RAY_ANGLES_DEGREES)
    ray_channel_count = len(RAY_CHANNEL_NAMES)
    ray_value_count = ray_count * ray_channel_count
    rays = observation[:, :ray_value_count].reshape(
        PLAYER_COUNT,
        ray_count,
        ray_channel_count,
    )
    scalars = observation[:, ray_value_count:].reshape(
        PLAYER_COUNT,
        len(SCALAR_NAMES),
    )
    return rays.astype(np.float32, copy=True), scalars.astype(np.float32, copy=True)


def _reward_map(rewards: np.ndarray, *, player_ids: tuple[str, str]) -> dict[str, float]:
    return {player_id: float(rewards[index]) for index, player_id in enumerate(player_ids)}


def _validate_vector_step_batch(batch: VectorTrainerBatch1v1NoBonus) -> int:
    if not isinstance(batch, VectorTrainerBatch1v1NoBonus):
        raise VectorEnvReplayRecorderError("batch must be VectorTrainerBatch1v1NoBonus")
    observation = np.asarray(batch.observation)
    if observation.ndim != 3:
        raise VectorEnvReplayRecorderError("batch.observation must have shape [B,2,D]")
    batch_size, player_count, obs_dim = observation.shape
    if player_count != PLAYER_COUNT or obs_dim != LIGHTZERO_FLAT_OBSERVATION_SHAPE[0]:
        raise VectorEnvReplayRecorderError(
            "batch.observation shape mismatch: expected "
            f"[B,{PLAYER_COUNT},{LIGHTZERO_FLAT_OBSERVATION_SHAPE[0]}], "
            f"got {list(observation.shape)}"
        )
    _expect_shape(
        batch.action_mask,
        "batch.action_mask",
        (batch_size, PLAYER_COUNT, ACTION_COUNT),
    )
    _expect_shape(
        batch.lightzero_action_mask,
        "batch.lightzero_action_mask",
        (batch_size, PLAYER_COUNT, ACTION_COUNT),
    )
    _expect_shape(batch.to_play, "batch.to_play", (batch_size, PLAYER_COUNT))
    _expect_shape(batch.reward, "batch.reward", (batch_size, PLAYER_COUNT))
    _expect_shape(batch.done, "batch.done", (batch_size,))
    _expect_shape(batch.terminated, "batch.terminated", (batch_size,))
    _expect_shape(batch.truncated, "batch.truncated", (batch_size,))
    done = np.asarray(batch.done, dtype=bool)
    expected_done = np.logical_or(
        np.asarray(batch.terminated, dtype=bool),
        np.asarray(batch.truncated, dtype=bool),
    )
    if not np.array_equal(done, expected_done):
        raise VectorEnvReplayRecorderError("batch.done must equal terminated | truncated")
    if bool(done.any()):
        if batch.final_observation is None:
            raise VectorEnvReplayRecorderError(
                "terminal vector env batch requires final_observation"
            )
        if batch.final_reward is None:
            raise VectorEnvReplayRecorderError("terminal vector env batch requires final_reward")
        _expect_shape(
            batch.final_observation,
            "batch.final_observation",
            (batch_size, PLAYER_COUNT, LIGHTZERO_FLAT_OBSERVATION_SHAPE[0]),
        )
        _expect_shape(batch.final_reward, "batch.final_reward", (batch_size, PLAYER_COUNT))
    return int(batch_size)


def _step_actions(value: ArrayLike, batch_size: int) -> np.ndarray:
    array = np.asarray(value)
    expected = (batch_size, PLAYER_COUNT)
    if array.shape != expected:
        raise VectorEnvReplayRecorderError(
            f"actions shape mismatch: expected {list(expected)}, got {list(array.shape)}"
        )
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise VectorEnvReplayRecorderError("actions must contain integer action ids")
    if bool(((array < 0) | (array >= ACTION_COUNT)).any()):
        raise VectorEnvReplayRecorderError(
            f"actions values must be in [0, {ACTION_COUNT})"
        )
    return array.astype(np.int16, copy=True)


def _step_action_weights(value: ArrayLike, batch_size: int) -> np.ndarray:
    array = np.asarray(value)
    expected = (batch_size, PLAYER_COUNT, ACTION_COUNT)
    if array.shape != expected:
        raise VectorEnvReplayRecorderError(
            "action_weights shape mismatch: expected "
            f"{list(expected)}, got {list(array.shape)}"
        )
    return _finite_float32(array, "action_weights")


def _step_root_value(value: ArrayLike, batch_size: int) -> np.ndarray:
    array = np.asarray(value)
    expected = (batch_size, PLAYER_COUNT)
    if array.shape != expected:
        raise VectorEnvReplayRecorderError(
            f"root_value shape mismatch: expected {list(expected)}, got {list(array.shape)}"
        )
    return _finite_float32(array, "root_value")


def _finite_float32(value: np.ndarray, key: str) -> np.ndarray:
    try:
        array = value.astype(np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise VectorEnvReplayRecorderError(f"{key} must be convertible to float32") from exc
    if not bool(np.isfinite(array).all()):
        raise VectorEnvReplayRecorderError(f"{key} must be finite")
    return array


def _expect_shape(value: Any, key: str, expected: tuple[int, ...]) -> None:
    array = np.asarray(value)
    if array.shape != expected:
        raise VectorEnvReplayRecorderError(
            f"{key} shape mismatch: expected {list(expected)}, got {list(array.shape)}"
        )


def _metadata_episode_id(
    value: str | Sequence[str] | ArrayLike | None,
    batches: Sequence[VectorTrainerBatch1v1NoBonus],
    *,
    batch_size: int,
) -> str | Sequence[str]:
    raw = value if value is not None else _find_info_array(batches, "episode_id")
    raw = raw if raw is not None else _find_info_array(batches, "reset_episode_id")
    if raw is None:
        raise VectorEnvReplayRecorderError(
            "episode_id is required when batch info does not provide it"
        )
    return _episode_id_strings(raw, batch_size=batch_size)


def _metadata_reset_seed(
    value: ArrayLike | None,
    batches: Sequence[VectorTrainerBatch1v1NoBonus],
    *,
    batch_size: int,
) -> np.ndarray:
    raw = value if value is not None else _find_info_array(batches, "reset_seed")
    if raw is None:
        raise VectorEnvReplayRecorderError(
            "reset_seed is required when batch info does not provide it"
        )
    array = np.asarray(raw)
    if array.ndim != 1 or array.shape != (batch_size,):
        raise VectorEnvReplayRecorderError(
            f"reset_seed shape mismatch: expected {[batch_size]}, got {list(array.shape)}"
        )
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise VectorEnvReplayRecorderError("reset_seed must contain integers")
    if bool((array < 0).any()):
        raise VectorEnvReplayRecorderError("reset_seed must be non-negative")
    if np.issubdtype(array.dtype, np.unsignedinteger) and bool(
        (array > np.iinfo(np.int64).max).any()
    ):
        raise VectorEnvReplayRecorderError("reset_seed values must fit int64")
    return array.astype(np.int64, copy=True)


def _metadata_reset_source(
    value: str | Sequence[str] | ArrayLike | None,
    batches: Sequence[VectorTrainerBatch1v1NoBonus],
    *,
    batch_size: int,
) -> str | Sequence[str]:
    raw = value if value is not None else _find_info_array(batches, "reset_source")
    if raw is None:
        raise VectorEnvReplayRecorderError(
            "reset_source is required when batch info does not provide it"
        )
    return _reset_source_strings(raw, batch_size=batch_size)


def _find_info_array(
    batches: Sequence[VectorTrainerBatch1v1NoBonus],
    key: str,
) -> Any | None:
    for batch in batches:
        found = _find_public_info_array_in_mapping(batch.info, key)
        if found is not None:
            return found
    for batch in batches:
        found = _find_nested_info_array_in_mapping(batch.info, key)
        if found is not None:
            return found
    return None


def _find_public_info_array_in_mapping(mapping: Any, key: str) -> Any | None:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    if value is not None:
        return value
    return None


def _find_nested_info_array_in_mapping(mapping: Any, key: str) -> Any | None:
    if not isinstance(mapping, dict):
        return None
    for nested_key in (
        "reset_info",
        "autoreset_reset_info",
        "autoreset_plan",
        "reset_metadata",
        "reset_arrays_kwargs",
    ):
        nested = mapping.get(nested_key)
        found = _find_public_info_array_in_mapping(nested, key)
        if found is not None:
            return found
        found = _find_nested_info_array_in_mapping(nested, key)
        if found is not None:
            return found
    return None


def _vector_env_row_metadata(
    batches: Sequence[VectorTrainerBatch1v1NoBonus],
    *,
    batch_size: int,
    chunk_episode_id: ArrayLike,
    chunk_reset_seed: ArrayLike,
    chunk_reset_source: ArrayLike,
) -> dict[str, Any]:
    """Build JSON-safe row reset identity metadata from public env batch info."""

    final_batch = batches[-1]
    final_info = final_batch.info
    missing_batch_info_fields: list[str] = []
    returned_episode_id = _find_public_info_array_in_mapping(
        final_info,
        "returned_episode_id",
    )
    returned_reset_seed = _find_public_info_array_in_mapping(
        final_info,
        "returned_reset_seed",
    )
    returned_reset_source = _find_public_info_array_in_mapping(
        final_info,
        "returned_reset_source",
    )
    for key, value in {
        "returned_episode_id": returned_episode_id,
        "returned_reset_seed": returned_reset_seed,
        "returned_reset_source": returned_reset_source,
    }.items():
        if value is None:
            missing_batch_info_fields.append(key)

    terminal_mask = np.asarray(final_batch.done, dtype=bool)
    metadata: dict[str, Any] = {
        "schema_id": VECTOR_ENV_ROW_METADATA_SCHEMA_ID,
        "env_id": VECTOR_ENV_ROW_METADATA_ENV_ID,
        "source": "VectorTrainerEnv1v1NoBonus.batch.info",
        "row_id": [int(row) for row in range(batch_size)],
        "terminal_row_id": [int(row) for row in np.flatnonzero(terminal_mask)],
        "terminal_row_mask": [bool(value) for value in terminal_mask.tolist()],
        "chunk_episode_id": _string_list(chunk_episode_id, "chunk_episode_id", batch_size),
        "chunk_reset_seed": _nonnegative_int_list(
            chunk_reset_seed,
            "chunk_reset_seed",
            batch_size,
        ),
        "chunk_reset_source": _string_list(
            _reset_source_strings(chunk_reset_source, batch_size=batch_size),
            "chunk_reset_source",
            batch_size,
        ),
        "returned_identity_available": not missing_batch_info_fields,
        "missing_batch_info_fields": missing_batch_info_fields,
        "rng_state_available": False,
        "rng_state_policy": VECTOR_ENV_RNG_STATE_POLICY,
    }
    if returned_episode_id is not None:
        metadata["returned_episode_id"] = _string_list(
            returned_episode_id,
            "returned_episode_id",
            batch_size,
        )
    if returned_reset_seed is not None:
        metadata["returned_reset_seed"] = _nonnegative_int_list(
            returned_reset_seed,
            "returned_reset_seed",
            batch_size,
        )
    if returned_reset_source is not None:
        metadata["returned_reset_source"] = _string_list(
            _reset_source_strings(returned_reset_source, batch_size=batch_size),
            "returned_reset_source",
            batch_size,
        )

    returned_observation_source = _find_public_info_array_in_mapping(
        final_info,
        "returned_observation_source",
    )
    if returned_observation_source is not None:
        metadata["returned_observation_source"] = _string_list(
            returned_observation_source,
            "returned_observation_source",
            batch_size,
        )
    return metadata


def _episode_id_strings(value: Any, *, batch_size: int) -> str | Sequence[str]:
    if isinstance(value, str):
        if batch_size != 1:
            raise VectorEnvReplayRecorderError("episode_id must have one entry per B row")
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        if batch_size != 1:
            raise VectorEnvReplayRecorderError("episode_id must have one entry per B row")
        entries = [array.item()]
    else:
        if array.shape != (batch_size,):
            raise VectorEnvReplayRecorderError(
                f"episode_id shape mismatch: expected {[batch_size]}, got {list(array.shape)}"
            )
        entries = list(array.tolist())
    return [str(entry) for entry in entries]


def _string_list(value: Any, key: str, batch_size: int) -> list[str]:
    array = np.asarray(value)
    if array.ndim == 0:
        if batch_size != 1:
            raise VectorEnvReplayRecorderError(f"{key} must have one entry per B row")
        entries = [array.item()]
    else:
        if array.shape != (batch_size,):
            raise VectorEnvReplayRecorderError(
                f"{key} shape mismatch: expected {[batch_size]}, got {list(array.shape)}"
            )
        entries = list(array.tolist())
    return [str(entry) for entry in entries]


def _nonnegative_int_list(value: Any, key: str, batch_size: int) -> list[int]:
    array = np.asarray(value)
    if array.ndim == 0:
        if batch_size != 1:
            raise VectorEnvReplayRecorderError(f"{key} must have one entry per B row")
        array = array.reshape(1)
    if array.shape != (batch_size,):
        raise VectorEnvReplayRecorderError(
            f"{key} shape mismatch: expected {[batch_size]}, got {list(array.shape)}"
        )
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise VectorEnvReplayRecorderError(f"{key} must contain integers")
    if bool((array < 0).any()):
        raise VectorEnvReplayRecorderError(f"{key} must be non-negative")
    return [int(entry) for entry in array.tolist()]


def _reset_source_strings(value: Any, *, batch_size: int) -> str | Sequence[str]:
    if isinstance(value, str):
        if batch_size != 1:
            raise VectorEnvReplayRecorderError("reset_source must have one entry per B row")
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        if batch_size != 1:
            raise VectorEnvReplayRecorderError("reset_source must have one entry per B row")
        entries = [array.item()]
    else:
        if array.shape != (batch_size,):
            raise VectorEnvReplayRecorderError(
                f"reset_source shape mismatch: expected {[batch_size]}, got {list(array.shape)}"
            )
        entries = list(array.tolist())

    normalized: list[str] = []
    for entry in entries:
        if isinstance(entry, (str, np.str_)):
            normalized.append(str(entry))
            continue
        code = int(entry)
        if code not in vector_reset.RESET_SOURCE_CODE_NAMES:
            raise VectorEnvReplayRecorderError(f"unknown reset_source code {code}")
        normalized.append(vector_reset.RESET_SOURCE_CODE_NAMES[code])
    return normalized


def _manifest_string_list(value: Sequence[str], key: str) -> list[str]:
    if isinstance(value, str):
        raise VectorEnvReplayRecorderError(f"{key} must be a sequence of strings")
    normalized = [
        _manifest_non_empty_string(entry, f"{key}[{index}]")
        for index, entry in enumerate(value)
    ]
    if not normalized:
        raise VectorEnvReplayRecorderError(f"{key} must be non-empty")
    return normalized


def _manifest_non_empty_string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise VectorEnvReplayRecorderError(f"{key} must be a non-empty string")
    return value


def _manifest_int_list(value: Any, key: str) -> list[int]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise VectorEnvReplayRecorderError(f"{key} must be a 1-D integer array")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise VectorEnvReplayRecorderError(f"{key} must contain integers")
    return [int(entry) for entry in array.tolist()]


def _manifest_bool_list(value: Any, key: str, *, batch_size: int) -> list[bool]:
    array = np.asarray(value)
    if array.dtype != np.dtype(bool) or array.shape != (batch_size,):
        raise VectorEnvReplayRecorderError(
            f"{key} must be a bool array with shape {[batch_size]}"
        )
    return [bool(entry) for entry in array.tolist()]


__all__ = [
    "PRODUCER",
    "STRICT_VECTOR_FEATURE_FLAGS",
    "VECTOR_ENV_IMPL_ID",
    "VECTOR_ENV_REPLAY_MANIFEST_SCHEMA_ID",
    "VectorEnvReplayRecorder",
    "VectorEnvReplayRecorderError",
    "build_vector_env_replay_chunk_v0",
    "build_vector_env_replay_manifest_v0",
    "vector_step_batches_to_trainer_rows",
]
