"""Metadata-only replay packaging for public multiplayer env output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from curvyzero.env.vector_multiplayer_observation import (
    MULTIPLAYER_OBSERVATION_SCHEMA_HASH,
    MULTIPLAYER_OBSERVATION_SCHEMA_ID,
    MULTIPLAYER_OBSERVATION_SHAPE,
)
from curvyzero.training.multiplayer_replay_contract import (
    MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID,
    validate_multiplayer_replay_metadata_guard,
)
from curvyzero.training.replay_chunk_v0 import ReplayCompatibilityError
from curvyzero.training.replay_chunk_v0 import stable_contract_hash


MULTIPLAYER_REPLAY_RECORD_SCHEMA_ID = "curvyzero_multiplayer_metadata_replay_record/v0"
MULTIPLAYER_REPLAY_CHUNK_SCHEMA_ID = "curvyzero_multiplayer_metadata_replay_chunk/v0"
MULTIPLAYER_REPLAY_CONTRACT_ID = "curvyzero_multiplayer_metadata_replay/v0"
MULTIPLAYER_REPLAY_TERMINAL_BARRIER_POLICY = (
    "terminal_or_final_public_row_closes_metadata_sequence/v0"
)
MULTIPLAYER_REPLAY_BONUS_METADATA_AUDIT_CLAIM = (
    "public_bonus_state_metadata_audit_only_not_trainer_replay_or_full_replay_arrays/v0"
)
MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_CONTRACT_ID = (
    "curvyzero_multiplayer_scalar_observation_replay_shape/v0"
)
MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_ID = (
    "curvyzero_multiplayer_scalar_observation_replay_schema/v0"
)
MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_ID = (
    "curvyzero_multiplayer_scalar_observation_replay_record/v0"
)
MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_NON_CLAIMS = (
    "not_full_trainer_replay",
    "not_visual_replay",
    "not_full_source_fidelity_completion",
    "not_policy_targets",
    "not_search_targets",
    "not_value_targets",
)
MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_ARRAY_KEYS = (
    "observation",
    "action_mask",
    "lightzero_action_mask",
    "env_row_id",
    "ego_player_id",
    "row_mask",
    "source_shape",
)
MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA = {
    "schema_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_ID,
    "observation_schema_id": MULTIPLAYER_OBSERVATION_SCHEMA_ID,
    "public_metadata_source": MULTIPLAYER_REPLAY_RECORD_SCHEMA_ID,
    "records": "active scalar rows only; padded rows stay in row_mask arrays",
    "required_trace_fields": (
        "episode_id",
        "reset_episode_id",
        "reset_episode_id_policy",
        "round_id",
        "source_round_id",
        "source_round_id_policy",
        "episode_end_mode",
        "lifecycle_policy_id",
        "step_index",
        "tick_index",
        "elapsed_ms",
        "reset_seed",
        "reset_source",
        "random_tape_cursor",
        "random_tape_draw_count",
    ),
    "non_claims": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_NON_CLAIMS,
}
MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_HASH = stable_contract_hash(
    MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA
)
MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA = {
    "schema_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_ID,
    "contract_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_CONTRACT_ID,
    "record_schema_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_ID,
    "record_schema_hash": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_HASH,
    "observation_schema_id": MULTIPLAYER_OBSERVATION_SCHEMA_ID,
    "arrays": {
        "observation": ("float32", ("row", *MULTIPLAYER_OBSERVATION_SHAPE)),
        "action_mask": ("bool", ("row", "action")),
        "lightzero_action_mask": ("int8", ("row", "action")),
        "env_row_id": ("int32", ("row",)),
        "ego_player_id": ("int16", ("row",)),
        "row_mask": ("bool", ("row",)),
        "source_shape": ("int32", (2,)),
    },
    "scope": "scalar_observation_rows_plus_public_env_metadata_trace_only/v0",
    "non_claims": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_NON_CLAIMS,
}
MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_HASH = stable_contract_hash(
    MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA
)


@dataclass(frozen=True, slots=True)
class MultiplayerReplayChunkV0:
    """A metadata-only multiplayer replay chunk."""

    metadata: dict[str, Any]
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class MultiplayerScalarObservationReplayArtifactV0:
    """A replay-shaped scalar-observation artifact, not a trainer replay."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    records: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class MultiplayerMetadataReplayRecorder:
    """Collect metadata-only public multiplayer batches into one replay chunk.

    This mirrors the strict 1v1 terminal-barrier policy without claiming a
    trainer replay surface: once any public row is terminal or carries final-row
    metadata, later steps belong in a new recorder or after ``reset()``.
    """

    _records: list[dict[str, Any]] = field(default_factory=list)
    _batch_size: int | None = None
    _player_count: int | None = None
    _batch_count: int = 0
    _closed_by_terminal: bool = False

    @property
    def record_count(self) -> int:
        """Number of row records collected so far."""

        return len(self._records)

    @property
    def batch_count(self) -> int:
        """Number of public vector batches recorded so far."""

        return self._batch_count

    @property
    def batch_size(self) -> int | None:
        """Vector row count once the first batch has been recorded."""

        return self._batch_size

    @property
    def player_count(self) -> int | None:
        """Player count once the first batch has been recorded."""

        return self._player_count

    @property
    def closed_by_terminal(self) -> bool:
        """Whether a terminal/final row closed this sequence."""

        return self._closed_by_terminal

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """A copy of the collected metadata records."""

        return tuple(dict(record) for record in self._records)

    def reset(self) -> None:
        """Clear collected rows so the recorder can start a new sequence."""

        self._records.clear()
        self._batch_size = None
        self._player_count = None
        self._batch_count = 0
        self._closed_by_terminal = False

    def record_batch(
        self,
        batch: Any,
        *,
        opponent_policy_sidecar: Mapping[str, Any] | None = None,
        rng_history_ref: str | Sequence[str] | np.ndarray | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Append one public ``VectorMultiplayerEnv`` step batch."""

        if self._closed_by_terminal:
            raise ReplayCompatibilityError(
                "terminal/final multiplayer replay batch must be the final "
                "recorded batch before reset or a new recorder"
            )
        info = _public_batch_info(batch)
        batch_size = _batch_size(_public_batch_attr(batch, "done"), "done")
        player_count = _metadata_int(info, "player_count")
        action_count = _public_batch_action_count(batch, batch_size, player_count)
        if self._batch_size is None:
            self._batch_size = batch_size
            self._player_count = player_count
        elif batch_size != self._batch_size:
            raise ReplayCompatibilityError(
                f"B row mismatch: expected {self._batch_size}, got {batch_size}"
            )
        elif player_count != self._player_count:
            raise ReplayCompatibilityError(
                f"player_count mismatch: expected {self._player_count}, got {player_count}"
            )

        rng_refs = _rng_history_refs(rng_history_ref, batch_size=batch_size)
        opponent_sidecar = _opponent_policy_sidecar_rows(
            opponent_policy_sidecar,
            batch_size=batch_size,
            player_count=player_count,
            action_count=action_count,
        )
        batch_index = self._batch_count
        appended: list[dict[str, Any]] = []
        for row in range(batch_size):
            record = build_multiplayer_replay_record_from_public_env_output(
                info=info,
                action_mask=_public_batch_attr(batch, "action_mask"),
                reward=_public_batch_attr(batch, "reward"),
                done=_public_batch_attr(batch, "done"),
                terminated=_public_batch_attr(batch, "terminated"),
                truncated=_public_batch_attr(batch, "truncated"),
                row=row,
                rng_history_ref=rng_refs[row],
            )
            record["sequence_index"] = len(self._records) + len(appended)
            record["batch_index"] = batch_index
            record["batch_row"] = row
            if opponent_sidecar is not None:
                record["opponent_policy_sidecar"] = opponent_sidecar[row]
            validate_multiplayer_replay_metadata_guard(record)
            appended.append(record)

        self._records.extend(appended)
        self._batch_count += 1
        self._closed_by_terminal = _public_batch_has_terminal_or_final_rows(batch)
        return tuple(dict(record) for record in appended)

    def build_chunk(self) -> MultiplayerReplayChunkV0:
        """Build a metadata-only multiplayer replay chunk from recorded rows."""

        chunk = build_multiplayer_replay_chunk_v0(self._records)
        metadata = dict(chunk.metadata)
        metadata.update(
            {
                "recorded_batch_count": self._batch_count,
                "closed_by_terminal": self._closed_by_terminal,
                "terminal_barrier_policy": MULTIPLAYER_REPLAY_TERMINAL_BARRIER_POLICY,
                "opponent_policy_sidecar_present": any(
                    "opponent_policy_sidecar" in record for record in chunk.records
                ),
                "opponent_policy_sidecar_schema_id": (
                    MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID
                ),
                "trainer_replay_claim": False,
                "learned_observation_claim": False,
            }
        )
        return MultiplayerReplayChunkV0(metadata=metadata, records=chunk.records)


def build_multiplayer_replay_record_from_public_env_output(
    *,
    info: Mapping[str, Any],
    action_mask: Any,
    reward: Any,
    done: Any,
    terminated: Any,
    truncated: Any,
    row: int = 0,
    joint_action: Any | None = None,
    rng_history_ref: str | None = None,
) -> dict[str, Any]:
    """Package one public metadata-env row into an honest replay record."""

    if not isinstance(info, Mapping):
        raise ReplayCompatibilityError("info must be a metadata object")
    batch_size = _batch_size(done, "done")
    row = _row_index(row, batch_size=batch_size)
    player_count = _metadata_int(info, "player_count")
    player_ids = _sequence(_required(info, "player_ids"), "player_ids")
    source_player_ids = _sequence(_required(info, "source_player_ids"), "source_player_ids")
    if len(player_ids) != player_count or len(source_player_ids) != player_count:
        raise ReplayCompatibilityError("player id metadata must match player_count")

    sidecar = _required(info, "action_sidecar")
    if joint_action is None:
        if "joint_action" in info:
            joint_action = info["joint_action"]
        elif isinstance(sidecar, Mapping) and "player_action" in sidecar:
            joint_action = sidecar["player_action"]
        else:
            joint_action = np.full((batch_size, player_count), -1, dtype=np.int16)
    if rng_history_ref is None:
        rng_history_ref = _optional_info_ref(info, "rng_history_ref", row=row)
        if rng_history_ref is None:
            rng_history_ref = _optional_info_ref(
                info,
                "random_tape_history_ref",
                row=row,
            )

    present = _row_bool_vector(
        _required(info, "present"),
        "present",
        row=row,
        player_count=player_count,
    )
    alive = _row_bool_vector(
        _required(info, "alive"),
        "alive",
        row=row,
        player_count=player_count,
    )
    done_value = _row_bool_scalar(done, "done", row=row)
    action_mask_row = _row_bool_matrix(
        action_mask,
        "action_mask",
        row=row,
        player_count=player_count,
    )
    action_required = [
        bool(player_present and player_alive and not done_value)
        for player_present, player_alive in zip(present, alive, strict=True)
    ]
    death_player = _row_int_vector(
        _required(info, "death_player"),
        "death_player",
        row=row,
        player_count=player_count,
    )
    death_count = _row_int_scalar(_required(info, "death_count"), "death_count", row=row)
    record: dict[str, Any] = {
        "replay_contract_id": MULTIPLAYER_REPLAY_CONTRACT_ID,
        "record_schema_id": MULTIPLAYER_REPLAY_RECORD_SCHEMA_ID,
        "public_env_contract_id": _metadata_string(info, "public_env_contract_id"),
        "env_impl_id": _metadata_string(info, "env_impl_id"),
        "ruleset_id": _metadata_string(info, "ruleset_id"),
        "rules_hash": _metadata_string(info, "rules_hash"),
        "native_control_model_id": _metadata_string(info, "native_control_model_id"),
        "trainer_control_wrapper_id": _metadata_string(
            info,
            "trainer_control_wrapper_id",
        ),
        "decision_ms": _metadata_number(info, "decision_ms"),
        "lifecycle_policy_id": _metadata_string(info, "lifecycle_policy_id"),
        "episode_id": _json_scalar(_row_scalar(_required(info, "episode_id"), "episode_id", row=row)),
        "reset_episode_id": _json_scalar(
            _row_scalar(_required(info, "reset_episode_id"), "reset_episode_id", row=row)
        ),
        "reset_episode_id_policy": _metadata_string(info, "reset_episode_id_policy"),
        "round_id": _json_scalar(_row_scalar(_required(info, "round_id"), "round_id", row=row)),
        "source_round_id": _json_scalar(
            _row_scalar(_required(info, "source_round_id"), "source_round_id", row=row)
        ),
        "source_round_id_policy": _metadata_string(info, "source_round_id_policy"),
        "episode_end_mode": _metadata_string(info, "episode_end_mode"),
        "step_index": _row_int_scalar(_required(info, "step_index"), "step_index", row=row),
        "tick_index": _row_int_scalar(_required(info, "tick_index"), "tick_index", row=row),
        "elapsed_ms": _row_number_scalar(_required(info, "elapsed_ms"), "elapsed_ms", row=row),
        "player_count": player_count,
        "player_ids": [_json_scalar(value) for value in player_ids],
        "source_player_ids": [_json_scalar(value) for value in source_player_ids],
        "present": present,
        "alive": alive,
        "action_mask": action_mask_row,
        "joint_action": _row_int_vector(
            joint_action,
            "joint_action",
            row=row,
            player_count=player_count,
        ),
        "reward": _row_number_vector(
            reward,
            "reward",
            row=row,
            player_count=player_count,
        ),
        "done": done_value,
        "terminated": _row_bool_scalar(terminated, "terminated", row=row),
        "truncated": _row_bool_scalar(truncated, "truncated", row=row),
        "score": _row_int_vector(
            _required(info, "score"),
            "score",
            row=row,
            player_count=player_count,
        ),
        "round_score": _row_int_vector(
            _required(info, "round_score"),
            "round_score",
            row=row,
            player_count=player_count,
        ),
        "death_order": death_player[:death_count],
        "death_player": death_player,
        "death_count": death_count,
        "round_done": _row_bool_scalar(_required(info, "round_done"), "round_done", row=row),
        "match_done": _row_bool_scalar(_required(info, "match_done"), "match_done", row=row),
        "needs_reset": _row_bool_scalar(_required(info, "needs_reset"), "needs_reset", row=row),
        "terminal_reason": _json_scalar(
            _row_scalar(_required(info, "terminal_reason"), "terminal_reason", row=row)
        ),
        "truncation_reason": _json_scalar(
            _row_scalar(_required(info, "truncation_reason"), "truncation_reason", row=row)
        ),
        "winner": _row_int_scalar(_required(info, "winner"), "winner", row=row),
        "round_winner": _row_int_scalar(
            _required(info, "round_winner"),
            "round_winner",
            row=row,
        ),
        "match_winner": _row_int_scalar(
            _required(info, "match_winner"),
            "match_winner",
            row=row,
        ),
        "draw": _row_bool_scalar(_required(info, "draw"), "draw", row=row),
        "winner_ids": _row_indexed_int_list(_required(info, "winner_ids"), row=row),
        "round_winner_ids": _row_indexed_int_list(
            _required(info, "round_winner_ids"),
            row=row,
            name="round_winner_ids",
        ),
        "match_winner_ids": _row_indexed_int_list(
            _required(info, "match_winner_ids"),
            row=row,
            name="match_winner_ids",
        ),
        "reset_seed": _row_int_scalar(_required(info, "reset_seed"), "reset_seed", row=row),
        "reset_source": _json_scalar(
            _row_scalar(_required(info, "reset_source"), "reset_source", row=row)
        ),
        "random_tape_cursor": _row_int_scalar(
            _required(info, "random_tape_cursor"),
            "random_tape_cursor",
            row=row,
        ),
        "random_tape_draw_count": _row_int_scalar(
            _required(info, "random_tape_draw_count"),
            "random_tape_draw_count",
            row=row,
        ),
        "rng_history_ref": _optional_ref(rng_history_ref),
        "action_sidecar": _row_action_sidecar(
            sidecar,
            batch_size=batch_size,
            row=row,
            action_mask=action_mask_row,
            action_required=action_required,
        ),
        "observation_schema_id": _metadata_string(info, "observation_schema_id"),
        "observation_schema_hash": _metadata_string(info, "observation_schema_hash"),
        "action_space_id": _metadata_string(info, "action_space_id"),
        "action_space_hash": _metadata_string(info, "action_space_hash"),
        "reward_schema_id": _metadata_string(info, "reward_schema_id"),
        "reward_schema_hash": _metadata_string(info, "reward_schema_hash"),
        "final_observation_policy": _metadata_string(info, "final_observation_policy"),
        "metadata_only": _metadata_bool(info, "metadata_only"),
        "trainer_observation_claim": _metadata_bool(info, "trainer_observation_claim"),
        "trainer_replay_claim": False,
        "learned_observation_claim": False,
    }
    _copy_optional_rng_metadata(info, record, row=row, batch_size=batch_size)
    _copy_optional_public_bonus_metadata(
        info,
        record,
        row=row,
        batch_size=batch_size,
        player_count=player_count,
    )
    _copy_optional_terminal_final_metadata(info, record)
    validate_multiplayer_replay_metadata_guard(record)
    _validate_optional_public_bonus_record_metadata(record)
    return record


def build_multiplayer_replay_chunk_v0(
    records: Sequence[Mapping[str, Any]],
) -> MultiplayerReplayChunkV0:
    """Package validated metadata-only records into one chunk."""

    if not records:
        raise ReplayCompatibilityError("multiplayer replay chunk must contain records")
    normalized = tuple(dict(record) for record in records)
    for record in normalized:
        validate_multiplayer_replay_metadata_guard(record)
        _validate_optional_public_bonus_record_metadata(record)

    consistency_fields = (
        "public_env_contract_id",
        "env_impl_id",
        "ruleset_id",
        "rules_hash",
        "native_control_model_id",
        "trainer_control_wrapper_id",
        "decision_ms",
        "lifecycle_policy_id",
        "reset_episode_id_policy",
        "source_round_id_policy",
        "player_count",
        "observation_schema_id",
        "observation_schema_hash",
        "action_space_id",
        "action_space_hash",
        "reward_schema_id",
        "reward_schema_hash",
        "final_observation_policy",
    )
    first_values = {name: normalized[0][name] for name in consistency_fields}
    for record in normalized:
        for name in consistency_fields:
            if record[name] != first_values[name]:
                raise ReplayCompatibilityError(f"chunk records must share {name}")

    return MultiplayerReplayChunkV0(
        metadata={
            "replay_contract_id": MULTIPLAYER_REPLAY_CONTRACT_ID,
            "chunk_schema_id": MULTIPLAYER_REPLAY_CHUNK_SCHEMA_ID,
            "record_schema_id": MULTIPLAYER_REPLAY_RECORD_SCHEMA_ID,
            "record_count": len(normalized),
            **first_values,
            "metadata_only": True,
            "trainer_observation_claim": False,
            "trainer_replay_claim": False,
            "learned_observation_claim": False,
            "terminal_barrier_policy": MULTIPLAYER_REPLAY_TERMINAL_BARRIER_POLICY,
            "rng_provenance_policy": "reset_seed_source_cursor_draw_count_optional_ref/v0",
            "opponent_policy_sidecar_present": any(
                "opponent_policy_sidecar" in record for record in normalized
            ),
            "opponent_policy_sidecar_schema_id": (
                MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID
            ),
        },
        records=normalized,
    )


def build_multiplayer_scalar_observation_replay_artifact_v0(
    observation_rows: Any,
    *,
    public_info: Mapping[str, Any] | None = None,
    batch: Any | None = None,
    info: Mapping[str, Any] | None = None,
    public_action_mask: Any | None = None,
    reward: Any | None = None,
    done: Any | None = None,
    terminated: Any | None = None,
    truncated: Any | None = None,
    rng_history_ref: str | Sequence[str] | np.ndarray | None = None,
) -> MultiplayerScalarObservationReplayArtifactV0:
    """Package scalar ego rows with public-row replay metadata.

    This is intentionally replay-shaped, not trainer replay. It stores the
    scalar observation arrays and links each active scalar row back to the
    already-versioned public multiplayer metadata row.
    """

    if public_info is not None:
        if info is not None and info is not public_info:
            raise ReplayCompatibilityError("public_info and info must not conflict")
        info = public_info

    (
        info,
        public_action_mask,
        reward,
        done,
        terminated,
        truncated,
    ) = _public_batch_or_fields(
        batch=batch,
        info=info,
        public_action_mask=public_action_mask,
        reward=reward,
        done=done,
        terminated=terminated,
        truncated=truncated,
    )
    arrays = _scalar_observation_replay_arrays(observation_rows)
    source_batch_size, source_player_count = _source_shape_value(arrays["source_shape"])
    batch_size = _batch_size(done, "done")
    if source_batch_size != batch_size:
        raise ReplayCompatibilityError(
            f"source_shape batch mismatch: expected {batch_size}, got {source_batch_size}"
        )
    player_count = _metadata_int(info, "player_count")
    if source_player_count != player_count:
        raise ReplayCompatibilityError(
            "source_shape player count mismatch: "
            f"expected {player_count}, got {source_player_count}"
        )
    if player_count not in (3, 4):
        raise ReplayCompatibilityError("scalar observation replay rows require 3P or 4P")

    public_mask = np.asarray(public_action_mask)
    action_count = int(arrays["action_mask"].shape[1])
    if public_mask.shape != (batch_size, player_count, action_count):
        raise ReplayCompatibilityError("public_action_mask must have shape [B,P,A]")
    if public_mask.dtype != np.dtype(bool):
        raise ReplayCompatibilityError("public_action_mask must contain booleans")

    rng_refs = _rng_history_refs(rng_history_ref, batch_size=batch_size)
    public_record_cache: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for scalar_row_index in np.flatnonzero(arrays["row_mask"]):
        env_row = int(arrays["env_row_id"][int(scalar_row_index)])
        ego_player = int(arrays["ego_player_id"][int(scalar_row_index)])
        if env_row not in public_record_cache:
            public_record_cache[env_row] = (
                build_multiplayer_replay_record_from_public_env_output(
                    info=info,
                    action_mask=public_action_mask,
                    reward=reward,
                    done=done,
                    terminated=terminated,
                    truncated=truncated,
                    row=env_row,
                    rng_history_ref=rng_refs[env_row],
                )
            )
        public_record = public_record_cache[env_row]
        if not bool(public_record["present"][ego_player]) or not bool(
            public_record["alive"][ego_player]
        ):
            raise ReplayCompatibilityError(
                "active scalar observation row must reference a present+alive ego player"
            )
        records.append(
            _scalar_observation_replay_record(
                arrays=arrays,
                scalar_row_index=int(scalar_row_index),
                env_row=env_row,
                ego_player=ego_player,
                public_record=public_record,
            )
        )

    metadata = _scalar_observation_replay_metadata(
        info,
        arrays=arrays,
        records=records,
        player_count=player_count,
    )
    artifact = MultiplayerScalarObservationReplayArtifactV0(
        metadata=metadata,
        arrays=arrays,
        records=tuple(records),
    )
    return validate_multiplayer_scalar_observation_replay_artifact_v0(artifact)


def validate_multiplayer_scalar_observation_replay_artifact_v0(
    artifact: MultiplayerScalarObservationReplayArtifactV0,
) -> MultiplayerScalarObservationReplayArtifactV0:
    """Validate the narrow scalar-row replay-shaped artifact."""

    if not isinstance(artifact, MultiplayerScalarObservationReplayArtifactV0):
        raise ReplayCompatibilityError(
            "scalar observation replay artifact must be "
            "MultiplayerScalarObservationReplayArtifactV0"
        )
    metadata = artifact.metadata
    if not isinstance(metadata, Mapping):
        raise ReplayCompatibilityError("scalar observation replay metadata must be a mapping")
    arrays = _validate_scalar_observation_replay_array_mapping(artifact.arrays)
    records = tuple(dict(record) for record in artifact.records)

    for key, expected in {
        "replay_contract_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_CONTRACT_ID,
        "replay_schema_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_ID,
        "replay_schema_hash": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_HASH,
        "record_schema_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_ID,
        "record_schema_hash": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_HASH,
        "observation_schema_id": MULTIPLAYER_OBSERVATION_SCHEMA_ID,
        "observation_schema_hash": MULTIPLAYER_OBSERVATION_SCHEMA_HASH,
    }.items():
        if metadata.get(key) != expected:
            raise ReplayCompatibilityError(f"{key} mismatch")

    row_count = int(arrays["row_mask"].shape[0])
    active_indices = [int(index) for index in np.flatnonzero(arrays["row_mask"])]
    if metadata.get("row_count") != row_count:
        raise ReplayCompatibilityError("row_count metadata mismatch")
    if metadata.get("active_row_count") != len(active_indices):
        raise ReplayCompatibilityError("active_row_count metadata mismatch")
    if metadata.get("record_count") != len(records):
        raise ReplayCompatibilityError("record_count metadata mismatch")
    if len(records) != len(active_indices):
        raise ReplayCompatibilityError("records must cover active scalar rows only")
    if metadata.get("source_shape") != arrays["source_shape"].astype(int).tolist():
        raise ReplayCompatibilityError("source_shape metadata mismatch")
    if metadata.get("array_shapes") != _array_shapes(arrays):
        raise ReplayCompatibilityError("array_shapes metadata mismatch")
    if metadata.get("array_dtypes") != _array_dtypes(arrays):
        raise ReplayCompatibilityError("array_dtypes metadata mismatch")

    if metadata.get("contains_scalar_observation_rows") is not True:
        raise ReplayCompatibilityError("contains_scalar_observation_rows must be true")
    for key in (
        "trainer_ready_env_claim",
        "trainer_replay_claim",
        "visual_replay_claim",
        "source_fidelity_completion_claim",
        "policy_targets_claim",
        "search_targets_claim",
        "value_targets_claim",
    ):
        if metadata.get(key) is not False:
            raise ReplayCompatibilityError(f"{key} must be false")
    if tuple(metadata.get("non_claims", ())) != MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_NON_CLAIMS:
        raise ReplayCompatibilityError("non_claims metadata mismatch")

    for expected_index, record in zip(active_indices, records, strict=True):
        _validate_scalar_observation_replay_record(
            record,
            arrays=arrays,
            expected_scalar_row_index=expected_index,
        )
    return MultiplayerScalarObservationReplayArtifactV0(
        metadata=dict(metadata),
        arrays=arrays,
        records=records,
    )


def _public_batch_attr(batch: Any, name: str) -> Any:
    if not hasattr(batch, name):
        raise ReplayCompatibilityError(f"public multiplayer batch missing {name}")
    return getattr(batch, name)


def _public_batch_info(batch: Any) -> Mapping[str, Any]:
    info = _public_batch_attr(batch, "info")
    if not isinstance(info, Mapping):
        raise ReplayCompatibilityError("public multiplayer batch info must be metadata")
    return info


def _public_batch_or_fields(
    *,
    batch: Any | None,
    info: Mapping[str, Any] | None,
    public_action_mask: Any | None,
    reward: Any | None,
    done: Any | None,
    terminated: Any | None,
    truncated: Any | None,
) -> tuple[Mapping[str, Any], Any, Any, Any, Any, Any]:
    if batch is not None:
        info = _public_batch_info(batch) if info is None else info
        public_action_mask = (
            _public_batch_attr(batch, "action_mask")
            if public_action_mask is None
            else public_action_mask
        )
        reward = _public_batch_attr(batch, "reward") if reward is None else reward
        done = _public_batch_attr(batch, "done") if done is None else done
        terminated = (
            _public_batch_attr(batch, "terminated") if terminated is None else terminated
        )
        truncated = _public_batch_attr(batch, "truncated") if truncated is None else truncated

    missing = [
        name
        for name, value in (
            ("info", info),
            ("public_action_mask", public_action_mask),
            ("reward", reward),
            ("done", done),
            ("terminated", terminated),
            ("truncated", truncated),
        )
        if value is None
    ]
    if missing:
        raise ReplayCompatibilityError(
            "missing scalar observation replay inputs: " + ", ".join(missing)
        )
    if not isinstance(info, Mapping):
        raise ReplayCompatibilityError("info must be a metadata object")
    return info, public_action_mask, reward, done, terminated, truncated


def _scalar_observation_replay_arrays(observation_rows: Any) -> dict[str, np.ndarray]:
    schema_id = getattr(observation_rows, "schema_id", None)
    if schema_id != MULTIPLAYER_OBSERVATION_SCHEMA_ID:
        raise ReplayCompatibilityError(
            "scalar observation rows must use "
            f"{MULTIPLAYER_OBSERVATION_SCHEMA_ID!r}"
        )
    schema_hash = getattr(observation_rows, "schema_hash", None)
    if schema_hash != MULTIPLAYER_OBSERVATION_SCHEMA_HASH:
        raise ReplayCompatibilityError("scalar observation schema_hash mismatch")
    arrays = {
        "observation": np.asarray(_rows_attr(observation_rows, "observation")),
        "action_mask": np.asarray(_rows_attr(observation_rows, "action_mask")),
        "lightzero_action_mask": np.asarray(
            _rows_attr(observation_rows, "lightzero_action_mask")
        ),
        "env_row_id": np.asarray(_rows_attr(observation_rows, "env_row_id")),
        "ego_player_id": np.asarray(_rows_attr(observation_rows, "ego_player_id")),
        "row_mask": np.asarray(_rows_attr(observation_rows, "row_mask")),
        "source_shape": np.asarray(_rows_attr(observation_rows, "source_shape")),
    }
    return _validate_scalar_observation_replay_array_mapping(arrays)


def _validate_scalar_observation_replay_array_mapping(
    arrays: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    missing = sorted(set(MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_ARRAY_KEYS) - set(arrays))
    if missing:
        raise ReplayCompatibilityError(
            "missing scalar observation replay arrays: " + ", ".join(missing)
        )
    unexpected = sorted(set(arrays) - set(MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_ARRAY_KEYS))
    if unexpected:
        raise ReplayCompatibilityError(
            "unexpected scalar observation replay arrays: " + ", ".join(unexpected)
        )

    observation = np.asarray(arrays["observation"])
    if observation.dtype != np.dtype("float32"):
        raise ReplayCompatibilityError("observation must be float32")
    if observation.ndim != 2 or observation.shape[1:] != MULTIPLAYER_OBSERVATION_SHAPE:
        raise ReplayCompatibilityError("observation must have shape [R,27]")
    if not bool(np.isfinite(observation).all()):
        raise ReplayCompatibilityError("observation must contain finite values")
    row_count = int(observation.shape[0])

    action_mask = _scalar_array(
        arrays["action_mask"],
        "action_mask",
        dtype=np.dtype(bool),
        shape=(row_count, None),
    )
    if action_mask.shape[1] <= 0:
        raise ReplayCompatibilityError("action_mask must include at least one action")
    lightzero_action_mask = _scalar_array(
        arrays["lightzero_action_mask"],
        "lightzero_action_mask",
        dtype=np.dtype("int8"),
        shape=action_mask.shape,
    )
    if not np.array_equal(lightzero_action_mask, action_mask.astype(np.int8)):
        raise ReplayCompatibilityError("lightzero_action_mask must equal action_mask as int8")

    env_row_id = _integer_vector(arrays["env_row_id"], "env_row_id", row_count=row_count)
    ego_player_id = _integer_vector(
        arrays["ego_player_id"],
        "ego_player_id",
        row_count=row_count,
    )
    row_mask = _scalar_array(
        arrays["row_mask"],
        "row_mask",
        dtype=np.dtype(bool),
        shape=(row_count,),
    )
    source_shape = np.asarray(arrays["source_shape"])
    source_batch_size, source_player_count = _source_shape_value(source_shape)

    active = row_mask.astype(bool, copy=False)
    if bool(active.any()):
        if bool(((env_row_id[active] < 0) | (env_row_id[active] >= source_batch_size)).any()):
            raise ReplayCompatibilityError("active env_row_id values must be in source_shape")
        if bool(
            ((ego_player_id[active] < 0) | (ego_player_id[active] >= source_player_count)).any()
        ):
            raise ReplayCompatibilityError("active ego_player_id values must be in source_shape")
    padded = ~active
    if bool(padded.any()):
        if bool((env_row_id[padded] != -1).any()) or bool(
            (ego_player_id[padded] != -1).any()
        ):
            raise ReplayCompatibilityError("padded scalar rows must use -1 ids")
        if bool(action_mask[padded].any()) or bool(lightzero_action_mask[padded].any()):
            raise ReplayCompatibilityError("padded scalar rows must have empty action masks")
        if bool(np.any(observation[padded] != 0.0)):
            raise ReplayCompatibilityError("padded scalar observations must be zero-filled")

    return {
        "observation": observation.copy(),
        "action_mask": action_mask.copy(),
        "lightzero_action_mask": lightzero_action_mask.copy(),
        "env_row_id": env_row_id.astype(np.int32, copy=True),
        "ego_player_id": ego_player_id.astype(np.int16, copy=True),
        "row_mask": row_mask.copy(),
        "source_shape": np.asarray([source_batch_size, source_player_count], dtype=np.int32),
    }


def _scalar_observation_replay_metadata(
    info: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray],
    records: Sequence[Mapping[str, Any]],
    player_count: int,
) -> dict[str, Any]:
    row_count = int(arrays["row_mask"].shape[0])
    active_row_count = int(arrays["row_mask"].sum())
    return {
        "replay_contract_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_CONTRACT_ID,
        "replay_schema_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_ID,
        "replay_schema_hash": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_HASH,
        "record_schema_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_ID,
        "record_schema_hash": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_HASH,
        "record_count": len(records),
        "row_count": row_count,
        "active_row_count": active_row_count,
        "source_shape": arrays["source_shape"].astype(int).tolist(),
        "player_count": player_count,
        "array_shapes": _array_shapes(arrays),
        "array_dtypes": _array_dtypes(arrays),
        "observation_schema_id": MULTIPLAYER_OBSERVATION_SCHEMA_ID,
        "observation_schema_hash": MULTIPLAYER_OBSERVATION_SCHEMA_HASH,
        "public_env_observation_schema_id": _metadata_string(info, "observation_schema_id"),
        "public_env_observation_schema_hash": _metadata_string(
            info,
            "observation_schema_hash",
        ),
        "public_env_contract_id": _metadata_string(info, "public_env_contract_id"),
        "env_impl_id": _metadata_string(info, "env_impl_id"),
        "ruleset_id": _metadata_string(info, "ruleset_id"),
        "rules_hash": _metadata_string(info, "rules_hash"),
        "native_control_model_id": _metadata_string(info, "native_control_model_id"),
        "trainer_control_wrapper_id": _metadata_string(
            info,
            "trainer_control_wrapper_id",
        ),
        "decision_ms": _metadata_number(info, "decision_ms"),
        "lifecycle_policy_id": _metadata_string(info, "lifecycle_policy_id"),
        "reset_episode_id_policy": _metadata_string(info, "reset_episode_id_policy"),
        "source_round_id_policy": _metadata_string(info, "source_round_id_policy"),
        "episode_end_mode": _metadata_string(info, "episode_end_mode"),
        "action_space_id": _metadata_string(info, "action_space_id"),
        "action_space_hash": _metadata_string(info, "action_space_hash"),
        "reward_schema_id": _metadata_string(info, "reward_schema_id"),
        "reward_schema_hash": _metadata_string(info, "reward_schema_hash"),
        "final_observation_policy": _metadata_string(info, "final_observation_policy"),
        "round_id_policy": (
            "preserved_from_public_env_rows_reset_starts_at_1_"
            "next_round_warmdown_increments_match_end_keeps_final/v0"
        ),
        "public_env_metadata_only": _metadata_bool(info, "metadata_only"),
        "public_env_trainer_observation_claim": _metadata_bool(
            info,
            "trainer_observation_claim",
        ),
        "contains_scalar_observation_rows": True,
        "contains_scalar_learned_observation_schema": True,
        "replay_scope": "scalar_observation_rows_plus_public_env_metadata_trace_only/v0",
        "trainer_ready_env_claim": False,
        "trainer_replay_claim": False,
        "visual_replay_claim": False,
        "source_fidelity_completion_claim": False,
        "policy_targets_claim": False,
        "search_targets_claim": False,
        "value_targets_claim": False,
        "non_claims": list(MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_NON_CLAIMS),
    }


def _scalar_observation_replay_record(
    *,
    arrays: Mapping[str, np.ndarray],
    scalar_row_index: int,
    env_row: int,
    ego_player: int,
    public_record: Mapping[str, Any],
) -> dict[str, Any]:
    reward = list(public_record["reward"])
    record = {
        "record_schema_id": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_ID,
        "record_schema_hash": MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_HASH,
        "scalar_row_index": scalar_row_index,
        "row_mask": True,
        "env_row_id": env_row,
        "ego_player_id": ego_player,
        "source_shape": arrays["source_shape"].astype(int).tolist(),
        "observation_schema_id": MULTIPLAYER_OBSERVATION_SCHEMA_ID,
        "observation_schema_hash": MULTIPLAYER_OBSERVATION_SCHEMA_HASH,
        "observation_shape": [int(arrays["observation"].shape[1])],
        "observation_dtype": str(arrays["observation"].dtype),
        "action_mask": [bool(flag) for flag in arrays["action_mask"][scalar_row_index]],
        "lightzero_action_mask": [
            int(flag) for flag in arrays["lightzero_action_mask"][scalar_row_index]
        ],
        "public_env_record_schema_id": public_record["record_schema_id"],
        "public_env_replay_contract_id": public_record["replay_contract_id"],
        "public_env_record": dict(public_record),
        "episode_id": public_record["episode_id"],
        "reset_episode_id": public_record["reset_episode_id"],
        "reset_episode_id_policy": public_record["reset_episode_id_policy"],
        "round_id": public_record["round_id"],
        "source_round_id": public_record["source_round_id"],
        "source_round_id_policy": public_record["source_round_id_policy"],
        "episode_end_mode": public_record["episode_end_mode"],
        "lifecycle_policy_id": public_record["lifecycle_policy_id"],
        "step_index": public_record["step_index"],
        "tick_index": public_record["tick_index"],
        "elapsed_ms": public_record["elapsed_ms"],
        "reset_seed": public_record["reset_seed"],
        "reset_source": public_record["reset_source"],
        "random_tape_cursor": public_record["random_tape_cursor"],
        "random_tape_draw_count": public_record["random_tape_draw_count"],
        "player_count": public_record["player_count"],
        "player_ids": list(public_record["player_ids"]),
        "source_player_ids": list(public_record["source_player_ids"]),
        "present": list(public_record["present"]),
        "alive": list(public_record["alive"]),
        "reward": reward,
        "ego_reward": float(reward[ego_player]),
        "done": public_record["done"],
        "terminated": public_record["terminated"],
        "truncated": public_record["truncated"],
        "needs_reset": public_record["needs_reset"],
        "round_done": public_record["round_done"],
        "match_done": public_record["match_done"],
        "terminal_reason": public_record["terminal_reason"],
        "truncation_reason": public_record["truncation_reason"],
        "score": list(public_record["score"]),
        "round_score": list(public_record["round_score"]),
        "trainer_replay_claim": False,
        "visual_replay_claim": False,
        "source_fidelity_completion_claim": False,
        "policy_targets_claim": False,
        "search_targets_claim": False,
        "value_targets_claim": False,
        "non_claims": list(MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_NON_CLAIMS),
    }
    for key in (
        "random_tape_source",
        "random_tape_length",
        "rng_impl_id",
        "source_fixture_ref",
    ):
        if key in public_record:
            record[key] = public_record[key]
    return record


def _validate_scalar_observation_replay_record(
    record: Mapping[str, Any],
    *,
    arrays: Mapping[str, np.ndarray],
    expected_scalar_row_index: int,
) -> None:
    if record.get("record_schema_id") != MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_ID:
        raise ReplayCompatibilityError("scalar record_schema_id mismatch")
    if (
        record.get("record_schema_hash")
        != MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_HASH
    ):
        raise ReplayCompatibilityError("scalar record_schema_hash mismatch")
    if record.get("scalar_row_index") != expected_scalar_row_index:
        raise ReplayCompatibilityError("scalar row records must be ordered by active rows")
    if record.get("row_mask") is not True:
        raise ReplayCompatibilityError("scalar row record row_mask must be true")
    if record.get("env_row_id") != int(arrays["env_row_id"][expected_scalar_row_index]):
        raise ReplayCompatibilityError("scalar record env_row_id mismatch")
    if record.get("ego_player_id") != int(arrays["ego_player_id"][expected_scalar_row_index]):
        raise ReplayCompatibilityError("scalar record ego_player_id mismatch")
    if record.get("source_shape") != arrays["source_shape"].astype(int).tolist():
        raise ReplayCompatibilityError("scalar record source_shape mismatch")
    if record.get("observation_schema_id") != MULTIPLAYER_OBSERVATION_SCHEMA_ID:
        raise ReplayCompatibilityError("scalar record observation_schema_id mismatch")
    if record.get("observation_schema_hash") != MULTIPLAYER_OBSERVATION_SCHEMA_HASH:
        raise ReplayCompatibilityError("scalar record observation_schema_hash mismatch")
    if record.get("action_mask") != [
        bool(flag) for flag in arrays["action_mask"][expected_scalar_row_index]
    ]:
        raise ReplayCompatibilityError("scalar record action_mask mismatch")
    if record.get("lightzero_action_mask") != [
        int(flag) for flag in arrays["lightzero_action_mask"][expected_scalar_row_index]
    ]:
        raise ReplayCompatibilityError("scalar record lightzero_action_mask mismatch")

    public_record = record.get("public_env_record")
    if not isinstance(public_record, Mapping):
        raise ReplayCompatibilityError("scalar record missing public_env_record")
    validate_multiplayer_replay_metadata_guard(public_record)
    ego_player = int(record["ego_player_id"])
    if not bool(public_record["present"][ego_player]) or not bool(
        public_record["alive"][ego_player]
    ):
        raise ReplayCompatibilityError("public_env_record must keep scalar ego present+alive")
    if record.get("round_id") != public_record["round_id"]:
        raise ReplayCompatibilityError("scalar record must preserve public round_id")
    for key in (
        "trainer_replay_claim",
        "visual_replay_claim",
        "source_fidelity_completion_claim",
        "policy_targets_claim",
        "search_targets_claim",
        "value_targets_claim",
    ):
        if record.get(key) is not False:
            raise ReplayCompatibilityError(f"scalar record {key} must be false")


def _rows_attr(rows: Any, name: str) -> Any:
    if not hasattr(rows, name):
        raise ReplayCompatibilityError(f"scalar observation rows missing {name}")
    return getattr(rows, name)


def _scalar_array(
    value: Any,
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise ReplayCompatibilityError(f"{name} must have dtype {dtype}")
    if array.ndim != len(shape):
        raise ReplayCompatibilityError(f"{name} rank mismatch")
    for actual, expected in zip(array.shape, shape, strict=True):
        if expected is not None and actual != expected:
            raise ReplayCompatibilityError(f"{name} shape mismatch")
    return array


def _integer_vector(value: Any, name: str, *, row_count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape != (row_count,):
        raise ReplayCompatibilityError(f"{name} must have shape [R]")
    if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.integer):
        raise ReplayCompatibilityError(f"{name} must contain integers")
    return array.astype(np.int64, copy=False)


def _source_shape_value(value: Any) -> tuple[int, int]:
    array = np.asarray(value)
    if array.shape != (2,):
        raise ReplayCompatibilityError("source_shape must have shape [2]")
    if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.integer):
        raise ReplayCompatibilityError("source_shape must contain integers")
    batch_size, player_count = (int(item) for item in array.tolist())
    if batch_size <= 0:
        raise ReplayCompatibilityError("source_shape batch size must be positive")
    if player_count not in (3, 4):
        raise ReplayCompatibilityError("source_shape player count must be 3 or 4")
    return batch_size, player_count


def _array_shapes(arrays: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    return {key: [int(axis) for axis in np.asarray(value).shape] for key, value in arrays.items()}


def _array_dtypes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {key: str(np.asarray(value).dtype) for key, value in arrays.items()}


def _public_batch_action_count(batch: Any, batch_size: int, player_count: int) -> int:
    action_mask = np.asarray(_public_batch_attr(batch, "action_mask"))
    if action_mask.ndim != 3 or action_mask.shape[:2] != (batch_size, player_count):
        raise ReplayCompatibilityError("action_mask must have shape [B,P,A]")
    return int(action_mask.shape[2])


def _public_batch_has_terminal_or_final_rows(batch: Any) -> bool:
    for name in ("done", "terminated", "truncated"):
        value = np.asarray(_public_batch_attr(batch, name))
        if value.ndim == 1 and bool(value.astype(bool).any()):
            return True
    if _public_batch_attr(batch, "final_observation") is not None:
        return True
    if _public_batch_attr(batch, "final_reward") is not None:
        return True

    info = _public_batch_info(batch)
    for key in ("final_observation", "final_reward_map"):
        if info.get(key) is not None:
            return True
    for key in ("final_observation_row_mask", "final_reward_row_mask"):
        if key in info and bool(np.asarray(info[key], dtype=bool).any()):
            return True
    return False


def _rng_history_refs(
    value: str | Sequence[str] | np.ndarray | None,
    *,
    batch_size: int,
) -> list[str | None]:
    if value is None:
        return [None for _ in range(batch_size)]
    if isinstance(value, str):
        return [_optional_ref(value) for _ in range(batch_size)]
    if isinstance(value, np.ndarray):
        values = value.tolist()
    else:
        values = list(value)
    if len(values) != batch_size:
        raise ReplayCompatibilityError("rng_history_ref length must match batch size")
    return [_optional_ref(item) for item in values]


def _opponent_policy_sidecar_rows(
    value: Mapping[str, Any] | None,
    *,
    batch_size: int,
    player_count: int,
    action_count: int,
) -> tuple[dict[str, Any], ...] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ReplayCompatibilityError("opponent_policy_sidecar must be a metadata object")
    policy_id = _metadata_string(value, "policy_id")
    policy_version = _metadata_string(value, "policy_version")
    seeds = _opponent_sidecar_seeds(
        _required(value, "seed"),
        batch_size=batch_size,
    )
    actions = _opponent_sidecar_actions(
        _required(value, "actions"),
        batch_size=batch_size,
        player_count=player_count,
        action_count=action_count,
    )
    return tuple(
        {
            "schema_id": MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID,
            "policy_id": policy_id,
            "policy_version": policy_version,
            "seed": int(seeds[row]),
            "actions": [int(action) for action in actions[row].tolist()],
            "metadata_only": True,
            "trainer_replay_claim": False,
            "learned_observation_claim": False,
        }
        for row in range(batch_size)
    )


def _opponent_sidecar_seeds(value: Any, *, batch_size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        if isinstance(array.item(), bool) or not np.issubdtype(array.dtype, np.integer):
            raise ReplayCompatibilityError(
                "opponent_policy_sidecar seed must contain integers"
            )
        seeds = np.full(batch_size, int(array.item()), dtype=np.int64)
    elif array.ndim == 1 and array.shape[0] == batch_size:
        if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.integer):
            raise ReplayCompatibilityError(
                "opponent_policy_sidecar seed must contain integers"
            )
        seeds = array.astype(np.int64, copy=False)
    else:
        raise ReplayCompatibilityError("opponent_policy_sidecar seed must be scalar or [B]")
    if bool((seeds < 0).any()):
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar seed must be non-negative"
        )
    return seeds


def _opponent_sidecar_actions(
    value: Any,
    *,
    batch_size: int,
    player_count: int,
    action_count: int,
) -> np.ndarray:
    actions = np.asarray(value)
    if actions.shape != (batch_size, player_count):
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar actions must have shape [B,P]"
        )
    if actions.dtype == np.dtype(bool) or not np.issubdtype(actions.dtype, np.integer):
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar actions must contain integers"
        )
    actions = actions.astype(np.int64, copy=False)
    invalid = (actions < -1) | (actions >= action_count)
    if bool(invalid.any()):
        raise ReplayCompatibilityError(
            "opponent_policy_sidecar actions must be -1 or valid action ids"
        )
    return actions


def _row_action_sidecar(
    value: Any,
    *,
    batch_size: int,
    row: int,
    action_mask: Sequence[Sequence[bool]],
    action_required: Sequence[bool],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayCompatibilityError("action_sidecar must be a metadata object")
    sidecar = {
        str(key): _jsonable(_maybe_row_slice(item, batch_size=batch_size, row=row))
        for key, item in value.items()
    }
    sidecar["player_action_mask"] = [
        [bool(flag) for flag in player_mask] for player_mask in action_mask
    ]
    sidecar["action_required"] = [bool(flag) for flag in action_required]
    return sidecar


def _maybe_row_slice(value: Any, *, batch_size: int, row: int) -> Any:
    array = np.asarray(value) if isinstance(value, np.ndarray) else None
    if array is not None and array.ndim > 0 and array.shape[0] == batch_size:
        return array[row]
    return value


def _row_bool_matrix(value: Any, name: str, *, row: int, player_count: int) -> list[list[bool]]:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[1] != player_count:
        raise ReplayCompatibilityError(f"{name} must have shape [B,P,A]")
    if array.dtype != np.dtype(bool):
        raise ReplayCompatibilityError(f"{name} must contain booleans")
    return [[bool(flag) for flag in player] for player in array[row].tolist()]


def _row_player_array(value: Any, name: str, *, row: int, player_count: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != player_count:
        raise ReplayCompatibilityError(f"{name} must have shape [B,P]")
    return array[row]


def _row_bool_vector(value: Any, name: str, *, row: int, player_count: int) -> list[bool]:
    array = _row_player_array(value, name, row=row, player_count=player_count)
    if array.dtype != np.dtype(bool):
        raise ReplayCompatibilityError(f"{name} must contain booleans")
    return [bool(item) for item in array.tolist()]


def _row_int_vector(value: Any, name: str, *, row: int, player_count: int) -> list[int]:
    array = _row_player_array(value, name, row=row, player_count=player_count)
    if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.integer):
        raise ReplayCompatibilityError(f"{name} must contain integers")
    return [int(item) for item in array.tolist()]


def _row_number_vector(value: Any, name: str, *, row: int, player_count: int) -> list[float]:
    array = _row_player_array(value, name, row=row, player_count=player_count)
    if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.number):
        raise ReplayCompatibilityError(f"{name} must contain numbers")
    if not bool(np.isfinite(array.astype(np.float64)).all()):
        raise ReplayCompatibilityError(f"{name} must contain finite values")
    return [float(item) for item in array.tolist()]


def _row_scalar(value: Any, name: str, *, row: int) -> Any:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ReplayCompatibilityError(f"{name} must have shape [B]")
    item = array[row]
    return item.item() if isinstance(item, np.generic) else item


def _row_bool_scalar(value: Any, name: str, *, row: int) -> bool:
    item = _row_scalar(value, name, row=row)
    if not isinstance(item, bool | np.bool_):
        raise ReplayCompatibilityError(f"{name} must contain booleans")
    return bool(item)


def _row_int_scalar(value: Any, name: str, *, row: int) -> int:
    item = _row_scalar(value, name, row=row)
    if isinstance(item, bool | np.bool_) or not isinstance(item, int | np.integer):
        raise ReplayCompatibilityError(f"{name} must contain integers")
    return int(item)


def _row_number_scalar(value: Any, name: str, *, row: int) -> float:
    item = _row_scalar(value, name, row=row)
    if isinstance(item, bool | np.bool_) or not isinstance(item, int | float | np.number):
        raise ReplayCompatibilityError(f"{name} must contain numbers")
    number = float(item)
    if not np.isfinite(number):
        raise ReplayCompatibilityError(f"{name} must contain finite values")
    return number


def _row_indexed_int_list(value: Any, *, row: int, name: str = "winner_ids") -> list[int]:
    if not isinstance(value, list | tuple):
        raise ReplayCompatibilityError(f"{name} must be row-indexed")
    row_value = value[row]
    if not isinstance(row_value, list | tuple):
        raise ReplayCompatibilityError(f"{name} rows must be sequences")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in row_value):
        raise ReplayCompatibilityError(f"{name} rows must contain integers")
    return [int(item) for item in row_value]


def _batch_size(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] <= 0:
        raise ReplayCompatibilityError(f"{name} must have shape [B]")
    return int(array.shape[0])


def _row_index(value: int, *, batch_size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayCompatibilityError("row must be an integer")
    if value < 0 or value >= batch_size:
        raise ReplayCompatibilityError("row is out of range")
    return int(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise ReplayCompatibilityError(f"{name} must be a sequence")
    return list(value)


def _required(info: Mapping[str, Any], key: str) -> Any:
    if key not in info:
        raise ReplayCompatibilityError(f"{key} is required")
    return info[key]


def _metadata_bool(info: Mapping[str, Any], key: str) -> bool:
    value = _required(info, key)
    if not isinstance(value, bool):
        raise ReplayCompatibilityError(f"{key} must be a boolean")
    return bool(value)


def _metadata_int(info: Mapping[str, Any], key: str) -> int:
    value = _required(info, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayCompatibilityError(f"{key} must be an integer")
    return int(value)


def _metadata_string(info: Mapping[str, Any], key: str) -> str:
    value = _required(info, key)
    if not isinstance(value, str) or not value:
        raise ReplayCompatibilityError(f"{key} must be a non-empty string")
    return value


def _metadata_number(info: Mapping[str, Any], key: str) -> float:
    value = _required(info, key)
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ReplayCompatibilityError(f"{key} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise ReplayCompatibilityError(f"{key} must be finite")
    return number


def _optional_ref(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ReplayCompatibilityError("rng_history_ref must be null or a non-empty string")
    return value


def _optional_info_ref(info: Mapping[str, Any], key: str, *, row: int) -> str | None:
    if key not in info:
        return None
    value = _row_or_metadata_scalar(info[key], key, row=row)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ReplayCompatibilityError(f"{key} must be null or a non-empty string")
    return value


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    raise ReplayCompatibilityError("metadata value must be JSON-scalar compatible")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    raise ReplayCompatibilityError("metadata value must be JSON-compatible")


def _row_metadata_sidecar(
    value: Any,
    name: str,
    *,
    row: int,
    batch_size: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayCompatibilityError(f"{name} must be a metadata object")
    return {
        str(key): _jsonable(_maybe_row_slice(item, batch_size=batch_size, row=row))
        for key, item in value.items()
    }


def _copy_optional_rng_metadata(
    info: Mapping[str, Any],
    record: dict[str, Any],
    *,
    row: int,
    batch_size: int,
) -> None:
    if "random_tape_source" in info:
        record["random_tape_source"] = _json_scalar(
            _row_or_metadata_scalar(info["random_tape_source"], "random_tape_source", row=row)
        )
    if "random_tape_length" in info:
        record["random_tape_length"] = _int_row_or_metadata_scalar(
            info["random_tape_length"],
            "random_tape_length",
            row=row,
        )
    if "rng_impl_id" in info:
        record["rng_impl_id"] = _json_scalar(
            _row_or_metadata_scalar(info["rng_impl_id"], "rng_impl_id", row=row)
        )
    if "source_fixture_ref" in info:
        record["source_fixture_ref"] = _json_scalar(
            _row_or_metadata_scalar(info["source_fixture_ref"], "source_fixture_ref", row=row)
        )
    if "reset_provenance" in info:
        record["reset_provenance"] = _row_metadata_sidecar(
            info["reset_provenance"],
            "reset_provenance",
            row=row,
            batch_size=batch_size,
        )


def _copy_optional_public_bonus_metadata(
    info: Mapping[str, Any],
    record: dict[str, Any],
    *,
    row: int,
    batch_size: int,
    player_count: int,
) -> None:
    if "base_public_env_contract_id" in info:
        record["base_public_env_contract_id"] = _metadata_string(
            info,
            "base_public_env_contract_id",
        )
    if "seeded_bonus_public_env_contract_id" in info:
        record["seeded_bonus_public_env_contract_id"] = _metadata_string(
            info,
            "seeded_bonus_public_env_contract_id",
        )
    if "borderless" in info:
        borderless = _bool_row_or_metadata_scalar(info["borderless"], "borderless", row=row)
        record["borderless"] = borderless
    if "bonus_support" not in info:
        unsupported_metadata = _bonus_unsupported_metadata(
            info,
            row=row,
            batch_size=batch_size,
        )
        if unsupported_metadata:
            record["bonus_support_unsupported_metadata"] = unsupported_metadata
        return

    support = info["bonus_support"]
    if not isinstance(support, Mapping):
        raise ReplayCompatibilityError("bonus_support must be a metadata object")
    support_mode = _metadata_string(support, "mode")
    if "bonus_support_mode" in info:
        mode = _metadata_string(info, "bonus_support_mode")
        if mode != support_mode:
            raise ReplayCompatibilityError("bonus_support_mode must match bonus_support.mode")
    else:
        mode = support_mode
    row_mode = _string_row_or_metadata_scalar(
        info.get("bonus_support_mode_by_row", _required(support, "mode_by_row")),
        "bonus_support_mode_by_row",
        row=row,
    )
    if "enabled_by_row" in support:
        enabled = _bool_row_or_metadata_scalar(
            support["enabled_by_row"],
            "bonus_support.enabled_by_row",
            row=row,
        )
    elif "enabled" in support:
        enabled = _bool_row_or_metadata_scalar(
            support["enabled"],
            "bonus_support.enabled",
            row=row,
        )
    else:
        raise ReplayCompatibilityError("bonus_support.enabled_by_row is required")
    active_count = _nonnegative_int_row_or_metadata_scalar(
        _required(support, "active_count"),
        "bonus_support.active_count",
        row=row,
    )
    stack_count = _row_nonnegative_int_vector_or_metadata_vector(
        _required(support, "stack_count"),
        "bonus_support.stack_count",
        row=row,
        player_count=player_count,
    )
    supported_types, supported_seeded_types = _bonus_supported_types(support)
    natural_source = (
        support["natural_bonus_spawn"]
        if "natural_bonus_spawn" in support
        else info.get("natural_bonus_spawn")
    )
    if natural_source is None:
        raise ReplayCompatibilityError("bonus_support.natural_bonus_spawn is required")
    natural_bonus_spawn = _bool_row_or_metadata_scalar(
        natural_source,
        "bonus_support.natural_bonus_spawn",
        row=row,
    )
    borderless = record.get("borderless")
    if borderless is None and "borderless" in support:
        borderless = _bool_row_or_metadata_scalar(
            support["borderless"],
            "bonus_support.borderless",
            row=row,
        )

    bonus_support_record = {
        "policy_id": _metadata_string(support, "policy_id"),
        "claim": _metadata_string(support, "claim"),
        "metadata_audit_claim": MULTIPLAYER_REPLAY_BONUS_METADATA_AUDIT_CLAIM,
        "mode": mode,
        "mode_by_row": row_mode,
        "enabled": enabled,
        "active_count": active_count,
        "stack_count": stack_count,
        "supported_types": supported_types,
        "natural_bonus_spawn": natural_bonus_spawn,
        "public_env_contract_id": record["public_env_contract_id"],
        "ruleset_id": record["ruleset_id"],
        "metadata_only": True,
        "trainer_replay_claim": False,
        "full_replay_arrays_claim": False,
    }
    if supported_seeded_types is not None:
        bonus_support_record["supported_seeded_bonus_types"] = supported_seeded_types
    if borderless is not None:
        bonus_support_record["borderless"] = bool(borderless)
        record["borderless"] = bool(borderless)

    unsupported_metadata = _bonus_unsupported_metadata(
        info,
        row=row,
        batch_size=batch_size,
    )
    unsupported_metadata.update(
        _bonus_unsupported_metadata(
            support,
            row=row,
            batch_size=batch_size,
        )
    )
    if unsupported_metadata:
        bonus_support_record.update(unsupported_metadata)
        bonus_support_record["unsupported_metadata"] = dict(unsupported_metadata)
        record["bonus_support_unsupported_metadata"] = dict(unsupported_metadata)

    record["bonus_support"] = bonus_support_record
    record["bonus_metadata_audit_claim"] = MULTIPLAYER_REPLAY_BONUS_METADATA_AUDIT_CLAIM
    record["bonus_support_mode"] = mode
    record["bonus_support_mode_by_row"] = row_mode
    record["bonus_support_enabled"] = enabled
    record["bonus_support_active_count"] = active_count
    record["bonus_support_stack_count"] = stack_count
    record["bonus_support_supported_types"] = supported_types
    if supported_seeded_types is not None:
        record["bonus_support_supported_seeded_bonus_types"] = supported_seeded_types
    record["bonus_support_natural_bonus_spawn"] = natural_bonus_spawn
    record["natural_bonus_spawn"] = natural_bonus_spawn
    if borderless is not None:
        record["bonus_support_borderless"] = bool(borderless)


def _validate_optional_public_bonus_record_metadata(metadata: Mapping[str, Any]) -> None:
    player_count = _metadata_int(metadata, "player_count")
    if "borderless" in metadata and not isinstance(metadata["borderless"], bool):
        raise ReplayCompatibilityError("borderless must be a boolean")
    if "bonus_support_borderless" in metadata:
        value = metadata["bonus_support_borderless"]
        if not isinstance(value, bool):
            raise ReplayCompatibilityError("bonus_support_borderless must be a boolean")
        if "borderless" in metadata and value != metadata["borderless"]:
            raise ReplayCompatibilityError("bonus_support_borderless must match borderless")

    if "bonus_support_unsupported_metadata" in metadata:
        _validate_bonus_unsupported_metadata_mapping(
            metadata["bonus_support_unsupported_metadata"],
            "bonus_support_unsupported_metadata",
        )
    if "bonus_support" not in metadata:
        return

    support = metadata["bonus_support"]
    if not isinstance(support, Mapping):
        raise ReplayCompatibilityError("bonus_support must be a metadata object")
    for key in ("policy_id", "claim", "mode", "mode_by_row"):
        _metadata_string(support, key)
    if support.get("metadata_audit_claim") != MULTIPLAYER_REPLAY_BONUS_METADATA_AUDIT_CLAIM:
        raise ReplayCompatibilityError("bonus_support metadata_audit_claim mismatch")
    if support.get("metadata_only") is not True:
        raise ReplayCompatibilityError("bonus_support metadata_only must be true")
    if support.get("trainer_replay_claim") is not False:
        raise ReplayCompatibilityError("bonus_support trainer_replay_claim must be false")
    if support.get("full_replay_arrays_claim") is not False:
        raise ReplayCompatibilityError("bonus_support full_replay_arrays_claim must be false")
    if support.get("enabled") is not metadata.get("bonus_support_enabled"):
        raise ReplayCompatibilityError("bonus_support enabled mismatch")
    if support.get("active_count") != metadata.get("bonus_support_active_count"):
        raise ReplayCompatibilityError("bonus_support active_count mismatch")
    if support.get("stack_count") != metadata.get("bonus_support_stack_count"):
        raise ReplayCompatibilityError("bonus_support stack_count mismatch")
    if support.get("natural_bonus_spawn") is not metadata.get("natural_bonus_spawn"):
        raise ReplayCompatibilityError("bonus_support natural_bonus_spawn mismatch")
    if support.get("natural_bonus_spawn") is not metadata.get(
        "bonus_support_natural_bonus_spawn",
    ):
        raise ReplayCompatibilityError("bonus_support_natural_bonus_spawn mismatch")

    _metadata_bool(support, "enabled")
    active_count = _metadata_int(support, "active_count")
    if active_count < 0:
        raise ReplayCompatibilityError("bonus_support active_count must be non-negative")
    stack_count = _expect_nonnegative_int_sequence(
        support,
        "stack_count",
        length=player_count,
    )
    if any(value < 0 for value in stack_count):
        raise ReplayCompatibilityError("bonus_support stack_count must be non-negative")
    _string_sequence(support.get("supported_types"), "bonus_support.supported_types")
    if "supported_seeded_bonus_types" in support:
        seeded_types = _string_sequence(
            support["supported_seeded_bonus_types"],
            "bonus_support.supported_seeded_bonus_types",
        )
        if seeded_types != metadata.get("bonus_support_supported_seeded_bonus_types"):
            raise ReplayCompatibilityError("bonus_support_supported_seeded_bonus_types mismatch")
    if support.get("supported_types") != metadata.get("bonus_support_supported_types"):
        raise ReplayCompatibilityError("bonus_support_supported_types mismatch")
    _metadata_bool(support, "natural_bonus_spawn")
    if "borderless" in support:
        if not isinstance(support["borderless"], bool):
            raise ReplayCompatibilityError("bonus_support borderless must be a boolean")
        if support["borderless"] != metadata.get("borderless"):
            raise ReplayCompatibilityError("bonus_support borderless mismatch")

    if "unsupported_metadata" in support:
        _validate_bonus_unsupported_metadata_mapping(
            support["unsupported_metadata"],
            "bonus_support.unsupported_metadata",
        )
    for key, value in support.items():
        if _is_bonus_unsupported_metadata_key(str(key)):
            _validate_bonus_unsupported_metadata_value(
                value,
                f"bonus_support.{key}",
                boolean_only=_bonus_unsupported_key_requires_bool(str(key)),
            )
    for key in ("bonus_active", "bonus_type"):
        if key in support:
            raise ReplayCompatibilityError(f"bonus_support {key} full arrays are not replayed")


def _copy_optional_terminal_final_metadata(
    info: Mapping[str, Any],
    record: dict[str, Any],
) -> None:
    for key in (
        "terminal_rows",
        "final_observation_rows",
        "final_observation_row_mask",
        "final_observation_row_policy",
        "final_reward_rows",
        "final_reward_row_mask",
        "final_reward_row_policy",
    ):
        if key in info:
            record[key] = _jsonable(info[key])


def _row_or_metadata_scalar(value: Any, name: str, *, row: int) -> Any:
    if isinstance(value, str) or value is None:
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        return item.item() if isinstance(item, np.generic) else item
    return _row_scalar(value, name, row=row)


def _int_row_or_metadata_scalar(value: Any, name: str, *, row: int) -> int:
    item = _row_or_metadata_scalar(value, name, row=row)
    if isinstance(item, bool | np.bool_) or not isinstance(item, int | np.integer):
        raise ReplayCompatibilityError(f"{name} must contain integers")
    return int(item)


def _nonnegative_int_row_or_metadata_scalar(value: Any, name: str, *, row: int) -> int:
    item = _int_row_or_metadata_scalar(value, name, row=row)
    if item < 0:
        raise ReplayCompatibilityError(f"{name} must be non-negative")
    return item


def _bool_row_or_metadata_scalar(value: Any, name: str, *, row: int) -> bool:
    item = _row_or_metadata_scalar(value, name, row=row)
    if not isinstance(item, bool | np.bool_):
        raise ReplayCompatibilityError(f"{name} must contain booleans")
    return bool(item)


def _string_row_or_metadata_scalar(value: Any, name: str, *, row: int) -> str:
    item = _row_or_metadata_scalar(value, name, row=row)
    if not isinstance(item, str) or not item:
        raise ReplayCompatibilityError(f"{name} must contain non-empty strings")
    return item


def _string_sequence(value: Any, name: str) -> list[str]:
    values = _sequence(value, name)
    if any(not isinstance(item, str) or not item for item in values):
        raise ReplayCompatibilityError(f"{name} must contain non-empty strings")
    return [str(item) for item in values]


def _expect_nonnegative_int_sequence(
    metadata: Mapping[str, Any],
    key: str,
    *,
    length: int,
) -> list[int]:
    values = _sequence(_required(metadata, key), key)
    if len(values) != length:
        raise ReplayCompatibilityError(f"{key} length must match player_count")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise ReplayCompatibilityError(f"{key} must contain integers")
    if any(int(item) < 0 for item in values):
        raise ReplayCompatibilityError(f"{key} must be non-negative")
    return [int(item) for item in values]


def _row_nonnegative_int_vector_or_metadata_vector(
    value: Any,
    name: str,
    *,
    row: int,
    player_count: int,
) -> list[int]:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == player_count:
        vector = array[row]
    elif array.ndim == 1 and array.shape[0] == player_count:
        vector = array
    else:
        raise ReplayCompatibilityError(f"{name} must have shape [B,P] or [P]")
    if vector.dtype == np.dtype(bool) or not np.issubdtype(vector.dtype, np.integer):
        raise ReplayCompatibilityError(f"{name} must contain integers")
    values = [int(item) for item in vector.tolist()]
    if any(item < 0 for item in values):
        raise ReplayCompatibilityError(f"{name} must be non-negative")
    return values


def _bonus_supported_types(support: Mapping[str, Any]) -> tuple[list[str], list[str] | None]:
    seeded_types: list[str] | None = None
    if "supported_seeded_bonus_types" in support:
        seeded_types = _string_sequence(
            support["supported_seeded_bonus_types"],
            "bonus_support.supported_seeded_bonus_types",
        )
    if "supported_bonus_types" in support:
        supported_types = _string_sequence(
            support["supported_bonus_types"],
            "bonus_support.supported_bonus_types",
        )
    elif seeded_types is not None:
        supported_types = list(seeded_types)
    else:
        raise ReplayCompatibilityError("bonus_support supported types are required")
    return supported_types, seeded_types


def _bonus_unsupported_metadata(
    metadata: Mapping[str, Any],
    *,
    row: int,
    batch_size: int,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in metadata.items():
        name = str(key)
        if not _is_bonus_unsupported_metadata_key(name):
            continue
        row_value = _jsonable(_maybe_row_slice(value, batch_size=batch_size, row=row))
        _validate_bonus_unsupported_metadata_value(
            row_value,
            name,
            boolean_only=_bonus_unsupported_key_requires_bool(name),
        )
        values[name] = row_value
    return values


def _is_bonus_unsupported_metadata_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return "unsupported" in normalized and (
        "bonus" in normalized or "effect" in normalized
    )


def _bonus_unsupported_key_requires_bool(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return (
        normalized.startswith(("has_", "is_", "contains_"))
        or normalized.endswith(("_flag", "_flags", "_present", "_enabled"))
    )


def _validate_bonus_unsupported_metadata_mapping(value: Any, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ReplayCompatibilityError(f"{name} must be a metadata object")
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ReplayCompatibilityError(f"{name} keys must be non-empty strings")
        if not _is_bonus_unsupported_metadata_key(key):
            raise ReplayCompatibilityError(f"{name} key {key!r} is not unsupported bonus metadata")
        _validate_bonus_unsupported_metadata_value(
            item,
            f"{name}.{key}",
            boolean_only=_bonus_unsupported_key_requires_bool(key),
        )


def _validate_bonus_unsupported_metadata_value(
    value: Any,
    name: str,
    *,
    boolean_only: bool,
) -> None:
    if isinstance(value, np.generic):
        value = value.item()
    if boolean_only:
        if isinstance(value, bool):
            return
        if isinstance(value, list | tuple):
            for index, item in enumerate(value):
                _validate_bonus_unsupported_metadata_value(
                    item,
                    f"{name}[{index}]",
                    boolean_only=True,
                )
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise ReplayCompatibilityError(
                        f"{name} keys must be non-empty strings"
                    )
                _validate_bonus_unsupported_metadata_value(
                    item,
                    f"{name}.{key}",
                    boolean_only=True,
                )
            return
        raise ReplayCompatibilityError(f"{name} must contain booleans")

    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if not value:
            raise ReplayCompatibilityError(f"{name} must contain non-empty strings")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_bonus_unsupported_metadata_value(
                item,
                f"{name}[{index}]",
                boolean_only=False,
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ReplayCompatibilityError(f"{name} keys must be non-empty strings")
            _validate_bonus_unsupported_metadata_value(
                item,
                f"{name}.{key}",
                boolean_only=False,
            )
        return
    raise ReplayCompatibilityError(
        f"{name} must contain boolean/string unsupported bonus metadata"
    )


__all__ = [
    "MULTIPLAYER_REPLAY_CHUNK_SCHEMA_ID",
    "MULTIPLAYER_REPLAY_CONTRACT_ID",
    "MULTIPLAYER_REPLAY_BONUS_METADATA_AUDIT_CLAIM",
    "MULTIPLAYER_REPLAY_OPPONENT_POLICY_SIDECAR_SCHEMA_ID",
    "MULTIPLAYER_REPLAY_RECORD_SCHEMA_ID",
    "MULTIPLAYER_REPLAY_TERMINAL_BARRIER_POLICY",
    "MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_CONTRACT_ID",
    "MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_NON_CLAIMS",
    "MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_HASH",
    "MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_RECORD_SCHEMA_ID",
    "MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_HASH",
    "MULTIPLAYER_SCALAR_OBSERVATION_REPLAY_SCHEMA_ID",
    "MultiplayerMetadataReplayRecorder",
    "MultiplayerReplayChunkV0",
    "MultiplayerScalarObservationReplayArtifactV0",
    "build_multiplayer_replay_chunk_v0",
    "build_multiplayer_replay_record_from_public_env_output",
    "build_multiplayer_scalar_observation_replay_artifact_v0",
    "validate_multiplayer_scalar_observation_replay_artifact_v0",
]
