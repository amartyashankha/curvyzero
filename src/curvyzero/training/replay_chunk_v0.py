"""Replay chunk v0 contract for 1v1/no-bonus training.

This module defines one narrow local ``.npz`` payload: arrays plus a JSON
metadata record. It validates compatibility and shape contracts, but it is not
a learner input pipeline or storage layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPLAY_CONTRACT_ID = "curvyzero_1v1_no_bonus_replay_chunk/v0"
REPLAY_METADATA_SCHEMA_ID = "curvyzero_1v1_no_bonus_replay_chunk_metadata/v0"
REPLAY_SCHEMA_ID = "curvyzero_1v1_no_bonus_replay_schema/v0"

PLAYER_COUNT = 2

REPLAY_ARRAY_KEYS = (
    "observation",
    "reward",
    "action",
    "action_weights",
    "root_value",
    "done",
    "terminated",
    "truncated",
    "episode_id",
    "reset_seed",
    "reset_source",
    "final_observation",
    "final_reward_map",
)

REPLAY_NUMERIC_ARRAY_DTYPES = {
    "observation": np.dtype("float32"),
    "reward": np.dtype("float32"),
    "action": np.dtype("int16"),
    "action_weights": np.dtype("float32"),
    "root_value": np.dtype("float32"),
    "done": np.dtype("bool"),
    "terminated": np.dtype("bool"),
    "truncated": np.dtype("bool"),
    "reset_seed": np.dtype("int64"),
    "final_observation": np.dtype("float32"),
    "final_reward_map": np.dtype("float32"),
}

REPLAY_STRING_ARRAY_KEYS = ("episode_id", "reset_source")

REQUIRED_COMPATIBILITY_HASH_KEYS = (
    "replay_schema_hash",
    "rules_hash",
    "observation_schema_hash",
    "action_space_hash",
    "reward_schema_hash",
)

OPTIONAL_COMPATIBILITY_KEYS = (
    "ruleset_id",
    "observation_schema_id",
    "action_space_id",
    "reward_schema_id",
)

REPLAY_DONE_SEMANTICS = "done == terminated | truncated"
FINAL_REWARD_MAP_SEMANTICS = "player_indexed_final_reward_no_bonus"

_REQUIRED_METADATA_KEYS = {
    "metadata_schema_id",
    "replay_contract_id",
    "replay_schema_id",
    "replay_schema_hash",
    "rules_hash",
    "observation_schema_hash",
    "action_space_hash",
    "reward_schema_hash",
    "chunk_steps",
    "batch_size",
    "player_count",
    "obs_dim",
    "action_count",
    "array_shapes",
    "array_dtypes",
    "done_semantics",
    "final_reward_map_semantics",
}

_OPTIONAL_METADATA_STRING_KEYS = {
    "ruleset_id",
    "observation_schema_id",
    "action_space_id",
    "reward_schema_id",
    "producer",
    "created_at",
}

_OPTIONAL_METADATA_MAPPING_KEYS = {
    "vector_env_row_metadata",
}


class ReplayCompatibilityError(ValueError):
    """Raised when a replay chunk does not satisfy the v0 compatibility contract."""


@dataclass(frozen=True, slots=True)
class ReplayChunkV0:
    """A validated replay chunk and its metadata."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def stable_contract_hash(payload: Mapping[str, Any]) -> str:
    """Return the short deterministic hash used by replay compatibility metadata."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def replay_schema_hash(*, obs_dim: int, action_count: int) -> str:
    """Hash the v0 replay field schema for this observation/action width."""

    return stable_contract_hash(
        {
            "schema_id": REPLAY_SCHEMA_ID,
            "metadata_schema_id": REPLAY_METADATA_SCHEMA_ID,
            "contract_id": REPLAY_CONTRACT_ID,
            "done_semantics": REPLAY_DONE_SEMANTICS,
            "final_reward_map_semantics": FINAL_REWARD_MAP_SEMANTICS,
            "fields": _replay_field_spec(obs_dim=obs_dim, action_count=action_count),
        }
    )


def build_replay_chunk_v0_metadata(
    arrays: Mapping[str, Any],
    *,
    rules_hash: str,
    observation_schema_hash: str,
    action_space_hash: str,
    reward_schema_hash: str,
    ruleset_id: str | None = None,
    observation_schema_id: str | None = None,
    action_space_id: str | None = None,
    reward_schema_id: str | None = None,
    producer: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build metadata for one validated replay chunk."""

    normalized = _validate_arrays(arrays)
    chunk_steps, batch_size, _player_count, obs_dim = normalized["observation"].shape
    action_count = normalized["action_weights"].shape[-1]
    metadata: dict[str, Any] = {
        "metadata_schema_id": REPLAY_METADATA_SCHEMA_ID,
        "replay_contract_id": REPLAY_CONTRACT_ID,
        "replay_schema_id": REPLAY_SCHEMA_ID,
        "replay_schema_hash": replay_schema_hash(
            obs_dim=int(obs_dim),
            action_count=int(action_count),
        ),
        "rules_hash": _non_empty_string(rules_hash, "rules_hash"),
        "observation_schema_hash": _non_empty_string(
            observation_schema_hash,
            "observation_schema_hash",
        ),
        "action_space_hash": _non_empty_string(action_space_hash, "action_space_hash"),
        "reward_schema_hash": _non_empty_string(
            reward_schema_hash,
            "reward_schema_hash",
        ),
        "chunk_steps": int(chunk_steps),
        "batch_size": int(batch_size),
        "player_count": PLAYER_COUNT,
        "obs_dim": int(obs_dim),
        "action_count": int(action_count),
        "array_shapes": _array_shapes(normalized),
        "array_dtypes": _array_dtypes(normalized),
        "done_semantics": REPLAY_DONE_SEMANTICS,
        "final_reward_map_semantics": FINAL_REWARD_MAP_SEMANTICS,
    }
    for key, value in {
        "ruleset_id": ruleset_id,
        "observation_schema_id": observation_schema_id,
        "action_space_id": action_space_id,
        "reward_schema_id": reward_schema_id,
        "producer": producer,
        "created_at": created_at,
    }.items():
        if value is not None:
            metadata[key] = _non_empty_string(value, key)

    validate_replay_chunk_v0(arrays=normalized, metadata=metadata)
    return metadata


def write_replay_chunk_v0(
    path: str | Path,
    *,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Write one replay chunk as ``.npz`` arrays plus JSON metadata."""

    normalized = validate_replay_chunk_v0(arrays=arrays, metadata=metadata)
    metadata_json = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez(handle, metadata=np.array(metadata_json), **normalized)


def read_replay_chunk_v0(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> ReplayChunkV0:
    """Read and validate one replay chunk."""

    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as payload:
        names = set(payload.files)
        required = {*REPLAY_ARRAY_KEYS, "metadata"}
        missing = sorted(required - names)
        if missing:
            raise ReplayCompatibilityError(
                f"{input_path} is missing replay payload keys: {', '.join(missing)}"
            )
        unexpected = sorted(names - required)
        if unexpected:
            raise ReplayCompatibilityError(
                f"{input_path} has unexpected replay payload keys: {', '.join(unexpected)}"
            )

        metadata = _load_metadata(payload["metadata"])
        arrays = {key: np.asarray(payload[key]) for key in REPLAY_ARRAY_KEYS}

    normalized = validate_replay_chunk_v0(arrays=arrays, metadata=metadata)
    if expected_metadata is not None:
        validate_replay_chunk_v0_compatibility(
            actual_metadata=metadata,
            expected_metadata=expected_metadata,
        )
    return ReplayChunkV0(metadata=dict(metadata), arrays=normalized)


def validate_replay_chunk_v0(
    *,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Validate arrays and metadata for the replay chunk v0 contract."""

    normalized = _validate_arrays(arrays)
    _validate_metadata(metadata=metadata, arrays=normalized)
    return normalized


def validate_replay_chunk_v0_compatibility(
    *,
    actual_metadata: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
) -> None:
    """Reject a replay chunk whose compatibility hashes do not match expectations."""

    for key in REQUIRED_COMPATIBILITY_HASH_KEYS:
        actual = _metadata_string(actual_metadata, key)
        expected = _metadata_string(expected_metadata, key)
        if actual != expected:
            raise ReplayCompatibilityError(
                f"{key} mismatch: expected {expected!r}, got {actual!r}"
            )

    for key in OPTIONAL_COMPATIBILITY_KEYS:
        if key not in expected_metadata:
            continue
        actual = _metadata_string(actual_metadata, key)
        expected = _metadata_string(expected_metadata, key)
        if actual != expected:
            raise ReplayCompatibilityError(
                f"{key} mismatch: expected {expected!r}, got {actual!r}"
            )


def compatibility_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Extract the compatibility record a reader should compare."""

    keys = (*REQUIRED_COMPATIBILITY_HASH_KEYS, *OPTIONAL_COMPATIBILITY_KEYS)
    return {key: _metadata_string(metadata, key) for key in keys if key in metadata}


def _validate_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    missing = [key for key in REPLAY_ARRAY_KEYS if key not in arrays]
    if missing:
        raise ReplayCompatibilityError(f"missing replay arrays: {', '.join(missing)}")
    unexpected = sorted(set(arrays) - set(REPLAY_ARRAY_KEYS))
    if unexpected:
        raise ReplayCompatibilityError(f"unexpected replay arrays: {', '.join(unexpected)}")

    normalized = {key: np.asarray(arrays[key]) for key in REPLAY_ARRAY_KEYS}
    for key, expected_dtype in REPLAY_NUMERIC_ARRAY_DTYPES.items():
        if normalized[key].dtype != expected_dtype:
            raise ReplayCompatibilityError(
                f"{key} dtype mismatch: expected {expected_dtype}, got {normalized[key].dtype}"
            )
    for key in REPLAY_STRING_ARRAY_KEYS:
        if normalized[key].dtype.kind != "U":
            raise ReplayCompatibilityError(
                f"{key} dtype mismatch: expected unicode string, got {normalized[key].dtype}"
            )

    observation = normalized["observation"]
    if observation.ndim != 4:
        raise ReplayCompatibilityError(
            f"observation must have shape [T,B,2,D], got {list(observation.shape)}"
        )
    chunk_steps, batch_size, player_count, obs_dim = observation.shape
    if min(chunk_steps, batch_size, player_count, obs_dim) <= 0:
        raise ReplayCompatibilityError(
            f"observation dimensions must be positive, got {list(observation.shape)}"
        )
    if player_count != PLAYER_COUNT:
        raise ReplayCompatibilityError(
            f"player_count mismatch: expected {PLAYER_COUNT}, got {player_count}"
        )

    _expect_shape(normalized["reward"], "reward", (chunk_steps, batch_size, PLAYER_COUNT))
    _expect_shape(normalized["action"], "action", (chunk_steps, batch_size, PLAYER_COUNT))
    _expect_shape(
        normalized["root_value"],
        "root_value",
        (chunk_steps, batch_size, PLAYER_COUNT),
    )
    _expect_shape(normalized["done"], "done", (chunk_steps, batch_size))
    _expect_shape(normalized["terminated"], "terminated", (chunk_steps, batch_size))
    _expect_shape(normalized["truncated"], "truncated", (chunk_steps, batch_size))
    _expect_shape(normalized["episode_id"], "episode_id", (batch_size,))
    _expect_shape(normalized["reset_seed"], "reset_seed", (batch_size,))
    _expect_shape(normalized["reset_source"], "reset_source", (batch_size,))
    _expect_shape(
        normalized["final_observation"],
        "final_observation",
        (batch_size, PLAYER_COUNT, obs_dim),
    )
    _expect_shape(normalized["final_reward_map"], "final_reward_map", (batch_size, PLAYER_COUNT))

    action_weights = normalized["action_weights"]
    if action_weights.ndim != 4:
        raise ReplayCompatibilityError(
            f"action_weights must have shape [T,B,2,A], got {list(action_weights.shape)}"
        )
    if action_weights.shape[:3] != (chunk_steps, batch_size, PLAYER_COUNT):
        raise ReplayCompatibilityError(
            "action_weights prefix shape mismatch: expected "
            f"{[chunk_steps, batch_size, PLAYER_COUNT]}, got {list(action_weights.shape[:3])}"
        )
    action_count = action_weights.shape[-1]
    if action_count <= 0:
        raise ReplayCompatibilityError("action_weights action count must be positive")

    _expect_finite(normalized["observation"], "observation")
    _expect_finite(normalized["reward"], "reward")
    _expect_finite(normalized["action_weights"], "action_weights")
    _expect_finite(normalized["root_value"], "root_value")
    _expect_finite(normalized["final_observation"], "final_observation")
    _expect_finite(normalized["final_reward_map"], "final_reward_map")
    if np.any(normalized["action_weights"] < 0.0):
        raise ReplayCompatibilityError("action_weights must be non-negative")
    if np.any(normalized["reset_seed"] < 0):
        raise ReplayCompatibilityError("reset_seed must be non-negative")
    if np.any(normalized["action"] < 0) or np.any(normalized["action"] >= action_count):
        raise ReplayCompatibilityError(
            f"action values must be in [0, {action_count}), got invalid action"
        )
    for key in REPLAY_STRING_ARRAY_KEYS:
        if np.any(np.char.str_len(normalized[key]) == 0):
            raise ReplayCompatibilityError(f"{key} entries must be non-empty strings")

    expected_done = np.logical_or(normalized["terminated"], normalized["truncated"])
    if not np.array_equal(normalized["done"], expected_done):
        raise ReplayCompatibilityError("done must equal terminated | truncated")

    return normalized


def _validate_metadata(
    *,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    missing = sorted(_REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        raise ReplayCompatibilityError(f"missing replay metadata: {', '.join(missing)}")
    allowed = (
        _REQUIRED_METADATA_KEYS
        | _OPTIONAL_METADATA_STRING_KEYS
        | _OPTIONAL_METADATA_MAPPING_KEYS
    )
    unexpected = sorted(set(metadata) - allowed)
    if unexpected:
        raise ReplayCompatibilityError(
            f"unexpected replay metadata: {', '.join(unexpected)}"
        )

    metadata_schema_id = _metadata_string(metadata, "metadata_schema_id")
    if metadata_schema_id != REPLAY_METADATA_SCHEMA_ID:
        raise ReplayCompatibilityError(
            "metadata_schema_id mismatch: expected "
            f"{REPLAY_METADATA_SCHEMA_ID!r}, got {metadata_schema_id!r}"
        )

    contract_id = _metadata_string(metadata, "replay_contract_id")
    if contract_id != REPLAY_CONTRACT_ID:
        raise ReplayCompatibilityError(
            f"replay_contract_id mismatch: expected {REPLAY_CONTRACT_ID!r}, got {contract_id!r}"
        )

    schema_id = _metadata_string(metadata, "replay_schema_id")
    if schema_id != REPLAY_SCHEMA_ID:
        raise ReplayCompatibilityError(
            f"replay_schema_id mismatch: expected {REPLAY_SCHEMA_ID!r}, got {schema_id!r}"
        )

    obs_dim = int(arrays["observation"].shape[-1])
    action_count = int(arrays["action_weights"].shape[-1])
    expected_schema_hash = replay_schema_hash(obs_dim=obs_dim, action_count=action_count)
    actual_schema_hash = _metadata_string(metadata, "replay_schema_hash")
    if actual_schema_hash != expected_schema_hash:
        raise ReplayCompatibilityError(
            "replay_schema_hash mismatch: expected "
            f"{expected_schema_hash!r}, got {actual_schema_hash!r}"
        )

    for key in (
        "rules_hash",
        "observation_schema_hash",
        "action_space_hash",
        "reward_schema_hash",
    ):
        _metadata_string(metadata, key)

    for key in _OPTIONAL_METADATA_STRING_KEYS:
        if key in metadata:
            _metadata_string(metadata, key)
    for key in _OPTIONAL_METADATA_MAPPING_KEYS:
        if key in metadata:
            _metadata_mapping(metadata, key)

    expected_shapes = _array_shapes(arrays)
    expected_dtypes = _array_dtypes(arrays)
    actual_shapes = _metadata_mapping(metadata, "array_shapes")
    actual_dtypes = _metadata_mapping(metadata, "array_dtypes")
    if dict(actual_shapes) != expected_shapes:
        raise ReplayCompatibilityError(
            f"array_shapes mismatch: expected {expected_shapes!r}, got {dict(actual_shapes)!r}"
        )
    if dict(actual_dtypes) != expected_dtypes:
        raise ReplayCompatibilityError(
            f"array_dtypes mismatch: expected {expected_dtypes!r}, got {dict(actual_dtypes)!r}"
        )

    expected_dimensions = {
        "chunk_steps": int(arrays["observation"].shape[0]),
        "batch_size": int(arrays["observation"].shape[1]),
        "player_count": PLAYER_COUNT,
        "obs_dim": int(arrays["observation"].shape[3]),
        "action_count": int(arrays["action_weights"].shape[-1]),
    }
    for key, expected_value in expected_dimensions.items():
        actual = _metadata_int(metadata, key)
        if actual != expected_value:
            raise ReplayCompatibilityError(
                f"{key} mismatch: expected {expected_value!r}, got {actual!r}"
            )

    done_semantics = _metadata_string(metadata, "done_semantics")
    if done_semantics != REPLAY_DONE_SEMANTICS:
        raise ReplayCompatibilityError(
            f"done_semantics mismatch: expected {REPLAY_DONE_SEMANTICS!r}, got {done_semantics!r}"
        )

    final_reward_semantics = _metadata_string(metadata, "final_reward_map_semantics")
    if final_reward_semantics != FINAL_REWARD_MAP_SEMANTICS:
        raise ReplayCompatibilityError(
            "final_reward_map_semantics mismatch: expected "
            f"{FINAL_REWARD_MAP_SEMANTICS!r}, got {final_reward_semantics!r}"
        )


def _replay_field_spec(*, obs_dim: int, action_count: int) -> dict[str, list[Any]]:
    return {
        "observation": ["T", "B", PLAYER_COUNT, int(obs_dim), "float32"],
        "reward": ["T", "B", PLAYER_COUNT, "float32"],
        "action": ["T", "B", PLAYER_COUNT, "int16"],
        "action_weights": ["T", "B", PLAYER_COUNT, int(action_count), "float32"],
        "root_value": ["T", "B", PLAYER_COUNT, "float32"],
        "done": ["T", "B", "bool", REPLAY_DONE_SEMANTICS],
        "terminated": ["T", "B", "bool"],
        "truncated": ["T", "B", "bool"],
        "episode_id": ["B", "unicode"],
        "reset_seed": ["B", "int64"],
        "reset_source": ["B", "unicode"],
        "final_observation": ["B", PLAYER_COUNT, int(obs_dim), "float32"],
        "final_reward_map": ["B", PLAYER_COUNT, "float32", FINAL_REWARD_MAP_SEMANTICS],
    }


def _array_shapes(arrays: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    return {key: [int(value) for value in arrays[key].shape] for key in REPLAY_ARRAY_KEYS}


def _array_dtypes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {key: str(arrays[key].dtype) for key in REPLAY_ARRAY_KEYS}


def _expect_shape(array: np.ndarray, key: str, expected: tuple[int, ...]) -> None:
    if array.shape != expected:
        raise ReplayCompatibilityError(
            f"{key} shape mismatch: expected {list(expected)}, got {list(array.shape)}"
        )


def _expect_finite(array: np.ndarray, key: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ReplayCompatibilityError(f"{key} must contain only finite values")


def _load_metadata(value: np.ndarray) -> dict[str, Any]:
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ReplayCompatibilityError("metadata payload must be a JSON string")
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ReplayCompatibilityError("metadata payload must decode to an object")
    return loaded


def _metadata_mapping(metadata: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = metadata.get(key)
    if not isinstance(value, Mapping):
        raise ReplayCompatibilityError(f"{key} must be a metadata object")
    return value


def _metadata_string(metadata: Mapping[str, Any], key: str) -> str:
    return _non_empty_string(metadata.get(key), key)


def _metadata_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayCompatibilityError(f"{key} must be an integer")
    return value


def _non_empty_string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayCompatibilityError(f"{key} must be a non-empty string")
    return value
