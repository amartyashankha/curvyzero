"""Local replay chunk v0 helpers for the debug actor-loop bridge.

This module is intentionally small: it writes one local ``.npz`` chunk with a
JSON metadata record and the fixed in-memory arrays already staged by the actor
bridge benchmark. It is a compatibility guard, not a storage layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPLAY_METADATA_SCHEMA_ID = "curvyzero_debug_actor_loop_replay_chunk_metadata/v0"
REPLAY_SCHEMA_ID = "curvyzero_debug_actor_loop_replay_chunk/v0"

REPLAY_ARRAY_KEYS = (
    "obs",
    "reward",
    "action",
    "action_weights",
    "root_value",
    "done",
    "ego_mask",
)

REPLAY_ARRAY_DTYPES = {
    "obs": np.dtype("float32"),
    "reward": np.dtype("float32"),
    "action": np.dtype("int8"),
    "action_weights": np.dtype("float32"),
    "root_value": np.dtype("float32"),
    "done": np.dtype("bool"),
    "ego_mask": np.dtype("bool"),
}

REQUIRED_COMPATIBILITY_HASH_KEYS = (
    "replay_schema_hash",
    "rules_hash",
    "observation_schema_hash",
    "action_space_hash",
    "reward_schema_hash",
)

OPTIONAL_COMPATIBILITY_KEYS = (
    "replay_schema_id",
    "ruleset_id",
    "observation_schema_id",
    "action_space_id",
    "reward_schema_id",
    "env_impl_id",
    "env_impl_version",
    "env_impl_hash",
)

REPLAY_METADATA_POLICY_FIELDS = {
    "episode_id_policy": "absent_debug_sample_only",
    "reset_seed_policy": "absent_debug_sample_only",
    "reset_source_policy": "absent_debug_sample_only",
    "terminated_truncated_done_policy": "done_debug_surface_only_absent_terminated_truncated",
    "final_observation_policy": "absent_debug_sample_only_current_obs_only",
}


class ReplayCompatibilityError(ValueError):
    """Raised when a replay chunk does not match the expected v0 contract."""


@dataclass(frozen=True, slots=True)
class DebugActorLoopReplayChunk:
    """A loaded local replay chunk plus its metadata."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def stable_contract_hash(payload: Mapping[str, Any]) -> str:
    """Return the short deterministic hash used by local replay metadata."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def replay_schema_hash(*, obs_dim: int, action_count: int) -> str:
    """Hash the v0 replay field schema for this observation/action width."""

    return stable_contract_hash(
        {
            "schema_id": REPLAY_SCHEMA_ID,
            "metadata_schema": REPLAY_METADATA_SCHEMA_ID,
            "fields": _replay_field_spec(obs_dim=obs_dim, action_count=action_count),
        }
    )


def build_debug_actor_loop_replay_metadata(
    arrays: Mapping[str, Any],
    *,
    ruleset_id: str,
    rules_hash: str,
    observation_schema_id: str,
    observation_schema_hash: str,
    action_space_id: str,
    action_space_hash: str,
    reward_schema_id: str,
    reward_schema_hash: str,
    env_impl_id: str,
    env_impl_version: str,
    env_impl_hash: str | None = None,
    producer: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build metadata for one local debug actor-loop replay chunk."""

    normalized = _validate_arrays(arrays)
    chunk_steps, batch_size, player_count, obs_dim = normalized["obs"].shape
    action_count = normalized["action_weights"].shape[-1]
    metadata = {
        "metadata_schema_id": REPLAY_METADATA_SCHEMA_ID,
        "replay_schema_id": REPLAY_SCHEMA_ID,
        "replay_schema_hash": replay_schema_hash(
            obs_dim=int(obs_dim),
            action_count=int(action_count),
        ),
        "ruleset_id": _non_empty_string(ruleset_id, "ruleset_id"),
        "rules_hash": _non_empty_string(rules_hash, "rules_hash"),
        "observation_schema_id": _non_empty_string(
            observation_schema_id,
            "observation_schema_id",
        ),
        "observation_schema_hash": _non_empty_string(
            observation_schema_hash,
            "observation_schema_hash",
        ),
        "action_space_id": _non_empty_string(action_space_id, "action_space_id"),
        "action_space_hash": _non_empty_string(action_space_hash, "action_space_hash"),
        "reward_schema_id": _non_empty_string(reward_schema_id, "reward_schema_id"),
        "reward_schema_hash": _non_empty_string(
            reward_schema_hash,
            "reward_schema_hash",
        ),
        "env_impl_id": _non_empty_string(env_impl_id, "env_impl_id"),
        "env_impl_version": _non_empty_string(env_impl_version, "env_impl_version"),
        "env_impl_hash": env_impl_hash
        if env_impl_hash is not None
        else stable_contract_hash(
            {
                "env_impl_id": env_impl_id,
                "env_impl_version": env_impl_version,
            }
        ),
        "producer": producer,
        "created_at": created_at,
        "chunk_steps": int(chunk_steps),
        "batch_size": int(batch_size),
        "player_count": int(player_count),
        "obs_dim": int(obs_dim),
        "action_count": int(action_count),
        "array_shapes": _array_shapes(normalized),
        "array_dtypes": _array_dtypes(normalized),
        **REPLAY_METADATA_POLICY_FIELDS,
    }
    validate_debug_actor_loop_replay_chunk(arrays=normalized, metadata=metadata)
    return metadata


def write_debug_actor_loop_replay_chunk(
    path: str | Path,
    *,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Write one local replay chunk as ``.npz`` arrays plus JSON metadata."""

    normalized = validate_debug_actor_loop_replay_chunk(arrays=arrays, metadata=metadata)
    metadata_json = json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.array(metadata_json),
            **normalized,
        )


def read_debug_actor_loop_replay_chunk(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> DebugActorLoopReplayChunk:
    """Read and validate one local replay chunk."""

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

    normalized = validate_debug_actor_loop_replay_chunk(arrays=arrays, metadata=metadata)
    if expected_metadata is not None:
        validate_replay_chunk_compatibility(
            actual_metadata=metadata,
            expected_metadata=expected_metadata,
        )
    return DebugActorLoopReplayChunk(metadata=dict(metadata), arrays=normalized)


def validate_debug_actor_loop_replay_chunk(
    *,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Validate arrays and metadata for the local debug actor-loop replay chunk."""

    normalized = _validate_arrays(arrays)
    _validate_metadata(metadata=metadata, arrays=normalized)
    return normalized


def validate_replay_chunk_compatibility(
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
    """Extract the small compatibility record a reader should compare."""

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
    for key, expected_dtype in REPLAY_ARRAY_DTYPES.items():
        if normalized[key].dtype != expected_dtype:
            raise ReplayCompatibilityError(
                f"{key} dtype mismatch: expected {expected_dtype}, got {normalized[key].dtype}"
            )

    obs = normalized["obs"]
    if obs.ndim != 4:
        raise ReplayCompatibilityError(f"obs must have shape [T,B,P,D], got {list(obs.shape)}")
    chunk_steps, batch_size, player_count, obs_dim = obs.shape
    if min(chunk_steps, batch_size, player_count, obs_dim) <= 0:
        raise ReplayCompatibilityError(f"obs dimensions must be positive, got {list(obs.shape)}")

    _expect_shape(normalized["reward"], "reward", (chunk_steps, batch_size, player_count))
    _expect_shape(normalized["action"], "action", (chunk_steps, batch_size, player_count))
    action_weights = normalized["action_weights"]
    if action_weights.ndim != 4:
        raise ReplayCompatibilityError(
            f"action_weights must have shape [T,B,P,A], got {list(action_weights.shape)}"
        )
    if action_weights.shape[:3] != (chunk_steps, batch_size, player_count):
        raise ReplayCompatibilityError(
            "action_weights prefix shape mismatch: expected "
            f"{[chunk_steps, batch_size, player_count]}, got {list(action_weights.shape[:3])}"
        )
    if action_weights.shape[-1] <= 0:
        raise ReplayCompatibilityError("action_weights action count must be positive")
    _expect_shape(normalized["root_value"], "root_value", (chunk_steps, batch_size, player_count))
    _expect_shape(normalized["done"], "done", (chunk_steps, batch_size))
    _expect_shape(normalized["ego_mask"], "ego_mask", (chunk_steps, batch_size, player_count))
    return normalized


def _validate_metadata(
    *,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    metadata_schema_id = _metadata_string(metadata, "metadata_schema_id")
    if metadata_schema_id != REPLAY_METADATA_SCHEMA_ID:
        raise ReplayCompatibilityError(
            "metadata_schema_id mismatch: expected "
            f"{REPLAY_METADATA_SCHEMA_ID!r}, got {metadata_schema_id!r}"
        )

    replay_schema_id = _metadata_string(metadata, "replay_schema_id")
    if replay_schema_id != REPLAY_SCHEMA_ID:
        raise ReplayCompatibilityError(
            f"replay_schema_id mismatch: expected {REPLAY_SCHEMA_ID!r}, got {replay_schema_id!r}"
        )

    obs_dim = int(arrays["obs"].shape[-1])
    action_count = int(arrays["action_weights"].shape[-1])
    expected_schema_hash = replay_schema_hash(obs_dim=obs_dim, action_count=action_count)
    actual_schema_hash = _metadata_string(metadata, "replay_schema_hash")
    if actual_schema_hash != expected_schema_hash:
        raise ReplayCompatibilityError(
            "replay_schema_hash mismatch: expected "
            f"{expected_schema_hash!r}, got {actual_schema_hash!r}"
        )

    for key in (
        "ruleset_id",
        "rules_hash",
        "observation_schema_id",
        "observation_schema_hash",
        "action_space_id",
        "action_space_hash",
        "reward_schema_id",
        "reward_schema_hash",
        "env_impl_id",
        "env_impl_version",
        "env_impl_hash",
    ):
        _metadata_string(metadata, key)

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
        "chunk_steps": int(arrays["obs"].shape[0]),
        "batch_size": int(arrays["obs"].shape[1]),
        "player_count": int(arrays["obs"].shape[2]),
        "obs_dim": int(arrays["obs"].shape[3]),
        "action_count": int(arrays["action_weights"].shape[-1]),
    }
    for key, expected_value in expected_dimensions.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise ReplayCompatibilityError(
                f"{key} mismatch: expected {expected_value!r}, got {actual!r}"
            )

    for key, expected_value in REPLAY_METADATA_POLICY_FIELDS.items():
        actual = _metadata_string(metadata, key)
        if actual != expected_value:
            raise ReplayCompatibilityError(
                f"{key} mismatch: expected {expected_value!r}, got {actual!r}"
            )


def _replay_field_spec(*, obs_dim: int, action_count: int) -> dict[str, list[Any]]:
    return {
        "obs": ["chunk_steps", "B", "P", int(obs_dim), "float32"],
        "reward": ["chunk_steps", "B", "P", "float32"],
        "action": ["chunk_steps", "B", "P", "int8"],
        "action_weights": ["chunk_steps", "B", "P", int(action_count), "float32"],
        "root_value": ["chunk_steps", "B", "P", "float32"],
        "done": ["chunk_steps", "B", "bool"],
        "ego_mask": ["chunk_steps", "B", "P", "bool"],
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


def _non_empty_string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayCompatibilityError(f"{key} must be a non-empty string")
    return value
