import json

import numpy as np
import pytest

from curvyzero.training import replay_chunk_v0 as replay


def test_replay_chunk_v0_round_trips_npz_payload_and_metadata(tmp_path):
    arrays = _arrays()
    metadata = _metadata(arrays)
    path = tmp_path / "chunk-000000.npz"

    replay.write_replay_chunk_v0(path, arrays=arrays, metadata=metadata)

    loaded = replay.read_replay_chunk_v0(
        path,
        expected_metadata=replay.compatibility_metadata(metadata),
    )

    assert loaded.metadata == metadata
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded.arrays[key], value)


def test_replay_chunk_v0_contract_keeps_required_fields_explicit():
    arrays = _arrays()
    metadata = _metadata(arrays)

    assert replay.REPLAY_ARRAY_KEYS == (
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
    assert replay.REQUIRED_COMPATIBILITY_HASH_KEYS == (
        "replay_schema_hash",
        "rules_hash",
        "observation_schema_hash",
        "action_space_hash",
        "reward_schema_hash",
    )
    assert metadata["player_count"] == 2
    assert metadata["done_semantics"] == "done == terminated | truncated"
    assert (
        metadata["final_reward_map_semantics"]
        == "player_indexed_final_reward_no_bonus"
    )
    assert metadata["array_shapes"] == {
        "observation": [2, 2, 2, 4],
        "reward": [2, 2, 2],
        "action": [2, 2, 2],
        "action_weights": [2, 2, 2, 3],
        "root_value": [2, 2, 2],
        "done": [2, 2],
        "terminated": [2, 2],
        "truncated": [2, 2],
        "episode_id": [2],
        "reset_seed": [2],
        "reset_source": [2],
        "final_observation": [2, 2, 4],
        "final_reward_map": [2, 2],
    }


@pytest.mark.parametrize("field", replay.REQUIRED_COMPATIBILITY_HASH_KEYS)
def test_replay_chunk_v0_reader_rejects_hash_mismatch(tmp_path, field):
    arrays = _arrays()
    metadata = _metadata(arrays)
    path = tmp_path / "chunk-000000.npz"
    replay.write_replay_chunk_v0(path, arrays=arrays, metadata=metadata)
    expected_metadata = replay.compatibility_metadata(metadata)
    expected_metadata[field] = f"wrong-{field}"

    with pytest.raises(replay.ReplayCompatibilityError, match=field):
        replay.read_replay_chunk_v0(path, expected_metadata=expected_metadata)


def test_replay_chunk_v0_reader_rejects_tampered_replay_schema_hash(tmp_path):
    arrays = _arrays()
    metadata = _metadata(arrays)
    metadata["replay_schema_hash"] = "wrong-replay-schema-hash"
    path = tmp_path / "chunk-000000.npz"
    _write_raw_npz(path, arrays=arrays, metadata=metadata)

    with pytest.raises(replay.ReplayCompatibilityError, match="replay_schema_hash"):
        replay.read_replay_chunk_v0(path)


def test_replay_chunk_v0_reader_rejects_missing_metadata_field(tmp_path):
    arrays = _arrays()
    metadata = _metadata(arrays)
    metadata.pop("rules_hash")
    path = tmp_path / "chunk-000000.npz"
    _write_raw_npz(path, arrays=arrays, metadata=metadata)

    with pytest.raises(replay.ReplayCompatibilityError, match="rules_hash"):
        replay.read_replay_chunk_v0(path)


def test_replay_chunk_v0_reader_rejects_missing_array(tmp_path):
    arrays = _arrays()
    metadata = _metadata(arrays)
    arrays.pop("final_observation")
    path = tmp_path / "chunk-000000.npz"
    _write_raw_npz(path, arrays=arrays, metadata=metadata)

    with pytest.raises(replay.ReplayCompatibilityError, match="final_observation"):
        replay.read_replay_chunk_v0(path)


def test_replay_chunk_v0_writer_rejects_array_metadata_mismatch(tmp_path):
    arrays = _arrays()
    metadata = _metadata(arrays)
    metadata["array_shapes"] = {
        **metadata["array_shapes"],
        "final_reward_map": [2, 3],
    }

    with pytest.raises(replay.ReplayCompatibilityError, match="array_shapes"):
        replay.write_replay_chunk_v0(
            tmp_path / "chunk-000000.npz",
            arrays=arrays,
            metadata=metadata,
        )


def test_replay_chunk_v0_writer_rejects_wrong_dtype(tmp_path):
    arrays = _arrays()
    arrays["action"] = arrays["action"].astype(np.int64)
    metadata = _metadata(_arrays())

    with pytest.raises(replay.ReplayCompatibilityError, match="action dtype"):
        replay.write_replay_chunk_v0(
            tmp_path / "chunk-000000.npz",
            arrays=arrays,
            metadata=metadata,
        )


def test_replay_chunk_v0_writer_rejects_object_episode_id(tmp_path):
    arrays = _arrays()
    arrays["episode_id"] = np.array(["episode-a", "episode-b"], dtype=object)
    metadata = _metadata(_arrays())

    with pytest.raises(replay.ReplayCompatibilityError, match="episode_id dtype"):
        replay.write_replay_chunk_v0(
            tmp_path / "chunk-000000.npz",
            arrays=arrays,
            metadata=metadata,
        )


def test_replay_chunk_v0_writer_rejects_done_semantics_mismatch(tmp_path):
    arrays = _arrays()
    arrays["done"] = np.zeros_like(arrays["done"], dtype=bool)
    metadata = _metadata(_arrays())

    with pytest.raises(replay.ReplayCompatibilityError, match=r"done must equal"):
        replay.write_replay_chunk_v0(
            tmp_path / "chunk-000000.npz",
            arrays=arrays,
            metadata=metadata,
        )


def test_replay_chunk_v0_writer_rejects_non_1v1_shape(tmp_path):
    arrays = _arrays()
    arrays["observation"] = np.zeros((2, 2, 3, 4), dtype=np.float32)
    metadata = _metadata(_arrays())

    with pytest.raises(replay.ReplayCompatibilityError, match="player_count"):
        replay.write_replay_chunk_v0(
            tmp_path / "chunk-000000.npz",
            arrays=arrays,
            metadata=metadata,
        )


def _arrays() -> dict[str, np.ndarray]:
    chunk_steps = 2
    batch_size = 2
    player_count = 2
    obs_dim = 4
    action_count = 3
    observation = np.arange(
        chunk_steps * batch_size * player_count * obs_dim,
        dtype=np.float32,
    ).reshape(chunk_steps, batch_size, player_count, obs_dim)
    reward = np.linspace(
        -1.0,
        1.0,
        chunk_steps * batch_size * player_count,
        dtype=np.float32,
    ).reshape(chunk_steps, batch_size, player_count)
    action = np.array(
        [
            [[0, 1], [2, 1]],
            [[1, 0], [2, 2]],
        ],
        dtype=np.int16,
    )
    action_weights = np.zeros(
        (chunk_steps, batch_size, player_count, action_count),
        dtype=np.float32,
    )
    for action_id in range(action_count):
        action_weights[..., action_id] = np.float32(1.0 / action_count)
    root_value = np.linspace(
        0.25,
        -0.25,
        chunk_steps * batch_size * player_count,
        dtype=np.float32,
    ).reshape(chunk_steps, batch_size, player_count)
    terminated = np.array([[False, True], [False, False]], dtype=bool)
    truncated = np.array([[False, False], [True, False]], dtype=bool)
    done = np.logical_or(terminated, truncated)
    final_observation = np.arange(
        batch_size * player_count * obs_dim,
        dtype=np.float32,
    ).reshape(batch_size, player_count, obs_dim)
    final_reward_map = np.array([[1.0, -1.0], [0.0, 0.0]], dtype=np.float32)
    return {
        "observation": observation,
        "reward": reward,
        "action": action,
        "action_weights": action_weights,
        "root_value": root_value,
        "done": done,
        "terminated": terminated,
        "truncated": truncated,
        "episode_id": np.array(["episode-a", "episode-b"], dtype=np.str_),
        "reset_seed": np.array([1001, 1002], dtype=np.int64),
        "reset_source": np.array(["source-fixture", "source-fixture"], dtype=np.str_),
        "final_observation": final_observation,
        "final_reward_map": final_reward_map,
    }


def _metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return replay.build_replay_chunk_v0_metadata(
        arrays,
        rules_hash="rules-hash-001",
        observation_schema_hash="obs-hash-001",
        action_space_hash="action-hash-001",
        reward_schema_hash="reward-hash-001",
        ruleset_id="curvytron-v1-reference",
        observation_schema_id="curvyzero_global_player_observation/v0",
        action_space_id="curvyzero_source_move_action_space/v0",
        reward_schema_id="curvyzero_1v1_no_bonus_reward/v0",
        producer="test_replay_chunk_v0",
        created_at="2026-05-09T00:00:00Z",
    )


def _write_raw_npz(
    path,
    *,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    with path.open("wb") as handle:
        np.savez(
            handle,
            metadata=np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            **arrays,
        )
