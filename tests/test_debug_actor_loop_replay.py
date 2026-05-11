import json

import numpy as np
import pytest

from curvyzero.training import debug_actor_loop_replay as replay


def test_debug_actor_loop_replay_chunk_round_trips_with_metadata(tmp_path):
    arrays = _arrays()
    metadata = _metadata(arrays)
    path = tmp_path / "chunk-000000.npz"

    replay.write_debug_actor_loop_replay_chunk(path, arrays=arrays, metadata=metadata)

    loaded = replay.read_debug_actor_loop_replay_chunk(
        path,
        expected_metadata=replay.compatibility_metadata(metadata),
    )

    assert loaded.metadata == metadata
    for key, value in arrays.items():
        np.testing.assert_array_equal(loaded.arrays[key], value)


@pytest.mark.parametrize("field", replay.REQUIRED_COMPATIBILITY_HASH_KEYS)
def test_debug_actor_loop_replay_reader_rejects_hash_mismatch(tmp_path, field):
    arrays = _arrays()
    metadata = _metadata(arrays)
    path = tmp_path / "chunk-000000.npz"
    replay.write_debug_actor_loop_replay_chunk(path, arrays=arrays, metadata=metadata)
    expected_metadata = replay.compatibility_metadata(metadata)
    expected_metadata[field] = f"wrong-{field}"

    with pytest.raises(replay.ReplayCompatibilityError, match=field):
        replay.read_debug_actor_loop_replay_chunk(
            path,
            expected_metadata=expected_metadata,
        )


def test_debug_actor_loop_replay_reader_rejects_env_impl_hash_mismatch(tmp_path):
    arrays = _arrays()
    metadata = _metadata(arrays)
    path = tmp_path / "chunk-000000.npz"
    replay.write_debug_actor_loop_replay_chunk(path, arrays=arrays, metadata=metadata)
    expected_metadata = replay.compatibility_metadata(metadata)
    expected_metadata["env_impl_hash"] = "wrong-env-impl-hash"

    with pytest.raises(replay.ReplayCompatibilityError, match="env_impl_hash"):
        replay.read_debug_actor_loop_replay_chunk(
            path,
            expected_metadata=expected_metadata,
        )


def test_debug_actor_loop_replay_reader_rejects_tampered_replay_schema_hash(tmp_path):
    arrays = _arrays()
    metadata = _metadata(arrays)
    metadata["replay_schema_hash"] = "wrong-replay-schema-hash"
    path = tmp_path / "chunk-000000.npz"
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            **arrays,
        )

    with pytest.raises(replay.ReplayCompatibilityError, match="replay_schema_hash"):
        replay.read_debug_actor_loop_replay_chunk(path)


def test_debug_actor_loop_replay_writer_rejects_array_metadata_mismatch(tmp_path):
    arrays = _arrays()
    metadata = _metadata(arrays)
    metadata["array_shapes"] = {
        **metadata["array_shapes"],
        "obs": [1, 2, 2, 4],
    }

    with pytest.raises(replay.ReplayCompatibilityError, match="array_shapes"):
        replay.write_debug_actor_loop_replay_chunk(
            tmp_path / "chunk-000000.npz",
            arrays=arrays,
            metadata=metadata,
        )


def test_debug_actor_loop_replay_metadata_contract_keeps_blockers_and_hashes_explicit():
    arrays = _arrays()
    metadata = _metadata(arrays)

    assert replay.REPLAY_METADATA_POLICY_FIELDS == {
        "episode_id_policy": "absent_debug_sample_only",
        "reset_seed_policy": "absent_debug_sample_only",
        "reset_source_policy": "absent_debug_sample_only",
        "terminated_truncated_done_policy": (
            "done_debug_surface_only_absent_terminated_truncated"
        ),
        "final_observation_policy": "absent_debug_sample_only_current_obs_only",
    }
    assert replay.REQUIRED_COMPATIBILITY_HASH_KEYS == (
        "replay_schema_hash",
        "rules_hash",
        "observation_schema_hash",
        "action_space_hash",
        "reward_schema_hash",
    )
    assert {key: metadata[key] for key in replay.REPLAY_METADATA_POLICY_FIELDS} == (
        replay.REPLAY_METADATA_POLICY_FIELDS
    )
    assert set(replay.REQUIRED_COMPATIBILITY_HASH_KEYS) <= set(metadata)


def test_debug_actor_loop_replay_reader_rejects_legacy_metadata_without_absent_policy(
    tmp_path,
):
    arrays = _arrays()
    metadata = _metadata(arrays)
    metadata.pop("episode_id_policy")
    path = tmp_path / "chunk-000000.npz"
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            **arrays,
        )

    with pytest.raises(replay.ReplayCompatibilityError, match="episode_id_policy"):
        replay.read_debug_actor_loop_replay_chunk(path)


@pytest.mark.parametrize("field", replay.REPLAY_METADATA_POLICY_FIELDS)
def test_debug_actor_loop_replay_writer_requires_explicit_absent_policy(tmp_path, field):
    arrays = _arrays()
    metadata = _metadata(arrays)
    metadata.pop(field)

    with pytest.raises(replay.ReplayCompatibilityError, match=field):
        replay.write_debug_actor_loop_replay_chunk(
            tmp_path / "chunk-000000.npz",
            arrays=arrays,
            metadata=metadata,
        )


@pytest.mark.parametrize("field", replay.REPLAY_METADATA_POLICY_FIELDS)
def test_debug_actor_loop_replay_writer_rejects_unsupported_absent_policy(tmp_path, field):
    arrays = _arrays()
    metadata = _metadata(arrays)
    metadata[field] = "production_replay_claim"

    with pytest.raises(replay.ReplayCompatibilityError, match=field):
        replay.write_debug_actor_loop_replay_chunk(
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
    obs = np.arange(
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
        dtype=np.int8,
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
    done = np.array([[False, True], [False, False]], dtype=bool)
    ego_mask = np.array(
        [
            [[True, True], [True, False]],
            [[True, True], [False, False]],
        ],
        dtype=bool,
    )
    return {
        "obs": obs,
        "reward": reward,
        "action": action,
        "action_weights": action_weights,
        "root_value": root_value,
        "done": done,
        "ego_mask": ego_mask,
    }


def _metadata(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return replay.build_debug_actor_loop_replay_metadata(
        arrays,
        ruleset_id="curvytron-v1-reference",
        rules_hash="rules-hash-001",
        observation_schema_id="curvyzero_debug_global_player_obs/v0",
        observation_schema_hash="obs-hash-001",
        action_space_id="curvyzero_source_move_action_space/v0",
        action_space_hash="action-hash-001",
        reward_schema_id="curvyzero_debug_score_round_delta_death_penalty/v0",
        reward_schema_hash="reward-hash-001",
        env_impl_id="fixture_seeded_numpy_vector_actor_loop_bridge",
        env_impl_version="curvyzero_vector_actor_loop_bridge/v1",
        producer="test_debug_actor_loop_replay",
        created_at="2026-05-09T00:00:00Z",
    )
