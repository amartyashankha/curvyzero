import json
from pathlib import Path

import numpy as np
import pytest

from curvyzero.env import VectorTrainerEnv1v1NoBonus
from curvyzero.env import vector_reset
from curvyzero.training import trainer_replay_v0_builder
from curvyzero.training import vector_env_replay_recorder as recorder


_RULES_HASH = "unit-test-vector-env-rules"
_RULESET_ID = "unit-test-vector-env"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_NORMAL_WALL_SAME_FRAME_DRAW = (
    _REPO_ROOT / "scenarios" / "environment" / "source_normal_wall_same_frame_draw_step.json"
)


def test_vector_env_replay_recorder_uses_batch_info_for_nonterminal_chunk():
    env = VectorTrainerEnv1v1NoBonus(batch_size=2, seed=10)
    env.reset(seed=np.asarray([101, 202], dtype=np.uint64))
    replay_recorder = recorder.VectorEnvReplayRecorder()

    actions_0 = np.asarray([[1, 1], [0, 2]], dtype=np.int64)
    batch_0 = env.step(actions_0)
    weights_0 = _action_weights(row_bias=0)
    values_0 = _root_value(offset=0.0)
    replay_recorder.record_step(
        batch_0,
        actions=actions_0,
        action_weights=weights_0,
        root_value=values_0,
    )

    actions_1 = np.asarray([[0, 2], [1, 0]], dtype=np.int64)
    batch_1 = env.step(actions_1)
    weights_1 = _action_weights(row_bias=1)
    values_1 = _root_value(offset=0.5)
    replay_recorder.record_step(
        batch_1,
        actions=actions_1,
        action_weights=weights_1,
        root_value=values_1,
    )

    chunk = replay_recorder.build_chunk(
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )

    assert chunk.metadata["chunk_steps"] == 2
    assert chunk.metadata["batch_size"] == 2
    np.testing.assert_array_equal(chunk.arrays["observation"][0], batch_0.observation)
    np.testing.assert_array_equal(chunk.arrays["observation"][1], batch_1.observation)
    np.testing.assert_array_equal(chunk.arrays["reward"][0], batch_0.reward)
    np.testing.assert_array_equal(chunk.arrays["reward"][1], batch_1.reward)
    np.testing.assert_array_equal(chunk.arrays["action"][0], actions_0)
    np.testing.assert_array_equal(chunk.arrays["action"][1], actions_1)
    np.testing.assert_array_equal(chunk.arrays["action_weights"][0], weights_0)
    np.testing.assert_array_equal(chunk.arrays["root_value"][1], values_1)
    np.testing.assert_array_equal(chunk.arrays["done"], np.zeros((2, 2), dtype=bool))
    np.testing.assert_array_equal(
        chunk.arrays["final_observation"],
        batch_1.observation,
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_reward_map"],
        np.zeros((2, 2), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        chunk.arrays["episode_id"],
        batch_0.info["episode_id"].astype(str),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reset_seed"],
        batch_0.info["reset_seed"].astype(np.int64),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reset_source"],
        np.asarray(["manual", "manual"], dtype="<U6"),
    )


def test_vector_env_replay_recorder_uses_batch_info_for_terminal_autoreset_rows():
    env = VectorTrainerEnv1v1NoBonus(batch_size=2, seed=20, decision_ms=300.0)
    env.reset(seed=np.asarray([303, 404], dtype=np.uint64))
    env.state["pos"][0, 1] = [env.map_size - 1.0, env.map_size / 2.0]
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["heading"][0, 1] = 0.0
    env.state["speed"][0, 1] = 200.0

    actions = np.asarray([[1, 1], [1, 1]], dtype=np.int64)
    terminal_batch = env.step(actions)
    assert terminal_batch.final_observation is not None
    assert terminal_batch.final_reward is not None
    assert bool(terminal_batch.done[0]) is True
    assert bool(terminal_batch.done[1]) is False
    assert not np.allclose(terminal_batch.observation[0], terminal_batch.final_observation[0])

    chunk = recorder.build_vector_env_replay_chunk_v0(
        [terminal_batch],
        actions=actions.reshape(1, 2, 2),
        action_weights=_action_weights(row_bias=2).reshape(
            1,
            2,
            2,
            trainer_replay_v0_builder.ACTION_COUNT,
        ),
        root_value=_root_value(offset=1.0).reshape(1, 2, 2),
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )

    np.testing.assert_array_equal(
        chunk.arrays["done"],
        np.asarray([[True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        chunk.arrays["terminated"],
        np.asarray([[True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        chunk.arrays["observation"][0, 0],
        terminal_batch.final_observation[0],
    )
    np.testing.assert_array_equal(
        chunk.arrays["observation"][0, 1],
        terminal_batch.observation[1],
    )
    np.testing.assert_array_equal(
        chunk.arrays["reward"][0, 0],
        terminal_batch.final_reward[0],
    )
    np.testing.assert_array_equal(
        chunk.arrays["reward"][0, 1],
        terminal_batch.reward[1],
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_observation"][0],
        terminal_batch.final_observation[0],
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_observation"][1],
        terminal_batch.observation[1],
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_reward_map"],
        np.asarray([[1.0, -1.0], [0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        chunk.arrays["episode_id"],
        terminal_batch.info["episode_id"].astype(str),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reset_seed"],
        terminal_batch.info["reset_seed"].astype(np.int64),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reset_source"],
        np.asarray(["manual", "manual"], dtype="<U6"),
    )
    row_metadata = chunk.metadata["vector_env_row_metadata"]
    assert row_metadata["schema_id"] == recorder.VECTOR_ENV_ROW_METADATA_SCHEMA_ID
    assert row_metadata["env_id"] == "VectorTrainerEnv1v1NoBonus"
    assert row_metadata["row_id"] == [0, 1]
    assert row_metadata["terminal_row_id"] == [0]
    assert row_metadata["terminal_row_mask"] == [True, False]
    assert row_metadata["chunk_episode_id"] == ["1", "1"]
    assert row_metadata["chunk_reset_seed"] == [303, 404]
    assert row_metadata["chunk_reset_source"] == ["manual", "manual"]
    assert row_metadata["returned_episode_id"] == ["2", "1"]
    assert row_metadata["returned_reset_source"] == ["autoreset", "manual"]
    assert row_metadata["returned_observation_source"] == [
        "post_autoreset",
        "live_state",
    ]
    assert row_metadata["returned_reset_seed"][1] == 404
    assert row_metadata["returned_reset_seed"][0] == int(
        terminal_batch.info["returned_reset_seed"][0]
    )
    assert row_metadata["returned_identity_available"] is True
    assert row_metadata["missing_batch_info_fields"] == []
    assert row_metadata["rng_state_available"] is False
    assert row_metadata["rng_state_policy"] == recorder.VECTOR_ENV_RNG_STATE_POLICY
    assert int(env.state["reset_source"][0]) == vector_reset.RESET_SOURCE_AUTORESET


def test_vector_env_replay_recorder_packs_same_frame_wall_draw_terminal_rows():
    scenario = _load_scenario(_SOURCE_NORMAL_WALL_SAME_FRAME_DRAW)
    step_ms = float(scenario["time_policy"]["step_ms"])
    env = VectorTrainerEnv1v1NoBonus(batch_size=1, seed=25, decision_ms=step_ms)
    _seed_same_frame_wall_draw_row(env, scenario)

    actions = np.asarray([[1, 1]], dtype=np.int64)
    terminal_batch = env.step(actions)

    np.testing.assert_array_equal(terminal_batch.done, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.terminated, np.asarray([True], dtype=bool))
    np.testing.assert_array_equal(terminal_batch.truncated, np.asarray([False], dtype=bool))
    np.testing.assert_array_equal(
        terminal_batch.reward,
        np.zeros((1, 2), dtype=np.float32),
    )
    assert terminal_batch.final_observation is not None
    assert terminal_batch.final_reward is not None
    np.testing.assert_array_equal(
        terminal_batch.final_reward,
        np.zeros((1, 2), dtype=np.float32),
    )
    assert int(terminal_batch.info["terminal_reason"][0]) == (
        vector_reset.TERMINAL_REASON_ALL_DEAD_DRAW
    )
    assert int(terminal_batch.info["winner"][0]) == -1
    assert bool(terminal_batch.info["draw"][0]) is True

    chunk = recorder.build_vector_env_replay_chunk_v0(
        [terminal_batch],
        actions=actions.reshape(1, 1, 2),
        action_weights=np.full(
            (1, 1, 2, trainer_replay_v0_builder.ACTION_COUNT),
            np.float32(1.0 / trainer_replay_v0_builder.ACTION_COUNT),
            dtype=np.float32,
        ),
        root_value=np.zeros((1, 1, 2), dtype=np.float32),
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )

    np.testing.assert_array_equal(chunk.arrays["done"], np.asarray([[True]], dtype=bool))
    np.testing.assert_array_equal(
        chunk.arrays["terminated"],
        np.asarray([[True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        chunk.arrays["truncated"],
        np.asarray([[False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        chunk.arrays["observation"][0, 0],
        terminal_batch.final_observation[0],
    )
    np.testing.assert_array_equal(
        chunk.arrays["reward"][0, 0],
        terminal_batch.final_reward[0],
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_observation"][0],
        terminal_batch.final_observation[0],
    )
    np.testing.assert_array_equal(
        chunk.arrays["final_reward_map"],
        np.zeros((1, 2), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        chunk.arrays["episode_id"],
        terminal_batch.info["episode_id"].astype(str),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reset_seed"],
        terminal_batch.info["reset_seed"].astype(np.int64),
    )
    np.testing.assert_array_equal(chunk.arrays["reset_source"], np.asarray(["manual"]))


def test_vector_env_replay_recorder_terminal_row_closes_until_fresh_recorder():
    env = VectorTrainerEnv1v1NoBonus(batch_size=2, seed=30, decision_ms=300.0)
    env.reset(seed=np.asarray([505, 606], dtype=np.uint64))
    env.state["pos"][0, 1] = [env.map_size - 1.0, env.map_size / 2.0]
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["heading"][0, 1] = 0.0
    env.state["speed"][0, 1] = 200.0

    terminal_actions = np.asarray([[1, 1], [1, 1]], dtype=np.int64)
    terminal_batch = env.step(terminal_actions)
    assert bool(terminal_batch.done[0]) is True
    assert bool(terminal_batch.done[1]) is False
    assert int(env.state["reset_source"][0]) == vector_reset.RESET_SOURCE_AUTORESET
    terminal_episode_id = terminal_batch.info["episode_id"].copy()
    terminal_reset_seed = terminal_batch.info["reset_seed"].copy()
    terminal_reset_source = terminal_batch.info["reset_source"].copy()

    terminal_recorder = recorder.VectorEnvReplayRecorder()
    terminal_recorder.record_step(
        terminal_batch,
        actions=terminal_actions,
        action_weights=_action_weights(row_bias=0),
        root_value=_root_value(offset=2.0),
    )

    next_actions = np.asarray([[1, 1], [0, 2]], dtype=np.int64)
    next_batch = env.step(next_actions)
    assert not bool(next_batch.done.any())
    next_episode_id = next_batch.info["episode_id"].copy()
    next_reset_seed = next_batch.info["reset_seed"].copy()
    next_reset_source = next_batch.info["reset_source"].copy()

    with pytest.raises(
        recorder.VectorEnvReplayRecorderError,
        match="terminal vector env batch must be the final recorded timestep",
    ):
        terminal_recorder.record_step(
            next_batch,
            actions=next_actions,
            action_weights=_action_weights(row_bias=1),
            root_value=_root_value(offset=2.5),
        )

    terminal_chunk = terminal_recorder.build_chunk(
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )
    np.testing.assert_array_equal(
        terminal_chunk.arrays["episode_id"],
        terminal_episode_id.astype(str),
    )
    np.testing.assert_array_equal(
        terminal_chunk.arrays["reset_seed"],
        terminal_reset_seed.astype(np.int64),
    )
    np.testing.assert_array_equal(
        terminal_chunk.arrays["reset_source"],
        np.asarray(["manual", "manual"]),
    )

    fresh_recorder = recorder.VectorEnvReplayRecorder()
    fresh_recorder.record_step(
        next_batch,
        actions=next_actions,
        action_weights=_action_weights(row_bias=1),
        root_value=_root_value(offset=2.5),
    )
    fresh_chunk = fresh_recorder.build_chunk(
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )
    np.testing.assert_array_equal(
        fresh_chunk.arrays["episode_id"],
        next_episode_id.astype(str),
    )
    np.testing.assert_array_equal(
        fresh_chunk.arrays["reset_seed"],
        next_reset_seed.astype(np.int64),
    )
    np.testing.assert_array_equal(
        fresh_chunk.arrays["reset_source"],
        np.asarray(["autoreset", "manual"]),
    )
    np.testing.assert_array_equal(
        terminal_reset_source.astype(np.int64),
        np.asarray(
            [
                vector_reset.RESET_SOURCE_MANUAL,
                vector_reset.RESET_SOURCE_MANUAL,
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        next_reset_source.astype(np.int64),
        np.asarray(
            [
                vector_reset.RESET_SOURCE_AUTORESET,
                vector_reset.RESET_SOURCE_MANUAL,
            ],
            dtype=np.int64,
        ),
    )
    assert terminal_chunk.arrays["episode_id"][0] != fresh_chunk.arrays["episode_id"][0]
    assert terminal_chunk.arrays["episode_id"][1] == fresh_chunk.arrays["episode_id"][1]


def test_vector_env_replay_recorder_row_metadata_uses_final_batch_returned_identity():
    env = VectorTrainerEnv1v1NoBonus(batch_size=2, seed=45, decision_ms=300.0)
    env.reset(seed=np.asarray([1111, 2222], dtype=np.uint64))

    first_actions = np.asarray([[1, 1], [0, 2]], dtype=np.int64)
    first_batch = env.step(first_actions)
    assert not bool(first_batch.done.any())

    env.state["pos"][0, 1] = [env.map_size - 1.0, env.map_size / 2.0]
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["heading"][0, 1] = 0.0
    env.state["speed"][0, 1] = 200.0

    terminal_actions = np.asarray([[1, 1], [1, 1]], dtype=np.int64)
    terminal_batch = env.step(terminal_actions)
    assert bool(terminal_batch.done[0]) is True
    assert bool(terminal_batch.done[1]) is False

    chunk = recorder.build_vector_env_replay_chunk_v0(
        [first_batch, terminal_batch],
        actions=np.stack([first_actions, terminal_actions]),
        action_weights=np.stack(
            [
                _action_weights(row_bias=0),
                _action_weights(row_bias=1),
            ],
        ),
        root_value=np.stack(
            [
                _root_value(offset=0.0),
                _root_value(offset=0.5),
            ],
        ),
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )

    row_metadata = chunk.metadata["vector_env_row_metadata"]
    assert row_metadata["chunk_episode_id"] == ["1", "1"]
    assert row_metadata["chunk_reset_seed"] == [1111, 2222]
    assert row_metadata["chunk_reset_source"] == ["manual", "manual"]
    assert row_metadata["returned_episode_id"] == ["2", "1"]
    assert row_metadata["returned_reset_seed"][0] == int(
        terminal_batch.info["returned_reset_seed"][0]
    )
    assert row_metadata["returned_reset_seed"][1] == 2222
    assert row_metadata["returned_reset_source"] == ["autoreset", "manual"]
    assert row_metadata["terminal_row_id"] == [0]
    assert row_metadata["rng_state_available"] is False


def test_vector_env_replay_recorder_records_row_local_seed_rng_history_across_autoreset():
    env = VectorTrainerEnv1v1NoBonus(batch_size=2, seed=50, decision_ms=300.0)
    env.reset(seed=np.asarray([1001, 2002], dtype=np.uint64))
    env.state["pos"][0, 1] = [env.map_size - 1.0, env.map_size / 2.0]
    env.state["prev_pos"][0, 1] = env.state["pos"][0, 1]
    env.state["heading"][0, 1] = 0.0
    env.state["speed"][0, 1] = 200.0

    terminal_actions = np.asarray([[1, 1], [1, 1]], dtype=np.int64)
    terminal_batch = env.step(terminal_actions)
    assert bool(terminal_batch.done[0]) is True
    assert bool(terminal_batch.done[1]) is False

    reset_metadata = terminal_batch.info["autoreset_plan"]["reset_metadata"]
    autoreset_seed = int(reset_metadata["reset_seed"][0])
    np.testing.assert_array_equal(reset_metadata["rows"], np.asarray([0], dtype=np.int32))
    np.testing.assert_array_equal(
        reset_metadata["reset_source"],
        np.asarray([vector_reset.RESET_SOURCE_AUTORESET], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["returned_reset_seed"],
        np.asarray([autoreset_seed, 2002], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        terminal_batch.info["returned_reset_source"],
        np.asarray(
            [vector_reset.RESET_SOURCE_AUTORESET, vector_reset.RESET_SOURCE_MANUAL],
            dtype=np.int16,
        ),
    )

    expected_tape = np.random.default_rng(autoreset_seed).random(
        env.random_tape_capacity,
        dtype=np.float64,
    )
    np.testing.assert_allclose(env.reset_template["random_tape_values"][0], expected_tape)
    np.testing.assert_allclose(env.state["random_tape_values"][0], expected_tape)
    assert int(env.state["reset_seed"][0]) == autoreset_seed
    assert int(env.state["reset_source"][0]) == vector_reset.RESET_SOURCE_AUTORESET

    spawn_info = terminal_batch.info["autoreset_reset_info"]["spawn_info"]
    assert {int(call["row"]) for call in spawn_info["random_calls"]} == {0}
    assert int(spawn_info["random_draw_count_delta"][0]) > 0
    assert int(spawn_info["random_draw_count_delta"][1]) == 0

    next_actions = np.asarray([[0, 2], [1, 0]], dtype=np.int64)
    next_batch = env.step(next_actions)
    fresh_recorder = recorder.VectorEnvReplayRecorder()
    fresh_recorder.record_step(
        next_batch,
        actions=next_actions,
        action_weights=_action_weights(row_bias=1),
        root_value=_root_value(offset=3.0),
    )
    chunk = fresh_recorder.build_chunk(
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )

    np.testing.assert_array_equal(
        chunk.arrays["reset_seed"],
        np.asarray([autoreset_seed, 2002], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reset_source"],
        np.asarray(["autoreset", "manual"]),
    )
    np.testing.assert_array_equal(chunk.arrays["episode_id"], np.asarray(["2", "1"]))


def test_vector_env_replay_recorder_caller_metadata_overrides_batch_info():
    env = VectorTrainerEnv1v1NoBonus(batch_size=2, seed=40)
    env.reset(seed=np.asarray([707, 808], dtype=np.uint64))
    actions = np.asarray([[1, 1], [0, 2]], dtype=np.int64)
    batch = env.step(actions)

    replay_recorder = recorder.VectorEnvReplayRecorder()
    replay_recorder.record_step(
        batch,
        actions=actions,
        action_weights=_action_weights(row_bias=0),
        root_value=_root_value(offset=0.0),
    )

    chunk = replay_recorder.build_chunk(
        episode_id=["override-row-0", "override-row-1"],
        reset_seed=np.asarray([901, 902], dtype=np.int64),
        reset_source=["override-manual", "override-autoreset"],
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )

    np.testing.assert_array_equal(
        chunk.arrays["episode_id"],
        np.asarray(["override-row-0", "override-row-1"]),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reset_seed"],
        np.asarray([901, 902], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        chunk.arrays["reset_source"],
        np.asarray(["override-manual", "override-autoreset"]),
    )


def test_vector_env_replay_manifest_records_strict_sidecar_contract():
    env = VectorTrainerEnv1v1NoBonus(
        batch_size=2,
        seed=60,
        decision_ms=125.0,
        body_capacity=128,
        event_capacity=8,
        timer_capacity=4,
        random_tape_capacity=64,
        event_mode="no-event",
    )
    env.reset(seed=np.asarray([111, 222], dtype=np.uint64))
    actions = np.asarray([[1, 1], [0, 2]], dtype=np.int64)
    batch = env.step(actions)
    chunk = recorder.build_vector_env_replay_chunk_v0(
        [batch],
        actions=actions.reshape(1, 2, 2),
        action_weights=_action_weights(row_bias=0).reshape(
            1,
            2,
            2,
            trainer_replay_v0_builder.ACTION_COUNT,
        ),
        root_value=_root_value(offset=0.0).reshape(1, 2, 2),
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )

    manifest = recorder.build_vector_env_replay_manifest_v0(
        chunk=chunk,
        env=env,
        step_batches=[batch],
        source_claim_id="source-claim-unit-test",
        included_stages=["reset", "env.step", "replay-v0.build"],
    )

    assert json.loads(json.dumps(manifest, sort_keys=True)) == manifest
    assert (
        manifest["manifest_schema_id"]
        == recorder.VECTOR_ENV_REPLAY_MANIFEST_SCHEMA_ID
    )
    assert manifest["env_impl_id"] == "VectorTrainerEnv1v1NoBonus"
    assert manifest["ruleset_id"] == _RULESET_ID
    assert manifest["rules_hash"] == _RULES_HASH
    assert manifest["source_claim_id"] == "source-claim-unit-test"
    assert manifest["feature_flags"] == ["strict_1v1", "no_bonus", "P=2"]
    assert manifest["event_mode"] == "no-event"
    assert manifest["decision_ms"] == 125.0
    assert manifest["capacities"] == {
        "batch_size": 2,
        "player_count": 2,
        "obs_dim": 106,
        "action_count": trainer_replay_v0_builder.ACTION_COUNT,
        "chunk_steps": 1,
        "body_capacity": 128,
        "event_capacity": 8,
        "timer_capacity": 4,
        "random_tape_capacity": 64,
    }
    assert manifest["reset_seed"] == [111, 222]
    assert manifest["reset_source"] == ["manual", "manual"]
    assert manifest["included_stages"] == ["reset", "env.step", "replay-v0.build"]
    assert manifest["replay_contract_id"] == chunk.metadata["replay_contract_id"]
    assert manifest["replay_schema_id"] == chunk.metadata["replay_schema_id"]
    assert manifest["replay_schema_hash"] == chunk.metadata["replay_schema_hash"]

    final_policy = manifest["final_observation_policy"]
    assert final_policy["array"] == "final_observation"
    assert final_policy["source"] == "last_step_batch.info.final_observation_policy"
    assert final_policy["present"] is False
    assert final_policy["terminal_rows"] == []
    assert final_policy["row_mask"] == [False, False]
    assert final_policy["terminal_rows_only"] is True


def _action_weights(*, row_bias: int) -> np.ndarray:
    values = np.full(
        (2, 2, trainer_replay_v0_builder.ACTION_COUNT),
        np.float32(1.0 / trainer_replay_v0_builder.ACTION_COUNT),
        dtype=np.float32,
    )
    values[row_bias % 2, :, 0] += np.float32(0.01)
    return values


def _root_value(*, offset: float) -> np.ndarray:
    return (
        np.asarray(
            [
                [0.1, -0.1],
                [0.2, -0.2],
            ],
            dtype=np.float32,
        )
        + np.float32(offset)
    )


def _load_scenario(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_same_frame_wall_draw_row(
    env: VectorTrainerEnv1v1NoBonus,
    scenario: dict[str, object],
) -> None:
    env.reset(seed=np.asarray([909], dtype=np.uint64))
    row = 0
    source_setup = scenario["source_setup"]
    assert isinstance(source_setup, dict)
    game = source_setup["game"]
    assert isinstance(game, dict)
    players = scenario["players"]
    assert isinstance(players, list)
    assert len(players) == 2

    env.state["episode_id"][row] = 1
    env.state["episode_step"][row] = 0
    env.state["tick"][row] = 0
    env.state["elapsed_ms"][row] = 0.0
    env.state["done"][row] = False
    env.state["terminated"][row] = False
    env.state["truncated"][row] = False
    env.state["reset_pending"][row] = False
    env.state["terminal_reason"][row] = vector_reset.TERMINAL_REASON_NONE
    env.state["winner"][row] = -1
    env.state["draw"][row] = False
    env.state["started"][row] = bool(game["started"])
    env.state["in_round"][row] = bool(game["in_round"])
    env.state["world_active"][row] = bool(game["world_active"])
    env.state["borderless"][row] = bool(game["borderless"])
    env.state["map_size"][row] = float(source_setup["map_size"])

    env.state["world_body_count"][row] = 0
    env.state["body_active"][row, ...] = False
    env.state["body_pos"][row, ...] = 0.0
    env.state["body_radius"][row, ...] = 0.0
    env.state["body_owner"][row, ...] = -1
    env.state["body_num"][row, ...] = -1
    env.state["body_insert_tick"][row, ...] = -1
    env.state["body_insert_kind"][row, ...] = -1
    env.state["body_write_cursor"][row] = 0
    env.state["visible_trail_count"][row, ...] = 0
    env.state["has_visible_trail_last"][row, ...] = False
    env.state["visible_trail_last_pos"][row, ...] = 0.0
    env.state["has_draw_cursor"][row, ...] = False
    env.state["draw_cursor_pos"][row, ...] = 0.0

    for index, player in enumerate(players):
        assert isinstance(player, dict)
        initial = player["initial"]
        assert isinstance(initial, dict)
        env.state["pos"][row, index] = [float(initial["x"]), float(initial["y"])]
        env.state["prev_pos"][row, index] = env.state["pos"][row, index]
        env.state["heading"][row, index] = float(initial["angle_rad"])
        env.state["alive"][row, index] = True
        env.state["present"][row, index] = True
        env.state["score"][row, index] = 0
        env.state["round_score"][row, index] = 0
        env.state["printing"][row, index] = bool(initial["printing"])
        env.state["print_manager_active"][row, index] = False
        env.state["print_manager_distance"][row, index] = 0.0
        env.state["print_manager_last_pos"][row, index] = 0.0
        env.state["live_body_num"][row, index] = 0
        env.state["body_count"][row, index] = 0
