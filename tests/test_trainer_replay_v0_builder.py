from dataclasses import replace

import numpy as np
import pytest

from curvyzero.env import vector_runtime
from curvyzero.env import vector_trainer_observation
from curvyzero.training import replay_chunk_v0 as replay
from curvyzero.training import trainer_replay_v0_builder as builder


_RULES_HASH = "unit-test-rules-hash"
_RULESET_ID = "unit-test-ruleset"


def test_trainer_replay_v0_arrays_pack_nonterminal_trainer_payload():
    nonterminal, _terminal = _trainer_batches()

    arrays = builder.build_replay_v0_arrays_from_trainer_batches(
        [nonterminal],
        actions=_actions(1),
        action_weights=_action_weights(1),
        root_value=_root_value(1),
        episode_id="episode-a",
        reset_seed=7,
        reset_source="unit-test-reset",
    )
    metadata = builder.build_trainer_replay_v0_metadata(
        arrays,
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )
    validated = replay.validate_replay_chunk_v0(arrays=arrays, metadata=metadata)

    assert metadata["observation_schema_id"] == "curvyzero_egocentric_rays/v0"
    assert metadata["action_space_id"] == "curvyzero_turn3/v0"
    assert metadata["reward_schema_id"] == "curvyzero_sparse_round_outcome/v0"
    assert validated["observation"].shape == (1, 1, 2, 106)
    np.testing.assert_array_equal(validated["observation"][0, 0], nonterminal.observation)
    np.testing.assert_array_equal(validated["reward"][0, 0], nonterminal.rewards)
    np.testing.assert_array_equal(validated["done"], np.array([[False]], dtype=np.bool_))
    np.testing.assert_array_equal(
        validated["final_observation"][0],
        nonterminal.observation,
    )
    np.testing.assert_array_equal(
        validated["final_reward_map"],
        np.zeros((1, 2), dtype=np.float32),
    )


def test_trainer_replay_v0_chunk_round_trips_terminal_sequence(tmp_path):
    nonterminal, terminal = _trainer_batches()
    path = tmp_path / "trainer-replay-v0.npz"

    chunk = builder.build_trainer_replay_chunk_v0(
        [nonterminal, terminal],
        actions=_actions(2),
        action_weights=_action_weights(2),
        root_value=_root_value(2),
        episode_id="episode-terminal",
        reset_seed=7,
        reset_source="unit-test-reset",
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )
    builder.write_trainer_replay_chunk_v0(path, chunk)

    loaded = replay.read_replay_chunk_v0(
        path,
        expected_metadata=replay.compatibility_metadata(chunk.metadata),
    )

    assert loaded.metadata == chunk.metadata
    np.testing.assert_array_equal(loaded.arrays["observation"][0, 0], nonterminal.observation)
    np.testing.assert_array_equal(loaded.arrays["observation"][1, 0], terminal.observation)
    np.testing.assert_array_equal(loaded.arrays["reward"][1, 0], terminal.rewards)
    np.testing.assert_array_equal(
        loaded.arrays["done"],
        np.array([[False], [True]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        loaded.arrays["terminated"],
        np.array([[False], [True]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        loaded.arrays["truncated"],
        np.array([[False], [False]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(loaded.arrays["final_observation"][0], terminal.observation)
    np.testing.assert_array_equal(
        loaded.arrays["final_reward_map"],
        np.array([[1.0, -1.0]], dtype=np.float32),
    )


def test_trainer_replay_v0_arrays_pack_two_vector_rows_with_mixed_terminal_state():
    rows = _vector_batch_rows()

    chunk = builder.build_trainer_replay_chunk_v0(
        rows,
        actions=_actions(2, batch_size=2),
        action_weights=_action_weights(2, batch_size=2),
        root_value=_root_value(2, batch_size=2),
        episode_id=["episode-row-0", "episode-row-1"],
        reset_seed=np.asarray([7, 8], dtype=np.int64),
        reset_source=["manual-reset", "fixture-reset"],
        rules_hash=_RULES_HASH,
        ruleset_id=_RULESET_ID,
    )
    metadata = chunk.metadata
    validated = chunk.arrays

    assert metadata["batch_size"] == 2
    assert validated["observation"].shape == (2, 2, 2, 106)
    assert validated["reward"].shape == (2, 2, 2)
    assert validated["action"].shape == (2, 2, 2)
    assert validated["action_weights"].shape == (2, 2, 2, builder.ACTION_COUNT)
    assert validated["root_value"].shape == (2, 2, 2)
    np.testing.assert_array_equal(validated["observation"][0, 0], rows[0][0].observation)
    np.testing.assert_array_equal(validated["observation"][0, 1], rows[0][1].observation)
    np.testing.assert_array_equal(validated["observation"][1, 0], rows[1][0].observation)
    np.testing.assert_array_equal(validated["observation"][1, 1], rows[1][1].observation)
    np.testing.assert_array_equal(
        validated["done"],
        np.asarray([[False, False], [True, False]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        validated["terminated"],
        np.asarray([[False, False], [True, False]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(
        validated["truncated"],
        np.asarray([[False, False], [False, False]], dtype=np.bool_),
    )
    np.testing.assert_array_equal(validated["final_observation"][0], rows[1][0].observation)
    np.testing.assert_array_equal(validated["final_observation"][1], rows[1][1].observation)
    np.testing.assert_array_equal(
        validated["final_reward_map"],
        np.asarray([[1.0, -1.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_trainer_replay_v0_builder_rejects_vector_terminal_row_before_final_step():
    rows = _vector_batch_rows()

    with pytest.raises(replay.ReplayCompatibilityError, match="final supplied step"):
        builder.build_trainer_replay_chunk_v0(
            [rows[1], rows[0]],
            actions=_actions(2, batch_size=2),
            action_weights=_action_weights(2, batch_size=2),
            root_value=_root_value(2, batch_size=2),
            episode_id=["episode-row-0", "episode-row-1"],
            reset_seed=np.asarray([7, 8], dtype=np.int64),
            reset_source=["manual-reset", "fixture-reset"],
            rules_hash=_RULES_HASH,
            ruleset_id=_RULESET_ID,
        )


def test_trainer_replay_v0_builder_rejects_bad_policy_shapes():
    nonterminal, _terminal = _trainer_batches()

    with pytest.raises(replay.ReplayCompatibilityError, match="actions shape"):
        builder.build_replay_v0_arrays_from_trainer_batches(
            [nonterminal],
            actions=np.zeros((1, 3), dtype=np.int64),
            action_weights=_action_weights(1),
            root_value=_root_value(1),
            episode_id="episode-a",
            reset_seed=7,
            reset_source="unit-test-reset",
        )


def test_trainer_replay_v0_builder_rejects_bad_reset_metadata():
    nonterminal, _terminal = _trainer_batches()

    with pytest.raises(replay.ReplayCompatibilityError, match="episode_id"):
        builder.build_replay_v0_arrays_from_trainer_batches(
            [nonterminal],
            actions=_actions(1),
            action_weights=_action_weights(1),
            root_value=_root_value(1),
            episode_id="",
            reset_seed=7,
            reset_source="unit-test-reset",
        )


def test_trainer_replay_v0_builder_rejects_inconsistent_terminal_metadata():
    nonterminal, terminal = _trainer_batches()
    bad_done = replace(nonterminal, done=True)
    bad_final_reward_map = replace(terminal, final_reward_map=None)

    with pytest.raises(replay.ReplayCompatibilityError, match=r"done must equal"):
        builder.build_replay_v0_arrays_from_trainer_batches(
            [bad_done],
            actions=_actions(1),
            action_weights=_action_weights(1),
            root_value=_root_value(1),
            episode_id="episode-a",
            reset_seed=7,
            reset_source="unit-test-reset",
        )

    with pytest.raises(replay.ReplayCompatibilityError, match="final_reward_map"):
        builder.build_replay_v0_arrays_from_trainer_batches(
            [bad_final_reward_map],
            actions=_actions(1),
            action_weights=_action_weights(1),
            root_value=_root_value(1),
            episode_id="episode-a",
            reset_seed=7,
            reset_source="unit-test-reset",
        )


def _trainer_batches():
    state = _vector_state(batch_size=1)
    nonterminal = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        state,
        0,
        decision_ms=300.0,
        max_ticks=100,
    )

    terminal_state = _vector_state(batch_size=1)
    terminal_state["alive"][0] = [True, False]
    terminal_state["done"][0] = True
    terminal_state["terminated"][0] = True
    terminal_state["terminal_reason"][0] = vector_runtime.TERMINAL_REASON_SURVIVOR_WIN
    terminal_state["winner"][0] = 0
    terminal = vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
        terminal_state,
        0,
        decision_ms=300.0,
        max_ticks=100,
    )
    return nonterminal, terminal


def _vector_batch_rows():
    state = _vector_state(batch_size=2)
    step_0 = tuple(
        vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
            state,
            row,
            decision_ms=300.0,
            max_ticks=100,
        )
        for row in range(2)
    )

    final_state = _vector_state(batch_size=2)
    final_state["tick"] += 1
    final_state["alive"][0] = [True, False]
    final_state["done"][0] = True
    final_state["terminated"][0] = True
    final_state["terminal_reason"][0] = vector_runtime.TERMINAL_REASON_SURVIVOR_WIN
    final_state["winner"][0] = 0
    step_1 = tuple(
        vector_trainer_observation.observe_vector_1v1_egocentric_rays_v0(
            final_state,
            row,
            decision_ms=300.0,
            max_ticks=100,
        )
        for row in range(2)
    )
    return (step_0, step_1)


def _vector_state(batch_size: int, body_capacity: int = 6) -> dict[str, np.ndarray]:
    state = {
        "pos": np.zeros((batch_size, 2, 2), dtype=np.float64),
        "heading": np.zeros((batch_size, 2), dtype=np.float64),
        "alive": np.ones((batch_size, 2), dtype=np.bool_),
        "tick": np.arange(batch_size, dtype=np.int32) + 5,
        "map_size": np.full(batch_size, 64.0, dtype=np.float64),
        "radius": np.ones((batch_size, 2), dtype=np.float64),
        "speed": np.full((batch_size, 2), 16.0, dtype=np.float64),
        "angular_velocity_per_ms": np.full(
            (batch_size, 2),
            2.8 / 1000.0,
            dtype=np.float64,
        ),
        "body_active": np.zeros((batch_size, body_capacity), dtype=np.bool_),
        "body_pos": np.zeros((batch_size, body_capacity, 2), dtype=np.float64),
        "body_radius": np.ones((batch_size, body_capacity), dtype=np.float64),
        "body_owner": np.full((batch_size, body_capacity), -1, dtype=np.int16),
        "borderless": np.zeros(batch_size, dtype=np.bool_),
        "done": np.zeros(batch_size, dtype=np.bool_),
        "terminated": np.zeros(batch_size, dtype=np.bool_),
        "truncated": np.zeros(batch_size, dtype=np.bool_),
        "terminal_reason": np.zeros(batch_size, dtype=np.int16),
        "winner": np.full(batch_size, -1, dtype=np.int16),
        "draw": np.zeros(batch_size, dtype=np.bool_),
    }
    state["pos"][0] = [[10.0, 10.0], [20.0, 10.0]]
    state["heading"][0] = [0.0, np.pi]
    state["body_active"][0, :4] = True
    state["body_pos"][0, :4] = [
        [10.0, 10.0],
        [14.0, 10.0],
        [10.0, 5.0],
        [20.0, 10.0],
    ]
    state["body_owner"][0, :4] = [0, 0, 1, 1]

    if batch_size > 1:
        state["pos"][1] = [[12.0, 12.0], [40.0, 12.0]]
        state["heading"][1] = [0.0, np.pi]
        state["body_active"][1, :2] = True
        state["body_pos"][1, :2] = [[12.0, 12.0], [40.0, 12.0]]
        state["body_owner"][1, :2] = [0, 1]
    return state


def _actions(chunk_steps: int, *, batch_size: int = 1) -> np.ndarray:
    values = np.asarray(
        [
            [[1, 1], [0, 2]],
            [[0, 2], [1, 0]],
        ][:chunk_steps],
        dtype=np.int64,
    )
    if batch_size == 1:
        return values[:, 0]
    return values[:, :batch_size]


def _action_weights(chunk_steps: int, *, batch_size: int = 1) -> np.ndarray:
    values = np.full(
        (chunk_steps, batch_size, 2, builder.ACTION_COUNT),
        np.float32(1.0 / builder.ACTION_COUNT),
        dtype=np.float32,
    )
    if batch_size == 1:
        return values[:, 0]
    return values


def _root_value(chunk_steps: int, *, batch_size: int = 1) -> np.ndarray:
    values = np.linspace(
        0.25,
        -0.25,
        chunk_steps * batch_size * 2,
        dtype=np.float32,
    ).reshape(chunk_steps, batch_size, 2)
    if batch_size == 1:
        return values[:, 0]
    return values
