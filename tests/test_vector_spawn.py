import math

import numpy as np

from curvyzero.env import vector_spawn


def _spawn_state(
    *,
    map_size: list[float],
    present: list[list[bool]],
    random_tape_values: list[list[float]],
    death_capacity: int | None = None,
) -> dict[str, np.ndarray]:
    batch_size = len(map_size)
    player_count = len(present[0])
    tape_capacity = max(len(row) for row in random_tape_values)
    tape = np.zeros((batch_size, tape_capacity), dtype=np.float64)
    tape_length = np.zeros(batch_size, dtype=np.int32)
    for row, values in enumerate(random_tape_values):
        tape[row, : len(values)] = values
        tape_length[row] = len(values)

    state = {
        "pos": np.zeros((batch_size, player_count, 2), dtype=np.float64),
        "prev_pos": np.zeros((batch_size, player_count, 2), dtype=np.float64),
        "heading": np.zeros((batch_size, player_count), dtype=np.float64),
        "alive": np.zeros((batch_size, player_count), dtype=bool),
        "present": np.asarray(present, dtype=bool),
        "map_size": np.asarray(map_size, dtype=np.float64),
        "random_tape_values": tape,
        "random_tape_length": tape_length,
        "random_tape_cursor": np.zeros(batch_size, dtype=np.int32),
        "random_tape_exhausted": np.ones(batch_size, dtype=bool),
        "random_tape_draw_count": np.zeros(batch_size, dtype=np.int32),
    }
    if death_capacity is not None:
        state["death_count"] = np.zeros(batch_size, dtype=np.int16)
        state["death_player"] = np.full((batch_size, death_capacity), -1, dtype=np.int16)
    return state


def test_spawn_round_rows_matches_promoted_2p_heading_retry_fixture():
    state = _spawn_state(
        map_size=[88.0],
        present=[[True, True]],
        random_tape_values=[
            [
                0.0,
                0.5,
                0.5,
                0.25,
                0.6794871794871795,
                0.5,
                0.5159154943091895,
            ]
        ],
    )

    info = vector_spawn.spawn_round_rows(
        state,
        np.asarray([True], dtype=bool),
        player_count=2,
    )

    assert info["schema"] == vector_spawn.SPAWN_INFO_SCHEMA_ID
    assert info["world_body_insert_count"] == 0
    np.testing.assert_array_equal(info["spawn_order"], np.asarray([1, 0], dtype=np.int16))
    np.testing.assert_allclose(state["pos"][0], [[58.0, 44.0], [5.0, 44.0]])
    np.testing.assert_allclose(state["prev_pos"][0], state["pos"][0])
    np.testing.assert_allclose(
        state["heading"][0],
        [0.5159154943091895 * math.tau, 0.25 * math.tau],
    )
    np.testing.assert_array_equal(state["alive"][0], np.asarray([True, True], dtype=bool))
    assert int(state["random_tape_cursor"][0]) == 7
    assert int(state["random_tape_draw_count"][0]) == 7
    assert bool(state["random_tape_exhausted"][0]) is False
    assert [
        (call["tape_index"], call["site"], call["player"], call["value"])
        for call in info["random_calls"]
    ] == [
        (0, "spawn.position_x", 1, 0.0),
        (1, "spawn.position_y", 1, 0.5),
        (2, "spawn.angle_attempt_0", 1, 0.5),
        (3, "spawn.angle_attempt_1", 1, 0.25),
        (4, "spawn.position_x", 0, 0.6794871794871795),
        (5, "spawn.position_y", 0, 0.5),
        (6, "spawn.angle_attempt_0", 0, 0.5159154943091895),
    ]


def test_spawn_round_rows_matches_promoted_3p_reverse_spawn_order_fixture():
    state = _spawn_state(
        map_size=[95.0],
        present=[[True, True, True]],
        random_tape_values=[
            [
                0.25,
                0.5,
                0.25,
                0.5,
                0.5,
                0.5,
                0.75,
                0.5,
                0.75,
            ]
        ],
    )
    state["world_body_count"] = np.asarray([123], dtype=np.int32)
    state["body_active"] = np.asarray([[True, False]], dtype=bool)
    body_active_before = state["body_active"].copy()

    info = vector_spawn.spawn_round_rows(
        state,
        np.asarray([True], dtype=bool),
        player_count=3,
    )

    np.testing.assert_allclose(
        state["pos"][0],
        [[68.575, 47.5], [47.5, 47.5], [26.425, 47.5]],
    )
    np.testing.assert_allclose(
        state["heading"][0],
        [0.75 * math.tau, 0.5 * math.tau, 0.25 * math.tau],
    )
    np.testing.assert_array_equal(state["alive"][0], np.asarray([True, True, True], dtype=bool))
    assert int(state["random_tape_cursor"][0]) == 9
    assert int(state["random_tape_draw_count"][0]) == 9
    assert [call["player"] for call in info["random_calls"]] == [2, 2, 2, 1, 1, 1, 0, 0, 0]
    assert [call["site"] for call in info["random_calls"]] == [
        "spawn.position_x",
        "spawn.position_y",
        "spawn.angle_attempt_0",
        "spawn.position_x",
        "spawn.position_y",
        "spawn.angle_attempt_0",
        "spawn.position_x",
        "spawn.position_y",
        "spawn.angle_attempt_0",
    ]
    assert info["world_body_insert_count"] == 0
    assert int(state["world_body_count"][0]) == 123
    np.testing.assert_array_equal(state["body_active"], body_active_before)


def test_spawn_round_rows_matches_promoted_3p_present_absent_fixture():
    state = _spawn_state(
        map_size=[95.0],
        present=[[True, False, True]],
        random_tape_values=[
            [
                0.25,
                0.5,
                0.25,
                0.75,
                0.5,
                0.75,
            ]
        ],
        death_capacity=3,
    )
    state["pos"][0, 1] = [vector_spawn.SOURCE_AVATAR_RADIUS, vector_spawn.SOURCE_AVATAR_RADIUS]

    info = vector_spawn.spawn_round_rows(
        state,
        np.asarray([True], dtype=bool),
        player_count=3,
    )

    np.testing.assert_allclose(
        state["pos"][0],
        [[68.575, 47.5], [0.6, 0.6], [26.425, 47.5]],
    )
    np.testing.assert_allclose(
        state["heading"][0],
        [0.75 * math.tau, 0.0, 0.25 * math.tau],
    )
    np.testing.assert_array_equal(state["alive"][0], np.asarray([True, False, True], dtype=bool))
    assert int(state["random_tape_cursor"][0]) == 6
    assert int(state["random_tape_draw_count"][0]) == 6
    np.testing.assert_array_equal(
        info["spawned_player_mask"],
        np.asarray([[True, False, True]], dtype=bool),
    )
    np.testing.assert_array_equal(
        info["absent_player_mask"],
        np.asarray([[False, True, False]], dtype=bool),
    )
    assert int(state["death_count"][0]) == 1
    np.testing.assert_array_equal(state["death_player"][0], np.asarray([1, -1, -1], dtype=np.int16))
    assert [call["player"] for call in info["random_calls"]] == [2, 2, 2, 0, 0, 0]


def test_spawn_round_rows_matches_promoted_4p_reverse_spawn_order_fixture():
    state = _spawn_state(
        map_size=[101.0],
        present=[[True, True, True, True]],
        random_tape_values=[
            [
                0.2,
                0.5,
                0.25,
                0.4,
                0.5,
                0.015915494309189534,
                0.6,
                0.5,
                0.5,
                0.8,
                0.5,
                0.75,
            ]
        ],
    )

    info = vector_spawn.spawn_round_rows(
        state,
        np.asarray([True], dtype=bool),
        player_count=4,
    )

    np.testing.assert_array_equal(info["spawn_order"], np.asarray([3, 2, 1, 0], dtype=np.int16))
    np.testing.assert_allclose(
        state["pos"][0],
        [[77.41, 50.5], [59.47, 50.5], [41.53, 50.5], [23.59, 50.5]],
    )
    np.testing.assert_allclose(
        state["heading"][0],
        [0.75 * math.tau, 0.5 * math.tau, 0.015915494309189534 * math.tau, 0.25 * math.tau],
    )
    np.testing.assert_array_equal(
        state["alive"][0],
        np.asarray([True, True, True, True], dtype=bool),
    )
    assert int(state["random_tape_cursor"][0]) == 12
    assert int(state["random_tape_draw_count"][0]) == 12
    assert [call["player"] for call in info["random_calls"]] == [
        3,
        3,
        3,
        2,
        2,
        2,
        1,
        1,
        1,
        0,
        0,
        0,
    ]
    assert [call["site"] for call in info["random_calls"]] == [
        "spawn.position_x",
        "spawn.position_y",
        "spawn.angle_attempt_0",
        "spawn.position_x",
        "spawn.position_y",
        "spawn.angle_attempt_0",
        "spawn.position_x",
        "spawn.position_y",
        "spawn.angle_attempt_0",
        "spawn.position_x",
        "spawn.position_y",
        "spawn.angle_attempt_0",
    ]
