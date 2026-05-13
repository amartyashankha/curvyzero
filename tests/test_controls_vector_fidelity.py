import math

import numpy as np
import pytest

from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE
from curvyzero.env.vector_multiplayer_env import (
    VectorMultiplayerEnv,
    VectorMultiplayerEnvError,
)


SEED = 20260513


def test_public_vector_2p_one_source_frame_control_change_matrix() -> None:
    for player in (0, 1):
        for action_id, native_move in enumerate(ACTION_ID_TO_SOURCE_MOVE):
            env = _controlled_2p_env(decision_source_frames=1)
            before_pos = env.state["pos"][0].copy()
            before_heading = env.state["heading"][0].copy()
            actions = np.ones((1, 2), dtype=np.int16)
            actions[0, player] = action_id
            expected_moves = np.zeros((1, 2), dtype=np.int8)
            expected_moves[0, player] = native_move

            batch = env.step(actions)

            assert batch.info["source_frame_decision"] is True
            assert batch.info["decision_source_frames"] == 1
            np.testing.assert_array_equal(
                batch.info["source_physics_substeps_executed"],
                np.asarray([1], dtype=np.int32),
            )
            np.testing.assert_allclose(
                batch.info["source_physics_elapsed_ms"],
                np.asarray([env.source_physics_step_ms], dtype=np.float64),
            )
            np.testing.assert_array_equal(batch.info["source_moves"], expected_moves)
            np.testing.assert_array_equal(
                batch.info["action_sidecar"]["native_control_value"],
                expected_moves,
            )

            expected_pos, expected_heading = _expected_one_source_frame(
                env,
                pos=before_pos,
                heading=before_heading,
                source_moves=expected_moves[0],
            )
            np.testing.assert_allclose(env.state["heading"][0], expected_heading)
            np.testing.assert_allclose(env.state["pos"][0], expected_pos)

            observed_delta = env.state["heading"][0, player] - before_heading[player]
            expected_delta = (
                env.state["angular_velocity_per_ms"][0, player]
                * env.source_physics_step_ms
                * native_move
            )
            assert observed_delta == pytest.approx(expected_delta)
            if native_move == 0:
                assert observed_delta == pytest.approx(0.0)
            else:
                assert math.copysign(1.0, observed_delta) == math.copysign(
                    1.0,
                    native_move,
                )
            assert np.linalg.norm(env.state["pos"][0, player] - before_pos[player]) > 0.0


def test_public_vector_2p_held_control_over_decision_source_frames() -> None:
    decision_source_frames = 5
    actions = np.asarray([[0, 2]], dtype=np.int16)
    expected_moves = np.asarray([[-1, 1]], dtype=np.int8)
    bulk_env = _controlled_2p_env(decision_source_frames=decision_source_frames)
    repeated_env = _controlled_2p_env(decision_source_frames=1)

    bulk_batch = bulk_env.step(actions)
    for _ in range(decision_source_frames):
        repeated_env.step(actions)

    np.testing.assert_array_equal(bulk_batch.info["source_moves"], expected_moves)
    np.testing.assert_array_equal(
        bulk_batch.info["source_physics_substeps_executed"],
        np.asarray([decision_source_frames], dtype=np.int32),
    )
    np.testing.assert_allclose(
        bulk_batch.info["source_physics_elapsed_ms"],
        np.asarray(
            [decision_source_frames * bulk_env.source_physics_step_ms],
            dtype=np.float64,
        ),
    )
    assert int(bulk_env.state["tick"][0]) == decision_source_frames
    assert int(repeated_env.state["tick"][0]) == decision_source_frames
    np.testing.assert_allclose(bulk_env.state["heading"], repeated_env.state["heading"])
    np.testing.assert_allclose(bulk_env.state["pos"], repeated_env.state["pos"])
    np.testing.assert_allclose(
        bulk_env.state["elapsed_ms"],
        repeated_env.state["elapsed_ms"],
    )


@pytest.mark.parametrize(("turn_action", "native_move"), [(0, -1), (2, 1)])
def test_public_vector_2p_release_to_straight_stops_turning(
    turn_action: int,
    native_move: int,
) -> None:
    env = _controlled_2p_env(decision_source_frames=1)
    initial_heading = float(env.state["heading"][0, 0])

    turn_batch = env.step(np.asarray([[turn_action, 1]], dtype=np.int16))
    heading_after_turn = float(env.state["heading"][0, 0])
    pos_after_turn = env.state["pos"][0, 0].copy()

    np.testing.assert_array_equal(
        turn_batch.info["action_sidecar"]["native_control_value"],
        np.asarray([[native_move, 0]], dtype=np.int8),
    )
    assert heading_after_turn - initial_heading == pytest.approx(
        env.state["angular_velocity_per_ms"][0, 0]
        * env.source_physics_step_ms
        * native_move,
    )

    straight_batch = env.step(np.asarray([[1, 1]], dtype=np.int16))
    heading_after_release = float(env.state["heading"][0, 0])

    np.testing.assert_array_equal(
        straight_batch.info["action_sidecar"]["native_control_value"],
        np.asarray([[0, 0]], dtype=np.int8),
    )
    assert heading_after_release == pytest.approx(heading_after_turn)
    expected_release_pos = pos_after_turn + (
        np.asarray(
            [math.cos(heading_after_turn), math.sin(heading_after_turn)],
            dtype=np.float64,
        )
        * env.state["speed"][0, 0]
        * env.source_physics_step_ms
        / 1000.0
    )
    np.testing.assert_allclose(env.state["pos"][0, 0], expected_release_pos)
    assert np.linalg.norm(env.state["pos"][0, 0] - pos_after_turn) > 0.0


def test_public_vector_invalid_live_actions_and_inactive_noops_are_contractual() -> None:
    env = _controlled_2p_env(decision_source_frames=1)
    with pytest.raises(
        VectorMultiplayerEnvError,
        match="live present players require left/straight/right action ids",
    ):
        env.step(np.asarray([[3, 1]], dtype=np.int16))
    with pytest.raises(
        VectorMultiplayerEnvError,
        match="live present players require left/straight/right action ids",
    ):
        env.step(np.asarray([[-1, 1]], dtype=np.int16))
    with pytest.raises(VectorMultiplayerEnvError, match="actions must have shape"):
        env.step(np.asarray([[1]], dtype=np.int16))

    inactive_env = _controlled_2p_env(batch_size=2, decision_source_frames=1)
    inactive_env.state["present"][0] = np.asarray([True, False], dtype=bool)
    inactive_env.state["alive"][0] = np.asarray([True, False], dtype=bool)
    inactive_env.state["present"][1] = np.asarray([True, True], dtype=bool)
    inactive_env.state["alive"][1] = np.asarray([True, False], dtype=bool)
    inactive_pos_before = inactive_env.state["pos"][:, 1].copy()

    batch = inactive_env.step(np.asarray([[1, -1], [1, 2]], dtype=np.int16))

    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_required"],
        np.asarray([[True, False], [True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_source"],
        np.asarray(
            [
                ["external_joint_action", "absent_noop"],
                ["external_joint_action", "dead_noop"],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        batch.info["source_moves"],
        np.asarray([[0, 0], [0, 0]], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["native_control_value"],
        np.asarray([[0, 0], [0, 0]], dtype=np.int8),
    )
    np.testing.assert_allclose(inactive_env.state["pos"][:, 1], inactive_pos_before)

    invalid_inactive_env = _controlled_2p_env(decision_source_frames=1)
    invalid_inactive_env.state["present"][0, 1] = False
    invalid_inactive_env.state["alive"][0, 1] = False
    with pytest.raises(
        VectorMultiplayerEnvError,
        match="inactive action slots must be -1 or a left/straight/right action id",
    ):
        invalid_inactive_env.step(np.asarray([[1, -2]], dtype=np.int16))


def test_public_vector_terminal_padding_noop_under_decision_source_frames() -> None:
    decision_source_frames = 6
    env = _controlled_2p_env(batch_size=2, decision_source_frames=decision_source_frames)
    terminal_row = 0
    live_row = 1
    env.state["done"][terminal_row] = True
    env.state["terminated"][terminal_row] = True
    env.state["in_round"][terminal_row] = False
    env.state["alive"][terminal_row] = np.asarray([False, False], dtype=bool)
    env.state["death_count"][terminal_row] = 2
    env.state["death_player"][terminal_row, :2] = np.asarray([0, 1], dtype=np.int16)
    env._needs_reset[terminal_row] = False

    terminal_pos_before = env.state["pos"][terminal_row].copy()
    terminal_heading_before = env.state["heading"][terminal_row].copy()
    terminal_elapsed_before = float(env.state["elapsed_ms"][terminal_row])
    live_pos_before = env.state["pos"][live_row].copy()
    live_heading_before = env.state["heading"][live_row].copy()
    actions = np.asarray(
        [
            [0, 2],
            [0, 2],
        ],
        dtype=np.int16,
    )

    batch = env.step(actions)

    assert batch.done[terminal_row]
    assert not batch.done[live_row]
    np.testing.assert_array_equal(
        batch.info["terminal_rows"],
        np.asarray([terminal_row], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        batch.info["source_moves"],
        np.asarray(
            [
                [0, 0],
                [-1, 1],
            ],
            dtype=np.int8,
        ),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_required"],
        np.asarray(
            [
                [False, False],
                [True, True],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["action_source"],
        np.asarray(
            [
                ["terminal_padding", "terminal_padding"],
                ["external_joint_action", "external_joint_action"],
            ],
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["native_control_value"],
        np.asarray(
            [
                [0, 0],
                [-1, 1],
            ],
            dtype=np.int8,
        ),
    )
    np.testing.assert_array_equal(
        batch.info["action_sidecar"]["player_action_mask"][terminal_row],
        np.zeros((2, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        batch.info["source_physics_substeps_executed"],
        np.asarray([0, decision_source_frames], dtype=np.int32),
    )
    np.testing.assert_allclose(
        batch.info["source_physics_elapsed_ms"],
        np.asarray(
            [
                0.0,
                decision_source_frames * env.source_physics_step_ms,
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(env.state["pos"][terminal_row], terminal_pos_before)
    np.testing.assert_allclose(
        env.state["heading"][terminal_row],
        terminal_heading_before,
    )
    assert float(env.state["elapsed_ms"][terminal_row]) == pytest.approx(
        terminal_elapsed_before,
    )

    repeated_env = _controlled_2p_env(
        batch_size=1,
        decision_source_frames=1,
    )
    repeated_env.state["pos"][0] = live_pos_before
    repeated_env.state["heading"][0] = live_heading_before
    repeated_env.state["prev_pos"][0] = live_pos_before
    repeated_env.state["print_manager_last_pos"][0] = live_pos_before
    for _ in range(decision_source_frames):
        repeated_env.step(actions[live_row : live_row + 1])

    np.testing.assert_allclose(
        env.state["pos"][live_row],
        repeated_env.state["pos"][0],
    )
    np.testing.assert_allclose(
        env.state["heading"][live_row],
        repeated_env.state["heading"][0],
    )
    assert int(env.state["tick"][live_row]) == decision_source_frames
    assert int(env.state["tick"][terminal_row]) == 0


def _controlled_2p_env(
    *,
    batch_size: int = 1,
    decision_source_frames: int,
) -> VectorMultiplayerEnv:
    env = VectorMultiplayerEnv(
        batch_size=batch_size,
        player_count=2,
        seed=SEED,
        decision_source_frames=decision_source_frames,
        natural_bonus_spawn=False,
        body_capacity=64,
        event_capacity=16,
        random_tape_capacity=64,
    )
    env.reset(
        seed=SEED,
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    _install_safe_control_state(env)
    return env


def _install_safe_control_state(env: VectorMultiplayerEnv) -> None:
    state = env.state
    batch_size = env.batch_size
    state["timer_active"][:] = False
    state["timer_remaining_ms"][:] = 0.0
    state["timer_kind"][:] = 0
    state["timer_seq"][:] = 0
    state["timer_overflow"][:] = False
    state["done"][:] = False
    state["terminated"][:] = False
    state["truncated"][:] = False
    state["reset_pending"][:] = False
    state["overflow"][:] = False
    state["started"][:] = True
    state["in_round"][:] = True
    state["world_active"][:] = True
    state["world_body_count"][:] = 0
    state["alive"][:, :2] = True
    state["present"][:, :2] = True
    state["printing"][:, :2] = False
    state["print_manager_active"][:, :2] = False
    state["print_manager_distance"][:, :2] = 999.0
    state["body_active"][:] = False
    state["body_write_cursor"][:] = 0
    state["body_count"][:] = 0
    state["live_body_num"][:] = 0
    state["visual_trail_active"][:] = False
    state["visual_trail_write_cursor"][:] = 0
    state["visible_trail_count"][:] = 0
    state["has_visible_trail_last"][:] = False
    state["has_draw_cursor"][:] = False
    state["death_count"][:] = 0
    state["death_player"][:] = -1

    for row in range(batch_size):
        state["pos"][row] = np.asarray(
            [[25.0 + row, 30.0 + row], [70.0 + row, 65.0 + row]],
            dtype=np.float64,
        )
        state["heading"][row] = np.asarray([0.25, 1.75], dtype=np.float64)
    state["prev_pos"][:, :2] = state["pos"][:, :2]
    state["print_manager_last_pos"][:, :2] = state["pos"][:, :2]


def _expected_one_source_frame(
    env: VectorMultiplayerEnv,
    *,
    pos: np.ndarray,
    heading: np.ndarray,
    source_moves: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    step_ms = env.source_physics_step_ms
    angular_velocity = env.state["angular_velocity_per_ms"][0].copy()
    speed = env.state["speed"][0].copy()
    expected_heading = heading + angular_velocity * step_ms * source_moves
    distance = speed * step_ms / 1000.0
    expected_pos = pos.copy()
    expected_pos[:, 0] += np.cos(expected_heading) * distance
    expected_pos[:, 1] += np.sin(expected_heading) * distance
    return expected_pos, expected_heading
