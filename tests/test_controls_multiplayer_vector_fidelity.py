import math

import numpy as np
import pytest

from curvyzero.env.trainer_contract import ACTION_ID_TO_SOURCE_MOVE
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv


SEED = 20260513


@pytest.mark.parametrize("player_count", [3, 4])
def test_public_vector_3p_4p_one_source_frame_control_change_matrix(
    player_count: int,
) -> None:
    for player in range(player_count):
        for action_id, native_move in enumerate(ACTION_ID_TO_SOURCE_MOVE):
            env = _controlled_multiplayer_env(
                player_count=player_count,
                decision_source_frames=1,
            )
            before_pos = env.state["pos"][0].copy()
            before_heading = env.state["heading"][0].copy()
            actions = np.ones((1, player_count), dtype=np.int16)
            actions[0, player] = action_id
            expected_moves = np.zeros((1, player_count), dtype=np.int8)
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

            other_players = [other for other in range(player_count) if other != player]
            np.testing.assert_allclose(
                env.state["heading"][0, other_players],
                before_heading[other_players],
            )
            assert np.linalg.norm(env.state["pos"][0, player] - before_pos[player]) > 0.0


def test_public_vector_4p_held_control_over_decision_source_frames() -> None:
    player_count = 4
    decision_source_frames = 4
    actions = np.asarray([[0, 1, 2, 0]], dtype=np.int16)
    expected_moves = np.asarray([[-1, 0, 1, -1]], dtype=np.int8)
    bulk_env = _controlled_multiplayer_env(
        player_count=player_count,
        decision_source_frames=decision_source_frames,
    )
    repeated_env = _controlled_multiplayer_env(
        player_count=player_count,
        decision_source_frames=1,
    )

    bulk_batch = bulk_env.step(actions)
    for _ in range(decision_source_frames):
        repeated_env.step(actions)

    np.testing.assert_array_equal(bulk_batch.info["source_moves"], expected_moves)
    np.testing.assert_array_equal(
        bulk_batch.info["action_sidecar"]["native_control_value"],
        expected_moves,
    )
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


def _controlled_multiplayer_env(
    *,
    player_count: int,
    decision_source_frames: int,
) -> VectorMultiplayerEnv:
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=player_count,
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
    player_count = env.player_count
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
    state["alive"][:, :player_count] = True
    state["present"][:, :player_count] = True
    state["printing"][:, :player_count] = False
    state["print_manager_active"][:, :player_count] = False
    state["print_manager_distance"][:, :player_count] = 999.0
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

    state["pos"][0, :player_count] = _safe_positions(player_count)
    state["heading"][0, :player_count] = np.asarray(
        [0.25, 1.75, 3.0, 4.2][:player_count],
        dtype=np.float64,
    )
    state["prev_pos"][0, :player_count] = state["pos"][0, :player_count]
    state["print_manager_last_pos"][0, :player_count] = state["pos"][0, :player_count]


def _safe_positions(player_count: int) -> np.ndarray:
    return np.asarray(
        [
            [25.0, 30.0],
            [70.0, 65.0],
            [115.0, 30.0],
            [145.0, 85.0],
        ][:player_count],
        dtype=np.float64,
    )


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
