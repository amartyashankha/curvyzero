import numpy as np

from curvyzero.env import vector_runtime
from curvyzero.env.vector_multiplayer_env import VectorMultiplayerEnv
from curvyzero.env.vector_visual_observation import SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import render_source_state_rgb_canvas_like
from curvyzero.env.vector_visual_observation import rgb_canvas_like_to_gray64


def test_bonus_all_color_changes_browser_like_rgb_to_gray64_and_restores_on_expiry():
    env = VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        decision_ms=1.0,
        body_capacity=8,
        event_capacity=16,
        timer_capacity=4,
        random_tape_capacity=8,
        event_mode="debug-event",
    )
    env.reset(
        seed=np.asarray([101], dtype=np.uint64),
        source_fixture_new_round_time_ms=0.0,
        source_fixture_warmup_advance_ms=0.0,
    )
    env.decision_ms = 0.0
    env.state["timer_active"][0] = False
    env.state["pos"][0] = np.asarray([[25.0, 50.0], [75.0, 50.0]], dtype=np.float64)
    env.state["prev_pos"][0] = env.state["pos"][0]
    env.state["alive"][0] = True
    env.state["present"][0] = True
    env.state["printing"][0] = False
    env.state["print_manager_active"][0] = False
    env.state["body_active"][0] = False
    if "visual_trail_active" in env.state:
        env.state["visual_trail_active"][0] = False

    env.seed_active_bonus(
        row=0,
        bonus_type="BonusAllColor",
        x=float(env.state["pos"][0, 0, 0]),
        y=float(env.state["pos"][0, 0, 1]),
    )
    base_state = _copied_state_without_active_bonus(env.state)
    base_rgb = render_source_state_rgb_canvas_like(base_state, row=0)
    base_gray64 = rgb_canvas_like_to_gray64(base_rgb)
    np.testing.assert_array_equal(
        base_gray64,
        render_source_state_canvas_gray64(base_state, row=0),
    )
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[0, 1]]))

    catch_batch = env.step(np.asarray([[1, 1]], dtype=np.int16))

    assert catch_batch.info["step_counters"]["bonus_all_color_catches"] == 1
    assert catch_batch.info["step_counters"]["bonus_stack_appends"] == 2
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[1, 0]]))
    caught_rgb = render_source_state_rgb_canvas_like(env.state, row=0)
    caught_gray64 = rgb_canvas_like_to_gray64(caught_rgb)
    np.testing.assert_array_equal(
        caught_gray64,
        render_source_state_canvas_gray64(env.state, row=0),
    )
    np.testing.assert_array_equal(
        _avatar_center_rgb(caught_rgb, env, player=0),
        SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[1],
    )
    np.testing.assert_array_equal(
        _avatar_center_rgb(caught_rgb, env, player=1),
        SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0],
    )
    assert int(np.count_nonzero(caught_gray64 != base_gray64)) > 0

    expiry_batch = env.step(
        np.asarray([[1, 1]], dtype=np.int16),
        timer_advance_ms=vector_runtime.BONUS_ALL_COLOR_DURATION_MS,
    )

    assert expiry_batch.info["step_counters"]["bonus_all_color_expiries"] == 2
    np.testing.assert_array_equal(env.state["avatar_color"], np.asarray([[0, 1]]))
    restored_rgb = render_source_state_rgb_canvas_like(env.state, row=0)
    restored_gray64 = rgb_canvas_like_to_gray64(restored_rgb)
    np.testing.assert_array_equal(
        restored_gray64,
        render_source_state_canvas_gray64(env.state, row=0),
    )
    np.testing.assert_array_equal(
        _avatar_center_rgb(restored_rgb, env, player=0),
        SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0],
    )
    np.testing.assert_array_equal(
        _avatar_center_rgb(restored_rgb, env, player=1),
        SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[1],
    )
    np.testing.assert_array_equal(restored_rgb, base_rgb)
    np.testing.assert_array_equal(restored_gray64, base_gray64)


def _avatar_center_rgb(
    frame: np.ndarray,
    env: VectorMultiplayerEnv,
    *,
    player: int,
) -> np.ndarray:
    map_size = float(env.state["map_size"][0])
    size = int(frame.shape[0])
    pos = env.state["pos"][0, player]
    x = int(np.clip(np.rint((float(pos[0]) / map_size) * float(size - 1)), 0, size - 1))
    y = int(np.clip(np.rint((float(pos[1]) / map_size) * float(size - 1)), 0, size - 1))
    return frame[y, x]


def _copied_state_without_active_bonus(
    state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    copied = {name: value.copy() for name, value in state.items()}
    copied["bonus_world_active"][0] = False
    copied["bonus_active"][0] = False
    copied["bonus_count"][0] = 0
    copied["bonus_world_body_count"][0] = 0
    return copied
