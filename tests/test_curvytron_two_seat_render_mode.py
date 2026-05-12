from types import SimpleNamespace

import numpy as np
import pytest

from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
)
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BROWSER_LINES
from curvyzero.env.vector_visual_observation import normalize_source_state_gray64
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import (
    render_source_state_canvas_gray64_player_perspectives,
)
from curvyzero.training import curvytron_two_seat_lightzero_train as train_smoke
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    SourceStateGray64Stack4,
    player_perspective_rgb_palette,
)


def _small_source_state() -> dict[str, np.ndarray]:
    return {
        "tick": np.asarray([7], dtype=np.int32),
        "elapsed_ms": np.asarray([2100.0], dtype=np.float64),
        "map_size": np.asarray([64.0], dtype=np.float64),
        "present": np.asarray([[True, True]], dtype=bool),
        "alive": np.asarray([[True, True]], dtype=bool),
        "pos": np.asarray([[[10.0, 10.0], [42.0, 18.0]]], dtype=np.float64),
        "radius": np.asarray([[1.0, 1.0]], dtype=np.float64),
        "avatar_color": np.asarray([[3, 1]], dtype=np.int16),
        "body_active": np.asarray([[True, True, True, False]], dtype=bool),
        "body_pos": np.asarray(
            [[[8.0, 10.0], [40.0, 18.0], [44.0, 18.0], [0.0, 0.0]]],
            dtype=np.float64,
        ),
        "body_radius": np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
        "body_owner": np.asarray([[0, 1, 1, -1]], dtype=np.int16),
        "body_write_cursor": np.asarray([3], dtype=np.int32),
        "done": np.asarray([False], dtype=bool),
        "terminated": np.asarray([False], dtype=bool),
        "truncated": np.asarray([False], dtype=bool),
        "terminal_reason": np.asarray([0], dtype=np.int16),
    }


def test_two_seat_stack_defaults_to_browser_lines_rgb_to_gray_player_perspective():
    state = _small_source_state()
    env = SimpleNamespace(batch_size=1, player_count=2, state=state)
    stack = SourceStateGray64Stack4(batch_size=1, player_count=2)

    observation = stack.update(env)
    expected_player_0 = normalize_source_state_gray64(
        render_source_state_canvas_gray64(
            state,
            row=0,
            player_rgb=player_perspective_rgb_palette(
                state,
                row=0,
                controlled_player=0,
                player_count=2,
            ),
            trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
        )
    )[0]
    expected_player_1 = normalize_source_state_gray64(
        render_source_state_canvas_gray64(
            state,
            row=0,
            player_rgb=player_perspective_rgb_palette(
                state,
                row=0,
                controlled_player=1,
                player_count=2,
            ),
            trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
        )
    )[0]

    assert stack.render_metadata()["trail_render_mode"] == TRAIL_RENDER_MODE_BROWSER_LINES
    assert stack.render_metadata()["single_frame_render_api"] == (
        "render_source_state_canvas_gray64"
    )
    assert (
        stack.render_metadata()["rgb_source_frame_size"]
        == SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    )
    assert stack.render_metadata()["downsample_target_frame_size"] == 64
    assert observation.shape == (1, 2, 4, 64, 64)
    assert np.array_equal(observation[0, 0, -1], expected_player_0)
    assert np.array_equal(observation[0, 1, -1], expected_player_1)
    assert float(np.max(np.abs(observation[0, 0] - observation[0, 1]))) > 0.0


@pytest.mark.parametrize(
    "trail_render_mode",
    [TRAIL_RENDER_MODE_BROWSER_LINES, TRAIL_RENDER_MODE_BODY_CIRCLES_FAST],
)
def test_two_seat_perspective_reuse_matches_independent_renders(trail_render_mode):
    state = _small_source_state()
    state["visual_trail_active"] = np.asarray([[True, True, True, True]], dtype=bool)
    state["visual_trail_write_cursor"] = np.asarray([4], dtype=np.int32)
    state["visual_trail_pos"] = np.asarray(
        [[[7.5, 10.0], [13.5, 11.0], [40.0, 18.0], [45.0, 18.5]]],
        dtype=np.float64,
    )
    state["visual_trail_radius"] = np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    state["visual_trail_owner"] = np.asarray([[0, 0, 1, 1]], dtype=np.int16)
    state["visual_trail_break_before"] = np.asarray([[False, False, True, False]], dtype=bool)
    state["bonus_active"] = np.asarray([[True, True]], dtype=bool)
    state["bonus_pos"] = np.asarray([[[24.0, 24.0], [48.0, 48.0]]], dtype=np.float64)
    state["bonus_radius"] = np.asarray([[1.5, 2.0]], dtype=np.float64)
    state["bonus_type"] = np.asarray([[1, 10]], dtype=np.int16)

    palettes = [
        player_perspective_rgb_palette(
            state,
            row=0,
            controlled_player=player,
            player_count=2,
        )
        for player in range(2)
    ]
    reused = render_source_state_canvas_gray64_player_perspectives(
        state,
        row=0,
        player_rgbs=palettes,
        trail_render_mode=trail_render_mode,
    )
    independent = np.stack(
        [
            render_source_state_canvas_gray64(
                state,
                row=0,
                player_rgb=palettes[player],
                trail_render_mode=trail_render_mode,
            )
            for player in range(2)
        ],
        axis=0,
    )

    assert np.array_equal(reused, independent)


def test_two_seat_perspective_reuse_falls_back_for_ambiguous_base_palette():
    state = _small_source_state()
    duplicate_palette = (
        (96, 96, 96),
        (96, 96, 96),
    )
    palettes = (duplicate_palette, duplicate_palette)
    reused = render_source_state_canvas_gray64_player_perspectives(
        state,
        row=0,
        player_rgbs=palettes,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )
    independent = np.stack(
        [
            render_source_state_canvas_gray64(
                state,
                row=0,
                player_rgb=palettes[player],
                trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
            )
            for player in range(2)
        ],
        axis=0,
    )

    assert np.array_equal(reused, independent)


def test_two_seat_stack_accepts_fast_render_mode_and_rejects_unknown_mode():
    stack = SourceStateGray64Stack4(
        batch_size=1,
        player_count=2,
        trail_render_mode=TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    )

    assert stack.render_metadata()["trail_render_mode"] == TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    assert stack.render_metadata()["trail_renderer_is_approximation"] is True
    with pytest.raises(ValueError, match="trail_render_mode"):
        SourceStateGray64Stack4(batch_size=1, player_count=2, trail_render_mode="mystery")


def test_two_seat_stack_render_mode_changes_trail_geometry():
    state = _small_source_state()
    state["present"][:, :] = False
    state["alive"][:, :] = False
    state["body_active"][:, :] = False
    state["body_active"][0, :2] = True
    state["body_pos"][0, :2] = np.asarray([[8.0, 10.0], [18.0, 10.0]])
    state["body_radius"][0, :2] = np.asarray([0.5, 0.5])
    state["body_owner"][0, :2] = np.asarray([0, 0], dtype=np.int16)
    state["body_write_cursor"][0] = 2
    env = SimpleNamespace(batch_size=1, player_count=2, state=state)

    browser_observation = SourceStateGray64Stack4(
        batch_size=1,
        player_count=2,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    ).update(env)
    fast_observation = SourceStateGray64Stack4(
        batch_size=1,
        player_count=2,
        trail_render_mode=TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    ).update(env)

    assert float(np.max(np.abs(browser_observation - fast_observation))) > 0.0


def test_two_seat_runner_records_render_mode_when_lightzero_is_blocked(monkeypatch):
    monkeypatch.setattr(
        train_smoke,
        "_build_lightzero_policy",
        lambda **_kwargs: {"status": "blocked", "reason": "test policy unavailable"},
    )

    result = train_smoke.run_curvytron_two_seat_lightzero_train_smoke(
        batch_size=1,
        steps=1,
        outer_iterations=1,
        death_mode="profile_no_death",
        trail_render_mode=TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
        require_installed_lightzero=False,
    )

    assert result["ok"] is False
    assert result["inputs"]["trail_render_mode"] == TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    assert result["inputs"]["death_mode"] == "profile_no_death"
    assert result["inputs"]["death_suppression_for_profile"] is True
    assert result["surface"]["single_frame_schema_id"] == (
        SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
    )
    assert result["surface"]["render"]["trail_render_mode"] == (
        TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
    )
