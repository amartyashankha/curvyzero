from types import SimpleNamespace

import numpy as np
import pytest

from curvyzero.env.vector_visual_observation import SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
)
from curvyzero.env.vector_visual_observation import SourceStateBrowserLineTrailLayerCache
from curvyzero.env.vector_visual_observation import SourceStateCanvasGray64DirtyRenderCache
from curvyzero.env.vector_visual_observation import SourceStateGray64DownsampleScratch
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BODY_CIRCLES_FAST
from curvyzero.env.vector_visual_observation import TRAIL_RENDER_MODE_BROWSER_LINES
from curvyzero.env.vector_visual_observation import normalize_source_state_gray64
from curvyzero.env.vector_visual_observation import render_source_state_canvas_gray64
from curvyzero.env.vector_visual_observation import (
    render_source_state_canvas_gray64_player_perspectives,
)
from curvyzero.env.vector_visual_observation import render_source_state_rgb_canvas_like
from curvyzero.env.vector_visual_observation import rgb_canvas_like_to_gray64
from curvyzero.training import curvytron_two_seat_lightzero_train_smoke as train_smoke
from curvyzero.training.curvytron_current_policy_selfplay_smoke import (
    STACK_RENDER_MODE_FAST_GRAY64_DIRECT,
    SourceStateGray64Stack4,
    player_perspective_rgb_palette,
)
from curvyzero.training.curvyzero_stacked_debug_visual_survival_profile import (
    _strip_large_arrays,
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


def test_two_seat_fast_gray64_direct_uses_visual_trail_and_player_perspective():
    state = _small_source_state()
    state["visual_trail_active"] = np.asarray([[True, True]], dtype=bool)
    state["visual_trail_write_cursor"] = np.asarray([2], dtype=np.int32)
    state["visual_trail_pos"] = np.asarray([[[48.0, 8.0], [48.0, 24.0]]], dtype=np.float64)
    state["visual_trail_radius"] = np.asarray([[1.0, 1.0]], dtype=np.float64)
    state["visual_trail_owner"] = np.asarray([[0, 0]], dtype=np.int16)
    state["visual_trail_break_before"] = np.asarray([[True, False]], dtype=bool)
    state["bonus_active"] = np.asarray([[True, True]], dtype=bool)
    state["bonus_pos"] = np.asarray([[[24.0, 24.0], [30.0, 30.0]]], dtype=np.float64)
    state["bonus_radius"] = np.asarray([[1.0, 1.0]], dtype=np.float64)
    state["bonus_type"] = np.asarray([[3, 11]], dtype=np.int16)
    env = SimpleNamespace(batch_size=1, player_count=2, state=state)
    stack = SourceStateGray64Stack4(
        batch_size=1,
        player_count=2,
        trail_render_mode=STACK_RENDER_MODE_FAST_GRAY64_DIRECT,
    )

    observation = stack.update(env)

    def cell(x: float, y: float) -> tuple[int, int]:
        return (
            int(np.clip(np.rint((y / 64.0) * 63.0), 0, 63)),
            int(np.clip(np.rint((x / 64.0) * 63.0), 0, 63)),
        )

    trail_y, trail_x = cell(48.0, 8.0)
    self_head_y, self_head_x = cell(10.0, 10.0)
    other_head_y, other_head_x = cell(42.0, 18.0)
    bonus_a_y, bonus_a_x = cell(24.0, 24.0)
    bonus_b_y, bonus_b_x = cell(30.0, 30.0)

    assert stack.render_metadata()["trail_render_mode"] == STACK_RENDER_MODE_FAST_GRAY64_DIRECT
    assert stack.render_metadata()["rgb_to_gray64"] is False
    assert stack.render_metadata()["trail_renderer_is_approximation"] is True
    assert stack.render_metadata()["bonus_renderer_kind"] == "bonus_type_luma_circle"
    assert observation.shape == (1, 2, 4, 64, 64)
    assert observation[0, 0, -1, trail_y, trail_x] == pytest.approx(96.0 / 255.0)
    assert observation[0, 1, -1, trail_y, trail_x] == pytest.approx(128.0 / 255.0)
    assert observation[0, 0, -1, self_head_y, self_head_x] == pytest.approx(96.0 / 255.0)
    assert observation[0, 1, -1, self_head_y, self_head_x] == pytest.approx(128.0 / 255.0)
    assert observation[0, 0, -1, other_head_y, other_head_x] == pytest.approx(128.0 / 255.0)
    assert observation[0, 1, -1, other_head_y, other_head_x] == pytest.approx(96.0 / 255.0)
    assert observation[0, 0, -1, bonus_a_y, bonus_a_x] == pytest.approx(106.0 / 255.0)
    assert observation[0, 1, -1, bonus_a_y, bonus_a_x] == pytest.approx(106.0 / 255.0)
    assert observation[0, 0, -1, bonus_b_y, bonus_b_x] == pytest.approx(218.0 / 255.0)
    assert observation[0, 1, -1, bonus_b_y, bonus_b_x] == pytest.approx(218.0 / 255.0)
    assert float(np.max(np.abs(observation[0, 0] - observation[0, 1]))) > 0.0


def test_two_seat_fast_gray64_direct_semantic_mask_approximates_browser_reference():
    state = _small_source_state()
    state["visual_trail_active"] = np.asarray([[True, True, True, True, True, True]], dtype=bool)
    state["visual_trail_write_cursor"] = np.asarray([6], dtype=np.int32)
    state["visual_trail_pos"] = np.asarray(
        [[[8.0, 10.0], [12.0, 10.0], [16.0, 10.0], [40.0, 18.0], [44.0, 18.0], [48.0, 18.0]]],
        dtype=np.float64,
    )
    state["visual_trail_radius"] = np.asarray([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float64)
    state["visual_trail_owner"] = np.asarray([[0, 0, 0, 1, 1, 1]], dtype=np.int16)
    state["visual_trail_break_before"] = np.asarray(
        [[True, False, False, True, False, False]],
        dtype=bool,
    )
    state["bonus_active"] = np.asarray([[True]], dtype=bool)
    state["bonus_pos"] = np.asarray([[[24.0, 24.0]]], dtype=np.float64)
    state["bonus_radius"] = np.asarray([[1.0]], dtype=np.float64)
    state["bonus_type"] = np.asarray([[7]], dtype=np.int16)
    env = SimpleNamespace(batch_size=1, player_count=2, state=state)

    browser = SourceStateGray64Stack4(
        batch_size=1,
        player_count=2,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    ).update(env)
    fast = SourceStateGray64Stack4(
        batch_size=1,
        player_count=2,
        trail_render_mode=STACK_RENDER_MODE_FAST_GRAY64_DIRECT,
    ).update(env)

    threshold = 40.0 / 255.0
    browser_mask = browser[0, 0, -1] > threshold
    fast_mask = fast[0, 0, -1] > threshold
    assert int(browser_mask.sum()) > 0
    assert int(fast_mask.sum()) > 0

    def dilate(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        result = np.zeros_like(mask, dtype=bool)
        for y_offset in range(3):
            for x_offset in range(3):
                result |= padded[y_offset : y_offset + 64, x_offset : x_offset + 64]
        return result

    fast_near = dilate(fast_mask)
    browser_near = dilate(browser_mask)
    browser_recall = float((browser_mask & fast_near).sum()) / float(browser_mask.sum())
    fast_recall = float((fast_mask & browser_near).sum()) / float(fast_mask.sum())
    foreground_ratio = float(fast_mask.sum()) / float(browser_mask.sum())
    assert browser_recall > 0.35
    assert fast_recall > 0.35
    assert 0.25 < foreground_ratio < 2.5


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


def test_two_seat_perspective_reuse_accepts_trail_layer_cache_on_visual_trails():
    state = _small_source_state()
    state["visual_trail_active"] = np.asarray([[True, True, True, True, True]], dtype=bool)
    state["visual_trail_write_cursor"] = np.asarray([2], dtype=np.int32)
    state["visual_trail_pos"] = np.asarray(
        [[[8.0, 8.0], [20.0, 12.0], [32.0, 16.0], [44.0, 20.0], [56.0, 24.0]]],
        dtype=np.float64,
    )
    state["visual_trail_radius"] = np.asarray([[0.8, 0.8, 1.2, 1.2, 1.2]], dtype=np.float64)
    state["visual_trail_owner"] = np.asarray([[0, 0, 0, 1, 1]], dtype=np.int16)
    state["visual_trail_break_before"] = np.asarray(
        [[False, False, False, True, False]],
        dtype=bool,
    )
    palettes = [
        player_perspective_rgb_palette(
            state,
            row=0,
            controlled_player=player,
            player_count=2,
        )
        for player in range(2)
    ]
    cache = SourceStateBrowserLineTrailLayerCache(min_active_slots=1)
    scratch = SourceStateGray64DownsampleScratch()

    for cursor in (2, 3, 4, 5):
        state["visual_trail_write_cursor"][0] = cursor
        cached = render_source_state_canvas_gray64_player_perspectives(
            state,
            row=0,
            player_rgbs=palettes,
            trail_cache=cache,
            downsample_scratch=scratch,
            trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
        ).copy()
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
        assert np.array_equal(cached, independent)

    assert cache.stats.rebuilds == 1
    assert cache.stats.incremental_updates >= 1
    assert cache.stats.fallback_full_renders == 0
    assert cache.stats.mask_recolors >= 1


def test_two_seat_dirty_render_cache_matches_full_render_with_sprites():
    state = _small_source_state()
    state["visual_trail_active"] = np.asarray([[True, True, True, True]], dtype=bool)
    state["visual_trail_write_cursor"] = np.asarray([2], dtype=np.int32)
    state["visual_trail_pos"] = np.asarray(
        [[[8.0, 8.0], [20.0, 12.0], [32.0, 16.0], [44.0, 20.0]]],
        dtype=np.float64,
    )
    state["visual_trail_radius"] = np.asarray([[0.8, 0.8, 1.2, 1.2]], dtype=np.float64)
    state["visual_trail_owner"] = np.asarray([[0, 0, 1, 1]], dtype=np.int16)
    state["visual_trail_break_before"] = np.asarray([[False, False, True, False]], dtype=bool)
    state["bonus_active"] = np.asarray([[True]], dtype=bool)
    state["bonus_pos"] = np.asarray([[[24.0, 24.0]]], dtype=np.float64)
    state["bonus_radius"] = np.asarray([[1.5]], dtype=np.float64)
    state["bonus_type"] = np.asarray([[1]], dtype=np.int16)
    palettes = [
        player_perspective_rgb_palette(
            state,
            row=0,
            controlled_player=player,
            player_count=2,
        )
        for player in range(2)
    ]
    trail_cache = SourceStateBrowserLineTrailLayerCache(min_active_slots=1)
    dirty_cache = SourceStateCanvasGray64DirtyRenderCache(player_count=2)

    render_source_state_canvas_gray64_player_perspectives(
        state,
        row=0,
        player_rgbs=palettes,
        trail_cache=trail_cache,
        dirty_render_cache=dirty_cache,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )
    state["visual_trail_write_cursor"][0] = 3
    state["pos"][0, 0] = [32.0, 16.0]

    dirty = render_source_state_canvas_gray64_player_perspectives(
        state,
        row=0,
        player_rgbs=palettes,
        trail_cache=trail_cache,
        dirty_render_cache=dirty_cache,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )
    full = np.stack(
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

    assert np.array_equal(dirty, full)
    assert dirty_cache.stats.hits >= 1
    assert dirty_cache.stats.dirty_blocks_total > 0


def test_rgb_canvas_like_to_gray64_downsample_scratch_matches_baseline():
    state = _small_source_state()
    rgb = render_source_state_rgb_canvas_like(
        state,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )
    scratch = SourceStateGray64DownsampleScratch()

    baseline = rgb_canvas_like_to_gray64(rgb)
    optimized = rgb_canvas_like_to_gray64(
        rgb,
        out=np.empty_like(baseline),
        scratch=scratch,
    )

    assert np.array_equal(optimized, baseline)


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


def test_two_seat_policy_skip_ticks_send_noop_and_do_not_emit_replay(monkeypatch):
    def fake_policy_actions_batch(
        _policy,
        observations,
        legal_action_mask,
        *,
        player_id,
        step_index,
        mode,
        temperature,
        epsilon,
    ):
        del observations, mode, temperature, epsilon
        records = []
        for row, player in enumerate(np.asarray(player_id, dtype=np.int64)):
            action = 0 if int(player) == 0 else 2
            assert bool(legal_action_mask[row, action])
            records.append(
                {
                    "ok": True,
                    "status": "ok",
                    "step_index": int(step_index),
                    "api": "fake_policy_actions_batch",
                    "action": action,
                }
            )
        return {
            "ok": True,
            "status": "ok",
            "step_index": int(step_index),
            "records": records,
        }

    monkeypatch.setattr(train_smoke, "_policy_actions_batch", fake_policy_actions_batch)

    env = train_smoke.VectorMultiplayerEnv(
        batch_size=1,
        player_count=2,
        seed=123,
        max_ticks=32,
        death_mode="profile_no_death",
        natural_bonus_spawn=False,
    )
    visual_stack = SourceStateGray64Stack4(batch_size=1, player_count=2)
    batch = env.reset(seed=123)
    observation = visual_stack.update(env)

    collection = train_smoke._collect_current_policy_iteration(
        object(),
        env,
        visual_stack,
        batch=batch,
        observation=observation,
        iteration=1,
        decision_offset=0,
        collect_steps=3,
        alive_reward=1.0,
        dead_reward=0.0,
        terminal_outcome_reward_per_step=1.0,
        bonus_pickup_reward_per_catch=0.0,
        return_target_discount=1.0,
        action_selection_mode=train_smoke.ACTION_SELECTION_MODE_COLLECT,
        collect_temperature=1.0,
        collect_epsilon=0.0,
        action_noop_probability=0.0,
        action_noise_rng=np.random.default_rng(0),
        policy_action_repeat_min=3,
        policy_action_repeat_max=3,
        policy_action_repeat_extra_probability=0.0,
        policy_action_repeat_rng=np.random.default_rng(1),
        observation_noise_std=0.0,
        observation_noise_rng=np.random.default_rng(2),
    )

    assert collection["problems"] == []
    assert collection["policy_search_row_count"] == 2
    assert len(collection["records"]) == 3
    assert len(collection["replay_rows"]) == 2
    assert {row["iteration_step"] for row in collection["replay_rows"]} == {0}
    assert all(
        row["policy_noop_skip_count_after_action"] == 2
        for row in collection["replay_rows"]
    )

    first, second, third = collection["records"]
    assert first["fresh_policy_input_rows"] == 2
    assert first["policy_noop_skip_rows"] == 0
    assert np.array_equal(first["joint_action"], np.asarray([[0, 2]], dtype=np.int16))
    for skipped in (second, third):
        assert skipped["fresh_policy_input_rows"] == 0
        assert skipped["policy_noop_skip_rows"] == 2
        assert np.array_equal(
            skipped["joint_action"],
            np.full((1, 2), train_smoke.NOOP_ACTION_ID, dtype=np.int16),
        )
        assert np.array_equal(
            skipped["executed_actions"],
            np.full(2, train_smoke.NOOP_ACTION_ID, dtype=np.int16),
        )
        assert skipped["search"] == []

    stochasticity = collection["control_stochasticity"]
    assert stochasticity["schema_id"] == "curvyzero_two_seat_policy_noop_skip/v0"
    assert stochasticity["counts"]["fresh_policy_decision_rows"] == 2
    assert stochasticity["counts"]["policy_noop_skip_rows"] == 4
    assert stochasticity["policy_noop_skip_interval_histogram"] == {"3": 2}


def test_two_seat_frozen_opponent_mix_does_not_emit_frozen_replay(monkeypatch):
    def fake_policy_actions_batch(
        _policy,
        observations,
        legal_action_mask,
        *,
        player_id,
        step_index,
        mode,
        temperature,
        epsilon,
    ):
        del observations, mode, temperature, epsilon
        players = np.asarray(player_id, dtype=np.int64)
        assert players.tolist() == [0, 1, 0]
        records = []
        for row, player in enumerate(players):
            action = 0 if int(player) == 0 else 2
            assert bool(legal_action_mask[row, action])
            records.append(
                {
                    "ok": True,
                    "status": "ok",
                    "step_index": int(step_index),
                    "api": "fake_policy_actions_batch",
                    "action": action,
                }
            )
        return {
            "ok": True,
            "status": "ok",
            "step_index": int(step_index),
            "records": records,
        }

    class FakeFrozenSelection:
        def __init__(self, actions, opponent_mask):
            self.actions = actions
            self._opponent_mask = opponent_mask

        def sidecar(self):
            return {
                "policy_id": "fake_frozen_checkpoint",
                "actions": self.actions.copy(),
                "opponent_mask": self._opponent_mask.copy(),
            }

    class FakeFrozenPolicy:
        def select_actions(
            self,
            legal_action_mask,
            opponent_mask,
            *,
            decision_index,
            observation,
        ):
            del decision_index, observation
            expected = np.asarray([[False, False], [False, True]], dtype=bool)
            np.testing.assert_array_equal(opponent_mask, expected)
            assert bool(legal_action_mask[1, 1, 2])
            actions = np.full(opponent_mask.shape, train_smoke.NOOP_ACTION_ID, dtype=np.int16)
            actions[opponent_mask] = 2
            return FakeFrozenSelection(actions, opponent_mask)

    monkeypatch.setattr(train_smoke, "_policy_actions_batch", fake_policy_actions_batch)

    env = train_smoke.VectorMultiplayerEnv(
        batch_size=2,
        player_count=2,
        seed=123,
        max_ticks=32,
        death_mode="profile_no_death",
        natural_bonus_spawn=False,
    )
    visual_stack = SourceStateGray64Stack4(batch_size=2, player_count=2)
    batch = env.reset(seed=123)
    observation = visual_stack.update(env)

    collection = train_smoke._collect_current_policy_iteration(
        object(),
        env,
        visual_stack,
        batch=batch,
        observation=observation,
        iteration=1,
        decision_offset=0,
        collect_steps=1,
        alive_reward=1.0,
        dead_reward=0.0,
        terminal_outcome_reward_per_step=1.0,
        bonus_pickup_reward_per_catch=0.0,
        return_target_discount=1.0,
        action_selection_mode=train_smoke.ACTION_SELECTION_MODE_COLLECT,
        collect_temperature=1.0,
        collect_epsilon=0.0,
        action_noop_probability=0.0,
        action_noise_rng=np.random.default_rng(0),
        policy_action_repeat_min=1,
        policy_action_repeat_max=1,
        policy_action_repeat_extra_probability=0.0,
        policy_action_repeat_rng=np.random.default_rng(1),
        observation_noise_std=0.0,
        observation_noise_rng=np.random.default_rng(2),
        frozen_opponent_policy=FakeFrozenPolicy(),
        frozen_opponent_row_mask=np.asarray([False, True], dtype=bool),
        frozen_opponent_probability=0.5,
        frozen_opponent_mix_rng=np.random.default_rng(3),
        frozen_opponent_player_id=1,
        frozen_opponent_metadata={
            "enabled": True,
            "checkpoint_ref": "training/example/checkpoints/lightzero/iteration_50.pth.tar",
            "snapshot_ref": "example_snapshot",
        },
    )

    assert collection["problems"] == []
    assert collection["policy_search_row_count"] == 3
    assert len(collection["replay_rows"]) == 3
    assert all(
        row["action_source"] == train_smoke.ACTION_SOURCE_CURRENT_POLICY
        for row in collection["replay_rows"]
    )
    assert all(row["learner_controlled"] is True for row in collection["replay_rows"])
    mixed_rows = [
        row
        for row in collection["replay_rows"]
        if row["rollout_kind"] == train_smoke.ROLLOUT_KIND_CURRENT_POLICY_VS_FROZEN
    ]
    assert len(mixed_rows) == 1
    assert mixed_rows[0]["env_row_id"] == 1
    assert mixed_rows[0]["player_id"] == 0

    record = collection["records"][0]
    assert record["joint_action"][1, 1] == 2
    assert record["frozen_opponent_slot_mask"][1, 1]
    assert not record["current_policy_live_mask"][1, 1]
    assert (
        record["action_source_by_slot"][1, 1]
        == train_smoke.ACTION_SOURCE_FROZEN_CHECKPOINT
    )
    assert collection["opponent_mix"]["current_policy_vs_frozen_rows"] == 1


def test_two_seat_replay_sampler_rejects_frozen_rows():
    with pytest.raises(ValueError, match="current-policy controlled"):
        train_smoke._sample_replay_batch(
            [
                {
                    "action_source": train_smoke.ACTION_SOURCE_FROZEN_CHECKPOINT,
                    "learner_controlled": False,
                }
            ]
        )


def test_two_seat_replay_sampler_rejects_missing_control_metadata():
    with pytest.raises(ValueError, match="current-policy controlled"):
        train_smoke._sample_replay_batch([{"action": 1}])


def test_frozen_opponent_reset_resampling_does_not_force_singletons_frozen():
    original = np.asarray([True, False, False], dtype=bool)
    resampled = train_smoke._resample_frozen_opponent_rows(
        original,
        row_mask=np.asarray([False, True, False], dtype=bool),
        probability=0.25,
        rng=np.random.default_rng(0),
        enabled=True,
    )

    assert resampled.tolist() == [True, False, False]


def test_frozen_opponent_reset_resampling_can_freeze_singletons_by_probability():
    original = np.asarray([False, False, False], dtype=bool)
    resampled = train_smoke._resample_frozen_opponent_rows(
        original,
        row_mask=np.asarray([False, True, False], dtype=bool),
        probability=1.0,
        rng=np.random.default_rng(0),
        enabled=True,
    )

    assert resampled.tolist() == [False, True, False]


def test_strip_large_arrays_handles_string_metadata_arrays():
    summary = _strip_large_arrays(
        {
            "rollout_kind_by_row": np.asarray(
                [
                    train_smoke.ROLLOUT_KIND_CURRENT_POLICY_SELFPLAY,
                    train_smoke.ROLLOUT_KIND_CURRENT_POLICY_VS_FROZEN,
                ]
            )
        }
    )

    assert summary["rollout_kind_by_row"]["shape"] == [2]
    assert summary["rollout_kind_by_row"]["dtype"].startswith("<U")
    assert summary["rollout_kind_by_row"]["sample"] == [
        train_smoke.ROLLOUT_KIND_CURRENT_POLICY_SELFPLAY,
        train_smoke.ROLLOUT_KIND_CURRENT_POLICY_VS_FROZEN,
    ]
