import numpy as np
import pytest

from curvyzero.env import vector_runtime
from curvyzero.env.vector_visual_observation import (
    BONUS_RENDER_MODE_BROWSER_SPRITES,
    BONUS_RENDER_MODE_CIRCLES_FAST,
    BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    BONUS_SYMBOL_INNER_LUMA,
    BONUS_SYMBOL_INNER_LUMA_BY_SHAPE,
    BONUS_SYMBOL_OUTER_LUMA_BY_SHAPE,
    SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL,
    SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL,
    SOURCE_STATE_BONUS64_STACK4_GAME_BORDERLESS_CHANNEL,
    SOURCE_STATE_BONUS64_STACK4_GAME_TTL_CHANNEL,
    SOURCE_STATE_BONUS64_STACK4_MAX_BONUS_TYPE_CODE,
    SOURCE_STATE_BONUS64_STACK4_OTHER_STATUS_CHANNELS,
    SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SCHEMA_ID,
    SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SHAPE,
    SOURCE_STATE_BONUS64_STACK4_SELF_STATUS_CHANNELS,
    SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID,
    SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID,
    SOURCE_STATE_CANVAS_GRAY64_SHAPE,
    SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
    SOURCE_STATE_GRAY64_BONUS_VALUE,
    SOURCE_STATE_GRAY64_COMPARISON_TARGET,
    SOURCE_STATE_GRAY64_RENDERER_IMPL_HASH,
    SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
    SOURCE_STATE_GRAY64_SCHEMA_HASH,
    SOURCE_STATE_GRAY64_SCHEMA_ID,
    SOURCE_STATE_GRAY64_SOURCE_CLAIM_ID,
    SOURCE_STATE_GRAY64_STATE_FIELDS,
    SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_RGB,
    SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_NAMES,
    SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB,
    SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
    SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB,
    TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    TRAIL_RENDER_MODE_BROWSER_LINES,
    SourceStateGray64Renderer,
    VectorVisualObservationError,
    _bonus_sprite_index,
    _simple_bonus_symbol_base,
    _source_bonus_sprite_sheet_path,
    normalize_source_state_gray64,
    render_source_state_bonus64_stack4_player_perspective_v1,
    render_source_state_canvas_gray64,
    render_source_state_rgb_canvas_like,
    render_source_snapshot_canvas_gray64,
    render_source_snapshot_gray64,
    render_source_snapshot_rgb_canvas_like,
    render_source_state_gray64_fast_player_perspectives,
    render_source_state_gray64,
    rgb_canvas_like_to_gray64,
    source_state_bonus64_stack4_player_perspective_v1_schema,
    source_state_canvas_gray64_schema,
    source_state_gray64_metadata,
    source_state_gray64_schema,
)
from curvyzero.env import vector_visual_observation as visual_observation


def _small_source_state() -> dict[str, np.ndarray]:
    state = {
        "tick": np.asarray([7], dtype=np.int32),
        "elapsed_ms": np.asarray([2100.0], dtype=np.float64),
        "map_size": np.asarray([64.0], dtype=np.float64),
        "present": np.asarray([[True, True]], dtype=bool),
        "alive": np.asarray([[True, True]], dtype=bool),
        "pos": np.asarray([[[10.0, 10.0], [42.0, 18.0]]], dtype=np.float64),
        "radius": np.asarray([[1.0, 1.0]], dtype=np.float64),
        "body_active": np.asarray([[True, True, True, False]], dtype=bool),
        "body_pos": np.asarray(
            [[[8.0, 10.0], [40.0, 18.0], [63.0, 63.0], [0.0, 0.0]]],
            dtype=np.float64,
        ),
        "body_radius": np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.float64),
        "body_owner": np.asarray([[0, 1, 1, -1]], dtype=np.int16),
        "body_write_cursor": np.asarray([2], dtype=np.int32),
        "done": np.asarray([False], dtype=bool),
        "terminated": np.asarray([False], dtype=bool),
        "truncated": np.asarray([False], dtype=bool),
        "terminal_reason": np.asarray([0], dtype=np.int16),
    }
    return state


def _trail_only_source_state(
    points: list[tuple[float, float]],
    *,
    radii: float | list[float] = 0.0,
    owners: int | list[int] = 0,
    break_before: list[bool] | None = None,
) -> dict[str, np.ndarray]:
    state = _small_source_state()
    state["present"][:, :] = False
    state["alive"][:, :] = False
    state["body_active"][:, :] = False
    point_count = len(points)
    assert point_count <= state["body_active"].shape[1]
    state["body_write_cursor"][0] = point_count
    state["body_active"][0, :point_count] = True
    state["body_pos"][0, :point_count] = np.asarray(points, dtype=np.float64)
    if isinstance(radii, list):
        state["body_radius"][0, :point_count] = np.asarray(radii, dtype=np.float64)
    else:
        state["body_radius"][0, :point_count] = float(radii)
    if isinstance(owners, list):
        state["body_owner"][0, :point_count] = np.asarray(owners, dtype=np.int16)
    else:
        state["body_owner"][0, :point_count] = int(owners)
    if break_before is not None:
        state["body_break_before"] = np.zeros_like(state["body_active"])
        state["body_break_before"][0, :point_count] = np.asarray(
            break_before,
            dtype=bool,
        )
    return state


def _with_visual_trail_points(
    state: dict[str, np.ndarray],
    points: list[tuple[float, float]],
    *,
    radii: float | list[float] = 1.0,
    owners: int | list[int] = 0,
    break_before: list[bool] | None = None,
) -> dict[str, np.ndarray]:
    capacity = max(len(points), 1)
    state["visual_trail_active"] = np.zeros((1, capacity), dtype=bool)
    state["visual_trail_pos"] = np.zeros((1, capacity, 2), dtype=np.float64)
    state["visual_trail_radius"] = np.zeros((1, capacity), dtype=np.float64)
    state["visual_trail_owner"] = np.full((1, capacity), -1, dtype=np.int16)
    state["visual_trail_break_before"] = np.zeros((1, capacity), dtype=bool)
    state["visual_trail_write_cursor"] = np.asarray([len(points)], dtype=np.int32)
    if points:
        state["visual_trail_active"][0, : len(points)] = True
        state["visual_trail_pos"][0, : len(points)] = np.asarray(points, dtype=np.float64)
        if isinstance(radii, list):
            state["visual_trail_radius"][0, : len(points)] = np.asarray(
                radii,
                dtype=np.float64,
            )
        else:
            state["visual_trail_radius"][0, : len(points)] = float(radii)
        if isinstance(owners, list):
            state["visual_trail_owner"][0, : len(points)] = np.asarray(
                owners,
                dtype=np.int16,
            )
        else:
            state["visual_trail_owner"][0, : len(points)] = int(owners)
        if break_before is not None:
            state["visual_trail_break_before"][0, : len(points)] = np.asarray(
                break_before,
                dtype=bool,
            )
    return state


def _rgb_triplet(value: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(value, dtype=np.uint8)


def _reference_draw_world_circle(
    canvas: np.ndarray,
    x_value,
    y_value,
    radius_value,
    map_size: float,
    *,
    value: int,
) -> None:
    x = float(x_value)
    y = float(y_value)
    radius = float(radius_value)
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(radius):
        return
    if x + radius <= 0.0 or x - radius >= map_size:
        return
    if y + radius <= 0.0 or y - radius >= map_size:
        return
    radius_px = int(max(0, np.ceil((radius / map_size) * 64.0)))
    px = int(np.clip(np.rint((x / map_size) * 63.0), 0, 63))
    py = int(np.clip(np.rint((y / map_size) * 63.0), 0, 63))
    if radius_px == 0:
        canvas[py, px] = max(int(canvas[py, px]), int(value))
        return

    x0 = max(0, px - radius_px)
    x1 = min(63, px + radius_px)
    y0 = max(0, py - radius_px)
    y1 = min(63, py + radius_px)
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius_px**2
    view = canvas[y0 : y1 + 1, x0 : x1 + 1]
    view[mask] = np.maximum(view[mask], np.uint8(value))


def _reference_body_value(owner) -> int:
    owner_index = int(owner)
    if owner_index < 0:
        return 80
    return int(min(192, 96 + owner_index * 32))


def _reference_head_value(player: int) -> int:
    return int(min(255, 224 + int(player) * 8))


def _reference_render_source_state_gray64(state: dict[str, np.ndarray]) -> np.ndarray:
    frame = np.zeros((1, 64, 64), dtype=np.uint8)
    canvas = frame[0]
    map_size = float(state["map_size"][0])
    body_limit = int(np.clip(state["body_write_cursor"][0], 0, state["body_active"].shape[1]))
    for slot in range(body_limit):
        if not bool(state["body_active"][0, slot]):
            continue
        _reference_draw_world_circle(
            canvas,
            state["body_pos"][0, slot, 0],
            state["body_pos"][0, slot, 1],
            state["body_radius"][0, slot],
            map_size,
            value=_reference_body_value(state["body_owner"][0, slot]),
        )
    for player in range(state["pos"].shape[1]):
        if not bool(state["present"][0, player]) or not bool(state["alive"][0, player]):
            continue
        _reference_draw_world_circle(
            canvas,
            state["pos"][0, player, 0],
            state["pos"][0, player, 1],
            state["radius"][0, player],
            map_size,
            value=_reference_head_value(player),
        )
    return frame


def _random_source_state(seed: int, *, capacity: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    player_count = 2
    return {
        "tick": np.asarray([seed], dtype=np.int32),
        "elapsed_ms": np.asarray([float(seed) * 16.666], dtype=np.float64),
        "map_size": np.asarray([64.0], dtype=np.float64),
        "present": np.ones((1, player_count), dtype=bool),
        "alive": rng.random((1, player_count)) > 0.2,
        "pos": rng.uniform(-4.0, 68.0, size=(1, player_count, 2)).astype(np.float64),
        "radius": rng.choice([0.0, 0.2, 0.6, 1.3, 2.7], size=(1, player_count)).astype(np.float64),
        "body_active": rng.random((1, capacity)) > 0.2,
        "body_pos": rng.uniform(-4.0, 68.0, size=(1, capacity, 2)).astype(np.float64),
        "body_radius": rng.choice(
            [0.0, 0.2, 0.6, 1.3, 2.7, 4.2],
            size=(1, capacity),
        ).astype(np.float64),
        "body_owner": rng.choice([-1, 0, 1, 2, 3, 9], size=(1, capacity)).astype(np.int16),
        "body_write_cursor": np.asarray([rng.integers(0, capacity + 1)], dtype=np.int32),
        "done": np.asarray([False], dtype=bool),
        "terminated": np.asarray([False], dtype=bool),
        "truncated": np.asarray([False], dtype=bool),
        "terminal_reason": np.asarray([0], dtype=np.int16),
    }


def test_source_state_gray64_renders_small_state_shape_dtype_and_range():
    frame = render_source_state_gray64(_small_source_state())
    normalized = normalize_source_state_gray64(frame)

    assert frame.shape == (1, 64, 64)
    assert frame.dtype == np.uint8
    assert int(frame.min()) >= 0
    assert int(frame.max()) <= 255
    assert int(np.count_nonzero(frame)) > 0
    assert normalized.shape == (1, 64, 64)
    assert normalized.dtype == np.float32
    assert float(normalized.min()) >= 0.0
    assert float(normalized.max()) <= 1.0


def test_source_state_gray64_metadata_is_stable_and_explicitly_not_browser_pixels():
    schema = source_state_gray64_schema()
    metadata = source_state_gray64_metadata(includes_render_cost=True)

    assert schema == source_state_gray64_schema()
    assert metadata == source_state_gray64_metadata(includes_render_cost=True)
    assert schema["schema_id"] == SOURCE_STATE_GRAY64_SCHEMA_ID
    assert schema["schema_hash"] == SOURCE_STATE_GRAY64_SCHEMA_HASH
    assert schema["renderer_impl_id"] == SOURCE_STATE_GRAY64_RENDERER_IMPL_ID
    assert schema["renderer_impl_hash"] == SOURCE_STATE_GRAY64_RENDERER_IMPL_HASH
    assert schema["source_claim_id"] == SOURCE_STATE_GRAY64_SOURCE_CLAIM_ID
    assert schema["shape"] == [1, 64, 64]
    assert schema["dtype"] == "uint8"
    assert schema["range"] == [0, 255]
    assert schema["source_state_backed"] is True
    assert schema["browser_pixel_fidelity"] is SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY
    assert schema["browser_pixel_fidelity"] is False
    assert schema["browser_pixel_fidelity_claim"] == "not_claimed"
    assert schema["comparison_target"] == SOURCE_STATE_GRAY64_COMPARISON_TARGET
    assert schema["uses_ale"] is False
    assert schema["ale_usage"] == "none"

    assert metadata["observation_schema_hash"] == SOURCE_STATE_GRAY64_SCHEMA_HASH
    assert metadata["renderer_impl_hash"] == SOURCE_STATE_GRAY64_RENDERER_IMPL_HASH
    assert metadata["includes_render_cost"] is True
    assert metadata["perspective"] == "global_arena_source_state"
    assert metadata["frame_stack_owner"] == "optimizer"
    assert "terminal rows" in metadata["terminal_policy"]
    assert metadata["browser_pixel_fidelity"] is False
    assert metadata["uses_ale"] is False


def test_source_state_gray64_contract_names_state_fields_without_ale_or_browser_claim():
    schema = source_state_gray64_schema()

    assert schema["state_fields"] == list(SOURCE_STATE_GRAY64_STATE_FIELDS)
    assert "state.pos" in schema["state_fields"]
    assert "state.body_pos" in schema["state_fields"]
    assert "state.terminal_reason" in schema["state_fields"]
    assert "state.bonus_active" in schema["optional_state_fields"]
    assert schema["truth_level"] == "source_state_backed_non_browser_pixel"
    assert schema["source_fidelity_level"] == "source_vector_state_geometry_raster"
    assert "browser/canvas pixel parity" in schema["source_pixel_fidelity_blocker"]
    assert schema["browser_pixel_fidelity"] is False
    assert schema["uses_ale"] is False


def test_source_state_gray64_render_uses_body_write_cursor_and_reuses_out_buffer():
    state = _small_source_state()
    out = np.full((1, 64, 64), 255, dtype=np.uint8)
    renderer = SourceStateGray64Renderer()

    frame = renderer.render(state, out=out)
    state_without_tail = _small_source_state()
    state_without_tail["body_active"][0, 2] = False
    expected = render_source_state_gray64(state_without_tail)

    assert frame is out
    np.testing.assert_array_equal(frame, expected)


def test_source_state_gray64_renderer_matches_reference_edge_overlap_and_radius_cases():
    renderer = SourceStateGray64Renderer()

    for capacity in (0, 1, 4, 17, 128):
        for seed in range(10):
            state = _random_source_state(seed + capacity * 1000, capacity=capacity)

            actual = renderer.render(state)
            expected = _reference_render_source_state_gray64(state)

            np.testing.assert_array_equal(actual, expected)


def test_source_state_gray64_renders_optional_bonus_geometry():
    state = _small_source_state()
    state.update(
        {
            "bonus_active": np.asarray([[True, False]], dtype=bool),
            "bonus_pos": np.asarray([[[24.0, 24.0], [0.0, 0.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[1.0, 1.0]], dtype=np.float64),
        }
    )

    frame = render_source_state_gray64(state)

    px = int(np.rint((24.0 / 64.0) * 63.0))
    py = int(np.rint((24.0 / 64.0) * 63.0))
    assert int(frame[0, py, px]) == SOURCE_STATE_GRAY64_BONUS_VALUE


def test_source_state_rgb_canvas_like_renders_full_size_color_players_and_bonus():
    state = _small_source_state()
    state.update(
        {
            "bonus_active": np.asarray([[True, False]], dtype=bool),
            "bonus_pos": np.asarray([[[24.0, 24.0], [0.0, 0.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[1.0, 1.0]], dtype=np.float64),
        }
    )

    frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=96,
        bonus_render_mode=BONUS_RENDER_MODE_CIRCLES_FAST,
    )

    assert frame.shape == (96, 96, 3)
    assert frame.dtype == np.uint8
    assert frame.shape[0] > 64
    flat = frame.reshape(-1, 3)
    for color in SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[:2]:
        rgb = np.asarray(color, dtype=np.uint8)
        assert len(set(int(channel) for channel in rgb)) > 1
        assert bool(np.any(np.all(flat == rgb, axis=1)))
    bonus_rgb = np.asarray(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_RGB, dtype=np.uint8)
    assert bool(np.any(np.all(flat == bonus_rgb, axis=1)))


def test_source_state_rgb_canvas_like_defaults_to_browser_lines_and_connects_straight_trail():
    state = _trail_only_source_state([(24.0, 32.0), (40.0, 32.0)])
    player_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0])
    background_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)

    default_frame = render_source_state_rgb_canvas_like(state, frame_size=64)
    browser_lines_frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )
    body_circles_frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        trail_render_mode=TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    )

    np.testing.assert_array_equal(default_frame, browser_lines_frame)
    np.testing.assert_array_equal(default_frame[32, 32], player_rgb)
    np.testing.assert_array_equal(default_frame[32, 24], player_rgb)
    np.testing.assert_array_equal(default_frame[32, 40], player_rgb)
    np.testing.assert_array_equal(body_circles_frame[32, 33], background_rgb)


def test_source_state_rgb_canvas_like_body_circles_fast_preserves_old_bead_behavior():
    state = _trail_only_source_state([(32.0, 32.0), (33.0, 32.0)])
    player_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0])
    background_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)

    frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        trail_render_mode=TRAIL_RENDER_MODE_BODY_CIRCLES_FAST,
    )
    player_pixels = np.all(frame == player_rgb, axis=2)

    np.testing.assert_array_equal(frame[32, 32], player_rgb)
    np.testing.assert_array_equal(frame[32, 33], background_rgb)
    assert int(np.count_nonzero(player_pixels)) == 1


def test_source_state_rgb_canvas_like_browser_lines_connects_sparse_body_points():
    state = _trail_only_source_state([(24.0, 32.0), (40.0, 32.0)])
    player_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0])

    frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )

    np.testing.assert_array_equal(frame[32, 24], player_rgb)
    np.testing.assert_array_equal(frame[32, 32], player_rgb)
    np.testing.assert_array_equal(frame[32, 40], player_rgb)


def test_source_state_rgb_canvas_like_browser_lines_prefers_visual_trail_points():
    state = _trail_only_source_state([(8.0, 16.0), (24.0, 16.0)])
    _with_visual_trail_points(state, [(48.0, 8.0), (48.0, 24.0)])
    player_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0])
    background_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)

    frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )

    np.testing.assert_array_equal(frame[16, 16], background_rgb)
    np.testing.assert_array_equal(frame[16, 48], player_rgb)


def test_source_state_rgb_canvas_like_browser_lines_breaks_on_explicit_segment_state():
    state = _trail_only_source_state(
        [(24.0, 32.0), (40.0, 32.0)],
        break_before=[False, True],
    )
    player_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0])
    background_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)

    frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )

    np.testing.assert_array_equal(frame[32, 24], player_rgb)
    np.testing.assert_array_equal(frame[32, 32], background_rgb)
    np.testing.assert_array_equal(frame[32, 40], player_rgb)


def test_source_state_rgb_canvas_like_browser_lines_breaks_visual_trail_segments():
    state = _trail_only_source_state([(0.0, 0.0)])
    _with_visual_trail_points(
        state,
        [(24.0, 32.0), (40.0, 32.0)],
        break_before=[False, True],
    )
    player_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0])
    background_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)

    frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )

    np.testing.assert_array_equal(frame[32, 24], player_rgb)
    np.testing.assert_array_equal(frame[32, 32], background_rgb)
    np.testing.assert_array_equal(frame[32, 40], player_rgb)


def test_cpu_oracle_browser_lines_connect_interleaved_same_owner_trails_and_pin_crossing_order():
    state = _trail_only_source_state([(0.0, 0.0)])
    state["body_active"][:, :] = False
    _with_visual_trail_points(
        state,
        [(16.0, 32.0), (32.0, 16.0), (48.0, 32.0), (32.0, 48.0)],
        radii=0.6,
        owners=[0, 1, 0, 1],
    )
    player_0_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0])
    player_1_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[1])

    frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )

    np.testing.assert_array_equal(frame[32, 24], player_0_rgb)
    np.testing.assert_array_equal(frame[24, 32], player_1_rgb)
    np.testing.assert_array_equal(frame[32, 32], player_0_rgb)


def test_cpu_oracle_visual_trail_write_cursor_ignores_active_stale_tail_slots():
    state = _trail_only_source_state([(0.0, 0.0)])
    state["body_active"][:, :] = False
    _with_visual_trail_points(
        state,
        [(16.0, 16.0), (32.0, 16.0), (48.0, 48.0)],
        radii=0.6,
        owners=0,
    )
    state["visual_trail_write_cursor"][0] = 2
    player_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0])
    background_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)

    frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
    )

    np.testing.assert_array_equal(frame[16, 24], player_rgb)
    np.testing.assert_array_equal(frame[48, 48], background_rgb)


def test_cpu_oracle_live_heads_draw_over_bonus_symbols():
    state = _small_source_state()
    state["body_active"][:, :] = False
    state["body_write_cursor"][0] = 0
    state["present"][0, :] = [True, False]
    state["alive"][0, :] = [True, False]
    state["pos"][0, 0] = [32.0, 32.0]
    state["radius"][0, 0] = 1.0
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[vector_runtime.BONUS_TYPE_SELF_FAST]], dtype=np.int16),
            "bonus_pos": np.asarray([[[32.0, 32.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[6.0]], dtype=np.float64),
        }
    )
    player_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0])
    background_rgb = _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)

    frame = render_source_state_rgb_canvas_like(
        state,
        frame_size=64,
        bonus_render_mode=BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    )

    np.testing.assert_array_equal(frame[32, 32], player_rgb)
    crop = frame[26:39, 26:39]
    non_background = np.any(crop != background_rgb[None, None, :], axis=2)
    non_head = ~np.all(crop == player_rgb[None, None, :], axis=2)
    assert bool(np.any(non_background & non_head))


def test_source_state_canvas_gray64_is_downsampled_luminance_of_browser_like_rgb_for_both_trail_modes():
    state = _small_source_state()
    state["avatar_color"] = np.asarray([[1, 0]], dtype=np.int16)
    schema = source_state_canvas_gray64_schema()

    assert schema["schema_id"] == SOURCE_STATE_CANVAS_GRAY64_SCHEMA_ID
    assert schema["renderer_impl_id"] == SOURCE_STATE_CANVAS_GRAY64_RENDERER_IMPL_ID
    assert "state.body_break_before" in schema["optional_state_fields"]
    assert schema["rgb_source_frame_size"] == SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    assert schema["downsample_target_frame_size"] == 64
    assert schema["downsample_method"] == "integer_area_average_after_luma"
    assert schema["downsample_ratio"] == 11
    assert schema["default_bonus_render_mode"] == BONUS_RENDER_MODE_SIMPLE_SYMBOLS
    assert schema["supported_bonus_render_modes"] == [
        BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
        BONUS_RENDER_MODE_BROWSER_SPRITES,
        BONUS_RENDER_MODE_CIRCLES_FAST,
    ]
    assert schema["bonus_renderer_kind"] == "simple_symbol_masks"
    assert (
        schema["bonus_sprite_missing_fallback"]
        == "deterministic_type_coded_placeholder_stamp"
    )
    assert schema["bonus_sprite_cache"] == (
        "in_process_lru_stamp_cache_by_tile_index_and_pixel_size"
    )

    for mode in (TRAIL_RENDER_MODE_BROWSER_LINES, TRAIL_RENDER_MODE_BODY_CIRCLES_FAST):
        source_rgb = render_source_state_rgb_canvas_like(
            state,
            trail_render_mode=mode,
        )
        assert source_rgb.shape == (
            SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
            SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,
            3,
        )
        direct_rgb64 = render_source_state_rgb_canvas_like(
            state,
            frame_size=64,
            trail_render_mode=mode,
        )
        frame = render_source_state_canvas_gray64(state, trail_render_mode=mode)

        assert frame.shape == SOURCE_STATE_CANVAS_GRAY64_SHAPE
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, rgb_canvas_like_to_gray64(source_rgb))
        assert not np.array_equal(frame, rgb_canvas_like_to_gray64(direct_rgb64))
        px = int(np.rint((10.0 / 64.0) * 63.0))
        py = int(np.rint((10.0 / 64.0) * 63.0))
        assert int(frame[0, py, px]) > int(round(34.0))


def test_rgb_canvas_like_to_gray64_area_downsample_encodes_source_pixel_coverage():
    source_size = SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE
    target_size = SOURCE_STATE_CANVAS_GRAY64_SHAPE[1]
    ratio = source_size // target_size
    block_area = ratio * ratio
    y_block = 3
    x_block = 5
    background = np.asarray(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB, dtype=np.uint8)
    player = np.asarray(SOURCE_STATE_RGB_CANVAS_LIKE_PLAYER_RGB[0], dtype=np.uint8)
    background_luma = int(
        np.rint(0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2])
    )
    player_luma = 0.299 * float(player[0]) + 0.587 * float(player[1]) + 0.114 * float(player[2])
    rgb = np.empty((source_size, source_size, 3), dtype=np.uint8)
    rgb[:, :] = background

    empty = rgb_canvas_like_to_gray64(rgb)
    assert int(empty[0, y_block, x_block]) == background_luma

    y0 = y_block * ratio
    x0 = x_block * ratio
    rgb[y0, x0] = player
    one_pixel = rgb_canvas_like_to_gray64(rgb)
    expected_one = int(
        np.rint(background_luma + (player_luma - background_luma) / float(block_area))
    )
    assert int(one_pixel[0, y_block, x_block]) == expected_one
    assert expected_one == background_luma

    rgb[y0, x0 + 1] = player
    two_pixels = rgb_canvas_like_to_gray64(rgb)
    expected_two = int(
        np.rint(background_luma + 2.0 * (player_luma - background_luma) / float(block_area))
    )
    assert int(two_pixels[0, y_block, x_block]) == expected_two
    assert expected_two > background_luma
    assert np.count_nonzero(two_pixels[0] != background_luma) == 1


def test_source_state_rgb_canvas_like_renders_distinct_source_bonus_sprite_patches():
    state = _small_source_state()
    state["present"][:, :] = False
    state["alive"][:, :] = False
    state["body_active"][:, :] = False
    state["body_write_cursor"][0] = 0
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[0]], dtype=np.int16),
            "bonus_pos": np.asarray([[[32.0, 32.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[8.0]], dtype=np.float64),
        }
    )
    patches = []

    for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES:
        state["bonus_type"][0, 0] = int(code)
        frame = render_source_state_rgb_canvas_like(
            state,
            frame_size=96,
            bonus_render_mode=BONUS_RENDER_MODE_BROWSER_SPRITES,
        )
        patch = frame[36:60, 36:60].copy()

        assert bool(np.any(patch != _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)))
        patches.append(patch.tobytes())

    assert len(patches) == len(SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_NAMES) == 12
    assert len(set(patches)) == len(patches)


def test_source_bonus_sprite_sheet_path_falls_back_to_repo_mount(
    tmp_path,
    monkeypatch,
):
    relative_path = visual_observation.SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH
    sprite_path = tmp_path / relative_path
    sprite_path.parent.mkdir(parents=True)
    sprite_path.write_bytes(b"fake")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        visual_observation,
        "__file__",
        "/root/curvyzero/env/vector_visual_observation.py",
    )

    assert _source_bonus_sprite_sheet_path() == sprite_path


def test_source_state_rgb_canvas_like_uses_typed_bonus_placeholders_when_atlas_missing(
    monkeypatch,
):
    visual_observation._source_bonus_sprite_tiles.cache_clear()
    visual_observation._source_bonus_sprite_stamp.cache_clear()
    visual_observation._source_bonus_placeholder_stamp.cache_clear()
    monkeypatch.setattr(
        visual_observation,
        "SOURCE_STATE_RGB_CANVAS_LIKE_BONUS_SPRITE_SHEET_RELATIVE_PATH",
        "missing/bonus.png",
    )
    state = _small_source_state()
    state["present"][:, :] = False
    state["alive"][:, :] = False
    state["body_active"][:, :] = False
    state["body_write_cursor"][0] = 0
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[0]], dtype=np.int16),
            "bonus_pos": np.asarray([[[32.0, 32.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[8.0]], dtype=np.float64),
        }
    )
    patches = []

    try:
        for code in (vector_runtime.BONUS_TYPE_SELF_SMALL, vector_runtime.BONUS_TYPE_GAME_CLEAR):
            state["bonus_type"][0, 0] = int(code)
            frame = render_source_state_rgb_canvas_like(
                state,
                frame_size=96,
                bonus_render_mode=BONUS_RENDER_MODE_BROWSER_SPRITES,
            )
            patch = frame[36:60, 36:60].copy()

            assert bool(np.any(patch != _rgb_triplet(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB)))
            patches.append(patch.tobytes())

        assert patches[0] != patches[1]
    finally:
        visual_observation._source_bonus_sprite_tiles.cache_clear()
        visual_observation._source_bonus_sprite_stamp.cache_clear()
        visual_observation._source_bonus_placeholder_stamp.cache_clear()


@pytest.mark.parametrize(
    ("bonus_type_code", "sprite_index"),
    [
        (1, 9),
        (2, 2),
        (3, 0),
        (4, 5),
        (5, 3),
        (6, 1),
        (7, 6),
        (8, 8),
        (9, 11),
        (10, 4),
        (11, 7),
        (12, 10),
    ],
)
def test_source_state_rgb_canvas_like_bonus_type_codes_map_to_browser_atlas_tiles(
    bonus_type_code,
    sprite_index,
):
    assert _bonus_sprite_index(bonus_type_code) == sprite_index


def test_source_state_canvas_gray64_uses_sprite_bonus_path_by_default():
    state = _small_source_state()
    state["present"][:, :] = False
    state["alive"][:, :] = False
    state["body_active"][:, :] = False
    state["body_write_cursor"][0] = 0
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray(
                [[vector_runtime.BONUS_TYPE_SELF_SMALL]],
                dtype=np.int16,
            ),
            "bonus_pos": np.asarray([[[32.0, 32.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[8.0]], dtype=np.float64),
        }
    )

    sprite_rgb = render_source_state_rgb_canvas_like(state)
    direct_sprite_rgb64 = render_source_state_rgb_canvas_like(state, frame_size=64)
    sprite_gray64 = render_source_state_canvas_gray64(state)
    circle_gray64 = render_source_state_canvas_gray64(
        state,
        bonus_render_mode=BONUS_RENDER_MODE_CIRCLES_FAST,
    )

    np.testing.assert_array_equal(sprite_gray64, rgb_canvas_like_to_gray64(sprite_rgb))
    assert not np.array_equal(sprite_gray64, rgb_canvas_like_to_gray64(direct_sprite_rgb64))
    assert not np.array_equal(sprite_gray64, circle_gray64)


def test_direct_fast_simple_bonus_symbols_are_distinct_and_not_remapped():
    state = _small_source_state()
    state["present"][:, :] = False
    state["alive"][:, :] = False
    state["body_active"][:, :] = False
    state["body_write_cursor"][0] = 0
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[0]], dtype=np.int16),
            "bonus_pos": np.asarray([[[32.0, 32.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[3.0]], dtype=np.float64),
        }
    )
    background_luma = int(
        np.rint(
            0.299 * SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB[0]
            + 0.587 * SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB[1]
            + 0.114 * SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB[2]
        )
    )
    allowed_symbol_values = {
        int(BONUS_SYMBOL_INNER_LUMA),
        *(int(value) for value in BONUS_SYMBOL_OUTER_LUMA_BY_SHAPE),
        *(int(value) for value in BONUS_SYMBOL_INNER_LUMA_BY_SHAPE),
    }
    forbidden_remap_values = {80, 96, 128, 160, 192, 208, 224, 232, 240, 248}
    hashes = []
    crop_hashes = []

    for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES:
        state["bonus_type"][0, 0] = int(code)
        frames = render_source_state_gray64_fast_player_perspectives(
            state,
            player_count=2,
            bonus_render_mode=BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
        )
        np.testing.assert_array_equal(frames[0], frames[1])
        frame = frames[0, 0]
        crop = frame[29:36, 29:36]
        non_background = crop[crop != background_luma]

        assert non_background.size >= 9
        assert set(int(value) for value in np.unique(non_background)) <= allowed_symbol_values
        assert not (
            set(int(value) for value in np.unique(non_background))
            & forbidden_remap_values
        )
        hashes.append(frame.tobytes())
        crop_hashes.append(crop.tobytes())

    assert len(set(hashes)) == len(vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES)
    assert len(set(crop_hashes)) == len(vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES)


def test_direct_fast_simple_bonus_symbol_base_masks_keep_margin():
    stamps = [
        _simple_bonus_symbol_base(code).astype(np.int16)
        for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES
    ]
    assert len({stamp.tobytes() for stamp in stamps}) == len(stamps)

    min_l1 = None
    min_mismatch = None
    for left_index, left in enumerate(stamps):
        for right in stamps[left_index + 1 :]:
            diff = np.abs(left - right)
            l1 = int(diff.sum())
            mismatch = int((diff > 0).sum())
            min_l1 = l1 if min_l1 is None else min(min_l1, l1)
            min_mismatch = mismatch if min_mismatch is None else min(min_mismatch, mismatch)

    assert min_l1 is not None
    assert min_l1 >= 1300
    assert min_mismatch >= 10


def test_direct_fast_simple_bonus_symbols_stay_distinct_across_offsets_and_radii():
    state = _small_source_state()
    state["present"][:, :] = False
    state["alive"][:, :] = False
    state["body_active"][:, :] = False
    state["body_write_cursor"][0] = 0
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[0]], dtype=np.int16),
            "bonus_pos": np.asarray([[[0.0, 0.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[3.0]], dtype=np.float64),
        }
    )
    cases = (
        ((20.0, 20.0), 2.0),
        ((31.4, 32.6), 3.0),
        ((47.8, 18.2), 4.0),
        ((8.2, 55.1), 3.0),
    )

    for (x_value, y_value), radius in cases:
        hashes = []
        state["bonus_pos"][0, 0] = (x_value, y_value)
        state["bonus_radius"][0, 0] = radius
        for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES:
            state["bonus_type"][0, 0] = int(code)
            frames = render_source_state_gray64_fast_player_perspectives(
                state,
                player_count=2,
                bonus_render_mode=BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
            )
            np.testing.assert_array_equal(frames[0], frames[1])
            hashes.append(frames[0].tobytes())

        assert len(set(hashes)) == len(vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES)


def test_direct_fast_simple_bonus_symbols_overwrite_underlying_trails():
    background_luma = int(
        np.rint(
            0.299 * SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB[0]
            + 0.587 * SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB[1]
            + 0.114 * SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB[2]
        )
    )

    for use_visual_trail in (False, True):
        hashes = []
        for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES:
            state = _small_source_state()
            state["present"][:, :] = False
            state["alive"][:, :] = False
            state["body_active"][:, :] = False
            state["body_write_cursor"][0] = 0
            state.update(
                {
                    "bonus_active": np.asarray([[True]], dtype=bool),
                    "bonus_type": np.asarray([[int(code)]], dtype=np.int16),
                    "bonus_pos": np.asarray([[[32.0, 32.0]]], dtype=np.float64),
                    "bonus_radius": np.asarray([[3.0]], dtype=np.float64),
                }
            )
            bonus_only = render_source_state_gray64_fast_player_perspectives(
                state,
                player_count=2,
                bonus_render_mode=BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
            )[0, 0]

            if use_visual_trail:
                _with_visual_trail_points(
                    state,
                    [(32.0, 32.0)],
                    radii=5.0,
                    owners=3,
                )
            else:
                state["body_active"][0, 0] = True
                state["body_pos"][0, 0] = (32.0, 32.0)
                state["body_radius"][0, 0] = 5.0
                state["body_owner"][0, 0] = 3
                state["body_write_cursor"][0] = 1
            with_trail = render_source_state_gray64_fast_player_perspectives(
                state,
                player_count=2,
                bonus_render_mode=BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
            )[0, 0]

            bonus_crop = bonus_only[29:36, 29:36]
            trail_crop = with_trail[29:36, 29:36]
            symbol_mask = bonus_crop != background_luma
            assert symbol_mask.any()
            np.testing.assert_array_equal(trail_crop[symbol_mask], bonus_crop[symbol_mask])
            if (~symbol_mask).any():
                assert np.any(trail_crop[~symbol_mask] != background_luma)
            hashes.append(trail_crop.tobytes())

        assert len(set(hashes)) == len(vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES)


def test_cpu_oracle_simple_bonus_symbols_overlay_browser_line_trails():
    state = _small_source_state()
    state["present"][:, :] = False
    state["alive"][:, :] = False
    state["body_active"][:, :] = False
    state["body_write_cursor"][0] = 0
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[vector_runtime.BONUS_TYPE_SELF_FAST]], dtype=np.int16),
            "bonus_pos": np.asarray([[[32.0, 32.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[4.0]], dtype=np.float64),
        }
    )
    bonus_only = render_source_state_rgb_canvas_like(
        state,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
        bonus_render_mode=BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    )

    _with_visual_trail_points(
        state,
        [(30.0, 32.0), (32.0, 32.0), (34.0, 32.0)],
        radii=6.0,
        owners=1,
    )
    with_trail = render_source_state_rgb_canvas_like(
        state,
        trail_render_mode=TRAIL_RENDER_MODE_BROWSER_LINES,
        bonus_render_mode=BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    )

    background = np.asarray(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB, dtype=np.uint8)
    symbol_mask = np.any(bonus_only != background[None, None, :], axis=2)
    assert symbol_mask.any()
    np.testing.assert_array_equal(with_trail[symbol_mask], bonus_only[symbol_mask])


def test_cpu_oracle_simple_bonus_symbols_rgb_avoids_full_canvas_scratch(monkeypatch):
    background = np.asarray(SOURCE_STATE_RGB_CANVAS_LIKE_BACKGROUND_RGB, dtype=np.uint8)
    canvas = np.empty((SOURCE_STATE_RGB_CANVAS_LIKE_DEFAULT_FRAME_SIZE,) * 2 + (3,), dtype=np.uint8)
    canvas[:, :] = background
    original_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if tuple(shape) == canvas.shape[:2]:
            raise AssertionError("unexpected full-frame simple-symbol scratch")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(visual_observation.np, "zeros", guarded_zeros)

    visual_observation._draw_bonus_simple_symbols_rgb(
        canvas,
        np.asarray([[32.0, 32.0]], dtype=np.float64),
        np.asarray([3.0], dtype=np.float64),
        64.0,
        bonus_types=np.asarray([vector_runtime.BONUS_TYPE_SELF_FAST], dtype=np.int16),
    )

    assert bool(np.any(canvas != background[None, None, :]))


def test_direct_fast_legacy_luma_circle_mode_is_still_available():
    state = _small_source_state()
    state["present"][:, :] = False
    state["alive"][:, :] = False
    state["body_active"][:, :] = False
    state["body_write_cursor"][0] = 0
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[vector_runtime.BONUS_TYPE_SELF_FAST]], dtype=np.int16),
            "bonus_pos": np.asarray([[[32.0, 32.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[3.0]], dtype=np.float64),
        }
    )

    symbol_frame = render_source_state_gray64_fast_player_perspectives(
        state,
        player_count=2,
        bonus_render_mode=BONUS_RENDER_MODE_SIMPLE_SYMBOLS,
    )
    legacy_frame = render_source_state_gray64_fast_player_perspectives(
        state,
        player_count=2,
        bonus_render_mode=BONUS_RENDER_MODE_CIRCLES_FAST,
    )

    assert not np.array_equal(symbol_frame, legacy_frame)


def test_source_snapshot_gray64_matches_equivalent_vector_state_with_bonus_body():
    state = _small_source_state()
    state["body_active"][0, 2] = False
    state["body_write_cursor"][0] = 2
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[1]], dtype=np.int16),
            "bonus_pos": np.asarray([[[24.0, 24.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[1.0]], dtype=np.float64),
        }
    )
    snapshot = {
        "game": {"size": 64},
        "avatars": [
            {"id": 1, "x": 10.0, "y": 10.0, "alive": True, "present": True},
            {"id": 2, "x": 42.0, "y": 18.0, "alive": True, "present": True},
        ],
    }
    world_bodies = (
        {"x": 8.0, "y": 10.0, "radius": 1.0, "avatarId": 1},
        {"x": 40.0, "y": 18.0, "radius": 1.0, "avatarId": 2},
    )
    bonus_bodies = ({"id": 1, "type": "BonusSelfSmall", "x": 24.0, "y": 24.0, "radius": 1.0},)
    avatar_body_metadata = (
        {"id": 1, "radius": 1.0},
        {"id": 2, "radius": 1.0},
    )

    vector_frame = render_source_state_gray64(state)
    source_frame = render_source_snapshot_gray64(
        snapshot,
        world_bodies=world_bodies,
        bonus_bodies=bonus_bodies,
        avatar_body_metadata=avatar_body_metadata,
    )

    np.testing.assert_array_equal(source_frame, vector_frame)


def test_source_snapshot_canvas_gray64_matches_equivalent_vector_state_with_colors():
    state = _small_source_state()
    state["avatar_color"] = np.asarray([[0, 1]], dtype=np.int16)
    state["body_active"][0, 2] = False
    state["body_write_cursor"][0] = 2
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[1]], dtype=np.int16),
            "bonus_pos": np.asarray([[[24.0, 24.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[1.0]], dtype=np.float64),
        }
    )
    snapshot = {
        "game": {"size": 64},
        "avatars": [
            {
                "id": 1,
                "x": 10.0,
                "y": 10.0,
                "alive": True,
                "present": True,
                "color": "#ff0000",
            },
            {
                "id": 2,
                "x": 42.0,
                "y": 18.0,
                "alive": True,
                "present": True,
                "color": "#00ff00",
            },
        ],
    }
    world_bodies = (
        {"x": 8.0, "y": 10.0, "radius": 1.0, "avatarId": 1},
        {"x": 40.0, "y": 18.0, "radius": 1.0, "avatarId": 2},
    )
    bonus_bodies = ({"id": 1, "type": "BonusSelfSmall", "x": 24.0, "y": 24.0, "radius": 1.0},)
    avatar_body_metadata = (
        {"id": 1, "radius": 1.0},
        {"id": 2, "radius": 1.0},
    )

    np.testing.assert_array_equal(
        render_source_snapshot_rgb_canvas_like(
            snapshot,
            world_bodies=world_bodies,
            bonus_bodies=bonus_bodies,
            avatar_body_metadata=avatar_body_metadata,
            frame_size=64,
        ),
        render_source_state_rgb_canvas_like(state, frame_size=64),
    )
    np.testing.assert_array_equal(
        render_source_snapshot_canvas_gray64(
            snapshot,
            world_bodies=world_bodies,
            bonus_bodies=bonus_bodies,
            avatar_body_metadata=avatar_body_metadata,
        ),
        render_source_state_canvas_gray64(state),
    )


def test_source_snapshot_canvas_like_renderers_support_both_trail_modes():
    state = _trail_only_source_state([(32.0, 32.0), (33.0, 32.0)])
    snapshot = {
        "game": {"size": 64},
        "avatars": [
            {
                "id": 1,
                "x": 0.0,
                "y": 0.0,
                "alive": False,
                "present": False,
                "color": "#ff0000",
            },
        ],
    }
    world_bodies = (
        {"x": 32.0, "y": 32.0, "radius": 0.0, "avatarId": 1},
        {"x": 33.0, "y": 32.0, "radius": 0.0, "avatarId": 1},
    )

    for mode in (TRAIL_RENDER_MODE_BROWSER_LINES, TRAIL_RENDER_MODE_BODY_CIRCLES_FAST):
        np.testing.assert_array_equal(
            render_source_snapshot_rgb_canvas_like(
                snapshot,
                world_bodies=world_bodies,
                frame_size=64,
                trail_render_mode=mode,
            ),
            render_source_state_rgb_canvas_like(
                state,
                frame_size=64,
                trail_render_mode=mode,
            ),
        )
        np.testing.assert_array_equal(
            render_source_snapshot_canvas_gray64(
                snapshot,
                world_bodies=world_bodies,
                trail_render_mode=mode,
            ),
            render_source_state_canvas_gray64(state, trail_render_mode=mode),
        )


def test_source_state_gray64_skips_circles_fully_outside_source_arena():
    state = _small_source_state()
    state["body_active"][0, :] = False
    state["body_active"][0, 0] = True
    state["body_pos"][0, 0] = np.asarray([65.0, 32.0], dtype=np.float64)
    state["body_radius"][0, 0] = 0.5
    state["body_owner"][0, 0] = 0
    state["body_write_cursor"][0] = 1
    state["alive"][0, :] = False

    frame = render_source_state_gray64(state)

    assert int(np.count_nonzero(frame)) == 0


def test_source_state_bonus64_v1_schema_and_no_bonus_planes_are_separate_from_v0():
    state = _small_source_state()

    tensor = render_source_state_bonus64_stack4_player_perspective_v1(state)
    schema = source_state_bonus64_stack4_player_perspective_v1_schema()

    assert schema["schema_id"] == SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SCHEMA_ID
    assert schema["shape"] == list(SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SHAPE)
    assert schema["status"] == "promoted_2p_typed_bonus_parity_gate_not_lightzero_default"
    assert schema["parity_gate"] is True
    assert "active_map_bonus_mask_type" in schema["parity_scope"]
    assert "post_catch_status_planes" in schema["parity_scope"]
    assert schema["replaces_gray64_v0"] is False
    assert schema["base_geometry_schema_id"] == SOURCE_STATE_GRAY64_SCHEMA_ID
    assert tensor.shape == SOURCE_STATE_BONUS64_STACK4_PLAYER_PERSPECTIVE_SHAPE
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
    assert int(np.count_nonzero(tensor[:4])) > 0
    assert float(tensor[SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL].max()) == 0.0
    assert float(tensor[SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL].max()) == 0.0
    assert float(tensor[list(SOURCE_STATE_BONUS64_STACK4_SELF_STATUS_CHANNELS)].max()) == 0.0
    assert float(tensor[list(SOURCE_STATE_BONUS64_STACK4_OTHER_STATUS_CHANNELS)].max()) == 0.0


def test_source_state_bonus64_v1_renders_bonus_mask_and_type_code():
    state = _small_source_state()
    state.update(
        {
            "bonus_active": np.asarray([[True, False]], dtype=bool),
            "bonus_type": np.asarray([[10, 0]], dtype=np.int16),
            "bonus_id": np.asarray([[3, -1]], dtype=np.int32),
            "bonus_pos": np.asarray([[[24.0, 24.0], [0.0, 0.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[1.0, 1.0]], dtype=np.float64),
        }
    )

    tensor = render_source_state_bonus64_stack4_player_perspective_v1(state)

    px = int(np.rint((24.0 / 64.0) * 63.0))
    py = int(np.rint((24.0 / 64.0) * 63.0))
    assert tensor[SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL, py, px] == 1.0
    assert tensor[SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL, py, px] == np.float32(10.0 / 12.0)


def test_source_state_bonus64_v1_distinguishes_all_source_default_map_bonus_types():
    state = _small_source_state()
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
            "bonus_type": np.asarray([[0]], dtype=np.int16),
            "bonus_id": np.asarray([[1]], dtype=np.int32),
            "bonus_pos": np.asarray([[[24.0, 24.0]]], dtype=np.float64),
            "bonus_radius": np.asarray([[1.0]], dtype=np.float64),
        }
    )
    px = int(np.rint((24.0 / 64.0) * 63.0))
    py = int(np.rint((24.0 / 64.0) * 63.0))
    center_values = []

    for code in vector_runtime.SOURCE_DEFAULT_BONUS_TYPE_CODES:
        state["bonus_type"][0, 0] = int(code)
        tensor = render_source_state_bonus64_stack4_player_perspective_v1(state)
        expected = np.float32(float(code) / SOURCE_STATE_BONUS64_STACK4_MAX_BONUS_TYPE_CODE)

        assert tensor[SOURCE_STATE_BONUS64_STACK4_BONUS_MASK_CHANNEL, py, px] == 1.0
        assert tensor[SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL, py, px] == expected
        center_values.append(float(tensor[SOURCE_STATE_BONUS64_STACK4_BONUS_TYPE_CHANNEL, py, px]))

    assert len(center_values) == 12
    assert len(set(center_values)) == len(center_values)


def test_source_state_bonus64_v1_swaps_self_and_other_status_by_perspective():
    state = _small_source_state()
    state.update(
        {
            "base_radius": np.asarray([[1.0, 1.0]], dtype=np.float64),
            "speed": np.asarray([[16.0, 8.0]], dtype=np.float64),
            "base_speed": np.asarray([[16.0, 16.0]], dtype=np.float64),
            "inverse": np.asarray([[False, True]], dtype=bool),
            "invincible": np.asarray([[True, False]], dtype=bool),
            "printing": np.asarray([[False, True]], dtype=bool),
            "angular_velocity_per_ms": np.asarray([[0.0028, 0.0]], dtype=np.float64),
            "base_angular_velocity_per_ms": np.asarray([[0.0028, 0.0028]], dtype=np.float64),
            "bonus_stack_count": np.asarray([[1, 1]], dtype=np.int32),
            "bonus_stack_duration_ms": np.asarray([[[5000], [2500]]], dtype=np.int32),
        }
    )

    p0 = render_source_state_bonus64_stack4_player_perspective_v1(state, controlled_player=0)
    p1 = render_source_state_bonus64_stack4_player_perspective_v1(state, controlled_player=1)

    self_channels = SOURCE_STATE_BONUS64_STACK4_SELF_STATUS_CHANNELS
    other_channels = SOURCE_STATE_BONUS64_STACK4_OTHER_STATUS_CHANNELS
    assert p0[self_channels[2], 0, 0] == 0.0
    assert p0[other_channels[2], 0, 0] == 1.0
    assert p0[self_channels[4], 0, 0] == 1.0
    assert p0[other_channels[5], 0, 0] == 1.0
    assert p0[self_channels[6], 0, 0] == np.float32(0.5)
    assert p0[other_channels[6], 0, 0] == np.float32(0.25)

    assert p1[self_channels[2], 0, 0] == 1.0
    assert p1[other_channels[2], 0, 0] == 0.0
    assert p1[self_channels[5], 0, 0] == 1.0
    assert p1[other_channels[4], 0, 0] == 1.0


def test_source_state_bonus64_v1_renders_game_borderless_status():
    state = _small_source_state()
    state.update(
        {
            "bonus_game_stack_count": np.asarray([1], dtype=np.int32),
            "bonus_game_stack_borderless": np.asarray([[1]], dtype=np.int16),
            "bonus_game_stack_duration_ms": np.asarray([[7500]], dtype=np.int32),
        }
    )

    tensor = render_source_state_bonus64_stack4_player_perspective_v1(state)

    assert tensor[SOURCE_STATE_BONUS64_STACK4_GAME_BORDERLESS_CHANNEL, 0, 0] == 1.0
    assert tensor[SOURCE_STATE_BONUS64_STACK4_GAME_TTL_CHANNEL, 0, 0] == np.float32(0.75)


def test_source_state_gray64_render_rejects_invalid_body_write_cursor():
    state = _small_source_state()
    state["body_write_cursor"][0] = state["body_active"].shape[1] + 1

    try:
        render_source_state_gray64(state)
    except VectorVisualObservationError as exc:
        assert "body_write_cursor" in str(exc)
    else:
        raise AssertionError("expected invalid body_write_cursor to be rejected")


def test_canvas_like_renderers_reject_invalid_trail_render_mode():
    state = _small_source_state()
    snapshot = {
        "game": {"size": 64},
        "avatars": [{"id": 1, "x": 0.0, "y": 0.0}],
    }

    with pytest.raises(VectorVisualObservationError, match="trail_render_mode"):
        render_source_state_rgb_canvas_like(state, trail_render_mode="not-a-mode")
    with pytest.raises(VectorVisualObservationError, match="trail_render_mode"):
        render_source_state_canvas_gray64(state, trail_render_mode="not-a-mode")
    with pytest.raises(VectorVisualObservationError, match="trail_render_mode"):
        render_source_snapshot_rgb_canvas_like(snapshot, trail_render_mode="not-a-mode")
    with pytest.raises(VectorVisualObservationError, match="bonus_render_mode"):
        render_source_state_rgb_canvas_like(state, bonus_render_mode="not-a-mode")
    with pytest.raises(VectorVisualObservationError, match="bonus_render_mode"):
        render_source_state_canvas_gray64(state, bonus_render_mode="not-a-mode")
    with pytest.raises(VectorVisualObservationError, match="bonus_render_mode"):
        render_source_snapshot_rgb_canvas_like(snapshot, bonus_render_mode="not-a-mode")
