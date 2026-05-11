import numpy as np

from curvyzero.env.vector_visual_observation import (
    SOURCE_STATE_GRAY64_BROWSER_PIXEL_FIDELITY,
    SOURCE_STATE_GRAY64_BONUS_VALUE,
    SOURCE_STATE_GRAY64_COMPARISON_TARGET,
    SOURCE_STATE_GRAY64_RENDERER_IMPL_HASH,
    SOURCE_STATE_GRAY64_RENDERER_IMPL_ID,
    SOURCE_STATE_GRAY64_SCHEMA_HASH,
    SOURCE_STATE_GRAY64_SCHEMA_ID,
    SOURCE_STATE_GRAY64_SOURCE_CLAIM_ID,
    SOURCE_STATE_GRAY64_STATE_FIELDS,
    SourceStateGray64Renderer,
    VectorVisualObservationError,
    normalize_source_state_gray64,
    render_source_snapshot_gray64,
    render_source_state_gray64,
    source_state_gray64_metadata,
    source_state_gray64_schema,
)


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
        "radius": rng.choice([0.0, 0.2, 0.6, 1.3, 2.7], size=(1, player_count)).astype(
            np.float64
        ),
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


def test_source_snapshot_gray64_matches_equivalent_vector_state_with_bonus_body():
    state = _small_source_state()
    state["body_active"][0, 2] = False
    state["body_write_cursor"][0] = 2
    state.update(
        {
            "bonus_active": np.asarray([[True]], dtype=bool),
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
    bonus_bodies = (
        {"id": 1, "type": "BonusSelfSmall", "x": 24.0, "y": 24.0, "radius": 1.0},
    )
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


def test_source_state_gray64_render_rejects_invalid_body_write_cursor():
    state = _small_source_state()
    state["body_write_cursor"][0] = state["body_active"].shape[1] + 1

    try:
        render_source_state_gray64(state)
    except VectorVisualObservationError as exc:
        assert "body_write_cursor" in str(exc)
    else:
        raise AssertionError("expected invalid body_write_cursor to be rejected")
