import numpy as np
import pytest

from curvyzero.infra.modal import source_state_gpu_render_benchmark as bench


def _base_config(**overrides):
    config = {
        "batch_size": 1,
        "player_count": 2,
        "trail_slots": 4,
        "bonus_count": 0,
        "controlled_player": 1,
    }
    config.update(overrides)
    return config


def _minimal_production_state():
    return {
        "pos": np.zeros((1, 2, 2), dtype=np.float64),
        "radius": np.ones((1, 2), dtype=np.float64),
        "alive": np.ones((1, 2), dtype=bool),
        "present": np.ones((1, 2), dtype=bool),
        "visual_trail_pos": np.array(
            [[[1.0, 1.0], [2.0, 2.0], [99.0, 99.0], [100.0, 100.0]]],
            dtype=np.float64,
        ),
        "visual_trail_radius": np.ones((1, 4), dtype=np.float64),
        "visual_trail_owner": np.array([[0, 0, 1, 1]], dtype=np.int16),
        "visual_trail_active": np.array([[True, True, True, True]], dtype=bool),
        "visual_trail_write_cursor": np.array([2], dtype=np.int32),
        "visual_trail_break_before": np.zeros((1, 4), dtype=bool),
    }


def test_production_to_benchmark_masks_visual_trail_active_slots_past_cursor():
    state = bench._production_to_benchmark_source_state(
        np=np,
        production_state=_minimal_production_state(),
        config=_base_config(),
    )

    np.testing.assert_array_equal(state["trail_active"], np.array([[1, 1, 0, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(state["trail_write_cursor"], np.array([2], dtype=np.int32))
    np.testing.assert_array_equal(
        state["trail_x"][0], np.array([1.0, 2.0, 99.0, 100.0], dtype=np.float32)
    )


def test_cpu_reference_palette_respects_non_identity_avatar_color():
    state = {
        "avatar_color": np.array([[7, 3]], dtype=np.int16),
    }

    palette = bench._benchmark_player_rgb_palette_for_state(
        np=np,
        state=state,
        row=0,
        config=_base_config(player_count=2, controlled_player=1),
    )

    assert palette is not None
    assert palette[3] == (
        bench.PERSPECTIVE_SELF_LUMA,
        bench.PERSPECTIVE_SELF_LUMA,
        bench.PERSPECTIVE_SELF_LUMA,
    )
    assert palette[1] == (
        bench.PERSPECTIVE_OTHER_LUMA,
        bench.PERSPECTIVE_OTHER_LUMA,
        bench.PERSPECTIVE_OTHER_LUMA,
    )
    assert palette[7] == (
        bench.PERSPECTIVE_OTHER_LUMA,
        bench.PERSPECTIVE_OTHER_LUMA,
        bench.PERSPECTIVE_OTHER_LUMA,
    )


def test_production_to_benchmark_preserves_avatar_color_for_gpu_palette():
    production = _minimal_production_state()
    production["avatar_color"] = np.array([[2, 0]], dtype=np.int16)

    state = bench._production_to_benchmark_source_state(
        np=np,
        production_state=production,
        config=_base_config(player_count=2, controlled_player=1),
    )

    np.testing.assert_array_equal(state["avatar_color"], np.array([[2, 0]], dtype=np.int32))


def test_cpu_previous_owner_trail_slots_connects_interleaved_same_owner_slots():
    state = {
        "trail_x": np.array([[10.0, 50.0, 20.0]], dtype=np.float32),
        "trail_y": np.array([[11.0, 51.0, 21.0]], dtype=np.float32),
        "trail_radius": np.ones((1, 3), dtype=np.float32),
        "trail_owner": np.array([[0, 1, 0]], dtype=np.int32),
        "trail_active": np.ones((1, 3), dtype=np.uint8),
        "head_x": np.zeros((1, 2), dtype=np.float32),
    }

    prev_x, prev_y, prev_active = bench._cpu_previous_owner_trail_slots(
        np=np,
        state=state,
    )

    assert bool(prev_active[0, 2])
    assert prev_x[0, 2] == np.float32(10.0)
    assert prev_y[0, 2] == np.float32(11.0)
    assert prev_x[0, 2] != np.float32(50.0)
    assert prev_y[0, 2] != np.float32(51.0)


def test_owner_ordered_active_trail_slots_groups_invalid_then_valid_descending_stably():
    owners = np.array([0, -1, 2, 1, 2, -3, 1, -1, 0], dtype=np.int32)
    active = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=np.uint8)

    np.testing.assert_array_equal(
        bench._owner_ordered_active_trail_slots(np=np, owners=owners, active=active),
        np.array([5, 1, 7, 2, 4, 6, 0, 8], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        bench._owner_ordered_compact_trail_order(np=np, owners=owners, active=active),
        np.array([5, 1, 7, 2, 4, 6, 0, 8, 3], dtype=np.int32),
    )


def test_pack_compact_trails_in_owner_draw_order_preserves_stable_slot_payloads():
    state = {
        "trail_x": np.array([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]], dtype=np.float32),
        "trail_y": np.array([[20.0, 21.0, 22.0, 23.0, 24.0, 25.0]], dtype=np.float32),
        "trail_radius": np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
        "trail_owner": np.array([[0, 2, -1, 2, 0, 1]], dtype=np.int32),
        "trail_active": np.array([[1, 0, 1, 1, 1, 1]], dtype=np.uint8),
        "trail_break_before": np.array([[0, 1, 0, 1, 1, 0]], dtype=np.uint8),
        "head_x": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    }

    packed = bench._pack_compact_trails_in_owner_draw_order(
        np=np,
        state=state,
        config=_base_config(player_count=3, trail_slots=6),
    )

    np.testing.assert_array_equal(
        packed["trail_owner"],
        np.array([[-1, 2, 1, 0, 0, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        packed["trail_active"],
        np.array([[1, 1, 1, 1, 1, 0]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        packed["trail_x"],
        np.array([[12.0, 13.0, 15.0, 10.0, 14.0, 11.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(packed["trail_write_cursor"], np.array([5], dtype=np.int32))
    np.testing.assert_array_equal(packed["head_x"], state["head_x"])


def test_adversarial_fixture_masks_cursor_stale_slots_and_covers_owner_cases():
    config = bench._validate_config(
        {
            "state_source": bench.STATE_SOURCE_ADVERSARIAL_FIXTURE,
            "batch_size": 4,
            "player_count": 3,
            "trail_slots": 10,
            "bonus_count": 3,
            "controlled_player": 0,
            "render_surface": bench.RENDER_SURFACE_BLOCK_704_GRAY64,
        }
    )
    production = bench._adversarial_fixture_production_state(np=np, config=config)
    np.testing.assert_array_equal(
        production["avatar_color"][:4, :3],
        np.array(
            [
                [2, 0, 1],
                [0, 0, 2],
                [7, 3, 5],
                [1, 2, 1],
            ],
            dtype=np.int16,
        ),
    )
    state = bench._production_to_benchmark_source_state(
        np=np,
        production_state=production,
        config=config,
    )

    assert int(production["visual_trail_write_cursor"][0]) == 8
    assert production["visual_trail_active"][0, 8]
    assert production["visual_trail_active"][0, 9]
    np.testing.assert_array_equal(state["trail_active"][0, 8:], np.array([0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(
        state["trail_owner"][0, :4],
        np.array([0, 1, 0, 1], dtype=np.int32),
    )
    assert bool(state["trail_break_before"][0, 4])
    assert state["trail_radius"][0, 4] == np.float32(9.0)

    prev_x, _prev_y, prev_active = bench._cpu_previous_owner_trail_slots(np=np, state=state)
    assert not bool(state["trail_active"][1, 3])
    assert bool(state["trail_active"][1, 4])
    assert bool(prev_active[1, 4])
    assert prev_x[1, 4] == state["trail_x"][1, 2]

    assert int(production["visual_trail_write_cursor"][2]) == 0
    assert bool(production["visual_trail_active"][2].any())
    assert not bool(state["trail_active"][2].any())
    assert int(state["trail_owner"][1, 9]) == -1
    assert not bool(production["present"][3, 2])


def test_owner_ordered_compact_adversarial_fixture_matches_cpu_oracle_without_gpu():
    config = bench._validate_config(
        {
            "state_source": bench.STATE_SOURCE_ADVERSARIAL_FIXTURE,
            "batch_size": 4,
            "player_count": 3,
            "trail_slots": 10,
            "bonus_count": 3,
            "controlled_player": 0,
            "render_surface": bench.RENDER_SURFACE_BLOCK_704_GRAY64,
            "trail_composition": bench.TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT,
        }
    )
    production = bench._adversarial_fixture_production_state(np=np, config=config)
    compact = bench._production_to_benchmark_source_state(
        np=np,
        production_state=production,
        config=config,
    )
    packed = bench._prepare_compact_state_for_render(np=np, state=compact, config=config)

    np.testing.assert_array_equal(
        packed["trail_owner"][0, :8],
        np.array([2, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(packed["trail_write_cursor"], np.array([8, 9, 0, 9]))
    expected = bench._cpu_render_original_production_canvas_gray64(
        np=np,
        production_state=production,
        config=config,
    )
    got = bench._cpu_render_production_canvas_gray64(
        np=np,
        state=packed,
        config=config,
    )

    np.testing.assert_array_equal(got, expected)


def test_adversarial_fixture_cpu_oracle_renders_both_controlled_views():
    base_config = {
        "state_source": bench.STATE_SOURCE_ADVERSARIAL_FIXTURE,
        "batch_size": 4,
        "player_count": 3,
        "trail_slots": 10,
        "bonus_count": 3,
        "render_surface": bench.RENDER_SURFACE_BLOCK_704_GRAY64,
    }
    production = bench._adversarial_fixture_production_state(
        np=np,
        config=bench._validate_config({**base_config, "controlled_player": 0}),
    )
    p0 = bench._cpu_render_original_production_canvas_gray64(
        np=np,
        production_state=production,
        config=bench._validate_config({**base_config, "controlled_player": 0}),
    )
    p1 = bench._cpu_render_original_production_canvas_gray64(
        np=np,
        production_state=production,
        config=bench._validate_config({**base_config, "controlled_player": 1}),
    )

    assert p0.shape == (4, 1, 64, 64)
    assert p0.dtype == np.uint8
    assert p1.shape == p0.shape
    assert int(np.count_nonzero(p0 != p1)) > 0
    assert int(np.count_nonzero(p0[2] != bench.SYNTHETIC_BACKGROUND_LUMA)) > 0


@pytest.mark.parametrize(
    "trail_composition",
    [
        bench.TRAIL_COMPOSITION_PRIORITY_BUFFER,
        bench.TRAIL_COMPOSITION_OWNER_ORDERED_COMPACT,
    ],
)
def test_adversarial_fixture_jax_parity_when_jax_is_available(trail_composition):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    config = bench._validate_config(
        {
            "state_source": bench.STATE_SOURCE_ADVERSARIAL_FIXTURE,
            "batch_size": 4,
            "player_count": 3,
            "trail_slots": 10,
            "bonus_count": 3,
            "controlled_player": 0,
            "render_surface": bench.RENDER_SURFACE_BLOCK_704_GRAY64,
            "trail_composition": trail_composition,
            "verify_rows": 4,
        }
    )
    state, production = bench._source_state_and_reference_for_benchmark(
        np=np,
        config=config,
    )
    assert production is not None
    render_fn = bench._make_jax_render_fn(
        jax=jax,
        jnp=jnp,
        config=config,
        render_mode_id=bench.RENDER_MODE_IDS[bench.RENDER_MODE_BROWSER_LINES],
        bonus_render_mode_id=bench.BONUS_RENDER_MODE_IDS[bench.BONUS_RENDER_MODE_SIMPLE_SYMBOLS],
    )

    got = np.asarray(render_fn(bench._copy_state_to_device(jax=jax, state=state)))
    expected = bench._cpu_render_original_production_canvas_gray64(
        np=np,
        production_state=production,
        config=config,
    )

    np.testing.assert_array_equal(got, expected)
